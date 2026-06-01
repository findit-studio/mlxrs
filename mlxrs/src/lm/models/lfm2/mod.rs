//! LFM2 — the hybrid short-convolution + attention language model.
//!
//! Faithful 1:1 port of `mlx-lm/mlx_lm/models/lfm2.py` (the dense LM; the
//! MoE variant `lfm2_moe.py` is out of scope). LFM2 interleaves two mixer
//! kinds per decoder layer, selected by layer index:
//!
//! - **Attention** (`lfm2.py:53-109`) — grouped-query attention with a
//!   per-head RMSNorm on the queries/keys (QK-norm) before RoPE, then the
//!   fused
//!   [`scaled_dot_product_attention`](crate::lm::nn::attention::scaled_dot_product_attention)
//!   against a [`StandardKvCache`](crate::lm::cache::StandardKvCache).
//! - **ShortConv** (`lfm2.py:112-170`) — a gated causal **depthwise**
//!   1-D convolution. `in_proj` produces a `(B, C, x)` triple; `Bx = B * x`
//!   is masked, left-padded (prefill) or prefixed with the cached state
//!   (decode), depthwise-convolved, gated by `C`, then `out_proj`. The
//!   recurrent state lives in an [`ArraysCache`](crate::lm::cache::ArraysCache)
//!   of one slot.
//!
//! Each decoder layer is pre-norm: `h = x + mixer(operator_norm(x))`
//! then `out = h + feed_forward(ffn_norm(h))`. The decoder embeds the
//! tokens, runs the layers with the two precomputed masks, applies the final
//! [`RMSNorm`](crate::lm::nn::norm::RMSNorm), and the tied embedding head
//! ([`Lfm2`](crate::lm::models::lfm2::Lfm2)) projects back to the vocabulary
//! via `embed_tokens.as_linear`.
//!
//! Per-layer cache is **heterogeneous**: a
//! [`StandardKvCache`](crate::lm::cache::StandardKvCache) for every attention
//! layer and a one-slot [`ArraysCache`](crate::lm::cache::ArraysCache) for
//! every conv layer (`lfm2.py:312-316`).
//! [`Lfm2::make_cache`](crate::lm::models::lfm2::Lfm2::make_cache) builds it in
//! layer order.
//!
//! ## `conv.weight` sanitize
//!
//! PyTorch stores a depthwise `nn.Conv1d` weight as `(C, 1, K)`; MLX's
//! channels-last `conv1d` wants `(C, K, 1)`.
//! [`Lfm2::sanitize`](crate::lm::models::lfm2::Lfm2::sanitize) transposes any
//! `conv.weight` whose last axis is larger than its middle axis (the
//! `lfm2.py:298-306` rule), which is the load-bearing fix for the depthwise
//! convolution.

mod linear;

use std::ffi::CStr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    ArithmeticOverflowPayload, Error, InvariantViolationPayload, LengthMismatchPayload,
    MissingKeyPayload, NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, Result,
  },
  lm::{
    cache::{ArraysCache, KvCache, MaskMode, StandardKvCache},
    model::Model as LmModel,
    nn::{
      activations::swiglu,
      attention::{Mask, scaled_dot_product_attention},
      norm::RMSNorm,
      rope::Rope,
    },
  },
  ops::{
    self,
    conv::conv1d,
    indexing::{take_along_axis, take_axis},
    logical::select,
    misc::astype,
    shape::{
      concatenate, expand_dims_axes, pad, reshape, split_sections, swapaxes, transpose_axes,
    },
  },
};
use linear::Linear;
use smol_str::format_smolstr;

use crate::model_validation::{
  checked_add, checked_mul, require_cardinality, require_divisible, require_even, require_in_range,
  reserve_or_error, take_if,
};

/// `mlx_pad`'s constant-fill mode string.
const PAD_CONSTANT: &CStr = c"constant";

/// Inclusive upper bound for every *width*-like [`TextConfig`] field
/// (`hidden_size`, `vocab_size`, the MLP widths). `2^24` is far above any real
/// LFM2 checkpoint (the largest real value, `vocab_size`, is ~10^5) yet small
/// enough that the downstream `i32` shape arithmetic — the `2 * block_ff_dim`
/// and `block_multiple_of * ceil(..)` of [`adjusted_ff_dim`], the per-head
/// reshapes, and the vocab-width logits projection — cannot overflow `i32`.
/// Rejecting an oversized field here keeps a malformed config a recoverable
/// [`Error::OutOfRange`] instead of a wrapping multiply downstream.
///
/// `conv_L_cache` is deliberately **not** in this set: it sizes runtime
/// allocations rather than just a width, so it takes the far smaller
/// [`MAX_CONV_L_CACHE`] cardinality cap instead.
const MAX_CONFIG_DIM: i32 = 1 << 24;

/// Inclusive upper bound for every *cardinality*-like [`TextConfig`] field —
/// `num_hidden_layers` (a per-layer `Vec` of heavyweight
/// [`Lfm2DecoderLayer`]s is reserved up front) and the head counts. Unlike the
/// width fields these size *eager allocations* (the decoder-layer `Vec`, the
/// per-layer cache), so the overflow-safe `2^24` width cap is far too loose:
/// `2^24` layers would request a multi-gigabyte `Vec` before the first
/// missing-key error. The largest real LFM2 has tens of layers and ~10^1 heads;
/// `4096` is generous headroom yet keeps a malformed cardinality a recoverable
/// [`Error::OutOfRange`] (or, if it still slips through, a recoverable
/// [`Error::AllocFailure`] from the `try_reserve`d layer `Vec`) rather than an
/// allocator abort.
const MAX_CONFIG_CARDINALITY: i32 = 4096;

/// Inclusive upper bound for `conv_L_cache`, the short-convolution
/// kernel / cache window. Unlike the width fields it never names a matmul axis;
/// it directly **sizes allocations** — the recurrent conv-state array's middle
/// axis (`conv_L_cache - 1`), the prefill left-pad width, and (via
/// [`lengths_positions`]) a host `B * (conv_L_cache - 1)` index `Vec`. The
/// `2^24` width cap would let one field drive a ~16M-wide pad / a `B`-multiplied
/// gigabyte host allocation, so it takes a dedicated, much tighter cap.
///
/// Real LFM2 kernels are single-digit (`conv_L_cache == 3`); `256` is far
/// beyond any plausible short-conv window yet bounds the cache-state width and
/// the per-batch host index build to a small constant factor, keeping an
/// oversized value a recoverable [`Error::CapExceeded`] rather than an
/// allocation that aborts before the first typed error.
const MAX_CONV_L_CACHE: i32 = 256;

// ───────────────────────── config ─────────────────────────

/// LFM2 text-model configuration — mirrors `mlx-vlm`'s `lfm2_vl/config.py`
/// `TextConfig` (and `mlx-lm`'s `lfm2.ModelArgs`). Parsed from a
/// `config.json` (or the VL config's `text_config`) via [`from_json`].
///
/// Like mlx-lm's `BaseModelArgs.from_dict`, unmodeled keys are ignored
/// (serde does not `deny_unknown_fields`) and every field carries the
/// reference's default so a partial config still parses. The
/// `full_attn_idxs` derivation (`__post_init__`) is reproduced by
/// [`attention_layer_indices`](TextConfig::attention_layer_indices).
///
/// [`from_json`]: TextConfig::from_json
#[derive(Debug, Clone, serde::Deserialize)]
#[non_exhaustive]
pub struct TextConfig {
  /// Architecture id (`"lfm2"`).
  #[serde(default = "default_model_type")]
  pub model_type: String,
  /// Hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of decoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Number of attention (query) heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Number of key/value heads (GQA).
  #[serde(default = "default_num_key_value_heads")]
  pub num_key_value_heads: i32,
  /// Maximum positional context (carried; not used by the forward pass).
  #[serde(default = "default_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// RoPE base frequency.
  #[serde(default = "default_rope_theta")]
  pub rope_theta: f32,
  /// Vocabulary size (logits last-axis width).
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// RMSNorm variance floor.
  #[serde(default = "default_norm_eps")]
  pub norm_eps: f32,
  /// Whether the MLP applies the SwiGLU `auto_adjust_ff_dim` reduction.
  #[serde(default = "default_true")]
  pub block_auto_adjust_ff_dim: bool,
  /// MLP input/output width (equals `hidden_size`).
  #[serde(default = "default_hidden_size")]
  pub block_dim: i32,
  /// MLP feed-forward width before any `auto_adjust_ff_dim` reduction.
  #[serde(default = "default_block_ff_dim")]
  pub block_ff_dim: i32,
  /// `auto_adjust_ff_dim` post-multiplier.
  #[serde(default = "default_block_ffn_dim_multiplier")]
  pub block_ffn_dim_multiplier: f32,
  /// `auto_adjust_ff_dim` rounding granularity.
  #[serde(default = "default_block_multiple_of")]
  pub block_multiple_of: i32,
  /// Depthwise-conv kernel length (the cache window is `conv_L_cache - 1`).
  /// The `config.json` key is `conv_L_cache` (mlx-lm's field name).
  #[serde(rename = "conv_L_cache", default = "default_conv_l_cache")]
  pub conv_l_cache: i32,
  /// Whether the conv / its projections carry a bias.
  #[serde(default)]
  pub conv_bias: bool,
  /// Explicit per-layer kind list (`"full_attention"` vs the conv default).
  /// `None` ⇒ every layer is a conv layer unless [`full_attn_idxs`] names it.
  ///
  /// [`full_attn_idxs`]: TextConfig::full_attn_idxs
  #[serde(default)]
  pub layer_types: Option<Vec<String>>,
  /// Explicit attention-layer index list. When present it is authoritative;
  /// otherwise it is derived from `layer_types` (`__post_init__`).
  #[serde(default)]
  pub full_attn_idxs: Option<Vec<i32>>,
}

fn default_model_type() -> String {
  "lfm2".to_string()
}
fn default_hidden_size() -> i32 {
  1024
}
fn default_num_hidden_layers() -> i32 {
  16
}
fn default_num_attention_heads() -> i32 {
  16
}
fn default_num_key_value_heads() -> i32 {
  8
}
fn default_max_position_embeddings() -> i32 {
  128000
}
fn default_rope_theta() -> f32 {
  1_000_000.0
}
fn default_vocab_size() -> i32 {
  65536
}
fn default_norm_eps() -> f32 {
  1e-5
}
fn default_true() -> bool {
  true
}
fn default_block_ff_dim() -> i32 {
  6656
}
fn default_block_ffn_dim_multiplier() -> f32 {
  1.0
}
fn default_block_multiple_of() -> i32 {
  256
}
fn default_conv_l_cache() -> i32 {
  3
}

impl TextConfig {
  /// Parse a [`TextConfig`] from a `config.json` (or `text_config`) string.
  ///
  /// A serde failure (malformed JSON) maps to [`Error::Parse`]; missing keys
  /// fall back to the reference defaults rather than erroring (mlx-lm
  /// `from_dict` semantics). The parsed fields are then [`validate`]d so a
  /// malformed config (zero / negative / non-divisible / oversized dimension)
  /// is a recoverable error here rather than a panic downstream.
  ///
  /// [`validate`]: TextConfig::validate
  pub fn from_json(json: &str) -> Result<TextConfig> {
    let cfg: TextConfig = serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "TextConfig::from_json",
        "LFM2 text config JSON",
        e,
      ))
    })?;
    cfg.validate()?;
    Ok(cfg)
  }

  /// Reject a structurally invalid configuration before it can panic the
  /// forward pass. Mirrors the eager `validate()` discipline of
  /// [`crate::lm::generate::GenConfig`].
  ///
  /// Every width-like field must be a positive integer no larger than `2^24`
  /// (so the downstream `i32` shape arithmetic cannot overflow); the
  /// cardinality fields that size eager allocations (`num_hidden_layers`, the
  /// head counts) and `conv_L_cache` (which sizes the conv-state / pad / host
  /// index allocations) take their own far tighter caps. The two derived ratios
  /// must divide exactly:
  ///
  /// - `hidden_size` must be divisible by `num_attention_heads` — otherwise
  ///   [`head_dim`](TextConfig::head_dim) truncates and the per-head reshape
  ///   `(B, L, n_heads, head_dim)` disagrees with `q_proj`'s output width.
  /// - `num_attention_heads` must be divisible by `num_key_value_heads` —
  ///   the grouped-query head grouping (`lfm2.py:69-70`).
  ///
  /// `block_multiple_of` must be `>= 1` (it is the rounding divisor in the
  /// `auto_adjust_ff_dim` reduction; a zero divisor would divide-by-zero).
  ///
  /// Beyond the per-field bounds:
  ///
  /// - the derived `head_dim` must be **even** — LFM2 builds
  ///   [`Rope::new(head_dim)`](crate::lm::nn::rope::Rope::new), whose rotation
  ///   pairs feature `k` with `k + head_dim/2`; an odd `head_dim` loads but
  ///   only fails inside the forward pass.
  /// - `block_ffn_dim_multiplier` must be **finite and positive**, and the
  ///   resulting `auto_adjust_ff_dim` width must itself stay within the width
  ///   cap — a huge multiplier would otherwise saturate / overflow the `i32`
  ///   MLP-width arithmetic at load time.
  ///
  /// Returns the first violation as [`Error::OutOfRange`] /
  /// [`Error::DivisibilityConstraint`] / [`Error::NonFiniteScalar`]; `Ok(())`
  /// when every field is sound.
  pub fn validate(&self) -> Result<()> {
    // Width-like fields: positive and within the overflow-safe `2^24` cap.
    // `require_in_range(_, 1, MAX_CONFIG_DIM)` rejects both a non-positive and
    // an oversized value as one [`Error::OutOfRange`].
    for (name, value) in [
      ("hidden_size", self.hidden_size),
      ("vocab_size", self.vocab_size),
      ("block_dim", self.block_dim),
      ("block_ff_dim", self.block_ff_dim),
      // `block_multiple_of` is a width-arithmetic divisor: positive (a zero
      // would divide-by-zero in `adjusted_ff_dim`) and within the width cap.
      ("block_multiple_of", self.block_multiple_of),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Cardinality-like fields size eager allocations (the decoder-layer `Vec`,
    // the per-layer cache), so they take the much smaller realistic cap: a
    // non-positive count is [`Error::OutOfRange`], an over-cap one is
    // [`Error::CapExceeded`].
    for (name, value) in [
      ("num_hidden_layers", self.num_hidden_layers),
      ("num_attention_heads", self.num_attention_heads),
      ("num_key_value_heads", self.num_key_value_heads),
    ] {
      require_cardinality(name, i64::from(value), MAX_CONFIG_CARDINALITY as u64)?;
    }
    // `conv_L_cache` sizes runtime allocations — the conv-state array's middle
    // axis (`conv_L_cache - 1`), the prefill left-pad, and the host
    // `B * (conv_L_cache - 1)` index `Vec` of [`ShortConv::forward`] — so it
    // takes its own tight [`MAX_CONV_L_CACHE`] cardinality cap (positive, and
    // small enough that no single field can drive an oversized allocation),
    // not the loose `2^24` width cap.
    require_cardinality(
      "conv_L_cache",
      i64::from(self.conv_l_cache),
      MAX_CONV_L_CACHE as u64,
    )?;
    // head_dim = hidden_size / num_attention_heads must divide exactly.
    require_divisible(
      "hidden_size",
      self.hidden_size,
      "num_attention_heads",
      self.num_attention_heads,
    )?;
    // RoPE rotates feature `k` with `k + head_dim/2`, so `head_dim` must be
    // even; an odd one loads but only fails inside the attention forward pass.
    require_even("head_dim", self.head_dim())?;
    // GQA: query heads must be an integer multiple of key/value heads.
    require_divisible(
      "num_attention_heads",
      self.num_attention_heads,
      "num_key_value_heads",
      self.num_key_value_heads,
    )?;
    // The SwiGLU `auto_adjust_ff_dim` multiplier is a free-floating f32: reject
    // a non-finite / non-positive value, then run the reduction eagerly so a
    // multiplier that would saturate / overflow the i32 MLP-width arithmetic is
    // a config-time error and the resulting width stays within the cap.
    if !self.block_ffn_dim_multiplier.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "TextConfig::validate (block_ffn_dim_multiplier)",
        f64::from(self.block_ffn_dim_multiplier),
      )));
    }
    if self.block_ffn_dim_multiplier <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "TextConfig::validate (block_ffn_dim_multiplier)",
        "must be a positive float",
        format_smolstr!("block_ffn_dim_multiplier={}", self.block_ffn_dim_multiplier),
      )));
    }
    let adjusted = adjusted_ff_dim(
      self.block_ff_dim,
      self.block_multiple_of,
      self.block_auto_adjust_ff_dim,
      self.block_ffn_dim_multiplier,
    )?;
    if adjusted > MAX_CONFIG_DIM {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "TextConfig::validate (adjusted feed-forward width)",
        "auto_adjust_ff_dim result must be <= 2^24",
        format_smolstr!("adjusted_ff_dim={adjusted}"),
      )));
    }
    Ok(())
  }

  /// The per-head dimension (`hidden_size / num_attention_heads`).
  pub fn head_dim(&self) -> i32 {
    self.hidden_size / self.num_attention_heads
  }

  /// The sorted attention-layer indices, reproducing `__post_init__`: the
  /// explicit `full_attn_idxs` when present, else every index whose
  /// `layer_types` entry is `"full_attention"`, else (no `layer_types`) the
  /// empty set — every layer is a conv layer.
  ///
  /// Each source list is a per-layer description, so its length is bounded by
  /// the realistic `MAX_CONFIG_CARDINALITY` cap **before** the derived `Vec` is
  /// built: a malformed config (a tiny `num_hidden_layers` paired with a huge
  /// `full_attn_idxs` / `layer_types`) would otherwise drive a large host
  /// allocation — the wholesale clone of `full_attn_idxs`, or the
  /// `"full_attention"` collect over `layer_types` — before any index is
  /// range-checked. An over-cap length is a recoverable [`Error::CapExceeded`];
  /// the derived `Vec` is then [`reserve_or_error`]'d and each index validated
  /// against `[0, num_hidden_layers)` as it is appended (an out-of-range index
  /// is [`Error::OutOfRange`], since it would otherwise mis-key the cache / mask
  /// precomputation).
  pub fn attention_layer_indices(&self) -> Result<Vec<i32>> {
    let n = self.num_hidden_layers;
    let mut idxs: Vec<i32> = Vec::new();
    match (&self.full_attn_idxs, &self.layer_types) {
      (Some(explicit), _) => {
        // Bound the source length before cloning so a hostile `full_attn_idxs`
        // cannot over-allocate; an empty list (no attention layers) is valid
        // and needs no cap check.
        if !explicit.is_empty() {
          require_cardinality(
            "full_attn_idxs length",
            explicit.len() as i64,
            MAX_CONFIG_CARDINALITY as u64,
          )?;
          reserve_or_error(&mut idxs, "attention layer index", explicit.len())?;
        }
        for &i in explicit {
          idxs.push(checked_attention_index(i, n)?);
        }
      }
      (None, Some(types)) => {
        // Bound the per-layer list length before the `"full_attention"` collect
        // so a hostile `layer_types` cannot over-allocate. The enumerate index
        // stays within `i32` because the cap (`4096`) is well below `i32::MAX`.
        if !types.is_empty() {
          require_cardinality(
            "layer_types length",
            types.len() as i64,
            MAX_CONFIG_CARDINALITY as u64,
          )?;
          reserve_or_error(&mut idxs, "attention layer index", types.len())?;
        }
        for (i, t) in types.iter().enumerate() {
          if t.as_str() == "full_attention" {
            idxs.push(checked_attention_index(i as i32, n)?);
          }
        }
      }
      (None, None) => {}
    }
    Ok(idxs)
  }
}

/// Validate one attention-layer index against `[0, num_hidden_layers)`,
/// returning it unchanged when in range and [`Error::OutOfRange`] otherwise.
///
/// Used by [`TextConfig::attention_layer_indices`] to reject an out-of-range
/// index **as it is appended** (a malformed index would otherwise mis-key the
/// cache / mask precomputation).
fn checked_attention_index(idx: i32, num_hidden_layers: i32) -> Result<i32> {
  if idx < 0 || idx >= num_hidden_layers {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "TextConfig: attention layer index (must be in [0, num_hidden_layers))",
      "in [0, num_hidden_layers)",
      format_smolstr!("idx={idx}, num_hidden_layers={num_hidden_layers}"),
    )));
  }
  Ok(idx)
}

/// Whether layer `idx` is an attention layer, given the sorted attention
/// index set.
fn is_attention_layer(attn_idxs: &[i32], idx: i32) -> bool {
  attn_idxs.contains(&idx)
}

/// `mlx-lm`'s `MLP` SwiGLU `auto_adjust_ff_dim` reduction (`lfm2.py:183-187`):
/// `ff = round_up(multiplier * (2/3 * ff_dim), multiple_of)` (multiplier
/// applied only when `> 0`). Returns `ff_dim` unchanged when `adjust` is
/// false. All integer arithmetic mirrors Python's truncating `int(...)`.
///
/// Every step is **checked**: `multiplier` is a free-floating `f32` from the
/// config, so a large value (or one that, multiplied in, exceeds `i32`) would
/// saturate the `as i32` cast to `i32::MAX` and then wrap (or panic) on the
/// `ff + multiple_of - 1` round-up. The integer-multiply / -add steps go
/// through [`checked_mul`] / [`checked_add`] (typed [`Error::ArithmeticOverflow`]
/// on overflow); the f32-multiplier product is computed in `f64` and
/// range-checked before the truncating cast, reported as a typed
/// [`Error::OutOfRange`] when it would exceed `i32` (rather than a silently-wrong
/// width or an overflow panic). [`TextConfig::validate`] runs this eagerly so the
/// failure surfaces at config time, not mid-load. `multiplier` is assumed
/// already validated finite + positive by the caller; `multiple_of >= 1`
/// likewise.
fn adjusted_ff_dim(ff_dim: i32, multiple_of: i32, adjust: bool, multiplier: f32) -> Result<i32> {
  if !adjust {
    return Ok(ff_dim);
  }
  // `int(2 * ff_dim / 3)` — Python truncates toward zero.
  let two_ff = checked_mul("adjusted_ff_dim: 2 * ff_dim", "ff_dim", ff_dim, "two", 2)?;
  let base = two_ff / 3;
  // `int(ffn_dim_multiplier * ff_dim)` — mlx-lm always applies this when the
  // multiplier is not None; the VL config defaults it to 1.0. The product is
  // computed in f64 and range-checked before the truncating cast so a large
  // multiplier becomes a typed error rather than an `i32::MAX` saturation.
  let scaled_f = f64::from(multiplier) * f64::from(base);
  if !(scaled_f >= 0.0 && scaled_f <= f64::from(i32::MAX)) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "adjusted_ff_dim (auto_adjust_ff_dim overflowed i32)",
      "multiplier * (2/3 * ff_dim) must fit in i32",
      format_smolstr!("ff_dim={ff_dim}, multiple_of={multiple_of}, multiplier={multiplier}"),
    )));
  }
  let ff = scaled_f as i32;
  // `multiple_of * ((ff + multiple_of - 1) // multiple_of)` — round up.
  let bumped = checked_add(
    "adjusted_ff_dim: ff + multiple_of - 1",
    "ff",
    ff,
    "multiple_of_minus_one",
    multiple_of - 1,
  )?;
  checked_mul(
    "adjusted_ff_dim: multiple_of * ceil(ff / multiple_of)",
    "multiple_of",
    multiple_of,
    "ceil",
    bumped / multiple_of,
  )
}

// ───────────────────────── attention ─────────────────────────

/// Grouped-query attention with per-head QK-norm (`lfm2.py:53-109`).
#[derive(Debug)]
struct Attention {
  n_heads: i32,
  n_kv_heads: i32,
  scale: f32,
  q_layernorm: RMSNorm,
  k_layernorm: RMSNorm,
  q_proj: Linear,
  k_proj: Linear,
  v_proj: Linear,
  out_proj: Linear,
  rope: Rope,
}

impl Attention {
  /// `queries/keys/values = {q,k,v}_proj(x)`, per-head reshape +
  /// q/k RMSNorm, RoPE at the cache offset, `cache.update_and_fetch`, then
  /// the fused [`scaled_dot_product_attention`] and `out_proj`.
  ///
  /// `cache` is the attention layer's [`StandardKvCache`]; `mask` is the
  /// attention mask mode for this forward pass.
  fn forward(&self, x: &Array, mask: &MaskMode, cache: &mut StandardKvCache) -> Result<Array> {
    let shape = x.shape();
    let (b, l) = (shape[0] as i32, shape[1] as i32);

    let queries = self.q_proj.forward(x)?;
    let keys = self.k_proj.forward(x)?;
    let values = self.v_proj.forward(x)?;

    // Per-head reshape `(B, L, n_heads, head_dim)`, q/k RMSNorm over the last
    // axis, then transpose to `(B, n_heads, L, head_dim)`.
    let queries = reshape(&queries, &[b, l, self.n_heads, -1])?;
    let queries = self.q_layernorm.forward(&queries)?;
    let queries = transpose_axes(&queries, &[0, 2, 1, 3])?;

    let keys = reshape(&keys, &[b, l, self.n_kv_heads, -1])?;
    let keys = self.k_layernorm.forward(&keys)?;
    let keys = transpose_axes(&keys, &[0, 2, 1, 3])?;

    let values = reshape(&values, &[b, l, self.n_kv_heads, -1])?;
    let values = transpose_axes(&values, &[0, 2, 1, 3])?;

    // RoPE at the cache offset, then append+fetch the running K/V.
    let offset = cache.offset() as i32;
    let queries = self.rope.apply(&queries, offset)?;
    let keys = self.rope.apply(&keys, offset)?;
    let (keys, values) = cache.update(&keys, &values)?;

    let attn_mask = mask_mode_to_mask(mask);
    let output = scaled_dot_product_attention(&queries, &keys, &values, self.scale, attn_mask)?;
    // `(B, n_heads, L, head_dim)` -> `(B, L, n_heads*head_dim)`.
    let output = transpose_axes(&output, &[0, 2, 1, 3])?;
    let output = reshape(&output, &[b, l, -1])?;
    self.out_proj.forward(&output)
  }
}

/// Map a [`MaskMode`] to the attention [`Mask`] selector. `Array` borrows
/// the mode's owned array for the call's duration.
fn mask_mode_to_mask(mode: &MaskMode) -> Mask<'_> {
  match mode {
    MaskMode::None => Mask::None,
    MaskMode::Causal => Mask::Causal,
    MaskMode::Array(a) => Mask::Array(a),
  }
}

// ───────────────────────── short conv ─────────────────────────

/// Gated causal depthwise short-convolution (`lfm2.py:112-170`).
#[derive(Debug)]
struct ShortConv {
  hidden_size: i32,
  /// The conv kernel / cache window (`conv_L_cache`); the recurrent state keeps
  /// `l_cache - 1` frames. Bounded by [`MAX_CONV_L_CACHE`] at config time, so
  /// every allocation it sizes (the conv-state array, the prefill pad, the host
  /// index `Vec`) stays small.
  l_cache: i32,
  /// Depthwise conv weight, `(hidden, K, 1)` (MLX channels-last layout, the
  /// transposed form [`Lfm2::sanitize`] produces from PyTorch's
  /// `(hidden, 1, K)`).
  conv_weight: Array,
  conv_bias: Option<Array>,
  in_proj: Linear,
  out_proj: Linear,
}

impl ShortConv {
  /// `in_proj(x)` → split `(B, C, x)`; `Bx = B*x` masked to 0 where `mask` is
  /// false; left-pad (prefill) or prefix the cached state (decode); depthwise
  /// `conv1d`; `y = C * conv_out`; `out_proj(y)`.
  ///
  /// - `mask`: the conv mask (`[B, N]` boolean) or `None`.
  /// - `cache`: `Some` ⇒ the decode path (reads/stashes the one-slot state
  ///   and advances by the sequence length); `None` ⇒ the prefill path
  ///   (left-pad by `L_cache - 1`).
  fn forward(
    &self,
    x: &Array,
    mask: Option<&Array>,
    cache: Option<&mut ArraysCache>,
  ) -> Result<Array> {
    let bcx = self.in_proj.forward(x)?;
    // `mx.split(BCx, 3, axis=-1)` — three equal `(B, L, hidden)` chunks.
    let h = self.hidden_size;
    let parts = split_sections(&bcx, &[h, 2 * h], -1)?;
    let b_gate = &parts[0];
    let c_gate = &parts[1];
    let x_in = &parts[2];

    let mut bx = b_gate.multiply(x_in)?;
    if let Some(m) = mask {
      // `mx.where(mask[..., None], Bx, 0)` — broadcast the `[B, N]` mask to
      // `[B, N, 1]` and zero the masked positions.
      let m3 = expand_dims_axes(m, &[-1])?;
      let zero = astype(&Array::full::<f32>(&[0i32; 0], 0.0)?, bx.dtype()?)?;
      bx = select(&m3, &bx, &zero)?;
    }

    let conv_in = match cache {
      Some(cache) => {
        // `t = x.shape[1]` — the current sequence length (x is the split
        // chunk, `(B, L, hidden)`).
        let t = x_in.shape()[1] as i32;
        let n_keep = self.l_cache - 1;
        let batch = bx.shape()[0] as i32;
        // `state = cache[0]` or zeros `(B, L_cache-1, hidden)`.
        let state = match cache.get(0) {
          Some(s) => s.try_clone()?,
          None => {
            let dtype = bx.dtype()?;
            zeros_like_dtype(&[batch, n_keep, self.hidden_size], dtype)?
          }
        };
        // `Bx = concat([state, Bx], axis=1)`.
        let bx_cat = concatenate(&[&state, &bx], 1)?;
        // Stash the trailing `n_keep` frames into `cache[0]`, honoring
        // `cache.lengths` via `take_along_axis` when set.
        let new_state = match cache.lengths() {
          Some(lengths) => {
            // `positions = (clip(lengths, 0, t)[:, None] +
            // arange(n_keep))[..., None]`. `lengths` must carry exactly one
            // entry per batch row; a mismatch would otherwise build an index
            // whose batch axis disagrees with `Bx` and fail deep inside
            // `take_along_axis` instead of as a typed error here.
            if lengths.len() != batch as usize {
              return Err(Error::LengthMismatch(LengthMismatchPayload::new(
                "ShortConv::forward: cache.lengths length vs batch",
                batch as usize,
                lengths.len(),
              )));
            }
            // The clamp is folded into the checked/reserved host-index loop so
            // no separate (infallible) `ends` buffer is allocated first.
            let positions = lengths_positions(lengths, t, n_keep)?;
            take_along_axis(&bx_cat, &positions, 1)?
          }
          None => {
            // `Bx[:, -n_keep:, :]`.
            let total = bx_cat.shape()[1] as i32;
            ops::indexing::slice(
              &bx_cat,
              &[0, total - n_keep, 0],
              &[batch, total, self.hidden_size],
              &[1, 1, 1],
            )?
          }
        };
        cache.set(0, new_state)?;
        cache.advance(t as usize)?;
        bx_cat
      }
      None => {
        // Prefill: `mx.pad(Bx, [(0,0), (L_cache-1, 0), (0,0)])`.
        pad(
          &bx,
          &[1],
          &[self.l_cache - 1],
          &[0],
          &astype(&Array::full::<f32>(&[0i32; 0], 0.0)?, bx.dtype()?)?,
          PAD_CONSTANT,
        )?
      }
    };

    // Depthwise `conv1d(groups=hidden, kernel=L_cache)`, stride 1, no pad.
    let conv_out = conv1d(&conv_in, &self.conv_weight, 1, 0, 1, self.hidden_size)?;
    let conv_out = match &self.conv_bias {
      Some(bias) => conv_out.add(bias)?,
      None => conv_out,
    };
    // `y = C * conv_out`.
    let y = c_gate.multiply(&conv_out)?;
    self.out_proj.forward(&y)
  }
}

/// `(clip(lengths, 0, t)[:, None] + arange(n_keep))[..., None]` as an `I32`
/// `[B, n_keep, 1]` index array for `take_along_axis(_, axis=1)`.
///
/// `n_keep` is `conv_L_cache - 1`, bounded small by [`MAX_CONV_L_CACHE`];
/// `lengths` is one entry per batch row (the caller has already asserted
/// `lengths.len() == batch`). The per-row clamp `clip(lengths, 0, t)` is folded
/// into the build loop so no separate (infallible) `ends` buffer is allocated
/// first. The host buffer is sized `B * n_keep` — both factors are non-negative
/// (`n_keep >= 0` since `conv_L_cache >= 1`), but the element count is still
/// **checked** and **fallibly reserved** so no config / batch combination can
/// drive an unbounded allocation that aborts before a typed error: the
/// `B * n_keep` product goes through `usize` [`checked_mul`]-style overflow
/// detection, [`reserve_or_error`] turns an allocator failure into
/// [`Error::AllocFailure`], and each clamped `end + k` index goes through
/// [`checked_add`] so a near-`i32::MAX` clamped length cannot wrap.
fn lengths_positions(lengths: &[i32], t: i32, n_keep: i32) -> Result<Array> {
  // `n_keep >= 0` (guaranteed by the `conv_L_cache >= 1` cap) so the cast is
  // lossless; size the host index buffer with a checked `usize` multiply.
  let n_keep_usize = n_keep as usize;
  let capacity = lengths.len().checked_mul(n_keep_usize).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "lengths_positions: B * (conv_L_cache - 1) index count",
      "usize",
      [
        ("batch", lengths.len() as u64),
        ("n_keep", n_keep_usize as u64),
      ],
    ))
  })?;
  let mut data: Vec<i32> = Vec::new();
  reserve_or_error(&mut data, "conv cache position index", capacity)?;
  for &length in lengths {
    // `clip(length, 0, t)` — `t = x.shape[1] >= 0`, so the clamp result is in
    // `[0, t]` and thus non-negative.
    let end = length.clamp(0, t);
    for k in 0..n_keep {
      // `end` is clamped to `[0, t]`; `k < n_keep`. Both non-negative, but
      // guard the sum so a near-`i32::MAX` clamp cannot wrap into UB.
      data.push(checked_add(
        "lengths_positions: end + offset",
        "end",
        end,
        "offset",
        k,
      )?);
    }
  }
  Array::from_slice::<i32>(&data, &(lengths.len(), n_keep_usize, 1usize))
}

/// A zeros array of the given shape cast to `dtype` (the conv state's dtype
/// must match `Bx` for the `concatenate`).
fn zeros_like_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
  let z = Array::zeros::<f32>(&shape)?;
  if dtype == Dtype::F32 {
    Ok(z)
  } else {
    astype(&z, dtype)
  }
}

// ───────────────────────── MLP ─────────────────────────

/// Dense SwiGLU feed-forward (`lfm2.py:173-194`): `w2(swiglu(w1(x), w3(x)))`.
#[derive(Debug)]
struct Mlp {
  w1: Linear,
  w3: Linear,
  w2: Linear,
}

impl Mlp {
  fn forward(&self, x: &Array) -> Result<Array> {
    let gate = self.w1.forward(x)?;
    let up = self.w3.forward(x)?;
    let act = swiglu(&gate, &up)?;
    self.w2.forward(&act)
  }
}

// ───────────────────────── decoder layer ─────────────────────────

/// One LFM2 decoder layer's mixer — either attention or short-conv
/// (`Lfm2DecoderLayer.__init__` `lfm2.py:200-205`).
#[derive(Debug)]
enum Mixer {
  Attention(Attention),
  Conv(ShortConv),
}

/// A pre-norm LFM2 decoder layer (`lfm2.py:197-234`).
#[derive(Debug)]
struct Lfm2DecoderLayer {
  mixer: Mixer,
  feed_forward: Mlp,
  operator_norm: RMSNorm,
  ffn_norm: RMSNorm,
}

impl Lfm2DecoderLayer {
  /// `h = x + mixer(operator_norm(x), mask, cache)` then
  /// `out = h + feed_forward(ffn_norm(h))`. The mixer dispatches on the
  /// layer kind and consumes its own concrete cache.
  fn forward(&self, x: &Array, mask: &MaskMode, cache: &mut dyn KvCache) -> Result<Array> {
    let normed = self.operator_norm.forward(x)?;
    let r = match &self.mixer {
      Mixer::Attention(attn) => {
        let kv = cache
          .as_any_mut()
          .downcast_mut::<StandardKvCache>()
          .ok_or_else(|| layer_cache_err("attention layer expects a KVCache"))?;
        attn.forward(&normed, mask, kv)?
      }
      Mixer::Conv(conv) => {
        let arr = cache
          .as_any_mut()
          .downcast_mut::<ArraysCache>()
          .ok_or_else(|| layer_cache_err("conv layer expects an ArraysCache"))?;
        let mask_arr = match mask {
          MaskMode::Array(a) => Some(a),
          MaskMode::None | MaskMode::Causal => None,
        };
        // The conv mask comes from the conv cache's `make_mask`, which only
        // ever yields `None` / `Array` (never `Causal`) — so a borrowed
        // mask array is taken before the `&mut` cache borrow below.
        let mask_owned = mask_arr.map(|a| a.try_clone()).transpose()?;
        conv.forward(&normed, mask_owned.as_ref(), Some(arr))?
      }
    };
    let hidden = x.add(&r)?;
    let ffn = self
      .feed_forward
      .forward(&self.ffn_norm.forward(&hidden)?)?;
    hidden.add(&ffn)
  }
}

/// The typed error for a per-layer cache that does not downcast to the kind
/// the layer's mixer requires.
fn layer_cache_err(detail: &'static str) -> Error {
  Error::InvariantViolation(InvariantViolationPayload::new(
    "LFM2 per-layer cache kind",
    detail,
  ))
}

// ───────────────────────── model ─────────────────────────

/// The LFM2 decoder stack (`Lfm2Model`, `lfm2.py:237-279`): token embedding,
/// the per-layer mixers with two precomputed masks, and the final RMSNorm.
#[derive(Debug)]
struct Lfm2Model {
  /// `(vocab, hidden)` token-embedding table; also the tied output head.
  embed_tokens: Array,
  layers: Vec<Lfm2DecoderLayer>,
  embedding_norm: RMSNorm,
  /// The attention-layer index set (for the per-layer mixer dispatch and the
  /// `fa_idx`/`conv_idx` mask-source selection).
  attn_idxs: Vec<i32>,
}

impl Lfm2Model {
  /// `lfm2.py:250-256` — the first attention-layer index (`fa_idx`) and the
  /// first conv-layer index (`conv_idx`), used to source the two masks.
  /// `fa_idx` is `attn_idxs[0]`; `conv_idx` is the count of leading attention
  /// layers (i.e. the first non-attention index). Either may be absent
  /// (all-conv or all-attention models), in which case that mask is `None`.
  fn mask_source_indices(&self, n_layers: i32) -> (Option<i32>, Option<i32>) {
    let fa_idx = self.attn_idxs.first().copied();
    // `conv_idx` = first i not in attn (mlx-lm increments while i is an attn
    // layer, breaking at the first conv index).
    let mut conv_idx = 0;
    while conv_idx < n_layers && is_attention_layer(&self.attn_idxs, conv_idx) {
      conv_idx += 1;
    }
    let conv_idx = if conv_idx < n_layers {
      Some(conv_idx)
    } else {
      None
    };
    (fa_idx, conv_idx)
  }

  /// Run the decoder over precomputed `h` (`(B, L, hidden)`), updating each
  /// layer's cache in place; returns the final-normed hidden states.
  ///
  /// The per-layer cache must hold exactly one entry per decoder layer (as
  /// [`Lfm2::make_cache`] builds it). A mismatched count is a recoverable
  /// [`Error::LengthMismatch`] rather than an out-of-bounds index panic on
  /// the mask-source lookups / a silently truncated `zip` over the layers.
  fn forward_hidden(&self, h: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    if cache.len() != self.layers.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "Lfm2Model::forward: per-layer cache count vs decoder layers",
        self.layers.len(),
        cache.len(),
      )));
    }
    let n_layers = self.layers.len() as i32;
    let n = h.shape()[1];
    let (fa_idx, conv_idx) = self.mask_source_indices(n_layers);

    // Build the two masks once (mlx-lm `create_attention_mask` /
    // `create_ssm_mask`). The attention mask comes from the first attention
    // layer's cache; the conv mask from the first conv layer's cache.
    let attn_mask = match fa_idx {
      Some(i) => cache[i as usize].make_mask(n, None, false)?,
      None => MaskMode::None,
    };
    let conv_mask = match conv_idx {
      Some(i) => cache[i as usize].make_mask(n, None, false)?,
      None => MaskMode::None,
    };

    let mut h = h.try_clone()?;
    for (idx, (layer, c)) in self.layers.iter().zip(cache.iter_mut()).enumerate() {
      let mask = if is_attention_layer(&self.attn_idxs, idx as i32) {
        &attn_mask
      } else {
        &conv_mask
      };
      h = layer.forward(&h, mask, c.as_mut())?;
    }
    self.embedding_norm.forward(&h)
  }

  /// Embed `tokens` (`(B, L)` integer ids) via the embedding table, then run
  /// the decoder.
  fn forward_tokens(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let h = take_axis(&self.embed_tokens, tokens, 0)?;
    self.forward_hidden(&h, cache)
  }
}

/// The LFM2 causal language model (`Model`, `lfm2.py:282-316`): the decoder
/// stack plus the tied embedding head.
#[derive(Debug)]
pub struct Lfm2 {
  config: TextConfig,
  model: Lfm2Model,
}

impl Lfm2 {
  /// Read-only view of the parsed configuration.
  pub fn config(&self) -> &TextConfig {
    &self.config
  }

  /// `embed_tokens.as_linear(out)` — the tied output head (`lfm2.py:296`):
  /// `out @ embed_tokens.T`, projecting hidden states back to vocab logits.
  fn as_linear(&self, hidden: &Array) -> Result<Array> {
    let wt = swapaxes(&self.model.embed_tokens, -1, -2)?;
    ops::linalg_basic::matmul(hidden, &wt)
  }

  /// `mlx-lm`'s `Model.sanitize` (`lfm2.py:298-306`): transpose any
  /// `conv.weight` stored as PyTorch `(C, 1, K)` to MLX's channels-last
  /// `(C, K, 1)` — i.e. when the last axis exceeds the middle axis. Every
  /// other weight passes through unchanged. Operates on a name → [`Array`]
  /// map in place; the load path applies this before constructing the model.
  pub fn sanitize(weights: &mut std::collections::HashMap<String, Array>) -> Result<()> {
    let conv_keys: Vec<String> = weights
      .keys()
      .filter(|k| k.contains("conv.weight"))
      .cloned()
      .collect();
    for key in conv_keys {
      let needs_transpose = {
        let w = &weights[&key];
        let shape = w.shape();
        // `param.shape[-1] > param.shape[1]` — rank-3 `(C, ?, ?)`.
        shape.len() == 3 && shape[2] > shape[1]
      };
      if needs_transpose {
        let w = &weights[&key];
        // `param.transpose(0, 2, 1)`.
        let t = transpose_axes(w, &[0, 2, 1])?;
        weights.insert(key, t);
      }
    }
    Ok(())
  }

  /// Build the heterogeneous per-layer cache (`make_cache`, `lfm2.py:312-316`):
  /// a [`StandardKvCache`] (`"KVCache"`) for every attention layer and a
  /// one-slot [`ArraysCache`] for every conv layer, in layer order.
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    let n = self.config.num_hidden_layers;
    (0..n)
      .map(|i| -> Box<dyn KvCache> {
        if is_attention_layer(&self.model.attn_idxs, i) {
          Box::new(StandardKvCache::new())
        } else {
          Box::new(ArraysCache::new(1))
        }
      })
      .collect()
  }
}

impl LmModel for Lfm2 {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let hidden = self.model.forward_tokens(tokens, cache)?;
    self.as_linear(&hidden)
  }

  fn forward_embeddings(
    &self,
    embeddings: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    let hidden = self.model.forward_hidden(embeddings, cache)?;
    self.as_linear(&hidden)
  }

  fn supports_input_embeddings(&self) -> bool {
    true
  }
}

// ───────────────────────── weight loading ─────────────────────────

/// Pull a required weight out of the map by `name`, erroring with the key on
/// absence (mlx's `model.update(tree_unflatten(weights))` would raise).
fn take_weight(
  weights: &mut std::collections::HashMap<String, Array>,
  name: &str,
) -> Result<Array> {
  weights.remove(name).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "LFM2 weight map",
      format_smolstr!("{name}"),
    ))
  })
}

impl Lfm2 {
  /// Construct an LFM2 model from a parsed [`TextConfig`] and a flat
  /// name → [`Array`] weight map (already [`sanitize`](Lfm2::sanitize)d).
  ///
  /// Weight keys follow mlx-lm's `model.*` tree: `model.embed_tokens.weight`,
  /// `model.embedding_norm.weight`, and per-layer
  /// `model.layers.{i}.{operator_norm,ffn_norm}.weight`,
  /// `model.layers.{i}.feed_forward.{w1,w2,w3}.weight`, and either the
  /// attention (`self_attn.*`) or conv (`conv.*`) sub-tree. The map is
  /// drained (weights are moved out); a missing required weight is an
  /// [`Error::MissingKey`].
  pub fn from_weights(
    config: TextConfig,
    mut weights: std::collections::HashMap<String, Array>,
  ) -> Result<Lfm2> {
    config.validate()?;
    let attn_idxs = config.attention_layer_indices()?;
    let eps = config.norm_eps;
    let head_dim = config.head_dim();

    let embed_tokens = take_weight(&mut weights, "model.embed_tokens.weight")?;
    let embedding_norm = RMSNorm::new(
      take_weight(&mut weights, "model.embedding_norm.weight")?,
      eps,
    );

    // `num_hidden_layers` is bounded by `MAX_CONFIG_CARDINALITY` in `validate`,
    // but reserve fallibly so even a within-cap heavyweight per-layer `Vec` that
    // the allocator cannot satisfy is a recoverable [`Error::AllocFailure`]
    // rather than `with_capacity`'s abort.
    let mut layers: Vec<Lfm2DecoderLayer> = Vec::new();
    reserve_or_error(
      &mut layers,
      "Lfm2DecoderLayer",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      let p = format!("model.layers.{i}");
      let operator_norm = RMSNorm::new(
        take_weight(&mut weights, &format!("{p}.operator_norm.weight"))?,
        eps,
      );
      let ffn_norm = RMSNorm::new(
        take_weight(&mut weights, &format!("{p}.ffn_norm.weight"))?,
        eps,
      );

      // The feed-forward width is validated eagerly in `validate` (and again by
      // the loaded weight shapes), so no per-layer recomputation is needed here.
      let feed_forward = Mlp {
        w1: Linear::new(
          take_weight(&mut weights, &format!("{p}.feed_forward.w1.weight"))?,
          None,
        ),
        w3: Linear::new(
          take_weight(&mut weights, &format!("{p}.feed_forward.w3.weight"))?,
          None,
        ),
        w2: Linear::new(
          take_weight(&mut weights, &format!("{p}.feed_forward.w2.weight"))?,
          None,
        ),
      };

      let mixer = if is_attention_layer(&attn_idxs, i) {
        let q = format!("{p}.self_attn");
        Mixer::Attention(Attention {
          n_heads: config.num_attention_heads,
          n_kv_heads: config.num_key_value_heads,
          scale: (head_dim as f32).powf(-0.5),
          q_layernorm: RMSNorm::new(
            take_weight(&mut weights, &format!("{q}.q_layernorm.weight"))?,
            eps,
          ),
          k_layernorm: RMSNorm::new(
            take_weight(&mut weights, &format!("{q}.k_layernorm.weight"))?,
            eps,
          ),
          q_proj: Linear::new(
            take_weight(&mut weights, &format!("{q}.q_proj.weight"))?,
            None,
          ),
          k_proj: Linear::new(
            take_weight(&mut weights, &format!("{q}.k_proj.weight"))?,
            None,
          ),
          v_proj: Linear::new(
            take_weight(&mut weights, &format!("{q}.v_proj.weight"))?,
            None,
          ),
          out_proj: Linear::new(
            take_weight(&mut weights, &format!("{q}.out_proj.weight"))?,
            None,
          ),
          rope: Rope::new(head_dim, false, config.rope_theta, 1.0),
        })
      } else {
        let c = format!("{p}.conv");
        let conv_weight = take_weight(&mut weights, &format!("{c}.conv.weight"))?;
        let in_weight = take_weight(&mut weights, &format!("{c}.in_proj.weight"))?;
        let out_weight = take_weight(&mut weights, &format!("{c}.out_proj.weight"))?;
        // `conv_bias` is authoritative (`lfm2.py` builds the conv + its two
        // projections with `bias=config.conv_bias`): when set, all three bias
        // tensors are required; when unset, none may be present (a stray bias
        // key would otherwise be silently applied). `take_if` enforces that
        // gate per tensor — [`Error::MissingKey`] when required-but-absent,
        // [`Error::KeyCollision`] when forbidden-but-present.
        let conv_bias = take_if(
          &mut weights,
          "conv_bias",
          config.conv_bias,
          &format!("{c}.conv.bias"),
        )?;
        let in_bias = take_if(
          &mut weights,
          "conv_bias",
          config.conv_bias,
          &format!("{c}.in_proj.bias"),
        )?;
        let out_bias = take_if(
          &mut weights,
          "conv_bias",
          config.conv_bias,
          &format!("{c}.out_proj.bias"),
        )?;
        Mixer::Conv(ShortConv {
          hidden_size: config.hidden_size,
          l_cache: config.conv_l_cache,
          conv_weight,
          conv_bias,
          in_proj: Linear::new(in_weight, in_bias),
          out_proj: Linear::new(out_weight, out_bias),
        })
      };

      layers.push(Lfm2DecoderLayer {
        mixer,
        feed_forward,
        operator_norm,
        ffn_norm,
      });
    }

    let model = Lfm2Model {
      embed_tokens,
      layers,
      embedding_norm,
      attn_idxs,
    };
    Ok(Lfm2 { config, model })
  }
}

#[cfg(test)]
mod tests;
