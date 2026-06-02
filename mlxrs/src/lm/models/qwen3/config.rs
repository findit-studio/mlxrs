//! Qwen3 text-model configuration.
//!
//! Mirrors `mlx-lm`'s `qwen3.ModelArgs` (`qwen3.py:15-29`): parsed from a
//! `config.json` via [`Qwen3Config::from_json`]. Like mlx-lm's
//! `BaseModelArgs.from_dict`, unmodeled keys are ignored (serde does not
//! `deny_unknown_fields`) and every field carries the reference default so a
//! partial config still parses. The parsed fields are then [`validate`]d so a
//! malformed config (zero / negative / non-divisible / oversized dimension) is
//! a recoverable error here rather than a panic deep in the forward pass.
//!
//! [`validate`]: Qwen3Config::validate

use smol_str::format_smolstr;

use crate::{
  error::{Error, NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, Result},
  model_validation::{
    checked_mul, require_cardinality, require_divisible, require_even, require_in_range,
  },
};

/// Inclusive upper bound for every *width*-like config field (`hidden_size`,
/// `intermediate_size`, `vocab_size`, `head_dim`). `2^24` is far above any real
/// Qwen3 checkpoint (the largest real value, `vocab_size`, is ~1.5·10^5) yet
/// small enough that the downstream `i32` shape arithmetic — the per-head
/// reshapes, the `num_attention_heads * head_dim` query-projection width, and
/// the vocab-width logits projection — cannot overflow `i32`. Rejecting an
/// oversized field here keeps a malformed config a recoverable
/// [`Error::OutOfRange`] instead of a wrapping multiply downstream.
const MAX_CONFIG_DIM: i32 = 1 << 24;

/// Inclusive upper bound for every *cardinality*-like config field —
/// `num_hidden_layers` (a per-layer `Vec` of heavyweight transformer blocks is
/// reserved up front) and the head counts. Unlike the width fields these size
/// *eager allocations* (the decoder-layer `Vec`, the per-layer cache), so the
/// overflow-safe `2^24` width cap is far too loose: `2^24` layers would request
/// a multi-gigabyte `Vec` before the first missing-key error. The largest real
/// Qwen3 has tens of layers and ~10^1 heads; `4096` is generous headroom yet
/// keeps a malformed cardinality a recoverable [`Error::OutOfRange`] /
/// [`Error::CapExceeded`] (or, if it still slips through, a recoverable
/// [`Error::AllocFailure`] from the `try_reserve`d layer `Vec`) rather than an
/// allocator abort.
const MAX_CONFIG_CARDINALITY: i32 = 4096;

/// Qwen3 text-model configuration — a serde-parsed mirror of `mlx-lm`'s
/// `qwen3.ModelArgs` (`qwen3.py:15-29`).
///
/// Distributed-training (`rope_scaling`) and the always-`"qwen3"`
/// `model_type` are carried for parse-completeness; the forward pass uses the
/// dimensions, head counts, `rms_norm_eps`, `rope_theta`, `head_dim`, and
/// `tie_word_embeddings`. `rope_scaling` is parsed as an opaque JSON value (the
/// base Qwen3 checkpoints set it to `null`; a non-null scaling config is a
/// later-phase feature and currently rejected by
/// [`validate`](Qwen3Config::validate)).
#[derive(Debug, Clone, serde::Deserialize)]
#[non_exhaustive]
pub struct Qwen3Config {
  /// Architecture id (`"qwen3"`).
  #[serde(default = "default_model_type")]
  pub model_type: String,
  /// Hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of decoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// MLP feed-forward (gate/up) width.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of attention (query) heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// RMSNorm variance floor.
  #[serde(default = "default_rms_norm_eps")]
  pub rms_norm_eps: f32,
  /// Vocabulary size (logits last-axis width).
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// Number of key/value heads (GQA).
  #[serde(default = "default_num_key_value_heads")]
  pub num_key_value_heads: i32,
  /// Maximum positional context (carried; not used by the forward pass).
  #[serde(default = "default_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// RoPE base frequency (`theta`).
  #[serde(default = "default_rope_theta")]
  pub rope_theta: f32,
  /// Per-head dimension. Unlike many architectures Qwen3 carries this
  /// explicitly (the query projection is `Linear(hidden, n_heads * head_dim)`),
  /// so it is **not** derived from `hidden_size / num_attention_heads`.
  #[serde(default = "default_head_dim")]
  pub head_dim: i32,
  /// Whether the output projection reuses the input embedding table.
  #[serde(default = "default_tie_word_embeddings")]
  pub tie_word_embeddings: bool,
  /// Optional RoPE scaling config (opaque JSON). The base Qwen3 checkpoints set
  /// this to `null`; a non-null value is a later-phase feature currently
  /// rejected by [`validate`](Qwen3Config::validate).
  #[serde(default)]
  pub rope_scaling: Option<serde_json::Value>,
}

fn default_model_type() -> String {
  "qwen3".to_string()
}
fn default_hidden_size() -> i32 {
  2048
}
fn default_num_hidden_layers() -> i32 {
  28
}
fn default_intermediate_size() -> i32 {
  6144
}
fn default_num_attention_heads() -> i32 {
  16
}
fn default_rms_norm_eps() -> f32 {
  1e-6
}
fn default_vocab_size() -> i32 {
  151936
}
fn default_num_key_value_heads() -> i32 {
  8
}
fn default_max_position_embeddings() -> i32 {
  65536
}
fn default_rope_theta() -> f32 {
  1_000_000.0
}
fn default_head_dim() -> i32 {
  128
}
fn default_tie_word_embeddings() -> bool {
  true
}

impl Default for Qwen3Config {
  /// The reference `qwen3.ModelArgs` defaults (the same per-field defaults
  /// serde applies to an absent key). Known-valid — [`validate`](Self::validate)
  /// passes on it.
  fn default() -> Self {
    Self {
      model_type: default_model_type(),
      hidden_size: default_hidden_size(),
      num_hidden_layers: default_num_hidden_layers(),
      intermediate_size: default_intermediate_size(),
      num_attention_heads: default_num_attention_heads(),
      rms_norm_eps: default_rms_norm_eps(),
      vocab_size: default_vocab_size(),
      num_key_value_heads: default_num_key_value_heads(),
      max_position_embeddings: default_max_position_embeddings(),
      rope_theta: default_rope_theta(),
      head_dim: default_head_dim(),
      tie_word_embeddings: default_tie_word_embeddings(),
      rope_scaling: None,
    }
  }
}

impl Qwen3Config {
  /// Parse a [`Qwen3Config`] from a `config.json` string.
  ///
  /// A serde failure (malformed JSON) maps to [`Error::Parse`]; missing keys
  /// fall back to the reference defaults rather than erroring (mlx-lm
  /// `from_dict` semantics). The parsed fields are then [`validate`]d so a
  /// malformed config is a recoverable error here rather than a panic
  /// downstream.
  ///
  /// [`validate`]: Qwen3Config::validate
  pub fn from_json(json: &str) -> Result<Qwen3Config> {
    let cfg: Qwen3Config = serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Qwen3Config::from_json",
        "Qwen3 text config JSON",
        e,
      ))
    })?;
    cfg.validate()?;
    Ok(cfg)
  }

  /// Reject a structurally invalid configuration before it can panic the
  /// forward pass. Mirrors the eager `validate()` discipline of
  /// [`crate::lm::models::lfm2`]'s `TextConfig`.
  ///
  /// Every width-like field (`hidden_size`, `intermediate_size`, `vocab_size`,
  /// `head_dim`) must be a positive integer no larger than `2^24` (so the
  /// downstream `i32` shape arithmetic cannot overflow); the cardinality fields
  /// that size eager allocations (`num_hidden_layers`, the head counts) take
  /// their own far tighter cap. Beyond the per-field bounds:
  ///
  /// - `num_attention_heads` must be divisible by `num_key_value_heads` — the
  ///   grouped-query head grouping (`qwen3.py:39`, and the
  ///   [`scaled_dot_product_attention`](crate::lm::nn::attention::scaled_dot_product_attention)
  ///   kernel's `N_q % N_kv == 0` contract).
  /// - the explicit `head_dim` must be **even** — RoPE rotates feature `k` with
  ///   `k + head_dim/2` (`traditional=false`); an odd `head_dim` loads but only
  ///   fails inside the attention forward pass.
  /// - the query-projection width `num_attention_heads * head_dim` must itself
  ///   stay within the width cap (a checked product), so the per-head reshape /
  ///   `o_proj` input width cannot overflow `i32`.
  /// - `rope_theta` must be **finite and positive** (it is the RoPE
  ///   angular-frequency base) and `rms_norm_eps` must be **finite** (the
  ///   variance floor).
  /// - `rope_scaling` must be absent / `null` — a non-null scaling config is a
  ///   later-phase feature this port does not yet implement, so accepting one
  ///   silently would apply *unscaled* RoPE and diverge from the reference.
  ///
  /// Returns the first violation as [`Error::OutOfRange`] /
  /// [`Error::CapExceeded`] / [`Error::DivisibilityConstraint`] /
  /// [`Error::ArithmeticOverflow`] / [`Error::NonFiniteScalar`]; `Ok(())` when
  /// every field is sound.
  pub fn validate(&self) -> Result<()> {
    // Width-like fields: positive and within the overflow-safe `2^24` cap.
    for (name, value) in [
      ("hidden_size", self.hidden_size),
      ("intermediate_size", self.intermediate_size),
      ("vocab_size", self.vocab_size),
      ("head_dim", self.head_dim),
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
    // GQA: query heads must be an integer multiple of key/value heads.
    require_divisible(
      "num_attention_heads",
      self.num_attention_heads,
      "num_key_value_heads",
      self.num_key_value_heads,
    )?;
    // RoPE rotates feature `k` with `k + head_dim/2`, so `head_dim` must be
    // even; an odd one loads but only fails inside the attention forward pass.
    require_even("head_dim", self.head_dim)?;
    // The query projection emits `num_attention_heads * head_dim` features; a
    // checked product keeps it within the width cap so the per-head reshape and
    // `o_proj` input width cannot overflow `i32`.
    let q_width = checked_mul(
      "Qwen3Config::validate: num_attention_heads * head_dim",
      "num_attention_heads",
      self.num_attention_heads,
      "head_dim",
      self.head_dim,
    )?;
    require_in_range("num_attention_heads * head_dim", q_width, 1, MAX_CONFIG_DIM)?;
    // `rope_theta` is the RoPE angular-frequency base — a non-finite or
    // non-positive base would produce NaN/garbage rotations.
    if !self.rope_theta.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Qwen3Config::validate (rope_theta)",
        f64::from(self.rope_theta),
      )));
    }
    if self.rope_theta <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3Config::validate (rope_theta)",
        "must be a positive float",
        format_smolstr!("rope_theta={}", self.rope_theta),
      )));
    }
    // `rms_norm_eps` is the variance floor under the rsqrt — a non-finite value
    // would propagate NaN through every norm.
    if !self.rms_norm_eps.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Qwen3Config::validate (rms_norm_eps)",
        f64::from(self.rms_norm_eps),
      )));
    }
    // A non-null `rope_scaling` is a later-phase feature; reject it rather than
    // silently applying unscaled RoPE.
    if let Some(scaling) = &self.rope_scaling
      && !scaling.is_null()
    {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3Config::validate (rope_scaling)",
        "RoPE scaling is not yet supported (must be null/absent)",
        format_smolstr!("rope_scaling={scaling}"),
      )));
    }
    Ok(())
  }
}
