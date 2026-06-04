//! LFM2.5-VL pixel-unshuffle + multimodal projector and the image-feature
//! merge — faithful 1:1 port of the projector / `PixelUnshuffleBlock` /
//! `merge_input_ids_with_image_features` of
//! `mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py`.
//!
//! ## Pixel unshuffle ([`PixelUnshuffleBlock`])
//!
//! `lfm2_vl.py`'s `PixelUnshuffleBlock(factor)` is a pure pad / reshape /
//! transpose that folds a `factor x factor` patch neighborhood into the channel
//! axis: an `(N, W, H, C)` grid becomes `(N, W/factor, H/factor, C·factor²)`
//! (each odd spatial dim is first zero-padded up to a multiple of `factor`). For
//! the base checkpoint (`factor = 2`, vision `hidden = 768`) the projector input
//! width is `768 · 2² = 3072`. No new primitive is introduced — only the present
//! [`crate::ops::shape::concatenate`] (the zero pad),
//! [`crate::ops::shape::reshape`], and
//! [`crate::ops::shape::transpose_axes`] ops.
//!
//! ## Projector ([`Lfm2VlMultiModalProjector`])
//!
//! `lfm2_vl.py`'s `Lfm2VlMultiModalProjector` maps the unshuffled vision width
//! into the LM's token space: an optional `LayerNorm(in)` (when
//! `projector_use_layernorm`), then `Linear(in, projector_hidden) → gelu →
//! Linear(projector_hidden, text_hidden)`. Both `Linear`s are built with
//! `bias=config.projector_bias` (`lfm2_vl.py:20-30`), so `projector_bias` is the
//! authoritative gate on the two `linear_*.bias` tensors — modeled here through
//! the [`take_if`] +
//! [`MaybeQuantizedLinear::from_weights_with_bias`](crate::nn::MaybeQuantizedLinear::from_weights_with_bias)
//! seam (required when `true`, forbidden when `false`) rather than an opportunistic
//! auto-consume. Both `Linear`s are routed through the shared quantize-aware
//! [`crate::nn::MaybeQuantizedLinear`], so the 8-bit
//! `LiquidAI/LFM2.5-VL-450M-MLX-8bit` checkpoint loads through the same code
//! path as a dense one (each layer's `.scales` sibling is the load-bearing
//! "this layer is quantized" signal). The activation is `nn.gelu` — the EXACT
//! erf GELU ([`crate::lm::nn::activations::gelu`]), distinct from the vision
//! tower's `nn.GELU(approx="precise")` tanh approximation.
//!
//! ## Image-feature merge ([`merge_input_ids_with_image_features`])
//!
//! `lfm2_vl.py`'s `merge_input_ids_with_image_features` builds
//! `special_image_mask = (input_ids == image_token_index)`, asserts the masked
//! position count equals the projected-feature row count (a typed error on
//! mismatch, mirroring `lfm2_vl.py:173-176`), and scatters the feature rows into
//! the masked positions (`masked_scatter`, `lfm2_vl.py:75-93`).
//!
//! ### Why a faithful masked splice and not the span-replace seam
//!
//! mlxrs already ships a default
//! [`merge_embeddings`](crate::vlm::model::Model::merge_embeddings) that
//! span-replaces a caller-supplied set of contiguous `(start, end)` ranges. For
//! LFM2.5-VL's contiguous `<image>` runs (the chat template brackets each image
//! with `image_start … image_token×N … image_end`) the two produce identical
//! output. The reference's entry point, however, is **mask-driven** — it derives
//! the splice positions directly from `input_ids == image_token_index` rather
//! than from pre-extracted spans, and its contract is the whole-mask count check
//! at `lfm2_vl.py:173-176`. This module ports that contract verbatim: it
//! consumes `input_ids` + `image_token_index` (no span pre-extraction), and its
//! splice is a general `where(mask, scattered_features, inputs_embeds)` that
//! matches `masked_scatter` for **any** mask layout (not only contiguous runs).
//! The splice stays fully lazy (no in-place mutation, faithful to mlx's
//! lazy-graph contract): the masked flat positions are read host-side (the same
//! tiny-array host read the vision tower does for `spatial_shapes`) only to
//! build the per-position gather index, and every tensor step appends to the
//! graph.

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result},
  lm::nn::{activations::gelu, norm::LayerNorm},
  model_validation::{checked_mul, require_positive, reserve_or_error, take_if},
  nn::MaybeQuantizedLinear,
  ops::{
    self,
    comparison::equal,
    indexing::take_axis,
    logical::select,
    shape::{broadcast_to, concatenate, expand_dims_axes, reshape, transpose_axes},
  },
  vlm::models::lfm2_vl::config::ModelConfig,
};

/// A per-layer quantization resolver: maps a layer's module path to its
/// `(group_size, bits, mode)` scheme (or `None` when that layer is dense). The
/// mode string's lifetime `'q` is the parsed config's (the resolver borrows the
/// mode from the config, not from the queried path), the same shape as the LFM2
/// LM's `quant_for` closure and the vision tower's resolver. A resolver
/// returning `None` everywhere loads a dense checkpoint unchanged.
type QuantResolver<'q> = dyn Fn(&str) -> Option<(i32, i32, &'q str)> + 'q;

// ═══════════════════════════ PixelUnshuffleBlock ═══════════════════════════

/// `lfm2_vl.py`'s `PixelUnshuffleBlock(factor)` — a pure pad / reshape /
/// transpose that folds a `factor x factor` spatial neighborhood into the
/// channel axis: `(N, W, H, C)` → `(N, W/factor, H/factor, C·factor²)`.
///
/// Each odd spatial dim is first zero-padded up to the next multiple of
/// `factor` (the reference's `mx.concatenate([x, mx.zeros(...)], axis=…)`), so a
/// `(1, 5, 5, 768)` grid with `factor = 2` pads to `(1, 6, 6, 768)` and unfolds
/// to `(1, 3, 3, 3072)`.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug, Clone, Copy)]
pub struct PixelUnshuffleBlock {
  /// The downsample factor (`downsample_factor`, `2` for the base checkpoint).
  factor: i32,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl PixelUnshuffleBlock {
  /// Construct from the downsample `factor`. `factor` must be `>= 1` (the
  /// reference only constructs the block when `downsample_factor > 1`; `1` is a
  /// structural no-op the caller substitutes `Identity` for, but it is accepted
  /// here so the block is total).
  ///
  /// # Errors
  /// [`Error::OutOfRange`] if `factor < 1`.
  pub fn new(factor: i32) -> Result<Self> {
    require_positive("lfm2_vl PixelUnshuffleBlock: factor", factor)?;
    Ok(Self { factor })
  }

  /// The downsample factor.
  #[inline(always)]
  pub fn factor(&self) -> i32 {
    self.factor
  }

  /// `(N, W, H, C)` → `(N, W/factor, H/factor, C·factor²)`.
  ///
  /// Zero-pads `W` and/or `H` up to a multiple of `factor` when odd, then
  /// reshapes + transposes to fold the `factor x factor` neighborhood into the
  /// channel axis (`lfm2_vl.py:45-72`).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `x` is not rank-4 `(N, W, H, C)`;
  /// - [`Error::ArithmeticOverflow`] if a folded width
  ///   (`C·factor`, `C·factor²`) overflows `i32`;
  /// - propagates the pad / reshape / transpose op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 4 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl PixelUnshuffleBlock: input must be rank-4 (N, W, H, C)",
        shape.len() as u32,
        shape,
      )));
    }
    let n = dim_i32(&shape, 0, "lfm2_vl PixelUnshuffleBlock: N")?;
    let mut w = dim_i32(&shape, 1, "lfm2_vl PixelUnshuffleBlock: W")?;
    let mut h = dim_i32(&shape, 2, "lfm2_vl PixelUnshuffleBlock: H")?;
    let c = dim_i32(&shape, 3, "lfm2_vl PixelUnshuffleBlock: C")?;
    let f = self.factor;

    // Zero-pad W (axis 1) up to a multiple of `factor` when odd:
    // `mx.concatenate([x, mx.zeros((n, f - (w % f), h, c))], axis=1)`.
    let mut x = if w % f != 0 {
      let pad = f - (w % f);
      let zeros = zeros_like_dtype(&[n, pad, h, c], x.dtype()?)?;
      let out = concatenate(&[x, &zeros], 1)?;
      w += pad;
      out
    } else {
      x.try_clone()?
    };

    // Zero-pad H (axis 2) up to a multiple of `factor` when odd:
    // `mx.concatenate([x, mx.zeros((n, w, f - (h % f), c))], axis=2)`.
    if h % f != 0 {
      let pad = f - (h % f);
      let zeros = zeros_like_dtype(&[n, w, pad, c], x.dtype()?)?;
      x = concatenate(&[&x, &zeros], 2)?;
      h += pad;
    }

    // `x.reshape(n, w, h/f, c*f)` — fold one factor of H into C.
    let h_div = h / f;
    let c_f = checked_mul("lfm2_vl PixelUnshuffleBlock: C*factor", "C", c, "factor", f)?;
    let x = reshape(&x, &[n, w, h_div, c_f])?;
    // `x.transpose(0, 2, 1, 3)` — (N, H/f, W, C*f).
    let x = transpose_axes(&x, &[0, 2, 1, 3])?;
    // `x.reshape(n, h/f, w/f, c*f**2)` — fold the remaining factor of W into C.
    let w_div = w / f;
    let c_f2 = checked_mul(
      "lfm2_vl PixelUnshuffleBlock: C*factor^2",
      "C*factor",
      c_f,
      "factor",
      f,
    )?;
    let x = reshape(&x, &[n, h_div, w_div, c_f2])?;
    // `x.transpose(0, 2, 1, 3)` — (N, W/f, H/f, C*f^2).
    transpose_axes(&x, &[0, 2, 1, 3])
  }
}

// ════════════════════════ Lfm2VlMultiModalProjector ════════════════════════

/// `lfm2_vl.py`'s `Lfm2VlMultiModalProjector` — the vision-to-LM-token-space
/// projector: an optional `LayerNorm(in)` (when `projector_use_layernorm`), then
/// `Linear(in, projector_hidden) → gelu → Linear(projector_hidden,
/// text_hidden)`.
///
/// `in = vision_hidden · downsample_factor²` (the [`PixelUnshuffleBlock`]
/// output width). Both `Linear`s route through the shared quantize-aware
/// [`MaybeQuantizedLinear`] (per-layer `.scales` auto-detect), so the 8-bit
/// checkpoint loads through the same path as a dense one. The activation is
/// `nn.gelu` (the exact erf GELU), not the vision tower's tanh `precise`
/// approximation.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct Lfm2VlMultiModalProjector {
  /// The input `LayerNorm` (present iff `projector_use_layernorm`). Built with
  /// mlx's `nn.LayerNorm` default `eps` (`1e-5`) — the reference constructs it
  /// as `nn.LayerNorm(in_channels)` with no explicit `eps`.
  layer_norm: Option<LayerNorm>,
  linear_1: MaybeQuantizedLinear,
  linear_2: MaybeQuantizedLinear,
}

/// mlx's `nn.LayerNorm` default `eps` (`1e-5`) — the projector's `LayerNorm` is
/// built with no explicit `eps` (`lfm2_vl.py:19`).
#[cfg(feature = "lfm2-vl")]
const LAYERNORM_DEFAULT_EPS: f32 = 1e-5;

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl Lfm2VlMultiModalProjector {
  /// Build the projector from a validated [`ModelConfig`] and the
  /// (already-sanitized) weight map.
  ///
  /// Weight keys follow `lfm2_vl.py`'s module tree under the projector prefix
  /// (the VL-level `multi_modal_projector` prefix already stripped):
  /// `layer_norm.{weight,bias}` (present iff `projector_use_layernorm`),
  /// `linear_1.{weight,bias,scales,biases}`,
  /// `linear_2.{weight,bias,scales,biases}`. The `linear_*.bias` presence is
  /// gated by the authoritative `projector_bias` config flag
  /// (`lfm2_vl.py:20-30` builds both `Linear`s with `bias=config.projector_bias`):
  /// when `projector_bias`, each `.bias` is REQUIRED; when not, a stray `.bias`
  /// is rejected. The bias is taken through the
  /// [`take_if`] gate and applied on both the
  /// dense and quantized paths via
  /// [`MaybeQuantizedLinear::from_weights_with_bias`] (NOT the auto-consuming
  /// `from_weights`).
  ///
  /// `quant` resolves a layer path's `(group_size, bits, mode)` from the parsed
  /// quantization config (`None` for a dense layer); a quantized layer is
  /// detected by the presence of its `<prefix>.scales` sibling, so passing a
  /// resolver that returns `None` everywhere (or `&|_| None`) loads a dense
  /// checkpoint unchanged.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] for an absent required weight (the `linear_*`
  ///   weights, the `layer_norm` affine params when `projector_use_layernorm`,
  ///   or a `linear_*.bias` required by `projector_bias`);
  /// - [`Error::KeyCollision`] for a stray `linear_*.bias` present when
  ///   `projector_bias` is `false`;
  /// - propagates the [`MaybeQuantizedLinear`] quantized-triple validation.
  pub fn from_weights(
    config: &ModelConfig,
    weights: &mut HashMap<String, Array>,
    quant: &QuantResolver<'_>,
  ) -> Result<Self> {
    // Idempotent re-validation (the constructor is public): pins the projector
    // dims + both tower configs before any tensor is built.
    config.validate()?;

    let layer_norm = if config.projector_use_layernorm {
      let weight = take_weight(weights, "layer_norm.weight")?;
      let bias = take_weight(weights, "layer_norm.bias")?;
      Some(LayerNorm::new(
        Some(weight),
        Some(bias),
        LAYERNORM_DEFAULT_EPS,
      ))
    } else {
      None
    };

    // The two projection widths are read from the checkpoint shapes by the
    // quantize-aware builder (it needs no per-axis width); the config dims are
    // validated above and pinned again by the loaded weight shapes.
    //
    // `projector_bias` is authoritative (`lfm2_vl.py:20-30` builds both
    // projector `Linear`s with `bias=config.projector_bias`): when set, each
    // REQUIRES its dense `.bias`; when unset, a stray projection bias is a
    // collision. `take_if` enforces that gate and TAKES the bias, which is then
    // passed explicitly to `from_weights_with_bias` (which does NOT auto-consume
    // an optional `.bias`), applied on both the dense and quantized paths — so
    // the dense-bias arity matches the reference whether the projection is dense
    // or quantized, mirroring the LFM2 LM's `conv_bias` gating. The plain
    // `from_weights` would instead silently apply a stray bias (or silently omit
    // a required one), ignoring `projector_bias`.
    let linear_1_bias = take_if(
      weights,
      "projector_bias",
      config.projector_bias,
      "linear_1.bias",
    )?;
    let linear_1 = MaybeQuantizedLinear::from_weights_with_bias(
      weights,
      "linear_1",
      quant("linear_1"),
      linear_1_bias,
    )?;
    let linear_2_bias = take_if(
      weights,
      "projector_bias",
      config.projector_bias,
      "linear_2.bias",
    )?;
    let linear_2 = MaybeQuantizedLinear::from_weights_with_bias(
      weights,
      "linear_2",
      quant("linear_2"),
      linear_2_bias,
    )?;

    Ok(Self {
      layer_norm,
      linear_1,
      linear_2,
    })
  }

  /// `(..., in) → (..., text_hidden)`: `linear_2(gelu(linear_1(layer_norm?(x))))`
  /// (`lfm2_vl.py:32-37`).
  ///
  /// # Errors
  /// Propagates the LayerNorm / Linear / gelu op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let h = match &self.layer_norm {
      Some(ln) => ln.forward(x)?,
      None => x.try_clone()?,
    };
    let h = self.linear_1.forward(&h)?;
    let h = gelu(&h)?;
    self.linear_2.forward(&h)
  }

  /// `true` if `linear_1` was loaded from a quantized checkpoint (the projector
  /// quantizes both `Linear`s or neither, matching the checkpoint).
  #[cfg(test)]
  pub fn is_quantized(&self) -> bool {
    self.linear_1.is_quantized()
  }
}

// ══════════════════ merge_input_ids_with_image_features ═════════════════════

/// `lfm2_vl.py`'s `merge_input_ids_with_image_features` (`:162-182`) + the
/// `masked_scatter` it calls (`:75-93`): splice the projected `image_features`
/// into `inputs_embeds` at the `<image>`-token positions.
///
/// - `image_features` — the projected `(N, D)` image rows (post-projector,
///   already in LM token space; `D` is the LM hidden width).
/// - `inputs_embeds` — the text embeddings `(B, T, D)` (`embed_tokens(input_ids)`).
/// - `input_ids` — the `(B, T)` integer token ids.
/// - `image_token_index` — the `<image>` placeholder id (`396` for the base
///   checkpoint; [`ModelConfig::image_token_index`]).
///
/// Builds `special_image_mask = (input_ids == image_token_index)`, asserts the
/// masked-position count equals `N` (the feature row count) — a typed
/// [`Error::LengthMismatch`] on mismatch, mirroring the reference's
/// `ValueError` at `lfm2_vl.py:173-176` — and replaces every masked position's
/// embedding with the matching feature row (in row-major mask order). The result
/// is `(B, T, D)`.
///
/// The splice is fully lazy (no in-place mutation): the masked flat positions
/// are read host-side (a tiny `(B, T)` bool read) only to build the
/// per-position gather index, then the merged embeddings are assembled with
/// [`take_axis`] + [`select`] — every tensor step appends to the lazy graph,
/// faithful to mlx's lazy-graph contract and to `masked_scatter`'s semantics for
/// **any** mask layout.
///
/// # Errors
/// - [`Error::RankMismatch`] if `inputs_embeds` is not rank-3 `(B, T, D)`,
///   `input_ids` is not rank-2 `(B, T)`, or `image_features` is not rank-2
///   `(N, D)`;
/// - [`Error::LengthMismatch`] if `input_ids` / `inputs_embeds` disagree on
///   `(B, T)`, the hidden widths `D` differ, or the masked-position count `!= N`;
/// - propagates the comparison / gather / select op errors.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn merge_input_ids_with_image_features(
  image_features: &Array,
  inputs_embeds: &Array,
  input_ids: &Array,
  image_token_index: i32,
) -> Result<Array> {
  let emb_shape = inputs_embeds.shape();
  if emb_shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfm2_vl merge: inputs_embeds must be rank-3 (B, T, D)",
      emb_shape.len() as u32,
      emb_shape,
    )));
  }
  let ids_shape = input_ids.shape();
  if ids_shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfm2_vl merge: input_ids must be rank-2 (B, T)",
      ids_shape.len() as u32,
      ids_shape,
    )));
  }
  let feat_shape = image_features.shape();
  if feat_shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfm2_vl merge: image_features must be rank-2 (N, D)",
      feat_shape.len() as u32,
      feat_shape,
    )));
  }
  let (b, t, d) = (emb_shape[0], emb_shape[1], emb_shape[2]);
  // `input_ids` and `inputs_embeds` must share (B, T) — a mismatch would
  // otherwise build a mask whose flat layout disagrees with the embeddings.
  if ids_shape[0] != b || ids_shape[1] != t {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl merge: input_ids (B, T) vs inputs_embeds (B, T) cell count",
      b.saturating_mul(t),
      ids_shape[0].saturating_mul(ids_shape[1]),
    )));
  }
  let n = feat_shape[0];
  if feat_shape[1] != d {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl merge: hidden width D (image_features vs inputs_embeds)",
      d,
      feat_shape[1],
    )));
  }

  // `special_image_mask = (input_ids == image_token_index)` — a (B, T) bool.
  // The scalar is built in `input_ids`' dtype so the broadcast equality is
  // exact (token ids are i32).
  let token_scalar = scalar_i32_like(image_token_index, input_ids)?;
  let mask_2d = equal(input_ids, &token_scalar)?;

  // Read the masked flat positions host-side (a tiny (B, T) bool array — the
  // same host read the vision tower does for `spatial_shapes`). These index the
  // per-position gather built below; the splice itself stays lazy.
  let masked_positions = masked_flat_positions(&mask_2d, b, t)?;

  // The masked-position count must equal the feature row count `N` — the
  // reference's `if n_image_mask_elements != image_features.size` check
  // (`lfm2_vl.py:173-176`; its `.size` is the broadcast `N*D` element count,
  // i.e. `N` masked rows once the per-D broadcast is divided out — equivalently
  // the masked (B, T) position count here).
  if masked_positions.len() != n {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl merge: masked <image> position count vs image feature rows N \
         (image features and image tokens do not match)",
      n,
      masked_positions.len(),
    )));
  }

  // No `<image>` tokens AND no features ⇒ the merged embeddings are exactly the
  // text embeddings (the all-text path; the reference's `masked_scatter` over an
  // all-false mask is a no-op). Returning the embeddings unchanged avoids a
  // degenerate zero-row gather.
  if n == 0 {
    return inputs_embeds.try_clone();
  }

  // Build the per-position gather index `(B*T,)`: the k-th masked flat position
  // gathers feature row `k`; every non-masked position gathers row 0 (its
  // gathered value is discarded by the `select` below). `select` then picks the
  // gathered feature at masked positions and the original embedding elsewhere —
  // the lazy analogue of `masked_scatter`'s flat row-major assignment.
  let cell_count = checked_mul_usize("lfm2_vl merge: B*T", b, t)?;
  let mut gather_idx: Vec<i32> = Vec::new();
  reserve_or_error(&mut gather_idx, "lfm2_vl merge gather index", cell_count)?;
  gather_idx.resize(cell_count, 0);
  for (k, &pos) in masked_positions.iter().enumerate() {
    // `k < n <= cell_count` fits i32 (cell_count is a valid array element count,
    // bounded by i32::MAX shape arithmetic); `pos < cell_count`.
    gather_idx[pos] = k as i32;
  }
  let gather_arr = Array::from_slice::<i32>(&gather_idx, &(cell_count,))?;
  // `image_features[gather_idx]` → (B*T, D), reshaped to (B, T, D).
  let gathered = take_axis(image_features, &gather_arr, 0)?;
  let gathered = reshape(&gathered, &[b as i32, t as i32, d as i32])?;

  // `select(mask[..., None].broadcast(B, T, D), gathered, inputs_embeds)`.
  let mask_3d = expand_dims_axes(&mask_2d, &[-1])?;
  let mask_3d = broadcast_to(&mask_3d, &[b as i32, t as i32, d as i32])?;
  select(&mask_3d, &gathered, inputs_embeds)
}

// ───────────────────────── helpers ─────────────────────────

/// Evaluate the `(B, T)` bool `mask` host-side and return the masked flat
/// positions (row-major), each in `[0, B*T)`. The flat order matches the
/// reference's `np.where(image_mask_expanded_flattened)[0]` row-major scan.
#[cfg(feature = "lfm2-vl")]
fn masked_flat_positions(mask: &Array, b: usize, t: usize) -> Result<Vec<usize>> {
  let cell_count = checked_mul_usize("lfm2_vl merge: B*T", b, t)?;
  // The mask is a tiny `(B, T)` array; eval + read it host-side (a cast to
  // i32 so the host read is over a concrete integer dtype).
  let mut m = ops::misc::astype(mask, Dtype::I32)?;
  m.eval()?;
  let flags = m.to_vec::<i32>()?;
  if flags.len() != cell_count {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "lfm2_vl merge: evaluated mask element count vs B*T",
      cell_count,
      flags.len(),
    )));
  }
  let mut positions: Vec<usize> = Vec::new();
  // Bound the position buffer to the cell count (the worst case is an all-true
  // mask); a fallible reservation keeps an OOM recoverable.
  reserve_or_error(&mut positions, "lfm2_vl merge masked position", cell_count)?;
  for (i, &flag) in flags.iter().enumerate() {
    if flag != 0 {
      positions.push(i);
    }
  }
  Ok(positions)
}

/// A rank-0 `i32` scalar carrying `value`, in the same dtype as `like` (token
/// ids are i32) so the broadcast `equal` is exact. Built as a 1-element array
/// (mlx broadcasts a `(1,)` operand against any shape).
#[cfg(feature = "lfm2-vl")]
fn scalar_i32_like(value: i32, like: &Array) -> Result<Array> {
  let scalar = Array::from_slice::<i32>(&[value], &(1usize,))?;
  let dtype = like.dtype()?;
  if dtype == Dtype::I32 {
    Ok(scalar)
  } else {
    ops::misc::astype(&scalar, dtype)
  }
}

/// A zeros array of the given shape in `dtype` (the pad block's dtype must match
/// `x` for the `concatenate`).
#[cfg(feature = "lfm2-vl")]
fn zeros_like_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
  let z = Array::zeros::<f32>(&shape)?;
  if dtype == Dtype::F32 {
    Ok(z)
  } else {
    ops::misc::astype(&z, dtype)
  }
}

/// Pop a required weight by exact key from the (sanitized) checkpoint map,
/// erroring with the key if absent.
#[cfg(feature = "lfm2-vl")]
fn take_weight(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(crate::error::MissingKeyPayload::new(
      "lfm2_vl::Lfm2VlMultiModalProjector::from_weights",
      key,
    ))
  })
}

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "lfm2-vl")]
fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      smol_str::format_smolstr!("{d}"),
    ))
  })
}

/// `a * b` as `usize`, with a typed [`Error::ArithmeticOverflow`] on overflow.
#[cfg(feature = "lfm2-vl")]
fn checked_mul_usize(context: &'static str, a: usize, b: usize) -> Result<usize> {
  a.checked_mul(b).ok_or_else(|| {
    Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::with_operands(
      context,
      "usize",
      [("a", a as u64), ("b", b as u64)],
    ))
  })
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
