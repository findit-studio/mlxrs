//! The SenseVoice-Small SANM (Self-Attention Network with FSMN memory)
//! encoder — a faithful port of the five reference blocks in
//! [`sensevoice.py`][sv]:
//!
//! - [`SinusoidalPositionEncoder`] — additive absolute sinusoid
//!   (`sensevoice.py:106-122`), with the swift truncate/pad + dtype-cast guards
//!   (`SenseVoiceModel.swift:8-31`).
//! - [`PositionwiseFeedForward`] — `w_2(relu(w_1(x)))`, ReLU not GELU
//!   (`sensevoice.py:125-132`).
//! - [`MultiHeadedAttentionSANM`] — the one novel block: a fused QKV projection,
//!   a parallel depthwise-FSMN memory branch over the value sequence, and the
//!   `att_out + fsmn_memory` post-projection sum (`sensevoice.py:135-198`).
//! - [`EncoderLayerSANM`] — pre-norm, with the no-residual-on-width-change rule
//!   (the residual around attention is dropped iff `in_size != size`,
//!   `sensevoice.py:201-237`).
//! - [`Encoder`] — the tower: `encoders0` (one width-changing `560 -> 512`
//!   block) + `encoders` (`num_blocks - 1`) + `after_norm` + `tp_encoders`
//!   (`tp_blocks`) + `tp_norm`, fronted by an `xs * sqrt(output_size)` scale and
//!   the sinusoidal PE (`sensevoice.py:240-338`).
//!
//! Every `nn.Linear` (`linear_q_k_v`, `linear_out`, `feed_forward.w_1/w_2`)
//! is routed through the shared quantize-aware
//! [`crate::nn::MaybeQuantizedLinear`], auto-detecting a quantized checkpoint by
//! the per-layer `<prefix>.scales` sibling. The depthwise FSMN conv stays dense
//! (mlx does not quantize conv).
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    Error, LengthMismatchPayload, MissingKeyPayload, OutOfRangePayload, RankMismatchPayload,
    Result, ShapePairMismatchPayload,
  },
  lm::{
    nn::{
      attention::{Mask, scaled_dot_product_attention},
      norm::LayerNorm,
    },
    quant::PerLayerQuantization,
  },
  model_validation::{checked_mul, reserve_or_error},
  nn::MaybeQuantizedLinear,
  ops,
};

use super::config::EncoderConfig;

/// Resolve the `(group_size, bits, mode)` tuple the shared quantize-aware
/// builders ([`MaybeQuantizedLinear::from_weights`]) take for one consumed
/// `prefix`, from the parsed [`PerLayerQuantization`] — the qwen3 per-prefix
/// resolution ([`crate::lm::models::qwen3`] /
/// [`PerLayerQuantization::quantization_for`]).
///
/// Returns `None` for a dense load (`quant == None`), a per-layer
/// [`Skip`](crate::lm::quant::QuantizationOption::Skip) override, OR no global
/// default with no override for this layer — exactly the cases the per-layer
/// builder treats as "build the dense arm". When a `<prefix>.scales` is
/// nevertheless present, the shared builder rejects the mismatch with a typed
/// [`Error::InvariantViolation`] (the weights say quantized, the config resolved
/// no scheme for THIS layer). The resolved tuple is per-prefix, so a per-layer
/// parameter override builds that layer with its own `(group_size, bits, mode)`
/// rather than a single collapsed global tuple.
///
/// `quantization_for` returns an owned `Copy` [`crate::lm::quant::Quantization`]
/// and `mode.as_str()` is `&'static`, so the borrowed tuple has no lifetime tie
/// to the resolver.
pub(super) fn resolve_layer_quant(
  quant: Option<&PerLayerQuantization>,
  prefix: &str,
) -> Option<(i32, i32, &'static str)> {
  quant
    .and_then(|q| q.quantization_for(prefix))
    .map(|q| (q.group_size, q.bits, q.mode.as_str()))
}

/// LayerNorm variance floor. The reference `nn.LayerNorm(dim)` uses mlx's
/// default `eps = 1e-5` (`mlx/python/mlx/nn/layers/normalization.py`).
const LAYERNORM_EPS: f32 = 1e-5;

/// The constant-pad mode C-string for the FSMN time-axis padding
/// (`mx.pad` defaults to zero-fill, `sensevoice.py:171-174`).
const PAD_CONSTANT: &std::ffi::CStr = c"constant";

/// Pop a required `<key>` tensor from the weight map, or a typed
/// [`Error::MissingKey`] naming the absent key.
fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "sensevoice encoder: required weight not found in checkpoint",
      key.to_string(),
    ))
  })
}

/// Build a [`LayerNorm`] from `<prefix>.weight` / `<prefix>.bias`
/// (`nn.LayerNorm` always carries both), pinning each affine vector's length to
/// `dim` at load. `eps` is fixed at [`LAYERNORM_EPS`].
///
/// `nn.LayerNorm(dim)` carries `weight` / `bias` 1-D vectors of length `dim`
/// (`sensevoice.py:214-215` etc.); a stale / mis-sized affine vector would
/// broadcast silently against the `(B, T, dim)` activations (or fail deep in the
/// norm op) — pin both lengths here so a wrong shard reports a typed error at
/// load.
///
/// # Errors
/// - [`Error::MissingKey`] for an absent `weight` / `bias`;
/// - [`Error::LengthMismatch`] if either vector's length (its sole axis) is not
///   `dim`.
fn build_layer_norm(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  dim: i32,
) -> Result<LayerNorm> {
  let weight = take(weights, &format!("{prefix}.weight"))?;
  pin_vector_len(
    &weight,
    dim,
    "sensevoice encoder: LayerNorm weight length vs dim",
  )?;
  let bias = take(weights, &format!("{prefix}.bias"))?;
  pin_vector_len(
    &bias,
    dim,
    "sensevoice encoder: LayerNorm bias length vs dim",
  )?;
  Ok(LayerNorm::new(Some(weight), Some(bias), LAYERNORM_EPS))
}

/// Pin a [`MaybeQuantizedLinear`]'s logical `(out_features, in_features)` shape
/// to the config-derived `(out, in)` at load — the SANM/FFN linear analogue of
/// the head shape-pin in [`super::model::build_head`].
///
/// `mlx.nn.Linear` stores `weight` as `(out_features, in_features)`; a stale /
/// hostile shard whose width disagrees with the config would otherwise only
/// mis-project (or fail deep in the matmul) at the first forward. The quantized
/// arm is pinned identically through its dequantized logical shape (no
/// materialization). The optional dense `<prefix>.bias` is pinned alongside the
/// weight via [`pin_dense_linear_bias`], so a dense linear whose bias would
/// broadcast a single wrong offset across every channel is rejected here too.
/// Reads only `shape()` metadata (no eval).
///
/// # Errors
/// - [`Error::ShapePairMismatch`] if the logical shape is not `(out, in)`;
/// - [`Error::RankMismatch`] / [`Error::LengthMismatch`] if the dense bias is
///   present but is not rank-1 with length `out` (via [`pin_dense_linear_bias`]);
/// - propagates [`MaybeQuantizedLinear::logical_shape`]'s rank / range errors.
fn pin_linear_shape(
  linear: &MaybeQuantizedLinear,
  out: i32,
  in_features: i32,
  context: &'static str,
) -> Result<()> {
  let shape = linear.logical_shape()?;
  if shape != (out, in_features) {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      context,
      vec![out.max(0) as usize, in_features.max(0) as usize],
      vec![shape.0.max(0) as usize, shape.1.max(0) as usize],
    )));
  }
  // Pin the optional dense bias to `(out,)` alongside the weight (see
  // `pin_dense_linear_bias`).
  pin_dense_linear_bias(linear, out)
}

/// The dense output bias of a [`MaybeQuantizedLinear`] (`<prefix>.bias`,
/// singular) — `Some` only when the checkpoint carried one, for either arm.
///
/// The shared enum exposes no unified `bias()` accessor (it would have to live
/// in `crate::nn::quantized`, which is shared by every model and must not be
/// changed for one model), so this reads the per-arm
/// [`crate::nn::Linear::bias`] / [`crate::nn::QuantizedLinear::bias`] accessor
/// locally. Distinct from the quantized per-group `biases` (the dense `bias` is
/// the layer's `Linear.bias`).
fn dense_linear_bias(linear: &MaybeQuantizedLinear) -> Option<&Array> {
  match linear {
    MaybeQuantizedLinear::Dense(l) => l.bias(),
    MaybeQuantizedLinear::Quantized(q) => q.bias(),
  }
}

/// Pin a [`MaybeQuantizedLinear`]'s optional dense output bias to rank-1
/// `(out_features,)` at load.
///
/// `mlx.nn.Linear` stores its bias as a 1-D `(out_features,)` vector and adds it
/// to the `(..., out_features)` projection. A stale / hostile shard whose dense
/// bias is shaped `(1,)` (or `(1, out)`, or any other length) broadcasts a
/// single wrong offset across every output channel — a SILENT wrong output that
/// no weight-shape pin catches — or fails only deep in the add at the first
/// forward. The quantized arm already validates this dense bias in
/// [`crate::nn::QuantizedLinear::from_parts`]; the DENSE arm of the shared
/// [`MaybeQuantizedLinear`] does NOT, so SenseVoice pins it here for every dense
/// linear in its load path. When no dense bias is present (the common
/// `bias=False` case for the SANM/FFN projections in a clean checkpoint) this is
/// a no-op. Reads only `shape()` metadata (no eval).
///
/// `pub(super)` so [`super::model::build_head`] reuses it for the `ctc_lo` head.
///
/// # Errors
/// - [`Error::RankMismatch`] if a present dense bias is not rank-1;
/// - [`Error::LengthMismatch`] if its sole axis is not `out_features`.
pub(super) fn pin_dense_linear_bias(
  linear: &MaybeQuantizedLinear,
  out_features: i32,
) -> Result<()> {
  if let Some(bias) = dense_linear_bias(linear) {
    pin_vector_len(
      bias,
      out_features,
      "sensevoice encoder: dense Linear bias must be rank-1 (out_features,)",
    )?;
  }
  Ok(())
}

/// Pin a 1-D vector [`Array`]'s length to `expected` at load. Used for the
/// LayerNorm affine vectors (`weight` / `bias`).
///
/// # Errors
/// - [`Error::RankMismatch`] if the array is not rank-1;
/// - [`Error::LengthMismatch`] if its sole axis is not `expected`.
fn pin_vector_len(vector: &Array, expected: i32, context: &'static str) -> Result<()> {
  let shape = vector.shape();
  if shape.len() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape,
    )));
  }
  let want = expected.max(0) as usize;
  if shape[0] != want {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      context, want, shape[0],
    )));
  }
  Ok(())
}

/// Pin the depthwise FSMN conv weight to its post-`sanitize` MLX layout
/// `(n_feat, kernel_size, 1)` at load.
///
/// The reference builds `Conv1d(n_feat, n_feat, kernel_size, groups=n_feat,
/// bias=False)` (`sensevoice.py:154-162`); torch stores its weight as
/// `(C_out, C_in/groups, K) = (n_feat, 1, kernel_size)`, which `sanitize`'s
/// `transpose(0, 2, 1)` turns into the MLX `(C_out, K, C_in/groups) =
/// (n_feat, kernel_size, 1)` the depthwise [`ops::conv::conv1d`] consumes
/// (`frontend.rs` sanitize). A stale conv weight would otherwise mis-group the
/// per-channel convolution (or fail deep in the conv) at the first forward.
///
/// # Errors
/// - [`Error::RankMismatch`] if the weight is not rank-3;
/// - [`Error::ShapePairMismatch`] if its shape is not `(n_feat, kernel_size, 1)`.
fn pin_fsmn_weight_shape(weight: &Array, n_feat: i32, kernel_size: i32) -> Result<()> {
  let context = "sensevoice encoder: fsmn_block.weight must be (output_size, kernel_size, 1)";
  let shape = weight.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape,
    )));
  }
  let expected = [n_feat.max(0) as usize, kernel_size.max(0) as usize, 1usize];
  if shape[..] != expected[..] {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      context,
      expected.to_vec(),
      shape,
    )));
  }
  Ok(())
}

// ──────────────────── SinusoidalPositionEncoder ────────────────────

/// The additive absolute sinusoidal position encoder
/// (`SinusoidalPositionEncoder`, `sensevoice.py:106-122`).
///
/// Stateless — the encoding is recomputed per call from the input shape. The
/// reference forward (`:107-122`):
/// `positions = arange(1, T+1)` (1-indexed); `half = D // 2`;
/// `incr = log(10000) / (half - 1)`;
/// `inv = exp(arange(half) * -incr)`;
/// `encoding = concat([sin(positions·inv), cos(positions·inv)], axis=-1)`;
/// returns `x + encoding`.
///
/// Two swift robustness clauses (`SenseVoiceModel.swift:8-31`) are ported on
/// top of the python source: the `concat([sin, cos])` width is `2 * (D // 2)`,
/// which for an odd `D` is `D - 1` — the encoding is then zero-padded to `D`
/// (and, defensively, truncated if it ever exceeds `D`); and the encoding is
/// cast back to `x.dtype()` before the add (the activation-dtype-preservation
/// discipline, so a `bf16` / `f16` checkpoint stays in its dtype). For the real
/// `input_dim = 560` (even) neither guard fires, but they make the block exact
/// for odd widths and half-precision activations.
#[derive(Debug, Default)]
pub struct SinusoidalPositionEncoder;

impl SinusoidalPositionEncoder {
  /// Add the sinusoidal position encoding to `x` of shape `(B, T, D)`.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `x` is not rank-3, or `T` / `D` exceed
  ///   `i32::MAX`, or `D < 2` (the `half - 1` increment divisor would be
  ///   non-positive);
  /// - propagates the arange / exp / trig / concat / pad / cast / add op
  ///   errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 3 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEncoder: input rank",
        "must be rank-3 (B, T, D)",
        format_smolstr!("{}", shape.len()),
      )));
    }
    let timesteps = shape[1];
    let input_dim = shape[2];
    if input_dim < 2 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEncoder: input_dim",
        "must be >= 2 (half_dim - 1 increment divisor)",
        format_smolstr!("{input_dim}"),
      )));
    }
    let t_i32 = i32::try_from(timesteps).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEncoder: T",
        "must fit in i32",
        format_smolstr!("{timesteps}"),
      ))
    })?;
    let d_i32 = i32::try_from(input_dim).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEncoder: D",
        "must fit in i32",
        format_smolstr!("{input_dim}"),
      ))
    })?;
    let half_dim = input_dim / 2;
    let half_i32 = (half_dim) as i32;

    // `positions = arange(1, T+1)` (1-indexed, `sensevoice.py:109`).
    let positions = Array::arange::<f32>(1.0, f64::from(t_i32) + 1.0, 1.0)?;
    // `incr = log(10000) / (half_dim - 1)`; `inv = exp(arange(half) * -incr)`
    // (`sensevoice.py:113-114`).
    let log_timescale_increment = (10000.0_f64).ln() / (f64::from(half_i32) - 1.0);
    let ramp = Array::arange::<f32>(0.0, f64::from(half_i32), 1.0)?;
    let neg_incr = Array::full::<f32>(&[0i32; 0], -log_timescale_increment as f32)?;
    let inv_timescales = ops::arithmetic::exp(&ramp.multiply(&neg_incr)?)?;

    // `scaled_time = positions[:, None] * inv[None, :]` -> (T, half).
    let positions_col = ops::shape::reshape(&positions, &[t_i32, 1])?;
    let inv_row = ops::shape::reshape(&inv_timescales, &[1, half_i32])?;
    let scaled_time = positions_col.multiply(&inv_row)?;

    // `encoding = concat([sin, cos], axis=-1)` -> (T, 2*half).
    let sin = ops::arithmetic::sin(&scaled_time)?;
    let cos = ops::arithmetic::cos(&scaled_time)?;
    let mut encoding = ops::shape::concatenate(&[&sin, &cos], 1)?;

    // Swift width guards (`SenseVoiceModel.swift:14-27`): the encoding width is
    // `2 * half_dim`; for an odd `D` that is `D - 1`. Zero-pad up to `D`
    // (defensively truncate if it ever exceeds `D`).
    let enc_width = 2 * half_i32;
    if enc_width > d_i32 {
      encoding = ops::indexing::slice(&encoding, &[0, 0], &[t_i32, d_i32], &[1, 1])?;
    } else if enc_width < d_i32 {
      let zero = Array::full::<f32>(&[0i32; 0], 0.0)?;
      encoding = ops::shape::pad(
        &encoding,
        &[1],
        &[0],
        &[d_i32 - enc_width],
        &zero,
        PAD_CONSTANT,
      )?;
    }

    // Add a leading batch axis -> (1, T, D); broadcasts over the batch in the
    // add. Cast back to `x.dtype()` (`SenseVoiceModel.swift:29`) so a half-
    // precision activation is not promoted to f32.
    let encoding = ops::shape::reshape(&encoding, &[1, t_i32, d_i32])?;
    let encoding = cast_to(&encoding, x.dtype()?)?;
    x.add(&encoding)
  }
}

/// Cast `a` to `dtype` (a no-op when already that dtype). Used to preserve the
/// activation dtype across the f32-built position encoding.
fn cast_to(a: &Array, dtype: Dtype) -> Result<Array> {
  if a.dtype()? == dtype {
    a.try_clone()
  } else {
    a.astype(dtype)
  }
}

// ──────────────────── PositionwiseFeedForward ────────────────────

/// The position-wise feed-forward: `w_2(relu(w_1(x)))`
/// (`PositionwiseFeedForward`, `sensevoice.py:125-132`). The activation is
/// **ReLU**, not GELU (`sensevoice.py:132`). Both projections are quantize-
/// aware.
#[derive(Debug)]
pub struct PositionwiseFeedForward {
  w_1: MaybeQuantizedLinear,
  w_2: MaybeQuantizedLinear,
}

impl PositionwiseFeedForward {
  /// Run `w_2(relu(w_1(x)))`.
  ///
  /// # Errors
  /// Propagates the linear / `maximum` op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.w_1.forward(x)?;
    let h = relu(&h)?;
    self.w_2.forward(&h)
  }

  /// Build from `<prefix>.w_1` / `<prefix>.w_2`, each quantize-aware via
  /// [`MaybeQuantizedLinear::from_weights`], pinning both projection shapes at
  /// load.
  ///
  /// `w_1` is `nn.Linear(idim, hidden_units)` and `w_2` is
  /// `nn.Linear(hidden_units, idim)` (`sensevoice.py:128-129`), so the
  /// `(out_features, in_features)` weight layouts are `(hidden_units, idim)` and
  /// `(idim, hidden_units)`. `idim` is the encoder hidden width (`output_size`)
  /// and `hidden_units` is `linear_units` (`sensevoice.py:262-265`); a stale
  /// shard whose FFN width disagrees would otherwise only fail deep in the
  /// matmul at the first forward.
  ///
  /// Each projection's optional dense bias is pinned to `(out_features,)`
  /// alongside its weight (through `pin_linear_shape` → [`pin_dense_linear_bias`]).
  ///
  /// `quant` is the parsed [`PerLayerQuantization`]; `w_1` / `w_2` each resolve
  /// their `(group_size, bits, mode)` PER PREFIX via [`resolve_layer_quant`]
  /// (the qwen3 per-prefix idiom), so a per-layer override / `Skip` is honored at
  /// the right granularity rather than a single collapsed global tuple.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] for an absent `w_1` / `w_2`;
  /// - [`Error::ShapePairMismatch`] if either projection's logical shape
  ///   disagrees with `(idim, hidden_units)`;
  /// - [`Error::RankMismatch`] / [`Error::LengthMismatch`] if a present `w_1` /
  ///   `w_2` dense bias is not rank-1 `(out_features,)`;
  /// - [`Error::InvariantViolation`] if a projection carries a `<prefix>.scales`
  ///   but the config resolved no scheme for that layer;
  /// - propagates the [`MaybeQuantizedLinear::from_weights`] / `logical_shape`
  ///   errors.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    idim: i32,
    hidden_units: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let w_1_prefix = format!("{prefix}.w_1");
    let w_1 = MaybeQuantizedLinear::from_weights(
      weights,
      &w_1_prefix,
      resolve_layer_quant(quant, &w_1_prefix),
    )?;
    pin_linear_shape(
      &w_1,
      hidden_units,
      idim,
      "sensevoice encoder: feed_forward.w_1 must be (linear_units, output_size)",
    )?;
    let w_2_prefix = format!("{prefix}.w_2");
    let w_2 = MaybeQuantizedLinear::from_weights(
      weights,
      &w_2_prefix,
      resolve_layer_quant(quant, &w_2_prefix),
    )?;
    pin_linear_shape(
      &w_2,
      idim,
      hidden_units,
      "sensevoice encoder: feed_forward.w_2 must be (output_size, linear_units)",
    )?;
    Ok(Self { w_1, w_2 })
  }
}

/// ReLU via `maximum(x, 0)` (`nn.relu`). Builds the `0` as a 0-D scalar so the
/// max broadcasts over any shape.
fn relu(x: &Array) -> Result<Array> {
  let zero = Array::full::<f32>(&[0i32; 0], 0.0)?;
  x.maximum(&zero)
}

// ──────────────────── MultiHeadedAttentionSANM ────────────────────

/// The SANM self-attention with an FSMN memory branch
/// (`MultiHeadedAttentionSANM`, `sensevoice.py:135-198`).
///
/// Differs from a vanilla MHA in two ways: a **fused** QKV projection
/// (`linear_q_k_v: in_feat -> n_feat * 3`, one matmul, split into q/k/v) and a
/// parallel **FSMN memory** branch — a depthwise `Conv1d(n_feat, n_feat, k,
/// groups=n_feat, bias=False)` over the value sequence, with an asymmetric
/// `(left, right)` pad and a `+ inputs` residual, whose output is added to the
/// projected attention output (`att_out + fsmn_memory`, NOT inside the softmax).
#[derive(Debug)]
pub struct MultiHeadedAttentionSANM {
  /// Fused `(in_feat) -> (n_feat * 3)` QKV projection.
  linear_q_k_v: MaybeQuantizedLinear,
  /// Output projection `(n_feat) -> (n_feat)`.
  linear_out: MaybeQuantizedLinear,
  /// Depthwise FSMN conv weight, MLX layout `(n_feat, kernel_size, 1)`
  /// (post-`sanitize`). Dense (mlx does not quantize conv).
  fsmn_weight: Array,
  /// Head count `h`.
  n_head: i32,
  /// Per-head width `d_k = n_feat / n_head`.
  d_k: i32,
  /// Total width `n_feat` (the merged QKV / attention-output width).
  n_feat: i32,
  /// FSMN left pad: `(k - 1) // 2 (+ sanm_shift if > 0)`
  /// (`sensevoice.py:164-167`).
  left_padding: i32,
  /// FSMN right pad: `k - 1 - left_padding` (`sensevoice.py:168`).
  right_padding: i32,
}

impl MultiHeadedAttentionSANM {
  /// The FSMN memory branch (`_forward_fsmn`, `sensevoice.py:170-177`): pad the
  /// time axis by `(left_padding, right_padding)`, run the depthwise conv, then
  /// add the un-padded `inputs` residual.
  ///
  /// `inputs` is `(B, T, n_feat)` (channels-last). The conv runs over the full
  /// `n_feat` channel dim with `groups = n_feat` (depthwise), so each channel
  /// is convolved by its own length-`k` kernel.
  fn forward_fsmn(&self, inputs: &Array) -> Result<Array> {
    // `mx.pad(inputs, ((0,0),(left,right),(0,0)))` — pad only the time axis
    // (axis 1) with zeros (`sensevoice.py:171-174`).
    let zero = Array::full::<f32>(&[0i32; 0], 0.0)?;
    let padded = ops::shape::pad(
      inputs,
      &[1],
      &[self.left_padding],
      &[self.right_padding],
      &zero,
      PAD_CONSTANT,
    )?;
    // Depthwise conv: input (B, L, C_in) channels-last, weight
    // (C_out, K, C_in/groups) = (n_feat, k, 1), groups = n_feat.
    let conv = ops::conv::conv1d(&padded, &self.fsmn_weight, 1, 0, 1, self.n_feat)?;
    // `x = x + inputs` (the FSMN memory shortcut, `sensevoice.py:176`).
    conv.add(inputs)
  }

  /// Run the SANM attention (`__call__`, `sensevoice.py:179-198`).
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `x` is not rank-3 or `B`/`T` exceed `i32::MAX`;
  /// - propagates the projection / split / FSMN / reshape / SDPA op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 3 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "MultiHeadedAttentionSANM: input rank",
        "must be rank-3 (B, T, in_feat)",
        format_smolstr!("{}", shape.len()),
      )));
    }
    let b = i32::try_from(shape[0]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "MultiHeadedAttentionSANM: B",
        "must fit in i32",
        format_smolstr!("{}", shape[0]),
      ))
    })?;
    let t = i32::try_from(shape[1]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "MultiHeadedAttentionSANM: T",
        "must fit in i32",
        format_smolstr!("{}", shape[1]),
      ))
    })?;

    // Fused QKV: `q_k_v = linear_q_k_v(x)`; `q, k, v = split(q_k_v, 3, -1)`
    // (`sensevoice.py:182-183`). `split_sections([n_feat, 2*n_feat])` gives the
    // three equal `n_feat`-wide parts.
    let qkv = self.linear_q_k_v.forward(x)?;
    let parts = ops::shape::split_sections(&qkv, &[self.n_feat, 2 * self.n_feat], -1)?;
    let q = &parts[0];
    let k = &parts[1];
    let v = &parts[2];

    // The FSMN memory runs on `v` BEFORE the head reshape, over the full
    // `n_feat` channel dim (`sensevoice.py:185`).
    let fsmn_memory = self.forward_fsmn(v)?;

    // Reshape q/k/v to (B, h, T, d_k) (`sensevoice.py:187-189`).
    let q_h = self.to_heads(q, b, t)?;
    let k_h = self.to_heads(k, b, t)?;
    let v_h = self.to_heads(v, b, t)?;

    // `scores = (q * d_k^-0.5) @ k.T`; `softmax`; `@ v_h`
    // (`sensevoice.py:191-193`). The shared SDPA computes exactly
    // `softmax(q @ k.T * scale) @ v` with `scale = d_k^-0.5` and no mask (the
    // encoder is fully bidirectional, `Mask::None`).
    let scale = (f64::from(self.d_k)).powf(-0.5) as f32;
    let att = scaled_dot_product_attention(&q_h, &k_h, &v_h, scale, Mask::None)?;

    // Merge heads back to (B, T, n_feat) (`sensevoice.py:195`), project, then
    // add the FSMN memory (`sensevoice.py:196-198`).
    let att = ops::shape::transpose_axes(&att, &[0, 2, 1, 3])?;
    let att = ops::shape::reshape(&att, &[b, t, self.n_feat])?;
    let att_out = self.linear_out.forward(&att)?;
    att_out.add(&fsmn_memory)
  }

  /// Reshape a `(B, T, n_feat)` projection to `(B, h, T, d_k)` heads.
  fn to_heads(&self, x: &Array, b: i32, t: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[b, t, self.n_head, self.d_k])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }

  /// Build from the layer config + the `<prefix>.{linear_q_k_v, linear_out,
  /// fsmn_block}` weights. `in_feat` is the projection input width (the layer's
  /// input width — `input_size` for the first block, `output_size` after).
  ///
  /// Every consumed weight is shape-pinned at load (the shape-pin discipline the
  /// head in [`super::model::build_head`] uses), so a stale / hostile shard
  /// reports a typed error here rather than splitting / reshaping with
  /// config-derived widths against a wrong tensor at the first forward:
  /// - `linear_q_k_v` is the fused `nn.Linear(in_feat, n_feat * 3)`
  ///   (`sensevoice.py:152`), whose `(out, in)` weight is `(3 * n_feat, in_feat)`
  ///   — `3 * n_feat` is computed with [`checked_mul`] so the fused width cannot
  ///   overflow `i32`;
  /// - `linear_out` is `nn.Linear(n_feat, n_feat)` (`sensevoice.py:151`), weight
  ///   `(n_feat, n_feat)`;
  /// - `fsmn_block.weight` is the depthwise `Conv1d(n_feat, n_feat, kernel_size,
  ///   groups=n_feat, bias=False)` in the post-`sanitize` MLX layout
  ///   `(n_feat, kernel_size, 1)` (`sensevoice.py:154-162`, `frontend.rs`
  ///   sanitize transpose).
  ///
  /// The optional dense `<prefix>.bias` of `linear_q_k_v` / `linear_out` is
  /// pinned to `(out_features,)` alongside its weight (through `pin_linear_shape`
  /// → [`pin_dense_linear_bias`]), so a stray `(1,)` / wrong-length bias that
  /// would broadcast a single wrong offset across every channel is rejected at
  /// load. The FSMN conv carries no bias (`bias=False`, `sensevoice.py:161`).
  ///
  /// `quant` is the parsed [`PerLayerQuantization`]; `linear_q_k_v` /
  /// `linear_out` each resolve their `(group_size, bits, mode)` PER PREFIX via
  /// [`resolve_layer_quant`] (the qwen3 per-prefix idiom), so a per-layer
  /// override / `Skip` is honored at the right granularity.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] for an absent weight;
  /// - [`Error::ArithmeticOverflow`] if `3 * n_feat` overflows `i32`;
  /// - [`Error::ShapePairMismatch`] if `linear_q_k_v` / `linear_out` disagrees
  ///   with its `(out, in)` layout;
  /// - [`Error::RankMismatch`] / [`Error::LengthMismatch`] if a present
  ///   `linear_q_k_v` / `linear_out` dense bias is not rank-1 `(out_features,)`;
  /// - [`Error::InvariantViolation`] if `linear_q_k_v` / `linear_out` carries a
  ///   `<prefix>.scales` but the config resolved no scheme for that layer;
  /// - [`Error::ShapePairMismatch`] / [`Error::RankMismatch`] if the FSMN conv
  ///   weight is not rank-3 `(n_feat, kernel_size, 1)`;
  /// - propagates the [`MaybeQuantizedLinear::from_weights`] / `logical_shape`
  ///   errors.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    enc: &EncoderConfig,
    in_feat: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let n_feat = enc.output_size();
    let n_head = enc.attention_heads();
    let kernel_size = enc.kernel_size();
    let sanm_shift = enc.sanm_shift();

    // `d_k = n_feat / n_head` (config.validate guarantees divisibility).
    let d_k = n_feat / n_head;

    // FSMN pad split (`sensevoice.py:164-168`).
    let mut left_padding = (kernel_size - 1) / 2;
    if sanm_shift > 0 {
      left_padding += sanm_shift;
    }
    let right_padding = kernel_size - 1 - left_padding;

    // The fused QKV output width `3 * n_feat` (`sensevoice.py:152`); checked so an
    // adversarial `output_size` near `i32::MAX / 3` cannot wrap the pinned width.
    let qkv_out = checked_mul(
      "sensevoice encoder: linear_q_k_v fused width 3 * output_size",
      "3",
      3,
      "encoder_conf.output_size",
      n_feat,
    )?;

    let qkv_prefix = format!("{prefix}.linear_q_k_v");
    let linear_q_k_v = MaybeQuantizedLinear::from_weights(
      weights,
      &qkv_prefix,
      resolve_layer_quant(quant, &qkv_prefix),
    )?;
    pin_linear_shape(
      &linear_q_k_v,
      qkv_out,
      in_feat,
      "sensevoice encoder: linear_q_k_v must be (3 * output_size, in_feat)",
    )?;
    let out_prefix = format!("{prefix}.linear_out");
    let linear_out = MaybeQuantizedLinear::from_weights(
      weights,
      &out_prefix,
      resolve_layer_quant(quant, &out_prefix),
    )?;
    pin_linear_shape(
      &linear_out,
      n_feat,
      n_feat,
      "sensevoice encoder: linear_out must be (output_size, output_size)",
    )?;
    // The depthwise FSMN conv is dense (mlx does not quantize conv); it carries
    // only a weight (bias=False, `sensevoice.py:161`). Pin its post-`sanitize`
    // MLX layout `(n_feat, kernel_size, 1)`.
    let fsmn_weight = take(weights, &format!("{prefix}.fsmn_block.weight"))?;
    pin_fsmn_weight_shape(&fsmn_weight, n_feat, kernel_size)?;

    Ok(Self {
      linear_q_k_v,
      linear_out,
      fsmn_weight,
      n_head,
      d_k,
      n_feat,
      left_padding,
      right_padding,
    })
  }
}

// ──────────────────── EncoderLayerSANM ────────────────────

/// One SANM encoder layer (`EncoderLayerSANM`, `sensevoice.py:201-237`),
/// pre-norm with the no-residual-on-width-change rule.
///
/// The subtlety (`sensevoice.py:227-230`): the residual around attention is
/// applied **iff** `in_size == size`. The first tower block changes width
/// (`560 -> 512`) and therefore drops that residual; every constant-width block
/// keeps it. `norm1` is sized to the INPUT width (`in_size`), `norm2` to the
/// output (`size`).
#[derive(Debug)]
pub struct EncoderLayerSANM {
  self_attn: MultiHeadedAttentionSANM,
  feed_forward: PositionwiseFeedForward,
  norm1: LayerNorm,
  norm2: LayerNorm,
  /// `true` when `in_size == size` (the residual-around-attention is kept).
  residual_attn: bool,
}

impl EncoderLayerSANM {
  /// Run the layer (`__call__`, `sensevoice.py:220-237`).
  ///
  /// # Errors
  /// Propagates the norm / attention / feed-forward / add op errors.
  pub fn forward(&self, x: &Array) -> Result<Array> {
    // residual = x; pre-norm; attn.
    let residual = x;
    let normed = self.norm1.forward(x)?;
    let attn_out = self.self_attn.forward(&normed)?;
    // `x = residual + attn_out` iff `in_size == size`, else `x = attn_out`.
    let x = if self.residual_attn {
      residual.add(&attn_out)?
    } else {
      attn_out
    };

    // residual = x; pre-norm; `x = residual + feed_forward(x)`.
    let normed = self.norm2.forward(&x)?;
    let ff = self.feed_forward.forward(&normed)?;
    x.add(&ff)
  }

  /// Build from `<prefix>.{self_attn, feed_forward, norm1, norm2}` with the
  /// given `in_size` (the layer input width). `norm1` is `(in_size,)`, `norm2`
  /// is `(output_size,)`.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    enc: &EncoderConfig,
    in_size: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let size = enc.output_size();
    let self_attn = MultiHeadedAttentionSANM::from_weights(
      weights,
      &format!("{prefix}.self_attn"),
      enc,
      in_size,
      quant,
    )?;
    // The FFN operates at the layer output width: `w_1(output_size ->
    // linear_units)`, `w_2(linear_units -> output_size)` (`sensevoice.py:262-265`).
    let feed_forward = PositionwiseFeedForward::from_weights(
      weights,
      &format!("{prefix}.feed_forward"),
      size,
      enc.linear_units(),
      quant,
    )?;
    // `norm1` is sized to the INPUT width (`in_size`), `norm2` to the output
    // (`size`) (`sensevoice.py:214-215`).
    let norm1 = build_layer_norm(weights, &format!("{prefix}.norm1"), in_size)?;
    let norm2 = build_layer_norm(weights, &format!("{prefix}.norm2"), size)?;
    Ok(Self {
      self_attn,
      feed_forward,
      norm1,
      norm2,
      residual_attn: in_size == size,
    })
  }
}

// ──────────────────── Encoder (the tower) ────────────────────

/// The full SenseVoice SANM encoder tower (`SenseVoiceEncoder`,
/// `sensevoice.py:240-338`).
///
/// `encoders0` (one width-changing `input_size -> output_size` block) +
/// `encoders` (`num_blocks - 1` constant-width blocks) + `after_norm` +
/// `tp_encoders` (`tp_blocks`) + `tp_norm`. The forward scales the input by
/// `sqrt(output_size)` and adds the sinusoidal PE before the stacks
/// (`sensevoice.py:323-324`).
#[derive(Debug)]
pub struct Encoder {
  embed: SinusoidalPositionEncoder,
  encoders0: Vec<EncoderLayerSANM>,
  encoders: Vec<EncoderLayerSANM>,
  after_norm: LayerNorm,
  tp_encoders: Vec<EncoderLayerSANM>,
  tp_norm: LayerNorm,
  /// `output_size`, for the `sqrt(output_size)` input scale.
  output_size: i32,
}

impl Encoder {
  /// Run the encoder over `xs` of shape `(B, T, input_size)`, producing
  /// `(B, T, output_size)` (`__call__`, `sensevoice.py:322-338`).
  ///
  /// # Errors
  /// Propagates the scale / PE / per-layer / norm op errors.
  pub fn forward(&self, xs: &Array) -> Result<Array> {
    // `xs = xs * sqrt(output_size)` (`sensevoice.py:323`). Build the scalar at
    // the activation dtype so a half-precision input is not promoted.
    let scale_val = (f64::from(self.output_size)).sqrt() as f32;
    let scale = cast_to(&Array::full::<f32>(&[0i32; 0], scale_val)?, xs.dtype()?)?;
    let mut h = xs.multiply(&scale)?;
    // `xs = embed(xs)` — additive sinusoidal PE (`sensevoice.py:324`).
    h = self.embed.forward(&h)?;

    for layer in &self.encoders0 {
      h = layer.forward(&h)?;
    }
    for layer in &self.encoders {
      h = layer.forward(&h)?;
    }
    h = self.after_norm.forward(&h)?;
    for layer in &self.tp_encoders {
      h = layer.forward(&h)?;
    }
    self.tp_norm.forward(&h)
  }

  /// The encoder hidden width (`output_size`).
  #[inline(always)]
  pub const fn output_size(&self) -> i32 {
    self.output_size
  }

  /// Build the tower from a checkpoint weight map under the `encoder.` prefix
  /// (`encoder.encoders0.*`, `encoder.encoders.*`, `encoder.after_norm`,
  /// `encoder.tp_encoders.*`, `encoder.tp_norm`).
  ///
  /// `input_size` is the LFR feature width (`560`) fed to the width-changing
  /// first block; every later block is `output_size`-wide. `quant` is the parsed
  /// [`PerLayerQuantization`] (a dense checkpoint passes `None`); each
  /// quantize-aware linear resolves its own `(group_size, bits, mode)` PER PREFIX
  /// via [`PerLayerQuantization::quantization_for`] in the per-layer builders (the
  /// qwen3 idiom), so a per-layer parameter override / `Skip` is honored at the
  /// right granularity.
  ///
  /// Every consumed weight is shape-pinned at load (the per-block qkv / out /
  /// FFN / FSMN-conv / LayerNorm affines, AND each dense linear's optional bias,
  /// via the per-layer builders), with checked arithmetic for the derived
  /// fused-QKV width `3 * output_size`, so a stale / hostile shard reports a
  /// typed shape error here rather than at the first forward.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `num_blocks` / `tp_blocks` is negative;
  /// - [`Error::AllocFailure`] if the `encoders` / `tp_encoders` block `Vec`
  ///   reservation fails for an over-large count (the fallible-reserve guard;
  ///   `Config::validate` also caps the count via `MAX_CONFIG_CARDINALITY`);
  /// - [`Error::MissingKey`] for an absent weight;
  /// - [`Error::ArithmeticOverflow`] if a derived width (`3 * output_size`)
  ///   overflows `i32`;
  /// - [`Error::ShapePairMismatch`] / [`Error::LengthMismatch`] /
  ///   [`Error::RankMismatch`] if any consumed weight's shape disagrees with the
  ///   config-derived layout;
  /// - propagates the per-layer build errors.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    input_size: i32,
    enc: &EncoderConfig,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let output_size = enc.output_size();
    let num_blocks = enc.num_blocks();
    let tp_blocks = enc.tp_blocks();
    if num_blocks < 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "encoder: num_blocks",
        "must be >= 1 (encoders0 holds the first block)",
        format_smolstr!("{num_blocks}"),
      )));
    }
    if tp_blocks < 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "encoder: tp_blocks",
        "must be >= 0",
        format_smolstr!("{tp_blocks}"),
      )));
    }

    // `encoders0`: ONE width-changing block (`input_size -> output_size`,
    // `sensevoice.py:250-270`).
    let block0 =
      EncoderLayerSANM::from_weights(weights, "encoder.encoders0.0", enc, input_size, quant)?;
    let encoders0 = vec![block0];

    // `encoders`: `num_blocks - 1` constant-width blocks
    // (`sensevoice.py:272-293`). Reserve fallibly so an over-large count
    // (bounded by `MAX_CONFIG_CARDINALITY` in `Config::validate`, but defended
    // here too) surfaces a typed `AllocFailure`, never an allocator abort.
    let mut encoders = Vec::new();
    reserve_or_error(
      &mut encoders,
      "sensevoice encoder.encoders blocks",
      (num_blocks - 1).max(0) as usize,
    )?;
    for i in 0..(num_blocks - 1) {
      let layer = EncoderLayerSANM::from_weights(
        weights,
        &format!("encoder.encoders.{i}"),
        enc,
        output_size,
        quant,
      )?;
      encoders.push(layer);
    }

    let after_norm = build_layer_norm(weights, "encoder.after_norm", output_size)?;

    // `tp_encoders`: `tp_blocks` constant-width blocks (`sensevoice.py:297-318`).
    // Reserve fallibly (same discipline as `encoders` above).
    let mut tp_encoders = Vec::new();
    reserve_or_error(
      &mut tp_encoders,
      "sensevoice encoder.tp_encoders blocks",
      tp_blocks.max(0) as usize,
    )?;
    for i in 0..tp_blocks {
      let layer = EncoderLayerSANM::from_weights(
        weights,
        &format!("encoder.tp_encoders.{i}"),
        enc,
        output_size,
        quant,
      )?;
      tp_encoders.push(layer);
    }

    let tp_norm = build_layer_norm(weights, "encoder.tp_norm", output_size)?;

    Ok(Self {
      embed: SinusoidalPositionEncoder,
      encoders0,
      encoders,
      after_norm,
      tp_encoders,
      tp_norm,
      output_size,
    })
  }
}

#[cfg(test)]
mod tests;
