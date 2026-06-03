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
  error::{Error, MissingKeyPayload, OutOfRangePayload, Result},
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::LayerNorm,
  },
  nn::MaybeQuantizedLinear,
  ops,
};

use super::config::EncoderConfig;

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
/// (`nn.LayerNorm` always carries both). `eps` is fixed at [`LAYERNORM_EPS`].
fn build_layer_norm(weights: &mut HashMap<String, Array>, prefix: &str) -> Result<LayerNorm> {
  let weight = take(weights, &format!("{prefix}.weight"))?;
  let bias = take(weights, &format!("{prefix}.bias"))?;
  Ok(LayerNorm::new(Some(weight), Some(bias), LAYERNORM_EPS))
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
  /// [`MaybeQuantizedLinear::from_weights`].
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let w_1 = MaybeQuantizedLinear::from_weights(weights, &format!("{prefix}.w_1"), quant)?;
    let w_2 = MaybeQuantizedLinear::from_weights(weights, &format!("{prefix}.w_2"), quant)?;
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
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    enc: &EncoderConfig,
    in_feat: i32,
    quant: Option<(i32, i32, &str)>,
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

    let linear_q_k_v =
      MaybeQuantizedLinear::from_weights(weights, &format!("{prefix}.linear_q_k_v"), quant)?;
    let linear_out =
      MaybeQuantizedLinear::from_weights(weights, &format!("{prefix}.linear_out"), quant)?;
    // The depthwise FSMN conv is dense (mlx does not quantize conv); it carries
    // only a weight (bias=False, `sensevoice.py:161`).
    let fsmn_weight = take(weights, &format!("{prefix}.fsmn_block.weight"))?;

    // `in_feat` is consumed only as documentation of the fused-projection input
    // width; the actual shape contract is pinned by the loader. Tie it to a
    // no-op so the parameter is part of the seam without an unused warning.
    let _ = in_feat;

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
    quant: Option<(i32, i32, &str)>,
  ) -> Result<Self> {
    let size = enc.output_size();
    let self_attn = MultiHeadedAttentionSANM::from_weights(
      weights,
      &format!("{prefix}.self_attn"),
      enc,
      in_size,
      quant,
    )?;
    let feed_forward =
      PositionwiseFeedForward::from_weights(weights, &format!("{prefix}.feed_forward"), quant)?;
    let norm1 = build_layer_norm(weights, &format!("{prefix}.norm1"))?;
    let norm2 = build_layer_norm(weights, &format!("{prefix}.norm2"))?;
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
  /// first block; every later block is `output_size`-wide. `quant` carries the
  /// resolved `(group_size, bits, mode)` applied to every quantize-aware
  /// linear (a dense checkpoint passes `None`); quant resolution is the loader's
  /// concern.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `num_blocks` / `tp_blocks` is negative;
  /// - [`Error::MissingKey`] for an absent weight;
  /// - propagates the per-layer build errors.
  pub fn from_weights(
    weights: &mut HashMap<String, Array>,
    input_size: i32,
    enc: &EncoderConfig,
    quant: Option<(i32, i32, &str)>,
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
    // (`sensevoice.py:272-293`).
    let mut encoders = Vec::with_capacity((num_blocks - 1).max(0) as usize);
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

    let after_norm = build_layer_norm(weights, "encoder.after_norm")?;

    // `tp_encoders`: `tp_blocks` constant-width blocks (`sensevoice.py:297-318`).
    let mut tp_encoders = Vec::with_capacity(tp_blocks.max(0) as usize);
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

    let tp_norm = build_layer_norm(weights, "encoder.tp_norm")?;

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
