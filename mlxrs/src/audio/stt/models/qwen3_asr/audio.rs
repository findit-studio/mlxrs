//! Qwen3-ASR audio encoder — the Conv2d stem + transformer encoder.
//!
//! Port of mlx-audio's `qwen3_asr.AudioEncoder` (and its `AudioAttention`,
//! `AudioEncoderLayer`, `SinusoidalPositionEmbedding`). The encoder maps a
//! log-mel spectrogram `[batch, num_mel_bins, time]` to audio embeddings
//! `[batch, time', output_dim]`:
//!
//! 1. **Conv2d stem** — the mel input is given a singleton channel axis
//!    (`[B, n_mels, T] -> [B, n_mels, T, 1]`, MLX channels-last
//!    `(N, H=freq, W=time, C_in)`), then three `Conv2d(kernel = 3, stride = 2,
//!    padding = 1)` layers each followed by GELU. Every conv halves both the
//!    freq and time axes (`out = (in + 1) / 2`), giving an ~8x downsample in
//!    each. The `(b, freq', time', hidden)` result is transposed to
//!    `(b, time', hidden, freq')` and flattened to
//!    `(b, time', hidden * freq')`, then a bias-free `conv_out` linear projects
//!    it to `d_model`.
//! 2. **Sinusoidal position embedding** — a precomputed
//!    `(max_source_positions, d_model)` table (`sin`/`cos` halves) is sliced to
//!    the post-CNN length and added.
//! 3. **Transformer encoder** — `encoder_layers` pre-norm blocks of
//!    multi-head self-attention (`head_dim**-0.5` passed as the SDPA `scale`) +
//!    a GELU feed-forward (`fc1`/`fc2`), with an optional additive attention
//!    mask. In `float16` the post-FFN-residual hidden state is clamped to
//!    `finfo(float16).max - 1000` (symmetric) so a valid half-precision
//!    checkpoint saturates rather than overflowing to `inf`/`NaN`.
//! 4. **Output head** — `ln_post`, then `gelu(proj1(.))`, then `proj2(.)` to
//!    `output_dim`.
//!
//! ### Block / chunked attention (windowed inference)
//!
//! The reference's full inference path splits a variable-length utterance into
//! `n_window * 2`-mel-frame conv chunks, runs the stem per chunk, then groups
//! the post-CNN frames into windows `n_window_infer / (n_window * 2)` wide and
//! builds a **block-diagonal additive mask** (`_create_block_attention_mask`)
//! so each window attends only within itself.
//! [`AudioEncoder::forward_single_window`] ports this for one utterance (the
//! aligner's audio path): the valid mel frames are split into conv chunks, the
//! stem runs per chunk, the sinusoidal positions reset per chunk, the valid
//! post-CNN frames are concatenated, and the transformer runs under the
//! block-diagonal mask — handling arbitrary-length audio. A short utterance
//! that fits one conv chunk collapses to the degenerate single-window case
//! (full attention). The plain [`AudioEncoder::forward`] runs full (unmasked)
//! attention over a precomputed mel and [`AudioEncoder::forward_with_mask`]
//! takes a caller-supplied additive mask; both are the building blocks the
//! single-window path drives.
//!
//! The mel **frontend itself is not re-ported here** — mlxrs already ships the
//! mel-spectrogram pipeline (see [`crate::audio::dsp`] and
//! [`crate::audio::features`]); the encoder consumes a precomputed
//! `input_features` mel tensor, exactly as the reference `AudioEncoder.__call__`
//! does.

use std::collections::HashMap;

use smol_str::format_smolstr;

use super::config::AudioEncoderConfig;
use crate::{
  Dtype,
  array::Array,
  error::{
    Error, LengthMismatchPayload, MissingKeyPayload, OutOfRangePayload, RankMismatchPayload,
    Result, ShapePairMismatchPayload,
  },
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::LayerNorm,
  },
  model_validation::reserve_or_error,
  ops::{
    conv::conv2d,
    linalg_basic::{addmm, matmul},
    shape::{concatenate, expand_dims_axes, pad, reshape, swapaxes, transpose_axes},
  },
};

// ───────────────────────────── linear helper ─────────────────────────────

/// `y = x @ wᵀ (+ bias)` — an `nn.Linear` forward. The reference's audio
/// encoder uses biased linears (`q/k/v/out_proj`, `fc1`/`fc2`, `proj1`/`proj2`)
/// and one bias-free linear (`conv_out`). HF stores a `Linear` weight as
/// `(out, in)`, so the matmul is `x @ wᵀ`. With a bias this is the fused
/// `addmm`; without, a plain `matmul`.
fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
  let wt = swapaxes(weight, -1, -2)?;
  match bias {
    Some(b) => addmm(b, x, &wt, 1.0, 1.0),
    None => matmul(x, &wt),
  }
}

/// Build a rank-0 `f32` scalar [`Array`] for broadcasting against a lazy
/// operand (rank-0 NumPy-broadcasts against any rank without lifting it).
fn scalar_f32(value: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], value)
}

/// The symmetric `float16` saturation bound `finfo(float16).max - 1000`.
///
/// The reference Qwen3-ASR audio encoder layer clamps a `torch.float16` hidden
/// state into `[-(finfo.max - 1000), finfo.max - 1000]` after the feed-forward
/// residual so a valid half-precision checkpoint saturates instead of
/// overflowing to `inf`/`NaN`. `finfo(float16).max` is `65504`, giving `64504`.
pub(super) const F16_CLAMP_BOUND: f32 = half::f16::MAX.to_f32_const() - 1000.0;

/// Saturate `x` into `[-F16_CLAMP_BOUND, F16_CLAMP_BOUND]` **only** when its
/// dtype is `float16`, leaving any other dtype untouched — the reference's
/// `if hidden_states.dtype == torch.float16: clamp(...)` guard.
///
/// The bound scalars are materialized in `float16` so the clamp does not
/// promote the `f16` activation to `f32` (`promote_types(f16, f32) == f32`),
/// preserving the half-precision activation dtype.
pub(super) fn clamp_if_f16(x: Array) -> Result<Array> {
  if x.dtype()? != Dtype::F16 {
    return Ok(x);
  }
  let hi = scalar_f32(F16_CLAMP_BOUND)?.astype(Dtype::F16)?;
  let lo = scalar_f32(-F16_CLAMP_BOUND)?.astype(Dtype::F16)?;
  x.clip(&lo, &hi)
}

// ───────────────────────── sinusoidal position embedding ─────────────────────────

/// Sinusoidal position embeddings (`SinusoidalPositionEmbedding`).
///
/// The `(max_source_positions, d_model)` table is precomputed once at
/// construction: `inv_timescales = exp(-(ln(max_timescale) / (C/2 - 1)) *
/// arange(C/2))`, `scaled_time = arange(length)[:, None] *
/// inv_timescales[None, :]`, then `concat([sin(scaled_time),
/// cos(scaled_time)], axis = 1)`. [`forward`](Self::forward) slices it to the
/// requested sequence length.
#[derive(Debug)]
pub(super) struct SinusoidalPositionEmbedding {
  /// `(max_source_positions, d_model)` precomputed embedding table.
  table: Array,
}

impl SinusoidalPositionEmbedding {
  /// `max_timescale` matching the reference (`10000.0`).
  const MAX_TIMESCALE: f64 = 10000.0;

  /// Precompute the `(length, channels)` table. `channels` must be even and
  /// `>= 2` (so `channels / 2 - 1 >= 0`); the config gate
  /// ([`AudioEncoderConfig::validate`]) guarantees both, but it is re-checked
  /// here as a typed error rather than a divide-by-zero.
  fn new(length: i32, channels: i32) -> Result<Self> {
    if channels < 2 || channels % 2 != 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEmbedding: channels",
        "must be even and >= 2",
        format_smolstr!("{channels}"),
      )));
    }
    let half = channels / 2;
    // log_timescale_increment = ln(max_timescale) / (half - 1). `half == 1`
    // (channels == 2) would divide by zero in the reference; guard it.
    if half < 2 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEmbedding: channels / 2",
        "must be >= 2 (channels >= 4)",
        format_smolstr!("{half}"),
      )));
    }
    let log_increment = Self::MAX_TIMESCALE.ln() / f64::from(half - 1);
    // inv_timescales = exp(-log_increment * arange(half))  → (half,)
    let idx = Array::arange::<f32>(0.0, f64::from(half), 1.0)?;
    let neg_inc = scalar_f32(-(log_increment as f32))?;
    let inv_timescales = idx.multiply(&neg_inc)?.exp()?;
    // positions = arange(length)[:, None]  → (length, 1)
    let positions = Array::arange::<f32>(0.0, f64::from(length), 1.0)?;
    let positions = expand_dims_axes(&positions, &[1])?;
    // inv_timescales[None, :]  → (1, half)
    let inv_row = expand_dims_axes(&inv_timescales, &[0])?;
    // scaled_time = positions * inv_timescales  → (length, half) (broadcast)
    let scaled_time = positions.multiply(&inv_row)?;
    let sin = scaled_time.sin()?;
    let cos = scaled_time.cos()?;
    let table = concatenate(&[&sin, &cos], 1)?;
    Ok(Self { table })
  }

  /// Test-only: build the table and return its first `seqlen` rows as a flat
  /// row-major `Vec<f32>` (`seqlen * channels`). Lets the oracle test compare
  /// the implementation's sinusoidal values against an independent closed form.
  #[cfg(test)]
  pub(super) fn eval_rows(length: i32, channels: i32, seqlen: i32) -> Result<Vec<f32>> {
    let emb = Self::new(length, channels)?;
    let mut rows = emb.forward(seqlen)?;
    rows.to_vec::<f32>()
  }

  /// The first `seqlen` rows of the precomputed table → `(seqlen, d_model)`.
  ///
  /// `seqlen` must not exceed the table's row count (`max_source_positions`).
  /// MLX `slice` **clamps** an out-of-range stop rather than erroring, so an
  /// unchecked `seqlen > rows` would silently yield a `(rows, d_model)` slice
  /// that then broadcasts position `0..rows` across a longer sequence — reusing
  /// positions through the public forward. Reject it with a typed
  /// [`Error::OutOfRange`] before slicing so the caller cannot get a silently
  /// truncated positional table.
  fn forward(&self, seqlen: i32) -> Result<Array> {
    let rows = i32::try_from(self.table.shape()[0]).unwrap_or(i32::MAX);
    if seqlen > rows {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SinusoidalPositionEmbedding::forward: sequence length vs max_source_positions",
        "post-CNN sequence length must not exceed the positional table row count (max_source_positions)",
        format_smolstr!("seqlen={seqlen}, max_source_positions={rows}"),
      )));
    }
    let d = i32::try_from(self.table.shape()[1]).unwrap_or(i32::MAX);
    let start = [0i32, 0];
    let stop = [seqlen, d];
    let strides = [1i32, 1];
    crate::ops::indexing::slice(&self.table, &start, &stop, &strides)
  }
}

// ───────────────────────────── attention ─────────────────────────────

/// Multi-head self-attention (`AudioAttention`).
///
/// `q/k/v/out_proj` are biased `Linear(d_model, d_model)`. The
/// `head_dim**-0.5` factor is passed to SDPA as its `scale` argument (rather
/// than pre-multiplied into the query), matching
/// `scaled_dot_product_attention(..., scale = self.scaling)` and the
/// `EmbeddingGemma` / SigLIP attention pattern. An optional additive mask (the
/// block-attention mask) is forwarded to SDPA.
#[derive(Debug)]
struct AudioAttention {
  q_weight: Array,
  q_bias: Array,
  k_weight: Array,
  k_bias: Array,
  v_weight: Array,
  v_bias: Array,
  out_weight: Array,
  out_bias: Array,
  num_heads: i32,
  head_dim: i32,
  embed_dim: i32,
  /// `head_dim**-0.5`, passed to SDPA as the softmax scale.
  scaling: f32,
}

impl AudioAttention {
  fn forward(&self, hidden_states: &Array, mask: Mask<'_>) -> Result<Array> {
    let shape = hidden_states.shape();
    let bsz = dim_i32(&shape, 0, "AudioAttention: batch")?;
    let seq_len = dim_i32(&shape, 1, "AudioAttention: seq")?;

    let q = linear(hidden_states, &self.q_weight, Some(&self.q_bias))?;
    let k = linear(hidden_states, &self.k_weight, Some(&self.k_bias))?;
    let v = linear(hidden_states, &self.v_weight, Some(&self.v_bias))?;

    // (B, L, C) → (B, n_heads, L, head_dim): reshape then transpose(0,2,1,3).
    let q = self.split_heads(&q, bsz, seq_len)?;
    let k = self.split_heads(&k, bsz, seq_len)?;
    let v = self.split_heads(&v, bsz, seq_len)?;

    // head_dim**-0.5 is the SDPA scale (not a query pre-multiply), so the
    // softmax-precision scaling happens inside the fused kernel and no dtype
    // promotion of the query can occur.
    let attn = scaled_dot_product_attention(&q, &k, &v, self.scaling, mask)?;

    // (B, n_heads, L, head_dim) → (B, L, C).
    let attn = transpose_axes(&attn, &[0, 2, 1, 3])?;
    let attn = reshape(&attn, &[bsz, seq_len, self.embed_dim])?;
    linear(&attn, &self.out_weight, Some(&self.out_bias))
  }

  /// `(B, L, C) → (B, n_heads, L, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

// ───────────────────────────── encoder layer ─────────────────────────────

/// A single pre-norm transformer encoder layer (`AudioEncoderLayer`):
/// `h = h + self_attn(self_attn_layer_norm(h)); h = h +
/// fc2(gelu(fc1(final_layer_norm(h))))`, with a `float16` saturation clamp on
/// the post-FFN-residual hidden state (see [`clamp_if_f16`]).
#[derive(Debug)]
struct AudioEncoderLayer {
  self_attn: AudioAttention,
  self_attn_layer_norm: LayerNorm,
  fc1_weight: Array,
  fc1_bias: Array,
  fc2_weight: Array,
  fc2_bias: Array,
  final_layer_norm: LayerNorm,
}

impl AudioEncoderLayer {
  fn forward(&self, hidden_states: &Array, mask: Mask<'_>) -> Result<Array> {
    let residual = hidden_states;
    let h = self.self_attn_layer_norm.forward(hidden_states)?;
    let h = self.self_attn.forward(&h, mask)?;
    let h = residual.add(&h)?;

    let residual = &h;
    let normed = self.final_layer_norm.forward(&h)?;
    let inter = linear(&normed, &self.fc1_weight, Some(&self.fc1_bias))?;
    let inter = crate::lm::nn::activations::gelu(&inter)?;
    let ff = linear(&inter, &self.fc2_weight, Some(&self.fc2_bias))?;
    // Clamp the post-FFN-residual hidden state in float16 (the reference's
    // `if dtype == float16: clamp(finfo.max - 1000)`); a no-op at other dtypes.
    // The reference does NOT clamp after the attention residual, only here.
    clamp_if_f16(residual.add(&ff)?)
  }
}

// ───────────────────────────── conv2d stem ─────────────────────────────

/// A single `Conv2d(kernel = 3, stride = 2, padding = 1)` of the stem, with a
/// channels-last MLX weight `(out, kH, kW, in)` (post-sanitize) and a `(out,)`
/// bias.
#[derive(Debug)]
struct Conv2dLayer {
  /// MLX channels-last conv weight `(out, kH, kW, in)`.
  weight: Array,
  /// `(out,)` bias added after the conv (channels-last, last axis).
  bias: Array,
}

impl Conv2dLayer {
  /// `gelu(conv2d(x) + bias)`. `x` is channels-last `(N, H, W, C_in)`; output
  /// is `(N, H', W', C_out)` with `H' = (H + 1) / 2`, `W' = (W + 1) / 2`
  /// (kernel 3, stride 2, padding 1).
  fn forward(&self, x: &Array) -> Result<Array> {
    let h = conv2d(x, &self.weight, (2, 2), (1, 1), (1, 1), 1)?;
    let h = h.add(&self.bias)?;
    crate::lm::nn::activations::gelu(&h)
  }
}

// ───────────────────────────── encoder ─────────────────────────────

/// The Qwen3-ASR audio encoder.
///
/// See the module documentation for the architecture and the block-attention
/// notes. Build with [`AudioEncoder::from_weights`]; run with
/// [`AudioEncoder::forward`] (full attention) or
/// [`AudioEncoder::forward_with_mask`] (caller-supplied block mask).
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr")))]
#[derive(Debug)]
pub struct AudioEncoder {
  config: AudioEncoderConfig,
  conv2d1: Conv2dLayer,
  conv2d2: Conv2dLayer,
  conv2d3: Conv2dLayer,
  /// `conv_out` bias-free linear `(d_model, downsample_hidden_size *
  /// freq_after_conv)`.
  conv_out_weight: Array,
  positional_embedding: SinusoidalPositionEmbedding,
  layers: Vec<AudioEncoderLayer>,
  ln_post: LayerNorm,
  proj1_weight: Array,
  proj1_bias: Array,
  proj2_weight: Array,
  proj2_bias: Array,
}

/// The reference's block-attention "masked-out" additive fill (`-1e9`, the
/// value `_create_block_attention_mask` writes off the window blocks). Not
/// `-inf`, matching mlx-audio so the saturated softmax probabilities agree.
const MASK_NEG: f32 = -1e9;

#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr")))]
impl AudioEncoder {
  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &AudioEncoderConfig {
    &self.config
  }

  /// Run the Conv2d stem + `conv_out` projection: mel `[B, n_mels, T]` →
  /// `[B, T', d_model]` (channels-last conv, then flatten `(channel, freq)`).
  ///
  /// This is the shared front half of [`forward`](Self::forward) /
  /// [`forward_with_mask`](Self::forward_with_mask), separated so the
  /// post-CNN length (`T'`) is observable for shape validation. The input must
  /// be rank-3 `[batch, num_mel_bins, time]`; its mel-bin axis must equal the
  /// configured `num_mel_bins`.
  pub fn encode_features(&self, input_features: &Array) -> Result<Array> {
    let shape = input_features.shape();
    if shape.len() != 3 {
      let rank = shape.len() as u32;
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "AudioEncoder::encode_features: input_features must be rank-3 [batch, num_mel_bins, time]",
        rank,
        shape,
      )));
    }
    let n_mels = dim_i32(&shape, 1, "AudioEncoder: num_mel_bins")?;
    if n_mels != self.config.num_mel_bins {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "AudioEncoder::encode_features: input mel-bin axis vs config.num_mel_bins",
        vec![self.config.num_mel_bins.max(0) as usize],
        vec![n_mels.max(0) as usize],
      )));
    }

    // [B, n_mels, T] → [B, n_mels, T, 1] (channels-last (N, H=freq, W=time, C_in=1)).
    let x = expand_dims_axes(input_features, &[3])?;
    let x = self.conv2d1.forward(&x)?;
    let x = self.conv2d2.forward(&x)?;
    let x = self.conv2d3.forward(&x)?;

    // x is (b, f', t', c). Transpose to (b, t', c, f') then flatten the last
    // two axes to (b, t', c * f') — the reference's `transpose(0,2,3,1)` +
    // `reshape(b, t, c*f)`.
    let xs = x.shape();
    let b = dim_i32(&xs, 0, "AudioEncoder: conv out batch")?;
    let f = dim_i32(&xs, 1, "AudioEncoder: conv out freq")?;
    let t = dim_i32(&xs, 2, "AudioEncoder: conv out time")?;
    let c = dim_i32(&xs, 3, "AudioEncoder: conv out channels")?;
    let x = transpose_axes(&x, &[0, 2, 3, 1])?;
    let cf = c.checked_mul(f).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder: conv out channels * freq",
        "overflows i32",
        format_smolstr!("{c} * {f}"),
      ))
    })?;
    let x = reshape(&x, &[b, t, cf])?;
    // conv_out: (b, t', c*f') → (b, t', d_model), bias-free.
    linear(&x, &self.conv_out_weight, None)
  }

  /// Full forward over a precomputed mel spectrogram with **full (unmasked)
  /// self-attention**: `[batch, num_mel_bins, time]` → audio embeddings
  /// `[batch, time', output_dim]`.
  ///
  /// `time'` is the three-fold stride-2 downsample of `time`
  /// ([`AudioEncoderConfig::time_after_conv`]). This is the correct behavior
  /// for a single window / short utterance; for the ragged multi-window
  /// inference path supply a block mask via
  /// [`forward_with_mask`](Self::forward_with_mask). No implicit eval — the
  /// returned [`Array`] is lazy.
  pub fn forward(&self, input_features: &Array) -> Result<Array> {
    self.forward_inner(input_features, Mask::None)
  }

  /// Full forward over a precomputed mel spectrogram with a caller-supplied
  /// additive attention `mask` (e.g. the ragged block-diagonal mask the
  /// reference builds for windowed inference). The mask must broadcast to the
  /// SDPA score shape `(batch, num_heads, time', time')`. See the
  /// module documentation block-attention notes.
  ///
  /// The **caller owns the mask shape**: it is forwarded to SDPA as-is (faithful
  /// to mlx-audio, which broadcasts the caller mask into the score tensor), so a
  /// mask that does not broadcast to `(batch, num_heads, time', time')` is a
  /// caller-contract violation rather than a guarded error here (issue #326).
  pub fn forward_with_mask(&self, input_features: &Array, mask: &Array) -> Result<Array> {
    self.forward_inner(input_features, Mask::Array(mask))
  }

  /// The reference's `_get_feat_extract_output_lengths` closed form for
  /// `mel_len` mel frames — a faithful port of that mlx-audio helper.
  ///
  /// The reference computes, with `leave = mel_len % 100`:
  /// `feat = floor((leave - 1) / 2) + 1`,
  /// `output = floor((floor((feat - 1) / 2) + 1 - 1) / 2) + 1 + (mel_len // 100)
  /// * 13`. This `% 100 … * 13` form is an **approximation** of the conv stack's
  /// output length that is exact only when the chunk size is the standard
  /// 100-frame conv chunk (`n_window = 50`): it equals the true three-fold
  /// stride-2 conv recurrence [`AudioEncoderConfig::time_after_conv`] for
  /// `mel_len <= 100` and is additive across 100-frame boundaries, but for any
  /// other chunk size it diverges from the actual post-conv row count (e.g.
  /// `feature_output_length(200) = 26` while `time_after_conv(200) = 25`).
  ///
  /// The audio token count and every windowed row-count therefore derive from
  /// the **exact** recurrence ([`windowed_output_length`](Self::windowed_output_length),
  /// built on `time_after_conv`), not from this helper; this function is retained
  /// as the literal port of the reference closed form (and for the
  /// 100-frame-chunk path, where the two agree). Computed in `i64`
  /// (overflow-safe for any realistic frame count).
  pub fn feature_output_length(mel_len: i64) -> i64 {
    if mel_len <= 0 {
      return 0;
    }
    // floor_div toward negative infinity matches the reference's `mx.floor`.
    let floor_div = |a: i64, b: i64| -> i64 { a.div_euclid(b) };
    let leave = mel_len % 100;
    let feat = floor_div(leave - 1, 2) + 1;
    let inner = floor_div(feat - 1, 2); // + 1 - 1 cancels
    floor_div(inner, 2)
      .saturating_add(1)
      .saturating_add((mel_len / 100) * 13)
  }

  /// The **exact** number of post-CNN audio rows the windowed encoder produces
  /// for `valid_len` valid mel frames at conv-chunk size `chunk = n_window * 2`.
  ///
  /// The valid frames are split into `chunk`-frame conv chunks (the last is the
  /// remainder), and each chunk contributes `time_after_conv(chunk_len)` rows —
  /// the true `kernel = 3, stride = 2, padding = 1` three-fold output length of
  /// that chunk ([`AudioEncoderConfig::time_after_conv`], the actual Conv1d/2d
  /// output-length recurrence the conv stem realizes). The total is the sum of
  /// the per-chunk counts, which is exactly the number of rows
  /// [`encode_features`](Self::encode_features) emits for the chunk batch.
  ///
  /// This is the single source of truth for both the windowed encoder's
  /// per-chunk keep / `seq_len` / block-mask sizing **and** the aligner's
  /// `<audio_pad>` token count, so the two always agree for any `n_window`. For
  /// the standard 100-frame chunk (`n_window = 50`) it equals
  /// [`feature_output_length`](Self::feature_output_length)`(valid_len)` (the
  /// reference closed form, additive across 100-frame boundaries), preserving
  /// the default-config counts; for any other chunk size it stays consistent
  /// with the conv stem where the closed form would not. A non-positive
  /// `valid_len` is `0`; a `chunk < 1` is treated as `1` (the caller validates
  /// `n_window >= 1`). Computed in `i64` (overflow-safe for any realistic frame
  /// count).
  pub fn windowed_output_length(valid_len: i64, chunk: i64) -> i64 {
    if valid_len <= 0 {
      return 0;
    }
    let chunk = chunk.max(1);
    let remainder = valid_len % chunk;
    let num_chunks = valid_len / chunk + i64::from(remainder != 0);
    let last_len = if remainder == 0 { chunk } else { remainder };
    let mut total: i64 = 0;
    for j in 0..num_chunks {
      let clen = if j == num_chunks - 1 { last_len } else { chunk };
      total = total.saturating_add(AudioEncoderConfig::time_after_conv(clen));
    }
    total
  }

  /// Encode a single-utterance mel for the aligner, trimming padded frames to
  /// the per-sample valid `feature_lengths` and running the reference's
  /// **windowed block-diagonal** audio encoder: `[1, num_mel_bins, time]` →
  /// `[1, time', output_dim]`.
  ///
  /// This is the aligner's audio path and faithfully mirrors mlx-audio's
  /// `AudioEncoder.__call__` (specialized to one utterance — the only shape the
  /// aligner produces). The valid mel frames are split into
  /// `chunk = n_window * 2`-frame conv chunks (the last chunk is the remainder),
  /// each chunk is conv-subsampled, its valid post-CNN frames are concatenated
  /// into one sequence, and the transformer runs under a block-diagonal additive
  /// mask so every `n_window_infer`-derived window attends only within itself.
  /// For a short utterance that
  /// fits one conv chunk this collapses to the degenerate single-window path
  /// (one chunk, full attention), preserving the prior behavior exactly. The
  /// post-CNN length is the exact conv recurrence over the chunk decomposition
  /// ([`windowed_output_length`](Self::windowed_output_length) of the valid
  /// frames) — the count of audio tokens the aligner splices.
  ///
  /// - the input must be batch `1` ([`Error::OutOfRange`] otherwise — the
  ///   reference flattens a *batch* into one ragged windowed sequence, which the
  ///   aligner never needs and is not ported);
  /// - the valid length (`feature_lengths[0]` if given, else the full `time`
  ///   axis) may span any number of conv windows; when `feature_lengths` is
  ///   given it must hold exactly one entry in `[0, time]` ([`Error::OutOfRange`]
  ///   for an out-of-range length, [`Error::LengthMismatch`] for a different
  ///   count) — an out-of-range length is rejected, not clamped;
  /// - padded trailing frames beyond the valid length are sliced off so they do
  ///   not contribute audio tokens or attention.
  ///
  /// `feature_lengths` is the per-sample count of valid (non-padded) mel frames
  /// (the reference's `feature_attention_mask.sum(-1)`); `None` means the whole
  /// `time` axis is valid (an unpadded utterance).
  pub fn forward_single_window(
    &self,
    input_features: &Array,
    feature_lengths: Option<&[i64]>,
  ) -> Result<Array> {
    let shape = input_features.shape();
    if shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "AudioEncoder::forward_single_window: input_features must be rank-3 [batch, num_mel_bins, time]",
        shape.len() as u32,
        shape,
      )));
    }
    let batch = shape[0];
    let time = i64::try_from(shape[2]).unwrap_or(i64::MAX);
    if batch != 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder::forward_single_window: batch size",
        "only single-utterance (batch == 1) is supported; the ragged windowed batch path is not yet ported",
        batch.to_string(),
      )));
    }

    // Per-sample valid length: from the mask sum if given, else the full time
    // axis. This is the single-utterance (batch == 1) path, so exactly one
    // length is expected; a different count is rejected. A length outside `[0,
    // time]` is rejected rather than clamped — clamping a negative or overlong
    // mask sum would silently rewrite malformed runtime input into a DIFFERENT
    // valid audio span (a zero-length tensor, or the full padded axis),
    // producing plausible-but-wrong timestamps instead of failing at the public
    // boundary.
    let valid_len = match feature_lengths {
      Some(lens) => {
        if lens.len() != 1 {
          return Err(Error::LengthMismatch(LengthMismatchPayload::new(
            "AudioEncoder::forward_single_window: feature_lengths vs batch",
            1,
            lens.len(),
          )));
        }
        let l = lens[0];
        if l < 0 || l > time {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "AudioEncoder::forward_single_window: feature_lengths[0]",
            "must be in [0, time] (the mel time axis)",
            format!("length={l}, time={time}"),
          )));
        }
        l
      }
      None => time,
    };

    // Trim padded trailing frames (slice the time axis to the valid length) so
    // padding contributes neither audio tokens nor attention.
    let n_mels = dim_i32(&input_features.shape(), 1, "AudioEncoder: num_mel_bins")?;
    let trimmed = if valid_len < time {
      let valid_i = i32::try_from(valid_len).unwrap_or(i32::MAX);
      crate::ops::indexing::slice(
        input_features,
        &[0, 0, 0],
        &[1, n_mels, valid_i],
        &[1, 1, 1],
      )?
    } else {
      input_features.try_clone()?
    };

    // A single conv chunk (the reference's degenerate one-window case) is full
    // (unmasked) attention — the block mask for one window is all-zeros. Reuse
    // the plain forward so the short-utterance numerics stay byte-identical.
    let chunk = i64::from(self.config.n_window).saturating_mul(2);
    if valid_len <= chunk {
      return self.forward_inner(&trimmed, Mask::None);
    }

    // Longer than one conv chunk → the reference's windowed block-diagonal path.
    self.forward_windowed(&trimmed, valid_len, n_mels)
  }

  /// The reference's windowed block-diagonal audio encoder for one utterance of
  /// `valid_len` valid mel frames (`> n_window * 2`, the multi-chunk case).
  ///
  /// Mirrors mlx-audio `AudioEncoder.__call__`
  /// (`stt/models/qwen3_asr/qwen3_asr.py:318`) for a single utterance:
  ///
  /// 1. **Chunk split** — the valid frames are split into
  ///    `chunk = n_window * 2`-frame chunks; the last chunk is the remainder
  ///    (or a full chunk when `valid_len` is a multiple of `chunk`).
  /// 2. **Per-chunk conv subsampling** — every chunk is right-padded with zeros
  ///    to `max_chunk_len` (the longest chunk = `chunk` whenever a full chunk
  ///    exists) and the chunks are stacked into a `(num_chunks, n_mels,
  ///    max_chunk_len)` batch. The batch is processed along the chunk axis in
  ///    `config.conv_chunksize`-sized slices — each slice runs
  ///    [`encode_features`](Self::encode_features) (the conv stem + `conv_out`)
  ///    and the encoded slices are concatenated back into
  ///    `(num_chunks, t', d_model)` with `t' = time_after_conv(max_chunk_len)`
  ///    (the exact conv output length of the padded chunk batch).
  ///    This mirrors the reference's `for i in range(0, total_chunks,
  ///    conv_chunksize)` conv loop, which bounds the conv working set without
  ///    changing the result: the conv is per-chunk independent, so slicing the
  ///    chunk/batch axis only caps the intermediate — the per-chunk outputs are
  ///    identical to a single all-at-once pass.
  /// 3. **Per-chunk sinusoidal positions** — `pos_emb[:t']` (positions
  ///    `0..t'`) is added to *every* chunk before trimming, so positions reset
  ///    per chunk exactly as in the reference (the table is f32; it is cast to
  ///    the activation dtype before the add so a bf16/f16 activation is not
  ///    promoted).
  /// 4. **Valid-frame concatenation** — each chunk keeps its leading
  ///    `time_after_conv(chunk_len)` rows (the exact conv output length of that
  ///    chunk; the partial last chunk keeps fewer), and the kept rows are
  ///    concatenated into one `(1, seq_len, d_model)` sequence with `seq_len =
  ///    windowed_output_length(valid_len, chunk)` (the sum of those exact
  ///    per-chunk counts).
  /// 5. **Block-diagonal mask** — the post-CNN sequence is grouped into windows
  ///    of `window_aftercnn = max_len_after_cnn * (n_window_infer / (n_window *
  ///    2))` frames (`_create_block_attention_mask` over the cumulative window
  ///    boundaries); a `(seq_len, seq_len)` additive mask is `0` inside each
  ///    window block and `-1e9` elsewhere, broadcast to `(1, 1, seq_len,
  ///    seq_len)` for SDPA. The mask host buffer is f32, cast to the activation
  ///    dtype before SDPA so it cannot promote the activation.
  ///
  /// `valid_len` is the (already trimmed) valid mel-frame count and `n_mels`
  /// its mel-bin axis. All host-side sizing is `i64`/`usize` with
  /// overflow-checked arithmetic (a typed error on a usize wrap) and fallible
  /// allocation (a typed error rather than a panic on OOM).
  fn forward_windowed(&self, valid: &Array, valid_len: i64, n_mels: i32) -> Result<Array> {
    let chunk = i64::from(self.config.n_window).saturating_mul(2);
    // chunk >= 1: validate() rejects n_window <= 0, and the caller only reaches
    // here with valid_len > chunk, so num_chunks >= 2.
    if chunk < 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder::forward_windowed: conv chunk size (n_window * 2)",
        "must be >= 1",
        format_smolstr!("{chunk}"),
      )));
    }

    // Per-chunk mel-frame lengths: full `chunk`s then the remainder tail.
    // ceil(valid_len / chunk) with chunk >= 1 (checked above).
    let remainder = valid_len % chunk;
    let num_chunks = valid_len / chunk + i64::from(remainder != 0);
    let last_len = if remainder == 0 { chunk } else { remainder };
    let num_chunks_usize = usize::try_from(num_chunks).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder::forward_windowed: chunk count",
        "exceeds usize::MAX",
        num_chunks.to_string(),
      ))
    })?;
    let mut chunk_lengths: Vec<i64> = Vec::new();
    reserve_or_error(
      &mut chunk_lengths,
      "AudioEncoder: chunk_lengths",
      num_chunks_usize,
    )?;
    for j in 0..num_chunks {
      chunk_lengths.push(if j == num_chunks - 1 { last_len } else { chunk });
    }
    // The longest chunk = `chunk` whenever a full chunk exists (num_chunks >= 2
    // here, so always), else the single short chunk — i.e. max of the lengths.
    let max_chunk_len = chunk_lengths.iter().copied().max().unwrap_or(last_len);
    let max_chunk_i = i32::try_from(max_chunk_len).unwrap_or(i32::MAX);

    // Build each chunk by slicing the valid mel and right-padding to
    // `max_chunk_len`, then stack into `(num_chunks, n_mels, max_chunk_len)`.
    // (Mirrors the reference's per-chunk `mx.pad` + `mx.stack`.)
    let pad_zero = scalar_f32(0.0)?.astype(valid.dtype()?)?;
    let mut chunks: Vec<Array> = Vec::new();
    reserve_or_error(&mut chunks, "AudioEncoder: chunk slices", num_chunks_usize)?;
    let mut pos: i64 = 0;
    for &clen in &chunk_lengths {
      let start = i32::try_from(pos).unwrap_or(i32::MAX);
      let stop = i32::try_from(pos + clen).unwrap_or(i32::MAX);
      // (n_mels, clen): slice the single utterance `(1, n_mels, valid_len)`.
      let sliced =
        crate::ops::indexing::slice(valid, &[0, 0, start], &[1, n_mels, stop], &[1, 1, 1])?;
      let clen_i = i32::try_from(clen).unwrap_or(i32::MAX);
      let padded = if clen_i < max_chunk_i {
        // Right-pad the time axis (axis 2) up to max_chunk_len.
        pad(
          &sliced,
          &[2],
          &[0],
          &[max_chunk_i - clen_i],
          &pad_zero,
          c"constant",
        )?
      } else {
        sliced
      };
      chunks.push(padded);
      pos += clen;
    }
    let chunk_refs: Vec<&Array> = chunks.iter().collect();
    // Each chunk is `(1, n_mels, max_chunk_len)` (a slice of the single
    // utterance), so concatenating on axis 0 yields the
    // `(num_chunks, n_mels, max_chunk_len)` batch directly.
    let padded_feature = concatenate(&chunk_refs, 0)?;

    // Conv subsample the chunk batch in `conv_chunksize`-sized slices along the
    // chunk axis, then concatenate the encoded slices → (num_chunks, t', d_model).
    // This mirrors the reference's `for i in range(0, total_chunks,
    // conv_chunksize)` conv loop: the conv stem + `conv_out` are per-chunk
    // independent, so slicing the chunk/batch axis bounds the conv working set
    // without changing any per-chunk output (the result is identical to a single
    // all-at-once pass). `conv_chunksize` is a model batching parameter, not an
    // input cap — `validate()` guarantees it is >= 1.
    let conv_chunksize = i64::from(self.config.conv_chunksize).max(1);
    let num_slices = num_chunks / conv_chunksize + i64::from(num_chunks % conv_chunksize != 0);
    let num_slices_usize = usize::try_from(num_slices).unwrap_or(usize::MAX);
    let mut encoded_slices: Vec<Array> = Vec::new();
    reserve_or_error(
      &mut encoded_slices,
      "AudioEncoder: encoded conv slices",
      num_slices_usize,
    )?;
    let mut slice_start: i64 = 0;
    while slice_start < num_chunks {
      let slice_stop = (slice_start + conv_chunksize).min(num_chunks);
      let start_i = i32::try_from(slice_start).unwrap_or(i32::MAX);
      let stop_i = i32::try_from(slice_stop).unwrap_or(i32::MAX);
      // padded_feature[slice_start:slice_stop] → (s, n_mels, max_chunk_len).
      let batch_slice = crate::ops::indexing::slice(
        &padded_feature,
        &[start_i, 0, 0],
        &[stop_i, n_mels, max_chunk_i],
        &[1, 1, 1],
      )?;
      encoded_slices.push(self.encode_features(&batch_slice)?);
      slice_start = slice_stop;
    }
    let encoded_refs: Vec<&Array> = encoded_slices.iter().collect();
    let mut x = concatenate(&encoded_refs, 0)?;
    let t_after = dim_i32(&x.shape(), 1, "AudioEncoder: windowed post-CNN seq")?;

    // Per-chunk sinusoidal positions (positions reset per chunk: pos_emb[:t']
    // is added to *every* chunk). Cast the f32 table to x's dtype first so a
    // bf16/f16 activation is not promoted to f32.
    let pos_emb = self.positional_embedding.forward(t_after)?;
    let pos_emb = pos_emb.astype(x.dtype()?)?;
    let pos_emb = expand_dims_axes(&pos_emb, &[0])?; // (1, t', d_model) broadcast
    x = x.add(&pos_emb)?;

    // Keep each chunk's leading valid post-CNN frames and concatenate into one
    // `(1, seq_len, d_model)` sequence. The per-chunk keep is the EXACT conv
    // output length `time_after_conv(chunk_len)` — the same recurrence the conv
    // stem realizes — so the slice never asks for more rows than `encode_features`
    // produced (`t' = time_after_conv(max_chunk_len)`), and it is monotonic in
    // the chunk length so every per-chunk count is <= t'. (The reference's
    // `_get_feat_extract_output_lengths` closed form would over-count here for a
    // non-100-frame chunk; the exact recurrence agrees with it at chunk 100.)
    let d_model = dim_i32(&x.shape(), 2, "AudioEncoder: windowed d_model")?;
    let mut valid_rows: Vec<Array> = Vec::new();
    reserve_or_error(
      &mut valid_rows,
      "AudioEncoder: valid post-CNN rows",
      num_chunks_usize,
    )?;
    let mut seq_len: i64 = 0;
    for (i, &clen) in chunk_lengths.iter().enumerate() {
      let keep = AudioEncoderConfig::time_after_conv(clen);
      let keep_i = i32::try_from(keep).unwrap_or(i32::MAX);
      let i_i = i32::try_from(i).unwrap_or(i32::MAX);
      // x[i, :keep, :]  → (1, keep, d_model) (keep the leading axis for concat).
      let rows =
        crate::ops::indexing::slice(&x, &[i_i, 0, 0], &[i_i + 1, keep_i, d_model], &[1, 1, 1])?;
      valid_rows.push(rows);
      seq_len = seq_len.saturating_add(keep);
    }
    let row_refs: Vec<&Array> = valid_rows.iter().collect();
    // Concatenate on the time axis (axis 1) → (1, seq_len, d_model).
    let hidden = concatenate(&row_refs, 1)?;

    // Block-diagonal additive mask over the post-CNN sequence. The window
    // grouping is sized from the EXACT post-conv length of a full chunk
    // (`time_after_conv(max_chunk_len)`), consistent with the per-chunk keep and
    // `seq_len` above, so the mask covers exactly the concatenated rows.
    let max_len_after_cnn = AudioEncoderConfig::time_after_conv(max_chunk_len);
    let mask = self.block_attention_mask(seq_len, max_len_after_cnn, hidden.dtype()?)?;
    // (seq_len, seq_len) → (1, 1, seq_len, seq_len) for SDPA broadcast.
    let mask = expand_dims_axes(&mask, &[0, 1])?;

    // Transformer layers under the block mask, then the output head — the same
    // tail as `forward_inner`, but the positions were already added per chunk.
    let mut h = hidden;
    for layer in &self.layers {
      h = layer.forward(&h, Mask::Array(&mask))?;
    }
    let h = self.ln_post.forward(&h)?;
    let h = linear(&h, &self.proj1_weight, Some(&self.proj1_bias))?;
    let h = crate::lm::nn::activations::gelu(&h)?;
    linear(&h, &self.proj2_weight, Some(&self.proj2_bias))
  }

  /// Build the reference's ragged block-diagonal additive attention mask
  /// (`_create_block_attention_mask`) for one utterance whose post-CNN sequence
  /// has `seq_len` frames, grouped into windows of `window_aftercnn =
  /// max_len_after_cnn * (n_window_infer / (n_window * 2))` frames.
  ///
  /// The mask is `(seq_len, seq_len)`, `0.0` on each window's diagonal block
  /// `[start:end, start:end]` and `-1e9` elsewhere (so each window attends only
  /// within itself). Built as a host f32 buffer then cast to `dtype` (the
  /// activation dtype) so it does not promote a bf16/f16 activation. The
  /// `seq_len^2` buffer size is overflow-checked and allocated fallibly (a
  /// typed error rather than a panic on a usize wrap or an allocation failure).
  fn block_attention_mask(
    &self,
    seq_len: i64,
    max_len_after_cnn: i64,
    dtype: Dtype,
  ) -> Result<Array> {
    // window_aftercnn = max_len_after_cnn * (n_window_infer / (n_window * 2)).
    // n_window_infer / (n_window*2) is integer division in the reference.
    let n_window_step = i64::from(self.config.n_window).saturating_mul(2);
    let windows_per_infer = if n_window_step > 0 {
      i64::from(self.config.n_window_infer) / n_window_step
    } else {
      0
    };
    let window_aftercnn = max_len_after_cnn.saturating_mul(windows_per_infer).max(1);

    let seq = usize::try_from(seq_len).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder::block_attention_mask: seq_len",
        "exceeds usize::MAX",
        seq_len.to_string(),
      ))
    })?;
    // The (seq_len, seq_len) mask buffer size. `saturating_mul` cannot wrap
    // usize; an oversized product is then caught by the fallible reservation
    // below (a typed error rather than a panic), so the allocation is sound.
    let mask_elems = seq.saturating_mul(seq);

    // Cumulative window boundaries over the single utterance's `seq_len` frames:
    // full `window_aftercnn` windows then a remainder window (the reference's
    // `cu_chunk_lens` / `cu_seqlens` for one utterance, which always sums to
    // `seq_len`).
    let mut bounds: Vec<usize> = Vec::new();
    reserve_or_error(&mut bounds, "AudioEncoder: window bounds", 2)?;
    bounds.push(0);
    let win = window_aftercnn.max(1);
    let mut acc: i64 = 0;
    let num_full = seq_len / win;
    for _ in 0..num_full {
      acc = acc.saturating_add(win);
      bounds.push(usize::try_from(acc).unwrap_or(usize::MAX));
    }
    if seq_len % win != 0 {
      bounds.push(seq);
    }

    // Host buffer: -1e9 everywhere, 0.0 on each window's diagonal block.
    let mut data: Vec<f32> = Vec::new();
    reserve_or_error(&mut data, "AudioEncoder: block mask buffer", mask_elems)?;
    data.resize(mask_elems, MASK_NEG);
    for w in bounds.windows(2) {
      let (start, end) = (w[0].min(seq), w[1].min(seq));
      for r in start..end {
        // `row + start <= r*seq + start < (r+1)*seq <= seq*seq == mask_elems`,
        // and `data` was sized to `mask_elems` above, so the slice is in range.
        let row = r.saturating_mul(seq);
        for slot in data.iter_mut().skip(row + start).take(end - start) {
          *slot = 0.0;
        }
      }
    }
    let seq_i = i32::try_from(seq_len).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoder::block_attention_mask: seq_len",
        "exceeds i32::MAX",
        seq_len.to_string(),
      ))
    })?;
    let mask = Array::from_slice(&data, &[seq_i, seq_i])?;
    mask.astype(dtype)
  }

  fn forward_inner(&self, input_features: &Array, mask: Mask<'_>) -> Result<Array> {
    let mut x = self.encode_features(input_features)?;
    // The reference's `AudioEncoder.__call__` does NOT scale the post-CNN
    // features by `embed_scale` (the field is computed in `__init__` but never
    // applied in the forward — a vestige of the Whisper-style encoder it is
    // adapted from), so neither do we, on any config.
    //
    // Add the sinusoidal position embedding sliced to the post-CNN length. The
    // table is built in f32; cast it to x's dtype before adding so a bf16/f16
    // activation is not promoted to f32 (matching the reference, which adds the
    // positional embedding in the activation dtype).
    let seqlen = dim_i32(&x.shape(), 1, "AudioEncoder: post-CNN seq")?;
    let pos = self.positional_embedding.forward(seqlen)?;
    let pos = pos.astype(x.dtype()?)?;
    // pos is (T', d_model); add with a leading batch broadcast axis.
    let pos = expand_dims_axes(&pos, &[0])?;
    x = x.add(&pos)?;

    for layer in &self.layers {
      x = layer.forward(&x, mask)?;
    }
    let x = self.ln_post.forward(&x)?;
    let x = linear(&x, &self.proj1_weight, Some(&self.proj1_bias))?;
    let x = crate::lm::nn::activations::gelu(&x)?;
    linear(&x, &self.proj2_weight, Some(&self.proj2_bias))
  }

  /// Build an [`AudioEncoder`] from a parsed [`AudioEncoderConfig`] and a flat
  /// name → [`Array`] weight map (already [`sanitize`](super::sanitize)d — the
  /// conv weights transposed to MLX channels-last). Weight keys follow the
  /// reference module tree (`conv2d1.{weight,bias}`, ...,
  /// `conv_out.weight`, `layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`,
  /// `layers.{i}.{self_attn_layer_norm,final_layer_norm}.{weight,bias}`,
  /// `layers.{i}.{fc1,fc2}.{weight,bias}`, `ln_post.{weight,bias}`,
  /// `proj1.{weight,bias}`, `proj2.{weight,bias}`). The map is drained; a
  /// missing required weight is [`Error::MissingKey`].
  pub fn from_weights(
    config: AudioEncoderConfig,
    mut weights: HashMap<String, Array>,
  ) -> Result<Self> {
    config.validate()?;
    let hidden = config.downsample_hidden_size;
    let d_model = config.d_model;
    let ffn = config.encoder_ffn_dim;
    let output_dim = config.output_dim;
    let conv_out_in = config.conv_out_in_features()?;

    // Conv2d stem. Post-sanitize MLX weight is (out, kH, kW, in): conv1 in=1,
    // conv2/conv3 in=hidden; all out=hidden, kernel 3x3.
    let conv2d1 = take_conv2d(&mut weights, "conv2d1", hidden, 3, 3, 1)?;
    let conv2d2 = take_conv2d(&mut weights, "conv2d2", hidden, 3, 3, hidden)?;
    let conv2d3 = take_conv2d(&mut weights, "conv2d3", hidden, 3, 3, hidden)?;

    // conv_out: bias-free Linear (d_model, hidden * freq_after_conv).
    let conv_out_weight = take_shaped(
      &mut weights,
      "conv_out.weight",
      "conv_out weight (d_model, downsample_hidden_size * freq_after_conv)",
      &[d_model, conv_out_in],
    )?;

    let positional_embedding =
      SinusoidalPositionEmbedding::new(config.max_source_positions, d_model)?;

    let num_layers = config.encoder_layers;
    let head_dim = config.head_dim();
    let scaling = (head_dim as f32).powf(-0.5);
    let mut layers: Vec<AudioEncoderLayer> = Vec::new();
    reserve_or_error(&mut layers, "AudioEncoderLayer", num_layers as usize)?;
    let proj = [d_model, d_model];
    for i in 0..num_layers {
      let p = format!("layers.{i}");
      let q = format!("{p}.self_attn");
      let self_attn = AudioAttention {
        q_weight: take_shaped(
          &mut weights,
          &format!("{q}.q_proj.weight"),
          "self_attn q_proj weight (d_model, d_model)",
          &proj,
        )?,
        q_bias: take_shaped(
          &mut weights,
          &format!("{q}.q_proj.bias"),
          "self_attn q_proj bias (d_model)",
          &[d_model],
        )?,
        k_weight: take_shaped(
          &mut weights,
          &format!("{q}.k_proj.weight"),
          "self_attn k_proj weight (d_model, d_model)",
          &proj,
        )?,
        k_bias: take_shaped(
          &mut weights,
          &format!("{q}.k_proj.bias"),
          "self_attn k_proj bias (d_model)",
          &[d_model],
        )?,
        v_weight: take_shaped(
          &mut weights,
          &format!("{q}.v_proj.weight"),
          "self_attn v_proj weight (d_model, d_model)",
          &proj,
        )?,
        v_bias: take_shaped(
          &mut weights,
          &format!("{q}.v_proj.bias"),
          "self_attn v_proj bias (d_model)",
          &[d_model],
        )?,
        out_weight: take_shaped(
          &mut weights,
          &format!("{q}.out_proj.weight"),
          "self_attn out_proj weight (d_model, d_model)",
          &proj,
        )?,
        out_bias: take_shaped(
          &mut weights,
          &format!("{q}.out_proj.bias"),
          "self_attn out_proj bias (d_model)",
          &[d_model],
        )?,
        num_heads: config.encoder_attention_heads,
        head_dim,
        embed_dim: d_model,
        scaling,
      };
      let self_attn_layer_norm = take_layernorm(
        &mut weights,
        &format!("{p}.self_attn_layer_norm"),
        d_model,
        "self_attn_layer_norm",
      )?;
      let fc1_weight = take_shaped(
        &mut weights,
        &format!("{p}.fc1.weight"),
        "fc1 weight (encoder_ffn_dim, d_model)",
        &[ffn, d_model],
      )?;
      let fc1_bias = take_shaped(
        &mut weights,
        &format!("{p}.fc1.bias"),
        "fc1 bias (encoder_ffn_dim)",
        &[ffn],
      )?;
      let fc2_weight = take_shaped(
        &mut weights,
        &format!("{p}.fc2.weight"),
        "fc2 weight (d_model, encoder_ffn_dim)",
        &[d_model, ffn],
      )?;
      let fc2_bias = take_shaped(
        &mut weights,
        &format!("{p}.fc2.bias"),
        "fc2 bias (d_model)",
        &[d_model],
      )?;
      let final_layer_norm = take_layernorm(
        &mut weights,
        &format!("{p}.final_layer_norm"),
        d_model,
        "final_layer_norm",
      )?;
      layers.push(AudioEncoderLayer {
        self_attn,
        self_attn_layer_norm,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        final_layer_norm,
      });
    }

    let ln_post = take_layernorm(&mut weights, "ln_post", d_model, "ln_post")?;
    let proj1_weight = take_shaped(
      &mut weights,
      "proj1.weight",
      "proj1 weight (d_model, d_model)",
      &proj,
    )?;
    let proj1_bias = take_shaped(
      &mut weights,
      "proj1.bias",
      "proj1 bias (d_model)",
      &[d_model],
    )?;
    let proj2_weight = take_shaped(
      &mut weights,
      "proj2.weight",
      "proj2 weight (output_dim, d_model)",
      &[output_dim, d_model],
    )?;
    let proj2_bias = take_shaped(
      &mut weights,
      "proj2.bias",
      "proj2 bias (output_dim)",
      &[output_dim],
    )?;

    Ok(Self {
      config,
      conv2d1,
      conv2d2,
      conv2d3,
      conv_out_weight,
      positional_embedding,
      layers,
      ln_post,
      proj1_weight,
      proj1_bias,
      proj2_weight,
      proj2_bias,
    })
  }
}

// ───────────────────────── weight-fetch helpers ─────────────────────────

/// Pull a weight by exact key, erroring with the key if absent.
fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights
    .remove(key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("AudioEncoder::from_weights", key)))
}

/// Assert a tensor's shape equals `expected` (rank + every dim) before it is
/// stored, so a checkpoint whose weight shape disagrees with the config-derived
/// expectation is rejected here rather than running a different graph. On
/// mismatch returns [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`]
/// naming `key`.
fn expect_shape(
  tensor: &Array,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<()> {
  let actual = tensor.shape();
  let matches = actual.len() == expected.len()
    && actual
      .iter()
      .zip(expected.iter())
      .all(|(&a, &e)| e >= 0 && a as i64 == i64::from(e));
  if !matches {
    let expected_usize: Vec<usize> = expected.iter().map(|&e| e.max(0) as usize).collect();
    return Err(Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
      key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        expected_usize,
        actual,
      )),
    )));
  }
  Ok(())
}

/// [`take`] a weight then assert its shape — the fused fetch-and-check used for
/// every tensor stored verbatim.
fn take_shaped(
  weights: &mut HashMap<String, Array>,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<Array> {
  let tensor = take(weights, key)?;
  expect_shape(&tensor, key, descriptor, expected)?;
  Ok(tensor)
}

/// Build a [`Conv2dLayer`] from `<prefix>.{weight,bias}`, pinning the
/// channels-last MLX weight shape `(out, kH, kW, in)` and the `(out,)` bias.
fn take_conv2d(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  out: i32,
  kh: i32,
  kw: i32,
  in_ch: i32,
) -> Result<Conv2dLayer> {
  let weight = take_shaped(
    weights,
    &format!("{prefix}.weight"),
    "conv2d weight (out, kH, kW, in) channels-last",
    &[out, kh, kw, in_ch],
  )?;
  let bias = take_shaped(
    weights,
    &format!("{prefix}.bias"),
    "conv2d bias (out)",
    &[out],
  )?;
  Ok(Conv2dLayer { weight, bias })
}

/// Build a [`LayerNorm`] from `<prefix>.{weight,bias}` of shape `(dims,)`.
fn take_layernorm(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  dims: i32,
  descriptor_stem: &'static str,
) -> Result<LayerNorm> {
  // Static descriptors (the &'static str payloads cannot carry the dynamic
  // per-layer prefix; the LayerKeyed wrapper carries the exact key).
  let (w_desc, b_desc): (&'static str, &'static str) = match descriptor_stem {
    "self_attn_layer_norm" => (
      "self_attn_layer_norm weight (d_model)",
      "self_attn_layer_norm bias (d_model)",
    ),
    "final_layer_norm" => (
      "final_layer_norm weight (d_model)",
      "final_layer_norm bias (d_model)",
    ),
    _ => ("ln_post weight (d_model)", "ln_post bias (d_model)"),
  };
  let weight = take_shaped(weights, &format!("{prefix}.weight"), w_desc, &[dims])?;
  let bias = take_shaped(weights, &format!("{prefix}.bias"), b_desc, &[dims])?;
  // LayerNorm eps: the reference uses nn.LayerNorm's default (1e-5).
  Ok(LayerNorm::new(Some(weight), Some(bias), 1e-5))
}

// ───────────────────────── small helpers ─────────────────────────

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
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
      format_smolstr!("{d}"),
    ))
  })
}
