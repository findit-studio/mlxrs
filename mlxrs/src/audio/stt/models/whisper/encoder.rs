//! The Whisper audio encoder — `AudioEncoder` (`whisper.py:409-437`).
//!
//! Faithful port of the `nn.Module` that turns a log-mel spectrogram into the
//! encoder hidden states the decoder cross-attends over:
//!
//! 1. `conv1` (`n_mels -> n_state`, kernel 3, pad 1) → exact GELU;
//! 2. `conv2` (`n_state -> n_state`, kernel 3, **stride 2**, pad 1) → exact
//!    GELU — this halves the time axis (`3000 -> 1500`);
//! 3. add the precomputed `sinusoids(n_ctx, n_state)` positional embedding;
//! 4. `n_audio_layer` × [`ResidualAttentionBlock`] (self-attention only — no
//!    cross-attention, no causal mask);
//! 5. a final `ln_post` LayerNorm.
//!
//! Input is the Whisper mel `(num_frames, n_mels)` (frames on axis 0, mel bins
//! on axis 1 — see [`super::audio`]); the convolutions run in MLX's
//! channels-last `(N, L, C)` layout, so the mel is lifted to `(1, num_frames,
//! n_mels)` before `conv1` and the singleton batch axis is carried through.
//! Output is `(1, n_ctx, n_audio_state)` (`(1, 1500, n_state)`).

use crate::{
  Array, Dtype, Result,
  error::{Error, ShapePairMismatchPayload},
  lm::nn::{activations::gelu, norm::LayerNorm},
  ops::conv::conv1d,
};

use super::layers::{ResidualAttentionBlock, sinusoids};

/// Whisper positional-embedding max timescale (`sinusoids(..., max_timescale=
/// 10000)`, `whisper.py:319`).
const MAX_TIMESCALE: f64 = 10_000.0;

/// A 1-D convolution layer with an additive bias — `nn.Conv1d`.
///
/// `mlx.nn.Conv1d` stores its `weight` channels-last `(C_out, K, C_in)` (the
/// same layout [`conv1d`] expects) and adds a `(C_out,)` bias on the channel
/// (last) axis after the convolution. mlx-c's `mlx_conv1d` has no bias slot, so
/// the bias add is reproduced here.
#[derive(Debug)]
struct Conv1dLayer {
  /// `(C_out, K, C_in)` convolution weight.
  weight: Array,
  /// `(C_out,)` bias, broadcast on the channel axis.
  bias: Array,
  /// Convolution stride along the time axis (1 for `conv1`, 2 for `conv2`).
  stride: i32,
  /// Convolution padding (1 for both Whisper convs).
  padding: i32,
}

impl Conv1dLayer {
  fn new(weight: Array, bias: Array, stride: i32, padding: i32) -> Self {
    Self {
      weight,
      bias,
      stride,
      padding,
    }
  }

  /// `conv1d(x) + bias`. `x` is `(N, L, C_in)`; the result is `(N, L_out,
  /// C_out)` with `bias` broadcast on the last (channel) axis.
  fn forward(&self, x: &Array) -> Result<Array> {
    // dilation 1, groups 1 (Whisper convs are dense).
    let y = conv1d(x, &self.weight, self.stride, self.padding, 1, 1)?;
    y.add(&self.bias)
  }
}

/// The Whisper audio encoder (`whisper.py:409-437`).
#[derive(Debug)]
pub(crate) struct AudioEncoder {
  conv1: Conv1dLayer,
  conv2: Conv1dLayer,
  /// Precomputed `sinusoids(n_ctx, n_state)` positional embedding `(n_ctx,
  /// n_state)`, added after the convolutions.
  positional_embedding: Array,
  blocks: Vec<ResidualAttentionBlock>,
  ln_post: LayerNorm,
}

impl AudioEncoder {
  /// Construct from the loaded conv weights/biases, the transformer blocks,
  /// the final LayerNorm, and the audio context length `n_ctx` (=
  /// `n_audio_ctx`, 1500) / state width `n_state`.
  ///
  /// `conv1_*` is the `n_mels -> n_state` (stride 1) projection and `conv2_*`
  /// is the `n_state -> n_state` **stride-2** projection; each weight is the
  /// channels-last `(C_out, 3, C_in)` MLX layout and each bias is `(C_out,)`.
  ///
  /// `positional_embedding` is computed eagerly via [`sinusoids`] (the
  /// reference precomputes it in `__init__` and never learns it), so it is
  /// **not** a checkpoint tensor. It is cast to the model `dtype` (the reference
  /// `self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)`,
  /// `whisper.py:422`), so adding it to the post-conv activations in
  /// [`Self::forward`] does not promote an f16/bf16 checkpoint's activations to
  /// f32. The cast is a no-op for an f32 model.
  ///
  /// # Errors
  /// Propagates the [`sinusoids`] op error (`n_state` must be even and `>= 2`,
  /// `n_ctx > 0`) and the dtype cast.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    conv1_weight: Array,
    conv1_bias: Array,
    conv2_weight: Array,
    conv2_bias: Array,
    n_ctx: usize,
    n_state: usize,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    dtype: Dtype,
  ) -> Result<Self> {
    // `sinusoids(...).astype(dtype)` (`whisper.py:422`): the table is computed in
    // f32 (the reference builds it in f32 inside `sinusoids`) and then cast to the
    // model dtype, so the `x + positional_embedding` add in `forward` stays in the
    // activation dtype. A no-op cast when `dtype == F32`.
    let positional_embedding = sinusoids(n_ctx, n_state, MAX_TIMESCALE)?.astype(dtype)?;
    Ok(Self {
      // conv1: stride 1, pad 1 (keeps the time axis).
      conv1: Conv1dLayer::new(conv1_weight, conv1_bias, 1, 1),
      // conv2: stride 2, pad 1 (halves the time axis, `3000 -> 1500`).
      conv2: Conv1dLayer::new(conv2_weight, conv2_bias, 2, 1),
      positional_embedding,
      blocks,
      ln_post,
    })
  }

  /// Read-only reference to the precomputed positional embedding (`(n_ctx,
  /// n_state)`).
  #[cfg(test)]
  pub(crate) fn positional_embedding_ref(&self) -> &Array {
    &self.positional_embedding
  }

  /// Read-only reference to the `conv1` weight in its materialized MLX
  /// channels-last `(C_out, K, C_in)` layout (test / introspection — confirms
  /// the HF `(out, in, k)` transpose ran during the build).
  #[cfg(test)]
  pub(crate) fn conv1_weight_ref(&self) -> &Array {
    &self.conv1.weight
  }

  /// Encode a log-mel spectrogram. Faithful port of `AudioEncoder.__call__`
  /// (`whisper.py:427-437`).
  ///
  /// `mel` is the Whisper mel `(num_frames, n_mels)` (or already-batched
  /// `(1, num_frames, n_mels)`); the convolutions and blocks run with a
  /// singleton batch axis and the result is `(1, n_ctx, n_audio_state)`.
  ///
  /// **Batching contract**: Whisper processes one 30-second segment at a time
  /// (the reference's `AudioEncoder` and the entire `decoding.py` seek loop are
  /// strictly single-segment — `encode` is called once per padded window), so
  /// the batch dimension must equal `1`. A 2-D mel is lifted to batch 1; an
  /// already-3-D input whose leading dimension is not `1` is rejected **before**
  /// `conv1` allocates, so an oversized batch cannot drive the conv1 activation
  /// to `B * N_FRAMES * n_audio_state` — past the `N_FRAMES * n_audio_state` the
  /// model config caps — toward an out-of-memory abort.
  ///
  /// The mel frame count (the conv-layout time axis) must likewise equal the
  /// encoder's expected pre-downsample count — `conv2.stride * n_ctx`, which for
  /// a config-built encoder is exactly the fixed `N_FRAMES` (`3000`) every
  /// Whisper segment is padded to (`n_audio_ctx` is pinned to `N_FRAMES /
  /// CONV_DOWNSAMPLE` and `conv2` runs at that stride). This is checked
  /// **before** `conv1` runs, so a wrong frame count is a typed error rather
  /// than an oversized conv activation that could OOM ahead of the post-conv
  /// shape check (and it ties the guard directly to the cap the model config
  /// bounds, `N_FRAMES * n_audio_state`). The mel/channel width is likewise
  /// pinned to the configured `n_mels` (the `conv1` input-channel dimension)
  /// **before** `conv1` contracts that axis, so a public caller cannot reach the
  /// convolution with an unbounded, caller-controlled `C_in`. The reference's
  /// post-conv assert (`x.shape[1:] == self._positional_embedding.shape`) is then
  /// also reproduced so the downsampled `(n_ctx, n_state)` matching the
  /// positional embedding is a typed error rather than a broadcast surprise in
  /// the `+ positional_embedding` step.
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if the input batch dimension is not `1`, the
  ///   mel frame count differs from the expected pre-downsample count, the mel
  ///   channel width differs from the configured `n_mels`, or the post-conv
  ///   `(T, n_state)` does not match the positional embedding `(n_ctx, n_state)`;
  /// - propagates the conv / GELU / add / LayerNorm / attention op errors.
  pub(crate) fn forward(&self, mel: &Array) -> Result<Array> {
    // Lift a 2-D mel `(num_frames, n_mels)` to the channels-last conv layout
    // `(1, num_frames, n_mels)`; an already-3-D input passes through.
    let x = match mel.ndim() {
      2 => {
        let shape = mel.shape();
        let frames = i32::try_from(shape[0]).map_err(|_| dim_overflow("num_frames"))?;
        let mels = i32::try_from(shape[1]).map_err(|_| dim_overflow("n_mels"))?;
        mel.reshape(&[1, frames, mels])?
      }
      _ => mel.try_clone()?,
    };

    // Whisper pads every segment to the fixed `N_FRAMES` mel frames before the
    // encoder, and `conv1` then materializes a `B * frames`-wide activation.
    // Reject a wrong batch OR frame count HERE — before `conv1` allocates — so a
    // hostile mel does not drive that activation to an out-of-memory abort ahead
    // of the post-conv positional-embedding shape check.
    //
    // Batch: Whisper is single-segment (one 30 s window per `encode`), so the
    // leading dimension must be 1 — an oversized batch would multiply the conv1
    // activation past the `N_FRAMES * n_audio_state` cap the model config bounds.
    //
    // Frames: the expected count is the encoder's own pre-downsample width
    // `conv2.stride * n_ctx` (= N_FRAMES once the config pin fixes `n_ctx` to
    // `N_FRAMES / CONV_DOWNSAMPLE`); deriving it from the encoder keeps the
    // sub-module self-consistent while bounding the input to exactly what the cap
    // and the post-conv check expect. The conv layout is `(B, frames, n_mels)`,
    // so the batch axis is dimension 0, the time axis is dimension 1, and `n_ctx`
    // is the positional embedding's row count.
    let n_ctx = self
      .positional_embedding
      .shape()
      .first()
      .copied()
      .unwrap_or(0);
    let expected_frames = n_ctx.saturating_mul(self.conv2.stride.max(0) as usize);
    // The mel/channel axis the conv front-end contracts: `conv1` is stored
    // channels-last `(C_out, K, C_in)`, so its last axis is the configured
    // `n_mels` the input must carry on its own last axis. Deriving it from the
    // conv weight keeps the sub-module self-consistent.
    let expected_mels = self.conv1.weight.shape().get(2).copied().unwrap_or(0);
    let x_shape = x.shape();
    let batch = x_shape.first().copied().unwrap_or(0);
    let frames = x_shape.get(1).copied().unwrap_or(0);
    let mels = x_shape.get(2).copied().unwrap_or(0);
    if x_shape.len() != 3 || batch != 1 {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "AudioEncoder: mel must be (1, N_FRAMES, n_mels) — Whisper encodes one 30s segment at a time (batch must be 1)",
        vec![1usize, expected_frames, expected_mels],
        vec![batch, frames, mels],
      )));
    }
    if frames != expected_frames {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "AudioEncoder: mel frame count must equal conv2.stride * n_ctx (= N_FRAMES) before conv1",
        vec![expected_frames],
        vec![frames],
      )));
    }
    // The mel/channel width must equal the configured `n_mels` (the `conv1`
    // input-channel dimension). Reject a wrong width HERE — before `conv1`
    // contracts the channel axis — so a public caller cannot drive the
    // convolution with an unbounded, caller-controlled `C_in` (the conv backend
    // would otherwise accept any channel count, scaling the gather/matmul work by
    // an extent no config cap bounds).
    if mels != expected_mels {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "AudioEncoder: mel channel width must equal n_mels (the conv1 input-channel dimension) before conv1",
        vec![expected_mels],
        vec![mels],
      )));
    }

    // conv1 → gelu (keeps the time axis), conv2 → gelu (stride 2 halves it).
    let x = gelu(&self.conv1.forward(&x)?)?;
    let x = gelu(&self.conv2.forward(&x)?)?;

    // Assert `x.shape[1:] == positional_embedding.shape` (the reference's
    // "incorrect audio shape" guard), then `x = x + positional_embedding`
    // (broadcast over the batch axis).
    let x_shape = x.shape();
    let pe_shape = self.positional_embedding.shape();
    if x_shape.len() != 3 || x_shape[1..] != pe_shape[..] {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "AudioEncoder: post-conv shape must match the positional embedding",
        pe_shape.to_vec(),
        x_shape.get(1..).unwrap_or(&x_shape).to_vec(),
      )));
    }
    let mut x = x.add(&self.positional_embedding)?;

    // Self-attention-only blocks: no encoder states, no mask, no cache.
    for block in &self.blocks {
      let (out, _) = block.forward(&x, None, None, None)?;
      x = out;
    }

    self.ln_post.forward(&x)
  }
}

/// A dimension exceeding `i32::MAX` when lifting the mel to the conv layout.
fn dim_overflow(which: &'static str) -> Error {
  use crate::error::OutOfRangePayload;
  Error::OutOfRange(OutOfRangePayload::new(
    "AudioEncoder: mel dimension",
    "must fit in i32",
    smol_str::format_smolstr!("{which} exceeds i32::MAX"),
  ))
}

#[cfg(test)]
mod tests;
