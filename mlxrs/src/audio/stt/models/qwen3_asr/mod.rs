//! Qwen3-ASR audio encoder (`qwen3_asr.AudioEncoder`).
//!
//! Phase 2 of the Qwen3 forced-aligner port: the audio tower that turns a
//! log-mel spectrogram into audio embeddings. A Conv2d frontend (three
//! `kernel = 3, stride = 2, padding = 1` convs ~8x downsampling freq + time)
//! over the mel features, then `encoder_layers` transformer self-attention +
//! GELU-FFN blocks with sinusoidal position embeddings, projecting to
//! `output_dim`. See [`AudioEncoder`] for the architecture and the
//! block/chunked attention notes, and [`AudioEncoderConfig`] for the typed
//! configuration and its validation bounds.
//!
//! The mel **frontend is reused** from [`crate::audio::dsp`] /
//! [`crate::audio::features`] (the encoder consumes a precomputed
//! `input_features` mel tensor); the conv op, GELU,
//! [`scaled_dot_product_attention`](crate::lm::nn::attention::scaled_dot_product_attention),
//! and [`LayerNorm`](crate::lm::nn::norm::LayerNorm) are reused from
//! [`crate::ops`] / [`crate::lm::nn`]. Only the Qwen3-ASR-specific assembly is
//! added here.
//!
//! Full numeric parity vs the reference belongs to the later
//! aligner-integration phase; this module is the encoder building block plus
//! its structural/shape and config validation.

#[cfg(feature = "qwen3-asr-aligner")]
mod aligner;
#[cfg(feature = "qwen3-asr-aligner")]
mod aligner_config;
mod audio;
mod config;
#[cfg(feature = "qwen3-asr-aligner")]
mod text;

use std::collections::HashMap;

#[cfg(feature = "qwen3-asr-aligner")]
pub use aligner::{
  AlignWord, ForcedAligner, JpKoSegmenter, PreTokenizedTranscript, RawAlignOptions, RawTranscript,
};
#[cfg(feature = "qwen3-asr-aligner")]
pub use aligner_config::ForcedAlignerConfig;
pub use audio::AudioEncoder;
pub use config::AudioEncoderConfig;
#[cfg(feature = "qwen3-asr-aligner")]
pub use text::{MRopeConfig, Qwen3AsrTextConfig, Qwen3AsrTextModel};

use crate::{
  array::Array, error::Result, model_validation::insert_unique, ops::shape::transpose_axes,
};

/// Rewrite a Qwen3-ASR checkpoint's audio-tower weights into the layout
/// [`AudioEncoder::from_weights`] loads — the Rust analogue of the audio-tower
/// portion of mlx-audio's `Qwen3ASRModel.sanitize`.
///
/// Rules (applied per `(key, value)`):
/// 1. Strip a leading `thinker.` prefix (the HF "thinker" wrapper).
/// 2. Strip a leading `audio_tower.` prefix so the audio-encoder submodule keys
///    (`conv2d1.*`, `layers.*`, `proj1.*`, ...) land at the top level this
///    port's builder reads. Keys that are **not** under `audio_tower.` after
///    the `thinker.` strip (the text decoder, `lm_head`, ...) are dropped —
///    this is the audio tower only.
/// 3. **Only for a raw HF/PyTorch checkpoint**, a 4-D `conv2d*.weight` is
///    transposed PyTorch `(out, in, kH, kW)` → MLX channels-last
///    `(out, kH, kW, in)` via `transpose(0, 2, 3, 1)`. Raw is detected exactly
///    as the reference does (`is_formatted = not any(k.startswith("thinker.")
///    for k in weights)`): the checkpoint is raw iff **any** input key carries
///    the `thinker.` wrapper, evaluated over the **whole** map before any
///    per-key rewrite. An already-converted checkpoint — e.g. the released
///    `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`, whose index has no
///    `thinker.` keys and whose conv kernels are already channels-last —
///    passes through untransposed; re-transposing would permute the kernels to
///    `(out, kW, in, kH)` and the shape-pinned conv loader
///    ([`AudioEncoder::from_weights`]) would reject the load.
/// 4. A duplicate destination key is rejected with [`Error::KeyCollision`]
///    (via [`insert_unique`]) rather than letting an arbitrary (per-run
///    nondeterministic) survivor win — e.g. a checkpoint carrying both the
///    `thinker.audio_tower.<x>` and `audio_tower.<x>` forms.
///
/// [`Error::KeyCollision`]: crate::error::Error::KeyCollision
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  // The reference's formatted-checkpoint gate (`is_formatted = not
  // any(k.startswith("thinker.") for k in weights)`), computed over the whole
  // input map BEFORE the per-key loop: one `thinker.` key anywhere marks the
  // checkpoint raw-HF (every 4-D conv kernel is transposed, prefixed or not);
  // none — including the empty map, vacuously formatted — skips the transpose
  // globally.
  let is_formatted = !weights.keys().any(|k| k.starts_with("thinker."));
  let mut out = HashMap::with_capacity(weights.len());
  for (mut k, mut v) in weights {
    // 1. Drop a leading `thinker.` wrapper prefix.
    if let Some(stripped) = k.strip_prefix("thinker.") {
      k = stripped.to_string();
    }
    // 2. Keep only the audio tower, stripping its prefix. Everything else (the
    //    text decoder, lm_head, ...) is not part of this submodule.
    let Some(rest) = k.strip_prefix("audio_tower.") else {
      continue;
    };
    k = rest.to_string();

    // 3. Raw-HF checkpoints only: PyTorch Conv2d weight (out, in, kH, kW) →
    //    MLX channels-last (out, kH, kW, in). Only the 4-D conv2d kernels need
    //    it; biases and the 1-D/2-D weights pass through unchanged. A
    //    formatted checkpoint's kernels are already channels-last and must not
    //    be re-permuted (the conv loader pins the channels-last shape).
    if !is_formatted && k.starts_with("conv2d") && k.ends_with(".weight") && v.ndim() == 4 {
      v = transpose_axes(&v, &[0, 2, 3, 1])?;
    }

    // 4. Insert, rejecting a duplicate destination key.
    insert_unique(&mut out, k, v, "Qwen3-ASR audio sanitize")?;
  }
  Ok(out)
}

#[cfg(all(test, feature = "qwen3-asr"))]
mod tests;
