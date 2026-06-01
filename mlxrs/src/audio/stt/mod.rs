//! Speech-to-text (STT) — the architecture-agnostic trait seam.
//!
//! [`model`] defines the three-layer trait architecture every concrete STT
//! model conforms to: the universal [`Transcribe`](model::Transcribe)
//! contract, and the two family traits [`CtcModel`](model::CtcModel)
//! (non-autoregressive, e.g. wav2vec2) and
//! [`AutoregressiveStt`](model::AutoregressiveStt) (encoder/decoder, e.g.
//! whisper). [`generate`] holds the shared decoding drivers, each a free
//! function a model calls from its own [`Transcribe`](model::Transcribe) impl:
//! [`greedy_ctc_transcribe`](generate::greedy_ctc_transcribe) (greedy CTC
//! collapse) and [`greedy_transcribe`](generate::greedy_transcribe) (the
//! generic autoregressive greedy loop).
//!
//! Per the project's no-per-model-arch rule, mlxrs ships **no** concrete STT
//! model implementations: those (the conv subsampling + transformer for
//! whisper, the conformer for parakeet, etc.) live in user code on top of
//! these traits. The submodules here are the shared support surface every
//! per-model STT decoder reuses.
//!
//! ## Pipeline
//!
//! 1. [`crate::audio::io::load_audio`] — mono `Vec<f32>` waveform in `[-1, 1]`,
//!    built into an [`crate::array::Array`].
//! 2. Optional [`generate::resample_waveform`] when the source rate differs
//!    from the model's [`model::MelConfig::sample_rate`].
//! 3. The model's frontend ([`model::AutoregressiveStt::log_mel`] /
//!    [`model::CtcModel::logits`]) → features.
//! 4. The family driver: per-frame greedy collapse (CTC) or the token-by-token
//!    greedy loop ([`generate::greedy_transcribe`]) for a simple autoregressive
//!    model — a complex model (whisper) runs its own
//!    [`Transcribe`](model::Transcribe) procedure reusing the same hooks.

pub mod generate;
pub mod load;
pub mod model;
/// Concrete STT model implementations (feature-gated per architecture).
///
/// Unlike the architecture-agnostic [`model::Model`] trait — which user code
/// implements for autoregressive cross-attention / joint decoders (whisper,
/// parakeet, …) — the models here are the small number of non-AR / CTC
/// architectures mlxrs ports directly because they do not fit that trait's
/// `encode_audio` + per-token `decode_step` + KV-cache shape. Each is behind
/// its own cargo feature.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub mod models;
pub mod serializers;
/// Streaming STT — incremental encoder + orchestration. Ports
/// `mlx-audio-swift/Sources/MLXAudioSTT/Streaming/`.
pub mod streaming;
