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
/// Hosts both family shapes the trait architecture spans: the CTC / non-AR
/// architectures (wav2vec2), which fit the [`CtcModel`](model::CtcModel)
/// family (or expose an inherent CTC API), and the autoregressive
/// encoder/decoder ones (whisper), which implement
/// [`AutoregressiveStt`](model::AutoregressiveStt) (`encode` + per-token
/// `decode_step` + KV cache) and run their own
/// [`Transcribe`](model::Transcribe) procedure. Each is behind its own cargo
/// feature.
#[cfg(any(feature = "wav2vec2", feature = "whisper"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "wav2vec2", feature = "whisper"))))]
pub mod models;
pub mod serializers;
/// Streaming STT — incremental encoder + orchestration. Ports
/// `mlx-audio-swift/Sources/MLXAudioSTT/Streaming/`.
pub mod streaming;
