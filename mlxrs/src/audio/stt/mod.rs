//! Speech-to-text (STT) — the architecture-agnostic generation seam.
//!
//! Ports the *shape* of mlx-audio's STT surface
//! ([`stt/models/base.py`][stt-base] + [`stt/generate.py`][stt-gen]) — the
//! [`Model`](model::Model) trait every concrete STT architecture
//! (whisper / parakeet / canary / …) implements, and the
//! [`stt_generate`](generate::stt_generate) Iterator that drives it.
//!
//! Per the project's no per-model arch porting rule, mlxrs ships
//! **no** concrete STT model implementations: those (the conv subsampling +
//! transformer for whisper, the conformer + RNN-T joint for parakeet, etc.)
//! live in user code on top of this trait. The two submodules here are the
//! shared support surface every per-model STT decoder reuses.
//!
//! ## Pipeline
//!
//! 1. [`crate::audio::io::load_audio`] — mono `Vec<f32>` in `[-1, 1]`.
//! 2. Optional [`crate::audio::io::resample_linear`] when the source rate
//!    differs from [`model::Model::mel_config`].
//! 3. [`generate::SttGenConfig::max_audio_seconds`] cap (BEFORE allocation).
//! 4. [`crate::audio::dsp::log_mel_spectrogram`] — `(n_mels, T)`.
//! 5. [`model::Model::encode_audio`] — one pass; states cached on the loop.
//! 6. Token-by-token [`model::Model::decode_step`] (seeded by
//!    [`model::Model::bos_token`]); sampled via the LM loop's
//!    [`crate::lm::generate::make_sampler`] /
//!    [`crate::lm::generate::make_logits_processors`] chain.
//!
//! [stt-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/base.py
//! [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py

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
