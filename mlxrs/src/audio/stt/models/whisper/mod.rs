//! OpenAI Whisper (speech-to-text).
//!
//! Faithful port of `mlx_audio.stt.models.whisper`
//! (`mlx-source/mlx-audio/mlx_audio/stt/models/whisper/`): the audio
//! front-end, the encoder/decoder transformer, the GPT-2 byte-level BPE
//! tokenizer wrapper, and the decoding task.
//!
//! Gated behind the `whisper` feature (`= ["audio", "tokenizer-bpe"]`).
//!
//! ## Layout
//! - [`audio`] — hyperparameters + `log_mel_spectrogram_whisper` (the
//!   Slaney mel + Whisper log-mel post) + `pad_or_trim`.
//! - [`config`] — [`config::ModelDimensions`] (`from_dict` for both the MLX
//!   and HuggingFace config layouts + eager `validate`).
//! - `layers` (crate-private) — the `Linear` / `Embedding` building blocks,
//!   the Whisper-variant `MultiHeadAttention`, the `ResidualAttentionBlock`,
//!   and `sinusoids`.
//! - `encoder` (crate-private) — the `AudioEncoder` (conv front-end +
//!   self-attention blocks).
//! - `decoder` (crate-private) — the `TextDecoder` (token + learned positional
//!   embedding, cross-attention blocks, weight-tied logits).
//! - [`tokenizer`] — the [`tokenizer::HFTokenizerWrapper`] over the crate's
//!   [`crate::tokenizer::Tokenizer`] + the language tables.
//! - [`model`] — [`model::WhisperModel`] (ties the encoder + decoder + dims,
//!   implements the
//!   [`AutoregressiveStt`](crate::audio::stt::model::AutoregressiveStt) family
//!   trait + the universal
//!   [`Transcribe`](crate::audio::stt::model::Transcribe) contract, sanitizes +
//!   loads weights).
//! - [`decoding`] — the full [`decoding::DecodingTask`] (greedy decode +
//!   the three logit filters + temperature fallback + the 30-second seek loop
//!   + language detection).

pub mod audio;
pub mod config;
pub(crate) mod decoder;
pub mod decoding;
pub(crate) mod encoder;
pub(crate) mod layers;
pub mod model;
pub mod tokenizer;
