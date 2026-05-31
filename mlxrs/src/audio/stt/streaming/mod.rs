//! Streaming speech-to-text — incremental encoder + orchestration.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/`][swift-dir]:
//!
//! - [`crate::audio::stt::streaming::mel_spectrogram::IncrementalMelSpectrogram`] —
//!   streaming mel-spec with overlap-save framing, mirrors
//!   `IncrementalMelSpectrogram.swift`.
//! - [`crate::audio::stt::streaming::encoder::StreamingEncoderBackend`]
//!   trait +
//!   [`crate::audio::stt::streaming::encoder::StreamingEncoder`] window
//!   accumulator — mirrors `StreamingEncoder.swift`.
//! - [`crate::audio::stt::streaming::session::StreamingInferenceSession`]
//!   + [`crate::audio::stt::streaming::session::StreamingDecoderBackend`]
//!   trait — orchestration, mirrors `StreamingInferenceSession.swift`.
//! - [`crate::audio::stt::streaming::types::DelayPreset`],
//!   [`crate::audio::stt::streaming::types::StreamingConfig`],
//!   [`crate::audio::stt::streaming::types::TranscriptionEvent`], and
//!   [`crate::audio::stt::streaming::types::StreamingStats`] —
//!   value-types, mirror `StreamingTypes.swift`.
//!
//! Async-stream → sync-batch shape:
//!
//! The Swift reference's
//! [`StreamingInferenceSession`][swift-session] yields
//! [`crate::audio::stt::streaming::types::TranscriptionEvent`] values
//! into an `AsyncStream<TranscriptionEvent>` and runs the decode pass
//! on a detached `Task`. mlxrs's port runs the decode pass synchronously
//! on the caller's thread and returns a `Vec<TranscriptionEvent>` from
//! each [`crate::audio::stt::streaming::session::StreamingInferenceSession::feed_audio`]
//! / [`crate::audio::stt::streaming::session::StreamingInferenceSession::stop`]
//! call instead. The event sequence is byte-identical; only the
//! delivery channel differs.
//!
//! [swift-session]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingInferenceSession.swift
//!
//! mlxrs ships **no** concrete encoder / decoder implementations:
//! per-architecture encoders / decoders implement the streaming traits
//! and pass themselves into the streaming session.
//!
//! [swift-dir]: https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioSTT/Streaming

pub mod encoder;
pub mod mel_spectrogram;
mod retry_state;
pub mod session;
pub mod types;

pub use encoder::{StreamingEncoder, StreamingEncoderBackend};
pub use mel_spectrogram::IncrementalMelSpectrogram;
pub use session::{StreamingDecoderBackend, StreamingInferenceSession, StreamingTokenizer};
pub use types::{DelayPreset, StreamingConfig, StreamingStats, TranscriptionEvent};
