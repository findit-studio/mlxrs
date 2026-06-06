//! Concrete voice-activity-detection model implementations.
//!
//! - `silero_vad` — Silero VAD (`snakers4/silero-vad`): a learned STFT-conv
//!   filterbank + four conv blocks + a single-layer LSTM + a `sigmoid`
//!   speech-probability head, run as a stream of fixed-size windows (512
//!   samples at 16 kHz, 256 at 8 kHz) and collapsed to speech-segment
//!   timestamps by a hysteresis state machine. Implements the shared
//!   [`crate::audio::vad::VadModel`] trait.

#[cfg(feature = "vad")]
#[cfg_attr(docsrs, doc(cfg(feature = "vad")))]
pub mod silero_vad;
