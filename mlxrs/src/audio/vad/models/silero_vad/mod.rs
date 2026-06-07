//! Silero VAD (`snakers4/silero-vad`) — a faithful port of mlx-audio's
//! `silero_vad` ([`vad/models/silero_vad/`][silero]).
//!
//! Silero is a lightweight per-frame voice activity detector: a learned
//! STFT-conv filterbank feeds four convolutional blocks, a single-layer LSTM,
//! and a `sigmoid` speech-probability head, producing one probability per
//! fixed-size audio window (512 samples at 16 kHz, 256 at 8 kHz). A whole
//! waveform is processed as a stream of those windows (each carrying a small
//! left context through the LSTM recurrent state), and the resulting per-frame
//! probabilities are collapsed to speech-segment timestamps by a hysteresis
//! state machine.
//!
//! ## Pipeline
//!
//! 1. **Config** ([`config`]) — the two per-rate `BranchConfig`s and the
//!    post-processing hyper-parameters (`threshold`, min-speech / min-silence,
//!    speech-pad), parsed from `config.json` with the reference's branch
//!    defaulting.
//! 2. **Model graph** ([`model`]) — `SileroVadBranch` (reflect-pad → STFT-conv
//!    → magnitude → 4 conv blocks → LSTM → `sigmoid` head), the recurrent
//!    chunking / streaming `feed`, and the `probs_to_timestamps` segment
//!    extractor — all faithful to `silero_vad.py`.
//! 3. **Loader** ([`loader`]) — `sanitize` (drop `val_*`), the weight-map →
//!    branch assembly, the safetensors shard walk, and the
//!    [`loader::load`] → [`crate::audio::vad::VadModel`] factory the VAD
//!    registry dispatches `model_type == "silero_vad"` to.
//!
//! Every piece is exercised by shape + closed-form + synthetic-checkpoint
//! oracle tests (mirroring `tests/test_silero_vad.py`); the gated e2e
//! numeric-parity test against a real Silero checkpoint is a separate change.
//!
//! [silero]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py

pub mod config;
pub mod loader;
pub mod model;

pub use config::{BranchConfig, MODEL_TYPE, ModelConfig};
pub use loader::{has_relevant_scales, load};
pub use model::{
  SileroVadBranch, SileroVadModel, SileroVadState, SpeechTimestampOptions, probs_to_timestamps,
  sanitize,
};

#[cfg(test)]
mod tests;
