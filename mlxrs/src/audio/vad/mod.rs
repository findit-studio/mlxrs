//! Voice Activity Detection / diarization — the architecture-agnostic
//! VAD seam, ported from
//! [`mlx_audio.vad`][vad-init] (the per-domain `load` / `load_model`
//! entry points + the [`VADOutput`][vad-output] result struct mlx-audio's
//! per-architecture `Model.generate` returns).
//!
//! Per the project's no per-model arch porting rule, mlxrs
//! ships **no** concrete VAD model implementations: the silero-vad
//! CNN+LSTM, the sortformer diarizer, the smart-turn endpoint model — all
//! per-model and excluded. The two submodules here are the shared support
//! surface every per-architecture VAD reuses:
//!
//! - [`output`] — the [`VadOutput`] result struct (plus [`SpeechSegment`])
//!   the per-architecture `Model.generate` returns.
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points that route through the shared
//!   [`crate::audio::load::base_load_model`] factory.
//!
//! [vad-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/utils.py
//! [vad-output]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/silero_vad.py#L21-L25

pub mod load;
pub mod output;

pub use load::{VadModel, load, load_model};
pub use output::{SpeechSegment, VadOutput};
