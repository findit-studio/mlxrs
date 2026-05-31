//! Language Identification (LID) — the architecture-agnostic LID seam,
//! ported from [`mlx_audio.lid`][lid-init] (the per-domain `load` /
//! `load_model` entry points + the `predict(audio, top_k=…)` result
//! shape mlx-audio's per-architecture `Model.predict` returns).
//!
//! Per the project's no per-model arch porting rule, mlxrs
//! ships **no** concrete LID model implementations: the
//! wav2vec2 / facebook/mms-lid CTC head, the SpeechBrain-port
//! ECAPA-TDNN classifier — both per-model and excluded. The two
//! submodules here are the shared support surface every per-architecture
//! LID reuses:
//!
//! - [`output`] — the [`LidOutput`] result struct (plus the typed
//!   [`LidPrediction`] entries) the per-architecture
//!   `Model.predict(audio, top_k=…)` returns.
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points that route through the shared
//!   [`crate::audio::load::base_load_model`] factory.
//!
//! [lid-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/lid/utils.py

pub mod load;
pub mod output;

pub use load::{LidModel, load, load_model};
pub use output::{LidOutput, LidPrediction};
