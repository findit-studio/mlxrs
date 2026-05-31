//! Neural-audio codecs — the architecture-agnostic codec seam.
//!
//! Ports the *shape* of mlx-audio's
//! [`mlx_audio.codec`][codec-init] surface (`DAC`, `Encodec`, `Mimi`,
//! `MossAudioTokenizer`, `Vocos`, `StepAudio2Token2Wav`, etc.) — every
//! codec the upstream package re-exports from its `models/` directory.
//!
//! Per the project's no per-model arch porting rule, mlxrs
//! ships **no** concrete codec model implementations: each codec (DAC,
//! Encodec, Mimi, Vocos, …) is per-architecture and excluded. The
//! single submodule here is the shared support surface every
//! per-architecture codec reuses:
//!
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points that route through the shared
//!   [`crate::audio::load::base_load_model`] factory, plus the
//!   [`CodecModel`] trait every concrete codec implements
//!   (`encode(audio) -> codes` / `decode(codes) -> audio`).
//!
//! mlx-audio's [`codec/__init__.py`][codec-init] is a bare re-export
//! list (no `load` helper at the codec level — each codec class exposes
//! its own `from_pretrained(...)`). mlxrs follows the per-domain
//! pattern the other audio domains use (VAD / LID / STS / STT / TTS):
//! a per-domain `load` entry point that routes through the shared
//! factory, so a downstream caller has a uniform load surface across
//! every audio domain.
//!
//! [codec-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/codec/__init__.py

pub mod load;

pub use load::{CodecModel, load, load_model};
