//! Speech-to-speech (STS) — the architecture-agnostic STS seam, ported
//! from [`mlx_audio.sts`][sts-init] (the per-domain `load` /
//! `load_model` entry points the upstream `__init__` re-exports).
//!
//! Per the project's [no per-model arch porting][noarch] rule, mlxrs
//! ships **no** concrete STS model implementations: the LFM-audio
//! end-to-end speech model, the SAM-audio source-separator,
//! DeepFilterNet, MossFormer2, Moshi — all per-model and excluded. The
//! single submodule here is the shared support surface every
//! per-architecture STS reuses:
//!
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points that route through the shared
//!   [`crate::audio::load::base_load_model`] factory, plus the
//!   [`StsModel`] trait every concrete STS architecture implements.
//!
//! mlx-audio's [`sts/voice_pipeline.py`][sts-pipeline] composes a full
//! voice-pipeline (VAD + STT + LLM + TTS); per the
//! [mirror-reference-structure][mirror] rule, mlxrs keeps STS as the
//! per-domain `load` surface and treats `VoicePipeline` as a separate
//! caller-level composition (it consumes the [`crate::audio::vad`] +
//! [`crate::audio::stt`] + [`crate::audio::tts`] surfaces directly; STS
//! itself is the surface for end-to-end speech-to-speech models, exactly
//! the upstream `sts.utils.load` shape).
//!
//! [sts-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py
//! [sts-pipeline]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
//! [mirror]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/mirror-reference-structure.md

pub mod load;

pub use load::{StsModel, load, load_model};
