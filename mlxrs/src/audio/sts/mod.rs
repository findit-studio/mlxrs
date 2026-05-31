//! Speech-to-speech (STS) — the architecture-agnostic STS seam, ported
//! from [`mlx_audio.sts`][sts-init] (the per-domain `load` /
//! `load_model` entry points the upstream `__init__` re-exports).
//!
//! Per the project's no per-model arch porting rule, mlxrs
//! ships **no** concrete STS model implementations: the LFM-audio
//! end-to-end speech model, the SAM-audio source-separator,
//! DeepFilterNet, MossFormer2, Moshi — all per-model and excluded. The
//! single submodule here is the shared support surface every
//! per-architecture STS reuses:
//!
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points that route through the shared
//!   [`crate::audio::load::base_load_model`] factory, plus the
//!   [`load::Model`] trait every concrete STS architecture implements.
//!
//! mlx-audio's [`sts/voice_pipeline.py`][sts-pipeline] composes a full
//! voice-pipeline (VAD + STT + LLM + TTS); per the
//! mirror-reference-structure rule, mlxrs keeps STS as the
//! per-domain `load` surface (the [`mod@load`] submodule) and lifts
//! the voice-pipeline composition into a separate [`mod@pipeline`]
//! sibling that consumes the [`crate::audio::vad`] +
//! [`crate::audio::stt`] + [`crate::audio::tts`] +
//! [`crate::audio::playback`] surfaces directly. STS itself stays the
//! surface for end-to-end speech-to-speech models (exactly the
//! upstream `sts.utils.load` shape); the [`mod@pipeline`] sibling is
//! the orchestration shape mlx-audio's `VoicePipeline` class exposes.
//!
//! - [`mod@load`] — the per-domain [`load::load`] / [`load::load_model`]
//!   entry points + the [`load::Model`] trait every concrete STS
//!   architecture implements.
//! - [`mod@pipeline`] — voice-pipeline orchestration (the
//!   [`pipeline::VoiceSession`] default + the [`pipeline::VoicePipeline`]
//!   trait + the chunker / barge-in / turn-taking primitive
//!   submodules).
//!
//! [sts-init]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/utils.py
//! [sts-pipeline]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py

pub mod load;
pub mod pipeline;

pub use load::{Model, load, load_model};
pub use pipeline::{
  AudioChunker, BargeInDetector, EnergyBargeInDetector, FixedSizeAudioChunker, LatencyProfile,
  LlmResponderAdapter, PreRollBuffer, SilenceTurnTakingPolicy, SttTurnAdapter, TtsStreamAdapter,
  TurnEvent, TurnTakingPolicy, VadFrameAdapter, VoicePipeline, VoicePipelineConfig, VoiceSession,
};
