//! Voice-pipeline orchestration — the architecture-agnostic
//! VAD + STT + LLM + TTS + audio-out composition surface, ported from
//! [`mlx_audio.sts.voice_pipeline`][vp].
//!
//! Per the no per-model arch porting rule and the
//! mirror-reference-structure rule, mlxrs lifts the
//! *orchestration shape* mlx-audio's [`VoicePipeline`][vp-class]
//! exposes as a typed trait surface every caller composes:
//!
//! - [`mod@config`] — [`VoicePipelineConfig`] (the typed argument
//!   bundle) + [`LatencyProfile`] (typed analogue of mlx-audio's
//!   `latency_profile: str`).
//! - [`mod@chunker`] — [`AudioChunker`] trait + [`FixedSizeAudioChunker`]
//!   default + [`PreRollBuffer`] (the sample-buffer primitives).
//! - [`mod@barge_in`] — [`BargeInDetector`] trait +
//!   [`EnergyBargeInDetector`] (the user-overlap decision shape).
//! - [`mod@turn_taking`] — [`TurnTakingPolicy`] trait +
//!   [`SilenceTurnTakingPolicy`] (the turn-end decision shape).
//! - [`mod@voice_pipeline`] — [`VoicePipeline`] trait (the high-level
//!   `run(mic_input, audio_out)` shape).
//!
//! - [`mod@orchestrator`] — [`VoiceSession`] (the default
//!   [`VoicePipeline`] implementor that composes every trait
//!   surface together into one synchronous mic-iterator-driven
//!   loop) plus the four per-step adapter traits
//!   [`orchestrator::VadFrameAdapter`] /
//!   [`orchestrator::SttTurnAdapter`] /
//!   [`orchestrator::LlmResponderAdapter`] /
//!   [`orchestrator::TtsStreamAdapter`] the orchestrator needs (a
//!   streaming view over the whole-utterance shapes the existing
//!   [`crate::audio::vad::VadModel`] /
//!   [`crate::audio::stt::model::Model`] /
//!   [`crate::lm::model::Model`] /
//!   [`crate::audio::tts::model::TtsModel`] trait surfaces
//!   expose).
//!
//! ## Scope cuts (explicit)
//!
//! mlx-audio's `voice_pipeline.py` carries 1500+ lines of
//! `asyncio` orchestration, real-time logging, and per-architecture
//! state (Voxtral streaming sessions, smart-turn endpoint
//! detectors). Per the project's match-official-binding-design
//! and no per-model arch porting rules, mlxrs ports only
//! the **shape** of that orchestration as composable traits + a
//! synchronous default `VoiceSession`. Out-of-scope:
//!
//! - **`sounddevice` mic capture** — mic input is an iterator the
//!   caller supplies (a `cpal::Stream` consumer, a file reader, a
//!   unit-test fixture). The orchestrator is sink-agnostic on the
//!   input side too.
//! - **`asyncio` worker / `MLXWorkScheduler`** — the orchestrator
//!   runs synchronously; a caller who needs the async fan-out wires
//!   their own runtime around the per-frame `step` call.
//! - **The per-model `VoxtralRealtimeTranscriber` /
//!   `SileroSpeechGate` / `SmartTurnEndpointDetector` /
//!   `LocalLLMResponseEngine` / `PocketTTSResponder` /
//!   `AudioOutputStream` classes** — these wrap concrete
//!   architectures (Voxtral STT, Silero VAD, etc.). mlxrs's no-arch
//!   rule pushes them to user code; the orchestrator instead consumes
//!   the architecture-agnostic [`crate::audio::stt::model::Model`] /
//!   [`crate::audio::vad::VadModel`] /
//!   [`crate::audio::tts::model::TtsModel`] / [`crate::lm::model::Model`]
//!   / [`crate::audio::playback::AudioOutputStream`] trait surfaces.
//! - **The full barge-in / echo-correlation state machine** — mlxrs
//!   ports the **decision shape** ([`BargeInDetector`]) but not the
//!   full echo-cancellation pipeline mlx-audio's
//!   `_handle_speech_started` / `_maybe_confirm_barge_candidate`
//!   build. The orchestrator wires the trait's `detect` call; the
//!   confirmation / suppression dance lives in user code that wraps
//!   the orchestrator's per-step events.
//!
//! [vp]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py
//! [vp-class]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L570-L572
//! [orch-session]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L570-L572

pub mod barge_in;
pub mod chunker;
pub mod config;
pub mod orchestrator;
pub mod turn_taking;
pub mod voice_pipeline;

pub use barge_in::{BargeInDetector, EnergyBargeInDetector};
pub use chunker::{AudioChunker, FixedSizeAudioChunker, PreRollBuffer};
pub use config::{LatencyProfile, VoicePipelineConfig};
pub use orchestrator::{
  LlmResponderAdapter, SttTurnAdapter, TtsStreamAdapter, TurnEvent, VadFrameAdapter, VoiceSession,
};
pub use turn_taking::{SilenceTurnTakingPolicy, TurnTakingPolicy};
pub use voice_pipeline::VoicePipeline;
