//! [`VoicePipeline`] — the high-level static-dispatch trait
//! "consume a mic input iterator + drive an audio sink" the public
//! voice-loop API exposes.
//!
//! Mirrors the public surface of mlx-audio's
//! [`VoicePipeline.start`][vp-start]: one `await pipeline.start()`
//! call drives the full VAD → STT → LLM → TTS → audio-out loop
//! until the mic stream closes. mlxrs lifts that into a typed
//! trait the caller composes:
//!
//! - [`VoicePipeline::config`] — the typed config bundle (parity
//!   with mlx-audio's `pipeline.config: VoicePipelineConfig`).
//! - [`VoicePipeline::run`] — the end-to-end drive call: takes a
//!   mic-frame iterator (the caller's choice — `cpal::Stream`
//!   consumer, file reader, unit-test fixture) plus an audio sink
//!   (a [`crate::audio::playback::AudioOutputStream`] implementor;
//!   the default device sink is
//!   [`crate::audio::playback::AudioPlayer`]) and runs the loop
//!   to mic-EOF.
//!
//! Static dispatch (`impl Iterator` + `S: AudioOutputStream`)
//! rather than `dyn` because (a) mlxrs's audio sinks are always
//! known at compile time (a unit test uses a recorder; production
//! code uses an `AudioPlayer`), and (b) the per-frame hot path
//! (chunker push + barge-in detect + turn-policy poll) inlines
//! away the trait dispatch — the same shape
//! [`crate::lm::generate`] uses for its sampler / logits-processor
//! boxed-closure surfaces ([`crate::lm::generate::Sampler`] +
//! [`crate::lm::generate::LogitsProcessor`]).
//!
//! The default implementor lives in [`super::orchestrator`]
//! ([`super::orchestrator::VoiceSession`]); a caller who needs a
//! custom run-loop (e.g. one that uses a different async runtime)
//! can implement [`VoicePipeline`] over a hand-rolled state
//! machine. Most callers should use the default `VoiceSession`.
//!
//! [vp-start]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L831-L850

use crate::{audio::playback::AudioOutputStream, error::Result};

use super::config::VoicePipelineConfig;

/// High-level voice-loop trait: consume a mic-frame iterator + drive
/// an audio sink with TTS chunks.
///
/// Static dispatch on the input iterator + audio sink: the caller
/// picks both at compile time. The implementor (the default is
/// [`super::orchestrator::VoiceSession`]) owns the inner
/// VAD / STT / LM / TTS trait objects and drives the loop until
/// `mic_input.next()` returns `None`.
pub trait VoicePipeline {
  /// The typed config bundle this pipeline was built with.
  fn config(&self) -> &VoicePipelineConfig;

  /// Drive the voice loop end-to-end:
  /// 1. Pull a frame from `mic_input`.
  /// 2. Push it through the chunker.
  /// 3. Run VAD over every emitted chunk.
  /// 4. On turn-end, finalize STT, dispatch to the LLM, stream
  ///    TTS into `output`.
  /// 5. Repeat until `mic_input.next()` returns `None`.
  ///
  /// Returns `Ok(())` when the mic iterator is exhausted; an `Err`
  /// is surfaced when any inner trait call (VAD / STT / LM / TTS /
  /// audio-out) fails irrecoverably.
  ///
  /// `mic_input` yields **`Vec<f32>`** frames of arbitrary length
  /// (the orchestrator's chunker handles re-alignment). `output`
  /// receives interleaved PCM at
  /// [`VoicePipelineConfig::output_sample_rate`] (or the TTS
  /// model's [`crate::audio::tts::model::TtsModel::sample_rate`]
  /// when that field is `None`).
  ///
  /// # Errors
  /// - [`crate::error::Error::Backend`] on inner trait errors
  ///   (VAD / STT / LM / TTS / audio-out).
  fn run<I, S>(&mut self, mic_input: I, output: S) -> Result<()>
  where
    I: Iterator<Item = Vec<f32>>,
    S: AudioOutputStream;
}
