//! [`AudioOutputStream`] — the producer-side trait the A8
//! speech-to-speech pipeline (`crate::audio::sts`) consumes to push
//! decoded PCM samples at an audio sink.
//!
//! Mirrors the role `AVAudioPlayerNode.scheduleBuffer` plays for
//! Swift's `MLXAudioCore.AudioPlayer.scheduleAudioChunk(_:withCrossfade:)`:
//! the high-level pipeline doesn't care whether the bytes end up on a
//! real device, a file, or a unit-test recorder — it only needs a
//! `write_samples(&[f32])` hook.
//!
//! The trait is intentionally narrow:
//! - [`write_samples`][AudioOutputStream::write_samples] enqueues
//!   PCM samples (returns how many it accepted; an `Err` signals a
//!   full / closed sink),
//! - [`flush`][AudioOutputStream::flush] signals the producer is done
//!   and blocks until the sink has drained,
//! - [`stop`][AudioOutputStream::stop] aborts immediately, dropping
//!   any queued samples (the cpal-equivalent of
//!   `AVAudioPlayerNode.stop()`),
//! - [`is_running`][AudioOutputStream::is_running] is the single
//!   state-introspection hook (mirrors Swift's `isPlaying`).
//!
//! [`super::player::AudioPlayer`] is the default device-backed
//! implementor; A8's pipeline tests can supply a mock via the same
//! trait without pulling in cpal.

use crate::error::Result;

/// A sink that accepts streamed PCM audio frames.
///
/// Implementors are responsible for whatever buffering / format
/// conversion / device handoff they need. The trait contract is the
/// minimum surface the upstream speech-to-speech pipeline
/// ([`crate::audio::sts`]) needs to push decoded PCM at an output:
///
/// - **Sample layout.** `samples` is interleaved PCM at the
///   implementor's negotiated channel count. For a mono stream that
///   means `samples` is a flat `[f32]`; for stereo it's
///   `[L0, R0, L1, R1, …]`. One *frame* = one `channels` group.
/// - **Backpressure.** [`write_samples`] returns `Ok(n)` where `n`
///   may be less than `samples.len()` if the sink could only accept a
///   prefix (the caller is responsible for retrying the remainder).
///   `Err(_)` means the sink rejected the write outright (queue
///   overflow, sink closed, device error).
/// - **Drop semantics.** Dropping an [`AudioOutputStream`] should
///   stop any in-flight playback and release the underlying
///   resources (cpal stream, mutex, etc.). The
///   [`super::player::AudioPlayer`] impl does this via its `Drop`
///   impl.
///
/// `Send` is required so the trait can cross thread boundaries (the
/// A8 pipeline runs the decoder on a worker thread and pushes
/// samples to the sink without dragging it back to the orchestrator
/// thread). `Sync` is *not* required — write paths are inherently
/// single-producer.
///
/// [`write_samples`]: AudioOutputStream::write_samples
pub trait AudioOutputStream: Send {
  /// Enqueue interleaved PCM samples. Returns the number of samples
  /// accepted (`<= samples.len()`).
  ///
  /// # Errors
  /// - [`crate::error::Error::Backend`] if the sink is full, closed,
  ///   or the underlying device errored. Callers that get a partial
  ///   accept (`Ok(n)` with `n < samples.len()`) should retry the
  ///   remainder; callers that get `Err` must not retry — the sink
  ///   has rejected the write.
  fn write_samples(&mut self, samples: &[f32]) -> Result<usize>;

  /// Signal the producer is done. The implementor MAY block until
  /// queued samples have been consumed by the underlying device /
  /// sink; the caller MUST treat `flush` as a synchronous drain
  /// barrier (the cpal-equivalent of Swift's
  /// `finishStreamingInput()` → `finishStreamIfDrained()` path).
  ///
  /// # Errors
  /// - [`crate::error::Error::Backend`] if the underlying sink
  ///   errored mid-drain (e.g. the device disconnected).
  fn flush(&mut self) -> Result<()>;

  /// Stop the sink immediately. Any queued samples MUST be dropped;
  /// subsequent [`write_samples`][Self::write_samples] calls MUST
  /// return `Err` (this is the **terminal-state** contract — once
  /// `stop()` returns, the sink rejects further writes until the
  /// caller drops the implementor and constructs a fresh one).
  /// Mirrors Swift's
  /// `MLXAudioCore.AudioPlayer.stopStreaming()`.
  ///
  /// **One-way transition.** `stop()` is a one-way latch on the
  /// implementor: any restart-style call on the same
  /// [`AudioOutputStream`] (e.g. a sink-specific `start()` /
  /// `resume()` / a second `stop()`) MUST NOT re-arm the producer
  /// surface. The caller MUST drop the implementor and construct a
  /// fresh one to resume — this is the contract that lets the A8
  /// pipeline treat `stop()` as a hard end-of-stream marker without
  /// auditing the implementor's internal state-machine on every
  /// transition. The [`super::player::AudioPlayer`] impl enforces
  /// this with a dedicated `SharedState::terminated` atomic flag
  /// checked BEFORE the playback-state tri-state on every producer
  /// method (`start`, `pause`, `resume`, `write_samples`).
  ///
  /// Distinct from a pause-style suspension: a pausable sink (e.g.
  /// [`super::player::AudioPlayer::pause`]) buffers writes for later
  /// resume; `stop()` does not. Implementors MUST NOT silently
  /// accept post-stop writes (a post-stop write that "succeeded"
  /// would surprise-replay on a later restart and violate the
  /// "dropped" contract on queued samples).
  ///
  /// # Errors
  /// - [`crate::error::Error::Backend`] if stopping the underlying
  ///   device failed (cpal's `Stream::pause()` returning an error,
  ///   for example).
  fn stop(&mut self) -> Result<()>;

  /// `true` if the sink is currently accepting samples (the
  /// cpal-equivalent of Swift's `isPlaying` for streaming mode).
  fn is_running(&self) -> bool;
}

/// Blanket impl forwarding [`AudioOutputStream`] through a mutable
/// reference — lets callers pass `&mut sink` to APIs that accept a
/// `S: AudioOutputStream` by value, retaining ownership for
/// post-call inspection (the [`VoicePipeline::run`] call site this
/// blanket enables).
///
/// The forwarding is a flat delegation; no buffering is added.
///
/// [`VoicePipeline::run`]: crate::audio::sts::pipeline::VoicePipeline::run
impl<T: AudioOutputStream + ?Sized> AudioOutputStream for &mut T {
  fn write_samples(&mut self, samples: &[f32]) -> Result<usize> {
    (**self).write_samples(samples)
  }

  fn flush(&mut self) -> Result<()> {
    (**self).flush()
  }

  fn stop(&mut self) -> Result<()> {
    (**self).stop()
  }

  fn is_running(&self) -> bool {
    (**self).is_running()
  }
}
