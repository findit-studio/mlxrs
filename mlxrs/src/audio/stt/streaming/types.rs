//! Supporting types for [`crate::audio::stt::streaming`] — `DelayPreset`,
//! `StreamingConfig`, `TranscriptionEvent`, `StreamingStats`.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/StreamingTypes.swift`][swift-ref]
//! (the orchestration-config + event-stream value-types — no per-model
//! wiring lives here).
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingTypes.swift

use derive_more::{IsVariant, TryUnwrap, Unwrap};

// --- Delay presets ------------------------------------------------------

/// Controls the tradeoff between latency and accuracy for streaming
/// transcription. Each variant resolves to a delay in **milliseconds** —
/// the minimum time a provisional token must survive before it can be
/// promoted to confirmed text.
///
/// Mirrors the Swift `DelayPreset` enum 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant)]
pub enum DelayPreset {
  /// ~200 ms delay — fastest feedback, may have more provisional
  /// corrections.
  Realtime,
  /// ~480 ms delay — balanced for voice-agent use cases.
  Agent,
  /// ~2400 ms delay — higher accuracy, suitable for subtitles.
  Subtitle,
  /// Custom delay in milliseconds.
  Custom(u32),
}

impl DelayPreset {
  /// Resolve this preset to its delay in milliseconds.
  pub const fn delay_ms(self) -> u32 {
    match self {
      DelayPreset::Realtime => 200,
      DelayPreset::Agent => 480,
      DelayPreset::Subtitle => 2400,
      DelayPreset::Custom(ms) => ms,
    }
  }

  /// Return a string label for the preset (non-`const` due to the
  /// `Custom` arm carrying a payload). Returns `"realtime"`, `"agent"`,
  /// `"subtitle"`, or `"custom"`.
  pub fn as_str(&self) -> &str {
    match self {
      DelayPreset::Realtime => "realtime",
      DelayPreset::Agent => "agent",
      DelayPreset::Subtitle => "subtitle",
      DelayPreset::Custom(_) => "custom",
    }
  }
}

impl Default for DelayPreset {
  /// `Agent` — the same default the Swift reference uses for
  /// [`StreamingConfig`].
  fn default() -> Self {
    DelayPreset::Agent
  }
}

// --- Streaming config ---------------------------------------------------

/// Configuration for a streaming inference session.
///
/// 1:1 with the Swift `StreamingConfig` struct — every field name and
/// default mirrors the reference. The defaults match the reference's
/// `init(...)` default-argument values exactly:
/// `decodeIntervalSeconds = 1.0`, `boundaryDecodeIntervalSeconds = 0.2`,
/// `boundaryBoostSeconds = 1.0`, `encoderWindowOverlapSeconds = 1.0`,
/// `maxCachedWindows = 60`, `delayPreset = .agent`, `language =
/// "English"`, `temperature = 0.0`, `maxTokensPerPass = 512`,
/// `minAgreementPasses = 2`, `boundaryMinAgreementPasses = 3`,
/// `maxDecodeWindows = 1`, `finalizeCompletedWindows = true`.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
  /// How often to run decode passes, in seconds.
  decode_interval_seconds: f64,
  /// Faster decode interval used briefly after an 8 s window boundary,
  /// in seconds.
  boundary_decode_interval_seconds: f64,
  /// Duration to keep the boundary fast cadence active, in seconds.
  boundary_boost_seconds: f64,
  /// Overlap duration between consecutive 8 s encoder windows, in
  /// seconds.
  encoder_window_overlap_seconds: f64,
  /// Maximum number of cached encoder windows (~8 s each).
  max_cached_windows: usize,
  /// Delay preset controlling the provisional → confirmed promotion.
  delay_preset: DelayPreset,
  /// Language for transcription.
  language: String,
  /// Sampling temperature (`0.0` = greedy).
  temperature: f32,
  /// Maximum tokens per decode pass.
  max_tokens_per_pass: usize,
  /// Minimum consecutive matching passes before a provisional token can
  /// promote.
  min_agreement_passes: usize,
  /// Stronger agreement threshold while the boundary boost is active.
  boundary_min_agreement_passes: usize,
  /// Maximum encoder windows visible to the decoder per pass.
  max_decode_windows: usize,
  /// Whether to run a one-shot decode on each completed 8 s window for
  /// accuracy.
  finalize_completed_windows: bool,
}

impl StreamingConfig {
  /// How often to run decode passes, in seconds.
  #[inline(always)]
  pub fn decode_interval_seconds(&self) -> f64 {
    self.decode_interval_seconds
  }

  /// Faster decode interval used briefly after an 8 s window boundary.
  #[inline(always)]
  pub fn boundary_decode_interval_seconds(&self) -> f64 {
    self.boundary_decode_interval_seconds
  }

  /// Duration to keep the boundary fast cadence active, in seconds.
  #[inline(always)]
  pub fn boundary_boost_seconds(&self) -> f64 {
    self.boundary_boost_seconds
  }

  /// Overlap duration between consecutive encoder windows, in seconds.
  #[inline(always)]
  pub fn encoder_window_overlap_seconds(&self) -> f64 {
    self.encoder_window_overlap_seconds
  }

  /// Maximum number of cached encoder windows.
  #[inline(always)]
  pub fn max_cached_windows(&self) -> usize {
    self.max_cached_windows
  }

  /// Delay preset controlling provisional → confirmed promotion.
  #[inline(always)]
  pub fn delay_preset(&self) -> DelayPreset {
    self.delay_preset
  }

  /// Language for transcription.
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }

  /// Sampling temperature.
  #[inline(always)]
  pub fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Maximum tokens per decode pass.
  #[inline(always)]
  pub fn max_tokens_per_pass(&self) -> usize {
    self.max_tokens_per_pass
  }

  /// Minimum consecutive matching passes before a provisional token can promote.
  #[inline(always)]
  pub fn min_agreement_passes(&self) -> usize {
    self.min_agreement_passes
  }

  /// Stronger agreement threshold while the boundary boost is active.
  #[inline(always)]
  pub fn boundary_min_agreement_passes(&self) -> usize {
    self.boundary_min_agreement_passes
  }

  /// Maximum encoder windows visible to the decoder per pass.
  #[inline(always)]
  pub fn max_decode_windows(&self) -> usize {
    self.max_decode_windows
  }

  /// Whether to run a one-shot decode on each completed 8 s window.
  #[inline(always)]
  pub fn finalize_completed_windows(&self) -> bool {
    self.finalize_completed_windows
  }

  /// Return `self` with `decode_interval_seconds` replaced.
  pub fn with_decode_interval_seconds(self, v: f64) -> Self {
    Self {
      decode_interval_seconds: v,
      ..self
    }
  }

  /// Return `self` with `boundary_decode_interval_seconds` replaced.
  pub fn with_boundary_decode_interval_seconds(self, v: f64) -> Self {
    Self {
      boundary_decode_interval_seconds: v,
      ..self
    }
  }

  /// Return `self` with `boundary_boost_seconds` replaced.
  pub fn with_boundary_boost_seconds(self, v: f64) -> Self {
    Self {
      boundary_boost_seconds: v,
      ..self
    }
  }

  /// Return `self` with `encoder_window_overlap_seconds` replaced.
  pub fn with_encoder_window_overlap_seconds(self, v: f64) -> Self {
    Self {
      encoder_window_overlap_seconds: v,
      ..self
    }
  }

  /// Return `self` with `max_cached_windows` replaced.
  pub fn with_max_cached_windows(self, v: usize) -> Self {
    Self {
      max_cached_windows: v,
      ..self
    }
  }

  /// Return `self` with `delay_preset` replaced.
  pub fn with_delay_preset(self, v: DelayPreset) -> Self {
    Self {
      delay_preset: v,
      ..self
    }
  }

  /// Return `self` with `language` replaced.
  pub fn with_language(self, v: impl Into<String>) -> Self {
    Self {
      language: v.into(),
      ..self
    }
  }

  /// Return `self` with `temperature` replaced.
  pub fn with_temperature(self, v: f32) -> Self {
    Self {
      temperature: v,
      ..self
    }
  }

  /// Return `self` with `max_tokens_per_pass` replaced.
  pub fn with_max_tokens_per_pass(self, v: usize) -> Self {
    Self {
      max_tokens_per_pass: v,
      ..self
    }
  }

  /// Return `self` with `min_agreement_passes` replaced.
  pub fn with_min_agreement_passes(self, v: usize) -> Self {
    Self {
      min_agreement_passes: v,
      ..self
    }
  }

  /// Return `self` with `boundary_min_agreement_passes` replaced.
  pub fn with_boundary_min_agreement_passes(self, v: usize) -> Self {
    Self {
      boundary_min_agreement_passes: v,
      ..self
    }
  }

  /// Return `self` with `max_decode_windows` replaced.
  pub fn with_max_decode_windows(self, v: usize) -> Self {
    Self {
      max_decode_windows: v,
      ..self
    }
  }

  /// Return `self` with `finalize_completed_windows` replaced.
  pub fn with_finalize_completed_windows(self, v: bool) -> Self {
    Self {
      finalize_completed_windows: v,
      ..self
    }
  }
}

impl Default for StreamingConfig {
  fn default() -> Self {
    Self {
      decode_interval_seconds: 1.0,
      boundary_decode_interval_seconds: 0.2,
      boundary_boost_seconds: 1.0,
      encoder_window_overlap_seconds: 1.0,
      max_cached_windows: 60,
      delay_preset: DelayPreset::Agent,
      language: "English".to_string(),
      temperature: 0.0,
      max_tokens_per_pass: 512,
      min_agreement_passes: 2,
      boundary_min_agreement_passes: 3,
      max_decode_windows: 1,
      finalize_completed_windows: true,
    }
  }
}

// --- Transcription event payloads ---------------------------------------

/// Payload for [`TranscriptionEvent::DisplayUpdate`].
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayUpdatePayload {
  /// The latest confirmed transcription text.
  confirmed: String,
  /// The latest provisional (unconfirmed) tail.
  provisional: String,
}

impl DisplayUpdatePayload {
  /// Construct a [`DisplayUpdatePayload`].
  pub fn new(confirmed: impl Into<String>, provisional: impl Into<String>) -> Self {
    Self {
      confirmed: confirmed.into(),
      provisional: provisional.into(),
    }
  }

  /// The latest confirmed transcription text.
  #[inline(always)]
  pub fn confirmed(&self) -> &str {
    &self.confirmed
  }

  /// The latest provisional (unconfirmed) tail.
  #[inline(always)]
  pub fn provisional(&self) -> &str {
    &self.provisional
  }
}

// --- Transcription events ----------------------------------------------

/// Events emitted by a streaming inference session
/// (`super::session::StreamingInferenceSession`, added in a follow-up
/// commit).
///
/// Mirrors Swift's `TranscriptionEvent` enum — the unit of output the
/// session produces. The Swift reference yields these into an
/// `AsyncStream<TranscriptionEvent>`; mlxrs's synchronous Rust API
/// returns batches (`Vec<TranscriptionEvent>`) from the session's
/// `feed_audio` / `stop` calls instead, matching the project's
/// foreground-only execution model.
#[derive(Debug, Clone, PartialEq, IsVariant, Unwrap, TryUnwrap)]
#[unwrap(ref, ref_mut)]
pub enum TranscriptionEvent {
  /// Provisional text that may still change.
  Provisional(String),
  /// Text that has been confirmed and will not change.
  Confirmed(String),
  /// Combined display update with both confirmed and provisional text.
  DisplayUpdate(DisplayUpdatePayload),
  /// Performance statistics.
  Stats(StreamingStats),
  /// Session has ended with the final full text.
  Ended(String),
}

impl TranscriptionEvent {
  /// Construct a [`TranscriptionEvent::Provisional`] event.
  pub fn provisional(text: impl Into<String>) -> Self {
    TranscriptionEvent::Provisional(text.into())
  }

  /// Construct a [`TranscriptionEvent::Confirmed`] event.
  pub fn confirmed(text: impl Into<String>) -> Self {
    TranscriptionEvent::Confirmed(text.into())
  }

  /// Construct a [`TranscriptionEvent::DisplayUpdate`] event.
  pub fn display_update(confirmed: impl Into<String>, provisional: impl Into<String>) -> Self {
    TranscriptionEvent::DisplayUpdate(DisplayUpdatePayload::new(confirmed, provisional))
  }

  /// Construct a [`TranscriptionEvent::Ended`] event.
  pub fn ended(full_text: impl Into<String>) -> Self {
    TranscriptionEvent::Ended(full_text.into())
  }
}

// --- Stats --------------------------------------------------------------

/// Performance statistics for a streaming session.
///
/// 1:1 with Swift's `StreamingStats`. `peak_memory_gb` reports the
/// process-global mlx allocator peak — see
/// [`crate::memory::peak_memory`]. The Swift reference also tracks a
/// `realTimeFactor` field that it always sets to `0`; mlxrs carries the
/// same field for shape parity.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamingStats {
  /// Number of encoder windows processed.
  pub encoded_window_count: usize,
  /// Total audio duration processed so far, in seconds.
  pub total_audio_seconds: f64,
  /// Tokens generated per second.
  pub tokens_per_second: f64,
  /// Real-time factor (`< 1.0` means faster than real-time). Carried
  /// for parity with the Swift reference; not currently populated.
  pub real_time_factor: f64,
  /// Peak memory usage in GB (process-global mlx allocator).
  pub peak_memory_gb: f64,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn delay_preset_resolves_to_documented_millisecond_values() {
    assert_eq!(DelayPreset::Realtime.delay_ms(), 200);
    assert_eq!(DelayPreset::Agent.delay_ms(), 480);
    assert_eq!(DelayPreset::Subtitle.delay_ms(), 2400);
    assert_eq!(DelayPreset::Custom(750).delay_ms(), 750);
  }

  #[test]
  fn delay_preset_default_is_agent() {
    assert_eq!(DelayPreset::default(), DelayPreset::Agent);
  }

  #[test]
  fn delay_preset_as_str() {
    assert_eq!(DelayPreset::Realtime.as_str(), "realtime");
    assert_eq!(DelayPreset::Agent.as_str(), "agent");
    assert_eq!(DelayPreset::Subtitle.as_str(), "subtitle");
    assert_eq!(DelayPreset::Custom(100).as_str(), "custom");
  }

  #[test]
  fn delay_preset_is_variant() {
    assert!(DelayPreset::Realtime.is_realtime());
    assert!(DelayPreset::Agent.is_agent());
    assert!(DelayPreset::Subtitle.is_subtitle());
    assert!(DelayPreset::Custom(0).is_custom());
  }

  #[test]
  fn streaming_config_default_matches_swift_reference_values() {
    let c = StreamingConfig::default();
    assert_eq!(c.decode_interval_seconds(), 1.0);
    assert_eq!(c.boundary_decode_interval_seconds(), 0.2);
    assert_eq!(c.boundary_boost_seconds(), 1.0);
    assert_eq!(c.encoder_window_overlap_seconds(), 1.0);
    assert_eq!(c.max_cached_windows(), 60);
    assert_eq!(c.delay_preset(), DelayPreset::Agent);
    assert_eq!(c.language(), "English");
    assert_eq!(c.temperature(), 0.0);
    assert_eq!(c.max_tokens_per_pass(), 512);
    assert_eq!(c.min_agreement_passes(), 2);
    assert_eq!(c.boundary_min_agreement_passes(), 3);
    assert_eq!(c.max_decode_windows(), 1);
    assert!(c.finalize_completed_windows());
  }

  #[test]
  fn transcription_event_constructors() {
    let p = TranscriptionEvent::provisional("hello");
    assert!(p.is_provisional());
    assert_eq!(p.unwrap_provisional(), "hello");

    let c = TranscriptionEvent::confirmed("world");
    assert!(c.is_confirmed());
    assert_eq!(c.unwrap_confirmed(), "world");

    let d = TranscriptionEvent::display_update("conf", "prov");
    assert!(d.is_display_update());
    let du = d.unwrap_display_update();
    assert_eq!(du.confirmed(), "conf");
    assert_eq!(du.provisional(), "prov");

    let e = TranscriptionEvent::ended("final text");
    assert!(e.is_ended());
    assert_eq!(e.unwrap_ended(), "final text");
  }
}
