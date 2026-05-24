//! Supporting types for [`crate::audio::stt::streaming`] — `DelayPreset`,
//! `StreamingConfig`, `TranscriptionEvent`, `StreamingStats`.
//!
//! Faithful port of
//! [`mlx-audio-swift/Sources/MLXAudioSTT/Streaming/StreamingTypes.swift`][swift-ref]
//! (the orchestration-config + event-stream value-types — no per-model
//! wiring lives here).
//!
//! [swift-ref]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioSTT/Streaming/StreamingTypes.swift

// --- Delay presets ------------------------------------------------------

/// Controls the tradeoff between latency and accuracy for streaming
/// transcription. Each variant resolves to a delay in **milliseconds** —
/// the minimum time a provisional token must survive before it can be
/// promoted to confirmed text.
///
/// Mirrors the Swift `DelayPreset` enum 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
  pub fn delay_ms(self) -> u32 {
    match self {
      DelayPreset::Realtime => 200,
      DelayPreset::Agent => 480,
      DelayPreset::Subtitle => 2400,
      DelayPreset::Custom(ms) => ms,
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
  pub decode_interval_seconds: f64,
  /// Faster decode interval used briefly after an 8 s window boundary,
  /// in seconds.
  pub boundary_decode_interval_seconds: f64,
  /// Duration to keep the boundary fast cadence active, in seconds.
  pub boundary_boost_seconds: f64,
  /// Overlap duration between consecutive 8 s encoder windows, in
  /// seconds.
  pub encoder_window_overlap_seconds: f64,
  /// Maximum number of cached encoder windows (~8 s each).
  pub max_cached_windows: usize,
  /// Delay preset controlling the provisional → confirmed promotion.
  pub delay_preset: DelayPreset,
  /// Language for transcription.
  pub language: String,
  /// Sampling temperature (`0.0` = greedy).
  pub temperature: f32,
  /// Maximum tokens per decode pass.
  pub max_tokens_per_pass: usize,
  /// Minimum consecutive matching passes before a provisional token can
  /// promote.
  pub min_agreement_passes: usize,
  /// Stronger agreement threshold while the boundary boost is active.
  pub boundary_min_agreement_passes: usize,
  /// Maximum encoder windows visible to the decoder per pass.
  pub max_decode_windows: usize,
  /// Whether to run a one-shot decode on each completed 8 s window for
  /// accuracy.
  pub finalize_completed_windows: bool,
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
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionEvent {
  /// Provisional text that may still change.
  Provisional {
    /// The latest in-flight (unconfirmed) transcription text.
    text: String,
  },
  /// Text that has been confirmed and will not change.
  Confirmed {
    /// The latest stable transcription text.
    text: String,
  },
  /// Combined display update with both confirmed and provisional text.
  DisplayUpdate {
    /// The latest confirmed transcription text.
    confirmed_text: String,
    /// The latest provisional (unconfirmed) tail.
    provisional_text: String,
  },
  /// Performance statistics.
  Stats(StreamingStats),
  /// Session has ended with the final full text.
  Ended {
    /// The final, complete transcription.
    full_text: String,
  },
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
  fn streaming_config_default_matches_swift_reference_values() {
    let c = StreamingConfig::default();
    assert_eq!(c.decode_interval_seconds, 1.0);
    assert_eq!(c.boundary_decode_interval_seconds, 0.2);
    assert_eq!(c.boundary_boost_seconds, 1.0);
    assert_eq!(c.encoder_window_overlap_seconds, 1.0);
    assert_eq!(c.max_cached_windows, 60);
    assert_eq!(c.delay_preset, DelayPreset::Agent);
    assert_eq!(c.language, "English");
    assert_eq!(c.temperature, 0.0);
    assert_eq!(c.max_tokens_per_pass, 512);
    assert_eq!(c.min_agreement_passes, 2);
    assert_eq!(c.boundary_min_agreement_passes, 3);
    assert_eq!(c.max_decode_windows, 1);
    assert!(c.finalize_completed_windows);
  }
}
