//! [`BargeInDetector`] + [`EnergyBargeInDetector`] — the
//! "user-just-spoke-while-TTS-was-playing" decision shape the
//! orchestrator queries every input frame.
//!
//! mlx-audio's `VoicePipeline` does **not** ship a standalone barge-in
//! detector class — instead the decision is fused into the main
//! `_handle_speech_started` / `_maybe_confirm_barge_candidate` flow
//! ([`voice_pipeline.py:951-1075`][vp-barge]), which inspects the
//! current playback state, runs an echo-correlation check via the
//! [`AudioOutputStream::echo_correlation`][vp-echo], and waits for a
//! partial transcript before confirming the candidate. Per the
//! mirror-reference-structure rule mlxrs lifts the **decision
//! shape** out as a trait so callers can swap in a custom detector
//! (e.g. one that uses a smarter echo cancellation model) without
//! rewriting [`super::orchestrator::VoiceSession`]; the default
//! [`EnergyBargeInDetector`] is the simplest correct implementation
//! (energy-RMS threshold over the user audio).
//!
//! The detector intentionally **does not** subsume the full mlx-audio
//! barge-in state machine (preroll, candidate confirmation, echo
//! correlation, partial-transcript gate); those belong to the
//! orchestrator and are wired in [`super::orchestrator`]. The detector
//! exists for the orchestrator to ask "given this frame and whether
//! TTS is currently playing, should we **consider** this a barge-in?"
//! — a single boolean the orchestrator then escalates through the
//! confirmation pipeline.
//!
//! [vp-barge]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L951-L1075
//! [vp-echo]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L544-L563

/// The shape the orchestrator queries per input frame to ask:
/// "given this user audio and whether TTS is playing, should this be
/// considered a barge-in candidate?"
///
/// `&self` because the detector should be a pure decision function
/// (every implementor mlx-audio uses is stateless); state that needs
/// to be carried across frames lives in
/// [`super::orchestrator::VoiceSession`] (timestamp of first
/// candidate frame, etc.), not the detector.
pub trait BargeInDetector {
  /// Inspect one frame of user audio + the TTS-playing flag and
  /// return whether the frame should be treated as a barge-in
  /// candidate (the orchestrator then runs the partial-transcript
  /// and min-duration confirmation steps mlx-audio's
  /// `_maybe_confirm_barge_candidate` performs).
  ///
  /// Returns `false` when TTS is not playing — barge-in is by
  /// definition a TTS-overlap event.
  fn detect(&self, user_audio: &[f32], tts_playing: bool) -> bool;
}

/// Energy-RMS based barge-in detector — the default implementor.
///
/// Computes the root-mean-square energy of `user_audio` and returns
/// `true` when it exceeds the configured threshold **and** TTS is
/// currently playing. This is the simplest correct barge-in heuristic
/// (matches the implicit energy gate mlx-audio's
/// `_handle_speech_started` runs implicitly via the Silero VAD's
/// frame-probability output).
///
/// The RMS amplitude threshold is in `[0, 1]`;
/// mlx-audio uses no explicit number (the gate is upstream of the
/// detector, in the VAD), so the default of `0.02` is a reasonable
/// "audible speech" floor — quiet room noise rarely crosses it, and
/// any voiced phone reliably does.
///
/// Construct via [`EnergyBargeInDetector::new`] or
/// [`Default::default`]; tune via [`EnergyBargeInDetector::with_energy_threshold`].
#[derive(Debug, Clone, Copy)]
pub struct EnergyBargeInDetector {
  /// RMS amplitude threshold in `[0, 1]`. Default `0.02` ≈ −34 dBFS.
  energy_threshold: f32,
}

impl Default for EnergyBargeInDetector {
  fn default() -> Self {
    Self {
      energy_threshold: 0.02,
    }
  }
}

impl EnergyBargeInDetector {
  /// Build a detector with an explicit RMS amplitude threshold (in
  /// `[0, 1]`; clamped to `>= 0` by the detect-time check).
  #[must_use]
  pub const fn new(energy_threshold: f32) -> Self {
    Self { energy_threshold }
  }

  /// The configured RMS amplitude threshold.
  #[inline(always)]
  #[must_use]
  pub const fn energy_threshold(&self) -> f32 {
    self.energy_threshold
  }

  /// Return a copy with a different RMS amplitude threshold.
  #[must_use]
  pub fn with_energy_threshold(self, energy_threshold: f32) -> Self {
    Self { energy_threshold }
  }

  /// Compute root-mean-square amplitude of `samples`. Returns `0`
  /// on an empty slice (matches mlx-audio's
  /// `AudioRecorderManager`-style `sqrt(sum(x^2) / max(N, 1))`
  /// guard).
  #[must_use]
  pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
      return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
  }
}

impl BargeInDetector for EnergyBargeInDetector {
  fn detect(&self, user_audio: &[f32], tts_playing: bool) -> bool {
    if !tts_playing {
      return false;
    }
    let rms = Self::rms(user_audio);
    rms >= self.energy_threshold
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// High-energy audio + TTS playing → barge-in.
  #[test]
  fn high_energy_with_tts_playing_is_barge_in() {
    let detector = EnergyBargeInDetector::default();
    let loud: Vec<f32> = (0..256).map(|i| 0.5 * ((i as f32).sin())).collect();
    assert!(detector.detect(&loud, true));
  }

  /// High-energy audio but TTS not playing → not a barge-in.
  #[test]
  fn high_energy_without_tts_is_not_barge_in() {
    let detector = EnergyBargeInDetector::default();
    let loud: Vec<f32> = (0..256).map(|i| 0.5 * ((i as f32).sin())).collect();
    assert!(!detector.detect(&loud, false));
  }

  /// Low-energy audio + TTS playing → still not a barge-in (heuristic
  /// floors out room noise).
  #[test]
  fn low_energy_with_tts_is_not_barge_in() {
    let detector = EnergyBargeInDetector::default();
    let quiet: Vec<f32> = vec![1e-4; 256];
    assert!(!detector.detect(&quiet, true));
  }

  /// Empty audio is never a barge-in.
  #[test]
  fn empty_audio_is_not_barge_in() {
    let detector = EnergyBargeInDetector::default();
    assert!(!detector.detect(&[], true));
    assert!(!detector.detect(&[], false));
  }

  /// RMS of constant signal equals its amplitude.
  #[test]
  fn rms_of_constant_signal_equals_amplitude() {
    let samples = vec![0.3_f32; 512];
    let rms = EnergyBargeInDetector::rms(&samples);
    assert!((rms - 0.3).abs() < 1e-6, "expected ≈0.3, got {rms}");
  }

  /// Custom threshold honored: lift threshold above RMS → no barge-in.
  #[test]
  fn custom_threshold_honored() {
    let detector = EnergyBargeInDetector::new(0.5);
    let medium: Vec<f32> = vec![0.3; 256];
    // RMS = 0.3 < 0.5 → no barge-in.
    assert!(!detector.detect(&medium, true));
    let loud: Vec<f32> = vec![0.7; 256];
    // RMS = 0.7 ≥ 0.5 → barge-in.
    assert!(detector.detect(&loud, true));
  }
}
