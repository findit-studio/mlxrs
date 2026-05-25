//! [`TurnTakingPolicy`] + [`SilenceTurnTakingPolicy`] — the
//! "user-is-done-speaking" decision shape the orchestrator queries
//! to decide when to finalize a turn and dispatch it to the LLM.
//!
//! mlx-audio's `VoicePipeline` fuses two signals for turn-end:
//!
//! 1. **A learned "smart turn" endpoint model**
//!    ([`SmartTurnEndpointDetector`][vp-smart-turn]) that predicts
//!    `complete: bool` from the full turn audio.
//! 2. **A silence-duration timeout** — the `turn_max_incomplete_silence_ms`
//!    ceiling that force-finalizes a turn after sustained silence
//!    even when the smart-turn model keeps voting "incomplete".
//!
//! mlxrs lifts both into one trait the orchestrator queries. The
//! default [`SilenceTurnTakingPolicy`] implements branch (2) — the
//! silence-threshold heuristic that does not need a learned model.
//! A caller who wants the smart-turn branch wires their own
//! [`TurnTakingPolicy`] impl over a loaded
//! [`crate::audio::vad::VadModel`] (the smart-turn model is a VAD-
//! family endpoint detector mlx-audio loads via
//! `mlx_audio.vad.load("mlx-community/smart-turn-v3")`).
//!
//! Per the [no per-model arch porting][noarch] rule, mlxrs does not
//! bundle a smart-turn impl — the default is the silence-only
//! policy which is correct without a model and lets callers opt into
//! the smart-turn flow on their own.
//!
//! [vp-smart-turn]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L287-L306
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

/// The shape the orchestrator queries to decide whether the user
/// has finished their turn.
///
/// `&self` because the policy should be a pure decision function;
/// stateful tracking (silence-start timestamp, frame counter, etc.)
/// lives in [`super::orchestrator::VoiceSession`].
pub trait TurnTakingPolicy {
  /// Given the recent audio and how long the current silence run
  /// has been (ms), return whether the user has finished speaking
  /// and the orchestrator should finalize the turn.
  ///
  /// `recent_audio` is the buffered turn audio so far (may be empty
  /// if the policy is being queried before any audio arrived);
  /// `silence_ms` is the elapsed silence duration since the last
  /// detected speech frame.
  fn user_finished(&self, recent_audio: &[f32], silence_ms: u32) -> bool;
}

/// Silence-threshold turn-taking policy — the default implementor.
///
/// Returns `true` when `silence_ms >= silence_threshold_ms`,
/// regardless of `recent_audio` content. Matches mlx-audio's
/// silence-timeout branch ([`voice_pipeline.py:1149-1160`][vp-end]),
/// the force-finalization path that fires when the smart-turn model
/// keeps voting "incomplete".
///
/// The default threshold is `1600` ms — mlx-audio's
/// `VoicePipelineConfig.turn_max_incomplete_silence_ms` default
/// ([`voice_pipeline.py:51`][vp-cfg]).
///
/// Construct via [`SilenceTurnTakingPolicy::new`] or
/// [`Default::default`]; tune via
/// [`SilenceTurnTakingPolicy::with_silence_threshold_ms`].
///
/// [vp-end]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L1149-L1160
/// [vp-cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/voice_pipeline.py#L51
#[derive(Debug, Clone, Copy)]
pub struct SilenceTurnTakingPolicy {
  /// Silence duration (ms) required to call the turn finished.
  /// Default `1600` matches mlx-audio's
  /// `turn_max_incomplete_silence_ms` default.
  silence_threshold_ms: u32,
}

impl Default for SilenceTurnTakingPolicy {
  fn default() -> Self {
    Self {
      silence_threshold_ms: 1600,
    }
  }
}

impl SilenceTurnTakingPolicy {
  /// Build a policy with an explicit silence threshold in ms.
  #[must_use]
  pub const fn new(silence_threshold_ms: u32) -> Self {
    Self {
      silence_threshold_ms,
    }
  }

  /// The configured silence threshold (ms).
  #[inline(always)]
  #[must_use]
  pub const fn silence_threshold_ms(&self) -> u32 {
    self.silence_threshold_ms
  }

  /// Return a copy with a different silence threshold (ms).
  #[must_use]
  pub fn with_silence_threshold_ms(self, silence_threshold_ms: u32) -> Self {
    Self {
      silence_threshold_ms,
    }
  }
}

impl TurnTakingPolicy for SilenceTurnTakingPolicy {
  fn user_finished(&self, _recent_audio: &[f32], silence_ms: u32) -> bool {
    silence_ms >= self.silence_threshold_ms
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Silence ≥ threshold → finished.
  #[test]
  fn silence_above_threshold_finishes_turn() {
    let policy = SilenceTurnTakingPolicy::default();
    assert!(policy.user_finished(&[], 1600));
    assert!(policy.user_finished(&[], 1601));
    assert!(policy.user_finished(&[], 99_999));
  }

  /// Silence < threshold → not finished.
  #[test]
  fn silence_below_threshold_keeps_turn_open() {
    let policy = SilenceTurnTakingPolicy::default();
    assert!(!policy.user_finished(&[], 0));
    assert!(!policy.user_finished(&[], 100));
    assert!(!policy.user_finished(&[], 1599));
  }

  /// Custom threshold honored.
  #[test]
  fn custom_threshold_honored() {
    let policy = SilenceTurnTakingPolicy::new(500);
    assert!(!policy.user_finished(&[], 499));
    assert!(policy.user_finished(&[], 500));
  }

  /// Recent audio content is ignored — only `silence_ms` matters
  /// for the silence policy.
  #[test]
  fn recent_audio_content_is_ignored() {
    let policy = SilenceTurnTakingPolicy::new(200);
    let loud = vec![0.9_f32; 1024];
    let quiet = vec![0.0_f32; 1024];
    assert!(policy.user_finished(&loud, 250));
    assert!(policy.user_finished(&quiet, 250));
    assert!(!policy.user_finished(&loud, 100));
    assert!(!policy.user_finished(&quiet, 100));
  }
}
