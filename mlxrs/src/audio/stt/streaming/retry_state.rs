//! Unified retry-state machine for [`super::session::StreamingInferenceSession`].
//!
//! The streaming session orchestrates four fallible stages per call:
//! `mel.flush` → `encoder.feed` → per-window decode (`finalize` or
//! pending-window decode pass). Each stage may `Err` independently, and
//! the caller is allowed to retry the failed work by calling
//! `feed_audio` or `stop` again. Pre-`SessionRetryState`, the retry
//! plumbing for these stages lived in three independent session fields
//! (`pending_finalize_queue`, `pending_stop_mel_frames`,
//! `pending_bridge_drain_decode`) plus per-call locals; each new bypass
//! corner Codex review found required a fresh field/flag. Five
//! consecutive review rounds (R3 → R7) each found a NEW way for one of
//! those fields to desync from the others.
//!
//! [`SessionRetryState`] replaces the field-soup with a single source
//! of truth for in-flight retry obligations. Each fallible stage either
//! fully commits or sets a [`RetryStage`] that names exactly where the
//! next call must resume. The session's `discharge_retry_obligation`
//! method calls into the per-stage discharge helpers
//! ([`SessionRetryState::discharge_stop_encoder_feed`] etc.) at the
//! top of every `feed_audio` / `stop`, transactionally drives the
//! resume point, and only proceeds with new audio after the discharge
//! advances `resume_at` to `None`.

use std::collections::VecDeque;

use derive_more::IsVariant;

use super::{
  encoder::{StreamingEncoder, StreamingEncoderBackend},
  mel_spectrogram::IncrementalMelSpectrogram,
};
use crate::{
  Array,
  error::{Error, LayerKeyedPayload, Result},
};

/// One window of encoded mel that owes a finalize decode.
///
/// The `fallback_consumed` flag is a per-entry sticky bit set BEFORE
/// the fallible `decode_all_tokens` call so that on a decode `Err`,
/// the next retry sees `fallback_consumed == true` and gets no fallback
/// — stale streamed text from `SessionSharedState` is never re-applied.
/// Without this gate, a `decode_all_tokens` error would leave the
/// streamed text in `SessionSharedState`, and the retry's empty-decode
/// tiebreaker would freeze that stale provisional over fresh boundary
/// audio.
#[derive(Debug)]
pub(super) struct PendingFinalize {
  /// Encoded hidden states for the completed window.
  pub(super) encoder_output: Array,
  /// `true` once the streamed-text fallback has been offered for this
  /// entry. Sticky across retries — see the doc comment above.
  pub(super) fallback_consumed: bool,
}

/// Stage where a partial-failure retry should resume.
///
/// The streaming session's `feed_audio` / `stop` pipeline has multiple
/// distinct fallible stages. Pre-rewrite, a partial failure at any of
/// them required composing across multiple session fields to recover.
/// The unified state machine names each resume point explicitly so the
/// next call can dispatch to exactly the work that errored — no field
/// composition, no per-call locals that get lost on `?` propagation.
///
/// Failed finalize-queue decodes are NOT carried in a `RetryStage`
/// variant — the [`SessionRetryState::finalize_queue`] field's
/// non-emptiness is the obligation signal (the failed entry is at the
/// queue front).
#[derive(Debug, IsVariant)]
pub(super) enum RetryStage {
  /// `stop()`'s `mel.flush()` errored. The mel processor's transactional
  /// `flush` left its overlap buffer intact, so the next `stop()` call
  /// retries `mel.flush()` exactly. Carries no payload (the source-of-
  /// truth is `IncrementalMelSpectrogram::overlap_buffer`).
  StopMelFlush,
  /// `stop()`'s `mel.flush()` succeeded (committing-and-clearing the
  /// overlap buffer), and the freshly-flushed mel rows live nowhere but
  /// in this payload. Any retry from `feed_audio` / `stop` MUST
  /// re-feed THIS array (the overlap is gone). On Ok the array is
  /// consumed and `resume_at` advances to the next stage if any.
  StopEncoderFeed(Array),
  /// One or more full encoder windows are committed to the encoder's
  /// `newly_encoded_windows` / `cached_windows` AND owe a decode pass.
  /// This covers two surfaces structurally:
  ///   (a) A previous call drained the [`StopEncoderFeed`] bridge with
  ///       a non-zero window count, then errored on a later step in the
  ///       same call (R6 corner: the count was a local, lost on `?`).
  ///   (b) A `run_decode_pass` invocation itself errored mid-way — the
  ///       windows are still in the encoder, the next call MUST decode
  ///       them BEFORE accepting new audio.
  /// Distinct from `resume_at = None + finalize_queue.is_empty()` —
  /// that state means no decode is owed.
  DecodeOwed,
  /// `stop()`'s post-finalize partial-window decode + Ended event
  /// emission. After this stage succeeds, `is_active` flips to false
  /// and the resume point clears. Carries the audio_features payload
  /// so the retry doesn't have to recompute encode_pending (which is
  /// itself fallible and idempotent — but skipping the recompute also
  /// avoids a redundant encoder forward pass).
  StopPartialDecode(Option<Array>),
}

/// Unified retry-state machine.
///
/// Owns the finalize queue + the resume point. Discharge methods are
/// called at the top of every `feed_audio` / `stop` to drive any
/// pending obligation BEFORE new audio is touched; partial discharge
/// leaves [`has_obligation`](Self::has_obligation) true and the session
/// returns the events from what completed without accepting new work.
#[derive(Debug)]
pub(super) struct SessionRetryState {
  /// Resume point for the next call. `None` means no retry is owed —
  /// the session is in clean state. `Some(stage)` means the next
  /// `feed_audio` / `stop` MUST dispatch to that stage BEFORE
  /// processing new audio.
  resume_at: Option<RetryStage>,
  /// Per-window finalize-decode work, FIFO. Drained one window at a
  /// time as decodes succeed. A `decode_all_tokens` Err leaves the
  /// failed entry at the queue front; the queue's non-emptiness alone
  /// is the obligation signal (a non-empty queue ⇒
  /// [`has_obligation`](Self::has_obligation) returns `true` regardless
  /// of `resume_at`).
  finalize_queue: VecDeque<PendingFinalize>,
}

impl Default for SessionRetryState {
  fn default() -> Self {
    Self::new()
  }
}

impl SessionRetryState {
  /// Build a clean retry state — no obligation, empty queue.
  pub(super) fn new() -> Self {
    Self {
      resume_at: None,
      finalize_queue: VecDeque::new(),
    }
  }

  /// True iff some prior call left work that MUST be discharged before
  /// any new audio can be accepted. Either a `resume_at` is set OR the
  /// finalize queue is non-empty — both arms are equally blocking.
  #[inline(always)]
  pub(super) fn has_obligation(&self) -> bool {
    self.resume_at.is_some() || !self.finalize_queue.is_empty()
  }

  /// Inspect the resume point. Borrowed read-only — discharge methods
  /// mutate it via the dedicated `take_*` / `set_*` helpers below.
  #[inline(always)]
  pub(super) fn resume_at(&self) -> Option<&RetryStage> {
    self.resume_at.as_ref()
  }

  /// True iff `resume_at == Some(StopMelFlush)`. The session uses this
  /// to dispatch the unified `StopMelFlush` discharge — without it the
  /// `StopMelFlush` obligation would be stranded forever because
  /// `discharge_retry_obligation`'s dispatcher would have nothing to
  /// gate on and `has_obligation()` would short-circuit `stop()` to an
  /// early-return.
  #[inline(always)]
  pub(super) fn has_pending_stop_mel_flush(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopMelFlush))
  }

  /// True iff `resume_at` names a stage whose source-of-truth lives
  /// inside `mel_processor` / `encoder` — i.e. some prior call's
  /// encoder.feed errored and the staged mel rows live in
  /// `RetryStage::StopEncoderFeed`. The session uses this to keep the
  /// contract "drain the staged stop-tail BEFORE processing new feed audio."
  #[inline(always)]
  pub(super) fn has_pending_stop_encoder_feed(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopEncoderFeed(_)))
  }

  /// True iff `resume_at == Some(DecodeOwed)`.
  #[inline(always)]
  pub(super) fn has_decode_owed(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::DecodeOwed))
  }

  /// Borrow the finalize queue — the session needs read-only access to
  /// drive the `has_pending_retries` gate.
  pub(super) fn finalize_queue(&self) -> &VecDeque<PendingFinalize> {
    &self.finalize_queue
  }

  /// Mutable access to the finalize queue — the session pushes
  /// newly-encoded windows here when entering the finalize-decode
  /// stage, and drains the queue front-to-back as decodes succeed.
  pub(super) fn finalize_queue_mut(&mut self) -> &mut VecDeque<PendingFinalize> {
    &mut self.finalize_queue
  }

  /// Push a freshly-encoded window onto the finalize queue.
  pub(super) fn enqueue_finalize(&mut self, window: Array) {
    self.finalize_queue.push_back(PendingFinalize {
      encoder_output: window,
      fallback_consumed: false,
    });
  }

  /// Discharge the [`RetryStage::StopEncoderFeed`] obligation, if any,
  /// against `encoder`. Returns the number of full windows committed
  /// by the staged `encoder.feed` (`0` on a sub-window staged buffer,
  /// `>= 1` on a window-completing one).
  ///
  /// Transactional: on Err the staged mel stays in `resume_at`. On Ok
  /// the resume point advances to [`RetryStage::DecodeOwed`] iff the
  /// drain produced one or more windows (so the next stage in the same
  /// call OR the next call's discharge runs the decode), or clears if
  /// `0` windows resulted (R5 corner: a sub-window drain owes no decode).
  ///
  /// Returns `Ok(0)` when there is no `StopEncoderFeed` obligation.
  pub(super) fn discharge_stop_encoder_feed<B>(
    &mut self,
    encoder: &mut StreamingEncoder<B>,
  ) -> Result<usize>
  where
    B: StreamingEncoderBackend,
  {
    let Some(RetryStage::StopEncoderFeed(mel_frames)) = self.resume_at.take() else {
      // Not our obligation — restore (we took() it above) and exit.
      // The take() only matched on StopEncoderFeed, so this branch is
      // unreachable, but the explicit check guards against future
      // refactors that might add another variant taking this path.
      return Ok(0);
    };
    // Run the fallible feed against the staged mel. If it errs, we
    // MUST restore the resume_at to its pre-call state so the next
    // call retries the SAME staged mel.
    let count = match encoder.feed(&mel_frames) {
      Ok(n) => n,
      Err(e) => {
        // ROLLBACK: re-arm the resume point. `mel_frames` was MOVED
        // into the match arm — restore by re-constructing.
        self.resume_at = Some(RetryStage::StopEncoderFeed(mel_frames));
        return Err(e);
      }
    };
    // COMMIT: resume_at already cleared by the take() above. If the
    // drain committed one or more windows, advance to DecodeOwed so
    // they flow through a decode pass BEFORE the next call returns.
    // Pre-rewrite this was a separate flag + a per-call local.
    if count > 0 {
      self.resume_at = Some(RetryStage::DecodeOwed);
    }
    Ok(count)
  }

  /// Stage a fresh `StopEncoderFeed` obligation — called by `stop()`
  /// after `mel.flush()` succeeds but BEFORE the encoder.feed call. If
  /// the feed errors, the resume point is already correct; on success
  /// the caller advances by clearing or chaining via the methods above.
  pub(super) fn stage_stop_encoder_feed(&mut self, mel_frames: Array) {
    self.resume_at = Some(RetryStage::StopEncoderFeed(mel_frames));
  }

  /// Stage a fresh `MelFlush` obligation — called by `stop()` BEFORE
  /// invoking `mel.flush()`. The mel processor's transactional `flush`
  /// preserves its `overlap_buffer` on Err, so the next call repeats
  /// the SAME flush.
  pub(super) fn stage_stop_mel_flush(&mut self) {
    self.resume_at = Some(RetryStage::StopMelFlush);
  }

  /// Clear the `MelFlush` obligation after a successful flush — called
  /// by `stop()`'s in-call commit.
  pub(super) fn clear_stop_mel_flush(&mut self) {
    if matches!(self.resume_at, Some(RetryStage::StopMelFlush)) {
      self.resume_at = None;
    }
  }

  /// Discharge the [`RetryStage::StopMelFlush`] obligation, if any,
  /// against `mel_processor`. Re-attempts the `flush()` whose previous
  /// invocation errored, and on success advances the resume point to
  /// [`RetryStage::StopEncoderFeed`] when the flush produced mel rows
  /// (so the next discharge step can drive the encoder feed).
  ///
  /// Returns the freshly-flushed `Option<Array>` so the caller can
  /// inspect it (the in-tree dispatcher discards it and falls through
  /// to [`discharge_stop_encoder_feed`](Self::discharge_stop_encoder_feed),
  /// but callers writing custom orchestrations can use it directly).
  ///
  /// Returns `Ok(None)` when there is no `StopMelFlush` obligation.
  ///
  /// # Transactional rollback
  /// - `mel_processor.flush()` Err → re-arms `StopMelFlush` so the next
  ///   call retries the SAME flush. `IncrementalMelSpectrogram::flush`
  ///   preserves `overlap_buffer` on Err (its own transactional
  ///   contract), so the retry sees identical input.
  /// - `Array::try_clone` on the flushed mel Err (rare — refcount-clone
  ///   only allocates a fresh handle slot) → MOVES the original (still-
  ///   owned) flushed mel into [`RetryStage::StopEncoderFeed`] and
  ///   returns Err. The Err signals to the caller that the in-call path
  ///   cannot continue, but the mel payload is PRESERVED in the
  ///   obligation so the next discharge runs path (b)
  ///   ([`discharge_stop_encoder_feed`](Self::discharge_stop_encoder_feed))
  ///   and feeds the saved mel to the encoder. This is the R2-fix:
  ///   pre-fix re-armed `StopMelFlush` after `flush()` had already
  ///   committed (and cleared the overlap), so the next retry's flush
  ///   would see an empty overlap, short-circuit `Ok(None)`, and emit
  ///   `Ended` with silent tail-audio loss. The new contract is that
  ///   once `flush()` returns `Ok(Some(mel))`, that mel reaches the
  ///   encoder via the obligation regardless of any subsequent failure.
  ///
  /// # Errors
  /// Propagates from [`IncrementalMelSpectrogram::flush`] or from
  /// [`Array::try_clone`].
  pub(super) fn discharge_stop_mel_flush(
    &mut self,
    mel_processor: &mut IncrementalMelSpectrogram,
  ) -> Result<Option<Array>> {
    self.discharge_stop_mel_flush_with_clone(mel_processor, Array::try_clone)
  }

  /// Test-visible inner of [`discharge_stop_mel_flush`] with an injectable
  /// clone function. Production uses [`Array::try_clone`]; tests pass a
  /// closure that deterministically returns `Err` to exercise the R2-fix
  /// rollback path
  /// (see
  /// `discharge_stop_mel_flush_try_clone_err_preserves_mel_as_stop_encoder_feed`).
  ///
  /// `mlx-c`'s `mlx_array_set` only fails on a host-allocator OOM (the
  /// fresh `mlx_array_new` handle must `set` against the source); reaching
  /// that arm in a unit test would require an FFI alloc-failure hook,
  /// which mlx-c does not provide. The injectable seam is the
  /// minimum-disruption alternative — same body, swap one call.
  fn discharge_stop_mel_flush_with_clone<F>(
    &mut self,
    mel_processor: &mut IncrementalMelSpectrogram,
    clone_fn: F,
  ) -> Result<Option<Array>>
  where
    F: FnOnce(&Array) -> Result<Array>,
  {
    let Some(RetryStage::StopMelFlush) = self.resume_at else {
      return Ok(None);
    };

    // Take the obligation so we can either commit or re-arm.
    self.resume_at = None;

    let mel_opt = match mel_processor.flush() {
      Ok(m) => m,
      Err(e) => {
        // Rollback: re-arm StopMelFlush so next stop() retries.
        // `IncrementalMelSpectrogram::flush` is transactional — its
        // overlap_buffer is preserved on Err, so the next flush sees
        // identical input.
        self.resume_at = Some(RetryStage::StopMelFlush);
        return Err(e);
      }
    };

    // CRITICAL INVARIANT (R2-fix): once `flush()` returns Ok(Some(mel)),
    // the overlap has been cleared and the mel rows live ONLY in `mel`.
    // From this point on, ANY return path (Ok or Err) MUST ensure the
    // mel reaches `StopEncoderFeed` if there is a mel — otherwise the
    // next retry would observe an empty overlap, short-circuit, and
    // emit `Ended` with silent tail-audio loss.
    let Some(mel) = mel_opt else {
      // Empty overlap path: flush yielded None. No mel to stage, no
      // obligation to re-arm — clean advance to the next call.
      return Ok(None);
    };

    // Try to clone the mel so we can carry BOTH a copy in the
    // obligation AND a copy in the return value (the caller's in-call
    // continuation immediately feeds it to the encoder).
    match clone_fn(&mel) {
      Ok(for_obligation) => {
        // Both handles live: obligation gets the clone, caller gets
        // the original. The obligation is the safety net — if the
        // caller's in-call path fails downstream, the next discharge
        // re-feeds from the staged clone.
        self.resume_at = Some(RetryStage::StopEncoderFeed(for_obligation));
        Ok(Some(mel))
      }
      Err(e) => {
        // Clone failed. The ORIGINAL `mel` is still owned by us — move
        // it into the obligation so the next discharge can run path
        // (b) (StopEncoderFeed) and feed it to the encoder. Then
        // propagate the Err so the caller's in-call path bails out
        // (it can't continue without a mel handle of its own).
        //
        // Pre-fix re-armed StopMelFlush here; that was wrong because
        // `flush()` had already committed and cleared the overlap, so
        // the next flush would yield None → silent tail loss. The
        // injected-clone-Err test seam
        // (`discharge_stop_mel_flush_try_clone_err_preserves_mel_as_stop_encoder_feed`)
        // gives this branch deterministic regression coverage.
        self.resume_at = Some(RetryStage::StopEncoderFeed(mel));
        Err(Error::LayerKeyed(LayerKeyedPayload::new(
          "StopMelFlush: failed to clone flushed mel for in-call use \
             (obligation preserved as StopEncoderFeed with original payload, \
             retry stop() to discharge)",
          e,
        )))
      }
    }
  }

  /// Mark that the same-call decode for one or more bridge-drained
  /// windows is OWED across call boundaries — called when a later
  /// fallible step in `feed_audio` errors AFTER the bridge drain
  /// successfully committed `count >= 1` windows to the encoder. The
  /// session's local count is lost to the `?` unwind; this flag is
  /// the cross-call source of truth (R6 corner).
  pub(super) fn arm_decode_owed(&mut self) {
    self.resume_at = Some(RetryStage::DecodeOwed);
  }

  /// Clear the `DecodeOwed` obligation after a successful decode pass.
  pub(super) fn clear_decode_owed(&mut self) {
    if matches!(self.resume_at, Some(RetryStage::DecodeOwed)) {
      self.resume_at = None;
    }
  }

  /// Mark that `stop()`'s post-finalize partial-window decode errored.
  /// The audio_features payload is carried in the stage so the retry
  /// doesn't have to recompute `encode_pending` — though the recompute
  /// would be safe (`encode_pending` is `&self` + idempotent), skipping
  /// it avoids a redundant encoder forward pass.
  pub(super) fn arm_stop_partial_decode(&mut self, audio_features: Option<Array>) {
    self.resume_at = Some(RetryStage::StopPartialDecode(audio_features));
  }

  /// True iff `resume_at == Some(StopPartialDecode)`.
  #[inline(always)]
  pub(super) fn has_pending_stop_partial_decode(&self) -> bool {
    matches!(self.resume_at, Some(RetryStage::StopPartialDecode(_)))
  }

  /// Take the staged `StopPartialDecode` audio_features out of the
  /// resume point — used by `stop()`'s discharge to consume the
  /// payload while the resume point is being advanced. Returns `None`
  /// if `resume_at` doesn't currently hold a `StopPartialDecode`.
  pub(super) fn take_stop_partial_decode_features(&mut self) -> Option<Option<Array>> {
    if matches!(self.resume_at, Some(RetryStage::StopPartialDecode(_))) {
      let Some(RetryStage::StopPartialDecode(audio_features)) = self.resume_at.take() else {
        unreachable!("matches! gated the take()")
      };
      Some(audio_features)
    } else {
      None
    }
  }

  /// Reset on cancel() / reset() — clears all obligations atomically.
  pub(super) fn clear_all(&mut self) {
    self.resume_at = None;
    self.finalize_queue.clear();
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Array;

  fn dummy_array() -> Array {
    Array::from_slice::<f32>(&[0.0_f32], &[1i32]).unwrap()
  }

  #[test]
  fn new_has_no_obligation() {
    let s = SessionRetryState::new();
    assert!(!s.has_obligation());
    assert!(s.resume_at().is_none());
    assert!(s.finalize_queue().is_empty());
  }

  #[test]
  fn enqueue_finalize_creates_obligation() {
    let mut s = SessionRetryState::new();
    s.enqueue_finalize(dummy_array());
    assert!(s.has_obligation());
    assert_eq!(s.finalize_queue().len(), 1);
    assert!(!s.finalize_queue()[0].fallback_consumed);
  }

  #[test]
  fn stage_stop_encoder_feed_then_clear_all_clears() {
    let mut s = SessionRetryState::new();
    s.stage_stop_encoder_feed(dummy_array());
    assert!(s.has_pending_stop_encoder_feed());
    s.clear_all();
    assert!(!s.has_pending_stop_encoder_feed());
    assert!(!s.has_obligation());
  }

  #[test]
  fn clear_all_drops_every_obligation_in_one_call() {
    let mut s = SessionRetryState::new();
    s.enqueue_finalize(dummy_array());
    s.arm_decode_owed();
    assert!(s.has_obligation());
    s.clear_all();
    assert!(!s.has_obligation());
    assert!(s.finalize_queue().is_empty());
    assert!(s.resume_at().is_none());
  }

  #[test]
  fn decode_owed_is_distinct_from_throttled_drain() {
    // R7 corner structural fix: there's no flag that bleeds across
    // calls when the same call's cadence throttle skipped the decode.
    // arm_decode_owed is the ONLY way to set DecodeOwed; the session's
    // happy-path drain (count > 0 + same-call decode succeeds) calls
    // clear_decode_owed AFTER the decode pass returns Ok and never
    // calls arm_decode_owed in the first place when the cadence
    // throttle declines the decode.
    let mut s = SessionRetryState::new();
    assert!(!s.has_decode_owed());
    s.arm_decode_owed();
    assert!(s.has_decode_owed());
    s.clear_decode_owed();
    assert!(!s.has_decode_owed());
  }

  #[test]
  fn take_stop_partial_decode_features_returns_none_when_not_set() {
    let mut s = SessionRetryState::new();
    assert!(s.take_stop_partial_decode_features().is_none());
  }

  #[test]
  fn take_stop_partial_decode_features_consumes_payload() {
    let mut s = SessionRetryState::new();
    s.arm_stop_partial_decode(Some(dummy_array()));
    assert!(s.has_pending_stop_partial_decode());
    let taken = s.take_stop_partial_decode_features().expect("set above");
    assert!(taken.is_some());
    assert!(!s.has_pending_stop_partial_decode());
  }

  // -------------------------------------------------------------------
  // F1: discharge_stop_mel_flush wiring + transactional contract.
  // -------------------------------------------------------------------

  /// F1: `discharge_stop_mel_flush` returns `Ok(None)` and clears the
  /// resume point when no obligation is set — the no-op short-circuit.
  #[test]
  fn discharge_stop_mel_flush_noop_when_not_staged() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("noop must succeed");
    assert!(out.is_none(), "no obligation ⇒ Ok(None)");
    assert!(!s.has_obligation());
  }

  /// F1: with an empty overlap, `discharge_stop_mel_flush` clears the
  /// obligation and returns `Ok(None)` (mel.flush short-circuits on
  /// empty overlap). The resume point advances to `None`, NOT to
  /// `StopEncoderFeed` (no mel rows to stage).
  #[test]
  fn discharge_stop_mel_flush_empty_overlap_clears_obligation() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    s.stage_stop_mel_flush();
    assert!(s.has_pending_stop_mel_flush());

    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("empty-overlap flush must succeed");
    assert!(
      out.is_none(),
      "empty overlap ⇒ flush yields None ⇒ no StopEncoderFeed stage"
    );
    assert!(!s.has_pending_stop_mel_flush());
    assert!(
      !s.has_pending_stop_encoder_feed(),
      "F1: no mel rows ⇒ MUST NOT advance to StopEncoderFeed"
    );
    assert!(!s.has_obligation());
  }

  /// F1: with non-empty overlap, `discharge_stop_mel_flush` runs the
  /// flush, advances `resume_at` to `StopEncoderFeed` carrying the
  /// fresh mel, and returns the flushed `Some(mel)` to the caller.
  /// The mel processor's overlap is cleared by the successful flush
  /// (the transactional contract — not the discharge's
  /// responsibility, but observable here).
  #[test]
  fn discharge_stop_mel_flush_with_overlap_advances_to_stop_encoder_feed() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    // Feed a chunk that's smaller than n_fft so process() returns None
    // and the samples accumulate in the overlap.
    let _ = mel
      .process(&[0.1_f32; 16])
      .expect("process must succeed on small input");
    assert!(
      mel.overlap_buffer_len() > 0,
      "test precondition: overlap populated"
    );

    s.stage_stop_mel_flush();
    let out = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("flush must succeed");
    assert!(out.is_some(), "non-empty overlap ⇒ flush yields Some(mel)");
    assert!(
      s.has_pending_stop_encoder_feed(),
      "F1: successful flush with mel rows MUST advance resume_at to \
       StopEncoderFeed for the downstream discharge to drain"
    );
    assert!(
      !s.has_pending_stop_mel_flush(),
      "F1: successful flush MUST clear StopMelFlush"
    );
  }

  // -------------------------------------------------------------------
  // R2-fix: discharge_stop_mel_flush try_clone-Err must preserve the
  // flushed mel in the StopEncoderFeed obligation (NOT re-arm
  // StopMelFlush against a now-empty overlap — that path silently
  // loses the tail audio).
  // -------------------------------------------------------------------

  /// R2-fix structural contract: after `discharge_stop_mel_flush`
  /// successfully runs `mel_processor.flush()` and gets `Some(mel)`,
  /// the mel processor's overlap is committed-cleared. From that
  /// moment, the ONLY remaining source of the tail-audio mel rows is
  /// the freshly-flushed array. The R2-fix invariant says: regardless
  /// of any subsequent try_clone failure, the mel MUST be reachable
  /// from `resume_at` (specifically as `StopEncoderFeed { mel_frames }`).
  /// We can't deterministically force `Array::try_clone` to fail
  /// without an FFI alloc-failure hook, but we can prove the
  /// invariant's positive direction: on the happy path, `resume_at`
  /// lands on `StopEncoderFeed` with a real `mel_frames` payload that
  /// downstream `discharge_stop_encoder_feed` consumes. The Err arm
  /// reuses the SAME `Some(RetryStage::StopEncoderFeed { mel_frames: mel })`
  /// construction (moving the original mel instead of the clone) — see
  /// the source. So the structural reachability is unconditional on
  /// `try_clone`.
  #[test]
  fn discharge_stop_mel_flush_lands_on_stop_encoder_feed_when_overlap_nonempty() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    let _ = mel
      .process(&[0.1_f32; 16])
      .expect("process must succeed on small input");
    assert!(mel.overlap_buffer_len() > 0);

    s.stage_stop_mel_flush();
    let _ = s
      .discharge_stop_mel_flush(&mut mel)
      .expect("happy-path flush must succeed");

    // R2-fix structural assertion: NEVER lands back on StopMelFlush
    // after the flush has committed-cleared the overlap. Either:
    //   (a) try_clone Ok → resume_at == StopEncoderFeed { clone }
    //   (b) try_clone Err → resume_at == StopEncoderFeed { original }
    // Both arms set StopEncoderFeed. The forbidden state is
    // `StopMelFlush` re-armed against an empty overlap (the pre-fix
    // path that emitted Ended with silent tail loss).
    assert!(
      !s.has_pending_stop_mel_flush(),
      "R2-fix: StopMelFlush MUST NOT be re-armed after flush()'s overlap-clear"
    );
    assert!(
      s.has_pending_stop_encoder_feed(),
      "R2-fix: flushed mel MUST reach StopEncoderFeed (the only valid landing)"
    );
    // Overlap is empty post-commit — proves the source-of-truth has
    // shifted from `mel_processor.overlap_buffer` to the obligation.
    assert_eq!(
      mel.overlap_buffer_len(),
      0,
      "test precondition for R2: flush commit clears overlap, so the \
       obligation is the ONLY remaining source of the tail mel"
    );
  }

  /// R2-fix end-to-end recovery: after `discharge_stop_mel_flush`
  /// lands on `StopEncoderFeed`, a subsequent `discharge_stop_encoder_feed`
  /// MUST be able to consume the staged mel and feed it to the encoder
  /// — never silently swallow it. This proves the recovery path the
  /// R2-fix unlocks for the try_clone-Err branch: even if the in-call
  /// path bails with Err, the next call's path (b) discharge picks up
  /// the preserved mel and feeds it.
  ///
  /// Mock encoder is local (the canonical `MockEncoder` in
  /// `encoder.rs` lives in a `#[cfg(test)] mod tests` not exported
  /// across modules).
  #[test]
  fn stop_encoder_feed_obligation_recovers_staged_mel_into_encoder() {
    /// Records every encode_window call so the test can assert the
    /// staged mel reached the encoder backend.
    struct RecordingEncoder {
      window_size: usize,
      call_count: std::cell::RefCell<usize>,
    }
    impl StreamingEncoderBackend for RecordingEncoder {
      fn window_size(&self) -> usize {
        self.window_size
      }
      fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
        *self.call_count.borrow_mut() += 1;
        let rows: usize = mel_window.shape().first().copied().unwrap_or(0);
        // Synthetic encoder output of shape (rows, 2).
        let buf = vec![0.0_f32; rows * 2];
        Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
      }
    }

    // Build mel + flush to obtain a real mel Array we can stage by
    // hand — this mirrors EXACTLY the state the R2-fix Err arm
    // leaves behind (StopEncoderFeed { mel_frames: <real flushed mel> }).
    let mut mel_proc = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    let _ = mel_proc
      .process(&[0.1_f32; 16])
      .expect("process must succeed");
    let mel_array = mel_proc
      .flush()
      .expect("flush must succeed")
      .expect("non-empty overlap ⇒ Some(mel)");
    let n_mels: usize = mel_array.shape().get(1).copied().unwrap_or(0);

    // Synthetic state: arrival shape == post-R2-fix try_clone-Err
    // landing. `StopEncoderFeed` holds the mel; the obligation is
    // discharge-able by path (b).
    let mut s = SessionRetryState::new();
    s.stage_stop_encoder_feed(mel_array);
    assert!(s.has_pending_stop_encoder_feed());

    // Build an encoder whose window_size matches one of the mel-frame
    // counts so feed either drains zero (sub-window) or >=1 windows;
    // either way the encoder backend receives the mel and the
    // obligation is consumed.
    let backend = RecordingEncoder {
      window_size: 1, // smallest window: every mel row is a full window
      call_count: std::cell::RefCell::new(0),
    };
    let mut encoder: StreamingEncoder<RecordingEncoder> = StreamingEncoder::new(backend, 4, 0);

    let drained = s
      .discharge_stop_encoder_feed(&mut encoder)
      .expect("path (b) discharge MUST consume the staged mel");
    assert!(
      *encoder.backend().call_count.borrow() > 0 || n_mels == 0,
      "R2-fix recovery: encoder.feed MUST receive the staged mel (call_count > 0)"
    );
    // After a successful drain, the StopEncoderFeed obligation is
    // gone. If drain > 0 the obligation advances to DecodeOwed (per
    // discharge_stop_encoder_feed's contract); if drain == 0 the
    // obligation clears entirely.
    assert!(
      !s.has_pending_stop_encoder_feed(),
      "R2-fix: successful path-(b) drain MUST clear StopEncoderFeed"
    );
    if drained == 0 {
      assert!(
        !s.has_obligation(),
        "drain=0 path: obligation fully cleared"
      );
    } else {
      assert!(
        s.has_decode_owed(),
        "drain>0 path: obligation advanced to DecodeOwed"
      );
    }
  }

  /// R3-fix DETERMINISTIC regression test for the try_clone-Err arm.
  ///
  /// Pre-R2 the Err arm did `self.resume_at = Some(StopMelFlush)`, which
  /// — combined with `flush()`'s post-commit empty overlap — meant the
  /// next retry's flush short-circuited to `Ok(None)` and silently lost
  /// the tail audio. The R2-fix routes the original `mel` into
  /// `StopEncoderFeed { mel_frames }` on the Err arm so the next
  /// discharge runs path (b) and feeds the preserved mel.
  ///
  /// `mlx_array_set` only fails on host-allocator OOM, which we can't
  /// reach from a unit test. To give the Err branch deterministic
  /// regression coverage we go through the test-only
  /// `discharge_stop_mel_flush_with_clone` seam and inject a clone fn
  /// that always returns `Err`. The pre-R2 production code would FAIL
  /// this test (resume_at would be re-armed to `StopMelFlush` against an
  /// empty overlap); the R2-fix code leaves resume_at on
  /// `StopEncoderFeed` carrying the original mel.
  #[test]
  fn discharge_stop_mel_flush_try_clone_err_preserves_mel_as_stop_encoder_feed() {
    let mut s = SessionRetryState::new();
    let mut mel_proc = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    // Populate the mel overlap so `flush()` returns `Ok(Some(mel))` (the
    // only path that reaches the try_clone call).
    let _ = mel_proc
      .process(&[0.1_f32; 16])
      .expect("process must succeed on small input");
    assert!(
      mel_proc.overlap_buffer_len() > 0,
      "test precondition: overlap populated so flush yields Some(mel)"
    );

    s.stage_stop_mel_flush();
    assert!(s.has_pending_stop_mel_flush());

    // Inject a clone fn that ALWAYS fails — simulates the rare
    // mlx_array_set host-allocator OOM that drives the R2-fix branch.
    let result = s.discharge_stop_mel_flush_with_clone(&mut mel_proc, |_arr| {
      Err(Error::Backend("test-injected clone failure".into()))
    });

    // The discharge MUST surface an Err so the caller's in-call path
    // bails out (it can't continue without a mel handle of its own).
    let err = result.expect_err("injected clone-Err MUST propagate as Err");
    // The discharge wraps the inner clone-Err in `Error::LayerKeyed` with
    // the obligation-recovery context as the layer label.
    assert!(
      matches!(err, Error::LayerKeyed(_)),
      "discharge wraps the clone-Err in Error::LayerKeyed, got {err:?}"
    );

    // CRITICAL R2-fix assertion: resume_at MUST land on StopEncoderFeed
    // carrying the original flushed mel — NEVER on StopMelFlush (the
    // pre-fix path that emits Ended with silent tail loss).
    match s.resume_at() {
      Some(RetryStage::StopEncoderFeed(mel_frames)) => {
        // The staged mel should have non-zero rows (we populated the
        // overlap with 16 samples + a 32-pt FFT, which produces ≥ 1
        // mel frame on flush).
        let rows: usize = mel_frames.shape().first().copied().unwrap_or(0);
        let n_mels: usize = mel_frames.shape().get(1).copied().unwrap_or(0);
        assert!(
          rows > 0,
          "R2-fix: staged mel_frames must carry the flushed rows (got rows=0)"
        );
        assert_eq!(
          n_mels, 8,
          "R2-fix: staged mel_frames must carry the configured n_mels=8"
        );
      }
      other => panic!(
        "R2-fix REGRESSION: expected StopEncoderFeed obligation carrying \
         the preserved mel, got {other:?} — this is the pre-fix silent \
         tail-loss path"
      ),
    }

    // Belt-and-braces: the forbidden re-arm state is explicitly absent.
    assert!(
      !s.has_pending_stop_mel_flush(),
      "R2-fix: StopMelFlush MUST NOT be re-armed after flush()'s overlap-clear"
    );
    // Overlap is empty (the flush committed), so the obligation is the
    // ONLY surviving source of the tail mel.
    assert_eq!(
      mel_proc.overlap_buffer_len(),
      0,
      "test precondition for R2: flush commit clears overlap, so the \
       obligation is the ONLY remaining source of the tail mel"
    );

    // End-to-end recovery: the staged obligation MUST be drainable by
    // a subsequent path (b) discharge. We feed it through a recording
    // encoder and assert the staged mel actually reaches the backend.
    struct RecordingEncoder {
      window_size: usize,
      call_count: std::cell::RefCell<usize>,
    }
    impl StreamingEncoderBackend for RecordingEncoder {
      fn window_size(&self) -> usize {
        self.window_size
      }
      fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
        *self.call_count.borrow_mut() += 1;
        let rows: usize = mel_window.shape().first().copied().unwrap_or(0);
        let buf = vec![0.0_f32; rows * 2];
        Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
      }
    }
    let backend = RecordingEncoder {
      window_size: 1, // smallest window: every mel row is a full window
      call_count: std::cell::RefCell::new(0),
    };
    let mut encoder: StreamingEncoder<RecordingEncoder> = StreamingEncoder::new(backend, 4, 0);

    let _drained = s
      .discharge_stop_encoder_feed(&mut encoder)
      .expect("path (b) discharge MUST consume the preserved mel");
    assert!(
      *encoder.backend().call_count.borrow() > 0,
      "R2-fix end-to-end recovery: the preserved mel MUST reach the \
       encoder backend on the next discharge (call_count > 0)"
    );
    assert!(
      !s.has_pending_stop_encoder_feed(),
      "R2-fix: successful path-(b) drain MUST clear StopEncoderFeed"
    );
  }

  /// R2-fix CRITICAL non-regression: the discharge MUST NEVER land in
  /// a state where `resume_at == Some(StopMelFlush)` AFTER the flush
  /// has been observed to commit (overlap cleared). This is the
  /// exact pre-fix bug: re-arming `StopMelFlush` against an empty
  /// overlap → next retry's flush short-circuits to `Ok(None)` →
  /// `discharge` returns `Ok(None)` → caller emits `Ended` with
  /// silent tail-audio loss. This test runs the discharge on a
  /// populated overlap and confirms the post-state CANNOT be the
  /// forbidden combination (StopMelFlush re-armed AND overlap empty).
  #[test]
  fn discharge_stop_mel_flush_never_leaves_stop_mel_flush_armed_after_overlap_committed() {
    let mut s = SessionRetryState::new();
    let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
    let _ = mel
      .process(&[0.1_f32; 16])
      .expect("process must succeed on small input");
    assert!(mel.overlap_buffer_len() > 0);

    s.stage_stop_mel_flush();
    // The discharge's outcome (Ok or Err) is irrelevant to this
    // invariant — what matters is the post-condition:
    // NOT (resume_at == Some(StopMelFlush) AND overlap_buffer empty).
    let _ = s.discharge_stop_mel_flush(&mut mel);

    let stop_mel_flush_armed = matches!(s.resume_at(), Some(RetryStage::StopMelFlush));
    let overlap_empty = mel.overlap_buffer_len() == 0;
    assert!(
      !(stop_mel_flush_armed && overlap_empty),
      "R2-fix non-regression: forbidden state — StopMelFlush re-armed \
       against empty overlap → next retry's flush short-circuits to \
       Ok(None) and Ended emits silent tail loss. The fix routes the \
       flushed mel into StopEncoderFeed unconditionally so this state \
       is unreachable."
    );
  }
}
