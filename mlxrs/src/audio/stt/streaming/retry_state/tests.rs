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
  // There's no flag that bleeds across calls when the same call's
  // cadence throttle skipped the decode.
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
// discharge_stop_mel_flush wiring + transactional contract.
// -------------------------------------------------------------------

/// `discharge_stop_mel_flush` returns `Ok(None)` and clears the
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

/// With an empty overlap, `discharge_stop_mel_flush` clears the
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
    "no mel rows ⇒ MUST NOT advance to StopEncoderFeed"
  );
  assert!(!s.has_obligation());
}

/// With non-empty overlap, `discharge_stop_mel_flush` runs the
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
    "successful flush with mel rows MUST advance resume_at to \
       StopEncoderFeed for the downstream discharge to drain"
  );
  assert!(
    !s.has_pending_stop_mel_flush(),
    "successful flush MUST clear StopMelFlush"
  );
}

// -------------------------------------------------------------------
// discharge_stop_mel_flush try_clone-Err must preserve the
// flushed mel in the StopEncoderFeed obligation (NOT re-arm
// StopMelFlush against a now-empty overlap — that path silently
// loses the tail audio).
// -------------------------------------------------------------------

/// Structural contract: after `discharge_stop_mel_flush`
/// successfully runs `mel_processor.flush()` and gets `Some(mel)`,
/// the mel processor's overlap is committed-cleared. From that
/// moment, the ONLY remaining source of the tail-audio mel rows is
/// the freshly-flushed array. The invariant says: regardless
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

  // Structural assertion: NEVER lands back on StopMelFlush
  // after the flush has committed-cleared the overlap. Either:
  //   (a) try_clone Ok → resume_at == StopEncoderFeed { clone }
  //   (b) try_clone Err → resume_at == StopEncoderFeed { original }
  // Both arms set StopEncoderFeed. The forbidden state is
  // `StopMelFlush` re-armed against an empty overlap (the path that
  // would emit Ended with silent tail loss).
  assert!(
    !s.has_pending_stop_mel_flush(),
    "StopMelFlush MUST NOT be re-armed after flush()'s overlap-clear"
  );
  assert!(
    s.has_pending_stop_encoder_feed(),
    "flushed mel MUST reach StopEncoderFeed (the only valid landing)"
  );
  // Overlap is empty post-commit — proves the source-of-truth has
  // shifted from `mel_processor.overlap_buffer` to the obligation.
  assert_eq!(
    mel.overlap_buffer_len(),
    0,
    "test precondition: flush commit clears overlap, so the \
       obligation is the ONLY remaining source of the tail mel"
  );
}

/// End-to-end recovery: after `discharge_stop_mel_flush`
/// lands on `StopEncoderFeed`, a subsequent `discharge_stop_encoder_feed`
/// MUST be able to consume the staged mel and feed it to the encoder
/// — never silently swallow it. This proves the recovery path
/// for the try_clone-Err branch: even if the in-call
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
  // hand — this mirrors EXACTLY the state the try_clone-Err arm
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

  // Synthetic state: arrival shape == the try_clone-Err
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
    "recovery: encoder.feed MUST receive the staged mel (call_count > 0)"
  );
  // After a successful drain, the StopEncoderFeed obligation is
  // gone. If drain > 0 the obligation advances to DecodeOwed (per
  // discharge_stop_encoder_feed's contract); if drain == 0 the
  // obligation clears entirely.
  assert!(
    !s.has_pending_stop_encoder_feed(),
    "successful path-(b) drain MUST clear StopEncoderFeed"
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

/// DETERMINISTIC regression test for the try_clone-Err arm.
///
/// A naive Err arm that did `self.resume_at = Some(StopMelFlush)`
/// — combined with `flush()`'s post-commit empty overlap — would mean
/// the next retry's flush short-circuits to `Ok(None)` and silently
/// loses the tail audio. The correct behavior routes the original
/// `mel` into `StopEncoderFeed { mel_frames }` on the Err arm so the
/// next discharge runs path (b) and feeds the preserved mel.
///
/// `mlx_array_set` only fails on host-allocator OOM, which we can't
/// reach from a unit test. To give the Err branch deterministic
/// regression coverage we go through the test-only
/// `discharge_stop_mel_flush_with_clone` seam and inject a clone fn
/// that always returns `Err`. A naive implementation would FAIL
/// this test (resume_at re-armed to `StopMelFlush` against an
/// empty overlap); the correct code leaves resume_at on
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
  // mlx_array_set host-allocator OOM that drives the fix branch.
  let result = s.discharge_stop_mel_flush_with_clone(&mut mel_proc, |_arr| {
    Err(Error::InvariantViolation(
      crate::error::InvariantViolationPayload::new(
        "discharge_stop_mel_flush_with_clone",
        "test-injected clone failure",
      ),
    ))
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

  // CRITICAL assertion: resume_at MUST land on StopEncoderFeed
  // carrying the original flushed mel — NEVER on StopMelFlush (the
  // path that would emit Ended with silent tail loss).
  match s.resume_at() {
    Some(RetryStage::StopEncoderFeed(mel_frames)) => {
      // The staged mel should have non-zero rows (we populated the
      // overlap with 16 samples + a 32-pt FFT, which produces ≥ 1
      // mel frame on flush).
      let rows: usize = mel_frames.shape().first().copied().unwrap_or(0);
      let n_mels: usize = mel_frames.shape().get(1).copied().unwrap_or(0);
      assert!(
        rows > 0,
        "staged mel_frames must carry the flushed rows (got rows=0)"
      );
      assert_eq!(
        n_mels, 8,
        "staged mel_frames must carry the configured n_mels=8"
      );
    }
    other => panic!(
      "REGRESSION: expected StopEncoderFeed obligation carrying \
         the preserved mel, got {other:?} — this is the silent \
         tail-loss path"
    ),
  }

  // Belt-and-braces: the forbidden re-arm state is explicitly absent.
  assert!(
    !s.has_pending_stop_mel_flush(),
    "StopMelFlush MUST NOT be re-armed after flush()'s overlap-clear"
  );
  // Overlap is empty (the flush committed), so the obligation is the
  // ONLY surviving source of the tail mel.
  assert_eq!(
    mel_proc.overlap_buffer_len(),
    0,
    "test precondition: flush commit clears overlap, so the \
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
    "end-to-end recovery: the preserved mel MUST reach the \
       encoder backend on the next discharge (call_count > 0)"
  );
  assert!(
    !s.has_pending_stop_encoder_feed(),
    "successful path-(b) drain MUST clear StopEncoderFeed"
  );
}

/// CRITICAL non-regression: the discharge MUST NEVER land in
/// a state where `resume_at == Some(StopMelFlush)` AFTER the flush
/// has been observed to commit (overlap cleared). The forbidden
/// bug is re-arming `StopMelFlush` against an empty
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
    "non-regression: forbidden state — StopMelFlush re-armed \
       against empty overlap → next retry's flush short-circuits to \
       Ok(None) and Ended emits silent tail loss. Routing the \
       flushed mel into StopEncoderFeed unconditionally keeps this state \
       unreachable."
  );
}
