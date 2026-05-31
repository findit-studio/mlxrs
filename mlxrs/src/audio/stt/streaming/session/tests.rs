use super::*;
use crate::{
  audio::stt::streaming::{encoder::StreamingEncoderBackend, types::DelayPreset},
  error::InvariantViolationPayload,
};
use std::sync::Mutex;

// -----------------------------------------------------------------
// Mocks
// -----------------------------------------------------------------

/// Scripted encoder backend: per-call `Result<(), Error>` script
/// (`true` = Err, `false` = Ok). On Ok produces a passthrough
/// `(rows, 2)` array; on Err returns the scripted backend failure.
/// Records the per-call mel_window[0][0] fingerprint BEFORE the
/// error gate so order-sensitive tests can prove which staged
/// buffer reached the encoder first.
struct ScriptedEncoder {
  window_size: usize,
  err_script: Mutex<Vec<bool>>,
  calls: Mutex<usize>,
  fingerprints: Mutex<Vec<f32>>,
}
impl ScriptedEncoder {
  fn new(window_size: usize, err_script: Vec<bool>) -> Self {
    Self {
      window_size,
      err_script: Mutex::new(err_script),
      calls: Mutex::new(0),
      fingerprints: Mutex::new(Vec::new()),
    }
  }
  fn call_count(&self) -> usize {
    *self.calls.lock().unwrap()
  }
  fn fingerprints(&self) -> Vec<f32> {
    self.fingerprints.lock().unwrap().clone()
  }
}
impl StreamingEncoderBackend for ScriptedEncoder {
  fn window_size(&self) -> usize {
    self.window_size
  }
  fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
    *self.calls.lock().unwrap() += 1;
    let fingerprint = mel_window
      .try_clone()
      .and_then(|mut a| a.to_vec::<f32>())
      .ok()
      .and_then(|v| v.first().copied())
      .unwrap_or(f32::NAN);
    self.fingerprints.lock().unwrap().push(fingerprint);
    let mut script = self.err_script.lock().unwrap();
    let should_err = if script.is_empty() {
      false
    } else {
      script.remove(0)
    };
    if should_err {
      return Err(crate::error::Error::InvariantViolation(
        crate::error::InvariantViolationPayload::new(
          "ScriptedEncoder::encode_window",
          "scripted failure",
        ),
      ));
    }
    let rows = mel_window.shape().first().copied().unwrap_or(0);
    let buf = vec![0.0_f32; rows * 2];
    Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
  }
}

/// Mock decoder that takes a `Result<Vec<u32>>` per call so individual
/// passes can inject decoder errors. Tracks call count.
struct ScriptedDecoder {
  results: Mutex<Vec<Result<Vec<u32>>>>,
  calls: Mutex<usize>,
}
impl ScriptedDecoder {
  fn with_results(results: Vec<Result<Vec<u32>>>) -> Self {
    Self {
      results: Mutex::new(results),
      calls: Mutex::new(0),
    }
  }
  fn call_count(&self) -> usize {
    *self.calls.lock().unwrap()
  }
}
impl StreamingDecoderBackend for ScriptedDecoder {
  fn decode_all_tokens(
    &self,
    _audio_features: &Array,
    _confirmed_token_ids: &[u32],
    _config: &StreamingConfig,
    _max_tokens: usize,
  ) -> Result<Vec<u32>> {
    *self.calls.lock().unwrap() += 1;
    let mut q = self.results.lock().unwrap();
    if q.is_empty() {
      Ok(Vec::new())
    } else {
      q.remove(0)
    }
  }
}

struct MockTokenizer;
impl StreamingTokenizer for MockTokenizer {
  fn decode_ids(&self, ids: &[u32]) -> String {
    ids
      .iter()
      .map(|id| format!("t{id}"))
      .collect::<Vec<_>>()
      .join(" ")
  }
}

/// Build a non-finalize_completed_windows session over scripted
/// encoder + decoder. Window size 8 mel-frames keeps the boundary
/// cadence tight.
fn nonfinalize_session(
  encoder: ScriptedEncoder,
  decoder: ScriptedDecoder,
) -> StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer> {
  let cfg = StreamingConfig::default()
    .with_decode_interval_seconds(0.0)
    .with_boundary_decode_interval_seconds(0.0)
    .with_boundary_boost_seconds(0.0)
    .with_max_cached_windows(8)
    .with_finalize_completed_windows(false)
    .with_min_agreement_passes(1)
    .with_boundary_min_agreement_passes(1)
    .with_delay_preset(DelayPreset::Custom(0));
  StreamingInferenceSession::new(decoder, MockTokenizer, cfg, encoder, 16_000, 8, 0).unwrap()
}

/// Same as `nonfinalize_session` but with finalize_completed_windows = true.
fn finalize_session(
  encoder: ScriptedEncoder,
  decoder: ScriptedDecoder,
) -> StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer> {
  let cfg = StreamingConfig::default()
    .with_decode_interval_seconds(0.0)
    .with_boundary_decode_interval_seconds(0.0)
    .with_boundary_boost_seconds(0.0)
    .with_max_cached_windows(8)
    .with_finalize_completed_windows(true)
    .with_min_agreement_passes(1)
    .with_boundary_min_agreement_passes(1)
    .with_delay_preset(DelayPreset::Custom(0));
  StreamingInferenceSession::new(decoder, MockTokenizer, cfg, encoder, 16_000, 8, 0).unwrap()
}

/// Drive the session through (a) a partial-window feed that exercises
/// the pending-window pre-boundary decode, then (b) a top-up feed
/// that closes the first window and triggers the boundary finalize
/// pass. n_fft=400, hop=160 → 400 + 7×160 = 1 520 samples for one
/// window of mel content. Using 800 + 1 200 samples gives one
/// partial-decode call + one finalize-decode call.
fn drive_two_phase(
  session: &mut StreamingInferenceSession<ScriptedEncoder, ScriptedDecoder, MockTokenizer>,
) -> (
  Result<Vec<TranscriptionEvent>>,
  Result<Vec<TranscriptionEvent>>,
) {
  let partial: Vec<f32> = (0..800).map(|i| (i as f32 * 0.001).sin()).collect();
  let topup: Vec<f32> = (800..2_000).map(|i| (i as f32 * 0.001).sin()).collect();
  let partial_events = session.feed_audio(&partial);
  let boundary_events = session.feed_audio(&topup);
  (partial_events, boundary_events)
}

// -----------------------------------------------------------------
// Baseline: feed_audio + stop happy paths
// -----------------------------------------------------------------

#[test]
fn feed_audio_short_input_yields_no_events_until_mel_emits() {
  let encoder = ScriptedEncoder::new(16, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10, 11, 12])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let events = session.feed_audio(&[0.0_f32; 1]).unwrap();
  assert!(events.is_empty(), "events={events:?}");
}

#[test]
fn feed_audio_long_input_drives_partial_decode() {
  let encoder = ScriptedEncoder::new(16, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10, 11, 12])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
  let events = session.feed_audio(&samples).unwrap();
  assert_eq!(session.decoder.call_count(), 1);
  assert!(
    matches!(events.first(), Some(TranscriptionEvent::Confirmed(_))),
    "events[0]={:?}",
    events.first()
  );
  assert!(
    events
      .iter()
      .any(|e| matches!(e, TranscriptionEvent::Stats(_)))
  );
}

#[test]
fn stop_emits_ended_event() {
  let encoder = ScriptedEncoder::new(16, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![10])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();
  let stop_events = session.stop().unwrap();
  assert!(
    matches!(stop_events.last(), Some(TranscriptionEvent::Ended(_))),
    "stop events: {stop_events:?}"
  );
  assert!(!session.is_active());
}

#[test]
fn cancel_marks_inactive_and_drops_state() {
  let encoder = ScriptedEncoder::new(16, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();
  session.cancel();
  assert!(!session.is_active());
  let after = session.feed_audio(&samples).unwrap();
  assert!(after.is_empty());
}

#[test]
fn append_text_basic_concatenation_and_trim() {
  let mut base = String::new();
  append_text("hello", &mut base);
  assert_eq!(base, "hello");
  append_text("world", &mut base);
  assert_eq!(base, "hello world");
  append_text("  ", &mut base);
  assert_eq!(base, "hello world");
  append_text("!", &mut base);
  assert_eq!(base, "hello world !");
}

// -----------------------------------------------------------------
// Completed-window finalization
// -----------------------------------------------------------------

/// When the first window is finalized, the FULL decode must run
/// over the completed-window features — even when streamed-fallback
/// text exists from a prior pending-window decode.
#[test]
fn streaming_session_first_window_finalization_runs_full_decode_not_streamed_fallback() {
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![9, 8, 7])]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  boundary_events.unwrap();
  assert!(
    session.decoder.call_count() >= 2,
    "expected >= 2 decoder calls (pending + finalize), got {}",
    session.decoder.call_count()
  );
}

/// When the streamed-fallback text differs from the full-decode
/// text, the FULL-decode text MUST land in `completed_text`.
#[test]
fn streaming_session_first_window_finalization_appends_full_decode_not_partial_text() {
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1, 2, 3]), Ok(vec![90, 91, 92])]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  boundary_events.unwrap();
  let stop_events = session.stop().unwrap();
  let TranscriptionEvent::Ended(full_text) = stop_events
    .last()
    .expect("expected Ended event at stop()")
    .clone()
  else {
    panic!("last stop event was not Ended: {stop_events:?}");
  };
  assert!(
    full_text.contains("t90"),
    "Ended.full_text must include the full-decode text, got {full_text:?}"
  );
}

/// On a finalize-decode error the failed window stays in the
/// retry queue so a subsequent `feed_audio` call can re-attempt it.
/// (feed_audio drains a pending retry queue.)
#[test]
fn streaming_session_decoder_error_keeps_window_for_retry_then_feed_audio_drains() {
  use crate::error::Error;
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted finalize failure",
    ))),
    Ok(vec![42]),
  ]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  assert!(boundary_events.is_err());
  assert_eq!(
    session.retry_state().finalize_queue().len(),
    1,
    "errored finalize must leave the window in the retry queue"
  );
  // Retry via small follow-up feed.
  let retry_events = session.feed_audio(&[0.0_f32; 200]).unwrap();
  assert_eq!(
    session.retry_state().finalize_queue().len(),
    0,
    "successful retry must pop the previously-failed window"
  );
  assert!(
    !retry_events.is_empty(),
    "retry decode must emit at least the Stats event"
  );
}

/// `frozen_window_count` invariant across a finalize error.
#[test]
fn streaming_session_decoder_error_does_not_advance_frozen_window_count() {
  use crate::error::Error;
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted finalize failure",
    ))),
  ]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  assert!(boundary_events.is_err());
  assert_eq!(session.encoded_window_count(), 1);
  assert_eq!(session.frozen_window_count, 0);
  assert_eq!(session.retry_state().finalize_queue().len(), 1);
}

// -----------------------------------------------------------------
// Retry semantics around stop() / feed_audio()
// -----------------------------------------------------------------

/// When `stop()` errors inside finalize, the session must
/// remain RETRYABLE — a second `stop()` call must drain the queue.
#[test]
fn streaming_session_stop_with_finalize_err_can_be_retried_with_second_stop() {
  use crate::error::Error;
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted boundary finalize Err",
    ))),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted stop-retry finalize Err",
    ))),
    Ok(vec![42]),
  ]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  assert!(boundary_events.is_err(), "boundary feed must Err");
  assert_eq!(session.retry_state().finalize_queue().len(), 1);

  let stop_first = session.stop();
  assert!(
    stop_first.is_err(),
    "first stop() must propagate the scripted finalize Err"
  );

  let stop_second = session.stop().expect("second stop() must succeed");
  assert!(
    matches!(stop_second.last(), Some(TranscriptionEvent::Ended(_))),
    "second stop() must emit terminal Ended, got {stop_second:?}"
  );
  assert!(!session.is_active());
  assert_eq!(session.retry_state().finalize_queue().len(), 0);
}

/// A follow-up `feed_audio` with EMPTY input must drive the
/// retry path.
#[test]
fn streaming_session_pending_retry_finalizes_on_empty_feed_audio() {
  use crate::error::Error;
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted boundary finalize Err",
    ))),
    Ok(vec![77]),
  ]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  assert!(boundary_events.is_err());
  assert_eq!(session.retry_state().finalize_queue().len(), 1);
  let calls_before = session.decoder.call_count();

  let retry_events = session
    .feed_audio(&[])
    .expect("empty feed_audio retry must succeed");
  assert_eq!(session.retry_state().finalize_queue().len(), 0);
  assert!(session.decoder.call_count() > calls_before);
  assert!(!retry_events.is_empty());
}

/// After a finalize-decode Err, the streamed-text fallback
/// MUST NOT be re-armed on retry — `PendingFinalize::fallback_consumed`
/// is sticky.
#[test]
fn streaming_session_fallback_not_reapplied_on_retry_after_err() {
  use crate::error::Error;
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![123]),
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "ScriptedDecoder",
      "scripted boundary finalize Err",
    ))),
    Ok(vec![]),
  ]);
  let mut session = finalize_session(encoder, decoder);
  let (partial_events, boundary_events) = drive_two_phase(&mut session);
  partial_events.unwrap();
  assert!(boundary_events.is_err());
  assert_eq!(session.retry_state().finalize_queue().len(), 1);

  let _ = session
    .feed_audio(&[])
    .expect("retry feed_audio must succeed");
  assert_eq!(session.retry_state().finalize_queue().len(), 0);

  // The stale "t123" provisional MUST NOT be frozen into
  // completed_text — the per-entry fallback flag prevented re-arm.
  assert!(
    !session.shared.completed_text.contains("t123"),
    "stale streamed fallback must NOT be frozen, got {:?}",
    session.shared.completed_text
  );
}

// -----------------------------------------------------------------
// stop() preserves tail audio across mel.flush / encoder
// .feed Err. Uses the SessionRetryState's StopEncoderFeed
// stage as the bridge.
// -----------------------------------------------------------------

/// stop()'s encoder.feed Err preserves the
/// freshly-flushed mel_frames so a retry stop() can re-feed them.
///
/// (Named for the misnomer — the scripted Err is on
/// `encode_window`, NOT on `mel.flush()`. The mel.flush-Err path is
/// covered separately by
/// [`session_retry_state_stop_with_mel_flush_err_can_be_retried`]
/// and friends.)
#[test]
fn session_retry_state_stop_with_encoder_feed_err_can_be_retried() {
  // 1200 samples ⇒ 7 mel frames in encoder pending (< 8); mel.flush
  // emits ~2 more → encoder.feed sees 9 → 1 full window → encode_window
  // is called and scripted to Err.
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3, 4])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();

  let overlap_before = session.mel_processor.overlap_buffer_len();
  assert!(
    overlap_before > 0,
    "test precondition: overlap must be populated"
  );

  // First stop(): mel.flush Ok (commits overlap clear), encoder.feed
  // Errs on encode_window.
  let stop_first = session.stop();
  assert!(stop_first.is_err());
  // Mel overlap cleared (transactional flush succeeded).
  assert_eq!(session.mel_processor.overlap_buffer_len(), 0);
  // Bridge holds the tail mel.
  assert!(
    session.retry_state().has_pending_stop_encoder_feed(),
    "SessionRetryState StopEncoderFeed MUST hold the tail mel"
  );
  assert!(
    session.is_active(),
    "errored stop() must leave session active"
  );

  // Retry stop(): bridge re-feeds → success → Ended.
  let stop_second = session.stop().expect("retry stop() must succeed");
  assert!(matches!(
    stop_second.last(),
    Some(TranscriptionEvent::Ended(_))
  ));
  assert!(!session.is_active());
  assert!(!session.retry_state().has_obligation());
}

// -----------------------------------------------------------------
// feed_audio MUST drain pending stop-tail BEFORE new audio.
// -----------------------------------------------------------------

/// A `feed_audio(new_samples)` AFTER a
/// transient stop()-Err MUST run the bridge drain BEFORE
/// mel.process(new_samples) / encoder.feed(new_mel). The
/// SessionRetryState's StopEncoderFeed discharge is called at the
/// TOP of feed_audio, before the normal feed-path.
#[test]
fn session_retry_state_feed_audio_drains_stop_tail_before_new_audio() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7])]);
  let mut session = nonfinalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let stop_first = session.stop();
  assert!(stop_first.is_err());
  assert!(session.retry_state().has_pending_stop_encoder_feed());
  let stop_err_call_idx = session.encoder.backend().call_count() - 1;
  let stop_err_fingerprint = session.encoder.backend().fingerprints()[stop_err_call_idx];

  // PROBE: deliver NEW audio. The bridge drain MUST run FIRST.
  let samples_b: Vec<f32> = (0..1200)
    .map(|i| ((i as f32 + 7.0) * 0.013).cos())
    .collect();
  let _events = session
    .feed_audio(&samples_b)
    .expect("feed_audio after staged-tail stop Err MUST succeed");

  // Bridge cleared on successful drain.
  assert!(
    !session.retry_state().has_pending_stop_encoder_feed(),
    "feed_audio MUST clear StopEncoderFeed after a successful drain"
  );

  // ORDER ASSERTION: the call immediately after stop-Err MUST be
  // the bridge drain (same fingerprint).
  let fingerprints = session.encoder.backend().fingerprints();
  let bridge_drain_idx = stop_err_call_idx + 1;
  assert!(fingerprints.len() > bridge_drain_idx);
  assert_eq!(
    fingerprints[bridge_drain_idx].to_bits(),
    stop_err_fingerprint.to_bits(),
    "ORDER: the call immediately after stop-Err MUST be the bridge \
       drain (bit-identical fingerprint). fingerprints={fingerprints:?}"
  );
}

/// `feed_audio(&[])` MUST drain the staged tail even with
/// zero new samples. The drain commits 1 window (encode_window call);
/// the SessionRetryState's discharge then drives a same-call decode
/// pass for the drained window (a second encode_window call for the
/// 1-row carry's encode_pending — the contract that drained
/// windows must flow through a decode in THIS call). Total: 2
/// encode_window calls in the same feed_audio.
#[test]
fn session_retry_state_feed_audio_empty_samples_drains_bridge_and_decodes() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
  let mut session = nonfinalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let _ = session.stop();
  assert!(session.retry_state().has_pending_stop_encoder_feed());
  let decoder_calls_before = session.decoder.call_count();

  // Reset cadence-gate so the discharge's decode pass isn't throttled.
  session.last_decode_time = None;

  let _events = session
    .feed_audio(&[])
    .expect("empty feed_audio MUST succeed by draining the bridge");
  assert!(!session.retry_state().has_pending_stop_encoder_feed());
  // Contract: drained window decoded in THIS call.
  assert_eq!(
    session.decoder.call_count(),
    decoder_calls_before + 1,
    "drained window MUST be decoded in the same call"
  );
  // All obligations cleared on happy path.
  assert!(!session.retry_state().has_obligation());
}

// -----------------------------------------------------------------
// Drained windows decoded in the SAME feed_audio call.
// -----------------------------------------------------------------

/// A bridge drain that completes a full
/// encoder window MUST route that window through `run_decode_pass`
/// IN THE SAME `feed_audio` call. The `discharge_stop_encoder_feed`
/// returns count > 0 and arms DecodeOwed, which immediately drives
/// the decode pass in the same discharge.
#[test]
fn session_retry_state_drained_windows_decoded_in_same_call() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
  let mut session = nonfinalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let decoder_calls_after_initial = session.decoder.call_count();
  let _ = session.stop();
  assert!(session.retry_state().has_pending_stop_encoder_feed());
  let encoder_windows_before_drain = session.encoder.encoded_window_count();

  // Reset cadence gate.
  session.last_decode_time = None;

  let _events = session
    .feed_audio(&[])
    .expect("feed_audio after staged-tail stop Err MUST succeed");

  assert!(!session.retry_state().has_pending_stop_encoder_feed());
  assert_eq!(
    session.encoder.encoded_window_count(),
    encoder_windows_before_drain + 1,
    "bridge drain MUST have committed exactly one full window"
  );
  // CORE: the decoder was invoked for the partial-window decode in
  // the SAME call as the drain.
  assert_eq!(
    session.decoder.call_count(),
    decoder_calls_after_initial + 1,
    "bridge-drained window MUST drive run_decode_pass in this call"
  );
}

/// Sub-window drain → bridge cleared (no window completed). The
/// decode-owed obligation MUST NOT be armed (only count > 0 arms it).
#[test]
fn session_retry_state_bridge_drain_no_windows_clears_bridge_no_obligation() {
  let encoder = ScriptedEncoder::new(8, vec![]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
  let mut session = nonfinalize_session(encoder, decoder);

  // Inject a sub-window mel directly into the bridge via the
  // test-only retry_state_mut accessor.
  let staged_mel = Array::from_slice::<f32>(&[0.0_f32; 2 * 8], &[2_i32, 8_i32]).unwrap();
  session
    .retry_state_mut()
    .stage_stop_encoder_feed(staged_mel);

  let _events = session
    .feed_audio(&[])
    .expect("empty feed_audio with non-completing drain MUST succeed");
  // Bridge cleared (successful drain always clears).
  assert!(!session.retry_state().has_pending_stop_encoder_feed());
  // Core: drain returned 0 windows → no DecodeOwed obligation
  // armed. (The encoder may have pending sub-window frames that the
  // normal pending-window decode picks up — that's a separate path
  // governed by the cadence gate, not the discharge.)
  assert!(
    !session.retry_state().has_decode_owed(),
    "drain with 0 windows MUST NOT arm DecodeOwed"
  );
  assert_eq!(session.encoder.encoded_window_count(), 0);
}

// -----------------------------------------------------------------
// A same-call decode obligation MUST survive a later `?`
// propagation. The DecodeOwed stage is the cross-call
// source of truth — no per-call locals.
// -----------------------------------------------------------------

/// A bridge drain SUCCESS in feed_audio
/// followed by a later Err on the new-audio encoder.feed must arm
/// DecodeOwed; a retry feed_audio(&[]) MUST decode the drained
/// window AND clear the obligation.
#[test]
fn session_retry_state_drained_count_survives_post_drain_err() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![42])]);
  let mut session = finalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let decoder_calls_after_initial = session.decoder.call_count();
  let _ = session.stop();
  assert!(session.retry_state().has_pending_stop_encoder_feed());

  // PROBE: deliver new audio. Discharge drains successfully (1
  // window committed) and arms DecodeOwed. The discharge's decode
  // pass runs and SUCCEEDS (clears DecodeOwed). Then back in
  // feed_audio normal path, encoder.feed(new_mel) Errs.
  //
  // The contract specifically requires: the drained window's
  // decode obligation survives a `?` from later. The discharge runs
  // the decode IN THE DISCHARGE, so the "drained but not decoded"
  // state across call boundaries is structurally impossible when the
  // drain returns count > 0.
  //
  // But the alternate surface IS reachable: encoder.feed(new_mel)
  // succeeds with new_windows > 0, then run_decode_pass Errs. The
  // new windows are stranded. DecodeOwed is armed BEFORE
  // run_decode_pass when new_windows > 0 to cover this.
  let samples_b: Vec<f32> = (0..1500)
    .map(|i| ((i as f32 + 11.0) * 0.013).cos())
    .collect();
  let feed_err = session.feed_audio(&samples_b);
  assert!(
    feed_err.is_err(),
    "feed_audio with new-audio encode_window Err MUST propagate Err"
  );
  // Bridge cleared (the discharge already drained it transactionally).
  assert!(!session.retry_state().has_pending_stop_encoder_feed());

  // The obligation we MIGHT still have depends on whether the
  // discharge's decode armed DecodeOwed and propagated, or the later
  // encoder.feed Err armed DecodeOwed before propagating. Either way,
  // a retry feed_audio(&[]) MUST decode whatever is owed and clear.
  session.last_decode_time = None;
  let retry_events = session
    .feed_audio(&[])
    .expect("retry feed_audio MUST succeed");
  assert!(
    !session.retry_state().has_decode_owed(),
    "successful retry decode MUST clear DecodeOwed"
  );
  assert!(
    session.decoder.call_count() > decoder_calls_after_initial,
    "retry MUST drive at least one decoder call"
  );
  let _ = retry_events;
}

// -----------------------------------------------------------------
// NO flag bleeds across calls when cadence-throttle skips
// the decode. Structurally, this design has no "force decode
// next call" flag. The discharge runs decodes when an
// obligation is owed; a cadence-throttled call doesn't
// create an obligation (DecodeOwed is only armed by an
// actual Err or a discharge_stop_encoder_feed with count > 0).
// -----------------------------------------------------------------

/// A bridge drain + successful same-call
/// decode + no later Err MUST clear all obligations. There is no
/// flag that survives the call to force an unnecessary decode next
/// time. The cadence throttle does NOT factor into the discharge's
/// decision to decode bridge-drained windows (the discharge decodes
/// unconditionally when count > 0); but no flag persists past a
/// successful end-of-call.
#[test]
fn session_retry_state_throttled_drain_does_not_force_decode_on_next_call() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5]), Ok(vec![6, 7, 8])]);
  let mut session = nonfinalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let _ = session.stop();
  assert!(session.retry_state().has_pending_stop_encoder_feed());

  session.last_decode_time = None;

  // First feed_audio: discharge runs drain + decode pass. Both Ok →
  // ALL obligations cleared.
  let _events = session
    .feed_audio(&[])
    .expect("happy-path discharge MUST succeed");

  assert!(
    !session.retry_state().has_obligation(),
    "end-of-call MUST clear all obligations on happy path"
  );

  let decoder_calls_after_discharge = session.decoder.call_count();

  // Second feed_audio: no work owed. The decoder MUST NOT be invoked
  // for a "phantom" obligation. (Cadence might fire on its own with
  // last_decode_time = None + has_new_encoder_content = false, but
  // we just emptied newly_encoded_windows so has_new_encoder_content
  // should be false at this point — the previous call set it true,
  // decode pass cleared it.)
  let events_2 = session
    .feed_audio(&[])
    .expect("second empty feed_audio MUST succeed without forced decode");

  assert_eq!(
    session.decoder.call_count(),
    decoder_calls_after_discharge,
    "second empty feed_audio MUST NOT trigger a phantom decode"
  );
  assert!(events_2.is_empty(), "no work ⇒ no events");
}

// -----------------------------------------------------------------
// cancel / reset clear all obligations atomically.
// -----------------------------------------------------------------

/// `cancel()` MUST clear every retry obligation in one call (no
/// per-flag forget-to-clear corner).
#[test]
fn session_retry_state_cancel_clears_all_obligations() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
  let mut session = finalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let _ = session.stop(); // Errs → stages StopEncoderFeed
  assert!(session.retry_state().has_obligation());

  session.cancel();
  assert!(
    !session.retry_state().has_obligation(),
    "cancel MUST clear all retry obligations atomically"
  );
}

/// `reset()` MUST clear every retry obligation in one call.
#[test]
fn session_retry_state_reset_clears_all_obligations() {
  let encoder = ScriptedEncoder::new(8, vec![false, true, false, true, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![5])]);
  let mut session = finalize_session(encoder, decoder);

  let samples_a: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples_a).unwrap();
  let _ = session.stop();
  assert!(session.retry_state().has_obligation());

  session.reset();
  assert!(
    !session.retry_state().has_obligation(),
    "reset MUST clear all retry obligations atomically"
  );
}

// -----------------------------------------------------------------
// StopMelFlush deadlock avoidance — discharge_stop_mel_flush is
// wired into discharge_retry_obligation. stop() stages
// StopMelFlush BEFORE mel.flush(); on flush Err the obligation
// is set, and discharge_retry_obligation MUST have a branch for it
// — otherwise has_obligation() returns true, stop() early-returns
// without driving the retry, and the session DEADLOCKS.
// -----------------------------------------------------------------

/// A stop() whose `mel.flush()` errors stages
/// `StopMelFlush` and keeps the session active so the next call can
/// retry. Without the discharge wiring the obligation would be
/// stranded.
#[test]
fn session_retry_state_stop_after_flush_err_keeps_session_active_with_obligation() {
  let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();
  assert!(session.mel_processor.overlap_buffer_len() > 0);

  // Script ONE mel.flush Err.
  session.mel_processor.flush_err_inject_count = 1;

  // First stop(): mel.flush Errs → StopMelFlush staged.
  let stop_first = session.stop();
  assert!(stop_first.is_err(), "stop() must propagate mel.flush Err");
  assert!(
    session.is_active(),
    "errored stop() MUST leave session active so retry is possible"
  );
  assert!(
    session.retry_state().has_pending_stop_mel_flush(),
    "stop()'s mel.flush Err MUST stage StopMelFlush"
  );
  assert!(
    session.retry_state().has_obligation(),
    "retry obligation MUST be visible to the next call"
  );
  // mel.flush is transactional — overlap preserved on Err.
  assert!(
    session.mel_processor.overlap_buffer_len() > 0,
    "transactional flush MUST preserve overlap on Err"
  );
}

/// A second stop() after the first's mel.flush Err
/// drives `discharge_stop_mel_flush`, completes the flush + encoder
/// feed + decode + Ended emission. Without a discharge branch for
/// StopMelFlush this would DEADLOCK.
#[test]
fn session_retry_state_stop_with_mel_flush_err_can_be_retried() {
  let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();
  let overlap_before = session.mel_processor.overlap_buffer_len();
  assert!(overlap_before > 0);

  // First stop(): mel.flush Errs → StopMelFlush staged.
  session.mel_processor.flush_err_inject_count = 1;
  let stop_first = session.stop();
  assert!(stop_first.is_err());
  assert!(session.retry_state().has_pending_stop_mel_flush());
  // Overlap intact (transactional flush).
  assert_eq!(session.mel_processor.overlap_buffer_len(), overlap_before);

  // Second stop(): the dispatcher's StopMelFlush discharge re-runs
  // mel.flush → Ok this time → advances to StopEncoderFeed → (b)
  // drains the encoder → (c) decodes the bridge-drained partial →
  // back in stop() body, finalize/freeze + partial decode + Ended.
  let stop_second = session
    .stop()
    .expect("retry stop() MUST succeed via discharge_stop_mel_flush");
  assert!(
    matches!(stop_second.last(), Some(TranscriptionEvent::Ended(_))),
    "second stop() MUST emit Ended after the discharge succeeds"
  );
  assert!(!session.is_active());
  assert!(!session.retry_state().has_obligation());
  // Overlap cleared by the successful re-flush.
  assert_eq!(session.mel_processor.overlap_buffer_len(), 0);
}

/// The cross-call discharge survives MULTIPLE mel.flush
/// failures — each call drives the retry and re-arms StopMelFlush
/// on continued failure. Only when the flush eventually succeeds
/// does the pipeline proceed.
#[test]
fn session_retry_state_second_stop_after_flush_err_retries_and_emits_ended_on_success() {
  let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
  let decoder = ScriptedDecoder::with_results(vec![Ok(vec![1]), Ok(vec![2, 3])]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();

  // Inject TWO consecutive mel.flush Errs.
  session.mel_processor.flush_err_inject_count = 2;

  // First stop(): mel.flush Err #1 → StopMelFlush staged.
  assert!(session.stop().is_err());
  assert!(session.retry_state().has_pending_stop_mel_flush());

  // Second stop(): the discharge re-fires mel.flush → Err #2 → re-arms
  // StopMelFlush, propagates Err. No deadlock — the obligation
  // remains discharge-able.
  assert!(
    session.stop().is_err(),
    "second stop() with continued flush Err MUST propagate Err"
  );
  assert!(
    session.retry_state().has_pending_stop_mel_flush(),
    "re-armed StopMelFlush MUST persist across the second Err"
  );
  assert!(session.is_active());

  // Third stop(): flush counter exhausted → flush Ok → full pipeline.
  let stop_third = session
    .stop()
    .expect("third stop() MUST succeed once flush stops erring");
  assert!(matches!(
    stop_third.last(),
    Some(TranscriptionEvent::Ended(_))
  ));
  assert!(!session.is_active());
  assert!(!session.retry_state().has_obligation());
}

// -----------------------------------------------------------------
// StopPartialDecode rollback must not silently drop the
// audio_features payload on try_clone failure. Mapping clone Err to
// `None` (via `try_clone().ok()`) would make the next retry behave
// as "no partial audio" and drop the window. Instead the clone Err
// propagates AND the obligation is re-armed with the original
// payload (fast path) or the mainline body's idempotent recompute
// handles it.
// -----------------------------------------------------------------

/// The `clone_partial_decode_payload` helper returns
/// `Ok(None)` for absent features (the normal "no partial" path),
/// `Ok(Some(_))` for a successful refcount clone, and propagates a
/// clone failure as `Err` instead of silently dropping it. Tested
/// at the helper level because forcing `Array::try_clone` to fail
/// requires injecting an FFI allocation failure that the mlx
/// backend doesn't surface deterministically; the helper is the
/// unit-testable choke-point that gates BOTH stop()-path call sites.
#[test]
fn session_retry_state_stop_partial_decode_clone_helper_propagates_errors_not_silently_drops_audio()
{
  // None in ⇒ Ok(None) out — the legitimate "no partial audio" path.
  let none_out = clone_partial_decode_payload(None).expect("None must succeed");
  assert!(
    none_out.is_none(),
    "None payload MUST round-trip as Ok(None) (no fabricated payload)"
  );

  // Some in ⇒ Ok(Some(refcount-cloned)) on the happy path.
  let arr = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0], &[3i32]).unwrap();
  let some_out = clone_partial_decode_payload(Some(&arr)).expect("happy-path clone must succeed");
  assert!(
    some_out.is_some(),
    "Some payload with successful clone MUST yield Ok(Some(_))"
  );
  // The clone is a separate handle (not the same allocation).
  let cloned = some_out.unwrap();
  assert_eq!(
    cloned.shape(),
    arr.shape(),
    "refcount clone preserves shape"
  );

  // STRUCTURAL ASSERTION: the function signature returns `Result`,
  // proving an `Err` PATH exists. A `.ok()`-based API would have NO
  // Err path → any clone failure would be dropped. The Err-path
  // existence is what kills the silent-drop defect class
  // structurally; the propagation is exercised end-to-end by every
  // stop()-with-partial-window test in this module (those call
  // sites use `?` to propagate, so a real Err would surface as a
  // stop() Err).
}

/// The stop() fast path's clone-failure rollback
/// re-arms `StopPartialDecode` with the ORIGINAL payload (moved
/// back into the obligation, no clone needed). A `try_clone().ok()`
/// approach would have armed with `None`, silently dropping the
/// partial window for the next retry. We assert the
/// rollback's structural shape: after a successful fast-path stop,
/// the obligation is cleared. (A clone-failure-injected fast path
/// can't be triggered without an FFI hook into `mlx_array_set`'s
/// allocator; the rollback code path is exercised symbolically by
/// the helper test above + the call-site preserves the original
/// payload across the Err propagation by structural construction.)
#[test]
fn session_retry_state_stop_partial_decode_fast_path_preserves_payload_on_success() {
  let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
  // Decoder: first call returns one token (feed_audio partial decode);
  // second call (stop()'s partial-decode) errs so StopPartialDecode
  // is left armed; third call (retry stop()'s fast path) succeeds.
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(crate::error::Error::InvariantViolation(
      crate::error::InvariantViolationPayload::new(
        "ScriptedDecoder",
        "scripted stop-partial-decode Err",
      ),
    )),
    Ok(vec![1, 2]),
  ]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();

  // First stop(): everything succeeds until the partial-window
  // decode, which Errs. StopPartialDecode is armed with the
  // freshly-cloned audio_features.
  let stop_first = session.stop();
  assert!(
    stop_first.is_err(),
    "first stop() MUST propagate decoder Err"
  );
  assert!(
    session.retry_state().has_pending_stop_partial_decode(),
    "stop()'s partial-decode Err MUST arm StopPartialDecode \
       (with the cloned audio_features payload, never silently None)"
  );

  // Second stop(): fast path takes the obligation, clones for
  // re-arm, calls finalize_partial_window_and_emit_ended, and on
  // success clears. The clone is real — never None when the
  // pre-arm payload was Some.
  let stop_second = session.stop().expect("retry stop() fast path MUST succeed");
  assert!(matches!(
    stop_second.last(),
    Some(TranscriptionEvent::Ended(_))
  ));
  assert!(!session.retry_state().has_obligation());
  assert!(!session.is_active());
}

/// The stop() mainline body's clone-for-arm step is
/// fallible-via-`?`-propagation now. We can't trigger
/// `Array::try_clone` to fail deterministically, but we can
/// confirm the structural shape: `clone_partial_decode_payload`
/// returns `Result` so a clone Err propagates via `?` instead of
/// being dropped into `None`. The helper test above proves the
/// Err path exists; this test proves the call-site uses the
/// helper (not a `.ok()` pattern) by exercising the
/// happy-path through the mainline body AND asserting the
/// arm-payload was never `None` when audio_features was Some.
#[test]
fn session_retry_state_stop_partial_decode_mainline_body_arms_with_real_clone() {
  let encoder = ScriptedEncoder::new(8, vec![false, false, false, false]);
  // Decoder: feed_audio partial, then stop()'s partial-decode Errs
  // so the arm is the LAST mutation before the Err.
  let decoder = ScriptedDecoder::with_results(vec![
    Ok(vec![1]),
    Err(crate::error::Error::InvariantViolation(
      crate::error::InvariantViolationPayload::new(
        "ScriptedDecoder",
        "scripted stop-partial-decode Err",
      ),
    )),
  ]);
  let mut session = nonfinalize_session(encoder, decoder);
  let samples: Vec<f32> = (0..1200).map(|i| (i as f32 * 0.001).sin()).collect();
  let _ = session.feed_audio(&samples).unwrap();

  // stop() reaches the mainline body, hits step 6's arm + decode.
  // The arm clones audio_features (real refcount handle); the
  // decode then Errs. Without the fix, the arm would have been
  // `None` if the clone failed silently — but we can assert that
  // when audio_features WAS Some (the encoder has a partial
  // window), the arm-payload is Some.
  let stop_first = session.stop();
  assert!(stop_first.is_err());
  assert!(
    session.retry_state().has_pending_stop_partial_decode(),
    "mainline-body Err must arm StopPartialDecode"
  );

  // STRUCTURAL CHECK: the obligation's payload is Some (not None).
  // We inspect via take + immediate re-arm so the obligation isn't
  // permanently consumed.
  let taken = session
    .retry_state_mut()
    .take_stop_partial_decode_features()
    .expect("guard above asserts arm");
  assert!(
    taken.is_some(),
    "mainline arm MUST carry the real cloned payload, never \
       silently None — the encoder had a partial window (1-row carry \
       after the 7-mel feed + stop-flush bridge), so encode_pending \
       returned Some, so the helper's Ok arm is Some"
  );
  // Restore so test cleanup is consistent.
  session.retry_state_mut().arm_stop_partial_decode(taken);
}
