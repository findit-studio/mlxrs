//! Integration tests for `audio::stt::streaming` —
//! `IncrementalMelSpectrogram` + `StreamingEncoder` +
//! `StreamingInferenceSession` exercised through their public APIs.
//!
//! Mocks are deterministic and dependency-free: a no-op encoder
//! returns `(rows, 2)` zero arrays, and a record-and-canned decoder
//! returns a fixed token sequence per pass. The tests verify
//! orchestration shapes (events emitted, call counts / shapes,
//! lifecycle) — not per-architecture model math, which lives in user
//! code per the no-per-model-arch rule.

#![cfg(feature = "audio")]

use std::sync::Mutex;

use mlxrs::{
  Array,
  audio::stt::streaming::{
    DelayPreset, IncrementalMelSpectrogram, StreamingConfig, StreamingDecoderBackend,
    StreamingEncoder, StreamingEncoderBackend, StreamingInferenceSession, StreamingTokenizer,
    TranscriptionEvent,
  },
  error::Result,
};

// --- Mock encoder ------------------------------------------------------

struct MockEncoder {
  window_size: usize,
  calls: Mutex<Vec<usize>>,
}

impl MockEncoder {
  fn new(window_size: usize) -> Self {
    Self {
      window_size,
      calls: Mutex::new(Vec::new()),
    }
  }
}

impl StreamingEncoderBackend for MockEncoder {
  fn window_size(&self) -> usize {
    self.window_size
  }

  fn encode_window(&self, mel_window: &Array, _valid_frames: usize) -> Result<Array> {
    let rows = mel_window.shape().first().copied().unwrap_or(0);
    self.calls.lock().unwrap().push(rows);
    let buf = vec![0.0_f32; rows * 2];
    Array::from_slice::<f32>(&buf, &[rows as i32, 2i32])
  }
}

// --- Mock decoder ------------------------------------------------------

struct MockDecoder {
  /// Queue of token sequences returned per call (front-popped).
  tokens: Mutex<Vec<Vec<u32>>>,
  /// (rows, confirmed_count, max_tokens) per call.
  calls: Mutex<Vec<(usize, usize, usize)>>,
}

impl MockDecoder {
  fn with_sequence(seqs: Vec<Vec<u32>>) -> Self {
    Self {
      tokens: Mutex::new(seqs),
      calls: Mutex::new(Vec::new()),
    }
  }
}

impl StreamingDecoderBackend for MockDecoder {
  fn decode_all_tokens(
    &self,
    audio_features: &Array,
    confirmed_token_ids: &[u32],
    _config: &StreamingConfig,
    max_tokens: usize,
  ) -> Result<Vec<u32>> {
    let rows = audio_features.shape().first().copied().unwrap_or(0);
    self
      .calls
      .lock()
      .unwrap()
      .push((rows, confirmed_token_ids.len(), max_tokens));
    let mut queue = self.tokens.lock().unwrap();
    Ok(if queue.is_empty() {
      Vec::new()
    } else {
      queue.remove(0)
    })
  }
}

// --- Mock tokenizer ----------------------------------------------------

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

// --- Helpers -----------------------------------------------------------

fn promote_immediate_config(window_size_frames: usize) -> StreamingConfig {
  // Force decode every feed + immediate promote so the integration
  // tests don't need wall-clock waits.
  StreamingConfig {
    decode_interval_seconds: 0.0,
    boundary_decode_interval_seconds: 0.0,
    boundary_boost_seconds: 0.0,
    max_cached_windows: window_size_frames.max(1),
    finalize_completed_windows: false,
    min_agreement_passes: 1,
    boundary_min_agreement_passes: 1,
    delay_preset: DelayPreset::Custom(0),
    ..StreamingConfig::default()
  }
}

fn zero_mel(rows: usize, n_mels: usize) -> Array {
  let buf = vec![0.0_f32; rows * n_mels];
  Array::from_slice::<f32>(&buf, &[rows as i32, n_mels as i32]).unwrap()
}

// --- Tests -------------------------------------------------------------

#[test]
fn mel_spectrogram_full_session_emits_then_flushes() {
  let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();

  let samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
  let out = mel.process(&samples).unwrap();
  assert!(out.is_some(), "first chunk should emit");

  let total_after_first = mel.total_frames();
  let out2 = mel.process(&samples).unwrap();
  assert!(out2.is_some(), "second chunk should emit");
  assert!(mel.total_frames() > total_after_first);

  let flushed = mel.flush().unwrap();
  assert!(flushed.is_some(), "flush should drain remaining samples");
}

#[test]
fn mel_spectrogram_reset_then_replay_is_deterministic() {
  let mut mel = IncrementalMelSpectrogram::new(16_000, 32, 16, 8).unwrap();
  let samples: Vec<f32> = (0..200).map(|i| (i as f32 * 0.005).sin()).collect();
  let _ = mel.process(&samples).unwrap();
  let before = mel.total_frames();
  assert!(before > 0);
  mel.reset();
  assert_eq!(mel.total_frames(), 0);
  let _ = mel.process(&samples).unwrap();
  assert_eq!(mel.total_frames(), before);
}

#[test]
fn streaming_encoder_records_each_full_window_then_drains_them() {
  let encoder = MockEncoder::new(8);
  let mut stream = StreamingEncoder::new(encoder, 4, 0);

  // Feed 16 rows → 2 windows.
  let n = stream.feed(&zero_mel(16, 4)).unwrap();
  assert_eq!(n, 2);
  assert_eq!(stream.encoded_window_count(), 2);

  let drained = stream.drain_newly_encoded_windows();
  assert_eq!(drained.len(), 2);
  let second_drain = stream.drain_newly_encoded_windows();
  assert!(second_drain.is_empty());
}

#[test]
fn streaming_session_round_trip_emits_lifecycle_events() {
  let config = promote_immediate_config(4);
  let mut session = StreamingInferenceSession::new(
    MockDecoder::with_sequence(vec![vec![1, 2, 3], vec![1, 2, 3, 4]]),
    MockTokenizer,
    config,
    MockEncoder::new(16),
    16_000,
    8,
    0,
  )
  .unwrap();

  let samples: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.001).sin()).collect();
  let feed_events = session.feed_audio(&samples).unwrap();
  // Decoder was called once; events include Confirmed + DisplayUpdate +
  // Stats.
  assert!(
    !feed_events.is_empty(),
    "expected events from feed_audio, got {feed_events:?}"
  );
  assert!(
    feed_events
      .iter()
      .any(|e| matches!(e, TranscriptionEvent::Stats(_)))
  );

  let stop_events = session.stop().unwrap();
  assert!(
    matches!(stop_events.last(), Some(TranscriptionEvent::Ended { .. })),
    "stop must emit Ended last, got {stop_events:?}"
  );
  assert!(!session.is_active());
}

#[test]
fn streaming_session_short_input_is_silent_until_mel_emits() {
  let config = promote_immediate_config(4);
  let mut session = StreamingInferenceSession::new(
    MockDecoder::with_sequence(vec![]),
    MockTokenizer,
    config,
    MockEncoder::new(16),
    16_000,
    8,
    0,
  )
  .unwrap();

  // 1 sample is far below the mel frame budget — no events.
  let events = session.feed_audio(&[0.0_f32]).unwrap();
  assert!(events.is_empty());
  assert!(session.is_active());
}

#[test]
fn streaming_session_cancel_is_idempotent() {
  let config = promote_immediate_config(4);
  let mut session = StreamingInferenceSession::new(
    MockDecoder::with_sequence(vec![]),
    MockTokenizer,
    config,
    MockEncoder::new(16),
    16_000,
    8,
    0,
  )
  .unwrap();
  session.cancel();
  session.cancel(); // second call must not panic.
  assert!(!session.is_active());
  // Post-cancel feed_audio returns empty.
  let after = session.feed_audio(&[0.0_f32; 100]).unwrap();
  assert!(after.is_empty());
}

#[test]
fn streaming_session_stop_without_feed_emits_only_ended_event() {
  let config = promote_immediate_config(4);
  let mut session = StreamingInferenceSession::new(
    MockDecoder::with_sequence(vec![]),
    MockTokenizer,
    config,
    MockEncoder::new(16),
    16_000,
    8,
    0,
  )
  .unwrap();

  let events = session.stop().unwrap();
  assert!(
    matches!(events.last(), Some(TranscriptionEvent::Ended { .. })),
    "stop must always end with Ended event, got {events:?}"
  );
}
