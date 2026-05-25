//! A7 — `audio::tts::TtsModel` trait + `audio::tts::tts_generate` Iterator:
//! end-to-end text → audio-chunk synthesis driven by a mock TTS model.
//!
//! Deterministic, dependency-free: a local `MockTtsModel` emits, per text
//! segment, a `[k]` ramp waveform whose length encodes the segment's
//! 0-based index — so the produced `AudioChunk`s' sample counts, segment
//! ids, and `is_final_chunk` flags are all fully predictable. Mirrors the
//! `tests/audio_stt.rs` mock-model pattern (replicated, not imported —
//! integration tests cannot see crate-private fixtures).
#![cfg(feature = "audio")]

use std::cell::RefCell;

use mlxrs::{
  Array,
  audio::tts::{
    TextProcessor,
    generate::{
      AudioChunk, AudioFormat, DEFAULT_MAX_TOKENS, DEFAULT_STREAMING_INTERVAL, DEFAULT_TEMPERATURE,
      DEFAULT_VOICE, TextSegmentation, TtsGenConfig, TtsReference, TtsSegment, join_audio,
      join_audio_with_reference, tts_generate, tts_generate_with_reference,
    },
    model::TtsModel,
  },
};

/// One recorded `synthesize_segment` call: `(text, voice, language, speed,
/// temperature, segment_idx)`. A named alias keeps the [`MockTtsModel::seen`]
/// field below the clippy `type_complexity` threshold.
type SeenSegment = (String, String, String, f32, f32, usize);

/// A deterministic, dependency-free [`TtsModel`].
///
/// - `synthesize_segment` emits a rank-1 `[len]` ramp waveform where
///   `len = base_len + segment_idx` — so a test reading a chunk's
///   `len_samples()` can recover which segment produced it, and can assert
///   the driver visited segments in order.
/// - Records every segment it received (text + voice + language + speed +
///   segment_idx + temperature) so config plumbing is observable end-to-end.
struct MockTtsModel {
  sample_rate: u32,
  /// Sample count of segment 0's waveform; segment `i` emits `base_len + i`.
  base_len: usize,
  /// Every recorded [`SeenSegment`] the model was asked to synthesize, in
  /// call order.
  seen: RefCell<Vec<SeenSegment>>,
}

impl MockTtsModel {
  fn new(sample_rate: u32, base_len: usize) -> Self {
    Self {
      sample_rate,
      base_len,
      seen: RefCell::new(Vec::new()),
    }
  }
}

impl TtsModel for MockTtsModel {
  fn synthesize_segment(&self, segment: &TtsSegment<'_>) -> mlxrs::Result<Array> {
    self.seen.borrow_mut().push((
      segment.text().to_string(),
      segment.voice().to_string(),
      segment.language().to_string(),
      segment.speed(),
      segment.temperature(),
      segment.segment_idx(),
    ));
    // `[base_len + segment_idx]` ramp — the length encodes the segment id.
    let len = self.base_len + segment.segment_idx();
    let samples: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
    let n = i32::try_from(len).unwrap();
    Array::from_slice::<f32>(&samples, &[n])
  }

  fn sample_rate(&self) -> u32 {
    self.sample_rate
  }
}

/// A `TtsModel` that does NOT override `synthesize_segment` — drives the
/// default "needs `synthesize_segment` override" `Err`.
struct DefaultSynthModel;
impl TtsModel for DefaultSynthModel {
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

/// A `synthesize_segment` returning a malformed (rank-2) audio tensor —
/// drives the driver's "must be rank-1 `[samples]`" shape guard.
struct BadShapeModel;
impl TtsModel for BadShapeModel {
  fn synthesize_segment(&self, _segment: &TtsSegment<'_>) -> mlxrs::Result<Array> {
    // Wrong rank: [1, 4] not [4].
    Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &[1, 4])
  }
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

/// A `synthesize_segment` that always errors — drives the "segment error
/// yielded once, iterator fuses" contract.
struct FailSynthModel;
impl TtsModel for FailSynthModel {
  fn synthesize_segment(&self, _segment: &TtsSegment<'_>) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "mock synthesize_segment failure".into(),
    })
  }
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

/// What a [`RecordingRefModel`] saw for one segment's voice-clone reference:
/// `(ref_audio_is_some, ref_text)`. The `ref_audio` is recorded only as a
/// presence flag (an `Array` is not `Clone`); `ref_text` is captured verbatim.
type SeenRef = (bool, Option<String>);

/// A [`TtsModel`] that records the [`TtsSegment::ref_audio`] /
/// [`TtsSegment::ref_text`] it received on every segment — so a test can prove
/// the voice-clone reference is threaded through the public path onto each
/// segment (Fix 1).
struct RecordingRefModel {
  /// One [`SeenRef`] per `synthesize_segment` call, in order.
  seen_refs: RefCell<Vec<SeenRef>>,
}

impl RecordingRefModel {
  fn new() -> Self {
    Self {
      seen_refs: RefCell::new(Vec::new()),
    }
  }
}

impl TtsModel for RecordingRefModel {
  fn synthesize_segment(&self, segment: &TtsSegment<'_>) -> mlxrs::Result<Array> {
    self.seen_refs.borrow_mut().push((
      segment.ref_audio().is_some(),
      segment.ref_text().map(str::to_string),
    ));
    // A valid rank-1 f32 waveform so synthesis succeeds.
    Array::from_slice::<f32>(&[0.0_f32, 0.1, 0.2], &[3])
  }
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

/// A `synthesize_segment` returning a rank-1 but NON-`f32` (here `i32`) tensor
/// — drives the driver's "must be f32 PCM" dtype guard (Fix 2). The shape is
/// valid rank-1, so only the dtype check can reject it.
struct NonF32Model;
impl TtsModel for NonF32Model {
  fn synthesize_segment(&self, _segment: &TtsSegment<'_>) -> mlxrs::Result<Array> {
    // Rank-1 [4] i32 — passes the rank-1 shape check, fails the f32 check.
    Array::from_slice::<i32>(&[1_i32, 2, 3, 4], &[4])
  }
  fn sample_rate(&self) -> u32 {
    24_000
  }
}

// ───────────────────────── pipeline smoke ─────────────────────────

/// End-to-end smoke: a single-segment input yields exactly one chunk,
/// stamped with the model's sample rate, segment 0, `is_final_chunk = true`.
#[test]
fn tts_generate_single_segment_smoke() {
  let model = MockTtsModel::new(24_000, 100);
  let cfg = TtsGenConfig::default();
  let chunks: Vec<AudioChunk> = tts_generate(&model, "hello world", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 1, "one segment ⇒ one chunk");
  let c = &chunks[0];
  assert_eq!(c.segment_idx(), 0);
  assert_eq!(
    c.sample_rate(),
    24_000,
    "chunk stamped with model sample rate"
  );
  assert_eq!(c.len_samples(), 100, "segment 0 emits base_len samples");
  assert!(
    !c.is_streaming_chunk(),
    "driver yields whole-segment chunks"
  );
  assert!(c.is_final_chunk(), "the only chunk is the final one");
  assert!(!c.is_empty());
  // The model saw the segment text verbatim (no phonemization by the driver).
  assert_eq!(model.seen.borrow().len(), 1);
  assert_eq!(model.seen.borrow()[0].0, "hello world");
}

/// Multi-segment (newline-split) input: each newline-separated line becomes
/// one segment; chunks come out in order with monotone `segment_idx`, and
/// only the last carries `is_final_chunk`.
#[test]
fn tts_generate_splits_on_newlines() {
  let model = MockTtsModel::new(16_000, 10);
  let cfg = TtsGenConfig::default(); // segmentation = Newlines
  let chunks: Vec<AudioChunk> = tts_generate(&model, "first\nsecond\nthird", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 3, "three lines ⇒ three segments");
  // segment_idx 0,1,2 in order; lengths 10,11,12 (base_len + idx).
  for (i, c) in chunks.iter().enumerate() {
    assert_eq!(c.segment_idx(), i, "segment ids monotone");
    assert_eq!(
      c.len_samples(),
      10 + i,
      "segment i emits base_len+i samples"
    );
    assert_eq!(
      c.is_final_chunk(),
      i == 2,
      "only the last segment's chunk is final"
    );
  }
  // The model saw the three lines, in order, with the right segment_idx.
  let seen = model.seen.borrow();
  assert_eq!(seen[0].0, "first");
  assert_eq!(seen[1].0, "second");
  assert_eq!(seen[2].0, "third");
  assert_eq!((seen[0].5, seen[1].5, seen[2].5), (0, 1, 2));
}

/// Consecutive newlines (the `\n+` collapse) and leading/trailing blank
/// lines are dropped — only non-blank segments survive.
#[test]
fn tts_generate_drops_blank_segments() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  // Leading newline, a doubled newline, a whitespace-only line, trailing
  // newlines — only "alpha" and "beta" are non-blank.
  let chunks: Vec<AudioChunk> = tts_generate(&model, "\nalpha\n\n   \nbeta\n\n", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 2, "blank/whitespace segments dropped");
  let seen = model.seen.borrow();
  assert_eq!(seen[0].0, "alpha");
  assert_eq!(seen[1].0, "beta");
}

/// `TextSegmentation::Whole`: the entire input (newlines and all) is one
/// segment — no splitting.
#[test]
fn tts_generate_whole_segmentation_single_chunk() {
  let model = MockTtsModel::new(24_000, 7);
  let cfg = TtsGenConfig::new().with_segmentation(TextSegmentation::Whole);
  let chunks: Vec<AudioChunk> = tts_generate(&model, "line one\nline two", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 1, "Whole ⇒ no split");
  // The model saw the whole input including the embedded newline.
  assert_eq!(model.seen.borrow()[0].0, "line one\nline two");
}

/// Config plumbing: voice / language / speed / temperature set on the
/// `TtsGenConfig` reach the model verbatim through every `TtsSegment`.
#[test]
fn tts_generate_plumbs_config_to_segments() {
  let model = MockTtsModel::new(24_000, 3);
  let cfg = TtsGenConfig::new()
    .with_voice("bf_emma")
    .with_language("en-gb")
    .with_speed(1.25)
    .with_temperature(0.4);
  let _ = tts_generate(&model, "one\ntwo", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect::<Vec<_>>();
  let seen = model.seen.borrow();
  assert_eq!(seen.len(), 2);
  for s in seen.iter() {
    assert_eq!(s.1, "bf_emma", "voice plumbed");
    assert_eq!(s.2, "en-gb", "language plumbed");
    assert!((s.3 - 1.25).abs() < 1e-6, "speed plumbed");
    assert!((s.4 - 0.4).abs() < 1e-6, "temperature plumbed");
  }
}

// ───────────────────────── join_audio ─────────────────────────

/// `join_audio` concatenates every segment's waveform into one tensor whose
/// length is the sum of the per-segment lengths.
#[test]
fn join_audio_concatenates_all_segments() {
  let model = MockTtsModel::new(24_000, 10);
  let cfg = TtsGenConfig::default();
  // Three segments ⇒ lengths 10 + 11 + 12 = 33.
  let mut joined = join_audio(&model, "a\nb\nc", &cfg).unwrap();
  assert_eq!(joined.shape(), vec![33], "joined length = sum of segments");
  // The materialized samples are the three ramps back-to-back: segment 0
  // contributes [0.00], segment 1 [0.00, 0.01], … — the first element of
  // every ramp is 0.0, so index 0 and index 10 (start of segment 1) are 0.
  let pcm = joined.to_vec::<f32>().unwrap();
  assert_eq!(pcm.len(), 33);
  assert_eq!(pcm[0], 0.0, "segment 0 ramp starts at 0");
  assert_eq!(pcm[10], 0.0, "segment 1 ramp starts at 0 (offset 10)");
}

/// `join_audio` on a single-segment input returns that segment's waveform
/// directly (mlx-audio's `len(audio_chunks) > 1` guard — no pointless
/// one-element concatenate).
#[test]
fn join_audio_single_segment_no_concat() {
  let model = MockTtsModel::new(24_000, 42);
  let cfg = TtsGenConfig::default();
  let joined = join_audio(&model, "just one line", &cfg).unwrap();
  assert_eq!(joined.shape(), vec![42], "single segment returned as-is");
}

/// `join_audio` propagates the first segment error.
#[test]
fn join_audio_propagates_segment_error() {
  let model = FailSynthModel;
  let cfg = TtsGenConfig::default();
  // `join_audio` returns `Result<Array>` (Array is Debug) so `expect_err`
  // applies here; the `tts_generate`-returning cases below keep
  // `.err().expect()` because `TtsGenerator` is not Debug.
  let err = join_audio(&model, "boom", &cfg).expect_err("synthesize failure propagates");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("synthesize_segment failure"),
        "got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
}

// ─────────────────── voice-clone reference (Fix 1) ───────────────────

/// `tts_generate_with_reference` threads the [`TtsReference`]'s `ref_audio` /
/// `ref_text` onto EVERY segment: a 3-segment input with a `Some` reference
/// makes the model see `Some` for BOTH on all three segments.
#[test]
fn tts_generate_threads_reference_to_every_segment() {
  let model = RecordingRefModel::new();
  let cfg = TtsGenConfig::default();
  // A rank-1 f32 reference waveform + a transcript.
  let ref_wav = Array::from_slice::<f32>(&[0.5_f32, -0.5, 0.25, -0.25], &[4]).unwrap();
  let reference = TtsReference::new(Some(&ref_wav), Some("the reference transcript"));
  // Three newline-split segments.
  let _ = tts_generate_with_reference(&model, "one\ntwo\nthree", &cfg, reference)
    .unwrap()
    .map(|r| r.unwrap())
    .collect::<Vec<_>>();
  let seen = model.seen_refs.borrow();
  assert_eq!(seen.len(), 3, "every segment synthesized");
  for (i, (has_audio, text)) in seen.iter().enumerate() {
    assert!(has_audio, "segment {i} received ref_audio = Some");
    assert_eq!(
      text.as_deref(),
      Some("the reference transcript"),
      "segment {i} received the ref_text verbatim"
    );
  }
}

/// A reference carrying ONLY `ref_audio` (no transcript) is threaded as such:
/// every segment sees `ref_audio = Some`, `ref_text = None` (the per-model
/// code would transcribe it). Proves the two fields are independently
/// optional.
#[test]
fn tts_generate_threads_audio_only_reference() {
  let model = RecordingRefModel::new();
  let cfg = TtsGenConfig::default();
  let ref_wav = Array::from_slice::<f32>(&[0.1_f32, 0.2], &[2]).unwrap();
  let reference = TtsReference::new(Some(&ref_wav), None);
  let _ = tts_generate_with_reference(&model, "alpha\nbeta", &cfg, reference)
    .unwrap()
    .map(|r| r.unwrap())
    .collect::<Vec<_>>();
  let seen = model.seen_refs.borrow();
  assert_eq!(seen.len(), 2);
  for (has_audio, text) in seen.iter() {
    assert!(has_audio, "ref_audio threaded");
    assert!(text.is_none(), "ref_text stays None (audio-only reference)");
  }
}

/// Back-compat: the plain `tts_generate` (no reference argument) makes the
/// model see `None` for BOTH ref fields on every segment — a non-cloning run.
#[test]
fn tts_generate_no_reference_passes_none() {
  let model = RecordingRefModel::new();
  let cfg = TtsGenConfig::default();
  let _ = tts_generate(&model, "one\ntwo", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect::<Vec<_>>();
  let seen = model.seen_refs.borrow();
  assert_eq!(seen.len(), 2);
  for (has_audio, text) in seen.iter() {
    assert!(!has_audio, "no reference ⇒ ref_audio = None");
    assert!(text.is_none(), "no reference ⇒ ref_text = None");
  }
}

/// `TtsReference::default()` is a both-`None` (non-cloning) reference, and
/// `tts_generate_with_reference` with it behaves exactly like `tts_generate`.
#[test]
fn tts_reference_default_is_non_cloning() {
  let model = RecordingRefModel::new();
  let cfg = TtsGenConfig::default();
  let _ = tts_generate_with_reference(&model, "x\ny", &cfg, TtsReference::default())
    .unwrap()
    .map(|r| r.unwrap())
    .collect::<Vec<_>>();
  let seen = model.seen_refs.borrow();
  assert_eq!(seen.len(), 2);
  for (has_audio, text) in seen.iter() {
    assert!(
      !has_audio && text.is_none(),
      "default reference is None/None"
    );
  }
}

/// `join_audio_with_reference` also threads the reference (proves the join
/// entry point carries it too, not just the iterator one).
#[test]
fn join_audio_with_reference_threads_reference() {
  let model = RecordingRefModel::new();
  let cfg = TtsGenConfig::default();
  let ref_wav = Array::from_slice::<f32>(&[0.3_f32, 0.4, 0.5], &[3]).unwrap();
  let reference = TtsReference::new(Some(&ref_wav), Some("caption"));
  // RecordingRefModel emits a [3] f32 waveform per segment ⇒ 2 segments = [6].
  let joined = join_audio_with_reference(&model, "p\nq", &cfg, reference).unwrap();
  assert_eq!(joined.shape(), vec![6], "two [3] segments joined to [6]");
  let seen = model.seen_refs.borrow();
  assert_eq!(seen.len(), 2);
  for (has_audio, text) in seen.iter() {
    assert!(has_audio, "join_audio_with_reference threads ref_audio");
    assert_eq!(text.as_deref(), Some("caption"), "and ref_text");
  }
}

// ──────────────────── f32 PCM dtype enforcement (Fix 2) ────────────────────

/// A rank-1 but NON-`f32` (`i32`) audio tensor from the model surfaces a
/// recoverable `Err(DtypeMismatch)` at the generator boundary — NOT a
/// successful chunk. The shape is valid rank-1, so only the dtype guard can
/// reject it.
#[test]
fn tts_generate_rejects_non_f32_audio_dtype() {
  let model = NonF32Model;
  let cfg = TtsGenConfig::default();
  let mut it = tts_generate(&model, "hi", &cfg).unwrap();
  match it.next().expect("an item") {
    Err(mlxrs::Error::DtypeMismatch { expected, got }) => {
      assert_eq!(expected, mlxrs::Dtype::F32, "f32 expected");
      assert_eq!(got, mlxrs::Dtype::I32, "actual dtype named (i32)");
    }
    other => panic!("expected DtypeMismatch, got {other:?}"),
  }
  assert!(it.next().is_none(), "iterator fuses after the dtype Err");
}

/// `join_audio` over a non-`f32` model NEVER returns a non-`f32` tensor — it
/// propagates the dtype `Err` instead. (Guards the `AudioChunk` /
/// `join_audio` f32 invariant.)
#[test]
fn join_audio_rejects_non_f32_audio_dtype() {
  let model = NonF32Model;
  let cfg = TtsGenConfig::default();
  let err = join_audio(&model, "x\ny\nz", &cfg).expect_err("non-f32 audio rejected");
  assert!(
    matches!(err, mlxrs::Error::DtypeMismatch { .. }),
    "got {err:?}"
  );
}

/// A valid `f32` model still synthesizes successfully through the dtype guard
/// (the guard rejects only non-f32 — it must not break the happy path).
#[test]
fn tts_generate_accepts_f32_audio_dtype() {
  // MockTtsModel emits f32 ramps.
  let model = MockTtsModel::new(24_000, 8);
  let cfg = TtsGenConfig::default();
  let chunks: Vec<AudioChunk> = tts_generate(&model, "ok", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 1, "valid f32 audio ⇒ a successful chunk");
  assert_eq!(chunks[0].len_samples(), 8);
  assert_eq!(
    chunks[0].audio_ref().dtype().unwrap(),
    mlxrs::Dtype::F32,
    "the chunk's audio is f32"
  );
}

// ───────────────────────── edge cases ─────────────────────────

/// Empty input text: `tts_generate` returns a recoverable `Err` — there is
/// nothing to synthesize, the model is NEVER called.
#[test]
fn tts_generate_rejects_empty_text() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  let err = tts_generate(&model, "", &cfg)
    .err()
    .expect("empty text rejected");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(message.contains("non-blank"), "got {message}");
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert!(
    model.seen.borrow().is_empty(),
    "model NOT called for empty text"
  );
}

/// All-whitespace / all-newline input: same as empty — no non-blank
/// segments, recoverable `Err`, model never called.
#[test]
fn tts_generate_rejects_whitespace_only_text() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  let err = tts_generate(&model, "   \n\n  \t \n", &cfg)
    .err()
    .expect("whitespace-only text rejected");
  assert!(matches!(err, mlxrs::Error::Backend { .. }));
  assert!(model.seen.borrow().is_empty());
}

/// Over-`MAX_TEXT_BYTES` input is rejected up front (pre-allocation cap) —
/// the model is never called.
#[test]
fn tts_generate_rejects_oversized_text() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  // 1 MiB + 1 byte of 'a'.
  let big = "a".repeat(1024 * 1024 + 1);
  let err = tts_generate(&model, &big, &cfg)
    .err()
    .expect("oversized text rejected");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("cap"),
        "error mentions the cap, got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert!(
    model.seen.borrow().is_empty(),
    "model NOT called for oversized text"
  );
}

/// A zero-length waveform from `synthesize_segment` is a *valid* rank-1
/// tensor (`[0]`) — the chunk reports `is_empty()` / `len_samples() == 0`
/// and `duration_seconds() == 0.0`, iteration still terminates cleanly.
#[test]
fn tts_generate_handles_zero_length_output() {
  // base_len = 0 ⇒ segment 0 emits a `[0]` waveform.
  let model = MockTtsModel::new(24_000, 0);
  let cfg = TtsGenConfig::default();
  let chunks: Vec<AudioChunk> = tts_generate(&model, "silent", &cfg)
    .unwrap()
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(chunks.len(), 1);
  assert!(chunks[0].is_empty(), "zero-length waveform ⇒ empty chunk");
  assert_eq!(chunks[0].len_samples(), 0);
  assert_eq!(chunks[0].duration_seconds(), 0.0);
  assert!(chunks[0].is_final_chunk(), "still the final chunk");
}

/// The trait-default `synthesize_segment` returns a recoverable `Err` with
/// the "needs override" message — the iterator yields it as the first item,
/// then fuses.
#[test]
fn synthesize_segment_default_errors_with_clear_message() {
  let model = DefaultSynthModel;
  let cfg = TtsGenConfig::default();
  let mut it = tts_generate(&model, "hi", &cfg).unwrap();
  match it.next().expect("an item") {
    Err(mlxrs::Error::Backend { message }) => {
      assert!(
        message.contains("synthesize_segment"),
        "error mentions synthesize_segment, got {message}"
      );
    }
    other => panic!("expected Backend Err, got {other:?}"),
  }
  assert!(it.next().is_none(), "iterator fuses after the Err");
}

/// A malformed (non-rank-1) audio tensor from the model surfaces a
/// recoverable `Err(ShapeMismatch)` — the driver's audio-shape guard.
#[test]
fn tts_generate_rejects_bad_audio_shape() {
  let model = BadShapeModel;
  let cfg = TtsGenConfig::default();
  let mut it = tts_generate(&model, "hi", &cfg).unwrap();
  match it.next().expect("an item") {
    Err(mlxrs::Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("rank-1"),
        "error mentions rank-1, got {message}"
      );
    }
    other => panic!("expected ShapeMismatch, got {other:?}"),
  }
  assert!(it.next().is_none(), "iterator fuses after the Err");
}

/// A `synthesize_segment` error is yielded once and the iterator fuses —
/// the same contract the STT / LM loops guarantee. With a 3-segment input
/// the error on segment 0 must prevent segments 1 and 2 from being
/// synthesized.
#[test]
fn tts_generate_segment_error_fuses() {
  let model = FailSynthModel;
  let cfg = TtsGenConfig::default();
  let mut it = tts_generate(&model, "one\ntwo\nthree", &cfg).unwrap();
  let first = it.next().expect("an item");
  assert!(first.is_err(), "segment error yielded as Err");
  assert!(
    it.next().is_none(),
    "iteration ends after the error (no panic, no re-entry)"
  );
}

/// `segment_count` reports the number of segments before iteration.
#[test]
fn tts_generator_reports_segment_count() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  let it = tts_generate(&model, "a\nb\nc\nd", &cfg).unwrap();
  assert_eq!(it.segment_count(), 4, "four non-blank segments");
}

/// The iterator's `size_hint` upper bound is the segment count and shrinks
/// as chunks are produced.
#[test]
fn tts_generator_size_hint_tracks_remaining() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  let mut it = tts_generate(&model, "a\nb\nc", &cfg).unwrap();
  assert_eq!(it.size_hint(), (0, Some(3)), "3 segments remaining");
  let _ = it.next().unwrap().unwrap();
  assert_eq!(it.size_hint(), (0, Some(2)), "2 remaining after one chunk");
}

// ───────────────────────── chunk accessors ─────────────────────────

/// `AudioChunk::duration_seconds` is `len_samples / sample_rate`.
#[test]
fn audio_chunk_duration_seconds() {
  // 24000-sample waveform at 24 kHz ⇒ exactly 1.0 second.
  let model = MockTtsModel::new(24_000, 24_000);
  let cfg = TtsGenConfig::default();
  let chunk = tts_generate(&model, "one second", &cfg)
    .unwrap()
    .next()
    .unwrap()
    .unwrap();
  assert!(
    (chunk.duration_seconds() - 1.0).abs() < 1e-9,
    "24000 samples / 24000 Hz = 1.0s, got {}",
    chunk.duration_seconds()
  );
}

/// `AudioChunk::samples` materializes the audio into an owned `Vec<f32>` —
/// the explicit-eval `&mut` step.
#[test]
fn audio_chunk_samples_materializes_pcm() {
  let model = MockTtsModel::new(24_000, 5);
  let cfg = TtsGenConfig::default();
  let mut chunk = tts_generate(&model, "hi", &cfg)
    .unwrap()
    .next()
    .unwrap()
    .unwrap();
  let pcm = chunk.samples().unwrap();
  // base_len = 5 ⇒ ramp [0.00, 0.01, 0.02, 0.03, 0.04].
  assert_eq!(pcm.len(), 5);
  assert_eq!(pcm[0], 0.0);
  assert!((pcm[4] - 0.04).abs() < 1e-6);
}

// ───────────────────────── config defaults ─────────────────────────

/// `TtsGenConfig::default()` carries the mlx-audio `generate_audio`
/// defaults.
#[test]
fn tts_gen_config_defaults_match_mlx_audio() {
  let c = TtsGenConfig::default();
  assert_eq!(c.voice(), DEFAULT_VOICE);
  assert_eq!(c.voice(), "af_heart");
  assert_eq!(c.language(), "en");
  assert!((c.speed() - 1.0).abs() < 1e-6, "speed default 1.0");
  assert!(
    (c.temperature() - DEFAULT_TEMPERATURE).abs() < 1e-6,
    "temperature default 0.7"
  );
  assert_eq!(c.top_p(), 0.0, "top_p default off");
  assert_eq!(c.top_k(), 0, "top_k default off");
  assert!(
    c.repetition_penalty().is_none(),
    "repetition_penalty default off"
  );
  assert_eq!(c.max_tokens(), DEFAULT_MAX_TOKENS);
  assert_eq!(c.max_tokens(), 1200);
  assert_eq!(c.segmentation(), TextSegmentation::Newlines);
  assert_eq!(c.audio_format(), AudioFormat::Wav);
  assert!(
    (c.streaming_interval() - DEFAULT_STREAMING_INTERVAL).abs() < 1e-6,
    "streaming_interval default 2.0"
  );
}

/// `AudioFormat` and `TextSegmentation` defaults.
#[test]
fn tts_enum_defaults() {
  assert_eq!(AudioFormat::default(), AudioFormat::Wav);
  assert_eq!(TextSegmentation::default(), TextSegmentation::Newlines);
}

/// A model's `default_config` falls back to `TtsGenConfig::default()` when
/// not overridden.
#[test]
fn tts_model_default_config_is_gen_config_default() {
  let model = MockTtsModel::new(24_000, 5);
  assert_eq!(model.default_config(), TtsGenConfig::default());
}

// ───────────────────────── TextProcessor hook ─────────────────────────

/// A trivial uppercase `TextProcessor` — proves the hook trait is callable
/// and the default `prepare()` is a no-op.
struct UppercaseProcessor {
  prepared: bool,
}

impl TextProcessor for UppercaseProcessor {
  fn prepare(&mut self) -> mlxrs::Result<()> {
    self.prepared = true;
    Ok(())
  }
  fn process(&self, text: &str, language: Option<&str>) -> mlxrs::Result<String> {
    // Echo the language into the output so the test can prove it was passed.
    let lang = language.unwrap_or("none");
    Ok(format!("{}|{}", text.to_uppercase(), lang))
  }
}

#[test]
fn text_processor_hook_processes_and_prepares() {
  let mut p = UppercaseProcessor { prepared: false };
  p.prepare().unwrap();
  assert!(p.prepared, "prepare() ran");
  let out = p.process("hello", Some("en-us")).unwrap();
  assert_eq!(out, "HELLO|en-us", "process applied + language plumbed");
  let out_no_lang = p.process("hi", None).unwrap();
  assert_eq!(out_no_lang, "HI|none", "None language handled");
}

/// The trait-default `prepare()` is a no-op (a processor that does not
/// override it still works).
#[test]
fn text_processor_default_prepare_is_noop() {
  struct NoPrepProcessor;
  impl TextProcessor for NoPrepProcessor {
    fn process(&self, text: &str, _language: Option<&str>) -> mlxrs::Result<String> {
      Ok(text.to_string())
    }
  }
  let mut p = NoPrepProcessor;
  // Default prepare() returns Ok without doing anything.
  assert!(p.prepare().is_ok());
  assert_eq!(p.process("x", None).unwrap(), "x");
}
