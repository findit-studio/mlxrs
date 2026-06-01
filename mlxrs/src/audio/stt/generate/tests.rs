//! Tests for the STT decoding drivers: the CTC greedy-collapse driver
//! ([`greedy_ctc_transcribe`]) and the autoregressive [`greedy_transcribe`]
//! loop, plus the shared waveform helpers ([`default_log_mel`],
//! [`resample_waveform`], the metadata validation gate).
//!
//! Oracles are computed from the TEST INPUTS (the scripted token sequences,
//! the hand-built logits), never from the code under test.

use std::cell::Cell;

use super::*;
use crate::{
  audio::{
    dsp::LogFloor,
    stt::model::{
      AutoregressiveStt, CtcModel, MelConfig, Task, Transcribe, TranscribeOptions, Transcription,
    },
  },
  error::Error,
};

// ===========================================================================
// CTC family — the `greedy_ctc_transcribe` free function, delegated to from a
// model's own `Transcribe` impl.
// ===========================================================================

/// A CTC mock returning a fixed `(T', vocab)` logit grid. `decode_ids` maps
/// each surviving id `i` to the character `(b'a' + i)` so the collapsed text
/// is a directly-checkable oracle. Opts into the greedy decode by delegating
/// to `greedy_ctc_transcribe` from its own `Transcribe` impl.
struct MockCtcModel {
  logits: Array,
  blank_id: u32,
}

impl MockCtcModel {
  /// Build a `(T', vocab)` grid whose per-frame argmax is exactly
  /// `argmax_per_frame[t]` (that class gets logit `1.0`, the rest `0.0`).
  fn from_argmax(argmax_per_frame: &[u32], vocab: usize, blank_id: u32) -> Self {
    let t = argmax_per_frame.len();
    let mut data = vec![0.0_f32; t * vocab];
    for (frame, &cls) in argmax_per_frame.iter().enumerate() {
      data[frame * vocab + cls as usize] = 1.0;
    }
    let logits = Array::from_slice::<f32>(&data, &[t as i32, vocab as i32]).unwrap();
    Self { logits, blank_id }
  }
}

impl CtcModel for MockCtcModel {
  fn logits(&self, _waveform: &Array) -> Result<Array> {
    self.logits.try_clone()
  }
  fn blank_id(&self) -> u32 {
    self.blank_id
  }
  fn decode_ids(&self, ids: &[u32]) -> String {
    ids.iter().map(|&i| (b'a' + i as u8) as char).collect()
  }
}

impl Transcribe for MockCtcModel {
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription> {
    greedy_ctc_transcribe(self, audio, opts)
  }
}

/// A non-empty mono placeholder waveform for the CTC mocks. Its CONTENT is never
/// inspected (every CTC mock's `logits` ignores the waveform and returns its own
/// scripted grid); it only needs to pass `validate_waveform` and to carry enough
/// samples that a mock's hand-built logits frame count `T'` satisfies the
/// driver's `T' <= input samples` tie (the largest functional grid here is 8
/// frames). 64 samples is a comfortable margin; tests that specifically exercise
/// the `T' > samples` rejection build their own small waveform inline.
fn dummy_waveform() -> Array {
  let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();
  Array::from_slice::<f32>(&data, &[data.len() as i32]).unwrap()
}

#[test]
fn ctc_collapses_dups_and_blanks() {
  // vocab 4, blank = 3. Frame argmax sequence (a=0, b=1, c=2, _=blank):
  //   a a _ a b b _ c
  // CTC greedy: collapse consecutive dups -> a _ a b _ c ; drop blank -> a a b c.
  let blank = 3;
  let frames = [0, 0, blank, 0, 1, 1, blank, 2];
  let model = MockCtcModel::from_argmax(&frames, 4, blank);

  // Routes through the model's own `Transcribe` impl (delegating to
  // `greedy_ctc_transcribe`).
  let out = model
    .transcribe(&dummy_waveform(), &TranscribeOptions::new())
    .expect("ctc transcribe");

  assert_eq!(out.text(), "aabc");
  assert!(out.language().is_none());
  // CTC carries one untimed segment spanning the whole utterance.
  assert_eq!(out.segments_slice().len(), 1);
  assert_eq!(out.segments_slice()[0].text(), "aabc");
}

#[test]
fn ctc_all_blank_is_empty() {
  let blank = 0;
  let frames = [0, 0, 0];
  let model = MockCtcModel::from_argmax(&frames, 2, blank);

  let out = model
    .transcribe(&dummy_waveform(), &TranscribeOptions::new())
    .expect("ctc transcribe");
  assert_eq!(out.text(), "");
  assert_eq!(out.segments_slice().len(), 1);
}

#[test]
fn ctc_rejects_non_rank2_logits() {
  // A model handing back rank-1 logits is a per-model defect -> typed error.
  let err =
    greedy_ctc_transcribe(&BadCtc, &dummy_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
}

/// A CTC mock whose `logits` shape is parameterized, to drive the encoder-
/// output shape guards (rank, empty-vocab, empty-time) in `greedy_ctc_transcribe`.
struct BadCtc;
impl CtcModel for BadCtc {
  fn logits(&self, _waveform: &Array) -> Result<Array> {
    // Rank-1 -> typed RankMismatch.
    Array::from_slice::<f32>(&[0.0_f32, 1.0], &[2])
  }
  fn blank_id(&self) -> u32 {
    0
  }
  fn decode_ids(&self, _ids: &[u32]) -> String {
    String::new()
  }
}

#[test]
fn ctc_rejects_empty_vocab_axis() {
  // `(T', 0)` logits: argmax over an empty vocab axis is undefined -> typed
  // EmptyInput (mirrors the autoregressive empty-vocab guard).
  struct EmptyVocabCtc;
  impl CtcModel for EmptyVocabCtc {
    fn logits(&self, _waveform: &Array) -> Result<Array> {
      // 3 frames, 0-wide vocab.
      Array::from_slice::<f32>(&[] as &[f32], &[3, 0])
    }
    fn blank_id(&self) -> u32 {
      0
    }
    fn decode_ids(&self, _ids: &[u32]) -> String {
      String::new()
    }
  }
  let err = greedy_ctc_transcribe(&EmptyVocabCtc, &dummy_waveform(), &TranscribeOptions::new())
    .unwrap_err();
  assert!(matches!(err, Error::EmptyInput(_)));
}

#[test]
fn ctc_empty_time_axis_is_empty_transcription() {
  // `(0, vocab)` logits: no frames -> an explicit empty transcription (not a
  // panic, not an error).
  struct EmptyTimeCtc;
  impl CtcModel for EmptyTimeCtc {
    fn logits(&self, _waveform: &Array) -> Result<Array> {
      Array::from_slice::<f32>(&[] as &[f32], &[0, 4])
    }
    fn blank_id(&self) -> u32 {
      0
    }
    fn decode_ids(&self, _ids: &[u32]) -> String {
      // Must never be called with a non-empty id slice here.
      String::new()
    }
  }
  let out = greedy_ctc_transcribe(&EmptyTimeCtc, &dummy_waveform(), &TranscribeOptions::new())
    .expect("empty-time CTC is an empty transcription");
  assert_eq!(out.text(), "");
  assert_eq!(out.segments_slice().len(), 1);
  assert_eq!(out.segments_slice()[0].text(), "");
}

/// A CTC mock whose `decode_ids(&[])` (the EMPTY id slice) returns a non-empty
/// sentinel `"<empty>"`. Its `logits` shape is scriptable so one model instance
/// drives both the empty-time `(0, vocab)` and the all-blank `(T>0, vocab)`
/// collapse paths; both must collapse to zero surviving ids and so must reach
/// `decode_ids(&[])`, proving the two "no surviving ids" paths render the SAME
/// text rather than one hard-coding `String::new()`.
struct SentinelCtc {
  logits: Array,
  blank_id: u32,
}

impl SentinelCtc {
  /// All-blank grid: `t` frames, `vocab`-wide, every frame's argmax == `blank`
  /// (so the collapse drops every id and the survivor set is empty). `t == 0`
  /// yields the empty-time `(0, vocab)` grid.
  fn all_blank(t: usize, vocab: usize, blank: u32) -> Self {
    let mut data = vec![0.0_f32; t * vocab];
    for frame in 0..t {
      data[frame * vocab + blank as usize] = 1.0;
    }
    let logits = Array::from_slice::<f32>(&data, &[t as i32, vocab as i32]).unwrap();
    Self {
      logits,
      blank_id: blank,
    }
  }
}

impl CtcModel for SentinelCtc {
  fn logits(&self, _waveform: &Array) -> Result<Array> {
    self.logits.try_clone()
  }
  fn blank_id(&self) -> u32 {
    self.blank_id
  }
  fn decode_ids(&self, ids: &[u32]) -> String {
    // A model whose detokenizer emits a sentinel for the empty id sequence: if
    // the empty-time branch hard-coded `String::new()` instead of routing
    // through here, the two empty-collapse paths would disagree.
    if ids.is_empty() {
      "<empty>".to_string()
    } else {
      ids.iter().map(|&i| (b'a' + i as u8) as char).collect()
    }
  }
}

#[test]
fn ctc_empty_time_and_all_blank_render_identical_via_decode_ids() {
  // Two semantically-identical "no surviving ids" inputs must produce the SAME
  // transcript because BOTH route their empty collapse through
  // `decode_ids(&[])`:
  //   (a) empty-time `(0, vocab)` — no frames at all.
  //   (b) all-blank `(T>0, vocab)` — frames exist but every argmax is the blank
  //       id, so the collapse leaves zero survivors.
  // With a model whose `decode_ids(&[])` returns the sentinel "<empty>", both
  // paths must yield "<empty>" (NOT an empty string for the empty-time path).
  let blank = 0;
  let vocab = 4;

  let empty_time = SentinelCtc::all_blank(0, vocab, blank);
  let out_empty = greedy_ctc_transcribe(&empty_time, &dummy_waveform(), &TranscribeOptions::new())
    .expect("empty-time CTC");

  let all_blank = SentinelCtc::all_blank(5, vocab, blank);
  let out_blank = greedy_ctc_transcribe(&all_blank, &dummy_waveform(), &TranscribeOptions::new())
    .expect("all-blank CTC");

  // Both reach `decode_ids(&[])` -> the sentinel, and they agree.
  assert_eq!(out_empty.text(), "<empty>");
  assert_eq!(out_blank.text(), "<empty>");
  assert_eq!(out_empty.text(), out_blank.text());
  // The single untimed segment carries the same sentinel text.
  assert_eq!(out_empty.segments_slice().len(), 1);
  assert_eq!(out_empty.segments_slice()[0].text(), "<empty>");
  assert_eq!(out_blank.segments_slice()[0].text(), "<empty>");
}

#[test]
fn ctc_rejects_rank2_waveform() {
  // A 2-D waveform is rejected (RankMismatch) BEFORE the encoder forward — it
  // is NOT silently flattened to mono.
  let model = MockCtcModel::from_argmax(&[0, 1], 3, 2);
  let stereo = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
  let err = greedy_ctc_transcribe(&model, &stereo, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
}

#[test]
fn ctc_rejects_empty_waveform() {
  // An empty waveform is rejected (EmptyInput) BEFORE the encoder forward.
  let model = MockCtcModel::from_argmax(&[0, 1], 3, 2);
  let empty = Array::from_slice::<f32>(&[] as &[f32], &[0]).unwrap();
  let err = greedy_ctc_transcribe(&model, &empty, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::EmptyInput(_)));
}

#[test]
fn ctc_rejects_blank_id_outside_vocab() {
  // A `blank_id` >= the logits vocab size can never equal a per-frame argmax,
  // so its "blank" frames would survive the collapse and feed `decode_ids` ->
  // silent bad text. The driver rejects it with a typed OutOfRange BEFORE the
  // argmax/collapse. vocab = 3 (valid ids 0..=2), blank_id = 3 (out of range).
  let model = MockCtcModel::from_argmax(&[0, 1, 2], 3, 3);
  // Routes through the model's own `Transcribe` impl (delegating to
  // `greedy_ctc_transcribe`), confirming the guard fires on the public path.
  let err = model
    .transcribe(&dummy_waveform(), &TranscribeOptions::new())
    .unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));

  // Also reject a blank id far past the vocab axis, via the free function.
  let model_far = MockCtcModel::from_argmax(&[0, 1], 3, 99);
  let err_far =
    greedy_ctc_transcribe(&model_far, &dummy_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err_far, Error::OutOfRange(_)));
}

/// A CTC mock whose `blank_id` is STATEFUL: the first call returns
/// `first_blank`, every later call returns `later_blank`. Both go through the
/// `&self` method (a `Cell` provides the interior mutability), modelling a
/// hostile/buggy model that could answer the driver's range check and its
/// collapse with different blank ids. `from_argmax`-style fixed logits and the
/// `(b'a' + i)` `decode_ids` map make the collapsed text a direct oracle.
struct StatefulBlankCtc {
  logits: Array,
  first_blank: u32,
  later_blank: u32,
  calls: Cell<u32>,
}

impl StatefulBlankCtc {
  fn from_argmax(
    argmax_per_frame: &[u32],
    vocab: usize,
    first_blank: u32,
    later_blank: u32,
  ) -> Self {
    let t = argmax_per_frame.len();
    let mut data = vec![0.0_f32; t * vocab];
    for (frame, &cls) in argmax_per_frame.iter().enumerate() {
      data[frame * vocab + cls as usize] = 1.0;
    }
    let logits = Array::from_slice::<f32>(&data, &[t as i32, vocab as i32]).unwrap();
    Self {
      logits,
      first_blank,
      later_blank,
      calls: Cell::new(0),
    }
  }
}

impl CtcModel for StatefulBlankCtc {
  fn logits(&self, _waveform: &Array) -> Result<Array> {
    self.logits.try_clone()
  }
  fn blank_id(&self) -> u32 {
    let n = self.calls.get();
    self.calls.set(n + 1);
    if n == 0 {
      self.first_blank
    } else {
      self.later_blank
    }
  }
  fn decode_ids(&self, ids: &[u32]) -> String {
    ids.iter().map(|&i| (b'a' + i as u8) as char).collect()
  }
}

#[test]
fn ctc_blank_id_read_once_collapse_uses_validated_value() {
  // TOCTOU guard: `blank_id` is `&self`, so a model could return an in-range
  // blank to the driver's range check and a DIFFERENT blank to the collapse.
  // The driver must read `blank_id` EXACTLY ONCE and collapse against that
  // first (validated) value.
  //
  // Frames argmax (vocab 4): [0, 1, 2] — no consecutive dups.
  //   first_blank = 1 (validated, in range): drop id 1 -> survivors [0, 2] -> "ac".
  //   later_blank = 0 (the value a SECOND `blank_id` call would return): if the
  //   collapse re-read `blank_id`, it would drop id 0 -> survivors [1, 2] -> "bc".
  // Asserting "ac" proves the cached, validated blank (1) drove the collapse,
  // and that exactly one `blank_id` call was made.
  let model = StatefulBlankCtc::from_argmax(&[0, 1, 2], 4, 1, 0);
  let out = greedy_ctc_transcribe(&model, &dummy_waveform(), &TranscribeOptions::new())
    .expect("ctc transcribe");
  assert_eq!(out.text(), "ac");
  // Exactly one `blank_id` call: the validated read IS the used read.
  assert_eq!(model.calls.get(), 1);
}

#[test]
fn ctc_stateful_blank_first_read_out_of_range_is_rejected() {
  // The validated value is the FIRST read: if that first `blank_id` is out of
  // range, the driver rejects with a typed OutOfRange even though a later read
  // would have been in range — the guard never trusts a second read.
  // vocab 3 (valid 0..=2); first_blank = 3 (out of range), later_blank = 0.
  let model = StatefulBlankCtc::from_argmax(&[0, 1, 2], 3, 3, 0);
  let err =
    greedy_ctc_transcribe(&model, &dummy_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

/// A CTC mock whose `logits` is a LAZILY-shaped `(t, vocab)` `zeros` grid — the
/// shape is carried on the lazy array's metadata with NO host materialization,
/// so an oversized `t` or `vocab` is realized only as a `shape`, never as bytes.
/// Drives the model-provided-magnitude caps (time axis, vocab axis).
struct LazyShapeCtc {
  logits: Array,
}

impl LazyShapeCtc {
  fn new(t: i32, vocab: i32) -> Self {
    // `Array::zeros` builds a lazy node: the `(t, vocab)` shape exists without
    // materializing `t * vocab` floats, so a multi-Gi grid costs nothing here.
    Self {
      logits: Array::zeros::<f32>(&[t, vocab]).unwrap(),
    }
  }
}

impl CtcModel for LazyShapeCtc {
  fn logits(&self, _waveform: &Array) -> Result<Array> {
    self.logits.try_clone()
  }
  fn blank_id(&self) -> u32 {
    0
  }
  fn decode_ids(&self, _ids: &[u32]) -> String {
    String::new()
  }
}

#[test]
fn ctc_rejects_oversized_time_axis_before_materializing() {
  // A model returning a LAZILY-shaped `(MAX_DECODED_SAMPLES + 1, vocab)` logits
  // would, if un-capped, take an argmax + `to_vec::<u32>()` of one u32 per frame
  // -> a multi-GB OOM. An absurd time axis `T'` of `MAX_DECODED_SAMPLES + 1` far
  // exceeds the `dummy_waveform()` input sample count (and the absolute cap), so
  // the driver rejects it off the lazy `.shape()` with a typed OutOfRange BEFORE the
  // argmax/to_vec — this test allocates nothing (the `zeros` grid is never
  // materialized). The dedicated input-tie and absolute-cap relationship is
  // covered by `ctc_rejects_time_axis_exceeding_input_samples`.
  let model = LazyShapeCtc::new((audio_io::MAX_DECODED_SAMPLES + 1) as i32, 4);
  let err =
    greedy_ctc_transcribe(&model, &dummy_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

#[test]
fn ctc_rejects_oversized_vocab_axis_before_materializing() {
  // A LAZILY-shaped `(T, MAX_LOGITS_VOCAB + 1)` logits: the argmax over a
  // multi-Gi-wide vocab axis is rejected off the lazy `.shape()` with a typed
  // OutOfRange before any materialization. Small `T` so only the vocab cap fires.
  let model = LazyShapeCtc::new(2, (MAX_LOGITS_VOCAB + 1) as i32);
  let err =
    greedy_ctc_transcribe(&model, &dummy_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

#[test]
fn ctc_rejects_oversized_element_product_before_materializing() {
  // The PRODUCT guard: a LAZILY-shaped `(8193, 8192)` logits whose axes EACH
  // individually pass their per-axis cap — time `8193 <= input samples`
  // (16384 here) and `<= MAX_DECODED_SAMPLES`; vocab `8192 <= MAX_LOGITS_VOCAB`
  // (256 Ki) — but whose product `8193 * 8192 = 67_117_056 > MAX_LOGITS_ELEMENTS`
  // (64 Mi). Without the product guard the `argmax` + `to_vec::<u32>()` would
  // force `eval` of a 64-Mi-element tensor (an OOM); with it, the driver rejects
  // off the lazy `.shape()` (a `checked_mul`) BEFORE any materialization, so this
  // test allocates nothing (the `zeros` grid is never realized). The input is
  // 16384 samples so the time axis passes the T'-vs-input tie and the PRODUCT
  // guard is what fires.
  let t = 8193_usize;
  let vocab = 8192_usize;
  assert!(
    t <= audio_io::MAX_DECODED_SAMPLES,
    "time axis under absolute cap"
  );
  assert!(vocab <= MAX_LOGITS_VOCAB, "vocab axis under per-axis cap");
  assert!(
    t.checked_mul(vocab).unwrap() > MAX_LOGITS_ELEMENTS,
    "product exceeds the element budget"
  );
  let input: Vec<f32> = vec![0.0_f32; 16_384];
  let waveform = Array::from_slice::<f32>(&input, &[input.len() as i32]).unwrap();
  let model = LazyShapeCtc::new(t as i32, vocab as i32);
  let err = greedy_ctc_transcribe(&model, &waveform, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

#[test]
fn ctc_rejects_time_axis_exceeding_input_samples() {
  // The T'-vs-input tie: a model whose logits time axis `T'` exceeds the
  // validated input sample count — but is well under MAX_DECODED_SAMPLES — is
  // rejected with a typed OutOfRange, so a normal-length input cannot be
  // amplified into a huge frame axis. A dedicated 4-sample input here; the model
  // returns `(8, vocab)` lazy logits (`T' = 8 > 4` samples, yet `8` is tiny vs
  // the absolute cap and the product `8 * 4 = 32` is far under the element
  // budget), so the input-tie guard — not the absolute or product cap — is what
  // fires. The `zeros` grid is never materialized (rejected off `.shape()`).
  let small_input = Array::from_slice::<f32>(&[0.1_f32, -0.2, 0.3, -0.4], &[4]).unwrap();
  let model = LazyShapeCtc::new(8, 4);
  let err = greedy_ctc_transcribe(&model, &small_input, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

// ===========================================================================
// Autoregressive family — the `greedy_transcribe` loop.
// ===========================================================================

/// The mock's owned decode cache. A fresh one is minted per generation and
/// threaded through `decode_step` by `&mut`; it carries the per-call step
/// count so `decode_step` has owned mutable state to advance (the driver
/// owning + `&mut`-threading this value is the contract under test).
#[derive(Default)]
struct MockCache {
  steps: usize,
}

/// An autoregressive mock that scripts a fixed token sequence then `eot`.
///
/// `decode_step` emits, at decode position `k` (the number of tokens produced
/// after the prompt prefix), a one-hot `(vocab,)` logit row whose argmax is
/// `script[k]`, or `eot` once `k == script.len()`. So the driver's decoded ids
/// are exactly `script` — a closed-form oracle.
struct MockSttModel {
  script: Vec<u32>,
  prompt: Vec<u32>,
  eot: u32,
  vocab: usize,
  mel_cfg: MelConfig,
  /// Records the token slice `decode_step` last saw, to confirm the driver
  /// grows + threads the full sequence (prompt + decoded-so-far).
  last_tokens_len: Cell<usize>,
  /// Mirrors the owned cache's final step count for a post-hoc assertion (the
  /// cache itself is dropped inside the driver).
  steps_total: Cell<usize>,
  caches_minted: Cell<usize>,
}

impl MockSttModel {
  fn new(script: Vec<u32>, prompt: Vec<u32>, eot: u32, vocab: usize) -> Self {
    Self {
      script,
      prompt,
      eot,
      vocab,
      mel_cfg: MelConfig::whisper_default(),
      last_tokens_len: Cell::new(0),
      steps_total: Cell::new(0),
      caches_minted: Cell::new(0),
    }
  }

  fn with_mel_config(mut self, cfg: MelConfig) -> Self {
    self.mel_cfg = cfg;
    self
  }

  fn one_hot(&self, cls: u32) -> Result<Array> {
    let mut row = vec![0.0_f32; self.vocab];
    row[cls as usize] = 1.0;
    Array::from_slice::<f32>(&row, &[self.vocab as i32])
  }
}

impl AutoregressiveStt for MockSttModel {
  type Cache = MockCache;

  // Uses the DEFAULT `log_mel` (delegates to `default_log_mel` + `mel_config`)
  // — exercised by the threading test. `encode` is a trivial pass-through so
  // the loop has a non-trivial-but-deterministic encoder state to forward.
  fn encode(&self, mel: &Array) -> Result<Array> {
    mel.try_clone()
  }

  fn new_cache(&self) -> Self::Cache {
    self.caches_minted.set(self.caches_minted.get() + 1);
    MockCache::default()
  }

  fn decode_step(&self, cache: &mut Self::Cache, _enc: &Array, tokens: &[u32]) -> Result<Array> {
    // The decode position is read FROM the owned cache (then advanced), so the
    // test exercises the `&mut Self::Cache` threading: step `k` here must equal
    // the count of tokens decoded after the prompt prefix.
    let k = cache.steps;
    debug_assert_eq!(k, tokens.len() - self.prompt.len());
    cache.steps += 1;
    self.steps_total.set(cache.steps);
    self.last_tokens_len.set(tokens.len());
    let cls = self.script.get(k).copied().unwrap_or(self.eot);
    self.one_hot(cls)
  }

  fn initial_tokens(&self, _opts: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(self.prompt.clone())
  }

  fn eot(&self) -> u32 {
    self.eot
  }

  fn mel_config(&self) -> MelConfig {
    self.mel_cfg
  }
}

fn speech_waveform() -> Array {
  // 800 samples so the default whisper log-mel (n_fft 400, hop 160) frames.
  let data: Vec<f32> = (0..800).map(|i| (i as f32 * 0.01).sin()).collect();
  Array::from_slice::<f32>(&data, &[800]).unwrap()
}

#[test]
fn greedy_decodes_scripted_sequence() {
  // Script 3 tokens then eot; prompt is a 2-token prefix excluded from output.
  let model = MockSttModel::new(vec![5, 6, 7], vec![100, 101], 99, 128);
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");

  // Oracle: decoded ids == script -> "5 6 7".
  assert_eq!(out.text(), "5 6 7");
  assert_eq!(out.segments_slice().len(), 1);
  assert_eq!(out.segments_slice()[0].text(), "5 6 7");

  // The driver routed through the model: one fresh cache, and `decode_step`
  // ran 4 times (3 scripted tokens + the eot step), last seeing prompt(2) +
  // 3 decoded = 5 tokens.
  assert_eq!(model.caches_minted.get(), 1);
  assert_eq!(model.steps_total.get(), 4);
  assert_eq!(model.last_tokens_len.get(), 5);
}

#[test]
fn greedy_excludes_prompt_from_text() {
  // Immediate eot -> no decoded tokens -> empty text (prompt never leaks).
  let model = MockSttModel::new(vec![], vec![100, 101, 102], 99, 128);
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");
  assert_eq!(out.text(), "");
}

#[test]
fn greedy_threads_language_from_options() {
  // eot = 7 is in-vocab (the mock's one-hot row is `vocab`-wide), distinct
  // from the scripted token 1.
  let model = MockSttModel::new(vec![1], vec![0], 7, 8);
  let opts = TranscribeOptions::new()
    .with_language("de")
    .with_task(Task::Translate);
  let out = greedy_transcribe(&model, &speech_waveform(), &opts).expect("greedy transcribe");
  assert_eq!(out.language(), Some("de"));
  assert_eq!(out.text(), "1");
}

/// A model that NEVER emits eot (argmax is always class 1, eot is class 0), so
/// the greedy loop only terminates at the `max_context` bound. `prompt` and
/// `max_ctx` are parameterized to exercise the total-context cap.
struct NeverStops {
  vocab: usize,
  prompt: Vec<u32>,
  max_ctx: usize,
}
impl AutoregressiveStt for NeverStops {
  type Cache = ();
  fn encode(&self, mel: &Array) -> Result<Array> {
    mel.try_clone()
  }
  fn new_cache(&self) {}
  fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
    // Always argmax class 1; eot is class 0 -> never reached.
    let mut row = vec![0.0_f32; self.vocab];
    row[1] = 1.0;
    Array::from_slice::<f32>(&row, &[self.vocab as i32])
  }
  fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(self.prompt.clone())
  }
  fn eot(&self) -> u32 {
    0
  }
  fn max_context(&self) -> usize {
    self.max_ctx
  }
}

#[test]
fn greedy_stops_at_max_context_with_empty_prompt() {
  // Empty prompt + default `max_context` (448): the runaway loop decodes
  // exactly DEFAULT_MAX_DECODE_STEPS class-1 tokens (total == max_context).
  let model = NeverStops {
    vocab: 4,
    prompt: vec![],
    max_ctx: DEFAULT_MAX_DECODE_STEPS,
  };
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");
  let want: String = std::iter::repeat_n("1", DEFAULT_MAX_DECODE_STEPS)
    .collect::<Vec<_>>()
    .join(" ");
  assert_eq!(out.text(), want);
}

#[test]
fn greedy_caps_total_at_max_context_accounting_for_prompt() {
  // A non-empty prompt eats into the budget: prompt(3) + generated must never
  // exceed `max_context`(10), so at most 7 new tokens are decoded.
  let max_ctx = 10;
  let prompt = vec![20, 21, 22];
  let model = NeverStops {
    vocab: 4,
    prompt: prompt.clone(),
    max_ctx,
  };
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");
  // Oracle: max_ctx - prompt.len() = 10 - 3 = 7 generated tokens, all class 1.
  let n_generated = max_ctx - prompt.len();
  let want: String = std::iter::repeat_n("1", n_generated)
    .collect::<Vec<_>>()
    .join(" ");
  assert_eq!(out.text(), want);
  // The decoded count (7) + prompt (3) == max_context (10): the total never
  // exceeds the decoder's context.
  assert_eq!(out.text().split_whitespace().count(), n_generated);
  assert_eq!(prompt.len() + n_generated, max_ctx);
}

#[test]
fn greedy_caller_max_new_tokens_caps_below_context() {
  // Option 3: a never-eot model with default `max_context` (448) would run to
  // 448 generated tokens; `with_max_new_tokens(5)` lowers the loop bound so it
  // stops at EXACTLY 5 generated tokens (the caller limit, not `max_context`).
  let model = NeverStops {
    vocab: 4,
    prompt: vec![],
    max_ctx: DEFAULT_MAX_DECODE_STEPS,
  };
  let opts = TranscribeOptions::default().with_max_new_tokens(5);
  let out = greedy_transcribe(&model, &speech_waveform(), &opts).expect("greedy transcribe");
  // Decoded ids are 5 class-1 tokens -> "1 1 1 1 1".
  assert_eq!(out.text(), "1 1 1 1 1");
  assert_eq!(out.text().split_whitespace().count(), 5);
}

#[test]
fn greedy_caller_max_new_tokens_clamps_to_remaining_context() {
  // A caller limit LARGER than the remaining context is harmlessly clamped to
  // the context: prompt(3) + generated must still never exceed max_context(10),
  // so even `with_max_new_tokens(1000)` yields only `10 - 3 = 7` tokens.
  let max_ctx = 10;
  let prompt = vec![20, 21, 22];
  let model = NeverStops {
    vocab: 4,
    prompt: prompt.clone(),
    max_ctx,
  };
  let opts = TranscribeOptions::default().with_max_new_tokens(1000);
  let out = greedy_transcribe(&model, &speech_waveform(), &opts).expect("greedy transcribe");
  let n_generated = max_ctx - prompt.len(); // 7
  assert_eq!(out.text().split_whitespace().count(), n_generated);
  assert_eq!(prompt.len() + n_generated, max_ctx);
}

#[test]
fn greedy_max_new_tokens_none_falls_back_to_max_context() {
  // `max_new_tokens == None` (the default) uses the full `max_context`: a
  // never-eot model decodes exactly `max_context` tokens, identical to the
  // no-caller-limit `greedy_stops_at_max_context_with_empty_prompt` behaviour.
  // This pins the `map_or(cap_new, …)` fallback explicitly.
  let max_ctx = 12;
  let model = NeverStops {
    vocab: 4,
    prompt: vec![],
    max_ctx,
  };
  let opts = TranscribeOptions::default();
  assert_eq!(opts.max_new_tokens(), None);
  let out = greedy_transcribe(&model, &speech_waveform(), &opts).expect("greedy transcribe");
  assert_eq!(out.text().split_whitespace().count(), max_ctx);
}

#[test]
fn greedy_rejects_prompt_at_or_over_max_context() {
  // initial_tokens length >= max_context leaves no room to decode -> typed
  // OutOfRange (the prompt-exceeds-context guard).
  let max_ctx = 4;
  // Prompt length == max_context (the boundary): rejected.
  let model_eq = NeverStops {
    vocab: 4,
    prompt: vec![1, 2, 3, 4],
    max_ctx,
  };
  let err =
    greedy_transcribe(&model_eq, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));

  // Prompt length > max_context: also rejected.
  let model_over = NeverStops {
    vocab: 4,
    prompt: vec![1, 2, 3, 4, 5],
    max_ctx,
  };
  let err =
    greedy_transcribe(&model_over, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

#[test]
fn greedy_rejects_rank2_waveform_via_default_log_mel() {
  // A 2-D waveform reaches `default_log_mel` (the default `log_mel`) and is
  // rejected (RankMismatch) before any feature extraction — not flattened.
  let model = MockSttModel::new(vec![1], vec![0], 7, 8);
  let stereo = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
  let err = greedy_transcribe(&model, &stereo, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
}

#[test]
fn greedy_rejects_non_rank1_decode_logits() {
  struct BadStep;
  impl AutoregressiveStt for BadStep {
    type Cache = ();
    fn encode(&self, mel: &Array) -> Result<Array> {
      mel.try_clone()
    }
    fn new_cache(&self) {}
    fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
      // Rank-2 -> typed RankMismatch.
      Array::from_slice::<f32>(&[0.0_f32, 1.0], &[1, 2])
    }
    fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
      Ok(vec![0])
    }
    fn eot(&self) -> u32 {
      99
    }
  }
  let err = greedy_transcribe(&BadStep, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
}

/// An autoregressive mock whose `eot` id is OUT OF RANGE for its decode
/// logits. `decode_step` always argmaxes class `1` (in range, so the loop is
/// well-formed and reaches the eot range check), but `eot()` reports `vocab`
/// (one past the last valid class). A `Cell` records how many times `eot()` was
/// called, to confirm the driver reads it exactly once.
struct EotOutOfRange {
  vocab: usize,
  eot_calls: Cell<u32>,
}

impl AutoregressiveStt for EotOutOfRange {
  type Cache = ();
  fn encode(&self, mel: &Array) -> Result<Array> {
    mel.try_clone()
  }
  fn new_cache(&self) {}
  fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
    // Argmax is class 1, always in range — the `(vocab,)` row is well-formed,
    // so the loop reaches the cached-eot range check rather than failing on
    // shape.
    let mut row = vec![0.0_f32; self.vocab];
    row[1] = 1.0;
    Array::from_slice::<f32>(&row, &[self.vocab as i32])
  }
  fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(vec![0])
  }
  fn eot(&self) -> u32 {
    self.eot_calls.set(self.eot_calls.get() + 1);
    // One past the last valid class -> can never be produced by argmax.
    self.vocab as u32
  }
}

#[test]
fn greedy_rejects_eot_outside_vocab() {
  // An `eot` >= the decode_step vocab size can never equal a per-frame argmax,
  // so the greedy loop's `next == eot` stop would never fire and the loop would
  // run to `max_context`, returning bogus full-length output. The driver
  // range-checks the cached `eot` against the actual logits vocab the first
  // step and rejects out-of-range with a typed OutOfRange.
  // vocab = 4 (valid ids 0..=3), eot() = 4 (out of range).
  let model = EotOutOfRange {
    vocab: 4,
    eot_calls: Cell::new(0),
  };
  let err = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
  // `eot` is read exactly once (cached in a local), not per loop iteration.
  assert_eq!(model.eot_calls.get(), 1);
}

/// An autoregressive mock whose per-step logits WIDTH is scripted, to prove the
/// `eot` range-check re-runs against EVERY step's vocab (not once via a latch).
/// Step `k` (0-based) returns a `(widths[k],)` one-hot row whose argmax is class
/// `1` (always in range, so the loop reaches the eot check and never stops on
/// `next == eot`). A `Cell` advances the step index. With `eot` between two
/// widths (e.g. `eot = 4`, widths `[5, 3]`), step 0 passes (`4 < 5`) but step 1
/// is out of range (`4 >= 3`) — a one-shot latch would miss it and loop to
/// `max_context`.
struct ShrinkingVocab {
  widths: Vec<usize>,
  eot: u32,
  step: Cell<usize>,
}

impl AutoregressiveStt for ShrinkingVocab {
  type Cache = ();
  fn encode(&self, mel: &Array) -> Result<Array> {
    mel.try_clone()
  }
  fn new_cache(&self) {}
  fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
    let k = self.step.get();
    self.step.set(k + 1);
    // Past the script, keep the last width (the loop should already have
    // errored on the shrunk step, so this is only a safety net).
    let vocab = self
      .widths
      .get(k)
      .copied()
      .unwrap_or(*self.widths.last().unwrap());
    let mut row = vec![0.0_f32; vocab];
    // argmax class 1 — in range for every scripted width (all >= 2), so the
    // loop never stops via the argmax and always reaches the eot range-check.
    row[1] = 1.0;
    Array::from_slice::<f32>(&row, &[vocab as i32])
  }
  fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(vec![0])
  }
  fn eot(&self) -> u32 {
    self.eot
  }
}

#[test]
fn greedy_revalidates_eot_against_each_step_vocab() {
  // FIX: the `eot` range-check runs per step, not once. A model whose vocab
  // SHRINKS on a later step (passing the check on the first, larger step) must
  // still be rejected when `eot` falls out of the smaller step's range —
  // otherwise `next == eot` can never fire and the loop runs to `max_context`.
  //
  // eot = 4; step 0 width 5 (4 < 5, passes); step 1 width 3 (4 >= 3, rejected).
  let model = ShrinkingVocab {
    widths: vec![5, 3],
    eot: 4,
    step: Cell::new(0),
  };
  let err = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  // Rejected with the typed OutOfRange at the shrunk step (NOT run to
  // max_context returning bogus output).
  assert!(matches!(err, Error::OutOfRange(_)));
  // The loop reached exactly step 1 (the second `decode_step`) before erroring:
  // step 0 passed the check, step 1's row was produced then rejected. The step
  // counter therefore advanced past both (2), proving the FIRST step did NOT
  // error and the per-step re-check is what caught it.
  assert_eq!(model.step.get(), 2);
}

#[test]
fn greedy_rejects_oversized_max_context_before_loop() {
  // A model reporting an absurd `max_context` (> MAX_DECODE_CONTEXT) would make
  // `max_new = max_ctx - prompt_len` an effectively unbounded decode loop with
  // infallible `tokens.push` growth -> OOM (a never-eot model). The driver caps
  // `max_context` against MAX_DECODE_CONTEXT BEFORE deriving `max_new` / entering
  // the loop and rejects with a typed OutOfRange — so the loop never runs and
  // no oversized token `Vec` is grown.
  let model = NeverStops {
    vocab: 4,
    prompt: vec![],
    max_ctx: MAX_DECODE_CONTEXT + 1,
  };
  let err = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

/// An autoregressive mock whose `decode_step` returns a LAZILY-shaped
/// `(MAX_LOGITS_VOCAB + 1,)` `zeros` row — its width is on the lazy array's
/// metadata with NO host materialization — to drive the per-step vocab cap.
struct OversizedVocabStep;

impl AutoregressiveStt for OversizedVocabStep {
  type Cache = ();
  fn encode(&self, mel: &Array) -> Result<Array> {
    mel.try_clone()
  }
  fn new_cache(&self) {}
  fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
    // Lazy `(MAX_LOGITS_VOCAB + 1,)` zeros: an absurd vocab width that exists as
    // a shape only (no allocation of that many floats).
    Array::zeros::<f32>(&[(MAX_LOGITS_VOCAB + 1) as i32])
  }
  fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(vec![0])
  }
  fn eot(&self) -> u32 {
    99
  }
}

#[test]
fn greedy_rejects_oversized_step_vocab_before_materializing() {
  // A `decode_step` returning a LAZILY-shaped `(MAX_LOGITS_VOCAB + 1,)` row would,
  // if un-capped, take an argmax over a multi-Gi-wide vocab axis -> OOM. The
  // driver caps the step's vocab off the lazy `.shape()` and rejects with a typed
  // OutOfRange BEFORE the argmax — so this test allocates nothing.
  let err = greedy_transcribe(
    &OversizedVocabStep,
    &speech_waveform(),
    &TranscribeOptions::new(),
  )
  .unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

/// A tracker mock that records whether its `log_mel` and `encode` frontend
/// hooks were reached, to prove the driver-owned preflight (waveform metadata +
/// prompt-vs-context) runs BEFORE any model frontend call. `prompt` and
/// `max_ctx` are parameterized to drive the prompt-over-context gate; the
/// decode loop itself is irrelevant here (the preflight rejects before it).
struct FrontendTracker {
  prompt: Vec<u32>,
  max_ctx: usize,
  log_mel_called: Cell<bool>,
  encode_called: Cell<bool>,
}

impl FrontendTracker {
  fn new(prompt: Vec<u32>, max_ctx: usize) -> Self {
    Self {
      prompt,
      max_ctx,
      log_mel_called: Cell::new(false),
      encode_called: Cell::new(false),
    }
  }
}

impl AutoregressiveStt for FrontendTracker {
  type Cache = ();

  // Override `log_mel` to RECORD the call (and otherwise behave as the default
  // would): if the driver's preflight precedes the frontend, a rejected input
  // never sets this flag.
  fn log_mel(&self, audio: &Array) -> Result<Array> {
    self.log_mel_called.set(true);
    default_log_mel(&self.mel_config(), audio)
  }
  fn encode(&self, mel: &Array) -> Result<Array> {
    self.encode_called.set(true);
    mel.try_clone()
  }
  fn new_cache(&self) {}
  fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> Result<Array> {
    // eot is class 0 and argmax is class 0 here -> immediate stop (only reached
    // when the preflight passes).
    let mut row = vec![0.0_f32; 4];
    row[0] = 1.0;
    Array::from_slice::<f32>(&row, &[4])
  }
  fn initial_tokens(&self, _o: &TranscribeOptions) -> Result<Vec<u32>> {
    Ok(self.prompt.clone())
  }
  fn eot(&self) -> u32 {
    0
  }
  fn max_context(&self) -> usize {
    self.max_ctx
  }
}

#[test]
fn greedy_preflight_rejects_rank2_waveform_before_log_mel() {
  // A rank-2 waveform is rejected by the driver-owned `validate_waveform` gate
  // BEFORE the (overrideable) `log_mel` frontend is ever called.
  let model = FrontendTracker::new(vec![], DEFAULT_MAX_DECODE_STEPS);
  let stereo = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
  let err = greedy_transcribe(&model, &stereo, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
  // The frontend was never reached.
  assert!(!model.log_mel_called.get());
  assert!(!model.encode_called.get());
}

#[test]
fn greedy_preflight_rejects_empty_waveform_before_log_mel() {
  // An empty waveform is rejected by the driver-owned gate before `log_mel`.
  let model = FrontendTracker::new(vec![], DEFAULT_MAX_DECODE_STEPS);
  let empty = Array::from_slice::<f32>(&[] as &[f32], &[0]).unwrap();
  let err = greedy_transcribe(&model, &empty, &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::EmptyInput(_)));
  assert!(!model.log_mel_called.get());
  assert!(!model.encode_called.get());
}

#[test]
fn greedy_preflight_rejects_over_context_prompt_before_frontend() {
  // A prompt whose length >= max_context is rejected (OutOfRange) BEFORE the
  // frontend + encode run, so neither `log_mel` nor `encode` is reached (the
  // call-tracker proves the prompt gate precedes the frontend).
  let max_ctx = 4;
  let model = FrontendTracker::new(vec![1, 2, 3, 4], max_ctx);
  let err = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new()).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
  assert!(!model.log_mel_called.get());
  assert!(!model.encode_called.get());
}

#[test]
fn greedy_preflight_runs_frontend_when_gates_pass() {
  // Control: a valid waveform + in-range prompt DOES reach the frontend, so the
  // tracker is not a false-negative (the flags can be set).
  let model = FrontendTracker::new(vec![], DEFAULT_MAX_DECODE_STEPS);
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");
  // Immediate eot -> empty decoded text.
  assert_eq!(out.text(), "");
  assert!(model.log_mel_called.get());
  assert!(model.encode_called.get());
}

// ===========================================================================
// Shared waveform helpers.
// ===========================================================================

#[test]
fn default_log_mel_threads_config_and_floor() {
  let audio = speech_waveform();
  let cfg = MelConfig::whisper_default();

  // Oracle: the driver helper must equal a direct `log_mel_spectrogram_with`
  // with the SAME config params — confirming it threads every field.
  let mut via_helper = default_log_mel(&cfg, &audio).expect("default_log_mel");
  let mut via_dsp = crate::audio::dsp::log_mel_spectrogram_with(
    &audio,
    cfg.n_fft(),
    cfg.hop_length(),
    cfg.win_length(),
    cfg.n_mels(),
    cfg.sample_rate(),
    cfg.f_min(),
    cfg.f_max(),
    cfg.log_floor(),
  )
  .expect("dsp log_mel");
  assert_eq!(via_helper.shape(), via_dsp.shape());
  assert_eq!(
    via_helper.to_vec::<f32>().unwrap(),
    via_dsp.to_vec::<f32>().unwrap()
  );

  // Changing the log floor must change the output (the floor is threaded, not
  // hard-coded). The Kaldi floor (1e-8) lifts low-energy bins above the
  // Whisper floor (1e-10), so at least one element differs.
  let kaldi_cfg = cfg.with_log_floor(LogFloor::Kaldi);
  let mut via_kaldi = default_log_mel(&kaldi_cfg, &audio).expect("default_log_mel kaldi");
  assert_ne!(
    via_helper.to_vec::<f32>().unwrap(),
    via_kaldi.to_vec::<f32>().unwrap()
  );
}

#[test]
fn default_log_mel_uses_models_mel_config_via_log_mel_default() {
  // The `AutoregressiveStt::log_mel` DEFAULT must route through the model's
  // `mel_config`. A model with a Kaldi-floor config must produce the same mel
  // as `default_log_mel` with that config (oracle), and differ from the
  // whisper-floor default.
  let audio = speech_waveform();
  let kaldi_cfg = MelConfig::whisper_default().with_log_floor(LogFloor::Kaldi);
  let model = MockSttModel::new(vec![], vec![0], 7, 8).with_mel_config(kaldi_cfg);

  let mut via_default = model.log_mel(&audio).expect("model.log_mel default");
  let mut oracle = default_log_mel(&kaldi_cfg, &audio).expect("default_log_mel oracle");
  assert_eq!(
    via_default.to_vec::<f32>().unwrap(),
    oracle.to_vec::<f32>().unwrap()
  );
}

#[test]
fn resample_waveform_matches_resample_linear_oracle() {
  // Oracle: resample_waveform == resample_linear on the raw samples.
  let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
  let audio = Array::from_slice::<f32>(&data, &[100]).unwrap();

  let mut got = resample_waveform(&audio, 16_000, 8_000).expect("resample_waveform");
  let oracle = crate::audio::io::resample_linear(&data, 16_000, 8_000).expect("resample_linear");
  assert_eq!(got.to_vec::<f32>().unwrap(), oracle);
  // 16k -> 8k halves the sample count.
  assert_eq!(got.shape(), vec![oracle.len()]);
}

#[test]
fn resample_waveform_same_rate_is_verbatim() {
  let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
  let audio = Array::from_slice::<f32>(&data, &[4]).unwrap();
  let mut got = resample_waveform(&audio, 16_000, 16_000).expect("resample_waveform");
  assert_eq!(got.to_vec::<f32>().unwrap(), data);
}

#[test]
fn empty_waveform_is_rejected_by_helpers_and_autoregressive_driver() {
  let empty = Array::from_slice::<f32>(&[] as &[f32], &[0]).unwrap();

  // The shared waveform helpers reject an empty waveform directly.
  assert!(matches!(
    default_log_mel(&MelConfig::whisper_default(), &empty),
    Err(Error::EmptyInput(_))
  ));
  assert!(matches!(
    resample_waveform(&empty, 16_000, 8_000),
    Err(Error::EmptyInput(_))
  ));

  // The autoregressive driver rejects it through the default `log_mel`
  // (a CTC model's empty handling is its own `logits` frontend's concern).
  let ar = MockSttModel::new(vec![1], vec![0], 7, 8);
  assert!(matches!(
    greedy_transcribe(&ar, &empty, &TranscribeOptions::new()),
    Err(Error::EmptyInput(_))
  ));
}

#[test]
fn rank2_waveform_is_rejected_by_helpers() {
  // A 2-D waveform is rejected (RankMismatch) by the shared metadata gate —
  // BEFORE any materialization — so it is never silently flattened to mono.
  let stereo = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
  assert!(matches!(
    default_log_mel(&MelConfig::whisper_default(), &stereo),
    Err(Error::RankMismatch(_))
  ));
  assert!(matches!(
    resample_waveform(&stereo, 16_000, 8_000),
    Err(Error::RankMismatch(_))
  ));
}

// ===========================================================================
// Autoregressive cumulative decode-work budget (Option 1) — the pure
// `accumulate_decode_work` helper, the cheap & deterministic proof of the
// budget arithmetic (a real loop tripping `MAX_AR_DECODE_WORK` would have to
// argmax 256 Mi elements — exactly what this helper exists to avoid testing).
// ===========================================================================

#[test]
fn accumulate_decode_work_sums_below_budget() {
  // Below the budget, each call returns the running sum (and never errors).
  let mut work = 0usize;
  work = accumulate_decode_work(work, 1000).expect("step 1 under budget");
  assert_eq!(work, 1000);
  work = accumulate_decode_work(work, 50_000).expect("step 2 under budget");
  assert_eq!(work, 51_000);
  work = accumulate_decode_work(work, 0).expect("a zero-vocab step is a no-op");
  assert_eq!(work, 51_000);
}

#[test]
fn accumulate_decode_work_allows_exactly_the_budget() {
  // The budget is inclusive: a total of EXACTLY MAX_AR_DECODE_WORK is Ok; the
  // very next non-zero element tips it over and errors.
  let at_budget = accumulate_decode_work(0, MAX_AR_DECODE_WORK).expect("exactly the budget is Ok");
  assert_eq!(at_budget, MAX_AR_DECODE_WORK);
  let err = accumulate_decode_work(at_budget, 1).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
}

#[test]
fn accumulate_decode_work_rejects_over_budget() {
  // A running total that would exceed MAX_AR_DECODE_WORK is a typed OutOfRange.
  let err = accumulate_decode_work(MAX_AR_DECODE_WORK - 5, 6).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
  // One step whose own vocab alone exceeds the budget is rejected from zero.
  let err_single = accumulate_decode_work(0, MAX_AR_DECODE_WORK + 1).unwrap_err();
  assert!(matches!(err_single, Error::OutOfRange(_)));
}

#[test]
fn accumulate_decode_work_rejects_usize_overflow() {
  // A `checked_add` overflow (work near usize::MAX) is a typed OutOfRange, not
  // a wrap-around: the guard caps the cumulative work even against a pathologic
  // accumulator value.
  let err = accumulate_decode_work(usize::MAX, 1).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)));
  let err2 = accumulate_decode_work(usize::MAX - 3, 10).unwrap_err();
  assert!(matches!(err2, Error::OutOfRange(_)));
}
