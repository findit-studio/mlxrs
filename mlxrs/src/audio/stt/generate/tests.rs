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

fn dummy_waveform() -> Array {
  Array::from_slice::<f32>(&[0.1_f32, -0.2, 0.3, -0.4], &[4]).unwrap()
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
