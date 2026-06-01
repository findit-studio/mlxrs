//! Tests for the STT decoding drivers: the CTC greedy-collapse blanket impl
//! and the autoregressive [`greedy_transcribe`] loop, plus the shared waveform
//! helpers ([`default_log_mel`], [`resample_waveform`], non-empty validation).
//!
//! Oracles are computed from the TEST INPUTS (the scripted token sequences,
//! the hand-built logits), never from the code under test.

use std::cell::Cell;

use super::*;
use crate::{
  audio::{
    dsp::LogFloor,
    stt::model::{AutoregressiveStt, CtcModel, MelConfig, Task, Transcribe, TranscribeOptions},
  },
  error::Error,
};

// ===========================================================================
// CTC family — the blanket `impl<M: CtcModel> Transcribe`.
// ===========================================================================

/// A CTC mock returning a fixed `(T', vocab)` logit grid. `decode_ids` maps
/// each surviving id `i` to the character `(b'a' + i)` so the collapsed text
/// is a directly-checkable oracle.
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

fn dummy_waveform() -> Array {
  Array::from_slice::<f32>(&[0.1_f32, -0.2, 0.3, -0.4], &[4]).unwrap()
}

#[test]
fn ctc_blanket_collapses_dups_and_blanks() {
  // vocab 4, blank = 3. Frame argmax sequence (a=0, b=1, c=2, _=blank):
  //   a a _ a b b _ c
  // CTC greedy: collapse consecutive dups -> a _ a b _ c ; drop blank -> a a b c.
  let blank = 3;
  let frames = [0, 0, blank, 0, 1, 1, blank, 2];
  let model = MockCtcModel::from_argmax(&frames, 4, blank);

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
fn ctc_blanket_all_blank_is_empty() {
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
fn ctc_blanket_rejects_non_rank2_logits() {
  // A model handing back rank-1 logits is a per-model defect -> typed error.
  struct BadCtc;
  impl CtcModel for BadCtc {
    fn logits(&self, _waveform: &Array) -> Result<Array> {
      Array::from_slice::<f32>(&[0.0_f32, 1.0], &[2])
    }
    fn blank_id(&self) -> u32 {
      0
    }
    fn decode_ids(&self, _ids: &[u32]) -> String {
      String::new()
    }
  }
  let err = BadCtc
    .transcribe(&dummy_waveform(), &TranscribeOptions::new())
    .unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)));
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

#[test]
fn greedy_stops_at_max_decode_steps() {
  // A model that NEVER emits eot (eot id is unreachable: every script slot is
  // a distinct in-vocab token, and past the script we emit `script.last`,
  // never `eot`). The loop must terminate at the cap.
  struct NeverStops {
    vocab: usize,
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
      Ok(vec![7])
    }
    fn eot(&self) -> u32 {
      0
    }
  }
  let model = NeverStops { vocab: 4 };
  let out = greedy_transcribe(&model, &speech_waveform(), &TranscribeOptions::new())
    .expect("greedy transcribe");
  // Exactly DEFAULT_MAX_DECODE_STEPS class-1 tokens decoded.
  let want: String = std::iter::repeat_n("1", DEFAULT_MAX_DECODE_STEPS)
    .collect::<Vec<_>>()
    .join(" ");
  assert_eq!(out.text(), want);
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
