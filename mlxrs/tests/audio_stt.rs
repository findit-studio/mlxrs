//! Integration coverage for the `audio::stt` trait architecture from outside
//! the crate: the universal `Transcribe` contract, the `CtcModel` blanket
//! impl, the `AutoregressiveStt` + `greedy_transcribe` path, the shared
//! waveform helpers, and the `load` factory.
//!
//! Deterministic and dependency-free: external `MockCtcModel` /
//! `MockSttModel` fixtures (replicated, not imported — integration tests
//! cannot see crate-private test fixtures) script fixed outputs so the
//! decoded text is a closed-form oracle. Real on-disk WAV round-trips
//! (`save_wav` → `load_audio`) exercise the load → waveform-`Array` → trait
//! path end to end.
#![cfg(feature = "audio")]

use std::{path::PathBuf, process};

use mlxrs::{
  Array,
  audio::{
    dsp::LogFloor,
    io::{load_audio, save_wav},
    stt::{
      generate::{DEFAULT_MAX_DECODE_STEPS, default_log_mel, greedy_transcribe, resample_waveform},
      load::load,
      model::{
        AutoregressiveStt, CtcModel, MelConfig, Task, Transcribe, TranscribeOptions, Transcription,
      },
    },
  },
};

// ───────────────────────────── WAV helpers ─────────────────────────────

/// Process-scoped + named tempfile so parallel test binaries / cases never
/// collide. The audio I/O tests share the same convention.
fn temp_wav(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_audio_stt_{}_{}.wav", process::id(), name));
  p
}

/// Write a deterministic short WAV at `sample_rate` and load it back as a
/// mono waveform [`Array`] — the real load → waveform path a caller drives
/// before `transcribe`.
fn wav_waveform(name: &str, sample_rate: u32, n_samples: usize) -> Array {
  let path = temp_wav(name);
  let samples: Vec<f32> = (0..n_samples)
    .map(|i| ((i as f32 * 0.02).sin()) * 0.5)
    .collect();
  save_wav(&path, &samples, sample_rate).unwrap();
  let (loaded, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, sample_rate);
  let n = loaded.len() as i32;
  let arr = Array::from_slice::<f32>(&loaded, &[n]).unwrap();
  let _ = std::fs::remove_file(&path);
  arr
}

/// One second of audio at the Whisper sample rate.
const ONE_SECOND_16K: usize = 16_000;

// ───────────────────────── CTC family fixture ─────────────────────────

/// An external CTC mock returning a fixed `(T', vocab)` logit grid; its
/// `decode_ids` maps each surviving id `i` to `(b'a' + i)` so the collapsed
/// text is a directly-checkable oracle.
struct MockCtcModel {
  logits: Array,
  blank_id: u32,
}

impl MockCtcModel {
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
  fn logits(&self, _waveform: &Array) -> mlxrs::Result<Array> {
    self.logits.try_clone()
  }
  fn blank_id(&self) -> u32 {
    self.blank_id
  }
  fn decode_ids(&self, ids: &[u32]) -> String {
    ids.iter().map(|&i| (b'a' + i as u8) as char).collect()
  }
}

#[test]
fn ctc_blanket_transcribes_via_universal_contract() {
  // a a _ a b b _ c  →  collapse dups: a _ a b _ c  →  drop blank: a a b c
  let blank = 3;
  let model = MockCtcModel::from_argmax(&[0, 0, blank, 0, 1, 1, blank, 2], 4, blank);
  let audio = wav_waveform("ctc", 16_000, ONE_SECOND_16K);

  let out = model
    .transcribe(&audio, &TranscribeOptions::new())
    .expect("ctc transcribe");
  assert_eq!(out.text(), "aabc");
  assert!(out.language().is_none());
  assert_eq!(out.segments_slice().len(), 1, "CTC emits one segment");
}

#[test]
fn ctc_blanket_works_through_trait_object() {
  // The universal contract is object-safe: a `&dyn Transcribe` drives the
  // CTC blanket impl.
  let model = MockCtcModel::from_argmax(&[1, 1, 0], 3, 0);
  let audio = wav_waveform("ctc_dyn", 16_000, ONE_SECOND_16K);
  let dynm: &dyn Transcribe = &model;
  let out = dynm
    .transcribe(&audio, &TranscribeOptions::new())
    .expect("dyn transcribe");
  // frames 1 1 0 → collapse → 1 0 → drop blank(0) → 1 → 'b'.
  assert_eq!(out.text(), "b");
}

// ────────────────────── Autoregressive family fixture ──────────────────────

/// The mock's owned decode cache, threaded by `&mut` through `decode_step`.
#[derive(Default)]
struct MockCache {
  steps: usize,
}

/// An external autoregressive mock implementing the universal `Transcribe`
/// contract by forwarding to `greedy_transcribe` (the simple-model pattern).
///
/// `decode_step` emits, at decode position `k`, a one-hot `(vocab,)` logit row
/// whose argmax is `script[k]`, or `eot` once the script is exhausted — so the
/// decoded ids are exactly `script`.
struct MockSttModel {
  script: Vec<u32>,
  prompt: Vec<u32>,
  eot: u32,
  vocab: usize,
  mel_cfg: MelConfig,
}

impl MockSttModel {
  fn new(script: Vec<u32>, prompt: Vec<u32>, eot: u32, vocab: usize) -> Self {
    Self {
      script,
      prompt,
      eot,
      vocab,
      mel_cfg: MelConfig::whisper_default(),
    }
  }

  fn one_hot(&self, cls: u32) -> mlxrs::Result<Array> {
    let mut row = vec![0.0_f32; self.vocab];
    row[cls as usize] = 1.0;
    Array::from_slice::<f32>(&row, &[self.vocab as i32])
  }
}

impl AutoregressiveStt for MockSttModel {
  type Cache = MockCache;

  // Uses the DEFAULT `log_mel` (delegates to `default_log_mel` + `mel_config`).
  fn encode(&self, mel: &Array) -> mlxrs::Result<Array> {
    mel.try_clone()
  }

  fn new_cache(&self) -> Self::Cache {
    MockCache::default()
  }

  fn decode_step(
    &self,
    cache: &mut Self::Cache,
    _enc: &Array,
    tokens: &[u32],
  ) -> mlxrs::Result<Array> {
    let k = cache.steps;
    assert_eq!(k, tokens.len() - self.prompt.len(), "cache + tokens agree");
    cache.steps += 1;
    let cls = self.script.get(k).copied().unwrap_or(self.eot);
    self.one_hot(cls)
  }

  fn initial_tokens(&self, _opts: &TranscribeOptions) -> mlxrs::Result<Vec<u32>> {
    Ok(self.prompt.clone())
  }

  fn eot(&self) -> u32 {
    self.eot
  }

  fn mel_config(&self) -> MelConfig {
    self.mel_cfg
  }
}

impl Transcribe for MockSttModel {
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> mlxrs::Result<Transcription> {
    greedy_transcribe(self, audio, opts)
  }
}

#[test]
fn autoregressive_transcribe_decodes_scripted_sequence() {
  // Script 3 tokens then eot; the 2-token prompt prefix is excluded.
  let model = MockSttModel::new(vec![5, 6, 7], vec![100, 101], 99, 128);
  let audio = wav_waveform("ar", 16_000, ONE_SECOND_16K);
  // Through the universal `Transcribe` contract (which forwards to
  // `greedy_transcribe`).
  let out = model
    .transcribe(&audio, &TranscribeOptions::new())
    .expect("transcribe");
  assert_eq!(out.text(), "5 6 7");
  assert_eq!(out.segments_slice().len(), 1);
}

#[test]
fn autoregressive_threads_language_and_excludes_prompt() {
  let model = MockSttModel::new(vec![1], vec![0], 7, 8);
  let audio = wav_waveform("ar_lang", 16_000, ONE_SECOND_16K);
  let opts = TranscribeOptions::new()
    .with_language("fr")
    .with_task(Task::Translate);
  let out = model.transcribe(&audio, &opts).expect("transcribe");
  assert_eq!(out.language(), Some("fr"));
  assert_eq!(out.text(), "1", "prompt prefix excluded from decoded text");
}

#[test]
fn autoregressive_immediate_eot_is_empty() {
  let model = MockSttModel::new(vec![], vec![100, 101, 102], 7, 8);
  let audio = wav_waveform("ar_empty", 16_000, ONE_SECOND_16K);
  let out = model
    .transcribe(&audio, &TranscribeOptions::new())
    .expect("transcribe");
  assert_eq!(out.text(), "");
}

#[test]
fn autoregressive_bounds_runaway_decode_at_cap() {
  // A model that never emits eot must terminate at DEFAULT_MAX_DECODE_STEPS.
  struct NeverStops;
  impl AutoregressiveStt for NeverStops {
    type Cache = ();
    fn encode(&self, mel: &Array) -> mlxrs::Result<Array> {
      mel.try_clone()
    }
    fn new_cache(&self) {}
    fn decode_step(&self, _c: &mut (), _e: &Array, _t: &[u32]) -> mlxrs::Result<Array> {
      Array::from_slice::<f32>(&[0.0_f32, 1.0, 0.0, 0.0], &[4]) // argmax 1
    }
    fn initial_tokens(&self, _o: &TranscribeOptions) -> mlxrs::Result<Vec<u32>> {
      Ok(vec![5])
    }
    fn eot(&self) -> u32 {
      0 // never produced (argmax is always 1)
    }
  }
  let audio = wav_waveform("runaway", 16_000, ONE_SECOND_16K);
  let out = greedy_transcribe(&NeverStops, &audio, &TranscribeOptions::new()).expect("transcribe");
  let want = std::iter::repeat_n("1", DEFAULT_MAX_DECODE_STEPS)
    .collect::<Vec<_>>()
    .join(" ");
  assert_eq!(out.text(), want);
}

// ───────────────────────── shared waveform helpers ─────────────────────────

#[test]
fn resample_waveform_shrinks_frame_count_on_downsample() {
  // A 1 s 44.1 kHz WAV resampled to 16 kHz yields a much smaller mel frame
  // count than the 44.1 kHz source — the resample-on-sr-mismatch path.
  let audio_44k = wav_waveform("resample_src", 44_100, 44_100);
  let resampled = resample_waveform(&audio_44k, 44_100, 16_000).expect("resample");

  let cfg = MelConfig::whisper_default(); // 16 kHz target, n_mels 80, hop 160
  let mut mel_16k = default_log_mel(&cfg, &resampled).expect("mel of resampled");
  let mut mel_44k = default_log_mel(&cfg, &audio_44k).expect("mel of source");

  let s16 = mel_16k.shape();
  let s44 = mel_44k.shape();
  assert_eq!(s16[0], 80, "n_mels");
  assert_eq!(s44[0], 80, "n_mels");
  // ~101 frames (16 kHz) vs ~276 frames (44.1 kHz) at hop 160.
  assert!(
    s16[1] < 200 && s44[1] > 200,
    "resampled mel T={} should be ~101, source mel T={} should be ~276",
    s16[1],
    s44[1]
  );
  // Sanity: both materialize.
  assert!(!mel_16k.to_vec::<f32>().unwrap().is_empty());
  assert!(!mel_44k.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn default_log_mel_threads_n_mels_override() {
  // A canary-style n_mels=128 config must produce a (128, T) mel — proves the
  // model's `mel_config` is wired into the default `log_mel`, not a hardcoded
  // whisper 80.
  let audio = wav_waveform("mel_cfg", 16_000, ONE_SECOND_16K);
  let model = MockSttModel {
    mel_cfg: MelConfig::whisper_default().with_n_mels(128),
    ..MockSttModel::new(vec![], vec![0], 7, 8)
  };
  let mel = model.log_mel(&audio).expect("log_mel default");
  assert_eq!(mel.shape()[0], 128, "n_mels override threaded");
}

#[test]
fn default_log_mel_threads_log_floor() {
  // The Kaldi floor (1e-8) clamps low-energy bins higher than the Whisper
  // floor (1e-10): `ln(1e-8) ≈ -18.4` vs `ln(1e-10) ≈ -23.0`. Same audio
  // through two models differing only in `log_floor` ⇒ the Kaldi mel min is
  // strictly greater (proves the floor is threaded, not defaulted).
  let audio = wav_waveform("log_floor", 16_000, ONE_SECOND_16K);

  let whisper_model = MockSttModel::new(vec![], vec![0], 7, 8); // whisper floor
  let kaldi_model = MockSttModel {
    mel_cfg: MelConfig::whisper_default().with_log_floor(LogFloor::Kaldi),
    ..MockSttModel::new(vec![], vec![0], 7, 8)
  };

  let mut w = whisper_model.log_mel(&audio).expect("whisper mel");
  let mut k = kaldi_model.log_mel(&audio).expect("kaldi mel");
  let w_min = w
    .to_vec::<f32>()
    .unwrap()
    .into_iter()
    .fold(f32::INFINITY, f32::min);
  let k_min = k
    .to_vec::<f32>()
    .unwrap()
    .into_iter()
    .fold(f32::INFINITY, f32::min);
  assert!(
    k_min > w_min,
    "Kaldi floor must lift the mel min above the Whisper floor: kaldi={k_min} whisper={w_min}"
  );
}

// ─────────────────────────────── load factory ───────────────────────────────

#[test]
fn load_returns_boxed_transcribe_trait_object() {
  // `load` / `load_model` hand back the universal `Box<dyn Transcribe>`; the
  // concrete model (CTC or autoregressive) is the constructor's choice.
  let dir = std::env::temp_dir().join(format!("mlxrs_stt_load_{}", process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  std::fs::write(
    dir.join("config.json"),
    r#"{ "model_type": "whisper", "n_mels": 80 }"#,
  )
  .unwrap();

  let model: Box<dyn Transcribe> = load(&dir.to_string_lossy(), |_bundle| {
    Ok(Box::new(MockCtcModel::from_argmax(&[0, 1], 3, 2)))
  })
  .expect("load constructs a Transcribe via the factory");

  let audio = wav_waveform("load", 16_000, ONE_SECOND_16K);
  let out = model
    .transcribe(&audio, &TranscribeOptions::new())
    .expect("boxed transcribe");
  // frames 0 1 → collapse (no dups) → 0 1 → blank is 2 (neither) → 'a' 'b'.
  assert_eq!(out.text(), "ab");

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────── options + config locks ───────────────────────────

#[test]
fn transcribe_options_defaults_and_builders() {
  let d = TranscribeOptions::new();
  assert!(d.language().is_none(), "auto-detect by default");
  assert_eq!(d.task(), Task::Transcribe);
  assert_eq!(d.temperature(), 0.0, "greedy by default");
  assert!(!d.no_timestamps());

  let mut o = TranscribeOptions::new()
    .with_language("en")
    .with_task(Task::Translate)
    .with_temperature(0.5)
    .with_no_timestamps();
  assert_eq!(o.language(), Some("en"));
  assert_eq!(o.task(), Task::Translate);
  assert_eq!(o.temperature(), 0.5);
  assert!(o.no_timestamps());

  // set_/update_ chain through `&mut Self`; clear_* reverts to absent/default.
  o.set_language("es").update_no_timestamps(false);
  assert_eq!(o.language(), Some("es"));
  assert!(!o.no_timestamps());
  o.clear_language();
  assert!(o.language().is_none());
}

#[test]
fn task_as_str_round_trips() {
  assert_eq!(Task::Transcribe.as_str(), "transcribe");
  assert_eq!(Task::Translate.as_str(), "translate");
  assert_eq!(Task::default(), Task::Transcribe);
  // Display routes through as_str.
  assert_eq!(format!("{}", Task::Translate), "translate");
}

/// Locks in the Whisper preset values — load-bearing for every concrete
/// Whisper port's mel front-end; a silent change is a contract break.
#[test]
fn mel_config_whisper_default_values() {
  let m = MelConfig::whisper_default();
  assert_eq!(m.n_fft(), 400);
  assert_eq!(m.hop_length(), 160);
  assert!(m.win_length().is_none());
  assert_eq!(m.n_mels(), 80);
  assert_eq!(m.sample_rate(), 16_000);
  assert_eq!(m.f_min(), 0.0);
  assert!(m.f_max().is_none());
  assert!(m.log_floor().is_whisper());
}
