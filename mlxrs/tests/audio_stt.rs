//! M5 final piece — `audio::stt::Model` trait + `audio::stt::stt_generate`
//! Iterator: end-to-end audio → token decoding driven by a mock STT model.
//!
//! Deterministic, dependency-free: a local `MockSttModel` returns a canned
//! `[1, 1, 1]` encoder state (the trait/loop is opaque to the encoder
//! shape) and a per-step `[1, V]` logits ramp so the decode token sequence
//! is fully predictable. Mirrors the in-crate `tests/lm_generate.rs`
//! MockModel pattern (replicated, not imported — integration tests cannot
//! see crate-private test fixtures).
#![cfg(feature = "audio")]

use std::{fs, path::PathBuf, process};

use mlxrs::{
  Array,
  audio::{
    io::save_wav,
    stt::{
      generate::{SttGenConfig, encode_audio_file, stt_generate},
      model::{MelConfig, Model as SttModel},
    },
  },
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache},
    model::Model as LmModel,
  },
};

/// Process-scoped + named tempfile so parallel test binaries / cases never
/// collide. The audio I/O tests share the same convention.
fn temp_wav(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_audio_stt_{}_{}.wav", process::id(), name));
  p
}

/// Write a deterministic short WAV at the given `sample_rate` and return
/// the path. The exact samples are unimportant — the mock STT model is
/// opaque to the mel content; we just need a real on-disk WAV the loader
/// can parse.
fn make_wav(name: &str, sample_rate: u32, n_samples: usize) -> PathBuf {
  let path = temp_wav(name);
  let samples: Vec<f32> = (0..n_samples)
    .map(|i| ((i as f32) / (n_samples.max(1) as f32) - 0.5) * 0.5)
    .collect();
  save_wav(&path, &samples, sample_rate).unwrap();
  path
}

/// A deterministic, dependency-free [`SttModel`].
///
/// - `encode_audio` records the **shape** of the mel it received (the only
///   per-call observable, so `auto_resample` / `max_audio_seconds` /
///   `mel_config()` overrides can be checked end-to-end) and returns a
///   trivial `[1, 1, 1]` encoder-states array (opaque to the loop).
/// - `decode_step` returns a `[1, V]` ramp `[0..V]` so greedy argmax is
///   always `V - 1` — exactly the LM-loop's `MockModel::ramp(V)` convention.
/// - `bos_token` / `eos_token` are configurable so each test can pin a
///   stop-on-step-K (set `bias`'s argmax to `eos_token`) or never-stop
///   (`eos_token == V` ⇒ unreachable since argmax is `V - 1`).
struct MockSttModel {
  vocab: usize,
  bos: u32,
  eos: u32,
  mel_cfg: MelConfig,
  /// Records every `encode_audio` mel shape so tests can assert the
  /// auto-resample / mel-config-override paths drove a different mel size.
  last_mel_shape: std::cell::RefCell<Option<Vec<usize>>>,
  /// Counts `decode_step` invocations — distinguishes "iterator empty for
  /// 0-second audio" from "iterator yielded 0 tokens by some other path".
  decode_calls: std::cell::RefCell<usize>,
}

impl MockSttModel {
  fn new(vocab: usize) -> Self {
    Self {
      vocab,
      bos: 1,
      // Default: argmax is vocab-1; setting eos = vocab makes it
      // unreachable so the loop runs `max_tokens` to completion.
      eos: vocab as u32,
      mel_cfg: MelConfig::whisper_default(),
      last_mel_shape: std::cell::RefCell::new(None),
      decode_calls: std::cell::RefCell::new(0),
    }
  }
}

impl LmModel for MockSttModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    // The LM forward path is never reached by `stt_generate` — the STT
    // loop drives `decode_step` instead. Return an error so a defect that
    // accidentally routed through the LM forward surfaces loud.
    Err(mlxrs::Error::Backend {
      message: "MockSttModel::forward should not be called from stt_generate".into(),
    })
  }
}

impl SttModel for MockSttModel {
  fn encode_audio(&self, mel: &Array) -> mlxrs::Result<Array> {
    *self.last_mel_shape.borrow_mut() = Some(mel.shape());
    // Trivial encoder states; the loop forwards this unchanged to
    // `decode_step` without inspecting it.
    Array::from_slice::<f32>(&[0.0_f32], &[1, 1, 1])
  }

  fn decode_step(
    &self,
    _token: u32,
    _encoder_states: &Array,
    _cache: &mut [Box<dyn KvCache>],
  ) -> mlxrs::Result<Array> {
    *self.decode_calls.borrow_mut() += 1;
    let bias: Vec<f32> = (0..self.vocab).map(|i| i as f32).collect();
    Array::from_slice::<f32>(&bias, &(1_usize, self.vocab))
  }

  fn mel_config(&self) -> MelConfig {
    self.mel_cfg
  }

  fn bos_token(&self) -> u32 {
    self.bos
  }

  fn eos_token(&self) -> u32 {
    self.eos
  }
}

/// `decode_step` that returns the wrong logits shape — drives the loop's
/// `[1, V]` rank/zero-axis guard.
struct BadShapeModel;
impl LmModel for BadShapeModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "unused".into(),
    })
  }
}
impl SttModel for BadShapeModel {
  fn encode_audio(&self, _mel: &Array) -> mlxrs::Result<Array> {
    Array::from_slice::<f32>(&[0.0_f32], &[1, 1, 1])
  }
  fn decode_step(&self, _t: u32, _e: &Array, _c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    // Wrong shape: [V] not [1, V].
    Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0], &[3])
  }
  fn bos_token(&self) -> u32 {
    0
  }
  fn eos_token(&self) -> u32 {
    99
  }
}

/// `decode_step` that always errors — drives the "step error yielded once,
/// iterator fuses" contract.
struct FailDecodeModel;
impl LmModel for FailDecodeModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "unused".into(),
    })
  }
}
impl SttModel for FailDecodeModel {
  fn encode_audio(&self, _mel: &Array) -> mlxrs::Result<Array> {
    Array::from_slice::<f32>(&[0.0_f32], &[1, 1, 1])
  }
  fn decode_step(&self, _t: u32, _e: &Array, _c: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "mock decode_step failure".into(),
    })
  }
  fn bos_token(&self) -> u32 {
    0
  }
  fn eos_token(&self) -> u32 {
    99
  }
}

/// A `SttModel` that does NOT override `decode_step` — drives the default
/// "STT model needs `decode_step` override" `Err`.
struct DefaultDecodeModel;
impl LmModel for DefaultDecodeModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "unused".into(),
    })
  }
}
impl SttModel for DefaultDecodeModel {
  fn encode_audio(&self, _mel: &Array) -> mlxrs::Result<Array> {
    Array::from_slice::<f32>(&[0.0_f32], &[1, 1, 1])
  }
  // `decode_step` inherits the trait default (unimplemented).
  fn bos_token(&self) -> u32 {
    0
  }
  fn eos_token(&self) -> u32 {
    99
  }
}

fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&CacheConfig {
    num_hidden_layers: layers,
    sliding_window: None,
  })
}

/// One second of audio at the Whisper sample rate. Just enough samples for
/// the default `log_mel_spectrogram(n_fft=400, hop=160)` to fit a few
/// frames.
const ONE_SECOND_16K: usize = 16_000;

// ───────────────────────── pipeline smoke ─────────────────────────

/// End-to-end smoke: load a WAV at the model's expected sample rate, drive
/// the iterator, and verify it yields exactly `max_tokens` greedy-argmax
/// tokens (no EOS hit because `eos = vocab` is unreachable).
#[test]
fn stt_generate_pipeline_smoke() {
  let path = make_wav("smoke", 16_000, ONE_SECOND_16K);
  let model = MockSttModel::new(5); // ramp → argmax == 4 every step
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 3,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let it = stt_generate(&model, &path, cache(1), cfg).unwrap();
  let toks: Vec<u32> = it.map(|r| r.unwrap().token).collect();
  // `eos == vocab == 5` is never reached (argmax is vocab-1 == 4); the loop
  // runs to `max_tokens` and stops with "length"-finish-reason equivalent
  // (the STT loop doesn't carry the LM loop's detokenizer-side
  // `finish_reason` string, so we just observe the count).
  assert_eq!(toks, vec![4, 4, 4], "max_tokens greedy decode");
  assert_eq!(*model.decode_calls.borrow(), 3);
  let _ = fs::remove_file(&path);
}

/// EOS hit: setting `eos_token` to the model's argmax (`vocab - 1`) makes
/// the loop stop after the first token (yielded as the final step, mirror-
/// ing the LM loop's "yield-then-fuse" eos contract).
#[test]
fn stt_generate_stops_on_eos() {
  let path = make_wav("eos", 16_000, ONE_SECOND_16K);
  let mut model = MockSttModel::new(5);
  model.eos = 4; // argmax is 4 ⇒ first step IS the eos token.
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 100,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let toks: Vec<u32> = stt_generate(&model, &path, cache(1), cfg)
    .unwrap()
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(toks, vec![4], "EOS token yielded once, then iteration ends");
  let _ = fs::remove_file(&path);
}

/// Sample-rate mismatch + `auto_resample = true`: a 44.1 kHz WAV at a
/// 16 kHz model triggers resample → smaller mel-spec, but the loop still
/// drives `decode_step` once per step. The `last_mel_shape` is the
/// post-resample-driven `(n_mels, T)`, so its `T` axis is much smaller than
/// the no-resample case (proves the resample path ran without inspecting
/// the exact sample count, which depends on `audio_io::load_wav`'s rounded
/// sample count).
#[test]
fn stt_generate_resamples_when_sr_mismatch() {
  // 44.1 kHz, 1 second of audio.
  let path = make_wav("resample", 44_100, 44_100);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 1,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    auto_resample: true,
    ..SttGenConfig::default()
  };
  let toks: Vec<u32> = stt_generate(&model, &path, cache(1), cfg)
    .unwrap()
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(toks, vec![2], "ramp(3) argmax");
  // mel-spec received by the encoder must be (n_mels=80, T), with T set by
  // the post-resample 16 kHz sample count, not the original 44.1 kHz one.
  // Whisper default: n_mels=80, hop=160 ⇒ 16k samples → ~101 frames.
  let mel_shape = model.last_mel_shape.borrow().clone().unwrap();
  assert_eq!(mel_shape.len(), 2, "mel-spec rank");
  assert_eq!(mel_shape[0], 80, "n_mels (whisper default)");
  // Loose upper bound: a 1-second 16 kHz mel-spec at hop 160 is ~101 frames.
  // A 44.1 kHz mel-spec at the same hop would be ~276 frames — so anything
  // < 200 frames proves resample-to-16k drove the mel shape.
  assert!(
    mel_shape[1] < 200,
    "mel T={} should be the post-resample (16k) frame count (~101), \
     not the original 44.1k (~276); resample didn't run",
    mel_shape[1]
  );
  let _ = fs::remove_file(&path);
}

/// `auto_resample = false` with a sample-rate mismatch: pipeline errors out
/// before any model call — the recoverable `Err` documented on the cfg.
#[test]
fn stt_generate_rejects_sr_mismatch_when_resample_off() {
  let path = make_wav("no_resample", 44_100, 44_100);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig {
    auto_resample: false,
    ..SttGenConfig::default()
  };
  // SttGenerator is not Debug, so route via ok/err rather than pattern-
  // matching the full Result (which would need Debug on the iterator).
  let err = stt_generate(&model, &path, cache(1), cfg)
    .err()
    .expect("auto_resample=false rejects mismatched sample rate");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("auto_resample"),
        "error mentions auto_resample, got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert_eq!(
    *model.decode_calls.borrow(),
    0,
    "decode never ran (rejection before model call)"
  );
  let _ = fs::remove_file(&path);
}

/// `max_audio_seconds` cap: a 2-second WAV with a 1-second cap rejects
/// **before** the mel-spec allocation (the encoder MUST NOT have been
/// called).
#[test]
fn stt_generate_rejects_audio_longer_than_max() {
  let path = make_wav("too_long", 16_000, 2 * ONE_SECOND_16K);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig {
    max_audio_seconds: 1.0,
    ..SttGenConfig::default()
  };
  let err = stt_generate(&model, &path, cache(1), cfg)
    .err()
    .expect("over-cap audio rejected");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("max_audio_seconds"),
        "error mentions max_audio_seconds, got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert!(
    model.last_mel_shape.borrow().is_none(),
    "encoder was NOT called (cap rejected before allocation)"
  );
  let _ = fs::remove_file(&path);
}

/// Regression (Codex adversarial-review round 1, high): the
/// `max_audio_seconds` cap is checked against the **source** duration —
/// the load_wav `(samples, src_sr)` pair — BEFORE the resample pass
/// allocates a second buffer. A 2-second 44.1 kHz source (88200 samples)
/// at a 16 kHz-target model with a 1-second cap MUST reject without
/// resampling. The encoder never being called is the proxy for the
/// resample-then-reject path being closed.
#[test]
fn stt_generate_rejects_over_cap_before_resample() {
  // 2 seconds of audio at 44.1 kHz: 88200 samples; target_sr = 16000.
  let path = make_wav("over_cap_pre_resample", 44_100, 2 * 44_100);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig {
    max_audio_seconds: 1.0,
    auto_resample: true, // resample WOULD run if the cap check came after
    ..SttGenConfig::default()
  };
  let err = stt_generate(&model, &path, cache(1), cfg)
    .err()
    .expect("over-cap source rejected pre-resample");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("max_audio_seconds"),
        "error mentions max_audio_seconds, got {message}"
      );
      // The message must reference the SOURCE sample rate to prove the
      // check ran on the source duration, not the post-resample one.
      assert!(
        message.contains("44100"),
        "error references the source sample_rate=44100, got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert!(
    model.last_mel_shape.borrow().is_none(),
    "encoder was NOT called (cap rejected BEFORE resample)"
  );
  let _ = fs::remove_file(&path);
}

/// Regression (Codex adversarial-review round 1, medium): an empty WAV
/// surfaces as a clear `Error::Backend` from `stt_generate` — the encoder
/// is NEVER called. Concrete encoders can reasonably assume at least one
/// mel frame; fabricating a zero-frame mel would silently push the failure
/// deep into per-model code.
#[test]
fn stt_generate_rejects_empty_audio() {
  let path = make_wav("empty", 16_000, 0);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig::default();
  let err = stt_generate(&model, &path, cache(1), cfg)
    .err()
    .expect("empty audio rejected");
  match err {
    mlxrs::Error::Backend { message } => {
      assert!(
        message.contains("empty"),
        "error mentions empty, got {message}"
      );
    }
    other => panic!("expected Backend error, got {other:?}"),
  }
  assert!(
    model.last_mel_shape.borrow().is_none(),
    "encoder was NOT called for empty audio"
  );
  let _ = fs::remove_file(&path);
}

/// Custom `MelConfig` override (e.g. canary's `n_mels=128`): the encoder
/// receives a `(128, T)` mel-spec — proves `model.mel_config()` is wired
/// into the pipeline, not a hardcoded Whisper default.
#[test]
fn stt_generate_uses_mel_config_override() {
  let path = make_wav("mel_cfg", 16_000, ONE_SECOND_16K);
  let mut model = MockSttModel::new(3);
  model.mel_cfg = MelConfig {
    n_mels: 128,
    ..MelConfig::whisper_default()
  };
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 1,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let _ = stt_generate(&model, &path, cache(1), cfg)
    .unwrap()
    .map(|r| r.unwrap().token)
    .collect::<Vec<_>>();
  let mel_shape = model.last_mel_shape.borrow().clone().unwrap();
  assert_eq!(
    mel_shape[0], 128,
    "n_mels override = 128 (canary-style), not 80 (whisper default)"
  );
  let _ = fs::remove_file(&path);
}

/// `encode_audio_file` runs the load → resample → max-seconds → log-mel →
/// encode subset of the pipeline (steps 1-5 of `stt_generate`'s doc) and
/// returns the encoder states. The mock returns a `[1, 1, 1]` array;
/// observing that shape proves `encode_audio` was called.
#[test]
fn encode_audio_file_smoke() {
  let path = make_wav("enc_file", 16_000, ONE_SECOND_16K);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig::default();
  let enc = encode_audio_file(&model, &path, &cfg).unwrap();
  assert_eq!(
    enc.shape(),
    vec![1, 1, 1],
    "MockSttModel::encode_audio returns [1,1,1]"
  );
  assert!(model.last_mel_shape.borrow().is_some());
  assert_eq!(
    *model.decode_calls.borrow(),
    0,
    "encode_audio_file does NOT drive the decode loop"
  );
  let _ = fs::remove_file(&path);
}

/// The trait default `decode_step` returns a recoverable `Err` with the
/// "needs override" message — the iterator yields that as its first item.
#[test]
fn decode_step_default_errors_with_clear_message() {
  let path = make_wav("default_decode", 16_000, ONE_SECOND_16K);
  let model = DefaultDecodeModel;
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 5,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let mut it = stt_generate(&model, &path, cache(1), cfg).unwrap();
  match it.next().expect("an item") {
    Err(mlxrs::Error::Backend { message }) => {
      assert!(
        message.contains("decode_step"),
        "error mentions decode_step, got {message}"
      );
    }
    other => panic!("expected Backend Err, got {other:?}"),
  }
  // Fused after the error: no further items, no panic.
  assert!(it.next().is_none(), "iterator fuses after decode_step Err");
  let _ = fs::remove_file(&path);
}

/// A `decode_step` returning the wrong logits shape (`[V]` instead of
/// `[1, V]`) surfaces a recoverable `Err(ShapeMismatch)` — the loop's
/// per-step rank/zero-axis guard.
#[test]
fn stt_generate_rejects_bad_decode_step_shape() {
  let path = make_wav("bad_shape", 16_000, ONE_SECOND_16K);
  let model = BadShapeModel;
  let cfg = SttGenConfig::default();
  let mut it = stt_generate(&model, &path, cache(1), cfg).unwrap();
  match it.next().expect("an item") {
    Err(mlxrs::Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("[1, V]"),
        "error mentions [1, V], got {message}"
      );
    }
    other => panic!("expected ShapeMismatch, got {other:?}"),
  }
  assert!(it.next().is_none(), "iterator fuses after the Err");
  let _ = fs::remove_file(&path);
}

/// A `decode_step` error is yielded once and the iterator fuses — same
/// contract the LM loop guarantees.
#[test]
fn stt_generate_decode_step_error_fuses() {
  let path = make_wav("decode_fail", 16_000, ONE_SECOND_16K);
  let model = FailDecodeModel;
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 5,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let mut it = stt_generate(&model, &path, cache(1), cfg).unwrap();
  let first = it.next().expect("an item");
  assert!(first.is_err(), "decode_step error yielded as Err");
  assert!(
    it.next().is_none(),
    "iteration ends after the error (no panic, no re-entry)"
  );
  let _ = fs::remove_file(&path);
}

/// Locks in the Whisper preset values. These are documented in the doc-
/// comment and have load-bearing values (every concrete Whisper port
/// computes its mel-spec against these); a silent change is a contract
/// break.
#[test]
fn mel_config_whisper_default_values() {
  let m = MelConfig::whisper_default();
  assert_eq!(m.n_fft, 400);
  assert_eq!(m.hop_length, 160);
  assert!(m.win_length.is_none());
  assert_eq!(m.n_mels, 80);
  assert_eq!(m.sample_rate, 16_000);
  assert_eq!(m.f_min, 0.0);
  assert!(m.f_max.is_none());
}

/// `max_tokens == 0`: the iterator is empty (no decode_step calls).
#[test]
fn stt_generate_zero_max_tokens_is_empty() {
  let path = make_wav("zero_max", 16_000, ONE_SECOND_16K);
  let model = MockSttModel::new(3);
  let cfg = SttGenConfig {
    lm: mlxrs::lm::generate::GenConfig {
      max_tokens: 0,
      ..mlxrs::lm::generate::GenConfig::default()
    },
    ..SttGenConfig::default()
  };
  let n = stt_generate(&model, &path, cache(1), cfg).unwrap().count();
  assert_eq!(n, 0);
  assert_eq!(*model.decode_calls.borrow(), 0);
  let _ = fs::remove_file(&path);
}

/// `SttGenConfig::default()` carries the Whisper segment 30-second cap and
/// `auto_resample = true`.
#[test]
fn stt_gen_config_defaults_are_whisper_shape() {
  let c = SttGenConfig::default();
  assert!((c.max_audio_seconds - 30.0).abs() < 1e-6, "30s whisper cap");
  assert!(c.auto_resample, "auto_resample default = true");
}
