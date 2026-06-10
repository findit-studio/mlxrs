//! PROBE-ONLY tests for the Whisper mel-dtype promotion investigation.
//!
//! Hypothesis under test: the F32 log-mel is never cast to the model dtype, so
//! with an fp16 checkpoint MLX's upward type promotion silently runs the whole
//! encoder (and, through cross-attention, the decoder + KV cache) in F32 —
//! roughly doubling compute/bandwidth versus upstream `mlx_whisper` (which
//! casts mel to fp16 in `_get_audio_features`).
//!
//! All tests are `#[ignore]` — run explicitly:
//!
//! ```text
//! cargo test --release --features whisper --test whisper_dtype_probe \
//!   -- --ignored --nocapture
//! ```
//!
//! `probe_head_activation_dtype` / `probe_baseline_rtf` need the turbo
//! checkpoint dir (`MLXRS_WHISPER_BENCH_DIR`, default
//! `/private/tmp/whisper-turbo-bench`) and the probe audio dir
//! (`MLXRS_PROBE_AUDIO_DIR`, default `/private/tmp/whisper-bench-audio`).
#![cfg(feature = "whisper")]

use std::{path::PathBuf, time::Instant};

use mlxrs::{
  Array, Dtype,
  audio::{
    io::load_audio,
    stt::{
      model::{AutoregressiveStt, Transcribe, TranscribeOptions},
      models::whisper::{audio::pad_or_trim, config::ModelDimensions, model::WhisperModel},
    },
  },
  tokenizer::Tokenizer,
  transforms,
};

fn bench_dir() -> PathBuf {
  std::env::var_os("MLXRS_WHISPER_BENCH_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("/private/tmp/whisper-turbo-bench"))
}

fn audio_dir() -> PathBuf {
  std::env::var_os("MLXRS_PROBE_AUDIO_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("/private/tmp/whisper-bench-audio"))
}

fn load_model() -> WhisperModel {
  let dir = bench_dir();
  let cfg_path = dir.join("config.json");
  let cfg_bytes =
    std::fs::read(&cfg_path).unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display()));
  let cfg: serde_json::Value =
    serde_json::from_slice(&cfg_bytes).expect("parse whisper config.json");
  let dims = ModelDimensions::from_dict(&cfg).expect("ModelDimensions::from_dict");
  let model = WhisperModel::load(&dir, dims, Dtype::F16).expect("WhisperModel::load");
  let tokenizer = Tokenizer::from_path(&dir, None).expect("Tokenizer::from_path");
  model
    .with_tokenizer(tokenizer)
    .expect("attach Whisper tokenizer")
}

fn load_wav(name: &str) -> Array {
  let path = audio_dir().join(name);
  let (samples, sr) = load_audio(&path).unwrap_or_else(|e| panic!("load_audio {name}: {e}"));
  assert_eq!(sr, 16_000, "{name} must be 16 kHz mono");
  let n = samples.len() as i32;
  Array::from_slice::<f32>(&samples, &[n]).expect("waveform Array")
}

/// Mechanism probe (no model needed): which dtype do MLX binary ops produce
/// for mixed (F32, F16) inputs? The hypothesis requires upward promotion
/// (result F32); if these came out F16 the poisoning mechanism would be dead
/// on arrival.
#[test]
#[ignore = "investigation probe"]
fn probe_mixed_dtype_promotion() {
  let a32 = Array::full::<f32>(&[8, 16], 1.0).expect("a32");
  let w16 = Array::full::<f32>(&[16, 16], 0.5)
    .expect("w16 f32")
    .astype(Dtype::F16)
    .expect("w16 astype");

  let mm = a32.matmul(&w16).expect("matmul");
  let mm_dtype = mm.dtype().expect("mm dtype");
  println!("matmul(F32, F16) -> {mm_dtype:?}");

  let x32 = Array::full::<f32>(&[16, 16], 1.0).expect("x32");
  let sum = x32.add(&w16).expect("add");
  let sum_dtype = sum.dtype().expect("sum dtype");
  println!("add(F32, F16)    -> {sum_dtype:?}");

  // Control: pure-F16 ops must stay F16 (otherwise "cast mel to f16" could
  // not fix anything).
  let a16 = a32.astype(Dtype::F16).expect("a16");
  let mm16 = a16.matmul(&w16).expect("matmul f16");
  let mm16_dtype = mm16.dtype().expect("mm16 dtype");
  println!("matmul(F16, F16) -> {mm16_dtype:?}");

  assert_eq!(mm_dtype, Dtype::F32, "mixed matmul should promote to F32");
  assert_eq!(sum_dtype, Dtype::F32, "mixed add should promote to F32");
  assert_eq!(mm16_dtype, Dtype::F16, "pure f16 matmul must stay F16");
}

/// End-to-end activation-dtype probe on the real fp16 checkpoint: what dtype
/// does the mel have, and what dtype comes out of the encoder? Under the
/// hypothesis: mel=F32 and encoder output=F32 even though model.dtype()=F16.
#[test]
#[ignore = "needs the turbo checkpoint"]
fn probe_head_activation_dtype() {
  let model = load_model();
  println!("model.dtype() = {:?}", model.dtype());

  // Synthetic 30 s of 16 kHz audio (440 Hz sine) — dtype flow does not depend
  // on audio content.
  let n: usize = 16_000 * 30;
  let samples: Vec<f32> = (0..n)
    .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16_000.0).sin() * 0.1)
    .collect();
  let audio = Array::from_slice::<f32>(&samples, &[n as i32]).expect("audio");

  let mel = model.log_mel(&audio).expect("log_mel");
  println!("mel: dtype={:?} shape={:?}", mel.dtype().expect("mel dtype"), mel.shape());

  let mel_win = pad_or_trim(&mel, 3000, 0).expect("pad_or_trim");
  let enc = model.encode(&mel_win).expect("encode");
  let enc_dtype = enc.dtype().expect("enc dtype");
  println!("encoder output: dtype={enc_dtype:?} shape={:?}", enc.shape());

  // Report-only on the activation dtype: the assert documents the CURRENT
  // (hypothesized-broken) behavior so a fix flips it loudly.
  println!(
    "VERDICT: encoder runs in {enc_dtype:?} while model dtype is {:?} -> {}",
    model.dtype(),
    if enc_dtype == model.dtype() { "OK (hypothesis REFUTED)" } else { "POISONED (hypothesis SUPPORTED)" }
  );
}

/// Wall-clock baseline on real audio with coarse per-stage splits (mel /
/// encode / full transcribe). Run on HEAD for the "before", on the fix branch
/// for the "after".
#[test]
#[ignore = "needs the turbo checkpoint + probe audio"]
fn probe_baseline_rtf() {
  let fixtures: &[(&str, f64)] = &[
    ("test_3speakers.wav", 35.319),
    ("long_dialog_227s.wav", 226.96),
  ];

  let model = load_model();
  let opts = TranscribeOptions::new();

  // Stage probes on the short fixture (after a warm-up so Metal kernel
  // compilation does not pollute the splits).
  let short = load_wav(fixtures[0].0);
  let _ = model.transcribe(&short, &opts).expect("warm-up transcribe");

  let t0 = Instant::now();
  let mel = model.log_mel(&short).expect("log_mel");
  transforms::eval(&[&mel]).expect("eval mel");
  println!("[stage] mel: {:.3}s (dtype={:?})", t0.elapsed().as_secs_f64(), mel.dtype().unwrap());

  let mel_win = pad_or_trim(&mel, 3000, 0).expect("pad_or_trim");
  let t1 = Instant::now();
  let enc = model.encode(&mel_win).expect("encode");
  transforms::eval(&[&enc]).expect("eval enc");
  println!(
    "[stage] encode(1 window): {:.3}s (dtype={:?})",
    t1.elapsed().as_secs_f64(),
    enc.dtype().unwrap()
  );

  println!("\nfixture | duration_s | wall_s | RTF | n_chars | lang");
  for (name, duration_s) in fixtures {
    let audio = load_wav(name);
    let t = Instant::now();
    let out = model
      .transcribe(&audio, &opts)
      .unwrap_or_else(|e| panic!("transcribe {name}: {e}"));
    let wall = t.elapsed().as_secs_f64();
    let text = out.text();
    println!(
      "{name} | {duration_s:.1} | {wall:.2} | {:.4} | {} | {:?}",
      wall / duration_s,
      text.chars().count(),
      out.language()
    );
    eprintln!("[text] {name} :: {}", text.chars().take(120).collect::<String>());
  }
}
