//! PROBE-ONLY A/B harness for the PR #405 quantized-model dispute.
//!
//! Build+run this SAME file on both branches:
//!   - `fix/whisper-max-decoder-ctx-cap` (== origin/main, NO mel-dtype fix)
//!   - `fix/whisper-mel-dtype`           (the PR #405 fix)
//!
//! ```text
//! cargo test --release --features whisper --test whisper_quant_ab -- --ignored --nocapture
//! ```
//!
//! Needs the 8-bit checkpoint at /private/tmp/whisper-turbo-8bit (group_size=64,
//! bits=8, Linear-only — mirrors the colleague's config) and the probe audio at
//! /private/tmp/whisper-bench-audio.
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
  lm::quant::{PerLayerQuantization, Quantization},
  tokenizer::Tokenizer,
  transforms,
};

fn model_dir() -> PathBuf {
  PathBuf::from("/private/tmp/whisper-turbo-8bit")
}

fn audio_dir() -> PathBuf {
  PathBuf::from("/private/tmp/whisper-bench-audio")
}

fn load_quant_model() -> WhisperModel {
  let dir = model_dir();
  let cfg_bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
  let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&cfg).expect("ModelDimensions");
  let q = cfg
    .get("quantization")
    .map(|qv| {
      Quantization::affine(
        qv.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
        qv.get("bits").and_then(|v| v.as_i64()).unwrap_or(8) as i32,
      )
    })
    .map(PerLayerQuantization::from_global);
  let model =
    WhisperModel::load_quantized(&dir, dims, Dtype::F16, q.as_ref()).expect("load_quantized");
  let tokenizer = Tokenizer::from_path(&dir, None).expect("tokenizer");
  model.with_tokenizer(tokenizer).expect("attach tokenizer")
}

fn load_wav(name: &str) -> Array {
  let path = audio_dir().join(name);
  let (samples, sr) = load_audio(&path).unwrap_or_else(|e| panic!("load_audio {name}: {e}"));
  assert_eq!(sr, 16_000);
  Array::from_slice::<f32>(&samples, &[samples.len() as i32]).expect("waveform")
}

#[test]
#[ignore = "quantized A/B probe"]
fn quant_ab() {
  let model = load_quant_model();
  println!("model.dtype() = {:?}", model.dtype());

  // (1) dtype probe — synthetic 30 s sine.
  let n: usize = 16_000 * 30;
  let samples: Vec<f32> = (0..n)
    .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 16_000.0).sin() * 0.1)
    .collect();
  let synth = Array::from_slice::<f32>(&samples, &[n as i32]).expect("synth");
  let mel = model.log_mel(&synth).expect("log_mel");
  let mel_win = pad_or_trim(&mel, 3000, 0).expect("pad");
  let enc = model.encode(&mel_win).expect("encode");
  println!(
    "[dtype] mel={:?} encoder_out={:?}",
    mel.dtype().unwrap(),
    enc.dtype().unwrap()
  );

  // (2) pure-kernel encode timing: fixed work, no decode trajectory. Warm up
  // twice, then time 5 reps.
  for _ in 0..2 {
    let e = model.encode(&mel_win).expect("warm encode");
    transforms::eval(&[&e]).expect("eval");
  }
  let t = Instant::now();
  for _ in 0..5 {
    let e = model.encode(&mel_win).expect("timed encode");
    transforms::eval(&[&e]).expect("eval");
  }
  println!("[encode-kernel] {:.4}s / window (5-rep avg)", t.elapsed().as_secs_f64() / 5.0);

  // (3) decode comparisons on real audio.
  let fixtures: &[(&str, f64)] = &[
    ("test_3speakers.wav", 35.319),
    ("long_dialog_227s.wav", 226.96),
  ];
  let default_opts = TranscribeOptions::new();
  // No-fallback control: greedy temp 0, all quality thresholds disabled — wall
  // time differences reflect kernel speed, not retry-trajectory divergence.
  let nofb_opts = TranscribeOptions::new()
    .with_temperature(0.0)
    .with_compression_ratio_threshold(None)
    .with_logprob_threshold(None)
    .with_no_speech_threshold(None);

  // Warm-up.
  let _ = model.transcribe(&load_wav(fixtures[0].0), &default_opts).expect("warmup");

  for (label, opts) in [("default", &default_opts), ("no-fallback", &nofb_opts)] {
    for (name, dur) in fixtures {
      let audio = load_wav(name);
      let t = Instant::now();
      let out = model.transcribe(&audio, opts).unwrap_or_else(|e| panic!("{name}: {e}"));
      let wall = t.elapsed().as_secs_f64();
      println!(
        "[{label}] {name} | dur={dur:.0}s | wall={wall:.2}s | rtf={:.4} | chars={} | lang={:?}",
        wall / dur,
        out.text().chars().count(),
        out.language()
      );
    }
  }
}
