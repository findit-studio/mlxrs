//! MEASUREMENT-ONLY end-to-end Whisper RTF (real-time-factor) benchmark.
//!
//! `#[ignore]` by default — run explicitly:
//!
//! ```text
//! cargo test --release --features whisper --test whisper_rtf_bench \
//!   -- --ignored --nocapture
//! ```
//!
//! Loads `whisper-large-v3-turbo` (dense fp16) ONCE, loads each WAV fixture via
//! `mlxrs::audio::io::load_audio`, transcribes with default/greedy
//! `TranscribeOptions`, and prints `fixture | duration_s | wall_s | RTF`.
//! RTF = wall_seconds / audio_duration_seconds (lower is better; < 1 ⇒ faster
//! than real-time). Warm-up transcribes the shortest fixture once before timing
//! (model load + first-call activation traces + Metal kernel compile out of the
//! measurement), then one timed run per fixture.
//!
//! The checkpoint dir is resolved from `MLXRS_WHISPER_BENCH_DIR` (defaults to
//! the assembled `/private/tmp/whisper-turbo-bench`): it must hold the dense
//! `weights.safetensors`, the native-MLX `config.json`, and the Whisper
//! `tokenizer.json` (+ `tokenizer_config.json`). The fixtures dir is
//! `MLXRS_AUDIO_FIXTURES_DIR` (defaults to the repo-sibling
//! `audio-fixtures/pcm_s16le`).
#![cfg(feature = "whisper")]

use std::{path::PathBuf, time::Instant};

use mlxrs::{
  Array, Dtype,
  audio::{
    io::load_audio,
    stt::{
      model::{Transcribe, TranscribeOptions},
      models::whisper::{config::ModelDimensions, model::WhisperModel},
    },
  },
  tokenizer::Tokenizer,
};

/// One fixture: file name + manifest audio duration (seconds).
struct Fixture {
  name: &'static str,
  duration_s: f64,
}

/// The representative short→long set (the very longest are skipped to keep
/// runtime sane). Durations are the `audio-fixtures/manifest.json` values.
const FIXTURES: &[Fixture] = &[
  Fixture {
    name: "07_yuhewei_dongbei_english.wav",
    duration_s: 25.263313,
  },
  Fixture {
    name: "02_pyannote_sample.wav",
    duration_s: 30.000000,
  },
  Fixture {
    name: "04_three_speaker.wav",
    duration_s: 39.973313,
  },
  Fixture {
    name: "10_mrbeast_clean_water.wav",
    duration_s: 619.498688,
  },
  Fixture {
    name: "09_mrbeast_dollar_date.wav",
    duration_s: 1041.984000,
  },
];

fn bench_dir() -> PathBuf {
  std::env::var_os("MLXRS_WHISPER_BENCH_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("/private/tmp/whisper-turbo-bench"))
}

fn fixtures_dir() -> PathBuf {
  std::env::var_os("MLXRS_AUDIO_FIXTURES_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("/Users/al/Developer/findit-studio/audio-fixtures/pcm_s16le"))
}

/// Load a 16 kHz mono WAV fixture into a waveform `Array` (the real
/// `load_audio` path a caller drives before `transcribe`).
fn load_fixture(name: &str) -> Array {
  let path = fixtures_dir().join(name);
  let (samples, sr) = load_audio(&path).unwrap_or_else(|e| panic!("load_audio {name}: {e}"));
  assert_eq!(sr, 16_000, "{name} must be 16 kHz mono");
  let n = samples.len() as i32;
  Array::from_slice::<f32>(&samples, &[n]).expect("waveform Array")
}

/// Build the Whisper turbo model with its tokenizer attached from `bench_dir`.
fn load_model() -> WhisperModel {
  let dir = bench_dir();
  let cfg_path = dir.join("config.json");
  let cfg_bytes =
    std::fs::read(&cfg_path).unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display()));
  let cfg: serde_json::Value =
    serde_json::from_slice(&cfg_bytes).expect("parse whisper config.json");
  let dims = ModelDimensions::from_dict(&cfg).expect("ModelDimensions::from_dict");

  // Dense fp16 checkpoint (matches the on-disk weights dtype).
  let model = WhisperModel::load(&dir, dims, Dtype::F16).expect("WhisperModel::load");

  let tokenizer = Tokenizer::from_path(&dir, None).expect("Tokenizer::from_path (tokenizer.json)");
  model
    .with_tokenizer(tokenizer)
    .expect("attach Whisper tokenizer")
}

#[test]
#[ignore = "measurement-only RTF benchmark; needs the turbo checkpoint + fixtures"]
fn whisper_rtf_bench() {
  let model = load_model();
  let opts = TranscribeOptions::new(); // greedy default, auto-detect language, transcribe

  // Warm-up on the SHORTEST fixture (excluded from the timing): pays the
  // first-call activation traces + Metal kernel compile up front.
  let warm = FIXTURES
    .iter()
    .min_by(|a, b| a.duration_s.total_cmp(&b.duration_s))
    .expect("at least one fixture");
  {
    let audio = load_fixture(warm.name);
    let out = model.transcribe(&audio, &opts).expect("warm-up transcribe");
    eprintln!(
      "[warmup] {} -> {} chars, lang={:?}",
      warm.name,
      out.text().chars().count(),
      out.language()
    );
  }

  println!("\nfixture | duration_s | wall_s | RTF | n_chars | lang");
  println!("------- | ---------- | ------ | --- | ------- | ----");
  for fx in FIXTURES {
    let audio = load_fixture(fx.name);
    let t0 = Instant::now();
    let out = model
      .transcribe(&audio, &opts)
      .unwrap_or_else(|e| panic!("transcribe {}: {e}", fx.name));
    let wall_s = t0.elapsed().as_secs_f64();
    let rtf = wall_s / fx.duration_s;
    let text = out.text();
    let n_chars = text.chars().count();
    println!(
      "{} | {:.1} | {:.2} | {:.4} | {} | {:?}",
      fx.name,
      fx.duration_s,
      wall_s,
      rtf,
      n_chars,
      out.language()
    );
    // Sanity: a real decode must produce a non-trivial transcript (a broken
    // transcribe would yield an empty/degenerate string and a meaningless RTF).
    assert!(
      n_chars > 8,
      "{} produced an implausibly short transcript ({n_chars} chars): {:?}",
      fx.name,
      text.chars().take(80).collect::<String>()
    );
    // Echo a transcript preview to stderr for the plausibility check.
    eprintln!(
      "[text] {} :: {}",
      fx.name,
      text.chars().take(160).collect::<String>()
    );
  }
}
