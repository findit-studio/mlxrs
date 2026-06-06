//! End-to-end per-token decode timing — the measurement #366 lacked.
//!
//! These are `#[ignore]`d timing tests: they need a local whisper checkpoint and
//! are not deterministic. Unlike an isolated micro-bench of `GreedyDecoder::update`,
//! this drives the REAL production per-token path on a loaded model with
//! fabricated encoder features, so it actually sees the GPU<->host round-trip
//! that dominates per-token cost — the structural serialization tracked in #369.
//! The `update`-in-isolation bench is exactly what made #366 look like a 3x win
//! while the real decode loop did not improve.
//!
//! Run:
//! `cargo test -p mlxrs --release --features whisper --lib decode_per_token -- --ignored --nocapture`

use std::{path::Path, time::Instant};

use super::{
  ApplyTimestampRules, GreedyDecoder, HFTokenizerWrapper, Task, TranscribeOptions,
  last_position_row, transcribe,
};
use crate::{
  Array, Dtype, Result,
  audio::stt::{
    model::AutoregressiveStt,
    models::whisper::{config::ModelDimensions, model::WhisperModel},
  },
  ops,
  tokenizer::Tokenizer,
  transforms,
};

/// A local checkpoint to time against (gitignored model dir). Skipped if absent
/// so the test is a no-op on a machine without weights.
const MODEL_DIR: &str = "/Users/al/Developer/findit-studio/mlxrs/models/whisper-large-v3-turbo";
const SOT: u32 = 50258; // any in-vocab token id works for timing
const STEPS: usize = 64;
const WARMUP: usize = 8;

fn min_med(mut v: Vec<f64>) -> (f64, f64) {
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  (v[0], v[v.len() / 2])
}

/// Per-token cost of the CURRENT serialized loop:
/// `decode_tokens(&[u32]) -> last_position_row -> argmax -> blocking eval ->
/// item() host readback -> feed back`. Every token pays the full GPU<->host
/// round-trip with zero GPU/host overlap.
#[test]
#[ignore = "needs a local whisper checkpoint; timing harness — run with --ignored --nocapture"]
fn decode_per_token_baseline_serialized() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP decode_per_token_baseline_serialized: {MODEL_DIR} not found");
    return Ok(());
  }

  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let (n_ctx, n_state) = (dims.n_audio_ctx(), dims.n_audio_state());
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;

  // Fabricated encoder output (1, n_audio_ctx, n_audio_state). Content is
  // irrelevant to the per-token TIMING — we always feed back the argmax.
  let fkey = ops::random::key(0)?;
  let features = ops::random::normal(
    &[1i32, n_ctx as i32, n_state as i32],
    Dtype::F16,
    0.0,
    1.0,
    &fkey,
  )?;
  transforms::eval(&[&features])?;

  let mut cache = None;
  let mut tok = SOT;

  for _ in 0..WARMUP {
    let (logits3d, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&logits3d)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
  }

  let mut times = Vec::with_capacity(STEPS);
  for _ in 0..STEPS {
    let t0 = Instant::now();
    let (logits3d, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&logits3d)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
    times.push(t0.elapsed().as_secs_f64() * 1e3);
  }

  let (min, med) = min_med(times);
  println!("\nDECODE per-token (serialized / current main, whisper-large-v3 f16, {STEPS} steps):");
  println!("  min={min:.2}ms  median={med:.2}ms");
  Ok(())
}

/// Per-step decode cost as a function of self-attention CACHE LENGTH — the
/// mlxrs mirror of `py_decode_scaling.py`. Decodes 400 steps from SOT against
/// fabricated features (the cache grows 1 -> 400) and reports the mean per-step
/// time in cache-length buckets. Python's decode is flat with cache length
/// (weight-loading-bound); if mlxrs ramps, the per-token forward has an
/// O(cache-length) cost the reference does not — the decode-loop gap root cause.
#[test]
#[ignore = "needs a local whisper checkpoint; timing harness — run with --ignored --nocapture"]
fn decode_scaling_vs_cache() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP decode_scaling_vs_cache: {MODEL_DIR} not found");
    return Ok(());
  }

  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let (n_ctx, n_state) = (dims.n_audio_ctx(), dims.n_audio_state());
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;

  let fkey = ops::random::key(0)?;
  let features = ops::random::normal(
    &[1i32, n_ctx as i32, n_state as i32],
    Dtype::F16,
    0.0,
    1.0,
    &fkey,
  )?;
  transforms::eval(&[&features])?;

  // Warmup (JIT + caches), then decode 400 fresh steps from SOT.
  let mut cache = None;
  let mut tok = SOT;
  for _ in 0..WARMUP {
    let (logits3d, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&logits3d)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
  }

  let mut cache = None;
  let mut tok = SOT;
  let mut per: Vec<f64> = Vec::with_capacity(400);
  for _ in 0..400 {
    let t0 = Instant::now();
    let (logits3d, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&logits3d)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
    per.push(t0.elapsed().as_secs_f64() * 1e3);
  }

  let mean = |lo: usize, hi: usize| -> f64 {
    let s = &per[lo..hi];
    s.iter().sum::<f64>() / s.len() as f64
  };
  println!("\nMLXRS decode/step vs cache length (turbo f16, 400 steps from SOT):");
  println!("  cache ~32  (steps 0-64)    : {:.2} ms/step", mean(0, 64));
  println!(
    "  cache ~125 (steps 100-150) : {:.2} ms/step",
    mean(100, 150)
  );
  println!(
    "  cache ~225 (steps 200-250) : {:.2} ms/step",
    mean(200, 250)
  );
  println!(
    "  cache ~335 (steps 320-384) : {:.2} ms/step",
    mean(320, 384)
  );
  println!("  (Python reference is flat ~4-5 ms/step across all cache lengths)");
  Ok(())
}

/// Sustained-load decode heat test: decode many steps with the cache RESET every
/// 400 (so per-step compute stays in the same short-cache regime), sampling
/// per-step time and GPU memory. Isolates SUSTAINED-LOAD degradation from
/// cache-length scaling: if per-step climbs while memory is flat, the GPU is
/// throttling under sustained load (thermal/power); if memory grows, the port is
/// accumulating GPU buffers (fixable). The real-transcribe per-step is ~7-9 ms vs
/// this path's cold ~3 ms — this test attributes that gap.
#[test]
#[ignore = "needs a local whisper checkpoint; timing harness — run with --ignored --nocapture"]
fn decode_sustained_heat() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP decode_sustained_heat: {MODEL_DIR} not found");
    return Ok(());
  }

  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let (n_ctx, n_state) = (dims.n_audio_ctx(), dims.n_audio_state());
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;

  let fkey = ops::random::key(0)?;
  let features = ops::random::normal(
    &[1i32, n_ctx as i32, n_state as i32],
    Dtype::F16,
    0.0,
    1.0,
    &fkey,
  )?;
  transforms::eval(&[&features])?;

  let mut cache = None;
  let mut tok = SOT;
  for _ in 0..WARMUP {
    let (l, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&l)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
  }

  let total = 6000usize;
  let mut per: Vec<f64> = Vec::with_capacity(total);
  let mut cache = None;
  let mut tok = SOT;
  let mut depth = 0usize;
  for i in 0..total {
    let t0 = Instant::now();
    let (l, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
    cache = Some(c);
    let row = last_position_row(&l)?;
    let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
    transforms::eval(&[&idx])?;
    tok = idx.item::<u32>()?;
    per.push(t0.elapsed().as_secs_f64() * 1e3);
    depth += 1;
    if depth >= 400 {
      cache = None;
      tok = SOT;
      depth = 0;
    }
    if (i + 1) % 1500 == 0 {
      let act = crate::memory::active_memory().unwrap_or(0) / (1024 * 1024);
      let cac = crate::memory::cache_memory().unwrap_or(0) / (1024 * 1024);
      let avg = per[i + 1 - 500..=i].iter().sum::<f64>() / 500.0;
      println!(
        "  step {:5}: last-500 avg = {:.2} ms/step  | active={}MB cache={}MB",
        i + 1,
        avg,
        act,
        cac
      );
    }
  }

  let mean = |lo: usize, hi: usize| per[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
  println!(
    "\nMLXRS sustained decode heat ({total} steps, cache reset every 400, short-cache regime):"
  );
  println!("  first 500 : {:.2} ms/step", mean(0, 500));
  println!("  last 500  : {:.2} ms/step", mean(total - 500, total));
  println!(
    "  (last >> first with flat memory => sustained/thermal throttle; memory growth => accumulation)"
  );
  Ok(())
}

/// Encoder per-window cost (min/median over 20 warm forwards) — the mlxrs mirror
/// of `py_encoder_bench.py`. The 32-layer audio encoder runs a full
/// `(1500, 1500)` self-attention per block; this measures the fused-SDPA encoder
/// path against the Python naive baseline (~493 ms/window on turbo f16).
#[test]
#[ignore = "needs a local whisper checkpoint; timing harness — run with --ignored --nocapture"]
fn encoder_per_window_timing() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP encoder_per_window_timing: {MODEL_DIR} not found");
    return Ok(());
  }

  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let n_mels = dims.n_mels();
  let n_ctx = dims.n_audio_ctx();
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;

  // One 30s window of mel: (1, N_FRAMES = conv2.stride(2) * n_audio_ctx, n_mels).
  let frames = (2 * n_ctx) as i32;
  let mkey = ops::random::key(7)?;
  let mel = ops::random::normal(&[1i32, frames, n_mels as i32], Dtype::F16, 0.0, 1.0, &mkey)?;
  transforms::eval(&[&mel])?;

  for _ in 0..3 {
    let f = super::encode_once(&model, &mel)?;
    transforms::eval(&[&f])?;
  }
  let mut times = Vec::with_capacity(20);
  for _ in 0..20 {
    let t0 = Instant::now();
    let f = super::encode_once(&model, &mel)?;
    transforms::eval(&[&f])?;
    times.push(t0.elapsed().as_secs_f64() * 1e3);
  }
  let (min, med) = min_med(times);
  println!("\nMLXRS encoder/window (turbo f16, 32-layer audio enc, {frames}x{n_mels} mel):");
  println!("  min={min:.1}ms  median={med:.1}ms   (Python naive baseline ~493 ms/window)");
  Ok(())
}

/// Reproduce the transcribe workload's encode/decode INTERLEAVE: per "window",
/// run the 32-layer encoder, then decode N tokens against its output — exactly
/// the alternation the real seek loop does, which the pure-decode heat test
/// omits. If the decode per-step here jumps to the in-situ ~11 ms (vs the pure
/// ~3 ms) and/or memory climbs, the encode/decode interleave (memory pressure /
/// sustained-power throttle) is the decode-loop gap; if it stays ~3 ms, the gap
/// is elsewhere in the transcribe machinery.
#[test]
#[ignore = "needs a local whisper checkpoint; timing harness — run with --ignored --nocapture"]
fn decode_with_encoder_interleave() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP decode_with_encoder_interleave: {MODEL_DIR} not found");
    return Ok(());
  }

  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let n_mels = dims.n_mels() as i32;
  let frames = (2 * dims.n_audio_ctx()) as i32;
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;

  let mkey = ops::random::key(11)?;
  let mel = ops::random::normal(&[1i32, frames, n_mels], Dtype::F16, 0.0, 1.0, &mkey)?;
  transforms::eval(&[&mel])?;

  // Warm encoder + decode kernels once.
  {
    let f = model.encode(&mel)?;
    transforms::eval(&[&f])?;
    let (l, _) = model.decode_tokens(&[SOT], &f, None)?;
    transforms::eval(&[&l])?;
  }

  let windows = 30usize;
  let steps_per = 115usize;
  let mut win_decode_ms: Vec<f64> = Vec::with_capacity(windows);
  let mut enc_total_ms = 0.0;
  for w in 0..windows {
    let te = Instant::now();
    let features = model.encode(&mel)?;
    transforms::eval(&[&features])?;
    enc_total_ms += te.elapsed().as_secs_f64() * 1e3;

    let mut cache = None;
    let mut tok = SOT;
    let t0 = Instant::now();
    for _ in 0..steps_per {
      let (l, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
      cache = Some(c);
      let mut idx = ops::misc::argmax(&last_position_row(&l)?, Some(-1), false)?;
      transforms::eval(&[&idx])?;
      tok = idx.item::<u32>()?;
    }
    let per = t0.elapsed().as_secs_f64() * 1e3 / steps_per as f64;
    win_decode_ms.push(per);
    if w == 0 || w == windows / 2 || w == windows - 1 {
      let act = crate::memory::active_memory().unwrap_or(0) / (1024 * 1024);
      let cac = crate::memory::cache_memory().unwrap_or(0) / (1024 * 1024);
      println!(
        "  win {:2}: decode={:.2} ms/step  enc={:.0}ms  | active={}MB cache={}MB",
        w,
        per,
        enc_total_ms / (w + 1) as f64,
        act,
        cac
      );
    }
  }
  let mean = win_decode_ms.iter().sum::<f64>() / windows as f64;
  println!("\nMLXRS encode/decode INTERLEAVE (turbo f16, {windows} windows x {steps_per} steps):");
  println!(
    "  mean decode = {mean:.2} ms/step   enc = {:.0} ms/window",
    enc_total_ms / windows as f64
  );
  println!("  (pure-decode baseline ~3 ms/step; in-situ transcribe ~11 ms/step)");
  Ok(())
}

/// Phase-1 correctness: the lazy-input decode (`decode_token_lazy` on a `(1,1)`
/// token Array) must produce logits IDENTICAL to the host-slice decode
/// (`decode_tokens(&[tok])`) for the same token + cache state — they share
/// `run_from_array`, so the only difference is how the token array was built.
#[test]
#[ignore = "needs a local whisper checkpoint; run with --ignored --nocapture"]
fn lazy_decode_matches_host_decode() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP lazy_decode_matches_host_decode: {MODEL_DIR} not found");
    return Ok(());
  }
  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let (n_ctx, n_state) = (dims.n_audio_ctx(), dims.n_audio_state());
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;
  let fkey = ops::random::key(0)?;
  let features = ops::random::normal(
    &[1i32, n_ctx as i32, n_state as i32],
    Dtype::F16,
    0.0,
    1.0,
    &fkey,
  )?;

  // Prefill so both decodes share one cache state.
  let (_p, cache0) = model.decode_tokens(&[SOT], &features, None)?;
  let tok: u32 = 1000;

  let (mut logits_h, _) = model.decode_tokens(&[tok], &features, Some(&cache0))?;
  let tok_arr = crate::Array::from_slice::<u32>(&[tok], &[1, 1])?;
  let (mut logits_l, _) = model.decode_token_lazy(&tok_arr, &features, Some(&cache0))?;

  let mut am_h = ops::misc::argmax(&logits_h, Some(-1), false)?;
  let mut am_l = ops::misc::argmax(&logits_l, Some(-1), false)?;
  transforms::eval(&[&logits_h, &logits_l, &am_h, &am_l])?;
  assert_eq!(
    am_h.to_vec::<u32>()?,
    am_l.to_vec::<u32>()?,
    "lazy decode argmax must equal host decode argmax"
  );
  assert_eq!(
    logits_h.to_vec::<f32>()?,
    logits_l.to_vec::<f32>()?,
    "lazy decode logits must be bit-identical to host decode"
  );
  Ok(())
}

/// Phase-2 correctness: the device-op timestamp mask must EQUAL the host
/// `deterministic_mask` for every token-history shape. Pure logic on a small
/// synthetic vocab — no model, runs in CI.
#[test]
fn timestamp_mask_device_matches_host() -> Result<()> {
  let rules = ApplyTimestampRules {
    sample_begin: 3,
    timestamp_begin: 50,
    no_timestamps: 49,
    eot: 40,
    max_initial_timestamp_index: Some(10),
    n_vocab: 100,
  };
  let ts = rules.timestamp_begin;
  let prefix = [1u32, 2, 3]; // len == sample_begin (the sot prefix)
  let tails: &[&[u32]] = &[
    &[],           // step 0 (is_first)
    &[10],         // step 1, last = text
    &[55],         // step 1, last = timestamp
    &[55, 56],     // step 2, ts-ts
    &[10, 55],     // step 2, text-ts
    &[55, 10],     // step 2, ts-text
    &[55, 56, 10], // step 3, ts pair then text
    &[60, 10, 20], // step 3, early ts (monotonicity) then text
    &[10, 20, 30], // step 3, all text (no timestamp)
    &[50, 50],     // step 2, <|0.00|> pair (equal-allowed edge)
    &[56, 55],     // step 2, decreasing ts (impossible in valid decode; tests LAST != max)
  ];
  for tail in tails {
    let tokens: Vec<u32> = prefix.iter().chain(tail.iter()).copied().collect();
    let host = rules.deterministic_mask(&tokens);

    let seq = &tokens[rules.sample_begin..];
    let step = seq.len();
    let last_tok = *seq.last().unwrap_or(&0);
    let penult_tok = if seq.len() >= 2 {
      seq[seq.len() - 2]
    } else {
      0
    };
    // The host uses the MOST-RECENT timestamp (it overwrites each iteration),
    // not the max — match that.
    let last_ts = seq.iter().copied().rev().find(|&v| v >= ts).unwrap_or(0);

    let lt = crate::Array::from_slice::<i32>(&[last_tok as i32], &[1])?;
    let pt = crate::Array::from_slice::<i32>(&[penult_tok as i32], &[1])?;
    let lts = crate::Array::from_slice::<i32>(&[last_ts as i32], &[1])?;
    let mut dev = rules.deterministic_mask_device(&lt, &pt, &lts, step)?;
    transforms::eval(&[&dev])?;
    assert_eq!(
      host,
      dev.to_vec::<f32>()?,
      "device timestamp mask must equal host for tail={tail:?} (step={step})"
    );
  }
  Ok(())
}

/// Phase-3 correctness: `update_lazy` must apply the eot-stick and the
/// logprob-accumulation gate exactly as `update` — on device, lazily.
#[test]
fn update_lazy_eot_stick_and_gate() -> Result<()> {
  let eot = 2u32;
  // logits whose argmax is index 3.
  let logits = crate::Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 5.0, 0.0], &[5])?;
  let mut d = GreedyDecoder::new(0.0, eot, 0)?;

  // last != eot ⇒ next = argmax (3), not completed, logprob contributes (< 0).
  let last = crate::Array::from_slice::<u32>(&[1u32], &[1])?;
  let (mut next, mut completed, mut contrib) = d.update_lazy(&logits, &last)?;
  transforms::eval(&[&next, &completed, &contrib])?;
  assert_eq!(next.item::<u32>()?, 3);
  assert!(!completed.item::<bool>()?);
  assert!(
    contrib.item::<f32>()? < 0.0,
    "live logprob must be negative"
  );

  // last == eot ⇒ next sticks at eot, completed, contribution gated to 0.
  let last_eot = crate::Array::from_slice::<u32>(&[eot], &[1])?;
  let (mut n2, mut c2, mut ct2) = d.update_lazy(&logits, &last_eot)?;
  transforms::eval(&[&n2, &c2, &ct2])?;
  assert_eq!(n2.item::<u32>()?, eot);
  assert!(c2.item::<bool>()?);
  assert_eq!(
    ct2.item::<f32>()?,
    0.0,
    "post-eot logprob contribution must be gated to 0"
  );
  Ok(())
}

/// A/B: per-token decode cost — the CURRENT serialized path vs the #369
/// pipelined path, on the real model. Drives the decode primitives directly (no
/// tokenizer / filters) so it isolates the structural per-token GPU<->host
/// round-trip the fix removes: SERIAL blocks on `item::<u32>()` before the next
/// step; PIPELINED keeps the token on device, dispatches via `async_eval`, reads
/// only the PREVIOUS step's completion flag (one behind, overlapped), and defers
/// the token readback to a single post-loop `eval`. Reports ms/token + speedup.
#[test]
#[ignore = "needs a local whisper checkpoint; run with --ignored --nocapture"]
fn decode_pipeline_ab() -> Result<()> {
  let dir = Path::new(MODEL_DIR);
  if !dir.exists() {
    eprintln!("SKIP decode_pipeline_ab: {MODEL_DIR} not found");
    return Ok(());
  }
  let body = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config: serde_json::Value = serde_json::from_str(&body).expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let (n_ctx, n_state) = (dims.n_audio_ctx(), dims.n_audio_state());
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;
  let fkey = ops::random::key(0)?;
  let features = ops::random::normal(
    &[1i32, n_ctx as i32, n_state as i32],
    Dtype::F16,
    0.0,
    1.0,
    &fkey,
  )?;
  transforms::eval(&[&features])?;

  // SERIAL: decode_tokens(&[u32]) -> argmax -> blocking eval -> item readback.
  let serial_ms = {
    let mut cache = None;
    let mut tok = SOT;
    for _ in 0..WARMUP {
      let (l, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
      cache = Some(c);
      let row = last_position_row(&l)?;
      let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
      transforms::eval(&[&idx])?;
      tok = idx.item::<u32>()?;
    }
    let t = Instant::now();
    for _ in 0..STEPS {
      let (l, c) = model.decode_tokens(&[tok], &features, cache.as_ref())?;
      cache = Some(c);
      let row = last_position_row(&l)?;
      let mut idx = ops::misc::argmax(&row, Some(-1), false)?;
      transforms::eval(&[&idx])?;
      tok = idx.item::<u32>()?;
    }
    t.elapsed().as_secs_f64() * 1e3 / STEPS as f64
  };

  // PIPELINED: decode_token_lazy -> lazy argmax -> async_eval -> completed
  // one-behind -> deferred readback. The eot sentinel never matches, so all
  // STEPS run.
  let pipelined_ms = {
    let eot_arr = crate::Array::from_slice::<u32>(&[u32::MAX], &[1])?;
    let mut cache = None;
    let mut tok_1x1 = crate::Array::from_slice::<u32>(&[SOT], &[1, 1])?;
    let mut prev_completed: Option<Array> = None;
    // warmup
    for _ in 0..WARMUP {
      let (l, c) = model.decode_token_lazy(&tok_1x1, &features, cache.as_ref())?;
      cache = Some(c);
      let row = last_position_row(&l)?;
      let idx = ops::misc::argmax(&row, Some(-1), true)?;
      let completed = ops::comparison::equal(&idx, &eot_arr)?;
      crate::transforms::async_eval(&[&idx, &completed])?;
      if let Some(pc) = &mut prev_completed {
        let _ = pc.item::<bool>()?;
      }
      tok_1x1 = ops::shape::reshape(&idx, &[1i32, 1])?;
      prev_completed = Some(completed);
    }
    // timed
    let mut sampled: Vec<Array> = Vec::new();
    let t = Instant::now();
    for _ in 0..STEPS {
      let (l, c) = model.decode_token_lazy(&tok_1x1, &features, cache.as_ref())?;
      cache = Some(c);
      let row = last_position_row(&l)?;
      let idx = ops::misc::argmax(&row, Some(-1), true)?;
      let completed = ops::comparison::equal(&idx, &eot_arr)?;
      crate::transforms::async_eval(&[&idx, &completed])?;
      if let Some(pc) = &mut prev_completed
        && pc.item::<bool>()?
      {
        break;
      }
      sampled.push(idx.try_clone()?);
      tok_1x1 = ops::shape::reshape(&idx, &[1i32, 1])?;
      prev_completed = Some(completed);
    }
    let refs: Vec<&Array> = sampled.iter().collect();
    transforms::eval(&refs)?;
    t.elapsed().as_secs_f64() * 1e3 / STEPS as f64
  };

  // FULLY-LAZY diagnostic: build the whole STEPS-step decode as ONE lazy graph
  // (each token feeds the next lazily, no per-step readback at all), then a
  // SINGLE eval. This is the pipelining CEILING — if even this does not beat
  // serial, mlx-c does not overlap autoregressive decode command buffers (the
  // cache dependency serializes the GPU regardless of how we dispatch).
  let fully_lazy_ms = {
    let mut cache = None;
    let mut tok_1x1 = crate::Array::from_slice::<u32>(&[SOT], &[1, 1])?;
    for _ in 0..WARMUP {
      let (l, c) = model.decode_token_lazy(&tok_1x1, &features, cache.as_ref())?;
      cache = Some(c);
      let idx = ops::misc::argmax(&last_position_row(&l)?, Some(-1), true)?;
      tok_1x1 = ops::shape::reshape(&idx, &[1i32, 1])?;
      transforms::eval(&[&tok_1x1])?;
    }
    let t = Instant::now();
    let mut outs: Vec<Array> = Vec::new();
    for _ in 0..STEPS {
      let (l, c) = model.decode_token_lazy(&tok_1x1, &features, cache.as_ref())?;
      cache = Some(c);
      let idx = ops::misc::argmax(&last_position_row(&l)?, Some(-1), true)?;
      tok_1x1 = ops::shape::reshape(&idx, &[1i32, 1])?;
      outs.push(idx);
    }
    let refs: Vec<&Array> = outs.iter().collect();
    transforms::eval(&refs)?;
    t.elapsed().as_secs_f64() * 1e3 / STEPS as f64
  };

  println!("\nDECODE A/B (whisper-large-v3 f16, {STEPS} steps):");
  println!("  serial      : {serial_ms:.2} ms/token");
  println!(
    "  pipelined   : {pipelined_ms:.2} ms/token  ({:.2}x)",
    serial_ms / pipelined_ms
  );
  println!(
    "  fully-lazy  : {fully_lazy_ms:.2} ms/token  ({:.2}x)  [pipelining ceiling]",
    serial_ms / fully_lazy_ms
  );
  Ok(())
}

/// mlx-c pipelining investigation (#369 follow-up). Isolates the per-token
/// "bubble" from whisper: (1) the latency of a single trivial `eval` (is the
/// bubble a per-eval command-buffer cost?), and (2) whether `async_eval`
/// overlaps a DEPENDENT chain (autoregressive-like: x_{i+1} = x_i + c) at all —
/// serial (eval + read each step) vs pipelined (async_eval + read one-behind).
/// If the chain pipelines here but whisper decode does not, the decode path has
/// a specific sync; if it does not pipeline here either, mlx-c does not overlap
/// dependent command buffers.
#[test]
#[ignore = "timing microbenchmark — run with --ignored --nocapture"]
fn async_eval_pipelining_micro() -> Result<()> {
  use std::time::Instant;
  const N: usize = 300;
  let dim = 4096i32;
  let c = ops::random::normal(&[1i32, dim], Dtype::F32, 0.0, 1e-6, &ops::random::key(1)?)?;
  let x0 = ops::random::normal(&[1i32, dim], Dtype::F32, 0.0, 1.0, &ops::random::key(2)?)?;
  transforms::eval(&[&c, &x0])?;

  // (1) trivial per-eval latency (eval + scalar readback).
  let eval_us = {
    for _ in 0..20 {
      let mut y = ops::reduction::max(&ops::arithmetic::add(&x0, &c)?, false)?;
      transforms::eval(&[&y])?;
      let _ = y.item::<f32>()?;
    }
    let t = Instant::now();
    for _ in 0..N {
      let mut y = ops::reduction::max(&ops::arithmetic::add(&x0, &c)?, false)?;
      transforms::eval(&[&y])?;
      let _ = y.item::<f32>()?;
    }
    t.elapsed().as_secs_f64() * 1e6 / N as f64
  };

  // (2) dependent chain — serial (eval + read each step).
  let serial = {
    let mut x = x0.try_clone()?;
    for _ in 0..20 {
      x = ops::arithmetic::add(&x, &c)?;
      transforms::eval(&[&x])?;
    }
    let mut x = x0.try_clone()?;
    let t = Instant::now();
    for _ in 0..N {
      x = ops::arithmetic::add(&x, &c)?;
      let mut s = ops::reduction::max(&x, false)?;
      transforms::eval(&[&x])?;
      let _ = s.item::<f32>()?;
    }
    t.elapsed().as_secs_f64() * 1e6 / N as f64
  };

  // (3) dependent chain — pipelined (async_eval + read one-behind).
  let pipelined = {
    let mut x = x0.try_clone()?;
    for _ in 0..20 {
      x = ops::arithmetic::add(&x, &c)?;
      crate::transforms::async_eval(&[&x])?;
    }
    transforms::eval(&[&x])?;
    let mut x = x0.try_clone()?;
    let mut prev: Option<Array> = None;
    let t = Instant::now();
    for _ in 0..N {
      x = ops::arithmetic::add(&x, &c)?;
      let s = ops::reduction::max(&x, false)?;
      crate::transforms::async_eval(&[&x])?;
      if let Some(mut p) = prev.take() {
        let _ = p.item::<f32>()?;
      }
      prev = Some(s);
    }
    if let Some(mut p) = prev {
      let _ = p.item::<f32>()?;
    }
    t.elapsed().as_secs_f64() * 1e6 / N as f64
  };

  println!("\nmlx-c pipelining micro (dim={dim}, N={N}):");
  println!("  trivial eval+read : {eval_us:.1} us");
  println!("  chain serial      : {serial:.1} us/step");
  println!(
    "  chain pipelined   : {pipelined:.1} us/step  ({:.2}x)",
    serial / pipelined
  );
  Ok(())
}

/// End-to-end transcribe RTF on a real fixture — directly comparable to
/// mlx-audio's `Processing time / audio duration`. Runs the whisper-specific
/// `transcribe` (full pipeline: 30s windows, language detect, temperature
/// fallback, reference defaults) on `-asr-fp16` (weights + tokenizer); the audio
/// is pre-decoded to 16 kHz mono f32 PCM at `/private/tmp/fixture.f32`.
#[test]
#[ignore = "needs -asr-fp16 + the fixture PCM; run with --ignored --nocapture"]
fn fixture_transcribe_rtf() -> Result<()> {
  use std::{path::Path, time::Instant};
  let dir =
    Path::new("/Users/al/Developer/findit-studio/mlxrs/models/whisper-large-v3-turbo-asr-fp16");
  // The fixture PCM is overridable so the same harness can sweep multiple clips
  // (short vs long, English vs Chinese) to localize where the gap appears.
  let pcm_path_owned =
    std::env::var("MLXRS_FIXTURE_PCM").unwrap_or_else(|_| "/private/tmp/fixture.f32".to_string());
  let pcm_path = pcm_path_owned.as_str();
  if !dir.exists() || !Path::new(pcm_path).exists() {
    eprintln!("SKIP fixture_transcribe_rtf: model or fixture PCM missing");
    return Ok(());
  }

  let config: serde_json::Value =
    serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).expect("config.json"))
      .expect("parse config.json");
  let dims = ModelDimensions::from_dict(&config)?;
  let model = WhisperModel::load(dir, dims, Dtype::F16)?;
  let mdims = model.dims();
  // A/B: cap the MLX buffer cache (env MLXRS_CACHE_LIMIT_MB) to test whether the
  // unbounded ~6 GB cache is what slows the in-situ decode.
  if let Ok(mb) = std::env::var("MLXRS_CACHE_LIMIT_MB") {
    let bytes = mb
      .parse::<usize>()
      .unwrap_or(512)
      .saturating_mul(1024 * 1024);
    let mut prior = 0usize;
    // SAFETY: thin FFI to `mlx_set_cache_limit(size_t* res, size_t limit)`; the
    // out-param `prior` is written with the previous limit.
    let rc = unsafe { mlxrs_sys::mlx_set_cache_limit(&mut prior, bytes) };
    println!(
      "  [MLX cache limit set to {mb} MB, prior={} MB, rc={rc}]",
      prior / (1024 * 1024)
    );
  }
  let tok = Tokenizer::from_path(dir, None)?;
  let wrapper = HFTokenizerWrapper::new(
    &tok,
    mdims.is_multilingual(),
    mdims.num_languages(),
    None, // auto-detect language (matches mlx-audio's default)
    Task::Transcribe,
  )?;

  let bytes = std::fs::read(pcm_path).expect("read fixture.f32");
  let pcm: Vec<f32> = bytes
    .chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();
  let audio = Array::from_slice::<f32>(&pcm, &[pcm.len() as i32])?;
  let audio_secs = pcm.len() as f64 / 16000.0;

  // A/B: disable the temperature fallback (single temp 0) to isolate its cost
  // from the per-token decode. Toggle via env MLXRS_RTF_NOFALLBACK=1.
  let mut options = TranscribeOptions::default();
  let no_fallback = std::env::var("MLXRS_RTF_NOFALLBACK").is_ok();
  if no_fallback {
    options.temperatures = vec![0.0];
  }
  // mel frontend (one STFT over the whole clip), timed separately.
  let t0 = Instant::now();
  let mel = model.log_mel(&audio)?;
  transforms::eval(&[&mel])?;
  let mel_secs = t0.elapsed().as_secs_f64();
  let content_frames = mel.shape()[0];

  // one encoder forward on a single 30s window (1500-conv-frame ≈ 3000 mel frames)
  // to estimate the per-window encoder cost. Clips shorter than one 30s window
  // are zero-padded to 3000 frames inside transcribe but cannot warm the encoder
  // on a real window here, so their encoder est is reported as 0 (the transcribe
  // RTF below is still measured end to end).
  let n_mels = mel.shape()[1] as i32;
  let n_windows = content_frames.div_ceil(3000);
  let enc_one = if content_frames >= 3000 {
    let window = ops::indexing::slice(&mel, &[0, 0], &[3000, n_mels], &[1, 1])?;
    // WARM the encoder (JIT) so the per-window time is steady-state, matching the
    // windows inside transcribe (the first encode pays one-time kernel compile).
    for _ in 0..3 {
      let w = model.encode(&window)?;
      transforms::eval(&[&w])?;
    }
    let t1 = Instant::now();
    for _ in 0..5 {
      let e = model.encode(&window)?;
      transforms::eval(&[&e])?;
    }
    t1.elapsed().as_secs_f64() / 5.0
  } else {
    0.0
  };

  // full transcribe (encoder + decode loop + pipeline).
  let t2 = Instant::now();
  let result = transcribe(&model, &wrapper, &mel, content_frames, &options)?;
  let tr_secs = t2.elapsed().as_secs_f64();
  let total = mel_secs + tr_secs;
  let enc_est = enc_one * n_windows as f64;

  println!(
    "\nMLXRS (turbo f16, serial) on {audio_secs:.1}s audio (lang={}, {} chars):",
    result.language,
    result.text.chars().count()
  );
  println!("  mel(STFT)        = {mel_secs:.2}s");
  println!(
    "  transcribe(e+d)  = {tr_secs:.2}s   [~{enc_est:.1}s encoder est ({n_windows} win x {enc_one:.3}s), rest decode+pipeline]"
  );
  println!(
    "  TOTAL            = {total:.2}s   ->  RTF = {:.4}",
    total / audio_secs
  );
  {
    let act = crate::memory::active_memory().unwrap_or(0) / (1024 * 1024);
    let peak = crate::memory::peak_memory().unwrap_or(0) / (1024 * 1024);
    let cac = crate::memory::cache_memory().unwrap_or(0) / (1024 * 1024);
    println!("  MEMORY (post-transcribe): active={act}MB peak={peak}MB cache={cac}MB");
  }
  if std::env::var("MLXRS_TIMING2").is_ok() {
    use std::sync::atomic::Ordering::Relaxed;
    let enc = super::TIMING2_ENC_NS.load(Relaxed) as f64 / 1e9;
    let dec = super::TIMING2_DEC_NS.load(Relaxed) as f64 / 1e9;
    let steps = super::TIMING2_STEPS.load(Relaxed);
    let calls = super::TIMING2_CALLS.load(Relaxed);
    let per = if steps > 0 {
      dec * 1e3 / steps as f64
    } else {
      0.0
    };
    let warm = super::TIMING2_WARM_NS.load(Relaxed) as f64 / 1e9;
    let warm_steps = super::TIMING2_WARM_STEPS.load(Relaxed);
    let warm_per = if warm_steps > 0 {
      warm * 1e3 / warm_steps as f64
    } else {
      0.0
    };
    println!(
      "  TIMING2: encoder(eval)={enc:.1}s  decode(main_loop)={dec:.1}s   [{calls} run() calls, {steps} token-len = {per:.2} ms]"
    );
    println!(
      "  TIMING2: WARM loop only = {warm:.1}s over {warm_steps} warm steps = {warm_per:.2} ms/warm-step   (prefill = {:.1}s)",
      dec - warm
    );
  }
  Ok(())
}
