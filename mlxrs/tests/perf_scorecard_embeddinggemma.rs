//! Performance scorecard bench — EmbeddingGemma forward pass.
//!
//! Isolated forward-pass timing for the `google/embeddinggemma-300m` sentence
//! encoder (`encode_text`: bidirectional Gemma3 backbone → mean-pool → Dense →
//! normalize). Mirrors the back-to-back Python reference
//! (`mlx_embeddings` `gemma3_text.Model.__call__`) in
//! `docs/perf/py/embeddinggemma_bench.py`.
//!
//! `#[ignore]` + skip-when-absent: needs the gitignored checkpoint at
//! `MLXRS_EMBEDDINGGEMMA_MODEL_DIR` (or the HF-cache default). Run with:
//! `cargo test --release --features embeddinggemma --test
//! perf_scorecard_embeddinggemma -- --ignored --nocapture`.
#![cfg(feature = "embeddinggemma")]

use std::{
  path::{Path, PathBuf},
  time::Instant,
};

use mlxrs::{
  array::Array,
  embeddings::embeddinggemma::{EmbeddingGemmaModel, config::Gemma3Config, sanitize},
};

/// Sequence length of the fabricated forward input (a realistic encoder seq).
const SEQ_LEN: usize = 256;
/// Warm-up forwards (discarded — JIT / kernel-compile out of the measurement).
const WARMUP: usize = 6;
/// Timed iterations.
const ITERS: usize = 30;

fn model_dir() -> Option<PathBuf> {
  if let Ok(dir) = std::env::var("MLXRS_EMBEDDINGGEMMA_MODEL_DIR") {
    let p = PathBuf::from(dir);
    return p.is_dir().then_some(p);
  }
  // HF cache for `mlx-community/embeddinggemma-300m-bf16`.
  let home = std::env::var("HOME").ok()?;
  let base = PathBuf::from(home)
    .join(".cache/huggingface/hub/models--mlx-community--embeddinggemma-300m-bf16/snapshots");
  let snap = std::fs::read_dir(&base).ok()?.next()?.ok()?.path();
  snap.is_dir().then_some(snap)
}

/// config-parse → load every `*.safetensors` shard → sanitize → from_weights
/// (the canonical load path; pooling `None` ⇒ EmbeddingGemma's mean default).
fn load_model(dir: &Path) -> EmbeddingGemmaModel {
  let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
  let config = Gemma3Config::from_json(&config_json).expect("parse config.json");
  let mut raw = std::collections::HashMap::new();
  for entry in std::fs::read_dir(dir).expect("read model dir") {
    let path = entry.expect("dir entry").path();
    if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
      let part = mlxrs::io::load_safetensors(&path).expect("load safetensors shard");
      raw.extend(part);
    }
  }
  assert!(!raw.is_empty(), "no safetensors in {}", dir.display());
  let weights = sanitize(raw).expect("sanitize weights");
  EmbeddingGemmaModel::from_weights(config, weights, None).expect("build model")
}

fn percentiles(mut times: Vec<f64>) -> (f64, f64) {
  times.sort_by(|a, b| a.partial_cmp(b).unwrap());
  (times[0], times[times.len() / 2])
}

#[test]
#[ignore = "perf scorecard bench — run with --ignored --nocapture (needs the embeddinggemma-300m checkpoint)"]
fn bench_embeddinggemma_forward() {
  let Some(dir) = model_dir() else {
    eprintln!(
      "SKIP bench_embeddinggemma_forward: checkpoint absent \
       (set MLXRS_EMBEDDINGGEMMA_MODEL_DIR)"
    );
    return;
  };
  let model = load_model(&dir);

  // Fabricated rank-2 ids + all-ones {0,1} mask at the real shape (1, SEQ_LEN).
  let ids_v: Vec<i32> = (0..SEQ_LEN as i32).map(|i| (i % 1000) + 5).collect();
  let input_ids = Array::from_slice::<i32>(&ids_v, &(1usize, SEQ_LEN)).expect("ids");
  let mask_v: Vec<i32> = vec![1; SEQ_LEN];
  let attention_mask = Array::from_slice::<i32>(&mask_v, &(1usize, SEQ_LEN)).expect("mask");

  for _ in 0..WARMUP {
    let mut out = model
      .encode_text(&input_ids, &attention_mask)
      .expect("encode");
    out.eval().expect("eval");
  }
  let mut times = Vec::with_capacity(ITERS);
  for _ in 0..ITERS {
    let t0 = Instant::now();
    let mut out = model
      .encode_text(&input_ids, &attention_mask)
      .expect("encode");
    out.eval().expect("eval");
    times.push(t0.elapsed().as_secs_f64() * 1e3);
  }
  let (min, med) = percentiles(times);
  println!(
    "\nMLXRS embeddinggemma-300m encode_text forward (1x{SEQ_LEN}): min={min:.3}ms median={med:.3}ms  (bf16)"
  );
}
