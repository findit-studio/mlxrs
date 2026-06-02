//! Structural tests for the SigLIP2 NaFlex text tower.
//!
//! Deterministic, tiny-fixture, non-gated: a `hidden = 8`, `heads = 2`,
//! `layers = 1`, `vocab = 10`, `max_pos = 6`, `projection = 8` text tower is
//! built from synthetic weights and exercised through [`TextTower`]. These pin
//! the pooled output shape, the seq-length guard, and a couple of
//! consumed-weight shape-mismatch rejections.

use std::collections::HashMap;

use super::*;
use crate::embeddings::siglip2_naflex::config::TextConfig;

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const LAYERS: i32 = 1;
const INTER: i32 = 16;
const VOCAB: i32 = 10;
const MAX_POS: i32 = 6;
const PROJ: i32 = 8;

fn tiny_text_config() -> TextConfig {
  let json = format!(
    r#"{{
      "model_type": "siglip2_text_model",
      "vocab_size": {VOCAB},
      "max_position_embeddings": {MAX_POS},
      "hidden_size": {HIDDEN},
      "intermediate_size": {INTER},
      "num_attention_heads": {HEADS},
      "num_hidden_layers": {LAYERS},
      "layer_norm_eps": 1e-6
    }}"#
  );
  let cfg = TextConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| ((i % 5) as f32) * 0.01).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

fn insert_attn(w: &mut HashMap<String, Array>, prefix: &str) {
  for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(format!("{prefix}.{p}.weight"), mat(HIDDEN, HIDDEN));
    w.insert(format!("{prefix}.{p}.bias"), vec1(HIDDEN));
  }
}

fn insert_mlp(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.fc1.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{prefix}.fc1.bias"), vec1(INTER));
  w.insert(format!("{prefix}.fc2.weight"), mat(HIDDEN, INTER));
  w.insert(format!("{prefix}.fc2.bias"), vec1(HIDDEN));
}

fn insert_ln(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.weight"), vec1(HIDDEN));
  w.insert(format!("{prefix}.bias"), vec1(HIDDEN));
}

/// Build a full, correctly-shaped weight map for the tiny text tower (the
/// post-strip keys `TextTower::from_weights` consumes).
fn tiny_text_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(
    "embeddings.token_embedding.weight".to_string(),
    mat(VOCAB, HIDDEN),
  );
  w.insert(
    "embeddings.position_embedding.weight".to_string(),
    mat(MAX_POS, HIDDEN),
  );
  insert_ln(&mut w, "encoder.layers.0.layer_norm1");
  insert_attn(&mut w, "encoder.layers.0.self_attn");
  insert_ln(&mut w, "encoder.layers.0.layer_norm2");
  insert_mlp(&mut w, "encoder.layers.0.mlp");
  insert_ln(&mut w, "final_layer_norm");
  // The text head is a BIASED Linear(hidden, projection_size).
  w.insert("head.weight".to_string(), mat(PROJ, HIDDEN));
  w.insert("head.bias".to_string(), vec1(PROJ));
  w
}

/// A `(batch, seq_len)` i32 token-id batch with ids in `0..VOCAB`.
fn ids(batch: usize, seq: usize) -> Array {
  let data: Vec<i32> = (0..batch * seq)
    .map(|n| (n % VOCAB as usize) as i32)
    .collect();
  Array::from_slice::<i32>(&data, &(batch, seq)).unwrap()
}

fn eval_to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

#[test]
fn text_tower_pooled_output_shape() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  let tower = TextTower::from_weights(&cfg, &mut w).unwrap();
  // batch 3, seq 4 (< max_pos 6).
  let pooled = tower.forward(&ids(3, 4)).unwrap();
  // (batch, projection_size).
  assert_eq!(pooled.shape(), vec![3, PROJ as usize]);
  assert!(eval_to_vec(&pooled).iter().all(|x| x.is_finite()));
}

#[test]
fn text_tower_pooled_output_shape_full_length() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  let tower = TextTower::from_weights(&cfg, &mut w).unwrap();
  // seq == max_pos is allowed.
  let pooled = tower.forward(&ids(1, MAX_POS as usize)).unwrap();
  assert_eq!(pooled.shape(), vec![1, PROJ as usize]);
}

#[test]
fn text_tower_rejects_seq_over_max_position() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  let tower = TextTower::from_weights(&cfg, &mut w).unwrap();
  // seq = max_pos + 1 must be rejected.
  let err = tower.forward(&ids(1, MAX_POS as usize + 1));
  assert!(err.is_err(), "seq_len > max_position_embeddings must error");
}

#[test]
fn text_tower_rejects_empty_seq() {
  // A `(batch, 0)` input has an empty sequence axis: there is no last token to
  // pool. The tower must reject `seq_len == 0` with a typed `OutOfRange`
  // BEFORE any embedding lookup or last-token pooling — never fall through to
  // `index_last(0)` building `[-1]` and a backend / negative-index `take_axis`.
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  let tower = TextTower::from_weights(&cfg, &mut w).unwrap();
  let empty = Array::from_slice::<i32>(&[], &(2usize, 0usize)).unwrap();
  let err = tower.forward(&empty).err();
  assert!(
    matches!(err, Some(Error::OutOfRange(_))),
    "empty (batch, 0) sequence must be a typed OutOfRange, got {err:?}"
  );
}

#[test]
fn text_tower_rejects_non_rank2_input_ids() {
  // The public `encode_text` / `embed_text` accept an untrusted array. A rank-3
  // `input_ids` must be rejected with a typed `RankMismatch` BEFORE any op — a
  // `shape[1]`-only read would otherwise gather a different-rank graph. Pins the
  // same exact-shape discipline as the vision tower's runtime gate.
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  let tower = TextTower::from_weights(&cfg, &mut w).unwrap();
  // (1, 2, 3) i32 — a rank-3 input.
  let bad = Array::from_slice::<i32>(&[0, 1, 2, 3, 4, 5], &(1usize, 2usize, 3usize)).unwrap();
  let err = tower.forward(&bad).err();
  assert!(
    matches!(err, Some(Error::RankMismatch(_))),
    "rank-3 input_ids must be a typed RankMismatch, got {err:?}"
  );
}

#[test]
fn from_weights_rejects_wrong_token_table_shape() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  w.insert(
    "embeddings.token_embedding.weight".to_string(),
    mat(VOCAB + 1, HIDDEN),
  );
  let err = TextTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "wrong vocab dim must be rejected");
}

#[test]
fn from_weights_rejects_wrong_head_shape() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  // Wrong projection output dim.
  w.insert("head.weight".to_string(), mat(PROJ + 1, HIDDEN));
  let err = TextTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "wrong head projection dim must be rejected");
}

#[test]
fn from_weights_rejects_oversize_num_hidden_layers() {
  // A directly-built (unvalidated) config with a hostile `num_hidden_layers`
  // reaching the PUBLIC `from_weights` must be rejected with a typed
  // `CapExceeded` (its idempotent `config.validate()?` guard) BEFORE the
  // per-layer reservation/loop — never a `with_capacity` abort.
  let json = format!(
    r#"{{
      "model_type": "siglip2_text_model",
      "vocab_size": {VOCAB},
      "max_position_embeddings": {MAX_POS},
      "hidden_size": {HIDDEN},
      "intermediate_size": {INTER},
      "num_attention_heads": {HEADS},
      "num_hidden_layers": 1000000,
      "layer_norm_eps": 1e-6
    }}"#
  );
  let cfg = TextConfig::from_json(&json).unwrap();
  assert_eq!(cfg.num_hidden_layers, 1_000_000);
  let mut w = tiny_text_weights();
  let err = TextTower::from_weights(&cfg, &mut w).err();
  assert!(
    matches!(err, Some(Error::CapExceeded(_))),
    "oversize num_hidden_layers must be a typed CapExceeded, got {err:?}"
  );
}

#[test]
fn from_weights_rejects_missing_final_layer_norm() {
  let cfg = tiny_text_config();
  let mut w = tiny_text_weights();
  w.remove("final_layer_norm.weight");
  let err = TextTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "missing final LayerNorm must be rejected");
}
