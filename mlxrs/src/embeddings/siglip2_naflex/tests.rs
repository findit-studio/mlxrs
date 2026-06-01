//! Structural tests for the SigLIP2 NaFlex dual-tower model + sanitize +
//! factory registration.
//!
//! Deterministic, tiny-fixture, non-gated. A tiny dual-tower model (`hidden =
//! 8`, `heads = 2`, `layers = 1`, `patch = 2`, `num_patches = 4`, `vocab =
//! 10`, `max_pos = 6`, `projection = 8`) is assembled from synthetic weights
//! (the *pre-strip* `vision_model.vision_model.*` / `text_model.text_model.*`
//! key layout `from_weights` consumes) and exercised end-to-end:
//! `encode_image` / `encode_text` shapes + L2-normalization, the `logits`
//! shape, the `sanitize` namespacing + `in_proj` rename, the
//! classifier-head rejection, and the registry registration.
//!
//! Numeric oracle parity (vs the PyTorch reference `.npy`) is the gated e2e
//! test in the crate-level `tests/`.

use std::collections::HashMap;

use super::*;
use crate::embeddings::{
  EmbeddingModelTypeRegistry, LoadedEmbeddingModel, TextEmbedder,
  siglip2_naflex::processing::preprocess,
};

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const LAYERS: i32 = 1;
const PATCH: i32 = 2;
const CHANNELS: i32 = 3;
const INTER: i32 = 16;
const NUM_PATCHES: i32 = 4;
const PATCH_FEAT: i32 = PATCH * PATCH * CHANNELS; // 12
const VOCAB: i32 = 10;
const MAX_POS: i32 = 6;
const PROJ: i32 = 8;

/// A tiny dual-tower `config.json` body (only the fields the port reads).
fn tiny_config_json() -> String {
  format!(
    r#"{{
      "model_type": "siglip2",
      "num_labels": 0,
      "text_config": {{
        "model_type": "siglip2_text_model",
        "vocab_size": {VOCAB},
        "max_position_embeddings": {MAX_POS},
        "hidden_size": {HIDDEN},
        "intermediate_size": {INTER},
        "num_attention_heads": {HEADS},
        "num_hidden_layers": {LAYERS},
        "layer_norm_eps": 1e-6
      }},
      "vision_config": {{
        "model_type": "siglip2_vision_model",
        "image_size": 4,
        "patch_size": {PATCH},
        "num_channels": {CHANNELS},
        "hidden_size": {HIDDEN},
        "intermediate_size": {INTER},
        "num_attention_heads": {HEADS},
        "num_hidden_layers": {LAYERS},
        "layer_norm_eps": 1e-6,
        "vision_use_head": true,
        "num_patches": {NUM_PATCHES},
        "max_num_patches": {NUM_PATCHES}
      }}
    }}"#
  )
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

/// Build the full, correctly-shaped *post-sanitize* dual-tower weight map (the
/// `vision_model.vision_model.*` / `text_model.text_model.*` layout
/// `from_weights` consumes).
fn tiny_model_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();

  // ── vision tower (vision_model.vision_model.*) ──
  let vp = "vision_model.vision_model";
  w.insert(
    format!("{vp}.embeddings.patch_embedding.weight"),
    mat(HIDDEN, PATCH_FEAT),
  );
  w.insert(
    format!("{vp}.embeddings.patch_embedding.bias"),
    vec1(HIDDEN),
  );
  w.insert(
    format!("{vp}.embeddings.position_embedding.weight"),
    mat(NUM_PATCHES, HIDDEN),
  );
  insert_ln(&mut w, &format!("{vp}.encoder.layers.0.layer_norm1"));
  insert_attn(&mut w, &format!("{vp}.encoder.layers.0.self_attn"));
  insert_ln(&mut w, &format!("{vp}.encoder.layers.0.layer_norm2"));
  insert_mlp(&mut w, &format!("{vp}.encoder.layers.0.mlp"));
  insert_ln(&mut w, &format!("{vp}.post_layernorm"));
  w.insert(
    format!("{vp}.head.probe"),
    Array::from_slice::<f32>(
      &(0..HIDDEN as usize)
        .map(|i| (i as f32) * 0.01 + 0.01)
        .collect::<Vec<_>>(),
      &(1usize, 1usize, HIDDEN as usize),
    )
    .unwrap(),
  );
  w.insert(
    format!("{vp}.head.attention.in_proj.weight"),
    mat(3 * HIDDEN, HIDDEN),
  );
  w.insert(
    format!("{vp}.head.attention.in_proj.bias"),
    vec1(3 * HIDDEN),
  );
  w.insert(
    format!("{vp}.head.attention.out_proj.weight"),
    mat(HIDDEN, HIDDEN),
  );
  w.insert(format!("{vp}.head.attention.out_proj.bias"), vec1(HIDDEN));
  insert_ln(&mut w, &format!("{vp}.head.layernorm"));
  insert_mlp(&mut w, &format!("{vp}.head.mlp"));

  // ── text tower (text_model.text_model.*) ──
  let tp = "text_model.text_model";
  w.insert(
    format!("{tp}.embeddings.token_embedding.weight"),
    mat(VOCAB, HIDDEN),
  );
  w.insert(
    format!("{tp}.embeddings.position_embedding.weight"),
    mat(MAX_POS, HIDDEN),
  );
  insert_ln(&mut w, &format!("{tp}.encoder.layers.0.layer_norm1"));
  insert_attn(&mut w, &format!("{tp}.encoder.layers.0.self_attn"));
  insert_ln(&mut w, &format!("{tp}.encoder.layers.0.layer_norm2"));
  insert_mlp(&mut w, &format!("{tp}.encoder.layers.0.mlp"));
  insert_ln(&mut w, &format!("{tp}.final_layer_norm"));
  w.insert(format!("{tp}.head.weight"), mat(PROJ, HIDDEN));
  w.insert(format!("{tp}.head.bias"), vec1(PROJ));

  // ── top-level contrastive params ──
  w.insert(
    "logit_scale".to_string(),
    Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap(),
  );
  w.insert(
    "logit_bias".to_string(),
    Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap(),
  );
  w
}

fn tiny_model() -> Siglip2NaflexModel {
  let cfg = Siglip2NaflexConfig::from_json(&tiny_config_json()).unwrap();
  Siglip2NaflexModel::from_weights(cfg, tiny_model_weights()).unwrap()
}

fn tiny_image_inputs() -> NaflexInputs {
  let rgb = vec![100u8; (4 * 4 * 3) as usize];
  preprocess(
    &rgb,
    4,
    4,
    PATCH as u32,
    CHANNELS as u32,
    NUM_PATCHES as u32,
  )
  .unwrap()
}

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

/// L2 norm of each row of a `(rows, cols)` flat buffer.
fn row_norms(flat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
  (0..rows)
    .map(|r| {
      flat[r * cols..(r + 1) * cols]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
    })
    .collect()
}

#[test]
fn from_weights_builds_dual_tower() {
  let _model = tiny_model();
}

#[test]
fn encode_image_shape_and_l2_normalized() {
  let model = tiny_model();
  let img = model.encode_image(&tiny_image_inputs()).unwrap();
  assert_eq!(img.shape(), vec![1, HIDDEN as usize]);
  let flat = eval_to_vec(&img);
  let norms = row_norms(&flat, 1, HIDDEN as usize);
  assert!(
    (norms[0] - 1.0).abs() < 1e-4,
    "image embedding must be L2-normalized: norm = {}",
    norms[0]
  );
}

#[test]
fn encode_text_shape_and_l2_normalized() {
  let model = tiny_model();
  let txt = model.encode_text(&ids(3, 4)).unwrap();
  assert_eq!(txt.shape(), vec![3, PROJ as usize]);
  let flat = eval_to_vec(&txt);
  let norms = row_norms(&flat, 3, PROJ as usize);
  for (i, &n) in norms.iter().enumerate() {
    assert!(
      (n - 1.0).abs() < 1e-4,
      "text embedding row {i} must be L2-normalized: norm = {n}"
    );
  }
}

#[test]
fn logits_shape_text_by_image() {
  let model = tiny_model();
  let img = model.encode_image(&tiny_image_inputs()).unwrap(); // (1, dim)
  let txt = model.encode_text(&ids(3, 4)).unwrap(); // (3, dim)
  let logits = model.logits(&txt, &img).unwrap();
  // (n_text, n_image) = (3, 1).
  assert_eq!(logits.shape(), vec![3, 1]);
  assert!(eval_to_vec(&logits).iter().all(|x| x.is_finite()));
}

#[test]
fn logits_apply_scale_and_bias() {
  // With logit_scale = 0 (exp = 1) and logit_bias = 0, the logits are the raw
  // cosine similarities of the (already-normalized) embeddings, i.e. in
  // [-1, 1]. This pins the `* exp(scale) + bias` tail's identity behaviour.
  let model = tiny_model();
  let img = model.encode_image(&tiny_image_inputs()).unwrap();
  let txt = model.encode_text(&ids(2, 4)).unwrap();
  let logits = model.logits(&txt, &img).unwrap();
  for &v in eval_to_vec(&logits).iter() {
    assert!(
      (-1.0001..=1.0001).contains(&v),
      "cosine logit out of [-1, 1]: {v}"
    );
  }
}

#[test]
fn text_embedder_runs_text_tower() {
  // The model is its own universal `TextEmbedder` (via `Embed<TextInput>`): the
  // text-tower path yields the `(batch, projection)` L2-normalized text
  // embedding (== the model's `encode_text`). The optional padding mask is
  // unused by the sticky-EOS pooling, so it is `None` here.
  let model = tiny_model();
  let emb = model.embed_text(&ids(1, 4), None).unwrap();
  assert_eq!(emb.array().shape(), vec![1, PROJ as usize]);
  // The returned text embedding is L2-normalized (the SigLIP text feature).
  let flat = eval_to_vec(emb.array());
  let norms = row_norms(&flat, 1, PROJ as usize);
  assert!(
    (norms[0] - 1.0).abs() < 1e-4,
    "text embedding must be L2-normalized: norm = {}",
    norms[0]
  );
}

#[test]
fn from_weights_rejects_classifier_head_config() {
  // num_labels > 0 selects the (unported) classifier-head arm.
  let json = tiny_config_json().replace("\"num_labels\": 0", "\"num_labels\": 5");
  let cfg = Siglip2NaflexConfig::from_json(&json).unwrap();
  let err = Siglip2NaflexModel::from_weights(cfg, tiny_model_weights());
  assert!(err.is_err(), "num_labels > 0 must be rejected");
}

#[test]
fn from_weights_rejects_missing_logit_scale() {
  let cfg = Siglip2NaflexConfig::from_json(&tiny_config_json()).unwrap();
  let mut w = tiny_model_weights();
  w.remove("logit_scale");
  let err = Siglip2NaflexModel::from_weights(cfg, w);
  assert!(err.is_err(), "missing logit_scale must be rejected");
}

#[test]
fn sanitize_namespaces_towers_and_renames_in_proj() {
  // Raw HF-style keys: text_model.* / vision_model.* one level shallow, and
  // the MultiheadAttention combined-QKV `in_proj_weight` / `in_proj_bias`.
  let mut raw = HashMap::new();
  raw.insert(
    "text_model.embeddings.token_embedding.weight".to_string(),
    mat(VOCAB, HIDDEN),
  );
  raw.insert(
    "vision_model.head.attention.in_proj_weight".to_string(),
    mat(3 * HIDDEN, HIDDEN),
  );
  raw.insert(
    "vision_model.head.attention.in_proj_bias".to_string(),
    vec1(3 * HIDDEN),
  );
  // A position_ids buffer that must be dropped.
  raw.insert(
    "text_model.embeddings.position_ids".to_string(),
    Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3usize)).unwrap(),
  );
  let out = sanitize(raw).unwrap();
  assert!(
    out.contains_key("text_model.text_model.embeddings.token_embedding.weight"),
    "text tower must be namespaced one level deeper"
  );
  assert!(
    out.contains_key("vision_model.vision_model.head.attention.in_proj.weight"),
    "in_proj_weight must be renamed to in_proj.weight (and tower-namespaced)"
  );
  assert!(
    out.contains_key("vision_model.vision_model.head.attention.in_proj.bias"),
    "in_proj_bias must be renamed to in_proj.bias"
  );
  assert!(
    !out.keys().any(|k| k.contains("position_ids")),
    "position_ids buffers must be dropped"
  );
}

#[test]
fn sanitize_is_idempotent_for_already_nested_keys() {
  // An already-nested key must not be namespaced twice.
  let mut raw = HashMap::new();
  raw.insert(
    "text_model.text_model.final_layer_norm.weight".to_string(),
    vec1(HIDDEN),
  );
  let out = sanitize(raw).unwrap();
  assert!(out.contains_key("text_model.text_model.final_layer_norm.weight"));
  assert!(
    !out
      .keys()
      .any(|k| k.contains("text_model.text_model.text_model"))
  );
}

#[test]
fn register_adds_siglip_constructor() {
  let mut registry = EmbeddingModelTypeRegistry::new();
  assert!(!registry.contains(MODEL_TYPE));
  let prev = register(&mut registry);
  assert!(prev.is_none(), "first registration displaces nothing");
  assert!(registry.contains(MODEL_TYPE), "siglip must be registered");
}

#[test]
fn with_builtin_models_registers_siglip() {
  let registry = EmbeddingModelTypeRegistry::new().with_builtin_models();
  assert!(
    registry.contains("siglip2"),
    "with_builtin_models must register the siglip2-naflex model"
  );
}

#[test]
fn constructor_builds_model_from_loaded() {
  // The registry constructor builds the model from a LoadedEmbeddingModel
  // carrying the RAW (un-sanitized, HF-namespaced) weights + config JSON.
  // Build a raw map by un-namespacing the post-sanitize fixture one level.
  let post = tiny_model_weights();
  let mut raw = HashMap::with_capacity(post.len());
  for (k, v) in post {
    let raw_key = if let Some(rest) = k.strip_prefix("vision_model.vision_model.") {
      format!("vision_model.{rest}")
    } else if let Some(rest) = k.strip_prefix("text_model.text_model.") {
      format!("text_model.{rest}")
    } else {
      k
    };
    raw.insert(raw_key, v);
  }
  let loaded = LoadedEmbeddingModel::new("siglip2".to_string(), tiny_config_json(), raw);
  let ctor = constructor();
  let model = ctor(&loaded).expect("constructor must build the model");
  // The constructed model answers the umbrella's universal text capability; its
  // `embed_text` runs the text tower to a `(batch, projection)` embedding.
  let emb = model
    .as_text_embedder()
    .expect("siglip exposes the universal text embedder")
    .embed_text(&ids(1, 4), None)
    .unwrap();
  assert_eq!(emb.array().shape(), vec![1, PROJ as usize]);
}
