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
use crate::{
  embeddings::{
    EmbeddingModelTypeRegistry, LoadedEmbeddingModel, Padding, TextEmbedder,
    siglip2_naflex::processing::preprocess,
  },
  lm::quant::{PerLayerQuantization, Quantization},
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

/// An all-`1` `(batch, seq)` `f32` attention mask — the
/// [`Padding::FixedLength`] scheme SigLIP declares. SigLIP's `embed_text`
/// ignores the mask (sticky-EOS reads a fixed position), so any valid
/// `(batch, seq)` mask is accepted; this matches what `encode` would build.
fn mask(batch: usize, seq: usize) -> Array {
  Array::from_slice::<f32>(&vec![1.0_f32; batch * seq], &(batch, seq)).unwrap()
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
  // The model is its own universal `TextEmbedder`: the text-tower path yields
  // the `(batch, projection)` L2-normalized text embedding (== the model's
  // `encode_text`). The padding mask is unused by the sticky-EOS pooling, so an
  // all-`1` mask (the FixedLength scheme) is supplied and ignored.
  let model = tiny_model();
  let emb = model.embed_text(&ids(1, 4), &mask(1, 4)).unwrap();
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
fn text_encoding_declares_fixed_length_sticky_eos_scheme() {
  // SigLIP's `TextEncoding` must declare the fixed-length processor scheme the
  // sticky-EOS tower needs: special tokens on, and `FixedLength` padding to the
  // text tower's `max_position_embeddings` with the SigLIP2 Gemma `<pad>` id
  // `0` (NOT SigLIP1's pad == EOS == 1 — the Gemma tokenizer binds `<pad>` = 0,
  // `<eos>` = 1, so padding with 1 would fill the pooled last slot of every
  // short prompt with `<eos>` instead of `<pad>`), an all-`1`-mask scheme, and
  // `eos_token_id = Some(1)` so an overlength prompt keeps the EOS at its final
  // position on truncation (sticky-EOS pooling). The tokenizer truncation cap
  // is NOT carried in `max_length` here: the `FixedLength` scheme is itself the
  // truncation cap, and the generic pipeline derives the effective cap
  // (`length + 1` for this sticky-EOS scheme) CENTRALLY from the padding mode —
  // so the cap does not depend on this model remembering to set `max_length`.
  // The fixed length is read from the config (`MAX_POS` here), NOT hard-coded —
  // so it tracks the checkpoint.
  let model = tiny_model();
  let enc = model.text_encoding();
  assert!(enc.add_special_tokens, "siglip encodes with special tokens");
  assert_eq!(
    enc.max_length, None,
    "the tokenizer cap is derived centrally from the FixedLength scheme, not \
     carried in an optional per-model max_length field"
  );
  assert_eq!(
    enc.padding,
    Padding::FixedLength {
      length: MAX_POS as usize,
      pad_token_id: 0,
      eos_token_id: Some(1),
    },
    "siglip2 pads/truncates to max_position_embeddings with the Gemma <pad> id 0, \
     all-1 mask, EOS-preserving truncation (EOS = 1 is distinct from the pad fill)"
  );
}

#[test]
fn text_pad_token_id_defaults_to_gemma_pad_zero_and_is_overridable() {
  // A directly-built model (`from_weights`, no tokenizer metadata in reach)
  // defaults to the SigLIP2 Gemma `<pad>` id 0 — never SigLIP1's `1`, which is
  // the `<eos>` id under the Gemma vocab. `set_text_pad_token_id` overrides it
  // (the hook the factory constructor uses for checkpoint-resolved ids), and
  // the declared `TextEncoding` scheme tracks the override.
  let mut model = tiny_model();
  assert_eq!(model.text_pad_token_id(), 0, "default pad id is <pad> = 0");
  model.set_text_pad_token_id(7);
  assert_eq!(model.text_pad_token_id(), 7);
  let enc = model.text_encoding();
  assert_eq!(
    enc.padding,
    Padding::FixedLength {
      length: MAX_POS as usize,
      pad_token_id: 7,
      eos_token_id: Some(1),
    },
    "the declared fixed-length scheme pads with the overridden id"
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
fn strip_prefix_splits_and_removes_matching_keys() {
  // `strip_prefix` (now fallible — both the matched-key `Vec` and the
  // destination map reserve via `reserve_or_error`) must move every
  // prefix-matching key into the returned sub-map with the prefix removed, and
  // leave the non-matching keys in the source. A clean run (not an allocator
  // failure) returns `Ok` and the exact split.
  let mut w: HashMap<String, Array> = HashMap::new();
  w.insert("vision_model.vision_model.a".to_string(), vec1(HIDDEN));
  w.insert("vision_model.vision_model.b".to_string(), vec1(HIDDEN));
  w.insert("text_model.text_model.x".to_string(), vec1(HIDDEN));
  w.insert("logit_scale".to_string(), vec1(1));

  let vision = super::strip_prefix(&mut w, "vision_model.vision_model.").unwrap();
  assert_eq!(vision.len(), 2, "two vision keys moved out");
  assert!(vision.contains_key("a") && vision.contains_key("b"));
  // The matched keys are gone from the source; the others remain.
  assert!(!w.keys().any(|k| k.starts_with("vision_model.")));
  assert!(w.contains_key("text_model.text_model.x"));
  assert!(w.contains_key("logit_scale"));

  // A prefix that matches nothing yields an empty sub-map (the reserve-0 path)
  // and leaves the source untouched.
  let before = w.len();
  let none = super::strip_prefix(&mut w, "no_such_prefix.").unwrap();
  assert!(none.is_empty(), "no match → empty sub-map");
  assert_eq!(
    w.len(),
    before,
    "non-matching strip leaves the source intact"
  );
}

#[test]
fn strip_prefix_clones_full_matched_keys_through_fallible_path() {
  // ROOT-cause regression: `strip_prefix` builds its matched-key list (the FULL
  // checkpoint keys needed to `remove` each entry) through the fallible
  // `fallible_clone_str` path — not an infallible `String::clone`. Exercise that
  // path on a many-key map: every matched FULL key must be cloned + removed, the
  // stripped key inserted, and the result correct (a clean run returns `Ok`).
  let mut w: HashMap<String, Array> = HashMap::new();
  // A wide set of long, prefix-matching keys so the full-key clone loop runs many
  // times over realistically-sized checkpoint keys.
  for i in 0..32 {
    w.insert(
      format!("vision_model.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"),
      vec1(HIDDEN),
    );
  }
  w.insert("logit_scale".to_string(), vec1(1));

  let vision = super::strip_prefix(&mut w, "vision_model.vision_model.").unwrap();
  assert_eq!(
    vision.len(),
    32,
    "all 32 full-key matches were cloned + moved"
  );
  // Each entry is present under its STRIPPED key (the prefix removed) ...
  for i in 0..32 {
    let stripped = format!("encoder.layers.{i}.self_attn.q_proj.weight");
    assert!(
      vision.contains_key(&stripped),
      "stripped key {stripped} present"
    );
  }
  // ... and every FULL matched key was removed from the source (the removal uses
  // the fully-cloned key — the path ROOT FIX 2 made fallible).
  assert!(
    !w.keys().any(|k| k.starts_with("vision_model.")),
    "every full matched key was removed from the source"
  );
  assert!(w.contains_key("logit_scale"), "non-matching key retained");

  // The full-key clone itself goes through the fallible String path: a clean
  // clone of a full-length checkpoint key returns `Ok` and is byte-identical.
  let full = "vision_model.vision_model.encoder.layers.0.self_attn.q_proj.weight";
  let cloned = super::fallible_clone_str("strip_prefix: matched key", full).unwrap();
  assert_eq!(
    cloned, full,
    "the fallible full-key clone is byte-identical"
  );
}

#[test]
fn sanitize_and_strip_prefix_roundtrip_on_full_fixture() {
  // Exercise the full fallible allocation path end-to-end on the (larger)
  // model fixture: un-namespace the post-sanitize map to raw HF keys, re-run
  // `sanitize` (fallible destination-map reserve + per-key `insert_unique`),
  // then `strip_prefix` each tower (fallible matched-key `Vec` + sub-map
  // reserve). The split must reproduce the original per-tower key sets — proving
  // the reserve-based refactor preserved behavior on a real-sized key count.
  let post = tiny_model_weights();
  let mut raw: HashMap<String, Array> = HashMap::new();
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
  let mut sanitized = sanitize(raw).unwrap();
  let vision = super::strip_prefix(&mut sanitized, "vision_model.vision_model.").unwrap();
  let text = super::strip_prefix(&mut sanitized, "text_model.text_model.").unwrap();
  // The patch-embed weight is a vision key; the token-embedding a text key.
  assert!(vision.contains_key("embeddings.patch_embedding.weight"));
  assert!(text.contains_key("embeddings.token_embedding.weight"));
  // After both strips, only the top-level contrastive params remain.
  assert!(sanitized.contains_key("logit_scale") && sanitized.contains_key("logit_bias"));
  assert!(
    !sanitized
      .keys()
      .any(|k| k.starts_with("vision_model.") || k.starts_with("text_model.")),
    "every tower key was stripped"
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
  // SigLIP bakes its own pooling, so the constructor ignores the pooling config.
  let model = ctor(&loaded, None).expect("constructor must build the model");
  // The constructed model answers the umbrella's universal text capability; its
  // `embed_text` runs the text tower to a `(batch, projection)` embedding.
  let emb = model
    .as_text_embedder()
    .expect("siglip exposes the universal text embedder")
    .embed_text(&ids(1, 4), &mask(1, 4))
    .unwrap();
  assert_eq!(emb.array().shape(), vec![1, PROJ as usize]);
}

// ───────────────────── pad-id resolution from tokenizer metadata ─────────────────────
//
// The factory constructor refines the text pad id from the loaded TOKENIZER
// directory's metadata (`tokenizer_config.json` / `special_tokens_map.json` —
// the same directory `embeddings::load` builds the `Tokenizer` from, so a
// split `tokenizer_source` resolves from the tokenizer's own directory),
// falling back to the Gemma `<pad> = 0` default on any miss. These pin the
// resolution order, both HF token shapes (plain string and AddedToken object),
// and the robustness contract (missing files / malformed JSON never error —
// the default is already correct).

/// A fresh, writable per-test temp directory (the crate's no-`tempfile`-crate
/// convention: `temp_dir()` + pid + a process-unique counter so parallel tests
/// never collide). Created empty.
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-siglip2-padid-{tag}-{}-{n}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// The raw (HF-namespaced) weight map the `LoadedEmbeddingModel` carries —
/// the post-sanitize fixture un-namespaced one level.
fn raw_tiny_weights() -> HashMap<String, Array> {
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
  raw
}

/// Build via the registry constructor from a `LoadedEmbeddingModel` with the
/// tokenizer directory attached, and read back the pad id the constructed
/// model's `TextEncoding` declares.
fn constructed_pad_id(dir: &std::path::Path) -> u32 {
  let loaded = LoadedEmbeddingModel::new(
    "siglip2".to_string(),
    tiny_config_json(),
    raw_tiny_weights(),
  )
  .with_tokenizer_dir(dir);
  let model = constructor()(&loaded, None).expect("constructor must build the model");
  let enc = model
    .as_text_embedder()
    .expect("siglip exposes the universal text embedder")
    .text_encoding();
  match enc.padding {
    Padding::FixedLength { pad_token_id, .. } => pad_token_id,
    other => panic!("siglip declares FixedLength padding, got {other:?}"),
  }
}

#[test]
fn constructor_resolves_pad_id_from_tokenizer_config() {
  // The real `google/siglip2-base-patch16-naflex` layout: a string `pad_token`
  // plus the `added_tokens_decoder` binding. A NON-zero id (3) proves the value
  // is read from the metadata, not the 0 default.
  let dir = fresh_dir("cfg");
  std::fs::write(
    dir.join("tokenizer_config.json"),
    r#"{
      "add_eos_token": true,
      "added_tokens_decoder": {
        "3": { "content": "<pad>", "special": true },
        "1": { "content": "<eos>", "special": true }
      },
      "pad_token": "<pad>",
      "eos_token": "<eos>"
    }"#,
  )
  .unwrap();
  assert_eq!(constructed_pad_id(&dir), 3, "pad id read from metadata");
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constructor_resolves_pad_id_from_special_tokens_map_fallback() {
  // `pad_token` absent from tokenizer_config.json but present (in the
  // AddedToken object shape) in special_tokens_map.json; the id still resolves
  // through tokenizer_config.json's `added_tokens_decoder`.
  let dir = fresh_dir("special");
  std::fs::write(
    dir.join("tokenizer_config.json"),
    r#"{
      "added_tokens_decoder": {
        "5": { "content": "<pad>", "special": true },
        "1": { "content": "<eos>", "special": true }
      },
      "eos_token": "<eos>"
    }"#,
  )
  .unwrap();
  std::fs::write(
    dir.join("special_tokens_map.json"),
    r#"{ "pad_token": { "content": "<pad>", "lstrip": false } }"#,
  )
  .unwrap();
  assert_eq!(
    constructed_pad_id(&dir),
    5,
    "pad token string from special_tokens_map.json resolves via added_tokens_decoder"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constructor_pad_id_falls_back_to_zero_without_metadata() {
  // No tokenizer dir attached (a hand-built LoadedEmbeddingModel), a dir with
  // no metadata files, a dir with MALFORMED JSON, and a dir whose metadata
  // cannot resolve the token to an id — every miss keeps the Gemma `<pad> = 0`
  // default and never turns the load into an error.
  // (a) no tokenizer dir attached.
  let loaded = LoadedEmbeddingModel::new(
    "siglip2".to_string(),
    tiny_config_json(),
    raw_tiny_weights(),
  );
  let model = constructor()(&loaded, None).expect("constructor must build without a tokenizer dir");
  let enc = model.as_text_embedder().unwrap().text_encoding();
  assert!(
    matches!(
      enc.padding,
      Padding::FixedLength {
        pad_token_id: 0,
        ..
      }
    ),
    "no tokenizer dir → default pad id 0"
  );

  // (b) dir without metadata files.
  let empty = fresh_dir("empty");
  assert_eq!(constructed_pad_id(&empty), 0, "no metadata files → 0");
  let _ = std::fs::remove_dir_all(&empty);

  // (c) malformed JSON in both files.
  let bad = fresh_dir("malformed");
  std::fs::write(bad.join("tokenizer_config.json"), "{ not json !!").unwrap();
  std::fs::write(bad.join("special_tokens_map.json"), "[1, 2,").unwrap();
  assert_eq!(constructed_pad_id(&bad), 0, "malformed metadata → 0");
  let _ = std::fs::remove_dir_all(&bad);

  // (d) a pad_token string no added_tokens_decoder entry resolves.
  let unresolved = fresh_dir("unresolved");
  std::fs::write(
    unresolved.join("tokenizer_config.json"),
    r#"{ "pad_token": "<pad>", "added_tokens_decoder": { "1": { "content": "<eos>" } } }"#,
  )
  .unwrap();
  assert_eq!(
    constructed_pad_id(&unresolved),
    0,
    "unresolvable pad token → 0"
  );
  let _ = std::fs::remove_dir_all(&unresolved);
}

#[test]
fn read_text_pad_token_id_handles_both_hf_token_shapes() {
  // The `pad_token` field appears as a plain string OR an AddedToken-style
  // `{"content": …}` object across HF checkpoints; both must resolve. A
  // non-u32 decoder key (a corrupt "<id>") is a SKIPPED entry, not a panic and
  // not a scan abort.
  let dir = fresh_dir("shapes");
  // Object shape in tokenizer_config.json itself.
  std::fs::write(
    dir.join("tokenizer_config.json"),
    r#"{
      "pad_token": { "content": "<pad>" },
      "added_tokens_decoder": { "0": { "content": "<pad>" } }
    }"#,
  )
  .unwrap();
  assert_eq!(super::read_text_pad_token_id(&dir), Some(0));

  // ONLY a corrupt (non-numeric) decoder id → None (the caller's 0 default
  // applies — no numeric-keyed entry resolves the token).
  std::fs::write(
    dir.join("tokenizer_config.json"),
    r#"{
      "pad_token": "<pad>",
      "added_tokens_decoder": { "zero": { "content": "<pad>" } }
    }"#,
  )
  .unwrap();
  assert_eq!(super::read_text_pad_token_id(&dir), None);

  // A corrupt non-numeric entry ALONGSIDE the legitimate numeric binding: the
  // corrupt entry is skipped (continue), so the planted junk cannot shadow the
  // real `"0" → <pad>` binding regardless of map iteration order.
  std::fs::write(
    dir.join("tokenizer_config.json"),
    r#"{
      "pad_token": "<pad>",
      "added_tokens_decoder": {
        "junk": { "content": "<pad>" },
        "0": { "content": "<pad>" }
      }
    }"#,
  )
  .unwrap();
  assert_eq!(
    super::read_text_pad_token_id(&dir),
    Some(0),
    "a corrupt non-numeric decoder entry must not shadow the numeric binding"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────── quantized-checkpoint loading ─────────────────────
//
// No local quantized SigLIP2 checkpoint is available (`models/siglip2-naflex`
// is dense F32), so the quantized load path is covered by a SYNTHETIC quantized
// checkpoint: a quantization-friendly tiny config whose widths are divisible by
// the affine `group_size` (mlx requires `group_size ∈ {32, 64, 128}`,
// `mlx/ops.cpp:4740`), with the TEXT tower's `nn.Linear` weights + token / position
// embeddings replaced by the real `ops::quantized::quantize`
// `(weight, scales, biases)` triple — the exact on-disk layout an mlx-community
// 8-bit checkpoint ships. The VISION tower is left dense, mirroring
// `quantize_model`'s default `skip_vision=True`. The model must then construct
// (building `QuantizedLinear` / quantized embeddings) and run a full
// encode_image + encode_text forward to finite output.

/// A valid mlx affine group size (`mlx/ops.cpp:4740`) that divides the
/// quantization-friendly fixture's `hidden` / `intermediate`.
const QGROUP: i32 = 32;
/// 8-bit affine — the common mlx-community embedding scheme.
const QBITS: i32 = 8;
const Q_HIDDEN: i32 = 32;
const Q_INTER: i32 = 64;
const Q_HEADS: i32 = 2;
const Q_VOCAB: i32 = 12;
const Q_MAX_POS: i32 = 6;
const Q_PROJ: i32 = 32;

/// A quantization-friendly tiny `config.json` body: every quantized weight's
/// input axis (`hidden` for attn/fc1/head/embeddings, `intermediate` for fc2)
/// is a whole multiple of `QGROUP`. The vision tower uses the same small dims
/// but stays dense (its patch-embed width `P^2*C = 12` is not a `QGROUP`
/// multiple — exactly the divisibility skip `get_class_predicate` applies).
fn quant_config_json() -> String {
  format!(
    r#"{{
      "model_type": "siglip2",
      "num_labels": 0,
      "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS}, "mode": "affine" }},
      "text_config": {{
        "model_type": "siglip2_text_model",
        "vocab_size": {Q_VOCAB},
        "max_position_embeddings": {Q_MAX_POS},
        "hidden_size": {Q_HIDDEN},
        "intermediate_size": {Q_INTER},
        "num_attention_heads": {Q_HEADS},
        "num_hidden_layers": 1,
        "layer_norm_eps": 1e-6
      }},
      "vision_config": {{
        "model_type": "siglip2_vision_model",
        "image_size": 4,
        "patch_size": {PATCH},
        "num_channels": {CHANNELS},
        "hidden_size": {Q_HIDDEN},
        "intermediate_size": {Q_INTER},
        "num_attention_heads": {Q_HEADS},
        "num_hidden_layers": 1,
        "layer_norm_eps": 1e-6,
        "vision_use_head": true,
        "num_patches": {NUM_PATCHES},
        "max_num_patches": {NUM_PATCHES}
      }}
    }}"#
  )
}

/// A dense `(rows, cols)` ramp friendly to affine quantization.
fn qmat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c).map(|n| (n as f32) * 0.001 + 0.001).collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

fn qvec(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| (i as f32) * 0.001).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple (`<prefix>.weight` packed +
/// `<prefix>.scales` + `<prefix>.biases`) — how an mlx-community quantized
/// checkpoint stores a quantized `nn.Linear` / `nn.Embedding`. The
/// `<prefix>.bias` (dense output bias), if any, is left untouched.
fn quantize_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .expect("dense weight present");
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// Insert a dense (q-friendly) attention / mlp / ln block under a tower prefix.
fn q_insert_attn(w: &mut HashMap<String, Array>, prefix: &str) {
  for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(format!("{prefix}.{p}.weight"), qmat(Q_HIDDEN, Q_HIDDEN));
    w.insert(format!("{prefix}.{p}.bias"), qvec(Q_HIDDEN));
  }
}
fn q_insert_mlp(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.fc1.weight"), qmat(Q_INTER, Q_HIDDEN));
  w.insert(format!("{prefix}.fc1.bias"), qvec(Q_INTER));
  w.insert(format!("{prefix}.fc2.weight"), qmat(Q_HIDDEN, Q_INTER));
  w.insert(format!("{prefix}.fc2.bias"), qvec(Q_HIDDEN));
}
fn q_insert_ln(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.weight"), qvec(Q_HIDDEN));
  w.insert(format!("{prefix}.bias"), qvec(Q_HIDDEN));
}

/// The full *post-sanitize* dual-tower weight map for the quantization-friendly
/// config (dense everywhere first). The quantizing step below replaces the text
/// tower's Linears + embeddings with the affine triple (vision stays dense).
fn quant_model_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();

  // ── vision tower (vision_model.vision_model.*) — stays DENSE ──
  let vp = "vision_model.vision_model";
  w.insert(
    format!("{vp}.embeddings.patch_embedding.weight"),
    qmat(Q_HIDDEN, PATCH_FEAT),
  );
  w.insert(
    format!("{vp}.embeddings.patch_embedding.bias"),
    qvec(Q_HIDDEN),
  );
  w.insert(
    format!("{vp}.embeddings.position_embedding.weight"),
    qmat(NUM_PATCHES, Q_HIDDEN),
  );
  q_insert_ln(&mut w, &format!("{vp}.encoder.layers.0.layer_norm1"));
  q_insert_attn(&mut w, &format!("{vp}.encoder.layers.0.self_attn"));
  q_insert_ln(&mut w, &format!("{vp}.encoder.layers.0.layer_norm2"));
  q_insert_mlp(&mut w, &format!("{vp}.encoder.layers.0.mlp"));
  q_insert_ln(&mut w, &format!("{vp}.post_layernorm"));
  w.insert(
    format!("{vp}.head.probe"),
    Array::from_slice::<f32>(
      &(0..Q_HIDDEN as usize)
        .map(|i| (i as f32) * 0.01 + 0.01)
        .collect::<Vec<_>>(),
      &(1usize, 1usize, Q_HIDDEN as usize),
    )
    .unwrap(),
  );
  w.insert(
    format!("{vp}.head.attention.in_proj.weight"),
    qmat(3 * Q_HIDDEN, Q_HIDDEN),
  );
  w.insert(
    format!("{vp}.head.attention.in_proj.bias"),
    qvec(3 * Q_HIDDEN),
  );
  w.insert(
    format!("{vp}.head.attention.out_proj.weight"),
    qmat(Q_HIDDEN, Q_HIDDEN),
  );
  w.insert(format!("{vp}.head.attention.out_proj.bias"), qvec(Q_HIDDEN));
  q_insert_ln(&mut w, &format!("{vp}.head.layernorm"));
  q_insert_mlp(&mut w, &format!("{vp}.head.mlp"));

  // ── text tower (text_model.text_model.*) — DENSE first, then quantized ──
  let tp = "text_model.text_model";
  w.insert(
    format!("{tp}.embeddings.token_embedding.weight"),
    qmat(Q_VOCAB, Q_HIDDEN),
  );
  w.insert(
    format!("{tp}.embeddings.position_embedding.weight"),
    qmat(Q_MAX_POS, Q_HIDDEN),
  );
  q_insert_ln(&mut w, &format!("{tp}.encoder.layers.0.layer_norm1"));
  q_insert_attn(&mut w, &format!("{tp}.encoder.layers.0.self_attn"));
  q_insert_ln(&mut w, &format!("{tp}.encoder.layers.0.layer_norm2"));
  q_insert_mlp(&mut w, &format!("{tp}.encoder.layers.0.mlp"));
  q_insert_ln(&mut w, &format!("{tp}.final_layer_norm"));
  w.insert(format!("{tp}.head.weight"), qmat(Q_PROJ, Q_HIDDEN));
  w.insert(format!("{tp}.head.bias"), qvec(Q_PROJ));

  // ── top-level contrastive params ──
  w.insert(
    "logit_scale".to_string(),
    Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap(),
  );
  w.insert(
    "logit_bias".to_string(),
    Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap(),
  );

  // Quantize the TEXT tower's Linears + embeddings (vision stays dense, the
  // `skip_vision=True` default). token/position embeddings AND every Linear.
  let tl = "text_model.text_model.encoder.layers.0";
  for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    quantize_in_place(&mut w, &format!("{tl}.self_attn.{p}"));
  }
  quantize_in_place(&mut w, &format!("{tl}.mlp.fc1"));
  quantize_in_place(&mut w, &format!("{tl}.mlp.fc2"));
  quantize_in_place(&mut w, "text_model.text_model.head");
  quantize_in_place(&mut w, "text_model.text_model.embeddings.token_embedding");
  quantize_in_place(
    &mut w,
    "text_model.text_model.embeddings.position_embedding",
  );
  w
}

/// The parsed global 8-bit affine quantization config for the synthetic
/// checkpoint (the analogue of the `config.json` `quantization` block).
fn quant_config() -> PerLayerQuantization {
  PerLayerQuantization::from_global(Quantization::affine(QGROUP, QBITS))
}

fn quant_image_inputs() -> NaflexInputs {
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

fn q_ids(batch: usize, seq: usize) -> Array {
  let data: Vec<i32> = (0..batch * seq)
    .map(|n| (n % Q_VOCAB as usize) as i32)
    .collect();
  Array::from_slice::<i32>(&data, &(batch, seq)).unwrap()
}

#[test]
fn from_weights_quantized_builds_and_forwards_text_to_finite() {
  // An 8-bit text tower loads AND runs encode_text through the quantized
  // attention/MLP `quantized_matmul`, the quantized token/position embedding
  // (dequantize-gather / dequantized-table), and the quantized pooled head.
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let model =
    Siglip2NaflexModel::from_weights_quantized(cfg, quant_model_weights(), Some(&quant_config()))
      .unwrap();

  let text = model.encode_text(&q_ids(2, 4)).unwrap();
  assert_eq!(text.shape(), vec![2, Q_PROJ as usize]);
  for v in eval_to_vec(&text) {
    assert!(v.is_finite(), "quantized text embedding non-finite: {v}");
  }
  // L2-normalized rows.
  let flat = eval_to_vec(&text);
  for n in row_norms(&flat, 2, Q_PROJ as usize) {
    assert!((n - 1.0).abs() < 1e-4, "text row not unit-norm: {n}");
  }
}

#[test]
fn from_weights_quantized_runs_image_through_dense_vision() {
  // The vision tower is dense (skip_vision); it still forwards to a finite,
  // L2-normalized image embedding under a threaded quantization config.
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let model =
    Siglip2NaflexModel::from_weights_quantized(cfg, quant_model_weights(), Some(&quant_config()))
      .unwrap();

  let img = model.encode_image(&quant_image_inputs()).unwrap();
  assert_eq!(img.shape(), vec![1, Q_HIDDEN as usize]);
  for v in eval_to_vec(&img) {
    assert!(v.is_finite(), "image embedding non-finite: {v}");
  }
}

#[test]
fn from_weights_quantized_attention_pool_in_proj_dequantizes_to_dense() {
  // A fully-quantized checkpoint that ALSO quantizes the attention-pool head's
  // combined-QKV `in_proj` (shipping `in_proj.scales` / `in_proj.biases`
  // alongside the packed weight) must load: the `.scales` sibling gates the
  // dequantize-to-dense path (the row-sliced `in_proj` is logically dense), so
  // the head still forwards to a finite, L2-normalized image embedding rather
  // than running dense with the siblings ignored or failing on a packed shape.
  let mut weights = quant_model_weights();
  // Quantize the vision attention-pool `in_proj` (96, 32) — input width 32 is a
  // whole `QGROUP` multiple — leaving its DENSE `.bias` (the MHA additive bias)
  // untouched, exactly how `quantize_model` would write the triple.
  quantize_in_place(
    &mut weights,
    "vision_model.vision_model.head.attention.in_proj",
  );
  assert!(
    weights.contains_key("vision_model.vision_model.head.attention.in_proj.scales"),
    "fixture must carry the quantized in_proj `.scales` sibling"
  );

  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let model = Siglip2NaflexModel::from_weights_quantized(cfg, weights, Some(&quant_config()))
    .expect("a quantized in_proj must dequantize-to-dense and load");

  let img = model.encode_image(&quant_image_inputs()).unwrap();
  assert_eq!(img.shape(), vec![1, Q_HIDDEN as usize]);
  let flat = eval_to_vec(&img);
  for v in &flat {
    assert!(
      v.is_finite(),
      "quantized-in_proj image embedding non-finite: {v}"
    );
  }
  for n in row_norms(&flat, 1, Q_HIDDEN as usize) {
    assert!((n - 1.0).abs() < 1e-4, "image row not unit-norm: {n}");
  }
}

#[test]
fn from_weights_dense_in_proj_unchanged_by_quant_gate() {
  // The dense `in_proj` path (no `.scales` sibling) is byte-identical whether or
  // not a quantization config is threaded — the gate only diverts when the
  // sibling is present. The fully-dense tiny fixture carries no `.scales`
  // anywhere (in_proj included), so the attention-pool image embedding matches
  // the plain (no-config) load exactly, with or without a threaded config.
  let plain = tiny_model();
  let cfg = Siglip2NaflexConfig::from_json(&tiny_config_json()).unwrap();
  let with_cfg =
    Siglip2NaflexModel::from_weights_quantized(cfg, tiny_model_weights(), Some(&quant_config()))
      .unwrap();

  let a = eval_to_vec(&plain.encode_image(&tiny_image_inputs()).unwrap());
  let b = eval_to_vec(&with_cfg.encode_image(&tiny_image_inputs()).unwrap());
  assert_eq!(a.len(), b.len());
  for (x, y) in a.iter().zip(b.iter()) {
    assert!(
      (x - y).abs() < 1e-6,
      "dense in_proj diverged with quant config: {x} vs {y}"
    );
  }
}

#[test]
fn from_weights_quantized_constructor_path_parses_config_block() {
  // The registry constructor parses the `config.json` `quantization` block and
  // loads the quantized model end-to-end (un-namespace the post-sanitize fixture
  // to the raw HF layout the LoadedEmbeddingModel carries).
  let post = quant_model_weights();
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
  let loaded = LoadedEmbeddingModel::new("siglip2".to_string(), quant_config_json(), raw);
  let model = constructor()(&loaded, None).expect("constructor must build the quantized model");
  let emb = model
    .as_text_embedder()
    .unwrap()
    .embed_text(&q_ids(1, 4), &mask(1, 4))
    .unwrap();
  assert_eq!(emb.array().shape(), vec![1, Q_PROJ as usize]);
}

#[test]
fn from_weights_dense_checkpoint_unchanged_with_quant_config() {
  // A NON-quantized (dense) checkpoint loads identically whether or not a
  // quantization config is threaded — the `.scales` sibling is the load-bearing
  // signal; the dense tiny fixture has none, so the dense path runs regardless,
  // producing the SAME embedding as the plain `from_weights`.
  let plain = tiny_model();
  let cfg = Siglip2NaflexConfig::from_json(&tiny_config_json()).unwrap();
  let with_cfg =
    Siglip2NaflexModel::from_weights_quantized(cfg, tiny_model_weights(), Some(&quant_config()))
      .unwrap();

  let a = eval_to_vec(&plain.encode_text(&ids(1, 4)).unwrap());
  let b = eval_to_vec(&with_cfg.encode_text(&ids(1, 4)).unwrap());
  assert_eq!(a.len(), b.len());
  for (x, y) in a.iter().zip(b.iter()) {
    assert!(
      (x - y).abs() < 1e-6,
      "dense path diverged with quant config: {x} vs {y}"
    );
  }
}

#[test]
fn from_weights_quantized_scales_without_config_errors() {
  // Weights say quantized (`.scales` present) but no quantization config
  // resolved scheme params → a typed InvariantViolation, not a silent wrong
  // load. Thread an EMPTY per-layer config with no global default so
  // `quantization_for` returns None for the quantized text layers.
  let empty_cfg = PerLayerQuantization::new(None, HashMap::new());
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  // `Siglip2NaflexModel` is not `Debug`, so take the error via `.err()` rather
  // than `.unwrap_err()` (which would require the `Ok` value to be `Debug`).
  let err =
    Siglip2NaflexModel::from_weights_quantized(cfg, quant_model_weights(), Some(&empty_cfg))
      .err()
      .expect("scales-without-params must error");
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for `.scales` present but no resolved params, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_linear_scales_without_quant_config_errors() {
  // A `.scales` sibling selects the quantized path ALONE — even when NO
  // quantization config is threaded (`quant == None`, the plain public
  // `from_weights`). A text-tower Linear carrying `.scales` must then surface the
  // typed `.scales`-without-scheme InvariantViolation rather than silently
  // falling through to the dense loader (which could accept a malformed packed
  // `uint32` weight as a dense `Linear`). `quant_model_weights` quantizes the
  // text tower, so its Linears carry real `.scales` siblings; `from_weights`
  // threads `quantization == None`.
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let err = Siglip2NaflexModel::from_weights(cfg, quant_model_weights())
    .err()
    .expect("a `.scales`-bearing Linear with no quant config must error");
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for a quantized Linear `.scales` sibling with quant == None, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_patch_embedding_scales_without_quant_config_errors() {
  // The same `.scales`-sibling-alone rule for the vision patch embedding gate:
  // an `embeddings.patch_embedding.scales` sibling selects the quantized patch
  // path even with `quant == None`, and a layer that resolves no scheme params
  // must error (not silently load a malformed packed weight as a dense one).
  // Start from the all-dense vision tower and inject ONLY the `.scales` sibling
  // (the error fires on scheme resolution, before any shape check), then load
  // via the plain `from_weights` (`quantization == None`).
  let mut w = quant_model_weights();
  w.insert(
    "vision_model.vision_model.embeddings.patch_embedding.scales".to_string(),
    qvec(Q_HIDDEN),
  );
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let err = Siglip2NaflexModel::from_weights(cfg, w)
    .err()
    .expect("a `.scales`-bearing patch embedding with no quant config must error");
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for a patch-embedding `.scales` sibling with quant == None, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_logical_width_linear() {
  // A quantized text `q_proj` whose packed weight unpacks to the WRONG logical
  // input width must be rejected at load time (the quantized path reaches the
  // same config-shape gate the dense path does). Re-quantize a DENSE weight of a
  // deliberately-wrong input width (a whole multiple of QGROUP so quantize
  // succeeds, but != Q_HIDDEN) and splice it under the layer.
  let mut w = quant_model_weights();
  let prefix = "text_model.text_model.encoder.layers.0.self_attn.q_proj";
  // Remove the existing (correct) triple, splice a wrong-input-width one.
  w.remove(&format!("{prefix}.weight"));
  w.remove(&format!("{prefix}.scales"));
  w.remove(&format!("{prefix}.biases"));
  let wrong_in = Q_HIDDEN + QGROUP; // 64 != 32, still a QGROUP multiple
  let dense = qmat(Q_HIDDEN, wrong_in);
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(format!("{prefix}.biases"), biases.unwrap());

  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let err = Siglip2NaflexModel::from_weights_quantized(cfg, w, Some(&quant_config()))
    .err()
    .expect("wrong-input-width quantized Linear must error");
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed shape error for a wrong-input-width quantized Linear, got {err:?}"
  );
}

/// Splice a quantized token-embedding triple of a deliberately-wrong LOGICAL
/// `(rows, dim)` (a QGROUP multiple `dim` so `quantize` succeeds) into the text
/// tower's quantized weight map, replacing the correct one.
fn splice_wrong_token_embedding(w: &mut HashMap<String, Array>, rows: i32, dim: i32) {
  let prefix = "text_model.text_model.embeddings.token_embedding";
  w.remove(&format!("{prefix}.weight"));
  w.remove(&format!("{prefix}.scales"));
  w.remove(&format!("{prefix}.biases"));
  let dense = qmat(rows, dim);
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(format!("{prefix}.biases"), biases.unwrap());
}

#[test]
fn from_weights_quantized_rejects_wrong_vocab_token_embedding() {
  // A quantized token embedding whose LOGICAL row count (vocab) disagrees with
  // TextConfig must be rejected at LOAD — a mismatched packed table would
  // otherwise mis-gather (an out-of-range or wrong-row gather) at the first
  // forward. `Q_VOCAB + 1` rows, correct hidden.
  let mut w = quant_model_weights();
  splice_wrong_token_embedding(&mut w, Q_VOCAB + 1, Q_HIDDEN);
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let err = Siglip2NaflexModel::from_weights_quantized(cfg, w, Some(&quant_config()))
    .err()
    .expect("wrong-vocab quantized token embedding must error at load");
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed logical-shape error for a wrong-vocab quantized token embedding, got {err:?}"
  );
}

#[test]
fn from_weights_quantized_rejects_wrong_hidden_token_embedding() {
  // Same gate on the LOGICAL width (hidden): a quantized token embedding whose
  // dequantized `dim` disagrees with TextConfig is rejected at load. `Q_HIDDEN +
  // QGROUP` is a QGROUP multiple (so `quantize` succeeds) but != Q_HIDDEN.
  let mut w = quant_model_weights();
  splice_wrong_token_embedding(&mut w, Q_VOCAB, Q_HIDDEN + QGROUP);
  let cfg = Siglip2NaflexConfig::from_json(&quant_config_json()).unwrap();
  let err = Siglip2NaflexModel::from_weights_quantized(cfg, w, Some(&quant_config()))
    .err()
    .expect("wrong-hidden quantized token embedding must error at load");
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "expected a keyed logical-shape error for a wrong-hidden quantized token embedding, got {err:?}"
  );
}

#[test]
fn reproject_quant_keys_conflicting_dual_forms_is_key_collision() {
  // A per-layer override supplied in BOTH the shallow (`text_model.…`) and the
  // nested (`text_model.text_model.…`) form for the SAME layer with CONFLICTING
  // schemes must be a typed `KeyCollision` — not a silent nondeterministic
  // overwrite (whichever the source `HashMap` iterates last would otherwise win).
  use crate::lm::quant::QuantizationOption;
  let layer = "encoder.layers.0.self_attn.q_proj";
  let shallow = format!("text_model.{layer}");
  let nested = format!("text_model.text_model.{layer}");
  let mut per_layer = HashMap::new();
  per_layer.insert(
    shallow,
    QuantizationOption::Quantize(Quantization::affine(QGROUP, QBITS)),
  );
  per_layer.insert(
    nested,
    // DIFFERENT bits ⇒ a conflicting scheme for the same reprojected layer.
    QuantizationOption::Quantize(Quantization::affine(QGROUP, 4)),
  );
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, QBITS)), per_layer);
  let err = reproject_quant_keys(&quant, "text_model.text_model.")
    .expect_err("conflicting dual-form per-layer override must error");
  assert!(
    matches!(err, Error::KeyCollision(_)),
    "expected a KeyCollision for conflicting dual-form overrides, got {err:?}"
  );
}

#[test]
fn reproject_quant_keys_identical_dual_forms_load_fine() {
  // The control: the SAME layer in both the shallow and the nested form with
  // IDENTICAL options is NOT a conflict — one is kept and the reprojection
  // succeeds with the single tower-relative override.
  use crate::lm::quant::QuantizationOption;
  let layer = "encoder.layers.0.self_attn.q_proj";
  let opt = QuantizationOption::Quantize(Quantization::affine(QGROUP, QBITS));
  let mut per_layer = HashMap::new();
  per_layer.insert(format!("text_model.{layer}"), opt);
  per_layer.insert(format!("text_model.text_model.{layer}"), opt);
  let quant = PerLayerQuantization::new(Some(Quantization::affine(QGROUP, QBITS)), per_layer);
  let reprojected =
    reproject_quant_keys(&quant, "text_model.text_model.").expect("identical dual forms must load");
  // Exactly the one tower-relative key survives, carrying the shared option.
  assert_eq!(reprojected.per_layer_ref().len(), 1);
  assert_eq!(
    reprojected.quantization_for(layer),
    Some(Quantization::affine(QGROUP, QBITS))
  );
}

// ───────────────── parse_quantization per-mode group_size/bits defaults ─────────────────
//
// `parse_quantization` injects the PER-MODE `(group_size, bits)` default for an
// omitted key — the `mode` is parsed FIRST, then the shared
// `lm::convert::defaults_for_mode` table resolves the fallback (a blanket
// `group_size = 64` would mis-resolve a non-affine block that relies on its mode
// default). The resolved global scheme is read back via `quantization_for` (any
// layer name returns the global default when there is no per-layer override).

/// A minimal `config.json` body carrying ONLY a `quantization` block with the
/// given `mode` and NO `group_size` / `bits`, so the per-mode default is what
/// `parse_quantization` injects.
fn quant_block_mode_only(mode: &str) -> String {
  format!(r#"{{ "quantization": {{ "mode": "{mode}" }} }}"#)
}

#[test]
fn parse_quantization_mxfp4_omitted_group_size_resolves_to_32() {
  let plq = parse_quantization(&quant_block_mode_only("mxfp4"))
    .expect("mxfp4 block parses")
    .expect("a quantization block is present");
  let q = plq
    .quantization_for("any.layer")
    .expect("global default resolves for an unlisted layer");
  assert_eq!(q.group_size, 32, "mxfp4 per-mode default group_size");
  assert_eq!(q.bits, 4, "mxfp4 per-mode default bits");
  assert_eq!(q.mode, crate::lm::quant::QuantMode::Mxfp4);
}

#[test]
fn parse_quantization_mxfp8_omitted_group_size_resolves_to_32() {
  let plq = parse_quantization(&quant_block_mode_only("mxfp8"))
    .expect("mxfp8 block parses")
    .expect("a quantization block is present");
  let q = plq
    .quantization_for("any.layer")
    .expect("global default resolves for an unlisted layer");
  assert_eq!(q.group_size, 32, "mxfp8 per-mode default group_size");
  assert_eq!(q.bits, 8, "mxfp8 per-mode default bits");
  assert_eq!(q.mode, crate::lm::quant::QuantMode::Mxfp8);
}

#[test]
fn parse_quantization_nvfp4_omitted_group_size_resolves_to_16() {
  let plq = parse_quantization(&quant_block_mode_only("nvfp4"))
    .expect("nvfp4 block parses")
    .expect("a quantization block is present");
  let q = plq
    .quantization_for("any.layer")
    .expect("global default resolves for an unlisted layer");
  assert_eq!(q.group_size, 16, "nvfp4 per-mode default group_size");
  assert_eq!(q.bits, 4, "nvfp4 per-mode default bits");
  assert_eq!(q.mode, crate::lm::quant::QuantMode::Nvfp4);
}

#[test]
fn parse_quantization_affine_omitted_group_size_resolves_to_64() {
  // The affine default is unchanged (64, 4) — including the implicit-affine case
  // (no `mode` key at all, which the deserializer defaults to affine).
  for cfg in [
    quant_block_mode_only("affine"),
    r#"{ "quantization": { "bits": 4 } }"#.to_string(),
  ] {
    let plq = parse_quantization(&cfg)
      .expect("affine block parses")
      .expect("a quantization block is present");
    let q = plq
      .quantization_for("any.layer")
      .expect("global default resolves for an unlisted layer");
    assert_eq!(q.group_size, 64, "affine per-mode default group_size");
    assert_eq!(q.mode, crate::lm::quant::QuantMode::Affine);
  }
}

#[test]
fn parse_quantization_explicit_group_size_overrides_mode_default() {
  // An explicit `group_size` / `bits` in the block is preserved verbatim — the
  // per-mode default only fills an OMITTED key.
  let cfg = r#"{ "quantization": { "group_size": 128, "bits": 4, "mode": "mxfp4" } }"#;
  let plq = parse_quantization(cfg)
    .expect("block parses")
    .expect("a quantization block is present");
  let q = plq.quantization_for("any.layer").expect("global default");
  assert_eq!(
    q.group_size, 128,
    "explicit group_size wins over mode default"
  );
  assert_eq!(q.bits, 4);
  assert_eq!(q.mode, crate::lm::quant::QuantMode::Mxfp4);
}

#[test]
fn parse_quantization_unknown_mode_is_typed_error() {
  // A present-but-unrecognized `mode` tag cannot resolve a per-mode default, so
  // it is a typed `UnknownEnumValue` rather than a silent affine guess.
  let cfg = r#"{ "quantization": { "bits": 4, "mode": "gptq" } }"#;
  let err = parse_quantization(cfg).expect_err("unknown mode must error");
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "expected UnknownEnumValue for an unrecognized quantization mode, got {err:?}"
  );
}

#[test]
fn parse_quantization_absent_mode_resolves_to_affine() {
  // An omitted `mode` resolves to `affine` (swift's `_mode ?? .affine`), even when
  // explicit positive `group_size` / `bits` are supplied — pinning the
  // absent-mode default independent of the per-mode default-fill path. Asserted at
  // the top level AND on a per-layer override (the single strict path covers both).
  let cfg = format!(
    r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "{PER_LAYER_PATH}": {{ "group_size": 256, "bits": 4 }} }} }}"#
  );
  let plq = parse_quantization(&cfg)
    .expect("absent-mode block parses")
    .expect("a quantization block is present");
  let global = plq
    .quantization_for("some.other.layer")
    .expect("global default resolves");
  assert_eq!(
    (global.group_size, global.bits, global.mode),
    (128, 8, crate::lm::quant::QuantMode::Affine),
    "an absent top-level mode resolves to affine with its explicit group_size/bits"
  );
  let per_layer = plq
    .quantization_for(PER_LAYER_PATH)
    .expect("per-layer override resolves");
  assert_eq!(
    (per_layer.group_size, per_layer.bits, per_layer.mode),
    (256, 4, crate::lm::quant::QuantMode::Affine),
    "an absent per-layer mode resolves to affine with its explicit group_size/bits"
  );
}

// ───────────────── parse_quantization falsy group_size/bits truthiness ─────────────────
//
// mlx-lm resolves `group_size` / `bits` via `value or default` (`utils.py:808`),
// so a present `0` or an explicit `null` falls back to the per-mode default
// exactly like an absent key (`0` is falsy). A present positive value still wins.
// Each falsy spelling must route through `defaults_for_mode` to the correct
// per-mode default (affine 64/4, mxfp4 32/4, nvfp4 16/4, mxfp8 32/8), not survive
// as the invalid value. A present NEGATIVE value is the lone departure — invalid
// for the quantization kernel, it is a typed `OutOfRange` (covered by the
// negative-rejection tests), not a falsy fallback.

/// A `config.json` body whose `quantization` block carries `mode` plus the
/// given raw JSON `group_size` and `bits` literals (each a JSON token such as
/// `0` or `null`).
fn quant_block_with(mode: &str, group_size: &str, bits: &str) -> String {
  format!(
    r#"{{ "quantization": {{ "mode": "{mode}", "group_size": {group_size}, "bits": {bits} }} }}"#
  )
}

/// Resolve the global `(group_size, bits)` a `config.json` body produces.
fn resolved_group_size_bits(cfg: &str) -> (i32, i32) {
  let plq = parse_quantization(cfg)
    .expect("block parses")
    .expect("a quantization block is present");
  let q = plq
    .quantization_for("any.layer")
    .expect("global default resolves for an unlisted layer");
  (q.group_size, q.bits)
}

#[test]
fn parse_quantization_falsy_group_size_resolves_to_per_mode_default() {
  // (mode, default_group_size, default_bits) per the shared `defaults_for_mode`
  // table; `bits` is held at a valid positive so only `group_size` is exercised.
  for (mode, default_gs, valid_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    for falsy in ["0", "null"] {
      let cfg = quant_block_with(mode, falsy, &valid_bits.to_string());
      let (gs, bits) = resolved_group_size_bits(&cfg);
      assert_eq!(
        gs, default_gs,
        "{mode}: falsy group_size `{falsy}` must fall back to the per-mode default"
      );
      assert_eq!(
        bits, valid_bits,
        "{mode}: valid bits `{valid_bits}` preserved"
      );
    }
  }
}

#[test]
fn parse_quantization_falsy_bits_resolves_to_per_mode_default() {
  // Same falsy spellings on `bits`; `group_size` is held at a valid positive.
  for (mode, valid_gs, default_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    for falsy in ["0", "null"] {
      let cfg = quant_block_with(mode, &valid_gs.to_string(), falsy);
      let (gs, bits) = resolved_group_size_bits(&cfg);
      assert_eq!(
        bits, default_bits,
        "{mode}: falsy bits `{falsy}` must fall back to the per-mode default"
      );
      assert_eq!(
        gs, valid_gs,
        "{mode}: valid group_size `{valid_gs}` preserved"
      );
    }
  }
}

// ───────────────── parse_quantization oversized group_size/bits → OutOfRange ─────────────────
//
// A present JSON integer that does NOT fit `i32` (e.g. `2147483648`, one past
// `i32::MAX`) is a typed `OutOfRange` error — it must NOT silently collapse to
// the per-mode default (which would mask corrupt/hostile config metadata and let
// the checkpoint load under different quant params than the JSON declared) nor
// silently truncate. The strict typed `QuantSpec` deserialize reads each value
// through serde's `i32` deserialization, which rejects the out-of-range integer
// before `defaults_for_mode` ever sees it.

/// One past `i32::MAX`: a JSON integer that does not fit `i32`. Serializes as a
/// plain JSON integer that serde_json represents as an `i64`.
const OVERSIZED_I32: i64 = i32::MAX as i64 + 1;

/// One past `i64::MAX`: a JSON integer too large for `i64`, so serde_json
/// represents it as a `u64`. The old `Value::as_i64` walk returned `None` here
/// and silently collapsed it to the per-mode default; serde's strict `i32`
/// deserialization rejects it as out of range, which the reframe surfaces as a
/// typed `OutOfRange`. This is the load-bearing edge the typed path fixes.
const OVERSIZED_U64: &str = "9223372036854775808";

/// A JSON float and a JSON string — neither is a JSON integer, so serde's `i32`
/// deserialization rejects each by TYPE (not range); the reframe maps that to a
/// typed `Error::Parse` naming the field. Held as raw JSON tokens for the
/// `quant_block_*` helpers (the string keeps its embedded quotes).
const NON_INTEGER_FLOAT: &str = "64.5";
const NON_INTEGER_STRING: &str = r#""64""#;

#[test]
fn parse_quantization_oversized_group_size_is_out_of_range() {
  // Held across every mode so the oversized rejection is mode-independent;
  // `bits` stays a valid positive so only `group_size` is exercised.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, &OVERSIZED_I32.to_string(), "4");
    let err = parse_quantization(&cfg).expect_err("oversized group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for an oversized group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_oversized_bits_is_out_of_range() {
  // Same oversized literal on `bits`; `group_size` stays a valid positive.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, "64", &OVERSIZED_I32.to_string());
    let err = parse_quantization(&cfg).expect_err("oversized bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for an oversized bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_u64_overflow_group_size_is_out_of_range() {
  // A magnitude PAST `i64::MAX` (so serde_json holds it as a `u64`) on
  // `group_size`. The old `Value::as_i64` walk dropped this to `None` and
  // silently used the per-mode default; serde's strict `i32` deserialization
  // rejects it as out of range. Held across every mode; `bits` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, OVERSIZED_U64, "4");
    let err = parse_quantization(&cfg).expect_err("u64-overflow group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a u64-overflow group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_u64_overflow_bits_is_out_of_range() {
  // Same u64-overflow literal on `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, "64", OVERSIZED_U64);
    let err = parse_quantization(&cfg).expect_err("u64-overflow bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a u64-overflow bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_non_integer_group_size_is_typed_error() {
  // A float (`64.5`) and a string (`"64"`) are not JSON integers — serde rejects
  // each by TYPE, which the reframe maps to a typed `Error::Parse` (NOT a silent
  // default and NOT a truncation to `64`). Held across every mode; `bits` valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    for non_int in [NON_INTEGER_FLOAT, NON_INTEGER_STRING] {
      let cfg = quant_block_with(mode, non_int, "4");
      let err = parse_quantization(&cfg).expect_err("non-integer group_size must error");
      assert!(
        matches!(err, Error::Parse(_)),
        "{mode}: expected Parse for a non-integer group_size `{non_int}`, got {err:?}"
      );
    }
  }
}

#[test]
fn parse_quantization_non_integer_bits_is_typed_error() {
  // Same non-integer values on `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    for non_int in [NON_INTEGER_FLOAT, NON_INTEGER_STRING] {
      let cfg = quant_block_with(mode, "64", non_int);
      let err = parse_quantization(&cfg).expect_err("non-integer bits must error");
      assert!(
        matches!(err, Error::Parse(_)),
        "{mode}: expected Parse for a non-integer bits `{non_int}`, got {err:?}"
      );
    }
  }
}

// ───────────────── parse_quantization negative group_size/bits → OutOfRange ─────────────────
//
// A present NEGATIVE `group_size` / `bits` is the lone departure from python's
// `value or default` truthiness (python keeps a negative — it is truthy). A
// negative `group_size` / `bits` is invalid for the `quantized_matmul` kernel
// (mlx's `quantize` asserts positive), so it must NOT silently collapse to the
// per-mode default (which would load the checkpoint under a quant param the
// config never declared — the same silent malformed-numeric class the oversized
// rejections above close). The single `resolve_quant_spec` guard rejects it as a
// typed `OutOfRange` before `defaults_for_mode`, at BOTH the top-level spec and
// every per-layer override. The remaining falsy spellings (`0` / `null` /
// absent) still fall back — the `..._zero_..._still_resolves...` guards pin that.

#[test]
fn parse_quantization_negative_group_size_is_out_of_range() {
  // Held across every mode so the negative rejection is mode-independent; `bits`
  // stays a valid positive so only `group_size` is exercised.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, "-1", "4");
    let err = parse_quantization(&cfg).expect_err("negative group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a negative group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_negative_bits_is_out_of_range() {
  // Same negative literal on `bits`; `group_size` stays a valid positive.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_with(mode, "64", "-1");
    let err = parse_quantization(&cfg).expect_err("negative bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a negative bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_zero_group_size_bits_still_resolves_to_per_mode_default() {
  // Guard the preserved falsy-default path: a present `0` (falsy) on BOTH fields
  // still resolves to the per-mode default — the negative guard above must not
  // have regressed `0` into an error. Held across every mode.
  for (mode, default_gs, default_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    let cfg = quant_block_with(mode, "0", "0");
    let (gs, bits) = resolved_group_size_bits(&cfg);
    assert_eq!(
      (gs, bits),
      (default_gs, default_bits),
      "{mode}: a present `0` group_size/bits still falls back to the per-mode default"
    );
  }
}

#[test]
fn parse_quantization_positive_group_size_bits_still_kept_alongside_negative_guard() {
  // Guard the preserved override path: a present POSITIVE value still wins over
  // the per-mode default (the negative guard only rejects negatives, never a
  // valid positive override).
  let (gs, bits) = resolved_group_size_bits(&quant_block_with("mxfp4", "128", "8"));
  assert_eq!(
    (gs, bits),
    (128, 8),
    "an explicit positive group_size/bits survives alongside the negative guard"
  );
}

// ───────────────── parse_quantization PER-LAYER override falsy/default ─────────────────
//
// The same `value or default` truthiness contract applies to per-layer override
// objects (the nested `{ mode?, group_size?, bits? }` schemes), not just the
// top-level global spec. A single uniform normalization pass routes EVERY
// quant-spec object through `defaults_for_mode`, so a per-layer override with a
// `0` / absent / null `group_size` or `bits` resolves to the per-mode default
// exactly like the global spec — there is no top-level-vs-per-layer gap. A
// per-layer positive value still overrides; a per-layer negative is the same
// typed `OutOfRange` the global spec raises (the one shared guard).

/// A representative per-layer override path (non-reserved, so the deserializer
/// treats it as a layer key). `parse_quantization` normalizes before any tower
/// reprojection, so the override is read back under this exact key.
const PER_LAYER_PATH: &str = "text_model.encoder.layers.0.self_attn.q_proj";

/// A `config.json` body whose `quantization` block carries a VALID top-level
/// global spec plus a single per-layer override at [`PER_LAYER_PATH`] with the
/// given raw JSON `group_size` and `bits` literals and the given `mode`.
fn quant_block_per_layer(mode: &str, group_size: &str, bits: &str) -> String {
  // The global spec is a fixed valid affine 128/8 so it can never be confused
  // with a per-mode default the per-layer override resolves to.
  format!(
    r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "mode": "affine", "{PER_LAYER_PATH}": {{ "mode": "{mode}", "group_size": {group_size}, "bits": {bits} }} }} }}"#
  )
}

/// Resolve the per-layer override's `(group_size, bits, mode)` for
/// [`PER_LAYER_PATH`] from a `config.json` body.
fn resolved_per_layer(cfg: &str) -> Quantization {
  let plq = parse_quantization(cfg)
    .expect("block parses")
    .expect("a quantization block is present");
  plq
    .quantization_for(PER_LAYER_PATH)
    .expect("the per-layer override resolves a quantization (not skipped)")
}

#[test]
fn parse_quantization_per_layer_falsy_group_size_resolves_to_per_mode_default() {
  // (mode, default_group_size, valid_bits) per the shared `defaults_for_mode`
  // table; `bits` is held at a valid positive so only `group_size` is exercised.
  // `"absent"` drops the key entirely (an omitted override key).
  for (mode, default_gs, valid_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    for falsy in ["0", "null", "absent"] {
      let cfg = if falsy == "absent" {
        format!(
          r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "mode": "affine", "{PER_LAYER_PATH}": {{ "mode": "{mode}", "bits": {valid_bits} }} }} }}"#
        )
      } else {
        quant_block_per_layer(mode, falsy, &valid_bits.to_string())
      };
      let q = resolved_per_layer(&cfg);
      assert_eq!(
        q.group_size, default_gs,
        "{mode}: per-layer falsy/absent group_size `{falsy}` must fall back to the per-mode default"
      );
      assert_eq!(q.bits, valid_bits, "{mode}: valid per-layer bits preserved");
      assert_eq!(q.mode.as_str(), mode, "{mode}: per-layer mode preserved");
    }
  }
}

#[test]
fn parse_quantization_per_layer_falsy_bits_resolves_to_per_mode_default() {
  // Same falsy/absent spellings on the per-layer `bits`; `group_size` is held at
  // a valid positive.
  for (mode, valid_gs, default_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    for falsy in ["0", "null", "absent"] {
      let cfg = if falsy == "absent" {
        format!(
          r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "mode": "affine", "{PER_LAYER_PATH}": {{ "mode": "{mode}", "group_size": {valid_gs} }} }} }}"#
        )
      } else {
        quant_block_per_layer(mode, &valid_gs.to_string(), falsy)
      };
      let q = resolved_per_layer(&cfg);
      assert_eq!(
        q.bits, default_bits,
        "{mode}: per-layer falsy/absent bits `{falsy}` must fall back to the per-mode default"
      );
      assert_eq!(
        q.group_size, valid_gs,
        "{mode}: valid per-layer group_size preserved"
      );
      assert_eq!(q.mode.as_str(), mode, "{mode}: per-layer mode preserved");
    }
  }
}

#[test]
fn parse_quantization_per_layer_positive_value_overrides() {
  // A present POSITIVE per-layer value is preserved verbatim — the per-mode
  // default only fills a falsy/absent key. Exercised across every mode with a
  // non-default group_size to prove the override wins (not the mode default).
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, "256", "4");
    let q = resolved_per_layer(&cfg);
    assert_eq!(
      q.group_size, 256,
      "{mode}: explicit positive per-layer group_size wins over the mode default"
    );
    assert_eq!(
      q.bits, 4,
      "{mode}: explicit positive per-layer bits preserved"
    );
    assert_eq!(q.mode.as_str(), mode, "{mode}: per-layer mode preserved");
  }
}

#[test]
fn parse_quantization_per_layer_unknown_mode_is_typed_error() {
  // A present-but-unrecognized `mode` on a PER-LAYER override is the same typed
  // `UnknownEnumValue` the top-level spec raises — the single normalization pass
  // covers both, so the per-layer object cannot resolve the WRONG per-mode
  // default by silently guessing affine.
  let cfg = format!(
    r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "mode": "affine", "{PER_LAYER_PATH}": {{ "mode": "gptq", "bits": 4 }} }} }}"#
  );
  let err = parse_quantization(&cfg).expect_err("per-layer unknown mode must error");
  assert!(
    matches!(err, Error::UnknownEnumValue(_)),
    "expected UnknownEnumValue for an unrecognized per-layer quantization mode, got {err:?}"
  );
}

#[test]
fn parse_quantization_per_layer_oversized_group_size_is_out_of_range() {
  // A PER-LAYER override carrying a present integer that does not fit `i32` is
  // the same typed `OutOfRange` the top-level spec raises — the single
  // normalization pass covers both, so an oversized per-layer literal cannot
  // silently collapse to the per-mode default. `bits` stays a valid positive.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, &OVERSIZED_I32.to_string(), "4");
    let err = parse_quantization(&cfg).expect_err("oversized per-layer group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for an oversized per-layer group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_oversized_bits_is_out_of_range() {
  // Same oversized literal on the per-layer `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, "64", &OVERSIZED_I32.to_string());
    let err = parse_quantization(&cfg).expect_err("oversized per-layer bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for an oversized per-layer bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_negative_group_size_is_out_of_range() {
  // A PER-LAYER negative `group_size` is the same typed `OutOfRange` the
  // top-level spec raises — the single `resolve_quant_spec` guard covers both
  // levels, so a per-layer negative cannot silently collapse to the per-mode
  // default. `bits` stays a valid positive.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, "-1", "4");
    let err = parse_quantization(&cfg).expect_err("negative per-layer group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a negative per-layer group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_negative_bits_is_out_of_range() {
  // Same negative literal on the per-layer `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, "64", "-1");
    let err = parse_quantization(&cfg).expect_err("negative per-layer bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a negative per-layer bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_u64_overflow_group_size_is_out_of_range() {
  // A PER-LAYER `group_size` magnitude past `i64::MAX` (serde_json `u64`) is the
  // same typed `OutOfRange` the top-level spec raises — the single strict path
  // covers both, so the per-layer object cannot silently collapse a `u64`-overflow
  // literal to the per-mode default. `bits` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, OVERSIZED_U64, "4");
    let err = parse_quantization(&cfg).expect_err("u64-overflow per-layer group_size must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a u64-overflow per-layer group_size, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_u64_overflow_bits_is_out_of_range() {
  // Same u64-overflow literal on the per-layer `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    let cfg = quant_block_per_layer(mode, "64", OVERSIZED_U64);
    let err = parse_quantization(&cfg).expect_err("u64-overflow per-layer bits must error");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "{mode}: expected OutOfRange for a u64-overflow per-layer bits, got {err:?}"
    );
  }
}

#[test]
fn parse_quantization_per_layer_non_integer_group_size_is_typed_error() {
  // A PER-LAYER `group_size` that is a float (`64.5`) or a string (`"64"`) is the
  // same typed `Error::Parse` the top-level spec raises — serde rejects each by
  // TYPE through the single strict path. `bits` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    for non_int in [NON_INTEGER_FLOAT, NON_INTEGER_STRING] {
      let cfg = quant_block_per_layer(mode, non_int, "4");
      let err = parse_quantization(&cfg).expect_err("non-integer per-layer group_size must error");
      assert!(
        matches!(err, Error::Parse(_)),
        "{mode}: expected Parse for a non-integer per-layer group_size `{non_int}`, got {err:?}"
      );
    }
  }
}

#[test]
fn parse_quantization_per_layer_non_integer_bits_is_typed_error() {
  // Same non-integer values on the per-layer `bits`; `group_size` stays valid.
  for mode in ["affine", "mxfp4", "nvfp4", "mxfp8"] {
    for non_int in [NON_INTEGER_FLOAT, NON_INTEGER_STRING] {
      let cfg = quant_block_per_layer(mode, "64", non_int);
      let err = parse_quantization(&cfg).expect_err("non-integer per-layer bits must error");
      assert!(
        matches!(err, Error::Parse(_)),
        "{mode}: expected Parse for a non-integer per-layer bits `{non_int}`, got {err:?}"
      );
    }
  }
}

#[test]
fn parse_quantization_per_layer_skip_sentinel_is_preserved() {
  // A per-layer `false` (the skip sentinel) is NOT a spec object, so the
  // normalization pass leaves it untouched and the deserializer maps it to
  // `Skip` — `quantization_for` returns `None` (do not quantize this layer).
  let cfg = format!(
    r#"{{ "quantization": {{ "group_size": 128, "bits": 8, "mode": "affine", "{PER_LAYER_PATH}": false }} }}"#
  );
  let plq = parse_quantization(&cfg)
    .expect("skip-sentinel block parses")
    .expect("a quantization block is present");
  assert_eq!(
    plq.quantization_for(PER_LAYER_PATH),
    None,
    "a per-layer `false` is a Skip — the layer is not quantized"
  );
  // The global default is unaffected (still the valid top-level spec).
  assert_eq!(
    plq.quantization_for("some.other.layer"),
    Some(Quantization::affine(128, 8)),
    "the global default is preserved alongside a per-layer skip"
  );
}

#[test]
fn parse_quantization_mixed_top_level_and_per_layer_falsy_both_resolve() {
  // A FALSY top-level group_size AND a FALSY per-layer group_size on the same
  // block both resolve through the one normalization pass to their respective
  // per-mode defaults — top-level mxfp8 (32/8) and per-layer nvfp4 (16/4). Only
  // the falsy spellings `0` / `null` are exercised here (a present negative is a
  // typed error, covered separately), so every field below stays a fallback.
  let cfg = format!(
    r#"{{ "quantization": {{ "mode": "mxfp8", "group_size": 0, "bits": null, "{PER_LAYER_PATH}": {{ "mode": "nvfp4", "group_size": null, "bits": 0 }} }} }}"#
  );
  let plq = parse_quantization(&cfg)
    .expect("mixed falsy block parses")
    .expect("a quantization block is present");
  // The per-layer override resolves to the nvfp4 per-mode default.
  let per_layer = plq
    .quantization_for(PER_LAYER_PATH)
    .expect("per-layer override resolves");
  assert_eq!(
    (
      per_layer.group_size,
      per_layer.bits,
      per_layer.mode.as_str()
    ),
    (16, 4, "nvfp4"),
    "per-layer falsy values resolve to the nvfp4 per-mode default"
  );
  // The global default (any other layer) resolves to the mxfp8 per-mode default.
  let global = plq
    .quantization_for("some.other.layer")
    .expect("global default resolves");
  assert_eq!(
    (global.group_size, global.bits, global.mode.as_str()),
    (32, 8, "mxfp8"),
    "top-level falsy values resolve to the mxfp8 per-mode default"
  );
}

#[test]
fn parse_quantization_both_falsy_resolve_to_per_mode_defaults() {
  // Both keys present-but-falsy together still resolve to the per-mode pair —
  // a config MLX would treat as e.g. nvfp4 (16, 4) must not fail load here.
  for (mode, default_gs, default_bits) in [
    ("affine", 64, 4),
    ("mxfp4", 32, 4),
    ("nvfp4", 16, 4),
    ("mxfp8", 32, 8),
  ] {
    let cfg = quant_block_with(mode, "0", "null");
    let (gs, bits) = resolved_group_size_bits(&cfg);
    assert_eq!((gs, bits), (default_gs, default_bits), "{mode}: both falsy");
  }
}

#[test]
fn parse_quantization_positive_group_size_bits_still_override_default() {
  // A present POSITIVE value is truthy and wins over the per-mode default — the
  // falsy-fallback must not clobber a real explicit scheme override.
  let cfg = quant_block_with("mxfp4", "128", "8");
  let (gs, bits) = resolved_group_size_bits(&cfg);
  assert_eq!(
    (gs, bits),
    (128, 8),
    "explicit positive group_size/bits override the mxfp4 mode default"
  );
}
