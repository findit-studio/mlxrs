//! Structural tests for the EmbeddingGemma sentence-encoder.
//!
//! Deterministic, tiny-fixture, non-network: a `hidden = 8`, `heads = 2`,
//! `kv_heads = 1`, `head_dim = 4`, `layers = 2`, `vocab = 10`, `intermediate =
//! 16` model is built from synthetic weights and exercised through
//! [`EmbeddingGemmaModel`]. These pin the embedding output shape + L2-norm, the
//! bidirectional-mask construction, the dynamic-right-pad text encoding, the
//! baked pooling config, the sanitize key rewrites, and a couple of
//! consumed-weight shape-mismatch rejections.
//!
//! A `google/embeddinggemma-300m` checkpoint is not bundled, so there is **no**
//! end-to-end cosine-parity oracle here (see the module-level note in the port
//! report); the unit tests pin the architecture's shape + invariants instead.

use std::collections::HashMap;

use super::*;
use crate::{array::Array, error::Error};

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 4;
const LAYERS: i32 = 2;
const VOCAB: i32 = 10;
const INTER: i32 = 16;
const DENSE_INTER: i32 = HIDDEN * 4; // 32

/// A tiny `Gemma3Config` whose dims keep the synthetic weights small. `head_dim`
/// is decoupled from `hidden/heads` exactly as the real Gemma3 config is (here
/// `8/2 = 4 = head_dim`, but the field is authoritative).
fn tiny_config() -> Gemma3Config {
  let json = format!(
    r#"{{
      "model_type": "gemma3_text",
      "vocab_size": {VOCAB},
      "hidden_size": {HIDDEN},
      "num_hidden_layers": {LAYERS},
      "intermediate_size": {INTER},
      "num_attention_heads": {HEADS},
      "head_dim": {HEAD_DIM},
      "rms_norm_eps": 1e-6,
      "num_key_value_heads": {KV_HEADS},
      "rope_theta": 1000000.0,
      "rope_local_base_freq": 10000.0,
      "query_pre_attn_scalar": 256.0,
      "sliding_window": 512,
      "sliding_window_pattern": 6,
      "max_position_embeddings": 2048
    }}"#
  );
  let cfg = Gemma3Config::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// A `(rows, cols)` f32 matrix with small deterministic values.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A `(n,)` f32 vector with small deterministic values (the RMSNorm `weight`
/// delta — small so `1.0 + delta` is near unity).
fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| ((i % 5) as f32) * 0.001).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

fn insert_attn(w: &mut HashMap<String, Array>, prefix: &str) {
  let q_out = HEADS * HEAD_DIM; // 8
  let kv_out = KV_HEADS * HEAD_DIM; // 4
  w.insert(format!("{prefix}.q_proj.weight"), mat(q_out, HIDDEN));
  w.insert(format!("{prefix}.k_proj.weight"), mat(kv_out, HIDDEN));
  w.insert(format!("{prefix}.v_proj.weight"), mat(kv_out, HIDDEN));
  w.insert(format!("{prefix}.o_proj.weight"), mat(HIDDEN, q_out));
  w.insert(format!("{prefix}.q_norm.weight"), vec1(HEAD_DIM));
  w.insert(format!("{prefix}.k_norm.weight"), vec1(HEAD_DIM));
}

fn insert_mlp(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.gate_proj.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{prefix}.up_proj.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{prefix}.down_proj.weight"), mat(HIDDEN, INTER));
}

/// Build a full, correctly-shaped weight map in the **sanitized** layout (the
/// keys `EmbeddingGemmaModel::from_weights` consumes).
fn tiny_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert("model.embed_tokens.weight".to_string(), mat(VOCAB, HIDDEN));
  for i in 0..LAYERS {
    let p = format!("model.layers.{i}");
    w.insert(format!("{p}.input_layernorm.weight"), vec1(HIDDEN));
    insert_attn(&mut w, &format!("{p}.self_attn"));
    w.insert(format!("{p}.post_attention_layernorm.weight"), vec1(HIDDEN));
    w.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
    insert_mlp(&mut w, &format!("{p}.mlp"));
    w.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
  }
  w.insert("model.norm.weight".to_string(), vec1(HIDDEN));
  // Dense head: dense.0 (intermediate, hidden), dense.1 (hidden, intermediate).
  w.insert("dense.0.weight".to_string(), mat(DENSE_INTER, HIDDEN));
  w.insert("dense.1.weight".to_string(), mat(HIDDEN, DENSE_INTER));
  w
}

/// A `(batch, seq_len)` i32 token-id batch with ids in `0..VOCAB`.
fn ids(batch: usize, seq: usize) -> Array {
  let data: Vec<i32> = (0..batch * seq)
    .map(|n| (n % VOCAB as usize) as i32)
    .collect();
  Array::from_slice::<i32>(&data, &(batch, seq)).unwrap()
}

/// A `(batch, seq_len)` f32 attention mask, all-ones (no padding).
fn full_mask(batch: usize, seq: usize) -> Array {
  let data = vec![1.0_f32; batch * seq];
  Array::from_slice::<f32>(&data, &(batch, seq)).unwrap()
}

fn eval_to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

fn build_model() -> EmbeddingGemmaModel {
  EmbeddingGemmaModel::from_weights(tiny_config(), tiny_weights(), None).unwrap()
}

#[test]
fn encode_text_output_shape_is_batch_by_hidden() {
  let model = build_model();
  let out = model
    .encode_text(&ids(3, 4), &full_mask(3, 4))
    .expect("encode");
  // (batch, hidden) — the Dense head maps hidden*4 back to hidden, so the final
  // width is `hidden` (no matryoshka truncation by default).
  assert_eq!(out.shape(), vec![3, HIDDEN as usize]);
  assert!(eval_to_vec(&out).iter().all(|x| x.is_finite()));
}

#[test]
fn encode_text_rows_are_l2_normalized() {
  let model = build_model();
  let out = model
    .encode_text(&ids(2, 5), &full_mask(2, 5))
    .expect("encode");
  let v = eval_to_vec(&out);
  let hidden = HIDDEN as usize;
  for row in 0..2 {
    let norm_sq: f32 = v[row * hidden..(row + 1) * hidden]
      .iter()
      .map(|x| x * x)
      .sum();
    assert!(
      (norm_sq.sqrt() - 1.0).abs() < 1e-4,
      "row {row} must be unit-norm, got ||v|| = {}",
      norm_sq.sqrt()
    );
  }
}

#[test]
fn embed_text_matches_encode_text() {
  let model = build_model();
  let i = ids(2, 3);
  let m = full_mask(2, 3);
  let via_trait = model.embed_text(&i, &m).expect("embed_text");
  let via_inherent = model.encode_text(&i, &m).expect("encode_text");
  assert_eq!(via_trait.array().shape(), via_inherent.shape());
  let a = eval_to_vec(via_trait.array());
  let b = eval_to_vec(&via_inherent);
  for (x, y) in a.iter().zip(b.iter()) {
    assert!((x - y).abs() < 1e-6, "embed_text must equal encode_text");
  }
}

#[test]
fn padding_changes_pooled_result_vs_unpadded() {
  // A row with a masked-out (padding) trailing token must pool to the same
  // embedding as the same row without that token — the mask is honored.
  let model = build_model();
  // Row A: 3 real tokens (ids 1,2,3), seq 3, full mask.
  let ids_a = Array::from_slice::<i32>(&[1, 2, 3], &(1usize, 3usize)).unwrap();
  let mask_a = full_mask(1, 3);
  let out_a = model.encode_text(&ids_a, &mask_a).expect("encode A");
  // Row B: same 3 tokens + 1 padding token (id 0), seq 4, mask [1,1,1,0].
  let ids_b = Array::from_slice::<i32>(&[1, 2, 3, 0], &(1usize, 4usize)).unwrap();
  let mask_b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 0.0], &(1usize, 4usize)).unwrap();
  let out_b = model.encode_text(&ids_b, &mask_b).expect("encode B");
  let a = eval_to_vec(&out_a);
  let b = eval_to_vec(&out_b);
  for (x, y) in a.iter().zip(b.iter()) {
    assert!(
      (x - y).abs() < 1e-4,
      "masked padding must not change the pooled embedding: {x} vs {y}"
    );
  }
}

#[test]
fn text_encoding_is_dynamic_right_pad_with_gemma_pad_id() {
  let model = build_model();
  let enc = model.text_encoding();
  assert!(enc.add_special_tokens, "Gemma encodes with special tokens");
  assert_eq!(enc.max_length, None, "no explicit per-text truncation cap");
  match enc.padding {
    Padding::DynamicRightPad { pad_token_id } => {
      assert_eq!(pad_token_id, 0, "Gemma <pad> id is 0");
    }
    other => panic!("expected DynamicRightPad, got {other:?}"),
  }
}

#[test]
fn baked_pooling_defaults_to_mean_normalize_full_dim() {
  let model = build_model();
  let p = model.pooling();
  assert!(matches!(p.strategy, PoolingStrategy::Mean));
  assert!(p.normalize);
  assert_eq!(p.dimension, None);
  assert!(!p.layer_norm);
  assert!(!p.rms_norm);
}

#[test]
fn baked_pooling_honors_matryoshka_dimension_from_st_config() {
  // A `1_Pooling` config with mean + a matryoshka dimension bakes that
  // truncation, and the encode output narrows to it.
  let st = StPoolingConfig::new(PoolingStrategy::Mean, true, Some(4));
  let model = EmbeddingGemmaModel::from_weights(tiny_config(), tiny_weights(), Some(&st)).unwrap();
  assert_eq!(model.pooling().dimension, Some(4));
  let out = model
    .encode_text(&ids(1, 3), &full_mask(1, 3))
    .expect("encode");
  assert_eq!(out.shape(), vec![1, 4], "matryoshka-truncated to dim 4");
  // still unit-norm after truncation.
  let v = eval_to_vec(&out);
  let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
  assert!((norm - 1.0).abs() < 1e-4);
}

#[test]
fn non_mean_st_pooling_is_rejected() {
  // EmbeddingGemma is a mean-pooling encoder; a non-mean ST config must be a
  // typed rejection (loading the wrong pooling would silently corrupt output).
  let st = StPoolingConfig::new(PoolingStrategy::Cls, true, None);
  let err =
    EmbeddingGemmaModel::from_weights(tiny_config(), tiny_weights(), Some(&st)).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "non-mean ST pooling must be InvariantViolation, got {err:?}"
  );
}

#[test]
fn missing_weight_is_a_typed_error() {
  let cfg = tiny_config();
  let mut w = tiny_weights();
  w.remove("model.norm.weight");
  let err = EmbeddingGemmaModel::from_weights(cfg, w, None).unwrap_err();
  assert!(
    matches!(err, Error::MissingKey(_)),
    "missing weight must be MissingKey, got {err:?}"
  );
}

#[test]
fn wrong_shape_weight_is_a_typed_error() {
  let cfg = tiny_config();
  let mut w = tiny_weights();
  // Wrong embed-table shape (transposed).
  w.insert("model.embed_tokens.weight".to_string(), mat(HIDDEN, VOCAB));
  let err = EmbeddingGemmaModel::from_weights(cfg, w, None).unwrap_err();
  assert!(
    matches!(err, Error::LayerKeyed(_)),
    "wrong-shape weight must be LayerKeyed(ShapePairMismatch), got {err:?}"
  );
}

#[test]
fn embed_text_rejects_non_rank2_input_ids() {
  let model = build_model();
  let bad = Array::from_slice::<i32>(&[0, 1, 2, 3, 4, 5], &(1usize, 2usize, 3usize)).unwrap();
  let err = model.embed_text(&bad, &full_mask(1, 6)).err();
  assert!(
    matches!(err, Some(Error::RankMismatch(_))),
    "rank-3 input_ids must be RankMismatch, got {err:?}"
  );
}

// ─────────────────── dtype preservation (f16 / bf16) ───────────────────
//
// EmbeddingGemma's backbone is a faithful port of the mlx-embeddings Gemma3
// encoder, whose forward keeps the activation dtype end to end: the token
// embedding is scaled by `sqrt(hidden)` built in the weight dtype then cast to
// `h.dtype` (`embedding_scale_like`), each RMSNorm folds `1.0 + weight` in the
// weight dtype, RoPE / the four sandwich norms / SDPA all run through
// dtype-preserving fused kernels, the additive padding mask is built in the
// activation dtype, and the f16 `clip_residual` saturates back to f16. So a
// bf16 / f16 checkpoint's hidden states (`last_hidden_state`) stay in-dtype —
// these tests pin that.
//
// The *pooled* `text_embeds`, however, is f32 by faithful design: the
// reference `mean_pooling` casts the attention mask to `mx.float32`
// (`mlx_embeddings/models/pooling.py`), so `token_embeddings * mask` promotes
// to f32, and the Dense projection + L2-normalize then run in f32. Mirroring
// the reference exactly (the project's faithfulness-first rule) means the final
// embedding is f32 for every checkpoint dtype — forcing it back to bf16 / f16
// would *diverge* from mlx-embeddings. These tests pin that contract too.

/// A `tiny_weights()` map with every tensor cast to `dt` — a synthetic
/// bf16 / f16 dense checkpoint.
fn tiny_weights_in_dtype(dt: crate::dtype::Dtype) -> HashMap<String, Array> {
  let mut w = tiny_weights();
  for v in w.values_mut() {
    *v = crate::ops::misc::astype(v, dt).unwrap();
  }
  w
}

/// Run a forward to the backbone's `last_hidden_state` (post final RMSNorm) for
/// a `dt`-dtype model, returning that intermediate so a test can assert it
/// stays in `dt`. Mirrors the head of `encode_text`: build the additive mask in
/// the backbone dtype, run the backbone.
fn backbone_last_hidden_state(model: &EmbeddingGemmaModel, batch: usize, seq: usize) -> Array {
  let mask = crate::embeddings::embeddinggemma::backbone::build_additive_mask(
    &full_mask(batch, seq),
    model.backbone.embed_dtype().unwrap(),
  )
  .unwrap();
  model.backbone.forward(&ids(batch, seq), &mask).unwrap()
}

/// Shared assertions for a dense `dt` checkpoint: the in-dtype intermediates
/// (embed table, additive mask, `last_hidden_state`) stay in `dt`, while the
/// final embedding is the faithful f32 (reference mean-pool mask cast) and is
/// finite + L2-normalized.
fn assert_dtype_preserved_dense(dt: crate::dtype::Dtype) {
  let model = EmbeddingGemmaModel::from_weights(tiny_config(), tiny_weights_in_dtype(dt), None)
    .expect("load dtype checkpoint");

  // Embedding table dtype is the checkpoint dtype (the mask is derived from it).
  assert_eq!(
    model.backbone.embed_dtype().unwrap(),
    dt,
    "embed table must stay {dt:?}"
  );
  // The additive padding mask is built in the activation dtype (no f32 mask
  // fed to a bf16/f16 SDPA).
  let mask =
    crate::embeddings::embeddinggemma::backbone::build_additive_mask(&full_mask(2, 4), dt).unwrap();
  assert_eq!(mask.dtype().unwrap(), dt, "additive mask must be {dt:?}");

  // The backbone last_hidden_state stays in-dtype end to end (embedding scale,
  // RoPE, every RMSNorm, the residual clips — none silently promote to f32).
  let hidden = backbone_last_hidden_state(&model, 2, 4);
  assert_eq!(
    hidden.dtype().unwrap(),
    dt,
    "backbone last_hidden_state must stay {dt:?} (no f32 promotion in the forward)"
  );

  // The final embedding is f32 — faithful to the reference's f32 mean-pool mask
  // cast — for every checkpoint dtype, and is finite + unit-norm.
  let out = model
    .encode_text(&ids(2, 4), &full_mask(2, 4))
    .expect("encode");
  assert_eq!(
    out.dtype().unwrap(),
    crate::dtype::Dtype::F32,
    "pooled embedding is f32 by faithful design (reference mean_pooling casts the mask to float32)"
  );
  assert_eq!(out.shape(), vec![2, HIDDEN as usize]);
  let v = eval_to_vec(&out);
  assert!(v.iter().all(|x| x.is_finite()), "finite, got {v:?}");
  let hidden_w = HIDDEN as usize;
  for row in 0..2 {
    let norm: f32 = v[row * hidden_w..(row + 1) * hidden_w]
      .iter()
      .map(|x| x * x)
      .sum::<f32>()
      .sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "row {row} unit-norm, got {norm}");
  }
}

#[test]
fn bf16_checkpoint_preserves_backbone_dtype() {
  assert_dtype_preserved_dense(crate::dtype::Dtype::BF16);
}

#[test]
fn f16_checkpoint_preserves_backbone_dtype() {
  assert_dtype_preserved_dense(crate::dtype::Dtype::F16);
}

// ───────────────────────── sanitize ─────────────────────────

#[test]
fn sanitize_namespaces_backbone_keys() {
  // A raw backbone key (no `model.` prefix, no dense/linear) is namespaced.
  let mut raw = HashMap::new();
  raw.insert("embed_tokens.weight".to_string(), mat(VOCAB, HIDDEN));
  raw.insert("layers.0.input_layernorm.weight".to_string(), vec1(HIDDEN));
  let out = sanitize(raw).expect("sanitize");
  assert!(out.contains_key("model.embed_tokens.weight"));
  assert!(out.contains_key("model.layers.0.input_layernorm.weight"));
}

#[test]
fn sanitize_keeps_already_namespaced_backbone_keys() {
  let mut raw = HashMap::new();
  raw.insert("model.embed_tokens.weight".to_string(), mat(VOCAB, HIDDEN));
  let out = sanitize(raw).expect("sanitize");
  assert!(out.contains_key("model.embed_tokens.weight"));
  // Not double-prefixed.
  assert!(!out.contains_key("model.model.embed_tokens.weight"));
}

#[test]
fn sanitize_renames_st_dense_modules_by_module_number() {
  // The lower module number (2_Dense) → dense.0; the next (3_Dense) → dense.1,
  // independent of tensor shape. For this dense checkpoint that coincides with
  // the reference's width test (2_Dense is the expansion, 3_Dense the
  // contraction), so the result is identical.
  let mut raw = HashMap::new();
  // 2_Dense expands hidden → hidden*4 → dense.0.
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  // 3_Dense contracts hidden*4 → hidden → dense.1.
  raw.insert(
    "3_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );
  let out = sanitize(raw).expect("sanitize");
  assert!(out.contains_key("dense.0.weight"), "2_Dense → dense.0");
  assert!(out.contains_key("dense.1.weight"), "3_Dense → dense.1");
  assert_eq!(
    out["dense.0.weight"].shape(),
    vec![DENSE_INTER as usize, HIDDEN as usize]
  );
  assert_eq!(
    out["dense.1.weight"].shape(),
    vec![HIDDEN as usize, DENSE_INTER as usize]
  );
}

#[test]
fn sanitize_is_idempotent_on_renamed_dense() {
  let mut raw = HashMap::new();
  raw.insert("dense.0.weight".to_string(), mat(DENSE_INTER, HIDDEN));
  raw.insert("dense.1.weight".to_string(), mat(HIDDEN, DENSE_INTER));
  let out = sanitize(raw).expect("sanitize");
  assert!(out.contains_key("dense.0.weight"));
  assert!(out.contains_key("dense.1.weight"));
}

#[test]
fn sanitized_raw_checkpoint_loads_end_to_end() {
  // Build a "raw" checkpoint (backbone keys unprefixed + ST Dense names), run
  // it through sanitize, and confirm it loads.
  let mut raw = HashMap::new();
  raw.insert("embed_tokens.weight".to_string(), mat(VOCAB, HIDDEN));
  for i in 0..LAYERS {
    let p = format!("layers.{i}");
    raw.insert(format!("{p}.input_layernorm.weight"), vec1(HIDDEN));
    insert_attn(&mut raw, &format!("{p}.self_attn"));
    raw.insert(format!("{p}.post_attention_layernorm.weight"), vec1(HIDDEN));
    raw.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
    insert_mlp(&mut raw, &format!("{p}.mlp"));
    raw.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
  }
  raw.insert("norm.weight".to_string(), vec1(HIDDEN));
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  raw.insert(
    "3_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );

  let weights = sanitize(raw).expect("sanitize");
  let model = EmbeddingGemmaModel::from_weights(tiny_config(), weights, None).expect("load");
  let out = model
    .encode_text(&ids(1, 3), &full_mask(1, 3))
    .expect("encode");
  assert_eq!(out.shape(), vec![1, HIDDEN as usize]);
}

// ───────────────────────── registration ─────────────────────────

#[test]
fn registers_under_gemma3_text() {
  let mut registry = EmbeddingModelTypeRegistry::new();
  register(&mut registry);
  assert!(registry.contains(MODEL_TYPE));
  assert_eq!(MODEL_TYPE, "gemma3_text");
}

#[test]
fn answers_as_text_embedder_only() {
  let model = build_model();
  let dyn_model: &dyn EmbeddingModel = &model;
  assert!(
    dyn_model.as_text_embedder().is_some(),
    "EmbeddingGemma is a text embedder"
  );
  assert!(
    dyn_model.as_contrastive().is_none(),
    "EmbeddingGemma has no contrastive capability"
  );
  assert!(dyn_model.as_late_interaction().is_none());
  assert!(
    dyn_model
      .as_any()
      .downcast_ref::<EmbeddingGemmaModel>()
      .is_some(),
    "downcast back to the concrete model"
  );
}

// ───────────────────────── quantized load ─────────────────────────
//
// A small synthetic 8-bit affine-quantized EmbeddingGemma checkpoint, built by
// running the dense weights through the real `ops::quantized::quantize` op
// (exactly how an mlx-community quantized bundle stores a quantized
// `nn.Linear` / `nn.Embedding`), then asserting it loads through the model's
// `from_weights` and forwards to the right shape / finite, L2-normalized
// output. Mirrors the Whisper quantized-load tests.
//
// The dims are chosen so every quantized weight's last axis (the `in`
// dimension: 32, 64, or 128) is a whole number of affine groups
// (`group_size = 32`), which mlx's affine `quantize` requires.

/// Affine group size for the synthetic quantized checkpoint (divides every
/// quantized weight's last axis: 32, 64, 128).
const QGROUP: i32 = 32;
/// Bit depth for the synthetic quantized checkpoint.
const QBITS: i32 = 8;

const Q_HIDDEN: i32 = 32;
const Q_HEADS: i32 = 2;
const Q_KV_HEADS: i32 = 1;
const Q_HEAD_DIM: i32 = 16; // Q_HEADS * Q_HEAD_DIM = 32 = Q_HIDDEN
const Q_LAYERS: i32 = 1;
const Q_VOCAB: i32 = 64;
const Q_INTER: i32 = 64; // multiple of QGROUP
const Q_DENSE_INTER: i32 = Q_HIDDEN * 4; // 128

/// A tiny `Gemma3Config` (with a `quantization` block) whose every quantized
/// weight's last axis is a multiple of `QGROUP`.
fn quant_config() -> Gemma3Config {
  let json = format!(
    r#"{{
      "model_type": "gemma3_text",
      "vocab_size": {Q_VOCAB},
      "hidden_size": {Q_HIDDEN},
      "num_hidden_layers": {Q_LAYERS},
      "intermediate_size": {Q_INTER},
      "num_attention_heads": {Q_HEADS},
      "head_dim": {Q_HEAD_DIM},
      "rms_norm_eps": 1e-6,
      "num_key_value_heads": {Q_KV_HEADS},
      "rope_theta": 1000000.0,
      "rope_local_base_freq": 10000.0,
      "query_pre_attn_scalar": 256.0,
      "sliding_window": 512,
      "sliding_window_pattern": 6,
      "max_position_embeddings": 2048,
      "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS} }}
    }}"#
  );
  let cfg = Gemma3Config::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Replace the dense `<prefix>.weight` in `w` with the real
/// `ops::quantized::quantize` affine triple (`<prefix>.weight` packed +
/// `<prefix>.scales` + `<prefix>.biases`), mirroring how an mlx-community
/// quantized checkpoint stores a quantized `nn.Linear` / `nn.Embedding`.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
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

/// Build the dense `quant_config()`-sized weight map (sanitized layout), then
/// quantize every `nn.Linear` projection, the Dense head, AND the token
/// embedding to the 8-bit affine triple — mirroring an mlx-embeddings quantized
/// checkpoint (the `class_predicate` quantizes every `nn.Linear` /
/// `nn.Embedding`). The RMSNorm weights stay dense (not quantizable).
fn quant_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(
    "model.embed_tokens.weight".to_string(),
    mat(Q_VOCAB, Q_HIDDEN),
  );
  for i in 0..Q_LAYERS {
    let p = format!("model.layers.{i}");
    w.insert(format!("{p}.input_layernorm.weight"), vec1(Q_HIDDEN));
    // attention projections.
    let q_out = Q_HEADS * Q_HEAD_DIM; // 32
    let kv_out = Q_KV_HEADS * Q_HEAD_DIM; // 16
    w.insert(format!("{p}.self_attn.q_proj.weight"), mat(q_out, Q_HIDDEN));
    w.insert(
      format!("{p}.self_attn.k_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(
      format!("{p}.self_attn.v_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(format!("{p}.self_attn.o_proj.weight"), mat(Q_HIDDEN, q_out));
    w.insert(format!("{p}.self_attn.q_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(format!("{p}.self_attn.k_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(
      format!("{p}.post_attention_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    w.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    // MLP projections.
    w.insert(format!("{p}.mlp.gate_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.up_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.down_proj.weight"), mat(Q_HIDDEN, Q_INTER));
    w.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
  }
  w.insert("model.norm.weight".to_string(), vec1(Q_HIDDEN));
  w.insert("dense.0.weight".to_string(), mat(Q_DENSE_INTER, Q_HIDDEN));
  w.insert("dense.1.weight".to_string(), mat(Q_HIDDEN, Q_DENSE_INTER));

  // Now quantize every nn.Linear projection, the Dense head, and the embedding.
  for i in 0..Q_LAYERS {
    let p = format!("model.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.self_attn.{proj}"));
    }
    for proj in ["gate_proj", "up_proj", "down_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.mlp.{proj}"));
    }
  }
  quantize_weight_in_place(&mut w, "dense.0");
  quantize_weight_in_place(&mut w, "dense.1");
  quantize_weight_in_place(&mut w, "model.embed_tokens");
  w
}

fn q_ids(batch: usize, seq: usize) -> Array {
  let data: Vec<i32> = (0..batch * seq)
    .map(|n| (n % Q_VOCAB as usize) as i32)
    .collect();
  Array::from_slice::<i32>(&data, &(batch, seq)).unwrap()
}

#[test]
fn quantized_checkpoint_loads_and_builds_quantized_layers() {
  // With a quantization config and a checkpoint whose Linear/Embedding weights
  // carry `.scales`/`.biases`, the model builds quantized layers throughout.
  // (A packed `uint32` weight of a DIFFERENT shape than the dense `(out, in)`
  // would otherwise be rejected by the dense shape gate, so a successful load
  // is itself proof the quantized path ran — but assert the introspection too.)
  let model =
    EmbeddingGemmaModel::from_weights(quant_config(), quant_weights(), None).expect("load");
  assert!(
    model.embedding_is_quantized(),
    "the token embedding must load quantized"
  );
  assert!(
    model.all_projections_quantized(),
    "every attention + MLP projection must load quantized"
  );
  assert!(
    model.dense_head_is_quantized(),
    "both Dense-head layers must load quantized"
  );
}

#[test]
fn quantized_checkpoint_forwards_to_finite_l2_normalized_output() {
  // The real GOAL contract on a synthetic stand-in: an 8-bit checkpoint loads
  // AND runs a full forward (the quantized attention/MLP/Dense `quantized_matmul`
  // and the quantized token-embedding gather + dequantize all execute through
  // mlx-c) to a finite, L2-normalized embedding of the right shape.
  let model =
    EmbeddingGemmaModel::from_weights(quant_config(), quant_weights(), None).expect("load");
  let out = model
    .encode_text(&q_ids(2, 5), &full_mask(2, 5))
    .expect("quantized encode");
  assert_eq!(out.shape(), vec![2, Q_HIDDEN as usize]);
  let v = eval_to_vec(&out);
  assert!(
    v.iter().all(|x| x.is_finite()),
    "quantized encode must be finite, got {v:?}"
  );
  let hidden = Q_HIDDEN as usize;
  for row in 0..2 {
    let norm: f32 = v[row * hidden..(row + 1) * hidden]
      .iter()
      .map(|x| x * x)
      .sum::<f32>()
      .sqrt();
    assert!(
      (norm - 1.0).abs() < 1e-4,
      "quantized row {row} must be unit-norm, got {norm}"
    );
  }
}

#[test]
fn quantized_padding_is_masked_like_dense() {
  // The mask is honored on the quantized path too: a masked-out trailing token
  // must not change the pooled embedding.
  let model =
    EmbeddingGemmaModel::from_weights(quant_config(), quant_weights(), None).expect("load");
  let ids_a = Array::from_slice::<i32>(&[1, 2, 3], &(1usize, 3usize)).unwrap();
  let out_a = model
    .encode_text(&ids_a, &full_mask(1, 3))
    .expect("encode A");
  let ids_b = Array::from_slice::<i32>(&[1, 2, 3, 0], &(1usize, 4usize)).unwrap();
  let mask_b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 0.0], &(1usize, 4usize)).unwrap();
  let out_b = model.encode_text(&ids_b, &mask_b).expect("encode B");
  let a = eval_to_vec(&out_a);
  let b = eval_to_vec(&out_b);
  for (x, y) in a.iter().zip(b.iter()) {
    assert!(
      (x - y).abs() < 1e-4,
      "masked padding must not change the quantized embedding: {x} vs {y}"
    );
  }
}

// ─────────────── quantized-path dtype derivation (f16 / bf16) ───────────────
//
// On the quantized token-embedding path the dequantize-gather output dtype — and
// everything derived from it (the additive mask dtype, the post-embedding
// activation dtype) — must match what the DENSE path produces for the same
// checkpoint dtype. The model reads it from `embed_dtype()`, which on the
// quantized path returns the `scales` dtype (the activation dtype the checkpoint
// was quantized from, and the dtype `affine_dequantize` reconstructs to) — NOT a
// hard-coded f32 nor the dequantize default. These tests quantize a bf16 / f16
// dense checkpoint (so the scales carry that dtype) and pin that the quantized
// embedding dtype + the in-dtype backbone match the dense contract.

/// The dense `quant_config()`-sized weight map (sanitized layout, pre-quantize)
/// with every tensor cast to `dt`.
fn quant_dense_weights_in_dtype(dt: crate::dtype::Dtype) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(
    "model.embed_tokens.weight".to_string(),
    mat(Q_VOCAB, Q_HIDDEN),
  );
  for i in 0..Q_LAYERS {
    let p = format!("model.layers.{i}");
    w.insert(format!("{p}.input_layernorm.weight"), vec1(Q_HIDDEN));
    let q_out = Q_HEADS * Q_HEAD_DIM;
    let kv_out = Q_KV_HEADS * Q_HEAD_DIM;
    w.insert(format!("{p}.self_attn.q_proj.weight"), mat(q_out, Q_HIDDEN));
    w.insert(
      format!("{p}.self_attn.k_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(
      format!("{p}.self_attn.v_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(format!("{p}.self_attn.o_proj.weight"), mat(Q_HIDDEN, q_out));
    w.insert(format!("{p}.self_attn.q_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(format!("{p}.self_attn.k_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(
      format!("{p}.post_attention_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    w.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    w.insert(format!("{p}.mlp.gate_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.up_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.down_proj.weight"), mat(Q_HIDDEN, Q_INTER));
    w.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
  }
  w.insert("model.norm.weight".to_string(), vec1(Q_HIDDEN));
  w.insert("dense.0.weight".to_string(), mat(Q_DENSE_INTER, Q_HIDDEN));
  w.insert("dense.1.weight".to_string(), mat(Q_HIDDEN, Q_DENSE_INTER));
  for v in w.values_mut() {
    *v = crate::ops::misc::astype(v, dt).unwrap();
  }
  w
}

/// A quantized checkpoint whose pre-quantize activations are `dt`: build the
/// dense map in `dt`, then run every `nn.Linear`, the Dense head, and the token
/// embedding through the real affine `quantize` (so the resulting `.scales`
/// carry `dt`).
fn quant_weights_in_dtype(dt: crate::dtype::Dtype) -> HashMap<String, Array> {
  let mut w = quant_dense_weights_in_dtype(dt);
  for i in 0..Q_LAYERS {
    let p = format!("model.layers.{i}");
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.self_attn.{proj}"));
    }
    for proj in ["gate_proj", "up_proj", "down_proj"] {
      quantize_weight_in_place(&mut w, &format!("{p}.mlp.{proj}"));
    }
  }
  quantize_weight_in_place(&mut w, "dense.0");
  quantize_weight_in_place(&mut w, "dense.1");
  quantize_weight_in_place(&mut w, "model.embed_tokens");
  w
}

/// Shared assertions for a `dt`-activation quantized checkpoint: the quantized
/// embedding dtype (and the mask derived from it) is `dt` — derived from the
/// model's activation dtype, NOT hard-f32 — and the backbone last_hidden_state
/// stays in `dt`, exactly like the dense path; the final embedding is the
/// faithful f32 and is finite + L2-normalized.
fn assert_dtype_preserved_quant(dt: crate::dtype::Dtype) {
  let model = EmbeddingGemmaModel::from_weights(quant_config(), quant_weights_in_dtype(dt), None)
    .expect("load quantized dtype checkpoint");
  assert!(
    model.embedding_is_quantized(),
    "the token embedding must load quantized"
  );
  // The quantized dequantize-gather output dtype is the activation dtype the
  // checkpoint was quantized from (the `scales` dtype), matching the dense path
  // — NOT a hard-coded f32.
  assert_eq!(
    model.backbone.embed_dtype().unwrap(),
    dt,
    "quantized embed dtype must be the activation dtype {dt:?}, not f32"
  );
  let mask = crate::embeddings::embeddinggemma::backbone::build_additive_mask(
    &full_mask(2, 5),
    model.backbone.embed_dtype().unwrap(),
  )
  .unwrap();
  assert_eq!(
    mask.dtype().unwrap(),
    dt,
    "quantized-path additive mask must be {dt:?}"
  );
  // The backbone last_hidden_state stays in-dtype on the quantized path too.
  let hidden = model.backbone.forward(&q_ids(2, 5), &mask).unwrap();
  assert_eq!(
    hidden.dtype().unwrap(),
    dt,
    "quantized backbone last_hidden_state must stay {dt:?}"
  );
  // Final embedding: the faithful f32, finite + unit-norm.
  let out = model
    .encode_text(&q_ids(2, 5), &full_mask(2, 5))
    .expect("quantized encode");
  assert_eq!(
    out.dtype().unwrap(),
    crate::dtype::Dtype::F32,
    "quantized pooled embedding is f32 by faithful design (reference mean_pooling mask cast)"
  );
  assert_eq!(out.shape(), vec![2, Q_HIDDEN as usize]);
  let v = eval_to_vec(&out);
  assert!(v.iter().all(|x| x.is_finite()), "finite, got {v:?}");
  let hidden_w = Q_HIDDEN as usize;
  for row in 0..2 {
    let norm: f32 = v[row * hidden_w..(row + 1) * hidden_w]
      .iter()
      .map(|x| x * x)
      .sum::<f32>()
      .sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "row {row} unit-norm, got {norm}");
  }
}

#[test]
fn quantized_bf16_checkpoint_derives_dtype_from_activations() {
  assert_dtype_preserved_quant(crate::dtype::Dtype::BF16);
}

#[test]
fn quantized_f16_checkpoint_derives_dtype_from_activations() {
  assert_dtype_preserved_quant(crate::dtype::Dtype::F16);
}

#[test]
fn dense_checkpoint_with_quant_config_loads_dense() {
  // A NON-quantized checkpoint loads dense even when a quantization config is
  // threaded: the `.scales` sibling is the load-bearing signal, and a dense
  // checkpoint has none, so the dense path runs regardless (the auto-detect
  // contract). Build the dense `quant_config()`-sized weights WITHOUT quantizing.
  let mut w = HashMap::new();
  w.insert(
    "model.embed_tokens.weight".to_string(),
    mat(Q_VOCAB, Q_HIDDEN),
  );
  for i in 0..Q_LAYERS {
    let p = format!("model.layers.{i}");
    w.insert(format!("{p}.input_layernorm.weight"), vec1(Q_HIDDEN));
    let q_out = Q_HEADS * Q_HEAD_DIM;
    let kv_out = Q_KV_HEADS * Q_HEAD_DIM;
    w.insert(format!("{p}.self_attn.q_proj.weight"), mat(q_out, Q_HIDDEN));
    w.insert(
      format!("{p}.self_attn.k_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(
      format!("{p}.self_attn.v_proj.weight"),
      mat(kv_out, Q_HIDDEN),
    );
    w.insert(format!("{p}.self_attn.o_proj.weight"), mat(Q_HIDDEN, q_out));
    w.insert(format!("{p}.self_attn.q_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(format!("{p}.self_attn.k_norm.weight"), vec1(Q_HEAD_DIM));
    w.insert(
      format!("{p}.post_attention_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    w.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
    w.insert(format!("{p}.mlp.gate_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.up_proj.weight"), mat(Q_INTER, Q_HIDDEN));
    w.insert(format!("{p}.mlp.down_proj.weight"), mat(Q_HIDDEN, Q_INTER));
    w.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(Q_HIDDEN),
    );
  }
  w.insert("model.norm.weight".to_string(), vec1(Q_HIDDEN));
  w.insert("dense.0.weight".to_string(), mat(Q_DENSE_INTER, Q_HIDDEN));
  w.insert("dense.1.weight".to_string(), mat(Q_HIDDEN, Q_DENSE_INTER));

  let model = EmbeddingGemmaModel::from_weights(quant_config(), w, None).expect("dense load");
  assert!(
    !model.embedding_is_quantized(),
    "a dense checkpoint must load the embedding dense even with a quant config"
  );
  assert!(
    !model.all_projections_quantized(),
    "a dense checkpoint must load projections dense (no `.scales` siblings)"
  );
  // And it still forwards to the right shape.
  let out = model
    .encode_text(&q_ids(1, 3), &full_mask(1, 3))
    .expect("encode");
  assert_eq!(out.shape(), vec![1, Q_HIDDEN as usize]);
}

#[test]
fn quantized_scales_without_resolvable_scheme_is_typed_error() {
  // A `.scales` sibling present but the config resolving NO scheme parameters
  // for that layer (here: a per-layer `false` Skip on a layer whose checkpoint
  // is nonetheless quantized) is a config/checkpoint inconsistency — a typed
  // InvariantViolation, never a guessed scheme.
  let json = format!(
    r#"{{
      "model_type": "gemma3_text",
      "vocab_size": {Q_VOCAB},
      "hidden_size": {Q_HIDDEN},
      "num_hidden_layers": {Q_LAYERS},
      "intermediate_size": {Q_INTER},
      "num_attention_heads": {Q_HEADS},
      "head_dim": {Q_HEAD_DIM},
      "rms_norm_eps": 1e-6,
      "num_key_value_heads": {Q_KV_HEADS},
      "rope_theta": 1000000.0,
      "rope_local_base_freq": 10000.0,
      "query_pre_attn_scalar": 256.0,
      "sliding_window": 512,
      "sliding_window_pattern": 6,
      "max_position_embeddings": 2048,
      "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS},
                         "model.embed_tokens": false }}
    }}"#
  );
  let cfg = Gemma3Config::from_json(&json).expect("parse");
  let err = EmbeddingGemmaModel::from_weights(cfg, quant_weights(), None).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "a `.scales` sibling with a Skip override must be InvariantViolation, got {err:?}"
  );
}

/// Build the quantized checkpoint in the **raw** (un-sanitized) layout the real
/// loader path sees: backbone keys without the `model.` prefix, and the Dense
/// head as the SentenceTransformers `{n}_Dense.linear.<suffix>` modules rather
/// than the already-renamed `dense.{0,1}.<suffix>`.
///
/// Derived from [`quant_weights`] (so every quantized triple is the real
/// `ops::quantized::quantize` output) by reversing the canonical→raw mapping:
/// `dense.0.<suffix>` → `2_Dense.linear.<suffix>`, `dense.1.<suffix>` →
/// `3_Dense.linear.<suffix>`, and `model.<rest>` → `<rest>`. This preserves all
/// three siblings (`.weight`, `.scales`, `.biases`) of each Dense module, whose
/// packed `.weight` and `(out, in/group_size)` `.scales`/`.biases` shapes are
/// what a shape-based classifier would misroute.
fn raw_quant_weights() -> HashMap<String, Array> {
  let sanitized = quant_weights();
  let mut raw = HashMap::new();
  for (k, v) in sanitized {
    let new_key = if let Some(suffix) = k.strip_prefix("dense.0") {
      format!("2_Dense.linear{suffix}")
    } else if let Some(suffix) = k.strip_prefix("dense.1") {
      format!("3_Dense.linear{suffix}")
    } else if let Some(rest) = k.strip_prefix("model.") {
      rest.to_string()
    } else {
      k
    };
    raw.insert(new_key, v);
  }
  raw
}

#[test]
fn sanitize_routes_raw_quantized_dense_module_triples_by_number() {
  // The real loader path sees the UN-sanitized `{n}_Dense.linear.{weight,scales,
  // biases}` quantized triples. Classifying by tensor shape would misroute the
  // 3_Dense contraction's `.scales`/`.biases` — which are `(out, in/group_size)`,
  // so `shape[0] > shape[1]` is true — onto `dense.0`, colliding with the 2_Dense
  // siblings (KeyCollision). Classifying by module number routes every sibling of
  // 2_Dense → dense.0 and every sibling of 3_Dense → dense.1.
  let raw = raw_quant_weights();
  // Sanity: the raw map really uses the {n}_Dense.linear names, not dense.{id}.
  assert!(raw.contains_key("2_Dense.linear.scales"));
  assert!(raw.contains_key("3_Dense.linear.scales"));
  assert!(!raw.keys().any(|k| k.starts_with("dense.")));

  let out = sanitize(raw).expect("sanitize must not KeyCollision on raw quantized Dense triples");

  // (a) every sibling routed to the module-number slot, no collision.
  for suffix in [".weight", ".scales", ".biases"] {
    assert!(
      out.contains_key(&format!("dense.0{suffix}")),
      "2_Dense{suffix} → dense.0{suffix}"
    );
    assert!(
      out.contains_key(&format!("dense.1{suffix}")),
      "3_Dense{suffix} → dense.1{suffix}"
    );
  }
  // The 3_Dense contraction's scales are `(out=Q_HIDDEN, in/group_size)` — the
  // shape a width test would have misrouted to dense.0. Pin it at dense.1.
  assert_eq!(
    out["dense.1.scales"].shape(),
    vec![Q_HIDDEN as usize, (Q_DENSE_INTER / QGROUP) as usize],
    "3_Dense contraction scales must land at dense.1, not dense.0"
  );

  // (b) the full model loads via the quantized path from this raw map and
  // forwards to a finite, L2-normalized output.
  let model = EmbeddingGemmaModel::from_weights(quant_config(), out, None)
    .expect("load from sanitized raw quantized map");
  assert!(
    model.dense_head_is_quantized(),
    "Dense head loaded quantized"
  );
  let emb = model
    .encode_text(&q_ids(2, 5), &full_mask(2, 5))
    .expect("encode");
  assert_eq!(emb.shape(), vec![2, Q_HIDDEN as usize]);
  let v = eval_to_vec(&emb);
  assert!(v.iter().all(|x| x.is_finite()), "finite, got {v:?}");
  let hidden = Q_HIDDEN as usize;
  for row in 0..2 {
    let norm: f32 = v[row * hidden..(row + 1) * hidden]
      .iter()
      .map(|x| x * x)
      .sum::<f32>()
      .sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "row {row} unit-norm, got {norm}");
  }
}

#[test]
fn sanitize_rejects_more_than_two_distinct_dense_modules() {
  // EmbeddingGemma has exactly 2_Dense and 3_Dense. A third distinct ST Dense
  // module number is an unexpected layout → typed OutOfRange, never a guessed id.
  let mut raw = HashMap::new();
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  raw.insert(
    "3_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );
  raw.insert("4_Dense.linear.weight".to_string(), mat(HIDDEN, HIDDEN));
  let err = sanitize(raw).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "a third distinct Dense module must be OutOfRange, got {err:?}"
  );
}

#[test]
fn sanitize_rejects_one_two_dense_module_set() {
  // The distinct ST Dense module-number set must be exactly {2, 3}. A `{1, 2}`
  // set — `1_Dense` / `2_Dense`, otherwise-valid shapes — is an unexpected
  // layout: mapping it by ascending order would bind `1_Dense` to `dense.0` and
  // `2_Dense` to `dense.1`, silently loading the wrong ST modules. Rejected.
  let mut raw = HashMap::new();
  // Shapes that would otherwise pass the dense gate (expansion then contraction).
  raw.insert(
    "1_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );
  let err = sanitize(raw).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "a {{1, 2}} Dense module set must be OutOfRange (not {{2, 3}}), got {err:?}"
  );
}

#[test]
fn sanitize_rejects_two_four_dense_module_set() {
  // A `{2, 4}` set (a gap at 3) is likewise not the EmbeddingGemma layout: only
  // the exact `{2, 3}` set maps unambiguously. Ascending order would bind
  // `4_Dense` to `dense.1`, the wrong module. Rejected.
  let mut raw = HashMap::new();
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  raw.insert(
    "4_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );
  let err = sanitize(raw).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "a {{2, 4}} Dense module set must be OutOfRange (not {{2, 3}}), got {err:?}"
  );
}

#[test]
fn sanitize_rejects_single_dense_module() {
  // A lone `{2}` (only `2_Dense`, no `3_Dense`) is an incomplete head — the
  // exact-set check rejects it rather than binding a single module to dense.0
  // and leaving dense.1 to fail downstream as a MissingKey.
  let mut raw = HashMap::new();
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  let err = sanitize(raw).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "a lone {{2}} Dense module must be OutOfRange (not the exact {{2, 3}} set), got {err:?}"
  );
}

#[test]
fn sanitize_accepts_exact_two_three_dense_set_and_loads() {
  // The valid `{2, 3}` set still sanitizes and loads end to end (the exact-set
  // gate must not regress the real layout). Build a raw checkpoint with the
  // 2_Dense / 3_Dense head and confirm a full forward.
  let mut raw = HashMap::new();
  raw.insert("embed_tokens.weight".to_string(), mat(VOCAB, HIDDEN));
  for i in 0..LAYERS {
    let p = format!("layers.{i}");
    raw.insert(format!("{p}.input_layernorm.weight"), vec1(HIDDEN));
    insert_attn(&mut raw, &format!("{p}.self_attn"));
    raw.insert(format!("{p}.post_attention_layernorm.weight"), vec1(HIDDEN));
    raw.insert(
      format!("{p}.pre_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
    insert_mlp(&mut raw, &format!("{p}.mlp"));
    raw.insert(
      format!("{p}.post_feedforward_layernorm.weight"),
      vec1(HIDDEN),
    );
  }
  raw.insert("norm.weight".to_string(), vec1(HIDDEN));
  raw.insert(
    "2_Dense.linear.weight".to_string(),
    mat(DENSE_INTER, HIDDEN),
  );
  raw.insert(
    "3_Dense.linear.weight".to_string(),
    mat(HIDDEN, DENSE_INTER),
  );
  let weights = sanitize(raw).expect("the exact {2, 3} layout must sanitize");
  assert!(weights.contains_key("dense.0.weight"), "2_Dense → dense.0");
  assert!(weights.contains_key("dense.1.weight"), "3_Dense → dense.1");
  let model = EmbeddingGemmaModel::from_weights(tiny_config(), weights, None).expect("load");
  let out = model
    .encode_text(&ids(1, 3), &full_mask(1, 3))
    .expect("encode");
  assert_eq!(out.shape(), vec![1, HIDDEN as usize]);
}
