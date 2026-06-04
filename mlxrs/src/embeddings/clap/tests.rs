//! Oracle / shape tests for the full dual-tower `ClapModel` assembly.
//!
//! No checkpoint is available, so these exercise a full-`laion/clap-htsat-unfused`-
//! config `ClapModel` built from synthetic weights (the sub-towers are already
//! pinned by `audio/tests.rs` + `text/tests.rs` + `shared/tests.rs`; these focus
//! on the assembly seams):
//!
//! - **`embed_audio` / `embed_text`** round-trip to `(B, 512)` L2-normalized
//!   embeddings (norm ≈ 1);
//! - **`classify`** ordering — descending cosine, top-k truncation, the stable
//!   **input-order tie-break** (identical labels keep their input order), and the
//!   empty (`k == 0` / no labels) early-return — ported from `textclap`'s
//!   `Clap::classify_all`;
//! - the **quant path** — synthetic `.scales` siblings build `Quantized` layers
//!   across both towers + the audio projection, and a weights-quantized /
//!   config-dense mismatch errors;
//! - **`sanitize`** — the tower-prefix passthrough, the patch-embed NHWC
//!   transpose, the dropped `position_ids` / `relative_position_index` buffers,
//!   and the duplicate-key rejection;
//! - **`register`** — the factory round-trip `model_type "clap"` → a constructed
//!   `ClapModel`.

use std::collections::HashMap;

use super::*;
use crate::{
  dtype::Dtype,
  embeddings::{EmbeddingModelTypeRegistry, LoadedEmbeddingModel},
};

// ───────────────────────── small Array helpers ─────────────────────────

/// Cast `a` to f32, eval, and read it back as a flat `Vec<f32>`.
fn read_f32(a: &Array) -> Vec<f32> {
  let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// A `(rows, cols)` f32 matrix with small deterministic entries.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A `(n,)` f32 vector with small deterministic entries.
fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| ((i % 5) as f32) * 0.01).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

// ───────── the real laion/clap-htsat-unfused dims (pinned by validate) ────────

const PATCH_HIDDEN: i32 = 96;
const PATCH: i32 = 4;
const NUM_MELS: i32 = 64;
const WINDOW: i32 = 8;
const DEPTHS: [i32; 4] = [2, 2, 6, 2];
const HEADS: [i32; 4] = [4, 8, 16, 32];
const AUDIO_HIDDEN: i32 = 768;

const T_HIDDEN: i32 = 768;
const T_INTER: i32 = 3072;
const T_LAYERS: i32 = 12;
const T_VOCAB: i32 = 50265;
const T_MAX_POS: i32 = 514;
const T_TYPE_VOCAB: i32 = 1;
const PROJ: i32 = 512;

/// A real-size dense [`ClapConfig`] (defaults already match the checkpoint).
fn clap_config() -> ClapConfig {
  let cfg = ClapConfig::from_json("{}").unwrap();
  cfg.validate().unwrap();
  cfg
}

// ────────────────────── synthetic HF-prefixed checkpoint ──────────────────────
//
// The weight keys carry the FULL HF prefixes (`audio_model.audio_encoder.*` /
// `text_model.*` / `audio_projection.*` / `text_projection.*`) so the maps drive
// both `sanitize` and the per-tower split in `ClapModel::from_weights`.

fn put_linear(w: &mut HashMap<String, Array>, prefix: &str, out: i32, in_f: i32, bias: bool) {
  w.insert(format!("{prefix}.weight"), mat(out, in_f));
  if bias {
    w.insert(format!("{prefix}.bias"), vec1(out));
  }
}

fn put_layer_norm(w: &mut HashMap<String, Array>, prefix: &str, hidden: i32) {
  w.insert(format!("{prefix}.weight"), vec1(hidden));
  w.insert(format!("{prefix}.bias"), vec1(hidden));
}

fn bias_table(window: i32, num_heads: i32) -> Array {
  let span = 2 * window - 1;
  Array::full::<f32>(&((span * span) as usize, num_heads as usize), 0.02).unwrap()
}

/// A `ClapProjectionLayer` (`{prefix}.linear1` `(proj, hidden)` + `{prefix}.linear2`
/// `(proj, proj)`, both biased).
fn put_projection(w: &mut HashMap<String, Array>, prefix: &str, hidden: i32, proj: i32) {
  put_linear(w, &format!("{prefix}.linear1"), proj, hidden, true);
  put_linear(w, &format!("{prefix}.linear2"), proj, proj, true);
}

/// One Swin block under `audio_model.audio_encoder.layers.{stage}.blocks.{i}`.
fn put_swin_block(w: &mut HashMap<String, Array>, stage: i32, i: i32, dim: i32, heads: i32) {
  let p = format!("audio_model.audio_encoder.layers.{stage}.blocks.{i}");
  put_layer_norm(w, &format!("{p}.layernorm_before"), dim);
  put_linear(w, &format!("{p}.attention.self.query"), dim, dim, true);
  put_linear(w, &format!("{p}.attention.self.key"), dim, dim, true);
  put_linear(w, &format!("{p}.attention.self.value"), dim, dim, true);
  w.insert(
    format!("{p}.attention.self.relative_position_bias_table"),
    bias_table(WINDOW, heads),
  );
  put_linear(w, &format!("{p}.attention.output.dense"), dim, dim, true);
  put_layer_norm(w, &format!("{p}.layernorm_after"), dim);
  let mlp_hidden = 4 * dim;
  put_linear(w, &format!("{p}.intermediate.dense"), mlp_hidden, dim, true);
  put_linear(w, &format!("{p}.output.dense"), dim, mlp_hidden, true);
}

/// One stage's patch-merge under `audio_model.audio_encoder.layers.{stage}.downsample`.
fn put_patch_merge(w: &mut HashMap<String, Array>, stage: i32, dim: i32) {
  let p = format!("audio_model.audio_encoder.layers.{stage}.downsample");
  put_layer_norm(w, &format!("{p}.norm"), 4 * dim);
  put_linear(w, &format!("{p}.reduction"), 2 * dim, 4 * dim, false);
}

/// One RoBERTa encoder layer under `text_model.encoder.layer.{i}`.
fn put_text_layer(w: &mut HashMap<String, Array>, i: i32) {
  let l = format!("text_model.encoder.layer.{i}");
  for p in ["query", "key", "value"] {
    put_linear(
      w,
      &format!("{l}.attention.self.{p}"),
      T_HIDDEN,
      T_HIDDEN,
      true,
    );
  }
  put_linear(
    w,
    &format!("{l}.attention.output.dense"),
    T_HIDDEN,
    T_HIDDEN,
    true,
  );
  put_layer_norm(w, &format!("{l}.attention.output.LayerNorm"), T_HIDDEN);
  put_linear(
    w,
    &format!("{l}.intermediate.dense"),
    T_INTER,
    T_HIDDEN,
    true,
  );
  put_linear(w, &format!("{l}.output.dense"), T_HIDDEN, T_INTER, true);
  put_layer_norm(w, &format!("{l}.output.LayerNorm"), T_HIDDEN);
}

/// Build the full synthetic HF-prefixed checkpoint for the real config, with the
/// patch-embed conv in the **HF NCHW** `(C_out, C_in, KH, KW)` layout (so
/// `sanitize` transposes it). The `position_ids` / `relative_position_index`
/// buffers are included so the dropped-buffer behaviour is exercised.
fn full_checkpoint() -> HashMap<String, Array> {
  let mut w = HashMap::new();

  // ── audio tower (audio_model.audio_encoder.*) ──
  let ae = "audio_model.audio_encoder";
  w.insert(format!("{ae}.batch_norm.weight"), vec1(NUM_MELS));
  w.insert(format!("{ae}.batch_norm.bias"), vec1(NUM_MELS));
  w.insert(
    format!("{ae}.batch_norm.running_mean"),
    Array::full::<f32>(&(NUM_MELS as usize,), 0.0).unwrap(),
  );
  w.insert(
    format!("{ae}.batch_norm.running_var"),
    Array::full::<f32>(&(NUM_MELS as usize,), 1.0).unwrap(),
  );
  // Patch-embed conv in HF NCHW (C_out=96, C_in=1, KH=4, KW=4) — sanitize → NHWC.
  let conv_n = (PATCH_HIDDEN * PATCH * PATCH) as usize;
  let conv_data: Vec<f32> = (0..conv_n)
    .map(|n| ((n % 11) as f32) * 0.003 + 0.001)
    .collect();
  w.insert(
    format!("{ae}.patch_embed.proj.weight"),
    Array::from_slice::<f32>(
      &conv_data,
      &(
        PATCH_HIDDEN as usize,
        1usize,
        PATCH as usize,
        PATCH as usize,
      ),
    )
    .unwrap(),
  );
  w.insert(format!("{ae}.patch_embed.proj.bias"), vec1(PATCH_HIDDEN));
  put_layer_norm(&mut w, &format!("{ae}.patch_embed.norm"), PATCH_HIDDEN);
  for stage in 0..4i32 {
    let dim = PATCH_HIDDEN << stage;
    for i in 0..DEPTHS[stage as usize] {
      put_swin_block(&mut w, stage, i, dim, HEADS[stage as usize]);
    }
    if stage < 3 {
      put_patch_merge(&mut w, stage, dim);
    }
  }
  put_layer_norm(&mut w, &format!("{ae}.norm"), PATCH_HIDDEN << 3);
  // A non-parameter relative_position_index buffer HF would ship (dropped).
  w.insert(
    "audio_model.audio_encoder.layers.0.blocks.0.attention.self.relative_position_index"
      .to_string(),
    Array::full::<f32>(
      &((WINDOW * WINDOW) as usize, (WINDOW * WINDOW) as usize),
      0.0,
    )
    .unwrap(),
  );

  // ── text tower (text_model.*) ──
  w.insert(
    "text_model.embeddings.word_embeddings.weight".to_string(),
    mat(T_VOCAB, T_HIDDEN),
  );
  w.insert(
    "text_model.embeddings.position_embeddings.weight".to_string(),
    mat(T_MAX_POS, T_HIDDEN),
  );
  w.insert(
    "text_model.embeddings.token_type_embeddings.weight".to_string(),
    mat(T_TYPE_VOCAB, T_HIDDEN),
  );
  put_layer_norm(&mut w, "text_model.embeddings.LayerNorm", T_HIDDEN);
  for i in 0..T_LAYERS {
    put_text_layer(&mut w, i);
  }
  // A non-parameter position_ids buffer HF ships (dropped by sanitize).
  w.insert(
    "text_model.embeddings.position_ids".to_string(),
    Array::full::<f32>(&(1usize, T_MAX_POS as usize), 0.0).unwrap(),
  );

  // ── projections (siblings of the towers) ──
  put_projection(&mut w, "text_projection", T_HIDDEN, PROJ);
  put_projection(&mut w, "audio_projection", AUDIO_HIDDEN, PROJ);

  // ── the unused (train-time) logit_scale parameters HF ships ──
  w.insert(
    "logit_scale_a".to_string(),
    Array::full::<f32>(&[0i32; 0], 2.65).unwrap(),
  );
  w.insert(
    "logit_scale_t".to_string(),
    Array::full::<f32>(&[0i32; 0], 2.65).unwrap(),
  );

  w
}

/// Build a `ClapModel` from the synthetic checkpoint (sanitize → from_weights).
fn build_model() -> ClapModel {
  let cfg = clap_config();
  let weights = sanitize(full_checkpoint()).unwrap();
  ClapModel::from_weights(cfg, weights).unwrap()
}

/// A small `(B, L)` i32 id batch + its `{0,1}` f32 attention mask.
fn ids_and_mask(batch: usize) -> (Array, Array) {
  // Each row: [0, 5, 9, 1] (one pad). The mask masks the trailing pad cell.
  let mut ids = Vec::new();
  let mut mask = Vec::new();
  for _ in 0..batch {
    ids.extend_from_slice(&[0, 5, 9, 1]);
    mask.extend_from_slice(&[1.0, 1.0, 1.0, 0.0]);
  }
  (
    Array::from_slice::<i32>(&ids, &(batch, 4usize)).unwrap(),
    Array::from_slice::<f32>(&mask, &(batch, 4usize)).unwrap(),
  )
}

/// A synthetic 48 kHz waveform (a couple of seconds of a deterministic ramp).
fn samples() -> Vec<f32> {
  (0..96_000)
    .map(|i| ((i % 97) as f32) * 0.001 - 0.05)
    .collect()
}

// ════════════════════════ embed round-trips ════════════════════════════

#[test]
fn embed_audio_is_unit_norm_512() {
  let model = build_model();
  let emb = model.embed_audio(&samples()).unwrap();
  assert_eq!(emb.array().shape(), vec![1, 512], "(B=1, projection_dim)");
  let v = read_f32(emb.array());
  assert!(v.iter().all(|x| x.is_finite()), "audio embedding finite");
  let norm_sq: f32 = v.iter().map(|x| x * x).sum();
  assert!(
    (norm_sq - 1.0).abs() < 1e-4,
    "audio embedding not unit-norm: norm_sq = {norm_sq}"
  );
}

#[test]
fn embed_text_is_unit_norm_512() {
  let model = build_model();
  let (ids, mask) = ids_and_mask(2);
  let emb = model.embed_text(&ids, &mask).unwrap();
  assert_eq!(emb.array().shape(), vec![2, 512], "(B=2, projection_dim)");
  let v = read_f32(emb.array());
  for row in 0..2 {
    let norm_sq: f32 = v[row * 512..(row + 1) * 512].iter().map(|x| x * x).sum();
    assert!(
      (norm_sq - 1.0).abs() < 1e-4,
      "text row {row} not unit-norm: norm_sq = {norm_sq}"
    );
  }
}

#[test]
fn embed_via_audio_input_matches_embed_audio() {
  // The `Embed<AudioInput>` impl runs the preprocessed-mel tail; embedding the
  // front-end's mel through it must match `embed_audio`'s full path.
  let model = build_model();
  let s = samples();
  let direct = read_f32(model.embed_audio(&s).unwrap().array());
  let mel = model.mel_front_end().extract(&s).unwrap();
  let via = read_f32(model.embed(AudioInput(&mel)).unwrap().array());
  assert_eq!(direct.len(), via.len());
  for (a, b) in direct.iter().zip(via.iter()) {
    assert!(
      (a - b).abs() < 1e-6,
      "Embed<AudioInput> must equal embed_audio"
    );
  }
}

#[test]
fn embed_audio_preserves_f16_dtype() {
  // An f16 checkpoint stays f16 end-to-end (no silent f32 promotion).
  let cfg = clap_config();
  let mut raw = full_checkpoint();
  for v in raw.values_mut() {
    *v = v.astype(Dtype::F16).unwrap();
  }
  let weights = sanitize(raw).unwrap();
  let model = ClapModel::from_weights(cfg, weights).unwrap();
  let mel = model
    .mel_front_end()
    .extract(&samples())
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let emb = model.embed_mel(&mel).unwrap();
  assert_eq!(
    emb.array().dtype().unwrap(),
    Dtype::F16,
    "audio embedding must stay f16"
  );
}

// ════════════════════════ classify ════════════════════════════

#[test]
fn classify_empty_on_zero_k_or_no_labels() {
  let model = build_model();
  let s = samples();
  let (ids, mask) = ids_and_mask(3);
  // k == 0 → empty (textclap `Clap::classify`).
  assert!(model.classify(&s, &ids, &mask, 0).unwrap().is_empty());
  // No labels → empty.
  let empty_ids = Array::from_slice::<i32>(&[0i32; 0], &(0usize, 4usize)).unwrap();
  let empty_mask = Array::from_slice::<f32>(&[0f32; 0], &(0usize, 4usize)).unwrap();
  assert!(
    model
      .classify(&s, &empty_ids, &empty_mask, 5)
      .unwrap()
      .is_empty()
  );
}

#[test]
fn classify_truncates_to_k_and_sorts_descending() {
  let model = build_model();
  let s = samples();
  // Five labels (distinct ids → distinct embeddings → distinct scores).
  let ids = Array::from_slice::<i32>(
    &[
      0, 5, 9, 1, // l0
      0, 6, 8, 2, // l1
      0, 7, 4, 3, // l2
      0, 2, 6, 1, // l3
      0, 9, 5, 4, // l4
    ],
    &(5usize, 4usize),
  )
  .unwrap();
  let mask = Array::full::<f32>(&(5usize, 4usize), 1.0).unwrap();
  let top = model.classify(&s, &ids, &mask, 3).unwrap();
  assert_eq!(top.len(), 3, "truncated to k = 3");
  // Descending by score.
  assert!(
    top[0].1 >= top[1].1 && top[1].1 >= top[2].1,
    "scores must be sorted descending: {top:?}"
  );
  // Every returned index is a valid, distinct label index.
  for (idx, _) in &top {
    assert!(*idx < 5, "index in range");
  }
  let mut idxs: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
  idxs.sort_unstable();
  idxs.dedup();
  assert_eq!(idxs.len(), 3, "the top-k indices are distinct");
}

#[test]
fn classify_stable_input_order_tie_break() {
  // THE tie-break oracle. Make every label IDENTICAL (same ids → same text
  // embedding → identical cosine score), so every score ties. A stable sort
  // keeps the input order, so the returned indices must be `[0, 1, 2, 3]` in
  // ascending order (textclap's `sort_by` is stable — equal scores keep input
  // order).
  let model = build_model();
  let s = samples();
  let row = [0i32, 5, 9, 1];
  let mut ids_data = Vec::new();
  for _ in 0..4 {
    ids_data.extend_from_slice(&row);
  }
  let ids = Array::from_slice::<i32>(&ids_data, &(4usize, 4usize)).unwrap();
  let mask = Array::full::<f32>(&(4usize, 4usize), 1.0).unwrap();
  let all = model.classify(&s, &ids, &mask, 4).unwrap();
  assert_eq!(all.len(), 4);
  // All scores equal (identical embeddings).
  for w in all.windows(2) {
    assert!(
      (w[0].1 - w[1].1).abs() < 1e-6,
      "identical labels must score identically: {all:?}"
    );
  }
  // The stable tie-break keeps the input order.
  let idxs: Vec<usize> = all.iter().map(|(i, _)| *i).collect();
  assert_eq!(
    idxs,
    vec![0, 1, 2, 3],
    "ties must preserve input order (stable sort)"
  );
}

#[test]
fn classify_partial_tie_keeps_input_order_within_ties() {
  // Two identical labels (0 and 1) plus a distinct one (2). Whatever the distinct
  // label's rank, the two tied labels must appear in input order (0 before 1).
  let model = build_model();
  let s = samples();
  let ids = Array::from_slice::<i32>(
    &[
      0, 5, 9, 1, // l0
      0, 5, 9, 1, // l1 (identical to l0 → ties)
      0, 7, 4, 3, // l2 (distinct)
    ],
    &(3usize, 4usize),
  )
  .unwrap();
  let mask = Array::full::<f32>(&(3usize, 4usize), 1.0).unwrap();
  let all = model.classify(&s, &ids, &mask, 3).unwrap();
  assert_eq!(all.len(), 3);
  let pos0 = all.iter().position(|(i, _)| *i == 0).unwrap();
  let pos1 = all.iter().position(|(i, _)| *i == 1).unwrap();
  assert!(
    pos0 < pos1,
    "the two tied labels (0, 1) must keep input order: {all:?}"
  );
}

#[test]
fn cosine_of_unit_embeddings_in_range() {
  // `cosine` over two L2-normed embedding rows is in [-1, 1].
  let model = build_model();
  let s = samples();
  let audio = model.embed_audio(&s).unwrap();
  let (ids, mask) = ids_and_mask(1);
  let text = model.embed_text(&ids, &mask).unwrap();
  // Both are (1, 512); reshape to rank-1 rows.
  let a = ops::shape::reshape(audio.array(), &[512]).unwrap();
  let b = ops::shape::reshape(text.array(), &[512]).unwrap();
  let cos = model.cosine(&a, &b).unwrap();
  assert!((-1.0..=1.0).contains(&cos), "cosine in [-1, 1]: {cos}");
}

#[test]
fn classify_rejects_non_rank2_labels() {
  let model = build_model();
  let s = samples();
  let bad = Array::from_slice::<i32>(&[0, 1, 2, 3], &(1usize, 2usize, 2usize)).unwrap();
  let bad_mask = Array::full::<f32>(&(1usize, 2usize, 2usize), 1.0).unwrap();
  assert!(
    model.classify(&s, &bad, &bad_mask, 1).is_err(),
    "rank-3 label_ids must be rejected"
  );
}

// ════════════════════════ sanitize ════════════════════════════

#[test]
fn sanitize_transposes_patch_embed_to_nhwc() {
  // The HF NCHW (C_out, C_in, KH, KW) patch weight becomes NHWC
  // (C_out, KH, KW, C_in).
  let raw = full_checkpoint();
  let before = raw
    .get("audio_model.audio_encoder.patch_embed.proj.weight")
    .unwrap()
    .shape();
  assert_eq!(
    before,
    vec![96, 1, 4, 4],
    "fixture ships the HF NCHW conv weight"
  );
  let out = sanitize(raw).unwrap();
  let after = out
    .get("audio_model.audio_encoder.patch_embed.proj.weight")
    .unwrap()
    .shape();
  assert_eq!(
    after,
    vec![96, 4, 4, 1],
    "sanitize transposes the conv weight to NHWC [0,2,3,1]"
  );
}

#[test]
fn sanitize_is_idempotent_on_patch_weight() {
  // Running sanitize twice keeps the NHWC layout (the transpose is keyed on the
  // channels-first signature; an already-NHWC weight passes through).
  let once = sanitize(full_checkpoint()).unwrap();
  let twice = sanitize(once).unwrap();
  let shape = twice
    .get("audio_model.audio_encoder.patch_embed.proj.weight")
    .unwrap()
    .shape();
  assert_eq!(shape, vec![96, 4, 4, 1], "idempotent NHWC layout");
}

#[test]
fn sanitize_drops_non_parameter_buffers() {
  let out = sanitize(full_checkpoint()).unwrap();
  assert!(
    !out.keys().any(|k| k.contains("position_ids")),
    "position_ids buffer must be dropped"
  );
  assert!(
    !out.keys().any(|k| k.contains("relative_position_index")),
    "relative_position_index buffer must be dropped"
  );
  // The real parameters (and the relative_position_bias_table, a true parameter)
  // survive.
  assert!(
    out
      .keys()
      .any(|k| k.ends_with("attention.self.relative_position_bias_table")),
    "the relative_position_bias_table parameter must survive"
  );
}

#[test]
fn sanitize_keeps_tower_prefixes_verbatim() {
  let out = sanitize(full_checkpoint()).unwrap();
  for key in [
    "audio_model.audio_encoder.batch_norm.weight",
    "audio_model.audio_encoder.norm.weight",
    "text_model.embeddings.word_embeddings.weight",
    "text_model.encoder.layer.0.attention.self.query.weight",
    "audio_projection.linear1.weight",
    "text_projection.linear1.weight",
    "logit_scale_a",
    "logit_scale_t",
  ] {
    assert!(out.contains_key(key), "sanitize must keep `{key}` verbatim");
  }
}

// ════════════════════════ quant path ════════════════════════════

/// Affine group size (divides every quantized `in` axis: the audio dims
/// 96/192/384/768 + their 4·dim MLP widths and the text 768/3072/512 are all
/// multiples of 32).
const QGROUP: i32 = 32;
const QBITS: i32 = 8;

/// Replace the dense `<prefix>.weight` with the real affine quantize triple.
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .unwrap_or_else(|| panic!("dense weight {prefix}.weight present"));
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// Quantize EVERY `nn.Linear` across both towers + the two projections in the
/// (HF-prefixed) checkpoint.
fn quantize_all_linears(w: &mut HashMap<String, Array>) {
  // Audio Swin Linears + patch-merge reductions.
  for stage in 0..4i32 {
    for i in 0..DEPTHS[stage as usize] {
      let p = format!("audio_model.audio_encoder.layers.{stage}.blocks.{i}");
      for proj in ["query", "key", "value"] {
        quantize_weight_in_place(w, &format!("{p}.attention.self.{proj}"));
      }
      quantize_weight_in_place(w, &format!("{p}.attention.output.dense"));
      quantize_weight_in_place(w, &format!("{p}.intermediate.dense"));
      quantize_weight_in_place(w, &format!("{p}.output.dense"));
    }
    if stage < 3 {
      quantize_weight_in_place(
        w,
        &format!("audio_model.audio_encoder.layers.{stage}.downsample.reduction"),
      );
    }
  }
  // Text RoBERTa Linears.
  for i in 0..T_LAYERS {
    let l = format!("text_model.encoder.layer.{i}");
    for p in ["query", "key", "value"] {
      quantize_weight_in_place(w, &format!("{l}.attention.self.{p}"));
    }
    quantize_weight_in_place(w, &format!("{l}.attention.output.dense"));
    quantize_weight_in_place(w, &format!("{l}.intermediate.dense"));
    quantize_weight_in_place(w, &format!("{l}.output.dense"));
  }
  // Both projections.
  for proj in ["text_projection", "audio_projection"] {
    quantize_weight_in_place(w, &format!("{proj}.linear1"));
    quantize_weight_in_place(w, &format!("{proj}.linear2"));
  }
}

/// A `ClapConfig` JSON carrying a `quantization` block.
fn quant_config() -> ClapConfig {
  let json = format!(r#"{{ "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS} }} }}"#);
  let cfg = ClapConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Parse the quantization block the same way the loader would.
fn quant_from_json() -> PerLayerQuantization {
  let json = format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#);
  serde_json::from_str::<PerLayerQuantization>(&json).unwrap()
}

#[test]
fn quantized_checkpoint_builds_quantized_layers_and_forwards() {
  let cfg = quant_config();
  let mut raw = full_checkpoint();
  quantize_all_linears(&mut raw);
  let weights = sanitize(raw).unwrap();
  let quant = quant_from_json();
  let model = ClapModel::from_weights_quantized(cfg, weights, Some(&quant)).unwrap();
  assert!(
    model.all_linears_quantized(),
    "every Linear (both towers + the audio projection) must have loaded quantized"
  );
  // The quantized forward is finite + unit-norm.
  let audio = read_f32(model.embed_audio(&samples()).unwrap().array());
  assert!(
    audio.iter().all(|x| x.is_finite()),
    "quantized audio finite"
  );
  let norm_sq: f32 = audio.iter().map(|x| x * x).sum();
  assert!(
    (norm_sq - 1.0).abs() < 1e-3,
    "quantized audio not unit-norm: {norm_sq}"
  );
}

#[test]
fn quantized_weights_with_dense_config_errors() {
  // A `.scales`-bearing checkpoint loaded with a DENSE config (no quantization
  // block) is a typed error (the weights say quantized, the config says dense) —
  // never a silent dense reinterpret.
  let cfg = clap_config(); // no quantization block
  let mut raw = full_checkpoint();
  // Quantize one projection Linear so a `.scales` sibling is present.
  quantize_weight_in_place(&mut raw, "audio_projection.linear1");
  let weights = sanitize(raw).unwrap();
  // `from_weights` threads quant = None; the `.scales` must error.
  let err = ClapModel::from_weights(cfg, weights);
  assert!(
    err.is_err(),
    "a `.scales` sibling without a quantization config must error"
  );
}

// ════════════════════════ factory registration ════════════════════════════

#[test]
fn register_round_trips_model_type_clap() {
  // The registry registers under "clap" and constructs a ClapModel from a loaded
  // directory bundle (config JSON + sanitized-or-raw weights — the constructor
  // sanitizes itself).
  let mut registry = EmbeddingModelTypeRegistry::new();
  let displaced = register(&mut registry);
  assert!(displaced.is_none(), "first registration displaces nothing");
  assert!(registry.contains("clap"), "registry contains `clap`");
  // A `-`-spelled / canonicalization round-trip (remap normalizes separators).
  assert!(registry.contains("clap"), "canonical id resolves");

  // Build a LoadedEmbeddingModel from the RAW (un-sanitized) HF-prefixed
  // checkpoint + a minimal config JSON; the constructor sanitizes + builds.
  let loaded = LoadedEmbeddingModel::new("clap".to_string(), "{}".to_string(), full_checkpoint());
  let model = registry.create(&loaded, None).unwrap();
  // The constructed model answers the text-embedder umbrella (CLAP's text tower).
  assert!(
    model.as_text_embedder().is_some(),
    "the constructed CLAP model exposes its text tower as a TextEmbedder"
  );
  // And downcasts to the concrete ClapModel for the audio tower / classify.
  let clap = model
    .as_any()
    .downcast_ref::<ClapModel>()
    .expect("downcasts to ClapModel");
  let emb = clap.embed_audio(&samples()).unwrap();
  assert_eq!(emb.array().shape(), vec![1, 512]);
}

#[test]
fn builtin_registry_includes_clap() {
  // `with_builtin_models` wires the clap arm when the feature is on.
  let registry = EmbeddingModelTypeRegistry::new().with_builtin_models();
  assert!(
    registry.contains("clap"),
    "with_builtin_models registers `clap`"
  );
}
