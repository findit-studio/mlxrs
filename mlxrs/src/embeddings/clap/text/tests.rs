//! Oracle / shape tests for the CLAP RoBERTa text tower.
//!
//! No checkpoint is available, so these pin the layer math against closed-form
//! expectations and exercise a full-config-sized tower built from synthetic
//! weights:
//!
//! - the **RoBERTa position-id offset** (`pad_id + 1 + cumsum(non_pad)`) — the
//!   killer test (ids `[0, 5, 9, 1, 1]`, `pad = 1` → `[2, 3, 4, 1, 1]`);
//! - the **additive padding mask** (`{0, -inf}` form + the activation dtype);
//! - **CLS pooling** (`last_hidden_state[:, 0, :]`);
//! - the **ReLU** projection activation + the `linear1 → ReLU → linear2`
//!   projection math (hand-computed);
//! - one **RoBERTa post-norm layer** (shape + that the additive mask actually
//!   changes the output);
//! - the **whole tower** at the real `laion/clap-htsat-unfused` config size:
//!   the `(B, 512)` L2-normalized output, the rank-2 input guard, a
//!   quantized-checkpoint load + forward, and **f16 / bf16** dtype preservation.
//!
//! `ClapConfig::validate` pins every dim to the exact checkpoint values (it is a
//! fail-fast faithfulness gate, not a cardinality cap), so the whole-tower tests
//! use the real `hidden = 768` / `12`-layer config with constant-fill synthetic
//! weights; the sub-block tests use ad-hoc tiny [`LayerDims`] directly.

use std::collections::HashMap;

use super::*;
use crate::{dtype::Dtype, embeddings::clap::shared::RobertaLayer};

// ───────────────────────── small Array helpers ─────────────────────────

/// Cast `a` to f32, eval, and read it back as a flat `Vec<f32>` (`to_vec` is
/// dtype-strict + needs `&mut`; there is no implicit eval).
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

// ═══════════════════════ position-id offset (R2 killer test) ═══════════════

#[test]
fn position_ids_match_roberta_offset_closed_form() {
  // HF `create_position_ids_from_input_ids` with pad = 1:
  //   ids   = [0, 5, 9, 1, 1]
  //   mask  = [1, 1, 1, 0, 0]   (ids != 1)
  //   cumsum= [1, 2, 3, 3, 3]
  //   *mask = [1, 2, 3, 0, 0]
  //   +pad  = [2, 3, 4, 1, 1]
  let ids = Array::from_slice::<i32>(&[0, 5, 9, 1, 1], &(1usize, 5usize)).unwrap();
  let pos = position_ids_from_ids(&ids, 1).unwrap();
  assert_eq!(
    pos.shape(),
    vec![1, 5],
    "position ids keep the (B, L) shape"
  );
  let got = read_f32(&pos);
  assert_eq!(got, vec![2.0, 3.0, 4.0, 1.0, 1.0]);
}

#[test]
fn position_ids_all_pad_row_is_all_pad_id() {
  // An all-pad row: mask all-0, so cumsum * mask = 0 everywhere, +pad = pad.
  let ids = Array::from_slice::<i32>(&[1, 1, 1], &(1usize, 3usize)).unwrap();
  let pos = position_ids_from_ids(&ids, 1).unwrap();
  assert_eq!(read_f32(&pos), vec![1.0, 1.0, 1.0]);
}

#[test]
fn position_ids_no_pad_row_counts_from_two() {
  // No pad token: every position is real, so positions are pad+1, pad+2, …
  // ids = [4, 7, 2, 9], pad = 1 → mask all-1 → cumsum [1,2,3,4] → +1 [2,3,4,5].
  let ids = Array::from_slice::<i32>(&[4, 7, 2, 9], &(1usize, 4usize)).unwrap();
  let pos = position_ids_from_ids(&ids, 1).unwrap();
  assert_eq!(read_f32(&pos), vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn position_ids_batched_independent_rows() {
  // Two rows of different pad layouts — the cumsum runs per-row (axis 1).
  // row0 ids [3, 8, 1] → [2, 3, 1]; row1 ids [1, 5, 6] → [1, 2, 3].
  let ids = Array::from_slice::<i32>(&[3, 8, 1, 1, 5, 6], &(2usize, 3usize)).unwrap();
  let pos = position_ids_from_ids(&ids, 1).unwrap();
  assert_eq!(read_f32(&pos), vec![2.0, 3.0, 1.0, 1.0, 2.0, 3.0]);
}

#[test]
fn position_ids_stay_integer_dtype() {
  // The whole offset runs in i32 (the index dtype) — no float rounding can
  // corrupt a position index.
  let ids = Array::from_slice::<i32>(&[0, 5, 1], &(1usize, 3usize)).unwrap();
  let pos = position_ids_from_ids(&ids, 1).unwrap();
  assert_eq!(pos.dtype().unwrap(), Dtype::I32);
}

// ═══════════════════════════ additive padding mask ═════════════════════════

#[test]
fn additive_mask_maps_pad_to_neg_inf_real_to_zero() {
  // mask {1,1,0} → additive {0, 0, -inf}, reshaped (B, 1, 1, L).
  let mask = Array::from_slice::<f32>(&[1.0, 1.0, 0.0], &(1usize, 3usize)).unwrap();
  let add = build_additive_mask(&mask, Dtype::F32).unwrap();
  assert_eq!(add.shape(), vec![1, 1, 1, 3], "(B, 1, 1, L) key-axis mask");
  let v = read_f32(&add);
  assert_eq!(v[0], 0.0);
  assert_eq!(v[1], 0.0);
  assert!(v[2].is_infinite() && v[2] < 0.0, "pad cell is -inf");
}

#[test]
fn additive_mask_casts_to_activation_dtype() {
  // The mask must adopt the activation dtype (dtype preservation) so the fused
  // SDPA sees a matching-dtype additive mask.
  let mask = Array::from_slice::<f32>(&[1.0, 0.0], &(1usize, 2usize)).unwrap();
  for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
    let add = build_additive_mask(&mask, dtype).unwrap();
    assert_eq!(
      add.dtype().unwrap(),
      dtype,
      "additive mask must be {dtype:?}"
    );
  }
}

#[test]
fn additive_mask_rejects_non_rank2() {
  let mask = Array::from_slice::<f32>(&[1.0, 0.0], &(2usize,)).unwrap();
  let err = build_additive_mask(&mask, Dtype::F32);
  assert!(err.is_err(), "rank-1 mask must be rejected");
}

// ═══════════════════════════════ CLS pooling ═══════════════════════════════

#[test]
fn pool_cls_takes_first_sequence_position() {
  // (B=1, L=3, hidden=2): rows [[10,11],[20,21],[30,31]] → CLS = [10, 11].
  let h = Array::from_slice::<f32>(
    &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
    &(1usize, 3usize, 2usize),
  )
  .unwrap();
  let cls = pool_cls(&h).unwrap();
  assert_eq!(cls.shape(), vec![1, 2], "(B, hidden)");
  assert_eq!(read_f32(&cls), vec![10.0, 11.0]);
}

#[test]
fn pool_cls_batched() {
  // (B=2, L=2, hidden=2): batch row 0 CLS = [1,2], row 1 CLS = [5,6].
  let h = Array::from_slice::<f32>(
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    &(2usize, 2usize, 2usize),
  )
  .unwrap();
  let cls = pool_cls(&h).unwrap();
  assert_eq!(cls.shape(), vec![2, 2]);
  assert_eq!(read_f32(&cls), vec![1.0, 2.0, 5.0, 6.0]);
}

// ════════════════════════════════ ReLU ═════════════════════════════════════

#[test]
fn relu_clamps_negatives_to_zero() {
  let x = Array::from_slice::<f32>(&[-2.0, -0.1, 0.0, 0.5, 3.0], &(5usize,)).unwrap();
  let y = relu(&x).unwrap();
  assert_eq!(read_f32(&y), vec![0.0, 0.0, 0.0, 0.5, 3.0]);
}

#[test]
fn relu_preserves_activation_dtype() {
  // The `0` floor is dtype-matched so an f16/bf16 activation is not promoted.
  for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
    let x = Array::from_slice::<f32>(&[-1.0, 2.0], &(2usize,))
      .unwrap()
      .astype(dtype)
      .unwrap();
    assert_eq!(
      relu(&x).unwrap().dtype().unwrap(),
      dtype,
      "relu keeps {dtype:?}"
    );
  }
}

// ══════════════════════ ClapProjectionLayer math ═══════════════════════════

/// Build a [`ClapProjectionLayer`] from a synthetic dense weight map at the
/// given `(hidden, proj)` dims.
fn proj_weights(prefix: &str, hidden: i32, proj: i32) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(format!("{prefix}.linear1.weight"), mat(proj, hidden));
  w.insert(format!("{prefix}.linear1.bias"), vec1(proj));
  w.insert(format!("{prefix}.linear2.weight"), mat(proj, proj));
  w.insert(format!("{prefix}.linear2.bias"), vec1(proj));
  w
}

#[test]
fn projection_layer_is_linear1_relu_linear2() {
  // Hand-computed `(linear2 ∘ relu ∘ linear1)` at hidden=2, proj=2.
  //   linear1.W = [[1, 0], [-1, 2]], linear1.b = [0, 1]
  //   relu
  //   linear2.W = [[2, 0], [0, 3]], linear2.b = [1, -1]
  // x = [1, 1]:
  //   linear1: [1*1 + 0*1, -1*1 + 2*1] + [0, 1] = [1, 1] + [0, 1] = [1, 2]
  //   relu:    [1, 2]
  //   linear2: [2*1 + 0*2, 0*1 + 3*2] + [1, -1] = [2, 6] + [1, -1] = [3, 5]
  let mut w = HashMap::new();
  w.insert(
    "p.linear1.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 0.0, -1.0, 2.0], &(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "p.linear1.bias".to_string(),
    Array::from_slice::<f32>(&[0.0, 1.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "p.linear2.weight".to_string(),
    Array::from_slice::<f32>(&[2.0, 0.0, 0.0, 3.0], &(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "p.linear2.bias".to_string(),
    Array::from_slice::<f32>(&[1.0, -1.0], &(2usize,)).unwrap(),
  );
  let proj = ClapProjectionLayer::from_weights("p", &mut w, 2, 2, None).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 1.0], &(1usize, 2usize)).unwrap();
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 2]);
  assert_eq!(read_f32(&out), vec![3.0, 5.0]);
}

#[test]
fn projection_layer_negative_path_relu_zeroes() {
  // If linear1's output is all-negative, relu zeroes it and linear2 returns its
  // bias. linear1.W = [[-1,-1],[-1,-1]], b = [0,0]; x=[1,1] → [-2,-2] → relu [0,0]
  //   → linear2.W any, b=[7, -3] → [7, -3].
  let mut w = HashMap::new();
  w.insert(
    "p.linear1.weight".to_string(),
    Array::from_slice::<f32>(&[-1.0, -1.0, -1.0, -1.0], &(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "p.linear1.bias".to_string(),
    Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
  );
  w.insert(
    "p.linear2.weight".to_string(),
    Array::from_slice::<f32>(&[5.0, 5.0, 5.0, 5.0], &(2usize, 2usize)).unwrap(),
  );
  w.insert(
    "p.linear2.bias".to_string(),
    Array::from_slice::<f32>(&[7.0, -3.0], &(2usize,)).unwrap(),
  );
  let proj = ClapProjectionLayer::from_weights("p", &mut w, 2, 2, None).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 1.0], &(1usize, 2usize)).unwrap();
  let out = proj.forward(&x).unwrap();
  assert_eq!(read_f32(&out), vec![7.0, -3.0]);
}

#[test]
fn projection_layer_shape_at_clap_dims() {
  // The real CLAP projection: hidden 768 → proj 512 → 512.
  let mut w = proj_weights("text_projection", 768, 512);
  let proj = ClapProjectionLayer::from_weights("text_projection", &mut w, 768, 512, None).unwrap();
  let x = mat(2, 768); // (B=2, hidden)
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![2, 512], "(B, projection_dim)");
}

// ═════════════════════════ one RoBERTa post-norm layer ═════════════════════

const L_HIDDEN: i32 = 8;
const L_HEADS: i32 = 2;
const L_INTER: i32 = 16;

/// Build a synthetic weight map for a single `encoder.layer.0` at the small
/// `(L_HIDDEN, L_HEADS, L_INTER)` dims (the post-strip keys
/// `RobertaLayer::from_weights` consumes).
fn one_layer_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  let l = "encoder.layer.0";
  for p in ["query", "key", "value"] {
    w.insert(
      format!("{l}.attention.self.{p}.weight"),
      mat(L_HIDDEN, L_HIDDEN),
    );
    w.insert(format!("{l}.attention.self.{p}.bias"), vec1(L_HIDDEN));
  }
  w.insert(
    format!("{l}.attention.output.dense.weight"),
    mat(L_HIDDEN, L_HIDDEN),
  );
  w.insert(format!("{l}.attention.output.dense.bias"), vec1(L_HIDDEN));
  w.insert(
    format!("{l}.attention.output.LayerNorm.weight"),
    vec1(L_HIDDEN),
  );
  w.insert(
    format!("{l}.attention.output.LayerNorm.bias"),
    vec1(L_HIDDEN),
  );
  w.insert(
    format!("{l}.intermediate.dense.weight"),
    mat(L_INTER, L_HIDDEN),
  );
  w.insert(format!("{l}.intermediate.dense.bias"), vec1(L_INTER));
  w.insert(format!("{l}.output.dense.weight"), mat(L_HIDDEN, L_INTER));
  w.insert(format!("{l}.output.dense.bias"), vec1(L_HIDDEN));
  w.insert(format!("{l}.output.LayerNorm.weight"), vec1(L_HIDDEN));
  w.insert(format!("{l}.output.LayerNorm.bias"), vec1(L_HIDDEN));
  w
}

#[test]
fn roberta_layer_preserves_shape() {
  let dims = LayerDims::new(L_HIDDEN, L_INTER, L_HEADS, 1e-5).unwrap();
  let mut w = one_layer_weights();
  let layer = RobertaLayer::from_weights(&mut w, "encoder", 0, dims, None).unwrap();
  let x = mat(1, L_HIDDEN * 3); // flatten (1, 3, 8) below
  let x = ops::shape::reshape(&x, &[1, 3, L_HIDDEN]).unwrap();
  let mask = build_additive_mask(
    &Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(1usize, 3usize)).unwrap(),
    Dtype::F32,
  )
  .unwrap();
  let out = layer.forward(&x, Mask::Array(&mask)).unwrap();
  assert_eq!(
    out.shape(),
    vec![1, 3, L_HIDDEN as usize],
    "(B, L, hidden) preserved"
  );
}

#[test]
fn roberta_layer_additive_mask_changes_output() {
  // Masking a key position must change the attention output (proves the mask is
  // wired into SDPA, not ignored). Compare an all-real mask vs one with the last
  // key masked.
  let dims = LayerDims::new(L_HIDDEN, L_INTER, L_HEADS, 1e-5).unwrap();
  let mut w = one_layer_weights();
  let layer = RobertaLayer::from_weights(&mut w, "encoder", 0, dims, None).unwrap();
  let x = ops::shape::reshape(&mat(1, L_HIDDEN * 3), &[1, 3, L_HIDDEN]).unwrap();

  let full = build_additive_mask(
    &Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(1usize, 3usize)).unwrap(),
    Dtype::F32,
  )
  .unwrap();
  let masked = build_additive_mask(
    &Array::from_slice::<f32>(&[1.0, 1.0, 0.0], &(1usize, 3usize)).unwrap(),
    Dtype::F32,
  )
  .unwrap();

  let out_full = read_f32(&layer.forward(&x, Mask::Array(&full)).unwrap());
  let out_masked = read_f32(&layer.forward(&x, Mask::Array(&masked)).unwrap());
  let max_diff = out_full
    .iter()
    .zip(out_masked.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
  assert!(
    max_diff > 1e-5,
    "masking a key must change the layer output (got max diff {max_diff})"
  );
}

// ══════════════════ whole tower at the real CLAP config size ════════════════

/// The real `laion/clap-htsat-unfused` RoBERTa dims (pinned by
/// `ClapTextConfig::validate`).
const HIDDEN: i32 = 768;
const INTER: i32 = 3072;
const LAYERS: i32 = 12;
const VOCAB: i32 = 50265;
const MAX_POS: i32 = 514;
const TYPE_VOCAB: i32 = 1;
const PROJ: i32 = 512;

/// The full `laion/clap-htsat-unfused` config (defaults already match it; built
/// explicitly so the test is self-documenting and `validate` runs).
fn clap_config() -> ClapConfig {
  let cfg = ClapConfig::from_json("{}").unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Insert one `encoder.layer.{i}` block's dense weights.
fn insert_layer(w: &mut HashMap<String, Array>, i: i32) {
  let l = format!("encoder.layer.{i}");
  for p in ["query", "key", "value"] {
    w.insert(
      format!("{l}.attention.self.{p}.weight"),
      mat(HIDDEN, HIDDEN),
    );
    w.insert(format!("{l}.attention.self.{p}.bias"), vec1(HIDDEN));
  }
  w.insert(
    format!("{l}.attention.output.dense.weight"),
    mat(HIDDEN, HIDDEN),
  );
  w.insert(format!("{l}.attention.output.dense.bias"), vec1(HIDDEN));
  w.insert(
    format!("{l}.attention.output.LayerNorm.weight"),
    vec1(HIDDEN),
  );
  w.insert(format!("{l}.attention.output.LayerNorm.bias"), vec1(HIDDEN));
  w.insert(format!("{l}.intermediate.dense.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{l}.intermediate.dense.bias"), vec1(INTER));
  w.insert(format!("{l}.output.dense.weight"), mat(HIDDEN, INTER));
  w.insert(format!("{l}.output.dense.bias"), vec1(HIDDEN));
  w.insert(format!("{l}.output.LayerNorm.weight"), vec1(HIDDEN));
  w.insert(format!("{l}.output.LayerNorm.bias"), vec1(HIDDEN));
}

/// A full dense weight map for the real-config RoBERTa text tower (the
/// post-sanitize keys `ClapTextModel::from_weights` consumes).
fn clap_text_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert(
    "embeddings.word_embeddings.weight".to_string(),
    mat(VOCAB, HIDDEN),
  );
  w.insert(
    "embeddings.position_embeddings.weight".to_string(),
    mat(MAX_POS, HIDDEN),
  );
  w.insert(
    "embeddings.token_type_embeddings.weight".to_string(),
    mat(TYPE_VOCAB, HIDDEN),
  );
  w.insert("embeddings.LayerNorm.weight".to_string(), vec1(HIDDEN));
  w.insert("embeddings.LayerNorm.bias".to_string(), vec1(HIDDEN));
  for i in 0..LAYERS {
    insert_layer(&mut w, i);
  }
  w.insert(
    "text_projection.linear1.weight".to_string(),
    mat(PROJ, HIDDEN),
  );
  w.insert("text_projection.linear1.bias".to_string(), vec1(PROJ));
  w.insert(
    "text_projection.linear2.weight".to_string(),
    mat(PROJ, PROJ),
  );
  w.insert("text_projection.linear2.bias".to_string(), vec1(PROJ));
  w
}

/// A small `(B, L)` i32 id batch with a couple of pad tokens, and its `{0,1}`
/// f32 attention mask (matching the encode pipeline's layout).
fn ids_and_mask() -> (Array, Array) {
  // B=2, L=4. Row 0: [0, 5, 9, 1] (one pad). Row 1: [0, 7, 1, 1] (two pad).
  let ids = Array::from_slice::<i32>(&[0, 5, 9, 1, 0, 7, 1, 1], &(2usize, 4usize)).unwrap();
  let mask =
    Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], &(2usize, 4usize)).unwrap();
  (ids, mask)
}

#[test]
fn tower_forward_shape_and_unit_norm() {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  let (ids, mask) = ids_and_mask();
  let out = model.encode_text(&ids, &mask).unwrap();
  assert_eq!(out.shape(), vec![2, 512], "(B, projection_dim=512)");
  // L2-normalized: each row's norm² ≈ 1.
  let v = read_f32(&out);
  for row in 0..2 {
    let norm_sq: f32 = v[row * 512..(row + 1) * 512].iter().map(|x| x * x).sum();
    assert!(
      (norm_sq - 1.0).abs() < 1e-4,
      "row {row} not unit-norm: norm_sq = {norm_sq}"
    );
    assert!(
      v[row * 512..(row + 1) * 512].iter().all(|x| x.is_finite()),
      "row {row} has non-finite components"
    );
  }
}

#[test]
fn tower_embed_text_matches_encode_text() {
  // The TextEmbedder seam wraps `encode_text` — the two must agree.
  let cfg = clap_config();
  let mut w = clap_text_weights();
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  let (ids, mask) = ids_and_mask();
  let direct = read_f32(&model.encode_text(&ids, &mask).unwrap());
  let viaembed = read_f32(model.embed_text(&ids, &mask).unwrap().array());
  assert_eq!(direct.len(), viaembed.len());
  for (a, b) in direct.iter().zip(viaembed.iter()) {
    assert!((a - b).abs() < 1e-6, "embed_text must equal encode_text");
  }
}

#[test]
fn tower_text_encoding_is_dynamic_right_pad_capped_512() {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  let enc = model.text_encoding();
  assert!(enc.add_special_tokens, "RoBERTa adds <s>/</s>");
  assert_eq!(enc.max_length, Some(512), "capped at 512 real tokens");
  match enc.padding {
    Padding::DynamicRightPad { pad_token_id } => {
      assert_eq!(pad_token_id, 1, "RoBERTa pad id is 1");
    }
    other => panic!("expected DynamicRightPad, got {other:?}"),
  }
}

#[test]
fn tower_rejects_non_rank2_input() {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  // rank-3 ids.
  let ids = Array::from_slice::<i32>(&[0, 1, 2, 3], &(1usize, 2usize, 2usize)).unwrap();
  let mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(1usize, 4usize)).unwrap();
  assert!(
    model.encode_text(&ids, &mask).is_err(),
    "rank-3 input_ids must be rejected"
  );
}

#[test]
fn tower_rejects_broadcastable_mask_shape() {
  // HF RoBERTa contract: the attention mask must share the `(B, L)` shape of the
  // ids. The fused SDPA broadcasts a `(1, L)` row or `(B, 1)` column mask, which
  // would silently apply the wrong padding pattern — both must be rejected.
  let cfg = clap_config();
  let mut w = clap_text_weights();
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  let (ids, matched_mask) = ids_and_mask(); // B=2, L=4.

  // (1, 4) mask against a B=2 batch: the broadcast-row case.
  let row_mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 0.0], &(1usize, 4usize)).unwrap();
  assert!(
    matches!(
      model.encode_text(&ids, &row_mask),
      Err(Error::ShapePairMismatch(_))
    ),
    "a (1, 4) mask for a B=2 batch must be rejected (broadcast-row)"
  );

  // (2, 1) mask against an L=4 sequence: the broadcast-col case.
  let col_mask = Array::from_slice::<f32>(&[1.0, 1.0], &(2usize, 1usize)).unwrap();
  assert!(
    matches!(
      model.encode_text(&ids, &col_mask),
      Err(Error::ShapePairMismatch(_))
    ),
    "a (2, 1) mask for an L=4 sequence must be rejected (broadcast-col)"
  );

  // The matched (2, 4) mask still succeeds — no regression.
  assert!(
    model.encode_text(&ids, &matched_mask).is_ok(),
    "a matched (2, 4) mask must be accepted"
  );
}

#[test]
fn tower_missing_weight_errors() {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  w.remove("text_projection.linear2.weight");
  let err = ClapTextModel::from_weights(&cfg, &mut w);
  assert!(
    err.is_err(),
    "a missing projection weight must error at load"
  );
}

#[test]
fn tower_wrong_shape_weight_errors() {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  // Word embedding with the wrong hidden width.
  w.insert(
    "embeddings.word_embeddings.weight".to_string(),
    mat(VOCAB, HIDDEN + 1),
  );
  let err = ClapTextModel::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "a wrong-shape embedding must error at load");
}

// ───────────────────────── dtype preservation ─────────────────────────

/// The encode path on an f16 / bf16 checkpoint must stay in that dtype: the
/// position rows, the token-type row, the additive mask, and the ReLU floor are
/// all cast back to the activation dtype, so the tower output is f16 / bf16
/// rather than silently promoted to f32 (the recurring faithfulness bug).
fn assert_tower_preserves_dtype(dtype: Dtype) {
  let cfg = clap_config();
  let mut w = clap_text_weights();
  // Cast every weight to the target dtype (an fp16/bf16 checkpoint).
  for v in w.values_mut() {
    *v = v.astype(dtype).unwrap();
  }
  let model = ClapTextModel::from_weights(&cfg, &mut w).unwrap();
  let (ids, mask) = ids_and_mask();
  let out = model.encode_text(&ids, &mask).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "tower output must stay {dtype:?} (no silent f32 promotion)"
  );
  assert_eq!(out.shape(), vec![2, 512]);
}

#[test]
fn tower_preserves_f16() {
  assert_tower_preserves_dtype(Dtype::F16);
}

#[test]
fn tower_preserves_bf16() {
  assert_tower_preserves_dtype(Dtype::BF16);
}

// ───────────────────────── embeddings dtype preservation ─────────────────────────

#[test]
fn embeddings_sum_preserves_activation_dtype() {
  // The embeddings sub-block alone: word + position(offset) + token_type + LN.
  // A wrong (f32-promoting) position/token_type cast would surface here.
  let cfg = clap_config();
  let mut w = clap_text_weights();
  for v in w.values_mut() {
    *v = v.astype(Dtype::F16).unwrap();
  }
  let eps = cfg.text_config.layer_norm_eps as f32;
  let emb = ClapTextEmbeddings::from_weights(&cfg, &mut w, eps, None).unwrap();
  let ids = Array::from_slice::<i32>(&[0, 5, 9, 1], &(1usize, 4usize)).unwrap();
  let out = emb.forward(&ids).unwrap();
  assert_eq!(out.dtype().unwrap(), Dtype::F16, "embeddings stay f16");
  assert_eq!(out.shape(), vec![1, 4, 768]);
}

// ───────────────────────── quantized load + forward ─────────────────────────

/// Affine group size for the synthetic quantized checkpoint (divides every
/// quantized weight's `in` axis: 768, 3072, 512 are all multiples of 128).
const QGROUP: i32 = 128;
/// Bit depth for the synthetic quantized checkpoint.
const QBITS: i32 = 8;

/// Replace the dense `<prefix>.weight` with the real `ops::quantized::quantize`
/// affine triple (`<prefix>.weight` packed + `<prefix>.scales` +
/// `<prefix>.biases`), mirroring how an mlx-community quantized checkpoint
/// stores a quantized `nn.Linear`.
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

/// A `ClapConfig` JSON carrying a `quantization` block (group_size / bits), so
/// the loader resolves the per-layer scheme for the `.scales`-bearing weights.
fn quant_config() -> ClapConfig {
  let json = format!(r#"{{ "quantization": {{ "group_size": {QGROUP}, "bits": {QBITS} }} }}"#);
  let cfg = ClapConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Parse the quantization block from the config JSON the same way the loader
/// would (so the `.scales`-bearing layers resolve their scheme). Reuses the
/// shared `PerLayerQuantization` deserializer via a tiny JSON wrapper.
fn quant_from_json() -> PerLayerQuantization {
  // The CLAP config's `quantization` block is a global `{group_size, bits}`.
  let json = format!(r#"{{ "group_size": {QGROUP}, "bits": {QBITS} }}"#);
  serde_json::from_str::<PerLayerQuantization>(&json).unwrap()
}

#[test]
fn tower_loads_and_forwards_quantized_checkpoint() {
  let cfg = quant_config();
  let mut w = clap_text_weights();

  // Quantize every nn.Linear: the attention q/k/v + output dense, the FFN
  // intermediate/output dense, and the two projection layers. The word/position/
  // token_type embeddings and the LayerNorms stay dense (the position table is
  // sliced; LayerNorm weights are not quantizable). This mirrors quantizing only
  // the Linears, which still exercises the quantized load + forward path.
  for i in 0..LAYERS {
    let l = format!("encoder.layer.{i}");
    for p in ["query", "key", "value"] {
      quantize_weight_in_place(&mut w, &format!("{l}.attention.self.{p}"));
    }
    quantize_weight_in_place(&mut w, &format!("{l}.attention.output.dense"));
    quantize_weight_in_place(&mut w, &format!("{l}.intermediate.dense"));
    quantize_weight_in_place(&mut w, &format!("{l}.output.dense"));
  }
  quantize_weight_in_place(&mut w, "text_projection.linear1");
  quantize_weight_in_place(&mut w, "text_projection.linear2");

  let quant = quant_from_json();
  let model = ClapTextModel::from_weights_quantized(&cfg, &mut w, Some(&quant)).unwrap();
  assert!(
    model.all_projections_quantized(),
    "every Linear must have loaded the quantized variant"
  );
  assert!(
    !model.word_embedding_is_quantized(),
    "the dense word embedding stays dense in this fixture"
  );

  let (ids, mask) = ids_and_mask();
  let out = model.encode_text(&ids, &mask).unwrap();
  assert_eq!(out.shape(), vec![2, 512]);
  let v = read_f32(&out);
  assert!(
    v.iter().all(|x| x.is_finite()),
    "quantized forward is finite"
  );
  for row in 0..2 {
    let norm_sq: f32 = v[row * 512..(row + 1) * 512].iter().map(|x| x * x).sum();
    assert!(
      (norm_sq - 1.0).abs() < 1e-3,
      "quantized row {row} not unit-norm: norm_sq = {norm_sq}"
    );
  }
}

#[test]
fn tower_quantized_scales_without_config_errors() {
  // A `.scales` sibling with no resolvable quantization config is a typed error
  // (the weights say quantized, the config says dense) — never a silent dense
  // reinterpret or a guessed scheme.
  let cfg = clap_config(); // no quantization block
  let mut w = clap_text_weights();
  quantize_weight_in_place(&mut w, "text_projection.linear1");
  // `from_weights` passes `quant = None`; the `.scales` on linear1 must error.
  let err = ClapTextModel::from_weights(&cfg, &mut w);
  assert!(
    err.is_err(),
    "a `.scales` sibling without a quantization config must error"
  );
}
