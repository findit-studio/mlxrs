//! Structural tests for the SigLIP2 NaFlex vision tower.
//!
//! Deterministic, tiny-fixture, non-gated: a `hidden = 8`, `heads = 2`,
//! `layers = 1`, `patch = 2`, `num_patches = 4` (a `2 x 2` trained position
//! grid) vision tower is built from synthetic weights and exercised through
//! the public [`VisionTower`] surface. These pin the output shapes, the
//! per-image bicubic pos-embed resize path, the attention-pool head shape, and
//! a couple of consumed-weight shape-mismatch rejections. Numeric parity is
//! covered by the gated e2e oracle in the crate-level `tests/`.

use std::collections::HashMap;

use super::*;
use crate::embeddings::siglip2_naflex::{config::VisionConfig, processing::preprocess};

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const LAYERS: i32 = 1;
const PATCH: i32 = 2;
const CHANNELS: i32 = 3;
const INTER: i32 = 16;
const NUM_PATCHES: i32 = 4; // 2 x 2 trained position grid
const PATCH_FEAT: i32 = PATCH * PATCH * CHANNELS; // 12

/// A tiny NaFlex vision config (square `2 x 2` position grid, `vision_use_head
/// = true`). `max_num_patches` matches `num_patches` so the preprocessing
/// budget aligns with the trained grid.
fn tiny_vision_config(use_head: bool) -> VisionConfig {
  let json = format!(
    r#"{{
      "model_type": "siglip_vision_model",
      "image_size": 4,
      "patch_size": {PATCH},
      "num_channels": {CHANNELS},
      "hidden_size": {HIDDEN},
      "intermediate_size": {INTER},
      "num_attention_heads": {HEADS},
      "num_hidden_layers": {LAYERS},
      "layer_norm_eps": 1e-6,
      "vision_use_head": {use_head},
      "num_patches": {NUM_PATCHES},
      "max_num_patches": {NUM_PATCHES}
    }}"#
  );
  let cfg = VisionConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// A deterministic `(rows, cols)` weight: entry `[i, j] = ((i * cols + j) %
/// 7) * 0.01 + 0.001` — small, nonzero, reproducible.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A deterministic rank-1 bias of length `n`.
fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize).map(|i| ((i % 5) as f32) * 0.01).collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// Insert the `{prefix}.{q,k,v,out}_proj.{weight,bias}` attention tensors
/// (all `(hidden, hidden)` / `(hidden,)`).
fn insert_attn(w: &mut HashMap<String, Array>, prefix: &str) {
  for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(format!("{prefix}.{p}.weight"), mat(HIDDEN, HIDDEN));
    w.insert(format!("{prefix}.{p}.bias"), vec1(HIDDEN));
  }
}

/// Insert the `{prefix}.fc1.* / fc2.*` MLP tensors.
fn insert_mlp(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.fc1.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{prefix}.fc1.bias"), vec1(INTER));
  w.insert(format!("{prefix}.fc2.weight"), mat(HIDDEN, INTER));
  w.insert(format!("{prefix}.fc2.bias"), vec1(HIDDEN));
}

/// Insert a `{prefix}.weight / .bias` LayerNorm pair.
fn insert_ln(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.weight"), vec1(HIDDEN));
  w.insert(format!("{prefix}.bias"), vec1(HIDDEN));
}

/// Build a full, correctly-shaped weight map for the tiny vision tower (the
/// post-strip keys `VisionTower::from_weights` consumes).
fn tiny_vision_weights(use_head: bool) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  // Patch embed: flattened (hidden, P^2*C) form.
  w.insert(
    "embeddings.patch_embedding.weight".to_string(),
    mat(HIDDEN, PATCH_FEAT),
  );
  w.insert("embeddings.patch_embedding.bias".to_string(), vec1(HIDDEN));
  // Position embedding table (num_positions, hidden).
  w.insert(
    "embeddings.position_embedding.weight".to_string(),
    mat(NUM_PATCHES, HIDDEN),
  );
  // One encoder layer.
  insert_ln(&mut w, "encoder.layers.0.layer_norm1");
  insert_attn(&mut w, "encoder.layers.0.self_attn");
  insert_ln(&mut w, "encoder.layers.0.layer_norm2");
  insert_mlp(&mut w, "encoder.layers.0.mlp");
  // Post LayerNorm.
  insert_ln(&mut w, "post_layernorm");
  // Attention-pool head.
  if use_head {
    w.insert(
      "head.probe".to_string(),
      Array::from_slice::<f32>(
        &(0..HIDDEN as usize)
          .map(|i| (i as f32) * 0.01 + 0.01)
          .collect::<Vec<_>>(),
        &(1usize, 1usize, HIDDEN as usize),
      )
      .unwrap(),
    );
    w.insert(
      "head.attention.in_proj.weight".to_string(),
      mat(3 * HIDDEN, HIDDEN),
    );
    w.insert("head.attention.in_proj.bias".to_string(), vec1(3 * HIDDEN));
    w.insert(
      "head.attention.out_proj.weight".to_string(),
      mat(HIDDEN, HIDDEN),
    );
    w.insert("head.attention.out_proj.bias".to_string(), vec1(HIDDEN));
    insert_ln(&mut w, "head.layernorm");
    insert_mlp(&mut w, "head.mlp");
  }
  w
}

/// Preprocess a synthetic image to `NaflexInputs` at the tiny config's budget.
/// `w`/`h` are pixel dims; the tower's `max_num_patches` is `NUM_PATCHES`.
fn tiny_inputs(w: u32, h: u32) -> NaflexInputs {
  let rgb = vec![100u8; (w * h * 3) as usize];
  preprocess(
    &rgb,
    w,
    h,
    PATCH as u32,
    CHANNELS as u32,
    NUM_PATCHES as u32,
  )
  .unwrap()
}

fn eval_shape(a: &Array) -> Vec<usize> {
  a.shape()
}

fn eval_to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

#[test]
fn vision_tower_output_shapes_square_grid() {
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  // A 4x4 image -> 2x2 = 4 patches (fills the NUM_PATCHES budget exactly).
  let inputs = tiny_inputs(4, 4);
  let ss = {
    let mut s = inputs.spatial_shapes.try_clone().unwrap();
    s.eval().unwrap();
    s.to_vec::<i32>().unwrap()
  };
  assert_eq!(ss, vec![2, 2], "4x4 -> 2x2 grid");
  let (last_hidden, pooled) = tower.forward(&inputs).unwrap();
  // last_hidden: (1, num_patches, hidden).
  assert_eq!(
    eval_shape(&last_hidden),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize]
  );
  // pooled: (1, hidden).
  let pooled = pooled.expect("vision_use_head -> pooled output");
  assert_eq!(eval_shape(&pooled), vec![1, HIDDEN as usize]);
  // The output must be finite (the resize + encoder produced real numbers).
  assert!(eval_to_vec(&pooled).iter().all(|x| x.is_finite()));
}

#[test]
fn vision_tower_output_shapes_below_budget_grid() {
  // A genuinely below-budget grid (H_p*W_p < NUM_PATCHES) exercises the
  // per-image bicubic resize to a NON-square grid AND the zero-padding of the
  // padded patch rows. NaFlex's preprocessing tends to upsample to FILL the
  // budget, so a below-budget input is built directly here: a (1, 3) grid =
  // 3 active patches out of the 4-patch budget. The `pixel_values` /
  // `pixel_attention_mask` / `spatial_shapes` are constructed to match.
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  let inputs = below_budget_inputs(1, 3);
  let (last_hidden, pooled) = tower.forward(&inputs).unwrap();
  assert_eq!(
    eval_shape(&last_hidden),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize]
  );
  let pooled = pooled.expect("vision_use_head -> pooled output");
  assert_eq!(eval_shape(&pooled), vec![1, HIDDEN as usize]);
  assert!(eval_to_vec(&pooled).iter().all(|x| x.is_finite()));
}

/// Build a [`NaflexInputs`] for a chosen below-budget `(h_p, w_p)` grid:
/// `pixel_values` is `(NUM_PATCHES, P^2*C)` with the first `h_p*w_p` rows
/// nonzero and the rest zero-padded; `pixel_attention_mask` is `1` for the
/// first `h_p*w_p` rows and `0` after; `spatial_shapes` is `[h_p, w_p]`.
fn below_budget_inputs(h_p: i32, w_p: i32) -> NaflexInputs {
  let active = (h_p * w_p) as usize;
  assert!(active < NUM_PATCHES as usize, "must be below budget");
  let per_patch = PATCH_FEAT as usize;
  let total = NUM_PATCHES as usize * per_patch;
  let mut pv = vec![0.0f32; total];
  for (i, slot) in pv.iter_mut().enumerate().take(active * per_patch) {
    *slot = ((i % 7) as f32) * 0.01 + 0.01;
  }
  let pixel_values = Array::from_slice::<f32>(&pv, &(NUM_PATCHES as usize, per_patch)).unwrap();
  let mut mask = vec![0i32; NUM_PATCHES as usize];
  for m in mask.iter_mut().take(active) {
    *m = 1;
  }
  let pixel_attention_mask = Array::from_slice::<i32>(&mask, &(NUM_PATCHES as usize,)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[h_p, w_p], &(2usize,)).unwrap();
  NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  }
}

#[test]
fn vision_tower_headless_has_no_pooled_output() {
  let cfg = tiny_vision_config(false);
  let mut w = tiny_vision_weights(false);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  assert!(!tower.has_head());
  let inputs = tiny_inputs(4, 4);
  let (last_hidden, pooled) = tower.forward(&inputs).unwrap();
  assert_eq!(
    eval_shape(&last_hidden),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize]
  );
  assert!(
    pooled.is_none(),
    "vision_use_head=false -> no pooled output"
  );
}

#[test]
fn pos_embed_resize_produces_active_grid_times_hidden() {
  // Directly probe the bicubic resize: resizing the trained 2x2 grid to a 1x2
  // active grid must produce (1*2, hidden) active rows, padded to
  // (num_patches, hidden), lifted to (1, num_patches, hidden).
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  let pos = tower
    .resized_position_embedding(1, 2, &[1, NUM_PATCHES as usize, HIDDEN as usize])
    .unwrap();
  assert_eq!(
    eval_shape(&pos),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize]
  );
  // The first 1*2 = 2 patch rows are the resized (active) positions; rows 2..4
  // are zero padding.
  let flat = eval_to_vec(&pos);
  let h = HIDDEN as usize;
  for padded_row in 2..NUM_PATCHES as usize {
    for j in 0..h {
      assert_eq!(
        flat[padded_row * h + j],
        0.0,
        "padded pos row {padded_row} idx {j}"
      );
    }
  }
  // At least one active-row value is nonzero (the resize is not all-zero).
  assert!(
    flat[..2 * h].iter().any(|&x| x != 0.0),
    "active position rows must be nonzero"
  );
}

#[test]
fn pos_embed_resize_identity_when_grid_matches_trained() {
  // Resizing the 2x2 trained grid to a 2x2 active grid is the bicubic
  // identity fast-path (out == in): the active rows equal the trained table.
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  // Capture the trained table before it's moved into the tower.
  let table = mat(NUM_PATCHES, HIDDEN);
  let table_vals = eval_to_vec(&table);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  let pos = tower
    .resized_position_embedding(2, 2, &[1, NUM_PATCHES as usize, HIDDEN as usize])
    .unwrap();
  let got = eval_to_vec(&pos);
  // All NUM_PATCHES rows are active (2*2 == 4); compare to the trained table.
  for (i, (g, t)) in got.iter().zip(table_vals.iter()).enumerate() {
    assert!((g - t).abs() < 1e-6, "pos[{i}]: got {g}, want {t}");
  }
}

#[test]
fn from_weights_rejects_wrong_patch_embed_shape() {
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  // Wrong flattened width (PATCH_FEAT + 1).
  w.insert(
    "embeddings.patch_embedding.weight".to_string(),
    mat(HIDDEN, PATCH_FEAT + 1),
  );
  let err = VisionTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "wrong patch-embed width must be rejected");
}

#[test]
fn from_weights_rejects_wrong_position_table_shape() {
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  // Wrong num_positions (NUM_PATCHES + 1).
  w.insert(
    "embeddings.position_embedding.weight".to_string(),
    mat(NUM_PATCHES + 1, HIDDEN),
  );
  let err = VisionTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "wrong position-table rows must be rejected");
}

#[test]
fn from_weights_rejects_missing_attention_weight() {
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  w.remove("encoder.layers.0.self_attn.q_proj.weight");
  let err = VisionTower::from_weights(&cfg, &mut w);
  assert!(err.is_err(), "missing attention weight must be rejected");
}

#[test]
fn patch_embed_accepts_conv2d_rank4_weight() {
  // The Conv2d channels-last (hidden, P, P, C) form must be accepted and
  // flattened to (hidden, P^2*C) internally.
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  let conv = Array::from_slice::<f32>(
    &(0..(HIDDEN * PATCH * PATCH * CHANNELS) as usize)
      .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
      .collect::<Vec<_>>(),
    &(
      HIDDEN as usize,
      PATCH as usize,
      PATCH as usize,
      CHANNELS as usize,
    ),
  )
  .unwrap();
  w.insert("embeddings.patch_embedding.weight".to_string(), conv);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();
  let inputs = tiny_inputs(4, 4);
  let (last_hidden, _) = tower.forward(&inputs).unwrap();
  assert_eq!(
    eval_shape(&last_hidden),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize]
  );
}
