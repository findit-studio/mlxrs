//! Structural tests for the SigLIP2 NaFlex vision tower.
//!
//! Deterministic, tiny-fixture, non-gated: a `hidden = 8`, `heads = 2`,
//! `layers = 1`, `patch = 2`, `num_patches = 4` (a `2 x 2` trained position
//! grid) vision tower is built from synthetic weights and exercised through
//! the public [`VisionTower`] surface. These pin the output shapes, the
//! per-image bilinear pos-embed resize path, the attention-pool head shape, and
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
      "model_type": "siglip2_vision_model",
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
  // per-image bilinear resize to a NON-square grid AND the first-row padding of
  // the padded patch rows. NaFlex's preprocessing tends to upsample to FILL the
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
fn pos_embed_resize_produces_active_grid_with_first_row_padding() {
  // Directly probe the bilinear resize: resizing the trained 2x2 grid to a 1x2
  // active grid must produce (1*2, hidden) active rows, padded to
  // (num_patches, hidden), lifted to (1, num_patches, hidden). The padded rows
  // (2..4) take the FIRST resized position (HF's resized_embeddings[0]), not
  // zeros.
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
  let flat = eval_to_vec(&pos);
  let h = HIDDEN as usize;
  // Every padded row equals the first active row (the first resized position).
  for padded_row in 2..NUM_PATCHES as usize {
    for j in 0..h {
      assert!(
        (flat[padded_row * h + j] - flat[j]).abs() < 1e-6,
        "padded pos row {padded_row} idx {j} must equal resized[0]: \
         got {} want {}",
        flat[padded_row * h + j],
        flat[j]
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
  // Resizing the 2x2 trained grid to a 2x2 active grid is the bilinear
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
fn from_weights_num_hidden_layers_over_weights_is_typed_error() {
  // `mlxrs` is a library: `num_hidden_layers` carries no magnitude cap (the
  // consuming application owns input bounding). A config asking for MORE layers
  // than the checkpoint provides still fails GRACEFULLY with a typed error —
  // the per-layer `take` reports the first missing layer key as
  // `Error::MissingKey`, never a panic. (`tiny_vision_weights` supplies only
  // `encoder.layers.0.*`, so `num_hidden_layers = 2` misses layer 1.)
  let json = format!(
    r#"{{
      "model_type": "siglip2_vision_model",
      "image_size": 4,
      "patch_size": {PATCH},
      "num_channels": {CHANNELS},
      "hidden_size": {HIDDEN},
      "intermediate_size": {INTER},
      "num_attention_heads": {HEADS},
      "num_hidden_layers": 2,
      "layer_norm_eps": 1e-6,
      "vision_use_head": true,
      "num_patches": {NUM_PATCHES},
      "max_num_patches": {NUM_PATCHES}
    }}"#
  );
  let cfg = VisionConfig::from_json(&json).unwrap();
  let mut w = tiny_vision_weights(true);
  assert_eq!(cfg.num_hidden_layers, 2); // one more layer than the weights supply
  let err = VisionTower::from_weights(&cfg, &mut w).err();
  assert!(
    matches!(err, Some(Error::MissingKey(_))),
    "an over-the-weights num_hidden_layers must be a typed MissingKey, got {err:?}"
  );
}

#[test]
fn forward_rejects_mask_length_patch_dim_mismatch() {
  // A `NaflexInputs` whose `pixel_attention_mask` length disagrees with the
  // `pixel_values` leading patch dim is malformed; `forward` must reject it
  // with the typed exact-shape pin (not rely on the SDPA broadcast). The mask is
  // rank-1 with a wrong length, so it is a ShapePairMismatch against the
  // expected `(patch_dim,)`.
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  let tower = VisionTower::from_weights(&cfg, &mut w).unwrap();

  // pixel_values: (NUM_PATCHES, PATCH_FEAT); mask: deliberately one element
  // too short (NUM_PATCHES - 1); spatial_shapes a valid 2x2 grid.
  let per_patch = PATCH_FEAT as usize;
  let pv = vec![0.01f32; NUM_PATCHES as usize * per_patch];
  let pixel_values = Array::from_slice::<f32>(&pv, &(NUM_PATCHES as usize, per_patch)).unwrap();
  let short_mask = vec![1i32; NUM_PATCHES as usize - 1];
  let pixel_attention_mask =
    Array::from_slice::<i32>(&short_mask, &(NUM_PATCHES as usize - 1,)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::ShapePairMismatch(_))),
    "mask/patch-dim length mismatch must be a typed ShapePairMismatch, got {res:?}"
  );
}

/// Build the tiny tower once for a runtime-gate rejection test.
fn tiny_tower() -> VisionTower {
  let cfg = tiny_vision_config(true);
  let mut w = tiny_vision_weights(true);
  VisionTower::from_weights(&cfg, &mut w).unwrap()
}

#[test]
fn forward_rejects_rank3_pixel_values() {
  // A rank-3 `pixel_values` (an extra leading axis) must be rejected as a typed
  // RankMismatch BEFORE any op — a leading-dim-only gate would wave it through
  // (treating `shape[0]` as the patch dim) into the patch-embed / SDPA graph.
  let tower = tiny_tower();
  let per_patch = PATCH_FEAT as usize;
  let pv = vec![0.01f32; NUM_PATCHES as usize * per_patch];
  let pixel_values =
    Array::from_slice::<f32>(&pv, &(1usize, NUM_PATCHES as usize, per_patch)).unwrap();
  let pixel_attention_mask =
    Array::from_slice::<i32>(&vec![1i32; NUM_PATCHES as usize], &(NUM_PATCHES as usize,)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::RankMismatch(_))),
    "rank-3 pixel_values must be a typed RankMismatch, got {res:?}"
  );
}

#[test]
fn forward_rejects_wrong_pixel_values_feature_width() {
  // A rank-2 `pixel_values` whose last axis != the loaded patch_feature_dim is a
  // typed ShapePairMismatch (the full-shape pin), not a silent wrong matmul.
  let tower = tiny_tower();
  let bad_feat = PATCH_FEAT as usize + 1;
  let pv = vec![0.01f32; NUM_PATCHES as usize * bad_feat];
  let pixel_values = Array::from_slice::<f32>(&pv, &(NUM_PATCHES as usize, bad_feat)).unwrap();
  let pixel_attention_mask =
    Array::from_slice::<i32>(&vec![1i32; NUM_PATCHES as usize], &(NUM_PATCHES as usize,)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::ShapePairMismatch(_))),
    "wrong pixel_values feature width must be a typed ShapePairMismatch, got {res:?}"
  );
}

#[test]
fn forward_rejects_rank2_pixel_attention_mask_with_trailing_axis() {
  // A `pixel_attention_mask` shaped `(patch_dim, huge)` (a trailing junk axis,
  // correct leading dim) must be rejected by the exact rank-1 pin — a
  // leading-dim-only check would pass it (`shape[0] == patch_dim`) into the SDPA
  // mask path. The extra/rank-mismatched-axis evasion the exact gate closes.
  let tower = tiny_tower();
  let per_patch = PATCH_FEAT as usize;
  let pv = vec![0.01f32; NUM_PATCHES as usize * per_patch];
  let pixel_values = Array::from_slice::<f32>(&pv, &(NUM_PATCHES as usize, per_patch)).unwrap();
  // (NUM_PATCHES, 5) mask: leading dim matches the patch dim, trailing junk axis.
  let bad_mask = vec![1i32; NUM_PATCHES as usize * 5];
  let pixel_attention_mask =
    Array::from_slice::<i32>(&bad_mask, &(NUM_PATCHES as usize, 5usize)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::RankMismatch(_))),
    "rank-2 pixel_attention_mask (trailing junk axis) must be a typed RankMismatch, got {res:?}"
  );
}

#[test]
fn forward_rejects_wrong_rank_spatial_shapes() {
  // `spatial_shapes` must be exactly rank-1 `(2,)`. A rank-2 (or wrong-length)
  // shape is rejected at the gate, before any op — not only later in the
  // host-side `read_spatial_shape`.
  let tower = tiny_tower();
  let per_patch = PATCH_FEAT as usize;
  let pv = vec![0.01f32; NUM_PATCHES as usize * per_patch];
  let pixel_values = Array::from_slice::<f32>(&pv, &(NUM_PATCHES as usize, per_patch)).unwrap();
  let pixel_attention_mask =
    Array::from_slice::<i32>(&vec![1i32; NUM_PATCHES as usize], &(NUM_PATCHES as usize,)).unwrap();
  // (1, 2) spatial_shapes (rank-2) instead of (2,).
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(1usize, 2usize)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::RankMismatch(_))),
    "wrong-rank spatial_shapes must be a typed RankMismatch, got {res:?}"
  );
}

#[test]
fn forward_rejects_patch_dim_above_max_num_patches() {
  // A runtime patch count above the loaded `max_num_patches` budget is the DoS
  // boundary: it must surface as the dedicated CapExceeded (not a generic shape
  // mismatch), before the position-embed scatter / O(patch^2) SDPA. Build a
  // `(NUM_PATCHES + 1, PATCH_FEAT)` pixel tensor with a matching-length mask so
  // the only violated bound is the patch-count cap.
  let tower = tiny_tower();
  let over = NUM_PATCHES as usize + 1;
  let per_patch = PATCH_FEAT as usize;
  let pv = vec![0.01f32; over * per_patch];
  let pixel_values = Array::from_slice::<f32>(&pv, &(over, per_patch)).unwrap();
  let pixel_attention_mask = Array::from_slice::<i32>(&vec![1i32; over], &(over,)).unwrap();
  let spatial_shapes = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let inputs = NaflexInputs {
    pixel_values,
    pixel_attention_mask,
    spatial_shapes,
  };
  let res = tower.forward(&inputs);
  assert!(
    matches!(res, Err(Error::CapExceeded(_))),
    "patch_dim > max_num_patches must be a typed CapExceeded, got {res:?}"
  );
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

#[test]
fn patch_embed_accepts_raw_pytorch_conv2d_weight_same_as_channels_last() {
  // A RAW PyTorch / HF Conv2d patch weight is `(hidden, C, P, P)`
  // (`nn.Conv2d`'s `(out, in, kH, kW)`). It must be accepted and produce the
  // SAME patch projection as the equivalent MLX channels-last `(hidden, P, P, C)`
  // weight (the loader transposes `[0, 2, 3, 1]` internally). Oracle: build the
  // channels-last weight, transpose it to the PyTorch layout for the second
  // tower, and assert both towers' `last_hidden` match — proving the raw HF
  // checkpoint layout is loaded equivalently (the channels-last comparison is the
  // independent reference, not the code under test).
  let cfg = tiny_vision_config(true);

  // The MLX channels-last conv weight (hidden, P, P, C), shared by both towers.
  let channels_last = Array::from_slice::<f32>(
    &(0..(HIDDEN * PATCH * PATCH * CHANNELS) as usize)
      .map(|n| ((n % 7) as f32) * 0.013 + 0.002)
      .collect::<Vec<_>>(),
    &(
      HIDDEN as usize,
      PATCH as usize,
      PATCH as usize,
      CHANNELS as usize,
    ),
  )
  .unwrap();

  // Tower A: channels-last weight loaded directly.
  let mut wa = tiny_vision_weights(true);
  wa.insert(
    "embeddings.patch_embedding.weight".to_string(),
    channels_last.try_clone().unwrap(),
  );
  let tower_a = VisionTower::from_weights(&cfg, &mut wa).unwrap();

  // Tower B: the SAME weight transposed to the raw PyTorch `(hidden, C, P, P)`
  // layout (the inverse of the loader's `[0, 2, 3, 1]`: channels-last
  // `(hidden, P, P, C)` -> PyTorch `(hidden, C, P, P)` is `transpose(0, 3, 1, 2)`).
  let mut pytorch = ops::shape::transpose_axes(&channels_last, &[0, 3, 1, 2]).unwrap();
  pytorch.eval().unwrap();
  assert_eq!(
    pytorch.shape(),
    vec![
      HIDDEN as usize,
      CHANNELS as usize,
      PATCH as usize,
      PATCH as usize
    ],
    "the second tower's weight is in raw PyTorch (hidden, C, P, P) layout"
  );
  let mut wb = tiny_vision_weights(true);
  wb.insert("embeddings.patch_embedding.weight".to_string(), pytorch);
  let tower_b = VisionTower::from_weights(&cfg, &mut wb).unwrap();

  // Both towers must produce the identical patch projection.
  let inputs = tiny_inputs(4, 4);
  let (last_a, _) = tower_a.forward(&inputs).unwrap();
  let (last_b, _) = tower_b.forward(&inputs).unwrap();
  assert_eq!(eval_shape(&last_a), eval_shape(&last_b));
  let va = eval_to_vec(&last_a);
  let vb = eval_to_vec(&last_b);
  assert_eq!(va.len(), vb.len());
  for (i, (a, b)) in va.iter().zip(vb.iter()).enumerate() {
    assert!(
      (a - b).abs() < 1e-5,
      "raw-PyTorch vs channels-last patch projection diverge at {i}: {a} vs {b}"
    );
  }
}
