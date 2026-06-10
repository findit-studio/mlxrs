//! Structural tests for the LFM2.5-VL SigLIP2-style vision tower.
//!
//! Deterministic, tiny-fixture, non-gated: a `hidden = 8`, `heads = 2`,
//! `layers = 2`, `patch = 2`, `num_patches = 4` (a `2 x 2` trained position
//! grid) tower is built from synthetic weights and exercised through the public
//! [`VisionModel`] surface. These pin the patch-embed + forward output shapes,
//! the per-image bicubic position resize (active + padded patch rows), the
//! encoder truncation to the feature layer, and the 8-bit quantized load +
//! forward. Numeric parity against the reference is a later gated e2e oracle.

use std::collections::HashMap;

use super::*;
use crate::vlm::models::lfm2_vl::config::VisionConfig;

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const LAYERS: i32 = 2;
const PATCH: i32 = 2;
const CHANNELS: i32 = 3;
const INTER: i32 = 16;
const NUM_PATCHES: i32 = 4; // 2 x 2 trained position grid
const PATCH_FEAT: i32 = PATCH * PATCH * CHANNELS; // 12

/// The affine quantization group size / bit depth for the quantized fixture.
/// `HIDDEN` (8) and `PATCH_FEAT` (12) are not multiples of a realistic 32/64
/// group, so the quantized fixture below uses a config whose quantizable axes
/// are all multiples of `QGROUP`.
const QGROUP: i32 = 32;
const QBITS: i32 = 8;

/// A tiny LFM2.5-VL vision config (square `2 x 2` position grid).
fn tiny_vision_config() -> VisionConfig {
  let json = format!(
    r#"{{
      "model_type": "lfm2_vl",
      "hidden_size": {HIDDEN},
      "intermediate_size": {INTER},
      "num_hidden_layers": {LAYERS},
      "num_attention_heads": {HEADS},
      "num_channels": {CHANNELS},
      "image_size": 4,
      "patch_size": {PATCH},
      "num_patches": {NUM_PATCHES},
      "layer_norm_eps": 1e-6
    }}"#
  );
  let cfg = VisionConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// A deterministic `(rows, cols)` weight: entry `[i, j] = ((i*cols+j) % 7) *
/// 0.01 + 0.001` — small, nonzero, reproducible.
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

/// Insert `{prefix}.{q,k,v,out}_proj.{weight,bias}` (all `(hidden, hidden)`).
fn insert_attn(w: &mut HashMap<String, Array>, prefix: &str) {
  for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(format!("{prefix}.{p}.weight"), mat(HIDDEN, HIDDEN));
    w.insert(format!("{prefix}.{p}.bias"), vec1(HIDDEN));
  }
}

/// Insert `{prefix}.fc1.* / fc2.*` MLP tensors.
fn insert_mlp(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.fc1.weight"), mat(INTER, HIDDEN));
  w.insert(format!("{prefix}.fc1.bias"), vec1(INTER));
  w.insert(format!("{prefix}.fc2.weight"), mat(HIDDEN, INTER));
  w.insert(format!("{prefix}.fc2.bias"), vec1(HIDDEN));
}

/// Insert a `{prefix}.weight / .bias` LayerNorm pair (length `hidden`).
fn insert_ln(w: &mut HashMap<String, Array>, prefix: &str) {
  w.insert(format!("{prefix}.weight"), vec1(HIDDEN));
  w.insert(format!("{prefix}.bias"), vec1(HIDDEN));
}

/// Build a full, correctly-shaped dense weight map for the tiny vision tower
/// (the post-strip keys `VisionModel::from_weights` consumes). `n_layers`
/// encoder layers are emitted.
fn tiny_vision_weights(n_layers: i32) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  // Patch embed: (hidden, P^2*C) Linear weight + bias.
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
  for i in 0..n_layers {
    insert_ln(&mut w, &format!("encoder.layers.{i}.layer_norm1"));
    insert_attn(&mut w, &format!("encoder.layers.{i}.self_attn"));
    insert_ln(&mut w, &format!("encoder.layers.{i}.layer_norm2"));
    insert_mlp(&mut w, &format!("encoder.layers.{i}.mlp"));
  }
  insert_ln(&mut w, "post_layernorm");
  w
}

/// `pixel_values` for `n` images: a `(n, num_patches, P^2*C)` deterministic
/// tensor.
fn pixel_values(n: usize) -> Array {
  let total = n * NUM_PATCHES as usize * PATCH_FEAT as usize;
  let data: Vec<f32> = (0..total)
    .map(|i| ((i % 11) as f32) * 0.01 - 0.05)
    .collect();
  Array::from_slice::<f32>(&data, &(n, NUM_PATCHES as usize, PATCH_FEAT as usize)).unwrap()
}

/// `spatial_shapes` `(n, 2)` for the given per-image `(H_p, W_p)` pairs.
fn spatial_shapes(pairs: &[(i32, i32)]) -> Array {
  let mut data: Vec<i32> = Vec::new();
  for &(h, wd) in pairs {
    data.push(h);
    data.push(wd);
  }
  Array::from_slice::<i32>(&data, &(pairs.len(), 2usize)).unwrap()
}

/// A `(1, num_patches)` i32 patch attention mask: `1` for the first `active`
/// rows, `0` for the rest (one image).
fn patch_mask(active: i32, num_patches: i32) -> Array {
  let data: Vec<i32> = (0..num_patches).map(|i| (i < active) as i32).collect();
  Array::from_slice::<i32>(&data, &(1usize, num_patches as usize)).unwrap()
}

/// A `(1, num_patches)` i32 patch attention mask from explicit per-row values —
/// for constructing right-shaped but MALFORMED companions (the vision tower
/// must ignore the content and derive the mask from `spatial_shapes`).
fn patch_mask_from(vals: &[i32]) -> Array {
  Array::from_slice::<i32>(vals, &(1usize, vals.len())).unwrap()
}

/// No-op quantization resolver (every layer dense).
fn no_quant(_path: &str) -> Option<(i32, i32, &'static str)> {
  None
}

fn eval_to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

#[test]
fn patch_embed_output_shape() {
  // The patch-embed Linear alone: (1, num_patches, P^2*C) -> (1, num_patches,
  // hidden), via a 1-layer tower's patch_embedding field.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1);
  let embeds = tower.patch_embedding.forward(&pv).unwrap();
  assert_eq!(
    embeds.shape(),
    vec![1, NUM_PATCHES as usize, HIDDEN as usize],
    "patch embed (1, num_patches, hidden)"
  );
}

#[test]
fn vision_forward_full_grid_output_shape_and_finite() {
  // A full 2x2 = 4-patch grid (fills the NUM_PATCHES budget exactly) through
  // the whole tower: (1, num_patches, hidden) finite output.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1);
  let ss = spatial_shapes(&[(2, 2)]);
  let out = tower.forward(&pv, &ss, None).unwrap();
  assert_eq!(out.shape(), vec![1, NUM_PATCHES as usize, HIDDEN as usize]);
  assert!(
    eval_to_vec(&out).iter().all(|x| x.is_finite()),
    "forward output must be finite"
  );
}

#[test]
fn vision_forward_below_budget_grid_resizes_and_pads() {
  // A below-budget grid (H_p*W_p < num_patches) exercises the per-image bicubic
  // resize to a NON-square grid AND the first-row padding of the padded patch
  // rows: a (1, 3) grid = 3 active patches out of the 4-patch budget.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1);
  let ss = spatial_shapes(&[(1, 3)]);
  let out = tower.forward(&pv, &ss, None).unwrap();
  assert_eq!(out.shape(), vec![1, NUM_PATCHES as usize, HIDDEN as usize]);
  assert!(eval_to_vec(&out).iter().all(|x| x.is_finite()));
}

#[test]
fn vision_padded_mask_matches_unpadded_active_rows() {
  // The load-bearing native-resolution invariant: with the patch attention
  // mask, an image whose active grid is smaller than the patch budget
  // must produce the SAME active-patch encoder outputs as the equivalent
  // UNPADDED run — the padding (zero rows + first-resized-position) must have NO
  // effect on the active rows.
  //
  // Active grid (2, 1) = 2 active patches out of the 4-patch budget. The padded
  // run carries 4 rows (2 active + 2 padded) and a mask 1,1,0,0; the unpadded
  // run carries exactly the 2 active rows (its own 2-patch grid, no padding, no
  // mask needed). Both use the same spatial_shape (2, 1) so the per-image
  // position resize for the active rows is identical.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();

  let active = 2i32;
  let (h_p, w_p) = (2i32, 1i32); // 2 * 1 = 2 active patches
  let pf = PATCH_FEAT as usize;
  let h = HIDDEN as usize;
  let ss = spatial_shapes(&[(h_p, w_p)]);
  let mask = patch_mask(active, NUM_PATCHES);

  // Deterministic active-patch rows shared by every run.
  let active_data: Vec<f32> = (0..active as usize * pf)
    .map(|i| ((i % 11) as f32) * 0.01 - 0.05)
    .collect();

  // Helper: a full-budget pixel_values whose active rows are `active_data` and
  // whose padded rows are filled with the constant `pad`.
  let padded_pv = |pad: f32| -> Array {
    let mut data = active_data.clone();
    data.extend(std::iter::repeat_n(
      pad,
      (NUM_PATCHES - active) as usize * pf,
    ));
    Array::from_slice::<f32>(&data, &(1usize, NUM_PATCHES as usize, pf)).unwrap()
  };

  // Run A: padded + mask, padding = 7.0.
  let out_a = eval_to_vec(&tower.forward(&padded_pv(7.0), &ss, Some(&mask)).unwrap());
  // Run B: padded + mask, padding = -3.0 (DIFFERENT padding values).
  let out_b = eval_to_vec(&tower.forward(&padded_pv(-3.0), &ss, Some(&mask)).unwrap());

  // (1) Non-vacuous: with the mask, the active-row outputs are INDEPENDENT of
  // the padding row values (run A == run B on the active rows) — a weight-
  // magnitude-independent proof the mask isolates the active patches from the
  // padding. Without the mask the padded keys would feed into the active
  // queries' attention and these would differ.
  for row in 0..active as usize {
    for c in 0..h {
      let a = out_a[row * h + c];
      let b = out_b[row * h + c];
      assert!(
        (a - b).abs() < 1e-5,
        "active row {row} chan {c}: masked active output must NOT depend on padding \
         (pad=7 {a} vs pad=-3 {b})"
      );
    }
  }

  // (2) Equivalence: the masked padded active rows equal the UNPADDED run (its
  // own active-only 2-patch grid; no padding ⇒ no mask) — the padding has no
  // effect on the active rows.
  let pv_unpadded = Array::from_slice::<f32>(&active_data, &(1usize, active as usize, pf)).unwrap();
  let out_unpadded = tower.forward(&pv_unpadded, &ss, None).unwrap();
  assert_eq!(
    out_unpadded.shape(),
    vec![1, active as usize, HIDDEN as usize]
  );
  let unpadded_v = eval_to_vec(&out_unpadded);
  for row in 0..active as usize {
    for c in 0..h {
      let pv = out_a[row * h + c];
      let uv = unpadded_v[row * h + c];
      assert!(
        (pv - uv).abs() < 1e-4,
        "active row {row} chan {c}: padded+mask {pv} vs unpadded {uv} (mask must zero padding's effect)"
      );
    }
  }
}

#[test]
fn vision_mask_shape_mismatch_is_typed_error() {
  // A pixel_attention_mask whose shape disagrees with (N, num_patches) is a
  // typed RankMismatch, never a panic or a silent SDPA broadcast.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1); // (1, NUM_PATCHES, PATCH_FEAT)
  let ss = spatial_shapes(&[(2, 2)]);
  // Wrong trailing dim: (1, NUM_PATCHES + 1).
  let bad = patch_mask(2, NUM_PATCHES + 1);
  let err = tower.forward(&pv, &ss, Some(&bad)).unwrap_err();
  assert!(
    matches!(err, crate::error::Error::RankMismatch(_)),
    "got {err}"
  );
}

#[test]
fn vision_mask_content_ignored_derived_from_spatial_shapes() {
  // The KEY soundness invariant: the additive attention mask is DERIVED from
  // `spatial_shapes` (the source of truth), NOT from the companion
  // `pixel_attention_mask`'s content. A right-shaped but MALFORMED companion —
  // zeros in the active prefix, ones in the padding, or all-zero — must NOT
  // corrupt the result: every such companion, driven by the same spatial_shape,
  // produces output IDENTICAL to the well-formed-companion run.
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();

  let active = 2i32;
  let (h_p, w_p) = (2i32, 1i32); // 2 active patches out of NUM_PATCHES (4)
  let ss = spatial_shapes(&[(h_p, w_p)]);
  let pv = pixel_values(1);

  // Reference run: a well-formed companion (1,1,0,0) — active prefix set,
  // padding clear. The mask is spatial_shapes-derived, so this is the
  // ground-truth output for the (2,1) grid.
  let reference = eval_to_vec(
    &tower
      .forward(&pv, &ss, Some(&patch_mask(active, NUM_PATCHES)))
      .unwrap(),
  );

  // Three malformed-but-right-shaped companions that, if the content were
  // trusted, would each corrupt attention differently:
  //   - zeros in the active prefix + ones in the padding (fully inverted),
  //   - all-zero (would all-mask every key → NaN softmax rows if trusted),
  //   - ones in the padding only (re-admits padded keys if trusted).
  for malformed in [
    patch_mask_from(&[0, 0, 1, 1]), // inverted: active cleared, padding set
    patch_mask_from(&[0, 0, 0, 0]), // all-zero
    patch_mask_from(&[1, 1, 1, 1]), // padding re-admitted
  ] {
    let got = eval_to_vec(&tower.forward(&pv, &ss, Some(&malformed)).unwrap());
    assert_eq!(
      got.len(),
      reference.len(),
      "output length must be stable regardless of (ignored) companion content"
    );
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
      assert!(
        (g - r).abs() < 1e-6,
        "elem {i}: malformed companion content must NOT change the output \
         (derived-from-spatial_shapes mask): got {g} vs reference {r}"
      );
      // The reference output is itself finite (no all-masked NaN row): a sanity
      // pin that the derived mask leaves at least the active keys unmasked.
      assert!(r.is_finite(), "reference output elem {i} must be finite");
    }
  }
}

#[test]
fn vision_rejects_out_of_range_spatial_shapes() {
  // A spatial_shapes entry whose active grid H_p*W_p exceeds the patch budget
  // (num_patches) is a typed OutOfRange, never a panic / OOB slice — even though
  // its (N, 2) SHAPE is valid. (3*2 = 6 > NUM_PATCHES = 4.)
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1);
  let ss = spatial_shapes(&[(3, 2)]); // 6 > 4 patch budget
  let err = tower
    .forward(&pv, &ss, Some(&patch_mask(4, NUM_PATCHES)))
    .unwrap_err();
  assert!(
    matches!(err, crate::error::Error::OutOfRange(_)),
    "got {err}"
  );

  // A non-positive grid dimension (0 cols) is likewise a typed error.
  let ss_zero = spatial_shapes(&[(2, 0)]);
  let err0 = tower.forward(&pv, &ss_zero, None).unwrap_err();
  assert!(
    matches!(err0, crate::error::Error::OutOfRange(_)),
    "got {err0}"
  );
}

#[test]
fn vision_forward_multi_image_batch() {
  // Two images with different active grids in one call: the per-image position
  // resize loops over the batch, then concatenates → (2, num_patches, hidden).
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  let pv = pixel_values(2);
  let ss = spatial_shapes(&[(2, 2), (1, 3)]);
  let out = tower.forward(&pv, &ss, None).unwrap();
  assert_eq!(out.shape(), vec![2, NUM_PATCHES as usize, HIDDEN as usize]);
  assert!(eval_to_vec(&out).iter().all(|x| x.is_finite()));
}

#[test]
fn vision_encoder_truncates_to_feature_layer() {
  // layers_kept = 1 builds only one encoder layer; weights for a single layer
  // suffice (layer 1's keys are never consumed). The output shape is unchanged
  // (truncation only changes depth).
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(1); // only layer 0's weights present
  let tower = VisionModel::from_weights(&cfg, 1, &mut w, &no_quant).unwrap();
  let pv = pixel_values(1);
  let ss = spatial_shapes(&[(2, 2)]);
  let out = tower.forward(&pv, &ss, None).unwrap();
  assert_eq!(out.shape(), vec![1, NUM_PATCHES as usize, HIDDEN as usize]);
}

#[test]
fn vision_rejects_out_of_range_layers_kept() {
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  // 3 > num_hidden_layers (2).
  let err = VisionModel::from_weights(&cfg, 3, &mut w, &no_quant).unwrap_err();
  assert!(
    matches!(err, crate::error::Error::OutOfRange(_)),
    "got {err}"
  );
}

#[test]
fn vision_missing_weight_is_typed_error() {
  let cfg = tiny_vision_config();
  let mut w = tiny_vision_weights(LAYERS);
  w.remove("post_layernorm.weight");
  let err = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap_err();
  assert!(
    matches!(err, crate::error::Error::MissingKey(_)),
    "got {err}"
  );
}

#[test]
fn vision_sanitize_drops_position_ids() {
  let mut w = tiny_vision_weights(LAYERS);
  w.insert(
    "encoder.embeddings.position_ids".to_string(),
    Array::from_slice::<i32>(&[0, 1, 2, 3], &(4usize,)).unwrap(),
  );
  VisionModel::sanitize(&mut w);
  assert!(
    !w.keys().any(|k| k.contains("position_ids")),
    "position_ids must be dropped"
  );
  // A real parameter is untouched.
  assert!(w.contains_key("post_layernorm.weight"));
}

// ───────────────────── activation dtype (reduced-precision checkpoints) ─────────────────────

/// Cast every fixture weight to `dtype` — the reduced-precision (f16/bf16)
/// checkpoint analogue of the F32 synthetic map.
fn cast_weights(w: HashMap<String, Array>, dtype: Dtype) -> HashMap<String, Array> {
  w.into_iter()
    .map(|(k, v)| (k, v.astype(dtype).unwrap()))
    .collect()
}

/// Production hands the tower the raw F32 `pixel_values` straight from the
/// NaFlex processor ([`super::super::processor::preprocess_image`] /
/// `tile_image` both build `Array::from_slice::<f32>`) — the reference casts
/// them to the patch-embedding weight dtype *inside* the tower (`vision.py`'s
/// `pixel_values.astype(target_dtype)` entry cast + the post-embeddings
/// `x.astype(...)` re-pin), so the tower must own both casts. Without them
/// MLX's upward promotion (`bf16 op f32 → f32`) silently runs the whole
/// encoder — and everything downstream of it — in F32 on an f16/bf16
/// checkpoint (the whisper f32-mel regression class, 1723a5c).
#[test]
fn f32_pixel_values_on_bf16_tower_encode_in_bf16() {
  f32_pixel_values_encode_stay_model_dtype(Dtype::BF16);
}

/// Same for f16.
#[test]
fn f32_pixel_values_on_f16_tower_encode_in_f16() {
  f32_pixel_values_encode_stay_model_dtype(Dtype::F16);
}

fn f32_pixel_values_encode_stay_model_dtype(dtype: Dtype) {
  let cfg = tiny_vision_config();
  let mut w = cast_weights(tiny_vision_weights(LAYERS), dtype);
  let tower = VisionModel::from_weights(&cfg, LAYERS, &mut w, &no_quant).unwrap();
  // The pixel values exactly as the production processor produces them: F32,
  // NOT pre-cast by the caller.
  let pv = pixel_values(1);
  assert_eq!(
    pv.dtype().unwrap(),
    Dtype::F32,
    "precondition: the processor's pixel_values are F32"
  );

  // Full-budget grid (no padding ⇒ the mask is skipped): the pure dtype flow.
  let out = tower.forward(&pv, &spatial_shapes(&[(2, 2)]), None).unwrap();
  assert_eq!(out.shape(), vec![1, NUM_PATCHES as usize, HIDDEN as usize]);
  assert_eq!(
    out.dtype().unwrap(),
    dtype,
    "the tower must cast the F32 pixel_values to the patch-embedding weight \
     dtype (vision.py's target_dtype casts) — the encoder must not promote to \
     F32 on a reduced-precision checkpoint"
  );

  // Below-budget grid + companion mask: the masked-SDPA path. The additive
  // key mask must be built in the computation dtype — mlx's fast SDPA rejects
  // a mask dtype that cannot promote to the output dtype without widening it,
  // so an F32 mask against f16/bf16 activations is a hard backend error.
  let out_masked = tower
    .forward(
      &pv,
      &spatial_shapes(&[(1, 3)]),
      Some(&patch_mask(3, NUM_PATCHES)),
    )
    .unwrap();
  assert_eq!(
    out_masked.dtype().unwrap(),
    dtype,
    "the masked (padded-grid) path must also stay in the model dtype"
  );
  assert!(
    eval_to_vec(&out_masked.astype(Dtype::F32).unwrap())
      .iter()
      .all(|x| x.is_finite()),
    "masked reduced-precision forward must be finite"
  );
}

// ───────────────────── quantized-checkpoint loading ─────────────────────
//
// No local 8-bit checkpoint is available, so the quantized path is covered by a
// SYNTHETIC quantized fixture: a tiny tower whose quantizable axes are all
// multiples of the affine `group_size`, with every `nn.Linear` weight replaced
// by the real `ops::quantized::quantize` `(weight, scales, biases)` triple —
// the exact on-disk layout an mlx-community 8-bit checkpoint stores. The
// position-embedding table stays dense (MLX quantizes `nn.Linear` only, never
// `nn.Embedding` rows it gathers-then-resizes here). The tower must construct
// the quantized Linears and run a finite forward.

/// A tiny quantized-fixture config: `hidden = intermediate = QGROUP` and
/// `patch_size` chosen so `P^2*C = QGROUP` — every quantizable Linear's input
/// axis is a whole number of `QGROUP` groups. `num_patches = 4` (2x2 grid).
fn quant_vision_config() -> VisionConfig {
  // hidden = 32, intermediate = 32. patch_size = 2, channels = 8 ⇒ P^2*C = 32.
  let json = r#"{
    "model_type": "lfm2_vl",
    "hidden_size": 32,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_channels": 8,
    "image_size": 4,
    "patch_size": 2,
    "num_patches": 4,
    "layer_norm_eps": 1e-6
  }"#;
  let cfg = VisionConfig::from_json(json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Replace the dense `{prefix}.weight` with the real 8-bit affine quantize
/// triple (`{prefix}.weight` packed + `.scales` + `.biases`).
fn quantize_weight_in_place(w: &mut HashMap<String, Array>, prefix: &str) {
  let dense = w
    .remove(&format!("{prefix}.weight"))
    .unwrap_or_else(|| panic!("dense weight present for {prefix}"));
  let (w_q, scales, biases) =
    crate::ops::quantized::quantize(&dense, QGROUP, QBITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), w_q);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine produces per-group biases"),
  );
}

/// The quantized fixture's dense weight map (config `quant_vision_config`):
/// hidden = inter = 32, P^2*C = 32, 1 encoder layer.
fn quant_dense_weights() -> HashMap<String, Array> {
  let h = 32i32;
  let inter = 32i32;
  let patch_feat = 32i32; // P^2*C
  let num_patches = 4i32;
  let mut w = HashMap::new();
  w.insert(
    "embeddings.patch_embedding.weight".to_string(),
    mat(h, patch_feat),
  );
  w.insert("embeddings.patch_embedding.bias".to_string(), vec_n(h));
  w.insert(
    "embeddings.position_embedding.weight".to_string(),
    mat(num_patches, h),
  );
  // One encoder layer (hidden=32).
  let p = "encoder.layers.0";
  w.insert(format!("{p}.layer_norm1.weight"), vec_n(h));
  w.insert(format!("{p}.layer_norm1.bias"), vec_n(h));
  w.insert(format!("{p}.layer_norm2.weight"), vec_n(h));
  w.insert(format!("{p}.layer_norm2.bias"), vec_n(h));
  for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    w.insert(format!("{p}.self_attn.{proj}.weight"), mat(h, h));
    w.insert(format!("{p}.self_attn.{proj}.bias"), vec_n(h));
  }
  w.insert(format!("{p}.mlp.fc1.weight"), mat(inter, h));
  w.insert(format!("{p}.mlp.fc1.bias"), vec_n(inter));
  w.insert(format!("{p}.mlp.fc2.weight"), mat(h, inter));
  w.insert(format!("{p}.mlp.fc2.bias"), vec_n(h));
  w.insert("post_layernorm.weight".to_string(), vec_n(h));
  w.insert("post_layernorm.bias".to_string(), vec_n(h));
  w
}

/// A length-`n` deterministic vector (distinct from `vec1`, which is fixed to
/// the small `HIDDEN`).
fn vec_n(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize)
    .map(|i| 0.01 + (i % 7) as f32 * 0.005)
    .collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

#[test]
fn vision_quantized_layer_loads_and_forwards() {
  let cfg = quant_vision_config();
  let mut w = quant_dense_weights();
  // Quantize every nn.Linear: patch embed, the four attn projections, fc1/fc2.
  quantize_weight_in_place(&mut w, "embeddings.patch_embedding");
  for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
    quantize_weight_in_place(&mut w, &format!("encoder.layers.0.self_attn.{proj}"));
  }
  quantize_weight_in_place(&mut w, "encoder.layers.0.mlp.fc1");
  quantize_weight_in_place(&mut w, "encoder.layers.0.mlp.fc2");

  // The per-layer quant resolver: every Linear path uses the 8-bit affine
  // scheme (the patch embed + attn + mlp; the position embed is never asked).
  let quant = |_path: &str| -> Option<(i32, i32, &'static str)> { Some((QGROUP, QBITS, "affine")) };

  let tower = VisionModel::from_weights(&cfg, 1, &mut w, &quant).unwrap();
  // Every routed Linear is the quantized variant.
  assert!(
    tower.patch_embedding.is_quantized(),
    "patch_embedding quantized"
  );
  assert!(
    tower.layers[0].self_attn.q_proj.is_quantized(),
    "q_proj quantized"
  );
  assert!(tower.layers[0].mlp.fc1.is_quantized(), "fc1 quantized");

  // Forward a synthetic image (patch_feat = 32) through the quantized tower.
  let total = 4 * 32; // num_patches * P^2*C
  let data: Vec<f32> = (0..total)
    .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
    .collect();
  let pv = Array::from_slice::<f32>(&data, &(1usize, 4usize, 32usize)).unwrap();
  let ss = spatial_shapes(&[(2, 2)]);
  let out = tower.forward(&pv, &ss, None).unwrap();
  assert_eq!(out.shape(), vec![1, 4, 32]);
  assert!(
    eval_to_vec(&out).iter().all(|x| x.is_finite()),
    "quantized forward output must be finite"
  );
}
