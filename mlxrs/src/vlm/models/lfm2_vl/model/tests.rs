//! End-to-end assembly tests for the top-level LFM2.5-VL model + factory.
//!
//! Deterministic, tiny-fixture, non-gated. A synthetic checkpoint (a tiny
//! vision tower + projector + all-conv LFM2 LM, every weight built from a
//! reproducible generator) is assembled through [`Lfm2Vl::from_weights`] and
//! driven end-to-end: a prompt with `<image>` placeholders + per-image NaFlex
//! inputs → `[batch, seq, vocab]` logits; the merge-count contract; the
//! text-only path; the 8-bit quantized load + forward; and the
//! [`VlmTypeRegistry`] registration + constructor.

use std::collections::HashMap;

use super::*;
use crate::vlm::models::lfm2_vl::{config::ModelConfig, processor::Lfm2VlImageInputs};

// ───────────────────────── tiny synthetic checkpoint ─────────────────────────

const VHIDDEN: i32 = 8; // vision hidden
const VHEADS: i32 = 2;
const VLAYERS: i32 = 2;
const VPATCH: i32 = 2;
const VINTER: i32 = 16;
const VNUM_PATCHES: i32 = 4; // 2x2 trained position grid
const VPATCH_FEAT: i32 = VPATCH * VPATCH * 3; // 12
const FACTOR: i32 = 2;
const PROJ_IN: i32 = VHIDDEN * FACTOR * FACTOR; // 32
const PROJ_HIDDEN: i32 = 16;
const THIDDEN: i32 = 4; // text hidden
const TLAYERS: i32 = 2; // all-conv LFM2 layers
const TFF: i32 = 8;
const VOCAB: i32 = 32;
/// The `<image>` placeholder id — an in-vocab row of the tiny `VOCAB`-wide
/// embedding (the real checkpoint's 396 is in-range of its 65536 vocab; the
/// merge overwrites these positions regardless).
const IMG: i32 = 7;

/// A deterministic `(rows, cols)` weight: entry `[i, j] = ((i*cols+j) % 7) *
/// 0.01 + 0.001` — small, nonzero, reproducible.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A deterministic rank-1 vector of length `n`.
fn vec1(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize)
    .map(|i| 0.01 + (i % 5) as f32 * 0.005)
    .collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// The tiny LFM2.5-VL model config JSON (all-conv text tower, 2x2 position
/// grid, downsample_factor 2). `quantization` optionally present.
fn tiny_config_json(quantized: bool) -> String {
  let quant_block = if quantized {
    r#""quantization": {"group_size": 32, "bits": 8, "mode": "affine"},"#
  } else {
    ""
  };
  // The `<image>` placeholder id must be a valid embedding row of the tiny
  // vocab (the language adapter range-guards token ids before the gather; the
  // merge then overwrites the `<image>` positions). The real checkpoint's 396
  // is in-range of its 65536 vocab; the tiny fixture uses an in-vocab id.
  format!(
    r#"{{
      "model_type": "lfm2-vl",
      {quant_block}
      "downsample_factor": {FACTOR},
      "image_token_index": {IMG},
      "projector_hidden_size": {PROJ_HIDDEN},
      "projector_use_layernorm": true,
      "projector_bias": true,
      "vision_feature_layer": -1,
      "text_config": {{
        "model_type": "lfm2",
        "hidden_size": {THIDDEN},
        "num_hidden_layers": {TLAYERS},
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": {VOCAB},
        "block_dim": {THIDDEN},
        "block_ff_dim": {TFF},
        "block_auto_adjust_ff_dim": false,
        "block_multiple_of": 1,
        "conv_L_cache": 3,
        "layer_types": ["conv", "conv"]
      }},
      "vision_config": {{
        "model_type": "lfm2_vl",
        "hidden_size": {VHIDDEN},
        "intermediate_size": {VINTER},
        "num_hidden_layers": {VLAYERS},
        "num_attention_heads": {VHEADS},
        "num_channels": 3,
        "image_size": 4,
        "patch_size": {VPATCH},
        "num_patches": {VNUM_PATCHES},
        "layer_norm_eps": 1e-6
      }}
    }}"#
  )
}

fn tiny_config(quantized: bool) -> ModelConfig {
  let cfg = ModelConfig::from_json(&tiny_config_json(quantized)).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Insert the vision tower weights under the `vision_tower.` prefix (the
/// post-sanitize namespace `from_weights` drains).
fn insert_vision(w: &mut HashMap<String, Array>) {
  let pfx = "vision_tower";
  // patch embedding: Linear (hidden, patch_feat) + bias.
  w.insert(
    format!("{pfx}.embeddings.patch_embedding.weight"),
    mat(VHIDDEN, VPATCH_FEAT),
  );
  w.insert(
    format!("{pfx}.embeddings.patch_embedding.bias"),
    vec1(VHIDDEN),
  );
  // position embedding table (num_patches, hidden).
  w.insert(
    format!("{pfx}.embeddings.position_embedding.weight"),
    mat(VNUM_PATCHES, VHIDDEN),
  );
  for i in 0..VLAYERS {
    let lp = format!("{pfx}.encoder.layers.{i}");
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      w.insert(format!("{lp}.self_attn.{p}.weight"), mat(VHIDDEN, VHIDDEN));
      w.insert(format!("{lp}.self_attn.{p}.bias"), vec1(VHIDDEN));
    }
    w.insert(format!("{lp}.mlp.fc1.weight"), mat(VINTER, VHIDDEN));
    w.insert(format!("{lp}.mlp.fc1.bias"), vec1(VINTER));
    w.insert(format!("{lp}.mlp.fc2.weight"), mat(VHIDDEN, VINTER));
    w.insert(format!("{lp}.mlp.fc2.bias"), vec1(VHIDDEN));
    for ln in ["layer_norm1", "layer_norm2"] {
      w.insert(format!("{lp}.{ln}.weight"), vec1(VHIDDEN));
      w.insert(format!("{lp}.{ln}.bias"), vec1(VHIDDEN));
    }
  }
  w.insert(format!("{pfx}.post_layernorm.weight"), vec1(VHIDDEN));
  w.insert(format!("{pfx}.post_layernorm.bias"), vec1(VHIDDEN));
}

/// Insert the projector weights under `multi_modal_projector.`.
fn insert_projector(w: &mut HashMap<String, Array>) {
  let pfx = "multi_modal_projector";
  w.insert(format!("{pfx}.layer_norm.weight"), vec1(PROJ_IN));
  w.insert(format!("{pfx}.layer_norm.bias"), vec1(PROJ_IN));
  w.insert(format!("{pfx}.linear_1.weight"), mat(PROJ_HIDDEN, PROJ_IN));
  w.insert(format!("{pfx}.linear_1.bias"), vec1(PROJ_HIDDEN));
  w.insert(format!("{pfx}.linear_2.weight"), mat(THIDDEN, PROJ_HIDDEN));
  w.insert(format!("{pfx}.linear_2.bias"), vec1(THIDDEN));
}

/// Insert the all-conv LFM2 LM weights under `language_model.model.`.
fn insert_language_model(w: &mut HashMap<String, Array>) {
  let pfx = "language_model.model";
  w.insert(format!("{pfx}.embed_tokens.weight"), mat(VOCAB, THIDDEN));
  w.insert(format!("{pfx}.embedding_norm.weight"), vec1(THIDDEN));
  for i in 0..TLAYERS {
    let lp = format!("{pfx}.layers.{i}");
    w.insert(format!("{lp}.operator_norm.weight"), vec1(THIDDEN));
    w.insert(format!("{lp}.ffn_norm.weight"), vec1(THIDDEN));
    w.insert(format!("{lp}.feed_forward.w1.weight"), mat(TFF, THIDDEN));
    w.insert(format!("{lp}.feed_forward.w3.weight"), mat(TFF, THIDDEN));
    w.insert(format!("{lp}.feed_forward.w2.weight"), mat(THIDDEN, TFF));
    // Conv layer: conv weight in MLX (C, K, 1) layout + the two projections.
    let kc = 3i32; // conv_L_cache
    let conv_data: Vec<f32> = (0..(THIDDEN * kc) as usize)
      .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
      .collect();
    w.insert(
      format!("{lp}.conv.conv.weight"),
      Array::from_slice::<f32>(&conv_data, &(THIDDEN as usize, kc as usize, 1usize)).unwrap(),
    );
    w.insert(
      format!("{lp}.conv.in_proj.weight"),
      mat(3 * THIDDEN, THIDDEN),
    );
    w.insert(format!("{lp}.conv.out_proj.weight"), mat(THIDDEN, THIDDEN));
  }
}

/// The complete dense synthetic checkpoint (post-sanitize keys).
fn dense_weights() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  insert_vision(&mut w);
  insert_projector(&mut w);
  insert_language_model(&mut w);
  w
}

/// Build a dense model from the tiny config + synthetic checkpoint.
fn dense_model() -> Lfm2Vl {
  Lfm2Vl::from_weights(tiny_config(false), dense_weights(), None).unwrap()
}

/// A synthetic single-image NaFlex input: a square `side x side` fully-active
/// grid of `num_patches = side*side` rows (so the cross-model + native paths
/// agree). The pixel rows are deterministic; the projected feature count is
/// `(side/factor)^2`.
fn synth_image(side: i32) -> Lfm2VlImageInputs {
  let num_patches = (side * side) as usize;
  let pv = mat(num_patches as i32, VPATCH_FEAT);
  let mask = Array::from_slice::<i32>(&vec![1i32; num_patches], &(num_patches,)).unwrap();
  let spatial = Array::from_slice::<i32>(&[side, side], &(2usize,)).unwrap();
  Lfm2VlImageInputs::from_parts(pv, mask, spatial)
}

/// `(B, T)` i32 token ids.
fn ids(data: &[i32], b: usize, t: usize) -> Array {
  Array::from_slice::<i32>(data, &(b, t)).unwrap()
}

// ───────────────────────── construction ─────────────────────────

#[test]
fn from_weights_consumes_all_and_builds() {
  // A clean build consumes every checkpoint weight (no leftover).
  let _model = dense_model();
}

#[test]
fn from_weights_leftover_weight_is_typed_error() {
  let mut w = dense_weights();
  w.insert("vision_tower.bogus.weight".to_string(), vec1(VHIDDEN));
  let err = Lfm2Vl::from_weights(tiny_config(false), w, None).unwrap_err();
  assert!(matches!(err, Error::InvariantViolation(_)), "got {err}");
}

#[test]
fn from_weights_missing_weight_is_typed_error() {
  let mut w = dense_weights();
  w.remove("multi_modal_projector.linear_1.weight");
  let err = Lfm2Vl::from_weights(tiny_config(false), w, None).unwrap_err();
  assert!(matches!(err, Error::MissingKey(_)), "got {err}");
}

#[test]
fn from_weights_stores_a_rope_normalized_text_config() {
  // `__post_init__` (`lfm2.py:40-42`): `text_config.rope_parameters.rope_theta`
  // wins over the top-level `text_config.rope_theta`. A freshly deserialized
  // `ModelConfig` is already normalized by `TextConfig`'s `Deserialize`, so this
  // pins the corrective path for a `ModelConfig` materialized then MUTATED into a
  // stale `rope_parameters` / `rope_theta` pair (the `text_config` field is
  // public): `Lfm2Vl::from_weights` must re-apply the override on the STORED
  // config, so `config().text_config.rope_theta` reflects the override the built
  // LM actually uses — not the stale top-level value.
  let mut cfg = tiny_config(false);
  // Desynchronize the pair: a stale top-level base of 1000, an override of 31337.
  cfg.text_config.rope_parameters = Some(crate::lm::models::lfm2::RopeParameters {
    rope_theta: Some(31337.0),
  });
  cfg.text_config.rope_theta = 1000.0;
  let model = Lfm2Vl::from_weights(cfg, dense_weights(), None).unwrap();
  assert_eq!(
    model.config().text_config.rope_theta,
    31337.0,
    "the stored text_config must carry the rope_parameters override, not the stale top-level rope_theta"
  );
}

#[test]
fn from_weights_rejects_non_finite_rope_theta_via_text_config_validate() {
  // The effective `rope_theta` finite/positive check in `TextConfig::validate`
  // runs on the VLM path too (`ModelConfig::validate` validates the nested text
  // config). A stale-pair mutation that makes the EFFECTIVE base 0.0 must be a
  // typed config error at build, not a silently-built invalid `Rope`. Here the
  // override is 0.0 (it wins over a sound top-level base).
  let mut cfg = tiny_config(false);
  cfg.text_config.rope_parameters = Some(crate::lm::models::lfm2::RopeParameters {
    rope_theta: Some(0.0),
  });
  let err = Lfm2Vl::from_weights(cfg, dense_weights(), None).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

// ───────────────────────── encode_image_inputs ─────────────────────────

#[test]
fn encode_image_inputs_shape() {
  // A 4x4 fully-active grid -> pixel-unshuffle by 2 -> (2,2) -> 4 projected
  // feature rows of width THIDDEN.
  let model = dense_model();
  let img = synth_image(4);
  let feats = model.encode_image_inputs(&img).unwrap();
  // N_i = (4/2)*(4/2) = 4 rows, D = THIDDEN.
  assert_eq!(feats.shape(), vec![4, THIDDEN as usize]);
  let mut f = feats.try_clone().unwrap();
  f.eval().unwrap();
  assert!(f.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}

// ───────────────── single-sourced active grid (spatial_shapes) ─────────────────

/// A full-budget (`VNUM_PATCHES`-row, all-active) NaFlex input whose declared
/// active grid is `(h_p, w_p)`. The pixel rows + mask are identical regardless
/// of `(h_p, w_p)` (every row active), so the ONLY thing that varies the
/// downstream active-row slice + PixelUnshuffle reshape is `spatial_shapes` —
/// the single source of truth. (`h_p * w_p` must equal `VNUM_PATCHES` so the
/// fully-active mask is consistent with the grid.)
fn full_budget_image(h_p: i32, w_p: i32) -> Lfm2VlImageInputs {
  let n = VNUM_PATCHES as usize;
  let pv = mat(VNUM_PATCHES, VPATCH_FEAT);
  let mask = Array::from_slice::<i32>(&vec![1i32; n], &(n,)).unwrap();
  let spatial = Array::from_slice::<i32>(&[h_p, w_p], &(2usize,)).unwrap();
  Lfm2VlImageInputs::from_parts(pv, mask, spatial)
}

#[test]
fn grid_is_read_from_spatial_shapes_single_source() {
  // `grid()` derives the active grid from `spatial_shapes` alone (there is no
  // separate host-int grid field after the single-source refactor), so it
  // returns exactly the spatial_shapes contents — the two CANNOT disagree.
  // (Compile-level: `from_parts` no longer accepts a grid argument, so a
  // disagreeing grid is unrepresentable; this pins the read at runtime.)
  let a = Array::from_slice::<i32>(&[2, 1], &(2usize,)).unwrap();
  let inputs = Lfm2VlImageInputs::from_parts(mat(2, VPATCH_FEAT), ones_mask(2), a);
  assert_eq!(inputs.grid().unwrap(), (2, 1));

  let b = Array::from_slice::<i32>(&[4, 1], &(2usize,)).unwrap();
  let inputs_b = Lfm2VlImageInputs::from_parts(mat(4, VPATCH_FEAT), ones_mask(4), b);
  assert_eq!(inputs_b.grid().unwrap(), (4, 1));
}

#[test]
fn grid_rejects_non_2_spatial_shapes() {
  // A `spatial_shapes` that is not a `(2,)` array is a typed RankMismatch from
  // `grid()` (never a panic / OOB index), since `grid()` reads `[0]` / `[1]`.
  let bad = Array::from_slice::<i32>(&[2, 1, 3], &(3usize,)).unwrap();
  let inputs = Lfm2VlImageInputs::from_parts(mat(3, VPATCH_FEAT), ones_mask(3), bad);
  assert!(
    matches!(inputs.grid().unwrap_err(), Error::RankMismatch(_)),
    "non-(2,) spatial_shapes must be a typed RankMismatch"
  );
}

#[test]
fn encode_image_inputs_slice_and_unshuffle_follow_spatial_shapes() {
  // The KEY single-source proof for the active-row slice + PixelUnshuffle
  // reshape: with the SAME (full-budget, all-active) pixel_values + mask, the
  // projected feature ROW COUNT is determined ENTIRELY by `spatial_shapes` —
  // because the slice takes `H_p * W_p` rows and the reshape uses `(1, H_p, W_p,
  // hidden)` before the factor-2 unshuffle.
  //
  //   grid (4, 1): slice 4 rows -> reshape (1, 4, 1, h) -> unshuffle pads to
  //                (1, 4, 2, h) -> fold by 2 -> (1, 2, 1, 4h) -> 2 feature rows.
  //   grid (2, 2): slice 4 rows -> reshape (1, 2, 2, h) -> unshuffle (1, 1, 1,
  //                4h) -> 1 feature row.
  //
  // The pixel tensor + mask are byte-identical across the two; ONLY
  // spatial_shapes differs. If anything but spatial_shapes drove the slice /
  // reshape, the row counts could not both be grid-consistent.
  let model = dense_model();

  let feats_4x1 = model.encode_image_inputs(&full_budget_image(4, 1)).unwrap();
  assert_eq!(
    feats_4x1.shape(),
    vec![2, THIDDEN as usize],
    "grid (4,1): ceil(4/2)*ceil(1/2) = 2 feature rows (slice+reshape follow spatial_shapes)"
  );

  let feats_2x2 = model.encode_image_inputs(&full_budget_image(2, 2)).unwrap();
  assert_eq!(
    feats_2x2.shape(),
    vec![1, THIDDEN as usize],
    "grid (2,2): ceil(2/2)*ceil(2/2) = 1 feature row — same pixels, different spatial_shapes"
  );

  // Both finite (no all-masked NaN, no OOB slice).
  for f in [feats_4x1, feats_2x2] {
    let mut f = f.try_clone().unwrap();
    f.eval().unwrap();
    assert!(f.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
  }
}

#[test]
fn encode_image_inputs_below_budget_grid_slices_active_rows() {
  // A below-budget active grid (2, 1) = 2 active patches out of the 4-patch
  // budget: the slice takes exactly the 2 active rows (driven by spatial_shapes
  // = (2,1)), reshapes to (1, 2, 1, hidden), unshuffles by 2 to (1, 1, 1, 4h),
  // and projects to ceil(2/2)*ceil(1/2) = 1 feature row. The 2 padded patch rows
  // (masked in the vision tower) never reach the projector.
  let model = dense_model();
  let img = full_budget_image_with_padding(2, 1);
  let feats = model.encode_image_inputs(&img).unwrap();
  assert_eq!(
    feats.shape(),
    vec![1, THIDDEN as usize],
    "grid (2,1) -> exactly 2 active rows sliced -> 1 projected feature row"
  );
  let mut f = feats.try_clone().unwrap();
  f.eval().unwrap();
  assert!(f.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}

/// A full-budget (`VNUM_PATCHES`-row) input whose mask marks only the first
/// `h_p * w_p` rows active (the rest padding), with `spatial_shapes = (h_p,
/// w_p)`. Exercises the below-budget active-row slice (the active count < the
/// patch budget).
fn full_budget_image_with_padding(h_p: i32, w_p: i32) -> Lfm2VlImageInputs {
  let n = VNUM_PATCHES as usize;
  let active = (h_p * w_p) as usize;
  let pv = mat(VNUM_PATCHES, VPATCH_FEAT);
  let mask_data: Vec<i32> = (0..n).map(|i| (i < active) as i32).collect();
  let mask = Array::from_slice::<i32>(&mask_data, &(n,)).unwrap();
  let spatial = Array::from_slice::<i32>(&[h_p, w_p], &(2usize,)).unwrap();
  Lfm2VlImageInputs::from_parts(pv, mask, spatial)
}

/// An all-active `(n,)` i32 mask.
fn ones_mask(n: i32) -> Array {
  Array::from_slice::<i32>(&vec![1i32; n as usize], &(n as usize,)).unwrap()
}

// ───────────────────────── end-to-end forward ─────────────────────────

#[test]
fn forward_multimodal_produces_expected_logits_shape() {
  // One 4x4 image -> 4 image tokens. Prompt: <bos=1> <image>*4 <tok=2>, so T=6.
  // The 4 <image> placeholders match the 4 projected feature rows.
  let model = dense_model();
  let img = synth_image(4);
  let input_ids = ids(&[1, IMG, IMG, IMG, IMG, 2], 1, 6);
  let mut cache = model.make_cache();
  let logits = model
    .forward_multimodal(&input_ids, std::slice::from_ref(&img), &mut cache)
    .unwrap();
  // [batch=1, seq=6, vocab=VOCAB].
  assert_eq!(logits.shape(), vec![1, 6, VOCAB as usize]);
  let mut l = logits.try_clone().unwrap();
  l.eval().unwrap();
  assert!(l.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn get_input_embeddings_merges_image_features() {
  // The merged embeddings replace the <image> positions with projected
  // features; the text positions keep their token embeddings.
  let model = dense_model();
  let img = synth_image(4);
  let input_ids = ids(&[1, IMG, IMG, IMG, IMG, 2], 1, 6);
  let merged = model
    .get_input_embeddings(&input_ids, std::slice::from_ref(&img))
    .unwrap();
  // [1, 6, THIDDEN].
  assert_eq!(merged.shape(), vec![1, 6, THIDDEN as usize]);

  // The merged embedding at the <image> positions must equal the projected
  // features (an independent recompute), and the text positions must equal the
  // token embeddings.
  let feats = model.encode_image_inputs(&img).unwrap();
  let text_embeds = model.embed_tokens(&input_ids).unwrap();
  let mut m = merged.try_clone().unwrap();
  m.eval().unwrap();
  let mv = m.to_vec::<f32>().unwrap();
  let mut fv = feats.try_clone().unwrap();
  fv.eval().unwrap();
  let fvv = fv.to_vec::<f32>().unwrap();
  let mut tv = text_embeds.try_clone().unwrap();
  tv.eval().unwrap();
  let tvv = tv.to_vec::<f32>().unwrap();
  let d = THIDDEN as usize;
  // Position 0 (text token 1): merged == token embedding.
  for k in 0..d {
    assert!((mv[k] - tvv[k]).abs() < 1e-5, "text pos 0 chan {k}");
  }
  // Positions 1..5 (the 4 <image> tokens): merged == feature rows 0..4.
  for (row, pos) in (1..5).enumerate() {
    for k in 0..d {
      assert!(
        (mv[pos * d + k] - fvv[row * d + k]).abs() < 1e-5,
        "image pos {pos} (feat row {row}) chan {k}"
      );
    }
  }
}

#[test]
fn forward_multimodal_count_mismatch_is_typed_error() {
  // 4 image tokens but a 6x6 image -> (6/2)^2 = 9 projected features. The merge
  // count check rejects the feature-vs-token mismatch with a typed error.
  let model = dense_model();
  let img = synth_image(6); // 9 feature rows
  let input_ids = ids(&[1, IMG, IMG, IMG, IMG, 2], 1, 6); // 4 image tokens
  let mut cache = model.make_cache();
  let err = model
    .forward_multimodal(&input_ids, std::slice::from_ref(&img), &mut cache)
    .unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

// ───────────────────────── text-only path ─────────────────────────

#[test]
fn text_only_forward_works() {
  // No images: the text-only forward (no <image> tokens) runs the LM directly.
  let model = dense_model();
  let input_ids = ids(&[1, 5, 9, 2], 1, 4);
  let mut cache = model.make_cache();
  let logits = LmModel::forward(&model, &input_ids, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 4, VOCAB as usize]);
}

#[test]
fn get_input_embeddings_no_images_is_text_embeds() {
  // With an empty image slice and no <image> tokens, get_input_embeddings is
  // exactly embed_tokens(input_ids).
  let model = dense_model();
  let input_ids = ids(&[1, 5, 9], 1, 3);
  let merged = model.get_input_embeddings(&input_ids, &[]).unwrap();
  let text = model.embed_tokens(&input_ids).unwrap();
  let mut m = merged.try_clone().unwrap();
  m.eval().unwrap();
  let mut t = text.try_clone().unwrap();
  t.eval().unwrap();
  assert_eq!(m.to_vec::<f32>().unwrap(), t.to_vec::<f32>().unwrap());
}

// ───────────────────────── cross-model trait ─────────────────────────

#[test]
fn vlm_trait_encode_image_consumes_processed_image() {
  // The cross-model encode_image consumes a ProcessedImage carrying the
  // flattened patch tensor (num_patches, patch_feat) + the native-resolution
  // companions (spatial_shape [H_p, W_p] = [4, 4], a fully-active patch_mask).
  // Grid 4x4 -> (4/2)^2 = 4 feature rows.
  let model = dense_model();
  let pixels = mat(16, VPATCH_FEAT); // (16, patch_feat)
  let spatial = Array::from_slice::<i32>(&[4, 4], &(2usize,)).unwrap();
  let mask = Array::from_slice::<i32>(&[1i32; 16], &(16usize,)).unwrap();
  let processed = ProcessedImage::new(pixels, Some(NativeResolution::new(spatial, mask)), 4);
  let feats = VlmModel::encode_image(&model, &processed).unwrap();
  assert_eq!(feats.shape(), vec![4, THIDDEN as usize]);
}

#[test]
fn vlm_trait_encode_image_missing_native_errors() {
  // A fixed-grid ProcessedImage (native = None) cannot drive the NaFlex path;
  // LFM2.5-VL's encode_image rejects it with a typed InvariantViolation
  // (never a panic) directing the caller to the Lfm2VlImageProcessor.
  let model = dense_model();
  let pixels = mat(16, VPATCH_FEAT);
  let processed = ProcessedImage::new(pixels, None, 4);
  let err = VlmModel::encode_image(&model, &processed).unwrap_err();
  assert!(matches!(err, Error::InvariantViolation(_)), "got {err}");
}

#[test]
fn vlm_trait_image_processor_config_is_siglip() {
  let model = dense_model();
  let cfg = VlmModel::image_processor_config(&model);
  assert_eq!(cfg.mean(), [0.5, 0.5, 0.5]);
  assert_eq!(cfg.std(), [0.5, 0.5, 0.5]);
  assert_eq!(cfg.resample(), crate::vlm::image::ResizeFilter::Bilinear);
}

// ───────────────────────── native-resolution processor ─────────────────────────

/// Build a deterministic `w × h` RGB PNG at `path` (a real file
/// `vlm::image::load_image` decodes).
fn write_png(path: &std::path::Path, w: u32, h: u32) {
  let mut buf = ::image::RgbImage::new(w, h);
  for y in 0..h {
    for x in 0..w {
      buf.put_pixel(x, y, ::image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
    .save_with_format(path, ::image::ImageFormat::Png)
    .unwrap();
}

/// The [`Lfm2VlImageProcessor`] returns a [`ProcessedImage`] with `Some`
/// native companions and a per-image `num_tokens` equal to
/// `ceil(H_p / factor) * ceil(W_p / factor)` — the PixelUnshuffle ceil
/// arithmetic, recomputed independently from the reported grid.
#[test]
fn lfm2vl_image_processor_native_count_matches_pixel_unshuffle() {
  let model = dense_model();
  let proc = VlmModel::image_processor(&model, 0);
  // A 32×24 RGB image; the SigLIP2 NaFlex smart-resize picks the grid within
  // the 1024-patch budget (patch_size = 2).
  let dir = std::env::temp_dir().join(format!("mlxrs-lfm2vl-proc-{}", std::process::id()));
  std::fs::create_dir_all(&dir).unwrap();
  let path = dir.join("img.png");
  write_png(&path, 32, 24);
  let img = crate::vlm::image::load_image(&path).unwrap();

  let processed = proc.process(&img).expect("native processor succeeds");
  let _ = std::fs::remove_file(&path);
  let _ = std::fs::remove_dir(&dir);

  // Native companions present.
  let native = processed
    .native()
    .expect("native-resolution companions present");
  // spatial_shape is (2,) = [H_p, W_p].
  let mut ss = native.spatial_shape().try_clone().unwrap();
  let grid = ss.to_vec::<i32>().unwrap();
  assert_eq!(grid.len(), 2, "spatial_shape is [H_p, W_p]");
  let (h_p, w_p) = (grid[0], grid[1]);
  // patch_mask is (max_num_patches,) — the active prefix count is H_p*W_p.
  assert_eq!(
    native.patch_mask().shape(),
    vec![model.config().max_num_patches as usize],
    "patch_mask is max_num_patches long"
  );
  // pixels is (max_num_patches, patch_feat).
  assert_eq!(
    processed.pixels().shape(),
    vec![
      model.config().max_num_patches as usize,
      VPATCH_FEAT as usize
    ],
  );
  // Independent ceil recompute (the PixelUnshuffle pad): ceil(x/f) = (x + f-1)/f.
  let f = FACTOR;
  let want = ((h_p + f - 1) / f) * ((w_p + f - 1) / f);
  assert_eq!(
    processed.num_tokens(),
    want as usize,
    "num_tokens = ceil(H_p/f) * ceil(W_p/f) (grid {h_p}x{w_p}, factor {f})"
  );
  assert!(want > 0, "a real image expands to >= 1 image token");
}

// ───────────────────────── end-to-end generic vlm_generate ─────────────────────────

/// The generic [`crate::vlm::generate::vlm_generate`] loop drives [`Lfm2Vl`]
/// end-to-end through the per-model [`ImageProcessor`] seam — the path that
/// previously failed because the factory-loaded model could not satisfy the
/// fixed-`num_tokens_per_image` cross-model surface. A synthetic image + a
/// single-`<image>`-marker prompt yields finite logits via the processor →
/// encode_image → merge → forward chain.
#[test]
fn vlm_generate_drives_lfm2vl_end_to_end() {
  use crate::{
    lm::generate::GenConfig,
    vlm::{
      generate::{VlmGenConfig, vlm_generate},
      prompt::MarkerPolicy,
    },
  };

  let model = dense_model();
  let proc = VlmModel::image_processor(&model, 0);
  let dir = std::env::temp_dir().join(format!("mlxrs-lfm2vl-e2e-{}", std::process::id()));
  std::fs::create_dir_all(&dir).unwrap();
  let path = dir.join("img.png");
  // Small image so the native grid (and image-token run) is modest.
  write_png(&path, 16, 16);

  // Prompt: <bos=1> <image=IMG> <tok=2>. One marker run of length 1 → one
  // image; the processor expands the placeholder to the image's token count.
  let prompt: Vec<u32> = vec![1, IMG as u32, 2];
  let cache = model.make_cache();
  // `image_token_id == image_marker_id == IMG`; the per-image count comes from
  // the processor (the cfg `num_tokens_per_image` is unused on this native-res
  // path, but a positive value keeps the config well-formed).
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(2),
    IMG as u32,
    1,
    MarkerPolicy::Required,
  );

  let mut it = vlm_generate(
    &model,
    proc.as_ref(),
    &prompt,
    std::slice::from_ref(&path),
    cache,
    cfg,
  )
  .expect("vlm_generate constructs and runs the native-resolution prefill");
  let first = it
    .next()
    .expect("at least one decode step")
    .expect("the prefill + first decode succeed → finite logits");
  let _ = std::fs::remove_file(&path);
  let _ = std::fs::remove_dir(&dir);

  // A sampled token id is in-vocab; the logprobs vector is finite.
  assert!((first.token as i32) < VOCAB, "sampled token id is in-vocab");
  let lp = first.logprobs.expect("VLM path always yields logprobs");
  let mut lp = lp.try_clone().unwrap();
  assert!(
    lp.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
    "logprobs are finite"
  );
}

// ───────────────────────── quantized path ─────────────────────────

const QGROUP: i32 = 32;
const QBITS: i32 = 8;

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

/// The quantized synthetic checkpoint: every quantizable axis is a multiple of
/// `QGROUP` (so the quantize is well-formed), every nn.Linear / nn.Embedding
/// quantized, the conv weight + norms kept dense (MLX quantizes Linear /
/// Embedding only). Mirrors the on-disk mlx-community 8-bit layout.
fn quant_config() -> ModelConfig {
  // Widen the quantizable axes to multiples of QGROUP = 32.
  let json = r#"{
    "model_type": "lfm2-vl",
    "quantization": {"group_size": 32, "bits": 8, "mode": "affine"},
    "downsample_factor": 2,
    "image_token_index": 7,
    "projector_hidden_size": 32,
    "projector_use_layernorm": true,
    "projector_bias": true,
    "vision_feature_layer": -1,
    "text_config": {
      "model_type": "lfm2",
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "vocab_size": 32,
      "block_dim": 32,
      "block_ff_dim": 32,
      "block_auto_adjust_ff_dim": false,
      "block_multiple_of": 1,
      "conv_L_cache": 3,
      "layer_types": ["conv", "conv"]
    },
    "vision_config": {
      "model_type": "lfm2_vl",
      "hidden_size": 32,
      "intermediate_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_channels": 3,
      "image_size": 8,
      "patch_size": 4,
      "num_patches": 4,
      "layer_norm_eps": 1e-6
    }
  }"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// Build the quantized checkpoint for [`quant_config`] (all axes 32-wide).
fn quant_weights() -> HashMap<String, Array> {
  let h = 32i32; // vision + text hidden
  let inter = 32i32;
  let patch_feat = 4 * 4 * 3; // patch_size 4 -> 48
  let proj_in = h * 4; // downsample^2
  let proj_hidden = 32i32;
  let ff = 32i32;
  let vocab = 32i32;
  let mut w = HashMap::new();

  // ── vision ──
  let vp = "vision_tower";
  w.insert(
    format!("{vp}.embeddings.patch_embedding.weight"),
    mat(h, patch_feat),
  );
  w.insert(format!("{vp}.embeddings.patch_embedding.bias"), vec1(h));
  w.insert(
    format!("{vp}.embeddings.position_embedding.weight"),
    mat(4, h),
  );
  for i in 0..2 {
    let lp = format!("{vp}.encoder.layers.{i}");
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      w.insert(format!("{lp}.self_attn.{p}.weight"), mat(h, h));
      w.insert(format!("{lp}.self_attn.{p}.bias"), vec1(h));
    }
    w.insert(format!("{lp}.mlp.fc1.weight"), mat(inter, h));
    w.insert(format!("{lp}.mlp.fc1.bias"), vec1(inter));
    w.insert(format!("{lp}.mlp.fc2.weight"), mat(h, inter));
    w.insert(format!("{lp}.mlp.fc2.bias"), vec1(h));
    for ln in ["layer_norm1", "layer_norm2"] {
      w.insert(format!("{lp}.{ln}.weight"), vec1(h));
      w.insert(format!("{lp}.{ln}.bias"), vec1(h));
    }
  }
  w.insert(format!("{vp}.post_layernorm.weight"), vec1(h));
  w.insert(format!("{vp}.post_layernorm.bias"), vec1(h));

  // ── projector ──
  let pp = "multi_modal_projector";
  w.insert(format!("{pp}.layer_norm.weight"), vec1(proj_in));
  w.insert(format!("{pp}.layer_norm.bias"), vec1(proj_in));
  w.insert(format!("{pp}.linear_1.weight"), mat(proj_hidden, proj_in));
  w.insert(format!("{pp}.linear_1.bias"), vec1(proj_hidden));
  w.insert(format!("{pp}.linear_2.weight"), mat(h, proj_hidden));
  w.insert(format!("{pp}.linear_2.bias"), vec1(h));

  // ── language model ──
  let mp = "language_model.model";
  w.insert(format!("{mp}.embed_tokens.weight"), mat(vocab, h));
  w.insert(format!("{mp}.embedding_norm.weight"), vec1(h));
  for i in 0..2 {
    let lp = format!("{mp}.layers.{i}");
    w.insert(format!("{lp}.operator_norm.weight"), vec1(h));
    w.insert(format!("{lp}.ffn_norm.weight"), vec1(h));
    w.insert(format!("{lp}.feed_forward.w1.weight"), mat(ff, h));
    w.insert(format!("{lp}.feed_forward.w3.weight"), mat(ff, h));
    w.insert(format!("{lp}.feed_forward.w2.weight"), mat(h, ff));
    let kc = 3i32;
    let conv_data: Vec<f32> = (0..(h * kc) as usize)
      .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
      .collect();
    w.insert(
      format!("{lp}.conv.conv.weight"),
      Array::from_slice::<f32>(&conv_data, &(h as usize, kc as usize, 1usize)).unwrap(),
    );
    w.insert(format!("{lp}.conv.in_proj.weight"), mat(3 * h, h));
    w.insert(format!("{lp}.conv.out_proj.weight"), mat(h, h));
  }

  // Quantize every nn.Linear + nn.Embedding whose quantizable input axis is a
  // whole number of `QGROUP` groups (the conv weight + norms stay dense). The
  // patch_embedding's input axis is `patch_feat = 48` (not a multiple of 32),
  // so it stays dense — exactly as a real checkpoint leaves a non-group-aligned
  // layer unquantized.
  let mut quant_targets: Vec<String> = Vec::new();
  for i in 0..2 {
    let lp = format!("{vp}.encoder.layers.{i}");
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
      quant_targets.push(format!("{lp}.self_attn.{p}"));
    }
    quant_targets.push(format!("{lp}.mlp.fc1"));
    quant_targets.push(format!("{lp}.mlp.fc2"));
  }
  quant_targets.push(format!("{pp}.linear_1"));
  quant_targets.push(format!("{pp}.linear_2"));
  quant_targets.push(format!("{mp}.embed_tokens"));
  for i in 0..2 {
    let lp = format!("{mp}.layers.{i}");
    quant_targets.push(format!("{lp}.feed_forward.w1"));
    quant_targets.push(format!("{lp}.feed_forward.w3"));
    quant_targets.push(format!("{lp}.feed_forward.w2"));
    quant_targets.push(format!("{lp}.conv.in_proj"));
    quant_targets.push(format!("{lp}.conv.out_proj"));
  }
  for t in &quant_targets {
    quantize_weight_in_place(&mut w, t);
  }
  w
}

#[test]
fn quantized_path_loads_and_forwards() {
  let cfg = quant_config();
  let quant = resolve_quantization(&tiny_quant_config_json()).unwrap();
  let model = Lfm2Vl::from_weights(cfg, quant_weights(), quant.as_ref()).unwrap();

  // A 4x4 fully-active image (patch_size 4 -> patch_feat 48). The vision tower
  // num_patches = 4 (2x2 grid). Use a synthetic 2x2 grid image (4 patches).
  let num_patches = 4usize;
  let patch_feat = 48i32;
  let pv = mat(num_patches as i32, patch_feat);
  let mask = Array::from_slice::<i32>(&vec![1i32; num_patches], &(num_patches,)).unwrap();
  let spatial = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let img = Lfm2VlImageInputs::from_parts(pv, mask, spatial);
  // 2x2 grid, factor 2 -> (2/2)^2 = 1 projected feature row.
  let input_ids = ids(&[1, IMG, 2], 1, 3);
  let mut cache = model.make_cache();
  let logits = model
    .forward_multimodal(&input_ids, std::slice::from_ref(&img), &mut cache)
    .unwrap();
  assert_eq!(logits.shape(), vec![1, 3, 32]);
  let mut l = logits.try_clone().unwrap();
  l.eval().unwrap();
  assert!(
    l.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()),
    "quantized multimodal forward must be finite"
  );
}

/// The quant config JSON (for `resolve_quantization`).
fn tiny_quant_config_json() -> String {
  r#"{"quantization": {"group_size": 32, "bits": 8, "mode": "affine"}}"#.to_string()
}

// ───────────────────────── factory registration ─────────────────────────

#[test]
fn registry_registers_lfm2_vl() {
  let mut reg = VlmTypeRegistry::new();
  assert!(register(&mut reg).is_none());
  // The "lfm2-vl" config alias and the canonical "lfm2_vl" both resolve.
  assert!(reg.contains("lfm2-vl"));
  assert!(reg.contains("lfm2_vl"));
}

#[test]
fn constructor_builds_from_loaded_model() {
  // The factory constructor parses the config off the raw JSON, sanitizes the
  // (pre-sanitize-key) weights, and assembles the model. Use pre-sanitize keys
  // (the `model.` prefixes the VL sanitize strips/remaps) to exercise sanitize.
  let config_json = tiny_config_json(false);
  let weights = pre_sanitize_weights();
  let base = crate::vlm::load::VlmBaseConfig::from_json(&config_json).unwrap();
  let loaded = LoadedVlmModel::new(base, config_json, weights);
  let model = constructor()(&loaded).unwrap();
  // Drive a text-only forward through the boxed `dyn VlmModel` (a
  // `crate::lm::model::Model` supertrait). The tiny config has 2 all-conv
  // layers -> 2 ArraysCache(1).
  let input_ids = ids(&[1, 5, 2], 1, 3);
  let mut cache: Vec<Box<dyn crate::lm::cache::KvCache>> = (0..TLAYERS)
    .map(|_| -> Box<dyn crate::lm::cache::KvCache> {
      Box::new(crate::lm::cache::ArraysCache::new(1))
    })
    .collect();
  let logits = LmModel::forward(model.as_ref(), &input_ids, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, VOCAB as usize]);
}

/// The synthetic checkpoint with the PRE-sanitize keys the real checkpoint
/// ships (a leading `model.` on the vision / projector / LM keys that
/// `Lfm2Vl::sanitize` strips + remaps). Mirrors `lfm2_vl.py`'s `transform_key`
/// inputs.
fn pre_sanitize_weights() -> HashMap<String, Array> {
  // The dense (post-sanitize) keys, re-prefixed to their pre-sanitize form:
  //   vision_tower.* keys ship as model.vision_tower.* (the VL sanitize strips
  //     the `model.` for vision keys),
  //   language_model.model.* ships as model.language_model.* (sanitize maps
  //     `model.language_model` -> `language_model.model`),
  //   multi_modal_projector.* ships as model.multi_modal_projector.*.
  let mut out = HashMap::new();
  for (k, v) in dense_weights() {
    let pre = if let Some(rest) = k.strip_prefix("vision_tower.") {
      format!("model.vision_tower.{rest}")
    } else if let Some(rest) = k.strip_prefix("language_model.model.") {
      format!("model.language_model.{rest}")
    } else if let Some(rest) = k.strip_prefix("multi_modal_projector.") {
      format!("model.multi_modal_projector.{rest}")
    } else {
      k
    };
    out.insert(pre, v);
  }
  out
}
