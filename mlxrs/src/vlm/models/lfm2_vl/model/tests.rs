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
  Lfm2VlImageInputs::from_parts(pv, mask, spatial, side, side)
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
fn vlm_trait_encode_image_square_fully_active() {
  // The cross-model encode_image takes a single (1, num_patches, patch_feat)
  // Array for a square fully-active grid. num_patches = 16 -> side 4 ->
  // (4/2)^2 = 4 feature rows.
  let model = dense_model();
  let pv =
    crate::ops::shape::reshape(&mat(16, VPATCH_FEAT), &(1usize, 16, VPATCH_FEAT as usize)).unwrap();
  let feats = VlmModel::encode_image(&model, &pv).unwrap();
  assert_eq!(feats.shape(), vec![4, THIDDEN as usize]);
}

#[test]
fn vlm_trait_encode_image_non_square_errors() {
  // A non-perfect-square num_patches can't be realized by the single-Array
  // entry (use encode_image_inputs); it's a typed error.
  let model = dense_model();
  let pv =
    crate::ops::shape::reshape(&mat(6, VPATCH_FEAT), &(1usize, 6, VPATCH_FEAT as usize)).unwrap();
  let err = VlmModel::encode_image(&model, &pv).unwrap_err();
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
  let img = Lfm2VlImageInputs::from_parts(pv, mask, spatial, 2, 2);
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
