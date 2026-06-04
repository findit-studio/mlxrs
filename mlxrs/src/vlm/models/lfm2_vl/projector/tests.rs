//! Structural + oracle tests for the LFM2.5-VL pixel-unshuffle, multimodal
//! projector, and image-feature merge.
//!
//! Deterministic, tiny-fixture, non-gated. The pixel-unshuffle tests pin the
//! pad-odd-grid + fold shape (including the required odd-grid `(1,5,5,768) ->
//! (1,3,3,3072)` case) and a hand-computed numeric oracle; the projector tests
//! pin the `(N, in) -> (N, text_hidden)` shape on the dense AND the 8-bit
//! quantized path; the merge tests pin that the projected features land at the
//! `<image>`-token positions, that a count mismatch is a typed error, and the
//! rank guards.

use std::collections::HashMap;

use super::*;
use crate::vlm::models::lfm2_vl::config::ModelConfig;

// ───────────────────────── shared fixture helpers ─────────────────────────

/// Evaluate `a` to a host `Vec<f32>` (an explicit clone + eval; the public
/// surface keeps the data accessor `&mut`).
fn eval_to_vec(a: &Array) -> Vec<f32> {
  let mut a = a.try_clone().unwrap();
  a.eval().unwrap();
  a.to_vec::<f32>().unwrap()
}

/// A deterministic `(rows, cols)` matrix: entry `[i, j] = ((i*cols+j) % 7) *
/// 0.01 + 0.001` — small, nonzero, reproducible.
fn mat(rows: i32, cols: i32) -> Array {
  let (r, c) = (rows as usize, cols as usize);
  let data: Vec<f32> = (0..r * c)
    .map(|n| ((n % 7) as f32) * 0.01 + 0.001)
    .collect();
  Array::from_slice::<f32>(&data, &(r, c)).unwrap()
}

/// A length-`n` deterministic bias vector.
fn vec_n(n: i32) -> Array {
  let data: Vec<f32> = (0..n as usize)
    .map(|i| 0.01 + (i % 7) as f32 * 0.005)
    .collect();
  Array::from_slice::<f32>(&data, &(n as usize,)).unwrap()
}

/// No-op quantization resolver (every layer dense).
fn no_quant(_path: &str) -> Option<(i32, i32, &'static str)> {
  None
}

// ───────────────────────── PixelUnshuffleBlock ─────────────────────────

#[test]
fn pixel_unshuffle_even_grid_shape() {
  // (1, 4, 4, 768) factor 2 -> (1, 2, 2, 3072). No padding (both dims even).
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::zeros::<f32>(&(1usize, 4, 4, 768)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 2, 3072]);
}

#[test]
fn pixel_unshuffle_odd_grid_pads_then_folds() {
  // The required odd-grid case: (1, 5, 5, 768) factor 2 -> pad both spatial
  // dims to 6 -> (1, 3, 3, 3072).
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::zeros::<f32>(&(1usize, 5, 5, 768)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 3, 3, 3072]);
}

#[test]
fn pixel_unshuffle_odd_w_only() {
  // Only W odd: (1, 3, 4, 8) factor 2 -> pad W to 4 -> (1, 2, 2, 32).
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::zeros::<f32>(&(1usize, 3, 4, 8)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 2, 32]);
}

#[test]
fn pixel_unshuffle_odd_h_only() {
  // Only H odd: (1, 4, 3, 8) factor 2 -> pad H to 4 -> (1, 2, 2, 32).
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::zeros::<f32>(&(1usize, 4, 3, 8)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 2, 2, 32]);
}

#[test]
fn pixel_unshuffle_numeric_oracle() {
  // A (1, 2, 2, 1) grid with distinct channel values folds the 2x2 spatial
  // neighborhood into the channel axis. Following lfm2_vl.py exactly:
  //   input (N=1, W=2, H=2, C=1), values laid out [w, h, c] row-major:
  //     x[0,0,0,0]=10, x[0,0,1,0]=11, x[0,1,0,0]=12, x[0,1,1,0]=13
  //   reshape(1, 2, 1, 2): rows over w, each holding (h/f=1, c*f=2):
  //     [[ [10,11] ], [ [12,13] ]]
  //   transpose(0,2,1,3) -> (1,1,2,2): [[ [10,11],[12,13] ]]
  //   reshape(1,1,1,4): [[[ [10,11,12,13] ]]]
  //   transpose(0,2,1,3) -> (1,1,1,4): [[[ [10,11,12,13] ]]]
  // So the single output cell is [10, 11, 12, 13].
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::from_slice::<f32>(&[10.0, 11.0, 12.0, 13.0], &(1usize, 2, 2, 1)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 1, 4]);
  assert_eq!(eval_to_vec(&out), vec![10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn pixel_unshuffle_pad_zeros_oracle() {
  // A (1, 1, 1, 2) grid factor 2 pads BOTH W and H to 2 with zeros, then folds.
  // The single real cell [a, b] sits at spatial (0,0); the three padded cells
  // are zero. After the fold the output (1,1,1,8) channel order is the
  // row-major [w, h] scan of the (2,2) padded grid's per-cell [a,b]:
  //   (w=0,h=0)=[a,b], (w=0,h=1)=[0,0], (w=1,h=0)=[0,0], (w=1,h=1)=[0,0]
  // -> [a, b, 0, 0, 0, 0, 0, 0].
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::from_slice::<f32>(&[7.0, 9.0], &(1usize, 1, 1, 2)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 1, 8]);
  assert_eq!(
    eval_to_vec(&out),
    vec![7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  );
}

#[test]
fn pixel_unshuffle_rejects_non_rank4() {
  let block = PixelUnshuffleBlock::new(2).unwrap();
  let x = Array::zeros::<f32>(&(1usize, 4, 4)).unwrap(); // rank-3
  let err = block.forward(&x).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn pixel_unshuffle_rejects_zero_factor() {
  let err = PixelUnshuffleBlock::new(0).unwrap_err();
  assert!(matches!(err, Error::OutOfRange(_)), "got {err}");
}

#[test]
fn pixel_unshuffle_factor_one_is_identity_shape() {
  // factor 1: no fold, no pad — shape unchanged (the reference substitutes
  // Identity for factor 1, but the block is total and a no-op here).
  let block = PixelUnshuffleBlock::new(1).unwrap();
  let x = mat(1, 12); // reshape to (1, 3, 2, 2)
  let x = crate::ops::shape::reshape(&x, &(1usize, 3, 2, 2)).unwrap();
  let out = block.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 3, 2, 2]);
}

// ───────────────────────── multimodal projector ─────────────────────────
//
// A tiny projector fixture: in = vision_hidden * factor^2. The smallest
// configuration that mirrors the real shape contract uses vision_hidden = 8,
// downsample_factor = 2 (so in = 32), projector_hidden = 16, text_hidden = 4.

const PROJ_IN: i32 = 32; // 8 * 2^2
const PROJ_HIDDEN: i32 = 16;
const TEXT_HIDDEN: i32 = 4;

/// A tiny LFM2.5-VL model config whose projector dims are (in=32 implied by
/// vision_hidden*factor^2, projector_hidden=16, text_hidden=4). The text/vision
/// configs only need to validate; the projector reads its widths from the
/// loaded weight shapes, not the config.
fn tiny_model_config(use_layernorm: bool) -> ModelConfig {
  let json = format!(
    r#"{{
      "model_type": "lfm2-vl",
      "downsample_factor": 2,
      "image_token_index": 396,
      "projector_hidden_size": {PROJ_HIDDEN},
      "projector_use_layernorm": {use_layernorm},
      "projector_bias": true,
      "vision_feature_layer": -1,
      "text_config": {{
        "model_type": "lfm2",
        "hidden_size": {TEXT_HIDDEN},
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "block_dim": {TEXT_HIDDEN},
        "block_ff_dim": 8,
        "block_auto_adjust_ff_dim": false,
        "block_multiple_of": 1,
        "conv_L_cache": 3
      }},
      "vision_config": {{
        "model_type": "lfm2_vl",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_channels": 3,
        "image_size": 4,
        "patch_size": 2,
        "num_patches": 4,
        "layer_norm_eps": 1e-6
      }}
    }}"#
  );
  let cfg = ModelConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// The dense projector weight map: layer_norm (optional) + linear_1 + linear_2.
fn projector_dense_weights(use_layernorm: bool) -> HashMap<String, Array> {
  let mut w = HashMap::new();
  if use_layernorm {
    w.insert("layer_norm.weight".to_string(), vec_n(PROJ_IN));
    w.insert("layer_norm.bias".to_string(), vec_n(PROJ_IN));
  }
  // linear_1: (projector_hidden, in), bias (projector_hidden,).
  w.insert("linear_1.weight".to_string(), mat(PROJ_HIDDEN, PROJ_IN));
  w.insert("linear_1.bias".to_string(), vec_n(PROJ_HIDDEN));
  // linear_2: (text_hidden, projector_hidden), bias (text_hidden,).
  w.insert("linear_2.weight".to_string(), mat(TEXT_HIDDEN, PROJ_HIDDEN));
  w.insert("linear_2.bias".to_string(), vec_n(TEXT_HIDDEN));
  w
}

#[test]
fn projector_shape_with_layernorm() {
  let cfg = tiny_model_config(true);
  let mut w = projector_dense_weights(true);
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap();
  // (N=5, in=32) -> (N=5, text_hidden=4).
  let x = mat(5, PROJ_IN);
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![5, TEXT_HIDDEN as usize]);
  assert!(eval_to_vec(&out).iter().all(|v| v.is_finite()));
}

#[test]
fn projector_shape_without_layernorm() {
  let cfg = tiny_model_config(false);
  let mut w = projector_dense_weights(false);
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap();
  let x = mat(3, PROJ_IN);
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![3, TEXT_HIDDEN as usize]);
  assert!(!proj.is_quantized());
}

#[test]
fn projector_missing_layernorm_weight_is_typed_error() {
  let cfg = tiny_model_config(true);
  let mut w = projector_dense_weights(true);
  w.remove("layer_norm.bias");
  let err = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap_err();
  assert!(matches!(err, Error::MissingKey(_)), "got {err}");
}

/// A tiny config with an explicit `projector_bias` flag (otherwise identical to
/// [`tiny_model_config`], layernorm on).
fn tiny_model_config_bias(projector_bias: bool) -> ModelConfig {
  let json = format!(
    r#"{{
      "model_type": "lfm2-vl",
      "downsample_factor": 2,
      "projector_hidden_size": {PROJ_HIDDEN},
      "projector_use_layernorm": true,
      "projector_bias": {projector_bias},
      "text_config": {{
        "model_type": "lfm2", "hidden_size": {TEXT_HIDDEN},
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "block_dim": {TEXT_HIDDEN}, "block_ff_dim": 8,
        "block_auto_adjust_ff_dim": false, "block_multiple_of": 1, "conv_L_cache": 3
      }},
      "vision_config": {{
        "model_type": "lfm2_vl", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_channels": 3,
        "image_size": 4, "patch_size": 2, "num_patches": 4, "layer_norm_eps": 1e-6
      }}
    }}"#
  );
  let cfg = ModelConfig::from_json(&json).unwrap();
  cfg.validate().unwrap();
  cfg
}

/// The projector weight map WITHOUT the two `linear_*.bias` tensors (the
/// `projector_bias = false` checkpoint shape): layer_norm + the two weights.
fn projector_weights_no_bias() -> HashMap<String, Array> {
  let mut w = HashMap::new();
  w.insert("layer_norm.weight".to_string(), vec_n(PROJ_IN));
  w.insert("layer_norm.bias".to_string(), vec_n(PROJ_IN));
  w.insert("linear_1.weight".to_string(), mat(PROJ_HIDDEN, PROJ_IN));
  w.insert("linear_2.weight".to_string(), mat(TEXT_HIDDEN, PROJ_HIDDEN));
  w
}

#[test]
fn projector_bias_true_requires_bias_tensors() {
  // projector_bias=true + both linear biases present: loads and forwards (the
  // matched-true case).
  let cfg = tiny_model_config_bias(true);
  let mut w = projector_dense_weights(true);
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap();
  let out = proj.forward(&mat(2, PROJ_IN)).unwrap();
  assert_eq!(out.shape(), vec![2, TEXT_HIDDEN as usize]);
}

#[test]
fn projector_bias_true_missing_bias_is_missing_key() {
  // projector_bias=true but a required `linear_*.bias` is absent: a typed
  // MissingKey (the `take_if` required-when-true gate), NOT a silent omission.
  let cfg = tiny_model_config_bias(true);
  let mut w = projector_dense_weights(true);
  w.remove("linear_2.bias");
  let err = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap_err();
  assert!(matches!(err, Error::MissingKey(_)), "got {err}");
}

#[test]
fn projector_bias_false_loads_without_biases() {
  // projector_bias=false + no linear biases present: loads and forwards (the
  // matched-false case — the bias-free projector).
  let cfg = tiny_model_config_bias(false);
  let mut w = projector_weights_no_bias();
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap();
  let out = proj.forward(&mat(2, PROJ_IN)).unwrap();
  assert_eq!(out.shape(), vec![2, TEXT_HIDDEN as usize]);
  // The map is fully drained (no stray `.bias` left behind, none consumed).
  assert!(w.is_empty(), "no leftover weights, got {w:?}");
}

#[test]
fn projector_bias_false_stray_bias_is_key_collision() {
  // projector_bias=false but a `linear_*.bias` is present: a typed KeyCollision
  // (the `take_if` forbidden-when-false gate), NOT a silent auto-apply.
  let cfg = tiny_model_config_bias(false);
  let mut w = projector_weights_no_bias();
  w.insert("linear_1.bias".to_string(), vec_n(PROJ_HIDDEN));
  let err = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap_err();
  assert!(matches!(err, Error::KeyCollision(_)), "got {err}");
}

#[test]
fn projector_3d_input_preserves_leading_dims() {
  // The projector is applied to the pixel-unshuffled (1, H/f, W/f, in) grid in
  // the real flow; a rank-3 (1, M, in) input should map to (1, M, text_hidden).
  let cfg = tiny_model_config(true);
  let mut w = projector_dense_weights(true);
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &no_quant).unwrap();
  let x = crate::ops::shape::reshape(&mat(6, PROJ_IN), &(1usize, 6, PROJ_IN as usize)).unwrap();
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 6, TEXT_HIDDEN as usize]);
}

// ───────────────────── quantized projector path ─────────────────────
//
// No local 8-bit checkpoint is available, so the quantized path is covered by a
// SYNTHETIC fixture: the projector's two nn.Linear weights replaced by the real
// `ops::quantized::quantize` (weight, scales, biases) triple — the exact on-disk
// layout an mlx-community 8-bit checkpoint stores. The layer_norm stays dense
// (MLX quantizes nn.Linear only). The projector must construct the quantized
// Linears and run a finite forward.

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

/// A quantized-fixture config: projector in/hidden are multiples of `QGROUP`
/// (the quantizable Linear input axis is a whole number of groups). in = 32
/// (vision_hidden 8 * factor^2 4), projector_hidden = 32, text_hidden = 32.
fn quant_model_config() -> ModelConfig {
  let json = r#"{
    "model_type": "lfm2-vl",
    "downsample_factor": 2,
    "projector_hidden_size": 32,
    "projector_use_layernorm": true,
    "projector_bias": true,
    "text_config": {
      "model_type": "lfm2",
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "vocab_size": 32,
      "block_dim": 32,
      "block_ff_dim": 8,
      "block_auto_adjust_ff_dim": false,
      "block_multiple_of": 1,
      "conv_L_cache": 3
    },
    "vision_config": {
      "model_type": "lfm2_vl",
      "hidden_size": 8,
      "intermediate_size": 16,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_channels": 3,
      "image_size": 4,
      "patch_size": 2,
      "num_patches": 4,
      "layer_norm_eps": 1e-6
    }
  }"#;
  let cfg = ModelConfig::from_json(json).unwrap();
  cfg.validate().unwrap();
  cfg
}

#[test]
fn projector_quantized_loads_and_forwards() {
  let cfg = quant_model_config();
  let qin = 32i32;
  let qhidden = 32i32;
  let qtext = 32i32;
  let mut w = HashMap::new();
  w.insert("layer_norm.weight".to_string(), vec_n(qin));
  w.insert("layer_norm.bias".to_string(), vec_n(qin));
  w.insert("linear_1.weight".to_string(), mat(qhidden, qin));
  w.insert("linear_1.bias".to_string(), vec_n(qhidden));
  w.insert("linear_2.weight".to_string(), mat(qtext, qhidden));
  w.insert("linear_2.bias".to_string(), vec_n(qtext));
  // Quantize both projection weights.
  quantize_weight_in_place(&mut w, "linear_1");
  quantize_weight_in_place(&mut w, "linear_2");

  let quant = |_path: &str| -> Option<(i32, i32, &'static str)> { Some((QGROUP, QBITS, "affine")) };
  let proj = Lfm2VlMultiModalProjector::from_weights(&cfg, &mut w, &quant).unwrap();
  assert!(
    proj.is_quantized(),
    "linear_1 must be the quantized variant"
  );

  let x = mat(4, qin);
  let out = proj.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![4, qtext as usize]);
  assert!(
    eval_to_vec(&out).iter().all(|v| v.is_finite()),
    "quantized projector forward must be finite"
  );
}

// ───────────────────── merge_input_ids_with_image_features ─────────────────────

/// Build a `(B, T)` i32 token-id array.
fn ids(data: &[i32], b: usize, t: usize) -> Array {
  Array::from_slice::<i32>(data, &(b, t)).unwrap()
}

/// Build a `(B, T, D)` embeddings array where cell `[b, t, :]` is `base + t`
/// repeated across D — distinct per position so the splice is observable.
fn embeds(b: usize, t: usize, d: usize) -> Array {
  let mut data: Vec<f32> = Vec::with_capacity(b * t * d);
  for bi in 0..b {
    for ti in 0..t {
      for _ in 0..d {
        data.push((bi * 100 + ti) as f32);
      }
    }
  }
  Array::from_slice::<f32>(&data, &(b, t, d)).unwrap()
}

/// Build an `(N, D)` image-features array where row k is `1000 + k` repeated
/// across D — distinct from any text embedding so a misplacement is visible.
fn feats(n: usize, d: usize) -> Array {
  let mut data: Vec<f32> = Vec::with_capacity(n * d);
  for k in 0..n {
    for _ in 0..d {
      data.push((1000 + k) as f32);
    }
  }
  Array::from_slice::<f32>(&data, &(n, d)).unwrap()
}

const IMG: i32 = 396;

#[test]
fn merge_places_features_at_masked_positions() {
  // B=1, T=5, D=2; image tokens at t=1,2,3 (a contiguous run). Three feature
  // rows fill those positions in order; the text rows at t=0,4 survive.
  let d = 2usize;
  let input_ids = ids(&[5, IMG, IMG, IMG, 9], 1, 5);
  let inputs_embeds = embeds(1, 5, d);
  let image_features = feats(3, d);
  let out =
    merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG).unwrap();
  assert_eq!(out.shape(), vec![1, 5, d]);
  let v = eval_to_vec(&out);
  // t=0: text (base 0) ; t=1..3: features 1000,1001,1002 ; t=4: text (4).
  let expected = vec![
    0.0, 0.0, // t=0 text
    1000.0, 1000.0, // t=1 feat row 0
    1001.0, 1001.0, // t=2 feat row 1
    1002.0, 1002.0, // t=3 feat row 2
    4.0, 4.0, // t=4 text
  ];
  assert_eq!(v, expected);
}

#[test]
fn merge_handles_non_contiguous_mask() {
  // Image tokens at NON-contiguous positions t=0 and t=3 — the masked-scatter
  // semantics (not span-replace) place row 0 at t=0 and row 1 at t=3.
  let d = 1usize;
  let input_ids = ids(&[IMG, 5, 6, IMG, 9], 1, 5);
  let inputs_embeds = embeds(1, 5, d);
  let image_features = feats(2, d);
  let out =
    merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG).unwrap();
  let v = eval_to_vec(&out);
  // t0=feat0, t1=text1, t2=text2, t3=feat1, t4=text4.
  assert_eq!(v, vec![1000.0, 1.0, 2.0, 1001.0, 4.0]);
}

#[test]
fn merge_multi_batch_row_major_order() {
  // B=2, T=2, D=1; one image token per row. Row-major mask scan visits
  // (b=0,t=1) then (b=1,t=0): feature row 0 -> (0,1), row 1 -> (1,0).
  let d = 1usize;
  let input_ids = ids(&[5, IMG, IMG, 9], 2, 2);
  let inputs_embeds = embeds(2, 2, d);
  let image_features = feats(2, d);
  let out =
    merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG).unwrap();
  let v = eval_to_vec(&out);
  // (b0,t0)=text 0, (b0,t1)=feat0 1000, (b1,t0)=feat1 1001, (b1,t1)=text 101.
  assert_eq!(v, vec![0.0, 1000.0, 1001.0, 101.0]);
}

#[test]
fn merge_count_mismatch_is_typed_error() {
  // Two image tokens but three feature rows -> the lfm2_vl.py:173-176 mismatch.
  let d = 2usize;
  let input_ids = ids(&[IMG, IMG, 9], 1, 3);
  let inputs_embeds = embeds(1, 3, d);
  let image_features = feats(3, d);
  let err = merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG)
    .unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

#[test]
fn merge_count_mismatch_too_few_features() {
  // Three image tokens but two feature rows -> mismatch.
  let d = 2usize;
  let input_ids = ids(&[IMG, IMG, IMG], 1, 3);
  let inputs_embeds = embeds(1, 3, d);
  let image_features = feats(2, d);
  let err = merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG)
    .unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

#[test]
fn merge_no_image_tokens_returns_embeds_unchanged() {
  // No image tokens AND zero feature rows -> the all-text path returns the
  // embeddings unchanged.
  let d = 2usize;
  let input_ids = ids(&[5, 6, 7], 1, 3);
  let inputs_embeds = embeds(1, 3, d);
  let image_features = feats(0, d);
  let out =
    merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG).unwrap();
  assert_eq!(eval_to_vec(&out), eval_to_vec(&inputs_embeds));
}

#[test]
fn merge_rejects_rank2_embeds() {
  let d = 2usize;
  let input_ids = ids(&[IMG], 1, 1);
  let bad_embeds = mat(1, d as i32); // rank-2
  let image_features = feats(1, d);
  let err =
    merge_input_ids_with_image_features(&image_features, &bad_embeds, &input_ids, IMG).unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn merge_rejects_rank1_input_ids() {
  let d = 2usize;
  let bad_ids = Array::from_slice::<i32>(&[IMG], &(1usize,)).unwrap(); // rank-1
  let inputs_embeds = embeds(1, 1, d);
  let image_features = feats(1, d);
  let err = merge_input_ids_with_image_features(&image_features, &inputs_embeds, &bad_ids, IMG)
    .unwrap_err();
  assert!(matches!(err, Error::RankMismatch(_)), "got {err}");
}

#[test]
fn merge_rejects_hidden_width_mismatch() {
  // image_features D=3 but inputs_embeds D=2.
  let input_ids = ids(&[IMG, 9], 1, 2);
  let inputs_embeds = embeds(1, 2, 2);
  let image_features = feats(1, 3);
  let err = merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG)
    .unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}

#[test]
fn merge_rejects_bt_mismatch() {
  // input_ids (1,2) vs inputs_embeds (1,3).
  let input_ids = ids(&[IMG, 9], 1, 2);
  let inputs_embeds = embeds(1, 3, 2);
  let image_features = feats(1, 2);
  let err = merge_input_ids_with_image_features(&image_features, &inputs_embeds, &input_ids, IMG)
    .unwrap_err();
  assert!(matches!(err, Error::LengthMismatch(_)), "got {err}");
}
