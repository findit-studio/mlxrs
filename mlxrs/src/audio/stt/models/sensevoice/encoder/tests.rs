//! SANM/FSMN encoder oracles for SenseVoice-Small.
//!
//! Every expected value is computed independently of the code under test — the
//! sinusoidal PE by the closed-form sin/cos recurrence, the FSMN depthwise conv
//! by a hand-rolled channels-last grouped convolution over plain `Vec`s (the
//! round-trip-via-real-functions discipline: the FSMN numeric is checked
//! through the real [`MultiHeadedAttentionSANM::forward_fsmn`], never a parallel
//! copy of it), and the variable-width residual rule by a from-scratch
//! pre-norm layer arithmetic — never by invoking the implementation twice.

use std::collections::HashMap;

use super::*;
use crate::{
  array::Array,
  error::Error,
  lm::quant::{PerLayerQuantization, Quantization, QuantizationOption},
  nn::Linear,
};

/// The global affine quantization config the SANM quantized-path oracles thread
/// in (the common SenseVoice scheme — a single global `{ group_size, bits }`).
/// Each builder resolves its own `(group_size, bits, mode)` per prefix from it
/// via `quantization_for`.
fn global_quant() -> PerLayerQuantization {
  PerLayerQuantization::from_global(Quantization::affine(QUANT_GROUP, QUANT_BITS))
}

// ───────────────────────── SinusoidalPositionEncoder ─────────────────────────

#[test]
fn sinusoidal_pe_closed_form() {
  // Reference (`sensevoice.py:106-122`): 1-indexed positions, half = D/2,
  // incr = ln(10000) / (half - 1), inv[j] = exp(-incr * j),
  // encoding[t, j] = sin(pos_t * inv[j]); encoding[t, half + j] = cos(...).
  // Then `x + encoding`. Use x = 0 so the output IS the encoding.
  let t = 4usize;
  let d = 8usize;
  let half = d / 2;
  let x = Array::from_slice::<f32>(&vec![0.0f32; t * d], &[1, t as i32, d as i32]).unwrap();
  let mut out = SinusoidalPositionEncoder.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, t, d]);
  let got = out.to_vec::<f32>().unwrap();

  let incr = (10000.0f64).ln() / ((half as f64) - 1.0);
  for ti in 0..t {
    let pos = (ti + 1) as f64; // 1-indexed
    for j in 0..half {
      let inv = (-incr * (j as f64)).exp();
      let angle = pos * inv;
      let want_sin = angle.sin() as f32;
      let want_cos = angle.cos() as f32;
      let got_sin = got[ti * d + j];
      let got_cos = got[ti * d + half + j];
      assert!(
        (got_sin - want_sin).abs() < 1e-5,
        "sin[{ti},{j}] got {got_sin} want {want_sin}"
      );
      assert!(
        (got_cos - want_cos).abs() < 1e-5,
        "cos[{ti},{j}] got {got_cos} want {want_cos}"
      );
    }
  }
}

#[test]
fn sinusoidal_pe_adds_to_input() {
  // With a non-zero input the output is `input + encoding`; the difference from
  // the zero-input encoding must equal the input exactly.
  let t = 3usize;
  let d = 4usize;
  let zeros = Array::from_slice::<f32>(&vec![0.0f32; t * d], &[1, t as i32, d as i32]).unwrap();
  let enc = SinusoidalPositionEncoder
    .forward(&zeros)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();

  let input: Vec<f32> = (0..t * d).map(|i| i as f32 * 0.1).collect();
  let x = Array::from_slice::<f32>(&input, &[1, t as i32, d as i32]).unwrap();
  let out = SinusoidalPositionEncoder
    .forward(&x)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  for i in 0..t * d {
    assert!((out[i] - (enc[i] + input[i])).abs() < 1e-5, "idx {i}");
  }
}

#[test]
fn sinusoidal_pe_preserves_bf16_dtype() {
  // The swift PE casts the encoding back to x.dtype before the add
  // (`SenseVoiceModel.swift:29`); a bf16 input must yield a bf16 output (not a
  // promoted f32). [preserve-activation-dtype].
  let x = Array::from_slice::<f32>(&[0.0f32; 2 * 4], &[1, 2, 4])
    .unwrap()
    .astype(crate::dtype::Dtype::BF16)
    .unwrap();
  let out = SinusoidalPositionEncoder.forward(&x).unwrap();
  assert_eq!(out.dtype().unwrap(), crate::dtype::Dtype::BF16);
}

#[test]
fn sinusoidal_pe_rejects_non_rank3() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
  assert!(matches!(
    SinusoidalPositionEncoder.forward(&bad),
    Err(Error::OutOfRange(_))
  ));
}

// ───────────────────────── FSMN depthwise conv ─────────────────────────

/// An independent channels-last depthwise convolution reference: for input
/// `(T, C)` and per-channel kernels `(C, K)`, with an asymmetric `(left, right)`
/// zero pad on the time axis, output `(T, C)` where
/// `out[t, c] = sum_k kernel[c, k] * padded[t + k, c]`. This is the depthwise
/// (`groups == C`) grouped conv the FSMN block runs, shared with no code under
/// test.
fn depthwise_conv_ref(
  input: &[Vec<f32>],
  kernels: &[Vec<f32>],
  left: usize,
  right: usize,
) -> Vec<Vec<f32>> {
  let t = input.len();
  let c = input[0].len();
  let k = kernels[0].len();
  // Zero-pad the time axis.
  let mut padded: Vec<Vec<f32>> = Vec::with_capacity(t + left + right);
  for _ in 0..left {
    padded.push(vec![0.0; c]);
  }
  padded.extend(input.iter().cloned());
  for _ in 0..right {
    padded.push(vec![0.0; c]);
  }
  // Valid conv -> output length (t + left + right) - k + 1 = t (since
  // left + right = k - 1).
  let mut out = vec![vec![0.0f32; c]; t];
  for (ti, out_row) in out.iter_mut().enumerate() {
    for (ci, slot) in out_row.iter_mut().enumerate() {
      let mut acc = 0.0f32;
      for ki in 0..k {
        acc += kernels[ci][ki] * padded[ti + ki][ci];
      }
      *slot = acc;
    }
  }
  out
}

/// Construct a `MultiHeadedAttentionSANM` with dummy linears and a chosen FSMN
/// kernel, for exercising the private `forward_fsmn` in isolation. `kernels` is
/// `(C, K)`; the MLX conv weight layout is `(C_out, K, C_in/groups) = (C, K, 1)`.
fn sanm_with_fsmn(
  n_feat: i32,
  n_head: i32,
  kernels: &[Vec<f32>],
  left: i32,
  right: i32,
) -> MultiHeadedAttentionSANM {
  let c = n_feat as usize;
  let k = kernels[0].len();
  // Pack kernels into MLX (C, K, 1) row-major: channel-major, then kernel.
  let mut flat = Vec::with_capacity(c * k);
  for ch in kernels {
    flat.extend_from_slice(ch);
  }
  let fsmn_weight = Array::from_slice::<f32>(&flat, &[n_feat, k as i32, 1]).unwrap();
  // Identity-ish dummy linears (never invoked by forward_fsmn).
  let id = |n: i32| {
    let mut w = vec![0.0f32; (n * n) as usize];
    for i in 0..n as usize {
      w[i * n as usize + i] = 1.0;
    }
    MaybeQuantizedLinear::Dense(Linear::new(
      Array::from_slice::<f32>(&w, &[n, n]).unwrap(),
      None,
    ))
  };
  let qkv = {
    let n = n_feat * 3;
    let mut w = vec![0.0f32; (n * n_feat) as usize];
    for i in 0..n_feat as usize {
      w[i * n_feat as usize + i] = 1.0;
    }
    MaybeQuantizedLinear::Dense(Linear::new(
      Array::from_slice::<f32>(&w, &[n, n_feat]).unwrap(),
      None,
    ))
  };
  MultiHeadedAttentionSANM {
    linear_q_k_v: qkv,
    linear_out: id(n_feat),
    fsmn_weight,
    n_head,
    d_k: n_feat / n_head,
    n_feat,
    left_padding: left,
    right_padding: right,
  }
}

#[test]
fn fsmn_zero_kernel_is_pure_residual() {
  // An all-zero FSMN kernel -> conv = 0 -> `_forward_fsmn` returns `inputs`
  // (the `+ inputs` residual shortcut, `sensevoice.py:176`).
  let n_feat = 2;
  let kernels = vec![vec![0.0; 3], vec![0.0; 3]];
  let sanm = sanm_with_fsmn(n_feat, 1, &kernels, 1, 1);
  let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]).unwrap();
  let mut out = sanm.forward_fsmn(&input).unwrap();
  assert_eq!(out.shape(), vec![1, 3, 2]);
  assert_eq!(
    out.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn fsmn_matches_independent_depthwise_conv_plus_residual() {
  // A non-trivial per-channel kernel, k=3, left=1, right=1 (the k=3 symmetric
  // split). Compare `forward_fsmn` to (independent depthwise conv) + inputs.
  let n_feat = 2;
  // Channel 0 kernel [1, 0, -1] (a centered difference); channel 1 [0.5,0.5,0].
  let kernels = vec![vec![1.0, 0.0, -1.0], vec![0.5, 0.5, 0.0]];
  let sanm = sanm_with_fsmn(n_feat, 1, &kernels, 1, 1);

  // Input (T=4, C=2), row-major (B=1, T, C).
  let rows = vec![
    vec![1.0f32, 10.0],
    vec![2.0, 20.0],
    vec![3.0, 30.0],
    vec![4.0, 40.0],
  ];
  let mut flat = Vec::new();
  for r in &rows {
    flat.extend_from_slice(r);
  }
  let input = Array::from_slice::<f32>(&flat, &[1, 4, 2]).unwrap();

  let mut got = sanm.forward_fsmn(&input).unwrap();
  assert_eq!(got.shape(), vec![1, 4, 2]);

  // Reference: depthwise conv (left=1,right=1) + inputs.
  let conv_ref = depthwise_conv_ref(&rows, &kernels, 1, 1);
  let mut want = Vec::new();
  for (ti, r) in rows.iter().enumerate() {
    for ci in 0..2 {
      want.push(conv_ref[ti][ci] + r[ci]);
    }
  }
  let got_flat = got.to_vec::<f32>().unwrap();
  for (g, w) in got_flat.iter().zip(want.iter()) {
    assert!((g - w).abs() < 1e-4, "fsmn got {g} want {w}");
  }
}

#[test]
fn fsmn_asymmetric_pad_k11_default() {
  // The real config: k=11, sanm_shift=0 -> left=5, right=5 (symmetric). Verify
  // forward_fsmn matches the independent conv for the default geometry on a
  // small single-channel input.
  let kernels = vec![(0..11).map(|i| i as f32 * 0.01).collect::<Vec<f32>>()];
  let sanm = sanm_with_fsmn(1, 1, &kernels, 5, 5);
  let rows: Vec<Vec<f32>> = (0..6).map(|t| vec![(t + 1) as f32]).collect();
  let mut flat = Vec::new();
  for r in &rows {
    flat.extend_from_slice(r);
  }
  let input = Array::from_slice::<f32>(&flat, &[1, 6, 1]).unwrap();
  let got = sanm.forward_fsmn(&input).unwrap().to_vec::<f32>().unwrap();
  let conv_ref = depthwise_conv_ref(&rows, &kernels, 5, 5);
  for (t, r) in rows.iter().enumerate() {
    let want = conv_ref[t][0] + r[0];
    assert!(
      (got[t] - want).abs() < 1e-4,
      "t={t} got {} want {want}",
      got[t]
    );
  }
}

// ───────────────────────── SANM attention shape ─────────────────────────

#[test]
fn sanm_attention_forward_shape() {
  // Full `MultiHeadedAttentionSANM::forward` over (B=1, T=5, in_feat=8) with
  // n_feat=8, h=2 -> output (1, 5, 8). Built from a synthetic weight map.
  let mut w = HashMap::new();
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  let sanm = build_sanm(&mut w, "blk.self_attn", &enc, 8);
  let x = Array::from_slice::<f32>(
    &(0..5 * 8).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
    &[1, 5, 8],
  )
  .unwrap();
  let out = sanm.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 5, 8]);
}

// ───────────────────────── EncoderLayerSANM residual rule ─────────────────────────

#[test]
fn encoder_layer_width_change_drops_attn_residual() {
  // For in_size != size, the residual around attention is dropped
  // (`sensevoice.py:227-230`): the layer's `residual_attn` flag is false, so
  // the post-attention value is `attn_out` (not `residual + attn_out`).
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let layer_wc = build_encoder_layer_with_zero_attn(4, &enc); // in=4 != size=8
  let layer_same = build_encoder_layer_with_zero_attn(8, &enc); // in=8 == size=8

  // With a self-attention forced to 0 (zero out/qkv weights) and identity
  // norms, the post-attention value is `residual + 0 = residual` (same-width)
  // or `0` (width-change). The downstream FFN is also forced to 0, so the final
  // output is the post-attention value unchanged: width-change -> 0,
  // same-width -> input.
  let x8 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 8]).unwrap();

  // Same-width path: input flows through (residual kept, attn=0, ffn=0).
  let out_same = layer_same.forward(&x8).unwrap().to_vec::<f32>().unwrap();
  assert_eq!(out_same, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

  // Width-change path: project a (1,1,4) input; the dropped residual means the
  // output is purely `attn_out (=0) + ffn(=0) = 0`.
  let x4 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
  let out_wc = layer_wc.forward(&x4).unwrap().to_vec::<f32>().unwrap();
  assert_eq!(out_wc, vec![0.0; 8], "width-change drops the attn residual");
}

// ───────────────────────── full encoder tower shape ─────────────────────────

#[test]
fn encoder_tower_output_shape() {
  // A tiny but structurally complete tower: input_size=4, output_size=8,
  // num_blocks=2 (encoders0 + 1 encoders), tp_blocks=1. Input (B=1, T=6, 4) ->
  // output (1, 6, 8).
  let enc = synthetic_encoder_config(8, 2, 16, 2, 1, 3);
  let mut w = HashMap::new();
  build_full_tower_weights(&mut w, 4, &enc);
  let encoder = Encoder::from_weights(&mut w, 4, &enc, None).unwrap();
  assert_eq!(encoder.output_size(), 8);

  let x = Array::from_slice::<f32>(
    &(0..6 * 4).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
    &[1, 6, 4],
  )
  .unwrap();
  let out = encoder.forward(&x).unwrap();
  assert_eq!(out.shape(), vec![1, 6, 8]);

  // The weight map is fully consumed by the tower build (no stray keys).
  assert!(
    w.is_empty(),
    "all tower weights consumed; leftover: {:?}",
    w.keys()
  );
}

#[test]
fn encoder_tower_consumes_tp_blocks_zero() {
  // tp_blocks = 0: no second stage, but after_norm + tp_norm still present.
  let enc = synthetic_encoder_config(8, 2, 16, 2, 0, 3);
  let mut w = HashMap::new();
  build_full_tower_weights(&mut w, 4, &enc);
  let encoder = Encoder::from_weights(&mut w, 4, &enc, None).unwrap();
  let x = Array::from_slice::<f32>(
    &(0..3 * 4).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
    &[1, 3, 4],
  )
  .unwrap();
  assert_eq!(encoder.forward(&x).unwrap().shape(), vec![1, 3, 8]);
}

#[test]
fn encoder_fallible_reserve_builds_full_block_set() {
  // The fallibly-reserved `encoders` / `tp_encoders` `Vec`s build the COMPLETE
  // block set at a normal count: num_blocks = 4 -> encoders0 (1) + encoders (3);
  // tp_blocks = 3 -> tp_encoders (3). The tower forwards and consumes every key.
  let enc = synthetic_encoder_config(8, 2, 16, 4, 3, 3);
  let mut w = HashMap::new();
  build_full_tower_weights(&mut w, 4, &enc);
  let encoder = Encoder::from_weights(&mut w, 4, &enc, None).unwrap();

  // The reserved-then-filled block stacks have exactly the configured counts.
  assert_eq!(
    encoder.encoders0.len(),
    1,
    "encoders0 holds the first block"
  );
  assert_eq!(
    encoder.encoders.len(),
    3,
    "num_blocks - 1 = 3 constant blocks"
  );
  assert_eq!(encoder.tp_encoders.len(), 3, "tp_blocks = 3");

  let x = Array::from_slice::<f32>(
    &(0..5 * 4).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
    &[1, 5, 4],
  )
  .unwrap();
  assert_eq!(encoder.forward(&x).unwrap().shape(), vec![1, 5, 8]);
  assert!(w.is_empty(), "all tower weights consumed: {:?}", w.keys());
}

#[test]
fn encoder_missing_weight_is_typed_error() {
  // An incomplete weight map surfaces a typed MissingKey, not a panic.
  let enc = synthetic_encoder_config(8, 2, 16, 2, 0, 3);
  let mut w = HashMap::new();
  build_full_tower_weights(&mut w, 4, &enc);
  w.remove("encoder.tp_norm.weight");
  assert!(matches!(
    Encoder::from_weights(&mut w, 4, &enc, None),
    Err(Error::MissingKey(_))
  ));
}

// ───────────────────────── quantized linear path ─────────────────────────

#[test]
fn sanm_loads_and_forwards_quantized_linear() {
  // A `.scales` sibling on a SANM linear makes `MaybeQuantizedLinear` build the
  // quantized variant; the block loads and forwards. Uses a real affine
  // quantize of a dense weight so the triple is structurally valid.
  // `group_size = 64` needs `in_features >= 64`, so use `n_feat = 64`.
  let n_feat = 64i32;
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  // The fused QKV is quantized; the rest dense.
  insert_quantized_linear(&mut w, "blk.self_attn.linear_q_k_v", n_feat * 3, n_feat);
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );
  let sanm = MultiHeadedAttentionSANM::from_weights(
    &mut w,
    "blk.self_attn",
    &enc,
    n_feat,
    Some(&global_quant()),
  )
  .expect("quantized SANM build");
  assert!(matches!(
    sanm.linear_q_k_v,
    MaybeQuantizedLinear::Quantized(_)
  ));
  assert!(matches!(sanm.linear_out, MaybeQuantizedLinear::Dense(_)));

  let x = Array::from_slice::<f32>(
    &(0..4 * n_feat)
      .map(|i| (i as f32) * 0.01)
      .collect::<Vec<_>>(),
    &[1, 4, n_feat],
  )
  .unwrap();
  assert_eq!(
    sanm.forward(&x).unwrap().shape(),
    vec![1, 4, n_feat as usize]
  );
}

#[test]
fn quantized_scales_without_scheme_is_typed_error() {
  // A `.scales` sibling but `quant = None` (config says dense) is a typed
  // InvariantViolation, not a guess.
  let n_feat = 64i32;
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  insert_quantized_linear(&mut w, "blk.self_attn.linear_q_k_v", n_feat * 3, n_feat);
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, n_feat, None),
    Err(Error::InvariantViolation(_))
  ));
}

// ───────────────────────── load-time weight shape-pins ─────────────────────────

#[test]
fn sanm_rejects_wrong_qkv_shape() {
  // The fused `linear_q_k_v` must be `(3 * output_size, in_feat)`
  // (`sensevoice.py:152`). A wrong out-feature count is a typed shape error at
  // LOAD, not a deferred split / reshape failure at the first forward.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  // Overwrite the qkv with a wrong out width (3*8 = 24 expected; give 20).
  insert_dense_linear(&mut w, "blk.self_attn.linear_q_k_v", 20, 8);
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn sanm_rejects_wrong_qkv_in_feat() {
  // The qkv input width must equal the layer input width `in_feat`. A first
  // (width-changing) block consumes `in_feat = input_size`; a shard whose qkv
  // input width disagrees is a typed shape error at load.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  // Build qkv with in_feat = 6 but load with in_feat = 8.
  insert_dense_linear(&mut w, "blk.self_attn.linear_q_k_v", 24, 6);
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn sanm_rejects_wrong_linear_out_shape() {
  // `linear_out` must be `(output_size, output_size)` (`sensevoice.py:151`).
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", 8, 6); // (8, 6) != (8, 8)
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn sanm_rejects_wrong_fsmn_weight_shape() {
  // The depthwise FSMN conv weight must be `(output_size, kernel_size, 1)` in the
  // post-`sanitize` MLX layout. A wrong kernel extent is a typed shape error.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  // kernel_size = 3 expected; plant a (8, 5, 1) weight.
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (8 * 5) as usize], &[8, 5, 1]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn sanm_rejects_non_rank3_fsmn_weight() {
  // A FSMN conv weight of the wrong RANK (e.g. the pre-sanitize torch `(C, 1, K)`
  // left un-transposed would still be rank-3, but a rank-2 weight is a typed
  // RankMismatch) — the conv would otherwise mis-group at the first forward.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (8 * 3) as usize], &[8, 3]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::RankMismatch(_))
  ));
}

#[test]
fn sanm_qkv_fused_width_overflow_is_typed_not_panic() {
  // The fused QKV width `3 * output_size` is computed with checked arithmetic; an
  // `output_size` near `i32::MAX / 3` is a typed ArithmeticOverflow at load (the
  // check runs BEFORE any weight lookup, so an empty map suffices).
  let enc = synthetic_encoder_config(i32::MAX, 1, 16, 1, 0, 3);
  let mut w = HashMap::new();
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::ArithmeticOverflow(_))
  ));
}

#[test]
fn ffn_rejects_wrong_w1_shape() {
  // `feed_forward.w_1` must be `(linear_units, output_size)` (`sensevoice.py:128`).
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  // A full encoder layer, then corrupt w_1's out width (linear_units = 16).
  insert_encoder_layer(&mut w, "blk", 8, &enc);
  insert_dense_linear(&mut w, "blk.feed_forward.w_1", 12, 8); // (12, 8) != (16, 8)
  assert!(matches!(
    EncoderLayerSANM::from_weights(&mut w, "blk", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn ffn_rejects_wrong_w2_shape() {
  // `feed_forward.w_2` must be `(output_size, linear_units)` (`sensevoice.py:129`).
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_encoder_layer(&mut w, "blk", 8, &enc);
  insert_dense_linear(&mut w, "blk.feed_forward.w_2", 8, 12); // (8, 12) != (8, 16)
  assert!(matches!(
    EncoderLayerSANM::from_weights(&mut w, "blk", &enc, 8, None),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn layer_norm_rejects_wrong_affine_length() {
  // Each `nn.LayerNorm(dim)` affine vector must be length `dim`: `norm1` is sized
  // to the input width, `norm2` to the output. A mis-sized affine would broadcast
  // silently against the activations; pin it at load.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_encoder_layer(&mut w, "blk", 8, &enc);
  // norm2 must be length 8 (output_size); plant a length-6 weight.
  w.insert(
    "blk.norm2.weight".to_string(),
    Array::from_slice::<f32>(&[1.0f32; 6], &[6]).unwrap(),
  );
  assert!(matches!(
    EncoderLayerSANM::from_weights(&mut w, "blk", &enc, 8, None),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn after_norm_rejects_wrong_affine_length() {
  // The tower's `after_norm` / `tp_norm` are `(output_size,)`; a mis-sized
  // affine is a typed LengthMismatch at the full-tower build.
  let enc = synthetic_encoder_config(8, 2, 16, 2, 0, 3);
  let mut w = HashMap::new();
  build_full_tower_weights(&mut w, 4, &enc);
  w.insert(
    "encoder.after_norm.weight".to_string(),
    Array::from_slice::<f32>(&[1.0f32; 7], &[7]).unwrap(),
  );
  assert!(matches!(
    Encoder::from_weights(&mut w, 4, &enc, None),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn quantized_sanm_rejects_wrong_qkv_shape() {
  // The shape-pin applies to the QUANTIZED arm too (via the dequantized logical
  // shape): a packed qkv whose recovered `(3*n_feat, in_feat)` disagrees is a
  // typed shape error at load, not a deferred mis-projection.
  let n_feat = 64i32;
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  // Quantize a qkv with the WRONG out width (3*n_feat = 192 expected; give 128).
  insert_quantized_linear(&mut w, "blk.self_attn.linear_q_k_v", 128, n_feat);
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(
      &mut w,
      "blk.self_attn",
      &enc,
      n_feat,
      Some(&global_quant())
    ),
    Err(Error::ShapePairMismatch(_))
  ));
}

#[test]
fn sanm_resolves_quant_per_prefix_skip_builds_dense_layer_dense() {
  // Per-prefix resolution: with a global default scheme PLUS a `Skip` override
  // for `linear_out`, `linear_q_k_v` (a `.scales`-bearing layer) builds quantized
  // via the global default while `linear_out` (DENSE on disk, no `.scales`) is
  // left dense — its `Skip` prefix resolves to `None`, consistent with its dense
  // weights. A single collapsed global tuple could not express the per-layer
  // `Skip`; the per-prefix `None` is what threads through.
  let n_feat = 64i32;
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  // `linear_q_k_v` is quantized (carries `.scales`); `linear_out` is DENSE.
  insert_quantized_linear(&mut w, "blk.self_attn.linear_q_k_v", n_feat * 3, n_feat);
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );

  // Global affine default, but `linear_out` is explicitly skipped (matching its
  // on-disk dense weights).
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "blk.self_attn.linear_out".to_string(),
    QuantizationOption::Skip,
  );
  let quant = PerLayerQuantization::new(
    Some(Quantization::affine(QUANT_GROUP, QUANT_BITS)),
    per_layer,
  );

  let sanm =
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, n_feat, Some(&quant))
      .expect("per-prefix resolution: quantized qkv via global default, dense out via Skip");
  // `linear_q_k_v` resolved the global default → quantized; `linear_out`
  // resolved `Skip` → its prefix's tuple is `None`, and with no `.scales` the
  // dense arm builds.
  assert!(matches!(
    sanm.linear_q_k_v,
    MaybeQuantizedLinear::Quantized(_)
  ));
  assert!(
    matches!(sanm.linear_out, MaybeQuantizedLinear::Dense(_)),
    "a per-layer Skip on a dense-on-disk `linear_out` keeps it dense"
  );
}

#[test]
fn sanm_per_prefix_skip_on_scales_bearing_layer_is_typed_error() {
  // The dual of the above: a `Skip` override on a layer that DOES carry `.scales`
  // on disk is a checkpoint/config inconsistency — the per-prefix resolution
  // yields `None` for `linear_out` while its `.scales` sibling is present, which
  // the shared `MaybeQuantizedLinear` rejects with a typed `InvariantViolation`
  // (the per-prefix `None` reached the leaf, exactly the qwen3 contract).
  let n_feat = 64i32;
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  insert_quantized_linear(&mut w, "blk.self_attn.linear_q_k_v", n_feat * 3, n_feat);
  insert_quantized_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "blk.self_attn.linear_out".to_string(),
    QuantizationOption::Skip,
  );
  let quant = PerLayerQuantization::new(
    Some(Quantization::affine(QUANT_GROUP, QUANT_BITS)),
    per_layer,
  );
  assert!(
    matches!(
      MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, n_feat, Some(&quant)),
      Err(Error::InvariantViolation(_))
    ),
    "a Skip on a `.scales`-bearing layer is an InvariantViolation, not a silent quantized build"
  );
}

#[test]
fn sanm_resolves_quant_per_prefix_parameter_override() {
  // Per-prefix resolution honors a per-layer PARAMETER override: `linear_q_k_v`
  // is quantized at `group_size = 32` (its `.scales` width is `in/32 = 2`) while
  // the global default is `group_size = 64` (`in/64 = 1`). Resolving the WRONG
  // (global) tuple for `linear_q_k_v` would fail `check_quantized_shape` (the
  // scales-width recovery would expect 1 group, not 2). The build succeeding is
  // proof the OVERRIDE tuple — not the collapsed global tuple — was used.
  const OVERRIDE_GROUP: i32 = 32;
  let n_feat = 64i32; // multiple of both 32 and 64
  let enc = synthetic_encoder_config(n_feat, 4, 128, 1, 0, 3);
  let mut w = HashMap::new();
  insert_quantized_linear_params(
    &mut w,
    "blk.self_attn.linear_q_k_v",
    n_feat * 3,
    n_feat,
    OVERRIDE_GROUP,
    QUANT_BITS,
  );
  insert_dense_linear(&mut w, "blk.self_attn.linear_out", n_feat, n_feat);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );

  // Global default group_size = 64; `linear_q_k_v` overridden to group_size = 32.
  let mut per_layer = HashMap::new();
  per_layer.insert(
    "blk.self_attn.linear_q_k_v".to_string(),
    QuantizationOption::Quantize(Quantization::affine(OVERRIDE_GROUP, QUANT_BITS)),
  );
  let quant = PerLayerQuantization::new(
    Some(Quantization::affine(QUANT_GROUP, QUANT_BITS)),
    per_layer,
  );

  let sanm =
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, n_feat, Some(&quant))
      .expect("the per-layer group_size override must be used, not the global default");
  assert!(matches!(
    sanm.linear_q_k_v,
    MaybeQuantizedLinear::Quantized(_)
  ));

  // Cross-check: resolving the GLOBAL tuple (group_size = 64) for this packed
  // `group_size = 32` weight is a load-time shape error — confirming the two
  // schemes are genuinely distinguishable (the override is load-bearing).
  let mut w_wrong = HashMap::new();
  insert_quantized_linear_params(
    &mut w_wrong,
    "blk.self_attn.linear_q_k_v",
    n_feat * 3,
    n_feat,
    OVERRIDE_GROUP,
    QUANT_BITS,
  );
  insert_dense_linear(&mut w_wrong, "blk.self_attn.linear_out", n_feat, n_feat);
  w_wrong.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * 3) as usize], &[n_feat, 3, 1]).unwrap(),
  );
  let wrong = MultiHeadedAttentionSANM::from_weights(
    &mut w_wrong,
    "blk.self_attn",
    &enc,
    n_feat,
    Some(&global_quant()), // group_size = 64, the WRONG scheme for this layer
  );
  assert!(
    wrong.is_err(),
    "the global group_size must mis-decode the group_size=32 packed weight"
  );
}

// ───────────────────────── dense Linear bias shape-pins ─────────────────────────

#[test]
fn sanm_rejects_wrong_linear_out_bias_length() {
  // The dense arm of `MaybeQuantizedLinear` does NOT validate the optional
  // `<prefix>.bias`; SenseVoice pins it locally. `linear_out` is
  // `(output_size, output_size)`, so its bias must be rank-1 `(output_size,)`. A
  // wrong-length bias (correct weight) would broadcast a single wrong offset
  // across the channels — a typed LengthMismatch at LOAD.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  // Correct weight (8, 8); overwrite ONLY the bias to a wrong length (6 != 8).
  w.insert(
    "blk.self_attn.linear_out.bias".to_string(),
    Array::from_slice::<f32>(&[0.0f32; 6], &[6]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn sanm_rejects_scalar_qkv_bias() {
  // A stray `(1,)` fused-QKV bias (correct weight) would broadcast one offset
  // across all `3 * output_size` channels — the SILENT-wrong-output case. Pinned
  // to a typed LengthMismatch at load (expected length `3 * 8 = 24`).
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  w.insert(
    "blk.self_attn.linear_q_k_v.bias".to_string(),
    Array::from_slice::<f32>(&[0.0f32], &[1]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn sanm_rejects_rank2_linear_out_bias() {
  // A `(1, out)` bias (rank-2, correct weight) is a typed RankMismatch — the
  // mlx.nn.Linear bias must be the rank-1 `(out_features,)` vector.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  w.insert(
    "blk.self_attn.linear_out.bias".to_string(),
    Array::from_slice::<f32>(&[0.0f32; 8], &[1, 8]).unwrap(),
  );
  assert!(matches!(
    MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None),
    Err(Error::RankMismatch(_))
  ));
}

#[test]
fn ffn_rejects_wrong_w1_bias_length() {
  // `feed_forward.w_1` is `(linear_units, output_size)`, so its dense bias must
  // be `(linear_units,)`. A wrong-length bias (correct weight) is pinned at load.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_encoder_layer(&mut w, "blk", 8, &enc);
  // linear_units = 16; plant a length-12 bias on the otherwise-correct w_1.
  w.insert(
    "blk.feed_forward.w_1.bias".to_string(),
    Array::from_slice::<f32>(&[0.0f32; 12], &[12]).unwrap(),
  );
  assert!(matches!(
    EncoderLayerSANM::from_weights(&mut w, "blk", &enc, 8, None),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn sanm_accepts_correct_dense_biases() {
  // The bias-pin is a no-op for the well-formed checkpoint the other helpers
  // build (every `insert_dense_linear` writes a correct `(out,)` bias): the SANM
  // still loads + forwards. Guards against the pin rejecting valid biases.
  let enc = synthetic_encoder_config(8, 2, 16, 1, 0, 3);
  let mut w = HashMap::new();
  insert_sanm(&mut w, "blk.self_attn", 8, &enc);
  let sanm = MultiHeadedAttentionSANM::from_weights(&mut w, "blk.self_attn", &enc, 8, None)
    .expect("correct dense biases load");
  let x = Array::from_slice::<f32>(
    &(0..5 * 8).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
    &[1, 5, 8],
  )
  .unwrap();
  assert_eq!(sanm.forward(&x).unwrap().shape(), vec![1, 5, 8]);
}

// ───────────────────────── test helpers ─────────────────────────

const QUANT_GROUP: i32 = 64;
const QUANT_BITS: i32 = 4;

/// Build an [`EncoderConfig`] from explicit dims by round-tripping through
/// JSON (the only public construction path), so the test config is the same
/// shape a real `config.json` produces.
fn synthetic_encoder_config(
  output_size: i32,
  attention_heads: i32,
  linear_units: i32,
  num_blocks: i32,
  tp_blocks: i32,
  kernel_size: i32,
) -> EncoderConfig {
  let json = format!(
    r#"{{ "output_size": {output_size}, "attention_heads": {attention_heads},
          "linear_units": {linear_units}, "num_blocks": {num_blocks},
          "tp_blocks": {tp_blocks}, "kernel_size": {kernel_size}, "sanm_shift": 0 }}"#
  );
  serde_json::from_str(&json).expect("encoder config")
}

/// An identity-`(out, in)` dense Linear weight, padded/truncated to shape: row
/// `i` has a `1.0` at column `i` (when `i < in`). Used so synthetic linears are
/// well-conditioned (not all-zero) for the shape passes.
fn identity_linear_weight(out: i32, in_features: i32) -> Array {
  let mut w = vec![0.0f32; (out * in_features) as usize];
  for i in 0..(out.min(in_features) as usize) {
    w[i * in_features as usize + i] = 1.0;
  }
  Array::from_slice::<f32>(&w, &[out, in_features]).unwrap()
}

/// Insert a dense `<prefix>.weight` (identity) + `<prefix>.bias` (zeros).
fn insert_dense_linear(w: &mut HashMap<String, Array>, prefix: &str, out: i32, in_features: i32) {
  w.insert(
    format!("{prefix}.weight"),
    identity_linear_weight(out, in_features),
  );
  w.insert(
    format!("{prefix}.bias"),
    Array::from_slice::<f32>(&vec![0.0f32; out as usize], &[out]).unwrap(),
  );
}

/// Insert a `<prefix>.weight` (identity) with NO bias (for the FFN-less
/// width-change probe, which still needs a bias for `nn.Linear`).
fn insert_zero_linear(w: &mut HashMap<String, Array>, prefix: &str, out: i32, in_features: i32) {
  w.insert(
    format!("{prefix}.weight"),
    Array::from_slice::<f32>(
      &vec![0.0f32; (out * in_features) as usize],
      &[out, in_features],
    )
    .unwrap(),
  );
  w.insert(
    format!("{prefix}.bias"),
    Array::from_slice::<f32>(&vec![0.0f32; out as usize], &[out]).unwrap(),
  );
}

/// Insert a real affine-quantized `<prefix>.{weight,scales,biases,bias}` by
/// quantizing an identity-ish dense weight, so the triple is structurally
/// valid for `QuantizedLinear::from_parts`. `in_features` must be a multiple of
/// [`QUANT_GROUP`].
fn insert_quantized_linear(
  w: &mut HashMap<String, Array>,
  prefix: &str,
  out: i32,
  in_features: i32,
) {
  let dense = identity_linear_weight(out, in_features);
  let (packed, scales, biases) =
    crate::ops::quantized::quantize(&dense, QUANT_GROUP, QUANT_BITS, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), packed);
  w.insert(format!("{prefix}.scales"), scales);
  // Affine mode always yields per-group biases.
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine quantize yields biases"),
  );
  w.insert(
    format!("{prefix}.bias"),
    Array::from_slice::<f32>(&vec![0.0f32; out as usize], &[out]).unwrap(),
  );
}

/// Insert a real affine-quantized `<prefix>.{weight,scales,biases,bias}` at an
/// EXPLICIT `(group_size, bits)` (not the [`QUANT_GROUP`] / [`QUANT_BITS`]
/// default) — used by the per-layer-override oracle to plant a layer whose
/// scheme differs from the global default, so resolving the wrong (global) tuple
/// for it would mis-shape / mis-decode the packed triple.
fn insert_quantized_linear_params(
  w: &mut HashMap<String, Array>,
  prefix: &str,
  out: i32,
  in_features: i32,
  group_size: i32,
  bits: i32,
) {
  let dense = identity_linear_weight(out, in_features);
  let (packed, scales, biases) =
    crate::ops::quantized::quantize(&dense, group_size, bits, "affine", None).unwrap();
  w.insert(format!("{prefix}.weight"), packed);
  w.insert(format!("{prefix}.scales"), scales);
  w.insert(
    format!("{prefix}.biases"),
    biases.expect("affine quantize yields biases"),
  );
  w.insert(
    format!("{prefix}.bias"),
    Array::from_slice::<f32>(&vec![0.0f32; out as usize], &[out]).unwrap(),
  );
}

/// Insert the `<prefix>.{linear_q_k_v, linear_out, fsmn_block.weight}` of a
/// SANM attention (all dense, identity linears, zero FSMN kernel).
fn insert_sanm(w: &mut HashMap<String, Array>, prefix: &str, in_feat: i32, enc: &EncoderConfig) {
  let n_feat = enc.output_size();
  let k = enc.kernel_size();
  insert_dense_linear(w, &format!("{prefix}.linear_q_k_v"), n_feat * 3, in_feat);
  insert_dense_linear(w, &format!("{prefix}.linear_out"), n_feat, n_feat);
  w.insert(
    format!("{prefix}.fsmn_block.weight"),
    Array::from_slice::<f32>(&vec![0.0f32; (n_feat * k) as usize], &[n_feat, k, 1]).unwrap(),
  );
}

/// Build a `MultiHeadedAttentionSANM` from a weight map (thin wrapper for
/// readability in the shape test).
fn build_sanm(
  w: &mut HashMap<String, Array>,
  prefix: &str,
  enc: &EncoderConfig,
  in_feat: i32,
) -> MultiHeadedAttentionSANM {
  MultiHeadedAttentionSANM::from_weights(w, prefix, enc, in_feat, None).unwrap()
}

/// Identity LayerNorm params (`weight = 1`, `bias = 0`) at `dim`.
fn insert_layer_norm(w: &mut HashMap<String, Array>, prefix: &str, dim: i32) {
  w.insert(
    format!("{prefix}.weight"),
    Array::from_slice::<f32>(&vec![1.0f32; dim as usize], &[dim]).unwrap(),
  );
  w.insert(
    format!("{prefix}.bias"),
    Array::from_slice::<f32>(&vec![0.0f32; dim as usize], &[dim]).unwrap(),
  );
}

/// Insert one EncoderLayerSANM's weights at `<prefix>` with input width
/// `in_size` (dense identity linears, zero FSMN, identity norms).
fn insert_encoder_layer(
  w: &mut HashMap<String, Array>,
  prefix: &str,
  in_size: i32,
  enc: &EncoderConfig,
) {
  let size = enc.output_size();
  insert_sanm(w, &format!("{prefix}.self_attn"), in_size, enc);
  insert_dense_linear(
    w,
    &format!("{prefix}.feed_forward.w_1"),
    enc.linear_units(),
    size,
  );
  insert_dense_linear(
    w,
    &format!("{prefix}.feed_forward.w_2"),
    size,
    enc.linear_units(),
  );
  insert_layer_norm(w, &format!("{prefix}.norm1"), in_size);
  insert_layer_norm(w, &format!("{prefix}.norm2"), size);
}

/// Build a full synthetic weight map for the whole tower at `input_size`.
fn build_full_tower_weights(w: &mut HashMap<String, Array>, input_size: i32, enc: &EncoderConfig) {
  let size = enc.output_size();
  insert_encoder_layer(w, "encoder.encoders0.0", input_size, enc);
  for i in 0..(enc.num_blocks() - 1) {
    insert_encoder_layer(w, &format!("encoder.encoders.{i}"), size, enc);
  }
  insert_layer_norm(w, "encoder.after_norm", size);
  for i in 0..enc.tp_blocks() {
    insert_encoder_layer(w, &format!("encoder.tp_encoders.{i}"), size, enc);
  }
  insert_layer_norm(w, "encoder.tp_norm", size);
}

/// Build a single EncoderLayerSANM whose attention AND feed-forward are forced
/// to zero (zero qkv/out/w_1/w_2 weights) with identity norms, at input width
/// `in_size`. Used to isolate the residual rule from the (zeroed) sub-layers.
fn build_encoder_layer_with_zero_attn(in_size: i32, enc: &EncoderConfig) -> EncoderLayerSANM {
  let size = enc.output_size();
  let k = enc.kernel_size();
  let mut w = HashMap::new();
  // Zero qkv/out so attention contributes 0.
  insert_zero_linear(&mut w, "blk.self_attn.linear_q_k_v", size * 3, in_size);
  insert_zero_linear(&mut w, "blk.self_attn.linear_out", size, size);
  w.insert(
    "blk.self_attn.fsmn_block.weight".to_string(),
    Array::from_slice::<f32>(&vec![0.0f32; (size * k) as usize], &[size, k, 1]).unwrap(),
  );
  // Zero feed-forward so the FFN contributes 0.
  insert_zero_linear(&mut w, "blk.feed_forward.w_1", enc.linear_units(), size);
  insert_zero_linear(&mut w, "blk.feed_forward.w_2", size, enc.linear_units());
  insert_layer_norm(&mut w, "blk.norm1", in_size);
  insert_layer_norm(&mut w, "blk.norm2", size);
  EncoderLayerSANM::from_weights(&mut w, "blk", enc, in_size, None).unwrap()
}
