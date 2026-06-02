use super::*;

use crate::ops::quantized;

/// `group_size=64` needs at least one full group along the last (input) axis,
/// so the quantized fixtures use `in_features = 64`.
const QUANT_IN: usize = 64;

/// A dense `(out_features, in_features)` weight with a smooth ramp — friendly
/// to 4-bit affine quantization (matches the tolerance fixtures in
/// `tests/ops_quantized.rs` and `lm::nn::switch`'s parity tests).
fn dense_weight(out: usize) -> Array {
  let mut data = Vec::with_capacity(out * QUANT_IN);
  for o in 0..out {
    for i in 0..QUANT_IN {
      data.push(((o * 10 + i) as f32) * 0.001);
    }
  }
  Array::from_slice::<f32>(&data, &(out, QUANT_IN)).unwrap()
}

/// An input `(n, in_features)` ramp.
fn input(n: usize) -> Array {
  let mut data = Vec::with_capacity(n * QUANT_IN);
  for ni in 0..n {
    for i in 0..QUANT_IN {
      data.push(((ni * 50 + i) as f32) * 0.01);
    }
  }
  Array::from_slice::<f32>(&data, &(n, QUANT_IN)).unwrap()
}

/// Independent reference oracle for a quantized forward: dequantize the
/// packed triple back to a dense `(out, in)` weight, then run the plain
/// `x @ weightᵀ (+ bias)`. This shares no code with [`QuantizedLinear::forward`]
/// (which dispatches `quantized_matmul`), so agreement within the quantization
/// error band is a genuine cross-check.
#[allow(clippy::too_many_arguments)]
fn dequant_then_matmul(
  x: &Array,
  w_q: &Array,
  scales: &Array,
  q_biases: Option<&Array>,
  bias: Option<&Array>,
  group_size: i32,
  bits: i32,
  mode: &str,
) -> Vec<f32> {
  let dense =
    quantized::dequantize(w_q, scales, q_biases, group_size, bits, mode, None, None).unwrap();
  let wt = dense.transpose().unwrap();
  let mut y = x.matmul(&wt).unwrap();
  if let Some(b) = bias {
    y = y.add(b).unwrap();
  }
  y.to_vec::<f32>().unwrap()
}

/// Assert per-element drift between a quantized forward and the
/// dequantize-then-matmul reference stays within a band relative to the
/// reference magnitude (4-bit affine quant on a smooth ramp).
fn assert_within_quant_error(reference: &[f32], got: &[f32]) {
  assert_eq!(reference.len(), got.len(), "length mismatch");
  let max_abs = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
  for (r, g) in reference.iter().zip(got.iter()) {
    assert!(
      (r - g).abs() <= 0.1 * max_abs + 1e-3,
      "quantized Linear drift too large: reference={r} got={g}"
    );
  }
}

// ─────────────────────────── dense Linear ───────────────────────────

#[test]
fn linear_forward_no_bias() {
  // 2x3 weight, project a (1, 3) input.
  let weight = Array::from_slice::<f32>(
    &[
      1.0, 0.0, 0.0, // row 0 selects feature 0
      0.0, 0.0, 1.0, // row 1 selects feature 2
    ],
    &(2usize, 3usize),
  )
  .unwrap();
  let layer = Linear::new(weight, None);
  let x = Array::from_slice::<f32>(&[10.0, 20.0, 30.0], &(1usize, 3usize)).unwrap();
  let mut y = layer.forward(&x).unwrap();
  assert_eq!(y.shape(), vec![1, 2]);
  assert_eq!(y.to_vec::<f32>().unwrap(), vec![10.0, 30.0]);
}

#[test]
fn linear_forward_with_bias() {
  let weight =
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &(2usize, 3usize)).unwrap();
  let bias = Array::from_slice::<f32>(&[100.0, 200.0], &(2usize,)).unwrap();
  let layer = Linear::new(weight, Some(bias));
  let x = Array::from_slice::<f32>(&[10.0, 20.0, 30.0], &(1usize, 3usize)).unwrap();
  let mut y = layer.forward(&x).unwrap();
  assert_eq!(y.to_vec::<f32>().unwrap(), vec![110.0, 230.0]);
}

// ──────────────────── QuantizedLinear forward parity ────────────────────

#[test]
fn quantized_linear_forward_matches_dequant_matmul_no_bias() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  assert!(q_biases.is_some(), "affine produces per-group biases");

  let layer = QuantizedLinear::from_parts(
    w_q.try_clone().unwrap(),
    scales.try_clone().unwrap(),
    q_biases.as_ref().map(|b| b.try_clone().unwrap()),
    None,
    64,
    4,
    "affine",
  )
  .unwrap();

  let x = input(2);
  let mut got = layer.forward(&x).unwrap();
  assert_eq!(got.shape(), vec![2, 8]);
  let reference = dequant_then_matmul(&x, &w_q, &scales, q_biases.as_ref(), None, 64, 4, "affine");
  assert_within_quant_error(&reference, &got.to_vec::<f32>().unwrap());
}

#[test]
fn quantized_linear_forward_matches_dequant_matmul_with_bias() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // A dense output bias (distinct from the per-group affine biases).
  let bias =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(8usize,)).unwrap();

  let layer = QuantizedLinear::from_parts(
    w_q.try_clone().unwrap(),
    scales.try_clone().unwrap(),
    q_biases.as_ref().map(|b| b.try_clone().unwrap()),
    Some(bias.try_clone().unwrap()),
    64,
    4,
    "affine",
  )
  .unwrap();

  let x = input(2);
  let mut got = layer.forward(&x).unwrap();
  let reference = dequant_then_matmul(
    &x,
    &w_q,
    &scales,
    q_biases.as_ref(),
    Some(&bias),
    64,
    4,
    "affine",
  );
  assert_within_quant_error(&reference, &got.to_vec::<f32>().unwrap());
}

// ─────────────────── QuantizedLinear structural validation ───────────────────

#[test]
fn quantized_linear_from_parts_rejects_non_u32_weight() {
  // A rank-2 dense f32 weight with otherwise-plausible scales must be rejected
  // (quantized_matmul requires a uint32 packed weight).
  let dense_w = dense_weight(8);
  let (_w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err = QuantizedLinear::from_parts(
    dense_w, // f32, not uint32
    scales, q_biases, None, 64, 4, "affine",
  )
  .unwrap_err();
  assert!(matches!(err, crate::Error::InvariantViolation(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_rank3_weight() {
  // A rank-3 packed weight is the switch-layer layout, not a plain Linear.
  let dense3 = Array::from_slice::<f32>(
    &vec![0.001f32; 2 * 4 * QUANT_IN],
    &(2usize, 4usize, QUANT_IN),
  )
  .unwrap();
  let (w_q3, scales3, qb3) = quantized::quantize(&dense3, 64, 4, "affine", None).unwrap();
  let err = QuantizedLinear::from_parts(w_q3, scales3, qb3, None, 64, 4, "affine").unwrap_err();
  assert!(matches!(err, crate::Error::RankMismatch(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_affine_without_biases() {
  // affine mode REQUIRES per-group biases.
  let dense_w = dense_weight(8);
  let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err = QuantizedLinear::from_parts(w_q, scales, None, None, 64, 4, "affine").unwrap_err();
  assert!(matches!(err, crate::Error::InvariantViolation(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_unknown_mode() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err = QuantizedLinear::from_parts(w_q, scales, q_biases, None, 64, 4, "garbage").unwrap_err();
  assert!(matches!(err, crate::Error::UnknownEnumValue(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_zero_group_size() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err = QuantizedLinear::from_parts(w_q, scales, q_biases, None, 0, 4, "affine").unwrap_err();
  assert!(matches!(err, crate::Error::OutOfRange(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_higher_rank_bias() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // A rank-2 dense bias is malformed (Linear bias is rank-1 (out,)).
  let bad_bias = Array::from_slice::<f32>(&[1.0; 16], &(8usize, 2usize)).unwrap();
  let err = QuantizedLinear::from_parts(w_q, scales, q_biases, Some(bad_bias), 64, 4, "affine")
    .unwrap_err();
  assert!(matches!(err, crate::Error::RankMismatch(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_length_one_bias() {
  // A rank-1 dense bias of length 1 (not out_features) would broadcast across
  // every output channel in the forward — reject it as a malformed checkpoint.
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let bad_bias = Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap();
  let err = QuantizedLinear::from_parts(w_q, scales, q_biases, Some(bad_bias), 64, 4, "affine")
    .unwrap_err();
  assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
}

#[test]
fn quantized_linear_from_parts_accepts_exact_length_bias() {
  // The boundary case for the new length check: a bias of exactly out_features
  // (8) is accepted (the with-bias forward parity test exercises 8 already; this
  // pins that a correctly-sized bias is NOT rejected by the length gate).
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let bias = Array::from_slice::<f32>(&[1.0; 8], &(8usize,)).unwrap();
  assert!(QuantizedLinear::from_parts(w_q, scales, q_biases, Some(bias), 64, 4, "affine").is_ok());
}

#[test]
fn quantized_linear_from_parts_rejects_mismatched_scales_leading_dim() {
  let dense_w = dense_weight(8);
  let (w_q, _scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Scales with the wrong leading dim (out=4 instead of 8): quantize a
  // different-out dense to get plausible per-group scales of the wrong rows.
  let other_dense = dense_weight(4);
  let (_w2, bad_scales, _qb2) = quantized::quantize(&other_dense, 64, 4, "affine", None).unwrap();
  let err =
    QuantizedLinear::from_parts(w_q, bad_scales, q_biases, None, 64, 4, "affine").unwrap_err();
  assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
}

#[test]
fn quantized_linear_from_parts_rejects_wrong_scales_trailing_dim() {
  // The triple has a CORRECT rank (2) and a CORRECT leading dim (out=8) but a
  // wrong scales TRAILING (per-group) dim: the packed weight recovers
  // `in = packed * 32 / bits` while the scales recover `in = scales.shape(-1) *
  // group_size`, and the two must agree (mlx's `quantized_matmul` invariant).
  // Pairing a group_size-64 packed weight (scales `(8, 1)`) with group_size-32
  // scales (`(8, 2)`) under a declared `group_size = 64` makes the scales
  // recover `2 * 64 = 128` against the weight's `8 * 32 / 4 = 64` — the shared
  // constructor MUST reject this itself, not defer to the deep `quantized_matmul`
  // failure on the first forward.
  let dense_w = dense_weight(8);
  let (w_q, _scales64, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Same `(8, 64)` dense, but group_size 32 → scales `(8, 2)` (a different
  // trailing per-group count) with the matching leading dim (8).
  let (_w32, wrong_scales, _qb32) = quantized::quantize(&dense_w, 32, 4, "affine", None).unwrap();
  assert_eq!(
    wrong_scales.shape(),
    vec![8, 2],
    "fixture: group_size-32 scales"
  );
  let err =
    QuantizedLinear::from_parts(w_q, wrong_scales, q_biases, None, 64, 4, "affine").unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch for a wrong scales trailing dim, got {err:?}"
  );
}

#[test]
fn quantized_linear_from_parts_accepts_affine_integer_scales_floating_biases() {
  // The affine scale/bias dtype rule (mlx's
  // `issubdtype(result_type(scales, biases), floating)`) is deferred to mlx-c at
  // op-time, so construction does NOT validate it: a shape-correct affine triple
  // with INTEGER `scales` and FLOATING `biases` constructs OK here. (mlx ITSELF
  // accepts this triple — the pair promotes to floating — so even at op-time it
  // is valid.) The cast changes only the scales dtype (construction reads
  // metadata, never the values).
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let int_scales = scales.astype(Dtype::I32).unwrap();
  let got = QuantizedLinear::from_parts(w_q, int_scales, q_biases, None, 64, 4, "affine");
  assert!(
    got.is_ok(),
    "expected integer-scales + floating-biases affine triple to construct, got {got:?}"
  );
}

#[test]
fn quantized_linear_from_parts_accepts_affine_floating_scales_integer_biases() {
  // The mirror case: FLOATING `scales` with INTEGER `biases`. Construction does
  // not validate the affine scale/bias dtype (it is deferred to mlx-c at
  // op-time), so this triple constructs OK. (mlx ITSELF accepts it — the pair
  // promotes to floating.) The cast changes only the bias dtype.
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let int_biases = q_biases.unwrap().astype(Dtype::I32).unwrap();
  let got = QuantizedLinear::from_parts(w_q, scales, Some(int_biases), None, 64, 4, "affine");
  assert!(
    got.is_ok(),
    "expected floating-scales + integer-biases affine triple to construct, got {got:?}"
  );
}

#[test]
fn quantized_linear_from_parts_rejects_fp_mode_non_uint8_scales() {
  // The `fp` modes (mxfp4 / mxfp8 / nvfp4) are scale-only AND require
  // `scales.dtype() == uint8` (`validate_mode_with_type`). Reuse a real affine
  // packed weight + scales (gs=64, bits=4) under `mxfp4` with NO biases: the
  // width identity still holds (`in/8 * 32/4 == in/64 * 64`), so the triple is
  // shape-correct, but the affine scales are `f32` — the fp-mode uint8 rule must
  // reject them at construction rather than deferring to the first op.
  let dense_w = dense_weight(8);
  let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  assert!(
    scales.dtype().unwrap() != Dtype::U8,
    "fixture: affine scales are floating, not uint8"
  );
  let err = QuantizedLinear::from_parts(w_q, scales, None, None, 64, 4, "mxfp4").unwrap_err();
  assert!(
    matches!(err, crate::Error::UnsupportedDtype(_)),
    "expected UnsupportedDtype for non-uint8 fp-mode scales, got {err:?}"
  );
}

// ─────────────────── MaybeQuantizedLinear::from_weights routing ───────────────────

#[test]
fn maybe_quantized_picks_dense_when_no_scales() {
  let mut weights: HashMap<String, Array> = HashMap::new();
  let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2usize, 2usize)).unwrap();
  let bias = Array::from_slice::<f32>(&[7.0, 9.0], &(2usize,)).unwrap();
  weights.insert("blk.q.weight".to_string(), weight);
  weights.insert("blk.q.bias".to_string(), bias);

  let layer = MaybeQuantizedLinear::from_weights(&mut weights, "blk.q", None).unwrap();
  assert!(!layer.is_quantized());
  assert!(matches!(layer, MaybeQuantizedLinear::Dense(_)));
  // Consumed both tensors.
  assert!(weights.is_empty());

  let x = Array::from_slice::<f32>(&[3.0, 5.0], &(1usize, 2usize)).unwrap();
  let mut y = layer.forward(&x).unwrap();
  assert_eq!(y.to_vec::<f32>().unwrap(), vec![10.0, 14.0]);
}

#[test]
fn maybe_quantized_picks_quantized_when_scales_present() {
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let q_biases = q_biases.expect("affine biases");

  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.q.weight".to_string(), w_q.try_clone().unwrap());
  weights.insert("blk.q.scales".to_string(), scales.try_clone().unwrap());
  weights.insert("blk.q.biases".to_string(), q_biases.try_clone().unwrap());

  let layer =
    MaybeQuantizedLinear::from_weights(&mut weights, "blk.q", Some((64, 4, "affine"))).unwrap();
  assert!(layer.is_quantized());
  assert!(matches!(layer, MaybeQuantizedLinear::Quantized(_)));
  // All three quantized tensors consumed.
  assert!(weights.is_empty());

  // Forward agrees with the independent dequant-then-matmul reference.
  let x = input(2);
  let mut got = layer.forward(&x).unwrap();
  let reference = dequant_then_matmul(&x, &w_q, &scales, Some(&q_biases), None, 64, 4, "affine");
  assert_within_quant_error(&reference, &got.to_vec::<f32>().unwrap());
}

#[test]
fn maybe_quantized_carries_dense_bias_on_quantized_layer() {
  // A quantized Linear that ALSO ships a dense `.bias` (singular) must route
  // the `.bias` into the QuantizedLinear's dense-bias slot, distinct from the
  // per-group `.biases`.
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let q_biases = q_biases.expect("affine biases");
  let dense_bias = Array::from_slice::<f32>(&[1.0; 8], &(8usize,)).unwrap();

  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.q.weight".to_string(), w_q);
  weights.insert("blk.q.scales".to_string(), scales);
  weights.insert("blk.q.biases".to_string(), q_biases);
  weights.insert("blk.q.bias".to_string(), dense_bias);

  let layer =
    MaybeQuantizedLinear::from_weights(&mut weights, "blk.q", Some((64, 4, "affine"))).unwrap();
  match layer {
    MaybeQuantizedLinear::Quantized(q) => {
      assert!(
        q.bias().is_some(),
        "dense `.bias` must land in the bias slot"
      );
      assert!(q.quant_biases().is_some(), "affine `.biases` present");
    }
    _ => panic!("expected quantized variant"),
  }
  assert!(weights.is_empty());
}

#[test]
fn maybe_quantized_scales_present_but_no_config_errors() {
  // The weights say quantized (`.scales` present) but the config resolved no
  // scheme params — a checkpoint/config inconsistency, surfaced as a typed
  // error rather than a guess.
  let dense_w = dense_weight(8);
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.q.weight".to_string(), w_q);
  weights.insert("blk.q.scales".to_string(), scales);
  weights.insert("blk.q.biases".to_string(), q_biases.unwrap());

  let err = MaybeQuantizedLinear::from_weights(&mut weights, "blk.q", None).unwrap_err();
  assert!(matches!(err, crate::Error::InvariantViolation(_)));
}

#[test]
fn maybe_quantized_dense_missing_weight_errors() {
  let mut weights: HashMap<String, Array> = HashMap::new();
  let err = MaybeQuantizedLinear::from_weights(&mut weights, "blk.q", None).unwrap_err();
  assert!(matches!(err, crate::Error::MissingKey(_)));
}
