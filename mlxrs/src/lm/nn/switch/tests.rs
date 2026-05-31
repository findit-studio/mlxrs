use super::*;

/// Hand-traced golden: one weight matrix per expert (E=2), I=4, O=3.
/// Token 0 (expert 0) selects expert-0 weights; token 1 (expert 1) selects
/// expert-1 weights. Each weight matrix is `[O, I]`, so per-row dot product
/// is `sum(W[o] * x)` for o in 0..O.
///
/// Per-expert weights (in python `[E, O, I]` layout):
/// ```text
/// expert 0: [[1, 0, 0, 0],   expert 1: [[0, 1, 0, 0],
///            [0, 1, 0, 0],              [0, 0, 1, 0],
///            [0, 0, 1, 0]]              [0, 0, 0, 1]]
/// ```
/// Inputs (after `expand_dims(x, (-2, -3))`-style reshape — here we go
/// straight to `[N, 1, I]`):
/// ```text
/// token 0: [1, 2, 3, 4]   → expert 0 → [1, 2, 3]   (project to first 3 features)
/// token 1: [5, 6, 7, 8]   → expert 1 → [6, 7, 8]   (project to last 3 features)
/// ```
fn hand_traced_weight() -> Array {
  Array::from_slice::<f32>(
    &[
      // expert 0: I=4, O=3 → [O, I] = 3x4
      1.0, 0.0, 0.0, 0.0, // row 0
      0.0, 1.0, 0.0, 0.0, // row 1
      0.0, 0.0, 1.0, 0.0, // row 2
      // expert 1: 3x4
      0.0, 1.0, 0.0, 0.0, // row 0
      0.0, 0.0, 1.0, 0.0, // row 1
      0.0, 0.0, 0.0, 1.0, // row 2
    ],
    &(2, 3, 4),
  )
  .unwrap()
}

fn hand_traced_input() -> Array {
  // [N=2, 1, I=4]
  Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(2, 1, 4)).unwrap()
}

#[test]
fn switch_linear_shape_no_bias() {
  let weight = hand_traced_weight();
  let layer = SwitchLinear::from_parts(weight, None).unwrap();
  assert_eq!(layer.num_experts(), 2);
  assert_eq!(layer.output_dims(), 3);
  assert_eq!(layer.input_dims(), 4);

  let x = hand_traced_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2usize,)).unwrap();
  let out = layer.apply(&x, &indices, false).unwrap();
  assert_eq!(out.shape(), vec![2, 1, 3]);
  assert_eq!(out.dtype().unwrap(), Dtype::F32);
}

#[test]
fn switch_linear_hand_traced_no_bias() {
  let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
  let x = hand_traced_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
  let mut out = layer.apply(&x, &indices, false).unwrap();
  let got = out.to_vec::<f32>().unwrap();
  // Token 0 via expert 0: [1, 2, 3] (projects to features 0..3 of [1,2,3,4]).
  // Token 1 via expert 1: [6, 7, 8] (projects to features 1..4 of [5,6,7,8]).
  assert_eq!(got, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
}

#[test]
fn switch_linear_hand_traced_with_bias() {
  // bias[E=2, O=3]; expert-0 adds [10, 20, 30], expert-1 adds [40, 50, 60].
  let bias = Array::from_slice::<f32>(
    &[
      10.0, 20.0, 30.0, // expert 0
      40.0, 50.0, 60.0, // expert 1
    ],
    &(2, 3),
  )
  .unwrap();
  let layer = SwitchLinear::from_parts(hand_traced_weight(), Some(bias)).unwrap();
  let x = hand_traced_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
  let mut out = layer.apply(&x, &indices, false).unwrap();
  let got = out.to_vec::<f32>().unwrap();
  // Token 0: [1+10, 2+20, 3+30] = [11, 22, 33].
  // Token 1: [6+40, 7+50, 8+60] = [46, 57, 68].
  assert_eq!(got, vec![11.0, 22.0, 33.0, 46.0, 57.0, 68.0]);
}

#[test]
fn switch_linear_all_routed_to_one_expert_matches_plain_matmul() {
  // Edge: every token routed to expert 0 → output is equivalent to a plain
  // batched matmul `x @ weight[0]ᵀ`.
  let weight = hand_traced_weight();
  let layer = SwitchLinear::from_parts(weight, None).unwrap();
  let x = hand_traced_input();
  let indices = Array::from_slice::<u32>(&[0, 0], &(2,)).unwrap();
  let mut out = layer.apply(&x, &indices, false).unwrap();
  let got = out.to_vec::<f32>().unwrap();
  // Both tokens via expert 0: [1, 2, 3] (token 0) and [5, 6, 7] (token 1).
  assert_eq!(got, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
}

#[test]
fn switch_linear_sorted_indices_matches_unsorted() {
  // `sorted_indices=true` is a performance hint — the result must match the
  // `false` path bit-for-bit when the indices truly are sorted.
  let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
  let x = hand_traced_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap(); // already sorted
  let mut via_sorted = layer.apply(&x, &indices, true).unwrap();
  let mut via_unsorted = layer.apply(&x, &indices, false).unwrap();
  assert_eq!(
    via_sorted.to_vec::<f32>().unwrap(),
    via_unsorted.to_vec::<f32>().unwrap()
  );
}

#[test]
fn switch_linear_from_parts_rejects_2d_weight() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let err = SwitchLinear::from_parts(bad, None).unwrap_err();
  assert!(matches!(err, crate::Error::RankMismatch(_)));
}

#[test]
fn switch_linear_from_parts_rejects_mismatched_bias() {
  let weight = hand_traced_weight(); // [2, 3, 4]
  // Bad bias: [3, 3] (wrong E).
  let bad_bias =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &(3, 3)).unwrap();
  let err = SwitchLinear::from_parts(weight, Some(bad_bias)).unwrap_err();
  assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
}

/// Bias-rank-mismatch split: a rank-1 (or rank-3)
/// bias must surface as `RankMismatch`, not as `ShapePairMismatch`, so
/// typed-error consumers can distinguish the rank-vs-shape categories.
/// Pre-split, every malformed-rank bias was collapsed into
/// `ShapePairMismatch` by the single combined check.
#[test]
fn switch_linear_from_parts_rejects_rank_mismatch_bias() {
  let weight = hand_traced_weight(); // [2, 3, 4]
  // Rank-1 bias `[2]` (a plausible per-expert flat scalar) — must now
  // be `RankMismatch` with `actual == 1`.
  let bad_bias_rank1 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
  let err =
    SwitchLinear::from_parts(weight.try_clone().unwrap(), Some(bad_bias_rank1)).unwrap_err();
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 1, "rank-1 bias ⇒ actual rank 1");
      assert_eq!(payload.actual_shape(), &[2usize]);
    }
    other => panic!("expected RankMismatch on rank-1 bias, got {other:?}"),
  }
  // Rank-3 bias `[2, 3, 1]` — must also be `RankMismatch`.
  let bad_bias_rank3 =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3usize, 1usize)).unwrap();
  let err = SwitchLinear::from_parts(weight, Some(bad_bias_rank3)).unwrap_err();
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 3, "rank-3 bias ⇒ actual rank 3");
      assert_eq!(payload.actual_shape(), &[2usize, 3, 1]);
    }
    other => panic!("expected RankMismatch on rank-3 bias, got {other:?}"),
  }
}

#[test]
fn switch_linear_top_k_routing_shape() {
  // top-k=2 routing: indices is [N, k]; output is [N, k, O].
  let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
  // x must broadcast against the [N, k] indices on the leading batch dims —
  // mlx-lm's SwitchMLP feeds x as [..., 1, 1, I] (expand_dims (-2, -3)) so
  // the k slot broadcasts. Here we go straight to [N=2, k=2, 1, I=4] which
  // is the shape after the expand: token 0 will go through experts (0, 1),
  // token 1 through (1, 0).
  let x = Array::from_slice::<f32>(
    &[
      1.0, 2.0, 3.0, 4.0, // token 0 expert slot 0
      1.0, 2.0, 3.0, 4.0, // token 0 expert slot 1
      5.0, 6.0, 7.0, 8.0, // token 1 expert slot 0
      5.0, 6.0, 7.0, 8.0, // token 1 expert slot 1
    ],
    &(2, 2, 1, 4),
  )
  .unwrap();
  let indices = Array::from_slice::<u32>(&[0, 1, 1, 0], &(2, 2)).unwrap();
  let mut out = layer.apply(&x, &indices, false).unwrap();
  assert_eq!(out.shape(), vec![2, 2, 1, 3]);
  let got = out.to_vec::<f32>().unwrap();
  // token 0 slot 0 (expert 0): [1, 2, 3]
  // token 0 slot 1 (expert 1): [2, 3, 4]
  // token 1 slot 0 (expert 1): [6, 7, 8]
  // token 1 slot 1 (expert 0): [5, 6, 7]
  assert_eq!(
    got,
    vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0]
  );
}

// -------- QuantizedSwitchLinear --------

/// A larger weight stack so the quantizer's `group_size=64` actually has at
/// least one full group along the last axis (`I=64` here).
const QUANT_INPUT_DIMS: usize = 64;

fn quant_dense_weight() -> Array {
  let e: usize = 2;
  let o: usize = 4;
  let i = QUANT_INPUT_DIMS;
  let mut data = Vec::with_capacity(e * o * i);
  // Smooth ramp — friendly to 4-bit affine quant (per `ops_quantized.rs`).
  for ei in 0..e {
    for oi in 0..o {
      for ii in 0..i {
        data.push(((ei * 100 + oi * 10 + ii) as f32) * 0.001);
      }
    }
  }
  Array::from_slice::<f32>(&data, &(e, o, i)).unwrap()
}

fn quant_input() -> Array {
  let n: usize = 2;
  let i = QUANT_INPUT_DIMS;
  let mut data = Vec::with_capacity(n * i);
  for ni in 0..n {
    for ii in 0..i {
      data.push(((ni * 50 + ii) as f32) * 0.01);
    }
  }
  Array::from_slice::<f32>(&data, &(n, 1usize, i)).unwrap()
}

#[test]
fn quantized_switch_linear_parity_within_quant_error() {
  let dense_w = quant_dense_weight();
  let dense_layer = SwitchLinear::from_parts(dense_w.try_clone().unwrap(), None).unwrap();

  // Quantize the dense weight using the affine scheme (default).
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  assert!(
    q_biases.is_some(),
    "affine scheme produces per-group biases"
  );
  let q_layer =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 64, 4, "affine").unwrap();

  let x = quant_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
  let mut dense_out = dense_layer.apply(&x, &indices, false).unwrap();
  let mut quant_out = q_layer.apply(&x, &indices, false).unwrap();
  assert_eq!(dense_out.shape(), quant_out.shape());

  let dense = dense_out.to_vec::<f32>().unwrap();
  let quant = quant_out.to_vec::<f32>().unwrap();
  // 4-bit affine quant on a smooth ramp: per-element drift must stay within
  // a generous band relative to the dense magnitude (matches the tolerance
  // used in `tests/ops_quantized.rs::quantize_then_dequantize_round_trips_*`).
  let max_abs = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
  for (d, q) in dense.iter().zip(quant.iter()) {
    assert!(
      (d - q).abs() <= 0.1 * max_abs + 1e-3,
      "quantized SwitchLinear drift too large: dense={d} quant={q}"
    );
  }
}

#[test]
fn quantized_switch_linear_from_parts_rejects_mismatched_bias() {
  // Quantize a `[E=2, O=4, I=64]` dense stack so the packed `weight` has
  // `shape[0]=E=2` and `shape[1]=O=4`. A `[E, 1]` bias would silently
  // broadcast across every output channel in `apply` (`take_axis` →
  // `expand_dims(-2)` → `add`) without this rejection.
  let dense_w = quant_dense_weight();
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Bad bias: rank-2 but trailing dim is 1, not O=4.
  let bad_bias = Array::from_slice::<f32>(&[1.0, 2.0], &(2, 1)).unwrap();
  let err =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bad_bias), 64, 4, "affine")
      .unwrap_err();
  assert!(matches!(err, crate::Error::ShapePairMismatch(_)));
}

/// Bias-rank-mismatch split: a rank-1 bias on the
/// QUANTIZED layer must surface as `RankMismatch`, not as
/// `ShapePairMismatch` — same taxonomy as the dense [`SwitchLinear`]
/// sibling.
#[test]
fn quantized_switch_linear_from_parts_rejects_rank_mismatch_bias() {
  let dense_w = quant_dense_weight();
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Rank-1 bias `[2]` — must now be `RankMismatch` with `actual == 1`.
  let bad_bias_rank1 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
  let err =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bad_bias_rank1), 64, 4, "affine")
      .unwrap_err();
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 1, "rank-1 bias ⇒ actual rank 1");
      assert_eq!(payload.actual_shape(), &[2usize]);
    }
    other => panic!("expected RankMismatch on rank-1 bias, got {other:?}"),
  }
}

/// `quant_biases` rank must match `scales` rank — split out from the
/// shape-pair check: a divergent rank now surfaces
/// as `RankMismatch`, not `ShapePairMismatch`. Pre-split, `qb_shape !=
/// s_shape` collapsed both rank and shape divergences into the same
/// variant.
#[test]
fn quantized_switch_linear_from_parts_rejects_quant_biases_rank_mismatch() {
  // Valid affine triple — `scales` is rank-3 `[E, O, n_groups]`. Supply a
  // rank-2 `quant_biases` and observe `RankMismatch` with `actual == 2`.
  let dense_w = quant_dense_weight(); // [2, 4, 64]
  let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Bad rank-2 quant_biases `[2, 4]` — wrong rank entirely.
  let bad_qb =
    Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &(2usize, 4usize)).unwrap();
  let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(bad_qb), None, 64, 4, "affine")
    .unwrap_err();
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 2, "rank-2 quant_biases ⇒ actual rank 2");
      assert_eq!(payload.actual_shape(), &[2usize, 4]);
    }
    other => panic!("expected RankMismatch on rank-2 quant_biases, got {other:?}"),
  }
}

#[test]
fn quantized_switch_linear_with_bias_parity_within_quant_error() {
  // Valid `[E=2, O=4]` bias on both the dense and quantized layers; the
  // quantized output (with bias) must stay within the same quant-error band
  // as the bias-less parity test above.
  let dense_w = quant_dense_weight();
  // Distinct per-expert per-channel bias so any wrong-broadcast would visibly
  // diverge from the dense reference.
  let bias = Array::from_slice::<f32>(
    &[
      10.0, 20.0, 30.0, 40.0, // expert 0
      50.0, 60.0, 70.0, 80.0, // expert 1
    ],
    &(2, 4),
  )
  .unwrap();
  let dense_layer = SwitchLinear::from_parts(
    dense_w.try_clone().unwrap(),
    Some(bias.try_clone().unwrap()),
  )
  .unwrap();

  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let q_layer =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bias), 64, 4, "affine").unwrap();

  let x = quant_input();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2,)).unwrap();
  let mut dense_out = dense_layer.apply(&x, &indices, false).unwrap();
  let mut quant_out = q_layer.apply(&x, &indices, false).unwrap();
  assert_eq!(dense_out.shape(), quant_out.shape());

  let dense = dense_out.to_vec::<f32>().unwrap();
  let quant = quant_out.to_vec::<f32>().unwrap();
  let max_abs = dense.iter().fold(0.0f32, |m, v| m.max(v.abs()));
  for (d, q) in dense.iter().zip(quant.iter()) {
    assert!(
      (d - q).abs() <= 0.1 * max_abs + 1e-3,
      "quantized SwitchLinear (with bias) drift too large: dense={d} quant={q}"
    );
  }
}

// ─── QuantizedSwitchLinear::from_parts structural-invariant tests ───
//
// Mirrors the `classify_triple` `match (q.mode, b_opt)` mode-arity pattern
// (`mlxrs/src/lm/quant.rs:613-640`): validates STRUCTURAL invariants on the
// packed `(weight, scales, quant_biases)` triple. Per-mode value tables
// (`bits ∈ {2,3,4,5,6,8}` for affine; `mxfp4` / `nvfp4` require specific
// `(group_size, bits)` pairs — `mlx/ops.cpp:4745-4750,4808-4823`) are
// DEFERRED to mlx-c per `feedback_match_official_binding_design` —
// duplicating them in mlxrs would drift from upstream.
//
// Quantization fixtures: a smooth `(2, 4, 64)` ramp under `affine /
// group_size=64 / bits=4` (matches existing parity tests); a `(2, 4, 64)`
// ramp under `mxfp4 / group_size=32 / bits=4` (the only `(gs, b)` mlx-c
// accepts for `mxfp4`, `mlx/ops.cpp:4808-4823`).

/// `mxfp4` fixture: the only `(group_size, bits)` pair mlx-c accepts for
/// `mxfp4` is `(32, 4)` (`quantization_params_from_mode` in
/// `mlx/ops.cpp:4808-4823`). Reuses the same dense `[E=2, O=4, I=64]`
/// stack as `quant_dense_weight`; the resulting packed `scales` is
/// `[2, 4, 64/32 = 2]`, and `quant_biases == None` (bias-less fp scheme).
fn quant_mxfp4_triple() -> (Array, Array, Option<Array>) {
  let dense_w = quant_dense_weight();
  quantized::quantize(&dense_w, 32, 4, "mxfp4", None).unwrap()
}

#[test]
fn quantized_switch_linear_from_parts_rejects_mismatched_scales_leading_dims() {
  // `weight [E=2, O=4, I_packed]` paired with `scales [E=3, O=4, ..]` —
  // the leading `E` mismatches. mlx `quantize` always preserves the
  // leading shape across (weight, scales, biases) (`mlx/ops.cpp:4789-4798`),
  // so this combination is structurally impossible from a real
  // `quantize` call.
  let dense_w = quant_dense_weight(); // [2, 4, 64]
  let (w_q, _scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  // Build a mismatched scales with E=3 (instead of E=2). Use a fresh
  // `[3, 4, 1]` rank-3 array; the trailing-axis value is irrelevant
  // because the leading-dim check fires first.
  let bad_scales = Array::from_slice::<f32>(
    &[
      1.0, 1.0, 1.0, 1.0, // E=0
      1.0, 1.0, 1.0, 1.0, // E=1
      1.0, 1.0, 1.0, 1.0, // E=2 (extra)
    ],
    &(3usize, 4usize, 1usize),
  )
  .unwrap();
  let err =
    QuantizedSwitchLinear::from_parts(w_q, bad_scales, None, None, 64, 4, "mxfp4").unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch on scales leading dims, got {err:?}"
  );
}

#[test]
fn quantized_switch_linear_from_parts_rejects_non_u32_weight() {
  // A rank-3 dense `f32` weight `[2, 4, 8]` with otherwise-matching affine
  // `scales` / `quant_biases` passes every shape / mode-arity check but is
  // NOT a packed quantized weight — mlx packs `affine_quantize`'s `w_q`
  // into `uint32` words and `gather_qmm` rejects any non-`uint32` quantized
  // weight. Without the dtype guard `from_parts` returns `Ok` and the
  // failure surfaces deep inside the FFI on the first `apply`; the guard
  // moves it to construction. Mirrors `classify_triple`'s `.weight` ==
  // `U32` requirement for quantized triples.
  let dense_data = vec![0.5f32; 2 * 4 * 8];
  let dense_weight = Array::from_slice::<f32>(&dense_data, &(2usize, 4usize, 8usize)).unwrap();
  // `scales` / `quant_biases` shaped to match the leading `[E=2, O=4, ..]`
  // dims so the dtype check — placed right after the weight-rank check —
  // is what fires, not a downstream shape mismatch.
  let scales_data = vec![1.0f32; 2 * 4];
  let scales = Array::from_slice::<f32>(&scales_data, &(2usize, 4usize, 1usize)).unwrap();
  let qb_data = vec![0.0f32; 2 * 4];
  let quant_biases = Array::from_slice::<f32>(&qb_data, &(2usize, 4usize, 1usize)).unwrap();
  let err = QuantizedSwitchLinear::from_parts(
    dense_weight,
    scales,
    Some(quant_biases),
    None,
    64,
    4,
    "affine",
  )
  .unwrap_err();
  match &err {
    crate::Error::InvariantViolation(payload) => {
      assert!(
        payload.context().contains("weight dtype") || payload.requirement().contains("uint32"),
        "InvariantViolation context/requirement should name the dtype invariant, got context={:?} requirement={:?}",
        payload.context(),
        payload.requirement()
      );
    }
    other => panic!("expected InvariantViolation naming the dtype invariant, got {other:?}"),
  }
}

#[test]
fn quantized_switch_linear_from_parts_rejects_quant_biases_shape_mismatch() {
  // Valid affine triple but `quant_biases` has a shape distinct from
  // `scales` — `affine_quantize` writes them with identical
  // `[E, O, n_groups]` shape (`mlx/ops.cpp:4793-4798`), so a divergent
  // shape is structurally invalid. Use `group_size=32` here so `scales`
  // resolves to `[E=2, O=4, n_groups=2]` and the `[2, 4, 1]` bad
  // `quant_biases` truly mismatches (with the default `group_size=64`,
  // `scales` is `[2, 4, 1]` and a `[2, 4, 1]` bad would coincidentally
  // match, masking the check).
  let dense_w = quant_dense_weight(); // [2, 4, 64]
  let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 32, 4, "affine", None).unwrap();
  // scales is `[2, 4, 64/32 = 2]`; bad quant_biases is `[2, 4, 1]` —
  // trailing dim mismatches.
  let bad_qb = Array::from_slice::<f32>(
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    &(2usize, 4usize, 1usize),
  )
  .unwrap();
  let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(bad_qb), None, 32, 4, "affine")
    .unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch on quant_biases shape, got {err:?}"
  );
}

#[test]
fn quantized_switch_linear_from_parts_affine_requires_quant_biases() {
  // `affine` mode is the 3-output `affine_quantize` arity
  // (`mlx/ops.cpp:4793-4798`); a `None` `quant_biases` next to it is a
  // structurally incomplete triple, rejected at construction.
  let dense_w = quant_dense_weight();
  let (w_q, scales, _q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err =
    QuantizedSwitchLinear::from_parts(w_q, scales, None, None, 64, 4, "affine").unwrap_err();
  assert!(
    matches!(err, crate::Error::InvariantViolation(_)),
    "expected InvariantViolation on affine-missing-quant_biases, got {err:?}"
  );
}

#[test]
fn quantized_switch_linear_from_parts_mxfp4_forbids_quant_biases() {
  // `mxfp4` is scale-only (`fp_quantize` 2-output arity,
  // `mlx/ops.cpp:4890,4898-4904`); a stale `quant_biases` next to it
  // would be retained from an unrelated `affine` triple and is rejected
  // at construction.
  let (w_q, scales, _none_qb) = quant_mxfp4_triple();
  // Fabricate a stale `quant_biases` shaped to match `scales` so the
  // mode-arity check fires before the shape-match check.
  let s_shape = scales.shape();
  let n_groups = s_shape[2];
  let stale_qb_data = vec![0.0f32; 2 * 4 * n_groups];
  let stale_qb = Array::from_slice::<f32>(&stale_qb_data, &(2usize, 4usize, n_groups)).unwrap();
  let err = QuantizedSwitchLinear::from_parts(w_q, scales, Some(stale_qb), None, 32, 4, "mxfp4")
    .unwrap_err();
  assert!(
    matches!(err, crate::Error::InvariantViolation(_)),
    "expected InvariantViolation on mxfp4-with-stale-quant_biases, got {err:?}"
  );
}

#[test]
fn quantized_switch_linear_from_parts_unknown_mode() {
  // Unknown mode tag — neither `affine` nor any of the fp schemes — is
  // rejected so a typo doesn't reach mlx-c with an unfamiliar mode
  // string (where it would surface as a less-specific backend error).
  let dense_w = quant_dense_weight();
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let err =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 64, 4, "unknown").unwrap_err();
  assert!(
    matches!(err, crate::Error::UnknownEnumValue(_)),
    "expected UnknownEnumValue on unknown mode, got {err:?}"
  );
}

#[test]
fn quantized_switch_linear_from_parts_zero_bits_or_group_size() {
  // Basic non-zero sanity on `bits` / `group_size` (per-mode value tables
  // remain deferred to mlx-c — we just catch the trivial 0 here so the
  // FFI doesn't divide-by-zero downstream).
  let dense_w = quant_dense_weight();
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();

  let err_bits = QuantizedSwitchLinear::from_parts(
    w_q.try_clone().unwrap(),
    scales.try_clone().unwrap(),
    q_biases.as_ref().map(|q| q.try_clone().unwrap()),
    None,
    64,
    0,
    "affine",
  )
  .unwrap_err();
  assert!(
    matches!(err_bits, crate::Error::OutOfRange(_)),
    "expected OutOfRange on bits=0, got {err_bits:?}"
  );

  let err_gs =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, None, 0, 4, "affine").unwrap_err();
  assert!(
    matches!(err_gs, crate::Error::OutOfRange(_)),
    "expected OutOfRange on group_size=0, got {err_gs:?}"
  );
}

/// Regression: a valid `mxfp4` triple (scales-only, `quant_biases ==
/// None`) constructs cleanly. Closes the new structural-invariant block
/// over the bias-less fp branch; the existing affine parity tests already
/// cover the `(affine, Some)` branch.
#[test]
fn quantized_switch_linear_from_parts_mxfp4_scales_only_ok() {
  let (w_q, scales, none_qb) = quant_mxfp4_triple();
  assert!(none_qb.is_none(), "mxfp4 quantize must yield None biases");
  let layer = QuantizedSwitchLinear::from_parts(w_q, scales, None, None, 32, 4, "mxfp4").unwrap();
  assert_eq!(layer.weight_ref().shape()[0], 2); // E=2
  assert_eq!(layer.weight_ref().shape()[1], 4); // O=4
  assert!(layer.quant_biases().is_none());
  assert_eq!(layer.mode(), "mxfp4");
}

// ─── SwitchLinear / QuantizedSwitchLinear field-visibility regressions ───

/// `SwitchLinear`'s `weight` / `bias` are PRIVATE fields with read-only
/// public accessors. This test exercises the accessors and — by virtue of
/// compiling without reaching for the fields — confirms the read path
/// goes through them. Direct field access from outside `super::` would
/// fail to compile (the fields' visibility is module-private). External
/// code previously could write `layer.bias = Some(bad_bias)` (any shape)
/// and then `layer.apply(_)` would silently broadcast a malformed
/// `[E, 1]` bias across every output channel; with the fields private,
/// that mutation path is statically impossible — `from_parts` is the
/// only construction path, and its `[E, O]` check is the only path that
/// matters.
#[test]
fn switch_linear_fields_are_read_only_via_accessors() {
  let layer = SwitchLinear::from_parts(hand_traced_weight(), None).unwrap();
  assert_eq!(layer.weight_ref().shape(), vec![2, 3, 4]);
  assert!(layer.bias().is_none());

  let bias = Array::from_slice::<f32>(
    &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], // [E=2, O=3]
    &(2, 3),
  )
  .unwrap();
  let layer_with_bias = SwitchLinear::from_parts(hand_traced_weight(), Some(bias)).unwrap();
  assert_eq!(layer_with_bias.weight_ref().shape(), vec![2, 3, 4]);
  assert_eq!(layer_with_bias.bias().unwrap().shape(), vec![2, 3]);
  // (compile-fail) external `layer.weight = ...` and `layer.bias = ...`
  // are both private-field errors; trying them here from inside `super::`
  // would compile (same module), so we don't try — the visibility
  // guarantee is what the regression turns on, not a runtime check.
}

/// `QuantizedSwitchLinear`'s `weight` / `scales` / `quant_biases` /
/// `bias` / `group_size` / `bits` / `mode` are PRIVATE fields with
/// read-only public accessors. Same rationale as
/// [`switch_linear_fields_are_read_only_via_accessors`]: external
/// struct-literal construction `QuantizedSwitchLinear { bias:
/// Some(bad_bias), .. }` or `&mut` mutation would otherwise bypass
/// `from_parts`'s `[E, O]` bias-shape check, and post-construction
/// `bits = -1` / `group_size = 0` / `mode = "garbage"` would mis-decode
/// the packed weight inside the FFI.
#[test]
fn quantized_switch_linear_fields_are_read_only_via_accessors() {
  let dense_w = quant_dense_weight();
  let (w_q, scales, q_biases) = quantized::quantize(&dense_w, 64, 4, "affine", None).unwrap();
  let bias = Array::from_slice::<f32>(
    &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], // [E=2, O=4]
    &(2, 4),
  )
  .unwrap();
  let q_layer =
    QuantizedSwitchLinear::from_parts(w_q, scales, q_biases, Some(bias), 64, 4, "affine").unwrap();
  // All accessors return the constructor-validated values; the field
  // privacy is what guarantees no other write path exists.
  assert_eq!(q_layer.weight_ref().shape()[0], 2); // E=2
  assert_eq!(q_layer.weight_ref().shape()[1], 4); // O=4
  assert_eq!(q_layer.scales_ref().shape()[0], 2); // E
  assert!(q_layer.quant_biases().is_some()); // affine ⇒ Some
  assert_eq!(q_layer.bias().unwrap().shape(), vec![2, 4]);
  assert_eq!(q_layer.group_size(), 64);
  assert_eq!(q_layer.bits(), 4);
  assert_eq!(q_layer.mode(), "affine");
  // (compile-fail) external `q_layer.bits = -1`, `q_layer.mode =
  // "garbage".into()`, etc. are all private-field errors; trying them
  // here from inside `super::` would compile (same module), so we don't
  // try — the visibility guarantee is what the regression turns on, not
  // a runtime check.
}

// ─── SwitchGLU / SwitchMLP block tests ───
//
// Hand-traced over a tiny known expert set (E=2, I=H=2). The projections
// are built from explicit per-expert weight stacks so the forward math is
// exactly reproducible by hand; `silu`/identity activations keep the
// reference value closed-form.

/// Logistic sigmoid — the reference scalar formula.
fn sigmoid_ref(v: f32) -> f32 {
  1.0 / (1.0 + (-v).exp())
}

/// `silu(v) = v · σ(v)` — the reference scalar formula.
fn silu_ref(v: f32) -> f32 {
  v * sigmoid_ref(v)
}

/// Per-element near-equality (f32 op-graph vs f64-ish reference).
fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(
    got.len(),
    want.len(),
    "length mismatch: {got:?} vs {want:?}"
  );
  for (g, w) in got.iter().zip(want.iter()) {
    assert!(
      (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
      "block output mismatch: got {g}, want {w} (full got {got:?}, want {want:?})"
    );
  }
}

/// A `[E=2, O=2, I=2]` weight stack: expert 0 is the 2×2 identity, expert 1
/// is the 2×2 swap `[[0,1],[1,0]]`. Routing token 0 → expert 0 leaves its
/// features in place; routing → expert 1 swaps them — so a forward result
/// reveals which expert each token was routed through.
fn identity_then_swap_weight() -> Array {
  Array::from_slice::<f32>(
    &[
      // expert 0: identity
      1.0, 0.0, //
      0.0, 1.0, //
      // expert 1: swap
      0.0, 1.0, //
      1.0, 0.0, //
    ],
    &(2, 2, 2),
  )
  .unwrap()
}

/// A `[E=2, O=2, I=2]` all-identity weight stack — both experts are the 2×2
/// identity, so the projection is a no-op `y = x` regardless of routing.
fn all_identity_weight() -> Array {
  Array::from_slice::<f32>(
    &[
      1.0, 0.0, 0.0, 1.0, // expert 0: identity
      1.0, 0.0, 0.0, 1.0, // expert 1: identity
    ],
    &(2, 2, 2),
  )
  .unwrap()
}

#[test]
fn switch_glu_hand_traced_two_experts() {
  // gate_proj routes through identity (expert 0) / swap (expert 1);
  // up_proj and down_proj are pure identity. With the `silu` activation the
  // block computes `down(silu(gate(x)) · up(x)) = silu(gate_e(x)) · x`.
  let gate_proj = SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap();
  let up_proj = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
  let down_proj = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
  let glu = SwitchGLU::new(
    gate_proj,
    up_proj,
    down_proj,
    SwitchGLU::default_activation(), // silu
  )
  .unwrap();

  // Two tokens [1, 2] and [3, 4]; token 0 → expert 0, token 1 → expert 1.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
  let mut out = glu.forward(&x, &indices).unwrap();
  // forward(x) returns [N=2, k=1, I=2].
  assert_eq!(out.shape(), vec![2, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Token 0 via expert 0 (identity gate): silu([1,2]) · [1,2]
  //   = [silu(1)·1, silu(2)·2].
  // Token 1 via expert 1 (swap gate): silu(swap([3,4])) · [3,4]
  //   = silu([4,3]) · [3,4] = [silu(4)·3, silu(3)·4].
  let want = vec![
    silu_ref(1.0) * 1.0,
    silu_ref(2.0) * 2.0,
    silu_ref(4.0) * 3.0,
    silu_ref(3.0) * 4.0,
  ];
  assert_close(&got, &want);
}

#[test]
fn switch_glu_routing_selects_the_indexed_expert() {
  // Same block, but route BOTH tokens through expert 1 (swap). Every token
  // must show the swapped-gate math — proving `indices` actually selects
  // the expert rather than e.g. always using expert 0.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 5.0, 6.0], &(2, 2)).unwrap();
  let indices = Array::from_slice::<u32>(&[1, 1], &(2, 1)).unwrap();
  let mut out = glu.forward(&x, &indices).unwrap();
  let got = out.to_vec::<f32>().unwrap();
  // Both via expert 1 (swap gate): silu(swap(x)) · x.
  //   token 0: silu([2,1]) · [1,2] = [silu(2)·1, silu(1)·2].
  //   token 1: silu([6,5]) · [5,6] = [silu(6)·5, silu(5)·6].
  let want = vec![
    silu_ref(2.0) * 1.0,
    silu_ref(1.0) * 2.0,
    silu_ref(6.0) * 5.0,
    silu_ref(5.0) * 6.0,
  ];
  assert_close(&got, &want);
}

#[test]
fn switch_glu_sorted_path_matches_hand_trace() {
  // `do_sort` triggers at `indices.size() >= 64`. Route 64 tokens through
  // alternating experts (0, 1, 0, 1, …): the block sorts them by expert id
  // internally and must `scatter_unsort` the result back so each token's
  // output lands at its original position. A wrong unsort would scramble
  // the per-token values and fail the assertion below.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let n = 64usize;
  // Token t has features [t, t + 1]; expert id alternates 0 / 1.
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let mut out = glu.forward(&x, &indices).unwrap();
  assert_eq!(out.shape(), vec![n, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Reference: per token, silu(gate_e(x)) · x — expert 0 keeps features,
  // expert 1 swaps them.
  let mut want = Vec::with_capacity(n * 2);
  for t in 0..n {
    let (x0, x1) = (t as f32, t as f32 + 1.0);
    if t % 2 == 0 {
      // expert 0 (identity gate)
      want.push(silu_ref(x0) * x0);
      want.push(silu_ref(x1) * x1);
    } else {
      // expert 1 (swap gate): gate sees [x1, x0]
      want.push(silu_ref(x1) * x0);
      want.push(silu_ref(x0) * x1);
    }
  }
  assert_close(&got, &want);
}

#[test]
fn switch_glu_new_rejects_mismatched_projection_shapes() {
  // down_proj must be the [hidden→input] inverse of gate/up [input→hidden].
  // Here gate/up are [2→2] but down is [2→3] (wrong output_dims) — rejected.
  let bad_down_weight =
    Array::from_slice::<f32>(&[0.0f32; 2 * 3 * 2], &(2usize, 3usize, 2usize)).unwrap();
  let err = SwitchGLU::new(
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(bad_down_weight, None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch on mismatched down_proj, got {err:?}"
  );
}

#[test]
fn switch_glu_new_rejects_mismatched_num_experts() {
  // gate/up have E=2; a down_proj with E=3 is rejected.
  let down_e3 = Array::from_slice::<f32>(&[1.0f32; 3 * 2 * 2], &(3usize, 2usize, 2usize)).unwrap();
  let err = SwitchGLU::new(
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(down_e3, None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch on mismatched num_experts, got {err:?}"
  );
}

#[test]
fn switch_mlp_hand_traced_two_experts() {
  // fc1 routes through identity (expert 0) / swap (expert 1); fc2 is
  // identity. With a `square` activation the block computes
  // `fc2(square(fc1(x))) = (fc1_e(x))²`.
  let fc1 = SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap();
  let fc2 = SwitchLinear::from_parts(all_identity_weight(), None).unwrap();
  // Explicit closed-form activation so the trace is exact integer arithmetic
  // (the block's wiring is what's under test here; the reference activation
  // formulas are covered in `activations::tests`).
  let square: Activation = Box::new(|a: &Array| a.multiply(a));
  let mlp = SwitchMLP::new(fc1, fc2, square).unwrap();

  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
  let mut out = mlp.forward(&x, &indices).unwrap();
  assert_eq!(out.shape(), vec![2, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Token 0 via expert 0 (identity): square([1,2]) = [1, 4].
  // Token 1 via expert 1 (swap): square(swap([3,4])) = square([4,3]) = [16, 9].
  assert_eq!(got, vec![1.0, 4.0, 16.0, 9.0]);
}

#[test]
fn switch_mlp_default_activation_is_gelu_approx() {
  // `SwitchMLP::default_activation()` must be `gelu_approx` (the python
  // `nn.GELU(approx="precise")` default). With identity fc1/fc2 the block
  // collapses to the activation itself, so the output must equal
  // `activations::gelu_approx(x)`.
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchMLP::default_activation(),
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&[-1.0, 0.5, 1.0, 2.0], &(2, 2)).unwrap();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
  let mut out = mlp.forward(&x, &indices).unwrap();
  let got = out.to_vec::<f32>().unwrap();
  // Reference: gelu_approx applied element-wise (fc1/fc2 are identity).
  let mut reference = super::super::activations::gelu_approx(&x).unwrap();
  let want = reference.to_vec::<f32>().unwrap();
  assert_close(&got, &want);
}

#[test]
fn switch_mlp_forward_preserves_f16_dtype() {
  // `SwitchMLP::default_activation()` is `gelu_approx`, whose scalar
  // constants are dtype-matched (see `activations::scalar_like`). With F16
  // weights and an F16 input the whole block stays F16 — a stray F32
  // activation constant would promote the output to F32. Weights are cast
  // from f32 so no `half`-crate scalars are needed.
  let w16 = all_identity_weight().astype(Dtype::F16).unwrap();
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(w16.try_clone().unwrap(), None).unwrap(),
    SwitchLinear::from_parts(w16, None).unwrap(),
    SwitchMLP::default_activation(),
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&[-1.0, 0.5, 1.0, 2.0], &(2, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let indices = Array::from_slice::<u32>(&[0, 1], &(2, 1)).unwrap();
  let out = mlp.forward(&x, &indices).unwrap();
  assert_eq!(
    out.dtype().unwrap(),
    Dtype::F16,
    "SwitchMLP default forward must preserve the F16 input dtype"
  );
}

#[test]
fn switch_mlp_sorted_path_matches_hand_trace() {
  // Same `indices.size() >= 64` sorted-path exercise as the SwitchGLU
  // sibling test, for the un-gated `fc2(square(fc1(x)))` body.
  let square: Activation = Box::new(|a: &Array| a.multiply(a));
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    square,
  )
  .unwrap();
  let n = 64usize;
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let mut out = mlp.forward(&x, &indices).unwrap();
  assert_eq!(out.shape(), vec![n, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Reference: per token, square(fc1_e(x)) — expert 0 identity, expert 1 swap.
  let mut want = Vec::with_capacity(n * 2);
  for t in 0..n {
    let (x0, x1) = (t as f32, t as f32 + 1.0);
    if t % 2 == 0 {
      want.push(x0 * x0);
      want.push(x1 * x1);
    } else {
      // expert 1 swaps before squaring
      want.push(x1 * x1);
      want.push(x0 * x0);
    }
  }
  assert_close(&got, &want);
}

#[test]
fn switch_mlp_new_rejects_mismatched_projection_shapes() {
  // fc2 must be the [hidden→input] inverse of fc1 [input→hidden]. fc1 is
  // [2→2]; an fc2 of [2→3] (wrong output_dims) is rejected.
  let bad_fc2 = Array::from_slice::<f32>(&[0.0f32; 2 * 3 * 2], &(2usize, 3usize, 2usize)).unwrap();
  let err = SwitchMLP::new(
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(bad_fc2, None).unwrap(),
    SwitchMLP::default_activation(),
  )
  .unwrap_err();
  assert!(
    matches!(err, crate::Error::ShapePairMismatch(_)),
    "expected ShapePairMismatch on mismatched fc2, got {err:?}"
  );
}

#[test]
fn gather_sort_then_scatter_unsort_round_trips() {
  // `gather_sort` reorders rows by expert id; `scatter_unsort` (with the
  // returned `inv_order`) must restore the original order exactly. Round-
  // tripping the *index* array through the pair must yield the input.
  // indices: [N=3, k=2] with deliberately-unsorted expert ids.
  let indices = Array::from_slice::<u32>(&[2, 0, 1, 1, 0, 2], &(3, 2)).unwrap();
  // gather_sort's `x` arg is the post-expand_dims input — rank ≥ 3 with
  // trailing (1, 1, D). Build a [3, 2, 1, 1, D=1] x whose value encodes the
  // flattened (token, k) slot so a mis-sort is visible.
  let x = Array::from_slice::<f32>(
    &(0..6).map(|i| i as f32).collect::<Vec<_>>(),
    &(3usize, 2usize, 1usize, 1usize),
  )
  .unwrap();
  let x_expanded = shape::expand_dims_axes(&x, &[-1]).unwrap(); // [3,2,1,1,1]
  let (_x_sorted, mut idx_sorted, inv_order) = gather_sort(&x_expanded, &indices).unwrap();
  // The sorted expert ids must be non-decreasing.
  let sorted_ids = idx_sorted.to_vec::<u32>().unwrap();
  let mut expected_sorted = vec![2u32, 0, 1, 1, 0, 2];
  expected_sorted.sort_unstable();
  assert_eq!(sorted_ids, expected_sorted);
  // scatter_unsort of the sorted ids (reshaped to indices.shape) restores
  // the original [3, 2] index array.
  let idx_as_rows = shape::expand_dims_axes(&idx_sorted, &[-1]).unwrap(); // [6,1]
  let mut restored = scatter_unsort(&idx_as_rows, &inv_order, &[3, 2]).unwrap();
  assert_eq!(restored.shape(), vec![3, 2, 1]);
  let restored_flat = restored.to_vec::<u32>().unwrap();
  assert_eq!(restored_flat, vec![2, 0, 1, 1, 0, 2]);
}

// ─── `indices` shape-contract regression (silent-MoE-corruption guard) ───
//
// `gather_sort` (the `indices.size() >= 64` sorted path) reads `M =
// indices.shape[-1]` as the top-k count and maps a sorted flat slot back to
// a token row via `order // M`. A top-1 `indices` shaped like the batch with
// NO explicit trailing k axis (`[N]` for x=`[N, D]`, `[B, S]` for
// x=`[B, S, D]`) would have its last *batch* dim mis-read as `M` — for `[N]`
// every `order // N` collapses to row 0, so all routed rows silently reuse
// token 0, yet unsort + squeeze still return a plausible `[N, D]` output.
// `check_routing_indices` rejects those ambiguous shapes (the reference
// always carries an explicit k axis); a top-1 caller must pass `[N, 1]`,
// which sorts correctly (`M == 1`, `order // 1 == order`).

#[test]
fn switch_glu_sorted_path_rejects_ambiguous_flat_indices() {
  // 64 routed tokens, `indices` shaped `[N]` (no trailing k axis) — the
  // sorted path is entered (`size >= 64`) but the shape is ambiguous: `N`
  // would be mis-read as the top-k count `M`. Must be a recoverable
  // `RankMismatch`, not silent corruption.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let n = 64usize;
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  // `[N]` — rank-1, same length as x's batch dim, NO explicit k axis.
  let indices = Array::from_slice::<u32>(&idx_data, &(n,)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let err = glu.forward(&x, &indices).unwrap_err();
  // x=[N, D] ⇒ x_batch=[N], expected_rank=2; indices=[N] is rank-1 (missing
  // the trailing k axis) ⇒ now categorised as RankMismatch
  // rather than a misleading "expected [N], got [N]" ShapePairMismatch.
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 1, "rank-1 indices ⇒ actual rank 1");
      assert_eq!(payload.actual_shape(), &[64usize]);
    }
    other => panic!("expected RankMismatch on ambiguous [N] indices, got {other:?}"),
  }
}

#[test]
fn switch_glu_sorted_path_rejects_ambiguous_batch_indices() {
  // 64 routed tokens via a 2-D batch x=`[B=8, S=8, D=2]`, `indices` shaped
  // `[B, S]` (no trailing k axis). `S` would be mis-read as `M`; reject.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let (b, s) = (8usize, 8usize);
  let mut x_data = Vec::with_capacity(b * s * 2);
  let mut idx_data = Vec::with_capacity(b * s);
  for t in 0..(b * s) {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
  // `[B, S]` — rank matches x's batch dims exactly, NO explicit k axis.
  let indices = Array::from_slice::<u32>(&idx_data, &(b, s)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let err = glu.forward(&x, &indices).unwrap_err();
  // x=[B, S, D] ⇒ x_batch=[B, S], expected_rank=3; indices=[B, S] is rank-2
  // (missing the trailing k axis) ⇒ RankMismatch.
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 2, "rank-2 indices ⇒ actual rank 2");
      assert_eq!(payload.actual_shape(), &[8usize, 8]);
    }
    other => panic!("expected RankMismatch on ambiguous [B, S] indices, got {other:?}"),
  }
}

#[test]
fn switch_glu_sorted_path_top1_explicit_k_routes_each_token_to_its_expert() {
  // The accepted top-1 form: `indices` shaped `[N, 1]` (explicit k=1). On the
  // sorted path (`size >= 64`) every token must route to ITS OWN selected
  // expert. Tokens 0..32 → expert 0 (identity gate), tokens 32..64 → expert 1
  // (swap gate); every token has a DISTINCT feature pair, so a `[N]`-style
  // mis-route (all rows reuse token 0) would make every output equal token
  // 0's value and fail the per-row assertion below.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let n = 64usize;
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    // Distinct, non-zero per-token features so reusing token 0 is detectable.
    x_data.push(t as f32 + 1.0);
    x_data.push(t as f32 + 2.0);
    // First half → expert 0, second half → expert 1.
    idx_data.push(if t < n / 2 { 0u32 } else { 1u32 });
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let mut out = glu.forward(&x, &indices).unwrap();
  assert_eq!(out.shape(), vec![n, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Reference: silu(gate_e(x)) · x — expert 0 keeps features, expert 1 swaps.
  let mut want = Vec::with_capacity(n * 2);
  for t in 0..n {
    let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
    if t < n / 2 {
      // expert 0: identity gate
      want.push(silu_ref(x0) * x0);
      want.push(silu_ref(x1) * x1);
    } else {
      // expert 1: swap gate sees [x1, x0]
      want.push(silu_ref(x1) * x0);
      want.push(silu_ref(x0) * x1);
    }
  }
  assert_close(&got, &want);
}

#[test]
fn switch_glu_sorted_path_explicit_2d_batch_k_routes_each_token() {
  // The explicit-`[..batch.., k]` contract with a 2-D batch: x=`[B=8, S=8,
  // D=2]`, `indices`=`[B, S, k=1]` (one extra trailing axis beyond x's
  // batch dims). Accepted, sorted path, each token routed to its own expert.
  let glu = SwitchGLU::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    SwitchGLU::default_activation(),
  )
  .unwrap();
  let (b, s) = (8usize, 8usize);
  let mut x_data = Vec::with_capacity(b * s * 2);
  let mut idx_data = Vec::with_capacity(b * s);
  for t in 0..(b * s) {
    x_data.push(t as f32 + 1.0);
    x_data.push(t as f32 + 2.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(b, s, 1usize)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let mut out = glu.forward(&x, &indices).unwrap();
  // forward returns `[..batch.., k, input_dims]` == `[B, S, 1, 2]`.
  assert_eq!(out.shape(), vec![b, s, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  let mut want = Vec::with_capacity(b * s * 2);
  for t in 0..(b * s) {
    let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
    if t % 2 == 0 {
      want.push(silu_ref(x0) * x0);
      want.push(silu_ref(x1) * x1);
    } else {
      want.push(silu_ref(x1) * x0);
      want.push(silu_ref(x0) * x1);
    }
  }
  assert_close(&got, &want);
}

#[test]
fn switch_mlp_sorted_path_rejects_ambiguous_flat_indices() {
  // `SwitchMLP` sibling of `switch_glu_sorted_path_rejects_ambiguous_flat_indices`:
  // a `[N]` top-1 `indices` on the sorted path is rejected, not mis-routed.
  let square: Activation = Box::new(|a: &Array| a.multiply(a));
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    square,
  )
  .unwrap();
  let n = 64usize;
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(n,)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let err = mlp.forward(&x, &indices).unwrap_err();
  // Missing-k-axis case ⇒ RankMismatch (was a misleading
  // ShapePairMismatch with expected==actual).
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 1, "rank-1 indices ⇒ actual rank 1");
      assert_eq!(payload.actual_shape(), &[64usize]);
    }
    other => panic!("expected RankMismatch on ambiguous [N] indices, got {other:?}"),
  }
}

#[test]
fn switch_mlp_sorted_path_rejects_ambiguous_batch_indices() {
  // `SwitchMLP` sibling: a `[B, S]` top-1 `indices` (no k axis) on a 2-D
  // batch x=`[B, S, D]` is rejected on the sorted path.
  let square: Activation = Box::new(|a: &Array| a.multiply(a));
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    square,
  )
  .unwrap();
  let (b, s) = (8usize, 8usize);
  let mut x_data = Vec::with_capacity(b * s * 2);
  let mut idx_data = Vec::with_capacity(b * s);
  for t in 0..(b * s) {
    x_data.push(t as f32);
    x_data.push(t as f32 + 1.0);
    idx_data.push((t % 2) as u32);
  }
  let x = Array::from_slice::<f32>(&x_data, &(b, s, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(b, s)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let err = mlp.forward(&x, &indices).unwrap_err();
  // Missing-k-axis case ⇒ RankMismatch.
  match err {
    crate::Error::RankMismatch(payload) => {
      assert_eq!(payload.actual(), 2, "rank-2 indices ⇒ actual rank 2");
      assert_eq!(payload.actual_shape(), &[8usize, 8]);
    }
    other => panic!("expected RankMismatch on ambiguous [B, S] indices, got {other:?}"),
  }
}

#[test]
fn switch_mlp_sorted_path_top1_explicit_k_routes_each_token_to_its_expert() {
  // `SwitchMLP` sibling of the SwitchGLU `[N, 1]` regression: the accepted
  // explicit-k=1 top-1 form, sorted path, every token routed to ITS OWN
  // expert with distinct per-token features (a `[N]`-style mis-route reusing
  // token 0 would fail the per-row assertion).
  let square: Activation = Box::new(|a: &Array| a.multiply(a));
  let mlp = SwitchMLP::new(
    SwitchLinear::from_parts(identity_then_swap_weight(), None).unwrap(),
    SwitchLinear::from_parts(all_identity_weight(), None).unwrap(),
    square,
  )
  .unwrap();
  let n = 64usize;
  let mut x_data = Vec::with_capacity(n * 2);
  let mut idx_data = Vec::with_capacity(n);
  for t in 0..n {
    x_data.push(t as f32 + 1.0);
    x_data.push(t as f32 + 2.0);
    idx_data.push(if t < n / 2 { 0u32 } else { 1u32 });
  }
  let x = Array::from_slice::<f32>(&x_data, &(n, 2usize)).unwrap();
  let indices = Array::from_slice::<u32>(&idx_data, &(n, 1usize)).unwrap();
  assert!(indices.size() >= 64, "test must exercise the sorted path");
  let mut out = mlp.forward(&x, &indices).unwrap();
  assert_eq!(out.shape(), vec![n, 1, 2]);
  let got = out.to_vec::<f32>().unwrap();
  // Reference: square(fc1_e(x)) — expert 0 identity, expert 1 swap.
  let mut want = Vec::with_capacity(n * 2);
  for t in 0..n {
    let (x0, x1) = (t as f32 + 1.0, t as f32 + 2.0);
    if t < n / 2 {
      want.push(x0 * x0);
      want.push(x1 * x1);
    } else {
      want.push(x1 * x1);
      want.push(x0 * x0);
    }
  }
  assert_close(&got, &want);
}
