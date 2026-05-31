//! M3 — happy-path tests for the quantization ops.
//!
//! `quantize` → `dequantize` round-trips a matrix (4-bit affine; checks
//! shape/dtype + approximate value recovery), and `quantized_matmul` is
//! exercised against the dequantized reference for output shape parity.
//!
//! `quantize` returns `(w_q, scales, Option<biases>)`: the `"affine"` scheme
//! yields `Some(biases)` (3 mlx outputs), while the bias-less float schemes
//! (`"mxfp4"`/`"mxfp8"`/`"nvfp4"`) yield `None` (2 mlx outputs). Both arity
//! paths are exercised below — the affine round-trip/matmul and a focused
//! `mxfp4` non-affine round-trip asserting `biases == None`.

use mlxrs::{Array, Dtype, ops::quantized};

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const MODE: &str = "affine";

/// 8 x 128 f32 matrix with a smooth ramp so 4-bit affine quant stays close.
fn sample_matrix() -> Array {
  let rows = 8usize;
  let cols = 128usize;
  let mut data = Vec::with_capacity(rows * cols);
  for r in 0..rows {
    for c in 0..cols {
      data.push(((r * cols + c) as f32) * 0.01);
    }
  }
  Array::from_slice::<f32>(&data, &[rows as i32, cols as i32]).unwrap()
}

#[test]
fn quantize_then_dequantize_round_trips_shape_and_dtype() {
  let w = sample_matrix();
  let (w_q, scales, biases) = quantized::quantize(&w, GROUP_SIZE, BITS, MODE, None).unwrap();

  // Affine mode produces per-group biases (mlx's 3-output path).
  let biases = biases.expect("affine quantize yields Some(biases)");

  // Packed weights compress the 128-wide last dim; scales/biases are per-group.
  let cols = 128usize;
  let groups = cols / GROUP_SIZE as usize;
  assert_eq!(scales.shape(), vec![8, groups]);
  assert_eq!(biases.shape(), vec![8, groups]);

  let mut deq = quantized::dequantize(
    &w_q,
    &scales,
    Some(&biases),
    GROUP_SIZE,
    BITS,
    MODE,
    None,
    Some(Dtype::F32),
  )
  .unwrap();
  assert_eq!(deq.shape(), vec![8, cols]);
  assert_eq!(deq.dtype().unwrap(), Dtype::F32);

  // 4-bit affine quant of a smooth ramp recovers values within a loose band.
  let got = deq.to_vec::<f32>().unwrap();
  let mut w_copy = w;
  let want = w_copy.to_vec::<f32>().unwrap();
  let max_abs = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
  for (g, e) in got.iter().zip(want.iter()) {
    assert!(
      (g - e).abs() <= 0.1 * max_abs + 1e-3,
      "dequant drift too large: got={g} want={e}"
    );
  }
}

#[test]
fn quantized_matmul_output_shape() {
  let w = sample_matrix(); // [8, 128]
  let (w_q, scales, biases) = quantized::quantize(&w, GROUP_SIZE, BITS, MODE, None).unwrap();
  let biases = biases.expect("affine quantize yields Some(biases)");

  // x: [4, 128]; transpose=true multiplies by wᵀ → [4, 8].
  let x_data: Vec<f32> = (0..4 * 128).map(|i| (i as f32) * 0.001).collect();
  let x = Array::from_slice::<f32>(&x_data, &[4i32, 128i32]).unwrap();

  let mut out = quantized::quantized_matmul(
    &x,
    &w_q,
    &scales,
    Some(&biases),
    true,
    GROUP_SIZE,
    BITS,
    MODE,
  )
  .unwrap();
  assert_eq!(out.shape(), vec![4, 8]);
  assert_eq!(out.dtype().unwrap(), Dtype::F32);
  // Force materialization to ensure the graph actually evaluates.
  let _ = out.to_vec::<f32>().unwrap();
}

/// Non-affine (bias-less) `mxfp4` mode exercises the 2-output `quantize`
/// arity path: it MUST yield `biases == None` and round-trip through
/// `dequantize` (which takes `biases = None` for float modes).
///
/// `mxfp4` is constructible & runnable at the pinned mlx (v0.31.2): its
/// `fp_quantize` path is fully implemented on both the Metal backend
/// (`fast::Quantize::eval_gpu` has a `mxfp4_quantize_*` kernel) and the CPU
/// fallback, so this executes on the default stream like the affine tests.
/// `mxfp4` requires `group_size = 32`, `bits = 4` (mlx
/// `quantization_params_from_mode`); other values are rejected by mlx-c.
#[test]
fn quantize_mxfp4_is_bias_less_and_round_trips() {
  const MXFP4_GS: i32 = 32;
  const MXFP4_BITS: i32 = 4;

  let w = sample_matrix(); // [8, 128]
  let (w_q, scales, biases) = quantized::quantize(&w, MXFP4_GS, MXFP4_BITS, "mxfp4", None).unwrap();

  // The whole point: bias-less float modes return only (w_q, scales).
  assert!(
    biases.is_none(),
    "mxfp4 is bias-less: quantize must return biases == None"
  );

  let cols = 128usize;
  let groups = cols / MXFP4_GS as usize;
  assert_eq!(scales.shape(), vec![8, groups]);

  let mut deq = quantized::dequantize(
    &w_q,
    &scales,
    None, // bias-less float mode: no biases input
    MXFP4_GS,
    MXFP4_BITS,
    "mxfp4",
    None,
    Some(Dtype::F32),
  )
  .unwrap();
  assert_eq!(deq.shape(), vec![8, cols]);
  assert_eq!(deq.dtype().unwrap(), Dtype::F32);
  // Force materialization so the float-quant graph actually evaluates.
  let _ = deq.to_vec::<f32>().unwrap();
}

/// An unknown `mode` is rejected by mlx-c (`string_to_quantization_mode`
/// throws) and surfaces as a recoverable `Err`, never a panic — this also
/// exercises the non-3 arity / error plumbing on the `quantize` path.
#[test]
fn quantize_rejects_unknown_mode_without_panicking() {
  let w = sample_matrix();
  let err = quantized::quantize(&w, GROUP_SIZE, BITS, "not-a-real-mode", None);
  assert!(
    err.is_err(),
    "an invalid quantization mode must return Err, not panic or succeed"
  );
}
