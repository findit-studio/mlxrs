//! Real-device integration tests for `mlxrs::ops::fast::metal_kernel`.
//!
//! Each test below requires a Metal-capable GPU at apply time, so they are
//! gated behind `#[cfg(target_os = "macos")] #[ignore]`. Run locally with:
//!
//!     CARGO_TARGET_DIR=/tmp/mlxrs-metalkernel-iso \
//!     cargo +nightly test -p mlxrs --test ops_fast_metal_kernel \
//!         -- --ignored --test-threads=1
//!
//! Tests:
//!
//! - `exp_kernel_writes_e_to_every_element` — a 1-input, 1-output kernel that
//!   computes `out[gid] = exp(input[gid])` over an `Array::ones([8])` and
//!   asserts every output element ≈ e (f32 epsilon).
//! - `saxpy_kernel_uses_template_alpha` — a 2-input, 1-output kernel that
//!   computes `out[gid] = ALPHA * x[gid] + y[gid]` with `ALPHA` supplied as a
//!   `KernelTemplateArg::Int(2)`.
//! - `multi_output_kernel_emits_two_arrays` — a 1-input, 2-output kernel
//!   that emits `sum[gid] = input[gid] + 1` and `diff[gid] = input[gid] - 1`
//!   over an `Array::full([4], 5.0)`. Checks that `apply` returns two
//!   `Array`s with the declared shapes / dtypes and the expected values.
//! - `apply_rejects_shape_count_mismatch` — `MetalKernel::new` declares one
//!   output_name but the per-call `MetalKernelApplyConfig` supplies two
//!   output_shapes; `apply` returns `Error::LengthMismatch` without touching
//!   the device.
//! - `apply_accepts_valid_multi_dim_output_shape` — a 1-input, 1-output
//!   kernel produces a `[4, 8, 16]`-shaped output; sanity-checks that a
//!   ranked-3 output_shape routes through the FFI without tripping the
//!   wrapper's new empty / negative-dim rejections.

#![cfg(target_os = "macos")]

use mlxrs::{
  Array, Dtype,
  ops::fast::metal_kernel::{KernelTemplateArg, MetalKernel, MetalKernelApplyConfig},
};

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn exp_kernel_writes_e_to_every_element() {
  let input = Array::ones::<f32>(&[8]).unwrap();
  let kernel = MetalKernel::new(
    "exp_kernel",
    &["input"],
    &["out"],
    // mlx-c auto-generates the function signature; this is just the body.
    // `thread_position_in_grid.x` is mlx-c's launch coordinate variable.
    "uint elem = thread_position_in_grid.x;
     out[elem] = exp(input[elem]);",
    "",
    /* ensure_row_contiguous */ true,
    /* atomic_outputs */ false,
  )
  .unwrap();

  assert_eq!(kernel.output_arity(), 1);
  assert_eq!(kernel.output_names_slice(), &["out".to_string()]);

  let cfg = MetalKernelApplyConfig::new(
    /* grid */ [8, 1, 1],
    /* thread_group */ [8, 1, 1],
    /* output_shapes */ vec![vec![8]],
    /* output_dtypes */ vec![Dtype::F32],
  )
  .unwrap();
  let mut outs = kernel.apply(&[&input], &cfg).unwrap();
  assert_eq!(outs.len(), 1);
  assert_eq!(outs[0].shape(), vec![8]);
  let buf: Vec<f32> = outs[0].to_vec().unwrap();
  let e = std::f32::consts::E;
  for (i, v) in buf.iter().enumerate() {
    assert!((v - e).abs() < 1e-5, "out[{i}] = {v}, expected ≈ {e}");
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn saxpy_kernel_uses_template_alpha() {
  // x = [1, 2, 3, 4], y = [10, 20, 30, 40], ALPHA = 2  →  out = [12, 24, 36, 48]
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
  let y = Array::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();

  let kernel = MetalKernel::new(
    "saxpy_kernel",
    &["x", "y"],
    &["out"],
    "uint elem = thread_position_in_grid.x;
     out[elem] = float(ALPHA) * x[elem] + y[elem];",
    "",
    true,
    false,
  )
  .unwrap();

  let cfg = MetalKernelApplyConfig::new([4, 1, 1], [4, 1, 1], vec![vec![4]], vec![Dtype::F32])
    .unwrap()
    .with_template(vec![("ALPHA".to_string(), KernelTemplateArg::Int(2))]);
  let mut outs = kernel.apply(&[&x, &y], &cfg).unwrap();
  assert_eq!(outs.len(), 1);
  let buf: Vec<f32> = outs[0].to_vec().unwrap();
  assert_eq!(buf, vec![12.0, 24.0, 36.0, 48.0]);
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn multi_output_kernel_emits_two_arrays() {
  // input = [5, 5, 5, 5]  →  sum = [6, 6, 6, 6], diff = [4, 4, 4, 4].
  let input = Array::full::<f32>(&[4], 5.0).unwrap();

  let kernel = MetalKernel::new(
    "split_kernel",
    &["input"],
    &["sum", "diff"],
    "uint elem = thread_position_in_grid.x;
     sum[elem] = input[elem] + 1.0;
     diff[elem] = input[elem] - 1.0;",
    "",
    true,
    false,
  )
  .unwrap();

  assert_eq!(kernel.output_arity(), 2);

  let cfg = MetalKernelApplyConfig::new(
    [4, 1, 1],
    [4, 1, 1],
    vec![vec![4], vec![4]],
    vec![Dtype::F32, Dtype::F32],
  )
  .unwrap();
  let mut outs = kernel.apply(&[&input], &cfg).unwrap();
  assert_eq!(outs.len(), 2);
  assert_eq!(outs[0].shape(), vec![4]);
  assert_eq!(outs[1].shape(), vec![4]);
  let sum: Vec<f32> = outs[0].to_vec().unwrap();
  let diff: Vec<f32> = outs[1].to_vec().unwrap();
  assert_eq!(sum, vec![6.0, 6.0, 6.0, 6.0]);
  assert_eq!(diff, vec![4.0, 4.0, 4.0, 4.0]);
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn apply_accepts_valid_multi_dim_output_shape() {
  // Sanity-check the FFI round-trip for a ranked-3 output shape: every
  // dimension is positive and the slice is non-empty, so the wrapper's
  // new `validate_dims` + empty-shape guard pass through and `dim_ptr`
  // forwards `Vec::as_ptr()` unchanged. The kernel itself writes a
  // constant so the test asserts shape + dtype rather than per-element
  // contents.
  let input = Array::ones::<f32>(&[4, 8, 16]).unwrap();
  let kernel = MetalKernel::new(
    "constant_3d_kernel",
    &["input"],
    &["out"],
    "uint elem = thread_position_in_grid.x;
     out[elem] = input[elem] * 2.0;",
    "",
    /* ensure_row_contiguous */ true,
    /* atomic_outputs */ false,
  )
  .unwrap();

  let cfg = MetalKernelApplyConfig::new(
    /* grid */ [4 * 8 * 16, 1, 1],
    /* thread_group */ [32, 1, 1],
    /* output_shapes */ vec![vec![4, 8, 16]],
    /* output_dtypes */ vec![Dtype::F32],
  )
  .unwrap();
  let mut outs = kernel.apply(&[&input], &cfg).unwrap();
  assert_eq!(outs.len(), 1);
  assert_eq!(outs[0].shape(), vec![4, 8, 16]);
  assert_eq!(outs[0].dtype().unwrap(), Dtype::F32);
  let buf: Vec<f32> = outs[0].to_vec().unwrap();
  assert_eq!(buf.len(), 4 * 8 * 16);
  for (i, v) in buf.iter().enumerate() {
    assert!((v - 2.0_f32).abs() < 1e-5, "out[{i}] = {v}, expected 2.0");
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn apply_rejects_shape_count_mismatch() {
  // Kernel declares one output, but the per-call config supplies two
  // output_shapes. The wrapper rejects before reaching mlx-c.
  let kernel = MetalKernel::new(
    "noop",
    &["x"],
    &["out"],
    "uint elem = thread_position_in_grid.x; out[elem] = x[elem];",
    "",
    true,
    false,
  )
  .unwrap();
  let input = Array::ones::<f32>(&[4]).unwrap();
  let cfg = MetalKernelApplyConfig::new(
    [4, 1, 1],
    [4, 1, 1],
    vec![vec![4], vec![4]],
    vec![Dtype::F32, Dtype::F32],
  )
  .unwrap();
  let err = kernel
    .apply(&[&input], &cfg)
    .expect_err("declared 1 output_name but supplied 2 output_shapes");
  match err {
    mlxrs::Error::LengthMismatch(payload) => {
      // 1 output_name declared, 2 output_shapes supplied.
      assert_eq!(payload.expected(), 1, "expected count: {:?}", payload);
      assert_eq!(payload.actual(), 2, "actual count: {:?}", payload);
    }
    other => panic!("expected LengthMismatch, got: {other:?}"),
  }
}
