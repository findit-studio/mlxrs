//! Value/shape/dtype coverage for `array::construction` (#260).
//!
//! `array/construction.rs` had ZERO direct coverage. The pre-existing
//! `error_paths.rs` already exercises `from_slice`'s negative-dim /
//! overflow / zero-element guards and `archetypes.rs` smoke-tests the
//! happy path of `ones`/`zeros`/`arange`, so this file deliberately
//! targets the GENUINE gaps:
//!   * `eye` and `linspace` — completely untested anywhere; plus `eye`'s
//!     non-square (`m`) and off-diagonal (`k`) parameters (#259 construction
//!     parity).
//!   * `full`'s exact-fill contract: `value` is a `T`, so the 0-d scalar handed
//!     to `mlx_full` is already dtype `T` and no value cast occurs — verify on
//!     an integer dtype and on a wide `u32` value the old f32/i32 path couldn't
//!     represent.
//!   * `arange`'s half-open `[start, stop)` semantics with non-unit and
//!     negative steps, plus the empty (`start == stop`) result.
//!   * `linspace`'s inclusive `[start, stop]` endpoints and the `num == 1`
//!     special case (C++ returns `astype(array({start}), dtype)`).
//!   * `ones`/`zeros` dtype correctness on non-f32 element types.
//!
//! Crate accessor rule (`feedback_no_implicit_eval`): `item`/`to_vec`/
//! `as_slice` are `&mut self` and force an explicit eval inside themselves,
//! so the build → read pattern used by the sibling test files (`archetypes`,
//! `ops_arithmetic`) is followed verbatim — no separate `eval()` call is
//! needed before a `&mut self` accessor.

use mlxrs::{Array, Dtype};

// ───────── ones / zeros ─────────

#[test]
fn ones_values_shape_and_dtype() {
  let mut a = Array::ones::<f32>(&(2, 3)).unwrap();
  assert_eq!(a.shape(), vec![2, 3]);
  assert_eq!(a.size(), 6);
  assert_eq!(a.ndim(), 2);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0; 6]);
}

#[test]
fn ones_i32_has_integer_dtype_and_values() {
  // dtype is driven purely by the `T` type parameter, independent of value.
  let mut a = Array::ones::<i32>(&(4,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![1_i32; 4]);
}

#[test]
fn zeros_values_shape_and_dtype() {
  let mut a = Array::zeros::<f32>(&(3, 2)).unwrap();
  assert_eq!(a.shape(), vec![3, 2]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn zeros_u32_dtype_and_values() {
  let mut a = Array::zeros::<u32>(&(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U32);
  assert_eq!(a.to_vec::<u32>().unwrap(), vec![0_u32; 5]);
}

// ───────── full ─────────

#[test]
fn full_f32_fills_every_element() {
  let mut a = Array::full::<f32>(&(2, 2), 7.5).unwrap();
  assert_eq!(a.shape(), vec![2, 2]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![7.5; 4]);
}

#[test]
fn full_fills_requested_integer_dtype() {
  // `full::<i32>` takes an exact `i32` value; the 0-d scalar handed to
  // `mlx_full` is already I32, so no cast occurs and the result is the I32
  // value 3.
  let mut a = Array::full::<i32>(&(3,), 3).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![3_i32; 3]);
}

#[test]
fn full_i32_fill_keeps_dtype_and_value() {
  // #259: the fill value is an exact `i32`, so dtype + value round-trip exactly
  // (the 0-d scalar handed to `mlx_full` is already I32 — no float intermediate
  // that could lose precision above 2^24).
  let mut a = Array::full::<i32>(&(2,), 7).unwrap();
  assert_eq!(a.shape(), vec![2]);
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![7_i32, 7]);
}

#[test]
fn full_u32_wide_value_is_exact() {
  // #259: `value: T` makes the fill exact for any in-range `T` value. A `u32`
  // above i32::MAX (3e9) is faithfully represented — the old `impl Into<f64>` +
  // i32-scalar path could not (it rejected/clipped it). No cast, no wrap.
  let mut a = Array::full::<u32>(&(2,), 3_000_000_000u32).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U32);
  assert_eq!(
    a.to_vec::<u32>().unwrap(),
    vec![3_000_000_000, 3_000_000_000]
  );
}

// ───────── eye ─────────

#[test]
fn eye_3_is_identity_matrix() {
  // mlx `eye(n, n, k=0)` → row-major identity. Flattened, the 1.0s sit on
  // indices 0, 4, 8 (the i*(n+1) diagonal) and everything else is 0.0.
  let mut a = Array::eye::<f32>(3, None, 0).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![
      1.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, //
      0.0, 0.0, 1.0,
    ]
  );
}

#[test]
fn eye_1_is_single_one() {
  let mut a = Array::eye::<f32>(1, None, 0).unwrap();
  assert_eq!(a.shape(), vec![1, 1]);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn eye_non_square_uses_m_for_columns() {
  // #259: `m = Some(4)` → 3 rows, 4 columns; the main diagonal (k=0) holds
  // ones at flat indices 0, 5, 10 (i*(m+1)) and the rest are zero.
  let mut a = Array::eye::<f32>(3, Some(4), 0).unwrap();
  assert_eq!(a.shape(), vec![3, 4]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0, //
      0.0, 0.0, 1.0, 0.0,
    ]
  );
}

#[test]
fn eye_positive_k_is_super_diagonal() {
  // #259: `k = 1` shifts the ones one column above the main diagonal. For a
  // 3×3 that is flat indices 1 and 5; index 0 (main diagonal) is now 0.0.
  let mut a = Array::eye::<f32>(3, None, 1).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  let v = a.to_vec::<f32>().unwrap();
  assert_eq!(v[1], 1.0);
  assert_eq!(v[5], 1.0);
  assert_eq!(v[0], 0.0);
  assert_eq!(v[8], 0.0);
}

#[test]
fn eye_negative_k_is_sub_diagonal() {
  // #259: `k = -1` shifts the ones one row below the main diagonal — a valid
  // negative offset. For a 3×3 that is flat indices 3 and 7.
  let mut a = Array::eye::<f32>(3, None, -1).unwrap();
  assert_eq!(a.shape(), vec![3, 3]);
  let v = a.to_vec::<f32>().unwrap();
  assert_eq!(v[3], 1.0);
  assert_eq!(v[7], 1.0);
  assert_eq!(v[0], 0.0);
  assert_eq!(v[4], 0.0);
}

#[test]
fn eye_k_i32_min_is_rejected_not_ub() {
  // #259 / Codex: mlx's eye evaluates `-k`, so k == i32::MIN would overflow in
  // C++ (UB). The wrapper rejects it with a typed error instead of calling FFI.
  let err = Array::eye::<f32>(3, None, i32::MIN).unwrap_err();
  assert!(matches!(err, mlxrs::Error::OutOfRange(_)), "got {err:?}");
}

// ───────── arange ─────────

#[test]
fn arange_unit_step_is_half_open() {
  // [0, 5) step 1 → 0,1,2,3,4 (stop is exclusive). f32 dtype always.
  let mut a = Array::arange(0.0, 5.0, 1.0).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arange_non_unit_step() {
  // [0, 1) step 0.25 → 0, 0.25, 0.5, 0.75 (4 values, 1.0 excluded).
  let mut a = Array::arange(0.0, 1.0, 0.25).unwrap();
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
}

#[test]
fn arange_negative_step_counts_down() {
  // [5, 0) step -1 → 5,4,3,2,1 (stop 0 excluded).
  let mut a = Array::arange(5.0, 0.0, -1.0).unwrap();
  assert_eq!(a.to_vec::<f32>().unwrap(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn arange_empty_when_start_equals_stop() {
  let mut a = Array::arange(2.0, 2.0, 1.0).unwrap();
  assert_eq!(a.size(), 0);
  assert_eq!(a.to_vec::<f32>().unwrap(), Vec::<f32>::new());
}

// ───────── linspace ─────────

#[test]
fn linspace_includes_both_endpoints() {
  // [0, 1] with 5 samples → 0, 0.25, 0.5, 0.75, 1.0 (BOTH ends inclusive,
  // unlike arange). f32 dtype.
  let mut a = Array::linspace(0.0, 1.0, 5).unwrap();
  assert_eq!(a.size(), 5);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  let v = a.to_vec::<f32>().unwrap();
  let expected = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
  assert_eq!(v.len(), expected.len());
  for (got, want) in v.iter().zip(expected.iter()) {
    assert!((got - want).abs() < 1e-6, "linspace got {got}, want {want}");
  }
}

#[test]
fn linspace_num_1_returns_start_only() {
  // C++ special case: `num == 1` → `astype(array({start}), dtype)` — a
  // single-element array holding `start` (NOT `stop`).
  let mut a = Array::linspace(3.0, 9.0, 1).unwrap();
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<f32>().unwrap(), 3.0);
}

// ───────── from_slice ─────────

#[test]
fn from_slice_preserves_order_and_shape() {
  // Row-major flatten: the buffer order is preserved 1:1 for a contiguous
  // freshly-built array, and the shape product matches the buffer length.
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  assert_eq!(a.shape(), vec![2, 3]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn from_slice_i32_dtype_is_driven_by_element_type() {
  let mut a = Array::from_slice::<i32>(&[10, 20, 30], &(3,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert_eq!(a.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn from_slice_scalar_rank0_round_trips() {
  // Empty shape `[]` → rank-0 scalar (size 1). `item` reads it back.
  let empty: [i32; 0] = [];
  let mut a = Array::from_slice::<f32>(&[42.0], &empty).unwrap();
  assert_eq!(a.ndim(), 0);
  assert_eq!(a.shape(), Vec::<usize>::new());
  assert_eq!(a.size(), 1);
  assert_eq!(a.item::<f32>().unwrap(), 42.0);
}

#[test]
fn from_slice_length_mismatch_is_typed_error() {
  // shape product (2*2 = 4) != data.len() (3) → typed LengthMismatch
  // carrying expected=4, actual=3 (NOT a panic, NOT a Backend string).
  let r = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(2, 2));
  match r {
    Err(mlxrs::Error::LengthMismatch(p)) => {
      assert_eq!(
        p.context(),
        "Array::from_slice: shape product vs data.len()"
      );
      assert_eq!(p.expected(), 4);
      assert_eq!(p.actual(), 3);
    }
    other => panic!("expected Err(LengthMismatch) on shape/len mismatch, got {other:?}"),
  }
}
