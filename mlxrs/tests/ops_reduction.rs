//! Phase 4 Branch A: reduction op happy-path tests.
//!
//! Mean / max / min / prod each get a full-reduction scalar test plus an
//! axis-reduction test that exercises the `_axes` form. Empty-axes contract
//! (returns clone) is sanity-checked once per family via the shared pattern.

use mlxrs::Array;

// ───────── mean ─────────

#[test]
fn mean_of_2x2_ones_yields_1() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let mut r = a.mean(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 1.0);
}

#[test]
fn mean_axes_of_2x2_along_axis0() {
  // [[1, 2], [3, 4]] mean over axis 0 → [2, 3]
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.mean_axes(&[0], false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn mean_axes_empty_is_identity_for_float() {
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.mean_axes(&[], false).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn mean_axes_empty_promotes_int_to_float() {
  // mean over no axes is the identity in shape/value, but `mean` always
  // promotes int inputs to float — both the empty and non-empty paths must
  // agree on dtype. The previous short-circuit-via-try_clone preserved the
  // input dtype, splitting the contract (Codex PR #6 finding).
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  assert_eq!(a.dtype().unwrap(), mlxrs::Dtype::I32);
  let r_empty = a.mean_axes(&[], false).unwrap();
  let r_full = mlxrs::ops::reduction::mean(&a, false).unwrap();
  assert_eq!(
    r_empty.dtype().unwrap(),
    r_full.dtype().unwrap(),
    "empty-axes and full-reduction must agree on output dtype",
  );
  assert_eq!(
    r_empty.dtype().unwrap(),
    mlxrs::Dtype::F32,
    "mean of int promotes to f32",
  );
}

// ───────── max ─────────

#[test]
fn max_of_arange_yields_last() {
  // arange(0, 5) → [0, 1, 2, 3, 4]; max → 4
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let mut r = a.max(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 4.0);
}

#[test]
fn max_axes_of_2x2_along_axis1() {
  // [[1, 2], [3, 4]] max over axis 1 → [2, 4]
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.max_axes(&[1], false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
}

#[test]
fn max_axes_keepdims_preserves_axis() {
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let r = a.max_axes(&[1], true).unwrap();
  assert_eq!(r.shape(), vec![2, 1]);
}

#[test]
fn max_axes_empty_on_zero_size_errors() {
  // MLX checks size==0 BEFORE the no-axes early return for max/min, so empty
  // axes on a zero-size array must error (not silently return a clone).
  // Locks in the Codex PR #6 round-2 fix. Same contract for min_axes.
  let a = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  assert_eq!(a.size(), 0);
  let r_max = mlxrs::ops::reduction::max_axes(&a, &[], false);
  assert!(
    matches!(r_max, Err(mlxrs::Error::Backend { .. })),
    "expected Err(Backend) for max_axes(zero_size, &[]), got {r_max:?}",
  );
  let r_min = mlxrs::ops::reduction::min_axes(&a, &[], false);
  assert!(
    matches!(r_min, Err(mlxrs::Error::Backend { .. })),
    "expected Err(Backend) for min_axes(zero_size, &[]), got {r_min:?}",
  );
}

#[test]
fn max_axes_empty_on_non_zero_size_is_identity() {
  // For non-zero-size arrays, empty axes is the no-op identity (numpy
  // semantics) — MLX agrees once it passes the size>0 check.
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.max_axes(&[], false).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ───────── min ─────────

#[test]
fn min_of_arange_yields_first() {
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let mut r = a.min(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn min_axes_of_2x2_along_axis0() {
  // [[1, 2], [3, 4]] min over axis 0 → [1, 2]
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.min_axes(&[0], false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

// ───────── prod ─────────

#[test]
fn prod_of_2x2_twos_yields_16() {
  // 2 * 2 * 2 * 2 = 16
  let a = Array::full::<f32>(&(2, 2), 2.0).unwrap();
  let mut r = a.prod(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 16.0);
}

#[test]
fn prod_axes_of_2x2_along_axis1() {
  // [[1, 2], [3, 4]] prod over axis 1 → [2, 12]
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.prod_axes(&[1], false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 12.0]);
}

#[test]
fn prod_axes_empty_returns_clone() {
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.prod_axes(&[], false).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ───────── free-fn parity sanity ─────────

#[test]
fn mean_freefn_parity_with_method() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let mut method = a.mean(false).unwrap();
  let mut freefn = mlxrs::ops::reduction::mean(&a, false).unwrap();
  assert_eq!(method.item::<f32>().unwrap(), freefn.item::<f32>().unwrap());
}
