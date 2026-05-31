//! Reduction op happy-path tests.
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
  // input dtype, splitting the contract.
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
  let a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
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
  // Same contract for min_axes.
  let a = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  assert_eq!(a.size(), 0);
  // mlx C++ surfaces `[max]` / `[min]` "Cannot reduce zero size array …"
  // via the boundary handler; the typed-prefix parser maps the bracketed
  // op-name to `MlxOpKind::Pool` (the reduction-family bucket).
  let r_max = mlxrs::ops::reduction::max_axes(&a, &[], false);
  assert!(
    matches!(
      &r_max,
      Err(mlxrs::Error::MlxOp(p)) if matches!(p.op(), mlxrs::error::MlxOpKind::Pool)
    ),
    "expected Err(MlxOp(Pool)) for max_axes(zero_size, &[]), got {r_max:?}",
  );
  let r_min = mlxrs::ops::reduction::min_axes(&a, &[], false);
  assert!(
    matches!(
      &r_min,
      Err(mlxrs::Error::MlxOp(p)) if matches!(p.op(), mlxrs::error::MlxOpKind::Pool)
    ),
    "expected Err(MlxOp(Pool)) for min_axes(zero_size, &[]), got {r_min:?}",
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
  let a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
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

// ───────── median ─────────

#[test]
fn median_odd_count_is_middle_element() {
  // Odd element count: sorted [1, 2, 3], midpoint index 3/2 = 1 → element 2.
  // No averaging (flat_size % 2 != 0), so the result is exactly the middle.
  let a = Array::from_slice(&[3.0_f32, 1.0, 2.0], &(3,)).unwrap();
  let mut r = a.median(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 2.0);
}

#[test]
fn median_even_count_averages_two_midpoints() {
  // Even element count: sorted [1, 2, 3, 4], mp = 4/2 = 2; result =
  // (sorted[mp-1] + sorted[mp]) * 0.5 = (2 + 3) * 0.5 = 2.5. A "lower
  // midpoint" (numpy `lower` interpolation) impl would wrongly return 2.0.
  let a = Array::from_slice(&[4.0_f32, 1.0, 3.0, 2.0], &(4,)).unwrap();
  let mut r = a.median(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 2.5);
}

#[test]
fn median_promotes_int_input_to_float() {
  // `median` runs on `at_least_float(dtype)`; an i32 input promotes to f32
  // and the even-count midpoint average can be fractional (2 elements → 1.5),
  // which an identity-dtype (int-preserving) path could not represent.
  let a = Array::from_slice(&[1_i32, 2], &(2,)).unwrap();
  assert_eq!(a.dtype().unwrap(), mlxrs::Dtype::I32);
  let mut r = a.median(false).unwrap();
  assert_eq!(
    r.dtype().unwrap(),
    mlxrs::Dtype::F32,
    "median promotes int to f32"
  );
  assert_eq!(r.item::<f32>().unwrap(), 1.5);
}

#[test]
fn median_axes_over_axis1_with_keepdims() {
  // [[1, 2, 3, 4], [10, 20, 30, 40]] median over axis 1 (even count each row):
  // row0 → (2 + 3) * 0.5 = 2.5, row1 → (20 + 30) * 0.5 = 25.0.
  // keepdims=true retains the reduced axis as length 1 → shape [2, 1].
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &(2, 4)).unwrap();
  let r = a.median_axes(&[1], true).unwrap();
  assert_eq!(r.shape(), vec![2, 1]);
  let mut r = r;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5, 25.0]);
}

#[test]
fn median_axes_over_axis0_no_keepdims() {
  // [[1, 5], [3, 9], [2, 7]] median over axis 0 (odd count, 3 rows):
  // col0 sorted [1, 2, 3] → 2, col1 sorted [5, 7, 9] → 7. Drops the axis.
  let a = Array::from_slice(&[1.0_f32, 5.0, 3.0, 9.0, 2.0, 7.0], &(3, 2)).unwrap();
  let r = a.median_axes(&[0], false).unwrap();
  assert_eq!(r.shape(), vec![2]);
  // median transposes the reduce axis to the back, so the result is strided;
  // materialize it row-contiguous before to_vec (mlxrs to_vec requires
  // row-contiguous storage and otherwise returns Error::NonContiguous).
  let mut r = mlxrs::ops::shape::contiguous(&r, false).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 7.0]);
}

#[test]
fn median_axes_empty_is_rejected() {
  // Unlike mean/var (which promote-identity over empty axes), mlx core median
  // cannot reduce over zero axes: it transposes the reduce axes to the back and
  // flattens them, and an empty reduce set is a degenerate flatten that mlx
  // throws on. The binding rejects empty axes up front with a typed error
  // rather than surfacing the cryptic internal flatten message.
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  match a.median_axes(&[], false) {
    Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(p.context(), "median_axes: axes"),
    other => panic!("expected EmptyInput for empty axes, got {other:?}"),
  }
}

#[test]
fn median_scalar_rank0_is_identity() {
  // A rank-0 scalar's only axis list is empty, but mlx median special-cases
  // ndim == 0 (flatten reshapes to length 1), so median(scalar) is valid and
  // returns the scalar itself. Regression: the empty-axes guard must NOT reject
  // rank-0 (it only rejects an explicit empty axis list on rank >= 1 arrays).
  let a = Array::from_slice::<f32>(&[5.0], &[0i32; 0]).unwrap();
  assert_eq!(a.ndim(), 0);
  let mut r = a.median(false).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 5.0);
}

#[test]
fn median_scalar_rank0_promotes_int() {
  // Rank-0 int scalar median promotes to f32 (at_least_float), like every other
  // median path, rather than being rejected as empty axes.
  let a = Array::from_slice::<i32>(&[7], &[0i32; 0]).unwrap();
  assert_eq!(a.ndim(), 0);
  let mut r = a.median(false).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::F32);
  assert_eq!(r.item::<f32>().unwrap(), 7.0);
}

// ───────── free-fn parity sanity ─────────

#[test]
fn mean_freefn_parity_with_method() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let mut method = a.mean(false).unwrap();
  let mut freefn = mlxrs::ops::reduction::mean(&a, false).unwrap();
  assert_eq!(method.item::<f32>().unwrap(), freefn.item::<f32>().unwrap());
}
