//! Phase 4 Branch B — happy-path tests for indexing ops.

use mlxrs::{Array, ops};

#[test]
fn take_flat_indices() {
  // a = [10, 20, 30, 40, 50]; take indices [0, 2, 4] -> [10, 30, 50].
  let a = Array::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[0, 2, 4], &[3i32]).unwrap();
  let mut t = a.take(&idx).unwrap();
  assert_eq!(t.shape(), vec![3]);
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![10.0, 30.0, 50.0]);
}

#[test]
fn take_axis_picks_rows() {
  // (3, 4) matrix; take_axis(axis=0, indices=[0, 2]) -> (2, 4) rows 0 & 2.
  let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
  let a = Array::from_slice::<f32>(&data, &(3usize, 4)).unwrap();
  let idx = Array::from_slice::<i32>(&[0, 2], &[2i32]).unwrap();
  let mut t = a.take_axis(&idx, 0).unwrap();
  assert_eq!(t.shape(), vec![2, 4]);
  // Row 0: [0,1,2,3]; row 2: [8,9,10,11].
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]
  );
}

#[test]
fn take_along_axis_picks_per_row() {
  // 2x3 input; pick a single column per row via a 2x1 indices array.
  // a = [[1, 2, 3], [4, 5, 6]]; idx = [[2], [0]] -> [[3], [4]].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let idx = Array::from_slice::<i32>(&[2, 0], &(2, 1)).unwrap();
  let mut t = a.take_along_axis(&idx, 1).unwrap();
  assert_eq!(t.shape(), vec![2, 1]);
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn gather_single_axis_slice_sizes_one() {
  // 1-D source; gather([1, 3]) along axis 0 with slice_sizes=[1] -> 1-element
  // slices, output shape (2, 1).
  let a = Array::from_slice::<f32>(&[100.0, 200.0, 300.0, 400.0, 500.0], &[5i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[1, 3], &[2i32]).unwrap();
  let mut g = ops::indexing::gather(&a, &[&idx], &[0], &[1]).unwrap();
  // Output shape: indices.shape ++ slice_sizes = [2] ++ [1] = [2, 1].
  assert_eq!(g.shape(), vec![2, 1]);
  assert_eq!(g.to_vec::<f32>().unwrap(), vec![200.0, 400.0]);
}

#[test]
fn gather_rejects_empty_indices() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &[2i32]).unwrap();
  let r = ops::indexing::gather(&a, &[], &[], &[1]);
  // Production still emits the deprecated free-form `Error::ShapeMismatch`
  // for this guard (typed-variant migration pending); assert the message
  // content so the test fails when production is migrated AND the typed
  // variant is reached.
  match r {
    Err(mlxrs::Error::ShapeMismatch(msg)) => {
      assert_eq!(msg, "gather: indices slice is empty");
    }
    other => panic!("expected ShapeMismatch for empty indices, got {other:?}"),
  }
}

#[test]
fn gather_rejects_indices_axes_length_mismatch() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  // Only one indices array, but two axes -> rejected before FFI.
  let r = ops::indexing::gather(&a, &[&idx], &[0, 1], &[1, 1]);
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::LengthMismatch(ref p))
        if p.context() == "gather: indices.len() vs axes.len()"
          && p.expected() == 2
          && p.actual() == 1
    ),
    "expected LengthMismatch; got {r:?}"
  );
}

#[test]
fn gather_rejects_negative_slice_size() {
  // `slice_sizes` is a shape extent — negative values must be rejected before
  // they can reach mlx::core::Shape construction (Codex review).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  let r = ops::indexing::gather(&a, &[&idx], &[0], &[-1]);
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert_eq!(p.context(), "shape::validate_dims: dim");
      assert_eq!(p.requirement(), "must be non-negative");
      assert_eq!(p.value(), "dim[0]=-1");
    }
    other => panic!("expected OutOfRange (shape::validate_dims), got {other:?}"),
  }
}

#[test]
fn gather_rejects_slice_sizes_rank_mismatch() {
  // slice_sizes.len() must equal a.ndim().
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  // a.ndim() == 2 but slice_sizes is rank-1 -> rejected before FFI.
  let r = ops::indexing::gather(&a, &[&idx], &[0], &[1]);
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::LengthMismatch(ref p))
        if p.context() == "gather: slice_sizes.len() vs a.ndim()"
          && p.expected() == 2
          && p.actual() == 1
    ),
    "expected LengthMismatch; got {r:?}"
  );
}

#[test]
fn put_along_axis_method_form() {
  // Inverse of take_along_axis: scatter 9.0 into `a` at per-row column
  // indices [[2],[0]] along axis 1 → [[1,2,9],[9,5,6]].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let idx = Array::from_slice::<i32>(&[2, 0], &(2, 1)).unwrap();
  let vals = Array::from_slice::<f32>(&[9.0, 9.0], &(2, 1)).unwrap();
  let mut t = a.put_along_axis(&idx, &vals, 1).unwrap();
  assert_eq!(t.shape(), vec![2, 3]);
  assert_eq!(
    t.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 9.0, 9.0, 5.0, 6.0]
  );
}

#[test]
fn scatter_overwrites_block() {
  // The mlx `scatter` doc example #1 (verbatim): scatter the 1x2 row vector
  // [1, 2] into a 4x4 zero matrix at axis-0 location 2. `updates` shape is
  // indices.shape (=[1]) ++ a 1x2 block to write (=[1, 2]) -> [1, 1, 2].
  let a = Array::from_slice::<f32>(&[0.0; 16], &(4, 4)).unwrap();
  let idx = Array::from_slice::<i32>(&[2], &[1i32]).unwrap();
  let updates = Array::from_slice::<f32>(&[1.0, 2.0], &[1i32, 1, 2]).unwrap();
  let mut g = ops::indexing::scatter(&a, &[&idx], &updates, &[0]).unwrap();
  assert_eq!(g.shape(), vec![4, 4]);
  // Row 2 becomes [1, 2, 0, 0]; every other entry stays 0.
  #[rustfmt::skip]
  let expected = vec![
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
  ];
  assert_eq!(g.to_vec::<f32>().unwrap(), expected);
}

#[test]
fn scatter_add_single_accumulates() {
  // Single-axis convenience form. 1-D source [10,20,30,40,50]; add a scalar
  // 100 at positions 1 and 3 (axis 0). `updates` shape is indices.shape (=[2])
  // ++ a scalar block (=[1]) -> [2, 1]. Existing values are accumulated, so
  // -> [10, 120, 30, 140, 50].
  let a = Array::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[1, 3], &[2i32]).unwrap();
  let updates = Array::from_slice::<f32>(&[100.0, 100.0], &[2i32, 1]).unwrap();
  let mut r = ops::indexing::scatter_add_single(&a, &idx, &updates, 0).unwrap();
  assert_eq!(r.shape(), vec![5]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![10.0, 120.0, 30.0, 140.0, 50.0]
  );
}

#[test]
fn slice_update_overwrites_region() {
  // Overwrite the single element at [0:1, 1:2] of a 2x3 matrix with 99.
  // a = [[1,2,3],[4,5,6]] -> [[1,99,3],[4,5,6]].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let upd = Array::from_slice::<f32>(&[99.0], &(1, 1)).unwrap();
  let mut r = a.slice_update(&upd, &[0, 1], &[1, 2], &[1, 1]).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 99.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn slice_update_add_accumulates_into_row() {
  // Add [10,10,10] into row 1 (region [1:2, 0:3]) of a 2x3 matrix.
  // a = [[1,2,3],[4,5,6]] -> [[1,2,3],[14,15,16]].
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let upd = Array::from_slice::<f32>(&[10.0, 10.0, 10.0], &(1, 3)).unwrap();
  let mut r = a.slice_update_add(&upd, &[1, 0], &[2, 3], &[1, 1]).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 14.0, 15.0, 16.0]
  );
}

#[test]
fn scatter_rejects_indices_axes_length_mismatch() {
  // Mirror of gather_rejects_indices_axes_length_mismatch: one indices array
  // but two axes -> rejected before FFI with a typed LengthMismatch.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  let upd = Array::from_slice::<f32>(&[9.0], &[1i32, 1, 1]).unwrap();
  let r = ops::indexing::scatter(&a, &[&idx], &upd, &[0, 1]);
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::LengthMismatch(ref p))
        if p.context() == "scatter: indices.len() vs axes.len()"
          && p.expected() == 2
          && p.actual() == 1
    ),
    "expected LengthMismatch; got {r:?}"
  );
}

#[test]
fn scatter_rejects_too_many_index_arrays() {
  // core scatter caps indices.size() <= a.ndim(); the wrapper enforces it before
  // building the index vector. `a` is 1-D (ndim 1) but two index arrays are
  // passed (lengths match axes, so the equality check passes first).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  let upd = Array::from_slice::<f32>(&[9.0], &[1i32, 1]).unwrap();
  let r = ops::indexing::scatter(&a, &[&idx, &idx], &upd, &[0, 0]);
  match r {
    Err(mlxrs::Error::CapExceeded(p)) => {
      assert_eq!(p.context(), "scatter: number of index arrays");
      assert_eq!(p.cap(), 1);
      assert_eq!(p.observed(), 2);
    }
    other => panic!("expected CapExceeded for too many index arrays, got {other:?}"),
  }
}

#[test]
fn slice_update_rejects_zero_stride() {
  // mlx normalize_slice divides by each stride, so a zero stride is division by
  // zero; the wrapper rejects it with a typed error before the FFI call.
  let src = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let upd = Array::from_slice::<f32>(&[0.0], &[1i32, 1]).unwrap();
  let r = ops::indexing::slice_update(&src, &upd, &[0, 0], &[1, 1], &[1, 0]);
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert_eq!(p.context(), "slice-update: stride");
      assert_eq!(p.value(), "0");
    }
    other => panic!("expected OutOfRange for zero stride, got {other:?}"),
  }
}

#[test]
fn slice_update_rejects_overflowing_stride() {
  // mlx normalize_slice computes `axis_size + stride - 1` in int32, so a stride
  // whose magnitude pushes that past i32::MAX overflows before the FFI returns.
  // A normal negative stride (reverse slice) is fine; only out-of-range
  // magnitudes are rejected. Covers i32::MIN (unrepresentable magnitude) and a
  // large positive stride; both route through the same magnitude guard.
  let src = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let upd = Array::from_slice::<f32>(&[0.0], &[1i32, 1]).unwrap();
  for bad in [i32::MIN, i32::MAX] {
    match ops::indexing::slice_update(&src, &upd, &[0, 0], &[1, 1], &[1, bad]) {
      Err(mlxrs::Error::OutOfRange(p)) => assert_eq!(p.context(), "slice-update: stride"),
      other => panic!("expected OutOfRange for overflowing stride {bad}, got {other:?}"),
    }
  }
}

#[test]
fn slice_update_rejects_i32_min_stride_on_empty_axis() {
  // Regression (Codex R4): for a zero-length axis the magnitude bound
  // `0 + abs(i32::MIN) - 1` equals i32::MAX exactly (not greater), so i32::MIN
  // is NOT caught by the magnitude check — it must be rejected by the explicit
  // negation guard, since mlx normalize_slice computes `-stride` (UB at INT_MIN)
  // regardless of axis size.
  let src = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  let upd = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  match ops::indexing::slice_update(&src, &upd, &[0], &[0], &[i32::MIN]) {
    Err(mlxrs::Error::OutOfRange(p)) => assert_eq!(p.context(), "slice-update: stride"),
    other => panic!("expected OutOfRange for i32::MIN stride on empty axis, got {other:?}"),
  }
}

#[test]
fn scatter_rejects_bool_index() {
  // mlx core scatter throws an uncaught (non-std::exception) C++ exception for
  // bool indices; the wrapper rejects them before FFI with a typed error on
  // BOTH the multi-axis and single-axis paths.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let bool_idx = Array::from_slice::<bool>(&[true], &[1i32]).unwrap();
  let upd = Array::from_slice::<f32>(&[9.0], &[1i32, 1]).unwrap();
  match ops::indexing::scatter(&a, &[&bool_idx], &upd, &[0]) {
    Err(mlxrs::Error::InvariantViolation(p)) => assert_eq!(p.context(), "scatter: index dtype"),
    other => panic!("expected InvariantViolation for bool index (multi), got {other:?}"),
  }
  match ops::indexing::scatter_axis(&a, &bool_idx, &upd, 0) {
    Err(mlxrs::Error::InvariantViolation(p)) => assert_eq!(p.context(), "scatter: index dtype"),
    other => panic!("expected InvariantViolation for bool index (single), got {other:?}"),
  }
}
