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
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn gather_rejects_indices_axes_length_mismatch() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  // Only one indices array, but two axes -> rejected before FFI.
  let r = ops::indexing::gather(&a, &[&idx], &[0, 1], &[1, 1]);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn gather_rejects_negative_slice_size() {
  // `slice_sizes` is a shape extent — negative values must be rejected before
  // they can reach mlx::core::Shape construction (Codex review).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  let r = ops::indexing::gather(&a, &[&idx], &[0], &[-1]);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn gather_rejects_slice_sizes_rank_mismatch() {
  // slice_sizes.len() must equal a.ndim().
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let idx = Array::from_slice::<i32>(&[0], &[1i32]).unwrap();
  // a.ndim() == 2 but slice_sizes is rank-1 -> rejected before FFI.
  let r = ops::indexing::gather(&a, &[&idx], &[0], &[1]);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}
