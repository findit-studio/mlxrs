//! Phase 4 Branch B — happy-path tests for shape ops.

use std::ffi::CString;

use mlxrs::{Array, ops};

#[test]
fn transpose_2x3_swaps_to_3x2() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let t = a.transpose().unwrap();
  assert_eq!(t.shape(), vec![3, 2]);
}

#[test]
fn transpose_axes_3d_permutes() {
  // (2, 3, 4) with axes [2, 0, 1] -> (4, 2, 3)
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let t = a.transpose_axes(&[2, 0, 1]).unwrap();
  assert_eq!(t.shape(), vec![4, 2, 3]);
}

#[test]
fn transpose_axes_empty_for_scalar() {
  // 0-D scalar -> empty axes -> still scalar; exercises dim_ptr sentinel.
  let empty: [i32; 0] = [];
  let a = Array::from_slice::<f32>(&[7.0], &empty).unwrap();
  let mut t = a.transpose_axes(&[]).unwrap();
  assert_eq!(t.shape(), Vec::<usize>::new());
  assert_eq!(t.item::<f32>().unwrap(), 7.0);
}

#[test]
fn expand_dims_axes_inserts_dims() {
  let a = Array::ones::<f32>(&(3usize, 4)).unwrap();
  let e = a.expand_dims_axes(&[0, 2]).unwrap();
  // From (3, 4): insert at 0 -> (1, 3, 4); insert at 2 -> (1, 3, 1, 4).
  assert_eq!(e.shape(), vec![1, 3, 1, 4]);
}

#[test]
fn expand_dims_axes_empty_is_clone() {
  // Empty axes is a no-op identity (numpy semantics + cookbook archetype 2 rationale).
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.expand_dims_axes(&[]).unwrap();
  assert_eq!(r.shape(), a.shape());
  assert_eq!(r.to_vec::<f32>().unwrap(), a.to_vec::<f32>().unwrap());
}

#[test]
fn squeeze_axes_drops_size1() {
  // (1, 3, 1, 4) -> squeeze [0, 2] -> (3, 4).
  let a = Array::ones::<f32>(&(1usize, 3, 1, 4)).unwrap();
  let s = a.squeeze_axes(&[0, 2]).unwrap();
  assert_eq!(s.shape(), vec![3, 4]);
}

#[test]
fn squeeze_axes_empty_is_clone() {
  let mut a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.squeeze_axes(&[]).unwrap();
  assert_eq!(r.shape(), a.shape());
  assert_eq!(r.to_vec::<f32>().unwrap(), a.to_vec::<f32>().unwrap());
}

#[test]
fn broadcast_to_expands_shape() {
  // (1, 3) broadcast to (4, 3) -> shape (4, 3); content all-ones.
  let a = Array::ones::<f32>(&(1usize, 3)).unwrap();
  let b = a.broadcast_to(&(4usize, 3)).unwrap();
  assert_eq!(b.shape(), vec![4, 3]);
  assert_eq!(b.size(), 12);
}

#[test]
fn stack_two_2x2_along_axis0() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = ops::shape::stack(&[&a, &b]).unwrap();
  // Stack inserts a new axis 0: (2, 2, 2).
  assert_eq!(s.shape(), vec![2, 2, 2]);
}

#[test]
fn stack_axis_two_2x2_along_axis2() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = ops::shape::stack_axis(&[&a, &b], 2).unwrap();
  assert_eq!(s.shape(), vec![2, 2, 2]);
}

#[test]
fn stack_with_method_form() {
  let a = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let b = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let c = Array::ones::<f32>(&(2usize, 2)).unwrap();
  let s = a.stack_with(&[&b, &c], 0).unwrap();
  // 3 inputs stacked along axis 0 -> (3, 2, 2).
  assert_eq!(s.shape(), vec![3, 2, 2]);
}

#[test]
fn stack_rejects_empty_input() {
  let r = ops::shape::stack(&[]);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
  let r2 = ops::shape::stack_axis(&[], 0);
  assert!(matches!(r2, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn split_sections_at_indices_yields_three_parts() {
  // arange(0, 10) = [0,1,2,3,4,5,6,7,8,9]; split at [3, 5] -> 3 parts.
  let a = Array::arange(0.0, 10.0, 1.0).unwrap();
  let parts = a.split_sections(&[3, 5], 0).unwrap();
  assert_eq!(parts.len(), 3);
  assert_eq!(parts[0].shape(), vec![3]);
  assert_eq!(parts[1].shape(), vec![2]);
  assert_eq!(parts[2].shape(), vec![5]);
}

#[test]
fn split_sections_empty_indices_yields_single_part() {
  // Splitting at no indices = whole array as a single part. Exercises the
  // empty-slice dim_ptr sentinel.
  let a = Array::arange(0.0, 4.0, 1.0).unwrap();
  let parts = a.split_sections(&[], 0).unwrap();
  assert_eq!(parts.len(), 1);
  assert_eq!(parts[0].shape(), vec![4]);
}

#[test]
fn flatten_2x3_to_6() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut f = a.flatten(0, -1).unwrap();
  assert_eq!(f.shape(), vec![6]);
  assert_eq!(
    f.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn flatten_partial_range() {
  // (2, 3, 4) flatten dims [1, 2] -> (2, 12).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let f = a.flatten(1, 2).unwrap();
  assert_eq!(f.shape(), vec![2, 12]);
}

#[test]
fn swapaxes_swaps_axes() {
  // (2, 3, 4) swap (0, 2) -> (4, 3, 2).
  let a = Array::ones::<f32>(&(2usize, 3, 4)).unwrap();
  let s = a.swapaxes(0, 2).unwrap();
  assert_eq!(s.shape(), vec![4, 3, 2]);
}

#[test]
fn pad_constant_grows_axis() {
  // (3,) padded by 2 on the left and 1 on the right of axis 0 -> (6,).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let p = ops::shape::pad(&a, &[0], &[2], &[1], &zero, &mode).unwrap();
  assert_eq!(p.shape(), vec![6]);
}

#[test]
fn pad_rejects_length_mismatch() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[2], &[1, 2], &zero, &mode);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn pad_rejects_negative_low() {
  // `low`/`high` are shape extents — negatives must be rejected before
  // reaching mlx::core::Shape construction (Codex review).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[-1], &[1], &zero, &mode);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}

#[test]
fn pad_rejects_negative_high() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let zero = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let mode = CString::new("constant").unwrap();
  let r = ops::shape::pad(&a, &[0], &[1], &[-2], &zero, &mode);
  assert!(matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })));
}
