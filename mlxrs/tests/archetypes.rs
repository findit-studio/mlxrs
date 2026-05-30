//! Smoke tests for the Phase 3.5 archetype templates.
//! Each test exercises one canonical wrapping pattern.

use mlxrs::{Array, ops};

#[test]
fn sum_2x2_ones_yields_4() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let mut s = a.sum(false).unwrap();
  assert_eq!(s.item::<f32>().unwrap(), 4.0);
}

#[test]
fn slice_arange_first_three() {
  let a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
  let mut s = a.slice(&[0], &[3], &[1]).unwrap();
  assert_eq!(s.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn concatenate_two_2x2_along_axis0() {
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let b = Array::ones::<f32>(&(2, 2)).unwrap();
  let c = ops::shape::concatenate(&[&a, &b], 0).unwrap();
  assert_eq!(c.shape(), vec![4, 2]);
}

#[test]
fn addmm_2x2_alpha1_beta0_yields_2() {
  // alpha * (a @ b) + beta * c with a=ones, b=ones, c=zeros, alpha=1, beta=0
  // = (2x2 ones) @ (2x2 ones) = 2x2 of value 2
  let a = Array::ones::<f32>(&(2, 2)).unwrap();
  let b = Array::ones::<f32>(&(2, 2)).unwrap();
  let c = Array::zeros::<f32>(&(2, 2)).unwrap();
  let mut r = ops::linalg_basic::addmm(&c, &a, &b, 1.0, 0.0).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn argmax_arange_5_yields_4() {
  // argmax over [0, 1, 2, 3, 4] should return 4. mlx returns U32 for index outputs.
  let a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
  let mut r = a.argmax(None, false).unwrap();
  assert_eq!(r.item::<u32>().unwrap(), 4);
}
