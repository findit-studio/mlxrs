//! Branch D linalg_basic happy-path tests: matmul, inner, outer.
//! addmm is exercised in tests/archetypes.rs.

use mlxrs::{Array, ops};

#[test]
fn matmul_2x3_times_3x2_via_freefn_yields_3() {
  // (2,3) ones @ (3,2) ones = (2,2) of value 3.
  let a = Array::ones::<f32>(&(2, 3)).unwrap();
  let b = Array::ones::<f32>(&(3, 2)).unwrap();
  let mut r = ops::linalg_basic::matmul(&a, &b).unwrap();
  assert_eq!(r.shape(), vec![2, 2]);
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn matmul_method_form_matches_freefn() {
  let a = Array::ones::<f32>(&(2, 3)).unwrap();
  let b = Array::ones::<f32>(&(3, 4)).unwrap();
  let mut r = a.matmul(&b).unwrap();
  assert_eq!(r.shape(), vec![2, 4]);
  // Each cell is sum_{k=0..3} 1*1 = 3.
  let v = r.to_vec::<f32>().unwrap();
  assert!(v.iter().all(|&x| x == 3.0));
}

#[test]
fn inner_1d_arange_yields_dot_product() {
  // a = [0, 1, 2, 3], b = [0, 1, 2, 3] → inner = 0+1+4+9 = 14
  let a = Array::arange(0.0, 4.0, 1.0).unwrap();
  let b = Array::arange(0.0, 4.0, 1.0).unwrap();
  let mut r = ops::linalg_basic::inner(&a, &b).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 14.0);
}

#[test]
fn inner_method_form_matches_freefn() {
  let a = Array::ones::<f32>(&[3]).unwrap();
  let b = Array::ones::<f32>(&[3]).unwrap();
  let mut r = a.inner(&b).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 3.0);
}

#[test]
fn outer_1d_arange_yields_outer_product_matrix() {
  // a = [1, 2, 3], b = [4, 5] → outer is 3×2 with rows [4,5], [8,10], [12,15]
  let a = Array::arange(1.0, 4.0, 1.0).unwrap();
  let b = Array::arange(4.0, 6.0, 1.0).unwrap();
  let mut r = ops::linalg_basic::outer(&a, &b).unwrap();
  assert_eq!(r.shape(), vec![3, 2]);
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]
  );
}

#[test]
fn outer_method_form_matches_freefn() {
  let a = Array::ones::<f32>(&[2]).unwrap();
  let b = Array::ones::<f32>(&[3]).unwrap();
  let mut r = a.outer(&b).unwrap();
  assert_eq!(r.shape(), vec![2, 3]);
  assert!(r.to_vec::<f32>().unwrap().iter().all(|&x| x == 1.0));
}
