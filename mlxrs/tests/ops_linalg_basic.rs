//! Branch D linalg_basic happy-path tests: matmul, inner, outer, plus the
//! audit #259 structure ops (tensordot int/axes forms, diagonal, trace, tril,
//! triu) exercised through the public free-fn + method surfaces.
//! addmm is exercised in tests/archetypes.rs.

use mlxrs::{Array, Dtype, ops};

// [[1,2,3],[4,5,6],[7,8,9]] — shared 3x3 fixture.
fn mat3() -> Array {
  Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap()
}

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

// ─────────────────── tensordot (#259) ───────────────────

#[test]
fn tensordot_int_full_contraction_via_freefn() {
  // axis=2 fully contracts two 2x2 matrices: 1+4+9+16 = 30.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let mut c = ops::linalg_basic::tensordot(&a, &b, 2).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
}

#[test]
fn tensordot_int_method_form_matches_freefn() {
  // method form, axis=1 over 2-D operands == matmul.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
  let mut c = a.tensordot(&b, 1).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensordot_axes_via_freefn_is_matmul() {
  // Contract a-axis 1 with b-axis 0 -> matmul.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
  let mut c = ops::linalg_basic::tensordot_axes(&a, &b, &[1], &[0]).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensordot_axes_method_form_matches_freefn() {
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let mut c = a.tensordot_axes(&b, &[0, 1], &[0, 1]).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
}

#[test]
fn tensordot_axes_length_mismatch_is_err() {
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  assert!(ops::linalg_basic::tensordot_axes(&a, &b, &[0, 1], &[0]).is_err());
}

// ─────────────────── diagonal / trace (#259) ───────────────────

#[test]
fn diagonal_offsets_via_freefn() {
  // main diagonal -> [1,5,9]; offset=-1 -> [4,8]; offset=1 -> [2,6].
  let mut d0 = ops::linalg_basic::diagonal(&mat3(), 0, 0, 1).unwrap();
  assert_eq!(d0.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
  let mut dm = ops::linalg_basic::diagonal(&mat3(), -1, 0, 1).unwrap();
  assert_eq!(dm.to_vec::<f32>().unwrap(), vec![4.0, 8.0]);
  let mut dp = ops::linalg_basic::diagonal(&mat3(), 1, 0, 1).unwrap();
  assert_eq!(dp.to_vec::<f32>().unwrap(), vec![2.0, 6.0]);
}

#[test]
fn diagonal_method_form_matches_freefn() {
  let mut d = mat3().diagonal(0, -2, -1).unwrap();
  assert_eq!(d.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn trace_offsets_via_freefn() {
  // main-diagonal trace = 15; offset=1 -> 2+6 = 8; offset=-1 -> 4+8 = 12.
  let mut t0 = ops::linalg_basic::trace(&mat3(), 0, 0, 1, None).unwrap();
  assert_eq!(t0.to_vec::<f32>().unwrap(), vec![15.0]);
  let mut tp = ops::linalg_basic::trace(&mat3(), 1, 0, 1, None).unwrap();
  assert_eq!(tp.to_vec::<f32>().unwrap(), vec![8.0]);
  let mut tm = ops::linalg_basic::trace(&mat3(), -1, 0, 1, None).unwrap();
  assert_eq!(tm.to_vec::<f32>().unwrap(), vec![12.0]);
}

#[test]
fn trace_explicit_dtype_via_freefn() {
  // Integer input traced into F32: 1+4 = 5.0, output dtype is the requested F32.
  let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
  let mut t = ops::linalg_basic::trace(&a, 0, 0, 1, Some(Dtype::F32)).unwrap();
  assert_eq!(t.dtype().unwrap(), Dtype::F32);
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn trace_method_form_default_dtype_is_input() {
  // dtype=None infers the input dtype: I32 in -> I32 out.
  let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
  let mut t = a.trace(0, 0, 1, None).unwrap();
  assert_eq!(t.dtype().unwrap(), Dtype::I32);
  assert_eq!(t.to_vec::<i32>().unwrap(), vec![5]);
}

// ─────────────────── tril / triu (#259) ───────────────────

#[test]
fn tril_k_variants_via_freefn() {
  assert_eq!(
    ops::linalg_basic::tril(&mat3(), 0)
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
  );
  assert_eq!(
    ops::linalg_basic::tril(&mat3(), 1)
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  );
  assert_eq!(
    ops::linalg_basic::tril(&mat3(), -1)
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
  );
}

#[test]
fn triu_k_variants_via_freefn_and_method() {
  assert_eq!(
    ops::linalg_basic::triu(&mat3(), 0)
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
  );
  // method form, k=1.
  assert_eq!(
    mat3().triu(1).unwrap().to_vec::<f32>().unwrap(),
    vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
  );
  // k=-1 keeps the first sub-diagonal.
  assert_eq!(
    ops::linalg_basic::triu(&mat3(), -1)
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]
  );
}
