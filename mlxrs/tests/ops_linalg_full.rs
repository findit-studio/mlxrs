//! M2 piece D — happy-path tests for extended linalg ops (factorizations,
//! solvers, eigendecompositions, norms, cross product).

use std::ffi::CString;

use mlxrs::{Array, Dtype, ops::linalg_full};

const TOL: f32 = 1e-3;

fn close(a: f32, b: f32) -> bool {
  (a - b).abs() <= TOL
}

fn close_vec(got: &[f32], want: &[f32]) -> bool {
  got.len() == want.len() && got.iter().zip(want.iter()).all(|(a, b)| close(*a, *b))
}

// ─────────────────── inverses ───────────────────

#[test]
fn inv_of_identity_is_identity() {
  let a = Array::eye::<f32>(3).unwrap();
  let mut i = linalg_full::inv(&a).unwrap();
  let v = i.to_vec::<f32>().unwrap();
  let want = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
  assert!(close_vec(&v, &want), "inv(I) got {v:?} want {want:?}");
}

#[test]
fn tri_inv_of_identity_is_identity() {
  let a = Array::eye::<f32>(3).unwrap();
  let mut i = linalg_full::tri_inv(&a, true).unwrap();
  let v = i.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn pinv_of_identity_is_identity() {
  let a = Array::eye::<f32>(3).unwrap();
  let mut p = linalg_full::pinv(&a).unwrap();
  let v = p.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn cholesky_inv_of_identity_works() {
  let a = Array::eye::<f32>(2).unwrap();
  let i = linalg_full::cholesky_inv(&a, false).unwrap();
  assert_eq!(i.shape(), vec![2, 2]);
}

// ─────────────────── factorizations ───────────────────

#[test]
fn cholesky_of_identity_is_identity() {
  let a = Array::eye::<f32>(3).unwrap();
  let mut c = linalg_full::cholesky(&a, false).unwrap();
  let v = c.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn qr_of_identity_yields_identity_pair() {
  let a = Array::eye::<f32>(3).unwrap();
  let (mut q, mut r) = linalg_full::qr(&a).unwrap();
  assert_eq!(q.shape(), vec![3, 3]);
  assert_eq!(r.shape(), vec![3, 3]);
  // Q for I is +/- I; R is +/- I. Diagonal magnitude should be 1.
  let qv = q.to_vec::<f32>().unwrap();
  let rv = r.to_vec::<f32>().unwrap();
  assert!(close(qv[0].abs(), 1.0));
  assert!(close(rv[0].abs(), 1.0));
}

#[test]
fn svd_of_identity_yields_singular_values_one() {
  let a = Array::eye::<f32>(3).unwrap();
  // compute_uv = true → mlx returns [U, S, Vt].
  let mut parts = linalg_full::svd(&a, true).unwrap();
  assert_eq!(parts.len(), 3);
  // S is the second element; for I it should be all ones, length 3.
  let s = &mut parts[1];
  assert_eq!(s.shape(), vec![3]);
  let sv = s.to_vec::<f32>().unwrap();
  for x in sv {
    assert!(
      close(x, 1.0),
      "svd singular value of I should be 1, got {x}"
    );
  }
}

#[test]
fn svd_no_uv_yields_singular_values_only() {
  let a = Array::eye::<f32>(3).unwrap();
  let parts = linalg_full::svd(&a, false).unwrap();
  // compute_uv = false → just [S].
  assert_eq!(parts.len(), 1);
  assert_eq!(parts[0].shape(), vec![3]);
}

#[test]
fn lu_of_identity_yields_three_factors() {
  let a = Array::eye::<f32>(3).unwrap();
  let parts = linalg_full::lu(&a).unwrap();
  // mlx LU returns [P, L, U] (3 arrays).
  assert_eq!(parts.len(), 3);
}

#[test]
fn lu_factor_yields_two_outputs() {
  let a = Array::eye::<f32>(3).unwrap();
  let (lu, piv) = linalg_full::lu_factor(&a).unwrap();
  assert_eq!(lu.shape(), vec![3, 3]);
  // pivots are a 1-D vector.
  assert_eq!(piv.shape().len(), 1);
}

// ─────────────────── solvers ───────────────────

#[test]
fn solve_identity_yields_b() {
  let a = Array::eye::<f32>(3).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut x = linalg_full::solve(&a, &b).unwrap();
  let v = x.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn solve_triangular_identity_yields_b() {
  let a = Array::eye::<f32>(3).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut x = linalg_full::solve_triangular(&a, &b, false).unwrap();
  let v = x.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

// ─────────────────── eigendecompositions ───────────────────

#[test]
fn eigh_of_diagonal_matrix_yields_diagonal_eigenvalues() {
  // diag(1, 2, 3) — eigh should produce eigenvalues sorted ascending: [1, 2, 3].
  let data = [1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
  let a = Array::from_slice::<f32>(&data, &(3, 3)).unwrap();
  let uplo = CString::new("L").unwrap();
  let (mut vals, vecs) = linalg_full::eigh(&a, &uplo).unwrap();
  assert_eq!(vals.shape(), vec![3]);
  assert_eq!(vecs.shape(), vec![3, 3]);
  let mut v = vals.to_vec::<f32>().unwrap();
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn eigvalsh_of_diagonal_matrix_yields_diagonal() {
  let data = [1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
  let a = Array::from_slice::<f32>(&data, &(3, 3)).unwrap();
  let uplo = CString::new("L").unwrap();
  let mut vals = linalg_full::eigvalsh(&a, &uplo).unwrap();
  assert_eq!(vals.shape(), vec![3]);
  let mut v = vals.to_vec::<f32>().unwrap();
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn eig_yields_complex_pair() {
  let a = Array::eye::<f32>(2).unwrap();
  let (vals, vecs) = linalg_full::eig(&a).unwrap();
  assert_eq!(vals.shape(), vec![2]);
  assert_eq!(vecs.shape(), vec![2, 2]);
  assert_eq!(vals.dtype().unwrap(), Dtype::Complex64);
}

#[test]
fn eigvals_yields_complex() {
  let a = Array::eye::<f32>(2).unwrap();
  let vals = linalg_full::eigvals(&a).unwrap();
  assert_eq!(vals.shape(), vec![2]);
  assert_eq!(vals.dtype().unwrap(), Dtype::Complex64);
}

// ─────────────────── norms ───────────────────

#[test]
fn norm_l2_of_3_4_yields_5() {
  // ||[3, 4]||_2 = 5.
  let a = Array::from_slice::<f32>(&[3.0, 4.0], &[2i32]).unwrap();
  let mut n = linalg_full::norm_l2(&a, &[0], false).unwrap();
  assert!(close(n.item::<f32>().unwrap(), 5.0));
}

#[test]
fn norm_p2_matches_l2() {
  let a = Array::from_slice::<f32>(&[3.0, 4.0], &[2i32]).unwrap();
  let mut n = linalg_full::norm(&a, 2.0, &[0], false).unwrap();
  assert!(close(n.item::<f32>().unwrap(), 5.0));
}

#[test]
fn norm_matrix_fro_matches_l2() {
  // ||I_2||_F = sqrt(2).
  let a = Array::eye::<f32>(2).unwrap();
  let ord = CString::new("fro").unwrap();
  let mut n = linalg_full::norm_matrix(&a, &ord, &[0, 1], false).unwrap();
  assert!(close(n.item::<f32>().unwrap(), 2.0_f32.sqrt()));
}

// ─────────────────── cross ───────────────────

#[test]
fn cross_x_y_yields_z() {
  // [1,0,0] x [0,1,0] = [0,0,1].
  let a = Array::from_slice::<f32>(&[1.0, 0.0, 0.0], &[3i32]).unwrap();
  let b = Array::from_slice::<f32>(&[0.0, 1.0, 0.0], &[3i32]).unwrap();
  let mut c = linalg_full::cross(&a, &b, -1).unwrap();
  let v = c.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[0.0, 0.0, 1.0]));
}

// ─────────────────── method-form bridges ───────────────────

#[test]
fn inv_method_form_matches_freefn() {
  let a = Array::eye::<f32>(3).unwrap();
  let r = a.inv().unwrap();
  assert_eq!(r.shape(), vec![3, 3]);
}

#[test]
fn solve_method_form_matches_freefn() {
  let a = Array::eye::<f32>(3).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut x = a.solve(&b).unwrap();
  let v = x.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn cross_method_form_matches_freefn() {
  let a = Array::from_slice::<f32>(&[1.0, 0.0, 0.0], &[3i32]).unwrap();
  let b = Array::from_slice::<f32>(&[0.0, 1.0, 0.0], &[3i32]).unwrap();
  let mut c = a.cross(&b, -1).unwrap();
  let v = c.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[0.0, 0.0, 1.0]));
}
