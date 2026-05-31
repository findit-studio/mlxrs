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
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let mut i = linalg_full::inv(&a).unwrap();
  let v = i.to_vec::<f32>().unwrap();
  let want = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
  assert!(close_vec(&v, &want), "inv(I) got {v:?} want {want:?}");
}

#[test]
fn tri_inv_of_identity_is_identity() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let mut i = linalg_full::tri_inv(&a, true).unwrap();
  let v = i.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn pinv_of_identity_is_identity() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let mut p = linalg_full::pinv(&a).unwrap();
  let v = p.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn cholesky_inv_of_identity_works() {
  let a = Array::eye::<f32>(2, None, 0).unwrap();
  let i = linalg_full::cholesky_inv(&a, false).unwrap();
  assert_eq!(i.shape(), vec![2, 2]);
}

// ─────────────────── factorizations ───────────────────

#[test]
fn cholesky_of_identity_is_identity() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let mut c = linalg_full::cholesky(&a, false).unwrap();
  let v = c.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[4], 1.0));
  assert!(close(v[8], 1.0));
}

#[test]
fn qr_of_identity_yields_identity_pair() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
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
  let a = Array::eye::<f32>(3, None, 0).unwrap();
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
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let parts = linalg_full::svd(&a, false).unwrap();
  // compute_uv = false → just [S].
  assert_eq!(parts.len(), 1);
  assert_eq!(parts[0].shape(), vec![3]);
}

#[test]
fn lu_of_identity_yields_three_factors() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let parts = linalg_full::lu(&a).unwrap();
  // mlx LU returns [P, L, U] (3 arrays).
  assert_eq!(parts.len(), 3);
}

#[test]
fn lu_factor_yields_two_outputs() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let (lu, piv) = linalg_full::lu_factor(&a).unwrap();
  assert_eq!(lu.shape(), vec![3, 3]);
  // pivots are a 1-D vector.
  assert_eq!(piv.shape().len(), 1);
}

// ─────────────────── solvers ───────────────────

#[test]
fn solve_identity_yields_b() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
  let mut x = linalg_full::solve(&a, &b).unwrap();
  let v = x.to_vec::<f32>().unwrap();
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn solve_triangular_identity_yields_b() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
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
  let (mut vals, vecs) = linalg_full::eigh(&a, linalg_full::Uplo::Lower).unwrap();
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
  let mut vals = linalg_full::eigvalsh(&a, linalg_full::Uplo::Lower).unwrap();
  assert_eq!(vals.shape(), vec![3]);
  let mut v = vals.to_vec::<f32>().unwrap();
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  assert!(close_vec(&v, &[1.0, 2.0, 3.0]));
}

#[test]
fn eig_yields_complex_pair() {
  let a = Array::eye::<f32>(2, None, 0).unwrap();
  let (vals, vecs) = linalg_full::eig(&a).unwrap();
  assert_eq!(vals.shape(), vec![2]);
  assert_eq!(vecs.shape(), vec![2, 2]);
  assert_eq!(vals.dtype().unwrap(), Dtype::Complex64);
}

#[test]
fn eigvals_yields_complex() {
  let a = Array::eye::<f32>(2, None, 0).unwrap();
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
  let a = Array::eye::<f32>(2, None, 0).unwrap();
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
  let a = Array::eye::<f32>(3, None, 0).unwrap();
  let r = a.inv().unwrap();
  assert_eq!(r.shape(), vec![3, 3]);
}

#[test]
fn solve_method_form_matches_freefn() {
  let a = Array::eye::<f32>(3, None, 0).unwrap();
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

// ─────────────────── empty-matrix SVD guard (#257) ───────────────────
//
// Both `svd` and `pinv` are SVD-backed; mlx's CPU SVD kernel computes
// `num_matrices = a.size() / (m * n)` and only guards `ndim < 2`, so a `>= 2`-D
// matrix with a zero-length trailing dim (`0×0` / `0×n` / `m×0`) makes
// `m * n == 0` and triggers a `0 / 0` integer divide-by-zero (UB / SIGFPE).
// The shared `reject_empty_matrix` guard rejects these with `Error::EmptyInput`
// via a cheap shape check, so the call returns `Err` WITHOUT entering mlx
// (no `eval` / `to_vec`).

/// Build a float matrix whose data is empty (product of `dims` is 0).
fn empty_matrix(dims: &[i32]) -> Array {
  // `from_slice` takes `shape: &impl IntoShape`; a runtime `&[i32]` slice
  // satisfies `IntoShape` only when re-borrowed (the `&[i32]` impl), so pass
  // `&dims` rather than `dims`.
  Array::from_slice::<f32>(&[], &dims).unwrap()
}

#[test]
fn svd_rejects_empty_matrix_dims() {
  for dims in [[0i32, 0], [0, 3], [3, 0]] {
    let a = empty_matrix(&dims);
    match linalg_full::svd(&a, true) {
      Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(
        p.context(),
        "svd: input matrix has a zero-length row or column dimension"
      ),
      other => panic!("expected EmptyInput for svd of {dims:?}, got {other:?}"),
    }
    // compute_uv = false must take the same guard.
    match linalg_full::svd(&a, false) {
      Err(mlxrs::Error::EmptyInput(_)) => {}
      other => panic!("expected EmptyInput for svd(no-uv) of {dims:?}, got {other:?}"),
    }
  }
}

#[test]
fn pinv_rejects_empty_matrix_dims() {
  for dims in [[0i32, 0], [0, 3], [3, 0]] {
    let a = empty_matrix(&dims);
    match linalg_full::pinv(&a) {
      Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(
        p.context(),
        "pinv: input matrix has a zero-length row or column dimension"
      ),
      other => panic!("expected EmptyInput for pinv of {dims:?}, got {other:?}"),
    }
  }
}

// ─────────────── empty-matrix guard for SVD-backed norms (#257) ───────
//
// mlx's `matrix_norm` (`mlx/mlx/linalg.cpp`) routes the *spectral* scalar orders
// (`ord == 2.0` / `-2.0`) and the *nuclear* string order (`ord == "nuc"`) through
// `svd(a_matrix, false)` over the two selected reduction axes. An empty matrix on
// those modes would reach the same `0 / 0` SVD divide-by-zero as bare `svd`/
// `pinv`, so the wrappers fast-fail with `Error::EmptyInput` (a cheap shape check,
// no `eval`). The non-SVD orders (`fro`, `1`, `inf`, p-norms, 1-axis reductions)
// are deliberately left unguarded — those tests assert our guard does NOT fire.

const SPECTRAL_NORM_CONTEXT: &str =
  "norm: matrix has a zero-length axis for the SVD-backed spectral order (ord = 2 / -2)";

const NUCLEAR_NORM_CONTEXT: &str =
  "norm_matrix: matrix has a zero-length axis for the SVD-backed nuclear order (ord = \"nuc\")";

#[test]
fn norm_spectral_rejects_empty_matrix_dims() {
  // ord = 2.0 (max singular value) and ord = -2.0 (min singular value) both
  // route through SVD in mlx; both must fast-fail on an empty matrix.
  for ord in [2.0_f64, -2.0] {
    for dims in [[0i32, 0], [0, 3], [3, 0]] {
      let a = empty_matrix(&dims);
      match linalg_full::norm(&a, ord, &[0, 1], false) {
        Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(p.context(), SPECTRAL_NORM_CONTEXT),
        other => panic!("expected EmptyInput for norm(ord={ord}) of {dims:?}, got {other:?}"),
      }
    }
  }
}

#[test]
fn norm_spectral_rejects_empty_matrix_negative_axes() {
  // Negative axes must be normalized the same way mlx does before the check.
  let a = empty_matrix(&[0, 3]);
  match linalg_full::norm(&a, 2.0, &[-2, -1], false) {
    Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(p.context(), SPECTRAL_NORM_CONTEXT),
    other => panic!("expected EmptyInput for norm(ord=2, axes=[-2,-1]), got {other:?}"),
  }
}

#[test]
fn norm_spectral_empty_axis_is_not_svd_guarded() {
  // An empty `axis` slice is passed to mlx-c as a NON-null pointer with len 0,
  // which mlx-c turns into `Some(empty vec)` (NOT `nullopt`/full-reduction); mlx
  // then rejects zero axes with "too many axes" BEFORE any SVD, so this is not an
  // SVD-backed case and our spectral guard must NOT fire. The call
  // still errors (mlx's axis error) — it just isn't OUR EmptyInput.
  for dims in [[0i32, 0], [0, 3], [3, 0]] {
    let a = empty_matrix(&dims);
    match linalg_full::norm(&a, 2.0, &[], false) {
      Err(mlxrs::Error::EmptyInput(p)) if p.context() == SPECTRAL_NORM_CONTEXT => {
        panic!(
          "norm(ord=2, axis=[]) is not SVD-backed (mlx rejects empty axis first); guard must not fire"
        )
      }
      Err(_) => {}
      Ok(_) => panic!("expected an error for norm(ord=2, axis=[]) of {dims:?}"),
    }
  }
}

#[test]
fn norm_non_spectral_ord_is_not_guarded() {
  // ord = 1 / inf are plain reductions (NOT SVD-backed); our empty-matrix guard
  // must NOT fire for them. (mlx may itself accept or reject the empty
  // reduction; we only assert this wrapper does not raise OUR spectral
  // EmptyInput.)
  let a = empty_matrix(&[0, 3]);
  for ord in [1.0_f64, f64::INFINITY] {
    match linalg_full::norm(&a, ord, &[0, 1], false) {
      Err(mlxrs::Error::EmptyInput(p)) if p.context() == SPECTRAL_NORM_CONTEXT => {
        panic!("norm(ord={ord}) must NOT hit the SVD spectral guard (non-SVD order)")
      }
      _ => {}
    }
  }
}

#[test]
fn norm_spectral_single_axis_is_not_guarded() {
  // A 1-axis reduction is a *vector* norm (NOT `matrix_norm`/SVD), so even
  // ord = 2 over a single axis must not hit our matrix guard.
  let a = empty_matrix(&[0, 3]);
  match linalg_full::norm(&a, 2.0, &[1], false) {
    Err(mlxrs::Error::EmptyInput(p)) if p.context() == SPECTRAL_NORM_CONTEXT => {
      panic!("norm(ord=2) over a single axis is a vector norm, must not hit the matrix guard")
    }
    _ => {}
  }
}

#[test]
fn norm_spectral_out_of_range_axis_is_typed_error() {
  // mlx's `matrix_norm` adds `ndim` once to a negative axis with NO
  // range check, so an out-of-`[-ndim, ndim)` axis (positive OR negative) is not
  // caught by mlx and could route a zero-length axis into the SVD divide-by-zero.
  // The guard fully validates the axis range and returns a typed `OutOfRange`
  // BEFORE any SVD — NOT our spectral `EmptyInput`, and never a crash. Checked on
  // an empty matrix where a naive zero-length-first check would have masked it.
  let a = empty_matrix(&[0, 3]);
  for axes in [&[0i32, 99][..], &[99, 0][..], &[-3, 1][..], &[0, -3][..]] {
    match linalg_full::norm(&a, 2.0, axes, false) {
      Err(mlxrs::Error::OutOfRange(_)) => {}
      other => {
        panic!("norm(ord=2, axes={axes:?}) out-of-range axis must be OutOfRange, got {other:?}")
      }
    }
  }
}

#[test]
fn norm_matrix_nuc_out_of_range_axis_is_typed_error() {
  // Same full-range axis validation for the nuclear order — an
  // out-of-`[-ndim, ndim)` axis is a typed `OutOfRange` before any SVD, not our
  // nuclear `EmptyInput` and never a crash.
  let ord = CString::new("nuc").unwrap();
  let a = empty_matrix(&[0, 3]);
  for axes in [&[0i32, 99][..], &[99, 0][..], &[-3, 1][..], &[0, -3][..]] {
    match linalg_full::norm_matrix(&a, &ord, axes, false) {
      Err(mlxrs::Error::OutOfRange(_)) => {}
      other => {
        panic!(
          "norm_matrix(nuc, axes={axes:?}) out-of-range axis must be OutOfRange, got {other:?}"
        )
      }
    }
  }
}

#[test]
fn norm_matrix_nuc_rejects_empty_matrix_dims() {
  let ord = CString::new("nuc").unwrap();
  for dims in [[0i32, 0], [0, 3], [3, 0]] {
    let a = empty_matrix(&dims);
    match linalg_full::norm_matrix(&a, &ord, &[0, 1], false) {
      Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(p.context(), NUCLEAR_NORM_CONTEXT),
      other => panic!("expected EmptyInput for norm_matrix(nuc) of {dims:?}, got {other:?}"),
    }
  }
}

#[test]
fn norm_matrix_nuc_empty_axis_is_not_svd_guarded() {
  // Empty `axis` reaches mlx as `Some(empty vec)` → mlx rejects it with "too many
  // axes" before any SVD, so the nuclear guard must NOT fire.
  let ord = CString::new("nuc").unwrap();
  let a = empty_matrix(&[3, 0]);
  match linalg_full::norm_matrix(&a, &ord, &[], false) {
    Err(mlxrs::Error::EmptyInput(p)) if p.context() == NUCLEAR_NORM_CONTEXT => {
      panic!(
        "norm_matrix(nuc, axis=[]) is not SVD-backed (mlx rejects empty axis first); guard must not fire"
      )
    }
    Err(_) => {}
    Ok(_) => panic!("expected an error for norm_matrix(nuc, axis=[])"),
  }
}

#[test]
fn norm_spectral_duplicate_axes_is_typed_error() {
  // A matrix reduction needs two DISTINCT axes. Duplicate axes (both
  // resolving to the same dim) are BOTH in range and length 3 here, so only a
  // distinctness check catches them — mlx would otherwise collapse them and leak
  // the unselected zero-length dim into the SVD divide-by-zero.
  let a = empty_matrix(&[0, 3]);
  for axes in [[1i32, 1], [1, -1]] {
    match linalg_full::norm(&a, 2.0, &axes, false) {
      Err(mlxrs::Error::OutOfRange(_)) => {}
      other => {
        panic!("norm(ord=2, axes={axes:?}) duplicate axes must be OutOfRange, got {other:?}")
      }
    }
  }
}

#[test]
fn norm_matrix_nuc_duplicate_axes_is_typed_error() {
  // The nuclear `sum(svd(...))` path is where the duplicate-axis SVD
  // divide-by-zero was traced; reject duplicate axes (each in range, length > 0)
  // with a typed error before any SVD.
  let ord = CString::new("nuc").unwrap();
  for (dims, axes) in [
    ([0i32, 3], [1i32, 1]),
    ([0, 3], [1, -1]),
    ([3, 0], [0, 0]),
    ([3, 0], [0, -2]),
  ] {
    let a = empty_matrix(&dims);
    match linalg_full::norm_matrix(&a, &ord, &axes, false) {
      Err(mlxrs::Error::OutOfRange(_)) => {}
      other => panic!(
        "norm_matrix(nuc, dims={dims:?}, axes={axes:?}) duplicate axes must be OutOfRange, got {other:?}"
      ),
    }
  }
}

#[test]
fn norm_matrix_fro_is_not_guarded() {
  // "fro"/"f" is computed via l2_norm (NOT SVD); our guard must NOT fire. (mlx
  // may itself accept the empty Frobenius reduction; we only assert this wrapper
  // does not raise OUR nuclear EmptyInput.)
  let a = empty_matrix(&[0, 3]);
  for name in ["fro", "f"] {
    let ord = CString::new(name).unwrap();
    match linalg_full::norm_matrix(&a, &ord, &[0, 1], false) {
      Err(mlxrs::Error::EmptyInput(p)) if p.context() == NUCLEAR_NORM_CONTEXT => {
        panic!("norm_matrix(\"{name}\") must NOT hit the SVD nuclear guard (l2_norm, not SVD)")
      }
      _ => {}
    }
  }
}

// ─────────────────── determinant (det / slogdet) ───────────────────

#[test]
fn det_of_identity_is_one() {
  for n in [1usize, 2, 3, 4, 5] {
    let a = Array::eye::<f32>(n, None, 0).unwrap();
    let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
    assert!(close(v, 1.0), "det(I_{n}) = {v}, want 1.0");
  }
}

#[test]
fn det_2x2_known_small_path() {
  // [[1,2],[3,4]] = 1*4 - 2*3 = -2 (closed-form small-matrix path).
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!(close(v, -2.0), "det = {v}, want -2.0");
}

#[test]
fn det_3x3_known_small_path() {
  // [[6,1,1],[4,-2,5],[2,8,7]] = -306.
  let a = Array::from_slice::<f32>(
    &[6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0],
    &(3usize, 3),
  )
  .unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!((v - (-306.0)).abs() < 0.1, "det = {v}, want -306.0");
}

#[test]
fn det_4x4_general_lu_path() {
  // diag(2,3,4,5) = 120 — exercises the LU path (n > 3), positive det.
  let a = Array::from_slice::<f32>(
    &[
      2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
    ],
    &(4usize, 4),
  )
  .unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!((v - 120.0).abs() < 0.05, "det = {v}, want 120.0");
}

#[test]
fn det_4x4_negative_lu_path() {
  // diag(2,3,4,5) with rows 0 and 1 swapped → one permutation → det = -120.
  // Independently exercises the permutation-parity sign on the LU path.
  let a = Array::from_slice::<f32>(
    &[
      0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
    ],
    &(4usize, 4),
  )
  .unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!((v - (-120.0)).abs() < 0.05, "det = {v}, want -120.0");
}

#[test]
fn det_4x4_even_permutation_lu_path() {
  // diag(2,3,4,5) row-permuted by [1,0,3,2] = (0 1)(2 3): an EVEN permutation
  // that forces TWO getrf row swaps → det = +120 (sign unchanged). Catches a
  // regression where the pivot-mismatch parity is computed as `any` (nonzero→1)
  // rather than a true count mod 2 — which would wrongly flip an even swap
  // count back to -120.
  let a = Array::from_slice::<f32>(
    &[
      0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 4.0, 0.0,
    ],
    &(4usize, 4),
  )
  .unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!(
    (v - 120.0).abs() < 0.05,
    "det = {v}, want +120.0 (even permutation)"
  );
}

#[test]
fn det_singular_is_zero() {
  // [[1,2],[2,4]] is rank-1 → det 0.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 2.0, 4.0], &(2usize, 2)).unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!(v.abs() < TOL, "singular det = {v}, want ~0");
}

#[test]
fn slogdet_reconstructs_det_on_lu_path() {
  // Diagonally-dominant SPD 4x4 (n > 3): det == sign * exp(logabsdet), sign +1.
  let a = Array::from_slice::<f32>(
    &[
      2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 5.0,
    ],
    &(4usize, 4),
  )
  .unwrap();
  let (mut sign, mut logabs) = linalg_full::slogdet(&a).unwrap();
  let s = sign.item::<f32>().unwrap();
  let la = logabs.item::<f32>().unwrap();
  let det = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!(
    close(s * la.exp(), det),
    "sign*exp(logabs) = {} want det {det}",
    s * la.exp()
  );
  assert!(close(s, 1.0), "SPD matrix sign should be +1, got {s}");
}

#[test]
fn slogdet_singular_is_sign_zero_neg_inf() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 2.0, 4.0], &(2usize, 2)).unwrap();
  let (mut sign, mut logabs) = linalg_full::slogdet(&a).unwrap();
  assert_eq!(sign.item::<f32>().unwrap(), 0.0, "singular sign must be 0");
  let la = logabs.item::<f32>().unwrap();
  assert!(
    la.is_infinite() && la < 0.0,
    "singular logabsdet must be -inf, got {la}"
  );
}

#[test]
fn det_batched() {
  // batch [2,2,2]: [[1,2],[3,4]] → -2, [[2,0],[0,3]] → 6.
  let a =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 3.0], &(2usize, 2, 2)).unwrap();
  let v = linalg_full::det(&a).unwrap().to_vec::<f32>().unwrap();
  assert!(
    close_vec(&v, &[-2.0, 6.0]),
    "batched det = {v:?}, want [-2, 6]"
  );
}

#[test]
fn det_rejects_non_square_and_low_rank() {
  let rect = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3)).unwrap();
  assert!(matches!(
    linalg_full::det(&rect),
    Err(mlxrs::Error::InvariantViolation(_))
  ));
  let v1d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap();
  assert!(matches!(
    linalg_full::det(&v1d),
    Err(mlxrs::Error::RankMismatch(_))
  ));
}

#[test]
fn det_integer_input_promotes_to_f32() {
  // diag(2,3) as i32 → promoted to f32, det 6.
  let a = Array::from_slice::<i32>(&[2, 0, 0, 3], &(2usize, 2)).unwrap();
  let v = linalg_full::det(&a).unwrap().item::<f32>().unwrap();
  assert!(close(v, 6.0), "int det = {v}, want 6.0");
}
