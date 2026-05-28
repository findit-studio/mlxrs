//! Method-form bridges for the extended linalg ops.

use std::ffi::CStr;

use crate::{array::Array, error::Result, ops::linalg_full::Uplo};

impl Array {
  /// Matrix inverse. See [`crate::ops::linalg_full::inv`].
  pub fn inv(&self) -> Result<Array> {
    crate::ops::linalg_full::inv(self)
  }

  /// Triangular inverse. See [`crate::ops::linalg_full::tri_inv`].
  pub fn tri_inv(&self, upper: bool) -> Result<Array> {
    crate::ops::linalg_full::tri_inv(self, upper)
  }

  /// Moore-Penrose pseudo-inverse. See [`crate::ops::linalg_full::pinv`].
  pub fn pinv(&self) -> Result<Array> {
    crate::ops::linalg_full::pinv(self)
  }

  /// Cholesky inverse. See [`crate::ops::linalg_full::cholesky_inv`].
  pub fn cholesky_inv(&self, upper: bool) -> Result<Array> {
    crate::ops::linalg_full::cholesky_inv(self, upper)
  }

  /// Cholesky factor. See [`crate::ops::linalg_full::cholesky`].
  pub fn cholesky(&self, upper: bool) -> Result<Array> {
    crate::ops::linalg_full::cholesky(self, upper)
  }

  /// QR decomposition. See [`crate::ops::linalg_full::qr`].
  pub fn qr(&self) -> Result<(Array, Array)> {
    crate::ops::linalg_full::qr(self)
  }

  /// Singular value decomposition. See [`crate::ops::linalg_full::svd`].
  pub fn svd(&self, compute_uv: bool) -> Result<Vec<Array>> {
    crate::ops::linalg_full::svd(self, compute_uv)
  }

  /// LU decomposition (`[P, L, U]`). See [`crate::ops::linalg_full::lu`].
  pub fn lu(&self) -> Result<Vec<Array>> {
    crate::ops::linalg_full::lu(self)
  }

  /// Pivoted LU factorization. See [`crate::ops::linalg_full::lu_factor`].
  pub fn lu_factor(&self) -> Result<(Array, Array)> {
    crate::ops::linalg_full::lu_factor(self)
  }

  /// Solve `self @ x = b` for `x`. See [`crate::ops::linalg_full::solve`].
  pub fn solve(&self, b: &Array) -> Result<Array> {
    crate::ops::linalg_full::solve(self, b)
  }

  /// Solve triangular `self @ x = b`. See [`crate::ops::linalg_full::solve_triangular`].
  pub fn solve_triangular(&self, b: &Array, upper: bool) -> Result<Array> {
    crate::ops::linalg_full::solve_triangular(self, b, upper)
  }

  /// Eigendecomposition (general). See [`crate::ops::linalg_full::eig`].
  pub fn eig(&self) -> Result<(Array, Array)> {
    crate::ops::linalg_full::eig(self)
  }

  /// Eigendecomposition (symmetric / Hermitian). See [`crate::ops::linalg_full::eigh`].
  pub fn eigh(&self, uplo: Uplo) -> Result<(Array, Array)> {
    crate::ops::linalg_full::eigh(self, uplo)
  }

  /// Eigenvalues only (general). See [`crate::ops::linalg_full::eigvals`].
  pub fn eigvals(&self) -> Result<Array> {
    crate::ops::linalg_full::eigvals(self)
  }

  /// Eigenvalues only (symmetric / Hermitian). See [`crate::ops::linalg_full::eigvalsh`].
  pub fn eigvalsh(&self, uplo: Uplo) -> Result<Array> {
    crate::ops::linalg_full::eigvalsh(self, uplo)
  }

  /// p-norm reduction. See [`crate::ops::linalg_full::norm`].
  pub fn norm(&self, ord: f64, axis: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::linalg_full::norm(self, ord, axis, keepdims)
  }

  /// String-named matrix norm. See [`crate::ops::linalg_full::norm_matrix`].
  pub fn norm_matrix(&self, ord: &CStr, axis: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::linalg_full::norm_matrix(self, ord, axis, keepdims)
  }

  /// L2 / Frobenius norm. See [`crate::ops::linalg_full::norm_l2`].
  pub fn norm_l2(&self, axis: &[i32], keepdims: bool) -> Result<Array> {
    crate::ops::linalg_full::norm_l2(self, axis, keepdims)
  }

  /// Cross product. See [`crate::ops::linalg_full::cross`].
  pub fn cross(&self, b: &Array, axis: i32) -> Result<Array> {
    crate::ops::linalg_full::cross(self, b, axis)
  }
}
