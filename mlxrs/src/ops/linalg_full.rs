//! Extended linalg ops: factorizations (cholesky/lu/qr/svd), inverses
//! (inv/tri_inv/cholesky_inv/pinv), solvers (solve/solve_triangular),
//! eigendecompositions (eig/eigh/eigvals/eigvalsh), norms, and cross product.
//!
//! Basic matmul / inner / outer / addmm live in
//! [`crate::ops::linalg_basic`]; this module covers the rest of `linalg.h`.
//!
//! Multi-output ops follow two shapes from the cookbook:
//!   - Paired outputs (qr, eig, eigh, lu_factor): two `mlx_array` outparams.
//!   - Variadic outputs (lu, svd-with-uv): an `mlx_vector_array` outparam,
//!     drained into a `Vec<Array>` (see `ops::shape::split_sections`).
//!
//! Most factorization / decomposition / inverse / solver ops are NOT yet
//! supported on the Metal GPU — mlx-c routes them through `linalg::*` C++
//! kernels that hard-fail with "This op is not yet supported on the GPU.
//! Explicitly pass a CPU stream to run it." For those ops we maintain a
//! per-thread CPU stream via `linalg_cpu_stream` and route through it,
//! mirroring the per-thread GPU stream pattern in `crate::stream`.
//!
//! See [mlx linalg docs](https://ml-explore.github.io/mlx/build/html/python/linalg.html).

use std::{
  cell::Cell,
  ffi::{CStr, c_int},
};

use crate::{
  array::Array,
  error::{Result, check, check_vector_array_handle},
  shape::dim_ptr,
  stream::default_stream,
};

thread_local! {
  static CPU_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

/// Per-thread CPU stream for linalg ops that mlx-c rejects on the GPU
/// (factorizations, inverses, solvers). Pattern mirrors `stream::default_stream`:
/// lazy init on first call per thread, never freed (Metal/CPU stream teardown
/// at process exit can crash). `norm` / `cross` use the regular GPU stream.
fn linalg_cpu_stream() -> mlxrs_sys::mlx_stream {
  crate::error::ensure_handler_installed();
  // Honor the #13 cleared-thread poison contract (as `default_stream()` /
  // `Stream::default_cpu()` do): a CPU-routed op on a poisoned thread must
  // fail fast, not continue into mlx with torn-down stream state.
  crate::stream::assert_streams_not_cleared();
  CPU_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: `mlx_default_cpu_stream_new()` returns the thread's default CPU stream
    // handle; the error handler is installed first and the NULL-ctx case is
    // checked by the caller before the handle is cached/used.
    let s = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs::ops::linalg_full: mlx_default_cpu_stream_new returned NULL ctx — \
         CPU stream initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

// ─────────────────────────── inverses ───────────────────────────

/// Matrix inverse (square `a`). Runs on the per-thread CPU stream
/// (`linalg_cpu_stream`) — GPU kernel not yet implemented in mlx-c.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.inv.html).
pub fn inv(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_inv(&mut out.0, a.0, linalg_cpu_stream()) })?;
  Ok(out)
}

/// Inverse of a triangular matrix. `upper = true` for upper-triangular input,
/// `false` for lower.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.tri_inv.html).
pub fn tri_inv(a: &Array, upper: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_tri_inv(&mut out.0, a.0, upper, linalg_cpu_stream()) })?;
  Ok(out)
}

/// Moore-Penrose pseudo-inverse.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.pinv.html).
pub fn pinv(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_pinv(&mut out.0, a.0, linalg_cpu_stream()) })?;
  Ok(out)
}

/// Inverse via Cholesky factorization. `upper` selects the triangle of the
/// pre-computed Cholesky factor in `a`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky_inv.html).
pub fn cholesky_inv(a: &Array, upper: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_cholesky_inv(&mut out.0, a.0, upper, linalg_cpu_stream())
  })?;
  Ok(out)
}

// ─────────────────────────── factorizations ───────────────────────────

/// Cholesky factor of a positive-definite matrix `a`. `upper = true` returns
/// `R` such that `a = R^T @ R`; `false` returns `L` such that `a = L @ L^T`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky.html).
pub fn cholesky(a: &Array, upper: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_cholesky(&mut out.0, a.0, upper, linalg_cpu_stream()) })?;
  Ok(out)
}

/// QR decomposition. Returns `(Q, R)` such that `a = Q @ R`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.qr.html).
pub fn qr(a: &Array) -> Result<(Array, Array)> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut q = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut r = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_qr(&mut q.0, &mut r.0, a.0, linalg_cpu_stream()) })?;
  Ok((q, r))
}

/// Singular value decomposition. When `compute_uv = true`, returns
/// `[U, S, Vt]`; when `false`, returns `[S]` only.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.svd.html).
pub fn svd(a: &Array, compute_uv: bool) -> Result<Vec<Array>> {
  // Resolve the CPU stream FIRST — `linalg_cpu_stream()` runs the cleared-thread
  // poison guard (`assert_streams_not_cleared`) and installs the error handler
  // (`ensure_handler_installed`) before the fallible `mlx_vector_array_new()`
  // allocation. This is intentionally stronger than test coverage: a poisoned
  // thread must fail fast (panic) here rather than return `Err` if the
  // subsequent alloc fails under allocator pressure. No alloc-failure injection
  // hook exists, so guard order — not a test — enforces the fail-fast contract.
  let s = linalg_cpu_stream();
  // SAFETY: `mlx_vector_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; the RAII guard captures it before the
  // populating call so a partial/early-return vector is still freed.
  let mut vec_out = unsafe { mlxrs_sys::mlx_vector_array_new() };
  // `mlx_vector_array_new` is fallible: a null `ctx` means allocation failed
  // and an error sits in TLS. Validate (draining handler state) BEFORE the
  // guard so it only ever wraps a non-null handle (no leak / double-free).
  check_vector_array_handle(vec_out)?;
  let _vec_guard = VectorArrayGuard(vec_out);
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_svd(&mut vec_out, a.0, compute_uv, s) })?;
  drain_vector(vec_out)
}

/// LU decomposition. Returns the `[P, L, U]` triple as a `Vec<Array>`
/// (mlx-c packs them in a vector_array).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.lu.html).
pub fn lu(a: &Array) -> Result<Vec<Array>> {
  // See `svd`: resolve the CPU stream FIRST so the cleared-thread poison guard
  // and handler-install run before the fallible `mlx_vector_array_new()`. A
  // poisoned thread must panic here, not return `Err` under allocator pressure.
  let s = linalg_cpu_stream();
  // SAFETY: `mlx_vector_array_new()` returns a fresh empty out-param handle (NULL
  // ctx) per the mlx-c convention; the RAII guard captures it before the
  // populating call so a partial/early-return vector is still freed.
  let mut vec_out = unsafe { mlxrs_sys::mlx_vector_array_new() };
  // See `svd`: validate the fallible allocation (draining handler state)
  // before the guard so it only ever wraps a non-null handle.
  check_vector_array_handle(vec_out)?;
  let _vec_guard = VectorArrayGuard(vec_out);
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_lu(&mut vec_out, a.0, s) })?;
  drain_vector(vec_out)
}

/// Pivoted LU factorization: returns `(LU, pivots)` in mlx's compact form.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.lu_factor.html).
pub fn lu_factor(a: &Array) -> Result<(Array, Array)> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out0 = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out1 = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_lu_factor(&mut out0.0, &mut out1.0, a.0, linalg_cpu_stream())
  })?;
  Ok((out0, out1))
}

// ─────────────────────────── solvers ───────────────────────────

/// Solve `a @ x = b` for `x`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.solve.html).
pub fn solve(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_solve(&mut out.0, a.0, b.0, linalg_cpu_stream()) })?;
  Ok(out)
}

/// Solve `a @ x = b` where `a` is triangular. `upper = true` selects upper
/// triangular; `false` selects lower.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.solve_triangular.html).
pub fn solve_triangular(a: &Array, b: &Array, upper: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_solve_triangular(&mut out.0, a.0, b.0, upper, linalg_cpu_stream())
  })?;
  Ok(out)
}

// ─────────────────────────── eigendecompositions ───────────────────────────

/// Eigendecomposition of a general (not necessarily symmetric) matrix.
/// Returns `(eigenvalues, eigenvectors)`. Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eig.html).
pub fn eig(a: &Array) -> Result<(Array, Array)> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut vals = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut vecs = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_eig(&mut vals.0, &mut vecs.0, a.0, linalg_cpu_stream()) })?;
  Ok((vals, vecs))
}

/// Eigendecomposition of a symmetric / Hermitian matrix.
/// Returns `(eigenvalues, eigenvectors)`. `uplo` is `b"L\0"` or `b"U\0"`
/// indicating which triangle to use.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigh.html).
pub fn eigh(a: &Array, uplo: &CStr) -> Result<(Array, Array)> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut vals = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut vecs = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_eigh(
      &mut vals.0,
      &mut vecs.0,
      a.0,
      uplo.as_ptr(),
      linalg_cpu_stream(),
    )
  })?;
  Ok((vals, vecs))
}

/// Eigenvalues only of a general matrix. Output dtype is Complex64.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigvals.html).
pub fn eigvals(a: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_linalg_eigvals(&mut out.0, a.0, linalg_cpu_stream()) })?;
  Ok(out)
}

/// Eigenvalues only of a symmetric / Hermitian matrix.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigvalsh.html).
pub fn eigvalsh(a: &Array, uplo: &CStr) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_eigvalsh(&mut out.0, a.0, uplo.as_ptr(), linalg_cpu_stream())
  })?;
  Ok(out)
}

// ─────────────────────────── norms ───────────────────────────

/// p-norm reduction. `ord` is the scalar order (e.g. 1.0, 2.0, f64::INFINITY);
/// `axis` selects the axes to reduce over (empty = full reduction, routed
/// through the `dim_ptr` sentinel for FFI safety).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html).
pub fn norm(a: &Array, ord: f64, axis: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_norm(
      &mut out.0,
      a.0,
      ord,
      dim_ptr(axis),
      axis.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Matrix norm using a string-named order (`"fro"`, `"nuc"`, etc.).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html).
pub fn norm_matrix(a: &Array, ord: &CStr, axis: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_norm_matrix(
      &mut out.0,
      a.0,
      ord.as_ptr(),
      dim_ptr(axis),
      axis.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// L2 (Frobenius) norm — convenience wrapper for `norm(a, 2.0, axis, keepdims)`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html).
pub fn norm_l2(a: &Array, axis: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_norm_l2(
      &mut out.0,
      a.0,
      dim_ptr(axis),
      axis.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

// ─────────────────────────── cross ───────────────────────────

/// Cross product of two 3-vectors (or stacks of 3-vectors) along `axis`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cross.html).
pub fn cross(a: &Array, b: &Array, axis: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_cross(&mut out.0, a.0, b.0, axis as c_int, default_stream())
  })?;
  Ok(out)
}

// ─────────────────────────── helpers ───────────────────────────

/// Drain an `mlx_vector_array` into a `Vec<Array>`, copying out each handle.
fn drain_vector(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// RAII guard for a temporary `mlx_vector_array`.
struct VectorArrayGuard(mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Runs during `Drop` /
    // thread teardown: must not touch TLS, call `check()`, panic, or unwind
    // across `extern "C"`; the rc is discarded silently per the crate's
    // Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}
