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
  error::{EmptyInputPayload, Error, OutOfRangePayload, Result, check, check_vector_array_handle},
  ffi::{VectorArrayGuard, drain_vector},
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

/// Reject a matrix with a zero-length trailing dimension before it reaches an
/// SVD-backed mlx kernel.
///
/// mlx core's `linalg::svd` only guards `ndim < 2`. For a `>= 2`-D input whose
/// last two dims include a zero (`0×0` / `0×n` / `m×0`), `m * n == 0` and the
/// CPU kernel's `num_matrices = a.size() / (m * n)` is then an integer
/// divide-by-zero (`0 / 0`) — undefined behavior / a process crash
/// (`mlx/backend/cpu/svd.cpp`). Every SVD-backed safe wrapper (`svd`, `pinv`,
/// and — on its covariance — `random::multivariate_normal`) calls this first so
/// an empty matrix can never reach that path; it returns the recoverable
/// [`Error::EmptyInput`] instead.
///
/// `ndim < 2` is intentionally left unguarded here so mlx surfaces its own
/// precise error for that case. The check is a cheap shape inspection with no
/// `eval`, so it never enters mlx.
pub(crate) fn reject_empty_matrix(a: &Array, op: &'static str) -> Result<()> {
  let shape = a.shape();
  if shape.len() >= 2 && (shape[shape.len() - 1] == 0 || shape[shape.len() - 2] == 0) {
    return Err(Error::EmptyInput(EmptyInputPayload::new(op)));
  }
  Ok(())
}

/// Reject an empty matrix for an SVD-backed *matrix-norm* mode, where the SVD is
/// taken over two **explicitly selected** axes (not necessarily the trailing
/// two).
///
/// mlx's `matrix_norm` (`mlx/mlx/linalg.cpp`) routes the spectral orders
/// (`ord == 2.0` / `-2.0`) and the nuclear order (`ord == "nuc"`) through
/// `svd(a_matrix, false)` after `moveaxis`-ing the two reduction axes (`row_axis`,
/// `col_axis`) to the back. The CPU SVD kernel then computes
/// `num_matrices = a.size() / (m * n)`, where `m` and `n` are the sizes of those
/// two selected axes — so if **either selected axis** has length zero,
/// `m * n == 0` and that is a `0 / 0` integer divide-by-zero (UB / a process
/// crash, `mlx/backend/cpu/svd.cpp`). mlx's `svd`/`norm` only guard `ndim < 2`,
/// so this rejects the empty case first with a recoverable
/// [`Error::EmptyInput`].
///
/// `axes` are the two reduction axes (possibly negative, mlx-style). mlx's
/// `matrix_norm` normalizes a negative axis as `axis + ndim` with NO range check
/// and then `moveaxis`-es it, so an out-of-`[-ndim, ndim)` axis (e.g. `-3` on a
/// rank-2 input → `-1`) is silently accepted by `moveaxis` and can route a
/// zero-length axis into the SVD `m * n == 0` divide-by-zero. We therefore FULLY
/// validate the axis range here, rejecting an out-of-range axis with a typed
/// [`Error::OutOfRange`] BEFORE any SVD dispatch: yielding to mlx is
/// unsafe because it raises no axis error for these. Duplicate axes (both
/// resolving to the same dimension) are likewise rejected — mlx would collapse
/// them and leak an UNSELECTED zero-length dim into the SVD; a valid
/// matrix reduction is exactly two DISTINCT in-range axes. This is a cheap shape
/// inspection with no `eval`, so it never enters mlx.
fn reject_empty_matrix_axes(a: &Array, axes: [i32; 2], op: &'static str) -> Result<()> {
  let shape = a.shape();
  let ndim = shape.len();
  // Resolve and range-check BOTH axes (mlx-style `axis + a.ndim()` for
  // negatives). An out-of-range axis is rejected with a typed `OutOfRange`
  // BEFORE any SVD dispatch (mlx does not range-check matrix-norm
  // axes, so yielding to it is unsafe). Only when both axes are validly in range
  // do we fast-fail a zero-length selected axis ahead of the SVD divide-by-zero.
  let mut resolved = [0usize; 2];
  for (slot, ax) in resolved.iter_mut().zip(axes) {
    let r = if ax < 0 {
      ax as isize + ndim as isize
    } else {
      ax as isize
    };
    if r < 0 || (r as usize) >= ndim {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "linalg norm: matrix-norm reduction axis",
        "must be in range [-ndim, ndim)",
        format!("{ax}"),
      )));
    }
    *slot = r as usize;
  }
  // A matrix reduction needs two DISTINCT axes; if both resolve to the same
  // dimension (e.g. `[1, 1]` or `[1, -1]` on a rank-2 input) mlx's two `moveaxis`
  // calls collapse them and can leak an UNSELECTED zero-length dim into the
  // trailing SVD matrix → the same `m * n == 0` divide-by-zero (traced
  // through the nuclear `sum(svd(...))` path). Reject the duplicate selection
  // before the length check.
  if resolved[0] == resolved[1] {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "linalg norm: matrix-norm reduction axes",
      "the two reduction axes must be distinct (a matrix reduction needs two different axes)",
      format!("{}", resolved[0]),
    )));
  }
  if shape[resolved[0]] == 0 || shape[resolved[1]] == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(op)));
  }
  Ok(())
}

// ─────────────────────────── inverses ───────────────────────────

/// Matrix inverse (square `a`). Runs on the per-thread CPU stream
/// (`linalg_cpu_stream`) — GPU kernel not yet implemented in mlx-c.
///
/// # Singular / ill-conditioned input
///
/// mlx does **not** report a dedicated "singular matrix" error, and this thin
/// wrapper preserves that. The mlx CPU kernel (`mlx/backend/cpu/inverse.cpp`)
/// factorizes with LAPACK `getrf` then inverts with `getri`, and the outcome
/// depends on the kind of singularity:
///
/// - **Exactly singular** (a pivot is exactly zero): `getrf` returns a non-zero
///   `info`, so mlx raises a runtime error. It surfaces here (after `eval`) as
///   an [`Error::MlxC`] / [`Error::Backend`] carrying the LAPACK error code, not
///   a typed "singular" variant.
/// - **Numerically near-singular / ill-conditioned** (tiny but non-zero
///   pivots): `getrf`/`getri` succeed and the returned inverse contains huge or
///   non-finite (`±Inf` / `NaN`) entries. No error is raised — the non-finite
///   output is the only signal. This matches numpy/mlx semantics; check the
///   result with [`crate::ops::comparison::isfinite`] if you need to detect it.
///
/// Because the factorization happens lazily inside mlx, the exactly-singular
/// error only materializes when the result is evaluated (e.g. via `eval` /
/// `item` / `to_vec`), not at the `inv` call itself.
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
/// # Empty matrix (a zero-length last-two dimension)
///
/// `pinv` is computed via SVD (`mlx/mlx/linalg.cpp` `pinv` calls
/// `linalg::svd(a, true, s)`), and mlx's `pinv` only guards `ndim < 2` — so a
/// `>= 2`-D matrix with a zero-sized row or column dimension (`0×0` / `0×n` /
/// `m×0`) would forward straight to the SVD kernel's divide-by-zero (the same
/// path guarded in [`svd`]). This safe wrapper rejects it first with a
/// recoverable [`Error::EmptyInput`]. (`ndim < 2` is still delegated to mlx,
/// which raises its own precise error.)
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.pinv.html).
pub fn pinv(a: &Array) -> Result<Array> {
  // Guard the same SVD divide-by-zero as `svd`: `pinv` is SVD-backed and mlx's
  // `pinv` only checks `ndim < 2`, so an empty trailing matrix dim would reach
  // the kernel's `a.size() / (m * n)` (`0 / 0`, UB / SIGFPE). Reject before mlx.
  reject_empty_matrix(
    a,
    "pinv: input matrix has a zero-length row or column dimension",
  )?;
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
/// # Empty matrix (a zero-length last-two dimension)
///
/// A matrix with a zero-sized row or column dimension (e.g. `0×0`, `0×n`,
/// `m×0`) is rejected with a recoverable [`Error::EmptyInput`]. mlx core's
/// `linalg::svd` only guards `ndim < 2`; for a `≥ 2`-D input with an empty
/// trailing dim its CPU kernel computes `num_matrices = a.size() / (m * n)`
/// (`mlx/backend/cpu/svd.cpp`), and with `m == 0` or `n == 0` that is an
/// integer divide-by-zero (`0 / 0`) — undefined behavior / a process crash.
/// This safe wrapper fails fast instead so an empty matrix can never reach
/// that path. (`ndim < 2` is still delegated to mlx, which raises its own
/// precise error.)
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.svd.html).
pub fn svd(a: &Array, compute_uv: bool) -> Result<Vec<Array>> {
  // Guard the divide-by-zero in mlx's CPU SVD kernel: an `>= 2`-D matrix whose
  // last two dims include a zero (0×0 / 0×n / m×0) makes `m * n == 0`, and the
  // kernel's `a.size() / (m * n)` is then `0 / 0` (UB / SIGFPE). mlx only checks
  // `ndim < 2`, so we reject the empty-matrix case here before entering mlx-c.
  // `ndim < 2` is intentionally left to mlx so it surfaces its own error.
  reject_empty_matrix(
    a,
    "svd: input matrix has a zero-length row or column dimension",
  )?;
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

/// Triangle selection for symmetric / Hermitian decompositions (`eigh`,
/// `eigvalsh`). Maps to mlx-c's `const char* UPLO` parameter as either
/// `"U"` or `"L"`. The Rust enum is the idiomatic surface; upstream
/// mlx-swift / mlx-python use `String = "L"`, which would force callers
/// to construct a `CStr` in Rust — the enum closes that ergonomics gap
/// (audit issue #259).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Uplo {
  /// Upper triangle of the matrix is used.
  Upper,
  /// Lower triangle of the matrix is used. This is the upstream default.
  Lower,
}

impl Uplo {
  /// The `CStr` form mlx-c expects.
  #[inline(always)]
  pub const fn as_cstr(self) -> &'static std::ffi::CStr {
    match self {
      Uplo::Upper => c"U",
      Uplo::Lower => c"L",
    }
  }
}

impl Default for Uplo {
  /// Matches the upstream default (`UPLO = "L"`).
  fn default() -> Self {
    Uplo::Lower
  }
}

/// Eigendecomposition of a symmetric / Hermitian matrix.
/// Returns `(eigenvalues, eigenvectors)`. `uplo` is [`Uplo::Lower`] or
/// [`Uplo::Upper`] (default [`Uplo::Lower`] matches upstream).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigh.html).
pub fn eigh(a: &Array, uplo: Uplo) -> Result<(Array, Array)> {
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
      uplo.as_cstr().as_ptr(),
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

/// Eigenvalues only of a symmetric / Hermitian matrix. `uplo` is
/// [`Uplo::Lower`] or [`Uplo::Upper`] (default [`Uplo::Lower`] matches
/// upstream).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigvalsh.html).
pub fn eigvalsh(a: &Array, uplo: Uplo) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_linalg_eigvalsh(
      &mut out.0,
      a.0,
      uplo.as_cstr().as_ptr(),
      linalg_cpu_stream(),
    )
  })?;
  Ok(out)
}

// ─────────────────────────── norms ───────────────────────────

/// p-norm reduction. `ord` is the scalar order (e.g. 1.0, 2.0, f64::INFINITY);
/// `axis` selects the axes to reduce over (empty = full reduction, routed
/// through the `dim_ptr` sentinel for FFI safety).
///
/// # Empty matrix (spectral order over two axes)
///
/// The **only** SVD-backed scalar order is the spectral norm (`ord == 2.0` /
/// `-2.0`) applied as a *matrix* reduction over exactly two axes
/// (`mlx/mlx/linalg.cpp` `matrix_norm` calls `svd(a_matrix, false)`). For that
/// case, a zero-length reduction axis would forward straight to the SVD kernel's
/// `0 / 0` divide-by-zero (see [`svd`]); this wrapper rejects it first with a
/// recoverable [`Error::EmptyInput`]. All other orders (`0`, `1`, `±inf`, any
/// `p`-norm) and the 1-axis / >2-axis cases are **not** SVD-backed and are
/// passed through to mlx unchanged. (An empty `axis` slice reaches mlx as
/// `Some(empty vec)`, which mlx rejects with "too many axes" — not a full
/// reduction through this API — so it is not SVD-backed either.)
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html).
pub fn norm(a: &Array, ord: f64, axis: &[i32], keepdims: bool) -> Result<Array> {
  // mlx's `matrix_norm` routes ONLY the spectral orders (2.0 / -2.0) through
  // `svd` (`linalg.cpp`), and `matrix_norm` is selected ONLY when `axis.len() ==
  // 2`. (An empty `axis` slice reaches mlx-c as a non-null pointer with len 0 →
  // `Some(empty vec)`, NOT `nullopt`/full-reduction, so mlx rejects it with "too
  // many axes" before any SVD — it is not guarded here.) Guard exactly the
  // two-axis spectral case; every other order / axis arity is non-SVD, untouched.
  if ord == 2.0 || ord == -2.0 {
    let matrix_axes: Option<[i32; 2]> = match axis.len() {
      2 => Some([axis[0], axis[1]]),
      _ => None,
    };
    if let Some(axes) = matrix_axes {
      reject_empty_matrix_axes(
        a,
        axes,
        "norm: matrix has a zero-length axis for the SVD-backed spectral order \
         (ord = 2 / -2)",
      )?;
    }
  }
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
/// # Empty matrix (nuclear order)
///
/// `ord == "nuc"` (the nuclear norm) is the only SVD-backed string order:
/// `mlx/mlx/linalg.cpp` `matrix_norm` computes it via `svd(a_matrix, false)`. A
/// zero-length reduction axis would forward straight to the SVD kernel's `0 / 0`
/// divide-by-zero (see [`svd`]); this wrapper rejects it first with a
/// recoverable [`Error::EmptyInput`]. The Frobenius order (`"fro"` / `"f"`) is
/// computed via `l2_norm` (a plain reduction, **not** SVD) and is passed through
/// to mlx unchanged. (Nuclear reduces over exactly two axes — `axis.len() == 2`;
/// any other arity, including an empty `axis` slice, is left to mlx, which raises
/// its own "only supported for matrices" / "too many axes" error.)
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html).
pub fn norm_matrix(a: &Array, ord: &CStr, axis: &[i32], keepdims: bool) -> Result<Array> {
  // Only the nuclear order is SVD-backed (`matrix_norm` `ord == "nuc"` in mlx's
  // `linalg.cpp`); "fro"/"f" is a plain `l2_norm` reduction. Guard the empty
  // matrix against the SVD divide-by-zero ONLY for "nuc", over its two explicit
  // reduction axes (`axis.len() == 2`).
  if ord.to_bytes() == b"nuc" {
    let matrix_axes: Option<[i32; 2]> = match axis.len() {
      2 => Some([axis[0], axis[1]]),
      _ => None,
    };
    if let Some(axes) = matrix_axes {
      reject_empty_matrix_axes(
        a,
        axes,
        "norm_matrix: matrix has a zero-length axis for the SVD-backed nuclear \
         order (ord = \"nuc\")",
      )?;
    }
  }
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
