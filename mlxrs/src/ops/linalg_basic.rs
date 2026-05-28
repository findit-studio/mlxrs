//! Basic linalg ops: addmm (Phase 3.5 trinary+scalar template), matmul, inner, outer,
//! plus the gathered/batched matmul `gather_mm` (the mixture-of-experts primitive).

use crate::{
  array::Array,
  error::{Result, check},
  ffi::opt_array,
  stream::default_stream,
};

/// `alpha * (a @ b) + beta * c` — fused matmul + scaled add.
///
/// CANONICAL TRINARY+SCALAR TEMPLATE — pattern: 3 array inputs + 2 primitive
/// scalar inputs.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.addmm.html).
pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_addmm(&mut out.0, c.0, a.0, b.0, alpha, beta, default_stream()) })?;
  Ok(out)
}

/// Matrix multiplication: `a @ b`. Generalizes to batched matmul (last two
/// dims of each input are the matmul dims; leading dims broadcast).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.matmul.html).
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_matmul(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Ordinary inner product of two 1-D arrays. For higher-rank inputs, mlx
/// contracts over the last axis of each (matching numpy `inner`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.inner.html).
pub fn inner(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_inner(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Outer product of two 1-D arrays. Higher-rank inputs are flattened first
/// (matching numpy `outer`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html).
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_outer(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Batched/gathered matmul: like [`matmul`] but selects per-batch rows of `a` /
/// `b` via optional `lhs_indices` / `rhs_indices` flat batch indices. The
/// indices contain flat indices along the **batch** dimensions of each input
/// (all but the last two dims); the last two dims of each input are still the
/// matmul dims and contract normally.
///
/// This is the dense primitive behind mixture-of-experts (MoE) `SwitchLinear`:
/// `a` is `[N, 1, K]` per-token input, `b` is `[E, K, M]` per-expert weights,
/// and `rhs_indices` is the `[N]` per-token expert assignment — the result is
/// `[N, 1, M]`, with token `i` matmul'd against expert `rhs_indices[i]`.
///
/// `sorted_indices` promises `rhs_indices` is sorted, enabling a faster
/// kernel (mlx-lm's `SwitchGLU` sets this on the `_gather_sort` path).
///
/// Mirrors python `mx.gather_mm` and swift `gatherMM`. See
/// [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html).
pub fn gather_mm(
  a: &Array,
  b: &Array,
  lhs_indices: Option<&Array>,
  rhs_indices: Option<&Array>,
  sorted_indices: bool,
) -> Result<Array> {
  let (lhs_h, _lhs_guard) = opt_array(lhs_indices);
  let (rhs_h, _rhs_guard) = opt_array(rhs_indices);
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it) — `lhs_h` / `rhs_h` are either the borrowed
  // optional handles or NULL-ctx placeholders kept alive by their guards, which
  // `mlx_gather_mm` accepts for the optional `lhs_indices` / `rhs_indices`
  // (verified in vendor mlx/c/ops.cpp:1445-1469: each `ctx ? optional : nullopt`);
  // the out-param was freshly allocated above and is written by this call;
  // the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_gather_mm(
      &mut out.0,
      a.0,
      b.0,
      lhs_h,
      rhs_h,
      sorted_indices,
      default_stream(),
    )
  })?;
  Ok(out)
}
