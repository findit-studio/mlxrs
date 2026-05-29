//! Reduction ops: sum (Phase 3.5 template), mean/max/min/prod (Phase 4 Branch A),
//! var/std/all/any/logsumexp (M2a long-tail).
//!
//! Cum* (cumsum/cumprod/cummax/cummin) live in `misc.rs` per the Phase 4 LoC
//! rebalancing.
//!
//! # NaN propagation
//!
//! Every floating-point reduction in this module is **NaN-propagating**, matching
//! mlx-core's CPU/Metal reduce kernels (no NaN-skipping `nanmax`/`nansum`
//! equivalents exist in mlx). If the reduced set contains any NaN, the result is
//! NaN:
//!
//! - `sum` / `prod` / `mean` / `var` / `std` / `logsumexp` combine elements with
//!   plain IEEE-754 arithmetic (`+`, `*`), so a NaN operand poisons the running
//!   accumulator and the output is NaN (`mlx/backend/cpu/reduce.cpp`'s
//!   `SumReduce`/`ProdReduce` are bare `x + y` / `x * y`).
//! - `max` / `min` explicitly test for NaN and short-circuit to NaN: the kernel
//!   does `if (simd::any(x != x)) return NAN;` and the pairwise `maximum`/
//!   `minimum` return the NaN operand if either side is NaN (so NaN dominates
//!   `+Inf`/`-Inf`, unlike a naive `>`/`<` compare which would drop it).
//! - `median` sorts and averages the midpoint(s); a NaN in the reduce set sorts
//!   to an implementation-defined position and propagates into the midpoint.
//!
//! Integer reductions have no NaN concept. `all` / `any` reduce to `bool` — a
//! non-zero NaN bit-pattern is truthy, so a NaN element counts as `true`
//! (consistent with `astype(a, bool)`).
//!
//! This is documentation of mlx-core's existing behavior; these wrappers are
//! thin forwards and do not alter it.
//!
//! Identity-dtype reductions (`sum`, `prod`) short-circuit
//! `_axes(empty_slice, _)` to `try_clone()`: MLX itself returns `a`
//! unchanged for empty axes (`if (axes.empty()) return a;` in mlx
//! `ops.cpp`), matching numpy `op(a, axis=())`. Every other reduction
//! routes empty axes through the `_axes` C entry via a `dim_ptr` sentinel
//! so MLX's own empty-axes semantics run: `mean`/`var`/`std`/`logsumexp`
//! for dtype promotion, `max`/`min` for zero-size checks, and `all`/`any`
//! because their empty-axes result is `astype(a, bool)` — a dtype change,
//! NOT a no-op (numpy `all(a, axis=())` is bool too).

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{EmptyInputPayload, Error, OutOfRangePayload, Result, check},
  shape::dim_ptr,
  stream::default_stream,
};

/// Sum elements along the given axes.
///
/// CANONICAL REDUCTION TEMPLATE — every reduction (mean, max, min, var, std,
/// prod) follows this shape; just swap the `mlx_sum_axes` symbol.
///
/// **Empty `axes` is a no-op** that returns a refcount-sharing clone of `a`
/// (matching numpy `sum(a, axis=())` and mlx-python). `keepdims` has no
/// observable effect in this case because no dimensions were reduced — the
/// result already has `a`'s shape — and is intentionally ignored. This is
/// not the same as `sum(a, keepdims)` (the full reduction over all axes);
/// callers that want full reduction must call `sum` explicitly.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sum.html).
pub fn sum_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  if axes.is_empty() {
    return a.try_clone();
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_sum_axes(
      &mut out.0,
      a.0,
      axes.as_ptr() as *const c_int,
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Sum all elements (full reduction).
pub fn sum(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_sum(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Mean along the given axes.
///
/// `mean` promotes integer inputs to f32+; mlx handles the promotion. We must
/// NOT short-circuit `axes.is_empty()` to `try_clone` like the identity-dtype
/// reductions (`sum`/`max`/`min`) do — that would preserve the input dtype,
/// producing a dtype split between the empty and non-empty paths (the empty
/// branch would return int while every other path returns float). Empty axes
/// route through `mlx_mean_axes` with a `dim_ptr` sentinel so MLX's promotion
/// runs uniformly. Codex PR #6 finding.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.mean.html).
pub fn mean_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_mean_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Mean of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.mean.html).
pub fn mean(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_mean(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Maximum value along the given axes.
///
/// NaN-propagating (floating types): if any element along the reduced axes is
/// NaN the result is NaN — and NaN dominates `±Inf` (see the module-level
/// "NaN propagation" note). Integer inputs have no NaN concept.
///
/// `max` errors on zero-size inputs (no defined max for an empty set). Unlike
/// the identity-dtype reductions (`sum`/`prod`), we must NOT short-circuit
/// `axes.is_empty()` to `try_clone` — MLX checks `a.size() == 0` BEFORE the
/// no-axes early return, so a clone here would silently accept zero-size
/// inputs that every other reduction path rejects (Codex PR #6 round 2).
/// Empty axes route through `mlx_max_axes` with a `dim_ptr` sentinel.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.max.html).
pub fn max_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_max_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Maximum of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.max.html).
pub fn max(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_max(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Minimum value along the given axes.
///
/// Same contract as `max_axes`: zero-size inputs error, no `try_clone`
/// short-circuit. See `max_axes` doc for the rationale. Also NaN-propagating
/// for floating types (NaN dominates `±Inf`; see the module-level note).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.min.html).
pub fn min_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_min_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Minimum of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.min.html).
pub fn min(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_min(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Product along the given axes. Empty `axes` is a no-op; see `sum_axes`.
///
/// `prod` of int inputs may promote to i64; mlx handles this.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.prod.html).
pub fn prod_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  if axes.is_empty() {
    return a.try_clone();
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_prod_axes(
      &mut out.0,
      a.0,
      axes.as_ptr() as *const c_int,
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Product of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.prod.html).
pub fn prod(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_prod(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Variance along the given axes. `ddof` is the delta-degrees-of-freedom
/// (numpy convention: variance divides by `n - ddof`).
///
/// `var` promotes integer inputs to f32+; routes empty axes through
/// `mlx_var_axes` with a `dim_ptr` sentinel so the promotion runs uniformly.
/// See `mean_axes` for the rationale.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.var.html).
pub fn var_axes(a: &Array, axes: &[i32], keepdims: bool, ddof: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_var_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      ddof as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Variance of all elements (full reduction). `ddof` follows numpy convention.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.var.html).
pub fn var(a: &Array, keepdims: bool, ddof: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_var(&mut out.0, a.0, keepdims, ddof as c_int, default_stream()) })?;
  Ok(out)
}

/// Standard deviation along the given axes. `ddof` is the delta-degrees-of-
/// freedom (numpy convention: divides by `n - ddof`).
///
/// `std` promotes integer inputs to f32+; routes empty axes through
/// `mlx_std_axes` with a `dim_ptr` sentinel (same rationale as `mean_axes`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.std.html).
pub fn std_axes(a: &Array, axes: &[i32], keepdims: bool, ddof: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_std_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      ddof as c_int,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Standard deviation of all elements (full reduction). `ddof` follows numpy.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.std.html).
pub fn std(a: &Array, keepdims: bool, ddof: i32) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_std(&mut out.0, a.0, keepdims, ddof as c_int, default_stream()) })?;
  Ok(out)
}

/// Logical AND along the given axes. Result dtype is always `bool`.
///
/// Empty `axes` is **not** a dtype-preserving no-op: MLX flags the
/// empty-axes case `is_noop` in `compute_reduce_shape` and returns
/// `astype(a, bool)` (numpy `all(a, axis=())` is bool too). It therefore
/// routes through `mlx_all_axes` with a `dim_ptr` sentinel — a `try_clone`
/// short-circuit (correct for the identity-dtype `sum_axes`/`prod_axes`)
/// would here wrongly return `a`'s original dtype/values.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.all.html).
pub fn all_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_all_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Logical AND of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.all.html).
pub fn all(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_all(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Logical OR along the given axes. Result dtype is always `bool`.
///
/// Empty `axes` routes through `mlx_any_axes` with a `dim_ptr` sentinel for
/// the same reason as [`all_axes`]: MLX's empty-axes result is
/// `astype(a, bool)`, not a dtype-preserving no-op.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.any.html).
pub fn any_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_any_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Logical OR of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.any.html).
pub fn any(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_any(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// `log(sum(exp(a)))` along the given axes — numerically stable LSE.
///
/// `logsumexp` promotes integer inputs to f32+; routes empty axes through
/// `mlx_logsumexp_axes` with a `dim_ptr` sentinel (same rationale as
/// `mean_axes`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html).
pub fn logsumexp_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_logsumexp_axes(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// `log(sum(exp(a)))` of all elements (full reduction).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html).
pub fn logsumexp(a: &Array, keepdims: bool) -> Result<Array> {
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_logsumexp(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Median along the given axes. Promotes integer inputs to f32+ and computes
/// the midpoint of the sorted values along the reduce axes.
///
/// Unlike `mean_axes`/`var_axes`, explicit empty `axes` on a rank >= 1 array is
/// rejected with [`Error::EmptyInput`]: mlx core `median` cannot reduce over
/// zero axes of a non-scalar — it transposes the reduce axes to the back and
/// flattens them, and an empty reduce set is a degenerate flatten that mlx
/// throws on (the numpy-style identity-promote for empty axes is not part of
/// mlx). A rank-0 scalar is the exception: its only axis list is empty and mlx
/// special-cases `ndim == 0` (flatten reshapes to length 1), so a scalar is
/// allowed through to `mlx_median` and yields its own (float-promoted) value.
///
/// The result is a thin forward of `mlx_median` and may be strided (median
/// transposes the reduce axes), so call `crate::ops::shape::contiguous` before
/// [`Array::to_vec`] to read it.
///
/// DIRECT-ARG SOUNDNESS (issue #266): beyond the empty-axes rejection, no
/// bounded guard is needed. Axis values are validated C-side (out-of-bounds
/// throws) and core median iterates them via a range-for and a `std::set`, not
/// a signed `int i` over `axes.size()`, so there is no direct-argument
/// count-overflow path (matching the merged `sum_axes` family).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.median.html).
pub fn median_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  // A rank-0 scalar's only axis list is empty, and mlx median special-cases
  // ndim == 0 (flatten reshapes to length 1), so that case is valid and must
  // delegate to mlx. Only reject an explicit empty axis list on a rank >= 1
  // array, where mlx median throws on the degenerate flatten.
  if axes.is_empty() && a.ndim() > 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "median_axes: axes",
    )));
  }
  // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
  // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
  // early return / panic frees it, then populated by the following call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
  // not retained by mlx past it); the out-param was freshly allocated above
  // and is written by this call; the backend rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_median(
      &mut out.0,
      a.0,
      dim_ptr(axes),
      axes.len(),
      keepdims,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// Median of all elements (full reduction). Convenience over [`median_axes`]
/// covering every axis (mlx's `median(a, keepdims)` overload, which expands to
/// `median` over `axes_for_median(a)` = all axes).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.median.html).
pub fn median(a: &Array, keepdims: bool) -> Result<Array> {
  // `median` has no all-reduce mlx-c entry (unlike `sum`/`mean`, which call a
  // dedicated `mlx_sum`/`mlx_mean`), so the full reduction is expressed as
  // `median_axes` over every axis. `ndim` is an array rank — mlx stores it as a
  // C++ `int`, so it always fits `c_int` — but convert through a checked cast
  // rather than `as` so building the axis range can never silently wrap.
  let ndim = c_int::try_from(a.ndim()).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "median: array rank",
      "must fit in c_int to build the full-reduction axis list",
      "exceeds i32::MAX",
    ))
  })?;
  let all_axes: Vec<c_int> = (0..ndim).collect();
  median_axes(a, &all_axes, keepdims)
}
