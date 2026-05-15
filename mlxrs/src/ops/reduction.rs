//! Reduction ops: sum (Phase 3.5 template), mean/max/min/prod (Phase 4 Branch A).
//!
//! Cum* (cumsum/cumprod/cummax/cummin) live in `misc.rs` per the Phase 4 LoC
//! rebalancing. `var`/`std`/`all`/`any`/`logsumexp` land in Branch B.
//!
//! Each `_axes(empty_slice, _)` short-circuits to `try_clone()` (matches numpy
//! `op(a, axis=())` and mlx-python). `keepdims` has no observable effect in
//! that case; the documented contract is intentional and Codex-reviewed.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_mean(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Maximum value along the given axes.
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_max(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}

/// Minimum value along the given axes.
///
/// Same contract as `max_axes`: zero-size inputs error, no `try_clone`
/// short-circuit. See `max_axes` doc for the rationale.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.min.html).
pub fn min_axes(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
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
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_prod(&mut out.0, a.0, keepdims, default_stream()) })?;
  Ok(out)
}
