//! Reduction ops: sum (Phase 3.5 template), mean/max/min/var/std/prod fill in Phase 4.
//!
//! Cum* (cumsum/cumprod/cummax/cummin) live in `misc.rs` per the Phase 4 LoC rebalancing.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
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
