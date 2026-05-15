//! Logical ops: element-wise `logical_and`/`logical_or`/`logical_not` + `select` (where).
//!
//! `select(cond, x, y)` mirrors numpy.where / mlx.core.where: it picks `x[i]`
//! where `cond[i]` is true, else `y[i]`. Renamed from `where` because that is
//! a Rust keyword.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Element-wise logical AND: `out[i] = a[i] && b[i]` (with broadcasting).
/// Inputs are interpreted as truthy/falsy; output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logical_and.html).
pub fn logical_and(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_logical_and(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise logical OR: `out[i] = a[i] || b[i]` (with broadcasting).
/// Inputs are interpreted as truthy/falsy; output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logical_or.html).
pub fn logical_or(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_logical_or(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise logical NOT: `out[i] = !a[i]`.
/// Input is interpreted as truthy/falsy; output dtype is Bool.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logical_not.html).
pub fn logical_not(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_logical_not(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise selection: `out[i] = if condition[i] { x[i] } else { y[i] }`
/// (with broadcasting). Wraps `mlx.core.where` — renamed because `where` is a
/// Rust keyword.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.where.html).
pub fn select(condition: &Array, x: &Array, y: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_where(&mut out.0, condition.0, x.0, y.0, default_stream()) })?;
  Ok(out)
}
