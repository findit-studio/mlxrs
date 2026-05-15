//! Misc ops: argmax (Phase 3.5 template — optional axis tagged union, index output),
//! plus clip/sort/top_k/cum*/etc. fill in Phase 4.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Index of the maximum value, optionally along `axis`. Output dtype is U32.
///
/// CANONICAL OPTIONAL-AXIS TEMPLATE — pattern: dispatch between `mlx_argmax`
/// (full reduction) and `mlx_argmax_axis` (per-axis) based on `Option<i32>`.
/// Every fn with optional-axis semantics (argmin, all_axis, any_axis) follows
/// this shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argmax.html).
pub fn argmax(a: &Array, axis: Option<i32>, keepdims: bool) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    match axis {
      Some(ax) => {
        mlxrs_sys::mlx_argmax_axis(&mut out.0, a.0, ax as c_int, keepdims, default_stream())
      }
      None => mlxrs_sys::mlx_argmax(&mut out.0, a.0, keepdims, default_stream()),
    }
  })?;
  Ok(out)
}
