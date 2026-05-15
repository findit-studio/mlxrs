//! Indexing ops: slice (Phase 3.5 template — start/stop/strides), take/take_along_axis/gather/scatter fill in Phase 4.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Slice `a` with NumPy-style `start`/`stop`/`strides` per dimension.
///
/// CANONICAL INDEXING TEMPLATE — pattern: 3 parallel slices passed as
/// (ptr, len) triples to mlx-c.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html).
pub fn slice(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_slice(
      &mut out.0,
      a.0,
      start.as_ptr() as *const c_int,
      start.len(),
      stop.as_ptr() as *const c_int,
      stop.len(),
      strides.as_ptr() as *const c_int,
      strides.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}
