//! Indexing ops: slice (Phase 3.5 template — start/stop/strides), take/take_along_axis/gather/scatter fill in Phase 4.

use std::ffi::c_int;

use crate::{
  array::Array,
  error::{Error, Result, check},
  stream::default_stream,
};

/// Slice `a` with NumPy-style `start`/`stop`/`strides` per dimension.
///
/// CANONICAL INDEXING TEMPLATE — pattern: 3 parallel slices passed as
/// (ptr, len) triples to mlx-c.
///
/// All three slices must be the same length (one entry per axis of `a`) and
/// non-empty. Empty slices have no defined slicing semantics, and forwarding
/// Rust's dangling empty-slice pointer to C++ `std::vector<int>(p, p+0)` is
/// strictly UB on a singular iterator (Codex PR #5 finding 1).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html).
pub fn slice(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
  if start.is_empty() || stop.is_empty() || strides.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "slice: start/stop/strides must be non-empty (one entry per axis)".into(),
    });
  }
  if start.len() != stop.len() || start.len() != strides.len() {
    return Err(Error::ShapeMismatch {
      message: format!(
        "slice: length mismatch — start={}, stop={}, strides={}",
        start.len(),
        stop.len(),
        strides.len()
      ),
    });
  }
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
