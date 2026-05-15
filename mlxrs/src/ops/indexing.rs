//! Indexing ops: slice (Phase 3.5 template — start/stop/strides), take/take_along_axis/gather/scatter fill in Phase 4.

use crate::{
  array::Array,
  error::{Error, Result, check},
  shape::dim_ptr,
  stream::default_stream,
};

/// Slice `a` with NumPy-style `start`/`stop`/`strides` per dimension.
///
/// CANONICAL INDEXING TEMPLATE — pattern: 3 parallel slices passed as
/// (ptr, len) triples to mlx-c.
///
/// All three slices must be the same length and equal to `a.ndim()`. For a
/// 0-D scalar input that means three empty slices, which is the correct
/// no-op slice — empty inputs are routed through `dim_ptr`'s static sentinel
/// (the Codex PR #5 dangling-pointer concern), not rejected.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html).
pub fn slice(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
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
  if start.len() != a.ndim() {
    return Err(Error::ShapeMismatch {
      message: format!(
        "slice: start/stop/strides length {} != a.ndim() {}",
        start.len(),
        a.ndim()
      ),
    });
  }
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_slice(
      &mut out.0,
      a.0,
      dim_ptr(start),
      start.len(),
      dim_ptr(stop),
      stop.len(),
      dim_ptr(strides),
      strides.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}
