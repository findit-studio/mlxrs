//! Indexing ops: slice (Phase 3.5 template — start/stop/strides), plus the
//! Phase 4 Branch B fan-out: take / take_axis / take_along_axis / gather.

use std::ffi::c_int;

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

/// Take elements of `a` at the flat positions in `indices` (treating `a` as
/// a 1-D array and returning the flat-take). Output dtype matches `a`'s.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take.html).
pub fn take(a: &Array, indices: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_take(&mut out.0, a.0, indices.0, default_stream()) })?;
  Ok(out)
}

/// Take elements of `a` at `indices` along `axis`. Output dtype matches `a`'s.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take.html).
pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_take_axis(&mut out.0, a.0, indices.0, axis as c_int, default_stream())
  })?;
  Ok(out)
}

/// Take elements of `a` along `axis` using a per-position `indices` array
/// (broadcasts `indices` against the non-`axis` dims of `a`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html).
pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_take_along_axis(&mut out.0, a.0, indices.0, axis as c_int, default_stream())
  })?;
  Ok(out)
}

/// Gather slices of `a` indexed by `indices` along `axes`, with a per-axis
/// `slice_sizes`. The number of `indices` arrays must match `axes.len()`;
/// `slice_sizes.len()` must equal `a.ndim()` (one entry per dimension of `a`).
///
/// Mirrors `concatenate`'s variadic-input + handler-installed pattern. Empty
/// `indices` is rejected because `mlx_gather` requires at least one index
/// array (one per gather axis).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather.html).
pub fn gather(a: &Array, indices: &[&Array], axes: &[i32], slice_sizes: &[i32]) -> Result<Array> {
  if indices.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "gather: indices slice is empty".into(),
    });
  }
  if indices.len() != axes.len() {
    return Err(Error::ShapeMismatch {
      message: format!(
        "gather: indices.len() {} != axes.len() {}",
        indices.len(),
        axes.len()
      ),
    });
  }
  // slice_sizes is a shape extent (one per dim of `a`); it must be non-negative
  // and have rank == a.ndim(). Without these guards, negative or wrong-rank
  // values cross into mlx::core::Shape construction (Codex PR #7-target finding).
  if slice_sizes.len() != a.ndim() {
    return Err(Error::ShapeMismatch {
      message: format!(
        "gather: slice_sizes.len() {} != a.ndim() {}",
        slice_sizes.len(),
        a.ndim()
      ),
    });
  }
  crate::shape::validate_dims(slice_sizes)?;
  crate::error::ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = indices.iter().map(|a| a.0).collect();
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
  let _vec_guard = VectorArrayGuard(vec);
  if vec.ctx.is_null() {
    return Err(
      crate::error::LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend {
          message: "mlx_vector_array_new_data returned NULL".into(),
        }),
    );
  }
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe {
    mlxrs_sys::mlx_gather(
      &mut out.0,
      a.0,
      vec,
      dim_ptr(axes),
      axes.len(),
      dim_ptr(slice_sizes),
      slice_sizes.len(),
      default_stream(),
    )
  })?;
  Ok(out)
}

/// RAII guard for a temporary `mlx_vector_array`.
struct VectorArrayGuard(mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}
