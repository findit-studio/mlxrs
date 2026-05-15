//! Shape ops: reshape (Phase 3.5 archetype #3 — IntoShape pattern) and
//! concatenate (Phase 3.5 archetype #4 — variadic input). Phase 4 fills in
//! transpose/expand_dims/squeeze/stack/etc.

use crate::{
  array::Array,
  error::{Error, Result, check},
  shape::{IntoShape, dim_ptr, validate_dims},
  stream::default_stream,
};

/// Reshape `a` to a new shape. Errors on incompatible total element count
/// (the C++ side validates).
///
/// CANONICAL SHAPE ARCHETYPE — the `IntoShape::with_shape` callback pattern
/// used by every shape-taking op. Every reshape/expand_dims/squeeze/etc.
/// follows this exact shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html).
pub fn reshape(a: &Array, shape: &impl IntoShape) -> Result<Array> {
  shape.with_shape(|s| {
    validate_dims(s)?;
    let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
    check(unsafe {
      mlxrs_sys::mlx_reshape(&mut out.0, a.0, dim_ptr(s), s.len(), default_stream())
    })?;
    Ok(out)
  })
}

/// Concatenate `arrays` along `axis`.
///
/// CANONICAL VARIADIC-INPUT TEMPLATE — pattern: build an `mlx_vector_array`
/// on the C side from a Rust slice, RAII-wrap for cleanup. Every fn taking
/// `Vec<Array>` (stack, meshgrid, broadcast_arrays) follows this shape.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.concatenate.html).
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
  // Concatenating zero arrays has no defined result shape — reject before
  // FFI rather than constructing an empty vector_array (which would also
  // hand mlx-c a Rust dangling pointer for `Vec::as_ptr()` on an empty Vec).
  if arrays.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "concatenate: arrays slice is empty".into(),
    });
  }
  // Install the error handler before the first fallible FFI call. Without
  // this, mlx_vector_array_new_data could fail and trigger mlx-c's default
  // printf+exit handler before default_stream() (the usual install site)
  // is reached. Codex PR #5 finding 3.
  crate::error::ensure_handler_installed();

  // Build a contiguous Vec<mlx_array> (mlx_array is Copy) and pass to
  // mlx_vector_array_new_data. RAII-free the vector_array via guard.
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
  let _vec_guard = VectorArrayGuard(vec);

  // Drain the captured backend message immediately if vector construction
  // failed — passing a NULL vec into mlx_concatenate_axis would discard the
  // original error and surface a less useful "null vector" failure instead.
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
  check(unsafe { mlxrs_sys::mlx_concatenate_axis(&mut out.0, vec, axis, default_stream()) })?;
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
