//! Arithmetic ops: add (Phase 3 template), sub/mul/div/neg/... fill in Phase 4.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Element-wise addition: `out[i] = a[i] + b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.add.html).
///
/// CANONICAL TEMPLATE — every output-pattern fn follows this shape:
///   1. Wrap mlx_array_new() in Array(...) FIRST so RAII covers failure.
///   2. Call the C fn with &mut out.0 + default_stream() trailing arg.
///   3. check(rc)? to surface backend errors.
///   4. Ok(out).
pub fn add(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_add(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}
