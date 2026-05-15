//! Basic linalg ops: addmm (Phase 3.5 template — trinary + scalar coeffs), matmul fills in Phase 4.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// `alpha * (a @ b) + beta * c` — fused matmul + scaled add.
///
/// CANONICAL TRINARY+SCALAR TEMPLATE — pattern: 3 array inputs + 2 primitive
/// scalar inputs.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.addmm.html).
pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_addmm(&mut out.0, c.0, a.0, b.0, alpha, beta, default_stream()) })?;
  Ok(out)
}
