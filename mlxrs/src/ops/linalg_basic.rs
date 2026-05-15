//! Basic linalg ops: addmm (Phase 3.5 trinary+scalar template), matmul, inner, outer.

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

/// Matrix multiplication: `a @ b`. Generalizes to batched matmul (last two
/// dims of each input are the matmul dims; leading dims broadcast).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.matmul.html).
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_matmul(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Ordinary inner product of two 1-D arrays. For higher-rank inputs, mlx
/// contracts over the last axis of each (matching numpy `inner`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.inner.html).
pub fn inner(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_inner(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Outer product of two 1-D arrays. Higher-rank inputs are flattened first
/// (matching numpy `outer`).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.outer.html).
pub fn outer(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_outer(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}
