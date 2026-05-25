//! Bulk eval and async-eval over slices of [`crate::Array`].
//!
//! Mirrors `mlx-swift`'s `eval(...)` / `asyncEval(...)`
//! ([`Transforms+Eval.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms%2BEval.swift))
//! and `mlx.core.eval` / `mlx.core.async_eval`. Single-array eval is available
//! as the inherent [`crate::Array::eval`]; this module is for the n-array case
//! that mlx-c handles via `mlx_eval(mlx_vector_array)`.

use crate::{
  Array,
  error::{Result, check, ensure_handler_installed},
  stream::assert_streams_not_cleared,
  transforms::closure::vector_array_from_borrow,
};

/// Force evaluation of every array in `arrays`. Blocks until all results are
/// materialized.
///
/// `arrays` is borrowed (`&[&Array]`); the wrapper builds a temporary
/// `mlx_vector_array` of refcount-shared handles for the eval call and frees
/// it on return. Errors surface via [`crate::Error::Backend`].
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// let a = mlxrs::Array::ones::<f32>(&(2, 2))?;
/// let b = mlxrs::Array::ones::<f32>(&(2, 2))?;
/// mlxrs::transforms::eval(&[&a, &b])?;
/// # Ok(()) }
/// ```
pub fn eval(arrays: &[&Array]) -> Result<()> {
  ensure_handler_installed();
  assert_streams_not_cleared();
  if arrays.is_empty() {
    return Ok(());
  }
  let guard = vector_array_from_borrow(arrays)?;
  // SAFETY: `guard.0` is a freshly built `mlx_vector_array` of borrowed handles
  // live for this call; mlx-c iterates the vector and evals each, surfacing
  // any backend error via `check()`. The guard frees the vector on return.
  check(unsafe { mlxrs_sys::mlx_eval(guard.0) })
}

/// Asynchronously enqueue evaluation of every array in `arrays`. Returns
/// immediately; the results materialize in the background on the array's
/// stream. To synchronize on completion, follow with [`eval`] (or the inherent
/// [`crate::Array::eval`]) on any of the same arrays, which will block until
/// the async work finishes.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// let mut a = mlxrs::Array::ones::<f32>(&(2, 2))?;
/// let b = mlxrs::Array::ones::<f32>(&(2, 2))?;
/// mlxrs::transforms::async_eval(&[&a, &b])?;
/// // ... unrelated work ...
/// a.eval()?; // syncs
/// # Ok(()) }
/// ```
pub fn async_eval(arrays: &[&Array]) -> Result<()> {
  ensure_handler_installed();
  assert_streams_not_cleared();
  if arrays.is_empty() {
    return Ok(());
  }
  let guard = vector_array_from_borrow(arrays)?;
  // SAFETY: same contract as `eval` above; the call is non-blocking but
  // identical from a memory-safety standpoint — the vector lives for the call,
  // mlx-c iterates and enqueues, errors surface via `check()`.
  check(unsafe { mlxrs_sys::mlx_async_eval(guard.0) })
}
