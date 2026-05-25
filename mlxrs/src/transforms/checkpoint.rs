//! Gradient checkpointing: [`checkpoint`].
//!
//! Mirrors `mlx.core.checkpoint` (Python) /
//! [`mlx-swift`](https://github.com/ml-explore/mlx-swift)'s checkpointing
//! helpers, and the `mlx.nn.utils.checkpoint` recipe. Wraps a function so
//! its intermediate activations are *recomputed* during the backward pass
//! instead of being stored, trading compute for memory — useful when peak
//! memory dominates training cost (e.g. long-sequence transformers).
//!
//! ## Semantics
//!
//! - Forward pass: identical to the unwrapped function. Returns the same
//!   `Vec<Array>`.
//! - Backward pass (when differentiated via [`super::grad`] /
//!   [`super::value_and_grad`] / [`super::vjp`]): mlx re-traces the forward
//!   function to reconstruct activations on demand, rather than holding them
//!   live from the forward pass. Mathematically equivalent gradient; lower
//!   peak memory; ~2x forward compute over the wrapped region.
//!
//! ## Re-entrancy
//!
//! Like [`super::custom_vjp`], the underlying `mlx_closure` is built once at
//! construction time (held in `Rc` so the returned `Fn` can call it
//! repeatedly). The wrapped `mlx_closure` returned by `mlx_checkpoint` is
//! also built once and cached.

use std::rc::Rc;

use crate::{
  Array,
  error::{Result, check, check_vector_array_handle, ensure_handler_installed},
  stream::assert_streams_not_cleared,
  transforms::closure::{
    BoxedFn, Closure, RawClosureGuard, VectorArrayGuard, drain_vector, vector_array_from_slice,
  },
};

/// Wrap `f` so its activations are recomputed (not stored) during backward.
/// Forward pass is identical to invoking `f` directly.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::{checkpoint, grad}};
/// // Wrap a function in `checkpoint` — forward identical, backward
/// // recomputes the activations.
/// let cf = checkpoint(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]))?;
/// let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
/// let mut vals = cf(&[x.try_clone()?])?;
/// assert_eq!(vals[0].item::<f32>()?, 9.0);
///
/// // Gradient through the checkpointed function is identical to the
/// // non-checkpointed gradient (same math, different memory profile).
/// let g = grad(cf, &[0])?;
/// let mut grads = g(&[x])?;
/// assert_eq!(grads[0].item::<f32>()?, 6.0);
/// # Ok(()) }
/// ```
pub fn checkpoint<F>(f: F) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  // Hold the user `f` in an `Rc` so we can re-wrap on each invocation.
  let f: Rc<BoxedFn> = Rc::new(Box::new(f));
  Ok(move |inputs: &[Array]| -> Result<Vec<Array>> {
    let f = Rc::clone(&f);
    let fwd = Closure::new(move |xs: &[Array]| f(xs))?;
    // SAFETY: `mlx_closure_new()` returns the documented `{ctx: NULL}`
    // sentinel slot (see mlx-c `private/closure.h::mlx_closure_new_()`);
    // `mlx_checkpoint` internally calls `mlx_closure_set_(*res, …)` which
    // (on the NULL-ctx path) ALLOCATES a fresh `std::function<…>` and writes
    // the pointer into `wrapped.ctx`. No guard yet — a guard built on the
    // NULL copy would never see the post-set allocated ctx and the wrapped
    // closure would leak. We RAII-wrap AFTER the populating call.
    let mut wrapped = unsafe { mlxrs_sys::mlx_closure_new() };
    // SAFETY: `fwd.as_raw()` is a valid borrowed handle live for this call;
    // out-param populated by mlx-c via `_set_` (leaves `wrapped.ctx` non-null
    // on success); backend rc surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_checkpoint(&mut wrapped, fwd.as_raw()) })?;
    // Wrap the now-populated `wrapped` in RAII so the apply step's failure
    // path frees the allocation.
    let wrapped_guard = RawClosureGuard(wrapped);

    // Apply.
    assert_streams_not_cleared();
    let in_guard = vector_array_from_slice(inputs)?;
    // SAFETY: out-param fresh empty vector_array; wrapped in RAII before
    // populating call.
    let mut out = unsafe { mlxrs_sys::mlx_vector_array_new() };
    check_vector_array_handle(out)?;
    let _out_guard = VectorArrayGuard(out);
    // SAFETY: `wrapped_guard.0` is the populated checkpoint closure (live for
    // the call); `in_guard.0` is the input vector (live); out-param populated.
    check(unsafe { mlxrs_sys::mlx_closure_apply(&mut out, wrapped_guard.0, in_guard.0) })?;
    drain_vector(out)
  })
}
