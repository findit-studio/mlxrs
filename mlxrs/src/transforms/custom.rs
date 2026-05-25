//! User-defined VJP overrides: [`custom_vjp`] and [`custom_function`].
//!
//! `mlx.core.custom_function` / `mlx.core.custom_vjp` let a user wrap a
//! forward function with a hand-written backward (cotangent → cotangent)
//! function, overriding mlx's autograd-derived VJP. Useful for ops whose
//! autograd transcription is numerically unstable or absent.
//!
//! `custom_function` (the more general form) also accepts user-defined JVP
//! and vmap rules; we expose `fun_vjp` only and pass `NULL` for the other
//! two slots (matching the most-common Python use case). Callers that need
//! custom JVP / vmap can extend this module in a follow-up.
//!
//! The Rust signature mirrors mlx-c's `mlx_closure_custom`:
//! `(primals, cotangents, outputs) -> grads` — the same positional order
//! `mlx::core::CustomTransforms::vjp` uses when invoking its `vjp_fun_`
//! callback (`mlx/primitives.cpp::CustomTransforms::vjp`). The crate-private
//! `closure::BoxedFn3` alias documents the precise heap layout used to box
//! the user callable for the FFI trampoline.

use std::rc::Rc;

use crate::{
  Array,
  error::{Result, check, check_vector_array_handle, ensure_handler_installed},
  stream::assert_streams_not_cleared,
  transforms::closure::{
    BoxedFn, BoxedFn3, Closure, ClosureCustomGuard, RawClosureGuard, VectorArrayGuard,
    closure_custom_new, drain_vector, vector_array_from_slice,
  },
};

/// Wrap a forward function `f` with a user-defined VJP `vjp_fn`, returning a
/// new callable whose autograd-derived gradient is overridden by `vjp_fn`.
///
/// - `f: Fn(&[Array]) -> Result<Vec<Array>>` — the forward pass.
/// - `vjp_fn: Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>>` —
///   `(primals, cotangents, outputs) → grads`. Must return one gradient
///   per primal in the same shape / dtype. The triple order matches MLX
///   core's `CustomTransforms::vjp` callback signature in
///   `mlx/primitives.cpp` upstream.
///
/// The returned closure has the same signature as `f`. Differentiating it
/// (via [`super::grad`] / [`super::value_and_grad`] / [`super::vjp`]) routes
/// through `vjp_fn` instead of autograd.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::{custom_vjp, grad}};
/// // f(x) = x * x; custom VJP returns a constant 42 instead of 2x.
/// let f = custom_vjp(
///   |xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]),
///   |_primals, _cot, _outputs| {
///     Ok(vec![Array::full::<f32>(&[0i32; 0], 42.0)?])
///   },
/// )?;
/// let g = grad(f, &[0])?;
/// let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
/// let mut grads = g(&[x])?;
/// assert_eq!(grads[0].item::<f32>()?, 42.0);
/// # Ok(()) }
/// ```
pub fn custom_vjp<F, V>(f: F, vjp_fn: V) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
  V: Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  // Shared state: the user `f` is held in an `Rc<BoxedFn>` so each invocation
  // of the returned closure can re-box a fresh `mlx_closure` around it.
  let f: Rc<BoxedFn> = Rc::new(Box::new(f));
  // The custom-VJP function is bound *once* into the returned closure — we
  // build a single `mlx_closure_custom` at construction time and share it
  // across all invocations via `Rc`. mlx-c clones the closure-by-value on
  // each `mlx_custom_vjp` call, so the `mlx_closure_custom` survives the
  // sequence of construction → many wraps → drop-of-our-Rc.
  let vjp_closure: Rc<ClosureCustomGuard> = Rc::new(closure_custom_new(vjp_fn)?);
  Ok(move |inputs: &[Array]| -> Result<Vec<Array>> {
    let f = Rc::clone(&f);
    let vjp_closure = Rc::clone(&vjp_closure);
    // Build a per-call forward closure (re-using `f`).
    let fwd = Closure::new(move |xs: &[Array]| f(xs))?;
    // SAFETY: `mlx_closure_new()` returns a documented `{ctx: NULL}` sentinel
    // (see mlx-c `private/closure.h::mlx_closure_new_()`); `mlx_custom_vjp`
    // calls `mlx_closure_set_(*res, …)` which ALLOCATES a fresh
    // `std::function<…>` on the NULL-ctx path and writes the pointer into
    // `wrapped.ctx`. No guard yet — a guard on the NULL copy would never see
    // the post-set allocated ctx and the wrapped closure would leak.
    let mut wrapped = unsafe { mlxrs_sys::mlx_closure_new() };
    // SAFETY: `fwd.as_raw()` is a valid borrowed handle live for this call;
    // `vjp_closure.0` is an Rc-owned `mlx_closure_custom` live for the
    // call (Rc drops only after this scope returns); out-param's `_set_`
    // semantics described above leave `wrapped.ctx` non-null on success.
    check(unsafe { mlxrs_sys::mlx_custom_vjp(&mut wrapped, fwd.as_raw(), vjp_closure.0) })?;
    // Wrap the now-populated `wrapped` in RAII so the apply step's failure
    // path frees the allocation.
    let wrapped_guard = RawClosureGuard(wrapped);

    // Now apply the wrapped closure to `inputs`.
    let in_guard = vector_array_from_slice(inputs)?;
    // SAFETY: out-param is a fresh empty vector_array; wrapped in RAII before
    // the populating call.
    let mut out = unsafe { mlxrs_sys::mlx_vector_array_new() };
    check_vector_array_handle(out)?;
    let _out_guard = VectorArrayGuard(out);
    // SAFETY: all `mlx_*` handle args valid for the call; out-param populated.
    check(unsafe { mlxrs_sys::mlx_closure_apply(&mut out, wrapped_guard.0, in_guard.0) })?;
    drain_vector(out)
  })
}

/// Generic form: wrap `f` with optional custom VJP / JVP / vmap. This crate
/// currently exposes only the `vjp` slot (matching the `custom_vjp` API);
/// JVP and vmap pass `NULL` (mlx falls back to its autograd-derived rules
/// for those transforms).
///
/// Use [`custom_vjp`] directly when only the VJP override is needed; use
/// `custom_function` if you want the same wrapping under mlx's
/// `mlx_custom_function` entry point (semantically equivalent for the
/// VJP-only case, but routed through the more general API for future
/// extension to JVP/vmap).
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::{custom_function, grad}};
/// let f = custom_function(
///   |xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]),
///   |_primals, _cot, _outputs| {
///     Ok(vec![Array::full::<f32>(&[0i32; 0], 7.0)?])
///   },
/// )?;
/// let g = grad(f, &[0])?;
/// let x = Array::full::<f32>(&[0i32; 0], 2.0)?;
/// assert_eq!(g(&[x])?[0].item::<f32>()?, 7.0);
/// # Ok(()) }
/// ```
pub fn custom_function<F, V>(f: F, vjp_fn: V) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
  V: Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  let f: Rc<BoxedFn> = Rc::new(Box::new(f));
  let vjp_closure: Rc<ClosureCustomGuard> = Rc::new(closure_custom_new(vjp_fn)?);
  Ok(move |inputs: &[Array]| -> Result<Vec<Array>> {
    let f = Rc::clone(&f);
    let vjp_closure = Rc::clone(&vjp_closure);
    let fwd = Closure::new(move |xs: &[Array]| f(xs))?;
    // SAFETY: `mlx_closure_new()` returns a `{ctx: NULL}` sentinel slot (see
    // mlx-c `private/closure.h`); `mlx_custom_function` calls
    // `mlx_closure_set_(*res, …)` which allocates a fresh `std::function`
    // on NULL ctx and writes the pointer into `wrapped.ctx`. No guard yet —
    // a guard on the NULL copy would never see the post-set allocated ctx.
    let mut wrapped = unsafe { mlxrs_sys::mlx_closure_new() };
    let null_jvp = mlxrs_sys::mlx_closure_custom_jvp {
      ctx: std::ptr::null_mut(),
    };
    let null_vmap = mlxrs_sys::mlx_closure_custom_vmap {
      ctx: std::ptr::null_mut(),
    };
    // SAFETY: `fwd.as_raw()` is a valid borrowed handle; `vjp_closure.0` is
    // an Rc-owned `mlx_closure_custom` live for the call; the two NULL-ctx
    // `mlx_closure_custom_jvp` / `mlx_closure_custom_vmap` args are mlx-c's
    // documented null sentinel ("may be null" in `transforms.h`, branched on
    // `.ctx ? std::make_optional(...) : std::nullopt` in mlx-c
    // `transforms.cpp::mlx_custom_function`); out-param populated by mlx-c
    // via `_set_` (leaves `wrapped.ctx` non-null on success).
    check(unsafe {
      mlxrs_sys::mlx_custom_function(
        &mut wrapped,
        fwd.as_raw(),
        vjp_closure.0,
        null_jvp,
        null_vmap,
      )
    })?;
    // Wrap the now-populated `wrapped` in RAII so the apply step's failure
    // path frees the allocation.
    let wrapped_guard = RawClosureGuard(wrapped);

    let in_guard = vector_array_from_slice(inputs)?;
    // SAFETY: out-param fresh empty handle wrapped in RAII before populating call.
    let mut out = unsafe { mlxrs_sys::mlx_vector_array_new() };
    check_vector_array_handle(out)?;
    let _out_guard = VectorArrayGuard(out);
    // SAFETY: handle args valid for call; out-param written by mlx-c.
    check(unsafe { mlxrs_sys::mlx_closure_apply(&mut out, wrapped_guard.0, in_guard.0) })?;
    drain_vector(out)
  })
}

// Suppress an "unused import" warning when only one of the two custom paths
// uses `BoxedFn3` directly — it's referenced in the type system through
// `closure_custom_new`'s `where`-clause, but `cargo` may flag it.
#[allow(dead_code)]
type _BoxedFn3 = BoxedFn3;
// Stream-cleared guard is invoked transitively through the apply path; keep
// the import explicit so audits can trace it.
#[allow(dead_code)]
fn _streams_guard() {
  assert_streams_not_cleared();
}
