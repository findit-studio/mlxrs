//! Autograd transforms: [`value_and_grad`], [`grad`], [`vjp`], [`jvp`].
//!
//! Mirrors `mlx-swift`'s `valueAndGrad` / `grad` / `vjp` / `jvp`
//! (`Transforms.swift`, `Transforms+Grad.swift`, `Transforms+Internal.swift`)
//! and `mlx.core.{value_and_grad,grad,vjp,jvp}` on the Python side.
//!
//! ## Returned closures
//!
//! [`value_and_grad`] / [`grad`] return Rust closures (`impl Fn(...) ->
//! Result<...>`) that, when invoked, build the underlying
//! `mlx_closure_value_and_grad` *per call* (matching mlx-swift's Internal
//! `buildValueAndGradient`). Each call: builds a fresh `mlx_closure` around
//! the captured `F`, calls `mlx_value_and_grad`, applies the returned
//! `mlx_closure_value_and_grad`, drains both output `mlx_vector_array`s, and
//! frees everything. The user `F` is moved into a refcounted `Rc` so the
//! returned `Fn` can call it across many invocations.
//!
//! [`vjp`] / [`jvp`] are one-shot — they take `F`, `primals`, and
//! `tangents`/`cotangents` and return `(outputs, jvp_out)` /
//! `(outputs, vjp_out)` directly. They build and tear down the underlying
//! closure each call.
//!
//! ## `argnums`
//!
//! Both `value_and_grad` and `grad` accept an `argnums: &[i32]` slice naming
//! which positional inputs to differentiate. mlx-swift defaults `argnums` to
//! `[0]` for the unary forms; we keep the explicit slice for full parity with
//! mlx-c (callers can pass `&[0]` or any subset). The slice MUST be
//! non-empty: an empty `argnums` would name no inputs to differentiate
//! against (the resulting grad-closure would be a no-op), and the
//! underlying mlx-c entry point would receive a `NULL` data pointer
//! alongside `argnums_num == 0` and construct a `std::vector<int>(NULL,
//! NULL + 0)` — which, while typically benign in practice, is strictly
//! pointer-arithmetic on a NULL pointer (technical UB under the C++
//! standard). We reject `argnums.is_empty()` at the safe-wrapper boundary
//! ([`value_and_grad`]) before reaching the FFI call so neither the
//! semantic-no-op nor the spec-UB risk can surface.

use std::rc::Rc;

use crate::{
  Array,
  error::{Error, Result, check, check_vector_array_handle, ensure_handler_installed},
  stream::assert_streams_not_cleared,
  transforms::closure::{
    BoxedFn, Closure, ClosureValueAndGradGuard, VectorArrayGuard, drain_vector,
    vector_array_from_slice,
  },
};

/// Build a closure that computes both the value and the gradient of `f` with
/// respect to the inputs named by `argnums`. The returned closure can be
/// invoked many times with different input slices.
///
/// `f`'s contract: `f(&[Array]) -> Result<Vec<Array>>`. mlx differentiates a
/// *scalar* output — `f` must return a single-element `Vec<Array>` whose sole
/// element is a scalar `Array` (or convertible to one). If `f` returns more
/// than one element, mlx interprets it the same as Python's
/// `mx.value_and_grad` (the first element is the scalar to differentiate; the
/// rest are auxiliary outputs returned alongside the value).
///
/// Returns a closure of signature
/// `Fn(&[Array]) -> Result<(Vec<Array>, Vec<Array>)>` returning
/// `(values, gradients)`.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::value_and_grad};
/// // f(x) = x * x
/// let vag = value_and_grad(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]), &[0])?;
/// let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
/// let (mut vals, mut grads) = vag(&[x])?;
/// assert_eq!(vals[0].item::<f32>()?, 9.0);
/// assert_eq!(grads[0].item::<f32>()?, 6.0);
/// # Ok(()) }
/// ```
#[allow(clippy::type_complexity)]
pub fn value_and_grad<F>(
  f: F,
  argnums: &[i32],
) -> Result<impl Fn(&[Array]) -> Result<(Vec<Array>, Vec<Array>)>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  // Empty `argnums` is rejected at the safe-wrapper boundary:
  // - Semantically it would name no inputs to differentiate against (the
  //   resulting grad-closure would be a no-op returning empty grads).
  // - Mechanically the underlying mlx-c entry point
  //   (`mlx_value_and_grad`) would receive a NULL data pointer alongside
  //   `argnums_num == 0` and build `std::vector<int>(NULL, NULL + 0)` —
  //   pointer arithmetic on NULL is technical UB under the C++ standard
  //   ([expr.add]) even when the addend is 0. Failing fast here removes
  //   both the no-op semantic and the spec-UB exposure.
  if argnums.is_empty() {
    return Err(Error::Backend {
      message: "value_and_grad: argnums must be non-empty (at least one input index to differentiate w.r.t.)".into(),
    });
  }
  // The Rust closure `F` is shared across invocations of the returned `Fn`.
  // mlx-swift rebuilds the `mlx_closure` per inner call (no payload sharing
  // across calls); we do the same. `Rc<BoxedFn>` lets each call clone-and-
  // re-box without consuming `f`.
  let f: Rc<BoxedFn> = Rc::new(Box::new(f));
  // Argnums is fixed at build time; copy it into the closure capture.
  let argnums = argnums.to_vec();
  Ok(
    move |inputs: &[Array]| -> Result<(Vec<Array>, Vec<Array>)> {
      let f = Rc::clone(&f);
      // Per-call rebuild: closure → value_and_grad → apply → drain.
      let closure = Closure::new(move |xs: &[Array]| f(xs))?;
      let vag = build_value_and_grad(&closure, &argnums)?;
      apply_value_and_grad(&vag, inputs)
    },
  )
}

/// Build a closure that computes only the gradient of `f` with respect to
/// `argnums`, discarding the forward-pass value.
///
/// Convenience wrapper over [`value_and_grad`]: invokes the same `mlx-c`
/// machinery and drops the `values` half of the tuple.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::grad};
/// let g = grad(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]), &[0])?;
/// let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
/// let mut grads = g(&[x])?;
/// assert_eq!(grads[0].item::<f32>()?, 6.0);
/// # Ok(()) }
/// ```
pub fn grad<F>(f: F, argnums: &[i32]) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  let vag = value_and_grad(f, argnums)?;
  Ok(move |inputs: &[Array]| -> Result<Vec<Array>> { Ok(vag(inputs)?.1) })
}

/// One-shot vector-Jacobian product. Computes `cotangents^T · J_f(primals)`,
/// where `J_f` is the Jacobian of `f` at `primals`.
///
/// Returns `(outputs, vjp_outputs)`:
/// - `outputs`: the forward pass `f(primals)` (same length as `f`'s outputs);
/// - `vjp_outputs`: the VJP w.r.t. each primal (same length / shapes /
///   dtypes as `primals`).
///
/// `cotangents` must match `f`'s outputs in count, shape, and dtype.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::vjp};
/// let primals = vec![Array::full::<f32>(&[0i32; 0], 3.0)?];
/// let cot = vec![Array::full::<f32>(&[0i32; 0], 1.0)?];
/// let (mut vals, mut grads) =
///   vjp(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]), &primals, &cot)?;
/// assert_eq!(vals[0].item::<f32>()?, 9.0);
/// assert_eq!(grads[0].item::<f32>()?, 6.0); // 2x at x=3 with cotangent=1
/// # Ok(()) }
/// ```
pub fn vjp<F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  assert_streams_not_cleared();
  let closure = Closure::new(f)?;
  let p_guard = vector_array_from_slice(primals)?;
  let c_guard = vector_array_from_slice(cotangents)?;
  // SAFETY: `mlx_vector_array_new()` returns a populated empty container
  // (non-null ctx on success; NULL on alloc failure caught immediately
  // below). RAII-wrapped after the null-check so the guard only ever holds
  // a valid handle.
  let mut out0 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out0)?;
  let _out0_guard = VectorArrayGuard(out0);
  // SAFETY: same as `out0` above — populated empty container, validated then
  // RAII-wrapped.
  let mut out1 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out1)?;
  let _out1_guard = VectorArrayGuard(out1);
  // SAFETY: `closure.as_raw()`, `p_guard.0`, `c_guard.0` are valid borrowed
  // handles live for this call (none retained by mlx past it); `out0`/`out1`
  // are populated empty out-params written by mlx-c's in-place
  // `mlx_vector_array_set_` semantics (same `ctx` pointer, mutated contents);
  // backend rc surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_vjp(&mut out0, &mut out1, closure.as_raw(), p_guard.0, c_guard.0)
  })?;
  let values = drain_vector(out0)?;
  let grads = drain_vector(out1)?;
  Ok((values, grads))
}

/// One-shot Jacobian-vector product. Computes `J_f(primals) · tangents`,
/// where `J_f` is the Jacobian of `f` at `primals`.
///
/// Returns `(outputs, jvp_outputs)`:
/// - `outputs`: the forward pass `f(primals)`;
/// - `jvp_outputs`: the JVP w.r.t. the outputs (same length / shapes / dtypes
///   as `outputs`).
///
/// `tangents` must match `primals` in count, shape, and dtype.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::{Array, transforms::jvp};
/// let primals = vec![Array::full::<f32>(&[0i32; 0], 3.0)?];
/// let tan = vec![Array::full::<f32>(&[0i32; 0], 1.0)?];
/// let (mut vals, mut jvp_out) =
///   jvp(|xs| Ok(vec![mlxrs::ops::arithmetic::square(&xs[0])?]), &primals, &tan)?;
/// assert_eq!(vals[0].item::<f32>()?, 9.0);
/// assert_eq!(jvp_out[0].item::<f32>()?, 6.0); // d/dx[x^2]=2x at x=3 with tangent=1
/// # Ok(()) }
/// ```
pub fn jvp<F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
  F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  assert_streams_not_cleared();
  let closure = Closure::new(f)?;
  let p_guard = vector_array_from_slice(primals)?;
  let t_guard = vector_array_from_slice(tangents)?;
  // SAFETY: see `vjp` above — populated empty container, null-checked, then
  // RAII-wrapped so the guard only ever holds a valid handle.
  let mut out0 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out0)?;
  let _out0_guard = VectorArrayGuard(out0);
  // SAFETY: same as `out0` above.
  let mut out1 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out1)?;
  let _out1_guard = VectorArrayGuard(out1);
  // SAFETY: `closure.as_raw()`, `p_guard.0`, `t_guard.0` are valid borrowed
  // handles live for this call; `out0`/`out1` are populated empty out-params
  // written by mlx-c's in-place `_set_` semantics; backend rc surfaced via
  // `check()`.
  check(unsafe {
    mlxrs_sys::mlx_jvp(&mut out0, &mut out1, closure.as_raw(), p_guard.0, t_guard.0)
  })?;
  let values = drain_vector(out0)?;
  let jvp_out = drain_vector(out1)?;
  Ok((values, jvp_out))
}

// ─────────────────────── internal helpers ───────────────────────

/// Build an `mlx_closure_value_and_grad` from a [`Closure`] + argnums slice.
///
/// `mlx_closure_value_and_grad_new()` returns a `{ctx: NULL}` sentinel slot
/// (verified against mlx-c `private/closure.h::mlx_closure_value_and_grad_new_()`),
/// and `mlx_value_and_grad` internally calls
/// `mlx_closure_value_and_grad_set_(*res, …)` which (on NULL ctx) ALLOCATES
/// a fresh `std::function<…>` and writes the pointer into `res->ctx`. The
/// safe-wrapper consequence: we must construct the RAII guard AROUND the
/// populated handle (the local stack value `vag` AFTER the populating call),
/// not around the NULL sentinel from the initial `_new()` — a guard built on
/// the NULL copy would never see the new ctx and the populated handle would
/// leak.
fn build_value_and_grad(closure: &Closure, argnums: &[i32]) -> Result<ClosureValueAndGradGuard> {
  ensure_handler_installed();
  // SAFETY: returns the documented `{ctx: NULL}` sentinel — infallible in the
  // success path (the catch arm returns the same `{nullptr}`). NO guard yet:
  // a `Drop` over a NULL ctx is a no-op, so no leak risk if the next
  // `check(…)` short-circuits — and wrapping the NULL copy in a guard now
  // would prevent us from seeing the post-set allocated ctx (which `set_`
  // writes into the LOCAL `vag` slot, not into any prior copy).
  let mut vag = unsafe { mlxrs_sys::mlx_closure_value_and_grad_new() };
  // `value_and_grad`'s public-API entry point rejects empty `argnums`
  // (see [`value_and_grad`] doc + early-return); this private helper is
  // only reachable with a non-empty slice, so `argnums.as_ptr()` is a
  // valid pointer to `argnums.len()` i32s in caller-owned storage and
  // mlx-c never performs pointer arithmetic on a NULL pointer.
  debug_assert!(
    !argnums.is_empty(),
    "build_value_and_grad: empty argnums must be rejected at value_and_grad"
  );
  let argnums_ptr = argnums.as_ptr();
  // SAFETY: `closure.as_raw()` is a valid borrowed handle (alive for the call,
  // not retained by mlx past it); `argnums_ptr` is a valid pointer to
  // `argnums.len()` i32s in the slice's backing storage (mlx-c copies them);
  // empty `argnums` is rejected at the public-API boundary so this path is
  // never reached with a NULL data pointer.
  // mlx-c's `_set_` on a NULL ctx allocates a fresh `std::function<…>` and
  // writes the pointer into `vag.ctx`, leaving `vag.ctx` non-null on success.
  check(unsafe {
    mlxrs_sys::mlx_value_and_grad(&mut vag, closure.as_raw(), argnums_ptr, argnums.len())
  })?;
  // Wrap the now-populated `vag` in the RAII guard so a later early return
  // (e.g. from the apply step) frees the allocation.
  Ok(ClosureValueAndGradGuard(vag))
}

/// Apply an `mlx_closure_value_and_grad` to inputs, draining both output
/// vectors into `(values, gradients)`.
fn apply_value_and_grad(
  vag: &ClosureValueAndGradGuard,
  inputs: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
  ensure_handler_installed();
  assert_streams_not_cleared();
  let in_guard = vector_array_from_slice(inputs)?;
  // SAFETY: `mlx_vector_array_new()` returns a populated empty container
  // (non-null ctx on success); null-checked then RAII-wrapped.
  let mut out0 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out0)?;
  let _out0_guard = VectorArrayGuard(out0);
  // SAFETY: same as `out0` above.
  let mut out1 = unsafe { mlxrs_sys::mlx_vector_array_new() };
  check_vector_array_handle(out1)?;
  let _out1_guard = VectorArrayGuard(out1);
  // SAFETY: all `mlx_*` handle args are valid borrowed handles live for the
  // call; out-params written by mlx-c via in-place `_set_` semantics; backend
  // rc surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_closure_value_and_grad_apply(&mut out0, &mut out1, vag.0, in_guard.0)
  })?;
  let values = drain_vector(out0)?;
  let grads = drain_vector(out1)?;
  Ok((values, grads))
}
