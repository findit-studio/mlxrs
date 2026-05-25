//! Safe `mlx_closure` wrapper.
//!
//! `mlx_closure` is mlx-c's callable handle: a function-pointer + opaque
//! `void* payload` pair that the autograd / custom-VJP / checkpoint / compile
//! transforms accept as their user-supplied function argument. The trampoline
//! pattern here mirrors `mlx-swift`'s
//! [`new_mlx_closure`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Cmlx%2BUtil.swift)
//! (`Cmlx+Util.swift`) and the equivalent `pybind` shim on the Python side.
//!
//! ## Lifetime
//!
//! The Rust callable is boxed (`Box<Inner<F>>`) and `Box::into_raw`'d into a
//! stable `*mut c_void` payload pointer. A C destructor (`destroy_payload`)
//! reclaims the box via `Box::from_raw`. `mlx_closure_free` invokes the
//! destructor exactly once (mlx-c's `mlx_closure` is a shared_ptr-backed
//! handle, so dtor runs when the last reference drops, not necessarily at the
//! `mlx_closure_free` call). The [`Closure`] wrapper owns *one* reference to
//! the handle and frees it in [`Drop`]; the payload box is *not* owned by
//! [`Closure`] directly — it is owned by the C++ shared destructor.
//!
//! ## Re-entrancy and panics
//!
//! The trampoline catches Rust panics via [`std::panic::catch_unwind`] and
//! converts them to a non-zero rc — unwinding across the `extern "C"` boundary
//! is undefined behavior. The user function is required `Fn + 'static` (not
//! `FnMut`); aliasing the captured state across re-entrant mlx-c calls is
//! safe because `Fn` mandates `&self` access.

use std::{
  ffi::c_void,
  os::raw::c_int,
  panic::{AssertUnwindSafe, catch_unwind},
  ptr,
};

use crate::{
  Array,
  error::{Error, Result, ensure_handler_installed},
};

/// Boxed type-erased Rust callable invoked by the mlx-c trampoline.
///
/// `Box<dyn Fn(&[Array]) -> Result<Vec<Array>>>` is itself a fat pointer
/// (vtable + data), so we wrap it in an outer `Box` to land on a stable
/// thin `*mut c_void` (the inner `Box<dyn Fn>` already heap-allocates the
/// closure; the outer `Box` is the indirection layer mlx-c hands back).
pub(crate) type BoxedFn = Box<dyn Fn(&[Array]) -> Result<Vec<Array>> + 'static>;

/// Safe RAII wrapper around an `mlx_closure` that keeps the captured Rust
/// callable alive for the entire lifetime of the C handle.
///
/// Construct via [`Closure::new`]; the returned value owns one reference to
/// the underlying `mlx_closure` and frees it on [`Drop`]. To pass the handle
/// into mlx-c transforms (`mlx_value_and_grad`, `mlx_vjp`, …) use
/// [`Closure::as_raw`], which borrows the handle without transferring
/// ownership. The Rust callable is held alive by the closure's mlx-c
/// destructor, *not* by this struct.
///
/// `Closure` is intentionally `!Send` + `!Sync`: the captured `F` may
/// reference [`crate::Array`] handles (themselves `!Send`), and the mlx-c
/// closure's payload destructor must run on the thread that built it.
pub struct Closure {
  inner: mlxrs_sys::mlx_closure,
}

impl Closure {
  /// Construct a closure from a Rust callable. Returns `Err` if the underlying
  /// `mlx_closure_new_func_payload` allocation fails.
  ///
  /// `F` is required `Fn + 'static` so the mlx-c side can invoke it across
  /// arbitrary later re-entries (including from within `mlx_eval`).
  pub fn new<F>(f: F) -> Result<Self>
  where
    F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
  {
    ensure_handler_installed();
    // Box the user closure on the heap, then re-box the resulting fat trait-
    // object pointer so the payload we hand to C is a thin `*mut c_void`.
    // SAFETY of pointer round-trip: we recover the same `Box<BoxedFn>` via
    // `Box::from_raw` exactly once, in `destroy_payload`. mlx-c invokes the
    // destructor exactly once when the underlying `shared_ptr` reaches
    // refcount 0.
    let boxed: Box<BoxedFn> = Box::new(Box::new(f));
    let payload_ptr: *mut c_void = Box::into_raw(boxed).cast();

    // SAFETY: `trampoline::<F>` and `destroy_payload` are both `extern "C"`
    // with the exact signatures mlx-c expects. `payload_ptr` is a freshly
    // boxed `Box<BoxedFn>` whose lifetime is transferred to mlx-c by this
    // call: mlx-c IMMEDIATELY wraps it in `std::shared_ptr<void>(payload,
    // dtor)` as the very first statement of its `try` block (see vendored
    // `mlx-c/mlx/c/closure.cpp::mlx_closure_new_func_payload`, line 70).
    // From that point on, the shared_ptr OWNS the payload — even if any
    // later allocation inside the same `try` throws (e.g. the lambda
    // capture or `mlx_closure_new_(cpp_closure)`), the shared_ptr's
    // destructor runs `destroy_payload(payload_ptr)` as part of stack
    // unwinding before the `catch` clause returns a NULL closure to us.
    // Therefore the NULL-ctx return path below MUST NOT reclaim the box
    // ourselves — that would double-free / UAF.
    // In production we call mlx-c directly. In `cfg(test)` builds we route
    // through a swappable function pointer (`test_seam::closure_new_fn`) so
    // unit tests can inject a NULL-returning stub that exercises the
    // `inner.ctx.is_null()` branch where the pre-fix F1 double-free lived —
    // see the `tests::closure_new_returns_err_*` cases in this file. The
    // `#[cfg(test)]` arm defaults to the same FFI symbol; test stubs satisfy
    // the same ABI + ownership contract (see `test_seam` docs), so the
    // unsafe contract is identical between the two arms.
    // SAFETY: `trampoline` and `destroy_payload` have the exact extern "C"
    // signatures mlx-c expects; `payload_ptr` is a freshly leaked
    // `Box<BoxedFn>` whose ownership transfers to mlx-c's shared_ptr per
    // the contract documented above. The `#[cfg(test)]` arm is functionally
    // identical (defaults to the same FFI symbol).
    let inner = unsafe { call_closure_new_ffi(payload_ptr) };
    if inner.ctx.is_null() {
      // mlx-c already owns `payload_ptr` via the
      // `std::shared_ptr<void>(payload, dtor)` it constructed at the top
      // of its `try` block. If the C++ ctor threw post-shared_ptr-
      // construction, the shared_ptr destructor has ALREADY released the
      // payload via `destroy_payload`. Reclaiming with `Box::from_raw`
      // here would be a double-free / UAF.
      //
      // We accept the (tiny) leak on the alternate path where mlx-c
      // returns NULL without ever constructing the shared_ptr (i.e. the
      // `mlx_closure_new_()` infallible sentinel constructor on the
      // catch arm somehow surfaced NULL — not currently observed in any
      // mlx-c codepath but a defensive consideration). Leak is strictly
      // preferable to UAF.
      return Err(crate::error::take_last().unwrap_or(Error::Backend {
        message: "mlx_closure_new_func_payload returned NULL ctx".into(),
      }));
    }
    Ok(Self { inner })
  }

  /// Borrow the raw `mlx_closure` handle for a transient FFI call.
  ///
  /// The returned handle MUST NOT be retained past this `&self` borrow —
  /// `Drop` will free the underlying handle. mlx-c transforms that consume a
  /// closure by *value* internally take a shared_ptr copy, so passing
  /// `closure.as_raw()` into e.g. `mlx_value_and_grad` is sound.
  #[inline]
  pub fn as_raw(&self) -> mlxrs_sys::mlx_closure {
    self.inner
  }
}

impl Drop for Closure {
  fn drop(&mut self) {
    // SAFETY: frees the handle this `Closure` owns exactly once. The closure's
    // C++ shared_ptr refcount drops; when it hits 0 the payload destructor
    // we registered runs and reclaims the `Box<BoxedFn>`. Runs during `Drop`
    // so must not touch TLS / panic / unwind across `extern "C"` — the rc is
    // discarded silently per the crate's `Drop` convention.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_free(self.inner);
    }
  }
}

// ──────────────────────── FFI call indirection ────────────────────────

/// Invoke `mlx_closure_new_func_payload` (production) or the test-seam stub
/// (`#[cfg(test)]`). Kept in a single helper so the safety annotation lives in
/// exactly one place — see [`Closure::new`] and the `test_seam` docs for the
/// ownership contract on `payload_ptr`.
///
/// # Safety
/// Caller must ensure `payload_ptr` was produced by `Box::into_raw` on a
/// `Box<BoxedFn>` and that ownership is hereby transferred to mlx-c's
/// `shared_ptr<void>(payload, destroy_payload)`. The `#[cfg(test)]` arm
/// routes through a swappable function pointer that defaults to the same
/// FFI symbol; swapped-in stubs must satisfy the identical ABI + ownership
/// contract.
#[inline]
unsafe fn call_closure_new_ffi(payload_ptr: *mut c_void) -> mlxrs_sys::mlx_closure {
  #[cfg(not(test))]
  // SAFETY: forwarded from caller; this is the production direct-FFI arm.
  unsafe {
    mlxrs_sys::mlx_closure_new_func_payload(Some(trampoline), payload_ptr, Some(destroy_payload))
  }
  #[cfg(test)]
  // SAFETY: forwarded from caller; the seam defaults to the same FFI symbol.
  unsafe {
    (test_seam::closure_new_fn())(Some(trampoline), payload_ptr, Some(destroy_payload))
  }
}

/// Invoke `mlx_closure_custom_new_func_payload` (production) or the test-seam
/// stub (`#[cfg(test)]`). Same single-call-site rationale as
/// [`call_closure_new_ffi`].
///
/// # Safety
/// Caller must ensure `payload_ptr` was produced by `Box::into_raw` on a
/// `Box<BoxedFn3>` and that ownership transfers to mlx-c's `shared_ptr`.
#[inline]
unsafe fn call_closure_custom_new_ffi(payload_ptr: *mut c_void) -> mlxrs_sys::mlx_closure_custom {
  #[cfg(not(test))]
  // SAFETY: forwarded from caller; production direct-FFI arm.
  unsafe {
    mlxrs_sys::mlx_closure_custom_new_func_payload(
      Some(trampoline_custom),
      payload_ptr,
      Some(destroy_payload_3),
    )
  }
  #[cfg(test)]
  // SAFETY: forwarded from caller; seam defaults to the same FFI symbol.
  unsafe {
    (test_seam::closure_custom_new_fn())(
      Some(trampoline_custom),
      payload_ptr,
      Some(destroy_payload_3),
    )
  }
}

// ─────────────────────────── trampoline ───────────────────────────

/// `extern "C"` shim invoked by mlx-c whenever the closure is applied.
///
/// `outputs_out` is an out-parameter slot pre-allocated by the caller (NULL
/// `ctx`); we populate it via `mlx_vector_array_set_data`. `inputs` is owned
/// by mlx-c (we read it; we do NOT free it). `payload` is the `*mut c_void`
/// we registered.
///
/// Returns `0` on success, non-zero on user error or panic. On user error /
/// panic we leave `outputs_out` populated with an empty `mlx_vector_array`
/// (still a valid handle that mlx-c will free) and post a `Backend` message
/// into the TLS error slot so `crate::error::check(rc)` can drain it.
extern "C" fn trampoline(
  outputs_out: *mut mlxrs_sys::mlx_vector_array,
  inputs: mlxrs_sys::mlx_vector_array,
  payload: *mut c_void,
) -> c_int {
  // Wrap the entire body in `catch_unwind` — any panic across `extern "C"`
  // is UB. We restore the panic as a Backend error in the TLS slot.
  let result = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: `payload` is the `*mut c_void` we stored via `Box::into_raw`
    // (preserved by mlx-c across calls). We cast back to `*const BoxedFn` and
    // borrow — NOT take ownership; the box is reclaimed in `destroy_payload`.
    let f: &BoxedFn = unsafe { &*payload.cast::<BoxedFn>() };

    // Borrow the input handles WITHOUT taking ownership: we build a
    // `Vec<Array>` of fresh handles by copying each element via
    // `mlx_vector_array_get` (refcount bump) — the original `inputs` vector
    // is mlx-c's. We then call the user function with a `&[Array]` borrow.
    let inputs_vec = borrow_inputs(inputs)?;

    // Invoke user function.
    let outputs = f(&inputs_vec)?;

    // Marshal outputs back into the out-param `mlx_vector_array`. We use
    // `mlx_vector_array_set_data` which copies the array handles into the
    // existing vector slot (refcount bump on each).
    write_outputs(outputs_out, &outputs)?;
    Ok::<(), Error>(())
  }));
  match result {
    Ok(Ok(())) => 0,
    Ok(Err(e)) => {
      // Stash the user error in TLS so `check(rc)` drains it.
      crate::error::set_last(e);
      // Populate the out-param with an empty vector so mlx-c's later
      // `mlx_vector_array_free` is a defined no-op.
      // SAFETY: `outputs_out` is the caller-owned pre-allocated handle slot;
      // writing an empty vector handle is the safe way to leave it.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
    Err(panic_payload) => {
      let msg = if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
        (*s).to_string()
      } else if let Some(s) = panic_payload.downcast_ref::<String>() {
        s.clone()
      } else {
        "panic in mlxrs::transforms closure trampoline".to_string()
      };
      crate::error::set_last(Error::Backend {
        message: format!("mlxrs::transforms closure trampoline caught panic: {msg}"),
      });
      // SAFETY: same as above — leave the out-param holding an empty handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
  }
}

/// `extern "C"` destructor mlx-c invokes when the closure's last `shared_ptr`
/// copy drops. Reclaims the `Box<BoxedFn>` we leaked at construction.
extern "C" fn destroy_payload(payload: *mut c_void) {
  if payload.is_null() {
    return;
  }
  // SAFETY: `payload` is the `*mut c_void` produced by `Box::into_raw` on a
  // `Box<BoxedFn>` in `Closure::new`. mlx-c calls this destructor exactly
  // once per registration. Box ownership returns here and is dropped.
  // Wrap drop in `catch_unwind` so a panicking user closure-destructor
  // cannot unwind across the C++ boundary.
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: see fn doc above — payload is a Box<BoxedFn> we created.
    let _: Box<BoxedFn> = unsafe { Box::from_raw(payload.cast::<BoxedFn>()) };
  }));
}

// ─────────────────────── vector_array marshalling ───────────────────────

/// Build a `Vec<Array>` from an mlx-c `mlx_vector_array` by copying out each
/// handle (refcount bump on each via `mlx_array_set`).
pub(crate) fn drain_vector(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
    // ctx); wrapping in `Array` first ensures `Drop` reclaims on early return.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: valid `vec` handle; `part.0` is the freshly-allocated out-param
    // populated by this call. rc surfaced via `check()`.
    crate::error::check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// Borrow the input handles of a `mlx_vector_array` as a `Vec<Array>` of
/// fresh refcount-shared copies. Same effect as [`drain_vector`] but used
/// inside the trampoline where the source `vec` is owned by mlx-c (we MUST
/// NOT free it).
fn borrow_inputs(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  drain_vector(vec)
}

/// Pack a `&[Array]` into a freshly allocated `mlx_vector_array` and write
/// its handle into `out`. mlx-c copies refcount-shared array handles into
/// the new vector storage. The previous contents of `*out` are leaked — mlx-c
/// gives us a NULL-ctx slot on first entry, so this is a safe overwrite.
fn write_outputs(out: *mut mlxrs_sys::mlx_vector_array, outputs: &[Array]) -> Result<()> {
  // Collect raw handles into a contiguous `Vec<mlx_array>` for FFI.
  let raw: Vec<mlxrs_sys::mlx_array> = outputs.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `out` is the trampoline's caller-owned out-param. Per mlx-c's
  // convention on entry it is a NULL-ctx handle; we replace it with a fresh
  // vector populated from `raw` (mlx-c copies the array handles, refcount-
  // bumping each).
  unsafe {
    *out = mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len());
  }
  // SAFETY: post-write null-check — the constructor is fallible.
  if unsafe { (*out).ctx.is_null() } && !outputs.is_empty() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx in closure trampoline".into(),
    }));
  }
  Ok(())
}

// ─────────────────────── caller-side helpers ───────────────────────

/// RAII guard for a temporary `mlx_vector_array`. Constructed *before* the
/// populating call so an early return / panic still frees it.
pub(crate) struct VectorArrayGuard(pub(crate) mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. Same `Drop`
    // discipline as elsewhere — discard rc silently.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}

/// Pack a `&[Array]` (or `&[&Array]` via iterator) into a fresh
/// `mlx_vector_array`. Returns the handle wrapped in a guard for RAII free.
pub(crate) fn vector_array_from_borrow(arrays: &[&Array]) -> Result<VectorArrayGuard> {
  ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `data_ptr` is either NULL (n==0, mlx-c builds an empty vector) or
  // a valid pointer to `raw.len()` borrowed handles live for this call (mlx-c
  // copies into the new vector, refcount-bumping each).
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len()) };
  if vec.ctx.is_null() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx".into(),
    }));
  }
  Ok(VectorArrayGuard(vec))
}

/// Same as [`vector_array_from_borrow`] but takes `&[Array]` (most-common
/// caller convenience).
pub(crate) fn vector_array_from_slice(arrays: &[Array]) -> Result<VectorArrayGuard> {
  ensure_handler_installed();
  let raw: Vec<mlxrs_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
  let data_ptr = if raw.is_empty() {
    ptr::null()
  } else {
    raw.as_ptr()
  };
  // SAFETY: `data_ptr` is either NULL or a valid pointer to `raw.len()`
  // borrowed handles live for this call; mlx-c copies into the new vector.
  let vec = unsafe { mlxrs_sys::mlx_vector_array_new_data(data_ptr, raw.len()) };
  if vec.ctx.is_null() {
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_vector_array_new_data returned NULL ctx".into(),
    }));
  }
  Ok(VectorArrayGuard(vec))
}

/// RAII guard for a temporary `mlx_closure_value_and_grad`.
pub(crate) struct ClosureValueAndGradGuard(pub(crate) mlxrs_sys::mlx_closure_value_and_grad);
impl Drop for ClosureValueAndGradGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free,
    // rc discarded.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_value_and_grad_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_closure_custom`.
pub(crate) struct ClosureCustomGuard(pub(crate) mlxrs_sys::mlx_closure_custom);
impl Drop for ClosureCustomGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_custom_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_closure` that we own (e.g. the result of
/// `mlx_checkpoint` / `mlx_custom_function`).
pub(crate) struct RawClosureGuard(pub(crate) mlxrs_sys::mlx_closure);
impl Drop for RawClosureGuard {
  fn drop(&mut self) {
    // SAFETY: same discipline as `VectorArrayGuard` — single-owner free.
    unsafe {
      let _ = mlxrs_sys::mlx_closure_free(self.0);
    }
  }
}

/// Build a custom-VJP `mlx_closure_custom` from a Rust 3-input function.
///
/// The contract matches `mlx_custom_vjp`'s `fun_vjp` argument:
/// `(primals, cotangents, outputs) -> grads` — the same positional order
/// `mlx::core::CustomTransforms::vjp` invokes its `vjp_fun_` callback with
/// (`mlx/primitives.cpp::CustomTransforms::vjp`).
pub(crate) fn closure_custom_new<F>(f: F) -> Result<ClosureCustomGuard>
where
  F: Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static,
{
  ensure_handler_installed();
  let boxed: Box<BoxedFn3> = Box::new(Box::new(f));
  let payload_ptr: *mut c_void = Box::into_raw(boxed).cast();
  // SAFETY: trampoline + destructor have correct signatures. `payload_ptr` is
  // a freshly leaked `Box<BoxedFn3>` whose lifetime is transferred to mlx-c
  // by this call: mlx-c IMMEDIATELY wraps it in
  // `std::shared_ptr<void>(payload, dtor)` as the first statement of its
  // `try` block (see vendored
  // `mlx-c/mlx/c/closure.cpp::mlx_closure_custom_new_func_payload`,
  // line 471). From that point on the shared_ptr OWNS the payload — even
  // if any later allocation inside the same `try` throws, the shared_ptr
  // destructor runs `destroy_payload_3(payload_ptr)` during unwinding
  // before the `catch` clause returns NULL. Therefore the NULL-ctx return
  // path below MUST NOT reclaim the box — that would double-free / UAF.
  // Production: direct FFI; tests: route through the swappable seam so the
  // NULL-ctx branch (which used to double-free in F1) is exercised
  // deterministically by `tests::closure_custom_new_returns_err_*`.
  // SAFETY: `payload_ptr` is a freshly leaked `Box<BoxedFn3>` whose
  // ownership transfers to mlx-c per the contract documented above.
  let inner = unsafe { call_closure_custom_new_ffi(payload_ptr) };
  if inner.ctx.is_null() {
    // mlx-c already owns `payload_ptr` via its `shared_ptr<void>`; the
    // shared_ptr destructor has run (or will run on the natural drop
    // path) and released the payload via `destroy_payload_3`. DO NOT
    // reclaim manually — that would be a double-free / UAF. Same
    // rationale as `Closure::new` above: accept a (tiny) leak on the
    // unobserved-NULL path over a deterministic UAF.
    return Err(crate::error::take_last().unwrap_or(Error::Backend {
      message: "mlx_closure_custom_new_func_payload returned NULL ctx".into(),
    }));
  }
  Ok(ClosureCustomGuard(inner))
}

pub(crate) type BoxedFn3 =
  Box<dyn Fn(&[Array], &[Array], &[Array]) -> Result<Vec<Array>> + 'static>;

// MLX core `CustomTransforms::vjp` invokes its `vjp_fun_` callback with the
// positional argument order `(primals, cotangents, outputs)` — see
// `mlx/primitives.cpp::CustomTransforms::vjp` upstream:
//
// ```cpp
// auto all_vjps = vjp_fun_(inputs, cotangents, outputs);
// ```
//
// The Rust trampoline therefore names its second / third `mlx_vector_array`
// arguments `cotangents` / `outputs` to match — the user closure receives the
// triple in this same order via `f(&primals, &cotangents, &outputs)`.
extern "C" fn trampoline_custom(
  outputs_out: *mut mlxrs_sys::mlx_vector_array,
  primals: mlxrs_sys::mlx_vector_array,
  cotangents: mlxrs_sys::mlx_vector_array,
  outputs: mlxrs_sys::mlx_vector_array,
  payload: *mut c_void,
) -> c_int {
  let result = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: `payload` was produced by `Box::into_raw(Box<BoxedFn3>)` and
    // is preserved by mlx-c; borrow without taking ownership.
    let f: &BoxedFn3 = unsafe { &*payload.cast::<BoxedFn3>() };
    let p = borrow_inputs(primals)?;
    let c = borrow_inputs(cotangents)?;
    let o = borrow_inputs(outputs)?;
    let grads = f(&p, &c, &o)?;
    write_outputs(outputs_out, &grads)?;
    Ok::<(), Error>(())
  }));
  match result {
    Ok(Ok(())) => 0,
    Ok(Err(e)) => {
      crate::error::set_last(e);
      // SAFETY: leave out-param holding an empty vector handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
    Err(panic_payload) => {
      let msg = if let Some(s) = panic_payload.downcast_ref::<&'static str>() {
        (*s).to_string()
      } else if let Some(s) = panic_payload.downcast_ref::<String>() {
        s.clone()
      } else {
        "panic in mlxrs::transforms custom-VJP trampoline".to_string()
      };
      crate::error::set_last(Error::Backend {
        message: format!("mlxrs::transforms custom-VJP trampoline caught panic: {msg}"),
      });
      // SAFETY: leave out-param holding an empty vector handle.
      unsafe {
        if !outputs_out.is_null() {
          *outputs_out = mlxrs_sys::mlx_vector_array_new();
        }
      }
      1
    }
  }
}

extern "C" fn destroy_payload_3(payload: *mut c_void) {
  if payload.is_null() {
    return;
  }
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: payload is a Box<BoxedFn3> we created; reclaim ownership once.
    let _: Box<BoxedFn3> = unsafe { Box::from_raw(payload.cast::<BoxedFn3>()) };
  }));
}

// ─────────────────────────── test seam ───────────────────────────

/// Test-only function-pointer indirection over the mlx-c closure constructors.
///
/// Production builds (`#[cfg(not(test))]`) call
/// `mlxrs_sys::mlx_closure_*_new_func_payload` directly: zero indirection,
/// zero overhead. The compiler eliminates this module entirely.
///
/// In `#[cfg(test)]` builds the constructor call in `Closure::new` /
/// `closure_custom_new` routes through an [`AtomicPtr`]-backed function
/// pointer slot here, defaulting to the real mlx-c symbol. The unit tests
/// below swap in a deterministic stub that simulates mlx-c's
/// shared_ptr-then-throw failure mode (invokes the destructor we registered,
/// then returns NULL ctx) to exercise the `inner.ctx.is_null()` branch
/// where the pre-fix F1 double-free lived. Without this seam the NULL-ctx
/// branch is unreachable from Rust (we cannot inject OOM into mlx-c) and
/// CI would be blind to a regression that re-introduced the reclaim.
///
/// ## R3-F2 fix: serialization + non-reentrant install lock
///
/// Each per-constructor slot now has TWO collaborators:
///
/// * `*_slot()` — an `AtomicPtr<()>` holding the currently-installed fn
///   pointer. Read via lock-free `load(Acquire)` in `*_fn()`, which the
///   `call_*_ffi` helpers invoke synchronously during `Closure::new` /
///   `closure_custom_new`. Lock-free reads guarantee no deadlock if a test
///   has the install lock held: the install lock and the slot are
///   independent.
/// * `*_install_lock()` — a `Mutex<()>` held by the [`ScopedClosureCtor`] /
///   [`ScopedCustomCtor`] guard for its ENTIRE lifetime (install + use +
///   restore). This makes the install→use→restore sequence atomic w.r.t.
///   any other guard. Combined with the [`serial_guard`] mutex inside the
///   test module (which every seam test acquires as its first action),
///   the seam tests run strictly one-at-a-time even under default parallel
///   `cargo test`; the install lock is defense-in-depth in case future
///   tests forget to acquire `serial_guard`.
///
/// The earlier design held the slot mutex only across the fn-pointer
/// swap, then released it before the guarded `Closure::new` call. Two
/// parallel seam tests could install conflicting stubs between install and
/// use, and non-LIFO drops could restore an older stub atop a newer
/// guard's install. Both windows are closed by holding the install lock
/// for the entire guard lifetime.
#[cfg(test)]
pub(crate) mod test_seam {
  use std::sync::{
    Mutex, MutexGuard, OnceLock,
    atomic::{AtomicPtr, Ordering},
  };

  use super::*;

  /// Function-pointer type matching `mlx_closure_new_func_payload`'s ABI.
  pub(crate) type ClosureNewFn = unsafe extern "C" fn(
    fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure;

  /// Function-pointer type matching `mlx_closure_custom_new_func_payload`'s ABI.
  pub(crate) type ClosureCustomNewFn = unsafe extern "C" fn(
    fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure_custom;

  /// Slot storing the currently-installed `ClosureNewFn` pointer.
  ///
  /// Stored as `AtomicPtr<()>` so reads (in [`closure_new_fn`]) don't need
  /// to take a lock — critical because that read happens during
  /// `Closure::new` while a test may be holding the install lock for the
  /// guard's lifetime; locking here would deadlock.
  fn closure_new_slot() -> &'static AtomicPtr<()> {
    static SLOT: OnceLock<AtomicPtr<()>> = OnceLock::new();
    SLOT.get_or_init(|| AtomicPtr::new(mlxrs_sys::mlx_closure_new_func_payload as *mut ()))
  }

  /// Mutex held by a `ScopedClosureCtor` guard for its entire lifetime to
  /// serialize install→use→restore against any other guard installation.
  /// `Mutex<()>` (not `Mutex<FnPtr>`) so the held guard never blocks the
  /// lock-free `closure_new_fn()` reads on the slot.
  fn closure_new_install_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
  }

  /// Mirror of [`closure_new_slot`] for the custom-VJP constructor seam.
  fn closure_custom_new_slot() -> &'static AtomicPtr<()> {
    static SLOT: OnceLock<AtomicPtr<()>> = OnceLock::new();
    SLOT.get_or_init(|| AtomicPtr::new(mlxrs_sys::mlx_closure_custom_new_func_payload as *mut ()))
  }

  /// Mirror of [`closure_new_install_lock`] for the custom-VJP seam.
  fn closure_custom_new_install_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
  }

  /// Read the currently-installed constructor (default: real mlx-c symbol).
  ///
  /// Lock-free atomic load: must NOT block, because the calling
  /// `Closure::new` may be running inside a test that already holds the
  /// install lock via [`ScopedClosureCtor`].
  pub(crate) fn closure_new_fn() -> ClosureNewFn {
    let ptr = closure_new_slot().load(Ordering::Acquire);
    // SAFETY: SLOT only ever contains values written by:
    //   (a) initial `OnceLock` init: address of the real mlx-c FFI symbol;
    //   (b) `ScopedClosureCtor::install` / its `Drop`: the address of a
    //       `ClosureNewFn` (an `unsafe extern "C" fn`) cast `as *mut ()`,
    //       or a prior value of (a)/(b).
    // Both source forms are valid fn-pointers of type `ClosureNewFn`, so
    // round-tripping through `*mut ()` and transmuting back recovers the
    // exact original pointer with the same ABI. Fn pointers and `*mut ()`
    // have identical size + repr on all supported targets (the language
    // guarantees fn pointers are word-sized).
    unsafe { std::mem::transmute::<*mut (), ClosureNewFn>(ptr) }
  }

  /// Read the currently-installed custom-VJP constructor.
  pub(crate) fn closure_custom_new_fn() -> ClosureCustomNewFn {
    let ptr = closure_custom_new_slot().load(Ordering::Acquire);
    // SAFETY: mirror of `closure_new_fn` above — SLOT only ever stores
    // valid `ClosureCustomNewFn` fn-pointer addresses; round-tripping
    // through `*mut ()` is sound on all supported targets.
    unsafe { std::mem::transmute::<*mut (), ClosureCustomNewFn>(ptr) }
  }

  /// RAII guard: replace [`closure_new_fn`] with `stub` for the guard's
  /// lifetime, restore the previous symbol on drop.
  ///
  /// Holds [`closure_new_install_lock`] for the ENTIRE guard lifetime
  /// (install + test body + restore) so the swap→use→restore sequence is
  /// atomic with respect to any concurrent `ScopedClosureCtor` install on
  /// another thread. Test bodies don't read this lock directly — only
  /// other `install` calls block on it, which means a parallel seam test
  /// can't install a conflicting stub between this guard's install and
  /// the matching `Closure::new` call inside the test body.
  pub(crate) struct ScopedClosureCtor {
    // Holds the install lock for the entire guard lifetime, blocking any
    // other `install` call. Auto-released when this struct drops.
    _install_guard: MutexGuard<'static, ()>,
    prev: *mut (),
  }

  impl ScopedClosureCtor {
    pub(crate) fn install(stub: ClosureNewFn) -> Self {
      // Acquire the install lock first and hold it for the guard's
      // lifetime. Recover from poison: a prior seam test that panicked
      // mid-test should not block subsequent runs.
      let guard = closure_new_install_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
      let stub_ptr = stub as *mut ();
      let prev = closure_new_slot().swap(stub_ptr, Ordering::AcqRel);
      Self {
        _install_guard: guard,
        prev,
      }
    }
  }

  impl Drop for ScopedClosureCtor {
    fn drop(&mut self) {
      // Restore previous symbol even if the test panicked. The atomic
      // swap pairs with the matching swap in `install`; the install lock
      // is released when `_install_guard` drops at the end of this fn.
      closure_new_slot().store(self.prev, Ordering::Release);
    }
  }

  /// Mirror of [`ScopedClosureCtor`] for the custom-VJP constructor seam.
  pub(crate) struct ScopedCustomCtor {
    _install_guard: MutexGuard<'static, ()>,
    prev: *mut (),
  }

  impl ScopedCustomCtor {
    pub(crate) fn install(stub: ClosureCustomNewFn) -> Self {
      let guard = closure_custom_new_install_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
      let stub_ptr = stub as *mut ();
      let prev = closure_custom_new_slot().swap(stub_ptr, Ordering::AcqRel);
      Self {
        _install_guard: guard,
        prev,
      }
    }
  }

  impl Drop for ScopedCustomCtor {
    fn drop(&mut self) {
      closure_custom_new_slot().store(self.prev, Ordering::Release);
    }
  }
}

#[cfg(test)]
mod tests {
  //! Deterministic regression tests for F1 (NULL-ctx UAF) via the
  //! [`test_seam`] function-pointer indirection.
  //!
  //! Pre-fix `Closure::new` and `closure_custom_new` reclaimed the payload
  //! via `Box::from_raw(payload_ptr.cast())` when mlx-c returned a NULL
  //! `ctx`. Per `mlx-c/mlx/c/closure.cpp` lines 70 / 471 mlx-c constructs a
  //! `std::shared_ptr<void>(payload, dtor)` as the first statement of the
  //! `try` block, so on any later throw the shared_ptr destructor has
  //! already invoked the registered Rust destructor (`destroy_payload` /
  //! `destroy_payload_3`) during stack unwinding — the Rust-side reclaim
  //! was a double-free / UAF.
  //!
  //! We can't deterministically inject OOM into mlx-c, so the integration
  //! tests in `tests/transforms.rs` only ever exercise the success path
  //! where `inner.ctx` is non-null — meaning a regression that
  //! re-introduced the reclaim would not surface in CI. These tests close
  //! that gap by swapping in a stub constructor that simulates the
  //! shared_ptr-then-throw failure mode.
  //!
  //! ## R3-F1 fix: ground-truth Drop sentinel
  //!
  //! Earlier seam tests asserted on a `static CLOSURE_DTOR_CALLS` counter
  //! that the STUB itself incremented before invoking the destructor.
  //! That counter does NOT observe the actual `Box<BoxedFn>` drop: a
  //! regression re-introducing `Box::from_raw(payload_ptr)` on the
  //! NULL-ctx branch would still produce `CLOSURE_DTOR_CALLS == 1` (the
  //! stub bumps it exactly once), even though TWO drops happened (the
  //! stub's `d(payload)` call ran the destructor, then the Rust reclaim
  //! ran it again → UB). The deliberate-breakage test caught it
  //! incidentally via SIGSEGV, but that is not a deterministic
  //! observation: under different allocator state or with the `_no_dtor`
  //! variant the regression would silently pass.
  //!
  //! These tests now capture a [`DropSentinel`] in the user closure via
  //! `move`. The sentinel's `Drop` impl increments a per-test
  //! `Arc<AtomicUsize>` — counting the ACTUAL number of times the boxed
  //! closure was reclaimed. Pre-fix would produce `drop_counter == 2`
  //! (double-free); post-fix produces exactly `1` on the dtor-invoked
  //! stub and `0` on the no-dtor stub (leak-over-UAF contract: when
  //! mlx-c did not run the destructor, Rust MUST NOT reclaim — a leak is
  //! strictly preferable to UB).
  //!
  //! ## R3-F2 fix: serial_guard
  //!
  //! Every seam test acquires [`serial_guard`] as its FIRST action. This
  //! is a crate-local `Mutex<()>` that strictly serializes the seam
  //! tests, eliminating cross-test stub-pointer contamination even
  //! under default `cargo test` parallelism. Defense-in-depth on top of
  //! the per-slot install lock held by `ScopedClosureCtor` /
  //! `ScopedCustomCtor` for the guard's full lifetime — see the
  //! `test_seam` module docs for the full ordering rationale.

  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };

  use super::{
    test_seam::{ClosureCustomNewFn, ClosureNewFn, ScopedClosureCtor, ScopedCustomCtor},
    *,
  };

  // ─────────────────── shared test infrastructure ───────────────────

  /// Crate-local test serialization mutex. Every seam test acquires this
  /// guard as its FIRST line so the seam tests run strictly one-at-a-time
  /// even under default `cargo test` parallel scheduling. Combined with
  /// the install lock held by each `Scoped*Ctor` for its full lifetime
  /// (see `test_seam` module docs), this is belt-and-suspenders against
  /// cross-test stub-pointer contamination.
  ///
  /// Recovers from poison: a prior seam test that panicked under its own
  /// guard should not block subsequent tests.
  fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
  }

  /// Drop sentinel captured by `move` into the user closure passed to
  /// `Closure::new` / `closure_custom_new`. Its `Drop` impl increments
  /// the shared `Arc<AtomicUsize>`, giving ground truth on how many
  /// times the boxed closure was reclaimed.
  ///
  /// Expected post-fix counts:
  /// * dtor-invoked stub: 1 (mlx-c's shared_ptr destructor — modelled
  ///   by the stub — runs `destroy_payload`, which drops the box).
  /// * no-dtor stub: 0 (mlx-c surfaced NULL without ever constructing
  ///   the shared_ptr — Rust accepts a tiny leak rather than reclaim).
  ///
  /// Pre-fix (`Box::from_raw` on the NULL-ctx branch): 2 on the
  /// dtor-invoked path (double-free) and 1 on the no-dtor path (UAF on
  /// any subsequent mlx-c-internal shared_ptr drop the test can't
  /// observe; instrumented here so the regression fails deterministically
  /// instead of crashing the harness with SIGSEGV).
  struct DropSentinel {
    counter: Arc<AtomicUsize>,
  }

  impl Drop for DropSentinel {
    fn drop(&mut self) {
      self.counter.fetch_add(1, Ordering::SeqCst);
    }
  }

  // ────────────── Closure::new NULL-ctx regression tests ──────────────

  /// Stub that simulates mlx-c's NULL-after-throw path for
  /// `mlx_closure_new_func_payload`:
  ///   1. Invoke the registered destructor on `payload` (mirroring the
  ///      `shared_ptr<void>(payload, dtor)` destructor that runs during
  ///      stack unwinding when the C++ ctor throws after shared_ptr
  ///      construction).
  ///   2. Return an `mlx_closure` with NULL `ctx` (mirroring the value
  ///      the `catch` clause returns to Rust).
  unsafe extern "C" fn stub_closure_new_invokes_dtor_then_returns_null(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure {
    if let Some(d) = dtor {
      // SAFETY: `d` is `destroy_payload` (our own `extern "C"` fn),
      // called exactly once on the `payload` mlx-c received — same
      // contract as the real mlx-c implementation when its `try` block
      // throws after shared_ptr construction.
      unsafe { d(payload) };
    }
    mlxrs_sys::mlx_closure {
      ctx: ptr::null_mut(),
    }
  }

  /// Stub that returns NULL `ctx` WITHOUT invoking the destructor:
  /// models the "alternate path" referenced by the SAFETY comment in
  /// `Closure::new` where mlx-c surfaces a NULL closure without ever
  /// constructing the shared_ptr. Per the documented contract the Rust
  /// wrapper accepts a tiny leak here over an undefined-behavior reclaim.
  unsafe extern "C" fn stub_closure_new_returns_null_no_dtor(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    _payload: *mut c_void,
    _dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure {
    mlxrs_sys::mlx_closure {
      ctx: ptr::null_mut(),
    }
  }

  #[test]
  fn closure_new_returns_err_without_double_free_when_ffi_returns_null_after_invoking_destructor() {
    let _serial = serial_guard();
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let _guard =
      ScopedClosureCtor::install(stub_closure_new_invokes_dtor_then_returns_null as ClosureNewFn);

    let sentinel = DropSentinel {
      counter: Arc::clone(&drop_counter),
    };
    // `move` captures `sentinel` by value: the sentinel is owned by the
    // closure, which is itself boxed into the payload mlx-c receives.
    // The closure is `Fn` (we only `&`-borrow `sentinel` through the
    // `let _keep = &sentinel;`), satisfying `Closure::new`'s bound.
    let result = Closure::new(move |_xs: &[Array]| {
      let _keep = &sentinel;
      Ok(Vec::<Array>::new())
    });

    assert!(
      result.is_err(),
      "Closure::new must surface Err when mlx-c returns NULL ctx"
    );

    // CRITICAL F1 regression assert: the boxed closure was reclaimed
    // EXACTLY ONCE — via the stub's invocation of `destroy_payload`,
    // which `Box::from_raw`'s the payload and drops the inner closure
    // (and with it, the captured sentinel).
    //
    // Pre-fix the production NULL-ctx branch ALSO called
    // `Box::from_raw(payload_ptr)` — that's a second drop on the same
    // pointer → double-free / UB. The earlier `CLOSURE_DTOR_CALLS`
    // static (incremented in the STUB) could not detect this; it would
    // still read 1 because the stub only bumps it once. The sentinel-
    // backed `drop_counter` here counts ACTUAL drops, so a pre-fix
    // regression deterministically fails this assertion with the value 2.
    let observed = drop_counter.load(Ordering::SeqCst);
    assert_eq!(
      observed, 1,
      "F1 REGRESSION: boxed closure was dropped {observed} times; expected \
       EXACTLY 1 (a pre-fix `Box::from_raw` on the NULL-ctx branch produces \
       2 = double-free / UAF; a missing-dtor regression produces 0)."
    );
  }

  #[test]
  fn closure_new_returns_err_without_reclaim_when_ffi_returns_null_without_invoking_destructor() {
    let _serial = serial_guard();
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let _guard = ScopedClosureCtor::install(stub_closure_new_returns_null_no_dtor as ClosureNewFn);

    let sentinel = DropSentinel {
      counter: Arc::clone(&drop_counter),
    };
    let result = Closure::new(move |_xs: &[Array]| {
      let _keep = &sentinel;
      Ok(Vec::<Array>::new())
    });

    assert!(
      result.is_err(),
      "Closure::new must surface Err when mlx-c returns NULL ctx (no-dtor path)"
    );

    // Leak-over-UAF contract: if mlx-c did not invoke the destructor
    // (it never constructed the shared_ptr), Rust MUST NOT reclaim —
    // an mlx-c-internal later drop on the still-live payload would
    // make a Rust-side reclaim a UAF. We accept the (tiny) leak.
    //
    // The sentinel-backed `drop_counter` deterministically reads 0 in
    // the post-fix world. A pre-fix `Box::from_raw` regression on the
    // NULL-ctx branch would advance it to 1, failing the assertion.
    let observed = drop_counter.load(Ordering::SeqCst);
    assert_eq!(
      observed, 0,
      "Leak-over-UAF contract violated: boxed closure was dropped {observed} \
       times; expected EXACTLY 0 (Rust reclaim here would be UAF on any \
       subsequent mlx-c-internal payload drop)."
    );
  }

  // ─────── closure_custom_new NULL-ctx regression tests ───────

  /// `BoxedFn3`-flavored mirror of
  /// [`stub_closure_new_invokes_dtor_then_returns_null`].
  unsafe extern "C" fn stub_closure_custom_new_invokes_dtor_then_returns_null(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    payload: *mut c_void,
    dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure_custom {
    if let Some(d) = dtor {
      // SAFETY: `d` is `destroy_payload_3`, invoked exactly once on the
      // `payload` mlx-c received.
      unsafe { d(payload) };
    }
    mlxrs_sys::mlx_closure_custom {
      ctx: ptr::null_mut(),
    }
  }

  /// `BoxedFn3`-flavored mirror of
  /// [`stub_closure_new_returns_null_no_dtor`].
  unsafe extern "C" fn stub_closure_custom_new_returns_null_no_dtor(
    _fun: Option<
      unsafe extern "C" fn(
        *mut mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        mlxrs_sys::mlx_vector_array,
        *mut c_void,
      ) -> c_int,
    >,
    _payload: *mut c_void,
    _dtor: Option<unsafe extern "C" fn(*mut c_void)>,
  ) -> mlxrs_sys::mlx_closure_custom {
    mlxrs_sys::mlx_closure_custom {
      ctx: ptr::null_mut(),
    }
  }

  #[test]
  fn closure_custom_new_returns_err_without_double_free_when_ffi_returns_null_after_invoking_destructor()
   {
    let _serial = serial_guard();
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let _guard = ScopedCustomCtor::install(
      stub_closure_custom_new_invokes_dtor_then_returns_null as ClosureCustomNewFn,
    );

    let sentinel = DropSentinel {
      counter: Arc::clone(&drop_counter),
    };
    let result = closure_custom_new(move |_p: &[Array], _o: &[Array], _c: &[Array]| {
      let _keep = &sentinel;
      Ok(Vec::new())
    });

    assert!(
      result.is_err(),
      "closure_custom_new must surface Err when mlx-c returns NULL ctx"
    );

    // F1 regression assert for the `BoxedFn3` path. Same rationale as
    // `Closure::new`: pre-fix `Box::from_raw` produces 2 (double-free);
    // post-fix is exactly 1 (stub-invoked `destroy_payload_3`).
    let observed = drop_counter.load(Ordering::SeqCst);
    assert_eq!(
      observed, 1,
      "F1 REGRESSION (custom-VJP): boxed closure was dropped {observed} times; \
       expected EXACTLY 1 (pre-fix `Box::from_raw` on the NULL-ctx branch \
       produces 2 = double-free / UAF)."
    );
  }

  #[test]
  fn closure_custom_new_returns_err_without_reclaim_when_ffi_returns_null_without_invoking_destructor()
   {
    let _serial = serial_guard();
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let _guard =
      ScopedCustomCtor::install(stub_closure_custom_new_returns_null_no_dtor as ClosureCustomNewFn);

    let sentinel = DropSentinel {
      counter: Arc::clone(&drop_counter),
    };
    let result = closure_custom_new(move |_p: &[Array], _o: &[Array], _c: &[Array]| {
      let _keep = &sentinel;
      Ok(Vec::new())
    });

    assert!(
      result.is_err(),
      "closure_custom_new must surface Err when mlx-c returns NULL ctx (no-dtor path)"
    );

    // Leak-over-UAF contract for the `BoxedFn3` path. Same rationale as
    // `Closure::new`'s no-dtor test: Rust MUST NOT reclaim a pointer
    // mlx-c may still own.
    let observed = drop_counter.load(Ordering::SeqCst);
    assert_eq!(
      observed, 0,
      "Leak-over-UAF contract violated (custom-VJP): boxed closure was dropped \
       {observed} times; expected EXACTLY 0."
    );
  }
}
