//! Deterministic regression tests for the NULL-ctx UAF via the
//! [`test_seam`] function-pointer indirection.
//!
//! A naive `Closure::new` / `closure_custom_new` would reclaim the
//! payload via `Box::from_raw(payload_ptr.cast())` when mlx-c returned a
//! NULL `ctx`. Per `mlx-c/mlx/c/closure.cpp` lines 70 / 471 mlx-c
//! constructs a `std::shared_ptr<void>(payload, dtor)` as the first
//! statement of the `try` block, so on any later throw the shared_ptr
//! destructor has already invoked the registered Rust destructor
//! (`destroy_payload` / `destroy_payload_3`) during stack unwinding —
//! such a Rust-side reclaim would be a double-free / UAF.
//!
//! We can't deterministically inject OOM into mlx-c, so the integration
//! tests in `tests/transforms.rs` only ever exercise the success path
//! where `inner.ctx` is non-null — meaning a regression that
//! re-introduced the reclaim would not surface in CI. These tests close
//! that gap by swapping in a stub constructor that simulates the
//! shared_ptr-then-throw failure mode.
//!
//! ## Ground-truth Drop sentinel
//!
//! A `static CLOSURE_DTOR_CALLS` counter that the STUB itself
//! incremented before invoking the destructor would NOT observe the
//! actual `Box<BoxedFn>` drop: a regression re-introducing
//! `Box::from_raw(payload_ptr)` on the NULL-ctx branch would still
//! produce `CLOSURE_DTOR_CALLS == 1` (the stub bumps it exactly once),
//! even though TWO drops happened (the stub's `d(payload)` call ran the
//! destructor, then the Rust reclaim ran it again → UB). Such a
//! regression would surface only incidentally via SIGSEGV, which is not
//! a deterministic observation: under different allocator state or with
//! the `_no_dtor` variant the regression would silently pass.
//!
//! These tests instead capture a [`DropSentinel`] in the user closure
//! via `move`. The sentinel's `Drop` impl increments a per-test
//! `Arc<AtomicUsize>` — counting the ACTUAL number of times the boxed
//! closure was reclaimed. A `Box::from_raw` reclaim on the NULL-ctx
//! branch would produce `drop_counter == 2` (double-free); the correct
//! code produces exactly `1` on the dtor-invoked stub and `0` on the
//! no-dtor stub (leak-over-UAF contract: when mlx-c did not run the
//! destructor, Rust MUST NOT reclaim — a leak is strictly preferable to
//! UB).
//!
//! ## serial_guard
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
/// Expected counts for the correct code:
/// * dtor-invoked stub: 1 (mlx-c's shared_ptr destructor — modelled
///   by the stub — runs `destroy_payload`, which drops the box).
/// * no-dtor stub: 0 (mlx-c surfaced NULL without ever constructing
///   the shared_ptr — Rust accepts a tiny leak rather than reclaim).
///
/// A `Box::from_raw` reclaim on the NULL-ctx branch: 2 on the
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

  // Regression assert: the boxed closure was reclaimed
  // EXACTLY ONCE — via the stub's invocation of `destroy_payload`,
  // which `Box::from_raw`'s the payload and drops the inner closure
  // (and with it, the captured sentinel).
  //
  // A NULL-ctx branch that ALSO called `Box::from_raw(payload_ptr)`
  // would do a second drop on the same pointer → double-free / UB. A
  // `CLOSURE_DTOR_CALLS` static (incremented in the STUB) could not
  // detect this; it would still read 1 because the stub only bumps it
  // once. The sentinel-backed `drop_counter` here counts ACTUAL drops,
  // so such a regression deterministically fails this assertion with
  // the value 2.
  let observed = drop_counter.load(Ordering::SeqCst);
  assert_eq!(
    observed, 1,
    "REGRESSION: boxed closure was dropped {observed} times; expected \
       EXACTLY 1 (a `Box::from_raw` reclaim on the NULL-ctx branch produces \
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
  // The sentinel-backed `drop_counter` deterministically reads 0 for
  // the correct code. A `Box::from_raw` reclaim on the NULL-ctx branch
  // would advance it to 1, failing the assertion.
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

  // Regression assert for the `BoxedFn3` path. Same rationale as
  // `Closure::new`: a `Box::from_raw` reclaim produces 2 (double-free);
  // the correct code is exactly 1 (stub-invoked `destroy_payload_3`).
  let observed = drop_counter.load(Ordering::SeqCst);
  assert_eq!(
    observed, 1,
    "REGRESSION (custom-VJP): boxed closure was dropped {observed} times; \
       expected EXACTLY 1 (a `Box::from_raw` reclaim on the NULL-ctx branch \
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
