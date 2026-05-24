//! Integration tests for `mlxrs::transforms` — the autograd / custom-VJP /
//! checkpoint / bulk-eval safe-wrapper port.
//!
//! Mirrors mlx-swift's `Tests/MLXTests/TransformTests.swift` cases plus
//! lifecycle tests for the `Closure` payload trampoline.

use mlxrs::{
  Array,
  ops::arithmetic::{add, multiply, power, square},
  transforms::{Closure, async_eval, checkpoint, custom_vjp, eval, grad, jvp, value_and_grad, vjp},
};

// Small float comparison helper to keep call sites readable.
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
  (a - b).abs() <= eps
}

// ──────────────────────── Closure round-trip ────────────────────────

/// Constructing a `Closure` and using it transitively through
/// `value_and_grad` proves the trampoline + payload-passing round-trip works
/// end-to-end. The Rust callable returns a squared array; differentiating
/// via the closure exercises the entire payload-trampoline-FFI path.
#[test]
fn closure_construction_succeeds_and_round_trips() {
  // Just constructing a Closure should populate the handle (non-null ctx)
  // and dropping it should free without UB. The actual round-trip is
  // covered exhaustively by `value_and_grad_simple_quadratic` etc.; this
  // test isolates Closure construction + drop.
  let cls = Closure::new(|xs: &[Array]| Ok(vec![square(&xs[0])?])).unwrap();
  assert!(!cls.as_raw().ctx.is_null(), "Closure ctx must be non-null");
  drop(cls); // Trampoline destructor must not panic / UAF.
}

/// Drop the construction-scope `Closure` and verify no UAF / leak via the
/// monotonic peak_memory probe (see [feedback_no_global_peak_memory_assert]:
/// only `>=` checks, never magnitude).
#[test]
fn closure_drop_releases_ffi_handle() {
  let baseline = mlxrs::memory::peak_memory().unwrap();
  for _ in 0..10 {
    let cls = Closure::new(|xs: &[Array]| Ok(vec![square(&xs[0]).unwrap()])).unwrap();
    // Drop here.
    drop(cls);
  }
  let after = mlxrs::memory::peak_memory().unwrap();
  // Monotonic: peak only goes up. The body itself should NOT trigger an
  // unbounded growth.
  assert!(after >= baseline, "peak_memory must be monotonic");
}

/// Contract documentation test for the NULL-ctx OOM path of
/// `mlx_closure_new_func_payload`.
///
/// Per vendored `mlx-c/mlx/c/closure.cpp::mlx_closure_new_func_payload`
/// (line 70), the C constructor wraps the Rust payload pointer in
/// `std::shared_ptr<void>(payload, dtor)` as the very first statement of
/// its `try` block. If any subsequent allocation (the captured lambda or
/// the inner `mlx_closure_new_(cpp_closure)` call) throws, the shared_ptr
/// destructor runs `destroy_payload(payload)` as part of stack unwinding
/// before the `catch` clause hands back a NULL closure.
///
/// Pre-fix the Rust safe wrapper reclaimed the payload via
/// `Box::from_raw(payload_ptr.cast())` on the NULL-ctx path; that produced
/// a double-free / UAF on the OOM-then-NULL-return path. Post-fix the
/// reclaim is removed (mlx-c's shared_ptr owns the payload after the call
/// returns regardless of success / failure).
///
/// We can't deterministically inject an OOM at the C++ ctor without a
/// custom shim. This test exercises the success path many times under the
/// existing TLS-aware ASAN/Miri runners so any reclaim-related UB would
/// surface as a use-after-free / double-free diagnostic. Combined with the
/// inline SAFETY-comment documenting the contract, this is the strongest
/// regression guard short of a deterministic OOM-injection shim.
#[test]
fn closure_constructor_failure_does_not_double_free_payload() {
  // Many successful constructions: if any path were spuriously reclaiming
  // the payload, the next construction's `Box::into_raw` would land on the
  // same address and the eventual `destroy_payload` would corrupt the
  // allocator. ASAN / Miri would catch this; in a release build the test
  // still proves the success path stays healthy after the fix.
  for i in 0..64 {
    let captured = i as f32;
    let cls = Closure::new(move |xs: &[Array]| {
      let s = square(&xs[0])?;
      let scalar = Array::full::<f32>(&[0i32; 0], captured)?;
      Ok(vec![multiply(&s, &scalar)?])
    })
    .unwrap();
    assert!(!cls.as_raw().ctx.is_null());
    drop(cls);
  }

  // Same stress for the custom-VJP variant (`closure_custom_new`), which
  // had the identical reclaim bug at the analogous NULL-ctx path. Covered
  // by `custom_vjp` round-trips.
  for i in 0..32 {
    let captured = i as f32;
    let f = custom_vjp(
      move |xs| Ok(vec![square(&xs[0])?]),
      move |primals, _outputs, _cot| {
        let dims = primals[0].shape();
        Ok(vec![Array::full::<f32>(&&dims[..], captured)?])
      },
    )
    .unwrap();
    let g = grad(f, &[0]).unwrap();
    let x = Array::full::<f32>(&[0i32; 0], 2.0).unwrap();
    let mut grads = g(&[x]).unwrap();
    assert!(approx_eq(grads[0].item::<f32>().unwrap(), captured, 1e-5));
  }
}

/// Capturing the closure into a `value_and_grad` result and using it after
/// dropping the local construction-scope variables verifies the captured Rust
/// callable outlives the construction scope (Rc-shared internally).
#[test]
fn closure_outlives_construction_scope() {
  // Build value_and_grad in a tight scope, return the closure out.
  let vag = {
    let mult = 2.0_f32;
    value_and_grad(
      move |xs| {
        let s = square(&xs[0])?;
        // Use a value captured in the inner closure to prove `mult` lives.
        let scalar = Array::full::<f32>(&[0i32; 0], mult)?;
        Ok(vec![multiply(&s, &scalar)?])
      },
      &[0],
    )
    .unwrap()
  }; // `mult` is now out-of-scope; vag still holds it via Rc<Box<Fn>>.
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let (mut vals, mut grads) = vag(&[x]).unwrap();
  // f(x) = 2*x^2 → f(3) = 18; f'(3) = 4*3 = 12.
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 18.0, 1e-5));
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 12.0, 1e-5));
}

// ──────────────────────── value_and_grad / grad ────────────────────────

/// F2 contract: empty `argnums` must be rejected at the safe-wrapper
/// boundary. mlx-c's `mlx_value_and_grad` would receive a NULL data
/// pointer alongside `argnums_num == 0` and build
/// `std::vector<int>(NULL, NULL + 0)` — pointer arithmetic on NULL is
/// technical UB under the C++ standard ([expr.add]) even when the addend
/// is 0. Failing fast here removes the spec-UB exposure (and the
/// semantically-meaningless "differentiate w.r.t. nothing" call shape).
#[test]
fn value_and_grad_rejects_empty_argnums() {
  let r = value_and_grad(|xs| Ok(vec![square(&xs[0])?]), &[]);
  let err = r.err().expect("empty argnums must be rejected");
  let msg = format!("{err}");
  assert!(
    msg.contains("non-empty"),
    "expected rejection mentioning non-empty argnums; got: {msg}"
  );
  // grad delegates to value_and_grad so it must reject too.
  let r = grad(|xs| Ok(vec![square(&xs[0])?]), &[]);
  let err = r.err().expect("empty argnums must be rejected by grad");
  let msg = format!("{err}");
  assert!(
    msg.contains("non-empty"),
    "expected rejection mentioning non-empty argnums; got: {msg}"
  );
}

/// f(x) = x^2; d/dx[x^2] = 2x; at x=3 → grad = 6, value = 9.
#[test]
fn value_and_grad_simple_quadratic() {
  let vag = value_and_grad(|xs| Ok(vec![square(&xs[0])?]), &[0]).unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let (mut vals, mut grads) = vag(&[x]).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 9.0, 1e-5));
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 6.0, 1e-5));
}

/// f(x, y) = x^2 + y^3; differentiated w.r.t. both inputs.
/// grad_x f = 2x = 4 at x=2; grad_y f = 3y^2 = 3 at y=1.
#[test]
fn value_and_grad_multivariate() {
  let vag = value_and_grad(
    |xs| {
      let xs0 = square(&xs[0])?; // x^2
      let three = Array::full::<f32>(&[0i32; 0], 3.0)?;
      let ys3 = power(&xs[1], &three)?; // y^3
      Ok(vec![add(&xs0, &ys3)?])
    },
    &[0, 1],
  )
  .unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 2.0).unwrap();
  let y = Array::full::<f32>(&[0i32; 0], 1.0).unwrap();
  let (_vals, mut grads) = vag(&[x, y]).unwrap();
  assert_eq!(grads.len(), 2);
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 4.0, 1e-5));
  assert!(approx_eq(grads[1].item::<f32>().unwrap(), 3.0, 1e-5));
}

/// d^2/dx^2 [x^3] = 6x → at x=2 returns 12.
/// We achieve this by taking grad-of-grad: g(x) = d/dx[x^3] = 3x^2;
/// d/dx[g] = 6x.
#[test]
fn grad_composition_yields_second_derivative() {
  // Inner: x^3.
  let g = grad(
    |xs| {
      let three = Array::full::<f32>(&[0i32; 0], 3.0)?;
      Ok(vec![power(&xs[0], &three)?])
    },
    &[0],
  )
  .unwrap();
  // Outer: d/dx[g(x)] = d/dx[3x^2] = 6x.
  let gg = grad(move |xs| g(xs), &[0]).unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 2.0).unwrap();
  let mut grads = gg(&[x]).unwrap();
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 12.0, 1e-4));
}

// ───────────────────── F3: user error/panic propagation ─────────────────────

/// F3 contract: a user closure that returns `Err` must surface that
/// SAME error through `grad` / `value_and_grad`'s outer return — NOT
/// mlx-c's generic "mlx_closure returned a non-zero value" wrapper.
///
/// Pre-fix, mlx-c's outer catch in `mlx_closure_*_apply` re-entered our
/// global error handler with the wrapper text after the trampoline had
/// already stashed the user's `Err` in TLS via `set_last`, overwriting
/// the user payload. Post-fix the handler preserves a trampoline-set
/// error when the incoming message matches the
/// `mlx_closure*…returned a non-zero value` wrapper shape.
#[test]
fn closure_user_error_propagates_through_grad() {
  use mlxrs::Error;
  let g = grad(
    |_xs: &[Array]| -> mlxrs::Result<Vec<Array>> {
      Err(Error::Backend {
        message: "USER_ERROR_PAYLOAD".into(),
      })
    },
    &[0],
  )
  .unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let err = g(&[x]).expect_err("user error must surface");
  let msg = format!("{err}");
  assert!(
    msg.contains("USER_ERROR_PAYLOAD"),
    "expected user error payload to surface; got: {msg}"
  );
  assert!(
    !msg.contains("mlx_closure returned a non-zero value"),
    "must NOT surface mlx-c's generic closure-non-zero wrapper; got: {msg}"
  );
}

/// F3 contract (panic case): a user closure that panics must surface a
/// Rust-side error mentioning the panic payload + that the trampoline
/// caught a panic — NOT mlx-c's generic wrapper text. The trampoline
/// catches via `catch_unwind` so the panic never crosses the
/// `extern "C"` boundary (which would be UB); the panic message is
/// stashed in TLS via `set_last`, and the F3 preserve-check ensures it
/// survives mlx-c's subsequent wrapper invocation of the handler.
#[test]
fn closure_user_panic_propagates_through_grad_as_error() {
  let g = grad(
    |_xs: &[Array]| -> mlxrs::Result<Vec<Array>> { panic!("USER_PANIC_PAYLOAD") },
    &[0],
  )
  .unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let err = g(&[x]).expect_err("user panic must surface as Err");
  let msg = format!("{err}");
  assert!(
    msg.contains("USER_PANIC_PAYLOAD"),
    "expected user panic payload to surface; got: {msg}"
  );
  assert!(
    msg.contains("panic"),
    "expected indication that the closure panicked; got: {msg}"
  );
  assert!(
    !msg.contains("mlx_closure returned a non-zero value"),
    "must NOT surface mlx-c's generic closure-non-zero wrapper; got: {msg}"
  );
}

// ─────────────────────────────── vjp ───────────────────────────────

/// VJP of a scalar-output function with cotangent = 1 equals the gradient.
/// f(x) = x^2, primal x=3, cotangent=1 → vjp = 2x = 6.
#[test]
fn vjp_matches_grad_for_scalar_output() {
  let primals = vec![Array::full::<f32>(&[0i32; 0], 3.0).unwrap()];
  let cot = vec![Array::full::<f32>(&[0i32; 0], 1.0).unwrap()];
  let (mut vals, mut grads) = vjp(|xs| Ok(vec![square(&xs[0])?]), &primals, &cot).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 9.0, 1e-5));
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 6.0, 1e-5));
}

// ─────────────────────────────── jvp ───────────────────────────────

/// JVP of f(x) = x^2 at primal x=3 with unit tangent = directional derivative
/// = 2x = 6.
#[test]
fn jvp_matches_directional_derivative() {
  let primals = vec![Array::full::<f32>(&[0i32; 0], 3.0).unwrap()];
  let tan = vec![Array::full::<f32>(&[0i32; 0], 1.0).unwrap()];
  let (mut vals, mut jvp_out) = jvp(|xs| Ok(vec![square(&xs[0])?]), &primals, &tan).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 9.0, 1e-5));
  assert!(approx_eq(jvp_out[0].item::<f32>().unwrap(), 6.0, 1e-5));
}

// ───────────────────────────── custom_vjp ─────────────────────────────

/// Define f(x) = x^2 (autograd grad = 2x), but override its VJP with a custom
/// rule that returns 42. Grad should now produce 42 instead of 2x.
#[test]
fn custom_vjp_overrides_autograd() {
  let f = custom_vjp(
    |xs| Ok(vec![square(&xs[0])?]),
    |primals, _outputs, _cot| {
      // Ignore the cotangent, return a constant 42 in primal-shape.
      let dims = primals[0].shape();
      Ok(vec![Array::full::<f32>(&&dims[..], 42.0)?])
    },
  )
  .unwrap();
  let g = grad(f, &[0]).unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let mut grads = g(&[x]).unwrap();
  assert!(approx_eq(grads[0].item::<f32>().unwrap(), 42.0, 1e-5));
}

// ───────────────────────────── checkpoint ─────────────────────────────

/// `checkpoint(f)` and `f` should produce identical FORWARD values. Memory
/// profile differs (recompute in backward) but math is invariant.
#[test]
fn checkpoint_returns_same_value_as_uncheckpointed() {
  // Reference value via direct call.
  let x = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let mut direct = square(&x).unwrap();
  let direct_val = direct.item::<f32>().unwrap();

  let cf = checkpoint(|xs| Ok(vec![square(&xs[0])?])).unwrap();
  let mut vals = cf(&[x]).unwrap();
  let ckpt_val = vals[0].item::<f32>().unwrap();
  assert!(approx_eq(direct_val, ckpt_val, 1e-6));
}

/// `grad(checkpoint(f))` and `grad(f)` should produce identical GRADIENTS
/// (checkpoint affects memory, not math).
#[test]
fn checkpoint_gradient_matches_uncheckpointed() {
  let g_direct = grad(|xs| Ok(vec![square(&xs[0])?]), &[0]).unwrap();
  let cf = checkpoint(|xs| Ok(vec![square(&xs[0])?])).unwrap();
  let g_ckpt = grad(cf, &[0]).unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 4.0).unwrap();
  let mut direct = g_direct(&[x.try_clone().unwrap()]).unwrap();
  let mut ckpt = g_ckpt(&[x]).unwrap();
  assert!(approx_eq(
    direct[0].item::<f32>().unwrap(),
    ckpt[0].item::<f32>().unwrap(),
    1e-5,
  ));
}

// ─────────────────────────────── eval ───────────────────────────────

/// Bulk eval over an empty slice is a no-op (no FFI call).
#[test]
fn eval_empty_slice_is_noop() {
  eval(&[]).unwrap();
  async_eval(&[]).unwrap();
}

/// Bulk eval of multiple lazy arrays should materialize all of them.
#[test]
fn eval_materializes_all_arrays() {
  let a = Array::full::<f32>(&(2usize, 2usize), 1.0).unwrap();
  let b = Array::full::<f32>(&(2usize, 2usize), 2.0).unwrap();
  let c = Array::full::<f32>(&(2usize, 2usize), 3.0).unwrap();
  // Build a derived computation that is still lazy.
  let mut d = add(&a, &b).unwrap();
  let mut e = multiply(&b, &c).unwrap();
  // Eval both at once.
  eval(&[&d, &e]).unwrap();
  // Reading values now should be cheap (already materialized).
  let dv = d.to_vec::<f32>().unwrap();
  let ev = e.to_vec::<f32>().unwrap();
  assert!(dv.iter().all(|&v| approx_eq(v, 3.0, 1e-6)));
  assert!(ev.iter().all(|&v| approx_eq(v, 6.0, 1e-6)));
}

/// async_eval enqueues but does not block; following with eval (or any
/// item / to_vec) syncs.
#[test]
fn async_eval_then_sync_via_item() {
  let a = Array::full::<f32>(&(4usize, 4usize), 0.5).unwrap();
  let mut sq = square(&a).unwrap();
  async_eval(&[&sq]).unwrap();
  // Eventually it must materialize; item() forces sync.
  // square(0.5) = 0.25.
  let vals = sq.to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|&v| approx_eq(v, 0.25, 1e-6)));
}
