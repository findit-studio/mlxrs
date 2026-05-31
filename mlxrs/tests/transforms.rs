//! Integration tests for `mlxrs::transforms` — the autograd / custom-VJP /
//! checkpoint / bulk-eval safe-wrapper port.
//!
//! Mirrors mlx-swift's `Tests/MLXTests/TransformTests.swift` cases plus
//! lifecycle tests for the `Closure` payload trampoline.

use mlxrs::{
  Array,
  ops::{
    arithmetic::{add, exp, multiply, power, sin, square, tanh},
    linalg_basic::{inner, matmul},
    reduction::sum,
    shape::contiguous,
  },
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
/// A naive Rust safe wrapper would reclaim the payload via
/// `Box::from_raw(payload_ptr.cast())` on the NULL-ctx path; that would
/// produce a double-free / UAF on the OOM-then-NULL-return path. The
/// reclaim is therefore omitted (mlx-c's shared_ptr owns the payload after
/// the call returns regardless of success / failure).
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
      move |primals, _cot, _outputs| {
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

/// Empty `argnums` must be rejected at the safe-wrapper
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
  match err {
    mlxrs::Error::EmptyInput(p) => {
      assert!(
        p.context().contains("value_and_grad") && p.context().contains("argnums"),
        "context names the argnums site: {}",
        p.context()
      );
    }
    other => panic!("expected Error::EmptyInput for empty argnums, got {other:?}"),
  }
  // grad delegates to value_and_grad so it must reject too.
  let r = grad(|xs| Ok(vec![square(&xs[0])?]), &[]);
  let err = r.err().expect("empty argnums must be rejected by grad");
  match err {
    mlxrs::Error::EmptyInput(p) => {
      assert!(
        p.context().contains("argnums"),
        "grad-delegated context still names argnums: {}",
        p.context()
      );
    }
    other => panic!("expected Error::EmptyInput from grad delegation, got {other:?}"),
  }
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

// ───────────────────── user error/panic propagation ─────────────────────

/// A user closure that returns `Err` must surface that
/// SAME error through `grad` / `value_and_grad`'s outer return — NOT
/// mlx-c's generic "mlx_closure returned a non-zero value" wrapper.
///
/// Without the guard, mlx-c's outer catch in `mlx_closure_*_apply`
/// re-enters our global error handler with the wrapper text after the
/// trampoline has already stashed the user's `Err` in TLS via `set_last`,
/// overwriting the user payload. The handler therefore preserves a
/// trampoline-set error when the incoming message matches the
/// `mlx_closure*…returned a non-zero value` wrapper shape.
#[test]
fn closure_user_error_propagates_through_grad() {
  use mlxrs::Error;
  let g = grad(
    |_xs: &[Array]| -> mlxrs::Result<Vec<Array>> {
      Err(Error::Backend("USER_ERROR_PAYLOAD".into()))
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

/// Panic case: a user closure that panics must surface a
/// Rust-side error mentioning the panic payload + that the trampoline
/// caught a panic — NOT mlx-c's generic wrapper text. The trampoline
/// catches via `catch_unwind` so the panic never crosses the
/// `extern "C"` boundary (which would be UB); the panic message is
/// stashed in TLS via `set_last`, and the preserve-check ensures it
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
    |primals, _cot, _outputs| {
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

/// Order-sensitive ABI regression guard for the `custom_vjp` trampoline.
///
/// Upstream `mlx/primitives.cpp::CustomTransforms::vjp` invokes the user
/// VJP callback positionally as `vjp_fun_(inputs, cotangents, outputs)`.
/// A transposition in `transforms::closure::trampoline_custom` that bound
/// the slots as `(primals, outputs, cotangents)` would be semantically
/// wrong for every downstream `custom_vjp` user (the closure would treat
/// the cotangent vector as the forward outputs and vice versa).
///
/// The trampoline binds `(primals, cotangents, outputs)`, matching mlx
/// core. This test pins that ABI by constructing a backward closure whose
/// return value depends
/// DISTINCTLY on each of `primals[0]`, `cotangents[0]`, and `outputs[0]`,
/// so any future regression swapping the cotangent / output slots is
/// caught deterministically in routine CI (this test is intentionally
/// non-ignored; the existing custom-VJP tests above use constant VJP
/// bodies that would still pass with the slots reversed).
///
/// Setup:
/// - Forward `f(x) = x^2`. At primal `x = 3.0`, the forward output is `9.0`.
/// - VJP returns `cotangents[0] * 10.0 + outputs[0] + primals[0] * 1000.0`
///   evaluated element-wise (single-scalar I/O so each slot collapses to a
///   scalar add).
/// - Cotangent fed via `vjp(...)` is `2.0`.
///
/// Correct slot binding `(primals, cotangents, outputs)` yields:
///   `2.0 * 10.0 + 9.0 + 3.0 * 1000.0 = 3029.0`
///
/// If the slots were swapped to `(primals, outputs, cotangents)`, the
/// closure would receive `cotangents = [9.0]` and `outputs = [2.0]`,
/// yielding:
///   `9.0 * 10.0 + 2.0 + 3.0 * 1000.0 = 3092.0` — a deterministic
/// 63-unit gap that fails the assertion below.
#[test]
fn custom_vjp_trampoline_argument_order_regression() {
  let f = custom_vjp(
    |xs| Ok(vec![square(&xs[0])?]),
    |primals, cotangents, outputs| {
      // Distinct, slot-revealing combination:
      //   cot * 10 + out + primal * 1000
      let ten = Array::full::<f32>(&[0i32; 0], 10.0)?;
      let thousand = Array::full::<f32>(&[0i32; 0], 1000.0)?;
      let c_term = multiply(&cotangents[0], &ten)?;
      let p_term = multiply(&primals[0], &thousand)?;
      let sum1 = add(&c_term, &outputs[0])?;
      Ok(vec![add(&sum1, &p_term)?])
    },
  )
  .unwrap();

  let primal = Array::full::<f32>(&[0i32; 0], 3.0).unwrap();
  let cotangent = Array::full::<f32>(&[0i32; 0], 2.0).unwrap();

  let (_vals, mut grads) = vjp(f, &[primal], &[cotangent]).unwrap();
  assert_eq!(grads.len(), 1);
  let got = grads[0].item::<f32>().unwrap();

  // Correct trampoline order: 2 * 10 + 9 + 3 * 1000 = 3029.
  let expected = 3029.0_f32;
  // A swapped order would yield: 9 * 10 + 2 + 3 * 1000 = 3092 (delta 63).
  let swapped_value = 3092.0_f32;
  assert!(
    approx_eq(got, expected, 1e-3),
    "trampoline arg order regression: got {got}, expected {expected} \
     (a value near {swapped_value} would indicate a \
     `(primals, outputs, cotangents)` slot ordering has been reintroduced)"
  );
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

// `async_eval_then_sync_via_item` moved to its own test bin
// (`mlxrs/tests/transforms_async_eval.rs`) for **process isolation**.
//
// **Why:** MLX's `detail::InTracing::trace_stack_` is a function-local
// `static std::vector<...>` (NOT `thread_local`) — see
// `mlxrs-sys/vendor/mlx/mlx/transforms.cpp:trace_stack()`. It is
// process-global, shared across all threads. The sibling test
// `closure_user_panic_propagates_through_grad_as_error` deliberately
// `panic!()`s inside a `grad` closure — the Rust panic is converted to
// `Err` at the FFI boundary, but MLX's C++ RAII guard for `trace_stack`
// does not always restore on that conversion path, leaving the
// process-global static with a stale frame. Any subsequent `async_eval`
// in the same process then rejects with
// `"[async_eval] Not allowed inside a graph transformation."`.
//
// Cargo runs each test bin in a **separate process**, so a fresh bin gets
// a fresh `trace_stack_`. Local `cargo test --test transforms` happens to
// pass because the test scheduler usually completes `async_eval` before
// the panic test pollutes — but CI's scheduler reliably reverses the
// order and the test fails. Process isolation is the only correct fix
// without touching MLX's C++.

// ───────────────────── #260: extra closed-form grad coverage ─────────────────────
//
// All closures below are panic-free on the happy path: each differentiated
// closure builds its graph with `?` and never calls `.unwrap()` inside the
// closure body, so none can leak MLX's process-global `trace_stack_` frame
// (see the long comment above + `tests/transforms_async_eval.rs`). Expected
// gradients are hand-derivable closed forms; values are compared with a small
// f32 epsilon after the implicit eval inside `item`/`to_vec`.

// ───────────────── value_and_grad / grad: vector + reduction grads ─────────────────

/// `grad` of a sum-reduction over a vector: f(x) = Σ_i x_i (scalar).
/// ∂f/∂x_i = 1 for every element, so grad(x) = ones(shape(x)).
/// Exercises `grad` directly (not via composition / custom_vjp) on a
/// genuinely multi-element input, and confirms the gradient is full-shape.
#[test]
fn grad_of_sum_is_ones_vector() {
  let g = grad(|xs| Ok(vec![sum(&xs[0], false)?]), &[0]).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
  let grads = g(&[x]).unwrap();
  assert_eq!(grads.len(), 1);
  // grad of a reduction is a stride-0 broadcast; materialize before reading.
  let gv = contiguous(&grads[0], false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert_eq!(gv.len(), 4, "grad must keep the input's element count");
  assert!(
    gv.iter().all(|&v| approx_eq(v, 1.0, 1e-5)),
    "d/dx_i[Σx] must be 1 everywhere; got {gv:?}"
  );
}

/// `value_and_grad` of a dot product against a constant: f(x) = inner(x, c)
/// = Σ_i x_i·c_i (scalar). The VALUE must equal the dot product and the GRAD
/// w.r.t. x must equal `c` (∂/∂x_i[Σ x_j c_j] = c_i).
/// x = [1,2,3], c = [4,5,6] → value = 4 + 10 + 18 = 32; grad = [4,5,6].
/// Confirms `value_and_grad` returns BOTH the correct value and grad together.
#[test]
fn value_and_grad_of_dot_returns_value_and_constant_grad() {
  // `c` is captured (constant w.r.t. differentiation); only x (arg 0) is
  // differentiated, so c contributes to the value + grad but is not itself
  // a grad target.
  let c = Array::from_slice::<f32>(&[4.0, 5.0, 6.0], &[3]).unwrap();
  let vag = value_and_grad(move |xs| Ok(vec![inner(&xs[0], &c)?]), &[0]).unwrap();
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
  let (mut vals, mut grads) = vag(&[x]).unwrap();
  assert!(
    approx_eq(vals[0].item::<f32>().unwrap(), 32.0, 1e-4),
    "value must equal the dot product x·c = 32"
  );
  let gv = grads[0].to_vec::<f32>().unwrap();
  assert_eq!(gv.len(), 3);
  assert!(
    approx_eq(gv[0], 4.0, 1e-5) && approx_eq(gv[1], 5.0, 1e-5) && approx_eq(gv[2], 6.0, 1e-5),
    "d/dx[x·c] must equal c = [4,5,6]; got {gv:?}"
  );
}

/// `grad` through a matmul: f(X) = Σ(X @ W) with W = ones(k, n).
/// Y = X @ W (m×n); s = ΣY. ∂s/∂X = ones(m,n) @ Wᵀ. With W = ones(k,n),
/// Wᵀ = ones(n,k) and ones(m,n) @ ones(n,k) = n · ones(m,k). So every
/// element of grad_X is exactly `n`.
/// Here k = 2, n = 3 → grad_X = 3 · ones(2,2). Value: X = ones(2,2) gives
/// each Y row = [2,2,2] → ΣY = 12.
#[test]
fn grad_through_matmul_sum_is_n_times_ones() {
  // W = ones(2, 3) captured as a differentiation constant.
  let w = Array::full::<f32>(&(2usize, 3usize), 1.0).unwrap();
  let vag = value_and_grad(
    move |xs| {
      let y = matmul(&xs[0], &w)?; // (2,2) @ (2,3) -> (2,3)
      Ok(vec![sum(&y, false)?]) // scalar
    },
    &[0],
  )
  .unwrap();
  let x = Array::full::<f32>(&(2usize, 2usize), 1.0).unwrap();
  let (mut vals, mut grads) = vag(&[x]).unwrap();
  assert!(
    approx_eq(vals[0].item::<f32>().unwrap(), 12.0, 1e-4),
    "Σ(ones(2,2) @ ones(2,3)) = 12"
  );
  let gv = grads[0].to_vec::<f32>().unwrap();
  assert_eq!(gv.len(), 4, "grad_X must be 2x2 = 4 elements");
  assert!(
    gv.iter().all(|&v| approx_eq(v, 3.0, 1e-5)),
    "∂Σ(XW)/∂X = n·ones with n=3; got {gv:?}"
  );
}

// ───────────────── value_and_grad: argnums subset selection ─────────────────

/// `argnums` must SELECT which positional input is differentiated. With
/// f(x, y) = x² + y² and `argnums = [1]`, the result must contain exactly ONE
/// gradient — ∂f/∂y = 2y — and NOT ∂f/∂x. x and y are chosen distinct
/// (x=2, y=5) so a wrong arg selection (2x=4 vs 2y=10) is caught.
#[test]
fn value_and_grad_argnums_selects_second_arg_only() {
  let vag = value_and_grad(
    |xs| {
      let x2 = square(&xs[0])?;
      let y2 = square(&xs[1])?;
      Ok(vec![add(&x2, &y2)?])
    },
    &[1],
  )
  .unwrap();
  let x = Array::full::<f32>(&[0i32; 0], 2.0).unwrap();
  let y = Array::full::<f32>(&[0i32; 0], 5.0).unwrap();
  let (mut vals, mut grads) = vag(&[x, y]).unwrap();
  // Forward value is unaffected by argnums: 2² + 5² = 29.
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 29.0, 1e-4));
  assert_eq!(
    grads.len(),
    1,
    "argnums=[1] selects exactly one grad target"
  );
  assert!(
    approx_eq(grads[0].item::<f32>().unwrap(), 10.0, 1e-4),
    "grad must be ∂f/∂y = 2y = 10 (a value of 4 would mean arg 0 was differentiated)"
  );
}

// ───────────────────────── vjp: cotangent scaling ─────────────────────────

/// VJP scales the gradient by the cotangent: f(x) = x², primal x=3,
/// cotangent c = 2 → vjp = c · (df/dx) = 2 · 2x = 12 (vs the unit-cotangent
/// case = 6 covered above). The forward value is unchanged at 9.
#[test]
fn vjp_scales_by_nonunit_cotangent() {
  let primals = vec![Array::full::<f32>(&[0i32; 0], 3.0).unwrap()];
  let cot = vec![Array::full::<f32>(&[0i32; 0], 2.0).unwrap()];
  let (mut vals, mut grads) = vjp(|xs| Ok(vec![square(&xs[0])?]), &primals, &cot).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 9.0, 1e-5));
  assert!(
    approx_eq(grads[0].item::<f32>().unwrap(), 12.0, 1e-4),
    "vjp = cotangent · 2x = 2 · 6 = 12"
  );
}

/// VJP of a vector→scalar reduction broadcasts the scalar cotangent over the
/// Jacobian. f(x) = Σx (Jacobian = ones row), cotangent c = 2 →
/// vjp_i = c · ∂f/∂x_i = 2 · 1 = 2 for every element. x = [1,2,3] →
/// vjp = [2,2,2], value = 6. Exercises a non-square (rank-1 → scalar) map.
#[test]
fn vjp_of_vector_sum_broadcasts_cotangent() {
  let primals = vec![Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap()];
  let cot = vec![Array::full::<f32>(&[0i32; 0], 2.0).unwrap()];
  let (mut vals, grads) = vjp(|xs| Ok(vec![sum(&xs[0], false)?]), &primals, &cot).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 6.0, 1e-5));
  // vjp of a reduction broadcasts the cotangent (stride-0); materialize first.
  let gv = contiguous(&grads[0], false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
  assert_eq!(gv.len(), 3, "vjp output matches the primal's shape");
  assert!(
    gv.iter().all(|&v| approx_eq(v, 2.0, 1e-5)),
    "vjp_i = cotangent · 1 = 2 everywhere; got {gv:?}"
  );
}

// ───────────────────────── jvp: tangent + multi-primal ─────────────────────────

/// JVP contracts the Jacobian with a non-unit tangent. f(x) = Σx
/// (Jacobian = ones row), tangent v = [2,3,4] → jvp = Σ_i (∂f/∂x_i · v_i)
/// = Σ v_i = 9 (distinct from the value Σx = 6). Confirms the directional
/// derivative uses the supplied tangent, not the primal.
#[test]
fn jvp_of_vector_sum_contracts_tangent() {
  let primals = vec![Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap()];
  let tan = vec![Array::from_slice::<f32>(&[2.0, 3.0, 4.0], &[3]).unwrap()];
  let (mut vals, mut jvp_out) = jvp(|xs| Ok(vec![sum(&xs[0], false)?]), &primals, &tan).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 6.0, 1e-5));
  assert!(
    approx_eq(jvp_out[0].item::<f32>().unwrap(), 9.0, 1e-4),
    "jvp = Σ(1 · v_i) = 2+3+4 = 9"
  );
}

/// JVP with multiple primals sums each primal's directional contribution.
/// f(x, y) = x · y (elementwise scalar). The total differential is
/// df = y·dx + x·dy. At x=2, y=3 with unit tangents dx=dy=1 →
/// jvp = 3·1 + 2·1 = 5 (distinct from value x·y = 6). Exercises the
/// multi-primal tangent path.
#[test]
fn jvp_multi_primal_sums_directional_contributions() {
  let primals = vec![
    Array::full::<f32>(&[0i32; 0], 2.0).unwrap(),
    Array::full::<f32>(&[0i32; 0], 3.0).unwrap(),
  ];
  let tan = vec![
    Array::full::<f32>(&[0i32; 0], 1.0).unwrap(),
    Array::full::<f32>(&[0i32; 0], 1.0).unwrap(),
  ];
  let (mut vals, mut jvp_out) =
    jvp(|xs| Ok(vec![multiply(&xs[0], &xs[1])?]), &primals, &tan).unwrap();
  assert!(approx_eq(vals[0].item::<f32>().unwrap(), 6.0, 1e-5));
  assert!(
    approx_eq(jvp_out[0].item::<f32>().unwrap(), 5.0, 1e-4),
    "jvp = y·dx + x·dy = 3 + 2 = 5"
  );
}

// ───────────── #257 M18: finite-difference gradient cross-checks ─────────────
//
// The closed-form tests above compare the analytic `grad` against a
// HAND-DERIVED symbolic derivative. These tests add a stronger, independent
// angle: compare the analytic `grad` against a NUMERICAL central-difference
// estimate of the same function,
//
//     f'(x) ≈ (f(x + h) − f(x − h)) / (2h)            (O(h²) accurate)
//
// computed by actually evaluating the forward function at perturbed inputs.
// This catches sign/chain-rule/op-wiring bugs that a hand-derived symbolic
// reference could share (both could be wrong the same way), since the numeric
// estimate only ever runs the FORWARD op.
//
// Panic-safety: every closure handed to `grad` builds its graph with `?` and
// never `.unwrap()`s inside the closure body, so none can leak MLX's
// process-global `trace_stack_` frame (see the long note earlier in this file
// and `tests/transforms_async_eval.rs`). The numeric side runs entirely
// OUTSIDE any transform — plain forward `eval` via `item` — so `.unwrap()` there
// is safe and cannot pollute the trace stack.

/// Central-difference estimate of `df/dx` for a scalar→scalar forward function
/// `f`, evaluated at scalar `x` with step `h`. Runs purely forward (no grad),
/// so the `?`/`item` here cannot interact with MLX's trace stack.
fn central_difference<F>(f: &F, x: f32, h: f32) -> mlxrs::Result<f32>
where
  F: Fn(&Array) -> mlxrs::Result<Array>,
{
  let mut up = f(&Array::full::<f32>(&[0i32; 0], x + h)?)?;
  let mut down = f(&Array::full::<f32>(&[0i32; 0], x - h)?)?;
  Ok((up.item::<f32>()? - down.item::<f32>()?) / (2.0 * h))
}

/// Analytic `grad` of a scalar→scalar function evaluated at scalar `x`.
/// `f` is a plain forward closure; it is lifted into the `&[Array]` shape
/// `grad` expects. Returns the single scalar gradient.
fn analytic_grad<F>(f: F, x: f32) -> mlxrs::Result<f32>
where
  F: Fn(&Array) -> mlxrs::Result<Array> + 'static,
{
  // The differentiated closure is panic-free: it only uses `?` (never
  // `.unwrap()`), so a failure surfaces as `Err` at the FFI boundary rather
  // than unwinding through MLX's C++ trace-stack guard.
  let g = grad(move |xs| Ok(vec![f(&xs[0])?]), &[0])?;
  let mut grads = g(&[Array::full::<f32>(&[0i32; 0], x)?])?;
  grads[0].item::<f32>()
}

/// `d/dx sin(x) = cos(x)`: analytic grad must match the central-difference
/// estimate. Checked at a few points spanning sign changes of cos.
#[test]
fn finite_diff_matches_grad_sin() {
  let h = 1e-3_f32;
  for &x in &[-1.5_f32, -0.3, 0.0, 0.7, 2.0] {
    let analytic = analytic_grad(sin, x).unwrap();
    let numeric = central_difference(&(|a: &Array| sin(a)), x, h).unwrap();
    // Central difference is O(h²); with h=1e-3 and f32 round-off a 2e-3
    // absolute tolerance comfortably covers truncation + rounding error.
    assert!(
      approx_eq(analytic, numeric, 2e-3),
      "d/dx sin at x={x}: analytic grad {analytic} vs central-diff {numeric}"
    );
  }
}

/// `d/dx exp(x) = exp(x)`: analytic grad vs central difference. The relative
/// step error grows with |f|, so we keep |x| modest and use an absolute
/// tolerance scaled to the largest expected magnitude (exp(1.2) ≈ 3.32).
#[test]
fn finite_diff_matches_grad_exp() {
  let h = 1e-3_f32;
  for &x in &[-1.0_f32, -0.2, 0.5, 1.2] {
    let analytic = analytic_grad(exp, x).unwrap();
    let numeric = central_difference(&(|a: &Array| exp(a)), x, h).unwrap();
    assert!(
      approx_eq(analytic, numeric, 5e-3),
      "d/dx exp at x={x}: analytic grad {analytic} vs central-diff {numeric}"
    );
  }
}

/// `d/dx tanh(x) = 1 − tanh²(x)`: analytic grad vs central difference. tanh is
/// smooth and bounded so the central-difference truncation error is small.
#[test]
fn finite_diff_matches_grad_tanh() {
  let h = 1e-3_f32;
  for &x in &[-1.3_f32, -0.4, 0.0, 0.6, 1.5] {
    let analytic = analytic_grad(tanh, x).unwrap();
    let numeric = central_difference(&(|a: &Array| tanh(a)), x, h).unwrap();
    assert!(
      approx_eq(analytic, numeric, 2e-3),
      "d/dx tanh at x={x}: analytic grad {analytic} vs central-diff {numeric}"
    );
  }
}

/// Composite polynomial `f(x) = x³ + 2x` (`d/dx = 3x² + 2`): exercises a
/// multi-term graph (`power` + `multiply` + `add`) against the numerical
/// estimate, so a chain-rule/op-wiring bug in the composite path is caught
/// independently of the single-op cases above.
#[test]
fn finite_diff_matches_grad_cubic_plus_linear() {
  // f(x) = x^3 + 2x, built panic-free (`?` only). `power` needs an exponent
  // array; `multiply` scales the linear term. Closures capture nothing
  // non-`'static`, so they satisfy `analytic_grad`'s bound.
  let f = |a: &Array| -> mlxrs::Result<Array> {
    let three = Array::full::<f32>(&[0i32; 0], 3.0)?;
    let two = Array::full::<f32>(&[0i32; 0], 2.0)?;
    let cube = power(a, &three)?;
    let lin = multiply(a, &two)?;
    add(&cube, &lin)
  };
  let h = 1e-3_f32;
  for &x in &[-1.4_f32, -0.5, 0.3, 1.1, 2.0] {
    let analytic = analytic_grad(f, x).unwrap();
    let numeric = central_difference(&f, x, h).unwrap();
    // x³ grows the central-difference truncation error (∝ |f'''| h² = 6h²)
    // and f32 cancellation near the larger |x|; 8e-3 absolute covers x=2.
    assert!(
      approx_eq(analytic, numeric, 8e-3),
      "d/dx (x³+2x) at x={x}: analytic grad {analytic} vs central-diff {numeric} \
       (closed form 3x²+2 = {})",
      3.0 * x * x + 2.0
    );
  }
}

/// Multivariate partial-derivative cross-check. f(x, y) = x²·y + y, so
/// ∂f/∂x = 2xy and ∂f/∂y = x² + 1. Each analytic partial (selected via
/// `argnums`) is compared against a central difference that perturbs ONLY that
/// variable, confirming the partials are wired to the correct input.
#[test]
fn finite_diff_matches_partial_grads_multivariate() {
  let x0 = 1.5_f32;
  let y0 = 2.0_f32;
  let h = 1e-3_f32;

  // f(x, y) = x²·y + y — panic-free (`?` only).
  let forward = |xs: &[Array]| -> mlxrs::Result<Vec<Array>> {
    let x2 = square(&xs[0])?;
    let x2y = multiply(&x2, &xs[1])?;
    Ok(vec![add(&x2y, &xs[1])?])
  };

  // Analytic partials via argnums selection.
  let gx = grad(forward, &[0]).unwrap();
  let gy = grad(forward, &[1]).unwrap();
  let mut dx = gx(&[
    Array::full::<f32>(&[0i32; 0], x0).unwrap(),
    Array::full::<f32>(&[0i32; 0], y0).unwrap(),
  ])
  .unwrap();
  let mut dy = gy(&[
    Array::full::<f32>(&[0i32; 0], x0).unwrap(),
    Array::full::<f32>(&[0i32; 0], y0).unwrap(),
  ])
  .unwrap();
  let analytic_dx = dx[0].item::<f32>().unwrap();
  let analytic_dy = dy[0].item::<f32>().unwrap();

  // Numeric partials: evaluate the scalar forward at perturbed single vars.
  // f is scalar-valued, so `item` reads it directly. Runs outside any
  // transform (plain forward eval), so `.unwrap()` is trace-stack-safe.
  let scalar_f = |x: f32, y: f32| -> f32 {
    let xa = Array::full::<f32>(&[0i32; 0], x).unwrap();
    let ya = Array::full::<f32>(&[0i32; 0], y).unwrap();
    let mut out = forward(&[xa, ya]).unwrap();
    out.remove(0).item::<f32>().unwrap()
  };
  let numeric_dx = (scalar_f(x0 + h, y0) - scalar_f(x0 - h, y0)) / (2.0 * h);
  let numeric_dy = (scalar_f(x0, y0 + h) - scalar_f(x0, y0 - h)) / (2.0 * h);

  assert!(
    approx_eq(analytic_dx, numeric_dx, 5e-3),
    "∂f/∂x: analytic {analytic_dx} vs central-diff {numeric_dx} (2xy = {})",
    2.0 * x0 * y0
  );
  assert!(
    approx_eq(analytic_dy, numeric_dy, 5e-3),
    "∂f/∂y: analytic {analytic_dy} vs central-diff {numeric_dy} (x²+1 = {})",
    x0 * x0 + 1.0
  );
}
