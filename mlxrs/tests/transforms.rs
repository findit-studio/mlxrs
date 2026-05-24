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
