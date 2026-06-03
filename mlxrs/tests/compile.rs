//! Integration tests for `mlxrs::transforms::compile` — the safe `mx.compile`
//! graph-compilation wrapper.
//!
//! The parity tests assert the central contract: a compiled function produces
//! output *identical* to the same function run uncompiled (compilation is a
//! performance optimization, never a numeric change). They build genuinely
//! multi-element inputs and compare element-wise after the explicit `to_vec`
//! materialization.
//!
//! ## Why an integration (separate-binary) test
//!
//! These run as a dedicated test binary — not in-crate `#[cfg(test)]` unit
//! tests — for **process isolation**, the same discipline as
//! `tests/transforms_async_eval.rs`:
//!
//! * In a `#[cfg(test)]` crate build, `Closure::new` routes its constructor
//!   call through the `transforms::closure::test_seam` swappable function
//!   pointer. The closure NULL-ctx regression tests install a NULL-returning
//!   stub into that global pointer for their duration; a concurrent
//!   `compile()` (which builds a source `Closure`) would then observe the
//!   stub and spuriously fail. An integration binary links the **non-test**
//!   build of `mlxrs`, so `Closure::new` calls the real FFI directly — no
//!   seam, no race.
//! * MLX's `detail::InTracing::trace_stack_` is a process-global (not
//!   thread-local) static; a sibling transforms unit test that panics inside
//!   a grad closure can leave a stale frame that makes any later eval-during-
//!   trace (which compilation performs) reject. A separate process starts
//!   with a clean `trace_stack_`.
//!
//! The compile-*mode* test additionally serializes its process-global
//! enable/disable + mode toggles with a local mutex and restores the default
//! (compilation enabled, `Enabled` mode) before returning.

use std::sync::{
  Arc, Barrier, Mutex, MutexGuard,
  atomic::{AtomicUsize, Ordering},
};

use mlxrs::{
  Array,
  ops::arithmetic::{add, multiply, square},
  transforms::{
    CompileMode, compile, compile_fn, disable_compile, enable_compile, set_compile_mode,
  },
};

/// Serialize the compile-mode test's process-global toggles. The parity tests
/// deliberately do NOT take this lock: a compiled function stays numerically
/// correct whether or not compilation is globally enabled, so their assertions
/// hold under any concurrent mode state.
fn mode_guard() -> MutexGuard<'static, ()> {
  static MODE: Mutex<()> = Mutex::new(());
  MODE.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
  (a - b).abs() <= eps
}

// ──────────────── tracing / side-effect semantics ────────────────

/// A compiled function's Rust-level side effects run only while the graph is
/// traced (the first call, plus shape/dtype re-traces) — never on a cache hit.
/// A captured counter therefore increments exactly once across two same-shape
/// calls, pinning the documented "compile pure functions only" contract.
#[test]
fn rust_side_effects_run_on_trace_not_on_cache_hit() {
  let _mode = mode_guard();
  enable_compile().unwrap();

  let trace_count = Arc::new(AtomicUsize::new(0));
  let counter = Arc::clone(&trace_count);
  let f = move |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
    counter.fetch_add(1, Ordering::SeqCst);
    Ok(vec![square(&a[0])?])
  };

  let compiled = compile(f, false).unwrap();
  let inputs = [Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
  let mut o1 = compiled.call(&inputs).unwrap(); // cache miss → trace → runs f
  o1[0].to_vec::<f32>().unwrap();
  let mut o2 = compiled.call(&inputs).unwrap(); // same shape → cache hit → f not re-run
  o2[0].to_vec::<f32>().unwrap();

  assert_eq!(
    trace_count.load(Ordering::SeqCst),
    1,
    "f's Rust body runs once (the trace), never on a cache hit",
  );
}

// ─────────────────────── numeric parity ───────────────────────

/// A compiled element-wise `f(x) = x*x + x` must match the uncompiled result
/// element-for-element — the core "compile is not a numeric change" contract
/// on a multi-element input.
#[test]
fn compiled_matches_uncompiled_unary() {
  let xs = [1.0f32, 2.0, 3.0, -4.0, 0.5];
  let f = |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
    let sq = square(&a[0])?;
    Ok(vec![add(&sq, &a[0])?])
  };

  let compiled = compile(f, false).unwrap();
  let x = Array::from_slice::<f32>(&xs, &[5]).unwrap();
  let mut out = compiled.call(&[x]).unwrap();
  let got = out[0].to_vec::<f32>().unwrap();

  // Reference: same function, run directly (uncompiled).
  let x_ref = Array::from_slice::<f32>(&xs, &[5]).unwrap();
  let mut want = f(&[x_ref]).unwrap();
  let want = want[0].to_vec::<f32>().unwrap();

  assert_eq!(got.len(), xs.len());
  for (g, w) in got.iter().zip(want.iter()) {
    assert!(approx_eq(*g, *w, 1e-6), "compiled {g} != uncompiled {w}");
  }
  // And the closed form, independently: x*x + x.
  for (g, &v) in got.iter().zip(xs.iter()) {
    assert!(
      approx_eq(*g, v * v + v, 1e-6),
      "{g} != closed-form {}",
      v * v + v
    );
  }
}

/// A two-input compiled function `f(a, b) = a*b + b` must match uncompiled,
/// confirming multi-input marshalling through `mlx_closure_apply`.
#[test]
fn compiled_matches_uncompiled_binary() {
  let av = [1.0f32, 2.0, 3.0];
  let bv = [4.0f32, 5.0, 6.0];
  let f = |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
    let prod = multiply(&a[0], &a[1])?;
    Ok(vec![add(&prod, &a[1])?])
  };

  let compiled = compile(f, false).unwrap();
  let a = Array::from_slice::<f32>(&av, &[3]).unwrap();
  let b = Array::from_slice::<f32>(&bv, &[3]).unwrap();
  let mut out = compiled.call(&[a, b]).unwrap();
  let got = out[0].to_vec::<f32>().unwrap();

  for (i, g) in got.iter().enumerate() {
    let w = av[i] * bv[i] + bv[i];
    assert!(approx_eq(*g, w, 1e-6), "compiled {g} != {w}");
  }
}

/// A compiled function returning *multiple* outputs round-trips every output
/// vector (exercises the multi-element output drain path).
#[test]
fn compiled_multiple_outputs() {
  let xs = [2.0f32, 3.0];
  let compiled = compile(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![square(&a[0])?, add(&a[0], &a[0])?]) },
    false,
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&xs, &[2]).unwrap();
  let mut out = compiled.call(&[x]).unwrap();
  assert_eq!(out.len(), 2);
  let sq = out[0].to_vec::<f32>().unwrap();
  let dbl = out[1].to_vec::<f32>().unwrap();
  assert!(approx_eq(sq[0], 4.0, 1e-6) && approx_eq(sq[1], 9.0, 1e-6));
  assert!(approx_eq(dbl[0], 4.0, 1e-6) && approx_eq(dbl[1], 6.0, 1e-6));
}

/// The same compiled function applied repeatedly (the cache-reuse /
/// decode-loop pattern) must return the same result every call — reuse across
/// calls is the whole point of compilation.
#[test]
fn compiled_reused_across_calls_is_stable() {
  let compiled = compile(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![square(&a[0])?]) },
    false,
  )
  .unwrap();
  for v in [1.0f32, 2.0, 5.0, 10.0] {
    let x = Array::from_slice::<f32>(&[v], &[1]).unwrap();
    let mut out = compiled.call(&[x]).unwrap();
    let got = out[0].to_vec::<f32>().unwrap();
    assert!(
      approx_eq(got[0], v * v, 1e-6),
      "reuse call {v} -> {}",
      got[0]
    );
  }
}

/// `compile_fn` (the `impl Fn` ergonomic surface) matches uncompiled and is
/// itself re-callable like the autograd transforms' returned closures.
#[test]
fn compile_fn_matches_uncompiled() {
  let g = compile_fn(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![add(&a[0], &a[0])?]) },
    false,
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&[3.0f32, 4.0], &[2]).unwrap();
  let mut out = g(&[x]).unwrap();
  let got = out[0].to_vec::<f32>().unwrap();
  assert!(approx_eq(got[0], 6.0, 1e-6) && approx_eq(got[1], 8.0, 1e-6));
}

// ─────────────────────── shapeless ───────────────────────

/// A `shapeless = true` compiled function applied to *different input shapes*
/// must still produce the correct (uncompiled-identical) result for each
/// shape — the shapeless graph is reused across the shape change rather than
/// erroring.
#[test]
fn compiled_shapeless_handles_varying_shapes() {
  let compiled = compile(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![square(&a[0])?]) },
    true,
  )
  .unwrap();

  // shape [3]
  let x3 = Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
  let mut o3 = compiled.call(&[x3]).unwrap();
  assert_eq!(o3[0].to_vec::<f32>().unwrap(), vec![1.0, 4.0, 9.0]);

  // shape [4] — same number of dims, different size: reuses the shapeless graph
  let x4 = Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
  let mut o4 = compiled.call(&[x4]).unwrap();
  assert_eq!(o4[0].to_vec::<f32>().unwrap(), vec![1.0, 4.0, 9.0, 16.0]);
}

// ─────────────────────── error propagation ───────────────────────

/// A closure that returns `Err` surfaces that error through `Compiled::call`
/// (the backend reports the trampoline's non-zero rc; the error is drained
/// from the TLS slot) rather than panicking or returning a bogus result.
#[test]
fn compiled_propagates_closure_error() {
  let compiled = compile(
    |_a: &[Array]| -> mlxrs::Result<Vec<Array>> {
      Err(mlxrs::error::Error::EmptyInput(
        mlxrs::error::EmptyInputPayload::new("compile test: forced error"),
      ))
    },
    false,
  )
  .unwrap();
  let x = Array::from_slice::<f32>(&[1.0f32], &[1]).unwrap();
  let res = compiled.call(&[x]);
  assert!(
    res.is_err(),
    "a closure Err must propagate through Compiled::call"
  );
}

/// A first trace that fails (the Rust body returns `Err`) must NOT leave the
/// `Compiled` able to later return a stale empty success.
///
/// mlx marks a compiled cache entry non-empty *before* tracing it
/// (`mlx/compile.cpp:1126-1133`), so a failed first trace leaves a half-filled
/// entry; a subsequent matching `call` would otherwise `find` it, skip the
/// trace, and return empty outputs as a spurious `Ok`. The poison flag in
/// `Compiled` must turn every call after the first failure into an error.
///
/// A nullary closure (`call(&[])`) is the sharpest case: its inputs always
/// match, so a second call is a guaranteed cache `find` on the same entry —
/// exactly the stale-hit path. We pin both the nullary and the fixed-shape
/// shapes here.
#[test]
fn failed_first_trace_poisons_and_never_returns_stale_success() {
  let _mode = mode_guard();
  enable_compile().unwrap();
  set_compile_mode(CompileMode::Enabled).unwrap();

  // The closure always errors, so the FIRST trace fails. The error is
  // independent of the inputs, so a second same-input call would re-`find` the
  // (now half-filled) cache entry — the stale-success hazard.
  let make_err = |_a: &[Array]| -> mlxrs::Result<Vec<Array>> {
    Err(mlxrs::error::Error::EmptyInput(
      mlxrs::error::EmptyInputPayload::new("compile test: forced first-trace failure"),
    ))
  };

  // Nullary: `call(&[])` — inputs always match, so the second call is a
  // guaranteed cache `find` on the entry the first (failed) trace half-filled.
  {
    let compiled = compile(make_err, false).unwrap();
    let first = compiled.call(&[]);
    assert!(
      first.is_err(),
      "first nullary call must error (trace failed)"
    );
    let second = compiled.call(&[]);
    assert!(
      second.is_err(),
      "second nullary call must ALSO error, never a stale empty Ok",
    );
    // It must be the poison error specifically, proving the second call did NOT
    // re-enter the backend and scrape a half-filled cache entry.
    match second {
      Err(mlxrs::error::Error::InvariantViolation(_)) => {}
      other => panic!("second call must be the poison InvariantViolation, got {other:?}"),
    }
  }

  // Fixed-shape: same contract for a closure that takes an input.
  {
    let compiled = compile(make_err, false).unwrap();
    let x1 = Array::from_slice::<f32>(&[1.0f32, 2.0], &[2]).unwrap();
    assert!(
      compiled.call(&[x1]).is_err(),
      "first fixed-shape call errors"
    );
    let x2 = Array::from_slice::<f32>(&[1.0f32, 2.0], &[2]).unwrap();
    match compiled.call(&[x2]) {
      Err(mlxrs::error::Error::InvariantViolation(_)) => {}
      other => panic!("second fixed-shape call must be the poison error, got {other:?}"),
    }
  }
}

/// The mirror image of the poison test: a `Compiled` built while compilation is
/// **disabled** is a direct passthrough to `f` (no mlx cache), so a failed call
/// must NOT poison it — later valid calls must still run. This pins that the
/// poison flag is gated on the cache-backed wrapper only, never the passthrough.
///
/// Under `disable_compile()` mlx returns `f` unchanged at construction (its
/// `skip_compile()` is true), so every `call` runs the body directly. A closure
/// that errors on its first invocation and succeeds afterward must therefore:
/// (1) surface that first error, (2) succeed on the second call with the right
/// output (proving the wrapper was not bricked), and (3) have run its body on
/// BOTH calls — no caching, no poison.
#[test]
fn disabled_passthrough_never_poisons_and_retries_after_error() {
  let _mode = mode_guard();
  // Construct while disabled → mlx hands back `f` as a direct passthrough.
  disable_compile().unwrap();

  let runs = Arc::new(AtomicUsize::new(0));
  let body_runs = Arc::clone(&runs);
  let f = move |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
    // A passthrough runs the body on EVERY call; the first errors, the rest pass.
    let nth = body_runs.fetch_add(1, Ordering::SeqCst);
    if nth == 0 {
      return Err(mlxrs::error::Error::EmptyInput(
        mlxrs::error::EmptyInputPayload::new("compile test: forced passthrough error"),
      ));
    }
    Ok(vec![square(&a[0])?])
  };
  let compiled = compile(f, false).unwrap();

  // First call: the body's forced error surfaces (the wrapper is not poisoned —
  // it is a passthrough). It must NOT be the poison `InvariantViolation`.
  let x1 = Array::from_slice::<f32>(&[2.0f32, 3.0], &[2]).unwrap();
  match compiled.call(&[x1]) {
    Err(mlxrs::error::Error::InvariantViolation(_)) => {
      panic!("a passthrough's first error must be f's own, never the poison error")
    }
    Err(_) => {}
    Ok(_) => panic!("first call must surface f's forced error"),
  }

  // Second call with valid input: succeeds — a passthrough never poisons, so the
  // earlier error did not brick it — and the body ran again.
  let x2 = Array::from_slice::<f32>(&[2.0f32, 3.0], &[2]).unwrap();
  let mut out = compiled
    .call(&[x2])
    .expect("second disabled call must succeed — a passthrough never poisons");
  let got = out[0].to_vec::<f32>().unwrap();
  assert!(approx_eq(got[0], 4.0, 1e-6) && approx_eq(got[1], 9.0, 1e-6));
  assert_eq!(
    runs.load(Ordering::SeqCst),
    2,
    "a passthrough runs the body on every call (no caching), so both calls ran it",
  );

  // Restore the default backend state (enabled, `Enabled`) for sibling tests.
  enable_compile().unwrap();
  set_compile_mode(CompileMode::Enabled).unwrap();
}

// ─────────────────────── global controls ───────────────────────

/// The global enable/disable/mode controls all succeed, and a compiled
/// function is numerically correct under each — confirming the controls are a
/// performance switch, not a correctness one. Restores the default
/// (compilation enabled, `Enabled` mode) before returning so sibling tests
/// see the default backend state.
#[test]
fn compile_mode_controls_roundtrip_and_preserve_correctness() {
  let _guard = mode_guard();

  let compiled = compile(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![square(&a[0])?]) },
    false,
  )
  .unwrap();
  let check_correct = || {
    let x = Array::from_slice::<f32>(&[2.0f32, 3.0], &[2]).unwrap();
    let mut out = compiled.call(&[x]).unwrap();
    let got = out[0].to_vec::<f32>().unwrap();
    assert!(approx_eq(got[0], 4.0, 1e-6) && approx_eq(got[1], 9.0, 1e-6));
  };

  // Each mode must apply cleanly; the compiled fn stays correct under all.
  for mode in [
    CompileMode::Disabled,
    CompileMode::NoSimplify,
    CompileMode::NoFuse,
    CompileMode::Enabled,
  ] {
    set_compile_mode(mode).unwrap();
    check_correct();
  }

  // Disable globally: a compiled fn still evaluates correctly (un-fused).
  disable_compile().unwrap();
  check_correct();

  // Re-enable: restore the default for any sibling test/process.
  enable_compile().unwrap();
  set_compile_mode(CompileMode::Enabled).unwrap();
  check_correct();
}

/// `CompileMode::as_str` returns the lowercase enumerator name (the mandatory
/// unit-enum string accessor).
#[test]
fn compile_mode_as_str() {
  assert_eq!(CompileMode::Disabled.as_str(), "disabled");
  assert_eq!(CompileMode::NoSimplify.as_str(), "no_simplify");
  assert_eq!(CompileMode::NoFuse.as_str(), "no_fuse");
  assert_eq!(CompileMode::Enabled.as_str(), "enabled");
}

// ─────────────────────── concurrent-tracing soundness ───────────────────────

/// Concurrently build + first-call **independent** compiled closures on many
/// threads, forcing their *first traces* to race.
///
/// MLX's `detail::InTracing::trace_stack_` is a process-global (not
/// thread-local) `static std::vector` (see the module note in
/// `src/transforms/compile.rs` + `mlxrs-sys/vendor/mlx/mlx/transforms.cpp`).
/// A compiled closure traces lazily on its first call (a cache-miss);
/// `compile_trace` pushes/pops that one C++ vector. Two first-calls on
/// separate threads would otherwise mutate it concurrently → data race / UB,
/// which the process-wide trace lock in `Compiled::call` must prevent.
///
/// This does NOT rely on `Compiled` being `!Send`: each thread builds AND
/// calls its OWN closure entirely on that thread (independent closures are
/// exactly what `!Send` permits to exist simultaneously). The forcing
/// conditions for a genuine concurrent first-trace are:
///
/// * **Distinct closures** — every thread/iteration captures a different
///   constant `k` in a freshly-boxed closure, so each has a distinct backend
///   function identity → a guaranteed cache-miss → a real trace (never a
///   cache hit that would skip `compile_trace`).
/// * **A barrier** right before the first `call`, so all threads enter
///   `mlx_closure_apply` — and thus `compile_trace` — as simultaneously as
///   the scheduler allows, maximizing the race window the lock must close.
///
/// Correctness oracle: `f_k(x) = x*x + k` per element, compared to the
/// closed form. Closures are panic-free on the happy path (built with `?`, no
/// `.unwrap()` inside) so none can leak a `trace_stack_` frame for the rest of
/// this process (the reason this lives in its own integration bin).
///
/// **Compilation must actually be enabled** for this to exercise the race. A
/// sibling test mutates the process-global mode to `Disabled`, and libtest runs
/// tests in parallel; if this test's `compile(...)` ran while disabled, mlx
/// returns `f` unchanged (`mlx/compile.cpp:1209-1210`) and the numeric
/// assertions would still pass *without ever touching `compile_trace` or
/// `trace_stack_`* — a false pass. So this test holds [`mode_guard`] for its
/// whole body (serializing it against the mode-mutating tests), forces
/// compilation enabled before building any closure, and proves real compiled
/// execution with a per-closure side-effect counter: a compiled+cached closure
/// runs its Rust body once (the trace) and NOT on a same-shape cache hit, so a
/// second same-shape call must leave the counter at 1. A disabled passthrough
/// would re-run the body and bump it to 2 — failing this test loudly.
#[test]
fn concurrent_independent_first_traces_are_sound() {
  const THREADS: usize = 8;
  const ITERS: usize = 24;
  let xs = [1.0f32, 2.0, 3.0, -4.0, 0.5, 7.0];

  // Serialize against the mode-mutating tests and force compilation genuinely
  // ON, so the threads below really hit `compile_trace` (not the disabled
  // passthrough that would false-pass the numeric oracle).
  let _mode = mode_guard();
  enable_compile().unwrap();
  set_compile_mode(CompileMode::Enabled).unwrap();

  for iter in 0..ITERS {
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
      .map(|t| {
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
          // A distinct constant per (iter, thread) ⇒ a distinct closure
          // identity ⇒ a guaranteed first-trace cache-miss on this call.
          let k = (iter * THREADS + t) as f32;
          // Per-closure side-effect counter: incremented inside the Rust body,
          // so it counts traces. A compiled closure runs the body once (the
          // trace) and never on a cache hit; a disabled passthrough would run
          // it every call. Asserted below to prove compilation is really ON.
          let body_runs = Arc::new(AtomicUsize::new(0));
          let counter = Arc::clone(&body_runs);
          // Build the compiled closure ON this thread (`Compiled` is `!Send`).
          let compiled = compile(
            move |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
              counter.fetch_add(1, Ordering::SeqCst);
              let sq = square(&a[0])?;
              let kc = Array::from_slice::<f32>(&[k], &[1])?;
              Ok(vec![add(&sq, &kc)?])
            },
            false,
          )
          .expect("compile must succeed");
          let x = Array::from_slice::<f32>(&xs, &[6]).expect("input array");

          // Release all threads into `compile_trace` together: this is the
          // concurrent first-trace the trace lock must serialize.
          barrier.wait();
          let mut out = compiled.call(&[x]).expect("compiled call must succeed");
          let got = out[0].to_vec::<f32>().expect("materialize output");

          assert_eq!(got.len(), xs.len());
          for (g, &v) in got.iter().zip(xs.iter()) {
            assert!(
              approx_eq(*g, v * v + k, 1e-4),
              "thread {t} iter {iter}: {g} != closed-form {}",
              v * v + k
            );
          }

          // Prove this was a genuinely COMPILED closure, not the disabled
          // passthrough: a second same-shape call must hit the cache and NOT
          // re-run the Rust body. If compilation were somehow disabled the body
          // would run again and this counter would read 2 — failing loudly.
          let x2 = Array::from_slice::<f32>(&xs, &[6]).expect("input array");
          let mut out2 = compiled
            .call(&[x2])
            .expect("second compiled call must succeed");
          out2[0].to_vec::<f32>().expect("materialize output");
          assert_eq!(
            body_runs.load(Ordering::SeqCst),
            1,
            "thread {t} iter {iter}: compiled body must run once (the trace) and \
             not on a same-shape cache hit — a count of 2 means compilation was \
             disabled and this test was not exercising compile_trace",
          );
        })
      })
      .collect();

    for h in handles {
      // A panic inside a thread (incl. a data-race-induced corruption) surfaces
      // here as a join error, failing the test deterministically.
      h.join().expect("a concurrent-trace thread panicked");
    }
  }

  // Restore the default backend state for any sibling test, matching how the
  // compile-mode test cleans up.
  set_compile_mode(CompileMode::Enabled).unwrap();
}

/// Nested compile: a compiled closure whose body itself builds + calls another
/// compiled closure. The trace lock is taken on the inner `call` while already
/// held by the outer `call` on the SAME thread, so it must be reentrant — a
/// non-reentrant mutex would self-deadlock here.
#[test]
fn nested_compile_does_not_self_deadlock() {
  let outer = compile(
    |a: &[Array]| -> mlxrs::Result<Vec<Array>> {
      // Inner compiled closure, built + first-traced *inside* the outer trace.
      // It is called with the outer's own input slice (`Array` is not `Clone`),
      // so the inner `call` takes the trace lock while the outer already holds
      // it on this thread — the reentrancy path under test.
      let inner = compile(
        |b: &[Array]| -> mlxrs::Result<Vec<Array>> { Ok(vec![add(&b[0], &b[0])?]) },
        false,
      )?;
      let doubled = inner.call(a)?;
      Ok(vec![square(&doubled[0])?])
    },
    false,
  )
  .unwrap();

  let x = Array::from_slice::<f32>(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
  let mut out = outer.call(&[x]).unwrap();
  let got = out[0].to_vec::<f32>().unwrap();
  // (2x)^2 = 4 x^2.
  for (g, v) in got.iter().zip([1.0f32, 2.0, 3.0]) {
    assert!(approx_eq(*g, 4.0 * v * v, 1e-4), "{g} != {}", 4.0 * v * v);
  }
}
