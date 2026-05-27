//! Process-isolated regression for `async_eval`.
//!
//! `async_eval` rejects when MLX's `detail::InTracing::trace_stack_` is
//! non-empty. That stack is a function-local `static` (NOT `thread_local`;
//! see `mlxrs-sys/vendor/mlx/mlx/transforms.cpp::trace_stack()`), so it is
//! **process-global, shared across all threads**.
//!
//! In `tests/transforms.rs`, the sibling test
//! `closure_user_panic_propagates_through_grad_as_error` deliberately
//! `panic!()`s inside a `grad` closure. The Rust panic is converted to
//! `Err` at the mlx-c FFI boundary, but MLX's C++ RAII guard for the
//! tracing-stack frame does not always restore on that error path,
//! leaving a stale frame on `trace_stack_`. Any subsequent `async_eval`
//! in the same process then rejects with
//! `"[async_eval] Not allowed inside a graph transformation."`.
//!
//! Each `tests/*.rs` integration-test file is a **separate test bin** =
//! a separate process = a fresh `trace_stack_`. Putting this test alone
//! in its own bin sidesteps the leak class until MLX is patched
//! upstream.
//!
//! Local `cargo test --test transforms` happens to pass because cargo's
//! local test scheduler typically completes `async_eval_then_sync_via_item`
//! before the panic test pollutes — but the GitHub Actions macOS runner's
//! scheduler reliably reverses the order, so the failure only surfaces
//! on CI. This bin's process isolation makes the test deterministic
//! everywhere.

use mlxrs::{Array, ops::arithmetic::square, transforms::async_eval};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
  (a - b).abs() <= tol
}

/// `async_eval` enqueues but does not block; following with `to_vec` (or
/// any item / eval) syncs.
#[test]
fn async_eval_then_sync_via_item() {
  let a = Array::full::<f32>(&(4usize, 4usize), 0.5).unwrap();
  let mut sq = square(&a).unwrap();
  async_eval(&[&sq]).unwrap();
  // Eventually it must materialize; to_vec forces sync.
  // square(0.5) = 0.25.
  let vals = sq.to_vec::<f32>().unwrap();
  assert!(vals.iter().all(|&v| approx_eq(v, 0.25, 1e-6)));
}
