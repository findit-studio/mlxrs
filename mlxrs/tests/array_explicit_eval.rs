//! Regression tests for the `try_item` `&self` borrow-relaxed accessor
//! (CORE-2 / #118) and the `try_clone` doc-only audit (CORE-1 / #117).
//!
//! `try_item` was added so callers holding `&Array` (not `&mut Array`) can
//! still read a scalar. The strict no-implicit-eval contract from
//! [`feedback_no_implicit_eval`] is NOT enforced at the FFI level — see the
//! CORE-2 audit-finding doc on `Array::try_item` for why (mlx-c routes
//! through the non-const C++ `array::item()` overload which implicitly
//! evaluates). Enforcing the strict contract is a separate follow-up that
//! needs `_mlx_array_is_available` binding work.
//!
//! These tests verify:
//! 1. After explicit `eval()`, `try_item` returns the same value as the
//!    back-compat `&mut self` `item`.
//! 2. `try_item` works through `&Array` — no `&mut` borrow required (the
//!    whole reason for the parallel).
//! 3. The `&self` borrow ergonomics let multiple shared references coexist
//!    (the case the `&mut self` `item` could not handle).
//! 4. `try_item` evaluates lazy graphs implicitly (documenting current
//!    behavior, not endorsing it — the no-implicit-eval enforcement is
//!    blocked on follow-up work).
//! 5. `try_clone` round-trips correctly (anchor for the CORE-1 doc update).
//!
//! # Test scope (issues #215 + #223 resolution)
//!
//! The structural syn-based tests that previously guarded `try_item`'s
//! `ensure_handler_installed()` first-call requirement were removed after
//! a 7-round bypass-finding spiral. The behavioral tests in this module
//! cover the normal happy-path and basic borrow-relaxation contract; the
//! stripped-ctor abort scenario (where the process-global mlx error
//! handler was NOT installed by `#[ctor]`) is exercised separately by
//! the child-process fixture in `tests/stripped_ctor_try_item.rs`
//! (issue #223). Removing the `ensure_handler_installed()` call from
//! `Array::try_item` reproducibly fails that test, closing the loop on
//! the "code review IS the enforcement" caveat that previously stood
//! here.

use mlxrs::{Array, ops::arithmetic::add};

#[test]
fn try_item_after_explicit_eval_matches_mut_item() {
  let mut a = Array::full::<f32>(&(1,), 3.5).unwrap();
  a.eval().unwrap();
  let via_mut = a.item::<f32>().unwrap();
  let via_ref = a.try_item::<f32>().unwrap();
  assert_eq!(via_mut, via_ref);
  assert_eq!(via_mut, 3.5);
}

#[test]
fn try_item_works_through_shared_ref() {
  // The whole point of the `&self` accessor: it can be called from a
  // function that only holds `&Array`. The `&mut self` `item` cannot.
  fn read_scalar(a: &Array) -> f32 {
    a.try_item::<f32>().unwrap()
  }

  let mut a = Array::full::<f32>(&(1,), 7.0).unwrap();
  a.eval().unwrap();
  assert_eq!(read_scalar(&a), 7.0);
}

#[test]
fn try_item_allows_concurrent_shared_borrows() {
  // The ergonomic win the `&mut self` `item` couldn't deliver: two
  // simultaneous `&Array` references readable in the same expression.
  let mut a = Array::full::<f32>(&(1,), 5.0).unwrap();
  a.eval().unwrap();
  let r1: &Array = &a;
  let r2: &Array = &a;
  let v1 = r1.try_item::<f32>().unwrap();
  let v2 = r2.try_item::<f32>().unwrap();
  assert_eq!(v1, v2);
  assert_eq!(v1, 5.0);
}

#[test]
fn try_item_dtype_mismatch_errors_before_ffi() {
  // dtype check fires before any FFI call, so the contract is enforced
  // even before the implicit-eval path.
  let mut a = Array::full::<f32>(&(1,), 1.0).unwrap();
  a.eval().unwrap();
  let err = a.try_item::<i32>().unwrap_err();
  assert!(
    matches!(err, mlxrs::Error::DtypeMismatch { .. }),
    "expected DtypeMismatch, got {err:?}"
  );
}

#[test]
fn try_item_after_eval_succeeds_on_lazy_graph() {
  // Canonical recommended pattern from `feedback_no_implicit_eval` — even
  // though the no-implicit-eval contract isn't FFI-enforced today, the
  // explicit eval+read shape is still the right call-site style:
  //     a.eval()?;
  //     let v = a.try_item()?;
  let a = Array::from_slice(&[5.0_f32], &(1,)).unwrap();
  let b = Array::from_slice(&[7.0_f32], &(1,)).unwrap();
  let mut sum = add(&a, &b).unwrap();
  sum.eval().unwrap();
  let v: f32 = sum.try_item().unwrap();
  assert_eq!(v, 12.0);
}

#[test]
fn try_item_currently_implicitly_evaluates_lazy_graph() {
  // Documents the CORE-2 audit finding: mlx-c's non-const `item()` overload
  // implicitly evaluates the array (see the audit-finding doc on
  // `Array::try_item`). This test exists to ANCHOR the current behavior so
  // a future PR that wires `_mlx_array_is_available` for the strict
  // no-implicit-eval contract will fail this test, forcing the docstring +
  // expectations to be updated together.
  let a = Array::from_slice(&[2.0_f32], &(1,)).unwrap();
  let b = Array::from_slice(&[3.0_f32], &(1,)).unwrap();
  let lazy = add(&a, &b).unwrap();

  // No explicit eval. Current behavior: mlx-c evaluates internally.
  let v = lazy.try_item::<f32>().unwrap();
  assert_eq!(v, 5.0);
}

#[test]
fn try_item_works_when_array_enters_via_from_raw_before_any_constructor() {
  // This is a HAPPY-PATH test for the from_raw API integration. It does
  // NOT exercise the stripped-ctor failure path (the scalar value 2.5
  // doesn't trigger `mlx_error`). That regression is now guarded by the
  // child-process fixture in `tests/stripped_ctor_try_item.rs` (issue
  // #223), which spawns a binary with the eager `#[ctor]` install
  // suppressed and asserts `try_item` still returns `Err` instead of
  // aborting via mlx-c's default `exit(-1)`.
  //
  // What this test DOES cover: `try_item` on an `Array` constructed via
  // `from_raw` — the same FFI entry pathway the `error_paths.rs`
  // `transpose_non_contig_view` test uses — returns the expected scalar
  // when `try_item` is the first safe-layer fallible call on that handle
  // (no prior `eval`, `try_clone`, or arithmetic). This anchors the
  // functional contract of the from_raw → try_item composition.
  use mlxrs_sys::mlx_array_new_float32;

  // SAFETY: `mlx_array_new_float32` returns a fresh evaluated scalar
  // mlx_array handle with rc=1 and a populated `array_desc`. Standard
  // mlx-c ctor convention; no out-param or rc to check.
  let raw = unsafe { mlx_array_new_float32(2.5) };
  // SAFETY: `Array::from_raw` contract — `raw` is valid (just produced),
  // not aliased anywhere, and the safe `Array` now owns it and frees it on
  // `Drop`.
  let arr = unsafe { mlxrs::Array::from_raw(raw) };

  // `try_item` is the FIRST safe-layer fallible call on this handle —
  // no prior `eval`, `try_clone`, or arithmetic happened on this `Array`.
  let v: f32 = arr.try_item().unwrap();
  assert_eq!(v, 2.5);
}

#[test]
fn try_clone_doc_audit_round_trip() {
  // CORE-1 (#117) landed as a doc-only update: the `try_clone` heap alloc
  // is unavoidable through the mlx-c public API (`mlx_array_set` always
  // `new`s; see vendor/mlx-c/mlx/c/private/array.h:28). This test anchors
  // the behavioural contract that the audit explicitly preserved: a
  // refcount-sharing clone returns the same scalar as the original.
  let mut a = Array::full::<f32>(&(1,), 42.0).unwrap();
  a.eval().unwrap();
  let b = a.try_clone().unwrap();
  // Both must observe the same value through `try_item` (exercising both
  // APIs together — the borrow-relaxed accessor on the cloned handle).
  assert_eq!(a.try_item::<f32>().unwrap(), 42.0);
  assert_eq!(b.try_item::<f32>().unwrap(), 42.0);
}
