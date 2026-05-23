//! Behavior tests for the wired-memory policies and the
//! [`mlxrs::memory::WiredLimitGuard`] scope guard. Mirrors the assertions in
//! Swift's
//! [`mlx-swift-lm/Tests/MLXLMTests/WiredMemoryPolicyTests.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Tests/MLXLMTests/WiredMemoryPolicyTests.swift),
//! the authoritative behavior contract for the LM wired-memory policy layer.
//!
//! No `peak_memory()` magnitude asserts are made here — the underlying counter
//! is process-global and monotonic, so cross-test pollution under
//! `cargo test`'s default multi-thread runner would produce CI-flaky
//! magnitude checks. Functional contract + monotonic `>=` only.
//!
//! ## Threading: wired-limit tests are serialized
//! `mlx_set_wired_limit` is process-global; two concurrent guards would race
//! on the captured `old_limit` and leave the process in a stale state on
//! drop. Tests that mutate the wired limit acquire [`WIRED_LIMIT_LOCK`]
//! first so they run strictly serially within this binary. Per-policy
//! math tests do NOT acquire the lock (they are pure functions of their
//! inputs).

use std::sync::{Mutex, MutexGuard, PoisonError};

use mlxrs::memory::{
  WiredBudgetPolicy, WiredFixedPolicy, WiredLimitGuard, WiredMaxPolicy, WiredMemoryMeasurement,
  WiredMemoryPolicy, WiredSumPolicy, recommended_working_set_bytes, set_wired_limit,
};

/// Process-global serialization for any test that mutates the wired-memory
/// limit. Acquired by every `install` / `set_wired_limit` test in this file
/// so they run strictly serially even under cargo test's default
/// multi-thread runner.
static WIRED_LIMIT_LOCK: Mutex<()> = Mutex::new(());

fn lock_wired_limit() -> MutexGuard<'static, ()> {
  WIRED_LIMIT_LOCK
    .lock()
    .unwrap_or_else(PoisonError::into_inner)
}

// ─────────────────────────── Sum policy ────────────────────────────────────

/// Mirrors Swift `testWiredSumPolicyCapAffectsLimitAndAdmission` — the
/// explicit-cap path: `clamp(baseline + sum) == cap` when `baseline + sum >
/// cap`, and admission denies the new ticket if it would push past the cap.
#[test]
fn wired_sum_policy_clamps_to_cap() {
  let policy = WiredSumPolicy::new(Some(200));
  // baseline=100, active=[50, 100] → sum=150 → baseline+sum=250 → clamped to 200.
  assert_eq!(policy.limit(100, &[50, 100]), 200);
  // baseline=100, active=[50], new_size=50 → projected=200 → fits.
  assert!(policy.can_admit(100, &[50], 50));
  // baseline=100, active=[50], new_size=51 → projected=201 → over cap.
  assert!(!policy.can_admit(100, &[50], 51));
}

/// `clamp(baseline + sum)` stays below the cap → return the raw sum.
#[test]
fn wired_sum_policy_below_cap() {
  let policy = WiredSumPolicy::new(Some(1_000));
  // baseline=100, active=[50, 75] → sum=125 → baseline+sum=225 → fits under 1000.
  assert_eq!(policy.limit(100, &[50, 75]), 225);
  assert!(policy.can_admit(100, &[50, 75], 700)); // projected 925 <= 1000
}

/// Empty `active_sizes` → limit is the bare baseline (clamped).
#[test]
fn wired_sum_policy_empty_active_returns_baseline() {
  let policy = WiredSumPolicy::new(Some(500));
  assert_eq!(policy.limit(100, &[]), 100);
}

/// Sum policy with `cap = None` clamps to the system's
/// recommended-working-set-size IF available; otherwise returns the raw
/// sum. Either branch is correct — we only assert the no-panic contract +
/// that the result is bounded by `min(raw_sum, recommended)` if recommended
/// is observable.
#[test]
fn wired_sum_policy_no_cap_clamps_to_recommended_if_available() {
  let policy = WiredSumPolicy::new(None);
  let baseline = 100u64;
  let active = [50u64, 75];
  let raw_sum = baseline + active.iter().sum::<u64>();
  let result = policy.limit(baseline, &active);

  match recommended_working_set_bytes().expect("recommended_working_set_bytes FFI") {
    Some(rec) => {
      assert_eq!(
        result,
        raw_sum.min(rec),
        "cap=None: result == min(raw_sum, recommended)"
      );
    }
    None => {
      // No recommended value (non-Metal / unsupported) — clamp is a no-op.
      assert_eq!(result, raw_sum, "cap=None + no recommended: pass-through");
    }
  }
}

// ─────────────────────────── Max policy ────────────────────────────────────

/// Mirrors Swift `testWiredMaxPolicyReturnsLargestDemandOrBaseline` — the
/// max of (baseline, max(active_sizes)).
#[test]
fn wired_max_policy_max_active_or_baseline() {
  let policy = WiredMaxPolicy::new();
  // baseline=100, max(active)=150 → 150.
  assert_eq!(policy.limit(100, &[20, 150, 60]), 150);
  // baseline=200, max(active)=150 → 200 (baseline floor wins).
  assert_eq!(policy.limit(200, &[20, 150, 60]), 200);
}

/// Empty `active_sizes` → just the baseline. Mirrors `activeSizes.max() ??
/// 0` → `max(baseline, 0) == baseline` (for `u64` baseline).
#[test]
fn wired_max_policy_empty_active_returns_baseline() {
  let policy = WiredMaxPolicy::new();
  assert_eq!(policy.limit(100, &[]), 100);
}

/// Max policy admits unconditionally — no cap, no admission gate.
#[test]
fn wired_max_policy_admits_unconditionally() {
  let policy = WiredMaxPolicy::new();
  assert!(policy.can_admit(100, &[], u64::MAX / 2));
}

// ────────────────────────── Fixed policy ───────────────────────────────────

/// Mirrors Swift `testWiredFixedPolicyIgnoresActiveSizes` — fixed returns the
/// constant `limit_bytes` regardless of inputs.
#[test]
fn wired_fixed_policy_returns_limit() {
  let policy = WiredFixedPolicy::new(123);
  assert_eq!(policy.limit(0, &[]), 123);
  assert_eq!(policy.limit(500, &[1, 2, 3]), 123);
}

/// Fixed policy admits unconditionally — no cap, no admission gate.
#[test]
fn wired_fixed_policy_admits_unconditionally() {
  let policy = WiredFixedPolicy::new(100);
  assert!(policy.can_admit(0, &[], u64::MAX));
}

// ────────────────────────── Budget policy ──────────────────────────────────

/// Mirrors Swift `testWiredBudgetPolicyIdentityAndCapBehavior` — the
/// id-based equality + the cap-and-admission contract in a single test.
#[test]
fn wired_budget_policy_identity_and_cap_behavior() {
  let shared_id = "shared-test-id";
  let first = WiredBudgetPolicy::with_id(shared_id, 100, Some(300));
  let second = WiredBudgetPolicy::with_id(shared_id, 999, Some(999));
  let third = WiredBudgetPolicy::with_id("other-id", 100, Some(300));

  // Equality is by id alone (matches Swift `static func == (lhs, rhs) ->
  // Bool { lhs.identifier == rhs.identifier }`).
  assert_eq!(first, second);
  assert_ne!(first, third);

  // limit(baseline=50, active=[75]) = clamp(50+100+75) = clamp(225) = 225 (<= cap=300).
  assert_eq!(first.limit(50, &[75]), 225);

  // canAdmit(baseline=50, active=[75], new=75) → projected = 50+100+75+75 = 300 ≤ cap.
  assert!(first.can_admit(50, &[75], 75));
  // canAdmit(baseline=50, active=[75], new=76) → projected = 301 → over cap.
  assert!(!first.can_admit(50, &[75], 76));
}

/// Budget policy without an explicit id — auto-id ensures distinct
/// instances are not accidentally equal (mirrors Swift `init(...id: UUID =
/// UUID())` per-instance freshness).
#[test]
fn wired_budget_policy_auto_id_is_unique() {
  let a = WiredBudgetPolicy::new(100, Some(300));
  let b = WiredBudgetPolicy::new(100, Some(300));
  assert_ne!(a, b, "auto-id instances should be distinct");
  assert_ne!(a.id_str(), b.id_str());
}

/// Budget policy hashing matches equality — same id → same hash;
/// different id → (almost-certainly) different hash. Strictly we only
/// require the `Eq` ⇒ same-hash invariant.
#[test]
fn wired_budget_policy_hash_matches_eq() {
  use std::collections::HashSet;
  let id = "hash-test";
  let a = WiredBudgetPolicy::with_id(id, 100, Some(300));
  let b = WiredBudgetPolicy::with_id(id, 999, None);
  let mut set: HashSet<WiredBudgetPolicy> = HashSet::new();
  set.insert(a.clone());
  // `b` shares `a`'s id → considered equal → insert is a no-op.
  set.insert(b.clone());
  assert_eq!(set.len(), 1);
  assert!(set.contains(&a));
  assert!(set.contains(&b));
}

// ─────────────────────────── Measurement ───────────────────────────────────

/// `WiredMemoryMeasurement` is a public-field record; verify construction
/// + `total_bytes()` matches the Swift `var totalBytes: Int` formula.
#[test]
fn wired_memory_measurement_construction_and_total() {
  let m = WiredMemoryMeasurement {
    weight_bytes: 1_000,
    kv_bytes: 200,
    workspace_bytes: 50,
    peak_active_bytes: 1_400,
    token_count: 128,
    prefill_step_size: 32,
  };
  assert_eq!(m.weight_bytes, 1_000);
  assert_eq!(m.kv_bytes, 200);
  assert_eq!(m.workspace_bytes, 50);
  assert_eq!(m.peak_active_bytes, 1_400);
  assert_eq!(m.token_count, 128);
  assert_eq!(m.prefill_step_size, 32);
  assert_eq!(m.total_bytes(), 1_250);
}

// ────────────────────── WiredLimitGuard round-trip ─────────────────────────

/// `WiredLimitGuard::install` returns `Ok(Some(_))` on macOS-with-Metal /
/// `Ok(None)` on a CPU-only build. Whichever path: no panic, no error.
/// Holds [`WIRED_LIMIT_LOCK`] for the duration to avoid racing the other
/// guard-install test on the process-global limit.
#[test]
fn wired_limit_guard_install_succeeds_or_returns_none() {
  let _serialized = lock_wired_limit();
  // `model_bytes = 0` cannot trigger the >90% warning.
  let result = WiredLimitGuard::install(0, &[]);
  assert!(
    result.is_ok(),
    "install must not error on a healthy GPU/no-GPU host: {result:?}"
  );
  // Guard (if any) drops here, restoring the prior limit before the lock
  // releases — keeping the global wired-limit state consistent for the
  // next serialized test.
}

/// Round-trip: install captures the prior limit, the guard's `Drop`
/// restores it. We observe the round-trip by reading the limit back via
/// `set_wired_limit(prior)` (which returns the value-currently-set as its
/// out-param), then re-installing it to leave the process limit unchanged.
///
/// This test is a no-op (skipped) on a platform where
/// [`recommended_working_set_bytes`] returns `Ok(None)` — the guard's
/// install path is itself a no-op there, so there's no round-trip to
/// verify.
#[test]
fn wired_limit_guard_drop_restores_old_limit() {
  let _serialized = lock_wired_limit();

  let Ok(Some(recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  // Snapshot the current limit by setting it to itself (a no-op on the
  // limit value, returns the prior value which equals the current value).
  // We cannot read the wired limit without setting it; `mlx_set_wired_limit`
  // has no pure-getter API. So instead: install the guard, observe the
  // captured `old_limit`, drop the guard, then read it back.
  let guard = WiredLimitGuard::install(0, &[]).expect("install rc");
  let guard = guard.expect("guard installed (recommended is Some)");
  let captured_old = guard.old_limit();
  drop(guard);

  // After drop, the wired limit should match `captured_old`. Confirm by
  // setting to itself and observing the out-param.
  let observed_after_drop = set_wired_limit(captured_old).expect("set_wired_limit rc");
  assert_eq!(
    observed_after_drop, captured_old,
    "guard's Drop must restore the captured old_limit (recommended={recommended}, \
     captured_old={captured_old}, observed_after_drop={observed_after_drop})"
  );
}

// ───────────────── recommended_working_set_bytes contract ──────────────────

/// `recommended_working_set_bytes` returns either `Ok(Some(>0))` or
/// `Ok(None)` — never `Err` on a healthy host (FFI errors are reserved for
/// genuine backend failures, not "unsupported" — the latter is `Ok(None)`).
#[test]
fn recommended_working_set_bytes_returns_ok() {
  let result = recommended_working_set_bytes();
  assert!(result.is_ok(), "FFI rc must surface as Ok(...) on a healthy host");
  if let Ok(Some(n)) = result {
    assert!(n > 0, "Some-value contract: > 0 bytes");
  }
}

// ──────────────────────────── tune() stub ──────────────────────────────────

/// `tune()` returns an actionable `Err` until the LM concurrency surface
/// lands; verifies the contract so a caller knows what to handle.
#[test]
fn tune_returns_actionable_unimplemented_error() {
  let err = mlxrs::memory::tune(0, 0, 0, &[]).expect_err("tune is a stub");
  let msg = err.to_string();
  assert!(
    msg.contains("not yet implemented"),
    "tune error message should advertise its unimplemented status: {msg}"
  );
}
