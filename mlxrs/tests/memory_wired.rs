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
//! `mlx_set_wired_limit` is process-global. The [`WiredLimitGuard`] itself
//! is race-safe across concurrent installs (refcounted via the internal
//! mutex), but our tests also call the raw
//! [`set_wired_limit`] helper to snapshot/restore the limit; THOSE calls
//! would race with the guard tests' state-mutating install/drop cycles
//! without serialization. Tests that mutate the wired limit acquire
//! [`WIRED_LIMIT_LOCK`] first so they run strictly serially within this
//! binary. Per-policy math tests do NOT acquire the lock (they are pure
//! functions of their inputs). The test that intentionally exercises
//! concurrent installs across threads
//! (`concurrent_install_drop_restores_correct_old_limit_per_owner`) holds
//! the lock for its full duration to keep its observation of the live
//! process-global limit consistent. The deterministic in-scope-protection
//! contract is exercised single-threaded by
//! `sequential_install_then_install_then_drop_first_still_protects_second_guard`
//! — the refcount-semantic contract does not require threading and the
//! single-threaded version is immune to scheduler-dependent test-infra
//! defects. CROSS-THREAD in-scope protection (a future regression that
//! accidentally scoped the guard epoch per-thread would pass the
//! sequential test) is covered separately by
//! `scoped_threads_t1_drop_first_does_not_revoke_t2_protection`, which
//! uses [`std::thread::scope`] so workers cannot outlive the test's
//! restore-guard and worker panics propagate to the parent.

use std::sync::{Mutex, MutexGuard, PoisonError};

use mlxrs::{
  Stream,
  memory::{
    WiredBudgetPolicy, WiredFixedPolicy, WiredLimitGuard, WiredMaxPolicy, WiredMemoryMeasurement,
    WiredMemoryPolicy, WiredSumPolicy, recommended_working_set_bytes, set_wired_limit,
  },
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
  assert_ne!(a.id(), b.id());
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

/// `WiredMemoryMeasurement` construction + `total_bytes()` matches the
/// Swift `var totalBytes: Int` formula.
#[test]
fn wired_memory_measurement_construction_and_total() {
  let m = WiredMemoryMeasurement::new(1_000, 200, 50, 1_400, 128, 32);
  assert_eq!(m.weight_bytes(), 1_000);
  assert_eq!(m.kv_bytes(), 200);
  assert_eq!(m.workspace_bytes(), 50);
  assert_eq!(m.peak_active_bytes(), 1_400);
  assert_eq!(m.token_count(), 128);
  assert_eq!(m.prefill_step_size(), 32);
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

/// Regression guard — concurrent
/// install/Drop on different threads MUST leave the process-global
/// wired-memory limit at its original pre-install value, not at the
/// recommended budget. Coarse stress test that asserts the final
/// restored-state invariant; the deterministic in-scope-protection
/// invariant is asserted single-threaded by
/// [`sequential_install_then_install_then_drop_first_still_protects_second_guard`].
///
/// Failure mode without synchronization:
/// ```text
///   T1 install: captures L0,    sets recommended
///   T2 install: captures recommended,   sets recommended
///   T1 drop:    restores L0     (limit = L0)
///   T2 drop:    restores recommended    (limit = recommended  ← BUG)
/// ```
/// With refcounted-guard semantics: T1 install captures L0 + sets
/// limit; T2 install bumps refcount and gets its own `Some(guard)` (NOT
/// `Ok(None)` as in a single-active-guard design); whichever drops
/// first decrements; whichever drops last restores L0.
///
/// This test cannot reliably reproduce the deterministic interleave
/// without sleeps/yields (and even then, races are inherently flaky),
/// so it asserts the *invariant* instead: after any number of concurrent
/// install/drop cycles, the limit is exactly what it was before the
/// stress test began.
#[test]
fn concurrent_install_drop_restores_correct_old_limit_per_owner() {
  let _serialized = lock_wired_limit();

  let Ok(Some(_recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  // Snapshot the pre-stress limit via `set_wired_limit` round-trip
  // (returns prior; we immediately restore it so this round-trip is a
  // no-op on the live value).
  let snapshot_before = set_wired_limit(0).expect("snapshot via round-trip");
  let _ = set_wired_limit(snapshot_before).expect("restore snapshot baseline");

  // Spawn 2 worker threads each running multiple install/drop cycles.
  // Under the refcounted-guard semantics every successful install
  // yields `Ok(Some(_))` — the only valid `Ok(None)` outcome is
  // Metal-unavailable (excluded by the early-return above). The
  // invariant we verify is post-stress restoration; individual install
  // outcomes are uniformly `Some`.
  let handles: Vec<_> = (0..2)
    .map(|_| {
      std::thread::spawn(|| {
        for _ in 0..32 {
          match WiredLimitGuard::install(0, &[]) {
            Ok(Some(_guard)) => {
              // Guard drops at end of scope, decrementing the refcount.
              // The last drop in each epoch restores the captured L0.
            }
            Ok(None) => {
              // A guard-install on a Metal-available host MUST yield
              // Some — Ok(None) here means the refcount semantics
              // regressed back to a single-active flag.
              panic!(
                "refcounted semantics: every install on a Metal-available \
                 host must yield Some(guard); Ok(None) is reserved for the \
                 Metal-unavailable path. A single-active-guard design \
                 silently returns None on concurrent installs which loses \
                 in-scope protection — do not reintroduce that contract."
              );
            }
            Err(e) => panic!("install must not error on a healthy host: {e:?}"),
          }
        }
      })
    })
    .collect();

  for h in handles {
    h.join().expect("worker thread must not panic");
  }

  // Invariant: after all workers have joined, the process-global limit
  // must equal `snapshot_before`. Without synchronization the race would
  // leave the limit at `recommended` for a substantial fraction of runs
  // (the T2-overwrites-T1-restore interleave). The last-drop-restores
  // discipline of the refcounted design guarantees the limit returns to
  // the captured L0.
  let observed = set_wired_limit(snapshot_before).expect("read-back via set");
  assert_eq!(
    observed, snapshot_before,
    "after concurrent install/drop stress, the wired-memory limit \
     must equal the pre-stress snapshot (snapshot_before={snapshot_before}, \
     observed={observed}). A mismatch (typically observed = recommended) \
     indicates an install/Drop race — the refcounted shared state is \
     supposed to make it impossible for two guards to simultaneously \
     hold inconsistent captures, while still letting each guard's full \
     scope enjoy the recommended-limit protection."
  );
}

/// Regression guard — the deterministic
/// in-scope-protection invariant, exercised single-threaded via the
/// refcount-semantic contract (NOT via thread concurrency).
///
/// **Why single-threaded.** The in-scope-protection invariant is a
/// refcount-semantic contract — install bumps the refcount, drop decrements
/// it, and the captured prior limit is restored only when the refcount
/// reaches zero. That contract is fully exercised by a single thread
/// performing `install, install, drop, observe, drop, observe` — no
/// scheduler dependence is required. A threaded version of this test
/// (`Barrier`, an `mpsc` channel handshake, a timeout-bounded `recv`)
/// would invite a class of test-infra defects (barrier-deadlock, ordering,
/// channel disconnect ambiguity, timeout escape, worker-cleanup race); the
/// sequential form avoids that entire class. The coarser cross-thread
/// invariant (post-stress restored-state) is covered by
/// [`concurrent_install_drop_restores_correct_old_limit_per_owner`].
///
/// Single-active-guard failure mode:
/// ```text
///   g1 install → captures L0, sets recommended, returns Some(guard)
///   g2 install → flag set, returns Ok(None) (NO GUARD)
///   caller assumes g2 protected; in reality unprotected from the start
///   g1 drops   → restores L0 (the original, lower limit)
///   g2 still active → now at L0 for the rest of its scope (no protection)
/// ```
///
/// Refcounted-guard:
/// ```text
///   g1 install → state=Some(L0, 1), limit=recommended
///   g2 install → state=Some(L0, 2), limit STAYS recommended, Some(guard)
///   g1 drop    → state=Some(L0, 1), limit STAYS recommended  ← THE FIX
///   g2 still active → observes limit==recommended (verified by this test)
///   g2 drop    → state=None, limit=L0 (final restore)
/// ```
///
/// Under single-active semantics, `WiredLimitGuard::install(...)` while
/// `g1` is alive returns `Ok(None)` and the `expect(...)` on the second
/// install fires immediately — exact-same failure mode as the threaded test
/// it replaces, without any threading overhead.
#[test]
fn sequential_install_then_install_then_drop_first_still_protects_second_guard() {
  // Acquire the per-binary serializer for any test that mutates the
  // process-global wired-limit.
  let _serialize = lock_wired_limit();

  let Ok(Some(recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  // Deliberate baseline DIFFERENT from `recommended` so the post-drop
  // assertion discriminates regardless of ambient state. On large hosts use
  // recommended/2 (always distinct from `recommended`); on tiny hosts fall
  // back to a small bump-down that stays positive.
  let test_baseline: u64 = if recommended > 1024 * 1024 * 1024 {
    recommended / 2
  } else {
    recommended.saturating_sub(1024)
  };
  let original = set_wired_limit(test_baseline).expect("set baseline");

  // RAII cleanup — single owner on this thread, no detached worker, no
  // overlapping Drop with any other guard.
  struct RestoreOriginal(u64);
  impl Drop for RestoreOriginal {
    fn drop(&mut self) {
      let _ = set_wired_limit(self.0);
    }
  }
  let _restore = RestoreOriginal(original);

  // g1 install — first install in this epoch.
  let g1 = WiredLimitGuard::install(0, &[])
    .expect("g1 install: FFI error")
    .expect("g1 install: returned None on first install with Metal available");

  // g2 install while g1 is active — MUST return Some(guard) under
  // refcounted semantics. Under a single-active regression this returns
  // Ok(None) and the assertion fires with the exact diagnostic message.
  let g2 = WiredLimitGuard::install(0, &[])
    .expect("g2 install: FFI error")
    .expect(
      "g2 install: returned Ok(None) while g1 was active — \
       single-active regression (refcounted semantics must return Ok(Some) via refcount bump)",
    );

  // Refcount=2, both protected. Verify the live limit is `recommended`
  // (round-trip via set_wired_limit, which returns the prior value as its
  // out-param without permanently changing the live value when we pass the
  // value already in place).
  let observed_before_drop = set_wired_limit(recommended).expect("readback 1");
  assert_eq!(
    observed_before_drop, recommended,
    "before any drop, limit was {observed_before_drop} not recommended {recommended} — \
     refcount bookkeeping broken"
  );

  drop(g1); // refcount=1, g2 still alive.

  // CRITICAL ASSERT: limit STAYS at `recommended` after g1 drops, because
  // g2 still holds a refcount. This is the contract: in-scope protection
  // preserved across overlapping guard lifetimes. A single-active design
  // would restore to `test_baseline` and g2's remaining scope would be
  // silently unprotected.
  let observed_after_g1_drop = set_wired_limit(recommended).expect("readback 2");
  assert_eq!(
    observed_after_g1_drop, recommended,
    "after dropping g1 while g2 alive, limit was {observed_after_g1_drop} not recommended \
     {recommended} — in-scope protection lost"
  );

  drop(g2); // refcount=0, restore to `test_baseline`.

  // Final assert: limit restored to the baseline we set, proving the
  // last-drop-restores discipline fired correctly.
  let final_observed = set_wired_limit(test_baseline).expect("readback 3");
  assert_eq!(
    final_observed, test_baseline,
    "after all drops, limit was {final_observed} not test_baseline {test_baseline} — \
     restore broken"
  );
}

/// Regression guard — deterministic CROSS-THREAD
/// in-scope-protection coverage. Complements the same-thread refcount-math
/// test [`sequential_install_then_install_then_drop_first_still_protects_second_guard`]
/// by exercising the contract across two distinct OS threads: a future
/// regression that accidentally scoped the guard epoch per-thread/owner
/// (e.g. moved [`WIRED_LIMIT_STATE`] into `thread_local!`) would pass the
/// sequential test and the coarse post-stress invariant test, but would
/// fail this one because T1 dropping its guard would restore the limit
/// while T2 is still mid-scope on the other thread.
///
/// ## Why [`std::thread::scope`]
/// A detached worker (`std::thread::spawn`) could outlive the test's
/// restore-guard and race the next test's setup. [`std::thread::scope`]
/// (stable since Rust 1.63) closes that defect class structurally:
/// - Scoped threads CANNOT outlive the scope — the scope blocks until
///   every spawned thread has joined, so the test's restore-guard is
///   guaranteed to run AFTER every worker has finished its drops.
/// - Worker panics propagate to the parent at scope exit — no silent
///   panic-then-race; a failed `expect` in either worker surfaces as a
///   `JoinHandle::join` error that this test propagates via its own
///   `expect`.
/// - Scoped threads can borrow stack data directly (no `Arc` juggling).
///
/// The ordered handshake uses [`mpsc::channel`] with timeout-bounded
/// `recv` whose `expect` calls turn timeout into a test failure (no
/// schedule-dependent passing of broken code). [`mpsc`] is safe here
/// even on worker panic because `thread::scope` joins every worker
/// before returning, so a disconnected receiver always surfaces the
/// worker's panic via the join-propagation path rather than as an
/// ambiguous channel error.
///
/// Single-active failure: T2's `install` returns `Ok(None)`
/// → the first `expect` in T2 fires with the exact single-active diagnostic.
/// No synchronization or hypothetical per-thread state: T1's
/// drop restores the limit → T2's post-T1-drop readback observes
/// `test_baseline` instead of `recommended` → assertion fires with the
/// "cross-thread in-scope protection lost" diagnostic.
#[test]
fn scoped_threads_t1_drop_first_does_not_revoke_t2_protection() {
  let _serialize = lock_wired_limit();
  let Some(recommended) = recommended_working_set_bytes().expect("rw query") else {
    eprintln!("[skip] Metal not available");
    return;
  };
  // Deliberate baseline DIFFERENT from `recommended` so the cross-thread
  // readbacks discriminate regardless of ambient state.
  let test_baseline: u64 = if recommended > 1024 * 1024 * 1024 {
    recommended / 2
  } else {
    recommended.saturating_sub(1024)
  };
  let original = set_wired_limit(test_baseline).expect("set baseline");
  struct RestoreOriginal(u64);
  impl Drop for RestoreOriginal {
    fn drop(&mut self) {
      let _ = set_wired_limit(self.0);
    }
  }
  let _restore = RestoreOriginal(original);

  // Deterministic handshake via mpsc — channel disconnect on panic is
  // recoverable here because thread::scope joins both threads before
  // returning, so any worker panic surfaces via join rather than as an
  // ambiguous channel error.
  use std::sync::mpsc;
  let (t1_installed_tx, t1_installed_rx) = mpsc::channel::<()>();
  let (t2_observed_tx, t2_observed_rx) = mpsc::channel::<u64>(); // T2 sends observed limit after T1 drops
  let (t1_dropped_tx, t1_dropped_rx) = mpsc::channel::<()>();

  // `mpsc::Receiver` is `!Sync`, so the scoped closures MUST `move` their
  // endpoints; capturing by reference would fail E0277.
  let observed_after_t1_drop = std::thread::scope(|s| {
    s.spawn(move || {
      let g1 = WiredLimitGuard::install(0, &[])
        .expect("T1 install FFI error")
        .expect("T1 install returned None on first install with Metal");
      t1_installed_tx.send(()).expect("send t1_installed");
      // Wait for T2 to install + assert its protection.
      let _ = t2_observed_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("T2 never sent observation — T2 likely failed install");
      drop(g1);
      t1_dropped_tx.send(()).expect("send t1_dropped");
    });

    // Main scope waits for T1 install BEFORE T2 install (ordered).
    t1_installed_rx
      .recv_timeout(std::time::Duration::from_secs(10))
      .expect("T1 install never completed within 10s");

    let t2 = s.spawn(move || {
      // `_g2` (leading underscore on a NAME, not bare `_`) silences clippy's
      // unused-binding lint while still extending the guard's lifetime until
      // scope-end, which is REQUIRED — its Drop closes the refcount epoch.
      let _g2 = WiredLimitGuard::install(0, &[])
        .expect("T2 install FFI error")
        .expect(
          "T2 install returned None while T1 was active — \
           single-active regression OR per-thread-scoped guard regression",
        );
      // While both alive, verify limit is recommended.
      let limit_with_both = set_wired_limit(recommended).expect("readback both");
      assert_eq!(
        limit_with_both, recommended,
        "while both guards alive, limit was {limit_with_both} not recommended {recommended}"
      );
      // Tell T1 it may drop now.
      t2_observed_tx
        .send(limit_with_both)
        .expect("send t2_observed");
      // Wait for T1 to drop.
      t1_dropped_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("T1 drop never completed within 10s");
      // CRITICAL: limit MUST still be recommended (T2 still alive).
      let limit_after_t1 = set_wired_limit(recommended).expect("readback after T1");
      assert_eq!(
        limit_after_t1, recommended,
        "after T1 drop while T2 alive, limit was {limit_after_t1} not recommended \
         {recommended} — cross-thread in-scope protection lost (refcount regression OR \
         per-thread-scoped guards)"
      );
      // T2 returns the observed limit; scope drops g2 at return.
      limit_after_t1
    });
    t2.join()
      .expect("T2 panicked (assertion failure shown above)")
  });

  assert_eq!(observed_after_t1_drop, recommended);

  // After scope exit, both guards are dropped (g1 by T1, g2 by T2's scope-end).
  let final_limit = set_wired_limit(test_baseline).expect("readback final");
  assert_eq!(
    final_limit, test_baseline,
    "after all guards dropped, limit was {final_limit} not test_baseline {test_baseline} — \
     refcount or restore broken"
  );
}

/// Regression guard — a concurrent install (one already
/// active) MUST yield `Ok(Some(_))` not `Ok(None)`. Direct contract test
/// for the API shape change vs a single-active-guard design.
///
/// This is the simplest expression of the bug: under a single-active design,
/// T2's install returned `Ok(None)` and the caller had no way to distinguish
/// "Metal unavailable" from "another guard is active", and crucially had
/// no live guard to participate in the cleanup contract.
#[test]
fn concurrent_install_returns_some_guard_not_none_when_already_installed() {
  let _serialized = lock_wired_limit();

  let Ok(Some(_recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  let snapshot_before = set_wired_limit(0).expect("snapshot via round-trip");
  let _ = set_wired_limit(snapshot_before).expect("restore baseline");

  // Install on this thread (T1).
  let g1 = WiredLimitGuard::install(0, &[])
    .expect("t1 install rc")
    .expect("t1 must receive Some(guard)");

  // Spawn T2 and assert its install also yields Some — this is the
  // contract change. Done in a worker thread to prove the install path
  // does not depend on same-thread reentrancy detection.
  let t2 = std::thread::spawn(|| WiredLimitGuard::install(0, &[]).map(|opt| opt.is_some()));
  let t2_yielded_some = t2
    .join()
    .expect("t2 must not panic")
    .expect("t2 install rc");

  assert!(
    t2_yielded_some,
    "refcounted semantics: a concurrent install (T1 active) must yield \
     Ok(Some(_)) not Ok(None). A single-active-guard design returns \
     None here, silently losing in-scope protection for the caller. This \
     test asserts the refcounted API shape directly — if it ever fails, \
     a single-active regression has crept in."
  );

  // Drop T1 (T2's guard was already dropped at the end of its closure;
  // the boolean `is_some()` consumed-and-dropped the guard before
  // returning). The shared state's refcount is now 1; T1's drop is the
  // last in this epoch and restores `snapshot_before`.
  drop(g1);

  let observed = set_wired_limit(snapshot_before).expect("final read-back");
  assert_eq!(
    observed, snapshot_before,
    "After both T1 + T2 guards drop, limit must restore to snapshot_before"
  );
}

/// Regression guard — refcount discipline. Install guard
/// A, install guard B (refcount = 2), drop A → limit MUST be unchanged
/// (still at recommended). Then drop B → limit MUST restore.
///
/// This is a same-thread version of the deterministic ordering test
/// above; useful because it runs without any thread synchronization
/// overhead, exercises the same install/drop refcount transitions, and
/// can catch a regression where the refcount logic is correct on the
/// install path but wrong on the drop path (e.g. unconditionally
/// restoring on every drop).
#[test]
fn refcounted_guard_drop_does_not_restore_until_last_drop() {
  let _serialized = lock_wired_limit();

  let Ok(Some(recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  let snapshot_before = set_wired_limit(0).expect("snapshot via round-trip");
  let _ = set_wired_limit(snapshot_before).expect("restore baseline");

  // Sanity: discriminating power requires snapshot_before != recommended
  // (same reason as `sequential_install_then_install_then_drop_first_still_protects_second_guard`).
  if snapshot_before == recommended {
    eprintln!(
      "skipping: snapshot_before ({snapshot_before}) == recommended ({recommended}); \
       this test cannot discriminate refcount-discipline in that degenerate state"
    );
    return;
  }

  let a = WiredLimitGuard::install(0, &[])
    .expect("a install rc")
    .expect("a must receive Some(guard)");
  let b = WiredLimitGuard::install(0, &[])
    .expect("b install rc")
    .expect("b must receive Some(guard) under refcounted semantics");

  // Both guards alive, refcount = 2, limit = recommended.
  // Drop A — refcount becomes 1; limit MUST stay at recommended.
  drop(a);

  let observed_after_a_drop = set_wired_limit(recommended).expect("read-back after a drop");
  assert_eq!(
    observed_after_a_drop, recommended,
    "refcount discipline: dropping A while B is still alive must NOT \
     restore the limit (observed={observed_after_a_drop}, \
     recommended={recommended}). A mismatch here means Drop unconditionally \
     restores instead of refcount-gated restoring — the exact regression \
     the refcounted design exists to prevent."
  );
  // Re-set to recommended in case the read-back above mutated the live
  // value (it shouldn't have — we passed recommended which equals the
  // existing live value — but make this defensive).
  let _ = set_wired_limit(recommended).expect("re-set recommended");

  // Drop B — refcount becomes 0, limit MUST restore to snapshot_before.
  drop(b);

  let observed_after_b_drop = set_wired_limit(snapshot_before).expect("read-back after b drop");
  assert_eq!(
    observed_after_b_drop, snapshot_before,
    "last-drop-restores: after B (the last live guard in the epoch) \
     drops, limit must restore to snapshot_before (observed={observed_after_b_drop}, \
     snapshot_before={snapshot_before})."
  );
}

/// Regression guard — the guard's `Drop` MUST NOT
/// panic when `clear_current_thread_streams()` was called before scope
/// exit, AND MUST still restore the prior wired-memory limit.
///
/// A naive `Drop` impl that called `Stream::default_gpu()` /
/// `Stream::synchronize()` would panic via `assert_streams_not_cleared()`
/// once the thread has been stream-cleared, leaking the process-global
/// wired-memory limit (the restore line would be unreachable). `Drop`
/// instead detours around those panicking paths.
///
/// Run inside a dedicated worker thread because
/// `clear_current_thread_streams()` permanently poisons the calling
/// thread for any future mlx work — we don't want to poison the cargo
/// test runner.
#[test]
fn wired_limit_guard_drop_after_stream_cleanup_does_not_panic_and_still_restores() {
  let _serialized = lock_wired_limit();

  let Ok(Some(_recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  // Snapshot the current process-global limit BEFORE the worker thread
  // installs a guard. After the worker joins, this value must be the
  // observed limit — proving the worker's guard `Drop` successfully ran
  // the `set_wired_limit(old)` restore step despite its thread being
  // stream-cleared mid-scope.
  let snapshot_before = set_wired_limit(0).expect("snapshot via round-trip");
  let _ = set_wired_limit(snapshot_before).expect("restore snapshot baseline");

  let join = std::thread::spawn(move || {
    // Touch the GPU stream FIRST so the install path / Drop path have a
    // real per-thread default to interact with — this matches a realistic
    // generation scope where the body did mlx work.
    let _ = Stream::default_gpu();

    let guard = WiredLimitGuard::install(0, &[])
      .expect("install rc")
      .expect("guard installed");
    let captured = guard.old_limit();

    // POISON: clear this thread's streams. Subsequent
    // `Stream::default_gpu` / `Stream::synchronize` would panic via
    // `assert_streams_not_cleared`. The guard's `Drop` (about to run on
    // scope exit) MUST detour around those panicking paths.
    Stream::clear_current_thread_streams().expect("clear_streams shim rc");

    // Guard drops here — must not panic, must still restore `captured`.
    drop(guard);

    captured
  });

  let captured = join.join().expect(
    "worker thread MUST NOT panic — a WiredLimitGuard::drop that called \
     Stream::default_gpu()/synchronize() would panic on a stream-cleared \
     thread, and a panic-on-Drop would leak the process-global \
     wired-memory limit (worst case: double-panic abort).",
  );

  // Verify the restore actually happened: the limit must equal the value
  // captured at install-time (which was `snapshot_before` since nothing
  // else mutated it under the serialization lock).
  let observed = set_wired_limit(captured).expect("read-back via set");
  assert_eq!(
    observed, captured,
    "Drop must restore captured old_limit (captured={captured}, \
     observed={observed}, snapshot_before={snapshot_before}) — \
     Drop uses Stream::try_synchronize/try_default_gpu to skip the \
     sync step on a cleared thread while still running the limit restore."
  );
}

/// Regression guard — the guard's `Drop` running
/// during an already-in-flight panic MUST NOT double-panic (which would
/// abort the process). Uses `std::panic::catch_unwind` on a closure
/// that creates a live `WiredLimitGuard` and then panics; the catch
/// MUST return `Err(_)` (the original panic propagates) and the limit
/// MUST be restored.
///
/// Failure mode: the closure panics → unwind starts → guard's
/// `Drop` runs → `Stream::default_gpu()` succeeds (thread not cleared) →
/// `Stream::synchronize()` is called, also fine → restore runs. So the
/// double-panic risk specifically materializes when sync ALSO panics —
/// covered by the stream-cleanup test above. This test additionally
/// confirms the simpler "Drop during normal panic completes cleanly"
/// invariant.
#[test]
fn wired_limit_guard_drop_during_panic_does_not_double_panic() {
  let _serialized = lock_wired_limit();

  let Ok(Some(_recommended)) = recommended_working_set_bytes() else {
    eprintln!("skipping: recommended_working_set_bytes unavailable on this host");
    return;
  };

  let snapshot_before = set_wired_limit(0).expect("snapshot via round-trip");
  let _ = set_wired_limit(snapshot_before).expect("restore snapshot baseline");

  // catch_unwind needs an UnwindSafe closure; the guard is RAII-only and
  // does not implement UnwindSafe (it borrows `&[]`), so use
  // AssertUnwindSafe — we manually verify the post-state below.
  let captured_cell = std::sync::Mutex::new(None::<u64>);
  let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    let guard = WiredLimitGuard::install(0, &[])
      .expect("install rc")
      .expect("guard installed");
    *captured_cell.lock().unwrap() = Some(guard.old_limit());
    panic!("deliberate panic with live WiredLimitGuard");
  }));

  // The closure SHOULD have panicked; catch_unwind returns Err on panic.
  assert!(
    result.is_err(),
    "deliberate panic inside the closure must propagate as Err from catch_unwind"
  );

  let captured = captured_cell.lock().unwrap().expect(
    "guard install ran before the panic, so old_limit was recorded; if this is \
     None, the install path or panic ordering changed",
  );

  // Verify the restore happened despite the panic-on-Drop path.
  let observed = set_wired_limit(captured).expect("read-back via set");
  assert_eq!(
    observed, captured,
    "Drop-during-panic must restore captured old_limit (captured={captured}, \
     observed={observed}, snapshot_before={snapshot_before})"
  );
}

// ───────────────── recommended_working_set_bytes contract ──────────────────

/// `recommended_working_set_bytes` returns either `Ok(Some(>0))` or
/// `Ok(None)` — never `Err` on a healthy host (FFI errors are reserved for
/// genuine backend failures, not "unsupported" — the latter is `Ok(None)`).
#[test]
fn recommended_working_set_bytes_returns_ok() {
  let result = recommended_working_set_bytes();
  assert!(
    result.is_ok(),
    "FFI rc must surface as Ok(...) on a healthy host"
  );
  if let Ok(Some(n)) = result {
    assert!(n > 0, "Some-value contract: > 0 bytes");
  }
}

/// Regression guard — `recommended_working_set_bytes`
/// on macOS (every CI mac runner has Metal) MUST return `Ok(Some(bytes > 0))`,
/// not `Ok(None)`. Gating on the empty `mlx_device_info` handle's NULL ctx
/// immediately after `_new()` would return `None` *always* (because `_new()`
/// is the handle constructor — see
/// `mlxrs-sys/vendor/mlx-c/mlx/c/private/device.h::mlx_device_info_new_`),
/// silently turning the entire wired-memory feature into a no-op (every
/// `WiredLimitGuard::install`, every `WiredSumPolicy`/`WiredBudgetPolicy`
/// with `cap = None` skipping clamping).
///
/// Gated `#[cfg(target_os = "macos")]` because Metal is mac-only; non-Metal
/// hosts legitimately get `Ok(None)` from the unchanged graceful-None path.
#[cfg(target_os = "macos")]
#[test]
fn recommended_working_set_bytes_returns_some_on_metal() {
  let result = recommended_working_set_bytes().expect("FFI rc on healthy mac");
  let bytes = result.expect(
    "macOS host MUST surface Some(bytes) from the populated mlx_device_info \
     map — None here means an always-None regression has crept in",
  );
  assert!(
    bytes > 0,
    "Metal max_recommended_working_set_size must be > 0 (got {bytes})"
  );
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
