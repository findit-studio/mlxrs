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

/// CODEX R1 [HIGH] F2 regression guard — concurrent install/Drop on
/// different threads MUST leave the process-global wired-memory limit at
/// its original pre-install value, not at the recommended budget.
///
/// Pre-F2 failure mode (no synchronization):
/// ```text
///   T1 install: captures L0,    sets recommended
///   T2 install: captures recommended,   sets recommended
///   T1 drop:    restores L0     (limit = L0)
///   T2 drop:    restores recommended    (limit = recommended  ← BUG)
/// ```
/// Post-F2 (single-active-guard semantics): T2's install observes the
/// owner-lock is held → returns `Ok(None)` (no-op). Only T1 has a real
/// guard whose `Drop` restores L0.
///
/// This test cannot reliably reproduce the deterministic interleave
/// without sleeps/yields (and even then, races are inherently flaky),
/// so it asserts the *invariant* instead: after any number of concurrent
/// install/drop cycles, the limit is exactly what it was before the
/// stress test began. This invariant holds under the F2 fix; it failed
/// to hold under the pre-F2 design.
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
  // Some installs will collide on the owner-lock and return `Ok(None)`
  // (the F2 single-active-guard semantics); others will succeed with
  // `Ok(Some(_))`. Both branches are correct — the invariant we
  // verify is post-stress restoration, NOT individual install outcomes.
  let handles: Vec<_> = (0..2)
    .map(|_| {
      std::thread::spawn(|| {
        for _ in 0..32 {
          match WiredLimitGuard::install(0, &[]) {
            Ok(Some(_guard)) => {
              // Guard drops at end of scope, restoring under the F2 lock.
            }
            Ok(None) => {
              // Another guard was active OR Metal-unavailable — no-op.
              // Both branches are valid F2 outcomes.
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
  // must equal `snapshot_before`. Pre-F2, the race would leave the limit
  // at `recommended` for a substantial fraction of runs (the T2-overwrites-
  // T1-restore interleave).
  let observed = set_wired_limit(snapshot_before).expect("read-back via set");
  assert_eq!(
    observed, snapshot_before,
    "F2 fix: after concurrent install/drop stress, the wired-memory limit \
     must equal the pre-stress snapshot (snapshot_before={snapshot_before}, \
     observed={observed}). A mismatch (typically observed = recommended) \
     indicates the install/Drop race re-introduced — the single-active-guard \
     ownership lock is supposed to make it impossible for two guards to \
     simultaneously hold captures of inconsistent old_limits."
  );
}

/// CODEX R1 [HIGH] F3 regression guard — the guard's `Drop` MUST NOT
/// panic when `clear_current_thread_streams()` was called before scope
/// exit, AND MUST still restore the prior wired-memory limit.
///
/// The prior `Drop` impl called `Stream::default_gpu()` /
/// `Stream::synchronize()`, both of which panic via
/// `assert_streams_not_cleared()` once the thread has been
/// stream-cleared. That would leak the process-global wired-memory limit
/// (the restore line was unreachable).
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
    "worker thread MUST NOT panic — the F3 regression makes \
     WiredLimitGuard::drop call Stream::default_gpu()/synchronize() which \
     panic on a stream-cleared thread, and a panic-on-Drop would leak \
     the process-global wired-memory limit (worst case: double-panic abort).",
  );

  // Verify the restore actually happened: the limit must equal the value
  // captured at install-time (which was `snapshot_before` since nothing
  // else mutated it under the serialization lock).
  let observed = set_wired_limit(captured).expect("read-back via set");
  assert_eq!(
    observed, captured,
    "Drop must restore captured old_limit (captured={captured}, \
     observed={observed}, snapshot_before={snapshot_before}) — \
     the F3 fix uses Stream::try_synchronize/try_default_gpu to skip the \
     sync step on a cleared thread while still running the limit restore."
  );
}

/// CODEX R1 [HIGH] F3 regression guard — the guard's `Drop` running
/// during an already-in-flight panic MUST NOT double-panic (which would
/// abort the process). Uses `std::panic::catch_unwind` on a closure
/// that creates a live `WiredLimitGuard` and then panics; the catch
/// MUST return `Err(_)` (the original panic propagates) and the limit
/// MUST be restored.
///
/// Pre-F3 failure mode: the closure panics → unwind starts → guard's
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
    panic!("F3 test: deliberate panic with live WiredLimitGuard");
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

/// CODEX R1 [HIGH] F1 regression guard — `recommended_working_set_bytes`
/// on macOS (every CI mac runner has Metal) MUST return `Ok(Some(bytes > 0))`,
/// not `Ok(None)`. The prior implementation gated on the empty `mlx_device_info`
/// handle's NULL ctx immediately after `_new()` and returned `None` *always*
/// (because `_new()` is the handle constructor — see
/// `mlxrs-sys/vendor/mlx-c/mlx/c/private/device.h::mlx_device_info_new_`),
/// silently turning the entire wired-memory feature into a no-op (every
/// `WiredLimitGuard::install`, every `WiredSumPolicy`/`WiredBudgetPolicy`
/// with `cap = None` skipped clamping).
///
/// Gated `#[cfg(target_os = "macos")]` because Metal is mac-only; non-Metal
/// hosts legitimately get `Ok(None)` from the unchanged graceful-None path.
#[cfg(target_os = "macos")]
#[test]
fn recommended_working_set_bytes_returns_some_on_metal() {
  let result = recommended_working_set_bytes().expect("FFI rc on healthy mac");
  let bytes = result.expect(
    "macOS host MUST surface Some(bytes) from the populated mlx_device_info \
     map — None here means the F1 regression (always-None) has reappeared",
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
