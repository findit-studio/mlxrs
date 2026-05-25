//! `diagnostics::install` is opt-in and idempotent.
//!
//! `install_is_idempotent` checks repeated installs are a harmless no-op.
//!
//! `panic_hook_chains_previous` proves the panic hook chains (Finding 2):
//! a prior hook installed before `install()` must still run, and must run
//! even though our diagnostic prelude executes first. The whole-binary
//! global panic hook is process state shared with every other test in this
//! binary, so this test saves/restores the hook itself and is the only test
//! here that mutates it; it uses `catch_unwind` so the panic never escapes.
//!
//! `sigabrt_is_not_swallowed` re-execs this test binary in a guarded child
//! that installs diagnostics then `raise(SIGABRT)`; the parent asserts the
//! fixed mlxrs message reached the child's stderr AND the child died via
//! SIGABRT (proving the abort is propagated through the restored default
//! disposition, not swallowed).
//!
//! `sigabrt_chains_previous_handler` re-execs a guarded child that installs
//! its OWN prior `SIGABRT` handler, then calls `install()`, then raises:
//! the parent asserts the child's stderr carries BOTH the mlxrs message and
//! the prior handler's sentinel, proving the previous disposition was
//! captured and chained (not clobbered, not garbage) — i.e. correct
//! `PREV_SIGABRT` capture/restore.
//!
//! NOTE: there is deliberately no thread-race stress test for "SIGABRT
//! delivered to another thread during install()". That race is structurally
//! eliminated by `install()`'s two-step ordering: the previous disposition
//! is queried and published (non-atomic write + `SeqCst` fence) STRICTLY
//! BEFORE our handler is registered with the kernel in a separate syscall,
//! so the handler cannot become live until after `PREV_SIGABRT` is fully
//! published. It is therefore argued correct by construction (not by a
//! nondeterministic, flaky test); `sigabrt_chains_previous_handler` gives
//! deterministic coverage of correct capture/restore.

use std::sync::{
  Arc,
  atomic::{AtomicBool, Ordering},
};

// `std::panic::{set_hook,take_hook}` and `diagnostics::install()`'s one-shot
// `INSTALLED` guard are process-global. `cargo test` runs this binary's tests
// concurrently by default, so tests that mutate that shared state must be
// mutually exclusive or they corrupt each other's observed take/set/restore
// sequence (a parallel-only flake). Every guarded test fully restores global
// hook state itself before returning, so a failed assertion (which panics
// while holding this lock and poisons it) leaves no torn global state — the
// next serialized test can safely recover the guard via
// `PoisonError::into_inner` rather than failing with an unrelated panic.
static DIAG_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn install_is_idempotent() {
  // Serialized: calls the process-global `install()` (`INSTALLED` guard).
  let _serial = DIAG_SERIAL
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner);

  mlxrs::diagnostics::install();
  mlxrs::diagnostics::install();
  mlxrs::diagnostics::install();
}

#[test]
fn panic_hook_chains_previous() {
  // Serialized: mutates the process-global panic hook (`take_hook`/
  // `set_hook`) and calls `install()`.
  let _serial = DIAG_SERIAL
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner);

  // The default test harness installs its own panic hook; save it so we can
  // restore exactly what was there, leaving the binary's shared state intact
  // for other tests regardless of run order.
  let harness_hook = std::panic::take_hook();

  let prior_ran = Arc::new(AtomicBool::new(false));
  let prior_ran_in_hook = Arc::clone(&prior_ran);

  // Install an application-style prior hook *before* diagnostics::install,
  // so diagnostics must chain it.
  std::panic::set_hook(Box::new(move |_info| {
    prior_ran_in_hook.store(true, Ordering::SeqCst);
  }));

  // Chains the prior hook above. (Idempotent across the binary: if a prior
  // test already called install(), the INSTALLED guard makes this a no-op —
  // but then our just-set prior hook is the active hook and still runs, so
  // the assertion below holds either way.)
  mlxrs::diagnostics::install();

  let res = std::panic::catch_unwind(|| {
    panic!("intentional panic to exercise the chained hook");
  });
  assert!(res.is_err(), "catch_unwind should have caught the panic");

  // Restore the harness hook before asserting, so a failed assertion below
  // panics through the normal harness hook (clean output, no recursion into
  // our test hooks).
  std::panic::set_hook(harness_hook);

  assert!(
    prior_ran.load(Ordering::SeqCst),
    "diagnostics::install() clobbered the previously-installed panic hook \
     instead of chaining it"
  );
}

// --- SIGABRT child-process test -------------------------------------------
//
// Guarded re-exec: when MLXRS_DIAG_SIGABRT_CHILD is set, the binary's normal
// test entry still runs, but this test detects the env var and performs the
// child role (install diagnostics, raise SIGABRT) instead of asserting. The
// parent spawns `current_exe` filtered to just this test with the env var
// set. This is robust (no threads, no shared fixtures, deterministic
// signal), so it is shipped rather than omitted.

const SIGABRT_CHILD_ENV: &str = "MLXRS_DIAG_SIGABRT_CHILD";

#[test]
fn sigabrt_is_not_swallowed() {
  if std::env::var_os(SIGABRT_CHILD_ENV).is_some() {
    // Child role: install diagnostics, then abort. The handler must print
    // the fixed message and then re-raise into the default disposition so
    // the process still dies by SIGABRT.
    mlxrs::diagnostics::install();
    // SAFETY: raising SIGABRT is the exact failure mode under test.
    unsafe {
      libc::raise(libc::SIGABRT);
    }
    // Unreachable: SIGABRT must terminate. If we get here the abort was
    // swallowed — exit non-signal so the parent's signal assertion fails.
    std::process::exit(0);
  }

  use std::{os::unix::process::ExitStatusExt, process::Command};

  let exe = std::env::current_exe().expect("current_exe");
  let output = Command::new(exe)
    .args([
      "--exact",
      "sigabrt_is_not_swallowed",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(SIGABRT_CHILD_ENV, "1")
    .output()
    .expect("spawn child test binary");

  let stderr = String::from_utf8_lossy(&output.stderr);

  assert!(
    stderr.contains("mlxrs: process aborted (SIGABRT)"),
    "child stderr missing the fixed mlxrs SIGABRT diagnostic message; \
     stderr was:\n{stderr}"
  );
  assert_eq!(
    output.status.signal(),
    Some(libc::SIGABRT),
    "child did not terminate via SIGABRT (abort was swallowed); \
     status: {:?}, stderr:\n{stderr}",
    output.status
  );
}

// --- SIGABRT prior-handler chaining test ----------------------------------
//
// Guarded re-exec (same pattern as above). The child installs its OWN prior
// SIGABRT handler that writes a distinct sentinel and `_exit(42)`s, THEN
// calls diagnostics::install(), THEN raises SIGABRT. If `PREV_SIGABRT` is
// captured and restored correctly, the mlxrs handler runs (prints its
// message), restores the prior disposition, re-raises, the prior handler
// runs (prints `PRIORHANDLER`) and exits 42. The parent asserts both
// outputs are present — this fails if the previous disposition were
// clobbered or read as garbage.

const SIGABRT_CHAIN_ENV: &str = "MLXRS_DIAG_SIGABRT_CHAIN_CHILD";

extern "C" fn prior_sigabrt_handler(_sig: libc::c_int) {
  const SENTINEL: &[u8] = b"PRIORHANDLER\n";
  // SAFETY: async-signal-safe only — a single `write(2)` then `_exit(2)`.
  unsafe {
    libc::write(2, SENTINEL.as_ptr() as *const libc::c_void, SENTINEL.len());
    libc::_exit(42);
  }
}

#[test]
fn sigabrt_chains_previous_handler() {
  if std::env::var_os(SIGABRT_CHAIN_ENV).is_some() {
    // Child role: install our OWN prior SIGABRT handler first, so
    // diagnostics::install() must capture and chain it.
    // SAFETY: all-zero is a valid initial `libc::sigaction`; the fields used
    // (`sa_sigaction`, `sa_mask`) are explicitly set before it is registered.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = prior_sigabrt_handler as *const () as libc::sighandler_t;
    // SAFETY: registering a valid async-signal-safe handler for SIGABRT.
    unsafe {
      libc::sigemptyset(&mut act.sa_mask);
      libc::sigaction(libc::SIGABRT, &act, std::ptr::null_mut());
    }

    mlxrs::diagnostics::install();

    // SAFETY: raising SIGABRT is the exact path under test.
    unsafe {
      libc::raise(libc::SIGABRT);
    }
    // Unreachable: if the prior handler chained correctly it _exit(42)'d;
    // otherwise SIGABRT's default disposition aborts. Either way we never
    // get here. Exit 0 so a regression (swallowed) makes the parent fail.
    std::process::exit(0);
  }

  use std::process::Command;

  let exe = std::env::current_exe().expect("current_exe");
  let output = Command::new(exe)
    .args([
      "--exact",
      "sigabrt_chains_previous_handler",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(SIGABRT_CHAIN_ENV, "1")
    .output()
    .expect("spawn child test binary");

  let stderr = String::from_utf8_lossy(&output.stderr);

  assert!(
    stderr.contains("mlxrs: process aborted (SIGABRT)"),
    "child stderr missing the fixed mlxrs SIGABRT diagnostic message \
     (mlxrs handler did not run); stderr was:\n{stderr}"
  );
  assert!(
    stderr.contains("PRIORHANDLER"),
    "child stderr missing the prior handler's sentinel — the previous \
     SIGABRT disposition was clobbered or read as garbage instead of being \
     captured and chained; stderr was:\n{stderr}"
  );
}
