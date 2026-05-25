//! Smoke test for issue #215 / Codex P9-r2 finding — verifies that
//! every public `mlxrs::Array` constructor in `array::construction`
//! returns normally (rather than process-exiting) in a stripped-ctor
//! environment (eager `#[ctor]` install skipped, e.g. older rustc,
//! linker-stripped consumer binary, sandboxed
//! `__attribute__((constructor))`) on NORMAL inputs.
//!
//! # This is a SMOKE TEST, NOT a regression detector
//!
//! **Removing `ensure_handler_installed()` from any of the seven
//! constructors here does NOT flip this matrix from green to red on
//! normal inputs.** The reason is structural: no reachable first-FFI
//! call in any of these constructors throws an `std::exception` for
//! inputs we can craft from Rust without an allocator-failure shim. The
//! later `default_stream()` (which itself calls
//! `ensure_handler_installed`) then installs the handler before any
//! throw site we CAN reach actually throws. Surfacing the install-time
//! regression on a NORMAL call would require either targeted allocator
//! failure injection for the first constructor FFI call or a
//! source/AST invariant test for the seven `ensure_handler_installed`
//! placements — the former requires platform-specific build tooling
//! that is out of scope for this fixture, and the latter is forbidden
//! by issue #215 (a 7-round syn-based structural-test spiral that
//! ultimately deleted the original `try_item` structural test).
//!
//! **Therefore, code review of `mlxrs/src/array/construction.rs` is
//! the enforcement mechanism for the install-at-call-site requirement
//! on these seven constructors.** The CRITICAL comments on each
//! constructor's `ensure_handler_installed()` call cross-reference this
//! limitation.
//!
//! **What this test DOES verify**: the current normal-input constructor
//! paths return without process exit in a stripped-ctor environment
//! (no `exit(-1)` from mlx-c's default handler on the inputs exercised
//! here).
//!
//! **What this test does NOT verify**: that `ensure_handler_installed()`
//! is the FIRST executable statement in each constructor. The
//! handler-install placement requirement remains enforced ONLY by code
//! review of `mlxrs/src/array/construction.rs`.
//!
//! # How the smoke matrix detects a process-exit
//!
//! Each constructor's worst-case raw FFI is wrapped in mlx-c's standard
//! `try { ... } catch (std::exception& e) { mlx_error(e.what()); ... }`
//! boilerplate (see `mlxrs-sys/vendor/mlx-c/mlx/c/array.cpp` for
//! `mlx_array_new`, `mlx_array_new_float32`, `mlx_array_new_data`).
//! Without an installed handler, `mlx_error` falls through to the
//! default `printf("[FATAL ERROR] ...") + exit(-1)` (see
//! `mlx_default_error_handler` in
//! `mlxrs-sys/vendor/mlx-c/mlx/c/error.cpp`). The smoke matrix spawns
//! one child per public constructor with a NORMAL input that exercises
//! the FFI path without triggering a throw; the child exits 0 if the
//! constructor returned (Ok or Err — both observable as `exit(0)`/`exit(1)`),
//! and any process-exit from mlx-c's default handler would surface as
//! a non-zero parent-observed exit code other than 1.
//!
//! # Why a child process
//!
//! Identical setup to `stripped_ctor_try_item`: in a normal test binary
//! the `#[ctor]` in `mlxrs::error` runs unconditionally at static init,
//! so every test inherits the installed handler in-process and the
//! stripped-ctor exit(-1) path is unreachable. We reproduce a
//! stripped-ctor environment by re-execing the test binary with
//! `MLXRS_DISABLE_CTOR_FOR_TEST=1` (which the ctor reads at start and
//! returns early on).
//!
//! # First-FFI reachability survey (why normal inputs can't throw)
//!
//! | Constructor                    | First raw FFI                       | Reliably-craftable throw? |
//! | ------------------------------ | ----------------------------------- | ------------------------- |
//! | `ones` / `zeros` / `eye` /     | `mlx_array_new()`                   | No — body is just         |
//! | `arange` / `linspace`          | (returns `mlx_array({nullptr})`)    | `mlx_array({nullptr})`,   |
//! |                                |                                     | no allocation, no throw.  |
//! | `full`                         | `mlx_array_new_float32(val)`        | Only on `std::bad_alloc`  |
//! |                                | (`new mlx::core::array(val)` + a    | of a ~16-byte allocation; |
//! |                                | scalar `malloc(4)` in `init`)       | infeasible without an     |
//! |                                |                                     | allocator-shim test build.|
//! | `from_slice`                   | `mlx_array_new_data(...)`           | Only on `std::bad_alloc`  |
//! |                                | (does `malloc(nbytes)` inside)      | of the requested buffer;  |
//! |                                |                                     | shape-product overflow    |
//! |                                |                                     | is rejected earlier by    |
//! |                                |                                     | our `checked_mul` guard.  |
//!
//! The stripped-ctor exit(-1) bug CLASS is still reproduced as an
//! executable regression by
//! `stripped_ctor_try_item::try_item_survives_stripped_ctor_environment`,
//! which routes through `mlx_array_item_*` — a throw site reachable via
//! a normal-input `try_item::<f32>()` on a non-scalar. That test
//! protects the bug-class itself; this smoke matrix only verifies that
//! today's seven constructor calls return without process exit on the
//! normal inputs exercised here.

use std::process::Command;

const STRIPPED_CTOR_CHILD_ENV: &str = "MLXRS_STRIPPED_CTOR_CHILD";
const DISABLE_CTOR_ENV: &str = "MLXRS_DISABLE_CTOR_FOR_TEST";
const CONSTRUCTOR_SELECTOR_ENV: &str = "MLXRS_STRIPPED_CTOR_CONSTRUCTOR";

/// Constructors covered by the smoke matrix. Each variant maps to a
/// distinct child invocation that calls the named constructor with a
/// normal input.
const CONSTRUCTORS: &[&str] = &[
  "ones",
  "zeros",
  "full",
  "eye",
  "arange",
  "linspace",
  "from_slice",
];

/// Run the named constructor in the child role. Returns exit code 0 on
/// `Result::Ok`, 1 on `Err` (still observable, NOT a process-exit), and
/// 42 on unexpected absence. The parent treats both 0 and 1 as PASS
/// (the constructor returned normally); only a non-zero exit OTHER than
/// 1 (e.g. 255 from mlx-c's default `exit(-1)`) indicates a process-exit
/// from mlx-c's default handler.
fn invoke_constructor_child(name: &str) -> ! {
  // Each branch is the FIRST safe-layer call in this child process. Any
  // safe-layer entry point would install the handler via its own
  // `ensure_handler_installed()`, which would mask the call-site wiring
  // for the next call — so we only ever invoke ONE constructor per child.
  let result: mlxrs::Result<()> = match name {
    "ones" => mlxrs::Array::ones::<f32>(&(2, 2)).map(drop),
    "zeros" => mlxrs::Array::zeros::<f32>(&(2, 2)).map(drop),
    "full" => mlxrs::Array::full::<f32>(&(2, 2), 1.0).map(drop),
    "eye" => mlxrs::Array::eye::<f32>(3).map(drop),
    "arange" => mlxrs::Array::arange(0.0, 4.0, 1.0).map(drop),
    "linspace" => mlxrs::Array::linspace(0.0, 1.0, 5).map(drop),
    "from_slice" => mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).map(drop),
    other => {
      eprintln!("stripped_ctor_constructors: unknown constructor selector {other:?}");
      std::process::exit(42);
    }
  };
  match result {
    Ok(()) => std::process::exit(0),
    Err(e) => {
      // An `Err` return is still a PASS for the smoke matrix: it means
      // the handler was installed in time to capture the error and the
      // constructor returned a `Result`. A process-exit would manifest
      // as a NON-zero exit code OTHER than 1 (mlx-c's default abort
      // uses -1 / 255).
      eprintln!("stripped_ctor_constructors: {name} returned Err (still a PASS): {e}");
      std::process::exit(1);
    }
  }
}

/// SMOKE TEST (not a regression detector — see module docstring).
/// Verifies each public `Array` constructor returns normally on normal
/// inputs in a stripped-ctor environment. Will NOT flip red if
/// `ensure_handler_installed()` is removed from any of these
/// constructors on normal inputs (no reachable throw site fires before
/// `default_stream()` rescues the missing install). Code review of
/// `mlxrs/src/array/construction.rs` is the enforcement mechanism for
/// the install-at-call-site requirement on these seven functions.
#[test]
fn every_constructor_smoke_in_stripped_ctor_environment() {
  if let Some(name) = std::env::var_os(STRIPPED_CTOR_CHILD_ENV) {
    let selector = std::env::var(CONSTRUCTOR_SELECTOR_ENV).unwrap_or_else(|_| {
      eprintln!(
        "stripped_ctor_constructors: child role detected via {STRIPPED_CTOR_CHILD_ENV}={name:?} \
         but {CONSTRUCTOR_SELECTOR_ENV} was not set"
      );
      std::process::exit(42);
    });
    invoke_constructor_child(&selector);
  }

  let exe = std::env::current_exe().expect("current_exe");
  for &name in CONSTRUCTORS {
    let output = Command::new(&exe)
      .args([
        "--exact",
        "every_constructor_smoke_in_stripped_ctor_environment",
        "--nocapture",
        "--test-threads=1",
      ])
      .env(DISABLE_CTOR_ENV, "1")
      .env(STRIPPED_CTOR_CHILD_ENV, "1")
      .env(CONSTRUCTOR_SELECTOR_ENV, name)
      .output()
      .unwrap_or_else(|e| panic!("spawn child test binary for {name}: {e}"));

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let code = output.status.code();

    // PASS: child returned from the constructor (exit 0 for Ok, exit 1
    // for Err — both are observable returns and prove the constructor
    // didn't process-exit). FAIL: any other exit (most commonly 255 on
    // macOS for mlx-c's default `exit(-1)`). NOTE: this only verifies
    // that the normal-input constructor call returns without process
    // exit in a stripped-ctor environment; it does NOT detect removal
    // of `ensure_handler_installed()` from any of these seven
    // constructors, because the later `default_stream()` installs the
    // handler before any reachable throw site fires (see module
    // docstring). Code review of `mlxrs/src/array/construction.rs` is
    // the enforcement mechanism for the install-at-call-site
    // requirement.
    let passed = matches!(code, Some(0) | Some(1));
    assert!(
      passed,
      "stripped-ctor child for `Array::{name}` exited with {code:?}; expected 0 (Ok returned) \
       or 1 (Err returned — also a PASS, the constructor returned normally). A code of 255 (or \
       -1 on signal) means a process-exit from mlx-c's default `printf + exit(-1)` handler fired \
       inside `Array::{name}`'s end-to-end execution. See `mlxrs/src/array/construction.rs` and \
       issue #215. NOTE: this smoke matrix does NOT detect removal of `ensure_handler_installed()` \
       from this constructor on normal inputs (no reachable first-FFI throw site fires before \
       `default_stream()`'s install rescues it); code review of the seven `CRITICAL` comments in \
       `construction.rs` is the enforcement mechanism for that wiring.\n\
       child stdout:\n{stdout}\nchild stderr:\n{stderr}",
    );
  }
}
