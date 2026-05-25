//! Regression test for issues #215 + #223 — verifies that
//! [`mlxrs::Array::try_item`] returns `Err` (not `exit(-1)`) when the
//! process-global `mlx_set_error_handler` was NOT installed by the
//! crate's eager `#[ctor]`.
//!
//! # Why a child process
//!
//! The `#[ctor]` in `mlxrs::error` is process-global and unconditional in
//! a normal test binary: every test in this crate inherits the installed
//! handler at static-init time, so the stripped-ctor abort scenario
//! cannot be reproduced in-process. The defense-in-depth
//! `ensure_handler_installed()` call that opens [`mlxrs::Array::try_item`]
//! is the ONLY thing that converts a future `#[ctor]` strip / disable
//! into a normal `Err` return; if that call were removed, mlx-c's default
//! `printf + exit(-1)` would terminate the process before `check()` could
//! drain the error.
//!
//! To exercise that path deterministically without rebuilding the binary
//! we use the same re-exec pattern as `tests/diagnostics.rs`: the parent
//! spawns the current test executable with a fresh env var
//! (`MLXRS_STRIPPED_CTOR_CHILD=1`) AND with the test-only opt-out env
//! var (`MLXRS_DISABLE_CTOR_FOR_TEST=1`) that the crate's `#[ctor]`
//! reads to skip its eager install. The child re-runs this same test,
//! detects the env var, performs the actual try_item call on a
//! non-scalar (which the C++ `array::item()` overload throws
//! `std::invalid_argument` for — see
//! `mlxrs-sys/vendor/mlx/mlx/array.h:566-579`), and exits 0 on `Err` /
//! 42 on unexpected `Ok`. The parent asserts the child exited cleanly
//! with code 0.
//!
//! # Stash-and-verify
//!
//! Locally verified that the test reproducibly catches the regression:
//!   - With `ensure_handler_installed()` PRESENT in
//!     [`mlxrs::Array::try_item`]: test PASSES (child exits 0).
//!   - With `ensure_handler_installed()` REMOVED: test FAILS (child dies
//!     via `exit(-1)` from mlx-c's default handler — observable as
//!     `Some(255)` exit code on macOS).
//!   - With the call restored: test PASSES again.
//!
//! # `MLXRS_DISABLE_CTOR_FOR_TEST` is test-only
//!
//! That env var is read once, in the `#[ctor]` body in `mlxrs::error`,
//! and never elsewhere. Production binaries do not set it; only this
//! regression child process does. Its only effect is to skip the eager
//! handler install at static-init time so the `ensure_handler_installed`
//! defense-in-depth path is the active code under test.

use std::process::Command;

const STRIPPED_CTOR_CHILD_ENV: &str = "MLXRS_STRIPPED_CTOR_CHILD";
const DISABLE_CTOR_ENV: &str = "MLXRS_DISABLE_CTOR_FOR_TEST";

#[test]
fn try_item_survives_stripped_ctor_environment() {
  if std::env::var_os(STRIPPED_CTOR_CHILD_ENV).is_some() {
    // Child role: the eager `#[ctor]` install was suppressed by
    // `MLXRS_DISABLE_CTOR_FOR_TEST=1` in our env. `try_item`'s
    // first-line `ensure_handler_installed()` must therefore be the
    // installer that prevents mlx-c's default `printf + exit(-1)` from
    // terminating us when `mlx_array_item_*` reports the non-scalar
    // size-check failure.
    //
    // Build the non-scalar `[3]` array through raw FFI so `try_item` is
    // the FIRST safe-layer fallible call on this thread. Going through
    // `mlxrs::Array::from_slice` would defeat the test: every safe-layer
    // entry point also calls `ensure_handler_installed()`, which would
    // install the handler before `try_item` ran and mask the regression
    // (the test would always pass even if `try_item`'s own
    // `ensure_handler_installed()` were removed). Direct FFI keeps the
    // process-global handler uninstalled until `try_item` itself runs.
    //
    // `mlx::core::array::item()` throws
    // `std::invalid_argument("item can only be called on arrays of size 1.")`
    // (`mlxrs-sys/vendor/mlx/mlx/array.h:566-579`), and mlx-c routes
    // that to `mlx_error` → our handler → TLS → `check()` →
    // `Err(Error::Backend)`.
    let data: [f32; 3] = [1.0, 2.0, 3.0];
    let dims: [i32; 1] = [3];
    // SAFETY: a fresh `mlx_array_new_data` call with a typed `f32` buffer
    // and matching 1-D shape `[3]`; mlx-c copies the buffer in and
    // returns an owned handle. The handle is non-NULL because every
    // arg (non-null data ptr, positive dim, valid f32 dtype) is valid.
    let raw = unsafe {
      mlxrs_sys::mlx_array_new_data(
        data.as_ptr().cast::<std::ffi::c_void>(),
        dims.as_ptr(),
        1,
        mlxrs_sys::mlx_dtype::from(mlxrs::Dtype::F32),
      )
    };
    assert!(
      !raw.ctx.is_null(),
      "mlx_array_new_data returned NULL handle; the FFI fixture cannot \
       proceed and the regression assertion would not be meaningful"
    );
    // SAFETY: `raw` was just produced by `mlx_array_new_data`, is not
    // aliased, and the safe `Array` now owns it (freed on Drop).
    let arr = unsafe { mlxrs::Array::from_raw(raw) };
    match arr.try_item::<f32>() {
      Err(_) => std::process::exit(0),
      Ok(_) => std::process::exit(42),
    }
  }

  // Parent role: spawn the child with both env vars set.
  let exe = std::env::current_exe().expect("current_exe");
  let output = Command::new(exe)
    .args([
      "--exact",
      "try_item_survives_stripped_ctor_environment",
      "--nocapture",
      "--test-threads=1",
    ])
    .env(DISABLE_CTOR_ENV, "1")
    .env(STRIPPED_CTOR_CHILD_ENV, "1")
    .output()
    .expect("spawn child test binary");

  let stderr = String::from_utf8_lossy(&output.stderr);
  let stdout = String::from_utf8_lossy(&output.stdout);

  assert!(
    output.status.success(),
    "stripped-ctor child exited with {:?}; expected 0 (Err returned from \
     try_item on a non-scalar). A non-zero exit means either:\n  \
     - try_item's `ensure_handler_installed()` was removed/reordered, so \
       mlx-c's default `printf + exit(-1)` aborted the child before the \
       error reached `check()` (exit code 255 on macOS, see issues #215 \
       and #223 for context); OR\n  \
     - try_item returned Ok(_) for a non-scalar array (exit code 42, \
       which would indicate a regression in mlx-c's `array::item()` \
       size-check at `mlxrs-sys/vendor/mlx/mlx/array.h:566-579`).\n\
     child stdout:\n{stdout}\nchild stderr:\n{stderr}",
    output.status.code(),
  );
}
