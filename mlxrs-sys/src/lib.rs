//! Raw FFI bindings for mlx-c. Pre-committed bindgen output.
//!
//! Regenerate with:
//!
//! ```sh
//! LIBCLANG_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib \
//!   cargo run -p xtask -- regen-bindings
//! ```
//!
//! (CI uses the same path on `macos-14`. `$(xcode-select -p)/usr/lib` does NOT
//! contain libclang on Xcode 26.x; this is the canonical location.)

// Bindgen output is wrapped in a private module so its lints (clippy::all,
// missing_safety_doc, naming-convention warnings) are scoped — not bleeding
// into smoke tests or any future hand-written glue in this crate.
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#[allow(clippy::missing_safety_doc, clippy::all)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod generated {
  include!("generated/bindings.rs");
}

pub use generated::*;

// ───── first-party C++ shims ─────
//
// Bridges to `mlx::core` symbols the vendored mlx-c layer does not expose.
// The C++ sources live in `mlxrs-sys/shim/` and are compiled by build.rs
// against libmlx. These declarations are HAND-WRITTEN (not bindgen output)
// on purpose: the shim is first-party, not part of the vendored mlx-c
// surface, so it must stay out of the `regen-bindings` drift gate.
//
// Policy: every entry here is a tracked mlx-c coverage gap to be upstreamed
// to ml-explore/mlx-c so this block shrinks over time.
unsafe extern "C" {
  /// Bridges `mlx::core::clear_streams()` (declared in `mlx/stream.h`),
  /// which mlx-c does not expose. Destroys all streams created on the
  /// **current thread**, freeing their Metal command encoders. Returns 0
  /// on success, non-zero if the underlying C++ call threw.
  ///
  /// This is mlx's only stream-teardown primitive — it is thread-wide and
  /// bulk (no per-stream free), which is why the safe `Stream` wrapper
  /// cannot reclaim resources via `Drop`.
  pub fn mlxrs_shim_clear_streams() -> ::std::os::raw::c_int;
}

#[cfg(test)]
mod smoke {
  use super::*;
  use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
  };

  // mlx-c's default handler is `printf("MLX error: %s\n", msg); exit(-1);`
  // which would terminate the test process. Install a no-op handler first.
  extern "C" fn noop_handler(_msg: *const c_char, _data: *mut c_void) {}

  #[test]
  fn version_round_trip() {
    // SAFETY: all calls are mlx-c FFI. `noop_handler` is a valid `extern "C"`
    // fn and the data ptr is null (no dtor needed). `mlx_string_new` yields a
    // fresh handle that is read via `mlx_string_data` (result null-checked
    // before `CStr::from_ptr`) and freed exactly once by `mlx_string_free`
    // within this block. No handle escapes.
    unsafe {
      mlx_set_error_handler(Some(noop_handler), ptr::null_mut(), None);
      let mut s = mlx_string_new();
      assert_eq!(mlx_version(&mut s), 0, "mlx_version returned non-zero");
      let data = mlx_string_data(s);
      assert!(!data.is_null());
      let ver = CStr::from_ptr(data).to_string_lossy();
      assert!(!ver.is_empty());
      assert_eq!(mlx_string_free(s), 0);
    }
  }

  #[test]
  fn array_new_free_round_trip() {
    // SAFETY: all calls are mlx-c FFI. `noop_handler` is a valid `extern "C"`
    // fn with a null data ptr. `mlx_array_new` yields a fresh owned handle
    // that is freed exactly once by `mlx_array_free` within this block; it
    // does not escape and is not double-freed.
    unsafe {
      mlx_set_error_handler(Some(noop_handler), ptr::null_mut(), None);
      let arr = mlx_array_new();
      assert_eq!(mlx_array_free(arr), 0);
    }
  }
}
