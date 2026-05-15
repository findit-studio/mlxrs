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
mod generated {
  include!("generated/bindings.rs");
}

pub use generated::*;

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
    unsafe {
      mlx_set_error_handler(Some(noop_handler), ptr::null_mut(), None);
      let arr = mlx_array_new();
      assert_eq!(mlx_array_free(arr), 0);
    }
  }
}
