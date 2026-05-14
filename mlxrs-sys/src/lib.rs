//! Raw FFI bindings for mlx-c. M1: hand-written; Phase 2 swaps in bindgen output.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};

// Hand-written subset for Phase 0 link-only smoke. Replaced by bindgen in Phase 2.

#[repr(C)]
#[derive(Copy, Clone)]
pub struct mlx_string {
  pub ctx: *mut c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct mlx_array {
  pub ctx: *mut c_void,
}

pub type mlx_error_handler_func =
  Option<unsafe extern "C" fn(msg: *const c_char, data: *mut c_void)>;

pub type mlx_dealloc_func = Option<unsafe extern "C" fn(data: *mut c_void)>;

unsafe extern "C" {
  pub fn mlx_string_new() -> mlx_string;
  pub fn mlx_string_free(s: mlx_string) -> c_int;
  pub fn mlx_string_data(s: mlx_string) -> *const c_char;

  pub fn mlx_version(out: *mut mlx_string) -> c_int;

  pub fn mlx_set_error_handler(
    handler: mlx_error_handler_func,
    data: *mut c_void,
    dtor: mlx_dealloc_func,
  );

  pub fn mlx_array_new() -> mlx_array;
  pub fn mlx_array_free(arr: mlx_array) -> c_int;
}

#[cfg(test)]
mod smoke {
  use super::*;
  use std::{ffi::CStr, ptr};

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
      assert!(!data.is_null(), "mlx_string_data returned NULL");
      let ver = CStr::from_ptr(data).to_string_lossy();
      assert!(!ver.is_empty(), "version string is empty");
      assert!(
        ver.chars().next().unwrap().is_ascii_digit(),
        "version doesn't start with a digit: {ver:?}"
      );
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
