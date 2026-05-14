//! `Array` core: RAII handle around `mlxrs_sys::mlx_array`.
//!
//! See `docs/superpowers/specs/` §6.2 for design rationale (Drop must not
//! touch TLS; Clone is cheap-but-not-free; Send-only no Sync per audit).

use static_assertions::{assert_impl_all, assert_not_impl_any};

use crate::error::{Result, check};

pub mod construction;
pub mod conversion;
pub mod ops_impl;

/// MLX N-dimensional array — RAII handle around an mlx-c `mlx_array`.
#[repr(transparent)]
pub struct Array(pub(crate) mlxrs_sys::mlx_array);

// Compile-time guarantees colocated with the type definition.
//
// `assert_not_impl_any!` is canonical in static_assertions 1.1.0.
assert_not_impl_any!(Array: Copy);

// Send is required across the safe layer. True colocation with the !Copy
// assertion above (round-5 false-colocation fix per project memory).
assert_impl_all!(Array: Send);

impl Drop for Array {
  fn drop(&mut self) {
    // SAFETY: must NOT touch TLS (LAST), call check(), or panic.
    // Drop runs during thread destruction where TLS access can panic,
    // and a panic across `extern "C"` is UB. Discard rc silently.
    unsafe {
      let _ = mlxrs_sys::mlx_array_free(self.0);
    }
  }
}

impl Clone for Array {
  /// Independent handle whose underlying buffer is refcount-shared with `self`
  /// (no data copy). Cost is **tens to hundreds of ns** — heap allocation for
  /// a fresh `mlx::core::array` instance + refcount bump on the underlying buffer.
  /// Never `.clone()` in hot paths; pass `&Array` instead. Use `try_clone` if
  /// you want to handle the failure path explicitly.
  fn clone(&self) -> Self {
    self
      .try_clone()
      .expect("Array::clone: mlx_array_set failed")
  }
}

// Send: an Array can move between threads. Verified atomic refcount via
// docs/audits/send-soundness.md (Phase-3-entry audit): array_desc_ is
// std::shared_ptr<array_desc>, atomic by C++ standard.
unsafe impl Send for Array {}

// Sync intentionally NOT implemented. mlx C++ mutates non-atomic state
// through `const` methods (set_status const → array_desc_->status = s);
// two threads holding `&Array` calling eval/to_vec/item concurrently
// would race. M2 adds SyncArray<Mutex<...>>.

impl Array {
  /// Refcount-sharing clone. Returns `Result` so callers can handle the
  /// rare allocation-failure path explicitly.
  pub fn try_clone(&self) -> Result<Self> {
    // RAII coverage: wrap the fresh handle in `Self` BEFORE the fallible
    // `mlx_array_set` call so panic / early-return drops it via `Drop`.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    let rc = unsafe { mlxrs_sys::mlx_array_set(&mut out.0, self.0) };
    check(rc)?;
    Ok(out)
  }

  /// Force evaluation of this array (and its dependencies). Returns when the
  /// result is materialized in memory.
  ///
  /// Takes `&mut self` because eval is observable state mutation: the
  /// underlying buffer flips from NULL to materialized, and concurrent readers
  /// would race on it. With `Sync` not implemented (see above), `&mut self`
  /// lets the borrow checker enforce "no other reference is alive during eval."
  pub fn eval(&mut self) -> Result<()> {
    check(unsafe { mlxrs_sys::mlx_array_eval(self.0) })
  }

  /// Consumes the Array and returns the raw mlx_array handle. Caller is
  /// responsible for eventually calling `mlx_array_free` (or wrapping the
  /// handle back via `Array::from_raw`).
  ///
  /// # Safety
  /// Caller takes over the lifetime contract; `Drop` will not run.
  pub unsafe fn into_raw(self) -> mlxrs_sys::mlx_array {
    let raw = self.0;
    std::mem::forget(self);
    raw
  }

  /// Wraps a raw mlx_array handle into a safe Array.
  ///
  /// # Safety
  /// Caller asserts that `handle` is valid, was created by a compatible mlx-c
  /// API, and is not concurrently aliased.
  pub unsafe fn from_raw(handle: mlxrs_sys::mlx_array) -> Self {
    Self(handle)
  }
}
