//! `Array` core: RAII handle around `mlxrs_sys::mlx_array`.
//!
//! See `docs/superpowers/specs/` §6.2 for design rationale (Drop must not
//! touch TLS; Clone is cheap-but-not-free; M1 is single-thread only).

use static_assertions::assert_not_impl_any;

use crate::error::{Result, check};

pub mod construction;
pub mod conversion;
pub mod ops_impl;

/// MLX N-dimensional array — RAII handle around an mlx-c `mlx_array`.
#[repr(transparent)]
pub struct Array(pub(crate) mlxrs_sys::mlx_array);

// Compile-time guarantees colocated with the type definition.
//
// `Array` is intentionally `!Send` and `!Sync` in M1.
//
// The Phase-3 entry audit confirmed `array_desc_` is `std::shared_ptr<array_desc>`
// (atomic refcount) and `set_status const → array_desc_->status = s` (non-atomic
// mutation through const → `Sync` is unsound). What the audit missed is that
// `Send` + cheap `Clone` together also break soundness even without `Sync`:
// `Clone` produces a refcount-sharing handle (separate `Array`, same underlying
// `array_desc`); if `Array: Send`, two clones can move to two threads where
// each calls `eval`/`to_vec`/`item` on `&mut self`. Each thread sees a distinct
// `&mut Array`, so `!Sync` doesn't catch it — but the underlying C++
// `array_desc->status` write races. To preserve cheap `Clone`, `Send` must go.
// M2 will provide an explicit cross-thread story (likely `SharedArray =
// Arc<Mutex<Array>>` newtype with documented contract).
assert_not_impl_any!(Array: Copy, Send, Sync);

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

impl Array {
  /// Refcount-sharing clone. Returns `Result` so callers can handle the
  /// rare allocation-failure path explicitly.
  pub fn try_clone(&self) -> Result<Self> {
    crate::error::ensure_handler_installed();
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
    crate::error::ensure_handler_installed();
    // `eval` reaches mlx without going through `default_stream()`, so the
    // cleared-thread poison guard must be applied here too — otherwise
    // materializing an existing lazy array on a cleared thread would fail
    // cryptically in the backend instead of panicking immediately.
    // `item`/`to_vec`/`as_slice` all funnel through here, so they are
    // covered transitively.
    crate::stream::assert_streams_not_cleared();
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
