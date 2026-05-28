//! Internal FFI helpers shared across `ops/**`, `transforms/**`, etc.
//!
//! Hosts cross-module RAII guards and small Option-to-raw bridges that
//! were previously copied per call-site. Visibility is `pub(crate)` —
//! these are implementation details, not part of the public API.
//!
//! Centralises the duplicated helpers flagged by audit issue #259:
//! `VectorArrayGuard` (was duplicated 7×), `drain_vector` (3×), and
//! `opt_array` (2×).

use crate::{
  array::Array,
  error::{Result, check},
};

/// RAII guard for an `mlx_vector_array` handle obtained from a fallible
/// mlx-c call. The handle is freed exactly once on drop via
/// `mlx_vector_array_free`, which is a defined no-op on a NULL ctx
/// (sentinel-handle pattern, see `mlxrs-sys/vendor/mlx-c/mlx/c/vector.cpp`).
///
/// Callers typically pair `VectorArrayGuard` with a separate read of the
/// vector via [`drain_vector`]; the guard owns the free, `drain_vector`
/// owns the read. Keeping the two concerns separate is what allows
/// borrow-style consumers (e.g. `transforms::closure::borrow_inputs`) to
/// read without freeing when the vector is mlx-c-owned.
pub(crate) struct VectorArrayGuard(pub(crate) mlxrs_sys::mlx_vector_array);

impl VectorArrayGuard {
  /// Borrow the raw handle for a transient FFI call. Must not outlive `self`.
  #[allow(dead_code)]
  #[inline(always)]
  pub(crate) const fn as_raw(&self) -> mlxrs_sys::mlx_vector_array {
    self.0
  }
}

impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a
    // defined no-op on NULL ctx (sentinel-handle pattern). Runs during
    // `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently
    // per the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}

/// Build a `Vec<Array>` from an `mlx_vector_array` by copying out each
/// handle (refcount bump on each via `mlx_array_set` inside the mlx-c
/// getter). PURE READ — does NOT free `vec`. Callers that own `vec`
/// must pair this with a [`VectorArrayGuard`] in their own scope;
/// callers reading mlx-c-owned vectors (trampoline inputs) must NOT.
pub(crate) fn drain_vector(vec: mlxrs_sys::mlx_vector_array) -> Result<Vec<Array>> {
  // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does not
  // mutate or retain it and returns a plain length.
  let n = unsafe { mlxrs_sys::mlx_vector_array_size(vec) };
  let mut parts = Vec::with_capacity(n);
  for i in 0..n {
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle
    // (NULL ctx); wrapping in `Array` first ensures `Drop` reclaims on
    // early return.
    let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: valid `vec` handle; `part.0` is the freshly-allocated out-param
    // populated by this call. rc surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, vec, i) })?;
    parts.push(part);
  }
  Ok(parts)
}

/// Map an optional `&Array` to `(raw_handle, ownership_anchor)` for FFI
/// calls that accept a nullable array argument. Returning
/// `(mlx_array, Option<Array>)` instead of `Option<&Array>` gives
/// callers a stable raw pointer that mlx-c sees as either NULL (when
/// `a` is `None`) or pointing at a live handle (when `a` is `Some`).
/// The `Option<Array>` in the return tuple keeps the freshly-allocated
/// anchor alive for the duration of the FFI call.
pub(crate) fn opt_array(a: Option<&Array>) -> (mlxrs_sys::mlx_array, Option<Array>) {
  match a {
    Some(arr) => (arr.0, None),
    None => {
      // SAFETY: `mlx_array_new()` returns a fresh empty handle (NULL ctx)
      // per the mlx-c convention; wrapped in the RAII newtype FIRST so an
      // early return / panic frees it. This is NOT an out-param — the
      // NULL-ctx handle is the placeholder mlx-c's "may be null" optional
      // *input* parameters accept for an absent array; the returned anchor
      // keeps it alive across the FFI call.
      let anchor = Array(unsafe { mlxrs_sys::mlx_array_new() });
      (anchor.0, Some(anchor))
    }
  }
}
