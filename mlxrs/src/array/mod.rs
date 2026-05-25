//! `Array` core: RAII handle around `mlxrs_sys::mlx_array`.
//!
//! See `docs/superpowers/specs/` §6.2 for design rationale (Drop must not
//! touch TLS; the only duplication is the fallible refcount-sharing
//! [`Array::try_clone`]; M1 is single-thread only).

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
// `Array` also intentionally does **not** implement `Clone`. The only
// supported duplication is `Array::try_clone() -> Result<Self>`: a
// refcount-sharing handle dup (a fresh `mlx::core::array` over the same
// underlying `array_desc`, no data copy), fallible because the mlx-c handle
// alloc/`set` can fail. An infallible `Clone` would have to panic on that
// failure, so it is not provided.
//
// `!Send`/`!Sync` is required by the underlying mlx-c array/backend, NOT to
// keep any `Clone` cheap. The Phase-3 entry audit confirmed `array_desc_` is
// `std::shared_ptr<array_desc>` (atomic refcount) but `set_status const →
// array_desc_->status = s` is a non-atomic mutation through `const` — so a
// shared `&Array` across threads (`Sync`) would race on `array_desc->status`.
// The same non-atomic lazy/eval state also makes the handle unsound to move
// across threads alongside another handle to the same `array_desc`: a
// `try_clone`d pair on two threads would each call `eval`/`to_vec`/`item`
// (`&mut self`, so `!Sync` doesn't catch it) and race that `status` write.
// mlx's `eval` is itself not concurrency-safe. There is no shared-array
// wrapper: MLX's C++/Python/Swift APIs deliberately don't share arrays across
// threads. To cross threads, extract owned data via `to_vec`/`item`
// (`Send`). The `assert_not_impl_any!` below is the actual enforced contract.
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

impl Array {
  /// Refcount-sharing clone. Returns `Result` so callers handle the rare
  /// allocation-failure path explicitly. `Array` intentionally does **not**
  /// implement `Clone`: a panicking `Clone` impl would hide an FFI failure on
  /// a recoverable path, and a refcount-sharing handle is cheap-but-not-free
  /// (**tens to hundreds of ns**: a fresh `mlx::core::array` heap allocation +
  /// a refcount bump). Never `try_clone` in hot paths; pass `&Array` instead.
  ///
  /// ## Why the heap allocation is unavoidable
  ///
  /// mlx-c's only refcount-sharing primitive is `mlx_array_set(&dst, src)`,
  /// implemented in `vendor/mlx-c/mlx/c/private/array.h:28` as
  /// `new mlx::core::array(s)` — i.e. it always heap-allocates a fresh outer
  /// `mlx::core::array` (which is itself a `shared_ptr<array_desc>` wrapper)
  /// and copy-constructs from `src`, bumping the inner `array_desc` refcount.
  /// There is no public mlx-c entry point that mutates `dst.ctx` in-place from
  /// a stack-allocated `array`, and the mlx C++ `array(const array&)` copy
  /// constructor is private to the C++ side. So `try_clone` cannot elide the
  /// heap allocation through the supported FFI surface; reducing the cost
  /// requires upstream changes to the mlx-c API (issue #117 closes on this
  /// finding, with the alloc-discipline guidance to **avoid the call**, not
  /// to optimise it). Eliding the second alloc inside the `Self(mlx_array_new())`
  /// wrap is also impossible: `mlx_array_new` returns an empty handle whose
  /// `ctx` is NULL, and `mlx_array_set` always allocates when `ctx` is NULL
  /// (`private/array.h:24-31`); there is no path that yields a populated
  /// handle without one heap allocation.
  pub fn try_clone(&self) -> Result<Self> {
    crate::error::ensure_handler_installed();
    // RAII coverage: wrap the fresh handle in `Self` BEFORE the fallible
    // `mlx_array_set` call so panic / early-return drops it via `Drop`.
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL ctx)
    // per the mlx-c convention; it is wrapped in the RAII newtype FIRST so an
    // early return / panic frees it, then populated by the following call.
    let mut out = Self(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
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
    // SAFETY: all `mlx_*` handle args are valid borrowed handles (live for the call,
    // not retained by mlx past it); the out-param was freshly allocated above
    // and is written by this call; the backend rc is surfaced via `check()`.
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
