//! Error model for the safe wrapper.
//!
//! Two failure-surfacing paths (verified against `mlx-c/mlx/c/array.cpp`):
//!   - rc pattern: most `int`-returning fns return 0 on success, non-zero on
//!     failure. Internal `check` helper drains the captured message.
//!   - sentinel-handle pattern: `mlx_array`-returning constructors return
//!     a handle with NULL `ctx` on failure. Internal `check_handle` does
//!     the same drain.
//!
//! In both cases the error message itself is delivered via the global
//! `mlx_set_error_handler` callback we install eagerly via `#[ctor::ctor(unsafe)]`.
//! That callback writes into a thread-local; check drains it.
//!
//! The handler MUST be installed before any fallible mlx-c call. The default
//! mlx-c handler is `printf + exit(-1)`, which would terminate the process
//! before our `rc` ever reaches `check()`. Every safe-layer entry point that
//! invokes mlx-c calls `ensure_handler_installed` first as defense-in-depth
//! against a stripped/disabled `#[ctor]`.

use std::{
  cell::RefCell,
  ffi::{CStr, c_char, c_int, c_void},
  panic::{AssertUnwindSafe, catch_unwind},
  ptr,
  sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
  },
};

use crate::Dtype;

/// Errors surfaced from the mlx backend or detected at the safe-wrapper boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Shape mismatch detected by mlx during graph construction or eval.
  #[error("shape mismatch: {message}")]
  ShapeMismatch {
    /// Backend-provided message.
    message: String,
  },

  /// Dtype mismatch (e.g. requesting `as_slice::<f32>` on an i32 array).
  #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
  DtypeMismatch {
    /// Caller-asserted dtype.
    expected: Dtype,
    /// Actual array dtype.
    got: Dtype,
  },

  /// `TryFrom<mlxrs_sys::mlx_dtype>` failed — mlx returned a dtype we don't recognize.
  #[error("unknown dtype value from mlx: {0}")]
  UnknownDtype(u32),

  /// Out-of-memory during allocation (best-effort detection).
  #[error("out of memory")]
  OutOfMemory,

  /// `as_slice` or `to_vec` called on a non-contiguous (post-transpose,
  /// broadcast, or strided-slice) array. M2 will add `.contiguous()` to
  /// materialize a row-contiguous copy.
  #[error("array is not contiguous; M2 will add .contiguous() to materialize")]
  NonContiguous,

  /// Generic backend error with the message captured from mlx-c.
  #[error("mlx backend: {message}")]
  Backend {
    /// Message captured from mlx-c's error handler.
    message: String,
  },
}

/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
  pub(crate) static LAST: RefCell<Option<Error>> = const { RefCell::new(None) };
}

/// Set to `true` by the `#[ctor]` install. Read by the static-init smoke test
/// to verify the eager install ran (vs the lazy fallback rescuing it).
pub(crate) static INIT_VIA_CTOR: AtomicBool = AtomicBool::new(false);

extern "C" fn handler(msg: *const c_char, _data: *mut c_void) {
  // Panics across `extern "C"` are UB. Wrap everything in catch_unwind.
  let _ = catch_unwind(AssertUnwindSafe(|| {
    let s = unsafe { CStr::from_ptr(msg) }
      .to_string_lossy()
      .into_owned();
    let _ = LAST.try_with(|c| {
      if let Ok(mut g) = c.try_borrow_mut() {
        *g = Some(Error::Backend { message: s });
      }
    });
  }));
}

#[ctor::ctor(unsafe)]
fn install_handler() {
  // SAFETY: handler is a valid extern "C" fn; null data ptr; no dtor needed.
  unsafe {
    mlxrs_sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
  }
  INIT_VIA_CTOR.store(true, Ordering::Relaxed);
}

/// Defense-in-depth installer. Every safe-layer entry point that invokes
/// mlx-c calls this before the FFI call so that, if the eager `#[ctor]`
/// install was skipped (older rustc toolchains below the rust#133491 fix
/// MSRV, consumer binaries that never reference any `mlxrs` symbol so the
/// linker drops the ctor section, or sandbox environments that disable
/// `__attribute__((constructor))`), the handler is installed before mlx-c
/// can invoke its default `printf + exit(-1)` and terminate the process.
///
/// Fast path is an atomic load + branch — `INIT_VIA_CTOR` is `true` after
/// either the ctor or this fallback has run, so subsequent calls return
/// immediately without touching the OnceLock.
#[inline]
pub(crate) fn ensure_handler_installed() {
  if INIT_VIA_CTOR.load(Ordering::Relaxed) {
    return;
  }
  ensure_handler_installed_slow();
}

#[cold]
#[inline(never)]
fn ensure_handler_installed_slow() {
  static FALLBACK: OnceLock<()> = OnceLock::new();
  FALLBACK.get_or_init(|| {
    unsafe {
      mlxrs_sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
    }
    INIT_VIA_CTOR.store(true, Ordering::Relaxed);
  });
}

/// Hot path: rc-pattern check. Returns `Ok(())` if `rc == 0`, else drains
/// the TLS slot into `Err`. Does NOT install the handler — callers must
/// have called `ensure_handler_installed` before the FFI call, since by the
/// time `check` runs the default abort handler would already have fired.
#[inline]
pub(crate) fn check(rc: c_int) -> Result<()> {
  if rc == 0 {
    Ok(())
  } else {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend {
          message: format!("mlx returned {rc} with no message"),
        }),
    )
  }
}

/// Sentinel-handle pattern: for constructors that return `mlx_array` directly
/// with NULL `ctx` on failure (e.g. `mlx_array_new_data`). Same install
/// contract as [`check`].
#[inline]
pub(crate) fn check_handle(handle: mlxrs_sys::mlx_array) -> Result<crate::Array> {
  if handle.ctx.is_null() {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend {
          message: "mlx returned null handle".into(),
        }),
    )
  } else {
    Ok(crate::Array(handle))
  }
}

#[cfg(test)]
mod init_smoke {
  use super::*;

  #[test]
  fn ctor_fired() {
    assert!(
      INIT_VIA_CTOR.load(Ordering::Relaxed),
      "ctor install did not fire — likely symbol stripping or static-init ordering issue"
    );
  }

  #[test]
  fn failing_op_returns_err_not_abort() {
    // Clear stale TLS first — cargo test runs #[test] fns on the same
    // thread within a binary; a prior failing op could leave Some(..)
    // and produce a false-positive pass.
    super::LAST.with(|c| *c.borrow_mut() = None);

    let r = crate::Array::ones::<f32>(&(2, 2)).and_then(|a| a.reshape(&(3,)));

    assert!(
      matches!(r, Err(crate::Error::Backend { .. })),
      "failing op aborted process or produced wrong error variant; \
       mlx-c++ may have overwritten our handler post-ctor — got: {r:?}"
    );
  }
}
