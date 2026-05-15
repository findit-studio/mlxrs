//! Error model for the safe wrapper.
//!
//! Two failure-surfacing paths (verified against `mlx-c/mlx/c/array.cpp`):
//!   - rc pattern: most `int`-returning fns return 0 on success, non-zero on
//!     failure. Use [`check`].
//!   - sentinel-handle pattern: `mlx_array`-returning constructors return
//!     a handle with NULL `ctx` on failure. (Helper lands in sub-batch B
//!     once the safe `Array` newtype exists.)
//!
//! In both cases the error message itself is delivered via the global
//! `mlx_set_error_handler` callback we install eagerly via `#[ctor::ctor(unsafe)]`.
//! That callback writes into a thread-local; check drains it.

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

/// Lazy fallback for the eager `ctor` install. Defends against (a) older
/// rustc toolchains in CI matrices below the rust#133491 fix MSRV, (b) the
/// orthogonal "consumer binary never references any `mlxrs` symbol" stripping
/// case, and (c) sandbox environments that disable `__attribute__((constructor))`
/// symbols entirely. One extra branch on a cold path is cheap insurance.
#[inline]
fn ensure_init() {
  static FALLBACK: OnceLock<()> = OnceLock::new();
  FALLBACK.get_or_init(|| unsafe {
    mlxrs_sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
  });
}

/// Hot path: rc-pattern check. Returns `Ok(())` if `rc == 0`, else drains
/// the TLS slot into `Err`.
#[inline]
pub(crate) fn check(rc: c_int) -> Result<()> {
  if rc == 0 {
    Ok(())
  } else {
    ensure_init();
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
/// with NULL `ctx` on failure (e.g. `mlx_array_new_data`).
#[inline]
pub(crate) fn check_handle(handle: mlxrs_sys::mlx_array) -> Result<crate::Array> {
  if handle.ctx.is_null() {
    ensure_init();
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
