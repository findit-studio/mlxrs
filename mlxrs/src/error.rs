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

  /// Tokenizer subsystem error (HF tokenizer load/encode/decode, chat-template
  /// render, tool-call parse). Only constructed when the `tokenizer` feature
  /// is enabled. The message carries the underlying cause.
  #[cfg(feature = "tokenizer")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer")))]
  #[error("tokenizer: {message}")]
  Tokenizer {
    /// Human-readable description of the tokenizer failure.
    message: String,
  },

  /// Defense-in-depth shard-path collision:
  /// [`crate::lm::load::save_model`]'s atomic no-replace
  /// `std::fs::hard_link` of a shard tempfile onto its final shard path
  /// failed with [`std::io::ErrorKind::AlreadyExists`], meaning a file
  /// already occupies that final path. `link(2)` is atomic + no-replace
  /// by spec, so this surfaces in a single syscall with no silent-
  /// replace window (a `rename`-based publish would race a concurrent
  /// writer here). The collision-resistant `gen_id` (timestamp µs,
  /// PID, per-process counter) makes this statistically unreachable in
  /// normal operation; surfacing it as a hard `Err` keeps the save
  /// fail-closed (never silently overwrite a foreign file). Constructed
  /// only when the `lm` feature is enabled.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  #[error("shard path collision: {path}")]
  ShardPathCollision {
    /// The pre-existing final shard path that the atomic no-replace
    /// `hard_link` refused to overwrite.
    path: std::path::PathBuf,
  },

  /// Post-commit durability warning: a checkpoint or config file was
  /// successfully renamed into place (so the new content IS visible on
  /// disk + would be observed by a subsequent
  /// [`crate::lm::load::load_weights`] / [`crate::lm::load::load_config`])
  /// but a follow-up `fsync` of the parent directory failed. The
  /// directory-rename entry may not yet be durable on disk: a power loss
  /// before the filesystem internally drains could leave the directory
  /// pointing at the OLD entry. The caller knows the save is **logically
  /// committed**.
  ///
  /// Returned by [`crate::lm::load::save`] when [`crate::lm::load::save_model`]
  /// or the post-commit config rename produced a
  /// [`crate::lm::load::CommitOutcome::CommittedWithDurabilityWarning`].
  /// Constructed only when the `lm` feature is enabled.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  #[error("save committed but durability fsync failed (committed={committed}): {source}")]
  DurabilityWarning {
    /// Always `true` for now — this variant is constructed only AFTER the
    /// observable commit point (the index rename + the config rename) has
    /// succeeded. Kept in the public shape so a future caller can branch on
    /// it without an API break if a `committed=false` durability story is
    /// ever added.
    committed: bool,
    /// The underlying `fsync_dir` IO error.
    #[source]
    source: std::io::Error,
  },
}

#[cfg(feature = "tokenizer")]
impl Error {
  /// Construct a [`Error::Tokenizer`] from anything stringifiable. Used
  /// throughout the `tokenizer` module to funnel HF / minijinja / serde
  /// failures into the crate's unified error type.
  pub(crate) fn tokenizer(message: impl Into<String>) -> Self {
    Self::Tokenizer {
      message: message.into(),
    }
  }
}

/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Allocate a `Vec<T>` reserving exactly `cap` capacity, returning
/// [`Error::OutOfMemory`] instead of aborting the process on allocator
/// failure (which `Vec::with_capacity` / `vec![…; n]` do). Use for any
/// REQUEST-SCALED allocation on a hot path (sequence length, token /
/// image counts) so an oversized or hostile input fails recoverably
/// rather than terminating the process.
///
/// For small fixed-size allocations (a handful of elements) the infallible
/// `Vec::with_capacity` remains fine — this is for input-proportional
/// buffers.
///
/// Consumed by the `lm`, `vlm`, `audio`, and `embeddings` modules for
/// request-scaled host-side buffers (the VLM-9 allocation-hardening pass,
/// now extended across lm/audio/embeddings). Gated to exactly the features
/// that use it so `cargo hack --each-feature` sees no dead code (`vlm` and
/// `audio` both enable `lm`).
#[cfg(any(feature = "lm", feature = "embeddings"))]
pub(crate) fn try_with_capacity<T>(cap: usize) -> Result<Vec<T>> {
  let mut v = Vec::new();
  v.try_reserve_exact(cap).map_err(|_| Error::OutOfMemory)?;
  Ok(v)
}

/// Fallible [`slice::to_vec`]: clone `slice` into a freshly-reserved
/// `Vec`, returning [`Error::OutOfMemory`] instead of aborting on
/// allocation failure. The recoverable analogue of `slice.to_vec()` for
/// request-scaled slices. (Only the `vlm` module needs the owned-clone
/// form; lm/audio/embeddings use `try_with_capacity` + `extend` or
/// `try_extend_from_slice` directly, hence the narrower gate.)
#[cfg(feature = "vlm")]
pub(crate) fn try_to_vec<T: Clone>(slice: &[T]) -> Result<Vec<T>> {
  let mut v = try_with_capacity(slice.len())?;
  v.extend_from_slice(slice);
  Ok(v)
}

/// Fallible [`Vec::extend_from_slice`]: reserve room for `slice` and append,
/// returning [`Error::OutOfMemory`] instead of aborting on allocation
/// failure. Uses the AMORTIZED `try_reserve` (NOT `try_reserve_exact`):
/// callers grow the same `Vec` repeatedly (processor history accumulates the
/// prefill prompt, then one token per decode step), so exact reservation
/// would reallocate on every append and turn an O(n) accumulation into
/// O(n²). The recoverable analogue of `vec.extend_from_slice(slice)`.
#[cfg(feature = "lm")]
pub(crate) fn try_extend_from_slice<T: Clone>(v: &mut Vec<T>, slice: &[T]) -> Result<()> {
  v.try_reserve(slice.len()).map_err(|_| Error::OutOfMemory)?;
  v.extend_from_slice(slice);
  Ok(())
}

thread_local! {
  pub(crate) static LAST: RefCell<Option<Error>> = const { RefCell::new(None) };
}

/// The most recent backend error recorded on this thread, if any. Used by
/// [`crate::diagnostics`] to surface mlx context when a panic follows a
/// backend failure. Non-panicking: `try_with` keeps it safe during thread
/// teardown, and `try_borrow` keeps it safe when called from inside a panic
/// hook that interrupted code already holding the `RefCell` borrow — a
/// borrow conflict yields `None` rather than a (double-)panic.
pub(crate) fn last_error_message() -> Option<String> {
  LAST
    .try_with(|c| {
      c.try_borrow()
        .ok()
        .and_then(|g| g.as_ref().map(|e| e.to_string()))
    })
    .ok()
    .flatten()
}

/// Set to `true` by the `#[ctor]` install. Read by the static-init smoke test
/// to verify the eager install ran (vs the lazy fallback rescuing it).
pub(crate) static INIT_VIA_CTOR: AtomicBool = AtomicBool::new(false);

extern "C" fn handler(msg: *const c_char, _data: *mut c_void) {
  // Panics across `extern "C"` are UB. Wrap everything in catch_unwind.
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: mlx-c guarantees `msg` is a valid NUL-terminated C string for the
    // duration of this error-handler callback; the owned `String` copies it
    // out so nothing escapes the callback.
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
    // SAFETY: `handler` is a valid `extern "C"` fn pointer, the data pointer is
    // NULL, and no destructor is needed; installs the process-global mlx-c
    // error handler.
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

/// Sentinel-handle pattern for `mlx_vector_array`-returning constructors
/// (e.g. `mlx_vector_array_new`): they report failure via the error handler
/// and return a handle with NULL `ctx`. Unlike [`check_handle`] the caller
/// keeps ownership of its handle (it is passed by value into the subsequent
/// mlx-c call and freed by its own RAII guard), so this returns `Result<()>`
/// like [`check`] — draining `LAST` into `Err` when `ctx` is null. Same
/// install contract as [`check`].
#[inline]
pub(crate) fn check_vector_array_handle(handle: mlxrs_sys::mlx_vector_array) -> Result<()> {
  if handle.ctx.is_null() {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend {
          message: "mlx returned null vector_array handle".into(),
        }),
    )
  } else {
    Ok(())
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
