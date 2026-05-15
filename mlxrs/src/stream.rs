//! Stream module: internal per-thread default GPU stream + public `Stream`
//! handle (M2).
//!
//! ## Internal singleton (M1 carry-over)
//!
//! `default_stream()` is a per-thread cache of the default GPU stream used by
//! every `ops::*` free function. It is intentionally process-lifetime-leaked
//! (Metal frameworks tear down before destructors run, so calling
//! `mlx_stream_free` at exit would crash).
//!
//! Per-thread (not process-wide) because mlx-c++ stores the default stream and
//! its `CommandEncoder` in `thread_local` storage on the C++ side
//! (see `mlx/stream.cpp::default_stream_storage` and
//! `mlx/backend/metal/device.cpp::get_command_encoders`). A handle obtained on
//! one thread cannot be used to eval on another — eval throws
//! "There is no Stream(gpu, N) in current thread."
//!
//! ## Public `Stream` (M2)
//!
//! [`Stream`] is an explicit, owned RAII handle for callers who want lifetime
//! control: short-lived worker threads, multi-device pipelines, or fixtures
//! that need deterministic teardown. Drop calls `mlx_stream_free`. Same
//! per-thread caveat applies for GPU streams — a `Stream` obtained on thread
//! T can only be used to eval on T (mlx-c++ side asserts this; we cannot
//! enforce it at compile time without giving up `Send`, which the audit
//! confirms is sound for the POD `mlx::core::Stream`).

use std::{cell::Cell, ffi::CStr};

use static_assertions::assert_not_impl_any;

use crate::{
  device::Device,
  error::{Result, check, ensure_handler_installed},
};

thread_local! {
  static DEFAULT_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

pub(crate) fn default_stream() -> mlxrs_sys::mlx_stream {
  // Most safe-layer FFI consumers funnel through here; install the error
  // handler before any mlx-c call so a stripped/disabled #[ctor] cannot let
  // the default printf+exit handler fire on the very first failure.
  crate::error::ensure_handler_installed();
  DEFAULT_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: handler installed above; errors surface via TLS.
    let s = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs: mlx_default_gpu_stream_new returned NULL ctx — \
         GPU unavailable or initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

// INTENTIONAL: never freed at thread/process exit. Metal frameworks tear down
// before destructors run, so calling mlx_stream_free at exit would crash.
// Instruments will flag this as a leak on shutdown — that's expected.
//
// USAGE GUIDANCE: each thread that ever calls into mlxrs allocates its own
// GPU stream that lives until process exit. mlxrs is intended to be driven
// from a small, long-lived set of worker threads (a fixed-size thread pool
// or the main thread). Patterns that spawn a fresh OS thread per request or
// per task — rayon-with-thread-recycling, std::thread per HTTP request,
// short-lived spawn loops — will accumulate one mlx_stream per worker over
// the process lifetime and grow without bound. M2's public `Stream` API
// (below) provides explicit lifetime control for those cases.

// ───────────────────────── Public Stream API (M2) ─────────────────────────

/// MLX execution stream — RAII handle around `mlxrs_sys::mlx_stream`.
///
/// A stream targets a specific device and serializes work submitted to it.
/// Construct via [`Stream::default_gpu`], [`Stream::default_cpu`], or
/// [`Stream::new_on`]. Drop calls `mlx_stream_free`.
///
/// ## Threading
/// `Stream` is intentionally **`!Send` and `!Sync`**.
///
/// The `mlx::core::Stream` struct is a `{DeviceType, int}` POD, so the
/// Phase-3 audit originally concluded Send/Sync was sound. That conclusion
/// was layout-only and is wrong in practice: a `Stream` is an *index into
/// per-thread state*. mlx-c++ stores the default-stream and the per-stream
/// `CommandEncoder` in C++ thread-local storage, so a GPU stream constructed
/// on thread A cannot be used to eval (or `synchronize`) on thread B —
/// mlx-c++ throws `"There is no Stream(gpu, N) in current thread."`. This
/// was confirmed empirically by the `SharedArray` cross-thread experiment.
///
/// This is the same class of bug as the M1 `Array` Send revision: a
/// trivially-copyable handle whose *referent* has thread-affine state.
/// Marking the wrapper `Send` would let safe code move the handle across a
/// thread boundary and hit that failure path. Until a thread-checked or
/// CPU/GPU-split API exists (future milestone), `Stream` stays single-thread
/// like `Array`. (`Device` IS `Send + Sync` — it is a pure `{kind, index}`
/// descriptor with no thread-local referent.)
#[repr(transparent)]
pub struct Stream(pub(crate) mlxrs_sys::mlx_stream);

// NO `unsafe impl Send/Sync for Stream`. The raw `mlx_stream` contains a
// `*mut c_void`, so the auto-traits are already absent; the assertion below
// locks that in against an accidental future `unsafe impl`.
assert_not_impl_any!(Stream: Send, Sync);

impl Drop for Stream {
  fn drop(&mut self) {
    // SAFETY: must NOT touch TLS or panic (drop runs during thread teardown).
    // Discard rc silently — same convention as Array::drop.
    //
    // NOTE: this is the explicit-ownership path. The internal
    // `default_stream()` singleton is intentionally leaked at process exit
    // (Metal frameworks tear down before drop runs); the public `Stream`
    // type is owned by the caller, so we follow normal RAII.
    unsafe {
      let _ = mlxrs_sys::mlx_stream_free(self.0);
    }
  }
}

impl Clone for Stream {
  /// Independent handle that wraps a fresh `mlx_stream` ctx pointing at the
  /// same underlying `mlx::core::Stream` payload (same `{kind, index}`).
  fn clone(&self) -> Self {
    self
      .try_clone()
      .expect("Stream::clone: mlx_stream_set failed")
  }
}

impl Stream {
  /// New default-GPU stream. Wraps `mlx_default_gpu_stream_new`. The handle
  /// is owned by this `Stream` and freed on drop.
  ///
  /// On a thread that never spun up Metal, this triggers GPU initialization;
  /// returns `Err(Backend { .. })` if the GPU is unavailable.
  pub fn default_gpu() -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_default_gpu_stream_new() };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_default_gpu_stream_new returned NULL ctx \
                  (GPU unavailable or init failed)"
          .into(),
      });
    }
    Ok(Self(raw))
  }

  /// New default-CPU stream. Wraps `mlx_default_cpu_stream_new`.
  pub fn default_cpu() -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_default_cpu_stream_new returned NULL ctx".into(),
      });
    }
    Ok(Self(raw))
  }

  /// New stream targeting `device`. Wraps `mlx_stream_new_device`.
  pub fn new_on(device: &Device) -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_stream_new_device(device.0) };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: "mlx_stream_new_device returned NULL ctx".into(),
      });
    }
    Ok(Self(raw))
  }

  /// Refcount-style clone via `mlx_stream_set`. Returns `Result` so callers
  /// can handle the rare allocation-failure path explicitly.
  pub fn try_clone(&self) -> Result<Self> {
    ensure_handler_installed();
    // `mlx_stream_new` returns an empty handle (NULL ctx) intended to be
    // populated by `mlx_stream_set`/`mlx_get_default_stream` — same
    // out-param convention as `mlx_array_new`. Wrap in `Self` first so RAII
    // covers the fallible set.
    let mut out = Self(unsafe { mlxrs_sys::mlx_stream_new() });
    check(unsafe { mlxrs_sys::mlx_stream_set(&mut out.0, self.0) })?;
    Ok(out)
  }

  /// Block until all work submitted to this stream is complete. Wraps
  /// `mlx_synchronize`.
  pub fn synchronize(&self) -> Result<()> {
    ensure_handler_installed();
    check(unsafe { mlxrs_sys::mlx_synchronize(self.0) })
  }

  /// Returns the [`Device`] this stream targets. Wraps `mlx_stream_get_device`.
  pub fn device(&self) -> Result<Device> {
    ensure_handler_installed();
    let mut dev = Device(unsafe { mlxrs_sys::mlx_device_new() });
    check(unsafe { mlxrs_sys::mlx_stream_get_device(&mut dev.0, self.0) })?;
    Ok(dev)
  }

  /// Returns the index of this stream within its device. Wraps
  /// `mlx_stream_get_index`.
  pub fn index(&self) -> Result<i32> {
    ensure_handler_installed();
    let mut idx: i32 = 0;
    check(unsafe { mlxrs_sys::mlx_stream_get_index(&mut idx, self.0) })?;
    Ok(idx)
  }

  /// Whether two streams refer to the same `{device, index}` pair. Wraps
  /// `mlx_stream_equal`.
  pub fn equal(&self, other: &Stream) -> bool {
    unsafe { mlxrs_sys::mlx_stream_equal(self.0, other.0) }
  }

  /// Borrow the raw mlx-c handle (does not transfer ownership).
  ///
  /// # Safety
  /// Caller must not call `mlx_stream_free` on the returned handle and must
  /// not retain it past `self`'s lifetime.
  #[inline]
  pub unsafe fn as_raw(&self) -> mlxrs_sys::mlx_stream {
    self.0
  }
}

/// Returns the current process-wide default stream for `device`. Wraps
/// `mlx_get_default_stream`.
pub fn get_default_stream(device: &Device) -> Result<Stream> {
  ensure_handler_installed();
  let mut out = Stream(unsafe { mlxrs_sys::mlx_stream_new() });
  check(unsafe { mlxrs_sys::mlx_get_default_stream(&mut out.0, device.0) })?;
  Ok(out)
}

/// Install `stream` as the process-wide default for the device it targets.
/// Wraps `mlx_set_default_stream`.
///
/// Note: this does NOT swap the per-thread default-GPU stream cached by
/// `default_stream()` — internal `ops::*` calls keep using their cached
/// handle. Use this when interoperating with raw mlx-c calls or with the
/// `mlx_get_default_stream` API.
pub fn set_default_stream(stream: &Stream) -> Result<()> {
  ensure_handler_installed();
  check(unsafe { mlxrs_sys::mlx_set_default_stream(stream.0) })
}

impl PartialEq for Stream {
  fn eq(&self, other: &Self) -> bool {
    self.equal(other)
  }
}

impl Eq for Stream {}

impl std::fmt::Debug for Stream {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = unsafe { mlxrs_sys::mlx_string_new() };
    let rc = unsafe { mlxrs_sys::mlx_stream_tostring(&mut s, self.0) };
    let result = if rc == 0 {
      let p = unsafe { mlxrs_sys::mlx_string_data(s) };
      if p.is_null() {
        write!(f, "Stream(<unprintable>)")
      } else {
        let cs = unsafe { CStr::from_ptr(p) };
        write!(f, "Stream({})", cs.to_string_lossy())
      }
    } else {
      write!(f, "Stream(<unprintable>)")
    };
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(s);
    }
    result
  }
}
