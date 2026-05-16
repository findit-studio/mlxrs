//! Public `Device` API: an RAII handle around `mlxrs_sys::mlx_device`.
//!
//! The mlx-c++ side defines `mlx::core::Device = { DeviceType type, int index }`
//! (see `mlx/device.h`). The `mlx_device` C boundary wraps a heap-allocated
//! `mlx::core::Device*` behind `void* ctx`, so this safe wrapper takes
//! ownership and frees on `Drop`.
//!
//! `Device` is `Send + Sync` — the handle is a pure `{kind, index}`
//! descriptor with no thread-local referent (unlike `Stream`, which indexes
//! mlx-c++ per-thread CommandEncoder state and is therefore `!Send`).
//!
//! HOWEVER: the *mlx-c++ process-global default device* is a plain
//! non-atomic function-static (`mlx::core::mutable_default_device()` returns
//! `&static Device`, NOT `thread_local`). `set_default` writes it and
//! `current` reads it, so concurrent safe-Rust calls would be a C++ data
//! race. Both are serialized through `DEFAULT_DEVICE_LOCK`. (The default
//! *stream* getters/setters do NOT need this — mlx stores default streams in
//! `thread_local` storage, so each thread only touches its own.)

use std::{ffi::CStr, sync::Mutex};

use static_assertions::assert_impl_all;

use crate::error::{Result, check, ensure_handler_installed};

/// Serializes safe-Rust access to mlx-c++'s non-atomic global default
/// device. `Mutex::new` is const since Rust 1.63 (MSRV is far above that),
/// so a plain `static` works without `OnceLock`. This only protects callers
/// going through the safe `Device` API; raw `mlxrs-sys` FFI users that call
/// `mlx_set_default_device` directly are in `unsafe` territory and must
/// provide their own synchronization.
static DEFAULT_DEVICE_LOCK: Mutex<()> = Mutex::new(());

/// Device kind tag — mirrors `mlx_device_type` (`MLX_CPU` / `MLX_GPU`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
  /// CPU device (`mlx_device_type__MLX_CPU`).
  Cpu,
  /// GPU device (Metal on Apple silicon; `mlx_device_type__MLX_GPU`).
  Gpu,
}

impl DeviceKind {
  /// Convert to the raw mlx-c device-type tag.
  #[inline]
  fn to_raw(self) -> mlxrs_sys::mlx_device_type {
    match self {
      DeviceKind::Cpu => mlxrs_sys::mlx_device_type__MLX_CPU,
      DeviceKind::Gpu => mlxrs_sys::mlx_device_type__MLX_GPU,
    }
  }

  /// Convert from the raw mlx-c device-type tag.
  #[inline]
  fn from_raw(raw: mlxrs_sys::mlx_device_type) -> Result<Self> {
    match raw {
      mlxrs_sys::mlx_device_type__MLX_CPU => Ok(DeviceKind::Cpu),
      mlxrs_sys::mlx_device_type__MLX_GPU => Ok(DeviceKind::Gpu),
      other => Err(crate::Error::Backend {
        message: format!("unknown mlx_device_type: {other}"),
      }),
    }
  }

  /// Number of available devices of this kind. Wraps `mlx_device_count`.
  pub fn count(self) -> Result<usize> {
    ensure_handler_installed();
    let mut n: i32 = 0;
    check(unsafe { mlxrs_sys::mlx_device_count(&mut n, self.to_raw()) })?;
    Ok(n.max(0) as usize)
  }
}

/// MLX compute device — RAII handle around `mlxrs_sys::mlx_device`.
///
/// Constructed via [`Device::cpu`], [`Device::gpu`], [`Device::current`], or
/// [`Device::with_index`]. Cheap to clone (each clone is a distinct
/// heap-allocated `mlx::core::Device` with the same `{type, index}` payload).
#[repr(transparent)]
pub struct Device(pub(crate) mlxrs_sys::mlx_device);

// SAFETY: `mlx::core::Device` is `{DeviceType type, int index}` POD with no
// mutable shared state (verified in `docs/audits/send-soundness.md` for the
// matching `Stream` case; `Device` is the same shape). Two threads holding
// distinct `Device` values cannot race because the underlying C++ object
// has no atomics-required mutation; a `Device` is also effectively
// read-only after construction (mlx-c provides no mutation API beyond
// `mlx_device_set`, which fully replaces the handle's ctx).
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

assert_impl_all!(Device: Send, Sync);

impl Drop for Device {
  fn drop(&mut self) {
    // SAFETY: Drop must NOT touch TLS, call check(), or panic. mlx_device_free
    // is documented to deallocate the heap-backed ctx; we discard rc silently
    // following the same convention as Array::drop.
    unsafe {
      let _ = mlxrs_sys::mlx_device_free(self.0);
    }
  }
}

impl Clone for Device {
  /// Independent handle that wraps a fresh heap-allocated `mlx::core::Device`
  /// with the same `{kind, index}` payload as `self`.
  fn clone(&self) -> Self {
    self
      .try_clone()
      .expect("Device::clone: mlx_device_set failed")
  }
}

impl Device {
  /// Construct a CPU device with index `0`. Wraps `mlx_device_new_type`.
  pub fn cpu() -> Result<Self> {
    Self::with_index(DeviceKind::Cpu, 0)
  }

  /// Construct a GPU device with index `0`. Wraps `mlx_device_new_type`.
  pub fn gpu() -> Result<Self> {
    Self::with_index(DeviceKind::Gpu, 0)
  }

  /// Construct a device of the given `kind` and `index`. Wraps
  /// `mlx_device_new_type`. The resulting handle is heap-backed; this
  /// `Device` value owns it and frees on `Drop`.
  pub fn with_index(kind: DeviceKind, index: i32) -> Result<Self> {
    ensure_handler_installed();
    let raw = unsafe { mlxrs_sys::mlx_device_new_type(kind.to_raw(), index) };
    if raw.ctx.is_null() {
      return Err(crate::Error::Backend {
        message: format!("mlx_device_new_type returned NULL ctx for kind={kind:?} index={index}",),
      });
    }
    Ok(Self(raw))
  }

  /// Refcount-style clone: allocates a fresh `mlx_device` ctx and copies
  /// `self.0` into it via `mlx_device_set`. Returns `Result` so callers can
  /// handle the rare allocation-failure path explicitly.
  pub fn try_clone(&self) -> Result<Self> {
    ensure_handler_installed();
    // `mlx_device_new` returns an empty handle (NULL ctx) intended to be
    // populated by a subsequent set/get call — same out-param convention as
    // `mlx_array_new`. Wrap in `Self` first so RAII covers the fallible set.
    let mut out = Self(unsafe { mlxrs_sys::mlx_device_new() });
    check(unsafe { mlxrs_sys::mlx_device_set(&mut out.0, self.0) })?;
    Ok(out)
  }

  /// Returns the current process-wide default device. Wraps
  /// `mlx_get_default_device`.
  ///
  /// Serialized against [`Device::set_default`] via `DEFAULT_DEVICE_LOCK` —
  /// reading the non-atomic mlx-c++ global concurrently with a write would
  /// be a C++ data race.
  pub fn current() -> Result<Self> {
    ensure_handler_installed();
    let _g = DEFAULT_DEVICE_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    let mut out = Self(unsafe { mlxrs_sys::mlx_device_new() });
    check(unsafe { mlxrs_sys::mlx_get_default_device(&mut out.0) })?;
    Ok(out)
  }

  /// Install `self` as the mlx-c++ process-wide default device. Wraps
  /// `mlx_set_default_device`.
  ///
  /// **Limitation (read before relying on this):** this sets the *mlx-c++*
  /// global default, which only affects FFI calls that are passed the
  /// implicit/global stream. mlxrs's own ops do NOT consult it — every
  /// safe-layer op routes through an internal per-thread default **GPU**
  /// stream (`stream::default_stream`) that is created once and never
  /// re-derived from `Device::current()`. So `Device::gpu().set_default()`
  /// is a no-op for mlxrs ops (they were already on GPU) and
  /// `Device::cpu().set_default()` will NOT move existing mlxrs ops to the
  /// CPU. Explicit per-op device/stream selection is a future-milestone API
  /// (ops do not yet take a stream argument). This method is provided for
  /// (a) interop with raw `mlxrs-sys` FFI calls and (b) forward-compat with
  /// that future API.
  ///
  /// Serialized against itself and [`Device::current`] via
  /// `DEFAULT_DEVICE_LOCK`: mlx-c++'s default device is a non-atomic
  /// function-static, so concurrent safe-Rust mutation would be a C++ data
  /// race. The lock makes the safe API race-free; raw `mlxrs-sys` callers
  /// bypass it (their responsibility).
  pub fn set_default(&self) -> Result<()> {
    ensure_handler_installed();
    let _g = DEFAULT_DEVICE_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    check(unsafe { mlxrs_sys::mlx_set_default_device(self.0) })
  }

  /// Returns the [`DeviceKind`] (Cpu / Gpu) of this device. Wraps
  /// `mlx_device_get_type`.
  pub fn kind(&self) -> Result<DeviceKind> {
    ensure_handler_installed();
    let mut raw: mlxrs_sys::mlx_device_type = 0;
    check(unsafe { mlxrs_sys::mlx_device_get_type(&mut raw, self.0) })?;
    DeviceKind::from_raw(raw)
  }

  /// Returns the index of this device (0-based; useful for multi-GPU CUDA
  /// builds — Metal currently exposes a single GPU at index 0). Wraps
  /// `mlx_device_get_index`.
  pub fn index(&self) -> Result<i32> {
    ensure_handler_installed();
    let mut idx: i32 = 0;
    check(unsafe { mlxrs_sys::mlx_device_get_index(&mut idx, self.0) })?;
    Ok(idx)
  }

  /// Whether this device is available on the current system. Wraps
  /// `mlx_device_is_available`.
  pub fn is_available(&self) -> Result<bool> {
    ensure_handler_installed();
    let mut avail = false;
    check(unsafe { mlxrs_sys::mlx_device_is_available(&mut avail, self.0) })?;
    Ok(avail)
  }

  /// Whether two devices refer to the same `{kind, index}` pair. Wraps
  /// `mlx_device_equal`.
  pub fn equal(&self, other: &Device) -> bool {
    unsafe { mlxrs_sys::mlx_device_equal(self.0, other.0) }
  }

  /// Borrow the raw mlx-c handle (does not transfer ownership).
  ///
  /// # Safety
  /// Caller must not call `mlx_device_free` on the returned handle and must
  /// not retain it past `self`'s lifetime.
  #[inline]
  pub unsafe fn as_raw(&self) -> mlxrs_sys::mlx_device {
    self.0
  }
}

impl PartialEq for Device {
  fn eq(&self, other: &Self) -> bool {
    self.equal(other)
  }
}

impl Eq for Device {}

impl std::fmt::Debug for Device {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // Reaches fallible mlx-c (mlx_device_tostring); install the handler
    // first per the error.rs contract so a stripped/disabled ctor can't
    // let mlx's default printf+exit abort the process. (No poison concept
    // for Device — it is not thread-affine.)
    crate::error::ensure_handler_installed();
    // Borrow the raw bytes from mlx_string. RAII via the local guard so a
    // panic in `write!` still frees the string.
    let mut s = unsafe { mlxrs_sys::mlx_string_new() };
    let rc = unsafe { mlxrs_sys::mlx_device_tostring(&mut s, self.0) };
    let result = if rc == 0 {
      let p = unsafe { mlxrs_sys::mlx_string_data(s) };
      if p.is_null() {
        write!(f, "Device(<unprintable>)")
      } else {
        let cs = unsafe { CStr::from_ptr(p) };
        write!(f, "Device({})", cs.to_string_lossy())
      }
    } else {
      write!(f, "Device(<unprintable>)")
    };
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(s);
    }
    result
  }
}
