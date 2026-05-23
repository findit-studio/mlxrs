//! Wired-memory limit — `mlx_set_wired_limit` thin wrapper, recommended
//! working-set query, and the RAII [`WiredLimitGuard`] scope guard (port of
//! mlx-lm's `wired_limit(model, streams=None)` `@contextmanager`).
//!
//! ## References
//! - Python: [`mlx-lm/mlx_lm/generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
//!   `wired_limit` (lines 230-269) — `@contextmanager` that sets
//!   [`mlx_set_wired_limit`](mlxrs_sys::mlx_set_wired_limit) to the device's
//!   `max_recommended_working_set_size`, warns on >90% utilization, restores
//!   the prior limit on exit after [`Stream::synchronize`](crate::Stream::synchronize).
//! - Swift: [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
//!   `recommendedWorkingSetBytes()` (lines 5-12) — the equivalent
//!   `GPU.maxRecommendedWorkingSetBytes()` helper consulted by the LM-side
//!   policy clamps.
//! - C: [`mlx/c/memory.h`](https://github.com/ml-explore/mlx/blob/main/mlx/c/memory.h)
//!   `mlx_set_wired_limit` (out-param returns prior limit; non-zero rc on
//!   failure).
//!
//! The wired-memory limit is **process-global**: setting it on one thread
//! affects every subsequent allocation in the process, irrespective of which
//! [`crate::Device`] / [`crate::Stream`] requested it. Use the guard form to
//! ensure the prior value is restored even on early return / panic.

use crate::{
  Stream,
  error::{Result, check, ensure_handler_installed},
};

/// Set the wired-memory limit in bytes, returning the **prior** limit.
///
/// Wraps [`mlx_set_wired_limit`](mlxrs_sys::mlx_set_wired_limit). The mlx-c
/// signature is `int mlx_set_wired_limit(size_t* res, size_t limit)` where
/// `res` receives the prior limit value (mirrors Python's
/// `old = mx.set_wired_limit(new)`).
///
/// **Process-global.** Prefer the RAII [`WiredLimitGuard`] form so the prior
/// limit is restored on early return / panic. Use the raw form only for
/// ad-hoc diagnostics, or when wrapping a longer-lived control structure.
pub fn set_wired_limit(limit: u64) -> Result<u64> {
  ensure_handler_installed();
  let mut prior: usize = 0;
  // SAFETY: `&mut prior` is a valid writable `size_t*` for the call; mlx-c
  // does not retain the pointer past the call. The backend rc is surfaced
  // via `check()`.
  check(unsafe { mlxrs_sys::mlx_set_wired_limit(&mut prior, limit as usize) })?;
  Ok(prior as u64)
}

/// The Metal device's recommended working-set size in bytes, if available.
///
/// Mirrors the Swift LM-side helper
/// [`recommendedWorkingSetBytes()`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift),
/// which itself wraps `GPU.maxRecommendedWorkingSetBytes()` from mlx-swift
/// (`Source/MLX/GPU+Metal.swift::DeviceInfo.maxRecommendedWorkingSetSize`).
///
/// Returns `Ok(None)` (gracefully) when the value cannot be obtained — non-
/// Metal builds, a CPU-only mlx, or an mlx-c that does not surface the
/// `max_recommended_working_set_size` key. Returns `Err` only on a genuine
/// FFI failure of [`mlx_device_info_get_size`](mlxrs_sys::mlx_device_info_get_size)
/// for an existing key (a backend rc).
///
/// **Process-global.** The value depends only on the system GPU's
/// `recommendedMaxWorkingSetSize`; mlxrs queries it through the default GPU
/// device's `mlx_device_info` map.
pub fn recommended_working_set_bytes() -> Result<Option<u64>> {
  ensure_handler_installed();

  // GPU device — if construction fails (no Metal), surface as None per the
  // graceful-None contract; do NOT panic and do NOT propagate the FFI Err
  // (the caller asked "is there a recommended budget?", not "is the GPU
  // healthy?"). Same pattern as Python `wired_limit` gating on
  // `mx.metal.is_available()` before reading `device_info`.
  let Ok(device) = crate::Device::gpu() else {
    return Ok(None);
  };

  // SAFETY: `mlx_device_info_new()` returns a fresh empty out-param handle
  // (NULL ctx) per the mlx-c convention. Freed via the local RAII guard
  // below before any early return.
  let info = unsafe { mlxrs_sys::mlx_device_info_new() };
  if info.ctx.is_null() {
    return Ok(None);
  }
  // RAII guard so the info handle is freed on every exit path (including
  // panics inside `check`).
  struct InfoGuard(mlxrs_sys::mlx_device_info);
  impl Drop for InfoGuard {
    fn drop(&mut self) {
      // SAFETY: frees a handle this guard owns exactly once. Runs during
      // `Drop` / thread teardown: must not touch TLS, call `check()`,
      // panic, or unwind across `extern "C"`; the rc is discarded silently
      // per the crate's Drop convention.
      unsafe {
        let _ = mlxrs_sys::mlx_device_info_free(self.0);
      }
    }
  }
  let mut guard = InfoGuard(info);

  // SAFETY: `&mut guard.0` is a valid writable handle out-param; `device.0`
  // is a valid borrowed device handle for the duration of the call; mlx-c
  // does not retain either past the call; the backend rc is surfaced via
  // `check()`.
  check(unsafe { mlxrs_sys::mlx_device_info_get(&mut guard.0, device.0) })?;

  // mlx-c key for the recommended working-set size — see
  // `mlx/backend/metal/device_info.cpp`:
  //   {"max_recommended_working_set_size", ...}
  // Use a literal C string so no heap alloc / no fallible CString.
  const KEY: &[u8] = b"max_recommended_working_set_size\0";
  let key_ptr = KEY.as_ptr().cast::<std::os::raw::c_char>();

  // Probe presence first — Metal-disabled mlx builds (or hypothetical
  // future backends) may omit the key entirely. has_key=false → None.
  let mut has_key = false;
  // SAFETY: `&mut has_key` is a valid bool out-param, `guard.0` is owned by
  // this stack frame, `key_ptr` points at a `'static` NUL-terminated byte
  // literal; mlx-c does not retain any of them past the call; rc surfaced
  // via `check()`.
  check(unsafe { mlxrs_sys::mlx_device_info_has_key(&mut has_key, guard.0, key_ptr) })?;
  if !has_key {
    return Ok(None);
  }

  let mut bytes: usize = 0;
  // SAFETY: same contract as `has_key` — out-param + owned handle + static
  // key; rc surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_device_info_get_size(&mut bytes, guard.0, key_ptr) })?;
  if bytes == 0 {
    // Mirrors mlx-swift `maxRecommendedWorkingSetBytes()`'s `guard maxBytes
    // > 0 else { return nil }` — 0 means "no usable value".
    return Ok(None);
  }
  Ok(Some(bytes as u64))
}

/// RAII scope guard for the wired-memory limit — port of mlx-lm's
/// `wired_limit(model, streams=None)` `@contextmanager`.
///
/// On [`WiredLimitGuard::install`] the prior limit is captured and the limit
/// is set to the device's `max_recommended_working_set_size`. On [`Drop`] the
/// supplied streams (or the default stream if none) are synchronized and the
/// prior limit is restored.
///
/// `install` returns `Ok(None)` (no guard, no-op Drop) when the wired-limit
/// surface is unavailable on this platform — non-Metal builds, CPU-only mlx,
/// or any environment where [`recommended_working_set_bytes`] returns
/// `Ok(None)`. Mirrors the Python helper's `if not mx.metal.is_available():
/// yield` early return.
///
/// Emits a `[WARNING]` to stderr (via [`eprintln`]) when `model_bytes >
/// 0.9 * max_rec_size`, matching the Python helper's near-OOM warning
/// (mlx-lm `generate.py` lines 248-256).
///
/// ## References
/// - Python: [`mlx-lm/mlx_lm/generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
///   `wired_limit` (lines 230-269) and the `Tuner.close()` symmetric pattern
///   (lines 1545-1559).
///
/// ## Usage
///
/// ```rust,ignore
/// use mlxrs::memory::WiredLimitGuard;
///
/// // Caller computes `model_bytes` from their model's parameters (sum of
/// // `Array::nbytes()` over the weight tree) before calling install.
/// let _guard = WiredLimitGuard::install(model_bytes, &[])?;
/// // ... generation loop ...
/// // Guard drops here: synchronizes default stream, restores prior limit.
/// ```
#[must_use = "drop the guard at the end of the scope to restore the prior wired-memory limit"]
pub struct WiredLimitGuard<'a> {
  /// The previous wired limit captured by
  /// [`mlx_set_wired_limit`](mlxrs_sys::mlx_set_wired_limit)'s `res`
  /// out-param. Restored on [`Drop`].
  old_limit: u64,
  /// Streams to [`Stream::synchronize`] before restoring `old_limit`. Empty
  /// → synchronize the default GPU stream (mirrors Python's
  /// `mx.synchronize()` no-arg call).
  streams: &'a [Stream],
}

impl<'a> WiredLimitGuard<'a> {
  /// Install the wired-memory limit guard.
  ///
  /// - `model_bytes`: sum of `Array::nbytes()` over the model's weight tree
  ///   (the caller computes this; Python mlx-lm uses
  ///   `tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array)
  ///   else acc, model, 0)`).
  /// - `streams`: streams to synchronize on [`Drop`] before restoring the
  ///   prior limit. Pass `&[]` to synchronize the default stream only,
  ///   matching Python's `mx.synchronize()` no-arg call.
  ///
  /// Returns:
  /// - `Ok(Some(guard))` — limit installed; guard's [`Drop`] restores it.
  /// - `Ok(None)` — wired-memory surface unavailable on this platform (no
  ///   GPU / no Metal / no `max_recommended_working_set_size` key); no-op.
  /// - `Err(_)` — a genuine FFI failure of
  ///   [`mlx_set_wired_limit`](mlxrs_sys::mlx_set_wired_limit) or the
  ///   underlying device-info query.
  pub fn install(model_bytes: u64, streams: &'a [Stream]) -> Result<Option<Self>> {
    let Some(max_rec_size) = recommended_working_set_bytes()? else {
      // Mirrors `if not mx.metal.is_available(): yield` — pure no-op.
      return Ok(None);
    };

    // Mirrors mlx-lm `generate.py` lines 248-256: stderr warning when the
    // model alone consumes >=90% of the recommended budget (likely-OOM /
    // likely-slow combo). Threshold matches Python verbatim. Uses integer
    // math to avoid any f64/u64 conversion pitfall on huge models.
    let threshold = (max_rec_size / 10).saturating_mul(9);
    if model_bytes > threshold {
      let model_mb = model_bytes >> 20;
      let max_rec_mb = max_rec_size >> 20;
      eprintln!(
        "[WARNING] Generating with a model that requires {model_mb} MB \
         which is close to the maximum recommended size of {max_rec_mb} MB. \
         This can be slow. See the documentation for possible work-arounds: \
         https://github.com/ml-explore/mlx-lm/tree/main#large-models"
      );
    }

    let old_limit = set_wired_limit(max_rec_size)?;
    Ok(Some(Self { old_limit, streams }))
  }

  /// The prior wired-memory limit captured at [`install`] time, in bytes.
  /// Restored on [`Drop`]. Exposed for diagnostics / round-trip assertions.
  ///
  /// [`install`]: WiredLimitGuard::install
  pub fn old_limit(&self) -> u64 {
    self.old_limit
  }
}

impl std::fmt::Debug for WiredLimitGuard<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("WiredLimitGuard")
      .field("old_limit", &self.old_limit)
      .field("streams_count", &self.streams.len())
      .finish()
  }
}

impl Drop for WiredLimitGuard<'_> {
  fn drop(&mut self) {
    // Mirrors Python's `finally:` — synchronize the streams (or the default
    // stream if none), then restore `old_limit`. Errors are silently
    // dropped per the crate's `Drop` convention (must not panic, must not
    // call check() through the TLS), the same as `Stream::drop`.
    //
    // SAFETY (no `unsafe` block, but the contract): the streams slice is
    // borrowed for `'a`, so all stream handles are still live; the
    // wired-limit restore uses the safe `set_wired_limit` wrapper which
    // drains its own rc on failure (rc is discarded here).
    if self.streams.is_empty() {
      // Default stream synchronize. Mirrors Python's `mx.synchronize()`
      // (no-arg). The crate's default stream is a per-thread cache; if it
      // was never initialized on this thread, `default_gpu` succeeds the
      // first time (initializes Metal) but cannot panic.
      if let Ok(s) = Stream::default_gpu() {
        let _ = s.synchronize();
      }
    } else {
      for s in self.streams {
        let _ = s.synchronize();
      }
    }
    let _ = set_wired_limit(self.old_limit);
  }
}
