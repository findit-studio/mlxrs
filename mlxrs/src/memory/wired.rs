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

use std::sync::{Mutex, PoisonError};

use crate::{
  Stream,
  error::{Result, check, ensure_handler_installed},
};

/// Process-global ownership tracker for [`WiredLimitGuard`].
///
/// CODEX R1 [HIGH] F2 fix — the previous design had NO synchronization on
/// the install/Drop read-modify-restore cycle, so two concurrent guards
/// on different threads could interleave their captures and corrupt the
/// process-global `mlx_set_wired_limit` state:
/// ```text
///   T1 install: captures L0    (sets recommended)
///   T2 install: captures recommended    (sets recommended; thinks "old" = recommended!)
///   T1 drop:    restores L0    (limit = L0)
///   T2 drop:    restores recommended    (limit = recommended; T1's restore is gone)
/// ```
/// Net effect: process-global limit ends at *recommended*, not the
/// original L0, despite both scopes having "completed cleanly".
///
/// ## Design: single-active-guard
/// We adopt **single-active-guard** semantics:
/// - The first [`WiredLimitGuard::install`] holds the slot until its
///   `Drop` runs. While held, the inner `bool` is `true`.
/// - Any concurrent or nested [`WiredLimitGuard::install`] (same or
///   different thread) returns `Ok(None)` — a graceful no-op, matching
///   the same `Ok(None)` shape returned when Metal is unavailable. The
///   caller cannot distinguish "no Metal" from "already-installed
///   elsewhere"; both are legitimate "skipped" outcomes and we already
///   surface the platform-unavailable case the same way.
///
/// This is intentionally NOT Python `wired_limit`'s implicit stacking
/// (Python's GIL hides the race; under real concurrency the same bug
/// fires). True stacking would require a `Vec<(thread_id, old)>` stack
/// and stack-pop matching that opens its own holes (drop ordering ≠
/// install ordering between threads). Single-active is the simplest
/// race-free contract that preserves the "Drop restores the original"
/// invariant unconditionally.
///
/// ## Lock-hold discipline
/// The `Mutex` is held ONLY across the install's read-modify-write
/// (acquire → set_wired_limit → record state) and the Drop's
/// restore-and-release. It is NOT held during the scope's body, so user
/// code in a `WiredLimitGuard`-scoped region runs unsynchronized — that
/// is the correct behavior (the body shouldn't block on the limit
/// transition).
///
/// The lock is poison-tolerant: a panic mid-install (extremely rare —
/// `set_wired_limit`'s only failure path is the FFI rc) leaves the
/// payload `false` (the install captured nothing), so subsequent
/// installs proceed normally.
static WIRED_LIMIT_OWNER: Mutex<bool> = Mutex::new(false);

/// Restore the process-global wired-memory limit under the
/// [`WIRED_LIMIT_OWNER`] lock and release ownership. Called from
/// [`WiredLimitGuard::drop`]; SHOULD NOT panic (the lock is poison-
/// tolerant; the FFI rc is discarded per the crate Drop convention).
fn restore_wired_limit_under_lock(old_limit: u64) {
  let mut owner = WIRED_LIMIT_OWNER
    .lock()
    .unwrap_or_else(PoisonError::into_inner);
  // Restore even if the flag was somehow already `false` — the limit is
  // a process-global resource and a defensive "always restore on Drop"
  // is strictly safer than leaking a stale limit. The flag's role is to
  // serialize install/Drop, not to gate the FFI write.
  let _ = set_wired_limit(old_limit);
  *owner = false;
}

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

  // SAFETY: `mlx_device_info_new()` is the *handle constructor* — it returns
  // a fresh `mlx_device_info { ctx: nullptr }` (see
  // `mlxrs-sys/vendor/mlx-c/mlx/c/private/device.h::mlx_device_info_new_`).
  // The NULL ctx here is EXPECTED, not an error: the actual context is
  // allocated + populated by `mlx_device_info_get` below
  // (`mlxrs-sys/vendor/mlx-c/mlx/c/device.cpp::mlx_device_info_get` calls
  // `mlx_device_info_set_` which `new`s the heap map). Wrap in the local
  // RAII guard FIRST so `mlx_device_info_free` runs on every early-return
  // path (the free path is null-safe — see `mlx_device_info_free_`).
  //
  // CODEX R1 [HIGH] regression fix: the prior code branched
  // `if info.ctx.is_null() { return Ok(None); }` immediately after
  // `_new()`, making this function ALWAYS return None on every host
  // (because `_new()` *always* returns NULL ctx). That silently turned the
  // entire wired-memory feature into a no-op on supported Metal systems.
  // Don't reintroduce that check here; the populated-ctx check happens via
  // `has_key` / `get_size`'s rc surface below.
  let info = unsafe { mlxrs_sys::mlx_device_info_new() };
  // RAII guard so the info handle is freed on every exit path (including
  // panics inside `check`).
  struct InfoGuard(mlxrs_sys::mlx_device_info);
  impl Drop for InfoGuard {
    fn drop(&mut self) {
      // SAFETY: frees a handle this guard owns exactly once (null-safe per
      // `mlx_device_info_free_`). Runs during `Drop` / thread teardown:
      // must not touch TLS, call `check()`, panic, or unwind across
      // `extern "C"`; the rc is discarded silently per the crate's Drop
      // convention.
      unsafe {
        let _ = mlxrs_sys::mlx_device_info_free(self.0);
      }
    }
  }
  let mut guard = InfoGuard(info);

  // SAFETY: `&mut guard.0` is a valid writable handle out-param; `device.0`
  // is a valid borrowed device handle for the duration of the call; mlx-c
  // does not retain either past the call; the backend rc is surfaced via
  // `check()`. This call allocates + populates the underlying
  // `mlx_device_info_cpp` map (the `_new()` above only built the empty
  // handle box).
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
/// `install` returns `Ok(None)` (no guard, no-op Drop) in two cases:
/// - the wired-limit surface is unavailable on this platform (non-Metal
///   build, CPU-only mlx, or any environment where
///   [`recommended_working_set_bytes`] returns `Ok(None)`). Mirrors the
///   Python helper's `if not mx.metal.is_available(): yield` early return.
/// - another `WiredLimitGuard` is currently active on any thread. We
///   serialize ownership of the process-global wired-memory limit (see
///   [`WIRED_LIMIT_OWNER`]) for race-safety; the currently-active guard's
///   `Drop` correctly restores the prior limit, so a second install is a
///   no-op that does not corrupt the captured state. This deviates from
///   Python's implicit-stacking semantics — Python's GIL hides the same
///   race, but under real concurrency the stacking design corrupts the
///   limit (`F2` in the L6 design notes). Callers that need stacking must
///   coordinate at a higher level (e.g. a single long-lived install
///   wrapping the whole concurrent region).
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

    // CODEX R1 [HIGH] F2 fix — acquire the process-global ownership lock
    // BEFORE the read-modify-write so two concurrent installs cannot
    // interleave their captures (the prior `Drop`'s blind restore would
    // then clobber a still-active sibling's effect). See
    // WIRED_LIMIT_OWNER's doc-comment for the single-active-guard
    // semantics.
    let mut owner = WIRED_LIMIT_OWNER
      .lock()
      .unwrap_or_else(PoisonError::into_inner);
    if *owner {
      // Another `WiredLimitGuard` is currently active (same or different
      // thread). Bail out as a no-op rather than racing — the active
      // guard already pushed the limit to the recommended budget, so
      // the caller's intent ("limit at recommended for the scope") is
      // already satisfied; on the active guard's Drop the prior limit
      // is correctly restored.
      return Ok(None);
    }

    let old_limit = set_wired_limit(max_rec_size)?;
    *owner = true;
    drop(owner); // release before yielding to user code
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
    // CODEX R1 [HIGH] F3 fix — Drop MUST be infallible.
    // Previously, `Stream::default_gpu()` and `Stream::synchronize()` both
    // invoke `assert_streams_not_cleared()` which `panic!`s when the
    // thread's streams have been bulk-cleared via
    // `Stream::clear_current_thread_streams()`. A safe sequence —
    // install guard → `clear_current_thread_streams` → scope-exit drop —
    // would have panicked here, leaking the process-global wired-memory
    // limit; and if the scope-exit was *already* from an in-flight panic,
    // a panic-while-panicking double-panics → process abort.
    //
    // Use the crate-internal non-panicking variants
    // ([`Stream::try_default_gpu`] / [`Stream::try_synchronize`]) which
    // silently skip the sync step on a poisoned thread. The
    // `set_wired_limit` restore still runs unconditionally — the wired
    // limit is process-global, not per-stream, so restoring it is correct
    // (and required) even when the per-thread streams have been cleared.
    //
    // SAFETY (no `unsafe` block, but the contract): the streams slice is
    // borrowed for `'a`, so all stream handles are still live; the
    // wired-limit restore uses the safe `set_wired_limit` wrapper which
    // drains its own rc on failure (rc is discarded here).
    if self.streams.is_empty() {
      // Default stream synchronize. Mirrors Python's `mx.synchronize()`
      // (no-arg). `try_default_gpu` returns `None` on a poisoned thread
      // (or any FFI failure) instead of panicking — skip the sync step.
      if let Some(s) = Stream::try_default_gpu() {
        let _ = s.try_synchronize();
      }
    } else {
      for s in self.streams {
        // `try_synchronize` no-ops on a poisoned thread instead of
        // panicking — the original sync intent cannot complete (the
        // stream's encoders are gone) but the limit restore below still
        // runs.
        let _ = s.try_synchronize();
      }
    }
    // Restore the process-global wired-memory limit under the F2
    // owner-lock so we release the ownership slot for the next install
    // atomically with the FFI restore. See `restore_wired_limit_under_lock`
    // and the [`WIRED_LIMIT_OWNER`] doc-comment.
    restore_wired_limit_under_lock(self.old_limit);
  }
}
