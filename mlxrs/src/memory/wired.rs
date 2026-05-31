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

/// Shared install-state for the process-global wired-memory limit.
///
/// Carried inside the `WIRED_LIMIT_STATE` mutex; populated by the first
/// concurrent [`WiredLimitGuard::install`] in an epoch and cleared by the
/// last live guard's [`Drop`] (refcount → 0).
struct WiredLimitState {
  /// The wired-memory limit captured BEFORE this owner-group installed
  /// the recommended limit. Restored only when [`refcount`] drops to 0.
  ///
  /// [`refcount`]: WiredLimitState::refcount
  old_limit: u64,
  /// Number of currently-live [`WiredLimitGuard`] instances sharing this
  /// state. Incremented in [`WiredLimitGuard::install`], decremented in
  /// [`Drop`]. When this hits 0 the state is cleared and [`old_limit`]
  /// is restored.
  ///
  /// [`old_limit`]: WiredLimitState::old_limit
  refcount: usize,
}

/// Process-global install-state for [`WiredLimitGuard`].
///
/// A design with NO synchronization on
/// the install/Drop read-modify-restore cycle would let two concurrent guards
/// on different threads interleave their captures and corrupt the
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
/// ## Design: refcounted-guard
/// A **single-active-guard** approach (concurrent installs
/// returned `Ok(None)`) would silently lose
/// in-scope protection for the second concurrent install:
/// ```text
///   T1 install      → captures L0, sets recommended limit, returns Some(guard)
///   T2 install      → sees flag set, returns Ok(None) (NO GUARD)
///   T2 continues memory-sensitive work assuming protection
///   T1 drops        → restores L0 (the ORIGINAL, lower limit)
///   T2 still running → now has L0 (UN-PROTECTED) for the rest of its scope
/// ```
/// The flag-check captured *intent* correctly but lost *lifetime* protection
/// for T2. We replace it with **refcounted-guard** semantics:
/// - The first install in an epoch captures the prior limit (`L0`), sets
///   the limit to the recommended value, and creates a [`WiredLimitState`]
///   with `refcount = 1`.
/// - Every concurrent install in the same epoch bumps `refcount` and yields
///   its own `Some(guard)` (NOT `Ok(None)`); the limit stays at recommended
///   for the new guard's full scope.
/// - Every [`Drop`] decrements `refcount`. Only when `refcount` hits 0 does
///   the captured `L0` get restored and the state cleared.
///
/// ## Comparison vs Python `wired_limit` context manager
/// Python's `wired_limit` is a `@contextmanager` that simply saves the prior
/// limit on entry and restores it on exit. Under the GIL, nesting is safe
/// because only one context is active at a time; under genuine concurrency
/// (sub-interpreters, free-threaded Python) the same unsynchronized bug
/// would fire. mlxrs's refcounted design matches Python's *intended
/// semantics* — "stack installs, restore the original at the bottom of the
/// stack" — but enforces them race-free via this mutex. (A single-active-guard
/// design preserves race-freedom at the cost of in-scope protection for
/// the second concurrent install; the refcounted design preserves both.)
///
/// ## Lock-hold discipline
/// The `Mutex` is held ONLY across the install's read-modify-write
/// (acquire → set_wired_limit if first → record state) and the Drop's
/// refcount-and-conditional-restore. It is NOT held during the scope's
/// body, so user code in a `WiredLimitGuard`-scoped region runs
/// unsynchronized — that is the correct behavior (the body shouldn't
/// block on the limit transition).
///
/// The lock is poison-tolerant: a panic mid-install (extremely rare —
/// `set_wired_limit`'s only failure path is the FFI rc) leaves the
/// payload `None` (the install captured nothing), so subsequent installs
/// proceed normally as the start of a new epoch.
static WIRED_LIMIT_STATE: Mutex<Option<WiredLimitState>> = Mutex::new(None);

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
  // Regression guard: branching
  // `if info.ctx.is_null() { return Ok(None); }` immediately after
  // `_new()` would make this function ALWAYS return None on every host
  // (because `_new()` *always* returns NULL ctx). That would silently turn the
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
/// `install` returns `Ok(None)` ONLY when the wired-limit surface is
/// unavailable on this platform (non-Metal build, CPU-only mlx, or any
/// environment where [`recommended_working_set_bytes`] returns
/// `Ok(None)`). Mirrors the Python helper's `if not mx.metal.is_available():
/// yield` early return.
///
/// ## Concurrency
/// Concurrent installs use **refcounted-guard** semantics:
/// - The first install in an epoch captures the prior limit, sets the
///   limit to recommended, and yields `Ok(Some(guard))`.
/// - Every subsequent install while the first epoch is still live bumps
///   an internal refcount and *also* yields `Ok(Some(guard))`. The limit
///   stays at recommended for the new guard's full scope — its caller
///   gets the protection it asked for.
/// - Each [`Drop`] decrements the refcount. Only when the last guard in
///   the epoch drops does the limit restore to the originally-captured
///   prior value.
///
/// This matches Python's implicit-stacking semantics (Python's GIL hides
/// the race; mlxrs makes it genuinely race-free via the internal mutex).
/// See the crate-private `WIRED_LIMIT_STATE` static's doc-comment for the
/// refcounted design rationale.
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
  /// Streams to [`Stream::synchronize`] before this guard's contribution
  /// to the refcount is released. Empty → synchronize the default GPU
  /// stream (mirrors Python's `mx.synchronize()` no-arg call).
  ///
  /// The captured prior limit lives in the crate-private process-global
  /// `WIRED_LIMIT_STATE` static, shared across all currently-live guards
  /// in this epoch — only the LAST drop restores it.
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

    // Refcounted-guard semantics. Acquire the
    // process-global state mutex BEFORE the conditional read-modify-write
    // so two concurrent installs cannot both observe `None` and race on
    // the FFI capture. Crucially, do NOT bail out with `Ok(None)` on
    // a concurrent install — instead bump the refcount and yield a real
    // guard, so the second caller's scope is genuinely protected for its
    // full lifetime. See WIRED_LIMIT_STATE's doc-comment for the design.
    let mut state = WIRED_LIMIT_STATE
      .lock()
      .unwrap_or_else(PoisonError::into_inner);
    match &mut *state {
      Some(s) => {
        // Already-installed: bump refcount and yield a guard that
        // participates in the cleanup. Limit is ALREADY at recommended
        // (from this epoch's first install), so DO NOT re-set it. The
        // caller's scope is protected until its own Drop decrements the
        // refcount.
        s.refcount = s.refcount.saturating_add(1);
      }
      None => {
        // First install in this concurrent epoch: capture the prior
        // limit and set the recommended limit. If the FFI call fails,
        // leave the state as `None` (lock-guard drop) so the next
        // install starts a fresh epoch.
        let old_limit = set_wired_limit(max_rec_size)?;
        *state = Some(WiredLimitState {
          old_limit,
          refcount: 1,
        });
      }
    }
    drop(state); // release before yielding to user code
    Ok(Some(Self { streams }))
  }

  /// The prior wired-memory limit captured by the first install in this
  /// concurrent epoch, in bytes. Restored when the LAST guard in the
  /// epoch drops (refcount → 0). Exposed for diagnostics / round-trip
  /// assertions.
  ///
  /// Returns `0` if the shared state has been concurrently cleared between
  /// this guard's drop and the call (a degenerate case that does not occur
  /// while `self` is alive — the guard itself is part of the refcount).
  ///
  /// Note: under refcounted semantics, ALL live guards in a single epoch
  /// share the same `old_limit` (the value captured by the *first*
  /// install in the epoch). This matches the restored-final-state
  /// invariant: when the last guard drops, the limit returns to the value
  /// that was current BEFORE the epoch started.
  #[inline(always)]
  pub fn old_limit(&self) -> u64 {
    let state = WIRED_LIMIT_STATE
      .lock()
      .unwrap_or_else(PoisonError::into_inner);
    state.as_ref().map(|s| s.old_limit).unwrap_or(0)
  }
}

impl std::fmt::Debug for WiredLimitGuard<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("WiredLimitGuard")
      .field("streams_count", &self.streams.len())
      .finish()
  }
}

impl Drop for WiredLimitGuard<'_> {
  fn drop(&mut self) {
    // Mirrors Python's `finally:` — synchronize the streams (or the default
    // stream if none), then decrement the refcount and restore the prior
    // limit IFF this was the last live guard in the epoch. Errors are
    // silently dropped per the crate's `Drop` convention (must not panic,
    // must not call check() through the TLS), the same as `Stream::drop`.
    //
    // Drop MUST be infallible.
    // `Stream::default_gpu()` and `Stream::synchronize()` both
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

    // Refcount-aware restore. Decrement the shared
    // refcount under the state lock; only the LAST guard in the epoch
    // restores the captured prior limit and clears the state. See
    // [`WIRED_LIMIT_STATE`]'s doc-comment for the rationale.
    let mut state = WIRED_LIMIT_STATE
      .lock()
      .unwrap_or_else(PoisonError::into_inner);
    match &mut *state {
      Some(s) if s.refcount > 1 => {
        // Another live guard still depends on the recommended limit;
        // just decrement and leave the FFI limit untouched.
        s.refcount -= 1;
      }
      Some(s) => {
        // Last live guard in this epoch. Restore the originally-captured
        // limit and clear the state so the next install starts a fresh
        // epoch. The FFI rc is discarded per the crate Drop convention.
        let _ = set_wired_limit(s.old_limit);
        *state = None;
      }
      None => {
        // Unreachable — a live `WiredLimitGuard` always corresponds to a
        // `Some` state with refcount ≥ 1 (it was installed under the
        // same lock). Defensive: skip silently rather than panic in
        // Drop. The `debug_assert!` surfaces the violation in debug
        // builds without ever aborting on the user's path.
        debug_assert!(
          false,
          "WiredLimitGuard dropped with no WIRED_LIMIT_STATE — \
           install/Drop refcount invariant violated"
        );
      }
    }
  }
}
