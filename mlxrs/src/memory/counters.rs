//! `memory.h` peak / active / cache counter wrappers. Re-exported by
//! [`crate::memory`]; see the parent module's docs for the unified surface
//! description (counters + wired-limit guard + policies).

use crate::error::{Result, check, ensure_handler_installed};

/// The peak memory (in bytes) the process-global mlx allocator has ever
/// held since process start (or since the last [`reset_peak_memory`]).
///
/// Wraps `mlx_get_peak_memory`. mlx-lm's `mx.get_peak_memory()` (used in
/// `mlx_lm/generate.py` `stream_generate` to populate the
/// `GenerationResponse.peak_memory` field) is the same C++ entry point;
/// mlx-lm divides by `1e9` to report GB — the safe-Rust surface stays in
/// the raw byte count and lets the caller choose the scale.
pub fn peak_memory() -> Result<u64> {
  ensure_handler_installed();
  let mut bytes: usize = 0;
  // SAFETY: `&mut bytes` is a valid writable `size_t*` for the call; mlx-c
  // does not retain the pointer past the call. The backend rc is surfaced
  // via `check()`.
  check(unsafe { mlxrs_sys::mlx_get_peak_memory(&mut bytes) })?;
  Ok(bytes as u64)
}

/// Reset the peak-memory counter to the current active size.
///
/// Wraps `mlx_reset_peak_memory`. mlx-swift exposes the same hook as
/// `GPU.resetPeakMemory()`; mlx-lm uses it indirectly through
/// `mx.reset_peak_memory()` in its perf-eval scripts.
pub fn reset_peak_memory() -> Result<()> {
  ensure_handler_installed();
  // SAFETY: `mlx_reset_peak_memory` takes no arguments; the backend rc is
  // surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_reset_peak_memory() })
}

/// The mlx allocator's currently-resident bytes (excluding the recycled
/// cache).
///
/// Wraps `mlx_get_active_memory`. Mirrors mlx-swift's `GPU.activeMemory`
/// and mlx-lm's `mx.get_active_memory()` (debug / diagnostics use only;
/// `GenerationStats` itself reports the peak).
pub fn active_memory() -> Result<u64> {
  ensure_handler_installed();
  let mut bytes: usize = 0;
  // SAFETY: same `size_t*` out-param contract as `peak_memory`.
  check(unsafe { mlxrs_sys::mlx_get_active_memory(&mut bytes) })?;
  Ok(bytes as u64)
}

/// The mlx allocator's currently-recycled cache size in bytes.
///
/// Wraps `mlx_get_cache_memory`. Mirrors mlx-swift's `GPU.cacheMemory`
/// (`active + cache` is the allocator's total reservation).
pub fn cache_memory() -> Result<u64> {
  ensure_handler_installed();
  let mut bytes: usize = 0;
  // SAFETY: same `size_t*` out-param contract as `peak_memory`.
  check(unsafe { mlxrs_sys::mlx_get_cache_memory(&mut bytes) })?;
  Ok(bytes as u64)
}
