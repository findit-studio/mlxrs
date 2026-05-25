//! Process-global memory introspection — thin wrappers around mlx-c's
//! `memory.h` peak / active / cache counters, the wired-memory limit
//! [`mlx_set_wired_limit`] scope guard, and the LM-side wired-memory
//! [`WiredMemoryPolicy`] surface.
//!
//! Mirrors the surface
//! [mlx-swift `Source/MLX/Memory.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Memory.swift)
//! exposes as the `GPU.peakMemory` / `GPU.activeMemory` / `GPU.cacheMemory`
//! static properties (which in turn mirror mlx-c's
//! [`mlx_get_peak_memory`](https://github.com/ml-explore/mlx/blob/main/mlx/c/memory.h)
//! / `mlx_get_active_memory` / `mlx_get_cache_memory`), plus the
//! [`WiredMemoryPolicy`] policy layer
//! ([mlx-swift-lm `WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift))
//! and [`WiredLimitGuard`] scope guard (port of mlx-lm's
//! [`wired_limit`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
//! `@contextmanager`). Used by the
//! [`crate::lm::generate::GenerationStats`] peak-memory field (mlx-lm
//! `mx.get_peak_memory() / 1e9` in `mlx_lm/generate.py` `stream_generate`).
//!
//! All counter values are byte counts of the **process-global** mlx allocator
//! (`mlx::core::metal::allocator()`), not per-`Device` / per-`Stream`. The
//! peak counter is monotonically non-decreasing within a process unless
//! [`reset_peak_memory`] is called (mlx-c's only reset hook). The wired-memory
//! limit ([`set_wired_limit`] / [`WiredLimitGuard`]) is likewise
//! process-global.
//!
//! [`mlx_set_wired_limit`]: mlxrs_sys::mlx_set_wired_limit

mod counters;
mod policies;
mod wired;

pub use self::{
  counters::{active_memory, cache_memory, peak_memory, reset_peak_memory},
  policies::{
    WiredBudgetPolicy, WiredFixedPolicy, WiredMaxPolicy, WiredMemoryMeasurement, WiredMemoryPolicy,
    WiredSumPolicy, tune,
  },
  wired::{WiredLimitGuard, recommended_working_set_bytes, set_wired_limit},
};
