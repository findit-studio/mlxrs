//! Smoke tests for the process-global mlx memory introspection wrappers
//! ([`mlxrs::memory`]). All values are byte counts of the shared mlx
//! allocator (process-global; the peak counter is monotonic within a
//! process unless explicitly reset).

use mlxrs::{Array, memory};

/// `peak_memory` returns a sensible byte count after a small allocation:
/// `> 0` (the array data is now resident) and `>=` the prior reading
/// (monotonic non-decreasing — the underlying counter is the process
/// peak).
#[test]
fn peak_memory_is_monotonic() {
  let before = memory::peak_memory().expect("mlx peak-memory FFI available");
  // Allocate + materialize a non-trivial array to push the peak forward.
  let mut buf = Array::from_slice::<f32>(&vec![1.0_f32; 4096], &(4096_usize,)).unwrap();
  let _ = buf.eval(); // ensure the alloc actually lands
  let after = memory::peak_memory().expect("mlx peak-memory FFI available");
  assert!(after > 0, "peak_memory > 0 after a real allocation");
  assert!(
    after >= before,
    "peak_memory monotonic ({after} >= {before})"
  );
}

/// `active_memory` / `cache_memory` are addressable (any process the
/// allocator has touched produces a sensible byte count; even a fresh
/// process returns `0` not an error).
#[test]
fn active_and_cache_memory_addressable() {
  let _ = memory::active_memory().expect("mlx active-memory FFI available");
  let _ = memory::cache_memory().expect("mlx cache-memory FFI available");
}
