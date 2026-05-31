//! Focused tests for [`mlxrs::lm::cache::KvCache::reference_class_name`] —
//! the trait method that lifts mlx-swift-lm's `cacheClassName(_:)` switch
//! (`KVCache.swift:1381-1392`) onto the concrete cache itself.
//!
//! Each concrete cache must return the exact mlx-lm / mlx-swift reference
//! name (`type(c).__name__` / swift `cacheClassName`) so the persist
//! save/load round-trip stays byte-identical. The trait default
//! (`"KVCache"`) is the documented fallback (`KVCache.swift:1390`'s
//! `default: return "KVCache"`) and exercised here via a minimal mock
//! `KvCache` that omits the override.

#![cfg(feature = "lm")]

use mlxrs::{
  Array, Error, Result,
  lm::cache::{
    ArraysCache, BatchKvCache, BatchRotatingKvCache, CacheList, ChunkedKvCache, KvCache, MaskMode,
    RotatingKvCache, StandardKvCache, StandardQuantizedKvCache,
  },
};

#[test]
fn standard_kv_cache_reference_name_is_kvcache() {
  // mlx-lm `type(KVCache).__name__` (`cache.py:56`) / mlx-swift-lm
  // `case is KVCacheSimple: return "KVCache"` (`KVCache.swift:1388`).
  let c = StandardKvCache::new();
  assert_eq!(c.reference_class_name(), "KVCache");
}

#[test]
fn rotating_kv_cache_reference_name_is_rotating_kvcache() {
  // mlx-lm `type(RotatingKVCache).__name__` / mlx-swift-lm
  // `case is RotatingKVCache: return "RotatingKVCache"`
  // (`KVCache.swift:1386`).
  let c = RotatingKvCache::new(8, 4);
  assert_eq!(c.reference_class_name(), "RotatingKVCache");
}

#[test]
fn chunked_kv_cache_reference_name_is_chunked_kvcache() {
  // mlx-lm `type(ChunkedKVCache).__name__` / mlx-swift-lm
  // `case is ChunkedKVCache: return "ChunkedKVCache"`
  // (`KVCache.swift:1383`).
  let c = ChunkedKvCache::new(Some(16));
  assert_eq!(c.reference_class_name(), "ChunkedKVCache");
}

#[test]
fn quantized_kv_cache_reference_name_is_quantized_kvcache() {
  // mlx-lm `type(QuantizedKVCache).__name__` / mlx-swift-lm
  // `case is QuantizedKVCache: return "QuantizedKVCache"`
  // (`KVCache.swift:1387`).
  let c = StandardQuantizedKvCache::new(64, 4).unwrap();
  assert_eq!(c.reference_class_name(), "QuantizedKVCache");
}

#[test]
fn arrays_cache_reference_name_is_arrays_cache() {
  // mlx-lm `type(ArraysCache).__name__` / mlx-swift-lm
  // `case is ArraysCache: return "ArraysCache"` (`KVCache.swift:1385`).
  let c = ArraysCache::new(2);
  assert_eq!(c.reference_class_name(), "ArraysCache");
}

#[test]
fn cache_list_reference_name_is_cache_list() {
  // mlx-lm `type(CacheList).__name__` / mlx-swift-lm
  // `case is CacheList: return "CacheList"` (`KVCache.swift:1389`).
  let cl = CacheList::new(Vec::new());
  assert_eq!(cl.reference_class_name(), "CacheList");
  // And with children — the name does not depend on the contents.
  let cl2 = CacheList::new(vec![Box::new(StandardKvCache::new())]);
  assert_eq!(cl2.reference_class_name(), "CacheList");
}

#[test]
fn batch_kv_cache_reference_name_is_batch_kvcache() {
  // mlx-lm `type(BatchKVCache).__name__` (`cache.py:56`); mlx-swift-lm has
  // no `BatchKVCache` arm in `cacheClassName` (`KVCache.swift:1381-1392`).
  let c = BatchKvCache::new(&[]);
  assert_eq!(c.reference_class_name(), "BatchKVCache");
}

#[test]
fn batch_rotating_kv_cache_reference_name_is_batch_rotating_kvcache() {
  // mlx-lm `type(BatchRotatingKVCache).__name__` (`cache.py:56`);
  // mlx-swift-lm has no `BatchRotatingKVCache` arm in `cacheClassName`
  // (`KVCache.swift:1381-1392`).
  let c = BatchRotatingKvCache::new(0, &[]);
  assert_eq!(c.reference_class_name(), "BatchRotatingKVCache");
}

// ───────── required method (no default) ─────────
//
// Per issue #107, `KvCache::reference_class_name` is now a
// REQUIRED trait method (no default). Forgetting to declare it on a new
// in-tree cache is a compile error — closing the silent runtime
// label-loss the earlier `"KVCache"` default left open (a new
// state-shape-different cache would have round-tripped as a
// `StandardKvCache`).
//
// The earlier `default_dyn_kvcache_falls_back_to_kvcache` test that
// asserted the default returns `"KVCache"` is REMOVED — the default no
// longer exists, and the assertion is structurally incoherent. The
// `RotatingForwardingWrapper` tests are updated below to assert the
// required-method contract: a wrapper MUST forward
// `reference_class_name` to its inner cache (one-line method body) and
// in-tree caches/wrappers do so explicitly.

/// Minimal `KvCache` impl that declares `reference_class_name`
/// EXPLICITLY (required method, no default). The custom
/// `"MyCustomCache"` name demonstrates the required-method contract — a
/// new cache type chooses its own class name at trait-impl site, a
/// compile-time guarantee.
struct ExplicitMock;

impl KvCache for ExplicitMock {
  fn offset(&self) -> usize {
    0
  }
  fn update(&mut self, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
    Err(Error::Backend(
      "ExplicitMock::update is not used in this test".into(),
    ))
  }
  fn state(&self) -> Result<Vec<Array>> {
    Ok(Vec::new())
  }
  fn materialize(&mut self) -> Result<()> {
    Ok(())
  }
  fn set_state(&mut self, _state: Vec<Array>) -> Result<()> {
    Ok(())
  }
  fn make_mask(
    &self,
    _n: usize,
    _window_size: Option<usize>,
    _return_array: bool,
  ) -> Result<MaskMode> {
    Ok(MaskMode::None)
  }
  fn nbytes(&self) -> usize {
    0
  }
  fn is_empty(&self) -> bool {
    true
  }
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    Ok(Box::new(ExplicitMock))
  }
  // Issue #107: REQUIRED — no default.
  fn reference_class_name(&self) -> &'static str {
    "MyCustomCache"
  }
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

#[test]
fn explicit_kvcache_uses_declared_name() {
  // A new cache impl declares its name at trait-impl site (no
  // silent inheritance of a generic default). `ExplicitMock` returns
  // `"MyCustomCache"`, the exact string compiled into its impl body —
  // verifying the required-method contract holds via both direct call
  // and dynamic dispatch (the path `persist::reference_class_name`
  // uses).
  let m = ExplicitMock;
  assert_eq!(m.reference_class_name(), "MyCustomCache");
  let d: &dyn KvCache = &m;
  assert_eq!(
    d.reference_class_name(),
    "MyCustomCache",
    "the declared name MUST be observable via dynamic dispatch (the path persist::reference_class_name uses)"
  );
}

/// Wrapper around a `RotatingKvCache` that forwards every method,
/// INCLUDING `reference_class_name`. Per issue #107, every
/// impl MUST declare its class name explicitly — the trait-default
/// `"KVCache"` fallback is gone. This wrapper's `reference_class_name`
/// forwards to `self.inner.reference_class_name()` (a one-liner — the
/// documented pattern for any wrapper that wants the wrapped kind's
/// label). The earlier silent-default behavior (`"KVCache"` for any
/// type without override) would have been a compile error under the
/// required method, not silent label loss.
struct RotatingForwardingWrapper {
  inner: RotatingKvCache,
}

impl KvCache for RotatingForwardingWrapper {
  fn offset(&self) -> usize {
    self.inner.offset()
  }
  fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
    self.inner.update(keys, values)
  }
  fn state(&self) -> Result<Vec<Array>> {
    self.inner.state()
  }
  fn materialize(&mut self) -> Result<()> {
    self.inner.materialize()
  }
  fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
    self.inner.set_state(state)
  }
  fn meta_state(&self) -> Vec<String> {
    self.inner.meta_state()
  }
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    self.inner.set_meta_state(m)
  }
  fn max_size(&self) -> Option<usize> {
    self.inner.max_size()
  }
  fn make_mask(
    &self,
    n: usize,
    window_size: Option<usize>,
    return_array: bool,
  ) -> Result<MaskMode> {
    self.inner.make_mask(n, window_size, return_array)
  }
  fn nbytes(&self) -> usize {
    self.inner.nbytes()
  }
  fn is_empty(&self) -> bool {
    self.inner.is_empty()
  }
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    self.inner.copy()
  }
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
  // Issue #107: REQUIRED — forward to inner's declared name.
  fn reference_class_name(&self) -> &'static str {
    self.inner.reference_class_name()
  }
}

#[test]
fn rotating_forwarding_wrapper_inherits_inner_class_name() {
  // Contract: a wrapper that forwards
  // `reference_class_name` to its inner cache reports the inner's name —
  // exactly the documented one-liner for any wrapper that wants the
  // wrapped kind's label. Earlier this was the trait default
  // `"KVCache"` (a silent label loss for any state-shape-different
  // wrapper); now the explicit forward is the only path, which
  // means a future state-shape-different wrapper that omits the forward
  // is a COMPILE ERROR (the structural guarantee).
  let w = RotatingForwardingWrapper {
    inner: RotatingKvCache::new(8, 1),
  };
  // Sanity: the inner is a rotating cache.
  assert!(w.max_size().is_some());
  assert_eq!(w.meta_state().len(), 4);
  // The contract: explicit forward yields the inner's name.
  assert_eq!(
    w.reference_class_name(),
    "RotatingKVCache",
    "wrapper that forwards to inner.reference_class_name() reports the inner's name"
  );
  let d: &dyn KvCache = &w;
  assert_eq!(
    d.reference_class_name(),
    "RotatingKVCache",
    "via dynamic dispatch too (the path persist uses)"
  );
}

#[test]
fn rotating_forwarding_wrapper_nested_in_cache_list_inherits_inner_class_name() {
  // The "top-level vs nested" consistency check: a
  // forwarding wrapper's class name must be the same in top-level vs
  // nested-in-CacheList positions. With the trait method REQUIRED, the
  // wrapper's explicit `self.inner.reference_class_name()` is the only
  // path, so this test pins that nested classification == top-level
  // classification == inner's `RotatingKVCache`.
  let w = RotatingForwardingWrapper {
    inner: RotatingKvCache::new(8, 1),
  };
  let cl = CacheList::new(vec![Box::new(w)]);
  let meta = cl.meta_state();
  // Framing: ["1" (childCount), className, stateCount, metaCount, ...meta].
  // The className slot is meta[1] — must be the inner's declared
  // `"RotatingKVCache"`, exactly the top-level classification above.
  assert_eq!(meta[0], "1", "one child");
  assert_eq!(
    meta[1], "RotatingKVCache",
    "nested wrapper that forwards to inner reports the inner's name, \
     identical to its top-level classification"
  );
}
