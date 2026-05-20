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
    QuantizedKvCacheImpl, RotatingKvCache, StandardKvCache,
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
  let c = QuantizedKvCacheImpl::new(64, 4);
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

// ───────── trait default fallback ─────────

/// Minimal `KvCache` impl that omits the `reference_class_name` override —
/// exercises the trait default (mlx-lm `_BaseCache` / mlx-swift-lm
/// `default: return "KVCache"` at `KVCache.swift:1390`). Required so a
/// future third-party / forward-compat cache that omits the override still
/// produces a *parseable* (if generic) round-trip rather than failing to
/// dispatch.
struct DefaultMock;

impl KvCache for DefaultMock {
  fn offset(&self) -> usize {
    0
  }
  fn update(&mut self, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
    Err(Error::Backend {
      message: "DefaultMock::update is not used in this test".into(),
    })
  }
  fn state(&self) -> Result<Vec<Array>> {
    Ok(Vec::new())
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
    Ok(Box::new(DefaultMock))
  }
}

#[test]
fn default_dyn_kvcache_falls_back_to_kvcache() {
  // The trait default is `"KVCache"` — mlx-lm `_BaseCache` / mlx-swift-lm
  // `default: return "KVCache"` (`KVCache.swift:1390`). A cache that omits
  // the override inherits the generic-but-parseable name; `from_state`'s
  // `"KVCache"` arm routes such payloads to `StandardKvCache`.
  let m = DefaultMock;
  assert_eq!(m.reference_class_name(), "KVCache");
  let d: &dyn KvCache = &m;
  assert_eq!(
    d.reference_class_name(),
    "KVCache",
    "trait default must be observable via dynamic dispatch (the path persist::reference_class_name uses)"
  );
}

/// Wrapper around a `RotatingKvCache` that forwards every method **except**
/// `reference_class_name` — i.e. the exact "out-of-tree wrapper" shape the
/// adversarial review flagged. Mirrors swift's `cacheClassName` default
/// arm: a `KVCache`-conforming type that doesn't match any of the
/// `case is …:` arms falls through to `default: return "KVCache"`
/// (`KVCache.swift:1390`). The pre-PR Rust impl used a structural
/// heuristic (`max_size().is_some() && meta_state().len() == 4 ⇒
/// RotatingKVCache`) to back-discriminate such wrappers; this PR removes
/// that heuristic in favor of the trait method (faithful to swift's
/// class-identity switch). This test pins the documented behavior so a
/// future "fallback heuristic" re-add would surface as a deliberate test
/// change, not silent re-introduction of the very hack the trait method
/// replaces. (See the `On the removed heuristic` section of the trait
/// method doc in `mlxrs/src/lm/cache/mod.rs`.)
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
  // Deliberately NO `reference_class_name` override — exercises the trait
  // default fallback (the whole point of this test).
}

#[test]
fn rotating_forwarding_wrapper_without_override_classifies_as_kvcache() {
  // Mirrors swift's `default: return "KVCache"` arm exactly: a type that
  // forwards the full structural surface (`max_size().is_some()`,
  // `meta_state().len() == 4`) but omits the `reference_class_name`
  // override gets the documented default `"KVCache"`, NOT the
  // pre-PR heuristic-derived `"RotatingKVCache"`. This is the
  // swift-faithful behavior; a `forward.reference_class_name()` one-liner
  // is the standard fix for any real downstream wrapper that wants the
  // wrapped kind's label (see `mlxrs/src/lm/cache/mod.rs`
  // `# On the removed heuristic`).
  let w = RotatingForwardingWrapper {
    inner: RotatingKvCache::new(8, 1),
  };
  // Sanity: the structural shape that the pre-PR heuristic would have
  // used to classify this as `"RotatingKVCache"` is genuinely present.
  assert!(
    w.max_size().is_some(),
    "wrapper forwards max_size() (structural shape of a rotating)"
  );
  assert_eq!(
    w.meta_state().len(),
    4,
    "wrapper forwards rotating's 4-field meta_state (structural shape)"
  );
  // The documented post-PR behavior: trait-default `"KVCache"`, faithful
  // to swift's `default:` arm.
  assert_eq!(
    w.reference_class_name(),
    "KVCache",
    "wrapper without override falls through to the trait default (swift KVCache.swift:1390)"
  );
  let d: &dyn KvCache = &w;
  assert_eq!(
    d.reference_class_name(),
    "KVCache",
    "via dynamic dispatch too (the path persist uses)"
  );
}

#[test]
fn rotating_forwarding_wrapper_nested_in_cache_list_also_classifies_as_kvcache() {
  // The "top-level vs nested" consistency check: a defaulting wrapper's
  // class name must be the SAME `"KVCache"` whether it is the cache being
  // persisted directly OR a child of a `CacheList`. Pre-fix,
  // `cache_list::child_class_name_from_meta` had a structural
  // `is_cache_list_meta` fallback that would reclassify a defaulting
  // wrapper whose `meta_state` parsed as a CacheList frame back to
  // `"CacheList"`, creating an inconsistency with the trait method's
  // documented swift-default contract. This test pins the unified
  // behavior: nested == top-level == trait method == `"KVCache"`.
  let w = RotatingForwardingWrapper {
    inner: RotatingKvCache::new(8, 1),
  };
  let cl = CacheList::new(vec![Box::new(w)]);
  let meta = cl.meta_state();
  // Framing: ["1" (childCount), className, stateCount, metaCount, ...meta].
  // The className slot is meta[1] — must be the trait default `"KVCache"`,
  // not a structurally-reclassified `"CacheList"` or `"RotatingKVCache"`.
  assert_eq!(meta[0], "1", "one child");
  assert_eq!(
    meta[1], "KVCache",
    "nested defaulting wrapper must be named `KVCache` (trait default), \
     identical to its top-level classification — no structural fallback"
  );
}
