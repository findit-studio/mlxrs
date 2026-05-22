//! Integration tests for [`mlxrs::vlm::feature_cache::VisionFeatureCache`],
//! hand-traced 1:1 from `mlx-vlm/mlx_vlm/tests/test_vision_cache.py`
//! (`TestVisionFeatureCache`), the authoritative behavior spec for the
//! ported `vision_cache.py::VisionFeatureCache`.
//!
//! Each test below names the Python test it mirrors. Where the Python
//! asserts `mx.array_equal(result, features)`, the Rust analogue compares
//! the hit's `to_vec` content (the crate's standard array-identity check —
//! `Array` has no `PartialEq`; see `tests/lm_cache_arrays.rs`), which is a
//! strictly stronger check that the cached buffer round-trips byte-for-byte.

#![cfg(feature = "vlm")]

use mlxrs::{
  Array,
  error::Error,
  vlm::feature_cache::{DEFAULT_MAX_SIZE, Key, VisionFeatureCache},
};

/// A `[1, n, d]` feature tensor filled with `value` — the shape/`mx.ones`
/// pattern the Python tests use (e.g. `mx.ones((1, 280, 1536))`).
fn features(n: usize, d: usize, value: f32) -> Array {
  Array::full::<f32>(&(1usize, n, d), value).unwrap()
}

/// Materialize a feature `Array`'s contents for identity comparison.
fn contents(mut a: Array) -> Vec<f32> {
  a.to_vec::<f32>().unwrap()
}

// ───────────────────────── put / get round-trip ─────────────────────────

/// Mirrors `test_put_and_get`: cache a feature `Array` under a key, look
/// it up, assert the round-tripped contents match.
#[test]
fn put_and_get() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  let f = features(280, 1536, 1.0);
  cache.put(Key::from_source("image1.jpg"), &f).unwrap();

  let got = cache.get(&Key::from_source("image1.jpg")).unwrap();
  assert!(got.is_some(), "expected a cache hit for image1.jpg");
  // Stronger than the Python `array_equal`: the cached buffer round-trips
  // to identical contents AND the same shape.
  let got = got.unwrap();
  assert_eq!(got.shape(), vec![1, 280, 1536]);
  assert_eq!(contents(got), contents(f));
}

// ──────────────────────────── cache miss ────────────────────────────────

/// Mirrors `test_cache_miss`: lookup of an absent key returns `None`.
#[test]
fn cache_miss() {
  let mut cache = VisionFeatureCache::new();
  let got = cache.get(&Key::from_source("nonexistent.jpg")).unwrap();
  assert!(got.is_none(), "absent key must miss");
}

// ─────────────────────────── LRU eviction ───────────────────────────────

/// Mirrors `test_lru_eviction`: filling past capacity evicts the oldest
/// (least-recently-used) entry.
#[test]
fn lru_eviction() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();
  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap(); // evicts a

  assert!(
    cache.get(&Key::from("a.jpg")).unwrap().is_none(),
    "a.jpg should have been evicted as the LRU entry"
  );
  assert!(cache.get(&Key::from("b.jpg")).unwrap().is_some());
  assert!(cache.get(&Key::from("c.jpg")).unwrap().is_some());
  assert_eq!(cache.len(), 2, "capacity stays bounded at max_size");
}

/// Mirrors `test_lru_touch`: a `get` refreshes recency, so the entry it
/// touched is NOT the one evicted on the next overflow.
#[test]
fn lru_touch_refreshes_recency() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();

  // Touch `a` → `b` becomes the LRU.
  assert!(cache.get(&Key::from("a.jpg")).unwrap().is_some());

  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap(); // evicts b
  assert!(
    cache.get(&Key::from("a.jpg")).unwrap().is_some(),
    "a was touched and must survive"
  );
  assert!(
    cache.get(&Key::from("b.jpg")).unwrap().is_none(),
    "b was the LRU after a's touch and must be evicted"
  );
  assert!(cache.get(&Key::from("c.jpg")).unwrap().is_some());
}

// ──────────────────── distinct keys don't collide ───────────────────────

/// Distinct single-source keys map to distinct entries (no collision):
/// each key returns its own features.
#[test]
fn distinct_keys_do_not_collide() {
  let mut cache = VisionFeatureCache::with_max_size(4).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();
  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap();

  assert_eq!(
    contents(cache.get(&Key::from("a.jpg")).unwrap().unwrap())[0],
    1.0
  );
  assert_eq!(
    contents(cache.get(&Key::from("b.jpg")).unwrap().unwrap())[0],
    2.0
  );
  assert_eq!(
    contents(cache.get(&Key::from("c.jpg")).unwrap().unwrap())[0],
    3.0
  );
  assert_eq!(cache.len(), 3);
}

// ───────────────────────── multi-image key ──────────────────────────────

/// Mirrors `test_multi_image_key`: a list key is order-sensitive —
/// `["img1", "img2"]` and `["img2", "img1"]` are different entries.
#[test]
fn multi_image_key_order_matters() {
  let mut cache = VisionFeatureCache::new();
  let f = features(560, 1536, 1.0);
  cache
    .put(Key::from_sources(&["img1.jpg", "img2.jpg"]), &f)
    .unwrap();

  assert!(
    cache
      .get(&Key::from_sources(&["img1.jpg", "img2.jpg"]))
      .unwrap()
      .is_some(),
    "same-order multi-image key must hit"
  );
  assert!(
    cache
      .get(&Key::from_sources(&["img2.jpg", "img1.jpg"]))
      .unwrap()
      .is_none(),
    "reversed-order multi-image key must miss (order is significant)"
  );
}

/// A single-element `from_sources` equals `from_source` of that element
/// (faithful to Python `"|".join(["x"]) == "x"`), and an empty slice is
/// the empty-string key.
#[test]
fn from_sources_join_semantics() {
  assert_eq!(Key::from_sources(&["x.jpg"]), Key::from_source("x.jpg"));
  assert_eq!(Key::from_sources(&[]).as_str(), "");
  assert_eq!(Key::from_sources(&["a", "b"]).as_str(), "a|b");
}

// ─────────────────────────────── URL key ────────────────────────────────

/// Mirrors `test_url_key`: a URL string is a valid (verbatim) key.
#[test]
fn url_key() {
  let mut cache = VisionFeatureCache::new();
  let url = "https://example.com/image.jpg";
  cache
    .put(Key::from(url), &features(280, 1536, 1.0))
    .unwrap();
  assert!(cache.get(&Key::from(url)).unwrap().is_some());
}

// ──────────────────────────── content-hash key ──────────────────────────

/// The PIL/content-hash branch (`Key::from_bytes`): identical bytes hash
/// to the same key (hit), different bytes to different keys (no collision),
/// and the key is `pil:`-prefixed so it can't alias a literal path.
#[test]
fn content_hash_key() {
  let mut cache = VisionFeatureCache::new();
  let img_a = [1u8, 2, 3, 4, 5];
  let img_b = [9u8, 8, 7, 6, 5];

  cache
    .put(Key::from_bytes(&img_a), &features(10, 64, 1.0))
    .unwrap();

  // Same bytes → same key → hit.
  assert!(
    cache
      .get(&Key::from_bytes(&[1u8, 2, 3, 4, 5]))
      .unwrap()
      .is_some()
  );
  // Different bytes → different key → miss.
  assert!(cache.get(&Key::from_bytes(&img_b)).unwrap().is_none());
  // Prefixed so it never aliases a real path.
  assert!(Key::from_bytes(&img_a).as_str().starts_with("pil:"));
}

// ──────────────────────────────── clear ─────────────────────────────────

/// Mirrors `test_clear` + `test_clear_releases_all`: clear empties the
/// cache and every prior key subsequently misses; max_size is retained.
#[test]
fn clear_empties_and_releases_all() {
  let mut cache = VisionFeatureCache::with_max_size(8).unwrap();
  for i in 0..5 {
    cache
      .put(
        Key::from(format!("img{i}.jpg").as_str()),
        &features(10, 64, i as f32),
      )
      .unwrap();
  }
  assert_eq!(cache.len(), 5);

  cache.clear();
  assert_eq!(cache.len(), 0);
  assert!(cache.is_empty());
  assert_eq!(cache.max_size(), 8, "max_size survives clear");
  for i in 0..5 {
    assert!(
      cache
        .get(&Key::from(format!("img{i}.jpg").as_str()))
        .unwrap()
        .is_none(),
      "every cleared key must miss"
    );
  }
}

// ─────────────────────────────── contains ───────────────────────────────

/// Mirrors `test_contains`: `contains` reports membership without
/// disturbing recency.
#[test]
fn contains_reports_membership() {
  let mut cache = VisionFeatureCache::new();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  assert!(cache.contains(&Key::from("a.jpg")));
  assert!(!cache.contains(&Key::from("b.jpg")));
}

/// `contains` is a pure read: it does NOT refresh LRU recency (matching
/// the reference's `__contains__`, which does not `move_to_end`). After
/// `contains(a)`, `a` is still the LRU and is the one evicted on overflow.
#[test]
fn contains_does_not_refresh_recency() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();

  // `contains` must NOT touch recency, so `a` remains the LRU.
  assert!(cache.contains(&Key::from("a.jpg")));

  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap();
  assert!(
    cache.get(&Key::from("a.jpg")).unwrap().is_none(),
    "contains must not have refreshed a's recency; a stays the LRU"
  );
  assert!(cache.get(&Key::from("b.jpg")).unwrap().is_some());
}

// ──────────────────────── overwrite existing key ────────────────────────

/// Mirrors `test_overwrite_existing_key`: re-`put` of an existing key
/// overwrites the value without growing the cache.
#[test]
fn overwrite_existing_key() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 5.0))
    .unwrap();

  assert_eq!(cache.len(), 1, "overwrite must not grow the cache");
  let got = cache.get(&Key::from("a.jpg")).unwrap().unwrap();
  assert_eq!(contents(got)[0], 5.0, "value must be the overwritten one");
}

/// Overwriting also refreshes recency (the reference's `move_to_end` on
/// the present-key branch): an overwritten entry survives the next
/// overflow over a stale neighbor.
#[test]
fn overwrite_refreshes_recency() {
  let mut cache = VisionFeatureCache::with_max_size(2).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();

  // Re-put `a` → `a` is now MRU, `b` is the LRU.
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 9.0))
    .unwrap();

  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap(); // evicts b
  assert!(cache.get(&Key::from("a.jpg")).unwrap().is_some());
  assert!(
    cache.get(&Key::from("b.jpg")).unwrap().is_none(),
    "b was the LRU after a's overwrite and must be evicted"
  );
}

// ───────────────────────────── default size ─────────────────────────────

/// Mirrors `test_default_max_size`: the no-argument constructor uses the
/// reference default of 20.
#[test]
fn default_max_size_is_20() {
  let cache = VisionFeatureCache::new();
  assert_eq!(cache.max_size(), 20);
  assert_eq!(DEFAULT_MAX_SIZE, 20);
  assert_eq!(VisionFeatureCache::default().max_size(), 20);
}

// ─────────────────────── zero-capacity rejected ─────────────────────────

/// mlxrs deviation: a zero `max_size` is rejected (a cache that can hold
/// nothing is a misuse). Documented in the module's "Deviations" note.
#[test]
fn zero_max_size_is_rejected() {
  let err = VisionFeatureCache::with_max_size(0).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "zero capacity must be a ShapeMismatch error, got {err:?}"
  );
}

/// A pathological `max_size` (here `usize::MAX`) must construct cheaply.
///
/// CHANGED (Codex review): the constructor no longer pre-reserves the raw
/// `max_size`, so it allocates nothing proportional to it — the LRU eviction
/// self-bounds live entries to `max_size`, making the upfront reserve
/// unnecessary. Before this change the constructor `try_reserve`d `max_size`
/// slots, so `usize::MAX` overflowed the `capacity * size_of` computation and
/// returned `Err(OutOfMemory)`. With empty-init nothing is reserved, so
/// `usize::MAX` is just a (huge) eviction bound and construction succeeds
/// *without* allocating — the abort/DoS class is gone because nothing is
/// pre-allocated. (The zero-capacity `ShapeMismatch` guard still runs first;
/// `usize::MAX` is non-zero, so it passes the guard and constructs `Ok`.)
#[test]
fn with_max_size_pathological_capacity_is_cheap_not_abort() {
  let cache = VisionFeatureCache::with_max_size(usize::MAX)
    .expect("usize::MAX max_size must construct (empty-init reserves nothing)");
  // Nothing was allocated for the bound: the cache is empty and the huge
  // value is retained verbatim as the eviction bound.
  assert_eq!(cache.len(), 0, "no entries are pre-allocated");
  assert_eq!(
    cache.max_size(),
    usize::MAX,
    "the requested (pathological) max_size is kept as the eviction bound"
  );
}

/// A large-but-allocatable `max_size` must NOT eagerly allocate memory
/// proportional to it (Codex review — DoS-on-success guard).
///
/// `1 << 40` (~1 trillion) is allocatable as a *number* but reserving that
/// many HashMap/VecDeque slots would consume terabytes — the exact
/// "converted from abort to memory-exhaustion-on-success" hazard the
/// empty-init fix removes. With empty-init the constructor reserves nothing,
/// so this returns `Ok` immediately and cheaply: an empty cache whose
/// `max_size` is the requested value, ready to grow lazily (bounded by LRU
/// eviction) only as real entries arrive. Pre-fix this line would either
/// OOM-abort or consume gigabytes before returning.
#[test]
fn with_max_size_large_value_does_not_eagerly_allocate() {
  let big = 1usize << 40;
  let cache = VisionFeatureCache::with_max_size(big)
    .expect("a large max_size must construct cheaply with no upfront reserve");
  assert_eq!(cache.len(), 0, "empty-init: no entries pre-allocated");
  assert!(cache.is_empty());
  assert_eq!(
    cache.max_size(),
    big,
    "the requested max_size is the (lazily applied) eviction bound"
  );
}

// ───────────────── fill exactly to capacity (no eviction) ────────────────

/// Filling to EXACTLY capacity must not evict anything — eviction only
/// triggers on the entry that would exceed `max_size`.
#[test]
fn fill_to_capacity_no_eviction() {
  let mut cache = VisionFeatureCache::with_max_size(3).unwrap();
  cache
    .put(Key::from("a.jpg"), &features(10, 64, 1.0))
    .unwrap();
  cache
    .put(Key::from("b.jpg"), &features(10, 64, 2.0))
    .unwrap();
  cache
    .put(Key::from("c.jpg"), &features(10, 64, 3.0))
    .unwrap();

  assert_eq!(cache.len(), 3);
  assert!(cache.contains(&Key::from("a.jpg")));
  assert!(cache.contains(&Key::from("b.jpg")));
  assert!(cache.contains(&Key::from("c.jpg")));
}
