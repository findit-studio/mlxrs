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

/// `from_sources` is order-sensitive and produces a STABLE key (same input
/// → same key). The encoding is namespaced (`l:` tag) + length-prefixed, so
/// unlike the reference's bare `'|'`-join a single-element `["x"]` does NOT
/// equal `from_source("x")` — see `key_encoding_non_aliasing` for the full
/// non-aliasing matrix. Here we only pin the stability + order contract.
#[test]
fn from_sources_is_stable_and_order_sensitive() {
  // Same input → same key (a cache lookup with a re-derived key hits).
  assert_eq!(
    Key::from_sources(&["a", "b"]),
    Key::from_sources(&["a", "b"])
  );
  // Order is significant.
  assert_ne!(
    Key::from_sources(&["a", "b"]),
    Key::from_sources(&["b", "a"])
  );
  // The empty slice is a valid, stable key (the bare `l:` tag).
  assert_eq!(Key::from_sources(&[]), Key::from_sources(&[]));
}

/// **Codex review — key encoding must not alias distinct image identities.**
///
/// The previous encoding shared one raw-string namespace across all three
/// constructors, so distinct image identities could collide and a cache hit
/// would return a *different* image's features. The fix is a namespaced
/// (`s:` / `l:` / `b:` variant tag) + length-prefixed encoding. This test
/// pins every non-aliasing guarantee.
#[test]
fn key_encoding_non_aliasing() {
  // ── cross-variant: `from_source` vs `from_sources` ──
  // Old bug: `from_source("a|b")` == `from_sources(&["a","b"])` because the
  // list was `'|'`-joined into the same raw namespace. The `s:`/`l:` tags
  // make them unconditionally distinct.
  assert_ne!(
    Key::from_source("a|b"),
    Key::from_sources(&["a", "b"]),
    "a literal '|'-bearing path must NOT alias a 2-element list key"
  );

  // ── within-list: length-prefixing kills delimiter ambiguity ──
  // Old bug: `from_sources(&["a|b"])` == `from_sources(&["a","b"])` — a bare
  // '|'-join is not injective. Length-prefixed components (`l:3:a|b` vs
  // `l:1:a1:b`) make the list encoding injective.
  assert_ne!(
    Key::from_sources(&["a|b"]),
    Key::from_sources(&["a", "b"]),
    "['a|b'] (one component) must NOT alias ['a','b'] (two components)"
  );
  // A few more length-prefix injectivity cases: any '|'-rearrangement of the
  // joined text must stay distinct.
  assert_ne!(
    Key::from_sources(&["", "a|b"]),
    Key::from_sources(&["", "a", "b"]),
  );
  assert_ne!(
    Key::from_sources(&["a", "", "b"]),
    Key::from_sources(&["a", "b"]),
    "an empty component must not vanish (it is length-prefixed '0:')"
  );

  // ── cross-variant: `from_source` vs `from_bytes` ──
  // `from_bytes` keys carry a `b:` tag; a literal source string — even one
  // shaped exactly like a digest key — gets the `s:` tag, so it can never
  // alias a real content-hash key. Probe against every possible digest by
  // hashing several distinct byte strings.
  for bytes in [
    &b""[..],
    &b"deadbeef"[..],
    &b"\x00\x01\x02"[..],
    &[0xffu8; 32][..],
  ] {
    let hashed = Key::from_bytes(bytes);
    let digest = hashed.as_str(); // e.g. "b:0123456789abcdef0123456789abcdef"
    // A literal path equal to the *encoded* digest must not alias it.
    assert_ne!(
      Key::from_source(digest),
      hashed,
      "a literal path == the encoded digest must not alias from_bytes"
    );
    // A literal path equal to the digest with the `b:` tag stripped (the
    // old `pil:`-prefix-only scheme's collision shape) must not alias it.
    let payload = digest
      .strip_prefix("b:")
      .expect("from_bytes key is b:-tagged");
    assert_ne!(
      Key::from_source(payload),
      hashed,
      "a literal path == the bare digest payload must not alias from_bytes"
    );
  }

  // ── the variant tag is not user-spoofable ──
  // A user source that literally starts with another variant's tag is still
  // unambiguously an `s:` key (the tag is *prepended*, never matched from
  // within the user's bytes).
  assert_ne!(
    Key::from_source("l:1:a"),
    Key::from_sources(&["a"]),
    "a source literally 'l:1:a' is an s: key, not the list key l:1:a"
  );
  assert_ne!(
    Key::from_source("b:0000000000000000"),
    Key::from_bytes(b"anything"),
    "a source shaped like a b: key is an s: key, not a from_bytes key"
  );
  // `from_source("s:foo")` is `s:s:foo` — it cannot collide with any other
  // variant (no other variant starts `s:`), nor with `from_source("foo")`.
  assert_ne!(Key::from_source("s:foo"), Key::from_source("foo"));

  // ── round-trip stability: same input → same key (cache hit works) ──
  assert_eq!(Key::from_source("img.jpg"), Key::from_source("img.jpg"));
  assert_eq!(Key::from_bytes(b"abc"), Key::from_bytes(b"abc"));
  assert_eq!(
    Key::from_sources(&["a|b", "c"]),
    Key::from_sources(&["a|b", "c"])
  );
  // ── distinct inputs → distinct keys (no false hit) ──
  assert_ne!(Key::from_source("x"), Key::from_source("y"));
  assert_ne!(Key::from_bytes(b"abc"), Key::from_bytes(b"abd"));
}

/// The non-aliasing encoding holds *through the cache*: a `put` under one
/// constructor's key must NOT be retrievable via an aliasing key from a
/// different constructor. This is the functional consequence of
/// `key_encoding_non_aliasing` — a wrong-image cache hit.
#[test]
fn cache_does_not_serve_aliased_keys() {
  let mut cache = VisionFeatureCache::with_max_size(8).unwrap();

  // Store under a `from_source` key that, pre-fix, aliased a list key.
  cache
    .put(Key::from_source("a|b"), &features(10, 64, 1.0))
    .unwrap();
  assert!(
    cache
      .get(&Key::from_sources(&["a", "b"]))
      .unwrap()
      .is_none(),
    "the 2-element list key must MISS — it must not alias from_source('a|b')"
  );

  // Store under a list key; the within-list-aliasing partner must miss.
  cache
    .put(Key::from_sources(&["a|b"]), &features(10, 64, 2.0))
    .unwrap();
  assert!(
    cache
      .get(&Key::from_sources(&["a", "b"]))
      .unwrap()
      .is_none(),
    "['a','b'] must MISS — it must not alias the stored ['a|b']"
  );

  // Store under a `from_bytes` key; a literal-path lookup of the digest
  // (with and without the tag) must miss.
  let f = features(10, 64, 3.0);
  cache.put(Key::from_bytes(b"deadbeef"), &f).unwrap();
  let digest = Key::from_bytes(b"deadbeef").as_str().to_owned();
  assert!(
    cache.get(&Key::from_source(&digest)).unwrap().is_none(),
    "a literal path == the encoded digest must MISS"
  );
  let payload = digest.strip_prefix("b:").unwrap().to_owned();
  assert!(
    cache.get(&Key::from_source(&payload)).unwrap().is_none(),
    "a literal path == the bare digest payload must MISS"
  );

  // Sanity: the correctly-keyed lookups still HIT (encoding is stable).
  assert!(
    cache.get(&Key::from_source("a|b")).unwrap().is_some(),
    "the original from_source('a|b') key must still hit"
  );
  assert!(
    cache.get(&Key::from_bytes(b"deadbeef")).unwrap().is_some(),
    "the original from_bytes key must still hit"
  );
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
/// to the same key (hit), different bytes to different keys (no collision for
/// any practical input), and the key carries the `b:` variant tag so it can't
/// alias a literal path (cross-variant non-aliasing is exhaustively checked in
/// `key_encoding_non_aliasing`).
///
/// **Contract — collision-resistant, not injective.** Unlike `from_source` /
/// `from_sources` (which carry the full source string and are *injective* —
/// distinct sources always yield distinct keys), `from_bytes` is a fixed-width
/// **128-bit digest** of the raw bytes. A digest cannot be injective over an
/// unbounded byte space (pigeonhole), so the guarantee is collision-*resistance*,
/// not collision-freedom: distinct bytes practically never collide (birthday
/// bound ≈2⁶⁴ images), but a collision is possible in principle. The
/// "different bytes → different key" assertions below hold for these (and any
/// realistic) inputs precisely because of that 128-bit collision-resistance;
/// they are not claiming the map is injective. See `from_bytes_is_*` for the
/// stability + width contract.
#[test]
fn content_hash_key() {
  let mut cache = VisionFeatureCache::new();
  let img_a = [1u8, 2, 3, 4, 5];
  let img_b = [9u8, 8, 7, 6, 5];

  cache
    .put(Key::from_bytes(&img_a), &features(10, 64, 1.0))
    .unwrap();

  // Same bytes → same key → hit (the digest is stable).
  assert!(
    cache
      .get(&Key::from_bytes(&[1u8, 2, 3, 4, 5]))
      .unwrap()
      .is_some()
  );
  // Different bytes → different key → miss (collision-resistant: these
  // distinct inputs digest to distinct 128-bit values).
  assert!(cache.get(&Key::from_bytes(&img_b)).unwrap().is_none());
  // `b:`-tagged so it never aliases a real path (an `s:` key).
  assert!(Key::from_bytes(&img_a).as_str().starts_with("b:"));
}

/// `from_bytes` is a STABLE, fixed-width 128-bit digest: same bytes → same key
/// (so a cache lookup with a re-derived key hits), and the encoded form is
/// always `"b:"` + 32 hex chars regardless of input length.
///
/// This pins the *digest* contract — stability + fixed width — as distinct
/// from the *injectivity* the string-carrying variants (`from_source` /
/// `from_sources`) guarantee. A digest is collision-RESISTANT, not injective
/// (see `content_hash_key`'s doc); these assertions verify only that it is a
/// well-formed, stable, 128-bit digest.
#[test]
fn from_bytes_is_stable_fixed_width_digest() {
  // Stable: identical bytes always produce the identical key.
  assert_eq!(Key::from_bytes(b"abc"), Key::from_bytes(b"abc"));
  assert_eq!(Key::from_bytes(b""), Key::from_bytes(b""));
  assert_eq!(
    Key::from_bytes(&[0xffu8; 64]),
    Key::from_bytes(&[0xffu8; 64])
  );

  // Fixed width: `"b:"` (2) + 128-bit digest as 32 lowercase hex chars = 34,
  // for ANY input length (empty, short, long).
  for bytes in [
    &b""[..],
    &b"x"[..],
    &b"a longer payload than one hash block, exercising the streaming path"[..],
    &[0u8; 1024][..],
  ] {
    let key = Key::from_bytes(bytes);
    let s = key.as_str();
    assert_eq!(s.len(), 34, "b: + 32 hex chars (128-bit digest), got {s:?}");
    let payload = s.strip_prefix("b:").expect("from_bytes key is b:-tagged");
    assert_eq!(payload.len(), 32, "128-bit digest is 32 hex chars");
    assert!(
      payload.bytes().all(|c| c.is_ascii_hexdigit()),
      "digest payload must be lowercase hex, got {payload:?}"
    );
  }

  // Collision-resistance (not injectivity): distinct inputs yield distinct
  // keys for these practical samples. The 128-bit width is what makes this
  // hold for any realistic test corpus — it is NOT a claim of injectivity.
  let samples: [&[u8]; 6] = [b"", b"a", b"b", b"ab", b"ba", &[0u8; 5]];
  for (i, &x) in samples.iter().enumerate() {
    for &y in samples.iter().skip(i + 1) {
      assert_ne!(
        Key::from_bytes(x),
        Key::from_bytes(y),
        "distinct byte inputs must digest to distinct 128-bit keys"
      );
    }
  }

  // The two domain-separated halves carry independent information: a single
  // bit flip in the input changes the digest (would not be guaranteed if both
  // halves were the same un-separated SipHash pass).
  assert_ne!(Key::from_bytes(b"\x00"), Key::from_bytes(b"\x01"));
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
    matches!(err, Error::InvariantViolation(_)),
    "zero capacity must be an InvariantViolation error, got {err:?}"
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
/// pre-allocated. (The zero-capacity `InvariantViolation` guard still runs
/// first; `usize::MAX` is non-zero, so it passes the guard and constructs `Ok`.)
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
