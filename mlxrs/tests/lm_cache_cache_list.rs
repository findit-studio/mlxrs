//! Deterministic tests for [`mlxrs::lm::cache::CacheList`], the composite
//! cache, ported from `mlx_lm.models.cache.CacheList`
//! (`mlx_lm/models/cache.py:814-902`) and cross-checked against
//! mlx-swift-lm's `MLXLMCommon` `CacheList`
//! (`Libraries/MLXLMCommon/KVCache.swift:1248-1370`).
//!
//! Children are real `StandardKvCache` / `RotatingKvCache` instances built
//! via their public constructors. KV tensors are tiny 4-D
//! `[B, n_kv_heads, S, head_dim]` arrays so every retained-token identity is
//! readable from `to_vec`.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{CacheList, ChunkedKvCache, KvCache, RotatingKvCache, StandardKvCache, from_state},
};

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// each step's value being its 0-based token id.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

/// A `CacheList` over a `StandardKvCache` (child 0) and a `RotatingKvCache`
/// (child 1), each pre-fed a few tokens so every child has non-empty state.
fn populated_pair() -> CacheList {
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let mut r = RotatingKvCache::new(8, 4);
  for i in 0..5 {
    let t = kv(&[i as f32]);
    r.update(&t, &t).unwrap();
  }
  CacheList::new(vec![Box::new(s), Box::new(r)])
}

/// `__getitem__` (cache.py:818-819) / Swift `subscript` (KVCache.swift:1266):
/// `get(i)` yields the i-th child; out-of-range is `None` (no panic).
#[test]
fn get_indexes_children() {
  let cl = populated_pair();
  assert_eq!(cl.len(), 2);
  // child 0 is the StandardKvCache (offset 3); child 1 the Rotating (5).
  assert_eq!(cl.get(0).unwrap().offset(), 3);
  assert_eq!(cl.get(1).unwrap().offset(), 5);
  assert_eq!(cl.get(1).unwrap().max_size(), Some(8));
  assert!(cl.get(2).is_none(), "out-of-range index must be None");
}

/// `offset()` — Python `CacheList.size()` = `max(c.size() for c in caches)`
/// (cache.py:884-885); each child's `size()` is its `offset`, so the
/// composite offset is the max child offset (Standard=3, Rotating=5 -> 5).
#[test]
fn offset_is_max_child_offset() {
  let cl = populated_pair();
  assert_eq!(cl.offset(), 5);

  // Empty list -> 0 (max of nothing; mlx-lm `max(...)` would raise, but a
  // recoverable 0 matches `_BaseCache.size()`'s "always 0" default).
  let empty = CacheList::new(Vec::new());
  assert_eq!(empty.offset(), 0);
}

/// `is_trimmable()` = `all(c.is_trimmable() ...)` (cache.py:821-822 / Swift
/// KVCache.swift:1293-1295). Standard is always trimmable; Rotating only
/// while `offset < max_size`.
#[test]
fn is_trimmable_is_all_children() {
  // Standard (always trimmable) + Rotating with offset 5 < max_size 8
  // (trimmable) -> all trimmable.
  let cl = populated_pair();
  assert!(cl.is_trimmable());

  // Fill the Rotating past its window so it is NOT trimmable; the whole
  // list must then report not-trimmable.
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0]), &kv(&[0.0])).unwrap();
  let mut r = RotatingKvCache::new(4, 2);
  for i in 0..6 {
    let t = kv(&[i as f32]);
    r.update(&t, &t).unwrap();
  }
  assert!(!r.is_trimmable(), "rotating must be full / not trimmable");
  let cl2 = CacheList::new(vec![Box::new(s), Box::new(r)]);
  assert!(
    !cl2.is_trimmable(),
    "one non-trimmable child => list not trimmable"
  );

  // Empty list: `all(...)` over no children is True (cache.py:822).
  assert!(CacheList::new(Vec::new()).is_trimmable());
}

/// `trim(n)` loops every child calling `c.trim(n)` and returns the **last**
/// child's result (cache.py:824-827 / Swift KVCache.swift:1297-1304). Every
/// child must actually be trimmed (offset moved), but only the last child's
/// trimmed count is returned.
#[test]
fn trim_delegates_to_all_returns_last() {
  let mut cl = populated_pair();
  // Standard offset 3, Rotating offset 5. trim(2):
  //  - Standard trims min(3,2)=2 -> offset 1
  //  - Rotating  trims min(5,2)=2 -> offset 3  (returned value)
  let returned = cl.trim(2).unwrap();
  assert_eq!(returned, 2, "returns the LAST child's trimmed count");
  assert_eq!(cl.get(0).unwrap().offset(), 1, "child 0 also trimmed");
  assert_eq!(cl.get(1).unwrap().offset(), 3, "child 1 trimmed");

  // The "last result wins" semantics matter when children trim different
  // amounts. Standard offset 1 (can trim 1), Rotating offset 3 (can trim 3).
  // trim(3): Standard -> min(1,3)=1, Rotating -> min(3,3)=3; returns 3.
  let returned2 = cl.trim(3).unwrap();
  assert_eq!(
    returned2, 3,
    "last child (Rotating) trimmed 3 -> that is the returned value"
  );
  assert_eq!(cl.get(0).unwrap().offset(), 0);
  assert_eq!(cl.get(1).unwrap().offset(), 0);

  // Empty list: nothing to trim, like mlx-lm's loop never assigning `m`
  // (our recoverable choice: 0 trimmed).
  let mut empty = CacheList::new(Vec::new());
  assert_eq!(empty.trim(5).unwrap(), 0);
}

/// `state()` is the **flattened** concatenation of every child's state
/// (Swift `caches.flatMap { $0.state }`, KVCache.swift:1274-1275 — the only
/// shape representable by the merged `state() -> Vec<Array>` signature; the
/// per-child grouping Python keeps is recoverable from `meta_state`'s
/// per-child `stateCount`). Standard has 2 arrays, Rotating 2 -> 4 total.
#[test]
fn state_is_flattened_child_states() {
  let cl = populated_pair();
  let st = cl.state().unwrap();
  assert_eq!(st.len(), 4, "2 (Standard k,v) + 2 (Rotating k,v)");

  // First two are the Standard child's k/v (ids 0,1,2).
  let mut k0 = st[0].try_clone().unwrap();
  assert_eq!(k0.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);

  // An empty list has empty state.
  assert!(CacheList::new(Vec::new()).state().unwrap().is_empty());
}

/// `meta_state()` carries each child's REFERENCE class name plus the
/// per-child `stateCount`/`metaCount` framing so `from_state` can split the
/// flattened arrays back per child (Swift KVCache.swift:1315-1327 format
/// `[childCount, (className, stateCount, metaCount, ...meta)*]`; the
/// className is mlx-lm's `type(c).__name__`, cache.py:841).
#[test]
fn meta_state_carries_reference_class_names() {
  let cl = populated_pair();
  let meta = cl.meta_state();
  // Layout: ["2", "KVCache","2","0", "RotatingKVCache","2","4", <4 rot meta>]
  assert_eq!(meta[0], "2", "child count");
  assert_eq!(meta[1], "KVCache", "child 0 reference class name");
  assert_eq!(meta[2], "2", "child 0 state array count");
  assert_eq!(meta[3], "0", "child 0 (Standard) has no meta_state");
  assert_eq!(meta[4], "RotatingKVCache", "child 1 reference class name");
  assert_eq!(meta[5], "2", "child 1 state array count");
  assert_eq!(meta[6], "4", "child 1 (Rotating) meta_state has 4 values");
  // The 4 Rotating meta values (keep, max_size, offset, idx) follow.
  assert_eq!(&meta[7..11], &["4", "8", "5", "5"]);

  assert_eq!(
    CacheList::new(Vec::new()).meta_state(),
    vec!["0".to_string()],
    "empty list -> just child count 0"
  );
}

/// Full `state -> from_state("CacheList", ...)` round-trip rebuilds
/// equivalent concrete children via the crate's `from_state(className, ...)`
/// (cache.py:894-900 `globals()[c].from_state(s, m)`; Swift `fromState`
/// KVCache.swift:1335-1369). The rebuilt list must observably match.
#[test]
fn from_state_roundtrip_rebuilds_children() {
  let cl = populated_pair();
  let st = cl.state().unwrap();
  let meta = cl.meta_state();

  let rebuilt = from_state("CacheList", st, &meta).unwrap();
  // The composite offset (max child offset) is preserved.
  assert_eq!(rebuilt.offset(), 5);
  assert!(!rebuilt.is_empty());

  // Re-serializing the rebuilt list yields byte-identical meta_state and an
  // equal-length state (children reconstructed to the right concrete types).
  assert_eq!(rebuilt.meta_state(), cl.meta_state());
  let rst = rebuilt.state().unwrap();
  assert_eq!(rst.len(), 4);
  let mut k0 = rst[0].try_clone().unwrap();
  assert_eq!(k0.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);

  // The rebuilt list is itself a CacheList whose children are indexable and
  // of the right kind (child 1 carries the Rotating window).
  let again = from_state("CacheList", rebuilt.state().unwrap(), &rebuilt.meta_state()).unwrap();
  assert_eq!(again.offset(), 5);
}

/// Regression (adversarial review, [high]): a `RotatingKvCache` whose
/// `keep == 1` has `meta_state() == ["1", max_size, "0", "0"]`, which is
/// numerically a well-formed `childCount=1` CacheList frame. With the
/// `reference_class_name` trait method, `RotatingKvCache` directly
/// returns `"RotatingKVCache"` so this misclassification is structurally
/// impossible (no heuristic to mis-fire). The test pins both the trait
/// dispatch AND the prompt-cache round-trip (a misname would make
/// `from_state` recurse into the rotating integers and fail on an
/// unknown child kind).
#[test]
fn keep_one_rotating_child_is_not_misidentified_as_cache_list() {
  // Fresh keep=1 rotating child -> meta_state ["1","8","0","0"] (the
  // reviewer's exact counterexample shape).
  let r = RotatingKvCache::new(8, 1);
  assert_eq!(
    r.meta_state(),
    vec![
      "1".to_string(),
      "8".to_string(),
      "0".to_string(),
      "0".to_string()
    ],
    "precondition: keep=1 rotating meta is the ambiguous numeric shape"
  );
  let cl = CacheList::new(vec![Box::new(r)]);

  let meta = cl.meta_state();
  // Layout must be ["1", "RotatingKVCache", "<stateCount>", "4", <4 rot meta>],
  // NOT ["1", "CacheList", ...].
  assert_eq!(meta[0], "1", "one child");
  assert_eq!(
    meta[1], "RotatingKVCache",
    "the keep=1 rotating child must be named RotatingKVCache, NOT CacheList"
  );
  assert_eq!(meta[2], "0", "fresh rotating child has 0 state arrays");
  assert_eq!(meta[3], "4", "rotating meta_state has 4 values");
  assert_eq!(&meta[4..8], &["1", "8", "0", "0"]);

  // And it must round-trip (would fail if misnamed CacheList — `from_state`
  // would recurse into the rotating integers and reject "8" as a child
  // kind). `from_state` yields `Box<dyn KvCache>`; assert via trait methods
  // (`get` is a `CacheList`-inherent method, not on the trait) — a
  // byte-identical re-`meta_state()` proves the child was rebuilt as the
  // right concrete kind (a misname would change the framing).
  let rebuilt = from_state("CacheList", cl.state().unwrap(), &cl.meta_state()).unwrap();
  assert_eq!(
    rebuilt.meta_state(),
    cl.meta_state(),
    "round-trip meta must be byte-identical (child rebuilt as RotatingKVCache)"
  );
  assert_eq!(rebuilt.offset(), 0);
  assert!(
    rebuilt.is_empty(),
    "fresh rotating child -> composite empty"
  );

  // Also exercise a populated keep=1 rotating child (non-zero offset/idx,
  // still all-numeric meta) to be thorough.
  let mut r2 = RotatingKvCache::new(6, 1);
  for i in 0..3 {
    let t = kv(&[i as f32]);
    r2.update(&t, &t).unwrap();
  }
  let cl2 = CacheList::new(vec![Box::new(r2)]);
  assert_eq!(
    cl2.meta_state()[1],
    "RotatingKVCache",
    "populated keep=1 rotating child must also be named RotatingKVCache"
  );
  let rb2 = from_state("CacheList", cl2.state().unwrap(), &cl2.meta_state()).unwrap();
  assert_eq!(rb2.meta_state(), cl2.meta_state());
  // Composite offset == max child offset == the keep=1 rotating child's 3.
  assert_eq!(rb2.offset(), 3);
}

/// `set_state(state)` splits the flattened arrays back per child by each
/// child's current `state().len()` and assigns them (Swift
/// KVCache.swift:1276-1285). A round-trip through `state()`/`set_state()`
/// on a fresh structurally-identical list reproduces the children.
#[test]
fn set_state_splits_per_child() {
  let src = populated_pair();
  let st = src.state().unwrap();

  // A structurally identical (same child types/shapes) but differently-fed
  // target; `set_state` overwrites its children's state from the flat list.
  let mut s = StandardKvCache::new();
  s.update(&kv(&[9.0, 9.0, 9.0]), &kv(&[9.0, 9.0, 9.0]))
    .unwrap();
  let mut r = RotatingKvCache::new(8, 4);
  for _ in 0..5 {
    let t = kv(&[9.0]);
    r.update(&t, &t).unwrap();
  }
  let mut tgt = CacheList::new(vec![Box::new(s), Box::new(r)]);

  tgt.set_state(st).unwrap();
  let back = tgt.state().unwrap();
  assert_eq!(back.len(), 4);
  let mut k0 = back[0].try_clone().unwrap();
  assert_eq!(
    k0.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0],
    "child 0 state replaced from the flat list"
  );
}

/// Regression (adversarial review, [high]): `CacheList::set_state` must be
/// **transactional** — if a *later* child rejects its chunk, every
/// *earlier* child must remain exactly as it was (no half-applied
/// old/new mix that corrupts generation state). Two `StandardKvCache`
/// children (2 state arrays each); the flat state is valid for child 0 but
/// child 1's "keys" array is rank-2 (invalid) so child 1's `set_state`
/// errors. The whole `set_state` must return `Err` AND leave child 0
/// untouched.
#[test]
fn set_state_is_transactional_on_later_child_failure() {
  let mut s0 = StandardKvCache::new();
  s0.update(&kv(&[1.0, 2.0]), &kv(&[1.0, 2.0])).unwrap();
  let mut s1 = StandardKvCache::new();
  s1.update(&kv(&[3.0, 4.0]), &kv(&[3.0, 4.0])).unwrap();
  let mut cl = CacheList::new(vec![Box::new(s0), Box::new(s1)]);

  // Snapshot child 0's pre-call state (ids 1,2) and offset.
  let before_off0 = cl.get(0).unwrap().offset();
  let before_k0 = cl.state().unwrap()[0].to_vec::<f32>().unwrap();
  assert_eq!(before_off0, 2);
  assert_eq!(before_k0, vec![1.0, 2.0]);

  // Flat state: child 0 gets two valid 4-D arrays (ids 7,8,9); child 1
  // gets a *rank-2* "keys" array -> StandardKvCache::set_state rejects it.
  let good_k = kv(&[7.0, 8.0, 9.0]);
  let good_v = kv(&[7.0, 8.0, 9.0]);
  let bad_k = Array::from_slice::<f32>(&[5.0, 6.0], &(1usize, 2)).unwrap();
  let ok_v = kv(&[5.0, 6.0]);
  let flat = vec![good_k, good_v, bad_k, ok_v];

  let r = cl.set_state(flat);
  assert!(
    r.is_err(),
    "a later child rejecting its chunk must make set_state Err"
  );

  // Child 0 MUST be unchanged (still ids 1,2, offset 2) — NOT the 7,8,9
  // that a non-transactional impl would have already written.
  assert_eq!(
    cl.get(0).unwrap().offset(),
    before_off0,
    "child 0 offset must be unchanged after the failed restore"
  );
  let after_k0 = cl.state().unwrap()[0].to_vec::<f32>().unwrap();
  assert_eq!(
    after_k0, before_k0,
    "child 0 state must be byte-identical after the failed restore \
     (no partial mutation / half-applied old-new mix)"
  );
}

/// `copy()` deep-copies each child (cache.py uses `copy.deepcopy`; Swift
/// `caches.map { $0.copy() }`, KVCache.swift:1287-1291). The copy evolves
/// independently of the original.
#[test]
fn copy_is_independent() {
  let mut cl = populated_pair();
  let cp = cl.copy().unwrap();
  assert_eq!(cp.offset(), 5);

  // Mutating the original (trim) must not change the copy.
  cl.trim(5).unwrap();
  assert_eq!(cl.get(0).unwrap().offset(), 0);
  assert_eq!(
    cp.offset(),
    5,
    "the deep copy is unaffected by trimming the original"
  );
}

/// `nbytes()` is the SUM of children's `nbytes` (cache.py:891-892); empty
/// list -> 0. `is_empty()` mirrors `self.caches[0].empty()`
/// (cache.py:887-888).
#[test]
fn nbytes_sum_and_is_empty_is_first_child() {
  let cl = populated_pair();
  let s_only_k = kv(&[0.0, 1.0, 2.0]); // f32 [1,1,3,1] = 12 bytes; k+v per child
  // Standard: k(12)+v(12)=24. Rotating buffer >= 5 rows; exact size depends
  // on the over-allocated ring, so just assert it is the children's sum and
  // strictly greater than the Standard-only contribution.
  let total = cl.nbytes();
  assert!(total >= 24, "at least the Standard child's k+v bytes");
  assert!(
    total > 2 * s_only_k.size() * 4,
    "includes the Rotating child too"
  );

  assert_eq!(CacheList::new(Vec::new()).nbytes(), 0);

  // is_empty == first child empty. Fresh children -> empty.
  let empty_children = CacheList::new(vec![
    Box::new(StandardKvCache::new()),
    Box::new(RotatingKvCache::new(4, 2)),
  ]);
  assert!(empty_children.is_empty(), "first child (fresh) is empty");
  // After feeding child 0 it is no longer empty.
  let mut s = StandardKvCache::new();
  s.update(&kv(&[1.0]), &kv(&[1.0])).unwrap();
  let non_empty = CacheList::new(vec![Box::new(s), Box::new(RotatingKvCache::new(4, 2))]);
  assert!(!non_empty.is_empty());

  // An empty list: Python indexes `self.caches[0]` (IndexError); our
  // recoverable choice is "empty".
  assert!(CacheList::new(Vec::new()).is_empty());
}

/// `update()` on the composite is meaningless — children are accessed via
/// `get(i)` (Swift `update` is `fatalError("use subscript access instead")`,
/// KVCache.swift:1270-1272; Python `CacheList` defines no `update`). The
/// merged trait requires `update`, so this returns a recoverable `Err`
/// (the no-panic equivalent of Swift's trap), never a panic.
#[test]
fn update_is_unsupported_error_not_panic() {
  let mut cl = populated_pair();
  let t = kv(&[7.0]);
  assert!(
    cl.update(&t, &t).is_err(),
    "CacheList.update must be a recoverable Err, not a panic"
  );
}

/// `make_mask()` on the composite is likewise meaningless: neither
/// `_BaseCache` nor `CacheList` defines `make_mask` in mlx-lm
/// (cache.py:127-175, 814-902) — masking is per child via `get(i)`. The
/// merged trait requires `make_mask`, so this is a recoverable `Err`, not a
/// panic.
#[test]
fn make_mask_is_unsupported_error_not_panic() {
  let cl = populated_pair();
  assert!(
    cl.make_mask(1, None, false).is_err(),
    "CacheList.make_mask must be a recoverable Err (no _BaseCache mask)"
  );
}

/// `from_state` rejects malformed `CacheList` meta (truncated framing,
/// non-numeric counts, child-count mismatch) with a recoverable `Error`,
/// never a panic / out-of-bounds index (Swift `fromState` throws
/// `KVCacheError`, KVCache.swift:1336-1361).
#[test]
fn from_state_rejects_malformed_meta() {
  // Empty meta (missing child count).
  assert!(from_state("CacheList", Vec::new(), &[]).is_err());

  // Non-numeric child count.
  assert!(from_state("CacheList", Vec::new(), &["x".to_string()]).is_err());

  // Claims 1 child but the per-child framing is truncated.
  assert!(
    from_state("CacheList", Vec::new(), &["1".to_string()]).is_err(),
    "truncated per-child framing must error, not panic"
  );

  // Claims 1 child, names it, but stateCount is non-numeric.
  let bad = vec![
    "1".to_string(),
    "KVCache".to_string(),
    "two".to_string(),
    "0".to_string(),
  ];
  assert!(from_state("CacheList", Vec::new(), &bad).is_err());

  // Well-framed but the declared stateCount exceeds the arrays provided
  // (Swift clamps with `min(...)`; we reject the inconsistency rather than
  // silently truncate -> recoverable Err, no slice panic).
  let claims_two_arrays = vec![
    "1".to_string(),
    "KVCache".to_string(),
    "2".to_string(), // says 2 state arrays
    "0".to_string(),
  ];
  assert!(
    from_state("CacheList", Vec::new(), &claims_two_arrays).is_err(),
    "declared stateCount > provided arrays must error, not panic"
  );
}

/// Regression (adversarial review, [high]): a corrupt/forged prompt cache
/// whose leading `child_count` is a huge number (here `usize::MAX`) with
/// truncated metadata must be a recoverable `Error`, NOT a
/// `Vec::with_capacity` capacity-overflow panic / OOM abort. `child_count`
/// is bounded against the metadata length (>= 3 framing fields per child)
/// *before* any allocation, so this is rejected, not reserved.
#[test]
fn from_state_huge_child_count_is_err_not_panic_or_oom() {
  let huge = vec![usize::MAX.to_string()]; // count = usize::MAX, no frames
  let r = from_state("CacheList", Vec::new(), &huge);
  assert!(
    r.is_err(),
    "an absurd child_count must be a recoverable Err, never a capacity \
     panic / OOM abort on the public from_state load path"
  );

  // Also a large-but-not-MAX count with some (insufficient) framing: still
  // rejected before allocating that many slots.
  let big = vec![
    "1000000000".to_string(),
    "KVCache".to_string(),
    "0".to_string(),
    "0".to_string(),
  ];
  assert!(
    from_state("CacheList", Vec::new(), &big).is_err(),
    "child_count far exceeding the frame budget must error pre-allocation"
  );

  // Boundary: exactly the max possible children for the given meta length
  // must NOT be rejected by the pre-allocation bound (it proceeds to the
  // normal per-child validation). meta.len()=4 -> max = (4-1)/3 = 1.
  // A well-formed single empty-Standard child round-trips fine.
  let exactly_one = vec![
    "1".to_string(),
    "KVCache".to_string(),
    "0".to_string(), // 0 state arrays
    "0".to_string(), // 0 meta values
  ];
  let ok = from_state("CacheList", Vec::new(), &exactly_one);
  assert!(
    ok.is_ok(),
    "child_count == the frame-budget max must still construct, not be \
     spuriously rejected by the pre-allocation bound"
  );
  assert_eq!(ok.unwrap().offset(), 0);
}

/// Regression (adversarial review, [high]): a forged prompt cache can
/// encode an arbitrarily deep single-child `CacheList -> CacheList -> … ->
/// []` chain using **only metadata strings and zero state arrays** — every
/// level is a well-formed `childCount=1, stateCount=0` frame, so the
/// `child_count`/`stateCount` allocation+length guards never reject it.
/// Unbounded native recursion on that chain is a stack-overflow **process
/// abort** (not a recoverable `Error`) on the public `from_state` load
/// path. `cache_list_from_state` must carry a nesting-depth budget and
/// reject an over-deep chain as a recoverable `Error`, never aborting; a
/// *legitimately* nested (shallow) `CacheList` must still round-trip.
#[test]
fn from_state_deeply_nested_chain_is_err_not_stack_overflow() {
  // Build the flattened meta for an N-deep single-child chain with NO
  // arrays: each level is `["1","CacheList","0", <inner_len>, ...inner]`,
  // the innermost being `["0"]` (empty CacheList). Depth chosen far beyond
  // the implementation's ceiling so the over-deep guard must fire (and a
  // forged file could pick any depth — the loader must not abort for ANY).
  // Built **iteratively** (inner -> outer) so the *test helper* itself
  // uses no recursion: the only recursion under test is the deserializer's
  // (which is exactly what the depth bound must keep from aborting).
  fn nest(depth: usize) -> Vec<String> {
    let mut m = vec!["0".to_string()]; // innermost: empty CacheList
    for _ in 0..depth {
      let inner_len = m.len().to_string();
      let mut next = vec![
        "1".to_string(),         // one child
        "CacheList".to_string(), // the child is itself a CacheList
        "0".to_string(),         // child has 0 state arrays
        inner_len,               // child's metaCount
      ];
      next.append(&mut m);
      m = next;
    }
    m
  }

  // 5000 levels deep — a forged chain that, recursed unbounded, would blow
  // the test thread's stack. It must instead be a recoverable Err.
  let deep = nest(5000);
  let r = from_state("CacheList", Vec::new(), &deep);
  assert!(
    r.is_err(),
    "a pathologically deep nested-CacheList chain must be a recoverable \
     Err, never a stack-overflow process abort on the from_state load path"
  );

  // A legitimately shallow nested CacheList (a few levels) must STILL
  // round-trip — the depth bound must not reject any realistic nesting.
  let shallow = nest(3);
  let ok = from_state("CacheList", Vec::new(), &shallow);
  assert!(
    ok.is_ok(),
    "a shallow (depth-3) nested CacheList must still reconstruct — the \
     nesting bound must not reject realistic nesting"
  );
  // The reconstructed shallow chain is an empty composite (no children
  // carry state); offset 0, reported empty.
  let ok = ok.unwrap();
  assert_eq!(ok.offset(), 0);
  assert!(ok.is_empty());
  // It re-serializes to the same framing it was built from (byte-identical
  // round-trip of the legitimate nested structure).
  assert_eq!(
    ok.meta_state(),
    shallow,
    "shallow nested round-trip is exact"
  );
}

/// Nested `CacheList` inside a `CacheList` round-trips: a child whose
/// reference class name is `"CacheList"` is itself rebuilt recursively via
/// the same `from_state` arm (cache.py:898 `globals()["CacheList"]`).
#[test]
fn nested_cache_list_roundtrips() {
  let inner = populated_pair();
  let mut s = StandardKvCache::new();
  s.update(&kv(&[8.0, 8.0]), &kv(&[8.0, 8.0])).unwrap();
  let outer = CacheList::new(vec![Box::new(s), Box::new(inner)]);

  assert_eq!(outer.len(), 2);
  let meta = outer.meta_state();
  assert_eq!(meta[0], "2");
  assert_eq!(meta[1], "KVCache");
  // child 1 is itself a CacheList.
  // (its className appears after child 0's framing: idx 4)
  assert_eq!(meta[4], "CacheList");

  let rebuilt = from_state("CacheList", outer.state().unwrap(), &outer.meta_state()).unwrap();
  assert_eq!(rebuilt.meta_state(), outer.meta_state());
  // outer offset = max(child0=2, inner_offset=5) = 5.
  assert_eq!(rebuilt.offset(), 5);
}

/// `as_cache_list` / `as_cache_list_mut` are the trait-level downcast hooks
/// that let a hybrid model holding `Box<dyn KvCache>` per layer reach the
/// `CacheList`-inherent indexing API (faithful to swift's
/// `cache as? CacheList`). `CacheList` overrides them to `Some(self)`;
/// every other concrete cache inherits the defaulted `None`.
#[test]
fn as_cache_list_downcast_through_dyn() {
  // Build a `CacheList` and hold it behind `Box<dyn KvCache>`; the downcast
  // must succeed and give the indexing API back.
  let cl = CacheList::new(vec![
    Box::new(StandardKvCache::new()),
    Box::new(RotatingKvCache::new(8, 4)),
  ]);
  let mut b: Box<dyn KvCache> = Box::new(cl);
  assert!(
    b.as_cache_list().is_some(),
    "Box<dyn KvCache> wrapping a CacheList must downcast via as_cache_list"
  );
  // The `&` downcast yields the indexing API.
  let view = b.as_cache_list().unwrap();
  assert_eq!(view.len(), 2);
  assert_eq!(view.get(0).unwrap().offset(), 0);

  // The `&mut` downcast lets the generation loop reach a child's mutating
  // API (`get_mut(i).update(...)` — exactly the swift `cache as? CacheList`
  // path through `Box<dyn KvCache>`).
  {
    let view_mut = b.as_cache_list_mut().unwrap();
    let child0 = view_mut.get_mut(0).unwrap();
    let k = Array::from_slice::<f32>(&[5.0], &(1usize, 1, 1, 1)).unwrap();
    child0.update(&k, &k).unwrap();
  }
  assert_eq!(
    b.as_cache_list().unwrap().get(0).unwrap().offset(),
    1,
    "the through-dyn `&mut` downcast actually mutated the child"
  );

  // A non-CacheList cache MUST inherit the defaulted `None` (the downcast
  // must NOT spuriously succeed for an unrelated kind).
  let plain: Box<dyn KvCache> = Box::new(StandardKvCache::new());
  assert!(
    plain.as_cache_list().is_none(),
    "a non-CacheList cache must inherit the defaulted None downcast"
  );
  let plain_rot: Box<dyn KvCache> = Box::new(RotatingKvCache::new(8, 4));
  assert!(plain_rot.as_cache_list().is_none());
}

/// Regression (Codex adversarial review): a `CacheList` child that is a
/// `ChunkedKvCache` must be named `"ChunkedKVCache"` in `meta_state` (not
/// the previous silent fallback `"KVCache"`), so a round-trip rebuilds the
/// right concrete kind via the crate `from_state` dispatch. A misname
/// would drop `chunk_size`/`start_position` (the Standard arm ignores
/// meta) and corrupt mask/trim/update semantics after reload.
#[test]
fn cache_list_chunked_child_class_name_and_roundtrip() {
  // Populated Chunked child (its `set_state` requires exactly 2 arrays —
  // its setter unpacks `keys, values = v`, cache.py:782-787 — so the
  // round-trip needs a fed child; an empty one is not a `from_state`-valid
  // shape). `chunk_size = 64` survives the round-trip iff the child is
  // named correctly.
  let mut chunk = ChunkedKvCache::new(Some(64));
  chunk.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  let chunk_meta_before = chunk.meta_state();
  assert_eq!(
    chunk_meta_before.len(),
    2,
    "precondition: Chunked meta_state has the 2-value shape"
  );

  let cl = CacheList::new(vec![Box::new(chunk)]);
  let meta = cl.meta_state();
  // Layout: ["1","ChunkedKVCache","2","2", <2 chunked meta>]
  assert_eq!(meta[0], "1", "one child");
  assert_eq!(
    meta[1], "ChunkedKVCache",
    "Chunked child must be named ChunkedKVCache, NOT KVCache (silent \
     fallback would drop chunk_size/start_position on reload)"
  );
  assert_eq!(meta[2], "2", "populated Chunked child has 2 state arrays");
  assert_eq!(meta[3], "2", "Chunked meta_state has 2 values");
  assert_eq!(&meta[4..6], chunk_meta_before.as_slice());

  // Round-trip: rebuilt child must preserve `chunk_size` (re-serializes
  // identically); a misname would reach the Standard arm whose default
  // `set_meta_state` is a no-op, so the rebuilt re-`meta_state` would not
  // carry the Chunked values.
  let rebuilt = from_state("CacheList", cl.state().unwrap(), &cl.meta_state()).unwrap();
  assert_eq!(
    rebuilt.meta_state(),
    cl.meta_state(),
    "round-trip meta must be byte-identical (Chunked child rebuilt as the \
     right concrete kind, preserving chunk_size/start_position)"
  );
}

/// `state_count()` must equal `state().len()` on a populated `CacheList`
/// (the contract `meta_state`'s framing relies on — using `state_count`
/// instead of cloning every child's full state just to read its length).
#[test]
fn cache_list_state_count_matches_state_len() {
  let cl = populated_pair();
  assert_eq!(
    cl.state_count().unwrap(),
    cl.state().unwrap().len(),
    "CacheList::state_count must equal CacheList::state().len()"
  );

  // Empty list also matches (both are 0).
  let empty = CacheList::new(Vec::new());
  assert_eq!(empty.state_count().unwrap(), empty.state().unwrap().len());
  assert_eq!(empty.state_count().unwrap(), 0);

  // Through `Box<dyn KvCache>` the trait method dispatches to the override
  // (not the default), so the contract is preserved across the dyn boundary.
  let b: Box<dyn KvCache> = Box::new(populated_pair());
  assert_eq!(b.state_count().unwrap(), b.state().unwrap().len());
}

/// Regression (#83 — CacheList::trim transactional, Codex round-2 finding):
/// when any child is non-trimmable, `trim(n)` must short-circuit `Ok(0)`
/// BEFORE mutating any child — matching `cache.py:88-111`'s
/// `can_trim_prompt_cache`/`trim_prompt_cache` `all(is_trimmable())` gate.
/// Otherwise an earlier-trimmable child would be partially trimmed while a
/// later non-trimmable child silently returns 0 — leaving the composite
/// out of sync. (Faithful semantic; just brings the per-call atomicity
/// the references' deliberately-sequential design left as the caller's
/// responsibility into mlxrs's per-trim `Result`-returning API.)
#[test]
fn cache_list_trim_transactional_short_circuits_on_non_trimmable_child() {
  // `Standard::is_trimmable() == true` always; `Rotating::is_trimmable()`
  // is `offset < max_size` (mlx-lm `RotatingKVCache.is_trimmable` — false
  // once the ring fills). To exercise the non-trimmable short-circuit: a
  // trimmable populated Standard + a filled Rotating. Without the
  // transactional gate, the Standard child would be trimmed first
  // (partial mutation) and the loop would then mutate the Rotating too
  // (mlx-lm's RotatingKVCache.trim doesn't check is_trimmable internally
  // — the guard is the per-call CacheList gate). The fix enforces that
  // gate atomically: any non-trimmable child aborts the whole list-trim
  // with Ok(0) BEFORE mutating anyone.
  let mut populated_std = StandardKvCache::new();
  populated_std
    .update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let std_offset_before = populated_std.offset();
  let mut filled_rot = RotatingKvCache::new(4, 2);
  for i in 0..6 {
    let t = kv(&[i as f32]);
    filled_rot.update(&t, &t).unwrap();
  }
  assert!(
    !filled_rot.is_trimmable(),
    "sanity: filled Rotating is not trimmable"
  );
  let rot_offset_before = filled_rot.offset();
  let mut cl = CacheList::new(vec![Box::new(populated_std), Box::new(filled_rot)]);
  // CacheList::is_trimmable = all → false (Rotating is full).
  assert!(
    !cl.is_trimmable(),
    "sanity: filled-Rotating child makes the list non-trimmable"
  );
  // trim(2) must short-circuit Ok(0) — NOT trim either child.
  let r = cl.trim(2).unwrap();
  assert_eq!(r, 0, "trim must short-circuit Ok(0) on non-trimmable list");
  // Critical: the trimmable Standard child must NOT have been mutated —
  // the atomicity check happened BEFORE any per-child trim ran. (Without
  // the fix, std_offset would have dropped to std_offset_before - 2 = 1.)
  assert_eq!(
    cl.get(0).unwrap().offset(),
    std_offset_before,
    "TRANSACTIONAL: trimmable Standard child must NOT mutate when sibling is non-trimmable"
  );
  assert_eq!(
    cl.get(1).unwrap().offset(),
    rot_offset_before,
    "filled Rotating child also unchanged"
  );

  // Sanity: a list where ALL children ARE trimmable still trims normally.
  let mut all_trim = CacheList::new(vec![
    Box::new({
      let mut s = StandardKvCache::new();
      s.update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
        .unwrap();
      s
    }),
    Box::new({
      let mut r = RotatingKvCache::new(8, 4);
      for i in 0..5 {
        let t = kv(&[i as f32]);
        r.update(&t, &t).unwrap();
      }
      r
    }),
  ]);
  assert!(all_trim.is_trimmable(), "sanity: both children trimmable");
  let r2 = all_trim.trim(2).unwrap();
  assert!(
    r2 > 0,
    "all-trimmable list: trim must actually trim (returned {r2})"
  );
}

#[test]
fn state_into_buffer_reuse_matches_state_for_cache_list() {
  // KVC-7 (#104): `state_into(&mut buf)` is the buffer-reuse companion to
  // `state() -> Vec<Array>`. For a `CacheList` (which flattens every
  // child's state), `state_into` lets a parent composite avoid the per-
  // child `Vec<Array>` allocation the default trait method would pay.
  // The OBSERVABLE output (count of arrays appended) must match `state()`
  // byte-for-byte — the optimization is alloc-only, not behavior.
  let cl = populated_pair();
  let s = cl.state().unwrap();
  let mut buf: Vec<Array> = Vec::new();
  cl.state_into(&mut buf).unwrap();
  assert_eq!(
    s.len(),
    buf.len(),
    "state_into and state must append the same number of arrays"
  );
  // Append semantics: a non-empty buf is APPENDED to, NOT cleared. Reuse
  // the same buffer for a second call and verify total len doubles.
  cl.state_into(&mut buf).unwrap();
  assert_eq!(
    buf.len(),
    s.len() * 2,
    "state_into must APPEND, not clear (multi-cache callers depend on this)"
  );
}

#[test]
fn meta_state_into_buffer_reuse_matches_meta_state_for_cache_list() {
  // KVC-6 (#103): `meta_state_into(&mut buf)` is the buffer-reuse
  // companion to `meta_state() -> Vec<String>`. For a `CacheList`
  // (whose framing is O(children) — `[childCount, (className,
  // stateCount, metaCount, ...meta)*]`) this saves one Vec<String>
  // allocation per child compared to the per-child meta_state() +
  // extend pattern. Output must be byte-identical (the metaCount slot
  // is patched in place after the child appends).
  let cl = populated_pair();
  let m1 = cl.meta_state();
  let mut buf: Vec<String> = Vec::new();
  cl.meta_state_into(&mut buf);
  assert_eq!(
    m1, buf,
    "meta_state and meta_state_into must produce byte-identical output"
  );
  // Append semantics — second call must append, not clear.
  cl.meta_state_into(&mut buf);
  assert_eq!(
    buf.len(),
    m1.len() * 2,
    "meta_state_into must APPEND, not clear"
  );
}

#[test]
fn meta_state_into_default_delegates_to_meta_state_for_standard_cache() {
  // KVC-6 (#103): the trait DEFAULT `meta_state_into` delegates to
  // `meta_state()` and appends. A concrete cache that does NOT override
  // (e.g. `StandardKvCache` — its meta_state is the trait default empty
  // Vec, so meta_state_into appends nothing) must still produce the same
  // observable output.
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  let m = s.meta_state();
  let mut buf: Vec<String> = Vec::new();
  s.meta_state_into(&mut buf);
  assert_eq!(
    m, buf,
    "default meta_state_into must produce identical output to meta_state"
  );
  // For StandardKvCache the meta is empty (no override) — both forms
  // emit nothing.
  assert!(buf.is_empty(), "StandardKvCache has no meta_state");
}

#[test]
fn state_into_default_delegates_to_state_for_standard_cache() {
  // KVC-7 (#104): the trait DEFAULT `state_into` delegates to `state()`
  // and appends. A concrete cache that does NOT override (every cache
  // EXCEPT `CacheList`) must still produce the same observable output.
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let st = s.state().unwrap();
  let mut buf: Vec<Array> = Vec::new();
  s.state_into(&mut buf).unwrap();
  assert_eq!(
    st.len(),
    buf.len(),
    "default state_into must produce identical count to state"
  );
  // A populated StandardKvCache has (keys, values) — 2 arrays.
  assert_eq!(buf.len(), 2, "populated StandardKvCache state has 2 arrays");
}
