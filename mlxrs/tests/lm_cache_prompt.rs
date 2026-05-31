//! Deterministic tests for the prompt-reuse + on-disk persistence surface
//! (`mlxrs::lm::cache::{prompt, persist}`), ported 1:1 from
//! `mlx_lm.models.cache` (`TokenBuffer` :1487-1521, `PromptTrieResult`
//! :1523-1530, `PromptTrie` :1532-1620, `LRUPromptCache` :1623-1763, and the
//! persistence helpers :43-113) and cross-checked against mlx-swift-lm's
//! `MLXLMCommon/KVCache.swift` (`savePromptCache`/`loadPromptCache`,
//! `cacheClassName`, the `"i.j"` / `"0.i.j"` / `"1.key"` / `"2.i"`
//! `tree_flatten` wire format).
//!
//! These tests are hand-traced from the cited Python lines so each retained
//! identity / eviction order / on-disk key is checkable, not assumed.

#![cfg(feature = "lm")]

use std::{collections::HashMap, fs, path::PathBuf, process};

use mlxrs::{
  Array, io,
  lm::cache::{
    KvCache, LruPromptCache, PromptTrie, RotatingKvCache, StandardKvCache, TokenBuffer,
    can_trim_prompt_cache, load_prompt_cache, save_prompt_cache, trim_prompt_cache,
  },
};

fn temp_path(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_lm_cache_prompt_{}_{}", process::id(), name));
  p
}

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// each step's value being its 0-based token id (so retained ids / round-trip
/// values are directly readable). `S == ids.len()`.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

// ───────────────────────── TokenBuffer ─────────────────────────

#[test]
fn token_buffer_push_and_slice() {
  // mlx-lm `TokenBuffer.__init__([])` then `update_and_fetch` (cache.py
  // :1496-1512). `step == 256`, so a small push over-allocates to 256 but
  // `.tokens` is the logical `[:_size]` slice and the fetch is `[:end]`.
  let mut tb = TokenBuffer::new(&[]);
  assert_eq!(tb.tokens(), &[] as &[i32]);

  assert_eq!(tb.update_and_fetch(&[1, 2, 3]), &[1, 2, 3]);
  assert_eq!(tb.tokens(), &[1, 2, 3]);

  // Appending extends from the prior size (start = 3).
  assert_eq!(tb.update_and_fetch(&[4, 5]), &[1, 2, 3, 4, 5]);
  assert_eq!(tb.tokens(), &[1, 2, 3, 4, 5]);

  // Construct seeded (mlx-lm `TokenBuffer(tokens)`).
  let mut tb2 = TokenBuffer::new(&[7, 8]);
  assert_eq!(tb2.tokens(), &[7, 8]);
  assert_eq!(tb2.update_and_fetch(&[9]), &[7, 8, 9]);

  // Crossing the 256 step boundary still yields the exact logical prefix
  // (the over-allocated tail zeros are never observable — `tokens`/fetch
  // slice them off, mlx-lm cache.py:1512/1520).
  let mut big: Vec<i32> = (0..300).collect();
  let mut tb3 = TokenBuffer::new(&[]);
  assert_eq!(tb3.update_and_fetch(&big), big.as_slice());
  assert_eq!(tb3.len(), 300);
  big.push(300);
  assert_eq!(tb3.update_and_fetch(&[300]), big.as_slice());
  assert_eq!(tb3.tokens(), big.as_slice());
}

// ───────────────────────── PromptTrie ─────────────────────────

#[test]
fn prompt_trie_search_exact_shorter_longer() {
  // Hand-traced from mlx-lm `PromptTrie.search` (cache.py:1578-1620).
  let mut trie: PromptTrie<&'static str, i32> = PromptTrie::new();

  // Empty trie / unknown model -> all-None, common_prefix 0 (cache.py
  // :1579-1580).
  let r = trie.search(&"m", &[1, 2, 3]);
  assert_eq!(r.exact, None);
  assert_eq!(r.shorter, None);
  assert_eq!(r.longer, None);
  assert_eq!(r.common_prefix, 0);

  // add returns the previous value at that path (None first time).
  assert_eq!(trie.add(&"m", &[1, 2, 3, 4], 10), None);
  assert_eq!(trie.add(&"m", &[1, 2], 20), None);
  // Re-add same path -> previous value returned (cache.py:1545-1547).
  assert_eq!(trie.add(&"m", &[1, 2], 99), Some(20));
  assert_eq!(trie.add(&"m", &[1, 2], 20), Some(99));

  // get walks the path and returns __value__ (cache.py:1549-1553).
  assert_eq!(trie.get(&"m", &[1, 2]).copied(), Some(20));
  assert_eq!(trie.get(&"m", &[1, 2, 3, 4]).copied(), Some(10));

  // Exact match: tokens [1,2] has a value AND last_index == len-1.
  let r = trie.search(&"m", &[1, 2]);
  assert_eq!(r.exact, Some(vec![1, 2]));
  assert_eq!(r.shorter, None);
  assert_eq!(r.longer, None);
  assert_eq!(r.common_prefix, 0);

  // tokens [1,2,3,4] exact too.
  let r = trie.search(&"m", &[1, 2, 3, 4]);
  assert_eq!(r.exact, Some(vec![1, 2, 3, 4]));

  // tokens [1,2,3]: walk 1->2(value@idx1)->3 (no value). last_index==1,
  // index==3. Not exact (last_index 1 != 2). shorter = tokens[:2] = [1,2]
  // (last_index>0). longer: index>0, DFS from node after 3 finds the
  // shortest value-bearing extension [4] -> tokens[:3] + [4] = [1,2,3,4].
  // common_prefix == index == 3.
  let r = trie.search(&"m", &[1, 2, 3]);
  assert_eq!(r.exact, None);
  assert_eq!(r.shorter, Some(vec![1, 2]));
  assert_eq!(r.longer, Some(vec![1, 2, 3, 4]));
  assert_eq!(r.common_prefix, 3);

  // tokens [1,2,3,4,5]: walk to [1,2,3,4] (value@idx3), token 5 not in
  // trie -> stop. index==4, last_index==3. Not exact (3 != 4). shorter =
  // tokens[:4] = [1,2,3,4] (last_index 3 > 0). longer: index 4 > 0, DFS
  // from node-after-4 which has __value__ -> best == [] -> tokens[:4] + []
  // = [1,2,3,4]. common_prefix == 4.
  let r = trie.search(&"m", &[1, 2, 3, 4, 5]);
  assert_eq!(r.exact, None);
  assert_eq!(r.shorter, Some(vec![1, 2, 3, 4]));
  assert_eq!(r.longer, Some(vec![1, 2, 3, 4]));
  assert_eq!(r.common_prefix, 4);

  // tokens [9]: token 9 not in the trie at all. index==0, last_index==-1.
  // Not exact. shorter None (last_index !> 0). longer None (index !> 0).
  // common_prefix == 0.
  let r = trie.search(&"m", &[9]);
  assert_eq!(r.exact, None);
  assert_eq!(r.shorter, None);
  assert_eq!(r.longer, None);
  assert_eq!(r.common_prefix, 0);
}

#[test]
fn prompt_trie_empty_tokens_and_pop() {
  let mut trie: PromptTrie<u8, i32> = PromptTrie::new();
  assert_eq!(trie.add(&1u8, &[], 5), None);

  // Empty tokens with a __value__ at the root -> exact = [] (cache.py
  // :1584-1585).
  let r = trie.search(&1u8, &[]);
  assert_eq!(r.exact, Some(vec![]));
  assert_eq!(r.common_prefix, 0);

  // get([]) returns the root value.
  assert_eq!(trie.get(&1u8, &[]).copied(), Some(5));
  // pop([]) returns it and (no path nodes to prune).
  assert_eq!(trie.pop(&1u8, &[]), Some(5));

  // pop trims now-empty interior nodes (cache.py:1555-1567).
  let mut t2: PromptTrie<u8, i32> = PromptTrie::new();
  t2.add(&0u8, &[1, 2, 3], 7);
  t2.add(&0u8, &[1, 2], 8);
  // pop the deeper path; the [1,2] node still has a value so it survives.
  assert_eq!(t2.pop(&0u8, &[1, 2, 3]), Some(7));
  assert_eq!(t2.get(&0u8, &[1, 2]).copied(), Some(8));
  // longest path now gone; searching [1,2,3] is no longer exact.
  let r = t2.search(&0u8, &[1, 2, 3]);
  assert_eq!(r.exact, None);
  // Trace: walk 1 -> node[1] (no value) -> 2 -> node[1,2] (value 8 @ index
  // 1, so last_index==1) -> token 3: node[1,2,3] was pruned by the pop ->
  // absent, stop. index==2, last_index==1. Not exact (1 != len-1==2).
  // shorter = tokens[:last_index+1] = tokens[:2] = [1,2] (last_index 1 >
  // 0). longer: index 2 > 0, DFS from node[1,2] — its only value-bearing
  // node is itself (value 8) reached with extra==[] -> longer =
  // tokens[:2] + [] = [1,2]. common_prefix == index == 2.
  assert_eq!(r.shorter, Some(vec![1, 2]));
  assert_eq!(r.longer, Some(vec![1, 2]));
  assert_eq!(r.common_prefix, 2);

  // pop_prefixes pops every shorter value along the path (cache.py
  // :1569-1576).
  let mut t3: PromptTrie<u8, i32> = PromptTrie::new();
  t3.add(&0u8, &[1], 100);
  t3.add(&0u8, &[1, 2], 200);
  t3.add(&0u8, &[1, 2, 3], 300);
  let popped = t3.pop_prefixes(&0u8, &[1, 2, 3]);
  // Values strictly *before* reaching the full token path: at i=1 (after
  // consuming token "1", node [1] has value 100) and i=2 (node [1,2] has
  // value 200). The terminal [1,2,3] value is NOT popped.
  assert_eq!(popped, vec![(1usize, 100), (2usize, 200)]);
  assert_eq!(t3.get(&0u8, &[1, 2, 3]).copied(), Some(300));
  assert_eq!(t3.get(&0u8, &[1]).copied(), None);
  assert_eq!(t3.get(&0u8, &[1, 2]).copied(), None);
}

#[test]
fn prompt_trie_longer_tie_breaks_by_insertion_order() {
  // Regression: equal-length `longer` extensions must
  // resolve EXACTLY as mlx-lm's `search` (cache.py:1610-1620) —
  // Python-dict **insertion order** of children + a **LIFO** stack, with
  // `best` replaced only on a *strictly* shorter `extra`. Trace for
  // siblings inserted [1,2] then [1,3], `search(m, [1])`: walk consumes
  // token 1 -> node[1] (no value), index==1>0. DFS from node[1]; children
  // in insertion order are 2 (from [1,2]) then 3 (from [1,3]); pushed 2,3
  // onto the stack; `pop()` yields the LAST-inserted first: node[1,3]
  // (extra [3]) -> value present -> best=[3]. Next pop node[1,2] (extra
  // [2]); `len 1 < len(best) 1` is false -> best stays [3]. So
  // longer == tokens[:1] + [3] == [1,3] (the last-inserted sibling).
  // A `HashMap` would make this hash-order dependent; the insertion-
  // ordered `ChildMap` makes it byte-for-byte mlx-lm.
  let mut t: PromptTrie<&'static str, i32> = PromptTrie::new();
  t.add(&"m", &[1, 2], 12);
  t.add(&"m", &[1, 3], 13);
  let r = t.search(&"m", &[1]);
  assert_eq!(r.exact, None);
  assert_eq!(r.common_prefix, 1);
  assert_eq!(
    r.longer,
    Some(vec![1, 3]),
    "equal-length longer tie must pick the LAST-inserted sibling ([1,3]), \
     matching mlx-lm dict-insertion-order + LIFO DFS"
  );

  // Reverse the insertion order -> mlx-lm would deterministically pick the
  // other sibling. Locks that the choice tracks *insertion order*, not
  // token value or hashing.
  let mut t2: PromptTrie<&'static str, i32> = PromptTrie::new();
  t2.add(&"m", &[1, 3], 13);
  t2.add(&"m", &[1, 2], 12);
  let r2 = t2.search(&"m", &[1]);
  assert_eq!(
    r2.longer,
    Some(vec![1, 2]),
    "with [1,3] then [1,2] inserted, the last-inserted ([1,2]) wins"
  );

  // Behavioral consequence in `LruPromptCache.fetch_nearest_cache`
  // (cache.py:1674-1694): the `longer` branch reuses the chosen entry.
  // Insert two equal-length siblings; the *last-inserted* one is the
  // `longer` target. `fetch_nearest_cache([1])` -> not exact, shorter
  // None (no value at [1]), longer == [1,3] (last inserted),
  // common_prefix 1 > short_length 0, [1,3] is trimmable -> trim path:
  // prefix = min(len([1])-1, 1) = 0, num_to_trim = len([1,3]) - 0 = 2,
  // returns (Some(cache), tokens[0:]) == rest [1]. The point: which entry
  // is reused is deterministic and insertion-order-driven, exactly mlx-lm.
  let mut lru: LruPromptCache<&'static str> = LruPromptCache::new(10, 1 << 62);
  lru
    .insert_cache(&"m", &[1, 2], trimmable_entry(&[0.0, 1.0]), "assistant")
    .unwrap();
  lru
    .insert_cache(&"m", &[1, 3], trimmable_entry(&[0.0, 1.0]), "assistant")
    .unwrap();
  let (hit, rest) = lru.fetch_nearest_cache(&"m", &[1]).unwrap();
  assert!(hit.is_some());
  assert_eq!(rest, vec![1]);
}

// ───────────────────────── LruPromptCache ─────────────────────────

/// A minimal trimmable cache standing in for a per-layer prompt cache entry
/// (its `nbytes` drives the byte accounting; one `StandardKvCache` fed `n`
/// tokens has `is_trimmable() == true`).
fn trimmable_entry(seq: &[f32]) -> Vec<Box<dyn KvCache>> {
  let mut c = StandardKvCache::new();
  c.update(&kv(seq), &kv(seq)).unwrap();
  vec![Box::new(c)]
}

#[test]
fn lru_insert_get_and_eviction_order() {
  // Hand-traced from mlx-lm `LRUPromptCache` (cache.py:1623-1763). Default
  // ordering ["assistant","user","system"]; `pop()` (cache.py:1649-1657):
  // i=0 compare assistant vs user -> if assistant non-empty and
  // len(assistant) >= len(user) popleft(assistant); else i=1 compare user
  // vs system likewise; else popleft(system).
  let mut lru: LruPromptCache<&'static str> = LruPromptCache::new(3, 1 << 62);
  assert_eq!(lru.len(), 0);

  // Insert one assistant entry; fetch_nearest_cache exact hit returns a
  // deep copy and an empty remaining-token list (cache.py:1676-1678).
  lru
    .insert_cache(
      &"m",
      &[1, 2, 3],
      trimmable_entry(&[0.0, 1.0, 2.0]),
      "assistant",
    )
    .unwrap();
  assert_eq!(lru.len(), 1);
  let (hit, rest) = lru.fetch_nearest_cache(&"m", &[1, 2, 3]).unwrap();
  assert!(hit.is_some());
  assert_eq!(rest, Vec::<i32>::new());
  // The entry is still present (fetch returns a *copy*; it does not consume).
  assert_eq!(lru.len(), 1);

  // A prefix-only request: insert a longer non-trimmable-prefix entry then
  // ask for a shorter prefix. With only [1,2,3] present, asking [1,2]:
  // search -> not exact, shorter? last_index for [1,2]: walk 1->2 (no
  // value at [1,2], value is at [1,2,3]); last_index==-1. longer: index==2
  // > 0, DFS finds [3] -> longer=[1,2,3], common_prefix=2. short_length=0.
  // result.longer not None and common_prefix(2) > short_length(0) and the
  // [1,2,3] entry is trimmable -> trim path: copy, prefix =
  // min(len(tokens)-1, common_prefix) = min(1,2)=1, num_to_trim =
  // len(longer)-prefix = 3-1 = 2, returns (cache, tokens[1:]) -> rest==[2].
  let (hit, rest) = lru.fetch_nearest_cache(&"m", &[1, 2]).unwrap();
  assert!(hit.is_some());
  assert_eq!(rest, vec![2]);

  // No match at all -> (None, tokens) (cache.py:1694).
  let (hit, rest) = lru.fetch_nearest_cache(&"m", &[9, 9]).unwrap();
  assert!(hit.is_none());
  assert_eq!(rest, vec![9, 9]);

  // ---- Eviction order at capacity (max_size = 3) ----
  let mut lru: LruPromptCache<&'static str> = LruPromptCache::new(3, 1 << 62);
  // Distinct, non-prefix-related token paths so pop_prefixes never folds
  // entries (each path's first token differs).
  lru
    .insert_cache(&"m", &[10], trimmable_entry(&[0.0]), "system")
    .unwrap(); // system: [ (m,[10]) ]
  lru
    .insert_cache(&"m", &[20], trimmable_entry(&[0.0]), "user")
    .unwrap(); // user: [ (m,[20]) ]
  lru
    .insert_cache(&"m", &[30], trimmable_entry(&[0.0]), "assistant")
    .unwrap(); // assistant: [ (m,[30]) ]
  assert_eq!(lru.len(), 3);

  // Insert a 4th (assistant) -> over capacity, pop() runs once.
  // lrus: assistant=[ (m,[30]) ] (len1), user=[ (m,[20]) ] (len1),
  // system=[ (m,[10]) ] (len1). pop: i=0 assistant non-empty and
  // len(assistant)=1 >= len(user)=1 -> popleft assistant -> evicts
  // (m,[30]). Then push the new assistant entry.
  lru
    .insert_cache(&"m", &[40], trimmable_entry(&[0.0]), "assistant")
    .unwrap();
  assert_eq!(lru.len(), 3);
  // [30] was evicted.
  assert!(lru.fetch_nearest_cache(&"m", &[30]).unwrap().0.is_none());
  // [10],[20],[40] survive.
  assert!(lru.fetch_nearest_cache(&"m", &[10]).unwrap().0.is_some());
  assert!(lru.fetch_nearest_cache(&"m", &[20]).unwrap().0.is_some());
  assert!(lru.fetch_nearest_cache(&"m", &[40]).unwrap().0.is_some());

  // trim_to(n_sequences=1): pop until len <= 1 (cache.py:1745-1749).
  // Current lrus: assistant=[ (m,[40]) ], user=[ (m,[20]) ],
  // system=[ (m,[10]) ] (each len1).
  //  pop#1: i=0 assistant(1) >= user(1) -> evict assistant (m,[40]).
  //  pop#2: assistant now empty -> i=1 user(1) >= system(1) -> evict user
  //         (m,[20]). len==1 (only system [10] left) -> stop.
  lru.trim_to(Some(1), None);
  assert_eq!(lru.len(), 1);
  assert!(lru.fetch_nearest_cache(&"m", &[10]).unwrap().0.is_some());
  assert!(lru.fetch_nearest_cache(&"m", &[40]).unwrap().0.is_none());
  assert!(lru.fetch_nearest_cache(&"m", &[20]).unwrap().0.is_none());
}

#[test]
fn lru_prefix_pop_on_trimmable_insert() {
  // mlx-lm cache.py:1719-1725: inserting a trimmable cache pops all *prefix*
  // entries (they "just take space"). Insert [1] then [1,2,3]: [1] is a
  // strict prefix of [1,2,3], so the [1] entry is removed on the second
  // insert.
  let mut lru: LruPromptCache<&'static str> = LruPromptCache::new(10, 1 << 62);
  lru
    .insert_cache(&"m", &[1], trimmable_entry(&[0.0]), "assistant")
    .unwrap();
  assert_eq!(lru.len(), 1);
  lru
    .insert_cache(
      &"m",
      &[1, 2, 3],
      trimmable_entry(&[0.0, 1.0, 2.0]),
      "assistant",
    )
    .unwrap();
  // [1] was a prefix -> popped; only [1,2,3] remains.
  assert_eq!(lru.len(), 1);
  assert!(
    lru
      .fetch_nearest_cache(&"m", &[1, 2, 3])
      .unwrap()
      .0
      .is_some()
  );
}

#[test]
fn lru_unknown_cache_type_is_err_and_no_untracked_entry() {
  // Regression: authoritative mlx-lm indexes fixed
  // per-type dicts (`self._n_bytes_by_type[cache_type]` cache.py:1711,
  // `self._lrus[cache_type]` cache.py:1639) → an unsupported `cache_type`
  // raises `KeyError` BEFORE `self._trie.add` (cache.py:1712), so nothing
  // is durably inserted. The Rust mirror must likewise reject an unknown
  // type with a clean `Err` and **leave the cache completely untouched** —
  // never silently drop the bucket (which would leave a fetchable,
  // untracked, un-evictable entry that bypasses `max_size`/`max_bytes`,
  // invisible to `len`/`nbytes`/`stats_by_type`/`trim_to`).
  let mut lru: LruPromptCache<&'static str> = LruPromptCache::new(1, 1);
  let r = lru.insert_cache(&"m", &[1, 2, 3], trimmable_entry(&[0.0, 1.0, 2.0]), "tool");
  assert!(r.is_err(), "unknown cache_type must be rejected, got Ok");

  // The rejected insert must not have mutated ANY state: no entry, no
  // bytes, not fetchable (so it can never bypass the caps).
  assert_eq!(
    lru.len(),
    0,
    "rejected insert left a tracked/untracked entry"
  );
  assert_eq!(lru.nbytes(), 0, "rejected insert leaked bytes into n_bytes");
  assert!(
    lru
      .fetch_nearest_cache(&"m", &[1, 2, 3])
      .unwrap()
      .0
      .is_none(),
    "rejected insert left a FETCHABLE untracked entry (cap-bypass footgun)"
  );
  // Every known bucket's stats stay zero (no untracked-bucket leak).
  for (_t, s) in lru.stats_by_type() {
    assert_eq!(s.n_sequences, 0);
    assert_eq!(s.n_bytes, 0);
  }

  // A subsequent VALID insert still works (the rejection didn't corrupt
  // internal state). Generous caps here so this checks *state integrity
  // after a rejection*, not the (separately tested) tight-cap eviction —
  // under `new(1, 1)` a real KV entry rightly exceeds `max_bytes` and is
  // evicted immediately, which is the intended cap enforcement, not a bug.
  let mut lru2: LruPromptCache<&'static str> = LruPromptCache::new(4, 1 << 62);
  // The unknown-type rejection on a *fresh* cache also leaves it pristine.
  assert!(
    lru2
      .insert_cache(&"m", &[7], trimmable_entry(&[0.0]), "tool")
      .is_err()
  );
  assert_eq!(lru2.len(), 0);
  lru2
    .insert_cache(
      &"m",
      &[1, 2, 3],
      trimmable_entry(&[0.0, 1.0, 2.0]),
      "assistant",
    )
    .unwrap();
  assert_eq!(lru2.len(), 1);
  assert!(
    lru2
      .fetch_nearest_cache(&"m", &[1, 2, 3])
      .unwrap()
      .0
      .is_some()
  );
}

// ───────────────────────── persistence ─────────────────────────

#[test]
fn save_writes_reference_class_names_and_round_trips() {
  // mlx-lm `save_prompt_cache`/`load_prompt_cache` (cache.py:43-85),
  // wire-format cross-checked vs mlx-swift-lm `savePromptCache`/
  // `loadPromptCache` + `cacheClassName`.
  let path = temp_path("rt.safetensors");

  let mut std_c = StandardKvCache::new();
  std_c
    .update(&kv(&[1.0, 2.0, 3.0]), &kv(&[4.0, 5.0, 6.0]))
    .unwrap();
  // Rotating prefill (S=3>1) so it has real state + a populated meta_state
  // (keep,max_size,offset,idx).
  let mut rot_c = RotatingKvCache::new(8, 4);
  rot_c
    .update(&kv(&[7.0, 8.0, 9.0]), &kv(&[1.0, 1.0, 1.0]))
    .unwrap();
  let rot_meta = rot_c.meta_state();
  assert_eq!(rot_meta.len(), 4);

  let mut meta = HashMap::new();
  meta.insert("model".to_string(), "demo".to_string());

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(std_c), Box::new(rot_c)];
  save_prompt_cache(&path, &cache, &meta).unwrap();

  // The on-disk metadata MUST name caches by the *reference* Python class
  // names ("KVCache" / "RotatingKVCache") under the swift/python
  // `tree_flatten` "2.i" key, so the file is loadable by mlx-lm /
  // mlx-swift unchanged.
  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(raw_meta.get("2.0").map(String::as_str), Some("KVCache"));
  assert_eq!(
    raw_meta.get("2.1").map(String::as_str),
    Some("RotatingKVCache")
  );
  // User metadata under "1.<key>".
  assert_eq!(raw_meta.get("1.model").map(String::as_str), Some("demo"));
  // Rotating meta_state under "0.1.j" (cache index 1, the rotating one).
  assert_eq!(
    raw_meta.get("0.1.0").map(String::as_str),
    Some(rot_meta[0].as_str())
  );
  assert_eq!(
    raw_meta.get("0.1.1").map(String::as_str),
    Some(rot_meta[1].as_str())
  );
  assert_eq!(
    raw_meta.get("0.1.2").map(String::as_str),
    Some(rot_meta[2].as_str())
  );
  assert_eq!(
    raw_meta.get("0.1.3").map(String::as_str),
    Some(rot_meta[3].as_str())
  );
  // DELIBERATE: rotating meta_state is the *authoritative mlx-lm* 4-tuple
  // `(keep, max_size, offset, _idx)` (cache.py:533) — NOT mlx-swift-lm's
  // 5-field `(keep, maxCacheSize, step, offset, idx)`. There is no
  // `"0.1.4"`. This is the inherited upstream mlx-lm↔swift divergence (see
  // the persist.rs module-doc compatibility scope); the merged
  // `RotatingKvCache::meta_state` is fixed at 4 fields (#32), and matching
  // mlx-lm here is what keeps the mlx-lm round-trip (the authoritative
  // spec) byte-exact.
  assert_eq!(rot_meta.len(), 4);
  assert_eq!(raw_meta.get("0.1.4"), None);
  // Arrays flattened "i.j": cache 0 (standard) has 2 arrays, cache 1
  // (rotating) has 2 arrays.
  assert!(_arrays.contains_key("0.0"));
  assert!(_arrays.contains_key("0.1"));
  assert!(_arrays.contains_key("1.0"));
  assert!(_arrays.contains_key("1.1"));

  // Round-trip: load reconstructs the right concrete types via from_state.
  let (loaded, loaded_meta) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  assert_eq!(loaded_meta.get("model").map(String::as_str), Some("demo"));

  // Cache 0 reconstructed as a Standard-equivalent (trimmable, full
  // attention) with the original keys/values.
  assert!(loaded[0].is_trimmable());
  assert_eq!(loaded[0].offset(), 3);
  let mut s0 = loaded[0].state().unwrap();
  assert_eq!(s0.len(), 2);
  assert_eq!(s0[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
  assert_eq!(s0[1].to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);

  // Cache 1 reconstructed as a Rotating-equivalent: offset/keep/max_size
  // restored from meta_state, state round-trips.
  assert_eq!(loaded[1].offset(), 3);
  assert_eq!(loaded[1].max_size(), Some(8));
  let mut s1 = loaded[1].state().unwrap();
  assert_eq!(s1.len(), 2);
  assert_eq!(s1[0].to_vec::<f32>().unwrap(), vec![7.0, 8.0, 9.0]);
  assert_eq!(s1[1].to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);

  let _ = fs::remove_file(&path);
}

#[test]
fn can_trim_and_trim_prompt_cache() {
  // cache.py:88-111. all(is_trimmable) and trim returns the *first* cache's
  // trim count.
  let mut a = StandardKvCache::new();
  a.update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let mut b = StandardKvCache::new();
  b.update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let mut cache: Vec<Box<dyn KvCache>> = vec![Box::new(a), Box::new(b)];

  assert!(can_trim_prompt_cache(&cache));
  // trim 2 tokens -> returns 2 (first cache's trimmed count); both trimmed.
  let trimmed = trim_prompt_cache(&mut cache, 2).unwrap();
  assert_eq!(trimmed, 2);
  assert_eq!(cache[0].offset(), 2);
  assert_eq!(cache[1].offset(), 2);

  // Empty cache list -> 0 (cache.py:109).
  let mut empty: Vec<Box<dyn KvCache>> = Vec::new();
  assert_eq!(trim_prompt_cache(&mut empty, 5).unwrap(), 0);

  // A non-trimmable rotating cache (window full) makes can_trim false ->
  // trim returns 0 and does NOT mutate. max_size=2, push 3 single tokens
  // so offset(3) >= max_size(2) -> is_trimmable() == false.
  let mut rot = RotatingKvCache::new(2, 0);
  rot.update(&kv(&[0.0]), &kv(&[0.0])).unwrap();
  rot.update(&kv(&[1.0]), &kv(&[1.0])).unwrap();
  rot.update(&kv(&[2.0]), &kv(&[2.0])).unwrap();
  assert!(!rot.is_trimmable());
  let mut mixed: Vec<Box<dyn KvCache>> = vec![Box::new(StandardKvCache::new()), Box::new(rot)];
  assert!(!can_trim_prompt_cache(&mixed));
  assert_eq!(trim_prompt_cache(&mut mixed, 1).unwrap(), 0);
}

#[test]
fn corrupt_or_missing_cache_file_is_err_not_panic() {
  // A missing file -> recoverable Err (mlx-c open failure surfaced), no
  // panic.
  let missing = temp_path("does_not_exist.safetensors");
  let _ = fs::remove_file(&missing);
  assert!(load_prompt_cache(&missing).is_err());

  // A non-safetensors / garbage file -> Err, never panic.
  let garbage = temp_path("garbage.safetensors");
  fs::write(&garbage, b"not a safetensors file at all\x00\xff\xfe").unwrap();
  assert!(load_prompt_cache(&garbage).is_err());
  let _ = fs::remove_file(&garbage);

  // A non-regular path (a directory named like the file) -> Err, not panic
  // (mirrors lm::load's reject-non-regular discipline).
  let dir = temp_path("a_directory.safetensors");
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  assert!(load_prompt_cache(&dir).is_err());
  let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_wrong_rank_rotating_state_is_err_not_panic() {
  // Regression: a *well-formed safetensors* whose
  // cross-tool layout names cache 1 `RotatingKVCache` with a valid 4-item
  // meta_state but **rank-1** state arrays. `RotatingKvCache::set_state`
  // mirrors mlx-lm verbatim (`self.keys, self.values = v`, no rank check),
  // so without the loader's rank gate this would reconstruct as `Ok` and
  // then PANIC the first time a cache method indexes `shape()[2]`. The
  // load-path contract is "corrupt file => Err, never panic", so
  // `load_prompt_cache` must reject it.
  let path = temp_path("corrupt_rank.safetensors");

  let mut arrays = HashMap::new();
  // Cache 1 (the rotating one) gets two RANK-1 arrays — the corruption.
  arrays.insert(
    "1.0".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
  );
  arrays.insert(
    "1.1".to_string(),
    Array::from_slice::<f32>(&[4.0, 5.0, 6.0], &(3usize,)).unwrap(),
  );
  let mut side = HashMap::new();
  // Reference class names (cross-tool wire format): cache 0 KVCache (empty
  // state), cache 1 RotatingKVCache with a *valid* 4-item meta_state
  // (keep,max_size,offset,idx) — only the arrays are wrong-rank.
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("2.1".to_string(), "RotatingKVCache".to_string());
  side.insert("0.1.0".to_string(), "4".to_string());
  side.insert("0.1.1".to_string(), "8".to_string());
  side.insert("0.1.2".to_string(), "3".to_string());
  side.insert("0.1.3".to_string(), "3".to_string());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Must be a clean recoverable error, NOT an `Ok` (footgun cache) and NOT
  // a panic.
  let r = load_prompt_cache(&path);
  assert!(
    r.is_err(),
    "wrong-rank rotating state must be rejected, got Ok"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn swift_shaped_5field_rotating_meta_is_err_not_panic() {
  // Regression: a prompt-cache file in the shared
  // cross-tool wire format whose `RotatingKVCache` carries mlx-swift-lm's
  // **5**-field `meta_state` `(keep, maxCacheSize, step, offset, idx)`
  // (`MLXLMCommon/KVCache.swift`) instead of authoritative mlx-lm's
  // **4**-tuple `(keep, max_size, offset, _idx)` (cache.py:533). This is
  // the documented upstream mlx-lm↔swift divergence (see persist.rs
  // module-doc compatibility scope): the merged 4-field
  // `RotatingKvCache::set_meta_state` (via `from_state`) cannot accept 5
  // fields — exactly as authoritative mlx-lm cannot. The faithful,
  // hostile-file-safe contract is that this is a clean recoverable
  // `Err`, **never a panic** and never a silently-wrong `Ok`. (4-D state
  // arrays so the rank gate passes and the meta_state arity is what is
  // actually being exercised.)
  let path = temp_path("swift5_rotating.safetensors");
  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0, 2.0]));
  arrays.insert("0.1".to_string(), kv(&[3.0, 4.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "RotatingKVCache".to_string());
  // Swift 5-field shape: keep, maxCacheSize, step, offset, idx.
  side.insert("0.0.0".to_string(), "4".to_string());
  side.insert("0.0.1".to_string(), "8".to_string());
  side.insert("0.0.2".to_string(), "256".to_string()); // the extra `step`
  side.insert("0.0.3".to_string(), "2".to_string());
  side.insert("0.0.4".to_string(), "2".to_string());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  let r = load_prompt_cache(&path);
  assert!(
    r.is_err(),
    "Swift-shaped 5-field RotatingKVCache meta_state must be a clean Err \
     (inherited upstream mlx-lm/swift divergence), got Ok"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn empty_state_caches_round_trip() {
  // A cache that holds nothing yet still round-trips (state == [], the
  // class name is still written so the slot reconstructs as the right
  // empty type).
  let path = temp_path("empty.safetensors");
  let std_c = StandardKvCache::new();
  let rot_c = RotatingKvCache::new(4, 2);
  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(std_c), Box::new(rot_c)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  let (_a, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(raw_meta.get("2.0").map(String::as_str), Some("KVCache"));
  assert_eq!(
    raw_meta.get("2.1").map(String::as_str),
    Some("RotatingKVCache")
  );

  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded[0].is_empty());
  assert!(loaded[1].is_empty());
  // Rotating meta_state still restored (keep=2, max_size=4).
  assert_eq!(loaded[1].max_size(), Some(4));
  let _ = fs::remove_file(&path);
}

#[test]
fn rank_valid_but_inconsistent_rotating_state_is_faithful_not_panic() {
  // DELIBERATE faithfulness-vs-hardening boundary (see the persist.rs
  // "Validation-scope boundary" note). A `RotatingKVCache` file with
  // RANK-VALID 4-D state of `S == 1` but pre-window metadata
  // `(keep=0, max_size=8, offset=3, idx=0)` is *internally inconsistent*
  // (a faithful pre-window rotating cache has `S == offset == idx`). The
  // authoritative spec — mlx-lm `_BaseCache.from_state` →
  // `RotatingKVCache.state`/`meta_state` setters (cache.py:294-295,
  // 535-541) — performs **no** such cross-field consistency check; it
  // assigns raw, so mlx-lm's own `load_prompt_cache` likewise accepts
  // this file and yields a logically-wrong-but-non-crashing cache. This
  // port is a 1:1 port of that spec, so it MUST behave identically:
  //
  //  * It is **`Ok`** (faithful: rank+arity valid ⇒ accepted, exactly as
  //    mlx-lm — NOT an `Err`; adding a reject here would diverge from the
  //    authoritative spec, the task's prime directive).
  //  * It does **NOT panic / UB** — the rank gate already eliminated the
  //    only case that panicked (wrong rank → `shape()[2]` OOB); a
  //    rank-valid inconsistent cache is memory-safe (reading its state is
  //    fine), the actual hostile-file contract ("corrupt ⇒ Err **or**
  //    safe; never panic/UB") being met.
  //
  // This test LOCKS that deliberate, spec-faithful behavior so a future
  // change that "hardens" it into an `Err` (diverging from mlx-lm) is a
  // visible, intentional decision — not silently introduced.
  let path = temp_path("inconsistent_rotating.safetensors");
  let mut arrays = HashMap::new();
  // Rank-valid 4-D [1,1,1,1] key/value, S == 1.
  arrays.insert("0.0".to_string(), kv(&[1.0]));
  arrays.insert("0.1".to_string(), kv(&[2.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "RotatingKVCache".to_string());
  // 4-field mlx-lm meta_state, but inconsistent: keep=0, max_size=8,
  // offset=3, idx=0 — pre-window (offset 3 < max_size 8) yet S=1 != 3 and
  // idx 0 != 3.
  side.insert("0.0.0".to_string(), "0".to_string());
  side.insert("0.0.1".to_string(), "8".to_string());
  side.insert("0.0.2".to_string(), "3".to_string());
  side.insert("0.0.3".to_string(), "0".to_string());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Faithful to authoritative mlx-lm: accepted (Ok), not Err, NEVER a
  // panic/UB. `load_prompt_cache` returning here at all (no panic) plus a
  // safe state read is the contract.
  let (loaded, _m) = load_prompt_cache(&path).expect(
    "rank+arity-valid (if semantically inconsistent) rotating cache must load Ok, \
     faithfully matching authoritative mlx-lm (no load-time semantic check)",
  );
  assert_eq!(loaded.len(), 1);
  assert!(!loaded[0].is_empty());
  assert_eq!(loaded[0].max_size(), Some(8));
  assert_eq!(loaded[0].offset(), 3); // raw from meta_state, as mlx-lm
  // Reading the reconstructed state is memory-safe (no panic/UB) — the
  // arrays are rank-valid; this is the no-panic half of the contract.
  let mut st = loaded[0].state().unwrap();
  assert_eq!(st.len(), 2);
  let _ = st[0].shape();
  let _ = st[1].to_vec::<f32>().unwrap();
  let _ = fs::remove_file(&path);
}

#[test]
fn all_kvcache_save_emits_mlx_lm_scalar_meta_and_round_trips() {
  // Regression (cross-loadability): authoritative
  // mlx-lm `_BaseCache.meta_state` is the empty STRING "" (cache.py
  // :138-139), which `mlx.utils.tree_flatten` serializes as the SCALAR
  // key `"0.{i}" -> ""` per no-meta cache. An all-`KVCache`/StandardKvCache
  // save MUST emit that scalar (not nothing), else mlx-lm's
  // `tree_unflatten` sees `info == {}` and `zip(classes, arrays, info)`
  // truncates to ZERO caches — the common full-attention prompt cache
  // would be silently unloadable by mlx-lm.
  let path = temp_path("all_kvcache.safetensors");
  let mut a = StandardKvCache::new();
  a.update(&kv(&[1.0, 2.0, 3.0]), &kv(&[4.0, 5.0, 6.0]))
    .unwrap();
  let mut b = StandardKvCache::new();
  b.update(&kv(&[7.0, 8.0]), &kv(&[9.0, 9.0])).unwrap();
  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(a), Box::new(b)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  // mlx-lm-loadable scalar empty-string meta_state for EVERY no-meta
  // cache (one `"0.{i}"` per layer, value ""), and NO list-form
  // `"0.{i}.{j}"` (StandardKvCache's meta_state is empty).
  assert_eq!(raw_meta.get("0.0").map(String::as_str), Some(""));
  assert_eq!(raw_meta.get("0.1").map(String::as_str), Some(""));
  assert_eq!(raw_meta.get("0.0.0"), None);
  assert_eq!(raw_meta.get("0.1.0"), None);
  // Reference class names still present (cross-tool kind labeling).
  assert_eq!(raw_meta.get("2.0").map(String::as_str), Some("KVCache"));
  assert_eq!(raw_meta.get("2.1").map(String::as_str), Some("KVCache"));

  // And it still round-trips through our own loader (the scalar `"0.i"`
  // form is accepted as "empty meta_state for cache i").
  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded[0].is_trimmable());
  assert_eq!(loaded[0].offset(), 3);
  assert_eq!(loaded[1].offset(), 2);
  let mut s0 = loaded[0].state().unwrap();
  assert_eq!(s0[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
  let _ = fs::remove_file(&path);
}

#[test]
fn mlx_lm_style_scalar_meta_file_loads() {
  // The inverse direction: an mlx-lm-saved (or mlx-lm-shaped) all-KVCache
  // file carries `"0.{i}" -> ""` scalars (mlx-lm `tree_flatten` of the
  // empty-string `_BaseCache.meta_state`). Our loader must accept that
  // scalar form and reconstruct the caches (NOT drop them / NOT panic).
  // Hand-built to exactly mlx-lm's wire shape.
  let path = temp_path("mlxlm_scalar_meta.safetensors");
  let mut arrays = HashMap::new();
  // Two KVCache caches, each 2 state arrays (keys/values), rank-4.
  arrays.insert("0.0".to_string(), kv(&[1.0, 2.0]));
  arrays.insert("0.1".to_string(), kv(&[3.0, 4.0]));
  arrays.insert("1.0".to_string(), kv(&[5.0]));
  arrays.insert("1.1".to_string(), kv(&[6.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("2.1".to_string(), "KVCache".to_string());
  // mlx-lm scalar empty-string meta_state, one per cache.
  side.insert("0.0".to_string(), String::new());
  side.insert("0.1".to_string(), String::new());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  let (loaded, _m) = load_prompt_cache(&path).expect(
    "mlx-lm-shaped scalar `0.i`-empty-meta_state all-KVCache file must load \
     (cross-loadability), not drop caches or panic",
  );
  assert_eq!(loaded.len(), 2);
  assert!(loaded[0].is_trimmable() && loaded[1].is_trimmable());
  assert_eq!(loaded[0].offset(), 2);
  assert_eq!(loaded[1].offset(), 1);
  let _ = fs::remove_file(&path);
}

#[test]
fn malformed_no_meta_kvcache_metadata_is_err() {
  // Regression: authoritative mlx-lm
  // `_BaseCache.meta_state` SETTER `raise`s on any truthy value
  // (`if v is not None and v: raise ValueError`, cache.py:142-145). So a
  // file naming cache 0 `KVCache` but carrying a *truthy* meta_state —
  // either the scalar `"0.0"="garbage"` or a list `"0.0.0"="x"` — is
  // malformed/schema-drifted and mlx-lm rejects it. Our loader must do
  // the same: a clean recoverable `Err`, never a silently-wrong `Ok`
  // that discards the value.
  //
  // (a) Truthy SCALAR `"0.0"="garbage"` on a KVCache.
  let p1 = temp_path("malformed_scalar_kvcache.safetensors");
  let mut a1 = HashMap::new();
  a1.insert("0.0".to_string(), kv(&[1.0]));
  a1.insert("0.1".to_string(), kv(&[2.0]));
  let mut s1 = HashMap::new();
  s1.insert("2.0".to_string(), "KVCache".to_string());
  s1.insert("0.0".to_string(), "garbage".to_string()); // truthy => mlx-lm raises
  io::save_safetensors_with_metadata(&p1, &a1, &s1).unwrap();
  assert!(
    load_prompt_cache(&p1).is_err(),
    "truthy scalar meta_state on a no-meta KVCache must be rejected (mlx-lm raises), got Ok"
  );
  let _ = fs::remove_file(&p1);

  // (b) Truthy LIST `"0.0.0"="x"` on a KVCache (no-meta kind, yet a
  // non-empty list meta_state).
  let p2 = temp_path("malformed_list_kvcache.safetensors");
  let mut a2 = HashMap::new();
  a2.insert("0.0".to_string(), kv(&[1.0]));
  a2.insert("0.1".to_string(), kv(&[2.0]));
  let mut s2 = HashMap::new();
  s2.insert("2.0".to_string(), "KVCache".to_string());
  s2.insert("0.0.0".to_string(), "x".to_string()); // non-empty list => reject
  io::save_safetensors_with_metadata(&p2, &a2, &s2).unwrap();
  assert!(
    load_prompt_cache(&p2).is_err(),
    "non-empty list meta_state on a no-meta KVCache must be rejected, got Ok"
  );
  let _ = fs::remove_file(&p2);

  // (c) Sanity: the FAITHFUL empty scalar `"0.0"=""` is still accepted
  // (no false-reject — falsy `""` is exactly mlx-lm's no-meta form).
  let p3 = temp_path("ok_empty_scalar_kvcache.safetensors");
  let mut a3 = HashMap::new();
  a3.insert("0.0".to_string(), kv(&[1.0]));
  a3.insert("0.1".to_string(), kv(&[2.0]));
  let mut s3 = HashMap::new();
  s3.insert("2.0".to_string(), "KVCache".to_string());
  s3.insert("0.0".to_string(), String::new()); // falsy "" => OK
  io::save_safetensors_with_metadata(&p3, &a3, &s3).unwrap();
  let (loaded, _m) = load_prompt_cache(&p3)
    .expect("empty-string scalar meta_state is the faithful no-meta form, must load Ok");
  assert_eq!(loaded.len(), 1);
  assert_eq!(loaded[0].offset(), 1);
  let _ = fs::remove_file(&p3);
}
