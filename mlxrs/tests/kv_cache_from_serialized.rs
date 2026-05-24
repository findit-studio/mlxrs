//! Tests for `KvCache::from_serialized` — the transactional restore
//! introduced by KVC-1.
//!
//! Two axes per concrete cache that overrides:
//!
//! 1. **Round-trip**: `to_serialized` → save the `(state, meta)` pair →
//!    construct a fresh cache of the same kind → `from_serialized(state,
//!    &meta)` → verify `state()` / `meta_state()` match the original.
//! 2. **Leaves self unchanged on error**: prime a cache with non-default
//!    state, snapshot `state()` / `meta_state()`, call `from_serialized`
//!    with a deliberately malformed `meta`, assert `Err`, then re-read
//!    `state()` / `meta_state()` and assert they equal the snapshot.
//!
//! Each concrete cache is tested under its own (overriding)
//! `from_serialized`. The trait-DEFAULT sequential-setter implementation
//! — which no concrete cache uses, since all 8 override it — is covered
//! separately via [`DefaultProbeCache`], a minimal test-only `KvCache`
//! that inherits the default and records its `set_state` → `set_meta_state`
//! call order.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{
    ArraysCache, BatchKvCache, BatchRotatingKvCache, CacheList, ChunkedKvCache, KvCache, MaskMode,
    QuantizedKvCache, QuantizedKvCacheImpl, RotatingKvCache, StandardKvCache,
  },
};

/// A `[1, 1, S, 1]` KV tensor with each token's value being its f32 id —
/// identical to the single-`i32` ramp used across the cache module's
/// existing tests so retained identities are directly grep'pable in
/// failure messages.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

/// Equal arrays by flattened f32 contents AND shape. `Array::to_vec` is
/// `&mut self` in this crate (it may trigger eval), so the comparators
/// take `&mut Array` — the shape comparison is read-only on `&self`.
fn assert_arrays_eq(got: &mut Array, want: &mut Array, ctx: &str) {
  assert_eq!(got.shape(), want.shape(), "{ctx}: shape mismatch");
  let g = got.to_vec::<f32>().unwrap();
  let w = want.to_vec::<f32>().unwrap();
  assert_eq!(g, w, "{ctx}: contents mismatch");
}

fn assert_arrays_eq_i32(got: &mut Array, want: &mut Array, ctx: &str) {
  assert_eq!(got.shape(), want.shape(), "{ctx}: shape mismatch");
  let g = got.to_vec::<i32>().unwrap();
  let w = want.to_vec::<i32>().unwrap();
  assert_eq!(g, w, "{ctx}: contents mismatch");
}

/// Compare two `Vec<Array>` (state lists) element-wise via f32 contents.
/// Both `Vec`s must be owned — `to_vec` needs `&mut`, and the slices we
/// pass in carry their own arrays.
fn assert_state_eq(got: &mut [Array], want: &mut [Array], ctx: &str) {
  assert_eq!(got.len(), want.len(), "{ctx}: state.len() mismatch");
  for (i, (g, w)) in got.iter_mut().zip(want.iter_mut()).enumerate() {
    assert_arrays_eq(g, w, &format!("{ctx}[{i}]"));
  }
}

/// Same as `assert_state_eq` but for state entries that mix f32 KV
/// tensors and i32 metadata arrays (batch caches' `offset`/
/// `left_padding`).
fn assert_state_eq_mixed_kv_then_i32(
  got: &mut [Array],
  want: &mut [Array],
  i32_indices: &[usize],
  ctx: &str,
) {
  assert_eq!(got.len(), want.len(), "{ctx}: state.len() mismatch");
  for (i, (g, w)) in got.iter_mut().zip(want.iter_mut()).enumerate() {
    if i32_indices.contains(&i) {
      assert_arrays_eq_i32(g, w, &format!("{ctx}[{i} as i32]"));
    } else {
      assert_arrays_eq(g, w, &format!("{ctx}[{i} as f32]"));
    }
  }
}

// ----------------------------------------------------------------------
// 1. The trait default — sequential set_state then set_meta_state.
// ----------------------------------------------------------------------

/// A minimal `KvCache` that does NOT override `from_serialized`, so the
/// **trait-default** implementation (sequential `set_state` then
/// `set_meta_state`) is actually exercised. Every concrete cache now
/// overrides `from_serialized`, so without this probe the default path
/// would be untested (Copilot #62 finding). The setters record their
/// call order so a test can assert the default's sequencing; all other
/// required methods are unreachable in that single code path.
#[derive(Default)]
struct DefaultProbeCache {
  calls: Vec<&'static str>,
}

impl KvCache for DefaultProbeCache {
  fn offset(&self) -> usize {
    0
  }
  fn update(&mut self, _k: &Array, _v: &Array) -> mlxrs::Result<(Array, Array)> {
    unreachable!("DefaultProbeCache::update is not exercised by the from_serialized default test")
  }
  fn state(&self) -> mlxrs::Result<Vec<Array>> {
    Ok(Vec::new())
  }
  fn materialize(&mut self) -> mlxrs::Result<()> {
    unreachable!(
      "DefaultProbeCache::materialize is not exercised by the from_serialized default test"
    )
  }
  fn set_state(&mut self, _state: Vec<Array>) -> mlxrs::Result<()> {
    self.calls.push("set_state");
    Ok(())
  }
  fn set_meta_state(&mut self, _m: &[String]) -> mlxrs::Result<()> {
    self.calls.push("set_meta_state");
    Ok(())
  }
  fn make_mask(&self, _n: usize, _w: Option<usize>, _r: bool) -> mlxrs::Result<MaskMode> {
    unreachable!("DefaultProbeCache::make_mask is not exercised")
  }
  fn nbytes(&self) -> usize {
    0
  }
  fn is_empty(&self) -> bool {
    true
  }
  fn copy(&self) -> mlxrs::Result<Box<dyn KvCache>> {
    unreachable!("DefaultProbeCache::copy is not exercised")
  }
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

#[test]
fn trait_default_from_serialized_calls_set_state_then_meta() {
  // The trait-DEFAULT `from_serialized` (used by no concrete cache —
  // all 8 override it) must call `set_state` THEN `set_meta_state`,
  // faithful to mlx-lm `cache.py:170-175`. `DefaultProbeCache` is the
  // only impl that inherits the default, so it's the sole coverage of
  // this ordering (Copilot #62 finding).
  let mut probe = DefaultProbeCache::default();
  probe.from_serialized(vec![kv(&[0.0])], &[]).unwrap();
  assert_eq!(
    probe.calls,
    vec!["set_state", "set_meta_state"],
    "trait-default from_serialized must call set_state then set_meta_state, in order"
  );
}

#[test]
fn standard_kvcache_from_serialized_round_trip() {
  // StandardKvCache OVERRIDES `from_serialized` (staged set_state +
  // set_meta_state, committed on success). This verifies that override's
  // round-trip; the trait-DEFAULT path is covered separately by
  // `trait_default_from_serialized_calls_set_state_then_meta` above.
  let mut original = StandardKvCache::new();
  let (_, _) = original
    .update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();

  let mut restored = StandardKvCache::new();
  restored.from_serialized(saved_state, &saved_meta).unwrap();

  // offset / state / meta all round-trip.
  assert_eq!(restored.offset(), 4);
  assert!(!restored.is_empty());
  let mut s = restored.state().unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!(s[0].shape(), vec![1, 1, 4, 1]);
  assert_eq!(s[0].to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
  assert_eq!(s[1].to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
  assert!(restored.meta_state().is_empty());
}

// ----------------------------------------------------------------------
// 2. RotatingKvCache — override path.
// ----------------------------------------------------------------------

#[test]
fn rotating_from_serialized_round_trip() {
  let mut original = RotatingKvCache::new(8, 2);
  // Drive in a few tokens so the cache isn't trivially empty.
  original
    .update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  original.update(&kv(&[4.0]), &kv(&[4.0])).unwrap();

  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();
  let saved_offset = original.offset();

  let mut restored = RotatingKvCache::new(0, 0);
  restored.from_serialized(saved_state, &saved_meta).unwrap();

  assert_eq!(restored.offset(), saved_offset);
  assert_eq!(restored.meta_state(), saved_meta);
  let mut restored_state = restored.state().unwrap();
  // Re-derive original_state by calling state() again on the original
  // (state() returns the same value for the immutable post-update cache).
  let mut original_state_again = original.state().unwrap();
  assert_state_eq(&mut restored_state, &mut original_state_again, "rotating");
}

#[test]
fn rotating_from_serialized_invalid_meta_leaves_self_unchanged() {
  let mut cache = RotatingKvCache::new(8, 2);
  cache
    .update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let mut original_state = cache.state().unwrap();
  let original_meta = cache.meta_state();
  let original_offset = cache.offset();

  // Build a structurally-valid state (re-pull from the cache) but a
  // deliberately malformed meta. Use 4 entries — the right arity — but
  // with a non-numeric value at index 2 (`offset`).
  let bad_state = cache.state().unwrap();
  let bad_meta: Vec<String> = vec![
    "2".to_string(),
    "8".to_string(),
    "not_a_number".to_string(),
    "0".to_string(),
  ];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err(), "expected Err on non-numeric offset");

  // The cache MUST be byte-identical to its pre-call state.
  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  let mut after_state = cache.state().unwrap();
  assert_state_eq(&mut after_state, &mut original_state, "rotating-unchanged");
}

#[test]
fn rotating_from_serialized_wrong_arity_meta_leaves_self_unchanged() {
  // Wrong arity (3 entries instead of 4) hits set_meta_state's length
  // gate BEFORE any field parse — still must leave the cache untouched.
  let mut cache = RotatingKvCache::new(8, 2);
  cache.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  let mut original_state = cache.state().unwrap();
  let original_meta = cache.meta_state();

  let bad_state = cache.state().unwrap();
  let bad_meta: Vec<String> = vec!["2".into(), "8".into(), "0".into()]; // 3, want 4
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.meta_state(), original_meta);
  let mut after_state = cache.state().unwrap();
  assert_state_eq(
    &mut after_state,
    &mut original_state,
    "rotating-wrong-arity",
  );
}

// ----------------------------------------------------------------------
// 3. ChunkedKvCache — override path.
// ----------------------------------------------------------------------

#[test]
fn chunked_from_serialized_round_trip() {
  let mut original = ChunkedKvCache::new(Some(8));
  original
    .update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();
  let saved_offset = original.offset();

  let mut restored = ChunkedKvCache::new(None);
  restored.from_serialized(saved_state, &saved_meta).unwrap();
  assert_eq!(restored.offset(), saved_offset);
  assert_eq!(restored.meta_state(), saved_meta);
  let mut restored_state = restored.state().unwrap();
  let mut original_state_again = original.state().unwrap();
  assert_state_eq(&mut restored_state, &mut original_state_again, "chunked");
}

#[test]
fn chunked_from_serialized_invalid_meta_leaves_self_unchanged() {
  let mut cache = ChunkedKvCache::new(Some(8));
  cache
    .update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let mut original_state = cache.state().unwrap();
  let original_meta = cache.meta_state();
  let original_offset = cache.offset();

  let bad_state = cache.state().unwrap();
  // 2 entries (right arity); non-numeric start_position.
  let bad_meta: Vec<String> = vec!["8".into(), "garbage".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  let mut after_state = cache.state().unwrap();
  assert_state_eq(&mut after_state, &mut original_state, "chunked-unchanged");
}

// ----------------------------------------------------------------------
// 4. QuantizedKvCacheImpl — override path.
// ----------------------------------------------------------------------

/// 4-D ramp `[1, 1, S, 64]` for the quantized cache. group_size = 64, so
/// `head_dim` MUST be 64 for `mx.quantize` to produce a single group per
/// row. Each row is the same ramp 0..64 so the quantized representation
/// is uniform across rows.
fn kv_quant(n_steps: usize) -> Array {
  let mut data = Vec::with_capacity(n_steps * 64);
  for _ in 0..n_steps {
    for j in 0..64 {
      data.push(j as f32);
    }
  }
  Array::from_slice::<f32>(&data, &(1usize, 1, n_steps, 64usize)).unwrap()
}

#[test]
fn quantized_from_serialized_round_trip() {
  let mut original = QuantizedKvCacheImpl::new(64, 8);
  original
    .update_quantized(&kv_quant(3), &kv_quant(3))
    .unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();
  let saved_offset = original.offset();

  let mut restored = QuantizedKvCacheImpl::new(0, 0);
  restored.from_serialized(saved_state, &saved_meta).unwrap();
  assert_eq!(restored.offset(), saved_offset);
  assert_eq!(restored.meta_state(), saved_meta);
  assert_eq!(restored.group_size(), 64);
  assert_eq!(restored.bits(), 8);
  let restored_state = restored.state().unwrap();
  let original_state_again = original.state().unwrap();
  // Quantized state arrays are I32 (weight) + f32/f16 (scales/biases),
  // but for shape/length-equality we only need the count + shapes here;
  // the exact bit-equality is covered by the dedicated quantized tests.
  assert_eq!(restored_state.len(), original_state_again.len());
  for (i, (a, b)) in restored_state
    .iter()
    .zip(original_state_again.iter())
    .enumerate()
  {
    assert_eq!(a.shape(), b.shape(), "quantized-state[{i}].shape");
  }
}

#[test]
fn quantized_from_serialized_wrong_arity_meta_leaves_self_unchanged() {
  let mut cache = QuantizedKvCacheImpl::new(64, 8);
  cache.update_quantized(&kv_quant(2), &kv_quant(2)).unwrap();
  let original_offset = cache.offset();
  let original_meta = cache.meta_state();

  let bad_state = cache.state().unwrap();
  // Wrong arity: 5 entries (valid is 3 or 4).
  let bad_meta: Vec<String> = vec![
    "2".into(),
    "64".into(),
    "8".into(),
    "extra".into(),
    "extra2".into(),
  ];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(cache.group_size(), 64);
  assert_eq!(cache.bits(), 8);
  // The state list shape is still valid.
  assert!(!cache.is_empty());
}

#[test]
fn quantized_from_serialized_invalid_meta_value_leaves_self_unchanged() {
  // Right arity (3), but non-numeric `bits` — must hit the parse Err
  // path AFTER set_state has already done its own (set_state)
  // sub-mutation in the staged cache, BUT the staged cache is local so
  // our own `cache` is unaffected.
  let mut cache = QuantizedKvCacheImpl::new(64, 8);
  cache.update_quantized(&kv_quant(2), &kv_quant(2)).unwrap();
  let original_offset = cache.offset();
  let original_meta = cache.meta_state();
  let original_gs = cache.group_size();
  let original_bits = cache.bits();

  let bad_state = cache.state().unwrap();
  let bad_meta: Vec<String> = vec!["2".into(), "64".into(), "not_bits".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(cache.group_size(), original_gs);
  assert_eq!(cache.bits(), original_bits);
}

// ----------------------------------------------------------------------
// 5. CacheList — the highest-payoff override.
// ----------------------------------------------------------------------

fn build_heterogeneous_cache_list() -> CacheList {
  let mut std_cache = StandardKvCache::new();
  std_cache
    .update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0]))
    .unwrap();
  let mut rot_cache = RotatingKvCache::new(4, 1);
  rot_cache
    .update(&kv(&[10.0, 11.0]), &kv(&[10.0, 11.0]))
    .unwrap();
  let mut q_cache = QuantizedKvCacheImpl::new(64, 8);
  q_cache
    .update_quantized(&kv_quant(2), &kv_quant(2))
    .unwrap();

  CacheList::new(vec![
    Box::new(std_cache),
    Box::new(rot_cache),
    Box::new(q_cache),
  ])
}

#[test]
fn cache_list_from_serialized_round_trip() {
  let original = build_heterogeneous_cache_list();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();

  // A fresh empty CacheList — `from_serialized` is what populates it.
  let mut restored = CacheList::new(Vec::new());
  restored.from_serialized(saved_state, &saved_meta).unwrap();

  // Same number of children.
  assert_eq!(restored.len(), 3);
  // Same flattened meta_state (child class names, framing, child meta).
  assert_eq!(restored.meta_state(), saved_meta);
  // Same per-child offsets (a structural identity).
  assert_eq!(restored.get(0).unwrap().offset(), 2);
  assert_eq!(restored.get(1).unwrap().offset(), 2);
  assert_eq!(restored.get(2).unwrap().offset(), 2);
  // Same per-child reference_class_name.
  assert_eq!(restored.get(0).unwrap().reference_class_name(), "KVCache");
  assert_eq!(
    restored.get(1).unwrap().reference_class_name(),
    "RotatingKVCache"
  );
  assert_eq!(
    restored.get(2).unwrap().reference_class_name(),
    "QuantizedKVCache"
  );
}

#[test]
fn cache_list_from_serialized_unknown_class_name_leaves_self_unchanged() {
  // Build a cache list, snapshot, then call from_serialized with a meta
  // whose child class name is unknown — must Err and not corrupt the
  // existing children.
  let mut cache = build_heterogeneous_cache_list();
  let original_meta = cache.meta_state();
  let original_len = cache.len();
  let original_child0_class = cache.get(0).unwrap().reference_class_name();
  let original_child1_class = cache.get(1).unwrap().reference_class_name();

  // A "valid" framing but with an unknown class name for child 0.
  // Frame: [childCount, (className, stateCount, metaCount, ...meta)*]
  let bad_meta: Vec<String> = vec![
    "1".into(),                     // 1 child
    "ThisClassDoesNotExist".into(), // bogus className
    "0".into(),                     // stateCount = 0
    "0".into(),                     // metaCount = 0
  ];
  let bad_state: Vec<Array> = Vec::new();
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err(), "expected Err on unknown class name");

  // Unchanged: same number of children, same kinds, same meta_state.
  assert_eq!(cache.len(), original_len);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(
    cache.get(0).unwrap().reference_class_name(),
    original_child0_class
  );
  assert_eq!(
    cache.get(1).unwrap().reference_class_name(),
    original_child1_class
  );
}

#[test]
fn cache_list_from_serialized_truncated_meta_leaves_self_unchanged() {
  // A "child count" that exceeds the framing budget — caught BEFORE the
  // children Vec is built, so `self.caches` is untouched.
  let mut cache = build_heterogeneous_cache_list();
  let original_meta = cache.meta_state();
  let original_len = cache.len();

  // childCount = 1000 with only 1 meta token after — far less than 3
  // framing fields per child.
  let bad_meta: Vec<String> = vec!["1000".into(), "KVCache".into()];
  let bad_state: Vec<Array> = Vec::new();
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.len(), original_len);
  assert_eq!(cache.meta_state(), original_meta);
}

// ----------------------------------------------------------------------
// 6. BatchKvCache — override path.
// ----------------------------------------------------------------------

/// A `[2, 1, S, 1]` batched KV tensor (batch=2, 1 head, head_dim=1) —
/// the smallest shape that exercises BatchKvCache's per-sequence
/// `[B]` offset/left_padding arrays.
fn kv_batch(seqs: &[&[f32]]) -> Array {
  let b = seqs.len();
  let s = seqs[0].len();
  // Concatenate row-major: [b, 1, s, 1].
  let mut data: Vec<f32> = Vec::with_capacity(b * s);
  for row in seqs {
    assert_eq!(row.len(), s, "kv_batch: ragged input");
    data.extend_from_slice(row);
  }
  Array::from_slice::<f32>(&data, &(b, 1usize, s, 1usize)).unwrap()
}

#[test]
fn batch_from_serialized_round_trip() {
  let mut original = BatchKvCache::new(&[1, 0]);
  original
    .update(
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
    )
    .unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();
  let saved_idx = original.offset();

  let mut restored = BatchKvCache::new(&[]);
  restored.from_serialized(saved_state, &saved_meta).unwrap();
  assert_eq!(restored.offset(), saved_idx);
  let mut restored_state = restored.state().unwrap();
  let mut original_state_again = original.state().unwrap();
  // BatchKvCache state is [keys, values, offset(i32), left_padding(i32)]
  // (indices 2 and 3 are i32 metadata arrays).
  assert_state_eq_mixed_kv_then_i32(
    &mut restored_state,
    &mut original_state_again,
    &[2, 3],
    "batch",
  );
}

#[test]
fn batch_from_serialized_wrong_state_arity_leaves_self_unchanged() {
  let mut cache = BatchKvCache::new(&[1, 0]);
  cache
    .update(
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
    )
    .unwrap();
  let original_offset = cache.offset();
  let mut original_state = cache.state().unwrap();

  // 3 arrays (valid is 0 or 4) — set_state errors.
  let bad_state = vec![
    kv_batch(&[&[0.0], &[1.0]]),
    kv_batch(&[&[0.0], &[1.0]]),
    Array::from_slice::<i32>(&[0, 0], &(2usize,)).unwrap(),
  ];
  let result = cache.from_serialized(bad_state, &[]);
  assert!(result.is_err());

  assert_eq!(cache.offset(), original_offset);
  let mut after_state = cache.state().unwrap();
  assert_state_eq_mixed_kv_then_i32(
    &mut after_state,
    &mut original_state,
    &[2, 3],
    "batch-unchanged",
  );
}

// ----------------------------------------------------------------------
// 7. BatchRotatingKvCache — override path.
// ----------------------------------------------------------------------

#[test]
fn batch_rotating_from_serialized_round_trip() {
  let mut original = BatchRotatingKvCache::new(4, &[1, 0]);
  original
    .update(
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
    )
    .unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();
  let saved_off = original.offset();

  let mut restored = BatchRotatingKvCache::new(0, &[]);
  restored.from_serialized(saved_state, &saved_meta).unwrap();
  assert_eq!(restored.offset(), saved_off);
  assert_eq!(restored.meta_state(), saved_meta);
  assert_eq!(restored.max_size(), Some(4));
  let mut restored_state = restored.state().unwrap();
  let mut original_state_again = original.state().unwrap();
  assert_state_eq_mixed_kv_then_i32(
    &mut restored_state,
    &mut original_state_again,
    &[2, 3],
    "batch-rotating",
  );
}

#[test]
fn batch_rotating_from_serialized_invalid_meta_leaves_self_unchanged() {
  let mut cache = BatchRotatingKvCache::new(4, &[0, 0]);
  cache
    .update(
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
    )
    .unwrap();
  let original_offset = cache.offset();
  let original_meta = cache.meta_state();
  let original_max_size = cache.max_size();

  let bad_state = cache.state().unwrap();
  // Right arity (4), but `rotated` is non-bool.
  let bad_meta: Vec<String> = vec!["4".into(), "2".into(), "2".into(), "neither".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(cache.max_size(), original_max_size);
}

// ----------------------------------------------------------------------
// 8. ArraysCache — override (transactional combined restore).
// ----------------------------------------------------------------------

#[test]
fn arrays_from_serialized_round_trip() {
  // Sparse cache: 4 slots, only slot 2 written.
  let mut original = ArraysCache::new(4);
  let slot_arr = Array::from_slice::<f32>(&[42.0, 43.0], &(1usize, 2)).unwrap();
  original.set(2, slot_arr).unwrap();
  let saved_state = original.state().unwrap();
  let saved_meta = original.meta_state();

  let mut restored = ArraysCache::new(0);
  restored.from_serialized(saved_state, &saved_meta).unwrap();
  assert_eq!(restored.meta_state(), saved_meta);
  // Slot 2 came back at index 2, others are None (so `get(2)` is Some,
  // `get(0)` is None) — the slot identity round-trip the ArraysCache
  // override exists to provide.
  assert!(restored.get(0).is_none());
  assert!(restored.get(2).is_some());
  // To assert the array contents, read through `state()` (an owned
  // `Vec<Array>` we can iterate over with `&mut` for `to_vec`); a sparse
  // cache compacts to one entry (the present slot).
  let mut restored_state = restored.state().unwrap();
  assert_eq!(restored_state.len(), 1);
  assert_eq!(restored_state[0].to_vec::<f32>().unwrap(), vec![42.0, 43.0]);
}

#[test]
fn arrays_from_serialized_invalid_meta_leaves_self_unchanged() {
  let mut cache = ArraysCache::new(4);
  let slot_arr = Array::from_slice::<f32>(&[7.0, 8.0], &(1usize, 2)).unwrap();
  cache.set(1, slot_arr).unwrap();
  let original_meta = cache.meta_state();

  // Capture the BEFORE through state() (owned arrays we can mut-borrow).
  let mut before_state = cache.state().unwrap();
  assert_eq!(before_state.len(), 1);
  let before_slot_contents = before_state[0].to_vec::<f32>().unwrap();

  // Non-numeric slotCount (`m[0]` parse fails — set_meta_state's first
  // parse step in the staged copy). ArraysCache's from_serialized builds
  // a FRESH local; on this Err the original cache must still hold its
  // slot 1 content.
  let bad_state: Vec<Array> = vec![Array::from_slice::<f32>(&[99.0], &(1usize, 1)).unwrap()];
  let bad_meta: Vec<String> = vec!["not_a_count".into(), "0".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err());

  // Untouched: slot 1 still has the original content, meta_state matches.
  assert_eq!(cache.meta_state(), original_meta);
  assert!(cache.get(1).is_some());
  let mut after_state = cache.state().unwrap();
  assert_eq!(after_state.len(), 1);
  let after_slot_contents = after_state[0].to_vec::<f32>().unwrap();
  assert_eq!(after_slot_contents, before_slot_contents);
}

// ----------------------------------------------------------------------
// 9. Post-setter invariant guards (Codex round-2 hardening).
//
// The `from_serialized` overrides must reject the same forged (state,
// meta) combinations the canonical `super::from_state` loader rejects —
// otherwise the new public API is observably weaker than the loader.
// These tests construct malformed-but-parse-valid pairs (the setters
// individually succeed; the post-setter consistency guard rejects) and
// assert (a) the override returns `Err` and (b) the existing cache state
// is byte-identical to its pre-call state.
// ----------------------------------------------------------------------

#[test]
fn standard_from_serialized_nonempty_meta_leaves_self_unchanged() {
  // StandardKvCache has trivial meta (empty `Vec<String>`). The default
  // trait impl would `set_state(state)?` (mutating self) FIRST, then
  // `set_meta_state(non_empty_meta)?` which errors because
  // StandardKvCache has no meta — leaving self holding the new
  // serialized state. The override stages + commits, so a non-empty
  // meta now rolls back cleanly.
  let mut cache = StandardKvCache::new();
  cache
    .update(&kv(&[10.0, 11.0]), &kv(&[10.0, 11.0]))
    .unwrap();
  let original_offset = cache.offset();
  let mut original_state = cache.state().unwrap();

  // Forge a valid 2-array state + non-empty meta. `set_state` would
  // succeed (legal rank-4 KV pair), but `set_meta_state(&["bogus"])`
  // returns Err per cache.py:142-145.
  let bad_state = vec![kv(&[99.0, 88.0]), kv(&[99.0, 88.0])];
  let bad_meta: Vec<String> = vec!["bogus".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(
    result.is_err(),
    "must reject non-empty meta on StandardKvCache"
  );

  // Cache state is byte-identical to its pre-call configuration.
  assert_eq!(cache.offset(), original_offset);
  let mut after_state = cache.state().unwrap();
  assert_state_eq(&mut after_state, &mut original_state, "standard");
}

#[test]
fn rotating_from_serialized_empty_state_nonzero_meta_rejected() {
  // Pre-populated cache that must remain unchanged after the failing call.
  let mut cache = RotatingKvCache::new(8, 2);
  cache
    .update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let original_meta = cache.meta_state();
  let original_offset = cache.offset();
  let mut original_state = cache.state().unwrap();

  // Empty state + non-zero meta (offset=5, idx=5) — parseable but
  // structurally impossible from a real round-trip (keys=None ⇒
  // offset==idx==0). `from_state` rejects this; `from_serialized` must too.
  let bad_state: Vec<Array> = Vec::new();
  let bad_meta: Vec<String> = vec!["4".into(), "8".into(), "5".into(), "5".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err(), "must reject empty state + non-zero meta");

  // Untouched: state(), meta_state(), offset() all match the pre-call value.
  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  let mut after_state = cache.state().unwrap();
  assert_state_eq(&mut after_state, &mut original_state, "rotating");
}

#[test]
fn quantized_from_serialized_empty_state_nonzero_offset_rejected() {
  // Pre-populated quantized cache that must remain unchanged.
  let mut cache = QuantizedKvCacheImpl::new(64, 8);
  cache.update_quantized(&kv_quant(2), &kv_quant(2)).unwrap();
  let original_meta = cache.meta_state();
  let original_offset = cache.offset();
  let original_gs = cache.group_size();
  let original_bits = cache.bits();
  let original_state_count = cache.state().unwrap().len();

  // Empty state + offset=5 — parseable but structurally impossible
  // (`mlx_lm/models/cache.py:294-296` `self.keys, self.values = v`
  // requires non-empty `v` to unpack). `from_state` rejects this;
  // `from_serialized` must too.
  let bad_state: Vec<Array> = Vec::new();
  let bad_meta: Vec<String> = vec!["5".into(), "64".into(), "8".into()];
  let result = cache.from_serialized(bad_state, &bad_meta);
  assert!(result.is_err(), "must reject empty state + non-zero offset");

  // Cache state is byte-identical to its pre-call configuration
  // (offset, meta_state, group_size, bits, state.len()).
  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(cache.group_size(), original_gs);
  assert_eq!(cache.bits(), original_bits);
  assert_eq!(cache.state().unwrap().len(), original_state_count);
}

#[test]
fn batch_rotating_from_serialized_structural_inconsistency_rejected() {
  // Pre-populated batch-rotating cache that must remain unchanged.
  let mut cache = BatchRotatingKvCache::new(4, &[0, 0]);
  cache
    .update(
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
      &kv_batch(&[&[0.0, 1.0], &[10.0, 11.0]]),
    )
    .unwrap();
  let original_meta = cache.meta_state();
  let original_offset = cache.offset();
  let original_max_size = cache.max_size();

  // Take a real saved state, but corrupt the meta: _idx=99 with L=2.
  // The setters individually succeed (each value parses), but the
  // structural guard rejects "_idx > L" — a write cursor past the
  // physical buffer end, impossible from a real round-trip.
  let saved_state = cache.state().unwrap();
  let bad_meta: Vec<String> = vec!["4".into(), "2".into(), "99".into(), "false".into()];
  let result = cache.from_serialized(saved_state, &bad_meta);
  assert!(
    result.is_err(),
    "must reject _idx beyond physical buffer length"
  );

  assert_eq!(cache.offset(), original_offset);
  assert_eq!(cache.meta_state(), original_meta);
  assert_eq!(cache.max_size(), original_max_size);
}
