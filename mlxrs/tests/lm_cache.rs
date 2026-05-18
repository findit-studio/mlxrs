//! Deterministic tests for the M3 KV cache (`mlxrs::lm::cache`), ported from
//! `mlx_lm.models.cache` (`KVCache` / `ConcatenateKVCache` /
//! `RotatingKVCache`) and cross-checked against mlx-swift-lm's `MLXLMCommon`
//! KV cache.
//!
//! Caches are 4-D `[B, n_kv_heads, S, head_dim]` with the sequence axis at
//! `-2`, matching mlx-lm. Tensors are tiny and built so every retained-token
//! identity is checkable from `to_vec`.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{CacheConfig, KvCache, RotatingKvCache, StandardKvCache, make_prompt_cache},
};

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// each step's value being its 0-based token id (so retained ids are
/// directly readable). `S == ids.len()`.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

#[test]
fn standard_append_offset_trim() {
  let mut c = StandardKvCache::new();
  assert!(c.is_empty());

  // First update: tokens [0,1,2,3] -> seq len 4, returned == input.
  let (mut k1, mut v1) = c
    .update_and_fetch(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  assert!(!c.is_empty());
  assert_eq!(k1.shape(), vec![1, 1, 4, 1]);
  assert_eq!(c.offset(), 4);
  assert_eq!(k1.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
  assert_eq!(v1.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);

  // Second update: tokens [4,5] -> concatenated along seq axis, offset 6.
  let (mut k2, _) = c
    .update_and_fetch(&kv(&[4.0, 5.0]), &kv(&[4.0, 5.0]))
    .unwrap();
  assert_eq!(k2.shape(), vec![1, 1, 6, 1]);
  assert_eq!(c.offset(), 6);
  assert_eq!(
    k2.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  );

  // trim(2): drop the last 2, offset == 4, stored prefix kept in sync.
  let trimmed = c.trim(2).unwrap();
  assert_eq!(trimmed, 2);
  assert_eq!(c.offset(), 4);

  // Next append extends the *trimmed* prefix (mlx-lm KVCache semantics):
  // [0,1,2,3] + [9] -> [0,1,2,3,9], offset 5.
  let (mut k3, _) = c.update_and_fetch(&kv(&[9.0]), &kv(&[9.0])).unwrap();
  assert_eq!(c.offset(), 5);
  assert_eq!(k3.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 9.0]);

  // trim never removes more than offset.
  let mut c2 = StandardKvCache::new();
  c2.update_and_fetch(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0]))
    .unwrap();
  assert_eq!(c2.trim(10).unwrap(), 2);
  assert_eq!(c2.offset(), 0);
}

#[test]
fn standard_wrong_rank_errors() {
  let mut c = StandardKvCache::new();
  // 2-D, not the required 4-D [B, n_kv_heads, S, head_dim] -> ShapeMismatch.
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2)).unwrap();
  assert!(c.update_and_fetch(&bad, &bad).is_err());
}

#[test]
fn rotating_keeps_prefix_and_window() {
  // max_size=8, keep=4. Feeding 12 single tokens (ids 0..=11) exercises
  // mlx-lm `RotatingKVCache._update_in_place`: a linear fill, then the
  // physical ring overwriting slots `keep..max_size` IN PLACE. The expected
  // values are the *physical* buffer order (NOT temporal order) — traced
  // 1:1 from `mlx_lm/models/cache.py`: once the ring is active, token 8
  // overwrites slot 4 → [0,1,2,3,8,5,6,7] (Codex's parity counterexample),
  // token 9 slot 5, etc. `offset` is the raw monotone counter (mlx-lm
  // `.offset`), never capped at `max_size`.
  let mut c = RotatingKvCache::new(8, 4);
  assert!(c.is_empty());
  assert!(c.is_trimmable());

  let expected: [(&[f32], usize); 12] = [
    (&[0.0], 1),
    (&[0.0, 1.0], 2),
    (&[0.0, 1.0, 2.0], 3),
    (&[0.0, 1.0, 2.0, 3.0], 4),
    (&[0.0, 1.0, 2.0, 3.0, 4.0], 5),
    (&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 6),
    (&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 7),
    (&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 8),
    (&[0.0, 1.0, 2.0, 3.0, 8.0, 5.0, 6.0, 7.0], 9),
    (&[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 6.0, 7.0], 10),
    (&[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 7.0], 11),
    (&[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0], 12),
  ];
  for (step, (want, off)) in expected.iter().enumerate() {
    let t = kv(&[step as f32]);
    let (mut k, mut v) = c.update_and_fetch(&t, &t).unwrap();
    assert_eq!(&k.to_vec::<f32>().unwrap(), want, "keys, step {step}");
    assert_eq!(&v.to_vec::<f32>().unwrap(), want, "values, step {step}");
    assert_eq!(c.offset(), *off, "raw offset, step {step}");
    if step == 11 {
      // Full window keeps shape [B, n_kv, max_size, head_dim].
      assert_eq!(k.shape(), vec![1, 1, 8, 1], "full-window shape");
    }
  }
  // Raw offset counted every token (mask / RoPE position), uncapped.
  assert_eq!(c.offset(), 12);
  // Window full -> not trimmable (mlx-lm `RotatingKVCache.is_trimmable`).
  assert!(!c.is_trimmable());
}

#[test]
fn rotating_multi_token_prefill_then_decode() {
  // mlx-lm `_update_concat` (S>1 prefill) then `_update_in_place` (S==1
  // decode). max_size=6, keep=2. Traced 1:1 from `cache.py`: the empty-case
  // concat stores the chunk verbatim; the first decode grows by one into
  // the buffer; the next decode finds `_idx == max_size` so it rotates to
  // `keep` and overwrites slot 2 IN PLACE → physical [0,1,6,3,4,5] (NOT the
  // temporal [0,1,3,4,5,6]). `offset` is the raw uncapped counter.
  let mut c = RotatingKvCache::new(6, 2);
  let chunk = kv(&[0.0, 1.0, 2.0, 3.0, 4.0]); // 5-token prefill, <= max_size
  let (mut k, _) = c.update_and_fetch(&chunk, &chunk).unwrap();
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
  assert_eq!(c.offset(), 5);

  // Decode one token -> grows into the buffer, total 6 == max_size.
  let (mut k, _) = c.update_and_fetch(&kv(&[5.0]), &kv(&[5.0])).unwrap();
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  );
  assert_eq!(c.offset(), 6);

  // Next token: `_idx == max_size` -> rotate to keep=2, overwrite slot 2
  // in place (mlx-lm physical ring), NOT a temporal re-order.
  let (mut k, _) = c.update_and_fetch(&kv(&[6.0]), &kv(&[6.0])).unwrap();
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 6.0, 3.0, 4.0, 5.0]
  );
  assert_eq!(c.offset(), 7);
}

#[test]
fn rotating_keep_zero_is_pure_sliding_window() {
  // keep=0: the ring wraps to slot 0 (no pinned prefix). Traced 1:1 from
  // mlx-lm `cache.py`: fill [0],[0,1],[0,1,2], then `_idx == max_size`
  // rotates to keep=0 so token 3 overwrites slot 0 → physical [3,1,2],
  // token 4 slot 1 → [3,4,2], token 5 slot 2 → [3,4,5]. This is the
  // physical ring order, NOT a temporal last-3 window.
  let mut c = RotatingKvCache::new(3, 0);
  let expected: [&[f32]; 6] = [
    &[0.0],
    &[0.0, 1.0],
    &[0.0, 1.0, 2.0],
    &[3.0, 1.0, 2.0],
    &[3.0, 4.0, 2.0],
    &[3.0, 4.0, 5.0],
  ];
  for (step, want) in expected.iter().enumerate() {
    let t = kv(&[step as f32]);
    let (mut k, _) = c.update_and_fetch(&t, &t).unwrap();
    assert_eq!(&k.to_vec::<f32>().unwrap(), want, "step {step}");
    assert_eq!(c.offset(), step + 1, "raw offset, step {step}");
  }
}

#[test]
fn rotating_active_ring_then_concat() {
  // The mixed path mlx-lm `cache.py:456-466` exercises and the simpler
  // tests miss: an ALREADY-ROTATED ring (built by S==1 decode) followed by
  // an S>1 `_update_concat`. max_size=8, keep=4. Single tokens 0..=8 leave
  // the physical ring [0,1,2,3,8,5,6,7] (offset 9, cursor at slot 5). The
  // S=2 append [9,10] must: temporal-order to [0,1,2,3,5,6,7,8], reassign
  // `_idx = 8` (the reordered length, cache.py:458), so `trim_size =
  // 8 - 8 + 1 = 1` (NOT 0 from the stale cursor) -> drop one post-`keep`
  // token and append -> [0,1,2,3,6,7,8,9,10] (len 9). Computing trim from
  // the stale cursor would wrongly keep id 5 and return a len-10 buffer.
  let mut c = RotatingKvCache::new(8, 4);
  for id in 0..=8 {
    let t = kv(&[id as f32]);
    c.update_and_fetch(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 9, "after single-token fill+rotate");

  let app = kv(&[9.0, 10.0]); // S = 2 -> _update_concat (active-ring branch)
  let (mut k, mut v) = c.update_and_fetch(&app, &app).unwrap();
  let want = vec![0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0];
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    want,
    "keys: temporal-order trim"
  );
  assert_eq!(
    v.to_vec::<f32>().unwrap(),
    want,
    "values: temporal-order trim"
  );
  assert_eq!(k.shape(), vec![1, 1, 9, 1], "max_size + S - 1 = 9");
  assert_eq!(c.offset(), 11, "raw offset += S");
}

#[test]
fn make_prompt_cache_one_per_layer_and_kind() {
  // No sliding window -> all Standard, one per layer.
  let cfg = CacheConfig {
    num_hidden_layers: 3,
    sliding_window: None,
  };
  let cache = make_prompt_cache(&cfg);
  assert_eq!(cache.len(), 3);
  assert!(cache.iter().all(|c| matches!(c, KvCache::Standard(_))));

  // Sliding window set -> all Rotating(max_size=8, keep=4), one per layer.
  let cfg = CacheConfig {
    num_hidden_layers: 3,
    sliding_window: Some(8),
  };
  let cache = make_prompt_cache(&cfg);
  assert_eq!(cache.len(), 3);
  assert!(cache.iter().all(|c| matches!(c, KvCache::Rotating(_))));
}

#[test]
fn kvcache_enum_dispatches_methods() {
  // The enum's inherent methods dispatch without an exhaustive external
  // match (so a deferred Quantized variant stays additive).
  let mut c = KvCache::Standard(StandardKvCache::new());
  assert!(c.is_empty());
  let (k, _) = c
    .update_and_fetch(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  assert_eq!(k.shape(), vec![1, 1, 3, 1]);
  assert_eq!(c.offset(), 3);
  assert!(!c.is_empty());
  assert_eq!(c.trim(1).unwrap(), 1);
  assert_eq!(c.offset(), 2);

  let mut c = KvCache::Rotating(RotatingKvCache::new(4, 2));
  for i in 0..6 {
    let t = kv(&[i as f32]);
    c.update_and_fetch(&t, &t).unwrap();
  }
  // Raw mlx-lm `cache.offset` (uncapped) — 6 single-token appends, same
  // counter semantics as the Standard variant above.
  assert_eq!(c.offset(), 6);
}
