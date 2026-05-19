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
  lm::cache::{
    CacheConfig, KvCache, MaskMode, RopeOffset, RotatingKvCache, StandardKvCache,
    create_attention_mask, create_causal_mask, from_state, make_prompt_cache,
  },
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
    .update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  assert!(!c.is_empty());
  assert_eq!(k1.shape(), vec![1, 1, 4, 1]);
  assert_eq!(c.offset(), 4);
  assert_eq!(k1.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
  assert_eq!(v1.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);

  // Second update: tokens [4,5] -> concatenated along seq axis, offset 6.
  let (mut k2, _) = c.update(&kv(&[4.0, 5.0]), &kv(&[4.0, 5.0])).unwrap();
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
  let (mut k3, _) = c.update(&kv(&[9.0]), &kv(&[9.0])).unwrap();
  assert_eq!(c.offset(), 5);
  assert_eq!(k3.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 9.0]);

  // trim never removes more than offset.
  let mut c2 = StandardKvCache::new();
  c2.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  assert_eq!(c2.trim(10).unwrap(), 2);
  assert_eq!(c2.offset(), 0);
}

#[test]
fn standard_wrong_rank_errors() {
  let mut c = StandardKvCache::new();
  // 2-D, not the required 4-D [B, n_kv_heads, S, head_dim] -> ShapeMismatch.
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2)).unwrap();
  assert!(c.update(&bad, &bad).is_err());
}

/// Regression (rank-safety, no panic on a recoverable path): the merged
/// faithful-revert removed the non-faithful K/V seq cross-check, leaving
/// `RotatingKvCache::update_in_place` (S==1) reading `values.shape()[3]`
/// raw. A rank-invalid `values` (with valid 4-D `keys`) must surface as a
/// recoverable `Err(Error::ShapeMismatch{..})` (the faithful equivalent of
/// mlx-lm `cache.py:478` `values.shape[3]` raising a catchable
/// `IndexError`), NEVER a Rust slice out-of-bounds panic on the
/// `Result`-returning public `update`.
#[test]
fn rotating_update_in_place_rank_invalid_values_errors_no_panic() {
  let mut c = RotatingKvCache::new(8, 4);
  // Valid 4-D single-token `keys` -> the S==1 `_update_in_place` dispatch.
  let keys = kv(&[0.0]);
  // 2-D `values` (rank < 4): would hit the raw `values.shape()[3]`.
  let bad_values = Array::from_slice::<f32>(&[0.0, 1.0], &(1usize, 2)).unwrap();
  let r = c.update(&keys, &bad_values);
  assert!(
    matches!(&r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "rank-invalid values on the S==1 path must be a recoverable \
     ShapeMismatch, got {r:?}"
  );
}

/// Regression (rank-safety, no panic on a recoverable path): the S>1
/// `RotatingKvCache::update_concat` dispatch with valid 4-D `keys` but a
/// rank-invalid `values` must also be a recoverable `Err` (never a panic).
/// Critically, this must hold on the empty-cache branch too: if an empty
/// ring accepts rank-invalid `values` via direct assignment/clone, it can
/// cache a malformed 2-D buffer and only panic later when subsequent valid
/// updates index cached-value shape as if it were 4-D. The load-bearing
/// guarantee is identical on the public `Result` API: reject the bad
/// update recoverably and leave the cache usable.
#[test]
fn rotating_update_concat_rank_invalid_values_errors_no_panic() {
  let mut c = RotatingKvCache::new(6, 2);
  // Valid 4-D multi-token `keys` (S == 2) -> the S>1 `_update_concat`.
  // Use an empty cache so this exercises the empty-cache branch.
  let keys = kv(&[2.0, 3.0]);
  // 2-D `values` (rank < 4).
  let bad_values = Array::from_slice::<f32>(&[2.0, 3.0], &(1usize, 2)).unwrap();
  let r = c.update(&keys, &bad_values);
  assert!(
    r.is_err(),
    "rank-invalid values on the empty-cache S>1 path must be a recoverable \
     Err, got {r:?}"
  );
  // A subsequent valid update must still succeed, proving the failed call
  // did not store the malformed 2-D values buffer in the cache.
  c.update(&keys, &keys).unwrap();
}

/// Regression (rank-safety, `concat_parts` single-part fast path):
/// `RotatingKvCache::update_concat` with `max_size=1, keep=0` and a
/// populated cache drops every retained (empty) 4-D piece in `_trim`,
/// leaving ONLY the rank-invalid external `values` as the lone surviving
/// part. The `concat_parts` `[one]` fast path must NOT clone that through
/// (it would `Ok`-store a rank-invalid buffer that a *later* valid update
/// hits via a raw cached-shape read in `temporal_order`/`set_seq` and
/// panics); it must surface the same recoverable `Err` mlx-lm's
/// `mx.concatenate(to_cat, axis=2)` raises on a rank-mismatched lone
/// element — and a subsequent valid update must still work (no cache
/// corruption, no panic on the `Result` API).
#[test]
fn rotating_update_concat_single_part_fast_path_rank_invalid_no_corruption() {
  let mut c = RotatingKvCache::new(1, 0);
  // Seed a valid 1-token ring (offset=1, idx=1, buffer is 4-D len 1).
  let seed = kv(&[0.0]);
  c.update(&seed, &seed).unwrap();
  // S == 2 valid keys -> `_update_concat`; rank-invalid (2-D) values.
  // `_trim(trim_size>0, ...)` yields empty 4-D `keep`/tail slices that the
  // rank-safe filter drops, so only `bad_values` survives -> `[one]` arm.
  let keys = kv(&[1.0, 2.0]);
  let bad_values = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2)).unwrap();
  let r = c.update(&keys, &bad_values);
  assert!(
    r.is_err(),
    "rank-invalid lone-surviving values must be a recoverable Err \
     (mx.concatenate of a single rank-mismatched element raises), got {r:?}"
  );
  // The failed update must NOT have stored the rank-invalid buffer: a
  // subsequent VALID update must succeed without panicking on a raw
  // cached-shape read (`temporal_order`/`set_seq`).
  let good = kv(&[3.0]);
  let r2 = c.update(&good, &good);
  assert!(
    r2.is_ok(),
    "a valid update after a rejected rank-invalid one must succeed (no \
     cache corruption / no Result-path panic), got {r2:?}"
  );
}

/// Regression (rank-safety, no panic): `StandardKvCache::update` with valid
/// 4-D `keys` but a rank-invalid `values`. It has no raw `values.shape[N]`
/// metadata read (`keys` is `seq_len`-validated; `values` only flows
/// through `mx.concatenate`/`try_clone`), so the recoverable error comes
/// from the backend concat against the seeded 4-D buffer — still a
/// recoverable `Err`, never a panic, on the `Result` API.
#[test]
fn standard_rank_invalid_values_errors_no_panic() {
  let mut c = StandardKvCache::new();
  // Seed a 4-D buffer so the next update concatenates (where a rank-invalid
  // `values` reaches `mx.concatenate`).
  let seed = kv(&[0.0, 1.0]);
  c.update(&seed, &seed).unwrap();
  let keys = kv(&[2.0]);
  let bad_values = Array::from_slice::<f32>(&[2.0, 3.0], &(1usize, 2)).unwrap();
  let r = c.update(&keys, &bad_values);
  assert!(
    r.is_err(),
    "rank-invalid values must be a recoverable Err, got {r:?}"
  );
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
    let (mut k, mut v) = c.update(&t, &t).unwrap();
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
  let (mut k, _) = c.update(&chunk, &chunk).unwrap();
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
  assert_eq!(c.offset(), 5);

  // Decode one token -> grows into the buffer, total 6 == max_size.
  let (mut k, _) = c.update(&kv(&[5.0]), &kv(&[5.0])).unwrap();
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
  );
  assert_eq!(c.offset(), 6);

  // Next token: `_idx == max_size` -> rotate to keep=2, overwrite slot 2
  // in place (mlx-lm physical ring), NOT a temporal re-order.
  let (mut k, _) = c.update(&kv(&[6.0]), &kv(&[6.0])).unwrap();
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
    let (mut k, _) = c.update(&t, &t).unwrap();
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
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 9, "after single-token fill+rotate");

  let app = kv(&[9.0, 10.0]); // S = 2 -> _update_concat (active-ring branch)
  let (mut k, mut v) = c.update(&app, &app).unwrap();
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
  // No sliding window -> all Standard, one per layer. The faithful
  // observable kind distinction is `max_size()` (mlx-lm `cache.max_size`
  // / swift `maxSize`): a Standard cache is unbounded (`None`).
  let cfg = CacheConfig {
    num_hidden_layers: 3,
    sliding_window: None,
  };
  let cache = make_prompt_cache(&cfg);
  assert_eq!(cache.len(), 3);
  assert!(cache.iter().all(|c| c.max_size().is_none()));
  assert!(cache.iter().all(|c| c.is_empty()));

  // Sliding window set -> all Rotating(max_size=8, keep=4), one per layer
  // (a Rotating cache reports `max_size() == Some(8)`).
  let cfg = CacheConfig {
    num_hidden_layers: 3,
    sliding_window: Some(8),
  };
  let cache = make_prompt_cache(&cfg);
  assert_eq!(cache.len(), 3);
  assert!(cache.iter().all(|c| c.max_size() == Some(8)));
  assert!(cache.iter().all(|c| c.is_empty()));
}

#[test]
fn kvcache_trait_dispatch() {
  // `Box<dyn KvCache>` dispatches without an exhaustive external match (so
  // a deferred Quantized cache stays additive). Same hand-traced sequence
  // and expected values as the former enum-dispatch test.
  let mut c: Box<dyn KvCache> = Box::new(StandardKvCache::new());
  assert!(c.is_empty());
  let (k, _) = c
    .update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  assert_eq!(k.shape(), vec![1, 1, 3, 1]);
  assert_eq!(c.offset(), 3);
  assert!(!c.is_empty());
  assert_eq!(c.trim(1).unwrap(), 1);
  assert_eq!(c.offset(), 2);

  let mut c: Box<dyn KvCache> = Box::new(RotatingKvCache::new(4, 2));
  for i in 0..6 {
    let t = kv(&[i as f32]);
    c.update(&t, &t).unwrap();
  }
  // Raw mlx-lm `cache.offset` (uncapped) — 6 single-token appends, same
  // counter semantics as the Standard cache above.
  assert_eq!(c.offset(), 6);
}

// ── 0.2: mask.rs (create_causal_mask / create_attention_mask) ────────────

/// `create_causal_mask(N, offset, None)` — the lower-triangular boolean
/// `[N, offset+N]` mask `linds >= rinds`, traced 1:1 from mlx-lm
/// `models.base.create_causal_mask`.
#[test]
fn causal_mask_no_window_offset_zero() {
  // N=3, offset=0: rinds=[0,1,2], linds=[0,1,2] -> 3x3 lower triangular.
  let mut m = create_causal_mask(3, 0, None).unwrap();
  assert_eq!(m.shape(), vec![3, 3]);
  // row i (query) sees key j iff i >= j.
  assert_eq!(
    m.to_vec::<bool>().unwrap(),
    vec![
      true, false, false, // q0
      true, true, false, // q1
      true, true, true, // q2
    ]
  );
}

/// With a non-zero offset the query indices are shifted (`linds =
/// arange(offset, offset+N)`), so the mask is `[N, offset+N]`.
#[test]
fn causal_mask_with_offset() {
  // N=2, offset=3: rinds=[0..5), linds=[3,4]. mask[i][j] = (3+i) >= j.
  let mut m = create_causal_mask(2, 3, None).unwrap();
  assert_eq!(m.shape(), vec![2, 5]);
  assert_eq!(
    m.to_vec::<bool>().unwrap(),
    vec![
      true, true, true, true, false, // q at pos 3
      true, true, true, true, true, // q at pos 4
    ]
  );
}

/// A sliding `window_size` adds `linds < rinds + window_size`, masking
/// keys older than the window.
#[test]
fn causal_mask_windowed() {
  // N=4, offset=0, window_size=2: keep j where i >= j AND i < j + 2.
  let mut m = create_causal_mask(4, 0, Some(2)).unwrap();
  assert_eq!(m.shape(), vec![4, 4]);
  assert_eq!(
    m.to_vec::<bool>().unwrap(),
    vec![
      true, false, false, false, // q0: {0}
      true, true, false, false, // q1: {0,1}
      false, true, true, false, // q2: {1,2}
      false, false, true, true, // q3: {2,3}
    ]
  );
}

/// `create_attention_mask` (`cache.py:114-126`): the symbolic vs array vs
/// none decision tree.
#[test]
fn attention_mask_mode_decision_tree() {
  // window_size set -> always an array mask.
  assert!(matches!(
    create_attention_mask(4, 0, false, Some(2)).unwrap(),
    MaskMode::Array(_)
  ));
  // N == 1, no window -> None.
  assert!(matches!(
    create_attention_mask(1, 7, false, None).unwrap(),
    MaskMode::None
  ));
  // N > 1, return_array -> array.
  assert!(matches!(
    create_attention_mask(3, 0, true, None).unwrap(),
    MaskMode::Array(_)
  ));
  // N > 1, no window, not return_array -> symbolic "causal".
  assert!(matches!(
    create_attention_mask(3, 0, false, None).unwrap(),
    MaskMode::Causal
  ));
}

// ── 0.3 / 0.4: trait defaults, make_mask, state, from_state ──────────────

/// The [`KvCache`] trait defaults match the reference: a scalar
/// `rope_offset`, no metadata for Standard, `Rotating` carries
/// `(keep,max_size,offset,idx)`, and neither cache is a quantized /
/// batched refinement (the deferred-PR hooks return `None`).
#[test]
fn trait_defaults_rope_offset_and_meta_state() {
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  // rope_offset defaults to the scalar offset.
  match s.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 2),
    RopeOffset::Batch(_) => panic!("standard cache must use a scalar RoPE offset"),
  }
  // Standard has no meta_state (mlx-lm `_BaseCache`); not quantized/batched.
  assert!(s.meta_state().is_empty());
  assert!(s.as_quantized().is_none());
  assert!(s.as_batch_positioned().is_none());

  let mut r = RotatingKvCache::new(8, 4);
  for i in 0..3 {
    let t = kv(&[i as f32]);
    r.update(&t, &t).unwrap();
  }
  // mlx-lm `RotatingKVCache.meta_state` = (keep, max_size, offset, idx).
  assert_eq!(r.meta_state(), vec!["4", "8", "3", "3"]);
  match r.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 3),
    RopeOffset::Batch(_) => panic!("rotating cache must use a scalar RoPE offset"),
  }
}

/// `state()`/`nbytes()`/`copy()`: Standard returns `[keys, values]` (else
/// `[]`), `nbytes` is the byte sum, and `copy` is an independent deep clone.
#[test]
fn standard_state_nbytes_copy() {
  let mut s = StandardKvCache::new();
  assert!(s.state().unwrap().is_empty());
  assert_eq!(s.nbytes(), 0);

  s.update(&kv(&[0.0, 1.0, 2.0, 3.0]), &kv(&[0.0, 1.0, 2.0, 3.0]))
    .unwrap();
  let mut st = s.state().unwrap();
  assert_eq!(st.len(), 2);
  assert_eq!(st[0].to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
  // 4 keys + 4 values f32 (1x1x4x1 each) = 8 * 4 bytes.
  assert_eq!(s.nbytes(), 8 * 4);

  // copy is independent: mutating the copy must not move the original.
  // `copy()` is fallible (Array::try_clone): it surfaces a Result and is
  // never silently mapped to a half-populated cache.
  let mut c = s.copy().unwrap();
  c.update(&kv(&[9.0]), &kv(&[9.0])).unwrap();
  assert_eq!(c.offset(), 5);
  assert_eq!(s.offset(), 4);
}

/// `StandardKvCache::make_mask` is mlx-lm `KVCache.make_mask`
/// (`cache.py:393`): `create_attention_mask(N, offset, return_array,
/// window_size)` with the caller's `window_size` passed THROUGH unchanged
/// (never substituted with `max_size`). Every expected value is hand-traced
/// from `cache.py:393` -> `cache.py:114-126` / `base.py:create_causal_mask`,
/// with the cache driven to a known `offset == 3`.
#[test]
fn standard_make_mask_matches_create_attention_mask() {
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  assert_eq!(s.offset(), 3);

  // make_mask(1, None, false): cache.py:114 window_size is None ->
  // cache.py:119 N==1 -> None.
  assert!(matches!(
    s.make_mask(1, None, false).unwrap(),
    MaskMode::None
  ));

  // make_mask(3, None, false): window_size None, N=3 != 1,
  // return_array false -> cache.py:124 "causal" -> Causal.
  assert!(matches!(
    s.make_mask(3, None, false).unwrap(),
    MaskMode::Causal
  ));

  // make_mask(3, None, true): window_size None, N=3 != 1,
  // return_array true -> cache.py:122 create_causal_mask(3, offset=3, None).
  // base.py: rinds=arange(0,6)=[0..5], linds=arange(3,6)=[3,4,5];
  // mask[i][j] = linds[i] >= rinds[j]; shape [N, offset+N] = [3, 6].
  match s.make_mask(3, None, true).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.shape(), vec![3, 6]);
      assert_eq!(
        m.to_vec::<bool>().unwrap(),
        vec![
          true, true, true, true, false, false, // q@3
          true, true, true, true, true, false, // q@4
          true, true, true, true, true, true, // q@5
        ]
      );
    }
    _ => panic!("make_mask(3,None,true) must be an Array (cache.py:122)"),
  }

  // make_mask(2, Some(4), false): cache.py:117 window_size is not None ->
  // cache.py:118 create_causal_mask(2, offset=3, window_size=4). Compare
  // against the direct call AND the hand-traced values. base.py:
  // rinds=[0..4], linds=arange(3,5)=[3,4]; mask = linds>=rinds AND
  // linds < rinds+4; shape [2, offset+2] = [2, 5].
  match s.make_mask(2, Some(4), false).unwrap() {
    MaskMode::Array(mut m) => {
      let mut want = create_causal_mask(2, s.offset(), Some(4)).unwrap();
      assert_eq!(m.shape(), vec![2, 5]);
      assert_eq!(m.to_vec::<bool>().unwrap(), want.to_vec::<bool>().unwrap());
      assert_eq!(
        m.to_vec::<bool>().unwrap(),
        vec![
          true, true, true, true, false, // q@3
          false, true, true, true, true, // q@4
        ]
      );
    }
    _ => panic!("make_mask(2,Some(4),_) must be an Array (cache.py:118)"),
  }
}

/// `RotatingKvCache::make_mask` is mlx-lm `RotatingKVCache.make_mask`
/// (`cache.py:554-578`) — the rotating cache's OWN override (NOT the generic
/// `create_attention_mask`). Every expected value is hand-traced from
/// `cache.py:554-578` (and `base.py:create_causal_mask` for the array path),
/// with the cache driven by deterministic single-token updates to fixed
/// `offset` / `_idx`.
#[test]
fn rotating_make_mask_windowed_and_rolled() {
  // ── N>1, small offset -> "causal" (cache.py:557-563) ──
  // max_size=8, keep=4; 2 single-token updates -> offset=2, idx=2
  // (cache.py:469-510 `_update_in_place`: linear fill, no rotate).
  let mut c = RotatingKvCache::new(8, 4);
  for id in 0..2 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 2);
  // cache.py:558 ws = None or 8 = 8; cache.py:559 offset =
  // min(max_size-1=7, self.offset=2) = 2; cache.py:560
  // offset+N = 2+3 = 5 > ws(8)? no; return_array false -> "causal".
  assert!(matches!(
    c.make_mask(3, None, false).unwrap(),
    MaskMode::Causal
  ));
  // cache.py:565 N==1, window_size None -> None.
  assert!(matches!(
    c.make_mask(1, None, false).unwrap(),
    MaskMode::None
  ));
  assert!(matches!(
    c.make_mask(1, None, true).unwrap(),
    MaskMode::None
  ));

  // ── N>1, large offset -> windowed array (cache.py:560-561) ──
  // max_size=8, keep=4; 10 single-token updates -> offset=10.
  let mut c = RotatingKvCache::new(8, 4);
  for id in 0..10 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 10);
  // cache.py:558 ws = None or 8 = 8; cache.py:559 offset =
  // min(max_size-1=7, self.offset=10) = 7; cache.py:560
  // offset+N = 7+3 = 10 > ws(8)? yes -> cache.py:561
  // create_causal_mask(N=3, offset=7, window_size=8); base.py shape
  // [N, offset+N] = [3, 10].
  match c.make_mask(3, None, false).unwrap() {
    MaskMode::Array(m) => assert_eq!(m.shape(), vec![3, 10]),
    _ => panic!("large-offset N>1 must be a windowed Array (cache.py:561)"),
  }

  // ── N==1 rolled physical-ring mask (cache.py:565-578) ──
  // max_size=4, keep=2; 6 single-token updates (ids 0..5). Hand-traced
  // `_update_in_place`: steps 0-3 fill buffer [0,1,2,3] (offset 4, idx 4);
  // step 4 idx==max_size so rotate idx->keep=2, overwrite slot 2 -> buffer
  // [0,1,4,3] (offset 5, idx 3); step 5 no rotate, overwrite slot 3 ->
  // [0,1,4,5] (offset 6, idx 4).
  let mut c = RotatingKvCache::new(4, 2);
  for id in 0..6 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 6);

  // make_mask(1, Some(2), false), w=2: cache.py:568
  // offset(6) >= w(2) AND max_size(4) > w(2) -> enter rolled branch.
  // cache.py:569-571 idx = _idx(4); 4 >= max_size(4) -> idx = 0.
  // cache.py:572-575 offset(6) < max_size(4)? no -> mask_size = 4.
  // cache.py:576 mask = arange(4) >= (mask_size - w = 4-2 = 2)
  //   = [0,1,2,3] >= 2 = [F, F, T, T].
  // cache.py:577 mask = roll(mask, shift = idx+1 = 0+1 = 1):
  //   roll of len-4 by 1 -> out[i] = a[(i-1) mod 4]
  //   = [a[3], a[0], a[1], a[2]] = [T, F, F, T].
  match c.make_mask(1, Some(2), false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.shape(), vec![4]);
      assert_eq!(m.to_vec::<bool>().unwrap(), vec![true, false, false, true]);
    }
    _ => panic!("N==1 windowed rolled case must be an Array (cache.py:578)"),
  }

  // make_mask(1, Some(8), false): offset(6) >= w(8)? no -> Python falls
  // through with no return -> None (cache.py:568 guard false).
  assert!(matches!(
    c.make_mask(1, Some(8), false).unwrap(),
    MaskMode::None
  ));
}

/// `from_state` reconstructs Standard/Rotating from their
/// state+meta_state, keyed on the SOURCE class names a real prompt cache
/// writes (mlx-lm `save_prompt_cache` emits `type(c).__name__`,
/// `cache.py:56`; `load_prompt_cache` rebuilds via `globals()[name]`,
/// `cache.py:79-82`), accepts our Rust-name aliases for back-compat, and
/// rejects an unknown kind with `Error::Backend`.
#[test]
fn from_state_roundtrip_and_unknown_kind() {
  // Standard round-trip via the SOURCE name a real prompt cache writes:
  // mlx-lm `KVCache` (cache.py:56 -> cache.py:80). state()/meta_state()
  // produce exactly what `save_prompt_cache` serializes (cache.py:53-54).
  let mut s = StandardKvCache::new();
  s.update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  let s_state = s.state().unwrap();
  let s_meta = s.meta_state();
  let s2 = from_state("KVCache", s_state, &s_meta).unwrap();
  assert_eq!(s2.offset(), 3);
  assert!(!s2.is_empty());
  // The reconstructed cache's state matches the source (id-by-id).
  let s2_state = s2.state().unwrap();
  assert_eq!(s2_state.len(), 2);
  let mut s2k = s2_state[0].try_clone().unwrap();
  assert_eq!(s2k.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);

  // Rotating round-trip via the SOURCE name `RotatingKVCache`
  // (cache.py:56 -> cache.py:80); meta_state carries
  // (keep, max_size, offset, idx) per cache.py:531-533.
  let mut r = RotatingKvCache::new(8, 4);
  for i in 0..5 {
    let t = kv(&[i as f32]);
    r.update(&t, &t).unwrap();
  }
  let r_state = r.state().unwrap();
  let r_meta = r.meta_state();
  // Build the `from_state` argument via per-element `try_clone()` rather
  // than `r_state.clone()`: the project uses the fallible
  // `Array::try_clone` and removes the infallible `Array: Clone` (#33), so
  // tests must NOT depend on `Vec<Array>: Clone`. `r_state` is kept for the
  // later `r_state[0].try_clone()` source comparison below.
  let r_state_arg: Vec<Array> = r_state
    .iter()
    .map(|a| a.try_clone())
    .collect::<mlxrs::Result<Vec<_>>>()
    .unwrap();
  let r2 = from_state("RotatingKVCache", r_state_arg, &r_meta).unwrap();
  assert_eq!(r2.offset(), 5);
  assert_eq!(r2.max_size(), Some(8));
  // The reconstructed Rotating's state matches the source buffer.
  let r2_state = r2.state().unwrap();
  assert_eq!(r2_state.len(), 2);
  let mut r2k = r2_state[0].try_clone().unwrap();
  let mut r0k = r_state[0].try_clone().unwrap();
  assert_eq!(r2k.to_vec::<f32>().unwrap(), r0k.to_vec::<f32>().unwrap());

  // Back-compat: our own Rust struct name still loads (round-trip alias).
  let r3 = from_state("RotatingKvCache", r.state().unwrap(), &r.meta_state()).unwrap();
  assert_eq!(r3.offset(), 5);
  assert_eq!(r3.max_size(), Some(8));

  // Unknown kind -> recoverable Error::Backend (later PRs add more arms,
  // e.g. "ChunkedKVCache"/"QuantizedKVCache").
  assert!(from_state("ChunkedKvCache", Vec::new(), &[]).is_err());
}

/// Regression (FINDING 2): mlx-lm/Swift rotating state setters require a
/// non-empty (two-array) state, so an empty buffer paired with a non-zero
/// `offset`/`idx` is unreachable upstream. `from_state` must reject that
/// inconsistent combination (it would otherwise let the next `update`
/// surface placeholder zeros as "prior context"), while a genuinely-empty
/// cache (empty state + zero meta) must still construct fine.
#[test]
fn from_state_rotating_empty_with_nonzero_meta_is_invalid() {
  // Empty state + meta (keep, max_size, offset, idx) with NON-ZERO
  // offset/idx -> the impossible "keys=None but offset>0" cache. Reject.
  let bad_meta = vec![
    "4".to_string(), // keep
    "8".to_string(), // max_size
    "5".to_string(), // offset (non-zero)
    "5".to_string(), // idx    (non-zero)
  ];
  let bad = from_state("RotatingKVCache", Vec::new(), &bad_meta);
  assert!(
    bad.is_err(),
    "empty state + non-zero offset/idx must not yield a usable cache"
  );

  // Non-zero idx alone (offset == 0) is also inconsistent for an empty
  // cache (invariant: empty => offset==0 && idx==0).
  let bad_idx_only = vec![
    "0".to_string(), // keep
    "8".to_string(), // max_size
    "0".to_string(), // offset (zero)
    "3".to_string(), // idx    (non-zero)
  ];
  assert!(
    from_state("RotatingKVCache", Vec::new(), &bad_idx_only).is_err(),
    "empty state + non-zero idx must be rejected"
  );

  // Genuinely-empty cache: empty state + ALL-ZERO offset/idx -> Ok, an
  // empty rotating cache (invariant satisfied).
  let zero_meta = vec![
    "4".to_string(), // keep
    "8".to_string(), // max_size
    "0".to_string(), // offset (zero)
    "0".to_string(), // idx    (zero)
  ];
  let ok = from_state("RotatingKVCache", Vec::new(), &zero_meta).unwrap();
  assert!(
    ok.is_empty(),
    "empty state + zero meta is a valid empty cache"
  );
  assert_eq!(ok.offset(), 0);
  assert_eq!(ok.max_size(), Some(8));
}

// ── Copilot: `offset += S` ring-update overflow ──────────────────────────

/// Regression (Copilot): `RotatingKvCache`'s ring update advances
/// `self.offset += S` (mlx-lm `cache.py:464`/`:506`). A corrupt/hostile
/// prompt cache can restore `offset` near `usize::MAX` via `set_meta_state`
/// (a non-empty, minimal-valid state so `from_state`'s `empty ⇒
/// offset==0 && idx==0` invariant isn't tripped); the next update would then
/// wrap (release) / panic (debug). Both physical paths
/// (`_update_concat`, S>1; `_update_in_place`, S==1) must instead return
/// `Err(ShapeMismatch)` BEFORE mutating any ring state (no partial
/// mutation). Non-overflowing inputs are byte-identical (the ring algorithm
/// outcome is unchanged — exercised by the #31 parity tests, untouched).
#[test]
fn rotating_offset_overflow_is_rejected_without_partial_mutation() {
  // Minimal valid non-empty cache (so the empty-state invariant in
  // `from_state` is irrelevant; here we drive `set_meta_state` directly).
  let mut c = RotatingKvCache::new(8, 4);
  let t1 = kv(&[0.0]);
  c.update(&t1, &t1).unwrap();
  // Load `offset` at `usize::MAX` (keep/max_size/idx kept consistent &
  // non-empty-safe). meta_state == (keep, max_size, offset, idx).
  c.set_meta_state(&[
    "4".to_string(),
    "8".to_string(),
    usize::MAX.to_string(),
    "1".to_string(),
  ])
  .unwrap();
  let before_meta = c.meta_state();
  let before_state_len = c.state().unwrap().len();

  // S == 2 -> `_update_concat`: `offset (MAX) + S (2)` overflows.
  let two = kv(&[1.0, 2.0]);
  let r_concat = c.update(&two, &two);
  assert!(
    matches!(r_concat, Err(mlxrs::Error::ShapeMismatch { .. })),
    "concat-path offset overflow must be Err(ShapeMismatch), got {r_concat:?}"
  );
  // No partial mutation: all four meta fields + buffer presence unchanged.
  assert_eq!(
    c.meta_state(),
    before_meta,
    "overflow (concat) must not partially mutate ring state"
  );
  assert_eq!(c.state().unwrap().len(), before_state_len);

  // S == 1 -> `_update_in_place`: `offset (MAX) + S (1)` overflows.
  let one = kv(&[3.0]);
  let r_inplace = c.update(&one, &one);
  assert!(
    matches!(r_inplace, Err(mlxrs::Error::ShapeMismatch { .. })),
    "in-place-path offset overflow must be Err(ShapeMismatch), got {r_inplace:?}"
  );
  assert_eq!(
    c.meta_state(),
    before_meta,
    "overflow (in-place) must not partially mutate ring state"
  );
  assert_eq!(c.state().unwrap().len(), before_state_len);
}

// ── Copilot: `set_meta_state` atomic restore ─────────────────────────────

/// Regression (Copilot): `RotatingKvCache::set_meta_state` parses
/// `(keep, max_size, offset, idx)` (cache.py:531-533). A parse error on a
/// later value must NOT leave earlier fields mutated — the call is atomic
/// (all four parse OK, then all four assigned), so a failed `set_meta_state`
/// leaves the existing cache exactly as it was. `meta_state()` exposes all
/// four fields, so an unchanged round-trip proves nothing was touched.
#[test]
fn rotating_set_meta_state_is_atomic_on_malformed_input() {
  let mut c = RotatingKvCache::new(8, 4);
  for i in 0..5 {
    let t = kv(&[i as f32]);
    c.update(&t, &t).unwrap();
  }
  let before = c.meta_state();
  assert_eq!(before.len(), 4, "(keep, max_size, offset, idx)");

  // Valid keep/max_size, NON-numeric offset (3rd value) -> parse Err.
  let bad = c.set_meta_state(&[
    "4".to_string(),
    "8".to_string(),
    "not-a-number".to_string(),
    "2".to_string(),
  ]);
  assert!(
    bad.is_err(),
    "malformed offset must make set_meta_state fail"
  );
  // ALL four fields unchanged (keep was NOT mutated even though it parsed).
  assert_eq!(
    c.meta_state(),
    before,
    "a failed set_meta_state must leave keep/max_size/offset/idx unchanged"
  );

  // A subsequent VALID set_meta_state still succeeds and applies fully
  // (the atomic rewrite did not wedge the cache).
  c.set_meta_state(&[
    "2".to_string(),
    "16".to_string(),
    "9".to_string(),
    "3".to_string(),
  ])
  .unwrap();
  assert_eq!(
    c.meta_state(),
    vec![
      "2".to_string(),
      "16".to_string(),
      "9".to_string(),
      "3".to_string()
    ]
  );
}

// ── iarange f32-exactness bound (Copilot finding 1) ──────────────────────

/// `iarange(start, stop)` (mask.rs) builds index positions through the
/// `f32`-only `Array::arange` AND casts its own *exclusive* `stop` to `f32`.
/// `f32` represents every integer in `[0, 2^24]` exactly and rounds
/// `2^24 + 1` *down* to `2^24`. The bound therefore rejects `stop > 2^24`
/// (strictly): at the maximum allowed `stop == 2^24` both the `stop` cast
/// (so the element count) and every produced value `[start, stop-1]` are
/// exact. The earlier bound only guarded the largest produced value
/// (`stop-1 > 2^24`), so `stop == 2^24 + 1` passed yet
/// `(2^24+1) as f32 == 2^24`, making `arange` stop one element short and
/// emit a silently-too-short (corrupt) mask returned as `Ok`. This test
/// asserts EXACT shape+contents in range (so a rounded/short range would
/// fail), `Err` (not a short mask) at `stop == 2^24 + 1`, `Ok` at the
/// exact-max `stop == 2^24`, and that the `Err` propagates through
/// `RotatingKvCache::make_mask`'s own `iarange`. `iarange` is `pub(crate)`,
/// so it is exercised through its only public entrypoints
/// `create_causal_mask` (`rinds = iarange(0, offset+N)`,
/// `linds = iarange(offset, offset+N)`) and `RotatingKvCache::make_mask`
/// (`iarange(0, mask_size)`, cache.py:576).
#[test]
fn iarange_mask_exact_in_range_and_rejects_past_f32_limit() {
  const F32_EXACT_INT_MAX: usize = 1usize << 24;

  // ── In-range, EXACT shape + contents ──
  // `create_causal_mask(3, 0, None)` (base.py:create_causal_mask):
  // `rinds = iarange(0, offset+N) = iarange(0, 3) = [0,1,2]`,
  // `linds = rinds` (offset==0), `mask[i][j] = linds[i] >= rinds[j]`.
  // A rounded/short `iarange(0,3)` (e.g. [0,1]) would change shape AND
  // these booleans, so this pins `iarange(0,3) == [0,1,2]` exactly.
  let mut m = create_causal_mask(3, 0, None).unwrap();
  assert_eq!(m.shape(), vec![3, 3]);
  assert_eq!(
    m.to_vec::<bool>().unwrap(),
    vec![
      true, false, false, // q0
      true, true, false, // q1
      true, true, true, // q2
    ]
  );

  // Offset case exercises BOTH `iarange(0, offset+N)` AND the
  // offset-start `iarange(offset, offset+N)`. `create_causal_mask(2, 3,
  // None)` (base.py): `rinds = iarange(0, 5) = [0,1,2,3,4]`,
  // `linds = iarange(3, 5) = [3,4]`, `mask[i][j] = linds[i] >= rinds[j]`,
  // shape `[N, offset+N] = [2, 5]`. The exact boolean grid below holds
  // iff `iarange(0,5) == [0,1,2,3,4]` and `iarange(3,5) == [3,4]`.
  let mut mo = create_causal_mask(2, 3, None).unwrap();
  assert_eq!(mo.shape(), vec![2, 5]);
  assert_eq!(
    mo.to_vec::<bool>().unwrap(),
    vec![
      true, true, true, true, false, // linds[0]=3 >= rinds=[0,1,2,3,4]
      true, true, true, true, true, // linds[1]=4 >= rinds=[0,1,2,3,4]
    ]
  );

  // ── Boundary: stop == 2^24 + 1 -> Err (was wrongly Ok/short mask) ──
  // `create_causal_mask(2, 2^24 - 1, None)`: `iarange(0, offset+N)` has
  // `stop = (2^24 - 1) + 2 = 2^24 + 1`. Old guard (`stop-1 > 2^24`):
  // `stop-1 = 2^24` -> passed -> `(2^24+1) as f32 == 2^24` -> `arange`
  // stopped one element short -> a silently-too-short mask returned `Ok`.
  // Tightened guard (`stop > 2^24`) rejects it. (Array stays lazy/unevaled;
  // no 16M alloc on the Err path.)
  let over = create_causal_mask(2, F32_EXACT_INT_MAX - 1, None);
  assert!(
    over.is_err(),
    "stop == 2^24+1 must be rejected (it rounds to 2^24 -> short/corrupt mask), not returned Ok"
  );

  // ── Exact max: stop == 2^24 -> Ok (is_ok ONLY; do NOT materialize) ──
  // `create_causal_mask(1, 2^24 - 1, None)`: `iarange(0, offset+N)` has
  // `stop = (2^24 - 1) + 1 = 2^24`; `(2^24) as f32 == 2^24` exact, every
  // produced value in `[0, 2^24 - 1]` exact, count exact. The 16M-element
  // array is built lazily and never `.to_vec`'d / evaluated here.
  let at_max = create_causal_mask(1, F32_EXACT_INT_MAX - 1, None);
  assert!(
    at_max.is_ok(),
    "stop == 2^24 is exactly representable in f32 (cast + every index exact) -> must be Ok"
  );

  // ── Propagation through RotatingKvCache::make_mask (cache.py:576) ──
  // `RotatingKVCache.make_mask` builds `iarange(0, mask_size)` for the
  // rolled physical-ring decode mask (cache.py:572-576:
  // `mask_size = offset+1 if offset < max_size else max_size`,
  // `mask = arange(mask_size) >= mask_size - window_size`). Driving
  // `offset >= window`, `max_size > window`, and `offset < max_size`
  // pushes `mask_size = offset + 1` past 2^24, so `iarange` returns the
  // tightened `Err`, which `make_mask` must propagate via `?` (NOT build a
  // corrupted ring mask). State is restored via cheap meta-state (no eval,
  // no huge alloc).
  let mut rc = RotatingKvCache::new(F32_EXACT_INT_MAX + 8, 4);
  rc.set_meta_state(&[
    "4".to_string(),                     // keep
    (F32_EXACT_INT_MAX + 8).to_string(), // max_size
    (F32_EXACT_INT_MAX + 4).to_string(), // offset (>= window, < max_size)
    "5".to_string(),                     // idx
  ])
  .unwrap();
  // mask_size = offset + 1 = 2^24 + 5 > 2^24 -> iarange(0, mask_size) Err.
  let rmask = rc.make_mask(1, Some(2), false);
  assert!(
    rmask.is_err(),
    "rotating make_mask must propagate the iarange f32-limit Err, not build a corrupted ring mask"
  );
}

// ── Copilot finding 3: offset + N must not panic/wrap before iarange ─────

/// `create_causal_mask` computes `offset + N` with `checked_add` *before*
/// building any range. A hostile/corrupt loaded `offset` (mlx-lm prompt
/// cache `set_meta_state`) would otherwise overflow usize → a debug panic,
/// or a release wrap to a small value that then *passes* `iarange`'s `2^24`
/// guard and silently emits a wrong mask. The overflow must be a recoverable
/// `Err` (no panic, no wrap-then-wrong-mask). `create_causal_mask` is the
/// public entry mlx-lm `base.py:create_causal_mask` ports
/// (`rinds = arange(offset+N)`, `linds = arange(offset, offset+N)`), and
/// `StandardKvCache::make_mask` → `create_attention_mask` →
/// `create_causal_mask` forwards the raw `offset` unchanged (cache.py:393,
/// cache.py:117-118), so the same `Err` surfaces through the cache surface
/// too. (The rotating make_mask N>1 path caps `offset` at `max_size-1`
/// inside its cache.py:554 decision logic — out of scope to change — so the
/// raw-`offset` overflow is exercised at `create_causal_mask`, its faithful
/// public source-line entry.)
#[test]
fn create_causal_mask_offset_plus_n_overflow_is_err_not_panic() {
  // offset = usize::MAX, N = 2 -> offset + N overflows usize.
  // base.py: `rinds = mx.arange(offset + N)` — the very first line.
  let r = create_causal_mask(2, usize::MAX, None);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "offset + N overflow must be Err::ShapeMismatch (no debug panic, no release wrap)"
  );

  // Through the StandardKvCache::make_mask -> create_attention_mask ->
  // create_causal_mask forward (cache.py:393 -> cache.py:117-118): a
  // window_size selects the create_causal_mask branch with the raw offset.
  // Drive the same overflow via from_state's set_state (offset =
  // keys.shape[-2]); a usize::MAX-length array is impossible, so this leg
  // asserts the function-level guard the cache forward depends on holds for
  // the windowed branch too (offset + N still the first computation).
  let rw = create_causal_mask(3, usize::MAX, Some(4));
  assert!(
    matches!(rw, Err(mlxrs::Error::ShapeMismatch { .. })),
    "windowed create_causal_mask must also reject offset + N overflow before any range"
  );
}

// ── Copilot finding 4: window_size >= range is mlx-lm's unbounded no-op ───

/// mlx-lm `base.py:create_causal_mask` applies
/// `mask & (linds < rinds + window_size)` with unbounded Python ints, so a
/// `window_size` at least the full index range is a NO-OP (the term is
/// always true: `max(linds) = offset+N-1 = total-1`, `min(rinds) = 0`, so
/// `linds < rinds + window_size` holds for every position once
/// `window_size >= total`). The Rust port casts `window_size as i32`, which
/// wraps for `window_size > i32::MAX` and could wrongly mask valid
/// positions. The faithful + safe fix skips the windowing term entirely when
/// `window_size >= total` (== mlx-lm's no-op), and only otherwise applies it
/// — where `window_size < total <= 2^24 < i32::MAX` makes the cast exact.
#[test]
fn create_causal_mask_huge_window_is_unwindowed_noop() {
  // Small shape, derive expected from base.py with NO window_size.
  // N=4, offset=0: rinds=[0,1,2,3], linds=[0,1,2,3] (offset==0),
  // mask[i][j] = linds[i] >= rinds[j] -> 4x4 lower triangular.
  let mut plain = create_causal_mask(4, 0, None).unwrap();
  let plain_v = plain.to_vec::<bool>().unwrap();
  assert_eq!(
    plain_v,
    vec![
      true, false, false, false, // q0
      true, true, false, false, // q1
      true, true, true, false, // q2
      true, true, true, true, // q3
    ]
  );

  // total = offset + N = 0 + 4 = 4. window_size == total (4): mlx-lm's
  // `linds < rinds + 4` is `(<=3) < (>=0)+4` -> always true -> no-op.
  // Must equal the unwindowed causal mask exactly.
  let mut w_eq = create_causal_mask(4, 0, Some(4)).unwrap();
  assert_eq!(w_eq.shape(), vec![4, 4]);
  assert_eq!(
    w_eq.to_vec::<bool>().unwrap(),
    plain_v,
    "window_size == total must be the unwindowed causal mask (mlx-lm no-op)"
  );

  // window_size = usize::MAX (>> total, and >> i32::MAX so the old
  // `as i32` cast would wrap): still exactly the unwindowed causal mask.
  let mut w_max = create_causal_mask(4, 0, Some(usize::MAX)).unwrap();
  assert_eq!(w_max.shape(), vec![4, 4]);
  assert_eq!(
    w_max.to_vec::<bool>().unwrap(),
    plain_v,
    "window_size = usize::MAX must be the unwindowed causal mask (no lossy i32 wrap)"
  );

  // With a non-zero offset too: total = offset + N = 3 + 2 = 5.
  // window_size == total (5) is the no-op; equals create_causal_mask(2,3,None).
  let mut po = create_causal_mask(2, 3, None).unwrap();
  let mut pw = create_causal_mask(2, 3, Some(5)).unwrap();
  assert_eq!(pw.shape(), vec![2, 5]);
  assert_eq!(
    pw.to_vec::<bool>().unwrap(),
    po.to_vec::<bool>().unwrap(),
    "offset!=0: window_size == offset+N must equal the unwindowed causal mask"
  );

  // Sanity: a window_size STRICTLY < total still windows (regression guard
  // that the fix did not disable real windowing). N=4, offset=0, w=2:
  // base.py keeps j where i>=j AND i<j+2 -> the existing windowed grid.
  let mut w_small = create_causal_mask(4, 0, Some(2)).unwrap();
  assert_eq!(
    w_small.to_vec::<bool>().unwrap(),
    vec![
      true, false, false, false, // q0: {0}
      true, true, false, false, // q1: {0,1}
      false, true, true, false, // q2: {1,2}
      false, false, true, true, // q3: {2,3}
    ],
    "window_size < total must still apply the sliding window"
  );
}

// ── Copilot finding 3: RotatingKvCache::make_mask N>1 offset+N overflow ──

/// `RotatingKvCache::make_mask`'s `N > 1` branch (cache.py:557-563) computes
/// `offset + N` with `checked_add` (matching the round-2
/// `create_causal_mask` fix) BEFORE the `offset + N > window_size` decision.
/// A hostile/corrupt loaded `max_size`/`offset` (mlx-lm prompt cache
/// `set_meta_state`) near `usize::MAX` would otherwise overflow usize here —
/// a debug panic, or a release wrap that flips the `Causal`-vs-array
/// decision — BEFORE `create_causal_mask`'s own checked-add can catch it.
/// The overflow must be a recoverable `Err` (no panic, no wrong decision).
/// This is faithful to mlx-lm (Python's unbounded ints never overflow);
/// the decision OUTCOME is byte-identical to before for every valid
/// (non-overflowing) input — only the overflow edge becomes `Err`.
///
/// Derivation from `cache.py:554-563` for this input (N=2, window_size=None,
/// loaded `max_size = usize::MAX`, `offset = usize::MAX`):
///   line 558: `window_size = window_size or self.max_size` -> `max_size`
///   line 559: `offset = min(self.max_size - 1, self.offset)`
///             = `min(usize::MAX - 1, usize::MAX)` = `usize::MAX - 1`
///   line 560: `offset + N` = `(usize::MAX - 1) + 2` -> overflows usize
/// so the checked-add yields `Err::ShapeMismatch` instead of wrapping into
/// a wrong `create_causal_mask`/`"causal"` choice. A non-empty minimal valid
/// state keeps `is_empty()==false` so `from_state`'s empty-state invariant
/// is not tripped (that path is unrelated and stays intact).
#[test]
fn rotating_make_mask_n_gt_1_offset_plus_n_overflow_is_err_not_panic() {
  // Non-empty minimal valid state (equal-length K/V, seq len 1) so the
  // cache is NOT empty -> from_state's empty-state invariant is bypassed
  // and `set_meta_state` restores the hostile max_size/offset afterwards.
  let c = from_state(
    "RotatingKVCache",
    vec![kv(&[0.0]), kv(&[0.0])],
    &[
      "0".to_string(),        // keep
      usize::MAX.to_string(), // max_size (hostile)
      usize::MAX.to_string(), // offset   (hostile)
      "0".to_string(),        // idx
    ],
  )
  .unwrap();

  // N = 2 > 1 (cache.py:557 branch). window_size = None -> ws = max_size
  // (cache.py:558). offset = min(max_size-1, offset) = usize::MAX - 1
  // (cache.py:559). offset + N = (usize::MAX - 1) + 2 -> overflow ->
  // checked-add Err (NOT a debug panic, NOT a release wrap that would flip
  // the cache.py:560 decision to a wrong "causal"/array choice).
  let r = c.make_mask(2, None, false);
  assert!(
    matches!(r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "RotatingKvCache::make_mask N>1 offset+N overflow must be Err::ShapeMismatch \
     (no panic, no wrap-then-wrong-Causal-decision)"
  );

  // Same overflow with return_array=true: cache.py:560 is
  // `if offset + N > window_size or return_array` — `offset + N` is still
  // evaluated first, so the checked-add Err still surfaces (the `or
  // return_array` short-circuit does not skip the overflowing sum).
  let r_arr = c.make_mask(2, None, true);
  assert!(
    matches!(r_arr, Err(mlxrs::Error::ShapeMismatch { .. })),
    "RotatingKvCache::make_mask N>1 offset+N overflow must be Err even with return_array=true"
  );

  // Regression: a VALID (non-overflowing) N>1 input still produces the
  // unchanged cache.py:557-563 decision. RotatingKvCache::new(8,4), no
  // tokens: offset=0, max_size=8. make_mask(3, None, false):
  //   line 558: ws = max_size = 8
  //   line 559: offset = min(8-1, 0) = 0
  //   line 560: offset + N = 0 + 3 = 3; 3 > 8? no; return_array? no
  //   line 563: -> "causal" (MaskMode::Causal). Byte-identical to before.
  let valid = RotatingKvCache::new(8, 4);
  assert!(
    matches!(valid.make_mask(3, None, false).unwrap(), MaskMode::Causal),
    "valid (non-overflowing) N>1 decision must be unchanged (cache.py:560-563 -> \"causal\")"
  );
}

// ── Copilot FINDING 2: window_size `or self.max_size` Python truthiness ───

/// `RotatingKvCache::make_mask`'s `N > 1` branch ports
/// `window_size = window_size or self.max_size` (cache.py:558). Python `or`
/// is TRUTHINESS, not None-coalescing: `0` is falsy, so `Some(0)` MUST fall
/// back to `self.max_size` exactly like `None` does — a plain `unwrap_or`
/// would keep `0` and produce a wrong (all-windowed/empty) N>1 mask. This
/// asserts `Some(0)` ≡ `None` ≡ `Some(max_size)` for the N>1 branch on BOTH
/// of its decision outcomes (the symbolic `"causal"` arm and the windowed
/// `create_causal_mask` array arm), every expected value hand-derived from
/// `cache.py:557-563`. mlx-lm only ever feeds a real `window_size` (`None`
/// or a positive sliding window), so this is behaviorally identical for
/// valid inputs; it only fixes the `Some(0)` edge to match Python `or`.
#[test]
fn rotating_make_mask_n_gt_1_window_size_zero_is_max_size() {
  // ── Outcome A: small offset -> symbolic "causal" (cache.py:563) ──
  // RotatingKvCache::new(8, 4); 2 single-token updates -> offset=2, idx=2
  // (cache.py `_update_in_place`: linear fill below max_size, no rotate).
  let mut c = RotatingKvCache::new(8, 4);
  for id in 0..2 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 2);

  // cache.py:558 `ws = window_size or self.max_size`:
  //   None      -> ws = 8
  //   Some(0)   -> 0 is falsy -> ws = max_size = 8  (THE FIX)
  //   Some(8)   -> ws = 8     (8 == max_size)
  // cache.py:559 offset = min(max_size-1=7, self.offset=2) = 2
  // cache.py:560 offset+N = 2+3 = 5 > ws(8)? no; return_array false
  // cache.py:563 -> "causal" (MaskMode::Causal) for ALL THREE.
  assert!(
    matches!(c.make_mask(3, None, false).unwrap(), MaskMode::Causal),
    "N>1 small-offset, window_size None -> \"causal\" (cache.py:563)"
  );
  assert!(
    matches!(c.make_mask(3, Some(0), false).unwrap(), MaskMode::Causal),
    "N>1 small-offset, window_size Some(0) must match None (0 is falsy -> max_size, cache.py:558)"
  );
  assert!(
    matches!(c.make_mask(3, Some(8), false).unwrap(), MaskMode::Causal),
    "N>1 small-offset, window_size Some(8==max_size) -> \"causal\" (cache.py:563)"
  );

  // ── Outcome B: large offset -> windowed array (cache.py:561) ──
  // RotatingKvCache::new(8, 4); 10 single-token updates -> offset=10.
  // This is the outcome the bug actually corrupts: with `ws = 0`,
  // cache.py:560 `offset+N > 0` is ALWAYS true (forced array) AND
  // cache.py:561 `create_causal_mask(N, offset, window_size=0)` is a wrong
  // degenerate (all-windowed) mask. The fix makes `Some(0)` -> ws=8, so the
  // array is byte-identical to `None`/`Some(8)`. Compare via `to_vec` of
  // the Array arm (MaskMode has no PartialEq/Debug per #31 convention).
  let mut c = RotatingKvCache::new(8, 4);
  for id in 0..10 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 10);

  // cache.py:558 ws: None->8, Some(0)->8 (falsy->max_size), Some(8)->8.
  // cache.py:559 offset = min(7, 10) = 7.
  // cache.py:560 offset+N = 7+3 = 10 > ws(8)? yes -> cache.py:561
  // create_causal_mask(N=3, offset=7, window_size=8); base.py shape
  // [N, offset+N] = [3, 10]. Identical Array for all three windows.
  let mask_vec = |m: MaskMode| -> (Vec<usize>, Vec<bool>) {
    match m {
      MaskMode::Array(mut a) => (a.shape(), a.to_vec::<bool>().unwrap()),
      _ => panic!("large-offset N>1 must be a windowed Array (cache.py:561)"),
    }
  };
  let (sh_none, v_none) = mask_vec(c.make_mask(3, None, false).unwrap());
  let (sh_zero, v_zero) = mask_vec(c.make_mask(3, Some(0), false).unwrap());
  let (sh_max, v_max) = mask_vec(c.make_mask(3, Some(8), false).unwrap());
  assert_eq!(sh_none, vec![3, 10], "cache.py:561 shape [N, offset+N]");
  assert_eq!(
    (sh_zero, v_zero),
    (sh_none.clone(), v_none.clone()),
    "Some(0) must yield the SAME windowed mask as None (cache.py:558 `or max_size`)"
  );
  assert_eq!(
    (sh_max, v_max),
    (sh_none, v_none),
    "Some(max_size) must yield the SAME windowed mask as None (cache.py:558)"
  );
}

// ── Copilot FINDING 1: rope_offset default dispatch is non-batch-preserving

/// `KvCache::rope_offset`'s default now auto-dispatches through
/// [`KvCache::as_batch_positioned`] (mirroring mlx-swift-lm's
/// `BatchPositionedKVCache.ropeOffset` protocol-extension, which Rust cannot
/// express as automatic conformance). This is a regression guard that the
/// new dispatch default is BEHAVIOR-PRESERVING for every current (non-batch)
/// cache: neither `StandardKvCache` nor `RotatingKvCache` implements
/// `as_batch_positioned` (its default returns `None`), so `rope_offset` must
/// still be exactly `RopeOffset::Scalar(self.offset())` after updates —
/// identical to the pre-dispatch `RopeOffset::Scalar(self.offset())` body
/// (zero behavior change today; the #31/parity values are unaffected).
#[test]
fn rope_offset_default_is_scalar_for_non_batch_caches() {
  // StandardKvCache: not batch-positioned -> Scalar(offset) after updates.
  let mut s = StandardKvCache::new();
  assert!(
    s.as_batch_positioned().is_none(),
    "StandardKvCache must not be a batch-positioned refinement"
  );
  s.update(&kv(&[0.0, 1.0, 2.0]), &kv(&[0.0, 1.0, 2.0]))
    .unwrap();
  match s.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 3, "Standard rope_offset == offset (3)"),
    RopeOffset::Batch(_) => {
      panic!("non-batch Standard cache must still yield RopeOffset::Scalar")
    }
  }
  // After a further update the scalar still tracks offset (dispatch default
  // is a pure forward to offset() for non-batch caches).
  s.update(&kv(&[3.0, 4.0]), &kv(&[3.0, 4.0])).unwrap();
  match s.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 5, "Standard rope_offset tracks offset (5)"),
    RopeOffset::Batch(_) => panic!("non-batch Standard cache must still yield RopeOffset::Scalar"),
  }

  // RotatingKvCache: also not batch-positioned -> Scalar(offset).
  let mut r = RotatingKvCache::new(8, 4);
  assert!(
    r.as_batch_positioned().is_none(),
    "RotatingKvCache must not be a batch-positioned refinement"
  );
  for id in 0..5 {
    let t = kv(&[id as f32]);
    r.update(&t, &t).unwrap();
  }
  match r.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 5, "Rotating rope_offset == raw offset (5)"),
    RopeOffset::Batch(_) => {
      panic!("non-batch Rotating cache must still yield RopeOffset::Scalar")
    }
  }
}
