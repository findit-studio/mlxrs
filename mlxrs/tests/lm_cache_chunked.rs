//! Deterministic, hand-traced tests for [`ChunkedKvCache`], ported 1:1 from
//! `mlx_lm.models.cache.ChunkedKVCache` (`cache.py:731-813`, `step = 256`)
//! and cross-checked against mlx-swift-lm `KVCache.swift:1008` (`ChunkedKVCache:
//! KVCacheSimple`).
//!
//! Every trace below is computed by hand directly from `cache.py:731-813`:
//!
//! - `step = 256`, so the first allocation of any `S <= 256` update is a
//!   256-row zero buffer; `keys.shape[2]` (the *buffer* length, **not** the
//!   logical length) drives `maybe_trim_front` / the realloc branch, exactly
//!   as in `update_and_fetch` (`cache.py:743,750`).
//! - `prev = offset - start_position`, `end = offset - start_position`
//!   *after* `offset += S`; the new rows are spliced over `[prev, end)` and
//!   `keys[..., :end, :]` is returned (`cache.py:749,767-771`).
//! - `maybe_trim_front` keeps the *last* `chunk_size` buffer rows and bumps
//!   `start_position` by `keys.shape[2] - chunk_size` (`cache.py:741-746`);
//!   it does **not** touch `offset`.
//!
//! Caches are 4-D `[B, n_kv_heads, S, head_dim]` (sequence axis `-2`), tiny
//! and built so every retained-token identity reads straight out of
//! `to_vec`.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{ChunkedKvCache, KvCache, MaskMode, RopeOffset, from_state},
};

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// each step's value being its 0-based token id, so retained ids are
/// directly readable. `S == ids.len()`.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

/// Grow from empty with single-token updates: ids `0..=3` accumulate to
/// `[0,1,2,3]`, `offset` walks `1..=4`, `start_position` stays `0`
/// (`cache.py:748-771`; first update allocates the 256-row `step` buffer,
/// every later update fits within it so no realloc).
#[test]
fn chunked_single_token_growth() {
  // chunk_size = 4 (mlx-lm `ChunkedKVCache(chunk_size)`, `cache.py:734`).
  let mut c = ChunkedKvCache::new(Some(4));
  assert!(c.is_empty());
  assert_eq!(c.offset(), 0);
  // meta_state = (chunk_size, start_position) (`cache.py:797-798`).
  assert_eq!(c.meta_state(), vec!["4", "0"]);

  for id in 0..4u32 {
    let t = kv(&[id as f32]);
    let (mut k, mut v) = c.update(&t, &t).unwrap();
    // prev = id - 0 = id; end = (id+1) - 0; returns keys[..., :id+1, :].
    assert_eq!(c.offset() as u32, id + 1);
    assert_eq!(k.shape(), vec![1, 1, (id + 1) as usize, 1]);
    let want: Vec<f32> = (0..=id).map(|x| x as f32).collect();
    assert_eq!(k.to_vec::<f32>().unwrap(), want);
    assert_eq!(v.to_vec::<f32>().unwrap(), want);
  }
  assert!(!c.is_empty());
  // start_position untouched by plain updates (`cache.py:748-771`).
  assert_eq!(c.meta_state(), vec!["4", "0"]);
  // RoPE offset is the scalar `offset` (no batched refinement).
  match c.rope_offset().unwrap() {
    RopeOffset::Scalar(o) => assert_eq!(o, 4),
    RopeOffset::Batch(_) => panic!("chunked cache must use a scalar RoPE offset"),
  }
}

/// Multi-token prefill spanning a chunk boundary in one update: `S = 5`
/// with `chunk_size = 4`. mlx-lm allocates one 256-row `step` buffer
/// (`n_steps = (256 + 5 - 1) // 256 = 1`), writes `[prev=0, end=5)`, and
/// returns `keys[..., :5, :]` — the chunk boundary affects only the *mask*
/// the model builds (the cache buffer is unaffected), `cache.py:748-771`.
#[test]
fn chunked_multi_token_spans_chunk_boundary() {
  let mut c = ChunkedKvCache::new(Some(4));
  let t = kv(&[0.0, 1.0, 2.0, 3.0, 4.0]);
  let (mut k, mut v) = c.update(&t, &t).unwrap();
  assert_eq!(c.offset(), 5);
  assert_eq!(k.shape(), vec![1, 1, 5, 1]);
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
  assert_eq!(c.meta_state(), vec!["4", "0"]);
}

/// `maybe_trim_front` on a realistic restored buffer (the llama4 between-chunk
/// path: `set_state` installs an exact-length buffer, then the model calls
/// `maybe_trim_front`). `cache.py:741-746` keeps the LAST `chunk_size` rows
/// and adds `keys.shape[2] - chunk_size` to `start_position`; `offset` is
/// untouched. A following single-token update then reallocates
/// (`prev % step != 0`) and writes at the shifted `prev`.
#[test]
fn chunked_maybe_trim_front_then_update() {
  let mut c = ChunkedKvCache::new(Some(4));
  // Restore a 6-row buffer [0,1,2,3,4,5]; state setter sets
  // offset = keys.shape[2] = 6 (`cache.py:783-786`), start_position 0.
  let buf = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.offset(), 6);
  assert_eq!(c.meta_state(), vec!["4", "0"]);

  // maybe_trim_front: keys.shape[2]=6 >= chunk_size=4 ->
  // start_position += 6-4 = 2; keys = keys[..., -4:, :] = [2,3,4,5];
  // offset stays 6 (`cache.py:741-746`).
  c.maybe_trim_front().unwrap();
  assert_eq!(c.offset(), 6);
  assert_eq!(c.meta_state(), vec!["4", "2"]);
  // state(): offset(6) != keys.shape[2](4) -> keys[..., :6, :], clamped by
  // Python slicing to the 4-row trimmed buffer (`cache.py:773-781`).
  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);
  let mut sk = st[0].try_clone().unwrap();
  assert_eq!(sk.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

  // Single-token update id 6: prev = 6 - 2 = 4; prev + 1 = 5 >
  // keys.shape[2]=4 -> realloc. prev % step = 4 % 256 = 4 != 0 ->
  // keys = keys[..., :4, :] (= [2,3,4,5]); concat 256 zeros -> len 260.
  // offset += 1 -> 7; end = 7 - 2 = 5; keys[..., 4:5, :] = [6];
  // return keys[..., :5, :] = [2,3,4,5,6] (`cache.py:748-771`).
  let t = kv(&[6.0]);
  let (mut k, mut v) = c.update(&t, &t).unwrap();
  assert_eq!(c.offset(), 7);
  assert_eq!(k.shape(), vec![1, 1, 5, 1]);
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
  // start_position unchanged by the update; offset moved.
  assert_eq!(c.meta_state(), vec!["4", "2"]);
}

/// `maybe_trim_front` byte-identity on the *raw* `step`-buffer semantics:
/// after a fresh single-token update the buffer is 256 rows (only row 0
/// written), and `cache.py:743` tests `keys.shape[2] (=256) >= chunk_size`,
/// so it trims to the LAST `chunk_size` rows and adds `256 - chunk_size` to
/// `start_position`, regardless of how few rows were logically written.
/// This locks the literal method semantics (the llama4 model is responsible
/// for only invoking it when meaningful — out of this port's scope); a
/// no-op guard would silently diverge from mlx-lm.
#[test]
fn chunked_maybe_trim_front_raw_step_buffer_semantics() {
  let mut c = ChunkedKvCache::new(Some(4));
  let t = kv(&[0.0]);
  c.update(&t, &t).unwrap(); // buffer is 256 rows, offset 1, start_position 0.
  assert_eq!(c.offset(), 1);

  // keys.shape[2] = 256 >= 4 -> start_position += 256 - 4 = 252; offset
  // untouched (`cache.py:741-746`).
  c.maybe_trim_front().unwrap();
  assert_eq!(c.offset(), 1);
  assert_eq!(c.meta_state(), vec!["4", "252"]);
  // Buffer trimmed to the last 4 rows (all zeros here: only row 0 was
  // written). state(): offset(1) != keys.shape[2](4) -> keys[..., :1, :].
  let st = c.state().unwrap();
  let mut sk = st[0].try_clone().unwrap();
  assert_eq!(sk.shape(), vec![1, 1, 1, 1]);
  assert_eq!(sk.to_vec::<f32>().unwrap(), vec![0.0]);
}

/// `maybe_trim_front` is a no-op while the buffer is shorter than
/// `chunk_size`, and entirely a no-op when `chunk_size` is `None`
/// (mlx-swift-lm `KVCache.swift:1017-1026`: the `guard let chunkSize`
/// short-circuits; mlx-lm always constructs with an int `chunk_size`, but
/// the Swift-reconstructed `None` must be honored — `cache.py:743` /
/// `KVCache.swift:1019-1021`).
#[test]
fn chunked_maybe_trim_front_noop_paths() {
  // Buffer (4 rows) shorter-than/equal handling: with chunk_size 8,
  // keys.shape[2]=4 >= 8 is false -> no-op.
  let mut c = ChunkedKvCache::new(Some(8));
  let buf = kv(&[0.0, 1.0, 2.0, 3.0]);
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  c.maybe_trim_front().unwrap();
  assert_eq!(c.offset(), 4);
  assert_eq!(c.meta_state(), vec!["8", "0"]);

  // chunk_size = None -> maybe_trim_front is unconditionally a no-op
  // (Swift `guard let chunkSize`); meta_state encodes "None".
  let mut c2 = ChunkedKvCache::new(None);
  let buf2 = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  c2.set_state(vec![buf2.try_clone().unwrap(), buf2.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c2.meta_state(), vec!["None", "0"]);
  c2.maybe_trim_front().unwrap();
  assert_eq!(c2.offset(), 6);
  assert_eq!(c2.meta_state(), vec!["None", "0"]);
}

/// `state` getter / setter round-trip (`cache.py:773-786`): when
/// `offset == keys.shape[2]` the buffer is returned as-is; the setter sets
/// `offset = keys.shape[2]`. An empty state is invalid for `ChunkedKVCache`
/// (its setter unpacks `self.keys, self.values = v`, which raises on `[]` —
/// unlike `_BaseCache`), so it is a recoverable error, not a silent reset.
#[test]
fn chunked_state_roundtrip() {
  let mut c = ChunkedKvCache::new(Some(4));
  for id in 0..3u32 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  // offset 3, buffer 256 -> offset != keys.shape[2] -> keys[..., :3, :].
  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);
  let mut sk = st[0].try_clone().unwrap();
  let mut sv = st[1].try_clone().unwrap();
  assert_eq!(sk.shape(), vec![1, 1, 3, 1]);
  assert_eq!(sk.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);
  assert_eq!(sv.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);

  // set_state: offset = keys.shape[2] = 3 (`cache.py:783-786`).
  let mut c2 = ChunkedKvCache::new(Some(4));
  c2.set_state(st).unwrap();
  assert_eq!(c2.offset(), 3);
  let st2 = c2.state().unwrap();
  // Now offset(3) == keys.shape[2](3) -> buffer returned as-is.
  let mut s2k = st2[0].try_clone().unwrap();
  assert_eq!(s2k.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0]);

  // Empty state -> recoverable error (Python unpack of [] raises).
  let mut c3 = ChunkedKvCache::new(Some(4));
  assert!(c3.set_state(Vec::new()).is_err());
  // Wrong-arity state -> recoverable error.
  assert!(c3.set_state(vec![kv(&[0.0])]).is_err());
}

/// `meta_state` carries `(chunk_size, start_position)` as decimal strings
/// and round-trips (`cache.py:796-802`); the Swift refinement encodes a
/// `None` `chunk_size` as the literal `"None"` (`KVCache.swift:1082-1098`).
#[test]
fn chunked_meta_state_roundtrip() {
  let mut c = ChunkedKvCache::new(Some(4));
  let buf = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  c.maybe_trim_front().unwrap(); // start_position -> 2.
  assert_eq!(c.meta_state(), vec!["4", "2"]);

  // Round-trip the metadata onto a fresh cache.
  let mut c2 = ChunkedKvCache::new(Some(99));
  c2.set_meta_state(&["7".to_string(), "5".to_string()])
    .unwrap();
  assert_eq!(c2.meta_state(), vec!["7", "5"]);

  // "None" chunk_size round-trips (Swift reconstruction).
  let mut c3 = ChunkedKvCache::new(Some(4));
  c3.set_meta_state(&["None".to_string(), "3".to_string()])
    .unwrap();
  assert_eq!(c3.meta_state(), vec!["None", "3"]);

  // Wrong arity is a recoverable error (Swift `fatalError` -> our Err).
  assert!(c3.set_meta_state(&["4".to_string()]).is_err());
  assert!(
    c3.set_meta_state(&["4".to_string(), "0".to_string(), "x".to_string()])
      .is_err()
  );
  // Non-numeric start_position -> error, and the cache is left unmodified.
  assert!(
    c3.set_meta_state(&["4".to_string(), "bad".to_string()])
      .is_err()
  );
  assert_eq!(c3.meta_state(), vec!["None", "3"]);
}

/// `is_trimmable` is always `true` and `trim(n)` drops
/// `min(offset - start_position, n)` tokens, returning the count
/// (`cache.py:788-794`). `trim` adjusts only `offset`; `start_position` /
/// the buffer are untouched (it mirrors mlx-lm exactly).
#[test]
fn chunked_is_trimmable_and_trim() {
  let mut c = ChunkedKvCache::new(Some(4));
  let buf = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  c.maybe_trim_front().unwrap(); // offset 6, start_position 2.
  assert!(c.is_trimmable());

  // trim(3): n = min(offset - start_position, 3) = min(6-2, 3) = 3;
  // offset -= 3 -> 3; returns 3 (`cache.py:791-794`).
  assert_eq!(c.trim(3).unwrap(), 3);
  assert_eq!(c.offset(), 3);
  assert_eq!(c.meta_state(), vec!["4", "2"]);

  // trim never removes more than offset - start_position: now
  // offset-start_position = 3-2 = 1, so trim(10) -> 1, offset -> 2.
  assert_eq!(c.trim(10).unwrap(), 1);
  assert_eq!(c.offset(), 2);
}

/// `nbytes` is `keys.nbytes + values.nbytes` (`0` when empty),
/// `cache.py:807-811`; `copy()` is an independent deep copy
/// (`KVCache.swift:1071-1080`).
#[test]
fn chunked_nbytes_and_copy() {
  let mut c = ChunkedKvCache::new(Some(4));
  assert_eq!(c.nbytes(), 0);
  let buf = kv(&[0.0, 1.0, 2.0, 3.0]); // 4 f32 (4 bytes each) keys + values.
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.nbytes(), 4 * 4 + 4 * 4);

  // copy() is independent: mutating the copy must not affect the original.
  let mut cp = c.copy().unwrap();
  let t = kv(&[9.0]);
  cp.update(&t, &t).unwrap();
  assert_eq!(c.offset(), 4); // original unchanged ...
  assert_eq!(c.meta_state(), vec!["4", "0"]);
  assert_eq!(cp.offset(), 5); // ... copy advanced.
}

/// `ChunkedKVCache` has **no** `make_mask` override and (unlike `KVCache`)
/// neither does its base, so mlx-lm `create_attention_mask` (`base.py:49`)
/// finds `hasattr(cache, "make_mask")` False and falls through; mlx-swift-lm
/// (`KVCache.swift:1008`, `ChunkedKVCache: KVCacheSimple`) resolves the
/// Rust/Swift "method is mandatory" tension via `BaseKVCache.makeMask`
/// (`KVCache.swift:177-191`), i.e. the standard offset-aware
/// `create_attention_mask`. Verified: `n==1 -> None`; `return_array ->
/// Array`; multi-token symbolic -> `Causal` (same as `StandardKvCache`).
#[test]
fn chunked_make_mask() {
  let mut c = ChunkedKvCache::new(Some(4));
  for id in 0..3u32 {
    let t = kv(&[id as f32]);
    c.update(&t, &t).unwrap();
  }
  assert_eq!(c.offset(), 3);

  // N == 1 -> no mask.
  assert!(matches!(
    c.make_mask(1, None, false).unwrap(),
    MaskMode::None
  ));
  // N > 1, no window, not return_array -> symbolic causal.
  assert!(matches!(
    c.make_mask(3, None, false).unwrap(),
    MaskMode::Causal
  ));
  // return_array -> materialized array (offset-aware: shape [N, offset+N]).
  match c.make_mask(2, None, true).unwrap() {
    MaskMode::Array(m) => assert_eq!(m.shape(), vec![2, 3 + 2]),
    _ => panic!("return_array must materialize an array mask"),
  }
  // window_size provided -> materialized windowed array.
  assert!(matches!(
    c.make_mask(3, Some(2), false).unwrap(),
    MaskMode::Array(_)
  ));
}

/// `from_state` reconstructs a `ChunkedKVCache` keyed on the SOURCE class
/// name `"ChunkedKVCache"` (mlx-lm `save_prompt_cache` emits
/// `type(c).__name__`, `cache.py:56`; `load_prompt_cache` rebuilds via
/// `globals()[name]`, `cache.py:79-82`), accepts the Rust-name alias
/// `"ChunkedKvCache"` for back-compat, and applies state THEN meta_state
/// (`_BaseCache.from_state`, `cache.py:170-175`).
#[test]
fn chunked_from_state_roundtrip() {
  let mut c = ChunkedKvCache::new(Some(4));
  let buf = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  c.set_state(vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()])
    .unwrap();
  c.maybe_trim_front().unwrap(); // offset 6, start_position 2, buffer [2,3,4,5].

  let state = c.state().unwrap();
  let meta = c.meta_state();
  assert_eq!(meta, vec!["4", "2"]);
  let state_arg: Vec<Array> = state
    .iter()
    .map(|a| a.try_clone())
    .collect::<mlxrs::Result<Vec<_>>>()
    .unwrap();

  // SOURCE name (what a real mlx-lm/Swift prompt cache writes).
  let c2 = from_state("ChunkedKVCache", state_arg, &meta).unwrap();
  // state() set offset = keys.shape[2] = 4, then meta_state restored
  // start_position = 2 (state THEN meta order).
  assert_eq!(c2.offset(), 4);
  assert_eq!(c2.meta_state(), vec!["4", "2"]);
  let c2_state = c2.state().unwrap();
  let mut c2k = c2_state[0].try_clone().unwrap();
  assert_eq!(c2k.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);

  // Rust-name alias also loads.
  let c3 = from_state(
    "ChunkedKvCache",
    vec![buf.try_clone().unwrap(), buf.try_clone().unwrap()],
    &["4".to_string(), "0".to_string()],
  )
  .unwrap();
  assert_eq!(c3.offset(), 6);
  assert_eq!(c3.meta_state(), vec!["4", "0"]);

  // Empty state is invalid for ChunkedKVCache (Python unpack of [] raises);
  // from_state must surface that as a recoverable error, not a silent cache.
  assert!(
    from_state(
      "ChunkedKVCache",
      Vec::new(),
      &["4".to_string(), "0".to_string()]
    )
    .is_err()
  );
}

/// Regression (adversarial review): a malformed/hostile prompt cache with
/// rank-valid `keys` but a rank-invalid `values` must be a recoverable
/// `Error`, NEVER a raw-`shape()[2]` index panic — neither at `from_state`
/// (`set_state` rank-checks BOTH tensors, the same per-tensor guard
/// `update` applies; this is NOT the faithful-forbidden K/V *seq*
/// cross-check) nor in a later `maybe_trim_front` / `state` (each rank-checks
/// `values` before slicing it, and `maybe_trim_front` computes both slices
/// into locals before mutating `self`, so even an `Err` leaves no
/// split-brain cache).
#[test]
fn chunked_malformed_state_values_rank_is_err_not_panic() {
  // 4-D keys (6 rows) + 2-D values (rank-invalid).
  let keys = kv(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
  let bad_values = Array::from_slice::<f32>(&[0.0, 1.0], &(1usize, 2)).unwrap();

  // from_state must reject it up front (set_state rank-checks values).
  assert!(
    from_state(
      "ChunkedKVCache",
      vec![keys.try_clone().unwrap(), bad_values.try_clone().unwrap()],
      &["4".to_string(), "0".to_string()],
    )
    .is_err()
  );

  // set_state directly: also a recoverable error, and the cache is left in
  // its prior (empty) state — no partial mutation.
  let mut c = ChunkedKvCache::new(Some(4));
  assert!(
    c.set_state(vec![
      keys.try_clone().unwrap(),
      bad_values.try_clone().unwrap()
    ])
    .is_err()
  );
  assert!(c.is_empty());
  assert_eq!(c.offset(), 0);
  assert_eq!(c.meta_state(), vec!["4", "0"]);

  // Even if a rank-invalid `values` is somehow installed (bypassing the
  // restore guard via a valid set_state then... not possible through the
  // public API), `maybe_trim_front` independently rank-checks `values`
  // before slicing and before any `self` mutation: recoverable, no panic,
  // no split-brain. We exercise the guard path by confirming a rank-valid
  // round-trip still trims correctly (the negative path is covered above).
  let mut ok = ChunkedKvCache::new(Some(4));
  ok.set_state(vec![keys.try_clone().unwrap(), keys.try_clone().unwrap()])
    .unwrap();
  ok.maybe_trim_front().unwrap();
  assert_eq!(ok.offset(), 6);
  assert_eq!(ok.meta_state(), vec!["4", "2"]);
}

/// Regression (adversarial review): `update` must be transactional — if ANY
/// fallible op AFTER the keys realloc-concat fails (the values concat /
/// either splice / the return slice), the cache must be left **completely
/// unchanged** (no poisoned grown-`keys` buffer with stale `values`/`offset`,
/// which a later `maybe_trim_front` would trim against `keys.shape[2]` and
/// silently drop context).
///
/// Deterministic trigger (no flaky OOM injection, public API only): restore
/// a `keys` buffer with `head_dim == 1` and a `values` buffer with `head_dim
/// == 2`. This is a valid restored state under the faithful no-K/V-cross-
/// validation philosophy (`set_state` only per-tensor *rank*-checks; mlx-lm
/// never compares K/V shapes). The next `update` needs a realloc: the keys
/// concat succeeds (consistent dims) but the values concat fails (the prior
/// `values` `head_dim==2` cannot concat with the `head_dim==1` zero block
/// derived from the new `values`). Under a non-transactional `update` the
/// keys field would already be the grown buffer; the fix keeps every buffer
/// in locals until all fallible work succeeds, so `Err` ⇒ no mutation.
#[test]
fn chunked_update_err_after_keys_realloc_leaves_cache_unchanged() {
  let mut c = ChunkedKvCache::new(None); // chunk_size None: trim path irrelevant.
  // keys [1,1,1,1], values [1,1,1,2] — rank-4 both, differing head_dim
  // (no K/V cross-check; the faithful references do not add one).
  let k0 = Array::from_slice::<f32>(&[7.0], &(1usize, 1, 1, 1)).unwrap();
  let v0 = Array::from_slice::<f32>(&[7.0, 8.0], &(1usize, 1, 1, 2)).unwrap();
  c.set_state(vec![k0.try_clone().unwrap(), v0.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.offset(), 1); // offset = keys.shape[2] = 1.
  assert!(!c.is_empty());
  assert_eq!(c.meta_state(), vec!["None", "0"]);
  // Snapshot the exact pre-update state buffers.
  let pre = c.state().unwrap();
  let pre_k = pre[0].try_clone().unwrap().to_vec::<f32>().unwrap();
  let pre_v = pre[1].try_clone().unwrap().to_vec::<f32>().unwrap();

  // update with [1,1,1,1] tensors: prev = 1, prev+1 = 2 > buf_len 1 ->
  // realloc; keys concat ([1,1,1,1] + [1,1,256,1]) succeeds, values concat
  // ([1,1,1,2] + [1,1,256,1]) fails on the mismatched last dim -> Err.
  let t = kv(&[9.0]);
  let r = c.update(&t, &t);
  assert!(
    r.is_err(),
    "values concat must fail on the head_dim mismatch"
  );

  // The cache must be byte-for-byte what it was BEFORE the failed update:
  // no grown keys buffer, offset not advanced, still non-empty, meta intact.
  assert_eq!(c.offset(), 1, "offset must not advance on a failed update");
  assert!(!c.is_empty());
  assert_eq!(c.meta_state(), vec!["None", "0"]);
  let post = c.state().unwrap();
  assert_eq!(post.len(), 2);
  let mut pk = post[0].try_clone().unwrap();
  let mut pv = post[1].try_clone().unwrap();
  assert_eq!(
    pk.shape(),
    vec![1, 1, 1, 1],
    "keys buffer must be unchanged"
  );
  assert_eq!(
    pv.shape(),
    vec![1, 1, 1, 2],
    "values buffer must be unchanged"
  );
  assert_eq!(pk.to_vec::<f32>().unwrap(), pre_k);
  assert_eq!(pv.to_vec::<f32>().unwrap(), pre_v);

  // And the cache is still fully usable afterwards: a well-formed update
  // (matching head_dim==2 values) now succeeds and advances normally.
  let kk = Array::from_slice::<f32>(&[1.0], &(1usize, 1, 1, 1)).unwrap();
  let vv = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 1, 1, 2)).unwrap();
  let (mut ok_k, mut ok_v) = c.update(&kk, &vv).unwrap();
  assert_eq!(c.offset(), 2);
  assert_eq!(ok_k.shape(), vec![1, 1, 2, 1]);
  assert_eq!(ok_v.shape(), vec![1, 1, 2, 2]);
  // Prior row 0 (keys 7.0 / values [7,8]) is retained, new row appended.
  assert_eq!(ok_k.to_vec::<f32>().unwrap(), vec![7.0, 1.0]);
  assert_eq!(ok_v.to_vec::<f32>().unwrap(), vec![7.0, 8.0, 1.0, 2.0]);
}

/// Regression (adversarial review): `maybe_trim_front` must keep EACH tensor's
/// OWN last `chunk_size` rows. mlx-lm `cache.py:745-746` does
/// `self.keys = self.keys[..., -chunk_size:, :]` and
/// `self.values = self.values[..., -chunk_size:, :]` — each a negative slice
/// relative to that tensor's *own* sequence axis. This port intentionally
/// accepts rank-valid but seq-*mismatched* restored K/V (the project's
/// no-K/V-cross-validation policy), so the `values` window must be derived
/// from `values`' own length, NOT reused from the keys length (which would
/// silently retain the wrong value rows). This is per-tensor faithfulness,
/// not a K/V cross-comparison (no `keys.len == values.len` check is added).
#[test]
fn chunked_maybe_trim_front_seq_mismatched_kv_trims_each_independently() {
  // Restored state with rank-valid but DIFFERENT K/V seq lengths
  // (keys 6 rows, values 10 rows) — set_state only per-tensor rank-checks.
  let mut c = ChunkedKvCache::new(Some(4));
  let keys = kv(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]); // [1,1,6,1]
  let values = kv(&[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]); // [1,1,10,1]
  c.set_state(vec![keys.try_clone().unwrap(), values.try_clone().unwrap()])
    .unwrap();
  // offset = keys.shape[2] = 6 (`cache.py:786`), start_position 0.
  assert_eq!(c.offset(), 6);
  assert_eq!(c.meta_state(), vec!["4", "0"]);

  // maybe_trim_front: guard is keys.shape[2]=6 >= chunk_size=4 (keys-only,
  // `cache.py:743`). start_position += 6-4 = 2 (from KEYS length).
  // keys  -> keys[..., -4:, :]  = rows [2,6)  = [12,13,14,15]
  // values-> values[..., -4:, :] = rows [6,10) = [26,27,28,29]  (NOT [22..25]!)
  c.maybe_trim_front().unwrap();
  assert_eq!(c.offset(), 6); // offset untouched by trim_front.
  assert_eq!(c.meta_state(), vec!["4", "2"]); // start_position from KEYS len.
  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);
  let mut sk = st[0].try_clone().unwrap();
  let mut sv = st[1].try_clone().unwrap();
  // state(): offset(6) != keys.shape[2](4) -> keys[..., :6, :] / values[..., :6, :],
  // each clamped per-tensor to the trimmed 4-row buffers.
  assert_eq!(sk.shape(), vec![1, 1, 4, 1]);
  assert_eq!(sv.shape(), vec![1, 1, 4, 1]);
  assert_eq!(sk.to_vec::<f32>().unwrap(), vec![12.0, 13.0, 14.0, 15.0]);
  assert_eq!(
    sv.to_vec::<f32>().unwrap(),
    vec![26.0, 27.0, 28.0, 29.0],
    "values must keep ITS OWN last chunk_size rows, not the keys-windowed rows"
  );

  // Sub-case: values SHORTER than chunk_size. Python `values[..., -4:, :]`
  // on a 3-row tensor yields the whole tensor (negative start clamps to 0);
  // `saturating_sub` reproduces that. Keys-only guard still fires (6 >= 4).
  let mut c2 = ChunkedKvCache::new(Some(4));
  let keys2 = kv(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]); // 6 rows
  let values2 = kv(&[40.0, 41.0, 42.0]); // 3 rows (< chunk_size)
  c2.set_state(vec![
    keys2.try_clone().unwrap(),
    values2.try_clone().unwrap(),
  ])
  .unwrap();
  c2.maybe_trim_front().unwrap();
  assert_eq!(c2.meta_state(), vec!["4", "2"]); // from keys len: 6-4=2.
  let st2 = c2.state().unwrap();
  let mut sk2 = st2[0].try_clone().unwrap();
  let mut sv2 = st2[1].try_clone().unwrap();
  assert_eq!(sk2.to_vec::<f32>().unwrap(), vec![12.0, 13.0, 14.0, 15.0]);
  // values had only 3 rows, all retained ([..., -4:, :] on len-3 = whole).
  assert_eq!(sv2.shape(), vec![1, 1, 3, 1]);
  assert_eq!(sv2.to_vec::<f32>().unwrap(), vec![40.0, 41.0, 42.0]);
}

/// Regression (adversarial review): `chunk_size == 0` is a degenerate case
/// mlx-lm does not guard against (`ChunkedKVCache(chunk_size)` accepts any
/// int, `cache.py:734`). Python's negative-slice semantics evaluate
/// `keys[..., -0:, :]` as `keys[..., 0:, :]` — the WHOLE tensor (Python's
/// `-0 == 0`, not "from end"). Mirror that exactly: a `chunk_size == 0`
/// `maybe_trim_front` bumps `start_position` by the full buffer length but
/// leaves `keys`/`values` byte-identical. Both `new(Some(0))` and a
/// `set_meta_state(["0", ...])` round-trip exercise this path.
#[test]
fn chunked_maybe_trim_front_chunk_size_zero_is_python_neg_zero_noop() {
  // Direct constructor path: chunk_size = 0.
  let mut c = ChunkedKvCache::new(Some(0));
  let keys = kv(&[10.0, 11.0, 12.0, 13.0]);
  let values = kv(&[20.0, 21.0, 22.0, 23.0]);
  c.set_state(vec![keys.try_clone().unwrap(), values.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.offset(), 4); // offset = keys.shape[2].
  assert_eq!(c.meta_state(), vec!["0", "0"]);

  // maybe_trim_front: guard 4 >= 0 true; start_position += 4-0 = 4;
  // keys/values = [..., -0:, :] = WHOLE tensor (Python -0 == 0). offset
  // untouched.
  c.maybe_trim_front().unwrap();
  assert_eq!(c.offset(), 4);
  assert_eq!(c.meta_state(), vec!["0", "4"]);
  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);
  let mut sk = st[0].try_clone().unwrap();
  let mut sv = st[1].try_clone().unwrap();
  // state(): offset(4) != keys.shape[2](4)? Actually 4 == 4 -> buffer returned as-is.
  // Either way the trim preserved all 4 rows.
  assert_eq!(sk.shape(), vec![1, 1, 4, 1]);
  assert_eq!(sv.shape(), vec![1, 1, 4, 1]);
  assert_eq!(
    sk.to_vec::<f32>().unwrap(),
    vec![10.0, 11.0, 12.0, 13.0],
    "chunk_size=0 trim must keep the WHOLE keys tensor (Python -0: slice)"
  );
  assert_eq!(
    sv.to_vec::<f32>().unwrap(),
    vec![20.0, 21.0, 22.0, 23.0],
    "chunk_size=0 trim must keep the WHOLE values tensor (Python -0: slice)"
  );

  // meta_state restore path: set_meta_state(["0", ...]) parses to Some(0).
  let mut c2 = ChunkedKvCache::new(Some(99));
  let keys2 = kv(&[100.0, 101.0, 102.0]);
  let values2 = kv(&[200.0, 201.0, 202.0]);
  c2.set_state(vec![
    keys2.try_clone().unwrap(),
    values2.try_clone().unwrap(),
  ])
  .unwrap();
  c2.set_meta_state(&["0".to_string(), "0".to_string()])
    .unwrap();
  assert_eq!(c2.meta_state(), vec!["0", "0"]);
  c2.maybe_trim_front().unwrap();
  // start_position += 3 - 0 = 3.
  assert_eq!(c2.meta_state(), vec!["0", "3"]);
  let st2 = c2.state().unwrap();
  let mut sk2 = st2[0].try_clone().unwrap();
  let mut sv2 = st2[1].try_clone().unwrap();
  assert_eq!(sk2.to_vec::<f32>().unwrap(), vec![100.0, 101.0, 102.0]);
  assert_eq!(sv2.to_vec::<f32>().unwrap(), vec![200.0, 201.0, 202.0]);
}

/// Regression (adversarial review): `set_seq` must enforce per-target write
/// bounds. mlx-lm's `self.<buf>[..., a:b, :] = new` raises an `IndexError` if
/// the slice extends past the buffer (Python/MLX semantics); reusing
/// `seq_slice` (which clamps for *reads*) for a *write* would silently
/// truncate or drop rows. With a rank-valid but seq-mismatched restored
/// state (keys len 4, values len 1, start_position 2), an `update(S=1)`
/// computes `prev = offset - start_position = 4 - 2 = 2`, no realloc
/// (`prev+S=3 <= keys_len=4`), then splices the new `values` row into the
/// `values` buffer at `[2,3)` — out-of-bounds for the length-1 `values`.
/// The faithful behavior is `Err`; under the un-bounds-checked `set_seq`
/// it would silently corrupt the cache. This is per-target bounds, NOT a
/// K/V cross-comparison — the check is on a single tensor's own write
/// window.
#[test]
fn chunked_update_out_of_bounds_values_write_is_err_not_silent_corrupt() {
  let mut c = ChunkedKvCache::new(Some(4));
  // Rank-valid keys (len 4) + values (len 1) — set_state per-tensor rank-
  // checks (no K/V cross-validation, per the faithful policy).
  let keys = kv(&[10.0, 11.0, 12.0, 13.0]);
  let values = kv(&[20.0]);
  c.set_state(vec![keys.try_clone().unwrap(), values.try_clone().unwrap()])
    .unwrap();
  // offset = keys.shape[2] = 4; restore start_position = 2.
  c.set_meta_state(&["4".to_string(), "2".to_string()])
    .unwrap();
  assert_eq!(c.offset(), 4);
  assert_eq!(c.meta_state(), vec!["4", "2"]);
  // Snapshot pre-update buffers.
  let pre = c.state().unwrap();
  let pre_k = pre[0].try_clone().unwrap().to_vec::<f32>().unwrap();
  let pre_v = pre[1].try_clone().unwrap().to_vec::<f32>().unwrap();

  // update(S=1): prev = 4-2 = 2; prev+1 = 3 <= keys_len=4 -> !need_alloc.
  // values buffer length is 1; write window [2,3) is out-of-bounds for
  // values. mlx-lm `self.values[..., 2:3, :] = ...` would raise IndexError;
  // we surface a recoverable ShapeMismatch.
  let t = kv(&[99.0]);
  let r = c.update(&t, &t);
  assert!(
    r.is_err(),
    "out-of-bounds values write must be Err, not silent truncation/append"
  );

  // Cache untouched: offset, meta, both buffers byte-for-byte the same.
  assert_eq!(c.offset(), 4);
  assert_eq!(c.meta_state(), vec!["4", "2"]);
  let post = c.state().unwrap();
  let mut pk = post[0].try_clone().unwrap();
  let mut pv = post[1].try_clone().unwrap();
  assert_eq!(pk.shape(), vec![1, 1, 4, 1]);
  assert_eq!(pv.shape(), vec![1, 1, 1, 1]);
  assert_eq!(pk.to_vec::<f32>().unwrap(), pre_k);
  assert_eq!(pv.to_vec::<f32>().unwrap(), pre_v);
}

/// Regression (closes #78 P1 iter5 — structural class-kill at `set_seq`'s
/// boundary): a full-window `set_seq` write (head AND tail empty) must NOT
/// shortcut to silently returning `new`, mutating the cached buffer's
/// non-seq axes (batch / n_kv_heads / head_dim). mlx-lm's
/// `self.<buf>[..., a:a+s, :] = new` slice-assignment implicitly validates
/// `new` against the buffer on every non-seq axis (a non-broadcastable
/// mismatch raises at the mlx level); our write-emulation goes through
/// `concat_parts([head, new, tail])`, whose `[]` / `[one]` fast path skips
/// the non-seq-axes shape compatibility check that mlx's concat would
/// otherwise enforce.
///
/// Deterministic trigger via the public API: `set_state` a `[1,1,1,1]`
/// buffer (`offset = 1`), `trim(1)` so `offset` becomes `0` while the
/// buffer remains length `1`. The next `update` with a `[B', 1, 1, 1]`
/// key/value tensor (`B' != 1`) computes `prev = 0 - 0 = 0`, no realloc
/// (`prev+S = 1 <= buf_len = 1`), and hits a FULL-window `set_seq("keys",
/// buf=[1,1,1,1], a=0, s=1, new=[B',1,1,1])` → before the fix, `concat_parts`
/// would short-circuit (empty head + empty tail filtered → `[one]` arm
/// returning `new.try_clone()`) and silently mutate the cached buffer's
/// batch axis. The fix validates non-seq axes at the `set_seq` boundary
/// via `util::broadcast_write_rhs`, so EVERY window — partial or
/// full — surfaces a recoverable `Err(ShapeMismatch)` on a mismatched
/// non-seq axis. This is a single-tensor check (`new` vs target `buf`),
/// NOT the fenced K/V cross-validation (keys vs values).
#[test]
fn chunked_set_seq_full_window_rejects_mismatched_batch_dim() {
  // Mismatched-batch single-token KV (`[2,1,1,1]`); seeded buffer is
  // `[1,1,1,1]`.
  let bad_kv2 = Array::from_slice::<f32>(&[9.0, 9.5], &(2usize, 1, 1, 1)).unwrap();

  let mut c = ChunkedKvCache::new(None);
  // Seed: set_state with [1,1,1,1] sets offset = keys.shape[2] = 1.
  let seed = kv(&[7.0]);
  c.set_state(vec![seed.try_clone().unwrap(), seed.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.offset(), 1);
  // Trim 1 → offset 0; start_position stays 0; buffer still [1,1,1,1].
  assert_eq!(c.trim(1).unwrap(), 1);
  assert_eq!(c.offset(), 0);
  assert_eq!(c.meta_state(), vec!["None", "0"]);
  // Snapshot the pre-update buffer.
  let pre = c.state().unwrap();
  // offset(0) != keys.shape[2](1) -> keys[..., :0, :] -> empty.
  // We don't need pre's exact value for this regression — only that the
  // cache buffer is NOT silently rewritten by the failed update below.

  // update: prev = offset(0) - start_position(0) = 0; prev+S = 1 <=
  // buf_len 1 -> !need_alloc. set_seq("keys", buf=[1,1,1,1], 0, 1,
  // new=[2,1,1,1]): full-window (head & tail both empty). BEFORE FIX:
  // silently returned `new` (batch-axis silently mutated). WITH FIX:
  // Err(ShapeMismatch) at the broadcast_write_rhs boundary.
  let r = c.update(&bad_kv2, &bad_kv2);
  assert!(
    matches!(&r, Err(mlxrs::Error::ShapeMismatch { .. })),
    "full-window set_seq must reject batch-axis mismatch on the public update API \
     (closes #78 P1 iter5), got {r:?}"
  );
  // Cache must be unchanged (still single-batch). No silent mutation of
  // the buffer batch dim, no advanced offset, no split-brain.
  assert_eq!(c.offset(), 0, "offset must not advance on a failed update");
  let post = c.state().unwrap();
  assert_eq!(post.len(), pre.len());
  // Direct buffer batch-shape check via a follow-on operation: a well-formed
  // [1,1,1,1] update must succeed, demonstrating the buffer's batch axis was
  // not silently mutated to 2 (which would now reject the well-formed update).
  let ok = kv(&[42.0]);
  let (mut ok_k, _) = c.update(&ok, &ok).unwrap();
  // prev = 0; update at [0,1) of the [1,1,1,1] buffer; new_offset = 1; end = 1;
  // returns keys[..., :1, :] = [42.0].
  assert_eq!(c.offset(), 1);
  assert_eq!(ok_k.shape(), vec![1, 1, 1, 1]);
  assert_eq!(ok_k.to_vec::<f32>().unwrap(), vec![42.0]);
}

/// Companion regression for `chunked_set_seq_full_window_rejects_mismatched_batch_dim`:
/// the same full-window `set_seq` boundary must reject a mismatched
/// `n_kv_heads` axis (axis 1) and a mismatched `head_dim` axis (axis 3).
/// All three non-seq axes are guarded by the SAME structural helper at the
/// boundary (one helper, three axes — class-kill, not per-axis whack-a-mole).
#[test]
fn chunked_set_seq_full_window_rejects_mismatched_heads_and_head_dim() {
  // Mismatched n_kv_heads (axis 1) and head_dim (axis 3) single-token KV.
  // Seeded buffer is [1, 1, 1, 1] so EITHER mismatch is non-broadcastable.
  let bad_kv_heads = Array::from_slice::<f32>(&[9.0, 9.5, 9.7], &(1usize, 3, 1, 1)).unwrap();
  let bad_kv_hd = Array::from_slice::<f32>(&[9.0, 9.5], &(1usize, 1, 1, 2)).unwrap();

  // --- mismatched n_kv_heads (axis 1) ---------------------------------
  let mut c1 = ChunkedKvCache::new(None);
  let seed = kv(&[7.0]);
  c1.set_state(vec![seed.try_clone().unwrap(), seed.try_clone().unwrap()])
    .unwrap();
  c1.trim(1).unwrap();
  let r1 = c1.update(&bad_kv_heads, &bad_kv_heads);
  assert!(
    matches!(&r1, Err(mlxrs::Error::ShapeMismatch { .. })),
    "full-window set_seq must reject n_kv_heads (axis 1) mismatch, got {r1:?}"
  );
  // Buffer's n_kv_heads still 1: a well-formed [1,1,1,1] update succeeds.
  let ok = kv(&[8.0]);
  c1.update(&ok, &ok).unwrap();
  assert_eq!(c1.offset(), 1);

  // --- mismatched head_dim (axis 3) -----------------------------------
  let mut c2 = ChunkedKvCache::new(None);
  c2.set_state(vec![seed.try_clone().unwrap(), seed.try_clone().unwrap()])
    .unwrap();
  c2.trim(1).unwrap();
  let r2 = c2.update(&bad_kv_hd, &bad_kv_hd);
  assert!(
    matches!(&r2, Err(mlxrs::Error::ShapeMismatch { .. })),
    "full-window set_seq must reject head_dim (axis 3) mismatch, got {r2:?}"
  );
  // Buffer's head_dim still 1: a well-formed [1,1,1,1] update succeeds.
  c2.update(&ok, &ok).unwrap();
  assert_eq!(c2.offset(), 1);
}

/// Positive companion to the chunked rejection tests above: mlx-lm's
/// `self.<buf>[..., a:a+s, :] = new` slice-assignment routes through
/// `slice_update`, which broadcasts the RHS to the slice shape (`mlx/
/// ops.cpp:843` — `broadcast_to(update, upd_shape)`). A size-1 `new` axis
/// MUST broadcast up to a size-`d` buffer axis (`d > 1`) — mlx-lm accepts
/// this and updates every broadcast row. Our `set_seq` must mirror that
/// faithfully (NOT over-reject as a "non-seq axis mismatch"). Trigger via
/// the public API: restore a `[2,1,1,1]` chunked buffer (`offset = 1`),
/// `trim(1)` so `offset` becomes `0` while the buffer stays `[2,1,1,1]`.
/// The next `update` with a `[1,1,1,1]` key/value (size-1 batch RHS)
/// triggers a full-window `set_seq("keys", buf=[2,1,1,1], 0, 1,
/// new=[1,1,1,1])` — `broadcast_write_rhs` builds slice shape `[2,1,1,1]`
/// and broadcasts `new` up, preserving the buffer's batch dim 2 (NOT
/// shrinking to 1).
///
/// Faithful to mlx-lm: mlx's `slice_update` returns the broadcast view
/// directly when the entire src is the slice (`ops.cpp:847`), so the
/// resulting buffer is a stride-0 broadcast view on the batch axis. That
/// is byte-identical to what mlx-lm itself would produce; subsequent ops
/// (matmul, concat, slice) all support strided/broadcast views via mlx's
/// lazy primitives. We assert shape (the load-bearing claim — the buffer's
/// batch axis was NOT silently shrunk to 1), and that the cache stays
/// usable for a follow-on update; value-level readback via `to_vec` would
/// require materializing the broadcast view (M2-deferred `.contiguous()`)
/// and is not the load-bearing assertion here.
#[test]
fn chunked_set_seq_full_window_broadcasts_size_one_rhs() {
  // 2-batch [2,1,1,1] seed (values [10, 20]: per-batch distinguishable).
  let seed2 = Array::from_slice::<f32>(&[10.0, 20.0], &(2usize, 1, 1, 1)).unwrap();
  // Size-1 batch RHS [1,1,1,1] — broadcasts to [2,1,1,1] (both batches
  // see the same value, the faithful mlx-lm broadcast outcome).
  let rhs1 = kv(&[99.0]); // shape [1,1,1,1]

  let mut c = ChunkedKvCache::new(None);
  c.set_state(vec![seed2.try_clone().unwrap(), seed2.try_clone().unwrap()])
    .unwrap();
  assert_eq!(c.offset(), 1); // offset = keys.shape[2] = 1.
  c.trim(1).unwrap();
  assert_eq!(c.offset(), 0);

  // Full-window set_seq with size-1 batch RHS. Mlx-lm broadcasts; we
  // mirror; the result keeps the buffer's batch dim 2 (NOT shrunk to 1
  // by the prior `[one]`-arm shortcut which would have silently returned
  // the [1,...] RHS).
  let (k, v) = c.update(&rhs1, &rhs1).unwrap();
  assert_eq!(c.offset(), 1);
  assert_eq!(
    k.shape(),
    vec![2, 1, 1, 1],
    "size-1 batch RHS must broadcast to PRESERVE buffer batch dim 2 \
     (NOT silently shrink to [1, 1, 1, 1])"
  );
  assert_eq!(v.shape(), vec![2, 1, 1, 1]);

  // The cache stays a 2-batch cache for subsequent updates — a follow-on
  // [2,1,1,1] update succeeds (concat on a non-broadcast batch axis would
  // fail if the buffer had been silently shrunk to [1,...] by the bug).
  let rhs2 = Array::from_slice::<f32>(&[7.0, 8.0], &(2usize, 1, 1, 1)).unwrap();
  let (k2, _) = c.update(&rhs2, &rhs2).unwrap();
  assert_eq!(c.offset(), 2);
  assert_eq!(k2.shape(), vec![2, 1, 2, 1]);
}
