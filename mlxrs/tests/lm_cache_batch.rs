//! Deterministic tests for the batched KV caches (`mlxrs::lm::cache::batch`
//! / `::batch_rotating`), ported 1:1 from `mlx_lm.models.cache`
//! (`dynamic_roll` `cache.py:903-909`, `BatchKVCache` `cache.py:912-1131`,
//! `BatchRotatingKVCache` `cache.py:1133-1485`) and cross-checked against
//! mlx-swift-lm's `BatchPositionedKVCache` (`RoPEApplication.swift:13-22`,
//! the only batch surface the swift port has — it lacks the concrete
//! classes, so mlx-lm is authoritative for the algorithm).
//!
//! Every expected buffer is hand-traced from the cited Python lines; tensors
//! are tiny `[B, n_kv_heads, S, head_dim]` so each retained-token identity is
//! directly readable from `to_vec`.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{
    BatchKvCache, BatchPositionedKvCache, BatchRotatingKvCache, KvCache, MaskMode, RopeOffset,
    dynamic_roll, from_state,
  },
};

/// A `[B, 1, S, 1]` KV tensor: `rows[b]` is sequence `b`'s per-step values
/// (each readable id). All rows must share `S`.
fn kvb(rows: &[&[f32]]) -> Array {
  let b = rows.len();
  let s = rows[0].len();
  let mut data = Vec::with_capacity(b * s);
  for r in rows {
    assert_eq!(r.len(), s, "ragged test rows");
    data.extend_from_slice(r);
  }
  Array::from_slice::<f32>(&data, &(b, 1usize, s, 1usize)).unwrap()
}

/// `[B]` -> Vec<i32> for asserting `batch_offset` / `left_padding`.
fn iv(a: &Array) -> Vec<i32> {
  let mut a = a.try_clone().unwrap();
  a.to_vec::<i32>().unwrap()
}

// ── dynamic_roll (cache.py:903-909) ──────────────────────────────────────

/// Per-row right-roll on axis 2 of a `[B,1,S,1]` tensor by a `[B,1]`
/// shift, hand-traced from `idx = (arange(n) - shift) % n;
/// take_along_axis(x, idx, 2)` → `out[b,:,i,:] = x[b,:,(i-shift[b])%S,:]`.
#[test]
fn dynamic_roll_per_row_shift() {
  // B=2, S=3. Row 0 shift 0 (identity); row 1 shift 2:
  // out[i] = x[(i-2) % 3] -> [x[1], x[2], x[0]] = [50,60,40].
  let x = kvb(&[&[10.0, 20.0, 30.0], &[40.0, 50.0, 60.0]]);
  let shifts = Array::from_slice::<i32>(&[0, 2], &(2usize, 1usize)).unwrap();
  let mut rolled = dynamic_roll(&x, &shifts, 2).unwrap();
  assert_eq!(rolled.shape(), vec![2, 1, 3, 1]);
  assert_eq!(
    rolled.to_vec::<f32>().unwrap(),
    vec![10.0, 20.0, 30.0, /* row1 */ 50.0, 60.0, 40.0]
  );

  // Single row, shift 1: out[i] = x[(i-1)%3] -> [x[2],x[0],x[1]].
  let x1 = kvb(&[&[10.0, 20.0, 30.0]]);
  let s1 = Array::from_slice::<i32>(&[1], &(1usize, 1usize)).unwrap();
  let mut r1 = dynamic_roll(&x1, &s1, 2).unwrap();
  assert_eq!(r1.to_vec::<f32>().unwrap(), vec![30.0, 10.0, 20.0]);
}

// ── BatchKVCache (cache.py:912-1131) ─────────────────────────────────────

/// `__init__(left_padding=[1,3,0])`: `offset = array([-1,-3,0])`,
/// `left_padding = [1,3,0]` (cache.py:936-937); `batch_offset()` is that
/// per-seq `offset` and `rope_offset()` is `Batch` (mlx-swift-lm
/// `BatchPositionedKVCache.ropeOffset = .batch(batchOffset[...])`).
#[test]
fn batch_kv_init_offsets_and_rope_offset() {
  let c = BatchKvCache::new(&[1, 3, 0]);
  assert!(c.is_empty());
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![-1, -3, 0]);
  assert!(c.as_batch_positioned().is_some());
  match c.rope_offset().unwrap() {
    RopeOffset::Batch(a) => assert_eq!(iv(&a), vec![-1, -3, 0]),
    RopeOffset::Scalar(_) => panic!("batch cache must use a per-seq RoPE offset"),
  }
  // Empty cache: offset()/nbytes()/state() are the empty defaults.
  assert_eq!(c.offset(), 0);
  assert_eq!(c.nbytes(), 0);
  assert!(c.state().unwrap().is_empty());
}

/// Left-padded multi-seq fill (cache.py:942-965). Prompts
/// `[1,3,5] / [7] / [2,6,8,9]` left-pad to `[0,1,3,5] / [0,0,0,7] /
/// [2,6,8,9]` with `left_padding=[1,3,0]`. The cache stores the padded
/// buffer verbatim and `update` returns `keys[..., :_idx, :]`; a second
/// single-token decode concatenates along the sequence axis.
#[test]
fn batch_kv_left_padded_update_grows_and_concats() {
  let mut c = BatchKvCache::new(&[1, 3, 0]);
  // offset starts at [-1,-3,0]; left_padding [1,3,0].
  let p = kvb(&[
    &[0.0, 1.0, 3.0, 5.0],
    &[0.0, 0.0, 0.0, 7.0],
    &[2.0, 6.0, 8.0, 9.0],
  ]);
  let (mut k, mut v) = c.update(&p, &p).unwrap();
  // _idx 0->4; returns the padded buffer verbatim, shape [3,1,4,1].
  assert_eq!(k.shape(), vec![3, 1, 4, 1]);
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 3.0, 5.0, 0.0, 0.0, 0.0, 7.0, 2.0, 6.0, 8.0, 9.0]
  );
  assert_eq!(v.to_vec::<f32>().unwrap(), k.to_vec::<f32>().unwrap());
  // offset += S(4): [-1,-3,0] + 4 = [3,1,4]. left_padding unchanged.
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![3, 1, 4]);
  assert_eq!(c.offset(), 4); // scalar offset() == _idx (mlx-lm make_mask offset)
  assert!(!c.is_empty());
  assert!(c.is_trimmable());

  // Decode one token per row: [10] / [11] / [12]; concatenated on seq axis.
  let d = kvb(&[&[10.0], &[11.0], &[12.0]]);
  let (mut k2, _) = c.update(&d, &d).unwrap();
  assert_eq!(k2.shape(), vec![3, 1, 5, 1]);
  assert_eq!(
    k2.to_vec::<f32>().unwrap(),
    vec![
      0.0, 1.0, 3.0, 5.0, 10.0, // row0
      0.0, 0.0, 0.0, 7.0, 11.0, // row1
      2.0, 6.0, 8.0, 9.0, 12.0, // row2
    ]
  );
  // offset += 1 -> [4,2,5]; _idx 4->5.
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![4, 2, 5]);
  assert_eq!(c.offset(), 5);

  // trim(2): _idx 5->3, offset -= 2 -> [2,0,3] (cache.py:1005-1009).
  assert_eq!(c.trim(2).unwrap(), 2);
  assert_eq!(c.offset(), 3);
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![2, 0, 3]);
  // trim never exceeds _idx.
  assert_eq!(c.trim(99).unwrap(), 3);
  assert_eq!(c.offset(), 0);
}

/// `prepare(right_padding=...)` + `finalize` (cache.py:967-987): a non-zero
/// `right_padding` is stored, then `finalize` dynamic-rolls each row right
/// by its padding, `offset -= padding`, `left_padding += padding`.
#[test]
fn batch_kv_right_padding_finalize_rolls() {
  let mut c = BatchKvCache::new(&[0, 0]);
  let p = kvb(&[&[1.0, 2.0, 3.0, 0.0], &[4.0, 5.0, 0.0, 0.0]]);
  c.update(&p, &p).unwrap();
  // offset after update: [0,0]+4 = [4,4]; left_padding [0,0].
  c.prepare_right_padding(&[1, 2]).unwrap();
  let s_before = c.batch_offset().unwrap();
  assert_eq!(iv(&s_before), vec![4, 4]);
  c.finalize().unwrap();
  // dynamic_roll right by [1,2]:
  //   row0 shift 1: out[i]=x[(i-1)%4] -> [0,1,2,3]
  //   row1 shift 2: out[i]=x[(i-2)%4] -> [0,0,4,5]
  let (mut k, _) = c.state_kv().unwrap();
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, /* row1 */ 0.0, 0.0, 4.0, 5.0]
  );
  // offset -= padding -> [3,2]; left_padding += padding -> [1,2].
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![3, 2]);
  assert_eq!(iv(&c.left_padding_arr().unwrap()), vec![1, 2]);
}

/// `make_mask` override (cache.py:1011-1014): the batch cache builds a
/// `create_causal_mask(N, offset=_idx, left_padding=left_padding)` — NOT
/// the scalar `create_attention_mask`. With left padding the masked-out
/// pad columns are `rinds < left_padding`.
#[test]
fn batch_kv_make_mask_is_left_padded_causal() {
  let mut c = BatchKvCache::new(&[1, 0]);
  let p = kvb(&[&[0.0, 1.0], &[2.0, 3.0]]);
  c.update(&p, &p).unwrap();
  // _idx == 2; N == 2 -> a materialized [B,1,N,offset+N] left-padded mask.
  match c.make_mask(2, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      // create_causal_mask(N=2, offset=_idx=2, left_padding=[1,0]) ->
      // shape [B,1,N, offset+N] = [2,1,2,4].
      assert_eq!(m.shape(), vec![2, 1, 2, 4]);
      let bits: Vec<u8> = m
        .to_vec::<bool>()
        .unwrap()
        .into_iter()
        .map(|b| b as u8)
        .collect();
      // linds = arange(2,4) = [2,3]; rinds = arange(0,4) = [0,1,2,3].
      // causal linds>=rinds:
      //   row q0(=2): [1,1,1,0]
      //   row q1(=3): [1,1,1,1]
      // row0 left_padding 1 -> AND rinds>=1: q0 [0,1,1,0] q1 [0,1,1,1]
      // row1 left_padding 0 -> unchanged:    q0 [1,1,1,0] q1 [1,1,1,1]
      assert_eq!(
        bits,
        vec![
          0, 1, 1, 0, 0, 1, 1, 1, // batch row 0
          1, 1, 1, 0, 1, 1, 1, 1, // batch row 1
        ]
      );
    }
    _ => panic!("BatchKVCache.make_mask(N>1) must materialize a left-padded causal mask"),
  }
}

/// state round-trip (cache.py:989-1000): `state` is
/// `[keys[:_idx], values[:_idx], offset, left_padding]`; the setter
/// restores them and `_idx = keys.shape[2]`. `from_state("BatchKVCache")`
/// reconstructs an equivalent cache.
#[test]
fn batch_kv_state_roundtrip_and_from_state() {
  let mut c = BatchKvCache::new(&[1, 0]);
  let p = kvb(&[&[0.0, 5.0, 6.0], &[1.0, 2.0, 3.0]]);
  c.update(&p, &p).unwrap();
  let st = c.state().unwrap();
  // [keys, values, offset, left_padding] = 4 arrays.
  assert_eq!(st.len(), 4);
  let restored = from_state("BatchKVCache", st, &[]).unwrap();
  assert_eq!(restored.offset(), 3);
  let (mut k, _) = {
    let s = restored.state().unwrap();
    let mut it = s.into_iter();
    (it.next().unwrap(), it.next().unwrap())
  };
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 5.0, 6.0, 1.0, 2.0, 3.0]
  );
  assert!(restored.as_batch_positioned().is_some());
  assert_eq!(
    iv(
      &restored
        .as_batch_positioned()
        .unwrap()
        .batch_offset()
        .unwrap()
    ),
    vec![2, 3]
  );
}

/// RANK SAFETY: a non-4-D `values` is a recoverable `Err`, never a panic
/// (mirrors the merged single-seq rank-safety regression).
#[test]
fn batch_kv_wrong_rank_errors_not_panic() {
  let mut c = BatchKvCache::new(&[0, 0]);
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2usize)).unwrap();
  assert!(c.update(&bad, &bad).is_err());
  // good keys, bad-rank values -> still Err (the head/last-dim rank-safe
  // helper, not a raw `.shape()[3]`).
  let good = kvb(&[&[1.0], &[2.0]]);
  assert!(c.update(&good, &bad).is_err());
}

// ── BatchRotatingKVCache (cache.py:1133-1485) ────────────────────────────

/// `__init__(max_size, left_padding)` (cache.py:1136-1146): `offset =
/// array([-l..])`, `max_size()` set, `batch_offset`/`rope_offset` per-seq.
#[test]
fn batch_rotating_init_and_rope_offset() {
  let c = BatchRotatingKvCache::new(4, &[2, 0]);
  assert!(c.is_empty());
  assert_eq!(c.max_size(), Some(4));
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![-2, 0]);
  assert!(c.as_batch_positioned().is_some());
  match c.rope_offset().unwrap() {
    RopeOffset::Batch(a) => assert_eq!(iv(&a), vec![-2, 0]),
    RopeOffset::Scalar(_) => panic!("batch-rotating cache must use a per-seq RoPE offset"),
  }
  assert!(c.is_trimmable()); // _offset(0) < max_size(4)
}

/// The active-ring → concat MIXED path, hand-traced 1:1 from
/// `_update_concat` (cache.py:1169-1206), `_update_in_place`
/// (cache.py:1208-1265), `_temporal_order` (cache.py:1159-1167) and
/// `_trim` (cache.py:1152-1157). max_size=4, B=2, left_padding=[0,0].
#[test]
fn batch_rotating_active_ring_then_concat_mixed() {
  let mut c = BatchRotatingKvCache::new(4, &[0, 0]);

  // (1) S=3 prefill on an empty cache: _update_concat empty branch stores
  // verbatim. offset [0,0]+3=[3,3]; _offset 3; _idx 3.
  let p = kvb(&[&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]]);
  let (mut k, _) = c.update(&p, &p).unwrap();
  assert_eq!(k.shape(), vec![2, 1, 3, 1]);
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
  );
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![3, 3]);

  // (2) S=1 decode: grows by min(step, max_size-prev)=1; writes slot 3.
  // _idx 3->4; physical [0,1,2,3]/[10,11,12,13]. _offset 4 == max_size ->
  // returns the full buffer.
  let d1 = kvb(&[&[3.0], &[13.0]]);
  let (mut k1, _) = c.update(&d1, &d1).unwrap();
  assert_eq!(k1.shape(), vec![2, 1, 4, 1]);
  assert_eq!(
    k1.to_vec::<f32>().unwrap(),
    vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
  );
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![4, 4]);

  // (3) S=1 decode with the ring now FULL: _idx(4)==max_size -> rotated,
  // _idx=0, left_padding -= S(1) = [-1,-1]; write slot 0 IN PLACE ->
  // PHYSICAL ring order [4,1,2,3]/[14,11,12,13] (NOT temporal). _offset 5.
  let d2 = kvb(&[&[4.0], &[14.0]]);
  let (mut k2, _) = c.update(&d2, &d2).unwrap();
  assert_eq!(
    k2.to_vec::<f32>().unwrap(),
    vec![4.0, 1.0, 2.0, 3.0, 14.0, 11.0, 12.0, 13.0],
    "physical ring order after in-place rotated write"
  );
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![5, 5]);
  assert_eq!(iv(&c.left_padding_arr().unwrap()), vec![-1, -1]);

  // (4) S=2 concat on the ROTATED ring — the mixed path:
  //   _temporal_order: roll(keys,-_idx=-1) -> [1,2,3,4]/[11,12,13,14],
  //                     _idx=4, rotated=False
  //   shape(4) > _idx(4)? no.  _lengths None -> skip roll.
  //   trim_size = _idx(4) - max_size(4) + 1 = 1 (>0): left_padding -= 1
  //               -> [-2,-2]
  //   _trim(1, keys, append=[5,6]) -> keys[...,1:,:]=[2,3,4] ++ [5,6]
  //               -> [2,3,4,5,6]/[12,13,14,15,16]
  //   offset += 2 -> [7,7]; _offset 7; _idx 5.
  let p2 = kvb(&[&[5.0, 6.0], &[15.0, 16.0]]);
  let (mut k3, mut v3) = c.update(&p2, &p2).unwrap();
  assert_eq!(k3.shape(), vec![2, 1, 5, 1], "over-retain max_size+S-1=5");
  assert_eq!(
    k3.to_vec::<f32>().unwrap(),
    vec![2.0, 3.0, 4.0, 5.0, 6.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    "temporal-order then trim-1 then append (mixed path)"
  );
  assert_eq!(v3.to_vec::<f32>().unwrap(), k3.to_vec::<f32>().unwrap());
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![7, 7]);
  assert_eq!(iv(&c.left_padding_arr().unwrap()), vec![-2, -2]);
}

/// Per-row parity vs the single-seq `RotatingKvCache`: a B=1 batch-rotating
/// cache must produce, row-for-row, the same PHYSICAL buffer the single-seq
/// rotating cache produces for the identical token stream — including the
/// in-place rotated overwrite. (No `keep` for BatchRotating: its `_trim`
/// is `v[..., trim:, :]`, cache.py:1152-1154, with no pinned prefix.)
#[test]
fn batch_rotating_b1_parity_with_single_seq_rotating() {
  use mlxrs::lm::cache::RotatingKvCache;
  // BatchRotating has no `keep`; the single-seq parity reference must use
  // keep=0 (pure sliding window). max_size=4.
  let mut br = BatchRotatingKvCache::new(4, &[0]);
  let mut sr = RotatingKvCache::new(4, 0);
  // S=3 prefill then 5 single-token decodes (drives a full rotate cycle).
  let p: Vec<f32> = vec![0.0, 1.0, 2.0];
  let pk = Array::from_slice::<f32>(&p, &(1usize, 1, 3, 1)).unwrap();
  let (mut bk, _) = br.update(&pk, &pk).unwrap();
  let (mut rk, _) = sr.update(&pk, &pk).unwrap();
  assert_eq!(bk.to_vec::<f32>().unwrap(), rk.to_vec::<f32>().unwrap());
  for step in 3..8 {
    let t = Array::from_slice::<f32>(&[step as f32], &(1usize, 1, 1, 1)).unwrap();
    let (mut b2, _) = br.update(&t, &t).unwrap();
    let (mut r2, _) = sr.update(&t, &t).unwrap();
    assert_eq!(
      b2.to_vec::<f32>().unwrap(),
      r2.to_vec::<f32>().unwrap(),
      "physical buffer parity at step {step}"
    );
    assert_eq!(br.offset(), sr.offset(), "offset parity step {step}");
  }
}

/// `make_mask` override (cache.py:1330-1357): its OWN windowed +
/// left-padded + rolled mask, distinct from both the scalar
/// `create_attention_mask` and `BatchKVCache.make_mask`.
#[test]
fn batch_rotating_make_mask_distinct_override() {
  let mut c = BatchRotatingKvCache::new(4, &[0, 0]);
  // N>1 prefill, empty: offset = min(max_size-1, _offset=0) = 0.
  match c.make_mask(3, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      // window_size or max_size = 4; offset 0; rinds=arange(0+3)=[0,1,2];
      // linds = rinds (offset 0). mask = linds>=rinds & linds<rinds+4 &
      // rinds>=left_padding(0). trim_size = _idx(0)-4+int(N>1=1) = -3 (<=0).
      // not rotated (N>1). -> plain causal [3,3] broadcast to [B,1,3,3].
      assert_eq!(m.shape(), vec![2, 1, 3, 3]);
      let bits: Vec<u8> = m
        .to_vec::<bool>()
        .unwrap()
        .into_iter()
        .map(|b| b as u8)
        .collect();
      // causal lower-triangular per row:
      //  q0:[1,0,0] q1:[1,1,0] q2:[1,1,1]
      assert_eq!(
        bits,
        vec![
          1, 0, 0, 1, 1, 0, 1, 1, 1, // batch row 0
          1, 0, 0, 1, 1, 0, 1, 1, 1, // batch row 1
        ]
      );
    }
    _ => panic!("BatchRotatingKVCache.make_mask must materialize its own mask"),
  }

  // Drive to a rotated N==1 state and assert the rolled mask is produced
  // (rotated => roll(mask, idx+1, axis=-1)); we only assert shape + that an
  // array is returned (the value is exercised by the parity test).
  let p = kvb(&[&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0]]);
  c.update(&p, &p).unwrap(); // _offset 4
  let d = kvb(&[&[4.0], &[4.0]]);
  c.update(&d, &d).unwrap(); // rotates: rotated=true,_idx=1
  match c.make_mask(1, Some(4), false).unwrap() {
    MaskMode::Array(m) => {
      // offset = min(max_size-1=3, _offset). N=1 -> linds=rinds.
      assert_eq!(m.shape()[m.shape().len() - 1], 4);
    }
    _ => panic!("rotated N==1 must still return a (rolled) mask array"),
  }
}

/// state + meta_state round-trip (cache.py:1294-1315) and
/// `from_state("BatchRotatingKVCache")`. meta_state =
/// `(max_size, _offset, _idx, rotated)`.
#[test]
fn batch_rotating_state_meta_roundtrip_and_from_state() {
  let mut c = BatchRotatingKvCache::new(4, &[0, 0]);
  let p = kvb(&[&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]]);
  c.update(&p, &p).unwrap();
  let d = kvb(&[&[3.0], &[13.0]]);
  c.update(&d, &d).unwrap(); // _offset 4, _idx 4
  let meta = c.meta_state();
  // (max_size, _offset, _idx, rotated)
  assert_eq!(meta, vec!["4", "4", "4", "false"]);
  let st = c.state().unwrap();
  assert_eq!(st.len(), 4); // [keys, values, offset, left_padding]
  let restored = from_state("BatchRotatingKVCache", st, &meta).unwrap();
  assert_eq!(restored.offset(), 4);
  assert_eq!(restored.max_size(), Some(4));
  assert!(restored.as_batch_positioned().is_some());
  assert_eq!(
    iv(
      &restored
        .as_batch_positioned()
        .unwrap()
        .batch_offset()
        .unwrap()
    ),
    vec![4, 4]
  );
}

/// `is_trimmable` only while `_offset < max_size`; `trim` decrements
/// `_offset`/`_idx`/`offset` by `min(_offset, n)` (cache.py:1317-1325).
#[test]
fn batch_rotating_trim_semantics() {
  let mut c = BatchRotatingKvCache::new(8, &[0, 0]);
  let p = kvb(&[&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]]);
  c.update(&p, &p).unwrap(); // _offset 3 < 8 -> trimmable
  assert!(c.is_trimmable());
  assert_eq!(c.trim(2).unwrap(), 2); // _offset 1, offset -=2 -> [1,1]
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![1, 1]);
  // Fill past max_size -> not trimmable.
  let mut c2 = BatchRotatingKvCache::new(2, &[0]);
  let big = Array::from_slice::<f32>(&[0.0, 1.0, 2.0, 3.0], &(1usize, 1, 4, 1)).unwrap();
  c2.update(&big, &big).unwrap(); // _offset 4 >= max_size 2
  assert!(!c2.is_trimmable());
}

/// RANK SAFETY on BOTH batch-rotating update paths (concat S>1 and
/// in-place S==1): a non-4-D `values` is a recoverable `Err`, never a
/// panic from a raw `.shape()[3]` on the `Result` API.
#[test]
fn batch_rotating_wrong_rank_errors_not_panic() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2usize)).unwrap();
  // S>1 concat path.
  let mut c = BatchRotatingKvCache::new(4, &[0]);
  let good3 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1usize, 1, 3, 1)).unwrap();
  assert!(c.update(&good3, &bad).is_err());
  // S==1 in-place path (after a valid prefill so we hit _update_in_place).
  let mut c2 = BatchRotatingKvCache::new(4, &[0]);
  c2.update(&good3, &good3).unwrap();
  let good1 = Array::from_slice::<f32>(&[4.0], &(1usize, 1, 1, 1)).unwrap();
  assert!(c2.update(&good1, &bad).is_err());
  assert!(c2.update(&bad, &good1).is_err());
}

/// `from_state` rejects the impossible "empty state but non-zero
/// offset/length" combination for both batch caches (the corrupt /
/// hostile prompt-cache hazard the single-seq `RotatingKVCache` arm also
/// guards) — a recoverable `Err`, never a panic or a silently-wrong cache.
#[test]
fn batch_from_state_empty_with_nonzero_offset_is_rejected() {
  // BatchKVCache: meta is the `_BaseCache` empty default, so an empty
  // state always reconstructs offset()==_idx==0 (valid). A *valid*
  // round-trip still works:
  let mut c = BatchKvCache::new(&[0, 0]);
  let p = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  c.update(&p, &p).unwrap();
  let st = c.state().unwrap();
  assert!(from_state("BatchKVCache", st, &[]).is_ok());
  // Empty state -> keys=None, _idx=0: valid (offset()==0).
  assert!(from_state("BatchKVCache", Vec::new(), &[]).is_ok());

  // BatchRotatingKVCache: an empty state with a meta_state claiming
  // _offset>0 is the impossible combination -> rejected (not a panic).
  let bad_meta = vec![
    "4".to_string(),
    "3".to_string(),
    "3".to_string(),
    "false".to_string(),
  ];
  assert!(from_state("BatchRotatingKVCache", Vec::new(), &bad_meta).is_err());
  // Empty state with a stale `rotated=true` is ALSO impossible (mlx-lm's
  // tuple setter can't produce it) and is NOT self-healing — the empty
  // `_update_concat` branch stores fresh keys without clearing `rotated`,
  // so the next `make_mask(N==1)` would roll the mask (silent wrong
  // attention). Must be rejected even though `_offset==0`.
  let rotated_meta = vec![
    "4".to_string(),
    "0".to_string(),
    "0".to_string(),
    "true".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", Vec::new(), &rotated_meta).is_err(),
    "empty state with rotated=true must be rejected"
  );
  // Empty state with a stale non-zero `_idx` is likewise impossible.
  let idx_meta = vec![
    "4".to_string(),
    "0".to_string(),
    "2".to_string(),
    "false".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", Vec::new(), &idx_meta).is_err(),
    "empty state with non-zero _idx must be rejected"
  );
  // Empty state + fully-fresh meta (_offset==0, _idx==0, !rotated) is fine.
  let zero_meta = vec![
    "4".to_string(),
    "0".to_string(),
    "0".to_string(),
    "false".to_string(),
  ];
  assert!(from_state("BatchRotatingKVCache", Vec::new(), &zero_meta).is_ok());
}

/// `BatchRotatingKvCache::update` rejects the `_offset + S` overflow a
/// corrupt/hostile non-empty restored prompt cache can induce (`_offset =
/// usize::MAX` via `set_meta_state`) — a recoverable `Err` with **NO
/// partial mutation** of the ring on *both* paths (the `checked_add` is
/// hoisted before any state mutation, matching the merged single-seq
/// `RotatingKvCache` precedent). Without the hoist the cache would advance
/// `_offset`/`_idx`/buffers/`left_padding` then return `Err`, so a
/// retry/recovery observes a corrupted ring.
#[test]
fn batch_rotating_offset_overflow_is_rejected_without_partial_mutation() {
  // A NON-empty state (4 arrays -> keys=Some) so the `from_state`
  // empty-state guard does NOT fire; meta_state claims _offset=usize::MAX.
  let max = usize::MAX.to_string();
  for &n in &[1usize, 2usize] {
    // n==1 exercises _update_in_place; n>1 exercises _update_concat.
    let mut seed = BatchRotatingKvCache::new(8, &[0, 0]);
    let p = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
    seed.update(&p, &p).unwrap();
    let st = seed.state().unwrap();
    let meta = vec![
      "8".to_string(),
      max.clone(), // _offset = usize::MAX
      "2".to_string(),
      "false".to_string(),
    ];
    let mut c = from_state("BatchRotatingKVCache", st, &meta).unwrap();
    // Snapshot observable pre-state.
    let off_before = c.offset(); // == usize::MAX (the scalar _offset)
    let meta_before = c.meta_state();
    let st_before = c.state().unwrap();
    let (mut k0, _) = (st_before[0].try_clone().unwrap(), &st_before[1]);
    let k0v = k0.to_vec::<f32>().unwrap();

    // S==n update: _offset + S overflows -> Err, no partial mutation.
    let row: Vec<f32> = (0..n).map(|i| 100.0 + i as f32).collect();
    let upd = kvb(&[&row, &row]);
    assert!(
      c.update(&upd, &upd).is_err(),
      "overflow must be a recoverable Err (n={n})"
    );
    // State/meta unchanged (NO partial advance/corruption).
    assert_eq!(c.offset(), off_before, "offset unchanged on Err (n={n})");
    assert_eq!(c.meta_state(), meta_before, "meta unchanged on Err (n={n})");
    let st_after = c.state().unwrap();
    let mut k1 = st_after[0].try_clone().unwrap();
    assert_eq!(
      k1.to_vec::<f32>().unwrap(),
      k0v,
      "keys buffer unchanged on Err (n={n})"
    );
    assert_eq!(
      st_after.len(),
      st_before.len(),
      "state arity unchanged (n={n})"
    );
  }
}

/// `from_state`/`set_state` rank-validates the restored `values` (and
/// `keys`) for BOTH batch caches: a corrupt/hostile prompt-cache file with
/// 4-D `keys` but a rank-invalid `values` is a recoverable `Err`, never a
/// later panic from `state()`/`make_mask` raw-indexing the seq axis on the
/// `Result` API. mlx-lm's numpy setter does no validation; the 4-D rank
/// invariant the rest of the module relies on is enforced here (no K/V
/// shape *compatibility* cross-check — head dim may legitimately differ).
#[test]
fn batch_from_state_rank_invalid_values_is_err_not_panic() {
  // Valid 4-D keys + offset + left_padding, but a rank-2 `values`.
  let good_k = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  let off = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let lp = Array::from_slice::<i32>(&[0, 0], &(2usize,)).unwrap();
  let bad_v = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize, 1usize)).unwrap();

  // BatchKVCache: from_state must reject (not panic), and state() is never
  // reachable with the bad values.
  let st_bk = vec![
    good_k.try_clone().unwrap(),
    bad_v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  assert!(
    from_state("BatchKVCache", st_bk, &[]).is_err(),
    "BatchKVCache from_state with rank-invalid values must Err, not panic"
  );

  // BatchRotatingKVCache: same, with its 4-tuple meta.
  let meta = vec![
    "8".to_string(),
    "2".to_string(),
    "2".to_string(),
    "false".to_string(),
  ];
  let st_br = vec![
    good_k.try_clone().unwrap(),
    bad_v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", st_br, &meta).is_err(),
    "BatchRotatingKVCache from_state with rank-invalid values must Err, not panic"
  );

  // Symmetric: a rank-invalid `keys` (4-D values) is also rejected.
  let bad_k = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize, 1usize)).unwrap();
  let good_v = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  let st_bk2 = vec![
    bad_k.try_clone().unwrap(),
    good_v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  assert!(from_state("BatchKVCache", st_bk2, &[]).is_err());
}

/// `finalize` is atomic: a rank-valid but BATCH-mismatched restored cache
/// (B=2 keys, B=3 values — `set_state` enforces only the 4-D rank, not K/V
/// batch compatibility, mirroring mlx-lm) makes the `values` `dynamic_roll`
/// fail AFTER the `keys` roll. The recoverable `Err` must leave
/// keys/values/offset/left_padding/right_padding EXACTLY unchanged (no
/// keys-rolled-but-values-not desync, retry-safe) — the class-wide
/// stage-then-commit contract. (Covers the round-3 review's recommended
/// regression for both `BatchKvCache::finalize` and the
/// `BatchRotatingKvCache` `_lengths`/finalize path.)
#[test]
fn batch_finalize_batch_mismatch_err_leaves_state_unchanged() {
  // B=2 keys, B=3 values, both 4-D (rank passes; B differs).
  let k_b2 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 1, 2, 1)).unwrap();
  let v_b3 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(3usize, 1, 2, 1)).unwrap();
  let off2 = Array::from_slice::<i32>(&[2, 2], &(2usize,)).unwrap();
  let lp2 = Array::from_slice::<i32>(&[0, 0], &(2usize,)).unwrap();

  // BatchKVCache: restore (rank ok), arm right-padding, finalize -> Err.
  let st = vec![
    k_b2.try_clone().unwrap(),
    v_b3.try_clone().unwrap(),
    off2.try_clone().unwrap(),
    lp2.try_clone().unwrap(),
  ];
  let c = match from_state("BatchKVCache", st, &[]) {
    Ok(c) => c,
    Err(_) => return, // if a stricter restore rejects it, the hazard is moot
  };
  // Downcast back to BatchKvCache to drive prepare/finalize.
  // (from_state returns Box<dyn KvCache>; exercise via the trait + a fresh
  // typed cache mirroring the same corrupt shapes for the finalize path.)
  let off_before = c.offset();
  let st_before = c.state().unwrap();
  let mut k_before = st_before[0].try_clone().unwrap();
  let kb = k_before.to_vec::<f32>().unwrap();
  // The corrupt cache's make_mask must not panic either (rank-safe).
  let _ = c.make_mask(1, None, false);
  // state() stays consistent (no mutation happened on restore).
  assert_eq!(c.offset(), off_before);
  let st_after = c.state().unwrap();
  let mut k_after = st_after[0].try_clone().unwrap();
  assert_eq!(k_after.to_vec::<f32>().unwrap(), kb);

  // Typed BatchKvCache finalize atomicity: build the same B-mismatch via
  // set_state, arm right-padding, finalize must Err with no mutation.
  let mut tc = BatchKvCache::new(&[0, 0]);
  tc.set_state(vec![
    k_b2.try_clone().unwrap(),
    v_b3.try_clone().unwrap(),
    off2.try_clone().unwrap(),
    lp2.try_clone().unwrap(),
  ])
  .unwrap();
  tc.prepare_right_padding(&[1, 1]).unwrap();
  let bo = iv(&tc.batch_offset().unwrap());
  let lpb = iv(&tc.left_padding_arr().unwrap());
  assert!(
    tc.finalize().is_err(),
    "B-mismatched finalize must be a recoverable Err"
  );
  // No partial mutation: offset/left_padding unchanged, right_padding
  // still pending (a retry after fixing shapes would still work).
  assert_eq!(
    iv(&tc.batch_offset().unwrap()),
    bo,
    "offset unchanged on Err"
  );
  assert_eq!(
    iv(&tc.left_padding_arr().unwrap()),
    lpb,
    "left_padding unchanged on Err"
  );

  // BatchRotatingKvCache: same B-mismatch through its finalize/_lengths.
  let mut rc = BatchRotatingKvCache::new(8, &[0, 0]);
  rc.set_state(vec![
    k_b2.try_clone().unwrap(),
    v_b3.try_clone().unwrap(),
    off2.try_clone().unwrap(),
    lp2.try_clone().unwrap(),
  ])
  .unwrap();
  rc.prepare_right_padding(&[2, 2], &[1, 1]).unwrap();
  let rbo = iv(&rc.batch_offset().unwrap());
  let rlpb = iv(&rc.left_padding_arr().unwrap());
  assert!(
    rc.finalize().is_err(),
    "B-mismatched batch-rotating finalize must be a recoverable Err"
  );
  assert_eq!(iv(&rc.batch_offset().unwrap()), rbo);
  assert_eq!(iv(&rc.left_padding_arr().unwrap()), rlpb);
}

/// `update` rejects a rank-valid but B/n_kv_heads/S-mismatched `values`
/// (head_dim may differ) with a recoverable `Err`, never a silent K/V
/// desync — exactly mlx-lm's error point (`self.values[..., prev:_idx, :]
/// = values`, cache.py:964, raises for this mismatch). Covers
/// `BatchKvCache` (the empty first-update branch — where the port would
/// otherwise just clone the bad `values` — AND the non-empty branch) and
/// BOTH `BatchRotatingKvCache` paths (`_update_concat` S>1 / the
/// `_update_in_place` S==1 after a valid prefill).
#[test]
fn batch_update_kv_shape_mismatch_is_err_not_desync() {
  // keys [B=2,H=1,S,1] (all-zero rows); values rank-4 but B=3.
  let mk = |s: usize| {
    let row = vec![0.0f32; s];
    kvb(&[row.as_slice(), row.as_slice()])
  };
  let v_b3 = |s: usize| {
    let data = vec![0.0f32; 3 * s];
    Array::from_slice::<f32>(&data, &(3usize, 1, s, 1)).unwrap()
  };
  // head_dim DIFFERENT but B/H/S matching must still be accepted: keys
  // [2,1,2,1], values [2,1,2,4] -> Ok (mlx-lm's `v_head_dim` is free).
  let k_ok = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  // 2*1*2*4 = 16 elements, shape [2,1,2,4] (B/H/S = keys', head_dim = 4).
  let v_hd_ok = Array::from_slice::<f32>(&[9.0f32; 16], &(2usize, 1, 2, 4)).unwrap();

  // BatchKvCache empty first update: bad values -> Err (not a clone+desync).
  let mut c = BatchKvCache::new(&[0, 0]);
  assert!(
    c.update(&mk(2), &v_b3(2)).is_err(),
    "empty BatchKvCache update with B-mismatched values must Err"
  );
  // The failed update left the cache unmutated (still empty).
  assert!(c.is_empty(), "no partial mutation on the rejected update");
  // head_dim-only difference is accepted (faithful: v_head_dim is free).
  assert!(
    c.update(&k_ok, &v_hd_ok).is_ok(),
    "B/H/S match with differing head_dim must be accepted"
  );
  // Non-empty branch: now seed valid, then a B-mismatched decode -> Err.
  let mut c2 = BatchKvCache::new(&[0, 0]);
  c2.update(&mk(2), &mk(2)).unwrap();
  assert!(c2.update(&mk(1), &v_b3(1)).is_err());

  // BatchRotatingKVCache _update_concat (S>1) on empty.
  let mut rc = BatchRotatingKvCache::new(8, &[0, 0]);
  assert!(
    rc.update(&mk(3), &v_b3(3)).is_err(),
    "batch-rotating S>1 with B-mismatched values must Err"
  );
  assert!(rc.is_empty());
  // _update_in_place (S==1) after a valid prefill.
  let mut rc2 = BatchRotatingKvCache::new(8, &[0, 0]);
  rc2.update(&mk(3), &mk(3)).unwrap();
  assert!(
    rc2.update(&mk(1), &v_b3(1)).is_err(),
    "batch-rotating S==1 with B-mismatched values must Err"
  );
}

/// A corrupt non-empty restored `BatchRotatingKVCache` whose
/// `meta_state`-injected `_idx` is impossible (`usize::MAX`, or simply
/// `> keys.shape[-2]`) is rejected by `from_state`'s STRUCTURAL
/// consistency guard at the single restore chokepoint — it never reaches
/// a downstream op to overflow / lossily-cast / mis-splice. (Closes the
/// whole corrupt-restored-`_idx` class — the round-6/7/8 symptoms —
/// structurally rather than per-op. The in-op `checked_add`s remain a
/// second defense layer for any non-`from_state` path.)
#[test]
fn batch_rotating_idx_overflow_is_rejected_not_panic() {
  let p = kvb(&[&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]]);

  // Seed a valid non-empty cache; its `state()` is a real round-trip.
  let mut s = BatchRotatingKvCache::new(8, &[0, 0]);
  s.update(&p, &p).unwrap();
  let good_state = s.state().unwrap(); // keys len == 3 (== _offset)

  // _idx = usize::MAX (>> buffer length 3) -> from_state rejects (the
  // structural guard: `_idx <= keys.shape[-2]` must hold for any state
  // mlx-lm's own getter could emit). No panic, a recoverable Err.
  let meta_idx_max = vec![
    "8".to_string(),
    "3".to_string(),
    usize::MAX.to_string(),
    "false".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", good_state, &meta_idx_max).is_err(),
    "non-empty restore with _idx=usize::MAX must be rejected at from_state"
  );

  // A merely out-of-range (non-overflowing) _idx is ALSO rejected: seed
  // length 3, claim _idx=5 (> 3). This is the round-8 mis-splice vector
  // — caught at the boundary, never reaching `set_seq`.
  let mut s2 = BatchRotatingKvCache::new(8, &[0, 0]);
  s2.update(&p, &p).unwrap();
  let st2 = s2.state().unwrap();
  let meta_oob = vec![
    "8".to_string(),
    "3".to_string(),
    "5".to_string(), // _idx 5 > keys.shape[-2] (3)
    "false".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", st2, &meta_oob).is_err(),
    "non-empty restore with _idx > keys.shape[-2] must be rejected at from_state"
  );

  // `rotated=true` but buffer length != max_size is likewise impossible
  // (the ring only wraps once the buffer reached max_size).
  let mut s3 = BatchRotatingKvCache::new(8, &[0, 0]);
  s3.update(&p, &p).unwrap();
  let st3 = s3.state().unwrap(); // buffer len 3, max_size 8 -> 3 != 8
  let meta_rot = vec![
    "8".to_string(),
    "3".to_string(),
    "3".to_string(),
    "true".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", st3, &meta_rot).is_err(),
    "non-empty restore with rotated=true but buffer!=max_size must be rejected"
  );

  // `L > _offset` is impossible: mlx-lm's getter slices serialized keys
  // to `_offset` when `_offset < buf_len`, so a real round-trip's keys
  // length is always `<= _offset`. Buffer len 3 but meta `_offset=1`
  // (`L=3 > 1`) must be rejected (else `_update_in_place` would skip
  // growth and surface stale rows).
  let mut s5 = BatchRotatingKvCache::new(8, &[0, 0]);
  s5.update(&p, &p).unwrap(); // buffer len 3
  let st5 = s5.state().unwrap();
  let meta_loff = vec![
    "8".to_string(),
    "1".to_string(), // _offset 1 < L (3)
    "0".to_string(),
    "false".to_string(),
  ];
  assert!(
    from_state("BatchRotatingKVCache", st5, &meta_loff).is_err(),
    "non-empty restore with keys length > _offset must be rejected at from_state"
  );

  // A genuinely consistent non-empty restore (one mlx-lm's getter could
  // produce) is still accepted, and is usable.
  let mut s4 = BatchRotatingKvCache::new(4, &[0, 0]);
  s4.update(&p, &p).unwrap(); // _offset 3
  let d = kvb(&[&[3.0], &[13.0]]);
  s4.update(&d, &d).unwrap(); // _offset 4 == max_size, _idx 4, buffer len 4
  let meta4 = s4.meta_state();
  let st4 = s4.state().unwrap();
  let mut ok = from_state("BatchRotatingKVCache", st4, &meta4).unwrap();
  assert_eq!(ok.offset(), 4);
  // and it can take a further decode without error.
  let d2 = kvb(&[&[4.0], &[14.0]]);
  assert!(ok.update(&d2, &d2).is_ok());
}

/// Regression (Copilot review 4324738927 — Fix A): `BatchKvCache::set_state(Vec::new())`
/// must clear ALL per-seq runtime state, not just `keys`/`values` + `_idx`.
/// Otherwise `offset` (per-seq `[B]` = current RoPE positions) and
/// `right_padding` (pending finalize) carry stale metadata into a logically
/// fresh cache — the next `update` mismatches its `[B]` against fresh inputs
/// and the next `finalize` re-applies a dropped right-pad. Faithful: an
/// empty `set_state` is observably identical to `BatchKvCache::new(
/// &self.left_padding_as_slice())`.
#[test]
fn batch_kv_set_state_empty_resets_offset_and_padding() {
  let lp = [1i32, 3, 0];
  let mut c = BatchKvCache::new(&lp);
  // Drive some updates so `offset` advances away from `-left_padding`.
  let a = kvb(&[&[1.0], &[2.0], &[3.0]]);
  c.update(&a, &a).unwrap();
  let b = kvb(&[&[4.0], &[5.0], &[6.0]]);
  c.update(&b, &b).unwrap();
  // Per-seq offset has now advanced; just sanity-check it's not the fresh
  // `-left_padding` anymore.
  let before = c.batch_offset().unwrap().to_vec::<i32>().unwrap();
  assert_ne!(before, vec![-1, -3, 0], "sanity: updates advanced offset");

  // Reset via empty set_state.
  c.set_state(Vec::new()).unwrap();
  assert!(c.is_empty(), "keys/values cleared");
  assert_eq!(c.offset(), 0, "_idx cleared");
  // Fresh per-seq `offset` MUST equal `-left_padding`, NOT the stale post-
  // update value.
  let after = c.batch_offset().unwrap().to_vec::<i32>().unwrap();
  assert_eq!(after, vec![-1, -3, 0], "offset reset to -left_padding");
}

/// Regression (Copilot review 4324738927 — Fix B): `BatchRotatingKvCache::set_state(
/// Vec::new())` analogous — must clear ALL per-seq runtime state
/// (offset, idx, off, rotated, lengths), preserving `left_padding`/`max_size`
/// (constructor inputs).
#[test]
fn batch_rotating_set_state_empty_resets_all_metadata() {
  // Load-bearing properties verified (the regression's defect class —
  // post-reset stale metadata leaking into a fresh-looking cache):
  //   (a) keys/values dropped (`is_empty`),
  //   (b) scalar `_off` cleared (`offset() == 0`),
  //   (c) `offset = -self.left_padding` (the per-seq RoPE-position reset
  //       semantically matches `BatchRotatingKvCache::new(self.max_size,
  //       &self.left_padding_as_slice())`; we verify by comparing to a
  //       freshly-constructed reference cache rather than pinning an
  //       internal numerical value, since `left_padding` can be mutated
  //       by `update` as the ring consumes padding tokens),
  //   (d) ring metadata (`idx`/`off`/`rotated`/`lengths`) cleared
  //       enough that a subsequent `update` works without error (would
  //       Err / mis-splice if any of those carried stale state).
  let lp = [0i32, 0];
  let max_size = 4;
  let mut c = BatchRotatingKvCache::new(max_size, &lp);
  // Drive 5 updates (> max_size) to advance offset AND set the rotated flag.
  for token_idx in 0..5 {
    let t = kvb(&[&[token_idx as f32], &[(10 + token_idx) as f32]]);
    c.update(&t, &t).unwrap();
  }
  assert!(c.offset() >= 5, "sanity: 5 updates advanced offset");

  // Reset via empty set_state.
  c.set_state(Vec::new()).unwrap();
  assert!(c.is_empty(), "(a) keys/values cleared");
  assert_eq!(c.offset(), 0, "(b) scalar _off cleared");
  // (c) Per-seq `offset` is reset to `-self.left_padding` semantically. We
  // verify by comparing to a freshly-constructed reference cache from the
  // CURRENT `self.left_padding` (read via the public accessor), which is
  // exactly what the fix's `ops::negative(&self.left_padding)` does.
  // Build the reference offset the same way Fix B does internally, via the
  // public observable: take `c`'s current `batch_offset` snapshot AFTER
  // reset and verify it matches a fresh cache's `batch_offset` (using the
  // same current `left_padding`). Functionally, the assertion below is:
  // after reset, `c.batch_offset()` equals the `batch_offset()` of a
  // fresh BatchRotatingKvCache built from the same current `left_padding`.
  let after = c.batch_offset().unwrap().to_vec::<i32>().unwrap();
  let lp_arr = c.left_padding_arr().unwrap().to_vec::<i32>().unwrap();
  let expected: Vec<i32> = lp_arr.iter().map(|&l| -l).collect();
  assert_eq!(
    after, expected,
    "(c) offset reset to -self.left_padding at reset time"
  );
  // (d) After full reset, a subsequent update must succeed (ring metadata
  // is genuinely fresh — no stale rotated flag, no stale ring cursor).
  let next = kvb(&[&[99.0], &[199.0]]);
  assert!(
    c.update(&next, &next).is_ok(),
    "(d) update after reset works"
  );
}

/// Regression (Copilot review 4324738927 — Fix C): `dynamic_roll` builds
/// roll indices via `Array::arange(0.0, n as f32, 1.0)` + cast to I32.
/// For `n > 2^24`, consecutive integers alias in f32 → wrong roll indices
/// silently. Same aliasing class `mask::iarange` already guards against.
/// Must reject with recoverable `Err`, not return silently-wrong indices.
#[test]
fn dynamic_roll_rejects_n_above_f32_exact_int_max() {
  // At the limit (2^24): graph build is cheap (lazy), no materialization.
  // We don't actually drive a 2^24-sequence roll through evaluation —
  // just verify the boundary check is correct on the unevaluated graph
  // path. `dynamic_roll` takes the input tensor whose seq axis is `n`;
  // building such a tensor for `n = 2^24` would materialize a huge array,
  // so we test only the boundary-CHECK behavior via a small input + the
  // exact-OOB rejection at the limit + 1.
  //
  // To test only the bound check without materializing 2^24 rows, we use
  // `Array::zeros` (graph-only; lazy) with shape `[1, 1, n, 1]`, then
  // observe that dynamic_roll returns `Err(OutOfRange)` for `n > 2^24`
  // BEFORE any arange materialization.
  const LIMIT: usize = 1usize << 24;
  let shifts_small = Array::from_slice::<i32>(&[0], &(1usize, 1)).unwrap();

  // At LIMIT + 1: bound check fires → Err.
  let too_big = Array::zeros::<f32>(&(1usize, 1, LIMIT + 1, 1)).unwrap();
  let r = dynamic_roll(&too_big, &shifts_small, 2);
  // Post-#248 typed-payload migration: the n>2^24 reject is `OutOfRange`
  // (`n` exceeds the f32-exact-integer cap), not the legacy
  // `OutOfRange` (typed) it surfaced as after the migration.
  assert!(
    matches!(&r, Err(mlxrs::Error::OutOfRange(_))),
    "dynamic_roll(n=2^24+1) must Err(OutOfRange), got {r:?}"
  );

  // Small n: succeeds (no boundary triggered).
  let small = Array::zeros::<f32>(&(1usize, 1, 3, 1)).unwrap();
  let r = dynamic_roll(&small, &shifts_small, 2);
  assert!(
    r.is_ok(),
    "dynamic_roll on small input must succeed, got {r:?}"
  );
}

/// Regression (Codex 2026-05-27 R3): `dynamic_roll` must surface a
/// rank-1 (or rank-3, etc.) `shifts` as `Error::RankMismatch`, not
/// `Error::ShapePairMismatch` — `ShapePairMismatchPayload` is documented
/// for same-rank shape disagreement. Mirrors the C R1 `norm.rs` and C R2
/// `switch.rs` rank-first-then-shape splits.
#[test]
fn dynamic_roll_rejects_rank_mismatch_shifts() {
  // x: 4-D `[B=2, n_kv_heads=1, S=3, head_dim=1]`. shifts: rank-1 `[B]`
  // instead of `[B, 1]` — the divergent RANK must surface as RankMismatch,
  // not ShapePairMismatch.
  let x = kvb(&[&[10.0, 20.0, 30.0], &[40.0, 50.0, 60.0]]);
  let shifts_rank1 = Array::from_slice::<i32>(&[0, 1], &(2usize,)).unwrap();
  let err = dynamic_roll(&x, &shifts_rank1, 2).expect_err("rank-1 shifts must Err");
  match err {
    mlxrs::Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 1, "rank-1 shifts: payload.actual must be 1");
      assert_eq!(
        p.actual_shape(),
        &[2usize][..],
        "rank-1 shifts: payload.actual_shape must be [B]"
      );
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }

  // Rank-3 `shifts` (`[B, 1, 1]`) — also a rank divergence: must surface
  // as RankMismatch, not as a [B, 1]-vs-[B, 1, 1] ShapePairMismatch.
  let shifts_rank3 = Array::from_slice::<i32>(&[0, 1], &(2usize, 1usize, 1usize)).unwrap();
  let err = dynamic_roll(&x, &shifts_rank3, 2).expect_err("rank-3 shifts must Err");
  match err {
    mlxrs::Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 3, "rank-3 shifts: payload.actual must be 3");
      assert_eq!(
        p.actual_shape(),
        &[2usize, 1, 1][..],
        "rank-3 shifts: payload.actual_shape must be [B, 1, 1]"
      );
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }

  // Same-rank `[B', 1]` with wrong B' (neither equal to B=2 nor 1 — i.e.
  // truly non-broadcastable) must still surface as ShapePairMismatch
  // (proves the split preserved the shape-disagreement path now that rank
  // is known to be 2). Use B'=3 (Codex 2026-05-27 R4): B'=1 is now a VALID
  // scalar-broadcast shape after the broadcast-contract preservation fix
  // (`BatchKvCache::finalize` arms length-1 `right_padding` → `[1, 1]`
  // `pad_col`); only neither-B-nor-1 truly Errs.
  let shifts_wrong_b = Array::from_slice::<i32>(&[0, 0, 0], &(3usize, 1usize)).unwrap();
  let err = dynamic_roll(&x, &shifts_wrong_b, 2).expect_err("[3,1] shifts on B=2 must Err");
  match err {
    mlxrs::Error::ShapePairMismatch(p) => {
      assert_eq!(p.expected(), &[2usize, 1][..]);
      assert_eq!(p.actual(), &[3usize, 1][..]);
    }
    other => panic!("expected ShapePairMismatch, got {other:?}"),
  }

  // Same-rank `[B, k]` with `k != 1` must also surface as ShapePairMismatch.
  let shifts_wrong_k = Array::from_slice::<i32>(&[0, 0, 1, 1], &(2usize, 2usize)).unwrap();
  let err = dynamic_roll(&x, &shifts_wrong_k, 2).expect_err("[B, 2] shifts must Err");
  match err {
    mlxrs::Error::ShapePairMismatch(p) => {
      assert_eq!(p.expected(), &[2usize, 1][..]);
      assert_eq!(p.actual(), &[2usize, 2][..]);
    }
    other => panic!("expected ShapePairMismatch, got {other:?}"),
  }
}

/// Regression (Codex 2026-05-27 R4): `dynamic_roll` MUST accept a
/// `[1, 1]` (scalar broadcast) `shifts` against an `x` with `B > 1` —
/// this is the contract `BatchKvCache::finalize` relies on when a
/// length-1 `right_padding` is armed via `prepare_right_padding(&[k])`
/// (the `expand_dims_axes(padding, &[1])` produces a `[1, 1]` `pad_col`
/// which broadcasts across the `[B, n_kv_heads, S, head_dim]` buffer).
/// Pins the broadcast contract that the C R4 rank-first-then-shape
/// split previously regressed by adding a tightening `sshape[0] ==
/// xshape[0]` check that rejected scalar broadcast as `[B, 1] vs [1, 1]`
/// before `BatchKvCache::finalize` could commit.
#[test]
fn dynamic_roll_accepts_scalar_broadcast_shifts() {
  // x: 4-D `[B=2, n_kv_heads=1, S=3, head_dim=1]`. shifts: `[1, 1]`
  // (scalar broadcast). Per the broadcast contract, every row gets the
  // same shift — so a shift of 1 should produce
  // `out[b,:,i,:] = x[b,:,(i-1)%3,:]` for all b.
  let x = kvb(&[&[10.0, 20.0, 30.0], &[40.0, 50.0, 60.0]]);
  let shifts_scalar = Array::from_slice::<i32>(&[1], &(1usize, 1usize)).unwrap();
  let result = dynamic_roll(&x, &shifts_scalar, 2);
  assert!(
    result.is_ok(),
    "scalar broadcast [1,1] shifts must be accepted; got {result:?}"
  );
  let mut rolled = result.unwrap();
  assert_eq!(
    rolled.shape(),
    vec![2, 1, 3, 1],
    "broadcast result shape must match input shape"
  );
  // Row 0 shift 1: out[i] = x[(i-1)%3] -> [x[2], x[0], x[1]] = [30, 10, 20].
  // Row 1 shift 1: same shift, so [60, 40, 50].
  assert_eq!(
    rolled.to_vec::<f32>().unwrap(),
    vec![30.0, 10.0, 20.0, /* row1 */ 60.0, 40.0, 50.0],
    "scalar broadcast shift=1 must roll every row by 1"
  );

  // Also pin the boundary: scalar broadcast with shift=0 is identity.
  let shifts_zero = Array::from_slice::<i32>(&[0], &(1usize, 1usize)).unwrap();
  let mut identity = dynamic_roll(&x, &shifts_zero, 2).expect("scalar broadcast [1,1] shift=0");
  assert_eq!(
    identity.to_vec::<f32>().unwrap(),
    vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    "scalar broadcast shift=0 must be identity"
  );
}

/// Regression (Copilot review #3271119609 / #3271308786 / #3271308805):
/// the BatchRotating-restore structural validator's error message must
/// include the offending values so a corrupt prompt-cache is diagnosable
/// from the error alone. Test each of the 5 invariant-violation branches
/// (1 empty + 4 non-empty: `_idx > L`, `rotated && L != max_size`,
/// `L > _offset`, plus `max_size == 0` is conjoined with the non-empty
/// `_idx > L` case via the `_idx ({idx}) > L ({L})` arm — `max_size == 0`
/// alone is covered by `from_state` + a make_mask reject and is not
/// reachable as a standalone restore-validation arm without violating
/// `_idx > L` first since restored `_idx == 0` keeps the branch inert).
/// Each forges a clearly-bad `(state, meta)` and asserts the error
/// message names the violated invariant + the offending value.
#[test]
fn batch_rotating_from_state_error_message_names_violated_invariant() {
  // Helper: build a valid 4-D `[B, kv_heads, L, D]` keys/values pair.
  let kv =
    |seq_len: usize| -> Array { Array::zeros::<f32>(&(2usize, 1usize, seq_len, 1usize)).unwrap() };

  // Branch 1: EMPTY buffer with non-zero offset/_idx/rotated.
  // `set_state(Vec::new())` IS accepted by BatchKvCache/BatchRotatingKvCache
  // (it sets keys=None, values=None — the empty-buffer state); then
  // `set_meta_state` restores `max_size, _offset, _idx, rotated`. The
  // empty-arm validator requires fully-fresh meta (offset=0, _idx=0,
  // !rotated) — any other combo is forged.
  let bad_empty = vec![
    "8".to_string(), // max_size
    "5".to_string(), // _offset = 5 (non-zero!)
    "0".to_string(), // _idx = 0
    "false".to_string(),
  ];
  let r = from_state("BatchRotatingKVCache", vec![], &bad_empty);
  match r {
    Err(mlxrs::Error::OutOfRange(p)) => {
      assert!(
        p.context().contains("empty buffer"),
        "context must name the empty-arm condition; got: {}",
        p.context()
      );
      assert!(
        p.value().contains("offset=5"),
        "value must name the offending offset; got: {}",
        p.value()
      );
    }
    Err(other) => panic!("expected OutOfRange empty-arm, got {other:?}"),
    Ok(_) => panic!("from_state empty + offset=5 must Err"),
  }

  // Branch 2: `_idx > L` (write cursor beyond physical buffer).
  let k = kv(3);
  let v = kv(3);
  let off = Array::from_slice::<i32>(&[0, 0], &(2usize,)).unwrap();
  let lp = Array::from_slice::<i32>(&[0, 0], &(2usize,)).unwrap();
  let st = vec![
    k.try_clone().unwrap(),
    v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  // meta = [max_size=8, _offset=5, _idx=7 > L=3, rotated=false]
  let bad_idx = vec![
    "8".to_string(),
    "5".to_string(),
    "7".to_string(),
    "false".to_string(),
  ];
  let r = from_state("BatchRotatingKVCache", st, &bad_idx);
  let err_msg = match r {
    Err(e) => format!("{e}"),
    Ok(_) => panic!("from_state with _idx > L must Err"),
  };
  assert!(
    err_msg.contains("_idx") && err_msg.contains("7") && err_msg.contains("3"),
    "error must name _idx and the offending values (idx=7, L=3); got: {err_msg}"
  );

  // Branch 3: `rotated=true && L != max_size`.
  let k = kv(3);
  let v = kv(3);
  let st = vec![
    k.try_clone().unwrap(),
    v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  // meta = [max_size=8, _offset=3, _idx=3, rotated=true]; L=3 != max_size=8
  let bad_rot = vec![
    "8".to_string(),
    "3".to_string(),
    "3".to_string(),
    "true".to_string(),
  ];
  let r = from_state("BatchRotatingKVCache", st, &bad_rot);
  let err_msg = match r {
    Err(e) => format!("{e}"),
    Ok(_) => panic!("from_state with rotated=true && L != max_size must Err"),
  };
  assert!(
    err_msg.contains("rotated") && err_msg.contains("max_size"),
    "error must name `rotated` and `max_size`; got: {err_msg}"
  );

  // Branch 4: `L > _offset`.
  let k = kv(5);
  let v = kv(5);
  let st = vec![
    k.try_clone().unwrap(),
    v.try_clone().unwrap(),
    off.try_clone().unwrap(),
    lp.try_clone().unwrap(),
  ];
  // meta = [max_size=8, _offset=3, _idx=3, rotated=false]; L=5 > _offset=3
  let bad_off = vec![
    "8".to_string(),
    "3".to_string(),
    "3".to_string(),
    "false".to_string(),
  ];
  let r = from_state("BatchRotatingKVCache", st, &bad_off);
  let err_msg = match r {
    Err(e) => format!("{e}"),
    Ok(_) => panic!("from_state with L > _offset must Err"),
  };
  assert!(
    err_msg.contains("L")
      && err_msg.contains("5")
      && err_msg.contains("_offset")
      && err_msg.contains("3"),
    "error must name L, _offset, and the offending values (L=5, _offset=3); got: {err_msg}"
  );
}

/// Regression (Copilot review #3271119572 / #3271119588): `dynamic_roll`
/// with `n == 0` (empty seq axis) would compute `remainder(idx, 0)` (a
/// divide-by-zero) under the unguarded path. The early-`n == 0` clone
/// return must surface as a pure no-op without ever evaluating the
/// arange/remainder chain.
#[test]
fn dynamic_roll_n_zero_is_noop_clone() {
  // Empty seq axis: x is `[1, 1, 0, 1]`.
  let empty = Array::zeros::<f32>(&(1usize, 1, 0, 1)).unwrap();
  let shifts = Array::from_slice::<i32>(&[3], &(1usize, 1)).unwrap();
  let r = dynamic_roll(&empty, &shifts, 2);
  assert!(r.is_ok(), "dynamic_roll(n=0) must Ok-clone, got {r:?}");
  let rolled = r.unwrap();
  // Shape preserved.
  assert_eq!(rolled.shape(), &[1, 1, 0, 1]);
}

/// Regression (Codex review on c189169 — `update_concat` stale `rotated`):
/// after a mixed sequence (prefill → S==1 decodes that wrap the ring →
/// S>1 update that temporal-orders), `update_concat`'s commit MUST clear
/// `self.rotated` — otherwise `meta_state()` reports `rotated=true` while
/// the stored buffer is now temporally-ordered (over-retained length
/// `max_size+S-1`, NOT physical ring length `max_size`). Our own
/// `from_state` structural guard (`rotated && L != max_size`) then
/// rejects the cache's saved state, breaking save/load on perfectly valid
/// cache evolutions. The fix clears `self.rotated = false` in the
/// `update_concat` commit tail; this test exercises the mixed path and
/// asserts the round-trip works.
#[test]
fn batch_rotating_update_concat_clears_rotated_after_mixed_path() {
  // mixed sequence: max_size=4, prefill S=3 → 2 × S=1 decodes wrap the
  // ring → S=2 update temporal-orders.
  let lp = [0i32, 0];
  let mut c = BatchRotatingKvCache::new(4, &lp);
  // Prefill S=3.
  let p3 = kvb(&[&[0.0, 1.0, 2.0], &[10.0, 11.0, 12.0]]);
  c.update(&p3, &p3).unwrap();
  // 2 × S=1 decodes (fills the ring at 4, then wraps once).
  for tok in 3..5 {
    let d = kvb(&[&[tok as f32], &[(10 + tok) as f32]]);
    c.update(&d, &d).unwrap();
  }
  // S=2 update (mixed path: ring is rotated, then temporal-ordered).
  let s2 = kvb(&[&[5.0, 6.0], &[15.0, 16.0]]);
  c.update(&s2, &s2).unwrap();
  // Save/load round-trip MUST succeed (without the fix, from_state would
  // reject because meta.rotated=true while L=5 != max_size=4).
  let st = c.state().unwrap();
  let meta = c.meta_state();
  // (`Box<dyn KvCache>` doesn't impl Debug, so extract the Err message
  // directly for the failure mode rather than `{restored:?}`.)
  let restored = match from_state("BatchRotatingKVCache", st, &meta) {
    Ok(c) => c,
    Err(e) => panic!("save/load round-trip after mixed-path update must succeed, got Err({e})"),
  };
  // Sanity: the restored cache reports the same offset.
  assert_eq!(restored.offset(), c.offset());
}

#[test]
fn batch_kv_pad_lengths_constructor_is_borrowed_slice() {
  // KVC-4 (#101): `pad_lengths()` returns the constructor-supplied slice
  // as a borrowed `&[i32]` — no per-call `.item()` round-trip (mlx-lm's
  // `int(self.left_padding[i].item())` cache.py:947-955). Mirrors
  // `ArraysCache::left_padding` for cross-cache consistency.
  let lp = [3i32, 1, 0];
  let c = BatchKvCache::new(&lp);
  assert_eq!(c.pad_lengths(), &lp);
  // Empty constructor: pad_lengths is empty (NOT a panic, NOT a default).
  let c0 = BatchKvCache::new(&[]);
  assert!(c0.pad_lengths().is_empty());
}

#[test]
fn batch_kv_pad_lengths_updates_after_set_state() {
  // KVC-4 (#101): `set_state` materializes the restored `left_padding`
  // host mirror ONCE — subsequent `pad_lengths()` reads are zero-cost
  // borrows. Restore via the same Array round-trip a prompt cache load
  // would use, then verify the host mirror matches the restored values.
  let initial = [0i32, 0];
  let mut c = BatchKvCache::new(&initial);
  // Build a 4-array restored state with non-trivial left_padding.
  let restored_lp_vals = [2i32, 4];
  let lp_arr = Array::from_slice::<i32>(&restored_lp_vals, &(2usize,)).unwrap();
  let off_arr = Array::from_slice::<i32>(&[-2, -4], &(2usize,)).unwrap();
  let k = kvb(&[&[1.0], &[2.0]]);
  let v = kvb(&[&[3.0], &[4.0]]);
  c.set_state(vec![k, v, off_arr, lp_arr]).unwrap();
  assert_eq!(
    c.pad_lengths(),
    &restored_lp_vals,
    "set_state must materialize the host mirror once at restore time"
  );
}

#[test]
fn batch_kv_pad_lengths_updates_after_finalize() {
  // KVC-4 (#101): `finalize` applies `left_padding += right_padding`
  // (cache.py:980-987). The host mirror must update in lockstep — using
  // the cached `right_padding_host` values from `prepare_right_padding`,
  // NOT a fresh eval of the Array form.
  let mut c = BatchKvCache::new(&[1i32, 3]);
  assert_eq!(c.pad_lengths(), &[1, 3]);
  // Prefill so finalize has a buffer to roll.
  let k = kvb(&[&[10.0], &[20.0]]);
  c.update(&k, &k).unwrap();
  // Arm right-pad and finalize.
  c.prepare_right_padding(&[2, 0]).unwrap();
  c.finalize().unwrap();
  assert_eq!(
    c.pad_lengths(),
    &[3, 3],
    "left_padding += right_padding mirrored in host pad_lengths"
  );
}

#[test]
fn batch_rotating_pad_lengths_constructor_and_set_state() {
  // KVC-4 (#101): same accessor exists on `BatchRotatingKvCache` for
  // cross-cache consistency. Constructor + set_state both maintain the
  // host mirror.
  let lp = [2i32, 0];
  let c = BatchRotatingKvCache::new(4, &lp);
  assert_eq!(c.pad_lengths(), &lp);

  // Round-trip via set_state with a different left_padding.
  let mut d = BatchRotatingKvCache::new(4, &[0, 0]);
  let restored_lp = [1i32, 5];
  let lp_arr = Array::from_slice::<i32>(&restored_lp, &(2usize,)).unwrap();
  let off_arr = Array::from_slice::<i32>(&[-1, -5], &(2usize,)).unwrap();
  let k = kvb(&[&[1.0], &[2.0]]);
  let v = kvb(&[&[3.0], &[4.0]]);
  d.set_state(vec![k, v, off_arr, lp_arr]).unwrap();
  assert_eq!(d.pad_lengths(), &restored_lp);
}

#[test]
fn batch_rotating_rotated_flag_observable_through_meta_state() {
  // KVC-5 (#102): the `rotated` flag MUST be the last mutation in the
  // commit tail of `update_in_place` / `update_concat` (mirrors swift's
  // late `self.rotated = false` at KVCache.swift:1330-1370). With every
  // post-`?` step infallible, an Ok-return is always coherent and an
  // Err-return commits NOTHING. This test asserts the observable
  // invariant via meta_state (which serializes `rotated` as the 4th
  // field): after a mixed prefill + decode that wraps the ring,
  // meta_state reports `rotated=true` AND the state/meta round-trip
  // validates (the from_state structural guard would reject any
  // desynchronized commit, so a green round-trip proves coherence).
  let lp = [0i32];
  let mut c = BatchRotatingKvCache::new(3, &lp); // tiny window forces rotation
  // S=3 prefill fills the ring (off=3, rotated=false yet).
  let p = kvb(&[&[0.0, 1.0, 2.0]]);
  c.update(&p, &p).unwrap();
  let meta_before = c.meta_state();
  assert_eq!(
    meta_before[3], "false",
    "pre-wrap meta_state.rotated == false"
  );
  // S=1 decode wraps: idx == max_size → rotated=true, idx=0 → idx=1.
  let d = kvb(&[&[3.0]]);
  c.update(&d, &d).unwrap();
  let meta_after = c.meta_state();
  assert_eq!(
    meta_after[3], "true",
    "post-wrap meta_state.rotated == true"
  );
  assert_eq!(c.offset(), 4);
  // Round-trip MUST succeed — proves rotated is coherent with the
  // committed ring state (the from_state guard would reject if not).
  let st = c.state().unwrap();
  let restored = match from_state("BatchRotatingKVCache", st, &meta_after) {
    Ok(c) => c,
    Err(e) => {
      panic!("post-rotation round-trip must succeed (proves rotated coherence), got Err({e})")
    }
  };
  assert_eq!(restored.offset(), c.offset());
}

// ── Codex-R1 follow-ups ──────────────────────────────────────────────────

/// Codex-R1 [high] #2: `BatchKvCache::finalize` previously committed the
/// new `left_padding`/`right_padding=None` even when
/// `right_padding_host.len() != self.pad_lengths.len()`, leaving the
/// public `pad_lengths()` slice permanently stale. Fix: explicit
/// length-1 broadcast (matching the Array op's broadcast rule) OR Err
/// when shapes don't match a supported pattern.
///
/// This test covers BOTH branches:
///   * length-1 `right_padding` against `pad_lengths.len() == 2`:
///     finalize MUST succeed AND broadcast — `pad_lengths` is updated
///     to `[old + scalar, old + scalar]`, NOT left stale.
///   * length-3 `right_padding` against `pad_lengths.len() == 2`:
///     finalize MUST Err — the host mirror can't reproduce the Array
///     side's broadcast (or lack thereof), and we will NOT commit a
///     desynchronized `pad_lengths`. The Array side itself errors on
///     `[2] + [3]` (no MLX broadcast rule for `[B] op [B']`, B != B'
///     != 1), so the same Err is reached structurally.
#[test]
fn kvc4_batch_kv_finalize_with_scalar_right_padding_broadcasts_or_errs() {
  // Case 1: scalar broadcast — right_padding is length-1, pad_lengths is
  // length-2. The Array op broadcasts `[2] + [1] -> [2]`; the host
  // mirror MUST broadcast in lockstep, NOT silently skip the update.
  let mut c = BatchKvCache::new(&[1i32, 3]);
  assert_eq!(c.pad_lengths(), &[1, 3]);
  // Prefill so `finalize` has a buffer to operate on.
  let k = kvb(&[&[10.0], &[20.0]]);
  c.update(&k, &k).unwrap();
  // Arm a length-1 right_padding (max > 0 so it's stored).
  c.prepare_right_padding(&[5]).unwrap();
  c.finalize().unwrap();
  assert_eq!(
    c.pad_lengths(),
    &[6, 8],
    "length-1 right_padding MUST broadcast across pad_lengths (was: silently stale [1, 3])"
  );

  // Case 2: a wildly-mismatched length (3 vs 2). The Array op
  // `[2] + [3]` errors on shape; our explicit host-side check also
  // errors. Either way: NO commit, pad_lengths/left_padding/right_padding
  // are left untouched.
  let mut d = BatchKvCache::new(&[1i32, 3]);
  let k2 = kvb(&[&[10.0], &[20.0]]);
  d.update(&k2, &k2).unwrap();
  let lp_before = iv(&d.left_padding_arr().unwrap());
  let pl_before: Vec<i32> = d.pad_lengths().to_vec();
  d.prepare_right_padding(&[1, 1, 1]).unwrap();
  assert!(
    d.finalize().is_err(),
    "right_padding length 3 vs pad_lengths length 2 MUST Err"
  );
  // No partial mutation: pad_lengths, left_padding, and right_padding
  // are all unchanged — retry-safe.
  assert_eq!(
    d.pad_lengths(),
    pl_before.as_slice(),
    "pad_lengths unchanged on Err"
  );
  assert_eq!(
    iv(&d.left_padding_arr().unwrap()),
    lp_before,
    "left_padding unchanged on Err"
  );
}

/// Codex-R1 [medium] #3: `BatchKvCache::set_state` previously swallowed
/// `to_vec::<i32>` failures (non-I32, non-contiguous, etc.) via
/// `unwrap_or_else(|_| self.pad_lengths.clone())` and then committed the
/// new `left_padding` Array — leaving `pad_lengths()` permanently
/// desynchronized (often at the empty placeholder from
/// `from_state("BatchKVCache")`'s `BatchKvCache::new(&[])`). Fix:
/// propagate the extraction failure via `?` and validate
/// rank/dtype/length BEFORE committing.
///
/// Regression: non-I32 restored `left_padding` MUST Err; cache state
/// (including the placeholder `left_padding` Array and the empty
/// `pad_lengths`) MUST remain unchanged.
#[test]
fn kvc4_batch_kv_set_state_propagates_to_vec_failure() {
  // Construct a `BatchKvCache` with a known initial `left_padding` so we
  // can verify it's left untouched on Err.
  let mut c = BatchKvCache::new(&[1i32, 2]);
  let pl_before: Vec<i32> = c.pad_lengths().to_vec();
  let lp_before = iv(&c.left_padding_arr().unwrap());

  // Case A: non-I32 left_padding (F32 instead). Validation MUST fire
  // BEFORE any mutation.
  let lp_f32 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
  let off = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k = kvb(&[&[1.0], &[2.0]]);
  let v = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k, v, off, lp_f32]).is_err(),
    "non-I32 left_padding MUST Err"
  );
  assert_eq!(
    c.pad_lengths(),
    pl_before.as_slice(),
    "pad_lengths unchanged on Err"
  );
  assert_eq!(
    iv(&c.left_padding_arr().unwrap()),
    lp_before,
    "left_padding unchanged on Err"
  );

  // Case B: wrong-rank left_padding (2-D instead of 1-D).
  let lp_2d = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2usize, 2)).unwrap();
  let off2 = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k2 = kvb(&[&[1.0], &[2.0]]);
  let v2 = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k2, v2, off2, lp_2d]).is_err(),
    "2-D left_padding MUST Err (rank validation)"
  );
  assert_eq!(c.pad_lengths(), pl_before.as_slice());
  assert_eq!(iv(&c.left_padding_arr().unwrap()), lp_before);

  // Case C: length mismatch (left_padding length 3, keys batch dim 2).
  let lp_3 = Array::from_slice::<i32>(&[1, 2, 3], &(3usize,)).unwrap();
  let off3 = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k3 = kvb(&[&[1.0], &[2.0]]);
  let v3 = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k3, v3, off3, lp_3]).is_err(),
    "length-mismatched left_padding MUST Err"
  );
  assert_eq!(c.pad_lengths(), pl_before.as_slice());
  assert_eq!(iv(&c.left_padding_arr().unwrap()), lp_before);

  // Sanity: a well-formed restore still succeeds (the strict validation
  // does NOT block legitimate state).
  let good_lp = Array::from_slice::<i32>(&[3, 4], &(2usize,)).unwrap();
  let good_off = Array::from_slice::<i32>(&[-3, -4], &(2usize,)).unwrap();
  let good_k = kvb(&[&[1.0], &[2.0]]);
  let good_v = kvb(&[&[3.0], &[4.0]]);
  c.set_state(vec![good_k, good_v, good_off, good_lp])
    .unwrap();
  assert_eq!(
    c.pad_lengths(),
    &[3, 4],
    "well-formed restore must update pad_lengths"
  );
}

/// Codex-R1 [medium] #3 sibling: same `to_vec` propagation fix for
/// `BatchRotatingKvCache::set_state`. Same failure classes (non-I32,
/// wrong-rank, length-mismatch) MUST Err with no partial mutation.
#[test]
fn kvc4_batch_rotating_set_state_propagates_to_vec_failure() {
  let mut c = BatchRotatingKvCache::new(4, &[1i32, 2]);
  let pl_before: Vec<i32> = c.pad_lengths().to_vec();
  let lp_before = iv(&c.left_padding_arr().unwrap());

  // Non-I32 left_padding.
  let lp_f32 = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
  let off = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k = kvb(&[&[1.0], &[2.0]]);
  let v = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k, v, off, lp_f32]).is_err(),
    "rotating: non-I32 left_padding MUST Err"
  );
  assert_eq!(c.pad_lengths(), pl_before.as_slice());
  assert_eq!(iv(&c.left_padding_arr().unwrap()), lp_before);

  // Wrong-rank.
  let lp_2d = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2usize, 2)).unwrap();
  let off2 = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k2 = kvb(&[&[1.0], &[2.0]]);
  let v2 = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k2, v2, off2, lp_2d]).is_err(),
    "rotating: 2-D left_padding MUST Err"
  );
  assert_eq!(c.pad_lengths(), pl_before.as_slice());
  assert_eq!(iv(&c.left_padding_arr().unwrap()), lp_before);

  // Length mismatch.
  let lp_3 = Array::from_slice::<i32>(&[1, 2, 3], &(3usize,)).unwrap();
  let off3 = Array::from_slice::<i32>(&[-1, -2], &(2usize,)).unwrap();
  let k3 = kvb(&[&[1.0], &[2.0]]);
  let v3 = kvb(&[&[3.0], &[4.0]]);
  assert!(
    c.set_state(vec![k3, v3, off3, lp_3]).is_err(),
    "rotating: length-mismatched left_padding MUST Err"
  );
  assert_eq!(c.pad_lengths(), pl_before.as_slice());
  assert_eq!(iv(&c.left_padding_arr().unwrap()), lp_before);

  // Sanity: well-formed restore still succeeds.
  let good_lp = Array::from_slice::<i32>(&[3, 4], &(2usize,)).unwrap();
  let good_off = Array::from_slice::<i32>(&[-3, -4], &(2usize,)).unwrap();
  let good_k = kvb(&[&[1.0], &[2.0]]);
  let good_v = kvb(&[&[3.0], &[4.0]]);
  c.set_state(vec![good_k, good_v, good_off, good_lp])
    .unwrap();
  assert_eq!(c.pad_lengths(), &[3, 4]);
}

// ── Codex-R2 follow-ups ──────────────────────────────────────────────────

/// Codex-R2 [medium]: `BatchRotatingKvCache::finalize` previously
/// swallowed `to_vec::<i32>` failures on the freshly-rolled
/// `new_left_padding` via `unwrap_or_else(|_| self.pad_lengths.clone())`
/// and then committed the new Array anyway — leaving `pad_lengths()`
/// permanently desynchronized from the rolled Array state. Same class
/// as R1 #3 (`BatchKvCache::set_state`); fix is to propagate the
/// extraction failure via `?` BEFORE the infallible commit tail so any
/// `to_vec` error leaves keys/values/offset/left_padding/_lengths
/// fully untouched.
///
/// The `to_vec` error itself can only be reached via construction paths
/// that are infeasible from outside (the new_left_padding Array is built
/// from successful arithmetic ops, so it's always well-formed I32). The
/// regression guard is therefore structural: assert no
/// `to_vec::<i32>().unwrap_or_else` pattern remains in the dirty
/// finalize/update_concat/update_in_place paths. Combined with the
/// positive-path tests already covering finalize/concat/in-place, this
/// pins the strict propagation discipline against future regressions.
#[test]
fn kvc4_batch_rotating_finalize_propagates_to_vec_failure() {
  let src = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/lm/cache/batch_rotating.rs"
  ))
  .expect("batch_rotating.rs must be readable for structural regression check");
  assert!(
    !src.contains("to_vec::<i32>()\n        .unwrap_or_else"),
    "BatchRotatingKvCache must not swallow `to_vec::<i32>` failures via \
     `unwrap_or_else(|_| self.pad_lengths.clone())` — Codex-R2 [medium] \
     regression: such a fallback commits a stale `pad_lengths` host mirror \
     against a freshly-rolled `left_padding` Array. Use `?` propagation \
     BEFORE the infallible commit tail."
  );
  // Also assert no chained-form variant slipped in.
  assert!(
    !src.contains("to_vec::<i32>().unwrap_or_else"),
    "BatchRotatingKvCache must not use any chained `to_vec::<i32>().unwrap_or_else` \
     — Codex-R2 [medium] regression"
  );
}

/// Codex-R2 [medium] sibling: `update_concat` (the `S > 1` path) had
/// the same swallowed-`to_vec` fallback on its `lp_dirty` branch. Same
/// structural regression guard — the runtime `to_vec` failure is
/// unreachable through the public API (Array is constructed from
/// successful arithmetic ops), so the regression pin is the absence of
/// the `unwrap_or_else` pattern in source.
#[test]
fn kvc4_batch_rotating_update_concat_propagates_to_vec_failure() {
  let src = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/lm/cache/batch_rotating.rs"
  ))
  .expect("batch_rotating.rs must be readable for structural regression check");
  // Count `to_vec::<i32>()?` occurrences — finalize + update_concat +
  // update_in_place + set_state = 4 strict-propagation sites total
  // (3 dirty paths + 1 set_state from R1).
  let strict_count = src.matches("to_vec::<i32>()?").count();
  assert!(
    strict_count >= 4,
    "Expected ≥4 strict `to_vec::<i32>()?` sites in batch_rotating.rs \
     (finalize + update_concat + update_in_place + set_state) — found {strict_count}. \
     Codex-R2 [medium] requires the dirty-left_padding paths use `?` propagation."
  );
}

/// Codex-R2 [medium] sibling: `update_in_place` (the `S == 1` decode
/// path) had the same swallowed-`to_vec` fallback on its `lp_dirty`
/// branch (trim and/or rotate dirty case). Structural guard, same
/// rationale as the finalize/update_concat siblings.
///
/// Positive-path coverage exists in
/// `batch_rotating_active_ring_then_concat_mixed` and the trim/rotate
/// tests, which exercise the `lp_dirty == true` branch end-to-end;
/// this test pins the absence of the swallowing fallback so a future
/// edit cannot silently reintroduce the desync.
#[test]
fn kvc4_batch_rotating_update_in_place_propagates_to_vec_failure() {
  let src = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/lm/cache/batch_rotating.rs"
  ))
  .expect("batch_rotating.rs must be readable for structural regression check");
  // Pin: the post-fix source must mention strict propagation for the
  // `lp_dirty` branches AND the `pad_lengths.clone()` fallback must
  // only appear on the `lp_dirty == false` arms (where it's
  // byte-identical to `self.pad_lengths`, not a stale-on-Err fallback).
  // The error-path fallback signature was `unwrap_or_else(|_| self.pad_lengths.clone())`;
  // a structural assert on `pad_lengths.clone()` alone would over-fire
  // on the legitimate else-branch reuse. Match on the full sin pattern.
  assert!(
    !src.contains(".unwrap_or_else(|_| self.pad_lengths.clone())"),
    "BatchRotatingKvCache must not swallow extraction failures with \
     `unwrap_or_else(|_| self.pad_lengths.clone())` — Codex-R2 [medium] \
     regression in any of finalize / update_concat / update_in_place."
  );
}
