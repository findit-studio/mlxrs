use super::*;

/// A `[B, 1, S, 1]` KV tensor (one readable id per `[b, step]`), matching
/// the `tests/lm_cache_batch.rs` `kvb` helper so each retained-token
/// identity is directly readable from `to_vec`. All rows share `S`.
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

/// `[B]` `I32` -> `Vec<i32>` for asserting per-seq `offset` / `left_padding`.
fn iv(a: &Array) -> Vec<i32> {
  let mut a = a.try_clone().unwrap();
  a.to_vec::<i32>().unwrap()
}

// ── batch_head_dim: the generic-name `_ =>` context arm + rank error ─────

/// `batch_head_dim` on a non-4-D array with a name that is neither "keys"
/// nor "values" exercises the `_ =>` context arm (line 65) and the
/// `RankMismatch` return; a 4-D array returns `shape[-1]`.
#[test]
fn batch_head_dim_generic_name_rank_error_and_ok() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2usize)).unwrap();
  // name == "offset" hits the catch-all context arm (line 65).
  let err = batch_head_dim("offset", &bad).unwrap_err();
  assert!(
    matches!(err, Error::RankMismatch(_)),
    "non-4-D batch_head_dim must be RankMismatch, not panic"
  );
  // The two named arms (keys/values) and the success path.
  assert!(matches!(
    batch_head_dim("keys", &bad).unwrap_err(),
    Error::RankMismatch(_)
  ));
  assert!(matches!(
    batch_head_dim("values", &bad).unwrap_err(),
    Error::RankMismatch(_)
  ));
  let ok = kvb(&[&[1.0, 2.0, 3.0]]); // [1,1,3,1] -> head_dim 1
  assert_eq!(batch_head_dim("keys", &ok).unwrap(), 1);
}

// ── dynamic_roll: the rank guard (168-172) + the axis guard (174-178) ────

/// `dynamic_roll` on a non-4-D `x` is a recoverable `RankMismatch`
/// (lines 168-172), and a correct-rank `x` with a non-sequence `axis` is a
/// recoverable `OutOfRange` (lines 174-178) — never a panic.
#[test]
fn dynamic_roll_rank_and_axis_guards() {
  let shifts = Array::from_slice::<i32>(&[0], &(1usize, 1usize)).unwrap();
  // Non-4-D x -> RankMismatch (168-172).
  let bad_x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1usize, 3usize)).unwrap();
  assert!(matches!(
    dynamic_roll(&bad_x, &shifts, 2).unwrap_err(),
    Error::RankMismatch(_)
  ));
  // 4-D x but axis != KV_NDIM-2 (== 2) -> OutOfRange (174-178).
  let x = kvb(&[&[10.0, 20.0, 30.0]]);
  assert!(matches!(
    dynamic_roll(&x, &shifts, 1).unwrap_err(),
    Error::OutOfRange(_)
  ));
}

// ── empty_ivec: the success path (lines 473-474) ─────────────────────────

/// `empty_ivec` builds a real `[0]`-length `I32` array via `from_slice`
/// (lines 473-474; the `mlx_array_new` fallback at 481 is the unreachable
/// double-allocation-failure branch and is intentionally not exercised).
#[test]
fn empty_ivec_builds_zero_length_i32() {
  let mut a = empty_ivec();
  assert_eq!(a.shape(), vec![0usize]);
  assert_eq!(a.dtype().unwrap(), Dtype::I32);
  assert!(a.to_vec::<i32>().unwrap().is_empty());
}

// ── state_kv: the empty-cache InvariantViolation branch (461-463) ────────

/// `state_kv()` on a fresh (empty) cache returns the `InvariantViolation`
/// error branch (lines 461-463); after an update it returns the `_idx`-
/// sliced `(keys, values)` pair.
#[test]
fn state_kv_empty_errors_then_returns_pair() {
  let c = BatchKvCache::new(&[0, 0]);
  assert!(
    matches!(c.state_kv().unwrap_err(), Error::InvariantViolation(_)),
    "state_kv on an empty cache must be InvariantViolation, not panic"
  );
  let mut c = BatchKvCache::new(&[0, 0]);
  let p = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  c.update(&p, &p).unwrap();
  let (mut k, mut v) = c.state_kv().unwrap();
  assert_eq!(k.shape(), vec![2, 1, 2, 1]);
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
  assert_eq!(v.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ── nbytes: the Some(keys) (783) + Some(values) (786) accumulation ───────

/// `nbytes()` is 0 when empty and the sum of the two buffers' byte sizes
/// once populated (lines 783, 786). The exact byte count is an independent
/// closed form: `B*H*S*D` f32 elements * 4 bytes, doubled for K and V.
#[test]
fn nbytes_sums_key_and_value_buffers() {
  let c = BatchKvCache::new(&[0, 0]);
  assert_eq!(c.nbytes(), 0, "empty cache has 0 bytes");
  let mut c = BatchKvCache::new(&[0, 0]);
  // [B=2, H=1, S=2, D=1] f32 -> 4 elements * 4 bytes = 16 bytes per buffer.
  let p = kvb(&[&[1.0, 2.0], &[3.0, 4.0]]);
  c.update(&p, &p).unwrap();
  let per_buffer_elems = 2 * 2; // B(2) * H(1) * S(2) * D(1)
  assert_eq!(
    c.nbytes(),
    2 * per_buffer_elems * std::mem::size_of::<f32>(),
    "keys.nbytes + values.nbytes (each 16 bytes here)"
  );
}

// ── materialize: keys/values/offset/left_padding/right_padding eval ──────

/// `materialize()` force-evals every live buffer the next chunk reuses:
/// keys (573-575), values (576-578), offset (579), left_padding (580), and
/// the pending right_padding (581-583). State is observably unchanged after.
#[test]
fn materialize_evals_all_live_buffers() {
  let mut c = BatchKvCache::new(&[1, 0]);
  let p = kvb(&[&[5.0, 6.0], &[7.0, 8.0]]);
  c.update(&p, &p).unwrap();
  // Arm a right_padding so the `Some(rp)` materialize arm (581-583) runs.
  c.prepare_right_padding(&[1, 1]).unwrap();
  c.materialize().unwrap();
  // Materialize is a pure memory barrier: the observable state is identical.
  // Scalar offset() == _idx == S(2); per-seq batch_offset = -[1,0] + 2 = [1,2].
  assert_eq!(c.offset(), 2);
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![1, 2]);
  let (mut k, _) = c.state_kv().unwrap();
  assert_eq!(k.to_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);

  // Empty-cache materialize: keys/values/right_padding are all None, so only
  // the offset/left_padding evals run (573/576/581 false branches).
  let mut empty = BatchKvCache::new(&[0]);
  empty.materialize().unwrap();
  assert!(empty.is_empty());
}

// ── update: the `_idx + S` overflow checked_add closure (527-530) ────────

/// A corrupt/hostile `_idx == usize::MAX` makes `update`'s `_idx + S`
/// overflow; the `checked_add` closure (lines 527-530) surfaces it as a
/// recoverable `ArithmeticOverflow` with NO partial mutation (the cache is
/// left exactly as it was). Built via a struct literal because `set_state`
/// derives `_idx` from `keys.shape[2]` and cannot inject a hostile value.
#[test]
fn update_idx_overflow_is_rejected_without_partial_mutation() {
  let stored = kvb(&[&[1.0]]); // [B=1,H=1,S=1,D=1]
  let lp = ivec(&[0]).unwrap();
  let off = ivec(&[5]).unwrap();
  let mut c = BatchKvCache {
    keys: Some(stored.try_clone().unwrap()),
    values: Some(stored.try_clone().unwrap()),
    left_padding: lp,
    pad_lengths: vec![0],
    offset: off,
    idx: usize::MAX,
    right_padding: None,
    right_padding_host: None,
  };
  let upd = kvb(&[&[2.0]]); // S == 1 -> idx + 1 overflows usize::MAX
  let err = c.update(&upd, &upd).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "_idx + S overflow must be a recoverable ArithmeticOverflow"
  );
  // No partial mutation: _idx and the per-seq offset are unchanged.
  assert_eq!(c.offset(), usize::MAX, "_idx unchanged on the Err path");
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![5], "offset unchanged");
}

// ── trim: the trimmed==0 early return (731) + the keys=None arm (743) ────

/// `trim(0)` and a trim of an empty cache hit the `trimmed == 0` early
/// return (line 731).
#[test]
fn trim_zero_is_noop_early_return() {
  let mut c = BatchKvCache::new(&[0, 0]);
  let p = kvb(&[&[1.0], &[2.0]]);
  c.update(&p, &p).unwrap();
  assert_eq!(c.trim(0).unwrap(), 0, "trim(0) returns 0 immediately");
  assert_eq!(c.offset(), 1, "offset untouched by trim(0)");
  // Empty cache: n.min(_idx=0) == 0 -> same early return.
  let mut empty = BatchKvCache::new(&[0]);
  assert_eq!(empty.trim(3).unwrap(), 0);
}

/// `trim` with a positive `_idx` but `keys == None` exercises the sliced
/// `_ => None` match arm (line 743): `_idx`/`offset` still decrement, but
/// there is no buffer to slice. Built via a struct literal (a real
/// populated cache always has keys=Some when `_idx > 0`).
#[test]
fn trim_with_no_buffer_decrements_idx_and_offset() {
  let lp = ivec(&[0, 0]).unwrap();
  let off = ivec(&[5, 5]).unwrap();
  let mut c = BatchKvCache {
    keys: None,
    values: None,
    left_padding: lp,
    pad_lengths: vec![0, 0],
    offset: off,
    idx: 5,
    right_padding: None,
    right_padding_host: None,
  };
  assert_eq!(c.trim(2).unwrap(), 2, "trimmed = min(2, _idx=5)");
  assert_eq!(c.offset(), 3, "_idx 5 -> 3");
  // offset -= 2: [5,5] -> [3,3] (the array path runs even with no buffer).
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![3, 3]);
  assert!(c.is_empty(), "keys stayed None (the `_ => None` slice arm)");
}

// ── finalize: the right_padding_host None arm (400) + rolled None (431) ──

/// `finalize` with `right_padding = Some` but `right_padding_host = None`
/// exercises the `None => self.pad_lengths.clone()` arm (line 400); with
/// `keys/values == None` the rolled match also hits its `_ => None` arm
/// (line 431). offset/left_padding still update from the `[B]` padding.
#[test]
fn finalize_with_none_host_mirror_and_no_buffer() {
  let lp = ivec(&[0, 0]).unwrap();
  let off = ivec(&[4, 4]).unwrap();
  let padding = ivec(&[1, 2]).unwrap();
  let mut c = BatchKvCache {
    keys: None,
    values: None,
    left_padding: lp,
    pad_lengths: vec![0, 0],
    offset: off,
    idx: 0,
    // right_padding armed but its host mirror deliberately None (the
    // line-400 arm: pad_lengths is preserved as-is, no host-side add).
    right_padding: Some(padding),
    right_padding_host: None,
  };
  c.finalize().unwrap();
  // rolled hit `_ => None` (no keys/values); offset -= padding -> [3,2];
  // left_padding += padding -> [1,2]. pad_lengths cloned unchanged ([0,0]).
  assert_eq!(
    iv(&c.batch_offset().unwrap()),
    vec![3, 2],
    "offset -= padding"
  );
  assert_eq!(
    iv(&c.left_padding_arr().unwrap()),
    vec![1, 2],
    "lp += padding"
  );
  assert_eq!(
    c.pad_lengths(),
    &[0, 0],
    "None host mirror -> pad_lengths preserved (line 400 arm)"
  );
  // right_padding cleared in the commit tail: a second finalize is a no-op.
  c.finalize().unwrap();
  assert_eq!(iv(&c.batch_offset().unwrap()), vec![3, 2]);
}

/// The faithful `prepare_right_padding` + `finalize` path with NO buffer:
/// host mirror is Some (len == pad_lengths) so the elementwise-add arm
/// (401-406) runs and `pad_lengths` is refreshed, while rolled is `_ =>
/// None` (431) since keys/values were never written.
#[test]
fn finalize_no_buffer_refreshes_host_mirror() {
  let mut c = BatchKvCache::new(&[0, 0]);
  c.prepare_right_padding(&[1, 2]).unwrap();
  c.finalize().unwrap();
  assert_eq!(iv(&c.left_padding_arr().unwrap()), vec![1, 2]);
  assert_eq!(
    c.pad_lengths(),
    &[1, 2],
    "host mirror updated elementwise (B==B arm)"
  );
}

// ── copy: every Some(arr) clone arm (802-820) ────────────────────────────

/// `copy()` deep-copies a fully-populated cache (keys + values + a pending
/// right_padding all Some), exercising every `Some(a) => Some(a.try_clone())`
/// arm and the scalar/Vec field clones (lines 802-820). The copy is an
/// independent, equal cache.
#[test]
fn copy_clones_all_buffers_independently() {
  let mut c = BatchKvCache::new(&[1, 0]);
  let p = kvb(&[&[10.0, 20.0], &[30.0, 40.0]]);
  c.update(&p, &p).unwrap();
  c.prepare_right_padding(&[1, 1]).unwrap(); // arm right_padding (Some arm)

  let mut copied = c.copy().unwrap();
  // Trait-level observables match the source. Scalar offset() == _idx == S(2);
  // the per-seq batch_offset is -left_padding + S = [-1,0] + 2 = [1, 2].
  assert_eq!(copied.offset(), 2);
  assert!(!copied.is_empty());
  assert_eq!(copied.nbytes(), c.nbytes());
  assert_eq!(
    iv(
      &copied
        .as_batch_positioned()
        .unwrap()
        .batch_offset()
        .unwrap()
    ),
    vec![1, 2]
  );
  let st = copied.state().unwrap();
  assert_eq!(st.len(), 4, "[keys, values, offset, left_padding]");
  let mut k = st[0].try_clone().unwrap();
  assert_eq!(
    k.to_vec::<f32>().unwrap(),
    vec![10.0, 20.0, 30.0, 40.0],
    "copied keys are an exact independent duplicate"
  );

  // Independence: an empty `set_state` on the COPY resets its per-seq
  // offset, and must NOT perturb the ORIGINAL's offset/left_padding (MLX
  // value semantics — try_clone shares refcounts but the cache only ever
  // reassigns its arrays, so the source is fully decoupled).
  let before_off = iv(&c.batch_offset().unwrap());
  let before_lp = iv(&c.left_padding_arr().unwrap());
  copied.set_state(Vec::new()).unwrap();
  assert!(copied.is_empty(), "copy reset independently");
  assert_eq!(
    iv(&c.batch_offset().unwrap()),
    before_off,
    "original offset untouched by mutating the copy"
  );
  assert_eq!(
    iv(&c.left_padding_arr().unwrap()),
    before_lp,
    "original left_padding untouched by mutating the copy"
  );
}

// ── copy of an empty cache: the `None` clone arms ────────────────────────

/// A copy of an EMPTY cache (keys/values/right_padding all None) takes the
/// `None` arms of the same `copy` matches and yields an empty, valid cache.
#[test]
fn copy_of_empty_cache_is_empty() {
  let c = BatchKvCache::new(&[2, 0, 1]);
  let copied = c.copy().unwrap();
  assert!(copied.is_empty());
  assert_eq!(copied.offset(), 0);
  assert_eq!(copied.nbytes(), 0);
  assert!(copied.state().unwrap().is_empty());
  assert_eq!(
    iv(
      &copied
        .as_batch_positioned()
        .unwrap()
        .batch_offset()
        .unwrap()
    ),
    vec![-2, 0, -1],
    "copied empty cache preserves -left_padding"
  );
}

// ── create_causal_mask_batched: overflow (911-914), window (943-952),
//    right_padding (966-977) ───────────────────────────────────────────

/// `create_causal_mask_batched` rejects an `offset + N` overflow with a
/// recoverable `ArithmeticOverflow` (lines 910-914) — never an overflow
/// panic / silent wrong mask.
#[test]
fn causal_mask_batched_offset_overflow_is_rejected() {
  let err = create_causal_mask_batched(1, usize::MAX, None, None, None).unwrap_err();
  assert!(
    matches!(err, Error::ArithmeticOverflow(_)),
    "offset + N overflow must be ArithmeticOverflow, not panic"
  );
}

/// The windowed term (lines 943, 949-952): a `window_size < total` ANDs in
/// `linds < rinds + window_size`, banding the causal mask. Closed-form
/// oracle: offset 0, N 4 -> linds = rinds = [0,1,2,3]; causal lower-tri
/// then ALSO require `linds < rinds + 2`, i.e. keep only entries within 2
/// of the diagonal.
#[test]
fn causal_mask_batched_windowed_term() {
  // No left/right padding so we isolate the windowed band. With no [B,1,1,1]
  // padding term to broadcast against, the result stays the 2-D [N, total]
  // causal grid (the batch dim is added only by the left/right_padding terms).
  let mut m = create_causal_mask_batched(4, 0, Some(2), None, None).unwrap();
  assert_eq!(m.shape(), vec![4, 4], "no batch term -> [N, total]");
  let bits: Vec<u8> = m
    .to_vec::<bool>()
    .unwrap()
    .into_iter()
    .map(|b| b as u8)
    .collect();
  // mask[l][r] = (l >= r) && (l < r + 2):
  //   l0: r0 (0>=0 & 0<2) ; r>=1 -> 0           -> [1,0,0,0]
  //   l1: r0 (1<2? yes)=1, r1=1, r>=2 ->0        -> [1,1,0,0]
  //   l2: r0 (2<2? no)=0, r1 (2<3)=1, r2=1, r3=0 -> [0,1,1,0]
  //   l3: r1 (3<3? no)=0, r2 (3<4)=1, r3=1       -> [0,0,1,1]
  assert_eq!(
    bits,
    vec![
      1, 0, 0, 0, // l0
      1, 1, 0, 0, // l1
      0, 1, 1, 0, // l2
      0, 0, 1, 1, // l3
    ]
  );
  // window_size >= total is the no-op (the term is skipped): identical to
  // an unwindowed causal mask.
  let mut full = create_causal_mask_batched(4, 0, Some(99), None, None).unwrap();
  let full_bits: Vec<u8> = full
    .to_vec::<bool>()
    .unwrap()
    .into_iter()
    .map(|b| b as u8)
    .collect();
  assert_eq!(
    full_bits,
    vec![
      1, 0, 0, 0, // l0
      1, 1, 0, 0, // l1
      1, 1, 1, 0, // l2
      1, 1, 1, 1, // l3
    ],
    "window_size >= total is a no-op (plain causal)"
  );
}

/// The right_padding term (lines 966, 972-977): masks out the right-padded
/// tail columns via `rinds < (offset + N) - right_padding`. Closed-form
/// oracle: B=2, offset 0, N 3 -> total 3; right_padding [0,1] -> per-row
/// column bound [3, 2]; AND with the plain causal lower-triangle.
#[test]
fn causal_mask_batched_right_padding_term() {
  let rp = ivec(&[0, 1]).unwrap();
  let mut m = create_causal_mask_batched(3, 0, None, Some(&rp), None).unwrap();
  assert_eq!(
    m.shape(),
    vec![2, 1, 3, 3],
    "right_padding -> [B,1,N,total]"
  );
  let bits: Vec<u8> = m
    .to_vec::<bool>()
    .unwrap()
    .into_iter()
    .map(|b| b as u8)
    .collect();
  // causal lower-tri (l>=r), total 3:
  //   l0 [1,0,0]; l1 [1,1,0]; l2 [1,1,1]
  // row0 bound 3 -> rinds<3 keeps all -> unchanged.
  // row1 bound 2 -> rinds<2 zeroes column r==2 (already 0 in causal):
  //   l0 [1,0,0]; l1 [1,1,0]; l2 [1,1,0]  (l2,r2 cleared)
  assert_eq!(
    bits,
    vec![
      1, 0, 0, 1, 1, 0, 1, 1, 1, // batch row 0 (bound 3)
      1, 0, 0, 1, 1, 0, 1, 1, 0, // batch row 1 (bound 2, col r2 cleared)
    ]
  );
}
