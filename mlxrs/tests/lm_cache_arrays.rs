//! Deterministic tests for [`mlxrs::lm::cache::ArraysCache`], hand-traced
//! 1:1 from `mlx_lm.models.cache.ArraysCache`
//! (`mlx_lm/models/cache.py:594-730`, the authoritative spec) and
//! cross-checked against mlx-swift-lm's `MLXLMCommon` `ArraysCache` /
//! `MambaCache` (`KVCache.swift:1102` / `:1230`).
//!
//! `ArraysCache` is the generic *slot* cache SSM (Mamba-style) models use
//! via `[]` / `state` — it holds opaque per-slot state, NOT 4-D K/V, so its
//! trait `update` is an error (mlx-lm `ArraysCache` has no
//! `update_and_fetch`).

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  error::Error,
  lm::cache::{ArraysCache, KvCache, MAX_SLOT_COUNT, MaskMode, RopeOffset, from_state},
};

/// A `[1, 1, S, 1]`-ish slot tensor whose values are directly readable from
/// `to_vec` (slots are opaque to `ArraysCache`; any shape is fine).
fn slot(vals: &[f32]) -> Array {
  Array::from_slice::<f32>(vals, &(1usize, 1, vals.len(), 1)).unwrap()
}

/// A `[B, ...]` slot whose leading axis is `b` (for `batch_size` inference).
fn slot_b(b: usize) -> Array {
  Array::from_slice::<f32>(&vec![0.0f32; b], &(b, 1usize)).unwrap()
}

#[test]
fn new_is_empty_offset_nbytes() {
  // cache.py:601-602 `self.cache = [None] * size`; :723-724
  // `empty(): self.cache[0] is None`; :726-728 `nbytes = 0` (all None).
  // No `offset`/`size()` override -> `_BaseCache.size()` == 0.
  let c = ArraysCache::new(2);
  assert!(c.is_empty());
  assert_eq!(c.offset(), 0);
  assert_eq!(c.nbytes(), 0);
  // No state yet (Swift `compactMap` / Python all-None list serializes to
  // nothing through the `Vec<Array>` trait state).
  assert!(c.state().unwrap().is_empty());
  assert!(c.get(0).is_none());
  assert!(c.get(1).is_none());
}

#[test]
fn set_get_slots() {
  // cache.py:618-622 `__setitem__` / `__getitem__`.
  let mut c = ArraysCache::new(2);
  let a = slot(&[1.0, 2.0, 3.0]);
  let b = slot(&[4.0, 5.0]);
  c.set(0, a).unwrap();
  c.set(1, b).unwrap();

  assert!(!c.is_empty()); // cache[0] is not None
  let mut g0 = c.get(0).unwrap().try_clone().unwrap();
  let mut g1 = c.get(1).unwrap().try_clone().unwrap();
  assert_eq!(g0.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
  assert_eq!(g1.to_vec::<f32>().unwrap(), vec![4.0, 5.0]);

  // Out-of-range slot is a recoverable error, never a panic (Python would
  // `IndexError`).
  assert!(c.get(2).is_none());
  assert!(c.set(5, slot(&[0.0])).is_err());
}

#[test]
fn state_and_set_state_round_trip() {
  // cache.py:624-630 `state` getter / setter. With every slot populated,
  // the present-only (compacted) `Vec<Array>` state IS the full list, so a
  // bare `set_state` round-trip is exact even without the slot metadata
  // (the sparse case is covered by `sparse_slot_state_round_trips`).
  let mut c = ArraysCache::new(2);
  c.set(0, slot(&[1.0, 2.0])).unwrap();
  c.set(1, slot(&[3.0, 4.0])).unwrap();

  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);

  let mut c2 = ArraysCache::new(2);
  c2.set_state(st).unwrap();
  let mut r0 = c2.get(0).unwrap().try_clone().unwrap();
  let mut r1 = c2.get(1).unwrap().try_clone().unwrap();
  assert_eq!(r0.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
  assert_eq!(r1.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);

  // cache.py:628-630 setter replaces the whole list (size becomes len(v)).
  let mut c3 = ArraysCache::new(2);
  c3.set_state(vec![slot(&[9.0])]).unwrap();
  assert_eq!(c3.state().unwrap().len(), 1);
  assert!(c3.get(1).is_none());

  // Empty state resets to no slots.
  c3.set_state(Vec::new()).unwrap();
  assert!(c3.is_empty());
  assert!(c3.state().unwrap().is_empty());
}

#[test]
fn batch_size_inference() {
  // cache.py:606-616.
  // 1) first non-None slot -> c.shape[0].
  let mut c = ArraysCache::new(2);
  c.set(1, slot_b(4)).unwrap(); // slot 0 None, slot 1 present, batch 4
  assert_eq!(c.batch_size().unwrap(), 4);

  // 2) no slots, left_padding set -> left_padding.size.
  let c2 = ArraysCache::with_left_padding(2, &[1, 0, 1]);
  assert_eq!(c2.batch_size().unwrap(), 3);

  // 3) no slots / no left_padding, lengths set -> lengths.size.
  let mut c3 = ArraysCache::new(2);
  c3.prepare(&[5, 6]);
  assert_eq!(c3.batch_size().unwrap(), 2);

  // 4) nothing -> 1.
  let c4 = ArraysCache::new(2);
  assert_eq!(c4.batch_size().unwrap(), 1);
}

#[test]
fn make_mask_left_padding() {
  // cache.py:691-694: left_padding -> `mx.arange(N) >= left_padding[:,None]`.
  // left_padding=[1,0], N=3 -> row0 [F,T,T], row1 [T,T,T], shape [2,3].
  let c = ArraysCache::with_left_padding(2, &[1, 0]);
  match c.make_mask(3, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.shape(), vec![2, 3]);
      assert_eq!(
        m.to_vec::<bool>().unwrap(),
        vec![false, true, true, true, true, true]
      );
    }
    _ => panic!("ArraysCache with left_padding must return MaskMode::Array"),
  }
}

#[test]
fn make_mask_lengths() {
  // cache.py:695-697: elif lengths -> `mx.arange(N) < lengths[:,None]`.
  // lengths=[2,3], N=3 -> row0 [T,T,F], row1 [T,T,T].
  let mut c = ArraysCache::new(2);
  c.prepare(&[2, 3]);
  match c.make_mask(3, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.shape(), vec![2, 3]);
      assert_eq!(
        m.to_vec::<bool>().unwrap(),
        vec![true, true, false, true, true, true]
      );
    }
    _ => panic!("ArraysCache with lengths must return MaskMode::Array"),
  }
  // window_size / return_array are ignored by ArraysCache.make_mask (the
  // reference takes only N) — same result regardless.
  assert!(matches!(
    c.make_mask(3, Some(1), true).unwrap(),
    MaskMode::Array(_)
  ));
}

#[test]
fn make_mask_none() {
  // cache.py:698-699: neither left_padding nor lengths -> None.
  let c = ArraysCache::new(2);
  assert!(matches!(
    c.make_mask(4, None, false).unwrap(),
    MaskMode::None
  ));
}

#[test]
fn make_mask_arange_f32_boundary() {
  // Codex regression: `mx.arange(N)` must be integer-exact. This crate's
  // `Array::arange` is f32-only, so `N > 2^24` would silently round the
  // exclusive stop and return a WRONG-length mask. `make_mask` routes
  // through the guarded `mask::iarange`, so `N == 2^24` is accepted and
  // `N == 2^24 + 1` is a recoverable Err (never a shortened `Ok` mask) —
  // for BOTH the left_padding and lengths branches.
  const LIMIT: usize = 1 << 24; // 2^24, the exact f32 integer limit.

  let lp = ArraysCache::with_left_padding(1, &[0]);
  // At the limit: Ok (graph only, not materialized — cheap).
  assert!(matches!(
    lp.make_mask(LIMIT, None, false).unwrap(),
    MaskMode::Array(_)
  ));
  // Past the limit: Err, NOT a silently-truncated Ok mask.
  assert!(lp.make_mask(LIMIT + 1, None, false).is_err());

  let mut ln = ArraysCache::new(1);
  ln.prepare(&[0]);
  assert!(matches!(
    ln.make_mask(LIMIT, None, false).unwrap(),
    MaskMode::Array(_)
  ));
  assert!(ln.make_mask(LIMIT + 1, None, false).is_err());

  // No left_padding/lengths: `mx.arange` is NOT evaluated (cache.py:698-699
  // `else: return None`), so even a huge N is `None`, never an error.
  let none = ArraysCache::new(2);
  assert!(matches!(
    none.make_mask(LIMIT + 1, None, false).unwrap(),
    MaskMode::None
  ));
}

#[test]
fn prepare_advance_finalize() {
  // cache.py:678-689. prepare sets lengths; advance subtracts N from
  // lengths/left_padding; finalize clears both.
  let mut c = ArraysCache::with_left_padding(2, &[3, 5]);
  c.prepare(&[4, 6]);
  c.advance(2).unwrap(); // lengths -> [2,4], left_padding -> [1,3]

  // lengths now [2,4]: make_mask N=3 -> row0 [T,T,F], row1 [T,T,T]
  // (left_padding is present too, but cache.py:691 checks left_padding
  // FIRST: left_padding=[1,3] -> row0 [F,T,T], row1 [F,F,F]).
  match c.make_mask(3, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(
        m.to_vec::<bool>().unwrap(),
        vec![false, true, true, false, false, false]
      );
    }
    _ => panic!("expected array mask"),
  }

  c.finalize();
  assert!(matches!(
    c.make_mask(3, None, false).unwrap(),
    MaskMode::None
  ));
}

#[test]
fn update_returns_err_not_kv() {
  // mlx-lm `ArraysCache` has NO `update_and_fetch`: it is a generic slot
  // cache, not K/V. The trait `update` must be a recoverable error.
  // Surfaced as `Error::Backend` ("unsupported operation"), NOT
  // `ShapeMismatch` — the condition isn't a wrong-shaped tensor (Copilot
  // review #3271124426).
  let mut c = ArraysCache::new(2);
  let a = slot(&[1.0]);
  let err = c.update(&a, &a).unwrap_err();
  assert!(matches!(err, Error::Backend { .. }));
}

#[test]
fn nbytes_sums_present_slots() {
  // cache.py:726-728 `sum(c.nbytes for c in cache if c is not None)`.
  // f32 slot of 3 elems = 12 bytes; left_padding/lengths NOT counted.
  let mut c = ArraysCache::with_left_padding(2, &[0, 0]);
  assert_eq!(c.nbytes(), 0);
  c.set(0, slot(&[1.0, 2.0, 3.0])).unwrap(); // 3 * 4 = 12
  c.set(1, slot(&[4.0, 5.0])).unwrap(); // 2 * 4 = 8
  assert_eq!(c.nbytes(), 20);
}

#[test]
fn copy_is_deep_and_independent() {
  // mlx-lm `copy.deepcopy` / swift `copy()` (KVCache.swift:1130): an
  // independent cache with the same slots / left_padding.
  let mut c = ArraysCache::with_left_padding(2, &[1, 0]);
  c.set(0, slot(&[1.0, 2.0])).unwrap();
  c.set(1, slot(&[3.0, 4.0])).unwrap();

  let d = c.copy().unwrap();
  // Mutating the original's slots must not affect the copy.
  c.set(0, slot(&[9.0, 9.0])).unwrap();

  let st = d.state().unwrap();
  assert_eq!(st.len(), 2);
  let mut s0 = st[0].try_clone().unwrap();
  assert_eq!(s0.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);

  // left_padding survived the copy: mask still computed from [1,0].
  match d.make_mask(2, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.to_vec::<bool>().unwrap(), vec![false, true, true, true]);
    }
    _ => panic!("expected array mask from copied left_padding"),
  }
}

#[test]
fn rope_offset_and_defaults() {
  // No `offset`/batch-positioned override -> scalar 0; trait defaults.
  let mut c = ArraysCache::new(2);
  assert!(matches!(c.rope_offset().unwrap(), RopeOffset::Scalar(0)));
  assert_eq!(c.max_size(), None);
  assert!(!c.is_trimmable());
  assert_eq!(c.trim(5).unwrap(), 0);
  // Slot-aware metaState (swift `ArraysCache.metaState`): an empty 2-slot
  // cache with no left_padding -> ["2", ""] (slotCount, empty presentSlots).
  assert_eq!(c.meta_state(), vec!["2".to_string(), String::new()]);
  assert!(c.as_quantized().is_none());
  assert!(c.as_batch_positioned().is_none());
}

#[test]
fn from_state_arrays_cache() {
  // cache.py:79-82 load path. Empty meta `[]` -> swift's legacy branch
  // (KVCache.swift:1208-1211): the compacted state stands at slots 0..n.
  let st = vec![slot(&[7.0, 8.0]), slot(&[1.0])];
  let c = from_state("ArraysCache", st, &[]).unwrap();
  let rs = c.state().unwrap();
  assert_eq!(rs.len(), 2);
  let mut r0 = rs[0].try_clone().unwrap();
  let mut r1 = rs[1].try_clone().unwrap();
  assert_eq!(r0.to_vec::<f32>().unwrap(), vec![7.0, 8.0]);
  assert_eq!(r1.to_vec::<f32>().unwrap(), vec![1.0]);
  // No left_padding/lengths -> make_mask is None.
  assert!(matches!(
    c.make_mask(3, None, false).unwrap(),
    MaskMode::None
  ));

  // A malformed slot-aware meta (non-numeric slotCount) is a recoverable
  // error, never a panic.
  assert!(from_state("ArraysCache", vec![slot(&[1.0])], &["x".into()]).is_err());
}

#[test]
fn from_state_mamba_cache_is_arrays_cache_alias() {
  // mlx-swift-lm's `class MambaCache: ArraysCache` (KVCache.swift:1229) is a
  // 2-slot ArraysCache adding NO extra state/metadata; swift SAVES the kind
  // `"MambaCache"` (:1384) and its own load arm reconstructs it via the
  // identical `restoreFromMetaState` (:1531, == `ArraysCache(size: 2)`). So a
  // cross-tool `"MambaCache"`-kind prompt cache holds pure ArraysCache slot
  // state and must reconstruct via the EXACT same path as `"ArraysCache"`
  // (this crate's `from_state` is keyed on the reference class name) — no new
  // type, no Mamba arch. This asserts `"MambaCache"` is accepted and behaves
  // identically to `"ArraysCache"` on the same serialized bytes.

  // Build an ArraysCache with the MambaCache convention (size 2 =
  // `(conv_state, ssm_state)`), both slots populated with known tensors.
  let mut c = ArraysCache::new(2);
  c.set(0, slot(&[10.0, 11.0])).unwrap();
  c.set(1, slot(&[12.0])).unwrap();

  // Fully-populated 2-slot cache, no left_padding -> state() is the 2
  // compacted arrays; meta_state() == ["2", "0,1"] (hand-derived: slotCount
  // "2", presentSlots "0,1", no 3rd left_padding element).
  let st = c.state().unwrap();
  assert_eq!(st.len(), 2);
  let meta = c.meta_state();
  assert_eq!(meta, vec!["2".to_string(), "0,1".to_string()]);

  // `from_state("MambaCache", ...)` reconstructs an equivalent cache via the
  // exact same `arrays::from_state_arrays` path as `"ArraysCache"`.
  let m = from_state("MambaCache", st, &meta).unwrap();
  let rs = m.state().unwrap();
  assert_eq!(rs.len(), 2);
  let mut r0 = rs[0].try_clone().unwrap();
  let mut r1 = rs[1].try_clone().unwrap();
  assert_eq!(r0.to_vec::<f32>().unwrap(), vec![10.0, 11.0]);
  assert_eq!(r1.to_vec::<f32>().unwrap(), vec![12.0]);
  // No left_padding/lengths -> make_mask is None (identical to ArraysCache).
  assert!(matches!(
    m.make_mask(3, None, false).unwrap(),
    MaskMode::None
  ));

  // Behavioral identity: the SAME serialized bytes through `"ArraysCache"`
  // yield the same reconstructed state (the alias is a pure dispatch alias).
  let a = from_state("ArraysCache", c.state().unwrap(), &c.meta_state()).unwrap();
  let as_ = a.state().unwrap();
  assert_eq!(as_.len(), rs.len());
  let mut a0 = as_[0].try_clone().unwrap();
  let mut a1 = as_[1].try_clone().unwrap();
  assert_eq!(a0.to_vec::<f32>().unwrap(), vec![10.0, 11.0]);
  assert_eq!(a1.to_vec::<f32>().unwrap(), vec![12.0]);

  // A malformed slot-aware meta under `"MambaCache"` is the same recoverable
  // error as under `"ArraysCache"` (never a panic) — same path.
  assert!(from_state("MambaCache", vec![slot(&[1.0])], &["x".into()]).is_err());
}

#[test]
fn sparse_slot_state_round_trips() {
  // Codex regression: a cache with ONLY slot 1 populated must restore with
  // get(1) == the value and get(0) empty (NOT silently re-packed to slot
  // 0). Faithful to mlx-lm's full-slot-list state (cache.py:624-630) via
  // swift's slot-aware metaState (KVCache.swift:1173-1212).
  let mut c = ArraysCache::new(3);
  c.set(1, slot(&[42.0, 43.0])).unwrap();

  // state() is compacted (1 array); meta_state() carries slotCount +
  // presentSlots so the slot identity survives.
  let st = c.state().unwrap();
  assert_eq!(st.len(), 1);
  let meta = c.meta_state();
  assert_eq!(meta, vec!["3".to_string(), "1".to_string()]);

  // Restore on a concrete ArraysCache via the exact `from_state` order
  // (set_state then set_meta_state — `from_serialized` does precisely this;
  // the trait has no `as_any` to downcast the boxed form, so assert slot
  // identity directly on the concrete type, exercising the same path).
  let mut d = ArraysCache::new(0);
  d.set_state(st).unwrap();
  d.set_meta_state(&meta).unwrap();

  // Slot 1 holds the value; slots 0 and 2 are empty — slot identity
  // preserved (the Codex defect was slot 1 silently re-packed to slot 0).
  assert!(d.get(0).is_none());
  assert!(d.get(2).is_none());
  let mut g1 = d.get(1).unwrap().try_clone().unwrap();
  assert_eq!(g1.to_vec::<f32>().unwrap(), vec![42.0, 43.0]);
  assert_eq!(d.state().unwrap().len(), 1);
}

#[test]
fn set_meta_state_is_atomic_on_malformed_meta() {
  // Codex regression: a malformed slot-aware meta must NOT half-destroy the
  // cache. Both a non-numeric slotCount and a hostile huge slotCount (whose
  // slot buffer cannot be allocated) return Err with the prior `set_state`
  // arrays fully intact (never emptied), and never panic/abort.
  let restore = |meta: &[String]| {
    let mut d = ArraysCache::new(0);
    d.set_state(vec![slot(&[1.0, 2.0]), slot(&[3.0])]).unwrap();
    let err = d.set_meta_state(meta).is_err();
    // Cache survived unchanged: the 2 compacted arrays are still present.
    let s = d.state().unwrap();
    (err, s.len())
  };

  // Non-numeric slotCount.
  assert_eq!(restore(&["bogus".into(), "0,1".into()]), (true, 2));
  // Hostile huge slotCount: try_reserve_exact fails -> Err::OutOfMemory,
  // cache intact, process not aborted.
  assert_eq!(restore(&[usize::MAX.to_string(), "0,1".into()]), (true, 2));
  // Non-numeric present-slots CSV.
  assert_eq!(restore(&["2".into(), "0,x".into()]), (true, 2));
}

#[test]
fn left_padding_round_trips_via_meta() {
  // swift metaState appends leftPadding as a 3rd CSV element
  // (KVCache.swift:1179-1181); restore rebuilds it (1198-1207).
  let mut c = ArraysCache::with_left_padding(2, &[1, 0]);
  c.set(0, slot(&[5.0])).unwrap();
  c.set(1, slot(&[6.0])).unwrap();
  let meta = c.meta_state();
  assert_eq!(
    meta,
    vec!["2".to_string(), "0,1".to_string(), "1,0".to_string()]
  );

  let d = from_state("ArraysCache", c.state().unwrap(), &meta).unwrap();
  // left_padding restored -> make_mask([1,0], N=2): row0 [F,T], row1 [T,T].
  match d.make_mask(2, None, false).unwrap() {
    MaskMode::Array(mut m) => {
      assert_eq!(m.to_vec::<bool>().unwrap(), vec![false, true, true, true]);
    }
    _ => panic!("expected array mask from restored left_padding"),
  }
}

#[test]
fn set_meta_state_rejects_slot_count_above_max_cap() {
  // KVC-2 (#99): a forged/corrupt prompt cache with slotCount > MAX_SLOT_COUNT
  // is rejected fast with Error::Backend BEFORE try_reserve_exact runs (which
  // would otherwise hit the allocator with a multi-GB request). Realistic
  // SSM/Mamba caches have ≤ 64 slots, so the cap (1 << 20) is far above any
  // legitimate use. The prior `set_state` arrays must be left fully intact —
  // staged-then-commit transactional discipline (Copilot review #3271554056).
  let mut d = ArraysCache::new(0);
  d.set_state(vec![slot(&[1.0, 2.0]), slot(&[3.0])]).unwrap();
  let just_over = MAX_SLOT_COUNT + 1;
  let meta = vec![just_over.to_string(), "0,1".into()];
  let err = d.set_meta_state(&meta);
  match err {
    Err(Error::Backend { message }) => {
      assert!(
        message.contains("exceeds MAX_SLOT_COUNT"),
        "expected message to mention MAX_SLOT_COUNT, got: {message}"
      );
      assert!(
        message.contains(&just_over.to_string()),
        "expected message to include offending slot_count {just_over}, got: {message}"
      );
    }
    other => panic!("expected Err(Backend) about MAX_SLOT_COUNT, got {other:?}"),
  }
  // Cache survived unchanged: the 2 compacted arrays are still present.
  let s = d.state().unwrap();
  assert_eq!(
    s.len(),
    2,
    "cache must be left untouched on hostile slot_count"
  );
}

#[test]
fn kvc2_arrayscache_rejects_huge_slot_count_before_csv_parse() {
  // Codex-R1 [high] #1 (KVC-2 follow-up): a forged meta with a `slotCount >
  // MAX_SLOT_COUNT` AND a huge `presentSlots`/`leftPadding` CSV payload
  // must be rejected by the `MAX_SLOT_COUNT` gate BEFORE the CSV is parsed
  // into a `Vec<T>` — closing the "forged slotCount + huge CSV" evasion
  // where the prior version walked the CSV (large transient allocation +
  // parse work) before reaching the cap check.
  //
  // STRUCTURAL assertion: the error message MUST identify the
  // `MAX_SLOT_COUNT` gate (not the CSV `presentSlots`/`leftPadding`
  // element parse) — proving the cap fires first. If the CSV parse ran
  // first, a malformed token (e.g. `"X"`) would yield a parse error
  // mentioning `presentSlots`/`leftPadding`, not the cap.
  let mut d = ArraysCache::new(0);
  d.set_state(vec![slot(&[1.0])]).unwrap();
  // Huge slot_count AND a huge CSV payload containing UNPARSABLE tokens.
  // The cap MUST fire first; if the CSV parse ran first, the test would
  // hit the unparsable "X"/"Y" tokens and surface "presentSlots"/"X" in
  // the error message instead of "MAX_SLOT_COUNT".
  let huge_slot_count = (MAX_SLOT_COUNT + 10).to_string();
  // ~5 MB of garbage tokens — large enough that any pre-cap CSV parse
  // would be a measurable transient allocation. Building the input here
  // is the *test's* allocation, not the implementation's (and the test
  // is the producer-of-hostile-input — that work is unavoidable).
  let big_payload = (0..500_000usize)
    .map(|i| if i.is_multiple_of(2) { "X" } else { "Y" })
    .collect::<Vec<_>>()
    .join(",");
  let meta = vec![huge_slot_count, big_payload.clone()];
  match d.set_meta_state(&meta) {
    Err(Error::Backend { message }) => {
      assert!(
        message.contains("exceeds MAX_SLOT_COUNT"),
        "cap MUST fire first: expected MAX_SLOT_COUNT error, got: {message}"
      );
      // And NOT a CSV-parse error (would mean the parse ran first).
      assert!(
        !message.contains("presentSlots"),
        "MAX_SLOT_COUNT gate must precede CSV parse; got presentSlots-mentioning error: {message}"
      );
    }
    other => panic!("expected Err(Backend) about MAX_SLOT_COUNT, got {other:?}"),
  }
  // Also: the slot_count == 1 case (a small declared cap but a huge CSV
  // payload) — the per-CSV `max_elems = slot_count` bound MUST reject the
  // CSV before allocation, with a "element count exceeds bound" message
  // (NOT a token parse error). This closes the second variant of the
  // evasion where slot_count itself is small but the CSV is hostile.
  let meta_small_count_huge_csv = vec!["1".to_string(), big_payload];
  match d.set_meta_state(&meta_small_count_huge_csv) {
    Err(Error::Backend { message }) => {
      assert!(
        message.contains("element count") && message.contains("exceeds bound"),
        "CSV bound MUST fire before token parse: expected element-count error, got: {message}"
      );
      assert!(
        message.contains("presentSlots"),
        "CSV bound message should identify the offending CSV (presentSlots), got: {message}"
      );
    }
    other => panic!("small slot_count + huge CSV must hit CSV bound, got {other:?}"),
  }
  // Cache survived all rejected restores untouched.
  assert_eq!(
    d.state().unwrap().len(),
    1,
    "cache must be left untouched on hostile slot_count / hostile CSV"
  );
}

#[test]
fn kvc2_arrayscache_csv_bound_rejects_oversized_left_padding() {
  // Codex-R1 [high] #1 sibling assertion: the `leftPadding` CSV is also
  // bounded by `slot_count` — a forged meta with a valid small
  // `slot_count` + valid small `presentSlots` but a HUGE `leftPadding`
  // CSV must be rejected before allocating the `Vec<i32>`.
  let mut d = ArraysCache::new(0);
  d.set_state(vec![slot(&[1.0])]).unwrap();
  let huge_lp_csv = (0..10_000)
    .map(|i| i.to_string())
    .collect::<Vec<_>>()
    .join(",");
  // slot_count=2, presentSlots="0,1" (in-bound), leftPadding=10000 entries.
  let meta = vec!["2".to_string(), "0,1".to_string(), huge_lp_csv];
  match d.set_meta_state(&meta) {
    Err(Error::Backend { message }) => {
      assert!(
        message.contains("element count") && message.contains("exceeds bound"),
        "leftPadding CSV bound must fire: expected element-count error, got: {message}"
      );
      assert!(
        message.contains("leftPadding"),
        "CSV bound message should identify the offending CSV (leftPadding), got: {message}"
      );
    }
    other => panic!("huge leftPadding CSV must hit bound, got {other:?}"),
  }
  // Cache untouched.
  assert_eq!(d.state().unwrap().len(), 1);
}

#[test]
fn set_meta_state_accepts_slot_count_at_max_cap() {
  // KVC-2 sanity: a meta with `slotCount = MAX_SLOT_COUNT` (the boundary)
  // must succeed — only `> MAX_SLOT_COUNT` is rejected. Use an EMPTY
  // presentSlots CSV so the test does not actually allocate the full
  // MAX_SLOT_COUNT × sizeof::<Option<Array>>() backing buffer in CI; the
  // assertion is purely on the gate's boundary, not on the residual
  // try_reserve_exact (which IS still exercised by the call but may itself
  // OOM in a constrained CI environment — handled by accepting either
  // Ok or Err::OutOfMemory, both of which prove the MAX_SLOT_COUNT gate
  // did NOT fire).
  let mut d = ArraysCache::new(0);
  let meta = vec![MAX_SLOT_COUNT.to_string(), String::new()];
  match d.set_meta_state(&meta) {
    Ok(()) => {
      // The cap let it through and the allocator satisfied the request.
    }
    Err(Error::OutOfMemory) => {
      // The cap let it through but the allocator declined — both prove
      // the MAX_SLOT_COUNT gate is NOT firing at the boundary value.
    }
    other => panic!(
      "boundary slot_count = MAX_SLOT_COUNT must NOT be rejected by the cap gate; got {other:?}"
    ),
  }
}
