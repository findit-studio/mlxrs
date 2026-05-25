//! Deterministic tests for the M3 quantized KV cache
//! (`mlxrs::lm::cache::QuantizedKvCacheImpl`), hand-traced 1:1 from
//! `mlx_lm.models.cache.QuantizedKVCache` (`cache.py:232-324`) and
//! cross-checked against mlx-swift-lm's `MLXLMCommon.QuantizedKVCache`
//! (`KVCache.swift:744-1005`) + `QuantizedKVCacheProtocol`
//! (`KVCache.swift:111-136`).
//!
//! Caches are 4-D `[B, n_kv_heads, S, head_dim]` (sequence axis `-2`),
//! matching mlx-lm. `group_size = 64, bits = 8`: `el_per_int = 8 *
//! uint32.size // bits = 32 // 8 = 4`, so for `head_dim = 64` a quantized
//! tensor is the triple `(weight [B, H, S, 16], scales [B, H, S, 1],
//! biases [B, H, S, 1])`. Tensors are tiny and the round-trip is checked
//! against the original within the 8-bit affine quantization error band.

#![cfg(feature = "lm")]

use mlxrs::{
  Array,
  lm::cache::{KvCache, MaskMode, QuantizedKvCache, QuantizedKvCacheImpl, from_state},
  ops,
};

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 8;
const HEAD_DIM: usize = 64;
const MODE: &str = "affine";

/// A `[1, 1, S, HEAD_DIM]` KV tensor with a smooth, per-element-distinct
/// ramp (so 8-bit affine quant stays close and every position differs).
/// `S == n_steps`.
fn kv(n_steps: usize) -> Array {
  kv_base(n_steps, 0.0)
}

/// Like [`kv`] but every element is additionally shifted by `base`, so a
/// tensor built with a distinct `base` shares no value with another — used
/// to prove an overwrite landed the *new* token, not a stale one.
fn kv_base(n_steps: usize, base: f32) -> Array {
  let total = n_steps * HEAD_DIM;
  let data: Vec<f32> = (0..total)
    .map(|i| (i as f32) * 0.013 - 0.4 + base)
    .collect();
  Array::from_slice::<f32>(&data, &(1usize, 1, n_steps, HEAD_DIM)).unwrap()
}

/// Dequantize a `(w, scales, biases)` triple via the merged
/// `crate::ops::quantized::dequantize` (the #19 op — NOT a reimpl).
fn dequant(t: &(Array, Array, Option<Array>)) -> Array {
  ops::quantized::dequantize(&t.0, &t.1, t.2.as_ref(), GROUP_SIZE, BITS, MODE, None, None).unwrap()
}

/// Assert `got ≈ want` within the 8-bit affine quantization band
/// (relative to the magnitude of `want`).
fn assert_close(got: &mut Array, want: &mut Array) {
  let g = got.to_vec::<f32>().unwrap();
  let w = want.to_vec::<f32>().unwrap();
  assert_eq!(g.len(), w.len(), "length mismatch");
  let max_abs = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
  for (a, b) in g.iter().zip(w.iter()) {
    assert!(
      (a - b).abs() <= 0.05 * max_abs + 1e-3,
      "dequant drift too large: got={a} want={b}"
    );
  }
}

/// `update_quantized` quantizes + accumulates and returns the full triples;
/// dequantizing them recovers the original within the quant band; `offset`
/// tracks the sequence length; the over-allocated-buffer growth
/// (`expand_quant`, hand-traced from `cache.py:259-283`) is observably the
/// sequence-axis concatenation.
#[test]
fn update_quantized_roundtrips_and_grows() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  assert!(c.is_empty());
  assert_eq!(c.offset(), 0);
  assert_eq!(c.group_size(), GROUP_SIZE);
  assert_eq!(c.bits(), BITS);

  // First update: 3 steps. `el_per_int = 32 // 8 = 4`, head_dim 64 →
  // weight [1,1,3,16], scales/biases [1,1,3,1].
  let mut k1 = kv(3);
  let mut v1 = kv(3);
  let (qk1, qv1) = c.update_quantized(&k1, &v1).unwrap();
  assert!(!c.is_empty());
  assert_eq!(c.offset(), 3);
  assert_eq!(qk1.0.shape(), vec![1, 1, 3, HEAD_DIM / 4]);
  assert_eq!(qk1.1.shape(), vec![1, 1, 3, HEAD_DIM / GROUP_SIZE as usize]);
  assert!(qk1.2.is_some(), "affine quantize yields Some(biases)");
  assert_eq!(qv1.0.shape(), vec![1, 1, 3, HEAD_DIM / 4]);
  // Dequantizing the returned triples recovers the originals.
  let mut dk1 = dequant(&qk1);
  let mut dv1 = dequant(&qv1);
  assert_eq!(dk1.shape(), vec![1, 1, 3, HEAD_DIM]);
  assert_close(&mut dk1, &mut k1);
  assert_close(&mut dv1, &mut v1);

  // Second update: 2 more steps. This is the `expand_quant`/over-allocated
  // buffer growth (cache.py:259-283): observably the new quantized triple
  // concatenated on the sequence axis (`-2`), offset 5. Dequantizing the
  // returned triple must recover the FULL 5-step original [steps 0..5].
  let mut k2 = kv(2);
  let mut v2 = kv(2);
  let (qk2, qv2) = c.update_quantized(&k2, &v2).unwrap();
  assert_eq!(c.offset(), 5);
  assert_eq!(qk2.0.shape(), vec![1, 1, 5, HEAD_DIM / 4]);
  let mut dk2 = dequant(&qk2);
  assert_eq!(dk2.shape(), vec![1, 1, 5, HEAD_DIM]);
  // Expected accumulated original = [step0..3 of k1] ++ [step0..2 of k2]
  // along the sequence axis (each `kv(n)` is the same ramp prefix).
  let want_k: Vec<f32> = {
    let mut a = k1.to_vec::<f32>().unwrap();
    a.extend_from_slice(&k2.to_vec::<f32>().unwrap());
    a
  };
  let mut want_k_arr = Array::from_slice::<f32>(&want_k, &(1usize, 1, 5usize, HEAD_DIM)).unwrap();
  assert_close(&mut dk2, &mut want_k_arr);
  let mut dv2 = dequant(&qv2);
  let want_v: Vec<f32> = {
    let mut a = v1.to_vec::<f32>().unwrap();
    a.extend_from_slice(&v2.to_vec::<f32>().unwrap());
    a
  };
  let mut want_v_arr = Array::from_slice::<f32>(&want_v, &(1usize, 1, 5usize, HEAD_DIM)).unwrap();
  assert_close(&mut dv2, &mut want_v_arr);
}

/// The base `KvCache::update` returns the **dequantized** accumulated
/// `(keys, values)` (the documented mlx-swift-lm non-quantized fallback /
/// `toUnquantized` contract): it must approximate the original keys/values
/// within the quant band and keep `offset` consistent with
/// `update_quantized`.
#[test]
fn base_update_returns_dequantized() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  let mut k = kv(4);
  let mut v = kv(4);
  let (mut dk, mut dv) = c.update(&k, &v).unwrap();
  assert_eq!(c.offset(), 4);
  assert_eq!(dk.shape(), vec![1, 1, 4, HEAD_DIM]);
  assert_eq!(dv.shape(), vec![1, 1, 4, HEAD_DIM]);
  assert_close(&mut dk, &mut k);
  assert_close(&mut dv, &mut v);

  // A second base update accumulates (offset 6) and still dequantizes to
  // the full prefix.
  let mut k2 = kv(2);
  let (mut dk2, _) = c.update(&k2, &kv(2)).unwrap();
  assert_eq!(c.offset(), 6);
  assert_eq!(dk2.shape(), vec![1, 1, 6, HEAD_DIM]);
  let want: Vec<f32> = {
    let mut a = k.to_vec::<f32>().unwrap();
    a.extend_from_slice(&k2.to_vec::<f32>().unwrap());
    a
  };
  let mut want_arr = Array::from_slice::<f32>(&want, &(1usize, 1, 6usize, HEAD_DIM)).unwrap();
  assert_close(&mut dk2, &mut want_arr);
}

/// `quantized_state` is `None` before any update (mlx-swift-lm
/// `getQuantizedState` returns `nil` when empty), `Some` after, and the
/// `Some` triples dequantize back to the original.
#[test]
fn quantized_state_none_then_some() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  assert!(c.quantized_state().unwrap().is_none());
  assert!(c.as_quantized().is_some());

  let mut k = kv(3);
  let mut v = kv(3);
  c.update_quantized(&k, &v).unwrap();

  let st = c.quantized_state().unwrap();
  assert!(st.is_some());
  let (qk, qv) = st.unwrap();
  assert_eq!(qk.0.shape(), vec![1, 1, 3, HEAD_DIM / 4]);
  let mut dk = dequant(&qk);
  let mut dv = dequant(&qv);
  assert_close(&mut dk, &mut k);
  assert_close(&mut dv, &mut v);

  // `quantized_state` does NOT mutate (no extra steps).
  assert_eq!(c.offset(), 3);
  assert!(c.quantized_state().unwrap().is_some());
  assert_eq!(c.offset(), 3);
}

/// `state()`/`set_state()` round-trip the packed triples: the serialized
/// form is 6 arrays (`[k.w, k.s, k.b, v.w, v.s, v.b]` — affine has biases),
/// and a fresh cache restored from it dequantizes identically.
#[test]
fn state_set_state_roundtrip() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  // Empty cache: state is [] (mlx-swift-lm `KVCache.swift:919`).
  assert!(c.state().unwrap().is_empty());

  let mut k = kv(3);
  let mut v = kv(3);
  c.update_quantized(&k, &v).unwrap();

  let st = c.state().unwrap();
  // Affine → biases present → 6 arrays (mlx-swift-lm `KVCache.swift:932`).
  assert_eq!(st.len(), 6);

  // Restore into a fresh cache, then carry meta_state (offset/group/bits).
  let st_clone: Vec<Array> = st.iter().map(|a| a.try_clone().unwrap()).collect();
  let meta = c.meta_state();
  // mlx-lm `meta_state` = (offset, group_size, bits) (cache.py:300).
  assert_eq!(
    meta,
    vec!["3".to_string(), "64".to_string(), "8".to_string()]
  );

  let mut c2 = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  c2.set_state(st_clone).unwrap();
  c2.set_meta_state(&meta).unwrap();
  assert_eq!(c2.offset(), 3);
  assert!(!c2.is_empty());

  // The restored cache's quantized_state dequantizes to the original.
  let (qk, qv) = c2.quantized_state().unwrap().unwrap();
  let mut dk = dequant(&qk);
  let mut dv = dequant(&qv);
  assert_close(&mut dk, &mut k);
  assert_close(&mut dv, &mut v);

  // Empty round-trips: set_state([]) resets.
  c2.set_state(Vec::new()).unwrap();
  assert!(c2.is_empty());
  assert_eq!(c2.offset(), 0);
  assert!(c2.state().unwrap().is_empty());
}

/// `from_state` reconstructs the quantized cache from the SOURCE class name
/// a real prompt cache writes (`type(c).__name__ == "QuantizedKVCache"`,
/// `cache.py:56` → `globals()[name]`, `cache.py:80`); meta_state restores
/// `(offset, group_size, bits)` (`cache.py:302-304`).
#[test]
fn from_state_quantized_roundtrip() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  let mut k = kv(2);
  let mut v = kv(2);
  c.update_quantized(&k, &v).unwrap();

  let st: Vec<Array> = c
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let meta = c.meta_state();

  let c2 = from_state("QuantizedKVCache", st, &meta).unwrap();
  assert_eq!(c2.offset(), 2);
  assert!(!c2.is_empty());
  let q = c2.as_quantized().expect("reconstructed cache is quantized");
  assert_eq!(q.group_size(), GROUP_SIZE);
  assert_eq!(q.bits(), BITS);
  let (qk, qv) = q.quantized_state().unwrap().unwrap();
  let mut dk = dequant(&qk);
  let mut dv = dequant(&qv);
  assert_close(&mut dk, &mut k);
  assert_close(&mut dv, &mut v);

  // Unknown kind is a recoverable error (not a panic).
  assert!(from_state("NotACache", Vec::new(), &[]).is_err());

  // Empty state + non-zero restored offset is an impossible cache
  // (`keys=None` but `offset>0`): mlx-lm's `state` setter (cache.py:295,
  // `self.keys, self.values = v`) can't even unpack an empty `v`, so this
  // combination is unreachable there; `from_state` must reject it (mirrors
  // the `RotatingKvCache` empty/offset guard) rather than build a cache
  // whose next update diverges `offset` from the stored length.
  assert!(
    from_state(
      "QuantizedKVCache",
      Vec::new(),
      &["1".into(), "64".into(), "8".into()]
    )
    .is_err()
  );
  // The consistent empty restore (offset 0) is still accepted.
  let empty = from_state(
    "QuantizedKVCache",
    Vec::new(),
    &["0".into(), "64".into(), "8".into()],
  )
  .unwrap();
  assert!(empty.is_empty());
  assert_eq!(empty.offset(), 0);
}

/// `make_mask` forwards to the generic `create_attention_mask` with the
/// cache's `offset` (mlx-lm `cache.py:314-315`): `N == 1` → `None`,
/// multi-token → `Causal` (no array) unless `return_array`/`window_size`.
#[test]
fn make_mask_forwards_to_create_attention_mask() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  c.update_quantized(&kv(3), &kv(3)).unwrap();
  // N == 1 → no mask (offset != 0 but single decode token).
  assert!(matches!(
    c.make_mask(1, None, false).unwrap(),
    MaskMode::None
  ));
  // Multi-token, no return_array/window → symbolic causal.
  assert!(matches!(
    c.make_mask(2, None, false).unwrap(),
    MaskMode::Causal
  ));
  // return_array → a materialized [N, offset+N] mask.
  match c.make_mask(2, None, true).unwrap() {
    MaskMode::Array(m) => assert_eq!(m.shape(), vec![2, 3 + 2]),
    _ => panic!("make_mask(2, None, true) must be a materialized Array (cache.py:122)"),
  }
  // window_size → a materialized windowed mask.
  assert!(matches!(
    c.make_mask(2, Some(1), false).unwrap(),
    MaskMode::Array(_)
  ));
}

/// `trim` drops the most recent `min(offset, n)` tokens and adjusts only
/// `offset` (mlx-lm `cache.py:309-312`); `nbytes` sums every present triple
/// array's bytes (mlx-lm `cache.py:320-322`); `copy` is an independent
/// deep clone (mlx-swift-lm `KVCache.swift:972-980`).
#[test]
fn trim_nbytes_copy() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  assert_eq!(c.nbytes(), 0);
  assert!(c.is_trimmable());

  c.update_quantized(&kv(4), &kv(4)).unwrap();
  assert_eq!(c.offset(), 4);
  // weight [1,1,4,16] u32 = 4*16*4 bytes; scales/biases [1,1,4,1] f32 =
  // 4*1*4 bytes each; ×2 (keys+values). Just assert it is the positive
  // byte sum (no eval, pure metadata) — exact dtype byte math is covered
  // by util::nbytes' own tests.
  assert!(c.nbytes() > 0);

  // copy is independent: mutating the copy must not change the original.
  let mut cp = c.copy().unwrap();
  cp.update(&kv(1), &kv(1)).unwrap();
  assert_eq!(cp.offset(), 5);
  assert_eq!(
    c.offset(),
    4,
    "original must be unaffected by copy mutation"
  );

  // trim(3): offset 4 → 1.
  assert_eq!(c.trim(3).unwrap(), 3);
  assert_eq!(c.offset(), 1);
  // trim never removes more than offset.
  assert_eq!(c.trim(10).unwrap(), 1);
  assert_eq!(c.offset(), 0);
}

/// A wrong-rank (not 4-D `[B, n_kv_heads, S, head_dim]`) input is a
/// recoverable `Err` (via `util::seq_len`), never a panic / raw shape
/// index.
#[test]
fn wrong_rank_errors() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  let bad = Array::from_slice::<f32>(&[1.0, 2.0], &(1usize, 2)).unwrap();
  assert!(c.update_quantized(&bad, &bad).is_err());
  assert!(c.update(&bad, &bad).is_err());
  // Bad meta_state arity / non-numeric value → recoverable Err.
  assert!(c.set_meta_state(&["1".into(), "64".into()]).is_err());
  assert!(
    c.set_meta_state(&["x".into(), "64".into(), "8".into()])
      .is_err()
  );
  // Bad state arity → recoverable Err.
  assert!(c.set_state(vec![bad.try_clone().unwrap()]).is_err());
}

/// Faithful `update_quantized` → `trim` → `update_quantized` semantics
/// (hand-traced 1:1 from `cache.py:242-283`).
///
/// Reference trace, `step=256`, after `update_and_fetch(S=4)` the buffer
/// length is 4 and `offset==4`; `trim(3)` sets `offset=1` (mlx-lm only
/// decrements `offset` — its over-allocated buffer is overwritten in place
/// on the next update). The next `update_and_fetch(S=1)` has `prev=1`,
/// `prev+S = 2 <= 4` so the grow branch is skipped, `offset += 1 -> 2`,
/// `self.keys[i][..., 1:2, :] = quant(new)[i]` (the trimmed-off token at
/// physical position 1 is **overwritten**), and it returns
/// `tree_map(x[..., :2, :])` = `[token0, NEW_token]`.
///
/// So position 1 of the returned/`quantized_state` triple MUST be the new
/// token, NOT the stale trimmed `token1`. The new input uses a disjoint
/// value range (`base = 100.0`) so a stale token (original `token1`,
/// values ~0.43..1.25) is unmistakably distinguishable from the correct
/// new token (~99.6..100.4). This is the regression for the Codex
/// adversarial-review finding (append-onto-stale-storage after trim).
#[test]
fn update_after_trim_overwrites_not_appends() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);

  // update_and_fetch(S=4): tokens t0..t3 (ramp, base 0). offset 4.
  let mut k4 = kv(4);
  let mut v4 = kv(4);
  c.update_quantized(&k4, &v4).unwrap();
  assert_eq!(c.offset(), 4);

  // trim(3): offset 4 -> 1. (Storage must be re-sliced to length 1 so the
  // next update overwrites at prev=1, exactly mlx-lm's observable result.)
  assert_eq!(c.trim(3).unwrap(), 3);
  assert_eq!(c.offset(), 1);
  // After trim, quantized_state is length 1 and dequantizes to token0
  // (the first HEAD_DIM ramp values).
  let (qk_t, _) = c.quantized_state().unwrap().unwrap();
  assert_eq!(qk_t.0.shape(), vec![1, 1, 1, HEAD_DIM / 4]);
  let mut dk_t = dequant(&qk_t);
  let mut tok0 = kv(1); // == token0 of kv(4)
  assert_close(&mut dk_t, &mut tok0);

  // update_and_fetch(S=1) with a DISJOINT value range (base 100).
  let mut new_tok = kv_base(1, 100.0);
  let (qk, qv) = c.update_quantized(&new_tok, &new_tok).unwrap();
  assert_eq!(c.offset(), 2);
  // Returned triple length is 2 (token0 + NEW), NOT 5 (stale append).
  assert_eq!(qk.0.shape(), vec![1, 1, 2, HEAD_DIM / 4]);

  // Dequantized returned keys == [token0, NEW_token] along the seq axis.
  let mut dk = dequant(&qk);
  assert_eq!(dk.shape(), vec![1, 1, 2, HEAD_DIM]);
  let want: Vec<f32> = {
    let mut a = kv(1).to_vec::<f32>().unwrap(); // token0
    a.extend_from_slice(&new_tok.to_vec::<f32>().unwrap()); // NEW
    a
  };
  let mut want_arr = Array::from_slice::<f32>(&want, &(1usize, 1, 2usize, HEAD_DIM)).unwrap();
  assert_close(&mut dk, &mut want_arr);
  let mut dv = dequant(&qv);
  assert_close(&mut dv, &mut want_arr);

  // Explicitly assert position 1 is the NEW token, not the stale original
  // token1 (which trim dropped). Compare the second HEAD_DIM slice.
  let got = dk.to_vec::<f32>().unwrap();
  let pos1: Vec<f32> = got[HEAD_DIM..2 * HEAD_DIM].to_vec();
  let new_vals = new_tok.to_vec::<f32>().unwrap();
  let stale_tok1 = k4.to_vec::<f32>().unwrap()[HEAD_DIM..2 * HEAD_DIM].to_vec();
  let max_abs = new_vals.iter().fold(0.0f32, |m, x| m.max(x.abs()));
  for (g, n) in pos1.iter().zip(new_vals.iter()) {
    assert!(
      (g - n).abs() <= 0.05 * max_abs + 1e-3,
      "post-trim update position 1 must be the NEW token: got={g} want={n}"
    );
  }
  // And it must be far from the stale trimmed token1 (disjoint ranges).
  let drift_from_stale: f32 = pos1
    .iter()
    .zip(stale_tok1.iter())
    .map(|(g, s)| (g - s).abs())
    .fold(0.0, f32::max);
  assert!(
    drift_from_stale > 10.0,
    "position 1 must NOT be the stale trimmed token1 (drift {drift_from_stale} too small)"
  );

  // quantized_state mirrors the same accumulated content.
  let (qk2, _) = c.quantized_state().unwrap().unwrap();
  assert_eq!(qk2.0.shape(), vec![1, 1, 2, HEAD_DIM / 4]);
  let mut dk2 = dequant(&qk2);
  assert_close(&mut dk2, &mut want_arr);

  // touch v4 so it isn't flagged unused (kept for symmetry with k4).
  let _ = v4.to_vec::<f32>().unwrap();
}

/// FIX 1 regression — the quantized cache's defining capability
/// `update_quantized` (`&mut self`) MUST be reachable through the generic
/// `&mut dyn KvCache` / `Box<dyn KvCache>` a generation loop holds, via the
/// **mutable** `as_quantized_mut` downcast (mlx-swift-lm `cache as?
/// QuantizedKVCacheProtocol` on a class-mutable cache,
/// `KVCache.swift:101`). The non-`&mut` `as_quantized` alone cannot reach
/// it. And a non-quantized cache must still return `None` (the additive
/// defaulted-`None` is inherited unchanged — no sibling churn).
#[test]
fn as_quantized_mut_reaches_update_quantized_through_dyn() {
  // Hold the quantized cache ONLY as a generic boxed `dyn KvCache` (what a
  // generation loop / `make_prompt_cache` vector actually carries).
  let mut boxed: Box<dyn KvCache> = Box::new(QuantizedKvCacheImpl::new(GROUP_SIZE, BITS));

  // The mutable downcast must succeed and let us call `update_quantized`.
  {
    let q = boxed
      .as_quantized_mut()
      .expect("QuantizedKvCacheImpl must downcast via as_quantized_mut");
    assert_eq!(q.group_size(), GROUP_SIZE);
    assert_eq!(q.bits(), BITS);
    let mut k = kv(3);
    let mut v = kv(3);
    let (qk, qv) = q
      .update_quantized(&k, &v)
      .expect("update_quantized through &mut dyn KvCache must succeed");
    assert_eq!(qk.0.shape(), vec![1, 1, 3, HEAD_DIM / 4]);
    let mut dk = dequant(&qk);
    let mut dv = dequant(&qv);
    assert_close(&mut dk, &mut k);
    assert_close(&mut dv, &mut v);
  }
  // The mutation landed on the cache the box owns (offset advanced).
  assert_eq!(boxed.offset(), 3);
  // A second mutable downcast + update accumulates (offset 5) — proves the
  // downcast targets the live cache, not a transient.
  {
    let q = boxed.as_quantized_mut().unwrap();
    q.update_quantized(&kv(2), &kv(2)).unwrap();
  }
  assert_eq!(boxed.offset(), 5);

  // A non-quantized cache inherits the additive defaulted `None` (no
  // sibling override) for BOTH the shared and the mutable downcast.
  let mut std_boxed: Box<dyn KvCache> = Box::new(mlxrs::lm::cache::StandardKvCache::new());
  assert!(std_boxed.as_quantized().is_none());
  assert!(std_boxed.as_quantized_mut().is_none());
}

/// FIX 2 regression — `from_state("QuantizedKVCache", state, meta)` must
/// re-establish P2's storage invariant (stored triples are exactly
/// `offset`-length) by slicing each restored triple's sequence axis down to
/// the restored `offset`, so a forged/inconsistent serialized cache (triple
/// seq-len > meta `offset`) does NOT leak stale tokens past the logical
/// offset on the next `update_quantized`. This makes P2's offset-length
/// representation observably IDENTICAL to mlx-lm, whose `state` getter
/// already returns `[..., :offset, :]` (`cache.py:285-292`) — repr-
/// equivalence maintenance, NOT a reject, and a no-op for consistent
/// states.
#[test]
fn from_state_slices_forged_overlong_triples_to_offset() {
  // Build a cache with 5 real steps (ramp t0..t4), capture its honest
  // 6-array state (seq-len 5) but FORGE the meta_state so the restored
  // `offset` is 3 (< the triples' seq-len 5) — an inconsistent/forged
  // serialized prompt cache (mlx-lm's `state` setter assigns triples as-is;
  // its getter would have sliced to `[:offset]`, so a faithful save never
  // produces this, but a forged blob can).
  let mut src = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src.update_quantized(&kv(5), &kv(5)).unwrap();
  let st: Vec<Array> = src
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  assert_eq!(st.len(), 6, "affine → biased 6-array state");
  // Honest seq-len of the serialized triples is 5.
  assert_eq!(st[0].shape(), vec![1, 1, 5, HEAD_DIM / 4]);
  let forged_meta = vec!["3".to_string(), "64".to_string(), "8".to_string()];

  let mut c = from_state("QuantizedKVCache", st, &forged_meta).unwrap();
  assert_eq!(c.offset(), 3);
  // The reconstructed cache's stored triples were sliced to offset 3 (NOT
  // the forged seq-len 5): `quantized_state` is length 3 and dequantizes
  // to exactly the first 3 ramp steps (== kv(3)), no stale t3/t4.
  {
    let q = c.as_quantized().expect("reconstructed cache is quantized");
    let (qk, qv) = q.quantized_state().unwrap().unwrap();
    assert_eq!(
      qk.0.shape(),
      vec![1, 1, 3, HEAD_DIM / 4],
      "restored triples must be sliced to offset 3, not the forged seq-len 5"
    );
    let mut dk = dequant(&qk);
    let mut dv = dequant(&qv);
    let mut want3 = kv(3); // first 3 ramp steps of kv(5)
    assert_close(&mut dk, &mut want3);
    let mut want3v = kv(3);
    assert_close(&mut dv, &mut want3v);
  }
  // The next update_quantized must concat onto the SLICED (len-3) triples:
  // a DISJOINT new token at position 3 — NOT the stale forged t3.
  let mut new_tok = kv_base(1, 100.0);
  let (qk, _) = {
    let q = c
      .as_quantized_mut()
      .expect("reconstructed cache downcasts mutably");
    q.update_quantized(&new_tok, &new_tok).unwrap()
  };
  assert_eq!(c.offset(), 4);
  // Length 4 (t0,t1,t2,NEW) — NOT 6 (a stale append onto the un-sliced
  // forged seq-len-5 triple).
  assert_eq!(qk.0.shape(), vec![1, 1, 4, HEAD_DIM / 4]);
  let mut dk = dequant(&qk);
  let want: Vec<f32> = {
    let mut a = kv(3).to_vec::<f32>().unwrap(); // t0..t2
    a.extend_from_slice(&new_tok.to_vec::<f32>().unwrap()); // NEW
    a
  };
  let mut want_arr = Array::from_slice::<f32>(&want, &(1usize, 1, 4usize, HEAD_DIM)).unwrap();
  assert_close(&mut dk, &mut want_arr);
  // Explicitly: position 3 is the NEW token (~100 range), provably NOT the
  // stale forged t3 (kv(5) step 3, ~ small ramp range).
  let got = dk.to_vec::<f32>().unwrap();
  let pos3: Vec<f32> = got[3 * HEAD_DIM..4 * HEAD_DIM].to_vec();
  let new_vals = new_tok.to_vec::<f32>().unwrap();
  let stale_t3 = kv(5).to_vec::<f32>().unwrap()[3 * HEAD_DIM..4 * HEAD_DIM].to_vec();
  let max_abs = new_vals.iter().fold(0.0f32, |m, x| m.max(x.abs()));
  for (g, n) in pos3.iter().zip(new_vals.iter()) {
    assert!(
      (g - n).abs() <= 0.05 * max_abs + 1e-3,
      "post-restore update position 3 must be the NEW token: got={g} want={n}"
    );
  }
  let drift_from_stale: f32 = pos3
    .iter()
    .zip(stale_t3.iter())
    .map(|(g, s)| (g - s).abs())
    .fold(0.0, f32::max);
  assert!(
    drift_from_stale > 10.0,
    "position 3 must NOT be the stale forged t3 (drift {drift_from_stale} too small)"
  );

  // A CONSISTENT state (seq-len == offset) round-trips byte-identically:
  // the slice-to-offset is a pure no-op for a faithfully saved state.
  let mut src2 = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src2.update_quantized(&kv(4), &kv(4)).unwrap();
  let consistent_st: Vec<Array> = src2
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let consistent_meta = src2.meta_state(); // ("4","64","8") — offset == seq-len
  assert_eq!(consistent_meta[0], "4");
  // Capture the honest dequantized content BEFORE the round-trip.
  let (sk, sv) = src2.quantized_state().unwrap().unwrap();
  let mut want_k = dequant(&sk);
  let mut want_v = dequant(&sv);

  let rt = from_state("QuantizedKVCache", consistent_st, &consistent_meta).unwrap();
  assert_eq!(rt.offset(), 4, "consistent offset preserved");
  let q = rt.as_quantized().unwrap();
  let (rk, rv) = q.quantized_state().unwrap().unwrap();
  // Same shape (slice was a no-op: seq-len 4 sliced to offset 4 == itself).
  assert_eq!(rk.0.shape(), vec![1, 1, 4, HEAD_DIM / 4]);
  let mut got_k = dequant(&rk);
  let mut got_v = dequant(&rv);
  // Byte-identical dequantized content (no value drift from the no-op slice).
  let gk = got_k.to_vec::<f32>().unwrap();
  let wk = want_k.to_vec::<f32>().unwrap();
  assert_eq!(gk.len(), wk.len());
  for (g, w) in gk.iter().zip(wk.iter()) {
    assert_eq!(
      g, w,
      "consistent-state round-trip must be byte-identical (keys)"
    );
  }
  let gv = got_v.to_vec::<f32>().unwrap();
  let wv = want_v.to_vec::<f32>().unwrap();
  for (g, w) in gv.iter().zip(wv.iter()) {
    assert_eq!(
      g, w,
      "consistent-state round-trip must be byte-identical (values)"
    );
  }
}

/// FIX 3 regression — symmetric to the FIX 2 overlength slice: when the
/// restored state is UNDERLENGTH (stored triple seq-len < restored
/// `offset`), `enforce_offset_len_invariant` must clamp `self.offset` DOWN
/// to the actual stored seq-len. `slice_seq` uses mlx's NumPy-style
/// `std::min(end, n)` clamping (`mlx/ops.cpp:685`), so the trim returns the
/// full shorter array; without this symmetric clamp `self.offset` would
/// stay at the larger forged value while storage stayed shorter, and the
/// next `update_quantized` would land the new token past the storage end
/// (phantom-slot gap) — the next/concat_seq would then reflect an offset
/// that does NOT match the stored sequence length. The user-approved
/// "slice, don't reject" policy generalizes to "converge storage and
/// offset to the smaller of the two via NumPy-style clamping"; mlx-lm's
/// `state` getter already reports `[..., :offset, :]` =
/// `[:min(offset, buf_len)]`, so this maintains observable equivalence in
/// both directions. A consistent (offset == seq-len) state round-trips
/// byte-identically — the clamp is a no-op for it.
#[test]
fn from_state_underlength_state_clamps_offset_down() {
  // Build a cache with 3 real steps (ramp t0..t2), capture its honest
  // 6-array state (seq-len 3), and FORGE the meta_state so the restored
  // `offset` is 5 (> the triples' seq-len 3) — the inconsistent
  // underlength direction (mlx-lm's getter would have sliced to
  // `[:offset]` which is the full 3 here, so a faithful save never
  // produces this; a forged blob can).
  let mut src = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src.update_quantized(&kv(3), &kv(3)).unwrap();
  let st: Vec<Array> = src
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  assert_eq!(st.len(), 6, "affine → biased 6-array state");
  // Honest seq-len of the serialized triples is 3.
  assert_eq!(st[0].shape(), vec![1, 1, 3, HEAD_DIM / 4]);
  // Forge meta_state[0] = "5" (offset > stored seq-len 3).
  let forged_meta = vec!["5".to_string(), "64".to_string(), "8".to_string()];

  let mut c = from_state("QuantizedKVCache", st, &forged_meta).unwrap();
  // SYMMETRIC CLAMP: offset must be CLAMPED DOWN to the actual stored
  // seq-len (3), NOT the forged 5. Hand-derivation: `trim_triple(_, 5)`
  // on a seq-len-3 array → NumPy `std::min(5, 3) == 3`, returns the
  // unchanged 3-element triple; the post-trim seq-len read yields 3, so
  // `self.offset` clamps to 3.
  assert_eq!(
    c.offset(),
    3,
    "underlength forge: offset must clamp down to stored seq-len 3, not the forged 5"
  );
  // The reconstructed cache's stored triples are unchanged (slice was a
  // NumPy-clamped no-op): `quantized_state` is length 3 and dequantizes
  // to exactly kv(3).
  {
    let q = c.as_quantized().expect("reconstructed cache is quantized");
    let (qk, qv) = q.quantized_state().unwrap().unwrap();
    assert_eq!(
      qk.0.shape(),
      vec![1, 1, 3, HEAD_DIM / 4],
      "underlength forge: storage seq-len stays 3 (NumPy clamp on the trim)"
    );
    let mut dk = dequant(&qk);
    let mut dv = dequant(&qv);
    let mut want3k = kv(3);
    let mut want3v = kv(3);
    assert_close(&mut dk, &mut want3k);
    assert_close(&mut dv, &mut want3v);
  }
  // The next `update_quantized` must land the new token at position 3
  // (the CLAMPED offset), NOT 5 (the forged offset). Length 4 (t0,t1,t2,
  // NEW) — NOT 6 (which would be a phantom-slot gap from concat onto
  // length-3 storage with `prev = 5`, which mlx's `concat_seq` would
  // actually compute as the same length-4 array; the test of the
  // INVARIANT is offset and shape consistency after).
  let mut new_tok = kv_base(1, 100.0);
  let (qk, _) = {
    let q = c
      .as_quantized_mut()
      .expect("reconstructed cache downcasts mutably");
    q.update_quantized(&new_tok, &new_tok).unwrap()
  };
  // Clamped offset 3 + 1 step = 4 (NOT 5 + 1 = 6).
  assert_eq!(
    c.offset(),
    4,
    "post-clamp `update_quantized` lands at offset 3 + 1 = 4, NOT 5 + 1 = 6"
  );
  assert_eq!(
    qk.0.shape(),
    vec![1, 1, 4, HEAD_DIM / 4],
    "post-clamp `update_quantized` storage is length 4 (3 + 1), NOT length 6"
  );
  let mut dk = dequant(&qk);
  let want: Vec<f32> = {
    let mut a = kv(3).to_vec::<f32>().unwrap(); // t0..t2
    a.extend_from_slice(&new_tok.to_vec::<f32>().unwrap()); // NEW
    a
  };
  let mut want_arr = Array::from_slice::<f32>(&want, &(1usize, 1, 4usize, HEAD_DIM)).unwrap();
  assert_close(&mut dk, &mut want_arr);
  // Explicitly: position 3 is the NEW token (~100 range), NOT a phantom
  // zero/uninitialized slot, NOT a stale repeat of t2 — provably the
  // genuine post-clamp append.
  let got = dk.to_vec::<f32>().unwrap();
  let pos3: Vec<f32> = got[3 * HEAD_DIM..4 * HEAD_DIM].to_vec();
  let new_vals = new_tok.to_vec::<f32>().unwrap();
  let max_abs = new_vals.iter().fold(0.0f32, |m, x| m.max(x.abs()));
  for (g, n) in pos3.iter().zip(new_vals.iter()) {
    assert!(
      (g - n).abs() <= 0.05 * max_abs + 1e-3,
      "post-clamp append position 3 must be the NEW token: got={g} want={n}"
    );
  }
  // `quantized_state` is internally consistent: keys & values triples are
  // length 4 == offset, matching the offset-length storage invariant.
  {
    let q = c.as_quantized().expect("post-clamp cache still quantized");
    let (qk2, qv2) = q.quantized_state().unwrap().unwrap();
    assert_eq!(qk2.0.shape()[2], c.offset());
    assert_eq!(qv2.0.shape()[2], c.offset());
  }

  // A CONSISTENT state (seq-len == offset) round-trips byte-identically:
  // the symmetric clamp is a pure no-op for a faithfully saved state.
  let mut src2 = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src2.update_quantized(&kv(4), &kv(4)).unwrap();
  let consistent_st: Vec<Array> = src2
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let consistent_meta = src2.meta_state(); // ("4","64","8")
  assert_eq!(
    consistent_meta[0], "4",
    "honest meta offset matches seq-len"
  );
  let (sk, sv) = src2.quantized_state().unwrap().unwrap();
  let mut want_k = dequant(&sk);
  let mut want_v = dequant(&sv);

  let rt = from_state("QuantizedKVCache", consistent_st, &consistent_meta).unwrap();
  assert_eq!(
    rt.offset(),
    4,
    "consistent offset preserved (clamp is no-op)"
  );
  let q = rt.as_quantized().unwrap();
  let (rk, rv) = q.quantized_state().unwrap().unwrap();
  assert_eq!(rk.0.shape(), vec![1, 1, 4, HEAD_DIM / 4]);
  let mut got_k = dequant(&rk);
  let mut got_v = dequant(&rv);
  // Byte-identical dequantized content (clamp + slice both no-ops).
  let gk = got_k.to_vec::<f32>().unwrap();
  let wk = want_k.to_vec::<f32>().unwrap();
  assert_eq!(gk.len(), wk.len());
  for (g, w) in gk.iter().zip(wk.iter()) {
    assert_eq!(
      g, w,
      "consistent-state round-trip must be byte-identical under symmetric clamp (keys)"
    );
  }
  let gv = got_v.to_vec::<f32>().unwrap();
  let wv = want_v.to_vec::<f32>().unwrap();
  for (g, w) in gv.iter().zip(wv.iter()) {
    assert_eq!(
      g, w,
      "consistent-state round-trip must be byte-identical under symmetric clamp (values)"
    );
  }
}

/// **KVC-8 update (issue #105): post-fix behavior is REJECT, not clamp.**
/// Pre-KVC-8 the across-K/V asymmetric forge (keys stored seq-len 3,
/// values stored seq-len 5, meta offset 5) was accepted by `set_state` and
/// later *converged* via `enforce_offset_len_invariant` (silent fix-up).
/// Post-KVC-8 the eager K/V cross-validator in `set_state` REJECTS the
/// asymmetry upfront with a precise diagnostic at the load boundary — a
/// forged/corrupt prompt cache surfaces immediately instead of running
/// through a silent shape-converging code path. This is the documented
/// Rust-idiom upgrade for diagnosability (the lazy-error fix at the first
/// `update_quantized` is replaced with an eager error at the load
/// boundary).
#[test]
fn from_state_asymmetric_keys_shorter_is_rejected_at_set_state() {
  let mut src_short = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_short.update_quantized(&kv(3), &kv(3)).unwrap();
  let s_short: Vec<Array> = src_short
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let mut src_long = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_long.update_quantized(&kv(5), &kv(5)).unwrap();
  let s_long: Vec<Array> = src_long
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let forged_state: Vec<Array> = s_short[0..3]
    .iter()
    .map(|a| a.try_clone().unwrap())
    .chain(s_long[3..6].iter().map(|a| a.try_clone().unwrap()))
    .collect();
  let forged_meta = vec!["5".to_string(), "64".to_string(), "8".to_string()];

  let result = from_state("QuantizedKVCache", forged_state, &forged_meta);
  let err = match result {
    Err(e) => e,
    Ok(_) => panic!("post-KVC-8: asymmetric K/V forge must be REJECTED at set_state (not clamped)"),
  };
  let msg = err.to_string();
  assert!(
    msg.contains("set_state") && (msg.contains("K and V") || msg.contains("axis")),
    "diagnostic must name the load boundary + the K/V mismatch; got {msg}"
  );
}

/// **KVC-8 update (issue #105): see sibling test above.** Symmetric
/// counterpart: keys longer than values. Pre-KVC-8 the forge was clamped;
/// post-KVC-8 it is REJECTED at set_state with a precise diagnostic.
#[test]
fn from_state_asymmetric_values_shorter_is_rejected_at_set_state() {
  let mut src_short = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_short.update_quantized(&kv(3), &kv(3)).unwrap();
  let s_short: Vec<Array> = src_short
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let mut src_long = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_long.update_quantized(&kv(5), &kv(5)).unwrap();
  let s_long: Vec<Array> = src_long
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let forged_state: Vec<Array> = s_long[0..3]
    .iter()
    .map(|a| a.try_clone().unwrap())
    .chain(s_short[3..6].iter().map(|a| a.try_clone().unwrap()))
    .collect();
  let forged_meta = vec!["5".to_string(), "64".to_string(), "8".to_string()];

  let result = from_state("QuantizedKVCache", forged_state, &forged_meta);
  let err = match result {
    Err(e) => e,
    Ok(_) => panic!("post-KVC-8: asymmetric K/V forge must be REJECTED at set_state (not clamped)"),
  };
  let msg = err.to_string();
  assert!(
    msg.contains("set_state") && (msg.contains("K and V") || msg.contains("axis")),
    "diagnostic must name the load boundary + the K/V mismatch; got {msg}"
  );
}

/// Regression for the Copilot finding rid=4324739111 on top of the post-#80
/// across-K/V asymmetric-clamp landing: a forged state can have ASYMMETRIC
/// seq-lens *within* a single triple's `(weight, scales, biases)` components
/// (the analog of the across-K/V case one level down — mlx-lm's `state`
/// getter `tree_map(lambda x: x[..., :offset, :], ...)` applies the slice
/// per-component, and our `trim_triple` does the same, so each component
/// clamps to its own `min(offset, own_len)` independently — a forged blob
/// with mismatched component lengths is preserved by the per-component
/// clamp, NOT converged).
///
/// Construction: keys triple has `scales` forged to seq-len 3 (from a short
/// cache's `state[1]`) while `weight` and `biases` stay at seq-len 5 (from a
/// 5-step honest cache). Values triple stays honest (seq-len 5). meta
/// offset 5.
///
/// Expected post-restore: within-triple `min` of keys = `min(5, 3, 5) = 3`;
/// across-K/V `min(3, 5) = 3`; final `offset = 3`. Every component of every
/// triple is re-trimmed to seq-len 3 (the within-triple longer components
/// `w` and `biases` on keys MUST be sliced down; values triple MUST be
/// re-trimmed across-K/V down from 5 to 3). Subsequent `update_quantized`
/// lands the new token at position 3 (NOT 5) and produces internally
/// consistent triples.
#[test]
fn from_state_underlength_state_within_triple_asymmetric_clamps_to_min() {
  // Honest 3-step source: every component of every triple is seq-len 3.
  let mut src_short = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_short.update_quantized(&kv(3), &kv(3)).unwrap();
  let s_short: Vec<Array> = src_short
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  // Honest 5-step source: every component of every triple is seq-len 5.
  let mut src_long = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_long.update_quantized(&kv(5), &kv(5)).unwrap();
  let s_long: Vec<Array> = src_long
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  // Forge: keys triple = (w from long [seq=5], scales from short [seq=3],
  // biases from long [seq=5]); values triple = entire long (seq=5).
  // State layout (see `state()` getter @ quantized.rs:437-457):
  //   [0] = k_w, [1] = k_s, [2] = k_b, [3] = v_w, [4] = v_s, [5] = v_b.
  let forged_state: Vec<Array> = vec![
    s_long[0].try_clone().unwrap(),  // k_w (seq=5)
    s_short[1].try_clone().unwrap(), // k_s (seq=3) -- the forged short component
    s_long[2].try_clone().unwrap(),  // k_b (seq=5)
    s_long[3].try_clone().unwrap(),  // v_w (seq=5)
    s_long[4].try_clone().unwrap(),  // v_s (seq=5)
    s_long[5].try_clone().unwrap(),  // v_b (seq=5)
  ];
  // Sanity: confirm the forged within-triple asymmetry on keys.
  assert_eq!(
    forged_state[0].shape(),
    vec![1, 1, 5, HEAD_DIM / 4],
    "keys.weight forged seq=5"
  );
  assert_eq!(
    forged_state[1].shape(),
    vec![1, 1, 3, 1],
    "keys.scales forged seq=3 -- WITHIN-triple asymmetry"
  );
  assert_eq!(
    forged_state[2].shape(),
    vec![1, 1, 5, 1],
    "keys.biases forged seq=5"
  );
  let forged_meta = vec!["5".to_string(), "64".to_string(), "8".to_string()];

  // **KVC-8 update (issue #105): post-fix behavior is REJECT, not clamp.**
  // The within-triple asymmetry on keys (k_w seq=5, k_s seq=3) makes
  // k_s.shape != v_s.shape (v_s is seq=5), so the eager K/V validator
  // rejects at set_state — the across-K/V projection of the within-triple
  // inconsistency lands directly on the cross-validator's axis check. The
  // pre-KVC-8 silent "within-triple min" convergence path is gone.
  let result = from_state("QuantizedKVCache", forged_state, &forged_meta);
  let err = match result {
    Err(e) => e,
    Ok(_) => panic!(
      "post-KVC-8: within-triple asymmetry surfaces as a K/V scales shape mismatch \
       at set_state — REJECTED, not silently converged"
    ),
  };
  let msg = err.to_string();
  assert!(
    msg.contains("set_state") && (msg.contains("scales") || msg.contains("K and V")),
    "diagnostic must name the load boundary + the offending element; got {msg}"
  );
}

/// Edge case: a bias-less (4-array) state with within-triple asymmetry
/// across `(weight, scales)` only — `biases` is `None` and MUST NOT be
/// included in the within-triple min. Forged keys: `w` seq=5, `scales`
/// seq=3 (no biases). Final offset = min(5, 3) = 3 across K, min(3, 5) = 3
/// across K/V. The bias-less arm of `set_state` (4-array) is faithfully
/// preserved (no biases fabricated).
#[test]
fn from_state_underlength_state_within_triple_asymmetric_bias_less_clamps_to_min() {
  // Honest 3-step + 5-step sources, then strip biases (drop [2] and [5])
  // to build 4-array bias-less states.
  let mut src_short = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_short.update_quantized(&kv(3), &kv(3)).unwrap();
  let s_short_full: Vec<Array> = src_short
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let mut src_long = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  src_long.update_quantized(&kv(5), &kv(5)).unwrap();
  let s_long_full: Vec<Array> = src_long
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  // Build 4-array bias-less forged state: keys = (w from long, scales from
  // short); values = (w, scales) from long. Layout (no biases):
  //   [0] = k_w, [1] = k_s, [2] = v_w, [3] = v_s.
  let forged_state: Vec<Array> = vec![
    s_long_full[0].try_clone().unwrap(),  // k_w (seq=5)
    s_short_full[1].try_clone().unwrap(), // k_s (seq=3) -- forged short
    s_long_full[3].try_clone().unwrap(),  // v_w (seq=5)
    s_long_full[4].try_clone().unwrap(),  // v_s (seq=5)
  ];
  assert_eq!(forged_state.len(), 4, "bias-less state has 4 arrays");
  assert_eq!(forged_state[0].shape(), vec![1, 1, 5, HEAD_DIM / 4]);
  assert_eq!(forged_state[1].shape(), vec![1, 1, 3, 1]);
  let forged_meta = vec!["5".to_string(), "64".to_string(), "8".to_string()];

  // **KVC-8 update (issue #105): post-fix behavior is REJECT, not clamp.**
  // Same as the 6-array within-triple test above: the k_s seq=3 vs v_s
  // seq=5 mismatch is rejected by the eager K/V cross-validator at
  // set_state.
  let result = from_state("QuantizedKVCache", forged_state, &forged_meta);
  let err = match result {
    Err(e) => e,
    Ok(_) => panic!("post-KVC-8: bias-less within-triple asymmetry is REJECTED at set_state"),
  };
  let msg = err.to_string();
  assert!(
    msg.contains("set_state") && (msg.contains("scales") || msg.contains("K and V")),
    "diagnostic must name the load boundary + the offending element; got {msg}"
  );
}

/// `set_meta_state` accepts the mlx-swift-lm 4-string form `[step, offset,
/// groupSize, bits]` (`MLXLMCommon/KVCache.swift` `QuantizedKVCache.metaState`
/// setter ~line 952): `step` at index `[0]` is dropped on restore (it is a
/// pure over-allocation tuning param with no observable effect on the
/// cache's contract — same as swift, which restores only `offset` from
/// index `[1]`); `offset`/`group_size`/`bits` are restored from indices
/// `[1..=3]`. Resolves cross-runtime prompt-cache portability — a cache
/// saved by mlx-swift-lm now loads into the Rust runtime.
#[test]
fn set_meta_state_accepts_swift_4string_form() {
  // Start from a fresh cache with placeholder group_size/bits; the swift
  // 4-string meta restores them to the saved values.
  let mut c = QuantizedKvCacheImpl::new(0, 0);
  c.set_meta_state(&[
    "256".to_string(), // step (dropped, NOT stored)
    "10".to_string(),  // offset
    "64".to_string(),  // groupSize
    "4".to_string(),   // bits
  ])
  .unwrap();
  assert_eq!(c.offset(), 10, "offset restored from index [1]");
  let q = c.as_quantized().unwrap();
  assert_eq!(q.group_size(), 64, "group_size restored from index [2]");
  assert_eq!(q.bits(), 4, "bits restored from index [3]");
}

/// Regression guard for the pre-existing mlx-lm 3-string form `[offset,
/// group_size, bits]` (`cache.py:302-304`): extending the setter to also
/// accept the 4-string swift form must not change the 3-string path's
/// observable behavior.
#[test]
fn set_meta_state_accepts_mlx_lm_3string_form() {
  let mut c = QuantizedKvCacheImpl::new(0, 0);
  c.set_meta_state(&[
    "10".to_string(), // offset
    "64".to_string(), // group_size
    "4".to_string(),  // bits
  ])
  .unwrap();
  assert_eq!(c.offset(), 10);
  let q = c.as_quantized().unwrap();
  assert_eq!(q.group_size(), 64);
  assert_eq!(q.bits(), 4);
}

/// Any length other than 3 (mlx-lm) or 4 (mlx-swift-lm) is a recoverable
/// `Err` — both the 2-string and 5-string forms must be rejected with the
/// combined message (faithful semantics: same no-partial-mutation
/// invariant on bad arity).
#[test]
fn set_meta_state_rejects_2_or_5_string_form() {
  let mut c = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  // Too-short: 2 strings (was always rejected; combined message now lists
  // both accepted arities).
  let err2 = c
    .set_meta_state(&["10".to_string(), "64".to_string()])
    .unwrap_err()
    .to_string();
  assert!(
    err2.contains("3 (mlx-lm form)") && err2.contains("4 (mlx-swift-lm form)"),
    "2-string rejection message must list BOTH accepted forms; got: {err2}"
  );
  // Too-long: 5 strings.
  let err5 = c
    .set_meta_state(&[
      "10".to_string(),
      "20".to_string(),
      "64".to_string(),
      "4".to_string(),
      "1".to_string(),
    ])
    .unwrap_err()
    .to_string();
  assert!(
    err5.contains("3 (mlx-lm form)") && err5.contains("4 (mlx-swift-lm form)"),
    "5-string rejection message must list BOTH accepted forms; got: {err5}"
  );
  // The cache is unchanged after both failed attempts (no partial mutation).
  assert_eq!(c.offset(), 0);
  let q = c.as_quantized().unwrap();
  assert_eq!(q.group_size(), GROUP_SIZE);
  assert_eq!(q.bits(), BITS);
}

/// `from_state("QuantizedKVCache", state, swift_meta)` end-to-end:
/// reconstructing a cache from honest serialized triples + a 4-string
/// swift-form meta produces a cache observably equivalent to the same
/// reconstruction with the 3-string mlx-lm form (the swift `step` field
/// is dropped on restore — exactly mlx-swift-lm's own setter at
/// `KVCache.swift:952-961`).
#[test]
fn from_state_round_trip_via_swift_form() {
  // Build an honest 3-step source and capture its serialized state.
  let mut src = QuantizedKvCacheImpl::new(GROUP_SIZE, BITS);
  let mut k = kv(3);
  let mut v = kv(3);
  src.update_quantized(&k, &v).unwrap();
  let st: Vec<Array> = src
    .state()
    .unwrap()
    .iter()
    .map(|a| a.try_clone().unwrap())
    .collect();
  let st_a: Vec<Array> = st.iter().map(|a| a.try_clone().unwrap()).collect();
  let st_b: Vec<Array> = st.iter().map(|a| a.try_clone().unwrap()).collect();

  // mlx-lm 3-string meta: [offset, group_size, bits] = [3, 64, 8].
  let lm_meta = src.meta_state();
  assert_eq!(
    lm_meta,
    vec!["3".to_string(), "64".to_string(), "8".to_string()]
  );

  // mlx-swift-lm 4-string meta: [step, offset, groupSize, bits] =
  // [256, 3, 64, 8]. `step` is the swift over-allocation tuning param —
  // its value here does NOT affect the cache's observable contract.
  let swift_meta = vec![
    "256".to_string(),
    "3".to_string(),
    "64".to_string(),
    "8".to_string(),
  ];

  let c_lm = from_state("QuantizedKVCache", st_a, &lm_meta).unwrap();
  let c_sw = from_state("QuantizedKVCache", st_b, &swift_meta).unwrap();

  // Same offset / quantization params.
  assert_eq!(c_lm.offset(), c_sw.offset());
  assert_eq!(c_lm.offset(), 3);
  let q_lm = c_lm.as_quantized().unwrap();
  let q_sw = c_sw.as_quantized().unwrap();
  assert_eq!(q_lm.group_size(), q_sw.group_size());
  assert_eq!(q_lm.bits(), q_sw.bits());
  assert_eq!(q_lm.group_size(), GROUP_SIZE);
  assert_eq!(q_lm.bits(), BITS);

  // Byte-identical state shapes (same triples, same offset → same
  // post-`enforce_offset_len_invariant` no-op).
  let st_lm = c_lm.state().unwrap();
  let st_sw = c_sw.state().unwrap();
  assert_eq!(st_lm.len(), st_sw.len());
  for (a, b) in st_lm.iter().zip(st_sw.iter()) {
    assert_eq!(a.shape(), b.shape());
  }

  // Dequantized content is observably equivalent — both reconstructions
  // recover the original keys/values within the 8-bit affine quant band.
  let (qk_lm, qv_lm) = q_lm.quantized_state().unwrap().unwrap();
  let (qk_sw, qv_sw) = q_sw.quantized_state().unwrap().unwrap();
  let mut dk_lm = dequant(&qk_lm);
  let mut dv_lm = dequant(&qv_lm);
  let mut dk_sw = dequant(&qk_sw);
  let mut dv_sw = dequant(&qv_sw);
  assert_close(&mut dk_lm, &mut k);
  assert_close(&mut dv_lm, &mut v);
  assert_close(&mut dk_sw, &mut k);
  assert_close(&mut dv_sw, &mut v);
}
