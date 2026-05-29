//! Disk-persistence tests for `mlxrs::lm::cache::persist`
//! (`save_prompt_cache` / `load_prompt_cache` / `reference_class_name`),
//! ported 1:1 from `mlx_lm.models.cache` (`save_prompt_cache` /
//! `load_prompt_cache`, `cache.py:43-85`) and cross-checked against
//! mlx-swift-lm's `MLXLMCommon/KVCache.swift` (`savePromptCache` /
//! `loadPromptCache` + the `"i.j"` / `"0.i.j"` / `"1.key"` / `"2.i"`
//! `tree_flatten` wire format).
//!
//! Scope (#260): these cover the persist layer's *genuine* gaps not already
//! exercised by `lm_cache_prompt.rs` (which covers Standard/Rotating
//! round-trip, trim, missing/garbage/dir, wrong-rank, swift-5-field,
//! empty-state, inconsistent-rotating, all-`KVCache` scalar emission +
//! load, and the no-meta truthy-meta rejection). Specifically:
//!
//!   * round-trips through `save_prompt_cache` → `load_prompt_cache` for the
//!     cache kinds previously only tested through `from_serialized`
//!     (`ChunkedKVCache`, `QuantizedKVCache`, `CacheList`, `MambaCache`);
//!   * the `unflatten_arrays` / `unflatten_side` parser surface — out-of-
//!     order array keys, a trailing all-empty-state cache, dotted user-
//!     metadata keys;
//!   * the recoverable typed-`Err` paths the persist layer adds on top of
//!     `from_state` — unknown kind, out-of-range array group index, and the
//!     non-dense (corrupt-file) index gates in `dense_len`.
//!
//! Each on-disk key / reconstructed value is hand-traced from the cited
//! `persist.rs` doc lines so it is checkable, not assumed.

#![cfg(feature = "lm")]

use std::{collections::HashMap, fs, path::PathBuf, process};

use mlxrs::{
  Array, Error, io,
  lm::cache::{
    ArraysCache, CacheList, ChunkedKvCache, KvCache, QuantizedKvCache, QuantizedKvCacheImpl,
    RotatingKvCache, StandardKvCache, load_prompt_cache, reference_class_name, save_prompt_cache,
  },
  ops,
};

/// Unique temp path per test name (PID-scoped so parallel test bins do not
/// collide). Mirrors `lm_cache_prompt.rs::temp_path`.
fn temp_path(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_lm_cache_persist_{}_{}", process::id(), name));
  p
}

/// A `[1, 1, S, 1]` KV tensor whose only varying axis is the sequence axis,
/// each step's value being its 0-based token id (so retained ids / round-
/// trip values are directly readable). `S == ids.len()`. Identical to the
/// ramp used across the cache module's existing tests.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

/// Distinct, sequence-position-AND-column-varying KEY fixture `[1, 1, S, 64]`
/// for the quantized round-trip (`head_dim == 64`, one affine quant group per
/// row). Row `s`, column `j` holds `j + s*64`, so EVERY (step, column) cell is
/// unique and rows differ by step. This is deliberately NOT row-repeated and
/// NOT equal to [`kv_quant_values`] — so a persist bug that swaps K↔V,
/// duplicates one side, or reorders sequence rows (all shape-preserving) is
/// caught by the dequant-equality checks, which the old shared `kv_quant`
/// fixture (identical `[0..63]` row for both K and V, every step) could not
/// detect. Values stay small (max `63 + (S-1)*64`) so 8-bit affine
/// quantization (group_size 64) round-trips deterministically.
fn kv_quant_keys(n_steps: usize) -> Array {
  let mut data = Vec::with_capacity(n_steps * 64);
  for s in 0..n_steps {
    for j in 0..64 {
      data.push(j as f32 + s as f32 * 64.0);
    }
  }
  Array::from_slice::<f32>(&data, &(1usize, 1, n_steps, 64usize)).unwrap()
}

/// Distinct VALUE fixture `[1, 1, S, 64]` paired with [`kv_quant_keys`]. Row
/// `s`, column `j` holds `(63 - j)*2 + s*128 + 4096` — column-reversed,
/// sequence-position-varying, offset by +4096. Crucially the per-row SPAN is
/// 126 (spacing 2) versus the keys' span of 63 (R5): a DIFFERENT per-row range
/// yields a different affine SCALE, a different min yields a different BIAS, and
/// the reversed pattern yields different packed WEIGHTS — so all three of the
/// value's quantized slots differ from the key's, and a persist bug swapping
/// ANY single one of the six on-disk arrays (incl. K-scales ↔ V-scales or
/// K-biases ↔ V-biases) changes the dequantized output. The +128 per-step
/// offset keeps rows non-overlapping; values stay bounded so 8-bit affine
/// quantization (group_size 64) round-trips deterministically.
fn kv_quant_values(n_steps: usize) -> Array {
  let mut data = Vec::with_capacity(n_steps * 64);
  for s in 0..n_steps {
    for j in 0..64 {
      data.push((63 - j) as f32 * 2.0 + s as f32 * 128.0 + 4096.0);
    }
  }
  Array::from_slice::<f32>(&data, &(1usize, 1, n_steps, 64usize)).unwrap()
}

// ─────────────────── round-trips for kinds only previously ───────────────
// ─────────────────── tested through `from_serialized` ────────────────────

#[test]
fn chunked_kvcache_round_trips_through_persist() {
  // ChunkedKVCache carries a NON-EMPTY 2-element meta_state
  // `[chunk_size, start_position]` (chunked.rs:584-589), so it exercises
  // the `"0.{i}.{j}"` LIST meta_state write/read path of persist (not the
  // scalar `"0.{i}"` empty form), plus the `ChunkedKVCache` `from_state`
  // arm — neither of which the existing persist tests touch (they only use
  // Standard + Rotating). 1:1 port of cache.py:43-85.
  let path = temp_path("chunked_rt.safetensors");

  let mut c = ChunkedKvCache::new(Some(8));
  c.update(&kv(&[1.0, 2.0, 3.0]), &kv(&[4.0, 5.0, 6.0]))
    .unwrap();
  let want_offset = c.offset();
  let want_meta = c.meta_state();
  assert_eq!(
    want_meta.len(),
    2,
    "chunked meta_state is [chunk, start_pos]"
  );

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(c)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  // On disk: class name "ChunkedKVCache" under "2.0", and the 2-element
  // meta_state under the LIST form "0.0.0"/"0.0.1" (not a scalar "0.0").
  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(
    raw_meta.get("2.0").map(String::as_str),
    Some("ChunkedKVCache")
  );
  assert_eq!(
    raw_meta.get("0.0.0").map(String::as_str),
    Some(want_meta[0].as_str())
  );
  assert_eq!(
    raw_meta.get("0.0.1").map(String::as_str),
    Some(want_meta[1].as_str())
  );
  assert_eq!(raw_meta.get("0.0"), None, "list meta, not scalar");

  // Round-trip: reconstructs a ChunkedKVCache with identical offset,
  // meta_state, and key/value contents.
  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(loaded[0].reference_class_name(), "ChunkedKVCache");
  assert_eq!(loaded[0].offset(), want_offset);
  assert_eq!(loaded[0].meta_state(), want_meta);
  let mut s = loaded[0].state().unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!(s[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
  assert_eq!(s[1].to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);

  let _ = fs::remove_file(&path);
}

/// Dequantize a 6-array quantized `state()` (`[k.w, k.scales, k.biases, v.w,
/// v.scales, v.biases]`, quantized.rs:644-668) into `(dense_keys, dense_values)`
/// via the merged `ops::quantized::dequantize` (gs=64, bits=8, affine — the
/// `QuantizedKvCacheImpl::new(64, 8)` params). Identical packed bytes
/// dequantize to identical dense values, so this is an EXACT (not banded)
/// content comparison after a lossless safetensors round-trip.
fn dequant_quant_state(state: &[Array]) -> (Array, Array) {
  assert_eq!(state.len(), 6, "affine quantized state is 6 arrays");
  let dk = ops::quantized::dequantize(
    &state[0],
    &state[1],
    Some(&state[2]),
    64,
    8,
    "affine",
    None,
    None,
  )
  .unwrap();
  let dv = ops::quantized::dequantize(
    &state[3],
    &state[4],
    Some(&state[5]),
    64,
    8,
    "affine",
    None,
    None,
  )
  .unwrap();
  (dk, dv)
}

#[test]
fn quantized_kvcache_round_trips_through_persist() {
  // QuantizedKVCache carries a 3-element meta_state `[offset, group_size,
  // bits]` (quantized.rs:783-789) and a 6-array packed state — the
  // `"0.{i}.{0,1,2}"` list-meta path + the `QuantizedKVCache` from_state
  // arm, neither covered by the existing persist tests.
  //
  // Strengthened per Codex #3 (which noted the original compared only meta +
  // packed-state SHAPES, so a corrupted/zeroed packed payload would pass):
  // this now compares the packed state by CONTENT. `Array::to_vec` errors
  // `NonContiguous` on a strided slice and the packed weight is U32 while
  // scales/biases are F32, so a per-array `to_vec` is dtype-fragile; instead
  // the saved vs loaded state is compared through `dequantize` (the
  // contiguity-safe, dtype-uniform F32 reconstruction the cache semantically
  // holds — Codex's "compare dequantized state before/after"). Because
  // safetensors round-trips the packed bytes losslessly, the two dequantize
  // EXACTLY equal (not just within the quant band). 1:1 port of cache.py:43-85.
  let path = temp_path("quantized_rt.safetensors");

  let mut c = QuantizedKvCacheImpl::new(64, 8).unwrap();
  // Codex #3 R2: the keys/values fixtures are now DISTINCT and vary by
  // sequence position (`kv_quant_keys` row = `j + s*64`; `kv_quant_values`
  // row = `(63 - j) + s*64 + 4096`). The old test reused `kv_quant` (the
  // identical `[0..63]` row for K, V, and EVERY step), so its dequant-equality
  // checks were blind to a persist bug that swaps K↔V, duplicates one side, or
  // reorders sequence rows (all shape-preserving). With distinct, per-step,
  // per-column fixtures those bugs now change at least one dequantized cell.
  c.update_quantized(&kv_quant_keys(3), &kv_quant_values(3))
    .unwrap();
  let want_offset = c.offset();
  let want_meta = c.meta_state();
  assert_eq!(want_meta.len(), 3, "quantized meta is [offset, gs, bits]");
  // `Array::shape()` returns an owned `Vec<usize>` and does NOT eval, so it
  // is safe to read off the `&self` state arrays for a shape-only compare.
  let want_state = c.state().unwrap();
  let want_state_shapes: Vec<Vec<usize>> = want_state.iter().map(|a| a.shape()).collect();
  // Capture the saved dense K/V CONTENTS (before `c` is moved into `cache`).
  let (mut want_dk, mut want_dv) = dequant_quant_state(&want_state);
  // head_dim stays 64 (one affine quant group per row): dequantize reconstructs
  // the original `[1, 1, 3, 64]` dense shape, confirming the fixtures kept a
  // single quant group per row (group_size 64 == head_dim).
  assert_eq!(
    want_dk.shape(),
    vec![1, 1, 3, 64],
    "dequantized keys must be [1,1,3,64] -> head_dim stayed 64 (one group/row)"
  );
  assert_eq!(
    want_dv.shape(),
    vec![1, 1, 3, 64],
    "dequantized values [1,1,3,64]"
  );
  let want_dk_vec = want_dk.to_vec::<f32>().unwrap();
  let want_dv_vec = want_dv.to_vec::<f32>().unwrap();
  assert!(
    !want_dk_vec.is_empty() && !want_dv_vec.is_empty(),
    "dense state must be non-empty (3 steps x 64 dims)"
  );
  // PRECONDITIONS (Codex #3 R2) — prove the fixtures genuinely exercise
  // side (K vs V) and sequence-order sensitivity on the dequantized in-memory
  // state, so the round-trip equality below is not vacuous:
  //   (a) dequantized K differs from dequantized V (distinct sides, +4096
  //       offset survives 8-bit affine quantization);
  //   (b) within K, the first sequence row differs from the last (rows vary by
  //       step, so a row-reorder/duplicate bug is observable).
  assert_ne!(
    want_dk_vec, want_dv_vec,
    "K and V fixtures must dequantize to DIFFERENT dense values (else a K<->V \
     swap or one-sided duplicate in persist would pass undetected)"
  );
  let row_len = 64; // head_dim
  assert_eq!(want_dk_vec.len(), 3 * row_len, "3 steps x 64 dims");
  assert_ne!(
    &want_dk_vec[..row_len],
    &want_dk_vec[(want_dk_vec.len() - row_len)..],
    "K row 0 must differ from K row 2 (rows vary by sequence step, so a \
     row-reorder/duplicate in persist would be observable)"
  );
  assert_ne!(
    &want_dv_vec[..row_len],
    &want_dv_vec[(want_dv_vec.len() - row_len)..],
    "V row 0 must differ from V row 2 (rows vary by sequence step)"
  );

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(c)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  let (mut arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(
    raw_meta.get("2.0").map(String::as_str),
    Some("QuantizedKVCache")
  );
  assert_eq!(
    raw_meta.get("0.0.0").map(String::as_str),
    Some(want_meta[0].as_str())
  );
  assert_eq!(
    raw_meta.get("0.0.2").map(String::as_str),
    Some(want_meta[2].as_str())
  );

  // Codex R3+R4: verify the RAW on-disk array slots against an oracle that is
  // INDEPENDENT of `c.state()`. The six packed arrays are written under "0.{j}"
  // (cache 0, array j) in order [k.w, k.scales, k.biases, v.w, v.scales,
  // v.biases]. Dequantizing the on-disk slots IN ORDER and comparing them to the
  // ORIGINAL dense INPUT fixtures (`kv_quant_keys`/`kv_quant_values`, created by
  // the test — NOT derived from `state()`/`save_prompt_cache`) pins the semantic
  // wire format: a save/load that writes arrays to the wrong slots, swaps K<->V,
  // or reorders sequence rows diverges from the input here even if it round-trips
  // self-consistently. (R4: the prior compare to `want_*` was self-referential —
  // both `want_*` and the slots derive from `state()`.)
  let mut raw_state: Vec<Array> = (0..6)
    .map(|j| {
      arrays
        .remove(&format!("0.{j}"))
        .unwrap_or_else(|| panic!("missing on-disk quantized array slot 0.{j}"))
    })
    .collect();
  // R5: every one of the six slots must be individually distinguishable, so a
  // swap/duplicate of ANY single on-disk array changes the dequantized output.
  // Keys span 63 / values span 126 give DIFFERENT affine scales (slots 0.1 vs
  // 0.4); different mins give different biases (slots 0.2 vs 0.5). Assert those
  // slot pairs differ so a same-range regression cannot silently reappear.
  assert_ne!(
    raw_state[1].to_vec::<f32>().unwrap(),
    raw_state[4].to_vec::<f32>().unwrap(),
    "K-scales (slot 0.1) and V-scales (slot 0.4) must differ — distinct per-row ranges"
  );
  assert_ne!(
    raw_state[2].to_vec::<f32>().unwrap(),
    raw_state[5].to_vec::<f32>().unwrap(),
    "K-biases (slot 0.2) and V-biases (slot 0.5) must differ — distinct per-row mins"
  );
  let (mut raw_dk, mut raw_dv) = dequant_quant_state(&raw_state);
  let raw_dk_vec = raw_dk.to_vec::<f32>().unwrap();
  let raw_dv_vec = raw_dv.to_vec::<f32>().unwrap();
  // Independent oracle = the dense f32 inputs to `update_quantized`. Compare
  // within the 8-bit affine quant band: keys span 63 (max error 63/510 ~= 0.124),
  // values span 126 (max error 126/510 ~= 0.247); QUANT_TOL 0.5 sits above the
  // larger error yet far below the 1.0 key element spacing and the 64/128-per-row
  // and 4096-K-vs-V separations, so a wrong-slot / swap / reorder exceeds it.
  const QUANT_TOL: f32 = 0.5;
  let exp_keys = kv_quant_keys(3).to_vec::<f32>().unwrap();
  let exp_values = kv_quant_values(3).to_vec::<f32>().unwrap();
  assert_eq!(
    raw_dk_vec.len(),
    exp_keys.len(),
    "raw key slots dequantize to the input key element count"
  );
  assert_eq!(
    raw_dv_vec.len(),
    exp_values.len(),
    "raw value slots dequantize to the input value element count"
  );
  for (i, (got, exp)) in raw_dk_vec.iter().zip(exp_keys.iter()).enumerate() {
    assert!(
      (got - exp).abs() <= QUANT_TOL,
      "RAW on-disk KEY slot element {i}: dequant {got} vs original input fixture \
       {exp} exceeds the quant band (wrong wire slot / K<->V swap / row reorder)"
    );
  }
  for (i, (got, exp)) in raw_dv_vec.iter().zip(exp_values.iter()).enumerate() {
    assert!(
      (got - exp).abs() <= QUANT_TOL,
      "RAW on-disk VALUE slot element {i}: dequant {got} vs original input fixture \
       {exp} exceeds the quant band (wrong wire slot / K<->V swap / row reorder)"
    );
  }

  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(loaded[0].reference_class_name(), "QuantizedKVCache");
  assert_eq!(loaded[0].offset(), want_offset);
  assert_eq!(loaded[0].meta_state(), want_meta);
  let loaded_state = loaded[0].state().unwrap();
  let loaded_shapes: Vec<Vec<usize>> = loaded_state.iter().map(|a| a.shape()).collect();
  assert_eq!(
    loaded_shapes, want_state_shapes,
    "quantized packed-state shapes must round-trip"
  );
  // CONTENT round-trip: the loaded packed state dequantizes to the SAME dense
  // K/V as the saved one, byte-for-byte (lossless packed-byte round-trip).
  let (mut got_dk, mut got_dv) = dequant_quant_state(&loaded_state);
  assert_eq!(
    got_dk.to_vec::<f32>().unwrap(),
    want_dk_vec,
    "loaded quantized keys must dequantize to the saved dense keys exactly"
  );
  assert_eq!(
    got_dv.to_vec::<f32>().unwrap(),
    want_dv_vec,
    "loaded quantized values must dequantize to the saved dense values exactly"
  );

  let _ = fs::remove_file(&path);
}

#[test]
fn cache_list_round_trips_through_persist() {
  // A top-level `CacheList` is the composite kind: its flattened
  // `meta_state` is the framing list `[childCount, (class, stateCount,
  // metaCount, ...meta)*]` (cache_list.rs:325-352) written as
  // `"0.0.{j}"`, its `state` is every child's arrays flattened as
  // `"0.{j}"`, and its class is `"CacheList"` under `"2.0"`. `from_state`
  // rebuilds each child recursively. None of the existing persist tests
  // round-trip a CacheList through save/load — this covers the
  // `cache_list_from_state` arm via the persistence layer. 1:1 port of
  // cache.py:43-85 / KVCache.swift CacheList.
  let path = temp_path("cache_list_rt.safetensors");

  let mut child0 = StandardKvCache::new();
  child0.update(&kv(&[1.0, 2.0]), &kv(&[3.0, 4.0])).unwrap();
  let mut child1 = RotatingKvCache::new(8, 4);
  child1
    .update(&kv(&[5.0, 6.0, 7.0]), &kv(&[8.0, 8.0, 8.0]))
    .unwrap();
  let list = CacheList::new(vec![Box::new(child0), Box::new(child1)]);
  assert_eq!(list.reference_class_name(), "CacheList");

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(list)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(raw_meta.get("2.0").map(String::as_str), Some("CacheList"));

  // Round-trip: a single top-level CacheList holding the two children.
  let (mut loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(loaded[0].reference_class_name(), "CacheList");
  let restored = loaded[0]
    .as_cache_list_mut()
    .expect("loaded CacheList must downcast to a CacheList");
  assert_eq!(restored.len(), 2);
  // Child reference class names + offsets survive the recursive rebuild.
  assert_eq!(restored.get(0).unwrap().reference_class_name(), "KVCache");
  assert_eq!(
    restored.get(1).unwrap().reference_class_name(),
    "RotatingKVCache"
  );
  assert_eq!(restored.get(0).unwrap().offset(), 2);
  assert_eq!(restored.get(1).unwrap().offset(), 3);
  // Child 0 (StandardKvCache) key/value contents round-trip.
  let mut s0 = restored.get(0).unwrap().state().unwrap();
  assert_eq!(s0[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
  assert_eq!(s0[1].to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
  // Child 1 (RotatingKvCache) key/value contents ALSO round-trip
  // (Codex #3: the original only checked child 0, so a corrupted child-1
  // state would go undetected). The single S=3 prefill update from empty
  // takes `update_concat`'s empty-cache branch (`(keys, values).try_clone`,
  // rotating.rs:249-251), so the buffer is exactly [5,6,7] / [8,8,8]; with
  // offset 3 == buffer len 3 `state()` returns the full buffer (NOT a
  // shorter slice) (rotating.rs:469-483).
  let mut s1 = restored.get(1).unwrap().state().unwrap();
  assert_eq!(s1.len(), 2, "rotating child state is [keys, values]");
  assert_eq!(s1[0].to_vec::<f32>().unwrap(), vec![5.0, 6.0, 7.0]);
  assert_eq!(s1[1].to_vec::<f32>().unwrap(), vec![8.0, 8.0, 8.0]);

  let _ = fs::remove_file(&path);
}

#[test]
fn mamba_arrays_cache_round_trips_through_persist() {
  // A `MambaCache` (an `ArraysCache` with the swift `MambaCache`
  // provenance, KVC-9) is a NON-KV kind: it is in neither persist's
  // `KV_RANK_KINDS` (so the 4-D rank gate is skipped — forward-compat path
  // for non-4-D-state caches) NOR `NO_META_KINDS` (its meta_state is a
  // genuine slot-aware list). An EMPTY one still writes its class +
  // meta_state and reconstructs to the right concrete type with the
  // `"MambaCache"` provenance preserved across the round-trip
  // (arrays.rs:232-251, from_state MambaCache arm). No existing persist
  // test exercises a non-KV cache kind through save/load.
  let path = temp_path("mamba_rt.safetensors");

  let mamba = ArraysCache::mamba();
  assert_eq!(mamba.reference_class_name(), "MambaCache");
  let want_meta = mamba.meta_state();

  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(mamba)];
  save_prompt_cache(&path, &cache, &HashMap::new()).unwrap();

  // Class label is the swift "MambaCache" (NOT degraded to "ArraysCache").
  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(raw_meta.get("2.0").map(String::as_str), Some("MambaCache"));

  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  // Provenance survives: a save-after-load would re-emit "MambaCache".
  assert_eq!(loaded[0].reference_class_name(), "MambaCache");
  assert_eq!(loaded[0].meta_state(), want_meta);
  assert!(loaded[0].is_empty(), "an empty Mamba cache stays empty");

  let _ = fs::remove_file(&path);
}

// ───────────────── parser surface: unflatten_arrays / _side ───────────────

#[test]
fn out_of_order_array_keys_round_trip() {
  // mlx-c's safetensors map iteration order is unspecified, so
  // `unflatten_arrays` builds a doubly-sparse map and reorders the per-cache
  // arrays by their parsed sub-index `j` (persist.rs:335-371) rather than by
  // load-time map order. This locks the load-side invariant: state array `j`
  // lands at `state()[j]` (here a single KVCache whose two arrays carry
  // distinct, sub-index-tagged values so the ordering is directly readable).
  let path = temp_path("out_of_order.safetensors");

  let mut arrays = HashMap::new();
  // Insert sub-index 1 first, then 0 — order must NOT leak into the result.
  arrays.insert("0.1".to_string(), kv(&[40.0, 50.0]));
  arrays.insert("0.0".to_string(), kv(&[10.0, 20.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), String::new()); // faithful empty scalar meta
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  let mut s = loaded[0].state().unwrap();
  assert_eq!(s.len(), 2);
  // Array 0 (keys) is the [10,20] tensor, array 1 (values) the [40,50] one,
  // i.e. ordered by sub-index `j`, NOT by HashMap insertion order.
  assert_eq!(s[0].to_vec::<f32>().unwrap(), vec![10.0, 20.0]);
  assert_eq!(s[1].to_vec::<f32>().unwrap(), vec![40.0, 50.0]);

  let _ = fs::remove_file(&path);
}

#[test]
fn trailing_empty_state_cache_reconstructs_faithfully() {
  // persist.rs documents (313-329 / 636-646) that the cache COUNT is the
  // `cache_classes` length, NOT the array-map size, so a trailing cache
  // whose state is `[]` (emits no "{i}.{j}" array keys) is reconstructed
  // faithfully — where mlx-lm's `zip(classes, arrays, info)` would silently
  // DROP it and mlx-swift's `cacheData.count == cacheClasses.count` guard
  // would REJECT the file. Hand-build a 2-class file where only cache 0
  // has arrays; cache 1 (KVCache, empty) must still come back.
  let path = temp_path("trailing_empty.safetensors");

  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0, 2.0]));
  arrays.insert("0.1".to_string(), kv(&[3.0, 4.0]));
  // NOTE: no "1.*" array keys — cache 1's state is empty.
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("2.1".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), String::new());
  side.insert("0.1".to_string(), String::new());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  let (loaded, _m) = load_prompt_cache(&path).unwrap();
  assert_eq!(
    loaded.len(),
    2,
    "trailing empty-state cache must NOT be dropped"
  );
  // Cache 0 has its arrays; cache 1 is a fresh empty KVCache.
  assert_eq!(loaded[0].offset(), 2);
  assert!(!loaded[0].is_empty());
  assert!(loaded[1].is_empty());
  assert_eq!(loaded[1].offset(), 0);

  let _ = fs::remove_file(&path);
}

#[test]
fn user_metadata_key_with_dots_round_trips() {
  // User metadata is written under "1.{key}" and read back as the verbatim
  // remainder after the first '.', so a key that itself contains dots
  // (swift `components.dropFirst().joined(".")`, persist.rs:456-459) must
  // survive intact. The existing persist tests only use a dot-free key
  // ("model"); this locks the dotted-key remainder semantics.
  let path = temp_path("dotted_meta.safetensors");

  let mut meta = HashMap::new();
  meta.insert("a.b.c".to_string(), "nested-value".to_string());
  meta.insert("plain".to_string(), "v".to_string());

  let std_c = StandardKvCache::new();
  let cache: Vec<Box<dyn KvCache>> = vec![Box::new(std_c)];
  save_prompt_cache(&path, &cache, &meta).unwrap();

  // On disk: "1.a.b.c" (the whole dotted key after the "1." tag).
  let (_arrays, raw_meta) = io::load_safetensors_with_metadata(&path).unwrap();
  assert_eq!(
    raw_meta.get("1.a.b.c").map(String::as_str),
    Some("nested-value")
  );

  let (_loaded, loaded_meta) = load_prompt_cache(&path).unwrap();
  assert_eq!(
    loaded_meta.get("a.b.c").map(String::as_str),
    Some("nested-value"),
    "a dotted user-metadata key must round-trip verbatim"
  );
  assert_eq!(loaded_meta.get("plain").map(String::as_str), Some("v"));

  let _ = fs::remove_file(&path);
}

// ─────────────────── recoverable typed-Err paths ──────────────────────────

#[test]
fn unknown_cache_kind_is_err_not_panic() {
  // A file naming cache 0 with a kind `from_state` does not recognize
  // (`KvCacheKind::parse` → `Error::UnknownEnumValue`, wrapped by
  // load_prompt_cache in a `LayerKeyed`, persist.rs:815-823). Must be a
  // clean recoverable Err, never a panic. Not in `KV_RANK_KINDS` so the
  // rank gate is skipped and `from_state` is what rejects it.
  let path = temp_path("unknown_kind.safetensors");

  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "BogusCacheKind".to_string());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Codex #4: pin the CONCRETE nested variant, not just `is_err()` (a generic
  // earlier failure would also be Err). `from_state` rejects the kind via
  // `KvCacheKind::parse` → `Error::UnknownEnumValue` (mod.rs:763-764), which
  // `load_prompt_cache` wraps in a `LayerKeyed` (persist.rs:815-823).
  let err = load_prompt_cache(&path)
    .err()
    .expect("an unknown cache kind must be a recoverable Err");
  match &err {
    Error::LayerKeyed(p) => match p.inner() {
      Error::UnknownEnumValue(inner) => {
        assert_eq!(
          inner.type_name(),
          "KvCacheKind",
          "the wrapped variant must name the KvCacheKind enum"
        );
        assert_eq!(
          inner.value(),
          "BogusCacheKind",
          "and carry the offending kind string"
        );
      }
      other => panic!("LayerKeyed inner must be UnknownEnumValue, got {other:?}"),
    },
    other => panic!("unknown kind must be Err(LayerKeyed(UnknownEnumValue)), got {other:?}"),
  }

  let _ = fs::remove_file(&path);
}

#[test]
fn array_group_index_out_of_range_is_err() {
  // An array group key "{i}.{j}" whose `i >= cache_classes.len()` is the
  // ONLY genuine array/class inconsistency persist flags (a trailing empty
  // cache is fine; a trailing array WITHOUT a class is corrupt). Here one
  // class ("2.0") but an array keyed "5.0" → `Error::OutOfRange`
  // (persist.rs:647-654), never a panic.
  let path = temp_path("array_oob.safetensors");

  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0]));
  arrays.insert("0.1".to_string(), kv(&[2.0]));
  // Array group index 5 with no matching "2.5" class.
  arrays.insert("5.0".to_string(), kv(&[3.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), String::new());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Codex #4: pin the CONCRETE variant + payload. One class ("2.0") so
  // class_count=1; the lone array group with `i >= 1` is index 5, surfaced as
  // `Error::OutOfRange` with the index/class-count context (persist.rs:648-654).
  let err = load_prompt_cache(&path)
    .err()
    .expect("an array group index past the class count must be Err");
  match &err {
    Error::OutOfRange(p) => {
      assert_eq!(
        p.context(),
        "load_prompt_cache: array group index (corrupt or incompatible file)"
      );
      assert_eq!(p.requirement(), "must be < class count");
      assert!(
        p.value().contains("index=5") && p.value().contains("class_count=1"),
        "value must name the offending index and class count, got: {}",
        p.value()
      );
    }
    other => panic!("array-group OOB must be Err(OutOfRange), got {other:?}"),
  }

  let _ = fs::remove_file(&path);
}

#[test]
fn non_dense_class_indices_is_err() {
  // `cache_classes` is a dense list (one "2.{i}" per cache, 0..len).
  // A file with classes at "2.0" and "2.2" but a GAP at "2.1" is corrupt /
  // incompatible — `dense_len` (persist.rs:283-311, the "class" call site)
  // rejects a non-dense list as `Error::LengthMismatch` rather than silently
  // allocating a sparse list or panicking. Not exercised by existing tests.
  let path = temp_path("nondense_class.safetensors");

  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0]));
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("2.2".to_string(), "KVCache".to_string()); // gap at index 1
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Codex #4: pin the CONCRETE variant + the `dense_len` context. Classes are
  // present at indices {0,2} (present=2) with max index 2 (n = 2+1 = 3), so
  // `dense_len(.., "class")` rejects the gap as `Error::LengthMismatch` with
  // expected=present=2, actual=n=3 (persist.rs:299-309, "class" call site at
  // 513). The single dense array group ("0.0") passes `unflatten_arrays`
  // first, so this class check is the failing one.
  let err = load_prompt_cache(&path)
    .err()
    .expect("non-dense class indices (gap) must be a recoverable Err");
  match &err {
    Error::LengthMismatch(p) => {
      assert_eq!(
        p.context(),
        "prompt cache: non-dense class indices (corrupt or incompatible file)"
      );
      assert_eq!(p.expected(), 2, "present (distinct) class indices");
      assert_eq!(p.actual(), 3, "max class index + 1");
    }
    other => panic!("non-dense class indices must be Err(LengthMismatch), got {other:?}"),
  }

  let _ = fs::remove_file(&path);
}

#[test]
fn non_dense_array_sub_indices_is_err() {
  // The inner per-cache array list is dense (sub-indices 0..len). A cache
  // whose arrays are at "0.0" and "0.2" but with a GAP at "0.1" is corrupt
  // — `dense_len` (the "array sub" call site, persist.rs:357) rejects it as
  // `Error::LengthMismatch`, bounding the allocation by the present-key
  // count so a hostile sparse side-table cannot drive an unbounded Vec.
  let path = temp_path("nondense_array_sub.safetensors");

  let mut arrays = HashMap::new();
  arrays.insert("0.0".to_string(), kv(&[1.0]));
  arrays.insert("0.2".to_string(), kv(&[2.0])); // gap at sub-index 1
  let mut side = HashMap::new();
  side.insert("2.0".to_string(), "KVCache".to_string());
  side.insert("0.0".to_string(), String::new());
  io::save_safetensors_with_metadata(&path, &arrays, &side).unwrap();

  // Codex #4: pin the CONCRETE variant + the `dense_len` context. Cache 0's
  // arrays are at sub-indices {0,2} (present=2) with max 2 (n = 2+1 = 3), so
  // `dense_len(.., "array sub")` rejects the gap as `Error::LengthMismatch`
  // with expected=present=2, actual=n=3 (persist.rs:357, "array sub" call
  // site). `unflatten_arrays` runs before `unflatten_side`, so this is the
  // failing check.
  let err = load_prompt_cache(&path)
    .err()
    .expect("non-dense array sub-indices (gap) must be a recoverable Err");
  match &err {
    Error::LengthMismatch(p) => {
      assert_eq!(
        p.context(),
        "prompt cache: non-dense array sub indices (corrupt or incompatible file)"
      );
      assert_eq!(p.expected(), 2, "present (distinct) array sub-indices");
      assert_eq!(p.actual(), 3, "max array sub-index + 1");
    }
    other => panic!("non-dense array sub-indices must be Err(LengthMismatch), got {other:?}"),
  }

  let _ = fs::remove_file(&path);
}

// ─────────────────── free reference_class_name fn ─────────────────────────

#[test]
fn reference_class_name_free_fn_matches_concrete_kinds() {
  // The free `persist::reference_class_name(&dyn KvCache)` is a thin
  // forward to the trait method (persist.rs:182-184) — it is what the saver
  // calls per cache and what `from_state` keys on. The trait method itself
  // is covered in `lm_cache_reference_class_name.rs`; this asserts the FREE
  // function dispatches identically across a representative spread of kinds.
  let std_c = StandardKvCache::new();
  let rot_c = RotatingKvCache::new(8, 4);
  let chunk_c = ChunkedKvCache::new(Some(8));
  let quant_c = QuantizedKvCacheImpl::new(64, 8).unwrap();
  let mamba_c = ArraysCache::mamba();
  let list_c = CacheList::new(Vec::new());

  assert_eq!(reference_class_name(&std_c), "KVCache");
  assert_eq!(reference_class_name(&rot_c), "RotatingKVCache");
  assert_eq!(reference_class_name(&chunk_c), "ChunkedKVCache");
  assert_eq!(reference_class_name(&quant_c), "QuantizedKVCache");
  assert_eq!(reference_class_name(&mamba_c), "MambaCache");
  assert_eq!(reference_class_name(&list_c), "CacheList");
  // Free fn and trait method agree (it is just a forward).
  assert_eq!(reference_class_name(&std_c), std_c.reference_class_name());
}
