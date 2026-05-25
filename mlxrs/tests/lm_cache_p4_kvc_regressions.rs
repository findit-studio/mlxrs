//! P4 KVC structural-soundness regression tests — issues #98, #100, #105,
//! #106, #107. Each section pins the post-fix contract with a focused
//! negative + positive pair so a future regression is a deterministic test
//! failure rather than a silent re-introduction.

#![cfg(feature = "lm")]

use mlxrs::{
  Array, Error, Result,
  lm::cache::{
    ArraysCache, CacheList, KvCache, KvCacheKind, MaskMode, QuantizedKvCache, QuantizedKvCacheImpl,
    RotatingKvCache, StandardKvCache, from_state,
  },
};

/// A `[1, 1, S, 1]` KV tensor with each token's value being its f32 id —
/// the convention used across the cache module's existing tests so values
/// remain directly grep'pable in failure messages.
fn kv(ids: &[f32]) -> Array {
  Array::from_slice::<f32>(ids, &(1usize, 1, ids.len(), 1)).unwrap()
}

// =====================================================================
// #98 KVC-1 — transactional default from_serialized
// =====================================================================
//
// The trait-default `from_serialized` (used by no concrete cache — all 8
// override it) now snapshots state + meta BEFORE the 2-setter chain and
// rolls back on `set_meta_state` failure. Pre-KVC-1 the default was the
// verbatim Python sequence and left self half-restored on the meta error.

/// A test-only `KvCache` that:
/// (a) inherits the trait-default `from_serialized` (no override),
/// (b) records its `set_state` / `set_meta_state` call sequence so the
///     test can assert the snapshot + rollback ordering, and
/// (c) gates `set_meta_state` to fail on a sentinel meta string
///     `"FAIL_SENTINEL"` so the test can exercise the rollback arm
///     deterministically.
///
/// The recorded state is a single `u32` (the inner "set_state value")
/// updated only by `set_state`. The recorded meta is a single `String`
/// updated only by `set_meta_state`. Both default to empty / 0, so the
/// snapshot at the start of `from_serialized` captures `(empty, "")`.
#[derive(Default)]
struct RollbackProbeCache {
  /// Recorded "state-id": replaced by `set_state` with the first element
  /// of the incoming `state` Vec (treated as a `[1]`-shape `i32` array).
  recorded_state_id: i32,
  /// Recorded "meta-id": replaced by `set_meta_state` with the first
  /// element of the incoming meta slice.
  recorded_meta: String,
  /// `set_state` / `set_meta_state` call log — pinned for ordering.
  calls: Vec<&'static str>,
}

impl KvCache for RollbackProbeCache {
  fn offset(&self) -> usize {
    0
  }
  fn update(&mut self, _k: &Array, _v: &Array) -> Result<(Array, Array)> {
    unreachable!("RollbackProbeCache::update not used")
  }
  fn state(&self) -> Result<Vec<Array>> {
    // Reflect `recorded_state_id` as a single `[1]`-shape i32 array so
    // the snapshot round-trips through `set_state` exactly.
    Ok(vec![
      Array::from_slice::<i32>(&[self.recorded_state_id], &(1usize,)).unwrap(),
    ])
  }
  fn materialize(&mut self) -> Result<()> {
    Ok(())
  }
  fn set_state(&mut self, mut state: Vec<Array>) -> Result<()> {
    self.calls.push("set_state");
    if let Some(mut a) = state.pop() {
      let v = a.to_vec::<i32>().unwrap();
      self.recorded_state_id = *v.first().unwrap_or(&0);
    }
    // Sentinel that triggers the set_state failure path deterministically
    // — and crucially does so AFTER mutating `recorded_state_id` above, so
    // a missing rollback would leave the cache observably corrupt. The
    // trait does NOT require `set_state` to be atomic, so the default
    // `from_serialized` must restore the snapshot even on this arm.
    if self.recorded_state_id == -1 {
      return Err(Error::Backend {
        message: "RollbackProbeCache: forced set_state failure (post-mutation) for KVC-1 test"
          .into(),
      });
    }
    Ok(())
  }
  fn meta_state(&self) -> Vec<String> {
    vec![self.recorded_meta.clone()]
  }
  fn set_meta_state(&mut self, m: &[String]) -> Result<()> {
    self.calls.push("set_meta_state");
    // Sentinel that triggers the rollback path deterministically.
    if m.first().map(String::as_str) == Some("FAIL_SENTINEL") {
      return Err(Error::Backend {
        message: "RollbackProbeCache: forced failure for KVC-1 test".into(),
      });
    }
    self.recorded_meta = m.first().cloned().unwrap_or_default();
    Ok(())
  }
  fn make_mask(&self, _n: usize, _w: Option<usize>, _r: bool) -> Result<MaskMode> {
    unreachable!("RollbackProbeCache::make_mask not used")
  }
  fn nbytes(&self) -> usize {
    0
  }
  fn is_empty(&self) -> bool {
    self.recorded_state_id == 0
  }
  fn copy(&self) -> Result<Box<dyn KvCache>> {
    Ok(Box::new(RollbackProbeCache {
      recorded_state_id: self.recorded_state_id,
      recorded_meta: self.recorded_meta.clone(),
      calls: Vec::new(),
    }))
  }
  fn reference_class_name(&self) -> &'static str {
    "RollbackProbeCache"
  }
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

#[test]
fn kvc1_default_from_serialized_rolls_back_on_meta_failure() {
  // Prime the cache with a non-default state (id=7) + meta ("PRIMED").
  let mut cache = RollbackProbeCache::default();
  cache
    .set_state(vec![Array::from_slice::<i32>(&[7i32], &(1usize,)).unwrap()])
    .unwrap();
  cache.set_meta_state(&["PRIMED".to_string()]).unwrap();
  assert_eq!(cache.recorded_state_id, 7);
  assert_eq!(cache.recorded_meta, "PRIMED");
  // Reset the call log for clarity.
  cache.calls.clear();

  // Now call from_serialized with a NEW state and the sentinel meta — the
  // default path snapshots (id=7, "PRIMED"), applies set_state (id=99),
  // hits set_meta_state's failure on "FAIL_SENTINEL", and rolls back.
  let new_state = vec![Array::from_slice::<i32>(&[99i32], &(1usize,)).unwrap()];
  let bad_meta = vec!["FAIL_SENTINEL".to_string()];
  let result = cache.from_serialized(new_state, &bad_meta);
  assert!(
    result.is_err(),
    "expected Err on set_meta_state failure (sentinel)"
  );
  // The KEY post-fix assertion: the cache is rolled back to its pre-call
  // snapshot — NOT left half-restored with id=99 + still-"PRIMED" meta.
  assert_eq!(
    cache.recorded_state_id, 7,
    "rollback must restore state_id to the pre-call snapshot (KVC-1)"
  );
  assert_eq!(
    cache.recorded_meta, "PRIMED",
    "rollback must restore meta to the pre-call snapshot (KVC-1)"
  );
  // Call sequence pin: set_state (apply new) → set_meta_state (FAIL) →
  // set_state (rollback snapshot) → set_meta_state (rollback snapshot).
  assert_eq!(
    cache.calls,
    vec!["set_state", "set_meta_state", "set_state", "set_meta_state"],
    "rollback path must re-call set_state then set_meta_state with the snapshot"
  );
}

#[test]
fn kvc1_default_from_serialized_rolls_back_on_set_state_failure() {
  // R1 follow-up: the trait does NOT require `set_state` to be atomic.
  // An impl that mutates part of its state and THEN errors must still
  // leave `self` byte-identical to its pre-call state under the
  // transactional `from_serialized` contract. Pre-fix the default did
  // `self.set_state(state)?` (early-return on Err) and never restored
  // the snapshot on the `set_state` arm, leaving the cache corrupt.
  //
  // The probe `set_state` records the state_id BEFORE checking the
  // post-mutation sentinel (`-1`), so a `[-1]` input mutates
  // `recorded_state_id` to `-1` and THEN errors. With the fix in place
  // the rollback re-applies the snapshot (`set_state(snapshot_state)`
  // then `set_meta_state(snapshot_meta)`) and the cache is restored
  // exactly to its pre-call value.
  let mut cache = RollbackProbeCache::default();
  cache
    .set_state(vec![Array::from_slice::<i32>(&[7i32], &(1usize,)).unwrap()])
    .unwrap();
  cache.set_meta_state(&["PRIMED".to_string()]).unwrap();
  assert_eq!(cache.recorded_state_id, 7);
  assert_eq!(cache.recorded_meta, "PRIMED");
  cache.calls.clear();

  // Sentinel state `-1` triggers the post-mutation failure inside
  // `set_state` (the probe records `-1` then errors).
  let bad_state = vec![Array::from_slice::<i32>(&[-1i32], &(1usize,)).unwrap()];
  let any_meta = vec!["IGNORED".to_string()];
  let result = cache.from_serialized(bad_state, &any_meta);
  assert!(
    result.is_err(),
    "expected Err on set_state failure (post-mutation sentinel)"
  );
  // The KEY R1 post-fix assertion: the cache is rolled back to its
  // pre-call snapshot — NOT left mid-mutation at `recorded_state_id == -1`.
  assert_eq!(
    cache.recorded_state_id, 7,
    "rollback must restore state_id on set_state failure (KVC-1 R1 follow-up)"
  );
  assert_eq!(
    cache.recorded_meta, "PRIMED",
    "rollback must restore meta on set_state failure (KVC-1 R1 follow-up)"
  );
  // Call sequence pin: set_state (apply NEW = mutate then FAIL) →
  // set_state (rollback snapshot) → set_meta_state (rollback snapshot).
  // The forward `set_meta_state` is NEVER called because set_state errored.
  assert_eq!(
    cache.calls,
    vec!["set_state", "set_state", "set_meta_state"],
    "set_state failure path: forward set_state (FAIL) → rollback set_state \
     → rollback set_meta_state (NO forward set_meta_state)"
  );
}

#[test]
fn kvc1_default_from_serialized_success_path_does_not_invoke_rollback() {
  // Sanity: on the success path (no sentinel) the snapshot is captured
  // but never re-applied; only set_state + set_meta_state on the NEW
  // arguments run. The recorded state ends at the new values.
  let mut cache = RollbackProbeCache::default();
  cache
    .set_state(vec![Array::from_slice::<i32>(&[3i32], &(1usize,)).unwrap()])
    .unwrap();
  cache.set_meta_state(&["OLD".to_string()]).unwrap();
  cache.calls.clear();

  let new_state = vec![Array::from_slice::<i32>(&[42i32], &(1usize,)).unwrap()];
  let good_meta = vec!["NEW".to_string()];
  cache.from_serialized(new_state, &good_meta).unwrap();
  assert_eq!(cache.recorded_state_id, 42, "new state applied");
  assert_eq!(cache.recorded_meta, "NEW", "new meta applied");
  assert_eq!(
    cache.calls,
    vec!["set_state", "set_meta_state"],
    "success path runs set_state + set_meta_state once each (no rollback)"
  );
}

// =====================================================================
// #100 KVC-3 — typed enum dispatch (KvCacheKind)
// =====================================================================
//
// `from_state` now dispatches through `KvCacheKind::parse(...)` — the
// typed replacement for the pre-KVC-3 string-keyed `match kind { … }`
// (mirroring mlx-lm `globals()[class_name].from_state(...)`,
// cache.py:898). Parsing accepts every name + alias the prior match did;
// adding a new variant + arm is compile-checked exhaustively.

#[test]
fn kvc3_kvcachekind_parse_accepts_every_alias() {
  // All names the pre-KVC-3 `match kind { … }` accepted MUST still parse
  // — the typed dispatch is back-compat with every alias.
  for (name, expected) in &[
    ("KVCache", KvCacheKind::KvCache),
    ("ConcatenateKVCache", KvCacheKind::KvCache),
    ("KVCacheSimple", KvCacheKind::KvCache),
    ("StandardKvCache", KvCacheKind::KvCache),
    ("RotatingKVCache", KvCacheKind::RotatingKvCache),
    ("RotatingKvCache", KvCacheKind::RotatingKvCache),
    ("ChunkedKVCache", KvCacheKind::ChunkedKvCache),
    ("ChunkedKvCache", KvCacheKind::ChunkedKvCache),
    ("QuantizedKVCache", KvCacheKind::QuantizedKvCache),
    ("QuantizedKvCacheImpl", KvCacheKind::QuantizedKvCache),
    ("CacheList", KvCacheKind::CacheList),
    ("BatchKVCache", KvCacheKind::BatchKvCache),
    ("BatchKvCache", KvCacheKind::BatchKvCache),
    ("BatchRotatingKVCache", KvCacheKind::BatchRotatingKvCache),
    ("BatchRotatingKvCache", KvCacheKind::BatchRotatingKvCache),
    ("ArraysCache", KvCacheKind::ArraysCache),
    ("MambaCache", KvCacheKind::MambaCache),
  ] {
    assert_eq!(
      &KvCacheKind::parse(name).unwrap(),
      expected,
      "KvCacheKind::parse({name:?}) must round-trip"
    );
  }
}

#[test]
fn kvc3_kvcachekind_parse_unknown_is_recoverable_err() {
  // Unknown kinds are a recoverable Error::Backend (replaces the prior
  // string-keyed `other => Err(...)` arm). NEVER a panic.
  let result = KvCacheKind::parse("DefinitelyNotARealCacheKind");
  match result {
    Err(Error::Backend { message }) => {
      assert!(
        message.contains("unknown cache kind"),
        "diagnostic must name the failure mode; got {message:?}"
      );
      assert!(
        message.contains("DefinitelyNotARealCacheKind"),
        "diagnostic must include the offending kind string; got {message:?}"
      );
    }
    other => panic!("expected Err(Error::Backend), got {other:?}"),
  }
}

#[test]
fn kvc3_from_state_dispatches_through_typed_enum() {
  // Round-trip a Standard cache through `from_state` (via the
  // typed-dispatch entry point) and verify the rebuilt cache reports the
  // canonical `"KVCache"` reference class name — exactly the path
  // `globals()[class_name].from_state(...)` mirrored.
  let mut original = StandardKvCache::new();
  original.update(&kv(&[0.0, 1.0]), &kv(&[0.0, 1.0])).unwrap();
  let state = original.state().unwrap();
  let meta = original.meta_state();
  let rebuilt = from_state("KVCache", state, &meta).unwrap();
  assert_eq!(
    rebuilt.reference_class_name(),
    "KVCache",
    "typed-dispatch round-trip preserves the reference class name"
  );
  // Sanity: the offset survived too (i.e. the dispatch routed to the
  // correct arm, not some other one with matching reference class name).
  assert_eq!(rebuilt.offset(), 2);
}

// =====================================================================
// #105 KVC-8 — eager K/V cross-validation in QuantizedKvCacheImpl::set_state
// =====================================================================
//
// Pre-KVC-8: a 4 or 6-array state with K and V having different leading
// shapes (e.g. mismatched batch size or seq_len) was accepted and only
// errored at the first `update_quantized` `concat_seq` — lazy + far from
// the cause. Post-KVC-8: a precise diagnostic at the load boundary
// (`set_state`), so a corrupt prompt cache is surfaced eagerly.

#[test]
fn kvc8_quantized_set_state_4_arr_rejects_kv_shape_mismatch() {
  // Construct a 4-array state where K and V have DIFFERENT seq_len
  // (axis 2) — a forged prompt cache. Pre-KVC-8 this assigned silently;
  // post-KVC-8 it MUST surface a precise diagnostic at set_state.
  let k_w = Array::from_slice::<f32>(&[0.0; 8], &(1usize, 1, 4, 2)).unwrap();
  let v_w = Array::from_slice::<f32>(&[0.0; 6], &(1usize, 1, 3, 2)).unwrap();
  let k_s = Array::from_slice::<f32>(&[0.0; 4], &(1usize, 1, 4, 1)).unwrap();
  let v_s = Array::from_slice::<f32>(&[0.0; 3], &(1usize, 1, 3, 1)).unwrap();
  let state = vec![k_w, k_s, v_w, v_s];

  let mut cache = QuantizedKvCacheImpl::new(64, 4);
  let result = cache.set_state(state);
  match result {
    Err(Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("set_state"),
        "diagnostic must name the load boundary; got {message:?}"
      );
      assert!(
        message.contains("axis 2") || message.contains("axis 1") || message.contains("axis"),
        "diagnostic must name the mismatched axis; got {message:?}"
      );
    }
    other => panic!("expected Err(Error::ShapeMismatch) for K/V seq_len mismatch, got {other:?}"),
  }
  // Cache MUST be untouched (kept empty): the validation runs BEFORE
  // the field assignment.
  assert!(
    cache.is_empty(),
    "set_state must leave the cache untouched on Err"
  );
  assert_eq!(cache.offset(), 0);
}

#[test]
fn kvc8_quantized_set_state_6_arr_rejects_kv_bias_shape_mismatch() {
  // 6-array state where K_bias and V_bias have different leading shapes
  // — exactly the K-6-affine / V-4-bias-less kind of corruption the
  // issue named, surfaced here as a K_bias shape that doesn't match
  // V_bias (a 6-array forged where one bias is empty / wrong).
  let shape_ok = (1usize, 1, 2, 2);
  let k_w = Array::from_slice::<f32>(&[0.0; 4], &shape_ok).unwrap();
  let v_w = Array::from_slice::<f32>(&[0.0; 4], &shape_ok).unwrap();
  let k_s = Array::from_slice::<f32>(&[0.0; 2], &(1usize, 1, 2, 1)).unwrap();
  let v_s = Array::from_slice::<f32>(&[0.0; 2], &(1usize, 1, 2, 1)).unwrap();
  // Mismatched biases: k_b has seq_len 2, v_b has seq_len 3.
  let k_b = Array::from_slice::<f32>(&[0.0; 2], &(1usize, 1, 2, 1)).unwrap();
  let v_b = Array::from_slice::<f32>(&[0.0; 3], &(1usize, 1, 3, 1)).unwrap();
  let state = vec![k_w, k_s, k_b, v_w, v_s, v_b];

  let mut cache = QuantizedKvCacheImpl::new(64, 4);
  let result = cache.set_state(state);
  match result {
    Err(Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("biases"),
        "diagnostic must name the offending element (biases); got {message:?}"
      );
    }
    other => {
      panic!("expected Err(Error::ShapeMismatch) for K/V bias shape mismatch, got {other:?}")
    }
  }
  assert!(
    cache.is_empty(),
    "set_state must leave the cache untouched on Err"
  );
}

#[test]
fn kvc8_quantized_set_state_consistent_shapes_pass() {
  // Sanity: a consistent 4-array state (matching K/V shapes) passes the
  // eager validator unchanged — the validator only rejects corruption,
  // not faithful round-trip payloads.
  let shape = (1usize, 1, 4, 2);
  let k_w = Array::from_slice::<f32>(&[0.0; 8], &shape).unwrap();
  let v_w = Array::from_slice::<f32>(&[0.0; 8], &shape).unwrap();
  let k_s = Array::from_slice::<f32>(&[0.0; 4], &(1usize, 1, 4, 1)).unwrap();
  let v_s = Array::from_slice::<f32>(&[0.0; 4], &(1usize, 1, 4, 1)).unwrap();
  let state = vec![k_w, k_s, v_w, v_s];

  let mut cache = QuantizedKvCacheImpl::new(64, 4);
  cache
    .set_state(state)
    .expect("consistent 4-array state must pass");
  assert!(
    !cache.is_empty(),
    "consistent state must populate the cache"
  );
}

// ---------------------------------------------------------------------
// R1 follow-up: leading-axes-only validation (B, H, S); the last
// (payload) axis can legitimately differ when `v_head_dim != k_head_dim`.
// Pre-R1 `validate_kv_leading_axes_match` walked every axis including
// the payload axis, so a valid skewed cache failed its own saved state.
// ---------------------------------------------------------------------

#[test]
fn kvc8_quantized_set_state_accepts_different_v_head_dim() {
  // K shape `[1, 2, 4, 64]` and V shape `[1, 2, 4, 128]` — leading axes
  // (B=1, H=2, S=4) match; the last axis differs because v_head_dim
  // (128) != k_head_dim (64). `update_quantized` reads `keys.shape[-1]`
  // and `values.shape[-1]` independently (cache.py:243-244), so this
  // shape is a faithful round-trip — the eager validator must NOT
  // reject it.
  // Lengths (B=1, H=2, S=4) factored to constants to satisfy
  // clippy::identity_op (no `* 1`) and stay readable.
  let k_w = Array::from_slice::<f32>(&[0.0; 8 * 64], &(1usize, 2, 4, 64)).unwrap();
  let v_w = Array::from_slice::<f32>(&[0.0; 8 * 128], &(1usize, 2, 4, 128)).unwrap();
  let k_s = Array::from_slice::<f32>(&[0.0; 8], &(1usize, 2, 4, 1)).unwrap();
  let v_s = Array::from_slice::<f32>(&[0.0; 8 * 2], &(1usize, 2, 4, 2)).unwrap();
  let k_b = Array::from_slice::<f32>(&[0.0; 8], &(1usize, 2, 4, 1)).unwrap();
  let v_b = Array::from_slice::<f32>(&[0.0; 8 * 2], &(1usize, 2, 4, 2)).unwrap();
  let state = vec![k_w, k_s, k_b, v_w, v_s, v_b];

  let mut cache = QuantizedKvCacheImpl::new(64, 4);
  cache
    .set_state(state)
    .expect("leading axes (B, H, S) match; differing payload axis is the v_head_dim skew the cache contract allows");
  assert!(
    !cache.is_empty(),
    "valid skewed-head_dim state must populate the cache"
  );
}

#[test]
fn kvc8_quantized_set_state_rejects_non_4d_rank() {
  // A non-4-D K or V is a forged / corrupt state. The rank-only gate
  // rejects at the load boundary with a precise diagnostic instead of
  // panicking downstream via a blind `shape[axis]` index.
  let k_w = Array::from_slice::<f32>(&[0.0; 6], &(2usize, 3)).unwrap(); // rank 2 — invalid
  let v_w = Array::from_slice::<f32>(&[0.0; 16], &(1usize, 2, 4, 2)).unwrap();
  let k_s = Array::from_slice::<f32>(&[0.0; 6], &(2usize, 3)).unwrap();
  let v_s = Array::from_slice::<f32>(&[0.0; 8], &(1usize, 2, 4, 1)).unwrap();
  let state = vec![k_w, k_s, v_w, v_s];

  let mut cache = QuantizedKvCacheImpl::new(64, 4);
  let result = cache.set_state(state);
  match result {
    Err(Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("4-D") || message.contains("rank"),
        "rank-gate diagnostic must name the 4-D requirement; got {message:?}"
      );
    }
    other => panic!("expected Err(Error::ShapeMismatch) for non-4-D state, got {other:?}"),
  }
  assert!(
    cache.is_empty(),
    "set_state must leave the cache untouched on rank-gate Err"
  );
}

#[test]
fn kvc8_quantized_update_save_reload_with_v_head_dim_skew() {
  // End-to-end round-trip: `update_quantized` with skewed dims
  // (k_head_dim=64, v_head_dim=128), then `state()` + `meta_state()`,
  // then `from_state("QuantizedKVCache", …)` rebuilds a cache.
  // Pre-R1 fix: the rebuilt `from_state` path called `set_state` which
  // hit the over-eager every-axis validator and ERRORED on the cache's
  // own saved state ("K w axis 3 (=8 = k_head_dim/el_per_int)
  // != V w axis 3 (=16 = v_head_dim/el_per_int)"). Post-R1 fix: the
  // leading-axes-only validator accepts the skew, and the round-trip
  // succeeds.
  let group_size: i32 = 64;
  let bits: i32 = 8;
  let k_head_dim: usize = 64;
  let v_head_dim: usize = 128;
  let b: usize = 1;
  let h: usize = 2;
  let s: usize = 4;

  // K: [B=1, H=2, S=4, k_head_dim=64]; V: [B=1, H=2, S=4, v_head_dim=128].
  let k = {
    let n = b * h * s * k_head_dim;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    Array::from_slice::<f32>(&data, &(b, h, s, k_head_dim)).unwrap()
  };
  let v = {
    let n = b * h * s * v_head_dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    Array::from_slice::<f32>(&data, &(b, h, s, v_head_dim)).unwrap()
  };

  let mut original = QuantizedKvCacheImpl::new(group_size, bits);
  original
    .update_quantized(&k, &v)
    .expect("update_quantized with v_head_dim != k_head_dim must succeed");
  assert_eq!(original.offset(), s);

  // Save: `state()` produces the 6-array K/V triple list; `meta_state()`
  // produces the 3-string `[offset, group_size, bits]` tuple.
  let state = original.state().expect("state() on a populated cache");
  let meta = original.meta_state();
  // Sanity: confirm the saved state's K and V w-arrays really do have
  // different payload (last) axes — this is the very condition the
  // over-eager validator rejected. K_w[-1] = k_head_dim / el_per_int,
  // V_w[-1] = v_head_dim / el_per_int (el_per_int = 32 // bits = 4).
  let el_per_int = 32 / bits as usize;
  assert_eq!(state[0].shape(), vec![b, h, s, k_head_dim / el_per_int]);
  assert_eq!(state[3].shape(), vec![b, h, s, v_head_dim / el_per_int]);

  // Reload via the project-canonical typed-dispatch entry point — the
  // exact path that pre-R1 errored on its own saved state.
  let rebuilt = from_state("QuantizedKVCache", state, &meta)
    .expect("from_state round-trip with v_head_dim skew must succeed (R1 fix)");
  assert_eq!(
    rebuilt.reference_class_name(),
    "QuantizedKVCache",
    "typed-dispatch routed to the QuantizedKVCache arm"
  );
  assert_eq!(rebuilt.offset(), s, "offset survived the save/reload");
  assert!(!rebuilt.is_empty(), "rebuilt cache is populated");
}

// =====================================================================
// #106 KVC-9 — MambaCache provenance preserved via ArraysCache::is_mamba
// =====================================================================

#[test]
fn kvc9_mamba_constructor_reports_mambacache_class_name() {
  // Constructor-time provenance: `ArraysCache::mamba()` carries the
  // `MambaCache` class label across `reference_class_name` (state shape
  // identical to `ArraysCache::new(2)`).
  let mamba = ArraysCache::mamba();
  assert!(mamba.is_mamba(), "mamba() ctor sets the is_mamba flag");
  assert_eq!(
    mamba.reference_class_name(),
    "MambaCache",
    "an ArraysCache::mamba() reports MambaCache (not ArraysCache)"
  );
  // The corresponding ArraysCache::new(2) reports ArraysCache — same
  // state shape, different provenance label.
  let plain = ArraysCache::new(2);
  assert!(!plain.is_mamba());
  assert_eq!(plain.reference_class_name(), "ArraysCache");
}

#[test]
fn kvc9_mamba_round_trip_preserves_class_label() {
  // The KEY post-fix assertion: a swift-saved "MambaCache"-kind prompt
  // cache reloads via `from_state("MambaCache", …)` and reports
  // `MambaCache` again on `reference_class_name` — pre-KVC-9 it would
  // have degraded to `ArraysCache` on save-after-load.
  let mut original = ArraysCache::mamba();
  // Drive in one slot of state (a minimal real-looking SSM slot).
  let slot0 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1usize, 3)).unwrap();
  original.set(0, slot0).unwrap();
  let state = original.state().unwrap();
  let meta = original.meta_state();
  let rebuilt = from_state("MambaCache", state, &meta).unwrap();
  assert_eq!(
    rebuilt.reference_class_name(),
    "MambaCache",
    "from_state(\"MambaCache\", …) MUST preserve the MambaCache class label"
  );
}

#[test]
fn kvc9_from_state_arrayscache_arm_does_not_set_mamba_flag() {
  // The complementary case: `from_state("ArraysCache", …)` reconstructs
  // WITHOUT the mamba flag — the two arms are distinct (KvCacheKind has
  // separate ArraysCache and MambaCache variants).
  let mut original = ArraysCache::new(2);
  original
    .set(0, Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap())
    .unwrap();
  let state = original.state().unwrap();
  let meta = original.meta_state();
  let rebuilt = from_state("ArraysCache", state, &meta).unwrap();
  assert_eq!(
    rebuilt.reference_class_name(),
    "ArraysCache",
    "from_state(\"ArraysCache\", …) reconstructs WITHOUT the mamba flag"
  );
}

#[test]
fn kvc9_mamba_copy_preserves_provenance() {
  // `copy()` must preserve the `is_mamba` flag — otherwise a deep-copy
  // of a `MambaCache` would silently downgrade to `ArraysCache` on the
  // copy.
  let mamba = ArraysCache::mamba();
  let copied = mamba.copy().unwrap();
  assert_eq!(
    copied.reference_class_name(),
    "MambaCache",
    "ArraysCache::copy() MUST preserve the is_mamba flag"
  );
}

// =====================================================================
// #107 KVC-10 — required `reference_class_name` (compile-time enforcement)
// =====================================================================
//
// Compile-time enforcement can only be tested at COMPILE TIME (a `KvCache`
// impl that omits `reference_class_name` is a compile error). The runtime
// test below verifies the contract: every in-tree concrete cache declares
// a non-`"KVCache"` name (other than `StandardKvCache` which legitimately
// IS `"KVCache"`). The structural compile-time guarantee is itself a
// "test" in the form of the codebase compiling: removing
// `reference_class_name` from any impl causes `cargo build` to fail with
// `E0046: not all trait items implemented`.

#[test]
fn kvc10_every_concrete_cache_declares_its_reference_class_name() {
  // The required-method contract: every concrete cache returns its
  // declared name (no silent inheritance of a `"KVCache"` default).
  // This pins each in-tree cache's name; removing the trait method
  // override would have inherited `"KVCache"` pre-KVC-10 but is now a
  // compile error.
  let pairs: Vec<(&str, Box<dyn KvCache>)> = vec![
    ("KVCache", Box::new(StandardKvCache::new())),
    ("RotatingKVCache", Box::new(RotatingKvCache::new(8, 1))),
    (
      "QuantizedKVCache",
      Box::new(QuantizedKvCacheImpl::new(64, 4)),
    ),
    ("ArraysCache", Box::new(ArraysCache::new(2))),
    ("MambaCache", Box::new(ArraysCache::mamba())),
    ("CacheList", Box::new(CacheList::new(vec![]))),
  ];
  for (expected, cache) in &pairs {
    assert_eq!(
      cache.reference_class_name(),
      *expected,
      "concrete cache MUST declare its `reference_class_name` explicitly (KVC-10)"
    );
  }
}
