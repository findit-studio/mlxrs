//! Unit tests for [`StandardQuantizedKvCache`] internals that the public-API
//! integration suite (`tests/lm_cache_quantized.rs`) cannot reach, or only
//! exercises on the happy path:
//!
//!  * the infallible [`Default`] (`new_unchecked(64, 8)`);
//!  * the `set_state` K/V rank-mismatch context arms of
//!    [`validate_kv_leading_axes_match`] — every per-element branch
//!    (`"w"` / `"scales"` / `"biases"`) for BOTH the K-side and V-side
//!    rank gate (a non-4-D triple component is a forged/corrupt prompt
//!    cache, reachable only by feeding `set_state` a wrong-rank array);
//!  * the `compute_appended` `offset + num_steps` overflow closure
//!    (`offset == usize::MAX` injected via a struct literal — a faithful
//!    trace can never reach it);
//!  * the `set_meta_state` `group_size` parse-error closure;
//!  * `trim`'s 0-token early return and its `None`-storage arms;
//!  * `copy`'s `None`-storage arms;
//!  * `triple_component_len_range`'s `None`-bias arm (called directly,
//!    in-module, on a bias-less triple — `seq_len` is metadata-only);
//!  * `tree_map`'s `None`-bias arm via a `copy` of a bias-less cache
//!    (`try_clone` is a metadata refcount clone, no compute);
//!  * `concat_triple`'s mismatched-bias `InvariantViolation` arm;
//!  * `enforce_offset_len_invariant`'s asymmetric `(Some, None)` /
//!    `(None, Some)` storage arms;
//!  * `materialize`'s `Some`/`None` eval arms.
//!
//! Oracle discipline: every retained-value/shape assertion is checked
//! against a hand-traced expectation derived from the input fixtures, never
//! against the function under test. Validation branches are matched by typed
//! `Error` variant + payload accessors.
//!
//! Build note: the branches that short-circuit before any MLX op (the
//! rank-gate context arms, the `offset` overflow, the `group_size` parse,
//! the 0-token / `None`-storage `trim`/`copy`/`tree_map` arms,
//! `triple_component_len_range`) inspect only shapes/`Option`s and host
//! scalars — `Array::shape`/`ndim` are metadata reads, `try_clone` is a
//! refcount clone — so they need no Metal device. The remaining tests
//! (`materialize`'s `eval`, `enforce_offset_len_invariant`'s `slice_seq`,
//! `concat_triple`'s `concat_seq`) drive real MLX compute and are validated
//! on the Metal CI host.
use super::*;

/// A `[1, 1, S, dim]` 4-D KV tensor (the canonical `KV_NDIM == 4` layout)
/// with a per-element-distinct ramp, so each marker reads straight out of a
/// row-major host read. `S == n_steps`.
fn kv4(n_steps: usize, dim: usize, base: f32) -> Array {
  let total = n_steps * dim;
  let data: Vec<f32> = (0..total).map(|i| base + i as f32).collect();
  Array::from_slice::<f32>(&data, &(1usize, 1, n_steps, dim)).unwrap()
}

/// A `[1, 1, S, 1]` 4-D KV tensor whose single trailing element per row is
/// the given marker — the minimal valid quantized-triple component shape for
/// the shape/`Option`-only branches under test.
fn kv(vals: &[f32]) -> Array {
  Array::from_slice::<f32>(vals, &(1usize, 1, vals.len(), 1)).unwrap()
}

/// A non-4-D array of the given rank/shape, for the `set_state` rank gate.
/// `shape` is taken by value as a `Vec<usize>` (which implements
/// [`crate::shape::IntoShape`]) so the call site is an unambiguous single
/// reference.
#[allow(clippy::needless_pass_by_value)]
fn ranked(shape: Vec<usize>) -> Array {
  let total: usize = shape.iter().product();
  let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
  Array::from_slice::<f32>(&data, &shape).unwrap()
}

// ── Default::default — new_unchecked(64, 8) (lines 97, 100) ──────────────

/// [`Default`] yields the mlx-lm `QuantizedKVCache(group_size=64, bits=8)`
/// defaults via the infallible `new_unchecked` (lines 97-100): empty, offset
/// 0, with the canonical `group_size`/`bits` observable through the
/// [`QuantizedKvCache`] getters + `meta_state`.
#[test]
fn default_is_empty_group64_bits8() {
  let c = StandardQuantizedKvCache::default();
  assert!(c.is_empty(), "fresh default cache holds no keys");
  assert_eq!(c.offset(), 0);
  assert_eq!(c.group_size(), 64, "mlx-lm default group_size");
  assert_eq!(c.bits(), 8, "mlx-lm default bits");
  assert_eq!(
    c.meta_state(),
    vec!["0".to_string(), "64".to_string(), "8".to_string()],
    "meta_state serializes the default offset/group_size/bits"
  );
  // No quantized state until an update.
  assert!(c.quantized_state().unwrap().is_none());
}

// ── validate_kv_leading_axes_match: K-side rank gate (lines 284-297) ─────
//    incl. the per-element context arms at 287-290.

/// `set_state` with a non-4-D **K** `w` component (rank 2) trips the K-side
/// rank gate; the `"w"` context arm (line 286) names the offending component
/// and the `RankMismatch` payload carries the observed rank + full shape.
/// `self` is untouched (the validator runs before any assignment).
#[test]
fn set_state_k_w_wrong_rank_uses_w_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let bad_kw = ranked(vec![1, 4]); // rank 2, not 4
  let ok = kv(&[1.0]); // valid 4-D for the remaining slots
  // 6-array (with-biases) form so each named element gets a w/scales/biases
  // slot: order is [k_w, k_scales, k_biases, v_w, v_scales, v_biases].
  let st = vec![
    bad_kw,
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
  ];
  let err = c.set_state(st).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("K w must be 4-D"),
        "must select the K-side `w` context arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 2, "observed rank is 2");
      assert_eq!(p.actual_shape(), &[1, 4], "full observed shape carried");
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(
    c.is_empty(),
    "set_state must not mutate on the rank-gate Err"
  );
  assert_eq!(c.offset(), 0);
}

/// `set_state` with a non-4-D **K** `scales` component selects the
/// `"scales"` K-side context arm (lines 287-289). The `w` slots are valid
/// 4-D so the validator reaches the scales element before failing.
#[test]
fn set_state_k_scales_wrong_rank_uses_scales_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let ok = kv(&[1.0]);
  let bad_ks = ranked(vec![1, 1, 3]); // rank 3
  // [k_w, k_scales(bad), k_biases, v_w, v_scales, v_biases]
  let st = vec![
    ok.try_clone().unwrap(),
    bad_ks,
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
  ];
  match c.set_state(st).unwrap_err() {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("K scales must be 4-D"),
        "must select the K-side `scales` arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 3);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(c.is_empty());
}

/// `set_state` with a non-4-D **K** `biases` component selects the generic
/// `_ =>` K-side context arm (line 290) — biases is validated only in the
/// 6-array path and falls through the `match element` to the `_` default.
#[test]
fn set_state_k_biases_wrong_rank_uses_default_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let ok = kv(&[1.0]);
  let bad_kb = ranked(vec![5]); // rank 1
  // [k_w, k_scales, k_biases(bad), v_w, v_scales, v_biases]
  let st = vec![
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    bad_kb,
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
  ];
  match c.set_state(st).unwrap_err() {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("K must be 4-D")
          && !p.context().contains("w must")
          && !p.context().contains("scales must"),
        "biases must select the generic K-side `_` arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 1);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(c.is_empty());
}

// ── validate_kv_leading_axes_match: V-side rank gate (lines 298-310) ─────
//    incl. the per-element context arms at 299-309. K is valid 4-D so the
//    gate falls through to the V-side check.

/// `set_state` with a valid 4-D K `w` but a non-4-D **V** `w` trips the
/// V-side rank gate's `"w"` arm (line 300).
#[test]
fn set_state_v_w_wrong_rank_uses_v_w_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let ok = kv(&[1.0]);
  let bad_vw = ranked(vec![2, 2]); // rank 2
  // 4-array (bias-less) form: [k_w, k_scales, v_w(bad), v_scales].
  let st = vec![ok.try_clone().unwrap(), ok.try_clone().unwrap(), bad_vw, ok];
  match c.set_state(st).unwrap_err() {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("V w must be 4-D"),
        "must select the V-side `w` arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 2);
      assert_eq!(p.actual_shape(), &[2, 2]);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(c.is_empty());
}

/// `set_state` with a non-4-D **V** `scales` selects the V-side `"scales"`
/// arm (lines 301-303).
#[test]
fn set_state_v_scales_wrong_rank_uses_v_scales_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let ok = kv(&[1.0]);
  let bad_vs = ranked(vec![1, 1, 1, 1, 1]); // rank 5
  // 4-array form: [k_w, k_scales, v_w, v_scales(bad)].
  let st = vec![
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    bad_vs,
  ];
  match c.set_state(st).unwrap_err() {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("V scales must be 4-D"),
        "must select the V-side `scales` arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 5);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(c.is_empty());
}

/// `set_state` with a non-4-D **V** `biases` selects the generic V-side
/// `_ =>` arm (line 304). K and the leading V components are valid 4-D so
/// the validator reaches the V `biases` element.
#[test]
fn set_state_v_biases_wrong_rank_uses_v_default_context_arm() {
  let mut c = StandardQuantizedKvCache::default();
  let ok = kv(&[1.0]);
  let bad_vb = ranked(vec![7]); // rank 1
  // 6-array form: [k_w, k_scales, k_biases, v_w, v_scales, v_biases(bad)].
  let st = vec![
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    ok.try_clone().unwrap(),
    bad_vb,
  ];
  match c.set_state(st).unwrap_err() {
    Error::RankMismatch(p) => {
      assert!(
        p.context().contains("V must be 4-D")
          && !p.context().contains("w must")
          && !p.context().contains("scales must"),
        "V biases must select the generic V-side `_` arm, got: {}",
        p.context()
      );
      assert_eq!(p.actual(), 1);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  assert!(c.is_empty());
}

// ── compute_appended: offset + num_steps overflow (lines 404-410) ────────

/// A hostile restored `offset == usize::MAX` makes `offset + num_steps`
/// overflow; the `checked_add` closure (lines 405-409) surfaces a
/// recoverable `ArithmeticOverflow` carrying both operands, with NO partial
/// mutation. The check precedes `mx.quantize`, so a valid 4-D input reaches
/// the overflow purely via `seq_len` (a metadata read) — no Metal device
/// needed. Injected via struct literal (a faithful trace never reaches
/// `usize::MAX`).
#[test]
fn update_quantized_offset_overflow_is_rejected() {
  let mut c = StandardQuantizedKvCache {
    keys: None,
    values: None,
    offset: usize::MAX,
    group_size: 64,
    bits: 8,
  };
  let t = kv(&[1.0, 2.0]); // num_steps = 2 -> usize::MAX + 2 overflows
  let err = c.update_quantized(&t, &t).unwrap_err();
  match err {
    Error::ArithmeticOverflow(p) => {
      assert!(
        p.context().contains("offset + num_steps"),
        "context must name the offset + num_steps add, got: {}",
        p.context()
      );
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "offset" && *v == usize::MAX as u64),
        "operands must carry offset=usize::MAX, got: {:?}",
        p.operands()
      );
      assert!(
        p.operands()
          .iter()
          .any(|(n, v)| *n == "num_steps" && *v == 2),
        "operands must carry num_steps=2, got: {:?}",
        p.operands()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
  assert_eq!(c.offset(), usize::MAX, "offset unchanged on the Err path");
  assert!(c.is_empty(), "buffer not committed on the Err path");

  // The base `KvCache::update` shares `compute_appended`, so it overflows
  // identically (lines 404-410 again, via `update`).
  let mut c2 = StandardQuantizedKvCache {
    keys: None,
    values: None,
    offset: usize::MAX,
    group_size: 64,
    bits: 8,
  };
  assert!(matches!(
    c2.update(&t, &t),
    Err(Error::ArithmeticOverflow(_))
  ));
  assert_eq!(c2.offset(), usize::MAX);
}

// ── set_meta_state: group_size parse-error closure (lines 866-874) ───────

/// A non-numeric `group_size` token makes the `parse::<i32>()` fail; the
/// closure (lines 869-873) wraps it as a recoverable `Error::Parse` naming
/// `group_size` with `input_kind == "i32"`, leaving the cache unmutated (the
/// offset parsed before it succeeds, but the commit tail runs only after ALL
/// three parses). Exercised in BOTH the 3-string (mlx-lm) and 4-string
/// (mlx-swift-lm) forms — same parser, distinct index mapping.
#[test]
fn set_meta_state_group_size_parse_error_leaves_cache_unmutated() {
  // 3-string mlx-lm form: indices [offset, group_size, bits].
  let mut c = StandardQuantizedKvCache::new(64, 8).unwrap();
  let err = c
    .set_meta_state(&["3".to_string(), "not_a_number".to_string(), "8".to_string()])
    .unwrap_err();
  match err {
    Error::Parse(p) => {
      assert!(
        p.context().contains("group_size"),
        "context must name group_size, got: {}",
        p.context()
      );
      assert_eq!(p.input_kind(), "i32");
    }
    other => panic!("expected Parse, got {other:?}"),
  }
  // Unmutated: a valid `offset` was parsed into a local but never committed,
  // because the `group_size` parse failed before the infallible tail.
  assert_eq!(
    c.meta_state(),
    vec!["0".to_string(), "64".to_string(), "8".to_string()],
    "no field committed on the parse Err"
  );

  // 4-string mlx-swift-lm form: indices [step, offset, groupSize, bits];
  // `group_size` is at index 2.
  let mut c2 = StandardQuantizedKvCache::new(64, 8).unwrap();
  match c2
    .set_meta_state(&[
      "256".to_string(),
      "5".to_string(),
      "bad".to_string(),
      "4".to_string(),
    ])
    .unwrap_err()
  {
    Error::Parse(p) => {
      assert!(p.context().contains("group_size"), "got: {}", p.context());
      assert_eq!(p.input_kind(), "i32");
    }
    other => panic!("expected Parse, got {other:?}"),
  }
  assert_eq!(c2.offset(), 0, "offset not committed on the Err path");
  assert_eq!(
    c2.group_size(),
    64,
    "group_size not committed on the Err path"
  );
}

// ── trim: the 0-token early return (line 925) ────────────────────────────

/// `trim(0)` on a populated-by-literal cache returns `Ok(0)` via the
/// `trimmed == 0` early return (lines 921-925) — a no-op that slices
/// nothing and leaves `offset` + storage untouched (no compute path taken).
/// Also covers `trim(n)` on an EMPTY cache (`offset == 0` -> `trimmed == 0`).
#[test]
fn trim_zero_token_is_noop_early_return() {
  // Populated via struct literal so the no-op path is observable without an
  // update (the stored arrays are never sliced — `trimmed == 0` returns
  // before `trim_triple`).
  let kw = kv(&[10.0, 11.0, 12.0]);
  let ks = kv(&[1.0, 1.0, 1.0]);
  let vw = kv(&[100.0, 101.0, 102.0]);
  let vs = kv(&[2.0, 2.0, 2.0]);
  let mut c = StandardQuantizedKvCache {
    keys: Some((kw, ks, None)),
    values: Some((vw, vs, None)),
    offset: 3,
    group_size: 64,
    bits: 8,
  };
  assert_eq!(c.trim(0).unwrap(), 0, "0-token trim returns 0");
  assert_eq!(c.offset(), 3, "offset unchanged");
  // Storage shapes intact (never sliced).
  let st = c.state().unwrap();
  assert_eq!(st[0].shape(), vec![1, 1, 3, 1], "K w untrimmed");

  // Empty cache: offset 0 -> min(n, 0) == 0 -> same early return.
  let mut empty = StandardQuantizedKvCache::new(64, 8).unwrap();
  assert_eq!(empty.trim(5).unwrap(), 0, "empty cache trims nothing");
  assert_eq!(empty.offset(), 0);
}

// ── trim: the None keys/values arms (lines 942, 946) ─────────────────────

/// `trim(n)` with `n > 0` but `keys`/`values == None` (a hostile
/// `offset > 0` with empty storage, struct-literal injected) takes BOTH
/// `None` arms (lines 942 + 946): no slice is attempted, `offset` is
/// decremented, and the cache stays empty. No compute (the `Some` arm's
/// `trim_triple` is never called).
#[test]
fn trim_with_none_storage_takes_none_arms() {
  let mut c = StandardQuantizedKvCache {
    keys: None,
    values: None,
    offset: 4, // > 0 so trimmed = min(2, 4) = 2 is non-zero
    group_size: 64,
    bits: 8,
  };
  let trimmed = c.trim(2).unwrap();
  assert_eq!(trimmed, 2, "min(2, 4) trimmed");
  assert_eq!(c.offset(), 2, "offset decremented by the trimmed count");
  assert!(c.is_empty(), "storage stays None (None arms taken)");
}

// ── copy: the None keys/values arms (lines 1013, 1017) ───────────────────

/// `copy()` of an EMPTY cache exercises the `None` clone arms for both
/// `keys` (line 1013) and `values` (line 1017): the copy is an independent,
/// equal, still-empty cache carrying the same scalar `offset`/`group_size`/
/// `bits`. No `try_clone` is attempted (no compute).
#[test]
fn copy_empty_cache_takes_none_arms() {
  let c = StandardQuantizedKvCache::new(32, 4).unwrap();
  let cp = c.copy().unwrap();
  assert!(cp.is_empty(), "copied empty cache is still empty");
  assert_eq!(cp.offset(), 0);
  assert_eq!(cp.reference_class_name(), "QuantizedKVCache");
  // Scalars carried through the `None` arms.
  let q = cp.as_quantized().expect("copy is still a quantized cache");
  assert_eq!(q.group_size(), 32, "group_size copied");
  assert_eq!(q.bits(), 4, "bits copied");
  assert!(q.quantized_state().unwrap().is_none(), "no triples to copy");
}

// ── triple_component_len_range: the None-bias arm (line 237) ─────────────

/// [`triple_component_len_range`] on a bias-less triple (`t.2 == None`)
/// takes the `None` arm (line 237) and returns the `(min, max)` seq-len of
/// just `(w, scales)` — `seq_len` is a metadata read (no eval), so the
/// private helper is exercised directly in-module with no Metal device.
/// Closed-form oracle: `w` seq-len 5, `scales` seq-len 3 -> `(3, 5)`.
#[test]
fn triple_component_len_range_bias_less_uses_none_arm() {
  let w = kv(&[0.0, 1.0, 2.0, 3.0, 4.0]); // seq-len 5
  let s = kv(&[0.0, 1.0, 2.0]); // seq-len 3
  let triple: (Array, Array, Option<Array>) = (w, s, None);
  let (lo, hi) =
    StandardQuantizedKvCache::triple_component_len_range("bias-less", &triple).unwrap();
  assert_eq!(lo, 3, "min seq-len across (w=5, scales=3) is 3");
  assert_eq!(hi, 5, "max seq-len across (w=5, scales=3) is 5");

  // The Some-bias arm (line 236) for contrast: a longer bias widens `max`.
  let w2 = kv(&[0.0, 1.0]); // 2
  let s2 = kv(&[0.0, 1.0]); // 2
  let b2 = kv(&[0.0, 1.0, 2.0, 3.0]); // 4
  let triple2: (Array, Array, Option<Array>) = (w2, s2, Some(b2));
  let (lo2, hi2) =
    StandardQuantizedKvCache::triple_component_len_range("biased", &triple2).unwrap();
  assert_eq!((lo2, hi2), (2, 4), "bias seq-len 4 widens the max to 4");
}

// ── tree_map: the None-bias arm (line 175) via copy ──────────────────────

/// `copy()` of a cache whose stored triples are bias-less drives
/// [`tree_map`]'s `None`-bias arm (line 175) through the `clone_triple`
/// `try_clone` callback (a refcount clone — the `copy` itself touches no
/// Metal device). Both `keys` and `values` use the `Some(triple)` clone arms
/// (lines 1012/1016) with `biases == None`; the marker readback below routes
/// through `quantized_state` -> `slice_seq` -> `to_vec` (real MLX compute,
/// validated on the Metal host).
#[test]
fn copy_bias_less_triples_takes_tree_map_none_arm() {
  let kw = kv(&[10.0, 20.0]);
  let ks = kv(&[1.0, 1.0]);
  let vw = kv(&[100.0, 200.0]);
  let vs = kv(&[2.0, 2.0]);
  let c = StandardQuantizedKvCache {
    keys: Some((kw, ks, None)),
    values: Some((vw, vs, None)),
    offset: 2,
    group_size: 64,
    bits: 8,
  };
  let cp = c.copy().unwrap();
  assert_eq!(cp.offset(), 2, "copied offset matches");
  assert!(!cp.is_empty());
  let q = cp.as_quantized().unwrap();
  let (qk, qv) = q
    .quantized_state()
    .unwrap()
    .expect("copied cache still has triples");
  // The bias `None` survived the tree_map None arm on BOTH sides.
  assert!(qk.2.is_none(), "copied K triple is bias-less (None arm)");
  assert!(qv.2.is_none(), "copied V triple is bias-less (None arm)");
  // Marker fidelity: the copied w component carries the original markers.
  let mut kw_copy = ops::shape::contiguous(&qk.0, false).unwrap();
  assert_eq!(
    kw_copy.to_vec::<f32>().unwrap(),
    vec![10.0, 20.0],
    "copied K w markers preserved (refcount clone)"
  );
}

// ── concat_triple: the mismatched-bias InvariantViolation arm (359-361) ──

/// [`concat_triple`] of a `Some(biases)` prev with a `None`-biases new
/// triple (or vice versa) is a recoverable `InvariantViolation` (lines
/// 359-361) — the affine mode always yields `Some`, so a mixed pairing
/// means a bias-less state was loaded then an affine triple produced. The
/// two leading `concat_seq`s (w, scales) run first (real MLX compute,
/// Metal-host only); only the bias `match` raises. Exercised directly
/// in-module (the public path always pairs `Some`/`Some`).
#[test]
fn concat_triple_mismatched_bias_is_invariant_violation() {
  let pw = kv(&[10.0]);
  let ps = kv(&[1.0]);
  let pb = kv(&[0.5]);
  let nw = kv(&[20.0]);
  let ns = kv(&[2.0]);
  // prev has Some(biases), new has None -> the `_ =>` mismatched arm.
  let prev: (Array, Array, Option<Array>) = (pw, ps, Some(pb));
  let new: (Array, Array, Option<Array>) = (nw, ns, None);
  match StandardQuantizedKvCache::concat_triple(&prev, &new).unwrap_err() {
    Error::InvariantViolation(p) => {
      assert!(
        p.context().contains("concatenating quantized triples"),
        "context must name the triple concat, got: {}",
        p.context()
      );
      assert!(
        p.requirement().contains("biases must be present in both"),
        "requirement must describe the bias-presence invariant, got: {}",
        p.requirement()
      );
    }
    other => panic!("expected InvariantViolation, got {other:?}"),
  }

  // The opposite pairing (prev None, new Some) hits the same `_ =>` arm.
  let prev2: (Array, Array, Option<Array>) = (kv(&[10.0]), kv(&[1.0]), None);
  let new2: (Array, Array, Option<Array>) = (kv(&[20.0]), kv(&[2.0]), Some(kv(&[0.5])));
  assert!(matches!(
    StandardQuantizedKvCache::concat_triple(&prev2, &new2),
    Err(Error::InvariantViolation(_))
  ));
}

// ── enforce_offset_len_invariant: asymmetric (Some,None)/(None,Some) ─────
//    storage arms (lines 542-543)

/// `enforce_offset_len_invariant` with `keys == Some` but `values == None`
/// takes the `(Some(k), None)` arm (line 542): `new_offset` converges to the
/// keys' min component seq-len, the values stay `None`, and the keys are
/// re-trimmed only if a component exceeds `new_offset`. Drives real
/// `slice_seq`/`seq_len` (Metal-host only). Closed-form oracle: keys stored
/// seq-len 3, restored offset 5 -> NumPy clamp gives stored len 3, so
/// `new_offset` converges DOWN to 3.
#[test]
fn enforce_offset_invariant_keys_only_some_arm() {
  let kw = kv(&[10.0, 11.0, 12.0]); // seq-len 3
  let ks = kv(&[1.0, 1.0, 1.0]); // seq-len 3
  let mut c = StandardQuantizedKvCache {
    keys: Some((kw, ks, None)),
    values: None,
    offset: 5, // > stored 3 -> underlength clamp down to 3
    group_size: 64,
    bits: 8,
  };
  c.enforce_offset_len_invariant().unwrap();
  assert_eq!(
    c.offset(),
    3,
    "offset clamps down to the keys' stored seq-len"
  );
  assert!(
    c.values.is_none(),
    "values stay None through the (Some, None) arm"
  );
  // `state()`/`quantized_state()` both require keys AND values to be Some, so
  // with values == None they short-circuit; observe the kept keys directly
  // through the private field (in-module access). Both components keep their
  // stored seq-len 3 (offset 5 clamped DOWN by the NumPy slice).
  let (kept_w, kept_s, kept_b) = c.keys.as_ref().expect("keys retained");
  assert_eq!(
    kept_w.shape(),
    vec![1, 1, 3, 1],
    "K w at the clamped offset"
  );
  assert_eq!(
    kept_s.shape(),
    vec![1, 1, 3, 1],
    "K scales at the clamped offset"
  );
  assert!(kept_b.is_none(), "bias-less keys stay bias-less");

  // Symmetric mirror: values == Some, keys == None -> the `(None, Some)`
  // arm (line 543). Stored seq-len 2, restored offset 2 (consistent) -> a
  // no-op clamp.
  let vw = kv(&[100.0, 200.0]); // seq-len 2
  let vs = kv(&[2.0, 2.0]);
  let mut c2 = StandardQuantizedKvCache {
    keys: None,
    values: Some((vw, vs, None)),
    offset: 2,
    group_size: 64,
    bits: 8,
  };
  c2.enforce_offset_len_invariant().unwrap();
  assert_eq!(c2.offset(), 2, "consistent values offset is a no-op clamp");
  assert!(
    c2.keys.is_none(),
    "keys stay None through the (None, Some) arm"
  );
}

// ── materialize: the Some/None eval arms (lines 684-699) ─────────────────

/// `materialize()` force-evals every present triple array of `keys`
/// (lines 693-695) and `values` (lines 696-698) — including the `Some(b)`
/// bias eval (lines 688-690) — then leaves the observable state unchanged.
/// Drives real `Array::eval` (Metal-host only). An empty cache hits both
/// `None` guards (a pure no-op).
#[test]
fn materialize_evals_triples_and_empty_is_noop() {
  // With-bias triples so the `Some(b)` eval arm (688-690) is taken.
  let kw = kv4(2, 1, 10.0);
  let ks = kv(&[1.0, 1.0]);
  let kb = kv(&[0.5, 0.5]);
  let vw = kv4(2, 1, 100.0);
  let vs = kv(&[2.0, 2.0]);
  let vb = kv(&[0.25, 0.25]);
  let mut c = StandardQuantizedKvCache {
    keys: Some((kw, ks, Some(kb))),
    values: Some((vw, vs, Some(vb))),
    offset: 2,
    group_size: 64,
    bits: 8,
  };
  c.materialize().unwrap();
  // Pure memory barrier: offset and the logical triples are intact.
  assert_eq!(c.offset(), 2);
  let (qk, qv) = c.quantized_state().unwrap().unwrap();
  assert!(
    qk.2.is_some() && qv.2.is_some(),
    "biases survive materialize"
  );
  let mut kw_after = ops::shape::contiguous(&qk.0, false).unwrap();
  assert_eq!(
    kw_after.to_vec::<f32>().unwrap(),
    vec![10.0, 11.0],
    "K w markers unchanged by materialize"
  );
  let mut vw_after = ops::shape::contiguous(&qv.0, false).unwrap();
  assert_eq!(
    vw_after.to_vec::<f32>().unwrap(),
    vec![100.0, 101.0],
    "V w markers unchanged by materialize (own stream)"
  );

  // Empty cache: both `if let Some` guards are false -> a no-op.
  let mut empty = StandardQuantizedKvCache::new(64, 8).unwrap();
  empty.materialize().unwrap();
  assert!(empty.is_empty());
}
