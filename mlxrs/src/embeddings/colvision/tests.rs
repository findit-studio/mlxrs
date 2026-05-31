use super::*;

// ---------- score_single_vector ----------

/// python line 52: `if len(qs) == 0: raise ValueError("No queries
/// provided")`. Faithful error-message parity.
#[test]
fn score_single_vector_rejects_empty_queries() {
  let p = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let err = score_single_vector(&[], std::slice::from_ref(&p)).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("No queries provided"),
    "expected python parity msg, got {msg}"
  );
}

/// python line 54: `if len(ps) == 0: raise ValueError("No passages
/// provided")`.
#[test]
fn score_single_vector_rejects_empty_passages() {
  let q = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let err = score_single_vector(std::slice::from_ref(&q), &[]).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("No passages provided"),
    "expected python parity msg, got {msg}"
  );
}

/// python line 59: `mx.einsum("bd,cd->bc", qs, ps)` — for 2 queries
/// `[[1,0,0],[0,1,0]]` and 2 passages `[[1,1,1],[2,0,2]]` the
/// expected scores are `[[1,2],[1,0]]`. Hand-traced.
#[test]
fn score_single_vector_dot_product_shape_and_values() {
  let q0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0], &(3,)).unwrap();
  let q1 = Array::from_slice::<f32>(&[0.0, 1.0, 0.0], &(3,)).unwrap();
  let p0 = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(3,)).unwrap();
  let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 2.0], &(3,)).unwrap();
  let mut scores =
    score_single_vector(&[q0, q1], &[p0, p1]).expect("score_single_vector should succeed");
  assert_eq!(scores.shape(), vec![2, 2], "shape (B,C) = (2,2)");
  assert_eq!(scores.dtype().unwrap(), Dtype::F32, "f32 cast (python L63)");
  let v = scores.to_vec::<f32>().unwrap();
  // [[<q0,p0>=1, <q0,p1>=2], [<q1,p0>=1, <q1,p1>=0]]
  assert_eq!(v, vec![1.0, 2.0, 1.0, 0.0]);
}

/// REGRESSION (single-vector analog): a
/// `(0,)` query embedding (zero-element vector) must be rejected.
/// The single-vector path does not go through [`pad_to_max`]; it has
/// its own early guard. A `(0,)` vector would dot-product to `0.0`
/// against every passage regardless of content, silently collapsing
/// the ranking signal. Assertion: returns [`Error::OutOfRange`]
/// whose message contains `"zero tokens"`.
#[test]
fn score_single_vector_rejects_zero_token_query() {
  let q_empty = Array::from_slice::<f32>(&[], &(0,)).unwrap();
  let p = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let err =
    score_single_vector(std::slice::from_ref(&q_empty), std::slice::from_ref(&p)).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("queries"),
    "expected 'queries' in message, got {msg}"
  );
}

/// REGRESSION (single-vector analog): a
/// `(0,)` passage embedding must be rejected for the same reasons as
/// the query analog. Assertion: returns [`Error::OutOfRange`]
/// whose message contains `"zero tokens"` and identifies the
/// passage path.
#[test]
fn score_single_vector_rejects_zero_token_passage() {
  let q = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let p_empty = Array::from_slice::<f32>(&[], &(0,)).unwrap();
  let err =
    score_single_vector(std::slice::from_ref(&q), std::slice::from_ref(&p_empty)).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("passages"),
    "expected 'passages' in message, got {msg}"
  );
}

/// python line 63: `.astype(mx.float32)` — non-f32 input must come
/// back as f32 (dtype upcast for the score, but inputs themselves
/// stay in their dtype until the final cast).
#[test]
fn score_single_vector_casts_result_to_f32_from_f16() {
  let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(2,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let p0 = Array::from_slice::<f32>(&[1.0, 1.0], &(2,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let scores = score_single_vector(&[q0], &[p0]).unwrap();
  assert_eq!(scores.shape(), vec![1, 1]);
  assert_eq!(
    scores.dtype().unwrap(),
    Dtype::F32,
    "result must be f32 even with f16 inputs"
  );
}

// ---------- score_multi_vector ----------

/// python lines 75-78: same empty-input ValueError parity.
#[test]
fn score_multi_vector_rejects_empty_queries() {
  let p = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
  let err = score_multi_vector(&[], std::slice::from_ref(&p), 128).unwrap_err();
  assert!(format!("{err}").contains("No queries provided"));
}

#[test]
fn score_multi_vector_rejects_empty_passages() {
  let q = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
  let err = score_multi_vector(std::slice::from_ref(&q), &[], 128).unwrap_err();
  assert!(format!("{err}").contains("No passages provided"));
}

/// Mlxrs-only guard: a `batch_size == 0` would put the python
/// `range(0, len(qs), 0)` into a `ValueError`. Surface a
/// recoverable error instead of looping forever.
#[test]
fn score_multi_vector_rejects_zero_batch_size() {
  let q = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
  let p = Array::from_slice::<f32>(&[1.0; 4], &(2, 2)).unwrap();
  let err = score_multi_vector(std::slice::from_ref(&q), std::slice::from_ref(&p), 0).unwrap_err();
  assert!(format!("{err}").contains("batch_size"));
}

/// REGRESSION: a zero-token query is
/// rejected by `score_multi_vector` — even though the outer
/// `qs.is_empty()` guard passes, the per-array `shape[0] == 0` check
/// must fire. The contract is that callers filter out
/// empty-tokenization inputs before invoking the scorer; if they
/// don't, the failure must be observable and recoverable (not
/// non-finite scores).
///
/// REGRESSION: the message must carry the
/// path tag (`queries`) AND the *global* index, not a tile-local
/// `array N` from the inner `pad_to_max` helper.
///
/// Assertion: returns [`Error::OutOfRange`] whose message
/// contains `"zero tokens"` AND `"queries[0]"`, not a propagation of
/// `-inf` through the masked MaxSim.
#[test]
fn score_multi_vector_rejects_zero_token_query() {
  let q_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let err = score_multi_vector(
    std::slice::from_ref(&q_empty),
    std::slice::from_ref(&p),
    128,
  )
  .unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange from score_multi_vector pre-validation, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("queries") && msg.contains("index 0"),
    "expected 'queries' + 'index 0' in message, got {msg}"
  );
}

/// REGRESSION: the high-severity fixture.
/// `q = [[1, 0]]`, `p0 = [[0_size_2]]` (zero tokens), `p1 = [[2, 0],
/// [0, 1]]`. Without the zero-token guard, the `(c=2, s_max=2)` mask
/// row for `p0` would be all-`false`, [`select`] would replace every
/// `(b=1, c=0, n=1, s)` similarity with `-inf`, `max(axis=3)` would
/// return `-inf` for that passage, and `sum(axis=2)` would propagate
/// to a `-inf` ranking score. The guard surfaces a recoverable
/// [`Error::OutOfRange`] instead.
///
/// REGRESSION: the message must carry the
/// path tag (`passages`) AND the *global* index — here `passages[0]`
/// — even though `p0` is also at tile-local index 0 (the
/// distinguishing global-vs-local fixture is below).
#[test]
fn score_multi_vector_rejects_zero_token_passage_in_mixed_tile() {
  let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  // p0 is a zero-token passage `(0, 2)`; p1 has two real tokens.
  let p0 = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  // batch_size=2 forces both passages into the same tile, so the
  // padded `p0` would otherwise produce an all-masked row.
  let err = score_multi_vector(std::slice::from_ref(&q), &[p0, p1], 2).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange from pre-validation (NOT -inf propagation), got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("passages") && msg.contains("index 0"),
    "expected 'passages' + 'index 0' in message, got {msg}"
  );
}

/// REGRESSION: the distinguishing global-
/// vs-tile-local fixture for the QUERY path. With `qs.len() = 4`
/// and `batch_size = 2`, the offending zero-token query at global
/// index 3 lives in the SECOND tile and would have been reported by
/// a tile-local-only impl as `array 1` (tile-local within tile #1).
/// The pre-validate fix reports the global `queries[3]` instead.
#[test]
fn score_multi_vector_rejects_zero_token_query_at_non_zero_global_index() {
  let q_valid_0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let q_valid_1 = Array::from_slice::<f32>(&[0.0, 1.0], &(1, 2)).unwrap();
  let q_valid_2 = Array::from_slice::<f32>(&[1.0, 1.0], &(1, 2)).unwrap();
  let q_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  // Empty query at global #3 (NOT first tile when batch_size=2): tiles
  // {q_valid_0, q_valid_1} and {q_valid_2, q_empty}. q_empty is
  // tile-local index 1 within tile #1 but global index 3.
  let qs = vec![q_valid_0, q_valid_1, q_valid_2, q_empty];
  let result = score_multi_vector(&qs, std::slice::from_ref(&p), 2);
  let err = match result {
    Err(e) => e,
    Ok(_) => panic!("expected OutOfRange, got Ok"),
  };
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("queries") && msg.contains("index 3"),
    "expected 'queries' + global index 3, got: {msg}"
  );
  // Defense-in-depth: assert the tile-local forms are absent.
  assert!(
    !msg.contains("index 1") && !msg.contains("array 1"),
    "tile-local index leaked: {msg}"
  );
}

/// REGRESSION: the distinguishing global-
/// vs-tile-local fixture. With `ps.len() = 4` and `batch_size = 2`,
/// the offending zero-token passage at global index 3 lives in the
/// SECOND tile and would have been reported by the inner
/// `pad_to_max` helper as `array 1` (tile-local). The pre-validate
/// fix reports the global `passages[3]` instead.
#[test]
fn score_multi_vector_rejects_zero_token_passage_at_non_zero_global_index() {
  let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let p0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let p1 = Array::from_slice::<f32>(&[0.0, 1.0], &(1, 2)).unwrap();
  let p2 = Array::from_slice::<f32>(&[1.0, 1.0], &(1, 2)).unwrap();
  let p_empty = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  // batch_size=2 → tiles {p0, p1} and {p2, p_empty}; p_empty is
  // tile-local index 1 within its tile but global index 3.
  let err = score_multi_vector(std::slice::from_ref(&q), &[p0, p1, p2, p_empty], 2).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange from pre-validation, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("passages") && msg.contains("index 3"),
    "expected 'passages' + global 'index 3' (not tile-local 'array 1') in message, got {msg}"
  );
}

/// python lines 100-102: MaxSim for one query, one passage, both
/// rank-2 with the same `n`. Hand-traced expected score.
///
/// q = [[1, 0],         p = [[1, 0],
///      [0, 1]]              [0, 1]]
/// sim = q @ p.T = [[1,0],[0,1]] (n=2, s=2)
/// max over s (axis=1, i.e. axis=3 in the (b,c,n,s) tensor) = [1, 1]
/// sum over n (axis=0, i.e. axis=2 in (b,c,n)) = 2
/// → scores = [[2]] in f32.
#[test]
fn score_multi_vector_identity_pair() {
  let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let mut scores = score_multi_vector(&[q], &[p], 128).unwrap();
  assert_eq!(scores.shape(), vec![1, 1]);
  assert_eq!(scores.dtype().unwrap(), Dtype::F32);
  assert_eq!(scores.to_vec::<f32>().unwrap(), vec![2.0]);
}

/// python lines 80-91 + 100-102: MaxSim across two queries (with
/// different `n`) and two passages (with different `s`) at
/// `batch_size = 1` exercises the inner-loop, [`pad_to_max`]
/// padding, and cross-tile [`concatenate`].
///
/// q0 = [[1,0]]            (n=1)        q1 = [[1,0],[0,1]]    (n=2)
/// p0 = [[1,0],[0,1]]      (s=2)        p1 = [[1,0]]           (s=1)
///
/// MaxSim(q,p) = Σ_n max_s <q_n, p_s>
/// (q0,p0): max([<[1,0],[1,0]>, <[1,0],[0,1]>]) = max(1,0)=1 → sum=1
/// (q0,p1): max([<[1,0],[1,0]>]) = 1 → sum=1
/// (q1,p0): n=2 rows → row0 max(1,0)=1; row1 max(0,1)=1 → sum=2
/// (q1,p1): n=2 rows → row0 max(1)=1; row1 max(0)=0 → sum=1
///
/// expected (B=2, C=2) = [[1,1],[2,1]]
#[test]
fn score_multi_vector_ragged_n_and_s_with_batching() {
  let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let q1 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let p0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let p1 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let mut scores = score_multi_vector(&[q0, q1], &[p0, p1], 1).unwrap();
  assert_eq!(scores.shape(), vec![2, 2]);
  assert_eq!(scores.dtype().unwrap(), Dtype::F32);
  assert_eq!(scores.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 1.0]);
}

/// `batch_size >= len(qs)` must produce the same result as a tiled
/// run — the outer/inner loop semantics are batch-size-agnostic.
/// Re-runs the previous fixture with `batch_size = 128` (python
/// default).
#[test]
fn score_multi_vector_default_batch_size_matches_tiled() {
  let q0 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let q1 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let p0 = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let p1 = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let mut scores = score_multi_vector(&[q0, q1], &[p0, p1], 128).unwrap();
  assert_eq!(scores.shape(), vec![2, 2]);
  assert_eq!(scores.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 1.0]);
}

/// python line 110: `.astype(mx.float32)` — f16 multi-vector inputs
/// must produce an f32 score.
#[test]
fn score_multi_vector_casts_result_to_f32_from_f16() {
  let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let p = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let scores = score_multi_vector(&[q], &[p], 128).unwrap();
  assert_eq!(scores.shape(), vec![1, 1]);
  assert_eq!(scores.dtype().unwrap(), Dtype::F32);
}

// ---------- pad_to_max ----------

/// python lines 80-91: ragged inputs zero-padded along axis=0 to the
/// slice max, then stacked. Hand-traced shape + dtype.
#[test]
fn pad_to_max_pads_ragged_then_stacks() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap(); // n=1
  let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2)).unwrap(); // n=2
  let (mut padded, _lens) = pad_to_max(&[a, b]).unwrap();
  // (len=2, max_n=2, d=2). a is padded with one zero row; b is unchanged.
  assert_eq!(padded.shape(), vec![2, 2, 2]);
  let v = padded.to_vec::<f32>().unwrap();
  // a → [[1,2],[0,0]]; b → [[3,4],[5,6]]
  assert_eq!(v, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Empty slice precondition surfaces as a recoverable error rather
/// than the python `IndexError` (`arrays[0]` on `[]`).
#[test]
fn pad_to_max_rejects_empty_slice() {
  let err = pad_to_max(&[]).unwrap_err();
  assert!(format!("{err}").contains("empty"));
}

/// python `arrays[0].shape[1]` assumes rank-2 inputs; surface a
/// recoverable error on rank-1 instead of an FFI panic.
#[test]
fn pad_to_max_rejects_non_rank_2() {
  let bad = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let err = pad_to_max(std::slice::from_ref(&bad)).unwrap_err();
  assert!(format!("{err}").contains("rank-2"));
}

/// All arrays must share `emb_dim` (the python ref implicitly
/// requires this — `mx.stack` would fail on mismatched dims after
/// the per-array pad, but we catch it upfront with a clearer
/// message).
#[test]
fn pad_to_max_rejects_mismatched_emb_dim() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap();
  let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0], &(1, 3)).unwrap();
  let err = pad_to_max(&[a, b]).unwrap_err();
  assert!(format!("{err}").contains("emb_dim"));
}

/// REGRESSION: a `(0, d)` array must be
/// rejected. Without the guard, [`pad_to_max`] records `0` in
/// `original_lengths`, [`score_multi_vector`]'s mask loop builds an
/// all-`false` row for the offending passage, [`select`] replaces
/// every position with `-inf`, `max(axis=3)` returns `-inf`, and the
/// resulting ranking score is non-finite. Enforce the precondition
/// here so the failure is observable, recoverable, and named.
///
/// Assertion: returns [`Error::OutOfRange`] whose message contains
/// `"zero tokens"` and identifies the offending array index.
#[test]
fn pad_to_max_rejects_zero_token_array() {
  // `(0, 2)` — a zero-token query / passage embedding.
  let zero = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  let err = pad_to_max(std::slice::from_ref(&zero)).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("index 0"),
    "expected 'index 0' in message, got {msg}"
  );
  // Even when the zero-token array is not the first one in the slice,
  // the guard must still fire and identify its index.
  let good = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap();
  let bad = Array::from_slice::<f32>(&[], &(0, 2)).unwrap();
  let err2 = pad_to_max(&[good, bad]).unwrap_err();
  let msg2 = format!("{err2}");
  assert!(
    msg2.contains("index 1"),
    "expected 'index 1' in message, got {msg2}"
  );
}

/// dtype-fidelity: a half-precision input batch must stay
/// half-precision through `pad_to_max` (python `dtype=a.dtype` on
/// line 87). No silent f32 promotion of the padding rows.
#[test]
fn pad_to_max_preserves_f16_dtype() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let (padded, _lens) = pad_to_max(&[a, b]).unwrap();
  assert_eq!(padded.shape(), vec![2, 2, 2]);
  assert_eq!(
    padded.dtype().unwrap(),
    Dtype::F16,
    "padding must preserve input dtype (python L87 `dtype=a.dtype`)"
  );
}

/// Sanity check on the divergence-from-python tuple return shape:
/// `pad_to_max` reports the **original** lengths of each input array
/// (before zero-padding), in input order. The mask in
/// [`score_multi_vector`] depends on this contract.
#[test]
fn pad_to_max_returns_original_lengths() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0], &(1, 2)).unwrap(); // n=1
  let b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0, 6.0], &(2, 2)).unwrap(); // n=2
  let c = Array::from_slice::<f32>(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &(3, 2)).unwrap(); // n=3
  let (padded, lens) = pad_to_max(&[a, b, c]).unwrap();
  assert_eq!(padded.shape(), vec![3, 3, 2], "stacked to (3, max_n=3, 2)");
  assert_eq!(
    lens,
    vec![1, 2, 3],
    "original_lengths must mirror input order"
  );
}

/// REGRESSION: zero-padded passages must not
/// win `max(axis=3)` for signed embeddings. With
/// `q = [[1, 0]]`, `p0 = [[-1, 0]]`, `p1 = [[2, 0], [0, 1]]`:
/// - Scoring `p0` alone (`batch_size = 1`, no padding): MaxSim is
///   `max(<q, p0_0>) = max(-1) = -1` → sum = -1.
/// - Scoring `[p0, p1]` together with `batch_size = 2` (p0 gets a
///   zero-pad row to match p1's length 2): python ref returns
///   `max(-1, 0) = 0` (wrong; padded zero won). mlxrs must return
///   `-1.0` (the unpadded answer) in BOTH cases — ranking is
///   batch-size-agnostic.
///
/// Rationale: the upstream
/// `mlx_embeddings/colvision_processor.py` includes the zero-padded
/// columns in `mx.max(sim, axis=3)`. mlxrs masks them to
/// `f32::NEG_INFINITY` (cast to the input dtype) before the max.
#[test]
fn score_multi_vector_ragged_negative_similarity_batch_size_agnostic() {
  let q = Array::from_slice::<f32>(&[1.0, 0.0], &(1, 2)).unwrap();
  let p0 = Array::from_slice::<f32>(&[-1.0, 0.0], &(1, 2)).unwrap();
  let p1 = Array::from_slice::<f32>(&[2.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();

  // Branch A: batch_size=1, p0 is processed alone (no padding).
  let mut scores_b1 = score_multi_vector(
    std::slice::from_ref(&q),
    &[p0.try_clone().unwrap(), p1.try_clone().unwrap()],
    1,
  )
  .unwrap();
  assert_eq!(scores_b1.shape(), vec![1, 2]);
  let v_b1 = scores_b1.to_vec::<f32>().unwrap();

  // Branch B: batch_size=2, p0 is tile-padded with a zero row.
  let mut scores_tiled = score_multi_vector(std::slice::from_ref(&q), &[p0, p1], 2).unwrap();
  assert_eq!(scores_tiled.shape(), vec![1, 2]);
  let v_tiled = scores_tiled.to_vec::<f32>().unwrap();

  // p0 score (index 0) must be -1.0 in BOTH branches.
  assert_eq!(
    v_b1[0], -1.0,
    "p0 alone: <q,p0_0> = -1.0; sum over the single query token = -1.0"
  );
  assert_eq!(
    v_tiled[0], -1.0,
    "p0 tiled with p1: padded zero column must be masked → -1.0, not 0.0"
  );

  // p1's score should be the same in both branches too (sanity).
  assert_eq!(
    v_b1[1], v_tiled[1],
    "p1 score must be tile-invariant in both branches"
  );

  // The whole vector must be batch-size-agnostic.
  assert_eq!(
    v_b1, v_tiled,
    "score_multi_vector ranking must not depend on batch_size"
  );
}

// ---------- trait shape + impl ----------

/// A minimal in-test impl of [`BaseColVisionProcessor`] proves the
/// trait shape is implementable (and the contract closes over the
/// model-specific state a concrete processor would own). It
/// delegates `score` to [`score_multi_vector`] exactly like
/// `colidefics3.Processor.score` (python line 329).
struct TestProcessor;

impl BaseColVisionProcessor for TestProcessor {
  fn process_images(&self, images: &[Vec<u8>]) -> Result<ProcessorBatch> {
    // Test-only stub: deposit a single `(len(images),)` int tensor
    // recording the batch size — the seam only checks the contract
    // shape, not the model preprocessor semantics (which are
    // out of scope).
    let mut batch = ProcessorBatch::new();
    let count = i32::try_from(images.len()).unwrap_or(0);
    batch.insert(
      "pixel_values_count".into(),
      Array::from_slice::<i32>(&[count], &(1,))?,
    );
    Ok(batch)
  }
  fn process_queries(
    &self,
    queries: &[&str],
    _max_length: usize,
    _suffix: Option<&str>,
  ) -> Result<ProcessorBatch> {
    let mut batch = ProcessorBatch::new();
    let count = i32::try_from(queries.len()).unwrap_or(0);
    batch.insert(
      "input_ids_count".into(),
      Array::from_slice::<i32>(&[count], &(1,))?,
    );
    Ok(batch)
  }
  fn score(&self, qs: &[Array], ps: &[Array], batch_size: usize) -> Result<Array> {
    score_multi_vector(qs, ps, batch_size)
  }
}

/// The trait is dyn-compatible-via-impl: a test impl returns the
/// expected dict shape from both branches, and `score` delegates to
/// `score_multi_vector` (mirroring `colidefics3.Processor.score`).
#[test]
fn base_processor_trait_impl_round_trips() {
  let p = TestProcessor;
  // process_images: dummy 2-image batch.
  let imgs = vec![vec![0u8, 1, 2], vec![3u8, 4, 5]];
  let img_batch = p.process_images(&imgs).unwrap();
  assert!(img_batch.contains_key("pixel_values_count"));

  // process_queries: dummy 3-query batch.
  let queries = vec!["query one", "query two", "query three"];
  let q_batch = p.process_queries(&queries, 50, None).unwrap();
  assert!(q_batch.contains_key("input_ids_count"));

  // score: identity multi-vector pair → 2.0 (matches the standalone
  // `score_multi_vector_identity_pair` test).
  let q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let pp = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2)).unwrap();
  let mut scores = p.score(&[q], &[pp], 128).unwrap();
  assert_eq!(scores.shape(), vec![1, 1]);
  assert_eq!(scores.to_vec::<f32>().unwrap(), vec![2.0]);
}
