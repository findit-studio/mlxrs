//! M3 embeddings: pooling, dispatcher, normalization, ST-config, and
//! similarity tests.
//!
//! Reference basis:
//! - python `mlx-embeddings/tests/test_pooling.py`
//!   (`TestMaxPooling`, `TestPoolingExactValues`, `TestPoolByConfig`,
//!   `TestNormalizePoolingConfig` — including legacy `pooling_mode_*`
//!   keys), `models/pooling.py`, `models/base.py::normalize_embeddings`.
//! - swift `MLXEmbedders/Pooling.swift` (`Strategy`, dispatcher order,
//!   CLS > Mean > Max > Last config priority) +
//!   `MLXArray+Helper.l2Normalized` (eps `1e-12`).
//! - Expected values derived from those references / first principles.

#![cfg(feature = "embeddings")]

use mlxrs::{
  Array, Dtype, Error,
  embeddings::{
    DEFAULT_NORMALIZE_EPS, PoolingStrategy, SWIFT_L2_EPS, cls_pooling, cosine_similarity,
    cosine_similarity_matrix, first_token_pooling, l2_normalize, l2_normalize_eps,
    last_token_pooling, layer_norm, max_pooling, mean_pooling, normalize, pool, pool_post,
    pooling_from_st_config_bytes, pooling_from_st_config_path, pooling_from_st_config_str,
    rms_norm, truncate_last_dim,
  },
};

const TOL: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
  (a - b).abs() <= TOL
}

fn vclose(a: &[f32], b: &[f32]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

// python fixture (test_pooling.py): seq0 = 3 real + 1 pad, seq1 = 4 real.
fn fixture() -> (Array, Array) {
  let emb = Array::from_slice(
    &[
      1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0, 99.0, // seq 0
      10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, // seq 1
    ],
    &(2, 4, 2),
  )
  .unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], &(2, 4)).unwrap();
  (emb, mask)
}

// ───────────────── back-compat: the 6 pre-existing public fns ─────────────────

#[test]
fn mean_pooling_of_ones_with_full_mask_is_ones() {
  let emb = Array::ones::<f32>(&(1, 3, 2)).unwrap();
  let mask = Array::ones::<f32>(&(1, 3)).unwrap();
  let mut pooled = mean_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 2]);
  assert_eq!(pooled.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn mean_pooling_ignores_padding() {
  let emb = Array::from_slice(&[1.0_f32, 5.0, 99.0], &(1, 3, 1)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  let mut pooled = mean_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 1]);
  assert!(close(pooled.to_vec::<f32>().unwrap()[0], 3.0));
}

#[test]
fn cls_pooling_selects_first_real_token() {
  // mask [0,1,1]: python cls_pooling = argmax(mask)=1 -> row [2,2]
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 1.0, 1.0], &(1, 3)).unwrap();
  let mut pooled = cls_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 2]);
  assert_eq!(pooled.to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
}

#[test]
fn last_token_pooling_selects_last_real_token() {
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 9.0, 9.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  let mut pooled = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 2]);
  assert_eq!(pooled.to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
}

// ───────── F1: last_token_pooling left/mixed-pad correctness ─────────
//
// Codex round-2 [high]: the old `sum(mask)-1` index is correct only for
// RIGHT-padding; a left-padded row `[0,0,1,1]` gathered the padding at
// index 1 instead of the last real token at index 3 — silent wrong
// embeddings for left-padded last-token models (Qwen3-embed). The impl
// now matches python `mlx-embeddings` `lasttoken_pooling` exactly:
// `last = seq_len - 1 - argmax(flip(mask, axis=1), axis=1)` with the
// all-pad fallback `seq_len-1` and a trailing `* mask`. Expected values
// below are derived by hand-evaluating that python formula.

#[test]
fn last_token_pooling_left_padded_selects_last_real_token() {
  // seq_len 4, left-padded mask [0,0,1,1].
  // python: flipped=[1,1,0,0]; argmax=0; last = 4-0-1 = 3.
  // emb index 3 = [7,7] (a real token, mask=1 → *mask keeps it).
  let emb = Array::from_slice(
    &[
      9.0_f32, 9.0, // 0 (pad)
      8.0, 8.0, // 1 (pad)
      6.0, 6.0, // 2 (real)
      7.0, 7.0, // 3 (real, LAST real)
    ],
    &(1, 4, 2),
  )
  .unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0, 1.0, 1.0], &(1, 4)).unwrap();
  let mut pooled = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 2]);
  assert!(vclose(&pooled.to_vec::<f32>().unwrap(), &[7.0, 7.0]));
}

#[test]
fn last_token_pooling_mixed_left_and_right_pad_batch() {
  // Row 0 left-padded  [0,0,1,1]: flipped=[1,1,0,0], argmax=0, last=3.
  // Row 1 right-padded [1,1,0,0]: flipped=[0,0,1,1], argmax=2, last=1.
  let emb = Array::from_slice(
    &[
      // row 0
      90.0_f32, 90.0, // 0 (pad)
      80.0, 80.0, // 1 (pad)
      60.0, 60.0, // 2 (real)
      70.0, 70.0, // 3 (real, LAST real → expected)
      // row 1
      1.0, 1.0, // 0 (real)
      2.0, 2.0, // 1 (real, LAST real → expected)
      99.0, 99.0, // 2 (pad)
      99.0, 99.0, // 3 (pad)
    ],
    &(2, 4, 2),
  )
  .unwrap();
  let mask = Array::from_slice(
    &[
      0.0_f32, 0.0, 1.0, 1.0, // row 0 left-pad
      1.0, 1.0, 0.0, 0.0, // row 1 right-pad
    ],
    &(2, 4),
  )
  .unwrap();
  let mut pooled = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![2, 2]);
  assert!(vclose(
    &pooled.to_vec::<f32>().unwrap(),
    &[70.0, 70.0, 2.0, 2.0]
  ));
}

#[test]
fn last_token_pooling_all_pad_row_falls_back_to_zeros() {
  // python: max(flipped)==0 → flip_indices=seq_len-1=3; last=4-3-1=0;
  // gather (emb*mask)[0] and mask[0]==0 → zeros (python parity).
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(1, 3)).unwrap();
  let mut pooled = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 2]);
  assert!(vclose(&pooled.to_vec::<f32>().unwrap(), &[0.0, 0.0]));
}

#[test]
fn last_token_pooling_left_padded_via_dispatcher() {
  // Same left-pad row through `pool(.., PoolingStrategy::Last, ..)`.
  let emb = Array::from_slice(
    &[
      9.0_f32, 9.0, 8.0, 8.0, 6.0, 6.0, 7.0, 7.0, // left-pad row
    ],
    &(1, 4, 2),
  )
  .unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0, 1.0, 1.0], &(1, 4)).unwrap();
  let mut p = pool(
    &emb,
    &mask,
    PoolingStrategy::Last,
    false,
    None,
    false,
    false,
  )
  .unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[7.0, 7.0]));
}

#[test]
fn last_token_pooling_right_padded_unchanged_regression() {
  // The pre-existing right-padded fixture value MUST be unchanged:
  // seq0 [1,1,1,0] → flipped [0,1,1,1], argmax=1, last=4-1-1=2 → [5,6];
  // seq1 [1,1,1,1] → flipped [1,1,1,1], argmax=0, last=4-0-1=3 → [70,80].
  let (emb, mask) = fixture();
  let mut lt = last_token_pooling(&emb, &mask).unwrap();
  assert!(vclose(
    &lt.to_vec::<f32>().unwrap(),
    &[5.0, 6.0, 70.0, 80.0]
  ));
}

#[test]
fn l2_normalize_yields_unit_norm() {
  let v = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let n = l2_normalize(&v).unwrap();
  let mut nn = mlxrs::ops::linalg_full::norm(&n, 2.0, &[-1], false).unwrap();
  assert!(close(nn.item::<f32>().unwrap(), 1.0));
}

#[test]
fn cosine_similarity_identical_is_one() {
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  assert!(close(cosine_similarity(&a, &b).unwrap(), 1.0));
}

#[test]
fn cosine_similarity_orthogonal_is_zero() {
  let a = Array::from_slice(&[1.0_f32, 0.0], &(2,)).unwrap();
  let b = Array::from_slice(&[0.0_f32, 1.0], &(2,)).unwrap();
  assert!(close(cosine_similarity(&a, &b).unwrap(), 0.0));
}

#[test]
fn cosine_similarity_matrix_diagonal_is_one() {
  let m = Array::from_slice(&[1.0_f32, 0.0, 0.0, 2.0], &(2, 2)).unwrap();
  let mut sim = cosine_similarity_matrix(&m).unwrap();
  assert_eq!(sim.shape(), vec![2, 2]);
  let v = sim.to_vec::<f32>().unwrap();
  assert!(close(v[0], 1.0));
  assert!(close(v[3], 1.0));
  assert!(close(v[1], 0.0));
}

// ───────────────── max pooling (python TestMaxPooling) ─────────────────

#[test]
fn max_pooling_respects_attention_mask() {
  // python: last position has the largest value but is masked out.
  let emb = Array::from_slice(&[1.0_f32, 3.0, 5.0, 10.0], &(1, 4, 1)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 1.0, 0.0], &(1, 4)).unwrap();
  let mut pooled = max_pooling(&emb, &mask).unwrap();
  assert_eq!(pooled.shape(), vec![1, 1]);
  assert!(close(pooled.to_vec::<f32>().unwrap()[0], 5.0));
}

// ───────────────── exact values per mode (python TestPoolingExactValues) ─────

#[test]
fn pooling_exact_values_fixture() {
  let (emb, mask) = fixture();

  let mut m = mean_pooling(&emb, &mask).unwrap();
  assert!(vclose(&m.to_vec::<f32>().unwrap(), &[3.0, 4.0, 40.0, 50.0]));

  let mut mx = max_pooling(&emb, &mask).unwrap();
  assert!(vclose(
    &mx.to_vec::<f32>().unwrap(),
    &[5.0, 6.0, 70.0, 80.0]
  ));

  let mut lt = last_token_pooling(&emb, &mask).unwrap();
  assert!(vclose(
    &lt.to_vec::<f32>().unwrap(),
    &[5.0, 6.0, 70.0, 80.0]
  ));

  // token-0 path (swift .first / dispatcher .cls)
  let mut ft = first_token_pooling(&emb).unwrap();
  assert!(vclose(
    &ft.to_vec::<f32>().unwrap(),
    &[1.0, 2.0, 10.0, 20.0]
  ));
}

// ───────────────── dispatcher: every PoolingStrategy ─────────────────

#[test]
fn dispatcher_every_strategy_shapes_and_values() {
  let (emb, mask) = fixture();

  for (strat, expected) in [
    (PoolingStrategy::Mean, vec![3.0, 4.0, 40.0, 50.0]),
    (PoolingStrategy::Max, vec![5.0, 6.0, 70.0, 80.0]),
    (PoolingStrategy::Last, vec![5.0, 6.0, 70.0, 80.0]),
    (PoolingStrategy::First, vec![1.0, 2.0, 10.0, 20.0]),
    (PoolingStrategy::Cls, vec![1.0, 2.0, 10.0, 20.0]),
  ] {
    let mut p = pool(&emb, &mask, strat, false, None, false, false).unwrap();
    assert_eq!(p.shape(), vec![2, 2], "shape for {strat:?}");
    assert!(
      vclose(&p.to_vec::<f32>().unwrap(), &expected),
      "value for {strat:?}"
    );
  }
}

#[test]
fn dispatcher_none_is_passthrough() {
  let (emb, mask) = fixture();
  let mut p = pool(
    &emb,
    &mask,
    PoolingStrategy::None,
    false,
    None,
    false,
    false,
  )
  .unwrap();
  // None keeps the (batch, seq, hidden) rank, values unchanged.
  assert_eq!(p.shape(), vec![2, 4, 2]);
  let mut emb2 = emb;
  assert_eq!(p.to_vec::<f32>().unwrap(), emb2.to_vec::<f32>().unwrap());
}

#[test]
fn dispatcher_normalize_flag_yields_unit_rows() {
  let (emb, mask) = fixture();
  let p = pool(&emb, &mask, PoolingStrategy::Mean, true, None, false, false).unwrap();
  let mut n = mlxrs::ops::linalg_full::norm(&p, 2.0, &[-1], false).unwrap();
  let norms = n.to_vec::<f32>().unwrap();
  assert!(norms.iter().all(|&x| close(x, 1.0)), "rows must be unit");
}

// ───────────────── matryoshka dimension truncation ─────────────────

#[test]
fn truncate_last_dim_basic() {
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut t = truncate_last_dim(&x, 2).unwrap();
  assert_eq!(t.shape(), vec![2, 2]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 2.0, 4.0, 5.0]));
}

#[test]
fn truncate_last_dim_noop_when_ge_size() {
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut t = truncate_last_dim(&x, 5).unwrap();
  assert_eq!(t.shape(), vec![2, 2]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn dispatcher_matryoshka_truncation() {
  // (batch=1, seq=2, hidden=4); mean over seq then truncate to 2.
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0], &(1, 2, 4)).unwrap();
  let mask = Array::ones::<f32>(&(1, 2)).unwrap();
  let mut p = pool(
    &emb,
    &mask,
    PoolingStrategy::Mean,
    false,
    Some(2),
    false,
    false,
  )
  .unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  // mean = [2,3,4,5]; truncated to 2 = [2,3]
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[2.0, 3.0]));
}

// ───────────────── parameterized normalize (python base.py) ─────────────────

#[test]
fn normalize_l2_default() {
  let v = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut n = normalize(&v, 2.0, -1, true, DEFAULT_NORMALIZE_EPS).unwrap();
  assert!(vclose(&n.to_vec::<f32>().unwrap(), &[0.6, 0.8]));
}

#[test]
fn normalize_l1_p_ne_2() {
  // L1 norm of [3,4] = 7 -> [3/7, 4/7]
  let v = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut n = normalize(&v, 1.0, -1, true, DEFAULT_NORMALIZE_EPS).unwrap();
  assert!(vclose(&n.to_vec::<f32>().unwrap(), &[3.0 / 7.0, 4.0 / 7.0]));
}

#[test]
fn normalize_inf_norm() {
  // L-inf norm of [3,-4] = 4 -> [0.75, -1.0]
  let v = Array::from_slice(&[3.0_f32, -4.0], &(1, 2)).unwrap();
  let mut n = normalize(&v, f64::INFINITY, -1, true, DEFAULT_NORMALIZE_EPS).unwrap();
  assert!(vclose(&n.to_vec::<f32>().unwrap(), &[0.75, -1.0]));
}

#[test]
fn normalize_axis_0_keepdims() {
  // Normalize columns (axis 0). col0=[3,4] L2=5 -> [0.6,0.8]; col1=[0,0] -> 0
  let v = Array::from_slice(&[3.0_f32, 0.0, 4.0, 0.0], &(2, 2)).unwrap();
  let mut n = normalize(&v, 2.0, 0, true, DEFAULT_NORMALIZE_EPS).unwrap();
  assert!(vclose(&n.to_vec::<f32>().unwrap(), &[0.6, 0.0, 0.8, 0.0]));
}

#[test]
fn normalize_zero_vector_eps_floor_python_vs_swift() {
  // Zero vector: x / max(0, eps) = 0 either way; differing eps must
  // both keep the result finite (== 0). Documents the 1e-9 vs 1e-12
  // python/swift divergence.
  let z = Array::from_slice(&[0.0_f32, 0.0], &(1, 2)).unwrap();
  let mut py = l2_normalize_eps(&z, DEFAULT_NORMALIZE_EPS).unwrap();
  let mut sw = l2_normalize_eps(&z, SWIFT_L2_EPS).unwrap();
  assert!(vclose(&py.to_vec::<f32>().unwrap(), &[0.0, 0.0]));
  assert!(vclose(&sw.to_vec::<f32>().unwrap(), &[0.0, 0.0]));
  const { assert!(DEFAULT_NORMALIZE_EPS > SWIFT_L2_EPS) }; // 1e-9 > 1e-12
}

// ───────────────── fused post-pool norms (mlx-c) ─────────────────

#[test]
fn layer_norm_zero_mean_unit_var() {
  // LayerNorm over last dim of [1,2,3,4]: mean=2.5, normalized has
  // ~zero mean and ~unit variance.
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let mut ln = layer_norm(&x, None, None, 1e-5).unwrap();
  let v = ln.to_vec::<f32>().unwrap();
  let mean: f32 = v.iter().sum::<f32>() / 4.0;
  assert!(mean.abs() < 1e-3, "mean ~0, got {mean}");
  let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
  assert!((var - 1.0).abs() < 1e-2, "var ~1, got {var}");
}

#[test]
fn rms_norm_scales_by_rms() {
  // RMSNorm of [3,4]: rms = sqrt((9+16)/2) = sqrt(12.5); x/rms.
  let x = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut rn = rms_norm(&x, None, 1e-6).unwrap();
  let rms = (12.5_f32).sqrt();
  assert!(vclose(
    &rn.to_vec::<f32>().unwrap(),
    &[3.0 / rms, 4.0 / rms]
  ));
}

#[test]
fn dispatcher_apply_layer_norm_then_normalize() {
  let (emb, mask) = fixture();
  // mean-pool -> layer_norm -> l2-normalize; result rows unit norm.
  let p = pool(&emb, &mask, PoolingStrategy::Mean, true, None, true, false).unwrap();
  let mut n = mlxrs::ops::linalg_full::norm(&p, 2.0, &[-1], false).unwrap();
  assert!(n.to_vec::<f32>().unwrap().iter().all(|&x| close(x, 1.0)));
}

#[test]
fn dispatcher_apply_rms_norm_path() {
  let (emb, mask) = fixture();
  // rms-norm requested; layer_norm takes precedence only when both set,
  // here only rms is set so the rms path runs and shape is preserved.
  let p = pool(&emb, &mask, PoolingStrategy::Mean, false, None, false, true).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
}

#[test]
fn dispatcher_layer_norm_wins_over_rms_when_both_set() {
  let (emb, mask) = fixture();
  let mut both = pool(&emb, &mask, PoolingStrategy::Mean, false, None, true, true).unwrap();
  let mut just_ln = pool(&emb, &mask, PoolingStrategy::Mean, false, None, true, false).unwrap();
  assert!(vclose(
    &both.to_vec::<f32>().unwrap(),
    &just_ln.to_vec::<f32>().unwrap()
  ));
}

// ───────────────── ST-config parsing (python TestNormalizePoolingConfig) ─────

#[test]
fn st_config_modern_pooling_mode_key() {
  let cfg = pooling_from_st_config_str(r#"{"pooling_mode": "mean"}"#).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Mean);
  assert!(cfg.normalize());
  assert_eq!(cfg.dimension(), None);
}

#[test]
fn st_config_word_embedding_dimension_is_matryoshka_dim() {
  let cfg = pooling_from_st_config_str(
    r#"{"word_embedding_dimension": 384, "pooling_mode_cls_token": true}"#,
  )
  .unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Cls);
  assert_eq!(cfg.dimension(), Some(384));
}

#[test]
fn st_config_legacy_mean_only() {
  // python test_pooling_legacy_config_conversion: only mean flag true.
  let json = r#"{
    "embedding_dimension": 384,
    "pooling_mode_cls_token": false,
    "pooling_mode_mean_tokens": true,
    "pooling_mode_max_tokens": false,
    "pooling_mode_mean_sqrt_len_tokens": false,
    "pooling_mode_weightedmean_tokens": false,
    "pooling_mode_lasttoken": false,
    "include_prompt": true
  }"#;
  let cfg = pooling_from_st_config_bytes(json.as_bytes()).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Mean);
  assert_eq!(cfg.dimension(), Some(384));
}

#[test]
fn st_config_legacy_priority_cls_over_mean_over_max_over_last() {
  // CLS > Mean > Max > Last priority (swift Pooling(config:)). python's
  // _normalize_pooling_config would produce a ("cls","mean") tuple and
  // then pool_by_config would reject it; the task's stated priority rule
  // resolves multi-active to the highest-priority *supported* mode.
  let all_true = r#"{
    "pooling_mode_cls_token": true,
    "pooling_mode_mean_tokens": true,
    "pooling_mode_max_tokens": true,
    "pooling_mode_lasttoken": true
  }"#;
  assert_eq!(
    pooling_from_st_config_str(all_true).unwrap().strategy(),
    PoolingStrategy::Cls
  );

  let mean_max_last = r#"{
    "pooling_mode_cls_token": false,
    "pooling_mode_mean_tokens": true,
    "pooling_mode_max_tokens": true,
    "pooling_mode_lasttoken": true
  }"#;
  assert_eq!(
    pooling_from_st_config_str(mean_max_last)
      .unwrap()
      .strategy(),
    PoolingStrategy::Mean
  );

  let max_last = r#"{
    "pooling_mode_max_tokens": true,
    "pooling_mode_lasttoken": true
  }"#;
  assert_eq!(
    pooling_from_st_config_str(max_last).unwrap().strategy(),
    PoolingStrategy::Max
  );

  let last_only = r#"{"pooling_mode_lasttoken": true}"#;
  assert_eq!(
    pooling_from_st_config_str(last_only).unwrap().strategy(),
    PoolingStrategy::Last
  );
}

#[test]
fn st_config_legacy_all_false_defaults_to_mean() {
  // python _normalize_pooling_config: no active flag -> ("mean",).
  let json = r#"{
    "pooling_mode_cls_token": false,
    "pooling_mode_mean_tokens": false,
    "pooling_mode_max_tokens": false,
    "pooling_mode_lasttoken": false
  }"#;
  assert_eq!(
    pooling_from_st_config_str(json).unwrap().strategy(),
    PoolingStrategy::Mean
  );
}

#[test]
fn st_config_unsupported_mode_rejected() {
  assert!(pooling_from_st_config_str(r#"{"pooling_mode": "weightedmean"}"#).is_err());
  assert!(pooling_from_st_config_str(r#"{"pooling_mode_weightedmean_tokens": true}"#).is_err());
  assert!(pooling_from_st_config_str(r#"{"pooling_mode": "bogus"}"#).is_err());
}

#[test]
fn st_config_include_prompt_false_rejected() {
  // python pool_by_config raises for include_prompt=false (INSTRUCTOR).
  assert!(
    pooling_from_st_config_str(r#"{"pooling_mode": "mean", "include_prompt": false}"#).is_err()
  );
}

#[test]
fn st_config_concatenated_list_mode_rejected() {
  assert!(pooling_from_st_config_str(r#"{"pooling_mode": ["cls", "mean"]}"#).is_err());
}

#[test]
fn st_config_present_malformed_pooling_mode_rejected() {
  // C6 (Copilot review 4307622782, #3256688299): a present-but-non-
  // string/non-array `pooling_mode` (null / bool / number / object).
  // python `pool_by_config` does `mode = cfg["pooling_mode"]` and falls
  // through to `raise ValueError(f"Unknown pooling mode {mode!r}...")`
  // for such a value — it REJECTS, it does NOT silently fall back to
  // legacy/Mean. mlxrs previously fell through to the legacy path (silent
  // Mean), a divergence AND a silent-wrong-embedding. Must now be a
  // recoverable `Err(Error::Backend)`, NOT `Ok(Mean)`.
  for (json, what) in [
    (r#"{"pooling_mode": null}"#, "null"),
    (r#"{"pooling_mode": false}"#, "bool false"),
    (r#"{"pooling_mode": true}"#, "bool true"),
    (r#"{"pooling_mode": 2}"#, "number"),
    (r#"{"pooling_mode": 1.5}"#, "fractional number"),
    (r#"{"pooling_mode": {"a": 1}}"#, "object"),
  ] {
    let r = pooling_from_st_config_str(json);
    assert!(
      matches!(r, Err(Error::OutOfRange(_))),
      "present malformed pooling_mode ({what}) must be Err(OutOfRange), got {r:?}"
    );
    // Specifically must NOT silently resolve to a strategy (e.g. Mean).
    assert!(
      r.is_err(),
      "must not silently fall back to a strategy for {what}: {r:?}"
    );
  }

  // A present malformed `pooling_mode` is rejected EVEN when legacy flags
  // are also present (python leaves the present `pooling_mode` as-is and
  // `pool_by_config` rejects it; mlxrs must not let the legacy path mask
  // the malformed modern key).
  let r = pooling_from_st_config_str(r#"{"pooling_mode": null, "pooling_mode_mean_tokens": true}"#);
  assert!(
    matches!(r, Err(Error::OutOfRange(_))),
    "malformed pooling_mode alongside legacy flags must still be Err, got {r:?}"
  );
}

#[test]
fn st_config_present_invalid_dimension_rejected() {
  // C7 (Copilot review 4307622782, #3256688310): a present-but-invalid
  // `word_embedding_dimension`/`embedding_dimension` (negative,
  // fractional, non-numeric, > usize, or 0) previously went `as_u64()` →
  // `None` → treated as ABSENT → matryoshka truncation silently SKIPPED,
  // returning a full-width embedding the model author did not request — a
  // silent wrong embedding. python `mlx-embeddings` has NO matryoshka
  // truncation (no python reference), so per the standing "never silently
  // produce wrong embeddings" rule mlxrs rejects a present-but-invalid
  // dimension with a recoverable `Err` (intentional stricter-than-python
  // safety choice). Absent / valid dimensions are unchanged (re-pinned
  // elsewhere: `Some(384)`, `Some(1)`, `None`).
  for (json, what) in [
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": -1}"#,
      "negative",
    ),
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": 1.5}"#,
      "fractional",
    ),
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": "384"}"#,
      "string",
    ),
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": null}"#,
      "null",
    ),
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": false}"#,
      "bool",
    ),
    (
      r#"{"pooling_mode": "mean", "word_embedding_dimension": 0}"#,
      "zero (empty embedding)",
    ),
    (
      // > u64::MAX → serde_json cannot even hold it as an integer, so it
      // is a float → `as_u64()` None → rejected.
      r#"{"pooling_mode": "mean", "word_embedding_dimension": 99999999999999999999999999}"#,
      "overflow > usize",
    ),
    // Same for the legacy `embedding_dimension` alias.
    (
      r#"{"pooling_mode": "mean", "embedding_dimension": -5}"#,
      "negative (embedding_dimension alias)",
    ),
  ] {
    let r = pooling_from_st_config_str(json);
    // Invalid-dimension errors split between Parse (for the negative /
    // fractional / out-of-range scanner-detected forms, which carry the
    // byte-offset diagnostic) and OutOfRange (for the value-after-parse
    // forms — non-number / zero). Both are non-recoverable typed Errs;
    // accept either, mirroring the loose "any Err" contract the test
    // originally enforced via the deprecated Backend variant.
    assert!(
      matches!(r, Err(Error::Parse(_)) | Err(Error::OutOfRange(_))),
      "present invalid dimension ({what}) must be Err(Parse) or Err(OutOfRange), got {r:?}"
    );
  }

  // `word_embedding_dimension` precedence: a present-but-invalid
  // `word_embedding_dimension` is rejected and does NOT silently fall
  // back to a valid `embedding_dimension` (matches the > precedence).
  let r = pooling_from_st_config_str(
    r#"{"pooling_mode": "mean", "word_embedding_dimension": -1, "embedding_dimension": 384}"#,
  );
  assert!(
    matches!(r, Err(Error::Parse(_)) | Err(Error::OutOfRange(_))),
    "invalid primary key must reject, not fall back to the alias, got {r:?}"
  );

  // Absent + valid dimensions remain unchanged (regression guard).
  assert_eq!(
    pooling_from_st_config_str(r#"{"pooling_mode": "mean"}"#)
      .unwrap()
      .dimension(),
    None
  );
  assert_eq!(
    pooling_from_st_config_str(r#"{"pooling_mode": "mean", "word_embedding_dimension": 256}"#)
      .unwrap()
      .dimension(),
    Some(256)
  );
}

#[test]
fn st_config_end_to_end_drives_dispatcher() {
  let (emb, mask) = fixture();
  let cfg = pooling_from_st_config_str(
    r#"{"pooling_mode_max_tokens": true, "word_embedding_dimension": 1}"#,
  )
  .unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Max);
  let mut p = pool(
    &emb,
    &mask,
    cfg.strategy(),
    cfg.normalize(),
    cfg.dimension(),
    false,
    false,
  )
  .unwrap();
  // max = [[5,6],[70,80]] then truncate to dim 1 -> [[5],[70]], then
  // normalize (single element rows) -> sign-preserving unit -> [[1],[1]].
  assert_eq!(p.shape(), vec![2, 1]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 1.0]));
}

// ───────────────── PoolingStrategy::from_mode parsing ─────────────────

#[test]
fn pooling_strategy_from_mode() {
  assert_eq!(
    PoolingStrategy::from_mode("cls").unwrap(),
    PoolingStrategy::Cls
  );
  assert_eq!(
    PoolingStrategy::from_mode("lasttoken").unwrap(),
    PoolingStrategy::Last
  );
  assert_eq!(
    PoolingStrategy::from_mode("max").unwrap(),
    PoolingStrategy::Max
  );
  assert_eq!(
    PoolingStrategy::from_mode("mean").unwrap(),
    PoolingStrategy::Mean
  );
  assert_eq!(
    PoolingStrategy::from_mode("first").unwrap(),
    PoolingStrategy::First
  );
  assert_eq!(
    PoolingStrategy::from_mode("none").unwrap(),
    PoolingStrategy::None
  );
  assert!(PoolingStrategy::from_mode("weightedmean").is_err());
  assert!(PoolingStrategy::from_mode("xyzzy").is_err());
}

// ───────────── F1: config-driven CLS must be mask-aware ─────────────
//
// Codex round-1 [high]: the dispatcher routed `PoolingStrategy::Cls` to
// `first_token_pooling` (strict token-0, ignores the mask), so a
// LEFT-PADDED batch under `Cls` (incl. via ST config) silently embedded
// the pad token. python `mlx-embeddings` `pool_by_config` mode `"cls"`
// (and the ST `pooling_mode_cls_token` resolution) → `cls_pooling`,
// which is mask-aware: `argmax(attention_mask, axis=1)` selects the
// first *real* token. `Cls` now == mask-aware `cls_pooling`; `First`
// stays strict token-0 (swift `.first`).

// Left-padded fixture: seq0 has 2 pad then 2 real; seq1 has 1 pad then 3
// real. python `cls_pooling`: argmax(mask) → first real index.
//   seq0 mask [0,0,1,1] → idx 2 → row [3,3]
//   seq1 mask [0,1,1,1] → idx 1 → row [200,200]
fn left_padded_fixture() -> (Array, Array) {
  let emb = Array::from_slice(
    &[
      0.0_f32, 0.0, 9.0, 9.0, 3.0, 3.0, 4.0, 4.0, // seq 0 (pad,pad,real,real)
      0.0, 0.0, 200.0, 200.0, 300.0, 300.0, 400.0, 400.0, // seq 1 (pad,real,real,real)
    ],
    &(2, 4, 2),
  )
  .unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], &(2, 4)).unwrap();
  (emb, mask)
}

#[test]
fn cls_dispatcher_is_mask_aware_on_left_padded_batch() {
  let (emb, mask) = left_padded_fixture();

  // Cls (mask-aware) → first *real* token, NOT position 0.
  let mut cls = pool(&emb, &mask, PoolingStrategy::Cls, false, None, false, false).unwrap();
  assert_eq!(cls.shape(), vec![2, 2]);
  // Derived from py cls_pooling: argmax([0,0,1,1])=2 → [3,3];
  // argmax([0,1,1,1])=1 → [200,200].
  assert!(
    vclose(&cls.to_vec::<f32>().unwrap(), &[3.0, 3.0, 200.0, 200.0]),
    "Cls must select first real token (py cls_pooling), not pos-0"
  );

  // First (strict token-0) → the pad rows (position 0), unchanged.
  let mut first = pool(
    &emb,
    &mask,
    PoolingStrategy::First,
    false,
    None,
    false,
    false,
  )
  .unwrap();
  assert!(
    vclose(&first.to_vec::<f32>().unwrap(), &[0.0, 0.0, 0.0, 0.0]),
    "First must stay strict token-0 (swift .first)"
  );

  // Dispatcher Cls must agree with the standalone cls_pooling fn.
  let mut direct = cls_pooling(&emb, &mask).unwrap();
  assert_eq!(
    direct.to_vec::<f32>().unwrap(),
    cls.to_vec::<f32>().unwrap()
  );
}

#[test]
fn st_config_resolved_cls_drives_mask_aware_dispatcher() {
  let (emb, mask) = left_padded_fixture();

  // Modern key and legacy boolean flag both resolve to mask-aware Cls.
  for json in [
    r#"{"pooling_mode": "cls"}"#,
    r#"{"pooling_mode_cls_token": true}"#,
  ] {
    let cfg = pooling_from_st_config_str(json).unwrap();
    assert_eq!(
      cfg.strategy(),
      PoolingStrategy::Cls,
      "ST CLS key must map to Cls (mask-aware), not First: {json}"
    );
    let mut p = pool(&emb, &mask, cfg.strategy(), false, None, false, false).unwrap();
    assert!(
      vclose(&p.to_vec::<f32>().unwrap(), &[3.0, 3.0, 200.0, 200.0]),
      "ST-config CLS must select first real token (py cls_pooling): {json}"
    );
  }

  // from_mode("cls") is mask-aware Cls too (not First / token-0).
  assert_eq!(
    PoolingStrategy::from_mode("cls").unwrap(),
    PoolingStrategy::Cls
  );
}

// Right-padded fixture sanity: when token-0 IS the first real token
// (no left padding) Cls and First coincide — preserves the existing
// `dispatcher_every_strategy_shapes_and_values` expected value
// ([1,2,10,20]) so that back-compat test is unchanged.
#[test]
fn cls_and_first_coincide_when_no_left_padding() {
  let (emb, mask) = fixture(); // mask [1,1,1,0 | 1,1,1,1] — token-0 real
  let mut cls = pool(&emb, &mask, PoolingStrategy::Cls, false, None, false, false).unwrap();
  let mut first = pool(
    &emb,
    &mask,
    PoolingStrategy::First,
    false,
    None,
    false,
    false,
  )
  .unwrap();
  assert_eq!(cls.to_vec::<f32>().unwrap(), first.to_vec::<f32>().unwrap());
  assert!(vclose(
    &cls.to_vec::<f32>().unwrap(),
    &[1.0, 2.0, 10.0, 20.0]
  ));
}

// ───────────── F2: bound the ST config read (OOM guard) ─────────────
//
// Codex round-1 [medium]: `pooling_from_st_config_path` did a raw
// `std::fs::read` on an untrusted model dir → unbounded allocation. Now
// it stats first and rejects > 1 MiB with a recoverable Error::Backend
// (no OOM/panic). A normal small config still parses.

#[test]
fn st_config_path_rejects_oversize_file_without_oom() {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-q20-oversize-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  let pooling_dir = dir.join("1_Pooling");
  std::fs::create_dir_all(&pooling_dir).unwrap();
  let path = pooling_dir.join("config.json");

  // 2 MiB > 1 MiB cap. Valid JSON prefix so a (buggy) read+parse would
  // otherwise succeed — the guard must reject on size alone, pre-read.
  let mut blob = String::from(r#"{"pooling_mode": "mean", "_pad": ""#);
  blob.push_str(&"A".repeat(2 * 1024 * 1024));
  blob.push_str(r#""}"#);
  std::fs::write(&path, &blob).unwrap();

  let r = pooling_from_st_config_path(&dir);
  assert!(
    matches!(r, Err(Error::CapExceeded(_))),
    "oversize config must yield Err(CapExceeded), got {r:?}"
  );

  // A small valid config in the same layout still parses fine.
  std::fs::write(&path, r#"{"pooling_mode": "cls"}"#).unwrap();
  let cfg = pooling_from_st_config_path(&dir).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Cls);

  std::fs::remove_dir_all(&dir).ok();
}

// Codex round-2 [medium] completeness: the round-1 stat-then-read was
// TOCTOU/non-regular-file bypassable (FIFO/device/symlink report len 0,
// then `fs::read` streams unbounded). The path now opens ONCE, rejects a
// non-regular file from the opened handle's metadata, and reads via
// `take(cap+1)` so the allocation is hard-bounded regardless. A FIFO is
// not portably creatable in std on macOS without extra deps, so the
// non-regular case is exercised with a *directory* at the config.json
// location: `File::open` on a directory succeeds on Unix, but
// `metadata().is_file()` is false → the non-regular rejection fires
// (deterministic, dependency-free, portable).
#[test]
fn st_config_path_rejects_non_regular_file_without_hang() {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-q20-nonreg-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  // Make `<dir>/1_Pooling/config.json` itself a *directory* (non-regular).
  let cfg_as_dir = dir.join("1_Pooling").join("config.json");
  std::fs::create_dir_all(&cfg_as_dir).unwrap();

  let r = pooling_from_st_config_path(&dir);
  assert!(
    matches!(r, Err(Error::FileIo(_))),
    "non-regular (directory) config path must yield a recoverable \
     Err(FileIo) without hang/panic, got {r:?}"
  );

  // Replace the directory with a normal small config: still parses.
  std::fs::remove_dir_all(&cfg_as_dir).unwrap();
  std::fs::write(&cfg_as_dir, r#"{"pooling_mode": "max"}"#).unwrap();
  let cfg = pooling_from_st_config_path(&dir).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Max);

  std::fs::remove_dir_all(&dir).ok();
}

// An exactly-at-cap regular config still parses (boundary: `take` reads
// `cap+1`, the > comparison must NOT reject a file of exactly `cap`
// bytes). Uses a valid-JSON-with-padding body padded to the cap.
#[test]
fn st_config_path_accepts_file_at_exact_cap() {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-q20-atcap-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  let pooling_dir = dir.join("1_Pooling");
  std::fs::create_dir_all(&pooling_dir).unwrap();
  let path = pooling_dir.join("config.json");

  let prefix = r#"{"pooling_mode": "mean", "_pad": ""#;
  let suffix = r#""}"#;
  let cap = 1usize << 20;
  let pad = cap - prefix.len() - suffix.len();
  let mut blob = String::with_capacity(cap);
  blob.push_str(prefix);
  blob.push_str(&"A".repeat(pad));
  blob.push_str(suffix);
  assert_eq!(blob.len(), cap, "blob must be exactly the cap");
  std::fs::write(&path, &blob).unwrap();

  let cfg = pooling_from_st_config_path(&dir).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Mean);

  std::fs::remove_dir_all(&dir).ok();
}

// Codex round-3 [medium]: a *directory* at `config.json` (round-2 test)
// makes `open()` return immediately, so it never exercised the one
// non-regular file whose blocking `open()` HANGS: a FIFO. On Unix a
// read-only blocking `open()` of a writer-less FIFO blocks forever —
// before the `is_file()` rejection can run. The fix opens with
// `O_NONBLOCK`, so the open returns at once and the
// pre-read `is_file()` check rejects it. This test plants a real FIFO
// (`libc::mkfifo`, no writer) at the config path and asserts the call
// returns `Err` *promptly without hanging*.
//
// Determinism / non-flakiness: the call is run on a worker thread and
// joined with a generous 30 s budget. With the fix the open is
// instantaneous (sub-millisecond), so the budget is never approached;
// if the O_NONBLOCK fix regresses, the blocking `open()` hangs forever
// and the budget elapses → the test FAILS (loud) instead of wedging the
// whole suite. The thread is left detached on the (regression-only)
// timeout path rather than joined, so a regression cannot hang CI.
#[cfg(unix)]
#[test]
fn st_config_path_fifo_returns_err_without_hang() {
  use std::sync::mpsc;

  let dir = std::env::temp_dir().join(format!(
    "mlxrs-q20-fifo-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  let pooling_dir = dir.join("1_Pooling");
  std::fs::create_dir_all(&pooling_dir).unwrap();
  let path = pooling_dir.join("config.json");

  // Create a FIFO at the config path with NO writer ever opened. A
  // read-only blocking `open()` of this would block indefinitely.
  use std::os::unix::ffi::OsStrExt;
  let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
  // SAFETY: `c_path` is a valid NUL-terminated C string that outlives
  // the call; `mkfifo` only reads it and creates a filesystem node.
  let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
  assert_eq!(rc, 0, "mkfifo failed (errno-based rc {rc})");

  // Run on a worker thread so a *regression* (blocking open hang) fails
  // the test via the join budget instead of wedging the whole suite.
  let probe_dir = dir.clone();
  let (tx, rx) = mpsc::channel();
  let handle = std::thread::spawn(move || {
    let r = pooling_from_st_config_path(&probe_dir);
    let _ = tx.send(matches!(r, Err(Error::FileIo(_))));
  });

  match rx.recv_timeout(std::time::Duration::from_secs(30)) {
    Ok(is_recoverable_err) => {
      handle.join().unwrap();
      assert!(
        is_recoverable_err,
        "FIFO at config.json must yield a recoverable Err(FileIo) \
         (rejected by is_file()==false), not Ok"
      );
    }
    Err(_) => {
      // Regression: the O_NONBLOCK open was lost and the blocking
      // `open()` is wedged on the writer-less FIFO. Don't join (would
      // hang CI) — fail loudly. The detached thread dies with the
      // process.
      std::fs::remove_dir_all(&dir).ok();
      panic!(
        "pooling_from_st_config_path HUNG on a writer-less FIFO at \
         config.json — the O_NONBLOCK open regressed"
      );
    }
  }

  // A normal small config replacing the FIFO still parses fine.
  std::fs::remove_file(&path).unwrap();
  std::fs::write(&path, r#"{"pooling_mode": "last"}"#).unwrap();
  let cfg = pooling_from_st_config_path(&dir).unwrap();
  assert_eq!(cfg.strategy(), PoolingStrategy::Last);

  std::fs::remove_dir_all(&dir).ok();
}

// Codex round-6 [high]: round-3 added `O_NOFOLLOW` to the open flags,
// which makes `open()` fail (ELOOP) on a symlink at `config.json`. But
// HuggingFace Hub caches store `.../snapshots/<rev>/1_Pooling/config.json`
// as a *symlink into `.../blobs/<hash>`* — the dominant real cached-model
// layout — so `O_NOFOLLOW` broke `pooling_from_st_config_path` for a
// normal cached model (caller silently fell back to the wrong pooling
// strategy/matryoshka dim → wrong embeddings). The fix removes
// `O_NOFOLLOW` (keeping `O_NONBLOCK | O_CLOEXEC`); safety is preserved by
// fstat-of-opened-target (`is_file()` on the *resolved* target rejects
// symlink→FIFO/device/dir) + non-blocking open + capped read, NOT by
// refusing symlinks. This test reproduces the HF cache layout: a real
// regular blob file at one path, with `1_Pooling/config.json` a symlink
// into it; the call must follow the symlink and parse the declared
// config (strategy/normalize/dimension) — NOT return `Err`.
#[cfg(unix)]
#[test]
fn st_config_path_follows_symlink_to_regular_file() {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-q20-symlink-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  // Mirror the HF Hub cache layout: a real regular blob file elsewhere
  // in the tree, and `<dir>/1_Pooling/config.json` a symlink into it.
  let blobs_dir = dir.join("blobs");
  std::fs::create_dir_all(&blobs_dir).unwrap();
  let blob = blobs_dir.join("deadbeefcafef00d");
  std::fs::write(
    &blob,
    r#"{"pooling_mode": "cls", "word_embedding_dimension": 384}"#,
  )
  .unwrap();

  let pooling_dir = dir.join("1_Pooling");
  std::fs::create_dir_all(&pooling_dir).unwrap();
  let cfg_path = pooling_dir.join("config.json");
  std::os::unix::fs::symlink(&blob, &cfg_path).unwrap();

  // The symlink must be followed and the resolved regular file parsed,
  // returning the declared config — NOT an ELOOP/`O_NOFOLLOW` Err.
  let cfg = pooling_from_st_config_path(&dir).expect(
    "HF-cache symlink → regular config.json must be followed and parsed, \
     not rejected (O_NOFOLLOW regressed)",
  );
  assert_eq!(cfg.strategy(), PoolingStrategy::Cls);
  assert!(cfg.normalize());
  assert_eq!(cfg.dimension(), Some(384));

  std::fs::remove_dir_all(&dir).ok();
}

// ───────────── F3: validate rank before indexing shape ─────────────
//
// Codex round-1 [medium]: pooling helpers indexed shape[0]/shape[2]
// (and mask shape) before validating rank — a 1-D/2-D token_embeddings
// or wrong-rank mask panicked a safe public API. Each public helper now
// validates rank-3 token_embeddings + rank-2 mask up front, returning
// Err(RankMismatch) instead of panicking.

#[test]
fn pooling_helpers_reject_non_rank3_token_embeddings_without_panic() {
  let mask = Array::from_slice(&[1.0_f32, 1.0], &(1, 2)).unwrap();

  // 1-D token_embeddings.
  let emb_1d = Array::from_slice(&[1.0_f32, 2.0], &(2,)).unwrap();
  // 2-D token_embeddings.
  let emb_2d = Array::from_slice(&[1.0_f32, 2.0], &(1, 2)).unwrap();

  for emb in [&emb_1d, &emb_2d] {
    // The rank-3 token_embeddings guard now produces a typed `RankMismatch`.
    assert!(matches!(
      mean_pooling(emb, &mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      max_pooling(emb, &mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      cls_pooling(emb, &mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      last_token_pooling(emb, &mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      first_token_pooling(emb),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      pool(emb, &mask, PoolingStrategy::Mean, false, None, false, false),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      pool(emb, &mask, PoolingStrategy::Cls, false, None, false, false),
      Err(Error::RankMismatch(_))
    ));
  }
}

#[test]
fn pooling_helpers_reject_wrong_rank_mask_without_panic() {
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();

  // 1-D mask (should be rank-2 (batch, seq_len)).
  let mask_1d = Array::from_slice(&[1.0_f32, 1.0], &(2,)).unwrap();
  // 3-D mask.
  let mask_3d = Array::from_slice(&[1.0_f32, 1.0], &(1, 2, 1)).unwrap();

  for mask in [&mask_1d, &mask_3d] {
    // Wrong-rank mask → typed `RankMismatch`.
    assert!(matches!(
      mean_pooling(&emb, mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      max_pooling(&emb, mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      cls_pooling(&emb, mask),
      Err(Error::RankMismatch(_))
    ));
    assert!(matches!(
      last_token_pooling(&emb, mask),
      Err(Error::RankMismatch(_))
    ));
  }
}

#[test]
fn pooling_helpers_reject_mismatched_batch_or_seq_dims() {
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  // mask seq_len 3 != emb seq_len 2.
  let bad_mask = Array::from_slice(&[1.0_f32, 1.0, 1.0], &(1, 3)).unwrap();
  // (batch, seq_len) shape mismatch between emb + mask → ShapePairMismatch.
  assert!(matches!(
    mean_pooling(&emb, &bad_mask),
    Err(Error::ShapePairMismatch(_))
  ));
  assert!(matches!(
    cls_pooling(&emb, &bad_mask),
    Err(Error::ShapePairMismatch(_))
  ));
}

// ════════════════ Codex round-4: f16 / bf16 dtype fidelity ════════════════
//
// SYSTEMIC dtype bug guard. Before the fix, f32 constant `Array`s
// (`eps`/`-inf`/`0` floors, and `max_pooling`'s f32 mask cast) force-
// upcast a f16/bf16 embedding tensor to f32 via MLX type promotion —
// diverging from python `mlx-embeddings`, which casts the mask to
// `token_embeddings.dtype` (`astype`) and lets python scalars act as MLX
// *weak* scalars that adopt the array dtype. f32-only tests masked this;
// real embedding models commonly run f16/bf16.
//
// Each test asserts BOTH:
//   (a) OUTPUT dtype == INPUT dtype (no silent f32 upcast), and
//   (b) values match the python-reference computation done IN that dtype.
//
// `mean_pooling` is the documented exception: python
// `mean_pooling` explicitly does `input_mask_expanded.astype(mx.float32)`
// (pooling.py L10), so its output is f32 *by python design* regardless of
// input dtype — the test asserts F32 + the f32 value (parity, not a bug).
//
// Tolerance approach: the fixtures use only small integers (1..=80) and
// exact binary fractions, which are bit-exact in BOTH f16 (≤2048 int) and
// bf16 (≤256 int). Gather/select/max paths (cls/max/last/first) are then
// EXACT in-dtype — asserted with a 0-tolerance per-element compare on the
// values read back as f32 (the f16/bf16→f32 widening of an exactly-
// representable value is itself lossless). Paths with a divide
// (normalize / cosine matrix) carry genuine half-precision rounding —
// asserted against the same op computed by the crate at f32 then rounded
// to the target dtype (`f32→half→f32`), i.e. compared at the dtype's own
// ULP via `half_close`, the rigorous non-flaky bound.

// f16/bf16 quantization round-trip of an f32 value (one ULP-grid snap).
fn to_f16_f32(v: f32) -> f32 {
  half::f16::from_f32(v).to_f32()
}
fn to_bf16_f32(v: f32) -> f32 {
  half::bf16::from_f32(v).to_f32()
}

// Tolerance reflecting one rounding at the *output* dtype's precision:
// f16 has 10 mantissa bits (rel ~2^-10), bf16 has 7 (rel ~2^-7). The
// bound is relative-scaled to the magnitude plus a small absolute floor;
// derived from the dtype, NOT hand-tuned, so it is non-flaky.
fn half_close(dt: Dtype, got: f32, want: f32) -> bool {
  let rel = match dt {
    Dtype::F16 => 1.0 / 1024.0, // 2^-10
    Dtype::BF16 => 1.0 / 128.0, // 2^-7
    _ => TOL,
  };
  let tol = rel * want.abs().max(1.0) * 4.0; // 4 ULP headroom for chained ops
  (got - want).abs() <= tol
}

// Build the standard python `test_pooling.py` fixture in `dt`
// (token_embeddings) — the mask stays f32 (python passes an int/float
// mask; the helpers `astype` it internally, exactly as python does).
fn fixture_dt(dt: Dtype) -> (Array, Array) {
  let (emb_f32, mask) = fixture();
  (emb_f32.astype(dt).unwrap(), mask)
}

fn assert_dtype(a: &Array, want: Dtype, ctx: &str) {
  assert_eq!(a.dtype().unwrap(), want, "output dtype for {ctx}");
}

// ---- gather/select/max paths: dtype preserved AND values bit-exact ----

#[test]
fn max_pooling_f16_bf16_preserve_dtype_and_values() {
  // python max_pooling: mask.astype(token_embeddings.dtype);
  // where(mask==0, -inf, emb); max(axis=1). Output dtype == emb dtype.
  // Fixture (seq0: 3 real +1 pad, seq1: 4 real) → [5,6, 70,80] exactly.
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, mask) = fixture_dt(dt);
    let mut p = max_pooling(&emb, &mask).unwrap();
    assert_dtype(&p, dt, "max_pooling");
    let v = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    assert_eq!(v, vec![5.0, 6.0, 70.0, 80.0], "max_pooling {dt:?}");
  }
}

#[test]
fn cls_pooling_f16_bf16_preserve_dtype_and_values() {
  // pure gather (argmax mask -> take_along_axis): exact, dtype preserved.
  // fixture mask row0=[1,1,1,0] argmax=0 -> [1,2]; row1 all1 -> [10,20].
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, mask) = fixture_dt(dt);
    let mut p = cls_pooling(&emb, &mask).unwrap();
    assert_dtype(&p, dt, "cls_pooling");
    let v = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    assert_eq!(v, vec![1.0, 2.0, 10.0, 20.0], "cls_pooling {dt:?}");
  }
}

#[test]
fn first_token_pooling_f16_bf16_preserve_dtype_and_values() {
  // strict token-0 gather: exact, dtype preserved. fixture -> [1,2,10,20].
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, _mask) = fixture_dt(dt);
    let mut p = first_token_pooling(&emb).unwrap();
    assert_dtype(&p, dt, "first_token_pooling");
    let v = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    assert_eq!(v, vec![1.0, 2.0, 10.0, 20.0], "first_token_pooling {dt:?}");
  }
}

#[test]
fn last_token_pooling_f16_bf16_preserve_dtype_and_values() {
  // python lasttoken: mask.astype(emb.dtype); gather (emb*mask) at last
  // real idx. fixture row0 last real idx=2 ->[5,6], row1 idx=3 ->[70,80].
  // emb*mask is in-dtype; values exact (1*1, integers).
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, mask) = fixture_dt(dt);
    let mut p = last_token_pooling(&emb, &mask).unwrap();
    assert_dtype(&p, dt, "last_token_pooling");
    let v = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    assert_eq!(v, vec![5.0, 6.0, 70.0, 80.0], "last_token_pooling {dt:?}");
  }
}

// ---- mean_pooling: python forces f32 (pooling.py L10) — PARITY ----

#[test]
fn mean_pooling_f16_bf16_matches_python_f32_upcast() {
  // python mean_pooling does input_mask_expanded.astype(mx.float32), so
  // the output is F32 by python design even for a f16/bf16 input. This
  // asserts that exact parity (NOT a dtype-preservation requirement) and
  // that the value equals the f32 fixture mean.
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, mask) = fixture_dt(dt);
    let mut p = mean_pooling(&emb, &mask).unwrap();
    assert_dtype(&p, Dtype::F32, "mean_pooling (python forces f32)");
    assert!(
      vclose(&p.to_vec::<f32>().unwrap(), &[3.0, 4.0, 40.0, 50.0]),
      "mean_pooling value {dt:?}"
    );
  }
}

// ---- normalize / l2_normalize: dtype preserved, in-dtype value ----

#[test]
fn normalize_l2_f16_bf16_preserve_dtype_and_value() {
  // python normalize_embeddings: x / maximum(norm(x), eps); eps is a weak
  // scalar adopting x.dtype, output dtype == x.dtype. Expected = the same
  // op done at f32 then snapped to the target dtype (its own ULP grid).
  let base = [3.0_f32, 4.0, 0.0, 12.0]; // ||(3,4)||=5, ||(0,12)||=12
  let x_f32 = Array::from_slice(&base, &(2, 2)).unwrap();
  let mut ref_f32 = l2_normalize(&x_f32).unwrap();
  let exp_f32 = ref_f32.to_vec::<f32>().unwrap(); // [0.6,0.8, 0,1]

  for dt in [Dtype::F16, Dtype::BF16] {
    let x = Array::from_slice(&base, &(2, 2))
      .unwrap()
      .astype(dt)
      .unwrap();
    let mut p = l2_normalize(&x).unwrap();
    assert_dtype(&p, dt, "l2_normalize");
    let got = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    for (g, w) in got.iter().zip(&exp_f32) {
      let want = if dt == Dtype::F16 {
        to_f16_f32(*w)
      } else {
        to_bf16_f32(*w)
      };
      assert!(
        half_close(dt, *g, want),
        "l2_normalize {dt:?}: got {g} want ~{want}"
      );
    }
  }
}

#[test]
fn normalize_param_p_f16_bf16_preserve_dtype() {
  // parameterized normalize (p=1, L1) must also keep input dtype: the eps
  // floor adopts x.dtype (weak scalar), divide stays in-dtype.
  let base = [1.0_f32, 1.0, 2.0, 2.0];
  for dt in [Dtype::F16, Dtype::BF16] {
    let x = Array::from_slice(&base, &(2, 2))
      .unwrap()
      .astype(dt)
      .unwrap();
    let p = normalize(&x, 1.0, -1, true, DEFAULT_NORMALIZE_EPS).unwrap();
    assert_dtype(&p, dt, "normalize p=1");
    let p2 = normalize(&x, 2.0, -1, true, SWIFT_L2_EPS).unwrap();
    assert_dtype(&p2, dt, "normalize p=2 swift-eps");
  }
}

#[test]
fn normalize_zero_vector_f16_bf16_eps_floor_in_dtype() {
  // all-zero row: norm=0, clamped by eps; 0/eps = 0. The eps floor adopts
  // x.dtype (MLX weak-scalar / python `mx.maximum(norm, eps)`, swift
  // `Float.asMLXArray(dtype:)` → MLXArray(eps, dtype: x.dtype)).
  //
  // CRITICAL half-precision fidelity point: the python/crate DEFAULT eps
  // (`1e-9`) interacts with the *exponent range* of the weak-scalar's
  // adopted dtype (verified against mlx-swift `DType.swift`
  // `Float32.asMLXArray`: the scalar is materialized IN the array dtype,
  // no higher-precision retention):
  //   - f16 has 5 exponent bits (min subnormal ~6e-8); `1e-9` underflows
  //     to 0.0, so `0 / max(0, 0)` = NaN.
  //   - bf16 has the SAME 8 exponent bits as f32 (min normal ~1.2e-38);
  //     `1e-9` is representable, so `0 / max(0, 1e-9)` = 0.0.
  // Both are the *faithful* MLX/python result for that dtype — NOT a
  // regression. This test uses a half-representable eps (`1e-2`) to prove
  // the dtype-preserving `0/eps == 0` floor engages, then asserts the
  // exact per-dtype faithful default-eps behavior so the underflow
  // boundary is documented, not silently "fixed".
  for dt in [Dtype::F16, Dtype::BF16] {
    let x = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(1, 3))
      .unwrap()
      .astype(dt)
      .unwrap();

    // representable eps: floor engages, 0/eps = 0, dtype preserved.
    let mut p = l2_normalize_eps(&x, 1e-2).unwrap();
    assert_dtype(&p, dt, "l2_normalize zero-vector (eps 1e-2)");
    let got = match dt {
      Dtype::F16 => p
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => p
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    assert_eq!(got, vec![0.0, 0.0, 0.0], "zero-vector eps 1e-2 {dt:?}");

    // default eps 1e-9: dtype-dependent faithful result. Dtype is STILL
    // preserved either way (the bug class under test).
    let mut q = l2_normalize(&x).unwrap();
    assert_dtype(&q, dt, "l2_normalize zero-vector (default eps 1e-9)");
    let qv = match dt {
      Dtype::F16 => q
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => q
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    match dt {
      // f16: 1e-9 underflows → 0/0 = NaN (python/MLX faithful).
      Dtype::F16 => assert!(
        qv.iter().all(|v| v.is_nan()),
        "default-eps zero-vector F16 is python-faithful NaN (1e-9 underflows in f16), got {qv:?}"
      ),
      // bf16: 1e-9 representable (f32 exponent range) → 0/1e-9 = 0.0.
      _ => assert_eq!(
        qv,
        vec![0.0, 0.0, 0.0],
        "default-eps zero-vector BF16 is 0.0 (1e-9 representable in bf16)"
      ),
    }
  }
}

// ---- pool() dispatcher with normalize=true ----

#[test]
fn dispatcher_normalize_f16_bf16_preserve_dtype() {
  // pool(strategy, normalize=true): max/cls/last/first keep emb dtype
  // through the (in-dtype) normalize. (Mean is python-f32 by design — see
  // mean_pooling_f16_bf16_matches_python_f32_upcast — so it is excluded
  // from the dtype-preservation set here on purpose.)
  for dt in [Dtype::F16, Dtype::BF16] {
    let (emb, mask) = fixture_dt(dt);
    for strat in [
      PoolingStrategy::Max,
      PoolingStrategy::Cls,
      PoolingStrategy::Last,
      PoolingStrategy::First,
    ] {
      let mut p = pool(&emb, &mask, strat, true, None, false, false).unwrap();
      assert_dtype(&p, dt, &format!("pool {strat:?} normalize=true"));
      // each pooled-then-L2 row is a unit vector (within dtype ULP).
      let shape = p.shape();
      let cols = shape[1];
      let got = match dt {
        Dtype::F16 => p
          .to_vec::<half::f16>()
          .unwrap()
          .iter()
          .map(|x| x.to_f32())
          .collect::<Vec<_>>(),
        _ => p
          .to_vec::<half::bf16>()
          .unwrap()
          .iter()
          .map(|x| x.to_f32())
          .collect::<Vec<_>>(),
      };
      for row in got.chunks(cols) {
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
          half_close(dt, n, 1.0),
          "unit norm {strat:?} {dt:?}: |row|={n}"
        );
      }
    }
  }
}

#[test]
fn dispatcher_mean_normalize_f16_is_f32_python_parity() {
  // pool(Mean, normalize=true) on f16: mean_pooling upcasts to f32
  // (python), normalize keeps that f32 -> result is F32 (parity, not a
  // regression). Guards that the dispatcher doesn't accidentally "fix"
  // python's documented f32 mean behavior.
  let (emb, mask) = fixture_dt(Dtype::F16);
  let mut p = pool(&emb, &mask, PoolingStrategy::Mean, true, None, false, false).unwrap();
  assert_dtype(&p, Dtype::F32, "pool Mean normalize=true (python f32)");
  // unit rows of f32 mean [3,4]/5 and [40,50]/~64.03.
  let v = p.to_vec::<f32>().unwrap();
  for row in v.chunks(2) {
    let n = (row[0] * row[0] + row[1] * row[1]).sqrt();
    assert!(close(n, 1.0), "unit norm mean f32: {n}");
  }
}

// ---- cosine_similarity_matrix: dtype preserved ----

#[test]
fn cosine_similarity_matrix_f16_bf16_preserve_dtype() {
  // l2_normalize (in-dtype) then normalized @ normalized.T -> dtype
  // preserved; diagonal ~1 within the dtype ULP.
  let base = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 rows, dim 2
  for dt in [Dtype::F16, Dtype::BF16] {
    let x = Array::from_slice(&base, &(3, 2))
      .unwrap()
      .astype(dt)
      .unwrap();
    let mut m = cosine_similarity_matrix(&x).unwrap();
    assert_dtype(&m, dt, "cosine_similarity_matrix");
    assert_eq!(m.shape(), vec![3, 3]);
    let got = match dt {
      Dtype::F16 => m
        .to_vec::<half::f16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
      _ => m
        .to_vec::<half::bf16>()
        .unwrap()
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<_>>(),
    };
    for i in 0..3 {
      assert!(
        half_close(dt, got[i * 3 + i], 1.0),
        "diag[{i}] {dt:?} = {}",
        got[i * 3 + i]
      );
    }
  }
}

// ---- scalar cosine_similarity: accepts f16/bf16 (final-cast only) ----

#[test]
fn cosine_similarity_scalar_f16_bf16_returns_similarity() {
  // Regression: scalar `cosine_similarity` extracts with `item::<f32>()`,
  // which is STRICT (no implicit cast). After the round-4 dtype-preserving
  // fixes, the dot/norm/divide stay in the INPUT dtype, so for f16/bf16
  // inputs `sim` was f16/bf16 → `item::<f32>()` => Err(DtypeMismatch).
  // The fix widens ONLY the final scalar to f32 (lossless, python computes
  // cosine in the vectors' dtype; we widen the in-dtype result). This
  // asserts: (a) returns Ok (not Err(DtypeMismatch)), and (b) equals the
  // python/MLX cosine computed IN that dtype then widened to f32.
  //
  // Fixture: a=(3,4), b=(4,3). Components are integers (bit-exact in both
  // f16 ≤2048 and bf16 ≤256). python-parity reference = the SAME cosine
  // computed by the crate at f32 then snapped to the target dtype grid
  // (`f32→half→f32`), compared at the dtype's own ULP via `half_close`
  // (the rigorous non-flaky bound used by the round-4 divide-path tests).
  let av = [3.0_f32, 4.0];
  let bv = [4.0_f32, 3.0];

  // f32 reference cosine = 24/25 = 0.96 (exact).
  let a_f32 = Array::from_slice(&av, &(2,)).unwrap();
  let b_f32 = Array::from_slice(&bv, &(2,)).unwrap();
  let ref_f32 = cosine_similarity(&a_f32, &b_f32).unwrap();

  for dt in [Dtype::F16, Dtype::BF16] {
    let a = Array::from_slice(&av, &(2,)).unwrap().astype(dt).unwrap();
    let b = Array::from_slice(&bv, &(2,)).unwrap().astype(dt).unwrap();
    // Must NOT be Err(DtypeMismatch): the broken path returned exactly
    // that for half-precision input.
    let got = cosine_similarity(&a, &b)
      .unwrap_or_else(|e| panic!("cosine_similarity {dt:?} errored: {e:?}"));
    let want = if dt == Dtype::F16 {
      to_f16_f32(ref_f32)
    } else {
      to_bf16_f32(ref_f32)
    };
    assert!(
      half_close(dt, got, want),
      "cosine_similarity scalar {dt:?}: got {got} want ~{want}"
    );
  }
}

#[test]
fn cosine_similarity_scalar_f16_bf16_identical_is_one() {
  // Identical half-precision vectors → ~1.0 within the dtype ULP (and,
  // critically, no Err(DtypeMismatch) from the strict `item::<f32>()`).
  let v = [1.0_f32, 2.0, 3.0];
  for dt in [Dtype::F16, Dtype::BF16] {
    let a = Array::from_slice(&v, &(3,)).unwrap().astype(dt).unwrap();
    let b = Array::from_slice(&v, &(3,)).unwrap().astype(dt).unwrap();
    let got = cosine_similarity(&a, &b)
      .unwrap_or_else(|e| panic!("cosine_similarity {dt:?} errored: {e:?}"));
    assert!(
      half_close(dt, got, 1.0),
      "cosine_similarity identical {dt:?} = {got}"
    );
  }
}

// ---- f32 regression guard: the fix must be dtype-PRESERVING ----

#[test]
fn cosine_similarity_scalar_f32_unchanged_after_final_cast() {
  // The final `astype(F32)` is a no-op cast for f32 input, so the f32
  // return must be BIT-IDENTICAL to before. (3,4)·(4,3)/(5·5)=0.96 exact;
  // identical=1.0, orthogonal=0.0 — re-pins the pre-existing f32 values.
  let a = Array::from_slice(&[3.0_f32, 4.0], &(2,)).unwrap();
  let b = Array::from_slice(&[4.0_f32, 3.0], &(2,)).unwrap();
  assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.96_f32);

  let i = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  assert!(close(cosine_similarity(&i, &i).unwrap(), 1.0));

  let e1 = Array::from_slice(&[1.0_f32, 0.0], &(2,)).unwrap();
  let e2 = Array::from_slice(&[0.0_f32, 1.0], &(2,)).unwrap();
  assert_eq!(cosine_similarity(&e1, &e2).unwrap(), 0.0_f32);
}

// ---- scalar cosine_similarity: rank/length precondition ----
//
// Codex round-7: scalar `cosine_similarity` documented same-length 1-D
// vectors but never validated rank/shape before `multiply(a, b)`. MLX
// broadcasting let a length-1 (or otherwise mismatched) `b` broadcast
// across a longer `a` while `norm(b)` used the original 1-element vector,
// yielding a "cosine" that can be > 1 (mathematically impossible) — silent
// ranking corruption on a dim/config mismatch. The fn now validates rank-1
// + equal length up front, returning Err(LengthMismatch) / Err(RankMismatch)
// instead.

#[test]
fn cosine_similarity_rejects_broadcastable_length_mismatch() {
  // (3,) vs (1,): MLX would broadcast b → an invalid score (can be > 1).
  // Must be Err(LengthMismatch), and specifically NOT any Ok value.
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice(&[1.0_f32], &(1,)).unwrap();
  let r = cosine_similarity(&a, &b);
  // (3,) vs (1,) is equal-rank-1 + unequal length → LengthMismatch.
  assert!(
    matches!(r, Err(Error::LengthMismatch(_))),
    "expected Err(LengthMismatch), got {r:?}"
  );
  assert!(r.is_err(), "must not return a (possibly > 1) value: {r:?}");
}

#[test]
fn cosine_similarity_rejects_unequal_lengths() {
  // (4,) vs (3,): equal-rank but unequal length → LengthMismatch.
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let b = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  assert!(matches!(
    cosine_similarity(&a, &b),
    Err(Error::LengthMismatch(_))
  ));
}

#[test]
fn cosine_similarity_rejects_non_rank1() {
  // Wrong-rank inputs → RankMismatch.
  let m = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let s = Array::from_slice(&[1.0_f32], &(1, 1)).unwrap();
  // rank-2 a, rank-2 b.
  assert!(matches!(
    cosine_similarity(&m, &s),
    Err(Error::RankMismatch(_))
  ));
  // rank-1 a, rank-2 b (only one side wrong).
  let v = Array::from_slice(&[1.0_f32], &(1,)).unwrap();
  assert!(matches!(
    cosine_similarity(&v, &s),
    Err(Error::RankMismatch(_))
  ));
  // rank-2 a, rank-1 b (symmetric).
  assert!(matches!(
    cosine_similarity(&s, &v),
    Err(Error::RankMismatch(_))
  ));
}

// ---- scalar cosine_similarity: zero-norm / length-0 → finite 0.0 ----
//
// Copilot review 4307433657 (#3256523014): `cosine_similarity` divided
// `dot` by the raw `||a||*||b||` with no eps floor, so a zero vector (or a
// valid length-0 input the round-7 rank/length validator allows) returned
// NaN/Inf — silent retrieval/ranking corruption, and inconsistent with
// `cosine_similarity_matrix` (eps-guarded `l2_normalize`). The denominator
// is now eps-floored (`DEFAULT_NORMALIZE_EPS`, dtype-aware, same guard as
// `l2_normalize`/`normalize`): `dot == 0 / max(||.||, eps) == 0`.

#[test]
fn cosine_similarity_zero_vector_is_finite_zero() {
  // One operand all-zeros: dot == 0, ||zero|| == 0 → floored to eps, so
  // the result is a finite 0.0 (NOT NaN/Inf), exactly consistent with the
  // matrix path (`l2_normalize(zero) == 0` → similarity 0).
  let a = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(3,)).unwrap();
  let b = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,)).unwrap();
  let got = cosine_similarity(&a, &b).unwrap();
  assert!(
    got.is_finite(),
    "zero-vector cosine must be finite, got {got}"
  );
  assert_eq!(got, 0.0_f32, "zero-vector cosine must be exactly 0.0");

  // Symmetric (zero on the other side) + both-zero.
  let got_sym = cosine_similarity(&b, &a).unwrap();
  assert!(got_sym.is_finite() && got_sym == 0.0_f32);
  let z = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(3,)).unwrap();
  let got_both = cosine_similarity(&a, &z).unwrap();
  assert!(
    got_both.is_finite() && got_both == 0.0_f32,
    "both-zero cosine must be finite 0.0, got {got_both}"
  );
}

#[test]
fn cosine_similarity_length_zero_is_finite() {
  // The round-7 validator treats `(0,)` vs `(0,)` as equal-length rank-1
  // (rank == 1 each, lengths 0 == 0) so it passes through; the empty dot
  // sums to 0 and both norms are 0 → without the floor this was 0/0 = NaN.
  // With the eps floor it is a finite 0.0.
  let a = Array::from_slice::<f32>(&[], &(0,)).unwrap();
  let b = Array::from_slice::<f32>(&[], &(0,)).unwrap();
  let r = cosine_similarity(&a, &b);
  let got = r.unwrap_or_else(|e| {
    panic!("length-0 vs length-0 must pass the rank/length validator, got {e:?}")
  });
  assert!(
    got.is_finite(),
    "length-0 cosine must be finite (no NaN/Inf), got {got}"
  );
  assert_eq!(got, 0.0_f32, "length-0 cosine must be exactly 0.0");
}

#[test]
fn cosine_similarity_zero_vector_f16_bf16_is_finite_zero() {
  // C3 (Copilot review 4307622782, #3256688255): the prior eps-floor used
  // `scalar_like(1e-9, &norm)`, casting `1e-9` into the NORM dtype. For
  // f16/bf16 `1e-9` is below the half subnormal floor → rounds to `0.0`,
  // so a zero f16/bf16 vector still did `0 / (0 * ||b||) = 0/0 = NaN` —
  // the documented finite-0.0 guarantee was FALSE for halves (the
  // f32-only zero-vector test masked it). The fix computes the final
  // ratio in f32 with a REAL f32 `1e-9` floor, so the guarantee now holds
  // for f16 AND bf16 too. This is the exact gap C3 identifies.
  for dt in [Dtype::F16, Dtype::BF16] {
    let zero = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(3,))
      .unwrap()
      .astype(dt)
      .unwrap();
    let nonzero = Array::from_slice(&[1.0_f32, 2.0, 3.0], &(3,))
      .unwrap()
      .astype(dt)
      .unwrap();

    // zero vs non-zero (the case that underflowed to NaN before the fix).
    let got = cosine_similarity(&zero, &nonzero)
      .unwrap_or_else(|e| panic!("cosine_similarity {dt:?} zero/nonzero errored: {e:?}"));
    assert!(
      got.is_finite(),
      "{dt:?} zero-vector cosine must be finite (not NaN/Inf), got {got}"
    );
    assert_eq!(
      got, 0.0_f32,
      "{dt:?} zero-vector cosine must be exactly 0.0"
    );

    // symmetric.
    let got_sym = cosine_similarity(&nonzero, &zero)
      .unwrap_or_else(|e| panic!("cosine_similarity {dt:?} nonzero/zero errored: {e:?}"));
    assert!(
      got_sym.is_finite() && got_sym == 0.0_f32,
      "{dt:?} symmetric zero-vector cosine must be finite 0.0, got {got_sym}"
    );

    // both-zero.
    let zero2 = Array::from_slice(&[0.0_f32, 0.0, 0.0], &(3,))
      .unwrap()
      .astype(dt)
      .unwrap();
    let got_both = cosine_similarity(&zero, &zero2)
      .unwrap_or_else(|e| panic!("cosine_similarity {dt:?} both-zero errored: {e:?}"));
    assert!(
      got_both.is_finite() && got_both == 0.0_f32,
      "{dt:?} both-zero cosine must be finite 0.0, got {got_both}"
    );
  }
}

// ---- scalar cosine_similarity: scale-invariance (D2 regression) ----
//
// Codex scoped-delta re-review [high]: the C3 fix unconditionally clamped
// each f32-widened norm with `max(norm, 1e-9)`. Cosine is scale-invariant,
// so colinear `a=[1e-12]`, `b=[1.0]` MUST be `1.0` (`1e-12/(1e-12*1)`),
// but the unconditional clamp yielded `1e-12/(1e-9*1) ≈ 0.001` — silent
// corruption of *valid finite* inputs (the eps was only ever meant to
// avoid `0/0` for EXACTLY-zero norms). The fix substitutes eps ONLY for
// exactly-zero norms (no clamp on nonzero norms); these cases FAIL on the
// buggy unconditional clamp and PASS after.
#[test]
fn cosine_similarity_scale_invariant_tiny_norm_f32_bf16() {
  // (1) f32 single-element colinear, extreme scale gap: a=[1e-12],
  //     b=[1.0]. Cosine = 1e-12/(1e-12*1) = 1.0 exactly (scale-invariant).
  //     The buggy clamp gave 1e-12/(1e-9*1) ≈ 1e-3.
  let a = Array::from_slice(&[1e-12_f32], &(1,)).unwrap();
  let b = Array::from_slice(&[1.0_f32], &(1,)).unwrap();
  let got = cosine_similarity(&a, &b).unwrap();
  assert!(
    got.is_finite() && got == 1.0_f32,
    "tiny-norm colinear cosine must be exactly 1.0 (scale-invariant), got {got}"
  );

  // (2) f32 multi-element scaled-colinear: b = 1e8 * a. Cosine == 1.0 to
  //     within a tight ULP bound (the only error is f32 dot/norm rounding,
  //     NOT a denominator clamp). The buggy clamp drove a small-norm `a`
  //     toward ~0.
  let a2 = Array::from_slice(&[1e-7_f32, 2e-7, 3e-7], &(3,)).unwrap();
  let b2 = Array::from_slice(&[1e1_f32, 2e1, 3e1], &(3,)).unwrap(); // 1e8 * a2
  let got2 = cosine_similarity(&a2, &b2).unwrap();
  assert!(
    got2.is_finite() && (got2 - 1.0_f32).abs() <= 4.0 * f32::EPSILON,
    "scaled-colinear (b = 1e8*a) cosine must be ~1.0, got {got2}"
  );

  // (3) tiny-norm ANTI-colinear: a=[-1e-12], b=[1.0] → exactly -1.0.
  let an = Array::from_slice(&[-1e-12_f32], &(1,)).unwrap();
  let bn = Array::from_slice(&[1.0_f32], &(1,)).unwrap();
  let gotn = cosine_similarity(&an, &bn).unwrap();
  assert!(
    gotn.is_finite() && gotn == -1.0_f32,
    "tiny-norm anti-colinear cosine must be exactly -1.0, got {gotn}"
  );

  // (4) bf16 small-but-representable scaled-colinear → 1.0 within the bf16
  //     ULP. `6.1035e-5` is a normal bf16 (well above its subnormal
  //     floor); colinear with `1.0`. The zero-only guard (in f32, post-
  //     widening) does NOT clamp this nonzero norm, so cosine is the
  //     correct scale-invariant ~1.0 — NOT the clamp-corrupted value, and
  //     NOT the finite-0.0 zero path.
  let small = 6.1035e-5_f32; // 2^-14, an exact normal bf16
  let abf = Array::from_slice(&[small], &(1,))
    .unwrap()
    .astype(Dtype::BF16)
    .unwrap();
  let bbf = Array::from_slice(&[1.0_f32], &(1,))
    .unwrap()
    .astype(Dtype::BF16)
    .unwrap();
  let gotbf = cosine_similarity(&abf, &bbf).unwrap();
  assert!(
    gotbf.is_finite() && half_close(Dtype::BF16, gotbf, 1.0),
    "bf16 small scaled-colinear cosine must be ~1.0 (scale-invariant), got {gotbf}"
  );
}

// ---- scalar cosine_similarity: zero norm vs overflowed (+Inf) norm ----
//
// Codex scoped-delta re-review [high]: the D2 formulation derived the
// zero predicate from `denom = na_f32 * nb_f32` via `equal(denom, 0.0)`.
// If one vector is a zero vector (`na_f32 == 0`) while the other is a
// *finite valid* input whose f32 L2 norm overflows to `+Inf` (e.g.
// `b = [f32::MAX, f32::MAX]` → `‖b‖₂ = +Inf`), then `denom = 0 * Inf =
// NaN`, `equal(NaN, 0.0)` is false, the NaN `safe_denom` leaks through,
// and the divide returns `NaN` — violating the documented finite-`0.0`
// contract for a one-zero-norm input. The D3 fix computes the predicate
// `‖a‖₂ == 0 ∨ ‖b‖₂ == 0` DIRECTLY on the widened norms (a real L2 norm
// is only ever 0/finite/+Inf, never NaN), so this case is finite `0.0`.
// This test FAILS on the D2 product-derived predicate and PASSES after.
#[test]
fn cosine_similarity_zero_vs_overflowed_norm_is_finite_zero() {
  // (1) f32: a is the zero vector; b = [f32::MAX, f32::MAX] is a finite
  //     valid input whose f32 L2 norm overflows to +Inf
  //     (f32::MAX^2 = +Inf, sqrt(+Inf) = +Inf). The D2 code did
  //     0 * Inf = NaN → NaN leaked. Must be a finite, exact 0.0.
  let zero = Array::from_slice(&[0.0_f32, 0.0], &(2,)).unwrap();
  let overflowed = Array::from_slice(&[f32::MAX, f32::MAX], &(2,)).unwrap();
  let got = cosine_similarity(&zero, &overflowed).unwrap();
  assert!(
    got.is_finite(),
    "zero vs overflowed-norm cosine must be finite (NOT NaN/Inf), got {got}"
  );
  assert_eq!(
    got, 0.0_f32,
    "zero vs overflowed-norm cosine must be exactly 0.0, got {got}"
  );

  // (2) symmetric: overflowed-norm vs zero → same finite 0.0 (the
  //     `‖a‖₂ == 0 ∨ ‖b‖₂ == 0` predicate is order-independent; D2's
  //     `0 * Inf` is also `Inf * 0 = NaN`, so this direction broke too).
  let got_sym = cosine_similarity(&overflowed, &zero).unwrap();
  assert!(
    got_sym.is_finite() && got_sym == 0.0_f32,
    "symmetric (overflowed-norm vs zero) cosine must be finite 0.0, got {got_sym}"
  );

  // (3) f16 zero vs an f16 vector whose IN-DTYPE L2 norm overflows to
  //     +Inf: f16 max is 65504; the norm is computed in the input dtype
  //     (f16), and `65504^2` overflows f16 to +Inf, so `‖b‖₂(f16) = +Inf`
  //     → widened to f32 +Inf. This deterministically reproduces the
  //     `0 * Inf = NaN` hole for a half dtype too (the per-norm f32
  //     predicate still maps it to finite 0.0). bf16 max is ~3.39e38
  //     (≈ f32::MAX), so a bf16 overflow case is the same construction as
  //     (1) post-widening and is already covered by the f32 case +
  //     `cosine_similarity_zero_vector_f16_bf16_is_finite_zero`; not
  //     re-duplicated here.
  let f16_max = half::f16::MAX.to_f32(); // 65504.0
  let zero_h = Array::from_slice(&[0.0_f32, 0.0], &(2,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let overflowed_h = Array::from_slice(&[f16_max, f16_max], &(2,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let got_h = cosine_similarity(&zero_h, &overflowed_h)
    .unwrap_or_else(|e| panic!("f16 zero vs overflowed-norm errored: {e:?}"));
  assert!(
    got_h.is_finite(),
    "f16 zero vs overflowed-norm cosine must be finite (NOT NaN/Inf), got {got_h}"
  );
  assert_eq!(
    got_h, 0.0_f32,
    "f16 zero vs overflowed-norm cosine must be exactly 0.0, got {got_h}"
  );
  let got_h_sym = cosine_similarity(&overflowed_h, &zero_h)
    .unwrap_or_else(|e| panic!("f16 symmetric overflowed-norm vs zero errored: {e:?}"));
  assert!(
    got_h_sym.is_finite() && got_h_sym == 0.0_f32,
    "f16 symmetric (overflowed-norm vs zero) cosine must be finite 0.0, got {got_h_sym}"
  );
}

// ---- scalar cosine_similarity: numerically-stable max-abs scaling ----
//
// TERMINAL fix: scalar `cosine_similarity` now scales each vector by its
// max-abs (Chebyshev / ∞-norm) before the dot/norm, so all magnitudes are
// O(1) and underflow / overflow / `0*Inf` are STRUCTURALLY impossible.
// `s = max(|x|)` uses NO `square`, so it is exact: `s == 0` iff the vector
// is genuinely all-zero. The prior `sqrt(sum(square(x)))`-derived
// zero/result underflowed (`square(1e-23) → 0`) or overflowed
// (`square(f32::MAX) → +Inf`), misclassifying tiny nonzero vectors as zero
// and leaking `NaN`. These cases span the full tiny→huge finite range and
// the Codex round-4 f16 counterexample class; all are exact/ULP-correct
// scale-invariant cosines (NOT the finite-0.0 zero path).
#[test]
fn cosine_similarity_tiny_and_huge_nonzero_are_scale_invariant() {
  // (1) f32 tiny vs unit, colinear: a=[1e-23], b=[1.0]. The OLD norm path
  //     did `square(1e-23) = 1e-46 → 0` in f32, so `‖a‖₂ = 0` and `a` was
  //     misclassified as a zero vector → finite 0.0 (WRONG: it is a
  //     perfectly colinear nonzero vector, cosine = 1.0). Max-abs scaling:
  //     s_a=1e-23 → â=[1.0], so cosine = exactly 1.0.
  let a = Array::from_slice(&[1e-23_f32], &(1,)).unwrap();
  let b = Array::from_slice(&[1.0_f32], &(1,)).unwrap();
  let got = cosine_similarity(&a, &b).unwrap();
  assert!(
    got.is_finite() && got == 1.0_f32,
    "f32 tiny [1e-23] vs [1.0] colinear must be exactly 1.0 (scale-invariant, NOT the underflow-misclassified 0.0), got {got}"
  );

  // (2) f32 tiny ANTI-colinear: a=[1e-30], b=[-1e-30]. Both underflow
  //     `square` to 0 in the old path; max-abs scaling → â=[1.0],
  //     b̂=[-1.0] → exactly -1.0.
  let an = Array::from_slice(&[1e-30_f32], &(1,)).unwrap();
  let bn = Array::from_slice(&[-1e-30_f32], &(1,)).unwrap();
  let gotn = cosine_similarity(&an, &bn).unwrap();
  assert!(
    gotn.is_finite() && gotn == -1.0_f32,
    "f32 tiny [1e-30] vs [-1e-30] anti-colinear must be exactly -1.0, got {gotn}"
  );

  // (3) f32 huge colinear: a=[f32::MAX, f32::MAX], b=[1.0, 1.0]. The OLD
  //     path did `square(f32::MAX) = +Inf` so `‖a‖₂ = +Inf` → the ratio
  //     leaked Inf/NaN. Max-abs scaling: s_a=f32::MAX → â=[1.0, 1.0],
  //     b̂=[1.0, 1.0] → cosine = 1.0 within a tight ULP bound (the only
  //     error is the 2-element f32 norm rounding `sqrt(2)·sqrt(2) ≠ 2`,
  //     NOT the overflow — proving the overflow class is also terminal).
  let huge = Array::from_slice(&[f32::MAX, f32::MAX], &(2,)).unwrap();
  let ones = Array::from_slice(&[1.0_f32, 1.0], &(2,)).unwrap();
  let goth = cosine_similarity(&huge, &ones).unwrap();
  assert!(
    goth.is_finite() && (goth - 1.0_f32).abs() <= 4.0 * f32::EPSILON,
    "f32 huge [f32::MAX,f32::MAX] vs [1,1] colinear must be ~1.0 (overflow class terminal), got {goth}"
  );

  // (4) f16 tiny vs unit, colinear: a=[6.1035e-5], b=[1.0]. `6.1035e-5`
  //     is `2^-14`, the smallest normal f16 — the Codex round-4
  //     counterexample class (a half-cast tiny value whose `square`
  //     underflows in f16). Widened to f32, max-abs scaling → â=[1.0]
  //     (exact: `s/s == 1.0`), b̂=[1.0] → exactly 1.0 (single-element
  //     scaled vector is bit-exact ±1.0 in f32). (f16 1e-30 underflows to
  //     0 in f16 itself, so 6.1e-5 is the correct tiny-but-representable
  //     f16 magnitude per the round-4 class.)
  let small = 6.1035e-5_f32; // 2^-14, exact smallest normal f16
  let af16 = Array::from_slice(&[small], &(1,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let bf16 = Array::from_slice(&[1.0_f32], &(1,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let gotf = cosine_similarity(&af16, &bf16)
    .unwrap_or_else(|e| panic!("f16 tiny [6.1e-5] vs [1.0] errored: {e:?}"));
  assert!(
    gotf.is_finite() && gotf == 1.0_f32,
    "f16 tiny [6.1035e-5] vs [1.0] colinear must be exactly 1.0 (round-4 class), got {gotf}"
  );

  // (5) f16 tiny ANTI-colinear: a=[6.1035e-5], b=[-6.1035e-5] (both exact
  //     f16). Max-abs scaling → â=[1.0], b̂=[-1.0] → exactly -1.0. The
  //     round-4 counterexample direction (tiny half anti-colinear).
  let af16n = Array::from_slice(&[small], &(1,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let bf16n = Array::from_slice(&[-small], &(1,))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let gotfn = cosine_similarity(&af16n, &bf16n)
    .unwrap_or_else(|e| panic!("f16 tiny anti-colinear errored: {e:?}"));
  assert!(
    gotfn.is_finite() && gotfn == -1.0_f32,
    "f16 tiny [6.1035e-5] vs [-6.1035e-5] anti-colinear must be exactly -1.0, got {gotfn}"
  );
}

#[test]
fn f32_paths_bit_identical_after_dtype_fix() {
  // The scalar_like helper builds the floor/-inf/0 as f32 then astype to
  // x.dtype(); for f32 x that is a no-op cast, so every f32 result must
  // be unchanged. This re-asserts the canonical fixture expectations to
  // pin bit-identity (alongside the 51 pre-existing f32 tests).
  let (emb, mask) = fixture();
  assert_dtype(&emb, Dtype::F32, "fixture emb is f32");

  let mut mx = max_pooling(&emb, &mask).unwrap();
  assert_dtype(&mx, Dtype::F32, "max_pooling f32");
  assert_eq!(mx.to_vec::<f32>().unwrap(), vec![5.0, 6.0, 70.0, 80.0]);

  let x = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut n = l2_normalize(&x).unwrap();
  assert_dtype(&n, Dtype::F32, "l2_normalize f32");
  assert!(vclose(&n.to_vec::<f32>().unwrap(), &[0.6, 0.8]));

  let mut np = normalize(&x, 2.0, -1, true, DEFAULT_NORMALIZE_EPS).unwrap();
  assert!(vclose(&np.to_vec::<f32>().unwrap(), &[0.6, 0.8]));
}

// ════════════════ #260 coverage: pool_post tail, PoolingStrategy
// as_str/Display/IsVariant + from_mode("last") alias, and
// truncate_last_dim rank-1/rank-3 / None-passthrough truncation. ════════
//
// pooling.rs `pool_post` (the shared normalize/dimension/layer-norm tail
// `pool` runs after the strategy reduction, also called directly by
// `encode` on a model's trained `pooled_output`) had ZERO coverage — it
// was not even imported by this test module. These pin its documented
// step order (LayerNorm|RMSNorm → matryoshka truncation → L2-normalize),
// its no-transform passthrough, and its equivalence to the `pool`
// dispatcher tail. The PoolingStrategy `as_str`/Display/IsVariant accessor
// surface and the `from_mode("last")` alias were likewise untested, as
// were `truncate_last_dim` on rank-1 / rank-3 (only rank-2 was covered).

// ───────────────── pool_post: the shared post-pool tail ─────────────────

#[test]
fn pool_post_no_transform_is_passthrough() {
  // All flags off / dimension None → `pool_post` returns the pooled vector
  // unchanged (the documented by-value no-copy passthrough).
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut p = pool_post(x, false, None, false, false).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert_eq!(p.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pool_post_truncate_only() {
  // dimension=Some(2), no norm: matryoshka-truncate the last axis to 2.
  // Same gather as `truncate_last_dim_basic`: (2,3) -> (2,2) keeping cols
  // 0..2 of each row.
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut p = pool_post(x, false, Some(2), false, false).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 2.0, 4.0, 5.0]));
}

#[test]
fn pool_post_normalize_only_yields_unit_rows() {
  // normalize=true only: L2-normalize. [3,4] -> [0.6,0.8] (||.||=5).
  let x = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut p = pool_post(x, true, None, false, false).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[0.6, 0.8]));
}

#[test]
fn pool_post_truncate_then_normalize_order() {
  // Documented step order is truncate (dimension) BEFORE L2-normalize, so
  // each row is truncated to its first `dimension` entries and THEN scaled
  // to unit norm over those entries. Row0 [3,4,99] -trunc2-> [3,4]
  // -norm-> [0.6,0.8]; row1 [0,5,12] -trunc2-> [0,5] -norm-> [0,1].
  let x = Array::from_slice(&[3.0_f32, 4.0, 99.0, 0.0, 5.0, 12.0], &(2, 3)).unwrap();
  let mut p = pool_post(x, true, Some(2), false, false).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[0.6, 0.8, 0.0, 1.0]));
}

#[test]
fn pool_post_equivalent_to_pool_dispatcher_tail() {
  // STRUCTURAL equivalence only: `pool_post` IS the tail `pool` runs after
  // the strategy reduction, so for every (normalize, dimension, layer_norm,
  // rms_norm) combination `pool_post(mean_pooling(emb, mask), ..flags) ==
  // pool(emb, mask, Mean, ..flags)`. Because `pool` *delegates* to
  // `pool_post`, both sides share the norm code path and this CANNOT catch a
  // broken LayerNorm/RMSNorm (it would be identically wrong on both sides).
  // The CORRECTNESS of the LayerNorm / RMSNorm / precedence / order steps is
  // pinned independently of `pool` by the closed-form `pool_post_*_closed_form`
  // tests below; this one only guarantees `encode`'s factored-out tail and the
  // dispatcher stay wired to the same code.
  let (emb, mask) = fixture();
  for (normalize, dim, ln, rms) in [
    (false, None, false, false),
    (true, None, false, false),
    (false, Some(1), false, false),
    (true, Some(1), false, false),
    (false, None, true, false), // layer-norm
    (false, None, false, true), // rms-norm
    (true, None, true, false),  // layer-norm + normalize
    (false, None, true, true),  // both set -> layer-norm wins
  ] {
    let pooled = mean_pooling(&emb, &mask).unwrap();
    let mut via_post = pool_post(pooled, normalize, dim, ln, rms).unwrap();
    let mut via_pool = pool(&emb, &mask, PoolingStrategy::Mean, normalize, dim, ln, rms).unwrap();
    assert_eq!(
      via_post.shape(),
      via_pool.shape(),
      "shape mismatch for (norm={normalize}, dim={dim:?}, ln={ln}, rms={rms})"
    );
    assert!(
      vclose(
        &via_post.to_vec::<f32>().unwrap(),
        &via_pool.to_vec::<f32>().unwrap()
      ),
      "pool_post must equal pool tail for (norm={normalize}, dim={dim:?}, ln={ln}, rms={rms})"
    );
  }
}

// ── pool_post norm steps: CLOSED-FORM, independent of `pool` (Codex #1) ──
//
// These pin the LayerNorm / RMSNorm / precedence / step-order CONTRACT of
// `pool_post` against values hand-computed from the documented formulas in
// `embeddings/fast.rs` + the call-site eps in `embeddings/pooling.rs`, with
// NO reference to `pool` (which delegates to `pool_post`, so a `pool`-vs-
// `pool_post` comparison would be tautological — Codex finding #1). Formulas:
//   LayerNorm (no affine): (x-mean)/sqrt(var+eps), population var over the
//     last axis, eps = LAYER_NORM_EPS = 1e-5 (pooling.rs:34, applied at
//     pooling.rs:458 via fast::layer_norm, fast.rs:43-77).
//   RMSNorm  (no affine): x/sqrt(mean(x^2)+eps), eps = RMS_NORM_EPS = 1e-5
//     (pooling.rs:39, applied at pooling.rs:460 via fast::rms_norm,
//     fast.rs:79-104).
//   L2 (normalize): x/max(||x||_2, 1e-9) (DEFAULT_NORMALIZE_EPS,
//     normalize.rs:26, applied at pooling.rs:468).
// Step order (pooling.rs:457-471): norm → truncate(dimension) → L2.

#[test]
fn pool_post_layer_norm_closed_form() {
  // apply_layer_norm only. Row [1,2,3,4]: mean=2.5, population var=1.25,
  // denom=sqrt(1.25+1e-5)=1.11803842; output=(x-2.5)/denom =
  // [-1.5,-0.5,0.5,1.5]/1.11803842. Derived from the LayerNorm formula
  // alone — no `pool` call.
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let mut p = pool_post(x, false, None, true, false).unwrap();
  assert_eq!(p.shape(), vec![1, 4]);
  assert!(vclose(
    &p.to_vec::<f32>().unwrap(),
    &[-1.3416354, -0.4472118, 0.4472118, 1.3416354],
  ));
}

#[test]
fn pool_post_rms_norm_closed_form_eps_load_bearing() {
  // apply_rms_norm only, with a tiny-magnitude row [0.001, 0.001] chosen so
  // RMS_NORM_EPS=1e-5 DOMINATES the mean-square (1e-6): denom =
  // sqrt(1e-6 + 1e-5) = sqrt(1.1e-5) = 3.3166248e-3, output =
  // 0.001/3.3166248e-3 = 0.30151135 each. With eps=0 this would be 1.0
  // each, so the assertion FAILS if the impl used the wrong eps (or 0) —
  // pinning the exact RMS_NORM_EPS, not just the RMS shape. Derived from
  // the RMSNorm formula alone — no `pool` call.
  let x = Array::from_slice(&[0.001_f32, 0.001], &(1, 2)).unwrap();
  let mut p = pool_post(x, false, None, false, true).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(
    &p.to_vec::<f32>().unwrap(),
    &[0.30151135, 0.30151135]
  ));
}

#[test]
fn pool_post_layer_norm_wins_over_rms_closed_form() {
  // BOTH apply_layer_norm and apply_rms_norm set → LayerNorm must win
  // (pooling.rs:457 `if apply_layer_norm … else if apply_rms_norm`). Row
  // [1,2,3,4]: the result must equal the LAYERNORM closed-form
  // [-1.3416354,-0.4472118,0.4472118,1.3416354] and must NOT equal the
  // RMSNorm closed-form (x/sqrt(mean(x^2)+1e-5) =
  // [0.36514813,0.73029626,1.0954444,1.4605925]). Asserting both the
  // positive match AND the negative non-match pins precedence
  // independently of `pool`.
  let layer_norm_expected = [-1.3416354_f32, -0.4472118, 0.4472118, 1.3416354];
  let rms_expected = [0.36514813_f32, 0.73029626, 1.0954444, 1.4605925];
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let mut p = pool_post(x, false, None, true, true).unwrap();
  let got = p.to_vec::<f32>().unwrap();
  assert!(
    vclose(&got, &layer_norm_expected),
    "both flags set must yield the LayerNorm result, got {got:?}"
  );
  assert!(
    !vclose(&got, &rms_expected),
    "both flags set must NOT yield the RMSNorm result, got {got:?}"
  );
}

#[test]
fn pool_post_layer_norm_then_truncate_then_normalize_order_closed_form() {
  // Combined norm + truncate + normalize, pinning the documented step ORDER
  // (LayerNorm → truncate → L2). Row [-3,-1,1,3] (mean 0): LayerNorm denom =
  // sqrt(var+1e-5), var=5 → /sqrt(5.00001); truncate to first 2 →
  // [-3,-1]/sqrt(5.00001); L2-normalize over those 2: the common
  // 1/sqrt(5.00001) factor cancels, leaving the L2-normalization of [-3,-1]
  // = [-3,-1]/sqrt(10) = [-0.9486833, -0.31622776]. If the order were
  // truncate-then-LayerNorm, the LayerNorm would re-center [-3,-1] (mean -2)
  // → a different vector, so this pins ORDER, not just the individual steps.
  // Derived from the formulas alone — no `pool` call.
  let x = Array::from_slice(&[-3.0_f32, -1.0, 1.0, 3.0], &(1, 4)).unwrap();
  let mut p = pool_post(x, true, Some(2), true, false).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(
    &p.to_vec::<f32>().unwrap(),
    &[-0.9486833, -0.31622776],
  ));
}

// ───────────── PoolingStrategy: as_str / Display / IsVariant ─────────────

#[test]
fn pooling_strategy_as_str_canonical_names() {
  // The canonical lowercase mode strings (python `pool_by_config` modes +
  // swift `Pooling.Strategy` display names).
  assert_eq!(PoolingStrategy::Mean.as_str(), "mean");
  assert_eq!(PoolingStrategy::Cls.as_str(), "cls");
  assert_eq!(PoolingStrategy::First.as_str(), "first");
  assert_eq!(PoolingStrategy::Last.as_str(), "last");
  assert_eq!(PoolingStrategy::Max.as_str(), "max");
  assert_eq!(PoolingStrategy::None.as_str(), "none");
}

#[test]
fn pooling_strategy_display_matches_as_str() {
  // `#[display("{}", self.as_str())]` — Display must equal as_str().
  for s in [
    PoolingStrategy::Mean,
    PoolingStrategy::Cls,
    PoolingStrategy::First,
    PoolingStrategy::Last,
    PoolingStrategy::Max,
    PoolingStrategy::None,
  ] {
    assert_eq!(format!("{s}"), s.as_str());
  }
}

#[test]
fn pooling_strategy_is_variant_predicates() {
  // derive_more::IsVariant generates `is_<variant>()` snake_case
  // predicates. Each is true for its own variant and false for the others.
  assert!(PoolingStrategy::Mean.is_mean());
  assert!(PoolingStrategy::Cls.is_cls());
  assert!(PoolingStrategy::First.is_first());
  assert!(PoolingStrategy::Last.is_last());
  assert!(PoolingStrategy::Max.is_max());
  assert!(PoolingStrategy::None.is_none());

  // Cross-checks: a couple of variants are NOT another variant.
  assert!(!PoolingStrategy::Mean.is_cls());
  assert!(!PoolingStrategy::Cls.is_mean());
  assert!(!PoolingStrategy::First.is_last());
  assert!(!PoolingStrategy::Last.is_first());
  assert!(!PoolingStrategy::None.is_max());
}

#[test]
fn pooling_strategy_from_mode_last_alias() {
  // The impl accepts BOTH "lasttoken" (python `_SUPPORTED_POOL_MODES`) and
  // the "last" alias (swift strategy name) for `PoolingStrategy::Last`. The
  // pre-existing `pooling_strategy_from_mode` test only exercises
  // "lasttoken"; this pins the "last" alias.
  assert_eq!(
    PoolingStrategy::from_mode("last").unwrap(),
    PoolingStrategy::Last
  );
  // Round-trip: as_str() of Last is "last", which from_mode must accept.
  assert_eq!(
    PoolingStrategy::from_mode(PoolingStrategy::Last.as_str()).unwrap(),
    PoolingStrategy::Last
  );
}

// ───────── truncate_last_dim: rank-1, rank-3, and None-passthrough ────────

#[test]
fn truncate_last_dim_rank1() {
  // ndim==1 path: a bare vector truncates to its first `dimension` entries.
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let mut t = truncate_last_dim(&x, 2).unwrap();
  assert_eq!(t.shape(), vec![2]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 2.0]));
}

#[test]
fn truncate_last_dim_rank3_keeps_first_of_last_axis() {
  // rank-3 (the PoolingStrategy::None passthrough shape): truncate only the
  // last axis. (2,2,2) -> (2,2,1) keeping index 0 of the last axis.
  // Row-major [[[1,2],[3,4]],[[5,6],[7,8]]] -> [[[1],[3]],[[5],[7]]].
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(2, 2, 2)).unwrap();
  let mut t = truncate_last_dim(&x, 1).unwrap();
  assert_eq!(t.shape(), vec![2, 2, 1]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 3.0, 5.0, 7.0]));
}

#[test]
fn dispatcher_none_passthrough_with_matryoshka_truncation() {
  // PoolingStrategy::None skips pooling but still honors `dimension`
  // (last-axis truncation) on the rank-3 hidden states — documented in the
  // `pool` doc but only the no-post-processing None path was tested. emb
  // (1,2,3) -> None keeps (1,2,3) -> truncate last dim to 2 -> (1,2,2).
  // [[[1,2,3],[4,5,6]]] -> [[[1,2],[4,5]]].
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(1, 2, 3)).unwrap();
  let mask = Array::ones::<f32>(&(1, 2)).unwrap();
  let mut p = pool(
    &emb,
    &mask,
    PoolingStrategy::None,
    false,
    Some(2),
    false,
    false,
  )
  .unwrap();
  assert_eq!(p.shape(), vec![1, 2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 2.0, 4.0, 5.0]));
}
