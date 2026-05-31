// Inline unit tests for `pooling.rs`. The whole `embeddings` module is
// `#[cfg(feature = "embeddings")]` at the crate root, so these inherit
// that gate and need no extra `cfg`. `use super::*` reaches both the
// public surface AND the two *private* validators
// (`validate_token_embeddings_and_mask`, `validate_token_embeddings_rank3`)
// and the `*_EPS` consts, which the external `tests/embeddings.rs`
// integration file cannot import. Oracles are CLOSED-FORM: every expected
// value is computed by hand from the documented formula on a small known
// input — never by calling the function under test.
use super::*;
use crate::{array::Array, dtype::Dtype, error::Error};

const TOL: f32 = 1e-5;

fn close(a: f32, b: f32) -> bool {
  (a - b).abs() <= TOL
}

fn vclose(a: &[f32], b: &[f32]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

// ───────────── private validator: rank/shape contract ─────────────
// These hit the two private guards DIRECTLY (the public wrappers all
// delegate to them) and assert the typed-payload CONTENTS — the observed
// rank + shape — which the public-API integration tests never inspect.

#[test]
fn validate_ok_for_well_formed_rank3_emb_and_rank2_mask() {
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0], &(1, 2)).unwrap();
  assert!(validate_token_embeddings_and_mask(&emb, &mask).is_ok());
  assert!(validate_token_embeddings_rank3(&emb).is_ok());
}

#[test]
fn validate_rejects_non_rank3_emb_with_observed_rank_and_shape() {
  // rank-2 token_embeddings → RankMismatch carrying actual()==2 and the
  // full observed shape [1,2] (the payload fields the public tests skip).
  let emb_2d = Array::from_slice(&[1.0_f32, 2.0], &(1, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0], &(1, 2)).unwrap();
  match validate_token_embeddings_and_mask(&emb_2d, &mask) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(p.actual(), 2, "observed rank");
      assert_eq!(p.actual_shape(), &[1, 2], "observed shape");
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
  // The mask-free entry point reports the same on a rank-1 input.
  let emb_1d = Array::from_slice(&[1.0_f32, 2.0], &(2,)).unwrap();
  match validate_token_embeddings_rank3(&emb_1d) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(p.actual(), 1);
      assert_eq!(p.actual_shape(), &[2]);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

#[test]
fn validate_rejects_non_rank2_mask() {
  // emb rank-3 OK, but mask rank-3 → RankMismatch on the mask branch.
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  let mask_3d = Array::from_slice(&[1.0_f32, 1.0], &(1, 2, 1)).unwrap();
  match validate_token_embeddings_and_mask(&emb, &mask_3d) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1, 2, 1]);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

#[test]
fn validate_rejects_batch_or_seq_mismatch_with_both_shapes() {
  // emb (batch,seq)=(1,2), mask (batch,seq)=(1,3) → ShapePairMismatch
  // carrying expected==[1,2] (the emb side) and actual==[1,3] (the mask).
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  let bad_mask = Array::from_slice(&[1.0_f32, 1.0, 1.0], &(1, 3)).unwrap();
  match validate_token_embeddings_and_mask(&emb, &bad_mask) {
    Err(Error::ShapePairMismatch(p)) => {
      assert_eq!(p.expected(), &[1, 2], "emb (batch, seq_len)");
      assert_eq!(p.actual(), &[1, 3], "mask (batch, seq_len)");
    }
    other => panic!("expected ShapePairMismatch, got {other:?}"),
  }
}

// ───────────── mean_pooling: hand-averaged oracle ─────────────

#[test]
fn mean_pooling_hand_average_over_unmasked() {
  // emb (1,3,2) rows: [1,2],[3,4],[5,6]; mask [1,1,0] → keep rows 0,1.
  // Hand mean: ([1,2]+[3,4])/2 = [2,3]. Floor (sum_mask=2) irrelevant.
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  let mut p = mean_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[2.0, 3.0]));
}

#[test]
fn mean_pooling_all_pad_row_uses_1e9_floor_finite_near_zero() {
  // mask all 0 → sum_embeddings=[0,0]; sum_mask=0 floored to 1e-9 →
  // 0/1e-9 = 0 (FINITE, not NaN). Pins the documented `max(sum,1e-9)`
  // guard. emb values are irrelevant (all weighted by mask 0).
  let emb = Array::from_slice(&[9.0_f32, 9.0, 7.0, 7.0], &(1, 2, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0], &(1, 2)).unwrap();
  let mut p = mean_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  let v = p.to_vec::<f32>().unwrap();
  assert!(
    v.iter().all(|x| x.is_finite()),
    "floor must avoid NaN: {v:?}"
  );
  assert!(vclose(&v, &[0.0, 0.0]));
}

#[test]
fn mean_pooling_output_is_f32_even_when_input_is_f16() {
  // python `mean_pooling` does `mask.astype(mx.float32)`, so the output
  // is f32 by design regardless of input dtype (documented exception).
  let emb = Array::from_slice(&[2.0_f32, 4.0], &(1, 2, 1))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let mask = Array::ones::<f32>(&(1, 2)).unwrap();
  let mut p = mean_pooling(&emb, &mask).unwrap();
  assert_eq!(p.dtype().unwrap(), Dtype::F32);
  // mean of [2],[4] over full mask = 3.
  assert!(close(p.to_vec::<f32>().unwrap()[0], 3.0));
}

// ───────────── max_pooling: hand-max oracle ─────────────

#[test]
fn max_pooling_forces_pad_to_neg_inf_then_maxes() {
  // emb (1,3,2): [1,9],[8,2],[100,100]; mask [1,1,0] (last pos masked).
  // Pad row forced to -inf, so per-dim max over rows 0,1 = [max(1,8),
  // max(9,2)] = [8,9]. The big masked row never wins.
  let emb = Array::from_slice(&[1.0_f32, 9.0, 8.0, 2.0, 100.0, 100.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  let mut p = max_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[8.0, 9.0]));
}

#[test]
fn max_pooling_handles_negative_values_under_mask() {
  // All-negative emb so the -inf pad-fill cannot be mistaken for a real
  // max. emb (1,2,1): [-5],[-2]; mask [1,0] → only row 0 survives → -5.
  let emb = Array::from_slice(&[-5.0_f32, -2.0], &(1, 2, 1)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 0.0], &(1, 2)).unwrap();
  let mut p = max_pooling(&emb, &mask).unwrap();
  assert!(close(p.to_vec::<f32>().unwrap()[0], -5.0));
}

#[test]
fn max_pooling_preserves_f16_dtype() {
  // python casts the mask to emb.dtype (NOT f32), so f16 in → f16 out.
  // Values 1..4 are exact in f16; max over full mask = 4.
  let emb = Array::from_slice(&[1.0_f32, 4.0], &(1, 2, 1))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let mask = Array::ones::<f32>(&(1, 2)).unwrap();
  let p = max_pooling(&emb, &mask).unwrap();
  assert_eq!(p.dtype().unwrap(), Dtype::F16);
}

// ───────────── cls_pooling: hand-picked first-real row ─────────────

#[test]
fn cls_pooling_picks_argmax_mask_row_under_left_padding() {
  // mask [0,0,1]: argmax(mask)=2 → row 2. emb rows [1,1],[2,2],[3,3] →
  // [3,3]. Distinct from first_token (which would give row 0 = [1,1]).
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0, 1.0], &(1, 3)).unwrap();
  let mut p = cls_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[3.0, 3.0]));
}

#[test]
fn cls_pooling_all_pad_row_argmax_is_index0() {
  // mask all 0 → argmax of an all-equal row is index 0 (first max),
  // so cls gathers row 0 unchanged = [5,6]. Documents the all-pad
  // fallback (cls_pooling has no `*mask` zeroing, unlike last_token).
  let emb = Array::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], &(1, 2, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0], &(1, 2)).unwrap();
  let mut p = cls_pooling(&emb, &mask).unwrap();
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[5.0, 6.0]));
}

// ───────────── last_token_pooling: reversed-argmax oracle ─────────────

#[test]
fn last_token_pooling_left_padded_selects_last_real() {
  // seq_len 3, mask [0,1,1]. python: flipped=[1,1,0]; argmax=0;
  // last = 3-0-1 = 2. emb rows [1,1],[2,2],[3,3] → row 2 = [3,3]
  // (mask[2]==1 so `*mask` keeps it).
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 3.0, 3.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 1.0, 1.0], &(1, 3)).unwrap();
  let mut p = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[3.0, 3.0]));
}

#[test]
fn last_token_pooling_right_padded_selects_last_real() {
  // mask [1,1,0]. python: flipped=[0,1,1]; argmax=1; last=3-1-1=1.
  // emb rows [1,1],[2,2],[9,9] → row 1 = [2,2].
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 9.0, 9.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  let mut p = last_token_pooling(&emb, &mask).unwrap();
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[2.0, 2.0]));
}

#[test]
fn last_token_pooling_all_pad_falls_back_to_zeros() {
  // mask all 0: max(flipped)==0 → flip_indices=seq_len-1=1; last=2-1-1=0;
  // gather (emb*mask)[0], and mask[0]==0 → zeros (python `*mask` parity).
  let emb = Array::from_slice(&[3.0_f32, 4.0, 5.0, 6.0], &(1, 2, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 0.0], &(1, 2)).unwrap();
  let mut p = last_token_pooling(&emb, &mask).unwrap();
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[0.0, 0.0]));
}

#[test]
fn last_token_pooling_mixed_pad_batch() {
  // Row 0 left-pad [0,1]: flipped=[1,0],argmax=0,last=2-0-1=1 → row 1.
  // Row 1 right-pad [1,0]: flipped=[0,1],argmax=1,last=2-1-1=0 → row 0.
  // emb row0 [[1,1],[2,2]] → [2,2]; row1 [[7,7],[9,9]] → [7,7].
  let emb = Array::from_slice(&[1.0_f32, 1.0, 2.0, 2.0, 7.0, 7.0, 9.0, 9.0], &(2, 2, 2)).unwrap();
  let mask = Array::from_slice(&[0.0_f32, 1.0, 1.0, 0.0], &(2, 2)).unwrap();
  let mut p = last_token_pooling(&emb, &mask).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[2.0, 2.0, 7.0, 7.0]));
}

// ───────────── first_token_pooling: strict token-0, mask-ignored ─────────

#[test]
fn first_token_pooling_always_takes_row0_ignoring_mask() {
  // mask [0,1] would route cls to row 1, but `first` is strict token-0.
  // emb rows [1,2],[3,4] → [1,2] regardless of mask.
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  let mut p = first_token_pooling(&emb).unwrap();
  assert_eq!(p.shape(), vec![1, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 2.0]));
}

// ───────────── single-token sequence (seq_len == 1) edge case ─────────────

#[test]
fn single_token_sequence_all_strategies_return_that_token() {
  // (batch=1, seq=1, hidden=2), single real token [4,5], mask [1].
  // Every mask-aware strategy reduces over a 1-length seq → exactly that
  // token. mean=[4,5]/1; max=[4,5]; cls (argmax=0)=[4,5];
  // last (flipped=[1],argmax=0,last=0, *mask keeps it)=[4,5]; first=[4,5].
  let emb = Array::from_slice(&[4.0_f32, 5.0], &(1, 1, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32], &(1, 1)).unwrap();
  let want = [4.0_f32, 5.0];
  for (label, mut p) in [
    ("mean", mean_pooling(&emb, &mask).unwrap()),
    ("max", max_pooling(&emb, &mask).unwrap()),
    ("cls", cls_pooling(&emb, &mask).unwrap()),
    ("last", last_token_pooling(&emb, &mask).unwrap()),
    ("first", first_token_pooling(&emb).unwrap()),
  ] {
    assert_eq!(p.shape(), vec![1, 2], "shape for {label}");
    assert!(
      vclose(&p.to_vec::<f32>().unwrap(), &want),
      "value for {label}"
    );
  }
}

// ───────────── pool() dispatcher: routing per strategy ─────────────
// One fixture, hand-computed expected per strategy, no post-processing.
// Asserts the dispatcher routes to the matching reduction.

#[test]
fn pool_dispatches_each_strategy_to_its_reduction() {
  // emb (1,3,2) rows [2,2],[4,4],[8,8]; mask [1,1,0] (pos 2 masked).
  // mean: ([2,2]+[4,4])/2 = [3,3]
  // max : per-dim max over rows 0,1 (pad -inf) = [4,4]
  // cls : argmax([1,1,0])=0 → row 0 = [2,2]
  // first: strict row 0 = [2,2]
  // last: flipped=[0,1,1],argmax=1,last=3-1-1=1 → row 1 = [4,4]
  let emb = Array::from_slice(&[2.0_f32, 2.0, 4.0, 4.0, 8.0, 8.0], &(1, 3, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0, 0.0], &(1, 3)).unwrap();
  for (strat, want) in [
    (PoolingStrategy::Mean, [3.0_f32, 3.0]),
    (PoolingStrategy::Max, [4.0, 4.0]),
    (PoolingStrategy::Cls, [2.0, 2.0]),
    (PoolingStrategy::First, [2.0, 2.0]),
    (PoolingStrategy::Last, [4.0, 4.0]),
  ] {
    let mut p = pool(&emb, &mask, strat, false, None, false, false).unwrap();
    assert_eq!(p.shape(), vec![1, 2], "shape for {strat:?}");
    assert!(
      vclose(&p.to_vec::<f32>().unwrap(), &want),
      "value for {strat:?}"
    );
  }
}

#[test]
fn pool_none_is_rank3_passthrough() {
  // None skips pooling: keeps (batch,seq,hidden) rank + values, no norms.
  let emb = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 2, 2)).unwrap();
  let mask = Array::ones::<f32>(&(1, 2)).unwrap();
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
  assert_eq!(p.shape(), vec![1, 2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn pool_propagates_rank_mismatch_from_validator() {
  // The dispatcher must surface the validator's typed error (not panic)
  // for a rank-2 token_embeddings on a mask-aware strategy.
  let emb_2d = Array::from_slice(&[1.0_f32, 2.0], &(1, 2)).unwrap();
  let mask = Array::from_slice(&[1.0_f32, 1.0], &(1, 2)).unwrap();
  assert!(matches!(
    pool(
      &emb_2d,
      &mask,
      PoolingStrategy::Mean,
      false,
      None,
      false,
      false
    ),
    Err(Error::RankMismatch(_))
  ));
}

// ───────────── pool_post: closed-form norm/truncate/L2 tail ─────────────

#[test]
fn pool_post_no_transform_returns_input_unchanged() {
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut p = pool_post(x, false, None, false, false).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn pool_post_layer_norm_closed_form() {
  // LayerNorm (no affine) over last axis: (x-mean)/sqrt(var+1e-5),
  // population var. Row [1,2,3,4]: mean=2.5, var=1.25,
  // denom=sqrt(1.25001)=1.1180384 → [-1.5,-0.5,0.5,1.5]/1.1180384 =
  // [-1.3416354,-0.4472118,0.4472118,1.3416354]. eps=LAYER_NORM_EPS.
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
  // RMSNorm (no affine): x/sqrt(mean(x^2)+1e-5). Row [0.001,0.001] chosen
  // so RMS_NORM_EPS=1e-5 DOMINATES mean-square 1e-6: denom=
  // sqrt(1.1e-5)=3.3166248e-3 → 0.001/3.3166248e-3 = 0.30151135 each.
  // (With eps=0 it would be 1.0 each — so this pins the exact eps.)
  let x = Array::from_slice(&[0.001_f32, 0.001], &(1, 2)).unwrap();
  let mut p = pool_post(x, false, None, false, true).unwrap();
  assert!(vclose(
    &p.to_vec::<f32>().unwrap(),
    &[0.30151135, 0.30151135]
  ));
}

#[test]
fn pool_post_layer_norm_wins_when_both_norm_flags_set() {
  // Both flags → LayerNorm precedence (`if ln … else if rms`). Result
  // must equal the LayerNorm closed-form and NOT the RMSNorm one.
  let layer_norm_expected = [-1.3416354_f32, -0.4472118, 0.4472118, 1.3416354];
  let rms_expected = [0.36514813_f32, 0.73029626, 1.0954444, 1.4605925];
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let mut p = pool_post(x, false, None, true, true).unwrap();
  let got = p.to_vec::<f32>().unwrap();
  assert!(
    vclose(&got, &layer_norm_expected),
    "LayerNorm must win: {got:?}"
  );
  assert!(!vclose(&got, &rms_expected), "must not be RMSNorm: {got:?}");
}

#[test]
fn pool_post_normalize_only_yields_unit_row() {
  // L2 over last axis: [3,4] → /5 → [0.6,0.8].
  let x = Array::from_slice(&[3.0_f32, 4.0], &(1, 2)).unwrap();
  let mut p = pool_post(x, true, None, false, false).unwrap();
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[0.6, 0.8]));
}

#[test]
fn pool_post_truncate_before_normalize_order() {
  // Step order is truncate → L2. Row0 [3,4,99] -trunc2-> [3,4] -L2-> [0.6,0.8];
  // row1 [0,5,12] -trunc2-> [0,5] -L2-> [0,1].
  let x = Array::from_slice(&[3.0_f32, 4.0, 99.0, 0.0, 5.0, 12.0], &(2, 3)).unwrap();
  let mut p = pool_post(x, true, Some(2), false, false).unwrap();
  assert_eq!(p.shape(), vec![2, 2]);
  assert!(vclose(&p.to_vec::<f32>().unwrap(), &[0.6, 0.8, 0.0, 1.0]));
}

// ───────────── truncate_last_dim: rank-1 / rank-2 / rank-3 / no-op ─────────

#[test]
fn truncate_last_dim_rank2_keeps_first_cols() {
  // (2,3) → (2,2) keeping cols 0..2 of each row.
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut t = truncate_last_dim(&x, 2).unwrap();
  assert_eq!(t.shape(), vec![2, 2]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 2.0, 4.0, 5.0]));
}

#[test]
fn truncate_last_dim_rank1() {
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let mut t = truncate_last_dim(&x, 2).unwrap();
  assert_eq!(t.shape(), vec![2]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 2.0]));
}

#[test]
fn truncate_last_dim_rank3_truncates_only_last_axis() {
  // (2,2,2) → (2,2,1) keeping index 0 of the last axis.
  // [[[1,2],[3,4]],[[5,6],[7,8]]] → [[[1],[3]],[[5],[7]]].
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(2, 2, 2)).unwrap();
  let mut t = truncate_last_dim(&x, 1).unwrap();
  assert_eq!(t.shape(), vec![2, 2, 1]);
  assert!(vclose(&t.to_vec::<f32>().unwrap(), &[1.0, 3.0, 5.0, 7.0]));
}

#[test]
fn truncate_last_dim_noop_when_dimension_ge_last() {
  // dimension >= last size → clone unchanged (documented no-op).
  let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut eq = truncate_last_dim(&x, 2).unwrap();
  let mut gt = truncate_last_dim(&x, 5).unwrap();
  assert_eq!(eq.shape(), vec![2, 2]);
  assert_eq!(gt.shape(), vec![2, 2]);
  assert!(vclose(&eq.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]));
  assert!(vclose(&gt.to_vec::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]));
}

// ───────────── PoolingStrategy: as_str / Display / IsVariant / from_mode ──

#[test]
fn pooling_strategy_as_str_and_display_match() {
  for (s, name) in [
    (PoolingStrategy::Mean, "mean"),
    (PoolingStrategy::Cls, "cls"),
    (PoolingStrategy::First, "first"),
    (PoolingStrategy::Last, "last"),
    (PoolingStrategy::Max, "max"),
    (PoolingStrategy::None, "none"),
  ] {
    assert_eq!(s.as_str(), name);
    assert_eq!(format!("{s}"), name, "Display delegates to as_str");
  }
}

#[test]
fn pooling_strategy_is_variant_predicates() {
  assert!(PoolingStrategy::Mean.is_mean());
  assert!(PoolingStrategy::Cls.is_cls());
  assert!(PoolingStrategy::First.is_first());
  assert!(PoolingStrategy::Last.is_last());
  assert!(PoolingStrategy::Max.is_max());
  assert!(PoolingStrategy::None.is_none());
  assert!(!PoolingStrategy::Mean.is_max());
  assert!(!PoolingStrategy::First.is_cls());
}

#[test]
fn pooling_strategy_from_mode_accepts_known_modes_and_last_alias() {
  assert_eq!(
    PoolingStrategy::from_mode("cls").unwrap(),
    PoolingStrategy::Cls
  );
  assert_eq!(
    PoolingStrategy::from_mode("mean").unwrap(),
    PoolingStrategy::Mean
  );
  assert_eq!(
    PoolingStrategy::from_mode("max").unwrap(),
    PoolingStrategy::Max
  );
  assert_eq!(
    PoolingStrategy::from_mode("lasttoken").unwrap(),
    PoolingStrategy::Last
  );
  assert_eq!(
    PoolingStrategy::from_mode("last").unwrap(),
    PoolingStrategy::Last
  );
  assert_eq!(
    PoolingStrategy::from_mode("first").unwrap(),
    PoolingStrategy::First
  );
  assert_eq!(
    PoolingStrategy::from_mode("none").unwrap(),
    PoolingStrategy::None
  );
  // Round-trip: as_str() of every variant re-parses to itself.
  for s in [
    PoolingStrategy::Mean,
    PoolingStrategy::Cls,
    PoolingStrategy::First,
    PoolingStrategy::Last,
    PoolingStrategy::Max,
    PoolingStrategy::None,
  ] {
    assert_eq!(
      PoolingStrategy::from_mode(s.as_str()).unwrap(),
      s,
      "round-trip {s}"
    );
  }
}

#[test]
fn pooling_strategy_from_mode_rejects_unsupported_with_typed_payload() {
  // `weightedmean` / `mean_sqrt_len_tokens` are documented-unsupported;
  // anything else is the catch-all. Both → UnknownEnumValue carrying the
  // type name, the offending value, and the static supported set.
  match PoolingStrategy::from_mode("weightedmean") {
    Err(Error::UnknownEnumValue(p)) => {
      assert_eq!(p.type_name(), "embeddings::PoolingStrategy");
      assert_eq!(p.value(), "weightedmean");
      assert_eq!(p.supported(), &["cls", "lasttoken", "max", "mean"]);
    }
    other => panic!("expected UnknownEnumValue, got {other:?}"),
  }
  assert!(matches!(
    PoolingStrategy::from_mode("xyzzy"),
    Err(Error::UnknownEnumValue(_))
  ));
  assert!(matches!(
    PoolingStrategy::from_mode("mean_sqrt_len_tokens"),
    Err(Error::UnknownEnumValue(_))
  ));
}

// ───────────── eps constants pin the documented call-site defaults ─────────

#[test]
fn eps_constants_match_documented_defaults() {
  assert_eq!(LAYER_NORM_EPS, 1e-5);
  assert_eq!(RMS_NORM_EPS, 1e-5);
}
