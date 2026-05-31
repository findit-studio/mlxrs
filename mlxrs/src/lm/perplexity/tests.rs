//! L8 tests — perplexity over the deterministic [`MockModel`] fixture.
//!
//! The reference identity under test is
//! `ppl == exp(mean(logsumexp(logits, -1) - logits[target]))` over the
//! next-token targets of each window, with non-overlapping windowing and
//! the per-batch concatenation matching `eval_ppl`.

use super::*;
use crate::lm::{cache::KvCache, model::MockModel};

/// A `CacheConfig` with no layers — the `MockModel` ignores cache content
/// (it only advances whatever entries it is handed), so a single full
/// forward needs none. (`make_prompt_cache` returns an empty `Vec`.)
fn no_cache() -> CacheConfig {
  CacheConfig {
    num_hidden_layers: 0,
    sliding_window: None,
  }
}

/// Independently compute `exp(mean over targets of (logsumexp(row) -
/// row[target]))` from the raw per-vocab logits and the `[N, L]` token
/// matrix — the hand-traced ground truth for [`perplexity`]. `canned` is
/// the per-vocab logit row the `MockModel` tiles at every position.
fn expected_ppl(canned: &[f32], windows: &[&[i32]]) -> f64 {
  // logsumexp of the (constant) logit row, in f64 for a tight reference.
  let max = canned.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
  let sumexp: f64 = canned.iter().map(|&x| (x as f64 - max).exp()).sum();
  let lse = max + sumexp.ln();

  let mut total = 0.0f64;
  let mut n = 0usize;
  for row in windows {
    // Targets are `row[1..]`; each predicted from the constant logit row.
    for &t in &row[1..] {
      let score = canned[t as usize] as f64;
      total += lse - score;
      n += 1;
    }
  }
  (total / n as f64).exp()
}

fn matrix(rows: &[&[i32]]) -> Array {
  let n = rows.len();
  let l = rows[0].len();
  let mut data: Vec<i32> = Vec::with_capacity(n * l);
  for r in rows {
    assert_eq!(r.len(), l, "ragged test matrix");
    data.extend_from_slice(r);
  }
  Array::from_slice::<i32>(&data, &(n, l)).unwrap()
}

#[test]
fn hand_traced_single_window_matches_reference() {
  // vocab 5; the MockModel tiles logits [0, 1, 2, 3, 4] at every position.
  let model = MockModel::new(5);
  let row: &[i32] = &[0, 1, 2, 3, 4];
  let data = matrix(&[row]);

  let res = perplexity(&model, &data, 8, &no_cache()).unwrap();

  // 4 scored tokens (L - 1) in one window.
  assert_eq!(res.num_tokens, 4);
  let want = expected_ppl(&model.canned, &[row]);
  assert!(
    (res.perplexity as f64 - want).abs() < 1e-4,
    "ppl {} != hand-traced {want}",
    res.perplexity
  );
  // mean_loss is the log of the perplexity.
  assert!((res.mean_loss as f64 - want.ln()).abs() < 1e-4);
  // Per-token losses are materializable and have the right count.
  let mut losses = res.losses;
  assert_eq!(losses.to_vec::<f32>().unwrap().len(), 4);
}

#[test]
fn uniform_logits_ppl_approximates_vocab_size() {
  // A model emitting UNIFORM logits over V classes has, for every target,
  // -log_softmax = log V, so ppl == V exactly regardless of the targets.
  for vocab in [2usize, 7, 50] {
    // Uniform logits: all-equal canned row (value is irrelevant; use 0).
    let model = MockModel {
      canned: vec![0.0; vocab],
      n_kv_heads: 1,
      head_dim: 2,
    };
    // A window touching a spread of targets; the result is target-independent.
    let row: Vec<i32> = (0..vocab as i32).collect();
    let data = matrix(&[&row]);

    let res = perplexity(&model, &data, 4, &no_cache()).unwrap();
    assert!(
      (res.perplexity as f64 - vocab as f64).abs() < 1e-3,
      "uniform vocab {vocab}: ppl {} != V",
      res.perplexity
    );
  }
}

#[test]
fn multi_window_batched_matches_unbatched_aggregation() {
  // Several windows + a batch size that splits them across >1 batch must
  // aggregate identically to a single concatenated reduction. vocab 6.
  let model = MockModel::new(6);
  let rows: Vec<&[i32]> = vec![
    &[0, 1, 2, 3],
    &[5, 4, 3, 2],
    &[1, 1, 5, 0],
    &[2, 3, 4, 5],
    &[5, 5, 5, 5],
  ];
  let data = matrix(&rows);

  // batch_size 2 -> batches of [2, 2, 1] rows; must equal the closed form.
  let res = perplexity(&model, &data, 2, &no_cache()).unwrap();
  // 5 windows * (4 - 1) = 15 scored tokens.
  assert_eq!(res.num_tokens, 15);
  let want = expected_ppl(&model.canned, &rows);
  assert!(
    (res.perplexity as f64 - want).abs() < 1e-4,
    "batched ppl {} != hand-traced {want}",
    res.perplexity
  );

  // And batch_size covering everything in one shot gives the same answer.
  let res_one = perplexity(&model, &data, 64, &no_cache()).unwrap();
  assert!((res.perplexity as f64 - res_one.perplexity as f64).abs() < 1e-5);
}

#[test]
fn batch_size_zero_is_treated_as_one() {
  let model = MockModel::new(4);
  let data = matrix(&[&[0, 1, 2, 3], &[3, 2, 1, 0]]);
  // batch_size 0 must not loop forever; clamps to 1 and still aggregates.
  let res = perplexity(&model, &data, 0, &no_cache()).unwrap();
  assert_eq!(res.num_tokens, 6);
  let want = expected_ppl(&model.canned, &[&[0, 1, 2, 3], &[3, 2, 1, 0]]);
  assert!((res.perplexity as f64 - want).abs() < 1e-4);
}

#[test]
fn rejects_non_rank2_data() {
  let model = MockModel::new(4);
  // A rank-1 token vector is not the expected [N, L] matrix.
  let flat = Array::from_slice::<i32>(&[0, 1, 2, 3], &(4usize,)).unwrap();
  assert!(perplexity(&model, &flat, 8, &no_cache()).is_err());
}

#[test]
fn rejects_too_short_window() {
  let model = MockModel::new(4);
  // L == 1: no next-token target exists.
  let data = Array::from_slice::<i32>(&[0, 1], &(2usize, 1)).unwrap();
  assert!(perplexity(&model, &data, 8, &no_cache()).is_err());
}

#[test]
fn make_windows_drops_ragged_tail_and_reshapes() {
  // 7 tokens, window 3 -> 2 rows (6 tokens), 1 dropped.
  let toks: Vec<i32> = (0..7).collect();
  let mut windows = make_windows(&toks, 3).unwrap();
  assert_eq!(windows.shape(), vec![2, 3]);
  assert_eq!(windows.to_vec::<i32>().unwrap(), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn make_windows_rejects_short_input_and_tiny_window() {
  // Fewer tokens than one window.
  assert!(make_windows(&[0, 1], 5).is_err());
  // Window below MIN_WINDOW.
  assert!(make_windows(&[0, 1, 2, 3], 1).is_err());
}

#[test]
fn cross_entropy_none_matches_logsumexp_minus_score() {
  // A tiny [2, 3] logits / [2] targets case checked by hand. Mirrors the
  // mlx docstring example shape (batch of rows, class-index targets).
  let logits = Array::from_slice::<f32>(&[2.0, -1.0, 0.0, -1.0, 2.0, 0.5], &(2usize, 3)).unwrap();
  let targets = Array::from_slice::<i32>(&[0, 1], &(2usize,)).unwrap();
  let mut loss = cross_entropy_none(&logits, &targets).unwrap();
  let got = loss.to_vec::<f32>().unwrap();

  // Row 0: logsumexp([2,-1,0]) - 2 ; Row 1: logsumexp([-1,2,0.5]) - 2.
  let lse = |r: &[f64]| {
    let m = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + r.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
  };
  let want0 = lse(&[2.0, -1.0, 0.0]) - 2.0;
  let want1 = lse(&[-1.0, 2.0, 0.5]) - 2.0;
  assert!((got[0] as f64 - want0).abs() < 1e-5);
  assert!((got[1] as f64 - want1).abs() < 1e-5);
}

#[test]
fn cross_entropy_none_rejects_target_rank_mismatch() {
  let logits = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
  // targets with the SAME rank as logits is the (unsupported) probs case.
  let bad = Array::from_slice::<i32>(&[0, 1, 0, 1], &(2usize, 2)).unwrap();
  assert!(cross_entropy_none(&logits, &bad).is_err());
}

#[test]
fn cross_entropy_none_rejects_broadcastable_target_shape() {
  // The right rank but a non-matching (broadcastable) shape must be rejected,
  // not silently broadcast — `take_along_axis`/`subtract` would otherwise reuse
  // one label across every position. logits `[B=2, S=3, V=4]`.
  let b = 2usize;
  let s = 3usize;
  let v = 4usize;
  let logits = Array::from_slice::<f32>(&vec![0.0f32; b * s * v], &(b, s, v)).unwrap();

  // targets `[B, 1]` — class-index rank (ndim 2 == 3 - 1) but broadcasts over S.
  let bad_bs = Array::from_slice::<i32>(&[0, 1], &(b, 1usize)).unwrap();
  let err_bs = cross_entropy_none(&logits, &bad_bs)
    .expect_err("targets [B, 1] should be rejected, not broadcast across S");
  match err_bs {
    Error::ShapePairMismatch(payload) => {
      assert_eq!(payload.expected(), &[b, s][..]);
      assert_eq!(payload.actual(), &[b, 1][..]);
    }
    other => panic!("expected ShapePairMismatch, got: {other:?}"),
  }

  // targets `[1, S]` — also right rank, broadcasts over B.
  let bad_1s = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, s)).unwrap();
  let err_1s = cross_entropy_none(&logits, &bad_1s)
    .expect_err("targets [1, S] should be rejected, not broadcast across B");
  match err_1s {
    Error::ShapePairMismatch(payload) => {
      assert_eq!(payload.expected(), &[b, s][..]);
      assert_eq!(payload.actual(), &[1, s][..]);
    }
    other => panic!("expected ShapePairMismatch, got: {other:?}"),
  }

  // The exact-shape targets `[B, S]` are accepted.
  let good = Array::from_slice::<i32>(&[0, 1, 2, 3, 0, 1], &(b, s)).unwrap();
  let mut loss = cross_entropy_none(&logits, &good).unwrap();
  // Uniform-zero logits over V classes -> every loss is `log V`.
  let got = loss.to_vec::<f32>().unwrap();
  assert_eq!(got.len(), b * s);
  let want = (v as f64).ln();
  for x in got {
    assert!((x as f64 - want).abs() < 1e-5, "loss {x} != log V {want}");
  }
}

#[test]
fn many_batches_match_single_batch_after_per_batch_eval() {
  // Per-batch `eval` materializes each batch's losses incrementally
  // (bounding the lazy graph) without changing the result. Drive *many* small
  // batches (batch_size 1 over a tall matrix) and assert the PPL equals both
  // the hand-traced closed form and the all-in-one-batch reduction.
  let model = MockModel::new(6);
  let rows: Vec<&[i32]> = vec![
    &[0, 1, 2, 3, 4],
    &[5, 4, 3, 2, 1],
    &[1, 2, 3, 4, 5],
    &[0, 0, 5, 5, 0],
    &[2, 4, 1, 3, 5],
    &[5, 5, 0, 0, 5],
    &[3, 3, 3, 3, 3],
    &[1, 0, 2, 4, 1],
  ];
  let data = matrix(&rows);

  // batch_size 1 -> 8 separate batches, each eval'd before the next.
  let res = perplexity(&model, &data, 1, &no_cache()).unwrap();
  assert_eq!(res.num_tokens, rows.len() * (5 - 1));
  let want = expected_ppl(&model.canned, &rows);
  assert!(
    (res.perplexity as f64 - want).abs() < 1e-4,
    "many-batch ppl {} != hand-traced {want}",
    res.perplexity
  );

  // The incremental per-batch eval must not change the answer vs one big batch.
  let res_one = perplexity(&model, &data, 64, &no_cache()).unwrap();
  assert!((res.perplexity as f64 - res_one.perplexity as f64).abs() < 1e-5);
  assert!((res.mean_loss as f64 - res_one.mean_loss as f64).abs() < 1e-5);
  assert_eq!(res.num_tokens, res_one.num_tokens);
}

/// A `MockModel` with a non-`f32` compute dtype must still reduce in f32
/// (the reference `.astype(mx.float32)`s the logits before the loss). This
/// drives a model whose logits are emitted as `f16` to exercise the cast.
#[test]
fn logits_cast_to_f32_before_loss() {
  struct F16Model {
    canned: Vec<f32>,
  }
  impl Model for F16Model {
    fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let (batch, seq) = match tokens.shape().as_slice() {
        [b, s] => (*b, *s),
        other => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "F16Model::forward: tokens must be rank-2 [B, S]",
            other.len() as u32,
            other.to_vec(),
          )));
        }
      };
      let vocab = self.canned.len();
      let mut data = Vec::with_capacity(batch * seq * vocab);
      for _ in 0..batch * seq {
        data.extend_from_slice(&self.canned);
      }
      let f32_logits = Array::from_slice::<f32>(&data, &(batch, seq, vocab))?;
      // Emit in f16 so `perplexity`'s `.astype(F32)` is load-bearing.
      f32_logits.astype(Dtype::F16)
    }
  }

  let model = F16Model {
    canned: vec![0.0, 1.0, 2.0, 3.0],
  };
  let row: &[i32] = &[0, 1, 2, 3];
  let data = matrix(&[row]);
  let res = perplexity(&model, &data, 8, &no_cache()).unwrap();
  // f16 can represent these small integer logits exactly, so the f32
  // reference holds to a loose tolerance.
  let want = expected_ppl(&model.canned, &[row]);
  assert!(
    (res.perplexity as f64 - want).abs() < 1e-2,
    "f16-model ppl {} != {want}",
    res.perplexity
  );
}
