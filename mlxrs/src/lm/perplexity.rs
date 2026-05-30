//! Perplexity (PPL) evaluation, ported 1:1 from
//! [`mlx_lm.perplexity`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/perplexity.py)
//! (`eval_ppl` / the `load_data` windowing).
//!
//! Perplexity is the exponentiated mean per-token negative-log-likelihood of a
//! causal language model over a held-out token stream. The reference's
//! `eval_ppl(model, data, batch_size)` takes a pre-windowed `[N, L]` token
//! matrix (built by `load_data`'s `reshape(-1, sequence_length)`), runs the
//! model over each batch's `data[:, :-1]`, scores the next tokens
//! `data[:, 1:]` with cross-entropy (`reduction="none"`), collects every
//! per-token loss, and reports
//!
//! ```text
//! mean_loss = mean(all_losses)
//! ppl       = exp(mean_loss)
//! ```
//!
//! plus a delta-method standard error of the PPL
//! (`std(all_losses, ddof=1) / sqrt(num_tokens) * ppl`).
//!
//! **Windowing (`load_data`).** The reference does *not* stride a long
//! sequence with overlap or carry context between windows: it truncates a flat
//! token stream to a multiple of `sequence_length`, reshapes into
//! non-overlapping `[-1, sequence_length]` rows, and treats each row as an
//! independent sequence. Within a row the first token has no loss (it is the
//! first *input*, predicting positions `1..L`); each row therefore contributes
//! `L - 1` per-token losses. [`make_windows`] mirrors that reshape;
//! [`perplexity`] mirrors `eval_ppl`.
//!
//! **No incremental cache.** `eval_ppl` does a single full forward per batch
//! (`model(batch[:, :-1])`) with no KV cache reused across batches — every
//! batch starts from empty cache state. [`perplexity`] mirrors this by
//! building a fresh cache (via [`make_prompt_cache`]) for each batch from the
//! supplied [`CacheConfig`], exactly as the generation loop sizes its cache to
//! the model's decoder-layer count.
//!
//! **Cross-entropy.** [`cross_entropy_none`] mirrors `nn.losses.cross_entropy`
//! for the class-indices / `reduction="none"` case perplexity uses:
//! `loss = logsumexp(logits, axis=-1) - take_along_axis(logits, targets)`,
//! the numerically stable form of `-log_softmax(logits)[target]`.
//!
//! [`make_prompt_cache`]: crate::lm::cache::make_prompt_cache
//! [`CacheConfig`]: crate::lm::cache::CacheConfig

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    Error, LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload, try_with_capacity,
  },
  lm::{
    cache::{CacheConfig, make_prompt_cache},
    model::Model,
  },
  ops,
};

/// The smallest token window [`perplexity`] / [`make_windows`] can score:
/// scoring the next token needs at least one input token *and* one target, so
/// `L >= 2`. (`eval_ppl` slices `batch[:, :-1]` and `batch[:, 1:]`, both of
/// which are empty for `L < 2`.)
pub const MIN_WINDOW: usize = 2;

/// The default evaluation batch size, matching `mlx_lm.perplexity`'s
/// `--batch-size` / `eval_ppl(batch_size=8)` default.
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// The outcome of a perplexity evaluation, mirroring `eval_ppl`'s reported
/// quantities (it returns `(ppl, standard_error_ppl)`; the per-token losses,
/// their mean, and the token count are the intermediate values it computes and
/// prints).
///
/// `losses` is the flat per-token negative-log-likelihood vector — the
/// reference's `all_losses = mx.concatenate([...])`. Each batch's per-token
/// losses are materialized as they are computed (mirroring `eval_ppl`'s
/// per-batch `mx.eval`, which bounds the lazy graph to one batch); the only
/// residual graph node is the final `concatenate` over those already-evaluated
/// batches (a single batch is returned fully evaluated). The scalar fields are
/// likewise materialized (the reference calls `.item()` on each).
pub struct PerplexityResult {
  /// `exp(mean_loss)` — the perplexity (`eval_ppl`'s `ppl`).
  pub perplexity: f32,
  /// Delta-method standard error of [`Self::perplexity`]
  /// (`eval_ppl`'s `standard_error_ppl = ppl * std/sqrt(n)`).
  pub std_error: f32,
  /// Mean per-token negative-log-likelihood (`all_losses.mean()`).
  pub mean_loss: f32,
  /// Total number of scored tokens — `sum over rows of (L - 1)`
  /// (`all_losses.size`).
  pub num_tokens: usize,
  /// The flat `[num_tokens]` per-token NLL vector (`all_losses`); its
  /// per-batch constituents are already evaluated (see the type-level note).
  pub losses: Array,
}

/// Reshape a flat token stream into the non-overlapping `[N, L]` window matrix
/// `eval_ppl` consumes, mirroring `mlx_lm.perplexity.load_data`'s
/// `data[: (len // L) * L].reshape(-1, L)`.
///
/// `tokens` is truncated to the largest multiple of `sequence_length` and
/// reshaped into rows of length `sequence_length`; the trailing
/// `len % sequence_length` tokens are dropped (exactly the reference). The
/// result has `len(tokens) / sequence_length` rows.
///
/// Errors with [`Error::OutOfRange`] if `sequence_length < MIN_WINDOW`
/// (a row must hold at least one input + one target) or if `tokens` is too
/// short to fill a single window.
pub fn make_windows(tokens: &[i32], sequence_length: usize) -> Result<Array> {
  if sequence_length < MIN_WINDOW {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "perplexity::make_windows: sequence_length",
      "must be >= MIN_WINDOW (one input + one target)",
      format_smolstr!("{sequence_length} (MIN_WINDOW={MIN_WINDOW})"),
    )));
  }
  let num_rows = tokens.len() / sequence_length;
  if num_rows == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "perplexity::make_windows: tokens.len()",
      "must be >= sequence_length to fill one window",
      format_smolstr!("{} (sequence_length={sequence_length})", tokens.len()),
    )));
  }
  // `data[: (len // L) * L]` — drop the ragged tail, then `reshape(-1, L)`.
  let kept = num_rows * sequence_length;
  Array::from_slice::<i32>(&tokens[..kept], &(num_rows, sequence_length))
}

/// Cross-entropy loss for integer class-index `targets` with
/// `reduction="none"`, mirroring `mlx.nn.losses.cross_entropy` over `axis=-1`
/// (the only case `eval_ppl` exercises: `cross_entropy(logits, targets,
/// reduction="none")`).
///
/// `logits` is `[..., V]` (any leading shape) and `targets` must be **exactly**
/// that shape with the last axis dropped (`[...]`, an integer dtype). Returns
/// the per-position loss `[...]`:
///
/// ```text
/// score = take_along_axis(logits, targets[..., None], -1).squeeze(-1)
/// loss  = logsumexp(logits, -1) - score
/// ```
///
/// which is the numerically stable `-log_softmax(logits)[target]` the
/// reference relies on (it never forms `log_softmax` explicitly). Label
/// smoothing / weights / non-`none` reductions are out of scope here — the
/// reference perplexity uses none of them.
///
/// Errors with [`Error::RankMismatch`] if `logits` is rank-0, with
/// [`Error::LengthMismatch`] if `targets.ndim()` does not equal
/// `logits.ndim() - 1`, or with [`Error::ShapePairMismatch`] if the full
/// shape of `targets` does not equal `logits.shape()` with the class axis
/// removed — mirroring mlx's `cross_entropy`
/// (`targets.shape != _drop_dim(logits.shape, axis)` raises). A merely
/// broadcastable target shape (e.g. `[B, 1]` against `[B, S, V]`) is
/// **rejected**, not silently broadcast across the missing positions.
pub fn cross_entropy_none(logits: &Array, targets: &Array) -> Result<Array> {
  let logits_ndim = logits.ndim();
  if logits_ndim == 0 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "perplexity::cross_entropy_none: logits must have a vocab axis (ndim >= 1)",
      0,
      Vec::new(),
    )));
  }
  // Class indices: `targets.ndim == logits.ndim - 1` (mlx-lm checks
  // `targets.shape == logits.shape with axis removed`).
  if targets.ndim() != logits_ndim - 1 {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "perplexity::cross_entropy_none: targets ndim (must equal logits ndim - 1 for class-index targets)",
      logits_ndim - 1,
      targets.ndim(),
    )));
  }
  let axis = (logits_ndim - 1) as i32;
  // mlx's `cross_entropy` rejects targets whose shape isn't *exactly* the logits
  // shape with the class axis removed (`targets.shape != _drop_dim(logits.shape,
  // axis)`). The rank check above isn't enough: `take_along_axis` + `subtract`
  // broadcast, so e.g. `logits [B, S, V]` with targets `[B, 1]` would silently
  // reuse one label across all `S` positions instead of erroring. Compare the
  // full shape (class axis = last, so logits shape sans its final entry) BEFORE
  // `expand_dims`/`take_along_axis` so a broadcastable-but-wrong shape is
  // rejected, not silently broadcast.
  let logits_shape = logits.shape();
  let expected = &logits_shape[..logits_ndim - 1];
  let targets_shape = targets.shape();
  if targets_shape.as_slice() != expected {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      "perplexity::cross_entropy_none: targets must equal logits with the class axis removed",
      expected.to_vec(),
      targets_shape.to_vec(),
    )));
  }
  // `mx.take_along_axis(logits, mx.expand_dims(targets, axis), axis).squeeze(axis)`.
  let idx = targets.expand_dims_axes(&[axis])?;
  let score = ops::indexing::take_along_axis(logits, &idx, axis)?;
  let score = score.squeeze_axes(&[axis])?;
  // `loss = logsumexp(logits, axis) - score` (the `label_smoothing == 0` branch).
  let lse = ops::reduction::logsumexp_axes(logits, &[axis], false)?;
  ops::arithmetic::subtract(&lse, &score)
}

/// Evaluate the perplexity of `model` on the pre-windowed token matrix `data`,
/// mirroring `mlx_lm.perplexity.eval_ppl(model, data, batch_size)`.
///
/// `data` is a `[N, L]` integer array of `N` independent length-`L` windows
/// (build it with [`make_windows`]). For each `batch_size`-row batch the model
/// runs once over `data[:, :-1]`, its logits are scored against the next
/// tokens `data[:, 1:]` with [`cross_entropy_none`], and the resulting
/// per-token losses are **evaluated and collected per batch** (mirroring
/// `eval_ppl`'s per-batch `mx.eval`, so the lazy compute graph stays bounded to
/// one batch rather than growing with `N`); then
///
/// ```text
/// mean_loss = mean(all_losses)
/// ppl       = exp(mean_loss)
/// se_ppl    = ppl * std(all_losses, ddof=1) / sqrt(num_tokens)
/// ```
///
/// exactly as the reference (which casts the logits to `float32` before the
/// loss — mirrored here so the reduction matches regardless of the model's
/// compute dtype). Each batch gets a **fresh** cache from `cache_config`
/// (`eval_ppl` reuses no cache state across batches); pass the model's decoder
/// layer count via [`CacheConfig::num_hidden_layers`].
///
/// `batch_size` is clamped to at least 1. Errors with
/// [`Error::RankMismatch`] if `data` is not rank-2, or with
/// [`Error::OutOfRange`] if its window length is `< MIN_WINDOW`.
///
/// [`CacheConfig::num_hidden_layers`]: crate::lm::cache::CacheConfig::num_hidden_layers
pub fn perplexity<M: Model>(
  model: &M,
  data: &Array,
  batch_size: usize,
  cache_config: &CacheConfig,
) -> Result<PerplexityResult> {
  let shape = data.shape();
  let (num_rows, seq_len) = match shape.as_slice() {
    [n, l] => (*n, *l),
    other => {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "perplexity: data must be a rank-2 [N, L] token matrix",
        other.len() as u32,
        other.to_vec(),
      )));
    }
  };
  if seq_len < MIN_WINDOW {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "perplexity: window length (must hold one input + one target)",
      "must be >= MIN_WINDOW",
      format_smolstr!("{seq_len} (MIN_WINDOW={MIN_WINDOW})"),
    )));
  }
  // mlx-lm clamps nothing but a 0 batch size would loop forever; treat it as 1.
  let batch_size = batch_size.max(1);

  // Per-batch losses, concatenated at the end (mlx-lm: `all_losses` list ->
  // `mx.concatenate`). Bounded `Vec` (one entry per batch).
  let num_batches = num_rows.div_ceil(batch_size);
  let mut all_losses: Vec<Array> = try_with_capacity(num_batches)?;

  let mut start = 0usize;
  while start < num_rows {
    let stop = (start + batch_size).min(num_rows);
    // `batch = data[s : s + batch_size]` (row slice).
    let batch = ops::indexing::slice(
      data,
      &[start as i32, 0],
      &[stop as i32, seq_len as i32],
      &[1, 1],
    )?;

    // `inputs = batch[:, :-1]`, `targets = batch[:, 1:]`.
    let rows = (stop - start) as i32;
    let inputs = ops::indexing::slice(&batch, &[0, 0], &[rows, (seq_len - 1) as i32], &[1, 1])?;
    let targets = ops::indexing::slice(&batch, &[0, 1], &[rows, seq_len as i32], &[1, 1])?;

    // `logits = model(inputs).astype(float32)`. Fresh cache per batch — the
    // reference reuses no cache state between batches (a single full forward).
    let mut cache = make_prompt_cache(cache_config);
    let logits = model.forward(&inputs, &mut cache)?;
    let logits = logits.astype(Dtype::F32)?;

    // `losses = cross_entropy(logits, targets, reduction="none").flatten()`.
    let losses = cross_entropy_none(&logits, &targets)?;
    let mut losses = losses.flatten(0, -1)?;
    // mlx-lm's `eval_ppl` calls `mx.eval(losses)` each batch — materialize the
    // per-token NLLs now instead of accumulating lazy graphs and eval'ing one
    // giant graph at the end. This bounds the lazy compute graph to a single
    // batch (memory + incremental compute); the accumulated values are exactly
    // the per-token losses, so eval'ing them per batch is correct and faithful.
    losses.eval()?;
    all_losses.push(losses);

    start = stop;
  }

  // `all_losses = mx.concatenate(all_losses)`.
  let losses = if all_losses.len() == 1 {
    // Single batch: avoid a no-op concatenate (and its alloc).
    all_losses.into_iter().next().expect("len checked == 1")
  } else {
    let refs: Vec<&Array> = all_losses.iter().collect();
    ops::shape::concatenate(&refs, 0)?
  };

  // `mean_loss = all_losses.mean().item()`; `ppl = exp(mean_loss)`.
  let mut mean_loss_arr = ops::reduction::mean(&losses, false)?;
  let mean_loss: f32 = mean_loss_arr.item::<f32>()?;
  let perplexity = mean_loss.exp();

  // `std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()`;
  // `num_tokens = all_losses.size`;
  // `standard_error = std_dev / sqrt(num_tokens)`;
  // `standard_error_ppl = ppl * standard_error`.
  let num_tokens = losses.size();
  let mut var_arr = ops::reduction::var(&losses, false, 1)?;
  let std_dev: f32 = var_arr.item::<f32>()?.sqrt();
  let standard_error = std_dev / (num_tokens as f32).sqrt();
  let std_error = perplexity * standard_error;

  Ok(PerplexityResult {
    perplexity,
    std_error,
    mean_loss,
    num_tokens,
    losses,
  })
}

#[cfg(test)]
mod tests {
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
    // Finding 1: per-batch `eval` materializes each batch's losses incrementally
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
}
