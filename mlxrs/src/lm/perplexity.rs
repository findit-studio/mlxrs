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
mod tests;
