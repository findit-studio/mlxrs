//! Logits-transform sampling utilities, ported from
//! [`mlx_lm.sample_utils`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)
//! (the primary spec) and cross-checked against mlx-swift-lm's
//! `MLXLMCommon` samplers.
//!
//! The transforms operate on **log-probabilities** (`log_softmax` output
//! over the last `[..., vocab]` axis) — exactly as the `mlx_lm.sample_utils`
//! `apply_*` helpers do; the `log_softmax`/normalization is the caller's or
//! the (deferred) `make_sampler` composition's job (mirroring mlx-swift's
//! `TopPSampler`, which `logSoftmax`es before the per-transform helpers).
//! Each is a pure composition of [`crate::ops`] returning a `Result<Array>`
//! — no implicit eval.
//!
//! The three logits transforms ([`apply_top_k`], [`apply_top_p`],
//! [`apply_min_p`]) mask filtered tokens to `-inf` so a subsequent
//! [`categorical_sampling`] never draws them. Sampler chaining (mlx-lm's
//! `make_sampler`) is left to the caller — the generation loop is out of
//! scope for M3 sampling.

use crate::{
  array::Array,
  error::{Error, Result},
  ops,
};

/// Build a 1-element scalar array **in `like`'s dtype** (for broadcast
/// operands like `-inf`, `1/temp`, `log(min_p)`, `1 - top_p`). mlx broadcasts
/// a `[1]` array against any shape.
///
/// This mirrors mlx-lm's *weak* Python scalars (and its explicit
/// `mx.array(-inf, logprobs.dtype)`), which adopt the operand dtype: a
/// concrete f32 array would instead promote `select`/`put_along_axis`/
/// arithmetic on f16/bf16 `logprobs` up to f32, diverging from the
/// reference's *in-dtype* mask/threshold precision (not just dtype
/// metadata). Twin of the embeddings module's `scalar_like`, duplicated so
/// this module stays self-contained.
fn scalar_like(value: f32, like: &Array) -> Result<Array> {
  // `Array::full` runs the fallible `mlx_array_new_float32` ctor BEFORE its
  // `mlx_full` call (whose `default_stream()` arg is what triggers
  // `ensure_handler_installed()`), so with the eager `#[ctor]` stripped that
  // first ctor could reach mlx-c with no error handler installed → its
  // default `printf + exit(-1)` instead of a recoverable `Err`. Install at
  // the entry point, before any fallible scalar construction — the same
  // defense-in-depth as `embeddings::scalar_like` (Copilot 4307622782 C2)
  // per the #13/#24 crate-wide error-handler contract.
  crate::error::ensure_handler_installed();
  ops::misc::astype(&Array::full::<f32>(&(1,), value)?, like.dtype()?)
}

/// Slice the last axis to `[start, end)`, keeping every other axis full.
/// mlx-lm's `[..., k:]` indexing expressed via [`ops::indexing::slice`].
fn slice_last_axis(a: &Array, start: i32, end: i32) -> Result<Array> {
  let shape = a.shape();
  let ndim = shape.len();
  let mut starts = vec![0i32; ndim];
  let mut stops: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
  let strides = vec![1i32; ndim];
  if ndim > 0 {
    starts[ndim - 1] = start;
    stops[ndim - 1] = end;
  }
  ops::indexing::slice(a, &starts, &stops, &strides)
}

/// Sample from only the top `k` tokens ranked by probability.
///
/// Port of `mlx_lm.sample_utils.apply_top_k`: the `vocab - k` lowest-ranked
/// logits (found via `argpartition(-logprobs, k-1)[..., k:]`) are scattered
/// to `-inf`.
///
/// `top_k` must be in the open interval `(0, vocab_size)`.
pub fn apply_top_k(logprobs: &Array, top_k: i32) -> Result<Array> {
  let vocab_size = *logprobs.shape().last().unwrap_or(&0) as i32;
  if top_k <= 0 || top_k >= vocab_size {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`top_k` has to be an integer in the (0, {vocab_size}) interval, but is {top_k}"
      ),
    });
  }
  let neg = ops::arithmetic::negative(logprobs)?;
  let part = ops::misc::argpartition_axis(&neg, top_k - 1, -1)?;
  let mask_idx = slice_last_axis(&part, top_k, vocab_size)?;
  let neg_inf = scalar_like(f32::NEG_INFINITY, logprobs)?;
  ops::indexing::put_along_axis(logprobs, &mask_idx, &neg_inf, -1)
}

/// Apply min-p sampling: keep only tokens whose probability is at least
/// `min_p` times the top token's probability.
///
/// Port of `mlx_lm.sample_utils.apply_min_p`. Working in log space, the
/// threshold is `max(logprobs) + log(min_p)`; tokens below it become `-inf`.
/// `min_tokens_to_keep` (>= 1) tokens are always preserved.
///
/// `min_p` must be in `[0, 1]` and `min_tokens_to_keep` must be in
/// `[1, vocab_size]` (matching mlx-lm, which errors on a larger value).
pub fn apply_min_p(logprobs: &Array, min_p: f32, min_tokens_to_keep: i32) -> Result<Array> {
  if !(0.0..=1.0).contains(&min_p) {
    return Err(Error::ShapeMismatch {
      message: format!("`min_p` has to be a float in the [0, 1] interval, but is {min_p}"),
    });
  }
  if min_tokens_to_keep < 1 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
      ),
    });
  }
  let vocab_size = *logprobs.shape().last().unwrap_or(&0) as i32;
  if min_tokens_to_keep > vocab_size {
    // mlx-lm passes `kth = -min_tokens_to_keep` to argpartition, which is
    // out of range for `min_tokens_to_keep > vocab_size` and errors there.
    // Without this guard our pre-normalized `vocab_size - min_tokens_to_keep`
    // goes negative, MLX re-normalizes it, and we silently over-prune below
    // the documented keep guarantee — so reject it up front instead.
    return Err(Error::ShapeMismatch {
      message: format!(
        "`min_tokens_to_keep` ({min_tokens_to_keep}) must not exceed the vocabulary size ({vocab_size})"
      ),
    });
  }

  let top_logprobs = ops::reduction::max_axes(logprobs, &[-1], true)?;
  let scaled_min_p = ops::arithmetic::add(&top_logprobs, &scalar_like(min_p.ln(), &top_logprobs)?)?;
  let mut tokens_to_remove = ops::comparison::less(logprobs, &scaled_min_p)?;

  if min_tokens_to_keep > 1 {
    let part = ops::misc::argpartition_axis(logprobs, vocab_size - min_tokens_to_keep, -1)?;
    let top_indices = slice_last_axis(&part, vocab_size - min_tokens_to_keep, vocab_size)?;
    let keep = Array::full::<bool>(&(1,), 0.0)?;
    tokens_to_remove = ops::indexing::put_along_axis(&tokens_to_remove, &top_indices, &keep, -1)?;
  }

  let neg_inf = scalar_like(f32::NEG_INFINITY, logprobs)?;
  ops::logical::select(&tokens_to_remove, &neg_inf, logprobs)
}

/// Apply top-p (nucleus) sampling: keep the smallest set of tokens whose
/// cumulative probability mass exceeds `top_p`.
///
/// `logprobs` must be **log-probabilities** (`log_softmax` output), exactly
/// as `mlx_lm.sample_utils.apply_top_p` expects — then `exp(logprobs)` sums
/// to 1 and the `1 - top_p` cumulative threshold is meaningful. The
/// `log_softmax` is the caller's / (deferred) `make_sampler` composition's
/// responsibility (mirroring mlx-swift's `TopPSampler`). `top_p` must be
/// finite in `(0, 1]` (`1.0` keeps everything).
///
/// Port of `mlx_lm.sample_utils.apply_top_p`: `exp(logprobs)` → ascending
/// argsort → cumulative sum → scatter back to original order → mask tokens
/// whose cumulative prob is `<= 1 - top_p` to `-inf`.
pub fn apply_top_p(logprobs: &Array, top_p: f32) -> Result<Array> {
  if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
    return Err(Error::ShapeMismatch {
      message: format!("`top_p` has to be a float in the (0, 1] interval, but is {top_p}"),
    });
  }
  let probs = ops::arithmetic::exp(logprobs)?;
  let sorted_indices = ops::misc::argsort_axis(logprobs, -1)?;
  let sorted_probs = ops::indexing::take_along_axis(&probs, &sorted_indices, -1)?;
  let cumulative_probs = ops::misc::cumsum(&sorted_probs, -1, false, true)?;

  // Rearrange cumulative probs back to the original token order. The inverse
  // of the `sorted_indices` permutation is `argsort(sorted_indices)`, computed
  // entirely in the integer index dtype — EXACT for any vocab size. (The
  // earlier `arange(0, n) as f32` index build aliased token indices above
  // 2^24, where consecutive integers are no longer f32-representable.)
  let inverse_indices = ops::misc::argsort_axis(&sorted_indices, -1)?;
  let cumulative_probs = ops::indexing::take_along_axis(&cumulative_probs, &inverse_indices, -1)?;

  let threshold = scalar_like(1.0 - top_p, &cumulative_probs)?;
  let keep = ops::comparison::greater(&cumulative_probs, &threshold)?;
  let neg_inf = scalar_like(f32::NEG_INFINITY, logprobs)?;
  ops::logical::select(&keep, logprobs, &neg_inf)
}

/// Temperature-scaled categorical draw.
///
/// Port of `mlx_lm.sample_utils.categorical_sampling`:
/// `random.categorical(logits / temp)` along the last axis. `temp` must be a
/// finite positive float. mlx-lm only ever calls this with `temp > 0` (its
/// `make_sampler` dispatches `temp == 0` to argmax); since that
/// `make_sampler` dispatch is deferred here, this validates the precondition
/// the reference relies on instead of producing `inf`/`NaN` logits. Use
/// [`argmax_sample`] for greedy / `temp == 0` decoding.
pub fn categorical_sampling(logits: &Array, temp: f32, key: &Array) -> Result<Array> {
  if !temp.is_finite() || temp <= 0.0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`temp` has to be a finite positive float (use `argmax_sample` for temperature-0 / greedy decoding), but is {temp}"
      ),
    });
  }
  let scaled = ops::arithmetic::multiply(logits, &scalar_like(1.0 / temp, logits)?)?;
  ops::random::categorical(&scaled, -1, key)
}

/// Greedy (argmax) token selection along the last axis — the
/// temperature-0 branch of `mlx_lm.sample_utils.make_sampler`.
pub fn argmax_sample(logits: &Array) -> Result<Array> {
  ops::misc::argmax(logits, Some(-1), false)
}
