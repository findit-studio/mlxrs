//! Logits-transform sampling utilities, ported from
//! [`mlx_lm.sample_utils`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)
//! (the primary spec) and cross-checked against mlx-swift-lm's
//! `MLXLMCommon` samplers.
//!
//! This module has **two input domains**, matching mlx-lm exactly — mixing
//! them is a correctness bug:
//!
//! * **Samplers** — [`apply_top_k`], [`apply_top_p`], [`apply_min_p`],
//!   [`apply_xtc`], [`categorical_sampling`], [`argmax_sample`] — operate on
//!   **log-probabilities** (`log_softmax` output over the last `[..., vocab]`
//!   axis), exactly as `mlx_lm.sample_utils`'s `apply_top_k`/`apply_min_p`/
//!   `apply_xtc` and `make_sampler` do (cross-checked against mlx-swift's
//!   `TopPSampler`, which `logSoftmax`es first). The `log_softmax` is the
//!   caller's or the deferred `make_sampler`'s job.
//! * **Logits processors** — [`apply_repetition_penalty`],
//!   [`apply_presence_penalty`], [`apply_frequency_penalty`],
//!   [`apply_logit_bias`] — operate on **raw logits**, exactly as mlx-lm's
//!   `make_logits_processors` closures do: in `generate_step` they run
//!   *before* `logprobs = logits - logsumexp(logits)`. Applying them to
//!   normalized log-probs changes behavior — e.g.
//!   `apply_repetition_penalty`'s `logit < 0` sign branch is meaningful only
//!   on raw (mixed-sign) logits, never on all-negative log-probs.
//!
//! Each is a pure composition of [`crate::ops`] returning a `Result<Array>`
//! — no implicit eval.
//!
//! The three logits transforms ([`apply_top_k`], [`apply_top_p`],
//! [`apply_min_p`]) mask filtered tokens to `-inf` so a subsequent
//! [`categorical_sampling`] never draws them. Sampler chaining (mlx-lm's
//! `make_sampler`) is left to the caller — the generation loop is out of
//! scope for M3 sampling.
//!
//! [`apply_xtc`] is the exclude-top-choices sampler (mlx-lm's `apply_xtc`).
//! The four logits-processor primitives —
//! [`apply_repetition_penalty`]/[`apply_presence_penalty`]/
//! [`apply_frequency_penalty`]/[`apply_logit_bias`] — are the **per-call,
//! pure** transforms behind mlx-lm's `make_repetition/presence/frequency
//! _penalty`/`logit_bias` closures (cross-checked against mlx-swift's
//! `RepetitionContext`/`PresencePenaltyContext`/`FrequencyPenaltyContext`).
//! The stateful recent-token ring + `make_logits_processors` composition and
//! the generation loop are the caller's job and out of scope here: each
//! takes the already-sliced recent-token id set (`token_ids`, an integer
//! `[n]` array) explicitly, exactly as mlx-lm's closures receive `tokens`.

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

/// Scale `logits` by `1 / temp`, returning a result in the **original**
/// `logits` dtype. Dispatched per dtype:
/// - **F32**: in-dtype `divide(logits, scalar_like(temp))` (hot path,
///   bit-identical to the pre-fix path).
/// - **F16 / BF16**: upcast to f32, divide in f32, downcast back —
///   `temp` never gets cast down to the narrower dtype.
/// - **F64**: rejected with an `Error::Backend`. MLX's GPU stream
///   (`default_stream()`, which `ops::arithmetic::divide` routes through)
///   does not support `float64` (`"float64 is not supported on the GPU"`),
///   so a native F64 divide errors at eval; the prior implicit f32
///   roundtrip silently lost precision on near-tied logits instead of
///   surfacing the limitation (LM-6 R2 medium finding). The caller must
///   cast logits down to F32 (or F16/BF16) before sampling — the
///   reference Python `mlx_lm.sample_utils.categorical_sampling` only
///   ever runs on F32/F16/BF16 logits.
/// - **Anything else**: rejected with an `Error::Backend` (categorical
///   sampling only makes sense on floating-point logits).
///
/// **NaN-safety (LM-6 R1 follow-up).** The previous fix replaced
/// `multiply(logits, scalar_like(1/temp))` with
/// `divide(logits, scalar_like(temp, logits))`, which still casts `temp`
/// down to the logits dtype BEFORE the divide via `scalar_like`. For f16
/// logits any positive `temp` below f16's minimum subnormal (~5.96e-8)
/// rounds to 0 in that cast; bf16 hits the same trap below its own min
/// subnormal (~9.18e-41). A max-shifted row contains a 0 entry, so
/// `0 / 0 = NaN` leaks into [`crate::ops::random::categorical`]'s softmax — exactly
/// the original LM-6 attack surface, just reached through the dtype cast
/// instead of the `1/temp` overflow.
///
/// **Fix has three parts:**
///
/// 1. **f32-denominator upcast path for f16/bf16.** Upcast `logits` to
///    f32 first, build the divisor in f32 (no cast-to-half), divide,
///    then downcast the result back to the original dtype. `temp` never
///    gets cast to the narrower dtype, eliminating the f16/bf16
///    dtype-cast leg.
///
/// 2. **Below-`1/f32::MAX` clamp.** Empirically MLX's `divide` lowers to
///    multiply-by-reciprocal internally on Apple Silicon (the divisor's
///    f32 reciprocal is materialized inside the kernel), so the original
///    `1/temp` overflow path the prior LM-6 fix claimed to eliminate is
///    actually still active for `temp < 1/f32::MAX ≈ 2.94e-39` (it just
///    moved into mlx-c). Without this clamp, `0 / temp` produces NaN
///    even after the upcast (since the kernel computes `0 * +Inf`).
///    Clamping `temp` from below to [`f32::MIN_POSITIVE`] (smallest f32
///    normal, `~1.18e-38`, so `1/temp` is finite) preserves the divide's
///    correctness for any sub-normal `temp` the validator accepts —
///    `softmax(logits/temp)` is mathematically equivalent to argmax in
///    this limit anyway, and the post-divide `±Inf` overflows for
///    extreme logits resolve correctly inside
///    [`crate::ops::random::categorical`]'s internal softmax (one-hot at
///    the max). This is the secondary recommendation in the LM-6 R1
///    Codex finding ("explicitly route temperatures that would round to
///    zero ... to an argmax-after-filtering path") for the bf16-only
///    sub-min-subnormal regime where the f32 reciprocal trap is
///    unavoidable (bf16 and f32 share an exponent range, so any temp
///    below bf16 min subnormal is also below `1/f32::MAX`).
///
/// 3. **F64 + non-floating dtype rejection (LM-6 R2 follow-up).** The
///    prior single non-F32 branch quietly funneled F64 through an f32
///    roundtrip, so near-tied f64 logits at small `temp` could lose
///    ordering before the Gumbel draw while still returning an F64
///    array. MLX's GPU stream does not support F64 (`"float64 is not
///    supported on the GPU"`), so a native F64 divide would error at
///    eval rather than preserve precision; F64 is now rejected up front
///    with a clear `Error::Backend` instructing the caller to cast
///    before sampling — bit-honest about the backend's actual F64
///    capability. Any non-floating dtype is likewise rejected (mirroring
///    the dtype-rejection pattern in `kl_div_loss`), so an i32/u32/etc.
///    array does not silently astype through f32.
///
/// **Handler safety.** `ensure_handler_installed()` runs as the FIRST
/// executable statement — `Array::full::<f32>` invokes the fallible
/// `mlx_array_new_float32` ctor BEFORE its `mlx_full` call (whose
/// `default_stream()` arg is what triggers the lazy install), so with
/// the eager `#[ctor]` stripped that first ctor could reach mlx-c with
/// no error handler installed → its default `printf + exit(-1)` instead
/// of a recoverable `Err`. Same defense-in-depth as `scalar_like` and
/// `embeddings::scalar_like` (LM-6 R2 Codex finding).
///
/// Exposed as `pub` so the [`categorical_sampling`] regression test can
/// assert directly on the scaled logits (not just the eventual sampled
/// index, which is uninformative under a NaN distribution); also genuinely
/// useful as a building block for custom sampler compositions.
pub fn scale_logits_by_temp(logits: &Array, temp: f32) -> Result<Array> {
  // Install the error handler BEFORE any fallible mlx-c ctor — `Array::full`
  // runs `mlx_array_new_float32` before its `mlx_full(default_stream())`
  // would lazily install it, so a ctor-stripped first sampling call could
  // otherwise trip mlx-c's default `printf + exit(-1)` on scalar allocation
  // failure instead of returning `Err` (LM-6 R2 Codex finding).
  crate::error::ensure_handler_installed();
  if !temp.is_finite() || temp <= 0.0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`temp` has to be a finite positive float (use `argmax_sample` for temperature-0 / greedy decoding), but is {temp}"
      ),
    });
  }
  // Clamp from below to ensure `1/temp` is finite — see docstring part (2).
  // This keeps MLX's internal multiply-by-reciprocal on Apple Silicon from
  // computing `0 * +Inf = NaN` on the max-shifted zero entries. Above this
  // threshold the clamp is a no-op (`temp` is returned unchanged).
  let temp = temp.max(f32::MIN_POSITIVE);
  let dtype = logits.dtype()?;
  match dtype {
    crate::Dtype::F32 => {
      // Bit-identical to the pre-fix path on the hot path most callers
      // hit (no extra cast-roundtrip).
      let divisor = Array::full::<f32>(&(1,), temp)?;
      ops::arithmetic::divide(logits, &divisor)
    }
    crate::Dtype::F16 | crate::Dtype::BF16 => {
      // Half precision — upcast-divide-downcast so `temp` never gets
      // cast to the narrower dtype (the LM-6 R1 dtype-cast leg).
      let logits_f32 = ops::misc::astype(logits, crate::Dtype::F32)?;
      let divisor = Array::full::<f32>(&(1,), temp)?;
      let scaled_f32 = ops::arithmetic::divide(&logits_f32, &divisor)?;
      ops::misc::astype(&scaled_f32, dtype)
    }
    crate::Dtype::F64 => Err(Error::Backend {
      message:
        "categorical_sampling does not support F64 logits — MLX's GPU stream \
         does not implement float64, so a native F64 divide would error at \
         eval and the prior implicit F32 roundtrip silently lost precision on \
         near-tied logits (LM-6 R2 finding). Cast logits with \
         .astype(Dtype::F32) (or F16/BF16) before sampling."
          .to_string(),
    }),
    other => Err(Error::Backend {
      message: format!(
        "categorical_sampling requires floating-point logits (F32, F16, or BF16); got {other:?}. Cast logits with .astype(Dtype::F32) before sampling."
      ),
    }),
  }
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
///
/// **NaN-safety (LM-6).** The scaling is delegated to
/// [`scale_logits_by_temp`], which uses an f32-denominator path so the
/// inverse temperature is never materialized AND `temp` never gets cast
/// down to f16/bf16 (where positive sub-min-subnormal values round to
/// zero, opening a `0 / 0 = NaN` hole under the L3-R2 max-shift —
/// the R1 follow-up to the original LM-6 fix).
pub fn categorical_sampling(logits: &Array, temp: f32, key: &Array) -> Result<Array> {
  let scaled = scale_logits_by_temp(logits, temp)?;
  ops::random::categorical(&scaled, -1, key)
}

/// Greedy (argmax) token selection along the last axis — the
/// temperature-0 branch of `mlx_lm.sample_utils.make_sampler`.
pub fn argmax_sample(logits: &Array) -> Result<Array> {
  ops::misc::argmax(logits, Some(-1), false)
}

/// XTC (exclude-top-choices) sampling.
///
/// Port of `mlx_lm.sample_utils.apply_xtc`. With probability
/// `xtc_probability`, every token whose softmax probability exceeds the
/// *smallest* probability that is still `> xtc_threshold` is masked to
/// `-inf` (so [`categorical_sampling`] never draws the over-confident head),
/// except the `xtc_special_tokens` ids, which are always preserved.
///
/// `xtc_threshold` must be finite in `[0, 0.5]` and `xtc_probability` finite
/// in `[0, 1]` — exactly mlx-lm's `ValueError` bounds (its `threshold` gate
/// is `[0, 0.5]`, not `[0, 1]`), surfaced via the file's `ShapeMismatch`
/// idiom plus an explicit finiteness check (a `NaN` bound would slip mlx-lm's
/// bare `<=` comparison and silently no-op the mask).
///
/// `key` seeds the single Bernoulli gate, mirroring mlx-lm's scalar
/// `mx.random.uniform(0, 1)` (one draw per call, broadcast over all logits),
/// but threaded explicitly like [`categorical_sampling`] (mlx-lm splits it
/// off `mx.random.state`; the deferred `make_sampler` owns the split).
///
/// mlx-lm reduces with the *scalar* `.min()` because it only ever runs on a
/// `[1, vocab]` row; the per-row `min` along `-1` (keepdims) below is its
/// exact equivalent there and the correct generalization for a batched
/// `[..., vocab]` input.
pub fn apply_xtc(
  logits: &Array,
  xtc_probability: f32,
  xtc_threshold: f32,
  xtc_special_tokens: &[i32],
  key: &Array,
) -> Result<Array> {
  if !xtc_threshold.is_finite() || !(0.0..=0.5).contains(&xtc_threshold) {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`xtc_threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
      ),
    });
  }
  if !xtc_probability.is_finite() || !(0.0..=1.0).contains(&xtc_probability) {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`xtc_probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
      ),
    });
  }

  // mlx-lm: `mx.softmax(logits, -1)` — `precise=False` (the mlx default), so
  // pass `false` here for bit-level parity rather than the higher-precision
  // accumulation path.
  let probs = ops::misc::softmax_axis(logits, -1, false)?;
  // `where(probs > xtc_threshold, probs, +inf).min(-1)` — the smallest prob
  // still above the threshold; `+inf` neutralizes the sub-threshold tail in
  // the min. Threshold/`+inf`/`-inf` are built in `probs`/`logits` dtype
  // (weak-scalar parity), so f16/bf16 stays in-dtype.
  let thr = scalar_like(xtc_threshold, &probs)?;
  let pos_inf = scalar_like(f32::INFINITY, &probs)?;
  let above = ops::comparison::greater(&probs, &thr)?;
  let candidates = ops::logical::select(&above, &probs, &pos_inf)?;
  let cutoff = ops::reduction::min_axes(&candidates, &[-1], true)?;
  let mut mask = ops::comparison::greater(&probs, &cutoff)?;

  // `mask[..., xtc_special_tokens] = False` — scatter `false` at the special
  // columns (1-D indices broadcast over any leading dims). Skipped when the
  // set is empty, mirroring mlx-lm's `if xtc_special_tokens:` guard (an empty
  // index array is a valid but pointless scatter).
  if !xtc_special_tokens.is_empty() {
    let special = token_index(logits, xtc_special_tokens)?;
    let off = Array::full::<bool>(&(1,), 0.0)?;
    mask = ops::indexing::put_along_axis(&mask, &special, &off, -1)?;
  }

  // One Bernoulli gate: `where(uniform(0,1) > xtc_probability, logits,
  // where(mask, -inf, logits))`. mlx-lm's `mx.random.uniform(0, 1)` draws a
  // single value (Python scalars broadcast to `()`); `scalar_like` builds
  // `low`/`high` as `[1]`, so the draw shape must be `[1]` (mlx rejects a
  // `()` request from `(1)`-shaped bounds) — a `[1]` gate broadcasts over
  // `[…, vocab]` logits identically to mlx-lm's scalar draw.
  let zero = scalar_like(0.0, logits)?;
  let one = scalar_like(1.0, logits)?;
  let u = ops::random::uniform(&zero, &one, &[1i32], logits.dtype()?, key)?;
  let prob = scalar_like(xtc_probability, logits)?;
  let gate = ops::comparison::greater(&u, &prob)?;
  let neg_inf = scalar_like(f32::NEG_INFINITY, logits)?;
  let masked = ops::logical::select(&mask, &neg_inf, logits)?;
  ops::logical::select(&gate, logits, &masked)
}

/// Build an index array (in the int dtype) addressing `n` token-id columns
/// of the last axis. `put_along_axis`/`take_along_axis`/`scatter_add_axis`
/// require the index rank to equal the operand's, with the non-axis dims
/// broadcasting; so the shape is `[1, …, 1, n]` (`like.ndim()` dims, all
/// leading `1`s broadcast against any `[…, vocab]` logits/mask).
fn token_index(like: &Array, ids: &[i32]) -> Result<Array> {
  let ndim = like.ndim().max(1);
  let mut shape = vec![1i32; ndim];
  let last = shape.len() - 1;
  shape[last] = ids.len() as i32;
  let dims: &[i32] = &shape;
  Array::from_slice::<i32>(ids, &dims)
}

/// Sign-aware multiplicative repetition penalty.
///
/// Port of `mlx_lm.sample_utils.make_repetition_penalty`'s closure
/// (cross-checked against mlx-swift `RepetitionContext.process`): for every
/// id in `token_ids`, `logit < 0 → logit * penalty` else `logit / penalty`,
/// scattered back into a copy of `logits`. The caller passes the (already
/// `context_size`-sliced) recent-token id set — the stateful ring is out of
/// scope. `penalty` must be finite and non-negative (mlx-lm's `ValueError`).
///
/// `put_along_axis` is last-write-wins on duplicate ids, exactly matching
/// mlx-lm's `logits[:, tokens] = selected_logits` fancy-index assignment
/// (the per-column scaled value is deterministic, so duplicates are a no-op).
pub fn apply_repetition_penalty(logits: &Array, token_ids: &[i32], penalty: f32) -> Result<Array> {
  if !penalty.is_finite() || penalty < 0.0 {
    return Err(Error::ShapeMismatch {
      message: format!("`penalty` must be a non-negative float, but is {penalty}"),
    });
  }
  if token_ids.is_empty() {
    return logits.try_clone();
  }
  let idx = token_index(logits, token_ids)?;
  let selected = ops::indexing::take_along_axis(logits, &idx, -1)?;
  let p = scalar_like(penalty, &selected)?;
  let scaled_down = ops::arithmetic::multiply(&selected, &p)?;
  let scaled_up = ops::arithmetic::divide(&selected, &p)?;
  let is_neg = ops::comparison::less(&selected, &scalar_like(0.0, &selected)?)?;
  let new_selected = ops::logical::select(&is_neg, &scaled_down, &scaled_up)?;
  ops::indexing::put_along_axis(logits, &idx, &new_selected, -1)
}

/// Presence penalty: subtract `penalty` **once** from every logit whose id
/// occurs in `token_ids`.
///
/// Port of `mlx_lm.sample_utils.make_presence_penalty`'s closure (the OpenAI
/// `presence_penalty`; cross-checked against mlx-swift
/// `PresencePenaltyContext.process`). The caller supplies the recent-token
/// id set. Like mlx-lm's `logits[:, tokens] -= penalty` this is a fancy-index
/// *assignment* (`take` → subtract → `put_along_axis`), so a duplicated id is
/// penalized once — not once per occurrence (that is the frequency penalty).
pub fn apply_presence_penalty(logits: &Array, token_ids: &[i32], penalty: f32) -> Result<Array> {
  if token_ids.is_empty() {
    return logits.try_clone();
  }
  let idx = token_index(logits, token_ids)?;
  let selected = ops::indexing::take_along_axis(logits, &idx, -1)?;
  let reduced = ops::arithmetic::subtract(&selected, &scalar_like(penalty, &selected)?)?;
  ops::indexing::put_along_axis(logits, &idx, &reduced, -1)
}

/// Frequency penalty: subtract `penalty * occurrence_count` from every
/// logit, where the count is how many times the id appears in `token_ids`.
///
/// Port of `mlx_lm.sample_utils.make_frequency_penalty`'s closure (the OpenAI
/// `frequency_penalty`), cross-checked against mlx-swift
/// `FrequencyPenaltyContext`. Implemented by scatter-adding `-penalty`
/// **directly** onto `logits`, once per occurrence of each id — repeated ids
/// accumulate, so a token mentioned `k` times gets `-penalty * k`, exactly
/// mlx-lm's repeated-index `logits.at[:, tokens].subtract(penalty)`. The
/// earlier dense `logits - histogram * penalty` form is deliberately *not*
/// used: it arithmetics every column, so a low-precision `0 * penalty` (or
/// an over-magnitude penalty) NaN-bleeds / flips signed zeros into untouched
/// logits. `scatter_add_axis` does no arithmetic on non-indexed positions,
/// so every untouched column is the bitwise-identical input for all
/// dtypes/penalty magnitudes (see the implementation note below).
pub fn apply_frequency_penalty(logits: &Array, token_ids: &[i32], penalty: f32) -> Result<Array> {
  if token_ids.is_empty() {
    return logits.try_clone();
  }
  // Indexed scatter-add of `-penalty` once per occurrence DIRECTLY onto
  // `logits` (no dense intermediate, mirroring `apply_logit_bias`): repeated
  // ids accumulate, so a mentioned token gets `-penalty * count` — exactly
  // mlx-lm's `logits.at[:, tokens].subtract(penalty)`. Crucially,
  // `scatter_add_axis` performs NO arithmetic on non-indexed positions, so
  // every UNtouched column is the bitwise-identical input for ALL
  // dtypes/penalty magnitudes: no `0 * inf` NaN-bleed (the original
  // `histogram * scalar` bug) AND no signed-zero flip / NaN canonicalization
  // (a global `logits + delta` would still arithmetic untouched columns).
  // `idx`/`neg_pen` are rank-matched to `logits` via `token_index` (same as
  // the sibling penalty transforms), so the non-axis broadcast holds for a
  // batched `[B, vocab]` input too.
  let idx = token_index(logits, token_ids)?;
  let ndim = logits.ndim().max(1);
  let mut vshape = vec![1i32; ndim];
  let last = vshape.len() - 1;
  vshape[last] = token_ids.len() as i32;
  let vdims: &[i32] = &vshape;
  let neg_pen = ops::shape::reshape(
    &ops::misc::astype(
      &Array::full::<f32>(&(token_ids.len(),), -penalty)?,
      logits.dtype()?,
    )?,
    &vdims,
  )?;
  ops::indexing::scatter_add_axis(logits, &idx, &neg_pen, -1)
}

/// Additive logit bias: add `values[i]` to the logit at column `indices[i]`.
///
/// Port of `mlx_lm.sample_utils`' inline `logit_bias_processor`
/// (`logits.at[:, indices].add(values)`). `indices` (an int `[n]` array) and
/// `values` (a numeric `[n]` array) are paired by position; duplicate indices
/// **accumulate** (mlx `.at[].add` semantics), unlike the assignment-based
/// repetition/presence penalties. A scatter-add over the last axis,
/// broadcasting `[1, n]` indices/values against `[..., vocab]` logits.
pub fn apply_logit_bias(logits: &Array, indices: &[i32], values: &Array) -> Result<Array> {
  // Validate length BEFORE the empty short-circuit: an empty `indices` with
  // non-empty `values` is a caller length-mismatch, not a no-op (the
  // shortcut would otherwise silently drop the supplied bias).
  if values.size() != indices.len() {
    return Err(Error::ShapeMismatch {
      message: format!(
        "`logit_bias` indices ({}) and values ({}) must have the same length",
        indices.len(),
        values.size()
      ),
    });
  }
  if indices.is_empty() {
    return logits.try_clone();
  }
  let idx = token_index(logits, indices)?;
  let ndim = logits.ndim().max(1);
  let mut vshape = vec![1i32; ndim];
  let last = vshape.len() - 1;
  vshape[last] = indices.len() as i32;
  let vdims: &[i32] = &vshape;
  let v = ops::shape::reshape(&ops::misc::astype(values, logits.dtype()?)?, &vdims)?;
  ops::indexing::scatter_add_axis(logits, &idx, &v, -1)
}
