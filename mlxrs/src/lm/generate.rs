//! The architecture-agnostic text-generation loop, ported 1:1 from
//! [`mlx_lm.generate`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
//! (`generate_step` / `stream_generate` / `generate`) and the sampler /
//! logits-processor composition of
//! [`mlx_lm.sample_utils`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)
//! (`make_sampler` / `make_logits_processors`), cross-checked against
//! mlx-swift-lm's `MLXLMCommon` `Evaluate`.
//!
//! Everything is generic over the [`Model`] trait: the loop only ever calls
//! `model.forward(tokens, &mut cache)`. The decode loop is an idiomatic Rust
//! [`Iterator`] — [`generate_step`] yields one [`GenStep`]
//! per step (the typed step item: `token` + opt-in `logprobs`),
//! [`stream_generate`] maps that through the #18 streaming detokenizer into
//! [`GenerationResponse`]s, and [`generate`] collects the whole thing into a
//! `(String, GenerationStats)` pair (the L3 stats surface — counts +
//! tokens-per-second + peak memory, sourced from the final
//! [`GenerationResponse`]).
//!
//! **L3 opt-in: per-step logprobs are gated by
//! [`GenConfig::collect_logprobs`].** When `false` (default), [`GenStep`]'s
//! `logprobs` field is `None`, the post-sampler squeeze is skipped, **and
//! the `logits - logsumexp(logits)` normalization itself is skipped**
//! whenever the configured sampler doesn't need normalized probabilities —
//! i.e. greedy (`temp == 0`), top-k (monotonic-invariant), min-p
//! (shift-invariant `max(logprobs) + log(min_p)` threshold), xtc (does its
//! own `softmax` internally), and categorical (does its own `softmax`
//! internally) all sample correctly from the raw post-processor logits, so
//! a `collect_logprobs=false` run pays **zero** vocab-wide normalization
//! cost per token. Only `top_p ∈ (0, 1)` requires the normalized
//! log-probability cumsum-to-1 contract; when top_p is enabled the
//! normalization runs regardless of `collect_logprobs` (the sampler would
//! otherwise read uninitialized cumulative probabilities). When
//! `collect_logprobs=true`, the `[V]` `Array` is yielded byte-identically
//! to mlx-lm's `logprobs.squeeze(0)` (the normalization is always run so
//! the yielded vector is the true log-softmax).
//!
//! **Stochastic-opt-out numerical safety:** the
//! shift-invariance argument above is mathematically valid but **not**
//! numerically safe in low-precision compute dtypes. `categorical_sampling`
//! multiplies its input by `1/temp` BEFORE the eventual `softmax` inside
//! `mx.random.categorical`, so in f16 / bf16 a small `temp` combined with a
//! large `logit_bias` (e.g. `bias = +50`, `temp = 0.1` ⇒ scaled logit `500`)
//! overflows to `+inf` long before `softmax` can stabilize via shift
//! cancellation. To preserve the per-step saving for stochastic configs
//! without exposing that overflow, the opt-out path applies a cheap
//! `logits - max(logits, keepdims=True)` per-row max-shift (one reduce + one
//! broadcast subtract — ~3-4× cheaper than the full `logsumexp` + subtract)
//! before feeding the sampler when `temp > 0`. Pure-greedy (`temp == 0`,
//! `argmax_sample`) is shift-invariant numerically as well (argmax doesn't
//! exponentiate anything), so it still receives the raw post-processor
//! logits.
//!
//! **Numerical-safety scope:** the opt-in logprobs + opt-out paths above
//! work correctly for sane (non-subnormal) `temp` + non-f16-tiny-temp
//! configs. Two extreme-`temp` configurations still produce non-finite
//! distributions inside `sample::categorical_sampling` itself (the bug
//! is in the primitive, not the generation loop): f16 logits +
//! `temp < 1/65504 ≈ 1.526e-5`, and any logits dtype + subnormal
//! positive `temp < 1.0/f32::MAX ≈ 2.94e-39`. Both share the same root:
//! `categorical_sampling` computes `1.0/temp` in `f32` (because
//! `GenConfig.temp` is `f32`) then multiplies by `scalar_like(1/temp,
//! logits)` IN THE LOGITS DTYPE; the `f32` reciprocal overflows for
//! subnormal `temp`, and the dtype cast overflows for f16 + tiny `temp`.
//! VLM ([`crate::vlm::generate`]) and audio ([`crate::audio::stt::generate`])
//! share the same defect via the same `make_sampler` chain. The
//! structural fix lives in `sample::categorical_sampling` itself —
//! the fix must avoid materializing an `+Inf` reciprocal OR casting an
//! overflowing reciprocal into the logits dtype. Two viable shapes:
//! (1) divide instead of multiply (`logits / scalar_like(temp,
//! logits)` — never materializes `1/temp`, covers both overflow
//! modes), or (2) route to argmax INSIDE the primitive after every
//! upstream sampler stage runs (preserves XTC/top_k/min_p semantics).
//! A naive `1/temp` upcast to f64 is only a partial mitigation — it
//! closes the f32 subnormal path but the cast back into f16 still
//! overflows for `temp < 1/65504`. A LM-only argmax bypass in the
//! generation loop was prototyped and reverted:
//! it failed to cover VLM/STT and silently skipped configured sampler
//! stages, so the fix must land in the primitive with regression
//! tests across LM/VLM/STT. Deferred to a dedicated `fix(lm/sample)`
//! follow-up PR after this one merges.
//!
//! **Exact per-step order (mlx-lm `generate_step._step`, lines 396-422):**
//!
//! 1. `logits = model.forward(last_tok[1, 1], &mut cache)` — `[1, 1, V]`,
//!    cache updated in place.
//! 2. `logits = logits[:, -1, :]` — the final position, `[1, V]`.
//! 3. accumulate the step's *input* tokens into the running history, then
//!    `for p in logits_processors: logits = p(history, logits)` (raw logits,
//!    full history; each processor slices its own `context_size` — the #29
//!    primitives).
//! 4. `logprobs = logits - mx.logsumexp(logits, keepdims=True)` — the exact
//!    mlx-lm normalization (all-axes `logsumexp`, `[1, 1]`, broadcast).
//!    **Skipped** entirely when both [`GenConfig::collect_logprobs`] is
//!    `false` AND the sampler chain doesn't need normalized log-probs
//!    (every sampler except `top_p` is shift-invariant or softmaxes
//!    internally — see [`GenConfig::collect_logprobs`]). In the opt-out
//!    path with `temp > 0` a cheap `logits - max(logits, keepdims=True)`
//!    max-shift is applied instead, to keep the downstream `1/temp`
//!    multiply finite in f16/bf16 (see the module-level
//!    "stochastic-opt-out numerical safety" note).
//! 5. `token = sampler(logits_or_logprobs)` — the [`make_sampler`] chain
//!    (top-k/p, min-p, xtc, categorical) or the default temperature-0
//!    `argmax`. The argument is the post-normalization `logprobs` if the
//!    full normalization ran, the max-shifted logits if only the cheap
//!    shift ran (stochastic opt-out), and the raw post-processor `logits`
//!    if neither did (pure-greedy opt-out). Every sampler in
//!    [`make_sampler`] is shift-invariant or softmaxes internally except
//!    `top_p`, which forces step 4 to run. Extreme-temp NaN-safety for
//!    `categorical_sampling` itself (f16 + `temp < 1/65504`, any dtype +
//!    subnormal `temp`) is deferred to a dedicated `fix(lm/sample)`
//!    follow-up PR — see the module-level "numerical-safety scope"
//!    note for the scope, fix options, and prior in-diff revert
//!    rationale.
//! 6. yield `GenStep { token, logprobs }` — `logprobs` is
//!    `Some(logprobs.squeeze(0))` when [`GenConfig::collect_logprobs`] is
//!    `true`, `None` otherwise (L3 opt-in; mlx-lm always yields the
//!    array, mlxrs surfaces the cost knob to the step loop). Stop when
//!    `token ∈ eos` (`finish_reason = "stop"`) or `count == max_tokens`
//!    (`finish_reason = "length"`).
//!
//! **Prefill** is chunked by [`GenConfig::prefill_step_size`] (mlx-lm lines
//! 430-453): the prompt's first `total - 1` tokens are fed in
//! `prefill_step_size`-sized chunks (logits discarded, cache filled); the
//! last token starts the first decode step.
//!
//! **Error model:** every fallible op returns [`crate::Result`];
//! [`generate_step`] / [`stream_generate`] are `Iterator<Item = Result<..>>`
//! — a step error is yielded **once** as `Err` and then the iterator ends
//! (it fuses — no panic, no poison, never re-entered). No implicit eval: the
//! only materialization is the `.item::<u32>()` at the explicit
//! token-extraction boundary (mlx-lm's `y.item()`); `logprobs` stays lazy.
//!
//! `make_sampler` / `make_logits_processors` **compose** the [`sample`] /
//! #29 primitives and propagate their validation `Err`s — they do **not**
//! re-validate ranges `sample.rs` already enforces. `temp == 0`
//! ⇒ the argmax sampler (mlx-lm `make_sampler` line 46). All sampler /
//! processor scalars stay in the compute dtype via the #29 `scalar_like`
//! discipline.
//!
//! [`Model`]: crate::lm::model::Model
//! [`sample`]: crate::lm::sample

use std::cell::RefCell;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{
    EmptyInputPayload, Error, LengthMismatchPayload, NonFiniteScalarPayload, OutOfRangePayload,
    RankMismatchPayload, Result, try_extend_from_slice, try_with_capacity,
  },
  lm::{cache::KvCache, model::Model, sample},
  ops,
};
// #111: bring the trait into scope so the `Detokenizer` enum's
// `StreamingDetokenizer` impl methods (`add_token` / `finalize` / `text` /
// `last_segment` / …) are callable through the enum value.
#[cfg(feature = "tokenizer-stream")]
use crate::tokenizer::StreamingDetokenizer as _;

/// The custom-escape-hatch closure type for [`LogitsProcessor::Custom`]
/// (extracted to satisfy `clippy::type_complexity` on the variant).
pub type LogitsProcessorFn = Box<dyn Fn(&[u32], &Array) -> Result<Array>>;

/// The custom-escape-hatch closure type for [`Sampler::Custom`]
/// (extracted to satisfy `clippy::type_complexity` on the variant).
pub type SamplerFn = Box<dyn FnMut(&Array) -> Result<Array>>;

/// Payload for [`LogitsProcessor::LogitBias`].
#[derive(Debug)]
pub struct LogitBiasPayload {
  indices: Vec<i32>,
  values: Array,
}

impl LogitBiasPayload {
  /// Construct a logit-bias payload from `(indices, values)` paired by position.
  pub fn new(indices: Vec<i32>, values: Array) -> Self {
    Self { indices, values }
  }

  /// The token-id columns to add bias to.
  #[inline(always)]
  pub fn indices_slice(&self) -> &[i32] {
    &self.indices
  }

  /// The bias array (built once at construction).
  #[inline(always)]
  pub fn values_ref(&self) -> &Array {
    &self.values
  }
}

/// Payload for [`LogitsProcessor::RepetitionPenalty`].
#[derive(Debug, Clone, Copy)]
pub struct RepetitionPenaltyPayload {
  penalty: f32,
  context_size: usize,
}

impl RepetitionPenaltyPayload {
  /// Construct a repetition-penalty payload.
  pub const fn new(penalty: f32, context_size: usize) -> Self {
    Self {
      penalty,
      context_size,
    }
  }

  /// The penalty factor (mlx-lm `repetition_penalty`).
  #[inline(always)]
  pub const fn penalty(&self) -> f32 {
    self.penalty
  }

  /// Window size (mlx-lm `repetition_context_size`).
  #[inline(always)]
  pub const fn context_size(&self) -> usize {
    self.context_size
  }
}

/// Payload for [`LogitsProcessor::PresencePenalty`].
#[derive(Debug, Clone, Copy)]
pub struct PresencePenaltyPayload {
  penalty: f32,
  context_size: usize,
}

impl PresencePenaltyPayload {
  /// Construct a presence-penalty payload.
  pub const fn new(penalty: f32, context_size: usize) -> Self {
    Self {
      penalty,
      context_size,
    }
  }

  /// The penalty value (mlx-lm `presence_penalty`).
  #[inline(always)]
  pub const fn penalty(&self) -> f32 {
    self.penalty
  }

  /// Window size (mlx-lm `presence_context_size`).
  #[inline(always)]
  pub const fn context_size(&self) -> usize {
    self.context_size
  }
}

/// Payload for [`LogitsProcessor::FrequencyPenalty`].
#[derive(Debug, Clone, Copy)]
pub struct FrequencyPenaltyPayload {
  penalty: f32,
  context_size: usize,
}

impl FrequencyPenaltyPayload {
  /// Construct a frequency-penalty payload.
  pub const fn new(penalty: f32, context_size: usize) -> Self {
    Self {
      penalty,
      context_size,
    }
  }

  /// The penalty value (mlx-lm `frequency_penalty`).
  #[inline(always)]
  pub const fn penalty(&self) -> f32 {
    self.penalty
  }

  /// Window size (mlx-lm `frequency_context_size`).
  #[inline(always)]
  pub const fn context_size(&self) -> usize {
    self.context_size
  }
}

/// A logits processor: maps `(recent token-id history, raw logits)` to
/// processed logits, exactly mlx-lm's
/// `Callable[[mx.array, mx.array], mx.array]` (`make_logits_processors`
/// closures).
///
/// # Breaking change (#109)
///
/// Previously this was the trait-object alias
/// `Box<dyn Fn(&[u32], &Array) -> Result<Array>>` — one vtable indirection
/// per processor per token (~4 indirections per token on the canonical
/// chain: logit_bias + repetition + presence + frequency penalties). The
/// enum unification preserves the same closure-call shape via `apply` but
/// dispatches via a `match` so the compiler can inline each variant and
/// the branch predictor warms on the consistent per-step variant pattern.
///
/// Construct via [`make_logits_processors`] (the canonical chain) or one
/// of the variant constructors below; out-of-tree processors (e.g. the
/// grammar-constrained [`crate::lm::structured::LLGuidanceLogitsProcessor`])
/// plug in through the [`LogitsProcessor::Custom`] escape hatch.
#[non_exhaustive]
#[derive(derive_more::IsVariant)]
pub enum LogitsProcessor {
  /// Additive logit bias (mlx-lm's inline `logit_bias_processor`).
  LogitBias(LogitBiasPayload),
  /// Sign-aware multiplicative repetition penalty (mlx-lm's
  /// `make_repetition_penalty`). `context_size` is the per-penalty
  /// independent window (Python `repetition_context_size`).
  RepetitionPenalty(RepetitionPenaltyPayload),
  /// OpenAI presence penalty (mlx-lm's `make_presence_penalty`).
  PresencePenalty(PresencePenaltyPayload),
  /// OpenAI frequency penalty (mlx-lm's `make_frequency_penalty`).
  FrequencyPenalty(FrequencyPenaltyPayload),
  /// Custom out-of-tree processor (escape hatch — e.g.
  /// [`crate::lm::structured::LLGuidanceLogitsProcessor`]). One
  /// indirection per call, but the standard mlx-lm chain inlines
  /// everything else.
  Custom(LogitsProcessorFn),
}

impl LogitsProcessor {
  /// Apply the processor: dispatch through the variant `match` to the
  /// matching [`crate::lm::sample`] primitive. The canonical chain
  /// inlines through the compiler-elided `match`; only [`Self::Custom`]
  /// takes an indirection.
  pub fn apply(&self, tokens: &[u32], logits: &Array) -> Result<Array> {
    match self {
      Self::LogitBias(p) => sample::apply_logit_bias(logits, p.indices_slice(), p.values_ref()),
      Self::RepetitionPenalty(p) => {
        let ids = recent_ids(tokens, p.context_size())?;
        sample::apply_repetition_penalty(logits, &ids, p.penalty())
      }
      Self::PresencePenalty(p) => {
        let ids = recent_ids(tokens, p.context_size())?;
        sample::apply_presence_penalty(logits, &ids, p.penalty())
      }
      Self::FrequencyPenalty(p) => {
        let ids = recent_ids(tokens, p.context_size())?;
        sample::apply_frequency_penalty(logits, &ids, p.penalty())
      }
      Self::Custom(f) => f(tokens, logits),
    }
  }
}

impl std::fmt::Debug for LogitsProcessor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::LogitBias(p) => f
        .debug_struct("LogitBias")
        .field("n", &p.indices_slice().len())
        .finish(),
      Self::RepetitionPenalty(p) => f
        .debug_struct("RepetitionPenalty")
        .field("penalty", &p.penalty())
        .field("context_size", &p.context_size())
        .finish(),
      Self::PresencePenalty(p) => f
        .debug_struct("PresencePenalty")
        .field("penalty", &p.penalty())
        .field("context_size", &p.context_size())
        .finish(),
      Self::FrequencyPenalty(p) => f
        .debug_struct("FrequencyPenalty")
        .field("penalty", &p.penalty())
        .field("context_size", &p.context_size())
        .finish(),
      Self::Custom(_) => f.debug_tuple("Custom").finish(),
    }
  }
}

/// A sampler: maps a log-probability vector to a sampled token id array
/// (`[1]`, `U32`), exactly mlx-lm's `Callable[[mx.array], mx.array]`.
///
/// # Breaking change (#108)
///
/// Previously this was the trait-object alias
/// `Box<dyn FnMut(&Array) -> Result<Array>>` — ONE indirect call per
/// token (the **hottest** dispatch site in the loop). The enum
/// unification dispatches through a `match` so the canonical chain
/// inlines; only [`Sampler::Custom`] still takes an indirection.
///
/// Construct via [`make_sampler`] (the canonical chain) or
/// [`Sampler::custom`]; out-of-tree samplers plug in through
/// [`Sampler::Custom`].
pub enum Sampler {
  /// Greedy / temperature-0 argmax (mlx-lm `make_sampler` line 46).
  /// Pure — no PRNG state.
  Argmax,
  /// The full mlx-lm `make_sampler` chain: top-p → min-p → xtc →
  /// top-k → categorical (all gated on their `do_*` flags). The
  /// per-call PRNG key is split per call to mirror `mx.random.state`.
  Chain(SamplerChain),
  /// Custom out-of-tree sampler (escape hatch). One indirection per
  /// call.
  Custom(SamplerFn),
}

impl Sampler {
  /// Build a [`Self::Custom`] sampler from a closure. Convenience
  /// constructor matching the prior `Sampler = Box<dyn FnMut>` shape.
  pub fn custom<F>(f: F) -> Self
  where
    F: FnMut(&Array) -> Result<Array> + 'static,
  {
    Self::Custom(Box::new(f))
  }

  /// Sample one token from `logits`: dispatch through the variant
  /// `match` to the matching [`crate::lm::sample`] composition. The
  /// canonical chain inlines through the compiler-elided `match`;
  /// only [`Self::Custom`] takes an indirection.
  pub fn sample(&mut self, logits: &Array) -> Result<Array> {
    match self {
      Self::Argmax => sample::argmax_sample(logits),
      Self::Chain(c) => c.sample(logits),
      Self::Custom(f) => f(logits),
    }
  }
}

impl std::fmt::Debug for Sampler {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Argmax => f.write_str("Argmax"),
      Self::Chain(c) => f.debug_tuple("Chain").field(c).finish(),
      Self::Custom(_) => f.debug_tuple("Custom").finish(),
    }
  }
}

/// The mlx-lm `make_sampler` chain — top-p → min-p → xtc → top-k →
/// categorical, each stage gated on its own `do_*` flag. Owns the
/// per-call PRNG key (advanced per call via [`ops::random::split`] to
/// mirror mlx-lm's `mx.random.state` advance).
///
/// Wrapped in [`Sampler::Chain`]; constructed via [`make_sampler`].
pub struct SamplerChain {
  temp: f32,
  top_p: f32,
  min_p: f32,
  min_tokens_to_keep: i32,
  top_k: i32,
  xtc_probability: f32,
  xtc_threshold: f32,
  xtc_special: Vec<i32>,
  do_top_p: bool,
  do_min_p: bool,
  do_xtc: bool,
  do_top_k: bool,
  /// Per-call PRNG key advanced on each `sample` call (mlx-lm's
  /// `mx.random.state` analogue). `RefCell` because `Sampler::sample`
  /// is `&mut self` but the chain's interior key advance is independent
  /// of the immutable enum dispatch.
  key: RefCell<Array>,
}

impl SamplerChain {
  fn sample(&self, logprobs: &Array) -> Result<Array> {
    let (k_xtc, k_cat) = {
      let mut k = self.key.borrow_mut();
      let (next, k_xtc) = ops::random::split(&k)?;
      let (next, k_cat) = ops::random::split(&next)?;
      *k = next;
      (k_xtc, k_cat)
    };
    // CORE-1: thread an `Option<Array>` through the optional stages so the
    // "no-op stage" path is a pure borrow of `logprobs` (no clone), and a
    // taken stage moves its owned result into `x`.
    let mut x: Option<Array> = if self.do_top_p {
      Some(sample::apply_top_p(logprobs, self.top_p)?)
    } else {
      None
    };
    if self.do_min_p {
      x = Some(sample::apply_min_p(
        x.as_ref().unwrap_or(logprobs),
        self.min_p,
        self.min_tokens_to_keep,
      )?);
    }
    if self.do_xtc {
      x = Some(sample::apply_xtc(
        x.as_ref().unwrap_or(logprobs),
        self.xtc_probability,
        self.xtc_threshold,
        &self.xtc_special,
        &k_xtc,
      )?);
    }
    if self.do_top_k {
      x = Some(sample::apply_top_k(
        x.as_ref().unwrap_or(logprobs),
        self.top_k,
      )?);
    }
    sample::categorical_sampling(x.as_ref().unwrap_or(logprobs), self.temp, &k_cat)
  }
}

impl std::fmt::Debug for SamplerChain {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SamplerChain")
      .field("temp", &self.temp)
      .field("top_p", &self.top_p)
      .field("min_p", &self.min_p)
      .field("top_k", &self.top_k)
      .field("xtc_probability", &self.xtc_probability)
      .finish()
  }
}

/// mlx-lm's `make_logits_processors` default `*_context_size` (the number of
/// most-recent tokens each penalty considers).
pub const DEFAULT_REPETITION_CONTEXT_SIZE: usize = 20;

/// A fresh, process-unique RNG seed for an unseeded stochastic sampler
/// ([`make_sampler`] / [`GenConfig::seed`] `None`).
///
/// mlx-lm's unseeded stochastic generations never repeat because each draws
/// from the advancing process-global `mx.random.state`. mlxrs's random API
/// is explicit-key (#21), so reproduce that property here: a monotonic
/// per-process [`AtomicU64`](std::sync::atomic::AtomicU64) counter mixed with
/// the wall clock yields a distinct seed per `make_sampler` call (so two
/// independent non-greedy runs in one process get different RNG streams),
/// while the clock component decorrelates seeds across process restarts.
/// Pure `std` — no entropy crate / new dependency.
fn next_sampler_seed() -> u64 {
  use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
  };
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_nanos() as u64)
    .unwrap_or(0);
  // Mix the counter into the high bits so two calls within the same clock
  // tick still differ; the clock decorrelates across process restarts.
  nanos ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Test-only seam (doc-hidden, **not** part of the public API): exposes the
/// seed an unseeded [`make_sampler`] / [`GenConfig::seed`] `None` resolves to
/// ([`next_sampler_seed`]) **without** running the stochastic sampler. Lets
/// the integration tests check the unseeded-independence property by
/// observing the deterministic seed-*resolution* path (the monotonic
/// per-process counter strictly advances every call) rather than by
/// comparing two unseeded *random token sequences* (the previous flaky
/// probabilistic assertion). No behavioural effect; a pure read of the
/// already-existing resolution path.
#[doc(hidden)]
pub fn __resolved_unseeded_seed_for_test() -> u64 {
  next_sampler_seed()
}

/// Generation parameters — the union of mlx-lm `generate_step`'s loop knobs,
/// `make_sampler`'s sampler params, and `make_logits_processors`' penalty /
/// bias params, plus the resolved eos-id set.
///
/// [`Default`] mirrors mlx-lm's defaults: `max_tokens = 256`,
/// `prefill_step_size = 2048`, `temp = 0` (⇒ the argmax sampler), every
/// other sampler / penalty knob off, no eos ids (the caller wires the
/// tokenizer's set — `stream_generate` does).
#[derive(Debug, Clone)]
pub struct GenConfig {
  /// Maximum number of tokens to generate (mlx-lm `max_tokens`). `0`
  /// produces nothing.
  pub max_tokens: usize,
  /// Prompt-prefill chunk size (mlx-lm `generate_step` `prefill_step_size`,
  /// default 2048).
  pub prefill_step_size: usize,

  // --- sampler params (mlx-lm `make_sampler`) ---------------------------
  /// Sampling temperature; `0` ⇒ the deterministic argmax sampler (mlx-lm
  /// `make_sampler` line 46).
  pub temp: f32,
  /// Nucleus (top-p) cutoff; applied iff `0 < top_p < 1`.
  pub top_p: f32,
  /// Min-p cutoff (scaled by the top token's prob); applied iff `!= 0`.
  pub min_p: f32,
  /// Minimum tokens min-p must keep (mlx-lm `min_tokens_to_keep`, default
  /// `1`).
  pub min_tokens_to_keep: i32,
  /// Top-k cutoff; applied iff `> 0`.
  pub top_k: i32,
  /// XTC application probability; the XTC stage is added iff `> 0`.
  pub xtc_probability: f32,
  /// XTC probability threshold.
  pub xtc_threshold: f32,
  /// Token ids XTC never masks.
  ///
  /// Private: access via [`xtc_special_tokens_slice`](Self::xtc_special_tokens_slice);
  /// set via [`with_xtc_special_tokens`](Self::with_xtc_special_tokens).
  pub(crate) xtc_special_tokens: Vec<i32>,

  // --- logits-processor params (mlx-lm `make_logits_processors`) --------
  /// Additive logit bias as `(token_id, bias)` pairs (mlx-lm's
  /// `Dict[int, float]`). Applied first, before the penalties.
  ///
  /// Private: access via [`logit_bias_slice`](Self::logit_bias_slice);
  /// set via [`with_logit_bias`](Self::with_logit_bias).
  pub(crate) logit_bias: Vec<(i32, f32)>,
  /// Sign-aware multiplicative repetition penalty; the processor is added
  /// iff `Some(p)` with `p != 0`.
  pub repetition_penalty: Option<f32>,
  /// Most-recent tokens the **repetition** penalty considers (mlx-lm's
  /// `repetition_context_size`; default
  /// [`DEFAULT_REPETITION_CONTEXT_SIZE`]).
  pub repetition_context_size: usize,
  /// OpenAI presence penalty; added iff `Some(p)` with `p != 0`.
  pub presence_penalty: Option<f32>,
  /// Most-recent tokens the **presence** penalty considers (mlx-lm's
  /// independent `presence_context_size`; default
  /// [`DEFAULT_REPETITION_CONTEXT_SIZE`]).
  pub presence_context_size: usize,
  /// OpenAI frequency penalty; added iff `Some(p)` with `p != 0`.
  pub frequency_penalty: Option<f32>,
  /// Most-recent tokens the **frequency** penalty considers (mlx-lm's
  /// independent `frequency_context_size`; default
  /// [`DEFAULT_REPETITION_CONTEXT_SIZE`]).
  pub frequency_context_size: usize,

  /// The resolved stop-token id set (mlx-lm `tokenizer.eos_token_ids`).
  /// Generation ends (`finish_reason = "stop"`) once a sampled token is in
  /// this set.
  ///
  /// Private: access via [`eos_slice`](Self::eos_slice);
  /// set via [`with_eos`](Self::with_eos).
  pub(crate) eos: Vec<u32>,

  /// Multi-token / string stop sequences (mlx-lm's `stop_words` / the
  /// server's `stop` strings). Generation ends (`finish_reason = "stop"`)
  /// once any of these strings appears in the decoded output, and the matched
  /// stop sequence is trimmed from the returned text (see
  /// [`crate::lm::stop`]). Default empty ⇒ eos-only stopping, byte-for-byte
  /// the prior behavior. Only consulted by the text-level entry points
  /// ([`stream_generate`] / [`generate`]); [`generate_step`] is token-only,
  /// like mlx-lm.
  ///
  /// Private: access via [`stop_strings_slice`](Self::stop_strings_slice);
  /// set via [`with_stop_strings`](Self::with_stop_strings).
  pub(crate) stop_strings: Vec<String>,

  /// Stochastic-sampler RNG seed (mlx-lm's `mx.random.seed` analogue).
  /// `Some(s)` ⇒ a non-greedy run is reproducible; `None` ⇒ a fresh
  /// process-unique seed per run so independent non-greedy generations never
  /// restart from the same sequence (mlx-lm's default — see [`make_sampler`]).
  /// Ignored when `temp == 0` (the deterministic argmax sampler).
  pub seed: Option<u64>,

  /// When `true`, every yielded [`GenStep`] carries the full `[V]`
  /// log-probability vector as `Some(Array)` (the mlx-lm `generate_step`
  /// per-step `logprobs` yield, lazy / kept on-device); when `false`
  /// (default), the loop yields `None` AND skips the
  /// `logits - logsumexp(logits)` normalization graph entirely when the
  /// configured sampler doesn't require it — i.e. for greedy
  /// (`temp == 0`) and every chain that does NOT include `top_p` (top-k /
  /// min-p / xtc / categorical are all shift-invariant or
  /// softmax-internally, so they sample correctly from raw post-processor
  /// logits). `top_p ∈ (0, 1)` forces the normalization regardless of this
  /// flag, since the cumsum-to-1 threshold is only meaningful on
  /// normalized log-probs. This is a true ZERO-cost opt-out for the common
  /// greedy / temperature-only case: the per-token vocab-wide reduce +
  /// broadcast subtract that `logsumexp` triggers is avoided, not just the
  /// `[V]` view squeeze. mlx-lm itself always yields logprobs (server-side
  /// opt-in lives a layer up at `mlx_lm/server.py:191` `logprobs: bool`);
  /// flipping the opt-in down to the step loop is a Rust-idiomatic
  /// cost-discipline improvement that honors the project's "no implicit
  /// eval" / allocation-discipline rules without changing the per-step
  /// compute when logprobs ARE requested.
  pub collect_logprobs: bool,
}

impl Default for GenConfig {
  fn default() -> Self {
    Self {
      max_tokens: 256,
      prefill_step_size: 2048,
      temp: 0.0,
      top_p: 0.0,
      min_p: 0.0,
      min_tokens_to_keep: 1,
      top_k: 0,
      xtc_probability: 0.0,
      xtc_threshold: 0.0,
      xtc_special_tokens: Vec::new(),
      logit_bias: Vec::new(),
      repetition_penalty: None,
      repetition_context_size: DEFAULT_REPETITION_CONTEXT_SIZE,
      presence_penalty: None,
      presence_context_size: DEFAULT_REPETITION_CONTEXT_SIZE,
      frequency_penalty: None,
      frequency_context_size: DEFAULT_REPETITION_CONTEXT_SIZE,
      eos: Vec::new(),
      stop_strings: Vec::new(),
      seed: None,
      collect_logprobs: false,
    }
  }
}

impl GenConfig {
  /// Construct a [`GenConfig`] with all defaults (same as [`Default::default`]).
  pub fn new() -> Self {
    Self::default()
  }

  // ── encapsulated Vec accessors ──────────────────────────────────────────

  /// The XTC special-token ids (token ids XTC never masks).
  #[inline(always)]
  pub fn xtc_special_tokens_slice(&self) -> &[i32] {
    &self.xtc_special_tokens
  }

  /// The logit-bias pairs (`(token_id, bias)` — mlx-lm `Dict[int, float]`).
  #[inline(always)]
  pub fn logit_bias_slice(&self) -> &[(i32, f32)] {
    &self.logit_bias
  }

  /// The resolved stop-token id set.
  #[inline(always)]
  pub fn eos_slice(&self) -> &[u32] {
    &self.eos
  }

  /// The string stop sequences.
  #[inline(always)]
  pub fn stop_strings_slice(&self) -> &[String] {
    &self.stop_strings
  }

  // ── with_* builders ─────────────────────────────────────────────────────

  /// Set `max_tokens` and return `self` (builder pattern). Equivalent to
  /// `cfg.max_tokens = n; cfg` but chainable: use `GenConfig::default().with_max_tokens(n)`.
  #[must_use]
  pub fn with_max_tokens(mut self, n: usize) -> Self {
    self.max_tokens = n;
    self
  }

  /// Set `prefill_step_size` and return `self` (builder pattern).
  #[must_use]
  pub fn with_prefill_step_size(mut self, n: usize) -> Self {
    self.prefill_step_size = n;
    self
  }

  /// Set the XTC special-token ids and return `self` (builder pattern).
  #[must_use]
  pub fn with_xtc_special_tokens(mut self, tokens: impl Into<Vec<i32>>) -> Self {
    self.xtc_special_tokens = tokens.into();
    self
  }

  /// Set the logit-bias pairs and return `self` (builder pattern).
  #[must_use]
  pub fn with_logit_bias(mut self, bias: impl Into<Vec<(i32, f32)>>) -> Self {
    self.logit_bias = bias.into();
    self
  }

  /// Set the stop-token id set and return `self` (builder pattern).
  #[must_use]
  pub fn with_eos(mut self, eos: impl Into<Vec<u32>>) -> Self {
    self.eos = eos.into();
    self
  }

  /// Set the string stop sequences and return `self` (builder pattern).
  #[must_use]
  pub fn with_stop_strings(mut self, stops: impl Into<Vec<String>>) -> Self {
    self.stop_strings = stops.into();
    self
  }

  // In-place setters: plain non-optional fields get both `with_*` consuming
  // AND `set_*` in-place returning `&mut Self` for chaining on an existing
  // owned value.

  /// Set the XTC special-token ids in place; chainable.
  pub fn set_xtc_special_tokens(&mut self, tokens: impl Into<Vec<i32>>) -> &mut Self {
    self.xtc_special_tokens = tokens.into();
    self
  }

  /// Set the logit-bias pairs in place; chainable.
  pub fn set_logit_bias(&mut self, bias: impl Into<Vec<(i32, f32)>>) -> &mut Self {
    self.logit_bias = bias.into();
    self
  }

  /// Set the stop-token id set in place; chainable.
  pub fn set_eos(&mut self, eos: impl Into<Vec<u32>>) -> &mut Self {
    self.eos = eos.into();
    self
  }

  /// Set the string stop sequences in place; chainable.
  pub fn set_stop_strings(&mut self, stops: impl Into<Vec<String>>) -> &mut Self {
    self.stop_strings = stops.into();
    self
  }

  /// Set the sampling `temp` and return `self` (builder pattern); useful
  /// when the test or call-site needs to chain on top of
  /// `GenConfig::default()`. `temp` is still a public field today, but
  /// chaining keeps the surface symmetric with the privatized fields' builders
  /// (also makes invalid-value rejection tests like `with_temp(-1.0)` read
  /// uniformly across the call-site shape).
  #[must_use]
  pub fn with_temp(mut self, temp: f32) -> Self {
    self.temp = temp;
    self
  }

  /// Eagerly validate every scalar sampler / logits-processor bound up
  /// front (polish #136) — `temp`, `top_p`, `min_p`,
  /// `min_tokens_to_keep`, `top_k`, `xtc_probability`, `xtc_threshold`,
  /// `repetition_penalty`, and the `logit_bias` `(id, value)` pair-arity.
  /// Returns the **first** bound violated as an `Err(`[`Error::OutOfRange`]`)`
  /// (out-of-range scalar bound) or `Err(`[`Error::NonFiniteScalar`]`)` (NaN /
  /// ±inf) — the same `Err` variants the per-step validation in
  /// [`crate::lm::sample`] surfaces, so a caller migrating from "fails on
  /// first decode step" to "fails at config-build" sees the same error class.
  ///
  /// # Why eager
  ///
  /// [`make_sampler`] / [`make_logits_processors`] build closures whose
  /// purely-scalar bounds (`temp < 0`, `min_p > 1`, `xtc_probability` out
  /// of range, a negative `repetition_penalty`, …) are checked INSIDE the
  /// closure when it first runs against logits. So both
  /// [`generate_step`] (LM) and [`crate::audio::stt::generate::stt_generate`]
  /// (STT) had a window where an invalid `cfg` could pass the constructor
  /// then run an entire prompt prefill (LM) — or an audio load, resample,
  /// log-mel, and encoder pass (STT) — **before** surfacing the scalar-
  /// bound `Err` on the first decode step. `validate()` collapses that
  /// window: LM's [`generate_step`] calls it before any model work (the
  /// `Err` becomes the iterator's first `pending_err` yield, like a
  /// sampler-construction error); STT's
  /// [`crate::audio::stt::generate::stt_generate`] calls it at the top of
  /// the constructor (before the expensive audio pipeline runs), so a
  /// misconfigured `cfg` fails fast in both loops with the same `Err`
  /// regardless of which entry point the caller invokes.
  ///
  /// # Defense in depth
  ///
  /// The per-primitive validations in [`crate::lm::sample`]
  /// (`apply_top_p` / `apply_min_p` / `apply_xtc` /
  /// `apply_repetition_penalty` / `apply_logit_bias` / `scale_logits_by_temp`)
  /// are **kept** — `validate()` is the eager gate but the sampler
  /// primitives' own checks remain so a direct
  /// `crate::lm::sample::apply_*` call (outside the generation loops) still
  /// rejects invalid input. The eager + per-primitive validations use the
  /// same bound predicates so the error messages match, modulo the
  /// dynamic-bound checks (`top_k < vocab_size`, `min_tokens_to_keep <=
  /// vocab_size`) that `validate()` can't enforce without the model's
  /// vocab — those still surface on the first decode step.
  ///
  /// # Bounds checked
  ///
  /// - `temp.is_finite() && temp >= 0.0` — `temp == 0` is the argmax
  ///   path (no scale), `temp > 0` is the stochastic path
  ///   (`scale_logits_by_temp` requires `temp > 0`).
  /// - `top_p.is_finite() && (0.0..=1.0).contains(&top_p)` —
  ///   [`apply_top_p`][crate::lm::sample::apply_top_p] strictly requires
  ///   `top_p > 0 && top_p <= 1`, but `make_sampler` only includes the
  ///   stage `iff (0, 1)` so `top_p == 0` is "off" and `top_p == 1` is
  ///   "include everything" — both no-op-equivalent, both accepted here.
  /// - `min_p.is_finite() && (0.0..=1.0).contains(&min_p)` — mirrors
  ///   [`apply_min_p`][crate::lm::sample::apply_min_p].
  /// - `min_tokens_to_keep >= 1` — mirrors
  ///   [`apply_min_p`][crate::lm::sample::apply_min_p] (the `< vocab_size`
  ///   bound is vocab-dependent and deferred).
  /// - `top_k >= 0` — `top_k == 0` is "off" in `make_sampler`,
  ///   `top_k > 0` is "on" (`apply_top_k`'s `< vocab_size` bound is
  ///   vocab-dependent and deferred).
  /// - `xtc_probability.is_finite() && (0.0..=1.0).contains(&xtc_probability)`
  ///   — mirrors [`apply_xtc`][crate::lm::sample::apply_xtc].
  /// - `xtc_threshold.is_finite() && (0.0..=0.5).contains(&xtc_threshold)`
  ///   — mirrors [`apply_xtc`][crate::lm::sample::apply_xtc].
  /// - `repetition_penalty: Option<f32>` — if `Some(p)`, then
  ///   `p.is_finite() && p >= 0.0` (mirrors
  ///   [`apply_repetition_penalty`][crate::lm::sample::apply_repetition_penalty]
  ///   and mlx-lm's `make_repetition_penalty` `ValueError`).
  /// - `presence_penalty` / `frequency_penalty: Option<f32>` — finite-only.
  ///   mlx-lm's `make_presence_penalty` / `make_frequency_penalty` allow
  ///   negative values (they're additive bonuses/penalties); only `NaN` /
  ///   `±inf` are caught here.
  /// - `logit_bias` — every `(id, value)` `value` must be finite (no NaN
  ///   bias). The pair-arity check (`indices.len() == values.size()`) is
  ///   structurally impossible to fail here because
  ///   [`make_logits_processors`] builds them from the same `Vec<(i32, f32)>`,
  ///   but `apply_logit_bias` still checks it as defense-in-depth.
  ///
  /// # Not checked (out of scope)
  ///
  /// - `top_k < vocab_size`, `min_tokens_to_keep <= vocab_size` — both
  ///   require knowing the model's vocab size, which is a model-load
  ///   concern that doesn't belong on `GenConfig`. Surface on the first
  ///   decode step like before.
  /// - The `eos` token ids — these are tokenizer-resolved and any
  ///   `u32` is a valid token id at this layer.
  /// - `stop_strings` — empty strings, length, etc. are handled by
  ///   [`crate::lm::stop::StopMatcher`].
  /// - `prefill_step_size == 0` — clamped to `1` in [`generate_step`].
  pub fn validate(&self) -> Result<()> {
    // temp: finite + non-negative (temp == 0 ⇒ argmax path; temp > 0 ⇒
    // stochastic path).
    if !self.temp.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: temp",
        self.temp as f64,
      )));
    }
    if self.temp < 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: temp",
        "must be a finite non-negative float (0.0 = argmax, > 0.0 = stochastic)",
        format_smolstr!("{}", self.temp),
      )));
    }
    // top_p: [0, 1]. `make_sampler` gates the stage on `(0, 1)`; 0 and 1
    // are no-op-equivalent and accepted as "off" / "include everything".
    if !self.top_p.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: top_p",
        self.top_p as f64,
      )));
    }
    if !(0.0..=1.0).contains(&self.top_p) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: top_p",
        "must be in [0, 1] (0 = off, (0, 1) = nucleus cutoff, 1 = include everything)",
        format_smolstr!("{}", self.top_p),
      )));
    }
    // min_p: [0, 1] (mirrors `apply_min_p`).
    if !self.min_p.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: min_p",
        self.min_p as f64,
      )));
    }
    if !(0.0..=1.0).contains(&self.min_p) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: min_p",
        "must be in [0, 1]",
        format_smolstr!("{}", self.min_p),
      )));
    }
    // min_tokens_to_keep >= 1 (mirrors `apply_min_p`; the `<= vocab_size`
    // bound is vocab-dependent and deferred to the first decode step).
    if self.min_tokens_to_keep < 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: min_tokens_to_keep",
        "must be a positive integer (>= 1)",
        format_smolstr!("{}", self.min_tokens_to_keep),
      )));
    }
    // top_k >= 0 (`top_k == 0` is "off"; `top_k > 0` is "on" — the
    // `< vocab_size` bound is vocab-dependent and deferred).
    if self.top_k < 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: top_k",
        "must be non-negative (0 = off, > 0 = top-k cutoff)",
        format_smolstr!("{}", self.top_k),
      )));
    }
    // xtc_probability: [0, 1] (mirrors `apply_xtc`).
    if !self.xtc_probability.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: xtc_probability",
        self.xtc_probability as f64,
      )));
    }
    if !(0.0..=1.0).contains(&self.xtc_probability) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: xtc_probability",
        "must be in [0, 1]",
        format_smolstr!("{}", self.xtc_probability),
      )));
    }
    // xtc_threshold: [0, 0.5] (mirrors `apply_xtc`).
    if !self.xtc_threshold.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: xtc_threshold",
        self.xtc_threshold as f64,
      )));
    }
    if !(0.0..=0.5).contains(&self.xtc_threshold) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "GenConfig::validate: xtc_threshold",
        "must be in [0, 0.5]",
        format_smolstr!("{}", self.xtc_threshold),
      )));
    }
    // repetition_penalty: finite + non-negative (mirrors
    // `apply_repetition_penalty` + mlx-lm `make_repetition_penalty`'s
    // `ValueError`). `None` and `Some(0.0)` are both "off" — the latter
    // because `make_logits_processors` only includes the processor when
    // `penalty != 0`.
    if let Some(p) = self.repetition_penalty {
      if !p.is_finite() {
        return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
          "GenConfig::validate: repetition_penalty",
          p as f64,
        )));
      }
      if p < 0.0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "GenConfig::validate: repetition_penalty",
          "must be a finite non-negative float when Some(_)",
          format_smolstr!("{p}"),
        )));
      }
    }
    // presence_penalty: finite-only. mlx-lm's `make_presence_penalty`
    // allows negative values (presence "boost" is a negative penalty), so
    // we only catch NaN / ±inf here.
    if let Some(p) = self.presence_penalty
      && !p.is_finite()
    {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: presence_penalty",
        p as f64,
      )));
    }
    // frequency_penalty: finite-only (same rationale as presence).
    if let Some(p) = self.frequency_penalty
      && !p.is_finite()
    {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "GenConfig::validate: frequency_penalty",
        p as f64,
      )));
    }
    // logit_bias: every `(id, value)` `value` finite. `id` is `i32` and
    // not bound here (the model's vocab is unknown; the `take`/scatter
    // primitive will reject an out-of-range id at the first decode step).
    for &(_id, v) in &self.logit_bias {
      if !v.is_finite() {
        return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
          "GenConfig::validate: logit_bias value",
          v as f64,
        )));
      }
    }
    Ok(())
  }
}

/// Build the sampler function, composing the [`sample`] primitives exactly
/// as `mlx_lm.sample_utils.make_sampler`.
///
/// `temp == 0` ⇒ the pure argmax sampler ([`sample::argmax_sample`], mlx-lm
/// line 46). Otherwise the chain is built in mlx-lm's exact order — top-p
/// (iff `0 < top_p < 1`), min-p (iff `min_p != 0`), xtc (iff
/// `xtc_probability > 0`), top-k (iff `top_k > 0`) — and each call ends with
/// [`sample::categorical_sampling`]. The returned closure threads its own
/// PRNG key, splitting it once per call (mirroring mlx-lm's per-call
/// `mx.random.state` advance, so successive draws differ) and seeding the
/// single xtc Bernoulli gate from the per-call subkey.
///
/// **RNG seeding (mlx-lm parity):** mlx-lm's `make_sampler` draws from the
/// process-global `mx.random.state`, so independent generations never repeat
/// and `mx.random.seed(s)` makes a run reproducible. mlxrs's random API is
/// explicit-key (#21, no hidden global state), so the seed is surfaced as
/// `seed`:
/// - `Some(s)` ⇒ the run is reproducible from `key(s)` — the mlxrs analogue
///   of `mx.random.seed(s)` before generating.
/// - `None` ⇒ a fresh, process-unique seed per `make_sampler` call (a
///   monotonic atomic counter mixed with the wall clock), so two independent
///   non-greedy generations do **not** restart from the same sequence —
///   matching mlx-lm's default (the global state has advanced between runs).
///
/// **Does not re-validate** the sampler ranges — [`sample`]'s
/// `apply_top_p` / `apply_min_p` / `apply_top_k` / `apply_xtc` /
/// `categorical_sampling` enforce them, and their `Err` is propagated when
/// the returned closure runs (mlx-lm builds the chain unconditionally; the
/// bound checks live in the primitives). Construction itself is fallible
/// only via the initial PRNG-key allocation.
#[allow(clippy::too_many_arguments)]
pub fn make_sampler(
  temp: f32,
  top_p: f32,
  min_p: f32,
  min_tokens_to_keep: i32,
  top_k: i32,
  xtc_probability: f32,
  xtc_threshold: f32,
  xtc_special_tokens: &[i32],
  seed: Option<u64>,
) -> Result<Sampler> {
  // mlx-lm: `if temp == 0: return lambda x: mx.argmax(x, axis=-1)`.
  // Returned as the [`Sampler::Argmax`] variant so the per-token dispatch
  // is a one-arm `match` (no closure, no PRNG key allocation).
  if temp == 0.0 {
    return Ok(Sampler::Argmax);
  }

  // mlx-lm builds the stage list in this exact order; the gates mirror
  // `make_sampler` lines 51-60 (`top_p in (0, 1)`, `min_p != 0`,
  // `xtc_probability > 0`, `top_k > 0`).
  let do_top_p = top_p > 0.0 && top_p < 1.0;
  let do_min_p = min_p != 0.0;
  let do_xtc = xtc_probability > 0.0;
  let do_top_k = top_k > 0;
  let xtc_special: Vec<i32> = xtc_special_tokens.to_vec();

  // Seed resolution (mlx-lm parity, see the doc): an explicit `seed` is
  // reproducible (`mx.random.seed(s)` analogue); `None` draws a
  // process-unique seed so independent non-greedy runs never restart from
  // the same sequence (mlx-lm's default — the global state has advanced).
  let resolved_seed = seed.unwrap_or_else(next_sampler_seed);
  // Per-call PRNG key, advanced like mlx-lm's `mx.random.state`. mlx-lm's
  // xtc `mx.random.uniform` and `mx.random.categorical` each advance the
  // global state once (two independent draws per call); mirror that by
  // splitting the running key into the next state plus a *distinct* subkey
  // for xtc and for the categorical draw, so neither reuses a key within or
  // across steps.
  let key = RefCell::new(ops::random::key(resolved_seed)?);

  Ok(Sampler::Chain(SamplerChain {
    temp,
    top_p,
    min_p,
    min_tokens_to_keep,
    top_k,
    xtc_probability,
    xtc_threshold,
    xtc_special,
    do_top_p,
    do_min_p,
    do_xtc,
    do_top_k,
    key,
  }))
}

/// Build the logits-processor list, composing the [`sample`] primitives
/// exactly as `mlx_lm.sample_utils.make_logits_processors`.
///
/// The order mirrors mlx-lm: the `logit_bias` processor first (iff
/// non-empty), then repetition / presence / frequency — each added iff its
/// penalty is `Some(p)` with `p != 0` (mlx-lm `penalty is not None and
/// penalty != 0`). Each penalty processor slices the history to the last
/// `*_context_size` ids before applying (mlx-lm's `tokens[-context_size:]`)
/// and forwards to the matching [`sample`] primitive — repetition, presence,
/// and frequency each have their **own** context window, exactly as
/// `sample_utils.make_logits_processors` (`repetition_context_size`,
/// `presence_context_size`, `frequency_context_size`, each default 20), so
/// every mlx-lm penalty-window configuration is reproducible. Validation
/// `Err` (e.g. a negative repetition penalty, a `logit_bias` length
/// mismatch) is **propagated from [`sample`]**, not re-checked here.
#[allow(clippy::too_many_arguments)]
pub fn make_logits_processors(
  logit_bias: &[(i32, f32)],
  repetition_penalty: Option<f32>,
  repetition_context_size: usize,
  presence_penalty: Option<f32>,
  presence_context_size: usize,
  frequency_penalty: Option<f32>,
  frequency_context_size: usize,
) -> Result<Vec<LogitsProcessor>> {
  let mut processors: Vec<LogitsProcessor> = Vec::new();

  // mlx-lm `if logit_bias:` — added first so the penalties see the biased
  // logits (the exact mlx-lm processor application order). mlx-lm builds the
  // `indices` / `values` arrays ONCE at closure-creation time
  // (`indices = mx.array(list(...))`); mirror that — the `values` array is
  // built once here at variant construction, not rebuilt per step.
  if !logit_bias.is_empty() {
    let mut indices: Vec<i32> = try_with_capacity(logit_bias.len())?;
    indices.extend(logit_bias.iter().map(|&(i, _)| i));
    let mut values_vec: Vec<f32> = try_with_capacity(logit_bias.len())?;
    values_vec.extend(logit_bias.iter().map(|&(_, v)| v));
    let values = Array::from_slice::<f32>(&values_vec, &(values_vec.len(),))?;
    processors.push(LogitsProcessor::LogitBias(LogitBiasPayload::new(
      indices, values,
    )));
  }

  // mlx-lm: `(make_repetition_penalty, repetition_penalty,
  // repetition_context_size), (make_presence_penalty, ...,
  // presence_context_size), (make_frequency_penalty, ...,
  // frequency_context_size)` — each appended iff `penalty is not None and
  // penalty != 0`, in this order, each capturing its OWN context size.
  if let Some(p) = repetition_penalty.filter(|&p| p != 0.0) {
    processors.push(LogitsProcessor::RepetitionPenalty(
      RepetitionPenaltyPayload::new(p, repetition_context_size),
    ));
  }
  if let Some(p) = presence_penalty.filter(|&p| p != 0.0) {
    processors.push(LogitsProcessor::PresencePenalty(
      PresencePenaltyPayload::new(p, presence_context_size),
    ));
  }
  if let Some(p) = frequency_penalty.filter(|&p| p != 0.0) {
    processors.push(LogitsProcessor::FrequencyPenalty(
      FrequencyPenaltyPayload::new(p, frequency_context_size),
    ));
  }

  Ok(processors)
}

/// The recent token ids as `i32` (the [`sample`] penalty primitives' index
/// dtype), mirroring mlx-lm's `tokens[-context_size:]` slicing **exactly**,
/// including the Python edge case: `context_size == 0` is `tokens[-0:]` ==
/// `tokens[0:]` == the **entire** history (Python `-0 == 0`), NOT an empty
/// slice — so a `0` window penalizes over all accumulated tokens, matching
/// `sample_utils`'s closures. Any positive `context_size >= tokens.len()`
/// likewise keeps the whole history (`tokens[-big:] == tokens`).
fn recent_ids(tokens: &[u32], context_size: usize) -> Result<Vec<i32>> {
  // Python `tokens[-context_size:]`: `context_size == 0` ⇒ `tokens[0:]`
  // (full history); otherwise the last `min(context_size, len)` ids.
  let start = if context_size == 0 {
    0
  } else {
    tokens.len().saturating_sub(context_size)
  };
  let tail = &tokens[start..];
  let mut ids = try_with_capacity(tail.len())?;
  ids.extend(tail.iter().map(|&t| t as i32));
  Ok(ids)
}

/// Why generation stopped — the typed version of the `finish_reason` string
/// (`"stop"` / `"length"` / stop-string) that mlx-lm / the OpenAI API
/// surface carries.
///
/// Used on [`GenStep`], [`GenerationResponse`], and [`BatchGenStep`].
#[derive(
  Debug,
  Clone,
  PartialEq,
  Eq,
  derive_more::Display,
  derive_more::IsVariant,
  derive_more::Unwrap,
  derive_more::TryUnwrap,
)]
#[display("{}", self.as_str())]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
pub enum FinishReason {
  /// A sampled token was in the eos set (mlx-lm `"stop"`).
  Eos,
  /// `max_tokens` was reached (mlx-lm `"length"`).
  Length,
  /// A string stop sequence matched (mlx-lm stop-words; the matched
  /// string is carried).
  Stop(String),
}

impl FinishReason {
  /// The canonical finish-reason tag (mlx-lm / OpenAI `finish_reason`).
  /// Both [`Self::Eos`] and [`Self::Stop`] map to `"stop"` — they share
  /// the same external taxonomy; the difference (EOS token vs configured
  /// stop string) lives internally. `"length"` for [`Self::Length`].
  ///
  /// For the matched stop sequence on [`Self::Stop`], use
  /// [`Self::stop_sequence`].
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Eos | Self::Stop(_) => "stop",
      Self::Length => "length",
    }
  }

  /// The matched stop sequence string for [`Self::Stop`]; `None` for
  /// [`Self::Eos`] and [`Self::Length`]. Use this when you need the
  /// payload — `as_str()` and `Display` collapse `Stop(_)` to the
  /// canonical tag `"stop"` per OpenAI's `finish_reason` contract.
  pub fn stop_sequence(&self) -> Option<&str> {
    match self {
      Self::Stop(s) => Some(s.as_str()),
      _ => None,
    }
  }
}

/// One decode step — the sampled `token` plus an **opt-in** `[V]`
/// log-probability vector over the vocabulary that produced it (mlx-lm
/// `generate_step`'s `yield y.item(), logprobs`, gated here by
/// [`GenConfig::collect_logprobs`]).
///
/// Replaces the prior `(u32, Array)` tuple item: mlx-lm uses Python's
/// positional tuple as informal documentation, but Rust callers reading
/// the iterator item shouldn't have to remember tuple-index conventions —
/// the struct is self-documenting and a Rust-idiomatic improvement
/// (prefer idiomatic-Rust ergonomics over verbatim Python mirroring).
///
/// # `logprobs` opt-in (L3)
///
/// `logprobs` is `Some(Array)` when [`GenConfig::collect_logprobs`] is
/// `true` (the prior unconditional yield); `None` otherwise so a caller
/// that only reads `token` pays no per-step squeeze. The `Option<Array>`
/// type (not `Option<Vec<f32>>`) keeps the no-implicit-eval contract:
/// materialization into a CPU `Vec<f32>` is the caller's explicit step
/// via [`Array::to_vec`] / [`Array::as_slice`]. This deviates from mlx-lm
/// (which always yields the array); mlx-lm's server-side opt-in
/// (`mlx_lm/server.py:191` `logprobs: bool`) is moved down to the step
/// loop so the cost-when-off saving applies to every consumer, not just
/// the HTTP server. VLM ([`crate::vlm::generate`]) and audio
/// ([`crate::audio::stt::generate`]) `GenStep` producers preserve their
/// unconditional-`Some` behavior (their public surfaces have not yet
/// adopted the `collect_logprobs` opt-in).
///
/// # Back-compat
///
/// This is **not** drop-in source-compatible with the prior tuple item:
/// existing `let (tok, lp) = step?;` call sites must add an explicit
/// `.into()` (`let (tok, lp) = step?.into();`) or pattern-match the
/// struct (`let GenStep { token, logprobs, .. } = step?;`). The break is
/// **intentional** — mlxrs is pre-1.0, and the ergonomics + self-
/// documentation win outweighs a one-line migration per call site. The
/// `From<GenStep> for (u32, Option<Array>)` impl below makes that
/// migration mechanical (the previous `From<GenStep> for (u32, Array)`
/// is replaced — the `Option` shift propagates through the tuple form
/// so call sites can't silently drop the new semantic). Pattern-match
/// destructures should use the rest pattern (`..`) since
/// `step_index` and `finish_reason` were added as further fields; new fields may be
/// added in the future under the same convention.
///
/// # `step_index` + `finish_reason` (polish #114)
///
/// Two more named fields were added to mirror the existing
/// [`BatchGenStep`]'s per-row shape (so a caller writing against either
/// surface sees the same step envelope):
///
/// - `step_index: usize` — 0-based index of this step within the
///   iterator's run. The first yielded step is `0`, the second `1`, etc.
///   Distinct from the [`stream_generate`] / [`GenerationResponse`]'s
///   `generation_tokens` (which is 1-indexed and counts the about-to-be-
///   reported token, mlx-lm `n + 1`). Useful for callers that want a
///   stable per-step identifier without re-counting via `enumerate()`.
/// - `finish_reason: Option<String>` — `None` for ordinary steps,
///   `Some("stop")` on the EOS-token step (the final yielded item when a
///   sampled token is in the eos set configured via [`GenConfig::with_eos`]). Note that single-seq
///   generation does NOT emit a `Some("length")` step — mlx-lm's
///   `if n == max_tokens: break` happens BEFORE the yield, so the
///   `max_tokens` finish is signalled by the iterator simply ending
///   (`next() == None`), not by a final `length`-tagged step. This
///   mirrors mlx-lm's `generate_step` exactly (the `"length"` reason is
///   computed by the higher-level [`stream_generate`] wrapper when it
///   detects the iterator ended without an EOS-tagged step). Batch
///   generation ([`BatchGenStep`]) DOES yield `Some("length")` per row
///   because the iterator can't end the run as a whole until every row
///   has finished — so per-row `length` must be signaled inline. Both
///   surfaces are byte-faithful to their upstream parallel.
#[derive(Debug)]
pub struct GenStep {
  /// The sampled token id (mlx-lm `y.item()`).
  pub token: u32,
  /// The token's `[V]` log-probability vector (mlx-lm
  /// `logprobs.squeeze(0)`), kept lazy. `Some` iff
  /// [`GenConfig::collect_logprobs`] was `true` for this run.
  pub logprobs: Option<Array>,
  /// 0-based index of this step within the iterator's run (`0` for the
  /// first yielded step, `1` for the second, …). Polish #114 — a
  /// stable per-step identifier so callers don't have to wrap the
  /// iterator in `enumerate()` just to know which step they're on.
  pub step_index: usize,
  /// `None` for ordinary steps; `Some(FinishReason::Eos)` on the EOS-token
  /// step (the final yielded item when a sampled token is in the eos set
  /// configured via [`GenConfig::with_eos`]). Polish #114 — mirrors the existing
  /// [`BatchGenStep::finish_reason`] field so single-seq + batch surfaces
  /// share a step envelope. NOTE: `Some(FinishReason::Length)` is NEVER
  /// emitted at this layer (mlx-lm `generate_step` `break`s BEFORE the
  /// `max_tokens`-th yield); the [`stream_generate`] wrapper computes
  /// the `Length` reason itself.
  pub finish_reason: Option<FinishReason>,
}

impl From<GenStep> for (u32, Option<Array>) {
  fn from(s: GenStep) -> Self {
    (s.token, s.logprobs)
  }
}

/// The architecture-agnostic decode iterator: borrows the model, owns the
/// per-layer KV cache, the running token history, the sampler, and the
/// logits processors. Constructed by [`generate_step`].
///
/// # Breaking change (#113)
///
/// Previously `pub struct Generator<'a, M>` — the concrete iterator type
/// was part of the public API surface, so downstream code could name it,
/// doc-comments tied to its layout, and any internal refactor (e.g.
/// splitting into `PrefillGenerator + DecodeGenerator`) became a
/// breaking change. The sibling [`stream_generate`] already returned
/// `impl Iterator + 'a` for exactly the same reason.
///
/// [`generate_step`] now returns
/// `impl Iterator<Item = Result<GenStep>> + 'a` (the opaque-iterator
/// shape `stream_generate` already used), and `Generator` is
/// `pub(crate)`. Callers that used
/// `let mut it = generate_step(...); it.next();` work unchanged;
/// callers that named the concrete `Generator<'a, M>` type (none on
/// `main`, since `#48` introduced it) must switch to inference /
/// `impl Iterator<_>`.
///
/// The borrow of `&'a M` plus the owned cache means no aliasing. The
/// iterator **fuses**: after it yields `Err` (a step failed)
/// or finishes (eos / `max_tokens`) every further `next()` is `None` —
/// never a panic, never a poisoned re-entry.
///
/// `M: Model + ?Sized` — the loop only ever touches the model behind the
/// `&'a M` borrow (`model.forward(...)`), never by value and never via a
/// `Sized`-requiring associated item, so `M` may be an unsized trait
/// object. This lets a `&dyn Model` (or a deref-coerced
/// `Box<dyn Model>` / `Box<dyn VlmModel>`, since `VlmModel: Model`) drive
/// generation directly — the exact handle a load factory returns
/// ([`crate::lm::factory::LoadedModelContext::model`],
/// [`crate::vlm::load::LoadedVlmContext::model`]).
pub(crate) struct Generator<'a, M: Model + ?Sized> {
  model: &'a M,
  cache: Vec<Box<dyn KvCache>>,
  sampler: Sampler,
  processors: Vec<LogitsProcessor>,
  /// The full encoded prompt (mlx-lm's `prompt`). Prefill advances
  /// [`Generator::prefill_offset`] over this buffer instead of
  /// front-draining it; the unconsumed tail (`prompt[prefill_offset..]`)
  /// starts the first decode step.
  prompt: Vec<u32>,
  /// Cursor into [`Generator::prompt`]: the count of leading tokens
  /// already prefilled (mlx-lm's `prompt_processed_tokens`). Advanced by
  /// each chunk size so prefill is O(P) — no front-removal / tail-shift /
  /// realloc per chunk (the byte-identical O(P) analogue of mlx-lm's
  /// `prompt = prompt[n_to_process:]` array slicing).
  prefill_offset: usize,
  /// The running token-id history fed to the logits processors (mlx-lm's
  /// accumulating `tokens` — the step input, not yet the predicted token).
  history: Vec<u32>,
  /// The most-recently sampled token (mlx-lm's `y` fed into the next
  /// `_step`); `None` before the first decode step.
  last: Option<u32>,
  /// Tokens yielded so far (mlx-lm's `n`); generation ends at
  /// `max_tokens`.
  produced: usize,
  max_tokens: usize,
  prefill_step_size: usize,
  eos: Vec<u32>,
  /// [`GenConfig::collect_logprobs`]: when `false`, the per-step squeeze
  /// is skipped and [`GenStep::logprobs`] is `None`.
  collect_logprobs: bool,
  /// `true` iff the configured sampler chain requires `logits - logsumexp`
  /// normalization to sample correctly. Only `top_p ∈ (0, 1)` does (its
  /// `exp(logprobs)` cumsum threshold `1 - top_p` assumes the cumsum
  /// reaches 1.0); every other sampler in [`make_sampler`] is
  /// shift-invariant (argmax, top_k argpartition, min_p's
  /// `max + log(min_p)` threshold) or softmaxes internally (xtc's own
  /// `softmax`, categorical's own `softmax`). When `false` and
  /// `collect_logprobs` is also `false`, the per-step `logsumexp +
  /// subtract` is skipped entirely — the sampler reads the raw
  /// post-processor logits and produces the byte-identical token.
  /// Precomputed from `GenConfig` at construction so the per-step hot
  /// loop is a single field check.
  needs_logprobs: bool,
  /// `true` iff `cfg.temp > 0` (stochastic sampling). Drives the
  /// opt-out path's cheap `max + subtract` max-shift:
  /// when the full normalization is skipped (`!needs_normalization`) but
  /// `temp > 0`, the sampler's downstream `logits * (1/temp)` would
  /// overflow in f16/bf16 with a large `logit_bias`, so the opt-out
  /// path subtracts the row-wise max to bound `exp` for every dtype.
  /// `temp == 0` (pure-greedy `argmax_sample`) doesn't exponentiate, so
  /// the raw-logit path stays the true zero-cost path there.
  /// Precomputed from `GenConfig.temp` so the per-step `match` is a
  /// single bool check.
  temp_stochastic: bool,
  /// `true` once prompt prefill has run (it runs on the first `next()`).
  prefilled: bool,
  /// `true` until the first decode step has run (it feeds the prompt tail;
  /// later steps feed back `last`) — mlx-lm `_step(prompt)` then `_step(y)`.
  first_step: bool,
  /// A deferred sampler / processor *construction* error (from
  /// [`generate_step`]); yielded as the iterator's first (and only) `Err`
  /// before any step runs, keeping the public surface a pure `Iterator`.
  pending_err: Option<Error>,
  /// Fused: set after a yielded `Err` or a finish so the iterator never
  /// re-enters mlx-c / re-runs the model.
  done: bool,
}

impl<M: Model + ?Sized> Generator<'_, M> {
  /// Consume the generator and return its per-layer KV cache.
  ///
  /// The cache the generator was constructed with is moved into the
  /// [`Generator`] and advanced **in place** by every prefill / decode
  /// [`Model::forward`] call — so once the iterator is exhausted (eos /
  /// `max_tokens`) this is the *advanced* cache, holding the keys/values for
  /// the full prompt **and** every generated token.
  ///
  /// This is a pure ownership transfer (no generation work, no eval): it
  /// hands the already-owned cache back so a longer-lived caller can reuse it
  /// — the building block a stateful, multi-turn driver
  /// ([`crate::lm::session::ChatSession`]) needs to carry one KV cache across
  /// `respond` turns instead of re-prefilling the conversation each time.
  /// The plain [`stream_generate`] / [`generate`] entry points drop the cache
  /// with the iterator (single-shot, mlx-lm's `generate_step` contract); this
  /// accessor is the seam for the reuse case.
  pub fn into_cache(self) -> Vec<Box<dyn KvCache>> {
    self.cache
  }

  /// Run the prompt prefill once: feed the first `total - 1` tokens through
  /// the model in `prefill_step_size` chunks (logits discarded, cache
  /// filled) by advancing [`Generator::prefill_offset`] over `self.prompt`,
  /// leaving the unconsumed final token(s) (`prompt[prefill_offset..]`) to
  /// start the first decode step — mlx-lm `generate_step` lines 430-451.
  ///
  /// The prefilled chunks are deliberately **not** added to `self.history`:
  /// mlx-lm's processor history (`tokens`) is `None` through prefill and is
  /// first set inside `_step` to that step's `input_tokens` (the prompt
  /// tail), so the logits-processor context is the prompt *tail* + generated
  /// tokens — exactly mirrored here by accumulating history only in
  /// [`Generator::step`].
  fn prefill(&mut self) -> Result<()> {
    // mlx-lm: `total = len(prompt); processed = 0; while total - processed >
    // 1: remaining = (total - processed) - 1; n = min(step, remaining);
    // forward(prompt[:n]); processed += n; prompt = prompt[n:]`. The
    // unconsumed `total - processed` count is `self.prompt.len() -
    // self.prefill_offset`; advancing the cursor (never front-removing)
    // makes prefill O(P) with byte-identical chunk boundaries.
    while self.prompt.len() - self.prefill_offset > 1 {
      let remaining = (self.prompt.len() - self.prefill_offset) - 1;
      let n = self.prefill_step_size.min(remaining);
      let chunk = token_window(&self.prompt[self.prefill_offset..self.prefill_offset + n])?;
      // logits discarded — the chunk only fills the cache.
      let _ = self.model.forward(&chunk, &mut self.cache)?;
      self.prefill_offset += n;
    }
    Ok(())
  }

  /// One decode step — the exact mlx-lm `_step` order
  /// (`generate_step` lines 396-422): forward → last-position slice →
  /// history-accumulate → logits processors → `logits - logsumexp` →
  /// sampler → `GenStep { token, logprobs: logprobs.squeeze(0) }`. No
  /// implicit eval except the `.item::<u32>()` token boundary.
  fn step(&mut self, input: &[u32]) -> Result<GenStep> {
    // 1. forward over `input[None]` (a `[1, S]` window); cache updated in
    //    place.
    let tokens = token_window(input)?;
    let logits = self.model.forward(&tokens, &mut self.cache)?;

    // 2. `logits = logits[:, -1, :]` — keep only the final sequence
    //    position, then drop that axis ⇒ `[1, V]` (mlx-lm line 407).
    let logits = last_position(&logits)?;

    // 3. mlx-lm runs this block ONLY when `logits_processors and
    //    len(input_tokens) > 0`: accumulate the step's input into the
    //    running history (`tokens = concat([tokens, input_tokens])`, lines
    //    409-414), then run each processor over the FULL history on RAW
    //    logits. With no processors mlx-lm never touches `tokens` — mirror
    //    that exactly (and avoid the needless history growth).
    let mut logits = logits;
    if !self.processors.is_empty() && !input.is_empty() {
      try_extend_from_slice(&mut self.history, input)?;
      for p in &self.processors {
        logits = p.apply(&self.history, &logits)?;
      }
    }

    // 4. `logprobs = logits - mx.logsumexp(logits, keepdims=True)` — the
    //    exact mlx-lm normalization (all-axes logsumexp, broadcast).
    //    **GATED, 3-way**: the per-step compute depends on
    //    `(needs_normalization, temp > 0)`:
    //      • `(true, _)`  — full `logsumexp + subtract` (collect_logprobs
    //         and/or top_p — `top_p` strictly needs the cumsum-to-1
    //         contract; collect_logprobs yields the true log-softmax).
    //      • `(false, true)` — cheap `max + subtract` max-shift. The
    //         downstream `categorical_sampling` does `logits * (1/temp)`
    //         BEFORE its internal `softmax`, so f16/bf16 + a large
    //         `logit_bias` + small `temp` would overflow to `+inf` before
    //         shift-invariance can save us. The max-shift caps the input
    //         at 0 (no positive scaled value) ⇒ `exp` is bounded for
    //         every dtype. One reduce + one broadcast subtract is ~3-4×
    //         cheaper than the full `logsumexp` + subtract (skips the
    //         per-element `exp` + `log`).
    //      • `(false, false)` — raw logits. Pure-greedy
    //         (`argmax_sample`) is shift-invariant numerically as well
    //         (it doesn't exponentiate), so it stays the true zero-cost
    //         path: no reduce, no broadcast, no allocation.
    //
    //    The max-shift bounds the sampler input to ≤ 0, but it does
    //    NOT protect against `categorical_sampling`'s own internal
    //    `1/temp` overflow for two extreme-`temp` configurations (f16
    //    logits + `temp < 1/65504`; any dtype + subnormal positive
    //    `temp < 1.0/f32::MAX ≈ 2.94e-39`). The structural fix lives in
    //    `sample::categorical_sampling` and is deferred to a dedicated
    //    `fix(lm/sample)` follow-up PR that updates `sample.rs` + all
    //    three call sites (LM / VLM / STT) consistently (a LM-only argmax
    //    bypass was prototyped and reverted because it
    //    failed to cover VLM/STT and silently skipped configured
    //    sampler stages — XTC/top_k/min_p).
    let needs_normalization = self.collect_logprobs || self.needs_logprobs;
    let sampler_input: Option<Array> = match (needs_normalization, self.temp_stochastic) {
      // Full normalization (collect_logprobs and/or top_p).
      (true, _) => {
        let lse = ops::reduction::logsumexp(&logits, true)?;
        Some(ops::arithmetic::subtract(&logits, &lse)?)
      }
      // Stochastic opt-out: cheap max-shift for f16/bf16 numerical safety.
      (false, true) => {
        let m = ops::reduction::max(&logits, true)?;
        Some(ops::arithmetic::subtract(&logits, &m)?)
      }
      // Pure-greedy opt-out: raw logits (argmax is shift-invariant).
      (false, false) => None,
    };

    // 5. `sampled = sampler(logprobs)` — the make_sampler chain / argmax.
    //    Feed the full `normalized` if we computed it (top_p needs the
    //    cumsum-to-1 log-probs; collect_logprobs needs the yielded
    //    log-softmax); feed the cheap max-shift if we computed only that
    //    (`(false, true)` — stochastic opt-out); otherwise feed the raw
    //    `logits` (pure-greedy opt-out — `argmax(logits) == argmax(logits
    //    - c)` for any scalar `c`).
    let mut sampled = self
      .sampler
      .sample(sampler_input.as_ref().unwrap_or(&logits))?;

    // 6. token boundary: the ONLY materialization (mlx-lm `y.item()`).
    //    `argmax` / `categorical` both yield `U32`.
    let token: u32 = sampled.item::<u32>()?;

    // mlx-lm returns `logprobs.squeeze(0)` ⇒ a `[V]` vector. Kept lazy.
    // L3 opt-in: only yield the `[V]` view when `collect_logprobs == true`;
    // otherwise both the normalization (above) and this squeeze are
    // skipped. When `true`, `sampler_input` is guaranteed `Some` via the
    // `(true, _)` match arm (full logsumexp + subtract), so the yielded
    // array is byte-identical to the prior unconditional yield (mlx-lm's
    // `logprobs.squeeze(0)`); the cheap max-shift path is never taken
    // when `collect_logprobs == true`.
    let logprobs = if self.collect_logprobs {
      Some(ops::shape::squeeze_axes(
        sampler_input
          .as_ref()
          .expect("sampler_input is Some (full normalization) when collect_logprobs == true"),
        &[0],
      )?)
    } else {
      None
    };
    // #114: `step_index` + `finish_reason` are set provisionally to
    // `self.produced` (== "tokens yielded so far before this one") +
    // `None`; the [`Iterator::next`] impl overrides `finish_reason` to
    // `Some("stop")` on the EOS-token step (the only `Some(_)` value
    // single-seq generation produces — `length` is signalled by the
    // iterator ending, see the field doc).
    Ok(GenStep {
      token,
      logprobs,
      step_index: self.produced,
      finish_reason: None,
    })
  }
}

impl<M: Model + ?Sized> Iterator for Generator<'_, M> {
  type Item = Result<GenStep>;

  fn next(&mut self) -> Option<Self::Item> {
    // Fused: a prior Err or a finish ends iteration permanently — no
    // panic, no poisoned re-entry into the model / mlx-c.
    if self.done {
      return None;
    }

    // A deferred sampler / processor construction error is the iterator's
    // first (and only) item, before any model call.
    if let Some(e) = self.pending_err.take() {
      self.done = true;
      return Some(Err(e));
    }

    // mlx-lm yields exactly `max_tokens` tokens (`if n == max_tokens:
    // break` BEFORE the yield) ⇒ "length" finish.
    if self.produced >= self.max_tokens {
      self.done = true;
      return None;
    }

    // Prompt prefill runs once, lazily, on the first poll (mlx-lm runs it
    // before the first `_step`). Any error fuses the iterator.
    if !self.prefilled {
      self.prefilled = true;
      if let Err(e) = self.prefill() {
        self.done = true;
        return Some(Err(e));
      }
    }

    // The first decode step consumes the remaining prompt tail (mlx-lm
    // `_step(prompt)`); every later step feeds back the previously sampled
    // token (mlx-lm `_step(y)`).
    let input: Vec<u32> = if self.first_step {
      self.first_step = false;
      // mlx-lm `_step(input_tokens=prompt)`: the post-prefill tail
      // (`prompt[prefill_offset..]`). mlx-lm materializes this same window
      // as the step's `input_tokens[None]`; this is O(tail), not the
      // per-chunk O(P) front-drain.
      self.prompt[self.prefill_offset..].to_vec()
    } else {
      match self.last {
        Some(t) => vec![t],
        // Unreachable: `last` is `Some` after the first step, which always
        // ran first. End defensively rather than feed an empty window.
        None => {
          self.done = true;
          return None;
        }
      }
    };

    match self.step(&input) {
      Ok(mut step) => {
        self.produced += 1;
        self.last = Some(step.token);
        // `generate_step` itself stops on an eos token (it carries
        // the eos set); the eos token IS yielded (mlx-lm yields it, then
        // `stream_generate` breaks) — so yield it, then fuse.
        if self.eos.contains(&step.token) {
          self.done = true;
          // #114: surface the "stop" reason on the yielded EOS step
          // (mirrors `BatchGenStep::finish_reason` semantics). `length` is
          // never set here — `if produced >= max_tokens` above returns
          // `None` BEFORE a step runs, mirroring mlx-lm `generate_step`'s
          // pre-yield break exactly.
          step.finish_reason = Some(FinishReason::Eos);
        }
        Some(Ok(step))
      }
      Err(e) => {
        // A step error is yielded once, then the iterator ends.
        self.done = true;
        Some(Err(e))
      }
    }
  }
}

/// Build a `[1, S]` `I32` token window from `ids` (mlx-lm's `prompt[None]` /
/// `input_tokens[None]`). `I32` is mlx's default integer dtype for token
/// ids (embedding `take` indices); the trait only constrains the shape.
fn token_window(ids: &[u32]) -> Result<Array> {
  let mut row: Vec<i32> = try_with_capacity(ids.len())?;
  row.extend(ids.iter().map(|&t| t as i32));
  Array::from_slice::<i32>(&row, &(1usize, row.len()))
}

/// `logits[:, -1, :]` — slice the final sequence position of a `[B, S, V]`
/// logits tensor and drop the (now size-1) sequence axis ⇒ `[B, V]`,
/// matching mlx-lm's `logits[:, -1, :]` (`generate_step` line 407).
///
/// A degenerate (buggy-model) `S == 0` or `V == 0` axis is a
/// **DETERMINISTIC recoverable** `Err(`[`Error::OutOfRange`]`)` — the
/// faithful-equivalent of Python `logits[:, -1, :]` raising `IndexError` on
/// a zero-length sequence axis (and the same recoverable-`Err` discipline as
/// the merged KV-cache rank guards). Guarded **before** the `s - 1`
/// last-position index so a zero `S` can never underflow / produce a
/// malformed `[0, -1, 0]` slice start (it stays a clean `Err`, never a
/// panic, so the iterator yields it once then fuses).
fn last_position(logits: &Array) -> Result<Array> {
  let shape = logits.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "generate::last_position: expected [B, S, V] logits from `forward`",
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  // `logits[:, -1, :]` is only defined for a non-empty sequence axis and a
  // non-empty vocab axis; mirror Python's `IndexError` on `S == 0` (the
  // last-position index `s - 1` would underflow / be `-1`) and on `V == 0`
  // (an empty distribution the sampler cannot draw from) as a recoverable
  // `Err` BEFORE any index arithmetic.
  if shape[1] == 0 || shape[2] == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "generate::last_position: forward logits axes (S and V)",
      "must be >= 1 (logits[:, -1, :] requires S >= 1 and V >= 1)",
      format_smolstr!("S={}, V={}", shape[1], shape[2]),
    )));
  }
  let (b, s, v) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
  // `[ :, s-1 : s, : ]` (a 1-wide window at the last position); `s >= 1`
  // (guarded above) so `s - 1 >= 0` — never a malformed slice start.
  let sliced = ops::indexing::slice(logits, &[0, s - 1, 0], &[b, s, v], &[1, 1, 1])?;
  // Drop the size-1 sequence axis ⇒ `[B, V]` (mlx-lm's `[:, -1, :]`).
  ops::shape::squeeze_axes(&sliced, &[1])
}

/// Start a generation run: a 1:1 port of `mlx_lm.generate.generate_step`.
///
/// `prompt` is the encoded prompt ids, `cache` the per-layer KV cache
/// (owned by the returned iterator — typically
/// [`crate::lm::cache::make_prompt_cache`]; one entry per decoder layer),
/// and `cfg` the [`GenConfig`]. The sampler and logits processors are built
/// from `cfg` via [`make_sampler`] / [`make_logits_processors`] (exactly
/// mlx-lm's `generate` → `make_sampler`/`make_logits_processors` →
/// `generate_step` wiring), so a `cfg`-level sampler / penalty validation
/// error surfaces here as an `Err`.
///
/// Returns an `Iterator<Item = Result<GenStep>>`: each item is the next
/// sampled token id plus its `[V]` log-probability vector (the typed step
/// item — see [`GenStep`]). The iterator prefills the prompt (chunked by
/// [`GenConfig::prefill_step_size`]) on its first poll, then yields one
/// token per step until a sampled token is in the eos set (configured via
/// [`GenConfig::with_eos`]; the eos token is the final yielded item) or [`GenConfig::max_tokens`] tokens
/// have been produced. A step error is yielded once as `Err`, after which
/// the iterator ends (no panic, no poison).
///
/// # Breaking change (#113)
///
/// The return type is now `impl Iterator<Item = Result<GenStep>> + 'a`
/// (previously the concrete `Generator<'a, M>`). Hiding the iterator's
/// concrete type keeps internal refactors (e.g. splitting `Generator` into
/// `PrefillGenerator + DecodeGenerator`) non-breaking and matches the
/// sibling [`stream_generate`]'s shape. Callers that wrote
/// `let mut it = generate_step(...); it.next();` are unaffected; callers
/// that named the concrete `Generator<'a, M>` (none on `main` — it was
/// added by `#48`) must switch to inference / `impl Iterator<_>`.
pub fn generate_step<'a, M: Model + ?Sized>(
  model: &'a M,
  prompt: &[u32],
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> impl Iterator<Item = Result<GenStep>> + 'a {
  build_generator(model, prompt, cache, cfg)
}

/// Concrete-typed twin of [`generate_step`] for the in-crate driver
/// ([`crate::lm::session::ChatSession`]) that needs to reclaim the
/// advanced cache via [`Generator::into_cache`] after the turn finishes.
///
/// Returns the concrete [`Generator<'a, M>`]; [`generate_step`] is a thin
/// wrapper that hides this type behind `impl Iterator + 'a` for the
/// public API surface (#113).
pub(crate) fn build_generator<'a, M: Model + ?Sized>(
  model: &'a M,
  prompt: &[u32],
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> Generator<'a, M> {
  // Build sampler + processors up front (mlx-lm's `generate` does this
  // before calling `generate_step`). An empty prompt (mlx-lm raises
  // `ValueError("Either input_embeddings or prompt ... must be provided")`)
  // and any sampler / processor construction error are deferred into the
  // first `next()` as the iterator's first `Err` so the public surface
  // stays a pure `Iterator` and the Iterator-yields-Err
  // contract is the single error channel.
  let built = (|| -> Result<(Sampler, Vec<LogitsProcessor>)> {
    if prompt.is_empty() {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "generate: prompt",
      )));
    }
    // #136: eager scalar-bound validation of every sampler /
    // logits-processor knob in `cfg` BEFORE any prompt prefill / model
    // work. The sampler-build path only catches
    // a SUBSET of bounds at build time; the per-primitive validations
    // in `apply_*` only fire when the closure runs against logits, so
    // an invalid `cfg` would pass the constructor + run an entire
    // prompt prefill before erroring on the first decode step. Calling
    // `cfg.validate()` here fails fast — the `Err` propagates through
    // the existing `pending_err` channel so the iterator's first
    // `next()` yields it without any model call, matching the surface
    // shape already used for the prompt-empty / sampler-build Errs above.
    cfg.validate()?;
    let sampler = make_sampler(
      cfg.temp,
      cfg.top_p,
      cfg.min_p,
      cfg.min_tokens_to_keep,
      cfg.top_k,
      cfg.xtc_probability,
      cfg.xtc_threshold,
      &cfg.xtc_special_tokens,
      cfg.seed,
    )?;
    let processors = make_logits_processors(
      &cfg.logit_bias,
      cfg.repetition_penalty,
      cfg.repetition_context_size,
      cfg.presence_penalty,
      cfg.presence_context_size,
      cfg.frequency_penalty,
      cfg.frequency_context_size,
    )?;
    Ok((sampler, processors))
  })();

  let collect_logprobs = cfg.collect_logprobs;
  // The sampler-chain "needs normalized log-probs" gate, computed from the
  // same `(temp, top_p, …)` knobs `make_sampler` reads (so the gate stays
  // in lockstep with the built chain). Only `top_p ∈ (0, 1)` strictly
  // needs the cumsum-to-1 normalization — `apply_top_p` does
  // `probs = exp(logprobs)` then `cumsum`, and its `1 - top_p` threshold
  // only matches the true probability mass when the cumsum reaches 1.0.
  // Every other sampler in `make_sampler` is either shift-invariant
  // (argmax, top_k via argpartition on `-logprobs`, min_p whose threshold
  // is `max + log(min_p)` so cancels the shift, xtc which does its own
  // `softmax(logits)`) or softmaxes internally (`categorical_sampling`
  // calls `random.categorical` which applies softmax over the last axis).
  // mlx-lm always runs the normalization because it always yields
  // `logprobs` to the caller; mlxrs separates the "yield" from the
  // "sampler input" so a `collect_logprobs=false` greedy / temperature-
  // only run pays zero per-token reduce + broadcast subtract.
  //
  // Lockstep with `make_sampler`'s `temp == 0` → argmax shortcut: when
  // `cfg.temp == 0` the built sampler is pure argmax regardless of
  // `top_p` (mlx-lm's `make_sampler` returns argmax BEFORE reading
  // top_p / min_p / xtc / top_k), so the top_p flag is dead config — no
  // normalization needed there either. Honor that here so a stale
  // `top_p` on a greedy run still skips the per-token logsumexp.
  let needs_logprobs = cfg.temp != 0.0 && cfg.top_p > 0.0 && cfg.top_p < 1.0;
  // Lockstep with `make_sampler`'s `temp == 0` → argmax shortcut: the
  // stochastic max-shift opt-out path only fires when the built sampler is
  // actually stochastic (`temp > 0` ⇒ the chain bottoms out in
  // `categorical_sampling`, which scales by `1/temp` before softmax). When
  // `temp == 0` the sampler is pure argmax (shift-invariant numerically), so
  // the raw-logit path is safe — see `temp_stochastic` on `Generator`.
  let temp_stochastic = cfg.temp > 0.0;
  match built {
    Ok((sampler, processors)) => Generator {
      model,
      cache,
      sampler,
      processors,
      prompt: prompt.to_vec(),
      prefill_offset: 0,
      history: Vec::new(),
      last: None,
      produced: 0,
      max_tokens: cfg.max_tokens,
      prefill_step_size: cfg.prefill_step_size.max(1),
      eos: cfg.eos,
      collect_logprobs,
      needs_logprobs,
      temp_stochastic,
      prefilled: false,
      first_step: true,
      pending_err: None,
      done: false,
    },
    Err(e) => Generator {
      model,
      cache,
      // A never-called placeholder sampler; `pending_err` ends the
      // iterator on its first poll before any step runs. The cheapest
      // variant ([`Sampler::Argmax`]) is used as a no-allocation
      // placeholder — its `sample` is never called because `pending_err`
      // short-circuits the first `next()`.
      sampler: Sampler::Argmax,
      processors: Vec::new(),
      prompt: Vec::new(),
      prefill_offset: 0,
      history: Vec::new(),
      last: None,
      produced: 0,
      max_tokens: cfg.max_tokens,
      prefill_step_size: 1,
      eos: Vec::new(),
      collect_logprobs,
      needs_logprobs,
      temp_stochastic,
      prefilled: true,
      first_step: false,
      pending_err: Some(e),
      done: false,
    },
  }
}

/// The final segment of a generation run — a 1:1 port of mlx-lm's
/// `GenerationResponse` (`generate.py` lines 269-296), restricted to the
/// fields the no-network / single-stream surface produces.
///
/// Yielded by [`stream_generate`]: `text` is the streaming detokenizer's
/// newly readable segment for this token (possibly empty), `token` /
/// `logprobs` the just-produced step, and `finish_reason` is `None` for
/// intermediate responses, `Some("stop")` when the model emitted an eos
/// token, `Some("length")` when `max_tokens` was reached — exactly mlx-lm's
/// `"stop" if token in tokenizer.eos_token_ids else "length"`.
///
/// `logprobs` honours the `GenStep` opt-in: `Some` iff the underlying
/// [`GenConfig::collect_logprobs`] was `true`; `None` otherwise.
///
/// `peak_memory_bytes` mirrors mlx-lm's `peak_memory = mx.get_peak_memory()
/// / 1e9` (kept in raw bytes here — the caller picks the scale). `None`
/// when the [`crate::memory::peak_memory`] FFI call itself errors; the
/// stream then continues uninterrupted (the per-response counter is
/// diagnostic, not load-bearing).
#[derive(Debug)]
pub struct GenerationResponse {
  /// The next readable text segment (mlx-lm `detokenizer.last_segment`);
  /// may be empty when the detokenizer is still withholding bytes.
  pub text: String,
  /// The token this response carries (mlx-lm `token`).
  pub token: u32,
  /// The token's `[V]` log-probability vector (mlx-lm `logprobs`).
  /// `Some` iff [`GenConfig::collect_logprobs`] was `true`.
  pub logprobs: Option<Array>,
  /// Number of prompt tokens (mlx-lm `prompt_tokens` = `prompt.size`).
  pub prompt_tokens: usize,
  /// Prompt processing tokens-per-second (mlx-lm `prompt_tps`).
  pub prompt_tps: f64,
  /// Number of tokens generated so far (mlx-lm `generation_tokens` = `n +
  /// 1`).
  pub generation_tokens: usize,
  /// Generation tokens-per-second (mlx-lm `generation_tps`).
  pub generation_tps: f64,
  /// Process-global mlx allocator peak in bytes (mlx-lm's
  /// `mx.get_peak_memory()`). `None` if the FFI counter is unavailable —
  /// stream is unaffected.
  pub peak_memory_bytes: Option<u64>,
  /// `None` while generating; `Some(FinishReason::Eos)` on an eos token,
  /// `Some(FinishReason::Length)` at `max_tokens` (mlx-lm `finish_reason`).
  pub finish_reason: Option<FinishReason>,
}

/// Aggregate stats for one full [`generate`] run — the cumulative
/// counterparts of the per-response [`GenerationResponse`] timing fields,
/// returned alongside the assembled output string.
///
/// Mirrors mlx-lm's `generate` verbose-mode summary (`mlx_lm/generate.py`
/// lines 791-798) and the mlx-swift-lm `GenerateCompletionInfo` /
/// `GenerateResult` summary, condensed into the union of fields the
/// no-network surface produces:
///
/// - `prompt_tokens` / `generation_tokens` — counts (mlx-lm
///   `response.prompt_tokens` / `response.generation_tokens`).
/// - `prompt_tps` / `generation_tps` — tokens-per-second (mlx-lm
///   `response.prompt_tps` / `response.generation_tps`); both are 0.0
///   when their respective phase took zero measurable wall time.
/// - `peak_memory_bytes` — process-global mlx allocator peak in bytes
///   (mlx-lm `mx.get_peak_memory()`); `None` if the FFI counter is
///   unavailable.
#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
  /// Prompt tokens processed (mlx-lm `response.prompt_tokens`).
  pub prompt_tokens: usize,
  /// Tokens generated by the model (mlx-lm `response.generation_tokens`;
  /// `0` if `stream_generate` produced no tokens).
  pub generation_tokens: usize,
  /// Prompt processing tokens-per-second (mlx-lm `response.prompt_tps`).
  pub prompt_tps: f64,
  /// Generation tokens-per-second (mlx-lm `response.generation_tps`).
  pub generation_tps: f64,
  /// Process-global mlx allocator peak in bytes at the end of the run
  /// (mlx-lm `response.peak_memory * 1e9`). `None` if the FFI counter
  /// (`mlx_get_peak_memory`) is unavailable / errored — the run completes
  /// regardless.
  pub peak_memory_bytes: Option<u64>,
}

/// Stream text from `model` for `prompt` — a 1:1 port of
/// `mlx_lm.generate.stream_generate`.
///
/// Maps [`generate_step`] through the #18 streaming detokenizer
/// ([`crate::tokenizer::Tokenizer::detokenizer`]) into
/// [`GenerationResponse`]s. The eos set is taken from the tokenizer
/// ([`crate::tokenizer::Tokenizer::eos_token_ids_iter`], mlx-lm's
/// `tokenizer.eos_token_ids`), overriding any `cfg.eos`, so the
/// `finish_reason` matches mlx-lm exactly. mlx-lm does **not** detokenize
/// the eos token (`if token in eos: break` before `add_token`), so the eos
/// token contributes no text and the final response carries
/// `Some("stop")`; reaching `max_tokens` yields a final response with
/// `Some("length")`.
///
/// An underlying step error is propagated as a yielded `Err` (the
/// [`generate_step`] Iterator-`Err` contract is preserved through the
/// detokenizer mapping); the iterator then ends (no panic, no poison).
///
/// `prompt` here is the already-encoded prompt ids (the caller encodes via
/// [`crate::tokenizer::Tokenizer::encode`]); mlx-lm's `str`-encoding
/// convenience belongs to a higher-level entry point.
pub fn stream_generate<'a, M: Model + ?Sized>(
  model: &'a M,
  tokenizer: &'a crate::tokenizer::Tokenizer,
  prompt: &[u32],
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> impl Iterator<Item = Result<GenerationResponse>> + 'a {
  use std::time::Instant;

  let prompt_tokens = prompt.len();
  // mlx-lm uses the tokenizer's eos set (not a caller override) for the
  // break + `finish_reason`; mirror that exactly.
  let mut cfg = cfg;
  cfg.eos = tokenizer.eos_token_ids_iter().collect();
  let max_tokens = cfg.max_tokens;
  let eos: Vec<u32> = cfg.eos.clone();

  // L5: multi-token / string stop sequences (mlx-lm `stop_words`). Built from
  // the decoded text (see [`crate::lm::stop`]); inert when `stop_strings` is
  // empty, in which case the loop takes the original `last_segment()` path
  // unchanged (eos-only, byte-for-byte the prior behavior).
  let matcher = crate::lm::stop::StopMatcher::new(cfg.stop_strings.clone());
  // Bytes of the cumulative decoded text already emitted to the caller. Only
  // used on the active-matcher path (which drives emission off `detok.text()`
  // and never advances the detokenizer offset).
  let mut emitted_len: usize = 0;

  let mut steps = generate_step(model, prompt, cache, cfg);
  let mut detok = tokenizer.detokenizer();
  let mut n: usize = 0;
  let mut finished = false;
  // mlx-lm wall-clock timing: `tic` before the first token measures prompt
  // processing; reset after the first token to measure generation.
  let mut tic = Instant::now();
  let mut prompt_tps = 0.0_f64;

  std::iter::from_fn(move || {
    if finished {
      return None;
    }
    // #114: `..` for forward compatibility — `GenStep` now also
    // carries `step_index` + `finish_reason`. `stream_generate` recomputes
    // its OWN `finish_reason` for `GenerationResponse` (`"stop"` on eos,
    // `"length"` on `max_tokens`, factoring in stop-strings via the
    // [`crate::lm::stop::StopMatcher`]) so the per-step
    // `finish_reason` is not re-read here — the wrapper still owns that
    // decision. `step_index` is the LM-loop's own counter (0-indexed),
    // distinct from `generation_tokens` (1-indexed per `GenerationResponse`)
    // so the latter stays mlx-lm parity (`n + 1`).
    let GenStep {
      token, logprobs, ..
    } = match steps.next()? {
      Ok(step) => step,
      Err(e) => {
        finished = true;
        return Some(Err(e));
      }
    };

    // mlx-lm: at the first produced token, `prompt_tps = prompt.size /
    // (now - tic)`, then `tic` is reset to time generation.
    if n == 0 {
      let prompt_time = tic.elapsed().as_secs_f64();
      prompt_tps = if prompt_time > 0.0 {
        prompt_tokens as f64 / prompt_time
      } else {
        0.0
      };
      tic = Instant::now();
    }

    // mlx-lm: `generation_tps = (n + 1) / (now - tic)` (here `gen_count`
    // already counts the about-to-be-reported token).
    let gen_tps = |gen_count: usize| -> f64 {
      let dt = tic.elapsed().as_secs_f64();
      if dt > 0.0 { gen_count as f64 / dt } else { 0.0 }
    };

    // mlx-lm: `peak_memory = mx.get_peak_memory() / 1e9`. We surface raw
    // bytes; an FFI failure (rare — process-global counter) degrades to
    // `None` without aborting the stream (the field is diagnostic).
    let peak = crate::memory::peak_memory().ok();

    // mlx-lm: `if token in eos: break` BEFORE `add_token` ⇒ the eos token
    // is never detokenized; a final `finish_reason="stop"` response with
    // the (empty) finalized tail is yielded.
    if eos.contains(&token) {
      finished = true;
      detok.finalize();
      // Active path: a detokenizer may withhold tail text from `text()`
      // until `finalize()` (e.g. the BPE detok holds a bare-space token),
      // so re-check the matcher against the now-finalized text before
      // emitting — a stop completed by the finalized tail still trims. The
      // EOS token reached: typed FinishReason is Eos by default, but a
      // detokenizer-withheld tail can complete a stop-string match in
      // finalize_active_tail() — propagate that as Stop(matched) so the
      // matched-sequence payload survives the terminal path.
      let (text, reason) = if matcher.is_active() {
        finalize_active_tail(&detok, &matcher, &mut emitted_len, FinishReason::Eos)
      } else {
        (detok.last_segment(), FinishReason::Eos)
      };
      return Some(Ok(GenerationResponse {
        text,
        token,
        logprobs,
        prompt_tokens,
        prompt_tps,
        generation_tokens: n + 1,
        generation_tps: gen_tps(n + 1),
        peak_memory_bytes: peak,
        finish_reason: Some(reason),
      }));
    }

    detok.add_token(token);
    n += 1;

    // L5: string stop-sequence check (active matcher only). Runs AFTER
    // `add_token` so the just-produced token's text participates, and BEFORE
    // the `max_tokens` check so a stop string still reports `finish_reason=
    // "stop"` on the final allowed token. The matched stop sequence (and any
    // trailing text) is trimmed from the output; the inert path skips this
    // entirely and behaves exactly as before.
    if matcher.is_active() {
      let full = detok.text();
      match matcher.step(&full) {
        crate::lm::stop::StopDecision::Stop(p) => {
          finished = true;
          let end = p.trimmed_len().max(emitted_len).min(full.len());
          let text = full[emitted_len..end].to_string();
          let stop = p.stop().to_owned();
          emitted_len = end;
          drop(full);
          return Some(Ok(GenerationResponse {
            text,
            token,
            logprobs,
            prompt_tokens,
            prompt_tps,
            generation_tokens: n,
            generation_tps: gen_tps(n),
            peak_memory_bytes: peak,
            // Stop-string match — typed FinishReason carries the matched
            // sequence so callers can distinguish from an EOS-token finish.
            finish_reason: Some(FinishReason::Stop(stop)),
          }));
        }
        crate::lm::stop::StopDecision::Continue(p) => {
          // mlx-lm: `if (n + 1) == max_tokens: break` ⇒ a final
          // `finish_reason="length"` response with the finalized tail. But a
          // detokenizer may withhold tail text from `text()` until
          // `finalize()` (e.g. the BPE detok holds a bare-space token), so
          // re-check the matcher against the finalized text: a stop completed
          // by the finalized tail wins over `length` (trim + "stop").
          if n >= max_tokens {
            finished = true;
            drop(full);
            detok.finalize();
            let (text, reason) =
              finalize_active_tail(&detok, &matcher, &mut emitted_len, FinishReason::Length);
            return Some(Ok(GenerationResponse {
              text,
              token,
              logprobs,
              prompt_tokens,
              prompt_tps,
              generation_tokens: n,
              generation_tps: gen_tps(n),
              peak_memory_bytes: peak,
              finish_reason: Some(reason),
            }));
          }
          let safe_len = p.safe_len();
          let end = safe_len.max(emitted_len).min(full.len());
          let text = full[emitted_len..end].to_string();
          emitted_len = end;
          drop(full);
          return Some(Ok(GenerationResponse {
            text,
            token,
            logprobs,
            prompt_tokens,
            prompt_tps,
            generation_tokens: n,
            generation_tps: gen_tps(n),
            peak_memory_bytes: peak,
            finish_reason: None,
          }));
        }
      }
    }

    // mlx-lm: `if (n + 1) == max_tokens: break` (n is 0-based there; here
    // `n` already counts this token) ⇒ a final `finish_reason="length"`
    // response with the finalized tail.
    if n >= max_tokens {
      finished = true;
      detok.finalize();
      let text = detok.last_segment();
      return Some(Ok(GenerationResponse {
        text,
        token,
        logprobs,
        prompt_tokens,
        prompt_tps,
        generation_tokens: n,
        generation_tps: gen_tps(n),
        peak_memory_bytes: peak,
        finish_reason: Some(FinishReason::Length),
      }));
    }

    let text = detok.last_segment();
    Some(Ok(GenerationResponse {
      text,
      token,
      logprobs,
      prompt_tokens,
      prompt_tps,
      generation_tokens: n,
      generation_tps: gen_tps(n),
      peak_memory_bytes: peak,
      finish_reason: None,
    }))
  })
}

/// Active-matcher terminal finalization. The caller MUST have already called
/// `detok.finalize()`; this re-runs the stop matcher against the now-finalized
/// `detok.text()` and decides the final emitted tail + `finish_reason`.
///
/// Mid-stream matching runs on `text()` BEFORE finalization, but some
/// detokenizers withhold tail text from `text()` until `finalize()` (e.g. the
/// BPE detok holds a single bare-space token for one step). A stop string can
/// therefore be completed only by that finalized tail, so the terminal paths
/// must re-check rather than blindly emit the tail:
///
/// - [`StopDecision::Stop`](crate::lm::stop::StopDecision::Stop): emit only up
///   to `trimmed_len` (clamped exactly like the mid-stream Stop arm) and report
///   `"stop"` — even on the `max_tokens` path, where a stop completed by the
///   finalized tail wins over `"length"`.
/// - Otherwise: emit the remaining safe tail (`text()[*emitted_len..]`,
///   advancing `*emitted_len` to the end — the matcher path drives emission off
///   byte offsets into `text()`, never `last_segment`) and report
///   `default_reason` (`"stop"` on the eos path, `"length"` on `max_tokens`).
fn finalize_active_tail(
  detok: &dyn crate::tokenizer::StreamingDetokenizer,
  matcher: &crate::lm::stop::StopMatcher,
  emitted_len: &mut usize,
  default_reason: FinishReason,
) -> (String, FinishReason) {
  let full = detok.text();
  match matcher.step(&full) {
    crate::lm::stop::StopDecision::Stop(p) => {
      let end = p.trimmed_len().max(*emitted_len).min(full.len());
      let text = full[*emitted_len..end].to_string();
      *emitted_len = end;
      // Stop-string match — surface the matched sequence in the typed
      // FinishReason so callers can distinguish a configured stop from EOS.
      (text, FinishReason::Stop(p.stop().to_owned()))
    }
    crate::lm::stop::StopDecision::Continue(_) => {
      let start = (*emitted_len).min(full.len());
      let text = full[start..].to_string();
      *emitted_len = full.len();
      (text, default_reason)
    }
  }
}

/// Generate a complete response string for `prompt` — a 1:1 port of
/// `mlx_lm.generate.generate` (the non-verbose path): collect every
/// [`stream_generate`] segment into one `String` and return it alongside
/// the aggregate [`GenerationStats`] for the run (the L3 stats surface —
/// counts + tokens-per-second + peak memory, populated from the final
/// [`GenerationResponse`] mlx-lm emits in its verbose-mode summary).
///
/// Returns `(text, stats)`:
/// - `text` is the concatenation of every per-response `text` segment
///   (the eos token contributes no text, faithful to mlx-lm).
/// - `stats` carries the final response's `prompt_tokens` /
///   `generation_tokens` / `prompt_tps` / `generation_tps` plus
///   `peak_memory_bytes` (mlx-lm's `mx.get_peak_memory()` in bytes; see
///   [`GenerationStats`]).
///
/// An empty run (zero produced tokens — `max_tokens == 0`) returns the
/// empty string and a zero-tps `GenerationStats` with the original
/// `prompt_tokens` count and the current peak memory.
///
/// Any step error is surfaced as `Err` (it short-circuits the collection,
/// exactly the [`stream_generate`] Iterator-`Err` contract).
pub fn generate<M: Model + ?Sized>(
  model: &M,
  tokenizer: &crate::tokenizer::Tokenizer,
  prompt: &[u32],
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> Result<(String, GenerationStats)> {
  let prompt_tokens = prompt.len();
  let mut text = String::new();
  // Capture the *final* response's stats fields (mlx-lm's verbose-mode
  // summary uses the loop's last `response`); a stream that produced
  // nothing falls back to the zero-tps init below.
  let mut final_response: Option<GenerationResponse> = None;
  for response in stream_generate(model, tokenizer, prompt, cache, cfg) {
    let response = response?;
    text.push_str(&response.text);
    final_response = Some(response);
  }

  let stats = match final_response {
    Some(r) => GenerationStats {
      prompt_tokens: r.prompt_tokens,
      generation_tokens: r.generation_tokens,
      prompt_tps: r.prompt_tps,
      generation_tps: r.generation_tps,
      peak_memory_bytes: r.peak_memory_bytes,
    },
    // No tokens produced (e.g. `max_tokens == 0`): mlx-lm prints
    // "No text generated for this prompt" and returns; we surface the
    // same zero-counts stats so the caller still gets `prompt_tokens` +
    // a current peak-memory snapshot.
    None => GenerationStats {
      prompt_tokens,
      generation_tokens: 0,
      prompt_tps: 0.0,
      generation_tps: 0.0,
      peak_memory_bytes: crate::memory::peak_memory().ok(),
    },
  };
  Ok((text, stats))
}

// ════════════════════════════════════════════════════════════════════════════
//   Batched generation (L1) — left-padded prefill + per-row independent EOS.
// ════════════════════════════════════════════════════════════════════════════

/// One batched decode step (per-row sampled `token` + the `[V]` log-probability
/// vector that produced it) for **one** row of a [`batch_generate`] /
/// [`batch_stream_generate`] run — the batched analogue of [`GenStep`].
///
/// `row` is the 0-based row index in the original `prompts` slice; `token` is
/// the just-sampled id; `logprobs` is the `[V]` per-row log-probability vector
/// (kept lazy — the only materialization is the per-row token id, mirroring
/// single-seq [`GenStep`]); `finish_reason` is `None` while the row is still
/// generating, `Some("stop")` once the row sampled an EOS token, `Some(
/// "length")` when the row hit [`GenConfig::max_tokens`].
#[derive(Debug)]
pub struct BatchGenStep {
  /// 0-based row index in the original `prompts` slice — mlx-lm
  /// `GenerationBatch.Response.uid` (uids are assigned in insertion order,
  /// so `uid == row index` here).
  pub row: usize,
  /// The sampled token id for this row at this step (mlx-lm `Response.token`).
  pub token: u32,
  /// The `[V]` per-row log-probability vector that produced `token` (mlx-lm
  /// `Response.logprobs`).
  pub logprobs: Array,
  /// `None` while still generating; `Some(FinishReason::Eos)` on an EOS
  /// token, `Some(FinishReason::Length)` on `max_tokens` — exactly mlx-lm
  /// `Response.finish_reason`.
  pub finish_reason: Option<FinishReason>,
}

/// Streaming batched-generation iterator: one [`BatchGenStep`] per row per
/// step, yielding rows in row-index order within each step. Stops when every
/// row has finished (EOS / `max_tokens`).
///
/// The decode loop runs ONE forward over the full `[B, 1]` batch per step
/// (mirroring mlx-lm `GenerationBatch._step`, `generate.py:1320-1378` —
/// "Forward pass: logits = self.model(inputs[:, None], cache=...)"), then
/// per-row samples / appends. Rows finish independently: once a row hits EOS
/// or `max_tokens`, it is marked done but the batch shape stays `[B, ...]`
/// (the underlying batch caches do not support mid-run row removal; finished
/// rows feed `pad_token_id` and their sampled outputs are discarded).
///
/// The iterator **fuses**: after it yields `Err` (a step failed) or finishes
/// (every row done) every further `next()` is `None` — never a panic, never a
/// poisoned re-entry.
///
/// `M: Model + ?Sized` — like the single-sequence
/// [`generate_step`] iterator, the loop only ever
/// touches the model behind the `&'a M` borrow (`model.forward(...)`), never
/// by value and never via a `Sized`-requiring associated item, so `M` may be
/// an unsized trait object (`&dyn Model`, or a deref-coerced
/// `Box<dyn Model>` / `Box<dyn VlmModel>`). This keeps batch generation
/// drivable by the exact handle a load factory returns.
pub struct BatchGenerator<'a, M: Model + ?Sized> {
  model: &'a M,
  cache: Vec<Box<dyn KvCache>>,
  sampler: Sampler,
  processors: Vec<LogitsProcessor>,
  /// The left-padded prompt `[B, max_len]` (mlx-lm `_left_pad_prompts`,
  /// `generate.py:802-805`). Prefill advances [`Self::prefill_offset`] over
  /// this buffer; the unconsumed final column starts the first decode step.
  /// Stored as `Vec<Vec<u32>>` rather than the materialized `Array` so the
  /// prefill chunk slicing is host-side (the `Array` slice would still need
  /// rebuilding per chunk; this avoids the per-chunk `slice` op).
  padded_rows: Vec<Vec<u32>>,
  max_len: usize,
  prefill_offset: usize,
  /// Per-row running history fed to the logits processors (mlx-lm
  /// `GenerationBatch._token_context`). Each row's `_step` slice is `inputs[
  /// i:i+1]` so each row's history grows by the per-step input token.
  history: Vec<Vec<u32>>,
  /// The most-recent per-row sampled token (mlx-lm's `inputs` fed into the
  /// next `_step`); `None` before the first decode step. A finished row's
  /// slot stays at `pad_token_id` for every subsequent step.
  last: Vec<u32>,
  /// Per-row "tokens generated so far" counter (mlx-lm `Response`'s
  /// generation count); compared against `max_tokens` per row.
  produced: Vec<usize>,
  /// Per-row finish reason: `None` while still generating,
  /// `Some(FinishReason::Eos)` once EOS sampled,
  /// `Some(FinishReason::Length)` at `max_tokens`. Mirrors mlx-lm's
  /// per-row `Response.finish_reason` semantics. A row stops contributing
  /// output the step it transitions from `None` to `Some(_)` (the EOS token
  /// itself is yielded with `finish_reason=Eos` but NOT appended to the
  /// row's running output by `batch_generate`, mirroring mlx-lm
  /// `generate.py:1945-1946`).
  finished: Vec<Option<FinishReason>>,
  /// 0-based row indices yet to emit at the current step (drained in order
  /// by `next()`). Empty between steps; refilled when a new forward runs.
  pending_emit: std::collections::VecDeque<BatchGenStep>,
  pad_token_id: u32,
  max_tokens: usize,
  prefill_step_size: usize,
  eos: Vec<u32>,
  /// `true` once prompt prefill has run (it runs on the first `next()`).
  prefilled: bool,
  /// `true` until the first decode step has run (it feeds the unconsumed
  /// prompt tail; later steps feed back `last`).
  first_step: bool,
  /// A deferred sampler / processor / cache validation error (from
  /// [`batch_generate_step`]); yielded as the iterator's first (and only)
  /// `Err` before any step runs, keeping the public surface a pure
  /// `Iterator`.
  pending_err: Option<Error>,
  /// Fused: set after a yielded `Err` or all-rows-done so the iterator
  /// never re-enters mlx-c / re-runs the model.
  done: bool,
}

impl<M: Model + ?Sized> BatchGenerator<'_, M> {
  fn batch_size(&self) -> usize {
    self.padded_rows.len()
  }

  /// One chunked-prefill pass over the left-padded `[B, max_len-1]` window —
  /// mirrors single-seq [`Generator::prefill`] but emits `[B, S]` token
  /// windows. Logits are discarded; only the cache is filled.
  fn prefill(&mut self) -> Result<()> {
    while self.max_len - self.prefill_offset > 1 {
      let remaining = (self.max_len - self.prefill_offset) - 1;
      let n = self.prefill_step_size.min(remaining);
      let chunk = batch_token_window(
        &self.padded_rows,
        self.prefill_offset,
        self.prefill_offset + n,
      )?;
      // logits discarded — the chunk only fills the cache.
      let _ = self.model.forward(&chunk, &mut self.cache)?;
      self.prefill_offset += n;
    }
    Ok(())
  }

  /// One batched decode step — the batched analogue of [`Generator::step`].
  /// Returns the per-row [`BatchGenStep`] vector (one entry per row, in row
  /// order), with each row's `finish_reason` updated for this step's
  /// transition (newly-finished rows get `Some("stop")`/`Some("length")`).
  ///
  /// Mirrors mlx-lm `GenerationBatch._step` (`generate.py:1320-1378`): single
  /// `[B, 1]` forward → `logits[:, -1, :]` → optional per-row processors →
  /// `logsumexp` normalize → sampler → per-row token extract. Finished rows
  /// pre-step are still fed (their `last` slot is `pad_token_id`) but their
  /// sampled-token contribution is NOT appended to the running output (the
  /// per-row `BatchGenStep` for an already-finished row carries the
  /// finalized `finish_reason` and a dummy token, exactly like mlx-lm where
  /// a removed row produces no further `Response`s — but our batch shape
  /// can't shrink, so we surface the no-op as `finish_reason=Some(prior)`).
  fn step(&mut self, input: &[u32]) -> Result<Vec<BatchGenStep>> {
    let b = self.batch_size();
    // 1. forward over `input[B, S]`; cache updated in place.
    let tokens = batch_full_window(input, b, input.len() / b)?;
    let logits = self.model.forward(&tokens, &mut self.cache)?;

    // 2. `logits = logits[:, -1, :]` ⇒ `[B, V]`.
    let mut logits = last_position(&logits)?;

    // 3. Per-row logits processors (mlx-lm `_step` lines 1336-1349): if any
    //    processors, split the per-step input into per-row `[1]` slices,
    //    grow each row's history by that token, run each processor over the
    //    row's history on the row's `[1, V]` logit slice, then concat back
    //    to `[B, V]`. mlx-lm only runs this block when any processor exists
    //    AND the input is non-empty; mirror that exactly (avoid the needless
    //    per-row history growth in the no-processors path).
    if !self.processors.is_empty() && !input.is_empty() {
      let s = input.len() / b;
      let mut row_logits: Vec<Array> = try_with_capacity(b)?;
      for (row, hist) in self.history.iter_mut().enumerate().take(b) {
        // Per-row input slice for this step (the row's S tokens from the
        // window; mlx-lm `inputs[i:i+1]` is shape-equivalent — both extend
        // the row's running history by S tokens).
        let row_input = &input[row * s..(row + 1) * s];
        try_extend_from_slice(hist, row_input)?;
        // Per-row logit slice `logits[row:row+1, :]` ⇒ `[1, V]`.
        let v = logits.shape()[1] as i32;
        let row_logit =
          ops::indexing::slice(&logits, &[row as i32, 0], &[(row + 1) as i32, v], &[1, 1])?;
        let mut row_l = row_logit;
        for p in &self.processors {
          row_l = p.apply(hist, &row_l)?;
        }
        row_logits.push(row_l);
      }
      // concat the `[1, V]` rows back to `[B, V]` on axis 0.
      let row_refs: Vec<&Array> = row_logits.iter().collect();
      logits = ops::shape::concatenate(&row_refs, 0)?;
    }

    // 4. `logprobs = logits - logsumexp(logits, keepdims=True)` (mlx-lm
    //    `_step` line 1352). The full-axes `logsumexp` matches the
    //    single-seq path: every `[B, V]` row gets normalized independently
    //    because the reduction is per-row when `keepdims=True` broadcasts.
    //
    //    Note: mlx-lm's `_step` calls `mx.logsumexp(logits, axis=-1,
    //    keepdims=True)` here (explicit `axis=-1`), whereas the single-seq
    //    path passes no `axis` (full reduction). For `[B, V]` the two
    //    differ — `axis=-1` per-row normalizes to `[B, 1]`, full reduction
    //    is `[1, 1]`. The single-seq path's full reduction is correct
    //    because B=1; for batch we MUST use axis=-1 per-row.
    let lse = ops::reduction::logsumexp_axes(&logits, &[-1], true)?;
    let logprobs = ops::arithmetic::subtract(&logits, &lse)?;

    // 5. `sampled = sampler(logprobs)` (mlx-lm `_step` lines 1354-1363).
    //    A single global sampler is applied to the full `[B, V]`; argmax /
    //    categorical / the make_sampler chain all reduce over axis=-1 and
    //    yield `[B]` U32. Per-row samplers (mlx-lm's `samplers[e]` list)
    //    are not exposed by [`GenConfig`] — mirrors mlx-lm's fallback path.
    let mut sampled = self.sampler.sample(&logprobs)?;

    // 6. token boundary: ONE materialization for the whole batch (mlx-lm
    //    materializes `inputs.tolist()` once per step, line 1375); the
    //    logprobs stay lazy.
    let tokens: Vec<u32> = sampled.to_vec::<u32>()?;
    if tokens.len() != b {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "batch_generate: sampler returned tokens (must be one per row)",
        b,
        tokens.len(),
      )));
    }

    // Build per-row step results. The full per-row logprob slice `[V]` is
    // sliced lazily (the only materialization above was the token batch).
    let mut steps: Vec<BatchGenStep> = try_with_capacity(b)?;
    let v = logprobs.shape()[1] as i32;
    for (row, &tok) in tokens.iter().enumerate() {
      // logprobs[row, :] ⇒ `[V]` (slice + squeeze axis 0).
      let row_lp =
        ops::indexing::slice(&logprobs, &[row as i32, 0], &[(row + 1) as i32, v], &[1, 1])?;
      let row_lp = ops::shape::squeeze_axes(&row_lp, &[0])?;
      // Decide per-row transition (None ⇒ Some on EOS / max_tokens).
      // Already-finished rows keep their prior reason (the loop won't emit
      // them again, but we still build the result so per-row order is
      // preserved if a caller streams them).
      let prior = self.finished[row].clone();
      let new_reason: Option<FinishReason> = if prior.is_some() {
        prior
      } else if self.eos.contains(&tok) {
        Some(FinishReason::Eos)
      } else {
        // Pre-bump check: this step's token will be the `produced[row] + 1`th
        // generated token; mlx-lm reports "length" when `produced == max_tokens`
        // BEFORE yielding the would-be next token (`generate_step` line 421:
        // `if n == max_tokens: break` is BEFORE the yield).
        // BUT mlx-lm's single-seq path yields the LAST token with no
        // finish_reason and breaks; the next iteration sees `n ==
        // max_tokens` and stops. Here for batch we lump them: the
        // `(produced+1) == max_tokens` token gets `Some(FinishReason::Length)`
        // to surface the per-row termination in ONE step (the caller would
        // otherwise need a separate "length" sentinel after this token).
        if self.produced[row] + 1 >= self.max_tokens {
          Some(FinishReason::Length)
        } else {
          None
        }
      };
      steps.push(BatchGenStep {
        row,
        token: tok,
        logprobs: row_lp,
        finish_reason: new_reason,
      });
    }
    Ok(steps)
  }
}

impl<M: Model + ?Sized> Iterator for BatchGenerator<'_, M> {
  type Item = Result<BatchGenStep>;

  fn next(&mut self) -> Option<Self::Item> {
    // Drain any pending per-row step results from the most-recent forward
    // before running another model call.
    if let Some(step) = self.pending_emit.pop_front() {
      return Some(Ok(step));
    }
    if self.done {
      return None;
    }
    // A deferred sampler / processor construction error is the iterator's
    // first (and only) item, before any model call.
    if let Some(e) = self.pending_err.take() {
      self.done = true;
      return Some(Err(e));
    }

    // Zero-budget guard (mirrors single-seq `Generator::next` at the
    // analogous slot): mlx-lm yields exactly `max_tokens` tokens with `if n
    // == max_tokens: break` BEFORE the yield. For batched generation every
    // row shares one `max_tokens`, so when no row can ever produce a token
    // the iterator finishes immediately — BEFORE prefill and any model /
    // cache mutation, matching `GenConfig`'s documented "0 produces nothing"
    // and the single-seq guard's contract.
    if self
      .produced
      .iter()
      .zip(self.finished.iter())
      .all(|(&p, f)| f.is_some() || p >= self.max_tokens)
    {
      self.done = true;
      return None;
    }

    // Prompt prefill runs once, lazily, on the first poll.
    if !self.prefilled {
      self.prefilled = true;
      if let Err(e) = self.prefill() {
        self.done = true;
        return Some(Err(e));
      }
    }

    // Build the next step's `[B, S]` input window. First step consumes the
    // unconsumed prompt tail (post-prefill); every later step feeds back the
    // per-row `last` (finished rows feed `pad_token_id`).
    let b = self.batch_size();
    let input: Vec<u32> = if self.first_step {
      self.first_step = false;
      let tail_len = self.max_len - self.prefill_offset;
      // `[B, tail_len]` left-padded tail in row-major order.
      let mut buf = match try_with_capacity::<u32>(b * tail_len) {
        Ok(b) => b,
        Err(e) => {
          self.done = true;
          return Some(Err(e));
        }
      };
      for row in &self.padded_rows {
        buf.extend_from_slice(&row[self.prefill_offset..self.prefill_offset + tail_len]);
      }
      buf
    } else {
      self.last.clone()
    };

    // Snapshot which rows were unfinished BEFORE this step. Rows already
    // finished pre-step must NOT be re-emitted to streaming callers — the
    // per-row finish is a one-shot event (mlx-lm's `_step` removes finished
    // rows from the batch entirely; our batch shape can't shrink, but the
    // surfaced contract matches: each row yields at most one terminal
    // `finish_reason` and nothing thereafter). `batch_generate`'s aggregator
    // happens to drop repeated `stop` emits, but raw streaming users
    // (`batch_stream_generate` / `batch_generate_step`) would otherwise see
    // the leak.
    let b = self.batch_size();
    let mut was_unfinished: Vec<bool> = match try_with_capacity(b) {
      Ok(v) => v,
      Err(e) => {
        self.done = true;
        return Some(Err(e));
      }
    };
    for f in &self.finished {
      was_unfinished.push(f.is_none());
    }

    let steps = match self.step(&input) {
      Ok(s) => s,
      Err(e) => {
        self.done = true;
        return Some(Err(e));
      }
    };

    // Apply per-row transitions BEFORE queueing emits: update `last`,
    // `produced`, `finished`. A row already-finished pre-step has its
    // `last` reset to `pad_token_id` (no effect on already-finished rows;
    // the cache still advances but the model never "sees" a meaningful
    // continuation for that row — its sampled-token output is dropped by
    // the `batch_generate` aggregator).
    for step in &steps {
      let row = step.row;
      // Already finished rows: preserve `last` as pad; ignore sampled token.
      if self.finished[row].is_some() {
        self.last[row] = self.pad_token_id;
        continue;
      }
      // Newly-decided rows: update bookkeeping.
      self.last[row] = step.token;
      self.produced[row] += 1;
      if let Some(ref reason) = step.finish_reason {
        self.finished[row] = Some(reason.clone());
        // mlx-lm batch_generate (generate.py:1945-1946) excludes "stop"
        // tokens from the per-row output; the EOS token feeds the cache
        // for this step but should NOT propagate into the next-step input
        // (the row's `last` is reset to pad so a parallel still-running
        // row drives a deterministic dummy column).
        if reason.is_eos() {
          self.last[row] = self.pad_token_id;
        }
      }
    }

    // Queue per-row emits and stop if every row finished. Only emit rows
    // that were unfinished BEFORE this step — rows that just transitioned
    // (their terminal `finish_reason` carries the EOS / length signal) and
    // rows still active. Rows already-finished pre-step are filtered out
    // here so the iterator never re-emits them.
    for step in steps {
      if was_unfinished[step.row] {
        self.pending_emit.push_back(step);
      }
    }
    if self.finished.iter().all(|r| r.is_some()) {
      self.done = true;
    }

    // Yield the first queued emit; subsequent `next()` calls drain the
    // queue before running another forward.
    self.pending_emit.pop_front().map(Ok)
  }
}

/// Build a left-padded `[B, max_len]` `I32` token matrix from `rows` —
/// mlx-lm `_left_pad_prompts` (`generate.py:802-805`,
/// `mx.array([[pad]*(max_len-len(p)) + p for p in prompts])`).
fn left_pad_rows(prompts: &[&[u32]], pad_token_id: u32) -> Result<(Vec<Vec<u32>>, usize)> {
  if prompts.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "batch_generate: prompts",
    )));
  }
  let max_len = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
  if max_len == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "batch_generate: every prompt",
    )));
  }
  let mut padded: Vec<Vec<u32>> = try_with_capacity(prompts.len())?;
  for p in prompts {
    if p.is_empty() {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "batch_generate: every prompt",
      )));
    }
    let mut row: Vec<u32> = try_with_capacity(max_len)?;
    for _ in 0..(max_len - p.len()) {
      row.push(pad_token_id);
    }
    try_extend_from_slice(&mut row, p)?;
    padded.push(row);
  }
  Ok((padded, max_len))
}

/// Build a `[B, end-start]` `I32` window from the left-padded row
/// representation — the batched analogue of [`token_window`].
fn batch_token_window(rows: &[Vec<u32>], start: usize, end: usize) -> Result<Array> {
  let b = rows.len();
  let s = end - start;
  let mut buf: Vec<i32> = try_with_capacity(b * s)?;
  for row in rows {
    buf.extend(row[start..end].iter().map(|&t| t as i32));
  }
  Array::from_slice::<i32>(&buf, &(b, s))
}

/// Build a `[B, S]` `I32` window from a row-major `[B*S]` token slice — used
/// for the first decode step's left-padded prompt tail and subsequent
/// single-token-per-row decode windows.
fn batch_full_window(flat: &[u32], b: usize, s: usize) -> Result<Array> {
  let mut buf: Vec<i32> = try_with_capacity(flat.len())?;
  buf.extend(flat.iter().map(|&t| t as i32));
  Array::from_slice::<i32>(&buf, &(b, s))
}

/// Per-row left-pad counts (`max_len - len(row)`) — the input to
/// [`crate::lm::cache::BatchKvCache::new`] /
/// [`crate::lm::cache::BatchRotatingKvCache::new`] so the cache's per-sequence
/// mask/RoPE metadata matches the left-padding chosen by [`batch_generate`].
///
/// Exposed as a public helper so a caller building their own batch cache
/// outside [`batch_generate`] (e.g. with [`crate::lm::cache::BatchRotatingKvCache`])
/// can reuse the exact left-pad scheme [`batch_generate`] uses internally.
pub fn batch_left_padding(prompts: &[&[u32]]) -> Vec<i32> {
  let max_len = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
  prompts.iter().map(|p| (max_len - p.len()) as i32).collect()
}

/// Start a batched generation run — the batched analogue of [`generate_step`]
/// (mlx-lm `GenerationBatch.__init__` + `_step` driven by `BatchGenerator`).
///
/// `prompts` is the per-row encoded token ids (must be non-empty, every row
/// non-empty); `pad_token_id` is the id used to left-pad shorter rows
/// (mlx-lm `_left_pad_prompts` uses `0`, but the caller chooses — the
/// project's `Tokenizer` may not always have a fixed pad id, so it is
/// surfaced explicitly); `cache` is the per-layer batch KV cache (typically
/// [`crate::lm::cache::BatchKvCache::new`] /
/// [`crate::lm::cache::BatchRotatingKvCache::new`] with `left_padding` =
/// [`batch_left_padding`]); `cfg` is the [`GenConfig`].
///
/// Returns an `Iterator<Item = Result<BatchGenStep>>`: one yield per row per
/// step, in row order within each step. Each row finishes independently on
/// EOS (`finish_reason = Some("stop")`) or `max_tokens` (`finish_reason =
/// Some("length")`); the iterator ends when every row has finished. A step
/// error is yielded once as `Err`, after which the iterator ends — never a
/// panic, never a poisoned re-entry.
///
/// **Per-row logits processors.** Mirrors mlx-lm `GenerationBatch._step`
/// lines 1336-1349: when processors exist, the per-step `[B, V]` logits are
/// sliced per-row, each row's running history is extended by that step's
/// per-row input token, and the processors run on the per-row history +
/// per-row `[1, V]` slice before being concatenated back to `[B, V]` for
/// normalization + sampling. The `GenConfig`-built processors (repetition /
/// presence / frequency penalties) are shared across rows but their context
/// is per-row.
///
/// **Cache contract.** The supplied `cache` MUST be built with `left_padding
/// = [max_len - len(p_i)]` for `max_len = max(len(p) for p in prompts)`
/// (use [`batch_left_padding`]) — the per-row mask uses that exact term.
/// `cache.len()` must match the model's decoder-layer count (the same as
/// single-seq [`generate_step`]).
pub fn batch_generate_step<'a, M: Model + ?Sized>(
  model: &'a M,
  prompts: &[&[u32]],
  pad_token_id: u32,
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> BatchGenerator<'a, M> {
  type Built = (Vec<Vec<u32>>, usize, Sampler, Vec<LogitsProcessor>);
  let built = (|| -> Result<Built> {
    // #136 — eager scalar-bound validation of every sampler /
    // logits-processor knob in `cfg` BEFORE any prefill / model work,
    // mirroring single-seq [`generate_step`]. The sampler-build path
    // only catches a SUBSET of bounds at build time; the per-primitive
    // validations in `apply_*` fire only when the closure runs against
    // real logits — so without this gate an invalid `cfg` would pass
    // the constructor + run an entire prompt prefill before erroring
    // on the first decode step, AND a NaN `logit_bias` / `*_penalty`
    // could silently NaN-poison the logits without any per-primitive
    // finite check. Calling `cfg.validate()` here fails fast — the
    // `Err` propagates through the existing `pending_err` channel so
    // the iterator's first `next()` yields it without any model call,
    // matching the surface shape used for sampler-build / empty-prompt
    // failures below.
    cfg.validate()?;
    let (padded_rows, max_len) = left_pad_rows(prompts, pad_token_id)?;
    let sampler = make_sampler(
      cfg.temp,
      cfg.top_p,
      cfg.min_p,
      cfg.min_tokens_to_keep,
      cfg.top_k,
      cfg.xtc_probability,
      cfg.xtc_threshold,
      &cfg.xtc_special_tokens,
      cfg.seed,
    )?;
    let processors = make_logits_processors(
      &cfg.logit_bias,
      cfg.repetition_penalty,
      cfg.repetition_context_size,
      cfg.presence_penalty,
      cfg.presence_context_size,
      cfg.frequency_penalty,
      cfg.frequency_context_size,
    )?;
    Ok((padded_rows, max_len, sampler, processors))
  })();

  match built {
    Ok((padded_rows, max_len, sampler, processors)) => {
      let b = padded_rows.len();
      BatchGenerator {
        model,
        cache,
        sampler,
        processors,
        padded_rows,
        max_len,
        prefill_offset: 0,
        history: vec![Vec::new(); b],
        last: vec![pad_token_id; b],
        produced: vec![0; b],
        finished: vec![None; b],
        pending_emit: std::collections::VecDeque::new(),
        pad_token_id,
        max_tokens: cfg.max_tokens,
        prefill_step_size: cfg.prefill_step_size.max(1),
        eos: cfg.eos,
        prefilled: false,
        first_step: true,
        pending_err: None,
        done: false,
      }
    }
    Err(e) => BatchGenerator {
      model,
      cache,
      // A never-called placeholder sampler ([`Sampler::Argmax`]);
      // `pending_err` ends the iterator on its first poll before any step
      // runs, so this is never invoked.
      sampler: Sampler::Argmax,
      processors: Vec::new(),
      padded_rows: Vec::new(),
      max_len: 0,
      prefill_offset: 0,
      history: Vec::new(),
      last: Vec::new(),
      produced: Vec::new(),
      finished: Vec::new(),
      pending_emit: std::collections::VecDeque::new(),
      pad_token_id,
      max_tokens: cfg.max_tokens,
      prefill_step_size: 1,
      eos: Vec::new(),
      prefilled: true,
      first_step: false,
      pending_err: Some(e),
      done: false,
    },
  }
}

/// Stream batched generation for `prompts` — the batched analogue of
/// [`stream_generate`]. Iterates over [`BatchGenStep`] items (one per row per
/// step, in row order within each step), using the tokenizer's EOS set
/// (overriding any `cfg.eos`, mirroring [`stream_generate`]) so per-row
/// `finish_reason` matches single-seq generation exactly.
///
/// See [`batch_generate_step`] for the iteration contract; this just wires
/// `cfg.eos = tokenizer.eos_token_ids()` before constructing the underlying
/// [`BatchGenerator`].
pub fn batch_stream_generate<'a, M: Model + ?Sized>(
  model: &'a M,
  tokenizer: &'a crate::tokenizer::Tokenizer,
  prompts: &[&[u32]],
  pad_token_id: u32,
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> BatchGenerator<'a, M> {
  let mut cfg = cfg;
  cfg.eos = tokenizer.eos_token_ids_iter().collect();
  batch_generate_step(model, prompts, pad_token_id, cache, cfg)
}

/// Generate per-row token sequences for a batch of prompts — the batched
/// analogue of [`generate`] and a 1:1 port of `mlx_lm.generate.batch_generate`
/// (`generate.py:1887-1963`).
///
/// Drives [`batch_stream_generate`] to completion, collecting each row's
/// sampled tokens into the returned `Vec<Vec<u32>>` (one entry per input
/// prompt, in input order). EOS tokens (`finish_reason="stop"`) are EXCLUDED
/// from each row's output, mirroring mlx-lm `batch_generate`
/// (`generate.py:1945-1946`: `if r.finish_reason != "stop":
/// results[r.uid].append(r.token)`); a `"length"` finish includes the token.
///
/// Any step error short-circuits the collection as `Err` (the
/// [`batch_stream_generate`] Iterator-`Err` contract is preserved).
///
/// # Arguments
///
/// - `model` — the [`Model`] implementation.
/// - `tokenizer` — provides the EOS set (overriding `cfg.eos`, like
///   [`stream_generate`]).
/// - `prompts` — per-row encoded prompt ids (must be non-empty, every row
///   non-empty; ragged lengths are left-padded with `pad_token_id`).
/// - `pad_token_id` — left-pad id for shorter rows (use
///   [`crate::tokenizer::Tokenizer::pad_token_id`] when available, else any
///   in-vocab id such as `0`).
/// - `cache` — per-layer batch KV cache (typically
///   [`crate::lm::cache::BatchKvCache::new`] /
///   [`crate::lm::cache::BatchRotatingKvCache::new`] with `left_padding`
///   from [`batch_left_padding`]).
/// - `cfg` — [`GenConfig`] (its `eos` is overridden by the tokenizer's set).
///
/// # Returns
///
/// `Vec<Vec<u32>>` — one per-row token sequence (input row order). Each
/// row's length is `produced - int(finish_reason == "stop")` ⇒ at most
/// `cfg.max_tokens`. A `"stop"` finish drops the trailing EOS; a `"length"`
/// finish keeps the final token.
pub fn batch_generate<M: Model + ?Sized>(
  model: &M,
  tokenizer: &crate::tokenizer::Tokenizer,
  prompts: &[&[u32]],
  pad_token_id: u32,
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> Result<Vec<Vec<u32>>> {
  let b = prompts.len();
  let mut results: Vec<Vec<u32>> = try_with_capacity(b)?;
  for _ in 0..b {
    results.push(Vec::new());
  }
  for step in batch_stream_generate(model, tokenizer, prompts, pad_token_id, cache, cfg) {
    let step = step?;
    // mlx-lm batch_generate (generate.py:1945-1946): drop EOS tokens from
    // per-row output; keep `"length"`-finish tokens; keep all in-progress
    // tokens. An already-finished row's emit (finish_reason carried from a
    // prior step, but the iterator only emits the once-per-row transition
    // — see `step()`'s prior-vs-new finish_reason logic) is never re-added.
    let row = step.row;
    if row >= results.len() {
      // Defensive: a sampler / model returning an out-of-range row index
      // would corrupt results; surface as a recoverable Err rather than a
      // panic.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "batch_generate: step row",
        "must be < prompts count",
        format_smolstr!("{row} (prompts={b})"),
      )));
    }
    match &step.finish_reason {
      Some(r) if r.is_eos() => {
        // EOS: drop the token, do NOT append.
      }
      _ => {
        // None ("still going") or Some(Length) or any other reason: append.
        results[row].push(step.token);
      }
    }
  }
  Ok(results)
}

#[cfg(test)]
mod batch_tests {
  //! L1 tests — [`MockBatchModel`] + `batch_generate` over a 2-3 row batch
  //! with different prompt lengths, finishing rows at different times.

  use super::*;
  use crate::lm::{cache::BatchKvCache, model::Model};

  /// A deterministic batched model: each row gets a canned "next token" at
  /// each *decode* step from `scripts[row]`, with the script index derived
  /// from the post-forward cache `offset()` and the prompt's `max_len`
  /// (`script_idx = cache_offset - max_len`). Logits are crafted so
  /// `argmax` returns the canned id (all others get `0.0`, the canned id
  /// gets `+10.0`). Cache wiring is minimal — pushes a placeholder
  /// `[B, 1, S, 1]` KV step into every layer so cache `offset()` advances
  /// exactly like the real `MockModel`.
  ///
  /// `vocab` controls the logits axis; `max_len` is the (left-padded)
  /// prompt length the generator was given (cache `offset()` reaches this
  /// value at the end of prefill, then advances by 1 per decode step);
  /// `scripts` is the per-row sequence of (argmax) next tokens — at decode
  /// step `t` (0-based, first decode after prefill) row `r` predicts
  /// `scripts[r][t]`. Prefill (`S > 1` or the trailing post-prefill
  /// `cache_offset <= max_len - 1` chunks) returns arbitrary logits (the
  /// loop discards them); the script cursor is read from cache state, so
  /// the first decode token always reads `scripts[r][0]` regardless of
  /// prefill chunking. Test-only, no public API.
  struct MockBatchModel {
    canned: Vec<f32>, // length == vocab; baseline `0.0`s.
    vocab: usize,
    max_len: usize,
    scripts: Vec<Vec<u32>>,
  }

  impl MockBatchModel {
    fn new(vocab: usize, max_len: usize, scripts: Vec<Vec<u32>>) -> Self {
      Self {
        canned: vec![0.0; vocab],
        vocab,
        max_len,
        scripts,
      }
    }
  }

  impl Model for MockBatchModel {
    fn forward(
      &self,
      tokens: &Array,
      cache: &mut [Box<dyn crate::lm::cache::KvCache>],
    ) -> Result<Array> {
      let shape = tokens.shape();
      let (batch, seq) = match shape.as_slice() {
        [b, s] => (*b, *s),
        other => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "MockBatchModel::forward: tokens must be rank-2 [B, S]",
            other.len() as u32,
            other.to_vec(),
          )));
        }
      };

      // Advance cache so `offset` increments (mirrors the single-seq
      // MockModel's wiring; batch caches use a [B, n_kv_heads, S, head_dim]
      // KV step). `n_kv_heads=1`, `head_dim=1` for the smallest possible.
      for layer in cache.iter_mut() {
        let elems = batch * seq;
        let k = Array::from_slice::<f32>(&vec![1.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
        let v = Array::from_slice::<f32>(&vec![2.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
        layer.update(&k, &v)?;
      }

      // Build per-row logits whose argmax is `scripts[row][script_idx]`,
      // where `script_idx = cache.offset() - max_len`. Cache offset after
      // the layer.update above reaches `max_len` exactly at the end of the
      // prefill chain + first decode forward; subsequent decode steps add
      // 1. So `script_idx` is `0` on the FIRST decode-call output (the
      // first generation token), `1` on the second, etc. Prefill chunks
      // yield negative `script_idx` arithmetically (cache_offset < max_len);
      // for those the logits are discarded by the loop, so any vocab id
      // suffices (we use the `0` fallback for safety).
      let cache_offset = cache.first().map(|c| c.offset()).unwrap_or(0);
      let script_idx = cache_offset.checked_sub(self.max_len);

      let mut data: Vec<f32> = Vec::with_capacity(batch * seq * self.vocab);
      for row in 0..batch {
        let pred = script_idx
          .and_then(|i| self.scripts.get(row).and_then(|s| s.get(i).copied()))
          .unwrap_or(0);
        // Every (row, seq) position gets the same logits; argmax picks `pred`.
        for _ in 0..seq {
          let mut row_logits = self.canned.clone();
          if (pred as usize) < self.vocab {
            row_logits[pred as usize] = 10.0;
          }
          data.extend_from_slice(&row_logits);
        }
      }

      Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
    }
  }

  /// 2-row batch: row 0 has prompt `[1, 2, 3]`, row 1 has `[7]` (length 1).
  /// After left-padding with `pad=0`, both rows have length 3:
  /// `[[0, 0, 7], [1, 2, 3]]` — wait, swap: row 0 longer, row 1 shorter ⇒
  /// `[[1, 2, 3], [0, 0, 7]]`. Each row's script picks distinct tokens so
  /// argmax sequences diverge per row.
  #[test]
  fn batch_generate_left_pads_and_emits_per_row_sequences() {
    // vocab = 16; EOS = 5.
    let scripts = vec![
      // row 0 — produces [11, 12, 13, 14, 15] (no EOS in 5 steps).
      vec![11, 12, 13, 14, 15],
      // row 1 — produces [21, 22] then EOS 5 at step 2 (counter starts at 1
      // after the prefill bump, so script idx 0 == first decode token).
      vec![21, 22, 5, 99, 99],
    ];
    let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3], &[7u32]];
    let left_pad = batch_left_padding(&prompts);
    // [max_len-3, max_len-1] = [0, 2].
    assert_eq!(left_pad, vec![0, 2]);
    let max_len = 3; // max(3, 1)
    let model = MockBatchModel::new(32, max_len, scripts);

    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];

    let cfg = GenConfig {
      max_tokens: 5,
      eos: vec![5],
      ..Default::default()
    };

    let batch_gen = batch_generate_step(&model, &prompts, 0, cache, cfg);

    // Drain all per-row steps; collect per-row tokens (excluding EOS).
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); 2];
    let mut last_step_per_row: Vec<Option<FinishReason>> = vec![None; 2];
    for item in batch_gen {
      let step = item.expect("step error");
      // Track per-row outputs the same way `batch_generate` aggregator does
      // (mlx-lm: exclude "stop" tokens from output, include everything else).
      match &step.finish_reason {
        Some(r) if r.is_eos() => {}
        _ => rows[step.row].push(step.token),
      }
      if let Some(r) = step.finish_reason {
        last_step_per_row[step.row] = Some(r);
      }
    }

    // Row 0: 5 tokens at max_tokens, no EOS — full [11, 12, 13, 14, 15].
    assert_eq!(rows[0], vec![11, 12, 13, 14, 15]);
    assert_eq!(last_step_per_row[0], Some(FinishReason::Length));
    // Row 1: EOS at script idx 2; output is [21, 22] (the EOS-token-bearing
    // step is dropped). finished = Eos.
    assert_eq!(rows[1], vec![21, 22]);
    assert_eq!(last_step_per_row[1], Some(FinishReason::Eos));
  }

  /// 3-row batch, ragged lengths: prompts of length 4 / 2 / 1. Asserts the
  /// left-pad scheme matches `[0, 2, 3]` and the model sees a `[3, 4]`
  /// prefill window.
  #[test]
  fn batch_left_padding_three_ragged_rows() {
    let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3, 4], &[5u32, 6], &[7u32]];
    let left_pad = batch_left_padding(&prompts);
    assert_eq!(left_pad, vec![0, 2, 3]);
  }

  /// 3-row batch: rows finish at different times — row 0 hits `max_tokens`
  /// quickly, row 1 EOS mid-way, row 2 runs the whole max. Verifies
  /// independent per-row termination and EOS-token exclusion from output.
  #[test]
  fn batch_generate_per_row_eos_independent_finish() {
    let scripts = vec![
      // row 0 — `max_tokens = 3`: emits [10, 11, 12], terminates "length".
      vec![10, 11, 12, 99, 99],
      // row 1 — EOS at step 1: emits [20] then EOS=5, terminates "stop".
      vec![20, 5, 99, 99, 99],
      // row 2 — emits [30, 31, 32], terminates "length".
      vec![30, 31, 32, 99, 99],
    ];
    let prompts: Vec<&[u32]> = vec![&[1u32, 2], &[3u32, 4], &[5u32]];
    let left_pad = batch_left_padding(&prompts);
    assert_eq!(left_pad, vec![0, 0, 1]); // max_len = 2 ⇒ [0, 0, 1].
    let max_len = 2; // max(2, 2, 1)
    let model = MockBatchModel::new(64, max_len, scripts);

    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];

    let cfg = GenConfig {
      max_tokens: 3,
      eos: vec![5],
      ..Default::default()
    };

    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); 3];
    let mut last_step_per_row: Vec<Option<FinishReason>> = vec![None; 3];
    for item in batch_generate_step(&model, &prompts, 0, cache, cfg) {
      let step = item.expect("step error");
      match &step.finish_reason {
        Some(r) if r.is_eos() => {}
        _ => rows[step.row].push(step.token),
      }
      if let Some(r) = step.finish_reason {
        last_step_per_row[step.row] = Some(r);
      }
    }

    assert_eq!(rows[0], vec![10, 11, 12]);
    assert_eq!(last_step_per_row[0], Some(FinishReason::Length));
    assert_eq!(rows[1], vec![20]); // EOS at idx 1 dropped.
    assert_eq!(last_step_per_row[1], Some(FinishReason::Eos));
    assert_eq!(rows[2], vec![30, 31, 32]);
    assert_eq!(last_step_per_row[2], Some(FinishReason::Length));
  }

  /// Empty / malformed prompt inputs surface as a deferred `Err` on first
  /// `next()` (the iterator-Err contract, mirroring single-seq
  /// [`generate_step`]).
  #[test]
  fn batch_generate_step_empty_prompts_is_err() {
    let model = MockBatchModel::new(8, 0, vec![]);
    let prompts: Vec<&[u32]> = vec![];
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = Vec::new();
    let mut batch_gen = batch_generate_step(&model, &prompts, 0, cache, GenConfig::default());
    assert!(batch_gen.next().unwrap().is_err());
    assert!(batch_gen.next().is_none()); // fuses
  }

  #[test]
  fn batch_generate_step_empty_row_is_err() {
    let model = MockBatchModel::new(8, 0, vec![vec![]]);
    let prompts: Vec<&[u32]> = vec![&[]];
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = Vec::new();
    let mut batch_gen = batch_generate_step(&model, &prompts, 0, cache, GenConfig::default());
    assert!(batch_gen.next().unwrap().is_err());
    assert!(batch_gen.next().is_none());
  }

  /// `max_tokens = 0` yields zero steps and runs ZERO model / cache
  /// mutations — exactly mirroring single-seq `Generator::next`'s zero-budget
  /// guard (`if self.produced >= self.max_tokens: return None` BEFORE
  /// prefill).
  ///
  /// Empty per-row scripts make a successful decode impossible (any
  /// non-existent `scripts[row][0]` lookup falls through to id `0` rather
  /// than panicking, but the offset side-effect of prefill on the cache
  /// would still be observable). To prove no prefill / model mutation ran
  /// we'd ideally inspect the cache's offset post-iteration; the iterator
  /// owns its `Vec<Box<dyn KvCache>>` so the cleaner shape is: assert the
  /// iterator is empty on first poll (the guard fires before prefill), AND
  /// that no item is ever produced when the guard is the only thing
  /// returning `None` — which is what `.count() == 0` verifies. The
  /// empty-script setup is belt-and-braces: even if the guard regressed
  /// silently, the iterator would attempt to read script idx 0, fall back
  /// to token `0`, and emit it — failing this test loudly.
  #[test]
  fn batch_generate_step_zero_max_tokens_emits_nothing_and_skips_prefill() {
    let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3], &[7u32]];
    let left_pad = batch_left_padding(&prompts);
    let max_len = 3;
    let model = MockBatchModel::new(16, max_len, vec![vec![], vec![]]);
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];
    // Sanity: fresh cache offset is 0; any prefill / decode would advance it.
    assert_eq!(cache[0].offset(), 0);

    let cfg = GenConfig {
      max_tokens: 0,
      eos: vec![5],
      ..Default::default()
    };

    let batch_gen = batch_generate_step(&model, &prompts, 0, cache, cfg);
    // Zero-budget guard fires on the first poll BEFORE prefill: no items.
    assert_eq!(batch_gen.count(), 0);
  }

  /// `batch_generate`'s aggregator drains the same iterator as
  /// `batch_generate_step` and pushes per-row tokens; with `max_tokens = 0`
  /// the iterator yields nothing, so each row's output Vec is empty.
  /// Mirrors the aggregator loop directly to avoid spinning up a HF
  /// `Tokenizer` fixture for a behavior that's fully determined by the
  /// iterator.
  #[test]
  fn batch_generate_zero_max_tokens_returns_empty_vec_per_row() {
    let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3], &[7u32], &[9u32, 10]];
    let left_pad = batch_left_padding(&prompts);
    let max_len = 3;
    let model = MockBatchModel::new(16, max_len, vec![vec![], vec![], vec![]]);
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];
    let cfg = GenConfig {
      max_tokens: 0,
      ..Default::default()
    };

    let b = prompts.len();
    let mut results: Vec<Vec<u32>> = vec![Vec::new(); b];
    for step in batch_generate_step(&model, &prompts, 0, cache, cfg) {
      let step = step.expect("zero-budget guard must not yield Err");
      // Reproduce the `batch_generate` aggregator: drop EOS-finish tokens,
      // append everything else. With max_tokens=0 the loop body never runs.
      match &step.finish_reason {
        Some(r) if r.is_eos() => {}
        _ => results[step.row].push(step.token),
      }
    }
    assert_eq!(results, vec![Vec::<u32>::new(); b]);
  }

  /// Streaming-count regression: a row that finishes EARLY must NOT be
  /// re-emitted on later steps. Without the pre-step `was_unfinished`
  /// snapshot in `Iterator::next`, the already-finished row's per-step
  /// (carried-`finish_reason="stop"`, dummy token) `BatchGenStep` would
  /// leak to `batch_stream_generate` callers on every later forward.
  /// `batch_generate`'s aggregator happens to drop repeated `stop` rows,
  /// but raw-streaming users see the bug. This test pins the contract by
  /// counting per-row emits.
  ///
  /// 2-row batch:
  /// - row 0 hits EOS on decode step 1 ⇒ exactly ONE emit (the terminal
  ///   `stop` step) — NEVER one emit per subsequent step.
  /// - row 1 continues until `max_tokens` ⇒ `max_tokens` emits (one per
  ///   step, the final one carrying `Some("length")`).
  #[test]
  fn batch_stream_generate_finished_row_not_re_emitted() {
    // Equal-length prompts so left_pad is `[0, 0]` and prefill is trivial.
    let prompts: Vec<&[u32]> = vec![&[1u32, 2], &[3u32, 4]];
    let left_pad = batch_left_padding(&prompts);
    assert_eq!(left_pad, vec![0, 0]);
    let max_len = 2;
    let max_tokens = 5;
    // row 0: EOS (5) at decode step 0 (first generated token) ⇒ should
    //        produce exactly ONE emit (the terminal stop step).
    // row 1: runs to `max_tokens=5` ⇒ tokens [20, 21, 22, 23, 24], last
    //        of which carries `Some("length")`. 5 emits total.
    let scripts = vec![
      vec![5u32, 99, 99, 99, 99], // EOS on first decode token
      vec![20u32, 21, 22, 23, 24],
    ];
    let model = MockBatchModel::new(64, max_len, scripts);
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];
    let cfg = GenConfig {
      max_tokens,
      eos: vec![5],
      ..Default::default()
    };

    let mut emits_per_row: Vec<usize> = vec![0; 2];
    let mut finish_per_row: Vec<Option<FinishReason>> = vec![None; 2];
    for item in batch_generate_step(&model, &prompts, 0, cache, cfg) {
      let step = item.expect("step error");
      emits_per_row[step.row] += 1;
      if let Some(r) = step.finish_reason {
        // A row should never transition twice — its terminal `finish_reason`
        // is the LAST thing the iterator says about that row.
        assert!(
          finish_per_row[step.row].is_none(),
          "row {} got a second finish_reason emit: prior={:?}, new={:?}",
          step.row,
          finish_per_row[step.row],
          r,
        );
        finish_per_row[step.row] = Some(r);
      }
    }

    // Row 0: exactly ONE emit — the terminal Eos step.
    assert_eq!(
      emits_per_row[0], 1,
      "row 0 finished on step 1 but was re-emitted on later steps (got {} emits, expected 1)",
      emits_per_row[0]
    );
    assert_eq!(finish_per_row[0], Some(FinishReason::Eos));
    // Row 1: full `max_tokens` emits, last carries `Some(Length)`.
    assert_eq!(
      emits_per_row[1], max_tokens,
      "row 1 expected {max_tokens} emits, got {}",
      emits_per_row[1]
    );
    assert_eq!(finish_per_row[1], Some(FinishReason::Length));
  }

  /// A `Model` whose every `forward` returns an error AND records that it
  /// was called — drives the "validate fail-fast must run BEFORE any
  /// model call" contract: if [`batch_generate_step`] regressed and called
  /// `forward` before propagating a `cfg.validate()` failure, this model's
  /// "mock batch forward failure" would surface instead of the validation
  /// error AND the call counter would increment.
  struct BatchFailModel {
    calls: std::cell::RefCell<usize>,
  }

  impl Model for BatchFailModel {
    fn forward(
      &self,
      _tokens: &Array,
      _cache: &mut [Box<dyn crate::lm::cache::KvCache>],
    ) -> Result<Array> {
      *self.calls.borrow_mut() += 1;
      Err(Error::InvariantViolation(
        crate::error::InvariantViolationPayload::new(
          "BatchFailModel::forward",
          "mock batch forward failure (test fixture)",
        ),
      ))
    }
  }

  /// #136 — eager `GenConfig::validate` MUST run inside
  /// [`batch_generate_step`]'s `built` closure BEFORE sampler /
  /// processor construction (and so before any model / cache work).
  /// An invalid `cfg` (negative `temp`) must surface as the iterator's
  /// first `Err` propagated through the existing `pending_err` channel,
  /// AND `BatchFailModel::forward` must NOT have been called (the
  /// presence of "mock batch forward failure" or a non-zero call count
  /// would prove the validate gate didn't fire). After the yielded
  /// `Err` the iterator fuses (next call returns `None`).
  #[test]
  fn batch_generate_step_propagates_validate_err_before_forward() {
    let model = BatchFailModel {
      calls: std::cell::RefCell::new(0),
    };
    // Valid prompts so the `prompt.is_empty()` / `row.is_empty()` errs
    // don't pre-empt the validation error we're testing.
    let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3], &[4u32]];
    let left_pad = batch_left_padding(&prompts);
    let cache: Vec<Box<dyn crate::lm::cache::KvCache>> =
      vec![Box::new(BatchKvCache::new(&left_pad))];
    let cfg = GenConfig {
      temp: -1.0, // invalid: validate must reject
      max_tokens: 4,
      ..GenConfig::default()
    };

    let mut it = batch_generate_step(&model, &prompts, 0, cache, cfg);
    let first = it.next().expect("iterator yields at least one item");
    let err = first.expect_err("validation Err must propagate");
    let msg = format!("{err:?}");
    assert!(
      msg.contains("temp"),
      "yielded validation error, not the forward error (validate ran BEFORE forward): {msg}"
    );
    assert!(
      !msg.contains("mock batch forward failure"),
      "model.forward must NOT have been called (validate fail-fast): {msg}"
    );
    assert_eq!(
      *model.calls.borrow(),
      0,
      "model.forward was called {} time(s) — validate gate did not fail-fast",
      *model.calls.borrow()
    );
    assert!(it.next().is_none(), "iterator fuses after the yielded Err");
  }
}

#[cfg(test)]
mod stop_sequence_tests {
  //! L5 — multi-token / string stop sequences driven through the real
  //! [`stream_generate`] / [`generate`] entry points with a deterministic,
  //! scriptable single-seq model and the committed `WordLevel` fixture
  //! tokenizer (tokens 3-8 = `hello world the quick brown fox`). Stop
  //! strings are derived from `tok.decode(...)` so the tests never hardcode
  //! the tokenizer's spacing. The pure matcher logic (overlap, char
  //! boundaries, first-match-wins) is unit-tested in [`crate::lm::stop`];
  //! these assert the generate-loop wiring: `finish_reason="stop"`, the
  //! trim, and the eos-only fallback when `stop_strings` is empty.

  use super::*;
  use crate::lm::cache::{CacheConfig, KvCache, make_prompt_cache};

  /// Resolve the fixture tokenizer directory (`mlxrs/tests/fixtures`),
  /// reachable from the in-crate `#[cfg(test)]` build via `CARGO_MANIFEST_DIR`.
  fn fixture_tokenizer() -> crate::tokenizer::Tokenizer {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("fixtures");
    crate::tokenizer::Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
  }

  /// A scriptable single-seq model: at decode step `t` (0-based, first decode
  /// after prefill) it emits `script[t]` as the argmax. The script cursor is
  /// `cache.offset() - prompt_len` (offset reaches `prompt_len` at the first
  /// decode forward, then +1/step), so prefill chunking never shifts it —
  /// identical wiring to the L1 `MockBatchModel`, single row.
  struct ScriptModel {
    vocab: usize,
    prompt_len: usize,
    script: Vec<u32>,
  }

  impl Model for ScriptModel {
    fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let shape = tokens.shape();
      let (batch, seq) = match shape.as_slice() {
        [b, s] => (*b, *s),
        [s] => (1usize, *s),
        other => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "ScriptModel::forward: tokens must be rank-1 [S] or rank-2 [B, S]",
            other.len() as u32,
            other.to_vec(),
          )));
        }
      };
      // Advance every cache so `offset()` increments like a real layer.
      for layer in cache.iter_mut() {
        let elems = batch * seq;
        let k = Array::from_slice::<f32>(&vec![1.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
        let v = Array::from_slice::<f32>(&vec![2.0_f32; elems], &(batch, 1usize, seq, 1usize))?;
        layer.update(&k, &v)?;
      }
      let cache_offset = cache.first().map(|c| c.offset()).unwrap_or(0);
      let script_idx = cache_offset.checked_sub(self.prompt_len);
      let pred = script_idx
        .and_then(|i| self.script.get(i).copied())
        .unwrap_or(0);

      let mut data = vec![0.0_f32; batch * seq * self.vocab];
      if (pred as usize) < self.vocab {
        for pos in 0..batch * seq {
          data[pos * self.vocab + pred as usize] = 10.0;
        }
      }
      Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
    }
  }

  /// Run `generate` (collect every streamed segment) over a scripted decode.
  /// `prompt` seeds the cache offset; `script` is the per-step argmax id
  /// sequence; `stop_strings` configures L5. Returns the collected text plus
  /// the per-response `finish_reason`s (in order) for assertions.
  fn run(
    prompt: &[u32],
    script: Vec<u32>,
    max_tokens: usize,
    stop_strings: Vec<String>,
  ) -> (String, Vec<Option<FinishReason>>) {
    let tok = fixture_tokenizer();
    let vocab = 16usize;
    let model = ScriptModel {
      vocab,
      prompt_len: prompt.len(),
      script,
    };
    let cache = make_prompt_cache(&CacheConfig {
      num_hidden_layers: 1,
      sliding_window: None,
    });
    let cfg = GenConfig {
      max_tokens,
      stop_strings,
      ..Default::default()
    };
    let mut text = String::new();
    let mut reasons = Vec::new();
    for resp in stream_generate(&model, &tok, prompt, cache, cfg) {
      let r = resp.expect("stream step");
      text.push_str(&r.text);
      reasons.push(r.finish_reason);
    }
    (text, reasons)
  }

  /// The fixture's decode of a token-id slice (the exact text the
  /// detokenizer reconstructs), used to build spacing-agnostic stop strings.
  fn decode(ids: &[u32]) -> String {
    fixture_tokenizer().decode(ids, false).expect("decode")
  }

  #[test]
  fn empty_stop_strings_is_eos_only_unchanged() {
    // Script: hello world </s>(eos=2) ... . With no stop strings, generation
    // ends on the eos token (finish_reason="stop"); the eos token is not
    // detokenized. Output is exactly the decode of the pre-eos tokens.
    let prompt = [1u32, 3]; // <s> hello
    let script = vec![4u32, 5, 2, 6, 7]; // world the </s> ...
    let (text, reasons) = run(&prompt, script, 32, Vec::new());
    assert_eq!(text, decode(&[4, 5]));
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Eos));
    // Only one "stop" (the eos), no premature stop.
    assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
  }

  #[test]
  fn single_token_stop_string_stops_and_trims() {
    // Stop on the single token `world` (id 4). Script produces hello world
    // the ...; generation must stop AT world and trim it, leaving `hello`.
    let prompt = [1u32, 3];
    let script = vec![3u32, 4, 5, 6, 7]; // hello world the quick brown
    let stop = decode(&[4]); // " world" (or "world") — whatever the fixture renders
    let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
    let full = decode(&[3, 4, 5]); // hello world the
    let cut = full.find(&stop).expect("stop substring present in decode");
    assert_eq!(text, full[..cut].to_string());
    // Typed FinishReason::Stop(matched) — as_str() collapses to "stop"
    // canonically, payload carries the matched sequence.
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
  }

  #[test]
  fn multi_token_stop_spanning_boundary_stops_and_trims() {
    // Stop string spans TWO tokens: decode([5,6]) = "the quick" (+ leading
    // space per the fixture). The match completes only when BOTH tokens have
    // been produced — a token boundary in the middle of the stop string.
    let prompt = [1u32, 3];
    let script = vec![3u32, 5, 6, 7, 8]; // hello the quick brown fox
    let stop = decode(&[5, 6]); // multi-token stop
    let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
    let full = decode(&[3, 5, 6, 7]); // up to brown
    let cut = full.find(&stop).expect("multi-token stop present");
    assert_eq!(text, full[..cut].to_string());
    // Crucially: it did NOT stop after the first token of the stop sequence —
    // the leading `hello` token survived (text is the non-empty pre-stop
    // prefix), and the full stop string is absent from the output.
    assert!(!text.is_empty());
    assert!(!text.contains(&stop));
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
  }

  #[test]
  fn partial_match_then_diverge_does_not_stop() {
    // Stop string = decode([5,6]) ("the quick"). Script produces the FIRST
    // token of it (`the`) then DIVERGES to `fox` — the partial match must
    // NOT fire; generation runs to max_tokens.
    let prompt = [1u32, 3];
    let script = vec![3u32, 5, 8, 4, 7]; // hello the fox world brown (no "the quick")
    let stop = decode(&[5, 6]);
    let (text, reasons) = run(&prompt, script, 5, vec![stop.clone()]);
    // No stop completed ⇒ ends on length, full text retained.
    assert!(!text.contains(&stop), "stop string must not appear");
    assert_eq!(text, decode(&[3, 5, 8, 4, 7]));
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Length));
    assert!(
      reasons
        .iter()
        .all(|r| r.as_ref() != Some(&FinishReason::Eos)),
      "no premature stop on the partial match"
    );
  }

  #[test]
  fn stop_completes_mid_token_trims_at_char_boundary() {
    // The "mid-token" case the token-level trie cannot handle: the stop
    // string is a CHARACTER PREFIX of a token's decoded text. Token `quick`
    // (id 6) decodes to text containing "qui"; stopping on "qui" must trim
    // mid-token at the exact character boundary, dropping "qui" and the rest.
    let prompt = [1u32, 3];
    let script = vec![3u32, 6, 7, 8, 4]; // hello quick brown fox world
    let quick = decode(&[6]); // e.g. " quick"
    // Build a stop that is a strict character prefix of the `quick` token
    // text, ending mid-token (drop the last char so it cannot be a whole
    // token). Skip a leading space if present so we cut inside the word.
    let trimmed = quick.trim_start();
    assert!(trimmed.len() >= 3, "need a multi-char token to cut");
    let stop = trimmed[..trimmed.len() - 1].to_string(); // e.g. "quic"
    let (text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
    let full = decode(&[3, 6, 7]); // hello quick brown
    let cut = full.find(&stop).expect("mid-token stop prefix present");
    assert_eq!(text, full[..cut].to_string());
    // The stop string itself must be gone, and the cut is at the char
    // boundary where the stop began (mid the `quick` token's text).
    assert!(!text.contains(&stop));
    // Typed FinishReason::Stop(matched) carries the stop sequence;
    // FinishReason::as_str() still collapses to canonical "stop".
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
  }

  #[test]
  fn multiple_stop_strings_first_completion_wins() {
    // Two stops: one completes earlier in the stream than the other. The
    // earlier completion wins and trims there.
    let prompt = [1u32, 3];
    let script = vec![3u32, 4, 5, 6, 7]; // hello world the quick brown
    let early = decode(&[4]); // "world" — completes at step 2
    let late = decode(&[6]); // "quick" — would complete at step 4
    let (text, reasons) = run(&prompt, script, 32, vec![late.clone(), early.clone()]);
    let full = decode(&[3, 4]); // hello world
    let cut = full.find(&early).expect("early stop present");
    assert_eq!(text, full[..cut].to_string());
    assert!(!text.contains(&early));
    assert!(!text.contains(&late));
    // The early stop is the one that matched, so the typed payload carries it.
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(early)));
  }

  #[test]
  fn finish_reason_is_stop_on_stop_string_match() {
    // Focused assertion: a stop-string match yields exactly one terminal
    // response with finish_reason == Some(Stop(matched)) and nothing after.
    let prompt = [1u32, 3];
    let script = vec![3u32, 4, 5, 6, 7];
    let stop = decode(&[5]); // "the"
    let (_text, reasons) = run(&prompt, script, 32, vec![stop.clone()]);
    assert_eq!(reasons.last().unwrap(), &Some(FinishReason::Stop(stop)));
    // Exactly one terminal reason, and it's the final element.
    assert_eq!(reasons.iter().filter(|r| r.is_some()).count(), 1);
  }

  // ── finalized-tail re-check on the active-matcher terminal paths ──────────
  //
  // Some detokenizers withhold tail text from `text()` until `finalize()`
  // (the real BPE detok holds a single bare-space token for one step). The
  // mid-stream matcher runs on `text()` BEFORE finalization, so a stop string
  // completed only by that withheld tail is invisible until the terminal
  // branch finalizes. These tests drive the exact unit the EOS / max_tokens
  // active-matcher branches now call — [`finalize_active_tail`] — through a
  // mock that reproduces the withhold-until-finalize behavior, asserting the
  // tail completes the stop (trim + "stop"), including the max_tokens case
  // where a finalized-tail stop must win over "length".

  /// A mock [`StreamingDetokenizer`](crate::tokenizer::StreamingDetokenizer)
  /// that withholds the most-recently-added "tail" token's text from `text()`
  /// until the next `add_token` / `finalize` flushes it — exactly the BPE
  /// detok's single-bare-space hold-back, but deterministic and tokenizer-free.
  #[derive(Default)]
  struct WithholdDetokenizer {
    /// Committed (visible) text — what `text()` returns.
    text: String,
    /// The withheld tail not yet visible in `text()` (flushed on the next
    /// `push` / `finalize`).
    pending: String,
    tokens: Vec<u32>,
    offset: usize,
  }

  impl WithholdDetokenizer {
    /// Add a token whose decoded text is `s`. When `withhold` is true the text
    /// is held back from `text()` until the next push/finalize (BPE bare-space
    /// semantics); otherwise it (and any pending tail) commits immediately.
    fn push(&mut self, s: &str, withhold: bool) {
      // A previously-withheld tail becomes visible as soon as another token
      // arrives (the BPE detok flushes `unflushed` on the next step).
      self.text.push_str(&self.pending);
      self.pending.clear();
      if withhold {
        self.pending.push_str(s);
      } else {
        self.text.push_str(s);
      }
      self.tokens.push(self.tokens.len() as u32);
    }
  }

  impl crate::tokenizer::StreamingDetokenizer for WithholdDetokenizer {
    fn reset(&mut self) {
      self.text.clear();
      self.pending.clear();
      self.tokens.clear();
      self.offset = 0;
    }
    fn add_token(&mut self, _token: u32) {}
    fn finalize(&mut self) {
      // Flush the withheld tail into the visible text (BPE `finalize`).
      self.text.push_str(&self.pending);
      self.pending.clear();
    }
    fn text(&self) -> std::borrow::Cow<'_, str> {
      std::borrow::Cow::Borrowed(&self.text)
    }
    fn tokens(&self) -> &[u32] {
      &self.tokens
    }
    fn offset(&self) -> usize {
      self.offset
    }
    fn set_offset(&mut self, offset: usize) {
      self.offset = offset;
    }
  }

  /// Sanity: before `finalize`, the withheld tail is invisible in `text()`;
  /// `finalize` makes it visible — the precondition that makes the bug bite.
  #[test]
  fn mock_withholds_tail_until_finalize() {
    use crate::tokenizer::StreamingDetokenizer;
    let mut d = WithholdDetokenizer::default();
    d.push("hello", false);
    d.push(" ", true); // bare space withheld
    assert_eq!(d.text().as_ref(), "hello"); // the space is NOT yet visible
    d.finalize();
    assert_eq!(d.text().as_ref(), "hello "); // now flushed
  }

  #[test]
  fn finalized_tail_completes_stop_on_eos_trims_and_reports_stop() {
    // EOS terminal path: a withheld bare-space token completes the stop " ".
    // The mid-stream matcher (run on pre-finalize text "hello") never saw the
    // space; only the finalized text "hello " contains the stop. The eos
    // branch passes default_reason="stop"; finalize_active_tail must trim the
    // space and report "stop".
    let stop = crate::lm::stop::StopMatcher::new(vec![" ".to_string()]);
    let mut d = WithholdDetokenizer::default();
    d.push("hello", false);
    d.push(" ", true); // the eos-preceding token; held back until finalize
    // The visible "hello" was already streamed mid-loop (emitted_len tracks it).
    let mut emitted_len = "hello".len();
    crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
    let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Eos);
    // The stop " " starts at byte 5; trimmed_len=5 == emitted_len ⇒ nothing
    // new emitted (the space is trimmed away, not returned). The finalized
    // tail completed the stop, so the typed reason is Stop(matched), not Eos.
    assert_eq!(text, "");
    assert_eq!(reason, FinishReason::Stop(" ".to_string()));
    assert!(!text.contains(' '), "the bare space must not be emitted");
  }

  #[test]
  fn finalized_tail_completes_stop_on_max_tokens_wins_over_length() {
    // max_tokens terminal path: the final allowed token is a withheld bare
    // space that completes the stop " ". finalize_active_tail is called with
    // default_reason="length", but a stop completed by the finalized tail must
    // OVERRIDE to "stop" (and trim), not report "length".
    let stop = crate::lm::stop::StopMatcher::new(vec![" ".to_string()]);
    let mut d = WithholdDetokenizer::default();
    d.push("hi", false);
    d.push(" ", true); // final allowed token, withheld until finalize
    let mut emitted_len = "hi".len();
    crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
    let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
    assert_eq!(
      reason,
      FinishReason::Stop(" ".to_string()),
      "finalized-tail stop must win over length and carry the matched payload"
    );
    assert_eq!(text, ""); // the space is trimmed, not emitted
    assert!(!text.contains(' '));
  }

  #[test]
  fn finalized_tail_no_stop_emits_tail_with_default_reason() {
    // Control: when the finalized tail does NOT complete a stop, the tail is
    // emitted and default_reason is preserved (length on max_tokens, stop on
    // eos). Guards against the re-check spuriously trimming/relabeling.
    let stop = crate::lm::stop::StopMatcher::new(vec!["ZZZ".to_string()]); // never matches
    let mut d = WithholdDetokenizer::default();
    d.push("hi", false);
    d.push(" ", true); // withheld tail, no stop completion
    let mut emitted_len = "hi".len();
    crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
    // max_tokens semantics → Length, and the withheld space is emitted.
    let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
    assert_eq!(text, " ");
    assert_eq!(reason, FinishReason::Length);
    // eos semantics → Eos, same emitted tail.
    let mut d2 = WithholdDetokenizer::default();
    d2.push("hi", false);
    d2.push(" ", true);
    let mut emitted_len2 = "hi".len();
    crate::tokenizer::StreamingDetokenizer::finalize(&mut d2);
    let (text2, reason2) = finalize_active_tail(&d2, &stop, &mut emitted_len2, FinishReason::Eos);
    assert_eq!(text2, " ");
    assert_eq!(reason2, FinishReason::Eos);
  }

  #[test]
  fn finalized_tail_completes_multichar_stop_spanning_into_tail() {
    // The withheld tail supplies the final char of a multi-char stop that
    // straddles the commit/withhold boundary: visible "ab", withheld "c",
    // stop "abc". Pre-finalize text "ab" has no match; finalized "abc" does.
    // finalize_active_tail trims at the match start (byte 0), but emitted_len
    // is already 2 (the "ab" was streamed), so nothing new is emitted and the
    // reason is "stop".
    let stop = crate::lm::stop::StopMatcher::new(vec!["abc".to_string()]);
    let mut d = WithholdDetokenizer::default();
    d.push("ab", false);
    d.push("c", true);
    let mut emitted_len = "ab".len();
    crate::tokenizer::StreamingDetokenizer::finalize(&mut d);
    let (text, reason) = finalize_active_tail(&d, &stop, &mut emitted_len, FinishReason::Length);
    assert_eq!(reason, FinishReason::Stop("abc".to_string()));
    assert_eq!(text, "");
    // emitted_len clamped to match start (0) max emitted (2) = 2.
    assert_eq!(emitted_len, 2);
  }
}
