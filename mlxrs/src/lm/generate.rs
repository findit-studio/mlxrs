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
//! [`Iterator`] (the spec's A1) — [`generate_step`] yields one [`GenStep`]
//! per step (the typed step item: `token` + `logprobs`),
//! [`stream_generate`] maps that through the #18 streaming detokenizer into
//! [`GenerationResponse`]s, and [`generate`] collects the whole thing into a
//! `String`.
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
//! 5. `token = sampler(logprobs)` — the [`make_sampler`] chain
//!    (top-k/p, min-p, xtc, categorical) or the default temperature-0
//!    `argmax`.
//! 6. yield `GenStep { token, logprobs: logprobs.squeeze(0) }`; stop when
//!    `token ∈ eos` (`finish_reason = "stop"`) or `count == max_tokens`
//!    (`finish_reason = "length"`).
//!
//! **Prefill** is chunked by [`GenConfig::prefill_step_size`] (mlx-lm lines
//! 430-453): the prompt's first `total - 1` tokens are fed in
//! `prefill_step_size`-sized chunks (logits discarded, cache filled); the
//! last token starts the first decode step.
//!
//! **Error model (spec §4):** every fallible op returns [`crate::Result`];
//! [`generate_step`] / [`stream_generate`] are `Iterator<Item = Result<..>>`
//! — a step error is yielded **once** as `Err` and then the iterator ends
//! (it fuses — no panic, no poison, never re-entered). No implicit eval: the
//! only materialization is the `.item::<u32>()` at the explicit
//! token-extraction boundary (mlx-lm's `y.item()`); `logprobs` stays lazy.
//!
//! `make_sampler` / `make_logits_processors` **compose** the [`sample`] /
//! #29 primitives and propagate their validation `Err`s — they do **not**
//! re-validate ranges `sample.rs` already enforces (spec §2/§4). `temp == 0`
//! ⇒ the argmax sampler (mlx-lm `make_sampler` line 46). All sampler /
//! processor scalars stay in the compute dtype via the #29 `scalar_like`
//! discipline.
//!
//! [`Model`]: crate::lm::model::Model
//! [`sample`]: crate::lm::sample

use std::cell::RefCell;

use crate::{
  array::Array,
  error::{Error, Result},
  lm::{cache::KvCache, model::Model, sample},
  ops,
};

/// A logits processor: maps `(recent token-id history, raw logits)` to
/// processed logits, exactly mlx-lm's
/// `Callable[[mx.array, mx.array], mx.array]` (`make_logits_processors`
/// closures). Boxed so the heterogeneous bias / repetition / presence /
/// frequency closures share one list.
pub type LogitsProcessor = Box<dyn Fn(&[u32], &Array) -> Result<Array>>;

/// A sampler: maps a log-probability vector to a sampled token id array
/// (`[1]`, `U32`), exactly mlx-lm's `Callable[[mx.array], mx.array]`. `FnMut`
/// because the stochastic chain owns and advances its own PRNG key per call
/// (mirroring mlx-lm's per-call `mx.random.state` split); the default
/// temperature-0 `argmax` sampler is pure.
pub type Sampler = Box<dyn FnMut(&Array) -> Result<Array>>;

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
/// bias params, plus the resolved eos-id set (spec §2).
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
  pub xtc_special_tokens: Vec<i32>,

  // --- logits-processor params (mlx-lm `make_logits_processors`) --------
  /// Additive logit bias as `(token_id, bias)` pairs (mlx-lm's
  /// `Dict[int, float]`). Applied first, before the penalties.
  pub logit_bias: Vec<(i32, f32)>,
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
  pub eos: Vec<u32>,

  /// Stochastic-sampler RNG seed (mlx-lm's `mx.random.seed` analogue).
  /// `Some(s)` ⇒ a non-greedy run is reproducible; `None` ⇒ a fresh
  /// process-unique seed per run so independent non-greedy generations never
  /// restart from the same sequence (mlx-lm's default — see [`make_sampler`]).
  /// Ignored when `temp == 0` (the deterministic argmax sampler).
  pub seed: Option<u64>,
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
      seed: None,
    }
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
  if temp == 0.0 {
    return Ok(Box::new(|logprobs: &Array| sample::argmax_sample(logprobs)));
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

  let sampler = move |logprobs: &Array| -> Result<Array> {
    let (k_xtc, k_cat) = {
      let mut k = key.borrow_mut();
      let (next, k_xtc) = ops::random::split(&k)?;
      let (next, k_cat) = ops::random::split(&next)?;
      *k = next;
      (k_xtc, k_cat)
    };
    let mut x = if do_top_p {
      sample::apply_top_p(logprobs, top_p)?
    } else {
      logprobs.try_clone()?
    };
    if do_min_p {
      x = sample::apply_min_p(&x, min_p, min_tokens_to_keep)?;
    }
    if do_xtc {
      x = sample::apply_xtc(&x, xtc_probability, xtc_threshold, &xtc_special, &k_xtc)?;
    }
    if do_top_k {
      x = sample::apply_top_k(&x, top_k)?;
    }
    sample::categorical_sampling(&x, temp, &k_cat)
  };
  Ok(Box::new(sampler))
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
  // built once here, not rebuilt per step (faithful + no per-step alloc).
  if !logit_bias.is_empty() {
    let indices: Vec<i32> = logit_bias.iter().map(|&(i, _)| i).collect();
    let values_vec: Vec<f32> = logit_bias.iter().map(|&(_, v)| v).collect();
    let values = Array::from_slice::<f32>(&values_vec, &(values_vec.len(),))?;
    let proc: LogitsProcessor = Box::new(move |_tokens: &[u32], logits: &Array| {
      sample::apply_logit_bias(logits, &indices, &values)
    });
    processors.push(proc);
  }

  // mlx-lm: `(make_repetition_penalty, repetition_penalty,
  // repetition_context_size), (make_presence_penalty, ...,
  // presence_context_size), (make_frequency_penalty, ...,
  // frequency_context_size)` — each appended iff `penalty is not None and
  // penalty != 0`, in this order, each capturing its OWN context size.
  if let Some(p) = repetition_penalty.filter(|&p| p != 0.0) {
    let ctx = repetition_context_size;
    let proc: LogitsProcessor = Box::new(move |tokens: &[u32], logits: &Array| {
      let ids = recent_ids(tokens, ctx);
      sample::apply_repetition_penalty(logits, &ids, p)
    });
    processors.push(proc);
  }
  if let Some(p) = presence_penalty.filter(|&p| p != 0.0) {
    let ctx = presence_context_size;
    let proc: LogitsProcessor = Box::new(move |tokens: &[u32], logits: &Array| {
      let ids = recent_ids(tokens, ctx);
      sample::apply_presence_penalty(logits, &ids, p)
    });
    processors.push(proc);
  }
  if let Some(p) = frequency_penalty.filter(|&p| p != 0.0) {
    let ctx = frequency_context_size;
    let proc: LogitsProcessor = Box::new(move |tokens: &[u32], logits: &Array| {
      let ids = recent_ids(tokens, ctx);
      sample::apply_frequency_penalty(logits, &ids, p)
    });
    processors.push(proc);
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
fn recent_ids(tokens: &[u32], context_size: usize) -> Vec<i32> {
  // Python `tokens[-context_size:]`: `context_size == 0` ⇒ `tokens[0:]`
  // (full history); otherwise the last `min(context_size, len)` ids.
  let start = if context_size == 0 {
    0
  } else {
    tokens.len().saturating_sub(context_size)
  };
  tokens[start..].iter().map(|&t| t as i32).collect()
}

/// One decode step — the sampled `token` plus the `[V]` log-probability
/// vector over the vocabulary that produced it (mlx-lm
/// `generate_step`'s `yield y.item(), logprobs`).
///
/// Replaces the prior `(u32, Array)` tuple item: mlx-lm uses Python's
/// positional tuple as informal documentation, but Rust callers reading
/// the iterator item shouldn't have to remember tuple-index conventions —
/// the struct is self-documenting and a Rust-idiomatic improvement
/// (prefer idiomatic-Rust ergonomics over verbatim Python mirroring).
///
/// # Back-compat
///
/// This is **not** drop-in source-compatible with the prior tuple item:
/// existing `let (tok, lp) = step?;` call sites must add an explicit
/// `.into()` (`let (tok, lp) = step?.into();`) or pattern-match the
/// struct (`let GenStep { token, logprobs } = step?;`). The break is
/// **intentional** — mlxrs is pre-1.0, and the ergonomics + self-
/// documentation win outweighs a one-line migration per call site. The
/// `From<GenStep> for (u32, Array)` impl below makes that migration
/// mechanical.
#[derive(Debug)]
pub struct GenStep {
  /// The sampled token id (mlx-lm `y.item()`).
  pub token: u32,
  /// The token's `[V]` log-probability vector (mlx-lm
  /// `logprobs.squeeze(0)`), kept lazy.
  pub logprobs: Array,
}

impl From<GenStep> for (u32, Array) {
  fn from(s: GenStep) -> Self {
    (s.token, s.logprobs)
  }
}

/// The architecture-agnostic decode iterator: borrows the model, owns the
/// per-layer KV cache, the running token history, the sampler, and the
/// logits processors. `impl Iterator<Item = Result<GenStep>>` — a 1:1 port
/// of `mlx_lm.generate.generate_step`.
///
/// Construct via [`generate_step`]. The borrow of `&'a M` plus the owned
/// cache means no aliasing (spec §7.5). The iterator **fuses**: after it
/// yields `Err` (a step failed) or finishes (eos / `max_tokens`) every
/// further `next()` is `None` — never a panic, never a poisoned re-entry
/// (spec §4).
pub struct Generator<'a, M: Model> {
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

impl<M: Model> Generator<'_, M> {
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
      self.history.extend_from_slice(input);
      for p in &self.processors {
        logits = p(&self.history, &logits)?;
      }
    }

    // 4. `logprobs = logits - mx.logsumexp(logits, keepdims=True)` — the
    //    exact mlx-lm normalization (all-axes logsumexp, broadcast).
    let lse = ops::reduction::logsumexp(&logits, true)?;
    let logprobs = ops::arithmetic::subtract(&logits, &lse)?;

    // 5. `sampled = sampler(logprobs)` — the make_sampler chain / argmax.
    let mut sampled = (self.sampler)(&logprobs)?;

    // 6. token boundary: the ONLY materialization (mlx-lm `y.item()`).
    //    `argmax` / `categorical` both yield `U32`.
    let token: u32 = sampled.item::<u32>()?;

    // mlx-lm returns `logprobs.squeeze(0)` ⇒ a `[V]` vector. Kept lazy.
    let logprobs = ops::shape::squeeze_axes(&logprobs, &[0])?;
    Ok(GenStep { token, logprobs })
  }
}

impl<M: Model> Iterator for Generator<'_, M> {
  type Item = Result<GenStep>;

  fn next(&mut self) -> Option<Self::Item> {
    // Fused: a prior Err or a finish ends iteration permanently — no
    // panic, no poisoned re-entry into the model / mlx-c (spec §4).
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
      Ok(step) => {
        self.produced += 1;
        self.last = Some(step.token);
        // Spec §3: `generate_step` itself stops on an eos token (it carries
        // the eos set); the eos token IS yielded (mlx-lm yields it, then
        // `stream_generate` breaks) — so yield it, then fuse.
        if self.eos.contains(&step.token) {
          self.done = true;
        }
        Some(Ok(step))
      }
      Err(e) => {
        // A step error is yielded once, then the iterator ends (spec §4).
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
  let row: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
  Array::from_slice::<i32>(&row, &(1usize, row.len()))
}

/// `logits[:, -1, :]` — slice the final sequence position of a `[B, S, V]`
/// logits tensor and drop the (now size-1) sequence axis ⇒ `[B, V]`,
/// matching mlx-lm's `logits[:, -1, :]` (`generate_step` line 407).
///
/// A degenerate (buggy-model) `S == 0` or `V == 0` axis is a
/// **DETERMINISTIC recoverable** `Err(Error::ShapeMismatch)` — the
/// faithful-equivalent of Python `logits[:, -1, :]` raising `IndexError` on
/// a zero-length sequence axis (and the same recoverable-`Err` discipline as
/// the merged KV-cache rank guards). Guarded **before** the `s - 1`
/// last-position index so a zero `S` can never underflow / produce a
/// malformed `[0, -1, 0]` slice start (it stays a clean `Err`, never a
/// panic, so the iterator yields it once then fuses).
fn last_position(logits: &Array) -> Result<Array> {
  let shape = logits.shape();
  if shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!("generate: expected [B, S, V] logits from `forward`, got {shape:?}"),
    });
  }
  // `logits[:, -1, :]` is only defined for a non-empty sequence axis and a
  // non-empty vocab axis; mirror Python's `IndexError` on `S == 0` (the
  // last-position index `s - 1` would underflow / be `-1`) and on `V == 0`
  // (an empty distribution the sampler cannot draw from) as a recoverable
  // `Err` BEFORE any index arithmetic.
  if shape[1] == 0 || shape[2] == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "generate: `forward` returned logits with a zero-length axis (got [B, S, V] {shape:?}); \
         `logits[:, -1, :]` requires S >= 1 and V >= 1"
      ),
    });
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
/// token per step until a sampled token is in [`GenConfig::eos`] (the eos
/// token is the final yielded item) or [`GenConfig::max_tokens`] tokens
/// have been produced. A step error is yielded once as `Err`, after which
/// the iterator ends (spec §4 — no panic, no poison).
pub fn generate_step<'a, M: Model>(
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
  // stays a pure `Iterator` (the spec's A1) and the Iterator-yields-Err
  // contract is the single error channel.
  let built = (|| -> Result<(Sampler, Vec<LogitsProcessor>)> {
    if prompt.is_empty() {
      return Err(Error::ShapeMismatch {
        message: "generate: prompt must be non-empty".into(),
      });
    }
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
      prefilled: false,
      first_step: true,
      pending_err: None,
      done: false,
    },
    Err(e) => Generator {
      model,
      cache,
      // A never-called placeholder sampler; `pending_err` ends the
      // iterator on its first poll before any step runs.
      sampler: Box::new(|_| {
        Err(Error::Backend {
          message: "generate: sampler construction failed".into(),
        })
      }),
      processors: Vec::new(),
      prompt: Vec::new(),
      prefill_offset: 0,
      history: Vec::new(),
      last: None,
      produced: 0,
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

/// The final segment of a generation run — a 1:1 port of mlx-lm's
/// `GenerationResponse` (`generate.py` lines 269-296), restricted to the
/// fields the no-network / single-stream WS-C surface produces.
///
/// Yielded by [`stream_generate`]: `text` is the streaming detokenizer's
/// newly readable segment for this token (possibly empty), `token` /
/// `logprobs` the just-produced step, and `finish_reason` is `None` for
/// intermediate responses, `Some("stop")` when the model emitted an eos
/// token, `Some("length")` when `max_tokens` was reached — exactly mlx-lm's
/// `"stop" if token in tokenizer.eos_token_ids else "length"`.
#[derive(Debug)]
pub struct GenerationResponse {
  /// The next readable text segment (mlx-lm `detokenizer.last_segment`);
  /// may be empty when the detokenizer is still withholding bytes.
  pub text: String,
  /// The token this response carries (mlx-lm `token`).
  pub token: u32,
  /// The token's `[V]` log-probability vector (mlx-lm `logprobs`).
  pub logprobs: Array,
  /// Number of prompt tokens (mlx-lm `prompt_tokens` = `prompt.size`).
  pub prompt_tokens: usize,
  /// Prompt processing tokens-per-second (mlx-lm `prompt_tps`).
  pub prompt_tps: f64,
  /// Number of tokens generated so far (mlx-lm `generation_tokens` = `n +
  /// 1`).
  pub generation_tokens: usize,
  /// Generation tokens-per-second (mlx-lm `generation_tps`).
  pub generation_tps: f64,
  /// `None` while generating; `Some("stop")` on an eos token, `Some(
  /// "length")` at `max_tokens` (mlx-lm `finish_reason`).
  pub finish_reason: Option<String>,
}

/// Stream text from `model` for `prompt` — a 1:1 port of
/// `mlx_lm.generate.stream_generate`.
///
/// Maps [`generate_step`] through the #18 streaming detokenizer
/// ([`crate::tokenizer::Tokenizer::detokenizer`]) into
/// [`GenerationResponse`]s. The eos set is taken from the tokenizer
/// ([`crate::tokenizer::Tokenizer::eos_token_ids`], mlx-lm's
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
pub fn stream_generate<'a, M: Model>(
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
  cfg.eos = tokenizer.eos_token_ids().iter().copied().collect();
  let max_tokens = cfg.max_tokens;
  let eos: Vec<u32> = cfg.eos.clone();

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
    let GenStep { token, logprobs } = match steps.next()? {
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

    // mlx-lm: `if token in eos: break` BEFORE `add_token` ⇒ the eos token
    // is never detokenized; a final `finish_reason="stop"` response with
    // the (empty) finalized tail is yielded.
    if eos.contains(&token) {
      finished = true;
      detok.finalize();
      let text = detok.last_segment();
      return Some(Ok(GenerationResponse {
        text,
        token,
        logprobs,
        prompt_tokens,
        prompt_tps,
        generation_tokens: n + 1,
        generation_tps: gen_tps(n + 1),
        finish_reason: Some("stop".to_string()),
      }));
    }

    detok.add_token(token);
    n += 1;

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
        finish_reason: Some("length".to_string()),
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
      finish_reason: None,
    }))
  })
}

/// Generate a complete response string for `prompt` — a 1:1 port of
/// `mlx_lm.generate.generate` (the non-verbose path): collect every
/// [`stream_generate`] segment into one `String`.
///
/// Any step error is surfaced as `Err` (it short-circuits the collection,
/// exactly the [`stream_generate`] Iterator-`Err` contract).
pub fn generate<M: Model>(
  model: &M,
  tokenizer: &crate::tokenizer::Tokenizer,
  prompt: &[u32],
  cache: Vec<Box<dyn KvCache>>,
  cfg: GenConfig,
) -> Result<String> {
  let mut text = String::new();
  for response in stream_generate(model, tokenizer, prompt, cache, cfg) {
    text.push_str(&response?.text);
  }
  Ok(text)
}
