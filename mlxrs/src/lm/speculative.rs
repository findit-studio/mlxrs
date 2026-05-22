//! Speculative decoding for `mlxrs::lm::generate`, ported 1:1 from
//! [`mlx_lm.generate.speculative_generate_step`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
//! and the `draft_model` branch of `mlx_lm.generate.stream_generate`.
//!
//! Per step (mlx-lm `speculative_generate_step.while True:` loop, lines
//! 611-654):
//!
//! 1. **Draft.** Run the draft model autoregressively for
//!    [`DraftConfig::n_draft_tokens`] steps from the current input `y` (one
//!    `forward` per draft, each advancing the draft cache by 1).
//! 2. **Verify.** Run the target model in ONE forward over the concatenated
//!    `[y, draft_tokens]` window (length `N+1`); slice the last `N+1`
//!    positions to obtain per-position predictions.
//! 3. **Accept.** For each draft `i` in `0..N`, compare the draft's sampled
//!    token to the target's argmax/sample at position `i`; **break on first
//!    mismatch**. All matched drafts are yielded as accepted tokens; then the
//!    target's position-`n` token is yielded as the **bonus** (one new token
//!    per step regardless of acceptance rate).
//! 4. **Rewind caches.** Both caches were over-advanced — target by `N - n`
//!    (it ran `N+1` predictions, kept `n + 1`), draft by `max(N - n - 1, 0)`
//!    (the `- 1` is mlx-lm's quirky bookkeeping for the case `n == N`, where
//!    the next iteration re-feeds the last draft together with the bonus).
//!    Implemented via [`crate::lm::cache::trim_prompt_cache`] (the existing
//!    cache-trim surface — no new trait method needed).
//! 5. **Next iter.** `y = [bonus]`; if `n == N`, prefix `draft_y` with the
//!    last (unfed) draft sample, faithful to mlx-lm lines 642-648.
//!
//! **Acceptance rule (mlx-lm faithful).** mlx-lm checks token equality
//! `tn != dtn` regardless of temperature — the comparison is on the
//! sampled token from each model, not a probability ratio. For greedy
//! decoding (`temp == 0`) this is exactly the "argmax match" test. For
//! stochastic decoding the equality is what mlx-lm uses (a weaker
//! acceptance than the standard Metropolis-style rejection rule, but
//! correct: target's distribution is always the one that decides each
//! yielded token, so the output sequence remains a sample from the target
//! distribution conditioned on the prefix).
//!
//! **Byte-identical to plain `generate` for greedy (temp=0).** When draft is
//! the same model (self-draft) every draft token IS the target's argmax, so
//! every draft is accepted and the bonus is the next argmax — exactly the
//! sequence plain [`crate::lm::generate::generate`] produces. When draft is
//! a different model the accepted prefix may be shorter, but the bonus is
//! still target's argmax and the rejected drafts are discarded — so the
//! yielded sequence remains exactly target's greedy output.
//!
//! **KV cache rollback.** Existing [`crate::lm::cache::KvCache::trim`] + the
//! [`crate::lm::cache::trim_prompt_cache`] helper (the port of mlx-lm
//! `trim_prompt_cache`, `cache.py:95-111`) already provide exactly the
//! rollback semantics the algorithm needs — no new trait method added. The
//! caches the caller passes in MUST be trimmable
//! ([`crate::lm::cache::can_trim_prompt_cache`]); a non-trimmable cache
//! surfaces as an early `Err`, mirroring mlx-lm lines 529-533.

use crate::{
  array::Array,
  error::{Error, Result, try_extend_from_slice, try_with_capacity},
  lm::{
    cache::{KvCache, can_trim_prompt_cache, trim_prompt_cache},
    generate::{GenConfig, GenerationResponse, make_logits_processors, make_sampler},
    model::Model,
  },
  ops,
};

/// Statistics reported alongside a speculative-decoding run.
///
/// Mirrors mlx-lm's per-run accounting (the `verbose=True` path of
/// `mlx_lm.generate.generate` prints prompt/generation tokens-per-second and
/// the speculative `accept_rate = accepted / proposed`). The accept-rate
/// counters track the per-step acceptance:
///
/// - [`Self::proposed_drafts`] — total `n_draft_tokens` proposed across all
///   speculative steps (mlx-lm `num_draft` summed). The per-step bonus token
///   is NOT counted as proposed (it is always emitted, never a proposal that
///   could be rejected).
/// - [`Self::accepted_drafts`] — total drafts whose token matched target
///   (mlx-lm `n` summed at each step's accept loop).
/// - [`Self::generated_tokens`] — total tokens yielded (accepted drafts +
///   bonuses); equals what plain [`crate::lm::generate::generate`] would
///   produce.
///
/// `accept_rate` is the convenience ratio in `[0, 1]` (returns `0.0` if no
/// drafts were ever proposed — e.g. a `n_draft_tokens == 0` config that
/// degenerates to plain greedy decoding).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationStats {
  /// Total drafts proposed across all speculative steps.
  pub proposed_drafts: usize,
  /// Total drafts accepted (the bonus token is NOT a draft).
  pub accepted_drafts: usize,
  /// Total tokens yielded to the caller (accepted drafts + per-step bonuses).
  pub generated_tokens: usize,
}

impl GenerationStats {
  /// `accepted_drafts / proposed_drafts` (or `0.0` when no drafts were
  /// proposed). Always in `[0, 1]`.
  pub fn accept_rate(&self) -> f32 {
    if self.proposed_drafts == 0 {
      0.0
    } else {
      self.accepted_drafts as f32 / self.proposed_drafts as f32
    }
  }
}

/// Configuration for the draft model used in speculative decoding —
/// mirrors mlx-lm's `(draft_model, num_draft_tokens)` parameters of
/// `speculative_generate_step` (`generate.py:473-487`).
pub struct DraftConfig {
  /// The smaller / faster draft model that proposes tokens for the target
  /// to verify (mlx-lm `draft_model`).
  pub draft_model: Box<dyn Model>,
  /// One full speculative step proposes `n_draft_tokens` drafts; the target
  /// then verifies + emits 1 bonus, so each step yields between `1` and
  /// `n_draft_tokens + 1` tokens (mlx-lm `num_draft_tokens`, default `2`).
  ///
  /// `0` degenerates to plain non-speculative decoding (each step yields
  /// just the target's argmax/sample as the bonus). A value larger than
  /// `cfg.max_tokens - generated` is clamped per-step exactly like mlx-lm
  /// (`num_draft = min(max_tokens - ntoks, num_draft_tokens)`,
  /// `generate.py:613`).
  pub n_draft_tokens: usize,
}

// Intentionally no `Default for DraftConfig`: a draft config without a
// `draft_model` is meaningless (there is no sensible empty draft), so
// requiring an explicit constructor at the call site avoids the panicking-
// default antipattern. Construct with `DraftConfig { draft_model: ..,
// n_draft_tokens: .. }` directly.

/// Run speculative decoding to completion and return the assembled text +
/// per-run [`GenerationStats`] (the speculative analogue of plain
/// [`crate::lm::generate::generate`]).
///
/// Collects every [`speculative_stream_generate`] segment into one `String`,
/// returning the final stats. Errors propagate as `Err`, short-circuiting
/// the collection (exactly the [`speculative_stream_generate`]
/// `Iterator<Item = Result<..>>` contract).
pub fn speculative_generate(
  target: &dyn Model,
  tokenizer: &crate::tokenizer::Tokenizer,
  prompt: &[u32],
  target_cache: Vec<Box<dyn KvCache>>,
  draft_cache: Vec<Box<dyn KvCache>>,
  draft_cfg: DraftConfig,
  cfg: GenConfig,
) -> Result<(String, GenerationStats)> {
  let mut text = String::new();
  let mut stats = GenerationStats::default();
  for response in speculative_stream_generate(
    target,
    tokenizer,
    prompt,
    target_cache,
    draft_cache,
    draft_cfg,
    cfg,
  ) {
    let r = response?;
    text.push_str(&r.text);
    stats = r.stats;
  }
  Ok((text, stats))
}

/// One [`speculative_stream_generate`] response: the same per-token
/// [`GenerationResponse`] plain [`crate::lm::generate::stream_generate`]
/// produces, plus the running [`GenerationStats`] snapshot and a flag
/// indicating whether the token came from the draft (mlx-lm's `from_draft`,
/// `generate.py:290`).
#[derive(Debug)]
pub struct SpeculativeResponse {
  /// The per-token [`GenerationResponse`] (text segment, token id, logprobs,
  /// counts, finish_reason).
  pub response: GenerationResponse,
  /// `true` iff this token was generated by the draft model and accepted by
  /// the target; `false` for a bonus token (target's own argmax / sample).
  /// Mirrors mlx-lm `GenerationResponse.from_draft` (`generate.py:290`).
  pub from_draft: bool,
  /// Snapshot of the [`GenerationStats`] **after** this token is included.
  pub stats: GenerationStats,
}

impl SpeculativeResponse {
  /// Convenience: re-emits the inner [`GenerationResponse`]'s text so
  /// `for r in speculative_stream_generate(..) { acc.push_str(&r?.text);
  /// }` works without an extra dereference (matches the plain
  /// [`crate::lm::generate::stream_generate`] ergonomic).
  pub fn text(&self) -> &str {
    &self.response.text
  }
}

impl std::ops::Deref for SpeculativeResponse {
  type Target = GenerationResponse;
  fn deref(&self) -> &Self::Target {
    &self.response
  }
}

/// A streaming speculative-decoding run — the [`Iterator`]
/// [`speculative_stream_generate`] returns.
///
/// Yields one [`SpeculativeResponse`] per yielded token (accepted draft or
/// bonus). Unlike a bare `impl Iterator`, this is a named type so a caller
/// that abandons the stream mid-generation can still finalize the streaming
/// detokenizer via [`SpeculativeStream::finalize_tail`] — the buffered tail
/// a BPE/SPM detokenizer withholds until `finalize()` would otherwise be
/// lost (see that method's doc).
pub struct SpeculativeStream<'a> {
  /// The speculative driver, or `None` once the run is finished / failed at
  /// construction.
  driver: Option<SpeculativeDriver<'a>>,
  /// A construction-time error, deferred so the public surface stays a pure
  /// `Iterator`: yielded once on the first poll, then the iterator fuses.
  /// `Error` is not `Clone`, hence the `take()`-on-first-poll `Option`.
  pending_err: Option<Error>,
  /// The streaming detokenizer — maps yielded token ids to text segments.
  detok: crate::tokenizer::wrapper::BoxedDetokenizer,
  /// Encoded prompt length, surfaced on every [`GenerationResponse`].
  prompt_tokens: usize,
  /// The eos id set (from `tokenizer.eos_token_ids()`): generation ends on
  /// the first eos token.
  eos: Vec<u32>,
  /// `cfg.max_tokens` — the "length" stop.
  max_tokens: usize,
  /// Whether the caller opted into per-position logprobs (`GenConfig`).
  collect_logprobs: bool,
  /// Tokens yielded so far (mlx-lm's `n`).
  n: usize,
  /// `true` once the run has ended (eos / `max_tokens` / `Err`); the
  /// iterator fuses afterwards.
  finished: bool,
  /// Start instant for the prompt-/generation-tps measurement.
  tic: std::time::Instant,
  /// Tokens-per-second over the prompt prefill (set after the first token).
  prompt_tps: f64,
}

impl SpeculativeStream<'_> {
  /// Finalize the streaming detokenizer and return whatever tail it had
  /// withheld — the text that an EOS / `max_tokens` poll *would* have
  /// flushed but that an interrupted run never reaches.
  ///
  /// BPE/SPM detokenizers buffer trailing bytes / spaces and only release
  /// them on `finalize()`. A caller that drops this stream mid-generation
  /// (e.g. [`crate::lm::session::ChatSession`]'s interrupted speculative
  /// turn) must call this to make the recorded text token-complete;
  /// otherwise the last produced token's tail is lost. Idempotent: once the
  /// run finished naturally (eos / `max_tokens` already finalized), or once
  /// this has been called, it returns an empty string.
  pub fn finalize_tail(&mut self) -> String {
    self.detok.finalize();
    self.detok.last_segment()
  }
}

impl Iterator for SpeculativeStream<'_> {
  type Item = Result<SpeculativeResponse>;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(e) = self.pending_err.take() {
      // Construction-time failure: yield once, then fuse.
      self.finished = true;
      return Some(Err(e));
    }
    if self.finished {
      return None;
    }
    let d = self.driver.as_mut()?;

    let TokenOut {
      token,
      logprobs,
      from_draft,
      stats,
    } = match d.next_token() {
      Ok(Some(t)) => t,
      Ok(None) => {
        self.finished = true;
        return None;
      }
      Err(e) => {
        self.finished = true;
        return Some(Err(e));
      }
    };

    if self.n == 0 {
      let prompt_time = self.tic.elapsed().as_secs_f64();
      self.prompt_tps = if prompt_time > 0.0 {
        self.prompt_tokens as f64 / prompt_time
      } else {
        0.0
      };
      self.tic = std::time::Instant::now();
    }

    let dt = self.tic.elapsed().as_secs_f64();
    let gen_tps = |gen_count: usize| -> f64 { if dt > 0.0 { gen_count as f64 / dt } else { 0.0 } };

    // mlx-lm: `if token in eos: break` BEFORE `add_token` — the eos token
    // is yielded with an empty (or only the finalized tail) text segment.
    if self.eos.contains(&token) {
      self.finished = true;
      self.detok.finalize();
      let text = self.detok.last_segment();
      return Some(Ok(SpeculativeResponse {
        response: GenerationResponse {
          text,
          token,
          logprobs: self.collect_logprobs.then_some(logprobs),
          prompt_tokens: self.prompt_tokens,
          prompt_tps: self.prompt_tps,
          generation_tokens: self.n + 1,
          generation_tps: gen_tps(self.n + 1),
          peak_memory_bytes: crate::memory::peak_memory().ok(),
          finish_reason: Some("stop".to_string()),
        },
        from_draft,
        stats,
      }));
    }

    self.detok.add_token(token);
    self.n += 1;

    if self.n >= self.max_tokens {
      self.finished = true;
      self.detok.finalize();
      let text = self.detok.last_segment();
      return Some(Ok(SpeculativeResponse {
        response: GenerationResponse {
          text,
          token,
          logprobs: self.collect_logprobs.then_some(logprobs),
          prompt_tokens: self.prompt_tokens,
          prompt_tps: self.prompt_tps,
          generation_tokens: self.n,
          generation_tps: gen_tps(self.n),
          peak_memory_bytes: crate::memory::peak_memory().ok(),
          finish_reason: Some("length".to_string()),
        },
        from_draft,
        stats,
      }));
    }

    let text = self.detok.last_segment();
    Some(Ok(SpeculativeResponse {
      response: GenerationResponse {
        text,
        token,
        logprobs: self.collect_logprobs.then_some(logprobs),
        prompt_tokens: self.prompt_tokens,
        prompt_tps: self.prompt_tps,
        generation_tokens: self.n,
        generation_tps: gen_tps(self.n),
        peak_memory_bytes: crate::memory::peak_memory().ok(),
        finish_reason: None,
      },
      from_draft,
      stats,
    }))
  }
}

/// Stream speculative-decoded text from `target` for `prompt`, using
/// `draft_cfg.draft_model` to propose tokens — port of the `draft_model`
/// branch of `mlx_lm.generate.stream_generate` (`generate.py:711-746`).
///
/// Returns a [`SpeculativeStream`], an [`Iterator`] yielding one
/// [`SpeculativeResponse`] per yielded token (accepted draft or bonus). The
/// streaming detokenizer + finish-reason wiring is the same as plain
/// [`crate::lm::generate::stream_generate`] — the eos set is taken from
/// `tokenizer.eos_token_ids()` (overriding any `cfg.eos`), so the
/// `finish_reason` matches mlx-lm exactly.
///
/// **Cache trimmability is required.** Both `target_cache` and `draft_cache`
/// must consist of trimmable caches ([`can_trim_prompt_cache`]); a
/// non-trimmable cache surfaces as the iterator's first (and only) `Err`,
/// mirroring mlx-lm's `cache.py:529-533` `ValueError`.
pub fn speculative_stream_generate<'a>(
  target: &'a dyn Model,
  tokenizer: &'a crate::tokenizer::Tokenizer,
  prompt: &[u32],
  target_cache: Vec<Box<dyn KvCache>>,
  draft_cache: Vec<Box<dyn KvCache>>,
  draft_cfg: DraftConfig,
  cfg: GenConfig,
) -> SpeculativeStream<'a> {
  let prompt_tokens = prompt.len();
  // mlx-lm uses the tokenizer's eos set for the break / finish_reason.
  let mut cfg = cfg;
  cfg.eos = tokenizer.eos_token_ids().iter().copied().collect();
  let max_tokens = cfg.max_tokens;
  let eos: Vec<u32> = cfg.eos.clone();
  // L3 opt-in (`GenConfig::collect_logprobs`): the speculative driver always
  // computes the per-position `[V]` log-probs internally (the verification +
  // sampling needs them), but the public `GenerationResponse.logprobs` only
  // exposes that `[V]` view when the caller opted in — `None` otherwise,
  // matching plain `stream_generate`'s opt-in contract.
  let collect_logprobs = cfg.collect_logprobs;
  let prompt: Vec<u32> = prompt.to_vec();

  // Build the core driver up front so any construction error surfaces as the
  // iterator's first (and only) `Err`, keeping the public surface a pure
  // `Iterator`. Cache trimmability is checked here too.
  let (driver, pending_err) =
    match SpeculativeDriver::new(target, draft_cfg, prompt, target_cache, draft_cache, &cfg) {
      Ok(d) => (Some(d), None),
      Err(e) => (None, Some(e)),
    };

  SpeculativeStream {
    driver,
    pending_err,
    detok: tokenizer.detokenizer(),
    prompt_tokens,
    eos,
    max_tokens,
    collect_logprobs,
    n: 0,
    finished: false,
    tic: std::time::Instant::now(),
    prompt_tps: 0.0,
  }
}

/// The per-token output of the speculative driver — accepted draft or
/// bonus, together with the running stats and `from_draft` flag.
struct TokenOut {
  token: u32,
  logprobs: Array,
  from_draft: bool,
  stats: GenerationStats,
}

/// A queued token awaiting yield: the [`TokenOut`] payload (sans final
/// `stats`) plus the per-token `stats` **delta** to apply at yield time.
///
/// Stats are tracked at YIELD time, not at step-build time, so an early
/// EOS that drops un-yielded pending entries does NOT count them as
/// generated / accepted / proposed (per Fix 2). The `delta` carries:
///
/// - `proposed`: how many drafts this yield "covers" (1 per accepted-draft
///   yield, `num_draft - n_accept` on the bonus so the **bonus** yield
///   accounts for any drafts that the verifier saw but the accept loop
///   broke on — rejected proposals are still proposals).
/// - `accepted`: 1 iff this yield is an accepted draft, 0 for the bonus.
/// - `generated`: always 1 (every yielded token is a generated token).
struct PendingToken {
  token: u32,
  logprobs: Array,
  from_draft: bool,
  delta: StatsDelta,
}

#[derive(Debug, Clone, Copy, Default)]
struct StatsDelta {
  proposed: usize,
  accepted: usize,
  generated: usize,
}

/// The speculative-decoding driver: owns both caches, the per-step
/// scratch buffer of accepted tokens awaiting yield, the sampler, and the
/// `y_input` / `draft_y_input` continuation state. Mirrors mlx-lm's
/// `speculative_generate_step` generator state machine.
struct SpeculativeDriver<'a> {
  target: &'a dyn Model,
  draft_model: Box<dyn Model>,
  n_draft_tokens: usize,
  max_tokens: usize,
  /// Target's per-layer KV cache.
  target_cache: Vec<Box<dyn KvCache>>,
  /// Draft's per-layer KV cache.
  draft_cache: Vec<Box<dyn KvCache>>,
  /// Sampler closure: temp 0 ⇒ argmax. We re-call it per position when
  /// verifying drafts (each position's `[V]` logits → 1 token).
  sampler: crate::lm::generate::Sampler,
  /// Logits processors (applied at each step, full history sees them).
  processors: Vec<crate::lm::generate::LogitsProcessor>,
  /// Full token history (prompt + everything yielded so far), fed to
  /// logits-processor closures' `(history, logits)` signature exactly as
  /// plain [`crate::lm::generate::generate_step`] does.
  history: Vec<u32>,
  /// Tokens already yielded.
  produced: usize,
  /// `true` once initial prefill has run.
  prefilled: bool,
  /// Prefill chunk size.
  prefill_step_size: usize,
  /// The full encoded prompt (used for prefill).
  prompt: Vec<u32>,
  /// Current `y` for the target's verification step — initialized to the
  /// post-prefill prompt tail, then `[bonus]` after each step.
  y_input: Vec<u32>,
  /// Current `draft_y` for the draft autoregressive proposals — initialized
  /// to the post-prefill prompt tail, then updated per mlx-lm's
  /// `n == num_draft` special case.
  draft_y_input: Vec<u32>,
  /// Pending tokens to yield (accepted drafts + bonus), drained one per
  /// `next_token` call so each yields a single token to the streaming
  /// detokenizer. Each entry carries a per-token stats **delta** that is
  /// applied at yield time (Fix 2 — early EOS that drops the remaining
  /// entries does NOT count them).
  pending: std::collections::VecDeque<PendingToken>,
  /// Running stats.
  stats: GenerationStats,
  /// `true` once we've exhausted (max_tokens reached / cache failure).
  exhausted: bool,
}

impl<'a> SpeculativeDriver<'a> {
  fn new(
    target: &'a dyn Model,
    draft_cfg: DraftConfig,
    prompt: Vec<u32>,
    target_cache: Vec<Box<dyn KvCache>>,
    draft_cache: Vec<Box<dyn KvCache>>,
    cfg: &GenConfig,
  ) -> Result<Self> {
    // mlx-lm `cache.py:529-533`: speculative decoding REQUIRES trimmable
    // caches (we use `trim_prompt_cache` per step to rewind on partial
    // accept).
    if !can_trim_prompt_cache(&target_cache) {
      return Err(Error::Backend {
        message: "speculative_generate: target_cache must be trimmable (see mlx-lm \
                  generate.py:529-533)"
          .into(),
      });
    }
    if !can_trim_prompt_cache(&draft_cache) {
      return Err(Error::Backend {
        message: "speculative_generate: draft_cache must be trimmable (see mlx-lm \
                  generate.py:529-533)"
          .into(),
      });
    }
    if prompt.is_empty() {
      return Err(Error::ShapeMismatch {
        message: "speculative_generate: prompt must be non-empty".into(),
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
    Ok(Self {
      target,
      draft_model: draft_cfg.draft_model,
      n_draft_tokens: draft_cfg.n_draft_tokens,
      max_tokens: cfg.max_tokens,
      target_cache,
      draft_cache,
      sampler,
      processors,
      history: Vec::new(),
      produced: 0,
      prefilled: false,
      prefill_step_size: cfg.prefill_step_size.max(1),
      prompt,
      y_input: Vec::new(),
      draft_y_input: Vec::new(),
      pending: std::collections::VecDeque::new(),
      stats: GenerationStats::default(),
      exhausted: false,
    })
  }

  /// Run prefill for both caches: feed the first `len - 1` tokens through
  /// the model in `prefill_step_size` chunks (cache filled, logits
  /// discarded). The last token becomes the initial `y_input` /
  /// `draft_y_input` (mlx-lm `_prefill` returns `y` of size 1).
  fn prefill(&mut self) -> Result<()> {
    let mut offset = 0usize;
    while self.prompt.len() - offset > 1 {
      let remaining = (self.prompt.len() - offset) - 1;
      let n = self.prefill_step_size.min(remaining);
      let chunk = token_window(&self.prompt[offset..offset + n])?;
      let _ = self.target.forward(&chunk, &mut self.target_cache)?;
      let _ = self.draft_model.forward(&chunk, &mut self.draft_cache)?;
      offset += n;
    }
    // Tail = the unconsumed final token(s); mlx-lm's `_prefill` keeps
    // `y.size == 1` post-prefill (the last token).
    let tail = self.prompt[offset..].to_vec();
    self.y_input = tail.clone();
    self.draft_y_input = tail;
    self.prefilled = true;
    Ok(())
  }

  /// Pull the next token (or `None` at end). Drains the pending queue from
  /// the prior speculative step before starting a new one. Per Fix 2,
  /// committing the pending entry's [`StatsDelta`] happens HERE (yield
  /// time), not at step-build time — so an early EOS in `stream_generate`
  /// that drops the remaining pending tokens leaves `self.stats` reflecting
  /// only what was actually yielded.
  fn next_token(&mut self) -> Result<Option<TokenOut>> {
    if let Some(t) = self.pending.pop_front() {
      return Ok(Some(self.commit_pending(t)));
    }
    if self.exhausted {
      return Ok(None);
    }
    if self.produced >= self.max_tokens {
      self.exhausted = true;
      return Ok(None);
    }
    if !self.prefilled {
      self.prefill()?;
    }
    self.run_speculative_step()?;
    Ok(self.pending.pop_front().map(|t| self.commit_pending(t)))
  }

  /// Apply the per-token stats delta to the live counters, then return the
  /// [`TokenOut`] with the snapshot AT YIELD time. The bonus's `proposed`
  /// delta accounts for any drafts rejected this step (so the bonus yield
  /// covers `num_draft - n_accept` proposals); accepted-draft yields each
  /// account for one proposal. If EOS interrupts before the bonus yields,
  /// those rejected drafts are NOT counted as proposed — which matches
  /// the "discard unyielded" semantics of the Fix 2 spec.
  fn commit_pending(&mut self, t: PendingToken) -> TokenOut {
    self.stats.proposed_drafts += t.delta.proposed;
    self.stats.accepted_drafts += t.delta.accepted;
    self.stats.generated_tokens += t.delta.generated;
    TokenOut {
      token: t.token,
      logprobs: t.logprobs,
      from_draft: t.from_draft,
      stats: self.stats,
    }
  }

  /// One speculative step: draft, verify, accept-prefix, emit bonus,
  /// rewind. Pushes every yielded token onto `self.pending`.
  ///
  /// **History semantics (Fix 1).** The processor history (`self.history`)
  /// is "tokens FED to the target model up to and including the current
  /// step's `y_input`", exactly matching plain
  /// [`crate::lm::generate`]'s `Generator::history` after the same number
  /// of yielded tokens. Per step, after `n_accept` is known, we advance
  /// `self.history` by `y_input + accepted_drafts[0..n_accept]` (the
  /// tokens FED this step that the verifier kept). We do NOT push the
  /// bonus — it is fed on the NEXT step (when it becomes the new
  /// `y_input`) — and we do NOT push rejected drafts. This makes
  /// speculative output byte-identical to plain `generate` for any
  /// history-sensitive logits-processor (rep / presence / frequency
  /// penalties, logit bias).
  ///
  /// **Stats semantics (Fix 2).** `self.stats` is mutated only when a
  /// pending token is YIELDED (see [`Self::commit_pending`]); this method
  /// just records each yield's [`StatsDelta`] alongside the [`PendingToken`].
  /// An early EOS in `stream_generate` that drops the remaining pending
  /// tokens leaves their deltas un-applied — so the final stats reflect
  /// only what was actually yielded.
  fn run_speculative_step(&mut self) -> Result<()> {
    // mlx-lm `generate.py:613`:
    //   `num_draft = min(max_tokens - ntoks, num_draft_tokens)`
    // Faithful 1:1 clamp — DO NOT reserve a slot for the bonus. When
    // `remaining == 1` we still propose 1 draft; the bonus block then
    // skips because `produced` reaches `max_tokens` after the accept loop
    // (`hit_max == true`), and the pending entries are committed at yield
    // time only (R2's `commit_pending` already drops any over-enqueued
    // pending past the length boundary because `next_token` stops calling
    // `run_speculative_step` once `produced >= max_tokens`). This matches
    // mlx-lm exactly: a final 1-token remainder still gets ONE draft
    // proposal and (for self-draft) ONE accepted draft with
    // `from_draft = true`.
    let remaining = self.max_tokens.saturating_sub(self.produced);
    let num_draft = self.n_draft_tokens.min(remaining);

    // (1) Draft autoregressively for `num_draft` steps.
    let draft_tokens = self.draft_generate(num_draft)?;

    // (2) Verify: target forward on `[y_input, draft_tokens]`, length
    // `y_input.len() + num_draft`. The mlx-lm code only ever uses
    // `y_input.len() == 1` after the first iteration, but on the FIRST
    // iteration it is the full post-prefill tail (typically also length 1).
    // We mirror the structure: feed the concatenation, slice the last
    // `num_draft + 1` positions, compare per-position.
    let mut combined: Vec<u32> = try_with_capacity(self.y_input.len() + draft_tokens.len())?;
    try_extend_from_slice(&mut combined, &self.y_input)?;
    try_extend_from_slice(&mut combined, &draft_tokens)?;
    let n_predict = num_draft + 1; // mlx-lm `num_draft + 1`.
    let combined_arr = token_window(&combined)?;
    let logits = self.target.forward(&combined_arr, &mut self.target_cache)?;
    // mlx-lm `logits[:, -n_predict:, :]` (line 556). Slice the last
    // `n_predict` sequence positions.
    let per_pos_logits = last_n_positions(&logits, n_predict)?;

    // (3) For each position, run logits-processors (if any) over the
    // history AT THAT POSITION and sample.
    //
    // mlx-lm's `_step` for `n_predict > 1` runs processors per position;
    // here we follow the same per-position loop. For greedy (`temp == 0`)
    // and NO processors this is just argmax per row — cheap. For the
    // general case we slice each row, apply processors with the
    // appropriate history, then call the sampler.
    //
    // The per-position `history_snapshot` is a CLONE of `self.history`
    // (the "permanent" history before this step) that we extend by the
    // tokens FED at each position (`combined[pos]`). `self.history`
    // itself is NOT mutated here — we wait until `n_accept` is known
    // (the Fix 1 invariant: only kept-by-verifier tokens get
    // committed).
    let mut target_tokens: Vec<u32> = try_with_capacity(n_predict)?;
    let mut target_logprobs: Vec<Array> = try_with_capacity(n_predict)?;
    let have_procs = !self.processors.is_empty();
    let mut history_snapshot = if have_procs {
      self.history.clone()
    } else {
      Vec::new()
    };
    for pos in 0..n_predict {
      // Slice `[B, V]` (single position) — drop the seq axis.
      let row = slice_position(&per_pos_logits, pos as i32)?;
      // Apply processors (mlx-lm's per-position history).
      let mut row = row;
      if have_procs {
        // History grows by `combined[pos]` for this position's input.
        try_extend_from_slice(&mut history_snapshot, &combined[pos..pos + 1])?;
        for p in &self.processors {
          row = p(&history_snapshot, &row)?;
        }
      }
      // `logprobs = row - logsumexp(row)` (the exact mlx-lm
      // normalization).
      let lse = ops::reduction::logsumexp(&row, true)?;
      let logprobs = ops::arithmetic::subtract(&row, &lse)?;
      // Sample (temp 0 ⇒ argmax).
      let mut sampled = (self.sampler)(&logprobs)?;
      let tok = sampled.item::<u32>()?;
      target_tokens.push(tok);
      target_logprobs.push(ops::shape::squeeze_axes(&logprobs, &[0])?);
    }

    // (4) Accept loop (mlx-lm lines 622-634): walk drafts, break on first
    // mismatch, then emit bonus. Per Fix 2, we push `PendingToken`s with
    // a `StatsDelta` but DO NOT mutate `self.stats` here — the delta is
    // applied at yield time in [`Self::commit_pending`].
    let mut n_accept = 0usize;
    let mut hit_max = false;
    for i in 0..num_draft {
      let t_n = target_tokens[i];
      let d_n = draft_tokens[i];
      if t_n != d_n {
        break;
      }
      n_accept += 1;
      self.produced += 1;
      // Take the logprobs by index (avoid clone — drain).
      let lp = std::mem::replace(&mut target_logprobs[i], empty_logprobs()?);
      // Each accepted-draft yield "covers" one proposal (the bonus's
      // delta picks up the remaining `num_draft - n_accept` rejected
      // proposals — see below). Per Fix 2, a yield that never happens
      // does not count toward `proposed_drafts`.
      self.pending.push_back(PendingToken {
        token: t_n,
        logprobs: lp,
        from_draft: true,
        delta: StatsDelta {
          proposed: 1,
          accepted: 1,
          generated: 1,
        },
      });
      if self.produced >= self.max_tokens {
        hit_max = true;
        break;
      }
    }

    if !hit_max && self.produced < self.max_tokens {
      // Emit bonus: target's prediction at position `n_accept`.
      let bonus = target_tokens[n_accept];
      let bonus_lp = std::mem::replace(&mut target_logprobs[n_accept], empty_logprobs()?);
      self.produced += 1;
      // The bonus yield "covers" the `num_draft - n_accept` proposals
      // that the verifier saw but the accept loop broke on (rejected
      // proposals). Together with the per-accepted-draft `proposed: 1`,
      // a fully-yielded step accounts for exactly `num_draft` proposals.
      self.pending.push_back(PendingToken {
        token: bonus,
        logprobs: bonus_lp,
        from_draft: false,
        delta: StatsDelta {
          proposed: num_draft - n_accept,
          accepted: 0,
          generated: 1,
        },
      });
    }

    // (Fix 1) Permanent history update: advance `self.history` by tokens
    // FED this step that the verifier kept — `y_input` (the current
    // input) plus `accepted_drafts[0..n_accept]` (the drafts that target
    // verification accepted). Do NOT push the bonus (it gets fed on the
    // next step as the new `y_input`); do NOT push rejected drafts.
    // `combined[0..1 + n_accept]` == `[y_input, accepted_drafts]` since
    // an accepted draft equals its target sample by construction.
    let committed_len = self.y_input.len() + n_accept;
    try_extend_from_slice(&mut self.history, &combined[..committed_len])?;

    // (5) Rewind caches (mlx-lm `_rewind_cache`, lines 589-591):
    //   target trim: `num_draft - n_accept`
    //   draft trim:  `max(num_draft - n_accept - 1, 0)`
    // Both trims are no-ops when `num_draft == 0`.
    let target_trim = num_draft - n_accept;
    let draft_trim = (num_draft.saturating_sub(n_accept)).saturating_sub(1);
    if target_trim > 0 {
      trim_prompt_cache(&mut self.target_cache, target_trim)?;
    }
    if draft_trim > 0 {
      trim_prompt_cache(&mut self.draft_cache, draft_trim)?;
    }

    // (6) Set up next iter (mlx-lm lines 639-648):
    //   y_input = [bonus]
    //   draft_y = [bonus]; if n_accept == num_draft: draft_y = [last_draft, bonus]
    if self.produced < self.max_tokens && !hit_max {
      let bonus = self.pending.back().map(|p| p.token).expect("bonus pending");
      self.y_input = vec![bonus];
      if num_draft > 0 && n_accept == num_draft {
        // All accepted: the last draft was sampled but NEVER fed to draft;
        // include it so draft's cache catches up next step.
        let mut d = try_with_capacity(2)?;
        d.push(draft_tokens[num_draft - 1]);
        d.push(bonus);
        self.draft_y_input = d;
      } else {
        self.draft_y_input = vec![bonus];
      }
    } else {
      self.exhausted = true;
    }

    Ok(())
  }

  /// Draft model autoregressively samples `num_draft` tokens, advancing
  /// `draft_cache`. Returns the drafted token ids (in order). Uses the same
  /// sampler / processors as the target (mlx-lm `_step(draft_model,
  /// draft_cache, y)`, same per-call sampler).
  ///
  /// **Draft processor history (Fix 1).** The draft's `draft_history`
  /// snapshot starts as `self.history.clone()` (the permanent history
  /// after the previous step's commit), then grows by exactly ONE token
  /// per draft iteration — the most recently committed yielded token. At
  /// iter 0 this is the LAST element of `draft_y_input` (the bonus from
  /// the previous step); at iter `k > 0` it is the draft sampled in iter
  /// `k - 1`. We extend by `draft_y_input.last()` (not the full window) in
  /// the all-accepted case because `draft_y_input = [last_draft, bonus]`
  /// there — `last_draft` is ALREADY in `self.history` (it was the last
  /// accepted draft of the previous step), so extending the snapshot by
  /// it would DOUBLE-count. The first FED token here is just the bonus
  /// (the one that's new vs `self.history`).
  fn draft_generate(&mut self, num_draft: usize) -> Result<Vec<u32>> {
    if num_draft == 0 {
      return Ok(Vec::new());
    }
    let mut drafts = try_with_capacity(num_draft)?;
    let mut y = self.draft_y_input.clone();
    let have_procs = !self.processors.is_empty();
    let mut draft_history = if have_procs {
      self.history.clone()
    } else {
      Vec::new()
    };
    // The "next token to extend draft_history by" — at iter 0 the FED
    // input window is `draft_y_input`, but only its LAST element is the
    // newly committed token (the rest is the previous step's
    // last-accepted draft already in `self.history`). At subsequent
    // iters it is the prior iter's sampled draft.
    let mut next_history_token: u32 = *y.last().ok_or_else(|| Error::ShapeMismatch {
      message: "speculative_generate: draft_y_input must be non-empty".into(),
    })?;
    for _ in 0..num_draft {
      let arr = token_window(&y)?;
      let logits = self.draft_model.forward(&arr, &mut self.draft_cache)?;
      // mlx-lm `logits[:, -1:, :]` then squeeze → `[B, V]`. We use the
      // same `last_n_positions` then per-position slice for symmetry.
      let last = last_n_positions(&logits, 1)?;
      let mut row = slice_position(&last, 0)?;
      if have_procs {
        draft_history.push(next_history_token);
        for p in &self.processors {
          row = p(&draft_history, &row)?;
        }
      }
      let lse = ops::reduction::logsumexp(&row, true)?;
      let lp = ops::arithmetic::subtract(&row, &lse)?;
      let mut sampled = (self.sampler)(&lp)?;
      let tok = sampled.item::<u32>()?;
      drafts.push(tok);
      y = vec![tok];
      next_history_token = tok;
    }
    Ok(drafts)
  }
}

/// Build a `[1, S]` int token window (the model's `forward` input shape).
/// Duplicate of [`crate::lm::generate`]'s private `token_window` — kept
/// here to keep speculative independent of that file's internals.
fn token_window(ids: &[u32]) -> Result<Array> {
  let mut row: Vec<i32> = try_with_capacity(ids.len())?;
  row.extend(ids.iter().map(|&t| t as i32));
  Array::from_slice::<i32>(&row, &(1usize, row.len()))
}

/// Slice the last `n` positions of `[B, S, V]` logits ⇒ `[B, n, V]`,
/// mlx-lm's `logits[:, -n:, :]` (`speculative_generate_step._step` line
/// 556). Recoverable `Err` on a degenerate shape (S == 0, V == 0, or n > S).
fn last_n_positions(logits: &Array, n: usize) -> Result<Array> {
  let shape = logits.shape();
  if shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "speculative_generate: expected [B, S, V] logits from `forward`, got {shape:?}"
      ),
    });
  }
  if shape[1] == 0 || shape[2] == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "speculative_generate: `forward` returned logits with a zero-length axis (got [B, S, V] \
         {shape:?}); slicing the last `n` positions requires S >= 1 and V >= 1"
      ),
    });
  }
  if n == 0 || n > shape[1] {
    return Err(Error::ShapeMismatch {
      message: format!(
        "speculative_generate: cannot slice last {n} positions from logits with S = {}",
        shape[1]
      ),
    });
  }
  let (b, s, v) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
  let start = s - n as i32;
  ops::indexing::slice(logits, &[0, start, 0], &[b, s, v], &[1, 1, 1])
}

/// Slice one sequence position from `[B, S, V]` logits ⇒ `[B, V]`, the
/// per-position `[1, V]` row the sampler operates on. `pos` is the
/// 0-based position into the S axis.
fn slice_position(logits: &Array, pos: i32) -> Result<Array> {
  let shape = logits.shape();
  if shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!("slice_position: expected [B, S, V], got {shape:?}"),
    });
  }
  let s = shape[1] as i32;
  if pos < 0 || pos >= s {
    return Err(Error::ShapeMismatch {
      message: format!("slice_position: pos {pos} out of range for S = {s}"),
    });
  }
  let (b, v) = (shape[0] as i32, shape[2] as i32);
  let sliced = ops::indexing::slice(logits, &[0, pos, 0], &[b, pos + 1, v], &[1, 1, 1])?;
  ops::shape::squeeze_axes(&sliced, &[1])
}

/// A placeholder `[0]` logprobs array used to `mem::replace` a yielded
/// logprobs slot without cloning the real array. The placeholder is never
/// observed — the original is yielded to the caller.
fn empty_logprobs() -> Result<Array> {
  Array::from_slice::<f32>(&[], &(0usize,))
}
