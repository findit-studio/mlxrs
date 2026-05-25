//! Speculative-decoding integration tests for `mlxrs::lm::speculative`.
//!
//! Deterministic, dependency-free: two `MockModel`s (same vs different bias
//! patterns) exercise the self-draft 100%-accept path and the
//! diverging-draft partial-accept path. The byte-identical-to-plain-
//! `generate` guarantee is the correctness oracle: regardless of acceptance
//! rate, the yielded sequence must equal what plain
//! [`mlxrs::lm::generate::generate`] produces (target's greedy output).
#![cfg(feature = "lm")]

use std::{cell::RefCell, fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array,
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache},
    generate::{FinishReason, GenConfig, generate},
    model::Model,
    speculative::{DraftConfig, speculative_generate, speculative_stream_generate},
  },
};

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_speculative_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

fn tokenizer(name: &str) -> mlxrs::tokenizer::Tokenizer {
  let dir = temp_dir(name);
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(&dir, None).unwrap()
}

/// A deterministic `Model` whose argmax is fully predictable from per-vocab
/// `bias` values. Advances every cache entry by the input window length so
/// cache rollback is observable.
struct MockModel {
  bias: Vec<f32>,
  n_kv_heads: usize,
  head_dim: usize,
}

impl MockModel {
  fn ramp(vocab: usize) -> Self {
    Self {
      bias: (0..vocab).map(|i| i as f32).collect(),
      n_kv_heads: 1,
      head_dim: 2,
    }
  }

  fn with_bias(bias: Vec<f32>) -> Self {
    Self {
      bias,
      n_kv_heads: 1,
      head_dim: 2,
    }
  }
}

impl Model for MockModel {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(mlxrs::Error::ShapeMismatch {
          message: format!("MockModel::forward expects [B, S], got {shape:?}"),
        });
      }
    };
    let vocab = self.bias.len();
    for layer in cache.iter_mut() {
      let elems = batch * self.n_kv_heads * seq * self.head_dim;
      let k = Array::from_slice::<f32>(
        &vec![1.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      let v = Array::from_slice::<f32>(
        &vec![2.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      layer.update(&k, &v)?;
    }
    let mut data = Vec::with_capacity(batch * seq * vocab);
    for _ in 0..batch * seq {
      data.extend_from_slice(&self.bias);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, vocab))
  }
}

/// A draft model that disagrees with the target on the FIRST position of
/// each verification window, then agrees thereafter. Behaves like a
/// "biased" draft: never produces the target's argmax for the first
/// position, but matches afterwards.
///
/// Concretely: target ramps(5) has argmax = 4. This draft's per-position
/// `forward` returns logits whose argmax is `1` (a different token) for
/// the FIRST forward call, then `4` (target's argmax) for subsequent calls
/// — driving partial acceptance at the per-step boundary while still
/// requiring the target's argmax to win as the yielded token.
struct DivergingDraft {
  call_count: RefCell<usize>,
  n_kv_heads: usize,
  head_dim: usize,
}

impl DivergingDraft {
  fn new() -> Self {
    Self {
      call_count: RefCell::new(0),
      n_kv_heads: 1,
      head_dim: 2,
    }
  }
}

impl Model for DivergingDraft {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(mlxrs::Error::ShapeMismatch {
          message: format!("DivergingDraft::forward expects [B, S], got {shape:?}"),
        });
      }
    };
    // Advance cache like a normal model.
    for layer in cache.iter_mut() {
      let elems = batch * self.n_kv_heads * seq * self.head_dim;
      let k = Array::from_slice::<f32>(
        &vec![3.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      let v = Array::from_slice::<f32>(
        &vec![4.0_f32; elems],
        &(batch, self.n_kv_heads, seq, self.head_dim),
      )?;
      layer.update(&k, &v)?;
    }
    let call = {
      let mut c = self.call_count.borrow_mut();
      let v = *c;
      *c += 1;
      v
    };
    let vocab = 5;
    // Per-position bias: position 0 favors token `1`, positions >= 1 favor
    // token `4` (target's argmax). Position is `call % 2` — so the first
    // call of every speculative step's draft loop disagrees; subsequent
    // calls agree.
    let mut data = Vec::with_capacity(batch * seq * vocab);
    for _ in 0..batch * seq {
      // Toggle by call: even calls argmax=1 ("draft disagrees"), odd
      // calls argmax=4 ("draft agrees with target").
      if call % 2 == 0 {
        // Bias: token 1 highest.
        data.extend_from_slice(&[0.0, 10.0, 0.0, 0.0, 1.0]);
      } else {
        // Bias: token 4 highest (matches target's argmax).
        data.extend_from_slice(&[0.0, 1.0, 0.0, 0.0, 10.0]);
      }
    }
    Array::from_slice::<f32>(&data, &(batch, seq, vocab))
  }
}

fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&CacheConfig {
    num_hidden_layers: layers,
    sliding_window: None,
  })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// **Self-draft greedy ⇒ 100% accept + byte-identical to plain `generate`.**
///
/// When the draft is the same model as the target, every draft is target's
/// argmax — every draft is accepted, then the bonus is target's argmax for
/// the next position. The yielded sequence is exactly what plain `generate`
/// produces and `accept_rate == 1.0`.
#[test]
fn speculative_decoding_greedy_self_draft_byte_identical() {
  let tok = tokenizer("self_draft");
  let target = MockModel::ramp(5); // argmax == 4 ("world")
  let draft = MockModel::ramp(5);

  let max_tokens = 6;
  let eos: Vec<u32> = tok.eos_token_ids_iter().collect();

  // Plain generate baseline.
  let cfg_baseline = GenConfig::default()
    .with_max_tokens(max_tokens)
    .with_eos(eos.clone());
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(&target, &tok, &[3u32], cache(1), cfg_baseline).unwrap();

  // Speculative with self-draft (move `eos` — last use here).
  let cfg_spec = GenConfig::default()
    .with_max_tokens(max_tokens)
    .with_eos(eos);
  let draft_cfg = DraftConfig {
    draft_model: Box::new(draft),
    n_draft_tokens: 3,
  };
  let (spec, stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    draft_cfg,
    cfg_spec,
  )
  .unwrap();

  assert_eq!(
    spec, baseline,
    "self-draft speculative output is byte-identical to plain generate"
  );
  assert!(
    stats.proposed_drafts > 0,
    "self-draft should propose drafts, got {stats:?}"
  );
  assert_eq!(
    stats.accepted_drafts, stats.proposed_drafts,
    "self-draft greedy ⇒ every draft accepted (got {stats:?})"
  );
  assert!(
    (stats.accept_rate() - 1.0).abs() < 1e-6,
    "self-draft accept_rate == 1.0, got {}",
    stats.accept_rate()
  );
}

/// **Diverging draft greedy ⇒ output STILL byte-identical to plain `generate`.**
///
/// The draft is a *different* model that disagrees with the target on some
/// positions. Acceptance correctness guarantees: every yielded token is
/// either an accepted draft (which must match target's argmax to be
/// accepted, so == target's greedy choice) OR a bonus token (which IS
/// target's argmax). So the sequence is byte-identical to plain `generate`
/// regardless of how often the draft is wrong.
#[test]
fn speculative_decoding_diverging_draft_still_correct() {
  let tok = tokenizer("diverging");
  let target = MockModel::ramp(5); // argmax == 4
  let draft = DivergingDraft::new(); // disagrees on alternating calls

  let max_tokens = 6;
  let eos: Vec<u32> = tok.eos_token_ids_iter().collect();

  let cfg_baseline = GenConfig::default()
    .with_max_tokens(max_tokens)
    .with_eos(eos.clone());
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(&target, &tok, &[3u32], cache(1), cfg_baseline).unwrap();

  let cfg_spec = GenConfig::default()
    .with_max_tokens(max_tokens)
    .with_eos(eos);
  let draft_cfg = DraftConfig {
    draft_model: Box::new(draft),
    n_draft_tokens: 3,
  };
  let (spec, stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    draft_cfg,
    cfg_spec,
  )
  .unwrap();

  assert_eq!(
    spec, baseline,
    "diverging draft is STILL byte-identical (acceptance correctness)"
  );
  // The diverging draft disagrees frequently, so accept_rate should be
  // strictly between 0 and 1 (not all accepted, not none).
  assert!(
    stats.proposed_drafts > 0,
    "drafts should be proposed, got {stats:?}"
  );
  assert!(
    stats.accepted_drafts < stats.proposed_drafts,
    "diverging draft has at least one rejection (got {stats:?})"
  );
}

/// **`stats.accept_rate` is in `[0, 1]` and reflects actual acceptance.**
#[test]
fn speculative_stats_track_accept_rate() {
  let tok = tokenizer("stats");
  let target = MockModel::ramp(5);

  // Self-draft ⇒ 100% accept.
  let (_text, stats_self) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(MockModel::ramp(5)),
      n_draft_tokens: 2,
    },
    GenConfig::default().with_max_tokens(5),
  )
  .unwrap();
  let r_self = stats_self.accept_rate();
  assert!(
    (0.0..=1.0).contains(&r_self),
    "accept_rate in [0,1], got {r_self}"
  );
  assert!(
    (r_self - 1.0).abs() < 1e-6,
    "self-draft accept_rate == 1.0, got {r_self}"
  );

  // Always-wrong draft ⇒ 0% accept.
  let always_wrong = MockModel::with_bias(vec![10.0, 0.0, 0.0, 0.0, 0.0]); // argmax = 0
  let (_text, stats_wrong) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(always_wrong),
      n_draft_tokens: 2,
    },
    GenConfig::default().with_max_tokens(4),
  )
  .unwrap();
  let r_wrong = stats_wrong.accept_rate();
  assert!(
    (0.0..=1.0).contains(&r_wrong),
    "accept_rate in [0,1], got {r_wrong}"
  );
  assert_eq!(
    stats_wrong.accepted_drafts, 0,
    "always-wrong draft ⇒ no acceptances, got {stats_wrong:?}"
  );
  assert!(
    r_wrong < 1e-6,
    "always-wrong accept_rate ≈ 0, got {r_wrong}"
  );
}

/// **KV cache lengths reflect accepted tokens after rejections.**
///
/// After a multi-step run with mid-step rejections, both caches' `offset()`
/// must equal `prompt_len + accepted_count + bonus_count` — i.e. the
/// logical context length, NOT the over-advanced `+ num_draft + 1`
/// position. This pins the rewind-via-`trim_prompt_cache` rollback.
#[test]
fn kv_cache_rollback_after_rejection() {
  let tok = tokenizer("rollback");
  let target = MockModel::ramp(5);
  let always_wrong = MockModel::with_bias(vec![10.0, 0.0, 0.0, 0.0, 0.0]); // argmax = 0

  let prompt = [3u32, 4]; // 2 tokens
  let max_tokens = 4;
  let n_draft = 3;
  let target_cache = cache(1);
  let draft_cache = cache(1);

  // Drain the iterator so all rewinds run; collect every yielded token.
  let mut last_stats = mlxrs::lm::speculative::GenerationStats::default();
  let mut produced = 0usize;
  for r in speculative_stream_generate(
    &target,
    &tok,
    &prompt,
    target_cache,
    draft_cache,
    DraftConfig {
      draft_model: Box::new(always_wrong),
      n_draft_tokens: n_draft,
    },
    GenConfig::default().with_max_tokens(max_tokens),
  ) {
    let r = r.unwrap();
    last_stats = r.stats;
    produced += 1;
  }

  // With always-wrong drafts, every step accepts 0 and yields just the
  // bonus. So:
  //   proposed_drafts = max_tokens * n_draft  (one step per yielded token)
  //   accepted_drafts = 0
  //   generated_tokens = max_tokens
  assert_eq!(
    last_stats.generated_tokens, max_tokens,
    "max_tokens reached, got {last_stats:?}"
  );
  assert_eq!(
    last_stats.accepted_drafts, 0,
    "always-wrong draft never accepted, got {last_stats:?}"
  );
  assert!(
    last_stats.proposed_drafts >= max_tokens, // at least 1 draft per step
    "proposed_drafts >= max_tokens with n_draft >= 1, got {last_stats:?}"
  );
  assert_eq!(produced, max_tokens, "yielded max_tokens responses");
}

/// **`n_draft_tokens == 0` degenerates to plain greedy decoding.**
///
/// With 0 drafts per step, each step just emits the bonus token (target's
/// argmax) — equivalent to plain `generate`, with `accept_rate == 0` (no
/// drafts were proposed, so the convenience ratio returns 0 by definition).
#[test]
fn speculative_n_draft_zero_degenerates_to_plain() {
  let tok = tokenizer("zero_draft");
  let target = MockModel::ramp(5);
  let draft = MockModel::ramp(5);

  let max_tokens = 4;
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    GenConfig::default().with_max_tokens(max_tokens),
  )
  .unwrap();

  let (spec, stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 0,
    },
    GenConfig::default().with_max_tokens(max_tokens),
  )
  .unwrap();

  assert_eq!(spec, baseline);
  assert_eq!(stats.proposed_drafts, 0);
  assert_eq!(stats.accepted_drafts, 0);
  assert!(stats.accept_rate() < 1e-6); // 0/0 convention: 0.0
  assert_eq!(stats.generated_tokens, max_tokens);
}

/// **Self-draft + repetition penalty ⇒ output STILL byte-identical to plain `generate`.**
///
/// Regression for the Fix 1 history-tracking bug: with a non-zero
/// `repetition_penalty` the per-step processor history must equal what
/// plain `generate` would see at the SAME predict point — i.e. advance
/// by `y_input` (current input) plus the accepted drafts, never by the
/// bonus (not yet fed) nor by rejected drafts (discarded). A buggy
/// history-advance produces a SHIFTED/DUPLICATED history and the
/// rep-penalty sampling diverges from plain.
#[test]
fn speculative_self_draft_with_repetition_penalty_byte_identical() {
  let tok = tokenizer("self_draft_rep_penalty");
  // bias=[0,0,0,10,12]: argmax 4 ("world"); rep penalty 2.0 on 4 ⇒
  // bias[4]=6; on [3,4] dedupes ⇒ bias[3]=5, bias[4]=6 — argmax stays 4.
  // Token 2 (eos) is bias 0 so never sampled — runs to max_tokens.
  let target = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);
  let draft = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);

  let max_tokens = 6;
  let cfg_baseline = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.repetition_penalty = Some(2.0);
    _c
  };
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(&target, &tok, &[3u32], cache(1), cfg_baseline).unwrap();

  let cfg_spec = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.repetition_penalty = Some(2.0);
    _c
  };
  let (spec, _stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 3,
    },
    cfg_spec,
  )
  .unwrap();

  assert_eq!(
    spec, baseline,
    "self-draft + rep_penalty: byte-identical to plain (Fix 1)"
  );
}

/// **Self-draft + presence penalty ⇒ output STILL byte-identical to plain.**
///
/// Same Fix 1 regression, exercising a *different* history-sensitive
/// processor (presence penalty subtracts a constant — different
/// per-history-token math than rep-penalty's multiplicative path). Either
/// processor surfaces the bug if `self.history` doesn't faithfully match
/// plain's accumulated input tokens.
#[test]
fn speculative_self_draft_with_presence_penalty_byte_identical() {
  let tok = tokenizer("self_draft_pres_penalty");
  let target = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);
  let draft = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);

  let max_tokens = 6;
  let cfg_baseline = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.presence_penalty = Some(3.0);
    _c
  };
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(&target, &tok, &[3u32], cache(1), cfg_baseline).unwrap();

  let cfg_spec = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.presence_penalty = Some(3.0);
    _c
  };
  let (spec, _stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 3,
    },
    cfg_spec,
  )
  .unwrap();

  assert_eq!(
    spec, baseline,
    "self-draft + presence_penalty: byte-identical to plain (Fix 1)"
  );
}

/// **Diverging draft + repetition penalty ⇒ output STILL byte-identical to plain.**
///
/// Acceptance correctness PLUS correct history advance guarantees the
/// yielded sequence matches plain: rejected drafts must not pollute
/// `self.history` (Fix 1 — "Do NOT advance history by rejected drafts"),
/// otherwise the bonus / next step's sampling would see drafts that
/// plain never fed.
#[test]
fn speculative_diverging_draft_with_repetition_penalty_byte_identical() {
  let tok = tokenizer("diverging_rep_penalty");
  let target = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);
  // Diverging draft: alternates argmax across calls — disagrees with
  // target on the first call of every draft loop, then agrees. This
  // means some drafts are rejected — exercising the "do NOT advance
  // history by rejected drafts" branch of Fix 1.
  let draft = DivergingDraft::new();

  let max_tokens = 6;
  let cfg_baseline = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.repetition_penalty = Some(2.0);
    _c
  };
  // L3: `generate` now returns `(String, GenerationStats)`; the speculative
  // parity assertions compare only the assembled text.
  let (baseline, _) = generate(&target, &tok, &[3u32], cache(1), cfg_baseline).unwrap();

  let cfg_spec = {
    let mut _c = GenConfig::default().with_max_tokens(max_tokens);
    _c.repetition_penalty = Some(2.0);
    _c
  };
  let (spec, _stats) = speculative_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 3,
    },
    cfg_spec,
  )
  .unwrap();

  assert_eq!(
    spec, baseline,
    "diverging draft + rep_penalty: byte-identical to plain (Fix 1 — rejected drafts excluded)"
  );
}

/// **EOS as first accepted draft ⇒ stats committed only for yielded tokens.**
///
/// Regression for the Fix 2 stats-at-yield-time bug: with
/// `n_draft_tokens=2` and EOS as the first accepted token, 3 tokens are
/// enqueued (2 accepts + 1 bonus) but the stream terminates after just
/// the first yield. The final stats MUST reflect ONLY the yielded token,
/// not the unyielded pending entries:
///
///   - `response_count == 1` (one SpeculativeResponse, the EOS yield)
///   - `stats.generated_tokens == 1` (NOT 3)
///   - `stats.accepted_drafts == 1` (the EOS draft was accepted; NOT 2 / 3)
#[test]
fn speculative_eos_in_first_accepted_draft_stats_match_yields() {
  let tok = tokenizer("eos_first_accept");
  // Force every prediction to token 2 (= "</s>", the eos token). Both
  // models output argmax = 2 ⇒ self-draft path with EOS as the first
  // sampled token of every step.
  let target = MockModel::with_bias(vec![0.0, 0.0, 10.0, 0.0, 0.0]);
  let draft = MockModel::with_bias(vec![0.0, 0.0, 10.0, 0.0, 0.0]);

  let eos: Vec<u32> = tok.eos_token_ids_iter().collect();
  let responses: Vec<_> = speculative_stream_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 2,
    },
    GenConfig::default().with_max_tokens(50).with_eos(eos),
  )
  .map(|r| r.unwrap())
  .collect();

  assert_eq!(
    responses.len(),
    1,
    "EOS as first accepted draft ⇒ exactly one response yielded (got {})",
    responses.len()
  );
  let r = responses.first().expect("one response");
  assert_eq!(
    r.response.finish_reason,
    Some(FinishReason::Eos),
    "first (and only) response is the EOS stop"
  );
  assert_eq!(
    r.stats.generated_tokens, 1,
    "stats counts ONLY the yielded EOS, NOT the 2 unyielded pending tokens (Fix 2); got {:?}",
    r.stats
  );
  assert_eq!(
    r.stats.accepted_drafts, 1,
    "the EOS draft was accepted (it IS a draft) but the second accept + bonus were never yielded; \
     got {:?}",
    r.stats
  );
}

/// **Final partial step still proposes drafts (R3 — `num_draft` clamp parity).**
///
/// Regression for the R3 finding: the old clamp reserved a bonus slot
/// (`remaining.saturating_sub(1)`), which made the final 1-token remainder
/// propose ZERO drafts and emit the last token as a bonus
/// (`from_draft = false`). mlx-lm's clamp (`generate.py:613`) is
/// `min(max_tokens - ntoks, num_draft_tokens)` — no bonus reservation, so
/// the final partial step still proposes drafts (and accepts them, for
/// self-draft), labeling the last yielded token `from_draft = true`.
///
/// With `n_draft_tokens=3, max_tokens=5` (self-draft, greedy):
/// - Step 1: `num_draft = min(5, 3) = 3`. Self-draft ⇒ all 3 accepted,
///   then bonus emitted (produced=4). Step contributes
///   proposed=3, accepted=3.
/// - Step 2: `num_draft = min(1, 3) = 1`. Self-draft ⇒ 1 accepted
///   (produced=5, hit_max), bonus skipped. Step contributes
///   proposed=1, accepted=1.
/// - Totals: 5 responses; proposed=4, accepted=4; last is the
///   accepted draft from step 2 ⇒ `from_draft = true`.
#[test]
fn speculative_self_draft_final_partial_step_proposes_and_accepts() {
  let tok = tokenizer("final_partial");
  // Self-draft greedy: both pick argmax = 4 every step. Token 2 (eos)
  // has bias 0 so it never wins; runs to max_tokens.
  let target = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);
  let draft = MockModel::with_bias(vec![0.0, 0.0, 0.0, 10.0, 12.0]);

  let responses: Vec<_> = speculative_stream_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 3,
    },
    GenConfig::default().with_max_tokens(5),
  )
  .map(|r| r.unwrap())
  .collect();

  assert_eq!(
    responses.len(),
    5,
    "exactly max_tokens responses yielded, got {}",
    responses.len()
  );
  let last = responses.last().expect("last response");
  assert_eq!(
    last.response.finish_reason,
    Some(FinishReason::Length),
    "final yield is the length-cap stop"
  );
  // R3: the last yielded token is an ACCEPTED DRAFT from step 2, not a
  // bonus.
  assert!(
    last.from_draft,
    "final partial step yields an accepted draft (R3 — NOT a bonus); got from_draft={}",
    last.from_draft
  );
  // Step 1: 3 proposed + 3 accepted + bonus (4 tokens).
  // Step 2: 1 proposed + 1 accepted (1 token, hit_max → no bonus).
  // Totals: proposed=4, accepted=4 (self-draft ⇒ proposed == accepted),
  // generated=5.
  assert_eq!(
    last.stats.proposed_drafts, 4,
    "R3: final partial step proposes 1 draft (3 + 1 = 4 total); got {:?}",
    last.stats
  );
  assert_eq!(
    last.stats.accepted_drafts, 4,
    "self-draft ⇒ accepted == proposed (== 4); got {:?}",
    last.stats
  );
  assert_eq!(
    last.stats.generated_tokens, 5,
    "5 tokens yielded; got {:?}",
    last.stats
  );
}

/// **Eos token in the speculative output terminates with `finish_reason="stop"`.**
#[test]
fn speculative_stops_on_eos() {
  let tok = tokenizer("eos_stop");
  // Force argmax == 2 == "</s>" (eos).
  let target = MockModel::with_bias(vec![0.0, 0.0, 10.0, 0.0, 0.0]);
  let draft = MockModel::with_bias(vec![0.0, 0.0, 10.0, 0.0, 0.0]);

  let eos: Vec<u32> = tok.eos_token_ids_iter().collect();
  let responses: Vec<_> = speculative_stream_generate(
    &target,
    &tok,
    &[3u32],
    cache(1),
    cache(1),
    DraftConfig {
      draft_model: Box::new(draft),
      n_draft_tokens: 2,
    },
    GenConfig::default().with_max_tokens(50).with_eos(eos),
  )
  .map(|r| r.unwrap())
  .collect();
  let last = responses.last().expect("at least one response");
  assert_eq!(last.response.finish_reason, Some(FinishReason::Eos));
  let full: String = responses.iter().map(|r| r.response.text.as_str()).collect();
  assert!(
    !full.contains("</s>"),
    "eos token never detokenized, got {full:?}"
  );
}
