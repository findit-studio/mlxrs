//! M3 WS-C PR-3 — the architecture-agnostic generation loop
//! (`mlxrs::lm::generate`), ported 1:1 from `mlx_lm.generate`
//! (`generate_step` / `stream_generate` / `generate`) and
//! `mlx_lm.sample_utils` (`make_sampler` / `make_logits_processors`).
//!
//! Deterministic, dependency-free: a local `MockModel` returns fixed
//! `[1, 1, vocab]` logits and advances every cache entry so the loop's
//! per-step order, EOS / max-token finish reasons, the streaming-detokenizer
//! text assembly, and the sampler / logits-processor composition order are
//! all checkable without a real model or any network. Mirrors the in-crate
//! `model::MockModel` fixture pattern (replicated, not imported — integration
//! tests cannot see the `#[cfg(test)] pub(crate)` helper).
#![cfg(feature = "lm")]

use std::{collections::HashSet, fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array,
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache},
    generate::{
      __resolved_unseeded_seed_for_test, GenConfig, GenStep, GenerationStats, generate,
      generate_step, make_logits_processors, make_sampler, stream_generate,
    },
    model::Model,
  },
};

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

/// A unique temp directory for one test (process-scoped + named so parallel
/// test binaries / cases never collide).
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_generate_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// Write the committed tokenizer fixtures into `dir` and build a real
/// [`mlxrs::tokenizer::Tokenizer`] from them (vocab: `<unk>`=0, `<s>`=1,
/// `</s>`=2 [eos], `hello`=3, `world`=4, `the`=5, `quick`=6, `brown`=7,
/// `fox`=8, `<think>`=9, `</think>`=10).
fn tokenizer(name: &str) -> mlxrs::tokenizer::Tokenizer {
  let dir = temp_dir(name);
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(&dir, None).unwrap()
}

/// A deterministic, dependency-free [`Model`] (replicating the in-crate
/// `model::MockModel`): `forward` advances every cache entry by the
/// token-window length and returns a fixed `[B, S, vocab]` logits array
/// whose per-vocab values are `bias[v]` (so the argmax / sampled token is
/// fully predictable).
struct MockModel {
  /// Per-vocab logit value; `bias.len()` is the vocab size.
  bias: Vec<f32>,
  n_kv_heads: usize,
  head_dim: usize,
}

impl MockModel {
  /// `bias[v] = v` ⇒ greedy argmax is always the last vocab index.
  fn ramp(vocab: usize) -> Self {
    Self {
      bias: (0..vocab).map(|i| i as f32).collect(),
      n_kv_heads: 1,
      head_dim: 2,
    }
  }

  /// Explicit per-vocab logits ⇒ argmax is the index of the max entry.
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
          message: format!("MockModel::forward expects [B, S] tokens, got {shape:?}"),
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

/// A `Model` whose every `forward` returns an [`mlxrs::Error`] — drives the
/// "a step error is yielded as `Err` and ends iteration (no panic, no
/// poison)" contract (spec §4).
struct FailModel;
impl Model for FailModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    Err(mlxrs::Error::Backend {
      message: "mock forward failure".into(),
    })
  }
}

/// A `Model` that records the sequence length of every `forward` call (the
/// per-call input-window length) so a test can assert the exact prefill
/// chunk-boundary sequence — the deterministic pin for the O(P) index-cursor
/// prefill (it must be byte-identical to the old front-drain chunking).
/// Behaves like `MockModel::ramp` otherwise (argmax == vocab-1).
struct RecordingModel {
  bias: Vec<f32>,
  /// One entry per `forward` call: that call's `S` (the chunk / step
  /// window length). Interior mutability — `forward` is `&self`.
  seq_lens: std::cell::RefCell<Vec<usize>>,
}
impl RecordingModel {
  fn ramp(vocab: usize) -> Self {
    Self {
      bias: (0..vocab).map(|i| i as f32).collect(),
      seq_lens: std::cell::RefCell::new(Vec::new()),
    }
  }
}
impl Model for RecordingModel {
  fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(mlxrs::Error::ShapeMismatch {
          message: format!("RecordingModel::forward expects [B, S] tokens, got {shape:?}"),
        });
      }
    };
    self.seq_lens.borrow_mut().push(seq);
    let vocab = self.bias.len();
    let mut data = Vec::with_capacity(batch * seq * vocab);
    for _ in 0..batch * seq {
      data.extend_from_slice(&self.bias);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, vocab))
  }
}

/// A `Model` returning logits with a deliberately **degenerate** axis
/// (`[1, 0, V]` when `zero_seq`, else `[1, 1, 0]`) — drives the
/// `last_position` `S == 0` / `V == 0` recoverable-`Err` guard (must be a
/// clean `Err(ShapeMismatch)` then a fused iterator, never a `usize`
/// underflow / malformed slice / panic).
struct ZeroAxisModel {
  zero_seq: bool,
}
impl Model for ZeroAxisModel {
  fn forward(&self, _tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    if self.zero_seq {
      // `[1, 0, 3]` — empty sequence axis (`logits[:, -1, :]` ⇒ IndexError).
      Array::from_slice::<f32>(&[], &(1usize, 0usize, 3usize))
    } else {
      // `[1, 1, 0]` — empty vocab axis (no distribution to sample).
      Array::from_slice::<f32>(&[], &(1usize, 1usize, 0usize))
    }
  }
}

fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&CacheConfig {
    num_hidden_layers: layers,
    sliding_window: None,
  })
}

// ---------------------------------------------------------------------------
// generate_step — exact mlx-lm per-step order & finish reasons
// ---------------------------------------------------------------------------

/// A `ramp(vocab)` model's argmax is always `vocab-1`, so greedy decoding
/// (default sampler, temp 0) yields the constant token `vocab-1` for exactly
/// `max_tokens` steps with no eos in the set.
#[test]
fn generate_step_greedy_deterministic_sequence() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 4,
    eos: vec![], // never stop on eos
    ..GenConfig::default()
  };
  let prompt = [1u32, 2, 3];
  let toks: Vec<u32> = generate_step(&model, &prompt, cache(2), cfg)
    .map(|r| r.unwrap().token)
    .collect();
  // argmax of [0,1,2,3,4] == 4 every step; exactly max_tokens of them.
  assert_eq!(toks, vec![4, 4, 4, 4]);
}

/// `finish_reason == "length"`: no eos hit, the iterator ends after exactly
/// `max_tokens` tokens.
#[test]
fn generate_step_stops_at_max_tokens_length() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 3,
    eos: vec![],
    ..GenConfig::default()
  };
  let n = generate_step(&model, &[1u32], cache(1), cfg).count();
  assert_eq!(n, 3);
}

/// `finish_reason == "stop"`: the model always argmaxes vocab-1; making that
/// the eos id stops generation after the first token (the eos token IS
/// yielded by `generate_step`, exactly as mlx-lm yields it then breaks in
/// `stream_generate`).
#[test]
fn generate_step_stops_on_eos_token() {
  let model = MockModel::ramp(5); // argmax == 4
  let cfg = GenConfig {
    max_tokens: 100,
    eos: vec![4],
    ..GenConfig::default()
  };
  let toks: Vec<u32> = generate_step(&model, &[1u32, 2], cache(1), cfg)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(toks, vec![4], "eos token yielded once, then iteration ends");
}

/// The yielded logprobs are the exact mlx-lm normalization
/// `logits - logsumexp(logits, keepdims=True)`: a `[vocab]` vector summing
/// (in prob space) to 1, argmax-aligned with the sampled token.
#[test]
fn generate_step_yields_logprobs_normalized() {
  let model = MockModel::with_bias(vec![0.0, 0.0, 10.0]); // argmax == 2
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    // L3: logprobs are opt-in; request them so the squeeze + per-step
    // `Some(Array)` yield is enabled.
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let GenStep {
    token: tok,
    logprobs: lp,
  } = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  assert_eq!(tok, 2);
  let mut lp = lp.expect("collect_logprobs=true ⇒ Some(Array)");
  assert_eq!(lp.shape(), vec![3], "logprobs squeezed to [vocab]");
  let v = lp.to_vec::<f32>().unwrap();
  // sum(exp(logprobs)) == 1 (normalized) and argmax preserved.
  let s: f32 = v.iter().map(|x| x.exp()).sum();
  assert!((s - 1.0).abs() < 1e-4, "exp(logprobs) sums to 1, got {s}");
  assert!(v[2] > v[0] && v[2] > v[1], "argmax preserved in logprobs");
}

/// Prefill chunking by `prefill_step_size`: a prompt longer than the chunk
/// size must be consumed in chunks (cache filled by the full prompt before
/// the first decode), and decoding still produces the right tokens. The
/// `MockModel` errors if a chunk's shape is wrong, so a correct token
/// sequence proves the chunked prefill drove `forward` correctly.
#[test]
fn generate_step_prefill_chunks_by_prefill_step_size() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 2,
    prefill_step_size: 2, // force multi-chunk prefill of a 5-token prompt
    eos: vec![],
    ..GenConfig::default()
  };
  let prompt = [1u32, 2, 3, 4, 1];
  let toks: Vec<u32> = generate_step(&model, &prompt, cache(1), cfg)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(toks, vec![4, 4], "chunked prefill + 2 decode steps");
}

/// Regression (Copilot fix #1): the O(P) index-cursor prefill must produce
/// **byte-identical** chunk boundaries to mlx-lm `generate_step`
/// (lines 430-453) — `n = min(prefill_step_size, (total - processed) - 1)`
/// per chunk, then the first decode step consumes the unconsumed tail
/// `prompt[processed..]`. A [`RecordingModel`] records every `forward`
/// window length; the prefill chunk-length sequence + the first-step /
/// per-decode window lengths must be exactly the mlx-lm sequence (this is
/// the only observable contract the front-drain → cursor change must keep,
/// and it catches an off-by-one in the boundary or the tail hand-off).
#[test]
fn generate_step_prefill_chunk_boundaries_are_exact() {
  // Case A: `K = 2`, prompt of 6 ⇒ mlx-lm trace
  // processed: 0→2 (n=min(2,5)=2), 2→4 (n=min(2,3)=2), 4→5 (n=min(2,1)=1;
  // the boundary-defining clamp), then `6-5=1` not `> 1` ⇒ exit; the first
  // `_step` forwards the length-1 tail `prompt[5..6]`. So prefill windows
  // == [2, 2, 1], then 3 decode steps each window length 1.
  let model = RecordingModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 3,
    prefill_step_size: 2,
    eos: vec![],
    ..GenConfig::default()
  };
  let prompt = [1u32, 2, 3, 4, 1, 2]; // P = 6
  let toks: Vec<u32> = generate_step(&model, &prompt, cache(1), cfg)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(
    toks,
    vec![4, 4, 4],
    "ramp(5) argmax == 4 for every decode step"
  );
  assert_eq!(
    *model.seq_lens.borrow(),
    vec![
      2, 2, 1, /* first decode (tail) */ 1, /* decode */ 1, /* decode */ 1
    ],
    "prefill chunk boundaries byte-identical to mlx-lm min(step, remaining) \
     (incl. the size-1 final-chunk clamp) + the length-1 tail hand-off"
  );

  // Case B: `K = 3`, prompt of 7 ⇒ processed: 0→3 (n=min(3,6)=3), 3→6
  // (n=min(3,3)=3), then `7-6=1` not `> 1` ⇒ exit; first `_step` forwards
  // the length-1 tail. Prefill == [3, 3], then 2 decode steps length 1.
  let model_b = RecordingModel::ramp(5);
  let cfg_b = GenConfig {
    max_tokens: 2,
    prefill_step_size: 3,
    eos: vec![],
    ..GenConfig::default()
  };
  let prompt_b = [1u32, 2, 3, 4, 1, 2, 3]; // P = 7
  let _: Vec<u32> = generate_step(&model_b, &prompt_b, cache(1), cfg_b)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(
    *model_b.seq_lens.borrow(),
    vec![3, 3, /* tail */ 1, /* decode */ 1],
    "multi-chunk prefill + non-multiple length-1 tail: exact mlx-lm boundaries"
  );

  // Case C: chunk size larger than the prompt ⇒ the single prefill chunk
  // is clamped to `remaining = P - 1`, not `prefill_step_size`. mlx-lm
  // trace (P=3, K=8): processed 0; `3-0 > 1` ⇒ remaining=2, n=min(8,2)=2,
  // forward len 2, processed=2; `3-2=1` not `> 1` ⇒ exit; first `_step`
  // forwards the length-1 tail `prompt[2..3]`. So prefill == [2], tail
  // step length 1.
  let model_c = RecordingModel::ramp(5);
  let cfg_c = GenConfig {
    max_tokens: 1,
    prefill_step_size: 8, // larger than the prompt
    eos: vec![],
    ..GenConfig::default()
  };
  let prompt_c = [1u32, 2, 3]; // P = 3
  let _: Vec<u32> = generate_step(&model_c, &prompt_c, cache(1), cfg_c)
    .map(|r| r.unwrap().token)
    .collect();
  assert_eq!(
    *model_c.seq_lens.borrow(),
    vec![2, /* tail */ 1],
    "prefill clamps the single chunk to remaining=P-1, tail is the last token"
  );
}

/// Regression (Copilot fix #2): a buggy model returning logits with a
/// zero-length sequence (`[1, 0, V]`) or vocab (`[1, S, 0]`) axis must
/// surface a **recoverable** `Err(ShapeMismatch)` (mirroring Python
/// `logits[:, -1, :]` raising `IndexError`) — never a `usize` underflow on
/// `s - 1`, a malformed `[0, -1, 0]` slice, or a panic. The iterator yields
/// the `Err` once and then fuses (spec §4).
#[test]
fn generate_step_zero_length_logits_axis_is_recoverable_err() {
  for zero_seq in [true, false] {
    let model = ZeroAxisModel { zero_seq };
    let cfg = GenConfig {
      max_tokens: 8,
      eos: vec![],
      ..GenConfig::default()
    };
    let mut it = generate_step(&model, &[1u32, 2], cache(1), cfg);
    let first = it.next().expect("an item is produced (no panic/underflow)");
    match first {
      Err(mlxrs::Error::ShapeMismatch { .. }) => {}
      other => panic!(
        "expected a recoverable Err(ShapeMismatch) for a zero-length {} axis, got {other:?}",
        if zero_seq { "S" } else { "V" }
      ),
    }
    assert!(
      it.next().is_none(),
      "iterator fuses after the zero-axis Err (no panic, no re-entry)"
    );
  }
}

/// A `forward` error is surfaced as `Err` and the iterator then ends — no
/// panic, no poison (spec §4).
#[test]
fn generate_step_forward_error_yields_err_then_ends() {
  let model = FailModel;
  let cfg = GenConfig {
    max_tokens: 8,
    eos: vec![],
    ..GenConfig::default()
  };
  let mut it = generate_step(&model, &[1u32, 2], cache(1), cfg);
  let first = it.next().expect("an item is produced");
  assert!(first.is_err(), "the forward error is yielded as Err");
  assert!(it.next().is_none(), "iteration ends after the error");
}

/// `max_tokens == 0` produces no tokens (no panic).
#[test]
fn generate_step_zero_max_tokens_is_empty() {
  let model = MockModel::ramp(3);
  let cfg = GenConfig {
    max_tokens: 0,
    eos: vec![],
    ..GenConfig::default()
  };
  assert_eq!(generate_step(&model, &[1u32], cache(1), cfg).count(), 0);
}

// ---------------------------------------------------------------------------
// make_sampler / make_logits_processors — composition order
// ---------------------------------------------------------------------------

/// Default sampler (temp 0) is argmax: returns a `[1]` u32 with the argmax
/// index, no randomness.
#[test]
fn make_sampler_default_is_argmax() {
  let mut s = make_sampler(0.0, 0.0, 0.0, 1, 0, 0.0, 0.0, &[], None).unwrap();
  let lp = Array::from_slice::<f32>(&[-3.0, -1.0, -2.0], &[1, 3]).unwrap();
  let mut tok = s(&lp).unwrap();
  assert_eq!(tok.to_vec::<u32>().unwrap(), vec![1], "argmax index");
}

/// `make_logits_processors` composes the #29 primitives: an empty config
/// yields no processors; a `logit_bias` processor shifts exactly its target
/// logit; a repetition penalty (temp-0-irrelevant) divides a repeated
/// positive token's logit. Asserts the *processor application order*
/// (logit_bias first), mirroring `sample_utils.make_logits_processors`.
#[test]
fn make_logits_processors_composition_and_order() {
  // Empty → no processors.
  let none = make_logits_processors(&[], None, 20, None, 20, None, 20).unwrap();
  assert!(none.is_empty());

  // logit_bias only: +5 on column 0.
  let procs = make_logits_processors(&[(0, 5.0)], None, 20, None, 20, None, 20).unwrap();
  assert_eq!(procs.len(), 1);
  let logits = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
  let mut out = procs[0](&[7u32], &logits).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![6.0, 2.0, 3.0]);

  // Order: logit_bias is index 0, repetition penalty index 1.
  let procs = make_logits_processors(&[(1, 1.0)], Some(2.0), 20, None, 20, None, 20).unwrap();
  assert_eq!(procs.len(), 2, "logit_bias + repetition_penalty");
  // First processor (logit_bias) adds 1.0 at col 1; second (rep penalty,
  // token id 0, positive logit) divides col 0 by 2.
  let logits = Array::from_slice::<f32>(&[4.0, 8.0, 1.0], &[1, 3]).unwrap();
  let mut a = procs[0](&[0u32], &logits).unwrap();
  assert_eq!(
    a.to_vec::<f32>().unwrap(),
    vec![4.0, 9.0, 1.0],
    "bias first"
  );
  let mut b = procs[1](&[0u32], &a).unwrap();
  assert_eq!(
    b.to_vec::<f32>().unwrap(),
    vec![2.0, 9.0, 1.0],
    "rep penalty divides repeated positive token id 0"
  );
}

/// Regression (Codex adversarial-review): repetition / presence / frequency
/// penalties each use their **own** context window, exactly as
/// `sample_utils.make_logits_processors`'s independent
/// `repetition_context_size` / `presence_context_size` /
/// `frequency_context_size`. With a long repetition window but a
/// presence/frequency window of `1`, only the most-recent token is penalized
/// by presence/frequency — a configuration impossible if all three reused
/// `repetition_context_size`.
#[test]
fn make_logits_processors_independent_context_windows() {
  // repetition window 20 (unused — no rep penalty), presence window 1,
  // frequency window 1. History [0, 1]: only token id 1 (the last) is in
  // the size-1 presence/frequency window.
  let procs = make_logits_processors(&[], None, 20, Some(0.5), 1, Some(0.25), 1).unwrap();
  assert_eq!(procs.len(), 2, "presence + frequency processors");
  let logits = Array::from_slice::<f32>(&[10.0, 10.0, 10.0], &[1, 3]).unwrap();

  // Presence: subtract 0.5 once from ids in tokens[-1:] == [1] ⇒ only col 1.
  let mut p = procs[0](&[0u32, 1], &logits).unwrap();
  assert_eq!(
    p.to_vec::<f32>().unwrap(),
    vec![10.0, 9.5, 10.0],
    "presence used its OWN size-1 window (only the last token), not the size-20 repetition window"
  );

  // Frequency: subtract 0.25 per occurrence in tokens[-1:] == [1] ⇒ col 1.
  let mut f = procs[1](&[0u32, 1], &logits).unwrap();
  assert_eq!(
    f.to_vec::<f32>().unwrap(),
    vec![10.0, 9.75, 10.0],
    "frequency used its OWN size-1 window"
  );

  // Widen presence to 2 ⇒ tokens[-2:] == [0, 1] ⇒ BOTH cols 0 and 1 drop,
  // proving the window is the presence-specific param (now distinct from
  // the still-size-20 repetition default and the prior size-1 result).
  let procs2 = make_logits_processors(&[], None, 20, Some(0.5), 2, None, 20).unwrap();
  let mut p2 = procs2[0](&[0u32, 1], &logits).unwrap();
  assert_eq!(
    p2.to_vec::<f32>().unwrap(),
    vec![9.5, 9.5, 10.0],
    "presence window 2 penalizes both recent tokens — independent of repetition_context_size"
  );
}

/// Regression (Codex adversarial-review): a `*_context_size` of `0` must
/// mirror Python `tokens[-0:]` == `tokens[0:]` == the **entire** history
/// (NOT an empty slice / no-op), for repetition, presence, AND frequency —
/// the `sample_utils.make_logits_processors` closures' exact slicing edge
/// case. With history `[0, 1]` and a `0` window, both columns 0 and 1 are
/// penalized (an empty slice would leave the logits unchanged — the defect).
#[test]
fn make_logits_processors_zero_context_is_full_history() {
  let logits = || Array::from_slice::<f32>(&[10.0, 10.0, 10.0], &[1, 3]).unwrap();
  let hist = &[0u32, 1];

  // Repetition penalty 2.0, context 0 ⇒ full history [0,1] ⇒ cols 0,1
  // divided by 2 (positive logits ⇒ `logit / penalty`).
  let rep = make_logits_processors(&[], Some(2.0), 0, None, 20, None, 20).unwrap();
  let mut r = rep[0](hist, &logits()).unwrap();
  assert_eq!(
    r.to_vec::<f32>().unwrap(),
    vec![5.0, 5.0, 10.0],
    "repetition context 0 == full history (cols 0 and 1), not a no-op"
  );

  // Presence penalty 0.5, context 0 ⇒ full history ⇒ cols 0,1 minus 0.5.
  let pre = make_logits_processors(&[], None, 20, Some(0.5), 0, None, 20).unwrap();
  let mut p = pre[0](hist, &logits()).unwrap();
  assert_eq!(
    p.to_vec::<f32>().unwrap(),
    vec![9.5, 9.5, 10.0],
    "presence context 0 == full history, not a no-op"
  );

  // Frequency penalty 0.25, context 0, history [0,1] (each once) ⇒ cols
  // 0,1 minus 0.25.
  let fre = make_logits_processors(&[], None, 20, None, 20, Some(0.25), 0).unwrap();
  let mut f = fre[0](hist, &logits()).unwrap();
  assert_eq!(
    f.to_vec::<f32>().unwrap(),
    vec![9.75, 9.75, 10.0],
    "frequency context 0 == full history, not a no-op"
  );
}

/// `make_sampler` propagates a sample.rs validation `Err` (it does NOT
/// re-validate): an out-of-range `top_k` surfaces sample.rs's error lazily
/// when the sampler runs (mlx-lm builds the chain unconditionally; the bound
/// check lives in `apply_top_k`).
#[test]
fn make_sampler_propagates_sample_rs_errors() {
  // temp != 0 so the chain is built; top_k == vocab is out of (0, vocab).
  let mut s = make_sampler(0.7, 0.0, 0.0, 1, 3, 0.0, 0.0, &[], None).unwrap();
  let lp = Array::from_slice::<f32>(&[-1.0, -2.0, -3.0], &[1, 3]).unwrap();
  assert!(s(&lp).is_err(), "out-of-range top_k error propagates");
}

/// Regression (Codex adversarial-review): stochastic (`temp > 0`) generation
/// must be reproducible under an explicit `seed` and must NOT restart from
/// the same RNG sequence on independent unseeded runs — mirroring mlx-lm's
/// advancing process-global `mx.random.state` (+ `mx.random.seed`).
///
/// **Deterministic by construction** (no probabilistic pass condition):
/// - *Seeded reproducibility*: two runs with the **same** explicit seed are
///   byte-identical; two runs with two **fixed distinct** explicit seeds
///   differ — both are fully pinned by the explicit seed (a fixed-input
///   regression, not a comparison of random outputs).
/// - *Unseeded independence*: asserted via the deterministic seed-
///   *resolution* path ([`__resolved_unseeded_seed_for_test`], the seed an
///   unseeded `make_sampler` resolves to). The monotonic per-process counter
///   strictly advances every call, so a batch of unseeded resolutions is
///   pairwise distinct — proving independent unseeded runs get distinct RNG
///   streams **without** comparing two random token sequences (the previous
///   flaky probabilistic `assert_ne!` on unseeded outputs is gone).
#[test]
fn stochastic_sampler_independent_runs_and_seed_reproducible() {
  // --- Seeded reproducibility: fully pinned by the explicit seed ---------
  // Uniform logits ⇒ the categorical draw exercises the RNG every step.
  let model = MockModel::with_bias(vec![0.0; 8]);
  let run = |seed: u64| -> Vec<u32> {
    let cfg = GenConfig {
      max_tokens: 24,
      temp: 1.0, // stochastic chain (not argmax)
      eos: vec![],
      seed: Some(seed),
      ..GenConfig::default()
    };
    generate_step(&model, &[1u32], cache(1), cfg)
      .map(|r| r.unwrap().token)
      .collect()
  };

  // Same explicit seed ⇒ bit-identical (mx.random.seed parity).
  let s1 = run(12345);
  let s2 = run(12345);
  assert_eq!(s1.len(), 24);
  assert_eq!(s1, s2, "a fixed seed reproduces the exact token sequence");
  // Two FIXED distinct seeds ⇒ different sequences. Deterministic for these
  // exact constants (the run is fully pinned by the seed — not a random
  // output comparison; it is a fixed-input regression).
  let s3 = run(67890);
  assert_ne!(
    s1, s3,
    "two fixed distinct seeds give different (reproducible) sequences"
  );

  // --- Unseeded independence: deterministic seed-resolution path ---------
  // An unseeded `make_sampler` resolves a fresh seed via a strictly-
  // advancing monotonic per-process counter, so successive unseeded
  // resolutions never repeat ⇒ independent unseeded non-greedy runs get
  // distinct RNG streams. Asserted by observing the resolution path (not by
  // comparing random sampler outputs): a batch of resolutions is pairwise
  // distinct. The OLD seed-0 hardcode (the defect) would make every unseeded
  // resolution identical — caught here deterministically.
  const N: usize = 256;
  let seeds: Vec<u64> = (0..N)
    .map(|_| __resolved_unseeded_seed_for_test())
    .collect();
  let unique: HashSet<u64> = seeds.iter().copied().collect();
  assert_eq!(
    unique.len(),
    N,
    "every unseeded seed resolution is distinct (monotonic-counter advance) \
     ⇒ independent unseeded runs never restart from the same RNG sequence; \
     a constant/hardcoded seed would collapse these to 1"
  );
}

/// `make_logits_processors` propagates a sample.rs `Err` (mismatched
/// logit_bias indices/values is impossible via the `(i32, f32)` pair API, so
/// drive the repetition-penalty negative-penalty validation instead — it is
/// surfaced by sample.rs, not re-checked here): a negative repetition
/// penalty errors when the processor runs.
#[test]
fn make_logits_processors_propagates_sample_rs_errors() {
  let procs = make_logits_processors(&[], Some(-1.0), 20, None, 20, None, 20).unwrap();
  assert_eq!(procs.len(), 1);
  let logits = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2]).unwrap();
  assert!(
    procs[0](&[0u32], &logits).is_err(),
    "negative penalty error propagates from sample.rs"
  );
}

// ---------------------------------------------------------------------------
// generate_step composition — processors run BEFORE logsumexp BEFORE sampler
// ---------------------------------------------------------------------------

/// The exact mlx-lm order is observable end-to-end: a `GenConfig.logit_bias`
/// that overwhelmingly boosts a non-argmax column must change the sampled
/// token (the processor `make_logits_processors` built from the config is
/// applied to raw logits *before* `logits - logsumexp` *before* the argmax
/// sampler). Proves processors run, in raw-logit space, before
/// normalization, before sampling — through the full `generate_step` config
/// path mlx-lm's `generate` uses.
#[test]
fn generate_step_applies_processors_before_logsumexp_before_sampler() {
  let model = MockModel::with_bias(vec![0.0, 0.0, 1.0]); // raw argmax == 2
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    logit_bias: vec![(0, 100.0)], // post-processor argmax == 0
    ..GenConfig::default()
  };
  let step = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  assert_eq!(
    step.token, 0,
    "logit_bias steered the argmax ⇒ processor ran on raw logits before logsumexp before sampler"
  );
}

// ---------------------------------------------------------------------------
// stream_generate / generate — text assembly, counts, finish_reason
// ---------------------------------------------------------------------------

/// `stream_generate` maps `generate_step` through the #18 streaming
/// detokenizer into `GenerationResponse`s; the assembled text is the decode
/// of the produced tokens, `finish_reason` is `"length"` when max_tokens is
/// reached, and the prompt/generation token counts are populated.
#[test]
fn stream_generate_text_assembly_and_counts_length() {
  let tok = tokenizer("stream_len");
  // argmax == 4 ("world"); never an eos (eos id is 2 = "</s>").
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 3,
    eos: tok.eos_token_ids().iter().copied().collect(),
    ..GenConfig::default()
  };
  let prompt = [3u32]; // "hello"
  let responses: Vec<_> = stream_generate(&model, &tok, &prompt, cache(1), cfg)
    .map(|r| r.unwrap())
    .collect();
  assert!(!responses.is_empty());
  let last = responses.last().unwrap();
  assert_eq!(last.finish_reason.as_deref(), Some("length"));
  assert_eq!(last.prompt_tokens, 1);
  assert_eq!(last.generation_tokens, 3);
  let full: String = responses.iter().map(|r| r.text.as_str()).collect();
  // 3 × token 4 == "world world world" (decoded; exact spacing is the
  // detokenizer's, so assert the decoded token content is present).
  assert!(
    full.contains("world"),
    "decoded text contains the token, got {full:?}"
  );
}

/// `finish_reason == "stop"` when the model emits an eos token: the eos token
/// is NOT detokenized into the text (mlx-lm breaks before `add_token`), and
/// the final response carries `"stop"`.
#[test]
fn stream_generate_stop_finish_reason_on_eos() {
  let tok = tokenizer("stream_stop");
  // Force argmax == 2 == "</s>" (the eos id).
  let model = MockModel::with_bias(vec![0.0, 0.0, 10.0, 0.0, 0.0]);
  let cfg = GenConfig {
    max_tokens: 50,
    eos: tok.eos_token_ids().iter().copied().collect(),
    ..GenConfig::default()
  };
  let responses: Vec<_> = stream_generate(&model, &tok, &[3u32], cache(1), cfg)
    .map(|r| r.unwrap())
    .collect();
  let last = responses.last().unwrap();
  assert_eq!(last.finish_reason.as_deref(), Some("stop"));
  // The eos token never reaches the detokenizer, so no "</s>" text.
  let full: String = responses.iter().map(|r| r.text.as_str()).collect();
  assert!(
    !full.contains("</s>"),
    "eos token not detokenized, got {full:?}"
  );
}

/// `generate` collects `stream_generate` into the full `String` (the eos
/// token contributes no text) and returns the aggregate
/// [`mlxrs::lm::generate::GenerationStats`] alongside.
#[test]
fn generate_collects_to_string() {
  let tok = tokenizer("gen_str");
  let model = MockModel::ramp(5); // argmax == 4 == "world"
  let cfg = GenConfig {
    max_tokens: 2,
    eos: tok.eos_token_ids().iter().copied().collect(),
    ..GenConfig::default()
  };
  let (out, stats) = generate(&model, &tok, &[3u32], cache(1), cfg).unwrap();
  assert!(out.contains("world"), "collected text, got {out:?}");
  assert_eq!(stats.prompt_tokens, 1);
  assert_eq!(stats.generation_tokens, 2);
}

/// A `forward` error inside `stream_generate` surfaces as a yielded `Err`
/// (the underlying `generate_step` Iterator-Err contract is preserved
/// through the detokenizer mapping) — no panic, no poison.
#[test]
fn stream_generate_propagates_forward_error() {
  let tok = tokenizer("stream_err");
  let model = FailModel;
  let cfg = GenConfig {
    max_tokens: 8,
    eos: vec![],
    ..GenConfig::default()
  };
  let mut it = stream_generate(&model, &tok, &[3u32], cache(1), cfg);
  let first = it.next().expect("an item");
  assert!(
    first.is_err(),
    "forward error propagated through stream_generate"
  );
  assert!(it.next().is_none(), "iteration ends after the error");
}

// ---------------------------------------------------------------------------
// GenStep typed item — field-equivalence, Debug, and tuple back-compat
// ---------------------------------------------------------------------------

/// `GenStep.token` matches the value the prior `(u32, Array)` tuple's `.0`
/// would have carried — the typed-struct refactor (LM-3) is a pure
/// ergonomics upgrade with **no semantic change** to the iterator's
/// payload. `MockModel::ramp(5)`'s argmax is always 4 (the last vocab id),
/// the exact same value the old tuple's `.0` would have yielded.
#[test]
fn gen_step_token_field_matches_prior_tuple_zero() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    // Request logprobs so the prior `.1 == [V] Array` contract is
    // preserved verbatim under the L3 opt-in.
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let step = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  assert_eq!(
    step.token, 4,
    "GenStep.token == argmax (ramp(5) ⇒ 4); identical to the prior (u32, Array).0"
  );
  // And `.logprobs` carries the [V] vector (prior .1 contract).
  let lp = step.logprobs.as_ref().expect("collect_logprobs=true");
  assert_eq!(lp.shape(), vec![5]);
}

/// `GenStep` is `Debug` so test failures stay diagnosable (and the public
/// API doc-comment promises it via `#[derive(Debug)]`). Enforced via a
/// compile-time `T: Debug` bound — Rust's `Debug` format string is not
/// guaranteed stable across compiler versions (Copilot review #3272760827),
/// so asserting on the formatted *contents* is brittle. The bound check
/// catches the only regression that matters (the derive being removed).
#[test]
fn gen_step_is_debug() {
  fn assert_debug<T: std::fmt::Debug>() {}
  assert_debug::<GenStep>();

  // Smoke-call the formatter on a real step too, so a `Debug` impl that
  // panics at runtime (e.g. an over-clever manual impl that calls into
  // a fallible path) is caught — but DON'T assert on the resulting
  // string contents (format is rustc-version-dependent).
  let model = MockModel::ramp(3);
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    ..GenConfig::default()
  };
  let step = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  let _ = format!("{step:?}");
}

/// `From<GenStep> for (u32, Option<Array>)` back-compat: a `GenStep`
/// round-trips into the (u32, Option<Array>) tuple via `.into()`, so call
/// sites that preferred tuple destructure (`let (tok, lp) = step.into();`)
/// keep working unchanged (the inner `Option` honors the L3 opt-in).
#[test]
fn gen_step_into_tuple_roundtrip() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let step = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  // Capture the typed fields before the move so we can cross-check.
  let expected_token = step.token;
  let expected_shape = step
    .logprobs
    .as_ref()
    .expect("collect_logprobs=true ⇒ Some(Array)")
    .shape();
  let (tok, lp): (u32, Option<Array>) = step.into();
  assert_eq!(
    tok, expected_token,
    "tuple .0 == GenStep.token via From<GenStep>"
  );
  let mut lp = lp.expect("collect_logprobs=true ⇒ tuple .1 carries Some(Array)");
  assert_eq!(
    lp.shape(),
    expected_shape,
    "tuple .1 carries the same [V] logprobs Array (no clone, no shape change)"
  );
  // Sanity: the moved-out logprobs is still the normalized [V] vector
  // (mlx-lm `logprobs.squeeze(0)`), summing to 1 in prob space.
  let s: f32 = lp.to_vec::<f32>().unwrap().iter().map(|x| x.exp()).sum();
  assert!((s - 1.0).abs() < 1e-4, "exp(logprobs) sums to 1, got {s}");
}

// ---------------------------------------------------------------------------
// L3 — `collect_logprobs` opt-in + `GenerationStats` aggregate
// ---------------------------------------------------------------------------

/// Default flow: with [`GenConfig::collect_logprobs`] = `false` (the
/// default), every yielded [`GenStep::logprobs`] is `None` — the squeeze
/// is skipped, no MLX node materialized per step. The token is still
/// produced exactly as before.
#[test]
fn generate_step_default_skips_logprobs() {
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 3,
    eos: vec![],
    ..GenConfig::default()
  };
  // Sanity: the default knob is `false`.
  assert!(
    !GenConfig::default().collect_logprobs,
    "default GenConfig.collect_logprobs is false (opt-in)"
  );
  let steps: Vec<GenStep> = generate_step(&model, &[1u32], cache(1), cfg)
    .map(|r| r.unwrap())
    .collect();
  assert_eq!(steps.len(), 3);
  for step in &steps {
    assert!(
      step.logprobs.is_none(),
      "collect_logprobs=false ⇒ every GenStep.logprobs is None"
    );
    assert_eq!(step.token, 4, "token still produced (ramp(5) argmax)");
  }
}

/// Hand-traced parity (L3 spec): with a deterministic argmax sampler
/// (`temp == 0`, the default) and an explicit `[V]` logits bias, the
/// yielded `[V]` logprobs is **exactly** `log_softmax(logits)` —
/// equivalently `logits - logsumexp(logits)` (mlx-lm `generate_step` line
/// 416, the only normalization in the loop) — and the sampled token's
/// logprob (`logprobs[token]`) is the maximum entry of the vector.
///
/// Computes the closed-form `log_softmax` from the same bias and asserts
/// elementwise agreement; this pins the per-step logprob extraction to
/// the mathematical definition the mlx-lm server (`server.py:891`
/// `r.logprobs[r.token].item()`) relies on.
#[test]
fn generate_step_logprob_matches_log_softmax() {
  // Three vocab entries: 1.0, 2.0, 5.0 ⇒ argmax == 2 ⇒ sampler picks
  // token 2 ⇒ its logprob is `log_softmax([1, 2, 5])[2]`.
  let bias = [1.0_f32, 2.0, 5.0];
  let model = MockModel::with_bias(bias.to_vec());
  let cfg = GenConfig {
    max_tokens: 1,
    eos: vec![],
    collect_logprobs: true, // L3 opt-in
    ..GenConfig::default()
  };
  let step = generate_step(&model, &[1u32], cache(1), cfg)
    .next()
    .unwrap()
    .unwrap();
  assert_eq!(step.token, 2, "argmax of [1, 2, 5] is index 2");

  let mut lp = step.logprobs.expect("collect_logprobs=true ⇒ Some(Array)");
  assert_eq!(lp.shape(), vec![3]);
  let got = lp.to_vec::<f32>().unwrap();

  // Closed-form `log_softmax(bias) = bias - log(sum(exp(bias)))`.
  let max = bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let log_sum_exp = max + (bias.iter().map(|x| (x - max).exp()).sum::<f32>()).ln();
  let expected: [f32; 3] = [
    bias[0] - log_sum_exp,
    bias[1] - log_sum_exp,
    bias[2] - log_sum_exp,
  ];

  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    assert!(
      (g - e).abs() < 1e-5,
      "logprobs[{i}] mismatch: got {g}, expected {e} (log_softmax of {bias:?})"
    );
  }
  // Sampled-token logprob == the max entry (mlx-lm `r.logprobs[r.token]`).
  let sampled_lp = got[step.token as usize];
  assert!(
    (sampled_lp - expected[2]).abs() < 1e-5,
    "sampled token's logprob == log_softmax(bias)[token]"
  );
  assert!(
    sampled_lp >= got[0] && sampled_lp >= got[1],
    "argmax-of-logprobs == sampled token (the sampler picked the maximum)"
  );
}

/// `generate` returns the aggregate [`GenerationStats`] (counts + tps +
/// peak_memory). Counts must match the per-response final values; tps
/// fields are computed from wall-clock timing (assert `>= 0.0`, since on a
/// trivial in-process model the prompt/generation phases can fall within
/// one clock tick → 0.0 by mlx-lm's same divide-by-zero guard).
#[test]
fn generate_returns_generation_stats() {
  let tok = tokenizer("gen_stats");
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 3,
    eos: tok.eos_token_ids().iter().copied().collect(),
    ..GenConfig::default()
  };
  let prompt = [3u32, 5u32]; // arbitrary 2-token prompt
  let (_text, stats): (String, GenerationStats) =
    generate(&model, &tok, &prompt, cache(1), cfg).unwrap();

  assert_eq!(stats.prompt_tokens, prompt.len());
  assert_eq!(stats.generation_tokens, 3);
  assert!(
    stats.prompt_tps >= 0.0,
    "prompt_tps is a measured tokens-per-second (>=0)"
  );
  assert!(
    stats.generation_tps >= 0.0,
    "generation_tps is a measured tokens-per-second (>=0)"
  );
  // peak_memory_bytes is `Option<u64>`: under the in-process MLX runtime
  // the counter is always available, so the Option is `Some` and
  // non-zero (we've allocated at least the prompt + cache).
  let peak = stats
    .peak_memory_bytes
    .expect("mlx_get_peak_memory is available in-process");
  assert!(peak > 0, "peak_memory_bytes > 0 after an in-process run");
}

/// Per-response `peak_memory_bytes` is `Some(u64)` under the in-process
/// MLX runtime (mlx-lm's `mx.get_peak_memory() / 1e9` analogue) and
/// non-decreasing across the stream (the underlying counter is a
/// monotonic process-global peak).
#[test]
fn stream_generate_peak_memory_is_monotonic() {
  let tok = tokenizer("stream_peak");
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 4,
    eos: vec![],
    ..GenConfig::default()
  };
  let responses: Vec<_> = stream_generate(&model, &tok, &[3u32], cache(1), cfg)
    .map(|r| r.unwrap())
    .collect();
  assert!(!responses.is_empty());
  let peaks: Vec<u64> = responses
    .iter()
    .map(|r| {
      r.peak_memory_bytes
        .expect("peak_memory available in-process")
    })
    .collect();
  for w in peaks.windows(2) {
    assert!(
      w[1] >= w[0],
      "peak_memory is monotonically non-decreasing across the stream, got {peaks:?}"
    );
  }
}

/// L3 zero-cost opt-out (Codex review fix): with
/// `collect_logprobs == false` and the default greedy sampler
/// (`temp == 0`), the `logits - logsumexp(logits)` normalization is
/// SKIPPED entirely — not just the `[V]` squeeze. The sampler reads raw
/// post-processor logits, and `argmax(logits) == argmax(logits - lse)` so
/// the sampled token is byte-identical to a `collect_logprobs == true`
/// run.
///
/// The "zero-cost" property — that the logsumexp graph node is never built
/// on the opt-out path — is guaranteed STRUCTURALLY by the 3-way
/// `match (needs_normalization, temp_stochastic)` in `generate.rs`, which
/// only constructs the logsumexp node in the `(true, _)` arm. It is
/// verified FUNCTIONALLY here by the `step.logprobs.is_none()` assertion on
/// the opt-out path (a `None` logprobs is the observable signal that the
/// normalization branch — the only producer of `Some(logprobs)` — did not
/// run).
///
/// A peak-memory MAGNITUDE comparison (e.g. `peak(opt-out) <= peak(opt-in)`)
/// is deliberately NOT asserted: [`mlxrs::memory::peak_memory`] wraps mlx-c's
/// `mlx_get_peak_memory`, a process-global monotonic high-water mark that
/// never decreases and is shared by every concurrently-running test. Under
/// the default multi-threaded test harness, other tests allocate during this
/// test's measurement window and inflate the global counter unpredictably, so
/// an absolute cross-sub-run peak comparison is fundamentally unreliable. The
/// structural + functional checks above prove the contract without reading a
/// shared global counter.
#[test]
fn generate_step_default_skips_logprobs_node() {
  // Same prompt + model fixture for both runs, so the only difference is
  // the L3 normalization gate.
  let prompt = [1u32];
  let max_tokens = 8;
  let bias = [1.0_f32, 2.0, 5.0, 3.0, 4.0];

  // Run A: opt-in. Both the normalization AND the squeeze run; every
  // step yields `Some([V])` logprobs. Materialize each logprobs Array via
  // `to_vec` (which evaluates the lazy graph) so the logsumexp + subtract
  // nodes are actually computed, not just built.
  let model_a = MockModel::with_bias(bias.to_vec());
  let cfg_a = GenConfig {
    max_tokens,
    eos: vec![],
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let mut tokens_a: Vec<u32> = Vec::new();
  for step in generate_step(&model_a, &prompt, cache(1), cfg_a) {
    let s = step.unwrap();
    tokens_a.push(s.token);
    // `collect_logprobs == true` ⇒ Some(Array): the normalization branch
    // ran and produced logprobs. Evaluate to force the graph nodes.
    let mut lp = s.logprobs.expect("collect_logprobs=true ⇒ Some(Array)");
    let _ = lp.to_vec::<f32>().unwrap();
  }

  // Run B: opt-out (default). The normalization is skipped, the squeeze
  // is skipped, every step yields `None` — the functional signal that the
  // logsumexp node (the only producer of Some(logprobs)) was never built.
  let model_b = MockModel::with_bias(bias.to_vec());
  let cfg_b = GenConfig {
    max_tokens,
    eos: vec![],
    // collect_logprobs: false (default)
    ..GenConfig::default()
  };
  let mut tokens_b: Vec<u32> = Vec::new();
  for step in generate_step(&model_b, &prompt, cache(1), cfg_b) {
    let s = step.unwrap();
    assert!(
      s.logprobs.is_none(),
      "collect_logprobs=false ⇒ GenStep.logprobs is None (logsumexp node skipped)"
    );
    tokens_b.push(s.token);
  }

  // Behavioural: same tokens (argmax is shift-invariant; the gate did not
  // change sampling). bias[2] == 5.0 ⇒ argmax == 2 every step.
  assert_eq!(tokens_a.len(), max_tokens);
  assert_eq!(tokens_b.len(), max_tokens);
  for (a, b) in tokens_a.iter().zip(tokens_b.iter()) {
    assert_eq!(
      a, b,
      "opt-out path must sample the same token as opt-in (argmax is shift-invariant)"
    );
    assert_eq!(*a, 2, "argmax(bias) == 2 (bias[2] == 5.0)");
  }
}

/// L3 normalization gate respects `top_p` (the only sampler in
/// [`make_sampler`] that strictly requires normalized log-probs — its
/// `exp(logprobs)` cumsum assumes a `1.0` total). With `top_p ∈ (0, 1)`
/// AND `collect_logprobs == false`, the gate MUST still run the
/// normalization so the sampler reads true log-probs (otherwise top_p's
/// `1 - top_p` cumulative threshold is on raw `exp(logits)` and the
/// nucleus cutoff is wrong). The observable proof: the sampled token is
/// the same as the `collect_logprobs == true` run on the same `top_p`
/// config — i.e. the sampler did NOT silently regress to a raw-logit
/// input.
#[test]
fn generate_step_top_p_forces_normalization_even_when_off() {
  // bias[2] is the dominant token (5.0 vs 1-4.0 others), so under any
  // reasonable top_p the nucleus contains it; with `temp` small enough
  // the categorical draw concentrates on token 2.
  let bias = [1.0_f32, 2.0, 5.0, 3.0, 4.0];
  let model_a = MockModel::with_bias(bias.to_vec());
  // Seed the stochastic sampler so the two runs draw the SAME PRNG
  // stream — any token divergence would then be a normalization bug, not
  // PRNG drift.
  let cfg_a = GenConfig {
    max_tokens: 4,
    eos: vec![],
    temp: 0.1,  // small temp → very concentrated draw
    top_p: 0.9, // forces the normalization gate
    seed: Some(42),
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let steps_a: Vec<u32> = generate_step(&model_a, &[1u32], cache(1), cfg_a)
    .map(|r| r.unwrap().token)
    .collect();

  let model_b = MockModel::with_bias(bias.to_vec());
  let cfg_b = GenConfig {
    max_tokens: 4,
    eos: vec![],
    temp: 0.1,
    top_p: 0.9,
    seed: Some(42),
    // collect_logprobs: false (default) — the gate must STILL run the
    // normalization because top_p is enabled.
    ..GenConfig::default()
  };
  let steps_b: Vec<u32> = generate_step(&model_b, &[1u32], cache(1), cfg_b)
    .map(|r| r.unwrap().token)
    .collect();

  assert_eq!(
    steps_a, steps_b,
    "top_p must force normalization even with collect_logprobs=false — \
     same PRNG seed ⇒ identical token stream"
  );
}

/// L3 zero-cost opt-out — stochastic max-shift numerical safety (Codex
/// review R2). The opt-out path used to feed RAW post-processor logits to
/// `categorical_sampling`, which multiplies by `1/temp` BEFORE the eventual
/// internal `softmax`. With a large `logit_bias` (e.g. `+50`) and a small
/// `temp` (e.g. `0.1` ⇒ `1/temp = 10`), the scaled logit reaches `+500`,
/// which overflows to `+inf` in f16/bf16 long before shift-invariance can
/// save us. The fix applies a cheap `logits - max(logits, keepdims=True)`
/// max-shift in the opt-out path when `temp > 0`, capping the input at 0 so
/// `exp` is bounded for every dtype.
///
/// The observable proof: the same fixed-seed stochastic config — large
/// positive bias on one entry, small `temp` — must produce the same token
/// stream whether `collect_logprobs == true` (full normalization runs) or
/// `collect_logprobs == false` (only the cheap max-shift runs). Before the
/// fix, the opt-out path's `1/temp` scaling on the raw `+50` bias would
/// reach `+inf` after the upstream cast paths normalize through the
/// graph; with the max-shift the input is always ≤ 0 and the
/// `categorical_sampling` softmax is stable.
#[test]
fn generate_step_opt_out_max_shift_stable_with_large_bias() {
  // Vocab 5; bias = [+50 on entry 0, near-zero elsewhere] ⇒ entry 0
  // dominates regardless of normalization or shift, so the same seed must
  // sample the same token stream in both runs.
  let bias = vec![50.0_f32, 0.1, 0.0, -0.1, 0.2];
  let prompt = [1u32];
  let max_tokens = 4;
  let cfg_base = || GenConfig {
    max_tokens,
    eos: vec![],
    temp: 0.1,
    seed: Some(42),
    // logit_bias adds another +50 on entry 0 ON TOP of the per-vocab bias
    // — drives the scaled-by-1/temp value well past any f16/bf16 finite
    // range. The fix's max-shift caps it back to 0.
    logit_bias: vec![(0, 50.0)],
    ..GenConfig::default()
  };

  // Run A: full normalization (collect_logprobs=true ⇒ logsumexp +
  // subtract). The reference token stream — the sampler reads true
  // log-probs, no overflow risk.
  let model_a = MockModel::with_bias(bias.clone());
  let cfg_a = GenConfig {
    collect_logprobs: true,
    ..cfg_base()
  };
  let tokens_a: Vec<u32> = generate_step(&model_a, &prompt, cache(1), cfg_a)
    .map(|r| r.unwrap().token)
    .collect();

  // Run B: opt-out (default collect_logprobs=false). With the R2 fix the
  // sampler input is `logits - max(logits)` ⇒ same argmax as full
  // normalization, same stable softmax in `categorical_sampling`.
  let model_b = MockModel::with_bias(bias);
  let cfg_b = cfg_base(); // collect_logprobs: false (default)
  let tokens_b: Vec<u32> = generate_step(&model_b, &prompt, cache(1), cfg_b)
    .map(|r| r.unwrap().token)
    .collect();

  assert_eq!(tokens_a.len(), max_tokens);
  assert_eq!(tokens_b.len(), max_tokens);
  assert_eq!(
    tokens_a, tokens_b,
    "max-shift opt-out must sample the same token stream as the full-normalization \
     reference under large logit_bias + small temp (no +inf overflow in the \
     `1/temp` multiply that would scramble `categorical_sampling`)"
  );
  // Every sampled token must be the dominant entry — proves the sampler
  // didn't degenerate to a NaN/uniform draw under the overflow regime.
  for t in &tokens_a {
    assert_eq!(*t, 0, "biased entry must dominate every step");
  }
}

/// L3 zero-cost opt-out — pure-greedy (`temp == 0`) still feeds RAW logits
/// (no max-shift), matching the documented "true zero-cost path" for
/// `argmax_sample`. The sampled token must be byte-identical to the
/// `collect_logprobs == true` (full normalization) reference, since
/// `argmax` is shift-invariant numerically as well as mathematically (it
/// doesn't exponentiate anything).
#[test]
fn generate_step_opt_out_greedy_zero_temp() {
  let bias = vec![0.0_f32, 5.0, 2.0, 3.0, 1.0]; // argmax == 1
  let prompt = [1u32];
  let max_tokens = 4;

  // Run A: opt-in (full normalization).
  let model_a = MockModel::with_bias(bias.clone());
  let cfg_a = GenConfig {
    max_tokens,
    eos: vec![],
    temp: 0.0, // greedy
    collect_logprobs: true,
    ..GenConfig::default()
  };
  let tokens_a: Vec<u32> = generate_step(&model_a, &prompt, cache(1), cfg_a)
    .map(|r| r.unwrap().token)
    .collect();

  // Run B: opt-out (no normalization, no max-shift — pure-greedy is the
  // `(false, false)` arm of the 3-way match: raw logits straight to
  // argmax).
  let model_b = MockModel::with_bias(bias);
  let cfg_b = GenConfig {
    max_tokens,
    eos: vec![],
    temp: 0.0, // greedy ⇒ argmax_sample, shift-invariant numerically
    // collect_logprobs: false (default)
    ..GenConfig::default()
  };
  let tokens_b: Vec<u32> = generate_step(&model_b, &prompt, cache(1), cfg_b)
    .map(|r| r.unwrap().token)
    .collect();

  assert_eq!(
    tokens_a, tokens_b,
    "greedy opt-out must be byte-identical to opt-in"
  );
  for t in &tokens_a {
    assert_eq!(*t, 1, "argmax(bias) == 1 (bias[1] == 5.0)");
  }
}

/// `generate` with `max_tokens == 0` produces no tokens; the returned
/// `GenerationStats` carries zero-counts + a zero tps + the original
/// `prompt_tokens`.
#[test]
fn generate_zero_max_tokens_stats() {
  let tok = tokenizer("gen_zero");
  let model = MockModel::ramp(5);
  let cfg = GenConfig {
    max_tokens: 0,
    eos: tok.eos_token_ids().iter().copied().collect(),
    ..GenConfig::default()
  };
  let (text, stats) = generate(&model, &tok, &[3u32, 5], cache(1), cfg).unwrap();
  assert_eq!(text, "", "no tokens produced ⇒ empty output");
  assert_eq!(stats.prompt_tokens, 2);
  assert_eq!(stats.generation_tokens, 0);
  assert_eq!(stats.prompt_tps, 0.0);
  assert_eq!(stats.generation_tps, 0.0);
}
