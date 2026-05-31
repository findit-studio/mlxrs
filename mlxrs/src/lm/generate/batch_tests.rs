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

  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];

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

  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];

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
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
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
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
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
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
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
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
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

// ════════════════════════════════════════════════════════════════════════
//   left_pad_rows error paths + batch_generate / batch_stream_generate
// ════════════════════════════════════════════════════════════════════════

/// `left_pad_rows` rejects an empty `prompts` slice and an all-empty batch
/// (`max_len == 0`) up front with `EmptyInput`.
#[test]
fn left_pad_rows_rejects_empty_inputs() {
  let empty: Vec<&[u32]> = vec![];
  assert!(matches!(
    left_pad_rows(&empty, 0),
    Err(Error::EmptyInput(_))
  ));
  // Non-empty slice but every row empty ⇒ max_len == 0 ⇒ EmptyInput.
  let all_empty: Vec<&[u32]> = vec![&[], &[]];
  assert!(matches!(
    left_pad_rows(&all_empty, 0),
    Err(Error::EmptyInput(_))
  ));
}

/// `left_pad_rows` rejects a RAGGED batch where `max_len > 0` but one row is
/// empty — the per-row empty check fires AFTER the `max_len == 0` guard, so
/// `[1,2]` + `[]` reaches the per-row branch and errs with `EmptyInput`.
#[test]
fn left_pad_rows_rejects_ragged_empty_row() {
  let ragged: Vec<&[u32]> = vec![&[1u32, 2], &[]];
  let err = left_pad_rows(&ragged, 0).unwrap_err();
  assert!(
    matches!(err, Error::EmptyInput(ref p) if p.context().contains("every prompt")),
    "a ragged empty row ⇒ EmptyInput(every prompt), got {err:?}"
  );
}

/// `left_pad_rows` left-pads shorter rows with `pad_token_id` to `max_len`,
/// preserving each row's tail. Closed-form oracle: `[1,2,3]` + `[7]` with
/// pad=99 ⇒ `[[1,2,3],[99,99,7]]`, max_len=3.
#[test]
fn left_pad_rows_pads_and_preserves_tail() {
  let prompts: Vec<&[u32]> = vec![&[1u32, 2, 3], &[7u32]];
  let (padded, max_len) = left_pad_rows(&prompts, 99).unwrap();
  assert_eq!(max_len, 3);
  assert_eq!(padded, vec![vec![1, 2, 3], vec![99, 99, 7]]);
}

/// Resolve the committed fixture tokenizer (`</s>` == id 2 ⇒ eos set {2}).
fn fixture_tokenizer() -> crate::tokenizer::Tokenizer {
  let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures");
  crate::tokenizer::Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
}

/// `batch_generate` aggregates per-row tokens: EOS-finish tokens are DROPPED,
/// `length`-finish + in-progress tokens are KEPT (mlx-lm `batch_generate`
/// `generate.py:1945-1946`). The tokenizer's eos set ({2}) overrides cfg.eos.
///
/// Equal-length prompts ⇒ left_pad `[0, 0]`, trivial prefill. Row 0 scripts
/// `[7, 2]` (token 7, then eos) ⇒ output `[7]` (eos dropped). Row 1 scripts
/// `[8, 9, 10]` and runs to `max_tokens = 3` ⇒ output `[8, 9, 10]` (the
/// length-finish token 10 is kept).
#[test]
fn batch_generate_drops_eos_keeps_length_tokens() {
  let tok = fixture_tokenizer();
  let prompts: Vec<&[u32]> = vec![&[1u32, 1], &[1u32, 1]];
  let left_pad = batch_left_padding(&prompts);
  assert_eq!(left_pad, vec![0, 0]);
  let max_len = 2;
  let scripts = vec![
    vec![7u32, 2, 99, 99], // token 7 then eos(2)
    vec![8u32, 9, 10, 99], // runs to max_tokens
  ];
  let model = MockBatchModel::new(32, max_len, scripts);
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
  let cfg = GenConfig {
    max_tokens: 3,
    ..Default::default()
  };

  let out = batch_generate(&model, &tok, &prompts, 0, cache, cfg).expect("batch_generate ok");
  assert_eq!(out.len(), 2, "one output row per prompt");
  assert_eq!(out[0], vec![7], "row 0: token 7 kept, eos(2) dropped");
  assert_eq!(out[1], vec![8, 9, 10], "row 1: length-finish token kept");
}

/// `batch_generate` with `max_tokens == 0` returns an empty `Vec` per row
/// (the zero-budget guard fires before any step). Independent of the model
/// script (no decode runs).
#[test]
fn batch_generate_zero_max_tokens_empty_rows() {
  let tok = fixture_tokenizer();
  let prompts: Vec<&[u32]> = vec![&[1u32, 1], &[1u32, 1], &[1u32, 1]];
  let left_pad = batch_left_padding(&prompts);
  let model = MockBatchModel::new(16, 2, vec![vec![], vec![], vec![]]);
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
  let cfg = GenConfig {
    max_tokens: 0,
    ..Default::default()
  };
  let out = batch_generate(&model, &tok, &prompts, 0, cache, cfg).unwrap();
  assert_eq!(out, vec![Vec::<u32>::new(); 3]);
}

/// `batch_stream_generate` overrides `cfg.eos` with the tokenizer's set
/// ({2}) before constructing the iterator, so a row scripting token 2
/// terminates with `FinishReason::Eos` even when `cfg.eos` was left empty.
#[test]
fn batch_stream_generate_uses_tokenizer_eos() {
  let tok = fixture_tokenizer();
  let prompts: Vec<&[u32]> = vec![&[1u32, 1]];
  let left_pad = batch_left_padding(&prompts);
  let model = MockBatchModel::new(16, 2, vec![vec![5u32, 2, 99]]); // token 5 then eos(2)
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
  // cfg.eos intentionally empty — the tokenizer's {2} must take over.
  let cfg = GenConfig {
    max_tokens: 5,
    ..Default::default()
  };

  let mut last_reason: Option<FinishReason> = None;
  let mut tokens = Vec::new();
  for item in batch_stream_generate(&model, &tok, &prompts, 0, cache, cfg) {
    let step = item.expect("step ok");
    match &step.finish_reason {
      Some(r) if r.is_eos() => last_reason = step.finish_reason.clone(),
      _ => tokens.push(step.token),
    }
  }
  assert_eq!(tokens, vec![5], "token 5 emitted before eos");
  assert_eq!(
    last_reason,
    Some(FinishReason::Eos),
    "tokenizer eos {{2}} drove the stop even with empty cfg.eos"
  );
}

/// A batch run WITH a logits processor (repetition penalty) exercises the
/// per-row slice → process → concat branch in `BatchGenerator::step`. The
/// scripted tokens are all distinct per step, so the penalty never down-
/// weights the (fresh) argmax token — the output stays exactly the script,
/// giving a deterministic oracle while still driving the processor code path.
#[test]
fn batch_generate_with_repetition_penalty_runs_per_row_processor() {
  let tok = fixture_tokenizer();
  let prompts: Vec<&[u32]> = vec![&[1u32, 1], &[1u32, 1]];
  let left_pad = batch_left_padding(&prompts);
  let max_len = 2;
  // Distinct tokens per step per row ⇒ no repeats ⇒ penalty is a no-op on
  // the argmax, so the script is reproduced exactly.
  let scripts = vec![vec![10u32, 11, 12, 99], vec![20u32, 21, 22, 99]];
  let model = MockBatchModel::new(32, max_len, scripts);
  let cache: Vec<Box<dyn crate::lm::cache::KvCache>> = vec![Box::new(BatchKvCache::new(&left_pad))];
  let cfg = GenConfig {
    max_tokens: 3,
    repetition_penalty: Some(2.0), // ⇒ make_logits_processors yields 1 processor
    ..Default::default()
  };

  let out = batch_generate(&model, &tok, &prompts, 0, cache, cfg).expect("ok");
  assert_eq!(
    out[0],
    vec![10, 11, 12],
    "row 0 unaffected by no-repeat penalty"
  );
  assert_eq!(
    out[1],
    vec![20, 21, 22],
    "row 1 unaffected by no-repeat penalty"
  );
}
