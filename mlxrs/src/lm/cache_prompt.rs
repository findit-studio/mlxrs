//! Prompt-cache fill + save driver, ported from
//! [`mlx_lm.cache_prompt`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py)
//! (`main`, the `--prompt-cache-file` CLI). mlx-swift-lm has **no** standalone
//! equivalent (its `ChatSession.saveCache(to:)` /
//! `MLXLMCommon/KVCache.swift::savePromptCache` cover the same "prefill a
//! shared context once, persist it, restore later" prefix-caching idea but
//! ship no separate driver), so `cache_prompt.py` is the authoritative
//! reference.
//!
//! The driver is the small piece that ties **tokenize → prefill → persist**
//! together: it encodes a prompt, runs a **prefill-only** forward over the
//! full prompt to populate the per-layer KV `cache` (no sampling, no token
//! generation), then writes the populated cache plus metadata to disk via the
//! existing [`crate::lm::cache::save_prompt_cache`]. It is the **support
//! surface**, not the CLI (`argparse` / stdin / progress printing /
//! `mx.get_peak_memory` are CLI concerns, intentionally omitted — exactly the
//! "port the driver, not the CLI" scope).
//!
//! ## What it reuses (no reimplementation)
//!
//! - **Persist:** the save is [`crate::lm::cache::save_prompt_cache`] verbatim
//!   (the #22/#31 wire format), so a cache written here loads back through the
//!   matching [`crate::lm::cache::load_prompt_cache`] and interoperates with
//!   mlx-lm exactly as that module documents.
//! - **Forward:** the prefill calls only [`Model::forward`] — the same
//!   architecture-agnostic seam [`crate::lm::generate`] drives. The cache is
//!   advanced in place by the model's attention blocks; the driver never
//!   reaches into a concrete cache type.
//!
//! ## The prefill (and the deliberate `max_tokens == 0` divergence)
//!
//! `cache_prompt.py` fills the cache by running
//! `generate_step(y, model, max_tokens=0, prompt_cache=cache, …)` and
//! discarding every step (the `for _ in …: pass` loop). In mlx-lm,
//! `generate_step` first prefills the prompt's leading `total - 1` tokens
//! (chunked by `prefill_step_size`), **then** runs the first `_step` over the
//! final token *before* the `while True` loop's `max_tokens` check — so with
//! `max_tokens == 0` the whole prompt (all `P` tokens) lands in the cache and
//! nothing is yielded (cache.py drives this exactly).
//!
//! mlxrs's [`crate::lm::generate::generate_step`] checks `produced >=
//! max_tokens` **before** prefill on the first `next()`, so consuming it with
//! `max_tokens == 0` would do *no* forward at all and save an **empty** cache.
//! This driver therefore does **not** route through `generate_step`; it runs
//! the same forward sequence `generate_step(max_tokens=0)` performs in mlx-lm
//! directly — chunk the leading `P - 1` tokens, then one forward over the last
//! token — leaving the cache offset at `P`, byte-identical to the reference
//! (and to a `generate_step` decode that *did* prefill). No sampler, no
//! logits-processors, no `logprobs` — none of which `cache_prompt` needs.

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{Error, Result, try_with_capacity},
  lm::{
    cache::{KvCache, save_prompt_cache},
    model::Model,
  },
  tokenizer::Tokenizer,
};

/// Metadata key for the model identity string — mlx-lm
/// `metadata["model"] = args.model` (cache_prompt.py:143).
pub const META_MODEL: &str = "model";

/// Metadata key for the serialized tokenizer config — mlx-lm
/// `metadata["tokenizer_config"] = json.dumps(tokenizer_config)`
/// (cache_prompt.py:144).
pub const META_TOKENIZER_CONFIG: &str = "tokenizer_config";

/// Summary of a [`cache_prompt`] run.
///
/// `cache_prompt.py` prints (`Processed {processed} tokens`) but persists
/// only `model` / `tokenizer_config` in the on-disk metadata — the processed
/// count is recoverable from the saved cache's offset, not a wire field. This
/// struct returns it to the caller (the reference's printed `processed`)
/// without changing the persist wire format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePromptInfo {
  /// Number of prompt tokens processed into the cache — the full encoded
  /// prompt length (mlx-lm's `total_prompt_tokens`, the cache's final
  /// offset). `0` only for the empty-prompt error path (which returns `Err`
  /// before producing this).
  pub tokens_processed: usize,
}

/// Encode `prompt` the way `cache_prompt.py` does (cache_prompt.py:100-109):
/// the chat template when the tokenizer has one (`add_generation_prompt=False,
/// continue_final_message=True` — a single `user` message), else the plain
/// [`Tokenizer::encode`].
///
/// `continue_final_message=True` has no first-class flag on
/// [`Tokenizer::apply_chat_template_ids`]; it is faithfully reproduced by
/// appending the prompt as the final message with **no** generation prompt
/// (the cache should end exactly at the prompt's last token, ready for a
/// later continuation — not after an injected assistant-turn opener).
fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
  if tokenizer.has_chat_template() {
    let messages = serde_json::json!([{ "role": "user", "content": prompt }]);
    let ids = tokenizer
      .apply_chat_template_ids(&messages, None, false, None)
      .map_err(|e| Error::Backend {
        message: format!("cache_prompt: apply_chat_template failed: {e}"),
      })?;
    Ok(ids)
  } else {
    // mlx-lm `tokenizer.encode(args.prompt)` — the tokenizer's default
    // special-token handling (transformers `encode` adds specials).
    tokenizer.encode(prompt, true).map_err(|e| Error::Backend {
      message: format!("cache_prompt: encode failed: {e}"),
    })
  }
}

/// Build a `[1, S]` `I32` token window from `ids` (mlx-lm's `prompt[:n][None]`
/// / `input_tokens[None]`). `I32` is mlx's default integer dtype for token ids
/// (embedding `take` indices); the [`Model`] trait only constrains the shape.
/// Mirrors [`crate::lm::generate`]'s identical private `token_window`, kept
/// local so the driver depends only on the public [`Model::forward`] seam.
fn token_window(ids: &[u32]) -> Result<Array> {
  let mut row: Vec<i32> = try_with_capacity(ids.len())?;
  row.extend(ids.iter().map(|&t| t as i32));
  Array::from_slice::<i32>(&row, &(1usize, row.len()))
}

/// Run a **prefill-only** forward over the full encoded `prompt`, advancing
/// `cache` in place — the exact forward sequence mlx-lm's
/// `generate_step(max_tokens=0)` performs (the prompt-fill `cache_prompt.py`
/// relies on), minus all sampling.
///
/// mlx-lm prefills the leading `total - 1` tokens in `prefill_step_size`
/// chunks (`generate.py:430-451`, logits discarded) and then forwards the
/// final token in the first `_step` (`generate.py:454`) — together the whole
/// prompt. This reproduces that precisely: the same chunk boundaries for the
/// first `P - 1` tokens, then a final 1-token forward. The result is a cache
/// at offset `P`, byte-identical to a `generate_step` run that prefilled the
/// same prompt. No `logits` are kept (every `forward` return is dropped — the
/// chunk only fills the cache, no implicit eval beyond what `forward` itself
/// does).
///
/// `prefill_step_size` is clamped to `>= 1` (a `0` would not make progress),
/// matching [`crate::lm::generate::generate_step`]'s `prefill_step_size.max(1)`.
fn prefill_full<M: Model>(
  model: &M,
  prompt: &[u32],
  cache: &mut [Box<dyn KvCache>],
  prefill_step_size: usize,
) -> Result<()> {
  let step = prefill_step_size.max(1);
  // mlx-lm: `while total - processed > 1: n = min(step, (total-processed)-1);
  // forward(prompt[:n]); processed += n`. Advance a cursor (never
  // front-drain) for O(P) with byte-identical chunk boundaries.
  let mut processed = 0usize;
  while prompt.len() - processed > 1 {
    let remaining = (prompt.len() - processed) - 1;
    let n = step.min(remaining);
    let chunk = token_window(&prompt[processed..processed + n])?;
    // logits discarded — the chunk only fills the cache.
    let _ = model.forward(&chunk, cache)?;
    processed += n;
  }
  // mlx-lm `_step(input_tokens=prompt)` over the final unconsumed token: the
  // same forward, just without sampling its logits. After the loop exactly
  // one token remains (`prompt[processed..]`).
  let tail = token_window(&prompt[processed..])?;
  let _ = model.forward(&tail, cache)?;
  Ok(())
}

/// Tokenize `prompt`, prefill `cache` with it, and save the populated cache
/// (plus metadata) to `out_path` — the support-surface port of
/// `mlx_lm.cache_prompt.main` (cache_prompt.py:83-145), minus the CLI.
///
/// Mirrors the reference end to end:
///
/// 1. **Tokenize** via `encode_prompt` (chat template when present, else
///    [`Tokenizer::encode`]) — cache_prompt.py:100-109.
/// 2. **Prefill** the full prompt into `cache` via `prefill_full` (the
///    `generate_step(max_tokens=0)` forward sequence; no sampling) —
///    cache_prompt.py:111-136. The empty-prompt case is rejected up front as
///    a recoverable [`Error::Backend`] (mlx-lm's `generate_step` raises
///    `ValueError` on an empty prompt; a prefill over zero tokens would be a
///    no-op saving an empty cache, so failing fast is the faithful behavior).
/// 3. **Save** the cache via [`crate::lm::cache::save_prompt_cache`] with the
///    metadata cache_prompt.py writes — `metadata["model"] = model_id` and
///    `metadata["tokenizer_config"] = tokenizer_config_json`
///    (cache_prompt.py:142-145). `extra_metadata` lets a caller add further
///    keys (e.g. an explicit processed-count) without altering the wire
///    format; the two reference keys take precedence on collision.
///
/// `cache` is the caller-built per-layer KV cache
/// ([`crate::lm::cache::make_prompt_cache`]) the model mutates in place — it
/// must have one entry per decoder layer, exactly as the generation loop
/// requires. It is borrowed `&mut` (advanced to offset `P`) and then saved;
/// the caller still owns it afterwards (e.g. to keep generating from it).
///
/// Returns a [`CachePromptInfo`] with the number of tokens processed (the
/// reference's printed `processed` count / the cache's final offset). Any
/// failure — encode, a prefill `forward`, or the save I/O — is a recoverable
/// [`crate::Error`]; the driver never panics.
#[allow(clippy::too_many_arguments)]
pub fn cache_prompt<M: Model>(
  model: &M,
  tokenizer: &Tokenizer,
  prompt: &str,
  cache: &mut [Box<dyn KvCache>],
  out_path: &std::path::Path,
  model_id: &str,
  tokenizer_config_json: &str,
  prefill_step_size: usize,
  extra_metadata: &HashMap<String, String>,
) -> Result<CachePromptInfo> {
  // 1. tokenize (cache_prompt.py:100-109).
  let prompt_ids = encode_prompt(tokenizer, prompt)?;
  cache_prompt_ids(
    model,
    &prompt_ids,
    cache,
    out_path,
    model_id,
    tokenizer_config_json,
    prefill_step_size,
    extra_metadata,
  )
}

/// [`cache_prompt`] over a pre-encoded prompt — the lower-level entry that
/// skips the tokenizer (mirroring how [`crate::lm::generate::generate_step`]
/// takes already-encoded ids, so a caller that has tokenized once need not
/// re-encode, and tests can drive the prefill+save without a `Tokenizer`).
///
/// Performs steps 2-3 of [`cache_prompt`]: prefill the full `prompt_ids` into
/// `cache` (the `generate_step(max_tokens=0)` forward sequence), then save via
/// [`crate::lm::cache::save_prompt_cache`] with the `model` /
/// `tokenizer_config` metadata cache_prompt.py writes (plus `extra_metadata`).
/// An empty `prompt_ids` is a recoverable [`Error::Backend`] (faithful to
/// mlx-lm's empty-prompt `ValueError`); nothing is written in that case.
#[allow(clippy::too_many_arguments)]
pub fn cache_prompt_ids<M: Model>(
  model: &M,
  prompt_ids: &[u32],
  cache: &mut [Box<dyn KvCache>],
  out_path: &std::path::Path,
  model_id: &str,
  tokenizer_config_json: &str,
  prefill_step_size: usize,
  extra_metadata: &HashMap<String, String>,
) -> Result<CachePromptInfo> {
  // mlx-lm `generate_step` raises `ValueError` on an empty prompt; a prefill
  // over zero tokens would silently save an empty cache, so reject it (the
  // save below never runs, so no file is written).
  if prompt_ids.is_empty() {
    return Err(Error::Backend {
      message:
        "cache_prompt: prompt must be non-empty (mlx-lm raises ValueError on an empty prompt)"
          .into(),
    });
  }

  // 2. prefill the full prompt into the cache (cache_prompt.py:111-136).
  prefill_full(model, prompt_ids, cache, prefill_step_size)?;

  // 3. save with the reference metadata (cache_prompt.py:142-145).
  // Build `extra` first so the two reference keys (`model` /
  // `tokenizer_config`) deterministically win on collision — mirroring
  // cache_prompt.py, which sets them last/unconditionally.
  let mut metadata: HashMap<String, String> = extra_metadata.clone();
  metadata.insert(META_MODEL.to_string(), model_id.to_string());
  metadata.insert(
    META_TOKENIZER_CONFIG.to_string(),
    tokenizer_config_json.to_string(),
  );
  save_prompt_cache(out_path, cache, &metadata)?;

  Ok(CachePromptInfo {
    tokens_processed: prompt_ids.len(),
  })
}

#[cfg(test)]
mod tests {
  //! In-crate prefill-boundary unit tests for the driver, sharing the
  //! `model::MockModel` fixture (`#[cfg(test)] pub(crate)`, visible here).
  //! The cross-tool round-trip (fill → save → load → continue) lives in the
  //! integration test `tests/lm_cache_prompt_driver.rs`, which exercises the
  //! public `save_prompt_cache`/`load_prompt_cache`.

  use super::*;
  use crate::lm::{
    cache::{CacheConfig, make_prompt_cache},
    model::MockModel,
  };

  fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
    make_prompt_cache(&CacheConfig {
      num_hidden_layers: layers,
      sliding_window: None,
    })
  }

  /// `prefill_full` advances every layer's cache to exactly the prompt length
  /// (offset `P`), regardless of the chunk size — the whole prompt lands in
  /// the cache (the `generate_step(max_tokens=0)` fill contract).
  #[test]
  fn prefill_full_fills_cache_to_prompt_len() {
    let model = MockModel::new(5);
    let prompt = [1u32, 2, 3, 4, 5, 6, 7];
    // A small chunk so the leading P-1 loop runs multiple chunks + the tail.
    let mut c = cache(2);
    prefill_full(&model, &prompt, &mut c, 3).unwrap();
    assert!(
      c.iter().all(|x| x.offset() == prompt.len()),
      "every layer cache must be at offset P after a full prefill"
    );
  }

  /// The chunk boundaries are byte-identical to mlx-lm's
  /// `generate_step(max_tokens=0)`: the leading `P-1` tokens in
  /// `prefill_step_size` chunks, then a final 1-token forward. A
  /// seq-len-recording model pins the exact `forward` window sequence.
  #[test]
  fn prefill_full_chunk_boundaries_match_reference() {
    use std::cell::RefCell;
    struct Recorder {
      bias: Vec<f32>,
      seq_lens: RefCell<Vec<usize>>,
    }
    impl Model for Recorder {
      fn forward(&self, tokens: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
        let s = match tokens.shape().as_slice() {
          [_, s] => *s,
          [s] => *s,
          other => {
            return Err(Error::ShapeMismatch {
              message: format!("Recorder: expected [B,S], got {other:?}"),
            });
          }
        };
        self.seq_lens.borrow_mut().push(s);
        let vocab = self.bias.len();
        let mut data = Vec::with_capacity(s * vocab);
        for _ in 0..s {
          data.extend_from_slice(&self.bias);
        }
        Array::from_slice::<f32>(&data, &(1usize, s, vocab))
      }
    }
    let model = Recorder {
      bias: vec![0.0, 1.0, 2.0],
      seq_lens: RefCell::new(Vec::new()),
    };
    // P = 7, step = 3 ⇒ leading P-1 = 6 tokens as chunks [3, 3], then the
    // final token as [1]. Exactly mlx-lm's prefill loop + first `_step`.
    let prompt = [1u32, 2, 1, 2, 1, 2, 1];
    let mut c: Vec<Box<dyn KvCache>> = Vec::new();
    prefill_full(&model, &prompt, &mut c, 3).unwrap();
    assert_eq!(model.seq_lens.into_inner(), vec![3, 3, 1]);
  }

  /// A single-token prompt: the leading-`P-1` loop never runs (0 tokens), and
  /// only the final 1-token forward fires — cache ends at offset 1. Uses the
  /// cache-advancing [`MockModel`] (`forward` updates each layer) so the
  /// offset is observable; the count side is pinned by the dedicated
  /// chunk-boundary test above.
  #[test]
  fn prefill_full_single_token_prompt() {
    let model = MockModel::new(4);
    let mut c = cache(1);
    prefill_full(&model, &[42u32], &mut c, 8).unwrap();
    assert!(
      c.iter().all(|x| x.offset() == 1),
      "a 1-token prompt fills the cache to offset 1 via the single tail forward"
    );
  }

  /// A `prefill_step_size` of `0` is clamped to `1` (still makes progress),
  /// mirroring `generate_step`'s `prefill_step_size.max(1)`.
  #[test]
  fn prefill_full_zero_step_is_clamped() {
    let model = MockModel::new(4);
    let mut c = cache(1);
    prefill_full(&model, &[1u32, 2, 3], &mut c, 0).unwrap();
    assert!(c.iter().all(|x| x.offset() == 3));
  }

  /// `cache_prompt_ids` over an empty prompt is a recoverable `Err` and
  /// writes no file (faithful to mlx-lm's empty-prompt `ValueError`).
  #[test]
  fn cache_prompt_ids_empty_prompt_errors_without_writing() {
    let model = MockModel::new(4);
    let mut c = cache(1);
    let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_empty_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join("empty.safetensors");
    let _ = std::fs::remove_file(&out);
    let r = cache_prompt_ids(&model, &[], &mut c, &out, "mock", "{}", 8, &HashMap::new());
    assert!(r.is_err(), "empty prompt must error");
    assert!(
      !out.exists(),
      "no cache file written on the empty-prompt error"
    );
  }
}
