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

use std::{
  collections::HashMap,
  fs,
  path::{Path, PathBuf},
};

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
/// the chat template when the tokenizer has one
/// (`add_generation_prompt=False, continue_final_message=True` — a single
/// `user` message), else the plain [`Tokenizer::encode`].
///
/// `continue_final_message=true` is passed through to
/// [`Tokenizer::apply_chat_template_ids`]'s first-class flag, which ports HF
/// Transformers' post-render trim: the rendered prompt ends exactly at the
/// final message's content, with the trailing end-of-turn / EOS tokens the
/// template would otherwise append stripped — so the cache offset matches
/// mlx-lm's exactly (the cache must end at the prompt's last *content* token,
/// ready for a later continuation, not after an injected turn terminator).
fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
  if tokenizer.has_chat_template() {
    let messages = serde_json::json!([{ "role": "user", "content": prompt }]);
    let ids = tokenizer
      .apply_chat_template_ids(&messages, None, false, true, None)
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

/// Force-evaluate every per-layer cache's **own stored arrays in place** —
/// the prefill memory-barrier mlx-lm runs after every prompt chunk
/// (`generate.py:442`: `mx.eval([c.state for c in prompt_cache])`).
///
/// `mlxrs::Array` is lazy (an op only records a graph node), so without this
/// each prefill chunk's `forward` would *append* to a graph spanning every
/// prior chunk and nothing would materialize until the final save — making
/// peak memory grow with the whole prompt and defeating `prefill_step_size`
/// (a long prompt could OOM/abort at the end).
///
/// This drives the [`KvCache::materialize`] hook on every layer (`&mut` via
/// [`slice::iter_mut`]), which evals each cache's **genuine stored
/// `keys`/`values`** (and quantized triples / per-sequence position arrays /
/// SSM slots / child caches) directly. It deliberately does **not** route
/// through [`KvCache::state`]: a sliding-window / chunked / batched cache
/// over-allocates its ring/step buffer and `state()` returns
/// `seq_slice(self.keys, 0, offset)` serialization views whenever `offset <
/// buffer_len` (the regime an `S == 1` update reaches after growing the ring,
/// also the `prefill_step_size == 1` / `0`-clamp path) — evaluating those
/// temporary slices would materialize the slice's output buffer, not the
/// stored buffer the next chunk's `update` reads and extends, so the live
/// graph could still chain across chunks and peak memory would not be bounded
/// (the Codex finding this hook closes). Evaluating the live arrays via the
/// `&mut` hook is faithful to mlx-lm's `mx.eval([c.state ...])` (per-chunk
/// full materialization of the live cache) without that hazard.
///
/// mlxrs has no safe vector-eval wrapper (mlx-c's `mlx_eval(mlx_vector_array)`
/// is unbound here), so each cache's arrays are evaluated individually —
/// observably identical to a single `mx.eval` over the list (each array's
/// graph is forced to its buffer; order is irrelevant). An empty cache (one
/// that holds no arrays) is a no-op.
fn materialize_caches(cache: &mut [Box<dyn KvCache>]) -> Result<()> {
  for layer in cache.iter_mut() {
    layer.materialize()?;
  }
  Ok(())
}

/// Run a **prefill-only** forward over the full encoded `prompt`, advancing
/// `cache` in place — the exact forward sequence mlx-lm's
/// `generate_step(max_tokens=0)` performs (the prompt-fill `cache_prompt.py`
/// relies on), minus all sampling.
///
/// mlx-lm prefills the leading `total - 1` tokens in `prefill_step_size`
/// chunks (`generate.py:430-451`, logits discarded, **evaluating the cache
/// state after each chunk** at `generate.py:442`) and then forwards the final
/// token in the first `_step` (`generate.py:454`) — together the whole prompt.
/// This reproduces that precisely: the same chunk boundaries for the first
/// `P - 1` tokens with a per-chunk [`materialize_caches`] barrier, then a
/// final 1-token forward. The result is a cache at offset `P`, byte-identical
/// to a `generate_step` run that prefilled the same prompt. No `logits` are
/// kept (every `forward` return is dropped — the chunk only fills the cache).
///
/// ## Why the per-chunk barrier (Codex finding — memory-bounded prefill)
///
/// `mlxrs::Array` is lazy, so without [`materialize_caches`] the chunk loop
/// would accumulate a single lazy graph spanning **every** chunk and only
/// force it at the final save — `prefill_step_size` would bound nothing and a
/// long prompt could OOM/abort. Materializing each cache's live stored arrays
/// after each chunk (via the [`KvCache::materialize`] hook — **not** the
/// serializable `state()`, whose over-allocated-buffer slices would leave the
/// stored buffers lazy and chaining) caps the live graph to one chunk's work
/// (mlx-lm's exact discipline). The final tail token's forward is left to be
/// materialized by the save (mlx-lm `async_eval`s it rather than blocking) —
/// `save_prompt_cache` reads `state()` and writes it, forcing that last step.
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
  // forward(prompt[:n]); mx.eval([c.state ...]); processed += n`. Advance a
  // cursor (never front-drain) for O(P) with byte-identical chunk boundaries.
  let mut processed = 0usize;
  while prompt.len() - processed > 1 {
    let remaining = (prompt.len() - processed) - 1;
    let n = step.min(remaining);
    let chunk = token_window(&prompt[processed..processed + n])?;
    // logits discarded — the chunk only fills the cache.
    let _ = model.forward(&chunk, cache)?;
    // mlx-lm `generate.py:442`: materialize each cache's live stored arrays
    // so the lazy graph does not span every chunk (memory-bounded prefill).
    materialize_caches(cache)?;
    processed += n;
  }
  // mlx-lm `_step(input_tokens=prompt)` over the final unconsumed token: the
  // same forward, just without sampling its logits. After the loop exactly
  // one token remains (`prompt[processed..]`). mlx-lm `async_eval`s this last
  // step rather than blocking on it; here the subsequent `save_prompt_cache`
  // reads `state()` and materializes it, so no extra barrier is needed.
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
///    A non-fresh `cache` (any layer with cached tokens) is likewise rejected
///    up front — the prefill only appends, so a reused cache would persist a
///    `[stale + new]` prefix.
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
/// requires. It **must be fresh/empty** (every layer at `offset() == 0`): the
/// prefill only appends, and the persisted cache must represent exactly this
/// prompt, so a reused / pre-populated cache is rejected with a recoverable
/// [`Error::Backend`] before any prefill or save (a reused cache would
/// persist stale state and leak a prior request's context). It is borrowed
/// `&mut` (advanced to offset `P`) and then saved; the caller still owns it
/// afterwards (e.g. to keep generating from it).
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
///
/// **Precondition — `cache` must be fresh/empty.** The prefill only ever
/// *appends* to `cache`; the saved cache must represent *exactly this
/// prompt*. A reused / pre-populated cache (any layer with `offset() != 0`)
/// is rejected with a recoverable [`Error::Backend`] before the prefill runs
/// (so nothing is written) — saving a reused cache would persist a
/// `[stale + new]` prefix and leak a prior request's context. Pass a freshly
/// built [`crate::lm::cache::make_prompt_cache`] cache.
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

  // The cache MUST be fresh/empty: `cache_prompt` saves a cache representing
  // *exactly this prompt*, but `prefill_full` only ever *appends* (it never
  // resets the cache). A reused / pre-populated cache would have the new
  // prompt prefilled on top of its stale state, and the save would persist
  // the combined `[stale + new]` prefix while `tokens_processed` reports the
  // new count alone — the saved cache would not represent the requested
  // prompt (in a service, a later generation would be conditioned on a prior
  // request's context: a cross-request context leak). mlx-lm never hits this
  // because `cache_prompt.py` allocates the cache itself
  // (`make_prompt_cache(model)`); here the cache is caller-provided, so the
  // freshness precondition is enforced explicitly. Reject ANY layer that
  // holds cached tokens (`offset() != 0`) or already holds key buffers
  // (`!is_empty()`) — a genuinely fresh `make_prompt_cache` cache satisfies
  // both. Checked before `prefill_full`, so the save never runs and no file
  // is written. Mirrors the empty-prompt guard above.
  if let Some((layer_idx, stale)) = cache
    .iter()
    .enumerate()
    .find(|(_, layer)| layer.offset() != 0 || !layer.is_empty())
  {
    return Err(Error::Backend {
      message: format!(
        "cache_prompt: the provided cache must be fresh/empty — layer {layer_idx} already \
         holds {} cached token(s); cache_prompt saves a cache representing exactly this \
         prompt, so a reused cache would persist stale state. Pass a freshly built cache \
         (`make_prompt_cache`).",
        stale.offset()
      ),
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
  // Atomic save: write to a same-directory tempfile, fsync, then rename over
  // the destination — a crash / IO error mid-save never leaves a partial or
  // corrupt cache at `out_path` (Codex finding). Mirrors `audio::io::save_wav`.
  save_prompt_cache_atomic(out_path, cache, &metadata)?;

  Ok(CachePromptInfo {
    tokens_processed: prompt_ids.len(),
  })
}

/// The path `mlx_save_safetensors` actually writes for `path`: mlx core's
/// `save_safetensors(std::string, …)` appends `".safetensors"` unless the
/// path already ends with it (`mlx/io/safetensors.cpp`). The atomic save must
/// (a) make its tempfile end with `".safetensors"` so mlx writes *that exact*
/// tempfile (no surprise second extension), and (b) rename onto the SAME
/// effective path the direct `save_prompt_cache(path, …)` would have produced
/// — so behavior (including the auto-appended extension) is unchanged.
fn effective_safetensors_path(path: &Path) -> PathBuf {
  let ends_with_ext = path.as_os_str().to_string_lossy().ends_with(".safetensors");
  if ends_with_ext {
    path.to_path_buf()
  } else {
    let mut s = path.as_os_str().to_owned();
    s.push(".safetensors");
    PathBuf::from(s)
  }
}

/// Open an exclusively-created (`O_CREAT|O_EXCL`), randomized tempfile in the
/// SAME directory as `final_path`, of the form
/// `<file_name>.<pid>.<rand>.tmp.safetensors`. Returns the temp path; the
/// created [`fs::File`] is dropped immediately (mlx-c reopens the path to
/// write it) — the exclusive create guarantees the path is a regular file we
/// own (never an attacker-precreated symlink), so the subsequent mlx truncate
/// of the same path follows no symlink. The trailing `.safetensors` keeps mlx
/// from appending a second extension (see [`effective_safetensors_path`]).
/// Same-directory keeps the later [`fs::rename`] single-fs (atomic on
/// POSIX/Windows; a cross-fs rename silently degrades to copy+unlink, losing
/// atomicity). Mirrors `audio::io::save_wav`'s `open_excl_tempfile` discipline.
fn open_excl_temp_safetensors(final_path: &Path, max_retries: u32) -> Result<PathBuf> {
  use std::{
    fs::OpenOptions,
    io::ErrorKind,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
  };
  static COUNTER: AtomicU64 = AtomicU64::new(0);

  let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
  let file_name = final_path
    .file_name()
    .ok_or_else(|| Error::Backend {
      message: format!(
        "cache_prompt: destination {} has no file_name component",
        final_path.display()
      ),
    })?
    .to_string_lossy()
    .into_owned();
  let pid = std::process::id();
  let mut last_err: Option<std::io::Error> = None;
  for _ in 0..max_retries {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let rand = nanos ^ counter.rotate_left(17);
    // The trailing `.safetensors` is required so mlx-c writes this exact path
    // (it would otherwise append the extension).
    let candidate = parent.join(format!("{file_name}.{pid}.{rand:016x}.tmp.safetensors"));
    match OpenOptions::new()
      .write(true)
      .create_new(true)
      .open(&candidate)
    {
      Ok(file) => {
        // mlx-c reopens this path by name to write it; drop our handle now.
        drop(file);
        return Ok(candidate);
      }
      Err(e) if e.kind() == ErrorKind::AlreadyExists => {
        last_err = Some(e);
        continue;
      }
      Err(e) => {
        return Err(Error::Backend {
          message: format!(
            "cache_prompt: create_new tempfile {} failed: {e}",
            candidate.display()
          ),
        });
      }
    }
  }
  Err(Error::Backend {
    message: format!(
      "cache_prompt: exhausted {max_retries} tempfile retries (last error: {})",
      last_err
        .map(|e| e.to_string())
        .unwrap_or_else(|| "<none>".into())
    ),
  })
}

/// Atomically save `cache` (+ `metadata`) to `out_path` — the durable,
/// crash-safe form of [`crate::lm::cache::save_prompt_cache`].
///
/// `save_prompt_cache` calls `mlx_save_safetensors` straight onto the final
/// path (no temp / fsync / rename), so a crash or IO error mid-save would
/// leave a partial/corrupt `.safetensors` at the destination — clobbering a
/// previously valid cache (Codex finding). This mirrors `audio::io::save_wav`'s
/// atomic discipline: write to a same-directory `O_EXCL` tempfile, fsync it to
/// durable storage, restore the destination's prior permissions, then
/// `fs::rename` it over the destination (atomic-within-fs). On ANY failure the
/// tempfile is removed (best-effort) and the destination is left
/// absent/unchanged — never a partial file.
fn save_prompt_cache_atomic(
  out_path: &Path,
  cache: &[Box<dyn KvCache>],
  metadata: &HashMap<String, String>,
) -> Result<()> {
  // The path mlx would actually write (extension auto-appended), so the
  // permission-capture + rename target match the direct-save destination.
  let dest = effective_safetensors_path(out_path);

  // Capture the destination's existing permissions (if any) so the renamed
  // file keeps the user's chosen mode (otherwise a private 0600 cache would
  // silently widen to the tempfile's umask-granted mode). `None` ⇒ no prior
  // file, so the tempfile's umask default stands. Mirrors `save_wav`.
  let existing_perms = fs::metadata(&dest).ok().map(|m| m.permissions());

  // Exclusively-created, same-directory, `.safetensors`-suffixed tempfile.
  const MAX_TEMPFILE_OPEN_RETRIES: u32 = 16;
  let tmp_path = open_excl_temp_safetensors(&dest, MAX_TEMPFILE_OPEN_RETRIES)?;

  // Inner closure so any failure cleans up the tempfile before returning.
  let write_result = (|| -> Result<()> {
    // mlx-c writes the cache to the tempfile path (it reopens + truncates the
    // regular file we exclusively created — no symlink follow). `tmp_path`
    // already ends in `.safetensors`, so mlx writes exactly it.
    save_prompt_cache(&tmp_path, cache, metadata)?;

    // fsync the tempfile so the bytes are durable before the rename — a
    // delayed-allocation / NFS / quota writeback failure must surface here,
    // not after we've renamed a not-yet-on-disk file into place. mlx-c does
    // not fsync; reopen the path read-only and `sync_all` it.
    let f = fs::File::open(&tmp_path).map_err(|e| Error::Backend {
      message: format!(
        "cache_prompt: reopen tempfile {} for fsync failed: {e}",
        tmp_path.display()
      ),
    })?;
    f.sync_all().map_err(|e| Error::Backend {
      message: format!(
        "cache_prompt: fsync tempfile {} failed: {e}",
        tmp_path.display()
      ),
    })?;
    drop(f);
    Ok(())
  })();

  if let Err(err) = write_result {
    let _ = fs::remove_file(&tmp_path);
    return Err(err);
  }

  // Restore the destination's prior permissions BEFORE the rename (skipped
  // when the destination did not previously exist). A failure here is handled
  // like any write-path failure: clean up the tempfile and propagate.
  if let Some(perms) = existing_perms
    && let Err(e) = fs::set_permissions(&tmp_path, perms)
  {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::Backend {
      message: format!(
        "cache_prompt: set_permissions on tempfile {} failed: {e}",
        tmp_path.display()
      ),
    });
  }

  // Atomic-within-fs rename: no observer can see a half-written cache at the
  // destination. On failure, remove the tempfile and propagate (the
  // destination keeps whatever it had before — never a partial file).
  if let Err(e) = fs::rename(&tmp_path, &dest) {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::Backend {
      message: format!(
        "cache_prompt: rename {} -> {} failed: {e}",
        tmp_path.display(),
        dest.display()
      ),
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  //! In-crate prefill-boundary unit tests for the driver, sharing the
  //! `model::MockModel` fixture (`#[cfg(test)] pub(crate)`, visible here).
  //! The cross-tool round-trip (fill → save → load → continue) lives in the
  //! integration test `tests/lm_cache_prompt_driver.rs`, which exercises the
  //! public `save_prompt_cache`/`load_prompt_cache`.

  use std::{cell::Cell, rc::Rc};

  use super::*;
  use crate::lm::{
    cache::{CacheConfig, MaskMode, StandardKvCache, make_prompt_cache},
    model::MockModel,
  };

  fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
    make_prompt_cache(&CacheConfig {
      num_hidden_layers: layers,
      sliding_window: None,
    })
  }

  /// A [`KvCache`] that delegates to a [`StandardKvCache`] but counts every
  /// [`materialize`](KvCache::materialize) call into a shared counter — used
  /// to observe the per-chunk [`materialize_caches`] barrier firing during
  /// prefill (the barrier calls `materialize()` once per layer per chunk).
  struct CountingCache {
    inner: StandardKvCache,
    materialize_calls: Rc<Cell<usize>>,
  }

  impl KvCache for CountingCache {
    fn offset(&self) -> usize {
      self.inner.offset()
    }
    fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
      self.inner.update(keys, values)
    }
    fn state(&self) -> Result<Vec<Array>> {
      self.inner.state()
    }
    fn materialize(&mut self) -> Result<()> {
      self.materialize_calls.set(self.materialize_calls.get() + 1);
      self.inner.materialize()
    }
    fn set_state(&mut self, state: Vec<Array>) -> Result<()> {
      self.inner.set_state(state)
    }
    fn make_mask(&self, n: usize, w: Option<usize>, ret: bool) -> Result<MaskMode> {
      self.inner.make_mask(n, w, ret)
    }
    fn nbytes(&self) -> usize {
      self.inner.nbytes()
    }
    fn is_empty(&self) -> bool {
      self.inner.is_empty()
    }
    fn copy(&self) -> Result<Box<dyn KvCache>> {
      self.inner.copy()
    }
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

  /// The per-chunk cache materialization barrier (mlx-lm `generate.py:442`)
  /// fires once per layer per leading chunk on a multi-chunk prompt (`P >
  /// step`): the lazy graph is materialized between chunks, not accumulated to
  /// the save. A [`CountingCache`] counts `materialize()` calls during
  /// prefill; with `P = 7`, `step = 2` the leading `P-1 = 6` tokens are 3
  /// chunks `[2,2,2]`, so the barrier runs **3 times** (> 1 chunk) — proving
  /// the prefill is memory-bounded (the graph never spans the whole prompt).
  #[test]
  fn prefill_full_materializes_caches_per_chunk() {
    let model = MockModel::new(5);
    let counter = Rc::new(Cell::new(0usize));
    let mut c: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
      inner: StandardKvCache::new(),
      materialize_calls: Rc::clone(&counter),
    })];
    // P = 7, step = 2 ⇒ leading 6 tokens as [2,2,2] ⇒ 3 barrier calls.
    let prompt = [1u32, 2, 3, 4, 5, 6, 7];
    prefill_full(&model, &prompt, &mut c, 2).unwrap();
    assert_eq!(
      counter.get(),
      3,
      "the per-chunk materialize barrier must fire once per leading chunk (3 chunks for P=7, step=2)"
    );
    assert!(
      counter.get() > 1,
      "a multi-chunk prefill runs the barrier more than once"
    );
    // The cache still ends at offset P (the tail forward ran after the loop).
    assert!(c.iter().all(|x| x.offset() == prompt.len()));
  }

  /// A single-chunk prompt (`P - 1 <= step`) runs the barrier exactly once
  /// (the lone leading chunk); a `P == 1` prompt (no leading chunk) never
  /// enters the loop, so the barrier does not fire — both still leave the
  /// cache at offset `P` via the tail forward.
  #[test]
  fn prefill_full_barrier_count_matches_chunking() {
    // P = 4, step = 8 ⇒ leading 3 tokens in ONE chunk ⇒ 1 barrier call.
    let model = MockModel::new(5);
    let one = Rc::new(Cell::new(0usize));
    let mut c1: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
      inner: StandardKvCache::new(),
      materialize_calls: Rc::clone(&one),
    })];
    prefill_full(&model, &[1u32, 2, 3, 4], &mut c1, 8).unwrap();
    assert_eq!(one.get(), 1, "a single leading chunk runs the barrier once");

    // P = 1 ⇒ no leading chunk ⇒ 0 barrier calls (only the tail forward).
    let zero = Rc::new(Cell::new(0usize));
    let mut c0: Vec<Box<dyn KvCache>> = vec![Box::new(CountingCache {
      inner: StandardKvCache::new(),
      materialize_calls: Rc::clone(&zero),
    })];
    prefill_full(&model, &[42u32], &mut c0, 8).unwrap();
    assert_eq!(
      zero.get(),
      0,
      "a 1-token prompt has no leading chunk, so no barrier"
    );
    assert!(c0.iter().all(|x| x.offset() == 1));
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

  /// `cache_prompt_ids` rejects a **non-fresh** cache (a reused / pre-populated
  /// cache with `offset() > 0`) as a recoverable `Err` and writes no file —
  /// `cache_prompt` saves a cache representing exactly the requested prompt, so
  /// prefilling on top of stale state (which `prefill_full` would silently do,
  /// it only appends) would persist a `[stale + new]` prefix and leak prior
  /// context. Mirrors the empty-prompt guard's structure.
  #[test]
  fn cache_prompt_ids_non_fresh_cache_errors_without_writing() {
    let model = MockModel::new(4);
    // Pre-populate the cache by prefilling a first prompt into it (offset > 0).
    let mut c = cache(2);
    prefill_full(&model, &[1u32, 2, 3], &mut c, 8).unwrap();
    assert!(
      c.iter().all(|x| x.offset() == 3),
      "the cache is pre-populated (offset > 0) before the rejected call"
    );

    let dir = std::env::temp_dir().join(format!(
      "mlxrs_cache_prompt_nonfresh_{}",
      std::process::id()
    ));
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join("nonfresh.safetensors");
    let _ = std::fs::remove_file(&out);

    let r = cache_prompt_ids(
      &model,
      &[4u32, 5],
      &mut c,
      &out,
      "mock",
      "{}",
      8,
      &HashMap::new(),
    );
    match r {
      Err(Error::Backend { message }) => {
        assert!(
          message.contains("fresh") || message.contains("empty"),
          "the error must indicate the cache has to be fresh/empty: {message}"
        );
      }
      other => panic!("a reused cache must be a recoverable Error::Backend, got {other:?}"),
    }
    assert!(
      !out.exists(),
      "no cache file may be written when a reused cache is rejected"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// A genuinely **fresh** cache still succeeds: `cache_prompt_ids` over a
  /// freshly built [`make_prompt_cache`] cache prefills + saves normally (the
  /// freshness guard does not regress the positive path).
  #[test]
  fn cache_prompt_ids_fresh_cache_succeeds() {
    let model = MockModel::new(4);
    let mut c = cache(2);
    assert!(
      c.iter().all(|x| x.offset() == 0 && x.is_empty()),
      "a make_prompt_cache cache is fresh (offset 0, empty)"
    );
    let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_fresh_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join("fresh.safetensors");
    let _ = std::fs::remove_file(&out);

    let info = cache_prompt_ids(
      &model,
      &[1u32, 2, 3],
      &mut c,
      &out,
      "mock",
      "{}",
      8,
      &HashMap::new(),
    )
    .expect("a fresh cache must prefill + save successfully");
    assert_eq!(info.tokens_processed, 3);
    assert!(c.iter().all(|x| x.offset() == 3));
    assert!(out.exists(), "a fresh-cache run must write the cache file");
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// [`effective_safetensors_path`] mirrors mlx core's extension-append: a
  /// path already ending in `.safetensors` is unchanged; any other path gets
  /// `.safetensors` appended (so the atomic rename target == the path mlx
  /// would write for a direct save).
  #[test]
  fn effective_safetensors_path_matches_mlx_extension_rule() {
    assert_eq!(
      effective_safetensors_path(Path::new("/tmp/cache.safetensors")),
      PathBuf::from("/tmp/cache.safetensors"),
    );
    assert_eq!(
      effective_safetensors_path(Path::new("/tmp/cache")),
      PathBuf::from("/tmp/cache.safetensors"),
    );
    assert_eq!(
      effective_safetensors_path(Path::new("/tmp/cache.bin")),
      PathBuf::from("/tmp/cache.bin.safetensors"),
    );
  }

  /// Atomic-save crash safety (Codex finding): when the save FAILS, a
  /// previously valid cache at the destination is left **intact** and no
  /// partial tempfile remains. We first write a good cache, then point a
  /// second save into a directory made read-only so mlx's write into the
  /// tempfile fails — the original `out_path` must still load, byte-identical.
  ///
  /// Read-only-dir failure injection is POSIX (`unix`); on other targets the
  /// no-partial-file path is covered by
  /// [`save_prompt_cache_atomic_failed_save_to_fresh_path_leaves_nothing`].
  #[cfg(unix)]
  #[test]
  fn save_prompt_cache_atomic_failed_save_keeps_original_intact() {
    use std::os::unix::fs::PermissionsExt;

    use crate::lm::cache::load_prompt_cache;

    let model = MockModel::new(4);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_cache_prompt_atomic_intact_{}",
      std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("cache.safetensors");

    // 1. Write a good cache.
    let mut c = cache(2);
    cache_prompt_ids(
      &model,
      &[1u32, 2, 3, 4],
      &mut c,
      &out,
      "good",
      "{}",
      2,
      &HashMap::new(),
    )
    .unwrap();
    assert!(out.exists(), "the first save must produce a cache");
    let (orig_loaded, orig_meta) = load_prompt_cache(&out).unwrap();
    let orig_offsets: Vec<usize> = orig_loaded.iter().map(|x| x.offset()).collect();

    // 2. Make the directory read-only so the next save's tempfile create /
    //    mlx write fails. (Root could bypass this; CI/dev users are not root.)
    let mut perms = fs::metadata(&dir).unwrap().permissions();
    let orig_mode = perms.mode();
    perms.set_mode(0o500); // r-x------ : no write ⇒ create/write fails
    fs::set_permissions(&dir, perms).unwrap();

    let mut c2 = cache(2);
    let r = cache_prompt_ids(
      &model,
      &[5u32, 6, 7, 8],
      &mut c2,
      &out,
      "SHOULD-NOT-WIN",
      "{}",
      2,
      &HashMap::new(),
    );

    // Restore write perms BEFORE asserting (so cleanup + reads work even if an
    // assert fails).
    let mut restore = fs::metadata(&dir).unwrap().permissions();
    restore.set_mode(orig_mode);
    fs::set_permissions(&dir, restore).unwrap();

    assert!(r.is_err(), "a save into a read-only dir must fail");

    // 3. The original cache is untouched: same metadata + offsets, and no
    //    leftover `.tmp.safetensors` partial file in the directory.
    assert!(out.exists(), "the failed save must not delete the original");
    let (after_loaded, after_meta) = load_prompt_cache(&out).unwrap();
    assert_eq!(
      after_meta.get(META_MODEL).map(String::as_str),
      Some("good"),
      "the original cache's metadata must survive the failed save (not 'SHOULD-NOT-WIN')"
    );
    assert_eq!(after_meta, orig_meta, "original metadata must be unchanged");
    let after_offsets: Vec<usize> = after_loaded.iter().map(|x| x.offset()).collect();
    assert_eq!(
      after_offsets, orig_offsets,
      "the original cache contents must be unchanged"
    );
    let leftover_tmp = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
    assert!(
      !leftover_tmp,
      "no partial tempfile may remain after a failed save"
    );

    let _ = fs::remove_dir_all(&dir);
  }

  /// Atomic-save crash safety (Codex finding), fresh-path variant: a FAILED
  /// save to a destination that did not previously exist leaves it **absent**
  /// (no partial file). Injects failure via a read-only parent directory.
  #[cfg(unix)]
  #[test]
  fn save_prompt_cache_atomic_failed_save_to_fresh_path_leaves_nothing() {
    use std::os::unix::fs::PermissionsExt;

    let model = MockModel::new(4);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_cache_prompt_atomic_fresh_{}",
      std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("never.safetensors");

    let mut perms = fs::metadata(&dir).unwrap().permissions();
    let orig_mode = perms.mode();
    perms.set_mode(0o500);
    fs::set_permissions(&dir, perms).unwrap();

    let mut c = cache(1);
    let r = cache_prompt_ids(
      &model,
      &[1u32, 2, 3],
      &mut c,
      &out,
      "m",
      "{}",
      8,
      &HashMap::new(),
    );

    let mut restore = fs::metadata(&dir).unwrap().permissions();
    restore.set_mode(orig_mode);
    fs::set_permissions(&dir, restore).unwrap();

    assert!(r.is_err(), "save into a read-only dir must fail");
    assert!(
      !out.exists(),
      "a failed save to a fresh path must leave no file at the destination"
    );
    let any_file = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).count();
    assert_eq!(
      any_file, 0,
      "no partial / tempfile may remain in the directory"
    );

    let _ = fs::remove_dir_all(&dir);
  }

  /// Atomic-save cleanup AFTER the tempfile is written: when the final
  /// `rename` fails (here the destination path is an existing **directory**,
  /// which `fs::rename(file -> dir)` rejects), the tempfile that mlx already
  /// wrote must be removed — no `.tmp.safetensors` leftover, and the
  /// destination directory is untouched. This exercises the write-succeeds /
  /// rename-fails branch the read-only-dir tests (which fail at tempfile
  /// *create*) do not reach.
  #[test]
  fn save_prompt_cache_atomic_rename_failure_cleans_up_tempfile() {
    let model = MockModel::new(4);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs_cache_prompt_atomic_rename_{}",
      std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    // The destination is itself a directory ⇒ the final rename (file -> dir)
    // fails, but the tempfile create + mlx write succeed first.
    let out = dir.join("dest.safetensors");
    fs::create_dir_all(&out).unwrap();

    let mut c = cache(1);
    let r = cache_prompt_ids(
      &model,
      &[1u32, 2, 3],
      &mut c,
      &out,
      "m",
      "{}",
      8,
      &HashMap::new(),
    );
    assert!(r.is_err(), "rename onto an existing directory must fail");
    // The dest directory still exists (untouched) and is still a directory.
    assert!(out.is_dir(), "the destination directory must be untouched");
    // No leftover tempfile: the post-write rename failure cleaned it up.
    let leftover_tmp = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).any(|e| {
      e.file_name()
        .to_string_lossy()
        .ends_with(".tmp.safetensors")
    });
    assert!(
      !leftover_tmp,
      "the tempfile mlx wrote must be removed when the rename fails"
    );

    let _ = fs::remove_dir_all(&dir);
  }
}
