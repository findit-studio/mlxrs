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

#[cfg(test)]
use crate::error::RankMismatchPayload;
use crate::{
  array::Array,
  error::{
    EmptyInputPayload, Error, FileIoPayload, FileOp, MissingFieldPayload, ParsePayload, Result,
    try_with_capacity,
  },
  lm::{
    cache::{CacheConfig, KvCache, make_prompt_cache, save_prompt_cache},
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
      .map_err(|e| {
        Error::Parse(ParsePayload::new(
          "cache_prompt: apply_chat_template",
          "chat template",
          std::io::Error::other(e.to_string()),
        ))
      })?;
    Ok(ids)
  } else {
    // mlx-lm `tokenizer.encode(args.prompt)` — the tokenizer's default
    // special-token handling (transformers `encode` adds specials).
    tokenizer.encode(prompt, true).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "cache_prompt: encode",
        "prompt tokens",
        std::io::Error::other(e.to_string()),
      ))
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
/// (the hazard this hook closes). Evaluating the live arrays via the
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
/// ## Why the per-chunk barrier (memory-bounded prefill)
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

/// Tokenize `prompt`, prefill a freshly allocated cache with it, and save the
/// populated cache (plus metadata) to `out_path` — the support-surface port of
/// `mlx_lm.cache_prompt.main` (cache_prompt.py:83-145), minus the CLI.
///
/// Mirrors the reference end to end:
///
/// 1. **Tokenize** via `encode_prompt` (chat template when present, else
///    [`Tokenizer::encode`]) — cache_prompt.py:100-109.
/// 2. **Allocate** a fresh per-layer KV cache via
///    [`crate::lm::cache::make_prompt_cache`] — exactly cache_prompt.py:111
///    (`cache = make_prompt_cache(model, args.max_kv_size)`). The cache is
///    *internally allocated*, never caller-provided, so it is fresh by
///    construction: there is no prior-request state to leak.
/// 3. **Prefill** the full prompt into that cache via `prefill_full` (the
///    `generate_step(max_tokens=0)` forward sequence; no sampling) —
///    cache_prompt.py:111-136. The empty-prompt case is rejected up front as
///    a recoverable [`Error::Backend`] (mlx-lm's `generate_step` raises
///    `ValueError` on an empty prompt; a prefill over zero tokens would be a
///    no-op saving an empty cache, so failing fast is the faithful behavior).
/// 4. **Save** the cache via [`crate::lm::cache::save_prompt_cache`] with the
///    metadata cache_prompt.py writes — `metadata["model"] = model_id` and
///    `metadata["tokenizer_config"] = tokenizer_config_json`
///    (cache_prompt.py:142-145). `extra_metadata` lets a caller add further
///    keys (e.g. an explicit processed-count) without altering the wire
///    format; the two reference keys take precedence on collision.
///
/// `cache_config` is the model-appropriate cache spec
/// ([`crate::lm::cache::CacheConfig`] — `num_hidden_layers` and the optional
/// `sliding_window`). In mlx-lm `make_prompt_cache(model)` reads this off the
/// model object; mlxrs's [`Model`] trait carries no such introspection seam,
/// so the spec is passed explicitly — `make_prompt_cache` then builds the
/// matching cache (a [`crate::lm::cache::RotatingKvCache`] per layer for a
/// sliding-window model, a [`crate::lm::cache::StandardKvCache`] otherwise).
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
  cache_config: &CacheConfig,
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
    cache_config,
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
/// Performs steps 2-4 of [`cache_prompt`]: allocate a fresh per-layer KV
/// cache via [`crate::lm::cache::make_prompt_cache`] (`cache_config`), prefill
/// the full `prompt_ids` into it (the `generate_step(max_tokens=0)` forward
/// sequence), then save via [`crate::lm::cache::save_prompt_cache`] with the
/// `model` / `tokenizer_config` metadata cache_prompt.py writes (plus
/// `extra_metadata`). An empty `prompt_ids` is a recoverable
/// [`Error::Backend`] (faithful to mlx-lm's empty-prompt `ValueError`);
/// nothing is written in that case.
///
/// The cache is **allocated internally**, exactly as `cache_prompt.py:111`
/// does (`cache = make_prompt_cache(model, args.max_kv_size)`) — never
/// caller-provided. An internally-allocated cache is *fresh by construction*,
/// so the saved cache represents exactly this prompt: there is no caller cache
/// to reuse and therefore no way to persist a prior request's state. (This is
/// strictly more faithful to the reference than a caller-cache parameter would
/// be.) `cache_config` is the model-appropriate cache spec — see
/// [`cache_prompt`].
#[allow(clippy::too_many_arguments)]
pub fn cache_prompt_ids<M: Model>(
  model: &M,
  prompt_ids: &[u32],
  cache_config: &CacheConfig,
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
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "cache_prompt: prompt",
    )));
  }

  // 2. Allocate a fresh per-layer KV cache — cache_prompt.py:111
  // (`cache = make_prompt_cache(model, args.max_kv_size)`). `cache_prompt`
  // saves a cache representing *exactly this prompt*, and `prefill_full` only
  // ever *appends* (it never resets a cache). Allocating the cache here — as
  // the reference does — makes it fresh by construction: there is no
  // caller-provided cache that could carry a prior request's state into the
  // save, so a cross-request context leak is structurally impossible.
  let mut cache = make_prompt_cache(cache_config);

  // 3. prefill the full prompt into the cache (cache_prompt.py:111-136).
  prefill_full(model, prompt_ids, &mut cache, prefill_step_size)?;

  // 4. save with the reference metadata (cache_prompt.py:142-145).
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
  // corrupt cache at `out_path`. Mirrors `audio::io::save_wav`.
  save_prompt_cache_atomic(out_path, &cache, &metadata)?;

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
    .ok_or_else(|| {
      Error::MissingField(MissingFieldPayload::new(
        "cache_prompt: destination path",
        "file_name component",
      ))
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
        return Err(Error::FileIo(FileIoPayload::new(
          "cache_prompt: create_new tempfile",
          FileOp::Create,
          candidate,
          e,
        )));
      }
    }
  }
  Err(Error::FileIo(FileIoPayload::new(
    "cache_prompt: exhausted tempfile create_new retries (every candidate collided with an existing path)",
    FileOp::Create,
    final_path.to_path_buf(),
    last_err.unwrap_or_else(|| std::io::Error::from(std::io::ErrorKind::AlreadyExists)),
  )))
}

/// Atomically save `cache` (+ `metadata`) to `out_path` — the durable,
/// crash-safe form of [`crate::lm::cache::save_prompt_cache`].
///
/// `save_prompt_cache` calls `mlx_save_safetensors` straight onto the final
/// path (no temp / fsync / rename), so a crash or IO error mid-save would
/// leave a partial/corrupt `.safetensors` at the destination — clobbering a
/// previously valid cache. This mirrors `audio::io::save_wav`'s
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
    let f = fs::File::open(&tmp_path).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "cache_prompt: reopen tempfile",
        FileOp::Open,
        tmp_path.to_path_buf(),
        e,
      ))
    })?;
    f.sync_all().map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "cache_prompt: fsync tempfile",
        FileOp::Fsync,
        tmp_path.to_path_buf(),
        e,
      ))
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
    return Err(Error::FileIo(FileIoPayload::new(
      "cache_prompt: set_permissions on tempfile",
      FileOp::Other("set_permissions"),
      tmp_path,
      e,
    )));
  }

  // Atomic-within-fs rename: no observer can see a half-written cache at the
  // destination. On failure, remove the tempfile and propagate (the
  // destination keeps whatever it had before — never a partial file).
  if let Err(e) = fs::rename(&tmp_path, &dest) {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::FileIo(FileIoPayload::new(
      "cache_prompt: rename tempfile -> destination",
      FileOp::Rename,
      tmp_path,
      e,
    )));
  }
  Ok(())
}

#[cfg(test)]
mod tests;
