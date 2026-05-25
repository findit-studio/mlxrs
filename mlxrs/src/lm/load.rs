//! Architecture-agnostic model-load **and -save support surface**, ported
//! from
//! [`mlx_lm.utils`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
//! (the authoritative spec) and cross-checked against `mlx-swift-lm`'s
//! `MLXLMCommon` loader.
//!
//! This module ports **only** the arch-agnostic pieces of `mlx_lm.utils`:
//!
//! - [`load_config`] mirrors `utils.load_config` — read `config.json` **once**
//!   and return both the typed [`Config`] subset and the verbatim JSON body
//!   (forward-compatible: unknown keys are ignored, so a checkpoint's
//!   `quantization_config`/`text_config`/etc. never break the parse), with the
//!   `generation_config.json` `eos_token_id` override applied. The pure-parse
//!   step is [`Config::from_json`].
//! - [`load_weights`] mirrors the weight-discovery + merge of
//!   `utils.load_model`, honoring `model.safetensors.index.json` (the HF
//!   safetensors-sharded authoritative manifest) as the SINGLE source of
//!   truth for which shards are part of the checkpoint when present, then
//!   falling back to a single `model.safetensors`, a legacy
//!   `weights.safetensors`, or a single `*.gguf` (mirroring mlx-lm's GGUF
//!   path). Quantized triples (`*.weight` / `*.scales` / `*.biases`) are
//!   kept **verbatim**.
//! - [`load`] mirrors `utils.load` — wire `config.json` + weights + the #18
//!   [`Tokenizer`](crate::tokenizer::Tokenizer) into the parts a (per-usecase)
//!   architecture assembles itself.
//! - [`make_shards`] / [`save_model`] / [`save_config`] / [`save`] mirror the
//!   model-**save** half of `mlx_lm.utils`: split a [`Weights`] map into
//!   `≤ max-shard-size` `.safetensors` shards, write them plus the
//!   `model.safetensors.index.json` weight-map index, write back a sorted
//!   `config.json`, and the `save` driver that wires the two together.
//! - [`get_total_parameters`] / [`compute_bits_per_weight`] /
//!   [`does_model_support_input_embeddings`] mirror the model-introspection
//!   helpers (`utils.py:196-215,979-991`).
//!
//! **Deliberately NOT ported** (per the project's no-model-arch scoping):
//! `utils.load_model`'s per-architecture `model_class(model_args)`
//! construction, `model.sanitize(weights)` key-remap, the `_quantize` /
//! `class_predicate` quantization *application* (it mutates a constructed
//! model), the legacy AWQ/bitnet transforms, `_download` /
//! `create_model_card` / `upload_to_hub` (HuggingFace Hub — this is
//! local-path-only, no network), and `sharded_load` / `pipeline_load`
//! (distributed). [`load`] returns the raw `(Config, Weights, Tokenizer)`
//! triple; assembling and (de)quantizing a concrete model is the per-usecase
//! architecture's job. The [`Config::quantization`] field merely *carries*
//! `config["quantization"]` (mlx-lm `utils.py`'s
//! `config.get("quantization")`) so a later arch can apply it — `load` itself
//! never quantizes. The save side is symmetric: it works on the arch-agnostic
//! [`Weights`] name map (mlx-lm's `dict(tree_flatten(model.parameters()))`),
//! not a constructed `nn.Module`.
//!
//! Conventions mirror [`crate::lm::sample`] and
//! [`crate::embeddings::config`]: `Result`-fallible, no implicit eval (the
//! weight `Array`s are returned lazily — no `eval`/`item`/`to_vec` here),
//! recoverable IO / parse / missing-file failures map to
//! [`Error::Backend`] with a clear message (the codebase's config-loading
//! convention), and the `config.json` read is bounded against an untrusted
//! model directory exactly as `embeddings::config`'s pooling-config read.
//!
//! [`Error::Backend`]: crate::Error::Backend

use std::{
  collections::{BTreeMap, HashMap},
  path::{Path, PathBuf},
};

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, Result},
};

/// Upper bound on a `config.json` we will read into memory, mirroring
/// `embeddings::config`'s `MAX_ST_POOLING_CONFIG_BYTES`. A real model's
/// `config.json` is well under 1 MiB; a hostile model dir cannot make us
/// allocate unbounded memory by planting a huge `config.json`.
///
/// `pub(crate)` so the [`crate::lm::factory`] load path shares the *one* bound
/// (rather than restating it) — both read `config.json` through the same cap.
pub(crate) const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// Upper bound on a `model.safetensors.index.json` we will read into memory.
/// The index carries one `weight_name -> shard_name` entry per tensor; even
/// a Llama-3-405B-class model lists well under 100 000 tensors, comfortably
/// under 16 MiB of JSON. A hostile model directory cannot OOM us by planting
/// a multi-GB index. Twin of [`MAX_CONFIG_BYTES`] for the larger
/// per-tensor-keyed index file.
const MAX_INDEX_BYTES: u64 = 16 << 20;

/// Quantization parameters from a checkpoint's `config.json` `quantization`
/// block (mlx-lm `utils.py` `config["quantization"]`: `{ "group_size": int,
/// "bits": int, [ "mode": str ] }`).
///
/// Re-export of the canonical [`crate::lm::quant::Quantization`] (the
/// swift-faithful schema with `mode` — `BaseConfiguration.swift:22-56`).
/// Carried so a per-usecase architecture can apply quantization itself;
/// [`load`] never quantizes. For the per-layer-aware "fine-grained" schema
/// (per-layer overrides, `Skip` entries), see
/// [`crate::lm::quant::PerLayerQuantization`] / the
/// [`crate::lm::quant::parse_quantization`] entry point.
pub use crate::lm::quant::Quantization;

/// The `config.json` subset the loader / generation loop needs, mirroring
/// the keys `mlx_lm.utils.load_config` feeds into a model's `ModelArgs`.
///
/// **Forward-compatible by design:** `#[serde(deny_unknown_fields)]` is
/// deliberately NOT set, so a checkpoint carrying extra keys
/// (`quantization_config`, `text_config`, `max_position_embeddings`, future
/// fields, …) parses cleanly — exactly as mlx-lm's `ModelArgs.from_dict`
/// ignores unmodeled keys. A missing **required** field is a parse error
/// (→ [`Error::Backend`]), matching mlx-lm raising on an incomplete config.
///
/// `sliding_window` and `quantization` are optional (`#[serde(default)]` →
/// `None` when absent). `num_hidden_layers` and `sliding_window` are the two
/// fields the (separately-stacked) `make_prompt_cache` bridge needs, so they
/// are always carried here.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
  /// Architecture id (`config.json` `model_type`, e.g. `"qwen3"`).
  pub model_type: String,
  /// Model hidden / embedding dimension.
  pub hidden_size: i32,
  /// Number of decoder layers — one KV-cache entry per layer.
  pub num_hidden_layers: i32,
  /// Number of attention (query) heads.
  pub num_attention_heads: i32,
  /// Number of key/value heads (GQA; equals `num_attention_heads` for MHA).
  pub num_key_value_heads: i32,
  /// Per-head dimension.
  pub head_dim: i32,
  /// RoPE base frequency (`config.json` `rope_theta`).
  pub rope_theta: f32,
  /// Vocabulary size (logits last-axis width).
  pub vocab_size: i32,
  /// Whether the LM head reuses the input-embedding weights.
  pub tie_word_embeddings: bool,
  /// Sliding-attention window, if any (`None` ⇒ full attention). Drives the
  /// later `make_prompt_cache` Rotating-vs-Standard choice.
  #[serde(default)]
  pub sliding_window: Option<i32>,
  /// Weight-quantization parameters, if the checkpoint is quantized
  /// (mlx-lm `config["quantization"]`). Carried, not applied.
  #[serde(default)]
  pub quantization: Option<Quantization>,
  /// `config.json` `eos_token_id` (a single id or a list) — mlx-lm's base
  /// stop-id set. A *truthy* `generation_config.json` `eos_token_id`
  /// overrides it; the result is the tokenizer's COMPLETE eos set (it
  /// REPLACES the tokenizer-config default — see [`load`]). `None` ⇒ fall
  /// back to the tokenizer's own `eos_token`.
  #[serde(default)]
  pub eos_token_id: Option<EosTokenId>,
}

/// `config.json` / `generation_config.json` `eos_token_id`: HF checkpoints
/// write it as either a single integer (`128001`) or a list
/// (`[128001, 128009]`); mlx-lm accepts both. Untagged so serde tries the
/// scalar form first, then the list.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
  /// A single stop id.
  Single(u32),
  /// An explicit list of stop ids.
  Many(Vec<u32>),
}

impl EosTokenId {
  /// Flatten to the id list (a scalar becomes a one-element list), the form
  /// the tokenizer's eos set is built from.
  pub fn into_ids(self) -> Vec<u32> {
    match self {
      EosTokenId::Single(x) => vec![x],
      EosTokenId::Many(v) => v,
    }
  }
}

impl Config {
  /// Parse a [`Config`] from an in-memory `config.json` string.
  ///
  /// Mirrors `mlx_lm.utils.load_config` (`json.load(config.json)`) restricted
  /// to the typed subset. A serde failure (malformed JSON or a missing
  /// required key) maps to [`Error::Backend`] with the underlying cause —
  /// the codebase's config-parse error convention (twin of
  /// `embeddings::config`'s `serde_json::from_str(..).map_err(Backend)`).
  pub fn from_json(json: &str) -> Result<Config> {
    serde_json::from_str(json).map_err(|e| Error::Backend {
      message: format!("invalid model config JSON: {e}"),
    })
  }
}

/// A flat name → [`Array`] weight map (mlx-lm's `weights` dict /
/// `mx.load(...)` result).
///
/// Quantized layers appear as the verbatim triple `*.weight` / `*.scales` /
/// `*.biases` (mlx's `QuantizedLinear` layout). [`load_weights`] performs
/// **no** key remapping or `sanitize` — that is the per-usecase
/// architecture's responsibility (mlx-lm `model.sanitize(weights)`), kept out
/// of this arch-agnostic surface.
pub type Weights = HashMap<String, Array>;

/// Discover and merge a model's weights from `dir`, mirroring the
/// weight-loading half of `mlx_lm.utils.load_model` while honoring the
/// HF/safetensors `model.safetensors.index.json` weight-map as the
/// **authoritative** shard manifest.
///
/// Resolution order (first match wins):
///
/// 1. **`model.safetensors.index.json` (authoritative).** If the index file
///    is present, it is the SINGLE source of truth for which shards belong
///    to the checkpoint. The unique shard filenames listed in its
///    `weight_map` are loaded from `<dir>/<shard>` and merged. Stale
///    `model*.safetensors` files in `dir` whose names are NOT in the index
///    are **ignored** (so the [`save_model`] index-rename single-commit-point
///    can leave new-but-not-yet-published shards on disk, or stale-from-an-
///    earlier-checkpoint shards, without corrupting load). This is the
///    standard HF safetensors-sharded convention (and mlx-lm's own
///    distributed-load path: `utils.py:557-558` reads the same `weight_map`
///    to pick local files; the local-only path's `glob("model*.safetensors")`
///    happens to converge on the same files only because a fresh
///    `save_model` directory has no stragglers).
/// 2. **Single `model.safetensors`** (no index). The HF un-sharded
///    convention: load the one file directly.
/// 3. **Legacy `weights.safetensors`.** Pre-HF-convention back-compat: if a
///    directory carries only this name, load it.
/// 4. **GGUF fallback:** if no safetensors of any of the three layouts above
///    is present, a single `*.gguf` in `dir` is loaded via
///    [`crate::io::load_gguf`] (mlx-lm's GGUF load path). Requires the
///    `gguf` feature; without it a present `*.gguf` is reported as
///    unsupported.
///
/// No safetensors and no usable GGUF → [`Error::Backend`] (mlx-lm's
/// `FileNotFoundError("No safetensors found in {model_path}")`). Keys are
/// returned **verbatim** (no remap/sanitize — spec §7.2).
///
/// The index is parsed with the same bounded-IO / `O_NONBLOCK` /
/// non-regular-reject discipline `read_bounded_config_file` uses for
/// `config.json` (capped at 16 MiB — well above a Llama-3-405B-class
/// index); a malformed or out-of-spec index is a recoverable
/// [`Error::Backend`].
pub fn load_weights(dir: &Path) -> Result<Weights> {
  // 1. Index-honoring path: the index, if present, IS the authoritative
  //    shard manifest. Stale `model*.safetensors` files NOT listed in the
  //    index are invisible to load — that is what makes the [`save_model`]
  //    index-rename single-commit-point safe.
  if let Some(weights) = load_via_index(dir)? {
    return Ok(weights);
  }

  // 2. Single, un-sharded `model.safetensors` (HF convention without an
  //    index file).
  let single = dir.join("model.safetensors");
  if path_is_file(&single)? {
    return crate::io::load_safetensors(&single);
  }

  // 3. Legacy back-compat: a `weights.safetensors`-only directory (pre-HF
  //    naming). Kept so a hand-rolled or older checkpoint that uses this
  //    name still loads.
  let legacy = dir.join("weights.safetensors");
  if path_is_file(&legacy)? {
    return crate::io::load_safetensors(&legacy);
  }

  // 4. No safetensors → try a single `*.gguf` (mlx-lm's GGUF load path).
  let ggufs = collect_sorted(dir, |name| name.ends_with(".gguf"))?;
  if let Some(gguf) = ggufs.first() {
    #[cfg(feature = "gguf")]
    {
      let (weights, _meta) = crate::io::load_gguf(gguf)?;
      return Ok(weights);
    }
    #[cfg(not(feature = "gguf"))]
    {
      return Err(Error::Backend {
        message: format!(
          "found a GGUF weight file ({}) but the `gguf` feature is disabled; \
           enable it to load GGUF checkpoints",
          gguf.display()
        ),
      });
    }
  }

  Err(Error::Backend {
    message: format!(
      "no model weights found in {}: expected `model.safetensors.index.json`, \
       `model.safetensors`, `weights.safetensors`, or a single `*.gguf`",
      dir.display()
    ),
  })
}

/// Load a checkpoint via its `model.safetensors.index.json`, if present.
///
/// Returns `Ok(Some(weights))` when an index file is found and successfully
/// drives the load; `Ok(None)` when no index file is present (the caller's
/// "try the next candidate" signal — twin of [`read_bounded_config_file`]'s
/// absent-file convention); `Err` on any structural problem (malformed
/// index, missing referenced shard, IO failure).
///
/// The index is the HF/safetensors authoritative shard manifest: its
/// `weight_map` lists every weight-name → shard-file-name binding. The
/// **unique** shard file names from `weight_map`'s values are collected, the
/// shards are loaded in sorted filename order (the same determinism
/// convention [`make_shards`] uses), and the per-shard maps are merged. A
/// shard file named in the index but absent on disk is an
/// [`Error::Backend`] naming the offending shard. A weight key listed in
/// `weight_map` whose corresponding tensor is NOT in the named shard's
/// safetensors body is silently tolerated (mlx-lm tolerates the symmetric
/// case — see `utils.py:557-558` — and the surrounding model construction
/// will raise on the missing key with a much better diagnostic).
///
/// The index body is bounded at [`MAX_INDEX_BYTES`] via the shared
/// [`read_bounded_text_file`] primitive.
fn load_via_index(dir: &Path) -> Result<Option<Weights>> {
  let index_path = dir.join("model.safetensors.index.json");
  let Some(text) = read_bounded_text_file(&index_path, "model weight index", MAX_INDEX_BYTES)?
  else {
    return Ok(None);
  };

  let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::Backend {
    message: format!(
      "model weight index {} is not valid JSON: {e}",
      index_path.display()
    ),
  })?;
  let weight_map = parsed
    .get("weight_map")
    .and_then(|v| v.as_object())
    .ok_or_else(|| Error::Backend {
      message: format!(
        "model weight index {} is missing a `weight_map` object",
        index_path.display()
      ),
    })?;

  // Collect the UNIQUE shard filenames from the index (a `BTreeSet` so the
  // load order is deterministic — the same sorted-filename convention used
  // by the pre-index `glob`+merge path). Reject a non-string entry, an
  // empty name, or a name carrying a path separator (an absolute or
  // parent-traversing shard name would escape `dir`; the HF convention is
  // bare basenames living in the same directory).
  let mut shard_names: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
  for (weight_key, shard_value) in weight_map {
    let shard = shard_value.as_str().ok_or_else(|| Error::Backend {
      message: format!(
        "model weight index {}: `weight_map[{}]` is not a string",
        index_path.display(),
        weight_key
      ),
    })?;
    if shard.is_empty()
      || shard.contains('/')
      || shard.contains('\\')
      || shard == "."
      || shard == ".."
    {
      return Err(Error::Backend {
        message: format!(
          "model weight index {}: `weight_map[{}]` -> {shard:?} is not a bare \
           shard basename (must live in the same directory as the index)",
          index_path.display(),
          weight_key,
        ),
      });
    }
    shard_names.insert(shard);
  }

  let mut weights: Weights = HashMap::new();
  for shard in &shard_names {
    let shard_path = dir.join(shard);
    if !path_is_file(&shard_path)? {
      return Err(Error::Backend {
        message: format!(
          "model weight index {} lists shard {shard:?} but {} is missing",
          index_path.display(),
          shard_path.display(),
        ),
      });
    }
    let part = crate::io::load_safetensors(&shard_path)?;
    weights.extend(part);
  }
  Ok(Some(weights))
}

/// Whether `path` exists AND its (symlink-resolved) target is a regular
/// file. A symlink whose target is a regular file qualifies (HF Hub snapshot
/// dirs store these as symlinks into `blobs/<hash>`, the same convention the
/// [`collect_sorted`] / [`read_bounded_config_file`] paths intentionally
/// follow). A missing path is `Ok(false)`; any other stat failure is an
/// [`Error::Backend`].
fn path_is_file(path: &Path) -> Result<bool> {
  match std::fs::metadata(path) {
    Ok(m) => Ok(m.is_file()),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
    Err(e) => Err(Error::Backend {
      message: format!("cannot stat {}: {e}", path.display()),
    }),
  }
}

/// List the entries of `dir` whose file name matches `pred`, returning their
/// full paths sorted by name. A non-readable directory (absent / not a
/// directory / permission) maps to [`Error::Backend`]. Only regular files
/// are considered (a directory named `model….safetensors` is ignored).
fn collect_sorted(dir: &Path, pred: impl Fn(&str) -> bool) -> Result<Vec<std::path::PathBuf>> {
  let entries = std::fs::read_dir(dir).map_err(|e| Error::Backend {
    message: format!("cannot read model directory {}: {e}", dir.display()),
  })?;
  let mut out = Vec::new();
  for entry in entries {
    let entry = entry.map_err(|e| Error::Backend {
      message: format!("cannot read an entry of {}: {e}", dir.display()),
    })?;
    let name = entry.file_name();
    let Some(name) = name.to_str() else { continue };
    if !pred(name) {
      continue;
    }
    // Require the *resolved* target to be a regular file (a hostile dir
    // could name a subdir / FIFO `model.safetensors`; mlx-c would then fail
    // opaquely on open). `DirEntry::file_type()` does NOT follow symlinks,
    // but HF Hub snapshot dirs store `model*.safetensors` as symlinks into
    // `blobs/<hash>` (mlx-lm's `glob(...) + mx.load(wf)` follows them) — so
    // resolve via `fs::metadata` (follows symlinks) and gate on the target,
    // exactly as `load_config` does for `config.json`. The original
    // (possibly-symlink) path is passed through; the IO loader opens it,
    // following the link.
    match std::fs::metadata(entry.path()) {
      Ok(m) if m.is_file() => out.push(entry.path()),
      Ok(_) => continue,
      Err(e) => {
        return Err(Error::Backend {
          message: format!("cannot stat {} in {}: {e}", name, dir.display()),
        });
      }
    }
  }
  out.sort();
  Ok(out)
}

/// Read `<dir>/config.json` **once**, returning both the typed [`Config`] and
/// the verbatim JSON body it was parsed from (the exact same bytes).
///
/// Mirrors `mlx_lm.utils.load_config`'s `open(model_path / "config.json")`,
/// and additionally applies the `generation_config.json` `eos_token_id`
/// override in place (see below) so the returned [`Config`] is the one the
/// generation loop / tokenizer should use — exactly `load(return_config=True)`.
///
/// Returning the raw text alongside the typed value closes a TOCTOU/divergence
/// hole the [`crate::lm::factory`] loader otherwise hit: a single open means a
/// constructor that consumes both the typed [`Config`] and the raw
/// [`String`](for model-specific keys outside the typed subset) can never get
/// them from two different on-disk versions of `config.json`.
///
/// The read is bounded against an untrusted model directory exactly as
/// `embeddings::config`'s pooling-config read: open **once** (closing the
/// stat-then-read TOCTOU window), reject a non-regular file (FIFO / device /
/// directory / symlink-to-special — all of which a pre-read size check would
/// see as `len() == 0` yet still stream unbounded data) **before any read**,
/// and cap the body at `MAX_CONFIG_BYTES` via `Read::take`. On Unix the
/// open carries `O_NONBLOCK | O_CLOEXEC` so a planted FIFO returns
/// immediately instead of hanging the caller; symlinks are intentionally
/// followed (HF Hub caches store `config.json` as a symlink into
/// `blobs/<hash>` — refusing symlinks would break every cached model) since
/// the post-open `is_file()` fstat enforces the guarantee on the *resolved*
/// target. Every failure path (absent, non-regular, oversized, unreadable,
/// invalid/incomplete JSON) is a recoverable [`Error::Backend`].
///
/// The eos override: a *truthy* `generation_config.json` `eos_token_id`
/// OVERWRITES `config["eos_token_id"]` IN PLACE
/// (`if eos_token_id := generation_config.get("eos_token_id", False):
/// config["eos_token_id"] = eos_token_id`), so the returned `Config`'s
/// `eos_token_id` is the tokenizer's COMPLETE set; `None` ⇒ the tokenizer's
/// own `eos_token`. The raw JSON [`String`] is the literal `config.json` body
/// and is **not** rewritten — it carries the on-disk model-specific keys
/// verbatim for a constructor's `Codable`-style init.
pub fn load_config(dir: &Path) -> Result<(Config, String)> {
  let path = dir.join("config.json");
  let Some(text) = read_bounded_config_file(&path, "model config")? else {
    return Err(Error::Backend {
      message: format!(
        "cannot open model config {}: file not found",
        path.display()
      ),
    });
  };
  let mut config = Config::from_json(&text)?;

  // mlx-lm `utils.load_config`: a *truthy* `generation_config.json`
  // `eos_token_id` OVERWRITES `config["eos_token_id"]` IN PLACE, so the
  // RETURNED config (and any tokenizer eos set derived from it) reflects the
  // generation-config override (`load(return_config=True)` parity). The raw
  // `text` is left untouched (it is the on-disk `config.json` verbatim).
  if let Some(eos_override) = read_generation_eos(dir) {
    config.eos_token_id = Some(eos_override);
  }

  Ok((config, text))
}

/// Bounded, TOCTOU-closed read of a config-style file at `path`.
///
/// Shared bounded-config-file primitive used by every config-JSON reader in
/// the loader (`config.json`, `generation_config.json`,
/// `(pre)processor_config.json`, VLM base-config). Behavior:
///
/// - `Ok(Some(text))` on a successful, bounded, valid-UTF-8 read.
/// - `Ok(None)` if the file is absent (`ENOENT`) — the caller's "try the
///   next candidate" / "absent is OK" signal. The caller decides whether
///   absence is a hard error (e.g. [`load_config`]) or simply *no override*
///   (e.g. [`read_generation_eos`], the VLM processor-config fallback).
/// - `Err(Error::Backend)` on every other failure (open failure other than
///   `NotFound`, not a regular file, oversized, IO failure during read,
///   non-UTF-8). Messages name the offending path and the `label` (one of
///   `"model config"`, `"generation config"`, `"processor config"`, …).
///
/// Discipline mirrors `embeddings::config`'s pooling-config read: open
/// **once** with `O_NONBLOCK | O_CLOEXEC` on unix (so a planted FIFO returns
/// immediately and never hangs the loader), post-open `is_file()` fstat
/// rejects non-regular targets even when reached via a symlink (HF Hub
/// snapshot caches store these files as symlinks into `blobs/<hash>`, which
/// is intentionally followed since the post-open stat enforces the
/// guarantee on the *resolved* target), and the body is capped at
/// [`MAX_CONFIG_BYTES`] via `Read::take` so a hostile model directory
/// cannot OOM us by planting a huge config.
pub(crate) fn read_bounded_config_file(path: &Path, label: &str) -> Result<Option<String>> {
  read_bounded_text_file(path, label, MAX_CONFIG_BYTES)
}

/// Shared bounded-text-file primitive parametrized on the byte cap. Identical
/// hardening (open-once + non-regular-reject + `O_NONBLOCK | O_CLOEXEC` on
/// unix + cap-via-`Read::take`) as [`read_bounded_config_file`]; factored out
/// so the larger [`MAX_INDEX_BYTES`] cap for `model.safetensors.index.json`
/// can reuse the *one* hardening path rather than restating it.
fn read_bounded_text_file(path: &Path, label: &str, max_bytes: u64) -> Result<Option<String>> {
  use std::io::Read;

  #[cfg(unix)]
  let open_result = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(path)
  };
  #[cfg(not(unix))]
  let open_result = std::fs::File::open(path);

  let file = match open_result {
    Ok(f) => f,
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
    Err(e) => {
      return Err(Error::Backend {
        message: format!("cannot open {label} {}: {e}", path.display()),
      });
    }
  };

  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("cannot stat opened {label} {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "{label} {} is not a regular file; refusing to read",
        path.display()
      ),
    });
  }

  let mut bytes = Vec::new();
  file
    .take(max_bytes + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("cannot read {label} {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > max_bytes {
    return Err(Error::Backend {
      message: format!(
        "{label} {} exceeds the {max_bytes}-byte cap; refusing to read",
        path.display(),
      ),
    });
  }

  let text = String::from_utf8(bytes).map_err(|e| Error::Backend {
    message: format!("{label} {} is not valid UTF-8: {e}", path.display()),
  })?;
  Ok(Some(text))
}

/// The *truthy* `eos_token_id` override from optional
/// `<dir>/generation_config.json`, **shape-preserving** (scalar → `Single`,
/// list → `Many`), mirroring `mlx_lm.utils.load_config:276-277`
/// (`if eos_token_id := generation_config.get("eos_token_id", False):
/// config["eos_token_id"] = eos_token_id`). `Some` only when the value is
/// truthy (scalar `0` and empty list are falsy → `None`); `except
/// json.JSONDecodeError: pass`, so an absent / malformed / missing-key /
/// non-regular / oversized file simply yields `None` (never errors — this is
/// optional metadata). Same bounded / `O_NONBLOCK` / non-regular-reject
/// discipline as [`load_config`] so a planted FIFO `generation_config.json`
/// cannot hang or OOM the loader. [`load_config`] writes this back into the
/// returned `Config.eos_token_id` (exactly Python's in-place overwrite) so
/// the returned config and the tokenizer's eos set always agree.
///
/// `pub(crate)` so the [`crate::vlm::load`] base-config reader applies the
/// same override through the same code path (mlx-vlm's `load_config` has
/// the identical generation-config block — `mlx_vlm/utils.py:506-515`).
pub(crate) fn read_generation_eos(dir: &Path) -> Option<EosTokenId> {
  let path = dir.join("generation_config.json");

  // mlx-lm's `except json.JSONDecodeError: pass` is widened here: any
  // bounded-read failure (absent, non-regular, oversized, IO failure,
  // non-UTF-8) AND any subsequent JSON-parse failure collapses to `None`,
  // since this is optional metadata — exactly the Python `except: pass`
  // semantics. The bounded-read primitive itself enforces the hardening
  // (FIFO/oversized/non-regular rejection happens BEFORE we ever try to
  // parse, so a planted FIFO cannot hang here even though we ignore the
  // resulting error).
  let bytes = read_bounded_config_file(&path, "generation config")
    .ok()
    .flatten()?;
  let v = serde_json::from_str::<serde_json::Value>(&bytes).ok()?;
  match v.get("eos_token_id") {
    // mlx-lm overwrites only when truthy (`if eos_token_id := ...`): a
    // scalar `0` is falsy → `None`; a NON-empty list is truthy regardless
    // of contents (so `[0, ..]` keeps the `0`); an empty list is falsy →
    // `None`. Shape is preserved (scalar → `Single`, list → `Many`) so the
    // value written back into `Config.eos_token_id` matches Python's
    // `config["eos_token_id"] = eos_token_id` exactly.
    Some(serde_json::Value::Number(n)) => n
      .as_u64()
      .filter(|&x| x != 0)
      .and_then(|x| u32::try_from(x).ok())
      .map(EosTokenId::Single),
    Some(serde_json::Value::Array(a)) if !a.is_empty() => Some(EosTokenId::Many(
      a.iter()
        .filter_map(|e| e.as_u64().and_then(|x| u32::try_from(x).ok()))
        .collect(),
    )),
    _ => None,
  }
}

/// Load a model directory's architecture-agnostic parts.
///
/// Mirrors `mlx_lm.utils.load` restricted to the local-path, no-network,
/// no-model-construction surface. It reads `config.json` into a [`Config`],
/// discovers and merges the weights via [`load_weights`], then builds the
/// #18 [`Tokenizer`] from the same directory (`tokenizer.json` plus an
/// optional `tokenizer_config.json`, through
/// [`Tokenizer::from_path`](crate::tokenizer::Tokenizer::from_path)) with
/// the mlx-lm-resolved eos set: a *truthy* `generation_config.json`
/// `eos_token_id` overrides `config.json`'s, and the result (if any)
/// REPLACES the tokenizer-config default entirely — exactly
/// `mlx_lm.utils.load_config` + `TokenizerWrapper`. The returned
/// `(Config, Weights, Tokenizer)` triple is what a per-usecase
/// architecture then assembles (and, if `Config.quantization` is set,
/// quantizes) itself.
///
/// Every recoverable failure (missing / oversized / invalid `config.json`,
/// missing weights, tokenizer load) is an [`Error::Backend`] whose message
/// names the offending path. No implicit eval — the returned weight
/// `Array`s are not materialized here.
///
/// [`Tokenizer`]: crate::tokenizer::Tokenizer
pub fn load(dir: &Path) -> Result<(Config, Weights, crate::tokenizer::Tokenizer)> {
  // Thin wrapper over the finer pieces, in model-directory order: parse the
  // config once (which also applies the `generation_config.json` eos override,
  // so the returned config and the tokenizer's eos set always agree —
  // `load(return_config=True)` parity), discover/merge the weights, then build
  // the tokenizer from the SAME directory with the resolved eos set. The
  // post-override `config.eos_token_id` is the tokenizer's COMPLETE set —
  // `TokenizerWrapper` `set(eos_token_ids)` REPLACES the tokenizer-config
  // default (NOT unioned); `None` ⇒ the tokenizer's own `eos_token`.
  let (config, _config_json) = load_config(dir)?;
  let weights = load_weights(dir)?;
  let tokenizer = load_tokenizer(dir, &config)?;
  Ok((config, weights, tokenizer))
}

/// Build the #18 [`Tokenizer`](crate::tokenizer::Tokenizer) from `dir` with
/// the eos set already resolved on `config` (its post-override
/// `eos_token_id` — see [`load_config`]). Factored out of [`load`] so the
/// [`crate::lm::factory`] loader builds the tokenizer from its
/// (optionally separate) tokenizer directory through the *same* eos-resolution
/// path. A tokenizer-load failure is a recoverable [`Error::Backend`] naming
/// the directory.
pub fn load_tokenizer(dir: &Path, config: &Config) -> Result<crate::tokenizer::Tokenizer> {
  load_tokenizer_with_eos(dir, config.eos_token_id.as_ref())
}

/// Build the #18 [`Tokenizer`](crate::tokenizer::Tokenizer) from `dir`,
/// given the already-resolved [`EosTokenId`] (mirroring `TokenizerWrapper`'s
/// `eos_token_ids`: if `Some(ids)` it REPLACES the tokenizer-config default
/// entirely; `None` ⇒ the tokenizer's own `eos_token`). The single
/// `eos_token_ids`-aware tokenizer-build primitive every loader funnels
/// through — both [`load_tokenizer`] (which reads it off [`Config`]) and
/// the [`crate::vlm::load`] base-config loader (whose minimal
/// `VlmBaseConfig` carries the same `eos_token_id`) call this so the eos
/// resolution stays uniform across LM and VLM.
pub fn load_tokenizer_with_eos(
  dir: &Path,
  eos_token_id: Option<&EosTokenId>,
) -> Result<crate::tokenizer::Tokenizer> {
  let resolved_eos = eos_token_id.cloned().map(EosTokenId::into_ids);
  crate::tokenizer::Tokenizer::from_path(dir, resolved_eos.as_deref()).map_err(|e| Error::Backend {
    message: format!("cannot load tokenizer from {}: {e}", dir.display()),
  })
}

// ───────────────────────────── save side ─────────────────────────────
//
// The model-**save** half of `mlx_lm.utils`: shard a weight map, write the
// shards plus the `model.safetensors.index.json` index, write back a sorted
// `config.json`, and the `save` driver. All of these work on the
// arch-agnostic [`Weights`] name map (mlx-lm's
// `dict(tree_flatten(model.parameters()))`) — mlxrs has no `nn.Module` tree
// to flatten (the per-usecase architecture owns that), so a caller flattens
// its own model into a [`Weights`] map and hands it here, exactly as
// [`crate::lm::quant`] takes a [`Weights`] map rather than walking modules.

/// Default per-shard size cap, in **gigabytes**, mirroring
/// `mlx_lm.utils.MAX_FILE_SIZE_GB` (`utils.py:57`). [`make_shards`] /
/// [`save_model`] split a weight map so no single `.safetensors` shard
/// exceeds this; the on-the-wire byte cap is `MAX_FILE_SIZE_GB << 30`
/// (`utils.py:609` — gibibytes, `1 << 30` per "GB").
pub const MAX_FILE_SIZE_GB: u64 = 5;

/// In-memory byte size of one [`Dtype`] element — mlx-c's `mlx_dtype_size`,
/// reproduced as a pure Rust mapping so `array_nbytes` needs no FFI/eval
/// (twin of `crate::lm::cache`'s private `dtype_size`, restated here so the
/// save side carries no `cache`-module dependency).
fn dtype_size(d: Dtype) -> usize {
  match d {
    Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
    Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
    Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
    Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
  }
}

/// Byte size of a weight `Array` — `elem_count * dtype_size` (mlx's
/// `array.nbytes`, which mlx-lm's `make_shards` / `save_model` /
/// `compute_bits_per_weight` all read). A **pure metadata read**: it forces
/// no `eval` and allocates nothing (no implicit eval — spec rule). An
/// unrecognized dtype surfaces as a recoverable [`Error::Backend`].
fn array_nbytes(a: &Array) -> Result<usize> {
  Ok(a.size() * dtype_size(a.dtype()?))
}

/// One weight shard: a **borrowed** sub-view of a [`Weights`] map whose
/// arrays' combined `array_nbytes` respects the [`make_shards`] size cap
/// (except a lone over-cap weight, which still gets its own shard — see
/// [`make_shards`]). Borrowed (`&'a str` keys, `&'a Array` values), not
/// owned: [`make_shards`] **partitions** the input map rather than
/// duplicating any `Array` — `Array` is intentionally not `Clone` (a
/// refcount-sharing duplicate is [`Array::try_clone`](crate::array::Array::try_clone)),
/// and a save partition has no reason to clone. Written verbatim by
/// [`crate::io::save_safetensors_view`] (the no-clone shard writer).
pub type Shard<'a> = BTreeMap<&'a str, &'a Array>;

/// Split a weight map into smaller shards, mirroring
/// `mlx_lm.utils.make_shards` (`utils.py:598-619`).
///
/// Weights are accumulated into the current shard until adding the next one
/// would push the shard past `max_file_size_gb` gibibytes
/// (`max_file_size_bytes = max_file_size_gb << 30`, `utils.py:609`); at that
/// point the current shard is flushed and a fresh one started. The check is
/// `shard_size + v.nbytes > max_file_size_bytes` **before** the weight is
/// added (`utils.py:613`) — verbatim, with **no** empty-shard guard, exactly
/// as the reference. Two edge cases follow directly from that and are
/// reproduced faithfully (F6 is a faithful port — it matches mlx-lm even in
/// the edge cases):
///
/// - A single weight larger than the cap is flushed onto its own shard and
///   still placed (the cap is never *enforced*, only used as a split point).
/// - When the **first** sorted weight already exceeds the cap (in
///   particular for a `0` cap, where `0 + nbytes > 0` for every non-empty
///   weight), the flush fires while the current `shard` is still **empty**,
///   so the result begins with an **empty leading shard** — e.g. a `0` cap
///   over weights `a, b, c, d` yields `[{}, {a}, {b}, {c}, {d}]`, and a `0`
///   cap over a lone weight `solo` yields `[{}, {solo}]`. mlx-lm produces
///   the same empty leading shard, so the shard list (and thus shard file
///   names + index data) matches the reference.
///
/// The result is always non-empty: a final (possibly empty, if `weights`
/// itself was empty) shard is always appended (`utils.py:618`).
///
/// Python `dict` preserves insertion order, so `make_shards` is
/// order-sensitive; a Rust [`HashMap`] is unordered, so this port iterates
/// the keys in **sorted order** — the same determinism convention
/// [`load_weights`] applies to its shard merge. Shard *contents* (which keys
/// land together) are therefore deterministic and reproducible; the final
/// `model.safetensors.index.json` `weight_map` [`save_model`] writes is
/// sorted regardless, so a load-back is byte-identical either way.
///
/// The returned [`Shard`]s **borrow** from `weights` — no `Array` is cloned
/// (each shard is a partition, not a copy).
///
/// `max_file_size_gb` is in gibibytes (the reference's "GB" = `1 << 30`); a
/// `0` cap puts every weight on its own shard (`0 + n > 0` for any non-empty
/// weight triggers a flush each iteration) — with the very first flush
/// emitting an empty leading shard, see above. A weight whose dtype is
/// unrecognized surfaces as a recoverable [`Error::Backend`] from
/// `array_nbytes`.
pub fn make_shards(weights: &Weights, max_file_size_gb: u64) -> Result<Vec<Shard<'_>>> {
  // `gb << 30` in `u64` so there is no `usize` truncation on a 32-bit host;
  // `MAX_FILE_SIZE_GB` (5) << 30 is ~5.4e9, well within `u64`.
  let max_file_size_bytes: u64 = max_file_size_gb << 30;

  // Deterministic, reproducible split order: sorted keys (a `HashMap` has no
  // insertion order, unlike the Python `dict` the reference iterates).
  let sorted: BTreeMap<&str, &Array> = weights.iter().map(|(k, v)| (k.as_str(), v)).collect();

  let mut shards: Vec<Shard<'_>> = Vec::new();
  let mut shard: Shard<'_> = BTreeMap::new();
  let mut shard_size: u64 = 0;
  for (k, v) in sorted {
    let nbytes = array_nbytes(v)? as u64;
    // mlx-lm `utils.py:613`: flush BEFORE adding when the running size plus
    // the next weight would exceed the cap. The split condition is verbatim
    // `shard_size + v.nbytes > max_file_size_bytes` — the reference has NO
    // empty-shard guard, so a `0` cap, or a first sorted weight already
    // over the cap, flushes the *empty* leading `shard` before placing that
    // weight. This port replicates that faithfully (a leading empty shard
    // is part of mlx-lm's edge-case output). `saturating_add` keeps a
    // pathological multi-exabyte map from wrapping the comparison (it would
    // only ever push the sum *higher*, never spuriously under the cap).
    if shard_size.saturating_add(nbytes) > max_file_size_bytes {
      shards.push(std::mem::take(&mut shard));
      shard_size = 0;
    }
    shard.insert(k, v);
    shard_size = shard_size.saturating_add(nbytes);
  }
  // mlx-lm always appends the trailing shard (`utils.py:618`), even when
  // `weights` is empty — then the single shard is itself empty.
  shards.push(shard);
  Ok(shards)
}

/// Total parameter count of a weight map, mirroring
/// `mlx_lm.utils.get_total_parameters` (`utils.py:196-207`).
///
/// The reference walks the model's leaf `nn.Module`s: a **quantized** module
/// (one carrying a `bits` attribute) contributes `m.bias.size + weight.size *
/// 32 // bits` — the *logical* (unpacked) parameter count, since a quantized
/// `weight` is a `uint32`-packed matrix holding `32 / bits` logical weights
/// per element — while a dense module contributes the plain sum of its
/// parameters' `.size`. Note `m.bias` there is the quantized module's
/// optional **real** bias (the `+ bias` of a biased linear), *not* its
/// affine `scales`/`biases` buffers — those are never summed, since
/// `get_total_parameters` reaches a quantized module through the special
/// `hasattr(m, "bits")` branch and never falls into the dense
/// `tree_flatten(m.parameters())` sum.
///
/// mlxrs has no `nn.Module` tree, so this port walks the [`Weights`] **name
/// map** (exactly as [`crate::lm::quant`] does). A quantized layer is
/// detected by a `<path>.scales` sibling next to a `<path>.weight` — the
/// very signal mlx-lm's own loader uses (`class_predicate`'s `f"{p}.scales"
/// in weights`, `utils.py:354`). For such a layer the packed `<path>.weight`
/// (a `uint32` matrix) contributes `weight.size * 32 / bits` logical
/// parameters; the `<path>.scales` **and** the `<path>.biases` arrays — both
/// affine-quantization metadata (the `affine_quantize` scale + zero-point
/// buffers, see [`crate::lm::quant`]), not model parameters — are, like the
/// reference, **not** counted. Every other array contributes its plain
/// `.size`: a dense `.weight`, a genuine module `.bias` (singular — a real
/// linear bias, with no `.scales` sibling), norms, embeddings, … — the dense
/// `sum(v.size …)` branch.
///
/// `bits` per quantized layer is resolved through
/// [`PerLayerQuantization::quantization_for`](crate::lm::quant::PerLayerQuantization::quantization_for)
/// (`quant` carries the global + per-layer overrides). A quantized triple
/// whose layer has no resolvable [`Quantization`] in `quant` is a
/// configuration error → [`Error::Backend`].
///
/// A **pure metadata read**: no `eval`, no allocation of weight data.
pub fn get_total_parameters(
  weights: &Weights,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<u64> {
  let mut total: u64 = 0;
  for (key, arr) in weights {
    // A quantized `<path>.weight` is the one whose `<path>.scales` sibling
    // is also present (mlx-lm's `f"{p}.scales" in weights` signal). Its
    // packed `uint32` matrix unpacks to `size * 32 / bits` logical weights.
    if let Some(path) = key.strip_suffix(".weight") {
      let scales_key = format!("{path}.scales");
      if weights.contains_key(&scales_key) {
        let q = quant.quantization_for(path).ok_or_else(|| Error::Backend {
          message: format!(
            "get_total_parameters: quantized layer {path:?} (has `.scales`) \
             but no quantization params for it in the config"
          ),
        })?;
        if q.bits <= 0 {
          return Err(Error::Backend {
            message: format!(
              "get_total_parameters: quantized layer {path:?} has non-positive bits {}",
              q.bits
            ),
          });
        }
        // `weight.size * 32 // bits` (`utils.py:204`), in `u64` so a large
        // packed matrix cannot overflow the multiply on a 32-bit host.
        total = total.saturating_add(arr.size() as u64 * 32 / q.bits as u64);
        continue;
      }
    }
    // A `<path>.scales` for a quantized layer is metadata, not a parameter
    // — the reference's quantized branch counts only the unpacked `weight`
    // and a real module `bias` (`utils.py:203-204`). Skip a `.scales` whose
    // `.weight` sibling exists.
    if let Some(path) = key.strip_suffix(".scales")
      && weights.contains_key(&format!("{path}.weight"))
    {
      continue;
    }
    // A `<path>.biases` (note the trailing `s`) sitting next to a
    // `<path>.weight` + `<path>.scales` triple is the **affine**
    // `affine_quantize` zero-point buffer — metadata, not a model
    // parameter (it is the quantized module's `biases` buffer, never
    // summed by mlx-lm's quantized branch; counting it would inflate
    // `total_parameters` and depress `compute_bits_per_weight`).
    //
    // But that is true ONLY under affine quantization: the scale-only
    // schemes (`mxfp4` / `mxfp8` / `nvfp4`) have NO `.biases` output and
    // explicitly reject one (`mlx/ops.cpp:5085-5099`; see
    // [`crate::lm::quant`]). So resolve the layer's [`QuantMode`] before
    // deciding — skip the `.biases` as metadata only for
    // [`QuantMode::Affine`]; under a scale-only mode a `.biases` sibling
    // is structurally invalid data and is flagged as an
    // [`Error::Backend`] rather than silently dropped. A genuine dense
    // module bias is named `.bias` (singular, no `.scales` sibling) and
    // falls through to the dense count below regardless.
    if let Some(path) = key.strip_suffix(".biases")
      && weights.contains_key(&format!("{path}.weight"))
      && weights.contains_key(&format!("{path}.scales"))
    {
      use crate::lm::quant::QuantMode;
      // The same `quantization_for` resolution the `.weight` branch uses
      // (an unresolvable quantized layer is the same configuration error
      // there; HashMap iteration may reach `.biases` first, so error
      // here too rather than relying on the `.weight` visit).
      let q = quant.quantization_for(path).ok_or_else(|| Error::Backend {
        message: format!(
          "get_total_parameters: quantized layer {path:?} (has `.weight` \
           + `.scales` + `.biases`) but no quantization params for it in \
           the config"
        ),
      })?;
      match q.mode {
        // Affine zero-point buffer — metadata, skip (do not count).
        QuantMode::Affine => continue,
        // mxfp4 / mxfp8 / nvfp4 carry no `.biases`; a present one means
        // an invalid checkpoint — flag it, do not silently drop it.
        QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4 => {
          return Err(Error::Backend {
            message: format!(
              "get_total_parameters: layer {path:?} is quantized with \
               scale-only mode `{}`, which has no `.biases` buffer, yet a \
               `{key}` tensor is present — invalid checkpoint",
              q.mode.as_mlx_str()
            ),
          });
        }
      }
    }
    // Everything else — a dense `.weight`, a real module `.bias`, the
    // unpacked-bias of a quantized layer counted via its packed `.weight`
    // above, norms, embeddings — is a plain parameter counted by its
    // element count (the dense `sum(v.size for _, v in
    // tree_flatten(m.parameters()))` branch).
    total = total.saturating_add(arr.size() as u64);
  }
  Ok(total)
}

/// Bits-per-weight of a weight map, mirroring
/// `mlx_lm.utils.compute_bits_per_weight` (`utils.py:210-215`).
///
/// `model_bytes * 8 / model_params`, where `model_bytes` is the sum of every
/// array's `array_nbytes` (the reference's `tree_reduce(... + x.nbytes
/// ...)`, `utils.py:211-213` — it sums **every** array, `scales` / `biases`
/// included) and `model_params` is [`get_total_parameters`]. The result is
/// the average number of stored bits backing each logical parameter; for a
/// `b`-bit quantized model it lands near `b` plus the scale/bias overhead.
///
/// `quant` supplies the per-layer [`Quantization`] [`get_total_parameters`]
/// needs. A `model_params == 0` map (no weights) is an [`Error::Backend`]
/// rather than a divide-by-zero. A **pure metadata read** — no `eval`.
pub fn compute_bits_per_weight(
  weights: &Weights,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<f64> {
  let mut model_bytes: u64 = 0;
  for arr in weights.values() {
    model_bytes = model_bytes.saturating_add(array_nbytes(arr)? as u64);
  }
  let model_params = get_total_parameters(weights, quant)?;
  if model_params == 0 {
    return Err(Error::Backend {
      message: "compute_bits_per_weight: model has zero parameters".into(),
    });
  }
  Ok(model_bytes as f64 * 8.0 / model_params as f64)
}

/// Whether `model` accepts pre-computed input embeddings, mirroring
/// `mlx_lm.utils.does_model_support_input_embeddings` (`utils.py:979-991`).
///
/// The reference inspects `model.__call__` for an `input_embeddings`
/// parameter. mlxrs has no runtime call-signature introspection, so the
/// capability is declared on the [`Model`](crate::lm::model::Model) trait
/// itself: this is a thin forward to
/// [`Model::supports_input_embeddings`](crate::lm::model::Model::supports_input_embeddings)
/// (text-only models inherit the `false` default; a VLM that overrides
/// [`forward_embeddings`](crate::lm::model::Model::forward_embeddings) also
/// overrides that to `true`). Kept as a free function so the public name
/// matches the reference helper.
pub fn does_model_support_input_embeddings(model: &dyn crate::lm::model::Model) -> bool {
  model.supports_input_embeddings()
}

/// Process-global counter feeding [`new_gen_id`]: a `fetch_add(1)` per save
/// gives every save *from this process* a distinct counter value, closing
/// the same-millisecond / same-microsecond collision the F6 R5 timestamp-
/// only tag left open. Combined with the PID (unique among live processes)
/// and the µs timestamp (monotone-ish across reboots) this makes the
/// resulting `gen_id` collision-resistant by construction.
static SAVE_GEN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Test-only override for `new_gen_id`: when `Some(s)`, the next call
// returns the override (cloned) and clears the override; otherwise the
// regular `{ts_us}-{pid}-{ctr}` path runs. Used by the
// `save_model_refuses_to_overwrite_existing_shard_basename` test to
// deterministically force a save to predict a specific basename so we
// can plant a colliding file at that exact path and assert
// `Error::ShardPathCollision` fires.
#[cfg(test)]
thread_local! {
  static GEN_ID_OVERRIDE: std::cell::RefCell<Option<String>> =
    const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn force_next_gen_id(id: &str) {
  GEN_ID_OVERRIDE.with(|c| *c.borrow_mut() = Some(id.to_string()));
}

/// Build a collision-resistant **generation id** for one save: a
/// `{ts_us:013}-{pid:08x}-{ctr:016x}` tag captured once at the start of
/// [`save_model`] so every shard of a single save shares it.
///
/// Why three components:
///
/// - `ts_us` (microseconds since the UNIX epoch, zero-padded to 13 digits):
///   monotone-ish across reboots, so two saves separated by even a single
///   filesystem tick land on different basenames; a clock error degrades to
///   `0` rather than failing the save.
/// - `pid` (the process id, lowercase hex, 8 digits): unique among **live
///   processes**, so two saves from two different processes — say, two
///   concurrent CLI invocations — can never collide on the PID component.
/// - `ctr` (a process-global atomic `fetch_add(1)`, 16 hex digits): unique
///   *within* this process across the lifetime of the binary, so two saves
///   from the same process in the same microsecond can never collide on the
///   counter. The mlx-lm reference produced fixed names that *did* collide;
///   the F6 R5 timestamp-only tag collided whenever two saves landed in the
///   same millisecond. This counter closes that hole.
///
/// Concatenated together the resulting tag is collision-resistant by
/// construction — no collision surface remains for any two `save_model`
/// calls anywhere in any process.
fn new_gen_id() -> String {
  // Test-only one-shot override (always `None` in production builds —
  // the thread-local lives behind `#[cfg(test)]`).
  #[cfg(test)]
  if let Some(forced) = GEN_ID_OVERRIDE.with(|c| c.borrow_mut().take()) {
    return forced;
  }

  use std::{
    sync::atomic::Ordering,
    time::{SystemTime, UNIX_EPOCH},
  };
  let ts_us = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_micros())
    .unwrap_or(0);
  let pid = std::process::id();
  let ctr = SAVE_GEN_COUNTER.fetch_add(1, Ordering::Relaxed);
  format!("{ts_us:013}-{pid:08x}-{ctr:016x}")
}

/// Per-shard file name, **generation-tagged** to avoid colliding with any
/// pre-existing shard on disk. The reference (`utils.py:728-732`) names a
/// lone shard `model.safetensors` and a multi-shard set
/// `model-{:05d}-of-{:05d}.safetensors`; both of those *do* collide with a
/// previously-saved checkpoint's shards on the same path, which then lets a
/// torn rename between the shard publish and the index publish corrupt the
/// OLD checkpoint (the OLD index still points at a path whose content is now
/// the NEW shard's bytes — `save_model` step 5's caveat).
///
/// mlxrs avoids that class of corruption by using a **generation-unique**
/// basename — `model-gen-{gen_id}-{idx:05}-of-{N:05}.safetensors` — with
/// `gen_id` built once at the start of [`save_model`] from a µs timestamp,
/// the PID, and a process-global counter (see [`new_gen_id`]). Two saves
/// from the same process can never collide on the counter; two saves from
/// different processes can never collide on the PID; across reboots the
/// timestamp is monotone-ish. The result: no collision surface remains.
///
/// The single-shard case uses the uniform `…-00001-of-00001` form (no
/// special case) so loader code never has to distinguish single- and
/// multi-shard layouts.
///
/// The index records these exact names verbatim, and the loader
/// ([`load_weights`]) follows the index, so any shard basename — including
/// these generation-tagged ones — works transparently.
fn shard_file_name(gen_id: &str, index_1based: usize, shards_count: usize) -> String {
  format!("model-gen-{gen_id}-{index_1based:05}-of-{shards_count:05}.safetensors")
}

/// Outcome of [`save_model`]'s **observable commit point** (the index
/// rename).
///
/// `save_model` returns `Err` ONLY for **pre-commit** failures — staging a
/// shard, renaming a shard tempfile, or renaming the index. Once the index
/// rename succeeds the NEW checkpoint IS observable on disk and
/// [`load_weights`] would now load it. A subsequent `fsync_dir` failure
/// does NOT roll the rename back (it cannot — the rename is durable in
/// the directory inode, just possibly not yet durable in the on-disk
/// directory metadata).
///
/// To distinguish the two cases cleanly, `save_model` returns
/// `Result<CommitOutcome, Error>`:
///
/// - `Ok(Committed)` — index rename succeeded **and** the post-commit
///   `fsync_dir` of the parent directory succeeded; the save is fully
///   durable.
/// - `Ok(CommittedWithDurabilityWarning(_))` — index rename succeeded, the
///   visible checkpoint loads correctly, but the post-commit `fsync_dir`
///   returned an error. The caller (e.g. [`save`]) MUST still proceed to
///   commit any other staged state (the config), and surface the warning
///   to its own caller via [`crate::Error::DurabilityWarning`].
///
/// This contract keeps the [`save`] driver from dropping a staged
/// config-tempfile guard (and its cleanup-on-drop tempfile delete) just
/// because a post-commit fsync hiccupped — which would leave NEW
/// weights+index visible against the OLD config, the exact mismatch
/// Finding 2 flagged.
#[derive(Debug)]
pub enum CommitOutcome {
  /// The index rename succeeded and the post-commit parent-directory
  /// `fsync` succeeded — the save is fully durable.
  Committed,
  /// The index rename succeeded (the NEW checkpoint IS visible on disk and
  /// would be observed by [`load_weights`]), but the post-commit parent-
  /// directory `fsync` returned an error. The save is logically committed
  /// but the directory-entry write may not yet be durable on disk.
  CommittedWithDurabilityWarning(std::io::Error),
}

/// Write a weight map as sharded `.safetensors` plus the
/// `model.safetensors.index.json` weight-map index into `save_path`,
/// mirroring `mlx_lm.utils.save_model` (`utils.py:714-771`) with three
/// **structural** additions: (1) the [`load_weights`] side reads the index
/// as the authoritative shard manifest, so the index rename here becomes
/// the **SINGLE** commit point — a mid-save crash leaves the previously-
/// valid checkpoint atomically intact; (2) shard basenames carry a
/// **collision-resistant generation id**
/// (`model-gen-{gen_id}-{idx:05}-of-{N:05}.safetensors`, with `gen_id` =
/// `{ts_us}-{pid:hex}-{ctr:hex}` built once at save start from the
/// microsecond timestamp, the PID, and a process-global atomic counter)
/// so a new save's shards can never overwrite an older checkpoint's
/// shards on disk regardless of clock or process; (3) the shard publish
/// uses an **atomic no-replace `hard_link`** rather than a `rename` —
/// `link(2)` succeeds creating a second directory entry at the FINAL
/// path or fails `EEXIST`, with no replace window, so a pre-existing
/// **final-path** entry (collision-resistant `gen_id` collision, stale
/// leftover, or a peer racing for the same FINAL name) surfaces as
/// [`crate::Error::ShardPathCollision`] in a single syscall — a
/// `symlink_metadata` + `rename` pre-check would have a TOCTOU gap
/// where `rename(2)` silently REPLACES the racing peer's bytes between
/// the stat and the rename. The index continues to use `rename` (its
/// commit semantics ARE "the latest rename wins"; it intentionally
/// overwrites the OLD index). **The `EEXIST` failure mode protects
/// against collisions at the FINAL shard path only; it does NOT
/// protect against staging-directory races on the temp NAME — see the
/// "Hostile-directory caveat" below.**
///
/// Steps, in reference order:
///
/// 1. `save_path` is created (`mkdir -p`, `utils.py:723`); a single
///    collision-resistant generation id (`ts_us`-`pid`-`ctr`) is
///    captured up front.
/// 2. The weights are sharded via [`make_shards`] at [`MAX_FILE_SIZE_GB`]
///    (`utils.py:726`).
/// 3. `total_size` (the sum of every weight's `array_nbytes`) and
///    `total_parameters` ([`get_total_parameters`]) are computed for the
///    index `metadata` block (`utils.py:734-741`).
/// 4. Each shard is written — with the `{"format": "mlx"}` safetensors
///    metadata mlx writes (`utils.py:756`) and the generation-tagged
///    `shard_file_name` — to a **same-directory, exclusively created
///    (`O_EXCL`) `.safetensors` tempfile**, which is then **fsync**ed.
///    The `index.json` body is likewise serialized to its own same-
///    directory `O_EXCL` tempfile and fsynced. **Nothing is written to a
///    final path yet.** Any failure here removes every staged tempfile.
/// 5. **Publish — single commit point.** Every shard tempfile is
///    published to its final shard name FIRST via an **atomic no-
///    replace `hard_link`** (`link(2)`) on the temp PATHNAME: a second
///    directory entry is created at the final shard path pointing at
///    whatever inode the temp NAME currently resolves to, or the call
///    fails `EEXIST` and the save aborts with
///    [`crate::Error::ShardPathCollision`] (no silent-replace window at
///    the FINAL path — unlike `rename(2)`); the tempfile is then
///    unlinked, freeing the temp name while the inode survives via the
///    final-path entry. **`hard_link` operates by pathname**, so the
///    inode that ends up published is the inode the temp NAME resolves
///    to AT THE LINK CALL — see the "Hostile-directory caveat" below
///    for what that means when the staging directory is user-writable. The parent directory is `fsync`ed so those
///    `link` + unlink entries are durable. The staged index is then
///    atomically `rename`d over `model.safetensors.index.json` LAST
///    (preceded by a `symlink_metadata` check — `NotFound` proceeds,
///    anything else errors; the loader IS allowed to read the OLD
///    `index.json`, and that file is the ONE final path we
///    intentionally overwrite). The parent directory is `fsync`ed again
///    to make the index rename durable. The index rename is THE
///    observable commit point: before it, load follows the OLD index
///    (if any) to the OLD shards (any new-named shards on disk are
///    invisible — [`load_weights`] only loads what the index lists);
///    after it, load follows the NEW index to the new shards. POSIX
///    `rename(2)` is atomic-within-fs and `link(2)` is atomic no-
///    replace (same-dir tempfiles keep both single-fs — `EXDEV` is
///    impossible by construction); the new checkpoint becomes visible
///    only at that final index rename. Because every NEW shard's
///    basename carries the collision-resistant `gen_id` it can never
///    collide with an OLD shard — a failure between the shard publish
///    and the index rename leaves the OLD index → OLD shards untouched
///    and still loadable; the not-yet-published staged index is
///    removed, and the linked new-named shards become silently-orphan
///    files (load ignores them).
///
/// **Failure-vs-commit boundary (`CommitOutcome`).** `save_model` returns
/// `Result<CommitOutcome, Error>`:
///
/// - `Err(_)` is returned **only** for pre-commit failures (directory
///   create, staging, shard publish — including the atomic no-replace
///   `hard_link` defense-in-depth that catches a colliding final path,
///   the index rename itself). On any such failure the OLD checkpoint
///   is byte-identical to its pre-save state.
/// - Once the index rename succeeds the NEW checkpoint IS observable and
///   `save_model` returns `Ok(_)`:
///   - `Ok(CommitOutcome::Committed)` — the post-commit parent-directory
///     `fsync` also succeeded; the save is fully durable.
///   - `Ok(CommitOutcome::CommittedWithDurabilityWarning(e))` — the
///     post-commit `fsync_dir` returned `e`. The visible checkpoint
///     loads correctly; the directory-entry write may not yet be durable
///     (a power loss before the FS internally drains could lose the
///     directory entry). The caller (e.g. [`save`]) MUST still proceed
///     to commit any other staged state.
///
/// This contract closes the F6 R6 Finding-2 hole where a post-index-
/// commit `fsync_dir` failure would propagate as `Err` and drop the
/// staged config-tempfile guard (deleting its tempfile via [`Drop`]),
/// leaving NEW weights+index visible against the OLD config.
///
/// # Hostile-directory caveat
///
/// Shards are written through [`crate::io::save_safetensors_to_file`]
/// (fd-bound — every byte goes to the inode this function opened) and
/// published via `std::fs::hard_link(temp_path, final_path)` + unlink.
/// The fd-bound write itself is protected against `unlink + symlink`
/// write-redirection on the temp NAME: the bytes go to the inode the
/// crate opened, regardless of what the temp directory entry currently
/// points to.
///
/// **However, the publication step links BY PATHNAME.** If the staging
/// directory (`save_path`) is user-writable, an attacker with write
/// access can `unlink(temp_path)` and substitute their own file at the
/// same name AT ANY TIME between the `O_EXCL` create and the
/// `hard_link` — including after the fd-bound write and fsync. The
/// `hard_link` then resolves the temp pathname to the attacker's inode
/// and atomically publishes IT at the final shard name; the fd-written
/// bytes survive only as an orphan inode no directory entry points to.
/// `linkat` / `renameat` with a directory fd do NOT close this race —
/// they anchor the parent directory but still look the temp entry up by
/// name. The `EEXIST` failure mode described above only protects
/// against pre-existing entries (or concurrent peers) at the **FINAL**
/// shard path; it does NOT protect against staging-directory races on
/// the temp NAME.
///
/// **For full hostile-directory safety**, pass a `save_path` that is
/// a **trusted (not user-writable) directory** — the simplest and most
/// portable solution. The publication step is then safe because no
/// unprivileged attacker can substitute the temp entry between fsync
/// and link. See the **"Scope of this guarantee"** section of
/// [`crate::io::save_safetensors_to_file`] for the broader discussion
/// of publication-step trust requirements (and the platform-specific
/// fd-bound publish primitives this crate does NOT provide).
///
/// **Prior-generation shards on disk.** This implementation **does not**
/// inline-prune `model*.safetensors` files left over from earlier
/// checkpoints. The loader follows the NEW index, never a glob, so those
/// orphans are invisible to load and only leak disk space. An explicit
/// prune API can be added later if needed; doing so as part of `save_model`
/// would re-introduce a race we just removed (an unlink concurrent with a
/// reader's `mmap`).
///
/// The `weight_map` (`weight name → shard file name`) is assembled and
/// **sorted by key** (`utils.py:762-764`); the whole `index_data`
/// (`{ "metadata": { total_size, total_parameters }, "weight_map": … }`)
/// is serialized with 4-space indentation (`json.dump(..., indent=4)`,
/// `utils.py:766-771`).
///
/// `quant` supplies the per-layer [`Quantization`]
/// [`get_total_parameters`] needs (pass
/// [`PerLayerQuantization::default`](crate::lm::quant::PerLayerQuantization)
/// for an unquantized model). Unlike the reference's `donate_model` /
/// `weights.clear()` memory-frugality dance (`utils.py:742-760` — it drops
/// each shard's `Array`s as it goes), this port borrows `&Weights` and never
/// takes ownership, so there is nothing to donate; the on-disk result is
/// identical.
///
/// A recoverable failure (directory create, a shard write, a shard-name
/// collision, the index write) is an [`Error`] naming the path
/// ([`Error::Backend`] for IO failures, [`crate::Error::ShardPathCollision`]
/// for a pre-existing final shard path).
pub fn save_model(
  save_path: &Path,
  weights: &Weights,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<CommitOutcome> {
  // 1. `save_path.mkdir(parents=True, exist_ok=True)` (`utils.py:723`).
  std::fs::create_dir_all(save_path).map_err(|e| Error::Backend {
    message: format!(
      "save_model: cannot create directory {}: {e}",
      save_path.display()
    ),
  })?;

  // Capture a collision-resistant generation id ONCE up front so every
  // shard of this save shares the same tag. The id is
  // `{ts_us:013}-{pid:08x}-{ctr:016x}` — see [`new_gen_id`] for the per-
  // component collision argument. Two saves from the same process can
  // never collide on `ctr`; two saves from different processes can never
  // collide on `pid` (PIDs are unique among LIVE processes); across
  // reboots `ts_us` is monotone-ish. No collision surface remains. The
  // value is not load-critical (the loader follows the index, never
  // parses the basename); a clock error degrades `ts_us` to `0` without
  // weakening the `pid`+`ctr` uniqueness.
  let gen_id = new_gen_id();

  // 2. shard (`utils.py:726`).
  let shards = make_shards(weights, MAX_FILE_SIZE_GB)?;
  let shards_count = shards.len();

  // 3. index `metadata` block (`utils.py:734-741`): `total_size` is the sum
  //    of every weight's `nbytes`, `total_parameters` the unpacked count.
  let mut total_size: u64 = 0;
  for arr in weights.values() {
    total_size = total_size.saturating_add(array_nbytes(arr)? as u64);
  }
  let total_parameters = get_total_parameters(weights, quant)?;

  // 4. Stage to same-directory `O_EXCL` tempfiles + fsync — NOTHING
  //    touches a final path yet. Shards and the index are staged
  //    separately so step 5 can publish shards FIRST and the index LAST
  //    (the single commit point). The `{"format": "mlx"}` safetensors
  //    metadata matches mlx-lm (`mx.save_safetensors(..., metadata={
  //    "format": "mlx"})`, `utils.py:756`). The shards are borrowed views,
  //    so they go through `save_safetensors_view` — no `Array` is cloned.
  let mut shard_metadata: HashMap<String, String> = HashMap::with_capacity(1);
  shard_metadata.insert("format".to_string(), "mlx".to_string());
  // `weight_map` collected sorted-by-key so the written index is
  // deterministic (`utils.py:762-764` sorts it before the `json.dump`).
  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  // `(tmp_path, final_path)` for every shard to be atomically published in
  // step 5. Held so any failure can remove every staged tempfile.
  let mut staged_shards: Vec<(PathBuf, PathBuf)> = Vec::with_capacity(shards_count);
  // Same for the staged index — kept separate so the publish step can
  // commit shards first and the index LAST.
  let mut staged_index: Option<(PathBuf, PathBuf)> = None;

  // Inner closure so ANY failure below cleans up every tempfile staged so
  // far before returning — no leaked `.tmp.safetensors` on a failed save.
  let staged: Result<()> = (|| {
    for (i, shard) in shards.iter().enumerate() {
      let shard_name = shard_file_name(&gen_id, i + 1, shards_count);
      let final_path = save_path.join(&shard_name);
      // Exclusively created, same-directory, `.safetensors`-suffixed
      // tempfile: we keep the open `File` through to the write so the
      // bytes go to THIS fd (no reopen-by-name TOCTOU window).
      let (mut tmp_file, tmp_path) = open_excl_temp_shard(&final_path)?;
      staged_shards.push((tmp_path.clone(), final_path));
      crate::io::save_safetensors_to_file(
        &mut tmp_file,
        shard.iter().map(|(&k, &v)| (k, v)),
        &shard_metadata,
      )?;
      // Pin durability to the same fd `open_excl_temp_shard` returned —
      // a path-reopen here would re-introduce the TOCTOU window the
      // `O_EXCL`-`File`-through-to-write change just closed. The path
      // is still threaded through so the test-only fsync_path injector
      // (`arm_fsync_path_fault*`) keeps firing identically.
      fsync_open_file_for_path(&tmp_file, &tmp_path)?;
      drop(tmp_file);
      for &weight_name in shard.keys() {
        weight_map.insert(weight_name.to_string(), shard_name.clone());
      }
    }

    // assemble `model.safetensors.index.json` with `indent=4`
    // (`utils.py:735-771`). `serde_json::Value` preserves the reference
    // key order (`metadata` before `weight_map`); `weight_map` is a
    // `BTreeMap`, so its keys serialize sorted.
    let index = serde_json::json!({
      "metadata": {
        "total_size": total_size,
        "total_parameters": total_parameters,
      },
      "weight_map": weight_map,
    });
    let index_final = save_path.join("model.safetensors.index.json");
    let (mut index_file, index_tmp) = open_excl_temp_shard(&index_final)?;
    staged_index = Some((index_tmp.clone(), index_final));
    write_json_pretty(
      &mut index_file,
      &index,
      "save_model: model.safetensors.index.json",
    )?;
    fsync_open_file_for_path(&index_file, &index_tmp)?;
    drop(index_file);
    Ok(())
  })();
  if let Err(err) = staged {
    for (tmp, _) in &staged_shards {
      let _ = std::fs::remove_file(tmp);
    }
    if let Some((tmp, _)) = &staged_index {
      let _ = std::fs::remove_file(tmp);
    }
    return Err(err);
  }

  // 5. Publish — single commit point.
  //
  //    a. Publish every shard tempfile to its final shard name FIRST via
  //       an **atomic no-replace `hard_link`**. `link(2)` (`std::fs::
  //       hard_link`) creates a new directory entry at `final_path`
  //       pointing at whatever inode the temp PATHNAME currently
  //       resolves to, or fails with `EEXIST` (`ErrorKind::
  //       AlreadyExists`) — atomically, with no replace window AT THE
  //       FINAL PATH. Unlike a `symlink_metadata` pre-check + `rename`
  //       sequence (which has a TOCTOU gap where a concurrent writer
  //       can race in between the stat and the rename; `rename(2)`
  //       then SILENTLY replaces the racing peer's bytes at the final
  //       path), `hard_link` cannot overwrite a pre-existing final-path
  //       entry — it either creates the final directory entry or
  //       returns the collision error in a single syscall. The tempfile
  //       is then unlinked: the inode survives via the new final-path
  //       entry (refcount stays at 1), so no bytes are lost. Each shard
  //       basename also carries a collision-resistant `gen_id`
  //       (`model-gen-{ts_us}-{pid}-{ctr}-…`) making a FINAL-PATH
  //       collision statistically unreachable; `hard_link`'s atomic
  //       no-replace is the fail-closed defense in depth on the
  //       residual final-path-collision case (`gen_id` collision,
  //       stale leftover, or a peer racing for the same FINAL name).
  //       A collision (or other IO failure) here leaves any still-
  //       staged shard + the staged index as tempfiles, which are
  //       cleaned up before propagating. Already-published shards
  //       remain on disk as silently-invisible files (the OLD index,
  //       if any, still points to its OLD shard names — which are
  //       untouched).
  //
  //       Hostile-directory caveat: `hard_link` resolves the temp
  //       PATHNAME at the link call — so if `save_path` is user-
  //       writable, an attacker who can `unlink(tmp) + create(tmp,
  //       attacker's inode)` between fsync and the link below will
  //       cause `final_path` to publish THEIR inode rather than the
  //       fd-written one. The `EEXIST` branch above does NOT catch
  //       this (it only catches collisions at the FINAL path).
  //       Defense lives at the caller layer — see the
  //       "Hostile-directory caveat" section in `save_model`'s doc
  //       comment for the full discussion and the
  //       trusted-staging-directory mitigation.
  //
  //       Same-filesystem guarantee: `open_excl_temp_shard` creates the
  //       tempfile in `final_path.parent()` (so tmp + final_path always
  //       share a directory), so `EXDEV` cross-device errors from
  //       `hard_link` are impossible by construction.
  for idx in 0..staged_shards.len() {
    let (tmp, final_path) = &staged_shards[idx];
    match std::fs::hard_link(tmp, final_path) {
      Ok(()) => {
        // The bytes are now reachable via `final_path` (a second name
        // for the same inode). Best-effort unlink the tmp name; the
        // inode survives via the final-path entry, so no data loss
        // even if the unlink fails.
        let _ = std::fs::remove_file(tmp);
      }
      Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
        // Atomic no-replace caught a pre-existing entry at the FINAL
        // path (the statistically-unreachable `gen_id` collision, a
        // stale leftover, or a peer racing for the same FINAL name).
        // Note this does NOT catch staging-directory races on the
        // temp NAME — see the "Hostile-directory caveat" in
        // `save_model`'s doc comment. Clean every still-unpublished
        // shard tempfile (including this one) + the staged index,
        // then surface the collision.
        for (leftover, _) in &staged_shards[idx..] {
          let _ = std::fs::remove_file(leftover);
        }
        if let Some((index_tmp, _)) = &staged_index {
          let _ = std::fs::remove_file(index_tmp);
        }
        return Err(Error::ShardPathCollision {
          path: final_path.clone(),
        });
      }
      Err(e) => {
        // Clean every still-unpublished shard tempfile + the staged
        // index, then surface the IO failure.
        for (leftover, _) in &staged_shards[idx..] {
          let _ = std::fs::remove_file(leftover);
        }
        if let Some((index_tmp, _)) = &staged_index {
          let _ = std::fs::remove_file(index_tmp);
        }
        return Err(Error::Backend {
          message: format!(
            "save_model: cannot hard_link shard {} -> {}: {e}",
            tmp.display(),
            final_path.display()
          ),
        });
      }
    }
  }

  //    a'. fsync the parent directory so all shard `hard_link` + unlink
  //        entries are durable on disk before we publish the index.
  //        Without this, a crash between the shard publish and the
  //        index rename can lose the directory entries, leaving the
  //        index referencing shards that appear missing on the next
  //        mount. A no-op on platforms where directory fsync is not
  //        supported.
  //
  //        This fsync is BEFORE the observable commit point (the index
  //        rename), so a failure here is a genuine pre-commit error and
  //        propagates as `Err`. The post-commit fsync (b') is the one
  //        that gets demoted to a `CommitOutcome` warning.
  fsync_dir(save_path).map_err(|e| Error::Backend {
    message: format!(
      "save_model: fsync parent directory {} after shard publish failed: {e}",
      save_path.display()
    ),
  })?;

  //    b. Atomically rename the staged index over
  //       `model.safetensors.index.json` LAST. This is THE observable
  //       commit point — until it succeeds the OLD index (if any) still
  //       drives [`load_weights`]; once it succeeds the NEW index does.
  //       On failure, remove the staged index tempfile and propagate —
  //       the directory now holds renamed new-named shards alongside the
  //       OLD checkpoint (load follows the OLD index → still correct, the
  //       new-named shards are invisible).
  //
  //       The index is the ONE final path we intentionally overwrite —
  //       the load contract is: "the index IS the manifest, the latest
  //       rename wins". So `symlink_metadata` returning `Ok(_)` is
  //       expected here (the OLD index) and is NOT a collision; we still
  //       check for a non-`NotFound`-and-non-`Ok(_)` stat error to avoid
  //       blindly renaming over a fs that refuses to stat.
  let (index_tmp, index_final) = staged_index
    .as_ref()
    .expect("index was successfully staged in step 4");
  // Pre-rename existence stat: the OLD index existing (Ok(_)) is the
  // intentional-overwrite path — the load contract is "the index IS the
  // manifest, the latest rename wins". A `NotFound` (first save into the
  // dir) also proceeds. Anything else (e.g. EACCES) is a genuine stat
  // failure that aborts the save rather than blindly renaming over a
  // filesystem we cannot inspect.
  if let Err(e) = std::fs::symlink_metadata(index_final)
    && e.kind() != std::io::ErrorKind::NotFound
  {
    let _ = std::fs::remove_file(index_tmp);
    return Err(Error::Backend {
      message: format!(
        "save_model: cannot stat index final path {} before rename: {e}",
        index_final.display()
      ),
    });
  }
  if let Err(e) = std::fs::rename(index_tmp, index_final) {
    let _ = std::fs::remove_file(index_tmp);
    return Err(Error::Backend {
      message: format!(
        "save_model: cannot rename index {} -> {}: {e}",
        index_tmp.display(),
        index_final.display()
      ),
    });
  }

  //    b'. fsync the parent directory again so the index rename — THE
  //        observable commit point — is durable. **An error here does
  //        NOT roll the commit back** (the rename is durable in the
  //        directory inode; only the on-disk directory metadata may not
  //        yet be drained). The NEW checkpoint IS observable on disk and
  //        [`load_weights`] would now load it; the caller MUST treat
  //        this as a logically-committed save with a durability warning.
  //
  //        Returned as `Ok(CommittedWithDurabilityWarning(_))` rather
  //        than `Err(_)`, so the [`save`] driver does not drop a still-
  //        staged [`StagedConfig`] (which would delete its tempfile via
  //        [`Drop`]), the F6 R6 Finding-2 hole. The caller surfaces it
  //        to its own caller via [`crate::Error::DurabilityWarning`].
  match fsync_dir(save_path) {
    Ok(()) => Ok(CommitOutcome::Committed),
    Err(e) => Ok(CommitOutcome::CommittedWithDurabilityWarning(e)),
  }
}

/// Open an exclusively created (`O_CREAT|O_EXCL`), randomized tempfile in
/// the SAME directory as `final_path`, of the form
/// `<file_name>.<pid>.<rand>.tmp.safetensors`, and return **both** the open
/// [`File`] **and** its path. Callers continue to write through the
/// returned [`File`] (via [`crate::io::save_safetensors_to_file`] for
/// shard bodies / [`write_json_pretty`] for the index + config JSON) so
/// the original-open identity carries through to every write.
///
/// **TOCTOU rationale.** The earlier shape returned only the [`PathBuf`]
/// and then dropped the open handle, leaving every subsequent write to
/// re-open `path` by name. Between the `O_EXCL` create + that reopen, an
/// attacker with write access to the destination directory could
/// `unlink(path) + symlink(path, /etc/passwd)` and redirect the write —
/// defeating the no-symlink guarantee `O_EXCL` was meant to provide.
/// Returning the [`File`] eliminates the reopen-by-name step entirely: all
/// further writes go through the same fd, which is pinned to the inode
/// `O_EXCL` created.
///
/// The trailing `.safetensors` is required for the shard tempfiles:
/// `mlx_save_safetensors` appends `.safetensors` to a path that lacks it
/// (`mlx/io/safetensors.cpp`), so a temp name ending in `.safetensors`
/// makes mlx write *exactly* this path even if the path-based writer is
/// used. The index tempfile reuses the same suffix harmlessly (it is
/// written via `write_json_pretty(&mut File, ...)` then renamed onto the
/// `.index.json` name). Same-directory keeps the later [`std::fs::rename`]
/// single-fs (atomic on POSIX/Windows; a cross-fs rename silently
/// degrades to copy+unlink, losing atomicity). Mirrors `cache_prompt`'s
/// `open_excl_temp_safetensors` / `audio::io::save_wav`'s
/// `open_excl_tempfile` discipline.
fn open_excl_temp_shard(final_path: &Path) -> Result<(std::fs::File, PathBuf)> {
  use std::{
    fs::OpenOptions,
    io::ErrorKind,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
  };
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  const MAX_RETRIES: u32 = 16;

  let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
  let file_name = final_path
    .file_name()
    .ok_or_else(|| Error::Backend {
      message: format!(
        "save: destination {} has no file_name component",
        final_path.display()
      ),
    })?
    .to_string_lossy()
    .into_owned();
  let pid = std::process::id();
  let mut last_err: Option<std::io::Error> = None;
  for _ in 0..MAX_RETRIES {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let rand = nanos ^ counter.rotate_left(17);
    let candidate = parent.join(format!("{file_name}.{pid}.{rand:016x}.tmp.safetensors"));
    match OpenOptions::new()
      .write(true)
      .create_new(true)
      .open(&candidate)
    {
      Ok(file) => {
        // Hand the open `File` back to the caller so every subsequent
        // write goes through THIS fd (closing the post-create / pre-
        // reopen TOCTOU window). The path is returned alongside for the
        // later atomic `fs::rename`.
        return Ok((file, candidate));
      }
      Err(e) if e.kind() == ErrorKind::AlreadyExists => {
        last_err = Some(e);
        continue;
      }
      Err(e) => {
        return Err(Error::Backend {
          message: format!(
            "save: create_new tempfile {} failed: {e}",
            candidate.display()
          ),
        });
      }
    }
  }
  Err(Error::Backend {
    message: format!(
      "save: exhausted {MAX_RETRIES} tempfile retries (last error: {})",
      last_err
        .map(|e| e.to_string())
        .unwrap_or_else(|| "<none>".into())
    ),
  })
}

/// fsync `path` so its bytes are durable on disk before it is renamed into
/// place — a delayed-allocation / NFS / quota writeback failure must surface
/// *here*, not after a not-yet-on-disk file has been renamed over a
/// previously-valid checkpoint. mlx-c does not fsync; reopen the path
/// read-only and `sync_all` it. Mirrors `cache_prompt::save_prompt_cache_atomic`.
///
/// `pub(crate)` so sibling modules ([`crate::lm::convert`]'s post-copy
/// durability step, F7 R4) can call the same well-tested helper rather
/// than re-implement the open + `sync_all` boilerplate. Test-only fault
/// injection is wired through [`arm_fsync_path_fault`].
///
/// Returns a crate-wide [`Error::Backend`] on IO failure. Callers that
/// need to preserve the underlying [`std::io::ErrorKind`] (ENOSPC / EIO /
/// PermissionDenied / ...) end-to-end through a structured aggregate
/// should instead call [`fsync_path_io`] (the sibling `io::Result<()>`
/// variant that does NOT collapse the kind into a string-wrapped
/// `Error::Backend`).
///
/// **TOCTOU note.** Reopening `path` by name to fsync it widens the same
/// TOCTOU window the path-based safetensors writer does — an attacker
/// with directory write access can `unlink(path) + symlink(path,
/// /etc/passwd)` between the original `O_EXCL` create + this reopen.
/// In-tree save callers therefore use the fd-bound
/// [`fsync_open_file_for_path`] which `sync_all`s the original-open fd
/// directly (still routing through the same `FSYNC_PATH_FAULT_*`
/// injector). This path-based variant is retained for sibling modules
/// (`convert.rs`'s post-copy durability step has no original-open fd to
/// reuse since the files were `std::fs::copy`d not `File::open`d) and
/// for back-compat.
#[allow(dead_code)]
pub(crate) fn fsync_path(path: &Path) -> Result<()> {
  // Forward to the kind-preserving inner so the production path + the
  // (test-only) injector both produce a single canonical io::Error which
  // we then wrap in [`Error::Backend`] (string-collapsing the kind — the
  // historic API shape for callers that want the crate-wide Error type).
  fsync_path_inner(path).map_err(|e| Error::Backend {
    message: format!("save: fsync tempfile {} failed: {e}", path.display()),
  })
}

/// Sibling of [`fsync_path`] that returns the raw [`std::io::Result`] so
/// callers can preserve the underlying [`std::io::ErrorKind`]
/// (ENOSPC / EIO / PermissionDenied / ...) instead of collapsing it into
/// the string-wrapped [`Error::Backend`] that [`fsync_path`] returns.
///
/// Used by [`crate::lm::convert::copy_tokenizer_and_extras`] so its
/// post-copy per-file fsync warnings carry a machine-readable
/// [`std::io::ErrorKind`] through to the typed
/// [`crate::error::ConvertDurabilityWarnings`] aggregate (without this
/// the post_copy_file warning's kind would be uniformly
/// [`std::io::ErrorKind::Other`] — callers couldn't distinguish a
/// writeback-quota failure from a permission failure without parsing the
/// message text). The save-side callers that want the crate-wide Error
/// shape stay on the original [`fsync_path`] (the two variants share the
/// same underlying IO + the same test-only injector — see
/// [`fsync_path_inner`]).
pub(crate) fn fsync_path_io(path: &Path) -> std::io::Result<()> {
  fsync_path_inner(path)
}

/// `fsync_path`-equivalent that operates on an **already-open** [`File`]
/// (typed as `&std::fs::File` so callers keep their borrow). Issues
/// `sync_all` on the supplied fd instead of reopening `path` by name, so
/// the durability gate is pinned to the original-open identity — closes
/// the same post-`O_EXCL` / pre-reopen TOCTOU window that
/// [`open_excl_temp_shard`] returning the `File` closes for the write
/// itself.
///
/// `path` is still threaded through so the test-only fault injector keyed
/// on `fsync_path` (`FSYNC_PATH_FAULT_*`) fires identically — exposes the
/// same fault surface to `convert.rs`'s post-copy / save-side tests as
/// the path-based [`fsync_path`] / [`fsync_path_io`] do (the injector
/// formats the path into its synthetic [`std::io::Error`] message; in the
/// remove-then-fail variant the path is removed before falling through
/// to the natural `sync_all` call, which will then return the OS's real
/// error on the now-stale fd).
pub(crate) fn fsync_open_file_for_path(file: &std::fs::File, path: &Path) -> Result<()> {
  fsync_open_file_for_path_inner(file, path).map_err(|e| Error::Backend {
    message: format!("save: fsync tempfile {} failed: {e}", path.display()),
  })
}

fn fsync_open_file_for_path_inner(
  file: &std::fs::File,
  #[cfg_attr(not(test), allow(unused_variables))] path: &Path,
) -> std::io::Result<()> {
  #[cfg(test)]
  if let Some(remaining) = FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.get()) {
    if remaining == 0 {
      let kind = FSYNC_PATH_FAULT_KIND.with(|c| c.get().unwrap_or(std::io::ErrorKind::Other));
      FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(None));
      FSYNC_PATH_FAULT_KIND.with(|c| c.set(None));
      return Err(std::io::Error::new(
        kind,
        format!("injected fsync_path failure for {}", path.display()),
      ));
    } else {
      FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(Some(remaining - 1)));
    }
  }
  #[cfg(test)]
  if let Some(remaining) = FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.get()) {
    if remaining == 0 {
      FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(None));
      let _ = std::fs::remove_file(path);
      // Fall through — `file.sync_all()` on the now-unlinked fd will
      // still succeed on POSIX (the inode stays live while the fd is
      // open), so the F7 R7 "real failure" path that depends on the
      // file disappearing is exercised through the path-based variant
      // ([`fsync_path_io`]) which DOES reopen by name. This helper's
      // remove-then-fail behavior matches POSIX semantics; the test
      // matrix is unchanged because all `arm_fsync_path_fault_remove_*`
      // call sites go through `fsync_path_io` (`copy_tokenizer_and_extras`
      // in `convert.rs`), not this fd-based variant.
    } else {
      FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(Some(remaining - 1)));
    }
  }
  file.sync_all()
}

/// Inner fsync helper shared by [`fsync_path`] (which wraps the io::Error
/// in [`Error::Backend`] for the crate-wide Error type) and
/// [`fsync_path_io`] (which returns the raw [`std::io::Result`] for
/// callers that need to preserve [`std::io::ErrorKind`] end-to-end).
///
/// Routing the test-only fault injector through this single inner — not
/// either variant's wrapper — means every caller observes the same
/// injected failure regardless of which surface they used. The injector
/// produces a real [`std::io::Error`] (with an injectable kind via
/// [`arm_fsync_path_fault_with_kind`]) so the kind-preservation property
/// the `_io` variant exposes is testable end-to-end.
fn fsync_path_inner(path: &Path) -> std::io::Result<()> {
  // Test-only fault-injection knob — see [`arm_fsync_path_fault`] /
  // [`arm_fsync_path_fault_with_kind`]. Mirrors the `fsync_dir` injector
  // shape so the F7 R4 post-copy durability tests can exercise the
  // "file content is durable but fsync warned" branch without needing a
  // hostile filesystem. Production code path: always `None`, falls
  // straight through to the real fsync.
  #[cfg(test)]
  if let Some(remaining) = FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.get()) {
    if remaining == 0 {
      let kind = FSYNC_PATH_FAULT_KIND.with(|c| c.get().unwrap_or(std::io::ErrorKind::Other));
      FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(None));
      FSYNC_PATH_FAULT_KIND.with(|c| c.set(None));
      return Err(std::io::Error::new(
        kind,
        format!("injected fsync_path failure for {}", path.display()),
      ));
    } else {
      FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(Some(remaining - 1)));
    }
  }

  // F7 R7 Finding test hook — "real OS failure" injector. Unlike the
  // pre-existing injector above (which synthesizes a formatted
  // io::Error string that incidentally includes the path), this one
  // removes the target file then falls through to the natural
  // [`std::fs::File::open`] call so the test observes the AUTHENTIC
  // [`std::io::Error`] the OS produces — a context-free OS-level
  // message like `"No such file or directory (os error 2)"` with NO
  // path embedded. Used by `convert.rs`'s F7 R7 "real failure" test to
  // prove the call-site wrap in `copy_tokenizer_and_extras` adds path
  // context INDEPENDENT of any injector-format coincidence.
  //
  // Production code path: always `None`, falls straight through to the
  // real fsync (the entire `#[cfg(test)]` block is compiled out).
  #[cfg(test)]
  if let Some(remaining) = FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.get()) {
    if remaining == 0 {
      FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(None));
      // Best-effort remove; if it doesn't exist we still want the
      // natural File::open below to produce the real NotFound error.
      let _ = std::fs::remove_file(path);
      // Fall through to the natural code path — File::open(path) will
      // now return `io::ErrorKind::NotFound` with the OS-level message
      // (no path in `.to_string()`).
    } else {
      FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(Some(remaining - 1)));
    }
  }

  let f = std::fs::File::open(path)?;
  f.sync_all()
}

// Test-only fault injector for `fsync_path` / `fsync_path_io`: when set
// on the current thread, the next `n` `fsync_path*` calls succeed
// normally and the (n+1)-th returns
// `std::io::Error::new(kind, "injected fsync_path ...")` (kind defaults
// to `ErrorKind::Other`; can be overridden via
// [`arm_fsync_path_fault_with_kind`]). Used by F7 R4's post-copy
// durability tests to drive the
// `CopyOutcome::CommittedWithDurabilityWarning` branch without needing a
// hostile filesystem. Always `None` in production code (the thread-local
// is only set inside `#[test]` fns and unset on test exit).
#[cfg(test)]
thread_local! {
  static FSYNC_PATH_FAULT_SKIP_THEN_FAIL: std::cell::Cell<Option<usize>> =
    const { std::cell::Cell::new(None) };
  /// Optional override for the injected failure's
  /// [`std::io::ErrorKind`]. `None` means the default
  /// [`std::io::ErrorKind::Other`] (the historic injector shape).
  /// Cleared in lockstep with `FSYNC_PATH_FAULT_SKIP_THEN_FAIL` when the
  /// injector fires.
  static FSYNC_PATH_FAULT_KIND: std::cell::Cell<Option<std::io::ErrorKind>> =
    const { std::cell::Cell::new(None) };
  /// F7 R7 — "real failure" injector skip counter. When `Some(n)`, the
  /// next `n` `fsync_path_inner` calls pass and the (n+1)-th
  /// `remove_file`s the target path then falls through to the natural
  /// [`std::fs::File::open`] which produces a real OS-level
  /// [`std::io::ErrorKind::NotFound`] error (no path in the message).
  /// Used to verify the call-site wrap in
  /// `crate::lm::convert::copy_tokenizer_and_extras` adds path context
  /// independent of any injector-format coincidence.
  static FSYNC_PATH_FAULT_REMOVE_THEN_FAIL: std::cell::Cell<Option<usize>> =
    const { std::cell::Cell::new(None) };
}

/// Arm the [`fsync_path`] / [`fsync_path_io`] fault injector to skip
/// `skip` successful calls then make the next call fail with
/// [`std::io::ErrorKind::Other`]. Returns a [`Drop`] guard that disarms
/// the injector when it goes out of scope (so a test panic still leaves
/// the thread in a clean state for the next test).
///
/// `pub(crate)` (test-only) so sibling modules' tests (e.g.
/// [`crate::lm::convert`]'s F7 R4 post-copy durability closure) can
/// drive the [`CopyOutcome::CommittedWithDurabilityWarning`] branch
/// through the public [`crate::lm::convert::convert`] entrypoint.
#[cfg(test)]
pub(crate) fn arm_fsync_path_fault(skip: usize) -> FsyncPathFaultGuard {
  arm_fsync_path_fault_with_kind(skip, std::io::ErrorKind::Other)
}

/// Variant of [`arm_fsync_path_fault`] that lets the caller pick the
/// injected [`std::io::ErrorKind`] (e.g.
/// [`std::io::ErrorKind::PermissionDenied`] /
/// [`std::io::ErrorKind::StorageFull`]). Used by F7 R6's
/// kind-preservation tests so the post-copy file fsync warning's
/// `.kind()` can be asserted against a SPECIFIC non-`Other` kind —
/// proving the convert()-side aggregate preserves the kind end-to-end
/// (the F7 R6 finding was that the R5 fix's `fsync_copied` closure
/// re-wrapped the injected io::Error via `io::Error::other(message)`,
/// collapsing every kind to `Other`).
#[cfg(test)]
pub(crate) fn arm_fsync_path_fault_with_kind(
  skip: usize,
  kind: std::io::ErrorKind,
) -> FsyncPathFaultGuard {
  FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(Some(skip)));
  FSYNC_PATH_FAULT_KIND.with(|c| c.set(Some(kind)));
  FsyncPathFaultGuard
}

#[cfg(test)]
pub(crate) struct FsyncPathFaultGuard;

#[cfg(test)]
impl Drop for FsyncPathFaultGuard {
  fn drop(&mut self) {
    FSYNC_PATH_FAULT_SKIP_THEN_FAIL.with(|c| c.set(None));
    FSYNC_PATH_FAULT_KIND.with(|c| c.set(None));
  }
}

/// F7 R7 — arm the "real OS failure" injector: skip `skip` successful
/// `fsync_path_inner` calls, then on the (skip+1)-th call remove the
/// target file before the natural [`std::fs::File::open`] runs. The
/// resulting [`std::io::Error`] is the AUTHENTIC OS-level error (kind
/// [`std::io::ErrorKind::NotFound`], message like `"No such file or
/// directory (os error 2)"`) with NO path embedded — exactly the kind
/// of context-free failure the F7 R7 call-site wrap is designed to
/// catch. Returns a [`Drop`] guard that disarms the injector on scope
/// exit (so a test panic still leaves the thread clean).
///
/// `pub(crate)` (test-only) so [`crate::lm::convert`]'s F7 R7
/// "real failure" test can drive a non-synthesized post-copy fsync
/// warning through the public [`crate::lm::convert::convert`]
/// entrypoint and assert the call-site wrap added path + operation
/// context to a path-free OS-level error.
#[cfg(test)]
pub(crate) fn arm_fsync_path_fault_remove_then_fail(skip: usize) -> FsyncPathRemoveFaultGuard {
  FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(Some(skip)));
  FsyncPathRemoveFaultGuard
}

#[cfg(test)]
pub(crate) struct FsyncPathRemoveFaultGuard;

#[cfg(test)]
impl Drop for FsyncPathRemoveFaultGuard {
  fn drop(&mut self) {
    FSYNC_PATH_FAULT_REMOVE_THEN_FAIL.with(|c| c.set(None));
  }
}

// Test-only fault injector for `fsync_dir`: when set on the current
// thread, the next `n` `fsync_dir` calls succeed normally and the
// (n+1)-th returns `std::io::Error::new(kind, ...)` (kind defaults to
// `ErrorKind::Other`; can be overridden via
// [`arm_fsync_dir_fault_with_kind`]). Used by the post-index-commit
// durability-warning tests to drive the
// `CommitOutcome::CommittedWithDurabilityWarning` branch without needing
// a hostile filesystem. Always `None` in production code (the thread-
// local is only set inside `#[test]` fns and unset on test exit).
#[cfg(test)]
thread_local! {
  static FSYNC_DIR_FAULT_SKIP_THEN_FAIL: std::cell::Cell<Option<usize>> =
    const { std::cell::Cell::new(None) };
  /// Optional override for the injected failure's
  /// [`std::io::ErrorKind`]. `None` means the default
  /// [`std::io::ErrorKind::Other`] (the historic injector shape).
  /// Cleared in lockstep with `FSYNC_DIR_FAULT_SKIP_THEN_FAIL` when the
  /// injector fires.
  static FSYNC_DIR_FAULT_KIND: std::cell::Cell<Option<std::io::ErrorKind>> =
    const { std::cell::Cell::new(None) };
}

/// Arm the [`fsync_dir`] fault injector to skip `skip` successful calls
/// then make the next call fail with [`std::io::ErrorKind::Other`].
/// Returns a [`Drop`] guard that disarms the injector when it goes out
/// of scope (so a test panic still leaves the thread in a clean state
/// for the next test).
///
/// `pub(crate)` (test-only) so sibling modules' tests (e.g.
/// [`crate::lm::convert`]'s F7 R1 Finding-4 closure) can drive the same
/// post-commit durability path through the public [`save`] entrypoint.
#[cfg(test)]
pub(crate) fn arm_fsync_dir_fault(skip: usize) -> FsyncDirFaultGuard {
  arm_fsync_dir_fault_with_kind(skip, std::io::ErrorKind::Other)
}

/// Variant of [`arm_fsync_dir_fault`] that lets the caller pick the
/// injected [`std::io::ErrorKind`]. Used by F7 R6's kind-preservation
/// tests so the save-side warning + post-copy dir warning both carry a
/// specific machine-readable kind end-to-end through the
/// [`crate::error::ConvertDurabilityWarnings`] aggregate.
#[cfg(test)]
pub(crate) fn arm_fsync_dir_fault_with_kind(
  skip: usize,
  kind: std::io::ErrorKind,
) -> FsyncDirFaultGuard {
  FSYNC_DIR_FAULT_SKIP_THEN_FAIL.with(|c| c.set(Some(skip)));
  FSYNC_DIR_FAULT_KIND.with(|c| c.set(Some(kind)));
  FsyncDirFaultGuard
}

#[cfg(test)]
pub(crate) struct FsyncDirFaultGuard;

#[cfg(test)]
impl Drop for FsyncDirFaultGuard {
  fn drop(&mut self) {
    FSYNC_DIR_FAULT_SKIP_THEN_FAIL.with(|c| c.set(None));
    FSYNC_DIR_FAULT_KIND.with(|c| c.set(None));
  }
}

/// fsync the directory `path` so any rename / unlink / create entries inside
/// it are durable on disk before we return — without this, a `rename(2)` is
/// allowed to land in the parent directory's in-memory inode but be lost on
/// a crash, leaving the index referencing shards the kernel never wrote the
/// directory entry for. Called once after every batch of renames in
/// [`save_model`] (after the shard renames, after the index rename) and
/// after the config rename in [`commit_staged_config`].
///
/// Unix implementation opens the directory read-only with `O_DIRECTORY`
/// (so a non-directory path errors) and calls `sync_all`, mirroring the
/// well-trodden POSIX `fsync(dirfd)` durability pattern that mlx-c does
/// not perform. On non-Unix platforms (Windows, WASI), this is a no-op:
/// Windows has no public API to fsync a directory handle (NTFS metadata
/// is journaled by the filesystem, not user-flushable), and the call
/// silently succeeds rather than propagate a platform-specific error
/// that has no corresponding fix at this layer.
/// `pub(crate)` so sibling modules ([`crate::lm::convert`]'s post-copy
/// directory-fsync step, F7 R4) can call the same well-tested helper
/// rather than re-implement the `O_DIRECTORY` + `sync_all` boilerplate.
/// Test-only fault injection is wired through [`arm_fsync_dir_fault`].
pub(crate) fn fsync_dir(path: &Path) -> std::io::Result<()> {
  // Test-only fault-injection knob — see [`arm_fsync_dir_fault`] /
  // [`arm_fsync_dir_fault_with_kind`]. Threaded through here rather than
  // at each call site so every fsync_dir call (shard-fsync, index-fsync,
  // config-fsync) is uniformly faultable.
  #[cfg(test)]
  if let Some(remaining) = FSYNC_DIR_FAULT_SKIP_THEN_FAIL.with(|c| c.get()) {
    if remaining == 0 {
      let kind = FSYNC_DIR_FAULT_KIND.with(|c| c.get().unwrap_or(std::io::ErrorKind::Other));
      FSYNC_DIR_FAULT_SKIP_THEN_FAIL.with(|c| c.set(None));
      FSYNC_DIR_FAULT_KIND.with(|c| c.set(None));
      return Err(std::io::Error::new(
        kind,
        format!("injected fsync_dir failure for {}", path.display()),
      ));
    } else {
      FSYNC_DIR_FAULT_SKIP_THEN_FAIL.with(|c| c.set(Some(remaining - 1)));
    }
  }

  #[cfg(unix)]
  {
    use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};
    // `O_DIRECTORY` makes the open fail with `ENOTDIR` if `path` is not a
    // directory — a strong precondition check before we waste a syscall.
    // `read(true)` is the portable POSIX way to ask for a read-only fd;
    // a directory open for read is allowed, an open for write is not.
    let f = OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_DIRECTORY)
      .open(path)?;
    f.sync_all()
  }
  #[cfg(not(unix))]
  {
    // No portable user-space "fsync a directory" on Windows / WASI. The
    // FS journals directory metadata transparently; returning Ok keeps
    // the call site platform-agnostic.
    let _ = path;
    Ok(())
  }
}

/// Write back a model configuration as `config.json`, mirroring
/// `mlx_lm.utils.save_config` (`utils.py:899-922`).
///
/// `config` is the verbatim `config.json` JSON body (the raw [`String`]
/// [`load_config`] returns alongside the typed [`Config`]). The reference's
/// clean-up is applied to the parsed object:
///
/// - the `_name_or_path` and `vision_config` keys are dropped
///   (`config.pop("_name_or_path"/"vision_config", None)`, `utils.py:912-913`);
/// - if a `quantization` key is present it is **also** copied to
///   `quantization_config` (`utils.py:914-915` — HF model-tree interop);
/// - the keys are **sorted** for readability (`dict(sorted(...))`,
///   `utils.py:918`);
///
/// then the result is written to `config_path` with 4-space indentation
/// (`json.dump(config, fid, indent=4)`, `utils.py:921-922`).
///
/// The write is **failure-atomic** — the same temp + fsync + rename
/// discipline `save_model` uses for its shards: the JSON is written to a
/// same-directory `O_EXCL` tempfile and fsynced, then atomically renamed over
/// `config_path`. A previously-valid `config.json` is therefore left fully
/// intact if the write fails partway, and the tempfile is cleaned up on every
/// error path.
///
/// `config` must be a JSON **object**; anything else (or invalid JSON) is an
/// [`Error::Backend`]. A write failure is an [`Error::Backend`] naming the
/// path. A post-rename `fsync_dir` failure is surfaced as
/// [`Error::DurabilityWarning`] with `committed: true` — the NEW
/// `config.json` IS observable on disk, but the directory-entry write may
/// not yet be durable (matching the [`save`] driver's contract).
///
/// # Hostile-directory caveat
///
/// The body is written fd-bound (every byte goes to the inode this
/// function opened) but published via `std::fs::rename(temp_path,
/// config_path)`, which operates BY PATHNAME. If `config_path`'s
/// parent directory is user-writable, an attacker with write access
/// can `unlink(temp_path)` and substitute their own file at the same
/// name between the fsync and the rename, causing the published
/// `config.json` to be the attacker's inode rather than the fd-written
/// one. Mirror of [`save_model`]'s "Hostile-directory caveat": use a
/// trusted (not user-writable) parent directory for full safety. See
/// the "Scope of this guarantee" section of
/// [`crate::io::save_safetensors_to_file`] for the broader discussion.
pub fn save_config(config: &str, config_path: &Path) -> Result<()> {
  let staged = stage_config(config, config_path)?;
  match commit_staged_config(staged, config_path)? {
    CommitOutcome::Committed => Ok(()),
    CommitOutcome::CommittedWithDurabilityWarning(source) => Err(Error::DurabilityWarning {
      committed: true,
      source,
    }),
  }
}

/// A `config.json` body that has been parsed, validated, cleaned, sorted,
/// written to a same-directory `O_EXCL` tempfile, and fsynced — but not yet
/// renamed into place. [`save`] stages the config FIRST (so an invalid
/// config aborts before [`save_model`] touches any weight), then atomically
/// renames it via [`commit_staged_config`] AFTER the index commits.
///
/// The tempfile is removed in [`Drop`] if the staging step is dropped
/// without a successful [`commit_staged_config`] (a `save_model` failure
/// in between). Holds only its tempfile path — no in-memory JSON.
struct StagedConfig {
  tmp_path: PathBuf,
  /// `false` once [`commit_staged_config`] has taken ownership of the
  /// rename; the [`Drop`] then becomes a no-op so a successful commit
  /// does not race with another concurrent process for the same temp
  /// basename.
  cleanup_on_drop: bool,
}

impl Drop for StagedConfig {
  fn drop(&mut self) {
    if self.cleanup_on_drop {
      let _ = std::fs::remove_file(&self.tmp_path);
    }
  }
}

/// Parse, validate, clean, sort, write + fsync a `config.json` body to a
/// same-directory `O_EXCL` tempfile next to `config_path`, returning the
/// staged tempfile. The body is NOT renamed yet — the caller is responsible
/// for atomically renaming it via [`commit_staged_config`] (or dropping the
/// returned guard to clean the tempfile up). Used by [`save`] so an invalid
/// config or a tempfile-create failure aborts BEFORE [`save_model`] touches
/// any weight (the previously-valid checkpoint stays fully intact).
///
/// Identical parse / clean / sort behavior as [`save_config`]; see that
/// function's doc-comment for the schema. Invalid JSON or a non-object
/// body is an [`Error::Backend`] surfaced here (before any IO failure can
/// destroy a still-valid checkpoint).
fn stage_config(config: &str, config_path: &Path) -> Result<StagedConfig> {
  let value: serde_json::Value = serde_json::from_str(config).map_err(|e| Error::Backend {
    message: format!("save_config: config is not valid JSON: {e}"),
  })?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::Backend {
      message: "save_config: config JSON must be an object".into(),
    });
  };

  // Clean unused keys (`utils.py:912-913`).
  map.remove("_name_or_path");
  map.remove("vision_config");
  // Mirror `quantization` into `quantization_config` (`utils.py:914-915`).
  if let Some(q) = map.get("quantization").cloned() {
    map.insert("quantization_config".to_string(), q);
  }

  // Sort keys for readability (`dict(sorted(config.items()))`,
  // `utils.py:918`). A plain `serde_json::Map` keeps insertion order (the
  // `preserve_order` feature is on workspace-wide for `minijinja`), so a
  // `BTreeMap` round-trip is the explicit sort.
  let sorted: BTreeMap<String, serde_json::Value> = map.into_iter().collect();
  let sorted_value = serde_json::to_value(sorted).map_err(|e| Error::Backend {
    message: format!("save_config: cannot re-serialize sorted config: {e}"),
  })?;

  let (mut tmp_file, tmp_path) = open_excl_temp_shard(config_path)?;
  let staged = StagedConfig {
    tmp_path,
    cleanup_on_drop: true,
  };
  write_json_pretty(&mut tmp_file, &sorted_value, "save_config: config.json")?;
  fsync_open_file_for_path(&tmp_file, &staged.tmp_path)?;
  drop(tmp_file);
  Ok(staged)
}

/// Atomically rename a [`StagedConfig`]'s tempfile over `config_path`, then
/// `fsync` the parent directory so the rename is durable. Consumes the
/// staging guard so a successful rename does not also delete the just-
/// published file via [`Drop`]. On a rename failure the staging guard's
/// `Drop` cleans the tempfile up.
///
/// Returns `Result<CommitOutcome, Error>` with the same shape as
/// [`save_model`]:
///
/// - `Err(_)` for a pre-commit failure (the rename itself failed; the
///   OLD `config.json`, if any, is untouched).
/// - `Ok(Committed)` — rename + post-commit `fsync_dir` both succeeded.
/// - `Ok(CommittedWithDurabilityWarning(e))` — rename succeeded (the NEW
///   `config.json` IS observable on disk), but the post-rename
///   `fsync_dir` returned `e`. The caller (e.g. [`save`]) MUST treat this
///   as a logically-committed config-write with a durability warning,
///   matching `save_model`'s contract.
fn commit_staged_config(mut staged: StagedConfig, config_path: &Path) -> Result<CommitOutcome> {
  if let Err(e) = std::fs::rename(&staged.tmp_path, config_path) {
    // Leave `cleanup_on_drop = true` so `Drop` removes the tempfile.
    return Err(Error::Backend {
      message: format!(
        "save_config: cannot rename {} -> {}: {e}",
        staged.tmp_path.display(),
        config_path.display()
      ),
    });
  }
  // The rename consumed the tempfile (it IS now `config_path`); the
  // tempfile path no longer exists, so don't try to `unlink` it on Drop.
  staged.cleanup_on_drop = false;
  // fsync the parent directory so the rename entry is durable on disk.
  // **An error here does NOT roll the commit back** (the rename is
  // durable in the directory inode; only the on-disk directory metadata
  // may not yet be drained). The NEW `config.json` IS observable on
  // disk. Returned as `Ok(CommittedWithDurabilityWarning(_))` rather
  // than `Err(_)` so the caller can surface it via
  // [`crate::Error::DurabilityWarning`] without rolling the commit back.
  if let Some(parent) = config_path.parent()
    && let Err(e) = fsync_dir(parent)
  {
    return Ok(CommitOutcome::CommittedWithDurabilityWarning(e));
  }
  Ok(CommitOutcome::Committed)
}

/// Save a model — weights and config — into `dst_path`, mirroring the
/// local-directory core of `mlx_lm.utils.save` (`utils.py:925-950`).
///
/// The ordering differs from the reference's `save_model` → `save_config`
/// sequence in service of the **same** atomicity guarantee [`save_model`]
/// provides for the weights, extended to the config:
///
/// 1. `dst_path.mkdir(parents=True, exist_ok=True)` is performed up front
///    so the same-directory tempfiles in steps 2 and 3 have a directory
///    to land in even on a brand-new destination.
/// 2. The config body is **parsed + validated + staged** to a same-
///    directory `O_EXCL` tempfile and fsynced (the staging step of
///    [`save_config`]). An invalid JSON / non-object config errors HERE,
///    **before** any weight file is touched — so a malformed config never
///    publishes a partial checkpoint over a previously-valid one.
/// 3. [`save_model`] writes the sharded `.safetensors` + the
///    `model.safetensors.index.json` index, with the index rename as the
///    single observable commit point (`utils.py:942` plus the structural
///    atomicity fix; see [`save_model`]). A
///    [`CommitOutcome::CommittedWithDurabilityWarning`] is NOT a failure —
///    the visible checkpoint is on disk and complete; `save` PROCEEDS to
///    commit the staged config (otherwise the staging guard's `Drop` would
///    delete the staged config tempfile, leaving NEW weights+index against
///    the OLD config — the F6 R6 Finding-2 hole). The warning is
///    accumulated for the final return.
/// 4. The staged config is **atomically renamed** over
///    `<dst_path>/config.json` LAST (`utils.py:943`). The config rename's
///    own post-rename `fsync_dir` failure is similarly demoted to a
///    durability warning.
/// 5. If any [`CommitOutcome::CommittedWithDurabilityWarning`] was
///    observed (from `save_model` or from `commit_staged_config`) `save`
///    returns [`Error::DurabilityWarning`] with `committed: true` — the
///    new checkpoint IS visible + complete (weights + index + config) on
///    disk, but a parent-dir `fsync` did not return success and a power
///    loss before the FS internally drains could lose the directory entry.
///    Otherwise `save` returns `Ok(())`.
///
/// Failure semantics:
///
/// - A config validation / staging failure in step 2 aborts BEFORE any
///   weight or config file is touched — the entire previous checkpoint
///   (weights, index, config) is byte-identical to the pre-save state.
/// - A pre-index-commit `save_model` failure in step 3 leaves the
///   previous weights+index intact (the index rename single-commit-point —
///   see [`save_model`]); the staged config tempfile is removed by the
///   staging-guard `Drop`.
/// - A post-index-commit durability-warning in step 3 is NOT a failure;
///   `save` PROCEEDS to step 4. (Step 5 surfaces the warning at the end.)
/// - A config-rename failure in step 4 is the one residual mismatch
///   window: the new weights+index are committed but the OLD `config.json`
///   survives, so a subsequent load sees the NEW weights against the OLD
///   config. This is rare (it would take a transient ENOENT/EBUSY on the
///   final rename of a fsynced, exclusively-created tempfile) and is
///   accepted as the ceiling without a true cross-file transaction.
///
/// **Deliberately not ported** from `utils.save`:
///
/// - `tokenizer.save_pretrained(dst_path)` (`utils.py:944`) — the mlxrs
///   [`Tokenizer`](crate::tokenizer::Tokenizer) is **load-only** (no
///   `save_pretrained` equivalent is ported; re-serializing the full
///   tokenizer file set is tokenizer-architecture surface, outside this
///   model-save/shard/introspect scope). A caller that needs the tokenizer
///   alongside the model copies its source files itself.
/// - the `glob("*.py" / "generation_config.json")` source-file copy
///   (`utils.py:946-948`) — needs a separate `src_path`, model-arch-adjacent.
/// - `create_model_card` (`utils.py:950`) — HuggingFace Hub, excluded as
///   network/hub surface (consistent with the `_download` /
///   `create_model_card` / `upload_to_hub` omission noted in the
///   module-level docs, and the project's local-path-only scope).
///
/// `quant` supplies the per-layer [`Quantization`] [`save_model`] needs
/// (pass [`PerLayerQuantization::default`](crate::lm::quant::PerLayerQuantization)
/// for an unquantized model). Any step's recoverable failure is the
/// [`Error`] that step produced.
///
/// # Hostile-directory caveat
///
/// `save` inherits the publication-step trust requirement from both
/// [`save_model`] (shards + index `hard_link` / `rename`) and
/// [`save_config`] (config `rename`): every publish operates BY
/// PATHNAME in `dst_path`. If `dst_path` is user-writable, an attacker
/// can substitute the temp entries between fsync and publish, causing
/// the final shards / index / config to point at the attacker's
/// inodes rather than the fd-written ones. **For full hostile-
/// directory safety, pass a trusted (not user-writable) `dst_path`.**
/// See the "Hostile-directory caveat" sections on [`save_model`] /
/// [`save_config`] and the "Scope of this guarantee" section of
/// [`crate::io::save_safetensors_to_file`] for the broader discussion.
pub fn save(
  dst_path: &Path,
  weights: &Weights,
  config: &str,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<()> {
  // 1. Create the destination directory up front so step 2's same-dir
  //    tempfile open can land in it even on a brand-new destination.
  //    `save_model` would otherwise be the first step to create it.
  std::fs::create_dir_all(dst_path).map_err(|e| Error::Backend {
    message: format!(
      "save: cannot create destination directory {}: {e}",
      dst_path.display()
    ),
  })?;

  // 2. Validate + stage the config FIRST so an invalid JSON aborts BEFORE
  //    `save_model` touches any weight. The staging guard's `Drop` removes
  //    the tempfile on any failure between here and the final commit, so
  //    a `save_model` failure leaves no leftover config tempfile either.
  let config_path = dst_path.join("config.json");
  let staged_config = stage_config(config, &config_path)?;

  // 3. Save the weights + index. An `Err` is a pre-commit failure (the
  //    previous checkpoint is untouched); the staging guard's `Drop`
  //    cleans the config tempfile. An `Ok(CommittedWithDurabilityWarning)`
  //    is NOT a failure — the index rename succeeded, so the visible
  //    checkpoint loads correctly and we MUST proceed to commit the
  //    config (dropping the staged config here would delete the config
  //    tempfile, leaving NEW weights+index against the OLD config —
  //    the F6 R6 Finding-2 hole this entire shape closes). The warning
  //    is accumulated for the final return.
  let mut durability: Option<std::io::Error> = match save_model(dst_path, weights, quant)? {
    CommitOutcome::Committed => None,
    CommitOutcome::CommittedWithDurabilityWarning(e) => Some(e),
  };

  // 4. Atomically rename the staged config LAST. A rename failure is a
  //    true `Err` (the residual mismatch window — see above). A post-
  //    rename `fsync_dir` failure is demoted to a durability warning,
  //    same shape as `save_model`'s — the config IS visible on disk.
  match commit_staged_config(staged_config, &config_path)? {
    CommitOutcome::Committed => {}
    CommitOutcome::CommittedWithDurabilityWarning(e) => {
      // Prefer the FIRST warning observed (the index commit) so the
      // surfaced error names the canonical commit point; otherwise
      // attach the config commit's.
      if durability.is_none() {
        durability = Some(e);
      }
    }
  }

  // 5. Surface any accumulated durability warning as a non-fatal `Err`
  //    with `committed: true`. The on-disk save is logically complete
  //    (weights + index + config); only the parent-directory fsync
  //    didn't return success. A successful `Ok(())` therefore means
  //    "fully durable".
  if let Some(source) = durability {
    return Err(Error::DurabilityWarning {
      committed: true,
      source,
    });
  }
  Ok(())
}

/// Write `value` to an **already-open** [`std::fs::File`] as
/// 4-space-indented JSON, mirroring Python's `json.dump(value, f, indent=4)`
/// byte-for-byte: a 4-space indent and the `,` / `": "` separators
/// `serde_json::ser::PrettyFormatter` already emits (Python's `indent=N`
/// uses the same — see [`crate::tokenizer::chat`]'s `tojson` note). A
/// trailing newline is **not** added — `json.dump` writes none.
///
/// **TOCTOU rationale.** Earlier this function took a [`Path`] and
/// reopened it via `fs::write` — even when the caller had just created
/// the path via `open_excl_temp_shard`'s `O_EXCL` tempfile, the reopen
/// gave an attacker with directory write access a window to
/// `unlink + symlink` the path between create + write. Writing through
/// the caller's own [`File`] pins the write to the `O_EXCL`-created
/// inode and closes that window.
fn write_json_pretty(
  file: &mut std::fs::File,
  value: &serde_json::Value,
  label: &str,
) -> Result<()> {
  use std::io::Write;
  let mut buf = Vec::new();
  let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
  let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
  serde::Serialize::serialize(value, &mut ser).map_err(|e| Error::Backend {
    message: format!("{label}: cannot serialize JSON: {e}"),
  })?;
  file.write_all(&buf).map_err(|e| Error::Backend {
    message: format!("{label}: cannot write JSON body: {e}"),
  })?;
  Ok(())
}

/// Test/back-compat convenience: open `path` for `O_CREAT|O_TRUNC` write
/// and emit pretty-JSON through [`write_json_pretty`]. Test fixtures that
/// hand-craft a sidecar (e.g. an index file) call this rather than
/// reproducing the `File::create` + `write_json_pretty` pair. Not used in
/// the production save path — that path opens its files via
/// [`open_excl_temp_shard`] and keeps the fd through to publish.
#[cfg(test)]
fn write_json_pretty_to_path(path: &Path, value: &serde_json::Value, label: &str) -> Result<()> {
  let mut f = std::fs::File::create(path).map_err(|e| Error::Backend {
    message: format!("{label}: cannot create {}: {e}", path.display()),
  })?;
  write_json_pretty(&mut f, value, label)
}

#[cfg(test)]
mod save_tests {
  //! F6 — model save / shard / introspect, in isolation. Shard boundaries
  //! are hand-computed for a chosen cap; `save` then `load_weights` round-
  //! trips (weights byte-equal, index.json correct); introspection helpers
  //! are checked against hand-verified counts. No `peak_memory()` assert.

  use super::*;
  use crate::lm::quant::{PerLayerQuantization, QuantMode, Quantization};

  /// A fresh, writable per-test temp directory (the crate's
  /// no-`tempfile`-crate convention — `temp_dir()` + pid + a process-unique
  /// counter, mirroring `lm::factory`'s `fresh_dir`).
  fn fresh_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mlxrs-lm-save-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// An `f32` weight of `n` elements, shape `[n]` — `n * 4` bytes.
  fn f32_weight(n: usize) -> Array {
    Array::from_slice::<f32>(&vec![0.0_f32; n], &(n,)).unwrap()
  }

  // ─────────────────────── array_nbytes ───────────────────────

  #[test]
  fn array_nbytes_is_count_times_dtype_size() {
    // f32 → 4 bytes/elem; 10 elems → 40 bytes.
    assert_eq!(array_nbytes(&f32_weight(10)).unwrap(), 40);
    // u8 → 1 byte/elem.
    let u8s = Array::from_slice::<u8>(&[1u8, 2, 3], &(3usize,)).unwrap();
    assert_eq!(array_nbytes(&u8s).unwrap(), 3);
    // u32 → 4 bytes/elem; a `[2, 8]` packed matrix → 16 elems → 64 bytes.
    let u32s = Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap();
    assert_eq!(array_nbytes(&u32s).unwrap(), 64);
  }

  // ─────────────────────── make_shards ───────────────────────

  /// All-fits case: four 100-byte weights (`a`..`d`, each 25 `f32` elems =
  /// 100 bytes = 400 bytes total) under the default 5-GiB cap land on a
  /// single shard — the cap is never reached, so the loop never flushes.
  #[test]
  fn make_shards_all_fits_one_shard() {
    let mut w: Weights = HashMap::new();
    for name in ["a", "b", "c", "d"] {
      w.insert(name.to_string(), f32_weight(25)); // 100 bytes each
    }
    let one = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
    assert_eq!(one.len(), 1, "4×100 bytes fits in one 5-GiB shard");
    assert_eq!(one[0].len(), 4);
  }

  /// Hand-traced zero-cap split, matching mlx-lm `make_shards`
  /// (`utils.py:598-619`) EXACTLY — including the empty leading shard the
  /// reference's guard-free `shard_size + v.nbytes > cap` produces. With a
  /// `0`-GiB cap, `0 + nbytes > 0` holds for every non-empty weight, so the
  /// split fires every iteration — *including the first*, while `shard` is
  /// still empty. Hand-trace over sorted weights `a, b, c, d` (100 bytes
  /// each), exactly as `utils.py`: at `a`, `0 + 100 > 0` pushes the empty
  /// `{}` and resets, then `shard = {a}`; at `b`, `100 + 100 > 0` pushes
  /// `{a}`, then `shard = {b}`; `c` pushes `{b}`, `shard = {c}`; `d` pushes
  /// `{c}`, `shard = {d}`; after the loop the trailing `{d}` is pushed —
  /// giving `[{}, {a}, {b}, {c}, {d}]`, 5 shards with an empty leading one.
  /// (Run `mlx_lm.utils.make_shards({"a":…,"b":…,"c":…,"d":…}, 0)` to
  /// confirm.)
  #[test]
  fn make_shards_zero_cap_empty_leading_then_one_weight_per_shard() {
    let mut w: Weights = HashMap::new();
    for name in ["a", "b", "c", "d"] {
      w.insert(name.to_string(), f32_weight(25));
    }
    let shards = make_shards(&w, 0).unwrap();
    // 5 shards: an empty leading shard + one per weight (mlx-lm parity).
    assert_eq!(shards.len(), 5);
    assert!(
      shards[0].is_empty(),
      "guard-free split flushes an empty leading shard"
    );
    // Sorted-key order in the trailing single-weight shards.
    assert!(shards[1].contains_key("a"));
    assert!(shards[2].contains_key("b"));
    assert!(shards[3].contains_key("c"));
    assert!(shards[4].contains_key("d"));
    assert!(shards[1..].iter().all(|s| s.len() == 1));
  }

  /// An over-cap **first** sorted tensor. mlx-lm `make_shards` has no
  /// empty-shard guard, so when the first tensor already exceeds the cap
  /// the split fires immediately, flushing the still-empty initial shard.
  /// Hand-trace `make_shards({"big": 400-byte, "small": 4-byte}, cap=0)`
  /// from `utils.py:611-618`: at `big`, `0 + 400 > 0` pushes the empty `{}`
  /// and resets, then `shard = {big}`; at `small`, `400 + 4 > 0` pushes
  /// `{big}` and resets, then `shard = {small}`; after the loop the
  /// trailing `{small}` is pushed — giving `[{}, {big}, {small}]`: an empty
  /// leading shard, then the over-cap tensor on its own shard, then the
  /// remainder. This port must produce the identical sequence (same shard
  /// filenames + index data).
  #[test]
  fn make_shards_over_cap_first_tensor_empty_leading_shard() {
    let mut w: Weights = HashMap::new();
    w.insert("big".to_string(), f32_weight(100)); // 400 bytes — over a 0 cap
    w.insert("small".to_string(), f32_weight(1)); // 4 bytes
    let shards = make_shards(&w, 0).unwrap();
    assert_eq!(
      shards.len(),
      3,
      "empty leading + over-cap tensor + remainder"
    );
    assert!(
      shards[0].is_empty(),
      "over-cap first tensor flushes an empty leading shard"
    );
    // Sorted key order: `big` < `small`.
    assert_eq!(shards[1].len(), 1);
    assert!(shards[1].contains_key("big"));
    assert_eq!(shards[2].len(), 1);
    assert!(shards[2].contains_key("small"));
  }

  /// An empty weight map still yields exactly one (empty) shard — mlx-lm's
  /// `shards.append(shard)` after the loop (`utils.py:618`) always runs.
  #[test]
  fn make_shards_empty_map_yields_one_empty_shard() {
    let w: Weights = HashMap::new();
    let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
    assert_eq!(shards.len(), 1);
    assert!(shards[0].is_empty());
  }

  /// A single weight under an ample cap — one shard holding it. (For the
  /// `0`-cap / over-cap edge case, where a lone weight yields an empty
  /// leading shard `[{}, {solo}]`, see
  /// [`make_shards_over_cap_first_tensor_empty_leading_shard`].)
  #[test]
  fn make_shards_single_weight_one_shard() {
    let mut w: Weights = HashMap::new();
    w.insert("solo".to_string(), f32_weight(7));
    let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].len(), 1);
    assert!(shards[0].contains_key("solo"));
  }

  // ─────────────────────── get_total_parameters ───────────────────────

  /// Dense model: every array contributes its plain element count
  /// (`sum(v.size …)`). Two weights of 25 + 7 elems → 32 parameters.
  #[test]
  fn get_total_parameters_dense_sums_sizes() {
    let mut w: Weights = HashMap::new();
    w.insert("model.embed.weight".to_string(), f32_weight(25));
    w.insert("model.norm.weight".to_string(), f32_weight(7));
    let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
    assert_eq!(total, 32);
  }

  /// Quantized affine layer: a `<path>.weight` (`uint32` packed) with a
  /// `<path>.scales` sibling counts as `weight.size * 32 / bits` logical
  /// params. Both affine-quantization metadata buffers — `<path>.scales`
  /// AND `<path>.biases` (the zero-point array, NOT a real module bias) —
  /// are NOT counted, matching mlx-lm `get_total_parameters`'s quantized
  /// branch (`m.weight.size * 32 // m.bits` plus only a genuine `m.bias`,
  /// `utils.py:203-204`). Hand-trace: packed `.weight` = 16 `u32` elems,
  /// `bits = 4` → `16 * 32 / 4 = 128` logical weights; `.scales` (2 elems)
  /// → +0; `.biases` (2 elems) → +0 (quantization metadata, skipped). Plus
  /// a dense `model.norm.weight` of 7 → +7. Total = 128 + 7 = 135.
  #[test]
  fn get_total_parameters_quantized_unpacks_weight_skips_scales_and_biases() {
    let mut w: Weights = HashMap::new();
    let packed = Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap();
    w.insert("model.layers.0.q_proj.weight".to_string(), packed);
    w.insert(
      "model.layers.0.q_proj.scales".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    w.insert(
      "model.layers.0.q_proj.biases".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    w.insert("model.norm.weight".to_string(), f32_weight(7));

    let quant = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let total = get_total_parameters(&w, &quant).unwrap();
    assert_eq!(total, 128 + 7);
  }

  /// A genuine module bias (`.bias`, singular, with NO `.scales` sibling)
  /// is a real model parameter and IS counted — only an affine
  /// quantization `.biases` (plural, sibling to a `.weight` + `.scales`
  /// triple) is skipped as metadata. Hand-trace: dense `model.fc.weight`
  /// of 5 → +5; `model.fc.bias` of 3 → +3 (no `model.fc.scales`, so it is
  /// a plain parameter). Total = 8.
  #[test]
  fn get_total_parameters_counts_genuine_module_bias() {
    let mut w: Weights = HashMap::new();
    w.insert("model.fc.weight".to_string(), f32_weight(5));
    w.insert("model.fc.bias".to_string(), f32_weight(3));
    let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
    assert_eq!(total, 8);
  }

  /// An orphan `.biases` (plural) with a `.weight` sibling but NO `.scales`
  /// sibling is not a valid affine triple, so it must NOT be skipped — it
  /// falls through to the dense count. (mlx-lm never produces this shape;
  /// the skip is gated on BOTH a `.weight` and a `.scales` sibling so a
  /// stray `.biases` is still accounted for.) Hand-trace: `model.x.weight`
  /// of 4 → +4; `model.x.biases` of 2, no `model.x.scales` → +2. Total = 6.
  #[test]
  fn get_total_parameters_orphan_biases_without_scales_is_counted() {
    let mut w: Weights = HashMap::new();
    w.insert("model.x.weight".to_string(), f32_weight(4));
    w.insert("model.x.biases".to_string(), f32_weight(2));
    let total = get_total_parameters(&w, &PerLayerQuantization::default()).unwrap();
    assert_eq!(total, 6);
  }

  /// A quantized triple (`.scales` present) with no resolvable
  /// [`Quantization`] for its layer is a configuration error.
  #[test]
  fn get_total_parameters_quantized_without_params_errors() {
    let mut w: Weights = HashMap::new();
    w.insert(
      "model.q.weight".to_string(),
      Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
    );
    w.insert(
      "model.q.scales".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    // No global default, no per-layer override → unresolvable.
    let err = get_total_parameters(&w, &PerLayerQuantization::default());
    assert!(matches!(err, Err(Error::Backend { .. })));
  }

  // ─────────────────────── compute_bits_per_weight ───────────────────────

  /// `model_bytes * 8 / model_params`. A single dense `f32` weight of 10
  /// elems: `model_bytes = 40`, `model_params = 10` → `40 * 8 / 10 = 32.0`
  /// bits per weight (exactly f32's 32 bits — a dense float model).
  #[test]
  fn compute_bits_per_weight_dense_f32_is_32() {
    let mut w: Weights = HashMap::new();
    w.insert("model.w.weight".to_string(), f32_weight(10));
    let bpw = compute_bits_per_weight(&w, &PerLayerQuantization::default()).unwrap();
    assert!((bpw - 32.0).abs() < 1e-9, "expected 32.0, got {bpw}");
  }

  /// Quantized: `model_bytes` sums EVERY array (`scales`/`biases` too —
  /// the reference's `tree_reduce` over `model`, `utils.py:211-213`), but
  /// `model_params` is the *unpacked* count with the affine `scales` AND
  /// `biases` excluded as metadata. Hand-trace: packed `.weight` 16 `u32`
  /// = 64 bytes; `.scales` 2 `f32` = 8 bytes; `.biases` 2 `f32` = 8 bytes
  /// → `model_bytes = 80`. `model_params = packed_weight.size * 32 / bits`
  /// `= 16 * 32 / 4 = 128` (the affine `.scales` AND `.biases` are NOT in
  /// the denominator — they are quantization metadata). `bpw = 80 * 8 /
  /// 128 = 5.0`.
  #[test]
  fn compute_bits_per_weight_quantized_includes_scale_overhead() {
    let mut w: Weights = HashMap::new();
    w.insert(
      "model.q.weight".to_string(),
      Array::from_slice::<u32>(&[0u32; 16], &(2usize, 8)).unwrap(),
    );
    w.insert(
      "model.q.scales".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    w.insert(
      "model.q.biases".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    let quant = PerLayerQuantization::from_global(Quantization::affine(64, 4));
    let bpw = compute_bits_per_weight(&w, &quant).unwrap();
    // model_bytes * 8 / (packed_weight.size * 32 / bits)  — `.biases` is
    // no longer in the denominator.
    let expected = 80.0 * 8.0 / 128.0;
    assert!(
      (bpw - expected).abs() < 1e-9,
      "expected {expected}, got {bpw}"
    );
  }

  /// An empty weight map has zero parameters → a clean error, not a
  /// divide-by-zero NaN.
  #[test]
  fn compute_bits_per_weight_zero_params_errors() {
    let w: Weights = HashMap::new();
    let err = compute_bits_per_weight(&w, &PerLayerQuantization::default());
    assert!(matches!(err, Err(Error::Backend { .. })));
  }

  // ─────────────────── does_model_support_input_embeddings ───────────────

  #[test]
  fn does_model_support_input_embeddings_false_for_text_model() {
    // The text-only `MockModel` inherits the `false` default.
    let model = crate::lm::model::MockModel::new(4);
    assert!(!does_model_support_input_embeddings(&model));
  }

  // ─────────────────────── shard_file_name ───────────────────────

  /// The generation-tagged basename: `model-gen-{gen_id}-{idx:05}-of-{N:05}
  /// .safetensors`. Uniform across single- and multi-shard sets so the
  /// publish path has one code path. The exact `gen_id` value is not load-
  /// critical (the loader follows the index, not the basename), it is
  /// just a uniqueness handle so new shards never collide with a prior
  /// checkpoint's shard names.
  #[test]
  fn shard_file_name_generation_tagged() {
    let gen_id = "1234567890123-deadbeef-00000000cafef00d";
    assert_eq!(
      shard_file_name(gen_id, 1, 1),
      format!("model-gen-{gen_id}-00001-of-00001.safetensors")
    );
    assert_eq!(
      shard_file_name(gen_id, 1, 3),
      format!("model-gen-{gen_id}-00001-of-00003.safetensors")
    );
    assert_eq!(
      shard_file_name(gen_id, 3, 3),
      format!("model-gen-{gen_id}-00003-of-00003.safetensors")
    );
    // Two distinct generation ids produce distinct basenames — the
    // property that lets new-checkpoint shards never overwrite old-
    // checkpoint shards on disk.
    assert_ne!(
      shard_file_name("first-gen-id", 1, 1),
      shard_file_name("second-gen-id", 1, 1),
      "different generation ids must produce different shard names"
    );
  }

  /// [`new_gen_id`] returns the expected `{ts_us:013}-{pid:08x}-{ctr:016x}`
  /// shape (the `:013` is a MINIMUM width pad — a 2026-and-later µs
  /// timestamp is naturally 16 digits and is left unpadded by the
  /// format spec), the counter advances each call (so two saves from
  /// the same process can never share a `gen_id`), and the PID + ctr
  /// widths stay constant.
  #[test]
  fn new_gen_id_shape_and_counter_advance() {
    let a = new_gen_id();
    let b = new_gen_id();
    // Two calls produce two distinct ids (the counter component differs).
    assert_ne!(a, b, "successive new_gen_id() calls must differ");
    // Shape: three `-`-separated components.
    for id in [&a, &b] {
      let parts: Vec<&str> = id.split('-').collect();
      assert_eq!(
        parts.len(),
        3,
        "gen_id has 3 dash-separated components: {id}"
      );
      // ts_us is decimal digits, minimum 13 wide (the format-spec pad;
      // a real 2026-and-later µs-since-epoch is 16 digits naturally).
      assert!(
        parts[0].len() >= 13,
        "ts_us is at least 13 chars wide (the format-spec pad): {}",
        parts[0]
      );
      assert!(
        parts[0].chars().all(|c| c.is_ascii_digit()),
        "ts_us is decimal: {}",
        parts[0]
      );
      assert_eq!(parts[1].len(), 8, "pid is 8 hex chars: {}", parts[1]);
      assert!(
        parts[1].chars().all(|c| c.is_ascii_hexdigit()),
        "pid is hex: {}",
        parts[1]
      );
      assert_eq!(parts[2].len(), 16, "ctr is 16 hex chars: {}", parts[2]);
      assert!(
        parts[2].chars().all(|c| c.is_ascii_hexdigit()),
        "ctr is hex: {}",
        parts[2]
      );
    }
    // The pid component is identical across two calls in the same
    // process — only the counter (and possibly the timestamp) advances.
    let a_parts: Vec<&str> = a.split('-').collect();
    let b_parts: Vec<&str> = b.split('-').collect();
    assert_eq!(
      a_parts[1], b_parts[1],
      "PID stable across calls in the same process"
    );
  }

  // ─────────────────────── save_model round-trip ───────────────────────

  /// `save_model` writes a single generation-tagged shard (the 3 small
  /// weights fit one 5-GiB shard) plus a `model.safetensors.index.json`;
  /// [`load_weights`] reads the weights back byte-equal, and the index JSON
  /// has the expected `metadata` + sorted `weight_map`.
  #[test]
  fn save_model_single_shard_round_trips() {
    let dir = fresh_dir("save-model-single");
    let mut w: Weights = HashMap::new();
    // Distinct values so byte-equality is meaningful.
    w.insert(
      "model.b.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
    );
    w.insert(
      "model.a.weight".to_string(),
      Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
    );

    save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();

    // Exactly one shard file, named with the generation-tagged
    // `…-00001-of-00001` form (uniform single- + multi-shard naming).
    let shards = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with("-00001-of-00001.safetensors")
    })
    .unwrap();
    assert_eq!(
      shards.len(),
      1,
      "exactly one generation-tagged single shard file"
    );
    assert!(dir.join("model.safetensors.index.json").is_file());

    // Weights round-trip byte-equal via the index.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(
      loaded
        .get_mut("model.a.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![4.0, 5.0]
    );
    assert_eq!(
      loaded
        .get_mut("model.b.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0, 3.0]
    );

    // index.json: metadata + sorted weight_map.
    let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    // total_size = (2 + 3) elems × 4 bytes = 20.
    assert_eq!(index["metadata"]["total_size"], 20);
    // dense → total_parameters = 2 + 3 = 5.
    assert_eq!(index["metadata"]["total_parameters"], 5);
    let wm = index["weight_map"].as_object().unwrap();
    assert_eq!(wm.len(), 2);
    // Both weights are in the same single shard. The shard basename in the
    // index matches the on-disk file.
    let shard_basename = shards[0]
      .file_name()
      .unwrap()
      .to_string_lossy()
      .into_owned();
    assert_eq!(wm["model.a.weight"], shard_basename);
    assert_eq!(wm["model.b.weight"], shard_basename);
    // weight_map keys are sorted (a before b).
    let keys: Vec<&String> = wm.keys().collect();
    assert_eq!(keys, vec!["model.a.weight", "model.b.weight"]);

    // 4-space indent — Python `json.dump(indent=4)` parity.
    assert!(index_text.contains("\n    \"metadata\""));
    assert!(
      !index_text.ends_with('\n'),
      "json.dump writes no trailing newline"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `make_shards` borrows, never clones: each [`Shard`] entry points at the
  /// very same `Array` object in the input `weights` map. Verified by
  /// pointer identity (the shard's `&Array` is the input map's `&Array`).
  #[test]
  fn make_shards_borrows_without_cloning() {
    let mut w: Weights = HashMap::new();
    w.insert("x".to_string(), f32_weight(3));
    let shards = make_shards(&w, MAX_FILE_SIZE_GB).unwrap();
    assert_eq!(shards.len(), 1);
    let shard_ref: &Array = shards[0]["x"];
    let map_ref: &Array = w.get("x").unwrap();
    assert!(
      std::ptr::eq(shard_ref, map_ref),
      "make_shards must borrow the input array, not clone it"
    );
  }

  /// Multi-shard path: a `0`-GiB cap can't be passed to `save_model`
  /// (it hard-codes [`MAX_FILE_SIZE_GB`]), so this exercises the multi-shard
  /// *file naming + index* through `shard_file_name` +
  /// [`crate::io::save_safetensors_view`] directly, then confirms a
  /// hand-built 2-shard layout — published with its `weight_map` index —
  /// reloads via [`load_weights`] (index-honoring path). Asserts the
  /// generation-tagged naming scheme (`model-gen-{ts}-{idx:05}-of-{N:05}
  /// .safetensors`) at the basename level + that the on-disk files exactly
  /// match the index's `weight_map` values.
  #[test]
  fn save_model_multi_shard_naming_and_index_reload() {
    let dir = fresh_dir("save-model-multi");
    // Two weights; write them as a 2-shard layout by hand using the same
    // primitives `save_model` uses, to exercise the multi-shard names.
    let w0 = Array::from_slice::<f32>(&[10.0], &(1usize,)).unwrap();
    let w1 = Array::from_slice::<f32>(&[20.0, 21.0], &(2usize,)).unwrap();
    let shards: Vec<Shard<'_>> = vec![BTreeMap::from([("w0", &w0)]), BTreeMap::from([("w1", &w1)])];
    let count = shards.len();
    // Single generation id for the whole save — exactly what
    // `save_model` does internally (here a hand-crafted fixed value so
    // the asserted basenames are deterministic; production uses
    // `new_gen_id()`).
    let gen_id = "1234567890123-deadbeef-00000000cafef00d";
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    let mut written_basenames: Vec<String> = Vec::new();
    for (i, s) in shards.iter().enumerate() {
      let name = shard_file_name(gen_id, i + 1, count);
      // Generation-tagged scheme + zero-padded indices.
      assert_eq!(
        name,
        format!(
          "model-gen-{gen_id}-{:05}-of-{:05}.safetensors",
          i + 1,
          count
        )
      );
      crate::io::save_safetensors_view(&dir.join(&name), s.iter().map(|(&k, &v)| (k, v)), &meta)
        .unwrap();
      for &k in s.keys() {
        weight_map.insert(k.to_string(), name.clone());
      }
      written_basenames.push(name);
    }
    // The index makes the shard set discoverable by the index-honoring
    // [`load_weights`] path (without it, an absent `model.safetensors` /
    // `weights.safetensors` / `*.gguf` would error — and the bare-glob
    // resurrection of pre-index code is intentionally gone).
    write_json_pretty_to_path(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 12, "total_parameters": 3 },
        "weight_map": weight_map,
      }),
      "test: 2-shard index",
    )
    .unwrap();

    // Indices listed in the JSON exactly match the on-disk shard files
    // (no orphan shards on disk, no dangling index references).
    let on_disk: std::collections::BTreeSet<String> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap()
    .into_iter()
    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    .collect();
    let indexed: std::collections::BTreeSet<String> = weight_map.values().cloned().collect();
    assert_eq!(
      on_disk, indexed,
      "index `weight_map` values must exactly match the on-disk shard set"
    );
    let expected: std::collections::BTreeSet<String> = written_basenames.into_iter().collect();
    assert_eq!(
      indexed, expected,
      "index lists every generation-tagged shard we wrote, no more, no less"
    );

    // Both shard files reload + merge via the index-honoring `load_weights`.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(
      loaded.get_mut("w0").unwrap().to_vec::<f32>().unwrap(),
      vec![10.0]
    );
    assert_eq!(
      loaded.get_mut("w1").unwrap().to_vec::<f32>().unwrap(),
      vec![20.0, 21.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─────────────────────── save_config ───────────────────────

  /// `save_config` drops `_name_or_path` / `vision_config`, mirrors
  /// `quantization` into `quantization_config`, sorts the keys, and writes
  /// 4-space-indented JSON with no trailing newline.
  #[test]
  fn save_config_cleans_mirrors_and_sorts() {
    let dir = fresh_dir("save-config");
    let path = dir.join("config.json");
    let src = r#"{
      "model_type": "qwen3",
      "_name_or_path": "/tmp/should-be-dropped",
      "vision_config": {"drop": "me"},
      "hidden_size": 64,
      "quantization": {"group_size": 64, "bits": 4}
    }"#;
    save_config(src, &path).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let obj = v.as_object().unwrap();

    // Dropped keys.
    assert!(!obj.contains_key("_name_or_path"));
    assert!(!obj.contains_key("vision_config"));
    // `quantization` preserved AND mirrored to `quantization_config`.
    assert_eq!(obj["quantization"]["bits"], 4);
    assert_eq!(obj["quantization_config"]["bits"], 4);
    assert_eq!(obj["quantization_config"]["group_size"], 64);
    // Surviving content keys.
    assert_eq!(obj["model_type"], "qwen3");
    assert_eq!(obj["hidden_size"], 64);
    // Keys sorted ascending.
    let keys: Vec<&String> = obj.keys().collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "config.json keys must be sorted");

    // 4-space indent, no trailing newline.
    assert!(text.contains("\n    \""));
    assert!(!text.ends_with('\n'));
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn save_config_rejects_non_object_json() {
    let dir = fresh_dir("save-config-bad");
    let err = save_config("[1, 2, 3]", &dir.join("config.json"));
    assert!(matches!(err, Err(Error::Backend { .. })));
    let err2 = save_config("not json at all", &dir.join("config.json"));
    assert!(matches!(err2, Err(Error::Backend { .. })));
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─────────────────────── save (driver) ───────────────────────

  /// The `save` driver writes both the sharded weights+index and the
  /// cleaned `config.json`; the weights reload byte-equal and the config
  /// is the cleaned/sorted form.
  #[test]
  fn save_driver_writes_weights_and_config() {
    let dir = fresh_dir("save-driver");
    let mut w: Weights = HashMap::new();
    w.insert(
      "model.embed_tokens.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4usize,)).unwrap(),
    );
    let config = r#"{"model_type": "qwen3", "_name_or_path": "drop", "hidden_size": 8}"#;

    save(&dir, &w, config, &PerLayerQuantization::default()).unwrap();

    // Weights side: a single generation-tagged shard plus the index.
    let shards = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert_eq!(
      shards.len(),
      1,
      "the save driver produced exactly one generation-tagged shard"
    );
    assert!(dir.join("model.safetensors.index.json").is_file());
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(
      loaded
        .get_mut("model.embed_tokens.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0, 3.0, 4.0]
    );

    // Config side: `_name_or_path` dropped, keys sorted.
    let cfg_text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
    assert!(!cfg.as_object().unwrap().contains_key("_name_or_path"));
    assert_eq!(cfg["model_type"], "qwen3");
    assert_eq!(cfg["hidden_size"], 8);

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─────────────────────── save_model overwrite semantics ───────────────────────

  /// Overwriting a pre-existing checkpoint with the structurally-different
  /// generation-tagged naming: the loader follows the NEW index, so only
  /// the new weights are visible. Stale-shard files from the OLD
  /// checkpoint may remain on disk as orphans — they are deliberately
  /// invisible to load (the index is the authoritative manifest) and
  /// they are NOT inline-cleaned by `save_model` (the inline cleanup was
  /// removed in F6 round-5 because it raced concurrent readers; see
  /// `save_model` rustdoc). This test asserts the *load* contract — only
  /// the new keys appear — while letting orphan shards exist on disk;
  /// `save_model_no_overwrite_of_old_shards` covers the on-disk side.
  #[test]
  fn save_model_overwrite_loads_only_new_weights() {
    let dir = fresh_dir("save-model-overwrite-loads-new");

    // Stale 3-shard checkpoint, hand-written with the OLD reference-
    // style multi-shard names (the form a pre-F6-round-5 build, or any
    // hand-crafted checkpoint, could leave behind).
    let stale_vals = [
      ("stale.a.weight", vec![100.0_f32]),
      ("stale.b.weight", vec![200.0_f32, 201.0]),
      ("stale.c.weight", vec![300.0_f32, 301.0, 302.0]),
    ];
    let stale_arrays: Vec<(&str, Array)> = stale_vals
      .iter()
      .map(|(k, v)| (*k, Array::from_slice::<f32>(v, &(v.len(),)).unwrap()))
      .collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    let stale_count = stale_arrays.len();
    let mut stale_map: BTreeMap<String, String> = BTreeMap::new();
    for (i, (k, arr)) in stale_arrays.iter().enumerate() {
      let name = format!("model-{:05}-of-{:05}.safetensors", i + 1, stale_count);
      crate::io::save_safetensors_view(&dir.join(&name), std::iter::once((*k, arr)), &meta)
        .unwrap();
      stale_map.insert((*k).to_string(), name);
    }
    // A stale index too — the new save's index rename overwrites it.
    write_json_pretty_to_path(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 24, "total_parameters": 6 },
        "weight_map": stale_map,
      }),
      "test: stale index",
    )
    .unwrap();

    // Overwrite with a smaller single-shard checkpoint.
    let mut new_w: Weights = HashMap::new();
    new_w.insert(
      "fresh.x.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap(),
    );
    new_w.insert(
      "fresh.y.weight".to_string(),
      Array::from_slice::<f32>(&[3.0], &(1usize,)).unwrap(),
    );
    save_model(&dir, &new_w, &PerLayerQuantization::default()).unwrap();

    // `load_weights` sees ONLY the new checkpoint's keys — the stale
    // shards on disk are invisible because the new index does not list
    // them.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2, "only the two new weights load back");
    assert!(loaded.contains_key("fresh.x.weight"));
    assert!(loaded.contains_key("fresh.y.weight"));
    assert!(!loaded.contains_key("stale.a.weight"));
    assert!(!loaded.contains_key("stale.b.weight"));
    assert!(!loaded.contains_key("stale.c.weight"));
    assert_eq!(
      loaded
        .get_mut("fresh.x.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0]
    );

    // The index `weight_map` lists only the new keys; their values
    // reference exactly one generation-tagged shard.
    let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    let wm = index["weight_map"].as_object().unwrap();
    assert_eq!(wm.len(), 2);
    let shard_x = wm["fresh.x.weight"].as_str().unwrap();
    let shard_y = wm["fresh.y.weight"].as_str().unwrap();
    assert_eq!(
      shard_x, shard_y,
      "both new weights land in the same single shard"
    );
    assert!(
      shard_x.starts_with("model-gen-") && shard_x.ends_with("-00001-of-00001.safetensors"),
      "new shard is generation-tagged: got {shard_x}"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Re-saving the *same* checkpoint to a directory is a structurally
  /// safe operation: each save publishes its own generation-tagged
  /// shard, and the loader follows the latest index. The test asserts
  /// the load contract is stable across two consecutive saves.
  #[test]
  fn save_model_resave_same_checkpoint_is_stable() {
    let dir = fresh_dir("save-model-resave");
    let mut w: Weights = HashMap::new();
    w.insert("m.w.weight".to_string(), f32_weight(4));

    save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();
    save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();

    // Each save writes its own generation-tagged shard; the loader sees
    // exactly the latest one (one entry in the index, one weight loaded).
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(loaded.contains_key("m.w.weight"));
    assert_eq!(
      loaded
        .get_mut("m.w.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![0.0, 0.0, 0.0, 0.0]
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Generation-unique shard names mean a NEW save can never overwrite an
  /// OLD save's shard files on disk: after two consecutive saves to the
  /// same directory, BOTH saves' shard files coexist on disk, but only
  /// the SECOND save's shards are listed in the current index, and the
  /// loader returns exactly the second save's weights. This is the load-
  /// time guarantee the inline-cleanup removal trades for: prior-
  /// generation shards leak disk space but never corrupt the
  /// previously-valid checkpoint.
  #[test]
  fn save_model_no_overwrite_of_old_shards() {
    let dir = fresh_dir("save-no-overwrite");

    // FIRST save: a single weight whose value is byte-distinct from the
    // second save's, so a confused load would surface obviously.
    let mut first: Weights = HashMap::new();
    first.insert(
      "w.first.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
    );
    save_model(&dir, &first, &PerLayerQuantization::default()).unwrap();
    let first_shards: Vec<String> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap()
    .into_iter()
    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    .collect();
    assert_eq!(first_shards.len(), 1, "first save writes one shard");

    // Sleep so the millisecond timestamps of the two saves cannot
    // coincide (a 1-ms tick is enough; we add a small margin for
    // coarser-clock CI).
    std::thread::sleep(std::time::Duration::from_millis(5));

    // SECOND save: a different weight name + value.
    let mut second: Weights = HashMap::new();
    second.insert(
      "w.second.weight".to_string(),
      Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap(),
    );
    save_model(&dir, &second, &PerLayerQuantization::default()).unwrap();
    let all_shards: Vec<String> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap()
    .into_iter()
    .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
    .collect();

    // (1) Both saves' shard files coexist on disk — the second save did
    // NOT inline-clean the first save's shard (no overwrite was possible
    // because the basenames carry different generation timestamps).
    assert_eq!(
      all_shards.len(),
      2,
      "both saves' shard files coexist on disk (no inline cleanup); got {all_shards:?}"
    );
    for s in &first_shards {
      assert!(
        all_shards.contains(s),
        "the first save's shard {s} must survive the second save"
      );
    }

    // (2) Only the SECOND save's shards are listed in the current
    // index — the orphan first-save shards are invisible to load.
    let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    let wm = index["weight_map"].as_object().unwrap();
    assert_eq!(wm.len(), 1, "second save's index lists one weight");
    let indexed: std::collections::BTreeSet<String> = wm
      .values()
      .filter_map(|v| v.as_str().map(|s| s.to_string()))
      .collect();
    assert_eq!(
      indexed.len(),
      1,
      "all keys in the new index reference exactly one shard"
    );
    let indexed_shard = indexed.iter().next().unwrap().clone();
    assert!(
      !first_shards.contains(&indexed_shard),
      "the second save's index must not reference the first save's shard"
    );

    // (3) The loader returns exactly the SECOND save's weights via the
    // new index — no resurrected first-save tensors.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 1, "load sees only the new checkpoint");
    assert!(
      loaded.contains_key("w.second.weight"),
      "the second save's weight loads"
    );
    assert!(
      !loaded.contains_key("w.first.weight"),
      "the first save's weight is invisible to load (orphan on disk only)"
    );
    assert_eq!(
      loaded
        .get_mut("w.second.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![10.0, 20.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─────────────────── save_model failure-atomicity ───────────────────

  /// Failure-atomic save (Codex finding): when a `save_model` overwrite
  /// FAILS partway, the previously-valid checkpoint in the directory is
  /// left **fully intact and loadable**, and no partial `.tmp.safetensors`
  /// remains. A direct (non-atomic) per-shard write would clobber a
  /// still-valid shard *before* the new checkpoint is durable, leaving the
  /// directory neither the old checkpoint nor the new one.
  ///
  /// Failure is injected by making the checkpoint directory read-only so
  /// the next save's shard-tempfile `create_new` fails (mirrors
  /// `cache_prompt`'s read-only-dir injection). POSIX-only (`unix`): the
  /// permission bits are the failure lever.
  #[cfg(unix)]
  #[test]
  fn save_model_failed_save_keeps_previous_checkpoint_intact() {
    use std::os::unix::fs::PermissionsExt;

    let dir = fresh_dir("save-model-failed-intact");

    // 1. Write a good single-shard checkpoint.
    let mut orig: Weights = HashMap::new();
    orig.insert(
      "orig.a.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
    );
    orig.insert(
      "orig.b.weight".to_string(),
      Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
    );
    save_model(&dir, &orig, &PerLayerQuantization::default()).unwrap();
    // The original generation-tagged shard set (snapshotted before the
    // failed save so we can assert it survives byte-identical).
    let orig_shards: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert!(
      !orig_shards.is_empty(),
      "the original save produced at least one generation-tagged shard"
    );
    let orig_shard_bytes: BTreeMap<std::path::PathBuf, Vec<u8>> = orig_shards
      .iter()
      .map(|p| (p.clone(), std::fs::read(p).unwrap()))
      .collect();
    let orig_index = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();

    // 2. Make the directory read-only so the next save's tempfile
    //    `create_new` fails. (Root could bypass this; CI/dev users are not.)
    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    let orig_mode = perms.mode();
    perms.set_mode(0o500); // r-x------ : no write ⇒ create_new fails
    std::fs::set_permissions(&dir, perms).unwrap();

    // 3. Attempt to overwrite with a different checkpoint — must fail.
    let mut replacement: Weights = HashMap::new();
    replacement.insert("SHOULD.NOT.WIN.weight".to_string(), f32_weight(7));
    let r = save_model(&dir, &replacement, &PerLayerQuantization::default());

    // Restore write perms BEFORE asserting so cleanup + reads work even if
    // an assert fails.
    let mut restore = std::fs::metadata(&dir).unwrap().permissions();
    restore.set_mode(orig_mode);
    std::fs::set_permissions(&dir, restore).unwrap();

    assert!(r.is_err(), "a save into a read-only dir must fail");

    // 4. The original checkpoint is untouched: same shard set + index, it
    //    still `load_weights`-loads byte-equal, and no leftover tempfile.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2, "only the original two weights load back");
    assert!(loaded.contains_key("orig.a.weight"));
    assert!(loaded.contains_key("orig.b.weight"));
    assert!(
      !loaded.contains_key("SHOULD.NOT.WIN.weight"),
      "the failed save's weight must not have leaked in"
    );
    assert_eq!(
      loaded
        .get_mut("orig.a.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
      loaded
        .get_mut("orig.b.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![4.0, 5.0]
    );
    assert_eq!(
      std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap(),
      orig_index,
      "the original index.json must survive the failed save unchanged"
    );
    // Every original generation-tagged shard file is still on disk and
    // byte-identical to its pre-failed-save state.
    for (path, bytes) in &orig_shard_bytes {
      assert!(
        path.is_file(),
        "original shard {} must survive the failed save",
        path.display()
      );
      assert_eq!(
        &std::fs::read(path).unwrap(),
        bytes,
        "original shard {} must be byte-identical after the failed save",
        path.display()
      );
    }
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "no partial tempfile may remain after a failed save"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Failure-atomic save, rename-failure branch: when the final atomic
  /// `rename` of the **index** fails (here the index path pre-exists as a
  /// **directory**, which `fs::rename(file -> dir)` rejects), every staged
  /// `.tmp.safetensors` is cleaned up — no leftover tempfile. Note that
  /// the shard renames *do* succeed (their basenames are generation-
  /// tagged and never collide with any pre-existing file), so this test
  /// exercises specifically the index-rename failure path; the renamed
  /// shards become orphan files (deliberately not inline-cleaned).
  #[test]
  fn save_model_failed_save_rename_failure_cleans_up_tempfiles() {
    let dir = fresh_dir("save-model-failed-rename");
    // Pre-create the INDEX path as a directory so the final
    // `rename(file -> dir)` of the staged index fails.
    std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

    let mut w: Weights = HashMap::new();
    w.insert("m.w.weight".to_string(), f32_weight(4));
    let r = save_model(&dir, &w, &PerLayerQuantization::default());
    assert!(
      r.is_err(),
      "rename of the index onto an existing directory must fail"
    );

    // The colliding directory at the index path is untouched.
    assert!(
      dir.join("model.safetensors.index.json").is_dir(),
      "the colliding directory at the index path must be left untouched"
    );
    // No `.tmp.safetensors` leftover — every staged tempfile was removed
    // on the rename-failure cleanup path.
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "every staged tempfile must be removed when a rename fails"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `save_config` is failure-atomic too: a FAILED config write leaves a
  /// previously-valid `config.json` fully intact and removes the tempfile.
  /// Failure is injected with a read-only directory (POSIX-only).
  #[cfg(unix)]
  #[test]
  fn save_config_failed_write_keeps_previous_config_intact() {
    use std::os::unix::fs::PermissionsExt;

    let dir = fresh_dir("save-config-failed-intact");
    let config_path = dir.join("config.json");

    // 1. Write a good config.
    save_config(r#"{"model_type": "good", "hidden_size": 8}"#, &config_path).unwrap();
    let orig = std::fs::read_to_string(&config_path).unwrap();

    // 2. Make the directory read-only so the next write's tempfile
    //    `create_new` fails.
    let mut perms = std::fs::metadata(&dir).unwrap().permissions();
    let orig_mode = perms.mode();
    perms.set_mode(0o500);
    std::fs::set_permissions(&dir, perms).unwrap();

    // 3. Attempt to overwrite — must fail.
    let r = save_config(r#"{"model_type": "SHOULD-NOT-WIN"}"#, &config_path);

    let mut restore = std::fs::metadata(&dir).unwrap().permissions();
    restore.set_mode(orig_mode);
    std::fs::set_permissions(&dir, restore).unwrap();

    assert!(r.is_err(), "a config write into a read-only dir must fail");

    // 4. The original config is byte-identical, no leftover tempfile.
    assert_eq!(
      std::fs::read_to_string(&config_path).unwrap(),
      orig,
      "the original config.json must survive the failed write unchanged"
    );
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "no partial tempfile may remain after a failed config write"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ────────── load_weights: index-honoring + fallback resolution ──────────

  /// `load_weights` only loads shards listed in
  /// `model.safetensors.index.json` — a stale `model-*.safetensors` left
  /// on disk that is NOT in the index is invisible (the structural fix
  /// that makes the [`save_model`] index-rename single-commit-point safe).
  /// Hand-wires a single `model.safetensors` published via the
  /// `weight_map`, plus a stale `model-00099-of-00099.safetensors` carrying
  /// an extra weight; `load_weights` must return ONLY the indexed weight.
  #[test]
  fn load_weights_ignores_stale_shards_not_in_index() {
    let dir = fresh_dir("load-ignores-stale");
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());

    // The "real" indexed shard — a single `model.safetensors`.
    let real = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model.safetensors"),
      std::iter::once(("real.weight", &real)),
      &meta,
    )
    .unwrap();
    // The stale shard — present on disk, but NOT in the index. The
    // pre-structural-fix `load_weights` (which globbed
    // `model*.safetensors`) would have resurrected this tensor; the
    // index-honoring `load_weights` must NOT.
    let stale = Array::from_slice::<f32>(&[99.0], &(1usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model-00099-of-00099.safetensors"),
      std::iter::once(("stale.weight", &stale)),
      &meta,
    )
    .unwrap();
    // An index that names ONLY the real shard.
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    weight_map.insert("real.weight".to_string(), "model.safetensors".to_string());
    write_json_pretty_to_path(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 12, "total_parameters": 3 },
        "weight_map": weight_map,
      }),
      "test: index ignores stale",
    )
    .unwrap();

    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(
      loaded.len(),
      1,
      "only the indexed weight loads; the stale shard is invisible"
    );
    assert!(loaded.contains_key("real.weight"));
    assert!(
      !loaded.contains_key("stale.weight"),
      "an out-of-index shard must NOT resurrect tensors on load"
    );
    assert_eq!(
      loaded
        .get_mut("real.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0, 3.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// An un-sharded checkpoint that has only `model.safetensors` (no index
  /// file at all — the simple HF single-file convention) still loads via
  /// the second-tier fallback. Back-compat for fresh-from-`huggingface_hub`
  /// directories that don't carry an index.
  #[test]
  fn load_weights_no_index_single_model_safetensors_loads() {
    let dir = fresh_dir("load-single-no-index");
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());

    let w = Array::from_slice::<f32>(&[7.0, 8.0], &(2usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model.safetensors"),
      std::iter::once(("only.weight", &w)),
      &meta,
    )
    .unwrap();
    // No `model.safetensors.index.json`.
    assert!(!dir.join("model.safetensors.index.json").exists());

    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(
      loaded
        .get_mut("only.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![7.0, 8.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Legacy `weights.safetensors`-only directory (pre-HF naming) still
  /// loads via the third-tier fallback. No index, no `model.safetensors`,
  /// just `weights.safetensors`. Back-compat for older hand-rolled or
  /// pre-HF-convention checkpoints.
  #[test]
  fn load_weights_legacy_weights_safetensors_fallback_loads() {
    let dir = fresh_dir("load-legacy-weights");
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());

    let w = Array::from_slice::<f32>(&[42.0], &(1usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("weights.safetensors"),
      std::iter::once(("legacy.weight", &w)),
      &meta,
    )
    .unwrap();
    assert!(!dir.join("model.safetensors").exists());
    assert!(!dir.join("model.safetensors.index.json").exists());

    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(
      loaded
        .get_mut("legacy.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![42.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `load_weights` errors when the index lists a shard that does NOT exist
  /// on disk (the load-side counterpart to a torn-publish scenario where
  /// only some shards were renamed). The message names the missing shard.
  #[test]
  fn load_weights_index_lists_missing_shard_errors() {
    let dir = fresh_dir("load-index-missing-shard");
    // Index references `model-00001-of-00002.safetensors` +
    // `model-00002-of-00002.safetensors`, but only the first is on disk.
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    let w = Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model-00001-of-00002.safetensors"),
      std::iter::once(("a.weight", &w)),
      &meta,
    )
    .unwrap();

    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    weight_map.insert(
      "a.weight".to_string(),
      "model-00001-of-00002.safetensors".to_string(),
    );
    weight_map.insert(
      "b.weight".to_string(),
      "model-00002-of-00002.safetensors".to_string(),
    );
    write_json_pretty_to_path(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 8, "total_parameters": 2 },
        "weight_map": weight_map,
      }),
      "test: missing-shard index",
    )
    .unwrap();

    let r = load_weights(&dir);
    let Err(Error::Backend { message }) = r else {
      panic!("a missing indexed shard must be an Error::Backend, got {r:?}");
    };
    assert!(
      message.contains("model-00002-of-00002.safetensors"),
      "error must name the missing shard, got: {message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `load_weights` rejects an index whose `weight_map` value carries a
  /// path component (an absolute or `..`-traversing shard name would
  /// escape `dir`; HF convention is bare basenames in the same directory).
  #[test]
  fn load_weights_index_with_path_traversal_errors() {
    let dir = fresh_dir("load-index-path-traversal");
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    weight_map.insert("evil.weight".to_string(), "../../etc/passwd".to_string());
    write_json_pretty_to_path(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 0, "total_parameters": 0 },
        "weight_map": weight_map,
      }),
      "test: path-traversal index",
    )
    .unwrap();

    let r = load_weights(&dir);
    let Err(Error::Backend { message }) = r else {
      panic!("a path-traversal shard name must be an Error::Backend, got {r:?}");
    };
    assert!(
      message.contains("bare shard basename"),
      "error must call out the basename rule, got: {message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `load_weights` rejects a malformed (non-JSON) index file rather than
  /// silently falling through to the next tier — an unparseable index is a
  /// genuine corruption signal.
  #[test]
  fn load_weights_malformed_index_errors() {
    let dir = fresh_dir("load-index-malformed");
    std::fs::write(
      dir.join("model.safetensors.index.json"),
      b"this is not valid JSON {{{",
    )
    .unwrap();
    let r = load_weights(&dir);
    assert!(matches!(r, Err(Error::Backend { .. })));

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// An empty model directory (no index, no safetensors, no GGUF) is the
  /// final fall-through to an error. Lists every layout the resolver
  /// considered in the message.
  #[test]
  fn load_weights_empty_dir_errors_listing_layouts() {
    let dir = fresh_dir("load-empty");
    let r = load_weights(&dir);
    let Err(Error::Backend { message }) = r else {
      panic!("an empty dir must be an Error::Backend, got {r:?}");
    };
    // Lists each resolver tier.
    assert!(
      message.contains("model.safetensors.index.json")
        && message.contains("model.safetensors")
        && message.contains("weights.safetensors"),
      "the error must list each resolver tier, got: {message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ────────── save_model: index-rename single commit point ──────────

  /// **The structural atomicity test.** A save that fails AFTER the shards
  /// rename but BEFORE the index rename must leave the OLD checkpoint
  /// loadable EXACTLY (every weight key + value byte-identical), with the
  /// OLD `model.safetensors.index.json` untouched. The failure is injected
  /// by pre-creating `model.safetensors.index.json` as a *directory* so the
  /// final atomic `rename(file -> dir)` fails after every shard has been
  /// renamed into place. Because new shards are generation-tagged
  /// (`model-gen-{ts}-…`), the renames never collide with the OLD
  /// `model-00001-of-00002.safetensors` and
  /// `model-00002-of-00002.safetensors` files — the OLD shards are
  /// untouched by construction.
  #[test]
  fn save_model_torn_publish_before_index_rename_keeps_old_checkpoint() {
    let dir = fresh_dir("torn-publish-before-index-rename");
    // 1. Write an OLD 2-shard checkpoint with NON-colliding names.
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    let old_a = Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap();
    let old_b = Array::from_slice::<f32>(&[3.0, 4.0, 5.0], &(3usize,)).unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model-00001-of-00002.safetensors"),
      std::iter::once(("old.a.weight", &old_a)),
      &meta,
    )
    .unwrap();
    crate::io::save_safetensors_view(
      &dir.join("model-00002-of-00002.safetensors"),
      std::iter::once(("old.b.weight", &old_b)),
      &meta,
    )
    .unwrap();
    // The OLD index file — to be left untouched after the failed save.
    let mut old_wm: BTreeMap<String, String> = BTreeMap::new();
    old_wm.insert(
      "old.a.weight".to_string(),
      "model-00001-of-00002.safetensors".to_string(),
    );
    old_wm.insert(
      "old.b.weight".to_string(),
      "model-00002-of-00002.safetensors".to_string(),
    );
    let old_index_text = serde_json::to_string(&serde_json::json!({
      "metadata": { "total_size": 20, "total_parameters": 5 },
      "weight_map": old_wm,
    }))
    .unwrap();
    // Sanity: confirm the OLD shards are loadable when we put the OLD
    // index in place.
    std::fs::write(
      dir.join("model.safetensors.index.json"),
      old_index_text.as_bytes(),
    )
    .unwrap();
    let mut sanity = load_weights(&dir).unwrap();
    assert_eq!(sanity.len(), 2);
    assert_eq!(
      sanity
        .get_mut("old.a.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0]
    );
    drop(sanity);
    // 2. Remove the OLD index file and plant a directory in its place so
    //    the final atomic `rename(file -> dir)` of the NEW index fails
    //    AFTER every NEW shard has been renamed into place. We will assert
    //    that after the failed save, restoring the OLD index file lets
    //    load follow it to the still-intact OLD shards.
    std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
    std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

    // 3. Attempt the new save — a smaller single-shard checkpoint.
    let mut new_w: Weights = HashMap::new();
    new_w.insert(
      "new.x.weight".to_string(),
      Array::from_slice::<f32>(&[100.0], &(1usize,)).unwrap(),
    );
    let r = save_model(&dir, &new_w, &PerLayerQuantization::default());
    assert!(
      r.is_err(),
      "the index rename onto an existing directory must fail"
    );

    // 4. The OLD shards must be untouched, and byte-identical.
    let old_a_path = dir.join("model-00001-of-00002.safetensors");
    let old_b_path = dir.join("model-00002-of-00002.safetensors");
    assert!(
      old_a_path.is_file(),
      "OLD shard 1 must survive the failed save"
    );
    assert!(
      old_b_path.is_file(),
      "OLD shard 2 must survive the failed save"
    );
    // The NEW shard was renamed into place before the failed index
    // rename; load ignores it as long as it isn't indexed. The OLD index
    // doesn't list it, so it's invisible to a load via the OLD index —
    // exactly the design's promise.
    let new_shards_on_disk: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert_eq!(
      new_shards_on_disk.len(),
      1,
      "the NEW shard rename SHOULD have succeeded (it's the index rename that fails); \
       this asserts the torn-publish scenario the test is targeting"
    );
    // No staged tempfile remains.
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "every staged tempfile must be removed when the index rename fails"
    );

    // 5. Restore the OLD index file (replacing the directory we used as
    //    the failure lever) and confirm load follows it to the still-
    //    intact OLD shards. The NEW shard is on disk but is invisible —
    //    load only sees the OLD-indexed shards.
    std::fs::remove_dir_all(dir.join("model.safetensors.index.json")).unwrap();
    std::fs::write(
      dir.join("model.safetensors.index.json"),
      old_index_text.as_bytes(),
    )
    .unwrap();
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(
      loaded.len(),
      2,
      "the OLD checkpoint loads EXACTLY (both weights)"
    );
    assert_eq!(
      loaded
        .get_mut("old.a.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0],
      "old.a is byte-identical"
    );
    assert_eq!(
      loaded
        .get_mut("old.b.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![3.0, 4.0, 5.0],
      "old.b is byte-identical"
    );
    assert!(
      !loaded.contains_key("new.x.weight"),
      "the NEW shard is on disk but the OLD index ignores it"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// The new torn-publish guarantee, end-to-end via the public
  /// [`save_model`] API: stage a save over an EXISTING checkpoint,
  /// complete shard staging + shard renames, then fail the index rename
  /// (by planting a directory at the index destination path). Because
  /// the NEW shard basenames are generation-tagged, they cannot
  /// overwrite the OLD shards — and the OLD index is left intact, so
  /// the loader still returns the OLD checkpoint EXACTLY.
  ///
  /// Distinct from `save_model_torn_publish_before_index_rename_keeps_old_checkpoint`:
  /// that test hand-builds the OLD layout with the reference-style names
  /// to assert the structural intent; this one round-trips through
  /// `save_model` for both saves to prove the end-to-end guarantee
  /// holds against the production code path.
  #[test]
  fn save_model_torn_after_shard_rename_before_index_rename_keeps_old_checkpoint() {
    let dir = fresh_dir("torn-after-shard-before-index");

    // 1. FIRST save: produce a legitimate checkpoint via `save_model`.
    let mut first: Weights = HashMap::new();
    first.insert(
      "first.alpha.weight".to_string(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap(),
    );
    first.insert(
      "first.beta.weight".to_string(),
      Array::from_slice::<f32>(&[4.0, 5.0], &(2usize,)).unwrap(),
    );
    save_model(&dir, &first, &PerLayerQuantization::default()).unwrap();

    // Snapshot the OLD checkpoint: every shard's bytes + the OLD index
    // body, all for a post-failure byte-equality check.
    let old_shard_paths: Vec<std::path::PathBuf> = collect_sorted(&dir, |n| {
      n.starts_with("model-gen-") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert!(
      !old_shard_paths.is_empty(),
      "first save produced at least one shard"
    );
    let old_shard_bytes: BTreeMap<std::path::PathBuf, Vec<u8>> = old_shard_paths
      .iter()
      .map(|p| (p.clone(), std::fs::read(p).unwrap()))
      .collect();
    let old_index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();

    // 2. Plant a directory at the index path AFTER removing the OLD
    //    index file (so the OLD shards still sit on disk untouched, but
    //    the next `save_model`'s index rename will fail).
    std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
    std::fs::create_dir_all(dir.join("model.safetensors.index.json")).unwrap();

    // Sleep so the generation timestamp of the second save is guaranteed
    // distinct from the first save's, even on coarser-clock platforms.
    std::thread::sleep(std::time::Duration::from_millis(5));

    // 3. SECOND save: must fail at the index rename, after the new
    //    shard(s) have been renamed into place.
    let mut second: Weights = HashMap::new();
    second.insert(
      "second.gamma.weight".to_string(),
      Array::from_slice::<f32>(&[100.0, 200.0], &(2usize,)).unwrap(),
    );
    let r = save_model(&dir, &second, &PerLayerQuantization::default());
    assert!(
      r.is_err(),
      "the index rename onto an existing directory must fail"
    );

    // 4. Every OLD shard is still on disk + byte-identical to its
    //    pre-failed-save state (the unique generation-tagged basenames
    //    of the SECOND save guaranteed they could not overwrite anything).
    for (path, bytes) in &old_shard_bytes {
      assert!(
        path.is_file(),
        "OLD shard {} must survive the failed save",
        path.display()
      );
      assert_eq!(
        &std::fs::read(path).unwrap(),
        bytes,
        "OLD shard {} must be byte-identical after the failed save",
        path.display()
      );
    }

    // 5. Restore the OLD index file (replacing the failure-lever
    //    directory) and confirm the loader returns the OLD checkpoint
    //    EXACTLY — no resurrected second-save weights.
    std::fs::remove_dir_all(dir.join("model.safetensors.index.json")).unwrap();
    std::fs::write(
      dir.join("model.safetensors.index.json"),
      old_index_text.as_bytes(),
    )
    .unwrap();
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains_key("first.alpha.weight"));
    assert!(loaded.contains_key("first.beta.weight"));
    assert!(
      !loaded.contains_key("second.gamma.weight"),
      "the SECOND save's shard is on disk but the OLD index ignores it"
    );
    assert_eq!(
      loaded
        .get_mut("first.alpha.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
      loaded
        .get_mut("first.beta.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![4.0, 5.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Smoke test the `fsync_dir` helper: open + fsync + close on a
  /// writable tmpdir works without error. The function returns
  /// `io::Result<()>` rather than `Result<()>` so the call sites in
  /// `save_model` / `commit_staged_config` wrap with their own error
  /// context.
  #[test]
  fn fsync_dir_helper_basic() {
    let dir = fresh_dir("fsync-dir-helper");
    // Sanity: the helper signature is `fsync_dir(&Path) -> io::Result<()>`.
    let r: std::io::Result<()> = fsync_dir(&dir);
    r.expect("fsync_dir must succeed on a writable tmpdir");
    let _ = std::fs::remove_dir_all(&dir);
  }

  // ────────── save: config-staging cheap fix ──────────

  /// `save` validates + stages the config BEFORE [`save_model`] touches any
  /// weight. An invalid config (malformed JSON) over an existing checkpoint
  /// leaves the checkpoint **byte-identical** to its pre-save state — every
  /// weight, the index, and the `config.json` are untouched.
  #[test]
  fn save_invalid_config_keeps_existing_checkpoint_byte_identical() {
    let dir = fresh_dir("save-invalid-config-intact");

    // 1. Write a valid initial checkpoint via the public `save` driver.
    let mut w: Weights = HashMap::new();
    w.insert(
      "orig.a.weight".to_string(),
      Array::from_slice::<f32>(&[10.0, 20.0], &(2usize,)).unwrap(),
    );
    w.insert(
      "orig.b.weight".to_string(),
      Array::from_slice::<f32>(&[30.0], &(1usize,)).unwrap(),
    );
    let good_config = r#"{"model_type": "qwen3", "hidden_size": 64}"#;
    save(&dir, &w, good_config, &PerLayerQuantization::default()).unwrap();

    // Capture the pre-failed-save byte snapshot of every file.
    let snapshot = |dir: &Path| -> BTreeMap<String, Vec<u8>> {
      let mut m: BTreeMap<String, Vec<u8>> = BTreeMap::new();
      for e in std::fs::read_dir(dir).unwrap().flatten() {
        if e.file_type().unwrap().is_file() {
          let name = e.file_name().to_string_lossy().into_owned();
          let bytes = std::fs::read(e.path()).unwrap();
          m.insert(name, bytes);
        }
      }
      m
    };
    let before = snapshot(&dir);
    assert!(
      before
        .keys()
        .any(|k| k.starts_with("model-gen-") && k.ends_with(".safetensors")),
      "the initial save produced a generation-tagged shard"
    );
    assert!(before.contains_key("model.safetensors.index.json"));
    assert!(before.contains_key("config.json"));

    // 2. Attempt a second save with an INVALID config — must fail.
    let bad_config = "this is not valid JSON at all";
    let other_weights: Weights = {
      let mut m: Weights = HashMap::new();
      m.insert(
        "SHOULD.NOT.WIN.weight".to_string(),
        Array::from_slice::<f32>(&[999.0], &(1usize,)).unwrap(),
      );
      m
    };
    let r = save(
      &dir,
      &other_weights,
      bad_config,
      &PerLayerQuantization::default(),
    );
    assert!(r.is_err(), "an invalid config must abort the save");

    // 3. EVERY file is byte-identical to the pre-failed-save state — the
    //    cheap config-staging fix's promise.
    let after = snapshot(&dir);
    // Filter out any tempfile (there should be none, but if any leaks
    // we want to assert separately and not have it pollute the byte-equal
    // comparison).
    let strip_tmp = |m: BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Vec<u8>> {
      m.into_iter()
        .filter(|(k, _)| !k.ends_with(".tmp.safetensors"))
        .collect()
    };
    let leftover_tmp = after.keys().any(|k| k.ends_with(".tmp.safetensors"));
    assert_eq!(
      strip_tmp(before),
      strip_tmp(after),
      "every file under {} must be byte-identical after an invalid-config save",
      dir.display()
    );
    assert!(
      !leftover_tmp,
      "no staged config tempfile may remain after an invalid-config save"
    );

    // 4. The original checkpoint still loads cleanly.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(
      loaded
        .get_mut("orig.a.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![10.0, 20.0]
    );
    assert!(!loaded.contains_key("SHOULD.NOT.WIN.weight"));

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ───────────── get_total_parameters: scale-only `.biases` ─────────────

  /// A `.biases` tensor present under a **scale-only** quant mode
  /// (`mxfp4` / `mxfp8` / `nvfp4`) is structurally invalid — those layouts
  /// have no zero-point buffer and reject one. `get_total_parameters` must
  /// flag it as an [`Error::Backend`], NOT silently skip it as it does for
  /// the affine zero-point. Checked for all three scale-only modes.
  #[test]
  fn get_total_parameters_scale_only_biases_is_error() {
    for mode in [QuantMode::Mxfp4, QuantMode::Mxfp8, QuantMode::Nvfp4] {
      let mut w: Weights = HashMap::new();
      w.insert(
        "model.layers.0.q_proj.weight".to_string(),
        Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
      );
      w.insert(
        "model.layers.0.q_proj.scales".to_string(),
        Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
      );
      // A stale `.biases` sibling — invalid under a scale-only layout.
      w.insert(
        "model.layers.0.q_proj.biases".to_string(),
        Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
      );
      let quant = PerLayerQuantization::from_global(Quantization {
        group_size: 32,
        bits: 4,
        mode,
      });
      let err = get_total_parameters(&w, &quant);
      assert!(
        matches!(err, Err(Error::Backend { .. })),
        "a `.biases` under scale-only `{}` must be an Error::Backend, got {err:?}",
        mode.as_mlx_str()
      );
      // The error names the offending layer and mode.
      if let Err(Error::Backend { message }) = err {
        assert!(
          message.contains("q_proj") && message.contains(mode.as_mlx_str()),
          "error should name the layer + the scale-only mode, got: {message}"
        );
      }
    }
  }

  /// The affine counterpart: under `QuantMode::Affine` the `.biases`
  /// zero-point buffer is still correctly skipped as metadata (not
  /// counted, no error). Hand-trace: packed `.weight` 8 `u32`, `bits = 4`
  /// → `8 * 32 / 4 = 64` logical weights; `.scales` + `.biases` → +0.
  /// Total = 64.
  #[test]
  fn get_total_parameters_affine_biases_still_skipped() {
    let mut w: Weights = HashMap::new();
    w.insert(
      "model.layers.0.q_proj.weight".to_string(),
      Array::from_slice::<u32>(&[0u32; 8], &(2usize, 4)).unwrap(),
    );
    w.insert(
      "model.layers.0.q_proj.scales".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    w.insert(
      "model.layers.0.q_proj.biases".to_string(),
      Array::from_slice::<f32>(&[0.0, 0.0], &(2usize,)).unwrap(),
    );
    let quant = PerLayerQuantization::from_global(Quantization::affine(32, 4));
    let total = get_total_parameters(&w, &quant).unwrap();
    assert_eq!(
      total, 64,
      "affine `.biases` skipped, only unpacked weight counts"
    );
  }

  // ────────── F6 R6: collision-resistant gen_id + fail-closed rename ──────────

  /// Two consecutive `save_model` calls from the same process — even in a
  /// tight loop where the µs timestamp may not advance between calls —
  /// produce on-disk shards with distinct basenames, because the process-
  /// global counter component of [`new_gen_id`] always advances. Without
  /// the counter the timestamp-only F6 R5 tag would have collided
  /// whenever two saves landed in the same ms / µs tick (and the second
  /// save would have overwritten the first save's shard via
  /// `fs::rename`); the counter closes that hole.
  #[test]
  fn gen_id_is_collision_resistant_across_same_ms_saves() {
    let dir_a = fresh_dir("gen-id-collision-a");
    let dir_b = fresh_dir("gen-id-collision-b");
    let mut w: Weights = HashMap::new();
    w.insert("w.weight".to_string(), f32_weight(2));
    // Back-to-back saves to two distinct dirs to keep the assertion
    // about basenames, not about a single-dir overwrite (that's covered
    // by `save_model_no_overwrite_of_old_shards`).
    save_model(&dir_a, &w, &PerLayerQuantization::default()).unwrap();
    save_model(&dir_b, &w, &PerLayerQuantization::default()).unwrap();

    let basenames = |dir: &Path| -> Vec<String> {
      collect_sorted(dir, |n| {
        n.starts_with("model-gen-") && n.ends_with(".safetensors")
      })
      .unwrap()
      .into_iter()
      .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
      .collect()
    };
    let a = basenames(&dir_a);
    let b = basenames(&dir_b);
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    // Even if the µs timestamp tick did not advance between the two
    // saves (e.g. on a coarser-clock host), the counter advances, so
    // the basenames differ.
    assert_ne!(
      a[0], b[0],
      "two same-process saves must produce distinct gen_id-tagged basenames; \
       got {a:?} == {b:?}"
    );

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
  }

  /// Defense-in-depth: if a pre-existing file occupies one of the
  /// predicted final shard paths (the collision-resistant `gen_id`
  /// makes this statistically unreachable, so the test plants the file
  /// by hand after forcing the gen_id via the test-only
  /// `force_next_gen_id` helper) `save_model`'s atomic no-replace
  /// `std::fs::hard_link` MUST fail with `ErrorKind::AlreadyExists`
  /// and the save MUST surface that as
  /// [`crate::Error::ShardPathCollision`] naming the offending path —
  /// the planted file is byte-identical (the no-replace primitive
  /// cannot overwrite, unlike `rename(2)`) and no staged tempfiles
  /// leak.
  #[test]
  fn save_model_refuses_to_overwrite_existing_shard_basename() {
    let dir = fresh_dir("save-refuses-overwrite");

    // 1. Pick a known gen_id and plant a decoy file at the shard-1-of-1
    //    path that gen_id will predict.
    let forced_gen_id = "9999999999999-cafebabe-0000000000000042";
    let collision_path = dir.join(shard_file_name(forced_gen_id, 1, 1));
    let decoy_bytes = b"pre-existing decoy bytes that must NOT be overwritten";
    std::fs::write(&collision_path, decoy_bytes).unwrap();

    // 2. Force `save_model`'s next gen_id to match the planted path.
    force_next_gen_id(forced_gen_id);

    let mut w: Weights = HashMap::new();
    w.insert("w.weight".to_string(), f32_weight(2));
    let r = save_model(&dir, &w, &PerLayerQuantization::default());

    // 3. The save aborts with `Error::ShardPathCollision` naming the
    //    offending path — the atomic no-replace `hard_link` mapped
    //    `ErrorKind::AlreadyExists` to this variant.
    match r {
      Err(Error::ShardPathCollision { path }) => {
        assert_eq!(
          path, collision_path,
          "the collision error names the planted path"
        );
      }
      other => panic!("expected Err(ShardPathCollision), got {other:?}"),
    }

    // 4. The decoy file is byte-identical — `hard_link`'s no-replace
    //    semantics guarantee no overwrite.
    assert!(
      collision_path.is_file(),
      "the planted decoy at {} must still be a file",
      collision_path.display()
    );
    assert_eq!(
      std::fs::read(&collision_path).unwrap(),
      decoy_bytes,
      "the planted decoy must be byte-identical (hard_link refused to replace)"
    );

    // 5. No staged `.tmp.safetensors` leaks (every staged tempfile was
    //    cleaned up on the collision-cleanup path).
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "no staged tempfile may remain after a ShardPathCollision abort"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// F6 R7: the shard publish primitive must be **atomic no-replace**,
  /// not a `symlink_metadata` + `rename` pre-check. A check-then-act
  /// has a TOCTOU window: the stat returns `NotFound`, a concurrent
  /// writer creates the final path, then `rename(2)` SILENTLY replaces
  /// the racing peer's bytes. With `std::fs::hard_link` the race is
  /// closed at the syscall boundary — the call either creates the new
  /// directory entry or fails `AlreadyExists`, never overwriting.
  ///
  /// The simplest faithful simulation of the race is to plant the
  /// colliding file BEFORE calling `save_model`: from `hard_link`'s
  /// perspective the final path already exists when the syscall runs,
  /// which is exactly the state a TOCTOU race would leave the
  /// filesystem in. (A `symlink_metadata` + `rename` implementation
  /// would also catch this specific pre-plant via the pre-check, but
  /// the contract under test is "the primitive is atomic no-replace",
  /// not "the pre-check happens to catch a pre-plant". Together with
  /// the original-test pre-plant case both arms are exercised: this
  /// test ALSO asserts no-tempfile-leak + no-NEW-index-commit, which
  /// the original does not.) Contract:
  ///
  /// 1. `save_model` returns `Err(Error::ShardPathCollision { path })`
  ///    naming the planted path.
  /// 2. The planted file is byte-identical — `hard_link` cannot
  ///    overwrite, so no bytes are clobbered.
  /// 3. No `.tmp.safetensors` leaks — the collision-cleanup path
  ///    removed every staged tempfile.
  /// 4. No NEW `model.safetensors.index.json` exists in the directory
  ///    — the index rename is the observable commit point and the
  ///    save aborted BEFORE it; the directory has no index file at
  ///    all (we started from a `fresh_dir`).
  #[test]
  fn save_model_concurrent_create_at_final_path_returns_collision_error_not_silent_overwrite() {
    let dir = fresh_dir("save-toctou-no-silent-overwrite");

    // Predict the final shard path from a forced gen_id and plant a
    // file there — equivalent to a concurrent peer winning the race
    // against a `symlink_metadata` + `rename` pre-check (from
    // `hard_link`'s perspective the path is already there when the
    // syscall runs).
    let forced_gen_id = "7777777777777-feedface-00000000beefcafe";
    let final_shard = dir.join(shard_file_name(forced_gen_id, 1, 1));
    let racer_bytes = b"racer-bytes: a concurrent writer's payload that MUST survive";
    std::fs::write(&final_shard, racer_bytes).unwrap();

    force_next_gen_id(forced_gen_id);

    let mut w: Weights = HashMap::new();
    w.insert("z.weight".to_string(), f32_weight(3));
    let r = save_model(&dir, &w, &PerLayerQuantization::default());

    // (1) `Err(ShardPathCollision { path: final_shard })`.
    match r {
      Err(Error::ShardPathCollision { path }) => {
        assert_eq!(
          path, final_shard,
          "collision error names the planted (racer) path"
        );
      }
      other => {
        panic!("expected Err(ShardPathCollision) from atomic no-replace hard_link, got {other:?}")
      }
    }

    // (2) Planted bytes survive byte-identical — `hard_link` is no-
    // replace; a silent overwrite would have clobbered these.
    assert!(
      final_shard.is_file(),
      "the racer file at {} must still be a regular file",
      final_shard.display()
    );
    assert_eq!(
      std::fs::read(&final_shard).unwrap(),
      racer_bytes,
      "racer bytes must be byte-identical — atomic no-replace forbids silent overwrite"
    );

    // (3) No leftover tempfile in the dir.
    let leftover_tmp = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(
      !leftover_tmp,
      "no staged .tmp.safetensors may remain after a ShardPathCollision"
    );

    // (4) No NEW index — the save aborted before the index rename
    // (the observable commit point). Directory was fresh, so the
    // index file must not exist.
    let index_path = dir.join("model.safetensors.index.json");
    assert!(
      !index_path.exists(),
      "no index commit may occur when shard publish fails: {} exists",
      index_path.display()
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ────────── F6 R6: post-index-commit durability warning ──────────

  /// `save_model` returns `Ok(CommitOutcome::CommittedWithDurabilityWarning)`
  /// — NOT `Err` — when the post-index-rename `fsync_dir` fails. The
  /// visible checkpoint loads correctly; only the parent-directory
  /// fsync hiccupped. This is the F6 R6 Finding-2 hole: returning `Err`
  /// here would propagate through [`save`] and drop the staged
  /// [`StagedConfig`], deleting its tempfile and leaving NEW
  /// weights+index against the OLD config.
  ///
  /// Driven via the test-only `arm_fsync_dir_fault(skip)`: `skip=1`
  /// makes the shard-fsync succeed and the INDEX-fsync (the
  /// observable-commit-point fsync) fail. The contract:
  ///
  /// 1. `save_model` returns `Ok(CommittedWithDurabilityWarning(_))`.
  /// 2. The on-disk checkpoint loads correctly (`load_weights` sees the
  ///    new weights).
  #[test]
  fn save_model_post_index_fsync_failure_keeps_visible_checkpoint() {
    let dir = fresh_dir("post-index-fsync-failure");

    let mut w: Weights = HashMap::new();
    w.insert(
      "v.alpha.weight".to_string(),
      Array::from_slice::<f32>(&[7.0, 8.0, 9.0], &(3usize,)).unwrap(),
    );
    w.insert(
      "v.beta.weight".to_string(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );

    // Arm: skip the FIRST fsync_dir call (after shard renames) then
    // fail the second (after the index rename — the durability fsync
    // that follows the observable commit point).
    let _guard = arm_fsync_dir_fault(1);
    let outcome = save_model(&dir, &w, &PerLayerQuantization::default())
      .expect("post-index fsync failure must NOT propagate as Err — it is a durability warning");
    drop(_guard);

    // (1) The returned outcome is the warning variant carrying the
    // injected error.
    let underlying = match outcome {
      CommitOutcome::CommittedWithDurabilityWarning(e) => e,
      CommitOutcome::Committed => {
        panic!("expected CommittedWithDurabilityWarning, got Committed")
      }
    };
    let underlying_msg = underlying.to_string();
    assert!(
      underlying_msg.contains("injected fsync_dir failure"),
      "the durability warning carries the underlying io::Error: got {underlying_msg}"
    );

    // (2) The visible checkpoint loads correctly — the index rename
    // succeeded, so `load_weights` sees the NEW weights.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains_key("v.alpha.weight"));
    assert!(loaded.contains_key("v.beta.weight"));
    assert_eq!(
      loaded
        .get_mut("v.alpha.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![7.0, 8.0, 9.0]
    );
    assert_eq!(
      loaded
        .get_mut("v.beta.weight")
        .unwrap()
        .to_vec::<f32>()
        .unwrap(),
      vec![1.0]
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// `save` proceeds to commit the staged config even when `save_model`
  /// returned `CommittedWithDurabilityWarning` — the NEW `config.json`
  /// MUST be byte-equal to the staged (cleaned/sorted) content, the OLD
  /// `config.json` is gone, and `save`'s final return is
  /// [`Error::DurabilityWarning`] with `committed: true`. This is the
  /// end-to-end F6 R6 Finding-2 closure.
  #[test]
  fn save_post_commit_durability_warning_still_commits_config() {
    let dir = fresh_dir("save-post-commit-warning-commits-config");

    // 1. Initial save with a "before" config so we can prove the OLD
    //    config.json is gone after the second save.
    let mut w0: Weights = HashMap::new();
    w0.insert("w.weight".to_string(), f32_weight(2));
    let before_config = r#"{"model_type": "OLD", "hidden_size": 4}"#;
    save(&dir, &w0, before_config, &PerLayerQuantization::default()).unwrap();
    let old_cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    assert!(
      old_cfg.contains("\"OLD\""),
      "the OLD config.json was written"
    );

    // 2. Second save with a "after" config + a fsync injection that
    //    fires AFTER the index rename inside save_model (skip=1 — the
    //    shard fsync passes, the index fsync fails). The save MUST
    //    still commit the config (otherwise the staged-config Drop
    //    would delete its tempfile and we'd be left with NEW
    //    weights+index against the OLD config).
    let mut w1: Weights = HashMap::new();
    w1.insert(
      "w.weight".to_string(),
      Array::from_slice::<f32>(&[5.0, 6.0], &(2usize,)).unwrap(),
    );
    let after_config = r#"{"model_type": "NEW", "hidden_size": 8}"#;

    let _guard = arm_fsync_dir_fault(1);
    let r = save(&dir, &w1, after_config, &PerLayerQuantization::default());
    drop(_guard);

    // (1) save's final return is `Err(DurabilityWarning{committed:true})`.
    match r {
      Err(Error::DurabilityWarning { committed, source }) => {
        assert!(
          committed,
          "save's DurabilityWarning must carry committed=true"
        );
        assert!(
          source.to_string().contains("injected fsync_dir failure"),
          "the underlying io::Error must be preserved: got {source}"
        );
      }
      other => panic!("expected Err(DurabilityWarning), got {other:?}"),
    }

    // (2) The NEW config.json is on disk and byte-equal to the staged
    //    (cleaned/sorted) form of `after_config`.
    let new_cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    assert!(
      new_cfg.contains("\"NEW\""),
      "the NEW config.json must be on disk: got {new_cfg}"
    );
    assert!(
      !new_cfg.contains("\"OLD\""),
      "the OLD config.json content must be gone: got {new_cfg}"
    );
    // The cleaned-and-sorted form of `after_config` (4-space indented,
    // sorted keys, no trailing newline).
    let expected_cfg = {
      let v: serde_json::Value = serde_json::from_str(after_config).unwrap();
      let obj = v.as_object().unwrap().clone();
      let sorted: BTreeMap<String, serde_json::Value> = obj.into_iter().collect();
      let mut buf = Vec::new();
      let fmt = serde_json::ser::PrettyFormatter::with_indent(b"    ");
      let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
      serde::Serialize::serialize(&sorted, &mut ser).unwrap();
      String::from_utf8(buf).unwrap()
    };
    assert_eq!(
      new_cfg, expected_cfg,
      "the NEW config.json must be byte-equal to the staged (cleaned/sorted) form"
    );

    // (3) The visible weights are the NEW ones — `load_weights` loads
    //    via the NEW index that the (warned-on) save did commit.
    let mut loaded = load_weights(&dir).unwrap();
    assert_eq!(
      loaded.get_mut("w.weight").unwrap().to_vec::<f32>().unwrap(),
      vec![5.0, 6.0]
    );

    // (4) No staged tempfile leaks behind.
    let leftover = std::fs::read_dir(&dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .any(|e| {
        e.file_name()
          .to_string_lossy()
          .ends_with(".tmp.safetensors")
      });
    assert!(!leftover, "no staged tempfile may leak");

    let _ = std::fs::remove_dir_all(&dir);
  }

  // ─────────────────────── LOAD-1 (#145): fd-bound atomic-save writers ───────────────────────

  /// [`open_excl_temp_shard`] returns BOTH the open [`File`] and the path
  /// so callers can write through the original-open fd (no reopen-by-name
  /// TOCTOU window). The pre-LOAD-1 signature returned only the path; this
  /// test asserts the post-fix shape, verifies the file was actually
  /// created on disk, that we can write through the fd, and that the bytes
  /// land on the inode the path points at.
  #[test]
  fn open_excl_temp_shard_returns_file_and_path() {
    use std::io::Write as _;
    let dir = fresh_dir("load1-open-excl-shape");
    let final_path = dir.join("model-00001-of-00001.safetensors");
    let (mut f, tmp) = open_excl_temp_shard(&final_path).unwrap();
    // The path is a same-directory `.tmp.safetensors` sibling of the
    // final path (no cross-directory tempfile).
    assert_eq!(tmp.parent().unwrap(), final_path.parent().unwrap());
    assert!(
      tmp
        .file_name()
        .unwrap()
        .to_string_lossy()
        .ends_with(".tmp.safetensors"),
      "tempfile must keep the .tmp.safetensors suffix, got {}",
      tmp.display()
    );
    // It exists on disk.
    assert!(tmp.exists(), "open_excl_temp_shard must create the file");
    // Writing through the returned `File` is observable at the path —
    // proves the `File` is bound to the same on-disk object as `tmp`.
    let payload = b"LOAD-1: fd-bound shard tempfile";
    f.write_all(payload).unwrap();
    drop(f);
    let on_disk = std::fs::read(&tmp).unwrap();
    assert_eq!(
      on_disk, payload,
      "bytes written through the returned File must land at the returned path"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **The LOAD-1 TOCTOU regression test for the safetensors writer.**
  /// Replace the staging tempfile with a symlink pointing at a "decoy"
  /// file AFTER opening the staging fd, then call the fd-bound writer.
  /// The writes must land in the ORIGINAL fd's inode (now anonymous —
  /// the path resolves to the decoy), not the decoy. Inode comparison
  /// catches the case where a reopen-by-name would have followed the
  /// symlink.
  #[test]
  fn save_safetensors_to_file_writes_via_fd_not_reopen_by_path() {
    use std::os::unix::fs::MetadataExt;
    let dir = fresh_dir("load1-safetensors-fd-not-reopen");
    let staging = dir.join("staging.tmp.safetensors");
    let decoy = dir.join("decoy.target");
    // Plant the decoy with known bytes.
    std::fs::write(&decoy, b"DECOY: must not be overwritten").unwrap();
    let decoy_meta_before = std::fs::metadata(&decoy).unwrap();
    let decoy_inode_before = decoy_meta_before.ino();
    // Open the staging fd via the same primitive `save_model` uses (an
    // `O_EXCL` create).
    let (mut staging_file, staging_path) = open_excl_temp_shard(&staging).unwrap();
    let staging_inode = std::fs::metadata(&staging_path).unwrap().ino();
    assert_ne!(
      staging_inode, decoy_inode_before,
      "test sanity: staging tempfile + decoy must be distinct inodes"
    );
    // Simulate the attack: unlink the staging path + symlink it to the
    // decoy. A reopen-by-name from this point on would follow the symlink
    // and write into the decoy. The staging fd we just opened, however,
    // is still pinned to the original (now-anonymous) inode.
    std::fs::remove_file(&staging_path).unwrap();
    std::os::unix::fs::symlink(&decoy, &staging_path).unwrap();
    // Drive the fd-bound writer with a small array.
    let arr = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3usize,)).unwrap();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    crate::io::save_safetensors_to_file(&mut staging_file, std::iter::once(("w", &arr)), &meta)
      .unwrap();
    drop(staging_file);
    // Assert the decoy is byte-for-byte unchanged + still the same inode.
    let decoy_after = std::fs::read(&decoy).unwrap();
    assert_eq!(
      decoy_after, b"DECOY: must not be overwritten",
      "decoy must not be touched by the fd-bound writer"
    );
    let decoy_meta_after = std::fs::metadata(&decoy).unwrap();
    assert_eq!(
      decoy_meta_after.ino(),
      decoy_inode_before,
      "decoy inode must not have changed"
    );
    // Also: the symlink itself still resolves to the decoy (the staging
    // path entry is the symlink, not a new file).
    let lmeta = std::fs::symlink_metadata(&staging_path).unwrap();
    assert!(lmeta.file_type().is_symlink());
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **The LOAD-1 TOCTOU regression test for the JSON writer.** Same
  /// shape as the safetensors test: replace the staging path with a
  /// symlink to a decoy AFTER opening the staging fd, call the fd-bound
  /// `write_json_pretty`, and assert the decoy is untouched. The
  /// pre-LOAD-1 `write_json_pretty(&Path,...)` would `fs::write` the
  /// symlinked decoy.
  #[test]
  fn write_json_pretty_writes_via_fd_not_reopen_by_path() {
    use std::os::unix::fs::MetadataExt;
    let dir = fresh_dir("load1-json-fd-not-reopen");
    let staging = dir.join("staging.tmp.safetensors");
    let decoy = dir.join("decoy.json");
    std::fs::write(&decoy, b"{\"decoy\": true}").unwrap();
    let decoy_inode_before = std::fs::metadata(&decoy).unwrap().ino();
    let (mut staging_file, staging_path) = open_excl_temp_shard(&staging).unwrap();
    std::fs::remove_file(&staging_path).unwrap();
    std::os::unix::fs::symlink(&decoy, &staging_path).unwrap();
    // Drive the fd-bound JSON writer.
    let value = serde_json::json!({
      "metadata": { "total_size": 0, "total_parameters": 0 },
      "weight_map": {},
    });
    write_json_pretty(&mut staging_file, &value, "LOAD-1: json fd-bound").unwrap();
    drop(staging_file);
    let decoy_after = std::fs::read(&decoy).unwrap();
    assert_eq!(
      decoy_after, b"{\"decoy\": true}",
      "decoy JSON must be untouched by the fd-bound writer"
    );
    assert_eq!(
      std::fs::metadata(&decoy).unwrap().ino(),
      decoy_inode_before,
      "decoy inode must not have changed"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Functional round-trip for the fd-bound safetensors writer: an array
  /// written through `save_safetensors_to_file` reloads byte-for-byte via
  /// [`crate::io::load_safetensors`]. Confirms the custom `mlx_io_writer`
  /// (which delegates `tell`/`seek`/`write` to the supplied `&mut File`)
  /// drives mlx-c through a correct safetensors layout — JSON header,
  /// per-tensor `data_offsets`, then the contiguous tensor-data section
  /// — equivalent in semantics to the path-based writer. The on-disk
  /// byte sequence cannot be asserted equal to a path-based write because
  /// mlx-c serializes the entry map (an `std::unordered_map`) in a
  /// non-deterministic order — the safetensors LAYOUT is invariant, the
  /// per-tensor offsets are not.
  #[test]
  fn save_safetensors_to_file_round_trips_via_path_load() {
    let dir = fresh_dir("load1-fd-round-trip");
    let arr_a = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(4usize,)).unwrap();
    let arr_b = Array::from_slice::<f32>(&[10.0_f32, 20.0], &(2usize,)).unwrap();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());

    // Fd-based write into a freshly-created `File`.
    let path = dir.join("via_fd.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    crate::io::save_safetensors_to_file(&mut f, [("a", &arr_a), ("b", &arr_b)], &meta).unwrap();
    f.sync_all().unwrap();
    drop(f);

    // Reload through the path-based loader — proves the on-disk
    // safetensors layout is valid (parseable header, correct offsets,
    // correct dtype + shape encoding).
    let mut loaded = crate::io::load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    let a_read = loaded.get_mut("a").unwrap().to_vec::<f32>().unwrap();
    let b_read = loaded.get_mut("b").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(a_read, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(b_read, vec![10.0, 20.0]);

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Happy-path: prefilled file at non-zero cursor → clean
  /// safetensors.** Without the internal rewind in
  /// `save_safetensors_to_file`, a caller-supplied `File` at a non-zero
  /// cursor would receive a safetensors header at the current cursor +
  /// stale prefilled bytes as the prefix — producing a corrupt file
  /// that `load_safetensors` could not parse, while the writer returned
  /// `Ok(())`. This test pre-fills the file with 100 bytes, seeks to
  /// byte 50, drives the writer with a small array, and asserts the
  /// reload succeeds + the on-disk size equals exactly the new
  /// safetensors payload size (no leading 50 bytes of garbage, no
  /// trailing 50 bytes of garbage). Documents the destructive truncate
  /// is part of the happy-path contract — see the "Destructive
  /// mutation" section of `save_safetensors_to_file`'s doc comment.
  #[test]
  fn save_safetensors_to_file_truncates_prefilled_file_at_nonzero_offset() {
    use std::io::{Seek, SeekFrom, Write as _};
    let dir = fresh_dir("load1-fd-prefilled-nonzero");
    let path = dir.join("prefilled_nonzero.safetensors");
    // Pre-fill the file with 100 bytes of obviously-not-safetensors data,
    // then seek to byte 50. The writer must reset to byte 0 + truncate
    // before writing — otherwise the on-disk bytes would start with the
    // first 50 prefill bytes and the safetensors payload would follow at
    // offset 50, yielding an unparseable file.
    let mut f = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .create(true)
      .truncate(true)
      .open(&path)
      .unwrap();
    f.write_all(&[0xAB_u8; 100]).unwrap();
    f.seek(SeekFrom::Start(50)).unwrap();
    let arr = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(4usize,)).unwrap();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    crate::io::save_safetensors_to_file(&mut f, std::iter::once(("w", &arr)), &meta).unwrap();
    f.sync_all().unwrap();
    drop(f);
    // The file must now parse as a clean safetensors with exactly the
    // one array we wrote — no leading garbage from the prefill prefix.
    let mut loaded = crate::io::load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 1, "expected exactly one tensor in the file");
    let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
    // And the on-disk size must equal exactly the fresh safetensors
    // payload size — written-via-`save_safetensors_view` to a control
    // path with the same array + metadata. A retained prefill prefix
    // or suffix would push the size past the control. We can't hard-
    // code the byte count because mlx-c's JSON-header layout (key
    // order, whitespace) is an implementation detail, but parity with
    // the path-based writer is the contract this fix establishes.
    let control_path = dir.join("control.safetensors");
    let mut control_arrays: HashMap<String, &Array> = HashMap::new();
    control_arrays.insert("w".to_string(), &arr);
    crate::io::save_safetensors_view(
      &control_path,
      control_arrays.iter().map(|(k, &v)| (k.as_str(), v)),
      &meta,
    )
    .unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len();
    let control_size = std::fs::metadata(&control_path).unwrap().len();
    assert_eq!(
      on_disk, control_size,
      "fd-bound writer on a prefilled-at-offset-50 file must produce the same \
       byte count as the path-based writer on a fresh file (proves rewind+truncate \
       wiped the 100-byte prefill); fd={on_disk}, control={control_size}"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Happy-path: prefilled file longer than new payload → trailing
  /// bytes truncated.** Without the internal `set_len(0)` truncate, a
  /// caller-supplied `File` that already held a much larger blob would
  /// retain the trailing tail bytes after the new (shorter)
  /// safetensors — the resulting file's prefix would parse but its
  /// overall byte length would lie about the payload size, and
  /// downstream tooling that mmaps / hashes / verifies the whole file
  /// would see garbage past the safetensors EOF. Pre-fills 10000 bytes,
  /// rewinds to 0, writes a small payload, asserts the final file size
  /// matches a fresh small payload (well under 10000) and reloads
  /// correctly. Documents the destructive truncate is part of the
  /// happy-path contract — see the "Destructive mutation" section of
  /// `save_safetensors_to_file`'s doc comment.
  #[test]
  fn save_safetensors_to_file_truncates_prefilled_file_longer_than_new_payload() {
    use std::io::{Seek, SeekFrom, Write as _};
    let dir = fresh_dir("load1-fd-prefilled-longer");
    let path = dir.join("prefilled_longer.safetensors");
    let mut f = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .create(true)
      .truncate(true)
      .open(&path)
      .unwrap();
    // 10000 bytes of obviously-not-safetensors data, then rewind to 0.
    f.write_all(&[0xCD_u8; 10000]).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    let arr = Array::from_slice::<f32>(&[7.0_f32, 8.0, 9.0], &(3usize,)).unwrap();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    crate::io::save_safetensors_to_file(&mut f, std::iter::once(("w", &arr)), &meta).unwrap();
    f.sync_all().unwrap();
    drop(f);
    let mut loaded = crate::io::load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(w, vec![7.0, 8.0, 9.0]);
    // Final file size must equal exactly a fresh control write — any
    // retained trailing prefill (the bytes past the new shorter
    // payload) would push it past the control. The control is
    // `save_safetensors_view` on a fresh path with the same single
    // array + metadata.
    let control_path = dir.join("control.safetensors");
    let mut control_arrays: HashMap<String, &Array> = HashMap::new();
    control_arrays.insert("w".to_string(), &arr);
    crate::io::save_safetensors_view(
      &control_path,
      control_arrays.iter().map(|(k, &v)| (k.as_str(), v)),
      &meta,
    )
    .unwrap();
    let on_disk = std::fs::metadata(&path).unwrap().len();
    let control_size = std::fs::metadata(&control_path).unwrap().len();
    assert_eq!(
      on_disk, control_size,
      "fd-bound writer on a 10000-byte-prefilled file must produce the same byte \
       count as the path-based writer on a fresh file (proves set_len(0) truncated \
       trailing prefill); fd={on_disk}, control={control_size}"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Defense-in-depth: interior-NUL in metadata key leaves file
  /// untouched.** Verifies the structural ordering inside
  /// `save_safetensors_to_file`: input-validation `Err` from
  /// `build_string_map` (interior-NUL in a metadata key) returns BEFORE
  /// the destructive `seek(0)` + `set_len(0)`, so a caller-owned
  /// prefilled file is byte-identical to its pre-call state on this
  /// error path.
  ///
  /// NOT a contract — see the "Destructive mutation" section of
  /// `save_safetensors_to_file`'s doc comment. Callers MUST NOT rely on
  /// byte preservation across save failures; use the fd-bound
  /// tempfile-staging pattern (open a same-directory `O_EXCL` `File`,
  /// pass it to `save_safetensors_to_file`, `sync_all`, then `rename` /
  /// `hard_link` to the final path — the open/write/fsync/drop fd-bound
  /// steps are exemplified by `save_model` above at lines 1359-1372) to
  /// preserve the fd-bound write-redirection mitigation through the
  /// staging write. The fd-bound mitigation covers the WRITE PATH only;
  /// the publication step (`rename` / `hard_link` by `temp_path`) is
  /// pathname-based and still subject to directory-entry substitution
  /// any time after the `O_EXCL` create and before publication (not
  /// just after fsync). See the "Scope of this guarantee" caveat in
  /// `save_safetensors_to_file`'s doc comment (its `# Destructive
  /// mutation` doc section) for the publication-race options. This test
  /// guards the defense-in-depth ordering does not regress, not a
  /// behavioral contract callers can depend on.
  #[test]
  fn save_safetensors_to_file_preserves_existing_file_on_interior_nul_metadata() {
    let dir = fresh_dir("load1-fd-r2-nul-meta");
    let path = dir.join("preexisting_meta.safetensors");
    // Pre-fill the file with known content (NOT a valid safetensors —
    // we are testing that the file is left UNCHANGED on Err, not that
    // it's still loadable). The byte sequence is arbitrary; the
    // assertion is byte-equality to `original_bytes`.
    let original_bytes: &[u8] = b"existing valid safetensors payload here";
    std::fs::write(&path, original_bytes).unwrap();
    let original_len = original_bytes.len() as u64;
    // Sanity: file is exactly the prefill before the call.
    assert_eq!(
      std::fs::metadata(&path).unwrap().len(),
      original_len,
      "pre-call: file size must equal prefill length"
    );

    let mut file = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .open(&path)
      .unwrap();
    let array = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap();
    let mut bad_metadata: HashMap<String, String> = HashMap::new();
    // Interior NUL in the key — `CString::new` rejects this, so
    // `build_string_map` returns `Err` before any FFI call.
    bad_metadata.insert("key\0with-nul".to_string(), "value".to_string());

    let result = crate::io::save_safetensors_to_file(
      &mut file,
      std::iter::once(("name", &array)),
      &bad_metadata,
    );

    // The call must surface an interior-NUL `Error::Backend`.
    assert!(
      result.is_err(),
      "expected Err from interior-NUL in metadata key, got Ok"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
      err_msg.contains("NUL") || err_msg.contains("nul"),
      "expected an interior-NUL error message, got: {err_msg}"
    );

    // Defense-in-depth: the file must be byte-identical to the prefill
    // because `build_string_map` rejected the interior-NUL key BEFORE
    // the destructive truncate ran. A regression that re-ordered the
    // truncate ahead of the validation step would zero this file.
    drop(file);
    let bytes_after = std::fs::read(&path).unwrap();
    assert_eq!(
      bytes_after, original_bytes,
      "DEFENSE-IN-DEPTH REGRESSION: input-validation Err from build_string_map must \
       return before the destructive seek+set_len so a caller-owned prefilled file is \
       byte-identical to its pre-call state on this error path. NOT a contract — see \
       save_safetensors_to_file's Destructive mutation doc section."
    );
    assert_eq!(
      bytes_after.len() as u64,
      original_len,
      "post-call: file size must still equal prefill length"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Defense-in-depth: interior-NUL in array name leaves file
  /// untouched.** Symmetric to the metadata-key case above, exercising
  /// the OTHER fallible map-build site (`build_array_map`). Verifies
  /// the structural ordering: input-validation `Err` from
  /// `build_array_map` returns BEFORE the destructive truncate.
  ///
  /// NOT a contract — see the "Destructive mutation" section of
  /// `save_safetensors_to_file`'s doc comment. Callers MUST NOT rely on
  /// byte preservation across save failures.
  #[test]
  fn save_safetensors_to_file_preserves_existing_file_on_interior_nul_array_name() {
    let dir = fresh_dir("load1-fd-r2-nul-name");
    let path = dir.join("preexisting_name.safetensors");
    let original_bytes: &[u8] = b"another distinct prefilled payload, array-name path";
    std::fs::write(&path, original_bytes).unwrap();
    let original_len = original_bytes.len() as u64;
    assert_eq!(
      std::fs::metadata(&path).unwrap().len(),
      original_len,
      "pre-call: file size must equal prefill length"
    );

    let mut file = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .open(&path)
      .unwrap();
    let array = Array::from_slice::<f32>(&[3.0_f32, 4.0, 5.0], &(3usize,)).unwrap();
    let good_metadata: HashMap<String, String> = HashMap::new();

    // Interior NUL in the array name — `build_array_map`'s
    // `CString::new(k)` rejects this and returns `Err` BEFORE any FFI
    // call.
    let bad_name = "arr\0with-nul";
    let result = crate::io::save_safetensors_to_file(
      &mut file,
      std::iter::once((bad_name, &array)),
      &good_metadata,
    );

    assert!(
      result.is_err(),
      "expected Err from interior-NUL in array name, got Ok"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
      err_msg.contains("NUL") || err_msg.contains("nul"),
      "expected an interior-NUL error message, got: {err_msg}"
    );

    drop(file);
    let bytes_after = std::fs::read(&path).unwrap();
    assert_eq!(
      bytes_after, original_bytes,
      "DEFENSE-IN-DEPTH REGRESSION (array-name path): input-validation Err from \
       build_array_map must return before the destructive seek+set_len so a \
       caller-owned prefilled file is byte-identical to its pre-call state on this \
       error path. NOT a contract — see save_safetensors_to_file's Destructive \
       mutation doc section."
    );
    assert_eq!(
      bytes_after.len() as u64,
      original_len,
      "post-call: file size must still equal prefill length"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Defense-in-depth: empty-metadata save succeeds through the
  /// NULL-sentinel guard.** Verifies the `ctx.is_null()` check in
  /// `build_string_map` (installed to surface a hypothetical
  /// `mlx_map_string_to_string_new()` allocation-failure sentinel) does
  /// not reject valid handles: with empty metadata the insert loop runs
  /// zero times, so the structural NULL guard is the only filter
  /// between the `_new()` and the caller. A bug that inverted the
  /// predicate or compared the wrong field would surface here as an
  /// `Err` on the most common save shape. The structural test below
  /// verifies the source carries the explicit check.
  ///
  /// NOT a contract — verifies the defense-in-depth ordering does not
  /// regress on the success path. See `save_safetensors_to_file`'s
  /// Destructive mutation doc section.
  #[test]
  fn save_safetensors_to_file_empty_metadata_succeeds_post_r3_null_check() {
    let dir = fresh_dir("load1-fd-r3-empty-meta-ok");
    let path = dir.join("empty_meta_ok.safetensors");
    let mut file = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .create_new(true)
      .open(&path)
      .unwrap();
    let arr = Array::from_slice::<f32>(&[1.5_f32, 2.5, 3.5], &(3usize,)).unwrap();
    // Empty `HashMap<String, String>` metadata is the shape that
    // bypasses every `_insert` call in `build_string_map`, so the
    // structural `is_null()` guard is the only thing between a
    // hypothetical NULL-ctx sentinel from `_new()` and the caller. A
    // valid (non-NULL) handle must pass through unchanged.
    let empty_metadata: HashMap<String, String> = HashMap::new();
    crate::io::save_safetensors_to_file(&mut file, std::iter::once(("w", &arr)), &empty_metadata)
      .expect(
        "DEFENSE-IN-DEPTH REGRESSION: empty-metadata save_safetensors_to_file must \
         succeed — the NULL-sentinel guard in build_string_map must not reject valid \
         handles. See save_safetensors_to_file's Destructive mutation doc section.",
      );
    file.sync_all().unwrap();
    drop(file);

    let mut loaded = crate::io::load_safetensors(&path).unwrap();
    assert_eq!(loaded.len(), 1, "round-trip must yield exactly one array");
    let w = loaded.get_mut("w").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(
      w,
      vec![1.5, 2.5, 3.5],
      "round-trip values must match the pre-save array"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Defense-in-depth structural: map-helper NULL-sentinel guards.**
  /// Real allocation failure inside `mlx_map_string_to_array_new()` /
  /// `mlx_map_string_to_string_new()` cannot be deterministically
  /// injected from a unit test (no allocator-fault hook is plumbed
  /// through to the C++ vendored map ctor), and the empty-input shape
  /// makes EVERY post-construction defensive `_insert` call a no-op so
  /// behavioral coverage cannot trip the NULL path on a real machine.
  /// This test reads `mlxrs/src/io.rs` and asserts both `build_array_map`
  /// and `build_string_map` carry an explicit `ctx.is_null()` check
  /// immediately after the corresponding `_new()` constructor, and
  /// drain `crate::error::LAST` rather than peek. A regression that
  /// removes either check (e.g. a refactor that drops the guard or
  /// reorders the call past the file mutation) will fail this test.
  ///
  /// Guards the defense-in-depth ordering, not a byte-preservation
  /// contract — see `save_safetensors_to_file`'s Destructive mutation
  /// doc section.
  #[test]
  fn build_map_helpers_carry_r3_null_sentinel_check() {
    // Read the SOURCE we shipped (not the compiled binary) so a future
    // edit that deletes the guard fails this test deterministically,
    // independent of inlining / optimization. The path is relative to
    // the cargo manifest dir of the `mlxrs` crate.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/io.rs"))
      .expect("must be able to read mlxrs/src/io.rs to verify NULL-sentinel guards");

    // Locate `fn build_array_map` and assert the body contains an
    // explicit `is_null()` predicate AND a `mlx_map_string_to_array_new`
    // call within the same logical region. We slice the source at the
    // function header and check the next ~3 KiB — comfortably larger
    // than either helper body but small enough that any NULL check
    // found belongs to the function it follows.
    let array_fn = src
      .find("fn build_array_map")
      .expect("build_array_map must exist in io.rs");
    let array_window = &src[array_fn..(array_fn + 3000).min(src.len())];
    assert!(
      array_window.contains("mlx_map_string_to_array_new"),
      "DEFENSE-IN-DEPTH STRUCTURAL: build_array_map must still call \
       mlx_map_string_to_array_new"
    );
    assert!(
      array_window.contains("ctx.is_null()"),
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: build_array_map must check \
       `guard.0.ctx.is_null()` immediately after `mlx_map_string_to_array_new()` to \
       surface allocation-failure sentinels; the check appears to have been removed."
    );

    let string_fn = src
      .find("fn build_string_map")
      .expect("build_string_map must exist in io.rs");
    let string_window = &src[string_fn..(string_fn + 3000).min(src.len())];
    assert!(
      string_window.contains("mlx_map_string_to_string_new"),
      "DEFENSE-IN-DEPTH STRUCTURAL: build_string_map must still call \
       mlx_map_string_to_string_new"
    );
    assert!(
      string_window.contains("ctx.is_null()"),
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: build_string_map must check \
       `guard.0.ctx.is_null()` immediately after `mlx_map_string_to_string_new()` — \
       without this guard, an allocation failure on the empty-metadata save path \
       returns a NULL-ctx sentinel through `Ok(NULL)` to the caller. The check \
       appears to have been removed."
    );

    // Both windows must also drain LAST (not peek). The drain is the
    // crate's `crate::error::take_last()` / `LAST.with(...).take()`
    // idiom; either spelling is acceptable — but SOMETHING must
    // consume the TLS so a stale Err does not poison the next call.
    let drains_last = |window: &str| {
      window.contains("take_last()")
        || window.contains("LAST.with")
        || window.contains("crate::error::take_last")
    };
    assert!(
      drains_last(array_window),
      "DEFENSE-IN-DEPTH STRUCTURAL: build_array_map's NULL branch must DRAIN \
       crate::error::LAST (via take_last() or LAST.with(..).take()), not peek \
       — leaving a stale Err in the TLS pollutes later mlx-c calls on this thread."
    );
    assert!(
      drains_last(string_window),
      "DEFENSE-IN-DEPTH STRUCTURAL: build_string_map's NULL branch must DRAIN \
       crate::error::LAST (via take_last() or LAST.with(..).take()), not peek \
       — leaving a stale Err in the TLS pollutes later mlx-c calls on this thread."
    );
  }

  /// **Defense-in-depth structural: writer-new precedes truncate.**
  /// `mlx_io_writer_new` allocates a `cwriter_holder` +
  /// `std::shared_ptr<CWriter>` inside its `try`/`catch` (vendored
  /// `mlx-c/mlx/c/private/io.h:126-129` +
  /// `mlx-c/mlx/c/io_types.cpp:48-54`) and converts a `std::bad_alloc`
  /// (or any other exception) into a `mlx_io_writer({nullptr})`
  /// sentinel. Real allocation failure inside that ctor cannot be
  /// deterministically injected from a unit test (no allocator-fault
  /// hook is plumbed through to the vendored C++ ctor), so this test
  /// guards the structural ordering: it reads `mlxrs/src/io.rs` and
  /// asserts the lexical ordering — (1) `mlx_io_writer_new` is called
  /// BEFORE `seek(SeekFrom::Start(0))`, (2) an explicit
  /// `ctx.is_null()` check appears within ~10 lines of
  /// `mlx_io_writer_new`, (3) a `take_last()` (or
  /// `crate::error::take_last`) drain appears within ~20 lines of
  /// `mlx_io_writer_new`, (4) `set_len(0)` appears AFTER the
  /// `is_null()` check.
  ///
  /// Guards the defense-in-depth ordering, not a byte-preservation
  /// contract — see `save_safetensors_to_file`'s Destructive mutation
  /// doc section.
  #[test]
  fn save_safetensors_to_file_writer_new_precedes_truncate_r4_structural() {
    // Read the SOURCE we shipped (not the compiled binary) so a future
    // edit that re-orders the writer ctor past the truncate fails this
    // test deterministically, independent of inlining / optimization.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/io.rs"))
      .expect("must be able to read mlxrs/src/io.rs to verify writer-precedes-truncate ordering");

    // Restrict the search to the `save_safetensors_to_file` function
    // body so unrelated occurrences (the `WriterGuard` impl, the
    // vtable factory, the `cb_seek` callback) cannot satisfy the
    // assertions by accident.
    let fn_start = src
      .find("pub fn save_safetensors_to_file")
      .expect("save_safetensors_to_file must exist in io.rs");
    // The next sibling item header in this file is the
    // `// ─────── mlx_io_writer backed by &mut File ──────` divider
    // immediately followed by `struct WriterState`. Slice up to that
    // point to capture only the function body.
    let fn_tail = src[fn_start..]
      .find("struct WriterState")
      .expect("WriterState declaration must follow save_safetensors_to_file in io.rs");
    let body = &src[fn_start..fn_start + fn_tail];

    // Locate each landmark by byte offset within the function body so
    // we can compare lexical ordering directly.
    let writer_new_off = body.find("mlx_io_writer_new(").expect(
      "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must construct the \
       mlx_io_writer via `mlx_io_writer_new(...)`; the writer-new call appears to \
       have been removed or renamed.",
    );
    let seek_off = body.find("seek(SeekFrom::Start(0))").expect(
      "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must rewind via \
       `seek(SeekFrom::Start(0))`; the rewind appears to have been removed or renamed.",
    );
    let set_len_off = body.find("set_len(0)").expect(
      "DEFENSE-IN-DEPTH STRUCTURAL: save_safetensors_to_file must truncate via \
       `set_len(0)`; the truncate appears to have been removed or renamed.",
    );

    // Invariant 1: writer-new BEFORE seek.
    assert!(
      writer_new_off < seek_off,
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: `mlx_io_writer_new(...)` must appear \
       BEFORE `seek(SeekFrom::Start(0))` inside save_safetensors_to_file so an \
       allocation failure in the writer ctor surfaces as Err before the destructive \
       truncate. Current ordering has writer-new at byte {writer_new_off} and \
       seek at byte {seek_off}.",
    );

    // Invariant 2: explicit `is_null()` check within 10 lines after writer-new.
    let post_writer_new = &body[writer_new_off..];
    let next_lines: Vec<&str> = post_writer_new.lines().take(11).collect();
    let null_check_window = next_lines.join("\n");
    assert!(
      null_check_window.contains(".ctx.is_null()") || null_check_window.contains("ctx.is_null()"),
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: within 10 lines after \
       `mlx_io_writer_new(...)` there must be an explicit `.ctx.is_null()` check \
       that drains the NULL-ctx sentinel before any destructive file mutation. \
       The check appears to have been removed.",
    );

    // Invariant 3: `take_last()` (or `crate::error::take_last`) drain
    // within 20 lines after writer-new.
    let drain_lines: Vec<&str> = post_writer_new.lines().take(21).collect();
    let drain_window = drain_lines.join("\n");
    assert!(
      drain_window.contains("take_last()") || drain_window.contains("crate::error::take_last"),
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: within 20 lines after \
       `mlx_io_writer_new(...)` there must be a `take_last()` (or \
       `crate::error::take_last`) DRAIN of the TLS error slot — peeking would \
       leave a stale Err and poison the next unrelated mlx-c call on this thread. \
       The drain appears to have been removed or replaced with a peek.",
    );

    // Invariant 4: `set_len(0)` AFTER the `is_null()` check.
    let null_check_local_off = null_check_window
      .find("ctx.is_null()")
      .expect("invariant-2 guard above asserted this exists; cannot fail here");
    let null_check_abs_off = writer_new_off + null_check_local_off;
    assert!(
      null_check_abs_off < set_len_off,
      "DEFENSE-IN-DEPTH STRUCTURAL REGRESSION: `set_len(0)` must appear AFTER the \
       `ctx.is_null()` check so a NULL-ctx writer sentinel cannot bypass the guard \
       and trigger the destructive truncate. Current ordering has the null check at \
       byte {null_check_abs_off} and set_len at byte {set_len_off}.",
    );
  }

  /// **Defense-in-depth behavioral: writer-construction reached on
  /// happy path.** Pre-fills a file with 50 bytes, then calls
  /// `save_safetensors_to_file` with valid empty metadata and one tiny
  /// array. Asserts the call returns `Ok(())` AND that the file ends up
  /// bearing the safetensors header bytes (a valid `load_safetensors`
  /// round-trip), proving the writer-construction is still reached and
  /// the write still happens after the structural reorder. The
  /// structural test above is the primary guard; this behavioral one
  /// documents the happy path so a future refactor that BREAKS the
  /// write itself fails fast.
  #[test]
  fn save_safetensors_to_file_writer_construction_precedes_truncate() {
    let dir = fresh_dir("load1-fd-r4-writer-precedes-truncate");
    let path = dir.join("prefilled_50_bytes.safetensors");
    {
      let mut prefill = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
      std::io::Write::write_all(&mut prefill, &[0xA5_u8; 50]).unwrap();
      prefill.sync_all().unwrap();
    }
    // Reopen at the start so the existing 50-byte prefix would corrupt
    // the safetensors header without the rewind+truncate. On the happy
    // path the structural ordering still produces a valid round-trippable
    // file because writer-new succeeded and the truncate ran before the
    // write.
    let mut file = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .open(&path)
      .unwrap();
    let arr = Array::from_slice::<f32>(&[7.0_f32, 8.0, 9.0], &(3usize,)).unwrap();
    let empty_metadata: HashMap<String, String> = HashMap::new();
    crate::io::save_safetensors_to_file(&mut file, std::iter::once(("v", &arr)), &empty_metadata)
      .expect(
        "DEFENSE-IN-DEPTH REGRESSION: happy-path save_safetensors_to_file with \
         empty metadata must succeed (writer construction reached + write completed)",
      );
    file.sync_all().unwrap();
    drop(file);

    // Round-trip the file: a valid safetensors header (which
    // `load_safetensors` parses) is the proof that the truncate +
    // write both happened after the writer-new succeeded.
    let mut loaded = crate::io::load_safetensors(&path).unwrap();
    assert_eq!(
      loaded.len(),
      1,
      "DEFENSE-IN-DEPTH REGRESSION: round-trip must yield exactly one array \
       (saved one)"
    );
    let v = loaded.get_mut("v").unwrap().to_vec::<f32>().unwrap();
    assert_eq!(
      v,
      vec![7.0, 8.0, 9.0],
      "DEFENSE-IN-DEPTH REGRESSION: round-trip values must match the pre-save \
       array — a mismatch would indicate the write did not run or wrote stale \
       prefix bytes"
    );

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// **Documents the destructive contract for MLX-internal errors.**
  /// Pre-fills a file with 50 bytes, then calls
  /// `save_safetensors_to_file` with a zero-element `Array` that mlx-c
  /// rejects inside `mlx_save_safetensors_writer`. Asserts the call
  /// returns `Err` AND that the file is truncated to 0 bytes (NOT
  /// preserved).
  ///
  /// This is the EXPECTED behavior per the "Destructive mutation"
  /// section of `save_safetensors_to_file`'s doc comment — once the
  /// defense-in-depth ordering has cleared (Rust map builds, FFI map
  /// ctors, FFI writer ctor all `Ok`), the function commits to the
  /// destructive `seek(0)` + `set_len(0)` before invoking
  /// `mlx_save_safetensors_writer`. Any error from the writer (eval
  /// failure, MLX-internal array-set rejection, header-build failure,
  /// write-callback failure) leaves the file in a truncated /
  /// partially-mutated state. Callers that need atomic-replace
  /// semantics must use the fd-bound tempfile-staging pattern (open a
  /// same-directory `O_EXCL` `File`, pass it to
  /// `save_safetensors_to_file`, `sync_all`, then `rename` /
  /// `hard_link` to the final path — the open/write/fsync/drop
  /// fd-bound steps are exemplified by `save_model` above at lines
  /// 1359-1372). The fd-bound mitigation protects the WRITE PATH from
  /// `unlink + symlink` redirection; the publication step (`rename` /
  /// `hard_link` by `temp_path`) is pathname-based and still subject
  /// to directory-entry substitution any time after the `O_EXCL`
  /// create and before publication (not just after fsync). See the
  /// "Scope of this guarantee" caveat in
  /// `save_safetensors_to_file`'s doc comment (its `# Destructive
  /// mutation` doc section) for the publication-race options. Do NOT
  /// use the path-taking `save_safetensors_view` for atomic
  /// replacement: it reopens by name and reintroduces the write-path
  /// TOCTOU window LOAD-1 closed.
  ///
  /// A regression that "fixed" this by preserving the prefill on a
  /// writer-error would silently change the function's contract; this
  /// test catches such a regression by asserting the file IS
  /// destructively truncated on the writer-error path.
  #[test]
  fn save_safetensors_to_file_truncates_on_mlx_internal_error_zero_element_array() {
    let dir = fresh_dir("load1-fd-destructive-zero-elem");
    let path = dir.join("destructive_contract.safetensors");
    let original_bytes: &[u8] = &[0xC3_u8; 50];
    std::fs::write(&path, original_bytes).unwrap();
    let original_len = original_bytes.len() as u64;
    assert_eq!(
      std::fs::metadata(&path).unwrap().len(),
      original_len,
      "pre-call: file size must equal prefill length"
    );

    let mut file = std::fs::OpenOptions::new()
      .read(true)
      .write(true)
      .open(&path)
      .unwrap();
    // A zero-element array constructs successfully in Rust (see e.g.
    // `embeddings::colvision` tests), so all the up-front validation +
    // FFI ctor steps succeed (input maps build, writer-new returns
    // non-NULL). mlx-c's safetensors writer then rejects the
    // zero-element shape inside `mlx_save_safetensors_writer` — AFTER
    // the destructive `seek(0)` + `set_len(0)` have already run. This
    // exercises the "Partially mutated or zero-length" branch of the
    // documented Destructive mutation contract.
    let zero_arr = Array::from_slice::<f32>(&[], &(0usize,)).unwrap();
    let empty_metadata: HashMap<String, String> = HashMap::new();

    let result = crate::io::save_safetensors_to_file(
      &mut file,
      std::iter::once(("zero", &zero_arr)),
      &empty_metadata,
    );

    assert!(
      result.is_err(),
      "expected Err from save_safetensors_to_file on a zero-element array — mlx-c's \
       safetensors writer rejects this shape. If the writer started accepting \
       zero-element arrays, pick another MLX-internal-rejection trigger to keep \
       coverage of the destructive-contract path."
    );

    drop(file);
    let post_len = std::fs::metadata(&path).unwrap().len();
    // The destructive `seek(0)` + `set_len(0)` run BEFORE
    // `mlx_save_safetensors_writer` is invoked, so the file is
    // truncated to 0 bytes (or written as a partial safetensors
    // header if mlx-c emitted some bytes before rejecting the
    // zero-element shape). The strict assertion the documented
    // contract makes is "not byte-identical to the prefill" — the
    // file is partially mutated or zero-length. Asserting
    // `post_len < original_len` covers both cases (early reject ⇒
    // 0 bytes; mid-stream reject ⇒ some bytes of partial header).
    // The original prefill (50 bytes of 0xC3) is not a valid
    // safetensors prefix, so a `post_len == original_len` with
    // byte-identical contents is the regression we are guarding
    // against.
    assert!(
      post_len < original_len,
      "DESTRUCTIVE CONTRACT: save_safetensors_to_file MUST destructively mutate the \
       file on an MLX-internal writer error (the destructive seek+set_len runs \
       before mlx_save_safetensors_writer is invoked). The file size went from \
       {original_len} bytes to {post_len} bytes — if this assertion fires because \
       the file is BYTE-IDENTICAL to the prefill, the function silently regained a \
       byte-preservation contract it explicitly disclaims. See \
       save_safetensors_to_file's Destructive mutation doc section."
    );

    let _ = std::fs::remove_dir_all(&dir);
  }
}
