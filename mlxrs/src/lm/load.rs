//! Architecture-agnostic model-load **support surface**, ported from
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
//!   `utils.load_model`: `glob("model*.safetensors")`, `mx.load` each, then
//!   `weights.update(...)` to merge — plus a single-`*.gguf` fallback
//!   (mirroring mlx-lm's GGUF path). Quantized triples (`*.weight` /
//!   `*.scales` / `*.biases`) are kept **verbatim**.
//! - [`load`] mirrors `utils.load` — wire `config.json` + weights + the #18
//!   [`Tokenizer`](crate::tokenizer::Tokenizer) into the parts a (per-usecase)
//!   architecture assembles itself.
//!
//! **Deliberately NOT ported** (per the project's no-model-arch scoping):
//! `utils.load_model`'s per-architecture `model_class(model_args)`
//! construction, `model.sanitize(weights)` key-remap, the `_quantize` /
//! `class_predicate` quantization *application* (it mutates a constructed
//! model), the legacy AWQ/bitnet transforms, `_download` (HuggingFace Hub —
//! this is local-path-only, no network), and `make_shards` / `save` /
//! `sharded_load`. [`load`] returns the raw `(Config, Weights, Tokenizer)`
//! triple; assembling and (de)quantizing a concrete model is the per-usecase
//! architecture's job. The [`Config::quantization`] field merely *carries*
//! `config["quantization"]` (mlx-lm `utils.py`'s
//! `config.get("quantization")`) so a later arch can apply it — `load` itself
//! never quantizes.
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

use std::{collections::HashMap, path::Path};

use crate::{
  array::Array,
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

/// Quantization parameters from a checkpoint's `config.json` `quantization`
/// block (mlx-lm `utils.py` `config["quantization"]`: `{ "group_size": int,
/// "bits": int, ... }`). Carried so a per-usecase architecture can apply
/// quantization itself; [`load`] never quantizes. Extra keys in the block
/// (e.g. `"mode"`) are ignored — only the two `mlx.core.quantize` always
/// needs are modeled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub struct Quantization {
  /// Elements per quantization group (`mlx.core.quantize` `group_size`).
  pub group_size: i32,
  /// Bits per weight (`mlx.core.quantize` `bits`).
  pub bits: i32,
}

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
/// weight-loading half of `mlx_lm.utils.load_model`.
///
/// Resolution order:
///
/// 1. **Sharded / single safetensors:** every `model*.safetensors` in `dir`
///    (mlx-lm `glob.glob(model_path / "model*.safetensors")`), iterated in
///    **sorted filename order** so a deterministic, reproducible merge —
///    `mx.load` each and `weights.update(...)` (later shard wins on a
///    duplicate key, which a well-formed shard set never produces). This
///    covers both `model.safetensors` and
///    `model-00001-of-000NN.safetensors` shard sets.
/// 2. **GGUF fallback:** if there is no `model*.safetensors`, a single
///    `*.gguf` in `dir` is loaded via [`crate::io::load_gguf`] (mlx-lm's
///    GGUF load path). Requires the `gguf` feature; without it a present
///    `*.gguf` is reported as unsupported.
///
/// No safetensors and no usable GGUF → [`Error::Backend`] (mlx-lm's
/// `FileNotFoundError("No safetensors found in {model_path}")`). Keys are
/// returned **verbatim** (no remap/sanitize — spec §7.2).
pub fn load_weights(dir: &Path) -> Result<Weights> {
  let mut shards = collect_sorted(dir, |name| {
    name.starts_with("model") && name.ends_with(".safetensors")
  })?;

  if !shards.is_empty() {
    // Deterministic merge in sorted filename order (mlx-lm globs then
    // `weights.update(...)`; sorting makes the dup-key tie-break — which a
    // valid shard set never hits — reproducible).
    shards.sort();
    let mut weights: Weights = HashMap::new();
    for shard in &shards {
      let part = crate::io::load_safetensors(shard)?;
      weights.extend(part);
    }
    return Ok(weights);
  }

  // No safetensors → try a single `*.gguf` (mlx-lm's GGUF load path).
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
      "no model weights found in {}: expected `model*.safetensors` (or a single `*.gguf`)",
      dir.display()
    ),
  })
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
  use std::io::Read;

  let path = dir.join("config.json");

  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
      .map_err(|e| Error::Backend {
        message: format!("cannot open model config {}: {e}", path.display()),
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(&path).map_err(|e| Error::Backend {
    message: format!("cannot open model config {}: {e}", path.display()),
  })?;

  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("cannot stat opened model config {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "model config {} is not a regular file; refusing to read",
        path.display()
      ),
    });
  }

  let mut bytes = Vec::new();
  file
    .take(MAX_CONFIG_BYTES + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("cannot read model config {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > MAX_CONFIG_BYTES {
    return Err(Error::Backend {
      message: format!(
        "model config {} exceeds the {}-byte cap; refusing to read",
        path.display(),
        MAX_CONFIG_BYTES
      ),
    });
  }

  let text = String::from_utf8(bytes).map_err(|e| Error::Backend {
    message: format!("model config {} is not valid UTF-8: {e}", path.display()),
  })?;
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
fn read_generation_eos(dir: &Path) -> Option<EosTokenId> {
  use std::io::Read;

  let path = dir.join("generation_config.json");

  #[cfg(unix)]
  let Ok(file) = ({
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
  }) else {
    return None;
  };
  #[cfg(not(unix))]
  let Ok(file) = std::fs::File::open(&path) else {
    return None;
  };

  match file.metadata() {
    Ok(m) if m.is_file() => {}
    _ => return None,
  }

  let mut bytes = Vec::new();
  if file
    .take(MAX_CONFIG_BYTES + 1)
    .read_to_end(&mut bytes)
    .is_err()
    || bytes.len() as u64 > MAX_CONFIG_BYTES
  {
    return None;
  }

  let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
    return None;
  };
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
  let resolved_eos = config.eos_token_id.clone().map(EosTokenId::into_ids);
  crate::tokenizer::Tokenizer::from_path(dir, resolved_eos.as_deref()).map_err(|e| Error::Backend {
    message: format!("cannot load tokenizer from {}: {e}", dir.display()),
  })
}
