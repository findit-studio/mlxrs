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
//!   `utils.load_model`: `glob("model*.safetensors")`, `mx.load` each, then
//!   `weights.update(...)` to merge — plus a single-`*.gguf` fallback
//!   (mirroring mlx-lm's GGUF path). Quantized triples (`*.weight` /
//!   `*.scales` / `*.biases`) are kept **verbatim**.
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
  path::Path,
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
    .take(MAX_CONFIG_BYTES + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("cannot read {label} {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > MAX_CONFIG_BYTES {
    return Err(Error::Backend {
      message: format!(
        "{label} {} exceeds the {}-byte cap; refusing to read",
        path.display(),
        MAX_CONFIG_BYTES
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

/// Per-shard file name, mirroring `mlx_lm.utils.save_model`'s
/// `shard_file_format` (`utils.py:728-732`): a lone shard is the bare
/// `model.safetensors`; a multi-shard set uses the
/// `model-{:05d}-of-{:05d}.safetensors` HF convention (**1-based** shard
/// index, zero-padded to 5 digits).
fn shard_file_name(index_1based: usize, shards_count: usize) -> String {
  if shards_count > 1 {
    format!("model-{index_1based:05}-of-{shards_count:05}.safetensors")
  } else {
    "model.safetensors".to_string()
  }
}

/// Write a weight map as sharded `.safetensors` plus the
/// `model.safetensors.index.json` weight-map index into `save_path`,
/// mirroring `mlx_lm.utils.save_model` (`utils.py:714-771`).
///
/// Steps, in reference order:
///
/// 1. `save_path` is created (`mkdir -p`, `utils.py:723`).
/// 2. The weights are sharded via [`make_shards`] at [`MAX_FILE_SIZE_GB`]
///    (`utils.py:726`).
/// 3. `total_size` (the sum of every weight's `array_nbytes`) and
///    `total_parameters` ([`get_total_parameters`]) are computed for the
///    index `metadata` block (`utils.py:734-741`).
/// 4. Each shard is written with [`crate::io::save_safetensors_with_metadata`]
///    and the `{"format": "mlx"}` safetensors metadata mlx writes
///    (`utils.py:756`); the shard file name comes from `shard_file_name`.
/// 5. Any **stale** pre-existing `model*.safetensors` shard whose name is
///    not in the freshly written set is removed, so the destination ends
///    up holding *only* the new checkpoint. The reference does not do
///    this (`save_model` assumes a fresh directory), but mlxrs
///    [`load_weights`] reads **every** `model*.safetensors` in the
///    directory — not just the ones named in the index — so overwriting,
///    say, a 3-shard checkpoint with a single-`model.safetensors` one
///    would otherwise leave `model-00002-of-00003.safetensors` …
///    behind and silently resurrect stale tensors on the next load. This
///    cleanup makes `save_model` idempotent: a second save to the same
///    directory yields exactly the second checkpoint.
/// 6. The `weight_map` (`weight name → shard file name`) is assembled, then
///    **sorted by key** (`utils.py:762-764`), and the whole `index_data`
///    (`{ "metadata": { total_size, total_parameters }, "weight_map": … }`)
///    is written to `model.safetensors.index.json` with 4-space indentation
///    (`json.dump(..., indent=4)`, `utils.py:766-771`). The index file name
///    is constant, so the new index always overwrites any prior one.
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
/// A recoverable failure (directory create, a shard write, the index write,
/// an unrecognized weight dtype) is an [`Error::Backend`] naming the path.
pub fn save_model(
  save_path: &Path,
  weights: &Weights,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<()> {
  // 1. `save_path.mkdir(parents=True, exist_ok=True)` (`utils.py:723`).
  std::fs::create_dir_all(save_path).map_err(|e| Error::Backend {
    message: format!(
      "save_model: cannot create directory {}: {e}",
      save_path.display()
    ),
  })?;

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

  // 4. write each shard, recording `weight name → shard file name`. The
  //    `{"format": "mlx"}` safetensors metadata matches mlx-lm
  //    (`mx.save_safetensors(..., metadata={"format": "mlx"})`,
  //    `utils.py:756`). The shards are borrowed views, so they go through
  //    `save_safetensors_view` — no `Array` is cloned for the write.
  let mut shard_metadata: HashMap<String, String> = HashMap::with_capacity(1);
  shard_metadata.insert("format".to_string(), "mlx".to_string());
  // `weight_map` collected sorted-by-key so the written index is
  // deterministic (`utils.py:762-764` sorts it before the `json.dump`).
  let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
  // The names of the shards just written — every other `model*.safetensors`
  // in `save_path` is stale and removed in step 5.
  let mut written_shards: std::collections::HashSet<String> =
    std::collections::HashSet::with_capacity(shards_count);
  for (i, shard) in shards.iter().enumerate() {
    let shard_name = shard_file_name(i + 1, shards_count);
    let shard_path = save_path.join(&shard_name);
    crate::io::save_safetensors_view(
      &shard_path,
      shard.iter().map(|(&k, &v)| (k, v)),
      &shard_metadata,
    )?;
    for &weight_name in shard.keys() {
      weight_map.insert(weight_name.to_string(), shard_name.clone());
    }
    written_shards.insert(shard_name);
  }

  // 5. idempotency: drop any pre-existing `model*.safetensors` left over
  //    from an earlier checkpoint that the new shard set did not just
  //    rewrite. Without this, [`load_weights`] (which merges *every*
  //    `model*.safetensors`, not the index) would resurrect stale tensors
  //    when a multi-shard checkpoint is overwritten by a smaller one. The
  //    `model.safetensors.index.json` has a constant name and is
  //    overwritten unconditionally in step 6, so it needs no cleanup.
  //    `collect_sorted` applies the same `model*.safetensors` predicate
  //    `load_weights` uses, so exactly the files a later load would pick
  //    up are considered.
  let existing_shards = collect_sorted(save_path, |name| {
    name.starts_with("model") && name.ends_with(".safetensors")
  })?;
  for stale in &existing_shards {
    let is_stale = stale
      .file_name()
      .and_then(|n| n.to_str())
      .is_some_and(|n| !written_shards.contains(n));
    if is_stale {
      std::fs::remove_file(stale).map_err(|e| Error::Backend {
        message: format!(
          "save_model: cannot remove stale shard {}: {e}",
          stale.display()
        ),
      })?;
    }
  }

  // 6. assemble + write `model.safetensors.index.json` with `indent=4`
  //    (`utils.py:735-771`). `serde_json::Value` preserves the
  //    reference key order (`metadata` before `weight_map`); `weight_map`
  //    is a `BTreeMap`, so its keys serialize sorted.
  let index = serde_json::json!({
    "metadata": {
      "total_size": total_size,
      "total_parameters": total_parameters,
    },
    "weight_map": weight_map,
  });
  let index_path = save_path.join("model.safetensors.index.json");
  write_json_pretty(
    &index_path,
    &index,
    "save_model: model.safetensors.index.json",
  )?;
  Ok(())
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
/// `config` must be a JSON **object**; anything else (or invalid JSON) is an
/// [`Error::Backend`]. A write failure is an [`Error::Backend`] naming the
/// path.
pub fn save_config(config: &str, config_path: &Path) -> Result<()> {
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

  write_json_pretty(config_path, &sorted_value, "save_config: config.json")
}

/// Save a model — weights and config — into `dst_path`, mirroring the
/// local-directory core of `mlx_lm.utils.save` (`utils.py:925-950`).
///
/// This wires the save primitives together, in reference order:
///
/// 1. [`save_model`] writes the sharded `.safetensors` + the
///    `model.safetensors.index.json` index (`utils.py:942`).
/// 2. [`save_config`] writes `<dst_path>/config.json` (`utils.py:943`).
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
/// for an unquantized model). Either step's recoverable failure is the
/// [`Error::Backend`] that step produced.
pub fn save(
  dst_path: &Path,
  weights: &Weights,
  config: &str,
  quant: &crate::lm::quant::PerLayerQuantization,
) -> Result<()> {
  save_model(dst_path, weights, quant)?;
  save_config(config, &dst_path.join("config.json"))?;
  Ok(())
}

/// Write `value` to `path` as 4-space-indented JSON, mirroring Python's
/// `json.dump(value, f, indent=4)` byte-for-byte: a 4-space indent and the
/// `,` / `": "` separators `serde_json::ser::PrettyFormatter` already emits
/// (Python's `indent=N` uses the same — see [`crate::tokenizer::chat`]'s
/// `tojson` note). A trailing newline is **not** added — `json.dump` writes
/// none. The parent directory is assumed to exist (callers `mkdir -p` it).
fn write_json_pretty(path: &Path, value: &serde_json::Value, label: &str) -> Result<()> {
  let mut buf = Vec::new();
  let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
  let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
  serde::Serialize::serialize(value, &mut ser).map_err(|e| Error::Backend {
    message: format!("{label}: cannot serialize JSON: {e}"),
  })?;
  std::fs::write(path, &buf).map_err(|e| Error::Backend {
    message: format!("{label}: cannot write {}: {e}", path.display()),
  })?;
  Ok(())
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

  #[test]
  fn shard_file_name_single_vs_multi() {
    // 1 shard → bare `model.safetensors`.
    assert_eq!(shard_file_name(1, 1), "model.safetensors");
    // multi-shard → 1-based, 5-digit zero-padded HF convention.
    assert_eq!(shard_file_name(1, 3), "model-00001-of-00003.safetensors");
    assert_eq!(shard_file_name(3, 3), "model-00003-of-00003.safetensors");
  }

  // ─────────────────────── save_model round-trip ───────────────────────

  /// `save_model` writes a single `model.safetensors` (the 3 small weights
  /// fit one 5-GiB shard) plus a `model.safetensors.index.json`;
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

    // Exactly one shard file, named `model.safetensors`.
    assert!(dir.join("model.safetensors").is_file());
    assert!(dir.join("model.safetensors.index.json").is_file());

    // Weights round-trip byte-equal.
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
    assert_eq!(wm["model.a.weight"], "model.safetensors");
    assert_eq!(wm["model.b.weight"], "model.safetensors");
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
  /// hand-built 2-shard layout reloads via [`load_weights`].
  #[test]
  fn save_model_multi_shard_naming_and_index_reload() {
    let dir = fresh_dir("save-model-multi");
    // Two weights; write them as a 2-shard layout by hand using the same
    // primitives `save_model` uses, to exercise the multi-shard names.
    let w0 = Array::from_slice::<f32>(&[10.0], &(1usize,)).unwrap();
    let w1 = Array::from_slice::<f32>(&[20.0, 21.0], &(2usize,)).unwrap();
    let shards: Vec<Shard<'_>> = vec![BTreeMap::from([("w0", &w0)]), BTreeMap::from([("w1", &w1)])];
    let count = shards.len();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "mlx".to_string());
    for (i, s) in shards.iter().enumerate() {
      let name = shard_file_name(i + 1, count);
      assert_eq!(
        name,
        format!("model-{:05}-of-{:05}.safetensors", i + 1, count)
      );
      crate::io::save_safetensors_view(&dir.join(&name), s.iter().map(|(&k, &v)| (k, v)), &meta)
        .unwrap();
    }
    // Both shard files reload + merge via `load_weights` (`model*.safetensors`).
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

    // Weights side.
    assert!(dir.join("model.safetensors").is_file());
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

  // ─────────────────────── save_model idempotency ───────────────────────

  /// `save_model` is idempotent: overwriting a *multi-shard* checkpoint
  /// with a *smaller single-shard* one must leave the destination holding
  /// ONLY the new checkpoint — no stale `model-0000N-of-…safetensors`
  /// shards may survive. (mlxrs [`load_weights`] merges every
  /// `model*.safetensors` in the dir, so a stale shard would silently
  /// resurrect dead tensors.) A pre-existing 3-shard layout is hand-written
  /// (the same `save_safetensors_view` primitive `save_model` uses, since
  /// `save_model` hard-codes the 5-GiB cap and cannot be coerced into
  /// multi-shard from tiny weights), then `save_model` rewrites the dir
  /// with two small weights → one `model.safetensors`. After the save the
  /// dir must contain exactly `model.safetensors`, no `model-*-of-*` files,
  /// and `load_weights` must see only the two new keys.
  #[test]
  fn save_model_overwrite_multi_shard_with_single_is_idempotent() {
    let dir = fresh_dir("save-model-idempotent");

    // Stale 3-shard checkpoint, hand-written with the multi-shard names.
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
      let name = shard_file_name(i + 1, stale_count);
      crate::io::save_safetensors_view(&dir.join(&name), std::iter::once((*k, arr)), &meta)
        .unwrap();
      stale_map.insert((*k).to_string(), name);
    }
    // A stale index too — the new save must overwrite it (same name).
    write_json_pretty(
      &dir.join("model.safetensors.index.json"),
      &serde_json::json!({
        "metadata": { "total_size": 24, "total_parameters": 6 },
        "weight_map": stale_map,
      }),
      "test: stale index",
    )
    .unwrap();
    // Sanity: three multi-shard files are present before the overwrite.
    assert!(dir.join("model-00001-of-00003.safetensors").is_file());
    assert!(dir.join("model-00002-of-00003.safetensors").is_file());
    assert!(dir.join("model-00003-of-00003.safetensors").is_file());

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

    // Destination holds ONLY the new single shard — no stale shards left.
    assert!(dir.join("model.safetensors").is_file());
    assert!(
      !dir.join("model-00001-of-00003.safetensors").exists(),
      "stale shard 1 must be removed"
    );
    assert!(
      !dir.join("model-00002-of-00003.safetensors").exists(),
      "stale shard 2 must be removed"
    );
    assert!(
      !dir.join("model-00003-of-00003.safetensors").exists(),
      "stale shard 3 must be removed"
    );
    // Exactly one `model*.safetensors` survives.
    let survivors = collect_sorted(&dir, |n| {
      n.starts_with("model") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert_eq!(
      survivors.len(),
      1,
      "exactly one shard file after an idempotent overwrite"
    );

    // `load_weights` sees ONLY the new checkpoint's keys (no resurrected
    // stale tensors).
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

    // The index `weight_map` lists only the new keys, all → model.safetensors.
    let index_text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap();
    let index: serde_json::Value = serde_json::from_str(&index_text).unwrap();
    let wm = index["weight_map"].as_object().unwrap();
    assert_eq!(wm.len(), 2);
    assert_eq!(wm["fresh.x.weight"], "model.safetensors");
    assert_eq!(wm["fresh.y.weight"], "model.safetensors");

    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Re-saving the *same* checkpoint to a directory is a no-op on the file
  /// set (a plain idempotency check on the common single-shard path: the
  /// one shard is rewritten, nothing stale, nothing removed).
  #[test]
  fn save_model_resave_same_checkpoint_is_stable() {
    let dir = fresh_dir("save-model-resave");
    let mut w: Weights = HashMap::new();
    w.insert("m.w.weight".to_string(), f32_weight(4));

    save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();
    save_model(&dir, &w, &PerLayerQuantization::default()).unwrap();

    let survivors = collect_sorted(&dir, |n| {
      n.starts_with("model") && n.ends_with(".safetensors")
    })
    .unwrap();
    assert_eq!(survivors.len(), 1, "single shard, stable across re-saves");
    let loaded = load_weights(&dir).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!(loaded.contains_key("m.w.weight"));
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
}
