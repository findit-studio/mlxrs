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
//!   `utils.load_model`. It is a thin delegate to the feature-neutral
//!   [`crate::io::load_weights_from_dir`] (the discovery lives in the
//!   always-compiled `io` module so `embeddings` / `vlm` / `audio` reuse
//!   it), honoring `model.safetensors.index.json` (the HF
//!   safetensors-sharded authoritative manifest) as the SINGLE source of
//!   truth for which shards are part of the checkpoint when present, then
//!   falling back to a single `model.safetensors`, a legacy
//!   `weights.safetensors`, a single `*.gguf` (mlx-lm's GGUF path), or a
//!   single `*.npz`. Quantized triples
//!   (`*.weight` / `*.scales` / `*.biases`) are kept **verbatim**.
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
  error::{
    DurabilityWarningPayload, EmptyInputPayload, Error, FileIoPayload, FileOp,
    InvariantViolationPayload, KeyCollisionPayload, LayerKeyedPayload, OutOfRangePayload,
    ParsePayload, Result,
  },
};

/// Upper bound on a `config.json` we will read into memory (the shared
/// [`crate::io::read_bounded_config_file`] cap). Re-exported from the
/// feature-neutral `io` module — the *one* bound every config-JSON reader
/// (`crate::lm::load`, [`crate::lm::factory`], [`crate::lm::lora`],
/// [`crate::vlm::load`]) funnels through.
pub(crate) use crate::io::MAX_CONFIG_BYTES;
/// The config-read primitive this module itself funnels `config.json` /
/// `generation_config.json` through, re-exported from the feature-neutral
/// [`crate::io`] module (the discovery logic moved to `io`).
pub(crate) use crate::io::read_bounded_config_file;
/// Shared checkpoint-discovery + bounded-read primitives, re-exported from
/// the feature-neutral [`crate::io`] module so the long-standing
/// `crate::lm::load::{collect_sorted, path_is_file, read_bounded_text_file,
/// read_bounded_bytes_file}` call sites (in `crate::vlm`, `crate::audio`,
/// and this module's discovery tests) keep resolving unchanged after the
/// logic moved to `io`. `#[allow(unused_imports)]` because which of these
/// shims is exercised depends on the active feature set (a bare `lm` build
/// without `vlm`/`audio` and outside `#[cfg(test)]` uses none of them).
#[allow(unused_imports)]
pub(crate) use crate::io::{
  collect_sorted, path_is_file, read_bounded_bytes_file, read_bounded_text_file,
};

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
#[non_exhaustive]
pub struct Config {
  /// Architecture id (`config.json` `model_type`, e.g. `"qwen3"`).
  model_type: String,
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, derive_more::IsVariant)]
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
  /// Architecture id (`config.json` `model_type`, e.g. `"qwen3"`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Parse a [`Config`] from an in-memory `config.json` string.
  ///
  /// Mirrors `mlx_lm.utils.load_config` (`json.load(config.json)`) restricted
  /// to the typed subset. A serde failure (malformed JSON or a missing
  /// required key) maps to [`Error::Backend`] with the underlying cause —
  /// the codebase's config-parse error convention (twin of
  /// `embeddings::config`'s `serde_json::from_str(..).map_err(Backend)`).
  pub fn from_json(json: &str) -> Result<Config> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Config::from_json",
        "model config JSON",
        e,
      ))
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
/// **Thin delegate to [`crate::io::load_weights_from_dir`]** — the discovery
/// logic moved to the always-compiled, feature-neutral `io` module so every
/// consumer (`embeddings`, `vlm`, `audio`, and this LM loader) shares the one
/// sharding-aware multi-format loader. This wrapper preserves the historical
/// `crate::lm::load::load_weights(dir) -> Result<Weights>` signature for the
/// existing `lm` call sites and tests; [`Weights`] IS
/// `HashMap<String, Array>`, so the return forwards directly.
///
/// Resolution order (first match wins), see
/// [`crate::io::load_weights_from_dir`] for the full contract:
///
/// 1. **`model.safetensors.index.json`** (authoritative shard manifest;
///    requires the `serde_json` feature — always on under `lm`).
/// 2. **Single `model.safetensors`** (no index).
/// 3. **Legacy `weights.safetensors`**.
/// 4. **`*.gguf`** (`gguf` feature) → `crate::io::load_gguf`.
/// 5. **`*.npz`** (`npz` feature) → `crate::io::load_npz`.
///
/// No weights file → a typed [`Error::FileIo`] (mlx-lm's
/// `FileNotFoundError("No safetensors found in {model_path}")`). Keys are
/// returned **verbatim** (no remap/sanitize).
pub fn load_weights(dir: &Path) -> Result<Weights> {
  crate::io::load_weights_from_dir(dir)
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
    return Err(Error::FileIo(FileIoPayload::new(
      "load_config: model config",
      FileOp::Open,
      path,
      std::io::Error::from(std::io::ErrorKind::NotFound),
    )));
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
  crate::tokenizer::Tokenizer::from_path(dir, resolved_eos.as_deref())
    .map_err(|e| Error::LayerKeyed(LayerKeyedPayload::new(dir.display().to_string(), e)))
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
/// reproduced faithfully (this is a faithful port — it matches mlx-lm even
/// in the edge cases):
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
        let q = quant.quantization_for(path).ok_or_else(|| {
          Error::LayerKeyed(LayerKeyedPayload::new(
            path.to_string(),
            Error::InvariantViolation(InvariantViolationPayload::new(
              "get_total_parameters: quantized layer (has `.scales`)",
              "must have quantization params resolvable in the config",
            )),
          ))
        })?;
        if q.bits <= 0 {
          return Err(Error::LayerKeyed(LayerKeyedPayload::new(
            path.to_string(),
            Error::OutOfRange(OutOfRangePayload::new(
              "get_total_parameters: quantized layer bits",
              "must be > 0",
              q.bits.to_string(),
            )),
          )));
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
      let q = quant.quantization_for(path).ok_or_else(|| {
        Error::LayerKeyed(LayerKeyedPayload::new(
          path.to_string(),
          Error::InvariantViolation(InvariantViolationPayload::new(
            "get_total_parameters: quantized layer (has `.weight` + `.scales` + `.biases`)",
            "must have quantization params resolvable in the config",
          )),
        ))
      })?;
      match q.mode {
        // Affine zero-point buffer — metadata, skip (do not count).
        QuantMode::Affine => continue,
        // mxfp4 / mxfp8 / nvfp4 carry no `.biases`; a present one means
        // an invalid checkpoint — flag it, do not silently drop it.
        QuantMode::Mxfp4 | QuantMode::Mxfp8 | QuantMode::Nvfp4 => {
          return Err(Error::LayerKeyed(LayerKeyedPayload::new(
            key.clone(),
            Error::KeyCollision(KeyCollisionPayload::new(
              "get_total_parameters: layer is quantized with scale-only mode \
                (mxfp4 / mxfp8 / nvfp4) which has no `.biases` buffer; a present \
                `.biases` tensor signals an invalid checkpoint",
              key.clone(),
            )),
          )));
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
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "compute_bits_per_weight: model parameters",
    )));
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
/// the same-millisecond / same-microsecond collision a timestamp-only tag
/// would leave open. Combined with the PID (unique among live processes)
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
///   a timestamp-only tag would collide whenever two saves landed in the
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
/// weights+index visible against the OLD config.
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
/// This contract closes the hole where a post-index-commit `fsync_dir`
/// failure would propagate as `Err` and drop the staged config-tempfile
/// guard (deleting its tempfile via [`Drop`]), leaving NEW weights+index
/// visible against the OLD config.
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
  std::fs::create_dir_all(save_path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save_model: cannot create directory",
      FileOp::Create,
      save_path.to_path_buf(),
      e,
    ))
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
      &index_tmp,
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
        return Err(Error::ShardPathCollision(final_path.clone()));
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
        return Err(Error::FileIo(FileIoPayload::new(
          "save_model: cannot hard_link shard to final path",
          FileOp::Rename,
          final_path.clone(),
          e,
        )));
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
  fsync_dir(save_path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save_model: fsync parent directory",
      FileOp::Fsync,
      save_path.to_path_buf(),
      e,
    ))
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
    return Err(Error::FileIo(FileIoPayload::new(
      "save_model: cannot stat index final path before rename",
      FileOp::Stat,
      index_final.clone(),
      e,
    )));
  }
  if let Err(e) = std::fs::rename(index_tmp, index_final) {
    let _ = std::fs::remove_file(index_tmp);
    return Err(Error::FileIo(FileIoPayload::new(
      "save_model: cannot rename index",
      FileOp::Rename,
      index_final.clone(),
      e,
    )));
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
  //        [`Drop`]), leaving NEW weights+index visible against the OLD
  //        config. The caller surfaces it to its own caller via
  //        [`crate::Error::DurabilityWarning`].
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
/// **TOCTOU rationale.** Returning only the [`PathBuf`] and dropping the
/// open handle would leave every subsequent write to re-open `path` by
/// name. Between the `O_EXCL` create + that reopen, an
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
    .ok_or_else(|| {
      Error::FileIo(FileIoPayload::new(
        "save: destination has no file_name component",
        FileOp::Stat,
        final_path.to_path_buf(),
        std::io::Error::from(std::io::ErrorKind::InvalidInput),
      ))
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
        return Err(Error::FileIo(FileIoPayload::new(
          "save: create_new tempfile",
          FileOp::Create,
          candidate,
          e,
        )));
      }
    }
  }
  Err(Error::FileIo(FileIoPayload::new(
    "save: exhausted tempfile retries (the per-process gen_id collided with foreign \
      tempfile names MAX_RETRIES times — usually a hostile staging-dir race or a \
      filesystem refusing create_new)",
    FileOp::Create,
    parent.to_path_buf(),
    last_err.unwrap_or_else(|| std::io::Error::from(std::io::ErrorKind::AlreadyExists)),
  )))
}

/// fsync `path` so its bytes are durable on disk before it is renamed into
/// place — a delayed-allocation / NFS / quota writeback failure must surface
/// *here*, not after a not-yet-on-disk file has been renamed over a
/// previously-valid checkpoint. mlx-c does not fsync; reopen the path
/// read-only and `sync_all` it. Mirrors `cache_prompt::save_prompt_cache_atomic`.
///
/// `pub(crate)` so sibling modules ([`crate::lm::convert`]'s post-copy
/// durability step) can call the same well-tested helper rather
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
  fsync_path_inner(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save: fsync tempfile",
      FileOp::Fsync,
      path.to_path_buf(),
      e,
    ))
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
  fsync_open_file_for_path_inner(file, path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save: fsync tempfile",
      FileOp::Fsync,
      path.to_path_buf(),
      e,
    ))
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
      // open), so the "real failure" path that depends on the
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
  // shape so the post-copy durability tests can exercise the
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

  // Test hook — "real OS failure" injector. Unlike the
  // pre-existing injector above (which synthesizes a formatted
  // io::Error string that incidentally includes the path), this one
  // removes the target file then falls through to the natural
  // [`std::fs::File::open`] call so the test observes the AUTHENTIC
  // [`std::io::Error`] the OS produces — a context-free OS-level
  // message like `"No such file or directory (os error 2)"` with NO
  // path embedded. Used by `convert.rs`'s "real failure" test to
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
// [`arm_fsync_path_fault_with_kind`]). Used by the post-copy
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
  /// "Real failure" injector skip counter. When `Some(n)`, the
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
/// [`crate::lm::convert`]'s post-copy durability closure) can
/// drive the [`CopyOutcome::CommittedWithDurabilityWarning`] branch
/// through the public [`crate::lm::convert::convert`] entrypoint.
#[cfg(test)]
pub(crate) fn arm_fsync_path_fault(skip: usize) -> FsyncPathFaultGuard {
  arm_fsync_path_fault_with_kind(skip, std::io::ErrorKind::Other)
}

/// Variant of [`arm_fsync_path_fault`] that lets the caller pick the
/// injected [`std::io::ErrorKind`] (e.g.
/// [`std::io::ErrorKind::PermissionDenied`] /
/// [`std::io::ErrorKind::StorageFull`]). Used by the
/// kind-preservation tests so the post-copy file fsync warning's
/// `.kind()` can be asserted against a SPECIFIC non-`Other` kind —
/// proving the convert()-side aggregate preserves the kind end-to-end
/// (a `fsync_copied` closure that re-wrapped the injected io::Error via
/// `io::Error::other(message)` would collapse every kind to `Other`).
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

/// Arm the "real OS failure" injector: skip `skip` successful
/// `fsync_path_inner` calls, then on the (skip+1)-th call remove the
/// target file before the natural [`std::fs::File::open`] runs. The
/// resulting [`std::io::Error`] is the AUTHENTIC OS-level error (kind
/// [`std::io::ErrorKind::NotFound`], message like `"No such file or
/// directory (os error 2)"`) with NO path embedded — exactly the kind
/// of context-free failure the call-site wrap is designed to
/// catch. Returns a [`Drop`] guard that disarms the injector on scope
/// exit (so a test panic still leaves the thread clean).
///
/// `pub(crate)` (test-only) so [`crate::lm::convert`]'s
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
/// [`crate::lm::convert`]'s durability closure) can drive the same
/// post-commit durability path through the public [`save`] entrypoint.
#[cfg(test)]
pub(crate) fn arm_fsync_dir_fault(skip: usize) -> FsyncDirFaultGuard {
  arm_fsync_dir_fault_with_kind(skip, std::io::ErrorKind::Other)
}

/// Variant of [`arm_fsync_dir_fault`] that lets the caller pick the
/// injected [`std::io::ErrorKind`]. Used by the kind-preservation
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
/// directory-fsync step) can call the same well-tested helper
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
    CommitOutcome::CommittedWithDurabilityWarning(source) => Err(Error::DurabilityWarning(
      DurabilityWarningPayload::new(true, source),
    )),
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
  let value: serde_json::Value = serde_json::from_str(config)
    .map_err(|e| Error::Parse(ParsePayload::new("save_config: config", "JSON", e)))?;
  let serde_json::Value::Object(mut map) = value else {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "save_config: config JSON",
      "must be an object",
    )));
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
  let sorted_value = serde_json::to_value(sorted).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "save_config: cannot re-serialize sorted config",
      "JSON",
      e,
    ))
  })?;

  let (mut tmp_file, tmp_path) = open_excl_temp_shard(config_path)?;
  let staged = StagedConfig {
    tmp_path,
    cleanup_on_drop: true,
  };
  write_json_pretty(
    &mut tmp_file,
    &staged.tmp_path,
    &sorted_value,
    "save_config: config.json",
  )?;
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
    return Err(Error::FileIo(FileIoPayload::new(
      "save_config: cannot rename staged config tempfile over destination",
      FileOp::Rename,
      config_path.to_path_buf(),
      e,
    )));
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
///    the OLD config). The warning is accumulated for the final return.
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
  std::fs::create_dir_all(dst_path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save: cannot create destination directory",
      FileOp::Create,
      dst_path.to_path_buf(),
      e,
    ))
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
  //    the hole this entire shape closes). The warning is accumulated
  //    for the final return.
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
    return Err(Error::DurabilityWarning(DurabilityWarningPayload::new(
      true, source,
    )));
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
  path: &Path,
  value: &serde_json::Value,
  label: &'static str,
) -> Result<()> {
  use std::io::Write;
  let mut buf = Vec::new();
  let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
  let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
  serde::Serialize::serialize(value, &mut ser)
    .map_err(|e| Error::Parse(ParsePayload::new(label, "JSON serialize", e)))?;
  file.write_all(&buf).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      label,
      FileOp::Write,
      path.to_path_buf(),
      e,
    ))
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
fn write_json_pretty_to_path(
  path: &Path,
  value: &serde_json::Value,
  label: &'static str,
) -> Result<()> {
  let mut f = std::fs::File::create(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      label,
      FileOp::Create,
      path.to_path_buf(),
      e,
    ))
  })?;
  write_json_pretty(&mut f, path, value, label)
}

#[cfg(test)]
mod save_tests;
