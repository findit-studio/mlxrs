//! Local VLM **load factory** + a (`model_type`, `processor_class`) →
//! constructor registry pair, ported from the local-path slice of
//! [`mlx_vlm.utils`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py)
//! (`load` / `load_model` / `load_processor` / `load_image_processor` /
//! `load_config` / `get_model_and_args`) and `mlx-swift-lm`'s
//! [`VLMModelFactory`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/VLMModelFactory.swift)
//! (`VLMTypeRegistry` + `VLMProcessorTypeRegistry` + `VLMModelFactory._load`
//! + `BaseProcessorConfiguration` + `loadProcessorConfig`'s
//!   `preprocessor_config.json`-over-`processor_config.json` preference).
//!
//! This is the VLM analog of [`crate::lm::factory`] — same orchestration
//! shape (parse config ONCE → validate registries EARLY → select tokenizer
//! dir → load weights → load tokenizer → construct), reusing
//! [`crate::lm::load::load_config`] / [`crate::lm::load::load_weights`] /
//! [`crate::lm::load::load_tokenizer`] verbatim, and adding the two
//! VLM-specific concerns the LM loader does not have:
//!
//! - the **processor config** read (mlx-vlm `load_processor` /
//!   `load_image_processor`; mlx-swift-lm `loadProcessorConfig`), which
//!   reads `<dir>/preprocessor_config.json` **preferring it over**
//!   `<dir>/processor_config.json` (mirroring
//!   `VLMModelFactory.swift:438-454`) and decodes its `processor_class`
//!   field (mirroring `BaseProcessorConfiguration` at lines 45-51) to look
//!   up the processor constructor — exactly how the swift registry
//!   dispatches a per-model processor;
//! - the **processor type registry** ([`VlmProcessorTypeRegistry`]) — a
//!   `processor_class: String` → `ProcessorConstructor` table mirroring
//!   `VLMProcessorTypeRegistry.shared` at `VLMModelFactory.swift:104-135`.
//!   Per-model processors are **out of scope** (the project's no-model-arch
//!   rule), so the registry is the seam every per-usecase processor PR
//!   registers into.
//!
//! Per-model architectures (Qwen-VL / LLaVA / Pixtral / etc.) and per-model
//! processors are out of scope — this PR ships the seam, not the
//! architectures. The mock-driven test suite proves the end-to-end path
//! against a hand-traced mock model + mock processor.
//!
//! Conventions match [`crate::lm::factory`] (and the rest of the crate):
//! every fallible step returns [`Result`], recoverable failures
//! (missing/invalid config, no weights, unknown `model_type` /
//! `processor_class`, tokenizer load, processor-config parse) are
//! [`Error::Backend`] with a message naming the cause, borrows are
//! preferred over clones, and there is no implicit eval (the weight
//! `Array`s are handed to the constructor lazily, exactly as
//! [`crate::lm::load::load_weights`] returns them).
//!
//! [`Error::Backend`]: crate::Error::Backend

use std::{collections::HashMap, path::Path};

use crate::{
  error::{Error, Result},
  lm::{
    factory::{Identifier, ModelConfiguration},
    load::{self, EosTokenId, Quantization, Weights},
  },
  tokenizer::Tokenizer,
  vlm::{image::ImageProcessorConfig, model::Model as VlmModel},
};

/// The **minimal** VLM `config.json` subset the VLM load factory needs to
/// dispatch a checkpoint, mirroring `mlx-swift-lm`'s `BaseConfiguration`
/// (`MLXLMCommon/BaseConfiguration.swift`):
///
/// ```swift
/// public struct BaseConfiguration: Codable, Sendable {
///   public let modelType: String
///   public var eosTokenIds: IntOrIntArray?
///   var quantizationContainer: QuantizationContainer?
///   enum CodingKeys: String, CodingKey {
///     case modelType = "model_type"
///     case quantizationContainer = "quantization"
///     case eosTokenIds = "eos_token_id"
///   }
/// }
/// ```
///
/// Why this exists separately from [`crate::lm::load::Config`]: real VLM
/// checkpoints commonly nest the text-model fields (`hidden_size`,
/// `num_hidden_layers`, `num_attention_heads`, `head_dim`, `vocab_size`)
/// under `text_config` / `vision_config` and only carry `model_type` (and
/// optional `eos_token_id` / `quantization`) at the top — exactly mirrored
/// by `mlx_vlm.utils.load_config`'s `dict`-return + `load_model`'s
/// `config.setdefault("text_config", config.pop("llm_config", {}))` /
/// `config.setdefault("vision_config", {})` (`mlx_vlm/utils.py:239-240`).
/// Going through [`crate::lm::load::Config`] would *fatally* reject every
/// such checkpoint before any registered VLM constructor sees the raw JSON.
/// The per-model VLM constructor parses its arch-specific text-model /
/// vision-model fields off the verbatim
/// [`config_json`](LoadedVlmModel::config_json), exactly as each swift VLM's
/// per-model `Codable` init decodes the full config `Data` after the
/// `BaseConfiguration` is extracted (e.g. `Qwen25VL.ModelConfiguration.init`
/// at `Models/Qwen25VL.swift:1052`).
///
/// **Forward-compatible by design** (no `#[serde(deny_unknown_fields)]`):
/// every nested block / unknown top-level key is ignored at this layer and
/// flows to the constructor via the raw JSON — exactly as swift's
/// `BaseConfiguration` `Codable` does.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VlmBaseConfig {
  /// Architecture id (`config.json` `model_type`, e.g. `"qwen2_vl"`,
  /// `"mistral3"`). The [`VlmTypeRegistry`] dispatch key (after
  /// [`remap_vlm_model_type`]).
  pub model_type: String,
  /// `config.json` `eos_token_id` (a single id or a list). A *truthy*
  /// `generation_config.json` `eos_token_id` overrides it; the result is
  /// the tokenizer's COMPLETE eos set (REPLACES the tokenizer-config
  /// default — see [`load_vlm_base_config`]). `None` ⇒ fall back to the
  /// tokenizer's own `eos_token`. Optional so a VLM with no top-level
  /// `eos_token_id` (and a `text_config.eos_token_id`-only layout, which a
  /// per-model constructor would surface) still parses.
  #[serde(default)]
  pub eos_token_id: Option<EosTokenId>,
  /// Weight-quantization parameters (`config["quantization"]`), if the
  /// checkpoint carries them at the top level. Optional and forward-
  /// compatible: a VLM whose quantization sits under
  /// `text_config.quantization_config` (mlx-vlm's `load_model`
  /// translation at `mlx_vlm/utils.py:275-301`) parses with this `None`,
  /// and the per-model constructor extracts its own translation off the
  /// raw JSON if it needs to. Carried, not applied — same convention as
  /// [`crate::lm::load::Config::quantization`].
  #[serde(default)]
  pub quantization: Option<Quantization>,
}

impl VlmBaseConfig {
  /// Parse a [`VlmBaseConfig`] from an in-memory `config.json` string.
  /// Mirrors the swift `JSONDecoder().decode(BaseConfiguration.self, …)`
  /// in `VLMModelFactory._load` at `VLMModelFactory.swift:335`. A serde
  /// failure (malformed JSON or a missing `model_type`) maps to
  /// [`Error::Backend`] — the codebase config-parse convention.
  pub fn from_json(json: &str) -> Result<VlmBaseConfig> {
    serde_json::from_str(json).map_err(|e| Error::Backend {
      message: format!("invalid VLM base config JSON: {e}"),
    })
  }
}

/// Read `<dir>/config.json` **once** for a VLM checkpoint, returning both
/// the typed [`VlmBaseConfig`] and the verbatim JSON body it was parsed
/// from (the same bytes — so the per-model constructor's typed base config
/// and raw JSON can never come from two different on-disk versions).
///
/// VLM analog of [`crate::lm::load::load_config`]: same bounded
/// `O_NONBLOCK | O_CLOEXEC`, non-regular-reject, `MAX_CONFIG_BYTES`-capped
/// single read (via the same shared bounded-config-file primitive the LM
/// loader uses internally), and the SAME `generation_config.json`
/// `eos_token_id` override applied IN PLACE on the returned config (so a
/// tokenizer built from the resolved `eos_token_id` reflects the
/// generation-config override) — exactly mirroring
/// `mlx_vlm.utils.load_config` at `mlx_vlm/utils.py:506-515`, which has the
/// identical block.
///
/// The verbatim JSON body is returned alongside the typed value so a
/// per-model constructor can decode its model-specific (nested
/// `text_config` / `vision_config` / arch-specific) fields without
/// re-opening the file — exactly how `VLMModelFactory._load` hands each
/// model the same `configData: Data` at `VLMModelFactory.swift:343-344`.
/// Every recoverable failure (absent, non-regular, oversized, unreadable,
/// invalid JSON, missing `model_type`) is an [`Error::Backend`] naming
/// the offending path.
pub fn load_vlm_base_config(dir: &Path) -> Result<(VlmBaseConfig, String)> {
  let path = dir.join("config.json");
  let Some(text) = load::read_bounded_config_file(&path, "VLM base config")? else {
    return Err(Error::Backend {
      message: format!(
        "cannot open VLM base config {}: file not found",
        path.display()
      ),
    });
  };
  let mut config = VlmBaseConfig::from_json(&text)?;

  // If the top-level `eos_token_id` is absent, real VLM checkpoints
  // commonly nest it under `text_config` (the canonical mlx-vlm key) or
  // `llm_config` (the alias mlx-vlm promotes via
  // `config.setdefault("text_config", config.pop("llm_config", {}))` at
  // `mlx_vlm/utils.py:239`). The per-model dataclass that mlx-vlm/-swift
  // surfaces an `eos_token_id` off (see `mlx_vlm/utils.py:419`'s
  // `getattr(model.config, "eos_token_id", None)`) is not constructed in
  // this loader, so without this promotion the nested EOS would be
  // silently dropped and the tokenizer would fall back to its
  // `tokenizer_config` default — wrong generation stop. We promote the
  // nested value into [`VlmBaseConfig::eos_token_id`] BEFORE applying the
  // generation_config override (so the override still wins, exactly as it
  // does over the top-level value). Only the aliases the references
  // actually use (`text_config`, `llm_config`) are recognized; truthiness
  // rules match [`crate::lm::load::read_generation_eos`] (scalar must be
  // nonzero u32; list must be non-empty), shape-preserving (scalar →
  // `Single`, list → `Many`).
  if config.eos_token_id.is_none() {
    config.eos_token_id = read_nested_eos(&text);
  }

  // mlx-vlm `utils.load_config` (`mlx_vlm/utils.py:506-515`) and
  // mlx-swift-lm `VLMModelFactory._load` (`VLMModelFactory.swift:351-359`)
  // both overwrite the base config's `eos_token_id` with the
  // generation_config.json override IN PLACE — done here on the typed
  // base config so a tokenizer built from `config.eos_token_id` (via
  // `load_tokenizer_with_eos`) sees the same resolved set the LM path does
  // (`crate::lm::load::load_config` makes the same call through
  // `read_generation_eos`).
  if let Some(eos_override) = load::read_generation_eos(dir) {
    config.eos_token_id = Some(eos_override);
  }

  Ok((config, text))
}

/// Promote a nested `eos_token_id` out of the verbatim `config.json` JSON
/// when the top-level value is absent: try `text_config.eos_token_id`
/// first (mlx-vlm's canonical nested home for text-model fields), then
/// `llm_config.eos_token_id` (the alias mlx-vlm rewrites to `text_config`
/// via `config.setdefault("text_config", config.pop("llm_config", {}))`
/// at `mlx_vlm/utils.py:239`). Returns `None` if neither holds a *truthy*
/// value, matching [`crate::lm::load::read_generation_eos`]'s rules:
/// scalar must be a nonzero `u32`; list must be non-empty; any other
/// shape collapses to `None`. Shape is preserved (scalar → `Single`,
/// list → `Many`). A malformed `config.json` shouldn't reach here — it
/// would have failed [`VlmBaseConfig::from_json`] — but a re-parse
/// failure still collapses to `None` so this layer is strictly additive.
fn read_nested_eos(config_json: &str) -> Option<EosTokenId> {
  let v = serde_json::from_str::<serde_json::Value>(config_json).ok()?;
  // `text_config` first (canonical), then `llm_config` (alias) — the same
  // precedence mlx-vlm imposes with `setdefault(text_config, pop(llm_config))`.
  ["text_config", "llm_config"]
    .into_iter()
    .find_map(|key| v.get(key).and_then(|nested| nested.get("eos_token_id")))
    .and_then(parse_truthy_eos)
}

/// Truthy-parse an `eos_token_id` JSON value with the same semantics as
/// [`crate::lm::load::read_generation_eos`]'s match on the generation
/// config: scalar must be a nonzero `u32` (a scalar `0` is falsy → `None`);
/// list must be non-empty (and is preserved verbatim — a `[0, ..]` list
/// keeps the `0`); any other shape collapses to `None`. Pulled out so the
/// nested-EOS promotion and a future caller can share one rule.
fn parse_truthy_eos(value: &serde_json::Value) -> Option<EosTokenId> {
  match value {
    serde_json::Value::Number(n) => n
      .as_u64()
      .filter(|&x| x != 0)
      .and_then(|x| u32::try_from(x).ok())
      .map(EosTokenId::Single),
    serde_json::Value::Array(a) if !a.is_empty() => Some(EosTokenId::Many(
      a.iter()
        .filter_map(|e| e.as_u64().and_then(|x| u32::try_from(x).ok()))
        .collect(),
    )),
    _ => None,
  }
}

/// Re-export of [`crate::lm::factory::ModelConfiguration`] under the VLM
/// alias so the VLM factory matches the LM factory's public shape exactly
/// without duplicating the local-path-only `Identifier` + `tokenizer_source`
/// scaffolding. Mirrors how mlx-swift-lm's `VLMModelFactory` shares the
/// same `ModelConfiguration` type as `LLMModelFactory` (both go through
/// `ResolvedModelConfiguration`) — the source-location semantics are
/// identical across LM and VLM.
pub type VlmModelConfiguration = ModelConfiguration;

/// Re-export the [`Identifier`] enum for callers that match on the VLM
/// configuration's `id` field — same rationale as
/// [`VlmModelConfiguration`].
pub type VlmIdentifier = Identifier;

/// Architecture-id remapping, mirroring `mlx_vlm.utils.MODEL_REMAPPING`
/// (lines 30-46 of `mlx_vlm/utils.py`): some VLM checkpoints declare a
/// `model_type` that is an alias for another architecture's
/// implementation (e.g. `"lfm2-vl"` is served by `"lfm2_vl"`).
/// [`remap_vlm_model_type`] applies this before a [`VlmTypeRegistry`]
/// lookup so a registry only needs to register the *canonical* id.
///
/// Kept verbatim from `mlx_vlm.utils` (the authoritative spec) so a
/// checkpoint that loads in mlx-vlm dispatches to the same constructor
/// here. Sorted by key for a deterministic, reviewable table. This is
/// the VLM-specific remap table; the LM table at
/// [`crate::lm::factory::remap_model_type`] is independent (and an LM
/// alias like `"mistral" → "llama"` does NOT apply to VLM checkpoints).
const VLM_MODEL_REMAPPING: &[(&str, &str)] = &[
  ("bunny-llama", "llava_bunny"),
  ("cohere2_vision", "aya_vision"),
  ("falcon-perception", "falcon_perception"),
  ("granite-vision", "granite_vision"),
  ("granite4-vision", "granite4_vision"),
  ("granite4_vision", "granite4_vision"),
  ("jvlm", "jina_vlm"),
  ("lfm2-vl", "lfm2_vl"),
  ("llava-qwen2", "llava_bunny"),
  ("llava_qwen2", "fastvlm"),
  ("nemotronh_nano_omni_reasoning_v3", "nemotron_h_nano_omni"),
  ("phi4-siglip", "phi4_siglip"),
  ("rf-detr", "rfdetr"),
  ("sam3.1_video", "sam3_1"),
  ("sam3_video", "sam3"),
];

/// Canonicalize a VLM checkpoint's `model_type` via the
/// `VLM_MODEL_REMAPPING` table, mirroring `mlx_vlm.utils.get_model_and_args`'s
/// `model_type = MODEL_REMAPPING.get(model_type, model_type)` (lines
/// 115-117). An id with no alias is returned unchanged.
pub fn remap_vlm_model_type(model_type: &str) -> &str {
  VLM_MODEL_REMAPPING
    .iter()
    .find_map(|&(from, to)| (from == model_type).then_some(to))
    .unwrap_or(model_type)
}

/// Per-`model_type` processor override, mirroring
/// `VLMModelFactory.swift:399-403`'s `processorTypeOverrides` map:
/// some checkpoints declare a `processor_class` in their
/// `(pre)processor_config.json` that is wrong for the model
/// architecture and must be overridden — currently only Mistral3
/// models, which ship `"PixtralProcessor"` but need `"Mistral3Processor"`
/// to handle spatial merging correctly. Returns the override class name
/// for `model_type` (already canonicalized via [`remap_vlm_model_type`]),
/// or `None` if no override applies.
fn processor_class_override(model_type: &str) -> Option<&'static str> {
  match model_type {
    "mistral3" => Some("Mistral3Processor"),
    _ => None,
  }
}

/// The raw `processor_class` field of a VLM's processor config,
/// mirroring mlx-swift-lm's `BaseProcessorConfiguration` at
/// `VLMModelFactory.swift:45-51`:
/// ```swift
/// public struct BaseProcessorConfiguration: Codable, Sendable {
///     public let processorClass: String
///     enum CodingKeys: String, CodingKey {
///         case processorClass = "processor_class"
///     }
/// }
/// ```
///
/// Read from `<dir>/preprocessor_config.json` (preferred) or
/// `<dir>/processor_config.json` (fallback) by [`load_processor_config`].
/// The processor-config JSON is otherwise opaque to this layer; the
/// processor constructor receives the verbatim JSON body so a per-model
/// processor can decode its own model-specific fields (mirroring how
/// `BaseProcessorConfiguration` is JUST the registry-lookup key and the
/// per-model `Codable` init reads the rest of the file).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ProcessorConfig {
  /// The processor class id (the registry lookup key) — e.g.
  /// `"PixtralProcessor"`, `"Qwen2VLProcessor"`. Required.
  pub processor_class: String,
}

/// A **tolerant** parse of just the registry-dispatch field
/// (`processor_class`) off either `preprocessor_config.json` or
/// `processor_config.json`. Tolerant because a real HF VLM dir's
/// `preprocessor_config.json` is the *image-preprocessor* file — it
/// commonly carries only `image_mean` / `image_std` / `crop_size` etc. and
/// has NO `processor_class` field at all. A strict
/// `serde_json::from_str::<ProcessorConfig>` on such a file would error
/// even though the dispatch metadata is sitting one file over in
/// `processor_config.json` (mlx-vlm's `AutoProcessor` config), so we
/// instead read this `Option<String>` view and let
/// [`load_processor_config`] orchestrate the across-files fallback for the
/// missing dispatch key. Forward-compatible by design — every other
/// processor-config key (image-preprocessor metadata, model-specific
/// fields) flows opaquely to the constructor via the raw JSON body.
#[derive(Debug, Clone, serde::Deserialize)]
struct ProcessorClassOnly {
  #[serde(default)]
  processor_class: Option<String>,
}

/// Read the processor config from `dir`, preferring
/// `preprocessor_config.json` over `processor_config.json` (mirroring
/// mlx-swift-lm's `loadProcessorConfig` at `VLMModelFactory.swift:438-454`
/// — that helper checks for `preprocessor_config.json` first, falls back
/// to `processor_config.json`, then decodes `BaseProcessorConfiguration`).
///
/// Returns the parsed [`ProcessorConfig`] (the registry-lookup key) plus
/// the verbatim JSON body of **each** processor-config file that was
/// present (`preprocessor_config.json` and/or `processor_config.json`),
/// keyed by file identity — the same single-read pattern
/// [`crate::lm::load::load_config`] uses for `config.json`, so a
/// processor constructor consuming both the typed key and the raw JSON
/// (for model-specific fields outside the typed subset, mirroring the
/// swift per-processor `Codable` init that decodes the full
/// `processorConfigData`) can never get them from two different on-disk
/// versions of a file. Carrying *both* bodies (rather than only the one
/// the dispatch class was extracted from) means a per-model processor
/// that needs image-preprocessor metadata AND `processor_config.json`
/// processor-level fields never has to re-open a file. Also returns the
/// source filename (one of `"preprocessor_config.json"` /
/// `"processor_config.json"`) of the file the dispatch class + primary
/// image-preprocessor metadata came from, so error messages and the
/// [`LoadedProcessor`] hand-off can name the file the constructor saw.
///
/// **Processor DISPATCH vs IMAGE-preprocessor metadata.** A real HF VLM
/// directory's `preprocessor_config.json` is the *image-preprocessor*
/// file (`image_mean` / `image_std` / `crop_size` / etc.) and commonly
/// has NO `processor_class` field — the dispatch metadata sits in a
/// separate `processor_config.json` (the `AutoProcessor` combined
/// config). To support both layouts the resolution order is:
///
/// 1. If `preprocessor_config.json` is **absent**: fall back entirely to
///    `processor_config.json` — strict-parse it for `processor_class` and
///    use its body as the constructor JSON. Returns
///    `(class, None, Some(processor_body), "processor_config.json")`.
/// 2. If `preprocessor_config.json` is **present** and tolerant-parses to
///    a `processor_class`: use that class. Returns
///    `(class, Some(preprocessor_body), <Some processor_body if the file
///    exists, else None>, "preprocessor_config.json")` — the constructor
///    gets the preprocessor body (the image-preprocessor metadata it
///    expects).
/// 3. If `preprocessor_config.json` is **present** but has NO
///    `processor_class` (the image-preprocessor-only layout): read
///    `processor_config.json` for `processor_class` (dispatch). Returns
///    `(class_from_proc_config, Some(preprocessor_body),
///    Some(processor_body), "preprocessor_config.json")` — dispatch
///    metadata and image-preprocessor metadata can come from different
///    files, exactly as real HF VLM checkpoints ship, and **both** file
///    bodies are carried so a per-model processor needing image-
///    preprocessor metadata AND `processor_config.json` processor-level
///    fields reaches both without re-opening either file (the
///    TOCTOU/config-divergence this factory exists to avoid).
///
/// In every case the two `Option<String>` slots are keyed by file
/// identity — slot 2 is `preprocessor_config.json`'s body iff that file
/// is present, slot 3 is `processor_config.json`'s body iff that file is
/// present — so neither already-performed read is discarded.
///
/// The read is bounded by the same `MAX_CONFIG_BYTES` cap
/// [`crate::lm::load::load_config`] uses for `config.json` and shares the
/// same TOCTOU-closed `O_NONBLOCK`-on-unix open (a planted FIFO is
/// rejected immediately, an oversized file is rejected before unbounded
/// allocation). Every failure path (both files absent or both missing
/// `processor_class`, non-regular, oversized, unreadable, invalid JSON,
/// missing `processor_class`) is a recoverable [`Error::Backend`] naming
/// the offending path(s).
///
/// The "single bounded read" contract holds per file: each of
/// `preprocessor_config.json` / `processor_config.json` is read at most
/// once. Whenever `preprocessor_config.json` is present (cases 2 and 3)
/// `processor_config.json` is also bounded-read once if it exists — for
/// the dispatch class (case 3) or purely to carry its processor-level
/// body (case 2) — and when it *is* opened that one body is carried out
/// rather than discarded. An absent `processor_config.json` is the
/// `ENOENT` "no body" signal, leaving its slot `None`.
pub fn load_processor_config(
  dir: &Path,
) -> Result<(
  ProcessorConfig,
  Option<String>,
  Option<String>,
  &'static str,
)> {
  // Preference order matches swift `loadProcessorConfig`:
  // `preprocessor_config.json` first, then `processor_config.json`.
  // (Python `load_image_processor` reads `config.json` not these; its
  // per-model `ImageProcessor` decides what to look at. The swift behavior
  // is the cross-model convention we follow because it matches what HF
  // VLM checkpoints actually ship: `preprocessor_config.json` is the HF-
  // standard image-processor config name, `processor_config.json` is the
  // newer combined `AutoProcessor` config; either can be present.)
  const PREFERRED: &str = "preprocessor_config.json";
  const FALLBACK: &str = "processor_config.json";

  let preferred_path = dir.join(PREFERRED);
  let fallback_path = dir.join(FALLBACK);

  // (1) Try `preprocessor_config.json` first. The TOCTOU-closed
  // `O_NONBLOCK`-on-unix open inside `read_bounded_config_file` accepts
  // `ENOENT` as the "absent" signal that falls through to the
  // `processor_config.json`-only path; a NON-`ENOENT` IO failure (oversized,
  // non-regular, planted FIFO, …) is still a hard error here, faithful to
  // the previous behavior.
  let preferred_body = load::read_bounded_config_file(&preferred_path, "processor config")?;

  if let Some(body) = preferred_body {
    // Tolerant parse — image-preprocessor-only files (no `processor_class`)
    // are NOT a parse error at this layer; that's the case Fix 2 unblocks.
    // A truly malformed (non-JSON) preferred file is still an error, since
    // the constructor would also choke on it.
    let parsed: ProcessorClassOnly = serde_json::from_str(&body).map_err(|e| Error::Backend {
      message: format!(
        "invalid processor config JSON in {}: {e} (expected an OBJECT, optionally with a \
         `processor_class` string field)",
        preferred_path.display()
      ),
    })?;

    if let Some(processor_class) = parsed.processor_class {
      // (2) Preferred file has `processor_class` — that's the dispatch
      // class (precedence stays with `preprocessor_config.json`; the
      // bounded read of `processor_config.json` below NEVER overrides
      // it). But `processor_config.json` may ALSO exist and carry
      // processor-level fields a per-model processor needs *alongside*
      // the image-preprocessor metadata in `preprocessor_config.json`.
      // Bounded-read it through the SAME `read_bounded_config_file`
      // helper (each file still read at most once) so its body reaches
      // the constructor instead of forcing a re-open (the TOCTOU/
      // config-divergence this factory exists to prevent). If
      // `processor_config.json` is absent, `ENOENT` → `None` and the
      // slot stays `None` exactly as before.
      let fallback_body = load::read_bounded_config_file(&fallback_path, "processor config")?;
      return Ok((
        ProcessorConfig { processor_class },
        Some(body),
        fallback_body,
        PREFERRED,
      ));
    }

    // (3) Preferred file is image-preprocessor-only — keep its body for
    // the constructor (that's the image-preprocessor metadata the
    // processor expects), but read `processor_class` from
    // `processor_config.json` for dispatch.
    let Some(fallback_body) = load::read_bounded_config_file(&fallback_path, "processor config")?
    else {
      return Err(Error::Backend {
        message: format!(
          "{PREFERRED} in {} has no `processor_class` field and no {FALLBACK} present to fall back \
           to for the dispatch class",
          dir.display()
        ),
      });
    };
    let fallback_parsed: ProcessorClassOnly =
      serde_json::from_str(&fallback_body).map_err(|e| Error::Backend {
        message: format!(
          "invalid processor config JSON in {}: {e} (expected an OBJECT, optionally with a \
           `processor_class` string field)",
          fallback_path.display()
        ),
      })?;
    let Some(processor_class) = fallback_parsed.processor_class else {
      return Err(Error::Backend {
        message: format!(
          "neither {PREFERRED} nor {FALLBACK} in {} has a `processor_class` field for the \
           dispatch class",
          dir.display()
        ),
      });
    };
    // Carry BOTH file bodies: the constructor's primary image-
    // preprocessor metadata is the PREFERRED body, and the dispatch
    // class came from `processor_config.json` — whose body may ALSO
    // carry processor-level fields a per-model processor needs. That
    // read already happened (`fallback_body`); thread it through rather
    // than discard it, so the processor never has to re-open the file
    // (the TOCTOU/config-divergence this factory exists to avoid).
    // Filename still reflects the PREFERRED body, the source the
    // constructor decodes its primary metadata off.
    return Ok((
      ProcessorConfig { processor_class },
      Some(body),
      Some(fallback_body),
      PREFERRED,
    ));
  }

  // (4) `preprocessor_config.json` is absent — fall back entirely to
  // `processor_config.json` (strict parse for `processor_class`, use its
  // body as the constructor JSON).
  let Some(body) = load::read_bounded_config_file(&fallback_path, "processor config")? else {
    return Err(Error::Backend {
      message: format!(
        "no processor config found in {}: expected {PREFERRED} (preferred) or {FALLBACK}",
        dir.display()
      ),
    });
  };
  let parsed: ProcessorClassOnly = serde_json::from_str(&body).map_err(|e| Error::Backend {
    message: format!(
      "invalid processor config JSON in {}: {e} (expected an OBJECT, optionally with a \
       `processor_class` string field)",
      fallback_path.display()
    ),
  })?;
  let Some(processor_class) = parsed.processor_class else {
    return Err(Error::Backend {
      message: format!(
        "{FALLBACK} in {} has no `processor_class` field for the dispatch class \
         (and no {PREFERRED} present to fall back from)",
        dir.display()
      ),
    });
  };
  // No `preprocessor_config.json` → its body slot is `None`; the
  // constructor's metadata is entirely the `processor_config.json` body.
  Ok((
    ProcessorConfig { processor_class },
    None,
    Some(body),
    FALLBACK,
  ))
}

/// Everything [`load()`] resolved from a VLM model directory's *model*
/// inputs, handed to a [`VlmModelConstructor`] so it can assemble (and,
/// if [`VlmBaseConfig::quantization`] is set, quantize) a concrete VLM
/// architecture without re-reading the directory.
///
/// Borrowing — the constructor gets `&LoadedVlmModel`; it reads the typed
/// [`VlmBaseConfig`] (`model_type` / `eos_token_id` / `quantization`) and,
/// for everything else (nested `text_config` / `vision_config` /
/// arch-specific fields), the verbatim [`config_json`](Self::config_json)
/// text — the analogue of mlx-swift-lm passing the raw `config.json` `Data`
/// to each model's `Codable` init at `VLMModelFactory.swift:341-348` — and
/// takes the weight [`Array`](crate::array::Array)s it needs out of
/// [`weights`](Self::weights) **by reference** (no implicit eval; mlx
/// `Array` is a cheap refcounted handle, so an arch clones only the
/// handles it keeps). Same shape as [`crate::lm::factory::LoadedModel`],
/// but the typed config is the VLM-minimal [`VlmBaseConfig`] (not the
/// LM-required [`crate::lm::load::Config`]) — real VLMs nest the
/// text-model fields under `text_config`, so requiring them at the top
/// level would reject every real checkpoint before the per-model
/// constructor saw the raw JSON.
#[non_exhaustive]
pub struct LoadedVlmModel {
  /// The typed VLM base config (mlx-swift-lm's `BaseConfiguration`), with
  /// the generation-config eos override already applied (see
  /// [`load_vlm_base_config`]). Only the registry-dispatch + tokenizer
  /// fields are typed here; everything model-specific (nested
  /// `text_config` / `vision_config` / arch-specific keys) is read off
  /// [`config_json`](Self::config_json) by the per-model constructor.
  pub config: VlmBaseConfig,
  /// The verbatim `config.json` body, for every key outside the typed
  /// [`VlmBaseConfig`] subset — i.e. the nested `text_config` /
  /// `vision_config` / arch-specific fields each VLM constructor decodes
  /// itself, the analogue of mlx-swift-lm handing each model's `Codable`
  /// init the raw config `Data` at `VLMModelFactory.swift:343-344`. Always
  /// the bytes the typed [`config`](Self::config) was parsed from.
  pub config_json: String,
  /// The merged, name → [`Array`](crate::array::Array) weight map
  /// (mlx-vlm's `weights` dict). Keys are verbatim — the constructor
  /// applies any `sanitize`/remap itself, exactly as
  /// [`crate::lm::load::load_weights`] documents.
  pub weights: Weights,
}

/// Everything [`load()`] resolved for the *processor* side, handed to a
/// [`ProcessorConstructor`] so it can assemble a concrete VLM processor
/// (image processor + tokenizer pairing) without re-reading the
/// directory.
///
/// Mirrors mlx-swift-lm's `processorRegistry.createModel(configuration:
/// processorType:tokenizer:)` call shape at
/// `VLMModelFactory.swift:405-407`: the constructor receives the
/// verbatim processor-config JSON (for its per-model `Codable` init) AND
/// the already-built [`Tokenizer`] (so the processor can splice the
/// tokenizer's special-token ids into its preprocessing). Because a real
/// HF VLM checkpoint can ship the image-preprocessor metadata and the
/// `AutoProcessor` processor-level metadata in **two separate files**
/// ([`preprocessor_config_json`](Self::preprocessor_config_json) /
/// [`processor_config_json`](Self::processor_config_json)), both bodies
/// that were on disk are carried — a per-model processor needing fields
/// from either file reaches them without re-opening anything. The
/// [`config`](Self::config) is the VLM base config (`model_type` /
/// `eos_token_id` / `quantization`); a processor that needs model-specific
/// fields beyond those reads them off the verbatim model `config.json`
/// carried here as [`config_json`](Self::config_json) — the SAME
/// single-read body the *model* constructor received as
/// [`LoadedVlmModel::config_json`], NOT a re-read — so the processor and
/// model share one TOCTOU-consistent config view. The swift
/// processor-construction signature likewise receives the same
/// `BaseConfiguration` + raw config `Data` + `Tokenizer` triple. The
/// [`processor_class`](Self::processor_class) is the registry key the
/// constructor was dispatched on (after any
/// `processor_class_override`); the
/// [`processor_config_filename`](Self::processor_config_filename) names
/// the file the dispatch class + primary image-preprocessor metadata
/// came from (one of `"preprocessor_config.json"` /
/// `"processor_config.json"`) for diagnostic / round-trip purposes.
///
/// The trait the constructor returns is intentionally an opaque
/// `Box<dyn ProcessorTrait>` ([`Processor`]) rather than the concrete
/// [`ImageProcessorConfig`] — per-model processors carry per-model state
/// beyond just the ImageNet pipeline (custom crop modes, grid-aware
/// patchifiers, tokenizer-aware multimodal chat templates, etc.) and the
/// trait surfaces only the cross-model entry points
/// ([`Processor::image_processor_config`] for the
/// [`crate::vlm::image::preprocess`] pipeline). Per-model concrete impls
/// are out of scope and are added per-usecase, mirroring the
/// no-per-model-arch rule.
#[non_exhaustive]
pub struct LoadedProcessor<'a> {
  /// The typed VLM base config — same instance the
  /// [`VlmModelConstructor`] received (the model and processor share the
  /// SAME parsed config, mirroring `VLMModelFactory.swift:333-339`'s
  /// single `baseConfig` decode shared by both creator calls).
  pub config: &'a VlmBaseConfig,
  /// The verbatim model `config.json` body — the SAME single-read body
  /// the [`VlmModelConstructor`] received as
  /// [`LoadedVlmModel::config_json`] (NOT a re-read). A concrete
  /// processor whose downcast-only methods need arch fields that live
  /// only in `config.json` (e.g. a `hidden_size` / `image_token_index`
  /// not duplicated into the processor configs) reads them off this
  /// string, reusing the loader's single TOCTOU-consistent read instead
  /// of re-opening the file. The typed [`config`](Self::config) exposes
  /// only the registry-dispatch + tokenizer subset of these bytes;
  /// everything model-specific is decoded off this verbatim body, the
  /// analogue of mlx-swift-lm handing the raw config `Data` to both the
  /// model and processor `Codable` inits at `VLMModelFactory.swift:
  /// 343-344` / `405-407`. `config.json` is required for a model, so
  /// this is always present (`&str`, not `Option`).
  pub config_json: &'a str,
  /// The registry key this processor was looked up under (after any
  /// `processor_class_override`) — useful for diagnostics and for a
  /// constructor that wants to assert it was dispatched correctly.
  pub processor_class: &'a str,
  /// The verbatim body of `preprocessor_config.json` — the HF-standard
  /// *image-preprocessor* config (`image_mean` / `image_std` /
  /// `crop_size` / `patch_size` / per-channel overrides). `Some` iff the
  /// directory contained that file; `None` for the `processor_config.
  /// json`-only layout. The per-model constructor decodes its own
  /// image-preprocessor fields off this string.
  pub preprocessor_config_json: Option<&'a str>,
  /// The verbatim body of `processor_config.json` — the newer combined
  /// `AutoProcessor` config (carries `processor_class` plus
  /// processor-level fields like `image_seq_len`, chat-template knobs,
  /// …). `Some` iff the directory contained that file: the
  /// `processor_config.json`-only layout, the split layout where
  /// `preprocessor_config.json` lacked `processor_class` so this file
  /// supplied the dispatch class, AND the common layout where
  /// `preprocessor_config.json` carried `processor_class` but
  /// `processor_config.json` also exists with extra processor-level
  /// fields. A per-model processor needing BOTH image-preprocessor
  /// metadata AND processor-level metadata reads this alongside
  /// [`preprocessor_config_json`](Self::preprocessor_config_json) — neither
  /// body is discarded, so no file is re-opened. `None` only when
  /// `processor_config.json` is genuinely absent from the directory
  /// (see [`load_processor_config`]).
  pub processor_config_json: Option<&'a str>,
  /// Name of the file the dispatch class + primary image-preprocessor
  /// metadata came from (one of `"preprocessor_config.json"` /
  /// `"processor_config.json"`).
  pub processor_config_filename: &'static str,
  /// The fully-built [`Tokenizer`] — mlx-swift-lm passes the tokenizer
  /// into the processor constructor at `VLMModelFactory.swift:405-407`
  /// so a processor like Qwen2VLProcessor can splice the tokenizer's
  /// `<|image_pad|>` / `<|video_pad|>` / `<|vision_start|>` ids into its
  /// multimodal prompt assembly. By reference so the processor doesn't
  /// take ownership.
  pub tokenizer: &'a Tokenizer,
}

/// A registered VLM model constructor: assemble a [`VlmModel`] from the
/// already-resolved [`LoadedVlmModel`] (parsed config + raw config JSON +
/// weights).
///
/// Mirrors mlx-swift-lm's `VLMTypeRegistry` creator
/// `(Data) throws -> LanguageModel` at `VLMModelFactory.swift:80-102` —
/// but receives the *already-loaded* weights too (so a per-usecase
/// architecture never re-globs/re-reads the directory) and returns a
/// [`Result`] (Rust's `throws`). `Send + Sync` so a registry can be
/// shared across threads (e.g. a `static` shared registry, as
/// mlx-swift-lm's `VLMTypeRegistry.shared` is). The constructor itself
/// does **no** I/O; the directory was already read by [`load()`].
pub type VlmModelConstructor =
  Box<dyn Fn(&LoadedVlmModel) -> Result<Box<dyn VlmModel>> + Send + Sync + 'static>;

/// A `model_type: String` → [`VlmModelConstructor`] table, the VLM load
/// factory's architecture **extension point**.
///
/// Mirrors mlx-swift-lm's `VLMTypeRegistry.shared` at
/// `VLMModelFactory.swift:80-102` (and replaces `mlx_vlm.utils.
/// get_model_and_args`' `importlib.import_module(f"mlx_vlm.models.
/// {model_type}")` dynamic dispatch with an explicit registration
/// table). Per-model VLM architectures are out of scope for this PR, so
/// the registry starts [`empty`](Self::new); future per-usecase model
/// PRs call [`register`](Self::register) (or build one with
/// [`with`](Self::with)) to plug their architecture in. A `model_type`
/// is canonicalized via [`remap_vlm_model_type`] on both registration
/// and lookup, so callers register the *canonical* id and any alias
/// resolves to it.
#[derive(Default)]
pub struct VlmTypeRegistry {
  creators: HashMap<String, VlmModelConstructor>,
}

impl VlmTypeRegistry {
  /// An empty registry (mlx-swift-lm's `VLMTypeRegistry()` — no creators).
  pub fn new() -> Self {
    Self {
      creators: HashMap::new(),
    }
  }

  /// Register `constructor` for `model_type` (canonicalized via
  /// [`remap_vlm_model_type`]), mirroring mlx-swift-lm's
  /// `ModelTypeRegistry.registerModelType(_:creator:)`. A re-registration
  /// of the same (canonical) id replaces the previous constructor
  /// (last-writer-wins, as the Swift dictionary assignment does) and
  /// returns the displaced one.
  pub fn register(
    &mut self,
    model_type: &str,
    constructor: VlmModelConstructor,
  ) -> Option<VlmModelConstructor> {
    self
      .creators
      .insert(remap_vlm_model_type(model_type).to_owned(), constructor)
  }

  /// Builder form of [`register`](Self::register) for assembling a
  /// registry in one expression (the analogue of mlx-swift-lm's
  /// `ModelTypeRegistry(creators:)` init).
  #[must_use]
  pub fn with(mut self, model_type: &str, constructor: VlmModelConstructor) -> Self {
    self.register(model_type, constructor);
    self
  }

  /// `true` if a constructor is registered for `model_type` (after
  /// [`remap_vlm_model_type`]).
  pub fn contains(&self, model_type: &str) -> bool {
    self.creators.contains_key(remap_vlm_model_type(model_type))
  }

  /// Construct a [`VlmModel`] for `loaded`'s [`VlmBaseConfig::model_type`],
  /// mirroring mlx-swift-lm's `VLMTypeRegistry.createModel(configuration:
  /// modelType:)` call at `VLMModelFactory.swift:343-344`. The id is
  /// canonicalized via [`remap_vlm_model_type`]; an unregistered id is a
  /// recoverable [`Error::Backend`] (mlx-swift-lm's
  /// `ModelFactoryError.unsupportedModelType`, mlx-vlm's
  /// `ValueError("Model type … not supported.")`).
  pub fn create(&self, loaded: &LoadedVlmModel) -> Result<Box<dyn VlmModel>> {
    let model_type = remap_vlm_model_type(&loaded.config.model_type);
    let constructor = self
      .creators
      .get(model_type)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "unsupported VLM model type {:?}: no constructor registered (register one via \
           VlmTypeRegistry::register)",
          loaded.config.model_type
        ),
      })?;
    constructor(loaded)
  }
}

/// The cross-model VLM processor trait the per-model processors implement,
/// mirroring the per-model processor protocols in mlx-vlm
/// (e.g. `Qwen2VLProcessor`, `PixtralProcessor` — each carries its own
/// state and exposes the cross-model preprocessing entry point) and
/// mlx-swift-lm's per-model `UserInputProcessor` conformers at
/// `VLMModelFactory.swift:108-134`.
///
/// **Scope here:** only the **cross-model** entry point — the
/// [`ImageProcessorConfig`] the per-model encoder expects for its
/// [`crate::vlm::image::preprocess`] pipeline. Per-model multimodal
/// prompt assembly / video frame handling / tool-augmented chat
/// formatting are per-usecase per the no-per-model-arch rule and are
/// owned by the per-model processor's own (concrete-type) methods —
/// recover the concrete type off this trait object by downcasting
/// through [`as_any`](Processor::as_any) /
/// [`as_any_mut`](Processor::as_any_mut) (e.g.
/// `ctx.processor.as_any().downcast_ref::<Qwen2VLProcessor>()`) as
/// needed by the caller. (Future per-model processor PRs may add more
/// cross-model methods to this trait if a pattern shared by every VLM
/// emerges.)
///
/// `Send + Sync` for the same reason [`VlmModelConstructor`] is: a
/// registry can be shared across threads. `'static` so a constructed
/// `Box<dyn Processor>` is `Any`-downcastable back to the concrete
/// per-model processor.
pub trait Processor: Send + Sync + 'static {
  /// The [`ImageProcessorConfig`] this processor's per-model encoder
  /// expects for [`crate::vlm::image::preprocess`] (mean / std / size /
  /// resize-filter / channel order). Mirrors how
  /// [`crate::vlm::model::Model::image_processor_config`] surfaces the
  /// per-model config off the *model* — but exposed off the
  /// [`Processor`] too so a caller that already has the processor
  /// constructed (via the factory) does not have to also reach into
  /// the model for the preprocessing pipeline's parameters.
  fn image_processor_config(&self) -> ImageProcessorConfig;

  /// Upcast to [`&dyn Any`](std::any::Any) so a caller holding the
  /// erased [`Box<dyn Processor>`] (e.g. off
  /// [`LoadedVlmContext::processor`]) can `downcast_ref` back to the
  /// concrete per-model processor (`Qwen2VLProcessor` / `PixtralProcessor`
  /// / …) to reach its concrete-only methods (multimodal prompt assembly
  /// / video handling / tool+chat formatting). Each concrete impl returns
  /// `self`.
  fn as_any(&self) -> &dyn std::any::Any;

  /// Mutable counterpart of [`as_any`](Processor::as_any) for callers that
  /// need `downcast_mut` to mutate the concrete per-model processor in
  /// place. Each concrete impl returns `self`.
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// A registered VLM processor constructor: assemble a
/// [`Box<dyn Processor>`] from the already-resolved
/// [`LoadedProcessor`] (parsed processor config + raw processor JSON +
/// shared [`VlmBaseConfig`] + already-built [`Tokenizer`]).
///
/// Mirrors mlx-swift-lm's `ProcessorTypeRegistry` creator
/// `(Data, Tokenizer) throws -> UserInputProcessor` at
/// `VLMModelFactory.swift:63-75`. `Send + Sync` so a `static` shared
/// registry can be used from multiple threads.
pub type ProcessorConstructor =
  Box<dyn Fn(&LoadedProcessor<'_>) -> Result<Box<dyn Processor>> + Send + Sync + 'static>;

/// A `processor_class: String` → [`ProcessorConstructor`] table, the VLM
/// load factory's processor **extension point**.
///
/// Mirrors mlx-swift-lm's `VLMProcessorTypeRegistry.shared` at
/// `VLMModelFactory.swift:104-135` (and replaces mlx-vlm's
/// `AutoProcessor.from_pretrained(model_path, use_fast=True)` `transformers`
/// dynamic dispatch with an explicit registration table). Per-model VLM
/// processors are out of scope for this PR; the registry starts
/// [`empty`](Self::new) and future per-usecase processor PRs register
/// their constructors into it. Registration keys are the raw
/// `processor_class` strings (no canonicalization — the
/// `processor_class_override` applies on lookup only, mirroring the
/// swift `processorTypeOverrides` lookup at lines 399-403).
#[derive(Default)]
pub struct VlmProcessorTypeRegistry {
  creators: HashMap<String, ProcessorConstructor>,
}

impl VlmProcessorTypeRegistry {
  /// An empty registry (mlx-swift-lm's `ProcessorTypeRegistry()`).
  pub fn new() -> Self {
    Self {
      creators: HashMap::new(),
    }
  }

  /// Register `constructor` for `processor_class`, mirroring
  /// mlx-swift-lm's `ProcessorTypeRegistry.registerProcessorType(_:creator:)`.
  /// A re-registration of the same id returns the displaced constructor
  /// (last-writer-wins, as the Swift dictionary assignment does).
  pub fn register(
    &mut self,
    processor_class: &str,
    constructor: ProcessorConstructor,
  ) -> Option<ProcessorConstructor> {
    self
      .creators
      .insert(processor_class.to_owned(), constructor)
  }

  /// Builder form of [`register`](Self::register).
  #[must_use]
  pub fn with(mut self, processor_class: &str, constructor: ProcessorConstructor) -> Self {
    self.register(processor_class, constructor);
    self
  }

  /// `true` if a constructor is registered for `processor_class` (raw —
  /// no override applied; the override is consulted only on the lookup
  /// performed by [`load()`]).
  pub fn contains(&self, processor_class: &str) -> bool {
    self.creators.contains_key(processor_class)
  }

  /// Construct a [`Processor`] for `loaded`'s
  /// [`processor_class`](LoadedProcessor::processor_class), mirroring
  /// mlx-swift-lm's `ProcessorTypeRegistry.createModel(configuration:
  /// processorType:tokenizer:)` at `VLMModelFactory.swift:405-407`. The
  /// caller in [`load()`] is responsible for having already applied any
  /// `processor_class_override` to the lookup key. An unregistered id
  /// is a recoverable [`Error::Backend`].
  pub fn create(&self, loaded: &LoadedProcessor<'_>) -> Result<Box<dyn Processor>> {
    let constructor = self
      .creators
      .get(loaded.processor_class)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "unsupported VLM processor class {:?}: no constructor registered (register one via \
           VlmProcessorTypeRegistry::register)",
          loaded.processor_class
        ),
      })?;
    constructor(loaded)
  }
}

/// The product of [`load()`]: a constructed [`VlmModel`] plus the
/// [`Tokenizer`], the constructed [`Processor`], and the parsed
/// [`VlmBaseConfig`].
///
/// Analogue of mlx-swift-lm's `ModelContext` (constructed at
/// `VLMModelFactory.swift:422-425` — the `(configuration, model, processor,
/// tokenizer)` tuple every VLM caller receives). Restricted to the
/// already-modeled fields here; `defaultPrompt` / `extraEOSTokens` /
/// `toolCallFormat` are intentionally not modeled (the eos set is
/// already resolved on the [`Tokenizer`] / [`VlmBaseConfig`]; prompt and
/// tool-format are chat-pipeline concerns above this loader, same
/// boundary [`crate::lm::factory::LoadedModelContext`] holds).
#[non_exhaustive]
pub struct LoadedVlmContext {
  /// The constructed VLM model (from the [`VlmTypeRegistry`] constructor).
  pub model: Box<dyn VlmModel>,
  /// The model's tokenizer, built from the (optionally separate)
  /// tokenizer directory with the resolved eos set.
  pub tokenizer: Tokenizer,
  /// The constructed VLM processor (from the
  /// [`VlmProcessorTypeRegistry`] constructor).
  pub processor: Box<dyn Processor>,
  /// The parsed VLM base `config.json` subset, returned for callers that
  /// need the dispatch metadata (`model_type` / `eos_token_id` /
  /// `quantization`). Model-specific fields (nested `text_config` /
  /// `vision_config` / arch-specific keys) are NOT carried here — they
  /// live on the per-model VLM model constructed off the raw `config.json`
  /// JSON.
  pub config: VlmBaseConfig,
}

/// Load a VLM model + tokenizer + processor from a local
/// [`VlmModelConfiguration`], dispatching to `model_registry` on the
/// checkpoint's `model_type` and to `processor_registry` on the
/// `(pre)processor_config.json`'s `processor_class` (after applying any
/// per-model-type `processor_class_override`).
///
/// The end-to-end port of `mlx_vlm.utils.load` restricted to the
/// local-path, no-network surface (and mlx-swift-lm's
/// `VLMModelFactory._load` at `VLMModelFactory.swift:318-425`). The
/// orchestration order is chosen so the *cheap, recoverable* failures
/// come first — nothing heavy (weights, tokenizer, vision processor) is
/// touched until both registries are known to be able to handle the
/// checkpoint:
///
/// 1. Resolve the model directory ([`VlmModelConfiguration::model_directory`]
///    — local, no Hub download) and read `config.json` **once** via
///    [`load_vlm_base_config`], yielding both the typed [`VlmBaseConfig`]
///    (with the `generation_config.json` eos override applied) and the
///    verbatim JSON body. The VLM-minimal parse is deliberately NOT
///    [`crate::lm::load::load_config`] — real VLMs nest the text-model
///    fields under `text_config`, so requiring them at the top level
///    would reject every real checkpoint *before* a registered VLM
///    constructor saw the raw JSON; the swift loader has the same
///    minimal `BaseConfiguration` (`MLXLMCommon/BaseConfiguration.swift`).
/// 2. **Validate the `model_type` is registered** (after
///    [`remap_vlm_model_type`]) *before* loading anything heavy: an
///    unsupported checkpoint is a cheap, recoverable [`Error::Backend`]
///    here, with no weight/tokenizer/processor I/O — mlx-vlm's
///    `ValueError("Model type … not supported.")` /
///    mlx-swift-lm's `unsupportedModelType`.
/// 3. Read the processor config (`preprocessor_config.json` preferred,
///    `processor_config.json` fallback) ONCE via
///    [`load_processor_config`], get both the typed `processor_class`
///    and the verbatim JSON body, apply any
///    `processor_class_override` for the canonical model type, and
///    **validate the resulting processor class is registered** —
///    same early-fail discipline as step 2, so an unsupported processor
///    class is a cheap, recoverable error before any weight/tokenizer
///    I/O.
/// 4. Select the tokenizer directory
///    ([`tokenizer_source`](VlmModelConfiguration::tokenizer_source) if set,
///    else the model directory — mlx-swift-lm's `tokenizerDirectory`).
/// 5. Discover and merge the weights from the model directory via
///    [`crate::lm::load::load_weights`].
/// 6. Build the [`Tokenizer`] EXACTLY ONCE from the selected directory
///    via [`crate::lm::load::load_tokenizer_with_eos`] (with the eos set
///    already resolved on the [`VlmBaseConfig`] from step 1 — the same
///    primitive [`crate::lm::load::load_tokenizer`] funnels through, so
///    LM and VLM share one eos-resolution path).
/// 7. Construct the model via `model_registry` on the [`LoadedVlmModel`]
///    (parsed VLM base config + raw JSON + weights).
/// 8. Construct the processor via `processor_registry` on the
///    [`LoadedProcessor`] (parsed processor config + raw processor JSON +
///    shared VLM base config + tokenizer reference) and return a
///    [`LoadedVlmContext`].
///
/// Per-model construction is the registries' job (this PR ships no
/// architectures, no processors). No implicit eval — the weights reach
/// the constructor lazily.
pub fn load(
  configuration: &VlmModelConfiguration,
  model_registry: &VlmTypeRegistry,
  processor_registry: &VlmProcessorTypeRegistry,
) -> Result<LoadedVlmContext> {
  let model_dir = configuration.model_directory();

  // (1) Read config.json ONCE into the VLM-minimal `VlmBaseConfig` (+
  // generation_config eos override) AND the verbatim JSON body, from the
  // same bytes. Mirrors VLMModelFactory.swift:325-339 (single `Data` read
  // shared by both BaseConfiguration decode and the per-model creator).
  // We deliberately do NOT route through `lm::load::load_config` here —
  // its required top-level `hidden_size` / `num_hidden_layers` / … would
  // reject every real VLM checkpoint (those fields live nested under
  // `text_config`); a per-model constructor needs the raw JSON to decode
  // them itself (`text_config` / `vision_config` / arch-specific keys
  // outside the minimal base subset, exactly how mlx-swift-lm hands each
  // model the raw config `Data`). Reading once means the typed base and
  // raw text can never come from two different on-disk versions.
  let (config, config_json) = load_vlm_base_config(model_dir)?;

  // (2) Validate the (remapped) model_type is registered BEFORE any
  // weights / tokenizer / processor I/O. An unsupported checkpoint —
  // the common case, since per-model architectures are out of scope and
  // the registry is normally empty — is a cheap, recoverable error
  // here, never paying for the rest of the load pipeline.
  if !model_registry.contains(&config.model_type) {
    return Err(Error::Backend {
      message: format!(
        "unsupported VLM model type {:?}: no constructor registered (register one via \
         VlmTypeRegistry::register)",
        config.model_type
      ),
    });
  }

  // (3) Read the processor config (preprocessor_config.json preferred,
  // processor_config.json fallback) ONCE, then apply any per-model-type
  // override and validate the processor registry can handle it. Order:
  // we run this BEFORE loading weights/tokenizer for the same
  // early-fail-cheap reason as step (2) — a checkpoint whose processor
  // class is registered nowhere should not pay for weight I/O. Some
  // per-model overrides (Mistral3) mean the on-disk
  // `processor_class` is NOT the registry key we look up against:
  // resolve the override on the canonical model_type (the same key we
  // dispatched the model constructor on), exactly mirroring
  // VLMModelFactory.swift:399-403.
  let (proc_config, preprocessor_config_json, processor_config_json, proc_filename) =
    load_processor_config(model_dir)?;
  let canonical_model_type = remap_vlm_model_type(&config.model_type);
  let processor_class = processor_class_override(canonical_model_type)
    .unwrap_or(&proc_config.processor_class)
    .to_owned();
  if !processor_registry.contains(&processor_class) {
    return Err(Error::Backend {
      message: format!(
        "unsupported VLM processor class {processor_class:?} (from {proc_filename} in {}{}): \
         no constructor registered (register one via VlmProcessorTypeRegistry::register)",
        model_dir.display(),
        if processor_class != proc_config.processor_class {
          format!(
            ", overridden from on-disk {:?} for model_type {:?}",
            proc_config.processor_class, canonical_model_type,
          )
        } else {
          String::new()
        },
      ),
    });
  }

  // (4) Select the tokenizer directory FIRST: the separate
  // `tokenizer_source` if set (a real split layout where the model dir
  // has NO `tokenizer.json`), else the model directory (mlx-swift-lm's
  // `tokenizerDirectory`).
  let tokenizer_dir = configuration.tokenizer_directory();

  // (5) Discover/merge the weights from the model directory.
  let weights = load::load_weights(model_dir)?;

  // (6) Build the tokenizer EXACTLY ONCE from the selected directory,
  // through the shared eos-resolution path (the eos set already resolved
  // on `config`). We go through `load_tokenizer_with_eos` since our
  // `VlmBaseConfig` is not the LM `Config` `load_tokenizer` consumes, but
  // the underlying primitive — `Tokenizer::from_path(dir, eos)` — is
  // identical, so LM and VLM resolve the eos set through the same code
  // path.
  let tokenizer = load::load_tokenizer_with_eos(tokenizer_dir, config.eos_token_id.as_ref())?;

  // (7) Construct the model via the registry (already validated as
  // registered in step 2). The model receives the parsed `config`
  // (still owned here) by reference inside `LoadedVlmModel`.
  let loaded = LoadedVlmModel {
    config,
    config_json,
    weights,
  };
  let model = model_registry.create(&loaded)?;

  // (8) Construct the processor via its registry. The processor receives
  // the SAME parsed config the model received (the `loaded.config`
  // reference) and the already-built tokenizer — exactly the
  // `(configuration, processorType, tokenizer)` triple mlx-swift-lm
  // passes at VLMModelFactory.swift:405-407.
  let processor = {
    let loaded_proc = LoadedProcessor {
      config: &loaded.config,
      // SAME single-read body the model constructor received above as
      // `loaded.config_json` — NOT a re-read (preserves the loader's
      // TOCTOU consistency for a processor needing `config.json`-only
      // arch fields).
      config_json: &loaded.config_json,
      processor_class: &processor_class,
      preprocessor_config_json: preprocessor_config_json.as_deref(),
      processor_config_json: processor_config_json.as_deref(),
      processor_config_filename: proc_filename,
      tokenizer: &tokenizer,
    };
    processor_registry.create(&loaded_proc)?
  };

  Ok(LoadedVlmContext {
    model,
    tokenizer,
    processor,
    config: loaded.config,
  })
}

#[cfg(test)]
mod tests {
  //! End-to-end VLM load-factory tests, driven by mock model + mock
  //! processor types registered into fresh registries (per the
  //! no-model-arch rule, this PR ships the seam, not architectures or
  //! processors — so the end-to-end path is proven against hand-traced
  //! mocks over a temp model directory).

  use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
  };

  use super::*;
  use crate::{
    array::Array,
    lm::{cache::KvCache, generate::GenConfig, model::Model as LmModel},
    vlm::{
      generate::{VlmGenConfig, vlm_generate},
      image::{ColorOrder, ImageProcessorConfig, ResizeFilter},
      prompt::MarkerPolicy,
    },
  };

  /// A "flat" mock `config.json` for the mock VLM architecture: the
  /// dispatch-only key (`model_type`) plus `vocab_size` and `mock_extra`
  /// at the **top level**. The minimal [`VlmBaseConfig`] only needs
  /// `model_type`; the mock constructor reads `vocab_size` and
  /// `mock_extra` off the verbatim [`LoadedVlmModel::config_json`] so
  /// the registry-dispatch end-to-end path is proven against the same
  /// raw-JSON model-specific decode every real per-model VLM constructor
  /// performs (the nested-config layout — `text_config.vocab_size` —
  /// is exercised separately by [`mock_nested_config_json`] and
  /// [`load_succeeds_for_nested_vlm_config_with_no_top_level_lm_fields`]).
  fn mock_config_json(model_type: &str) -> String {
    format!(
      r#"{{
        "model_type": "{model_type}",
        "vocab_size": 5,
        "mock_extra": 11
      }}"#
    )
  }

  /// A `config.json` shaped like a **real** VLM checkpoint:
  /// `model_type` at the top level (the dispatch key, mirroring swift's
  /// `BaseConfiguration` at `MLXLMCommon/BaseConfiguration.swift:13-16`),
  /// every text-model field nested under `text_config` (mirroring how
  /// e.g. Qwen2-VL / LLaVA / Pixtral ship their configs, and how
  /// `mlx_vlm.utils.load_model:239-240` sets up
  /// `config.setdefault("text_config", ...)`), and an arbitrary
  /// `vision_config` block. NO top-level `hidden_size` / `num_hidden_layers`
  /// / `vocab_size` / etc. — the regression case the
  /// [`crate::lm::load::Config`] parse would have *fatally rejected*
  /// before this PR's fix (since those fields are required there).
  fn mock_nested_config_json(model_type: &str) -> String {
    format!(
      r#"{{
        "model_type": "{model_type}",
        "text_config": {{
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 2,
          "rope_theta": 10000.0,
          "vocab_size": 5,
          "tie_word_embeddings": false
        }},
        "vision_config": {{
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "image_size": 224
        }},
        "mock_extra": 11
      }}"#
    )
  }

  /// A minimal processor-config body (written to whichever of
  /// `preprocessor_config.json` / `processor_config.json` a test wants).
  /// `processor_class` is the registry key; `mock_image_size` is a
  /// model-specific key OUTSIDE the typed subset, used to prove the
  /// processor constructor reads the carried config body
  /// ([`LoadedProcessor::preprocessor_config_json`] /
  /// [`LoadedProcessor::processor_config_json`]).
  fn mock_preprocessor_config_json(processor_class: &str, image_size: u32) -> String {
    format!(
      r#"{{
        "processor_class": "{processor_class}",
        "mock_image_size": {image_size}
      }}"#
    )
  }

  /// A trivial VLM [`Model`] returned by the mock constructor. Implements
  /// the LM-side [`crate::lm::model::Model`] (vocab-aware zero logits) and
  /// the VLM-side [`crate::vlm::model::Model`]'s required entry points
  /// (text-embed lookup, image-encode passthrough); records both the
  /// raw-JSON-decoded `vocab_size` (which the per-model constructor reads
  /// off `LoadedVlmModel::config_json`, since `VlmBaseConfig` carries
  /// only the dispatch fields) and the raw-config `mock_extra` for
  /// assertions.
  struct MockVlmModel {
    vocab: i32,
    #[allow(dead_code)]
    mock_extra: i64,
  }

  impl LmModel for MockVlmModel {
    fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let (batch, seq) = match tokens.shape().as_slice() {
        [b, s] => (*b, *s),
        [s] => (1, *s),
        other => {
          return Err(Error::ShapeMismatch {
            message: format!("MockVlmModel::forward expects [B, S], got {other:?}"),
          });
        }
      };
      let vocab = self.vocab as usize;
      Array::from_slice::<f32>(&vec![0.0_f32; batch * seq * vocab], &(batch, seq, vocab))
    }
  }

  impl crate::vlm::model::Model for MockVlmModel {
    fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
      // [1, T] tokens → [1, T, hidden=8] zero embeds. Matches the typed
      // Config.hidden_size = 8 from `mock_config_json` so a chained
      // forward_embeddings would line up.
      let shape = tokens.shape();
      let (b, t) = match shape.as_slice() {
        [b, t] => (*b, *t),
        other => {
          return Err(Error::ShapeMismatch {
            message: format!("MockVlmModel::embed_tokens expects [B, T], got {other:?}"),
          });
        }
      };
      Array::from_slice::<f32>(&vec![0.0_f32; b * t * 8], &(b, t, 8usize))
    }

    fn encode_image(&self, _image: &Array) -> Result<Array> {
      // [1, 8] zero features — single placeholder per image into the
      // hidden_size = 8 space.
      Array::from_slice::<f32>(&[0.0_f32; 8], &(1usize, 8usize))
    }
  }

  /// A trivial [`Processor`] returned by the mock processor constructor.
  /// Records the typed `processor_class` it was dispatched on AND the
  /// model-specific `mock_image_size` it read off the raw processor
  /// JSON, so a test can assert both pieces of dispatch state arrived.
  struct MockVlmProcessor {
    #[allow(dead_code)]
    processor_class: String,
    image_size: u32,
  }

  impl Processor for MockVlmProcessor {
    fn image_processor_config(&self) -> ImageProcessorConfig {
      // Honor the image-size the processor decoded off the raw JSON, so
      // a test can assert the cross-model preprocessing parameters
      // round-trip through the registry.
      ImageProcessorConfig {
        size: (self.image_size, self.image_size),
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        rescale_factor: 1.0 / 255.0,
        do_resize: true,
        do_rescale: true,
        do_normalize: true,
        resample: ResizeFilter::Bilinear,
        color_order: ColorOrder::Rgb,
        ..ImageProcessorConfig::default()
      }
    }

    fn as_any(&self) -> &dyn std::any::Any {
      self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
      self
    }
  }

  /// Build a [`VlmModelConstructor`] for the mock VLM architecture: read
  /// the dispatch key off the typed [`LoadedVlmModel::config`]
  /// (`model_type` is the only field guaranteed by [`VlmBaseConfig`]),
  /// then decode model-specific fields (`vocab_size`, `mock_extra`) off
  /// the verbatim [`LoadedVlmModel::config_json`] — mirroring how a real
  /// per-model VLM constructor's `Codable` init reads its nested
  /// `text_config` / `vision_config` blocks off the raw JSON
  /// (`VLMModelFactory.swift:343-348`). `vocab_size` is looked up at the
  /// top level OR under `text_config` so the same mock works for both the
  /// "flat" and "nested" fixtures. Asserts at least one weight tensor
  /// arrived.
  fn mock_vlm_constructor() -> VlmModelConstructor {
    Box::new(|loaded: &LoadedVlmModel| -> Result<Box<dyn VlmModel>> {
      assert!(
        !loaded.weights.is_empty(),
        "constructor should receive the loaded weights"
      );
      let raw: serde_json::Value =
        serde_json::from_str(&loaded.config_json).map_err(|e| Error::Backend {
          message: format!("mock vlm ctor: bad config json: {e}"),
        })?;
      // Vocab can be top-level (the "flat" mock fixture) or nested under
      // text_config (the real-VLM-shaped mock fixture). The per-model
      // constructor decides how to decode its own model-specific fields;
      // both are equally legitimate dispatch outputs here, since
      // `VlmBaseConfig` only requires `model_type` and the rest flows
      // through the raw JSON.
      let vocab = raw
        .get("vocab_size")
        .or_else(|| raw.get("text_config").and_then(|t| t.get("vocab_size")))
        .and_then(serde_json::Value::as_i64)
        .and_then(|x| i32::try_from(x).ok())
        .ok_or_else(|| Error::Backend {
          message: "mock vlm ctor: missing vocab_size (top-level or text_config.vocab_size)".into(),
        })?;
      let mock_extra = raw
        .get("mock_extra")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Error::Backend {
          message: "mock vlm ctor: missing mock_extra".into(),
        })?;
      Ok(Box::new(MockVlmModel { vocab, mock_extra }))
    })
  }

  /// Build a [`ProcessorConstructor`] for the mock processor: read the
  /// processor class off [`LoadedProcessor::processor_class`] and the
  /// model-specific `mock_image_size` off whichever processor-config body
  /// carries it — [`LoadedProcessor::preprocessor_config_json`] when a
  /// `preprocessor_config.json` was present (the common + split layouts),
  /// otherwise [`LoadedProcessor::processor_config_json`] (the
  /// `processor_config.json`-only layout). Mirrors a real per-model
  /// processor decoding its image-preprocessor metadata from the file
  /// that actually carries it.
  fn mock_processor_constructor() -> ProcessorConstructor {
    Box::new(
      |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
        // The image-preprocessor metadata lives in `preprocessor_config.
        // json` when that file is present, else in the
        // `processor_config.json`-only body. Decode from whichever the
        // loader carried — both bodies that were on disk are available.
        let body = loaded
          .preprocessor_config_json
          .or(loaded.processor_config_json)
          .ok_or_else(|| Error::Backend {
            message: "mock vlm processor ctor: no processor-config body carried".into(),
          })?;
        let raw: serde_json::Value = serde_json::from_str(body).map_err(|e| Error::Backend {
          message: format!("mock vlm processor ctor: bad processor config json: {e}"),
        })?;
        let image_size = raw
          .get("mock_image_size")
          .and_then(serde_json::Value::as_u64)
          .and_then(|x| u32::try_from(x).ok())
          .ok_or_else(|| Error::Backend {
            message: "mock vlm processor ctor: missing mock_image_size".into(),
          })?;
        // Sanity-touch the tokenizer the swift `(Data, Tokenizer) ->
        // Processor` shape hands in — assert it can encode something so a
        // future change that hands the wrong (uninitialized / wrong-dir)
        // tokenizer surfaces here.
        let _ = loaded
          .tokenizer
          .encode("a", false)
          .expect("processor constructor must receive a working tokenizer");
        Ok(Box::new(MockVlmProcessor {
          processor_class: loaded.processor_class.to_owned(),
          image_size,
        }))
      },
    )
  }

  /// A fresh, writable per-test temp directory (the crate's
  /// no-`tempfile`-crate convention: `temp_dir()` + pid + a
  /// process-unique counter so parallel tests never collide). Created
  /// empty; the caller populates it.
  fn fresh_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs-vlm-factory-{tag}-{}-{n}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Serialize a minimal but loadable `tokenizer.json` (a 3-token
  /// WordLevel model with a Whitespace pre-tokenizer) into `dir` via
  /// the `tokenizers` crate — the same fixture style as the LM
  /// factory's tests, so the reused [`Tokenizer::from_path`] loads it.
  fn write_tokenizer(dir: &Path) {
    use tokenizers::{
      Tokenizer as HfTokenizer, models::wordlevel::WordLevel,
      pre_tokenizers::whitespace::Whitespace,
    };
    let vocab = [("a", 0u32), ("b", 1), ("c", 2)]
      .iter()
      .map(|(w, i)| ((*w).to_string(), *i))
      .collect();
    let wl = WordLevel::builder()
      .vocab(vocab)
      .unk_token("a".to_string())
      .build()
      .unwrap();
    let mut hf = HfTokenizer::new(wl);
    hf.with_pre_tokenizer(Some(Whitespace {}));
    hf.save(dir.join("tokenizer.json"), false).unwrap();
  }

  /// Populate `dir` with the VLM `config.json` + a tiny single-tensor
  /// `model.safetensors` + the named processor config (one of
  /// `"preprocessor_config.json"` / `"processor_config.json"`) — but
  /// **no** `tokenizer.json`. Basis for [`write_vlm_dir`] (which adds
  /// the tokenizer) and the split-layout test.
  fn write_vlm_dir_no_tokenizer(
    dir: &Path,
    model_type: &str,
    processor_filename: &str,
    processor_class: &str,
    image_size: u32,
  ) {
    std::fs::write(dir.join("config.json"), mock_config_json(model_type)).unwrap();
    std::fs::write(
      dir.join(processor_filename),
      mock_preprocessor_config_json(processor_class, image_size),
    )
    .unwrap();

    // A tiny one-tensor safetensors so `load_weights` finds non-empty
    // weights. `save_safetensors` writes the on-disk format the loader
    // reads.
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  }

  /// Populate `dir` as a minimal but *loadable* VLM model directory:
  /// `config.json`, a tiny single-tensor `model.safetensors`, the named
  /// processor config, and a `tokenizer.json`.
  fn write_vlm_dir(
    dir: &Path,
    model_type: &str,
    processor_filename: &str,
    processor_class: &str,
    image_size: u32,
  ) {
    write_vlm_dir_no_tokenizer(
      dir,
      model_type,
      processor_filename,
      processor_class,
      image_size,
    );
    write_tokenizer(dir);
  }

  #[test]
  fn load_dispatches_to_registered_mocks_and_returns_full_bundle() {
    let dir = fresh_dir("dispatch");
    write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 64);
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

    // The returned VLM base config carries the dispatch key + (here, no)
    // top-level eos override. vocab/etc. flow through the raw JSON to the
    // constructor, which proved them in `logits.shape()` below.
    assert_eq!(ctx.config.model_type, "mockvlm");
    assert_eq!(ctx.config.eos_token_id, None);
    assert_eq!(ctx.config.quantization, None);

    // The constructed model is the mock: drive one forward to confirm it
    // is wired and the constructor saw the right vocab off the raw JSON.
    let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
    let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
    let logits = LmModel::forward(ctx.model.as_ref(), &tokens, &mut cache).unwrap();
    assert_eq!(logits.shape(), vec![1, 3, 5]);

    // The constructed processor surfaces the image-size it decoded off
    // the raw processor JSON (64 from `write_vlm_dir`) — round-trip
    // proof that the processor constructor saw the right JSON body.
    let proc_cfg = ctx.processor.image_processor_config();
    assert_eq!(proc_cfg.size, (64, 64));

    // The tokenizer loaded from the same directory.
    let ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(ids.len(), 3);
  }

  #[test]
  fn loaded_model_drives_vlm_generate_end_to_end() {
    // Codex review (load↔generate integration gap): `load()` hands back a
    // `LoadedVlmContext` whose `model` is a `Box<dyn VlmModel>`, and the
    // public `vlm_generate` is generic over `M: vlm::Model + ?Sized`. This
    // test proves the loader's trait-object output drives the generation
    // loop *directly* — `&*ctx.model` deref-coerces `Box<dyn VlmModel>` to
    // `&dyn VlmModel`, an UNSIZED `M`, which satisfies the relaxed bound.
    // Before the `?Sized` relaxation this call did not compile at all (the
    // implicit `Sized` bound rejected `dyn VlmModel`), so the loader's
    // output was unusable by the generation loop; that regression is now
    // caught here. Zero-image path — `vlm_generate` dispatches straight to
    // `lm::generate::generate_step` (also `?Sized`-generic, accepted
    // because `VlmModel: Model`) — so this needs no image fixture.
    let dir = fresh_dir("e2e-generate");
    write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 64);
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

    // Drive `vlm_generate` on the LOADED model — `&*ctx.model` is a
    // `&dyn VlmModel` (the `Box<dyn VlmModel>` field deref-coerced). The
    // mock's `forward` returns `[B, S, vocab]` zero logits ⇒ greedy argmax
    // is token id 0 every step; an empty eos set lets it run to
    // `max_tokens`. The mock ignores the KV cache, so an empty cache is
    // sufficient to exercise the decode loop.
    let cfg = VlmGenConfig {
      lm: GenConfig {
        max_tokens: 4,
        ..Default::default()
      },
      image_token_id: 99,
      image_marker_id: None,
      num_tokens_per_image: 3,
      marker_policy: MarkerPolicy::Required,
    };
    let prompt = [0_u32, 1, 2];
    // mlx-vlm `generate(model, processor, …)` — the image-processor config is
    // supplied separately; the loaded processor carries the parsed config.
    let img_cfg = ctx.processor.image_processor_config();
    let steps = vlm_generate(&*ctx.model, &img_cfg, &prompt, &[], Vec::new(), cfg)
      .expect("vlm_generate constructs against the loaded trait-object model");

    let tokens: Vec<u32> = steps
      .map(|s| s.expect("each generation step succeeds").token)
      .collect();
    // The loaded model produced exactly `max_tokens` tokens — load→generate
    // works for the trait-object output. (Greedy argmax of all-zero logits
    // is the lowest index, 0.)
    assert_eq!(tokens, vec![0_u32, 0, 0, 0]);
  }

  #[test]
  fn loaded_processor_config_drives_image_preprocessing_not_model_default() {
    // Codex review (load↔generate gap, [high]): `vlm_generate` must
    // preprocess real image prompts with the *loaded* processor's
    // `ImageProcessorConfig` — the one parsed from
    // `preprocessor_config.json` / `processor_config.json` and carried on
    // `LoadedVlmContext.processor` — NOT one re-derived from the model via
    // `Model::image_processor_config()` (which falls back to the trait
    // default / a stale baked-in config). mlx-vlm's `generate(model,
    // processor, …)` takes the processor separately for exactly this
    // reason; `vlm_generate` now mirrors that with an explicit
    // `image_processor_config` parameter.
    //
    // This test wires the divergence concretely: the loaded processor
    // config's image size (48×48, parsed off the processor JSON by the
    // mock processor) DIFFERS from the model default (224×224). It loads
    // a VLM via `load()`, drives `vlm_generate` on the loaded model with
    // the loaded processor's config + one real image, and asserts the
    // model's `encode_image` saw a `[48, 48, 3]` preprocessed array —
    // proof the LOADED config (not the 224×224 model default) drove
    // preprocessing. Before the fix this preprocessed to `[224, 224, 3]`.
    use std::sync::{Arc, Mutex};

    // A VLM model whose `encode_image` records the shape of the
    // (preprocessed) array it receives, so the test can read back what
    // size `preprocess` resized to. `embed_tokens` / `forward` /
    // `forward_embeddings` are minimal but real so the full multimodal
    // prefill+decode path runs. `image_processor_config` is left as the
    // trait default (224×224) — the value that MUST NOT be used.
    struct RecordingVlmModel {
      /// Set once by `encode_image` to the preprocessed input's shape.
      seen_image_shape: Arc<Mutex<Option<Vec<usize>>>>,
    }
    impl LmModel for RecordingVlmModel {
      fn forward(&self, tokens: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
        let (b, s) = match tokens.shape().as_slice() {
          [b, s] => (*b, *s),
          [s] => (1, *s),
          other => {
            return Err(Error::ShapeMismatch {
              message: format!("RecordingVlmModel::forward expects [B, S], got {other:?}"),
            });
          }
        };
        // `[B, S, vocab=5]` zero logits — greedy argmax is token id 0.
        Array::from_slice::<f32>(&vec![0.0_f32; b * s * 5], &(b, s, 5usize))
      }
      fn forward_embeddings(
        &self,
        embeddings: &Array,
        _c: &mut [Box<dyn KvCache>],
      ) -> Result<Array> {
        // `[1, T, D]` merged embeds → `[1, T, vocab=5]` zero logits.
        let (b, t) = match embeddings.shape().as_slice() {
          [b, t, _d] => (*b, *t),
          other => {
            return Err(Error::ShapeMismatch {
              message: format!(
                "RecordingVlmModel::forward_embeddings expects [B, T, D], got {other:?}"
              ),
            });
          }
        };
        Array::from_slice::<f32>(&vec![0.0_f32; b * t * 5], &(b, t, 5usize))
      }
    }
    impl crate::vlm::model::Model for RecordingVlmModel {
      fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
        let (b, t) = match tokens.shape().as_slice() {
          [b, t] => (*b, *t),
          other => {
            return Err(Error::ShapeMismatch {
              message: format!("RecordingVlmModel::embed_tokens expects [B, T], got {other:?}"),
            });
          }
        };
        // hidden_size = 8, matching `encode_image`'s D below.
        Array::from_slice::<f32>(&vec![0.0_f32; b * t * 8], &(b, t, 8usize))
      }
      fn encode_image(&self, image: &Array) -> Result<Array> {
        // Record the preprocessed image shape — this is the observable
        // proof of which `ImageProcessorConfig` drove `preprocess`.
        *self.seen_image_shape.lock().unwrap() = Some(image.shape());
        // `[num_tokens_per_image = 1, D = 8]` features (one row per image,
        // satisfying `vlm_generate`'s `[num_tokens_per_image, D]` check).
        Array::from_slice::<f32>(&[0.0_f32; 8], &(1usize, 8usize))
      }
    }

    let recorded: Arc<Mutex<Option<Vec<usize>>>> = Arc::new(Mutex::new(None));
    // The model constructor captures a clone of the recording handle so
    // the test can read back `encode_image`'s input AFTER `load()` has
    // boxed the model into the `LoadedVlmContext`.
    let model_registry = {
      let recorded = Arc::clone(&recorded);
      VlmTypeRegistry::new().with(
        "recordingvlm",
        Box::new(
          move |loaded: &LoadedVlmModel| -> Result<Box<dyn VlmModel>> {
            assert!(
              !loaded.weights.is_empty(),
              "constructor should receive the loaded weights"
            );
            Ok(Box::new(RecordingVlmModel {
              seen_image_shape: Arc::clone(&recorded),
            }))
          },
        ),
      )
    };
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());

    // Processor config image size = 48 (≠ the 224×224 model default).
    let dir = fresh_dir("loaded-proc-drives-preprocess");
    write_vlm_dir(
      &dir,
      "recordingvlm",
      "preprocessor_config.json",
      "MockProc",
      48,
    );
    let config = VlmModelConfiguration::from_directory(&dir);
    let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

    // Sanity: the loaded processor's config carries the 48×48 size parsed
    // off the JSON, and it differs from the model's (default) 224×224.
    let loaded_img_cfg = ctx.processor.image_processor_config();
    assert_eq!(loaded_img_cfg.size, (48, 48));
    assert_eq!(
      crate::vlm::model::Model::image_processor_config(ctx.model.as_ref()).size,
      (224, 224),
      "the recording model uses the trait-default 224×224 — the value that must NOT drive preprocessing"
    );

    // A real PNG `vlm::image::load_image` can decode (size irrelevant —
    // `preprocess` resizes to the config's `size`).
    let img_path = dir.join("prompt.png");
    let mut buf = ::image::RgbImage::new(10, 7);
    for y in 0..7 {
      for x in 0..10 {
        buf.put_pixel(x, y, ::image::Rgb([(x * 20) as u8, (y * 30) as u8, 64]));
      }
    }
    ::image::DynamicImage::ImageRgb8(buf)
      .save_with_format(&img_path, ::image::ImageFormat::Png)
      .unwrap();

    // Drive `vlm_generate` on the LOADED model with the LOADED processor's
    // config + one image. marker=image_token=99, num_tokens_per_image=1.
    let cfg = VlmGenConfig {
      lm: GenConfig {
        max_tokens: 2,
        ..Default::default()
      },
      image_token_id: 99,
      image_marker_id: None,
      num_tokens_per_image: 1,
      marker_policy: MarkerPolicy::Required,
    };
    let prompt = [0_u32, 99, 1]; // one marker → one image
    let steps = vlm_generate(
      &*ctx.model,
      &loaded_img_cfg,
      &prompt,
      std::slice::from_ref(&img_path),
      Vec::new(),
      cfg,
    )
    .expect("vlm_generate constructs against the loaded model + loaded processor config");
    // Drain so the eager vision pipeline (load → preprocess → encode_image)
    // has definitely run.
    let tokens: Vec<u32> = steps
      .map(|s| s.expect("each generation step succeeds").token)
      .collect();
    assert_eq!(tokens, vec![0_u32, 0]);

    // THE ASSERTION: `encode_image` saw a `[48, 48, 3]` array — the loaded
    // processor config's size drove `preprocess`, NOT the model's 224×224
    // default. (`preprocess` emits channel-last `[H, W, 3]`.)
    let seen = recorded
      .lock()
      .unwrap()
      .clone()
      .expect("encode_image must have run on the single image prompt");
    assert_eq!(
      seen,
      vec![48, 48, 3],
      "image preprocessing must use the loaded processor config's size (48×48), \
       not the model's default 224×224"
    );
  }

  #[test]
  fn preprocessor_config_is_preferred_over_processor_config() {
    // Both files present, with DIFFERENT processor_class values. The
    // `preprocessor_config.json` MUST win (per
    // VLMModelFactory.swift:438-454's preference order); the registry
    // is set up so only the "Preferred" class can construct — the
    // "Fallback" class would resolve to a missing constructor.
    let dir = fresh_dir("prefer-preprocessor");
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("Preferred", 32),
    )
    .unwrap();
    std::fs::write(
      dir.join("processor_config.json"),
      mock_preprocessor_config_json("Fallback", 999),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("Preferred", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry)
      .expect("load should succeed using the preferred preprocessor_config.json");
    // The mock processor records the image_size off the raw JSON — `32`
    // proves the preferred file was used (would be `999` from the
    // fallback otherwise).
    assert_eq!(ctx.processor.image_processor_config().size, (32, 32));
  }

  #[test]
  fn processor_config_is_used_when_only_fallback_present() {
    // No preprocessor_config.json → fall back to processor_config.json.
    let dir = fresh_dir("fallback-processor-config");
    write_vlm_dir(&dir, "mockvlm", "processor_config.json", "MockProc", 48);
    assert!(!dir.join("preprocessor_config.json").exists());
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx =
      load(&config, &model_registry, &processor_registry).expect("fallback processor_config load");
    assert_eq!(ctx.processor.image_processor_config().size, (48, 48));
  }

  #[test]
  fn from_id_resolves_as_local_path() {
    // An `Identifier::Id` is treated as a LOCAL path (no network): pointing
    // it at the temp dir loads exactly as `from_directory` would.
    let dir = fresh_dir("idpath");
    write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 24);
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_id(dir.to_str().unwrap());
    assert_eq!(config.model_directory(), dir.as_path());

    let ctx = load(&config, &model_registry, &processor_registry)
      .expect("id-as-local-path load should succeed");
    assert_eq!(ctx.config.model_type, "mockvlm");
  }

  #[test]
  fn tokenizer_source_loads_from_separate_directory() {
    // Split layout: the model dir has config + processor config +
    // weights but NO tokenizer.json; a separate dir holds the tokenizer.
    // `tokenizer_source` points the load there, mirroring the LM
    // factory's analogous test.
    let model_dir = fresh_dir("split-model");
    write_vlm_dir_no_tokenizer(
      &model_dir,
      "mockvlm",
      "preprocessor_config.json",
      "MockProc",
      16,
    );
    assert!(!model_dir.join("tokenizer.json").exists());
    let tok_dir = fresh_dir("split-tok");
    write_tokenizer(&tok_dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&model_dir).with_tokenizer_source(&tok_dir);
    assert_eq!(config.tokenizer_directory(), tok_dir.as_path());

    let ctx = load(&config, &model_registry, &processor_registry).expect("split-tokenizer load");
    let ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(ids.len(), 3);
  }

  #[test]
  fn unknown_model_type_is_recoverable_error_with_no_io_beyond_config() {
    // config.json says "nope" but only "mockvlm" is registered →
    // unsupported-model-type Error (NOT a panic), naming the type. The
    // weights file is deliberately INVALID, the tokenizer is absent,
    // and the processor config is absent: any load attempt would
    // surface a different error. We must see the unsupported-model
    // error first (faithful to step (2) of the orchestration order).
    let dir = fresh_dir("unknown-model-cheap");
    std::fs::write(dir.join("config.json"), mock_config_json("nope")).unwrap();
    std::fs::write(
      dir.join("model.safetensors"),
      b"this is not a safetensors file",
    )
    .unwrap();
    assert!(!dir.join("tokenizer.json").exists());
    assert!(!dir.join("preprocessor_config.json").exists());

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &model_registry, &processor_registry) else {
      panic!("unknown VLM model_type must error");
    };
    let msg = err.to_string();
    assert!(msg.contains("unsupported VLM model type"), "got: {msg}");
    assert!(msg.contains("nope"), "error should name the type: {msg}");
    // The processor-config / weights / tokenizer paths must NOT have
    // run: their files are intentionally absent/invalid here, and a
    // failure on any of them surfaces a different error message.
    assert!(
      !msg.contains("safetensors") && !msg.contains("processor") && !msg.contains("tokenizer.json"),
      "weights/processor/tokenizer must not have been loaded, got: {msg}"
    );
  }

  #[test]
  fn unknown_processor_class_is_recoverable_error_with_no_weight_io() {
    // Model type IS registered, but the processor class on disk is
    // not. The unsupported-processor-class error must fire BEFORE any
    // weight load: weights file is deliberately invalid here.
    let dir = fresh_dir("unknown-processor-cheap");
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("WrongProc", 16),
    )
    .unwrap();
    std::fs::write(
      dir.join("model.safetensors"),
      b"this is not a safetensors file",
    )
    .unwrap();
    assert!(!dir.join("tokenizer.json").exists());

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &model_registry, &processor_registry) else {
      panic!("unknown processor class must error");
    };
    let msg = err.to_string();
    assert!(
      msg.contains("unsupported VLM processor class"),
      "got: {msg}"
    );
    assert!(
      msg.contains("WrongProc"),
      "error should name the class: {msg}"
    );
    assert!(
      msg.contains("preprocessor_config.json"),
      "error should name the source file: {msg}"
    );
    assert!(
      !msg.contains("safetensors") && !msg.contains("tokenizer.json"),
      "weights/tokenizer must not have been loaded, got: {msg}"
    );
  }

  #[test]
  fn missing_processor_config_is_recoverable_error() {
    // No preprocessor_config.json AND no processor_config.json present.
    let dir = fresh_dir("no-proc-config");
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &model_registry, &processor_registry) else {
      panic!("missing processor config must error");
    };
    let msg = err.to_string();
    assert!(msg.contains("no processor config found"), "got: {msg}");
    assert!(
      msg.contains("preprocessor_config.json") && msg.contains("processor_config.json"),
      "error should name both candidate filenames: {msg}"
    );
  }

  #[test]
  fn processor_class_override_applies_for_mistral3() {
    // Mistral3 ships processor_class = "PixtralProcessor" on disk but
    // VLMModelFactory.swift:399-403 overrides it to "Mistral3Processor"
    // because spatial-merge handling is different. The registry is set
    // up so only "Mistral3Processor" can construct; "PixtralProcessor"
    // would resolve to a missing constructor.
    let dir = fresh_dir("mistral3-override");
    write_vlm_dir(
      &dir,
      "mistral3",
      "preprocessor_config.json",
      "PixtralProcessor",
      40,
    );
    let model_registry = VlmTypeRegistry::new().with("mistral3", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("Mistral3Processor", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry)
      .expect("mistral3 override should dispatch to Mistral3Processor");
    assert_eq!(ctx.processor.image_processor_config().size, (40, 40));
  }

  #[test]
  fn vlm_remap_applies_on_registration_and_lookup() {
    // "lfm2-vl" canonicalizes to "lfm2_vl" (verbatim from
    // mlx_vlm.utils.MODEL_REMAPPING line 34). Registering under either
    // form, the registry finds it under both.
    let registry = VlmTypeRegistry::new().with("lfm2-vl", mock_vlm_constructor());
    assert!(registry.contains("lfm2-vl"));
    assert!(registry.contains("lfm2_vl"));
    assert!(!registry.contains("qwen3_vl"));
    assert_eq!(remap_vlm_model_type("lfm2-vl"), "lfm2_vl");
    assert_eq!(remap_vlm_model_type("qwen3_vl"), "qwen3_vl");
  }

  #[test]
  fn register_replaces_and_returns_previous() {
    let mut registry = VlmTypeRegistry::new();
    assert!(
      registry
        .register("mockvlm", mock_vlm_constructor())
        .is_none()
    );
    assert!(
      registry
        .register("mockvlm", mock_vlm_constructor())
        .is_some()
    );
    let mut proc_registry = VlmProcessorTypeRegistry::new();
    assert!(
      proc_registry
        .register("MockProc", mock_processor_constructor())
        .is_none()
    );
    assert!(
      proc_registry
        .register("MockProc", mock_processor_constructor())
        .is_some()
    );
  }

  #[test]
  fn raw_config_and_processor_json_reach_constructors() {
    // The constructors stash what they SAW; assert both pieces of
    // raw-JSON dispatch state arrived correctly (the model's
    // `mock_extra = 11` from `mock_config_json`, the processor's
    // `mock_image_size = 24` from `mock_preprocessor_config_json`).
    let dir = fresh_dir("raw-dispatch");
    write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 24);
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry).expect("load");
    assert_eq!(ctx.processor.image_processor_config().size, (24, 24));
  }

  #[test]
  fn load_succeeds_for_nested_vlm_config_with_no_top_level_lm_fields() {
    // **Regression** for the Codex finding: real VLM `config.json` files
    // commonly nest the text-model fields (`hidden_size` /
    // `num_hidden_layers` / `vocab_size` / etc.) under `text_config` and
    // only carry `model_type` at the top — exactly what
    // `mock_nested_config_json` shapes. Before the fix, the VLM load path
    // ran the LM `lm::load::Config` parse upfront, which REQUIRES those
    // top-level fields → every real VLM checkpoint fatally errored
    // BEFORE a registered VLM constructor could see the raw JSON. With
    // the VLM-minimal `VlmBaseConfig` parse, the dispatch goes through,
    // the per-model constructor reads its nested `text_config.vocab_size`
    // off the verbatim raw JSON, and the load completes — proven by the
    // shape of the forward pass driving the registered mock constructor.
    let dir = fresh_dir("nested-vlm-config");
    std::fs::write(dir.join("config.json"), mock_nested_config_json("mockvlm")).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 32),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let config = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &model_registry, &processor_registry)
      .expect("nested-config VLM should load (no top-level LM fields)");

    // The dispatch key arrived; vocab/hidden_size live under text_config
    // and are not on `VlmBaseConfig` (faithful to swift's BaseConfiguration).
    assert_eq!(ctx.config.model_type, "mockvlm");

    // Drive one forward — confirms the mock constructor decoded
    // `text_config.vocab_size = 5` off the raw JSON and the registry +
    // weight + tokenizer path all completed against the nested-shaped
    // config.
    let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
    let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
    let logits = LmModel::forward(ctx.model.as_ref(), &tokens, &mut cache).unwrap();
    assert_eq!(logits.shape(), vec![1, 3, 5]);
  }

  #[test]
  fn eos_token_id_on_vlm_config_flows_to_tokenizer() {
    // `eos_token_id` declared at the TOP LEVEL of a real-VLM-shaped
    // `config.json` (no top-level LM fields, nested `text_config`) must
    // be picked up by `VlmBaseConfig` and forwarded to the tokenizer via
    // `load_tokenizer_with_eos` — REPLACING the tokenizer-config default
    // (mirroring `TokenizerWrapper`'s `set(eos_token_ids)` semantics).
    let dir = fresh_dir("eos-from-config");
    let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": [1, 2],
      "text_config": {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": 5,
        "tie_word_embeddings": false
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry).expect("eos config load");

    // Base config carries the [1, 2] list verbatim (shape-preserving).
    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Many(vec![1, 2])),
      "base config should carry the top-level eos_token_id list"
    );
    // And the tokenizer's COMPLETE eos set is exactly {1, 2} — the
    // tokenizer-config default was REPLACED (not unioned) by the resolved
    // list, exactly as `TokenizerWrapper::set(eos_token_ids)` does.
    let eos_set = ctx.tokenizer.eos_token_ids();
    assert_eq!(
      eos_set.iter().copied().collect::<Vec<_>>(),
      vec![1u32, 2],
      "tokenizer eos set should be exactly the resolved {{1, 2}}"
    );
  }

  #[test]
  fn generation_config_eos_overrides_vlm_base_config_eos() {
    // A *truthy* `generation_config.json` `eos_token_id` OVERWRITES the
    // `config.json` value IN PLACE on the returned `VlmBaseConfig` — same
    // semantics as mlx-lm and mlx-vlm (`mlx_vlm/utils.py:506-515`). The
    // override is a scalar `2`; the on-disk config says `1`; the
    // resulting tokenizer eos set must be {2}.
    let dir = fresh_dir("eos-generation-override");
    let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": 1,
      "text_config": {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": 5,
        "tie_word_embeddings": false
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 2}"#).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("eos generation override load");

    // The returned base config reflects the generation-config override
    // (`1` → `2`) — exactly the in-place overwrite mlx-vlm performs.
    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Single(2)),
      "generation_config.json eos_token_id should override config.json"
    );
    // Tokenizer's COMPLETE eos set is the post-override {2}, not the
    // on-disk {1}.
    let eos_set = ctx.tokenizer.eos_token_ids();
    assert_eq!(
      eos_set.iter().copied().collect::<Vec<_>>(),
      vec![2u32],
      "tokenizer eos set should be the overridden {{2}}"
    );
  }

  #[test]
  fn vlm_base_config_parses_without_top_level_lm_fields() {
    // Pure parse-level proof of the contract: a JSON body with ONLY
    // `model_type` (no LM-required fields, no eos, no quantization)
    // parses into a `VlmBaseConfig` — the swift `BaseConfiguration`
    // shape — and the LM `Config` parse would have rejected the same
    // body (required `hidden_size` / etc. absent). Guards against a
    // future regression that re-adds a hard LM field to `VlmBaseConfig`.
    let cfg = r#"{ "model_type": "qwen2_vl" }"#;
    let base = VlmBaseConfig::from_json(cfg).expect("VLM base config should parse");
    assert_eq!(base.model_type, "qwen2_vl");
    assert_eq!(base.eos_token_id, None);
    assert_eq!(base.quantization, None);

    // Same body through the LM `Config` parse fails (missing
    // `hidden_size` and the rest of the required LM subset). This pins
    // *why* we need a separate VLM base parse.
    let lm_err = crate::lm::load::Config::from_json(cfg)
      .expect_err("LM Config should reject a model_type-only body");
    let msg = lm_err.to_string();
    assert!(
      msg.contains("hidden_size") || msg.contains("missing field"),
      "LM Config parse error should name the missing LM field, got: {msg}"
    );
  }

  // ────────────────────────────────────────────────────────────────────
  // Fix 1: nested-EOS promotion regression tests.
  // ────────────────────────────────────────────────────────────────────

  /// Write a `tokenizer_config.json` that pins the tokenizer-config
  /// fallback EOS to vocab id 2 (the `"c"` token in [`write_tokenizer`]'s
  /// 3-token WordLevel vocab). Used by the nested-EOS tests to prove the
  /// promoted nested EOS REPLACES this fallback (rather than the
  /// tokenizer silently dropping the nested value and defaulting to id 2).
  fn write_tokenizer_config_with_eos_c(dir: &Path) {
    std::fs::write(dir.join("tokenizer_config.json"), r#"{ "eos_token": "c" }"#).unwrap();
  }

  #[test]
  fn nested_text_config_eos_promotes_to_tokenizer() {
    // Real VLM layout: NO top-level `eos_token_id`, but `text_config`
    // carries a list `[42, 50]`. The tokenizer_config pins a different
    // fallback EOS (`"c"` → id 2). After load, the tokenizer's COMPLETE
    // eos set MUST be exactly {42, 50} — the nested promotion happened
    // and REPLACED the tokenizer-config default. Before Fix 1, the
    // nested value was silently dropped → eos set = {2}, wrong generation
    // stop.
    let dir = fresh_dir("nested-text-config-eos");
    let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);
    write_tokenizer_config_with_eos_c(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("nested text_config.eos_token_id should promote");

    // Base config carries the promoted [42, 50] list (shape-preserved).
    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Many(vec![42, 50])),
      "VlmBaseConfig should carry the promoted text_config.eos_token_id list"
    );
    // Tokenizer's COMPLETE eos set is exactly {42, 50} — the
    // tokenizer-config default ({2}) was REPLACED, not unioned.
    let eos_set = ctx.tokenizer.eos_token_ids();
    let mut eos_vec: Vec<u32> = eos_set.iter().copied().collect();
    eos_vec.sort_unstable();
    assert_eq!(
      eos_vec,
      vec![42u32, 50],
      "tokenizer eos set should be exactly the promoted {{42, 50}}, not the tokenizer-config fallback"
    );
  }

  #[test]
  fn top_level_eos_wins_over_nested_text_config_eos() {
    // Top-level `eos_token_id = 7` is present AND `text_config.eos_token_id
    // = [42, 50]` is present — top-level MUST win (the nested promotion
    // only triggers when the top-level is `None`). Faithful to swift
    // `BaseConfiguration`'s top-level-only decode (`MLXLMCommon/BaseConfiguration.swift:192-208`).
    let dir = fresh_dir("top-eos-wins-over-nested");
    let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": 7,
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("top-level eos with nested present");

    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Single(7)),
      "top-level eos_token_id must win over nested text_config.eos_token_id"
    );
    let eos_set = ctx.tokenizer.eos_token_ids();
    assert_eq!(
      eos_set.iter().copied().collect::<Vec<_>>(),
      vec![7u32],
      "tokenizer eos set must be the top-level {{7}}, not nested {{42, 50}}"
    );
  }

  #[test]
  fn generation_config_eos_overrides_promoted_nested_eos() {
    // Promotion happens BEFORE the generation_config override: nested
    // `text_config.eos_token_id = [42, 50]` is promoted, but
    // `generation_config.json eos_token_id = 9` then overrides on top —
    // exactly the same precedence Python's
    // `mlx_vlm/utils.py:506-515` block has for the top-level
    // `eos_token_id`.
    let dir = fresh_dir("gen-cfg-overrides-promoted-nested");
    let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 9}"#).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("generation_config override over promoted nested eos");

    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Single(9)),
      "generation_config.json eos_token_id must override the promoted nested value"
    );
    let eos_set = ctx.tokenizer.eos_token_ids();
    assert_eq!(
      eos_set.iter().copied().collect::<Vec<_>>(),
      vec![9u32],
      "tokenizer eos set must be the post-override {{9}}, not the promoted nested set"
    );
  }

  #[test]
  fn nested_llm_config_eos_promotes_when_text_config_absent() {
    // `llm_config` is the alias mlx-vlm rewrites to `text_config` via
    // `config.setdefault("text_config", config.pop("llm_config", {}))`
    // (`mlx_vlm/utils.py:239`). When the checkpoint only has the alias
    // and no canonical `text_config`, the nested-EOS promotion must still
    // pick the alias up so the tokenizer's eos set reflects it.
    let dir = fresh_dir("nested-llm-config-eos");
    // `vocab_size` at top level since the mock constructor only knows
    // top-level + `text_config.vocab_size`; the alias key is incidental
    // to this EOS-promotion test.
    let cfg = r#"{
      "model_type": "mockvlm",
      "vocab_size": 5,
      "llm_config": {
        "hidden_size": 8,
        "eos_token_id": [11, 13]
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("nested llm_config.eos_token_id alias should promote");

    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Many(vec![11, 13])),
      "VlmBaseConfig should carry the promoted llm_config.eos_token_id list"
    );
  }

  #[test]
  fn text_config_eos_wins_over_llm_config_alias_when_both_present() {
    // mlx-vlm's `setdefault(text_config, pop(llm_config))` makes
    // `text_config` the canonical destination (an existing `text_config`
    // is preserved, the `llm_config` alias is only consulted as a
    // fallback). Our promotion mirrors that precedence: when BOTH nested
    // blocks are present and carry different EOS values, `text_config`
    // wins.
    let dir = fresh_dir("text-config-wins-over-llm-config");
    let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "llm_config": {
        "eos_token_id": [11, 13]
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("text_config should win over llm_config alias when both present");

    assert_eq!(
      ctx.config.eos_token_id,
      Some(EosTokenId::Many(vec![42, 50])),
      "text_config.eos_token_id must take precedence over llm_config.eos_token_id"
    );
  }

  #[test]
  fn falsy_nested_eos_does_not_promote() {
    // Truthiness rules match `read_generation_eos`: a scalar `0` is
    // falsy, an empty list is falsy. Either way the promotion must
    // collapse to `None` and the tokenizer falls back to its own
    // `eos_token` from `tokenizer_config.json` (id 2 here). Pinning this
    // protects against a future change that drops the truthy filter and
    // starts forwarding `0`-shaped EOS to the tokenizer.
    let dir = fresh_dir("falsy-nested-eos");
    let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": 0
      },
      "mock_extra": 11
    }"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);
    write_tokenizer_config_with_eos_c(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry).expect("falsy eos");

    // Promotion collapses to `None`, tokenizer falls back to its own
    // `eos_token` ("c" → id 2 from the tokenizer_config above).
    assert_eq!(
      ctx.config.eos_token_id, None,
      "scalar 0 nested eos must not promote (falsy)"
    );
    let eos_set = ctx.tokenizer.eos_token_ids();
    assert_eq!(
      eos_set.iter().copied().collect::<Vec<_>>(),
      vec![2u32],
      "tokenizer should fall back to its tokenizer_config `eos_token` when nested is falsy"
    );
  }

  // ────────────────────────────────────────────────────────────────────
  // Fix 2: preprocessor + processor_config dispatch fallback.
  // ────────────────────────────────────────────────────────────────────

  /// A minimal **image-preprocessor-only** `preprocessor_config.json` —
  /// the real HF VLM layout: `image_mean` / `image_std` and a
  /// model-specific `mock_image_size`, NO `processor_class`. Used by the
  /// Fix 2 regression case where dispatch metadata must come from
  /// `processor_config.json` instead.
  fn mock_image_only_preprocessor_config_json(image_size: u32) -> String {
    format!(
      r#"{{
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "mock_image_size": {image_size}
      }}"#
    )
  }

  /// A `processor_config.json` carrying ONLY the dispatch class — the
  /// `AutoProcessor`-style combined config from real HF VLM checkpoints.
  fn mock_processor_class_only_config_json(processor_class: &str) -> String {
    format!(r#"{{ "processor_class": "{processor_class}" }}"#)
  }

  /// A `processor_config.json` carrying the dispatch class AND a
  /// **required non-class processor-level field** (`image_seq_len`) — the
  /// real `AutoProcessor` shape where `processor_config.json` holds
  /// processor metadata a per-model processor needs *in addition to* the
  /// image-preprocessor metadata that lives in `preprocessor_config.json`.
  fn mock_processor_config_with_seq_len(processor_class: &str, image_seq_len: u32) -> String {
    format!(r#"{{ "processor_class": "{processor_class}", "image_seq_len": {image_seq_len} }}"#)
  }

  #[test]
  fn processor_class_falls_back_to_processor_config_when_preprocessor_has_none() {
    // **Regression** for Codex Fix 2: real HF VLM dir where the
    // preprocessor file carries ONLY image-preprocessor metadata (no
    // `processor_class`) and `processor_config.json` carries the dispatch
    // class. Before the fix, the strict parse of `preprocessor_config.json`
    // failed immediately → `processor_config.json` was never tried →
    // otherwise-loadable VLM dir was rejected. After the fix: dispatch
    // class comes from `processor_config.json`, but the constructor still
    // sees the `preprocessor_config.json` body (image-preprocessor
    // metadata) — its `mock_image_size = 24` round-trips through the
    // mock processor, proving the constructor body source.
    let dir = fresh_dir("split-dispatch-preprocessor-no-class");
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_image_only_preprocessor_config_json(24),
    )
    .unwrap();
    std::fs::write(
      dir.join("processor_config.json"),
      mock_processor_class_only_config_json("MockProc"),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("dispatch class from processor_config.json + body from preprocessor_config.json");

    // The mock processor recorded the `mock_image_size = 24` it decoded
    // off the body — i.e. the constructor received the
    // `preprocessor_config.json` body (which has `mock_image_size`), NOT
    // the `processor_config.json` body (which has only
    // `processor_class`). Round-trip proof that the split-source
    // dispatch picked the right body.
    assert_eq!(
      ctx.processor.image_processor_config().size,
      (24, 24),
      "constructor must see preprocessor_config.json body (image-preprocessor metadata)"
    );
  }

  #[test]
  fn split_layout_carries_both_preprocessor_and_processor_config_bodies() {
    // **Regression** for the Codex finding: in the split layout
    // (`preprocessor_config.json` has the image-preprocessor metadata but
    // NO `processor_class`; `processor_config.json` supplies the dispatch
    // class) the loader used to extract ONLY the dispatch class from
    // `processor_config.json` and discard that file's body. A per-model
    // processor needing a processor-level field from `processor_config.
    // json` (here `image_seq_len`) AND the image-preprocessor metadata
    // had no way to reach the discarded body. After the fix BOTH bodies
    // are carried.
    let dir = fresh_dir("split-carries-both-bodies");
    // `preprocessor_config.json`: image-preprocessor metadata, no class.
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_image_only_preprocessor_config_json(24),
    )
    .unwrap();
    // `processor_config.json`: dispatch class + a REQUIRED non-class
    // processor-level field.
    std::fs::write(
      dir.join("processor_config.json"),
      mock_processor_config_with_seq_len("MockProc", 256),
    )
    .unwrap();

    // (a) `load_processor_config` directly: BOTH `Option<String>` slots
    // are populated, each keyed by file identity.
    let (proc_config, preprocessor_body, processor_body, filename) =
      load_processor_config(&dir).expect("split-layout processor config must resolve");
    assert_eq!(
      proc_config.processor_class, "MockProc",
      "dispatch class must come from processor_config.json"
    );
    assert_eq!(
      filename, "preprocessor_config.json",
      "primary-body filename is the preprocessor file (image-preprocessor metadata source)"
    );
    let preprocessor_body =
      preprocessor_body.expect("preprocessor_config.json body must be carried");
    assert!(
      preprocessor_body.contains("mock_image_size"),
      "preprocessor body must carry the image-preprocessor metadata, got: {preprocessor_body}"
    );
    let processor_body =
      processor_body.expect("processor_config.json body must be carried, not discarded");
    assert!(
      processor_body.contains("image_seq_len"),
      "processor_config.json body must survive with its non-class field, got: {processor_body}"
    );

    // (b) Through the full `load()` pipeline: the per-model processor
    // constructor sees BOTH bodies on `LoadedProcessor`. The constructor
    // closure asserts `processor_config_json` is `Some` and exposes
    // `image_seq_len = 256` AND `preprocessor_config_json` is `Some` with
    // the image-preprocessor metadata — if either is missing it returns
    // an `Err` and the `load()` below fails.
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let asserting_processor_ctor: ProcessorConstructor = Box::new(
      |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
        let preprocessor = loaded
          .preprocessor_config_json
          .ok_or_else(|| Error::Backend {
            message: "expected preprocessor_config.json body on LoadedProcessor".into(),
          })?;
        let processor = loaded.processor_config_json.ok_or_else(|| Error::Backend {
          message: "expected processor_config.json body on LoadedProcessor (the carried body)"
            .into(),
        })?;
        let pre: serde_json::Value =
          serde_json::from_str(preprocessor).map_err(|e| Error::Backend {
            message: format!("bad preprocessor body: {e}"),
          })?;
        let image_size = pre
          .get("mock_image_size")
          .and_then(serde_json::Value::as_u64)
          .and_then(|x| u32::try_from(x).ok())
          .ok_or_else(|| Error::Backend {
            message: "preprocessor body missing mock_image_size".into(),
          })?;
        let proc: serde_json::Value =
          serde_json::from_str(processor).map_err(|e| Error::Backend {
            message: format!("bad processor_config.json body: {e}"),
          })?;
        let seq_len = proc
          .get("image_seq_len")
          .and_then(serde_json::Value::as_u64)
          .ok_or_else(|| Error::Backend {
            message: "processor_config.json body missing image_seq_len (the discarded field)"
              .into(),
          })?;
        if seq_len != 256 {
          return Err(Error::Backend {
            message: format!("image_seq_len must round-trip as 256, got {seq_len}"),
          });
        }
        Ok(Box::new(MockVlmProcessor {
          processor_class: loaded.processor_class.to_owned(),
          image_size,
        }))
      },
    );

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", asserting_processor_ctor);
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("split-layout load must surface BOTH config bodies to the processor constructor");
    // The constructor read `mock_image_size = 24` off the preprocessor
    // body — proof the preprocessor body also reached it alongside the
    // processor_config.json body.
    assert_eq!(ctx.processor.image_processor_config().size, (24, 24));
  }

  #[test]
  fn preferred_class_layout_still_carries_processor_config_body() {
    // **Regression** for the Codex finding on the *preferred-class* path:
    // `preprocessor_config.json` carries the `processor_class` (so it
    // wins dispatch) AND its own image-preprocessor metadata, while
    // `processor_config.json` ALSO exists with a required non-class
    // processor-level field (`image_seq_len`). Before the fix this path
    // returned `processor_config_json: None` and discarded the
    // `processor_config.json` body even though it was on disk — a
    // per-model processor needing that field had to re-open the file
    // (the TOCTOU/config-divergence the loader exists to prevent). After
    // the fix BOTH bodies are carried, and the dispatch class still
    // comes from `preprocessor_config.json` (precedence unchanged).
    let dir = fresh_dir("preferred-class-carries-processor-body");
    // `preprocessor_config.json`: HAS `processor_class` + image metadata.
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 24),
    )
    .unwrap();
    // `processor_config.json`: also present, carrying a REQUIRED
    // non-class processor-level field (plus a class that must NOT be the
    // one used for dispatch).
    std::fs::write(
      dir.join("processor_config.json"),
      mock_processor_config_with_seq_len("OtherProc", 256),
    )
    .unwrap();

    // (a) `load_processor_config` directly: dispatch class is the
    // preferred file's, filename is the preferred file, and BOTH bodies
    // are populated — `processor_config.json`'s body is NOT discarded.
    let (proc_config, preprocessor_body, processor_body, filename) =
      load_processor_config(&dir).expect("preferred-class processor config must resolve");
    assert_eq!(
      proc_config.processor_class, "MockProc",
      "dispatch class must come from preprocessor_config.json (precedence unchanged)"
    );
    assert_eq!(
      filename, "preprocessor_config.json",
      "primary-body filename is the preprocessor file"
    );
    let preprocessor_body =
      preprocessor_body.expect("preprocessor_config.json body must be carried");
    assert!(
      preprocessor_body.contains("mock_image_size"),
      "preprocessor body must carry the image-preprocessor metadata, got: {preprocessor_body}"
    );
    let processor_body = processor_body
      .expect("processor_config.json body must be carried in the preferred-class path too");
    assert!(
      processor_body.contains("image_seq_len"),
      "processor_config.json body must survive with its non-class field, got: {processor_body}"
    );

    // (b) Through the full `load()` pipeline: the per-model processor
    // constructor sees BOTH bodies on `LoadedProcessor`. The constructor
    // asserts `processor_config_json` is `Some` exposing `image_seq_len
    // = 256` AND `preprocessor_config_json` is `Some` with the
    // image-preprocessor metadata — failure of either makes `load()`
    // error.
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let asserting_processor_ctor: ProcessorConstructor = Box::new(
      |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
        let preprocessor = loaded
          .preprocessor_config_json
          .ok_or_else(|| Error::Backend {
            message: "expected preprocessor_config.json body on LoadedProcessor".into(),
          })?;
        let processor = loaded.processor_config_json.ok_or_else(|| Error::Backend {
          message: "expected processor_config.json body on LoadedProcessor (the carried body)"
            .into(),
        })?;
        let pre: serde_json::Value =
          serde_json::from_str(preprocessor).map_err(|e| Error::Backend {
            message: format!("bad preprocessor body: {e}"),
          })?;
        let image_size = pre
          .get("mock_image_size")
          .and_then(serde_json::Value::as_u64)
          .and_then(|x| u32::try_from(x).ok())
          .ok_or_else(|| Error::Backend {
            message: "preprocessor body missing mock_image_size".into(),
          })?;
        let proc: serde_json::Value =
          serde_json::from_str(processor).map_err(|e| Error::Backend {
            message: format!("bad processor_config.json body: {e}"),
          })?;
        let seq_len = proc
          .get("image_seq_len")
          .and_then(serde_json::Value::as_u64)
          .ok_or_else(|| Error::Backend {
            message: "processor_config.json body missing image_seq_len (the discarded field)"
              .into(),
          })?;
        if seq_len != 256 {
          return Err(Error::Backend {
            message: format!("image_seq_len must round-trip as 256, got {seq_len}"),
          });
        }
        Ok(Box::new(MockVlmProcessor {
          processor_class: loaded.processor_class.to_owned(),
          image_size,
        }))
      },
    );

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    // ONLY `MockProc` is registered — if dispatch had used the
    // `processor_config.json` class (`OtherProc`) the lookup would miss.
    let processor_registry =
      VlmProcessorTypeRegistry::new().with("MockProc", asserting_processor_ctor);
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry).expect(
      "preferred-class load must dispatch on preprocessor_config.json's class and surface \
       BOTH config bodies",
    );
    // Round-trip proof: the constructor decoded `mock_image_size = 24`
    // off the preprocessor body, and only `MockProc` is registered so
    // dispatch used the preprocessor file's `processor_class`.
    assert_eq!(ctx.processor.image_processor_config().size, (24, 24));
  }

  #[test]
  fn neither_processor_config_file_has_processor_class_is_recoverable_error() {
    // Both files present, NEITHER has `processor_class`. The error must
    // be a recoverable `Backend` naming the dir; we additionally check the
    // message identifies the missing dispatch field so an operator can
    // diagnose without source-diving.
    let dir = fresh_dir("neither-has-processor-class");
    std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_image_only_preprocessor_config_json(16),
    )
    .unwrap();
    std::fs::write(
      dir.join("processor_config.json"),
      r#"{ "some_other_key": 1 }"#,
    )
    .unwrap();
    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry = VlmProcessorTypeRegistry::new();
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let Err(err) = load(&configuration, &model_registry, &processor_registry) else {
      panic!("missing-processor-class across both files must error");
    };
    let msg = err.to_string();
    assert!(
      msg.contains("processor_class") || msg.contains("dispatch class"),
      "error should name the missing dispatch field, got: {msg}"
    );
    assert!(
      msg.contains("preprocessor_config.json") && msg.contains("processor_config.json"),
      "error should name both candidate filenames, got: {msg}"
    );
  }

  /// A concrete processor with a method that is NOT on the [`Processor`]
  /// trait — standing in for the per-model concrete-only surface
  /// (multimodal prompt assembly / video handling / tool+chat
  /// formatting) a real `Qwen2VLProcessor` / `PixtralProcessor` carries.
  struct MockConcreteProcessor {
    special: u32,
  }

  impl MockConcreteProcessor {
    /// Concrete-only method unreachable through `dyn Processor` — only a
    /// successful downcast to the concrete type can call it.
    fn mock_special(&self) -> u32 {
      self.special
    }
  }

  impl Processor for MockConcreteProcessor {
    fn image_processor_config(&self) -> ImageProcessorConfig {
      ImageProcessorConfig {
        size: (1, 1),
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        rescale_factor: 1.0 / 255.0,
        do_resize: true,
        do_rescale: true,
        do_normalize: true,
        resample: ResizeFilter::Bilinear,
        color_order: ColorOrder::Rgb,
        ..ImageProcessorConfig::default()
      }
    }

    fn as_any(&self) -> &dyn std::any::Any {
      self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
      self
    }
  }

  #[test]
  fn loaded_processor_downcasts_to_concrete_per_model_type() {
    // Codex review: `LoadedVlmContext.processor` is an erased
    // `Box<dyn Processor>`, but a caller needs the CONCRETE per-model
    // processor (`Qwen2VLProcessor` / `PixtralProcessor` / …) to reach
    // its concrete-only methods (multimodal prompt assembly / video /
    // tool+chat formatting). The trait now upcasts to `Any` via
    // `as_any`, so the erased processor handed back by `load()` can be
    // downcast to its concrete type end-to-end. Before the `as_any` +
    // `'static` change there was no way to recover the concrete type off
    // `load()`'s output, so the concrete-only API was unreachable; this
    // proves the round-trip works.
    let dir = fresh_dir("processor-downcast");
    write_vlm_dir(
      &dir,
      "mockvlm",
      "preprocessor_config.json",
      "MockConcreteProc",
      64,
    );

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry = VlmProcessorTypeRegistry::new().with(
      "MockConcreteProc",
      Box::new(
        |_loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
          Ok(Box::new(MockConcreteProcessor { special: 4242 }))
        },
      ),
    );
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("load should construct the concrete processor");

    // Recover the concrete per-model processor off the erased
    // `Box<dyn Processor>` and call its concrete-only method.
    let concrete = ctx
      .processor
      .as_any()
      .downcast_ref::<MockConcreteProcessor>()
      .expect("loaded processor must downcast to its concrete per-model type");
    assert_eq!(concrete.mock_special(), 4242);
  }

  #[test]
  fn loaded_processor_reads_model_config_json_only_arch_field() {
    // Codex review (Finding 2): a concrete per-model processor's
    // downcast-only methods may need an arch field that lives ONLY in the
    // model `config.json` (e.g. a `hidden_size` / `image_token_index`
    // nested under `text_config`), NOT in either processor-config body.
    // Before this fix `LoadedProcessor` exposed only the processor configs
    // + the typed `VlmBaseConfig` subset, so such a processor had to
    // re-open `config.json` itself — losing the single-read TOCTOU
    // consistency the loader provides. `LoadedProcessor.config_json` now
    // carries the SAME body the model constructor received, so the
    // processor reads the field off the loader's single read.
    let dir = fresh_dir("processor-reads-model-config-json");
    // `config.json`: nested-shaped, carries `text_config.hidden_size = 8`
    // — an arch field present ONLY here (the processor configs below do
    // NOT carry it).
    std::fs::write(dir.join("config.json"), mock_nested_config_json("mockvlm")).unwrap();
    // The processor config carries `mock_image_size = 999` — DELIBERATELY
    // different from `text_config.hidden_size = 8`. The constructor below
    // ignores `mock_image_size` and instead drives `image_size` from the
    // `config.json`-only `hidden_size`, so a passing `(8, 8)` assertion
    // proves the value came from `config_json`, not the processor config.
    std::fs::write(
      dir.join("preprocessor_config.json"),
      mock_preprocessor_config_json("MockProc", 999),
    )
    .unwrap();
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
    write_tokenizer(&dir);

    let config_reading_ctor: ProcessorConstructor = Box::new(
      |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
        // Read the arch field off `LoadedProcessor.config_json` — the
        // SAME single-read body the model constructor saw. It is NOT in
        // either processor-config body.
        let cfg: serde_json::Value =
          serde_json::from_str(loaded.config_json).map_err(|e| Error::Backend {
            message: format!("processor ctor: bad model config_json: {e}"),
          })?;
        let hidden_size = cfg
          .get("text_config")
          .and_then(|t| t.get("hidden_size"))
          .and_then(serde_json::Value::as_u64)
          .and_then(|x| u32::try_from(x).ok())
          .ok_or_else(|| Error::Backend {
            message: "processor ctor: text_config.hidden_size must be readable off \
                      LoadedProcessor.config_json (config.json-only arch field)"
              .into(),
          })?;
        // Drive `image_size` from the config.json-only field (NOT the
        // processor config's `mock_image_size`) so the test can assert
        // the value round-tripped from `config_json`.
        Ok(Box::new(MockVlmProcessor {
          processor_class: loaded.processor_class.to_owned(),
          image_size: hidden_size,
        }))
      },
    );

    let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
    let processor_registry = VlmProcessorTypeRegistry::new().with("MockProc", config_reading_ctor);
    let configuration = VlmModelConfiguration::from_directory(&dir);

    let ctx = load(&configuration, &model_registry, &processor_registry)
      .expect("load must surface the model config.json to the processor constructor");

    // `text_config.hidden_size = 8` round-tripped through
    // `LoadedProcessor.config_json` into the processor — NOT the
    // processor config's `mock_image_size = 999`.
    assert_eq!(
      ctx.processor.image_processor_config().size,
      (8, 8),
      "processor must have read hidden_size=8 off LoadedProcessor.config_json, \
       not mock_image_size=999 off the processor config"
    );
  }
}
