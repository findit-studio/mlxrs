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
  error::{Error, MissingFieldPayload, MissingKeyPayload, ParsePayload, Result},
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
/// [`config_json`](LoadedVlmModel::config_json_ref), exactly as each swift VLM's
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
  model_type: String,
  /// `config.json` `eos_token_id` (a single id or a list). A *truthy*
  /// `generation_config.json` `eos_token_id` overrides it; the result is
  /// the tokenizer's COMPLETE eos set (REPLACES the tokenizer-config
  /// default — see [`load_vlm_base_config`]). `None` ⇒ fall back to the
  /// tokenizer's own `eos_token`. Optional so a VLM with no top-level
  /// `eos_token_id` (and a `text_config.eos_token_id`-only layout, which a
  /// per-model constructor would surface) still parses.
  #[serde(default)]
  eos_token_id: Option<EosTokenId>,
  /// Weight-quantization parameters (`config["quantization"]`), if the
  /// checkpoint carries them at the top level. Optional and forward-
  /// compatible: a VLM whose quantization sits under
  /// `text_config.quantization_config` (mlx-vlm's `load_model`
  /// translation at `mlx_vlm/utils.py:275-301`) parses with this `None`,
  /// and the per-model constructor extracts its own translation off the
  /// raw JSON if it needs to. Carried, not applied — same convention as
  /// [`crate::lm::load::Config::quantization`].
  #[serde(default)]
  quantization: Option<Quantization>,
}

impl VlmBaseConfig {
  /// Parse a [`VlmBaseConfig`] from an in-memory `config.json` string.
  /// Mirrors the swift `JSONDecoder().decode(BaseConfiguration.self, …)`
  /// in `VLMModelFactory._load` at `VLMModelFactory.swift:335`. A serde
  /// failure (malformed JSON or a missing `model_type`) maps to
  /// [`Error::Backend`] — the codebase config-parse convention.
  pub fn from_json(json: &str) -> Result<VlmBaseConfig> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "VlmBaseConfig::from_json",
        "VLM base config.json",
        e,
      ))
    })
  }

  // ── accessors ─────────────────────────────────────────────────────────────

  /// Architecture id (the [`VlmTypeRegistry`] dispatch key).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }
  /// Resolved eos token id set (generation-config override already applied).
  #[inline(always)]
  pub fn eos_token_id(&self) -> Option<&EosTokenId> {
    self.eos_token_id.as_ref()
  }
  /// Weight-quantization parameters (if present). `Quantization` is `Copy`.
  #[inline(always)]
  pub fn quantization(&self) -> Option<Quantization> {
    self.quantization
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
    return Err(Error::FileIo(crate::error::FileIoPayload::new(
      "load_vlm_base_config: VLM base config.json",
      crate::error::FileOp::Open,
      path,
      std::io::Error::from(std::io::ErrorKind::NotFound),
    )));
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
  processor_class: String,
}

impl ProcessorConfig {
  /// The processor class id (the [`VlmProcessorTypeRegistry`] lookup key).
  #[inline(always)]
  pub fn processor_class(&self) -> &str {
    &self.processor_class
  }
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
    // are NOT a parse error at this layer; the dispatch class comes from
    // `processor_config.json` in that case (handled below). A truly
    // malformed (non-JSON) preferred file is still an error, since the
    // constructor would also choke on it.
    let parsed: ProcessorClassOnly = serde_json::from_str(&body).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "load_processor_config: preferred processor config (expected an OBJECT, optionally with a \
           `processor_class` string field)",
        "JSON",
        e,
      ))
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
      return Err(Error::MissingField(MissingFieldPayload::new(
        "ProcessorConfig (preferred file has no `processor_class`, and no fallback file present)",
        "processor_class",
      )));
    };
    let fallback_parsed: ProcessorClassOnly =
      serde_json::from_str(&fallback_body).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "load_processor_config: fallback processor config (expected an OBJECT, optionally with \
             a `processor_class` string field)",
          "JSON",
          e,
        ))
      })?;
    let Some(processor_class) = fallback_parsed.processor_class else {
      return Err(Error::MissingField(MissingFieldPayload::new(
        "ProcessorConfig (neither preferred nor fallback file has `processor_class`)",
        "processor_class",
      )));
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
    return Err(Error::FileIo(crate::error::FileIoPayload::new(
      "load_processor_config: neither preferred nor fallback processor config present",
      crate::error::FileOp::Open,
      fallback_path,
      std::io::Error::from(std::io::ErrorKind::NotFound),
    )));
  };
  let parsed: ProcessorClassOnly = serde_json::from_str(&body).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "load_processor_config: fallback processor config (expected an OBJECT, optionally with a \
         `processor_class` string field)",
      "JSON",
      e,
    ))
  })?;
  let Some(processor_class) = parsed.processor_class else {
    return Err(Error::MissingField(MissingFieldPayload::new(
      "ProcessorConfig (fallback file has no `processor_class` field for the dispatch class, and \
         no preferred file present)",
      "processor_class",
    )));
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
/// arch-specific fields), the verbatim [`config_json`](Self::config_json_ref)
/// text — the analogue of mlx-swift-lm passing the raw `config.json` `Data`
/// to each model's `Codable` init at `VLMModelFactory.swift:341-348` — and
/// takes the weight [`Array`](crate::array::Array)s it needs out of
/// [`weights`](Self::weights_ref) **by reference** (no implicit eval; mlx
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
  /// [`config_json`](Self::config_json_ref) by the per-model constructor.
  config: VlmBaseConfig,
  /// The verbatim `config.json` body, for every key outside the typed
  /// [`VlmBaseConfig`] subset — i.e. the nested `text_config` /
  /// `vision_config` / arch-specific fields each VLM constructor decodes
  /// itself, the analogue of mlx-swift-lm handing each model's `Codable`
  /// init the raw config `Data` at `VLMModelFactory.swift:343-344`. Always
  /// the bytes the typed config was parsed from.
  config_json: String,
  /// The merged, name → [`Array`](crate::array::Array) weight map
  /// (mlx-vlm's `weights` dict). Keys are verbatim — the constructor
  /// applies any `sanitize`/remap itself, exactly as
  /// [`crate::lm::load::load_weights`] documents.
  weights: Weights,
}

impl LoadedVlmModel {
  /// Construct a [`LoadedVlmModel`] from its components.
  pub fn new(config: VlmBaseConfig, config_json: String, weights: Weights) -> Self {
    Self {
      config,
      config_json,
      weights,
    }
  }

  /// The typed VLM base config (dispatch key + eos + quantization).
  #[inline(always)]
  pub fn config_ref(&self) -> &VlmBaseConfig {
    &self.config
  }

  /// The verbatim `config.json` body the per-model constructor decodes its
  /// arch-specific fields off.
  #[inline(always)]
  pub fn config_json_ref(&self) -> &str {
    &self.config_json
  }

  /// The weight map handed to the per-model constructor.
  #[inline(always)]
  pub fn weights_ref(&self) -> &Weights {
    &self.weights
  }
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
/// [`LoadedVlmModel::config_json_ref`], NOT a re-read — so the processor and
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
  /// [`LoadedVlmModel::config_json_ref`] (NOT a re-read). A concrete
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
    let constructor = self.creators.get(model_type).ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "VlmTypeRegistry::create: no constructor registered for model_type (register one via \
           VlmTypeRegistry::register)",
        loaded.config.model_type.clone(),
      ))
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
    let constructor = self.creators.get(loaded.processor_class).ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "VlmProcessorTypeRegistry::create: no constructor registered for processor_class \
           (register one via VlmProcessorTypeRegistry::register)",
        loaded.processor_class.to_owned(),
      ))
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
  model: Box<dyn VlmModel>,
  /// The model's tokenizer, built from the (optionally separate)
  /// tokenizer directory with the resolved eos set.
  tokenizer: Tokenizer,
  /// The constructed VLM processor (from the
  /// [`VlmProcessorTypeRegistry`] constructor).
  processor: Box<dyn Processor>,
  /// The parsed VLM base `config.json` subset, returned for callers that
  /// need the dispatch metadata (`model_type` / `eos_token_id` /
  /// `quantization`). Model-specific fields (nested `text_config` /
  /// `vision_config` / arch-specific keys) are NOT carried here — they
  /// live on the per-model VLM model constructed off the raw `config.json`
  /// JSON.
  config: VlmBaseConfig,
}

impl LoadedVlmContext {
  /// Construct a [`LoadedVlmContext`] from its components.
  pub fn new(
    model: Box<dyn VlmModel>,
    tokenizer: Tokenizer,
    processor: Box<dyn Processor>,
    config: VlmBaseConfig,
  ) -> Self {
    Self {
      model,
      tokenizer,
      processor,
      config,
    }
  }

  /// The constructed VLM model.
  #[inline(always)]
  pub fn model(&self) -> &dyn VlmModel {
    self.model.as_ref()
  }

  /// Mutable reference to the constructed VLM model.
  #[inline(always)]
  pub fn model_mut(&mut self) -> &mut dyn VlmModel {
    self.model.as_mut()
  }

  /// The model's tokenizer.
  #[inline(always)]
  pub fn tokenizer(&self) -> &Tokenizer {
    &self.tokenizer
  }

  /// The constructed VLM processor.
  #[inline(always)]
  pub fn processor(&self) -> &dyn Processor {
    self.processor.as_ref()
  }

  /// Mutable reference to the constructed VLM processor.
  #[inline(always)]
  pub fn processor_mut(&mut self) -> &mut dyn Processor {
    self.processor.as_mut()
  }

  /// The parsed VLM base config (dispatch metadata).
  #[inline(always)]
  pub fn config_ref(&self) -> &VlmBaseConfig {
    &self.config
  }
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
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "load: no VLM model_type constructor registered (register one via \
         VlmTypeRegistry::register)",
      config.model_type.as_str(),
    )));
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
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "load: no VLM processor_class constructor registered (register one via \
         VlmProcessorTypeRegistry::register; a `processor_class_override` may have been applied \
         to derive this key from the on-disk processor_class)",
      processor_class.as_str(),
    )));
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
  let loaded = LoadedVlmModel::new(config, config_json, weights);
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

  Ok(LoadedVlmContext::new(
    model,
    tokenizer,
    processor,
    loaded.config,
  ))
}

#[cfg(test)]
mod tests;
