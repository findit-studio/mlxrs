//! Local model **load factory** + a [`model_type`](crate::lm::load::Config)
//! → constructor [`ModelTypeRegistry`], ported from the local-path slice of
//! [`mlx_lm.utils`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
//! (`load` / `load_model` / `load_config` / `_get_classes`) and
//! `mlx-swift-lm`'s `MLXLMCommon` (`ModelFactory` / `ModelConfiguration` /
//! `ModelTypeRegistry` / `BaseConfiguration`).
//!
//! This layer sits **on top of** [`crate::lm::load`] (which already ports the
//! arch-agnostic `config.json` parse + weight discovery + tokenizer build) and
//! adds the three pieces that turn a directory into a constructed model:
//!
//! - [`ModelConfiguration`] — the model's *location* (mlx-swift-lm's
//!   `ModelConfiguration.Identifier`). An [`Identifier::Id`] (an
//!   org/name string) is treated as a **local path** (there is **no**
//!   Hugging Face Hub download — the network slice of `_download` /
//!   `snapshot_download` is deliberately out of scope), exactly the
//!   `path_or_hf_repo` already-local branch of `mlx_lm.utils._download`. An
//!   optional [`ModelConfiguration::tokenizer_source`] lets the tokenizer load
//!   from a different local directory (mlx-swift-lm's `tokenizerSource`); when
//!   `None` the model directory is reused.
//! - [`ModelTypeRegistry`] — `model_type: &str` → a [`ModelConstructor`]
//!   closure, mirroring mlx-swift-lm's
//!   `ModelTypeRegistry<T>.creators: [String: (Data) throws -> T]` and
//!   replacing `_get_classes`' Python `importlib.import_module(
//!   "mlx_lm.models.{model_type}")` dynamic dispatch with an explicit,
//!   compile-time-safe registration table. Per-model architectures are **out
//!   of scope** (the project's no-model-arch rule), so the registry is the
//!   *extension point* future per-usecase model PRs register their constructor
//!   into — this PR ships the seam, not the architectures.
//! - [`load()`] — the end-to-end entry: resolve the directory → parse the
//!   `config.json` `model_type` + load the weights + build the tokenizer
//!   (all via [`crate::lm::load::load`]) → look the `model_type` up in the
//!   registry (after [`remap_model_type`], mirroring `MODEL_REMAPPING`) →
//!   invoke the constructor → return the `(Box<dyn Model>, Tokenizer)` pair.
//!
//! On top of that load surface sits [`ModelContext`] — a thin **owning
//! bundle** of the loaded `(model, tokenizer, config)` with ergonomic
//! convenience methods (`encode` / `decode` / `apply_chat_template` /
//! `generate` / `stream_generate`, each a thin forward to the tokenizer or
//! [`crate::lm::generate`]). It is the single-thread reduction of
//! mlx-swift-lm's `ModelContext` / `ModelContainer` — the actor concurrency of
//! the Swift `ModelContainer` is dropped because mlxrs's
//! [`Array`](crate::array::Array) is `!Send`/`!Sync` (see the
//! [`ModelContext`] type docs).
//!
//! Conventions match the rest of `lm`: every fallible step returns
//! [`Result`], recoverable failures (missing/invalid config, no weights,
//! unknown `model_type`, tokenizer load) are [`Error::Backend`] with a
//! message naming the cause, borrows are preferred over clones, and there is
//! no implicit eval (the weight `Array`s are handed to the constructor lazily,
//! exactly as [`crate::lm::load::load`] returns them).

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

#[cfg(test)]
use crate::error::RankMismatchPayload;
use crate::{
  error::{Error, MissingKeyPayload, Result},
  lm::{
    generate::{GenConfig, GenerationResponse, GenerationStats},
    load::{self, Config, Weights},
    model::Model,
  },
  tokenizer::Tokenizer,
};

/// Architecture-id remapping, mirroring `mlx_lm.utils.MODEL_REMAPPING`:
/// some checkpoints declare a `model_type` that is an alias for another
/// architecture's implementation (e.g. `"mistral"` is served by the `"llama"`
/// model). [`remap_model_type`] applies this before a [`ModelTypeRegistry`]
/// lookup so a registry only needs to register the *canonical* id.
///
/// Kept verbatim from `mlx_lm.utils` (the authoritative spec) so a checkpoint
/// that loads in mlx-lm dispatches to the same constructor here. Sorted by key
/// for a deterministic, reviewable table.
const MODEL_REMAPPING: &[(&str, &str)] = &[
  ("falcon_mamba", "mamba"),
  ("iquestcoder", "llama"),
  ("joyai_llm_flash", "deepseek_v3"),
  ("kimi_k2", "deepseek_v3"),
  ("llava", "mistral3"),
  ("minimax_m2", "minimax"),
  ("mistral", "llama"),
  ("phi-msft", "phixtral"),
  ("qwen2_5_vl", "qwen2_vl"),
];

/// Canonicalize a checkpoint's `model_type` via the `MODEL_REMAPPING` table,
/// mirroring `mlx_lm.utils._get_classes`'s
/// `model_type = MODEL_REMAPPING.get(model_type, model_type)`. An id with no
/// alias is returned unchanged.
pub fn remap_model_type(model_type: &str) -> &str {
  MODEL_REMAPPING
    .iter()
    .find_map(|&(from, to)| (from == model_type).then_some(to))
    .unwrap_or(model_type)
}

/// Everything [`crate::lm::load::load`] resolved from a model directory,
/// handed to a [`ModelConstructor`] so it can assemble (and, if
/// [`Config::quantization`] is set, quantize) a concrete architecture without
/// re-reading the directory.
///
/// Borrowing — the constructor gets `&LoadedModel`; it reads the typed
/// [`Config`] (and, for keys outside that typed subset, the verbatim
/// [`config_json`](Self::config_json) text — the analogue of mlx-swift-lm
/// passing the raw `config.json` `Data` to each model's `Codable` init) and
/// takes the weight [`Array`](crate::array::Array)s it needs out of
/// [`weights`](Self::weights) **by reference** (no implicit eval; mlx `Array`
/// is a cheap refcounted handle, so an arch clones only the handles it keeps).
#[non_exhaustive]
pub struct LoadedModel {
  /// The typed `config.json` subset (mlx-lm's `config` dict), with the
  /// generation-config eos override already applied (see
  /// [`crate::lm::load::load`]).
  pub config: Config,
  /// The verbatim `config.json` body, for model-specific keys outside the
  /// typed [`Config`] subset (the analogue of mlx-swift-lm handing each
  /// model's `Codable` init the raw config `Data`). Always the bytes the
  /// typed [`config`](Self::config) was parsed from.
  pub config_json: String,
  /// The merged, name → [`Array`](crate::array::Array) weight map
  /// (mlx-lm's `weights` dict). Keys are verbatim — the constructor applies
  /// any `sanitize`/remap itself.
  pub weights: Weights,
}

/// A registered model constructor: assemble a [`Model`] from the
/// already-resolved [`LoadedModel`] (parsed config + raw config JSON +
/// weights).
///
/// Mirrors mlx-swift-lm's `ModelTypeRegistry` creator
/// `(Data) throws -> T` — but receives the *already-loaded* weights too (so a
/// per-usecase architecture never re-globs/re-reads the directory) and returns
/// a [`Result`] (Rust's `throws`). `Send + Sync` so a registry can be shared
/// across threads (e.g. a `static` shared registry, as mlx-swift-lm's
/// `LLMTypeRegistry.shared` is). The constructor itself does **no** I/O; the
/// directory was already read by [`load()`].
pub type ModelConstructor =
  Box<dyn Fn(&LoadedModel) -> Result<Box<dyn Model>> + Send + Sync + 'static>;

/// A `model_type: String` → [`ModelConstructor`] table, the load factory's
/// architecture **extension point**.
///
/// Mirrors mlx-swift-lm's `ModelTypeRegistry<T>` (and replaces
/// `mlx_lm.utils._get_classes`' `importlib` dynamic dispatch with an explicit
/// registration table). Per-model architectures are out of scope for this PR,
/// so the registry starts [`empty`](Self::new); future per-usecase model PRs
/// call [`register`](Self::register) (or build one with
/// [`with`](Self::with)) to plug their architecture in. A `model_type` is
/// canonicalized via [`remap_model_type`] on both registration and lookup, so
/// callers register the *canonical* id and any alias resolves to it.
#[derive(Default)]
pub struct ModelTypeRegistry {
  creators: HashMap<String, ModelConstructor>,
}

impl ModelTypeRegistry {
  /// An empty registry (mlx-swift-lm's `ModelTypeRegistry()` — no creators).
  pub fn new() -> Self {
    Self {
      creators: HashMap::new(),
    }
  }

  /// Register `constructor` for `model_type` (canonicalized via
  /// [`remap_model_type`]), mirroring mlx-swift-lm's
  /// `registerModelType(_:creator:)`. A re-registration of the same
  /// (canonical) id replaces the previous constructor (last-writer-wins, as
  /// the Swift dictionary assignment does) and returns the displaced one.
  pub fn register(
    &mut self,
    model_type: &str,
    constructor: ModelConstructor,
  ) -> Option<ModelConstructor> {
    self
      .creators
      .insert(remap_model_type(model_type).to_owned(), constructor)
  }

  /// Builder form of [`register`](Self::register) for assembling a registry
  /// in one expression (the analogue of mlx-swift-lm's
  /// `ModelTypeRegistry(creators:)` init).
  #[must_use]
  pub fn with(mut self, model_type: &str, constructor: ModelConstructor) -> Self {
    self.register(model_type, constructor);
    self
  }

  /// `true` if a constructor is registered for `model_type` (after
  /// [`remap_model_type`]).
  pub fn contains(&self, model_type: &str) -> bool {
    self.creators.contains_key(remap_model_type(model_type))
  }

  /// Construct a [`Model`] for `loaded`'s [`Config::model_type`], mirroring
  /// mlx-swift-lm's `createModel(configuration:modelType:)`. The id is
  /// canonicalized via [`remap_model_type`]; an unregistered id is a
  /// recoverable [`Error::Backend`] (mlx-swift-lm's
  /// `ModelFactoryError.unsupportedModelType`, mlx-lm's
  /// `ValueError("Model type … not supported.")`).
  pub fn create(&self, loaded: &LoadedModel) -> Result<Box<dyn Model>> {
    let model_type = remap_model_type(loaded.config.model_type());
    let constructor = self.creators.get(model_type).ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "ModelTypeRegistry::create: unsupported model type (no constructor registered; register one via ModelTypeRegistry::register)",
        loaded.config.model_type(),
      ))
    })?;
    constructor(loaded)
  }
}

/// Which local directory holds a model (mlx-swift-lm's
/// `ModelConfiguration.Identifier`).
///
/// **No network**: an [`Id`](Self::Id) (an org/name string) is treated as a
/// *local path* — the already-local branch of `mlx_lm.utils._download`
/// (`Path(path_or_hf_repo)` when `model_path.exists()`); the
/// `snapshot_download` Hub fetch is out of scope. So both variants resolve to
/// a [`Path`] without any I/O beyond the later directory read in [`load()`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Identifier {
  /// A model identifier (`org/name`) treated as a **local path** (no Hub
  /// download). mlx-swift-lm's `.id(String, revision:)`, restricted to the
  /// local-path slice (the `revision` is meaningless without a network fetch
  /// and is intentionally dropped).
  Id(String),
  /// An explicit local directory. mlx-swift-lm's `.directory(URL)`.
  Directory(PathBuf),
}

impl Identifier {
  /// The local directory this identifier names. Both variants are local (see
  /// the type docs), so this is infallible and does **no** I/O — the
  /// directory's existence is validated when [`load()`] reads it.
  pub fn directory(&self) -> &Path {
    match self {
      Identifier::Id(id) => Path::new(id),
      Identifier::Directory(dir) => dir,
    }
  }
}

/// Where to load a model and (optionally) its tokenizer from, ported from the
/// **local-path slice** of mlx-swift-lm's `ModelConfiguration`.
///
/// Behavioural metadata that mlx-swift-lm's `ModelConfiguration` carries
/// (`defaultPrompt` / `extraEOSTokens` / `toolCallFormat`) is intentionally
/// **not** modeled here: the eos set is already resolved from
/// `config.json` + `generation_config.json` by [`crate::lm::load::load`]
/// (and lives on the [`Tokenizer`]), and prompt/tool-format are
/// chat-pipeline concerns layered above this loader. This type is purely the
/// *source location* (model dir + optional separate tokenizer dir).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfiguration {
  /// The model's location ([`Identifier::Directory`] or an
  /// [`Identifier::Id`] treated as a local path).
  pub id: Identifier,
  /// An optional **separate local directory** for the tokenizer
  /// (mlx-swift-lm's `tokenizerSource`). `None` ⇒ load the tokenizer from the
  /// model directory (the common case). Like [`Identifier`] this is
  /// local-only — no Hub download.
  pub tokenizer_source: Option<PathBuf>,
}

impl ModelConfiguration {
  /// A configuration for a model in a local `directory` (tokenizer loaded
  /// from the same directory). mlx-swift-lm's
  /// `ModelConfiguration(directory:)`.
  pub fn from_directory(directory: impl Into<PathBuf>) -> Self {
    Self {
      id: Identifier::Directory(directory.into()),
      tokenizer_source: None,
    }
  }

  /// A configuration for a model `id` (`org/name`) treated as a **local
  /// path** — *no* Hub download (see [`Identifier::Id`]). mlx-swift-lm's
  /// `ModelConfiguration(id:)`, restricted to the local-path slice.
  pub fn from_id(id: impl Into<String>) -> Self {
    Self {
      id: Identifier::Id(id.into()),
      tokenizer_source: None,
    }
  }

  /// Use a separate local `directory` for the tokenizer (mlx-swift-lm's
  /// `tokenizerSource`). Builder form for the rare split-tokenizer case.
  #[must_use]
  pub fn with_tokenizer_source(mut self, directory: impl Into<PathBuf>) -> Self {
    self.tokenizer_source = Some(directory.into());
    self
  }

  /// The resolved local model directory. Local-only, so infallible and
  /// I/O-free (mlx-swift-lm's `modelDirectory`, minus the unresolved-remote
  /// throw that cannot occur without a network identifier).
  pub fn model_directory(&self) -> &Path {
    self.id.directory()
  }

  /// The resolved local tokenizer directory:
  /// [`tokenizer_source`](Self::tokenizer_source) if set, else the model
  /// directory (mlx-swift-lm's `tokenizerDirectory` fallback).
  pub fn tokenizer_directory(&self) -> &Path {
    match &self.tokenizer_source {
      Some(dir) => dir,
      None => self.model_directory(),
    }
  }
}

/// The product of [`load()`]: a constructed [`Model`] plus the
/// [`Tokenizer`] and the parsed [`Config`], the analogue of mlx-swift-lm's
/// `ModelContext` (restricted to the text-LM essentials — no
/// `UserInputProcessor`, which is a chat-pipeline concern above this loader).
#[non_exhaustive]
pub struct LoadedModelContext {
  /// The constructed model (from the registry's constructor).
  pub model: Box<dyn Model>,
  /// The model's tokenizer, built from the (optionally separate) tokenizer
  /// directory with the resolved eos set.
  pub tokenizer: Tokenizer,
  /// The parsed `config.json` subset, returned for callers that need the
  /// architecture metadata (mlx-lm's `load(return_config=True)`).
  pub config: Config,
}

/// Load a model + tokenizer from a local [`ModelConfiguration`], dispatching
/// to `registry` on the checkpoint's `model_type`.
///
/// The end-to-end port of `mlx_lm.utils.load` restricted to the local-path,
/// no-network surface (and mlx-swift-lm's `GenericModelFactory._load`). The
/// orchestration order is chosen so the *cheap, recoverable* failures come
/// first — nothing heavy (weights, tokenizer) is touched until the checkpoint
/// is known to be loadable:
///
/// 1. Resolve the model directory ([`ModelConfiguration::model_directory`] —
///    local, no Hub download) and read `config.json` **once** via
///    [`crate::lm::load::load_config`], yielding both the typed [`Config`]
///    (with the `generation_config.json` eos override applied) and the
///    verbatim JSON body — the *same bytes* the typed config was parsed from,
///    so the constructor's typed [`Config`] and raw
///    [`config_json`](LoadedModel::config_json) can never diverge across two
///    opens.
/// 2. **Validate the `model_type` is registered** (after [`remap_model_type`])
///    *before* loading anything heavy: an unsupported checkpoint is a cheap,
///    recoverable [`Error::Backend`] here, with no weight/tokenizer I/O —
///    mlx-lm's `ValueError("Model type … not supported.")` /
///    mlx-swift-lm's `unsupportedModelType`.
/// 3. Select the tokenizer directory FIRST
///    ([`tokenizer_source`](ModelConfiguration::tokenizer_source) if set, else
///    the model directory — mlx-swift-lm's `tokenizerDirectory`).
/// 4. Discover and merge the weights from the model directory via
///    [`crate::lm::load::load_weights`].
/// 5. Build the [`Tokenizer`] EXACTLY ONCE from the selected directory (with
///    the eos set resolved on the [`Config`] from step 1).
/// 6. Construct the model via `registry` on the [`LoadedModel`] (parsed config
///    + raw JSON + weights) and return it with the tokenizer and config.
///
/// Per-model construction is the registry's job (this PR ships no
/// architectures). No implicit eval — the weights reach the constructor lazily.
pub fn load(
  configuration: &ModelConfiguration,
  registry: &ModelTypeRegistry,
) -> Result<LoadedModelContext> {
  let model_dir = configuration.model_directory();

  // (1) Read config.json ONCE: typed Config (+ generation_config eos
  // override) AND the verbatim JSON body, from the same bytes. The constructor
  // may need model-specific keys outside the typed subset (mlx-swift-lm hands
  // each model the raw config `Data`); reading once means they can never come
  // from two different on-disk versions of the file.
  let (config, config_json) = load::load_config(model_dir)?;

  // (2) Validate the (remapped) model_type is registered BEFORE loading any
  // weights or the tokenizer. An unsupported checkpoint — the common case,
  // since per-model architectures are out of scope and the registry is
  // normally empty — is a cheap, recoverable error here, never paying for
  // weight/tokenizer I/O (and never surfacing a weight error in place of the
  // recoverable unsupported-model one).
  if !registry.contains(config.model_type()) {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "ModelTypeRegistry::create: unsupported model type (no constructor registered; register one via ModelTypeRegistry::register)",
      config.model_type(),
    )));
  }

  // (3) Select the tokenizer directory FIRST: the separate `tokenizer_source`
  // if set (a real split layout where the model dir has NO `tokenizer.json`),
  // else the model directory (mlx-swift-lm's `tokenizerDirectory`).
  let tokenizer_dir = configuration.tokenizer_directory();

  // (4) Discover/merge the weights from the model directory.
  let weights = load::load_weights(model_dir)?;

  // (5) Build the tokenizer EXACTLY ONCE from the selected directory, through
  // the shared eos-resolution path (the eos set already resolved on `config`).
  let tokenizer = load::load_tokenizer(tokenizer_dir, &config)?;

  // (6) Construct via the registry (already validated as registered in step 2).
  let loaded = LoadedModel {
    config,
    config_json,
    weights,
  };
  let model = registry.create(&loaded)?;

  Ok(LoadedModelContext {
    model,
    tokenizer,
    config: loaded.config,
  })
}

/// An owning bundle of a loaded `(model, tokenizer, config)` with ergonomic
/// convenience entry points — the single-thread reduction of mlx-swift-lm's
/// `ModelContext` / `ModelContainer`.
///
/// # Relationship to [`LoadedModelContext`]
///
/// [`load()`] returns a [`LoadedModelContext`] — the loader's plain *product*
/// struct (public fields, no behavior). [`ModelContext`] is the **owning
/// context** layered on top: it takes the same three values by value and adds
/// the convenience surface (`encode` / `decode` / `apply_chat_template` /
/// `generate` / `stream_generate`) so a caller need not thread `&model`,
/// `&tokenizer` and a hand-built `CacheConfig` through every `lm::generate`
/// call. Build one straight from a load with [`From<LoadedModelContext>`]
/// (`load(..)?.into()`) or the [`ModelContext::load`] one-call convenience.
///
/// # Actor → single-thread divergence (intentional)
///
/// mlx-swift-lm's `ModelContainer` is a Swift **`actor`** — it exists to share
/// one model safely across threads, serializing access to the non-`Sendable`
/// `MLXArray`s inside. mlxrs's [`Array`](crate::array::Array) is deliberately
/// `!Send`/`!Sync` (single-thread, matching MLX's own threading model), so a
/// faithful actor port is **inapplicable**: there is no cross-thread sharing
/// to serialize. [`ModelContext`] therefore ports the *logic* of
/// `ModelContext` / `ModelContainer` — the `(model, tokenizer, config)`
/// ownership plus the convenience entry points — as a plain single-thread
/// owning struct, dropping only the actor concurrency machinery (the
/// `SerialAccessContainer`, the `perform` closure isolation, the `Sendable` /
/// `sending` annotations). This mirrors how the project already handles the
/// other Swift `actor` types it ports.
///
/// # API conventions
///
/// Matches the rest of `lm`: every fallible call returns [`Result`]; accessors
/// ([`model`](Self::model) / [`tokenizer`](Self::tokenizer) /
/// [`config`](Self::config)) borrow and never eval (no implicit eval — the
/// owned [`Array`](crate::array::Array) weights are touched only by an explicit
/// `generate` forward pass). The convenience methods are **thin forwards** —
/// `encode` / `decode` / `apply_chat_template` defer to the [`Tokenizer`],
/// `generate` / `stream_generate` defer to [`crate::lm::generate`] — they
/// re-implement nothing.
///
/// The generation methods take `&self`, not `&mut self`: a [`ModelContext`]
/// owns **no** KV cache (model weights are immutable after load — see the
/// [`Model`] trait — and [`crate::lm::generate`] takes `&M`). Each `generate`
/// / `stream_generate` call builds a *fresh* per-call cache (sized from
/// [`Config::num_hidden_layers`] / [`Config::sliding_window`] via
/// [`crate::lm::cache::make_prompt_cache`]) that the call consumes, so the
/// context is never mutated. A persistent multi-turn cache is a chat-session
/// concern layered above this bundle (mlx-swift-lm's `ChatSession`), not part
/// of the `ModelContext` reduction.
#[non_exhaustive]
pub struct ModelContext {
  /// The loaded model (mlx-swift-lm `ModelContext.model`).
  model: Box<dyn Model>,
  /// The model's tokenizer (mlx-swift-lm `ModelContext.tokenizer`).
  tokenizer: Tokenizer,
  /// The parsed `config.json` subset (the architecture metadata mlx-swift-lm
  /// keeps on `ModelContext.configuration`, restricted to the typed
  /// [`Config`] the load surface produces).
  config: Config,
}

impl ModelContext {
  /// Bundle an already-loaded `(model, tokenizer, config)` triple.
  ///
  /// The direct constructor — mlx-swift-lm's `ModelContext.init(...)`. Most
  /// callers instead use [`From<LoadedModelContext>`] (`load(..)?.into()`) or
  /// [`load`](Self::load); this is for callers assembling the triple by hand
  /// (e.g. tests, or a model built without the [`load()`] factory path).
  pub fn new(model: Box<dyn Model>, tokenizer: Tokenizer, config: Config) -> Self {
    Self {
      model,
      tokenizer,
      config,
    }
  }

  /// Load a model + tokenizer from a local [`ModelConfiguration`] and bundle
  /// the result into a [`ModelContext`] — the one-call convenience.
  ///
  /// Equivalent to `load(configuration, registry).map(ModelContext::from)`,
  /// the analogue of mlx-swift-lm's `loadModelContainer` (which loads a
  /// `ModelContext` and wraps it in the owning `ModelContainer`). The same
  /// recoverable [`Error::Backend`]s [`load()`] returns (missing/invalid
  /// config, no weights, unknown `model_type`, tokenizer load) propagate
  /// unchanged.
  pub fn load(configuration: &ModelConfiguration, registry: &ModelTypeRegistry) -> Result<Self> {
    load(configuration, registry).map(Self::from)
  }

  /// The bundled model (mlx-swift-lm `ModelContainer.perform { $0.model }`).
  pub fn model(&self) -> &dyn Model {
    self.model.as_ref()
  }

  /// The bundled tokenizer (mlx-swift-lm `ModelContainer.tokenizer`).
  pub fn tokenizer(&self) -> &Tokenizer {
    &self.tokenizer
  }

  /// The bundled parsed `config.json` subset (mlx-swift-lm
  /// `ModelContainer.configuration`).
  pub fn config(&self) -> &Config {
    &self.config
  }

  /// Decompose the bundle back into its owned `(model, tokenizer, config)`
  /// triple (the inverse of [`new`](Self::new) — mlx-swift-lm's `consuming`
  /// `ModelContext` move-out).
  pub fn into_parts(self) -> (Box<dyn Model>, Tokenizer, Config) {
    (self.model, self.tokenizer, self.config)
  }

  /// Encode `text` to token ids — a thin forward to
  /// [`Tokenizer::encode`](crate::tokenizer::Tokenizer::encode)
  /// (mlx-swift-lm's `ModelContainer.encode(_:)`).
  ///
  /// `add_special_tokens` is forwarded verbatim (mlx-swift-lm's bare `encode`
  /// uses the tokenizer default; this exposes the flag so callers keep full
  /// control).
  pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
    self.tokenizer.encode(text, add_special_tokens)
  }

  /// Decode token `ids` back to a string — a thin forward to
  /// [`Tokenizer::decode`](crate::tokenizer::Tokenizer::decode)
  /// (mlx-swift-lm's `ModelContainer.decode(tokenIds:)`).
  pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
    self.tokenizer.decode(ids, skip_special_tokens)
  }

  /// Render the chat template for `messages` — a thin forward to
  /// [`Tokenizer::apply_chat_template`](crate::tokenizer::Tokenizer::apply_chat_template)
  /// (mlx-swift-lm's `ModelContainer.applyChatTemplate(messages:)`).
  ///
  /// Returns the rendered prompt **string**; pair it with
  /// [`encode`](Self::encode) (or use [`apply_chat_template_ids`](Self::apply_chat_template_ids))
  /// to get token ids. All arguments forward verbatim — see the tokenizer
  /// method for `add_generation_prompt` / `continue_final_message` semantics
  /// (and their mutual exclusivity).
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template(
    &self,
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
    continue_final_message: bool,
    additional_context: Option<&serde_json::Value>,
  ) -> Result<String> {
    self.tokenizer.apply_chat_template(
      messages,
      tools,
      add_generation_prompt,
      continue_final_message,
      additional_context,
    )
  }

  /// Render the chat template and tokenize it in one step — a thin forward to
  /// [`Tokenizer::apply_chat_template_ids`](crate::tokenizer::Tokenizer::apply_chat_template_ids)
  /// (the `tokenize: true` form of mlx-swift-lm's `applyChatTemplate`).
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template_ids(
    &self,
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
    continue_final_message: bool,
    additional_context: Option<&serde_json::Value>,
  ) -> Result<Vec<u32>> {
    self.tokenizer.apply_chat_template_ids(
      messages,
      tools,
      add_generation_prompt,
      continue_final_message,
      additional_context,
    )
  }

  /// A [`CacheConfig`](crate::lm::cache::CacheConfig) for this model's KV
  /// cache, derived from the bundled [`Config`].
  ///
  /// `num_hidden_layers` and `sliding_window` are the two `config.json` keys
  /// [`make_prompt_cache`](crate::lm::cache::make_prompt_cache) needs; both
  /// are carried on [`Config`]. Used by [`generate`](Self::generate) /
  /// [`stream_generate`](Self::stream_generate) to size each fresh per-call
  /// cache. `num_hidden_layers` is an `i32` on [`Config`]; a negative or
  /// absurd value never reaches here on a real checkpoint, and a `0`/negative
  /// count simply yields an empty cache.
  fn cache_config(&self) -> crate::lm::cache::CacheConfig {
    crate::lm::cache::CacheConfig {
      num_hidden_layers: self.config.num_hidden_layers.max(0) as usize,
      sliding_window: self.config.sliding_window,
    }
  }

  /// Generate a complete response string for the already-encoded `prompt` —
  /// a thin forward to [`crate::lm::generate::generate`].
  ///
  /// The owning-context counterpart of mlx-swift-lm's
  /// `ModelContainer.generate(...)` collected to a string. Builds a **fresh**
  /// KV cache for this call (sized from [`Config::num_hidden_layers`] /
  /// [`Config::sliding_window`] via
  /// [`make_prompt_cache`](crate::lm::cache::make_prompt_cache)) and hands the
  /// bundled model + tokenizer to [`crate::lm::generate`];
  /// nothing on `&self` is mutated (see the type docs on why this is
  /// `&self`). `prompt` is the encoded prompt ids — encode a `&str` via
  /// [`encode`](Self::encode) or a chat prompt via
  /// [`apply_chat_template_ids`](Self::apply_chat_template_ids) first.
  ///
  /// Returns `(text, stats)` exactly as [`crate::lm::generate::generate`]; an
  /// underlying step error propagates as `Err`.
  pub fn generate(&self, prompt: &[u32], cfg: GenConfig) -> Result<(String, GenerationStats)> {
    let cache = crate::lm::cache::make_prompt_cache(&self.cache_config());
    crate::lm::generate::generate(self.model.as_ref(), &self.tokenizer, prompt, cache, cfg)
  }

  /// Stream the response for the already-encoded `prompt` as an iterator of
  /// [`GenerationResponse`]s — a thin forward to
  /// [`crate::lm::generate::stream_generate`].
  ///
  /// The owning-context counterpart of mlx-swift-lm's
  /// `ModelContainer.generate(...)` `AsyncStream`. Like [`generate`](Self::generate)
  /// it builds a fresh per-call cache; the returned iterator borrows `&self`
  /// for the model + tokenizer (so it cannot outlive the context). `prompt`
  /// is the encoded prompt ids.
  pub fn stream_generate(
    &self,
    prompt: &[u32],
    cfg: GenConfig,
  ) -> impl Iterator<Item = Result<GenerationResponse>> + '_ {
    let cache = crate::lm::cache::make_prompt_cache(&self.cache_config());
    crate::lm::generate::stream_generate(self.model.as_ref(), &self.tokenizer, prompt, cache, cfg)
  }
}

impl From<LoadedModelContext> for ModelContext {
  /// Wrap the loader's product struct into the owning convenience context —
  /// the analogue of mlx-swift-lm's `GenericModelFactory._wrap` (which boxes a
  /// freshly-`_load`ed `ModelContext` into the owning `ModelContainer`).
  fn from(loaded: LoadedModelContext) -> Self {
    Self::new(loaded.model, loaded.tokenizer, loaded.config)
  }
}

#[cfg(test)]
mod tests;
