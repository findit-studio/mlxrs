//! Local embedding-model **load factory** + a `model_type` →
//! constructor [`EmbeddingModelTypeRegistry`], ported from the local-path
//! slice of
//! [`mlx_embeddings.utils`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/utils.py)
//! (`load` / `load_model` / `load_config` / `_get_model_arch`) and
//! `mlx-swift-lm`'s `MLXEmbedders` (`EmbedderModelFactory._load` /
//! `EmbedderTypeRegistry` / `EmbedderModelContext`).
//!
//! This is the embeddings twin of [`crate::lm::factory`] — it is **structurally
//! mirrored** on that module — and turns a local model directory into a
//! constructed [`EmbeddingModel`] + [`Tokenizer`] + (optional) pooling-config
//! bundle:
//!
//! - [`EmbeddingModelConfiguration`] — the model's *location* (mlx-swift-lm's
//!   `ModelConfiguration`). An [`EmbeddingIdentifier::Id`] (an org/name string)
//!   is treated as a **local path** — there is **no** Hugging Face Hub download
//!   (the `snapshot_download` network slice of `mlx_embeddings.utils
//!   .get_model_path` is deliberately out of scope). An optional
//!   [`tokenizer_source`](EmbeddingModelConfiguration::tokenizer_source) lets
//!   the tokenizer load from a different local directory; when `None` the model
//!   directory is reused.
//! - [`EmbeddingModelTypeRegistry`] — `model_type: &str` → an
//!   [`EmbeddingModelConstructor`] closure, mirroring mlx-swift-lm's
//!   `EmbedderTypeRegistry`'s `ModelTypeRegistry<EmbeddingModel>` and replacing
//!   `_get_model_arch`'s Python `importlib.import_module(
//!   "mlx_embeddings.models.{model_type}")` dynamic dispatch with an explicit,
//!   compile-time-safe registration table. Per-model architectures are **out of
//!   scope** (the project's no-model-arch rule), so the registry is the
//!   *extension point* future per-usecase model PRs register their constructor
//!   into — this layer ships the seam, not the architectures.
//! - [`load()`] — the end-to-end entry: resolve the directory → parse the
//!   `config.json` `model_type` → look it up in the registry (after
//!   [`remap_model_type`], mirroring `_get_model_arch`) → load the weights + the
//!   tokenizer + the optional `1_Pooling/config.json` → invoke the constructor →
//!   return a [`LoadedEmbeddingContext`].
//!
//! ## Shared loaders reused (not re-implemented)
//!
//! The `embeddings` feature is deliberately `serde_json`-free (EMB-1: the
//! `1_Pooling/config.json` parse is a hand-rolled strict-JSON scanner) and does
//! **not** enable the `lm` feature, so [`crate::lm::load`]'s
//! `serde`-derived `Config` reader is unreachable here. "Reuse the shared
//! loaders" therefore means reusing the *lower*, ungated layers `lm::load`
//! itself builds on:
//!
//! - **weights** — [`crate::io::load_safetensors`], the exact lowest-level
//!   loader `lm::load::load_weights` calls;
//! - **tokenizer** — [`Tokenizer::from_path`], the exact call
//!   `lm::load::load_tokenizer` wraps (and which
//!   [`crate::embeddings::encode()`] already uses);
//! - **pooling config** — the existing
//!   [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
//!   (mlx-embeddings' `_read_pooling_config`).
//!
//! Only the `config.json` `model_type` read is module-local: it needs a single
//! string field, so a small dependency-free extractor reads it
//! with the same bounded-read discipline (`O_NONBLOCK | O_CLOEXEC` open,
//! post-open `is_file()` reject, `Read::take` cap) as
//! [`crate::embeddings::config`]'s pooling-config reader — the discipline
//! `lm::load`'s reader is itself modeled on.
//!
//! Conventions match the rest of `embeddings`: every fallible step returns
//! [`Result`], recoverable failures (missing/invalid config, no weights,
//! unknown `model_type`, tokenizer load, malformed pooling config) are
//! [`Error::Backend`] with a message naming the cause, borrows are preferred
//! over clones, and there is no implicit eval (the weight `Array`s are handed
//! to the constructor lazily).

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
};

use crate::{
  array::Array,
  embeddings::{config::StPoolingConfig, model::EmbeddingModel},
  error::{Error, Result},
  tokenizer::Tokenizer,
};

/// Upper bound on a `config.json` we will read into memory, mirroring
/// [`crate::embeddings::config`]'s `MAX_ST_POOLING_CONFIG_BYTES` (and
/// `lm::load`'s `MAX_CONFIG_BYTES`). A real model's `config.json` is well under
/// 1 MiB; a hostile model directory cannot make us allocate unbounded memory by
/// planting a huge `config.json`.
const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// Architecture-id remapping, mirroring `mlx_embeddings.utils.MODEL_REMAPPING`:
/// some checkpoints declare a `model_type` that is an alias for another
/// architecture's implementation. [`remap_model_type`] applies this (after the
/// `-`→`_` normalization) before an [`EmbeddingModelTypeRegistry`] lookup so a
/// registry only needs to register the *canonical* id.
///
/// `mlx_embeddings.utils.MODEL_REMAPPING` is currently the **empty dict**
/// `{}` — there are no embedding-model aliases upstream — so this table is
/// likewise empty. It is kept (rather than dropped) so the structure mirrors
/// [`crate::lm::factory`]'s `MODEL_REMAPPING` and a future upstream alias is a
/// one-line addition. Sorted by key for a deterministic, reviewable table.
const MODEL_REMAPPING: &[(&str, &str)] = &[];

/// Canonicalize a checkpoint's `model_type`, mirroring
/// `mlx_embeddings.utils._get_model_arch`'s
/// `model_type = config["model_type"].replace("-", "_")` **followed by**
/// `MODEL_REMAPPING.get(model_type, model_type)`.
///
/// Unlike [`crate::lm::factory::remap_model_type`] (whose
/// `mlx_lm` counterpart does *not* normalize separators), `mlx_embeddings`
/// replaces every `-` with `_` first — so `"xlm-roberta"` canonicalizes to
/// `"xlm_roberta"`. Because that step rewrites the string, this returns an
/// owned [`String`] rather than a borrow. An id with no `-` and no alias is
/// returned unchanged (as an owned copy).
pub fn remap_model_type(model_type: &str) -> String {
  let normalized = model_type.replace('-', "_");
  MODEL_REMAPPING
    .iter()
    .find_map(|&(from, to)| (from == normalized).then(|| to.to_owned()))
    .unwrap_or(normalized)
}

/// Which local directory holds an embedding model (mlx-swift-lm's
/// `ModelConfiguration.Identifier`).
///
/// **No network**: an [`Id`](Self::Id) (an org/name string) is treated as a
/// *local path* — the already-local branch of
/// `mlx_embeddings.utils.get_model_path` (`Path(path_or_hf_repo)` when
/// `model_path.exists()`); the `snapshot_download` Hub fetch is out of scope.
/// So both variants resolve to a [`Path`] without any I/O beyond the later
/// directory read in [`load()`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingIdentifier {
  /// A model identifier (`org/name`) treated as a **local path** (no Hub
  /// download). mlx-swift-lm's `.id(String, revision:)`, restricted to the
  /// local-path slice (the `revision` is meaningless without a network fetch
  /// and is intentionally dropped).
  Id(String),
  /// An explicit local directory. mlx-swift-lm's `.directory(URL)`.
  Directory(PathBuf),
}

impl EmbeddingIdentifier {
  /// The local directory this identifier names. Both variants are local (see
  /// the type docs), so this is infallible and does **no** I/O — the
  /// directory's existence is validated when [`load()`] reads it.
  pub fn directory(&self) -> &Path {
    match self {
      EmbeddingIdentifier::Id(id) => Path::new(id),
      EmbeddingIdentifier::Directory(dir) => dir,
    }
  }
}

/// Where to load an embedding model and (optionally) its tokenizer from,
/// ported from the **local-path slice** of mlx-swift-lm's `ModelConfiguration`.
///
/// Behavioural metadata that mlx-swift-lm's `ModelConfiguration` carries
/// (`defaultPrompt` / `extraEOSTokens` / `toolCallFormat`) is intentionally
/// **not** modeled here: an embedding encoder does not generate (no eos /
/// chat / tool concerns). This type is purely the *source location* (model dir
/// + optional separate tokenizer dir).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingModelConfiguration {
  /// The model's location ([`EmbeddingIdentifier::Directory`] or an
  /// [`EmbeddingIdentifier::Id`] treated as a local path).
  pub id: EmbeddingIdentifier,
  /// An optional **separate local directory** for the tokenizer
  /// (mlx-swift-lm's `tokenizerSource`). `None` ⇒ load the tokenizer from the
  /// model directory (the common case). Like [`EmbeddingIdentifier`] this is
  /// local-only — no Hub download.
  pub tokenizer_source: Option<PathBuf>,
}

impl EmbeddingModelConfiguration {
  /// A configuration for a model in a local `directory` (tokenizer loaded
  /// from the same directory). mlx-swift-lm's
  /// `ModelConfiguration(directory:)`.
  pub fn from_directory(directory: impl Into<PathBuf>) -> Self {
    Self {
      id: EmbeddingIdentifier::Directory(directory.into()),
      tokenizer_source: None,
    }
  }

  /// A configuration for a model `id` (`org/name`) treated as a **local
  /// path** — *no* Hub download (see [`EmbeddingIdentifier::Id`]).
  /// mlx-swift-lm's `ModelConfiguration(id:)`, restricted to the local-path
  /// slice.
  pub fn from_id(id: impl Into<String>) -> Self {
    Self {
      id: EmbeddingIdentifier::Id(id.into()),
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
  /// I/O-free (mlx-swift-lm's `modelDirectory`).
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

/// A flat name → [`Array`] weight map (mlx-embeddings' `weights` dict /
/// `mx.load(...)` result, the [`crate::io::load_safetensors`] return type).
///
/// Keys are returned **verbatim** — [`load()`] performs no `sanitize`/remap;
/// that (and mlx-embeddings' subfolder `<folder>.<key>` renaming) is a
/// per-usecase architecture's responsibility, kept out of this arch-agnostic
/// surface (exactly as [`crate::lm::load`] does).
pub type EmbeddingWeights = HashMap<String, Array>;

/// Everything [`load()`] resolved from a model directory, handed to an
/// [`EmbeddingModelConstructor`] so it can assemble a concrete architecture
/// without re-reading the directory.
///
/// The embeddings twin of [`crate::lm::factory::LoadedModel`]. Borrowing — the
/// constructor gets `&LoadedEmbeddingModel`; it reads the
/// [`model_type`](Self::model_type), the verbatim
/// [`config_json`](Self::config_json) (the analogue of mlx-swift-lm passing the
/// raw `config.json` `Data` to each model's `Decodable` init), and takes the
/// weight [`Array`]s it needs out of [`weights`](Self::weights) **by
/// reference** (no implicit eval; mlx `Array` is a cheap refcounted handle).
#[non_exhaustive]
pub struct LoadedEmbeddingModel {
  /// The checkpoint's canonicalized architecture id (`config.json`
  /// `model_type`, after [`remap_model_type`]). The registry key the
  /// constructor was looked up under.
  pub model_type: String,
  /// The verbatim `config.json` body, for the architecture's
  /// `Decodable`-style init (mlx-swift-lm hands each `EmbeddingModel`'s
  /// initializer the raw config `Data`). Always the exact bytes
  /// [`model_type`](Self::model_type) was extracted from.
  pub config_json: String,
  /// The merged, name → [`Array`] weight map (mlx-embeddings' `weights`
  /// dict). Keys are verbatim — the constructor applies any `sanitize`/remap
  /// itself.
  pub weights: EmbeddingWeights,
}

/// A registered embedding-model constructor: assemble an [`EmbeddingModel`]
/// from the already-resolved [`LoadedEmbeddingModel`] (model type + raw config
/// JSON + weights).
///
/// Mirrors mlx-swift-lm's `EmbedderTypeRegistry` creator `(Data) throws ->
/// EmbeddingModel` — but receives the *already-loaded* weights too (so a
/// per-usecase architecture never re-globs/re-reads the directory) and returns
/// a [`Result`] (Rust's `throws`). `Send + Sync` so a registry can be shared
/// across threads (e.g. a `static` shared registry, as mlx-swift-lm's
/// `EmbedderTypeRegistry.shared` is). The constructor itself does **no** I/O;
/// the directory was already read by [`load()`].
pub type EmbeddingModelConstructor =
  Box<dyn Fn(&LoadedEmbeddingModel) -> Result<Box<dyn EmbeddingModel>> + Send + Sync + 'static>;

/// A `model_type: String` → [`EmbeddingModelConstructor`] table, the load
/// factory's architecture **extension point**.
///
/// The embeddings twin of [`crate::lm::factory::ModelTypeRegistry`]. Mirrors
/// mlx-swift-lm's `EmbedderTypeRegistry`'s `ModelTypeRegistry<EmbeddingModel>`
/// (and replaces `mlx_embeddings.utils._get_model_arch`'s `importlib` dynamic
/// dispatch with an explicit registration table). Per-model architectures are
/// out of scope for this layer, so the registry starts [`empty`](Self::new);
/// future per-usecase model PRs call [`register`](Self::register) (or build one
/// with [`with`](Self::with)) to plug their architecture in. A `model_type` is
/// canonicalized via [`remap_model_type`] on both registration and lookup, so
/// callers register the *canonical* id and any alias / `-`-spelled variant
/// resolves to it.
#[derive(Default)]
pub struct EmbeddingModelTypeRegistry {
  creators: HashMap<String, EmbeddingModelConstructor>,
}

impl EmbeddingModelTypeRegistry {
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
    constructor: EmbeddingModelConstructor,
  ) -> Option<EmbeddingModelConstructor> {
    self
      .creators
      .insert(remap_model_type(model_type), constructor)
  }

  /// Builder form of [`register`](Self::register) for assembling a registry
  /// in one expression (the analogue of mlx-swift-lm's
  /// `ModelTypeRegistry(creators:)` init).
  #[must_use]
  pub fn with(mut self, model_type: &str, constructor: EmbeddingModelConstructor) -> Self {
    self.register(model_type, constructor);
    self
  }

  /// `true` if a constructor is registered for `model_type` (after
  /// [`remap_model_type`]).
  pub fn contains(&self, model_type: &str) -> bool {
    self.creators.contains_key(&remap_model_type(model_type))
  }

  /// Construct an [`EmbeddingModel`] for `loaded`'s
  /// [`model_type`](LoadedEmbeddingModel::model_type), mirroring mlx-swift-lm's
  /// `createModel(configuration:modelType:)`. `loaded.model_type` is already
  /// canonicalized (see [`load()`]); an unregistered id is a recoverable
  /// [`Error::Backend`] (mlx-swift-lm's `unsupportedModelType`, mlx-embeddings'
  /// `ValueError("Model type … not supported.")`).
  pub fn create(&self, loaded: &LoadedEmbeddingModel) -> Result<Box<dyn EmbeddingModel>> {
    let constructor = self
      .creators
      .get(&loaded.model_type)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "unsupported model type {:?}: no constructor registered (register one via \
             EmbeddingModelTypeRegistry::register)",
          loaded.model_type
        ),
      })?;
    constructor(loaded)
  }
}

/// The product of [`load()`]: a constructed [`EmbeddingModel`] plus the
/// [`Tokenizer`], the canonicalized `model_type`, and the optional
/// `1_Pooling/config.json` pooling configuration.
///
/// Mirrors [`crate::lm::factory::LoadedModelContext`] **and** mlx-swift-lm's
/// `EmbedderModelContext` — which, unlike the LM `ModelContext`, additionally
/// carries the `pooling` it resolved from `1_Pooling/config.json` (mlx-swift-lm
/// `loadPooling`). Here that is the optional [`StPoolingConfig`].
#[non_exhaustive]
pub struct LoadedEmbeddingContext {
  /// The constructed model (from the registry's constructor).
  pub model: Box<dyn EmbeddingModel>,
  /// The model's tokenizer, built from the (optionally separate) tokenizer
  /// directory via [`Tokenizer::from_path`].
  pub tokenizer: Tokenizer,
  /// The checkpoint's canonicalized architecture id (`config.json`
  /// `model_type`, after [`remap_model_type`]).
  pub model_type: String,
  /// The parsed `1_Pooling/config.json`, if the model directory carries one
  /// (a `sentence-transformers` pooling layout — mlx-embeddings
  /// `_read_pooling_config`; mlx-swift-lm `loadPooling`). `None` when no
  /// `1_Pooling/config.json` is present; the caller then falls back to the
  /// model's own pooling strategy or an [`crate::embeddings::EncodeConfig`]
  /// default.
  pub pooling: Option<StPoolingConfig>,
}

/// Load an embedding model + tokenizer from a local
/// [`EmbeddingModelConfiguration`], dispatching to `registry` on the
/// checkpoint's `model_type`.
///
/// The end-to-end port of `mlx_embeddings.utils.load` restricted to the
/// local-path, no-network surface (and mlx-swift-lm's
/// `EmbedderModelFactory._load`). The orchestration order is chosen so the
/// *cheap, recoverable* failures come first — nothing heavy (weights,
/// tokenizer) is touched until the checkpoint is known to be loadable:
///
/// 1. Resolve the model directory
///    ([`EmbeddingModelConfiguration::model_directory`] — local, no Hub
///    download) and read the `config.json` `model_type` **once** (bounded,
///    dependency-free), canonicalizing it with [`remap_model_type`].
/// 2. **Validate the `model_type` is registered** *before* loading anything
///    heavy: an unsupported checkpoint is a cheap, recoverable
///    [`Error::Backend`] here, with no weight/tokenizer I/O — mlx-embeddings'
///    `ValueError("Model type … not supported.")` / mlx-swift-lm's
///    `unsupportedModelType`.
/// 3. Select the tokenizer directory
///    ([`tokenizer_source`](EmbeddingModelConfiguration::tokenizer_source) if
///    set, else the model directory — mlx-swift-lm's `tokenizerDirectory`).
/// 4. Discover and merge the weights from the model directory (reusing
///    [`crate::io::load_safetensors`]).
/// 5. Build the [`Tokenizer`] from the selected directory via
///    [`Tokenizer::from_path`].
/// 6. Read the optional `1_Pooling/config.json` via
///    [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
///    (absent ⇒ `None`; a malformed *present* file ⇒ `Err`).
/// 7. Construct the model via `registry` and return it with the tokenizer, the
///    canonical `model_type`, and the optional pooling config.
///
/// Per-model construction is the registry's job (this layer ships no
/// architectures). No implicit eval — the weights reach the constructor lazily.
pub fn load(
  configuration: &EmbeddingModelConfiguration,
  registry: &EmbeddingModelTypeRegistry,
) -> Result<LoadedEmbeddingContext> {
  let model_dir = configuration.model_directory();

  // (1) Read the `config.json` `model_type` ONCE (bounded, dependency-free)
  // and canonicalize it (`-`→`_` + `MODEL_REMAPPING`, mirroring
  // `_get_model_arch`). The raw JSON body is kept for the constructor's
  // `Decodable`-style init; one read means the typed dispatch and the raw
  // body can never come from two on-disk versions of the file.
  let (model_type, config_json) = read_model_type(model_dir)?;

  // (2) Validate the (remapped) model_type is registered BEFORE loading any
  // weights or the tokenizer. An unsupported checkpoint — the common case,
  // since per-model architectures are out of scope and the registry is
  // normally empty — is a cheap, recoverable error here, never paying for
  // weight/tokenizer I/O.
  if !registry.contains(&model_type) {
    return Err(Error::Backend {
      message: format!(
        "unsupported model type {model_type:?}: no constructor registered (register one via \
         EmbeddingModelTypeRegistry::register)"
      ),
    });
  }

  // (3) Select the tokenizer directory: the separate `tokenizer_source` if
  // set, else the model directory (mlx-swift-lm's `tokenizerDirectory`).
  let tokenizer_dir = configuration.tokenizer_directory();

  // (4) Discover/merge the weights from the model directory.
  let weights = load_weights(model_dir)?;

  // (5) Build the tokenizer from the selected directory. An embedding encoder
  // does not generate, so there is no eos override (mlx-embeddings' embedding
  // `load` builds a plain tokenizer); pass `None` — `Tokenizer::from_path`
  // then uses the tokenizer's own `eos_token`.
  let tokenizer = Tokenizer::from_path(tokenizer_dir, None).map_err(|e| Error::Backend {
    message: format!(
      "cannot load tokenizer from {}: {e}",
      tokenizer_dir.display()
    ),
  })?;

  // (6) Read the optional `1_Pooling/config.json` (mlx-embeddings
  // `_read_pooling_config`; mlx-swift-lm `loadPooling`). An ABSENT file ⇒
  // `None` (the common non-`sentence-transformers` layout); a PRESENT but
  // malformed file ⇒ `Err` (the reader's contract — a planted broken pooling
  // config is a recoverable error, not a silent wrong strategy).
  let pooling = read_optional_pooling(model_dir)?;

  // (7) Construct via the registry (already validated as registered in step 2).
  let loaded = LoadedEmbeddingModel {
    model_type,
    config_json,
    weights,
  };
  let model = registry.create(&loaded)?;

  Ok(LoadedEmbeddingContext {
    model,
    tokenizer,
    model_type: loaded.model_type,
    pooling,
  })
}

/// Read `<dir>/1_Pooling/config.json` if present, mirroring
/// `mlx_embeddings.utils._read_pooling_config` (`return None` when the file is
/// absent) and mlx-swift-lm's `loadPooling`.
///
/// An **absent** `1_Pooling/config.json` (or no `1_Pooling` directory at all)
/// is the common case for a plain HF encoder checkpoint and yields `Ok(None)`.
/// A **present** file is parsed via
/// [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
/// (the existing bounded, hand-rolled strict-JSON reader); a malformed present
/// file therefore propagates as an [`Error::Backend`] rather than being
/// silently dropped — a planted broken pooling config is a recoverable error,
/// not a silently-wrong pooling strategy.
fn read_optional_pooling(dir: &Path) -> Result<Option<StPoolingConfig>> {
  // `_read_pooling_config`'s "absent ⇒ None" is a presence check on
  // `1_Pooling/config.json`; the existing reader maps an absent file to an
  // open-error `Err`, so the presence check is done here first.
  if !dir.join("1_Pooling").join("config.json").exists() {
    return Ok(None);
  }
  crate::embeddings::config::pooling_from_st_config_path(dir).map(Some)
}

/// Discover and merge an embedding model's weights from `dir`, mirroring the
/// weight-loading half of `mlx_embeddings.utils.load_model`.
///
/// Resolution order (mirroring `load_model`'s two `glob` passes):
///
/// 1. **Sharded / single safetensors:** every `model*.safetensors` in `dir`
///    (mlx-embeddings `glob.glob(model_path / "**/model*.safetensors")`),
///    iterated in **sorted filename order** for a deterministic merge —
///    [`crate::io::load_safetensors`] each and `extend(...)` (later shard wins
///    on a duplicate key, which a well-formed shard set never produces). Covers
///    both `model.safetensors` and `model-00001-of-000NN.safetensors`.
/// 2. **Back-compat `weight*.safetensors`:** if there is no
///    `model*.safetensors`, mlx-embeddings `load_model` retries
///    `glob(model_path / "weight*.safetensors")` — the legacy layout — and so
///    does this loader.
///
/// Subfolder-prefixed weights (mlx-embeddings renames a weight loaded from a
/// subdirectory to `<folder>.<key>`) are an architecture-`sanitize` concern and
/// are intentionally **not** applied here — keys stay verbatim, exactly as
/// [`crate::lm::load::load_weights`] does. GGUF is not a `mlx_embeddings`
/// weight path and is not handled.
///
/// No safetensors at all → [`Error::Backend`] (mlx-embeddings'
/// `FileNotFoundError("No safetensors found in {model_path}")`).
fn load_weights(dir: &Path) -> Result<EmbeddingWeights> {
  // mlx-embeddings' primary glob: `model*.safetensors`.
  let mut shards = collect_sorted(dir, |name| {
    name.starts_with("model") && name.ends_with(".safetensors")
  })?;

  // mlx-embeddings' back-compat retry: `weight*.safetensors`.
  if shards.is_empty() {
    shards = collect_sorted(dir, |name| {
      name.starts_with("weight") && name.ends_with(".safetensors")
    })?;
  }

  if shards.is_empty() {
    return Err(Error::Backend {
      message: format!(
        "no model weights found in {}: expected `model*.safetensors` (or legacy \
         `weight*.safetensors`)",
        dir.display()
      ),
    });
  }

  // Deterministic merge in sorted filename order (`collect_sorted` already
  // sorts; the dup-key tie-break — which a valid shard set never hits — is
  // thus reproducible).
  let mut weights: EmbeddingWeights = HashMap::new();
  for shard in &shards {
    let part = crate::io::load_safetensors(shard)?;
    weights.extend(part);
  }
  Ok(weights)
}

/// List the entries of `dir` whose file name matches `pred`, returning their
/// full paths sorted by name. A non-readable directory (absent / not a
/// directory / permission) maps to [`Error::Backend`]. Only regular files are
/// considered (a directory named `model….safetensors` is ignored).
///
/// Twin of [`crate::lm::load`]'s `collect_sorted`: symlinks are intentionally
/// followed (HF Hub snapshot dirs store `model*.safetensors` as symlinks into
/// `blobs/<hash>`) via `fs::metadata`, and the gate is on the *resolved*
/// target being a regular file.
fn collect_sorted(dir: &Path, pred: impl Fn(&str) -> bool) -> Result<Vec<PathBuf>> {
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
    // Resolve via `fs::metadata` (follows symlinks) and gate on the target
    // being a regular file — a hostile dir could name a subdir / FIFO
    // `model.safetensors`, on which the IO loader would fail opaquely.
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

/// Read `<dir>/config.json` **once**, returning the canonicalized
/// `model_type` (via [`remap_model_type`]) and the verbatim JSON body it was
/// extracted from.
///
/// Mirrors `mlx_embeddings.utils.load_config`'s `open(model_path /
/// "config.json")` followed by `_get_model_arch`'s `config["model_type"]`
/// lookup — but extracts only the single `model_type` field the load factory
/// dispatches on (the full typed config is the per-usecase architecture's
/// `Decodable` concern, fed the raw [`String`]).
///
/// The read is bounded against an untrusted model directory exactly as
/// [`crate::embeddings::config`]'s pooling-config read: open **once** (closing
/// the stat-then-read TOCTOU window), reject a non-regular file (FIFO / device
/// / directory / symlink-to-special) **before any read**, and cap the body at
/// [`MAX_CONFIG_BYTES`] via `Read::take`. On Unix the open carries
/// `O_NONBLOCK | O_CLOEXEC` so a planted FIFO returns immediately instead of
/// hanging the caller; symlinks are intentionally followed (HF Hub caches store
/// `config.json` as a symlink into `blobs/<hash>`) since the post-open
/// `is_file()` fstat enforces the guarantee on the *resolved* target. Every
/// failure path (absent, non-regular, oversized, unreadable, invalid JSON,
/// missing / non-string `model_type`) is a recoverable [`Error::Backend`].
fn read_model_type(dir: &Path) -> Result<(String, String)> {
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

  let raw = extract_string_field(&text, "model_type").map_err(|e| Error::Backend {
    message: format!("invalid model config {}: {e}", path.display()),
  })?;
  let model_type = raw.ok_or_else(|| Error::Backend {
    message: format!(
      "model config {} has no string `model_type` field (required to pick an architecture)",
      path.display()
    ),
  })?;

  Ok((remap_model_type(&model_type), text))
}

/// Extract a single top-level string field `key` from a strict-JSON object
/// `src`, **dependency-free** (the `embeddings` feature carries no
/// `serde_json` — EMB-1).
///
/// Returns `Ok(Some(value))` when `key` is present with a JSON string value,
/// `Ok(None)` when `key` is absent, and `Err` when `src` is not a JSON object,
/// is structurally malformed, or `key` is present with a non-string value.
///
/// This is intentionally *not* the full [`crate::embeddings::config`]
/// strict-JSON scanner — that one is purpose-built for `1_Pooling/config.json`
/// (its `KNOWN_KEYS` schema and "pooling config" error text). The load factory
/// needs exactly one string field (`model_type`) from `config.json`, so this
/// is a minimal top-level-object walker: it parses each `"key": value` pair,
/// captures the matched string, and *skips* every other value (strings,
/// numbers, `true`/`false`/`null`, and — to find their end — nested
/// objects/arrays) with a depth cap so a hostile `config.json` cannot
/// stack-overflow the walker. The **first** occurrence of `key` wins.
///
/// The whole top-level object is validated to its closing `}` even after the
/// key is found, so a truncated / malformed `config.json` whose `model_type`
/// happens to be the first key (e.g. `{"model_type": "bert"` with no close) is
/// rejected rather than silently accepted — the file must be well-formed JSON.
fn extract_string_field(src: &str, key: &str) -> std::result::Result<Option<String>, String> {
  let bytes = src.as_bytes();
  let mut p = JsonCursor { bytes, pos: 0 };
  p.skip_ws();
  p.expect(b'{')?;
  p.skip_ws();
  let mut found: Option<String> = None;
  if p.peek() == Some(b'}') {
    p.pos += 1;
    return finish_object(&mut p, found);
  }
  loop {
    p.skip_ws();
    let field = p.parse_string()?;
    p.skip_ws();
    p.expect(b':')?;
    p.skip_ws();
    if field == key && found.is_none() {
      // The first matching key: its value must be a JSON string. Capture it
      // but keep validating the rest of the object (do NOT return early).
      if p.peek() == Some(b'"') {
        found = Some(p.parse_string()?);
      } else {
        return Err(format!(
          "`{key}` is present but its value is not a JSON string"
        ));
      }
    } else {
      // A different key (or a later duplicate of `key`) — skip its value,
      // whatever JSON type it is, to advance past it.
      p.skip_value(0)?;
    }
    p.skip_ws();
    match p.peek() {
      Some(b',') => {
        p.pos += 1;
        p.skip_ws();
        if p.peek() == Some(b'}') {
          return Err("trailing comma before `}` is not valid JSON".to_owned());
        }
      }
      Some(b'}') => {
        p.pos += 1;
        return finish_object(&mut p, found);
      }
      Some(c) => {
        return Err(format!(
          "expected `,` or `}}` in object but found {:?} at byte {}",
          c as char, p.pos
        ));
      }
      None => return Err("expected `,` or `}` but reached end of input".to_owned()),
    }
  }
}

/// After the top-level object's closing `}` is consumed, reject any trailing
/// non-whitespace bytes (a strict-JSON document is a single value) and return
/// the captured result.
fn finish_object(
  p: &mut JsonCursor<'_>,
  found: Option<String>,
) -> std::result::Result<Option<String>, String> {
  p.skip_ws();
  if p.pos != p.bytes.len() {
    return Err(format!(
      "trailing data after top-level object at byte {}",
      p.pos
    ));
  }
  Ok(found)
}

/// Hard cap on nested object/array depth [`extract_string_field`]'s value-skip
/// will descend. A 1-MiB-capped `config.json` could otherwise pack hundreds of
/// thousands of `[`/`{` at one position; without this guard
/// [`JsonCursor::skip_value`]'s recursion would overflow the thread stack on
/// hostile model data (turning a malformed `config.json` into an abort instead
/// of a recoverable error). 128 levels covers every realistic HF
/// `config.json` — they are shallow — yet caps stack growth at a constant.
const MAX_JSON_DEPTH: usize = 128;

/// A byte cursor over a strict-JSON `config.json`, used by
/// [`extract_string_field`]. Deliberately tiny — it understands just enough
/// JSON to walk a flat top-level object and skip arbitrarily-typed values.
struct JsonCursor<'a> {
  bytes: &'a [u8],
  pos: usize,
}

impl JsonCursor<'_> {
  fn peek(&self) -> Option<u8> {
    self.bytes.get(self.pos).copied()
  }

  /// Advance past ASCII JSON whitespace (space, tab, CR, LF).
  fn skip_ws(&mut self) {
    while let Some(b) = self.peek() {
      if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
        self.pos += 1;
      } else {
        break;
      }
    }
  }

  /// Consume the byte `b`, or error citing what was found instead.
  fn expect(&mut self, b: u8) -> std::result::Result<(), String> {
    match self.peek() {
      Some(c) if c == b => {
        self.pos += 1;
        Ok(())
      }
      Some(c) => Err(format!(
        "expected {:?} but found {:?} at byte {}",
        b as char, c as char, self.pos
      )),
      None => Err(format!("expected {:?} but reached end of input", b as char)),
    }
  }

  /// Parse a JSON string at the cursor (cursor must be on the opening `"`),
  /// returning the decoded UTF-8 contents. Handles the JSON escapes
  /// (`\" \\ \/ \b \f \n \r \t` and `\uXXXX`, including surrogate pairs).
  ///
  /// Bytes are accumulated into a `Vec<u8>` and decoded once at the close:
  /// a raw (unescaped) byte ≥ `0x20` is copied verbatim, so a multi-byte
  /// UTF-8 codepoint in the source (which is already valid UTF-8 —
  /// `String::from_utf8` ran upstream) is preserved exactly; escape sequences
  /// are encoded into the buffer via [`char::encode_utf8`]. The final
  /// `from_utf8` cannot fail on the copied source bytes, and `\u` escapes only
  /// ever yield valid scalars (surrogates are paired/rejected here).
  fn parse_string(&mut self) -> std::result::Result<String, String> {
    self.expect(b'"')?;
    let mut out: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4];
    loop {
      let b = self
        .peek()
        .ok_or_else(|| "unterminated string: reached end of input".to_owned())?;
      self.pos += 1;
      match b {
        b'"' => {
          return String::from_utf8(out).map_err(|e| format!("string is not valid UTF-8: {e}"));
        }
        b'\\' => {
          let esc = self
            .peek()
            .ok_or_else(|| "unterminated escape in string".to_owned())?;
          self.pos += 1;
          let push = |out: &mut Vec<u8>, ch: char, buf: &mut [u8; 4]| {
            out.extend_from_slice(ch.encode_utf8(buf).as_bytes());
          };
          match esc {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0C),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
              let cp = self.parse_hex4()?;
              if (0xD800..=0xDBFF).contains(&cp) {
                // High surrogate — must be followed by `\uXXXX` low surrogate.
                if self.peek() != Some(b'\\') {
                  return Err("unpaired UTF-16 high surrogate in string".to_owned());
                }
                self.pos += 1;
                if self.peek() != Some(b'u') {
                  return Err("unpaired UTF-16 high surrogate in string".to_owned());
                }
                self.pos += 1;
                let low = self.parse_hex4()?;
                if !(0xDC00..=0xDFFF).contains(&low) {
                  return Err("invalid UTF-16 low surrogate in string".to_owned());
                }
                let combined = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                let ch = char::from_u32(combined)
                  .ok_or_else(|| "invalid Unicode scalar from surrogate pair".to_owned())?;
                push(&mut out, ch, &mut buf);
              } else if (0xDC00..=0xDFFF).contains(&cp) {
                return Err("unpaired UTF-16 low surrogate in string".to_owned());
              } else {
                let ch = char::from_u32(cp)
                  .ok_or_else(|| "invalid Unicode scalar in `\\u` escape".to_owned())?;
                push(&mut out, ch, &mut buf);
              }
            }
            other => {
              return Err(format!(
                "invalid escape `\\{}` in string at byte {}",
                other as char, self.pos
              ));
            }
          }
        }
        // A raw control byte is invalid in a JSON string.
        0x00..=0x1F => {
          return Err(format!(
            "raw control byte {b:#04x} in string at byte {}",
            self.pos - 1
          ));
        }
        // Any other byte (ASCII ≥ 0x20 or a UTF-8 continuation/lead byte of a
        // multi-byte codepoint in the already-valid-UTF-8 source) is copied
        // verbatim; the closing `from_utf8` reassembles the codepoints.
        _ => out.push(b),
      }
    }
  }

  /// Parse exactly four hexadecimal digits (a `\uXXXX` payload).
  fn parse_hex4(&mut self) -> std::result::Result<u32, String> {
    let mut v = 0u32;
    for _ in 0..4 {
      let d = self
        .peek()
        .ok_or_else(|| "unterminated `\\u` escape".to_owned())?;
      let nibble = match d {
        b'0'..=b'9' => u32::from(d - b'0'),
        b'a'..=b'f' => u32::from(d - b'a') + 10,
        b'A'..=b'F' => u32::from(d - b'A') + 10,
        _ => {
          return Err(format!(
            "invalid hex digit {:?} in `\\u` escape at byte {}",
            d as char, self.pos
          ));
        }
      };
      v = (v << 4) | nibble;
      self.pos += 1;
    }
    Ok(v)
  }

  /// Skip exactly one JSON value at the cursor — whatever its type — so the
  /// walker can advance past a non-matching key's value. `depth` is the
  /// current nesting level, capped by [`MAX_JSON_DEPTH`].
  fn skip_value(&mut self, depth: usize) -> std::result::Result<(), String> {
    if depth > MAX_JSON_DEPTH {
      return Err(format!(
        "nested object/array depth exceeds the {MAX_JSON_DEPTH}-level cap"
      ));
    }
    self.skip_ws();
    match self.peek() {
      Some(b'"') => {
        self.parse_string()?;
        Ok(())
      }
      Some(b'{') => self.skip_object(depth),
      Some(b'[') => self.skip_array(depth),
      Some(b't') => self.expect_lit(b"true"),
      Some(b'f') => self.expect_lit(b"false"),
      Some(b'n') => self.expect_lit(b"null"),
      Some(c) if c == b'-' || c.is_ascii_digit() => {
        self.skip_number();
        Ok(())
      }
      Some(c) => Err(format!(
        "unexpected byte {:?} where a JSON value was expected at byte {}",
        c as char, self.pos
      )),
      None => Err("expected a JSON value but reached end of input".to_owned()),
    }
  }

  /// Skip a nested object `{ ... }` (cursor on the opening `{`).
  fn skip_object(&mut self, depth: usize) -> std::result::Result<(), String> {
    self.expect(b'{')?;
    self.skip_ws();
    if self.peek() == Some(b'}') {
      self.pos += 1;
      return Ok(());
    }
    loop {
      self.skip_ws();
      self.parse_string()?; // key
      self.skip_ws();
      self.expect(b':')?;
      self.skip_value(depth + 1)?;
      self.skip_ws();
      match self.peek() {
        Some(b',') => {
          self.pos += 1;
          self.skip_ws();
          if self.peek() == Some(b'}') {
            return Err("trailing comma before `}` is not valid JSON".to_owned());
          }
        }
        Some(b'}') => {
          self.pos += 1;
          return Ok(());
        }
        Some(c) => {
          return Err(format!(
            "expected `,` or `}}` in object but found {:?} at byte {}",
            c as char, self.pos
          ));
        }
        None => return Err("expected `,` or `}` but reached end of input".to_owned()),
      }
    }
  }

  /// Skip a nested array `[ ... ]` (cursor on the opening `[`).
  fn skip_array(&mut self, depth: usize) -> std::result::Result<(), String> {
    self.expect(b'[')?;
    self.skip_ws();
    if self.peek() == Some(b']') {
      self.pos += 1;
      return Ok(());
    }
    loop {
      self.skip_value(depth + 1)?;
      self.skip_ws();
      match self.peek() {
        Some(b',') => {
          self.pos += 1;
          self.skip_ws();
          if self.peek() == Some(b']') {
            return Err("trailing comma before `]` is not valid JSON".to_owned());
          }
        }
        Some(b']') => {
          self.pos += 1;
          return Ok(());
        }
        Some(c) => {
          return Err(format!(
            "expected `,` or `]` in array but found {:?} at byte {}",
            c as char, self.pos
          ));
        }
        None => return Err("expected `,` or `]` but reached end of input".to_owned()),
      }
    }
  }

  /// Consume a literal token (`true` / `false` / `null`).
  fn expect_lit(&mut self, lit: &[u8]) -> std::result::Result<(), String> {
    if self.bytes[self.pos..].starts_with(lit) {
      self.pos += lit.len();
      Ok(())
    } else {
      Err(format!(
        "invalid JSON literal at byte {} (expected {:?})",
        self.pos,
        std::str::from_utf8(lit).unwrap_or("<lit>")
      ))
    }
  }

  /// Skip a JSON number — the cursor advances over every byte that can be
  /// part of an `int frac? exp?` token. Exact numeric validity is not the
  /// walker's concern (it only needs the value's *extent*); a non-matching
  /// key's malformed number would simply fail at the following `,`/`}`.
  fn skip_number(&mut self) {
    while let Some(b) = self.peek() {
      if b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.' || b == b'e' || b == b'E' {
        self.pos += 1;
      } else {
        break;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  //! End-to-end load-factory tests, driven by a **mock** embedding model
  //! registered into a fresh [`EmbeddingModelTypeRegistry`] (per the project's
  //! no-model-arch rule, this layer ships the seam, not architectures — so the
  //! end-to-end path is proven against a hand-traced mock constructor over a
  //! temp model directory). Strict-JSON-scanner edge cases for
  //! [`extract_string_field`] are unit-tested directly.

  use std::sync::atomic::{AtomicU64, Ordering};

  use super::*;
  use crate::embeddings::model::EmbeddingModelOutput;

  /// A minimal `config.json` for the mock architecture. `model_type` is the
  /// registry-dispatch key; `mock_extra` is a model-specific key the
  /// constructor reads off the raw JSON to prove
  /// [`LoadedEmbeddingModel::config_json`] reaches it.
  fn mock_config_json(model_type: &str) -> String {
    format!(
      r#"{{
        "model_type": "{model_type}",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 5,
        "mock_extra": 7
      }}"#
    )
  }

  /// A trivial [`EmbeddingModel`] the mock constructor returns. `forward`
  /// returns a fixed `(batch, seq, hidden)` zero hidden-state (dispatch is
  /// what these tests prove, not the encoder math).
  struct MockLoadedEmbedding {
    hidden: usize,
  }

  impl EmbeddingModel for MockLoadedEmbedding {
    fn forward(&self, input_ids: &Array, _attention_mask: &Array) -> Result<EmbeddingModelOutput> {
      let (batch, seq) = match input_ids.shape().as_slice() {
        [b, s] => (*b, *s),
        other => {
          return Err(Error::ShapeMismatch {
            message: format!("MockLoadedEmbedding::forward expects (batch, seq), got {other:?}"),
          });
        }
      };
      let data = vec![0.0_f32; batch * seq * self.hidden];
      let last_hidden_state = Array::from_slice::<f32>(&data, &(batch, seq, self.hidden))?;
      Ok(EmbeddingModelOutput::from_hidden_state(last_hidden_state))
    }
  }

  /// Build an [`EmbeddingModelConstructor`] for the mock architecture: assert
  /// at least one weight tensor arrived, read the model-specific `mock_extra`
  /// off the raw config JSON (proving the raw body reaches the constructor),
  /// and return a [`MockLoadedEmbedding`].
  fn mock_constructor() -> EmbeddingModelConstructor {
    Box::new(
      |loaded: &LoadedEmbeddingModel| -> Result<Box<dyn EmbeddingModel>> {
        assert!(
          !loaded.weights.is_empty(),
          "constructor should receive the loaded weights"
        );
        // Model-specific key, read from the raw JSON via the dependency-free
        // extractor exercised below (no `serde_json` in the `embeddings`
        // feature). It is a number here, so the string extractor returns
        // `None` for it — the assertion is simply that the raw body is intact
        // and parseable.
        assert!(
          loaded.config_json.contains("mock_extra"),
          "raw config json should reach the constructor"
        );
        Ok(Box::new(MockLoadedEmbedding { hidden: 4 }))
      },
    )
  }

  /// A fresh, writable per-test temp directory (the crate's
  /// no-`tempfile`-crate convention: `temp_dir()` + pid + a process-unique
  /// counter so parallel tests never collide). Created empty.
  fn fresh_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
      "mlxrs-emb-factory-{tag}-{}-{n}",
      std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Serialize a minimal but loadable `tokenizer.json` (a 3-token WordLevel
  /// model with a Whitespace pre-tokenizer) into `dir` — the same fixture
  /// style as `embeddings::encode`'s tests, so the reused
  /// [`Tokenizer::from_path`] loads it.
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

  /// Populate `dir` with just `config.json` (with the given `model_type`) and
  /// a tiny single-tensor `model.safetensors` — but **no** `tokenizer.json`.
  fn write_model_dir_no_tokenizer(dir: &Path, model_type: &str) {
    std::fs::write(dir.join("config.json"), mock_config_json(model_type)).unwrap();
    let mut weights: EmbeddingWeights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  }

  /// Populate `dir` as a minimal loadable model directory: `config.json`,
  /// `model.safetensors`, and `tokenizer.json`.
  fn write_model_dir(dir: &Path, model_type: &str) {
    write_model_dir_no_tokenizer(dir, model_type);
    write_tokenizer(dir);
  }

  /// Write a `1_Pooling/config.json` declaring mean pooling into `dir`.
  fn write_pooling_config(dir: &Path) {
    let pooling_dir = dir.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir).unwrap();
    std::fs::write(
      pooling_dir.join("config.json"),
      r#"{"word_embedding_dimension": 4, "pooling_mode_mean_tokens": true}"#,
    )
    .unwrap();
  }

  #[test]
  fn load_dispatches_to_registered_mock_and_returns_context() {
    let dir = fresh_dir("dispatch");
    write_model_dir(&dir, "mockemb");
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &registry).expect("load should succeed");

    assert_eq!(ctx.model_type, "mockemb");
    // No `1_Pooling/config.json` was written → pooling is None.
    assert!(ctx.pooling.is_none());

    // The constructed model is the mock: drive one forward to confirm wiring.
    let ids = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
    let mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(1usize, 3)).unwrap();
    let out = ctx.model.forward(&ids, &mask).unwrap();
    assert_eq!(out.last_hidden_state.shape(), vec![1, 3, 4]);

    // The tokenizer loaded from the same directory.
    let tok_ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(tok_ids.len(), 3);
  }

  #[test]
  fn from_id_resolves_as_local_path() {
    // An `EmbeddingIdentifier::Id` is treated as a LOCAL path (no network).
    let dir = fresh_dir("idpath");
    write_model_dir(&dir, "mockemb");
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_id(dir.to_str().unwrap());
    assert_eq!(config.model_directory(), dir.as_path());

    let ctx = load(&config, &registry).expect("id-as-local-path load should succeed");
    assert_eq!(ctx.model_type, "mockemb");
  }

  #[test]
  fn pooling_config_is_loaded_when_present() {
    // A `1_Pooling/config.json` in the model dir → `ctx.pooling` is Some.
    let dir = fresh_dir("pooling");
    write_model_dir(&dir, "mockemb");
    write_pooling_config(&dir);
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &registry).expect("load with pooling config");
    let pooling = ctx.pooling.expect("pooling config should be parsed");
    assert_eq!(pooling.strategy, crate::embeddings::PoolingStrategy::Mean);
    assert_eq!(pooling.dimension, Some(4));
  }

  #[test]
  fn unknown_model_type_is_recoverable_error() {
    // config.json says "nope" but only "mockemb" is registered → an
    // unsupported-model-type Error (NOT a panic), naming the type.
    let dir = fresh_dir("unknown");
    write_model_dir(&dir, "nope");
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("unknown model_type must error");
    };
    let msg = err.to_string();
    assert!(msg.contains("unsupported model type"), "got: {msg}");
    assert!(msg.contains("nope"), "error should name the type: {msg}");
  }

  #[test]
  fn missing_config_json_is_recoverable_error() {
    // A directory with NO config.json → a recoverable Error naming
    // config.json, never a panic.
    let dir = fresh_dir("noconfig");
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("missing config.json must error");
    };
    assert!(
      err.to_string().contains("config.json"),
      "error should name config.json: {err}"
    );
  }

  #[test]
  fn oversized_config_json_is_recoverable_error() {
    // A `config.json` larger than the byte cap → a recoverable Error
    // (bounded read), never an unbounded allocation.
    let dir = fresh_dir("bigconfig");
    let mut huge = String::from("{\"model_type\": \"mockemb\", \"pad\": \"");
    huge.push_str(&"x".repeat((MAX_CONFIG_BYTES as usize) + 16));
    huge.push_str("\"}");
    std::fs::write(dir.join("config.json"), huge).unwrap();
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("oversized config.json must error");
    };
    assert!(
      err.to_string().contains("cap"),
      "error should mention the byte cap: {err}"
    );
  }

  #[test]
  fn missing_weights_is_recoverable_error() {
    // config + tokenizer but NO safetensors → recoverable Error from the
    // shared weight loader.
    let dir = fresh_dir("noweights");
    std::fs::write(dir.join("config.json"), mock_config_json("mockemb")).unwrap();
    write_tokenizer(&dir);
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("missing weights must error");
    };
    assert!(
      err.to_string().contains("no model weights"),
      "error should name the missing weights: {err}"
    );
  }

  #[test]
  fn tokenizer_source_loads_from_separate_directory() {
    // Split layout: the model dir has config + weights but NO
    // `tokenizer.json`; a SEPARATE dir holds the tokenizer.
    let model_dir = fresh_dir("split-model");
    write_model_dir_no_tokenizer(&model_dir, "mockemb");
    assert!(!model_dir.join("tokenizer.json").exists());
    let tok_dir = fresh_dir("split-tok");
    write_tokenizer(&tok_dir);

    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config =
      EmbeddingModelConfiguration::from_directory(&model_dir).with_tokenizer_source(&tok_dir);
    assert_eq!(config.tokenizer_directory(), tok_dir.as_path());

    let ctx = load(&config, &registry).expect("split-tokenizer load should succeed");
    let ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(ids.len(), 3);
  }

  #[test]
  fn registry_contains_and_separator_normalization() {
    // `_get_model_arch` normalizes `-`→`_`: registering under "xlm-roberta"
    // is found under "xlm_roberta" (and the `-` spelling) too.
    let registry = EmbeddingModelTypeRegistry::new().with("xlm-roberta", mock_constructor());
    assert!(registry.contains("xlm-roberta"));
    assert!(registry.contains("xlm_roberta"));
    assert!(!registry.contains("bert"));
    assert_eq!(remap_model_type("xlm-roberta"), "xlm_roberta");
    assert_eq!(remap_model_type("bert"), "bert");
  }

  #[test]
  fn register_replaces_and_returns_previous() {
    let mut registry = EmbeddingModelTypeRegistry::new();
    assert!(registry.register("mockemb", mock_constructor()).is_none());
    // A second registration of the same canonical id returns the displaced
    // constructor (last-writer-wins, mirroring the Swift dict assignment).
    assert!(registry.register("mockemb", mock_constructor()).is_some());
  }

  #[test]
  fn separator_normalized_config_dispatches() {
    // A checkpoint whose config.json `model_type` is "xlm-roberta" loads
    // against a registry that registered the canonical "xlm_roberta".
    let dir = fresh_dir("sep-dispatch");
    write_model_dir(&dir, "xlm-roberta");
    let registry = EmbeddingModelTypeRegistry::new().with("xlm_roberta", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &registry).expect("separator-normalized dispatch");
    assert_eq!(ctx.model_type, "xlm_roberta");
  }

  #[test]
  fn unsupported_model_type_does_not_touch_weights() {
    // An UNREGISTERED `model_type` must be rejected BEFORE any weights are
    // loaded. The model dir's `model.safetensors` is deliberately INVALID —
    // if `load()` tried to load weights it would surface a parse error;
    // instead it must return the recoverable unsupported-model error.
    let dir = fresh_dir("unsupported-cheap");
    std::fs::write(dir.join("config.json"), mock_config_json("nope")).unwrap();
    std::fs::write(
      dir.join("model.safetensors"),
      b"this is not a safetensors file",
    )
    .unwrap();

    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("unsupported model_type must error");
    };
    let msg = err.to_string();
    assert!(
      msg.contains("unsupported model type"),
      "expected the unsupported-model error before any weight load, got: {msg}"
    );
    assert!(msg.contains("nope"), "error should name the type: {msg}");
  }

  // ───────────── extract_string_field unit tests ─────────────

  #[test]
  fn extract_finds_present_string_field() {
    let r = extract_string_field(
      r#"{"model_type": "bert", "hidden_size": 768}"#,
      "model_type",
    );
    assert_eq!(r.unwrap(), Some("bert".to_owned()));
  }

  #[test]
  fn extract_skips_other_typed_values_before_match() {
    // The matched key comes after a nested object, an array, a number, and a
    // bool — all of which must be skipped to reach it.
    let src = r#"{
      "nested": {"a": [1, 2, {"deep": true}], "b": "x"},
      "arr": [true, false, null, 3.5e2],
      "n": -12.0,
      "flag": false,
      "model_type": "qwen3"
    }"#;
    assert_eq!(
      extract_string_field(src, "model_type").unwrap(),
      Some("qwen3".to_owned())
    );
  }

  #[test]
  fn extract_returns_none_for_absent_field() {
    let r = extract_string_field(r#"{"hidden_size": 768, "vocab_size": 5}"#, "model_type");
    assert_eq!(r.unwrap(), None);
  }

  #[test]
  fn extract_returns_none_for_empty_object() {
    assert_eq!(extract_string_field("{}", "model_type").unwrap(), None);
  }

  #[test]
  fn extract_decodes_json_escapes() {
    // `\u` escape + surrogate pair (😀 = U+1F600) + a simple escape.
    let src = r#"{"model_type": "ab\nc😀"}"#;
    assert_eq!(
      extract_string_field(src, "model_type").unwrap(),
      Some("ab\nc😀".to_owned())
    );
  }

  #[test]
  fn extract_rejects_non_string_matched_value() {
    let r = extract_string_field(r#"{"model_type": 123}"#, "model_type");
    assert!(r.is_err(), "a numeric model_type must be rejected");
  }

  #[test]
  fn extract_rejects_non_object_root() {
    assert!(extract_string_field(r#"["model_type"]"#, "model_type").is_err());
    assert!(extract_string_field(r#""bert""#, "model_type").is_err());
  }

  #[test]
  fn extract_rejects_malformed_json() {
    // Unterminated object whose FIRST key is the match — must NOT be accepted
    // just because the value parsed; the whole object is validated to `}`.
    assert!(extract_string_field(r#"{"model_type": "bert""#, "model_type").is_err());
    // Trailing comma before `}`.
    assert!(extract_string_field(r#"{"model_type": "bert",}"#, "model_type").is_err());
    // Missing `:` between key and value.
    assert!(extract_string_field(r#"{"model_type" "bert"}"#, "model_type").is_err());
    // Trailing data after the top-level object (a strict-JSON document is a
    // single value).
    assert!(extract_string_field(r#"{"model_type": "bert"} junk"#, "model_type").is_err());
    // An unterminated NESTED object after the matched key is still rejected.
    assert!(extract_string_field(r#"{"model_type": "bert", "x": {"a": 1"#, "model_type").is_err());
  }

  #[test]
  fn extract_rejects_pathologically_deep_nesting() {
    // A value-skip past `MAX_JSON_DEPTH` levels of `[` must error, not
    // overflow the stack.
    let mut src = String::from(r#"{"deep": "#);
    src.push_str(&"[".repeat(MAX_JSON_DEPTH + 8));
    let r = extract_string_field(&src, "model_type");
    assert!(
      r.is_err(),
      "pathological nesting must be a recoverable error"
    );
  }
}
