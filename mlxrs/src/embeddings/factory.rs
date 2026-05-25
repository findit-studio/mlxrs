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

use glob::{MatchOptions, glob_with};

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
#[derive(Debug, Clone, PartialEq, Eq, derive_more::IsVariant)]
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
  id: EmbeddingIdentifier,
  /// An optional **separate local directory** for the tokenizer
  /// (mlx-swift-lm's `tokenizerSource`). `None` ⇒ load the tokenizer from the
  /// model directory (the common case). Like [`EmbeddingIdentifier`] this is
  /// local-only — no Hub download.
  tokenizer_source: Option<PathBuf>,
}

impl EmbeddingModelConfiguration {
  /// Construct from an [`EmbeddingIdentifier`] and an optional separate
  /// tokenizer source directory.
  pub fn new(id: EmbeddingIdentifier, tokenizer_source: Option<PathBuf>) -> Self {
    Self {
      id,
      tokenizer_source,
    }
  }

  /// The model identifier.
  #[inline(always)]
  pub fn id(&self) -> &EmbeddingIdentifier {
    &self.id
  }

  /// The optional separate tokenizer source directory.
  #[inline(always)]
  pub fn tokenizer_source(&self) -> Option<&Path> {
    self.tokenizer_source.as_deref()
  }

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
  /// [`tokenizer_source()`](Self::tokenizer_source) if set, else the model
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
/// Keys carry mlx-embeddings' subfolder `<folder>.<key>` namespacing for shards
/// loaded from a nested component directory (see [`load()`]'s weight discovery);
/// a root-level shard's keys are verbatim. [`load()`] performs no further
/// `sanitize`/remap — architecture-specific key rewriting is the per-usecase
/// constructor's responsibility (exactly as [`crate::lm::load`] leaves it), but
/// the multi-component subfolder prefix is applied at load time to match
/// `mlx_embeddings.utils.load_model`.
pub type EmbeddingWeights = HashMap<String, Array>;

/// Everything [`load()`] resolved from a model directory, handed to an
/// [`EmbeddingModelConstructor`] so it can assemble a concrete architecture
/// without re-reading the directory.
///
/// The embeddings twin of [`crate::lm::factory::LoadedModel`]. Borrowing — the
/// constructor gets `&LoadedEmbeddingModel`; it reads the
/// [`model_type()`](Self::model_type), the verbatim
/// [`config_json()`](Self::config_json) (the analogue of mlx-swift-lm passing the
/// raw `config.json` `Data` to each model's `Decodable` init), and takes the
/// weight [`Array`]s it needs out of [`weights_ref()`](Self::weights_ref) **by
/// reference** (no implicit eval; mlx `Array` is a cheap refcounted handle).
#[non_exhaustive]
pub struct LoadedEmbeddingModel {
  /// The checkpoint's canonicalized architecture id (`config.json`
  /// `model_type`, after [`remap_model_type`]). The registry key the
  /// constructor was looked up under.
  model_type: String,
  /// The verbatim `config.json` body, for the architecture's
  /// `Decodable`-style init (mlx-swift-lm hands each `EmbeddingModel`'s
  /// initializer the raw config `Data`). Always the exact bytes
  /// [`model_type()`](Self::model_type) was extracted from.
  config_json: String,
  /// The merged, name → [`Array`] weight map (mlx-embeddings' `weights`
  /// dict). Nested-component shards carry the `<folder>.` prefix (see
  /// [`load()`]'s weight discovery); root shards are verbatim. The constructor
  /// applies any further `sanitize`/remap itself.
  weights: EmbeddingWeights,
}

impl LoadedEmbeddingModel {
  /// Construct a [`LoadedEmbeddingModel`] from its three components.
  pub fn new(model_type: String, config_json: String, weights: EmbeddingWeights) -> Self {
    Self {
      model_type,
      config_json,
      weights,
    }
  }

  /// The checkpoint's canonicalized architecture id.
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// The verbatim `config.json` body.
  #[inline(always)]
  pub fn config_json(&self) -> &str {
    &self.config_json
  }

  /// The merged weight map.
  #[inline(always)]
  pub fn weights_ref(&self) -> &EmbeddingWeights {
    &self.weights
  }
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
  /// [`model_type()`](LoadedEmbeddingModel::model_type), mirroring mlx-swift-lm's
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
/// 0. Reject an **empty** model-directory (or, if a separate
///    [`tokenizer_source`](EmbeddingModelConfiguration::tokenizer_source) is
///    set, tokenizer-directory) path **up front** — before any `config.json`,
///    pooling, or shard/glob resolution. An empty directory argument is a
///    caller bug: it is not normalized to `"."`, it is rejected with a
///    recoverable [`Error::Backend`]. The reason this cannot be left to the
///    later steps is shard discovery — `collect_glob_shards` builds its glob
///    pattern as `"<dir>/<suffix>"`, so an empty `<dir>` yields the *absolute*
///    pattern `"/**/model*.safetensors"`, which recursively scans the
///    filesystem root `/` and could merge unrelated `safetensors` from outside
///    the intended directory (a filesystem-escape + wrong-weight load). The
///    same up-front spirit as the non-UTF-8 model-dir-path rejection in
///    `collect_glob_shards`, hoisted ahead of *all* I/O.
/// 1. Resolve the model directory
///    ([`EmbeddingModelConfiguration::model_directory`] — local, no Hub
///    download) and read the `config.json` `model_type` **once** (bounded,
///    dependency-free), canonicalizing it with [`remap_model_type`].
/// 2. **Validate the `model_type` is registered** *before* loading anything
///    heavy: an unsupported checkpoint is a cheap, recoverable
///    [`Error::Backend`] here, with no weight/tokenizer I/O — mlx-embeddings'
///    `ValueError("Model type … not supported.")` / mlx-swift-lm's
///    `unsupportedModelType`.
/// 3. Read the optional `1_Pooling/config.json` via
///    [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
///    (absent ⇒ `None`; a malformed *present* file ⇒ `Err`) — cheap,
///    recoverable metadata, validated **before** the heavy weight/tokenizer
///    loads so a broken pooling config fails fast.
/// 4. Select the tokenizer directory
///    ([`tokenizer_source`](EmbeddingModelConfiguration::tokenizer_source) if
///    set, else the model directory — mlx-swift-lm's `tokenizerDirectory`).
/// 5. Discover and merge the weights from the model directory (reusing
///    [`crate::io::load_safetensors`]), recursively including nested-component
///    shards with mlx-embeddings' `<folder>.` key prefix.
/// 6. Build the [`Tokenizer`] from the selected directory via
///    [`Tokenizer::from_path`].
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

  // (0) Reject an EMPTY model-directory path up front, before any I/O. An
  // empty path is a caller bug — and a load-bearing one: `collect_glob_shards`
  // forms its shard pattern as `"<dir>/<suffix>"`, so an empty `<dir>` yields
  // the ABSOLUTE pattern `"/**/model*.safetensors"`, recursively scanning the
  // filesystem root `/` and potentially merging unrelated weights. It is NOT
  // normalized to `"."`; it is rejected here, ahead of `config.json` / pooling
  // / shard resolution (the same up-front spirit as the non-UTF-8 model-dir
  // rejection inside `collect_glob_shards`).
  reject_empty_dir(model_dir, "model")?;
  // The tokenizer directory, when supplied separately via `tokenizer_source`,
  // is rejected here too: an empty separate tokenizer path is the same caller
  // bug. (When `tokenizer_source` is `None` the tokenizer dir IS the model dir,
  // already checked above.)
  if let Some(tokenizer_source) = &configuration.tokenizer_source {
    reject_empty_dir(tokenizer_source, "tokenizer")?;
  }

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

  // (3) Read the optional `1_Pooling/config.json` (mlx-embeddings
  // `_read_pooling_config`; mlx-swift-lm `loadPooling`) BEFORE any heavy I/O.
  // An ABSENT file ⇒ `None` (the common non-`sentence-transformers` layout); a
  // PRESENT but malformed file ⇒ `Err` (the reader's contract — a planted
  // broken pooling config is a recoverable error, not a silent wrong strategy).
  // This is cheap, recoverable metadata, so it is validated up front: a
  // malformed pooling config must not cost a full weights + tokenizer load
  // first.
  let pooling = read_optional_pooling(model_dir)?;

  // (4) Select the tokenizer directory: the separate `tokenizer_source` if
  // set, else the model directory (mlx-swift-lm's `tokenizerDirectory`).
  let tokenizer_dir = configuration.tokenizer_directory();

  // (5) Discover/merge the weights from the model directory.
  let weights = load_weights(model_dir)?;

  // (6) Build the tokenizer from the selected directory. An embedding encoder
  // does not generate, so there is no eos override (mlx-embeddings' embedding
  // `load` builds a plain tokenizer); pass `None` — `Tokenizer::from_path`
  // then uses the tokenizer's own `eos_token`.
  let tokenizer = Tokenizer::from_path(tokenizer_dir, None).map_err(|e| Error::Backend {
    message: format!(
      "cannot load tokenizer from {}: {e}",
      tokenizer_dir.display()
    ),
  })?;

  // (7) Construct via the registry (already validated as registered in step 2).
  let loaded = LoadedEmbeddingModel::new(model_type, config_json, weights);
  let model = registry.create(&loaded)?;

  Ok(LoadedEmbeddingContext {
    model,
    tokenizer,
    model_type: loaded.model_type,
    pooling,
  })
}

/// Reject an **empty** directory path with a recoverable [`Error::Backend`].
///
/// `role` is the human label for the path being checked (`"model"` /
/// `"tokenizer"`), so the message names which directory the caller passed
/// empty. An empty [`Path`] is a caller bug — it is *not* silently normalized
/// to the current directory (`"."`); see [`load()`] step 0 for why an empty
/// model directory in particular is dangerous (the absolute `"/**/…"` shard
/// pattern → filesystem-root scan).
///
/// The check is `Path::as_os_str().is_empty()` — a byte/`OsStr`-level test, so
/// it is correct for a non-UTF-8 path too and never panics. A path that is
/// merely whitespace or `"."` is *not* empty and is left to the existing I/O
/// steps to resolve or reject; only the genuinely empty path is caught here.
fn reject_empty_dir(dir: &Path, role: &str) -> Result<()> {
  if dir.as_os_str().is_empty() {
    return Err(Error::Backend {
      message: format!("embeddings load: {role} directory path must not be empty"),
    });
  }
  Ok(())
}

/// Read `<dir>/1_Pooling/config.json` if present, mirroring
/// `mlx_embeddings.utils._read_pooling_config` (`return None` when the file is
/// absent) and mlx-swift-lm's `loadPooling`.
///
/// A **genuinely absent** `1_Pooling/config.json` (or no `1_Pooling` directory
/// at all) is the common case for a plain HF encoder checkpoint and yields
/// `Ok(None)`. A **present** entry is parsed via
/// [`pooling_from_st_config_path`](crate::embeddings::pooling_from_st_config_path)
/// (the existing bounded, hand-rolled strict-JSON reader); a malformed present
/// file therefore propagates as an [`Error::Backend`] rather than being
/// silently dropped — a planted broken pooling config is a recoverable error,
/// not a silently-wrong pooling strategy.
///
/// The presence probe is [`Path::symlink_metadata`] (an `lstat` — it does
/// **not** follow symlinks), *not* [`Path::exists`]: `exists()` collapses every
/// error — a broken symlink, a symlink loop, permission-denied, any metadata
/// failure — into `false`, so a *present-but-unresolvable* pooling config would
/// be silently treated as ABSENT and the loader would fall back to the wrong
/// pooling strategy/dimension with no diagnostic. With `symlink_metadata`:
///
/// - `Err(NotFound)` ⇒ the path genuinely does not exist (not even a dangling
///   link) ⇒ `Ok(None)` — the faithful `_read_pooling_config` "absent" case.
/// - `Ok(_)` ⇒ an entry *is* present — including a **broken symlink**, which an
///   `lstat` reports as the link itself rather than erroring `NotFound`. The
///   parse proceeds: [`pooling_from_st_config_path`] opens (following symlinks)
///   and a broken/looping link, or a non-regular target, is rejected there as
///   an [`Error::Backend`]. A "present but bad" config is thus an error, never
///   silently-absent.
/// - any other `Err` (permission-denied, …) ⇒ a present-but-unresolvable
///   config ⇒ an [`Error::Backend`] naming the path, never `Ok(None)`.
fn read_optional_pooling(dir: &Path) -> Result<Option<StPoolingConfig>> {
  // `_read_pooling_config`'s "absent ⇒ None" is a presence check on
  // `1_Pooling/config.json`; the existing reader maps an absent file to an
  // open-error `Err`, so the presence check is done here first. Probe with
  // `symlink_metadata` (`lstat`) so a present-but-broken entry (dangling
  // symlink, EACCES) is NOT mistaken for absence — only a genuine `NotFound`
  // is the "absent ⇒ None" case.
  let config_path = dir.join("1_Pooling").join("config.json");
  match config_path.symlink_metadata() {
    Ok(_) => crate::embeddings::config::pooling_from_st_config_path(dir).map(Some),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
    Err(e) => Err(Error::Backend {
      message: format!(
        "embeddings load: pooling config at {} is present but unreadable: {e}",
        config_path.display()
      ),
    }),
  }
}

/// A discovered weight shard: its full path plus the **key prefix** to apply to
/// every tensor name it contributes (mlx-embeddings' subfolder rename), or
/// `None` for a root-level shard whose keys are merged verbatim.
struct DiscoveredShard {
  /// Full path to the `*.safetensors` file (a [`glob`] match).
  path: PathBuf,
  /// `Some(folder)` when the shard lives in a *child* directory of the model
  /// root — every key is rewritten to `<folder>.<key>` (mlx-embeddings'
  /// `f"{folder_name}.{key}"`, where `folder_name = Path(wf).parent.name`, the
  /// **immediate** parent's name). `None` for a root-level shard (keys
  /// verbatim) — and *only* for a genuine root shard: a nested shard whose
  /// immediate parent folder name is not valid UTF-8 cannot be a `String`
  /// prefix and is rejected by [`collect_glob_shards`] rather than collapsed to
  /// `None`.
  prefix: Option<String>,
}

/// Discover and merge an embedding model's weights from `dir`, mirroring the
/// weight-loading half of `mlx_embeddings.utils.load_model`.
///
/// Resolution order (a faithful port of `load_model`'s two `glob.glob` passes —
/// see [`collect_glob_shards`] for the [`glob`]-crate mechanics):
///
/// 1. **Sharded / single safetensors, RECURSIVELY:** every `model*.safetensors`
///    anywhere under `dir` —
///    `glob.glob(str(model_path / "**/model*.safetensors"), recursive=True)`
///    (mlx-embeddings `utils.py` line 159) — iterated in **sorted full-path
///    order** for a deterministic merge — [`crate::io::load_safetensors`] each
///    and `extend(...)` (later shard wins on a duplicate key, which a
///    well-formed shard set never produces). Covers both `model.safetensors`
///    and `model-00001-of-000NN.safetensors`, at the root and in nested
///    component folders (e.g. a ColVision-style `vision_model/model.safetensors`
///    + `text_model/model.safetensors`).
/// 2. **Back-compat `weight*.safetensors` (root only):** if there is no
///    `model*.safetensors` anywhere, mlx-embeddings `load_model` retries
///    `glob.glob(str(model_path / "weight*.safetensors"))` (`utils.py` line
///    163) — the legacy layout, **not** recursive (no `**`) — and so does this
///    loader. The fallback fires only on a genuinely empty `model*.safetensors`
///    match: a `model*.safetensors` path `glob` *yields* but
///    [`crate::io::load_safetensors`] cannot load (a non-regular target — see
///    [`collect_glob_shards`]'s stat gate) makes step 1 **error**, so a corrupt
///    primary shard fails the load loudly rather than silently degrading to a
///    stale legacy snapshot.
///
/// **Nested-shard key prefixing (mlx-embeddings parity):** a shard found in a
/// *child* directory has every tensor key rewritten to `<folder>.<key>` before
/// merge, where `folder` is the shard's **immediate** parent-directory name —
/// exactly `load_model`'s
/// `folder_name = Path(wf).parent.name; new_key = f"{folder_name}.{key}"` (so a
/// deeper `a/b/model.safetensors` prefixes with `b`, not `a.b`). Root-level
/// shards (`Path(wf).parent == model_path`) keep their keys verbatim. The
/// prefix is computed from the **glob-returned** path's immediate parent (which,
/// for a symlinked component dir, is the *link* name — `glob` yields the path it
/// walked, not the symlink's canonical target). This is the one place the
/// embeddings factory diverges from [`crate::lm::load::load_weights`]'s flat,
/// verbatim merge: multi-component embedding models (vision + text) ship
/// per-component shard folders the loader must namespace, and the prefixing is
/// done **here** (per shard, then merge), leaving the shared
/// [`crate::io::load_safetensors`] and the lm loader untouched. GGUF is not a
/// `mlx_embeddings` weight path and is not handled.
///
/// No safetensors at all → [`Error::Backend`] (mlx-embeddings'
/// `FileNotFoundError("No safetensors found in {model_path}")`).
fn load_weights(dir: &Path) -> Result<EmbeddingWeights> {
  // mlx-embeddings' primary glob: `**/model*.safetensors`, RECURSIVE — the
  // `glob` crate's `**` matches the model dir itself plus arbitrary
  // subdirectories, so a root shard and nested-component shards are all found.
  let mut shards = collect_glob_shards(dir, "**/model*.safetensors")?;

  // mlx-embeddings' back-compat retry: `weight*.safetensors` at the ROOT only
  // (the legacy glob is NOT recursive — `utils.py` line 163 has no `**`).
  if shards.is_empty() {
    shards = collect_glob_shards(dir, "weight*.safetensors")?;
  }

  if shards.is_empty() {
    return Err(Error::Backend {
      message: format!(
        "no model weights found in {}: expected `model*.safetensors` (recursively, or legacy \
         root-level `weight*.safetensors`)",
        dir.display()
      ),
    });
  }

  // Deterministic merge in sorted full-path order (`collect_glob_shards` sorts
  // by path; the cross-shard dup-key tie-break — which a valid shard set never
  // hits — is thus reproducible). A nested shard's keys are prefixed `<folder>.`
  // before merge (mlx-embeddings' subfolder rename); root shards merge verbatim.
  let mut weights: EmbeddingWeights = HashMap::new();
  for shard in &shards {
    let part = crate::io::load_safetensors(&shard.path)?;
    match &shard.prefix {
      Some(folder) => {
        weights.reserve(part.len());
        for (key, value) in part {
          weights.insert(format!("{folder}.{key}"), value);
        }
      }
      None => weights.extend(part),
    }
  }
  Ok(weights)
}

/// Run one `glob` pass under `dir` for the relative `pattern_suffix` (e.g.
/// `"**/model*.safetensors"` or `"weight*.safetensors"`), returning the matched
/// shards — each with its mlx-embeddings `<folder>.` key prefix already computed
/// — sorted by full path.
///
/// This is the faithful port of `mlx_embeddings.utils.load_model`'s
/// `glob.glob(str(model_path / "<suffix>"), recursive=True)`. Using the
/// maintained [`glob`] crate (`rust-lang/glob`) — rather than a hand-rolled
/// recursive directory walk — is what makes the Python-`glob` corner semantics
/// faithful by construction:
///
/// - **`**` recursion** is built into the pattern grammar: with the
///   `"**/model*.safetensors"` suffix `glob` matches `model*.safetensors` at the
///   model dir itself *and* in every subdirectory, exactly Python's
///   `recursive=True`. The legacy `"weight*.safetensors"` suffix has no `**`, so
///   it is root-only — matching `utils.py` line 163.
/// - **`include_hidden=False`** (Python `glob`'s default) excludes any path
///   whose name — at *any* component below the model dir — starts with `.`, so a
///   `.hidden/model.safetensors` directory shard and a root `.model.safetensors`
///   file shard are both excluded. The natural spelling would be
///   [`MatchOptions::require_literal_leading_dot`]` = true`, but `glob 0.3.3`
///   implements that hidden-filter by calling `file_name().to_str().unwrap()` on
///   **every** scanned directory child (`glob-0.3.3/src/lib.rs:953-955`) — a
///   single non-UTF-8 sibling name on a mounted NFS/exFAT/case-sensitive volume
///   would then *panic the process*. So the field is left `false` (which gates
///   off that `unwrap` path entirely) and the `.`-component exclusion is
///   re-implemented here ([`path_has_hidden_component`]) directly on the
///   returned `PathBuf`s' `OsStr` components — no UTF-8 unwrap, panic-free for
///   any filesystem.
/// - **`require_literal_separator` is forced `true`** by `glob_with` regardless
///   of the field, so a `*` never matches across a `/` — `model*.safetensors`
///   matches one path component, as in Python.
/// - **`case_sensitive` is `true`**: `model.safetensors` is matched, `MODEL.SAFETENSORS`
///   is not — Python `glob` is case-sensitive on a case-sensitive filesystem.
/// - **Directory symlinks are followed, and `scandir`/`OSError`s are
///   suppressed**, by the crate: a symlinked component directory is descended
///   (its shard discovered with the *link* name as the immediate-parent prefix),
///   a symlink **cycle** terminates (the crate does not recurse forever), and an
///   unreadable nested directory yields a per-entry [`glob::GlobError`] that is
///   **skipped** here (`continue`) so one bad subdirectory never aborts a load
///   whose real shards live elsewhere — matching Python `glob`, which swallows
///   the `scandir` `OSError`. (`**` recursion has no separate depth cap: the
///   crate bounds the walk itself, so the old hand-rolled `MAX_WEIGHT_DIR_DEPTH`
///   sanity ceiling is gone — there is no unbounded *our-code* recursion left to
///   guard.)
///
/// **Fail-loud on a `model*.safetensors`-named non-regular entry.** `glob`'s
/// match is **name-based** (like Python `glob`): a *directory*, a
/// symlink-to-directory, a FIFO/device, or a **dangling symlink** named
/// `model.safetensors` *is* yielded by the pattern. Each yielded path is
/// therefore `stat`-ed here (via [`std::fs::metadata`], which dereferences
/// symlinks) and a non-regular — or unresolvable — target is rejected with a
/// recoverable [`Error::Backend`] **naming the offending path**. This is the
/// explicit-stat form of the prior rounds' fail-loud contract: a broken
/// primary shard must fail the load, never silently vanish and let
/// [`load_weights`] degrade to a stale `weight*.safetensors` fallback. (HF Hub
/// snapshots store shards as symlinks into `blobs/<hash>`; a *valid* such
/// symlink resolves to a regular file and passes.)
///
/// **Non-UTF-8 paths — fully closed by a byte-level preflight.** Every path
/// component (the model dir, an intermediate directory, an immediate parent
/// folder, or the shard's own leaf file name) that is not valid UTF-8 is either
/// handled or produces a clean [`Error::Backend`] *before* a stale fallback can
/// fire:
///
/// - **Model dir path:** `glob_with` takes a `&str` pattern and internally
///   `unwrap()`s `Path::to_str()` on it, so a model directory whose own path is
///   not valid UTF-8 would *panic* inside the crate. It is rejected up front
///   here with a recoverable [`Error::Backend`].
/// - **Immediate parent folder:** the match is name-based, so a legitimately
///   -named (ASCII) `model*.safetensors` shard sitting in a *child directory
///   whose folder name is non-UTF-8* is yielded by the pattern, yet that
///   non-UTF-8 immediate-parent name cannot become the `String` key prefix —
///   that shard is rejected with a recoverable [`Error::Backend`] naming its
///   path (rather than silently mis-merging its keys as if it were a root
///   shard) by the prefix-derivation step below.
/// - **Shard leaf file name:** `glob 0.3.3` matches a non-recursive pattern
///   component via `file_name().and_then(|s| s.to_str())` and `continue`s on
///   `None` (`glob-0.3.3/src/lib.rs:463-467`, `// FIXME (#9639)`). So a
///   directory entry whose own leaf name is *not* valid UTF-8 — e.g. Unix bytes
///   `model\xff.safetensors` — is **never yielded** by `glob`, even though it
///   matches the `model*.safetensors` shard predicate. Left unchecked, such a
///   primary shard is silently dropped and [`load_weights`] degrades to a stale
///   `weight*.safetensors` fallback with no error. To close that hole,
///   [`collect_glob_shards`] runs a [`scan_non_utf8_shards`] **byte-level
///   preflight** alongside the `glob` pass: it inspects every entry's leaf name
///   at the `OsStr`/byte level (no `to_str`), and if any non-UTF-8 leaf matches
///   a shard pattern it returns an [`Error::Backend`] naming the path before
///   any weights are merged.
///
/// A non-UTF-8 *descendant* name that does **not** match a shard pattern is
/// simply skipped: with `require_literal_leading_dot: false` `glob` no longer
/// runs its `to_str().unwrap()` hidden-filter over directory children, so a
/// non-UTF-8 sibling on a mounted NFS/exFAT/case-sensitive volume no longer
/// panics the walk — it just does not match the ASCII pattern. The literal
/// `dir` portion of the pattern is [`glob::Pattern::escape`]d so a real
/// directory name containing a glob metacharacter (`*`, `?`, `[`, `]`) is
/// matched literally, not interpreted — only the `pattern_suffix` carries
/// pattern metacharacters.
///
/// A malformed *pattern* (a [`glob::PatternError`]) would be a bug in this
/// fixed, escaped pattern, not untrusted input, and maps to [`Error::Backend`].
/// `true` if `path` has a hidden (`.`-prefixed) component *strictly below* the
/// `root` model directory — the explicit, panic-free port of Python `glob`'s
/// `include_hidden=False` (and of `glob 0.3.3`'s `require_literal_leading_dot`,
/// which we cannot use directly: its implementation `unwrap()`s
/// `file_name().to_str()` on every scanned child and so panics on a non-UTF-8
/// sibling name — see [`collect_glob_shards`]).
///
/// Each component name is inspected as an [`OsStr`](std::ffi::OsStr) with **no
/// UTF-8 conversion**: on Unix via [`OsStrExt::as_bytes`] (testing the first
/// byte for `b'.'`), elsewhere via a lossy view. Either way this never panics on
/// a non-UTF-8 name — a non-UTF-8 component simply does not begin with an ASCII
/// `.` and so is not treated as hidden.
///
/// Only components *below* `root` are checked: the model directory the user
/// pointed at is theirs to name (it may itself sit under a `.`-prefixed path)
/// and matches Python `glob`, which only filters path segments it itself walked
/// *under* the glob root. The shard file name is included in the check — a
/// `.model.safetensors` is hidden — but `model*.safetensors` / `weight*.safetensors`
/// never begin with `.`, so a legitimate shard name is unaffected.
fn path_has_hidden_component(path: &Path, root: &Path) -> bool {
  // `strip_prefix` operates on `OsStr`-backed `Path`s with no UTF-8
  // requirement; `Err` (the glob result is somehow not under `root`) is treated
  // conservatively as "no hidden component" — `glob` always yields paths under
  // the *normalized* glob root ([`glob_root`]; `root` MUST be that normalized
  // form, not the raw `dir`), so this branch is unreachable in practice.
  let Ok(rel) = path.strip_prefix(root) else {
    return false;
  };
  rel.components().any(|component| {
    let std::path::Component::Normal(name) = component else {
      // `glob` yields plain descendant paths; `.` / `..` / prefixes never
      // appear, but if one did it is not a hidden *entry* name.
      return false;
    };
    starts_with_dot(name)
  })
}

/// `true` if the OS string `name` begins with an ASCII `.`, inspected without a
/// UTF-8 unwrap so a non-UTF-8 file name can never panic the check.
fn starts_with_dot(name: &std::ffi::OsStr) -> bool {
  #[cfg(unix)]
  {
    use std::os::unix::ffi::OsStrExt;
    name.as_bytes().first() == Some(&b'.')
  }
  #[cfg(not(unix))]
  {
    // Non-Unix has no `OsStrExt::as_bytes`; a lossy view still cannot panic and
    // a leading ASCII `.` survives any lossy conversion intact.
    name.to_string_lossy().starts_with('.')
  }
}

/// The model directory `dir` re-expressed in the **exact path shape the
/// [`glob`] crate yields matched paths in**, so a glob result can be
/// [`strip_prefix`](Path::strip_prefix)ed against it.
///
/// `glob` does **not** preserve a leading current-directory (`.`) component:
/// walking the pattern `"./**/model*.safetensors"` it yields a root shard as
/// `model.safetensors` — *not* `./model.safetensors` — and a root shard under
/// `"./model/..."` as `model/model.safetensors`, *not* `./model/...`. (It also
/// drops any further interior `.` segment of the pattern's `dir` portion.) A
/// raw `dir` of `"."` / `"./sub"` therefore is **not** a prefix of what glob
/// returns, and the previous `path.parent() == dir` test in
/// [`collect_glob_shards`] mis-classified a valid root shard as nested.
///
/// This rebuilds `dir` keeping only the components glob keeps:
/// [`Component::Normal`](std::path::Component) names, a leading
/// [`RootDir`](std::path::Component)/[`Prefix`](std::path::Component) (so an
/// absolute `dir` stays absolute — glob yields absolute results verbatim), and
/// [`ParentDir`](std::path::Component) (`..`) segments, while **dropping every
/// [`CurDir`](std::path::Component) (`.`)** — exactly glob's own normalization.
/// `"."` and `"./"` collapse to the **empty** path, which `strip_prefix` treats
/// as the identity prefix (every relative glob result strips cleanly against
/// it). No `canonicalize`, no symlink resolution: the result is purely a
/// lexical re-spelling of `dir`, so a symlinked component directory still
/// contributes its on-disk *link* name to the key prefix — the documented
/// behavior. Operates on [`OsStr`](std::ffi::OsStr) components, so a non-UTF-8
/// directory name is carried through losslessly and never panics.
fn glob_root(dir: &Path) -> PathBuf {
  let mut root = PathBuf::new();
  for component in dir.components() {
    match component {
      // `glob` strips `.` segments from the paths it yields — drop them so the
      // normalized root matches.
      std::path::Component::CurDir => {}
      // A leading `/` (Unix) or a Windows drive/UNC prefix: glob yields
      // absolute matches with this intact, so keep it.
      std::path::Component::RootDir | std::path::Component::Prefix(_) => {
        root.push(component.as_os_str());
      }
      // `..` and real directory names are preserved verbatim.
      std::path::Component::ParentDir | std::path::Component::Normal(_) => {
        root.push(component.as_os_str());
      }
    }
  }
  root
}

fn collect_glob_shards(dir: &Path, pattern_suffix: &str) -> Result<Vec<DiscoveredShard>> {
  // Defense-in-depth: reject an EMPTY `dir` before building the pattern.
  // `load()` already rejects an empty model/tokenizer directory up front, but
  // `collect_glob_shards` must not itself be a hole — an empty `dir` makes the
  // `format!("{}/{}", escape(dir_str), pattern_suffix)` below the ABSOLUTE
  // pattern `"/**/model*.safetensors"`, which `glob` then expands by
  // recursively scanning the filesystem root `/` (suppressing permission
  // errors per-entry) and could merge unrelated `safetensors` from outside the
  // intended directory. An empty `dir` is a bug, not a request to scan `/`.
  reject_empty_dir(dir, "model")?;

  // `glob_with` takes a `&str` and `unwrap()`s `to_str()` internally — reject a
  // non-UTF-8 model dir path here so that becomes a recoverable error, not a
  // panic inside the crate.
  let dir_str = dir.to_str().ok_or_else(|| Error::Backend {
    message: format!(
      "model directory path {} is not valid UTF-8; cannot glob for weight shards",
      dir.display()
    ),
  })?;

  // Byte-level preflight: `glob 0.3.3` matches a leaf component via
  // `file_name().and_then(|s| s.to_str())` and silently `continue`s on `None`
  // (`glob-0.3.3/src/lib.rs:463-467`) — so a directory entry whose OWN leaf
  // name is not valid UTF-8 (e.g. `model\xff.safetensors`) is never yielded by
  // `glob`, even though it matches the shard predicate. Were it merely skipped,
  // a non-UTF-8-named primary shard would vanish and `load_weights` would
  // degrade to a stale `weight*.safetensors` fallback with no error. The
  // preflight scans for exactly that case at the `OsStr`/byte level and fails
  // loudly *before* the merge — closing the last non-UTF-8 discovery hole. The
  // `glob` path below is unchanged for valid-UTF-8 shards.
  scan_non_utf8_shards(dir, pattern_suffix)?;

  // The literal model-dir prefix is escaped so a metacharacter in a real
  // directory name (`*`, `?`, `[`, `]`) is matched verbatim; only
  // `pattern_suffix` contributes pattern metacharacters (`**`, `model*`).
  let pattern = format!("{}/{}", glob::Pattern::escape(dir_str), pattern_suffix);

  // The model directory in the **exact path shape glob yields matches in** —
  // `glob` strips a leading `./` (and any interior `.` segment) from the paths
  // it returns, so the raw `dir` (`"."`, `"./sub"`, `"sub/"`, an absolute path,
  // ...) is generally NOT a literal prefix of a glob result. Every glob result
  // below is classified by `strip_prefix`-ing it against THIS normalized root
  // (root-vs-nested + key prefix) and by `path_has_hidden_component`; using the
  // raw `dir` instead mis-classified a valid root shard as nested for the very
  // common `.` / `./sub` spellings. `glob_root` is a purely lexical re-spelling
  // (no `canonicalize`, no symlink resolution).
  let glob_root = glob_root(dir);

  // `require_literal_leading_dot` is deliberately `false`, NOT `true`: the
  // `true` spelling would be the natural port of Python glob's
  // `include_hidden=False`, but `glob 0.3.3` implements that filter by calling
  // `file_name().to_str().unwrap()` on every scanned directory child
  // (`glob-0.3.3/src/lib.rs:953-955`) — a single non-UTF-8 sibling name would
  // panic the process. With the field `false` that `unwrap` path is never
  // reached; the `.`-component exclusion is re-applied below via
  // `path_has_hidden_component` on the returned `PathBuf`s (OsStr-level, no
  // UTF-8 unwrap). `case_sensitive: true` matches Python glob on a
  // case-sensitive filesystem; `require_literal_separator` is forced `true` by
  // `glob_with` regardless of the field.
  let options = MatchOptions {
    case_sensitive: true,
    require_literal_separator: false,
    require_literal_leading_dot: false,
  };

  let matches = glob_with(&pattern, options).map_err(|e| Error::Backend {
    // A `PatternError` from this fixed, escaped pattern would be an internal
    // bug, not untrusted input — surface it as a recoverable error all the same.
    message: format!("internal error building weight-shard glob pattern {pattern:?}: {e}"),
  })?;

  let mut out = Vec::new();
  for entry in matches {
    // Python `glob` swallows the `scandir` `OSError` of an unreadable
    // directory; the crate surfaces it as a per-entry `GlobError`. Skip just
    // that entry (the iterator continues over the rest) so one unreadable
    // nested directory never aborts a load whose real shards live elsewhere.
    let Ok(path) = entry else { continue };

    // `include_hidden=False`: with `require_literal_leading_dot: false`, `glob`
    // *does* now yield paths through `.`-prefixed (hidden) components — so the
    // exclusion is re-applied here, explicitly, on the returned `PathBuf`. This
    // is the panic-free replacement for `glob`'s own `to_str().unwrap()` hidden
    // -filter: a `.checkpoints/model.safetensors`, a root `.model.safetensors`,
    // and a normal shard under any `.`-prefixed ancestor are all skipped.
    // Stripped against the NORMALIZED `glob_root` (not the raw `dir`) so the
    // strip succeeds for a `.` / `./sub` spelling.
    if path_has_hidden_component(&path, &glob_root) {
      continue;
    }

    // `glob`'s match is name-based: a directory / symlink-to-dir / FIFO /
    // device / dangling symlink NAMED `model*.safetensors` is yielded. Stat the
    // yielded path (`fs::metadata` dereferences symlinks) and fail loudly on a
    // non-regular — or unresolvable — target, so a broken primary shard fails
    // the load instead of silently vanishing into a stale `weight*.safetensors`
    // fallback.
    match std::fs::metadata(&path) {
      Ok(m) if m.is_file() => {}
      Ok(_) => {
        return Err(Error::Backend {
          message: format!(
            "weight shard {} is a non-regular entry (directory / FIFO / device / socket); \
             refusing to load",
            path.display()
          ),
        });
      }
      Err(e) => {
        return Err(Error::Backend {
          message: format!(
            "weight shard {} cannot be resolved (broken symlink / unreadable target): {e}",
            path.display()
          ),
        });
      }
    }

    // mlx-embeddings: `folder_name = Path(wf).parent.name`, applied iff
    // `Path(wf).parent != model_path` — the IMMEDIATE parent's name
    // (`a/b/model.safetensors` → `b`), never the full relative path. `wf` is the
    // glob-returned path, so a symlinked component dir contributes its LINK
    // name (the path glob walked), not the canonical target.
    //
    // Root-vs-nested is decided by `strip_prefix`-ing the glob result against
    // the NORMALIZED `glob_root` (NOT a `path.parent() == dir` test on the raw
    // `dir`: glob strips a leading `./`, so for `dir == "."` a root shard's
    // parent is the empty path `!= "."` — mis-flagged nested — and for
    // `dir == "./sub"` a root shard's parent `sub != "./sub"` — mis-flagged
    // nested, corrupting root keys with a bogus `sub.` prefix). The stripped
    // remainder's `Normal` components are: the shard file name, preceded by 0+
    // directory names. ZERO directory names ⇒ a genuine ROOT shard ⇒ `None`
    // (keys verbatim). ONE+ ⇒ a NESTED shard ⇒ prefix = the LAST directory name
    // (the IMMEDIATE parent: `a/b/model.safetensors` → `b`).
    //
    // A NESTED shard whose immediate parent folder name is not valid UTF-8
    // cannot become a `String` prefix — and must NOT fall through to `None`,
    // which would silently mis-merge its keys verbatim (and collide with a real
    // root shard). Fail loudly with a path-naming `Error::Backend` instead.
    let relative = path.strip_prefix(&glob_root).map_err(|_| Error::Backend {
      // `glob` always yields matches under the normalized root, so this is
      // unreachable in practice — surface it as a recoverable error all the
      // same rather than silently mis-classifying the shard as root.
      message: format!(
        "weight shard {} is not under the model directory {}; cannot derive the key prefix",
        path.display(),
        glob_root.display()
      ),
    })?;
    // `Path::components()` already collapses any interior `.` segment, so only
    // `Normal` components remain to count; the last is the file name.
    let dir_names: Vec<&std::ffi::OsStr> = relative
      .components()
      .filter_map(|component| match component {
        std::path::Component::Normal(name) => Some(name),
        _ => None,
      })
      .collect();
    let prefix = match dir_names.split_last() {
      // `[.., immediate_parent, file_name]` — a NESTED shard; the prefix is the
      // immediate parent folder name (the component just before the file name).
      Some((_file_name, [.., immediate_parent])) => {
        let folder = immediate_parent.to_str().ok_or_else(|| Error::Backend {
          message: format!(
            "weight shard {} has a non-UTF-8 parent directory name; cannot derive the key \
             prefix",
            path.display()
          ),
        })?;
        Some(folder.to_owned())
      }
      // `[file_name]` (or, defensively, an empty remainder) — a genuine ROOT
      // shard directly under the model directory: keys merge verbatim.
      _ => None,
    };
    out.push(DiscoveredShard { path, prefix });
  }

  // `glob` yields in alphabetical order already, but sort explicitly so the
  // deterministic-merge contract does not silently depend on that.
  out.sort_by(|a, b| a.path.cmp(&b.path));
  Ok(out)
}

/// `true` if the OS string `name`'s **bytes** start with `prefix` and end with
/// `suffix` — the byte-level form of the `glob` shard predicate (`model*` /
/// `weight*` ... `*.safetensors`), evaluated with **no UTF-8 conversion** so a
/// non-UTF-8 leaf file name can be tested without a `to_str` (which `glob`'s
/// own matcher would drop on `None`).
///
/// On Unix the raw bytes are read via [`OsStrExt::as_bytes`]; elsewhere a lossy
/// view is used (non-Unix has no byte accessor, and a lossy view still cannot
/// panic — the ASCII `model`/`weight`/`.safetensors` literals survive any lossy
/// conversion intact, so a non-Unix host never mis-classifies a real shard).
fn name_bytes_match(name: &std::ffi::OsStr, prefix: &[u8], suffix: &[u8]) -> bool {
  #[cfg(unix)]
  let bytes: &[u8] = {
    use std::os::unix::ffi::OsStrExt;
    name.as_bytes()
  };
  #[cfg(not(unix))]
  let lossy = name.to_string_lossy();
  #[cfg(not(unix))]
  let bytes: &[u8] = lossy.as_bytes();

  bytes.len() >= prefix.len() + suffix.len() && bytes.starts_with(prefix) && bytes.ends_with(suffix)
}

/// Byte-level **preflight** for [`collect_glob_shards`]: detect any directory
/// entry whose **leaf file name is not valid UTF-8** yet matches a shard
/// pattern, and fail loudly with an [`Error::Backend`] naming it — *before* the
/// `glob` pass and the weight merge.
///
/// This is the structural backstop for `glob 0.3.3`'s non-recursive leaf match,
/// which reads `file_name().and_then(|s| s.to_str())` and silently `continue`s
/// on `None` (`glob-0.3.3/src/lib.rs:463-467`, `// FIXME (#9639)`). A non-UTF-8
/// -named `model*.safetensors` primary shard would therefore never be yielded
/// by `glob`, and [`load_weights`] would silently degrade to a stale
/// `weight*.safetensors` fallback. Detecting it here turns that silent
/// mis-load into a clean, recoverable error.
///
/// The scan **mirrors the two `glob` passes** [`collect_glob_shards`] runs,
/// dispatching on `pattern_suffix`:
///
/// - `"**/model*.safetensors"` → match a leaf whose bytes start with `b"model"`
///   and end with `b".safetensors"`, searched **recursively** (like the `**`
///   glob), descending every subdirectory.
/// - `"weight*.safetensors"` → match a leaf whose bytes start with `b"weight"`
///   and end with `b".safetensors"`, at the **root only** (the legacy fallback
///   glob has no `**`).
///
/// The SAME hidden-component exclusion the `glob` path applies
/// ([`path_has_hidden_component`] / [`starts_with_dot`]) is honoured: a
/// `.`-prefixed directory is *not* descended and a `.`-prefixed entry is *not*
/// flagged, so the preflight never errors on a path the `glob` pass would
/// itself skip (a non-UTF-8 leaf can never *itself* be hidden — it does not
/// begin with an ASCII `.` — but it may sit under a hidden ancestor). Recursion
/// uses [`std::fs::read_dir`] (no new crate dependency).
///
/// An **IO error is suppressed**, exactly as the `glob` pass suppresses a
/// `scandir` `OSError` (Python `glob` does likewise, and `collect_glob_shards`
/// `continue`s past a per-entry `GlobError`): a `read_dir` that fails — an
/// unreadable subdirectory, or even an unreadable model root — makes the
/// preflight skip that subtree rather than error. This keeps "one unreadable
/// nested directory must not abort a load whose real shards live elsewhere"
/// intact; and an unreadable directory's entries cannot be enumerated by `glob`
/// either, so there is no hidden non-UTF-8 shard to mis-load there. The
/// preflight changes behavior in exactly one way: a non-UTF-8 leaf that `glob`
/// *could* see (its parent is readable) but silently drops now errors.
///
/// The preflight only needs to **detect-and-error**: it does not replicate the
/// `glob` crate's sort order, symlink-cycle termination, or
/// regular-file/`stat` gate — those remain the `glob` pass's job for the
/// valid-UTF-8 shards it does yield. Entry **type is intentionally not
/// inspected**: a non-UTF-8 *directory* named `model\xff.safetensors` would
/// equally be yielded by `glob`'s name-based match (and rejected by the stat
/// gate); flagging any non-UTF-8 shard-named entry, file or not, keeps the
/// fail-loud contract complete.
///
/// **Known limitation (deliberate scope decision).** Recursion descends only
/// real directories (`file_type().is_dir()`, with a fallible `path().is_dir()`
/// fallback) — it does **not** follow directory *symlinks*, whereas the `glob`
/// pass does. So a non-UTF-8-named shard inside a *symlinked* component
/// directory is invisible to both the preflight (does not descend the symlink)
/// and `glob` (silently drops the non-UTF-8 leaf); a model directory stacking a
/// symlinked component dir + a non-UTF-8 shard name + a stale legacy
/// `weight*.safetensors` could then fall back to the legacy file instead of
/// erroring. This contrived layout is accepted as **out of scope** per E3's
/// "match the reference, trust the input" decision (see the project follow-ups
/// doc, `DEFERRED-3`); real Hugging Face model directories neither symlink
/// component directories nor use non-UTF-8 filenames.
fn scan_non_utf8_shards(dir: &Path, pattern_suffix: &str) -> Result<()> {
  // Dispatch on the exact `pattern_suffix` strings `collect_glob_shards` is
  // called with. `(prefix, suffix, recursive)` is the byte-level transcription
  // of the `glob` pattern; an unrecognized suffix is a caller bug, not
  // untrusted input — be conservative and scan nothing rather than guess.
  let (prefix, suffix, recursive): (&[u8], &[u8], bool) = match pattern_suffix {
    "**/model*.safetensors" => (b"model", b".safetensors", true),
    "weight*.safetensors" => (b"weight", b".safetensors", false),
    _ => return Ok(()),
  };

  // An unreadable directory is SKIPPED, mirroring `glob`'s `scandir` `OSError`
  // suppression (and `collect_glob_shards`' per-entry `GlobError` `continue`):
  // one unreadable subtree — or even an unreadable model root — must not abort
  // a load whose real shards live elsewhere. An unreadable directory's children
  // are invisible to `glob` too, so no non-UTF-8 shard is silently mis-loaded.
  let Ok(entries) = std::fs::read_dir(dir) else {
    return Ok(());
  };

  for entry in entries {
    // A per-entry enumeration error is likewise skipped (same `glob`-parity
    // suppression); it cannot have yielded a name to inspect anyway.
    let Ok(entry) = entry else { continue };
    let name = entry.file_name();

    // `include_hidden=False` parity: a `.`-prefixed entry is neither flagged
    // nor descended — exactly what the `glob` pass + `path_has_hidden_component`
    // do. (Checked byte-level, so a non-UTF-8 name never panics here.)
    if starts_with_dot(&name) {
      continue;
    }

    // A non-UTF-8 leaf that matches the shard predicate is the silent-skip hole
    // `glob` leaves open — fail loudly, naming the path (lossy display).
    if name.to_str().is_none() && name_bytes_match(&name, prefix, suffix) {
      let path = entry.path();
      return Err(Error::Backend {
        message: format!(
          "weight shard {} has a non-UTF-8 file name matching the `{}*{}` shard pattern; \
           `glob` silently skips it (glob 0.3.3 leaf match drops non-UTF-8 names), which would \
           let the load fall back to stale weights — refusing to load",
          path.display(),
          String::from_utf8_lossy(prefix),
          String::from_utf8_lossy(suffix),
        ),
      });
    }

    // Recurse into subdirectories for the `**/model*.safetensors` pass only —
    // the legacy `weight*.safetensors` glob is root-only. `file_type()` avoids
    // an extra `stat`; on its (rare) failure fall back to `Path::is_dir`.
    if recursive {
      let is_dir = match entry.file_type() {
        Ok(ft) => ft.is_dir(),
        Err(_) => entry.path().is_dir(),
      };
      if is_dir {
        // A symlinked component directory is descended (matching `glob`, which
        // follows directory symlinks); `read_dir` does not itself recurse, so
        // a symlink cycle cannot loop here without re-entering this call —
        // which it does only through real subdirectory names, and the bounded
        // model-directory trees this loads make that a non-concern (the
        // accepted no-DoS-hardening scope decision). A non-UTF-8-named
        // directory is descended too: its own `model.safetensors` child is
        // matched by the recursive scan, and `collect_glob_shards`' non-UTF-8
        // -parent reject covers the prefix.
        scan_non_utf8_shards(&entry.path(), pattern_suffix)?;
      }
    }
  }
  Ok(())
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
/// stack-overflow the walker.
///
/// **Duplicate-key semantics match a real JSON parser:** if `key` appears more
/// than once at the top level, the **last** occurrence wins — exactly what
/// `serde_json` deserialization into a struct field and Python's `json.load`
/// (which keeps the last value for a duplicate key) both do. Each occurrence's
/// value is still required to be a JSON string (a non-string duplicate is
/// rejected), and every occurrence is fully parsed/validated.
///
/// The whole top-level object is validated to its closing `}` even after the
/// key is found, so a truncated / malformed `config.json` whose `model_type`
/// happens to be the first key (e.g. `{"model_type": "bert"` with no close) is
/// rejected rather than silently accepted — the file must be well-formed JSON.
/// Numbers are validated against the RFC 8259 grammar
/// (`-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?`), so a malformed number anywhere in
/// the object (`01`, `1.`, `1e`) rejects the whole config as invalid JSON
/// rather than being silently accepted.
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
    if field == key {
      // A matching key: its value must be a JSON string. Capture it (a later
      // duplicate OVERWRITES an earlier capture — last-wins, matching
      // `serde_json` / Python `json.load` duplicate-key semantics) but keep
      // validating the rest of the object (do NOT return early).
      if p.peek() == Some(b'"') {
        found = Some(p.parse_string()?);
      } else {
        return Err(format!(
          "`{key}` is present but its value is not a JSON string"
        ));
      }
    } else {
      // A different key — skip its value, whatever JSON type it is, to advance
      // past it.
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
      Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
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

  /// Parse and **strictly validate** a JSON number against the RFC 8259
  /// grammar `-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?` (the cursor must be on the
  /// leading `-` or first digit). A malformed number — `01` (leading zero),
  /// `1.` (trailing dot, no fraction digit), `1e` / `1e+` (exponent with no
  /// digit), or a bare `-` — is rejected as invalid JSON rather than silently
  /// accepted, so the whole `config.json` is required to be well-formed.
  ///
  /// Unlike a lenient extent-skipper, exact validity *is* enforced here: the
  /// "whole config is valid JSON before dispatch" guarantee means a config the
  /// model code would later reject (via a real parser) must not be silently
  /// accepted by this walker.
  fn parse_number(&mut self) -> std::result::Result<(), String> {
    let start = self.pos;
    let invalid = |this: &Self| {
      // Report the malformed token from its start (not the byte we stopped on)
      // so the error points at the offending number.
      Err(format!("invalid JSON number at byte {start}: {:?}", {
        let end = this.pos.min(this.bytes.len());
        std::str::from_utf8(&this.bytes[start..end]).unwrap_or("<number>")
      }))
    };

    // Optional minus (JSON forbids a leading `+`).
    if self.peek() == Some(b'-') {
      self.pos += 1;
    }

    // Integer part: a single `0`, or a nonzero digit followed by more digits.
    // A leading zero (`01`) is invalid; an absent integer part (bare `-`) too.
    match self.peek() {
      Some(b'0') => {
        self.pos += 1;
        // `0` must NOT be followed by another digit (`01`, `00` are invalid).
        if matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
          return invalid(self);
        }
      }
      Some(d) if d.is_ascii_digit() => {
        // d is 1..=9 here (0 handled above). Consume the rest of the digits.
        self.pos += 1;
        while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
          self.pos += 1;
        }
      }
      _ => return invalid(self),
    }

    // Optional fraction: a `.` MUST be followed by at least one digit.
    if self.peek() == Some(b'.') {
      self.pos += 1;
      if !matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
        return invalid(self);
      }
      while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
        self.pos += 1;
      }
    }

    // Optional exponent: `e`/`E`, an optional sign, then ≥1 digit.
    if matches!(self.peek(), Some(b'e' | b'E')) {
      self.pos += 1;
      if matches!(self.peek(), Some(b'+' | b'-')) {
        self.pos += 1;
      }
      if !matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
        return invalid(self);
      }
      while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
        self.pos += 1;
      }
    }

    Ok(())
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
    assert_eq!(out.last_hidden_state().shape(), vec![1, 3, 4]);

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
    assert_eq!(pooling.strategy(), crate::embeddings::PoolingStrategy::Mean);
    assert_eq!(pooling.dimension(), Some(4));
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
  fn empty_model_directory_is_recoverable_error_before_any_scan() {
    // `from_directory("")`, `from_id("")`, and `PathBuf::new()` all yield an
    // EMPTY model-directory path. Without the up-front guard, shard discovery
    // builds the ABSOLUTE pattern `"/**/model*.safetensors"` and scans the
    // filesystem ROOT `/`. `load()` must reject the empty path with a
    // recoverable `Error::Backend` BEFORE any `config.json`/pooling/shard I/O —
    // i.e. it must error without ever performing that filesystem-root scan.
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    for config in [
      EmbeddingModelConfiguration::from_directory(""),
      EmbeddingModelConfiguration::from_id(""),
      EmbeddingModelConfiguration::from_directory(PathBuf::new()),
    ] {
      // Precondition: each constructor really did produce an empty path.
      assert!(
        config.model_directory().as_os_str().is_empty(),
        "fixture precondition: the model directory path must be empty"
      );
      let Err(err) = load(&config, &registry) else {
        panic!("an empty model directory path must be a recoverable error, not a load");
      };
      assert!(
        matches!(err, Error::Backend { .. }),
        "expected a recoverable Backend error; got {err:?}"
      );
      let msg = err.to_string();
      assert!(
        msg.contains("model directory path must not be empty"),
        "the error should explain the empty-path rejection; got: {msg}"
      );
      // The empty path is rejected BEFORE discovery: the error must NOT be a
      // downstream `config.json` / weights failure (which would mean step 0
      // did not fire and a `/`-root scan was attempted).
      assert!(
        !msg.contains("config.json") && !msg.contains("no model weights"),
        "the empty path must be rejected before config/shard resolution; got: {msg}"
      );
    }
  }

  #[test]
  fn empty_tokenizer_source_is_recoverable_error() {
    // A separately-supplied tokenizer directory that is EMPTY is the same
    // caller bug and must be rejected up front too (the model dir here is a
    // real, loadable directory, so only the empty tokenizer path can fail).
    let model_dir = fresh_dir("empty-tok-src");
    write_model_dir(&model_dir, "mockemb");
    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&model_dir).with_tokenizer_source("");
    assert!(
      config.tokenizer_directory().as_os_str().is_empty(),
      "fixture precondition: the tokenizer directory path must be empty"
    );

    let Err(err) = load(&config, &registry) else {
      panic!("an empty tokenizer_source path must be a recoverable error");
    };
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    let msg = err.to_string();
    assert!(
      msg.contains("tokenizer directory path must not be empty"),
      "the error should explain the empty tokenizer-path rejection; got: {msg}"
    );
  }

  #[test]
  fn collect_glob_shards_rejects_empty_dir() {
    // Defense-in-depth: `collect_glob_shards` itself must reject an empty
    // `dir` — an empty `dir` would otherwise build the absolute pattern
    // `"/**/model*.safetensors"` and recursively scan the filesystem root.
    // The guard fires before any glob/I/O, so no directory need exist.
    for suffix in ["**/model*.safetensors", "weight*.safetensors"] {
      let Err(err) = collect_glob_shards(Path::new(""), suffix) else {
        panic!("collect_glob_shards must reject an empty dir, not scan the filesystem root");
      };
      assert!(
        matches!(err, Error::Backend { .. }),
        "expected a recoverable Backend error; got {err:?}"
      );
      assert!(
        err
          .to_string()
          .contains("model directory path must not be empty"),
        "the error should explain the empty-path rejection; got: {err}"
      );
    }
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

  #[test]
  fn extract_duplicate_key_last_wins() {
    // A real JSON parser (serde_json into a field / Python `json.load`) keeps
    // the LAST value for a duplicate top-level key. The hand-rolled extractor
    // must match: the second `model_type` overwrites the first.
    let src = r#"{"model_type": "first", "model_type": "second"}"#;
    assert_eq!(
      extract_string_field(src, "model_type").unwrap(),
      Some("second".to_owned()),
      "last duplicate key must win"
    );
    // Three occurrences: the last still wins; intervening keys do not matter.
    let src3 = r#"{"model_type": "a", "x": 1, "model_type": "b", "model_type": "c"}"#;
    assert_eq!(
      extract_string_field(src3, "model_type").unwrap(),
      Some("c".to_owned())
    );
  }

  #[test]
  fn extract_duplicate_key_non_string_later_value_is_rejected() {
    // Every occurrence of the key is validated as a string — a later duplicate
    // with a non-string value rejects the whole config (it is still malformed
    // for our single-string-field contract).
    let src = r#"{"model_type": "ok", "model_type": 7}"#;
    assert!(
      extract_string_field(src, "model_type").is_err(),
      "a non-string duplicate value must be rejected"
    );
  }

  #[test]
  fn extract_rejects_rfc8259_malformed_numbers() {
    // Each malformed number (in a NON-matching key's value, exercised via the
    // value-skip path) must reject the whole object as invalid JSON.
    for bad in [
      r#"{"x": 01}"#,   // leading zero
      r#"{"x": 00}"#,   // leading zero
      r#"{"x": 1.}"#,   // trailing dot, no fraction digit
      r#"{"x": 1e}"#,   // exponent with no digit
      r#"{"x": 1e+}"#,  // exponent sign with no digit
      r#"{"x": 1E-}"#,  // uppercase exponent sign with no digit
      r#"{"x": -}"#,    // bare minus, no integer part
      r#"{"x": .5}"#,   // no integer part before the dot
      r#"{"x": 1..2}"#, // double dot
    ] {
      assert!(
        extract_string_field(bad, "model_type").is_err(),
        "malformed number must be rejected: {bad}"
      );
    }
  }

  #[test]
  fn extract_accepts_rfc8259_valid_numbers() {
    // Valid RFC 8259 numbers (in a non-matching key's value) must be accepted;
    // `model_type` is absent so the result is `Ok(None)`.
    for good in [
      r#"{"x": 1}"#,
      r#"{"x": 1.0}"#,
      r#"{"x": 1e3}"#,
      r#"{"x": -1.5e-2}"#,
      r#"{"x": 0}"#,
      r#"{"x": 0.5}"#,
      r#"{"x": 10}"#,
      r#"{"x": 1E+10}"#,
      r#"{"x": -0}"#,
    ] {
      assert_eq!(
        extract_string_field(good, "model_type").unwrap(),
        None,
        "valid number must be accepted (model_type absent ⇒ None): {good}"
      );
    }
    // A valid number AS the matched value is still rejected (model_type must be
    // a string) — this is the existing non-string-value contract, unchanged.
    assert!(extract_string_field(r#"{"model_type": 1.0}"#, "model_type").is_err());
  }

  // ───────────── name_bytes_match unit tests ─────────────
  //
  // `name_bytes_match` is the byte-level shard predicate the non-UTF-8
  // preflight (`scan_non_utf8_shards`) uses in place of a `to_str` match — it
  // must classify a leaf name purely from its raw bytes, including names that
  // are NOT valid UTF-8 (the case `glob` itself silently drops). These run on
  // every host, unlike the filesystem-fixture tests which skip on a
  // UTF-8-enforcing volume.

  #[cfg(unix)]
  #[test]
  fn name_bytes_match_classifies_shard_names_at_byte_level() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};

    let model_pat = (b"model".as_slice(), b".safetensors".as_slice());
    let weight_pat = (b"weight".as_slice(), b".safetensors".as_slice());

    // Plain ASCII shard names match their pattern.
    let os = |b: &[u8]| OsString::from_vec(b.to_vec());
    assert!(name_bytes_match(
      &os(b"model.safetensors"),
      model_pat.0,
      model_pat.1
    ));
    assert!(name_bytes_match(
      &os(b"model-00001-of-00002.safetensors"),
      model_pat.0,
      model_pat.1
    ));
    assert!(name_bytes_match(
      &os(b"weights.safetensors"),
      weight_pat.0,
      weight_pat.1
    ));

    // A NON-UTF-8 leaf still matches purely on its bytes — the whole point of
    // the preflight (`glob` would drop this on a `to_str()` `None`).
    let mut bad = b"model".to_vec();
    bad.push(0xFF);
    bad.extend_from_slice(b".safetensors");
    assert!(
      name_bytes_match(&os(&bad), model_pat.0, model_pat.1),
      "a non-UTF-8 `model\\xff.safetensors` leaf must match the model shard pattern"
    );
    assert!(
      OsString::from_vec(bad.clone()).to_str().is_none(),
      "fixture precondition: the leaf must be non-UTF-8"
    );

    // Non-matches: wrong prefix, wrong suffix, and a name too short to carry
    // both prefix and suffix.
    assert!(!name_bytes_match(
      &os(b"tokenizer.json"),
      model_pat.0,
      model_pat.1
    ));
    assert!(!name_bytes_match(
      &os(b"model.bin"),
      model_pat.0,
      model_pat.1
    ));
    assert!(!name_bytes_match(
      &os(b"model.safetensors"),
      weight_pat.0,
      weight_pat.1
    ));
    assert!(!name_bytes_match(
      &os(b".safetensors"),
      model_pat.0,
      model_pat.1
    ));
    // A non-UTF-8 name that does NOT match the pattern must not be flagged.
    assert!(!name_bytes_match(
      &os(&[b'x', 0xFF]),
      model_pat.0,
      model_pat.1
    ));
  }

  // ───────────── recursive nested-shard load tests ─────────────

  /// Write a single-tensor `<dir>/<name>` safetensors whose only key is
  /// `tensor_key`, for the recursive-load tests.
  fn write_one_tensor(path: &Path, tensor_key: &str) {
    let mut weights: EmbeddingWeights = HashMap::new();
    weights.insert(
      tensor_key.to_owned(),
      Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap(),
    );
    crate::io::save_safetensors(path, &weights).unwrap();
  }

  #[test]
  fn load_weights_recurses_and_prefixes_nested_shards() {
    // A multi-component layout: a ROOT `model.safetensors` plus a NESTED
    // `vision_model/model.safetensors`. mlx-embeddings loads both recursively
    // and prefixes the nested shard's keys with the IMMEDIATE parent folder
    // name (`vision_model.`); the root shard's keys stay verbatim.
    let dir = fresh_dir("recursive");
    write_one_tensor(&dir.join("model.safetensors"), "embeddings.weight");
    let vision = dir.join("vision_model");
    std::fs::create_dir_all(&vision).unwrap();
    write_one_tensor(&vision.join("model.safetensors"), "encoder.weight");

    let weights = load_weights(&dir).expect("recursive load");
    assert!(
      weights.contains_key("embeddings.weight"),
      "root-shard key must be verbatim; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      weights.contains_key("vision_model.encoder.weight"),
      "nested-shard key must be `<folder>.<key>`; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 2);
  }

  #[test]
  fn load_weights_handles_nested_only_models() {
    // A model with NO root shard, only nested-component shards (the case the
    // flat loader silently reported as "missing weights"). Both nested shards
    // load, each prefixed by its own immediate parent folder.
    let dir = fresh_dir("nested-only");
    let vision = dir.join("vision_model");
    let text = dir.join("text_model");
    std::fs::create_dir_all(&vision).unwrap();
    std::fs::create_dir_all(&text).unwrap();
    write_one_tensor(&vision.join("model.safetensors"), "w");
    write_one_tensor(&text.join("model.safetensors"), "w");

    let weights = load_weights(&dir).expect("nested-only load");
    assert!(weights.contains_key("vision_model.w"));
    assert!(weights.contains_key("text_model.w"));
    assert_eq!(weights.len(), 2);
  }

  #[test]
  fn load_weights_prefixes_with_immediate_parent_only() {
    // A shard two levels deep `a/b/model.safetensors` is prefixed with the
    // IMMEDIATE parent's name (`b.`), NOT the full relative path (`a.b.`) —
    // matching mlx-embeddings' `Path(wf).parent.name`.
    let dir = fresh_dir("deep-prefix");
    let deep = dir.join("a").join("b");
    std::fs::create_dir_all(&deep).unwrap();
    write_one_tensor(&deep.join("model.safetensors"), "w");

    let weights = load_weights(&dir).expect("deep nested load");
    assert!(
      weights.contains_key("b.w"),
      "prefix must be the immediate parent folder name; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(!weights.contains_key("a.b.w"));
  }

  #[test]
  fn load_weights_backcompat_weight_glob_is_root_only() {
    // The legacy `weight*.safetensors` retry is NOT recursive: a nested
    // `weight.safetensors` (with no `model*.safetensors` anywhere) must NOT be
    // discovered — only a ROOT-level `weight*.safetensors` is.
    let dir = fresh_dir("backcompat");
    write_one_tensor(&dir.join("weights.safetensors"), "root.w");
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    write_one_tensor(&sub.join("weight.safetensors"), "nested.w");

    let weights = load_weights(&dir).expect("back-compat load");
    assert!(weights.contains_key("root.w"));
    assert!(
      !weights.contains_key("sub.nested.w") && !weights.contains_key("nested.w"),
      "the legacy weight glob is root-only; nested weight*.safetensors must be ignored; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 1);
  }

  // ───────── glob-faithful recursive-walk tests (Codex review) ─────────
  // `collect_glob_shards` drives `glob::glob_with` with
  // `MatchOptions { require_literal_leading_dot: false, .. }` — the faithful
  // port of `glob.glob("**/model*.safetensors", recursive=True,
  // include_hidden=False)`: `**` recursion + directory-symlink follow (with
  // cycle termination) + `scandir`-error suppression are the `glob` crate's
  // job, the hidden (`.`-prefixed) component exclusion is re-applied explicitly
  // by `path_has_hidden_component` (the field is `false` so `glob`'s own
  // `to_str().unwrap()` hidden-filter — which panics on a non-UTF-8 sibling —
  // is never reached), and a `model*.safetensors`-named non-regular entry is
  // rejected by the per-match stat gate.

  #[test]
  fn load_weights_excludes_hidden_directory_shards() {
    // `include_hidden=False` (glob's default): a `.`-prefixed directory is NOT
    // descended — a stale `.hidden/model.safetensors` (e.g. an
    // `.ipynb_checkpoints/` backup) must not be loaded, while a normal
    // `vision_model/model.safetensors` IS. Were the hidden shard discovered, its
    // immediate-parent prefix scheme could let it silently override real
    // weights.
    let dir = fresh_dir("hidden-dir");
    let vision = dir.join("vision_model");
    std::fs::create_dir_all(&vision).unwrap();
    write_one_tensor(&vision.join("model.safetensors"), "encoder.weight");
    let hidden = dir.join(".hidden");
    std::fs::create_dir_all(&hidden).unwrap();
    write_one_tensor(&hidden.join("model.safetensors"), "stale.weight");

    let weights = load_weights(&dir).expect("load with a hidden sibling dir");
    assert!(
      weights.contains_key("vision_model.encoder.weight"),
      "the normal nested shard must load; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      !weights.contains_key(".hidden.stale.weight")
        && !weights.contains_key("hidden.stale.weight")
        && !weights.contains_key("stale.weight"),
      "a `.`-prefixed directory's shard must be excluded (include_hidden=False); got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 1);
  }

  #[test]
  fn load_weights_excludes_hidden_file_shards() {
    // `include_hidden=False` also excludes a `.`-prefixed FILE: a
    // `.model.safetensors` at the root is not matched even though it satisfies
    // the `model*.safetensors` predicate (the leading `.` makes it a hidden
    // path component glob skips).
    let dir = fresh_dir("hidden-file");
    write_one_tensor(&dir.join("model.safetensors"), "real.weight");
    write_one_tensor(&dir.join(".model.safetensors"), "hidden.weight");

    let weights = load_weights(&dir).expect("load with a hidden-file sibling");
    assert!(weights.contains_key("real.weight"));
    assert!(
      !weights.contains_key("hidden.weight"),
      "a `.`-prefixed shard file must be excluded (include_hidden=False); got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 1);
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_follows_symlinked_component_directory() {
    // `**` follows directory symlinks. A symlinked component
    // `text_model -> ../real_text_model` whose target holds a `model.safetensors`
    // IS followed and loaded — and the immediate-parent prefix is the SYMLINK
    // name (`text_model`), the path as glob walked it, NOT the canonicalized
    // target directory name (`real_text_model`).
    let base = fresh_dir("symlink-dir");
    let model_dir = base.join("model");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_one_tensor(&model_dir.join("model.safetensors"), "root.weight");
    // The real target lives OUTSIDE the model dir, so it is reachable only via
    // the symlink (proving the symlink itself is followed).
    let real_text = base.join("real_text_model");
    std::fs::create_dir_all(&real_text).unwrap();
    write_one_tensor(&real_text.join("model.safetensors"), "encoder.weight");
    std::os::unix::fs::symlink(&real_text, model_dir.join("text_model")).unwrap();

    let weights = load_weights(&model_dir).expect("symlinked component dir must load");
    assert!(weights.contains_key("root.weight"));
    assert!(
      weights.contains_key("text_model.encoder.weight"),
      "the symlinked component must load with the SYMLINK name as prefix; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      !weights.contains_key("real_text_model.encoder.weight"),
      "the prefix must be the path-as-walked symlink name, not the canonical target"
    );
    assert_eq!(weights.len(), 2);
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_symlink_cycle_terminates() {
    // A directory symlink pointing to an ANCESTOR creates a cycle. The load
    // must TERMINATE — never hang or stack-overflow — and still discover the
    // legitimate shards.
    //
    // Termination here is the `glob` crate's (and Python `glob`'s) behavior:
    // `**` follows directory symlinks and neither side detects the cycle
    // structurally — the walk down `sub/loop/sub/loop/...` simply stops once
    // the OS `ELOOP` limit refuses to resolve a path that deep. The prior
    // hand-rolled walk additionally deduped the cycle with a recursion-stack of
    // canonicalized paths, yielding *exactly* the 2 underlying shards; that
    // dedup was a DIVERGENCE from Python `glob` (Python `glob` does NOT dedup a
    // symlink cycle — it relies on `ELOOP` just like this). The structural port
    // to the `glob` crate faithfully drops that divergence, so the merged map
    // legitimately also carries the cycle's shards under extra `<folder>.`
    // prefixes (e.g. `loop.root.weight`) — the same keys Python `glob` +
    // `load_model`'s immediate-parent prefix would produce. The load-bearing
    // contract is unchanged: the load terminates, and every real tensor is
    // present and correct.
    let dir = fresh_dir("symlink-cycle");
    write_one_tensor(&dir.join("model.safetensors"), "root.weight");
    let sub = dir.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    write_one_tensor(&sub.join("model.safetensors"), "nested.weight");
    // `sub/loop` points back at the model root → `root/sub/loop/sub/loop/...`.
    std::os::unix::fs::symlink(&dir, sub.join("loop")).unwrap();

    // `expect` (rather than a hang / stack overflow) IS the termination
    // assertion — a non-terminating cycle would never reach it.
    let weights = load_weights(&dir).expect("a symlink cycle must terminate, not hang");
    // The two real shards are discovered, verbatim and immediate-parent-prefixed.
    assert!(
      weights.contains_key("root.weight"),
      "the root shard must load; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      weights.contains_key("sub.nested.weight"),
      "the nested shard must load with its immediate-parent prefix; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    // The cycle's extra prefixed aliases (`loop.…`) are expected and harmless —
    // they reference the SAME underlying tensors, so every value in the merged
    // map is one of the two single-element tensors `write_one_tensor` wrote.
    for (key, value) in &weights {
      assert_eq!(
        value.shape(),
        vec![2],
        "every merged weight (including a cycle alias) must be a real shard \
         tensor; key {key:?} has shape {:?}",
        value.shape()
      );
    }
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_walks_both_aliases_to_one_real_directory() {
    // Two DISTINCT walked component directories alias the SAME real directory:
    // a symlink `text_model -> real_text_model` sitting next to `real_text_model`
    // itself, both directly under the model root. `glob`'s `**` walks EACH path
    // with its own returned prefix, so BOTH `text_model.<key>` and
    // `real_text_model.<key>` must appear. A global, never-removed visited-set
    // would canonicalize-dedup the second alias `read_dir` reached, silently
    // dropping a whole component's tensors with a filesystem-iteration-order
    // -dependent surviving prefix. The recursion-stack guard (insert-before /
    // remove-after) only blocks an ANCESTOR, so two siblings aliasing one target
    // are each walked.
    let dir = fresh_dir("alias-dirs");
    write_one_tensor(&dir.join("model.safetensors"), "root.weight");
    // The real component directory, directly under the model root.
    let real_text = dir.join("real_text_model");
    std::fs::create_dir_all(&real_text).unwrap();
    write_one_tensor(&real_text.join("model.safetensors"), "encoder.weight");
    // A sibling symlink to it — a second, distinct path to the same real dir.
    std::os::unix::fs::symlink(&real_text, dir.join("text_model")).unwrap();

    let weights = load_weights(&dir).expect("aliased component dirs must both load");
    assert!(weights.contains_key("root.weight"));
    assert!(
      weights.contains_key("real_text_model.encoder.weight"),
      "the real directory alias must be walked; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      weights.contains_key("text_model.encoder.weight"),
      "the symlink alias to the SAME real dir must ALSO be walked (each path \
       keeps its own prefix); got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    // root + the two aliased shards' prefixed keys — neither alias dropped.
    assert_eq!(weights.len(), 3);
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_dangling_model_shard_fails_not_falls_back() {
    // A `model.safetensors` that is a DANGLING symlink, next to a real
    // `weight*.safetensors`. `glob`'s name-based match yields the broken
    // `model*.safetensors` path → the load FAILS loudly. The walker must NOT
    // silently skip the unresolvable shard and degrade to the stale legacy
    // `weight*.safetensors` snapshot.
    let dir = fresh_dir("dangling-shard");
    // A legacy root shard that the buggy fallback would have loaded instead.
    write_one_tensor(&dir.join("weights.safetensors"), "stale.weight");
    // `model.safetensors` -> a nonexistent target: a dangling symlink.
    std::os::unix::fs::symlink(
      dir.join("does-not-exist.safetensors"),
      dir.join("model.safetensors"),
    )
    .unwrap();

    let result = load_weights(&dir);
    let Err(err) = result else {
      panic!(
        "a dangling `model.safetensors` must fail the load, not fall back to \
         the stale `weight*.safetensors`"
      );
    };
    // A recoverable discovery error — must mention the broken shard, NOT be a
    // success that loaded `stale.weight`.
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    let msg = err.to_string();
    assert!(
      msg.contains("model.safetensors"),
      "the error must name the broken shard; got: {msg}"
    );
  }

  #[test]
  fn load_weights_model_named_directory_fails_not_falls_back() {
    // A `model.safetensors` that is a real DIRECTORY, next to a legacy root
    // `weights.safetensors`. The walk must REJECT the `model*.safetensors`-named
    // non-regular entry (a directory is non-regular) BEFORE descending into it —
    // descending an (empty) `model.safetensors/` directory would discover no
    // shard there, the canonical shard name would vanish, and `load_weights`
    // would silently degrade to the stale legacy `weight*.safetensors` snapshot.
    let dir = fresh_dir("model-named-dir");
    // The legacy root shard the buggy fallback would have loaded instead.
    write_one_tensor(&dir.join("weights.safetensors"), "stale.weight");
    // `model.safetensors` is a real directory, not a file.
    std::fs::create_dir_all(dir.join("model.safetensors")).unwrap();

    let result = load_weights(&dir);
    let Err(err) = result else {
      panic!(
        "a `model.safetensors` DIRECTORY must fail the load, not be descended \
         and fall back to the stale `weight*.safetensors`"
      );
    };
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    let msg = err.to_string();
    assert!(
      msg.contains("model.safetensors"),
      "the error must name the offending entry; got: {msg}"
    );
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_model_named_symlink_to_directory_fails() {
    // A `model.safetensors` that is a SYMLINK-TO-DIRECTORY — `fs::metadata`
    // dereferences it, so `meta.is_dir()` is true. Like a real directory, this
    // `model*.safetensors`-named non-regular entry must be rejected BEFORE the
    // directory-descent branch, not silently walked.
    let dir = fresh_dir("model-named-symlink-dir");
    write_one_tensor(&dir.join("weights.safetensors"), "stale.weight");
    // A real directory elsewhere, then a `model.safetensors` symlink to it.
    let real = dir.join("real_dir");
    std::fs::create_dir_all(&real).unwrap();
    std::os::unix::fs::symlink(&real, dir.join("model.safetensors")).unwrap();

    let result = load_weights(&dir);
    let Err(err) = result else {
      panic!("a `model.safetensors` symlink-to-directory must fail the load");
    };
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    let msg = err.to_string();
    assert!(
      msg.contains("model.safetensors"),
      "the error must name the offending entry; got: {msg}"
    );
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_suppresses_unreadable_subdir() {
    // `glob` swallows a `scandir` `OSError`: an unreadable nested directory is
    // SKIPPED and the walk still finds shards elsewhere — one bad nested dir
    // must not abort a load whose real weights live in a sibling directory.
    use std::os::unix::fs::PermissionsExt;

    let dir = fresh_dir("unreadable-subdir");
    let vision = dir.join("vision_model");
    std::fs::create_dir_all(&vision).unwrap();
    write_one_tensor(&vision.join("model.safetensors"), "encoder.weight");
    let locked = dir.join("locked");
    std::fs::create_dir_all(&locked).unwrap();
    // Mode 0o000: no read/execute → `read_dir` fails with EACCES.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();

    // Probe whether the env actually enforces the permission (running as root,
    // or some CI filesystems, bypass it). If not, this one case cannot be
    // exercised here — the suppress-error behavior is still implemented and
    // covered by the dangling-symlink path.
    let enforced = std::fs::read_dir(&locked).is_err();
    let result = load_weights(&dir);
    // Always restore so `fresh_dir`'s `remove_dir_all` cleanup can run.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

    if !enforced {
      eprintln!(
        "skipping unreadable-subdir assertion: this environment does not \
         enforce directory read permission"
      );
      return;
    }
    let weights = result.expect("an unreadable nested dir must be skipped, not fail the load");
    assert!(
      weights.contains_key("vision_model.encoder.weight"),
      "the readable sibling's shard must still load; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 1);
  }

  #[test]
  fn load_weights_glob_recurses_deeply_and_excludes_hidden() {
    // Combined `**`-recursion + explicit `path_has_hidden_component` exclusion:
    // the `glob` `**` matches `model*.safetensors` at the root AND in
    // arbitrarily deep subdirectories, while a `.`-prefixed component ANYWHERE
    // below the model dir — whether a hidden directory, a hidden FILE, or a
    // hidden directory ABOVE an otherwise-normal shard — is excluded
    // (`include_hidden=False`).
    let dir = fresh_dir("glob-deep-hidden");
    // (a) a root shard — `**` matches the model dir itself.
    write_one_tensor(&dir.join("model.safetensors"), "root.weight");
    // (b) a DEEPLY nested shard `a/b/c/model.safetensors` — `**` recurses with
    //     no depth cap; the prefix is the IMMEDIATE parent (`c`).
    let deep = dir.join("a").join("b").join("c");
    std::fs::create_dir_all(&deep).unwrap();
    write_one_tensor(&deep.join("model.safetensors"), "deep.weight");
    // (c) a hidden DIRECTORY shard — excluded (the `.`-component is not
    //     descended).
    let hidden_dir = dir.join(".checkpoints");
    std::fs::create_dir_all(&hidden_dir).unwrap();
    write_one_tensor(&hidden_dir.join("model.safetensors"), "hidden_dir.weight");
    // (d) a hidden FILE shard at the root — excluded (the leading `.` makes it
    //     a hidden path component `**`/`*` will not match).
    write_one_tensor(&dir.join(".model.safetensors"), "hidden_file.weight");
    // (e) a normal shard under a hidden ANCESTOR — excluded: a `.`-component
    //     anywhere on the path disqualifies the whole match.
    let under_hidden = dir.join(".secret").join("text_model");
    std::fs::create_dir_all(&under_hidden).unwrap();
    write_one_tensor(
      &under_hidden.join("model.safetensors"),
      "under_hidden.weight",
    );

    let weights = load_weights(&dir).expect("deep recursive glob load");
    assert!(
      weights.contains_key("root.weight"),
      "the root shard must load (** matches the model dir itself); got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      weights.contains_key("c.deep.weight"),
      "the deeply-nested shard must load, prefixed by its immediate parent `c`; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    // None of the hidden-path shards may appear, under any prefix spelling.
    for forbidden in [
      "hidden_dir.weight",
      ".checkpoints.hidden_dir.weight",
      "checkpoints.hidden_dir.weight",
      "hidden_file.weight",
      "under_hidden.weight",
      "text_model.under_hidden.weight",
    ] {
      assert!(
        !weights.contains_key(forbidden),
        "a `.`-prefixed path component must exclude its shard \
         (path_has_hidden_component); leaked {forbidden:?} in {:?}",
        weights.keys().collect::<Vec<_>>()
      );
    }
    // Exactly the two non-hidden shards.
    assert_eq!(weights.len(), 2);
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_non_utf8_model_dir_is_recoverable_error() {
    // `glob_with` takes a `&str` and `unwrap()`s `Path::to_str()` internally,
    // so a non-UTF-8 model directory path would PANIC inside the crate. The
    // up-front `dir.to_str()` guard turns that into a recoverable
    // `Error::Backend` instead — the check fires before any filesystem I/O, so
    // the directory need not (and, on a UTF-8-enforcing host filesystem,
    // cannot) actually exist on disk.
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

    // A directory path with a non-UTF-8 byte (0xFF) in its final component.
    let mut raw: Vec<u8> = b"/tmp/mlxrs-emb-non-utf8-".to_vec();
    raw.push(0xFF);
    let bad_dir = PathBuf::from(OsStr::from_bytes(&raw));
    assert!(
      bad_dir.to_str().is_none(),
      "test precondition: the constructed path must be non-UTF-8"
    );

    let Err(err) = load_weights(&bad_dir) else {
      panic!("a non-UTF-8 model dir path must be a recoverable error, not a panic");
    };
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    assert!(
      err.to_string().contains("not valid UTF-8"),
      "the error should explain the non-UTF-8 path rejection; got: {err}"
    );
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_non_utf8_descendant_does_not_panic() {
    // A non-UTF-8 *descendant* name (distinct from a non-UTF-8 model-dir path):
    // `glob 0.3.3`'s `require_literal_leading_dot: true` hidden-filter calls
    // `file_name().to_str().unwrap()` on EVERY scanned directory child
    // (`glob-0.3.3/src/lib.rs:953-955`), so a single non-UTF-8 sibling inside an
    // otherwise-UTF-8 model directory would panic the whole process. Driving
    // `glob_with` with `require_literal_leading_dot: false` gates that `unwrap`
    // path off; `collect_glob_shards` must therefore walk a directory holding a
    // non-UTF-8 entry WITHOUT panicking and still load the legitimate
    // `model.safetensors` shards.
    //
    // macOS/APFS enforces UTF-8 file names and will reject creating the
    // non-UTF-8 entry — the test then `return`s cleanly (the no-panic code path,
    // not this fixture, is the deliverable; on a mounted NFS/exFAT/case
    // -sensitive volume the entry creates and the walk is exercised for real).
    use std::os::unix::ffi::OsStringExt;

    let dir = fresh_dir("non-utf8-child");
    // A legitimate root shard plus a legitimate nested shard — both must still
    // load once the non-UTF-8 sibling no longer aborts the walk.
    write_one_tensor(&dir.join("model.safetensors"), "root.weight");
    let nested = dir.join("text_model");
    std::fs::create_dir_all(&nested).unwrap();
    write_one_tensor(&nested.join("model.safetensors"), "encoder.weight");

    // A non-UTF-8 file name (`m` then byte 0xFF) directly inside the model dir,
    // so `glob`'s `**` expansion `read_dir`s it as a sibling of the real shard.
    let non_utf8_name = std::ffi::OsString::from_vec(vec![b'm', 0xFF]);
    if std::fs::write(dir.join(&non_utf8_name), b"junk").is_err() {
      // APFS (and any UTF-8-enforcing filesystem) rejects the name — the
      // panic-free walk cannot be exercised here; skip without failing.
      return;
    }
    // Also place a non-UTF-8-named entry one level down, so the nested
    // directory's `read_dir` children list is non-UTF-8 too.
    let _ = std::fs::write(nested.join(&non_utf8_name), b"junk");

    // The deliverable: the walk completes without a panic. The non-UTF-8 entry
    // is not named `model*.safetensors` (it is not even ASCII) so it never
    // matches the pattern; the two legitimate shards still load.
    let weights = load_weights(&dir).expect("a non-UTF-8 descendant must not break the glob walk");
    assert!(
      weights.contains_key("root.weight"),
      "the root shard must still load past a non-UTF-8 sibling; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert!(
      weights.contains_key("text_model.encoder.weight"),
      "the nested shard must still load past a non-UTF-8 sibling; got {:?}",
      weights.keys().collect::<Vec<_>>()
    );
    assert_eq!(weights.len(), 2);
  }

  #[cfg(unix)]
  #[test]
  fn load_weights_non_utf8_nested_parent_is_recoverable_error() {
    // A NESTED shard `<bad>/model.safetensors` whose immediate parent FOLDER
    // name is not valid UTF-8. The shard file itself is ASCII-named, so the
    // `model*.safetensors` glob DOES yield it — but the non-UTF-8 parent folder
    // name cannot become the `String` key prefix. It must fail loudly with a
    // path-naming `Error::Backend`, NOT silently collapse to a `None` prefix
    // (which would mis-merge the nested shard's keys verbatim and could collide
    // with a real root shard).
    //
    // macOS/APFS enforces UTF-8 directory names and will reject creating the
    // non-UTF-8 folder — the test then `return`s cleanly (the error code path,
    // not this fixture, is the deliverable; on a mounted NFS/exFAT/case
    // -sensitive volume the folder creates and the error is exercised for real).
    use std::os::unix::ffi::OsStringExt;

    let dir = fresh_dir("non-utf8-parent");
    // A child directory whose name is `t` followed by an invalid byte (0xFF).
    let bad_folder = std::ffi::OsString::from_vec(vec![b't', 0xFF]);
    let nested = dir.join(&bad_folder);
    if std::fs::create_dir_all(&nested).is_err() {
      // APFS (and any UTF-8-enforcing filesystem) rejects the directory name —
      // the error code path cannot be exercised here; skip without failing.
      return;
    }
    // The ASCII-named shard inside the non-UTF-8 folder — this is what the glob
    // matches and what `collect_glob_shards` must reject for its bad prefix.
    write_one_tensor(&nested.join("model.safetensors"), "encoder.weight");

    let Err(err) = load_weights(&dir) else {
      panic!(
        "a nested shard under a non-UTF-8 parent folder must be a recoverable error, not a \
         silent root-merge or a panic"
      );
    };
    assert!(
      matches!(err, Error::Backend { .. }),
      "expected a recoverable Backend error; got {err:?}"
    );
    let msg = err.to_string();
    assert!(
      msg.contains("non-UTF-8 parent directory name"),
      "the error should explain the non-UTF-8 parent rejection; got: {msg}"
    );
    // The error must name the offending shard path: its UTF-8 prefix is the
    // model directory, which is present in the path `Display`.
    assert!(
      msg.contains(&dir.display().to_string()),
      "the error should name the offending shard path; got: {msg}"
    );
  }

  #[test]
  fn load_pooling_validated_before_heavy_io() {
    // A MALFORMED `1_Pooling/config.json` must fail the load even when the
    // weights/tokenizer would be expensive/invalid to load — proving the cheap
    // pooling validation runs BEFORE the heavy I/O. The `model.safetensors`
    // here is deliberately INVALID: if `load()` reached the weight load it
    // would surface a safetensors parse error instead of the pooling error.
    let dir = fresh_dir("pooling-first");
    std::fs::write(dir.join("config.json"), mock_config_json("mockemb")).unwrap();
    std::fs::write(dir.join("model.safetensors"), b"not a safetensors file").unwrap();
    // No tokenizer.json either — another heavy step that would fail later.
    let pooling_dir = dir.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir).unwrap();
    std::fs::write(pooling_dir.join("config.json"), b"{ not valid json").unwrap();

    let registry = EmbeddingModelTypeRegistry::new().with("mockemb", mock_constructor());
    let config = EmbeddingModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("malformed pooling config must error");
    };
    let msg = err.to_string();
    // The error must be about the pooling config, NOT a safetensors/tokenizer
    // failure (which would mean the heavy I/O ran first).
    assert!(
      !msg.contains("safetensors") && !msg.contains("no model weights"),
      "pooling must be validated before the weight load; got: {msg}"
    );
  }
}
