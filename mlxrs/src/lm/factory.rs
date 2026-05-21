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

use crate::{
  error::{Error, Result},
  lm::{
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
    let model_type = remap_model_type(&loaded.config.model_type);
    let constructor = self
      .creators
      .get(model_type)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "unsupported model type {:?}: no constructor registered (register one via \
         ModelTypeRegistry::register)",
          loaded.config.model_type
        ),
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
  if !registry.contains(&config.model_type) {
    return Err(Error::Backend {
      message: format!(
        "unsupported model type {:?}: no constructor registered (register one via \
         ModelTypeRegistry::register)",
        config.model_type
      ),
    });
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

#[cfg(test)]
mod tests {
  //! End-to-end load-factory tests, driven by a **mock** model type
  //! registered into a fresh [`ModelTypeRegistry`] (per the project's
  //! no-model-arch rule, this PR ships the seam, not architectures — so the
  //! end-to-end path is proven against a hand-traced mock constructor over a
  //! temp model directory).

  use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
  };

  use super::*;
  use crate::{array::Array, lm::cache::KvCache};

  /// A minimal `config.json` for the mock architecture. `model_type` is the
  /// registry key; the remaining fields are exactly the required keys of the
  /// typed [`Config`] (so the reused [`crate::lm::load::load`] parse
  /// succeeds). `mock_extra` is a model-specific key OUTSIDE the typed
  /// subset, used to prove the constructor can read
  /// [`LoadedModel::config_json`].
  fn mock_config_json(model_type: &str) -> String {
    format!(
      r#"{{
        "model_type": "{model_type}",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": 5,
        "tie_word_embeddings": false,
        "mock_extra": 7
      }}"#
    )
  }

  /// A trivial [`Model`] the mock constructor returns. It records the vocab
  /// size it was built with (read off [`Config::vocab_size`]) and the value
  /// of the model-specific `mock_extra` config key (read off the raw JSON),
  /// so a test can assert the constructor saw both the typed config and the
  /// raw config body. `forward` returns a fixed `[B, S, vocab]` zero logits
  /// (the generation loop is exercised elsewhere; here we only prove
  /// dispatch).
  struct MockLoadedModel {
    vocab: i32,
    #[allow(dead_code)]
    mock_extra: i64,
  }

  impl Model for MockLoadedModel {
    fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let (batch, seq) = match tokens.shape().as_slice() {
        [b, s] => (*b, *s),
        [s] => (1, *s),
        other => {
          return Err(Error::ShapeMismatch {
            message: format!("MockLoadedModel::forward expects [B, S], got {other:?}"),
          });
        }
      };
      let vocab = self.vocab as usize;
      Array::from_slice::<f32>(&vec![0.0_f32; batch * seq * vocab], &(batch, seq, vocab))
    }
  }

  /// Build a [`ModelConstructor`] for the mock architecture: read the typed
  /// `vocab_size` off [`LoadedModel::config`], parse the model-specific
  /// `mock_extra` off [`LoadedModel::config_json`], and assert at least one
  /// weight tensor arrived (proving the weights reached the constructor).
  fn mock_constructor() -> ModelConstructor {
    Box::new(|loaded: &LoadedModel| -> Result<Box<dyn Model>> {
      assert!(
        !loaded.weights.is_empty(),
        "constructor should receive the loaded weights"
      );
      // Model-specific key outside the typed Config subset, read from the
      // raw JSON (the analogue of mlx-swift-lm's per-model Codable init).
      let raw: serde_json::Value =
        serde_json::from_str(&loaded.config_json).map_err(|e| Error::Backend {
          message: format!("mock ctor: bad config json: {e}"),
        })?;
      let mock_extra = raw
        .get("mock_extra")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Error::Backend {
          message: "mock ctor: missing mock_extra".into(),
        })?;
      Ok(Box::new(MockLoadedModel {
        vocab: loaded.config.vocab_size,
        mock_extra,
      }))
    })
  }

  /// A fresh, writable per-test temp directory (the crate's
  /// no-`tempfile`-crate convention: `temp_dir()` + pid + a process-unique
  /// counter so parallel tests never collide). Created empty; the caller
  /// populates it.
  fn fresh_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
      std::env::temp_dir().join(format!("mlxrs-lm-factory-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// Serialize a minimal but loadable `tokenizer.json` (a 3-token WordLevel
  /// model with a Whitespace pre-tokenizer) into `dir` via the `tokenizers`
  /// crate — the same fixture style as `embeddings::encode`'s tests, so the
  /// reused [`Tokenizer::from_path`] loads it.
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

  /// Populate `dir` with just the model's `config.json` (with the given
  /// `model_type`) and a tiny single-tensor `model.safetensors` — but **no**
  /// `tokenizer.json`. The basis for both [`write_model_dir`] (which adds the
  /// tokenizer) and the real split-layout test (where the tokenizer lives in a
  /// separate directory).
  fn write_model_dir_no_tokenizer(dir: &Path, model_type: &str) {
    std::fs::write(dir.join("config.json"), mock_config_json(model_type)).unwrap();

    // A tiny one-tensor safetensors so `load_weights` finds non-empty
    // weights. `save_safetensors` writes the on-disk format the loader reads.
    let mut weights: Weights = HashMap::new();
    weights.insert(
      "mock.weight".to_owned(),
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
    );
    crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  }

  /// Populate `dir` as a minimal but *loadable* model directory: `config.json`
  /// (with the given `model_type`), a tiny single-tensor `model.safetensors`,
  /// and a `tokenizer.json`.
  fn write_model_dir(dir: &Path, model_type: &str) {
    write_model_dir_no_tokenizer(dir, model_type);
    write_tokenizer(dir);
  }

  #[test]
  fn load_dispatches_to_registered_mock_and_returns_model_and_tokenizer() {
    let dir = fresh_dir("dispatch");
    write_model_dir(&dir, "mockarch");
    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_directory(&dir);

    let ctx = load(&config, &registry).expect("load should succeed");

    // The returned config carries the parsed model_type + vocab.
    assert_eq!(ctx.config.model_type, "mockarch");
    assert_eq!(ctx.config.vocab_size, 5);

    // The constructed model is the mock: drive one forward to confirm it is
    // wired and saw the right vocab (logits last-axis == vocab_size).
    let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
    let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
    let logits = ctx.model.forward(&tokens, &mut cache).unwrap();
    assert_eq!(logits.shape(), vec![1, 3, 5]);

    // The tokenizer loaded from the same directory (encode the 3-token vocab).
    let ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(ids.len(), 3);
  }

  #[test]
  fn from_id_resolves_as_local_path() {
    // An `Identifier::Id` is treated as a LOCAL path (no network): pointing
    // it at the temp dir loads exactly as `from_directory` would.
    let dir = fresh_dir("idpath");
    write_model_dir(&dir, "mockarch");
    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_id(dir.to_str().unwrap());
    assert_eq!(config.model_directory(), dir.as_path());

    let ctx = load(&config, &registry).expect("id-as-local-path load should succeed");
    assert_eq!(ctx.config.model_type, "mockarch");
  }

  #[test]
  fn constructor_reads_model_specific_raw_config_key() {
    // The mock constructor reads `mock_extra` (outside the typed Config
    // subset) off the raw JSON; assert it sees the value the test wrote (7).
    let dir = fresh_dir("rawkey");
    write_model_dir(&dir, "mockarch");
    let registry = ModelTypeRegistry::new().with("mockarch", {
      Box::new(|loaded: &LoadedModel| -> Result<Box<dyn Model>> {
        let raw: serde_json::Value = serde_json::from_str(&loaded.config_json).unwrap();
        assert_eq!(raw.get("mock_extra").and_then(|v| v.as_i64()), Some(7));
        Ok(Box::new(MockLoadedModel {
          vocab: loaded.config.vocab_size,
          mock_extra: 7,
        }))
      }) as ModelConstructor
    });
    let config = ModelConfiguration::from_directory(&dir);
    let ctx = load(&config, &registry).expect("load");
    // Sanity: the model was built (the in-ctor assert is the real check).
    let _ = ctx.model;
  }

  #[test]
  fn unknown_model_type_is_recoverable_error() {
    // config.json says "nope" but only "mockarch" is registered → an
    // unsupported-model-type Error (NOT a panic), naming the type.
    let dir = fresh_dir("unknown");
    write_model_dir(&dir, "nope");
    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("unknown model_type must error");
    };
    let msg = err.to_string();
    assert!(msg.contains("unsupported model type"), "got: {msg}");
    assert!(msg.contains("nope"), "error should name the type: {msg}");
  }

  #[test]
  fn missing_config_json_is_recoverable_error() {
    // A directory with NO config.json → a recoverable Error from the reused
    // loader (naming config.json), never a panic.
    let dir = fresh_dir("noconfig");
    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("missing config.json must error");
    };
    assert!(
      err.to_string().contains("config.json"),
      "error should name config.json: {err}"
    );
  }

  #[test]
  fn registry_contains_and_remapping() {
    // Registration is keyed on the CANONICAL id: "mistral" remaps to
    // "llama", so registering under "mistral" is found under "llama" too.
    let registry = ModelTypeRegistry::new().with("mistral", mock_constructor());
    assert!(registry.contains("mistral"));
    assert!(registry.contains("llama"));
    assert!(!registry.contains("qwen3"));
    assert_eq!(remap_model_type("mistral"), "llama");
    assert_eq!(remap_model_type("qwen3"), "qwen3");
  }

  #[test]
  fn register_replaces_and_returns_previous() {
    let mut registry = ModelTypeRegistry::new();
    assert!(registry.register("mockarch", mock_constructor()).is_none());
    // A second registration of the same canonical id returns the displaced
    // constructor (last-writer-wins, mirroring the Swift dict assignment).
    assert!(registry.register("mockarch", mock_constructor()).is_some());
  }

  #[test]
  fn tokenizer_source_loads_from_separate_directory() {
    // REAL split layout (Codex finding #1): the model dir has config +
    // weights but NO `tokenizer.json`; a SEPARATE dir holds the tokenizer, and
    // `tokenizer_source` points the load there. This MUST fail on the old
    // orchestration (which always built `Tokenizer::from_path(model_dir)`
    // first, before ever consulting `tokenizer_source`) and succeed now that
    // the tokenizer dir is selected up front and loaded exactly once.
    let model_dir = fresh_dir("split-model");
    write_model_dir_no_tokenizer(&model_dir, "mockarch");
    // Prove there is genuinely no tokenizer in the model dir.
    assert!(!model_dir.join("tokenizer.json").exists());
    let tok_dir = fresh_dir("split-tok");
    write_tokenizer(&tok_dir);

    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_directory(&model_dir).with_tokenizer_source(&tok_dir);
    assert_eq!(config.tokenizer_directory(), tok_dir.as_path());

    let ctx = load(&config, &registry).expect("split-tokenizer load should succeed");
    let ids = ctx.tokenizer.encode("a b c", false).unwrap();
    assert_eq!(ids.len(), 3);
  }

  #[test]
  fn unsupported_model_type_does_not_touch_weights_or_tokenizer() {
    // Codex finding #2: an UNREGISTERED `model_type` must be rejected BEFORE
    // any weights/tokenizer are loaded. The model dir's `config.json` names an
    // unregistered type and its `model.safetensors` is deliberately INVALID
    // (not a real safetensors) — if `load()` tried to load weights it would
    // surface a parse/IO error from that file; instead it must return the
    // recoverable unsupported-model error, proving weights were never touched.
    // There is no `tokenizer.json` either, so a tokenizer load would also
    // fail; the unsupported-model error proves neither was attempted.
    let dir = fresh_dir("unsupported-cheap");
    std::fs::write(dir.join("config.json"), mock_config_json("nope")).unwrap();
    // Garbage where a safetensors would be: loading it would error loudly.
    std::fs::write(
      dir.join("model.safetensors"),
      b"this is not a safetensors file",
    )
    .unwrap();
    assert!(!dir.join("tokenizer.json").exists());

    // Registry knows only "mockarch"; "nope" is unregistered.
    let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
    let config = ModelConfiguration::from_directory(&dir);

    let Err(err) = load(&config, &registry) else {
      panic!("unsupported model_type must error");
    };
    // The error is the recoverable unsupported-model one (naming the type),
    // NOT a weights/tokenizer parse error.
    let msg = err.to_string();
    assert!(
      msg.contains("unsupported model type"),
      "expected the unsupported-model error before any weight load, got: {msg}"
    );
    assert!(msg.contains("nope"), "error should name the type: {msg}");
    // Belt-and-suspenders: the message must not be the invalid-weights one.
    assert!(
      !msg.contains("safetensors") && !msg.contains("weights"),
      "weights must not have been loaded, but the error mentions them: {msg}"
    );
  }

  #[test]
  fn raw_config_json_matches_parsed_config() {
    // Codex finding #3: the `config_json` handed to the constructor must be the
    // SAME content that was parsed into the typed `Config` (one read, not two).
    // The constructor captures both and asserts they agree: the raw JSON
    // parses back to the same `model_type`/`vocab_size`/`mock_extra`, and is
    // byte-identical to the on-disk `config.json` the test wrote.
    let dir = fresh_dir("raw-consistency");
    write_model_dir(&dir, "mockarch");
    let on_disk = std::fs::read_to_string(dir.join("config.json")).unwrap();

    let captured: std::sync::Arc<std::sync::Mutex<Option<String>>> =
      std::sync::Arc::new(std::sync::Mutex::new(None));
    let captured_in_ctor = std::sync::Arc::clone(&captured);
    let registry = ModelTypeRegistry::new().with("mockarch", {
      Box::new(move |loaded: &LoadedModel| -> Result<Box<dyn Model>> {
        // Raw JSON parses back to the SAME typed fields as `loaded.config`.
        let raw: serde_json::Value = serde_json::from_str(&loaded.config_json).unwrap();
        assert_eq!(
          raw.get("model_type").and_then(|v| v.as_str()),
          Some(loaded.config.model_type.as_str())
        );
        assert_eq!(
          raw.get("vocab_size").and_then(|v| v.as_i64()),
          Some(loaded.config.vocab_size as i64)
        );
        *captured_in_ctor.lock().unwrap() = Some(loaded.config_json.clone());
        Ok(Box::new(MockLoadedModel {
          vocab: loaded.config.vocab_size,
          mock_extra: 7,
        }))
      }) as ModelConstructor
    });
    let config = ModelConfiguration::from_directory(&dir);
    let _ctx = load(&config, &registry).expect("load");

    // The `config_json` the constructor saw is byte-identical to the file the
    // typed `Config` was parsed from (single read — no divergence window).
    let seen = captured.lock().unwrap().clone().expect("ctor ran");
    assert_eq!(seen, on_disk);
  }
}
