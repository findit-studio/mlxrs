//! Architecture-agnostic audio model-load support surface, ported from
//! [`mlx_audio.utils`][audio-utils] (the authoritative spec) and shared
//! across every per-domain loader (TTS / STT / STS / VAD / LID / codec).
//!
//! This is the audio analogue of [`crate::lm::load`]: the per-domain
//! `load` entry points (e.g. [`crate::audio::tts::load::load`]) all funnel
//! through these helpers, exactly as `mlx_audio.{tts,stt,sts,vad,lid}.utils.load`
//! all funnel through `mlx_audio.utils.base_load_model`. The helpers here
//! port **only** the arch-agnostic shape:
//!
//! - [`get_model_path`] — local-path resolver mirroring
//!   `mlx_audio.utils.get_model_path` ([utils.py:106-150][audio-utils-getpath]).
//!   Per the project's no-network policy, mlxrs **rejects** `hf://`-style
//!   paths with a clear [`Error::OutOfRange`] (hub path) or [`Error::MissingKey`]
//!   (missing local path) rather than calling `huggingface_hub.snapshot_download`
//!   — the (`hf://repo/id`) path is purely a hosted-Hub artifact, never a real
//!   on-disk path.
//! - [`load_config`] — bounded read of `<dir>/config.json`, reusing the
//!   shared [`crate::lm::load`] `read_bounded_config_file` primitive,
//!   so audio configs inherit the same hardening (TOCTOU-closed open,
//!   non-regular-reject, `O_NONBLOCK`-on-unix, byte-capped read) every
//!   other config reader in the crate already uses.
//! - [`apply_quantization`] — `nn.quantize(model, …)` analogue. mlx-audio's
//!   [utils.py:207-254][audio-utils-quant] delegates to mlx's `nn.quantize`
//!   with a per-layer predicate that consults the checkpoint's
//!   `weights` for `.scales`. mlxrs's [`crate::lm::quant`] already ports
//!   the equivalent weight-level quantize/dequantize (per-layer-aware
//!   [`crate::lm::quant::PerLayerQuantization`]); this helper is the
//!   audio-side entry point that **parses** the per-layer schema from
//!   `config.json` and returns it, leaving the actual weight
//!   quantization application to the architecture-specific loader (a
//!   model with no `quantization` block returns `Ok(None)`, the no-op
//!   path mlx-audio's [utils.py:222-225][audio-utils-quant] takes).
//! - [`base_load_model`] — the shared factory that resolves the path,
//!   reads `config.json`, parses the optional `quantization` block, and
//!   returns a [`LoadedAudioModel`] bundle — the
//!   `(path, config_json, quantization)` triple the architecture-specific
//!   constructor consumes. mlx-audio's
//!   `base_load_model` ([utils.py:319-414][audio-utils-base]) additionally
//!   instantiates the per-architecture `Model(model_config)` + loads the
//!   safetensors weights + calls `model.sanitize(weights)` +
//!   `model.load_weights(...)`; per the project's no-per-model-arch rule,
//!   mlxrs **does not** port the per-architecture construction — that is
//!   the per-domain loader's caller's job (the analog of mlx-audio's
//!   `model_class.Model(model_config)`).
//!
//! **Deliberately NOT ported** (per the no-network / no-per-model-arch
//! scoping): `huggingface_hub.snapshot_download` (Hub download — local-only
//! per project policy), `get_model_class` (per-arch `importlib.import_module`
//! — mlxrs has no analogous arch registry), per-model `Model(model_config)`
//! construction + `sanitize(weights)` (per-arch — out of scope), the
//! [utils.py:417-472][audio-utils-getters] lazy-import accessors
//! (`_get_tts_utils` / …) — mlxrs's per-domain modules export the load
//! entry points directly without a registry-of-registries.
//!
//! [audio-utils]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py
//! [audio-utils-getpath]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L106-L150
//! [audio-utils-quant]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L207-L254
//! [audio-utils-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L319-L414
//! [audio-utils-getters]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L417-L472

use std::path::{Path, PathBuf};

use smol_str::format_smolstr;

use crate::{
  error::{Error, MissingKeyPayload, OutOfRangePayload, ParsePayload, Result},
  lm::{load::read_bounded_config_file, quant::PerLayerQuantization},
};

/// The bundle [`base_load_model`] returns: the resolved local model
/// directory, the verbatim `config.json` body, and the parsed
/// per-layer-aware quantization (if any).
///
/// The per-domain loader (TTS / STT / STS / VAD / LID / codec) consumes
/// this to construct its concrete model trait object — mirroring the
/// `(model_path, config)` pair mlx-audio's
/// [`base_load_model`][audio-utils-base] threads through to
/// `model_class.Model(model_config)` (plus the `apply_quantization` call
/// that follows). Per the project's no-per-model-arch rule, the actual
/// per-architecture construction (the `model_class.Model(model_config)`
/// step) is the caller's responsibility — this bundle gives them the
/// inputs.
///
/// [audio-utils-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L319-L414
#[derive(Debug, Clone)]
pub struct LoadedAudioModel {
  /// The resolved local model directory (the output of [`get_model_path`]).
  model_path: PathBuf,
  /// The verbatim `config.json` body — kept as the source-of-truth string
  /// so an architecture-specific deserializer (`Codable`-style) can
  /// reparse model-specific keys outside the small typed subset
  /// [`load_config`] returns. Same TOCTOU-closed single-read convention
  /// as [`crate::lm::load::load_config`].
  config_json: String,
  /// Parsed per-layer-aware quantization, if `config.json` carried a
  /// `quantization` block. `None` ⇒ the checkpoint is dense — mlx-audio's
  /// no-op path ([utils.py:222-225][audio-utils-quant]).
  ///
  /// [audio-utils-quant]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L222-L225
  quantization: Option<PerLayerQuantization>,
}

impl LoadedAudioModel {
  /// Construct a [`LoadedAudioModel`] from the resolved model directory,
  /// verbatim `config.json` body, and optional parsed quantization.
  ///
  /// This is the canonical constructor — [`base_load_model`] builds its
  /// return value through this.
  pub fn new(
    model_path: PathBuf,
    config_json: String,
    quantization: Option<PerLayerQuantization>,
  ) -> Self {
    Self {
      model_path,
      config_json,
      quantization,
    }
  }

  /// The resolved local model directory.
  #[inline(always)]
  pub fn model_path(&self) -> &Path {
    &self.model_path
  }

  /// The verbatim `config.json` body.
  #[inline(always)]
  pub fn config_json(&self) -> &str {
    &self.config_json
  }

  /// Parsed per-layer-aware quantization, if any. `None` ⇒ dense checkpoint.
  #[inline(always)]
  pub fn quantization(&self) -> Option<&PerLayerQuantization> {
    self.quantization.as_ref()
  }
}

/// Ensure `path` is a local on-disk model directory — mirroring
/// `mlx_audio.utils.get_model_path` ([utils.py:106-150][audio-utils-getpath]),
/// minus the `huggingface_hub.snapshot_download` Hub-download branch.
///
/// **No-network**: per the project policy, mlxrs never downloads from
/// HuggingFace Hub. A path that mlx-audio would forward to
/// `snapshot_download` (a non-local repo id like `"mlx-community/foo"`
/// or an `hf://…` URL) is **rejected** with a clear typed error
/// naming the offending input, instead of being silently fetched.
/// Callers who need a Hub-fetched directory must run `huggingface-cli
/// download …` (or its programmatic equivalent) **out of process** and
/// pass mlxrs the resulting local path.
///
/// Resolution:
///
/// 1. Expand `~`-style home references (mlx-audio's
///    `Path(path_or_hf_repo).expanduser()`); if the result exists on
///    disk it is returned.
/// 2. If the input *looks* local (starts with `.` / `/` / `~` / `<drive>:`
///    — the [utils.py:96-103][audio-utils-islocal] heuristic) and the
///    path does NOT exist, surface
///    [`Error::MissingKey`] (mlx-audio's `FileNotFoundError`).
/// 3. Otherwise the input would have been forwarded to
///    `snapshot_download` — mlxrs surfaces
///    [`Error::OutOfRange`] explaining the no-network policy and pointing
///    at the out-of-process workaround.
///
/// [audio-utils-getpath]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L106-L150
/// [audio-utils-islocal]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L96-L103
pub fn get_model_path(path: &str) -> Result<PathBuf> {
  let expanded = expand_home(path);

  // (1) Existing local path → done. This matches mlx-audio's
  // `if model_path.exists(): return model_path`.
  if expanded.exists() {
    return Ok(expanded);
  }

  // (2) Looks like a local path but doesn't exist → mlx-audio's
  // FileNotFoundError. Surface a clear Error::MissingKey instead of
  // (3)'s "would have fetched from Hub" message.
  if is_local_path(path) {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "audio model local path not found",
      path,
    )));
  }

  // (3) mlx-audio would call snapshot_download here. mlxrs is local-only
  // by policy: reject with a clear, actionable message.
  //
  // Strip the Hub-URL prefix from the input before interpolating it
  // into the workaround. `huggingface-cli download` expects a
  // repo-id (`org/model`), so feeding it the user's raw `hf://org/model`
  // or `https://huggingface.co/org/model` would print broken advice —
  // strip both forms back to `org/model` first.
  let repo_id = path
    .strip_prefix("hf://")
    .or_else(|| path.strip_prefix("https://huggingface.co/"))
    .or_else(|| path.strip_prefix("http://huggingface.co/"))
    .unwrap_or(path);
  Err(Error::OutOfRange(OutOfRangePayload::new(
    "audio model path (mlxrs does not download from HuggingFace Hub; \
       fetch the model out of process via `huggingface-cli download <repo>` \
       and pass the local path)",
    "must be a local on-disk directory (starting with `.`, `/`, `~`, or `<drive>:`)",
    format_smolstr!("path={path:?}, repo_id={repo_id}"),
  )))
}

/// `path` starts with a marker that mlx-audio's
/// [`_is_local_path`][audio-utils-islocal] treats as "definitely local":
/// `.` (cwd-relative), `/` (absolute Unix), `~` (home), or `<drive>:`
/// (Windows). A repo id like `"mlx-community/foo"` matches none of these
/// and is treated as a Hub id by mlx-audio (rejected here).
///
/// [audio-utils-islocal]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L96-L103
fn is_local_path(path: &str) -> bool {
  path.starts_with('.')
    || path.starts_with('/')
    || path.starts_with('~')
    || (path.len() > 1 && path.as_bytes().get(1) == Some(&b':'))
}

/// Expand a leading `~` to the user's home directory; otherwise pass
/// through. Mirrors `pathlib.Path.expanduser()` for the only case
/// mlx-audio uses (a bare `~` or `~/...`).
fn expand_home(path: &str) -> PathBuf {
  if let Some(rest) = path.strip_prefix("~/") {
    if let Some(home) = home_dir() {
      return home.join(rest);
    }
  } else if path == "~"
    && let Some(home) = home_dir()
  {
    return home;
  }
  PathBuf::from(path)
}

/// `$HOME` (Unix) or `%USERPROFILE%` fallback (Windows). Returns `None`
/// if neither is set — the caller's `~` is then passed through verbatim,
/// matching `pathlib`'s no-op behavior in that environment.
fn home_dir() -> Option<PathBuf> {
  std::env::var_os("HOME")
    .or_else(|| std::env::var_os("USERPROFILE"))
    .map(PathBuf::from)
}

/// Read `<dir>/config.json` once and return its verbatim body — mirroring
/// `mlx_audio.utils.load_config` ([utils.py:153-174][audio-utils-config]),
/// minus the `get_model_path` recursion (the caller already resolved
/// `dir` via [`get_model_path`]).
///
/// Returns the raw JSON string (rather than a typed subset) because audio
/// configs are model-specific — there is no equivalent of the LM-side
/// [`crate::lm::load::Config`] subset that every audio architecture
/// shares. The per-domain loader passes this string to its model-specific
/// deserializer.
///
/// Same TOCTOU-closed / `O_NONBLOCK` / non-regular-reject / byte-capped
/// discipline as [`crate::lm::load::load_config`] — the read goes through
/// the shared [`crate::lm::load`] `read_bounded_config_file` primitive
/// (capped at 1 MiB — the `MAX_CONFIG_BYTES` constant in `lm::load`). A
/// hostile model directory cannot OOM the loader by planting a huge
/// `config.json`.
///
/// An absent `config.json` is a recoverable [`Error::MissingKey`] naming the
/// offending path — matching mlx-audio's `FileNotFoundError(f"Config not found
/// at {model_path}")`. A present-but-unusable `config.json` instead propagates
/// the typed error from the shared bounded reader: [`Error::FileIo`] for a
/// non-regular or unreadable file, [`Error::CapExceeded`] when it exceeds the
/// `MAX_CONFIG_BYTES` ceiling, and [`Error::LayerKeyed`] wrapping
/// [`Error::Parse`] for non-UTF-8 content.
///
/// [audio-utils-config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L153-L174
pub fn load_config(dir: &Path) -> Result<String> {
  let path = dir.join("config.json");
  match read_bounded_config_file(&path, "audio model config")? {
    Some(text) => Ok(text),
    None => Err(Error::MissingKey(MissingKeyPayload::new(
      "audio model config not found",
      format_smolstr!("{}", path.display()),
    ))),
  }
}

/// Parse the optional quantization block of `config_json` into a
/// [`PerLayerQuantization`] — mirroring the input half of mlx-audio's
/// [`apply_quantization`][audio-utils-quant] ([utils.py:207-254][audio-utils-quant]).
///
/// Audio-specific deviations from [`crate::lm::quant::parse_quantization`]
/// (the LM-side parser this previously delegated to verbatim), faithful
/// to mlx-audio's Python ([utils.py:221-226][audio-utils-quant]):
///
/// 1. **Key fallback** ([utils.py:221-223][audio-utils-quant]): try
///    top-level `"quantization"` first, and if that key is absent **or
///    explicitly `null`** fall back to `"quantization_config"` (the
///    longer key HF post-quantize artifacts use —
///    `mlx_audio.utils.apply_quantization` reads both). Python's
///    `config.get("quantization", None)` treats a missing key and a
///    `null` value identically, so an explicit `null` triggers the
///    fallback just like an absent key.
/// 2. **`group_size` default** ([utils.py:226][audio-utils-quant]): if
///    the chosen block has no `"group_size"`, default it to `64` (audio
///    convention — Python's `quantization.get("group_size", 64)`). The
///    LM-side parser rejects missing `group_size` outright (swift
///    `Quantization` requires it), so this is an audio-only relaxation.
///
/// Otherwise identical to the LM parser: the parsed block flows through
/// the shared [`PerLayerQuantization`] deserializer (per-layer overrides
/// and global default), so audio quantized checkpoints inherit the same
/// per-layer schema as LM ones.
///
/// Per the project's no-per-model-arch rule, mlxrs returns the **parsed
/// schema** (the `Option<PerLayerQuantization>`) rather than mutating a
/// `Model` instance — there is no concrete model trait object to consult
/// `to_quantized` on, and the actual quantization application is the
/// per-architecture loader's job (it routes through
/// [`crate::lm::quant::quantize_weights`] with its own
/// [`crate::lm::quant::Eligible`] predicate, the same path the
/// [`crate::lm::convert`] driver uses).
///
/// `Ok(None)` ⇒ the checkpoint is dense (mlx-audio's no-op early-return
/// at [utils.py:222-225][audio-utils-quant] — neither
/// `"quantization"` nor `"quantization_config"` is present, or both are
/// explicitly `null`).
/// `Ok(Some(plq))` ⇒ the checkpoint is quantized; `plq` carries the
/// global default (`group_size` / `bits` / `mode`) plus any per-layer
/// overrides. A malformed quantization block (e.g. `bits` missing or
/// `quantization` not a JSON object) is a recoverable [`Error::OutOfRange`]
/// (non-object block) or [`Error::Parse`] (invalid JSON / missing bits).
///
/// [audio-utils-quant]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L207-L254
pub fn apply_quantization(config_json: &str) -> Result<Option<PerLayerQuantization>> {
  use serde_json::Value;

  let value: Value = serde_json::from_str(config_json).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "audio apply_quantization: config",
      "JSON",
      e,
    ))
  })?;

  // (1) mlx-audio utils.py:221-223 — prefer top-level "quantization" if
  // NON-NULL, else fall back to "quantization_config" (the HF
  // post-quantize artifact key) if NON-NULL, else dense model (no-op).
  // Python's `config.get("quantization", None)` treats a missing key and
  // an explicit `null` value identically, so the fallback must trigger
  // in both cases — and `{"quantization_config": null}` is itself the
  // dense-model no-op early return at utils.py:222-225.
  let block = match value.get("quantization") {
    Some(b) if !b.is_null() => b,
    _ => match value.get("quantization_config") {
      Some(b) if !b.is_null() => b,
      // Neither key present (or both null) → dense model, the no-op
      // early return at mlx-audio utils.py:222-225.
      _ => return Ok(None),
    },
  };

  let Value::Object(map) = block else {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "audio apply_quantization: quantization block",
      "must be a JSON object",
      format_smolstr!("{block:?}"),
    )));
  };

  // (2) mlx-audio utils.py:226 — `group_size = quantization.get("group_size", 64)`.
  // The shared `PerLayerQuantization` deserializer requires `group_size`
  // at the top of the block (swift-faithful), so audio injects the
  // default here before delegating, preserving every other key
  // (including per-layer overrides) verbatim.
  let mut patched = map.clone();
  patched
    .entry("group_size".to_string())
    .or_insert_with(|| Value::from(64));

  let plq: PerLayerQuantization = serde_json::from_value(Value::Object(patched)).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "audio apply_quantization: quantization block",
      "JSON",
      e,
    ))
  })?;
  Ok(Some(plq))
}

/// Faithful port of `mlx_audio.utils.base_load_model`
/// ([utils.py:319-414][audio-utils-base]) — the shared model-load factory
/// every per-domain loader (TTS / STT / STS / VAD / LID / codec) routes
/// through.
///
/// Pipeline (matching mlx-audio's three-stage shape):
///
/// 1. [`get_model_path`] — resolve `path` to a local on-disk directory
///    (mlx-audio's `model_path = get_model_path(...)`).
/// 2. [`load_config`] — read `<dir>/config.json` once into a verbatim
///    JSON body (mlx-audio's `config = load_config(model_path)`).
/// 3. [`apply_quantization`] — parse the optional `quantization` block
///    into a per-layer-aware [`PerLayerQuantization`] (mlx-audio's
///    `apply_quantization(model, config, weights, …)` — the parsing
///    half; the per-architecture weight application is the caller's
///    responsibility).
///
/// **Out of scope** vs mlx-audio's `base_load_model`:
/// - The per-architecture `Model(model_config)` construction +
///   `model.sanitize(weights)` + `model.load_weights(...)` +
///   `model_class.post_load_hook(...)` — those are per-model and the
///   per-domain loader's caller's job (mlxrs has no `model_class`
///   registry — per the no-per-model-arch rule).
/// - The `get_model_class(...)` `importlib.import_module(...)` step —
///   mlxrs's per-domain modules export the load entry points directly.
/// - The `lazy` / `strict` flags — those parametrize the per-architecture
///   `load_weights(strict=…)` call that mlxrs leaves to the caller.
///
/// The returned [`LoadedAudioModel`] bundle is what the per-domain loader
/// consumes: `path` (to read weights from), `config_json` (to parse
/// model-specific args from), and `quantization` (to drive the optional
/// `quantize_weights` call).
///
/// Failures are typed: missing dir → [`Error::MissingKey`], hub path →
/// [`Error::OutOfRange`], malformed JSON → [`Error::Parse`].
/// No implicit eval — this helper reads JSON only, no Array allocation.
///
/// [audio-utils-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L319-L414
pub fn base_load_model(path: &str) -> Result<LoadedAudioModel> {
  let model_path = get_model_path(path)?;
  let config_json = load_config(&model_path)?;
  let quantization = apply_quantization(&config_json)?;
  Ok(LoadedAudioModel::new(model_path, config_json, quantization))
}

#[cfg(test)]
mod tests;
