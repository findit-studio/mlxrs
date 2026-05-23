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
//!   paths with a clear [`Error::Backend`] rather than calling
//!   `huggingface_hub.snapshot_download` — the (`hf://repo/id`)
//!   path is purely a hosted-Hub artifact, never a real on-disk path.
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
//! [`Error::Backend`]: crate::Error::Backend

use std::path::{Path, PathBuf};

use crate::{
  error::{Error, Result},
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
  pub model_path: PathBuf,
  /// The verbatim `config.json` body — kept as the source-of-truth string
  /// so an architecture-specific deserializer (`Codable`-style) can
  /// reparse model-specific keys outside the small typed subset
  /// [`load_config`] returns. Same TOCTOU-closed single-read convention
  /// as [`crate::lm::load::load_config`].
  pub config_json: String,
  /// Parsed per-layer-aware quantization, if `config.json` carried a
  /// `quantization` block. `None` ⇒ the checkpoint is dense — mlx-audio's
  /// no-op path ([utils.py:222-225][audio-utils-quant]).
  ///
  /// [audio-utils-quant]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L222-L225
  pub quantization: Option<PerLayerQuantization>,
}

/// Ensure `path` is a local on-disk model directory — mirroring
/// `mlx_audio.utils.get_model_path` ([utils.py:106-150][audio-utils-getpath]),
/// minus the `huggingface_hub.snapshot_download` Hub-download branch.
///
/// **No-network**: per the project policy, mlxrs never downloads from
/// HuggingFace Hub. A path that mlx-audio would forward to
/// `snapshot_download` (a non-local repo id like `"mlx-community/foo"`
/// or an `hf://…` URL) is **rejected** with a clear [`Error::Backend`]
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
///    [`Error::Backend`] (mlx-audio's `FileNotFoundError`).
/// 3. Otherwise the input would have been forwarded to
///    `snapshot_download` — mlxrs surfaces
///    [`Error::Backend`] explaining the no-network policy and pointing
///    at the out-of-process workaround.
///
/// [audio-utils-getpath]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L106-L150
/// [audio-utils-islocal]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L96-L103
/// [`Error::Backend`]: crate::Error::Backend
pub fn get_model_path(path: &str) -> Result<PathBuf> {
  let expanded = expand_home(path);

  // (1) Existing local path → done. This matches mlx-audio's
  // `if model_path.exists(): return model_path`.
  if expanded.exists() {
    return Ok(expanded);
  }

  // (2) Looks like a local path but doesn't exist → mlx-audio's
  // FileNotFoundError. Surface a clear Error::Backend instead of
  // (3)'s "would have fetched from Hub" message.
  if is_local_path(path) {
    return Err(Error::Backend {
      message: format!("audio model local path not found: {path}"),
    });
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
  Err(Error::Backend {
    message: format!(
      "audio model path {path:?} is not a local on-disk directory; \
       mlxrs does not download from HuggingFace Hub. Fetch the model \
       directory out of process (e.g. `huggingface-cli download {repo_id}` \
       or `hf download {repo_id}`) and pass the resulting local path."
    ),
  })
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
/// A missing / non-regular / oversized / unreadable / non-UTF-8
/// `config.json` is a recoverable [`Error::Backend`] naming the offending
/// path — matching mlx-audio's `FileNotFoundError(f"Config not found at
/// {model_path}")`.
///
/// [audio-utils-config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L153-L174
/// [`Error::Backend`]: crate::Error::Backend
pub fn load_config(dir: &Path) -> Result<String> {
  let path = dir.join("config.json");
  match read_bounded_config_file(&path, "audio model config")? {
    Some(text) => Ok(text),
    None => Err(Error::Backend {
      message: format!("audio model config not found at {}", path.display()),
    }),
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
/// `quantization` not a JSON object) is a recoverable
/// [`Error::Backend`].
///
/// [audio-utils-quant]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L207-L254
/// [`Error::Backend`]: crate::Error::Backend
pub fn apply_quantization(config_json: &str) -> Result<Option<PerLayerQuantization>> {
  use serde_json::Value;

  let value: Value = serde_json::from_str(config_json).map_err(|e| Error::Backend {
    message: format!("audio apply_quantization: invalid config JSON: {e}"),
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
    return Err(Error::Backend {
      message: format!(
        "audio apply_quantization: quantization block must be a JSON object, got {block:?}"
      ),
    });
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

  let plq: PerLayerQuantization =
    serde_json::from_value(Value::Object(patched)).map_err(|e| Error::Backend {
      message: format!("audio apply_quantization: invalid quantization block: {e}"),
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
/// Every recoverable failure (missing dir / config / quant-block-parse)
/// is an [`Error::Backend`] whose message names the offending path.
/// No implicit eval — this helper reads JSON only, no Array allocation.
///
/// [audio-utils-base]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/utils.py#L319-L414
/// [`Error::Backend`]: crate::Error::Backend
pub fn base_load_model(path: &str) -> Result<LoadedAudioModel> {
  let model_path = get_model_path(path)?;
  let config_json = load_config(&model_path)?;
  let quantization = apply_quantization(&config_json)?;
  Ok(LoadedAudioModel {
    model_path,
    config_json,
    quantization,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;

  /// A unique temp directory for one test (process-scoped + named so
  /// parallel test binaries / cases never collide).
  fn temp_dir(name: &str) -> PathBuf {
    let dir =
      std::env::temp_dir().join(format!("mlxrs_audio_load_{}_{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  /// `get_model_path("/abs/path/that/exists")` returns the canonical
  /// local PathBuf — mirror of mlx-audio's
  /// `if model_path.exists(): return model_path` early return.
  #[test]
  fn get_model_path_resolves_local_path() {
    let dir = temp_dir("resolves_local");
    let s = dir.to_string_lossy().into_owned();
    let resolved = get_model_path(&s).expect("local existing path resolves");
    assert_eq!(resolved, dir);
  }

  /// A repo-id-shaped input ("org/name") that does NOT exist locally is
  /// REJECTED with a clear no-Hub message — the no-network policy.
  #[test]
  fn get_model_path_rejects_hf_hub_path() {
    let err = get_model_path("mlx-community/silero-vad")
      .expect_err("non-local repo id must be rejected, not silently fetched");
    let msg = err.to_string();
    assert!(
      msg.contains("not a local on-disk directory"),
      "error should explain the no-network policy, got: {msg}"
    );
    assert!(
      msg.contains("huggingface-cli download"),
      "error should point at the out-of-process workaround, got: {msg}"
    );
  }

  /// A local-shaped path that does NOT exist surfaces a clear "not
  /// found" error rather than being treated as a Hub id.
  #[test]
  fn get_model_path_local_missing_is_clear_error() {
    let err = get_model_path("/definitely/does/not/exist/mlxrs-a9-missing")
      .expect_err("missing local path must error, not fetch");
    let msg = err.to_string();
    assert!(
      msg.contains("local path not found"),
      "error should name the local-not-found case, got: {msg}"
    );
  }

  /// `load_config` reads a small synthetic `config.json` and returns the
  /// verbatim body.
  #[test]
  fn load_config_reads_small_json() {
    let dir = temp_dir("load_config_small");
    let body = r#"{ "model_type": "silero_vad", "hidden_size": 128 }"#;
    fs::write(dir.join("config.json"), body).unwrap();
    let text = load_config(&dir).expect("config.json reads");
    assert_eq!(text, body);
  }

  /// A missing `config.json` is a recoverable Backend error naming the
  /// offending path.
  #[test]
  fn load_config_missing_is_clear_error() {
    let dir = temp_dir("load_config_missing");
    let err = load_config(&dir).expect_err("missing config.json must error");
    let msg = err.to_string();
    assert!(
      msg.contains("audio model config not found"),
      "error should name the missing-config case, got: {msg}"
    );
  }

  /// A `config.json` without a `quantization` block is the dense-model
  /// path: `apply_quantization` returns `Ok(None)`, matching mlx-audio's
  /// `if quantization is None: return` early return.
  #[test]
  fn apply_quantization_passes_through_unquantized_model() {
    let body = r#"{ "model_type": "silero_vad", "hidden_size": 128 }"#;
    let q = apply_quantization(body).expect("dense config parses");
    assert!(
      q.is_none(),
      "no quantization block → Ok(None), got Some(_) — broke the dense-model path"
    );
  }

  /// A `config.json` with a global `quantization` block parses into a
  /// [`PerLayerQuantization`] carrying the default.
  #[test]
  fn apply_quantization_parses_global_block() {
    let body = r#"{
      "model_type": "silero_vad",
      "quantization": { "group_size": 64, "bits": 4 }
    }"#;
    let q = apply_quantization(body).expect("quantized config parses");
    let plq = q.expect("Some(PerLayerQuantization) for quantized config");
    let global = plq.quantization.expect("global default present");
    assert_eq!(global.group_size, 64);
    assert_eq!(global.bits, 4);
  }

  /// `base_load_model` chains the three steps on a synthetic local dir
  /// (an empty `config.json` + a path that exists). The returned bundle
  /// carries the path, the verbatim JSON body, and the parsed
  /// (here-`None`) quantization.
  #[test]
  fn base_load_model_local_path_resolves() {
    let dir = temp_dir("base_load_local");
    let body = r#"{ "model_type": "silero_vad" }"#;
    fs::write(dir.join("config.json"), body).unwrap();
    let bundle = base_load_model(&dir.to_string_lossy()).expect("local dir loads");
    assert_eq!(bundle.model_path, dir);
    assert_eq!(bundle.config_json, body);
    assert!(bundle.quantization.is_none());
  }

  /// HF post-quantize artifact: `"quantization_config"` (the longer key)
  /// is the fallback mlx-audio's `utils.py:221-223` recognizes when the
  /// shorter `"quantization"` is absent. Both should parse identically.
  #[test]
  fn apply_quantization_parses_quantization_config_key() {
    let body = r#"{
      "model_type": "voxtral",
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
    let q = apply_quantization(body).expect("HF-key config parses");
    let plq = q.expect("Some(PerLayerQuantization) for HF-key config");
    let global = plq.quantization.expect("global default present");
    assert_eq!(global.group_size, 64);
    assert_eq!(global.bits, 4);
  }

  /// mlx-audio's `quantization.get("group_size", 64)` ([utils.py:226])
  /// silently defaults a missing `group_size` to 64. The LM-side parser
  /// would reject this; the audio parser injects the default.
  #[test]
  fn apply_quantization_defaults_missing_group_size_to_64() {
    let body = r#"{
      "model_type": "voxtral",
      "quantization": { "bits": 4 }
    }"#;
    let q = apply_quantization(body).expect("missing-group_size config parses");
    let plq = q.expect("Some(PerLayerQuantization) for default-injected config");
    let global = plq.quantization.expect("global default present");
    assert_eq!(global.group_size, 64, "audio default group_size is 64");
    assert_eq!(global.bits, 4);
  }

  /// When BOTH `"quantization"` and `"quantization_config"` are present,
  /// the top-level key (`"quantization"`) wins — matching mlx-audio's
  /// `config.get("quantization", None)` precedence at utils.py:221.
  #[test]
  fn apply_quantization_top_level_takes_precedence_over_quantization_config() {
    let body = r#"{
      "model_type": "voxtral",
      "quantization": { "bits": 8, "group_size": 32 },
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
    let q = apply_quantization(body).expect("both-keys config parses");
    let plq = q.expect("Some(PerLayerQuantization) for both-keys config");
    let global = plq.quantization.expect("global default present");
    assert_eq!(global.bits, 8, "top-level `quantization` wins");
    assert_eq!(global.group_size, 32, "top-level `quantization` wins");
  }

  /// Python's `config.get("quantization", None)` ([utils.py:221]) treats
  /// a missing key and an explicit `null` value identically — both fall
  /// through to the `quantization_config` retry. A `{"quantization":
  /// null, "quantization_config": {...}}` config must therefore select
  /// the non-null `quantization_config` block, not error on the null.
  #[test]
  fn apply_quantization_null_primary_falls_back_to_quantization_config() {
    let body = r#"{
      "model_type": "voxtral",
      "quantization": null,
      "quantization_config": { "bits": 4, "group_size": 64 }
    }"#;
    let q = apply_quantization(body).expect("null-primary config falls back");
    let plq = q.expect("Some(PerLayerQuantization) from quantization_config fallback");
    let global = plq.quantization.expect("global default present");
    assert_eq!(global.bits, 4, "fallback block's `bits` selected");
    assert_eq!(
      global.group_size, 64,
      "fallback block's `group_size` selected"
    );
  }

  /// `{"quantization_config": null}` is the dense-model no-op (Python's
  /// `config.get("quantization_config", None)` is `None`, the early
  /// return at [utils.py:222-225] fires), not an error on the null
  /// value.
  #[test]
  fn apply_quantization_only_null_quantization_config_returns_none() {
    let body = r#"{ "model_type": "voxtral", "quantization_config": null }"#;
    let q = apply_quantization(body).expect("null-only quantization_config parses as dense");
    assert!(
      q.is_none(),
      "null quantization_config → Ok(None), matches upstream's no-op early return"
    );
  }

  /// `{"quantization": null}` — same rationale as the
  /// `quantization_config: null` case: Python's `dict.get(_, None)` on a
  /// null value yields `None`, falling through to the no-op early
  /// return, not erroring on the null.
  #[test]
  fn apply_quantization_only_null_quantization_returns_none() {
    let body = r#"{ "model_type": "voxtral", "quantization": null }"#;
    let q = apply_quantization(body).expect("null-only quantization parses as dense");
    assert!(
      q.is_none(),
      "null quantization → Ok(None), matches upstream's no-op early return"
    );
  }

  /// Both keys explicitly `null` is the conjunction of the two
  /// preceding cases — the null-aware fallback must still reach the
  /// dense no-op early return, not error on either null block.
  #[test]
  fn apply_quantization_both_null_returns_none() {
    let body = r#"{
      "model_type": "voxtral",
      "quantization": null,
      "quantization_config": null
    }"#;
    let q = apply_quantization(body).expect("both-null config parses as dense");
    assert!(
      q.is_none(),
      "both keys null → Ok(None), matches upstream's no-op early return"
    );
  }

  /// `hf://org/model` repo-id-shaped input: the rejection message's
  /// **CLI workaround segment** must strip the `hf://` prefix so
  /// `huggingface-cli download <repo_id>` is actionable.
  ///
  /// The message echoes the user's raw input for context (e.g. `path
  /// "hf://org/model" is not a local on-disk directory`), but the CLI
  /// suggestion segment after "Fetch the model directory out of process"
  /// must contain only the clean repo id.
  #[test]
  fn get_model_path_hf_url_prefix_yields_clean_repo_id_in_error() {
    let err = get_model_path("hf://mlx-community/silero-vad")
      .expect_err("hf:// repo id must be rejected with a clean workaround");
    let msg = err.to_string();
    assert!(
      msg.contains("mlx-community/silero-vad"),
      "workaround should print the clean repo id, got: {msg}"
    );
    // Extract the CLI suggestion segment (between "Fetch" and the
    // closing ")") and assert the prefix isn't there — the leading
    // `audio model path "hf://…"` is the verbatim echo, deliberate.
    let workaround = msg
      .split_once("Fetch the model directory out of process")
      .map(|(_, after)| after)
      .expect("workaround section present");
    assert!(
      !workaround.contains("hf://"),
      "CLI workaround must not embed the `hf://` prefix, got: {workaround}"
    );
    assert!(
      workaround.contains("huggingface-cli download mlx-community/silero-vad"),
      "CLI workaround must use the clean repo id, got: {workaround}"
    );
  }

  /// `https://huggingface.co/org/model` URL: same — strip the URL
  /// prefix so the CLI workaround is correct.
  #[test]
  fn get_model_path_https_huggingface_url_yields_clean_repo_id_in_error() {
    let err = get_model_path("https://huggingface.co/mlx-community/silero-vad")
      .expect_err("https://huggingface.co/ URL must be rejected with a clean workaround");
    let msg = err.to_string();
    assert!(
      msg.contains("mlx-community/silero-vad"),
      "workaround should print the clean repo id, got: {msg}"
    );
    let workaround = msg
      .split_once("Fetch the model directory out of process")
      .map(|(_, after)| after)
      .expect("workaround section present");
    assert!(
      !workaround.contains("https://huggingface.co/"),
      "CLI workaround must not embed the full URL, got: {workaround}"
    );
    assert!(
      workaround.contains("huggingface-cli download mlx-community/silero-vad"),
      "CLI workaround must use the clean repo id, got: {workaround}"
    );
  }

  /// Every per-domain `audio::<domain>::load` module exposes its
  /// `MODEL_REMAPPING` table under the same uniform name (no
  /// per-domain prefix) so generic caller code can read
  /// `audio::<domain>::load::MODEL_REMAPPING` without a per-domain
  /// branch. Codec's table is empty (mlx-audio's `codec/__init__.py`
  /// ships no remapping); the others mirror their upstream
  /// `*-utils.py:MODEL_REMAPPING` tables.
  #[test]
  #[allow(non_snake_case)]
  fn per_domain_load_modules_expose_uniform_MODEL_REMAPPING() {
    let tts: &[(&str, &str)] = crate::audio::tts::load::MODEL_REMAPPING;
    let stt: &[(&str, &str)] = crate::audio::stt::load::MODEL_REMAPPING;
    let sts: &[(&str, &str)] = crate::audio::sts::load::MODEL_REMAPPING;
    let vad: &[(&str, &str)] = crate::audio::vad::load::MODEL_REMAPPING;
    let lid: &[(&str, &str)] = crate::audio::lid::load::MODEL_REMAPPING;
    let codec: &[(&str, &str)] = crate::audio::codec::load::MODEL_REMAPPING;

    assert!(
      codec.is_empty(),
      "codec's MODEL_REMAPPING must be empty per upstream's no-remapping shape, got: {codec:?}"
    );
    assert!(
      !tts.is_empty(),
      "TTS MODEL_REMAPPING must mirror upstream's non-empty alias table"
    );
    assert!(
      !stt.is_empty(),
      "STT MODEL_REMAPPING must mirror upstream's non-empty alias table"
    );
    assert!(
      !sts.is_empty(),
      "STS MODEL_REMAPPING must mirror upstream's non-empty alias table"
    );
    assert!(
      !vad.is_empty(),
      "VAD MODEL_REMAPPING must mirror upstream's non-empty alias table"
    );
    assert!(
      !lid.is_empty(),
      "LID MODEL_REMAPPING must mirror upstream's non-empty alias table"
    );
  }
}
