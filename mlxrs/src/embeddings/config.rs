//! `sentence-transformers` pooling-config parsing (`1_Pooling/config.json`).
//!
//! Ported from `mlx-embeddings` `utils._read_pooling_config` +
//! `models/pooling._normalize_pooling_config` (legacy `pooling_mode_*`
//! → `pooling_mode`) and swift `MLXEmbedders`
//! `Pooling.PoolingConfiguration` / `loadPooling` (the CLS > Mean > Max
//! > Last priority + `word_embedding_dimension` matryoshka dim).
//!
//! Reading the file off disk and the model-id registry are out of scope
//! (no-model-arch rule); this only parses already-obtained JSON
//! (path *or* in-memory bytes/str) into a [`PoolingStrategy`] +
//! `normalize` + `dimension` triple. Feature-gated: pulled in only by
//! `embeddings` (the sole user of the optional `serde_json` dep).

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

use super::pooling::PoolingStrategy;

/// Upper bound on the on-disk size of a `1_Pooling/config.json` we will
/// read into memory. Real `sentence-transformers` pooling configs are a
/// handful of boolean flags plus a dimension — well under 1 KiB. The cap
/// is deliberately generous (1 MiB) yet still hard-bounds the allocation
/// (enforced via `Read::take(cap + 1)` on the opened handle, so even a
/// hostile / corrupt model directory that races a TOCTOU swap or streams
/// from a special file cannot drive an unbounded read into an OOM).
/// Exceeding it yields a recoverable [`Error::Backend`], not a panic,
/// and the over-cap body is never parsed.
///
/// Reading is additionally **non-blocking against a non-regular file**:
/// on Unix the open uses `O_NONBLOCK | O_CLOEXEC` so a FIFO planted at
/// the config path by an untrusted model dir returns from `open()`
/// immediately (no indefinite wait for a writer). Symlinks **are**
/// followed (HuggingFace Hub caches store `snapshots/<rev>/1_Pooling/
/// config.json` as a symlink into `blobs/<hash>` — the dominant real
/// cached-model layout — so refusing symlinks would break normal cached
/// models). Safety does not rely on refusing symlinks: the opened
/// handle is `fstat`ed and rejected via `metadata().is_file()` **before
/// any read** (this stats the *resolved target* of any symlink, so a
/// symlink → FIFO/device/directory is still rejected), and a
/// non-blocking open prevents a symlink → FIFO from hanging. The only
/// guarantees a caller relies on (no hang, no unbounded read, no panic,
/// recoverable error) hold for FIFOs/devices/directories and for
/// symlinks to any of them too.
const MAX_ST_POOLING_CONFIG_BYTES: u64 = 1 << 20;

/// Legacy `sentence-transformers` boolean flag → mode name. Mirrors
/// python `_LEGACY_POOLING_MODE_KWARGS`.
const LEGACY_KEYS: &[(&str, &str)] = &[
  ("pooling_mode_cls_token", "cls"),
  ("pooling_mode_max_tokens", "max"),
  ("pooling_mode_mean_tokens", "mean"),
  ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"),
  ("pooling_mode_weightedmean_tokens", "weightedmean"),
  ("pooling_mode_lasttoken", "lasttoken"),
];

/// Parsed `1_Pooling/config.json` → pooling pipeline parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StPoolingConfig {
  /// Resolved pooling strategy.
  pub strategy: PoolingStrategy,
  /// Whether the embeddings should be L2-normalized after pooling. ST
  /// configs don't carry a normalize flag, so this is always `true`
  /// (the `mlx-embeddings` / `MLXEmbedders` convention is to normalize),
  /// surfaced explicitly so the caller can override.
  pub normalize: bool,
  /// Matryoshka output dimension (`word_embedding_dimension`), if the
  /// config declares one.
  pub dimension: Option<usize>,
}

fn resolve_strategy(cfg: &serde_json::Map<String, Value>) -> Result<PoolingStrategy> {
  // Modern key wins if present (python `pool_by_config` uses
  // `cfg["pooling_mode"]` directly when set).
  if let Some(Value::String(mode)) = cfg.get("pooling_mode") {
    return PoolingStrategy::from_mode(mode);
  }
  if let Some(v) = cfg.get("pooling_mode")
    && v.is_array()
  {
    return Err(Error::Backend {
      message: "concatenated pooling mode (list) is not supported; \
                only a single pooling mode is allowed"
        .into(),
    });
  }
  // C6 (Copilot review 4307622782, #3256688299): a present-but-non-
  // string/non-array `pooling_mode` (null / bool / number / object).
  //
  // python parity: `_normalize_pooling_config` only synthesizes
  // `pooling_mode` from legacy flags; with `pooling_mode` already present
  // it leaves the value as-is, then `pool_by_config` does
  // `mode = cfg["pooling_mode"]` and — since `None`/`False`/a number is
  // neither a tuple/list, nor a known-unsupported string, nor any of the
  // `if mode == "cls"/...` branches — falls through to
  // `raise ValueError(f"Unknown pooling mode {mode!r}...")`
  // (`models/pooling.py`; `tests/test_pooling.py::test_invalid_mode_-
  // raises` pins the analogous unknown-string path). python therefore
  // REJECTS a present-but-wrong-typed `pooling_mode`; it does NOT silently
  // fall back to legacy/Mean. mlxrs previously fell through to the legacy
  // path (silent Mean) — a divergence AND a silent-wrong-embedding (the
  // model author set `pooling_mode`; honoring it as a different strategy
  // is silently wrong). Reject with a recoverable `Err` to match python.
  if let Some(v) = cfg.get("pooling_mode") {
    return Err(Error::Backend {
      message: format!(
        "`pooling_mode` is present but not a string or list (got {}); \
         a malformed pooling mode is rejected (python `pool_by_config` \
         raises `ValueError` for a non-string/non-list mode) rather than \
         silently falling back to a different strategy",
        match v {
          Value::Null => "null".to_string(),
          Value::Bool(b) => format!("bool {b}"),
          Value::Number(n) => format!("number {n}"),
          Value::Object(_) => "object".to_string(),
          // String/Array handled above; unreachable, but no panic.
          _ => "an unsupported JSON type".to_string(),
        }
      ),
    });
  }

  // Legacy boolean flags. python `_normalize_pooling_config` picks the
  // *first active flag in legacy declaration order* and errors out of
  // `pool_by_config` if it is a known-unsupported mode; swift
  // `Pooling(config:)` instead applies a fixed CLS > Mean > Max > Last
  // priority. The task specifies the python priority **CLS > Mean > Max
  // > Last** (swift's order) — applied here over the *supported* flags.
  let truthy = |k: &str| cfg.get(k).and_then(Value::as_bool).unwrap_or(false);

  // Reject known-unsupported flags only if they are the *sole* active
  // ones (mirrors python: an unsupported mode that is the resolved one
  // raises; a supported one alongside it just wins via priority).
  if truthy("pooling_mode_cls_token") {
    return Ok(PoolingStrategy::Cls);
  }
  if truthy("pooling_mode_mean_tokens") {
    return Ok(PoolingStrategy::Mean);
  }
  if truthy("pooling_mode_max_tokens") {
    return Ok(PoolingStrategy::Max);
  }
  if truthy("pooling_mode_lasttoken") {
    return Ok(PoolingStrategy::Last);
  }

  // No supported flag set: surface a known-unsupported one if that is
  // what was declared (python `pool_by_config` NotImplementedError);
  // otherwise fall back to python `_normalize_pooling_config`'s `("mean",)`
  // default / swift's `.first`. python's no-active default is `"mean"`;
  // swift's is `.first`. We follow python (primary reference) → mean,
  // unless an unsupported flag is the only thing present.
  for (key, name) in LEGACY_KEYS {
    if (*name == "weightedmean" || *name == "mean_sqrt_len_tokens") && truthy(key) {
      return Err(Error::Backend {
        message: format!(
          "pooling mode {name:?} is not supported (supported: cls, lasttoken, max, mean)"
        ),
      });
    }
  }

  // Any legacy flag key present at all (even all-false) ⇒ python's
  // `("mean",)` default.
  let has_legacy = LEGACY_KEYS.iter().any(|(k, _)| cfg.contains_key(*k));
  if has_legacy {
    return Ok(PoolingStrategy::Mean);
  }

  Err(Error::Backend {
    message: "pooling config declares no pooling mode (no `pooling_mode` \
              and no legacy `pooling_mode_*` flags)"
      .into(),
  })
}

fn parse_value(v: &Value) -> Result<StPoolingConfig> {
  let cfg = v.as_object().ok_or_else(|| Error::Backend {
    message: "pooling config is not a JSON object".into(),
  })?;

  // python `pool_by_config` rejects `include_prompt: false` (INSTRUCTOR
  // prompt-aware pooling unsupported).
  if let Some(Value::Bool(false)) = cfg.get("include_prompt") {
    return Err(Error::Backend {
      message: "prompt-aware pooling (include_prompt=false) is not supported".into(),
    });
  }

  let strategy = resolve_strategy(cfg)?;

  // Matryoshka dim: swift `word_embedding_dimension`; python configs
  // also commonly use `embedding_dimension` (legacy ST). Accept either,
  // `word_embedding_dimension` taking precedence when both are present.
  //
  // C7 (Copilot review 4307622782, #3256688310): a present-but-invalid
  // value (negative / fractional / string / `> usize`) previously went
  // `as_u64()` → `None` → treated as ABSENT → matryoshka truncation
  // silently SKIPPED, so the caller got a full-width embedding when the
  // model author explicitly requested a truncated dimension — a silent
  // wrong embedding.
  //
  // python parity: python `mlx-embeddings` has NO matryoshka /
  // `word_embedding_dimension` truncation at all (grep-confirmed: the dim
  // is carried in the ST config but never used to slice the output; the
  // truncation is an mlxrs/swift-only feature), so there is no python
  // reference for malformed-dimension handling here. The user's standing
  // rule is "never silently produce wrong embeddings": a present key the
  // model author set MUST be honored or surfaced. A present-but-invalid
  // dimension is therefore a recoverable `Err` (an intentionally
  // stricter-than-python safety choice — python has no behavior to match,
  // and a silent full-width fallback is a silent-wrong-result).
  //
  // Only the FIRST present key is consulted (matching the
  // `word_embedding_dimension` > `embedding_dimension` precedence): if
  // `word_embedding_dimension` is present but invalid we reject it rather
  // than silently falling back to `embedding_dimension`.
  let dim_entry = cfg
    .get("word_embedding_dimension")
    .map(|v| ("word_embedding_dimension", v))
    .or_else(|| {
      cfg
        .get("embedding_dimension")
        .map(|v| ("embedding_dimension", v))
    });
  let dimension = match dim_entry {
    None => None,
    Some((key, v)) => {
      let d = v.as_u64().ok_or_else(|| Error::Backend {
        message: format!(
          "`{key}` is present but not a non-negative integer (got {v}); \
           a malformed matryoshka dimension is rejected rather than \
           silently skipping truncation (which would return a \
           full-width embedding the model author did not request)"
        ),
      })?;
      let d = usize::try_from(d).map_err(|_| Error::Backend {
        message: format!(
          "`{key}` = {d} exceeds usize::MAX; refusing to use it as a \
           matryoshka dimension"
        ),
      })?;
      if d == 0 {
        return Err(Error::Backend {
          message: format!(
            "`{key}` is 0; a zero matryoshka dimension would produce an \
             empty embedding (rejected rather than silently skipped)"
          ),
        });
      }
      Some(d)
    }
  };

  Ok(StPoolingConfig {
    strategy,
    normalize: true,
    dimension,
  })
}

/// Parse a `1_Pooling/config.json` from an in-memory JSON string.
///
/// Mirrors python `_read_pooling_config` + `_normalize_pooling_config`
/// (legacy `pooling_mode_*` keys, the modern `pooling_mode` key,
/// `include_prompt` guard) and swift `PoolingConfiguration` decoding —
/// resolved with the CLS > Mean > Max > Last priority over supported
/// flags.
pub fn pooling_from_st_config_str(json: &str) -> Result<StPoolingConfig> {
  let v: Value = serde_json::from_str(json).map_err(|e| Error::Backend {
    message: format!("invalid pooling config JSON: {e}"),
  })?;
  parse_value(&v)
}

/// Parse a `1_Pooling/config.json` from raw in-memory JSON bytes.
pub fn pooling_from_st_config_bytes(json: &[u8]) -> Result<StPoolingConfig> {
  let v: Value = serde_json::from_slice(json).map_err(|e| Error::Backend {
    message: format!("invalid pooling config JSON: {e}"),
  })?;
  parse_value(&v)
}

/// Read and parse `<model_dir>/1_Pooling/config.json`.
///
/// `model_dir` is the model root; the `1_Pooling/config.json` suffix is
/// appended (python `_read_pooling_config`, swift `loadPooling`'s
/// `appending(components: "1_Pooling", "config.json")`). Returns an error
/// if the file is absent or unreadable (python returns `None`; the
/// caller can map the error to a fallback).
///
/// The read is bounded against an untrusted model directory:
///
/// 1. the file is **opened once** — no separate `stat` that a TOCTOU
///    swap/extend could race past. On Unix the open carries
///    `O_NONBLOCK | O_CLOEXEC`: opening a **FIFO** returns immediately
///    instead of blocking until a writer appears (a hostile model dir
///    cannot hang the caller by planting a named pipe at `config.json`).
///    Symlinks **are** followed: HuggingFace Hub caches store
///    `snapshots/<rev>/1_Pooling/config.json` as a symlink into
///    `blobs/<hash>`, so `O_NOFOLLOW` would make `open()` fail (ELOOP)
///    for a normal cached model and the caller would silently fall back
///    to the wrong pooling strategy/dimension. Following the symlink is
///    safe because steps 2–3 enforce the actual guarantees on the
///    *resolved target* (a symlink → FIFO/device/dir is still rejected
///    by step 2's `is_file()` fstat of the opened target, and a
///    symlink → FIFO still cannot hang thanks to `O_NONBLOCK`). On
///    non-Unix targets a plain `File::open` is used (no
///    FIFO-open-blocking semantics to defend against).
/// 2. the *opened handle's* metadata must describe a **regular file**
///    (`metadata.is_file()`) and this is checked **before any read** — a
///    FIFO / device / directory / symlink-to-special (all of which
///    `fs::metadata().len()` would report as `0`, bypassing a pre-read
///    size check) is rejected here with a recoverable [`Error::Backend`].
///    `File::metadata()` `fstat`s the opened descriptor, i.e. the
///    *resolved target* of any symlink, so this check is what defends
///    against symlink → non-regular, not refusing symlinks at open.
///    Because the rejection precedes any `read`, the `O_NONBLOCK` handle
///    is never read from, so a non-blocking `EAGAIN` can never occur.
/// 3. the body is read through `Read::take(MAX + 1)` so at most one byte
///    past the 1 MiB cap is ever allocated; if that cap is exceeded the
///    config is rejected (recoverable [`Error::Backend`]), never parsed.
///
/// No panic and **no hang** — every failure path (absent, non-regular
/// incl. FIFO/device/symlink-to-special, oversized, unreadable, invalid
/// JSON) is a recoverable error the caller can map to a fallback (python
/// returns `None`). A symlink to an in-cap regular JSON file (the HF
/// cache layout) is followed and parsed normally.
pub fn pooling_from_st_config_path(model_dir: impl AsRef<Path>) -> Result<StPoolingConfig> {
  use std::io::Read;

  let path = model_dir.as_ref().join("1_Pooling").join("config.json");

  // Open ONCE: a single handle whose metadata and contents refer to the
  // same file object, closing the stat-then-read TOCTOU window.
  //
  // On Unix a read-only blocking `open()` of a FIFO blocks until a
  // writer appears, so an untrusted model dir that plants a named pipe
  // at `config.json` would hang the caller indefinitely *before* the
  // `is_file()` rejection below could ever run. Open with
  // `O_NONBLOCK | O_CLOEXEC`: a FIFO/non-regular open returns
  // immediately (no writer-wait). Symlinks are intentionally followed
  // (no `O_NOFOLLOW`): HuggingFace Hub caches store
  // `snapshots/<rev>/1_Pooling/config.json` as a symlink into
  // `blobs/<hash>`, so `O_NOFOLLOW` would fail (ELOOP) on a normal
  // cached model and the caller would silently use the wrong pooling.
  // This loses no safety: the `is_file()` check below fstats the
  // *opened (resolved) target*, so a symlink→FIFO/device/dir is still
  // rejected before any read, and the symlink→FIFO open still cannot
  // hang because of `O_NONBLOCK`. On non-Unix targets there is no
  // FIFO-open-blocking semantics to defend against — a plain
  // `File::open` is used.
  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(&path)
      .map_err(|e| Error::Backend {
        message: format!("cannot open pooling config {}: {e}", path.display()),
      })?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(&path).map_err(|e| Error::Backend {
    message: format!("cannot open pooling config {}: {e}", path.display()),
  })?;

  // Reject non-regular files from the OPENED handle, BEFORE any read.
  // `File::metadata()` fstats the open descriptor, i.e. the *resolved
  // target* of any symlink we followed at open — so a FIFO / device /
  // directory / symlink-to-any-of-those (all of which `len() == 0` to a
  // pre-read `fs::metadata` check yet still stream/block unbounded data
  // on read) is rejected here. The model dir is untrusted, so only a
  // regular file (or a symlink resolving to one, e.g. the HF blob
  // layout) is accepted. Doing this before any `read` also means the
  // `O_NONBLOCK` handle (which on an opened FIFO could yield `EAGAIN`)
  // is never read from: keep this ordering.
  let meta = file.metadata().map_err(|e| Error::Backend {
    message: format!("cannot stat opened pooling config {}: {e}", path.display()),
  })?;
  if !meta.is_file() {
    return Err(Error::Backend {
      message: format!(
        "pooling config {} is not a regular file; refusing to read",
        path.display()
      ),
    });
  }

  // Read at most `cap + 1` bytes: `take` hard-bounds the allocation
  // regardless of the reported size (a regular file can still be
  // extended between open and read; `take` makes that harmless). If we
  // got more than the cap the config is oversized → reject, never parse.
  let mut bytes = Vec::new();
  file
    .take(MAX_ST_POOLING_CONFIG_BYTES + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| Error::Backend {
      message: format!("cannot read pooling config {}: {e}", path.display()),
    })?;
  if bytes.len() as u64 > MAX_ST_POOLING_CONFIG_BYTES {
    return Err(Error::Backend {
      message: format!(
        "pooling config {} exceeds the {}-byte cap; refusing to read",
        path.display(),
        MAX_ST_POOLING_CONFIG_BYTES
      ),
    });
  }
  pooling_from_st_config_bytes(&bytes)
}
