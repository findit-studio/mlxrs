//! E3 — embeddings local load-factory (`embeddings::factory`).
//!
//! Integration tests reachable from *outside* the crate, mirroring
//! `tests/lm_load.rs` and the `lm/factory.rs` `#[cfg(test)]` block: they prove
//! the public re-exported surface (`load` + `EmbeddingModelTypeRegistry` +
//! `EmbeddingModelConfiguration` + the loaded context) is usable from a
//! consumer, against mock model-dir fixtures (a real `tokenizer.json` written
//! via the `tokenizers` crate + a real `model.safetensors` written via
//! `mlxrs::io::save_safetensors`). Gated on the `embeddings` feature.
#![cfg(feature = "embeddings")]

use std::{
  collections::HashMap,
  fs,
  path::{Path, PathBuf},
  process,
};

use mlxrs::{
  Array, Error,
  embeddings::{
    EmbeddingModel, EmbeddingModelConfiguration, EmbeddingModelConstructor, EmbeddingModelOutput,
    EmbeddingModelTypeRegistry, EmbeddingWeights, LoadedEmbeddingModel, PoolingStrategy, load,
    remap_model_type,
  },
  io,
};

/// A unique temp directory for one test (process-scoped + named so parallel
/// test binaries / cases never collide). Created fresh.
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_emb_load_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// A minimal `config.json` carrying the dispatch `model_type`.
fn config_json(model_type: &str) -> String {
  format!(r#"{{"model_type": "{model_type}", "hidden_size": 4, "vocab_size": 5}}"#)
}

/// A trivial public-surface [`EmbeddingModel`] the mock constructor returns.
struct MockEmbedding;

impl EmbeddingModel for MockEmbedding {
  fn forward(
    &self,
    input_ids: &Array,
    _attention_mask: &Array,
  ) -> Result<EmbeddingModelOutput, Error> {
    let (batch, seq) = match input_ids.shape().as_slice() {
      [b, s] => (*b, *s),
      other => {
        return Err(Error::ShapeMismatch {
          message: format!("expects (batch, seq), got {other:?}"),
        });
      }
    };
    let hidden = 4usize;
    let data = vec![0.0_f32; batch * seq * hidden];
    Ok(EmbeddingModelOutput::from_hidden_state(
      Array::from_slice::<f32>(&data, &(batch, seq, hidden)).unwrap(),
    ))
  }
}

fn mock_constructor() -> EmbeddingModelConstructor {
  Box::new(
    |loaded: &LoadedEmbeddingModel| -> Result<Box<dyn EmbeddingModel>, Error> {
      assert!(!loaded.weights.is_empty());
      Ok(Box::new(MockEmbedding))
    },
  )
}

/// Write a minimal but loadable `tokenizer.json` (3-token WordLevel +
/// Whitespace pre-tokenizer) into `dir`.
fn write_tokenizer(dir: &Path) {
  use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
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

/// Populate `dir` as a loadable model directory: config + weights + tokenizer.
fn write_model_dir(dir: &Path, model_type: &str) {
  fs::write(dir.join("config.json"), config_json(model_type)).unwrap();
  let mut weights: EmbeddingWeights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
  );
  io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(dir);
}

#[test]
fn load_produces_context_via_public_surface() {
  let dir = temp_dir("ctx");
  write_model_dir(&dir, "bert");
  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let ctx = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  )
  .expect("load should succeed");

  assert_eq!(ctx.model_type, "bert");
  assert!(ctx.pooling.is_none(), "no 1_Pooling/config.json written");

  let ids = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
  let mask = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(1usize, 3)).unwrap();
  let out = ctx.model.forward(&ids, &mask).unwrap();
  assert_eq!(out.last_hidden_state.shape(), vec![1, 3, 4]);

  let tok_ids = ctx.tokenizer.encode("a b c", false).unwrap();
  assert_eq!(tok_ids.len(), 3);
}

#[test]
fn load_parses_pooling_config_when_present() {
  let dir = temp_dir("pooling");
  write_model_dir(&dir, "bert");
  let pooling_dir = dir.join("1_Pooling");
  fs::create_dir_all(&pooling_dir).unwrap();
  fs::write(
    pooling_dir.join("config.json"),
    r#"{"word_embedding_dimension": 4, "pooling_mode_cls_token": true}"#,
  )
  .unwrap();

  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let ctx = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  )
  .unwrap();
  let pooling = ctx.pooling.expect("pooling config parsed");
  assert_eq!(pooling.strategy, PoolingStrategy::Cls);
  assert_eq!(pooling.dimension, Some(4));
}

#[test]
fn unknown_model_type_errors() {
  let dir = temp_dir("unknown");
  write_model_dir(&dir, "no_such_arch");
  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let Err(err) = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  ) else {
    panic!("unknown model_type must error");
  };
  let msg = err.to_string();
  assert!(msg.contains("unsupported model type"), "got: {msg}");
  assert!(msg.contains("no_such_arch"), "should name the type: {msg}");
}

#[test]
fn missing_config_errors() {
  let dir = temp_dir("noconfig");
  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let Err(err) = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  ) else {
    panic!("missing config must error");
  };
  assert!(err.to_string().contains("config.json"), "got: {err}");
}

#[test]
fn empty_model_directory_errors_before_filesystem_root_scan() {
  // `from_directory("")`, `from_id("")`, and `PathBuf::new()` each yield an
  // EMPTY model-directory path. Shard discovery builds its glob pattern as
  // `"<dir>/<suffix>"`, so an empty `<dir>` becomes the ABSOLUTE pattern
  // `"/**/model*.safetensors"` — which recursively scans the filesystem ROOT
  // `/`, suppresses permission errors, and can merge unrelated `safetensors`
  // from outside the intended directory (a filesystem-escape + wrong-weight
  // load). `load()` must reject the empty path with a recoverable
  // `Error::Backend` BEFORE any `config.json` / pooling / shard resolution —
  // i.e. it must error WITHOUT performing that filesystem-root scan.
  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  for config in [
    EmbeddingModelConfiguration::from_directory(""),
    EmbeddingModelConfiguration::from_id(""),
    EmbeddingModelConfiguration::from_directory(PathBuf::new()),
  ] {
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
    // Proof the rejection happened up front: the error is NOT a downstream
    // `config.json` / weights failure (which would mean a `/`-root scan ran).
    assert!(
      !msg.contains("config.json") && !msg.contains("no model weights"),
      "the empty path must be rejected before config/shard resolution; got: {msg}"
    );
  }
}

#[test]
fn empty_tokenizer_source_errors() {
  // A separately-supplied tokenizer directory that is EMPTY is the same caller
  // bug and is rejected up front too. The model directory here is real and
  // loadable, so only the empty tokenizer-source path can fail the load.
  let model_dir = temp_dir("empty-tok-src");
  write_model_dir(&model_dir, "bert");
  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let config = EmbeddingModelConfiguration::from_directory(&model_dir).with_tokenizer_source("");

  let Err(err) = load(&config, &registry) else {
    panic!("an empty tokenizer_source path must be a recoverable error");
  };
  assert!(
    matches!(err, Error::Backend { .. }),
    "expected a recoverable Backend error; got {err:?}"
  );
  assert!(
    err
      .to_string()
      .contains("tokenizer directory path must not be empty"),
    "the error should explain the empty tokenizer-path rejection; got: {err}"
  );
}

#[test]
fn separator_normalization_via_public_remap() {
  // `_get_model_arch`'s `-`→`_` normalization is reachable + applied on load.
  assert_eq!(remap_model_type("xlm-roberta"), "xlm_roberta");
  let dir = temp_dir("sep");
  write_model_dir(&dir, "xlm-roberta");
  let registry = EmbeddingModelTypeRegistry::new().with("xlm_roberta", mock_constructor());
  let ctx = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  )
  .unwrap();
  assert_eq!(ctx.model_type, "xlm_roberta");
}

// ───────────────── non-UTF-8 shard-leaf rejection (Unix) ─────────────────
//
// `glob 0.3.3` matches a non-recursive pattern component via
// `file_name().and_then(|s| s.to_str())` and silently `continue`s on `None`
// (`glob-0.3.3/src/lib.rs:463-467`, `// FIXME (#9639)`): a directory entry whose
// own LEAF file name is not valid UTF-8 (e.g. Unix bytes `model\xff.safetensors`)
// is never returned by `glob`, even though it matches the `model*.safetensors`
// shard pattern. Without a backstop the primary shard is silently dropped and
// the loader falls through to a stale `weight*.safetensors` root fallback. The
// `collect_glob_shards` byte-level preflight closes that hole: a non-UTF-8 leaf
// matching a shard pattern now produces a clean `Error::Backend` naming the
// path. These tests drive the PUBLIC `load()` so the fix is verified end-to-end.
//
// macOS/APFS enforces UTF-8 file names and rejects creating the non-UTF-8 entry;
// each test then `return`s cleanly — the error code path (not this fixture) is
// the deliverable, and on a mounted NFS/exFAT/case-sensitive volume the entry
// creates and the rejection is exercised for real. Same skip pattern as the
// in-crate `load_weights_non_utf8_*` tests.

/// Write a minimal single-tensor `model.safetensors`-shaped weights file at an
/// arbitrary `path` (which may be non-UTF-8). Returns `false` if the filesystem
/// rejected the (non-UTF-8) name, so the caller can skip cleanly.
#[cfg(unix)]
fn try_write_one_tensor(path: &Path, key: &str) -> bool {
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert(
    key.to_owned(),
    Array::from_slice::<f32>(&[1.0, 2.0], &(2usize,)).unwrap(),
  );
  io::save_safetensors(path, &weights).is_ok()
}

#[cfg(unix)]
#[test]
fn load_non_utf8_leaf_model_shard_is_recoverable_error() {
  // A UTF-8 model directory containing a `model<0xFF>.safetensors` shard whose
  // LEAF name is not valid UTF-8. `glob` silently skips it; the preflight must
  // reject it with an `Error::Backend` naming the path — no panic.
  use std::{ffi::OsString, os::unix::ffi::OsStringExt};

  let dir = temp_dir("nonutf8-leaf");
  fs::write(dir.join("config.json"), config_json("bert")).unwrap();
  // `model` + invalid byte 0xFF + `.safetensors` — matches the byte-level shard
  // predicate (`b"model"` ... `b".safetensors"`) but is not valid UTF-8.
  let mut raw = b"model".to_vec();
  raw.push(0xFF);
  raw.extend_from_slice(b".safetensors");
  let bad_leaf = OsString::from_vec(raw);
  if !try_write_one_tensor(&dir.join(&bad_leaf), "mock.weight") {
    return; // UTF-8-enforcing filesystem (e.g. APFS) — skip cleanly.
  }

  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let Err(err) = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  ) else {
    panic!("a non-UTF-8 model*.safetensors leaf must be a recoverable error, not a panic");
  };
  assert!(
    matches!(err, Error::Backend { .. }),
    "expected a recoverable Backend error; got {err:?}"
  );
  let msg = err.to_string();
  assert!(
    msg.contains("non-UTF-8 file name") && msg.contains("shard pattern"),
    "the error should explain the non-UTF-8 shard-name rejection; got: {msg}"
  );
  assert!(
    msg.contains(&dir.display().to_string()),
    "the error should name the offending shard path; got: {msg}"
  );
}

#[cfg(unix)]
#[test]
fn load_non_utf8_leaf_shard_wins_over_stale_weight_fallback() {
  // THE DANGEROUS CASE: a model dir with BOTH a non-UTF-8-named
  // `model<0xFF>.safetensors` primary shard AND a valid legacy
  // `weights.safetensors` fallback. `glob` silently drops the non-UTF-8 primary
  // shard, so without the preflight the loader would fall through to the legacy
  // `weight*.safetensors` and load STALE/WRONG weights with no error. The
  // preflight must make the load FAIL with the clean non-UTF-8 error instead —
  // it must NOT silently load the `weights.safetensors` fallback.
  use std::{ffi::OsString, os::unix::ffi::OsStringExt};

  let dir = temp_dir("nonutf8-stale");
  fs::write(dir.join("config.json"), config_json("bert")).unwrap();
  // The non-UTF-8 primary shard.
  let mut raw = b"model".to_vec();
  raw.push(0xFF);
  raw.extend_from_slice(b".safetensors");
  let bad_leaf = OsString::from_vec(raw);
  if !try_write_one_tensor(&dir.join(&bad_leaf), "primary.weight") {
    return; // UTF-8-enforcing filesystem — skip cleanly.
  }
  // A VALID legacy root-level `weight*.safetensors` fallback sitting alongside.
  assert!(
    try_write_one_tensor(&dir.join("weights.safetensors"), "stale.weight"),
    "the legacy fallback shard must write on any filesystem"
  );

  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let Err(err) = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  ) else {
    panic!(
      "a non-UTF-8 model*.safetensors primary shard must fail the load, NOT silently fall \
       back to the stale weight*.safetensors snapshot"
    );
  };
  assert!(
    matches!(err, Error::Backend { .. }),
    "expected a recoverable Backend error; got {err:?}"
  );
  let msg = err.to_string();
  // The error must be the non-UTF-8 shard rejection, proving the preflight
  // fired BEFORE the legacy fallback could load the stale snapshot.
  assert!(
    msg.contains("non-UTF-8 file name") && msg.contains("shard pattern"),
    "the load must fail with the non-UTF-8 shard error, not silently load the stale \
     weight*.safetensors fallback; got: {msg}"
  );
}

#[cfg(unix)]
#[test]
fn load_non_utf8_leaf_shard_nested_under_subfolder_is_recoverable_error() {
  // A non-UTF-8 leaf shard `text_model/model<0xFF>.safetensors` NESTED one level
  // down. The `**/model*.safetensors` glob recurses, and `glob` silently skips
  // the non-UTF-8 leaf at any depth — so the preflight's recursive scan must
  // also catch it nested, not only at the root, and error cleanly.
  use std::{ffi::OsString, os::unix::ffi::OsStringExt};

  let dir = temp_dir("nonutf8-nested-leaf");
  fs::write(dir.join("config.json"), config_json("bert")).unwrap();
  let nested = dir.join("text_model");
  fs::create_dir_all(&nested).unwrap();
  let mut raw = b"model".to_vec();
  raw.push(0xFF);
  raw.extend_from_slice(b".safetensors");
  let bad_leaf = OsString::from_vec(raw);
  if !try_write_one_tensor(&nested.join(&bad_leaf), "encoder.weight") {
    return; // UTF-8-enforcing filesystem — skip cleanly.
  }

  let registry = EmbeddingModelTypeRegistry::new().with("bert", mock_constructor());
  let Err(err) = load(
    &EmbeddingModelConfiguration::from_directory(&dir),
    &registry,
  ) else {
    panic!("a non-UTF-8 leaf shard nested under a subfolder must be a recoverable error");
  };
  assert!(
    matches!(err, Error::Backend { .. }),
    "expected a recoverable Backend error; got {err:?}"
  );
  let msg = err.to_string();
  assert!(
    msg.contains("non-UTF-8 file name") && msg.contains("shard pattern"),
    "the error should explain the non-UTF-8 shard-name rejection; got: {msg}"
  );
  assert!(
    msg.contains(&nested.display().to_string()),
    "the error should name the offending nested shard path; got: {msg}"
  );
}
