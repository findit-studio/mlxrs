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
      _ => {
        let shape = input_ids.shape();
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockLoadedEmbedding::forward expects rank-2 (batch, seq) ids",
          shape.len() as u32,
          shape,
        )));
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
      matches!(err, Error::EmptyInput(_)),
      "expected EmptyInput error; got {err:?}"
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
    matches!(err, Error::EmptyInput(_)),
    "expected EmptyInput error; got {err:?}"
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
      matches!(err, Error::EmptyInput(_)),
      "expected EmptyInput error; got {err:?}"
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

// ───────── glob-faithful recursive-walk tests ─────────
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
    matches!(err, Error::FileIo(_)),
    "expected FileIo error; got {err:?}"
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
    matches!(err, Error::FileIo(_)),
    "expected FileIo error; got {err:?}"
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
    matches!(err, Error::FileIo(_)),
    "expected FileIo error; got {err:?}"
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
    matches!(err, Error::FileIo(_)),
    "expected FileIo error; got {err:?}"
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
  // Strict typed FileIo destructure. The non-UTF-8 parent-directory
  // rejection MUST surface as Error::FileIo with FileOp::Other
  // ("weight_shard_discovery"), inner io::Error kind == InvalidData
  // (UTF-8 conversion failure), and path == the offending shard.
  let Error::FileIo(payload) = &err else {
    panic!("expected Error::FileIo; got {err:?}");
  };
  assert_eq!(
    payload.op(),
    FileOp::Other("weight_shard_discovery"),
    "non-UTF-8 parent rejection MUST surface as FileOp::Other(\"weight_shard_discovery\"); \
       got {:?}",
    payload.op()
  );
  assert_eq!(
    payload.inner().kind(),
    std::io::ErrorKind::InvalidData,
    "non-UTF-8 parent rejection MUST carry io::ErrorKind::InvalidData; got {:?}",
    payload.inner().kind()
  );
  let msg = err.to_string();
  assert!(
    msg.contains("non-UTF-8 parent directory name"),
    "the error should explain the non-UTF-8 parent rejection; got: {msg}"
  );
  // The error path must be the offending nested shard (an ASCII-named
  // model.safetensors under the bad-byte directory) — assert it sits
  // inside the model dir AND ends with the shard file name.
  assert!(
    payload.path().starts_with(&dir),
    "the error path must sit under the model directory; got {}",
    payload.path().display()
  );
  assert!(
    payload.path().ends_with("model.safetensors"),
    "the error path must end with the shard file name; got {}",
    payload.path().display()
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
