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
