//! End-to-end VLM load-factory tests, driven by mock model + mock
//! processor types registered into fresh registries (per the
//! no-model-arch rule, this PR ships the seam, not architectures or
//! processors — so the end-to-end path is proven against hand-traced
//! mocks over a temp model directory).

use std::{
  path::PathBuf,
  sync::atomic::{AtomicU64, Ordering},
};

use super::*;
use crate::{
  array::Array,
  error::{FileOp, MissingFieldPayload, RankMismatchPayload},
  lm::{cache::KvCache, generate::GenConfig, model::Model as LmModel},
  vlm::{
    generate::{VlmGenConfig, vlm_generate},
    image::{ColorOrder, ImageProcessorConfig, ResizeFilter},
    prompt::MarkerPolicy,
  },
};

/// A "flat" mock `config.json` for the mock VLM architecture: the
/// dispatch-only key (`model_type`) plus `vocab_size` and `mock_extra`
/// at the **top level**. The minimal [`VlmBaseConfig`] only needs
/// `model_type`; the mock constructor reads `vocab_size` and
/// `mock_extra` off the verbatim [`LoadedVlmModel::config_json`] so
/// the registry-dispatch end-to-end path is proven against the same
/// raw-JSON model-specific decode every real per-model VLM constructor
/// performs (the nested-config layout — `text_config.vocab_size` —
/// is exercised separately by [`mock_nested_config_json`] and
/// [`load_succeeds_for_nested_vlm_config_with_no_top_level_lm_fields`]).
fn mock_config_json(model_type: &str) -> String {
  format!(
    r#"{{
        "model_type": "{model_type}",
        "vocab_size": 5,
        "mock_extra": 11
      }}"#
  )
}

/// A `config.json` shaped like a **real** VLM checkpoint:
/// `model_type` at the top level (the dispatch key, mirroring swift's
/// `BaseConfiguration` at `MLXLMCommon/BaseConfiguration.swift:13-16`),
/// every text-model field nested under `text_config` (mirroring how
/// e.g. Qwen2-VL / LLaVA / Pixtral ship their configs, and how
/// `mlx_vlm.utils.load_model:239-240` sets up
/// `config.setdefault("text_config", ...)`), and an arbitrary
/// `vision_config` block. NO top-level `hidden_size` / `num_hidden_layers`
/// / `vocab_size` / etc. — the regression case the
/// [`crate::lm::load::Config`] parse would *fatally reject* (since those
/// fields are required there).
fn mock_nested_config_json(model_type: &str) -> String {
  format!(
    r#"{{
        "model_type": "{model_type}",
        "text_config": {{
          "hidden_size": 8,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 2,
          "rope_theta": 10000.0,
          "vocab_size": 5,
          "tie_word_embeddings": false
        }},
        "vision_config": {{
          "hidden_size": 16,
          "num_hidden_layers": 1,
          "image_size": 224
        }},
        "mock_extra": 11
      }}"#
  )
}

/// A minimal processor-config body (written to whichever of
/// `preprocessor_config.json` / `processor_config.json` a test wants).
/// `processor_class` is the registry key; `mock_image_size` is a
/// model-specific key OUTSIDE the typed subset, used to prove the
/// processor constructor reads the carried config body
/// ([`LoadedProcessor::preprocessor_config_json`] /
/// [`LoadedProcessor::processor_config_json`]).
fn mock_preprocessor_config_json(processor_class: &str, image_size: u32) -> String {
  format!(
    r#"{{
        "processor_class": "{processor_class}",
        "mock_image_size": {image_size}
      }}"#
  )
}

/// A trivial VLM [`Model`] returned by the mock constructor. Implements
/// the LM-side [`crate::lm::model::Model`] (vocab-aware zero logits) and
/// the VLM-side [`crate::vlm::model::Model`]'s required entry points
/// (text-embed lookup, image-encode passthrough); records both the
/// raw-JSON-decoded `vocab_size` (which the per-model constructor reads
/// off `LoadedVlmModel::config_json`, since `VlmBaseConfig` carries
/// only the dispatch fields) and the raw-config `mock_extra` for
/// assertions.
struct MockVlmModel {
  vocab: i32,
  #[allow(dead_code)]
  mock_extra: i64,
}

impl LmModel for MockVlmModel {
  fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let (batch, seq) = match tokens.shape().as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockVlmModel::forward expects [B, S] (rank 1 or 2)",
          tokens.shape().len() as u32,
          tokens.shape(),
        )));
      }
    };
    let vocab = self.vocab as usize;
    Array::from_slice::<f32>(&vec![0.0_f32; batch * seq * vocab], &(batch, seq, vocab))
  }
}

impl crate::vlm::model::Model for MockVlmModel {
  fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
    // [1, T] tokens → [1, T, hidden=8] zero embeds. Matches the typed
    // Config.hidden_size = 8 from `mock_config_json` so a chained
    // forward_embeddings would line up.
    let shape = tokens.shape();
    let (b, t) = match shape.as_slice() {
      [b, t] => (*b, *t),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockVlmModel::embed_tokens expects [B, T] (rank 2)",
          shape.len() as u32,
          shape,
        )));
      }
    };
    Array::from_slice::<f32>(&vec![0.0_f32; b * t * 8], &(b, t, 8usize))
  }

  fn encode_image(&self, _image: &Array) -> Result<Array> {
    // [1, 8] zero features — single placeholder per image into the
    // hidden_size = 8 space.
    Array::from_slice::<f32>(&[0.0_f32; 8], &(1usize, 8usize))
  }
}

/// A trivial [`Processor`] returned by the mock processor constructor.
/// Records the typed `processor_class` it was dispatched on AND the
/// model-specific `mock_image_size` it read off the raw processor
/// JSON, so a test can assert both pieces of dispatch state arrived.
struct MockVlmProcessor {
  #[allow(dead_code)]
  processor_class: String,
  image_size: u32,
}

impl Processor for MockVlmProcessor {
  fn image_processor_config(&self) -> ImageProcessorConfig {
    // Honor the image-size the processor decoded off the raw JSON, so
    // a test can assert the cross-model preprocessing parameters
    // round-trip through the registry.
    ImageProcessorConfig::new()
      .with_size((self.image_size, self.image_size))
      .with_mean([0.5, 0.5, 0.5])
      .with_std([0.5, 0.5, 0.5])
      .with_rescale_factor(1.0 / 255.0)
      .with_do_resize(true)
      .with_do_rescale(true)
      .with_do_normalize(true)
      .with_resample(ResizeFilter::Bilinear)
      .with_color_order(ColorOrder::Rgb)
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

/// Build a [`VlmModelConstructor`] for the mock VLM architecture: read
/// the dispatch key off the typed [`LoadedVlmModel::config`]
/// (`model_type` is the only field guaranteed by [`VlmBaseConfig`]),
/// then decode model-specific fields (`vocab_size`, `mock_extra`) off
/// the verbatim [`LoadedVlmModel::config_json`] — mirroring how a real
/// per-model VLM constructor's `Codable` init reads its nested
/// `text_config` / `vision_config` blocks off the raw JSON
/// (`VLMModelFactory.swift:343-348`). `vocab_size` is looked up at the
/// top level OR under `text_config` so the same mock works for both the
/// "flat" and "nested" fixtures. Asserts at least one weight tensor
/// arrived.
fn mock_vlm_constructor() -> VlmModelConstructor {
  Box::new(|loaded: &LoadedVlmModel| -> Result<Box<dyn VlmModel>> {
    assert!(
      !loaded.weights_ref().is_empty(),
      "constructor should receive the loaded weights"
    );
    let raw: serde_json::Value = serde_json::from_str(loaded.config_json_ref())
      .map_err(|e| Error::Parse(ParsePayload::new("mock vlm ctor: config.json", "JSON", e)))?;
    // Vocab can be top-level (the "flat" mock fixture) or nested under
    // text_config (the real-VLM-shaped mock fixture). The per-model
    // constructor decides how to decode its own model-specific fields;
    // both are equally legitimate dispatch outputs here, since
    // `VlmBaseConfig` only requires `model_type` and the rest flows
    // through the raw JSON.
    let vocab = raw
      .get("vocab_size")
      .or_else(|| raw.get("text_config").and_then(|t| t.get("vocab_size")))
      .and_then(serde_json::Value::as_i64)
      .and_then(|x| i32::try_from(x).ok())
      .ok_or_else(|| {
        Error::MissingField(MissingFieldPayload::new(
          "mock vlm ctor",
          "vocab_size (top-level or text_config.vocab_size)",
        ))
      })?;
    let mock_extra = raw
      .get("mock_extra")
      .and_then(serde_json::Value::as_i64)
      .ok_or(Error::MissingField(MissingFieldPayload::new(
        "mock vlm ctor",
        "mock_extra",
      )))?;
    Ok(Box::new(MockVlmModel { vocab, mock_extra }))
  })
}

/// Build a [`ProcessorConstructor`] for the mock processor: read the
/// processor class off [`LoadedProcessor::processor_class`] and the
/// model-specific `mock_image_size` off whichever processor-config body
/// carries it — [`LoadedProcessor::preprocessor_config_json`] when a
/// `preprocessor_config.json` was present (the common + split layouts),
/// otherwise [`LoadedProcessor::processor_config_json`] (the
/// `processor_config.json`-only layout). Mirrors a real per-model
/// processor decoding its image-preprocessor metadata from the file
/// that actually carries it.
fn mock_processor_constructor() -> ProcessorConstructor {
  Box::new(
    |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
      // The image-preprocessor metadata lives in `preprocessor_config.
      // json` when that file is present, else in the
      // `processor_config.json`-only body. Decode from whichever the
      // loader carried — both bodies that were on disk are available.
      let body = loaded
        .preprocessor_config_json
        .or(loaded.processor_config_json)
        .ok_or_else(|| {
          Error::MissingField(MissingFieldPayload::new(
            "mock vlm processor ctor",
            "preprocessor_config_json or processor_config_json",
          ))
        })?;
      let raw: serde_json::Value = serde_json::from_str(body).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "mock vlm processor ctor: processor config",
          "JSON",
          e,
        ))
      })?;
      let image_size = raw
        .get("mock_image_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or(Error::MissingField(MissingFieldPayload::new(
          "mock vlm processor ctor",
          "mock_image_size",
        )))?;
      // Sanity-touch the tokenizer the swift `(Data, Tokenizer) ->
      // Processor` shape hands in — assert it can encode something so a
      // future change that hands the wrong (uninitialized / wrong-dir)
      // tokenizer surfaces here.
      let _ = loaded
        .tokenizer
        .encode("a", false)
        .expect("processor constructor must receive a working tokenizer");
      Ok(Box::new(MockVlmProcessor {
        processor_class: loaded.processor_class.to_owned(),
        image_size,
      }))
    },
  )
}

/// A fresh, writable per-test temp directory (the crate's
/// no-`tempfile`-crate convention: `temp_dir()` + pid + a
/// process-unique counter so parallel tests never collide). Created
/// empty; the caller populates it.
fn fresh_dir(tag: &str) -> PathBuf {
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-vlm-factory-{tag}-{}-{n}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Serialize a minimal but loadable `tokenizer.json` (a 3-token
/// WordLevel model with a Whitespace pre-tokenizer) into `dir` via
/// the `tokenizers` crate — the same fixture style as the LM
/// factory's tests, so the reused [`Tokenizer::from_path`] loads it.
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

/// Populate `dir` with the VLM `config.json` + a tiny single-tensor
/// `model.safetensors` + the named processor config (one of
/// `"preprocessor_config.json"` / `"processor_config.json"`) — but
/// **no** `tokenizer.json`. Basis for [`write_vlm_dir`] (which adds
/// the tokenizer) and the split-layout test.
fn write_vlm_dir_no_tokenizer(
  dir: &Path,
  model_type: &str,
  processor_filename: &str,
  processor_class: &str,
  image_size: u32,
) {
  std::fs::write(dir.join("config.json"), mock_config_json(model_type)).unwrap();
  std::fs::write(
    dir.join(processor_filename),
    mock_preprocessor_config_json(processor_class, image_size),
  )
  .unwrap();

  // A tiny one-tensor safetensors so `load_weights` finds non-empty
  // weights. `save_safetensors` writes the on-disk format the loader
  // reads.
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
}

/// Populate `dir` as a minimal but *loadable* VLM model directory:
/// `config.json`, a tiny single-tensor `model.safetensors`, the named
/// processor config, and a `tokenizer.json`.
fn write_vlm_dir(
  dir: &Path,
  model_type: &str,
  processor_filename: &str,
  processor_class: &str,
  image_size: u32,
) {
  write_vlm_dir_no_tokenizer(
    dir,
    model_type,
    processor_filename,
    processor_class,
    image_size,
  );
  write_tokenizer(dir);
}

#[test]
fn load_dispatches_to_registered_mocks_and_returns_full_bundle() {
  let dir = fresh_dir("dispatch");
  write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 64);
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

  // The returned VLM base config carries the dispatch key + (here, no)
  // top-level eos override. vocab/etc. flow through the raw JSON to the
  // constructor, which proved them in `logits.shape()` below.
  assert_eq!(ctx.config_ref().model_type(), "mockvlm");
  assert_eq!(ctx.config_ref().eos_token_id().cloned(), None);
  assert_eq!(ctx.config_ref().quantization(), None);

  // The constructed model is the mock: drive one forward to confirm it
  // is wired and the constructor saw the right vocab off the raw JSON.
  let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
  let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
  let logits = LmModel::forward(ctx.model(), &tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, 5]);

  // The constructed processor surfaces the image-size it decoded off
  // the raw processor JSON (64 from `write_vlm_dir`) — round-trip
  // proof that the processor constructor saw the right JSON body.
  let proc_cfg = ctx.processor().image_processor_config();
  assert_eq!(proc_cfg.size(), (64, 64));

  // The tokenizer loaded from the same directory.
  let ids = ctx.tokenizer().encode("a b c", false).unwrap();
  assert_eq!(ids.len(), 3);
}

#[test]
fn loaded_model_drives_vlm_generate_end_to_end() {
  // load↔generate integration gap: `load()` hands back a
  // `LoadedVlmContext` whose `model` is a `Box<dyn VlmModel>`, and the
  // public `vlm_generate` is generic over `M: vlm::Model + ?Sized`. This
  // test proves the loader's trait-object output drives the generation
  // loop *directly* — `ctx.model()` deref-coerces `Box<dyn VlmModel>` to
  // `&dyn VlmModel`, an UNSIZED `M`, which satisfies the relaxed bound.
  // Without the `?Sized` relaxation this call would not compile at all
  // (the implicit `Sized` bound would reject `dyn VlmModel`), so the
  // loader's output would be unusable by the generation loop; that
  // regression is caught here. Zero-image path — `vlm_generate` dispatches straight to
  // `lm::generate::generate_step` (also `?Sized`-generic, accepted
  // because `VlmModel: Model`) — so this needs no image fixture.
  let dir = fresh_dir("e2e-generate");
  write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 64);
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

  // Drive `vlm_generate` on the LOADED model — `ctx.model()` is a
  // `&dyn VlmModel` (the `Box<dyn VlmModel>` field deref-coerced). The
  // mock's `forward` returns `[B, S, vocab]` zero logits ⇒ greedy argmax
  // is token id 0 every step; an empty eos set lets it run to
  // `max_tokens`. The mock ignores the KV cache, so an empty cache is
  // sufficient to exercise the decode loop.
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(4),
    99,
    3,
    MarkerPolicy::Required,
  );
  let prompt = [0_u32, 1, 2];
  // mlx-vlm `generate(model, processor, …)` — the image-processor config is
  // supplied separately; the loaded processor carries the parsed config.
  let img_cfg = ctx.processor().image_processor_config();
  let steps = vlm_generate(ctx.model(), &img_cfg, &prompt, &[], Vec::new(), cfg)
    .expect("vlm_generate constructs against the loaded trait-object model");

  let tokens: Vec<u32> = steps
    .map(|s| s.expect("each generation step succeeds").token)
    .collect();
  // The loaded model produced exactly `max_tokens` tokens — load→generate
  // works for the trait-object output. (Greedy argmax of all-zero logits
  // is the lowest index, 0.)
  assert_eq!(tokens, vec![0_u32, 0, 0, 0]);
}

#[test]
fn loaded_processor_config_drives_image_preprocessing_not_model_default() {
  // load↔generate gap: `vlm_generate` must
  // preprocess real image prompts with the *loaded* processor's
  // `ImageProcessorConfig` — the one parsed from
  // `preprocessor_config.json` / `processor_config.json` and carried on
  // `LoadedVlmContext.processor` — NOT one re-derived from the model via
  // `Model::image_processor_config()` (which falls back to the trait
  // default / a stale baked-in config). mlx-vlm's `generate(model,
  // processor, …)` takes the processor separately for exactly this
  // reason; `vlm_generate` now mirrors that with an explicit
  // `image_processor_config` parameter.
  //
  // This test wires the divergence concretely: the loaded processor
  // config's image size (48×48, parsed off the processor JSON by the
  // mock processor) DIFFERS from the model default (224×224). It loads
  // a VLM via `load()`, drives `vlm_generate` on the loaded model with
  // the loaded processor's config + one real image, and asserts the
  // model's `encode_image` saw a `[48, 48, 3]` preprocessed array —
  // proof the LOADED config (not the 224×224 model default) drove
  // preprocessing. Re-deriving the config from the model default would
  // instead preprocess to `[224, 224, 3]`.
  use std::sync::{Arc, Mutex};

  // A VLM model whose `encode_image` records the shape of the
  // (preprocessed) array it receives, so the test can read back what
  // size `preprocess` resized to. `embed_tokens` / `forward` /
  // `forward_embeddings` are minimal but real so the full multimodal
  // prefill+decode path runs. `image_processor_config` is left as the
  // trait default (224×224) — the value that MUST NOT be used.
  struct RecordingVlmModel {
    /// Set once by `encode_image` to the preprocessed input's shape.
    seen_image_shape: Arc<Mutex<Option<Vec<usize>>>>,
  }
  impl LmModel for RecordingVlmModel {
    fn forward(&self, tokens: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let (b, s) = match tokens.shape().as_slice() {
        [b, s] => (*b, *s),
        [s] => (1, *s),
        _ => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "RecordingVlmModel::forward expects [B, S] (rank 1 or 2)",
            tokens.shape().len() as u32,
            tokens.shape(),
          )));
        }
      };
      // `[B, S, vocab=5]` zero logits — greedy argmax is token id 0.
      Array::from_slice::<f32>(&vec![0.0_f32; b * s * 5], &(b, s, 5usize))
    }
    fn forward_embeddings(&self, embeddings: &Array, _c: &mut [Box<dyn KvCache>]) -> Result<Array> {
      // `[1, T, D]` merged embeds → `[1, T, vocab=5]` zero logits.
      let (b, t) = match embeddings.shape().as_slice() {
        [b, t, _d] => (*b, *t),
        _ => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "RecordingVlmModel::forward_embeddings expects [B, T, D] (rank 3)",
            embeddings.shape().len() as u32,
            embeddings.shape(),
          )));
        }
      };
      Array::from_slice::<f32>(&vec![0.0_f32; b * t * 5], &(b, t, 5usize))
    }
  }
  impl crate::vlm::model::Model for RecordingVlmModel {
    fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
      let (b, t) = match tokens.shape().as_slice() {
        [b, t] => (*b, *t),
        _ => {
          return Err(Error::RankMismatch(RankMismatchPayload::new(
            "RecordingVlmModel::embed_tokens expects [B, T] (rank 2)",
            tokens.shape().len() as u32,
            tokens.shape(),
          )));
        }
      };
      // hidden_size = 8, matching `encode_image`'s D below.
      Array::from_slice::<f32>(&vec![0.0_f32; b * t * 8], &(b, t, 8usize))
    }
    fn encode_image(&self, image: &Array) -> Result<Array> {
      // Record the preprocessed image shape — this is the observable
      // proof of which `ImageProcessorConfig` drove `preprocess`.
      *self.seen_image_shape.lock().unwrap() = Some(image.shape());
      // `[num_tokens_per_image = 1, D = 8]` features (one row per image,
      // satisfying `vlm_generate`'s `[num_tokens_per_image, D]` check).
      Array::from_slice::<f32>(&[0.0_f32; 8], &(1usize, 8usize))
    }
  }

  let recorded: Arc<Mutex<Option<Vec<usize>>>> = Arc::new(Mutex::new(None));
  // The model constructor captures a clone of the recording handle so
  // the test can read back `encode_image`'s input AFTER `load()` has
  // boxed the model into the `LoadedVlmContext`.
  let model_registry = {
    let recorded = Arc::clone(&recorded);
    VlmTypeRegistry::new().with(
      "recordingvlm",
      Box::new(
        move |loaded: &LoadedVlmModel| -> Result<Box<dyn VlmModel>> {
          assert!(
            !loaded.weights_ref().is_empty(),
            "constructor should receive the loaded weights"
          );
          Ok(Box::new(RecordingVlmModel {
            seen_image_shape: Arc::clone(&recorded),
          }))
        },
      ),
    )
  };
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());

  // Processor config image size = 48 (≠ the 224×224 model default).
  let dir = fresh_dir("loaded-proc-drives-preprocess");
  write_vlm_dir(
    &dir,
    "recordingvlm",
    "preprocessor_config.json",
    "MockProc",
    48,
  );
  let config = VlmModelConfiguration::from_directory(&dir);
  let ctx = load(&config, &model_registry, &processor_registry).expect("load should succeed");

  // Sanity: the loaded processor's config carries the 48×48 size parsed
  // off the JSON, and it differs from the model's (default) 224×224.
  let loaded_img_cfg = ctx.processor().image_processor_config();
  assert_eq!(loaded_img_cfg.size(), (48, 48));
  assert_eq!(
    crate::vlm::model::Model::image_processor_config(ctx.model()).size(),
    (224, 224),
    "the recording model uses the trait-default 224×224 — the value that must NOT drive preprocessing"
  );

  // A real PNG `vlm::image::load_image` can decode (size irrelevant —
  // `preprocess` resizes to the config's `size`).
  let img_path = dir.join("prompt.png");
  let mut buf = ::image::RgbImage::new(10, 7);
  for y in 0..7 {
    for x in 0..10 {
      buf.put_pixel(x, y, ::image::Rgb([(x * 20) as u8, (y * 30) as u8, 64]));
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
    .save_with_format(&img_path, ::image::ImageFormat::Png)
    .unwrap();

  // Drive `vlm_generate` on the LOADED model with the LOADED processor's
  // config + one image. marker=image_token=99, num_tokens_per_image=1.
  let cfg = VlmGenConfig::new(
    GenConfig::default().with_max_tokens(2),
    99,
    1,
    MarkerPolicy::Required,
  );
  let prompt = [0_u32, 99, 1]; // one marker → one image
  let steps = vlm_generate(
    ctx.model(),
    &loaded_img_cfg,
    &prompt,
    std::slice::from_ref(&img_path),
    Vec::new(),
    cfg,
  )
  .expect("vlm_generate constructs against the loaded model + loaded processor config");
  // Drain so the eager vision pipeline (load → preprocess → encode_image)
  // has definitely run.
  let tokens: Vec<u32> = steps
    .map(|s| s.expect("each generation step succeeds").token)
    .collect();
  assert_eq!(tokens, vec![0_u32, 0]);

  // THE ASSERTION: `encode_image` saw a `[48, 48, 3]` array — the loaded
  // processor config's size drove `preprocess`, NOT the model's 224×224
  // default. (`preprocess` emits channel-last `[H, W, 3]`.)
  let seen = recorded
    .lock()
    .unwrap()
    .clone()
    .expect("encode_image must have run on the single image prompt");
  assert_eq!(
    seen,
    vec![48, 48, 3],
    "image preprocessing must use the loaded processor config's size (48×48), \
       not the model's default 224×224"
  );
}

#[test]
fn preprocessor_config_is_preferred_over_processor_config() {
  // Both files present, with DIFFERENT processor_class values. The
  // `preprocessor_config.json` MUST win (per
  // VLMModelFactory.swift:438-454's preference order); the registry
  // is set up so only the "Preferred" class can construct — the
  // "Fallback" class would resolve to a missing constructor.
  let dir = fresh_dir("prefer-preprocessor");
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("Preferred", 32),
  )
  .unwrap();
  std::fs::write(
    dir.join("processor_config.json"),
    mock_preprocessor_config_json("Fallback", 999),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("Preferred", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry)
    .expect("load should succeed using the preferred preprocessor_config.json");
  // The mock processor records the image_size off the raw JSON — `32`
  // proves the preferred file was used (would be `999` from the
  // fallback otherwise).
  assert_eq!(ctx.processor().image_processor_config().size(), (32, 32));
}

#[test]
fn processor_config_is_used_when_only_fallback_present() {
  // No preprocessor_config.json → fall back to processor_config.json.
  let dir = fresh_dir("fallback-processor-config");
  write_vlm_dir(&dir, "mockvlm", "processor_config.json", "MockProc", 48);
  assert!(!dir.join("preprocessor_config.json").exists());
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx =
    load(&config, &model_registry, &processor_registry).expect("fallback processor_config load");
  assert_eq!(ctx.processor().image_processor_config().size(), (48, 48));
}

#[test]
fn from_id_resolves_as_local_path() {
  // An `Identifier::Id` is treated as a LOCAL path (no network): pointing
  // it at the temp dir loads exactly as `from_directory` would.
  let dir = fresh_dir("idpath");
  write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 24);
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_id(dir.to_str().unwrap());
  assert_eq!(config.model_directory(), dir.as_path());

  let ctx = load(&config, &model_registry, &processor_registry)
    .expect("id-as-local-path load should succeed");
  assert_eq!(ctx.config_ref().model_type(), "mockvlm");
}

#[test]
fn tokenizer_source_loads_from_separate_directory() {
  // Split layout: the model dir has config + processor config +
  // weights but NO tokenizer.json; a separate dir holds the tokenizer.
  // `tokenizer_source` points the load there, mirroring the LM
  // factory's analogous test.
  let model_dir = fresh_dir("split-model");
  write_vlm_dir_no_tokenizer(
    &model_dir,
    "mockvlm",
    "preprocessor_config.json",
    "MockProc",
    16,
  );
  assert!(!model_dir.join("tokenizer.json").exists());
  let tok_dir = fresh_dir("split-tok");
  write_tokenizer(&tok_dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&model_dir).with_tokenizer_source(&tok_dir);
  assert_eq!(config.tokenizer_directory(), tok_dir.as_path());

  let ctx = load(&config, &model_registry, &processor_registry).expect("split-tokenizer load");
  let ids = ctx.tokenizer().encode("a b c", false).unwrap();
  assert_eq!(ids.len(), 3);
}

#[test]
fn unknown_model_type_is_recoverable_error_with_no_io_beyond_config() {
  // config.json says "nope" but only "mockvlm" is registered →
  // unsupported-model-type Error (NOT a panic), naming the type. The
  // weights file is deliberately INVALID, the tokenizer is absent,
  // and the processor config is absent: any load attempt would
  // surface a different error. We must see the unsupported-model
  // error first (faithful to step (2) of the orchestration order).
  let dir = fresh_dir("unknown-model-cheap");
  std::fs::write(dir.join("config.json"), mock_config_json("nope")).unwrap();
  std::fs::write(
    dir.join("model.safetensors"),
    b"this is not a safetensors file",
  )
  .unwrap();
  assert!(!dir.join("tokenizer.json").exists());
  assert!(!dir.join("preprocessor_config.json").exists());

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let Err(err) = load(&config, &model_registry, &processor_registry) else {
    panic!("unknown VLM model_type must error");
  };
  match &err {
    Error::MissingKey(p) => {
      assert_eq!(p.key(), "nope", "error should carry the model_type as key");
      assert!(
        p.context().contains("model_type"),
        "context should name model_type, got: {}",
        p.context()
      );
    }
    _ => panic!("expected MissingKey, got: {err:?}"),
  }
  // The processor-config / weights / tokenizer paths must NOT have
  // run: their files are intentionally absent/invalid here, and a
  // failure on any of them surfaces a different error variant.
  let msg = err.to_string();
  assert!(
    !msg.contains("safetensors") && !msg.contains("tokenizer.json"),
    "weights/tokenizer must not have been loaded, got: {msg}"
  );
}

#[test]
fn unknown_processor_class_is_recoverable_error_with_no_weight_io() {
  // Model type IS registered, but the processor class on disk is
  // not. The unsupported-processor-class error must fire BEFORE any
  // weight load: weights file is deliberately invalid here.
  let dir = fresh_dir("unknown-processor-cheap");
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("WrongProc", 16),
  )
  .unwrap();
  std::fs::write(
    dir.join("model.safetensors"),
    b"this is not a safetensors file",
  )
  .unwrap();
  assert!(!dir.join("tokenizer.json").exists());

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let Err(err) = load(&config, &model_registry, &processor_registry) else {
    panic!("unknown processor class must error");
  };
  match &err {
    Error::MissingKey(p) => {
      assert_eq!(
        p.key(),
        "WrongProc",
        "error should carry the processor_class as key"
      );
      assert!(
        p.context().contains("processor_class"),
        "context should name processor_class, got: {}",
        p.context()
      );
    }
    _ => panic!("expected MissingKey, got: {err:?}"),
  }
  let msg = err.to_string();
  assert!(
    !msg.contains("safetensors") && !msg.contains("tokenizer.json"),
    "weights/tokenizer must not have been loaded, got: {msg}"
  );
}

#[test]
fn missing_processor_config_is_recoverable_error() {
  // No preprocessor_config.json AND no processor_config.json present.
  let dir = fresh_dir("no-proc-config");
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let Err(err) = load(&config, &model_registry, &processor_registry) else {
    panic!("missing processor config must error");
  };
  match &err {
    Error::FileIo(p) => {
      assert_eq!(p.op(), FileOp::Open);
      let s = p.path().to_string_lossy();
      assert!(
        s.contains("processor_config.json"),
        "FileIo path should be the fallback processor_config.json, got: {s}"
      );
    }
    _ => panic!("expected FileIo for missing processor config, got: {err:?}"),
  }
}

#[test]
fn processor_class_override_applies_for_mistral3() {
  // Mistral3 ships processor_class = "PixtralProcessor" on disk but
  // VLMModelFactory.swift:399-403 overrides it to "Mistral3Processor"
  // because spatial-merge handling is different. The registry is set
  // up so only "Mistral3Processor" can construct; "PixtralProcessor"
  // would resolve to a missing constructor.
  let dir = fresh_dir("mistral3-override");
  write_vlm_dir(
    &dir,
    "mistral3",
    "preprocessor_config.json",
    "PixtralProcessor",
    40,
  );
  let model_registry = VlmTypeRegistry::new().with("mistral3", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("Mistral3Processor", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry)
    .expect("mistral3 override should dispatch to Mistral3Processor");
  assert_eq!(ctx.processor().image_processor_config().size(), (40, 40));
}

#[test]
fn vlm_remap_applies_on_registration_and_lookup() {
  // "lfm2-vl" canonicalizes to "lfm2_vl" (verbatim from
  // mlx_vlm.utils.MODEL_REMAPPING line 34). Registering under either
  // form, the registry finds it under both.
  let registry = VlmTypeRegistry::new().with("lfm2-vl", mock_vlm_constructor());
  assert!(registry.contains("lfm2-vl"));
  assert!(registry.contains("lfm2_vl"));
  assert!(!registry.contains("qwen3_vl"));
  assert_eq!(remap_vlm_model_type("lfm2-vl"), "lfm2_vl");
  assert_eq!(remap_vlm_model_type("qwen3_vl"), "qwen3_vl");
}

#[test]
fn register_replaces_and_returns_previous() {
  let mut registry = VlmTypeRegistry::new();
  assert!(
    registry
      .register("mockvlm", mock_vlm_constructor())
      .is_none()
  );
  assert!(
    registry
      .register("mockvlm", mock_vlm_constructor())
      .is_some()
  );
  let mut proc_registry = VlmProcessorTypeRegistry::new();
  assert!(
    proc_registry
      .register("MockProc", mock_processor_constructor())
      .is_none()
  );
  assert!(
    proc_registry
      .register("MockProc", mock_processor_constructor())
      .is_some()
  );
}

#[test]
fn raw_config_and_processor_json_reach_constructors() {
  // The constructors stash what they SAW; assert both pieces of
  // raw-JSON dispatch state arrived correctly (the model's
  // `mock_extra = 11` from `mock_config_json`, the processor's
  // `mock_image_size = 24` from `mock_preprocessor_config_json`).
  let dir = fresh_dir("raw-dispatch");
  write_vlm_dir(&dir, "mockvlm", "preprocessor_config.json", "MockProc", 24);
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry).expect("load");
  assert_eq!(ctx.processor().image_processor_config().size(), (24, 24));
}

#[test]
fn load_succeeds_for_nested_vlm_config_with_no_top_level_lm_fields() {
  // **Regression** test: real VLM `config.json` files
  // commonly nest the text-model fields (`hidden_size` /
  // `num_hidden_layers` / `vocab_size` / etc.) under `text_config` and
  // only carry `model_type` at the top — exactly what
  // `mock_nested_config_json` shapes. Routing the VLM load path through
  // the LM `lm::load::Config` parse upfront would REQUIRE those
  // top-level fields → every real VLM checkpoint would fatally error
  // BEFORE a registered VLM constructor could see the raw JSON. With
  // the VLM-minimal `VlmBaseConfig` parse, the dispatch goes through,
  // the per-model constructor reads its nested `text_config.vocab_size`
  // off the verbatim raw JSON, and the load completes — proven by the
  // shape of the forward pass driving the registered mock constructor.
  let dir = fresh_dir("nested-vlm-config");
  std::fs::write(dir.join("config.json"), mock_nested_config_json("mockvlm")).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 32),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let config = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&config, &model_registry, &processor_registry)
    .expect("nested-config VLM should load (no top-level LM fields)");

  // The dispatch key arrived; vocab/hidden_size live under text_config
  // and are not on `VlmBaseConfig` (faithful to swift's BaseConfiguration).
  assert_eq!(ctx.config_ref().model_type(), "mockvlm");

  // Drive one forward — confirms the mock constructor decoded
  // `text_config.vocab_size = 5` off the raw JSON and the registry +
  // weight + tokenizer path all completed against the nested-shaped
  // config.
  let mut cache: Vec<Box<dyn KvCache>> = Vec::new();
  let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
  let logits = LmModel::forward(ctx.model(), &tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, 5]);
}

#[test]
fn eos_token_id_on_vlm_config_flows_to_tokenizer() {
  // `eos_token_id` declared at the TOP LEVEL of a real-VLM-shaped
  // `config.json` (no top-level LM fields, nested `text_config`) must
  // be picked up by `VlmBaseConfig` and forwarded to the tokenizer via
  // `load_tokenizer_with_eos` — REPLACING the tokenizer-config default
  // (mirroring `TokenizerWrapper`'s `set(eos_token_ids)` semantics).
  let dir = fresh_dir("eos-from-config");
  let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": [1, 2],
      "text_config": {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": 5,
        "tie_word_embeddings": false
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry).expect("eos config load");

  // Base config carries the [1, 2] list verbatim (shape-preserving).
  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Many(vec![1, 2])),
    "base config should carry the top-level eos_token_id list"
  );
  // And the tokenizer's COMPLETE eos set is exactly {1, 2} — the
  // tokenizer-config default was REPLACED (not unioned) by the resolved
  // list, exactly as `TokenizerWrapper::set(eos_token_ids)` does.
  let eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  assert_eq!(
    eos_vec,
    vec![1u32, 2],
    "tokenizer eos set should be exactly the resolved {{1, 2}}"
  );
}

#[test]
fn generation_config_eos_overrides_vlm_base_config_eos() {
  // A *truthy* `generation_config.json` `eos_token_id` OVERWRITES the
  // `config.json` value IN PLACE on the returned `VlmBaseConfig` — same
  // semantics as mlx-lm and mlx-vlm (`mlx_vlm/utils.py:506-515`). The
  // override is a scalar `2`; the on-disk config says `1`; the
  // resulting tokenizer eos set must be {2}.
  let dir = fresh_dir("eos-generation-override");
  let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": 1,
      "text_config": {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": 5,
        "tie_word_embeddings": false
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 2}"#).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("eos generation override load");

  // The returned base config reflects the generation-config override
  // (`1` → `2`) — exactly the in-place overwrite mlx-vlm performs.
  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Single(2)),
    "generation_config.json eos_token_id should override config.json"
  );
  // Tokenizer's COMPLETE eos set is the post-override {2}, not the
  // on-disk {1}.
  let eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  assert_eq!(
    eos_vec,
    vec![2u32],
    "tokenizer eos set should be the overridden {{2}}"
  );
}

#[test]
fn vlm_base_config_parses_without_top_level_lm_fields() {
  // Pure parse-level proof of the contract: a JSON body with ONLY
  // `model_type` (no LM-required fields, no eos, no quantization)
  // parses into a `VlmBaseConfig` — the swift `BaseConfiguration`
  // shape — and the LM `Config` parse would have rejected the same
  // body (required `hidden_size` / etc. absent). Guards against a
  // future regression that re-adds a hard LM field to `VlmBaseConfig`.
  let cfg = r#"{ "model_type": "qwen2_vl" }"#;
  let base = VlmBaseConfig::from_json(cfg).expect("VLM base config should parse");
  assert_eq!(base.model_type(), "qwen2_vl");
  assert_eq!(base.eos_token_id().cloned(), None);
  assert_eq!(base.quantization(), None);

  // Same body through the LM `Config` parse fails (missing
  // `hidden_size` and the rest of the required LM subset). This pins
  // *why* we need a separate VLM base parse.
  let lm_err = crate::lm::load::Config::from_json(cfg)
    .expect_err("LM Config should reject a model_type-only body");
  let msg = lm_err.to_string();
  assert!(
    msg.contains("hidden_size") || msg.contains("missing field"),
    "LM Config parse error should name the missing LM field, got: {msg}"
  );
}

// ────────────────────────────────────────────────────────────────────
// Nested-EOS promotion regression tests.
// ────────────────────────────────────────────────────────────────────

/// Write a `tokenizer_config.json` that pins the tokenizer-config
/// fallback EOS to vocab id 2 (the `"c"` token in [`write_tokenizer`]'s
/// 3-token WordLevel vocab). Used by the nested-EOS tests to prove the
/// promoted nested EOS REPLACES this fallback (rather than the
/// tokenizer silently dropping the nested value and defaulting to id 2).
fn write_tokenizer_config_with_eos_c(dir: &Path) {
  std::fs::write(dir.join("tokenizer_config.json"), r#"{ "eos_token": "c" }"#).unwrap();
}

#[test]
fn nested_text_config_eos_promotes_to_tokenizer() {
  // Real VLM layout: NO top-level `eos_token_id`, but `text_config`
  // carries a list `[42, 50]`. The tokenizer_config pins a different
  // fallback EOS (`"c"` → id 2). After load, the tokenizer's COMPLETE
  // eos set MUST be exactly {42, 50} — the nested promotion happened
  // and REPLACED the tokenizer-config default. Without the promotion,
  // the nested value would be silently dropped → eos set = {2}, wrong
  // generation stop.
  let dir = fresh_dir("nested-text-config-eos");
  let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);
  write_tokenizer_config_with_eos_c(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("nested text_config.eos_token_id should promote");

  // Base config carries the promoted [42, 50] list (shape-preserved).
  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Many(vec![42, 50])),
    "VlmBaseConfig should carry the promoted text_config.eos_token_id list"
  );
  // Tokenizer's COMPLETE eos set is exactly {42, 50} — the
  // tokenizer-config default ({2}) was REPLACED, not unioned.
  let mut eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  eos_vec.sort_unstable();
  assert_eq!(
    eos_vec,
    vec![42u32, 50],
    "tokenizer eos set should be exactly the promoted {{42, 50}}, not the tokenizer-config fallback"
  );
}

#[test]
fn top_level_eos_wins_over_nested_text_config_eos() {
  // Top-level `eos_token_id = 7` is present AND `text_config.eos_token_id
  // = [42, 50]` is present — top-level MUST win (the nested promotion
  // only triggers when the top-level is `None`). Faithful to swift
  // `BaseConfiguration`'s top-level-only decode (`MLXLMCommon/BaseConfiguration.swift:192-208`).
  let dir = fresh_dir("top-eos-wins-over-nested");
  let cfg = r#"{
      "model_type": "mockvlm",
      "eos_token_id": 7,
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("top-level eos with nested present");

  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Single(7)),
    "top-level eos_token_id must win over nested text_config.eos_token_id"
  );
  let eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  assert_eq!(
    eos_vec,
    vec![7u32],
    "tokenizer eos set must be the top-level {{7}}, not nested {{42, 50}}"
  );
}

#[test]
fn generation_config_eos_overrides_promoted_nested_eos() {
  // Promotion happens BEFORE the generation_config override: nested
  // `text_config.eos_token_id = [42, 50]` is promoted, but
  // `generation_config.json eos_token_id = 9` then overrides on top —
  // exactly the same precedence Python's
  // `mlx_vlm/utils.py:506-515` block has for the top-level
  // `eos_token_id`.
  let dir = fresh_dir("gen-cfg-overrides-promoted-nested");
  let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 9}"#).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("generation_config override over promoted nested eos");

  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Single(9)),
    "generation_config.json eos_token_id must override the promoted nested value"
  );
  let eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  assert_eq!(
    eos_vec,
    vec![9u32],
    "tokenizer eos set must be the post-override {{9}}, not the promoted nested set"
  );
}

#[test]
fn nested_llm_config_eos_promotes_when_text_config_absent() {
  // `llm_config` is the alias mlx-vlm rewrites to `text_config` via
  // `config.setdefault("text_config", config.pop("llm_config", {}))`
  // (`mlx_vlm/utils.py:239`). When the checkpoint only has the alias
  // and no canonical `text_config`, the nested-EOS promotion must still
  // pick the alias up so the tokenizer's eos set reflects it.
  let dir = fresh_dir("nested-llm-config-eos");
  // `vocab_size` at top level since the mock constructor only knows
  // top-level + `text_config.vocab_size`; the alias key is incidental
  // to this EOS-promotion test.
  let cfg = r#"{
      "model_type": "mockvlm",
      "vocab_size": 5,
      "llm_config": {
        "hidden_size": 8,
        "eos_token_id": [11, 13]
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("nested llm_config.eos_token_id alias should promote");

  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Many(vec![11, 13])),
    "VlmBaseConfig should carry the promoted llm_config.eos_token_id list"
  );
}

#[test]
fn text_config_eos_wins_over_llm_config_alias_when_both_present() {
  // mlx-vlm's `setdefault(text_config, pop(llm_config))` makes
  // `text_config` the canonical destination (an existing `text_config`
  // is preserved, the `llm_config` alias is only consulted as a
  // fallback). Our promotion mirrors that precedence: when BOTH nested
  // blocks are present and carry different EOS values, `text_config`
  // wins.
  let dir = fresh_dir("text-config-wins-over-llm-config");
  let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": [42, 50]
      },
      "llm_config": {
        "eos_token_id": [11, 13]
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("text_config should win over llm_config alias when both present");

  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    Some(EosTokenId::Many(vec![42, 50])),
    "text_config.eos_token_id must take precedence over llm_config.eos_token_id"
  );
}

#[test]
fn falsy_nested_eos_does_not_promote() {
  // Truthiness rules match `read_generation_eos`: a scalar `0` is
  // falsy, an empty list is falsy. Either way the promotion must
  // collapse to `None` and the tokenizer falls back to its own
  // `eos_token` from `tokenizer_config.json` (id 2 here). Pinning this
  // protects against a future change that drops the truthy filter and
  // starts forwarding `0`-shaped EOS to the tokenizer.
  let dir = fresh_dir("falsy-nested-eos");
  let cfg = r#"{
      "model_type": "mockvlm",
      "text_config": {
        "hidden_size": 8,
        "vocab_size": 5,
        "eos_token_id": 0
      },
      "mock_extra": 11
    }"#;
  std::fs::write(dir.join("config.json"), cfg).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);
  write_tokenizer_config_with_eos_c(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry).expect("falsy eos");

  // Promotion collapses to `None`, tokenizer falls back to its own
  // `eos_token` ("c" → id 2 from the tokenizer_config above).
  assert_eq!(
    ctx.config_ref().eos_token_id().cloned(),
    None,
    "scalar 0 nested eos must not promote (falsy)"
  );
  let eos_vec: Vec<u32> = ctx.tokenizer().eos_token_ids_iter().collect();
  assert_eq!(
    eos_vec,
    vec![2u32],
    "tokenizer should fall back to its tokenizer_config `eos_token` when nested is falsy"
  );
}

// ────────────────────────────────────────────────────────────────────
// preprocessor + processor_config dispatch fallback.
// ────────────────────────────────────────────────────────────────────

/// A minimal **image-preprocessor-only** `preprocessor_config.json` —
/// the real HF VLM layout: `image_mean` / `image_std` and a
/// model-specific `mock_image_size`, NO `processor_class`. Used by the
/// regression case where dispatch metadata must come from
/// `processor_config.json` instead.
fn mock_image_only_preprocessor_config_json(image_size: u32) -> String {
  format!(
    r#"{{
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "mock_image_size": {image_size}
      }}"#
  )
}

/// A `processor_config.json` carrying ONLY the dispatch class — the
/// `AutoProcessor`-style combined config from real HF VLM checkpoints.
fn mock_processor_class_only_config_json(processor_class: &str) -> String {
  format!(r#"{{ "processor_class": "{processor_class}" }}"#)
}

/// A `processor_config.json` carrying the dispatch class AND a
/// **required non-class processor-level field** (`image_seq_len`) — the
/// real `AutoProcessor` shape where `processor_config.json` holds
/// processor metadata a per-model processor needs *in addition to* the
/// image-preprocessor metadata that lives in `preprocessor_config.json`.
fn mock_processor_config_with_seq_len(processor_class: &str, image_seq_len: u32) -> String {
  format!(r#"{{ "processor_class": "{processor_class}", "image_seq_len": {image_seq_len} }}"#)
}

#[test]
fn processor_class_falls_back_to_processor_config_when_preprocessor_has_none() {
  // **Regression** test: real HF VLM dir where the
  // preprocessor file carries ONLY image-preprocessor metadata (no
  // `processor_class`) and `processor_config.json` carries the dispatch
  // class. A strict parse of `preprocessor_config.json` would fail
  // immediately → `processor_config.json` would never be tried →
  // otherwise-loadable VLM dir rejected. The tolerant resolution makes
  // the dispatch class come from `processor_config.json`, while the
  // constructor still sees the `preprocessor_config.json` body
  // (image-preprocessor metadata) — its `mock_image_size = 24`
  // round-trips through the mock processor, proving the constructor
  // body source.
  let dir = fresh_dir("split-dispatch-preprocessor-no-class");
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_image_only_preprocessor_config_json(24),
  )
  .unwrap();
  std::fs::write(
    dir.join("processor_config.json"),
    mock_processor_class_only_config_json("MockProc"),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", mock_processor_constructor());
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("dispatch class from processor_config.json + body from preprocessor_config.json");

  // The mock processor recorded the `mock_image_size = 24` it decoded
  // off the body — i.e. the constructor received the
  // `preprocessor_config.json` body (which has `mock_image_size`), NOT
  // the `processor_config.json` body (which has only
  // `processor_class`). Round-trip proof that the split-source
  // dispatch picked the right body.
  assert_eq!(
    ctx.processor().image_processor_config().size(),
    (24, 24),
    "constructor must see preprocessor_config.json body (image-preprocessor metadata)"
  );
}

#[test]
fn split_layout_carries_both_preprocessor_and_processor_config_bodies() {
  // **Regression** test: in the split layout
  // (`preprocessor_config.json` has the image-preprocessor metadata but
  // NO `processor_class`; `processor_config.json` supplies the dispatch
  // class) extracting ONLY the dispatch class from
  // `processor_config.json` and discarding that file's body would leave
  // a per-model processor needing a processor-level field from
  // `processor_config.json` (here `image_seq_len`) AND the
  // image-preprocessor metadata with no way to reach the discarded
  // body. BOTH bodies are therefore carried.
  let dir = fresh_dir("split-carries-both-bodies");
  // `preprocessor_config.json`: image-preprocessor metadata, no class.
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_image_only_preprocessor_config_json(24),
  )
  .unwrap();
  // `processor_config.json`: dispatch class + a REQUIRED non-class
  // processor-level field.
  std::fs::write(
    dir.join("processor_config.json"),
    mock_processor_config_with_seq_len("MockProc", 256),
  )
  .unwrap();

  // (a) `load_processor_config` directly: BOTH `Option<String>` slots
  // are populated, each keyed by file identity.
  let (proc_config, preprocessor_body, processor_body, filename) =
    load_processor_config(&dir).expect("split-layout processor config must resolve");
  assert_eq!(
    proc_config.processor_class(),
    "MockProc",
    "dispatch class must come from processor_config.json"
  );
  assert_eq!(
    filename, "preprocessor_config.json",
    "primary-body filename is the preprocessor file (image-preprocessor metadata source)"
  );
  let preprocessor_body = preprocessor_body.expect("preprocessor_config.json body must be carried");
  assert!(
    preprocessor_body.contains("mock_image_size"),
    "preprocessor body must carry the image-preprocessor metadata, got: {preprocessor_body}"
  );
  let processor_body =
    processor_body.expect("processor_config.json body must be carried, not discarded");
  assert!(
    processor_body.contains("image_seq_len"),
    "processor_config.json body must survive with its non-class field, got: {processor_body}"
  );

  // (b) Through the full `load()` pipeline: the per-model processor
  // constructor sees BOTH bodies on `LoadedProcessor`. The constructor
  // closure asserts `processor_config_json` is `Some` and exposes
  // `image_seq_len = 256` AND `preprocessor_config_json` is `Some` with
  // the image-preprocessor metadata — if either is missing it returns
  // an `Err` and the `load()` below fails.
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let asserting_processor_ctor: ProcessorConstructor = Box::new(
    |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
      let preprocessor = loaded.preprocessor_config_json.ok_or_else(|| {
        Error::MissingField(MissingFieldPayload::new(
          "LoadedProcessor",
          "preprocessor_config_json",
        ))
      })?;
      let processor = loaded.processor_config_json.ok_or_else(|| {
        Error::MissingField(MissingFieldPayload::new(
          "LoadedProcessor (the carried body)",
          "processor_config_json",
        ))
      })?;
      let pre: serde_json::Value = serde_json::from_str(preprocessor).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "asserting_processor_ctor: preprocessor body",
          "JSON",
          e,
        ))
      })?;
      let image_size = pre
        .get("mock_image_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or(Error::MissingField(MissingFieldPayload::new(
          "preprocessor body",
          "mock_image_size",
        )))?;
      let proc: serde_json::Value = serde_json::from_str(processor).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "asserting_processor_ctor: processor_config.json body",
          "JSON",
          e,
        ))
      })?;
      let seq_len = proc
        .get("image_seq_len")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
          Error::MissingField(MissingFieldPayload::new(
            "processor_config.json body (the discarded field)",
            "image_seq_len",
          ))
        })?;
      if seq_len != 256 {
        return Err(Error::LengthMismatch(
          crate::error::LengthMismatchPayload::new(
            "asserting_processor_ctor: image_seq_len round-trip",
            256,
            seq_len as usize,
          ),
        ));
      }
      Ok(Box::new(MockVlmProcessor {
        processor_class: loaded.processor_class.to_owned(),
        image_size,
      }))
    },
  );

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", asserting_processor_ctor);
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("split-layout load must surface BOTH config bodies to the processor constructor");
  // The constructor read `mock_image_size = 24` off the preprocessor
  // body — proof the preprocessor body also reached it alongside the
  // processor_config.json body.
  assert_eq!(ctx.processor().image_processor_config().size(), (24, 24));
}

#[test]
fn preferred_class_layout_still_carries_processor_config_body() {
  // **Regression** test for the *preferred-class* path:
  // `preprocessor_config.json` carries the `processor_class` (so it
  // wins dispatch) AND its own image-preprocessor metadata, while
  // `processor_config.json` ALSO exists with a required non-class
  // processor-level field (`image_seq_len`). Returning
  // `processor_config_json: None` and discarding the
  // `processor_config.json` body even though it was on disk would force
  // a per-model processor needing that field to re-open the file (the
  // TOCTOU/config-divergence the loader exists to prevent). BOTH bodies
  // are therefore carried, and the dispatch class still comes from
  // `preprocessor_config.json` (precedence unchanged).
  let dir = fresh_dir("preferred-class-carries-processor-body");
  // `preprocessor_config.json`: HAS `processor_class` + image metadata.
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 24),
  )
  .unwrap();
  // `processor_config.json`: also present, carrying a REQUIRED
  // non-class processor-level field (plus a class that must NOT be the
  // one used for dispatch).
  std::fs::write(
    dir.join("processor_config.json"),
    mock_processor_config_with_seq_len("OtherProc", 256),
  )
  .unwrap();

  // (a) `load_processor_config` directly: dispatch class is the
  // preferred file's, filename is the preferred file, and BOTH bodies
  // are populated — `processor_config.json`'s body is NOT discarded.
  let (proc_config, preprocessor_body, processor_body, filename) =
    load_processor_config(&dir).expect("preferred-class processor config must resolve");
  assert_eq!(
    proc_config.processor_class(),
    "MockProc",
    "dispatch class must come from preprocessor_config.json (precedence unchanged)"
  );
  assert_eq!(
    filename, "preprocessor_config.json",
    "primary-body filename is the preprocessor file"
  );
  let preprocessor_body = preprocessor_body.expect("preprocessor_config.json body must be carried");
  assert!(
    preprocessor_body.contains("mock_image_size"),
    "preprocessor body must carry the image-preprocessor metadata, got: {preprocessor_body}"
  );
  let processor_body = processor_body
    .expect("processor_config.json body must be carried in the preferred-class path too");
  assert!(
    processor_body.contains("image_seq_len"),
    "processor_config.json body must survive with its non-class field, got: {processor_body}"
  );

  // (b) Through the full `load()` pipeline: the per-model processor
  // constructor sees BOTH bodies on `LoadedProcessor`. The constructor
  // asserts `processor_config_json` is `Some` exposing `image_seq_len
  // = 256` AND `preprocessor_config_json` is `Some` with the
  // image-preprocessor metadata — failure of either makes `load()`
  // error.
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let asserting_processor_ctor: ProcessorConstructor = Box::new(
    |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
      let preprocessor = loaded.preprocessor_config_json.ok_or_else(|| {
        Error::MissingField(MissingFieldPayload::new(
          "LoadedProcessor",
          "preprocessor_config_json",
        ))
      })?;
      let processor = loaded.processor_config_json.ok_or_else(|| {
        Error::MissingField(MissingFieldPayload::new(
          "LoadedProcessor (the carried body)",
          "processor_config_json",
        ))
      })?;
      let pre: serde_json::Value = serde_json::from_str(preprocessor).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "asserting_processor_ctor: preprocessor body",
          "JSON",
          e,
        ))
      })?;
      let image_size = pre
        .get("mock_image_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or(Error::MissingField(MissingFieldPayload::new(
          "preprocessor body",
          "mock_image_size",
        )))?;
      let proc: serde_json::Value = serde_json::from_str(processor).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "asserting_processor_ctor: processor_config.json body",
          "JSON",
          e,
        ))
      })?;
      let seq_len = proc
        .get("image_seq_len")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
          Error::MissingField(MissingFieldPayload::new(
            "processor_config.json body (the discarded field)",
            "image_seq_len",
          ))
        })?;
      if seq_len != 256 {
        return Err(Error::LengthMismatch(
          crate::error::LengthMismatchPayload::new(
            "asserting_processor_ctor: image_seq_len round-trip",
            256,
            seq_len as usize,
          ),
        ));
      }
      Ok(Box::new(MockVlmProcessor {
        processor_class: loaded.processor_class.to_owned(),
        image_size,
      }))
    },
  );

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  // ONLY `MockProc` is registered — if dispatch had used the
  // `processor_config.json` class (`OtherProc`) the lookup would miss.
  let processor_registry =
    VlmProcessorTypeRegistry::new().with("MockProc", asserting_processor_ctor);
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry).expect(
    "preferred-class load must dispatch on preprocessor_config.json's class and surface \
       BOTH config bodies",
  );
  // Round-trip proof: the constructor decoded `mock_image_size = 24`
  // off the preprocessor body, and only `MockProc` is registered so
  // dispatch used the preprocessor file's `processor_class`.
  assert_eq!(ctx.processor().image_processor_config().size(), (24, 24));
}

#[test]
fn neither_processor_config_file_has_processor_class_is_recoverable_error() {
  // Both files present, NEITHER has `processor_class`. The error must
  // be a recoverable `Backend` naming the dir; we additionally check the
  // message identifies the missing dispatch field so an operator can
  // diagnose without source-diving.
  let dir = fresh_dir("neither-has-processor-class");
  std::fs::write(dir.join("config.json"), mock_config_json("mockvlm")).unwrap();
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_image_only_preprocessor_config_json(16),
  )
  .unwrap();
  std::fs::write(
    dir.join("processor_config.json"),
    r#"{ "some_other_key": 1 }"#,
  )
  .unwrap();
  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry = VlmProcessorTypeRegistry::new();
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let Err(err) = load(&configuration, &model_registry, &processor_registry) else {
    panic!("missing-processor-class across both files must error");
  };
  match &err {
    Error::MissingField(p) => {
      assert_eq!(
        p.field(),
        "processor_class",
        "MissingField should name processor_class as the field"
      );
      assert!(
        p.type_name().contains("ProcessorConfig"),
        "type_name should name ProcessorConfig, got: {}",
        p.type_name()
      );
    }
    _ => panic!("expected MissingField for missing processor_class, got: {err:?}"),
  }
}

/// A concrete processor with a method that is NOT on the [`Processor`]
/// trait — standing in for the per-model concrete-only surface
/// (multimodal prompt assembly / video handling / tool+chat
/// formatting) a real `Qwen2VLProcessor` / `PixtralProcessor` carries.
struct MockConcreteProcessor {
  special: u32,
}

impl MockConcreteProcessor {
  /// Concrete-only method unreachable through `dyn Processor` — only a
  /// successful downcast to the concrete type can call it.
  fn mock_special(&self) -> u32 {
    self.special
  }
}

impl Processor for MockConcreteProcessor {
  fn image_processor_config(&self) -> ImageProcessorConfig {
    ImageProcessorConfig::new()
      .with_size((1, 1))
      .with_mean([0.5, 0.5, 0.5])
      .with_std([0.5, 0.5, 0.5])
      .with_rescale_factor(1.0 / 255.0)
      .with_do_resize(true)
      .with_do_rescale(true)
      .with_do_normalize(true)
      .with_resample(ResizeFilter::Bilinear)
      .with_color_order(ColorOrder::Rgb)
  }

  fn as_any(&self) -> &dyn std::any::Any {
    self
  }

  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

#[test]
fn loaded_processor_downcasts_to_concrete_per_model_type() {
  // `LoadedVlmContext.processor` is an erased
  // `Box<dyn Processor>`, but a caller needs the CONCRETE per-model
  // processor (`Qwen2VLProcessor` / `PixtralProcessor` / …) to reach
  // its concrete-only methods (multimodal prompt assembly / video /
  // tool+chat formatting). The trait upcasts to `Any` via `as_any`, so
  // the erased processor handed back by `load()` can be downcast to its
  // concrete type end-to-end. Without the `as_any` + `'static` bound
  // there would be no way to recover the concrete type off `load()`'s
  // output and the concrete-only API would be unreachable; this proves
  // the round-trip works.
  let dir = fresh_dir("processor-downcast");
  write_vlm_dir(
    &dir,
    "mockvlm",
    "preprocessor_config.json",
    "MockConcreteProc",
    64,
  );

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry = VlmProcessorTypeRegistry::new().with(
    "MockConcreteProc",
    Box::new(
      |_loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
        Ok(Box::new(MockConcreteProcessor { special: 4242 }))
      },
    ),
  );
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("load should construct the concrete processor");

  // Recover the concrete per-model processor off the erased
  // `Box<dyn Processor>` and call its concrete-only method.
  let concrete = ctx
    .processor()
    .as_any()
    .downcast_ref::<MockConcreteProcessor>()
    .expect("loaded processor must downcast to its concrete per-model type");
  assert_eq!(concrete.mock_special(), 4242);
}

#[test]
fn loaded_processor_reads_model_config_json_only_arch_field() {
  // A concrete per-model processor's
  // downcast-only methods may need an arch field that lives ONLY in the
  // model `config.json` (e.g. a `hidden_size` / `image_token_index`
  // nested under `text_config`), NOT in either processor-config body.
  // If `LoadedProcessor` exposed only the processor configs + the typed
  // `VlmBaseConfig` subset, such a processor would have to re-open
  // `config.json` itself — losing the single-read TOCTOU consistency the
  // loader provides. `LoadedProcessor.config_json` instead carries the
  // SAME body the model constructor received, so the processor reads the
  // field off the loader's single read.
  let dir = fresh_dir("processor-reads-model-config-json");
  // `config.json`: nested-shaped, carries `text_config.hidden_size = 8`
  // — an arch field present ONLY here (the processor configs below do
  // NOT carry it).
  std::fs::write(dir.join("config.json"), mock_nested_config_json("mockvlm")).unwrap();
  // The processor config carries `mock_image_size = 999` — DELIBERATELY
  // different from `text_config.hidden_size = 8`. The constructor below
  // ignores `mock_image_size` and instead drives `image_size` from the
  // `config.json`-only `hidden_size`, so a passing `(8, 8)` assertion
  // proves the value came from `config_json`, not the processor config.
  std::fs::write(
    dir.join("preprocessor_config.json"),
    mock_preprocessor_config_json("MockProc", 999),
  )
  .unwrap();
  let mut weights: Weights = HashMap::new();
  weights.insert(
    "mock.weight".to_owned(),
    Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap(),
  );
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();
  write_tokenizer(&dir);

  let config_reading_ctor: ProcessorConstructor = Box::new(
    |loaded: &LoadedProcessor<'_>| -> Result<Box<dyn Processor>> {
      // Read the arch field off `LoadedProcessor.config_json` — the
      // SAME single-read body the model constructor saw. It is NOT in
      // either processor-config body.
      let cfg: serde_json::Value = serde_json::from_str(loaded.config_json).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "processor ctor: model config_json",
          "JSON",
          e,
        ))
      })?;
      let hidden_size = cfg
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| {
          Error::MissingField(MissingFieldPayload::new(
            "processor ctor: LoadedProcessor.config_json (config.json-only arch field)",
            "text_config.hidden_size",
          ))
        })?;
      // Drive `image_size` from the config.json-only field (NOT the
      // processor config's `mock_image_size`) so the test can assert
      // the value round-tripped from `config_json`.
      Ok(Box::new(MockVlmProcessor {
        processor_class: loaded.processor_class.to_owned(),
        image_size: hidden_size,
      }))
    },
  );

  let model_registry = VlmTypeRegistry::new().with("mockvlm", mock_vlm_constructor());
  let processor_registry = VlmProcessorTypeRegistry::new().with("MockProc", config_reading_ctor);
  let configuration = VlmModelConfiguration::from_directory(&dir);

  let ctx = load(&configuration, &model_registry, &processor_registry)
    .expect("load must surface the model config.json to the processor constructor");

  // `text_config.hidden_size = 8` round-tripped through
  // `LoadedProcessor.config_json` into the processor — NOT the
  // processor config's `mock_image_size = 999`.
  assert_eq!(
    ctx.processor().image_processor_config().size(),
    (8, 8),
    "processor must have read hidden_size=8 off LoadedProcessor.config_json, \
       not mock_image_size=999 off the processor config"
  );
}
