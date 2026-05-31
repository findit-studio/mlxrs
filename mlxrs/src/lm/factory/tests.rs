//! End-to-end load-factory tests, driven by a **mock** model type
//! registered into a fresh [`ModelTypeRegistry`] (per the project's
//! no-model-arch rule, this PR ships the seam, not architectures — so the
//! end-to-end path is proven against a hand-traced mock constructor over a
//! temp model directory).

use std::{
  path::PathBuf,
  sync::atomic::{AtomicU64, Ordering},
};

use crate::error::MissingFieldPayload;

use super::*;
use crate::{
  array::Array,
  lm::{cache::KvCache, generate::FinishReason},
};

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
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "MockLoadedModel::forward: tokens must be rank-1 [S] or rank-2 [B, S]",
          other.len() as u32,
          other.to_vec(),
        )));
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
    let raw: serde_json::Value = serde_json::from_str(&loaded.config_json).map_err(|e| {
      Error::Parse(crate::error::ParsePayload::new(
        "mock ctor: bad config json",
        "config.json",
        Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
      ))
    })?;
    let mock_extra = raw
      .get("mock_extra")
      .and_then(serde_json::Value::as_i64)
      .ok_or(Error::MissingField(MissingFieldPayload::new(
        "mock ctor",
        "mock_extra",
      )))?;
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
  let dir = std::env::temp_dir().join(format!("mlxrs-lm-factory-{tag}-{}-{n}", std::process::id()));
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
  assert_eq!(ctx.config.model_type(), "mockarch");
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
  assert_eq!(ctx.config.model_type(), "mockarch");
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
  // REAL split layout: the model dir has config +
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
  // An UNREGISTERED `model_type` must be rejected BEFORE
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
  // The `config_json` handed to the constructor must be the
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
        Some(loaded.config.model_type())
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

// ──────────────────────────────────────────────────────────────────────
//   ModelContext — the owning `(model, tokenizer, config)` bundle.
//
//   Hand-traced tests over the crate-shared deterministic `MockModel`
//   (`crate::lm::model::MockModel`) and the shared `tests/fixtures`
//   tokenizer (a tiny WordLevel model + a jinja chat template), proving
//   the bundle owns the triple and that `encode` / `decode` /
//   `apply_chat_template` / `generate` / `stream_generate` forward to the
//   same underlying calls a hand-wired `lm::generate` / tokenizer would.
//   No `peak_memory()` magnitude asserts (process-global counter).
// ──────────────────────────────────────────────────────────────────────

use crate::lm::{generate::GenConfig, model::MockModel};

/// Load the shared `tests/fixtures` tokenizer (WordLevel vocab + jinja chat
/// template + `</s>` eos), reachable from the in-crate `#[cfg(test)]` build
/// via `CARGO_MANIFEST_DIR` — the same fixture `lm::generate`'s tests use.
fn fixture_tokenizer() -> Tokenizer {
  let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("fixtures");
  Tokenizer::from_path(&dir, None).expect("load fixture tokenizer")
}

/// A minimal typed [`Config`] for the bundle tests: `vocab_size` and
/// `num_hidden_layers` are the keys [`ModelContext`] actually consults
/// (vocab for the mock, layer count for the per-call cache), the rest are
/// the typed [`Config`]'s required fields filled with inert values.
fn mock_config(vocab: i32, num_layers: i32) -> Config {
  Config::from_json(&format!(
    r#"{{
        "model_type": "mockarch",
        "hidden_size": 8,
        "num_hidden_layers": {num_layers},
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "rope_theta": 10000.0,
        "vocab_size": {vocab},
        "tie_word_embeddings": false
      }}"#
  ))
  .expect("mock config parses")
}

/// Build a [`ModelContext`] over a [`MockModel`] of the given `vocab` and a
/// matching fixture tokenizer / [`Config`]. `MockModel`'s greedy argmax is
/// the last vocab index, so `vocab` chooses the generated token id.
fn mock_context(vocab: i32, num_layers: i32) -> ModelContext {
  ModelContext::new(
    Box::new(MockModel::new(vocab as usize)),
    fixture_tokenizer(),
    mock_config(vocab, num_layers),
  )
}

#[test]
fn context_owns_and_exposes_model_tokenizer_config() {
  // The bundle owns the triple and the accessors hand back the SAME values
  // (the config's typed fields, a working tokenizer, a runnable model).
  let ctx = mock_context(8, 2);

  assert_eq!(ctx.config().model_type(), "mockarch");
  assert_eq!(ctx.config().vocab_size, 8);
  assert_eq!(ctx.config().num_hidden_layers, 2);

  // The tokenizer accessor returns a real, working tokenizer.
  let ids = ctx.tokenizer().encode("the quick brown", false).unwrap();
  assert_eq!(ids.len(), 3);

  // The model accessor returns the runnable mock: one forward yields
  // `[B, S, vocab]` logits (vocab == 8).
  let mut cache: Vec<Box<dyn crate::lm::cache::KvCache>> = Vec::new();
  let tokens = Array::from_slice::<i32>(&[1, 2, 3], &(1usize, 3)).unwrap();
  let logits = ctx.model().forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, 8]);
}

#[test]
fn encode_forwards_to_tokenizer() {
  // `ModelContext::encode` is a thin forward — it must produce byte-for-byte
  // the ids the bundled tokenizer's own `encode` produces.
  let ctx = mock_context(8, 1);
  let text = "the quick brown world";
  let via_context = ctx.encode(text, false).unwrap();
  let via_tokenizer = ctx.tokenizer().encode(text, false).unwrap();
  assert_eq!(via_context, via_tokenizer);
  // WordLevel over 4 known fixture words ⇒ 4 ids.
  assert_eq!(via_context.len(), 4);
}

#[test]
fn decode_forwards_to_tokenizer_and_round_trips_encode() {
  // `decode` forwards to the tokenizer, and `encode`→`decode` round-trips
  // the fixture vocab words (every word here is in-vocab).
  let ctx = mock_context(8, 1);
  let ids = ctx.encode("hello world", false).unwrap();
  let via_context = ctx.decode(&ids, true).unwrap();
  let via_tokenizer = ctx.tokenizer().decode(&ids, true).unwrap();
  assert_eq!(via_context, via_tokenizer);
  assert_eq!(via_context, "hello world");
}

#[test]
fn apply_chat_template_forwards_to_tokenizer() {
  // `apply_chat_template` forwards to the tokenizer; the fixture template is
  // `{{bos}}{% for m %}<|role|>content{% endfor %}{% gen-prompt %}`.
  let ctx = mock_context(8, 1);
  let messages = serde_json::json!([
    {"role": "user", "content": "hello"}
  ]);

  let via_context = ctx
    .apply_chat_template(&messages, None, true, false, None)
    .unwrap();
  let via_tokenizer = ctx
    .tokenizer()
    .apply_chat_template(&messages, None, true, false, None)
    .unwrap();
  assert_eq!(via_context, via_tokenizer);
  // Hand-trace the fixture template: bos + the user turn + the
  // add_generation_prompt assistant marker.
  assert_eq!(via_context, "<s><|user|>hello<|assistant|>");
}

#[test]
fn apply_chat_template_ids_forwards_and_equals_render_then_encode() {
  // The `tokenize: true` form forwards to the tokenizer AND equals
  // `apply_chat_template` followed by `encode` (its own documented
  // composition).
  let ctx = mock_context(8, 1);
  let messages = serde_json::json!([
    {"role": "user", "content": "the quick"}
  ]);

  let via_context = ctx
    .apply_chat_template_ids(&messages, None, true, false, None)
    .unwrap();
  let via_tokenizer = ctx
    .tokenizer()
    .apply_chat_template_ids(&messages, None, true, false, None)
    .unwrap();
  assert_eq!(via_context, via_tokenizer);

  let rendered = ctx
    .apply_chat_template(&messages, None, true, false, None)
    .unwrap();
  assert_eq!(via_context, ctx.encode(&rendered, false).unwrap());
}

#[test]
fn apply_chat_template_rejects_generation_prompt_with_continue() {
  // The mutually-exclusive-flags guard lives on the tokenizer; the forward
  // must surface that error (not panic) just as a direct call would.
  let ctx = mock_context(8, 1);
  let messages = serde_json::json!([{"role": "user", "content": "hello"}]);
  let err = ctx
    .apply_chat_template(
      &messages, None, /*gen*/ true, /*continue*/ true, None,
    )
    .expect_err("gen-prompt + continue must error");
  assert!(
    err.to_string().contains("continue_final_message"),
    "got: {err}"
  );
}

#[test]
fn generate_forwards_and_runs_to_length() {
  // `MockModel::new(8)`'s greedy argmax is the last index (7), and the
  // fixture eos is `</s>` (id 2) — so token 7 is never eos and a greedy run
  // proceeds for the full `max_tokens`. `generate` builds its own per-call
  // cache (sized from `num_hidden_layers`) and forwards to
  // `lm::generate::generate`.
  let ctx = mock_context(8, 2);
  let prompt = ctx.encode("hello world", false).unwrap();
  let cfg = GenConfig {
    max_tokens: 3,
    ..Default::default()
  };
  let (text, stats) = ctx.generate(&prompt, cfg).expect("generate");

  // Three non-eos tokens generated, the prompt counted, length-capped run.
  assert_eq!(stats.generation_tokens, 3);
  assert_eq!(stats.prompt_tokens, prompt.len());
  // The collected text is exactly the detokenization of the three argmax
  // tokens — i.e. forwarding to `lm::generate` produced a real decode.
  assert_eq!(text, ctx.decode(&[7, 7, 7], true).unwrap());
}

#[test]
fn generate_stops_on_eos_token() {
  // `MockModel::new(3)`'s greedy argmax is index 2 == the fixture `</s>`
  // eos id: the very first sampled token is eos, so generation stops
  // immediately with no produced text (mlx-lm never detokenizes the eos
  // token). Proves the bundle's eos handling is the `lm::generate` one.
  let ctx = mock_context(3, 1);
  let prompt = ctx.encode("hello", false).unwrap();
  let cfg = GenConfig {
    max_tokens: 16,
    ..Default::default()
  };
  let (text, stats) = ctx.generate(&prompt, cfg).expect("generate");
  assert!(
    text.is_empty(),
    "eos token contributes no text, got {text:?}"
  );
  // The eos token itself counts as one generation token (mlx-lm `n + 1`).
  assert_eq!(stats.generation_tokens, 1);
}

#[test]
fn stream_generate_forwards_and_yields_per_token_responses() {
  // `stream_generate` forwards to `lm::generate::stream_generate`: a greedy
  // run over `MockModel::new(8)` (argmax 7, never eos) yields one response
  // per token and the final response carries `finish_reason = "length"`.
  let ctx = mock_context(8, 2);
  let prompt = ctx.encode("the quick", false).unwrap();
  let cfg = GenConfig {
    max_tokens: 4,
    ..Default::default()
  };

  let mut reasons = Vec::new();
  let mut collected = String::new();
  for resp in ctx.stream_generate(&prompt, cfg) {
    let r = resp.expect("stream step");
    collected.push_str(&r.text);
    reasons.push(r.finish_reason);
  }

  // Four tokens ⇒ four responses; only the last has a finish_reason.
  assert_eq!(reasons.len(), 4);
  assert_eq!(reasons[0], None);
  assert_eq!(reasons[3], Some(FinishReason::Length));
  // Streaming and the collecting `generate` agree on the assembled text.
  let (gen_text, _) = ctx
    .generate(
      &prompt,
      GenConfig {
        max_tokens: 4,
        ..Default::default()
      },
    )
    .unwrap();
  assert_eq!(collected, gen_text);
}

#[test]
fn from_loaded_model_context_wraps_the_triple() {
  // `ModelContext` is buildable straight from the loader's product struct
  // (`load(..)?.into()`): load a real mock model dir, wrap it, and confirm
  // the bundle exposes the loaded model + tokenizer + config.
  let dir = fresh_dir("ctx-from-loaded");
  write_model_dir(&dir, "mockarch");
  let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
  let configuration = ModelConfiguration::from_directory(&dir);

  let loaded = load(&configuration, &registry).expect("load");
  let ctx: ModelContext = loaded.into();

  assert_eq!(ctx.config().model_type(), "mockarch");
  // The loaded mock arch has vocab 5 (see `mock_config_json`); the model
  // forwards `[B, S, 5]` logits.
  let mut cache: Vec<Box<dyn crate::lm::cache::KvCache>> = Vec::new();
  let tokens = Array::from_slice::<i32>(&[0, 1, 2], &(1usize, 3)).unwrap();
  let logits = ctx.model().forward(&tokens, &mut cache).unwrap();
  assert_eq!(logits.shape(), vec![1, 3, 5]);
  // The tokenizer loaded from the same dir (the 3-token fixture vocab).
  assert_eq!(ctx.encode("a b c", false).unwrap().len(), 3);
}

#[test]
fn context_load_convenience_equals_load_then_into() {
  // `ModelContext::load` is the one-call convenience — it must yield the
  // same bundle as `load(..)` followed by `.into()`.
  let dir = fresh_dir("ctx-load");
  write_model_dir(&dir, "mockarch");
  let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
  let configuration = ModelConfiguration::from_directory(&dir);

  let ctx = ModelContext::load(&configuration, &registry).expect("ModelContext::load");
  assert_eq!(ctx.config().model_type(), "mockarch");
  assert_eq!(ctx.config().vocab_size, 5);
  assert_eq!(ctx.encode("a b c", false).unwrap().len(), 3);
}

#[test]
fn context_load_propagates_unknown_model_type_error() {
  // The convenience `load` surfaces the same recoverable errors `load()`
  // does — an unregistered `model_type` is an `Error`, not a panic.
  let dir = fresh_dir("ctx-load-unknown");
  write_model_dir(&dir, "nope");
  let registry = ModelTypeRegistry::new().with("mockarch", mock_constructor());
  let configuration = ModelConfiguration::from_directory(&dir);

  let Err(err) = ModelContext::load(&configuration, &registry) else {
    panic!("unknown model_type must error");
  };
  assert!(
    err.to_string().contains("unsupported model type"),
    "got: {err}"
  );
}

#[test]
fn into_parts_round_trips_new() {
  // `into_parts` is the inverse of `new`: decomposing then rebuilding
  // preserves the config, and the model/tokenizer stay runnable.
  let ctx = mock_context(8, 3);
  let (model, tokenizer, config) = ctx.into_parts();
  assert_eq!(config.num_hidden_layers, 3);

  let rebuilt = ModelContext::new(model, tokenizer, config);
  assert_eq!(rebuilt.config().vocab_size, 8);
  assert_eq!(rebuilt.encode("hello", false).unwrap().len(), 1);
}
