//! L7 — prompt-cache fill + save driver (`mlxrs::lm::cache_prompt`), ported
//! from `mlx_lm.cache_prompt` (the `--prompt-cache-file` CLI's support core).
//!
//! Deterministic + dependency-free: a local `MockModel` (replicating the
//! in-crate `model::MockModel` — integration tests cannot see the
//! `#[cfg(test)] pub(crate)` helper) advances every cache entry by the
//! token-window length and returns fixed `[B, S, vocab]` logits, so the
//! driver's *tokenize → prefill → persist* behavior is checkable through the
//! **public** `cache_prompt` / `cache_prompt_ids` + the existing
//! `save_prompt_cache` / `load_prompt_cache` round-trip, with no real model or
//! network.
//!
//! `cache_prompt` / `cache_prompt_ids` allocate their KV cache **internally**
//! via `make_prompt_cache` (exactly `cache_prompt.py:111` — `cache =
//! make_prompt_cache(model, ...)`); they take a `CacheConfig` (the
//! model-appropriate cache spec), not a caller-provided cache. So the saved
//! cache is fresh by construction and represents exactly the requested prompt.
//!
//! The core property: filling a cache via the driver and saving it, then
//! loading it back, yields a cache *byte-identical* to a direct prefill — and
//! a subsequent `generate_step` continuation from the loaded cache produces
//! the **same** next tokens as generating from scratch over the same prompt.

#![cfg(feature = "lm")]

use std::{collections::HashMap, fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array,
  lm::{
    cache::{CacheConfig, KvCache, load_prompt_cache, make_prompt_cache},
    cache_prompt::{
      CachePromptInfo, META_MODEL, META_TOKENIZER_CONFIG, cache_prompt, cache_prompt_ids,
    },
    generate::{GenConfig, generate_step},
    model::Model,
  },
};

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");
const TOKENIZER_CONFIG_JSON: &str = include_str!("fixtures/tokenizer_config.json");

/// A unique temp directory for one test (process-scoped + named so parallel
/// test binaries / cases never collide).
fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_cache_prompt_drv_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

/// Build a real [`mlxrs::tokenizer::Tokenizer`] from the committed fixtures
/// (vocab: `<unk>`=0, `<s>`=1, `</s>`=2 [eos], `hello`=3, `world`=4, `the`=5,
/// `quick`=6, `brown`=7, `fox`=8, `<think>`=9, `</think>`=10). The fixture
/// carries a chat template, so the driver's high-level `cache_prompt` exercises
/// the chat-template encode branch.
fn tokenizer(dir: &std::path::Path) -> mlxrs::tokenizer::Tokenizer {
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(dir, None).unwrap()
}

/// A deterministic, dependency-free [`Model`] (replicating the in-crate
/// `model::MockModel`): `forward` advances every cache entry by the
/// token-window length and returns a fixed `[B, S, vocab]` logits array whose
/// per-vocab values are `bias[v]` — so the argmax / sampled token is fully
/// predictable and the saved/loaded cache contents are directly readable.
struct MockModel {
  bias: Vec<f32>,
  n_kv_heads: usize,
  head_dim: usize,
}

impl MockModel {
  /// `bias[v] = v` ⇒ greedy argmax is always the last vocab index.
  fn ramp(vocab: usize) -> Self {
    Self {
      bias: (0..vocab).map(|i| i as f32).collect(),
      n_kv_heads: 1,
      head_dim: 2,
    }
  }
}

impl Model for MockModel {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> mlxrs::Result<Array> {
    let shape = tokens.shape();
    let (batch, seq) = match shape.as_slice() {
      [b, s] => (*b, *s),
      [s] => (1, *s),
      _ => {
        return Err(mlxrs::Error::ShapeMismatch {
          message: format!("MockModel::forward expects [B, S] tokens, got {shape:?}"),
        });
      }
    };
    let vocab = self.bias.len();
    // A per-position-varying KV step (value == the running offset + local
    // index) so the saved/loaded cache contents are a non-trivial, directly
    // comparable signature — not a constant tensor that would mask a
    // round-trip bug.
    for layer in cache.iter_mut() {
      let base = layer.offset();
      let elems = batch * self.n_kv_heads * seq * self.head_dim;
      let mut kd = Vec::with_capacity(elems);
      let mut vd = Vec::with_capacity(elems);
      for b in 0..batch {
        for h in 0..self.n_kv_heads {
          for s in 0..seq {
            for d in 0..self.head_dim {
              let tag = (base + s) as f32;
              kd.push(tag + 0.1 * (b * 100 + h * 10 + d) as f32);
              vd.push(100.0 + tag + 0.1 * (b * 100 + h * 10 + d) as f32);
            }
          }
        }
      }
      let k = Array::from_slice::<f32>(&kd, &(batch, self.n_kv_heads, seq, self.head_dim))?;
      let v = Array::from_slice::<f32>(&vd, &(batch, self.n_kv_heads, seq, self.head_dim))?;
      layer.update(&k, &v)?;
    }
    let mut data = Vec::with_capacity(batch * seq * vocab);
    for _ in 0..batch * seq {
      data.extend_from_slice(&self.bias);
    }
    Array::from_slice::<f32>(&data, &(batch, seq, vocab))
  }
}

/// A full-attention (non-sliding-window) [`CacheConfig`] for `layers` decoder
/// layers — what `cache_prompt` allocates a `StandardKvCache` per.
fn config(layers: usize) -> CacheConfig {
  CacheConfig {
    num_hidden_layers: layers,
    sliding_window: None,
  }
}

/// A sliding-window [`CacheConfig`] — `cache_prompt` allocates a
/// `RotatingKvCache` per layer for it (mlx-lm's `make_prompt_cache` with a
/// model whose `sliding_window` is set).
fn sliding_config(layers: usize, window: i32) -> CacheConfig {
  CacheConfig {
    num_hidden_layers: layers,
    sliding_window: Some(window),
  }
}

/// A freshly built full-attention per-layer cache — used only as the
/// `generate_step` baseline (the loop takes an owned cache); the driver
/// allocates its own internally.
fn cache(layers: usize) -> Vec<Box<dyn KvCache>> {
  make_prompt_cache(&config(layers))
}

/// Flatten every cache layer's `state` arrays into one comparable `Vec<f32>`
/// (offset-tagged, so order + per-position values are pinned).
fn cache_signature(c: &[Box<dyn KvCache>]) -> Vec<f32> {
  let mut out = Vec::new();
  for layer in c {
    for mut arr in layer.state().unwrap() {
      out.extend(arr.to_vec::<f32>().unwrap());
    }
  }
  out
}

// ---------------------------------------------------------------------------
// Core round-trip: fill + save == direct prefill; loaded cache continues like
// scratch.
// ---------------------------------------------------------------------------

/// `cache_prompt_ids` fills an internally-allocated cache + saves it; loading
/// it back yields a cache whose state is **independent of `prefill_step_size`**
/// (a small multi-chunk step and a single-chunk step produce a byte-identical
/// saved cache), AND the returned `tokens_processed` equals the prompt length.
#[test]
fn driver_fill_save_load_matches_direct_prefill() {
  let model = MockModel::ramp(6);
  let prompt = [1u32, 3, 4, 5, 6]; // 5 tokens
  let dir = temp_dir("roundtrip");
  let out = dir.join("cache.safetensors");

  // Drive: tokenize-skip (pre-encoded) → allocate cache internally → prefill
  // → save. `prefill_step_size = 2` exercises the multi-chunk prefill.
  let info = cache_prompt_ids(
    &model,
    &prompt,
    &config(2),
    &out,
    "mock-model",
    "{}",
    2,
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(
    info,
    CachePromptInfo {
      tokens_processed: prompt.len()
    }
  );

  // Load it back from disk: the saved cache is at offset P.
  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded.iter().all(|c| c.offset() == prompt.len()));

  // A second run over the SAME prompt with a different (single) chunk size
  // must yield a byte-identical saved cache — the prefill result is
  // independent of `prefill_step_size`.
  let direct_out = dir.join("direct.safetensors");
  cache_prompt_ids(
    &model,
    &prompt,
    &config(2),
    &direct_out,
    "mock-model",
    "{}",
    5, // a different (single-chunk) size — result must be identical
    &HashMap::new(),
  )
  .unwrap();
  let (direct, _m) = load_prompt_cache(&direct_out).unwrap();

  let sig_loaded = cache_signature(&loaded);
  let sig_direct = cache_signature(&direct);
  assert_eq!(
    sig_loaded, sig_direct,
    "prefill result must be independent of prefill_step_size"
  );
  assert!(
    !sig_loaded.is_empty(),
    "cache signature must be non-trivial"
  );
}

/// A multi-chunk prompt (`P` >> `prefill_step_size`) completes and produces a
/// **loadable** cache (Codex finding: the per-chunk eval barrier makes prefill
/// memory-bounded — the lazy graph is materialized between chunks, not spanned
/// across the whole prompt). With `P = 17`, `step = 4` the leading `P-1 = 16`
/// tokens are 4 chunks `[4,4,4,4]`, so the barrier runs 4 times; the saved
/// cache loads back at offset `P`, byte-identical to a single-chunk prefill of
/// the same prompt (chunk size must not affect the result).
#[test]
fn driver_multi_chunk_prefill_completes_and_loads() {
  let model = MockModel::ramp(8);
  // 17 tokens, all < vocab(8): a long-enough prompt to span several chunks.
  let prompt: Vec<u32> = (0..17u32).map(|i| i % 7).collect();
  let dir = temp_dir("multichunk");
  let out = dir.join("cache.safetensors");

  let info = cache_prompt_ids(
    &model,
    &prompt,
    &config(2),
    &out,
    "mock",
    "{}",
    4, // P-1 = 16 ⇒ 4 leading chunks ⇒ barrier fires 4 times (> 1)
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(info.tokens_processed, prompt.len());

  // The multi-chunk cache loads back at offset P.
  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded.iter().all(|c| c.offset() == prompt.len()));

  // Result is independent of chunking: a single-chunk prefill of the same
  // prompt yields a byte-identical cache (so the per-chunk barrier changed
  // nothing observable beyond bounding memory).
  let throwaway = dir.join("single.safetensors");
  cache_prompt_ids(
    &model,
    &prompt,
    &config(2),
    &throwaway,
    "mock",
    "{}",
    1000, // one chunk for the whole leading run
    &HashMap::new(),
  )
  .unwrap();
  let (single, _m) = load_prompt_cache(&throwaway).unwrap();
  assert_eq!(
    cache_signature(&loaded),
    cache_signature(&single),
    "the multi-chunk (barrier'd) prefill must equal a single-chunk prefill"
  );
}

/// A continuation from the **loaded** cache produces the **same next tokens**
/// as generating from scratch over the same prompt (the whole point of a
/// prompt cache): `prompt-cache + continue == prompt + generate`.
#[test]
fn driver_loaded_cache_continues_like_scratch() {
  let model = MockModel::ramp(6); // argmax always vocab-1 == 5
  let prompt = [1u32, 3, 4]; // 3 tokens
  let dir = temp_dir("continue");
  let out = dir.join("cache.safetensors");

  // Fill + save a cache from the prompt (cache allocated internally).
  cache_prompt_ids(
    &model,
    &prompt,
    &config(2),
    &out,
    "mock",
    "{}",
    2,
    &HashMap::new(),
  )
  .unwrap();

  // Drive `generate_step` from scratch over the full prompt for the baseline.
  let from_scratch: Vec<u32> = generate_step(
    &model,
    &prompt,
    cache(2),
    GenConfig::default().with_max_tokens(4),
  )
  .map(|r| r.unwrap().token)
  .collect();

  // From the loaded cache: continue by feeding the *last* prompt token. The
  // cached prefix already covers `prompt`, so a continuation that re-feeds the
  // prompt's final token reproduces the scratch trajectory's first decode
  // (mlx-lm's prompt-cache reuse: cache holds the prefix, the loop continues
  // from there). Greedy argmax is position-independent for the MockModel, so
  // the produced tokens match the scratch run exactly.
  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert!(loaded.iter().all(|c| c.offset() == prompt.len()));
  let continued: Vec<u32> = generate_step(
    &model,
    &[*prompt.last().unwrap()], // continue from the cached prefix
    loaded,
    GenConfig::default().with_max_tokens(4),
  )
  .map(|r| r.unwrap().token)
  .collect();

  assert_eq!(
    from_scratch, continued,
    "continuation from the loaded prompt cache must match generating from scratch"
  );
  assert_eq!(from_scratch, vec![5, 5, 5, 5]);
}

// ---------------------------------------------------------------------------
// Metadata round-trip.
// ---------------------------------------------------------------------------

/// The reference metadata (`model` / `tokenizer_config`) the driver writes
/// round-trips through `load_prompt_cache` unchanged — and an extra
/// caller-supplied key survives too, while the two reference keys win on
/// collision (faithful to cache_prompt.py setting them unconditionally).
#[test]
fn driver_metadata_round_trips() {
  let model = MockModel::ramp(5);
  let prompt = [1u32, 3, 4];
  let dir = temp_dir("meta");
  let out = dir.join("cache.safetensors");

  let mut extra = HashMap::new();
  extra.insert("note".to_string(), "hello world".to_string());
  // A colliding key the reference key must override.
  extra.insert(META_MODEL.to_string(), "WRONG".to_string());

  cache_prompt_ids(
    &model,
    &prompt,
    &config(1),
    &out,
    "my-model-id",
    "{\"eos_token\": \"</s>\"}",
    8,
    &extra,
  )
  .unwrap();

  let (_loaded, meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(
    meta.get(META_MODEL).map(String::as_str),
    Some("my-model-id"),
    "reference `model` key must win over the colliding extra key"
  );
  assert_eq!(
    meta.get(META_TOKENIZER_CONFIG).map(String::as_str),
    Some("{\"eos_token\": \"</s>\"}")
  );
  assert_eq!(meta.get("note").map(String::as_str), Some("hello world"));
}

// ---------------------------------------------------------------------------
// Internally-allocated cache (the structural fix).
//
// `cache_prompt` / `cache_prompt_ids` allocate the per-layer KV cache
// themselves via `make_prompt_cache(cache_config)` — exactly cache_prompt.py:111
// (`cache = make_prompt_cache(model, args.max_kv_size)`). There is NO
// caller-provided cache parameter, so the saved cache is fresh by construction
// and represents *exactly* the requested prompt. The old caller-cache hazard
// (a reused / pre-populated cache persisting a `[stale + new]` prefix and
// leaking a prior request's context) is structurally impossible — there is no
// cache object to reuse.
// ---------------------------------------------------------------------------

/// Two back-to-back `cache_prompt_ids` runs over *different* prompts each
/// persist a cache at that run's own prompt length — the second (shorter)
/// run's saved cache is exactly its own prompt, never `[run1 + run2]`. Because
/// the cache is allocated inside `cache_prompt_ids` (never caller-provided),
/// run 2 cannot inherit run 1's state: a cross-request context leak is
/// structurally impossible.
#[test]
fn cache_prompt_ids_allocates_a_fresh_cache_per_call() {
  let model = MockModel::ramp(8);
  let dir = temp_dir("fresh_per_call");

  // Run 1: a 4-token prompt.
  let out1 = dir.join("run1.safetensors");
  let info1 = cache_prompt_ids(
    &model,
    &[1u32, 3, 4, 5],
    &config(2),
    &out1,
    "mock",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(info1.tokens_processed, 4);
  let (loaded1, _m1) = load_prompt_cache(&out1).unwrap();
  assert!(
    loaded1.iter().all(|c| c.offset() == 4),
    "run 1's saved cache represents exactly its 4-token prompt"
  );

  // Run 2: a *shorter* 2-token prompt. The freshly allocated cache makes the
  // saved cache exactly 2 tokens — not 4 + 2.
  let out2 = dir.join("run2.safetensors");
  let info2 = cache_prompt_ids(
    &model,
    &[6u32, 7],
    &config(2),
    &out2,
    "mock",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(
    info2.tokens_processed, 2,
    "run 2 processes only its own 2-token prompt"
  );
  let (loaded2, _m2) = load_prompt_cache(&out2).unwrap();
  assert!(
    loaded2.iter().all(|c| c.offset() == 2),
    "run 2's saved cache is exactly its 2-token prompt — no leaked prior-request \
     context (the internally-allocated cache is fresh by construction)"
  );
}

/// The high-level `cache_prompt` (tokenizer encode path) likewise allocates
/// its cache internally: two runs over different prompts each save a cache at
/// that run's own encoded-prompt length.
#[test]
fn cache_prompt_allocates_a_fresh_cache_per_call() {
  let dir = temp_dir("fresh_per_call_highlevel");
  let tok = tokenizer(&dir);
  let model = MockModel::ramp(64);

  // Run 1.
  let out1 = dir.join("run1.safetensors");
  let info1 = cache_prompt(
    &model,
    &tok,
    "hello world",
    &config(2),
    &out1,
    "fixture-model",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();
  let (loaded1, _m1) = load_prompt_cache(&out1).unwrap();
  assert!(
    loaded1.iter().all(|c| c.offset() == info1.tokens_processed),
    "run 1's saved cache is exactly its own encoded prompt"
  );

  // Run 2 over a different prompt: a fresh cache, exactly run 2's prompt.
  let out2 = dir.join("run2.safetensors");
  let info2 = cache_prompt(
    &model,
    &tok,
    "the quick brown fox",
    &config(2),
    &out2,
    "fixture-model",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();
  let (loaded2, _m2) = load_prompt_cache(&out2).unwrap();
  assert!(
    loaded2.iter().all(|c| c.offset() == info2.tokens_processed),
    "run 2's saved cache is exactly its own encoded prompt — no leaked context"
  );
}

// ---------------------------------------------------------------------------
// High-level `cache_prompt` (tokenizer encode path).
// ---------------------------------------------------------------------------

/// The high-level `cache_prompt` encodes the prompt via the tokenizer (the
/// fixture has a chat template, so the chat-template branch runs), fills +
/// saves an internally-allocated cache, and reports the processed-token count
/// == the encoded prompt length. The saved cache offset matches that count.
#[test]
fn driver_high_level_encodes_and_fills() {
  let dir = temp_dir("highlevel");
  let tok = tokenizer(&dir);
  let model = MockModel::ramp(64); // vocab >= any fixture id
  let out = dir.join("cache.safetensors");

  let info = cache_prompt(
    &model,
    &tok,
    "hello world",
    &config(2),
    &out,
    "fixture-model",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();

  assert!(
    info.tokens_processed > 0,
    "a non-empty prompt processes >0 tokens"
  );

  // The saved cache's every layer offset equals the processed count.
  let (loaded, meta) = load_prompt_cache(&out).unwrap();
  assert!(loaded.iter().all(|x| x.offset() == info.tokens_processed));
  assert_eq!(
    meta.get(META_MODEL).map(String::as_str),
    Some("fixture-model")
  );
}

/// The high-level `cache_prompt` over an empty string still encodes (the chat
/// template may inject tokens) — but if the encoding is empty it errors and
/// writes nothing (faithful to mlx-lm's empty-prompt `ValueError`). With the
/// fixture's chat template, an empty user message is non-empty after
/// templating, so this asserts the success path produces a consistent count.
#[test]
fn driver_high_level_empty_string_is_consistent() {
  let dir = temp_dir("emptystr");
  let tok = tokenizer(&dir);
  let model = MockModel::ramp(64);
  let out = dir.join("cache.safetensors");

  match cache_prompt(
    &model,
    &tok,
    "",
    &config(1),
    &out,
    "m",
    "{}",
    8,
    &HashMap::new(),
  ) {
    Ok(info) => {
      // Chat template injected tokens ⇒ a non-empty encode; cache filled.
      assert!(info.tokens_processed > 0);
      assert!(out.exists());
      let (loaded, _m) = load_prompt_cache(&out).unwrap();
      assert!(loaded.iter().all(|x| x.offset() == info.tokens_processed));
    }
    Err(_) => {
      // An empty encode ⇒ error, nothing written.
      assert!(!out.exists());
    }
  }
}

// ---------------------------------------------------------------------------
// `continue_final_message` regression (Codex finding).
//
// `cache_prompt.py` encodes the chat-template prompt with
// `add_generation_prompt=False, continue_final_message=True`. For a chat
// template that appends an end-of-turn token after the final user message,
// `continue_final_message=True` must STRIP that terminator so the saved KV
// cache ends exactly at the prompt's last content token (matching mlx-lm); a
// non-continued encode would cache an EXTRA terminator token, diverging the
// offset + later generation.
// ---------------------------------------------------------------------------

/// A `tokenizer_config.json` whose chat template appends `</s>` (an end-of-turn
/// terminator, vocab id 2 in the fixture) after EVERY message's content — the
/// Qwen/ChatML-style shape `continue_final_message` exists to handle.
const EOT_TEMPLATE_CONFIG_JSON: &str = r#"{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "clean_up_tokenization_spaces": false,
  "unk_token": "<unk>",
  "chat_template": "{{ bos_token }}{% for m in messages %}{{ '<|' + m['role'] + '|>' }}{{ m['content'] }}</s>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
}"#;

/// Build a [`Tokenizer`] from the committed `tokenizer.json` fixture plus a
/// caller-supplied `tokenizer_config.json` body — used to install the
/// terminator-appending chat template above.
fn tokenizer_with_config(dir: &std::path::Path, config_json: &str) -> mlxrs::tokenizer::Tokenizer {
  let mut tj = fs::File::create(dir.join("tokenizer.json")).unwrap();
  tj.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(config_json.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(dir, None).unwrap()
}

/// The chat-template encode used by `cache_prompt` (`continue_final_message =
/// true`) drops the trailing end-of-turn token a non-continued encode keeps —
/// so the cached prompt is one-or-more tokens SHORTER and ends exactly at the
/// final message's content (the Codex finding's correctness contract).
#[test]
fn continue_final_message_encode_drops_trailing_terminator() {
  let dir = temp_dir("cfm_encode");
  let tok = tokenizer_with_config(&dir, EOT_TEMPLATE_CONFIG_JSON);
  // One `user` message — exactly what `cache_prompt` builds.
  let messages = serde_json::json!([{ "role": "user", "content": "hello world" }]);

  // Non-continued: the template's trailing `</s>` is rendered + tokenized, so
  // the encoded ids END with the `</s>` terminator id (2).
  let plain = tok
    .apply_chat_template_ids(&messages, None, false, false, None)
    .unwrap();
  assert_eq!(
    plain.last().copied(),
    Some(2),
    "without continue_final_message the encode keeps the trailing </s> (id 2)"
  );

  // Continued (what `cache_prompt` uses): HF's post-render trim strips the
  // trailing `</s>`, so the encoded ids do NOT end with id 2 and are strictly
  // shorter — the cache offset is exactly that many tokens smaller.
  let continued = tok
    .apply_chat_template_ids(&messages, None, false, true, None)
    .unwrap();
  assert_ne!(
    continued.last().copied(),
    Some(2),
    "continue_final_message must strip the trailing </s> terminator"
  );
  assert!(
    continued.len() < plain.len(),
    "the continued encode ({} ids) is shorter than the plain encode ({} ids)",
    continued.len(),
    plain.len(),
  );
  // Exactly the terminator was removed: the continued encode is the plain
  // encode with its trailing `</s>` id(s) gone — it is a strict prefix.
  assert_eq!(
    continued.as_slice(),
    &plain[..continued.len()],
    "the continued encode is the plain encode minus the trailing terminator"
  );
}

/// L7 round-trip with a terminator-appending chat template: driving
/// `cache_prompt` (which encodes with `continue_final_message=true`) saves a
/// cache whose offset == the *continued* encode length (terminator stripped),
/// NOT the longer plain-encode length — and the loaded cache continues like a
/// from-scratch prefill of that same continued prompt.
#[test]
fn cache_prompt_chat_template_uses_continue_final_message_offset() {
  let dir = temp_dir("cfm_roundtrip");
  let tok = tokenizer_with_config(&dir, EOT_TEMPLATE_CONFIG_JSON);
  let model = MockModel::ramp(64); // vocab >= any fixture id
  let out = dir.join("cache.safetensors");

  let messages = serde_json::json!([{ "role": "user", "content": "the quick brown fox" }]);
  let plain = tok
    .apply_chat_template_ids(&messages, None, false, false, None)
    .unwrap();
  let continued = tok
    .apply_chat_template_ids(&messages, None, false, true, None)
    .unwrap();
  assert!(
    continued.len() < plain.len(),
    "sanity: the terminator-appending template makes the continued encode shorter"
  );

  // Drive the high-level `cache_prompt` (its chat-template branch encodes with
  // continue_final_message=true).
  let info = cache_prompt(
    &model,
    &tok,
    "the quick brown fox",
    &config(2),
    &out,
    "fixture-model",
    "{}",
    8,
    &HashMap::new(),
  )
  .unwrap();

  // The processed count + cache offset match the CONTINUED encode length (the
  // terminator-stripped prompt), not the longer plain encode.
  assert_eq!(
    info.tokens_processed,
    continued.len(),
    "cache_prompt must process the continue_final_message encode (terminator stripped)"
  );
  assert_ne!(
    info.tokens_processed,
    plain.len(),
    "cache_prompt must NOT cache the extra terminator token"
  );

  // Loaded cache continues like a from-scratch prefill of the continued
  // prompt: the cache holds the `continued` prefix, so re-feeding its last
  // token reproduces a scratch run's first decode (greedy argmax is
  // position-independent for the ramp MockModel).
  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert!(loaded.iter().all(|x| x.offset() == continued.len()));
  let from_scratch: Vec<u32> = generate_step(
    &model,
    &continued,
    cache(2),
    GenConfig::default().with_max_tokens(3),
  )
  .map(|r| r.unwrap().token)
  .collect();
  let from_cache: Vec<u32> = generate_step(
    &model,
    &[*continued.last().unwrap()],
    loaded,
    GenConfig::default().with_max_tokens(3),
  )
  .map(|r| r.unwrap().token)
  .collect();
  assert_eq!(
    from_scratch, from_cache,
    "loaded cache (continue_final_message prompt) continues like a from-scratch prefill"
  );
}

// ---------------------------------------------------------------------------
// Sliding-window / rotating-cache prefill (Codex finding).
//
// A sliding-window `CacheConfig` makes `cache_prompt` allocate a
// `RotatingKvCache` per layer (via `make_prompt_cache`). The per-chunk barrier
// routes through the `KvCache::materialize` hook so a `RotatingKvCache` whose
// ring buffer over-allocates (`offset < buffer_len`) materializes its genuine
// stored ring buffers, not the offset-length `state()` serialization slices.
// (The barrier-fires-on-the-over-allocated-ring observation is pinned by the
// in-crate `prefill_full` unit test, which can wrap a custom observing cache;
// the integration driver allocates its cache internally, so here the contract
// is "the sliding-window prefill completes + the saved cache round-trips".)
// ---------------------------------------------------------------------------

/// A multi-chunk prefill over a sliding-window config with `prefill_step_size
/// == 1` (each leading chunk is a single token ⇒ the `S == 1` `update_in_place`
/// path that grows the rotating ring): the driver completes and the saved
/// cache loads back at offset `P` as a `RotatingKVCache`.
#[test]
fn driver_sliding_window_prefill_step_one_completes_and_loads() {
  let model = MockModel::ramp(8);
  // P = 9 tokens, all < vocab(8). step = 1 ⇒ leading P-1 = 8 single-token
  // chunks, each an `S == 1` `update_in_place`. Window 8 > the default `keep`
  // (4) `make_prompt_cache` builds the RotatingKvCache with, so the ring
  // genuinely rotates over P=9 (mlx-lm's sliding-window models always have
  // sliding_window >> keep=4 — a window <= keep is a degenerate config).
  let prompt: Vec<u32> = (0..9u32).map(|i| i % 7).collect();
  let dir = temp_dir("sliding_step1");
  let out = dir.join("cache.safetensors");

  let info = cache_prompt_ids(
    &model,
    &prompt,
    &sliding_config(1, 8), // window 8 (> the default keep=4)
    &out,
    "rotating",
    "{}",
    1, // prefill_step_size == 1 ⇒ S==1 leading chunks
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(info.tokens_processed, prompt.len());

  // The saved cache loads back at offset P as a RotatingKVCache.
  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(loaded.len(), 1);
  assert!(loaded.iter().all(|c| c.offset() == prompt.len()));
  assert_eq!(
    loaded[0].reference_class_name(),
    "RotatingKVCache",
    "a sliding-window config persists the cache as a RotatingKVCache"
  );
}

/// The `prefill_step_size == 0` clamp (→ 1) over a sliding-window config: a `0`
/// step must still make progress (clamped to single-token chunks, the same
/// S==1 over-allocating ring path), completing and producing a loadable cache
/// byte-identical to an explicit `step == 1` run.
#[test]
fn driver_sliding_window_prefill_step_zero_clamped_completes() {
  let model = MockModel::ramp(8);
  let prompt: Vec<u32> = (0..7u32).map(|i| i % 5).collect(); // P = 7
  let dir = temp_dir("sliding_step0");
  let out_zero = dir.join("zero.safetensors");
  let out_one = dir.join("one.safetensors");

  // Window 8 > the default `keep` (4) so the ring genuinely rotates over the
  // P=7 prompt (a window <= keep cannot rotate — mlx-lm's models always have
  // sliding_window >> keep=4). step == 0 (clamped to 1 internally).
  let info = cache_prompt_ids(
    &model,
    &prompt,
    &sliding_config(2, 8),
    &out_zero,
    "rotating",
    "{}",
    0, // clamped to 1 (still makes progress)
    &HashMap::new(),
  )
  .unwrap();
  assert_eq!(info.tokens_processed, prompt.len());

  // Explicit step == 1: must yield a byte-identical cache (the 0-clamp is
  // exactly step 1).
  cache_prompt_ids(
    &model,
    &prompt,
    &sliding_config(2, 8),
    &out_one,
    "rotating",
    "{}",
    1,
    &HashMap::new(),
  )
  .unwrap();

  let (loaded_zero, _m0) = load_prompt_cache(&out_zero).unwrap();
  let (loaded_one, _m1) = load_prompt_cache(&out_one).unwrap();
  assert!(loaded_zero.iter().all(|c| c.offset() == prompt.len()));
  assert_eq!(
    cache_signature(&loaded_zero),
    cache_signature(&loaded_one),
    "the prefill_step_size==0 clamp must equal an explicit step==1 prefill"
  );
}

/// A multi-token-chunk (`prefill_step_size > 1`) prefill over a sliding-window
/// config (the `S > 1` `update_concat` path) ALSO completes and loads — the
/// barrier is correct on both rotating update paths, leaving the saved cache
/// at offset `P`.
#[test]
fn driver_sliding_window_multi_token_chunks_complete_and_load() {
  let model = MockModel::ramp(8);
  let prompt: Vec<u32> = (0..13u32).map(|i| i % 6).collect(); // P = 13
  let dir = temp_dir("sliding_multichunk");

  // Window 8 > the default `keep` (4) so the ring rotates over P=13.
  // step = 3 ⇒ leading 12 tokens as [3,3,3,3] (S==3 update_concat chunks).
  let out = dir.join("chunked.safetensors");
  cache_prompt_ids(
    &model,
    &prompt,
    &sliding_config(2, 8),
    &out,
    "rotating",
    "{}",
    3,
    &HashMap::new(),
  )
  .unwrap();

  let (loaded, _meta) = load_prompt_cache(&out).unwrap();
  assert_eq!(loaded.len(), 2);
  assert!(loaded.iter().all(|c| c.offset() == prompt.len()));
}
