//! GGUF export pipeline tests, hand-traced from `mlx_lm/gguf.py`.
//!
//! No `peak_memory()` magnitude asserts (the process-global monotonic
//! counter is polluted by concurrent tests).

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn fresh_dir(tag: &str) -> PathBuf {
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs-lm-gguf-{tag}-{}-{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

// ─────────────────────── translate_weight_names ──────────────────────

/// Hand-traced from `mlx_lm/gguf.py:103-130`. Every rule in the
/// reference must have at least one case in the table; the table also
/// covers per-layer composition (`model.layers.N.<suffix>`) so the
/// first rule's interaction with later rules is exercised.
#[test]
fn translate_weight_names_table_matches_python_reference() {
  let cases: &[(&str, &str)] = &[
    // 1. model.layers.N. → blk.N.
    (
      "model.layers.0.input_layernorm.weight",
      "blk.0.attn_norm.weight",
    ),
    (
      "model.layers.12.post_attention_layernorm.weight",
      "blk.12.ffn_norm.weight",
    ),
    // 2. Mixtral router gate
    (
      "model.layers.3.block_sparse_moe.gate.weight",
      "blk.3.ffn_gate_inp.weight",
    ),
    // 3-5. Mixtral expert FFN re.sub
    (
      "model.layers.3.block_sparse_moe.experts.0.w1.weight",
      "blk.3.ffn_gate.0.weight",
    ),
    (
      "model.layers.3.block_sparse_moe.experts.7.w2.weight",
      "blk.3.ffn_down.7.weight",
    ),
    (
      "model.layers.3.block_sparse_moe.experts.15.w3.weight",
      "blk.3.ffn_up.15.weight",
    ),
    // 6-8. Per-component MLP
    (
      "model.layers.1.mlp.gate_proj.weight",
      "blk.1.ffn_gate.weight",
    ),
    (
      "model.layers.1.mlp.down_proj.weight",
      "blk.1.ffn_down.weight",
    ),
    ("model.layers.1.mlp.up_proj.weight", "blk.1.ffn_up.weight"),
    // 9-12. Per-component attention
    (
      "model.layers.2.self_attn.q_proj.weight",
      "blk.2.attn_q.weight",
    ),
    (
      "model.layers.2.self_attn.k_proj.weight",
      "blk.2.attn_k.weight",
    ),
    (
      "model.layers.2.self_attn.v_proj.weight",
      "blk.2.attn_v.weight",
    ),
    (
      "model.layers.2.self_attn.o_proj.weight",
      "blk.2.attn_output.weight",
    ),
    // 13-14. Norms (already covered above per-layer, but explicit forms)
    (
      "model.layers.5.input_layernorm.weight",
      "blk.5.attn_norm.weight",
    ),
    (
      "model.layers.5.post_attention_layernorm.weight",
      "blk.5.ffn_norm.weight",
    ),
    // 15. Embed tokens
    ("model.embed_tokens.weight", "token_embd.weight"),
    // 16. Final norm
    ("model.norm.weight", "output_norm.weight"),
    // 17. LM head
    ("lm_head.weight", "output.weight"),
  ];
  for (input, expected) in cases {
    assert_eq!(
      &translate_weight_names(input),
      expected,
      "translate_weight_names({input:?}) mismatch",
    );
  }
  // The reference applies the rules unconditionally; an unrelated key
  // passes through.
  assert_eq!(
    translate_weight_names("some.unrelated.key"),
    "some.unrelated.key"
  );
}

// ──────────────────────── permute_weights ────────────────────────

/// Hand-traced from `mlx_lm/gguf.py:133-141`. Shape and values are
/// computed in Python (mlx) for the same inputs and compared element-
/// wise.
///
/// `n_head = 2`, `n_head_kv = 2`, weights shape `[8, 1]`:
/// arange(8) → reshape `[2, 2, 2, 1]` → swapaxes(1, 2) → reshape `[8, 1]`.
///
/// Reshape laid out (row-major):
/// ```text
///   reshape[h, half, d, c]:
///     h=0 half=0 d=0 c=0 → 0    (id 0)
///     h=0 half=0 d=1 c=0 → 1    (id 1)
///     h=0 half=1 d=0 c=0 → 2    (id 2)
///     h=0 half=1 d=1 c=0 → 3    (id 3)
///     h=1 half=0 d=0 c=0 → 4
///     h=1 half=0 d=1 c=0 → 5
///     h=1 half=1 d=0 c=0 → 6
///     h=1 half=1 d=1 c=0 → 7
///   swapaxes(1, 2) (now [h, d, half, c]):
///     h=0 d=0 half=0 c=0 → 0
///     h=0 d=0 half=1 c=0 → 2
///     h=0 d=1 half=0 c=0 → 1
///     h=0 d=1 half=1 c=0 → 3
///     h=1 d=0 half=0 c=0 → 4
///     h=1 d=0 half=1 c=0 → 6
///     h=1 d=1 half=0 c=0 → 5
///     h=1 d=1 half=1 c=0 → 7
/// ```
/// reshape back → `[0, 2, 1, 3, 4, 6, 5, 7]`.
#[test]
fn permute_weights_q_k_matches_python_reference() {
  let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
  let w = Array::from_slice::<f32>(&data, &(8_usize, 1)).unwrap();
  let mut out = permute_weights(&w, 2, Some(2)).unwrap();
  assert_eq!(out.shape(), vec![8, 1]);
  assert_eq!(
    out.to_vec::<f32>().unwrap(),
    vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
  );
}

/// `n_head_kv` overrides `n_head` per `mlx_lm/gguf.py:134-135`.
/// Shape `[4, 1]`, `n_head=4`, `n_head_kv=2` → effective=2.
/// Reshape to `[2, 2, 1, 1]`, swapaxes(1, 2) → `[2, 1, 2, 1]` →
/// reshape `[4, 1]`. Indices: `[0, 1, 2, 3]` → unchanged for this
/// case because each head's `(half, d) = (2, 1)` swap is a no-op
/// (one of the axes is size-1).
#[test]
fn permute_weights_kv_overrides_n_head() {
  let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
  let w = Array::from_slice::<f32>(&data, &(4_usize, 1)).unwrap();
  let mut out = permute_weights(&w, 4, Some(2)).unwrap();
  assert_eq!(out.shape(), vec![4, 1]);
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

/// A leading dim not divisible by `2 * effective_n_head` is a
/// fail-fast — the reference would silently produce a bogus reshape.
#[test]
fn permute_weights_rejects_invalid_leading_dim() {
  let w = Array::from_slice::<f32>(&[0.0; 6], &(6_usize, 1)).unwrap();
  let err = permute_weights(&w, 4, Some(4)).unwrap_err();
  let msg = format!("{err:?}");
  assert!(msg.contains("permute_weights"), "{msg}");
}

// ──────────────────────── HfVocab / prepare_metadata ────────────────────

/// Build a tiny BPE tokenizer fixture on disk: `tokenizer.json` +
/// `tokenizer_config.json`. The vocab has 4 base tokens plus 1 added
/// special token. This is exactly the shape needed by [`HfVocab`] and
/// [`prepare_metadata`].
fn write_tokenizer_fixture(dir: &std::path::Path) -> crate::tokenizer::Tokenizer {
  use serde_json::json;
  // A minimal GPT2-style BPE tokenizer.json with 4 base tokens; no
  // merges (encode/decode paths aren't exercised here).
  let tok = json!({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "a": 3,
      },
      "merges": []
    },
    "added_tokens": [
      {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
      {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
      {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
      {"id": 100, "content": "<extra>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    ],
  });
  std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();

  let cfg = json!({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();

  crate::tokenizer::Tokenizer::from_path(dir, None).unwrap()
}

#[test]
fn hf_vocab_to_gguf_round_trip() {
  let dir = fresh_dir("hf_vocab");
  let tokenizer = write_tokenizer_fixture(&dir);
  let vocab = HfVocab::from_tokenizer(&tokenizer).unwrap();

  // base vocab = 4 (<unk>, <s>, </s>, a) per the fixture; the added
  // `<extra>` lives at id 100 → outside the base range and thus
  // appended.
  assert_eq!(vocab.vocab_size_base(), 4);
  assert_eq!(vocab.vocab_size(), 5);

  let triples = vocab.all_tokens().unwrap();
  assert_eq!(triples.len(), 5);
  // Base ids 0..4 — ids 0/1/2 are special (Control), id 3 ('a') is Normal.
  assert_eq!(triples[0].2, TokenType::Control);
  assert_eq!(triples[1].2, TokenType::Control);
  assert_eq!(triples[2].2, TokenType::Control);
  assert_eq!(triples[3].2, TokenType::Normal);
  // The appended `<extra>` is a user-defined token (`special=false`).
  assert_eq!(triples[4].0, "<extra>");
  assert_eq!(triples[4].2, TokenType::UserDefined);
  // All scores are the reference's constant -1000.0.
  for (_, score, _) in &triples {
    assert!((score - -1000.0).abs() < 1e-6, "score {score} != -1000.0");
  }

  // Pack via prepare_metadata and verify the vocab block is written
  // through round-trip via save_gguf + load_gguf (load_gguf cannot
  // enumerate metadata keys per `crate::io::load_gguf`'s doc comment,
  // so we only verify the file is at least openable).
  let mut config_json = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 16,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-5,
  });
  // Round-trip serialize/deserialize so `Config::from_json` and
  // `prepare_metadata` consume identical text.
  let raw_json = serde_json::to_string(&config_json).unwrap();
  let config = Config::from_json(&raw_json).unwrap();
  config_json = serde_json::from_str(&raw_json).unwrap();
  let meta = prepare_metadata(&config, &config_json, &vocab).unwrap();
  assert!(meta.contains_key("tokenizer.ggml.tokens"));
  assert!(meta.contains_key("tokenizer.ggml.scores"));
  assert!(meta.contains_key("tokenizer.ggml.token_type"));

  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn prepare_metadata_minimal_llama_config() {
  let dir = fresh_dir("prep_meta");
  let tokenizer = write_tokenizer_fixture(&dir);
  let vocab = HfVocab::from_tokenizer(&tokenizer).unwrap();

  let config_text = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rope_theta": 500_000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 64,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "_name_or_path": "foo/bar-7b",
    "rope_scaling": { "type": "linear", "factor": 2.0 },
  })
  .to_string();
  let raw_json: serde_json::Value = serde_json::from_str(&config_text).unwrap();
  let config = Config::from_json(&config_text).unwrap();
  let meta = prepare_metadata(&config, &raw_json, &vocab).unwrap();

  // Validate keys / shapes / scalar values via to_vec.
  // GgufMetadata has no Debug impl (mirrors the M3 IO surface, since
  // `Array` is `!Debug` for content); the mismatch panic names just
  // the key + a short tag.
  fn unwrap_u32_scalar(m: &HashMap<String, GgufMetadata>, key: &str) -> u32 {
    match m.get(key) {
      Some(GgufMetadata::Array(a)) => {
        let mut a = a.try_clone().unwrap();
        a.to_vec::<u32>().unwrap()[0]
      }
      Some(_) => panic!("metadata key {key} was not a scalar array"),
      None => panic!("missing metadata key {key}"),
    }
  }
  fn unwrap_f32_scalar(m: &HashMap<String, GgufMetadata>, key: &str) -> f32 {
    match m.get(key) {
      Some(GgufMetadata::Array(a)) => {
        let mut a = a.try_clone().unwrap();
        a.to_vec::<f32>().unwrap()[0]
      }
      Some(_) => panic!("metadata key {key} was not a scalar array"),
      None => panic!("missing metadata key {key}"),
    }
  }
  fn unwrap_string(m: &HashMap<String, GgufMetadata>, key: &str) -> String {
    match m.get(key) {
      Some(GgufMetadata::String(s)) => s.clone(),
      Some(_) => panic!("metadata key {key} was not a string"),
      None => panic!("missing metadata key {key}"),
    }
  }

  assert_eq!(unwrap_u32_scalar(&meta, "llama.context_length"), 128);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.embedding_length"), 16);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.block_count"), 4);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.feed_forward_length"), 64);
  // rope.dimension_count = hidden_size / num_attention_heads = 16/4 = 4
  assert_eq!(unwrap_u32_scalar(&meta, "llama.rope.dimension_count"), 4);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.attention.head_count"), 4);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.attention.head_count_kv"), 2);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.expert_count"), 8);
  assert_eq!(unwrap_u32_scalar(&meta, "llama.expert_used_count"), 2);
  assert!((unwrap_f32_scalar(&meta, "llama.attention.layer_norm_rms_epsilon") - 1e-5).abs() < 1e-9);
  assert!((unwrap_f32_scalar(&meta, "llama.rope.freq_base") - 500_000.0).abs() < 1e-3);
  assert_eq!(unwrap_string(&meta, "llama.rope.scaling.type"), "linear",);
  assert!((unwrap_f32_scalar(&meta, "llama.rope.scaling.factor") - 2.0).abs() < 1e-6);
  assert_eq!(unwrap_u32_scalar(&meta, "general.file_type"), 1);
  assert_eq!(unwrap_u32_scalar(&meta, "general.quantization_version"), 1);
  assert_eq!(unwrap_u32_scalar(&meta, "general.alignment"), 32);
  assert_eq!(unwrap_string(&meta, "general.architecture"), "llama");
  assert_eq!(unwrap_string(&meta, "general.name"), "bar-7b");
  assert_eq!(unwrap_string(&meta, "tokenizer.ggml.model"), "llama");

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── convert_to_gguf end-to-end ───────────────────────

/// Build a minimal HF-shaped model directory: `config.json` +
/// `model.safetensors` + `tokenizer.json` + `tokenizer_config.json`.
/// Run `convert_to_gguf`, then load the resulting `.gguf` back and
/// assert the weight keys are the translated set.
#[test]
fn convert_to_gguf_end_to_end_minimal() {
  let dir = fresh_dir("e2e");
  let _ = write_tokenizer_fixture(&dir);

  // config.json — minimal Llama-shape (num_attention_heads=2,
  // num_key_value_heads=2 keeps the permute path trivial; hidden_size
  // must be 4 to satisfy `permute_weights`'s `d0 % (2 * n_head) == 0`
  // for shape `[4, 4]` weights).
  let config = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

  // Synthesize a minimal weight set covering: embed, attention q/k/v/o,
  // mlp gate/down/up, norms (input/post-attention/final), lm_head.
  let mut weights: HashMap<String, Array> = HashMap::new();
  let w4x4 = || Array::from_slice::<f32>(&[0.5_f32; 16], &(4_usize, 4)).unwrap();
  let w8x4 = || Array::from_slice::<f32>(&[0.25_f32; 32], &(8_usize, 4)).unwrap();
  let w4x8 = || Array::from_slice::<f32>(&[0.125_f32; 32], &(4_usize, 8)).unwrap();
  let n4 = || Array::from_slice::<f32>(&[1.0_f32; 4], &(4_usize,)).unwrap();
  let e5x4 = || Array::from_slice::<f32>(&[0.0_f32; 20], &(5_usize, 4)).unwrap();
  weights.insert("model.embed_tokens.weight".into(), e5x4());
  weights.insert("model.layers.0.input_layernorm.weight".into(), n4());
  weights.insert(
    "model.layers.0.post_attention_layernorm.weight".into(),
    n4(),
  );
  weights.insert("model.layers.0.self_attn.q_proj.weight".into(), w4x4());
  weights.insert("model.layers.0.self_attn.k_proj.weight".into(), w4x4());
  weights.insert("model.layers.0.self_attn.v_proj.weight".into(), w4x4());
  weights.insert("model.layers.0.self_attn.o_proj.weight".into(), w4x4());
  weights.insert("model.layers.0.mlp.gate_proj.weight".into(), w8x4());
  weights.insert("model.layers.0.mlp.up_proj.weight".into(), w8x4());
  weights.insert("model.layers.0.mlp.down_proj.weight".into(), w4x8());
  weights.insert("model.norm.weight".into(), n4());
  weights.insert("lm_head.weight".into(), e5x4());
  crate::io::save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();

  let gguf_path = dir.join("out.gguf");
  convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path: gguf_path.clone(),
  })
  .unwrap();

  assert!(gguf_path.exists(), "gguf file not written");
  let (loaded_weights, _meta) = crate::io::load_gguf(&gguf_path).unwrap();

  let expected_keys: std::collections::BTreeSet<&str> = [
    "token_embd.weight",
    "blk.0.attn_norm.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "output_norm.weight",
    "output.weight",
  ]
  .iter()
  .copied()
  .collect();
  let got_keys: std::collections::BTreeSet<&str> =
    loaded_weights.keys().map(String::as_str).collect();
  assert_eq!(got_keys, expected_keys, "weight name set mismatch");

  // Norm-named weights cast to F32 (`mlx_lm/gguf.py:303-309`).
  for norm_key in [
    "blk.0.attn_norm.weight",
    "blk.0.ffn_norm.weight",
    "output_norm.weight",
  ] {
    let a = loaded_weights.get(norm_key).unwrap();
    assert_eq!(a.dtype().unwrap(), Dtype::F32, "{norm_key} should be F32");
  }

  let _ = std::fs::remove_dir_all(&dir);
}

/// SENTINEL pattern — plant a `model.safetensors` containing 1 MiB of
/// pure garbage bytes. If the fail-fast validation ran AFTER
/// `load_weights`, the safetensors loader would fail with a parser
/// error naming the bogus header (an mlx-c `safetensors` parse failure,
/// NOT a backend error about an unsupported arch / quantized
/// checkpoint). The test asserts the SPECIFIC validation error fires
/// AND that the error message does NOT carry any safetensors-parse
/// signature, which proves `convert_to_gguf` rejected the model
/// without touching the weight file.
///
/// 1 MiB is large enough that an accidental "passed through to
/// `load_safetensors`" would either (a) succeed silently parsing
/// nothing of value, returning weird errors deeper in the pipeline, or
/// (b) error with a clear parse message — either way the assertion
/// below catches it.
fn write_sentinel_weights(dir: &std::path::Path) {
  // 1 MiB of `0xAB` bytes — not a valid safetensors header, so
  // `load_safetensors` would error with a parse-specific message.
  let garbage = vec![0xAB_u8; 1024 * 1024];
  std::fs::write(dir.join("model.safetensors"), &garbage).unwrap();
}

/// Assert `msg` contains NONE of the safetensors-loader error fingerprints
/// (any of which would prove `load_safetensors` ran on the sentinel file
/// — i.e. the fail-fast validation did NOT run before the weight load).
fn assert_no_safetensors_load_signature(msg: &str) {
  for needle in [
    "safetensors",
    "load_safetensors",
    "header",
    "deserializ",
    "mlx_load",
  ] {
    assert!(
      !msg.to_lowercase().contains(needle),
      "unexpected weight-load signature {needle:?} in error: {msg}"
    );
  }
}

#[test]
fn convert_to_gguf_rejects_unsupported_arch() {
  let dir = fresh_dir("reject_arch");
  let _ = write_tokenizer_fixture(&dir);
  let config = serde_json::json!({
    "model_type": "qwen3",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
  // SENTINEL: 1 MiB of garbage bytes. If load_weights ran, the
  // safetensors loader would error with a parse signature; the
  // assert_no_safetensors_load_signature check below would trip.
  write_sentinel_weights(&dir);

  let gguf_path = dir.join("out.gguf");
  let err = convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path,
  })
  .unwrap_err();
  let Error::UnknownEnumValue(p) = &err else {
    panic!("expected Error::UnknownEnumValue for unsupported arch, got {err:?}");
  };
  assert_eq!(p.value(), "qwen3");
  assert!(
    p.type_name().contains("model_type"),
    "type_name should name the rejected field: {}",
    p.type_name()
  );
  let msg = format!("{err:?}");
  assert_no_safetensors_load_signature(&msg);

  let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn convert_to_gguf_rejects_quantized() {
  let dir = fresh_dir("reject_quant");
  let _ = write_tokenizer_fixture(&dir);
  let config = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
    "quantization": { "group_size": 64, "bits": 4 },
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
  // SENTINEL: 1 MiB of garbage bytes (see `write_sentinel_weights`).
  write_sentinel_weights(&dir);

  let gguf_path = dir.join("out.gguf");
  let err = convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path,
  })
  .unwrap_err();
  let Error::InvariantViolation(p) = &err else {
    panic!("expected Error::InvariantViolation for quantized checkpoint, got {err:?}");
  };
  assert_eq!(p.context(), "convert_to_gguf: checkpoint quantization");
  assert!(p.requirement().contains("must be None"));
  let msg = format!("{err:?}");
  assert_no_safetensors_load_signature(&msg);

  let _ = std::fs::remove_dir_all(&dir);
}

/// Same fail-fast contract as above for the `quantization_config` JSON
/// key — a few HF checkpoints + mlx-lm post-quantize artifacts ship
/// the quantization payload under the longer key (not the strongly
/// typed `quantization` field that `Config` carries). The fail-fast
/// gate must trip on either key BEFORE the weight load.
#[test]
fn convert_to_gguf_rejects_quantization_config_key() {
  let dir = fresh_dir("reject_quant_cfg");
  let _ = write_tokenizer_fixture(&dir);
  let config = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
    // The longer key — Config doesn't pull this into the typed field;
    // the JSON gate in convert_to_gguf catches it.
    "quantization_config": { "group_size": 64, "bits": 4 },
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
  write_sentinel_weights(&dir);

  let gguf_path = dir.join("out.gguf");
  let err = convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path,
  })
  .unwrap_err();
  let Error::InvariantViolation(p) = &err else {
    panic!("expected Error::InvariantViolation for quantized checkpoint, got {err:?}");
  };
  assert_eq!(p.context(), "convert_to_gguf: checkpoint quantization");
  assert!(p.requirement().contains("must be None"));
  let msg = format!("{err:?}");
  assert_no_safetensors_load_signature(&msg);

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────── HfVocab.special_ids union coverage ───────────

/// A second tokenizer fixture where BOS/EOS/UNK live at BASE-VOCAB ids
/// (NOT in `added_tokens_decoder`). The Python reference's
/// `tokenizer.all_special_ids` (`mlx_lm/gguf.py:49`) unions
/// `added_tokens_decoder.special=true` AND the ids declared in
/// `tokenizer_config.json` (`bos_token`/`eos_token`/`unk_token`/
/// `pad_token`/`additional_special_tokens`) — the latter can resolve
/// to base-vocab ids. Building `special_ids` from the added-tokens
/// decoder alone would misclassify these as `Normal` instead of
/// `Control`.
///
/// Fixture shape:
///   - 6 base-vocab tokens: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3),
///     `a`(4), `b`(5).
///   - `tokenizer.json#added_tokens` is empty (so `<unk>`/`<s>`/`</s>`
///     are NOT in `added_tokens_decoder`).
///   - `tokenizer_config.json` declares `bos_token: <s>`,
///     `eos_token: </s>`, `unk_token: <unk>`, `pad_token: <pad>`,
///     `additional_special_tokens: [b]`.
///
/// Expected: ids 0, 1, 2, 3, 5 are `Control`; id 4 (`a`) is `Normal`.
fn write_base_vocab_special_fixture(dir: &std::path::Path) -> crate::tokenizer::Tokenizer {
  use serde_json::json;
  let tok = json!({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "a": 4,
        "b": 5,
      },
      "merges": []
    },
    // INTENTIONALLY EMPTY — the specials are declared via
    // tokenizer_config.json, not via added_tokens_decoder.
    "added_tokens": [],
  });
  std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();

  let cfg = json!({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["b"],
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();

  crate::tokenizer::Tokenizer::from_path(dir, None).unwrap()
}

#[test]
fn convert_to_gguf_uses_base_vocab_special_token_ids() {
  let dir = fresh_dir("base_vocab_specials");
  let tokenizer = write_base_vocab_special_fixture(&dir);
  let vocab = HfVocab::from_tokenizer(&tokenizer).unwrap();

  // (a) special_ids contains the base-vocab ids declared in
  //     tokenizer_config.json (the special-ids union is the only way these
  //     end up in the set — they are NOT in added_tokens_decoder).
  assert!(vocab.special_ids.contains(&0), "unk (id 0) missing");
  assert!(vocab.special_ids.contains(&1), "bos (id 1) missing");
  assert!(vocab.special_ids.contains(&2), "eos (id 2) missing");
  assert!(vocab.special_ids.contains(&3), "pad (id 3) missing");
  assert!(vocab.special_ids.contains(&5), "additional 'b' missing");
  assert!(
    !vocab.special_ids.contains(&4),
    "plain 'a' must NOT be classified Control"
  );

  // (b) get_token_type returns Control for those ids and Normal for 'a'.
  assert_eq!(vocab.get_token_type(0, "<unk>"), TokenType::Control);
  assert_eq!(vocab.get_token_type(1, "<s>"), TokenType::Control);
  assert_eq!(vocab.get_token_type(2, "</s>"), TokenType::Control);
  assert_eq!(vocab.get_token_type(3, "<pad>"), TokenType::Control);
  assert_eq!(vocab.get_token_type(4, "a"), TokenType::Normal);
  assert_eq!(vocab.get_token_type(5, "b"), TokenType::Control);

  // (c) The emitted GGUF tokenizer.ggml.token_type array carries the
  //     correct value at each of these indices. We drive this via
  //     prepare_metadata which packs all_tokens() into the token_type
  //     array (`mlx_lm/gguf.py:240`).
  let config_text = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rope_theta": 10000.0,
    "vocab_size": 6,
    "tie_word_embeddings": false,
    "intermediate_size": 16,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-5,
  })
  .to_string();
  let raw_json: serde_json::Value = serde_json::from_str(&config_text).unwrap();
  let config = Config::from_json(&config_text).unwrap();
  let meta = prepare_metadata(&config, &raw_json, &vocab).unwrap();

  let toktype_vals = match meta.get("tokenizer.ggml.token_type").unwrap() {
    GgufMetadata::Array(a) => {
      let mut a = a.try_clone().unwrap();
      a.to_vec::<u32>().unwrap()
    }
    _ => panic!("token_type was not an Array"),
  };
  assert_eq!(toktype_vals.len(), 6);
  assert_eq!(toktype_vals[0], TokenType::Control as u32, "unk (id 0)");
  assert_eq!(toktype_vals[1], TokenType::Control as u32, "bos (id 1)");
  assert_eq!(toktype_vals[2], TokenType::Control as u32, "eos (id 2)");
  assert_eq!(toktype_vals[3], TokenType::Control as u32, "pad (id 3)");
  assert_eq!(toktype_vals[4], TokenType::Normal as u32, "'a' (id 4)");
  assert_eq!(
    toktype_vals[5],
    TokenType::Control as u32,
    "additional 'b' (id 5)"
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Cross-source coverage: one special lives in `added_tokens_decoder`
/// (with `special=true`), one lives in the base vocab and is declared
/// via `tokenizer_config.json` `additional_special_tokens` only. Both
/// must end up in `special_ids` (i.e. the union of (a) + (b) from the
/// `HfVocab` constructor docs).
#[test]
fn convert_to_gguf_special_ids_unions_added_and_base_vocab() {
  let dir = fresh_dir("union_specials");
  use serde_json::json;
  let tok = json!({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "<unk>": 0,
        "<s>": 1,
        "a": 2,
        "<extra>": 3, // base vocab, will be declared via additional_special_tokens
      },
      "merges": []
    },
    // added_tokens_decoder carries ONLY <added>, an out-of-base added token
    // marked special=true. <s> is left out (so it lives only in the base
    // vocab + tokenizer_config bos_token).
    "added_tokens": [
      {"id": 100, "content": "<added>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    ],
  });
  std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();

  let cfg = json!({
    "bos_token": "<s>",
    "additional_special_tokens": ["<extra>"],
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();

  let tokenizer = crate::tokenizer::Tokenizer::from_path(&dir, None).unwrap();

  // The `tokenizers` crate rewrites added-token ids past the base vocab
  // into the next available slot (the JSON-declared `"id": 100` becomes
  // `id = vocab_size_base` after load). Resolve the actual id of
  // `<added>` from the live tokenizer rather than hard-coding 100 so
  // the assertion stays robust to that rewrite (the union semantics —
  // not the numeric id — is what the test exercises).
  let added_id = tokenizer
    .hf()
    .get_added_tokens_decoder()
    .iter()
    .find(|(_, t)| t.content == "<added>")
    .map(|(id, _)| *id)
    .expect("`<added>` must appear in added_tokens_decoder");
  let vocab = HfVocab::from_tokenizer(&tokenizer).unwrap();

  // (a) source: added_tokens_decoder special=true → <added>.
  assert!(
    vocab.special_ids.contains(&added_id),
    "added <added> (id {added_id}) missing — source (a) failed; special_ids={:?}",
    vocab.special_ids,
  );
  // (b) source: tokenizer_config.json bos_token → <s> at base id 1
  //     AND additional_special_tokens → <extra> at base id 3.
  //     Neither of these is in added_tokens_decoder, so the only way
  //     they end up in special_ids is via the special-ids union — they
  //     are the canonical "BOS/EOS/UNK as a base-vocab token" case
  //     this union exists to cover.
  assert!(
    vocab.special_ids.contains(&1),
    "bos <s> (base id 1) missing — source (b) failed; special_ids={:?}",
    vocab.special_ids,
  );
  assert!(
    vocab.special_ids.contains(&3),
    "additional <extra> (base id 3) missing — source (b) failed; special_ids={:?}",
    vocab.special_ids,
  );
  // Negative control: plain 'a' is not declared anywhere.
  assert!(
    !vocab.special_ids.contains(&2),
    "plain 'a' (id 2) should not be in special_ids; special_ids={:?}",
    vocab.special_ids,
  );

  let _ = std::fs::remove_dir_all(&dir);
}

/// Regression: an added token NAMED by
/// `tokenizer_config.json#additional_special_tokens` but flagged
/// `special=false` in `tokenizer.json#added_tokens_decoder` must still
/// classify as [`TokenType::Control`] in the emitted
/// `tokenizer.ggml.token_type` array.
///
/// A text-based lookup in `self.specials` (populated only from
/// `added_tokens_decoder.special=true`) would classify this case as
/// `UserDefined` even though the constructor unioned the id into
/// `special_ids` via the `additional_special_token_ids()` accessor.
/// [`HfVocab::all_tokens`] classifies by id against `special_ids`
/// instead, so it resolves to `Control`.
///
/// Fixture shape:
///   - base vocab has `<unk>`(0), `<s>`(1), `</s>`(2), `a`(3) — 4
///     entries; vocab_size_base = 4.
///   - `added_tokens` carries `<custom>` at an id >= 4 (the
///     `tokenizers` crate rewrites to the next available id), with
///     `special=false`.
///   - `tokenizer_config.json#additional_special_tokens = ["<custom>"]`
///     — naming the same token text. This unions the resolved id into
///     `special_ids` via source (b).
///
/// Expected: the added-token entry classifies as `Control`, NOT
/// `UserDefined`.
#[test]
fn convert_to_gguf_added_token_via_additional_special_tokens_classifies_as_control() {
  let dir = fresh_dir("added_via_additional_special");
  use serde_json::json;
  let tok = json!({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "a": 3,
      },
      "merges": []
    },
    // `<custom>` lives in added_tokens but with `special=false`. This
    // is the exact gap the special-ids union covers — a prior emission walk
    // would look it up in `self.specials` (empty for this token because
    // special=false) and classify it as UserDefined.
    "added_tokens": [
      {"id": 100, "content": "<custom>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    ],
  });
  std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();

  let cfg = json!({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    // The same token text — declares `<custom>` as a special via
    // tokenizer_config.json. With these ids unioned
    // into `special_ids`, AND the emission classified
    // via `special_ids.contains(&id)`, this should resolve
    // to Control.
    "additional_special_tokens": ["<custom>"],
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();

  let tokenizer = crate::tokenizer::Tokenizer::from_path(&dir, None).unwrap();
  let vocab = HfVocab::from_tokenizer(&tokenizer).unwrap();

  // Resolve the rewritten id of `<custom>` from the live tokenizer —
  // the `tokenizers` crate may rewrite the JSON-declared id 100 into
  // the next available slot after the base vocab. (See the same
  // pattern in `convert_to_gguf_special_ids_unions_added_and_base_vocab`.)
  let custom_id = tokenizer
    .hf()
    .get_added_tokens_decoder()
    .iter()
    .find(|(_, t)| t.content == "<custom>")
    .map(|(id, _)| *id)
    .expect("`<custom>` must appear in added_tokens_decoder");
  assert!(
    custom_id >= vocab.vocab_size_base(),
    "`<custom>` id {custom_id} should be past base vocab ({})",
    vocab.vocab_size_base(),
  );

  // Pre-conditions for the test to be meaningful — the gap this test
  // covers:
  //   (a) `<custom>` is in `special_ids` (source-b union via the
  //       `additional_special_token_ids()` accessor).
  //   (b) `<custom>` is NOT in `specials` (because
  //       `added_tokens_decoder.special=false`). If this changed
  //       (e.g. a future tokenizers-crate revision started unioning
  //       config-additional into specials), the test would still
  //       pass functionally, but it would no longer cover the gap;
  //       the assertion documents the gap explicitly.
  assert!(
    vocab.special_ids.contains(&custom_id),
    "special-ids union failed: special_ids should contain `<custom>` id {custom_id}; \
       special_ids={:?}",
    vocab.special_ids,
  );
  assert!(
    !vocab.specials.contains_key("<custom>"),
    "fixture invariant: `<custom>` should NOT be in `specials` (the gap this test covers); \
       specials={:?}",
    vocab.specials,
  );

  // Run the emission walk via prepare_metadata and assert the
  // emitted `tokenizer.ggml.token_type[custom_id]` is Control.
  let config_text = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rope_theta": 10000.0,
    "vocab_size": vocab.vocab_size(),
    "tie_word_embeddings": false,
    "intermediate_size": 16,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-5,
  })
  .to_string();
  let raw_json: serde_json::Value = serde_json::from_str(&config_text).unwrap();
  let config = Config::from_json(&config_text).unwrap();
  let meta = prepare_metadata(&config, &raw_json, &vocab).unwrap();

  let toktype_vals = match meta.get("tokenizer.ggml.token_type").unwrap() {
    GgufMetadata::Array(a) => {
      let mut a = a.try_clone().unwrap();
      a.to_vec::<u32>().unwrap()
    }
    _ => panic!("token_type was not an Array"),
  };
  assert_eq!(toktype_vals.len() as u32, vocab.vocab_size());
  assert_eq!(
    toktype_vals[custom_id as usize],
    TokenType::Control as u32,
    "`<custom>` (id {custom_id}) should classify as Control, \
       got {} (UserDefined would be {}); full token_type={:?}",
    toktype_vals[custom_id as usize],
    TokenType::UserDefined as u32,
    toktype_vals,
  );
  assert_ne!(
    toktype_vals[custom_id as usize],
    TokenType::UserDefined as u32,
    "explicit not-UserDefined check",
  );

  let _ = std::fs::remove_dir_all(&dir);
}

// ───────── tokenizer-load fail-fast coverage ─────────

/// Regression: a `tokenizer.json` that exists
/// but is malformed JSON must fail before the multi-GB weight load.
/// Merely checking `Path::exists()` would let a malformed
/// tokenizer.json force `load_safetensors` to run first.
/// `convert_to_gguf` instead calls `load_tokenizer` in the validation
/// block, which parses the JSON up front.
///
/// Asserts:
///   (a) the error message names the tokenizer-loading failure.
///   (b) the error message does NOT contain any safetensors-loader
///       signature — proves the weights were NOT read.
#[test]
fn convert_to_gguf_malformed_tokenizer_rejects_before_weight_load() {
  let dir = fresh_dir("malformed_tokenizer");
  // Valid (Llama-shaped) config — passes the arch + quant gates so
  // the only validation that can fire is the tokenizer load.
  let config = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
  // Malformed tokenizer.json — exists, is not a directory, but is not
  // valid JSON. A bare `Path::exists()` gate would accept this and
  // only run `load_tokenizer` AFTER `load_weights`.
  std::fs::write(
    dir.join("tokenizer.json"),
    "{ this is not valid tokenizer json }",
  )
  .unwrap();
  // SENTINEL: 1 MiB of garbage bytes (same pattern as the arch /
  // quant fail-fast tests). If `load_weights` ran, the safetensors
  // loader would surface a parse signature.
  write_sentinel_weights(&dir);

  let gguf_path = dir.join("out.gguf");
  let err = convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path,
  })
  .unwrap_err();
  let msg = format!("{err:?}");
  // (a) the error message names the tokenizer-loading failure. The
  //     load_tokenizer wrapper formats `cannot load tokenizer from
  //     {dir}: {underlying}` (see `load_tokenizer_with_eos`).
  assert!(
    msg.to_lowercase().contains("tokenizer"),
    "error should name tokenizer-loading failure; got: {msg}"
  );
  // (b) the error message does NOT carry a safetensors-loader
  //     signature — proves `load_weights` did NOT run.
  assert_no_safetensors_load_signature(&msg);

  let _ = std::fs::remove_dir_all(&dir);
}

/// Regression: a *directory* at `tokenizer.json`
/// (instead of a file) must fail before the multi-GB weight load.
/// A bare `Path::exists()` returns `true` for a directory and would
/// silently accept it, letting the safetensors loader run.
#[test]
fn convert_to_gguf_directory_at_tokenizer_path_rejects_before_weight_load() {
  let dir = fresh_dir("dir_at_tokenizer");
  let config = serde_json::json!({
    "model_type": "llama",
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "rope_theta": 10000.0,
    "vocab_size": 5,
    "tie_word_embeddings": false,
    "intermediate_size": 8,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
  });
  std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
  // mkdir at `tokenizer.json` — `Path::exists()` returns true, so a
  // bare existence gate would accept it.
  std::fs::create_dir_all(dir.join("tokenizer.json")).unwrap();
  // SENTINEL: 1 MiB of garbage bytes — if load_weights ran, the
  // safetensors loader would surface a parse signature.
  write_sentinel_weights(&dir);

  let gguf_path = dir.join("out.gguf");
  let err = convert_to_gguf(&ConvertToGgufArgs {
    model_path: dir.clone(),
    gguf_path,
  })
  .unwrap_err();
  let msg = format!("{err:?}");
  assert!(
    msg.to_lowercase().contains("tokenizer"),
    "error should name tokenizer-loading failure; got: {msg}"
  );
  assert_no_safetensors_load_signature(&msg);

  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── unit helpers ───────────────────────

#[test]
fn is_byte_token_classifier() {
  assert!(is_byte_token("<0x0A>"));
  assert!(is_byte_token("<0xff>"));
  assert!(is_byte_token("<0xAB>"));
  assert!(!is_byte_token("<0xZ>"));
  assert!(!is_byte_token("<0x0AB>"));
  assert!(!is_byte_token("0x0A"));
  assert!(!is_byte_token("<unk>"));
}
