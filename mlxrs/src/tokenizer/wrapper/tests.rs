//! Unit tests for the unified [`Tokenizer`] wrapper.
//!
//! Fixtures follow the repo idiom (see `lm/tuner/datasets/tests.rs` and
//! `embeddings/encode/tests.rs`): a minimal `WordLevel` HF tokenizer is built
//! in-memory via the public `tokenizers` API, serialized to a per-test temp
//! directory, and loaded through the feature-combo-agnostic
//! [`Tokenizer::from_path`]. Each test's tokenizer-config branches are driven
//! by a hand-written `tokenizer_config.json` so the expected encode/decode /
//! special-token / chat-override outputs are derived independently of the
//! function under test (closed-form oracles), never by calling that function
//! to produce its own expectation.

use std::{
  path::{Path, PathBuf},
  sync::atomic::{AtomicU64, Ordering},
};

use super::*;

// ───────────────────────── fixtures ─────────────────────────

/// A fresh, unique temp directory per call (process id + monotonic counter),
/// matching the `fresh_dir` idiom in `lm/tuner/datasets/tests.rs`. Leaked
/// intentionally (process-lifetime fixture).
fn fresh_dir(tag: &str) -> PathBuf {
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-tokenizer-wrapper-{tag}-{}-{n}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Build a `WordLevel` HF tokenizer over `vocab` (word → id) with the given
/// `unk` token and an optional `Whitespace` pre-tokenizer, save it to
/// `dir/tokenizer.json`.
///
/// `with_whitespace = false` leaves the pre-tokenizer unset, so the WHOLE
/// input string is one lookup against the vocab (`WordLevel::tokenize` maps
/// the entire pre-token to a single id, or `unk`). This is what makes the
/// `<|channel>thought` token-id oracle exact in
/// [`infer_thinking_channel_branch`].
fn write_wordlevel(dir: &Path, vocab: &[(&str, u32)], unk: &str, with_whitespace: bool) {
  use tokenizers::{
    Tokenizer as HfTokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };
  let map = vocab.iter().map(|(w, i)| ((*w).to_string(), *i)).collect();
  let wl = WordLevel::builder()
    .vocab(map)
    .unk_token(unk.to_string())
    .build()
    .unwrap();
  let mut hf = HfTokenizer::new(wl);
  if with_whitespace {
    hf.with_pre_tokenizer(Some(Whitespace {}));
  }
  hf.save(dir.join("tokenizer.json"), false).unwrap();
}

/// The standard small whitespace vocab shared by the config-driven tests
/// (no thinking markers, so `has_thinking()` is `false`).
const BASIC_VOCAB: &[(&str, u32)] = &[
  ("<unk>", 0),
  ("hello", 1),
  ("world", 2),
  ("<tool_call>", 3),
  ("</tool_call>", 4),
];

// ───────────────────────── cfg_str: object ("content") form ─────────────────────────

/// `cfg_str` line 134 — the `Value::Object(o)` AddedToken-style branch:
/// `bos_token` given as `{"content": "<s>"}` resolves to its `content`.
/// Oracle: the config literally puts `"<s>"` under `content`, so
/// `bos_token()` must read `"<s>"` back (independent of the parser).
#[cfg(feature = "tokenizer-config")]
#[test]
fn cfg_str_reads_object_content_form() {
  let dir = fresh_dir("cfg_str_obj");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({
    // object form for bos, plain-string form for eos — exercise both arms.
    "bos_token": {"content": "<s>", "lstrip": false},
    "eos_token": "</s>",
    // an object WITHOUT a "content" key resolves to None (the
    // `and_then(content)` short-circuit).
    "unk_token": {"not_content": "x"},
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert_eq!(tok.bos_token(), Some("<s>"));
  assert_eq!(tok.eos_token(), Some("</s>"));
  assert_eq!(tok.unk_token(), None);
}

// ───────────────────────── from_path: naive detok when no spm/bpe ─────────────────────────

/// `from_path` line 175 — the `detok_class = DetokenizerClass::Naive` arm,
/// compiled ONLY when `tokenizer-stream` is on while BOTH `tokenizer-spm`
/// and `tokenizer-bpe` are off (otherwise the decoder-parsing arm at 163-170
/// is selected instead). Oracle: with no decoder-class inference available,
/// the class is unconditionally `Naive`.
#[cfg(all(
  feature = "tokenizer-stream",
  not(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))
))]
#[test]
fn from_path_naive_class_without_spm_bpe() {
  let dir = fresh_dir("naive_noinfer");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert_eq!(tok.detokenizer_class(), DetokenizerClass::Naive);
  // The factory then yields the naive variant.
  assert!(tok.detokenizer().is_naive());
}

// ───────────────────────── tool parser: config-selected ─────────────────────────

/// `from_loaded` lines 276-278 (`tool_call_start`/`tool_call_end` from the
/// selected parser) + accessors 659-672, 677-678. A `tool_parser_type`
/// config key selects `json_tools` via `parser_by_name`. Oracle: the
/// `json_tools` marker table entry (generated.rs) is `<tool_call>` /
/// `</tool_call>`, and its `name()` is `"json_tools"`.
#[cfg(feature = "tokenizer-tools")]
#[test]
fn tool_parser_from_config_sets_delimiters_and_accessors() {
  let dir = fresh_dir("tool_parser_cfg");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "tool_parser_type": "json_tools" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  assert!(tok.has_tool_calling());
  assert_eq!(tok.tool_call_start(), Some("<tool_call>"));
  assert_eq!(tok.tool_call_end(), Some("</tool_call>"));
  assert_eq!(tok.tool_parser().map(|p| p.name()), Some("json_tools"));
}

/// Without a tool parser the accessors report "absent" — covers the
/// `None` projection through `tool_call_start`/`tool_call_end`/
/// `has_tool_calling`/`tool_parser` (the `as_deref()`/`is_some()` on `None`).
#[cfg(feature = "tokenizer-tools")]
#[test]
fn tool_parser_absent_accessors_are_none() {
  let dir = fresh_dir("tool_parser_none");
  // No `tool_parser_type` and a chat template that infers no parser.
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "chat_template": "{{ messages }}" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  assert!(!tok.has_tool_calling());
  assert_eq!(tok.tool_call_start(), None);
  assert_eq!(tok.tool_call_end(), None);
  assert!(tok.tool_parser().is_none());
}

// ───────────────────────── parse_tool_call ─────────────────────────

/// `parse_tool_call` lines 683, 688-692 — the configured-parser success
/// path. With `json_tools` selected, a bare JSON object parses to one
/// [`tools::ToolCall`]. Oracle: the input JSON literally names the call
/// `get_time` with empty arguments.
#[cfg(feature = "tokenizer-tools")]
#[test]
fn parse_tool_call_with_configured_parser() {
  let dir = fresh_dir("parse_tc_ok");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "tool_parser_type": "json_tools" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  let calls = tok
    .parse_tool_call(r#"{"name": "get_time", "arguments": {}}"#, None)
    .unwrap();
  assert_eq!(calls.len(), 1);
  assert_eq!(calls[0].name(), "get_time");
  assert_eq!(*calls[0].arguments(), serde_json::json!({}));
}

/// `parse_tool_call` lines 688-691 — the `ok_or_else` error arm when no
/// tool parser is configured. Oracle: a typed `Error::Tokenizer` whose
/// message names the missing parser.
#[cfg(feature = "tokenizer-tools")]
#[test]
fn parse_tool_call_without_parser_errors() {
  let dir = fresh_dir("parse_tc_err");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  // No tool parser configured.
  std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  let err = tok.parse_tool_call("{}", None).unwrap_err();
  match err {
    Error::Tokenizer(m) => assert!(m.contains("no tool parser"), "msg: {m}"),
    other => panic!("expected Error::Tokenizer, got {other:?}"),
  }
}

// ───────────────────────── from_parts (legacy constructor) ─────────────────────────

/// `from_parts` lines 330, 337 — the legacy `(hf, raw, config, detok_class,
/// eos)` constructor that forwards to `from_loaded`. Oracle: the forwarded
/// `detok_class` and `eos_token_ids` are observable verbatim on the result.
/// This is the cfg-gated signature, so the test is gated identically.
#[cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream"))]
#[test]
fn from_parts_forwards_to_from_loaded() {
  // `HfTokenizer` is already in scope via `use super::*` (wrapper.rs's
  // `use tokenizers::Tokenizer as HfTokenizer`).
  use tokenizers::models::wordlevel::WordLevel;
  let map = BASIC_VOCAB
    .iter()
    .map(|(w, i)| ((*w).to_string(), *i))
    .collect();
  let wl = WordLevel::builder()
    .vocab(map)
    .unk_token("<unk>".to_string())
    .build()
    .unwrap();
  let hf = HfTokenizer::new(wl);

  let raw = serde_json::json!({ "decoder": null });
  let config = serde_json::json!({ "eos_token": "</tool_call>" });
  // Caller-supplied eos set REPLACES the config default; first elem is primary.
  let tok = Tokenizer::from_parts(
    hf,
    raw,
    config,
    DetokenizerClass::Naive,
    Some(&[4u32, 3u32]),
  )
  .unwrap();

  assert_eq!(tok.detokenizer_class(), DetokenizerClass::Naive);
  let eos: Vec<u32> = tok.eos_token_ids_iter().collect();
  // BTreeSet sorts → {3,4}; both supplied ids present.
  assert_eq!(eos, vec![3, 4]);
  // Primary EOS = first supplied id (4), used by encode_with(add_eos=true).
  let out = tok
    .encode_with(
      "hello",
      &EncodeOptions::new()
        .with_add_eos(true)
        .with_add_special(false),
    )
    .unwrap();
  assert_eq!(out.ids().last(), Some(&4u32));
}

// ───────────────────────── additional_special_token_ids ─────────────────────────

/// `additional_special_token_ids` line 604 — `additional_special_tokens`
/// present but NOT an array (a bare string) ⇒ empty result. Oracle: the
/// `as_array()` guard returns `None`, so the method returns `Vec::new()`.
#[cfg(feature = "tokenizer-config")]
#[test]
fn additional_special_token_ids_non_array_is_empty() {
  let dir = fresh_dir("addl_non_array");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "additional_special_tokens": "<tool_call>" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(tok.additional_special_token_ids().is_empty());
}

/// `additional_special_token_ids` line 611 — the `_ => None` arm for an
/// array entry that is neither a string nor an object (a bare number is
/// skipped), plus the string + object("content") resolution. Oracle: only
/// the two known vocab tokens (`<tool_call>`=3, `</tool_call>`=4) resolve;
/// `42` is skipped by the match, and `"unknown_tok"` is skipped by the
/// `token_to_id` lookup miss.
#[cfg(feature = "tokenizer-config")]
#[test]
fn additional_special_token_ids_mixed_entries() {
  let dir = fresh_dir("addl_mixed");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({
    "additional_special_tokens": [
      "<tool_call>",                    // string → id 3
      {"content": "</tool_call>"},      // object → id 4
      42,                               // number → `_ => None` (line 611)
      "unknown_tok",                    // string, not in vocab → skipped
      {"no_content": true},             // object w/o content → skipped
    ]
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert_eq!(tok.additional_special_token_ids(), vec![3, 4]);
}

// ───────────────────────── detokenizer(): SPM branch ─────────────────────────

/// `detokenizer` lines 739-741 (the SPM arm) + `detokenizer_class` 773-774.
/// A `decoder` node matching `spm_decoder_target(true)` is inferred as
/// [`DetokenizerClass::Spm`] by `from_path`; the factory then builds the
/// `Spm` variant. Oracle: the decoder JSON is the exact SPM-with-strip
/// shape `infer_detokenizer_class` recognizes (see `stream.rs`), so the
/// class is `Spm` and `detokenizer().is_spm()` holds.
#[cfg(feature = "tokenizer-spm")]
#[test]
fn detokenizer_spm_branch() {
  let dir = fresh_dir("spm_detok");
  // Build a WordLevel tokenizer.json, then splice in the SPM decoder node so
  // `from_path`'s decoder-class inference fires.
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let path = dir.join("tokenizer.json");
  let bytes = std::fs::read(&path).unwrap();
  let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
  // The exact SPM-with-strip decoder `infer_detokenizer_class` matches.
  v["decoder"] = serde_json::json!({
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  });
  std::fs::write(&path, v.to_string()).unwrap();

  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert_eq!(tok.detokenizer_class(), DetokenizerClass::Spm);
  assert!(tok.detokenizer().is_spm());
}

// ───────────────────────── apply_chat_template: override path ─────────────────────────

/// `apply_chat_template` lines 821-833 — the registered-override branch.
/// A `chat_template_type: "deepseek_v32"` config selects the DeepseekV32
/// override, which takes precedence over any jinja template. With no
/// thinking markers in the vocab, `has_thinking()` is `false`, so
/// `enable_thinking` defaults to `false` (chat mode). Oracle: the exact
/// deepseek_v32 chat-mode rendering of a single user message (cross-checked
/// against the closed-form constants in `chat.rs`):
/// `<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>`.
#[cfg(feature = "tokenizer-deepseek-v32")]
#[test]
fn apply_chat_template_uses_registered_override() {
  let dir = fresh_dir("ds_override");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  // A jinja chat_template is ALSO present to prove the override wins.
  let cfg = serde_json::json!({
    "chat_template_type": "deepseek_v32",
    "chat_template": "JINJA-SHOULD-NOT-BE-USED",
  });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(tok.has_chat_template());
  assert!(!tok.has_thinking());

  let messages = serde_json::json!([{ "role": "user", "content": "hello" }]);
  let out = tok
    .apply_chat_template(&messages, None, true, false, None)
    .unwrap();
  assert_eq!(
    out,
    "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
  );
}

/// `apply_chat_template` line 826 — the override path's `messages.as_array()`
/// guard: a non-array `messages` value yields `Error::Tokenizer("messages
/// must be a list")`. Oracle: the typed error + its message.
#[cfg(feature = "tokenizer-deepseek-v32")]
#[test]
fn apply_chat_template_override_rejects_non_list_messages() {
  let dir = fresh_dir("ds_override_err");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "chat_template_type": "deepseek_v32" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  // messages is an object, not a list.
  let messages = serde_json::json!({ "role": "user", "content": "hello" });
  let err = tok
    .apply_chat_template(&messages, None, true, false, None)
    .unwrap_err();
  match err {
    Error::Tokenizer(m) => assert!(m.contains("messages must be a list"), "msg: {m}"),
    other => panic!("expected Error::Tokenizer, got {other:?}"),
  }
}

// ───────────────────────── config() accessor ─────────────────────────

/// `config` lines 883-884 — the raw parsed-config accessor returns the
/// `tokenizer_config.json` that was loaded. Oracle: a sentinel key written
/// into the config must read back through `config()`.
#[cfg(feature = "tokenizer-config")]
#[test]
fn config_accessor_returns_parsed_config() {
  let dir = fresh_dir("config_accessor");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let cfg = serde_json::json!({ "model_max_length": 4096, "eos_token": "</tool_call>" });
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  let c = tok.config();
  assert_eq!(
    c.get("model_max_length").and_then(|v| v.as_u64()),
    Some(4096)
  );
  assert_eq!(
    c.get("eos_token").and_then(|v| v.as_str()),
    Some("</tool_call>")
  );
}

// ───────────────────────── infer_thinking: <|channel> (gpt-oss) branch ─────────────────────────

/// `infer_thinking` lines 933-948 — the `<|channel>` / `<channel|>`
/// gpt-oss-style branch, reached only when neither `<think>`/`</think>` nor
/// `<longcat_think>`/`</longcat_think>` is in the vocab but BOTH `<|channel>`
/// and `<channel|>` are.
///
/// The tokenizer has NO whitespace pre-tokenizer, so `encode(s, false)` is a
/// whole-string vocab lookup. Oracle (closed-form): the marker strings are
/// the hardcoded literals `"<|channel>thought"` / `"<channel|>"`; the start
/// tokens are the whole-string lookup of `"<|channel>thought"` (vocab id 12)
/// and the end tokens the lookup of `"<channel|>"` (vocab id 11).
#[test]
fn infer_thinking_channel_branch() {
  let dir = fresh_dir("channel_think");
  let vocab: &[(&str, u32)] = &[
    ("<unk>", 0),
    ("hello", 1),
    // The condition keys (must both be present):
    ("<|channel>", 10),
    ("<channel|>", 11),
    // The exact strings `infer_thinking` re-encodes, as whole-string lookups:
    ("<|channel>thought", 12),
  ];
  // `with_whitespace = false` ⇒ whole-string lookup (no punctuation split).
  write_wordlevel(dir.as_path(), vocab, "<unk>", false);
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  assert!(tok.has_thinking());
  assert_eq!(tok.think_start(), Some("<|channel>thought"));
  assert_eq!(tok.think_end(), Some("<channel|>"));
  assert_eq!(tok.think_start_tokens(), Some(&[12u32][..]));
  assert_eq!(tok.think_end_tokens(), Some(&[11u32][..]));
}

/// Sanity floor for the no-thinking case used by the other fixtures: the
/// basic vocab has none of the thinking markers, so `infer_thinking` returns
/// the default (covers the `Thinking::default()` fall-through tail).
#[test]
fn infer_thinking_absent_when_no_markers() {
  let dir = fresh_dir("no_think");
  write_wordlevel(dir.as_path(), BASIC_VOCAB, "<unk>", true);
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  assert!(!tok.has_thinking());
  assert_eq!(tok.think_start(), None);
  assert_eq!(tok.think_end(), None);
  assert_eq!(tok.think_start_tokens(), None);
  assert_eq!(tok.think_end_tokens(), None);
}
