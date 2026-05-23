//! Integration tests for `mlxrs::lm::structured` — port of
//! `mlx_vlm/structured.py`'s `LLGuidanceLogitsProcessor` +
//! `build_json_schema_logits_processor` (V6 / issue #180).
//!
//! Uses an inline byte-level HF tokenizer fixture (single-token ASCII
//! vocab: every printable byte gets its own id + a special token) so the
//! `toktrie_hf_tokenizers` `ByteLevel` decoder-detection path accepts the
//! tokenizer without needing the full WordLevel/SPM fixture (which the
//! crate rejects: it requires `ByteLevel` or `ByteFallback`).
//!
//! Test scope (per V6 spec):
//!
//! - `build_json_schema_logits_processor_constructs` — sanity-check
//!   construction with a simple object schema.
//! - `json_schema_processor_masks_invalid_first_tokens` — apply the
//!   processor on a fixture logits row; assert tokens that DON'T lead to
//!   a valid JSON start (an alphabetic char before `{`) are masked to
//!   `-inf`, while `{` (the only valid first character for the
//!   `{"type":"object"}` schema) remains finite.
//! - `llguidance_regex_grammar_constructs` —
//!   `GrammarSpec::Regex(r"^[0-9]+$")` constructor succeeds.
//! - `llguidance_lark_grammar_constructs` — minimal Lark grammar
//!   constructor succeeds.
//! - `llguidance_processor_implements_logits_processor_trait` — compile-
//!   check that `into_logits_processor` plugs into the
//!   `make_logits_processors` trait alias.
//! - `llguidance_terminal_grammar_uses_mlxrs_configured_custom_eos_id` —
//!   R2 fix: pin `Tokenizer::eos_token_ids` to a custom id whose string
//!   is OUTSIDE `toktrie_hf_tokenizers`'s hardcoded auto-detect set,
//!   then assert the terminal-grammar EOS-only mask leaves ONLY that
//!   id finite (NOT the upstream-default id 0 or the auto-detected
//!   `</s>`).
//! - `llguidance_terminal_grammar_multi_eos_unmasks_all_configured_ids`
//!   — R2 fix: pin three caller-supplied eos ids (mixed
//!   auto-detect/non-auto-detect), assert all three remain finite in
//!   the EOS-only mask and every non-eos id is `-inf`.
//! - `llguidance_terminal_grammar_padded_eos_id_actually_unmasks_at_runtime`
//!   — R4 fix: padded-only EOS configuration (`[120]` with
//!   `model_vocab_size = Some(128)` over a `bt_vocab = 99` tokenizer)
//!   drives the terminal-grammar EOS-only mask to leave EXACTLY id
//!   120 finite. Under R3's filtered-FFI workaround the EOS-only mask
//!   silently defaulted to id 0 (the upstream auto-detect fallback)
//!   instead.
//! - `llguidance_terminal_grammar_mixed_in_range_plus_padded_eos` —
//!   R4 fix: mixed in-range + padded EOS configuration
//!   (`[CUSTOM_EOS_ID, 120]`) registers BOTH ids in the terminal-grammar
//!   EOS-only mask (under R3 the padded id was filtered out).

#![cfg(all(feature = "lm", feature = "llguidance"))]

use std::{fs, io::Write, path::PathBuf, process};

use mlxrs::{
  Array,
  lm::{generate::LogitsProcessor, structured},
};
use serde_json::json;

/// A minimal byte-level HF tokenizer (vocab = printable ASCII + a few
/// specials) accepted by `toktrie_hf_tokenizers::ByteTokenizer`:
/// `ByteLevel` pre-tokenizer + `ByteLevel` decoder (the two decoders the
/// crate's `check_decoder` recognizes), one-to-one byte→id model.
///
/// The vocab covers the 95 printable ASCII bytes (`0x20..=0x7E`), each
/// mapped via the `tokenizers::ByteLevel` byte→unicode char convention,
/// plus three special tokens (`<unk>`, `<s>`, `</s>`). All printable
/// bytes appear as their byte-level glyph (`tokenizers` automatically
/// maps non-printable + space using its standard byte→char table —
/// `Ġ` for `0x20` etc.).
fn build_byte_level_tokenizer_json() -> String {
  // The `tokenizers::ByteLevel` byte→unicode map: bytes `0x21..=0x7E`
  // map to themselves, byte `0x20` (space) maps to `Ġ` (0x120). Build
  // a minimal vocab covering exactly those + the 3 special tokens.
  let mut vocab_entries: Vec<String> = Vec::new();
  // Special tokens (ids 0..3); the byte-level adapter recognizes the
  // `<...>`-bracketed form as specials.
  vocab_entries.push("\"<unk>\": 0".to_string());
  vocab_entries.push("\"<s>\": 1".to_string());
  vocab_entries.push("\"</s>\": 2".to_string());
  // Printable ASCII via ByteLevel byte→char map.
  let mut next_id: u32 = 3;
  // Build the byte→char table the same way `tokenizers::ByteLevel`
  // does — `is_self_mapped` for `!..~` + `00A1..00AC` + `00AE..00FF`,
  // remap others starting at U+0100.
  let mut k: u32 = 0x100;
  let mut char_map: Vec<char> = Vec::with_capacity(256);
  for byte in 0..=255u8 {
    let c = byte as char;
    let mapped = match c {
      '!'..='~' => c,
      '\u{00A1}'..='\u{00AC}' => c,
      '\u{00AE}'..='\u{00FF}' => c,
      _ => {
        let m = char::from_u32(k).unwrap();
        k += 1;
        m
      }
    };
    char_map.push(mapped);
  }
  for byte in 0x20u8..=0x7Eu8 {
    let glyph = char_map[byte as usize];
    // JSON-escape the glyph as a Rust char literal.
    let escaped = match glyph {
      '"' => "\\\"".to_string(),
      '\\' => "\\\\".to_string(),
      c => c.to_string(),
    };
    vocab_entries.push(format!("\"{}\": {}", escaped, next_id));
    next_id += 1;
  }

  format!(
    r#"{{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
        {{"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}},
        {{"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}
      ],
      "normalizer": null,
      "pre_tokenizer": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "post_processor": null,
      "decoder": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "model": {{
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": {{
          {}
        }},
        "merges": []
      }}
    }}"#,
    vocab_entries.join(",\n      ")
  )
}

const TOKENIZER_CONFIG_JSON: &str = r#"{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "model_max_length": 2048
}"#;

fn temp_dir(name: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_lm_structured_{}_{}", process::id(), name));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

fn fixture_tokenizer(name: &str) -> mlxrs::tokenizer::Tokenizer {
  let dir = temp_dir(name);
  let tj_path = dir.join("tokenizer.json");
  let mut tj = fs::File::create(&tj_path).unwrap();
  tj.write_all(build_byte_level_tokenizer_json().as_bytes())
    .unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(&dir, None)
    .unwrap_or_else(|e| panic!("fixture tokenizer load failed: {e}"))
}

/// Look up the id for a single-byte token in our fixture vocab. The
/// printable-ASCII region starts at id 3.
fn id_for_byte(byte: u8) -> u32 {
  assert!(
    (0x20..=0x7E).contains(&byte),
    "byte {byte:#x} not in fixture vocab"
  );
  3 + (byte - 0x20) as u32
}

#[test]
fn build_json_schema_logits_processor_constructs() {
  let tok = fixture_tokenizer("build_json_schema_constructs");
  let schema = json!({
    "type": "object",
    "properties": {
      "name": { "type": "string" }
    }
  });
  let _proc = structured::build_json_schema_logits_processor(schema, &tok, None)
    .expect("processor construction should succeed for a simple schema");
}

#[test]
fn json_schema_processor_masks_invalid_first_tokens() {
  let tok = fixture_tokenizer("masks_invalid_first");
  // The simplest schema: any JSON object — only `{` (and optional
  // leading whitespace per the JSON grammar) can be the first token.
  let schema = json!({ "type": "object" });
  let proc = structured::build_json_schema_logits_processor(schema, &tok, None)
    .expect("processor construction should succeed");

  // Build a `[1, V]` logits row of zeros; the processor should mask
  // every token that can't start a valid JSON object to `-inf`.
  // Use the actual vocab size from the matcher's mask (after the
  // first apply call returns).
  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  let mut out = proc.apply(&[], &logits).expect("apply should succeed");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  // Token for `{` must be finite (allowed as the JSON object start).
  let open_brace = id_for_byte(b'{') as usize;
  assert!(
    out_v[open_brace].is_finite(),
    "`{{` token (id {open_brace}) must remain finite, got {}",
    out_v[open_brace]
  );

  // Token for `a` must be `-inf` (an alphabetic char cannot start a
  // JSON object — JSON grammar requires `{` or whitespace).
  let a = id_for_byte(b'a') as usize;
  assert!(
    out_v[a].is_infinite() && out_v[a] < 0.0,
    "`a` token (id {a}) must be masked to -inf, got {}",
    out_v[a]
  );

  // Token for `}` must also be masked (it can't be the FIRST character
  // of an object, only the LAST).
  let close_brace = id_for_byte(b'}') as usize;
  assert!(
    out_v[close_brace].is_infinite() && out_v[close_brace] < 0.0,
    "`}}` token (id {close_brace}) must be masked to -inf, got {}",
    out_v[close_brace]
  );
}

#[test]
fn llguidance_regex_grammar_constructs() {
  let tok = fixture_tokenizer("regex_constructs");
  let grammar = structured::GrammarSpec::Regex(r"[0-9]+".to_string());
  let _proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("regex grammar processor construction should succeed");
}

#[test]
fn llguidance_lark_grammar_constructs() {
  let tok = fixture_tokenizer("lark_constructs");
  // A minimal Lark grammar: a single string of digits.
  let lark = r#"start: DIGITS
DIGITS: /[0-9]+/
"#;
  let grammar = structured::GrammarSpec::Lark(lark.to_string());
  let _proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("lark grammar processor construction should succeed");
}

#[test]
fn llguidance_processor_implements_logits_processor_trait() {
  // Compile-time check: `into_logits_processor` returns the
  // `make_logits_processors` trait alias `Box<dyn Fn(&[u32], &Array)
  // -> Result<Array>>`; the binding's type pin enforces it.
  let tok = fixture_tokenizer("plug_into_chain");
  let proc = structured::build_json_schema_logits_processor(json!({"type": "object"}), &tok, None)
    .expect("processor construction should succeed");

  let boxed: LogitsProcessor = proc.into_logits_processor();
  // Exercise the boxed closure once to confirm the wiring round-trips.
  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();
  let _out = boxed(&[], &logits).expect("boxed closure call should succeed");
}

// ── Finding 1 (R1): terminal-grammar EOS-only mask ────────────────────
//
// `Matcher::compute_mask` errors out once the grammar reaches a stopped
// state (e.g. a `Regex("a")` grammar after the single `a` has been
// consumed). The fix uses `compute_mask_or_eos`, which returns an EOS-
// only mask when stopped. These two tests pin that behaviour: after
// consuming the only token the grammar accepts, the next mask must
// allow ONLY the tokenizer's eos id (every other position `-inf`).

/// Index of the tokenizer's EOS in the byte-level fixture vocab.
///
/// `TOKENIZER_CONFIG_JSON` sets `eos_token` = `</s>`; the added-tokens
/// section in `build_byte_level_tokenizer_json` assigns id `2` to
/// `</s>`. Codify this once so the two terminal-grammar tests can
/// assert against it.
const FIXTURE_EOS_ID: usize = 2;

#[test]
fn llguidance_terminal_regex_grammar_returns_eos_only_mask_after_consume() {
  let tok = fixture_tokenizer("terminal_regex_eos_only");
  // A regex grammar that accepts exactly the single byte `a`. The
  // matcher is in `StopReason::NotStopped` initially (one token may
  // still be sampled), then after consuming the `a` token it
  // transitions to a stopped state.
  let grammar = structured::GrammarSpec::Regex("a".to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("terminal regex processor construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  // Step 1: the first `apply` call returns the initial mask (the `a`
  // token is the only valid sample). The `is_first_token` flag is set,
  // so no history token is consumed yet.
  let mut first = proc
    .apply(&[], &logits)
    .expect("first apply should succeed (initial mask, not yet stopped)");
  let first_v = first.to_vec::<f32>().unwrap();
  let a_id = id_for_byte(b'a') as usize;
  assert!(
    first_v[a_id].is_finite(),
    "first-step `a` token (id {a_id}) must remain finite, got {}",
    first_v[a_id]
  );

  // Step 2: simulate consuming the `a` token. The matcher transitions
  // to `StopReason::EosTriggered` (the regex is now satisfied) — pre-
  // fix `compute_mask` would surface an `Err`; the fix routes through
  // `compute_mask_or_eos`, which auto-returns an EOS-only mask. The
  // returned logits must have EVERY position `-inf` except `FIXTURE_EOS_ID`.
  let a_token_id = a_id as u32;
  let mut out = proc
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed and return EOS-only mask");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  assert!(
    out_v[FIXTURE_EOS_ID].is_finite(),
    "EOS-only mask: eos token (id {FIXTURE_EOS_ID}) must remain finite, got {}",
    out_v[FIXTURE_EOS_ID]
  );
  // Every other token (including the previously-allowed `a`) must be
  // masked to `-inf`.
  for (i, v) in out_v.iter().enumerate() {
    if i == FIXTURE_EOS_ID {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "EOS-only mask: non-eos token id {i} must be -inf, got {v}"
    );
  }
}

#[test]
fn llguidance_terminal_lark_grammar_returns_eos_only_after_close() {
  let tok = fixture_tokenizer("terminal_lark_eos_only");
  // A finite Lark grammar that accepts exactly the single literal
  // `x`. After consuming the `x` token the grammar is satisfied and
  // the matcher transitions to a stopped state.
  let lark = r#"start: "x"
"#;
  let grammar = structured::GrammarSpec::Lark(lark.to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("terminal lark processor construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  // Step 1: prime the matcher (first call doesn't consume history).
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed (initial mask, not yet stopped)");

  // Step 2: consume the `x` token; the lark grammar is now finished.
  let x_token_id = id_for_byte(b'x');
  let mut out = proc
    .apply(&[x_token_id], &logits)
    .expect("post-close apply should succeed and return EOS-only mask");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  assert!(
    out_v[FIXTURE_EOS_ID].is_finite(),
    "EOS-only lark mask: eos (id {FIXTURE_EOS_ID}) must remain finite, got {}",
    out_v[FIXTURE_EOS_ID]
  );
  for (i, v) in out_v.iter().enumerate() {
    if i == FIXTURE_EOS_ID {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "EOS-only lark mask: non-eos token id {i} must be -inf, got {v}"
    );
  }
}

// ── Finding 2 (R1): padded model-vocab support ────────────────────────
//
// `tok_env_from_tokenizer` used to call `into_tok_env(None)`, so the
// matcher mask was sized to `tokenizer.get_vocab_size(true)`. Models
// with a padded LM head (logits last-dim > tokenizer vocab) would hit a
// shape-mismatch in `apply`. The fix threads a `model_vocab_size:
// Option<usize>` through to `into_tok_env`, padding the toktrie with
// placeholder special tokens up to the model's actual vocab width. The
// placeholder ids have no real byte sequence, so the grammar engine
// never allows them — they must always be `-inf` in the returned mask.

// ── Finding 1 (R2): mlxrs-configured EOS ids synced into ByteTokenizer ─
//
// `tok_env_from_tokenizer` used to build the `ByteTokenizer` via
// `from_json_bytes` + `into_tok_env` WITHOUT syncing
// `Tokenizer::eos_token_ids()` — upstream `from_tokenizer` only auto-
// detects a small hardcoded set of EOS strings (`</s>`, `<|endoftext|>`,
// `<|end_of_text|>`, DeepSeek's `<｜end▁of▁sentence｜>`, `<eos>`) and
// silently defaults `tok_eos` to id `0` for everything else. The previous
// terminal-grammar test happened to use a fixture with `</s>` at id 2,
// masking the bug. The fix calls
// [`ByteTokenizer::set_eos_tokens`](https://github.com/microsoft/llguidance/blob/main/toktrie_hf_tokenizers/src/lib.rs#L271-L282)
// (toktrie_hf_tokenizers/src/lib.rs:271-282) right after the
// ByteTokenizer is built and BEFORE `into_tok_env`, so the
// `tok_trie().eos_token_set()` returned by `compute_mask_or_eos` reflects
// the model's ACTUAL stop ids.
//
// Test 1: a fixture whose EOS string is NOT in upstream's hardcoded set
// (`<|im_end|>` — upstream classifies it as `tok_end_of_turn`, NOT
// `tok_eos`). Force `Tokenizer::from_path` to pin `eos_token_ids =
// [<im_end_id>]`; assert the terminal-grammar EOS-only mask leaves
// EXACTLY that id finite (NOT id 0, NOT id 2 `</s>`).
//
// Test 2: a fixture with MULTIPLE caller-supplied eos ids; assert ALL of
// them remain finite in the EOS-only mask, and every non-eos id is
// `-inf`.

/// Build a byte-level tokenizer JSON fixture with `extra_added` placed
/// after the base 3 specials + printable ASCII region. Each entry is
/// (`(id, content)`); content must be a `<...>`-bracketed special-token
/// string so `toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer`
/// classifies it as a special (lib.rs:181 `info.content.starts_with("<")
/// && info.content.ends_with(">")`). Ids MUST be > 97 (the existing
/// fixture's last printable-ASCII id is `3 + 95 - 1 = 97`).
fn build_byte_level_tokenizer_json_with_extras(extra_added: &[(u32, &str)]) -> String {
  // Reuse the base vocab body (specials + 95 printable ASCII tokens at
  // ids 3..=97), then splice extras into the `added_tokens` array AND
  // the BPE `vocab` map.
  let mut vocab_entries: Vec<String> = Vec::new();
  vocab_entries.push("\"<unk>\": 0".to_string());
  vocab_entries.push("\"<s>\": 1".to_string());
  vocab_entries.push("\"</s>\": 2".to_string());
  let mut next_id: u32 = 3;
  let mut k: u32 = 0x100;
  let mut char_map: Vec<char> = Vec::with_capacity(256);
  for byte in 0..=255u8 {
    let c = byte as char;
    let mapped = match c {
      '!'..='~' => c,
      '\u{00A1}'..='\u{00AC}' => c,
      '\u{00AE}'..='\u{00FF}' => c,
      _ => {
        let m = char::from_u32(k).unwrap();
        k += 1;
        m
      }
    };
    char_map.push(mapped);
  }
  for byte in 0x20u8..=0x7Eu8 {
    let glyph = char_map[byte as usize];
    let escaped = match glyph {
      '"' => "\\\"".to_string(),
      '\\' => "\\\\".to_string(),
      c => c.to_string(),
    };
    vocab_entries.push(format!("\"{}\": {}", escaped, next_id));
    next_id += 1;
  }
  // Splice extras (JSON-escape the content; brackets `<|...|>` are
  // JSON-safe as-is).
  let mut added_entries: Vec<String> = vec![
    "{\"id\": 0, \"content\": \"<unk>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}".to_string(),
    "{\"id\": 1, \"content\": \"<s>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}".to_string(),
    "{\"id\": 2, \"content\": \"</s>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}".to_string(),
  ];
  for &(id, content) in extra_added {
    assert!(
      id >= next_id,
      "extra-added id {id} must be > base vocab top id {}",
      next_id - 1
    );
    vocab_entries.push(format!("\"{}\": {}", content, id));
    added_entries.push(format!(
      "{{\"id\": {}, \"content\": \"{}\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}}",
      id, content
    ));
  }

  format!(
    r#"{{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {}
      ],
      "normalizer": null,
      "pre_tokenizer": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "post_processor": null,
      "decoder": {{
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true
      }},
      "model": {{
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": {{
          {}
        }},
        "merges": []
      }}
    }}"#,
    added_entries.join(",\n        "),
    vocab_entries.join(",\n          ")
  )
}

/// Same as [`fixture_tokenizer`] but installs `extra_added` specials AND
/// pins `eos_token_ids` to `eos_override` (replacing the `eos_token` =
/// `</s>` default from `TOKENIZER_CONFIG_JSON`).
fn fixture_tokenizer_with_eos_override(
  name: &str,
  extra_added: &[(u32, &str)],
  eos_override: &[u32],
) -> mlxrs::tokenizer::Tokenizer {
  let dir = temp_dir(name);
  let tj_path = dir.join("tokenizer.json");
  let mut tj = fs::File::create(&tj_path).unwrap();
  tj.write_all(build_byte_level_tokenizer_json_with_extras(extra_added).as_bytes())
    .unwrap();
  let mut tc = fs::File::create(dir.join("tokenizer_config.json")).unwrap();
  tc.write_all(TOKENIZER_CONFIG_JSON.as_bytes()).unwrap();
  mlxrs::tokenizer::Tokenizer::from_path(&dir, Some(eos_override))
    .unwrap_or_else(|e| panic!("fixture tokenizer load failed: {e}"))
}

/// Custom EOS id placed after the printable-ASCII region. Id 98 is the
/// next free slot (base ids 0..=2 are specials, 3..=97 are printables).
const CUSTOM_EOS_ID: u32 = 98;

#[test]
fn llguidance_terminal_grammar_uses_mlxrs_configured_custom_eos_id() {
  // `<|im_end|>` is NOT in `toktrie_hf_tokenizers`'s hardcoded `tok_eos`
  // detection list (it's classified as `tok_end_of_turn` —
  // `toktrie_hf_tokenizers/src/lib.rs:189`). Pre-fix, the resulting
  // `tok_eos` would default to `0`, and a terminal grammar's EOS-only
  // mask would leave ONLY id 0 finite. Post-fix, the sync via
  // `set_eos_tokens` carries the mlxrs-configured override
  // (`eos_token_ids = [CUSTOM_EOS_ID]`) into the toktrie, and only that
  // id is finite.
  let tok = fixture_tokenizer_with_eos_override(
    "terminal_custom_eos",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[CUSTOM_EOS_ID],
  );
  // Sanity: the mlxrs wrapper now reports exactly that one eos id.
  let set = tok.eos_token_ids();
  assert_eq!(
    set.iter().copied().collect::<Vec<u32>>(),
    vec![CUSTOM_EOS_ID],
    "mlxrs Tokenizer::eos_token_ids() must reflect the from_path override"
  );

  // Terminal regex grammar `a`: after consuming `a` the matcher is
  // stopped and `compute_mask_or_eos` returns `tok_trie().eos_token_set()`.
  let grammar = structured::GrammarSpec::Regex("a".to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("processor construction with custom eos override should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  // Prime, then drive to terminal state by consuming `a`.
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let a_token_id = id_for_byte(b'a');
  let mut out = proc
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  // Post-fix assertion: ONLY the custom eos id is finite.
  assert!(
    out_v[CUSTOM_EOS_ID as usize].is_finite(),
    "EOS-only mask: custom eos id {CUSTOM_EOS_ID} must remain finite, got {}",
    out_v[CUSTOM_EOS_ID as usize]
  );
  // Pre-fix this would be finite (upstream defaulted `tok_eos` to 0);
  // post-fix it must be `-inf`.
  assert!(
    out_v[0].is_infinite() && out_v[0] < 0.0,
    "EOS-only mask: id 0 (upstream's pre-fix default `tok_eos`) must be -inf, got {}",
    out_v[0]
  );
  // The previous hardcoded-default `</s>` id must NOT be unmasked
  // (proving the override REPLACES — not unions with — upstream's
  // auto-detection).
  assert!(
    out_v[FIXTURE_EOS_ID].is_infinite() && out_v[FIXTURE_EOS_ID] < 0.0,
    "EOS-only mask: hardcoded `</s>` id {FIXTURE_EOS_ID} must be -inf (override replaces), got {}",
    out_v[FIXTURE_EOS_ID]
  );
  // Every other id is `-inf`.
  for (i, v) in out_v.iter().enumerate() {
    if i as u32 == CUSTOM_EOS_ID {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }
}

#[test]
fn llguidance_terminal_grammar_multi_eos_unmasks_all_configured_ids() {
  // Multiple caller-supplied eos ids — the sync must register ALL of
  // them in the toktrie's EOS set, so `compute_mask_or_eos` leaves every
  // configured id finite (and only those). Use ids 1, 2, and 98:
  //   - id 1 = `<s>` (in the base fixture; NOT auto-eos upstream)
  //   - id 2 = `</s>` (in the base fixture; IS auto-eos upstream)
  //   - id 98 = `<|im_end|>` (added; NOT auto-eos upstream).
  // The mixed selection proves the path is "use the caller-supplied
  // set" — not "union" or "fall back to auto-detection".
  let tok = fixture_tokenizer_with_eos_override(
    "terminal_multi_eos",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[1, 2, CUSTOM_EOS_ID],
  );
  let set = tok.eos_token_ids();
  assert_eq!(
    set.iter().copied().collect::<Vec<u32>>(),
    vec![1, 2, CUSTOM_EOS_ID],
    "mlxrs Tokenizer::eos_token_ids() must hold all three caller-supplied ids"
  );

  let grammar = structured::GrammarSpec::Regex("a".to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None)
    .expect("processor construction with multi-eos override should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let a_token_id = id_for_byte(b'a');
  let mut out = proc
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  // All three configured eos ids must remain finite.
  for &eos in &[1u32, 2u32, CUSTOM_EOS_ID] {
    assert!(
      out_v[eos as usize].is_finite(),
      "EOS-only mask: configured eos id {eos} must remain finite, got {}",
      out_v[eos as usize]
    );
  }
  // Every other id (including id 0 = upstream's pre-fix default eos) is `-inf`.
  for (i, v) in out_v.iter().enumerate() {
    let id = i as u32;
    if id == 1 || id == 2 || id == CUSTOM_EOS_ID {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }
}

#[test]
fn llguidance_processor_accepts_padded_model_vocab() {
  let tok = fixture_tokenizer("padded_model_vocab");
  // Logits last-dim larger than the tokenizer vocab — simulate a
  // padded LM head (e.g. 32064 vs 32000 for Llama-style models).
  let tok_vocab = tok.hf().get_vocab_size(true);
  let model_vocab = tok_vocab + 8;

  // Pass `Some(model_vocab)` so the toktrie is padded to match the
  // logits' last-axis width.
  let proc = structured::build_json_schema_logits_processor(
    json!({ "type": "object" }),
    &tok,
    Some(model_vocab),
  )
  .expect("processor construction should succeed with padded model vocab");

  let zeros = vec![0.0f32; model_vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, model_vocab)).unwrap();

  // Pre-fix this would error out with "matcher mask length {tok_vocab}
  // < logits vocab {model_vocab}".
  let mut out = proc
    .apply(&[], &logits)
    .expect("apply should succeed when model vocab is padded via Some(n)");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), model_vocab);

  // The original tokenizer-vocab `{` token is allowed (JSON object
  // start), so we still have a finite anchor.
  let open_brace = id_for_byte(b'{') as usize;
  assert!(
    out_v[open_brace].is_finite(),
    "padded-vocab mask: `{{` (id {open_brace}) must remain finite, got {}",
    out_v[open_brace]
  );

  // Every padded-placeholder position (ids `tok_vocab..model_vocab`)
  // must be `-inf` — the grammar engine never allows them because
  // they have no real byte sequence.
  for (i, v) in out_v.iter().enumerate().take(model_vocab).skip(tok_vocab) {
    assert!(
      v.is_infinite() && *v < 0.0,
      "padded placeholder id {i} must be -inf, got {v}"
    );
  }
}

// ── Finding 1 (R3): out-of-range configured EOS ids → recoverable Err ──
//
// Upstream `ByteTokenizer::set_eos_tokens`
// (`toktrie_hf_tokenizers/src/lib.rs:271-282`) asserts
// `tok < self.info.vocab_size` for every supplied id and PANICS
// otherwise. The mlxrs `Tokenizer::eos_token_ids()` is populated from
// caller-supplied and/or `tokenizer_config.json`-derived ids without
// any vocab-size check (existing fixtures accept ids well past the
// tokenizer's actual vocab — see e.g. id 4242), so an out-of-range
// configured id used to take down the calling thread on the FFI side.
//
// R3 fix: validate every configured id against
// `model_vocab_size.unwrap_or(bt.tokrx_info().vocab_size as usize)`
// BEFORE crossing the FFI boundary, returning
// `Error::ShapeMismatch` with the offending id + bound. Padded LM
// heads (`Some(n)` > tokenizer vocab) are explicitly accepted: an
// EOS id inside `[tokenizer_vocab, model_vocab)` is legitimate.

#[test]
fn llguidance_terminal_grammar_rejects_out_of_range_eos_id_without_panic() {
  // Pin a single eos id far beyond any reasonable vocab. The fixture's
  // total vocab is at most 99 (3 base specials + 95 printable ASCII +
  // 1 added special). With `model_vocab_size = None`, the validation
  // bound is the tokenizer's own vocab, so id 4242 is out of range
  // and must surface as `Error::ShapeMismatch` — NOT a panic.
  let tok = fixture_tokenizer_with_eos_override(
    "out_of_range_eos",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[4242],
  );
  let bound = tok.hf().get_vocab_size(true);

  let grammar = structured::GrammarSpec::Regex("a".to_string());
  // Test framework catches panics as failures, so a `Result::Err`
  // return without a panic is sufficient evidence the constructor
  // converted the upstream `assert!` into a recoverable Error.
  // (`LLGuidanceLogitsProcessor` doesn't implement `Debug` — matcher
  // state is intentionally opaque — so the `Ok` arm has to panic
  // manually rather than via `expect_err`.)
  let result = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, None);
  match result {
    Err(mlxrs::Error::ShapeMismatch { message }) => {
      assert!(
        message.contains("4242"),
        "error message must name the offending id 4242, got: {message}"
      );
      assert!(
        message.contains("out of range"),
        "error message must say `out of range`, got: {message}"
      );
      assert!(
        message.contains(&bound.to_string()),
        "error message must name the vocab bound {bound}, got: {message}"
      );
    }
    Err(other) => panic!("expected Error::ShapeMismatch, got different Err: {other:?}"),
    Ok(_) => panic!("out-of-range eos id must yield Err, not Ok"),
  }
}

#[test]
fn llguidance_terminal_grammar_accepts_padded_model_eos_id() {
  // Padded-LM-head case: tokenizer vocab is N, `model_vocab_size =
  // Some(n)` with `n > N` widens the toktrie to `n` (placeholder
  // special tokens fill `[N, n)`). The validation bound must be
  // `model_vocab_size.unwrap_or(tokenizer_vocab)` so a model-config
  // EOS id within the padded range is not spuriously rejected — only
  // ids above BOTH bounds are genuinely out of range.
  //
  // Two sub-checks:
  //
  // (a) EOS id in the padded range (`[bt_vocab, model_vocab)`) is
  //     accepted AND actually drives the terminal-grammar EOS-only
  //     mask to unmask that exact id — proving the R4 path registers
  //     padded-range EOS ids through the widened
  //     `TokTrie::with_eos_tokens` (the earlier R3 path silently
  //     filtered them out, leaving the trie's EOS set defaulted to
  //     upstream's auto-detected id, often `0`).
  //
  // (b) EOS id in the UNPADDED range with a padded `model_vocab_size`
  //     supplied is unmasked correctly in the terminal-grammar
  //     EOS-only mask (the realistic padded-model setup: EOS lives in
  //     the tokenizer's actual vocab, model_vocab is just wider for
  //     hardware alignment).
  //
  // The fixture has 99 base ids (3 specials + 95 printables + 1 added
  // `<|im_end|>` at id 98). Pin `model_vocab_size = Some(128)`.

  // ─ Sub-check (a): padded-range eos id is ACCEPTED AND DRIVES the
  // EOS-only mask. The strengthened R4 form: construct the processor,
  // drive the regex grammar to terminal state by consuming `a`, and
  // assert the EOS-only mask leaves id 120 finite. Under the R3
  // filtered-FFI workaround this sub-check WOULD have surfaced the
  // silent-failure bug (id 120 was sliced out of the FFI call, so
  // `tok_trie.eos_token_set()` defaulted to id 0 — the post-consume
  // mask would have left 0 finite and 120 `-inf`, the opposite of what
  // the mlxrs Tokenizer reported).
  let tok_padded_only = fixture_tokenizer_with_eos_override(
    "padded_eos_accepted_padded_range",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[120],
  );
  let tok_vocab_a = tok_padded_only.hf().get_vocab_size(true);
  assert!(
    120 >= tok_vocab_a as u32,
    "test premise: eos id 120 must be in the PADDED range \
     (above the tokenizer vocab {tok_vocab_a})"
  );
  let model_vocab = 128usize;
  assert!(
    120usize < model_vocab,
    "test premise: eos id 120 must be inside model_vocab {model_vocab}"
  );
  let grammar_a = structured::GrammarSpec::Regex("a".to_string());
  let proc_a =
    structured::LLGuidanceLogitsProcessor::new(grammar_a, &tok_padded_only, Some(model_vocab))
      .expect(
        "padded-model EOS id (in [tokenizer_vocab, model_vocab)) must be ACCEPTED — \
         constructor must not panic and must not return Err",
      );
  let zeros_a = vec![0.0f32; model_vocab];
  let logits_a = Array::from_slice::<f32>(&zeros_a, &(1, model_vocab)).unwrap();
  let _ = proc_a
    .apply(&[], &logits_a)
    .expect("first apply should succeed");
  let a_token_id_a = id_for_byte(b'a');
  let mut out_a = proc_a
    .apply(&[a_token_id_a], &logits_a)
    .expect("post-consume apply should succeed and return EOS-only mask");
  let out_va = out_a.to_vec::<f32>().unwrap();
  assert_eq!(out_va.len(), model_vocab);
  // Padded-range eos id 120 must be the one finite position in the
  // EOS-only mask — proving option (a)'s post-widening
  // `with_eos_tokens` registered it.
  assert!(
    out_va[120].is_finite(),
    "padded-range EOS-only mask: padded eos id 120 must remain finite, got {}",
    out_va[120]
  );
  // Upstream's pre-R4 fallback default (`tok_eos = 0`) must NOT be
  // unmasked — the override REPLACES it entirely.
  assert!(
    out_va[0].is_infinite() && out_va[0] < 0.0,
    "padded-range EOS-only mask: id 0 (upstream fallback default) must be -inf, got {}",
    out_va[0]
  );
  // Every other id (including padded placeholders `[tok_vocab_a,
  // model_vocab)` other than 120) is `-inf`.
  for (i, v) in out_va.iter().enumerate() {
    if i == 120 {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "padded-range EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }

  // ─ Sub-check (b): in-unpadded-range eos id with padded model_vocab
  //   round-trips through the EOS-only mask (the realistic case the
  //   `model_vocab_size = Some(n)` path is designed for).
  let tok_unpadded = fixture_tokenizer_with_eos_override(
    "padded_eos_accepted_unpadded_range",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[CUSTOM_EOS_ID],
  );
  let tok_vocab_b = tok_unpadded.hf().get_vocab_size(true);
  assert!(
    (CUSTOM_EOS_ID as usize) < tok_vocab_b,
    "test premise: CUSTOM_EOS_ID ({CUSTOM_EOS_ID}) must be in the unpadded \
     tokenizer vocab {tok_vocab_b}"
  );
  let grammar_b = structured::GrammarSpec::Regex("a".to_string());
  let proc_b =
    structured::LLGuidanceLogitsProcessor::new(grammar_b, &tok_unpadded, Some(model_vocab))
      .expect("padded model_vocab with in-range EOS must be accepted");

  let zeros = vec![0.0f32; model_vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, model_vocab)).unwrap();
  let _ = proc_b
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let a_token_id = id_for_byte(b'a');
  let mut out = proc_b
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed and return EOS-only mask");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), model_vocab);

  // The configured eos id is finite; every other id (including the
  // padded placeholder positions in `[tok_vocab_b, model_vocab)`) is
  // `-inf`.
  assert!(
    out_v[CUSTOM_EOS_ID as usize].is_finite(),
    "padded-vocab EOS-only mask: id {CUSTOM_EOS_ID} must remain finite, got {}",
    out_v[CUSTOM_EOS_ID as usize]
  );
  for (i, v) in out_v.iter().enumerate() {
    if i as u32 == CUSTOM_EOS_ID {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "padded-vocab EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }
}

// ── Finding 1 (R4): padded-range EOS actually reaches the runtime mask ──
//
// R3 filtered configured EOS ids `>= bt.tokrx_info().vocab_size` out of
// the upstream `set_eos_tokens` call to dodge its `assert!`. That
// SILENTLY accepted padded-only EOS configs (e.g. `[120]` with
// `bt_vocab=99` + `model_vocab_size=Some(128)`): `in_range` was empty,
// `set_eos_tokens` was never called, and `tok_trie.eos_token_set()`
// defaulted to upstream's auto-detected `tok_eos` (often `0`). At the
// terminal-grammar stopped state, `compute_mask_or_eos` then unmasked
// the WRONG token id — mlxrs `Tokenizer::eos_token_ids()` reported
// `{120}` while the structured-decoder emitted id `0`.
//
// R4 fix: register configured EOS ids POST-widening via
// `ByteTokenizerEnv::new(bt, n_vocab)` + `tok_trie.with_eos_tokens`
// (`toktrie/src/toktree.rs:300-313`), whose assert is against the
// WIDENED `vocab_size()` — `120 < 128` is now valid and fully
// propagated. The strengthened
// `llguidance_terminal_grammar_accepts_padded_model_eos_id` (sub-check
// a) exercises the realistic case; this test is a directly-named
// regression pin so the silent-failure case is hard-locked.

#[test]
fn llguidance_terminal_grammar_padded_eos_id_actually_unmasks_at_runtime() {
  // Padded-only EOS configuration: configured eos = `[120]`, tokenizer
  // vocab = 99 (so `120 >= bt_vocab`), model_vocab_size = `Some(128)`.
  // Under R3 the FFI call was suppressed and the EOS-only mask
  // defaulted to id `0`. Under R4 the configured eos is registered
  // against the WIDENED trie, so the EOS-only mask leaves id `120`
  // finite and every other position (incl. `0`) `-inf`.
  let tok = fixture_tokenizer_with_eos_override(
    "r4_padded_eos_runtime_mask",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[120],
  );
  let bt_vocab = tok.hf().get_vocab_size(true);
  assert!(
    120u32 >= bt_vocab as u32,
    "test premise: eos id 120 must be padded-range \
     (above the unpadded tokenizer vocab {bt_vocab})"
  );
  let model_vocab = 128usize;
  assert!(
    120usize < model_vocab,
    "test premise: eos id 120 must be inside model_vocab {model_vocab}"
  );

  let grammar = structured::GrammarSpec::Regex("a".to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, Some(model_vocab))
    .expect("padded-only EOS construction must succeed under R4");

  let zeros = vec![0.0f32; model_vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, model_vocab)).unwrap();
  // Prime, then drive to terminal state by consuming `a`.
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let a_token_id = id_for_byte(b'a');
  let mut out = proc
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed and return EOS-only mask");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), model_vocab);

  // Post-R4 contract: padded-range eos id `120` is the ONLY finite
  // position in the EOS-only mask. Under R3 this would have FAILED:
  // id 0 (upstream's pre-R4 fallback) would have been finite and id
  // 120 would have been `-inf`.
  assert!(
    out_v[120].is_finite(),
    "R4 padded-range EOS-only mask: padded eos id 120 must remain finite, got {}",
    out_v[120]
  );
  assert!(
    out_v[0].is_infinite() && out_v[0] < 0.0,
    "R4 padded-range EOS-only mask: id 0 (upstream pre-R4 fallback default) \
     must be -inf, got {}",
    out_v[0]
  );
  // Every other id (every placeholder padded slot, every printable, every
  // base special other than the configured 120) must be `-inf`.
  for (i, v) in out_v.iter().enumerate() {
    if i == 120 {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "R4 padded-range EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }
}

#[test]
fn llguidance_terminal_grammar_mixed_in_range_plus_padded_eos() {
  // Mixed configuration: `eos_token_ids = [98, 120]` + `model_vocab_size
  // = Some(128)` with `bt_vocab = 99`. Under R3 the FFI call received
  // `[98]` (in-range filter), so `tok_trie.eos_token_set()` had `{98}`
  // and the EOS-only mask UNMASKED id 98 but MASKED id 120 — the
  // padded-range id was silently dropped. Under R4 both ids are
  // registered against the widened trie and both must be finite.
  let tok = fixture_tokenizer_with_eos_override(
    "r4_mixed_eos_in_range_plus_padded",
    &[(CUSTOM_EOS_ID, "<|im_end|>")],
    &[CUSTOM_EOS_ID, 120],
  );
  let bt_vocab = tok.hf().get_vocab_size(true);
  assert!(
    (CUSTOM_EOS_ID as usize) < bt_vocab,
    "test premise: CUSTOM_EOS_ID ({CUSTOM_EOS_ID}) must be in the unpadded \
     tokenizer vocab {bt_vocab}"
  );
  assert!(
    120u32 >= bt_vocab as u32,
    "test premise: eos id 120 must be padded-range \
     (above the unpadded tokenizer vocab {bt_vocab})"
  );
  let model_vocab = 128usize;

  let grammar = structured::GrammarSpec::Regex("a".to_string());
  let proc = structured::LLGuidanceLogitsProcessor::new(grammar, &tok, Some(model_vocab))
    .expect("mixed in-range + padded EOS construction must succeed under R4");

  let zeros = vec![0.0f32; model_vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, model_vocab)).unwrap();
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let a_token_id = id_for_byte(b'a');
  let mut out = proc
    .apply(&[a_token_id], &logits)
    .expect("post-consume apply should succeed and return EOS-only mask");
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), model_vocab);

  // BOTH configured eos ids are finite (post-R4: both reach the
  // widened trie's `eos_token_set`). Under R3 only `CUSTOM_EOS_ID`
  // (98) would have been finite; `120` would have been `-inf`.
  assert!(
    out_v[CUSTOM_EOS_ID as usize].is_finite(),
    "R4 mixed EOS-only mask: in-range eos id {CUSTOM_EOS_ID} must remain finite, got {}",
    out_v[CUSTOM_EOS_ID as usize]
  );
  assert!(
    out_v[120].is_finite(),
    "R4 mixed EOS-only mask: padded-range eos id 120 must remain finite, got {}",
    out_v[120]
  );
  // Every other id is `-inf`.
  for (i, v) in out_v.iter().enumerate() {
    if i as u32 == CUSTOM_EOS_ID || i == 120 {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "R4 mixed EOS-only mask: non-eos id {i} must be -inf, got {v}"
    );
  }
}
