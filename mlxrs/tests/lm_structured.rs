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
