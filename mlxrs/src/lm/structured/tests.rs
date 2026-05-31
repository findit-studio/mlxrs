//! Inline unit tests for the error / single-rank / state-reset branches of
//! [`LLGuidanceLogitsProcessor`] that the broad happy-path integration
//! coverage in `tests/lm_structured.rs` does not reach:
//!
//!   - `apply` on a single-rank `[V]` logits row (the `[v]` arm of every
//!     shape `match` — the integration tests only ever feed `[1, V]`);
//!   - the [`Error::RankMismatch`] arm for any rank that is neither `[V]`
//!     nor `[1, V]`;
//!   - the [`Matcher::consume_token`] error arm (a subsequent step feeding
//!     a token the grammar rejects);
//!   - the construction-time [`Matcher::get_error`] arm (a grammar that
//!     fails to compile);
//!   - the [`Error::LengthMismatch`] arm (matcher mask narrower than the
//!     logits' vocab, i.e. a padded LM head with `model_vocab_size = None`);
//!   - [`LLGuidanceLogitsProcessor::reset`] returning the processor to its
//!     first-step state.
//!
//! Fixtures mirror `tests/lm_structured.rs`: a minimal byte-level HF
//! tokenizer (printable ASCII + three specials) is the only shape the
//! `toktrie_hf_tokenizers::ByteTokenizer` adapter accepts (it requires a
//! `ByteLevel`/`ByteFallback` decoder). Every expectation is a closed-form
//! oracle: the allowed-token set is derived by hand from the grammar +
//! token sequence, never by calling the function under test.

use super::*;

use serde_json::json;

/// Build a minimal byte-level HF `tokenizer.json` string: the 95 printable
/// ASCII bytes (`0x20..=0x7E`) each mapped via the `tokenizers::ByteLevel`
/// byte→unicode convention, plus three specials (`<unk>`=0, `<s>`=1,
/// `</s>`=2). This matches the accepted fixture in `tests/lm_structured.rs`
/// (a `ByteLevel` pre-tokenizer + `ByteLevel` decoder are the two decoders
/// the adapter's `check_decoder` recognizes). Resulting vocab size is 98
/// (ids 0..=97).
fn byte_level_tokenizer_json() -> String {
  let mut vocab_entries: Vec<String> = Vec::new();
  vocab_entries.push("\"<unk>\": 0".to_string());
  vocab_entries.push("\"<s>\": 1".to_string());
  vocab_entries.push("\"</s>\": 2".to_string());
  let mut next_id: u32 = 3;
  // The `tokenizers::ByteLevel` byte→char table: self-mapped for `!..~`,
  // `00A1..00AC`, `00AE..00FF`; everything else remapped from U+0100.
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
      "pre_tokenizer": {{ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true }},
      "post_processor": null,
      "decoder": {{ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true }},
      "model": {{
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": {{ {} }},
        "merges": []
      }}
    }}"#,
    vocab_entries.join(",\n          ")
  )
}

const TOKENIZER_CONFIG_JSON: &str = r#"{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "model_max_length": 2048
}"#;

/// A fresh, unique temp directory per call (pid + monotonic counter),
/// matching the repo `fresh_dir` idiom. Leaked intentionally
/// (process-lifetime fixture).
fn temp_dir(name: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_structured_inline_{}_{}_{n}",
    std::process::id(),
    name
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Build + load the byte-level fixture tokenizer through the public
/// `Tokenizer::from_path` (cross-feature-combo-agnostic).
fn fixture_tokenizer(name: &str) -> Tokenizer {
  let dir = temp_dir(name);
  std::fs::write(dir.join("tokenizer.json"), byte_level_tokenizer_json()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_JSON).unwrap();
  Tokenizer::from_path(&dir, None).unwrap_or_else(|e| panic!("fixture tokenizer load failed: {e}"))
}

/// Id of a single-byte printable token in the fixture (region starts at 3).
fn id_for_byte(byte: u8) -> u32 {
  assert!(
    (0x20..=0x7E).contains(&byte),
    "byte {byte:#x} not in fixture vocab"
  );
  3 + (byte - 0x20) as u32
}

// ── single-rank `[V]` logits (the `[v]` arms + single-rank mask path) ──

/// `apply` on a `[V]` (rank-1) logits row exercises the `[v] => *v` vocab
/// arm (line 376), the `[v] => vec![*v as i32]` mask-shape arm (line 460),
/// and the single-rank `bool_mask = bool_mask_flat` branch (line 468) — the
/// integration suite only ever feeds `[1, V]`.
///
/// Oracle: for the JSON-object schema `{"type":"object"}` the only valid
/// first byte is `{` (or leading whitespace); an alphabetic byte cannot
/// start a JSON value, and `}` cannot be the first byte of an object. So
/// `out[{]` is finite while `out[a]` and `out[}]` are `-inf`. The output
/// shape stays rank-1 with the input vocab length.
#[test]
fn apply_single_rank_vector_logits_masks_by_grammar() {
  let tok = fixture_tokenizer("single_rank_vector");
  let proc = build_json_schema_logits_processor(json!({ "type": "object" }), &tok, None)
    .expect("processor construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  // Rank-1 `[V]` logits — NOT `[1, V]`.
  let logits = Array::from_slice::<f32>(&zeros, &(vocab,)).unwrap();

  let mut out = proc
    .apply(&[], &logits)
    .expect("apply should succeed on a rank-1 `[V]` logits row");
  // The returned array keeps the rank-1 input shape.
  assert_eq!(out.shape(), vec![vocab]);
  let out_v = out.to_vec::<f32>().unwrap();
  assert_eq!(out_v.len(), vocab);

  let open_brace = id_for_byte(b'{') as usize;
  assert!(
    out_v[open_brace].is_finite(),
    "`{{` (id {open_brace}) must remain finite in a `[V]` mask, got {}",
    out_v[open_brace]
  );
  let a = id_for_byte(b'a') as usize;
  assert!(
    out_v[a].is_infinite() && out_v[a] < 0.0,
    "`a` (id {a}) must be -inf in a `[V]` mask, got {}",
    out_v[a]
  );
  let close_brace = id_for_byte(b'}') as usize;
  assert!(
    out_v[close_brace].is_infinite() && out_v[close_brace] < 0.0,
    "`}}` (id {close_brace}) must be -inf in a `[V]` mask, got {}",
    out_v[close_brace]
  );
}

// ── rank-mismatch error arm ──

/// `apply` on a rank-2 batch `[2, V]` (not `[1, V]`) hits the catch-all
/// shape arm → [`Error::RankMismatch`] (lines 378-382). The payload carries
/// the observed rank (2) and the full observed shape.
#[test]
fn apply_rejects_batched_rank2_logits_with_rank_mismatch() {
  let tok = fixture_tokenizer("rank2_reject");
  let proc = build_json_schema_logits_processor(json!({ "type": "object" }), &tok, None)
    .expect("processor construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; 2 * vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(2, vocab)).unwrap();

  match proc.apply(&[], &logits) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(p.actual(), 2, "observed rank must be 2");
      assert_eq!(
        p.actual_shape(),
        [2usize, vocab].as_slice(),
        "payload must carry the full observed shape"
      );
      assert!(
        p.context().contains("[V]") && p.context().contains("[1, V]"),
        "context names the accepted shapes: {}",
        p.context()
      );
    }
    Err(other) => panic!("expected Error::RankMismatch, got: {other:?}"),
    Ok(_) => panic!("rank-2 `[2, V]` logits must be rejected, not accepted"),
  }
}

/// `apply` on a rank-3 `[1, 1, V]` row also hits the catch-all arm →
/// [`Error::RankMismatch`] with observed rank 3.
#[test]
fn apply_rejects_rank3_logits_with_rank_mismatch() {
  let tok = fixture_tokenizer("rank3_reject");
  let proc = build_json_schema_logits_processor(json!({ "type": "object" }), &tok, None)
    .expect("processor construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, 1, vocab)).unwrap();

  match proc.apply(&[], &logits) {
    Err(Error::RankMismatch(p)) => {
      assert_eq!(p.actual(), 3, "observed rank must be 3");
      assert_eq!(p.actual_shape(), [1usize, 1, vocab].as_slice());
    }
    Err(other) => panic!("expected Error::RankMismatch, got: {other:?}"),
    Ok(_) => panic!("rank-3 logits must be rejected"),
  }
}

// ── consume_token error arm (subsequent step, disallowed token) ──

/// After the first (priming) `apply`, feeding a token the grammar does NOT
/// allow makes [`Matcher::consume_token`] fail → [`Error::Parse`] with the
/// `"llguidance: consume_token"` context (lines 401-404).
///
/// Oracle: the `Regex("a")` grammar accepts ONLY the single byte `a` as its
/// first emission. The first `apply` does not consume any history (the
/// `is_first_token` branch). The second `apply` is told the model emitted
/// `b` (`id_for_byte(b'b')`), which the grammar never allowed — so the
/// matcher rejects it.
#[test]
fn apply_consume_disallowed_token_surfaces_parse_error() {
  let tok = fixture_tokenizer("consume_disallowed");
  let proc = LLGuidanceLogitsProcessor::new(GrammarSpec::Regex("a".to_string()), &tok, None)
    .expect("regex grammar construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();

  // Prime: first step consumes no history token.
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");

  // Second step claims the model emitted `b`, which the `a`-only grammar
  // never allowed → consume_token must error.
  let b_id = id_for_byte(b'b');
  match proc.apply(&[b_id], &logits) {
    Err(Error::Parse(p)) => {
      assert!(
        p.context().contains("consume_token"),
        "context must name consume_token: {}",
        p.context()
      );
      assert!(
        p.inner().to_string().contains(&b_id.to_string()),
        "the inner error should carry the offending token id {b_id}: {}",
        p.inner()
      );
    }
    Err(other) => panic!("expected Error::Parse(consume_token), got: {other:?}"),
    Ok(_) => panic!("consuming a grammar-disallowed token must error, not succeed"),
  }
}

// ── construction-time grammar-compile error (Matcher::get_error) ──

/// A Lark grammar that references an undefined rule fails to compile. The
/// failure is swallowed by `Matcher::new` into a sentinel error-state
/// matcher; `new` surfaces it via [`Matcher::get_error`] as an
/// [`Error::Parse`] with the `"llguidance: grammar compile"` context
/// (lines 338-341) at CONSTRUCTION time (not per-step).
///
/// Oracle: `start: undefined_rule` names `undefined_rule`, which has no
/// production — the upstream lark compiler raises `"unknown name:
/// \"undefined_rule\""`, so `create_parser` returns `Err` and the
/// matcher lands in its error state.
#[test]
fn new_with_uncompilable_grammar_errors_at_construction() {
  let tok = fixture_tokenizer("bad_grammar_compile");
  let lark = "start: undefined_rule\n".to_string();
  match LLGuidanceLogitsProcessor::new(GrammarSpec::Lark(lark), &tok, None) {
    Err(Error::Parse(p)) => {
      assert!(
        p.context().contains("grammar compile"),
        "context must name the grammar-compile stage: {}",
        p.context()
      );
      assert_eq!(p.input_kind(), "llguidance grammar");
    }
    Err(other) => panic!("expected Error::Parse(grammar compile), got: {other:?}"),
    Ok(_) => panic!("an uncompilable grammar must yield Err at construction, not Ok"),
  }
}

// ── length-mismatch arm (matcher mask narrower than logits vocab) ──

/// With `model_vocab_size = None` the matcher's mask width equals the
/// tokenizer's own vocab. Feeding logits whose last axis is WIDER than that
/// (a padded LM head used WITHOUT the `Some(n)` override) trips the
/// `mask.len() < vocab` guard → [`Error::LengthMismatch`] (lines 440-443).
///
/// Oracle: `ByteTokenizer` sets `vocab_size = get_vocab_size(true)` with no
/// rounding (and `SimpleVob::len()` returns exactly that), so `mask.len()`
/// is precisely the tokenizer vocab; logits of width `tok_vocab + 64` make
/// `expected = tok_vocab + 64` and `actual = tok_vocab`.
#[test]
fn apply_errors_when_mask_narrower_than_unpadded_logits() {
  let tok = fixture_tokenizer("mask_narrower_than_logits");
  let tok_vocab = tok.hf().get_vocab_size(true);
  let model_vocab = tok_vocab + 64;

  // `None` → matcher mask width stays at the tokenizer's vocab.
  let proc = build_json_schema_logits_processor(json!({ "type": "object" }), &tok, None)
    .expect("processor construction should succeed");

  let zeros = vec![0.0f32; model_vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, model_vocab)).unwrap();

  match proc.apply(&[], &logits) {
    Err(Error::LengthMismatch(p)) => {
      assert_eq!(
        p.expected(),
        model_vocab,
        "expected = logits vocab width ({model_vocab})"
      );
      assert_eq!(
        p.actual(),
        tok_vocab,
        "actual = matcher mask length ({tok_vocab})"
      );
      assert!(
        p.context().contains("mask vs logits vocab"),
        "context names the mask-vs-vocab comparison: {}",
        p.context()
      );
    }
    Err(other) => panic!("expected Error::LengthMismatch, got: {other:?}"),
    Ok(_) => panic!("a too-narrow matcher mask must error, not silently pass"),
  }
}

// ── reset() returns the processor to first-step state ──

/// [`LLGuidanceLogitsProcessor::reset`] (lines 486-495) rewinds the matcher
/// and re-arms the `is_first_token` flag, so the NEXT `apply` is treated as
/// the first step again (consuming no history) and computes the grammar's
/// initial mask.
///
/// Oracle: the `Regex("a")` grammar's initial mask allows ONLY `a`. We
/// advance to the terminal state (prime, then consume `a` → grammar
/// satisfied → next mask would be EOS-only). After `reset`, a fresh first
/// `apply` must again allow `a` and mask everything else — proving both the
/// matcher rewind AND the first-token re-arm. If `reset` had not rewound,
/// the post-`a` matcher would still be terminal and the mask would be
/// EOS-only (id 2 finite, `a` masked).
#[test]
fn reset_returns_to_initial_first_step_state() {
  let tok = fixture_tokenizer("reset_first_step");
  let proc = LLGuidanceLogitsProcessor::new(GrammarSpec::Regex("a".to_string()), &tok, None)
    .expect("regex grammar construction should succeed");

  let vocab = tok.hf().get_vocab_size(true);
  let zeros = vec![0.0f32; vocab];
  let logits = Array::from_slice::<f32>(&zeros, &(1, vocab)).unwrap();
  let a_id = id_for_byte(b'a');
  let eos_id = 2usize; // `</s>` in the fixture.

  // Prime (first step consumes no history), then a second `apply` that
  // tells the matcher the model emitted `a`: it consumes `a` (the grammar
  // is now satisfied / terminal) and returns the EOS-only mask. This is the
  // same two-call flow the integration suite uses to reach the terminal
  // state — do NOT add a third `apply`, which would re-consume `a` on an
  // already-stopped matcher and error.
  let _ = proc
    .apply(&[], &logits)
    .expect("first apply should succeed");
  let mut terminal = proc
    .apply(&[a_id], &logits)
    .expect("post-consume apply should return the EOS-only mask");

  // Sanity on the pre-reset terminal state: the mask is EOS-only, so `a` is
  // now masked and the eos id is finite. This anchors that reset's effect
  // (below) is real, not a no-op on an already-initial matcher.
  let terminal_v = terminal.to_vec::<f32>().unwrap();
  assert!(
    terminal_v[eos_id].is_finite(),
    "pre-reset terminal mask: eos id {eos_id} must be finite, got {}",
    terminal_v[eos_id]
  );
  assert!(
    terminal_v[a_id as usize].is_infinite() && terminal_v[a_id as usize] < 0.0,
    "pre-reset terminal mask: `a` (id {a_id}) must be -inf, got {}",
    terminal_v[a_id as usize]
  );

  // Reset: rewind + re-arm is_first_token.
  proc.reset().expect("reset should succeed");

  // The NEXT apply is the first step again over the INITIAL grammar mask:
  // only `a` is allowed; the eos id (and everything else) is masked.
  let mut after = proc
    .apply(&[], &logits)
    .expect("post-reset first apply should succeed");
  let after_v = after.to_vec::<f32>().unwrap();
  assert_eq!(after_v.len(), vocab);
  assert!(
    after_v[a_id as usize].is_finite(),
    "post-reset initial mask: `a` (id {a_id}) must be finite again, got {}",
    after_v[a_id as usize]
  );
  // Every non-`a` token is masked in the initial single-byte grammar mask
  // (including the eos id, which was finite in the terminal state) — this
  // is the closed-form initial allowed-set of `Regex("a")`.
  for (i, v) in after_v.iter().enumerate() {
    if i as u32 == a_id {
      continue;
    }
    assert!(
      v.is_infinite() && *v < 0.0,
      "post-reset initial mask: non-`a` id {i} must be -inf, got {v}"
    );
  }
}
