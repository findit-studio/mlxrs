//! Bare `tokenizer-stream` naive-detokenizer tests.
//!
//! Mirrors mlx-swift-lm's `NaiveStreamingDetokenizer`: the only detokenizer
//! available without `tokenizer-spm` / `tokenizer-bpe`. Re-decodes the
//! accumulated segment via the `tokenizers` crate's own configured decoder.
//! No SPM/BPE class inference, no tool parsers / override registry, no
//! `serde_json` on the streaming path.
#![cfg(all(feature = "tokenizer-stream", not(feature = "tokenizer-config")))]

use std::io::Write;

// P1 #111: `Tokenizer::detokenizer()` returns the enum-unified
// [`Detokenizer`]; methods like `add_token` / `text` / `last_segment`
// require the [`StreamingDetokenizer`] trait in scope.
use mlxrs::tokenizer::{StreamingDetokenizer, Tokenizer};

const TOKENIZER_JSON: &str = include_str!("fixtures/tokenizer.json");

/// Materialize the (deterministic, byte-identical-every-call) fixture
/// exactly once per test process. `cargo test` runs the tests in this
/// binary in parallel; creating + truncating + rewriting the same file on
/// every call races a concurrent `Tokenizer::from_path` read. The content
/// is constant, so a write-once `OnceLock` guard removes the race entirely
/// (every test then reads the same already-complete file) without
/// serializing the tests.
fn fixture_dir() -> std::path::PathBuf {
  static FIXTURE: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
  FIXTURE
    .get_or_init(|| {
      let dir =
        std::env::temp_dir().join(format!("mlxrs-tok-stream-fixture-{}", std::process::id()));
      std::fs::create_dir_all(&dir).unwrap();
      let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
      f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
      dir
    })
    .clone()
}

#[test]
fn naive_detokenizer_streams_and_reconstructs_full_decode() {
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let ids = tok
    .encode("hello world the quick brown fox", false)
    .unwrap();
  let full = tok.decode(&ids, false).unwrap();

  // Without `tokenizer-spm`/`tokenizer-bpe` the factory always yields the
  // naive re-decode detokenizer (graceful default). Stream tokens one at a
  // time and accumulate readable segments.
  let mut d = tok.detokenizer();
  d.reset();
  let mut streamed = String::new();
  for &t in &ids {
    d.add_token(t);
    streamed.push_str(&d.last_segment());
  }
  d.finalize();
  streamed.push_str(&d.last_segment());

  assert_eq!(d.text(), full);
  assert_eq!(d.tokens(), ids.as_slice());
  assert_eq!(streamed, full);
}

#[test]
fn encode_decode_round_trip_serde_json_free_core() {
  // Bare stream (no `tokenizer-config`): encode/decode work purely from
  // `tokenizer.json` with no `serde_json` linked on this path.
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();
  let ids = tok
    .encode("hello world the quick brown fox", false)
    .unwrap();
  assert_eq!(ids, vec![3, 4, 5, 6, 7, 8]);
  let text = tok.decode(&ids, false).unwrap();
  assert!(text.contains("hello"));
  assert!(text.contains("fox"));
  assert_eq!(tok.encode(&text, false).unwrap(), ids);
  // Thinking inference is vocab-based (no config / serde_json).
  assert!(tok.has_thinking());
  assert_eq!(tok.think_start(), Some("<think>"));
}
