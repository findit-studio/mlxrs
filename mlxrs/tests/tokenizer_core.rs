//! serde_json-free-core proof: bare `--features tokenizer` exposes
//! encode/decode (+ convert + thinking) built purely from `tokenizer.json`,
//! with **no `serde_json` on any reachable code path**.
//!
//! This is also a compile-level check: it is gated to the bare-`tokenizer`
//! configuration (no `tokenizer-config`, no `tokenizer-stream`), so it only
//! builds — and only links the dependency graph — of that minimal subset.
//! `cargo tree -p mlxrs --features tokenizer -e normal -i serde_json` is the
//! companion dependency-graph proof.
#![cfg(all(
  feature = "tokenizer",
  not(feature = "tokenizer-config"),
  not(feature = "tokenizer-stream")
))]

use std::io::Write;

use mlxrs::tokenizer::Tokenizer;

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
      let dir = std::env::temp_dir().join(format!("mlxrs-tok-core-fixture-{}", std::process::id()));
      std::fs::create_dir_all(&dir).unwrap();
      let mut f = std::fs::File::create(dir.join("tokenizer.json")).unwrap();
      f.write_all(TOKENIZER_JSON.as_bytes()).unwrap();
      dir
    })
    .clone()
}

#[test]
fn bare_tokenizer_encode_decode_convert() {
  let tok = Tokenizer::from_path(fixture_dir(), None).unwrap();

  let ids = tok
    .encode("hello world the quick brown fox", false)
    .unwrap();
  assert_eq!(ids, vec![3, 4, 5, 6, 7, 8]);

  let text = tok.decode(&ids, false).unwrap();
  assert!(text.contains("hello"));
  assert!(text.contains("fox"));
  assert_eq!(tok.encode(&text, false).unwrap(), ids);

  // Special tokens known to `tokenizer.json` resolve without any config.
  assert_eq!(tok.convert_token_to_id("hello"), Some(3));
  assert_eq!(tok.convert_id_to_token(3).as_deref(), Some("hello"));

  // Batch APIs.
  let batch = tok
    .encode_batch(vec!["hello".into(), "world".into()], false)
    .unwrap();
  assert_eq!(batch, vec![vec![3u32], vec![4u32]]);

  // _infer_thinking is vocab-based (no `serde_json`).
  assert!(tok.has_thinking());
  assert_eq!(tok.think_start(), Some("<think>"));
}
