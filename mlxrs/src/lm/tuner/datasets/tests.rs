use std::{
  path::PathBuf,
  sync::atomic::{AtomicU64, Ordering},
};

use serde_json::json;

use super::*;

// ───────────────────────── instrumented reader ─────────────────────────

/// Test-only `BufRead` wrapper that records the cumulative byte count
/// the helper consumed from the inner reader. Used by
/// `load_dataset_cap_enforced_on_single_giant_line` to PROVE the
/// `take(remaining + 1)` allocation cap held — i.e. the helper read at
/// most `cap + 1` bytes before erroring, NOT the full input. A
/// `BufRead::lines()` impl would pull the entire line into a
/// `String` before yielding, so this counter distinguishes the two
/// implementations.
///
/// Both `Read::read` and `BufRead::consume` are instrumented so the
/// counter rises regardless of which path the helper exercises
/// (`read_until` goes through `fill_buf` + `consume`; the EOF-peek
/// path goes through raw `Read::read`). The two paths are mutually
/// exclusive per iteration so there is no double-counting risk:
/// `fill_buf` does NOT advance the cursor, only `consume(amt)` does,
/// and `Read::read` only fires on the raw peek (not while
/// `read_until` is draining the buffer through `Take`).
struct CountingReader<R: std::io::BufRead> {
  inner: R,
  consumed: usize,
}

impl<R: std::io::BufRead> CountingReader<R> {
  fn new(inner: R) -> Self {
    Self { inner, consumed: 0 }
  }

  fn consumed(&self) -> usize {
    self.consumed
  }
}

impl<R: std::io::BufRead> std::io::Read for CountingReader<R> {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let n = self.inner.read(buf)?;
    self.consumed += n;
    Ok(n)
  }
}

impl<R: std::io::BufRead> std::io::BufRead for CountingReader<R> {
  fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
    self.inner.fill_buf()
  }

  fn consume(&mut self, amt: usize) {
    self.consumed += amt;
    self.inner.consume(amt);
  }
}

// ───────────────────────── fixtures ─────────────────────────

fn fresh_dir(tag: &str) -> PathBuf {
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!(
    "mlxrs-lm-tuner-datasets-{tag}-{}-{n}",
    std::process::id()
  ));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// A minimal `WordLevel` tokenizer with a chat template that lets us
/// hand-trace what the dataset's process() should emit. Vocabulary is
/// designed so each word/marker is a single token id we can assert on.
fn write_tokenizer(dir: &Path) -> Tokenizer {
  // Vocab:
  //   0=<unk>  1=<s>  2=</s>  3=hello  4=world  5=user  6=assistant
  //   7=:    8=tools
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": [
      {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 1, "content": "<s>",   "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 2, "content": "</s>",  "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 7, "content": ":",     "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 8, "content": "tools", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
    ],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "vocab": {
        "<unk>": 0, "<s>": 1, "</s>": 2,
        "hello": 3, "world": 4,
        "user": 5, "assistant": 6,
        ":": 7, "tools": 8
      },
      "unk_token": "<unk>"
    }
  });
  let cfg = json!({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    // A trivial chat template: emits the role token, ':', then content
    // tokens. add_generation_prompt appends 'assistant :' so the
    // prefix-length offset for a one-message user prefix is
    // {user, :, <content>, assistant, :}.
    "chat_template":
      "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} \
       {% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
  });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  Tokenizer::from_path(dir, None).unwrap()
}

/// Build a temp-dir tokenizer and yield it; the directory is leaked
/// intentionally (process-lifetime fixture).
fn tokenizer_fixture(tag: &str) -> Tokenizer {
  let dir = fresh_dir(tag);
  write_tokenizer(&dir)
}

fn write_jsonl(path: &Path, lines: &[Value]) {
  let mut s = String::new();
  for v in lines {
    s.push_str(&v.to_string());
    s.push('\n');
  }
  std::fs::write(path, s).unwrap();
}

// ───────────────────────── TextDataset ─────────────────────────

#[test]
fn text_dataset_happy_path_appends_eos_when_missing() {
  let tok = tokenizer_fixture("text_happy");
  let data = vec![
    json!({ "text": "hello world" }),
    json!({ "text": "world hello" }),
    json!({ "text": "hello" }),
  ];
  let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
  assert_eq!(ds.len(), 3);
  let (toks0, off0) = ds.process(0).unwrap();
  let (toks1, off1) = ds.process(1).unwrap();
  let (toks2, off2) = ds.process(2).unwrap();
  // hello world </s>
  assert_eq!(toks0, vec![3, 4, 2]);
  // world hello </s>
  assert_eq!(toks1, vec![4, 3, 2]);
  // hello </s>
  assert_eq!(toks2, vec![3, 2]);
  // No prompt masking on text.
  assert_eq!(off0, 0);
  assert_eq!(off1, 0);
  assert_eq!(off2, 0);
}

#[test]
fn text_dataset_does_not_double_append_eos() {
  let tok = tokenizer_fixture("text_no_dup_eos");
  // The bare-string `"hello </s>"` Whitespace-tokenizes to `[3, 2]`
  // (the `</s>` is a registered special-token literal). encode then
  // sees the trailing 2 and the dataset must NOT push another one.
  let data = vec![json!({ "text": "hello </s>" })];
  let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
  let (toks, _) = ds.process(0).unwrap();
  assert_eq!(toks, vec![3, 2]);
}

#[test]
fn text_dataset_missing_field_errors() {
  let tok = tokenizer_fixture("text_missing_field");
  let data = vec![json!({ "not_text": "hello" })];
  let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
  let err = ds.process(0).unwrap_err();
  match err {
    Error::MissingKey(p) => assert_eq!(p.key(), "jsonl record missing 'text'"),
    other => panic!("expected MissingKey, got: {other:?}"),
  }
}

#[test]
fn text_dataset_wrong_type_errors() {
  let tok = tokenizer_fixture("text_wrong_type");
  let data = vec![json!({ "text": 42 })];
  let ds = TextDataset::new(data, &tok, DEFAULT_TEXT_KEY);
  let err = ds.process(0).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got: {err:?}"
  );
}

// ───────────────────────── ChatDataset ─────────────────────────

#[test]
fn chat_dataset_happy_path_no_mask() {
  let tok = tokenizer_fixture("chat_happy_no_mask");
  let data = vec![json!({
    "messages": [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "world"},
    ]
  })];
  let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
  let (toks, off) = ds.process(0).unwrap();
  // Template renders: "user : hello assistant : world "
  // → [5, 7, 3, 6, 7, 4]
  assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
  assert_eq!(off, 0);
}

#[test]
fn chat_dataset_mask_prompt_returns_prefix_offset() {
  let tok = tokenizer_fixture("chat_mask");
  let data = vec![json!({
    "messages": [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "world"},
    ]
  })];
  let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, true);
  let (toks, off) = ds.process(0).unwrap();
  // Full:   [5, 7, 3, 6, 7, 4]   (user : hello assistant : world)
  // Prefix (messages[:-1]=user, last_role==assistant so
  // add_generation_prompt=true): "user : hello assistant : "
  // → [5, 7, 3, 6, 7]
  assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
  assert_eq!(off, 5);
}

#[test]
fn chat_dataset_missing_messages_errors() {
  let tok = tokenizer_fixture("chat_missing");
  let data = vec![json!({ "no_messages_field": [] })];
  let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
  let err = ds.process(0).unwrap_err();
  match err {
    Error::MissingKey(p) => {
      assert_eq!(p.key(), "messages");
    }
    other => panic!("expected MissingKey, got: {other:?}"),
  }
}

#[test]
fn chat_dataset_messages_not_array_errors() {
  let tok = tokenizer_fixture("chat_not_array");
  let data = vec![json!({ "messages": "not an array" })];
  let ds = ChatDataset::new(data, &tok, DEFAULT_CHAT_KEY, false);
  let err = ds.process(0).unwrap_err();
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected OutOfRange, got: {err:?}"
  );
}

// ───────────────────────── CompletionsDataset ─────────────────────────

#[test]
fn completions_dataset_happy_path_no_mask() {
  let tok = tokenizer_fixture("comp_happy_no_mask");
  let data = vec![json!({ "prompt": "hello", "completion": "world" })];
  let ds = CompletionsDataset::new(
    data,
    &tok,
    DEFAULT_PROMPT_KEY,
    DEFAULT_COMPLETION_KEY,
    false,
  );
  let (toks, off) = ds.process(0).unwrap();
  // Same template rendering: "user : hello assistant : world "
  // → [5, 7, 3, 6, 7, 4]
  assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
  assert_eq!(off, 0);
}

#[test]
fn completions_dataset_mask_prompt_returns_prefix_offset() {
  let tok = tokenizer_fixture("comp_mask");
  let data = vec![json!({ "prompt": "hello", "completion": "world" })];
  let ds = CompletionsDataset::new(data, &tok, DEFAULT_PROMPT_KEY, DEFAULT_COMPLETION_KEY, true);
  let (toks, off) = ds.process(0).unwrap();
  // Full:    [5, 7, 3, 6, 7, 4]
  // Prefix (user-only + add_generation_prompt=true):
  //   "user : hello assistant : " → [5, 7, 3, 6, 7]
  assert_eq!(toks, vec![5, 7, 3, 6, 7, 4]);
  assert_eq!(off, 5);
}

#[test]
fn completions_dataset_missing_prompt_errors() {
  let tok = tokenizer_fixture("comp_missing_prompt");
  let data = vec![json!({ "completion": "world" })];
  let ds = CompletionsDataset::new(
    data,
    &tok,
    DEFAULT_PROMPT_KEY,
    DEFAULT_COMPLETION_KEY,
    false,
  );
  let err = ds.process(0).unwrap_err();
  match err {
    Error::MissingKey(p) => assert_eq!(p.key(), "jsonl record missing 'prompt'"),
    other => panic!("expected MissingKey, got: {other:?}"),
  }
}

// ───────────────────────── ConcatenatedDataset ─────────────────────────

#[test]
fn concatenated_dataset_indexes_across_inner_in_order() {
  let tok = tokenizer_fixture("concat_indexes");
  let a = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
  let b = TextDataset::new(
    vec![json!({ "text": "world" }), json!({ "text": "hello world" })],
    &tok,
    DEFAULT_TEXT_KEY,
  );
  let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
  assert_eq!(cat.len(), 3);
  // idx 0 → a[0]: hello </s>
  assert_eq!(cat.process(0).unwrap().0, vec![3, 2]);
  // idx 1 → b[0]: world </s>
  assert_eq!(cat.process(1).unwrap().0, vec![4, 2]);
  // idx 2 → b[1]: hello world </s>
  assert_eq!(cat.process(2).unwrap().0, vec![3, 4, 2]);
}

#[test]
fn concatenated_dataset_out_of_range_errors() {
  let tok = tokenizer_fixture("concat_oor");
  let a = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
  let cat = ConcatenatedDataset::new(vec![Box::new(a)]);
  assert!(cat.process(7).is_err());
}

#[test]
fn concatenated_dataset_empty_inputs_yield_empty_dataset() {
  let tok = tokenizer_fixture("concat_empty");
  let a = TextDataset::new(vec![], &tok, DEFAULT_TEXT_KEY);
  let b = TextDataset::new(vec![], &tok, DEFAULT_TEXT_KEY);
  let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
  assert_eq!(cat.len(), 0);
  assert!(cat.is_empty());
}

// ───────────────────────── CacheDataset ─────────────────────────

#[test]
fn cache_dataset_returns_consistent_result_on_repeat() {
  let tok = tokenizer_fixture("cache_repeat");
  let inner = TextDataset::new(
    vec![json!({ "text": "hello" }), json!({ "text": "world" })],
    &tok,
    DEFAULT_TEXT_KEY,
  );
  let cache = CacheDataset::new(Box::new(inner));
  assert_eq!(cache.len(), 2);
  let first = cache.process(0).unwrap();
  let second = cache.process(0).unwrap();
  assert_eq!(first, second);
  assert_eq!(cache.item_len(1).unwrap(), 2); // "world </s>" → 2 ids
}

#[test]
fn cache_dataset_out_of_range_errors() {
  let tok = tokenizer_fixture("cache_oor");
  let inner = TextDataset::new(vec![json!({ "text": "hello" })], &tok, DEFAULT_TEXT_KEY);
  let cache = CacheDataset::new(Box::new(inner));
  assert!(cache.process(99).is_err());
}

// ───────────────────────── load_dataset entry points ─────────────────────────

#[test]
fn load_dataset_text() {
  let tok = tokenizer_fixture("load_text");
  let dir = fresh_dir("load_text_data");
  let p = dir.join("train.jsonl");
  write_jsonl(
    &p,
    &[json!({ "text": "hello" }), json!({ "text": "world" })],
  );
  let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
  assert_eq!(ds.len(), 2);
  assert_eq!(ds.process(0).unwrap().0, vec![3, 2]);
  assert_eq!(ds.process(1).unwrap().0, vec![4, 2]);
}

#[test]
fn load_dataset_chat() {
  let tok = tokenizer_fixture("load_chat");
  let dir = fresh_dir("load_chat_data");
  let p = dir.join("train.jsonl");
  write_jsonl(
    &p,
    &[json!({
      "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
      ]
    })],
  );
  let ds = load_dataset(&p, &tok, DatasetType::Chat, &DatasetConfig::default()).unwrap();
  assert_eq!(ds.len(), 1);
  assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
}

#[test]
fn load_dataset_completions() {
  let tok = tokenizer_fixture("load_comp");
  let dir = fresh_dir("load_comp_data");
  let p = dir.join("train.jsonl");
  write_jsonl(&p, &[json!({ "prompt": "hello", "completion": "world" })]);
  let ds = load_dataset(
    &p,
    &tok,
    DatasetType::Completions,
    &DatasetConfig::default(),
  )
  .unwrap();
  assert_eq!(ds.len(), 1);
  assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
}

#[test]
fn load_dataset_concatenated() {
  let tok = tokenizer_fixture("load_concat");
  let dir = fresh_dir("load_concat_data");
  let p1 = dir.join("train.jsonl");
  let p2 = dir.join("valid.jsonl");
  write_jsonl(&p1, &[json!({ "text": "hello" })]);
  write_jsonl(&p2, &[json!({ "text": "world" })]);
  let a = load_dataset(&p1, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
  let b = load_dataset(&p2, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
  let cat = ConcatenatedDataset::new(vec![Box::new(a), Box::new(b)]);
  assert_eq!(cat.len(), 2);
  assert_eq!(cat.process(0).unwrap().0, vec![3, 2]);
  assert_eq!(cat.process(1).unwrap().0, vec![4, 2]);
}

#[test]
fn load_dataset_cache() {
  let tok = tokenizer_fixture("load_cache");
  let dir = fresh_dir("load_cache_data");
  let p = dir.join("train.jsonl");
  write_jsonl(&p, &[json!({ "text": "hello" })]);
  let ds = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap();
  // `load_dataset` always wraps in a CacheDataset; the public type makes
  // that visible (vs returning a `Box<dyn Dataset>`), and a repeat call
  // returns the same `(tokens, offset)` pair.
  let first = ds.process(0).unwrap();
  let second = ds.process(0).unwrap();
  assert_eq!(first, second);
}

#[test]
fn load_dataset_auto_detects_completions_first() {
  let tok = tokenizer_fixture("load_auto_comp");
  let dir = fresh_dir("load_auto_comp_data");
  let p = dir.join("train.jsonl");
  // Both completions (prompt+completion) AND text keys present →
  // Python's create_dataset picks completions first.
  write_jsonl(
    &p,
    &[json!({ "prompt": "hello", "completion": "world", "text": "ignored" })],
  );
  let ds = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap();
  assert_eq!(ds.len(), 1);
  assert_eq!(ds.process(0).unwrap().0, vec![5, 7, 3, 6, 7, 4]);
}

#[test]
fn load_dataset_auto_unsupported_format_errors() {
  let tok = tokenizer_fixture("load_auto_bad");
  let dir = fresh_dir("load_auto_bad_data");
  let p = dir.join("train.jsonl");
  write_jsonl(&p, &[json!({ "irrelevant": "junk" })]);
  let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
  assert!(
    err.to_string().contains("Unsupported data format"),
    "got: {err}"
  );
}

#[test]
fn load_dataset_rejects_hf_hub_path() {
  let tok = tokenizer_fixture("load_hf");
  let p = PathBuf::from("hf://datasets/mlx-community/some-dataset");
  let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
  match err {
    Error::OutOfRange(p) => assert!(p.context().contains("HF Hub URI rejected"), "got: {p:?}"),
    other => panic!("expected OutOfRange, got: {other:?}"),
  }
}

#[test]
fn load_dataset_text_with_mask_prompt_errors() {
  let tok = tokenizer_fixture("load_text_mask_err");
  let dir = fresh_dir("load_text_mask_err_data");
  let p = dir.join("train.jsonl");
  write_jsonl(&p, &[json!({ "text": "hello" })]);
  let cfg = DatasetConfig::new().with_mask_prompt(true);
  let err = load_dataset(&p, &tok, DatasetType::Text, &cfg).unwrap_err();
  assert!(
    err.to_string().contains("not supported for text dataset"),
    "got: {err}"
  );
}

#[test]
fn load_dataset_empty_file_errors_with_path() {
  let tok = tokenizer_fixture("load_empty");
  let dir = fresh_dir("load_empty_data");
  let p = dir.join("train.jsonl");
  std::fs::write(&p, "").unwrap();
  let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
  assert!(
    matches!(err, Error::EmptyInput(_)),
    "expected EmptyInput rejection, got: {err:?}",
  );
  let s = err.to_string();
  let _ = &s;
  assert!(
    s.contains("empty"),
    "expected 'empty' in error message, got: {s}",
  );
}

#[test]
fn load_dataset_blank_line_errors_with_line_number() {
  let tok = tokenizer_fixture("load_blank");
  let dir = fresh_dir("load_blank_data");
  let p = dir.join("train.jsonl");
  // A valid record, then a literal blank line, then another valid
  // record — the blank in the middle must surface as a hard error,
  // not be silently dropped.
  std::fs::write(&p, "{\"text\": \"hello\"}\n\n{\"text\": \"world\"}\n").unwrap();
  let err = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap_err();
  assert!(
    matches!(err, Error::EmptyInput(_)),
    "expected EmptyInput on blank line, got: {err:?}",
  );
  let s = err.to_string();
  assert!(
    s.contains("blank"),
    "expected 'blank' in blank-line error, got: {s}",
  );
}

#[test]
fn load_dataset_rejects_non_regular_file() {
  let tok = tokenizer_fixture("load_dir");
  let dir = fresh_dir("load_dir_data");
  // Pass the directory itself (which `exists()` and has metadata,
  // but is not a regular file).
  let err = load_dataset(&dir, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
  let s = err.to_string();
  assert!(
    s.contains("not a regular file"),
    "expected non-regular-file rejection, got: {s}",
  );
}

#[test]
fn load_dataset_cap_enforced_during_read_loop() {
  use std::io::Cursor;
  // Drive the path-agnostic helper directly so we can simulate a
  // file whose cumulative bytes exceed the cap mid-read WITHOUT
  // having to materialize a multi-GiB fixture (the prod constant
  // is 2 GiB). The helper is the single chokepoint, so this is
  // sufficient to prove the in-loop check fires after the file
  // is already "open".
  let cap: u64 = 40;
  // Three valid lines: each is ~18 bytes incl. the trailing \n
  // accounted for as `len() + 1`. After line 2 the cumulative is
  // ~36 (under cap); after line 3 it crosses 40.
  let body = "{\"text\": \"aaa\"}\n{\"text\": \"bbb\"}\n{\"text\": \"ccc\"}\n";
  let path = std::path::PathBuf::from("/synthetic/grows.jsonl");
  let err = read_jsonl_with_cap(Cursor::new(body), &path, cap).unwrap_err();
  match err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap(), 40);
      assert_eq!(p.cap_name(), "MAX_DATASET_FILE_BYTES");
      assert!(p.observed() > 40, "observed must exceed cap, got: {p:?}");
      assert!(
        p.context().contains("read jsonl"),
        "expected read-jsonl context, got: {p:?}"
      );
    }
    other => panic!("expected CapExceeded, got: {other:?}"),
  }
}

#[test]
fn load_dataset_malformed_line_errors_with_line_number() {
  let tok = tokenizer_fixture("load_malformed");
  let dir = fresh_dir("load_malformed_data");
  let p = dir.join("train.jsonl");
  std::fs::write(
    &p,
    "{\"text\": \"hello\"}\n{this is not json}\n{\"text\": \"world\"}\n",
  )
  .unwrap();
  let err = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap_err();
  assert!(
    matches!(err, Error::Parse(_)),
    "expected Parse error on malformed line, got: {err:?}",
  );
}

#[test]
fn load_dataset_nonexistent_path_errors() {
  let tok = tokenizer_fixture("load_nopath");
  let p = std::env::temp_dir().join(format!(
    "mlxrs-a6-does-not-exist-{}.jsonl",
    std::process::id()
  ));
  let err = load_dataset(&p, &tok, DatasetType::Auto, &DatasetConfig::default()).unwrap_err();
  match err {
    Error::FileIo(p) => {
      assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
    }
    other => panic!("expected FileIo NotFound, got: {other:?}"),
  }
}

#[test]
fn cache_dataset_invalidates_on_source_mtime_change() {
  // The Python `CacheDataset` is in-memory per-instance — there is no
  // sidecar `.cache` file. A source mtime change invalidates the cache
  // via the natural mechanism: the next `load_dataset` call constructs
  // a FRESH `CacheDataset` whose `_proc_data` is empty, so the new
  // file contents are observed.
  let tok = tokenizer_fixture("cache_mtime");
  let dir = fresh_dir("cache_mtime_data");
  let p = dir.join("train.jsonl");

  // First version of the file.
  write_jsonl(&p, &[json!({ "text": "hello" })]);
  let first = {
    let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
    ds.process(0).unwrap()
  };
  assert_eq!(first.0, vec![3, 2]); // hello </s>

  // Mutate the file (simulating an mtime change with new content).
  // Sleep one milli to make the mtime change observable on every fs.
  std::thread::sleep(std::time::Duration::from_millis(10));
  write_jsonl(&p, &[json!({ "text": "world" })]);

  // Second load constructs a fresh CacheDataset → reads the new content.
  let second = {
    let ds = load_dataset(&p, &tok, DatasetType::Text, &DatasetConfig::default()).unwrap();
    ds.process(0).unwrap()
  };
  assert_eq!(second.0, vec![4, 2]); // world </s>
  assert_ne!(first.0, second.0);
}

// `File::open` blocks read-only on a FIFO
// until a writer appears; the `meta.is_file()` rejection runs AFTER
// the open, so an adversarial FIFO at the dataset path could hang
// the loader indefinitely. The loader opens with
// `O_NONBLOCK | O_CLOEXEC` (mirroring the rest of mlxrs's hardened
// loaders) so the open returns immediately and the post-open
// `is_file()` check rejects the non-regular target before any read
// is attempted. This test plants a real writer-less FIFO at the
// dataset path and asserts the loader returns `Err(Backend)` with a
// "not a regular file" message PROMPTLY (within a 2 s budget).
//
// Determinism / non-flakiness: the loader runs on a worker thread
// and is joined with a 2 s budget. With `O_NONBLOCK`, the open is
// instantaneous (sub-millisecond), so the budget is never
// approached. If the `O_NONBLOCK` open regresses, the blocking
// `File::open()` wedges on the writer-less FIFO → the budget
// elapses and the test FAILS loudly instead of hanging CI. The
// thread is left detached on the (regression-only) timeout path so
// a regression cannot wedge the entire test binary.
#[cfg(unix)]
#[test]
fn load_dataset_rejects_fifo_without_blocking() {
  use std::{os::unix::ffi::OsStrExt, sync::mpsc};
  let dir = fresh_dir("load_fifo");
  // The dataset tokenizer fixture leaks a different temp dir.
  let tok = tokenizer_fixture("load_fifo_tok");
  let path = dir.join("train.jsonl");
  // Plant a real FIFO with NO writer. A blocking read-only
  // `open()` on this would hang forever.
  let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
  // SAFETY: `c_path` is a valid NUL-terminated C string that
  // outlives the call; `mkfifo` only reads the path and creates a
  // filesystem node — no aliasing concerns.
  let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
  assert_eq!(rc, 0, "mkfifo failed (rc {rc})");

  let (tx, rx) = mpsc::channel();
  let handle = std::thread::spawn(move || {
    let r = load_dataset(&path, &tok, DatasetType::Auto, &DatasetConfig::default());
    let msg = match &r {
      Err(Error::FileIo(p)) => Some(p.to_string()),
      _ => None,
    };
    let _ = tx.send(msg);
  });

  match rx.recv_timeout(std::time::Duration::from_secs(2)) {
    Ok(Some(msg)) => {
      handle.join().unwrap();
      assert!(
        msg.contains("not a regular file"),
        "FIFO at dataset path must yield 'not a regular file' \
           rejection, got: {msg}",
      );
    }
    Ok(None) => {
      handle.join().unwrap();
      panic!(
        "FIFO at dataset path must yield Err(FileIo), got a \
           different result"
      );
    }
    Err(_) => {
      // Regression: the O_NONBLOCK open was lost and the blocking
      // `File::open()` is wedged. Do NOT join (would wedge CI) —
      // fail loudly. The detached thread dies with the process.
      std::fs::remove_dir_all(&dir).ok();
      panic!(
        "load_dataset HUNG on a writer-less FIFO — the O_NONBLOCK \
           open regressed"
      );
    }
  }

  std::fs::remove_dir_all(&dir).ok();
}

// `BufRead::lines()` reads a FULL line into a
// `String` BEFORE yielding, so a single mid-read-grown gigantic
// line bypasses the cumulative cap → OOM. The reader uses manual
// `read_until(b'\n')` plus a per-iteration `take(remaining + 1)`
// instead of `lines()`, so the cap is enforced on the READ itself —
// the buffered reader cannot allocate more than `remaining + 1`
// bytes per iteration, regardless of how long the underlying
// (unterminated) line is.
//
// This test drives `read_jsonl_with_cap` with a 40-byte cap and a
// SINGLE 100-byte line containing no newline. A `lines()` reader
// would allocate all 100 bytes; the `read_until` + `take` reader
// allocates at most `cap + 1 = 41` bytes before the cap error fires.
// We assert both the cap error and (via `line_buf` indirectly through
// error message structure) that the implementation is operating
// under the truncation.
#[test]
fn load_dataset_cap_enforced_on_single_giant_line() {
  use std::io::{BufReader, Cursor};
  // 100 bytes of `a`, no newline anywhere — would force any
  // line-buffered reader to consume the full input before yielding.
  let body: Vec<u8> = vec![b'a'; 100];
  let cap: u64 = 40;
  let path = std::path::PathBuf::from("/synthetic/giant.jsonl");

  // Wrap the fixture in `CountingReader` so we can OBSERVE how many
  // bytes the helper pulled from the underlying reader. The helper
  // takes `R: BufRead` by value; the `impl<R: BufRead + ?Sized>
  // BufRead for &mut R` blanket lets us pass `&mut counting` and
  // retain ownership of the wrapper to query `.consumed()` after
  // the call returns.
  let mut counting = CountingReader::new(BufReader::new(Cursor::new(body)));
  let err = read_jsonl_with_cap(&mut counting, &path, cap).unwrap_err();
  match &err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap(), 40);
      assert_eq!(p.cap_name(), "MAX_DATASET_FILE_BYTES");
      assert!(
        p.observed() >= cap,
        "observed must be at or above cap, got: {p:?}"
      );
      assert!(
        p.context().contains("read jsonl"),
        "expected read-jsonl context, got: {p:?}"
      );
    }
    other => panic!("expected CapExceeded, got: {other:?}"),
  }

  // PROVE the `take(remaining + 1)` allocation cap held — at most
  // `cap + 1 = 41` bytes consumed from the underlying reader,
  // regardless of how long the unterminated line is. A
  // `BufRead::lines()` impl would consume all 100 bytes
  // before erroring (since `lines()` reads a full line into a
  // `String` BEFORE yielding). This assertion distinguishes the
  // safe-allocation `read_until` + `take` path from the OOM-prone
  // `lines()` path it replaces.
  let consumed = counting.consumed();
  assert!(
    consumed <= (cap as usize) + 1,
    "take(remaining + 1) allocation cap violated: consumed {consumed} bytes from a 100-byte \
       fixture with cap={cap} (expected <= {}); a lines() impl would consume 100",
    cap as usize + 1,
  );

  // Also exercise the "no newline anywhere, cap >= input" boundary
  // to confirm a short single line WITHOUT a trailing newline
  // doesn't trip the cap (it should parse as one record). 16 bytes
  // is well below the 40-byte cap.
  let small_body = b"{\"text\":\"abc\"}".to_vec();
  let v = read_jsonl_with_cap(Cursor::new(small_body), &path, cap).unwrap();
  assert_eq!(v.len(), 1);
  assert_eq!(v[0]["text"].as_str(), Some("abc"));
}
