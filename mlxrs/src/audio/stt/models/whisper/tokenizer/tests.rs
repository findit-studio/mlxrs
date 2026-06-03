use super::*;
use crate::tokenizer::Tokenizer;
use serde_json::json;
use std::path::{Path, PathBuf};

/// The Whisper special tokens, with the ids this fixture assigns them. A few
/// language tokens (`<|en|>`, `<|zh|>`, `<|de|>`) are included so
/// `language_token` / `all_language_tokens` are exercisable.
const SPECIALS: &[(&str, u32)] = &[
  ("hello", 0),
  ("world", 1),
  ("<|endoftext|>", 2),
  ("<|startoftranscript|>", 3),
  ("<|en|>", 4),
  ("<|zh|>", 5),
  ("<|de|>", 6),
  ("<|translate|>", 7),
  ("<|transcribe|>", 8),
  ("<|startoflm|>", 9),
  ("<|startofprev|>", 10),
  ("<|nospeech|>", 11),
  ("<|notimestamps|>", 12),
  ("<|0.00|>", 13),
];

fn fresh_dir(tag: &str) -> PathBuf {
  let dir = std::env::temp_dir().join(format!("mlxrs_whisper_tok_{}_{}", std::process::id(), tag));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Build a Whisper-shaped tokenizer (WordLevel + the special tokens as
/// added_tokens) into `dir` and load it.
fn write_tokenizer(dir: &Path) -> Tokenizer {
  let vocab: serde_json::Map<String, serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();

  // Mark the `<|…|>` markers special (and `<|endoftext|>`); the plain word
  // tokens stay non-special.
  let added_tokens: Vec<serde_json::Value> = SPECIALS
    .iter()
    .map(|(tok, id)| {
      let special = tok.starts_with("<|");
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": special
      })
    })
    .collect();

  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });

  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  Tokenizer::from_path(dir, None).unwrap()
}

fn fixture(tag: &str) -> Tokenizer {
  write_tokenizer(&fresh_dir(tag))
}

#[test]
fn resolves_special_tokens_by_string() {
  let tok = fixture("specials");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  assert_eq!(w.sot(), 3);
  assert_eq!(w.eot(), 2);
  assert_eq!(w.transcribe(), 8);
  assert_eq!(w.translate(), 7);
  assert_eq!(w.sot_prev(), 10);
  assert_eq!(w.no_timestamps(), 12);
  assert_eq!(w.no_speech(), 11);
  assert_eq!(w.timestamp_begin(), 13);
}

#[test]
fn sot_sequence_multilingual_transcribe() {
  let tok = fixture("sot_multi_transcribe");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  // (sot, <|en|>, <|transcribe|>) = (3, 4, 8).
  assert_eq!(w.sot_sequence(), vec![3, 4, 8]);
}

#[test]
fn sot_sequence_multilingual_translate() {
  let tok = fixture("sot_multi_translate");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("de"), Task::Translate).unwrap();
  // (sot, <|de|>, <|translate|>) = (3, 6, 7).
  assert_eq!(w.sot_sequence(), vec![3, 6, 7]);
}

#[test]
fn sot_sequence_english_only_is_sot_alone() {
  let tok = fixture("sot_en_only");
  // multilingual=false → sot_sequence is just (sot,).
  let w = HFTokenizerWrapper::new(&tok, false, 1, None, Task::Transcribe).unwrap();
  assert_eq!(w.sot_sequence(), vec![3]);
}

#[test]
fn sot_sequence_including_notimestamps_appends() {
  let tok = fixture("sot_notimestamps");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  // (sot, <|en|>, <|transcribe|>, <|notimestamps|>) = (3, 4, 8, 12).
  assert_eq!(w.sot_sequence_including_notimestamps(), vec![3, 4, 8, 12]);
}

#[test]
fn language_passed_as_name_is_normalized() {
  let tok = fixture("lang_name");
  // "german" → "de"; language_token resolves <|de|> = 6.
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("german"), Task::Transcribe).unwrap();
  assert_eq!(w.language(), "de");
  assert_eq!(w.language_token(), 6);
}

#[test]
fn language_none_defaults_to_en() {
  let tok = fixture("lang_default");
  let w = HFTokenizerWrapper::new(&tok, true, 99, None, Task::Transcribe).unwrap();
  assert_eq!(w.language(), "en");
  assert_eq!(w.language_token(), 4);
}

#[test]
fn task_token_selects_by_task() {
  let tok = fixture("task_token");
  let transcribe = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  let translate = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Translate).unwrap();
  assert_eq!(transcribe.task_token(), 8);
  assert_eq!(translate.task_token(), 7);
}

#[test]
fn all_language_tokens_takes_first_n() {
  let tok = fixture("all_lang_tokens");
  // num_languages=3 → first three LANGUAGES codes are en, zh, de → ids 4,5,6.
  let w = HFTokenizerWrapper::new(&tok, true, 3, Some("en"), Task::Transcribe).unwrap();
  assert_eq!(w.all_language_tokens(), vec![4, 5, 6]);
  assert_eq!(w.all_language_codes(), vec!["en", "zh", "de"]);
}

#[test]
fn all_language_candidates_stay_aligned_when_a_language_is_missing() {
  // A tokenizer whose `<|zh|>` (the SECOND language) is absent — its
  // `convert_token_to_id` falls back to the `unk` id. `all_language_tokens`
  // filters that entry out (→ [en_id, de_id]) while `all_language_codes`
  // keeps the full prefix (→ [en, zh, de]); a positional zip of those two would
  // pair the de token id with the "zh" code. `all_language_candidates` pairs the
  // code WITH its id in one pass, so the surviving languages keep their own
  // codes and the missing one simply drops out.
  let dir = fresh_dir("lang_misalign");
  // Required Whisper specials + en/de language tokens, but NO `<|zh|>`.
  // `<|endoftext|>` (id 2) is both eos and unk, so a missing `<|zh|>` resolves
  // to id 2.
  let present: &[(&str, u32)] = &[
    ("hello", 0),
    ("world", 1),
    ("<|endoftext|>", 2),
    ("<|startoftranscript|>", 3),
    ("<|en|>", 4),
    ("<|de|>", 6),
    ("<|translate|>", 7),
    ("<|transcribe|>", 8),
    ("<|startoflm|>", 9),
    ("<|startofprev|>", 10),
    ("<|nospeech|>", 11),
    ("<|notimestamps|>", 12),
    ("<|0.00|>", 13),
  ];
  let vocab: serde_json::Map<String, serde_json::Value> = present
    .iter()
    .map(|(tok, id)| (tok.to_string(), json!(id)))
    .collect();
  let added_tokens: Vec<serde_json::Value> = present
    .iter()
    .map(|(tok, id)| {
      json!({
        "id": id, "content": tok, "single_word": false, "lstrip": false,
        "rstrip": false, "normalized": false, "special": tok.starts_with("<|")
      })
    })
    .collect();
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": added_tokens,
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
  });
  let cfg = json!({ "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>" });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();

  let w = HFTokenizerWrapper::new(&tok, true, 3, Some("en"), Task::Transcribe).unwrap();

  // The two separate vectors drift: tokens drop zh (→ [4, 6]) while codes keep
  // it (→ [en, zh, de]). A naive zip would pair (6, "zh") — the de id mislabeled.
  assert_eq!(w.all_language_tokens(), vec![4, 6]);
  assert_eq!(w.all_language_codes(), vec!["en", "zh", "de"]);

  // The paired pass keeps each surviving language with its OWN code, and drops
  // the missing one entirely — no positional drift.
  assert_eq!(w.all_language_candidates(), vec![("en", 4), ("de", 6)]);
}

#[test]
fn decode_filters_timestamp_tokens() {
  let tok = fixture("decode_filter");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  // tokens: [hello(0), <|0.00|>(13), world(1)] → timestamp 13 dropped.
  let text = w.decode(&[0, 13, 1], true).unwrap();
  assert!(text.contains("hello"), "got {text:?}");
  assert!(text.contains("world"), "got {text:?}");
}

#[test]
fn missing_special_token_errors() {
  // A tokenizer without the Whisper specials must fail HFTokenizerWrapper::new.
  let dir = fresh_dir("missing_specials");
  let vocab = json!({ "<unk>": 0, "hello": 1 });
  let tokenizer_json = json!({
    "version": "1.0",
    "added_tokens": [{
      "id": 0, "content": "<unk>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true
    }],
    "normalizer": null,
    "pre_tokenizer": { "type": "Whitespace" },
    "post_processor": null,
    "decoder": null,
    "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
  });
  std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
  let tok = Tokenizer::from_path(&dir, None).unwrap();
  let err = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap_err();
  assert!(
    matches!(err, Error::MissingKey(_)),
    "expected MissingKey, got {err:?}"
  );
}

#[test]
fn to_language_code_handles_codes_names_aliases() {
  // Already a code.
  assert_eq!(to_language_code("en"), Some("en"));
  // Full name.
  assert_eq!(to_language_code("german"), Some("de"));
  // Alias.
  assert_eq!(to_language_code("mandarin"), Some("zh"));
  assert_eq!(to_language_code("flemish"), Some("nl"));
  // Unknown.
  assert_eq!(to_language_code("klingon"), None);
}

#[test]
fn languages_table_matches_reference() {
  // The reference `LANGUAGES` dict has 100 entries (v3 added Cantonese as the
  // 100th); the `num_languages=99` constructor default is the v2 slice count,
  // a separate concept.
  assert_eq!(LANGUAGES.len(), 100);
  // First and last entries (order is load-bearing).
  assert_eq!(LANGUAGES[0], ("en", "english"));
  assert_eq!(LANGUAGES[99], ("yue", "cantonese"));
}

#[test]
fn split_to_word_tokens_merges_spaceless_subwords_and_splits_specials() {
  // The fixture's per-token decode yields no leading space, so consecutive
  // plain tokens merge into one word (`_split_tokens_on_spaces`: a subword that
  // is not special / leading-space / punctuation / first merges into the
  // previous word). A special token (`>= eot`) always starts its own word.
  let tok = fixture("split_words");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  let (words, word_tokens) = w.split_to_word_tokens(&[0, 1, 2]).unwrap();
  assert_eq!(
    words,
    vec!["helloworld".to_string(), "<|endoftext|>".to_string()]
  );
  assert_eq!(word_tokens, vec![vec![0u32, 1], vec![2]]);
  // The word_tokens partition the input exactly (no token dropped / duplicated).
  let flat: Vec<u32> = word_tokens.into_iter().flatten().collect();
  assert_eq!(flat, vec![0, 1, 2]);
}

#[test]
fn split_to_word_tokens_eot_starts_new_word() {
  // A plain token followed by a special (`>= eot`) token: the plain token is the
  // first word, and the special starts its own word (`special = tokens[0] >=
  // eot`). The partition is preserved exactly.
  let tok = fixture("split_eot");
  let w = HFTokenizerWrapper::new(&tok, true, 99, Some("en"), Task::Transcribe).unwrap();
  let (words, word_tokens) = w.split_to_word_tokens(&[0, 2]).unwrap();
  assert_eq!(word_tokens, vec![vec![0u32], vec![2]]);
  assert_eq!(words.len(), 2);
  assert_eq!(words[1], "<|endoftext|>");
}
