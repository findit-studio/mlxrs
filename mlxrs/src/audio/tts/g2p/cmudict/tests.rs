use super::*;

// === parse_line ===

// Mirrors `parsesBasicEntry`.
#[test]
fn parses_basic_entry() {
  let entry = parse_line("hello  HH AH0 L OW1", 1)
    .unwrap()
    .expect("entry");
  assert_eq!(entry.word(), "hello");
  assert_eq!(entry.arpabet(), ["HH", "AH0", "L", "OW1"]);
  assert_eq!(entry.variant(), None);
}

// Mirrors `parsesVariantEntry`.
#[test]
fn parses_variant_entry() {
  let entry = parse_line("the(2)  DH IY0", 1).unwrap().expect("entry");
  assert_eq!(entry.word(), "the");
  assert_eq!(entry.arpabet(), ["DH", "IY0"]);
  assert_eq!(entry.variant(), Some(2));
}

// Mirrors `skipsCommentLines`.
#[test]
fn skips_comment_lines() {
  assert_eq!(parse_line(";;; this is a comment", 1).unwrap(), None);
}

// Mirrors `skipsEmptyLines`.
#[test]
fn skips_empty_lines() {
  assert_eq!(parse_line("", 1).unwrap(), None);
  assert_eq!(parse_line("   ", 1).unwrap(), None);
}

// Mirrors `handlesSinglePhoneme`.
#[test]
fn handles_single_phoneme() {
  let entry = parse_line("a  AH0", 1).unwrap().expect("entry");
  assert_eq!(entry.word(), "a");
  assert_eq!(entry.arpabet(), ["AH0"]);
}

#[test]
fn lowercases_uppercase_word() {
  let entry = parse_line("HELLO  HH AH0", 1).unwrap().expect("entry");
  assert_eq!(entry.word(), "hello");
}

// Malformed row test — mlxrs surfaces an error (with line number) where
// swift silently drops the line. The error carries the offending line
// number so a bulk loader can point the caller at the bad row.
#[test]
fn malformed_line_errors_with_line_number() {
  let err = parse_line("nopronunciation", 42).unwrap_err();
  let Error::OutOfRange(payload) = &err else {
    panic!("malformed parse_line must be OutOfRange, got: {err:?}");
  };
  assert_eq!(
    payload.value(),
    "42",
    "OutOfRange payload value must carry the 1-indexed line number"
  );
}

// === parse (bulk) ===

// Mirrors `parsesBulkText`.
#[test]
fn parses_bulk_text() {
  let text = ";;; comment\n\
                hello  HH AH0 L OW1\n\
                world  W ER1 L D\n\
                the  DH AH0\n\
                the(2)  DH IY0";
  let entries = parse(text, false).unwrap();
  assert_eq!(entries.len(), 4);
  assert_eq!(entries[0].word(), "hello");
  assert_eq!(entries[3].variant(), Some(2));
}

// Mirrors `filtersPrimaryOnly`.
#[test]
fn filters_primary_only() {
  let text = "the  DH AH0\n\
                the(2)  DH IY0\n\
                hello  HH AH0 L OW1";
  let entries = parse(text, true).unwrap();
  assert_eq!(entries.len(), 2);
  assert!(entries.iter().all(|e| e.variant().is_none()));
}

#[test]
fn bulk_propagates_line_number_on_first_error() {
  let text = "hello  HH AH0\nbadline\nworld  W ER1 L D";
  let err = parse(text, false).unwrap_err();
  let Error::OutOfRange(payload) = &err else {
    panic!("bulk parse first error must be OutOfRange, got: {err:?}");
  };
  assert_eq!(
    payload.value(),
    "2",
    "OutOfRange payload value must carry the 1-indexed line number (2 for the malformed `badline` row)"
  );
}

// === CMUDict + Lexicon ===

fn fixture_dict() -> CMUDict {
  let raw = parse(
    "hello  HH AH0 L OW1\n\
       world  W ER1 L D\n\
       the  DH AH0\n\
       phone  F OW1 N",
    true,
  )
  .unwrap();
  CMUDict::from_raw_entries(raw).unwrap()
}

#[test]
fn lookup_returns_entry_for_known_word() {
  let dict = fixture_dict();
  let entry = dict.lookup("hello").expect("hello in dict");
  assert_eq!(entry.grapheme(), "hello");
  assert_eq!(entry.phonemes_slice(), ["h", "ə", "l", "oʊ"]);
}

#[test]
fn lookup_is_case_insensitive() {
  let dict = fixture_dict();
  assert!(dict.lookup("HELLO").is_some());
  assert!(dict.lookup("Hello").is_some());
  assert!(dict.lookup("hello").is_some());
}

#[test]
fn lookup_returns_none_for_unknown() {
  let dict = fixture_dict();
  assert!(dict.lookup("xyzzyplugh").is_none());
}

#[test]
fn from_entries_is_case_insensitive() {
  let dict = CMUDict::from_entries(vec![LexiconEntry::new(
    "HELLO".to_string(),
    vec!["h".into(), "ə".into(), "l".into(), "oʊ".into()],
  )]);
  assert!(dict.lookup("hello").is_some());
  assert_eq!(dict.len(), 1);
  assert!(!dict.is_empty());
}

/// `from_raw_entries` now uses the STRICT ARPAbet conversion path: an
/// unknown token mid-sequence fails the build with a backend error
/// carrying the word + offending token + 1-indexed line number.
///
/// Inverts the previous (silent-skip) behaviour deliberately — a dropped
/// token silently corrupted lexicon entries (empty / wrong-length
/// pronunciation, blocking the lexicon-first / neural-fallback pattern).
#[test]
fn from_raw_entries_strict_rejects_unknown_arpabet() {
  let raw = vec![RawEntry::new(
    "weird",
    vec!["HH".into(), "XX".into(), "L".into()],
    None,
    7,
  )];
  let err = CMUDict::from_raw_entries(raw).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("weird"), "expected word in {msg:?}");
  assert!(msg.contains("XX"), "expected offending token in {msg:?}");
  assert!(msg.contains("line 7"), "expected line number in {msg:?}");
}

// === Loader (file-system) ===

fn temp_dir(name: &str) -> std::path::PathBuf {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_audio_g2p_cmudict_{}_{}",
    std::process::id(),
    name
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

#[test]
fn loader_reads_cmudict_dict_from_directory() {
  let dir = temp_dir("loader_reads");
  fs::write(
    dir.join("cmudict.dict"),
    ";;; small fixture\n\
       hello  HH AH0 L OW1\n\
       world  W ER1 L D\n\
       the  DH AH0\n\
       the(2)  DH IY0\n",
  )
  .unwrap();

  let dict = CMUDictLoader::load(&dir).unwrap();
  assert!(dict.lookup("hello").is_some());
  assert!(dict.lookup("world").is_some());
  let the = dict.lookup("the").unwrap();
  // primary_only=true → the(2) variant is filtered out, the primary
  // (DH AH0 → ð ə) wins.
  assert_eq!(the.phonemes_slice(), ["ð", "ə"]);
}

#[test]
fn loader_errors_on_missing_file() {
  let dir = temp_dir("missing_file");
  let err = CMUDictLoader::load(&dir).unwrap_err();
  let Error::MissingKey(payload) = &err else {
    panic!("missing cmudict.dict must be MissingKey, got: {err:?}");
  };
  assert!(
    payload.key().ends_with("cmudict.dict"),
    "MissingKey payload key must name the missing cmudict.dict file, got: {}",
    payload.key()
  );
}

#[test]
fn loader_propagates_malformed_line_error_with_line_number() {
  let dir = temp_dir("malformed_line");
  fs::write(
    dir.join("cmudict.dict"),
    "hello  HH AH0\n\
       badline\n\
       world  W ER1 L D\n",
  )
  .unwrap();

  let err = CMUDictLoader::load(&dir).unwrap_err();
  let Error::OutOfRange(payload) = &err else {
    panic!("loader malformed-line error must be OutOfRange, got: {err:?}");
  };
  assert_eq!(
    payload.value(),
    "2",
    "OutOfRange payload value must carry the 1-indexed line number (2 for the malformed `badline` row)"
  );
}

/// Latin-1 fallback — bytes that aren't valid UTF-8 (e.g. an isolated
/// 0xE9 for é) decode as Latin-1.
#[test]
fn loader_decodes_latin1_when_utf8_invalid() {
  let dir = temp_dir("latin1");
  // 0xE9 is é in Latin-1 (but an invalid UTF-8 lead byte in isolation).
  let mut bytes: Vec<u8> = Vec::from(b"caf\xE9  K AE1 F EY0\n" as &[u8]);
  // Make sure UTF-8 decode fails (the leading bytes are ASCII but the
  // 0xE9 mid-stream is not a valid UTF-8 sequence start).
  assert!(std::str::from_utf8(&bytes).is_err());
  fs::write(dir.join("cmudict.dict"), &bytes).unwrap();
  bytes.clear();

  let dict = CMUDictLoader::load(&dir).unwrap();
  // After Latin-1 decode the word is "café" lowercased → "café".
  assert!(
    dict.lookup("café").is_some(),
    "café should be present after Latin-1 fallback"
  );
}

// ============================================================
// Strict variant-parser tests — every malformed `WORD(...)` shape
// surfaces as `Err(Backend)` carrying the 1-indexed line number + the
// offending token. Previously these slipped through and (often) wrote
// a malformed `variant: None` entry as if it were the canonical
// pronunciation, silently corrupting the lexicon.
// ============================================================

#[test]
fn cmudict_parse_line_rejects_non_digit_variant_paren() {
  let err = parse_line("the(x)  DH IY0", 13).unwrap_err();
  let msg = err.to_string();
  assert!(
    msg.contains("the(x)"),
    "expected offending token in {msg:?}"
  );
  assert!(msg.contains("line 13"), "expected line number in {msg:?}");
}

#[test]
fn cmudict_parse_line_rejects_empty_paren() {
  let err = parse_line("the()  DH IY0", 21).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("the()"), "expected offending token in {msg:?}");
  assert!(msg.contains("line 21"), "expected line number in {msg:?}");
}

#[test]
fn cmudict_parse_line_rejects_trailing_garbage_after_paren() {
  let err = parse_line("the(2)junk  DH IY0", 34).unwrap_err();
  let msg = err.to_string();
  assert!(
    msg.contains("the(2)junk"),
    "expected offending token in {msg:?}"
  );
  assert!(msg.contains("line 34"), "expected line number in {msg:?}");
}

#[test]
fn cmudict_parse_line_rejects_empty_word_before_paren() {
  let err = parse_line("(2)  DH IY0", 55).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("(2)"), "expected offending token in {msg:?}");
  assert!(msg.contains("line 55"), "expected line number in {msg:?}");
}

// ============================================================
// Strict ARPAbet-conversion tests — driven through the loader so we
// exercise the same path real callers hit (parse → from_raw_entries),
// and so the surfaced error carries the 1-indexed line number.
// ============================================================

#[test]
fn cmudict_loader_rejects_unknown_arpabet_token() {
  let dir = temp_dir("loader_unknown_arpabet");
  // Two real lines so the offending row is on line 2 (1-indexed):
  fs::write(
    dir.join("cmudict.dict"),
    "hello  HH AH0 L OW1\n\
       word  XX YY\n",
  )
  .unwrap();
  let err = CMUDictLoader::load(&dir).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("word"), "expected word in {msg:?}");
  assert!(msg.contains("XX"), "expected offending token in {msg:?}");
  assert!(msg.contains("line 2"), "expected line number in {msg:?}");
}

/// A row whose `arpabet` is non-empty pre-conversion but reduces to
/// nothing post-conversion must error (an empty pronunciation in a
/// lexicon is invalid by definition). This is a belt-and-suspenders
/// guard — currently unreachable because `try_convert_sequence_strict`
/// errors on the first unknown — but it locks the contract in case a
/// future refactor relaxes the strict path.
#[test]
fn cmudict_loader_rejects_empty_phonemes_after_conversion() {
  let raw = vec![RawEntry::new("ghostword", Vec::new(), None, 17)];
  let err = CMUDict::from_raw_entries(raw).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("ghostword"), "expected word in {msg:?}");
  assert!(
    msg.contains("empty"),
    "expected 'empty' diagnostic in {msg:?}"
  );
  assert!(msg.contains("line 17"), "expected line number in {msg:?}");
}

/// Sanity-check: the strict fix doesn't break the happy path. A
/// well-formed `WORD  PHONEMES` row still loads + maps correctly.
#[test]
fn cmudict_loader_accepts_well_known_word_after_fix() {
  let dir = temp_dir("loader_well_known_after_fix");
  fs::write(dir.join("cmudict.dict"), "hello  HH AH0 L OW1\n").unwrap();
  let dict = CMUDictLoader::load(&dir).unwrap();
  let entry = dict.lookup("hello").expect("hello in dict");
  assert_eq!(entry.phonemes_slice(), ["h", "ə", "l", "oʊ"]);
}
