//! CMU Pronouncing Dictionary lexicon — a 1:1 port of mlx-audio-swift's
//! [`CMUDictParser.swift`][parser] + [`CMUDictLoader.swift`][loader] +
//! [`InMemoryLexicon.swift`][mem].
//!
//! ## Format
//!
//! CMUDict ships as one row per line in the form
//! `WORD<spaces>PHONEME PHONEME …`. Variant pronunciations are flagged
//! `WORD(N)` (e.g. `the(2)  DH IY0`). Lines starting `;;;` are comments;
//! blank lines are skipped. The parser is whitespace-tolerant (single or
//! double space between word and pronunciation, the wild-style raw and
//! pre-formatted dict files).
//!
//! ## Local-file-only
//!
//! [`CMUDictLoader::load`] takes a directory path and reads the
//! `cmudict.dict` file inside it (no HF Hub, no network). The bytes are
//! decoded as UTF-8 first, falling back to Latin-1 (the upstream wild dict
//! is mostly ASCII but ships with a handful of accented loanwords; the
//! swift loader makes the same fallback).
//!
//! [parser]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/CMUDict/CMUDictParser.swift
//! [loader]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/CMUDict/CMUDictLoader.swift
//! [mem]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/InMemoryLexicon.swift

use std::{collections::HashMap, fs, path::Path};

use smol_str::format_smolstr;

use crate::{
  audio::tts::g2p::{
    arpabet,
    types::{Lexicon, LexiconEntry},
  },
  error::{Error, FileIoPayload, FileOp, MissingKeyPayload, OutOfRangePayload, Result},
};

/// One row of a parsed CMUDict file, BEFORE ARPAbet→IPA conversion.
/// Mirrors swift's `CMUDictRawEntry` (with one mlxrs extension:
/// `line_number` is carried through so the lexicon converter can surface
/// per-row errors with their source-line position).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawEntry {
  /// The lowercase word (the variant suffix `(N)` is stripped).
  word: String,
  /// The ARPAbet phoneme sequence as captured from the source line
  /// (still carries stress digits).
  arpabet: Vec<String>,
  /// `Some(n)` for variant pronunciations (`word(n)` syntax); `None` for
  /// the primary entry.
  variant: Option<u32>,
  /// The 1-indexed source-line number this row came from. Carried so the
  /// downstream ARPAbet → IPA converter can surface a malformed-token
  /// error with the offending line position.
  line_number: usize,
}

impl RawEntry {
  /// Construct a [`RawEntry`] from all its fields.
  pub fn new(
    word: impl Into<String>,
    arpabet: Vec<String>,
    variant: Option<u32>,
    line_number: usize,
  ) -> Self {
    Self {
      word: word.into(),
      arpabet,
      variant,
      line_number,
    }
  }

  /// The lowercase word (variant suffix `(N)` stripped).
  #[inline(always)]
  pub fn word(&self) -> &str {
    &self.word
  }

  /// The ARPAbet phoneme sequence (still carries stress digits).
  #[inline(always)]
  pub fn arpabet(&self) -> &[String] {
    &self.arpabet
  }

  /// `Some(n)` for variant pronunciations; `None` for the primary entry.
  #[inline(always)]
  pub fn variant(&self) -> Option<u32> {
    self.variant
  }

  /// The 1-indexed source-line number.
  #[inline(always)]
  pub fn line_number(&self) -> usize {
    self.line_number
  }
}

/// Construct a `malformed-word` [`Error::OutOfRange`] tagged with the
/// 1-indexed `line_number`, the offending `word_token`, and a short
/// `reason` describing why the parse failed.
///
/// `reason` must be a `&'static str` so it can live in the typed
/// `OutOfRangePayload::requirement` field.
fn malformed_word(word_token: &str, line_number: usize, reason: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "CMUDict parse: malformed word token (expected WORD or WORD(N))",
    reason,
    format_smolstr!("line {line_number}: '{word_token}'"),
  ))
}

/// Strict parse of a CMUDict word token into its base spelling and
/// (optional) variant index.
///
/// Accepts exactly two shapes:
/// - `WORD` — no parens, returns `(WORD, None)`.
/// - `WORD(N)` — a non-empty base, an open paren, ≥1 ASCII digits, and a
///   close paren that is the LAST character of the token. Returns
///   `(WORD, Some(N))`.
///
/// Rejects every other shape (`the(x)`, `the()`, `the(2)junk`, `(2)`,
/// `WORD(`) with a backend error tagged by `line_number` so the bulk
/// parse can point the caller at the offending row.
fn parse_word_and_variant(word_token: &str, line_number: usize) -> Result<(&str, Option<u32>)> {
  let Some(open_idx) = word_token.find('(') else {
    // No paren → bare WORD. (`word_part.is_empty()` is already rejected by
    // the caller, so we don't re-check here.)
    return Ok((word_token, None));
  };

  // Close paren MUST be the last char of the token (no trailing garbage
  // after `WORD(N)`).
  if !word_token.ends_with(')') {
    return Err(malformed_word(
      word_token,
      line_number,
      "trailing characters after closing paren (or missing closing paren)",
    ));
  }
  // `find('(')` returns a byte index in an ASCII-only path, and `(` / `)`
  // are single-byte; safe to slice byte-wise.
  let base = &word_token[..open_idx];
  if base.is_empty() {
    return Err(malformed_word(
      word_token,
      line_number,
      "empty base word before opening paren",
    ));
  }
  // The token is at least `<base>(<...>)` with `base` non-empty and
  // ending in `)`, so `open_idx + 1 <= word_token.len() - 1`.
  let variant_str = &word_token[open_idx + 1..word_token.len() - 1];
  if variant_str.is_empty() {
    return Err(malformed_word(
      word_token,
      line_number,
      "empty variant index between parens",
    ));
  }
  if !variant_str.bytes().all(|b| b.is_ascii_digit()) {
    return Err(malformed_word(
      word_token,
      line_number,
      "variant index must be 1+ ASCII digits",
    ));
  }
  let variant = variant_str.parse::<u32>().map_err(|_| {
    malformed_word(
      word_token,
      line_number,
      "variant index overflows u32 (>4_294_967_295)",
    )
  })?;
  Ok((base, Some(variant)))
}

/// Parse a single CMUDict source line into a [`RawEntry`].
///
/// Returns `None` for blank lines and comment lines (those starting
/// `;;;`); returns `Err` for malformed rows (a non-empty, non-comment
/// line that is missing whitespace or has a non-word/non-pronunciation
/// shape after the split).
///
/// The error carries `line_number` so a bulk loader can surface the
/// offending line position to the caller.
pub fn parse_line(line: &str, line_number: usize) -> Result<Option<RawEntry>> {
  let trimmed = line.trim();
  if trimmed.is_empty() || trimmed.starts_with(";;;") {
    return Ok(None);
  }

  // Split on first ASCII space: word is the first token, pronunciation is
  // the rest (handles both raw `cmudict.dict` and double-space formats).
  let Some(first_space) = trimmed.find(' ') else {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "CMUDict line",
      "must contain whitespace between word and pronunciation",
      format_smolstr!("{line_number}"),
    )));
  };

  let word_part = &trimmed[..first_space];
  let pron_part = trimmed[first_space + 1..].trim();

  if word_part.is_empty() || pron_part.is_empty() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "CMUDict line",
      "word and pronunciation must both be non-empty",
      format_smolstr!("{line_number}"),
    )));
  }

  // Strict split of `WORD` / `WORD(N)`: any other shape errors.
  let (word_str, variant) = parse_word_and_variant(word_part, line_number)?;
  let word = word_str.to_lowercase();

  let arpabet: Vec<String> = pron_part
    .split(' ')
    .filter(|s| !s.is_empty())
    .map(String::from)
    .collect();
  if arpabet.is_empty() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "CMUDict line",
      "pronunciation must be non-empty",
      format_smolstr!("{line_number}"),
    )));
  }

  Ok(Some(RawEntry::new(word, arpabet, variant, line_number)))
}

/// Parse a CMUDict text blob into a list of [`RawEntry`]. Comment / blank
/// lines are skipped. If `primary_only` is set, variant pronunciations
/// (`word(n)` rows) are filtered out.
///
/// Returns the first malformed line's error (with the offending line
/// number); a single bad row fails the whole parse.
pub fn parse(text: &str, primary_only: bool) -> Result<Vec<RawEntry>> {
  let mut out = Vec::new();
  for (idx, line) in text.lines().enumerate() {
    // Line numbers are 1-indexed for human consumption.
    let line_number = idx + 1;
    if let Some(entry) = parse_line(line, line_number)?
      && (!primary_only || entry.variant().is_none())
    {
      out.push(entry);
    }
  }
  Ok(out)
}

/// In-memory CMU Pronouncing Dictionary lexicon — case-insensitive
/// grapheme → IPA phoneme-sequence map. Mirrors swift's `InMemoryLexicon`
/// (used by `CMUDictLoader`).
#[derive(Debug, Clone)]
pub struct CMUDict {
  /// Lowercase grapheme → entry.
  entries: HashMap<String, LexiconEntry>,
}

impl CMUDict {
  /// Build a [`CMUDict`] from a list of [`LexiconEntry`]. Duplicate
  /// graphemes are collapsed (last wins, matching swift's
  /// `Dictionary(uniqueKeysWithValues:)` semantics: the *primary-only*
  /// parse passes a deduplicated list, so the unique-keys assumption holds
  /// in normal use; on dup we overwrite rather than panic, to stay robust).
  #[must_use]
  pub fn from_entries(entries: impl IntoIterator<Item = LexiconEntry>) -> Self {
    let mut map = HashMap::new();
    for entry in entries {
      let key = entry.grapheme().to_lowercase();
      map.insert(key, entry);
    }
    Self { entries: map }
  }

  /// Build a [`CMUDict`] from the parsed raw entries. Each row's ARPAbet
  /// sequence is mapped to IPA via
  /// [`arpabet::try_convert_sequence_strict`] — the STRICT path: the first
  /// unknown ARPAbet token in any row fails the whole build with
  /// [`Error::OutOfRange`] tagged by the source-line position.
  ///
  /// Empty post-conversion pronunciations (the row's `arpabet` was
  /// non-empty pre-conversion but every token was rejected — currently
  /// unreachable because
  /// [`arpabet::try_convert_sequence_strict`] errors on the first
  /// unknown, but a future refactor could relax that) are likewise
  /// surfaced as a backend error: an empty pronunciation in a lexicon is
  /// invalid by definition (it would block the lexicon-first /
  /// neural-fallback pattern: a `Some` with an empty `phonemes` masks the
  /// fallback case).
  pub fn from_raw_entries(raw: impl IntoIterator<Item = RawEntry>) -> Result<Self> {
    let mut entries = Vec::new();
    for r in raw {
      let phonemes = arpabet::try_convert_sequence_strict(r.arpabet()).map_err(|bad| {
        Error::OutOfRange(OutOfRangePayload::new(
          "CMUDict ARPAbet token",
          "must be a known ARPAbet symbol",
          format_smolstr!(
            "line {}: word '{}' token '{}'",
            r.line_number(),
            r.word(),
            bad.token(),
          ),
        ))
      })?;
      if phonemes.is_empty() {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "CMUDict line: pronunciation after ARPAbet → IPA conversion",
          "must be non-empty",
          format_smolstr!("line {}: word '{}'", r.line_number(), r.word()),
        )));
      }
      entries.push(LexiconEntry::new(r.word().to_owned(), phonemes));
    }
    Ok(Self::from_entries(entries))
  }

  /// Number of unique graphemes in the lexicon.
  #[must_use]
  pub fn len(&self) -> usize {
    self.entries.len()
  }

  /// `true` iff the lexicon is empty.
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

impl Lexicon for CMUDict {
  fn lookup(&self, grapheme: &str) -> Option<&LexiconEntry> {
    self.entries.get(&grapheme.to_lowercase())
  }
}

/// Loader for CMUDict files on the local filesystem.
///
/// **Local-file-only** — no HF Hub, no network. Reads `cmudict.dict` from
/// the given directory, decodes UTF-8 first (falling back to Latin-1 for
/// the handful of accented loanwords in the wild dict), parses
/// `primary_only=true` (so the lookup returns the canonical pronunciation
/// for each grapheme, not a variant), and converts ARPAbet to IPA on the
/// way in.
pub struct CMUDictLoader;

impl CMUDictLoader {
  /// Load `cmudict.dict` from `directory` and return an in-memory
  /// [`CMUDict`].
  ///
  /// Errors:
  /// - [`Error::MissingKey`] if the `cmudict.dict` file is not present in
  ///   `directory`.
  /// - [`Error::FileIo`] if the file cannot be read.
  /// - [`Error::OutOfRange`] (with the 1-indexed line number) on a malformed
  ///   row.
  pub fn load(directory: &Path) -> Result<CMUDict> {
    let path = directory.join("cmudict.dict");
    if !path.exists() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "CMUDictLoader::load: required file not found",
        format_smolstr!("{}", path.display()),
      )));
    }

    let bytes = fs::read(&path).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "read",
        FileOp::Read,
        path.to_path_buf(),
        e,
      ))
    })?;

    // UTF-8 first, then Latin-1 (every byte sequence is valid Latin-1, so
    // this never fails — matching the swift loader's fallback chain).
    let text = match std::str::from_utf8(&bytes) {
      Ok(s) => s.to_owned(),
      Err(_) => bytes.iter().map(|&b| b as char).collect(),
    };

    let raw = parse(&text, true)?;
    CMUDict::from_raw_entries(raw)
  }
}

#[cfg(test)]
mod tests {
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
}
