//! CMU Pronouncing Dictionary lexicon — a 1:1 port of mlx-audio-swift's
//! [`CMUDictParser.swift`][parser] + [`CMUDictLoader.swift`][loader] +
//! [`InMemoryLexicon.swift`][mem].
//!
//! ## Format
//!
//! CMUDict ships as one row per line in the form
//! `WORD<spaces>PHONEME PHONEME …`. Variant pronunciations are flagged
//! `WORD(N)` (e.g. `the(2)  DH IY0`). Lines starting `;;;` are comments;
//! blank lines are skipped. The canonical cmusphinx `cmudict.dict`
//! additionally carries inline `#` comments on the pronunciation side
//! (`aalborg AO1 L B AO0 R G # place, danish`) — a pronunciation token
//! beginning with `#` starts a comment that runs to end-of-line and is
//! stripped before ARPAbet → IPA conversion. The parser is
//! whitespace-tolerant (single or double space between word and
//! pronunciation, the wild-style raw and pre-formatted dict files).
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
/// Inline `#` comments on the pronunciation side (canonical cmusphinx
/// `cmudict.dict` style, e.g. `aalborg AO1 L B AO0 R G # place, danish`)
/// are stripped: the first pronunciation token beginning with `#` and
/// everything after it are dropped. A `#` in the WORD column is part of
/// the word; a row whose pronunciation is ONLY a comment errors (no
/// phonemes left).
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

  // Inline `#` comment stripping — the canonical cmusphinx `cmudict.dict`
  // annotates some rows with a trailing comment on the pronunciation side
  // (`aalborg AO1 L B AO0 R G # place, danish`). A whitespace-delimited
  // token BEGINNING with `#` starts the comment; it and everything after
  // it are dropped before ARPAbet → IPA conversion (otherwise the strict
  // converter would reject `#` and fail the whole file).
  //
  // Deliberately narrow:
  // - only the pronunciation side is stripped — a `#` in the WORD column
  //   (0.7b-style `#hash-mark`) is part of the word (it sits before the
  //   first space, so it never reaches this tokenizer);
  // - a `#` glued to the TAIL of a token (`G#`) does not start a comment:
  //   the token is kept as-is so the strict converter still rejects it
  //   loudly with the row's line/word context.
  let arpabet: Vec<String> = pron_part
    .split(' ')
    .filter(|s| !s.is_empty())
    .take_while(|s| !s.starts_with('#'))
    .map(String::from)
    .collect();
  if arpabet.is_empty() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "CMUDict line",
      "pronunciation must be non-empty (after inline `#` comment stripping)",
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
mod tests;
