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

// ============================================================
// Inline `#` comment stripping — the canonical cmusphinx
// `cmudict.dict` carries inline comments on the pronunciation side
// (22 rows in master, e.g. `aalborg AO1 L B AO0 R G # place, danish`).
// Pre-fix, the `#` and the comment words leaked into the ARPAbet token
// list, the strict converter rejected `#` as a BadArpabetToken, and the
// WHOLE canonical-file load failed on its first commented row (line 29).
// ============================================================

/// Verbatim line 29 of canonical cmusphinx/cmudict `cmudict.dict`.
#[test]
fn parse_line_strips_inline_hash_comment() {
  let entry = parse_line("aalborg AO1 L B AO0 R G # place, danish", 29)
    .unwrap()
    .expect("entry");
  assert_eq!(entry.word(), "aalborg");
  assert_eq!(entry.arpabet(), ["AO1", "L", "B", "AO0", "R", "G"]);
  assert_eq!(entry.variant(), None);
}

/// Verbatim line 28252 of canonical cmudict.dict — a `(N)` variant row
/// carrying an inline comment. The variant suffix must still parse.
#[test]
fn parse_line_strips_inline_comment_on_variant_row() {
  let entry = parse_line("dail(2) D OY1 L # org, irish", 28252)
    .unwrap()
    .expect("entry");
  assert_eq!(entry.word(), "dail");
  assert_eq!(entry.variant(), Some(2));
  assert_eq!(entry.arpabet(), ["D", "OY1", "L"]);
}

/// A `#` glued directly to comment text (` #place`) still starts a
/// comment — the comment marker is any whitespace-delimited token that
/// BEGINS with `#`, not only a lone `#`.
#[test]
fn parse_line_strips_hash_comment_glued_to_text() {
  let entry = parse_line("aalborg AO1 L B AO0 R G #place,danish", 1)
    .unwrap()
    .expect("entry");
  assert_eq!(entry.arpabet(), ["AO1", "L", "B", "AO0", "R", "G"]);
}

/// After comment stripping, the converted IPA must be identical to the
/// same row WITHOUT the comment (the comment must not perturb phonemes).
#[test]
fn inline_comment_entry_converts_same_as_uncommented() {
  let commented = parse("aalborg AO1 L B AO0 R G # place, danish", true).unwrap();
  let plain = parse("aalborg AO1 L B AO0 R G", true).unwrap();
  let dict_commented = CMUDict::from_raw_entries(commented).unwrap();
  let dict_plain = CMUDict::from_raw_entries(plain).unwrap();
  let a = dict_commented
    .lookup("aalborg")
    .expect("aalborg (commented)");
  let b = dict_plain.lookup("aalborg").expect("aalborg (plain)");
  assert_eq!(a.phonemes_slice(), b.phonemes_slice());
  assert_eq!(a.phonemes_slice(), ["ɔ", "l", "b", "ɔ", "ɹ", "ɡ"]);
}

/// A row whose pronunciation is ONLY a comment has no phonemes — that is
/// a malformed row and must error (with the 1-indexed line number), not
/// silently produce an empty-pronunciation lexicon entry.
#[test]
fn parse_line_comment_only_pronunciation_errors() {
  let err = parse_line("ghost # comment only", 9).unwrap_err();
  let Error::OutOfRange(payload) = &err else {
    panic!("comment-only pronunciation must be OutOfRange, got: {err:?}");
  };
  assert_eq!(
    payload.value(),
    "9",
    "OutOfRange payload value must carry the 1-indexed line number"
  );
}

/// A `#` glued to the TAIL of a token (`G#`) is NOT a comment start: the
/// token is kept as-is so the strict ARPAbet converter still rejects it
/// loudly (preserving the anti-corruption value of the strict path), and
/// the error still carries the word + token + 1-indexed line number.
#[test]
fn hash_glued_to_token_tail_is_not_a_comment_and_fails_strict() {
  let entry = parse_line("weird AO1 G# L", 5).unwrap().expect("entry");
  assert_eq!(entry.arpabet(), ["AO1", "G#", "L"]);
  let err = CMUDict::from_raw_entries(vec![entry]).unwrap_err();
  let msg = err.to_string();
  assert!(msg.contains("weird"), "expected word in {msg:?}");
  assert!(msg.contains("G#"), "expected offending token in {msg:?}");
  assert!(msg.contains("line 5"), "expected line number in {msg:?}");
}

/// A `#` in the WORD column is part of the word, never a comment
/// (cmudict-0.7b ships punctuation names like `#hash-mark`). Only the
/// pronunciation side is comment-stripped.
#[test]
fn hash_in_word_column_is_not_a_comment() {
  let entry = parse_line("#hash-mark HH AE1 SH M AA2 R K", 3)
    .unwrap()
    .expect("entry");
  assert_eq!(entry.word(), "#hash-mark");
  assert_eq!(entry.arpabet(), ["HH", "AE1", "SH", "M", "AA2", "R", "K"]);
}

/// Loader-level regression for the issue's exact failure mode: rows with
/// inline comments (verbatim from the canonical file) must load — pre-fix
/// the first `#` token failed the whole `CMUDictLoader::load`.
#[test]
fn loader_loads_rows_with_inline_comments() {
  let dir = temp_dir("loader_inline_comments");
  fs::write(
    dir.join("cmudict.dict"),
    "aalborg AO1 L B AO0 R G # place, danish\n\
     aalborg(2) AA1 L B AO0 R G\n\
     aalburg AE1 L B ER0 G # place, dutch\n",
  )
  .unwrap();
  let dict =
    CMUDictLoader::load(&dir).expect("canonical-style rows with inline comments must load");
  assert_eq!(
    dict.lookup("aalborg").expect("aalborg").phonemes_slice(),
    ["ɔ", "l", "b", "ɔ", "ɹ", "ɡ"]
  );
  assert_eq!(
    dict.lookup("aalburg").expect("aalburg").phonemes_slice(),
    ["æ", "l", "b", "ɚ", "ɡ"]
  );
}

/// Integration smoke over the first 100 lines of the canonical
/// cmusphinx/cmudict `cmudict.dict` (master @ 7479086), embedded verbatim
/// below — no network. The slice carries five inline `# place/name`
/// comment rows (29, 31, 32, 36, 37) and eight `(N)` variant rows.
#[test]
fn loader_smoke_canonical_head_100() {
  assert_eq!(
    CANONICAL_CMUDICT_HEAD_100.lines().count(),
    100,
    "fixture must be exactly the canonical head-100"
  );

  let dir = temp_dir("loader_canonical_head100");
  fs::write(dir.join("cmudict.dict"), CANONICAL_CMUDICT_HEAD_100).unwrap();
  let dict = CMUDictLoader::load(&dir).expect("canonical cmudict.dict head-100 must load");

  // 100 rows − 8 `(N)` variant rows (primary_only=true) = 92 graphemes.
  assert_eq!(dict.len(), 92);

  // Commented rows resolve to comment-free phonemes.
  assert_eq!(
    dict.lookup("aalborg").expect("aalborg").phonemes_slice(),
    ["ɔ", "l", "b", "ɔ", "ɹ", "ɡ"]
  );
  // `aalto AA1 L T OW2 # name, finnish`.
  assert_eq!(
    dict.lookup("aalto").expect("aalto").phonemes_slice(),
    ["ɑ", "l", "t", "oʊ"]
  );
  // Primary `a  AH0` wins over the filtered `a(2)  EY1` variant.
  assert_eq!(dict.lookup("a").expect("a").phonemes_slice(), ["ə"]);
  // Apostrophe-leading and dotted words survive the strict word parse.
  assert!(dict.lookup("'bout").is_some());
  assert!(dict.lookup("a.d.").is_some());
}

/// First 100 lines of canonical cmusphinx/cmudict `cmudict.dict`
/// (<https://github.com/cmusphinx/cmudict>, master @ 7479086), verbatim.
const CANONICAL_CMUDICT_HEAD_100: &str = r"'bout B AW1 T
'cause K AH0 Z
'course K AO1 R S
'cuse K Y UW1 Z
'em AH0 M
'frisco F R IH1 S K OW0
'gain G EH1 N
'kay K EY1
'm AH0 M
'n AH0 N
'round R AW1 N D
's EH1 S
'til T IH1 L
'tis T IH1 Z
'twas T W AH1 Z
a AH0
a(2) EY1
a's EY1 Z
a. EY1
a.'s EY1 Z
a.d. EY2 D IY1
a.m. EY2 EH1 M
a.s EY1 Z
aaa T R IH2 P AH0 L EY1
aaberg AA1 B ER0 G
aachen AA1 K AH0 N
aachener AA1 K AH0 N ER0
aaker AA1 K ER0
aalborg AO1 L B AO0 R G # place, danish
aalborg(2) AA1 L B AO0 R G
aalburg AE1 L B ER0 G # place, dutch
aalen AE1 L AH0 N # place, german
aalen(2) AA1 L AH0 N
aaliyah AA2 L IY1 AA2
aalseth AA1 L S EH0 TH
aalsmeer AA1 L S M IH0 R # place, dutch
aalto AA1 L T OW2 # name, finnish
aamodt AA1 M AH0 T
aancor AA1 N K AO2 R
aardema AA0 R D EH1 M AH0
aardvark AA1 R D V AA2 R K
aardvarks AA1 R D V AA2 R K S
aargh AA1 R G
aarhus AA2 HH UW1 S
aaron EH1 R AH0 N
aaron's EH1 R AH0 N Z
aarons EH1 R AH0 N Z
aaronson EH1 R AH0 N S AH0 N
aaronson(2) AA1 R AH0 N S AH0 N
aaronson's EH1 R AH0 N S AH0 N Z
aaronson's(2) AA1 R AH0 N S AH0 N Z
aarti AA1 R T IY2
aase AA1 S
aasen AA1 S AH0 N
ab AE1 B
ab(2) EY1 B IY1
aba EY2 B IY2 EY1
ababa AH0 B AA1 B AH0
ababa(2) AA1 B AH0 B AH0
abacha AE1 B AH0 K AH0
aback AH0 B AE1 K
abaco AE1 B AH0 K OW2
abacus AE1 B AH0 K AH0 S
abad AH0 B AA1 D
abadaka AH0 B AE1 D AH0 K AH0
abadi AH0 B AE1 D IY0
abadie AH0 B AE1 D IY0
abair AH0 B EH1 R
abalkin AH0 B AA1 L K IH0 N
abalone AE2 B AH0 L OW1 N IY0
abalones AE2 B AH0 L OW1 N IY0 Z
abalos AA0 B AA1 L OW0 Z
abandon AH0 B AE1 N D AH0 N
abandoned AH0 B AE1 N D AH0 N D
abandoning AH0 B AE1 N D AH0 N IH0 NG
abandonment AH0 B AE1 N D AH0 N M AH0 N T
abandonments AH0 B AE1 N D AH0 N M AH0 N T S
abandons AH0 B AE1 N D AH0 N Z
abanto AH0 B AE1 N T OW0
abarca AH0 B AA1 R K AH0
abare AA0 B AA1 R IY0
abascal AE1 B AH0 S K AH0 L
abash AH0 B AE1 SH
abashed AH0 B AE1 SH T
abasia AH0 B EY1 ZH Y AH0
abate AH0 B EY1 T
abated AH0 B EY1 T IH0 D
abatement AH0 B EY1 T M AH0 N T
abatements AH0 B EY1 T M AH0 N T S
abates AH0 B EY1 T S
abating AH0 B EY1 T IH0 NG
abattoir AE2 B AH0 T W AA1 R
abba AE1 B AH0
abbado AH0 B AA1 D OW0
abbas AH0 B AA1 S
abbasi AA0 B AA1 S IY0
abbate AA1 B EY0 T
abbatiello AA0 B AA0 T IY0 EH1 L OW0
abbe AE1 B IY0
abbe(2) AE0 B EY1
";
