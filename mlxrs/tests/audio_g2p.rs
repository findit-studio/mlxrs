//! Integration tests for `mlxrs::audio::tts::g2p` and
//! `mlxrs::audio::tts::text_processor`. Mirrors mlx-audio-swift's
//! `MLXAudioG2PTests.swift` + `MLXAudioG2PCMUDictTests.swift` (the
//! offline / unit-style cases; the network-gated `NeuralG2PIntegrationTests`
//! and the env-var-gated `CMUDictLoaderTests` upstream are out of scope —
//! we instead drive the loader against a small synthetic fixture written
//! per-test, matching the project's local-file-only policy).

#![cfg(feature = "audio")]

use std::fs;

use mlxrs::audio::tts::{
  BasicTextProcessor, TextProcessor,
  g2p::{
    CMUDict, CMUDictLoader, Lexicon, LexiconEntry, NeuralPhonemizer, PhonemeUnit, Phonemizer,
    arpabet::{convert_sequence, to_ipa},
    cmudict::{parse, parse_line},
  },
};

// ============================================================
// TextProcessor — round-trip via the trait, not internals.
// ============================================================

#[test]
fn text_processor_normalizes_known_input() {
  let tp = BasicTextProcessor::new();
  // NFC + lowercase + whitespace-collapse:
  //   "  Hello\u{0301}  Wörld  " → "hello\u{0301}" composed→"helló" (lowercase) → "helló wörld"
  // Wait — for "Hello\u{0301}": e + combining acute on the *previous* char.
  // Better: just use a simple normalized check.
  let out = tp.process("  Hello\tWorld  ", Some("en-us")).unwrap();
  assert_eq!(out, "hello world");
}

#[test]
fn text_processor_idempotent_on_already_normalized() {
  let tp = BasicTextProcessor::new();
  let once = tp.process("hello world", None).unwrap();
  let twice = tp.process(&once, None).unwrap();
  assert_eq!(once, twice);
}

#[test]
fn text_processor_nfc_compose() {
  let tp = BasicTextProcessor::new();
  let decomposed = "cafe\u{301}"; // 'e' + combining acute
  let out = tp.process(decomposed, None).unwrap();
  // After NFC: precomposed é (U+00E9).
  assert!(out.contains('\u{00E9}'), "{out:?}");
  assert!(!out.contains('\u{0301}'), "{out:?}");
}

// ============================================================
// ARPAbet → IPA mapper (table-driven, exhaustive).
// ============================================================

#[test]
fn arpabet_full_table_round_trips() {
  // Every (ARPAbet, IPA) pair from the swift ARPAbetMapper.mapping table.
  // Also covers the special-cased AH (stress-dependent) and ER
  // (stress-dependent) cases explicitly.
  let pairs: &[(&str, &str)] = &[
    // vowels (single IPA form, primary-stress sample)
    ("AA1", "ɑ"),
    ("AE1", "æ"),
    ("AO1", "ɔ"),
    ("AW1", "aʊ"),
    ("AY1", "aɪ"),
    ("EH1", "ɛ"),
    ("EY1", "eɪ"),
    ("IH1", "ɪ"),
    ("IY1", "i"),
    ("OW1", "oʊ"),
    ("OY1", "ɔɪ"),
    ("UH1", "ʊ"),
    ("UW1", "u"),
    // AH stress-dependent
    ("AH0", "ə"),
    ("AH1", "ʌ"),
    ("AH2", "ʌ"),
    // ER stress-dependent
    ("ER0", "ɚ"),
    ("ER1", "ɝ"),
    ("ER2", "ɝ"),
    // consonants
    ("B", "b"),
    ("CH", "tʃ"),
    ("D", "d"),
    ("DH", "ð"),
    ("F", "f"),
    ("G", "ɡ"),
    ("HH", "h"),
    ("JH", "dʒ"),
    ("K", "k"),
    ("L", "l"),
    ("M", "m"),
    ("N", "n"),
    ("NG", "ŋ"),
    ("P", "p"),
    ("R", "ɹ"),
    ("S", "s"),
    ("SH", "ʃ"),
    ("T", "t"),
    ("TH", "θ"),
    ("V", "v"),
    ("W", "w"),
    ("Y", "j"),
    ("Z", "z"),
    ("ZH", "ʒ"),
  ];
  for (arpa, ipa) in pairs {
    assert_eq!(to_ipa(arpa).as_deref(), Some(*ipa), "{arpa} → {ipa}");
  }
}

#[test]
fn arpabet_unknown_and_empty_return_none() {
  assert_eq!(to_ipa("XX"), None);
  assert_eq!(to_ipa(""), None);
}

#[test]
fn arpabet_convert_sequence_skips_unknown() {
  let ipa = convert_sequence(&["HH", "AH0", "L", "OW1"]);
  assert_eq!(ipa, vec!["h", "ə", "l", "oʊ"]);
  let with_unknown = convert_sequence(&["HH", "ZZ", "L"]);
  assert_eq!(with_unknown, vec!["h", "l"]);
}

// ============================================================
// CMUDict parser — known-good fixture + malformed-line-with-line-number.
// ============================================================

#[test]
fn cmudict_parse_known_good_fixture() {
  let text = ";;; comment\n\
              hello  HH AH0 L OW1\n\
              world  W ER1 L D\n\
              the  DH AH0\n\
              the(2)  DH IY0\n";
  let entries = parse(text, false).unwrap();
  assert_eq!(entries.len(), 4);
  assert_eq!(entries[0].word(), "hello");
  assert_eq!(entries[0].arpabet(), ["HH", "AH0", "L", "OW1"]);
  assert_eq!(entries[3].variant(), Some(2));
}

#[test]
fn cmudict_parse_primary_only_drops_variants() {
  let text = "the  DH AH0\nthe(2)  DH IY0\nhello  HH AH0 L OW1";
  let entries = parse(text, true).unwrap();
  assert_eq!(entries.len(), 2);
  assert!(entries.iter().all(|e| e.variant().is_none()));
}

#[test]
fn cmudict_parse_line_malformed_returns_err_with_line_number() {
  // Line 7 has no whitespace (i.e. word with no pronunciation) — must
  // surface as `Err(Error::OutOfRange)` whose payload carries the line
  // number in `value()`.
  let err = parse_line("nospaces", 7).unwrap_err();
  match err {
    mlxrs::Error::OutOfRange(p) => {
      assert_eq!(p.context(), "CMUDict line");
      assert_eq!(
        p.requirement(),
        "must contain whitespace between word and pronunciation"
      );
      assert_eq!(p.value(), "7");
    }
    other => panic!("expected OutOfRange with line 7, got {other:?}"),
  }
}

#[test]
fn cmudict_parse_bulk_malformed_surfaces_line_number() {
  let text = ";;; comment\n\
              hello  HH AH0 L OW1\n\
              malformed\n\
              world  W ER1 L D\n";
  // The bad row is on line 3 (1-indexed, comments count).
  let err = parse(text, false).unwrap_err();
  match err {
    mlxrs::Error::OutOfRange(p) => {
      assert_eq!(p.context(), "CMUDict line");
      assert_eq!(
        p.requirement(),
        "must contain whitespace between word and pronunciation"
      );
      assert_eq!(p.value(), "3");
    }
    other => panic!("expected OutOfRange with line 3, got {other:?}"),
  }
}

// ============================================================
// CMUDict lexicon + Loader (local-file-only synthetic fixture).
// ============================================================

fn temp_dir(name: &str) -> std::path::PathBuf {
  let dir = std::env::temp_dir().join(format!(
    "mlxrs_audio_g2p_it_{}_{}",
    std::process::id(),
    name
  ));
  let _ = fs::remove_dir_all(&dir);
  fs::create_dir_all(&dir).unwrap();
  dir
}

fn write_fixture(dir: &std::path::Path) {
  fs::write(
    dir.join("cmudict.dict"),
    ";;; mlxrs test fixture\n\
     hello  HH AH0 L OW1\n\
     world  W ER1 L D\n\
     the  DH AH0\n\
     the(2)  DH IY0\n\
     phone  F OW1 N\n\
     knight  N AY1 T\n",
  )
  .unwrap();
}

#[test]
fn cmudict_loader_loads_from_directory_then_looks_up() {
  let dir = temp_dir("loader_lookup");
  write_fixture(&dir);
  let dict = CMUDictLoader::load(&dir).unwrap();
  assert!(dict.lookup("hello").is_some());
  assert!(dict.lookup("world").is_some());
  assert!(dict.lookup("the").is_some());
  assert!(dict.lookup("phone").is_some());
  assert!(dict.lookup("knight").is_some());
}

#[test]
fn cmudict_loader_produces_correct_ipa() {
  let dir = temp_dir("loader_ipa");
  write_fixture(&dir);
  let dict = CMUDictLoader::load(&dir).unwrap();
  let hello = dict.lookup("hello").unwrap();
  assert_eq!(hello.phonemes_slice(), ["h", "ə", "l", "oʊ"]);
}

#[test]
fn cmudict_loader_lookup_is_case_insensitive() {
  let dir = temp_dir("loader_case");
  write_fixture(&dir);
  let dict = CMUDictLoader::load(&dir).unwrap();
  assert!(dict.lookup("HELLO").is_some());
  assert!(dict.lookup("Hello").is_some());
}

#[test]
fn cmudict_loader_unknown_word_returns_none() {
  let dir = temp_dir("loader_unknown");
  write_fixture(&dir);
  let dict = CMUDictLoader::load(&dir).unwrap();
  assert!(dict.lookup("xyzzyplugh").is_none());
}

#[test]
fn cmudict_loader_fixes_digraph_orthography() {
  // Sanity check: "phone" → contains "f" (not "p"); "knight" → starts "n"
  // (silent 'k'). Mirrors swift's `fixesDigraphsCorrectly`.
  let dir = temp_dir("loader_digraphs");
  write_fixture(&dir);
  let dict = CMUDictLoader::load(&dir).unwrap();

  let phone = dict.lookup("phone").unwrap();
  assert!(phone.phonemes_slice().contains(&"f".to_string()));
  assert!(!phone.phonemes_slice().contains(&"p".to_string()));

  let knight = dict.lookup("knight").unwrap();
  assert_eq!(
    knight.phonemes_slice().first().map(String::as_str),
    Some("n")
  );
}

#[test]
fn cmudict_loader_errors_on_missing_dict_file() {
  let dir = temp_dir("loader_missing");
  // Don't write the fixture.
  let err = CMUDictLoader::load(&dir).unwrap_err();
  match err {
    mlxrs::Error::MissingKey(p) => {
      assert_eq!(p.context(), "CMUDictLoader::load: required file not found");
      let expected_path = dir.join("cmudict.dict");
      assert_eq!(p.key(), expected_path.display().to_string());
    }
    other => panic!("expected MissingKey for missing cmudict.dict, got {other:?}"),
  }
}

// ============================================================
// NeuralPhonemizer trait — at least one concrete TextProcessor + G2P pair
// runs end-to-end on a known input (with a stubbed neural backend).
// ============================================================

/// A user-supplied `TextProcessor` that wires a [`NeuralPhonemizer`] into
/// the `audio::tts::TextProcessor` hook. Mirrors the swift
/// `MisakiTextProcessor` example from the swift `TextProcessor.swift`
/// docs.
struct G2pTextProcessor<F>
where
  F: Fn(&str, &str) -> mlxrs::error::Result<String>,
{
  g2p: NeuralPhonemizer<F>,
}

impl<F> TextProcessor for G2pTextProcessor<F>
where
  F: Fn(&str, &str) -> mlxrs::error::Result<String>,
{
  fn process(&self, text: &str, _language: Option<&str>) -> mlxrs::error::Result<String> {
    // Per-word phonemize, then re-join with whitespace (the simplest
    // sentence-level orchestration; per-model code does whatever it likes).
    let mut out = Vec::new();
    for word in text.split_whitespace() {
      let units = self.g2p.phonemize(word)?;
      let joined: String = units.into_iter().map(|u| u.symbol().to_owned()).collect();
      out.push(joined);
    }
    Ok(out.join(" "))
  }
}

#[test]
fn end_to_end_text_processor_with_neural_phonemizer_runs() {
  // The neural backend is a stub — in real usage it would be a ByT5
  // model loaded from disk and run by user code. We're proving the
  // *trait composition* works, not the model.
  let backend = |word: &str, _lang: &str| -> mlxrs::error::Result<String> {
    let stub = match word {
      "hello" => "h ə l oʊ",
      "world" => "w ɝ l d",
      _ => "ɑ", // arbitrary fallback
    };
    Ok(stub.to_string())
  };
  let g2p = NeuralPhonemizer::new(backend, "eng-us");
  let tp = G2pTextProcessor { g2p };

  let out = tp.process("hello world", Some("eng-us")).unwrap();
  // After per-word phonemize + join: "həloʊ wɝld"
  assert_eq!(out, "həloʊ wɝld");
}

#[test]
fn end_to_end_lexicon_first_with_neural_fallback() {
  // Common pattern: try CMUDict (deterministic), fall back to neural G2P
  // if the word is OOV. Proves the trait + lookup compose.
  let dict = CMUDict::from_entries(vec![LexiconEntry::new(
    "hello",
    vec!["h".into(), "ə".into(), "l".into(), "oʊ".into()],
  )]);
  let neural_called = std::sync::atomic::AtomicBool::new(false);
  let backend = |word: &str, _lang: &str| -> mlxrs::error::Result<String> {
    neural_called.store(true, std::sync::atomic::Ordering::SeqCst);
    let stub = match word {
      "xyz" => "ɛ k s",
      _ => "ɑ",
    };
    Ok(stub.to_string())
  };
  let neural = NeuralPhonemizer::new(backend, "eng-us");

  // "hello" hits the lexicon, "xyz" misses → neural fallback.
  let hello_units: Vec<PhonemeUnit> = if let Some(entry) = dict.lookup("hello") {
    entry
      .phonemes_slice()
      .iter()
      .map(|p| PhonemeUnit::new(p.clone()))
      .collect()
  } else {
    neural.phonemize("hello").unwrap()
  };
  let xyz_units: Vec<PhonemeUnit> = if let Some(entry) = dict.lookup("xyz") {
    entry
      .phonemes_slice()
      .iter()
      .map(|p| PhonemeUnit::new(p.clone()))
      .collect()
  } else {
    neural.phonemize("xyz").unwrap()
  };

  assert_eq!(
    hello_units,
    vec![
      PhonemeUnit::new("h"),
      PhonemeUnit::new("ə"),
      PhonemeUnit::new("l"),
      PhonemeUnit::new("oʊ"),
    ]
  );
  // OOV "xyz" → "ɛ k s" → split per char (whitespace dropped) → 3 units.
  assert_eq!(xyz_units.len(), 3);
  assert!(neural_called.load(std::sync::atomic::Ordering::SeqCst));
}
