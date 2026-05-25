//! ARPAbet → IPA mapper — a 1:1 port of mlx-audio-swift's
//! [`ARPAbetMapper.swift`][src].
//!
//! ARPAbet is the phoneme inventory CMU Pronouncing Dictionary uses (39
//! ASCII tokens — `AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B CH D DH
//! F G HH JH K L M N NG P R S SH T TH V W Y Z ZH`); each vowel optionally
//! carries a trailing stress digit (`0` unstressed, `1` primary, `2`
//! secondary). The mapper strips the stress, then table-translates to IPA
//! with one special case for `AH` (`0` → `ə`, `1`/`2` → `ʌ`) and `ER`
//! (`0` → `ɚ`, `1`/`2` → `ɝ`).
//!
//! Stress digits themselves are dropped; the mapping returns the bare IPA
//! glyph (matching the swift impl). Unknown tokens map to `None` (and the
//! batch [`convert_sequence`] silently skips them, matching swift's
//! `compactMap`).
//!
//! [src]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/CMUDict/ARPAbetMapper.swift

/// Map a single ARPAbet symbol (with optional trailing stress digit) to
/// its IPA equivalent. Returns `None` for empty input or unknown tokens.
///
/// Examples:
/// ```
/// # use mlxrs::audio::tts::g2p::arpabet::to_ipa;
/// assert_eq!(to_ipa("TH"), Some("θ".into()));
/// assert_eq!(to_ipa("AH0"), Some("ə".into()));
/// assert_eq!(to_ipa("AH1"), Some("ʌ".into()));
/// assert_eq!(to_ipa("ER0"), Some("ɚ".into()));
/// assert_eq!(to_ipa("ER1"), Some("ɝ".into()));
/// assert_eq!(to_ipa("XX"), None);
/// assert_eq!(to_ipa(""), None);
/// ```
#[must_use]
pub fn to_ipa(arpabet: &str) -> Option<String> {
  if arpabet.is_empty() {
    return None;
  }

  // Last char is a stress digit if it is '0', '1', or '2'.
  let last = arpabet.as_bytes()[arpabet.len() - 1];
  let (base, stress): (&str, Option<u8>) = if (b'0'..=b'2').contains(&last) {
    // ASCII-only safe split — stress digits are single bytes.
    (&arpabet[..arpabet.len() - 1], Some(last - b'0'))
  } else {
    (arpabet, None)
  };

  // Special cases (vowel-with-stress-dependent IPA).
  match base {
    "AH" => {
      return Some(if stress == Some(0) {
        "ə".into()
      } else {
        "ʌ".into()
      });
    }
    "ER" => {
      return Some(if stress == Some(0) {
        "ɚ".into()
      } else {
        "ɝ".into()
      });
    }
    _ => {}
  }

  arpabet_table(base).map(str::to_owned)
}

/// Batch-convert a list of ARPAbet symbols to IPA, silently skipping
/// unknown tokens (matching swift's `compactMap`).
///
/// This is the LAX path — appropriate for free-form text where dropping a
/// single mis-spelled token is preferable to failing the whole input. For
/// CMUDict loading use [`try_convert_sequence_strict`] instead: a dropped
/// token there silently corrupts the lexicon entry.
///
/// Examples:
/// ```
/// # use mlxrs::audio::tts::g2p::arpabet::convert_sequence;
/// assert_eq!(
///   convert_sequence(&["HH", "AH0", "L", "OW1"]),
///   vec!["h", "ə", "l", "oʊ"]
/// );
/// // Unknown tokens are dropped, not preserved.
/// assert_eq!(convert_sequence(&["HH", "XX", "L"]), vec!["h", "l"]);
/// ```
#[must_use]
pub fn convert_sequence<S: AsRef<str>>(arpabet: &[S]) -> Vec<String> {
  arpabet.iter().filter_map(|s| to_ipa(s.as_ref())).collect()
}

/// An ARPAbet token that [`try_convert_sequence_strict`] could not map to
/// IPA. Carries the offending source token so callers can surface a
/// precise error (e.g. `"word 'foo' has unknown ARPAbet token 'XX'"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadArpabetToken {
  /// The unrecognised source token (as it appeared in the input slice).
  pub token: String,
}

/// Batch-convert a list of ARPAbet symbols to IPA, returning `Err` on the
/// FIRST unknown token (i.e. one that [`to_ipa`] returns `None` for).
///
/// Use this for callers that must not silently drop tokens —
/// canonical example is CMUDict loading, where dropping a token corrupts
/// the lexicon entry (empty / wrong-length pronunciation, blocking the
/// lexicon-first / neural-fallback pattern). The lax [`convert_sequence`]
/// remains the right choice for free-form text.
///
/// Examples:
/// ```
/// # use mlxrs::audio::tts::g2p::arpabet::{try_convert_sequence_strict, BadArpabetToken};
/// assert_eq!(
///   try_convert_sequence_strict(&["HH", "AH0", "L", "OW1"]).unwrap(),
///   vec!["h", "ə", "l", "oʊ"]
/// );
/// assert_eq!(
///   try_convert_sequence_strict(&["HH", "XX", "L"]).unwrap_err(),
///   BadArpabetToken { token: "XX".to_string() }
/// );
/// ```
pub fn try_convert_sequence_strict<S: AsRef<str>>(
  arpabet: &[S],
) -> Result<Vec<String>, BadArpabetToken> {
  let mut out = Vec::with_capacity(arpabet.len());
  for s in arpabet {
    let token = s.as_ref();
    match to_ipa(token) {
      Some(ipa) => out.push(ipa),
      None => {
        return Err(BadArpabetToken {
          token: token.to_owned(),
        });
      }
    }
  }
  Ok(out)
}

/// The base ARPAbet → IPA table (the consonants + vowels with a single
/// IPA form). `AH` / `ER` have stress-dependent IPA handled in [`to_ipa`]
/// directly and are absent here.
fn arpabet_table(base: &str) -> Option<&'static str> {
  Some(match base {
    // Vowels (single IPA form)
    "AA" => "ɑ",
    "AE" => "æ",
    "AO" => "ɔ",
    "AW" => "aʊ",
    "AY" => "aɪ",
    "EH" => "ɛ",
    "EY" => "eɪ",
    "IH" => "ɪ",
    "IY" => "i",
    "OW" => "oʊ",
    "OY" => "ɔɪ",
    "UH" => "ʊ",
    "UW" => "u",
    // Consonants
    "B" => "b",
    "CH" => "tʃ",
    "D" => "d",
    "DH" => "ð",
    "F" => "f",
    "G" => "ɡ",
    "HH" => "h",
    "JH" => "dʒ",
    "K" => "k",
    "L" => "l",
    "M" => "m",
    "N" => "n",
    "NG" => "ŋ",
    "P" => "p",
    "R" => "ɹ",
    "S" => "s",
    "SH" => "ʃ",
    "T" => "t",
    "TH" => "θ",
    "V" => "v",
    "W" => "w",
    "Y" => "j",
    "Z" => "z",
    "ZH" => "ʒ",
    _ => return None,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  // Mirrors `ARPAbetMapperTests.mapsConsonant` in
  // Tests/MLXAudioG2PCMUDictTests.swift.
  #[test]
  fn maps_consonants() {
    assert_eq!(to_ipa("TH").as_deref(), Some("θ"));
    assert_eq!(to_ipa("SH").as_deref(), Some("ʃ"));
    assert_eq!(to_ipa("NG").as_deref(), Some("ŋ"));
    assert_eq!(to_ipa("HH").as_deref(), Some("h"));
    assert_eq!(to_ipa("CH").as_deref(), Some("tʃ"));
    assert_eq!(to_ipa("JH").as_deref(), Some("dʒ"));
    assert_eq!(to_ipa("ZH").as_deref(), Some("ʒ"));
  }

  // Mirrors `mapsVowelStrippingStress`.
  #[test]
  fn maps_ah_with_stress() {
    assert_eq!(to_ipa("AH0").as_deref(), Some("ə"));
    assert_eq!(to_ipa("AH1").as_deref(), Some("ʌ"));
    assert_eq!(to_ipa("AH2").as_deref(), Some("ʌ"));
  }

  #[test]
  fn maps_er_with_stress() {
    assert_eq!(to_ipa("ER0").as_deref(), Some("ɚ"));
    assert_eq!(to_ipa("ER1").as_deref(), Some("ɝ"));
    assert_eq!(to_ipa("ER2").as_deref(), Some("ɝ"));
  }

  // Mirrors `mapsRegularVowels`.
  #[test]
  fn maps_regular_vowels() {
    assert_eq!(to_ipa("AA0").as_deref(), Some("ɑ"));
    assert_eq!(to_ipa("AA1").as_deref(), Some("ɑ"));
    assert_eq!(to_ipa("AE1").as_deref(), Some("æ"));
    assert_eq!(to_ipa("EY0").as_deref(), Some("eɪ"));
    assert_eq!(to_ipa("OW1").as_deref(), Some("oʊ"));
    assert_eq!(to_ipa("AW0").as_deref(), Some("aʊ"));
    assert_eq!(to_ipa("AY1").as_deref(), Some("aɪ"));
    assert_eq!(to_ipa("OY0").as_deref(), Some("ɔɪ"));
  }

  // Mirrors `returnsNilForUnknown`.
  #[test]
  fn returns_none_for_unknown_and_empty() {
    assert_eq!(to_ipa("XX"), None);
    assert_eq!(to_ipa(""), None);
  }

  // Mirrors `convertsFullSequence`.
  #[test]
  fn converts_full_sequence() {
    let ipa = convert_sequence(&["HH", "AH0", "L", "OW1"]);
    assert_eq!(ipa, vec!["h", "ə", "l", "oʊ"]);
  }

  // Mirrors `convertsSequenceSkipsUnknown`.
  #[test]
  fn converts_sequence_skips_unknown() {
    let ipa = convert_sequence(&["HH", "XX", "L"]);
    assert_eq!(ipa, vec!["h", "l"]);
  }

  // === try_convert_sequence_strict ===

  #[test]
  fn try_convert_sequence_strict_round_trips_known() {
    let ipa = try_convert_sequence_strict(&["HH", "AH0", "L", "OW1"]).unwrap();
    assert_eq!(ipa, vec!["h", "ə", "l", "oʊ"]);
  }

  #[test]
  fn try_convert_sequence_strict_errors_on_first_unknown() {
    let err = try_convert_sequence_strict(&["HH", "XX", "YY", "L"]).unwrap_err();
    // The FIRST unknown token wins (so the surfaced message is precise).
    assert_eq!(err.token, "XX");
  }

  /// Regression guard: the lax [`convert_sequence`] API must keep
  /// dropping unknown tokens (the swift `compactMap` behaviour) for callers
  /// outside the CMUDict loader.
  #[test]
  fn arpabet_convert_sequence_lax_still_drops_unknown() {
    let ipa = convert_sequence(&["HH", "XX", "L", "ZZ"]);
    assert_eq!(ipa, vec!["h", "l"]);
  }

  /// Cover every consonant ARPAbet symbol exhaustively (table-driven,
  /// matching the inventory in the swift `mapping` table).
  #[test]
  fn covers_full_consonant_inventory() {
    let pairs: &[(&str, &str)] = &[
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
      assert_eq!(
        to_ipa(arpa).as_deref(),
        Some(*ipa),
        "consonant {arpa} → {ipa}"
      );
    }
  }

  /// Cover every vowel-with-stress-1 (table-driven, matching the swift
  /// `mapping` table for vowels).
  #[test]
  fn covers_full_vowel_inventory_with_primary_stress() {
    let pairs: &[(&str, &str)] = &[
      ("AA1", "ɑ"),
      ("AE1", "æ"),
      ("AH1", "ʌ"),
      ("AO1", "ɔ"),
      ("AW1", "aʊ"),
      ("AY1", "aɪ"),
      ("EH1", "ɛ"),
      ("ER1", "ɝ"),
      ("EY1", "eɪ"),
      ("IH1", "ɪ"),
      ("IY1", "i"),
      ("OW1", "oʊ"),
      ("OY1", "ɔɪ"),
      ("UH1", "ʊ"),
      ("UW1", "u"),
    ];
    for (arpa, ipa) in pairs {
      assert_eq!(to_ipa(arpa).as_deref(), Some(*ipa), "vowel {arpa} → {ipa}");
    }
  }
}
