//! Shared types for the grapheme-to-phoneme (G2P) subsystem — a 1:1 port
//! of mlx-audio-swift's [`G2PTypes.swift`][types] +
//! [`LexiconEntry.swift`][entry] + [`LexiconProviding.swift`][provider].
//!
//! [types]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/G2PTypes.swift
//! [entry]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/LexiconEntry.swift
//! [provider]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/Lexicon/LexiconProviding.swift

/// A single phoneme unit produced by a [`Phonemizer`] — an IPA glyph or
/// ARPAbet symbol carried as an owned string.
///
/// Mirrors swift's `PhonemeUnit` (the `symbol: String` wrapper). One-field
/// struct kept as a distinct type so callers can carry per-unit metadata
/// (stress / tone) in a future extension without breaking the slice API.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhonemeUnit {
  /// The phoneme's textual symbol (e.g. `"h"`, `"ə"`, `"oʊ"`).
  symbol: String,
}

impl PhonemeUnit {
  /// Construct a [`PhonemeUnit`] from any string-like value.
  pub fn new(symbol: impl Into<String>) -> Self {
    Self {
      symbol: symbol.into(),
    }
  }

  /// The phoneme's textual symbol (e.g. `"h"`, `"ə"`, `"oʊ"`).
  #[inline(always)]
  pub fn symbol(&self) -> &str {
    &self.symbol
  }
}

/// A single lexicon row — one grapheme spelling paired with its phoneme
/// sequence. Mirrors swift's `LexiconEntry`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LexiconEntry {
  /// The grapheme / spelling (e.g. `"hello"`).
  grapheme: String,
  /// The IPA phoneme sequence (e.g. `["h", "ə", "l", "oʊ"]`).
  phonemes: Vec<String>,
}

impl LexiconEntry {
  /// Construct a [`LexiconEntry`] from owned strings.
  pub fn new(grapheme: impl Into<String>, phonemes: Vec<String>) -> Self {
    Self {
      grapheme: grapheme.into(),
      phonemes,
    }
  }

  /// The grapheme / spelling (e.g. `"hello"`).
  #[inline(always)]
  pub fn grapheme(&self) -> &str {
    &self.grapheme
  }

  /// The IPA phoneme sequence as a slice (e.g. `["h", "ə", "l", "oʊ"]`).
  #[inline(always)]
  pub fn phonemes_slice(&self) -> &[String] {
    &self.phonemes
  }
}

/// A pronunciation lexicon — a grapheme → phoneme-sequence lookup.
///
/// Mirrors swift's `LexiconProviding` protocol. Implementors are typically
/// in-memory hash tables ([`crate::audio::tts::g2p::cmudict::CMUDict`])
/// but the trait deliberately abstracts the storage so a per-model crate
/// can plug in an on-disk / mmap'd / RPC-backed lexicon without changing
/// the orchestrator.
pub trait Lexicon {
  /// Look up `grapheme` (case-insensitive, lowercase-folded by the
  /// implementor) and return its [`LexiconEntry`], or `None` if the word
  /// is OOV. Returns a borrow so callers do not clone unless they choose
  /// to.
  fn lookup(&self, grapheme: &str) -> Option<&LexiconEntry>;
}

/// A grapheme-to-phoneme converter — turns a written word into a sequence
/// of [`PhonemeUnit`]s.
///
/// Mirrors swift's `Phonemizing` protocol. Concrete implementors compose a
/// lexicon ([`Lexicon`]) and an optional fallback (e.g.
/// [`crate::audio::tts::g2p::neural_phonemizer::NeuralPhonemizer`]) on top
/// of this trait.
pub trait Phonemizer {
  /// Convert a single grapheme (typically one word) into its phoneme
  /// sequence. Returns `Err` if conversion fails (empty input, unsupported
  /// locale, &c.) — see the variants on the returned error type.
  fn phonemize(&self, grapheme: &str) -> crate::error::Result<Vec<PhonemeUnit>>;
}
