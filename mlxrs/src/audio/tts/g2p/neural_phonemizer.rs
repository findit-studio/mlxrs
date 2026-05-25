//! Neural-G2P orchestrator — a port of mlx-audio-swift's
//! [`NeuralPhonemizer.swift`][src].
//!
//! ## What this is (and isn't)
//!
//! `NeuralPhonemizer` is the **orchestration seam** that wraps a neural
//! G2P backend (typically a ByT5 encoder-decoder) into a
//! [`Phonemizer`]: one input grapheme → `Vec<PhonemeUnit>`.
//!
//! Per the project's [no per-model arch porting][noarch] rule the
//! underlying T5 model architecture (encoder/decoder layers, attention,
//! relative-position bias, weight loader) is **NOT** ported. mlxrs ships
//! the trait + composition; user code on top of [`Phonemizer`] supplies
//! whatever neural backend it wants — ByT5, a small bigram model, or a
//! mocked-in-test stub.
//!
//! The Swift impl owns a `G2P` (which owns a `T5ForConditionalGeneration`
//! plus a `ByT5Tokenizer`); mlxrs deliberately abstracts the backend
//! behind a `convert: Fn(&str, &str) -> Result<String>` closure so
//! callers wire whatever they want. The post-processing pipeline (trim,
//! strip whitespace, split into per-glyph [`PhonemeUnit`]s) matches the
//! swift impl exactly.
//!
//! ## Composition
//!
//! Typical wiring (sketched, since the actual neural backend is
//! out-of-scope):
//!
//! ```no_run
//! use mlxrs::audio::tts::g2p::{NeuralPhonemizer, Phonemizer};
//! // A user-supplied neural backend (out of scope here — would be a
//! // ByT5 model load + greedy-decode).
//! let backend = |word: &str, lang: &str| -> mlxrs::error::Result<String> {
//!   Ok(format!("h ə l oʊ"))  // stub for example purposes
//! };
//! let phonemizer = NeuralPhonemizer::new(backend, "eng-us");
//! let units = phonemizer.phonemize("hello").unwrap();
//! ```
//!
//! [src]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/NeuralPhonemizer.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

use crate::{
  audio::tts::g2p::types::{PhonemeUnit, Phonemizer},
  error::{Error, Result},
};

/// A neural-G2P orchestrator parameterized over an arbitrary backend
/// `convert: Fn(&str, &str) -> Result<String>` (word, language → IPA
/// string).
///
/// Per the no-per-model-arch rule, the actual neural model (ByT5,
/// transformer, &c.) is supplied by user code through the `convert`
/// closure; mlxrs supplies the post-processing (trim + drop whitespace +
/// split into per-glyph [`PhonemeUnit`]s) and the error handling (empty
/// model output → [`Error::Backend`] with the offending grapheme), both
/// matching the swift impl 1:1.
///
/// The `language` field is the locale code threaded through to the
/// backend on every [`Phonemizer::phonemize`] call (e.g. `"eng-us"`).
pub struct NeuralPhonemizer<F> {
  convert: F,
  language: String,
}

impl<F> NeuralPhonemizer<F>
where
  F: Fn(&str, &str) -> Result<String>,
{
  /// Construct a [`NeuralPhonemizer`] from a backend closure + language
  /// code. The closure is called once per [`Phonemizer::phonemize`]
  /// invocation with the input grapheme and the configured language.
  pub fn new(convert: F, language: impl Into<String>) -> Self {
    Self {
      convert,
      language: language.into(),
    }
  }

  /// The locale code passed to the backend on each phonemize call.
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }
}

impl<F> Phonemizer for NeuralPhonemizer<F>
where
  F: Fn(&str, &str) -> Result<String>,
{
  fn phonemize(&self, grapheme: &str) -> Result<Vec<PhonemeUnit>> {
    let raw = (self.convert)(grapheme, &self.language)?;
    let ipa = raw.trim();
    if ipa.is_empty() {
      return Err(Error::Backend {
        message: format!("neural G2P returned empty output for token {grapheme:?}"),
      });
    }
    // Match swift: filter out whitespace, map each remaining char into
    // its own PhonemeUnit. Multi-codepoint IPA glyphs (e.g. tʃ, oʊ) end
    // up split here — same as the swift impl which also does
    // `ipa.filter { !$0.isWhitespace }.map { PhonemeUnit(symbol: String($0)) }`.
    let units: Vec<PhonemeUnit> = ipa
      .chars()
      .filter(|c| !c.is_whitespace())
      .map(|c| PhonemeUnit::new(c.to_string()))
      .collect();
    Ok(units)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn happy_path_splits_into_phoneme_units() {
    let backend = |_w: &str, _l: &str| Ok("h ə l oʊ".to_string());
    let p = NeuralPhonemizer::new(backend, "eng-us");
    let units = p.phonemize("hello").unwrap();
    // Whitespace dropped; multi-byte glyphs split per char.
    assert_eq!(
      units,
      vec![
        PhonemeUnit::new("h"),
        PhonemeUnit::new("ə"),
        PhonemeUnit::new("l"),
        PhonemeUnit::new("o"),
        PhonemeUnit::new("ʊ"),
      ]
    );
  }

  #[test]
  fn empty_backend_output_errors_with_token() {
    let backend = |_w: &str, _l: &str| Ok("   ".to_string());
    let p = NeuralPhonemizer::new(backend, "eng-us");
    let err = p.phonemize("ghost").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("ghost"), "expected token in {msg:?}");
    assert!(msg.contains("empty"), "expected 'empty' in {msg:?}");
  }

  #[test]
  fn backend_error_propagates() {
    let backend = |_w: &str, _l: &str| -> Result<String> {
      Err(Error::Backend {
        message: "model failure".into(),
      })
    };
    let p = NeuralPhonemizer::new(backend, "eng-us");
    let err = p.phonemize("test").unwrap_err();
    assert!(err.to_string().contains("model failure"));
  }

  #[test]
  fn language_is_threaded_to_backend() {
    let p = NeuralPhonemizer::new(
      |word: &str, lang: &str| Ok(format!("<{lang}>:{word}")),
      "es",
    );
    let units = p.phonemize("hola").unwrap();
    // Should encode the language and word into the symbols (then split per char).
    let joined: String = units.iter().map(|u| u.symbol()).collect();
    assert_eq!(joined, "<es>:hola");
    assert_eq!(p.language(), "es");
  }
}
