//! Text-preprocessing hook for the TTS synthesis pipeline.
//!
//! Ports mlx-audio-swift's [`TextProcessor`][swift-tp] protocol — the
//! *interface* a per-model phonemizer / normalizer plugs into.
//!
//! ## What lives here
//!
//! - [`TextProcessor`] — the trait, a 1:1 of the swift protocol (a
//!   [`prepare`][TextProcessor::prepare] hook + a
//!   [`process`][TextProcessor::process] method that takes natural-language
//!   text and an optional language hint, returns the per-model expected
//!   string).
//! - [`BasicTextProcessor`] — a no-G2P default impl that runs the three
//!   normalization passes most TTS frontends share — Unicode NFC
//!   ([`unicode-normalization`][un] is the only added dep) → ASCII
//!   lowercase → whitespace collapse — and returns the normalized text.
//!   Suitable for any TTS model that wants raw normalized text (not
//!   phonemized IPA). Per-model phonemizers (Misaki G2P, eSpeak adapter, …)
//!   implement [`TextProcessor`] directly without going through
//!   [`BasicTextProcessor`].
//!
//! [swift-tp]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/TextProcessor.swift
//! [un]: https://docs.rs/unicode-normalization

use unicode_normalization::UnicodeNormalization;

use crate::error::Result;

/// Text-preprocessing hook for the TTS synthesis pipeline.
///
/// A 1:1 port of mlx-audio-swift's [`TextProcessor`][swift-tp] protocol —
/// the *interface*, not a concrete phonemizer. Some TTS models (kokoro,
/// kitten-tts) require phonemized IPA input rather than raw text; a
/// [`TextProcessor`] converts natural-language text into the format the
/// target model expects.
///
/// Phonemization / G2P itself is **model-specific** and out of scope per
/// the project's no per-model arch porting rule — mlxrs ships the
/// hook, not a Misaki/eSpeak G2P implementation. A per-model crate
/// implements [`TextProcessor`] (e.g. a Misaki G2P adapter) and the model's
/// [`crate::audio::tts::model::TtsModel::synthesize_segment`] runs it; the
/// [`crate::audio::tts::generate::tts_generate`] driver itself never
/// phonemizes — it passes segment text through unchanged.
///
/// Why a separate hook trait rather than a method on
/// [`TtsModel`](crate::audio::tts::model::TtsModel): mlx-audio-swift keeps
/// `TextProcessor` distinct from `SpeechGenerationModel` precisely so one
/// G2P adapter can be shared across several models, and so a caller can
/// *inject* a custom processor at load time (the swift `loadModel(...,
/// textProcessor:)` parameter). mlxrs mirrors that separation — per the
/// mirror-reference-structure rule, the reference's two distinct
/// protocols stay two distinct traits.
///
/// [swift-tp]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioTTS/TextProcessor.swift
pub trait TextProcessor {
  /// Download or initialize any resources the processor needs before
  /// [`TextProcessor::process`] can run (a G2P lexicon, a weights file, …).
  ///
  /// Mirrors the swift `TextProcessor.prepare()`. The default impl is a
  /// no-op for processors that need no preparation (the swift protocol
  /// extension's default is likewise empty); call it once before the first
  /// `process`.
  fn prepare(&mut self) -> Result<()> {
    Ok(())
  }

  /// Convert input `text` into the format the target model expects (e.g.
  /// phonemized IPA).
  ///
  /// `language` is an optional locale code (`"en-us"`, `"en-gb"`, …) — the
  /// same `language: String?` argument the swift `process(text:language:)`
  /// takes; `None` lets the processor pick a default. Returns the processed
  /// string a per-model tokenizer then consumes.
  fn process(&self, text: &str, language: Option<&str>) -> Result<String>;
}

/// A no-G2P default [`TextProcessor`] that applies the three normalization
/// passes most TTS text frontends share — Unicode NFC composition,
/// lowercase folding, and whitespace collapse — and returns the
/// normalized text.
///
/// Suitable for any TTS model that wants raw normalized text (not
/// phonemized IPA); a per-model phonemizer (Misaki G2P, eSpeak adapter)
/// implements [`TextProcessor`] directly without composing
/// [`BasicTextProcessor`] (it does no phonemization).
///
/// The three passes (mirroring the Swift TTS-frontend convention):
///
/// 1. **Unicode NFC** — composes combining marks into precomposed forms so
///    `"café"` (with a combining acute accent) and `"café"` (with a
///    precomposed é) collide on the same string.
/// 2. **Lowercase** — uses Rust's [`str::to_lowercase`] (Unicode-aware,
///    matching Swift's `String.lowercased()`).
/// 3. **Whitespace collapse** — runs of Unicode whitespace (any
///    [`char::is_whitespace`]) collapse to a single ASCII space, and
///    leading / trailing whitespace is trimmed.
///
/// `language` is ignored — this processor does not branch on locale.
#[derive(Debug, Default, Clone, Copy)]
pub struct BasicTextProcessor;

impl BasicTextProcessor {
  /// Construct a fresh [`BasicTextProcessor`]. Zero-sized; equivalent to
  /// [`BasicTextProcessor::default`].
  #[must_use]
  pub const fn new() -> Self {
    Self
  }

  /// Run the three normalization passes (NFC → lowercase → whitespace
  /// collapse) on `text` and return the normalized string. Exposed as a
  /// free function so callers that do not want the trait machinery can
  /// reuse the same pipeline.
  #[must_use]
  pub fn normalize(text: &str) -> String {
    // 1. NFC compose
    let nfc: String = text.nfc().collect();
    // 2. lowercase (Unicode-aware)
    let lower = nfc.to_lowercase();
    // 3. whitespace collapse + trim
    collapse_whitespace(&lower)
  }
}

impl TextProcessor for BasicTextProcessor {
  fn process(&self, text: &str, _language: Option<&str>) -> Result<String> {
    Ok(Self::normalize(text))
  }
}

/// Collapse runs of Unicode whitespace to a single ASCII space and trim
/// leading / trailing whitespace. `"  hello \t world  "` → `"hello world"`.
fn collapse_whitespace(s: &str) -> String {
  let mut out = String::with_capacity(s.len());
  let mut in_ws = true; // skip leading whitespace
  for ch in s.chars() {
    if ch.is_whitespace() {
      if !in_ws {
        in_ws = true;
        out.push(' ');
      }
    } else {
      in_ws = false;
      out.push(ch);
    }
  }
  // Strip trailing single space we may have appended above.
  if out.ends_with(' ') {
    out.pop();
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn basic_lowercase() {
    let tp = BasicTextProcessor::new();
    let out = tp.process("Hello World", None).unwrap();
    assert_eq!(out, "hello world");
  }

  #[test]
  fn collapses_internal_whitespace() {
    assert_eq!(BasicTextProcessor::normalize("a   b\t\tc\nd"), "a b c d");
  }

  #[test]
  fn trims_leading_and_trailing_whitespace() {
    assert_eq!(BasicTextProcessor::normalize("  hello  "), "hello");
  }

  #[test]
  fn empty_input_round_trips() {
    assert_eq!(BasicTextProcessor::normalize(""), "");
    assert_eq!(BasicTextProcessor::normalize("   "), "");
  }

  /// NFC composes the decomposed e + combining acute (U+0065 U+0301) into
  /// the precomposed é (U+00E9).
  #[test]
  fn nfc_composes_combining_marks() {
    // decomposed: "cafe\u{301}" — 'e' + COMBINING ACUTE ACCENT
    let decomposed = "cafe\u{301}";
    let out = BasicTextProcessor::normalize(decomposed);
    // After NFC compose + lowercase: "café" (5 chars: c, a, f, é, but lowercase)
    // The é codepoint is U+00E9.
    assert!(
      out.contains('\u{00E9}'),
      "expected NFC-composed é in {out:?}"
    );
    // No standalone combining acute should remain after NFC.
    assert!(
      !out.contains('\u{0301}'),
      "combining acute should be composed away in {out:?}"
    );
  }

  #[test]
  fn unicode_lowercase() {
    assert_eq!(BasicTextProcessor::normalize("ÄPFEL"), "äpfel");
  }

  #[test]
  fn prepare_is_no_op() {
    let mut tp = BasicTextProcessor::new();
    tp.prepare().unwrap();
  }

  #[test]
  fn language_is_ignored() {
    let tp = BasicTextProcessor::new();
    let with_lang = tp.process("Hello", Some("en-us")).unwrap();
    let without_lang = tp.process("Hello", None).unwrap();
    assert_eq!(with_lang, without_lang);
  }
}
