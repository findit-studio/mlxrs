//! Grapheme-to-phoneme (G2P) subsystem — ported from
//! [mlx-audio-swift's `MLXAudioG2P`][src].
//!
//! ## Submodules
//!
//! - [`types`] — shared types ([`Phonemizer`], [`Lexicon`],
//!   [`PhonemeUnit`], [`LexiconEntry`]).
//! - [`arpabet`] — pure-algorithmic ARPAbet → IPA mapper.
//! - [`cmudict`] — CMU Pronouncing Dictionary in-memory lexicon + the
//!   local-file loader / parser.
//! - [`neural_phonemizer`] — the model-agnostic **orchestrator**
//!   ([`NeuralPhonemizer`] trait adapter): wires a [`Phonemizer`] backend
//!   into a [`crate::audio::tts::TextProcessor`].
//!
//! ## Scope
//!
//! Per the project's no per-model arch porting rule, mlxrs ships
//! the **orchestration** seam (the trait + composition) and the **purely
//! algorithmic** pieces (CMUDict / ARPAbet → IPA), NOT the ByT5 model
//! architecture itself. The neural-G2P backend (T5 encoder/decoder
//! weights, attention, relative-position bias) lives in user code on top
//! of [`Phonemizer`].
//!
//! [src]: https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioG2P

pub mod arpabet;
pub mod cmudict;
pub mod neural_phonemizer;
pub mod types;

pub use cmudict::{CMUDict, CMUDictLoader, RawEntry};
pub use neural_phonemizer::NeuralPhonemizer;
pub use types::{Lexicon, LexiconEntry, PhonemeUnit, Phonemizer};
