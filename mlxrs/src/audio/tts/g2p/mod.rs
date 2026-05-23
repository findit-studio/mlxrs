//! Grapheme-to-phoneme (G2P) subsystem — ported from
//! [mlx-audio-swift's `MLXAudioG2P`][src].
//!
//! ## Submodules
//!
//! - [`types`] — shared types ([`Phonemizer`](types::Phonemizer),
//!   [`Lexicon`](types::Lexicon), [`PhonemeUnit`](types::PhonemeUnit),
//!   [`LexiconEntry`](types::LexiconEntry)).
//! - [`arpabet`] — pure-algorithmic ARPAbet → IPA mapper.
//! - [`cmudict`] — CMU Pronouncing Dictionary in-memory lexicon + the
//!   local-file loader / parser.
//!
//! The neural-G2P orchestrator ([`NeuralPhonemizer`][np]) lives in a
//! follow-up commit on this branch.
//!
//! ## Scope
//!
//! Per the project's [no per-model arch porting][noarch] rule, mlxrs ships
//! the **orchestration** seam (the trait + composition) and the **purely
//! algorithmic** pieces (CMUDict / ARPAbet → IPA), NOT the ByT5 model
//! architecture itself.
//!
//! [src]: https://github.com/Blaizzy/mlx-audio-swift/tree/main/Sources/MLXAudioG2P
//! [np]: https://github.com/Blaizzy/mlx-audio-swift/blob/main/Sources/MLXAudioG2P/NeuralPhonemizer.swift
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md

pub mod arpabet;
pub mod cmudict;
pub mod types;

pub use cmudict::{CMUDict, CMUDictLoader, RawEntry};
pub use types::{Lexicon, LexiconEntry, PhonemeUnit, Phonemizer};
