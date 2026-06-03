//! CLAP-HTSAT-unfused — dual-tower audio+text embeddings model
//! (`laion/clap-htsat-unfused`).
//!
//! CLAP is a contrastive dual-tower model: an HTSAT Swin-Transformer audio
//! encoder and a RoBERTa text encoder each map their input to a 512-dim vector
//! in one shared, L2-normalized space, so `cosine(audio, text)` ranks
//! audio↔text relevance (the zero-shot-classification primitive). This port
//! targets the **unfused** checkpoint (`enable_fusion=False`): a single fixed
//! 10 s mel window, no feature-fusion branch.
//!
//! Sources:
//! - HF `transformers` `ClapModel` (`laion/clap-htsat-unfused`) — the
//!   authoritative architecture (`ClapAudioModel` + `ClapTextModel` + two
//!   projection MLPs + L2-normalize).
//! - The Findit-AI `textclap` crate — owns the mel front-end + the I/O
//!   contract verbatim, and its committed `golden_mel.npy` /
//!   `filterbank_row_*.npy` fixtures pin the mel front-end numerically (the
//!   [`mel`] oracle).
//!
//! ## Phase status
//! This module currently ships **phases 1–2**: the configuration
//! ([`config::ClapConfig`] = [`config::ClapAudioConfig`] +
//! [`config::ClapTextConfig`] + projection dims), the mel / spectrogram
//! front-end ([`mel`]), and the RoBERTa **text tower** ([`text::ClapTextModel`]
//! — the `ClapTextModel` embeddings + post-norm encoder + CLS pooling + the
//! text projection + L2-normalize, with the [`text::ClapTextModel`]
//! [`crate::embeddings::TextEmbedder`] impl). The HTSAT Swin **audio** tower
//! (phase 3), the full dual-tower `ClapModel` assembly + `classify` + the
//! factory registration (phase 4), and the end-to-end checkpoint-parity test
//! (phase 5) land in later phases; [`text::ClapTextModel`] exposes a clean
//! `embed_text` the phase-4 assembly consumes as the text tower.
//!
//! ## Reuse
//! The mel front-end reuses [`crate::audio::dsp`] —
//! [`mel_filter_bank_scaled`](crate::audio::dsp::mel_filter_bank_scaled)
//! (Slaney scale + Slaney normalization), [`stft`](crate::audio::dsp::stft),
//! and the framing / rfft machinery — rather than re-implementing the DSP. The
//! configuration mirrors the [`crate::embeddings::siglip2_naflex`] dual-tower
//! config precedent (serde `#[serde(default)]` + a `validate()` that pins
//! every architecture-defining field before any tensor is built).

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod config;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod mel;

#[cfg(feature = "clap")]
mod shared;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod text;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use config::{ClapAudioConfig, ClapConfig, ClapTextConfig};

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use mel::{MelFrontEnd, N_MELS, T_FRAMES};

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use text::ClapTextModel;
