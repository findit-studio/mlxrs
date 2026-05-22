//! Audio (TTS/STT/STS) — speech inference primitives.
//!
//! M5 ships the core IO + DSP primitives ported faithfully from
//! `mlx-audio` ([`audio_io.py`] + [`dsp.py`]):
//! - [`crate::audio::io`] — WAV load (via the `symphonia` crate, WAV +
//!   PCM features only) + WAV save (roll-our-own pure-Rust 16-bit PCM
//!   mono encoder with atomic-rename via tempfile + fsync-before-rename
//!   + permission preservation; ~80 LOC in `audio/io.rs::save_wav`).
//!   Naive linear resampling for the WAV-only surface; sinc/polyphase
//!   resamplers are planned follow-ups.
//! - [`crate::audio::dsp`] — window family (Hann/Hamming/Blackman/Bartlett
//!   + the `STR_TO_WINDOW_FN`-style [`crate::audio::dsp::window_from_name`]
//!   dispatch), STFT (over [`crate::ops::fft::rfft`] +
//!   [`crate::ops::shape::as_strided`]), inverse STFT
//!   ([`crate::audio::dsp::istft`], overlap-add reconstruction over
//!   [`crate::ops::fft::irfft`] +
//!   [`crate::ops::indexing::scatter_add_axis`]), mel filterbank, mel +
//!   log-mel spectrogram, 1-D IIR [`crate::audio::dsp::lfilter`], ITU-R
//!   BS.1770 K-weighted integrated loudness
//!   ([`crate::audio::dsp::integrated_loudness`]) +
//!   [`crate::audio::dsp::normalize_loudness`].
//!
//! - [`crate::audio::features`] — Kaldi-compatible log-mel-filterbank features
//!   (`mel_scale_kaldi` / `inverse_mel_scale_kaldi` / `get_mel_banks_kaldi` /
//!   `compute_fbank_kaldi`, sibling to the HTK/Whisper mel front-end in
//!   [`crate::audio::dsp`]).
//!
//! Out of scope for this PR (separate follow-ups per the M5 plan):
//! - The `ISTFTCache` batched/cached overlap-add helper.
//! - High-quality resampling (polyphase sinc, libsamplerate-style).
//! - Pitch shifting, time stretching, voice activity detection,
//!   `normalize_peak`, biquad filters, Kaldi `compute_deltas_kaldi` deltas.
//! - The Kaldi `snip_edges=false` reflect-bookend framing path (the standard
//!   ASR pipelines all use `snip_edges=true`; see
//!   [`crate::audio::features`] for the scope fence).
//! - MP3/FLAC/OGG codecs (additional symphonia feature flags become
//!   opt-in in future PRs; the `symphonia` crate already supports them,
//!   we just don't enable them yet to keep the dep tree minimal).
//! - Per-model architectures (Whisper, Sesame, CSM, etc.) — see the
//!   "no per-model arch porting" project rule.
//!
//! [`audio_io.py`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/audio_io.py
//! [`dsp.py`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py

pub mod dsp;
pub mod features;
pub mod io;
pub mod stt;
