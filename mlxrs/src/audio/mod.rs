//! Audio (TTS/STT/STS) — speech inference primitives.
//!
//! M5 ships the core IO + DSP primitives ported faithfully from
//! `mlx-audio` ([`audio_io.py`] + [`dsp.py`]):
//! - [`crate::audio::io`] — multi-format audio **load** (WAV / MP3 /
//!   FLAC / OGG-Vorbis, format auto-detected via the `symphonia`
//!   crate's probe — the four formats `mlx_audio.audio_io.read`
//!   decodes in-process via `miniaudio`) + WAV **save** (roll-our-own
//!   pure-Rust 16-bit PCM mono encoder with atomic-rename via tempfile
//!   + fsync-before-rename + permission preservation; ~80 LOC in
//!   `audio/io.rs::save_wav`). Naive linear resampling; sinc/polyphase
//!   resamplers are planned follow-ups.
//! - [`crate::audio::dsp`] — window family (Hann/Hamming/Blackman/Bartlett
//!   + the `STR_TO_WINDOW_FN`-style [`crate::audio::dsp::window_from_name`]
//!   dispatch), STFT (over [`crate::ops::fft::rfft`] +
//!   [`crate::ops::shape::as_strided`]), inverse STFT
//!   ([`crate::audio::dsp::istft`], overlap-add reconstruction over
//!   [`crate::ops::fft::irfft`] +
//!   [`crate::ops::indexing::scatter_add_axis`]), the cached / batched
//!   overlap-add [`crate::audio::dsp::ISTFTCache`] (streaming inverse STFT),
//!   mel filterbank, mel + log-mel spectrogram, 1-D IIR
//!   [`crate::audio::dsp::lfilter`], ITU-R BS.1770 K-weighted integrated
//!   loudness ([`crate::audio::dsp::integrated_loudness`]) +
//!   [`crate::audio::dsp::normalize_loudness`] +
//!   [`crate::audio::dsp::normalize_peak`] (peak dBFS normalization).
//!
//! - [`crate::audio::features`] — Kaldi-compatible log-mel-filterbank features
//!   (`mel_scale_kaldi` / `inverse_mel_scale_kaldi` / `get_mel_banks_kaldi` /
//!   `compute_fbank_kaldi` with both `snip_edges` paths, plus
//!   [`crate::audio::features::compute_deltas_kaldi`] delta/acceleration
//!   coefficients, sibling to the HTK/Whisper mel front-end in
//!   [`crate::audio::dsp`]).
//!
//! Out of scope for this PR (separate follow-ups per the M5 plan):
//! - High-quality resampling (polyphase sinc, libsamplerate-style).
//! - Pitch shifting, time stretching, voice activity detection, biquad filters.
//! - **Decode** of M4A/AAC, Opus, and WebM. `mlx_audio.audio_io.read`
//!   routes those through an external `ffmpeg` subprocess, and Opus has
//!   no pure-Rust `symphonia` codec in 0.6 (no `opus` feature / no
//!   `symphonia-codec-opus` crate — an open upstream issue). Adding any
//!   of them needs a heavy `libopus`/`ffmpeg` C dependency, which the
//!   minimal-deps project rule scopes out; see [`crate::audio::io`].
//! - **Encode** of any non-WAV format. `mlx-audio` shells out to
//!   `ffmpeg` for MP3/FLAC/OGG/Opus encoding; mlxrs's save path stays
//!   pure-Rust 16-bit PCM mono WAV (no encoder crate / `ffmpeg` dep).
//! - Per-model architectures (Whisper, Sesame, CSM, etc.) — see the
//!   "no per-model arch porting" project rule.
//!
//! [`audio_io.py`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/audio_io.py
//! [`dsp.py`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py

pub mod dsp;
pub mod features;
pub mod io;
pub mod stt;
pub mod tts;
