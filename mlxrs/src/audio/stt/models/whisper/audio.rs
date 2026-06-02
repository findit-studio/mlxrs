//! Whisper audio front-end: hard-coded hyperparameters, the log-mel
//! spectrogram, and `pad_or_trim`.
//!
//! Faithful port of `mlx_audio.stt.models.whisper.audio`
//! (`mlx-source/mlx-audio/mlx_audio/stt/models/whisper/audio.py`). The
//! log-mel post-processing is the Whisper-specific tail on top of
//! the shared Slaney mel filterbank ([`crate::audio::dsp::mel_filter_bank_scaled`]):
//! drop the last STFT frame, `magnitudes @ filters.T`, then
//! `log10` + dynamic-range clamp + the `(x + 4) / 4` affine renorm.
//!
//! Layout note: unlike [`crate::audio::dsp::log_mel_spectrogram`] (which is
//! `(n_mels, num_frames)`), the Whisper mel is **`(num_frames, n_mels)`** —
//! frames on axis 0, mel bins (the conv input channels) on axis 1 — because
//! the Whisper `AudioEncoder` feeds it straight into
//! `nn.Conv1d(n_mels, n_state, ...)` whose channel axis is last (`x.shape[-2]`
//! is the frame axis in the reference's `pad_or_trim(mel, N_FRAMES,
//! axis=-2)`).

use crate::{
  Array, Error, Result,
  audio::dsp::{self, MelScale, WindowPad},
  error::{InvariantViolationPayload, OutOfRangePayload},
  ops,
};

use smol_str::format_smolstr;

/// Whisper sample rate (Hz). `mlx-audio/.../whisper/audio.py:12`.
pub const SAMPLE_RATE: u32 = 16_000;
/// Whisper STFT FFT length. `audio.py:13`.
pub const N_FFT: usize = 400;
/// Whisper STFT hop length (samples). `audio.py:14`.
pub const HOP_LENGTH: usize = 160;
/// Whisper analysis chunk length (seconds). `audio.py:15`.
pub const CHUNK_LENGTH: usize = 30;
/// Samples in one [`CHUNK_LENGTH`]-second chunk (`30 * 16000 = 480000`).
/// `audio.py:16`.
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE as usize;
/// Mel frames in one chunk's spectrogram input (`480000 / 160 = 3000`).
/// `audio.py:17`.
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
/// Audio samples per output token — the initial conv has stride 2
/// (`160 * 2 = 320`). `audio.py:19`.
pub const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2;
/// Audio frames per second (`16000 / 160 = 100`, i.e. 10 ms per frame).
/// `audio.py:20`.
pub const FRAMES_PER_SECOND: usize = SAMPLE_RATE as usize / HOP_LENGTH;
/// Output tokens per second (`16000 / 320 = 50`, i.e. 20 ms per token).
/// `audio.py:21`.
pub const TOKENS_PER_SECOND: usize = SAMPLE_RATE as usize / N_SAMPLES_PER_TOKEN;

/// Whisper log-mel dynamic-range clamp: `log_spec = max(log_spec,
/// log_spec.max() - 8.0)` — clamp the spectrogram to 8 decades below its
/// peak. `audio.py:80`.
const LOG_SPEC_DYNAMIC_RANGE: f32 = 8.0;
/// Whisper log-mel affine renorm offset: `(log_spec + 4.0) / 4.0`.
/// `audio.py:81`.
const LOG_SPEC_AFFINE_OFFSET: f32 = 4.0;
/// Whisper log-mel affine renorm divisor: `(log_spec + 4.0) / 4.0`.
/// `audio.py:81`.
const LOG_SPEC_AFFINE_DIV: f32 = 4.0;
/// Whisper log-mel floor before `log10`: `max(mel_spec, 1e-10)`. `audio.py:79`.
const LOG_SPEC_FLOOR: f32 = 1e-10;

/// Compute the Whisper log-mel spectrogram of a 1-D mono `16 kHz` waveform.
///
/// Faithful port of `mlx_audio.stt.models.whisper.audio.log_mel_spectrogram`
/// (`audio.py:41-82`):
///
/// ```text
/// window = hanning(400)                          # symmetric (periodic=False)
/// freqs  = stft(audio, window, n_fft=400, hop=160)   # center=True, reflect
/// magnitudes = freqs[:-1, :].abs().square()      # DROP the last STFT frame
/// filters = mel_filters(16000, 400, n_mels, norm="slaney", mel_scale=None)
/// mel_spec = magnitudes @ filters.T
/// log_spec = maximum(mel_spec, 1e-10).log10()    # log10, NOT ln
/// log_spec = maximum(log_spec, log_spec.max() - 8.0)
/// log_spec = (log_spec + 4.0) / 4.0
/// ```
///
/// Returns shape `(num_frames - 1, n_mels)` `Dtype::F32` (frames on axis 0,
/// mel bins on axis 1 — see the module doc for the layout rationale). With
/// `padding` zero samples appended on the right (the reference's `padding`
/// argument), `num_frames` grows accordingly; pass the encoder's expected
/// frame count to [`pad_or_trim`] afterward to land on [`N_FRAMES`].
///
/// `samples` must be 1-D. The mel bank is fetched via
/// [`dsp::mel_filter_bank_scaled_cached`] (Slaney scale + Slaney
/// normalization), so repeated chunks reuse the per-thread cached bank.
///
/// # Errors
/// - [`Error::RankMismatch`] (via [`dsp::stft`]) if `samples` is not 1-D;
/// - [`Error::OutOfRange`] if `padding` exceeds `i32::MAX`, or the produced
///   spectrogram has fewer than 2 frames (so `freqs[:-1, :]` would be empty);
/// - propagates [`dsp::stft`] / [`dsp::mel_filter_bank_scaled_cached`] /
///   matmul errors.
pub fn log_mel_spectrogram_whisper(
  samples: &Array,
  n_mels: usize,
  padding: usize,
) -> Result<Array> {
  if samples.ndim() != 1 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "log_mel_spectrogram_whisper: samples",
      "must be 1-D (a mono waveform)",
    )));
  }

  // Optional right-zero-pad (`mx.pad(audio, (0, padding))`, `audio.py:70-71`).
  let padded = if padding > 0 {
    let pad = i32::try_from(padding).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "log_mel_spectrogram_whisper: padding",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{padding}"),
      ))
    })?;
    let zero = Array::zeros::<f32>(&[0i32; 0])?;
    ops::shape::pad(samples, &[0_i32], &[0_i32], &[pad], &zero, c"constant")?
  } else {
    samples.try_clone()?
  };

  // STFT with the Whisper hyperparameters. `dsp::stft` hardcodes
  // `center = true, pad_mode = "reflect"` (the reference default) and builds
  // the symmetric Hann window internally; `WindowPad::Right` (the mel
  // front-end convention) matches `win_length == n_fft` here (no padding).
  let spec = dsp::stft(&padded, N_FFT, HOP_LENGTH, None, WindowPad::Right)?;
  let freqs = spec.data_ref(); // (num_frames, n_freqs) Complex64

  // Drop the last STFT frame: `freqs[:-1, :]` (`audio.py:74`). With
  // `center=true` the reference's framing yields one extra trailing frame vs
  // the `n_samples / hop` count, which Whisper discards to land on N_FRAMES.
  let num_frames = freqs.shape()[0];
  if num_frames < 2 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "log_mel_spectrogram_whisper: num_frames (after STFT)",
      "must be >= 2 so dropping the last frame leaves >= 1 frame",
      format_smolstr!("num_frames={num_frames}"),
    )));
  }
  let n_freqs_i32 = i32::try_from(freqs.shape()[1]).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "log_mel_spectrogram_whisper: n_freqs",
      "must fit in i32",
      format_smolstr!("{}", freqs.shape()[1]),
    ))
  })?;
  let keep = i32::try_from(num_frames - 1).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "log_mel_spectrogram_whisper: num_frames - 1",
      "must fit in i32",
      format_smolstr!("{}", num_frames - 1),
    ))
  })?;
  // `freqs[0..num_frames-1, 0..n_freqs]`.
  let freqs_dropped = ops::indexing::slice(freqs, &[0, 0], &[keep, n_freqs_i32], &[1, 1])?;

  // `magnitudes = |freqs[:-1]|.square()` → (num_frames-1, n_freqs) F32.
  let magnitudes = freqs_dropped.abs()?.square()?;

  // Slaney mel filterbank (`mel_filters(16000, 400, n_mels, norm="slaney",
  // mel_scale=None)`) — Slaney scale + Slaney normalization, cached.
  let filters = dsp::mel_filter_bank_scaled_cached(
    n_mels,
    N_FFT,
    SAMPLE_RATE,
    0.0,
    None,
    MelScale::Slaney,
    true,
  )?; // (n_mels, n_freqs)

  // `mel_spec = magnitudes @ filters.T` → (num_frames-1, n_mels).
  let filters_t = filters.transpose()?;
  let mel_spec = ops::linalg_basic::matmul(&magnitudes, &filters_t)?;

  // `log_spec = maximum(mel_spec, 1e-10).log10()` (log10, NOT ln).
  let floor = Array::full::<f32>(&[0i32; 0], LOG_SPEC_FLOOR)?;
  let floored = ops::arithmetic::maximum(&mel_spec, &floor)?;
  let log_spec = floored.log10()?;

  // `log_spec = maximum(log_spec, log_spec.max() - 8.0)` — clamp to 8 decades
  // below the global peak. `reduction::max(.., keepdims=false)` is the scalar
  // global max; subtracting the dynamic range gives the broadcast floor.
  let peak = ops::reduction::max(&log_spec, false)?;
  let range = Array::full::<f32>(&[0i32; 0], LOG_SPEC_DYNAMIC_RANGE)?;
  let clamp_floor = ops::arithmetic::subtract(&peak, &range)?;
  let clamped = ops::arithmetic::maximum(&log_spec, &clamp_floor)?;

  // `log_spec = (log_spec + 4.0) / 4.0`.
  let offset = Array::full::<f32>(&[0i32; 0], LOG_SPEC_AFFINE_OFFSET)?;
  let div = Array::full::<f32>(&[0i32; 0], LOG_SPEC_AFFINE_DIV)?;
  let shifted = ops::arithmetic::add(&clamped, &offset)?;
  ops::arithmetic::divide(&shifted, &div)
}

/// Pad-or-trim a mel spectrogram (or waveform) to `length` along `axis`.
///
/// Faithful port of `mlx_audio.stt.models.whisper.audio.pad_or_trim`
/// (`audio.py:24-38`): if `array.shape[axis] > length`, slice
/// `[0..length]`; if `< length`, zero-pad `(0, length - shape[axis])` on the
/// right. Whisper calls this on the mel with `axis = -2` (the frame axis) to
/// land on [`N_FRAMES`].
///
/// `axis` is a non-negative index here (the reference's default is `-1`, but
/// Whisper always passes the resolved frame axis); callers pass the explicit
/// axis. For the canonical `(num_frames, n_mels)` mel, the frame axis is `0`.
///
/// # Errors
/// - [`Error::OutOfRange`] if `axis >= array.ndim()` or `length` /
///   sizes exceed `i32::MAX`;
/// - propagates [`ops::indexing::slice`] / [`ops::shape::pad`] errors.
pub fn pad_or_trim(array: &Array, length: usize, axis: usize) -> Result<Array> {
  let shape = array.shape();
  let ndim = shape.len();
  if axis >= ndim {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "pad_or_trim: axis",
      "must be < array.ndim()",
      format_smolstr!("axis={axis}, ndim={ndim}"),
    )));
  }
  let cur = shape[axis];
  if cur == length {
    return array.try_clone();
  }

  let length_i32 = i32::try_from(length).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "pad_or_trim: length",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{length}"),
    ))
  })?;

  if cur > length {
    // Trim: slice `[0..length]` on `axis`, full extent on every other axis.
    let mut start = vec![0_i32; ndim];
    let mut stop = vec![0_i32; ndim];
    let strides = vec![1_i32; ndim];
    for (d, &s) in shape.iter().enumerate() {
      let s_i32 = i32::try_from(s).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "pad_or_trim: dim",
          "must fit in i32",
          format_smolstr!("{s}"),
        ))
      })?;
      stop[d] = s_i32;
    }
    start[axis] = 0;
    stop[axis] = length_i32;
    return ops::indexing::slice(array, &start, &stop, &strides);
  }

  // Pad: append `length - cur` zeros on the right of `axis`.
  let mut lo = vec![0_i32; ndim];
  let mut hi = vec![0_i32; ndim];
  let axes: Vec<i32> = (0..ndim as i32).collect();
  let pad_amt = i32::try_from(length - cur).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "pad_or_trim: pad amount",
      "must fit in i32",
      format_smolstr!("{}", length - cur),
    ))
  })?;
  lo[axis] = 0;
  hi[axis] = pad_amt;
  let zero = Array::zeros::<f32>(&[0i32; 0])?;
  ops::shape::pad(array, &axes, &lo, &hi, &zero, c"constant")
}

#[cfg(test)]
mod tests;
