//! CLAP mel / spectrogram front-end.
//!
//! A faithful port of `textclap/src/mel.rs` (the proven CLAP mel pipeline,
//! itself matching HF `transformers.ClapFeatureExtractor` on
//! `laion/clap-htsat-unfused`): turn a 48 kHz mono `&[f32]` waveform into the
//! `(1, 1, T=1001, n_mels=64)` log-mel spectrogram the HTSAT tower consumes.
//!
//! Pipeline (matching `textclap/src/mel.rs:182-250` step-for-step):
//! 1. **repeat-pad / head-truncate** to `TARGET_SAMPLES` (`= 10 s`): tile the
//!    waveform `floor(target / len)` times, then zero-pad the remainder; if the
//!    input already reaches the target, head-truncate to it (HF `repeatpad`).
//! 2. **center reflect-pad** by `n_fft / 2` on each side (librosa `center=True`).
//! 3. **STFT** — frame at `hop`, window each frame with a **periodic** Hann of
//!    length `n_fft`, real-FFT, take `|X|²`.
//! 4. **mel projection** — `power @ filterbank.T` with the Slaney-scale,
//!    Slaney-normalized `(n_mels, n_freqs)` filterbank.
//! 5. **power → dB** — `10 · log10(max(amin, x))` (`amin = 1e-10`, `ref = 1.0`,
//!    no `top_db` clip), applied **once**, giving a `-100 dB` floor.
//!
//! ## Reuse vs new
//! The Slaney filterbank comes straight from
//! [`crate::audio::dsp::mel_filter_bank_scaled_cached`] (Slaney scale + Slaney
//! normalization, the `MelPrecision::Precise` f64 build cast to f32 — matching
//! `textclap`'s f64 filterbank), and the framing + real-FFT reuse
//! [`crate::ops::shape::as_strided`] + [`crate::ops::fft::rfft`] (the same
//! machinery [`crate::audio::dsp::stft`] frames with). The **only** piece this
//! module owns over `dsp::stft` is the **periodic** Hann window: `dsp::stft`
//! hardcodes a *symmetric* (`periodic=False`) Hann, whereas CLAP / HF / librosa
//! `center=True` use the *periodic* (`fftbins=True`) Hann, so the window — and
//! the repeat-pad + power-to-dB glue — are built here.
//!
//! ## Precision
//! `textclap` computes the STFT in `f64` to match HF (`mel.rs:35`); MLX's
//! [`rfft`](crate::ops::fft::rfft) is `f32`. The dB output therefore carries a
//! small (`~1e-1` dB worst-case) drift vs the `golden_mel.npy` fixture that the
//! oracle test absorbs with a relative / cosine tolerance (the filterbank rows,
//! built in f64, still match `filterbank_row_*.npy` tightly).

use crate::{
  Array, Error, Result,
  audio::dsp::{self, MelPrecision, MelScale},
  error::{InvariantViolationPayload, OutOfRangePayload},
  ops,
};

use smol_str::format_smolstr;

/// Audio sample rate the CLAP front-end expects (Hz).
/// `textclap/src/mel.rs:20` / `golden_params.json["sampling_rate"]`.
pub const SAMPLE_RATE: u32 = 48_000;
/// FFT window size. `textclap/src/mel.rs:17` /
/// `golden_params.json["fft_window_size"]`.
pub const N_FFT: usize = 1024;
/// STFT hop length (samples). `textclap/src/mel.rs:18` /
/// `golden_params.json["hop_length"]`.
pub const HOP: usize = 480;
/// Mel-bin count. `textclap/src/mel.rs:19` / `golden_params.json["feature_size"]`.
pub const N_MELS: usize = 64;
/// Target sample count (`10 s` at 48 kHz). `textclap/src/mel.rs:21`.
pub const TARGET_SAMPLES: usize = 480_000;
/// Mel time-frame count (HF `center=True` framing of [`TARGET_SAMPLES`]).
/// `textclap/src/mel.rs:15` / `golden_params.json["T_frames"]`.
pub const T_FRAMES: usize = 1001;
/// Mel lower frequency bound (Hz). `textclap/src/mel.rs:22`.
pub const FMIN: f32 = 50.0;
/// Mel upper frequency bound (Hz). `textclap/src/mel.rs:23`.
pub const FMAX: f32 = 14_000.0;
/// `power_to_db` floor: `10 · log10(max(amin, x))` with `amin = 1e-10` gives a
/// `-100 dB` floor. `textclap/src/mel.rs:24`.
const POWER_TO_DB_AMIN: f32 = 1e-10;

/// CLAP mel / spectrogram front-end. Owns the precomputed periodic Hann window
/// and the cached Slaney filterbank so repeated clips amortize their
/// construction.
///
/// Construct once with [`MelFrontEnd::new`], then call [`MelFrontEnd::extract`]
/// per clip. The output is the `(1, 1, T_FRAMES, N_MELS)` log-mel spectrogram
/// (HF feature-extractor layout: `(batch, channel, time, mel)`), `Dtype::F32`.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub struct MelFrontEnd {
  /// The `(n_fft,)` periodic Hann window, `Dtype::F32` (built in f64, cast to
  /// f32 — see [`periodic_hann_f32`]).
  window: Array,
  /// The `(n_freqs, n_mels)` transposed Slaney filterbank, `Dtype::F32`
  /// (transposed once so the per-clip projection is `power @ filterbank_t`).
  filterbank_t: Array,
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl MelFrontEnd {
  /// Build the front-end: precompute the periodic Hann window and the Slaney
  /// filterbank.
  ///
  /// # Errors
  /// Propagates [`Array`] construction and
  /// [`dsp::mel_filter_bank_scaled_cached`] errors.
  pub fn new() -> Result<Self> {
    let window = periodic_hann_f32(N_FFT)?;
    // Slaney scale + Slaney normalization, built in f64 then cast to f32 (the
    // `Precise` path) to match `textclap`'s f64 reference filterbank — this is
    // what makes the filterbank rows match `filterbank_row_*.npy` tightly.
    let filterbank = dsp::mel_filter_bank_scaled_with(
      N_MELS,
      N_FFT,
      SAMPLE_RATE,
      FMIN,
      Some(FMAX),
      MelPrecision::Precise,
      MelScale::Slaney,
      true,
    )?; // (n_mels, n_freqs)
    let filterbank_t = filterbank.transpose()?; // (n_freqs, n_mels)
    Ok(Self {
      window,
      filterbank_t,
    })
  }

  /// The `(n_mels, n_freqs)` Slaney filterbank, re-materialized from the cached
  /// transpose. Used by the oracle test to compare rows against
  /// `filterbank_row_*.npy`.
  ///
  /// # Errors
  /// Propagates [`Array::transpose`].
  pub fn filterbank(&self) -> Result<Array> {
    self.filterbank_t.transpose()
  }

  /// Extract the `(1, 1, T_FRAMES, N_MELS)` log-mel spectrogram of a 48 kHz
  /// mono waveform.
  ///
  /// The waveform is repeat-padded / head-truncated to [`TARGET_SAMPLES`],
  /// center reflect-padded, framed + periodic-Hann-windowed + real-FFT'd to
  /// `|X|²`, projected through the Slaney filterbank, and converted to dB. The
  /// result is laid out `(batch=1, channel=1, time=T_FRAMES, mel=N_MELS)` to
  /// match the HF `ClapFeatureExtractor` `input_features` layout.
  ///
  /// # Errors
  /// - [`Error::InvariantViolation`] if `samples` is empty (the repeat-pad
  ///   would divide by zero).
  /// - Propagates the framing / FFT / matmul / arithmetic errors and any
  ///   `i32` shape-overflow.
  pub fn extract(&self, samples: &[f32]) -> Result<Array> {
    if samples.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "clap::mel::extract: samples",
        "must be non-empty (the repeat-pad divides by the sample count)",
      )));
    }

    // 1. Repeat-pad / head-truncate to TARGET_SAMPLES (HF `repeatpad`).
    let padded = repeat_pad(samples)?; // (TARGET_SAMPLES,) F32

    // 2. center reflect-pad by n_fft / 2 (librosa `center=True`).
    let centered = reflect_pad(&padded, N_FFT / 2)?; // (TARGET_SAMPLES + n_fft,)

    // 3. STFT: frame, periodic-Hann window, real-FFT, |X|².
    let power = self.stft_power(&centered)?; // (T_FRAMES, n_freqs) F32

    // 4. mel projection: power @ filterbank.T → (T_FRAMES, n_mels).
    let mel = ops::linalg_basic::matmul(&power, &self.filterbank_t)?;

    // 5. power → dB: 10 · log10(max(amin, mel)), applied once (floor -100 dB).
    let amin = Array::full::<f32>(&[0i32; 0], POWER_TO_DB_AMIN)?;
    let floored = ops::arithmetic::maximum(&mel, &amin)?;
    let ten = Array::full::<f32>(&[0i32; 0], 10.0)?;
    let db = ops::arithmetic::multiply(&floored.log10()?, &ten)?; // (T_FRAMES, n_mels)

    // Reshape to the HF (batch, channel, time, mel) = (1, 1, T_FRAMES, N_MELS).
    let t = i32::try_from(T_FRAMES).map_err(|_| shape_overflow("T_FRAMES", T_FRAMES))?;
    let m = i32::try_from(N_MELS).map_err(|_| shape_overflow("N_MELS", N_MELS))?;
    db.reshape(&[1, 1, t, m])
  }

  /// Frame the centered signal at [`HOP`], window each `(n_fft,)` frame with the
  /// periodic Hann, real-FFT, and return the `(T_FRAMES, n_freqs)` power
  /// spectrum `|X|²`.
  fn stft_power(&self, centered: &Array) -> Result<Array> {
    let centered_len = centered.shape()[0];
    // The centered signal has length TARGET_SAMPLES + n_fft, so the framing
    // reaches exactly T_FRAMES frames: last frame ends at
    // (T_FRAMES-1)*HOP + N_FFT = 1000*480 + 1024 = 481024 = TARGET_SAMPLES + N_FFT.
    let expected = TARGET_SAMPLES + N_FFT;
    if centered_len != expected {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap::mel::stft_power: centered length",
        "must equal TARGET_SAMPLES + N_FFT",
        format_smolstr!("centered_len={centered_len}, expected={expected}"),
      )));
    }

    let t_frames_i32 = i32::try_from(T_FRAMES).map_err(|_| shape_overflow("T_FRAMES", T_FRAMES))?;
    let n_fft_i32 = i32::try_from(N_FFT).map_err(|_| shape_overflow("N_FFT", N_FFT))?;
    let hop_i64 = i64::try_from(HOP).map_err(|_| shape_overflow("HOP", HOP))?;

    // Strided frame view: (T_FRAMES, N_FFT) over the contiguous centered signal.
    let shape: &[i32] = &[t_frames_i32, n_fft_i32];
    // SAFETY: the strided view spans element indices
    //   { i * HOP + j  |  i in [0, T_FRAMES),  j in [0, N_FFT) }.
    // The maximum reachable index is (T_FRAMES-1)*HOP + (N_FFT-1) = centered_len
    // - 1, which is < centered_len (asserted equal above). `centered` is
    // row-contiguous (built via concatenate of 1-D slices), so its flattened
    // element count equals centered_len, satisfying mlx's `as_strided`
    // element-bounds contract. `offset = 0` so no out-of-front access.
    let frames = unsafe { ops::shape::as_strided(centered, &shape, &[hop_i64, 1], 0)? };

    // Window each frame (broadcast the (N_FFT,) window across the T_FRAMES rows),
    // then real-FFT along the frame axis.
    let windowed = ops::arithmetic::multiply(&frames, &self.window)?;
    let spectrum = ops::fft::rfft(&windowed, n_fft_i32, 1, ops::fft::FftNorm::Backward)?;

    // |X|² for the one-sided spectrum → (T_FRAMES, n_freqs) F32.
    spectrum.abs()?.square()
  }
}

#[cfg(feature = "clap")]
impl MelFrontEnd {
  // Intentionally no `Default` impl: `new()` is fallible (it allocates the
  // window + filterbank via mlx), so a `Default` returning `Self` cannot exist.
}

/// Build a periodic (`fftbins=True`) Hann window of length `n` as a `(n,)`
/// `Dtype::F32` [`Array`].
///
/// `w[k] = 0.5 − 0.5·cos(2π·k / n)` for `k ∈ [0, n)` — what
/// `numpy.hanning(n+1)[:-1]`, `torch.hann_window(n, periodic=True)`, and
/// `librosa.filters.get_window("hann", n, fftbins=True)` all return, and what
/// HF `transformers.audio_utils.window_function("hann", periodic=True)` (the
/// `ClapFeatureExtractor` window) uses. Computed in f64 then cast to f32 to
/// match `textclap/src/mel.rs:55-60`'s f64 window.
///
/// This differs from [`crate::audio::dsp::hann_window`], which is the
/// *symmetric* (`periodic=False`) Hann.
fn periodic_hann_f32(n: usize) -> Result<Array> {
  let denom = n as f64;
  let mut w: Vec<f32> = Vec::new();
  w.try_reserve_exact(n).map_err(|e| {
    Error::AllocFailure(crate::error::AllocFailurePayload::new(
      "clap::mel::periodic_hann_f32: window reservation",
      "f32 elements",
      n as u64,
      e,
    ))
  })?;
  for k in 0..n {
    let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (k as f64) / denom).cos();
    w.push(v as f32);
  }
  let n_i32 = i32::try_from(n).map_err(|_| shape_overflow("n_fft", n))?;
  Array::from_slice::<f32>(&w, &[n_i32])
}

/// Repeat-pad / head-truncate `samples` to [`TARGET_SAMPLES`], returning a
/// `(TARGET_SAMPLES,)` `Dtype::F32` [`Array`].
///
/// Mirrors HF `repeatpad` (`textclap/src/mel.rs:194-207`): if `samples` already
/// reaches the target, head-truncate to it; otherwise tile `floor(target / len)`
/// whole copies and zero-pad the remainder.
///
/// The padded buffer is built on the CPU (one `Vec<f32>` of `TARGET_SAMPLES`)
/// then handed to mlx in a single [`Array::from_slice`]; this is the one place
/// the tile/zero-pad is materialized eagerly (it is a fixed `10 s` buffer per
/// clip, independent of input length).
fn repeat_pad(samples: &[f32]) -> Result<Array> {
  debug_assert!(!samples.is_empty(), "caller guarantees non-empty input");
  let mut padded: Vec<f32> = Vec::new();
  padded.try_reserve_exact(TARGET_SAMPLES).map_err(|e| {
    Error::AllocFailure(crate::error::AllocFailurePayload::new(
      "clap::mel::repeat_pad: padded reservation",
      "f32 elements",
      TARGET_SAMPLES as u64,
      e,
    ))
  })?;
  if samples.len() >= TARGET_SAMPLES {
    padded.extend_from_slice(&samples[..TARGET_SAMPLES]);
  } else {
    let n_repeat = TARGET_SAMPLES / samples.len();
    for _ in 0..n_repeat {
      padded.extend_from_slice(samples);
    }
    // Zero-pad the remainder (HF `np.pad` constant 0).
    padded.resize(TARGET_SAMPLES, 0.0);
  }
  let len_i32 =
    i32::try_from(TARGET_SAMPLES).map_err(|_| shape_overflow("TARGET_SAMPLES", TARGET_SAMPLES))?;
  Array::from_slice::<f32>(&padded, &[len_i32])
}

/// Reflect-pad a 1-D [`Array`] by `padding` on each side (librosa `center=True`),
/// returning a `(len + 2·padding,)` [`Array`].
///
/// `prefix = samples[1..=padding][::-1]`, `suffix = samples[len-padding-1..len-1]
/// [::-1]`, then `concatenate([prefix, samples, suffix])` — the same
/// construction as [`crate::audio::dsp::stft`]'s internal reflect pad and
/// `textclap/src/mel.rs:209-223`. Built with mlx [`slice`](ops::indexing::slice)
/// + [`concatenate`](ops::shape::concatenate) (mlx-c has no `reflect` pad mode).
fn reflect_pad(samples: &Array, padding: usize) -> Result<Array> {
  if padding == 0 {
    return samples.try_clone();
  }
  let len = samples.shape()[0];
  // Need indices samples[1..=padding] AND samples[len-padding-1..len-1].
  if len < padding + 1 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap::mel::reflect_pad: samples len for reflect padding",
      "must be >= padding + 1",
      format_smolstr!("len={len}, padding={padding}"),
    )));
  }
  let p_i32 = i32::try_from(padding).map_err(|_| shape_overflow("padding", padding))?;
  let len_i32 = i32::try_from(len).map_err(|_| shape_overflow("samples len", len))?;

  // prefix: samples[padding], samples[padding-1], ..., samples[1] — `padding`
  // elements (slice start=padding, stop=0 exclusive, stride=-1).
  let prefix = ops::indexing::slice(samples, &[p_i32], &[0], &[-1])?;
  // suffix: samples[len-2], samples[len-3], ..., samples[len-padding-1] —
  // `padding` elements. For a negative stride mlx pre-normalizes a negative
  // `stop` by `+ len` before the per-stride logic, so the "left of index 0"
  // sentinel is `stop = -(len + 1)`; here `len - padding - 1 >= 1` (TARGET +
  // n_fft is large), so `stop = len - padding - 2` (the positive index before
  // the last-included one).
  let suffix_stop = len_i32 - p_i32 - 2;
  let suffix = ops::indexing::slice(samples, &[len_i32 - 2], &[suffix_stop], &[-1])?;

  ops::shape::concatenate(&[&prefix, samples, &suffix], 0)
}

/// Build an [`Error::OutOfRange`] for a `usize → i32` shape overflow.
#[inline]
fn shape_overflow(what: &'static str, value: usize) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "clap::mel: shape value",
    "must fit in i32 (i32::MAX = 2147483647)",
    format_smolstr!("{what}={value}"),
  ))
}

#[cfg(test)]
mod tests;
