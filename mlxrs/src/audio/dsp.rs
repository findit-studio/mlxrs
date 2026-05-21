//! DSP primitives: Hann window, STFT, mel filterbank, mel + log-mel spectrogram.
//!
//! Faithful 1:1 port of the corresponding `mlx_audio.dsp` core
//! (`hanning`, `stft`, `mel_filters`) at <https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py>.
//! Out of scope for this PR: iSTFT, ISTFTCache, Kaldi-style features, BS.1770
//! loudness, biquad filters, dither — see [`crate::audio`] for the scope fence.
//!
//! ## API conventions
//! - Window construction is **symmetric** (`periodic=False` in `mlx-audio`):
//!   the first and last samples are zero. This matches scipy's
//!   `windows.hann(N, sym=True)` and the `mlx-audio` default for STFT.
//! - STFT mirrors `mlx_audio.dsp.stft` defaults: `center=True`,
//!   `pad_mode="reflect"`. Output layout is **`(num_frames, n_fft / 2 + 1)`
//!   complex** (mlx-c `rfft` yields `Complex64` natively), as in the
//!   reference.
//! - Mel filterbank uses the HTK formula
//!   (`mel = 2595 * log10(1 + hz / 700)`) and returns shape
//!   **`(n_mels, n_fft / 2 + 1)`**.
//! - `log_mel_spectrogram` uses `log(max(mel, floor))` with `floor` chosen
//!   via the [`LogFloor`] enum (default [`LogFloor::Whisper`] = `1e-10`,
//!   matching the Whisper / mlx-audio front-end). [`LogFloor::Kaldi`] =
//!   `1e-8` matches the floor literal in `mlx-audio/mlx_audio/dsp.py:950`
//!   — floor-constant parity only; the upstream mel-filterbank
//!   `get_mel_banks_kaldi` path is out of scope (see the per-variant
//!   `LogFloor::Kaldi` docs). Tracks mlx-audio's literal, NOT the
//!   upstream kaldi-asr `FbankComputer` floor of `f32::EPSILON`.

use std::f32::consts::PI;

use crate::{
  Array, Error, Result,
  ops::{
    self,
    fft::{self, FftNorm},
  },
};

/// HTK mel formula scale: `mel = MEL_HZ_DIV * log10(1 + hz / MEL_HZ_BREAK)`.
/// Matches `mlx-audio/mlx_audio/dsp.py:510` (`hz_to_mel("htk")` branch).
const MEL_HZ_DIV: f32 = 2595.0;
/// HTK mel formula break frequency (Hz). Matches `mlx-audio/mlx_audio/dsp.py:510`.
const MEL_HZ_BREAK: f32 = 700.0;
/// Log base used by both the HTK forward formula (`log10`) and the inverse
/// (`10^x`). Centralized so a future Slaney-style mel port stays consistent.
const MEL_LOG_BASE: f32 = 10.0;

/// Whisper-style log-mel floor used by `mlx-audio`'s Whisper / mlx-audio
/// front-end path (`mlx-audio/mlx_audio/dsp.py` whisper-style mel path).
const LOG_FLOOR_WHISPER: f32 = 1e-10;
/// `mlx-audio`'s "Kaldi-style" log-mel floor: the literal `1e-8` baked into
/// `mlx-audio/mlx_audio/dsp.py:950` after `get_mel_banks_kaldi`. NOTE this
/// does NOT match the upstream kaldi-asr `FbankComputer` floor of
/// `f32::EPSILON` (~`1.19e-7`) — see [`LogFloor::Kaldi`] for the rationale.
const LOG_FLOOR_KALDI: f32 = 1e-8;

/// The numerical floor applied to mel energies before `log` to avoid
/// `log(0) = -inf` and to bound the dynamic range of the resulting
/// log-mel feature.
///
/// `mlx-audio` ships two distinct log-floor conventions that differ by
/// **2 orders of magnitude** with no rationale documented upstream —
/// `1e-10` in the Whisper-style front-end (deeper floor, wider dynamic
/// range) vs `1e-8` in the `get_mel_banks_kaldi` path. Mixed pipelines
/// produce subtly different features, so we expose the choice
/// explicitly rather than baking in either constant.
///
/// Defaults to [`LogFloor::Whisper`] (the mlxrs reference target;
/// preserves the previous port's behavior byte-identically).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LogFloor {
  /// `1e-10` — matches `mlx-audio`'s Whisper-style mel path.
  #[default]
  Whisper,
  /// `1e-8` — matches `mlx-audio/mlx_audio/dsp.py:950`'s literal floor
  /// (the value clamped before `log` in the `get_mel_banks_kaldi` path).
  ///
  /// **Floor-constant parity only.** This variant changes the
  /// `log(max(mel, floor))` clamp value to `1e-8`; the mel filterbank
  /// produced by [`mel_filter_bank`] is still the HTK formula (see the
  /// `# API conventions` section of this module's doc). Selecting
  /// [`LogFloor::Kaldi`] does NOT route through `get_mel_banks_kaldi`
  /// or otherwise reproduce the full kaldi-style mel pipeline (that
  /// path is out of scope for this PR per the module docs).
  ///
  /// This deliberately tracks `mlx-audio`'s `1e-8` literal — NOT the
  /// upstream kaldi-asr `FbankComputer` floor of `f32::EPSILON`
  /// (~`1.19e-7`). mlxrs is a faithful port of `mlx-audio`, so floor-
  /// constant parity for mlx-audio's two log-mel paths is the goal of
  /// this enum.
  Kaldi,
  /// A custom user-chosen floor. Useful for pipelines mixing mlx-audio's
  /// two floor choices, for floor-constant parity with upstream kaldi-asr
  /// via `LogFloor::Custom(f32::EPSILON)` (subject to the same caveat
  /// as [`LogFloor::Kaldi`]: this changes only the log clamp, not the
  /// upstream mel filterbank path), or for other reproducibility-
  /// sensitive workflows.
  ///
  /// Non-finite (`NaN`, `+/-inf`) and non-positive values (`<= 0.0`,
  /// including `-0.0`) get clamped to [`f32::MIN_POSITIVE`] inside
  /// [`LogFloor::value`] so the resulting `log(floor)` is always finite.
  Custom(f32),
}

impl LogFloor {
  /// The numeric floor value, guaranteed `> 0.0` and finite so
  /// `log(floor)` is always finite.
  pub fn value(self) -> f32 {
    match self {
      LogFloor::Whisper => LOG_FLOOR_WHISPER,
      LogFloor::Kaldi => LOG_FLOOR_KALDI,
      LogFloor::Custom(x) => {
        if x.is_finite() && x > 0.0 {
          x
        } else {
          f32::MIN_POSITIVE
        }
      }
    }
  }
}

/// Symmetric Hann window: `w[k] = 0.5 * (1 - cos(2π k / (n - 1)))` for
/// `k in 0..n`. The first and last samples are zero.
///
/// Matches `mlx_audio.dsp.hanning(n, periodic=False)` (the STFT default).
///
/// # Errors
/// - Returns [`Error::Backend`] when `n < 2`. The reference Python form
///   would divide by zero for `n == 1` (silently producing `NaN`); we
///   reject upfront. `n == 0` would produce an empty zero-length window
///   which is never useful for spectral analysis.
pub fn hann_window(n: usize) -> Result<Array> {
  if n < 2 {
    return Err(Error::Backend {
      message: format!("hann_window: n must be >= 2 (got {n})"),
    });
  }
  // Cap on public-input-driven allocation — defends against an
  // adversarial / fuzzer-supplied `n = usize::MAX` that would otherwise
  // attempt a 16 EiB infallible allocation. Real-world windows are
  // typically <= a few thousand samples; 64 Mi-samples (256 MiB of f32)
  // is a generous ceiling that still excludes pathological inputs.
  if n > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "hann_window: n {n} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }
  let n_i32 = i32::try_from(n).map_err(|_| Error::Backend {
    message: format!("hann_window: n {n} exceeds i32::MAX"),
  })?;

  // Materialize on the CPU (cheap; n is bounded above) via a
  // recoverable `try_reserve_exact` so the cap above (and any
  // future allocation budget) cannot abort the host on a fuzzer input.
  let denom = (n - 1) as f32;
  let mut buf: Vec<f32> = Vec::new();
  buf.try_reserve_exact(n).map_err(|e| Error::Backend {
    message: format!("hann_window: reservation for {n} elements failed: {e}"),
  })?;
  for k in 0..n {
    let theta = 2.0 * PI * (k as f32) / denom;
    buf.push(0.5 * (1.0 - theta.cos()));
  }
  Array::from_slice::<f32>(&buf, &[n_i32])
}

/// Manual `reflect`-mode pad along axis 0 (1-D arrays).
///
/// `prefix = samples[1..=padding][::-1]`, `suffix =
/// samples[len-padding-1..len-1][::-1]`, then `concatenate([prefix,
/// samples, suffix])`. Matches `mlx_audio.dsp.stft._pad(..., pad_mode="reflect")`
/// byte-for-byte. mlx-c's `mlx_pad` only supports `"constant"` and `"edge"`,
/// so reflect is built from slice + concatenate here (same construction
/// the python reference uses).
///
/// # Errors
/// - [`Error::Backend`] if `padding > samples_len - 1` (not enough samples
///   to reflect — would require `samples[len-padding-1]` which underflows
///   for `padding >= len`). The reference Python form would index out of
///   bounds and return a malformed array.
fn reflect_pad_1d(samples: &Array, padding: usize) -> Result<Array> {
  if padding == 0 {
    return samples.try_clone();
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!("reflect_pad_1d: expected 1-D input, got {}-D", shape.len()),
    });
  }
  let len = shape[0];
  // Need indices `samples[1..=padding]` AND `samples[len-padding-1..len-1]`
  // to exist — i.e. `len >= padding + 1`.
  if len < padding + 1 {
    return Err(Error::Backend {
      message: format!(
        "reflect_pad_1d: samples len {len} too short for reflect padding {padding} \
         (need len >= padding + 1)"
      ),
    });
  }

  let p_i32 = i32::try_from(padding).map_err(|_| Error::Backend {
    message: format!("reflect_pad_1d: padding {padding} exceeds i32::MAX"),
  })?;
  let len_i32 = i32::try_from(len).map_err(|_| Error::Backend {
    message: format!("reflect_pad_1d: samples len {len} exceeds i32::MAX"),
  })?;
  // prefix indices: `samples[padding], samples[padding-1], ..., samples[1]`.
  // `slice(start=padding, stop=0, strides=-1)` traverses `padding, padding-1,
  // ..., 1` (exclusive of `stop=0`), yielding exactly `padding` elements.
  // Boundary safe: `0` is a strictly-positive lower bound the slice never
  // reaches (the prefix never goes through index 0 — that would be a
  // double-edge reflect).
  let prefix = ops::indexing::slice(samples, &[p_i32], &[0], &[-1])?;
  // suffix indices: `samples[len-2], samples[len-3], ..., samples[len-padding-1]`,
  // exactly `padding` elements.
  //
  // mlx slice stop is exclusive of the destination, and for negative
  // strides `stop` follows mlx's `normalize_slice` rules (see
  // `mlx/ops.cpp:646` — a negative `stop` is pre-normalized by `+ n`
  // BEFORE the per-stride logic, so the post-normalize "position left of
  // 0" sentinel is `stop = -(n + 1)`, NOT `stop = -1` — `-1` would
  // post-normalize to `n - 1`).
  //
  // Two cases:
  //   1. `len - padding - 1 > 0`: traversal ends at index `len-padding-1`
  //      inclusive, so `stop = len-padding-2` (positive, the index BEFORE
  //      the last-included one).
  //   2. `len - padding - 1 == 0` (boundary: padding == len - 1): traversal
  //      must include index 0, so `stop` must post-normalize to `-1`
  //      ("position left of 0"). Using `stop = -(len + 1)` makes
  //      `e + n = -1`, exactly what mlx wants.
  let suffix_start = len_i32 - 2;
  let suffix_stop = if padding + 1 < len {
    // Inclusive-end is at index `len-padding-1 >= 1`, so the exclusive
    // stop is one less and strictly non-negative.
    len_i32 - p_i32 - 2
  } else {
    // `padding == len - 1`. Inclusive-end is index 0 — needs the
    // post-normalize-to-`-1` sentinel form (`stop = -(n + 1)`).
    //
    // Overflow note (Copilot review #3273868700): both `padding` and `len`
    // were checked to fit `i32` above via `i32::try_from`; combined with
    // `len == padding + 1` in this branch (`padding + 1 >= len` from the
    // else condition, and `len >= padding + 1` from the early check),
    // `len_i32` can be exactly `i32::MAX` (when `padding = i32::MAX - 1`,
    // `len = i32::MAX`). `len_i32 + 1` then overflows. Compute the
    // sentinel in `i64` and reject the (vanishingly rare) overflow as a
    // recoverable `Error::Backend` rather than debug-panicking / wrapping.
    let sentinel_i64 = -(i64::from(len_i32) + 1);
    i32::try_from(sentinel_i64).map_err(|_| Error::Backend {
      message: format!(
        "reflect_pad_1d: reflect-pad sentinel `-(len + 1) = {sentinel_i64}` overflows i32 \
         (len == padding + 1 == {len}, near i32::MAX boundary)"
      ),
    })?
  };
  let suffix = ops::indexing::slice(samples, &[suffix_start], &[suffix_stop], &[-1])?;
  ops::shape::concatenate(&[&prefix, samples, &suffix], 0)
}

/// Short-Time Fourier Transform along axis 0.
///
/// Faithful port of `mlx_audio.dsp.stft(x, n_fft, hop_length, win_length,
/// window="hann", center=True, pad_mode="reflect")`. The window is
/// constructed via [`hann_window`] (the only window kind in this PR;
/// hamming/blackman/bartlett are planned follow-ups). When `win_length`
/// (default = `n_fft`) is smaller than `n_fft`, the window is zero-padded
/// up to `n_fft`. `win_length > n_fft` is rejected — the reference would
/// concatenate zeros, but a longer window than the FFT length cannot occur
/// in any documented `mlx-audio` config.
///
/// Output: `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64`, where
/// `num_frames = 1 + (padded_len - n_fft) / hop_length`. Matches the
/// reference layout.
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `samples` is not 1-D,
///   - `n_fft == 0`, `hop_length == 0`, or `win_length == 0`,
///   - `win_length > n_fft`,
///   - the post-pad sample count is too short to fit a single frame
///     (matches the reference's `Input is too short` raise),
///   - any size exceeds `i32::MAX`.
pub fn stft(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
) -> Result<Array> {
  if n_fft == 0 {
    return Err(Error::Backend {
      message: "stft: n_fft must be > 0".into(),
    });
  }
  if hop_length == 0 {
    return Err(Error::Backend {
      message: "stft: hop_length must be > 0".into(),
    });
  }
  let win_length = win_length.unwrap_or(n_fft);
  if win_length == 0 {
    return Err(Error::Backend {
      message: "stft: win_length must be > 0".into(),
    });
  }
  if win_length > n_fft {
    return Err(Error::Backend {
      message: format!("stft: win_length {win_length} > n_fft {n_fft} (unsupported)"),
    });
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!("stft: expected 1-D input, got {}-D", shape.len()),
    });
  }

  // Window construction (hann; padded to n_fft if win_length < n_fft).
  let window = hann_window(win_length)?;
  let window = if win_length < n_fft {
    let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
    let pad_axes = [0_i32];
    let pad_low = [0_i32];
    let pad_high = [
      i32::try_from(n_fft - win_length).map_err(|_| Error::Backend {
        message: format!("stft: window pad {} exceeds i32::MAX", n_fft - win_length),
      })?,
    ];
    ops::shape::pad(
      &window,
      &pad_axes,
      &pad_low,
      &pad_high,
      &pad_value,
      c"constant",
    )?
  } else {
    window
  };

  // `center=True, pad_mode="reflect"` (reference default).
  let padded = reflect_pad_1d(samples, n_fft / 2)?;
  let padded_len = padded.shape()[0];

  // Pre-frame validation: need at least one frame.
  if padded_len < n_fft {
    return Err(Error::Backend {
      message: format!(
        "stft: input is too short (padded_len={padded_len}) for n_fft={n_fft} \
         (need padded_len >= n_fft)"
      ),
    });
  }
  let num_frames = 1 + (padded_len - n_fft) / hop_length;
  if num_frames == 0 {
    return Err(Error::Backend {
      message: format!(
        "stft: input is too short for n_fft={n_fft} hop_length={hop_length} \
         (computed num_frames = 0)"
      ),
    });
  }

  // SAFETY pre-condition: the reachable element range of the strided view
  // is `(num_frames - 1) * hop_length + n_fft - 1`. We assert this is
  // strictly less than `padded_len`, so every read is in-bounds.
  let last_element_index = (num_frames - 1)
    .checked_mul(hop_length)
    .and_then(|v| v.checked_add(n_fft))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "stft: reachable element range overflows usize \
         (num_frames={num_frames}, hop_length={hop_length}, n_fft={n_fft})"
      ),
    })?;
  if last_element_index > padded_len {
    return Err(Error::Backend {
      message: format!(
        "stft: derived frame bounds {last_element_index} > padded len {padded_len} \
         (n_fft={n_fft}, hop_length={hop_length}, num_frames={num_frames}) — \
         internal invariant violated"
      ),
    });
  }
  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| Error::Backend {
    message: format!("stft: num_frames {num_frames} exceeds i32::MAX"),
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| Error::Backend {
    message: format!("stft: n_fft {n_fft} exceeds i32::MAX"),
  })?;
  let hop_i64 = i64::try_from(hop_length).map_err(|_| Error::Backend {
    message: format!("stft: hop_length {hop_length} exceeds i64::MAX"),
  })?;

  // PR #50 changed `as_strided`'s shape param to `&impl IntoShape`; an
  // array literal `&[i32; 2]` doesn't impl `IntoShape`, so we bind a
  // slice first and pass `&shape` (matching `IntoShape for &[i32]`).
  let shape: &[i32] = &[num_frames_i32, n_fft_i32];
  // SAFETY: the strided view spans element indices
  //   { i * hop_length + j  |  i in [0, num_frames),  j in [0, n_fft) }
  // The maximum reachable index is
  //   (num_frames - 1) * hop_length + (n_fft - 1) = last_element_index - 1.
  // We asserted `last_element_index <= padded_len` above, so every reachable
  // element is in `[0, padded_len)`. `padded` is row-contiguous (built via
  // concatenate of 1-D slices), so its flattened element count equals
  // `padded_len`, satisfying mlx's `as_strided` element-bounds contract.
  // `offset=0` so no out-of-front access either.
  let frames = unsafe { ops::shape::as_strided(&padded, &shape, &[hop_i64, 1], 0)? };

  // `frames * window` broadcasts the `(n_fft,)` window across each frame.
  let windowed = ops::arithmetic::multiply(&frames, &window)?;
  // rfft over the last axis (axis 1) with explicit length n_fft.
  fft::rfft(&windowed, n_fft_i32, 1, FftNorm::Backward)
}

/// HTK mel scale: `mel = 2595 * log10(1 + hz / 700)`.
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
  MEL_HZ_DIV * (1.0 + hz / MEL_HZ_BREAK).log10()
}

/// Inverse HTK mel scale: `hz = 700 * (10^(mel / 2595) - 1)`.
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
  MEL_HZ_BREAK * (MEL_LOG_BASE.powf(mel / MEL_HZ_DIV) - 1.0)
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)`.
///
/// Faithful port of `mlx_audio.dsp.mel_filters(sample_rate, n_fft, n_mels,
/// f_min, f_max, norm=None, mel_scale="htk")` — the HTK formula only;
/// Slaney normalization is a planned follow-up.
///
/// `f_max` defaults to `sample_rate / 2` (Nyquist) when `None`. The reference
/// builds frequency points via `mx.linspace(0, sample_rate // 2, n_freqs)`
/// which integer-divides the Nyquist — we mirror that exactly (using
/// `sample_rate as f32 / 2.0` would drift by 0.5 for odd sample rates).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `n_fft == 0`,
///   - `n_mels == 0` (no filters requested),
///   - `f_min < 0` or `f_max <= f_min`,
///   - any size exceeds `i32::MAX`.
pub fn mel_filter_bank(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  if n_fft == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: n_fft must be > 0".into(),
    });
  }
  if n_mels == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: n_mels must be > 0".into(),
    });
  }
  if sample_rate == 0 {
    return Err(Error::Backend {
      message: "mel_filter_bank: sample_rate must be > 0".into(),
    });
  }
  let f_max = f_max.unwrap_or((sample_rate / 2) as f32);
  if !(f_min >= 0.0 && f_max > f_min) {
    return Err(Error::Backend {
      message: format!("mel_filter_bank: invalid f_min={f_min} / f_max={f_max}"),
    });
  }

  // `n_freqs = n_fft / 2 + 1`; `n_fft / 2 <= usize::MAX / 2`, so `+ 1`
  // cannot overflow `usize`. Bound on i32 happens after the multiplication
  // check below.
  let n_freqs = n_fft / 2 + 1;
  // `n_pts = n_mels + 2`; check for overflow on `n_mels = usize::MAX` /
  // `usize::MAX - 1` before we walk `0..n_pts`.
  let n_pts = n_mels.checked_add(2).ok_or_else(|| Error::Backend {
    message: format!("mel_filter_bank: n_mels {n_mels} + 2 overflows usize"),
  })?;
  // Bank size: `n_mels * n_freqs`. The reference uses an mlx broadcast
  // graph; we materialize one `Vec<f32>` of the same logical size, so we
  // must reject any combination that would attempt a multi-GB allocation
  // (the python form would silently swap or OOM-kill).
  let bank_len = n_mels.checked_mul(n_freqs).ok_or_else(|| Error::Backend {
    message: format!(
      "mel_filter_bank: n_mels * n_freqs overflows usize \
       (n_mels={n_mels}, n_freqs={n_freqs})"
    ),
  })?;
  // i32 bounds on the final mlx shape go here, BEFORE any large allocation.
  let n_mels_i32 = i32::try_from(n_mels).map_err(|_| Error::Backend {
    message: format!("mel_filter_bank: n_mels {n_mels} exceeds i32::MAX"),
  })?;
  let n_freqs_i32 = i32::try_from(n_freqs).map_err(|_| Error::Backend {
    message: format!("mel_filter_bank: n_freqs {n_freqs} exceeds i32::MAX"),
  })?;

  // `all_freqs[i] = i * (sample_rate / 2) / (n_freqs - 1)` for the python
  // `mx.linspace(0, sample_rate // 2, n_freqs)` form. Build CPU-side;
  // n_freqs is small for any reasonable n_fft (e.g. 201 for n_fft=400).
  // Use `try_reserve_exact` for the same reason as `bank` below — a
  // crafted n_fft can drive n_freqs into multi-GB territory.
  let nyq = (sample_rate / 2) as f32;
  let denom = (n_freqs as f32 - 1.0).max(1.0);
  let mut all_freqs: Vec<f32> = Vec::new();
  all_freqs
    .try_reserve_exact(n_freqs)
    .map_err(|e| Error::Backend {
      message: format!("mel_filter_bank: reservation for n_freqs={n_freqs} failed: {e}"),
    })?;
  for i in 0..n_freqs {
    all_freqs.push(i as f32 * nyq / denom);
  }

  // Mel grid: `n_mels + 2` points (the +2 give the outer triangle edges).
  let m_min = hz_to_mel(f_min);
  let m_max = hz_to_mel(f_max);
  let m_denom = (n_pts as f32 - 1.0).max(1.0);
  let mut f_pts: Vec<f32> = Vec::new();
  f_pts.try_reserve_exact(n_pts).map_err(|e| Error::Backend {
    message: format!("mel_filter_bank: reservation for n_pts={n_pts} failed: {e}"),
  })?;
  for i in 0..n_pts {
    let m = m_min + (m_max - m_min) * (i as f32) / m_denom;
    f_pts.push(mel_to_hz(m));
  }

  // Build the filterbank directly on the CPU as `(n_mels, n_freqs)` to
  // avoid the reference's allocation chain (linspace + 4 broadcast ops);
  // this is the only place we elide an mlx-graph step in this PR — the
  // mel filter is a one-shot constant matrix per `(sample_rate, n_fft,
  // n_mels)` triple, and the on-device construction has no perf benefit.
  // Logged in docs/rust-golden-standard-followups.md (AUDIO-2).
  //
  // Use `try_reserve_exact` so a multi-GB request from a forged input
  // returns a recoverable `Error::Backend` rather than aborting on the
  // allocator's OOM panic (Rust's default behavior is to abort, not
  // unwind, on allocation failure — `Vec::with_capacity` and `vec![]`
  // share that abort path).
  let mut bank: Vec<f32> = Vec::new();
  bank
    .try_reserve_exact(bank_len)
    .map_err(|e| Error::Backend {
      message: format!("mel_filter_bank: allocation of {bank_len} f32 elements failed: {e}"),
    })?;
  bank.resize(bank_len, 0.0);
  for m in 0..n_mels {
    let left = f_pts[m];
    let center = f_pts[m + 1];
    let right = f_pts[m + 2];
    let lc = center - left;
    let cr = right - center;
    // Guard against zero-width triangles (collapsed mel bins). The
    // reference would NaN/inf on the division; we keep the bin at zero.
    if lc <= 0.0 || cr <= 0.0 {
      continue;
    }
    for (f, &freq) in all_freqs.iter().enumerate() {
      let up = (freq - left) / lc;
      let down = (right - freq) / cr;
      let v = up.min(down).max(0.0);
      bank[m * n_freqs + f] = v;
    }
  }

  Array::from_slice::<f32>(&bank, &[n_mels_i32, n_freqs_i32])
}

/// Mel spectrogram: `mel_bank @ |stft(samples)|^2`.
///
/// Returns shape `(n_mels, num_frames)` `Dtype::F32`. Combines [`stft`],
/// magnitude-squared, and [`mel_filter_bank`] in the canonical Whisper /
/// mlx-audio order.
///
/// # Errors
/// Propagates from [`stft`] and [`mel_filter_bank`].
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  let spec = stft(samples, n_fft, hop_length, win_length)?;
  // `|stft|^2` — `abs` of Complex64 yields F32 magnitudes, then square.
  let mag = spec.abs()?;
  let power = mag.square()?;
  // `power` is `(num_frames, n_freqs)`; mel is `(n_mels, n_freqs)`.
  // Mel-spec layout in mlx-audio / Whisper is `(n_mels, num_frames)` =
  // `mel @ power.T`.
  let mel = mel_filter_bank(n_mels, n_fft, sample_rate, f_min, f_max)?;
  let power_t = power.transpose()?;
  ops::linalg_basic::matmul(&mel, &power_t)
}

/// Log-mel spectrogram: `log(max(mel_spectrogram, floor))` with `floor =
/// [`LogFloor::default`]` (= `1e-10`, Whisper / mlx-audio convention).
///
/// Thin forward to [`log_mel_spectrogram_with`] with the default floor —
/// output is byte-identical to the pre-`LogFloor` behavior. Use
/// [`log_mel_spectrogram_with`] to pick a different floor explicitly.
///
/// # Errors
/// Propagates from [`mel_spectrogram`].
#[allow(clippy::too_many_arguments)]
pub fn log_mel_spectrogram(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  log_mel_spectrogram_with(
    samples,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    sample_rate,
    f_min,
    f_max,
    LogFloor::default(),
  )
}

/// Log-mel spectrogram with an explicit log floor — `log(max(mel, floor.value()))`.
///
/// Lets the caller pick between [`LogFloor::Whisper`] (`1e-10`, the default
/// matching the mlx-audio Whisper-style front-end), [`LogFloor::Kaldi`]
/// (`1e-8`, matching the floor literal in `mlx-audio/mlx_audio/dsp.py:950`),
/// or [`LogFloor::Custom`] for downstream reproducibility-sensitive
/// workflows. See [`LogFloor`] for the rationale and the floor-constant-
/// only scope (the mel filterbank stays the HTK one — `LogFloor::Kaldi`
/// does NOT swap in `get_mel_banks_kaldi`).
///
/// # Errors
/// Propagates from [`mel_spectrogram`].
#[allow(clippy::too_many_arguments)]
pub fn log_mel_spectrogram_with(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  n_mels: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  floor: LogFloor,
) -> Result<Array> {
  let mel = mel_spectrogram(
    samples,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    sample_rate,
    f_min,
    f_max,
  )?;
  // `maximum(mel, floor)` then `log`. Build the floor as a 0-D scalar so
  // it broadcasts against `mel`'s `(n_mels, num_frames)` shape.
  let eps = Array::full::<f32>(&[0i32; 0], floor.value())?;
  let floored = ops::arithmetic::maximum(&mel, &eps)?;
  floored.log()
}
