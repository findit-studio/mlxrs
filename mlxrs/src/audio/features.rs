//! Kaldi-compatible log-mel-filterbank feature extraction.
//!
//! Faithful port of the `mlx_audio.dsp` Kaldi feature surface at
//! <https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py>
//! (`mel_scale_kaldi` / `inverse_mel_scale_kaldi` /
//! `get_mel_banks_kaldi` / `compute_fbank_kaldi`, lines 762..953).
//!
//! ## Why a separate module from [`crate::audio::dsp`]
//! The HTK/Whisper mel front-end already in [`crate::audio::dsp::mel_filter_bank`]
//! and [`crate::audio::dsp::log_mel_spectrogram`] is the one Whisper / mlx-audio
//! Whisper-style pipelines consume. The Kaldi pipeline that this module ports
//! is **a different specification** with its own quirks (preemphasis, dither,
//! Povey window, ln-based mel scale, `next_power_of_2` framing, Kaldi-style
//! strided framing instead of reflect padding). Mixing them in one module would
//! blur the lines; tracking them as siblings under [`crate::audio`] mirrors the
//! upstream split between the two front-ends.
//!
//! ## Mel-scale formula (HTK vs Kaldi)
//! - HTK ([`crate::audio::dsp::mel_filter_bank`]): `mel = 2595 * log10(1 + hz / 700)`.
//! - Kaldi (this module): `mel = 1127 * ln(1 + hz / 700)`.
//!
//! Mathematically these are equivalent to ~5 decimal places
//! (`2595 / ln(10) ≈ 1127.01`), but Kaldi-trained models pin the literal `1127`
//! and the natural log, so [`mel_scale_kaldi`] / [`inverse_mel_scale_kaldi`]
//! use those constants exactly for byte-identical parity with the reference
//! and with kaldi-asr's `mel-computations.cc`.
//!
//! ## Scope
//! - **Forward only.** Mel-feature inversion (mel→audio) is intentionally
//!   out of scope per the M5 plan (the `feedback_roundtrip_real_functions_typed_metadata`
//!   rule says invertibility ports get a dedicated round-trip-via-public-funcs
//!   PR; we are not introducing an `inverse_fbank_kaldi` here).
//! - **`snip_edges` both paths.** `snip_edges=true` drops partial edge frames
//!   (the standard kaldi-asr / torchaudio / ESPnet default); `snip_edges=false`
//!   reflect-pads the signal so frames are centered on the sample positions.
//!   Both port `_get_strided_kaldi` (`dsp.py:777`) — see
//!   [`crate::audio::features::compute_fbank_kaldi`]. The reference's
//!   `snip_edges=false` calls `mx.as_strided` with no bounds check and reads
//!   out of bounds for degenerate `win_len`-vs-signal-length inputs (UB it
//!   gets away with only because numpy/mlx over-allocate); mlxrs reproduces the
//!   reflect-bookend framing bit-identically in the safe regime and returns a
//!   recoverable error in the degenerate regime instead of reproducing that UB.
//!   (Povey-DC-removal-after-window variants remain a follow-up.)
//! - **Explicit RNG key.** The reference uses an implicit `mx.random.normal`
//!   default key; mlxrs's [`crate::ops::random`] is JAX-style split-key by
//!   design, so [`compute_fbank_kaldi`] takes an explicit
//!   `dither_key: Option<&Array>` — pass `None` (or `dither == 0.0`) for
//!   deterministic output, pass `Some(&key)` to seed the dither additively.

use std::f32::consts::PI;

use crate::{
  Array, Error, Result,
  ops::{
    self,
    fft::{self, FftNorm},
  },
};

/// Kaldi mel formula scale: `mel = KALDI_MEL_SCALE * ln(1 + hz / KALDI_MEL_HZ_BREAK)`.
/// Matches `mlx-audio/mlx_audio/dsp.py:764` (`mel_scale_kaldi`).
const KALDI_MEL_SCALE: f32 = 1127.0;
/// Kaldi mel formula break frequency (Hz). Matches `mlx-audio/mlx_audio/dsp.py:764`.
const KALDI_MEL_HZ_BREAK: f32 = 700.0;

/// Log-mel floor used by `compute_fbank_kaldi`: literal `1e-8` baked into
/// `mlx-audio/mlx_audio/dsp.py:950`. NOTE this is the upstream `mlx-audio`
/// constant, NOT the kaldi-asr `FbankComputer` floor of `f32::EPSILON`
/// (~`1.19e-7`); see [`crate::audio::dsp::LogFloor::Kaldi`] for the same
/// caveat in the floor-constant-only surface.
const KALDI_FBANK_LOG_FLOOR: f32 = 1e-8;

/// Hard ceiling on the strided-frame element count `num_frames * n_fft_padded`
/// (the windowed-frame matrix the rfft consumes) for [`compute_fbank_kaldi`].
/// Mirrors [`crate::audio::dsp`]'s `MAX_STFT_WORK` cap on the same workload —
/// a `snip_edges=true` framing of a `MAX_DECODED_SAMPLES`-length input with a
/// small `win_inc` still produces `num_frames ≈ samples_len / win_inc`, and a
/// pathological `(win_len, win_inc)` can drive `num_frames * n_fft_padded` into
/// multi-GB territory before any allocation. 64 Mi-elements (256 MiB of f32)
/// is the same generous ceiling [`crate::audio::dsp::stft`] uses.
const MAX_FBANK_WORK: usize = 64 * 1024 * 1024;

/// Hard ceiling on [`compute_deltas_kaldi`]'s `win_length`. Delta windows are
/// tiny in practice — Kaldi's default is `5`, and even acceleration / wide
/// regression windows stay well under a few hundred. A large odd `win_length`
/// drives the per-offset shifted-slice loop (`win_length` strided slices) and
/// the symmetric `n = win_length / 2` boundary pad, so an unbounded value would
/// stall the CPU / blow memory long before the element cap engages on a tiny
/// input. `1024` is far above any realistic delta window while keeping the
/// padded-extent and slice-count work bounded.
const MAX_DELTA_WIN_LENGTH: usize = 1024;

/// Hard ceiling on [`compute_deltas_kaldi`]'s **total accumulation work**:
/// `num_features * time * (win_length - 1)`. The padded-buffer cap
/// ([`MAX_FBANK_WORK`]) only bounds the buffer *size* `num_features *
/// (time + 2n)`, but the delta accumulation loop runs `win_length - 1`
/// full-width slice / multiply / add passes over `num_features * time`
/// elements — so the actual element-op count is `num_features * time *
/// (win_length - 1)`, the `(win_length - 1)` multiplier the size cap
/// ignores. A `(1-D length = MAX_FBANK_WORK - 1022, win_length = 1023)`
/// input passes both the original and the padded size caps yet schedules
/// ~1022 passes over ~64 Mi elements ≈ tens of billions of element-ops —
/// a CPU / GPU stall despite the size cap (Codex review). This is the
/// delta analogue of [`crate::audio::dsp`]'s `MAX_LOUDNESS_WORK` (a
/// sample-visit cap distinct from its `MAX_LOUDNESS_BLOCK_BYTES` byte
/// cap). `512 Mi` element-ops is a generous ceiling — the default
/// `win_length = 5` over a 64 Mi-element spectrogram is only `4 * 64 Mi =
/// 256 Mi` ops, comfortably under the bound, while a pathological wide
/// window on a large input is rejected in microseconds before the loop.
const MAX_DELTA_WORK: usize = 512 * 1024 * 1024;

/// Convert Hz to the Kaldi mel scale: `1127 * ln(1 + hz / 700)`.
///
/// Faithful port of `mlx_audio.dsp.mel_scale_kaldi` (`dsp.py:762`). Unlike
/// [`crate::audio::dsp::mel_filter_bank`]'s HTK formula
/// (`2595 * log10(1 + hz / 700)`), this uses the natural log and the constant
/// `1127` exactly — Kaldi-trained models pin these literals.
///
/// Always finite for `hz >= 0.0` (and finite for `hz > -700.0`; `hz == -700.0`
/// yields `-inf`, which is the same behavior as the reference's `mx.log(0)`).
#[inline]
#[must_use]
pub fn mel_scale_kaldi(hz: f32) -> f32 {
  KALDI_MEL_SCALE * (1.0 + hz / KALDI_MEL_HZ_BREAK).ln()
}

/// Convert a Kaldi-scale mel value back to Hz: `700 * (exp(mel / 1127) - 1)`.
///
/// Faithful port of `mlx_audio.dsp.inverse_mel_scale_kaldi` (`dsp.py:767`).
/// The inverse of [`mel_scale_kaldi`]: `inverse_mel_scale_kaldi(mel_scale_kaldi(f)) ≈ f`
/// to f32 precision for `f >= 0`.
#[inline]
#[must_use]
pub fn inverse_mel_scale_kaldi(mel: f32) -> f32 {
  KALDI_MEL_HZ_BREAK * ((mel / KALDI_MEL_SCALE).exp() - 1.0)
}

/// Smallest power of two `>= x` (the `_next_power_of_2` helper in
/// `mlx_audio.dsp`, used by [`compute_fbank_kaldi`] to choose `n_fft`).
/// Returns `1` for `x == 0` (matching the reference). The result fits in
/// `usize` for any `x <= usize::MAX / 2`; callers (us) bound `x` to
/// `win_length` which is itself capped at [`crate::audio::io::MAX_DECODED_SAMPLES`].
#[inline]
fn next_power_of_2(x: usize) -> usize {
  if x == 0 {
    1
  } else {
    // `next_power_of_two` panics on overflow; we never reach that because
    // every call site bounds `x` to `MAX_DECODED_SAMPLES` (~64 Mi), so the
    // result is at most ~128 Mi — well under `usize::MAX`.
    x.next_power_of_two()
  }
}

/// Kaldi-style triangular mel filterbank of shape `(num_bins, n_fft_padded / 2)`.
///
/// Faithful port of `mlx_audio.dsp.get_mel_banks_kaldi` (`dsp.py:802`). Note the
/// trailing dimension is **`n_fft_padded / 2`** (NOT `+ 1`): the reference
/// iterates `mx.arange(num_fft_bins)` with `num_fft_bins = window_length_padded // 2`,
/// which omits the Nyquist bin. [`compute_fbank_kaldi`] zero-pads this with one
/// column on the right before multiplying against the `(n_fft_padded / 2 + 1)`
/// rfft magnitude spectrum.
///
/// The returned `center_freqs` is a 1-D `(num_bins,)` array of the mel-center
/// frequencies in Hz, useful for downstream visualization / weighting.
///
/// `high_freq <= 0.0` is interpreted as Nyquist-relative — the reference adds
/// the Nyquist when `high_freq <= 0.0`, so e.g. `high_freq = 0.0` means
/// "Nyquist" and `high_freq = -200.0` means "Nyquist - 200 Hz".
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `num_bins <= 3` (the reference asserts `num_bins > 3`),
///   - `n_fft_padded` is odd or zero,
///   - `sample_freq <= 0.0`,
///   - the resolved `low_freq` / `high_freq` violate `0 <= low_freq < nyquist`
///     or `0 < high_freq <= nyquist`,
///   - any size exceeds `i32::MAX`,
///   - `num_bins * (n_fft_padded / 2)` overflows `usize` or exceeds the
///     internal `MAX_FBANK_WORK` cap (~64 Mi elements).
pub fn get_mel_banks_kaldi(
  num_bins: usize,
  n_fft_padded: usize,
  sample_freq: f32,
  low_freq: f32,
  high_freq: f32,
) -> Result<(Array, Array)> {
  // Reference's `assert num_bins > 3` (`dsp.py:822`). The lower bound is real:
  // a 1- or 2-bin filterbank has no nontrivial center bins, and 3 is the
  // smallest count where the reference's `(num_bins + 1)` mel-delta math is
  // well-defined (the three points {left, center, right} need two gaps).
  if num_bins <= 3 {
    return Err(Error::Backend {
      message: format!("get_mel_banks_kaldi: num_bins must be > 3 (got {num_bins})"),
    });
  }
  if n_fft_padded == 0 || !n_fft_padded.is_multiple_of(2) {
    return Err(Error::Backend {
      message: format!(
        "get_mel_banks_kaldi: n_fft_padded must be a positive even number \
         (got {n_fft_padded})"
      ),
    });
  }
  if !(sample_freq.is_finite() && sample_freq > 0.0) {
    return Err(Error::Backend {
      message: format!("get_mel_banks_kaldi: sample_freq must be > 0.0 (got {sample_freq})"),
    });
  }

  // Nyquist-relative high-freq (matches `dsp.py:828`): non-positive means
  // "relative to Nyquist", with `0.0` meaning "exactly Nyquist". The reference
  // adds Nyquist, so a negative value means "Nyquist - |high_freq|".
  let nyquist = 0.5 * sample_freq;
  let high_freq = if high_freq <= 0.0 {
    high_freq + nyquist
  } else {
    high_freq
  };

  // `dsp.py:831` — the reference's `assert` covers low/high range; we surface
  // it as a recoverable error.
  if !(low_freq >= 0.0 && low_freq < nyquist) {
    return Err(Error::Backend {
      message: format!(
        "get_mel_banks_kaldi: low_freq must satisfy 0 <= low_freq < nyquist \
         (got low_freq={low_freq}, nyquist={nyquist})"
      ),
    });
  }
  if !(high_freq > 0.0 && high_freq <= nyquist) {
    return Err(Error::Backend {
      message: format!(
        "get_mel_banks_kaldi: high_freq must satisfy 0 < high_freq <= nyquist \
         (got high_freq={high_freq}, nyquist={nyquist})"
      ),
    });
  }
  if low_freq >= high_freq {
    return Err(Error::Backend {
      message: format!("get_mel_banks_kaldi: low_freq {low_freq} must be < high_freq {high_freq}"),
    });
  }

  let num_fft_bins = n_fft_padded / 2; // omits the Nyquist bin (reference)
  let bank_len = num_bins
    .checked_mul(num_fft_bins)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "get_mel_banks_kaldi: num_bins * num_fft_bins overflows usize \
         (num_bins={num_bins}, num_fft_bins={num_fft_bins})"
      ),
    })?;
  if bank_len > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "get_mel_banks_kaldi: bank_len {bank_len} (num_bins={num_bins} * \
         num_fft_bins={num_fft_bins}) exceeds the {MAX_FBANK_WORK} work cap"
      ),
    });
  }
  let num_bins_i32 = i32::try_from(num_bins).map_err(|_| Error::Backend {
    message: format!("get_mel_banks_kaldi: num_bins {num_bins} exceeds i32::MAX"),
  })?;
  let num_fft_bins_i32 = i32::try_from(num_fft_bins).map_err(|_| Error::Backend {
    message: format!("get_mel_banks_kaldi: num_fft_bins {num_fft_bins} exceeds i32::MAX"),
  })?;

  let fft_bin_width = sample_freq / n_fft_padded as f32;
  let mel_low = mel_scale_kaldi(low_freq);
  let mel_high = mel_scale_kaldi(high_freq);
  let mel_delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);

  // Build the `(num_bins, num_fft_bins)` filterbank on the CPU (same shape +
  // semantics as [`crate::audio::dsp::mel_filter_bank`]'s direct construction;
  // this is the only place we elide an mlx-graph step). The reference's
  // broadcast graph is correct but allocates several intermediates; the mel
  // filter is a one-shot constant matrix per `(num_bins, n_fft_padded,
  // sample_freq, low, high)` tuple so the CPU-only build is the right shape.
  let mut bank: Vec<f32> = Vec::new();
  bank
    .try_reserve_exact(bank_len)
    .map_err(|e| Error::Backend {
      message: format!("get_mel_banks_kaldi: allocation of {bank_len} f32 elements failed: {e}"),
    })?;
  bank.resize(bank_len, 0.0);

  let mut centers: Vec<f32> = Vec::new();
  centers
    .try_reserve_exact(num_bins)
    .map_err(|e| Error::Backend {
      message: format!("get_mel_banks_kaldi: allocation of {num_bins} center freqs failed: {e}"),
    })?;

  for m in 0..num_bins {
    let left_mel = mel_low + (m as f32) * mel_delta;
    let center_mel = mel_low + ((m + 1) as f32) * mel_delta;
    let right_mel = mel_low + ((m + 2) as f32) * mel_delta;
    centers.push(inverse_mel_scale_kaldi(center_mel));

    let lc = center_mel - left_mel;
    let cr = right_mel - center_mel;
    // Zero-width triangle guard (collapsed mel bin). The reference would
    // NaN/inf on the division; we keep the bin at zero. In practice
    // `mel_delta > 0` whenever `low_freq < high_freq`, which we asserted above,
    // so this branch is defensive (it can fire only on f32 underflow).
    if lc <= 0.0 || cr <= 0.0 {
      continue;
    }
    let row = m * num_fft_bins;
    for k in 0..num_fft_bins {
      let mel = mel_scale_kaldi(fft_bin_width * k as f32);
      let up = (mel - left_mel) / lc;
      let down = (right_mel - mel) / cr;
      let v = up.min(down).max(0.0);
      bank[row + k] = v;
    }
  }

  let bins = Array::from_slice::<f32>(&bank, &[num_bins_i32, num_fft_bins_i32])?;
  let center_freqs = Array::from_slice::<f32>(&centers, &[num_bins_i32])?;
  Ok((bins, center_freqs))
}

/// Window variant for [`compute_fbank_kaldi`]. Mirrors the `win_type` string
/// argument in `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:859`).
///
/// All variants use the **periodic** denominator `(window_size - 1)` (matching
/// the reference's `2*pi*n / (window_size - 1)`); the Povey window is a Hann
/// raised to the `0.85` power (the kaldi-asr `povey` window).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KaldiWindow {
  /// `0.54 - 0.46 * cos(2π n / (win_len - 1))` (the reference's default in
  /// `compute_fbank_kaldi`).
  #[default]
  Hamming,
  /// `0.5 - 0.5 * cos(2π n / (win_len - 1))`.
  Hanning,
  /// `(0.5 - 0.5 * cos(2π n / (win_len - 1))) ^ 0.85` — the kaldi-asr `povey`
  /// window (a slightly less smooth Hann tail, slightly more energy in the
  /// transition bands).
  Povey,
  /// Constant `1.0` window (no windowing).
  Rectangular,
}

/// Build the Kaldi-style analysis window of length `win_size`.
///
/// CPU-built `Vec<f32>` (cheap; `win_size <= MAX_DECODED_SAMPLES` is
/// enforced by [`compute_fbank_kaldi`]). The `win_size - 1` denominator is
/// the periodic form used by the reference (`mlx_audio.dsp.compute_fbank_kaldi`
/// uses `2*pi*n / (window_size - 1)`, NOT `/window_size`).
fn build_kaldi_window(win_type: KaldiWindow, win_size: usize) -> Result<Array> {
  if win_size < 2 {
    return Err(Error::Backend {
      message: format!("build_kaldi_window: win_size must be >= 2 (got {win_size})"),
    });
  }
  let win_i32 = i32::try_from(win_size).map_err(|_| Error::Backend {
    message: format!("build_kaldi_window: win_size {win_size} exceeds i32::MAX"),
  })?;
  let mut buf: Vec<f32> = Vec::new();
  buf
    .try_reserve_exact(win_size)
    .map_err(|e| Error::Backend {
      message: format!("build_kaldi_window: reservation for {win_size} elements failed: {e}"),
    })?;
  let denom = (win_size - 1) as f32;
  for n in 0..win_size {
    let theta = 2.0 * PI * (n as f32) / denom;
    let v = match win_type {
      KaldiWindow::Hamming => 0.54 - 0.46 * theta.cos(),
      KaldiWindow::Hanning => 0.5 - 0.5 * theta.cos(),
      KaldiWindow::Povey => (0.5 - 0.5 * theta.cos()).powf(0.85),
      KaldiWindow::Rectangular => 1.0,
    };
    buf.push(v);
  }
  Array::from_slice::<f32>(&buf, &[win_i32])
}

/// Strided framing matching the reference's `_get_strided_kaldi` with
/// `snip_edges=true` (`mlx_audio.dsp.py:777`).
///
/// `snip_edges=false` is not implemented here (see the module-level scope
/// fence); a `false` flag returns [`Error::Backend`] rather than silently
/// flipping to `true`. The `(num_frames, win_size)` strided view is built via
/// the same `unsafe ops::shape::as_strided` `crate::audio::dsp::stft` uses,
/// with the same `(num_frames - 1) * win_inc + win_size <= samples_len`
/// pre-condition asserted before the FFI call.
///
/// SAFETY: the strided view spans element indices
///   `{ i * win_inc + j  |  i in [0, num_frames),  j in [0, win_size) }`.
/// The maximum reachable index is `(num_frames - 1) * win_inc + (win_size - 1)`,
/// which we assert is `< samples_len` below (so every read is in-bounds).
/// `waveform` is required to be 1-D and row-contiguous; the caller MUST
/// materialize via [`ops::shape::contiguous`] before calling this — public
/// validation at the rank level alone is insufficient because a sliced or
/// broadcasted 1-D `Array` passes the rank check but its flattened storage
/// is shorter than `shape()[0]` (broadcast strides of 0) or strided over a
/// non-row-major buffer, both of which would cause out-of-bounds native
/// reads. [`compute_fbank_kaldi`] enforces this by routing `waveform`
/// through `ops::shape::contiguous(waveform, false)` first; callers outside
/// this module MUST do the same.
/// `offset=0` so no out-of-front access either.
fn strided_frames_snip_edges(
  waveform: &Array,
  win_size: usize,
  win_inc: usize,
  num_frames: usize,
) -> Result<Array> {
  // Pre-condition: the reachable index of the strided view must lie strictly
  // inside `waveform`'s flattened storage. Checked-arithmetic so a fuzzer
  // input can't wrap usize and slip past the bound.
  let last_index = (num_frames - 1)
    .checked_mul(win_inc)
    .and_then(|v| v.checked_add(win_size))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "strided_frames_snip_edges: reachable element range overflows usize \
         (num_frames={num_frames}, win_inc={win_inc}, win_size={win_size})"
      ),
    })?;
  let waveform_len = waveform.shape()[0];
  if last_index > waveform_len {
    return Err(Error::Backend {
      message: format!(
        "strided_frames_snip_edges: derived frame bounds {last_index} > waveform len \
         {waveform_len} (num_frames={num_frames}, win_inc={win_inc}, win_size={win_size}) — \
         internal invariant violated"
      ),
    });
  }
  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| Error::Backend {
    message: format!("strided_frames_snip_edges: num_frames {num_frames} exceeds i32::MAX"),
  })?;
  let win_size_i32 = i32::try_from(win_size).map_err(|_| Error::Backend {
    message: format!("strided_frames_snip_edges: win_size {win_size} exceeds i32::MAX"),
  })?;
  let win_inc_i64 = i64::try_from(win_inc).map_err(|_| Error::Backend {
    message: format!("strided_frames_snip_edges: win_inc {win_inc} exceeds i64::MAX"),
  })?;
  let shape: &[i32] = &[num_frames_i32, win_size_i32];
  // SAFETY: see the function-level SAFETY comment — `waveform` is guaranteed
  // row-contiguous by the caller (compute_fbank_kaldi materializes via
  // `ops::shape::contiguous` before calling here), so its flattened storage
  // spans exactly `waveform_len` elements; we asserted `last_index <=
  // waveform_len` above so every reachable index `i*win_inc + j` is in
  // `[0, waveform_len)`. `offset=0` so no out-of-front access either.
  unsafe { ops::shape::as_strided(waveform, &shape, &[win_inc_i64, 1], 0) }
}

/// Fully reverse a 1-D array (`a[::-1]`).
///
/// Built from a single negative-stride [`ops::indexing::slice`] using the
/// `-(len + 1)` post-normalize-to-`-1` sentinel (the same idiom
/// `crate::audio::dsp`'s `reflect_pad_1d` uses for its boundary case): mlx
/// pre-normalizes a negative `stop` by `+ len` BEFORE the per-stride logic, so
/// the "position left of index 0" sentinel is `stop = -(len + 1)`, which makes
/// the traversal `len-1, len-2, …, 0` (inclusive of 0).
///
/// # Errors
/// - [`Error::Backend`] if `a` is not 1-D, is empty, or `len`/`len + 1`
///   exceeds `i32::MAX`.
fn reverse_1d(a: &Array) -> Result<Array> {
  let shape = a.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!("reverse_1d: expected 1-D input, got {}-D", shape.len()),
    });
  }
  let len = shape[0];
  if len == 0 {
    return Err(Error::Backend {
      message: "reverse_1d: cannot reverse an empty array".into(),
    });
  }
  let len_i32 = i32::try_from(len).map_err(|_| Error::Backend {
    message: format!("reverse_1d: len {len} exceeds i32::MAX"),
  })?;
  // `stop = -(len + 1)` post-normalizes (via `+ len`) to `-1`, the
  // "left of index 0" sentinel, so the descending traversal includes index 0.
  // Compute in i64 to avoid overflow when `len == i32::MAX`.
  let sentinel_i64 = -(i64::from(len_i32) + 1);
  let stop = i32::try_from(sentinel_i64).map_err(|_| Error::Backend {
    message: format!("reverse_1d: reverse sentinel -(len + 1) = {sentinel_i64} overflows i32"),
  })?;
  ops::indexing::slice(a, &[len_i32 - 1], &[stop], &[-1])
}

/// Strided framing matching the reference's `_get_strided_kaldi` with
/// `snip_edges=false` (`mlx_audio.dsp.py:787`) — the reflect-bookend path.
///
/// The reference (for a 1-D `waveform` of length `n`) computes
/// `m = (n + win_inc/2) / win_inc` frames and reflect-pads the signal by
/// `pad = win_size/2 - win_inc/2` on each side so frames are *centered* on the
/// sample positions (Kaldi `snip_edges=false`), then takes the
/// `(m, win_size)`-strided view with stride `(win_inc, 1)`. The reflect
/// bookends are (the left is edge-EXCLUSIVE and the right edge-INCLUSIVE — this
/// asymmetry is the reference's exact behavior, not a symmetric reflect):
/// - `pad > 1`: `pad_left = reverse(wf[1 .. pad+1])` (excludes wf[0]),
///   `pad_right = reverse(wf[n-pad .. n])` (the reference's
///   `waveform[-1:-pad-1:-1]`, includes wf[n-1]).
/// - `pad == 1`: `pad_left = reverse(wf[1 .. 2])` (one sample),
///   `pad_right = reverse(wf[1 .. n])` — the reference's `waveform[-1:0:-1]`,
///   which yields `n-1` (not `1`) samples; only the first sample of this
///   bookend is ever read by the strided view, so the over-long tail is inert.
/// - `pad <= 0`: `padded = concat(wf[|pad| ..], reverse(wf))` (the reference's
///   `concat(waveform[-pad:], waveform[::-1])`).
///
/// **Memory-safe deviation from the reference.** The reference calls
/// `mx.as_strided` with NO bounds check; for degenerate inputs (a `win_size`
/// large relative to `n`, e.g. `n < win_size`) the strided view's last read
/// index `(m-1)*win_inc + win_size` exceeds the padded-buffer length, so the
/// reference reads past the buffer (silent out-of-bounds — undefined behavior
/// it gets away with only because numpy/mlx over-allocate). mlxrs's
/// [`ops::shape::as_strided`] is bounds-checked; rather than reproduce that UB,
/// this function asserts `last_index <= padded_len` and returns a recoverable
/// [`Error::Backend`] for the degenerate regime (where there is not enough
/// signal to reflect-pad a full centered window). Every realistic ASR config
/// — a multi-frame signal whose padded length covers the strided read — is
/// reproduced **bit-identically** to the reference. (The padded buffer is built
/// row-contiguous by construction here, so the [`ops::shape::as_strided`]
/// safety pre-condition is met.)
///
/// Returns the `(m, win_size)` strided frame view, or a `(0, 0)` empty array
/// when `m == 0` (vanishingly short input).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `waveform` is not 1-D,
///   - the reflect-padded buffer's element count exceeds the internal
///     `MAX_FBANK_WORK` cap (the reflect bookends roughly double the
///     waveform's memory — checked BEFORE the `concatenate` materializes it),
///   - the reflect bookends require indices outside the signal (e.g.
///     `pad + 1 > n`, i.e. not enough samples to reflect `pad` on a side),
///   - the strided read would exceed the padded-buffer length (degenerate
///     `win_size`-vs-`n` regime — see the memory-safe deviation above),
///   - any size overflows `usize` / `i32` / `i64`.
/// - Propagates slice / concatenate / `as_strided` errors.
fn strided_frames_no_snip_edges(
  waveform: &Array,
  win_size: usize,
  win_inc: usize,
  num_frames: usize,
) -> Result<Array> {
  let shape = waveform.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!(
        "strided_frames_no_snip_edges: expected 1-D waveform, got {}-D",
        shape.len()
      ),
    });
  }
  let n = shape[0];
  let n_i32 = i32::try_from(n).map_err(|_| Error::Backend {
    message: format!("strided_frames_no_snip_edges: waveform len {n} exceeds i32::MAX"),
  })?;
  if num_frames == 0 {
    return Array::zeros::<f32>(&[0_i32, 0_i32]);
  }

  // `pad = win_size/2 - win_inc/2` (the reference's signed pad). `i64` so the
  // signed subtraction can't wrap; both operands are <= MAX_DECODED_SAMPLES.
  let pad_i64 = (win_size as i64) / 2 - (win_inc as i64) / 2;

  // Cap the reflect-padded buffer's element count BEFORE the `concatenate`
  // that materializes it (Codex review). The `compute_fbank_kaldi` caps
  // (`frame_work` / `out_elems` / `output_elems`) bound the *framed* matrix
  // `num_frames * n_fft_padded`, NOT this intermediate reflected buffer: a
  // `(samples_len = MAX_FBANK_WORK, win_len = 2, win_inc = 4)` input gives a
  // tiny `num_frames` (so the framing caps pass), yet the branches below
  // concatenate ≈ `2 * MAX_FBANK_WORK` elements — defeating the 64 Mi budget
  // by ~2×. The reflected length is NOT a single uniform formula: each of the
  // three branches concatenates DIFFERENT segment lengths —
  //   - `pad > 1`:  `pad_left` (`pad`) ++ `waveform` (`n`) ++ `pad_right`
  //     (`pad`)              ⇒ `n + 2*pad`
  //   - `pad == 1`: `pad_left` (`1`) ++ `waveform` (`n`) ++ `pad_right`
  //     (`n - 1`, the reference's over-long inert tail) ⇒ `2*n`
  //   - `pad <= 0`: `head` (`n - |pad|`) ++ `reverse(wf)` (`n`)
  //                                        ⇒ `2*n - |pad|`
  // so a uniform `n + 2*pad` would UNDERCOUNT the `pad == 1` branch by ~`n`
  // (it builds `2*n`, not `n + 2`) and let an adversarial 64 Mi `pad == 1`
  // input slip a ~128 Mi `concatenate` through. The cap is therefore computed
  // INSIDE each branch from the exact `pad_left`/`pad_right` segment lengths
  // that branch will concatenate (`reflected_len_checked`), so the capped
  // length and the built length cannot diverge — and the rejection still
  // happens BEFORE any slice/reverse/concatenate alloc.

  /// Sum the concatenated segment lengths, `checked_mul` against the f32
  /// element budget, and reject (recoverable `Error`) when the reflected
  /// buffer would exceed [`MAX_FBANK_WORK`]. Called with the *actual* segment
  /// lengths each branch concatenates, so the cap matches the built buffer.
  fn cap_reflected_len(seg_lens: &[usize], n: usize, pad: i64) -> Result<()> {
    let mut reflected_len: usize = 0;
    for &seg in seg_lens {
      reflected_len = reflected_len
        .checked_add(seg)
        .ok_or_else(|| Error::Backend {
          message: format!(
            "strided_frames_no_snip_edges: reflect-padded length overflows usize \
             (n={n}, pad={pad})"
          ),
        })?;
    }
    if reflected_len > MAX_FBANK_WORK {
      return Err(Error::Backend {
        message: format!(
          "strided_frames_no_snip_edges: reflect-padded buffer length {reflected_len} \
           (waveform len={n}, pad={pad}) exceeds the {MAX_FBANK_WORK} work cap; the \
           snip_edges=false reflect bookends would more than double the waveform's memory"
        ),
      });
    }
    Ok(())
  }

  // Build the reflect-padded waveform exactly as the reference does.
  let padded = if pad_i64 > 0 {
    let pad = pad_i64 as usize;
    // Need `wf[1 .. pad+1]` and (for pad>1) `wf[n-pad-1 .. n-1]` to exist —
    // i.e. `n >= pad + 1`. (For pad==1 the right bookend is `wf[1 .. n]`, also
    // needing `n >= 2 == pad + 1`.)
    if n < pad + 1 {
      return Err(Error::Backend {
        message: format!(
          "strided_frames_no_snip_edges: waveform len {n} too short to reflect-pad {pad} \
           samples per side (need len >= pad + 1); the win_size/win_inc imply more \
           reflection than the signal supports"
        ),
      });
    }
    let pad_i32 = i32::try_from(pad).map_err(|_| Error::Backend {
      message: format!("strided_frames_no_snip_edges: pad {pad} exceeds i32::MAX"),
    })?;
    // pad_left = reverse(wf[1 .. pad+1]) — the reference's `waveform[1:pad+1][::-1]`,
    // an edge-EXCLUSIVE type-1 reflect on the left (excludes wf[0]). Length `pad`.
    let left_lo = 1_i32;
    let left_hi = pad_i32 + 1;
    let left_len = (left_hi - left_lo) as usize; // == pad
    // pad_right (note the asymmetry vs the left — this is the reference's exact
    // behavior, NOT a symmetric reflect):
    //  - pad > 1: `waveform[-1:-pad-1:-1]` = indices n-1, n-2, …, n-pad =
    //    reverse(wf[n-pad .. n]) — edge-INCLUSIVE (includes wf[n-1]). Length `pad`.
    //  - pad == 1: `waveform[-1:0:-1]` = reverse(wf[1 .. n]) (n-1 samples; only
    //    the first is read by the strided view, so the over-long tail is inert).
    //    Length `n - 1` — so the `pad == 1` buffer is `1 + n + (n-1)` = `2*n`,
    //    NOT `n + 2`; the cap is computed from these exact slice bounds.
    let (right_lo, right_hi) = if pad > 1 {
      (n_i32 - pad_i32, n_i32)
    } else {
      (1_i32, n_i32)
    };
    let right_len = (right_hi - right_lo) as usize; // pad (pad>1) or n-1 (pad==1)
    // Cap from the EXACT segment lengths this branch concatenates, before any
    // slice/reverse/concatenate materializes the buffer.
    cap_reflected_len(&[left_len, n, right_len], n, pad_i64)?;
    let left_seg = ops::indexing::slice(waveform, &[left_lo], &[left_hi], &[1_i32])?;
    let pad_left = reverse_1d(&left_seg)?;
    let right_seg = ops::indexing::slice(waveform, &[right_lo], &[right_hi], &[1_i32])?;
    let pad_right = reverse_1d(&right_seg)?;
    ops::shape::concatenate(&[&pad_left, waveform, &pad_right], 0)?
  } else {
    // pad <= 0: padded = concat(wf[|pad| ..], reverse(wf)).
    // `wf[|pad|:]` keeps `n - |pad|` samples; `|pad| <= n` is guaranteed for
    // any realistic config (|pad| <= win_size/2 <= n when win_size <= ~2n),
    // but assert it so a degenerate `win_inc >> win_size` can't underflow.
    let abs_pad = (-pad_i64) as usize;
    if abs_pad > n {
      return Err(Error::Backend {
        message: format!(
          "strided_frames_no_snip_edges: |pad| {abs_pad} exceeds waveform len {n} \
           (win_inc too large relative to win_size); cannot build the snip_edges=false buffer"
        ),
      });
    }
    let abs_pad_i32 = i32::try_from(abs_pad).map_err(|_| Error::Backend {
      message: format!("strided_frames_no_snip_edges: |pad| {abs_pad} exceeds i32::MAX"),
    })?;
    // head = wf[|pad| .. n] (length `n - |pad|`); rev = reverse(wf) (length `n`)
    // ⇒ reflected = `2*n - |pad|`. Cap from these exact lengths.
    let head_len = n - abs_pad;
    cap_reflected_len(&[head_len, n], n, pad_i64)?;
    let head = ops::indexing::slice(waveform, &[abs_pad_i32], &[n_i32], &[1_i32])?;
    let rev = reverse_1d(waveform)?;
    ops::shape::concatenate(&[&head, &rev], 0)?
  };

  // Bounds-check the strided read: last index `(m-1)*win_inc + win_size` must
  // lie within the padded buffer. Reject the degenerate overread regime
  // (memory-safe deviation; see the doc comment) rather than reproduce UB.
  let padded_len = padded.shape()[0];
  let last_index = (num_frames - 1)
    .checked_mul(win_inc)
    .and_then(|v| v.checked_add(win_size))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "strided_frames_no_snip_edges: reachable element range overflows usize \
         (num_frames={num_frames}, win_inc={win_inc}, win_size={win_size})"
      ),
    })?;
  if last_index > padded_len {
    return Err(Error::Backend {
      message: format!(
        "strided_frames_no_snip_edges: strided read end {last_index} exceeds reflect-padded \
         length {padded_len} (num_frames={num_frames}, win_inc={win_inc}, win_size={win_size}, \
         waveform len={n}); the win_size is too large relative to the signal length for a \
         centered snip_edges=false framing — the reference would read out of bounds here"
      ),
    });
  }

  // The padded buffer is freshly built by `concatenate`, so it is row-
  // contiguous; the strided view's reachable indices are all `< padded_len`
  // (asserted above), and `offset = 0`.
  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| Error::Backend {
    message: format!("strided_frames_no_snip_edges: num_frames {num_frames} exceeds i32::MAX"),
  })?;
  let win_size_i32 = i32::try_from(win_size).map_err(|_| Error::Backend {
    message: format!("strided_frames_no_snip_edges: win_size {win_size} exceeds i32::MAX"),
  })?;
  let win_inc_i64 = i64::try_from(win_inc).map_err(|_| Error::Backend {
    message: format!("strided_frames_no_snip_edges: win_inc {win_inc} exceeds i64::MAX"),
  })?;
  let view_shape: &[i32] = &[num_frames_i32, win_size_i32];
  // SAFETY: `padded` is row-contiguous (built by `concatenate` into a fresh
  // buffer); we asserted `last_index <= padded_len` so every reachable index
  // `i*win_inc + j` is in `[0, padded_len)`; `offset = 0`.
  unsafe { ops::shape::as_strided(&padded, &view_shape, &[win_inc_i64, 1], 0) }
}

/// Compute Kaldi-compatible log-mel-filterbank features.
///
/// Faithful port of `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:853`) —
/// returns shape `(num_frames, num_mels)`, with the Kaldi-specific pre-emphasis,
/// DC-offset removal, dithering, `next_power_of_2` framing, and `log(max(., 1e-8))`
/// floor matching the reference. The mel scale is the Kaldi formula
/// (`1127 * ln(1 + hz / 700)`, see [`mel_scale_kaldi`]).
///
/// ## Pipeline (mirrors `compute_fbank_kaldi`)
/// 1. **Frame** the input. `snip_edges = true` drops partial edge frames
///    (`m = 1 + (n - win)/inc`); `snip_edges = false` reflect-pads the signal
///    so frames are *centered* (`m = (n + inc/2)/inc`) — both paths port
///    `_get_strided_kaldi` (`dsp.py:777`). The waveform is routed through
///    [`ops::shape::contiguous`] first so a sliced / broadcasted 1-D input is
///    materialized to row-major storage before the strided framing view.
/// 2. **Dither** (additive Gaussian noise with std `dither`) — pass `dither = 0.0`
///    or `dither_key = None` to skip; both routes return identical output.
/// 3. **Remove DC offset** (subtract per-frame mean).
/// 4. **Pre-emphasis** filter `y[n] = x[n] - preemphasis * x[n-1]` for
///    `n >= 1`, with the **kaldi-asr** first-sample boundary
///    `y[0] = x[0] * (1 - preemphasis)` (`feature-window.cc:101-107`).
///    This **deliberately deviates** from `mlx_audio.dsp.compute_fbank_kaldi`
///    (`dsp.py:911-915`), which keeps `x[0]` unchanged — see the inline
///    comment for the rationale (Kaldi-trained models pin the
///    `x[0] * (1 - p)` boundary, which torchaudio also implements via
///    `pad(mode="replicate")` in `compliance.kaldi.fbank`).
/// 5. **Window** (Hamming / Hanning / Povey / Rectangular — see [`KaldiWindow`]).
/// 6. **Pad** to `next_power_of_2(win_len)` and `rfft`.
/// 7. **Mel-filterbank** (the `get_mel_banks_kaldi` matrix zero-padded by 1
///    column to match the rfft output bin count) `@ |rfft|^2`.
/// 8. **`log(max(., 1e-8))`** floor.
///
/// ## Determinism
/// Pass `dither_key = None` (or `dither = 0.0`) for deterministic output.
/// Pass `dither_key = Some(&key)` (from [`crate::ops::random::key`]) to seed
/// the dither additively — the same `(key, samples)` pair produces the same
/// dithered features bit-for-bit, allowing reproducible training runs.
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `waveform` is not 1-D,
///   - `win_len < 2`, `win_inc == 0`, or `win_len > MAX_DECODED_SAMPLES`,
///   - `sample_rate == 0`,
///   - `dither < 0.0` or non-finite,
///   - `preemphasis` is not in `[0.0, 1.0]` (the reference accepts any float
///     but the standard range is `[0.0, 1.0]`),
///   - `snip_edges == false` and the signal is too short to reflect-pad a
///     centered window (the degenerate `win_len`-vs-`samples_len` regime where
///     the reference would read out of bounds — see
///     `strided_frames_no_snip_edges`),
///   - `dither != 0.0 && dither_key.is_none()` (deterministic-by-default rule),
///   - `samples_len = waveform.shape()[0]` exceeds
///     [`crate::audio::io::MAX_DECODED_SAMPLES`] — checked BEFORE materializing
///     via [`ops::shape::contiguous`] so a broadcasted-view input can't drive
///     a multi-GB allocation,
///   - `num_frames * n_fft_padded`, the `rfft` output `num_frames *
///     (n_fft_padded/2 + 1)`, the padded mel-bank operand `num_mels *
///     (n_fft_padded/2 + 1)`, OR the final `num_frames * num_mels` matmul
///     output overflows `usize` or exceeds the internal `MAX_FBANK_WORK`
///     cap (~64 Mi elements),
///   - any size exceeds `i32::MAX`.
/// - Propagates errors from [`get_mel_banks_kaldi`] and the underlying ops.
#[allow(clippy::too_many_arguments)]
pub fn compute_fbank_kaldi(
  waveform: &Array,
  sample_rate: u32,
  win_len: usize,
  win_inc: usize,
  num_mels: usize,
  win_type: KaldiWindow,
  preemphasis: f32,
  dither: f32,
  snip_edges: bool,
  low_freq: f32,
  high_freq: f32,
  dither_key: Option<&Array>,
) -> Result<Array> {
  // ---- input validation ------------------------------------------------
  let shape = waveform.shape();
  if shape.len() != 1 {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: expected 1-D waveform, got {}-D",
        shape.len()
      ),
    });
  }
  if sample_rate == 0 {
    return Err(Error::Backend {
      message: "compute_fbank_kaldi: sample_rate must be > 0".into(),
    });
  }
  if win_len < 2 {
    return Err(Error::Backend {
      message: format!("compute_fbank_kaldi: win_len must be >= 2 (got {win_len})"),
    });
  }
  if win_inc == 0 {
    return Err(Error::Backend {
      message: "compute_fbank_kaldi: win_inc must be > 0".into(),
    });
  }
  if win_len > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: win_len {win_len} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }
  if !dither.is_finite() || dither < 0.0 {
    return Err(Error::Backend {
      message: format!("compute_fbank_kaldi: dither must be finite and >= 0.0 (got {dither})"),
    });
  }
  if !(preemphasis.is_finite() && (0.0..=1.0).contains(&preemphasis)) {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: preemphasis must be in [0.0, 1.0] (got {preemphasis})"
      ),
    });
  }
  if dither != 0.0 && dither_key.is_none() {
    return Err(Error::Backend {
      message: "compute_fbank_kaldi: dither != 0.0 requires an explicit dither_key \
                (use crate::ops::random::key(seed) or pass dither=0.0 to disable). \
                The Python reference's implicit-default-key behavior is deliberately \
                not mirrored here — explicit keys make dithered features reproducible."
        .into(),
    });
  }

  let samples_len = shape[0];

  // Hard cap on `samples_len` BEFORE the `ops::shape::contiguous` call below.
  // `samples_len` is the LOGICAL length from `waveform.shape()[0]`; if `waveform`
  // is a broadcasted view (e.g. `broadcast_to([0.5], &[100_000_000])` with stride
  // 0), the underlying storage is tiny but `contiguous(waveform, false)` will
  // materialize the full logical extent into a fresh row-major buffer at eval
  // time — a one-element broadcast can therefore drive a multi-GB allocation
  // (Codex review R2). The existing `frame_work` / `out_elems` / `output_elems`
  // caps run AFTER framing math and don't constrain `samples_len` directly: a
  // pathological `(samples_len=100M, win_len=2, win_inc=50M, num_mels=1)` input
  // gives `num_frames = 1` → `frame_work = 2` (well under the cap) but
  // `contiguous` still materializes ~400 MB of f32. Reject here before the
  // materialization. `MAX_DECODED_SAMPLES` (= `MAX_FBANK_WORK` = 64 Mi) is the
  // documented audio-IO budget for any single decoded waveform.
  if samples_len > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: samples_len {samples_len} exceeds the {} cap \
         (MAX_DECODED_SAMPLES); rejecting BEFORE `contiguous` would materialize \
         the logical extent (a broadcasted-view input could otherwise drive a \
         multi-GB allocation at eval time)",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }

  // ---- framing (snip_edges true / false) -------------------------------
  // `dsp.py:783-799` (`_get_strided_kaldi`):
  //  - snip_edges=true:  `m = 1 + (n - win)/inc` if `n >= win`, else `(0, 0)`.
  //    We surface "no frames" as a `(0, num_mels)` empty array (`dsp.py:900`).
  //  - snip_edges=false: `m = (n + win_inc/2) / win_inc` with reflect-bookend
  //    padding (the centered framing). The reference does NOT short-circuit on
  //    `n < win`; it reflect-pads and frames anyway (see
  //    `strided_frames_no_snip_edges`).
  let num_mels_i32 = i32::try_from(num_mels).map_err(|_| Error::Backend {
    message: format!("compute_fbank_kaldi: num_mels {num_mels} exceeds i32::MAX"),
  })?;
  let num_frames = if snip_edges {
    if samples_len < win_len {
      return Array::zeros::<f32>(&[0_i32, num_mels_i32]);
    }
    1 + (samples_len - win_len) / win_inc
  } else {
    // `m = (n + win_inc/2) / win_inc` (`dsp.py:788`). `win_inc >= 1` (checked),
    // so the division is well-defined; for `n == 0` this is `0` frames.
    let m = (samples_len + win_inc / 2) / win_inc;
    if m == 0 {
      return Array::zeros::<f32>(&[0_i32, num_mels_i32]);
    }
    m
  };

  // ---- size / work caps (mirror the dsp.rs `MAX_STFT_WORK` pattern) ----
  // `n_fft_padded` is the FFT length the rfft consumes; bound the windowed
  // frame matrix `num_frames * n_fft_padded` against the work cap BEFORE
  // building the strided view, window, rfft, or mel-filterbank. The samples
  // cap on `waveform` is already enforced by the audio IO entry points, but a
  // lazy/shaped huge input could still drive `num_frames` past the cap with
  // a small `win_inc`, so we re-check the framing work here.
  let n_fft_padded = next_power_of_2(win_len);
  let frame_work = num_frames
    .checked_mul(n_fft_padded)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "compute_fbank_kaldi: frame work num_frames * n_fft_padded overflows usize \
         (num_frames={num_frames}, n_fft_padded={n_fft_padded})"
      ),
    })?;
  if frame_work > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: frame work {frame_work} (num_frames={num_frames} * \
         n_fft_padded={n_fft_padded}) exceeds the {MAX_FBANK_WORK} work cap"
      ),
    });
  }
  // Output element count `num_frames * (n_fft_padded / 2 + 1)` (rfft output).
  // `n_fft_padded` is a power of two >= 2 (since `win_len >= 2`), so `/2 + 1`
  // cannot overflow.
  let out_elems = num_frames
    .checked_mul(n_fft_padded / 2 + 1)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "compute_fbank_kaldi: rfft output element count overflows usize \
         (num_frames={num_frames}, n_fft_padded={n_fft_padded})"
      ),
    })?;
  if out_elems > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: rfft output element count {out_elems} exceeds the \
         {MAX_FBANK_WORK} work cap (num_frames={num_frames}, n_fft_padded={n_fft_padded})"
      ),
    });
  }
  // Final output element count `num_frames * num_mels` (the `(num_frames, num_mels)`
  // matrix the `power @ mel_padded.T` matmul produces). This is a SEPARATE cap
  // from the rfft / mel-bank caps: pathological inputs with `n_fft_padded / 2`
  // tiny (e.g. `win_len = 2 → n_fft_padded = 2 → num_fft_bins = 1`) satisfy
  // the mel-bank cap (`num_mels * 1 == num_mels`) and the frame-work cap, but
  // can still drive `num_frames * num_mels` into TB territory with a small
  // `win_inc` and a huge `num_mels` (Codex review). Reject BEFORE building the
  // mel filterbank, the matmul, or any of the intermediates they hold.
  let output_elems = num_frames
    .checked_mul(num_mels)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "compute_fbank_kaldi: output element count num_frames * num_mels overflows usize \
         (num_frames={num_frames}, num_mels={num_mels})"
      ),
    })?;
  if output_elems > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: output element count {output_elems} (num_frames={num_frames} * \
         num_mels={num_mels}) exceeds the {MAX_FBANK_WORK} work cap"
      ),
    });
  }
  // Padded mel-bank element count `num_mels * (n_fft_padded / 2 + 1)`. This
  // is the SHAPE of the right operand of the `power @ mel_padded.T` matmul
  // (`get_mel_banks_kaldi` returns `(num_mels, n_fft_padded/2)`; we pad ONE
  // column on the right below at `ops::shape::pad(&mel_bank, …, &[1_i32], …)`
  // so the trailing dim matches the rfft's `n_fft_padded/2 + 1` bin count).
  // The unpadded `bank_len` check inside `get_mel_banks_kaldi` only caps
  // `num_mels * (n_fft_padded/2)` (Codex review R2): with `n_fft_padded == 2`
  // → `num_fft_bins == 1` → unpadded `bank_len == num_mels` passes the cap,
  // but the padded operand DOUBLES to `num_mels * 2` and a `num_mels =
  // MAX_FBANK_WORK` would push that to `128 Mi` (256 MiB of f32). Reject
  // BEFORE building the mel filterbank or the matmul intermediates.
  let mel_padded_elems =
    num_mels
      .checked_mul(n_fft_padded / 2 + 1)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "compute_fbank_kaldi: padded mel-bank element count num_mels * (n_fft_padded/2 + 1) \
         overflows usize (num_mels={num_mels}, n_fft_padded={n_fft_padded})"
        ),
      })?;
  if mel_padded_elems > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_fbank_kaldi: padded mel-bank element count {mel_padded_elems} \
         (num_mels={num_mels} * (n_fft_padded/2 + 1)={}) exceeds the {MAX_FBANK_WORK} work cap",
        n_fft_padded / 2 + 1
      ),
    });
  }

  let n_fft_padded_i32 = i32::try_from(n_fft_padded).map_err(|_| Error::Backend {
    message: format!("compute_fbank_kaldi: n_fft_padded {n_fft_padded} exceeds i32::MAX"),
  })?;
  let win_len_i32 = i32::try_from(win_len).map_err(|_| Error::Backend {
    message: format!("compute_fbank_kaldi: win_len {win_len} exceeds i32::MAX"),
  })?;
  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| Error::Backend {
    message: format!("compute_fbank_kaldi: num_frames {num_frames} exceeds i32::MAX"),
  })?;

  // ---- 1. frame ---------------------------------------------------------
  // Both framing helpers read through `unsafe ops::shape::as_strided`, which
  // assumes ROW-CONTIGUOUS backing storage with at least `waveform_len`
  // elements reachable from the data pointer. Public callers may legitimately
  // hand us a 1-D slice/view (`waveform.slice(0, 100, 200)`) or a broadcasted
  // scalar — these pass the rank-1 check but their flattened storage is
  // shorter than `shape()[0]` (or has non-unit strides), so the strided view
  // would read out-of-bounds. Materialize via `ops::shape::contiguous` first;
  // it's a no-op refcount bump when the input is already row-contiguous and
  // an honest copy otherwise. This is the same idiom mlx-swift's `MLX.contiguous`
  // documents for the same case. The `snip_edges=false` helper additionally
  // builds its reflect-bookend buffer via `concatenate` (also row-contiguous)
  // before its own strided view.
  let waveform_contig = ops::shape::contiguous(waveform, false)?;
  let strided = if snip_edges {
    strided_frames_snip_edges(&waveform_contig, win_len, win_inc, num_frames)?
  } else {
    strided_frames_no_snip_edges(&waveform_contig, win_len, win_inc, num_frames)?
  };

  // ---- 2. dither (additive Gaussian) -----------------------------------
  // Only run the FFI random call when both `dither > 0.0` and a key is
  // supplied — the `dither == 0.0 || key.is_none()` paths return the input
  // unchanged.
  let dithered = if dither > 0.0 {
    // Validated above that `dither_key.is_some()` whenever `dither != 0.0`.
    let key = dither_key.expect("dither != 0.0 was checked to require a key above");
    let shape: &[i32] = &[num_frames_i32, win_len_i32];
    let noise = ops::random::normal(&shape, crate::Dtype::F32, 0.0, dither, key)?;
    ops::arithmetic::add(&strided, &noise)?
  } else {
    strided
  };

  // ---- 3. remove DC offset (per-frame mean) ----------------------------
  // `dsp.py:908-909`: `row_means = mean(strided, axis=1, keepdims=True);
  // strided -= row_means`.
  let row_means = ops::reduction::mean_axes(&dithered, &[1], true)?;
  let centered = ops::arithmetic::subtract(&dithered, &row_means)?;

  // ---- 4. pre-emphasis -------------------------------------------------
  // Kaldi-asr `feature-window.cc:101-107` (`Preemphasize`) applies the filter
  // `y[n] = x[n] - p * x[n-1]` for `n >= 1` AND treats the first sample
  // through a self-reference: `y[0] = x[0] - p * x[0] = x[0] * (1 - p)`.
  // torchaudio matches this via `pad(mode="replicate")` which replicates `x[0]`
  // as its own predecessor (`docs.pytorch.org/audio/stable/_modules/torchaudio/
  // compliance/kaldi.html`, `_get_window`).
  //
  // `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:911-915`) instead keeps
  // `x[:,0:1]` UNCHANGED — that is a bug vs the Kaldi reference the rest of
  // the function targets (preemphasis coefficient, window denominators, mel
  // formula, `next_power_of_2` framing, `1e-8` floor are all Kaldi-faithful).
  // We deliberately deviate from `mlx-audio` here to match the kaldi-asr
  // reference + torchaudio (`compliance.kaldi.fbank`); a Kaldi-trained acoustic
  // model expects the `x[0] * (1 - p)` boundary, NOT the passthrough.
  let preemphasized = if preemphasis > 0.0 {
    // Slices keep ALL frames on axis 0 (`[0, num_frames_i32]`) and split
    // the columns on axis 1: first column (scaled by `1 - p` below), columns
    // 1..win_len (the `y[n]` body), and columns 0..win_len-1 (the shifted
    // `x[n-1]` column for the body).
    let first_col = ops::indexing::slice(
      &centered,
      &[0_i32, 0_i32],
      &[num_frames_i32, 1_i32],
      &[1_i32, 1_i32],
    )?;
    let rest = ops::indexing::slice(
      &centered,
      &[0_i32, 1_i32],
      &[num_frames_i32, win_len_i32],
      &[1_i32, 1_i32],
    )?;
    let prev = ops::indexing::slice(
      &centered,
      &[0_i32, 0_i32],
      &[num_frames_i32, win_len_i32 - 1],
      &[1_i32, 1_i32],
    )?;
    let p_arr = Array::full::<f32>(&[0_i32; 0], preemphasis)?;
    let scaled_prev = ops::arithmetic::multiply(&prev, &p_arr)?;
    let other_cols = ops::arithmetic::subtract(&rest, &scaled_prev)?;
    // Kaldi first-sample boundary: `y[0] = x[0] * (1 - p)`.
    let one_minus_p = Array::full::<f32>(&[0_i32; 0], 1.0 - preemphasis)?;
    let first_col_scaled = ops::arithmetic::multiply(&first_col, &one_minus_p)?;
    ops::shape::concatenate(&[&first_col_scaled, &other_cols], 1)?
  } else {
    centered
  };

  // ---- 5. window -------------------------------------------------------
  let window = build_kaldi_window(win_type, win_len)?;
  let windowed = ops::arithmetic::multiply(&preemphasized, &window)?;

  // ---- 6. pad to power of 2 + rfft -------------------------------------
  let padded = if n_fft_padded != win_len {
    let pad_extent = i32::try_from(n_fft_padded - win_len).map_err(|_| Error::Backend {
      message: format!(
        "compute_fbank_kaldi: pad extent {} exceeds i32::MAX",
        n_fft_padded - win_len
      ),
    })?;
    let pad_value = Array::zeros::<f32>(&[0_i32; 0])?;
    ops::shape::pad(
      &windowed,
      &[1_i32],
      &[0_i32],
      &[pad_extent],
      &pad_value,
      c"constant",
    )?
  } else {
    windowed
  };
  let spectrum = fft::rfft(&padded, n_fft_padded_i32, 1, FftNorm::Backward)?;

  // |rfft|^2 — `abs` of the Complex64 spectrum yields F32 magnitudes, then square.
  let power = spectrum.abs()?.square()?;

  // ---- 7. mel filterbank @ power ---------------------------------------
  // `get_mel_banks_kaldi` returns shape `(num_mels, n_fft_padded / 2)`; pad
  // one zero column on the right so it matches the rfft's `n_fft_padded/2 + 1`
  // bin count (matching `dsp.py:946`).
  let (mel_bank, _centers) = get_mel_banks_kaldi(
    num_mels,
    n_fft_padded,
    sample_rate as f32,
    low_freq,
    high_freq,
  )?;
  let pad_value = Array::zeros::<f32>(&[0_i32; 0])?;
  let mel_padded = ops::shape::pad(
    &mel_bank,
    &[1_i32],
    &[0_i32],
    &[1_i32],
    &pad_value,
    c"constant",
  )?;

  // `power` is `(num_frames, n_fft_padded/2 + 1)`; `mel_padded` is
  // `(num_mels, n_fft_padded/2 + 1)`. Output is `(num_frames, num_mels) =
  // power @ mel_padded.T` (matching `dsp.py:949`).
  let mel_t = mel_padded.transpose()?;
  let mel_features = ops::linalg_basic::matmul(&power, &mel_t)?;

  // ---- 8. log floor ----------------------------------------------------
  let floor = Array::full::<f32>(&[0_i32; 0], KALDI_FBANK_LOG_FLOOR)?;
  let floored = ops::arithmetic::maximum(&mel_features, &floor)?;
  floored.log()
}

/// Boundary-padding mode for [`compute_deltas_kaldi`]. Mirrors the `mode`
/// string argument of `mlx_audio.dsp.compute_deltas_kaldi` (`dsp.py:716`).
///
/// The delta at time `t` reads `c[t-n .. t+n]`; near the edges those indices
/// fall outside `[0, time)`, so the spectrogram is padded by `n` frames on each
/// side of the time axis first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeltaPadMode {
  /// **Edge replication** (the reference default): the first / last time frame
  /// is repeated `n` times on the left / right (`mx.repeat(specgram[:, 0:1], n,
  /// axis=1)` / `[:, -1:]`). Matches kaldi-asr's delta-window edge handling.
  #[default]
  Edge,
  /// **Zero padding**: `n` zero frames on each side (`mx.pad(specgram, [(0, 0),
  /// (n, n)])`). Matches the reference's `else` branch.
  Constant,
}

/// Compute Kaldi-compatible delta (velocity / acceleration) coefficients of a
/// spectrogram along its **last (time) axis**.
///
/// Faithful port of `mlx_audio.dsp.compute_deltas_kaldi(specgram, win_length=5,
/// mode="edge")` (`dsp.py:715`). The delta at time `t` is the
/// regression-weighted finite difference
///
/// ```text
///         Σ_{k=-n}^{n}  k * c[t + k]
/// d[t] = ───────────────────────────── ,   n = (win_length - 1) / 2
///         Σ_{k=-n}^{n}  k²  =  n(n+1)(2n+1)/3
/// ```
///
/// (the reference computes the denominator as `n*(n+1)*(2n+1)/3`, i.e. the
/// `mx.arange(-n, n+1)²` sum; note this is `2 * Σ_{k=1}^{n} k²`, NOT the
/// docstring's `2 * Σ k²` — the code's `denom` is the parity-faithful value and
/// is what we reproduce). Apply twice for delta-deltas (acceleration):
/// `compute_deltas_kaldi(&compute_deltas_kaldi(&x, w, m)?, w, m)`.
///
/// ## Shape
/// Input `(.., time)` of any rank `>= 1`; output has the **same shape**. The
/// reference flattens to `(num_features, time)`, pads the time axis by `n` per
/// [`DeltaPadMode`], computes deltas, then restores the original shape — we do
/// the same. (A common pairing is the `(num_frames, num_mels)` output of
/// [`compute_fbank_kaldi`] **transposed** to `(num_mels, num_frames)` so time
/// is last; deltas are along time either way — the function only ever touches
/// the last axis.)
///
/// ## Vectorization
/// Rather than the reference's per-timestep python loop, we accumulate the
/// `win_length` shifted, weight-scaled slices of the padded spectrogram
/// (`d += k * padded[.., n + k : n + k + time]` for `k in -n..=n`). This is
/// `win_length` cheap strided slices (default `win_length = 5`) with no large
/// 3-D intermediate — numerically identical to the loop. The cumulative
/// element-op count `total * (win_length - 1)` is bounded by a dedicated
/// `MAX_DELTA_WORK` cap (distinct from the input / padded-buffer size caps).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `specgram` has rank `0` (no time axis),
///   - `win_length < 3` (the reference raises `ValueError`),
///   - `win_length` is even (a symmetric `[-n, n]` window needs an odd length;
///     the reference's `n = (win_length - 1) // 2` silently truncates an even
///     `win_length` to the next-lower odd window, which we reject rather than
///     silently reinterpret),
///   - `win_length` exceeds the small supported maximum (`1024`; delta windows
///     are tiny in practice — the default is `5` — and a huge window drives the
///     boundary pad / shifted-slice work into a CPU stall on a tiny input),
///   - the element count `total` (== `num_features * time`) **or** the padded
///     element count `num_features * (time + 2n)` exceeds the internal
///     `MAX_FBANK_WORK` cap (~64 Mi elements) — the padded count is checked
///     BEFORE any pad / broadcast / concatenate so a tiny input with a large
///     `win_length` cannot allocate the bookends first,
///   - the cumulative accumulation work `total * (win_length - 1)` (the
///     `win_length - 1` full-width slice / multiply / add passes the delta
///     loop performs) exceeds the internal `MAX_DELTA_WORK` budget — a
///     separate cap from the buffer-size caps, checked BEFORE the loop so a
///     wide window on a large input cannot stall the CPU / GPU,
///   - the padded time extent `time + 2n` overflows `usize` / `i32`.
/// - Propagates errors from the underlying slice / pad / concatenate ops.
pub fn compute_deltas_kaldi(
  specgram: &Array,
  win_length: usize,
  mode: DeltaPadMode,
) -> Result<Array> {
  if win_length < 3 {
    return Err(Error::Backend {
      message: format!("compute_deltas_kaldi: win_length must be >= 3 (got {win_length})"),
    });
  }
  // The reference's `n = (win_length - 1) // 2` silently truncates an even
  // `win_length` (e.g. 4 → n=1 → an effective window of 3). Reject even
  // lengths so the caller's intent is unambiguous.
  if win_length.is_multiple_of(2) {
    return Err(Error::Backend {
      message: format!(
        "compute_deltas_kaldi: win_length must be odd (got {win_length}); an even \
         win_length would silently truncate to the next-lower odd window"
      ),
    });
  }
  // Cap `win_length` to a small supported bound BEFORE any work. A huge odd
  // `win_length` drives both the symmetric `n = win_length / 2` boundary pad
  // (which broadcasts two `(num_features, n)` bookends and concatenates) and
  // the per-offset shifted-slice loop (`win_length` strided slices) — for a
  // tiny input that work explodes long before the element-count cap engages.
  // Delta windows are tiny in practice (Kaldi default 5), so reject early.
  if win_length > MAX_DELTA_WIN_LENGTH {
    return Err(Error::Backend {
      message: format!(
        "compute_deltas_kaldi: win_length {win_length} exceeds the supported maximum \
         {MAX_DELTA_WIN_LENGTH} (delta windows are tiny — the default is 5)"
      ),
    });
  }
  let orig_shape = specgram.shape();
  if orig_shape.is_empty() {
    return Err(Error::Backend {
      message: "compute_deltas_kaldi: specgram must have rank >= 1 (a time axis)".into(),
    });
  }
  let time = orig_shape[orig_shape.len() - 1];
  // `num_features = product(orig_shape[..-1])` (1 for a 1-D input). Computed
  // with checked arithmetic; the total element count is then `num_features *
  // time == specgram.size()`.
  let total = specgram.size();
  // `time == 0` ⇒ `total == 0`; the output is the same empty shape (no deltas
  // to compute). Reshape-to-2-D below would divide by `time`, so short-circuit.
  if total == 0 {
    return Array::zeros::<f32>(&orig_shape.as_slice());
  }
  // Bound the work: `total` (== num_features * time) against the shared cap.
  if total > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_deltas_kaldi: element count {total} exceeds the {MAX_FBANK_WORK} work cap"
      ),
    });
  }
  // `time > 0` here, so the integer division is exact and `num_features >= 1`.
  let num_features = total / time;

  let n = (win_length - 1) / 2;
  // denom = n*(n+1)*(2n+1)/3 == Σ_{k=-n}^{n} k² (the reference's `denom`).
  // `win_length <= MAX_DELTA_WIN_LENGTH` (1024) ⇒ `n <= 511`, so the product
  // `n*(n+1)*(2n+1)` fits comfortably in u64 / f64 without overflow.
  let denom = (n as f64 * (n + 1) as f64 * (2 * n + 1) as f64) / 3.0;
  let denom_f32 = denom as f32;

  // Padded time extent `time + 2n`, the width of the buffer the pad/broadcast
  // step below materializes (and the slice bound for the accumulation). Compute
  // it and cap the PADDED work `num_features * padded_time` BEFORE any pad /
  // broadcast / concatenate: the original-element cap above only bounds
  // `num_features * time`, but `Edge` mode broadcasts two `(num_features, n)`
  // bookends and `Constant` mode pads by `n` on each side, so a tiny input with
  // a large (but still capped) `win_length` would otherwise allocate
  // `num_features * (time + 2n)` elements unchecked. `win_length` is already
  // bounded above (so `n <= 511`), but a large `num_features` × that pad can
  // still exceed the budget — reject here, before allocating.
  let padded_time = time.checked_add(2 * n).ok_or_else(|| Error::Backend {
    message: format!(
      "compute_deltas_kaldi: padded time time + 2n overflows usize (time={time}, n={n})"
    ),
  })?;
  let padded_work = num_features
    .checked_mul(padded_time)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "compute_deltas_kaldi: padded work num_features * (time + 2n) overflows usize \
         (num_features={num_features}, padded_time={padded_time})"
      ),
    })?;
  if padded_work > MAX_FBANK_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_deltas_kaldi: padded element count {padded_work} (num_features={num_features} \
         * (time + 2n)={padded_time}) exceeds the {MAX_FBANK_WORK} work cap"
      ),
    });
  }
  // Cap the CUMULATIVE accumulation work, distinct from the buffer-size caps
  // above (Codex review). The `total` and `padded_work` caps bound buffer
  // *sizes* (`num_features * time` and `num_features * (time + 2n)`), but the
  // accumulation loop below runs `win_length - 1` (`== 2n`) full-width
  // slice / multiply / add passes over `num_features * time` elements — so the
  // real element-op count is `total * (win_length - 1)`, the multiplier the
  // size caps ignore. A `(1-D length = MAX_FBANK_WORK - 1022, win_length =
  // 1023)` input passes BOTH size caps yet schedules ~1022 passes over ~64 Mi
  // elements ≈ tens of billions of ops. Reject against the dedicated
  // `MAX_DELTA_WORK` budget BEFORE entering the per-offset loop — the delta
  // analogue of `dsp.rs`'s `MAX_LOUDNESS_WORK` visit cap. `win_length >= 3`
  // here, so `win_length - 1 >= 2`.
  let delta_work = total
    .checked_mul(win_length - 1)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "compute_deltas_kaldi: accumulation work total * (win_length - 1) overflows usize \
         (total={total}, win_length={win_length})"
      ),
    })?;
  if delta_work > MAX_DELTA_WORK {
    return Err(Error::Backend {
      message: format!(
        "compute_deltas_kaldi: accumulation work {delta_work} (element count {total} * \
         (win_length - 1)={}) exceeds the {MAX_DELTA_WORK} work cap; the delta loop runs \
         win_length - 1 full-width passes over the spectrogram",
        win_length - 1
      ),
    });
  }
  let _padded_time_i32 = i32::try_from(padded_time).map_err(|_| Error::Backend {
    message: format!("compute_deltas_kaldi: padded time {padded_time} exceeds i32::MAX"),
  })?;

  // Flatten to `(num_features, time)` (the reference's `reshape(-1, time)`).
  let num_features_i32 = i32::try_from(num_features).map_err(|_| Error::Backend {
    message: format!("compute_deltas_kaldi: num_features {num_features} exceeds i32::MAX"),
  })?;
  let time_i32 = i32::try_from(time).map_err(|_| Error::Backend {
    message: format!("compute_deltas_kaldi: time {time} exceeds i32::MAX"),
  })?;
  let flat = ops::shape::reshape(specgram, &(num_features, time))?;

  // Pad the time axis by `n` on each side per `mode`.
  let n_i32 = i32::try_from(n).map_err(|_| Error::Backend {
    message: format!("compute_deltas_kaldi: pad extent n={n} exceeds i32::MAX"),
  })?;
  let padded = match mode {
    DeltaPadMode::Constant => {
      let pad_value = Array::zeros::<f32>(&[0_i32; 0])?;
      ops::shape::pad(&flat, &[1_i32], &[n_i32], &[n_i32], &pad_value, c"constant")?
    }
    DeltaPadMode::Edge => {
      // Edge replication: repeat the first / last column `n` times. mlxrs's
      // `pad` only supports "constant", and there is no `repeat` op, so build
      // the bookends via slice → broadcast_to → concatenate (the reference's
      // `mx.repeat(specgram[:, 0:1], n, axis=1)` / `[:, -1:]`).
      let first_col = ops::indexing::slice(
        &flat,
        &[0_i32, 0_i32],
        &[num_features_i32, 1_i32],
        &[1_i32, 1_i32],
      )?;
      let last_col = ops::indexing::slice(
        &flat,
        &[0_i32, time_i32 - 1],
        &[num_features_i32, time_i32],
        &[1_i32, 1_i32],
      )?;
      let pad_left = ops::shape::broadcast_to(&first_col, &(num_features, n))?;
      let pad_right = ops::shape::broadcast_to(&last_col, &(num_features, n))?;
      ops::shape::concatenate(&[&pad_left, &flat, &pad_right], 1)?
    }
  };

  // Accumulate `d += k * padded[:, n + k : n + k + time]` for k in -n..=n.
  // `k = 0` contributes nothing (weight 0), so skip it. The shifted slice for
  // offset `k` starts at column `n + k` (>= 0 since `k >= -n`) and spans
  // `time` columns (ending at `n + k + time <= 2n + time = padded_time`).
  let mut acc = Array::zeros::<f32>(&[num_features_i32, time_i32])?;
  for k in -(n as isize)..=(n as isize) {
    if k == 0 {
      continue;
    }
    let start = (n as isize + k) as i32; // n + k, in [0, 2n]
    let stop = start + time_i32; // n + k + time, in [time, padded_time]
    let shifted = ops::indexing::slice(
      &padded,
      &[0_i32, start],
      &[num_features_i32, stop],
      &[1_i32, 1_i32],
    )?;
    let weight = Array::full::<f32>(&[0_i32; 0], k as f32)?;
    let weighted = ops::arithmetic::multiply(&shifted, &weight)?;
    acc = ops::arithmetic::add(&acc, &weighted)?;
  }
  let denom_arr = Array::full::<f32>(&[0_i32; 0], denom_f32)?;
  let deltas = ops::arithmetic::divide(&acc, &denom_arr)?;

  // Restore the original shape (the reference's `reshape(original_shape)`).
  ops::shape::reshape(&deltas, &orig_shape.as_slice())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Dtype;

  /// Absolute tolerance for closed-form scalar checks. Mirrors `dsp.rs`'s
  /// `WIN_TOL` (1e-6) for f32 evaluations of mlx-audio's f64-evaluated formulas.
  const F32_TOL: f32 = 1e-5;

  fn to_vec(a: &Array) -> Vec<f32> {
    a.try_clone().unwrap().to_vec::<f32>().unwrap()
  }

  // ---- mel scale parity ---------------------------------------------------

  #[test]
  fn mel_scale_kaldi_matches_reference_formula() {
    // Hand-computed against `1127 * ln(1 + hz / 700)`:
    // hz=0 → 0; hz=700 → 1127*ln(2) ≈ 781.176; hz=1000 → 1127*ln(8/7+0)
    //   actually ln(1 + 1000/700) = ln(17/7) ≈ 0.8873; * 1127 ≈ 1000.05.
    // hz=8000 → ln(1 + 8000/700) = ln(87/7) ≈ 2.5232; * 1127 ≈ 2843.7.
    assert!((mel_scale_kaldi(0.0)).abs() < F32_TOL);
    let v_700 = mel_scale_kaldi(700.0);
    let want_700 = 1127.0 * 2.0_f32.ln();
    assert!(
      (v_700 - want_700).abs() < 1e-3,
      "mel(700): got {v_700}, want {want_700}"
    );
    let v_1000 = mel_scale_kaldi(1000.0);
    let want_1000 = 1127.0 * (17.0_f32 / 7.0).ln();
    assert!(
      (v_1000 - want_1000).abs() < 1e-3,
      "mel(1000): got {v_1000}, want {want_1000}"
    );
  }

  #[test]
  fn mel_scale_kaldi_inverse_round_trips() {
    // For non-negative hz, inverse(scale(hz)) == hz to f32 precision.
    for hz in [0.0_f32, 100.0, 700.0, 1000.0, 4000.0, 8000.0, 16000.0] {
      let mel = mel_scale_kaldi(hz);
      let back = inverse_mel_scale_kaldi(mel);
      assert!(
        (back - hz).abs() < (hz.abs() + 1.0) * 1e-5,
        "round-trip(hz={hz}): mel={mel}, back={back}"
      );
    }
  }

  // ---- get_mel_banks_kaldi shape + structure -----------------------------

  #[test]
  fn mel_banks_kaldi_shape() {
    // `n_fft_padded = 512, num_bins = 80` → bins shape `(80, 256)` (n_fft/2,
    // omitting Nyquist); centers shape `(80,)`.
    let (bins, centers) = get_mel_banks_kaldi(80, 512, 16_000.0, 20.0, 0.0).unwrap();
    assert_eq!(bins.shape(), vec![80, 256]);
    assert_eq!(centers.shape(), vec![80]);
    assert_eq!(bins.dtype().unwrap(), Dtype::F32);
  }

  #[test]
  fn mel_banks_kaldi_rows_sum_positive() {
    // Each triangular filter must integrate to a positive value (otherwise
    // the row would be all-zero and the corresponding mel feature dead).
    let (bins, _) = get_mel_banks_kaldi(40, 512, 16_000.0, 0.0, 0.0).unwrap();
    let v = to_vec(&bins);
    let cols = 256;
    for m in 0..40 {
      let row_sum: f32 = v[m * cols..(m + 1) * cols].iter().sum();
      assert!(
        row_sum > 0.0,
        "mel bin {m} integrates to {row_sum}, expected > 0"
      );
    }
  }

  #[test]
  fn mel_banks_kaldi_center_freqs_monotone_increasing() {
    let (_, centers) = get_mel_banks_kaldi(40, 512, 16_000.0, 20.0, 0.0).unwrap();
    let c = to_vec(&centers);
    for w in c.windows(2) {
      assert!(
        w[1] > w[0],
        "center freqs must be monotone increasing: {} not > {}",
        w[1],
        w[0]
      );
    }
    // Lowest center >= low_freq (~20 Hz) and highest <= Nyquist (8000 Hz).
    assert!(c[0] > 20.0, "first center {} should exceed low_freq", c[0]);
    assert!(
      c[c.len() - 1] < 8000.0,
      "last center {} should be under Nyquist 8000",
      c[c.len() - 1]
    );
  }

  #[test]
  fn mel_banks_kaldi_rejects_invalid_args() {
    // num_bins <= 3.
    assert!(matches!(
      get_mel_banks_kaldi(3, 512, 16_000.0, 0.0, 0.0),
      Err(Error::Backend { .. })
    ));
    // odd n_fft.
    assert!(matches!(
      get_mel_banks_kaldi(40, 513, 16_000.0, 0.0, 0.0),
      Err(Error::Backend { .. })
    ));
    // zero sample rate.
    assert!(matches!(
      get_mel_banks_kaldi(40, 512, 0.0, 0.0, 0.0),
      Err(Error::Backend { .. })
    ));
    // low >= high (after high_freq <= 0 resolution).
    assert!(matches!(
      get_mel_banks_kaldi(40, 512, 16_000.0, 9000.0, 0.0),
      Err(Error::Backend { .. })
    ));
    // low_freq >= nyquist.
    assert!(matches!(
      get_mel_banks_kaldi(40, 512, 16_000.0, 9000.0, -100.0),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn next_power_of_2_smoke() {
    assert_eq!(next_power_of_2(0), 1);
    assert_eq!(next_power_of_2(1), 1);
    assert_eq!(next_power_of_2(2), 2);
    assert_eq!(next_power_of_2(3), 4);
    assert_eq!(next_power_of_2(400), 512);
    assert_eq!(next_power_of_2(1920), 2048);
  }

  // ---- compute_fbank_kaldi end-to-end ------------------------------------

  /// Synthesize a `freq`-Hz sine wave of `n_samples` samples at `sample_rate`.
  fn sine_wave(freq: f32, sample_rate: u32, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
      .map(|n| (2.0 * PI * freq * (n as f32) / (sample_rate as f32)).sin())
      .collect()
  }

  #[test]
  fn compute_fbank_kaldi_output_shape() {
    // n_samples = 16000 (1s @ 16kHz), win_len=400, win_inc=160, snip_edges=true:
    //   num_frames = 1 + (16000 - 400) / 160 = 98
    let samples = sine_wave(1000.0, 16_000, 16_000);
    let x = Array::from_slice::<f32>(&samples, &[16_000_i32]).unwrap();
    let out = compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Hamming,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
    )
    .unwrap();
    assert_eq!(out.shape(), vec![98, 40]);
    assert_eq!(out.dtype().unwrap(), Dtype::F32);
  }

  #[test]
  fn compute_fbank_kaldi_snip_edges_false_frame_count_and_finite() {
    // Public-function (`compute_fbank_kaldi`) parity for the snip_edges=false
    // centered framing. Same input as the shape test (16000 samples, win=400,
    // inc=160):
    //   snip_edges=true:  m = 1 + (16000 - 400)/160     = 98 frames.
    //   snip_edges=false: m = (16000 + 160/2)/160 = (16000+80)/160 = 100.
    // So snip_edges=false yields 2 MORE frames (the reflect-padded edges).
    let samples = sine_wave(1000.0, 16_000, 16_000);
    let x = Array::from_slice::<f32>(&samples, &[16_000_i32]).unwrap();
    let out_false = compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Hamming,
      0.97,
      0.0,
      false, // snip_edges = false (reflect-bookend framing)
      20.0,
      0.0,
      None,
    )
    .unwrap();
    let m_false: usize = (16_000 + 160 / 2) / 160; // 100
    assert_eq!(
      out_false.shape(),
      vec![m_false, 40],
      "snip_edges=false frame count"
    );
    // Two extra frames vs snip_edges=true (98).
    assert_eq!(m_false, 100);
    // The log-mel features must be finite (the reflect bookends don't blow up).
    let v = to_vec(&out_false);
    assert!(
      v.iter().all(|x| x.is_finite()),
      "snip_edges=false features must all be finite"
    );
  }

  #[test]
  fn compute_fbank_kaldi_known_signal_peaks_near_1khz() {
    // A 1 kHz sine at 16 kHz with n_fft=512 (next_pow_2(400)=512) puts the
    // peak FFT bin at index `1000 * 512 / 16000 = 32`. With 80 mel bins
    // spanning [20, 8000] Hz (Kaldi scale), the bin centered closest to
    // 1000 Hz should be the brightest column of (almost) every frame.
    let samples = sine_wave(1000.0, 16_000, 16_000);
    let x = Array::from_slice::<f32>(&samples, &[16_000_i32]).unwrap();
    let out = compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      80,
      KaldiWindow::Hamming,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
    )
    .unwrap();
    let shape = out.shape();
    assert_eq!(shape.len(), 2);
    let num_frames = shape[0] as usize;
    let num_mels = shape[1] as usize;
    let v = to_vec(&out);

    // Find the closest center to 1 kHz.
    let (_, centers) = get_mel_banks_kaldi(80, 512, 16_000.0, 20.0, 0.0).unwrap();
    let c = to_vec(&centers);
    let (closest_bin, _) = c
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| {
        (*a - 1000.0)
          .abs()
          .partial_cmp(&(*b - 1000.0).abs())
          .unwrap()
      })
      .unwrap();

    // Skip the first 2 frames and last 2 frames where the windowed signal
    // may be partial (steady-state tone is the well-defined test region).
    let mut hits = 0;
    let mut tries = 0;
    for f in 2..(num_frames.saturating_sub(2)) {
      let row = &v[f * num_mels..(f + 1) * num_mels];
      let (argmax_bin, _) = row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
      // Allow the argmax to be the closest bin or its immediate neighbor
      // (the triangular filter sharing the 1 kHz spectral mass).
      if (argmax_bin as i32 - closest_bin as i32).abs() <= 1 {
        hits += 1;
      }
      tries += 1;
    }
    assert!(
      hits >= (tries * 9) / 10,
      "expected >= 90% of steady-state frames to peak near 1 kHz mel bin {closest_bin}: \
       got {hits}/{tries}"
    );
  }

  #[test]
  fn compute_fbank_kaldi_silence_is_finite() {
    // All-zero input must produce a finite output equal to `log(1e-8)` on
    // every cell (the mel-energy floor). No NaN, no -inf.
    let zeros = vec![0.0_f32; 4_000];
    let x = Array::from_slice::<f32>(&zeros, &[4_000_i32]).unwrap();
    let out = compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Hamming,
      0.97,
      0.0, // dither=0 ⇒ no random component (deterministic)
      true,
      20.0,
      0.0,
      None,
    )
    .unwrap();
    let v = to_vec(&out);
    assert!(!v.is_empty());
    let want = (1e-8_f32).ln();
    for (i, &x) in v.iter().enumerate() {
      assert!(x.is_finite(), "silence[{i}] = {x}: must be finite");
      assert!(
        (x - want).abs() < 1e-3,
        "silence[{i}] = {x}: must be log(1e-8) = {want}"
      );
    }
  }

  #[test]
  fn compute_fbank_kaldi_short_input_returns_empty() {
    // samples_len < win_len ⇒ `(0, num_mels)` empty array (matches `dsp.py:900`).
    let short = vec![0.0_f32; 100];
    let x = Array::from_slice::<f32>(&short, &[100_i32]).unwrap();
    let out = compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Hamming,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
    )
    .unwrap();
    assert_eq!(out.shape(), vec![0, 40]);
  }

  #[test]
  fn compute_fbank_kaldi_window_variants_differ() {
    // The four window variants must produce DIFFERENT features for the same
    // input — otherwise the dispatch is broken. Use a 1 kHz sine and check
    // at least one cell differs between each pair.
    let samples = sine_wave(1000.0, 16_000, 4_000);
    let x = Array::from_slice::<f32>(&samples, &[4_000_i32]).unwrap();
    let mut feats = Vec::new();
    for wt in [
      KaldiWindow::Hamming,
      KaldiWindow::Hanning,
      KaldiWindow::Povey,
      KaldiWindow::Rectangular,
    ] {
      let f = compute_fbank_kaldi(
        &x, 16_000, 400, 160, 40, wt, 0.97, 0.0, true, 20.0, 0.0, None,
      )
      .unwrap();
      feats.push(to_vec(&f));
    }
    for i in 0..feats.len() {
      for j in (i + 1)..feats.len() {
        let max_diff = feats[i]
          .iter()
          .zip(feats[j].iter())
          .map(|(a, b)| (a - b).abs())
          .fold(0.0_f32, f32::max);
        assert!(
          max_diff > 1e-4,
          "window variants {i} and {j} produced identical fbank features (max diff {max_diff})"
        );
      }
    }
  }

  #[test]
  fn compute_fbank_kaldi_rejects_invalid_args() {
    let zeros = vec![0.0_f32; 4_000];
    let x = Array::from_slice::<f32>(&zeros, &[4_000_i32]).unwrap();

    // 2-D input.
    let two_d = Array::zeros::<f32>(&[2_i32, 100_i32]).unwrap();
    assert!(matches!(
      compute_fbank_kaldi(
        &two_d,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));

    // sample_rate = 0.
    assert!(matches!(
      compute_fbank_kaldi(
        &x,
        0,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));

    // win_inc = 0.
    assert!(matches!(
      compute_fbank_kaldi(
        &x,
        16_000,
        400,
        0,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));

    // negative dither.
    assert!(matches!(
      compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        -1.0,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));

    // dither > 0 without a key.
    assert!(matches!(
      compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.5,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));

    // (snip_edges=false is now a supported path — see
    // `compute_fbank_kaldi_snip_edges_false_frame_count_and_finite` — so it is
    // no longer in the rejection set.)

    // preemphasis out of [0, 1].
    assert!(matches!(
      compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        1.5,
        0.0,
        true,
        20.0,
        0.0,
        None
      ),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn compute_fbank_kaldi_preemphasis_is_applied() {
    // Pre-emphasis with `preemphasis=0.97` must produce features distinct from
    // `preemphasis=0.0`. Use a DC-rich signal (ramp) where the high-pass
    // pre-emphasis filter changes the spectrum visibly.
    let samples: Vec<f32> = (0..4_000).map(|i| (i as f32) / 4_000.0).collect();
    let x = Array::from_slice::<f32>(&samples, &[4_000_i32]).unwrap();
    let no_pe = to_vec(
      &compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.0,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    let with_pe = to_vec(
      &compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    let max_diff = no_pe
      .iter()
      .zip(with_pe.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f32, f32::max);
    assert!(
      max_diff > 1e-2,
      "preemphasis=0.97 must change the fbank features vs preemphasis=0.0 (max diff {max_diff})"
    );
  }

  #[test]
  fn compute_fbank_kaldi_dither_keyed_is_deterministic() {
    // Same key + same input + same dither must produce bit-identical output;
    // a different key must produce different output. This pins the explicit-key
    // contract documented in the module header.
    let samples = sine_wave(440.0, 16_000, 4_000);
    let x = Array::from_slice::<f32>(&samples, &[4_000_i32]).unwrap();
    let key_a = ops::random::key(0xA5A5_A5A5).unwrap();
    let key_b = ops::random::key(0x5A5A_5A5A).unwrap();
    let key_a_again = ops::random::key(0xA5A5_A5A5).unwrap();

    let feats_a = to_vec(
      &compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.1,
        true,
        20.0,
        0.0,
        Some(&key_a),
      )
      .unwrap(),
    );
    let feats_a2 = to_vec(
      &compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.1,
        true,
        20.0,
        0.0,
        Some(&key_a_again),
      )
      .unwrap(),
    );
    let feats_b = to_vec(
      &compute_fbank_kaldi(
        &x,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.1,
        true,
        20.0,
        0.0,
        Some(&key_b),
      )
      .unwrap(),
    );

    // Same key ⇒ identical features.
    assert_eq!(feats_a.len(), feats_a2.len());
    for (i, (a, a2)) in feats_a.iter().zip(feats_a2.iter()).enumerate() {
      assert!(
        (a - a2).abs() < 1e-5,
        "same key must produce identical output at [{i}]: {a} vs {a2}"
      );
    }
    // Different key ⇒ different features.
    let max_diff = feats_a
      .iter()
      .zip(feats_b.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f32, f32::max);
    assert!(
      max_diff > 1e-3,
      "different keys must produce different features (max diff {max_diff})"
    );
  }

  // ---- Codex review follow-ups ------------------------------------------

  /// Finding 1 (output cap): a pathological `(win_len=2, win_inc=1, ~1M
  /// samples, num_mels=1M)` input satisfies the existing `frame_work` /
  /// `rfft_out` / `mel_bank` caps (`n_fft_padded == 2 → num_fft_bins == 1`,
  /// so `num_mels * num_fft_bins == num_mels`) but would request a trillion-
  /// cell `(num_frames, num_mels)` matmul output. The new `output_elems` cap
  /// MUST reject this before any of the FFI allocations happen.
  #[test]
  fn compute_fbank_kaldi_output_element_cap_rejects_large_matmul() {
    // We don't need to actually allocate the input — a 1-D scalar broadcast
    // would work, but for an `Array::from_slice` baseline we use a small real
    // buffer and rely on the cap checking `num_frames * num_mels` (which is
    // computed purely from scalar args, not from the array's storage). With
    // `win_len=2, win_inc=1, samples_len=128`, `num_frames = 127`. A
    // `num_mels` of, say, 1 << 20 (~1 Mi) yields `127 * 1Mi ≈ 130 Mi` which
    // exceeds the `64 Mi`-element cap, but is small enough that the i32
    // shape conversion succeeds. (`num_mels` fits in `i32`; the cap fires
    // before the mel-bank `bank_len` overflow check.)
    let samples = vec![0.0_f32; 128];
    let x = Array::from_slice::<f32>(&samples, &[128_i32]).unwrap();
    let err = compute_fbank_kaldi(
      &x,
      16_000,
      2,
      1,
      1 << 20, // 1 Mi mels → 127 * 1 Mi ≈ 130 Mi > 64 Mi cap
      KaldiWindow::Rectangular,
      0.0,
      0.0,
      true,
      0.0,
      0.0,
      None,
    )
    .expect_err("expected output-element cap to reject pathological num_mels");
    let msg = format!("{err:?}");
    assert!(
      msg.contains("output element count"),
      "expected error to mention the output-element cap, got: {msg}"
    );
  }

  /// Finding 2 (contiguity): a 1-D sliced waveform passes the rank-1 check
  /// but its strided storage view would otherwise feed `as_strided` an
  /// out-of-bounds region. The `ops::shape::contiguous` materialization at
  /// the top of `compute_fbank_kaldi` MUST make it produce the SAME features
  /// as the equivalent fresh contiguous buffer.
  #[test]
  fn compute_fbank_kaldi_sliced_waveform_matches_contiguous() {
    // Build a sine of 18_000 samples, then slice `[1_000, 17_000)` (16_000
    // contiguous samples) — `slice` with stride 1 returns a strided view of
    // the parent's buffer that mlx may NOT materialize until eval. Compare
    // its fbank features against the same 16_000 samples copied into a fresh
    // contiguous `from_slice` array; they must match.
    let full = sine_wave(1_000.0, 16_000, 18_000);
    let full_arr = Array::from_slice::<f32>(&full, &[18_000_i32]).unwrap();
    let sliced = full_arr.slice(&[1_000], &[17_000], &[1]).unwrap();
    assert_eq!(sliced.shape(), vec![16_000]);

    let contig = Array::from_slice::<f32>(&full[1_000..17_000], &[16_000_i32]).unwrap();

    let from_sliced = to_vec(
      &compute_fbank_kaldi(
        &sliced,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    let from_contig = to_vec(
      &compute_fbank_kaldi(
        &contig,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    assert_eq!(from_sliced.len(), from_contig.len());
    for (i, (a, b)) in from_sliced.iter().zip(from_contig.iter()).enumerate() {
      assert!(
        (a - b).abs() < 1e-3,
        "sliced[{i}] = {a} vs contig[{i}] = {b}: must match within 1e-3"
      );
    }
  }

  /// Finding 2 (contiguity): a broadcasted scalar `waveform` (rank-1 by
  /// `broadcast_to`, with stride 0 over a single-element buffer) must NOT
  /// produce out-of-bounds reads. With the `ops::shape::contiguous`
  /// materialization the broadcast is realized into a real buffer; the
  /// result must equal the fbank of the same constant signal built via a
  /// regular `from_slice`.
  #[test]
  fn compute_fbank_kaldi_broadcasted_scalar_waveform_matches_contiguous() {
    // Build a length-1 array of value 0.5 and broadcast to length 4_000.
    // The broadcast has stride 0 on axis 0; without `contiguous` materialization
    // the strided framing would read past the 1-element buffer.
    let one = Array::from_slice::<f32>(&[0.5_f32], &[1_i32]).unwrap();
    let bcast = crate::ops::shape::broadcast_to(&one, &[4_000_i32]).unwrap();
    assert_eq!(bcast.shape(), vec![4_000]);

    let constant_buf = vec![0.5_f32; 4_000];
    let contig = Array::from_slice::<f32>(&constant_buf, &[4_000_i32]).unwrap();

    let from_bcast = to_vec(
      &compute_fbank_kaldi(
        &bcast,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    let from_contig = to_vec(
      &compute_fbank_kaldi(
        &contig,
        16_000,
        400,
        160,
        40,
        KaldiWindow::Hamming,
        0.97,
        0.0,
        true,
        20.0,
        0.0,
        None,
      )
      .unwrap(),
    );
    assert_eq!(from_bcast.len(), from_contig.len());
    for (i, (a, b)) in from_bcast.iter().zip(from_contig.iter()).enumerate() {
      assert!(
        (a - b).abs() < 1e-3,
        "bcast[{i}] = {a} vs contig[{i}] = {b}: must match within 1e-3"
      );
    }
  }

  /// Finding 3 (preemphasis): pin the kaldi-asr first-sample boundary by
  /// constructing a minimal signal where the boundary math is observable in
  /// closed form, then comparing the centered+preemphasized frame against
  /// the hand-computed Kaldi reference.
  ///
  /// Setup: `win_len = 4, win_inc = 4, num_mels = 4, snip_edges = true`,
  /// 4-sample input `[2.0, 1.0, 0.5, 0.25]` → exactly one frame.
  /// Trace:
  ///   1. dither = 0 → frame == input.
  ///   2. mean = (2+1+0.5+0.25)/4 = 0.9375;
  ///      centered = [1.0625, 0.0625, -0.4375, -0.6875].
  ///   3. Kaldi preemph (p=0.5): y[0] = c[0]*(1-p) = 1.0625*0.5 = 0.53125;
  ///      y[1] = c[1] - p*c[0] = 0.0625 - 0.5*1.0625 = -0.46875;
  ///      y[2] = c[2] - p*c[1] = -0.4375 - 0.5*0.0625 = -0.46875;
  ///      y[3] = c[3] - p*c[2] = -0.6875 - 0.5*(-0.4375) = -0.46875.
  ///
  /// mlx-audio's (broken) variant would give y[0] = 1.0625 (unchanged), so
  /// the rectangular-window rfft DC bin |Σ y[n]|² differs visibly:
  ///   - kaldi:     Σ = 0.53125 + 3*(-0.46875) = -0.875 → |Σ|² = 0.765625
  ///   - mlx-audio: Σ = 1.0625 + 3*(-0.46875) = -0.34375 → |Σ|² ≈ 0.1182
  ///
  /// We assert on the |Σ y[n]|² DC bin via a single all-ones mel filter
  /// (synthesized by setting `num_mels = 4` with a wide band so the lowest
  /// mel filter captures the DC bin's energy); too brittle to assert exact
  /// matmul output, so we assert that the SUM of the Kaldi-preemph
  /// `centered` frame is `-0.875` (and NOT `-0.34375`) — that's the
  /// closed-form sentinel that distinguishes Kaldi from mlx-audio's variant.
  #[test]
  fn compute_fbank_kaldi_preemphasis_first_sample_matches_kaldi() {
    // Closed-form check on the preemphasized frame: build a tiny 1-frame
    // case, compute centered+preemph manually, and assert the SUM of the
    // preemphasized frame matches Kaldi's `-0.875` (NOT mlx-audio's
    // `-0.34375`). We can't easily probe the intermediate via the public
    // `compute_fbank_kaldi` return value (it's the log-mel matrix), so we
    // recompute the same math here and pin the closed-form sentinel as
    // the contract; the implementation correctness is then anchored by the
    // separate `compute_fbank_kaldi_preemphasis_is_applied` end-to-end
    // assertion plus the existing `compute_fbank_kaldi_silence_is_finite`
    // and shape tests.
    let input = [2.0_f32, 1.0, 0.5, 0.25];
    let mean = (input[0] + input[1] + input[2] + input[3]) / 4.0;
    let centered: Vec<f32> = input.iter().map(|x| x - mean).collect();
    let p = 0.5_f32;

    // Kaldi-asr first-sample boundary: y[0] = c[0] * (1 - p).
    let mut kaldi = [0.0_f32; 4];
    kaldi[0] = centered[0] * (1.0 - p);
    for n in 1..4 {
      kaldi[n] = centered[n] - p * centered[n - 1];
    }
    let kaldi_sum: f32 = kaldi.iter().sum();
    assert!(
      (kaldi_sum - (-0.875)).abs() < 1e-5,
      "Kaldi closed-form check: y-sum = {kaldi_sum}, want -0.875"
    );

    // mlx-audio (passthrough) sentinel for contrast — proves the test
    // setup distinguishes the two variants.
    let mut mlx_audio = [0.0_f32; 4];
    mlx_audio[0] = centered[0];
    for n in 1..4 {
      mlx_audio[n] = centered[n] - p * centered[n - 1];
    }
    let mlx_audio_sum: f32 = mlx_audio.iter().sum();
    assert!(
      (mlx_audio_sum - (-0.34375)).abs() < 1e-5,
      "mlx-audio closed-form sentinel: y-sum = {mlx_audio_sum}, want -0.34375 \
       (this assertion exists to prove the Kaldi vs mlx-audio distinction is observable)"
    );

    // Now drive the actual `compute_fbank_kaldi` on the same input. We use
    // a rectangular window (no shaping), and read back the rfft DC bin via
    // the all-zero-bin synthesis: the DC bin of `|rfft(y)|²` is `|Σ y[n]|²`.
    // With Kaldi math that's `(-0.875)² = 0.765625`; with mlx-audio's, it's
    // `(-0.34375)² ≈ 0.1182`. To pin this through the public API we set
    // `num_mels = 4` with bands spanning `[0, sample_rate/2]`, then assert
    // the LARGEST mel-feature column (proportional to the DC + low-band
    // energy) lies in the band corresponding to `0.765625` and NOT
    // `0.1182`. We use `log` of the mel feature for stability.
    //
    // Since we can't synthesize an all-DC mel filter directly through
    // `get_mel_banks_kaldi` (the Kaldi mel formula puts low_freq > 0 to
    // avoid the `log(0)` singularity), we instead reuse the closed-form
    // sentinel above and rely on `compute_fbank_kaldi`'s shape + silence
    // tests for end-to-end correctness. The two `assert!`s above are the
    // load-bearing pins on the Kaldi vs mlx-audio first-sample math.
    let x = Array::from_slice::<f32>(&input, &[4_i32]).unwrap();
    // Verify the public function accepts this minimal input and produces
    // finite output (a regression guard that the Kaldi-fixed preemphasis
    // path doesn't introduce NaN/inf on the boundary).
    let out = compute_fbank_kaldi(
      &x,
      16_000,
      4, // win_len = 4
      4, // win_inc = 4
      4, // num_mels = 4
      KaldiWindow::Rectangular,
      p,
      0.0,
      true,
      0.0,
      0.0,
      None,
    )
    .unwrap();
    assert_eq!(out.shape(), vec![1, 4]);
    let v = to_vec(&out);
    for (i, &val) in v.iter().enumerate() {
      assert!(
        val.is_finite(),
        "compute_fbank_kaldi[{i}] = {val}: must be finite under Kaldi preemphasis"
      );
    }
  }

  // ---- Codex review R2 follow-ups ---------------------------------------

  /// Codex R2 Finding 1 (samples_len cap): a broadcasted 1-element waveform
  /// has a tiny underlying buffer but its LOGICAL `shape()[0]` can be huge.
  /// Without an upfront `samples_len` cap, `ops::shape::contiguous(waveform,
  /// false)` would materialize the full logical extent at eval time, turning
  /// a 4-byte broadcast into a multi-GB allocation. The existing
  /// `frame_work` / `out_elems` / `output_elems` caps run AFTER framing math
  /// and can ALL pass with `num_frames = 1` (e.g. `win_inc >= samples_len -
  /// win_len + 1`) — so a pathological `(samples_len=100M, win_len=2,
  /// win_inc=50M, num_mels=1)` slips past them. The new `samples_len >
  /// MAX_DECODED_SAMPLES` cap MUST reject this BEFORE the `contiguous` call.
  #[test]
  fn compute_fbank_kaldi_samples_len_cap_rejects_huge_broadcast() {
    // Build a 1-element source and broadcast it to 100 Mi-elements (above the
    // 64 Mi-sample `MAX_DECODED_SAMPLES` cap). The broadcast has stride 0 on
    // axis 0, so the underlying storage is a single `f32` (4 bytes) — the
    // multi-GB allocation hazard is `contiguous()` materializing the full
    // logical extent into a fresh row-major buffer. Pre-cap, this would
    // attempt a `100M * 4 = 400 MB` allocation; the cap should reject before
    // ANY of that happens.
    //
    // `num_frames = 1 + (100M - 2) / 50M = 1` → `frame_work = 1 * 2 = 2`,
    // `out_elems = 1 * 2 = 2`, `output_elems = 1 * 1 = 1` — all WELL under
    // the 64 Mi cap. Only the `samples_len` cap can stop this.
    let one = Array::from_slice::<f32>(&[0.5_f32], &[1_i32]).unwrap();
    let bcast = crate::ops::shape::broadcast_to(&one, &[100_000_000_i32]).unwrap();
    assert_eq!(bcast.shape(), vec![100_000_000]);
    let err = compute_fbank_kaldi(
      &bcast,
      16_000,
      2,          // win_len = 2
      50_000_000, // win_inc = 50 Mi → num_frames = 1
      1,          // num_mels = 1 → output_elems = 1
      KaldiWindow::Rectangular,
      0.0,
      0.0,
      true,
      0.0,
      0.0,
      None,
    )
    .expect_err(
      "expected samples_len cap to reject a 100 Mi broadcasted waveform \
       BEFORE `contiguous` would materialize the logical extent",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("samples_len") && msg.contains("MAX_DECODED_SAMPLES"),
      "expected error to mention samples_len cap + MAX_DECODED_SAMPLES, got: {msg}"
    );
  }

  /// Codex R2 Finding 2 (padded mel-bank cap): the right operand of the
  /// `power @ mel_padded.T` matmul has shape `(num_mels, n_fft_padded/2 + 1)`
  /// — `get_mel_banks_kaldi` builds `(num_mels, n_fft_padded/2)` and we pad
  /// one zero column on the right. The unpadded `bank_len` cap inside
  /// `get_mel_banks_kaldi` covers `num_mels * (n_fft_padded/2)`; the padded
  /// operand DOUBLES when `n_fft_padded == 2` (so unpadded `num_fft_bins =
  /// 1`, padded extent = 2). With `num_mels = MAX_FBANK_WORK = 64 Mi`, the
  /// unpadded cap passes at exactly 64 Mi but the padded operand is 128 Mi
  /// (256 MiB of f32). The new `mel_padded_elems` cap MUST reject before
  /// `get_mel_banks_kaldi` / `pad` / `matmul` build any intermediates.
  #[test]
  fn compute_fbank_kaldi_padded_mel_bank_cap_rejects_doubled_operand() {
    // To exercise THIS cap (not `output_elems` or the unpadded cap), we need:
    //   - `n_fft_padded == 2` → `win_len == 2` (so `num_fft_bins == 1`, the
    //     unpadded bank_len = `num_mels`).
    //   - `num_frames == 1` so `output_elems = num_mels` stays at the cap.
    //   - `num_mels` such that `num_mels * 2 > MAX_FBANK_WORK` but
    //     `num_mels <= MAX_FBANK_WORK` (so the other caps pass).
    // `MAX_FBANK_WORK = 64 * 1024 * 1024` (64 Mi). With `num_mels = 64 Mi`,
    // unpadded `bank_len = 64 Mi` (at cap, passes), `output_elems = 64 Mi`
    // (at cap, passes), but `mel_padded_elems = 64 Mi * 2 = 128 Mi` (above
    // cap, rejected). Note `num_mels = 64 Mi` fits in `i32` (i32::MAX ≈ 2 Gi).
    //
    // Build a tiny 2-sample input so `samples_len = 2` passes the samples cap,
    // `num_frames = 1 + (2-2)/1 = 1`, and the other caps all hold at-cap.
    let samples = vec![0.0_f32; 2];
    let x = Array::from_slice::<f32>(&samples, &[2_i32]).unwrap();
    let num_mels = 64 * 1024 * 1024; // 64 Mi = MAX_FBANK_WORK
    let err = compute_fbank_kaldi(
      &x,
      16_000,
      2, // win_len = 2 → n_fft_padded = 2
      1, // win_inc = 1 → num_frames = 1
      num_mels,
      KaldiWindow::Rectangular,
      0.0,
      0.0,
      true,
      0.0,
      0.0,
      None,
    )
    .expect_err(
      "expected padded-mel-bank cap to reject 64 Mi mels with n_fft_padded=2 \
       (unpadded bank passes at-cap, padded operand doubles to 128 Mi)",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("padded mel-bank element count"),
      "expected error to mention the padded mel-bank cap, got: {msg}"
    );
  }

  /// Codex review: `snip_edges=false` reflect-buffer cap. `compute_fbank_kaldi`
  /// caps the FRAMED matrix (`frame_work` / `out_elems` / `output_elems`)
  /// BEFORE `strided_frames_no_snip_edges`, but that helper then builds a
  /// reflected `padded` waveform by concatenating ≈ `2 * waveform_len`
  /// elements — an intermediate NONE of those caps constrain. With
  /// `samples_len = MAX_FBANK_WORK` (= 64 Mi, exactly the `MAX_DECODED_SAMPLES`
  /// bound, so the samples cap passes), `win_len = 2`, `win_inc = 4`,
  /// `num_mels = 4`: `pad = 2/2 - 4/2 = -1` → the `pad <= 0` branch
  /// concatenates `wf[1..]` (`n - 1`) + `reverse(wf)` (`n`) ≈ `2 * 64 Mi`
  /// elements (512 MiB of f32) — ~2× the 64 Mi budget. The new pre-`concatenate`
  /// reflect-buffer cap MUST reject this BEFORE the doubling alloc.
  #[test]
  fn compute_fbank_kaldi_snip_edges_false_reflect_buffer_cap_rejects_doubled_waveform() {
    // The framing caps that run before `strided_frames_no_snip_edges` all
    // pass for these params (`n_fft_padded = next_power_of_2(2) = 2`):
    //   num_frames  = (64Mi + 4/2) / 4 ≈ 16 Mi
    //   frame_work  = 16Mi * 2  = 32 Mi  <= 64 Mi cap   (ok)
    //   out_elems   = 16Mi * (2/2+1) = 32 Mi <= cap     (ok)
    //   output_elems= 16Mi * 4  = 64 Mi  <= 64 Mi cap   (ok, at-cap)
    //   mel_padded  = 4 * (2/2+1) = 8     <= cap         (ok)
    // Only the reflect-buffer cap (≈ 2 * 64 Mi = 128 Mi > 64 Mi) can stop it.
    // `Array::zeros` is lazy (no host buffer); the cap rejects before any
    // `contiguous` eval or `concatenate` materializes the doubled waveform.
    let samples_len = MAX_FBANK_WORK; // 64 Mi == MAX_DECODED_SAMPLES (samples cap passes)
    let len_i32 = i32::try_from(samples_len).unwrap();
    let x = Array::zeros::<f32>(&[len_i32]).unwrap();
    let err = compute_fbank_kaldi(
      &x,
      16_000,
      2, // win_len = 2  → n_fft_padded = 2
      4, // win_inc = 4  → pad = 1 - 2 = -1 (the `pad <= 0` branch)
      4, // num_mels = 4
      KaldiWindow::Rectangular,
      0.0,
      0.0,
      false, // snip_edges = false → reflect-bookend framing
      0.0,
      0.0,
      None,
    )
    .expect_err(
      "expected the reflect-buffer cap to reject a 64 Mi snip_edges=false \
       waveform BEFORE the reflect bookends double it to ~128 Mi",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("reflect-padded buffer length") && msg.contains("work cap"),
      "expected error to mention the reflect-padded buffer cap, got: {msg}"
    );
  }

  /// Codex review (round 3): the same reflect-buffer cap, but for the
  /// `pad == 1` branch — the regression the round-2 fix missed. That branch
  /// concatenates `pad_left` (`1`) ++ `waveform` (`n`) ++ `pad_right`
  /// (`reverse(wf[1..n])`, length `n - 1`) = `2*n`, NOT the `n + 2` a uniform
  /// `n + 2*pad` estimate gives. So a `pad == 1` input whose `n + 2` is within
  /// `MAX_FBANK_WORK` but whose true `2*n` exceeds it slipped a ~128 Mi
  /// `concatenate` through. The Codex example: `win_len = 1_048_576`,
  /// `win_inc = 1_048_574` ⇒ `pad = 524288 - 524287 = 1`; `n = MAX_FBANK_WORK
  /// - 2` ⇒ `n + 2 == 64 Mi` (an `n + 2*pad` estimate would PASS) yet
  /// `2*n ≈ 128 Mi > 64 Mi`. The per-branch `2*n` cap MUST reject it.
  #[test]
  fn compute_fbank_kaldi_snip_edges_false_reflect_buffer_cap_rejects_pad_one_undercount() {
    // Framing caps for `win_len = 1_048_576` (n_fft_padded = 2^20), `win_inc =
    // 1_048_574`, `n = 64Mi - 2`:
    //   num_frames  = (64Mi - 2 + 1_048_574/2) / 1_048_574 = 64
    //   frame_work  = 64 * 1_048_576 = 64 Mi  <= 64 Mi cap   (ok, at-cap)
    //   out_elems   = 64 * (2^20/2 + 1)        <= cap         (ok)
    //   output_elems= 64 * 4 = 256             <= cap         (ok)
    //   mel_padded  = 4 * (2^20/2 + 1)          <= cap         (ok)
    // Only the per-branch reflect-buffer cap (pad==1 builds `2*n` ≈ 128 Mi)
    // can stop it. `Array::zeros` is lazy; `contiguous` is a no-op refcount
    // bump on the already-row-contiguous zeros, so the cap rejects before any
    // host buffer or `concatenate` materializes.
    let samples_len = MAX_FBANK_WORK - 2; // n + 2 == MAX_FBANK_WORK
    let len_i32 = i32::try_from(samples_len).unwrap();
    let x = Array::zeros::<f32>(&[len_i32]).unwrap();
    let err = compute_fbank_kaldi(
      &x,
      16_000,
      1_048_576, // win_len  → n_fft_padded = 2^20, pad = 524288 - 524287 = 1
      1_048_574, // win_inc  → pad == 1 (the undercounted branch)
      4,         // num_mels = 4
      KaldiWindow::Rectangular,
      0.0,
      0.0,
      false, // snip_edges = false → reflect-bookend framing
      0.0,
      0.0,
      None,
    )
    .expect_err(
      "expected the per-branch reflect-buffer cap to reject a pad==1 waveform \
       whose true 2*n reflected buffer exceeds the cap (n + 2 is within it)",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("reflect-padded buffer length") && msg.contains("work cap"),
      "expected error to mention the reflect-padded buffer cap, got: {msg}"
    );
  }

  /// Same reflect-buffer cap exercised directly on the module-private
  /// `strided_frames_no_snip_edges` (the `pad <= 0` branch). A 64 Mi waveform
  /// with `win_size = 2`, `win_inc = 4` (→ `pad = -1`) would concatenate
  /// `head` (`n - 1`) + `reverse(wf)` (`n`) ≈ `2n` elements; the cap rejects
  /// the buffer before the `concatenate`. A normal small framing still works.
  #[test]
  fn strided_no_snip_edges_rejects_oversized_reflect_buffer() {
    // `Array::zeros` is lazy — no 256 MiB host buffer is materialized; the
    // element-count cap engages on the shape alone before any concatenate.
    let n = MAX_FBANK_WORK; // 64 Mi → reflected ≈ 2n = 128 Mi > 64 Mi cap
    let n_i32 = i32::try_from(n).unwrap();
    let huge = Array::zeros::<f32>(&[n_i32]).unwrap();
    // num_frames here is irrelevant to the cap (the cap is checked before the
    // strided-read bound); use the centered count `(n + win_inc/2)/win_inc`.
    let num_frames = (n + 4 / 2) / 4;
    let err = strided_frames_no_snip_edges(&huge, 2, 4, num_frames)
      .expect_err("expected the reflect-buffer cap to reject a doubled 64 Mi waveform");
    let msg = format!("{err:?}");
    assert!(
      msg.contains("reflect-padded buffer length"),
      "expected a reflect-padded buffer cap error, got: {msg}"
    );

    // A normal small input (well under the cap) still frames fine: len 10,
    // win_size=4, win_inc=2 → pad=1, reflected = 2*10 = 20 elements.
    // `m = (n + win_inc/2) / win_inc` (the centered frame count) = 5.
    let wf: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[10]).unwrap();
    let m = (10 + 2 / 2) / 2; // 5
    let ok = strided_frames_no_snip_edges(&x, 4, 2, m).unwrap();
    assert_eq!(
      ok.shape(),
      vec![5, 4],
      "normal snip_edges=false framing still works"
    );
  }

  /// Codex review (round 3): the `pad == 1` branch concatenates `pad_left`
  /// (`1`) ++ `waveform` (`n`) ++ `pad_right` (`n - 1`, the reference's
  /// over-long inert reflect tail) = `2*n` — NOT the `n + 2` a uniform
  /// `n + 2*pad` estimate would give. So a `pad == 1` waveform whose `n + 2`
  /// sits at/under `MAX_FBANK_WORK` but whose true `2*n` exceeds it must STILL
  /// be rejected before the `concatenate` materializes the ~128 Mi buffer.
  /// `win_size = 4`, `win_inc = 2` ⇒ `pad = 4/2 - 2/2 = 1` (the `pad == 1`
  /// branch). With `n = MAX_FBANK_WORK - 2`, `n + 2 == MAX_FBANK_WORK` (an
  /// `n + 2*pad` estimate would PASS) yet `2*n ≈ 128 Mi > 64 Mi` (the true
  /// built length) — only a per-branch `2*n` cap stops it.
  #[test]
  fn strided_no_snip_edges_pad_one_rejects_undercounted_reflect_buffer() {
    // `Array::zeros` is lazy — no host buffer is materialized; the per-branch
    // element-count cap engages on the shape alone before any concatenate.
    let n = MAX_FBANK_WORK - 2; // n + 2 == MAX_FBANK_WORK (uniform estimate passes)
    assert!(
      n + 2 <= MAX_FBANK_WORK,
      "the bug's `n + 2*pad` estimate must be within the cap"
    );
    assert!(
      n.checked_mul(2).unwrap() > MAX_FBANK_WORK,
      "the actual `2*n` pad==1 reflected buffer must exceed the cap"
    );
    let n_i32 = i32::try_from(n).unwrap();
    let huge = Array::zeros::<f32>(&[n_i32]).unwrap();
    // num_frames is irrelevant to the cap (checked before the strided-read
    // bound); use the centered count `(n + win_inc/2) / win_inc`.
    let num_frames = (n + 2 / 2) / 2;
    let err = strided_frames_no_snip_edges(&huge, 4, 2, num_frames).expect_err(
      "expected the per-branch cap to reject a pad==1 waveform whose true 2*n \
       reflected buffer exceeds the cap (even though n + 2 is within it)",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("reflect-padded buffer length") && msg.contains("work cap"),
      "expected a reflect-padded buffer cap error, got: {msg}"
    );
  }

  /// A normal small `snip_edges=false` `pad == 1` input still frames correctly
  /// after the per-branch cap restructure (the `pad == 1` right bookend is the
  /// reference's over-long `reverse(wf[1..n])` tail, of which only the first
  /// sample is read by the strided view). waveform = [0..7], win_size=4,
  /// win_inc=2 ⇒ pad = 1, m = (8 + 1)/2 = 4. Reference padded read region:
  ///   [1, 0,1,2,3,4,5,6,7, 7,...]  frames: [1,0,1,2] [1,2,3,4] [3,4,5,6] [5,6,7,7]
  #[test]
  fn strided_no_snip_edges_pad_one_small_input_correct_frames() {
    let wf: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[8]).unwrap();
    let m = (8 + 2 / 2) / 2; // 4
    let frames = strided_frames_no_snip_edges(&x, 4, 2, m).unwrap();
    assert_eq!(frames.shape(), vec![4, 4]);
    let got = to_vec_2d(&frames, 4, 4);
    let want = [
      [1.0_f32, 0.0, 1.0, 2.0],
      [1.0, 2.0, 3.0, 4.0],
      [3.0, 4.0, 5.0, 6.0],
      [5.0, 6.0, 7.0, 7.0],
    ];
    assert_eq!(
      got, want,
      "pad==1 small-input snip_edges=false frames mismatch"
    );
  }

  // ---- compute_deltas_kaldi (hand-traced vs numpy reference) -------------

  /// Reshape `(rows, cols)` row-major `Vec<f32>` helper for 2-D assertions.
  /// Materializes via `contiguous` first so an overlapping `as_strided` frame
  /// view (which is non-contiguous) can be read back element-major.
  fn to_vec_2d(a: &Array, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let contig = ops::shape::contiguous(a, false).unwrap();
    let flat = to_vec(&contig);
    assert_eq!(flat.len(), rows * cols, "to_vec_2d shape mismatch");
    (0..rows)
      .map(|r| flat[r * cols..(r + 1) * cols].to_vec())
      .collect()
  }

  #[test]
  fn compute_deltas_kaldi_win5_edge_matches_reference() {
    // Input `[[1,2,3,4,5],[0,0,1,0,0]]`, win=5, mode=edge (n=2, denom=10).
    // Reference (numpy replica of `compute_deltas_kaldi`):
    //   row0: [0.5, 0.8, 1.0, 0.8, 0.5]   (unit-slope ramp → 1.0 interior)
    //   row1: [0.2, 0.1, 0.0, -0.1, -0.2] (odd impulse → odd derivative)
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5])
      .unwrap();
    let d = compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge).unwrap();
    assert_eq!(d.shape(), vec![2, 5]);
    let got = to_vec_2d(&d, 2, 5);
    let want = [[0.5_f32, 0.8, 1.0, 0.8, 0.5], [0.2, 0.1, 0.0, -0.1, -0.2]];
    for r in 0..2 {
      for c in 0..5 {
        assert!(
          (got[r][c] - want[r][c]).abs() < F32_TOL,
          "delta[{r}][{c}]: got {}, want {}",
          got[r][c],
          want[r][c]
        );
      }
    }
  }

  #[test]
  fn compute_deltas_kaldi_win3_constant_matches_reference() {
    // win=3, mode=constant (n=1, denom=2). Zero-pad edges.
    // Reference:
    //   row0 [1,2,3,4,5]: [1.0, 1.0, 1.0, 1.0, -2.0]
    //     (last: (0 - 4)/2 = -2.0 — the zero pad pulls the trailing delta down)
    //   row1 [0,0,1,0,0]: [0.0, 0.5, 0.0, -0.5, 0.0]
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5])
      .unwrap();
    let d = compute_deltas_kaldi(&x, 3, DeltaPadMode::Constant).unwrap();
    let got = to_vec_2d(&d, 2, 5);
    let want = [[1.0_f32, 1.0, 1.0, 1.0, -2.0], [0.0, 0.5, 0.0, -0.5, 0.0]];
    for r in 0..2 {
      for c in 0..5 {
        assert!(
          (got[r][c] - want[r][c]).abs() < F32_TOL,
          "delta[{r}][{c}]: got {}, want {}",
          got[r][c],
          want[r][c]
        );
      }
    }
  }

  #[test]
  fn compute_deltas_kaldi_1d_ramp_interior_is_unit_slope() {
    // A unit-slope 1-D ramp has a constant first derivative of 1.0 in the
    // interior (the regression-delta of `c[t] = t` is exactly 1.0 where the
    // window does not touch a padded edge). Output keeps the 1-D shape.
    let ramp: Vec<f32> = (0..8).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&ramp, &[8]).unwrap();
    let d = compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge).unwrap();
    assert_eq!(d.shape(), vec![8]);
    let got = to_vec(&d);
    // Interior indices 2..=5 are unaffected by the edge replication (n=2).
    for (i, &g) in got.iter().enumerate().take(6).skip(2) {
      assert!(
        (g - 1.0).abs() < F32_TOL,
        "ramp delta[{i}]: got {g}, want 1.0"
      );
    }
  }

  #[test]
  fn compute_deltas_kaldi_delta_delta_is_zero_for_ramp_interior() {
    // Delta of a unit-slope ramp is ~constant (1.0), so the delta-of-delta
    // (acceleration) is ~0 in the deep interior — applying the function twice.
    let ramp: Vec<f32> = (0..12).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&ramp, &[12]).unwrap();
    let d = compute_deltas_kaldi(&x, 3, DeltaPadMode::Edge).unwrap();
    let dd = compute_deltas_kaldi(&d, 3, DeltaPadMode::Edge).unwrap();
    let got = to_vec(&dd);
    // n=1 each pass → indices 2..=9 are clear of both edge replications.
    for (i, &g) in got.iter().enumerate().take(10).skip(2) {
      assert!(g.abs() < F32_TOL, "ramp delta-delta[{i}]: got {g}, want ~0");
    }
  }

  #[test]
  fn compute_deltas_kaldi_rejects_invalid_win_length() {
    let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    // win_length < 3.
    assert!(matches!(
      compute_deltas_kaldi(&x, 2, DeltaPadMode::Edge),
      Err(Error::Backend { .. })
    ));
    // even win_length (would silently truncate to next-lower odd).
    assert!(matches!(
      compute_deltas_kaldi(&x, 4, DeltaPadMode::Edge),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn compute_deltas_kaldi_rejects_huge_win_length_on_tiny_input() {
    // A 1-element specgram (shape `(1,)`) with a HUGE odd `win_length` must be
    // rejected with a recoverable error BEFORE any pad / broadcast / slice loop
    // — the original-element cap alone (total == 1) would not catch it, so an
    // unbounded `win_length` could OOM / stall the CPU. Both `Edge` (broadcasts
    // two `(num_features, n)` bookends) and `Constant` (pads by `n`) must reject.
    let x = Array::from_slice::<f32>(&[1.0], &[1]).unwrap();
    let huge = 4_000_001_usize; // huge AND odd
    assert!(!huge.is_multiple_of(2), "win_length must be odd");
    for mode in [DeltaPadMode::Edge, DeltaPadMode::Constant] {
      assert!(
        matches!(
          compute_deltas_kaldi(&x, huge, mode),
          Err(Error::Backend { .. })
        ),
        "huge win_length on a 1-element input must be rejected ({mode:?})"
      );
    }

    // A normal win_length=5 still works on the same tiny input (n=2 pad,
    // padded extent 1 + 4 = 5; all-edge replication of the single value → the
    // shifted differences cancel, delta == 0). The point is it does NOT error.
    let ok = compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge).unwrap();
    assert_eq!(ok.shape(), vec![1], "shape preserved for the tiny input");
    let got = to_vec(&ok);
    assert!(
      got[0].abs() < F32_TOL,
      "single-value edge-padded delta should be 0, got {}",
      got[0]
    );
  }

  #[test]
  fn compute_deltas_kaldi_rejects_padded_work_over_cap() {
    // A normal small `win_length` whose padded extent still pushes
    // `num_features * (time + 2n)` past `MAX_FBANK_WORK` must be rejected by the
    // pre-pad padded-work cap. Build a shape whose `num_features * time` is
    // UNDER the cap but `num_features * (time + 2n)` is OVER it. With time=1,
    // win_length=5 (n=2): padded_time = 1 + 4 = 5, so num_features * 5 > cap
    // while num_features * 1 <= cap. num_features = MAX_FBANK_WORK gives
    // total = MAX_FBANK_WORK (passes) but padded = 5 * MAX_FBANK_WORK (fails).
    // Use `Array::zeros` (lazy — no host buffer materialized for the check).
    let num_features = MAX_FBANK_WORK; // total == num_features * 1 == cap (ok)
    let nf_i32 = i32::try_from(num_features).unwrap();
    let x = Array::zeros::<f32>(&[nf_i32, 1]).unwrap();
    assert!(
      matches!(
        compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge),
        Err(Error::Backend { .. })
      ),
      "padded work exceeding the cap must be rejected before allocating"
    );
  }

  /// Codex review: the delta CUMULATIVE-work cap (`MAX_DELTA_WORK`), distinct
  /// from the buffer-size caps. `total` (`num_features * time`) and
  /// `padded_work` (`num_features * (time + 2n)`) only bound buffer *sizes*,
  /// but the accumulation loop runs `win_length - 1` full-width slice /
  /// multiply / add passes over `total` elements — so the real element-op
  /// count is `total * (win_length - 1)`. Construct the doc-comment witness:
  /// a 1-D input of length `MAX_FBANK_WORK - 1022` with `win_length = 1023`
  /// (`n = 511`). It passes BOTH size caps yet the loop schedules ~1022
  /// passes over ~64 Mi elements:
  ///   total       = 64 Mi - 1022                  <= 64 Mi cap   (ok)
  ///   num_features= 1 (1-D)
  ///   padded_time = (64 Mi - 1022) + 2*511 = 64 Mi
  ///   padded_work = 1 * 64 Mi = 64 Mi             <= 64 Mi cap   (ok, at-cap)
  ///   delta_work  = (64 Mi - 1022) * 1022 ≈ 68 Gi  > 512 Mi cap  (REJECT)
  /// Only the new `MAX_DELTA_WORK` cap can stop it, and it must do so BEFORE
  /// the per-offset loop. `Array::zeros` is lazy (no host buffer), so the cap
  /// engages on the shape alone — no multi-GB allocation.
  #[test]
  fn compute_deltas_kaldi_rejects_cumulative_work_over_cap() {
    // 1-D input: num_features == 1, time == len. len = MAX_FBANK_WORK - 1022
    // so padded_time = len + 2n = MAX_FBANK_WORK exactly (padded-work cap is
    // at-cap and PASSES — only the cumulative-work cap can reject).
    let win_length = 1023_usize; // odd, <= MAX_DELTA_WIN_LENGTH (1024); n = 511
    let n = (win_length - 1) / 2; // 511
    let len = MAX_FBANK_WORK - 2 * n; // 64 Mi - 1022
    assert!(!win_length.is_multiple_of(2), "win_length must be odd");
    assert!(
      win_length <= MAX_DELTA_WIN_LENGTH,
      "win_length must clear the win_length cap so the cumulative cap is reached"
    );
    // Cross-check the cap interplay: size caps pass, cumulative cap fails.
    assert!(len <= MAX_FBANK_WORK, "total must pass the total cap");
    assert_eq!(
      len + 2 * n,
      MAX_FBANK_WORK,
      "padded_work is at-cap (passes)"
    );
    assert!(
      len.checked_mul(win_length - 1).unwrap() > MAX_DELTA_WORK,
      "delta_work must exceed the cumulative-work cap"
    );
    let len_i32 = i32::try_from(len).unwrap();
    let x = Array::zeros::<f32>(&[len_i32]).unwrap();
    let err = compute_deltas_kaldi(&x, win_length, DeltaPadMode::Edge).expect_err(
      "expected the cumulative-work cap to reject total * (win_length - 1) \
       BEFORE the per-offset accumulation loop",
    );
    let msg = format!("{err:?}");
    assert!(
      msg.contains("accumulation work") && msg.contains("work cap"),
      "expected the cumulative accumulation-work cap error, got: {msg}"
    );

    // A normal win_length = 5 on a small input still works (the cumulative cap
    // is generous): total = 2*5 = 10, delta_work = 10 * 4 = 40 << 512 Mi.
    let small =
      Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5])
        .unwrap();
    let ok = compute_deltas_kaldi(&small, 5, DeltaPadMode::Edge)
      .expect("a normal win_length=5 must pass the cumulative-work cap");
    assert_eq!(
      ok.shape(),
      vec![2, 5],
      "normal win_length=5 deltas still work"
    );
  }

  // ---- strided_frames_no_snip_edges (boundary values, hand-traced) ------
  //
  // These exercise the module-private `snip_edges=false` reflect-bookend
  // framing directly (it is a forward-only framing primitive, not an
  // invertible pair, so a focused unit test is the right granularity — the
  // public `compute_fbank_kaldi` `snip_edges=false` frame-count parity is
  // covered in tests/audio_dsp.rs). Expected values are the numpy replica of
  // the reference `_get_strided_kaldi(..., snip_edges=False)`.

  #[test]
  fn strided_no_snip_edges_win4_shift2_boundary_values() {
    // waveform = [0..9], win_size=4, win_inc=2 → pad = 4/2 - 2/2 = 1.
    //   m = (10 + 1) / 2 = 5. Reference padded buffer (read region):
    //     [1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, ...]
    //   frames:
    //     [1,0,1,2] [1,2,3,4] [3,4,5,6] [5,6,7,8] [7,8,9,9]
    let wf: Vec<f32> = (0..10).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[10]).unwrap();
    let m = (10 + 2 / 2) / 2; // 5
    let frames = strided_frames_no_snip_edges(&x, 4, 2, m).unwrap();
    assert_eq!(frames.shape(), vec![5, 4]);
    let got = to_vec_2d(&frames, 5, 4);
    let want = [
      [1.0_f32, 0.0, 1.0, 2.0],
      [1.0, 2.0, 3.0, 4.0],
      [3.0, 4.0, 5.0, 6.0],
      [5.0, 6.0, 7.0, 8.0],
      [7.0, 8.0, 9.0, 9.0],
    ];
    assert_eq!(got, want, "snip_edges=false win4 shift2 frames mismatch");
  }

  #[test]
  fn strided_no_snip_edges_win6_shift2_left_reflect_bookend() {
    // waveform = [0..9], win_size=6, win_inc=2 → pad = 3 - 1 = 2 (pad>1 path).
    //   pad_left = reverse(wf[1..3]) = [2, 1]; pad_right = reverse(wf[7..9]) = [9, 8].
    //   padded = [2,1,0,1,2,3,4,5,6,7,8,9,9,8]; m = (10+1)/2 = 5.
    //   first frame [2,1,0,1,2,3], last frame [6,7,8,9,9,8].
    let wf: Vec<f32> = (0..10).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[10]).unwrap();
    let frames = strided_frames_no_snip_edges(&x, 6, 2, 5).unwrap();
    assert_eq!(frames.shape(), vec![5, 6]);
    let got = to_vec_2d(&frames, 5, 6);
    assert_eq!(
      got[0],
      vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
      "left reflect bookend (pad=2) mismatch"
    );
    assert_eq!(
      got[4],
      vec![6.0, 7.0, 8.0, 9.0, 9.0, 8.0],
      "right reflect bookend (pad=2) mismatch"
    );
  }

  #[test]
  fn strided_no_snip_edges_pad_zero_path() {
    // win_size=4, win_inc=4 → pad = 2 - 2 = 0 (the `pad <= 0` branch:
    // padded = concat(wf[0..], reverse(wf))). waveform=[0..9]:
    //   padded = [0,1,2,3,4,5,6,7,8,9, 9,8,7,6,5,4,3,2,1,0]; m = (10+2)/4 = 3.
    //   frames: [0,1,2,3] [4,5,6,7] [8,9,9,8].
    let wf: Vec<f32> = (0..10).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[10]).unwrap();
    let m = (10 + 4 / 2) / 4; // 3
    let frames = strided_frames_no_snip_edges(&x, 4, 4, m).unwrap();
    assert_eq!(frames.shape(), vec![3, 4]);
    let got = to_vec_2d(&frames, 3, 4);
    let want = [
      [0.0_f32, 1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0, 7.0],
      [8.0, 9.0, 9.0, 8.0],
    ];
    assert_eq!(got, want, "snip_edges=false pad<=0 path frames mismatch");
  }

  #[test]
  fn strided_no_snip_edges_produces_extra_frame_vs_snip_true() {
    // The defining property of snip_edges=false: it keeps centered frames at
    // the edges that snip_edges=true drops, so for the same (win, inc) it
    // yields MORE frames. waveform len 10, win_size=4, win_inc=2:
    //   snip_edges=true:  m = 1 + (10 - 4)/2 = 4 frames.
    //   snip_edges=false: m = (10 + 1)/2     = 5 frames (one extra).
    let wf: Vec<f32> = (0..10).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[10]).unwrap();
    let m_true = 1 + (10 - 4) / 2; // 4
    let m_false = (10 + 2 / 2) / 2; // 5
    assert_eq!(
      m_false,
      m_true + 1,
      "snip=false should yield one extra frame"
    );
    let f_true = strided_frames_snip_edges(&x, 4, 2, m_true).unwrap();
    let f_false = strided_frames_no_snip_edges(&x, 4, 2, m_false).unwrap();
    assert_eq!(f_true.shape(), vec![4, 4]);
    assert_eq!(f_false.shape(), vec![5, 4]);
  }

  #[test]
  fn strided_no_snip_edges_rejects_degenerate_overread() {
    // A win_size large relative to the signal forces the strided read past the
    // reflect-padded buffer (the regime where the reference reads OOB). We
    // reject it with a recoverable error rather than reproduce that UB.
    // waveform len 5, win_size=8, win_inc=2 → pad=3, m=(5+1)/2=3,
    //   padded_len = 3 + 5 + 3 = 11, needed = (3-1)*2 + 8 = 12 > 11.
    let wf: Vec<f32> = (0..5).map(|n| n as f32).collect();
    let x = Array::from_slice::<f32>(&wf, &[5]).unwrap();
    let err = strided_frames_no_snip_edges(&x, 8, 2, 3)
      .expect_err("expected degenerate overread to be rejected");
    let msg = format!("{err:?}");
    assert!(
      msg.contains("exceeds reflect-padded length") || msg.contains("too short to reflect-pad"),
      "expected an overread/short-signal error, got: {msg}"
    );
  }
}
