//! DSP primitives: window family (Hann/Hamming/Blackman/Bartlett), STFT,
//! inverse STFT, mel filterbank, mel + log-mel spectrogram.
//!
//! Faithful 1:1 port of the corresponding `mlx_audio.dsp` core
//! (`hanning`, `hamming`, `blackman`, `bartlett`, `STR_TO_WINDOW_FN`, `stft`,
//! `istft`, `mel_filters`) at <https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py>.
//! Out of scope for this PR: the `ISTFTCache` batched/cached overlap-add
//! helper, Kaldi-style features, BS.1770 loudness, biquad filters, dither —
//! see [`crate::audio`] for the scope fence.
//!
//! ## API conventions
//! - Window construction is **symmetric** (`periodic=False` in `mlx-audio`):
//!   the first and last samples are zero. This matches scipy's
//!   `windows.hann(N, sym=True)` and the `mlx-audio` default for STFT. The
//!   string→window dispatch ([`window_from_name`]) mirrors `mlx-audio`'s
//!   `STR_TO_WINDOW_FN` table (`"hann"`/`"hanning"`/`"hamming"`/`"blackman"`/
//!   `"bartlett"`).
//! - STFT mirrors `mlx_audio.dsp.stft` defaults: `center=True`,
//!   `pad_mode="reflect"`. Output layout is **`(num_frames, n_fft / 2 + 1)`
//!   complex** (mlx-c `rfft` yields `Complex64` natively), as in the
//!   reference.
//! - [`istft`] inverts [`stft`] **in that same `(num_frames, n_fft / 2 + 1)`
//!   layout** (so `istft(&stft(x, ..)?, ..)` composes directly). This is a
//!   deliberate, semantics-preserving adaptation of `mlx_audio.dsp.istft`,
//!   which documents a frequency-major `(n_fft / 2 + 1, num_frames)` input
//!   and irffts along axis 0; see [`istft`] for the full rationale (the
//!   reference's `win_length` default is also derived from the frequency
//!   dimension here, fixing an axis bug in the upstream default formula).
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

/// Hard ceiling on [`istft`]'s overlap-add *work* — the number of
/// scatter/update elements `num_frames * frame_width` (`frame_width =
/// n_fft`). The OLA *output* length `t` is already capped at
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES), but with
/// small hops the scatter workload is orders of magnitude larger than `t`
/// (e.g. `num_frames=65536, n_fft=65536, hop=1` → `t≈131071` but the
/// scatter touches `4.29e9` indices). We therefore reject any
/// frame/window/hop combination whose real index count exceeds this cap
/// *before* allocating the index buffer (`try_reserve`) or building any
/// broadcast/flattened intermediate. 64 Mi-elements (256 MiB of `i32`
/// indices + matching f32 updates) is a generous ceiling that still admits
/// every realistic STFT round-trip while excluding pathological / lazily-
/// shaped inputs that would otherwise drive multi-GB allocation.
const MAX_OLA_WORK: usize = 64 * 1024 * 1024;

/// Coverage threshold for the overlap-add window-sum, shared by [`istft`]'s
/// normalization guard and its mandatory coverage check. A sample whose
/// window-sum is `<= COVERAGE_EPS` received negligible window energy in the
/// forward transform and cannot be recovered by dividing by that sum, so the
/// inverse would otherwise return a corrupt (un-normalized) value there.
/// Matches the reference's `mx.where(window_sum > 1e-10, ...)` literal.
const COVERAGE_EPS: f32 = 1e-10;

/// Placement of a short analysis/synthesis window (`win_length <= n_fft`)
/// inside the `n_fft`-wide frame, threaded identically through [`stft`] and
/// [`istft`] so the synthesis window always matches the analysis window.
///
/// When `win_length == n_fft` there is no padding and the two variants are
/// **identical**. They differ only for `win_length < n_fft`:
///
/// - [`WindowPad::Center`] (the default) places the window as
///   `[zeros(pad_low), w, zeros(pad_high)]` with `pad_low = (n_fft -
///   win_length) / 2` and `pad_high = n_fft - win_length - pad_low` — the
///   librosa `pad_center` convention. This gives full COLA coverage of the
///   centered output region, so [`istft`]'s coverage guard always passes and
///   the round-trip is exactly invertible.
/// - [`WindowPad::Right`] places the window as `[w, zeros(n_fft -
///   win_length)]` — the convention `mlx_audio.dsp` (and mlxrs's merged
///   [`stft`]) use. With `win_length <= n_fft / 2` this leaves a boundary
///   sample with **zero** window coverage; the forward transform discards it,
///   so it is not recoverable and [`istft`]'s coverage guard returns an
///   [`Error::Backend`] rather than silently emitting a corrupt sample.
///   `win_length > n_fft / 2` keeps full coverage and round-trips exactly.
///
/// `mlx-audio-swift` has no `win_length` (the window is always `n_fft`), so
/// it corresponds to `win_length == n_fft`, which both variants handle the
/// same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowPad {
  /// `[zeros, w, zeros]` centered in `n_fft` (librosa `pad_center`). The
  /// default: full COLA coverage, exactly invertible by [`istft`].
  #[default]
  Center,
  /// `[w, zeros]` right-padded in `n_fft` (mlx-audio / mlxrs `stft`). Only
  /// fully invertible when `win_length > n_fft / 2`; otherwise [`istft`]'s
  /// coverage guard fires (a boundary sample has no window coverage).
  Right,
}

/// Place a `win_length`-wide window `w` into an `n_fft`-wide frame per
/// [`WindowPad`], returning the `(n_fft,)` array. Shared by [`stft`] and
/// [`istft`] so analysis and synthesis windows are placed identically.
///
/// `win_length == n_fft` is a no-op (the window is already full width) for
/// both variants. For `win_length < n_fft`:
/// - [`WindowPad::Right`]  → pad `[0, n_fft - win_length]` (low, high).
/// - [`WindowPad::Center`] → pad `[(n_fft - win_length) / 2, rest]`.
///
/// `caller` only flavors the error message prefix.
///
/// # Errors
/// - [`Error::Backend`] if `w` is not 1-D, its length is not exactly
///   `win_length`, `win_length > n_fft`, or a pad extent exceeds `i32::MAX`.
fn place_window(
  caller: &str,
  w: &Array,
  win_length: usize,
  n_fft: usize,
  window_pad: WindowPad,
) -> Result<Array> {
  if w.ndim() != 1 {
    return Err(Error::Backend {
      message: format!("{caller}: window must be 1-D, got {}-D", w.ndim()),
    });
  }
  let w_len = w.shape()[0];
  if w_len != win_length {
    return Err(Error::Backend {
      message: format!("{caller}: window length {w_len} must equal win_length {win_length}"),
    });
  }
  if win_length > n_fft {
    return Err(Error::Backend {
      message: format!("{caller}: win_length {win_length} > n_fft {n_fft}"),
    });
  }
  if win_length == n_fft {
    return w.try_clone();
  }
  let total = n_fft - win_length;
  let (low, high) = match window_pad {
    WindowPad::Right => (0usize, total),
    WindowPad::Center => {
      let low = total / 2;
      (low, total - low)
    }
  };
  let pad_value = Array::zeros::<f32>(&[0i32; 0])?;
  let low_i32 = i32::try_from(low).map_err(|_| Error::Backend {
    message: format!("{caller}: window pad-low {low} exceeds i32::MAX"),
  })?;
  let high_i32 = i32::try_from(high).map_err(|_| Error::Backend {
    message: format!("{caller}: window pad-high {high} exceeds i32::MAX"),
  })?;
  ops::shape::pad(
    w,
    &[0_i32],
    &[low_i32],
    &[high_i32],
    &pad_value,
    c"constant",
  )
}

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

/// Shared scaffolding for the symmetric (`periodic=False`) window family:
/// validates `n`, applies the public-input allocation cap, and materializes
/// `[sample(k) for k in 0..n]` on the CPU via a recoverable
/// `try_reserve_exact`.
///
/// `name` only flavors the error messages so each public window keeps its
/// own diagnostic prefix; `sample` receives `(k, denom)` where
/// `denom = (n - 1) as f32` (the `periodic=False` denominator shared by
/// every `mlx-audio` window). The window kinds differ ONLY in this closure,
/// so the guards / cap / fallible allocation can't drift between them.
fn symmetric_window(name: &str, n: usize, sample: impl Fn(usize, f32) -> f32) -> Result<Array> {
  if n < 2 {
    return Err(Error::Backend {
      message: format!("{name}: n must be >= 2 (got {n})"),
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
        "{name}: n {n} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }
  let n_i32 = i32::try_from(n).map_err(|_| Error::Backend {
    message: format!("{name}: n {n} exceeds i32::MAX"),
  })?;

  // Materialize on the CPU (cheap; n is bounded above) via a
  // recoverable `try_reserve_exact` so the cap above (and any
  // future allocation budget) cannot abort the host on a fuzzer input.
  let denom = (n - 1) as f32;
  let mut buf: Vec<f32> = Vec::new();
  buf.try_reserve_exact(n).map_err(|e| Error::Backend {
    message: format!("{name}: reservation for {n} elements failed: {e}"),
  })?;
  for k in 0..n {
    buf.push(sample(k, denom));
  }
  Array::from_slice::<f32>(&buf, &[n_i32])
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
/// - Returns [`Error::Backend`] when `n` exceeds the
///   [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap or
///   `i32::MAX`, or if the backing allocation fails.
pub fn hann_window(n: usize) -> Result<Array> {
  symmetric_window("hann_window", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.5 * (1.0 - theta.cos())
  })
}

/// Symmetric Hamming window: `w[k] = 0.54 - 0.46 * cos(2π k / (n - 1))` for
/// `k in 0..n`. Endpoints are `0.08` (not zero, unlike Hann).
///
/// Matches `mlx_audio.dsp.hamming(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn hamming(n: usize) -> Result<Array> {
  symmetric_window("hamming", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.54 - 0.46 * theta.cos()
  })
}

/// Symmetric Blackman window:
/// `w[k] = 0.42 - 0.5 * cos(2π k / (n - 1)) + 0.08 * cos(4π k / (n - 1))`
/// for `k in 0..n`. Endpoints are zero (modulo f32 rounding ~`-1.4e-17`).
///
/// Matches `mlx_audio.dsp.blackman(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn blackman(n: usize) -> Result<Array> {
  symmetric_window("blackman", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos()
  })
}

/// Symmetric Bartlett (triangular) window:
/// `w[k] = 1 - 2 * |k - (n - 1) / 2| / (n - 1)` for `k in 0..n`. Rises
/// linearly to `1` at the center and back to `0` at both endpoints.
///
/// Matches `mlx_audio.dsp.bartlett(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn bartlett(n: usize) -> Result<Array> {
  symmetric_window("bartlett", n, |k, denom| {
    1.0 - 2.0 * (k as f32 - denom / 2.0).abs() / denom
  })
}

/// String → window dispatch, mirroring `mlx-audio`'s `STR_TO_WINDOW_FN`
/// table. The lookup is case-insensitive (matching the reference's
/// `window.lower()` in `stft`/`istft`):
/// - `"hann"` / `"hanning"` → [`hann_window`]
/// - `"hamming"` → [`hamming`]
/// - `"blackman"` → [`blackman`]
/// - `"bartlett"` → [`bartlett`]
///
/// All windows are the symmetric (`periodic=False`) form, as in `mlx-audio`.
///
/// # Errors
/// - [`Error::Backend`] for an unknown window name (mirrors the reference's
///   `ValueError(f"Unknown window function: {window}")`).
/// - Propagates the constructor errors of the selected window (see
///   [`hann_window`]).
pub fn window_from_name(name: &str, n: usize) -> Result<Array> {
  match name.to_ascii_lowercase().as_str() {
    "hann" | "hanning" => hann_window(n),
    "hamming" => hamming(n),
    "blackman" => blackman(n),
    "bartlett" => bartlett(n),
    other => Err(Error::Backend {
      message: format!("window_from_name: unknown window function: {other}"),
    }),
  }
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
/// up to `n_fft` per `window_pad` ([`WindowPad::Center`] — the librosa
/// `pad_center` convention and the default — or [`WindowPad::Right`] — the
/// `mlx_audio.dsp` convention). For `win_length == n_fft` the two are
/// identical (no padding). `win_length > n_fft` is rejected — the reference
/// would concatenate zeros, but a longer window than the FFT length cannot
/// occur in any documented `mlx-audio` config.
///
/// **Pair `istft` with the SAME `window_pad` and `win_length`** to invert:
/// the synthesis window must be placed identically to the analysis window.
/// [`WindowPad::Center`] is exactly invertible for any `win_length <= n_fft`;
/// [`WindowPad::Right`] is invertible only when `win_length > n_fft / 2`
/// (otherwise a boundary sample has no window coverage and [`istft`]'s
/// coverage guard errors).
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
  window_pad: WindowPad,
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

  // Window construction (hann of `win_length`, placed into the `n_fft` frame
  // per `window_pad`; no-op when `win_length == n_fft`).
  let window = hann_window(win_length)?;
  let window = place_window("stft", &window, win_length, n_fft, window_pad)?;

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

/// Synthesis window selector for [`istft`], the idiomatic translation of
/// `mlx_audio.dsp.istft`'s `window: mx.array | str` union:
/// - [`Window::Named`] resolves a `STR_TO_WINDOW_FN` name to the **periodic**
///   form the reference uses for synthesis (`window_fn(win_length + 1)`
///   truncated to `win_length` — i.e. the symmetric window of length
///   `win_length + 1` with its trailing duplicate sample dropped). This is
///   the COLA-friendly periodic window. The resulting `win_length`-wide
///   window is then placed into the `n_fft` frame per [`istft`]'s
///   `window_pad`.
/// - [`Window::Array`] supplies the synthesis window directly (the
///   reference's `else: w = window` branch). Its length must be exactly
///   `win_length`; it is then placed into the `n_fft` frame per
///   `window_pad`. Pass the SAME window [`stft`] used internally
///   ([`hann_window`] of `win_length`) with the SAME `window_pad` /
///   `win_length` and `normalized = true` for exact `istft(stft(x))`
///   reconstruction.
#[derive(Debug, Clone, Copy)]
pub enum Window<'a> {
  /// A `STR_TO_WINDOW_FN` name (case-insensitive). Built as the periodic
  /// window of length `win_length` (via the `win_length + 1` symmetric
  /// form, last sample dropped), then placed into `n_fft` per `window_pad`,
  /// matching `mlx_audio.dsp.istft`.
  Named(&'a str),
  /// A caller-supplied synthesis window array. Must be 1-D of length exactly
  /// `win_length`; it is then placed into the `n_fft` frame per `window_pad`
  /// (any other length is a recoverable [`Error::Backend`]).
  Array(&'a Array),
}

/// Inverse Short-Time Fourier Transform — overlap-add reconstruction, the
/// inverse of [`stft`].
///
/// Faithful port of `mlx_audio.dsp.istft(x, hop_length, win_length, window,
/// center=True, length=None, normalized=False)`, adapted to mlxrs's STFT
/// layout. **`x` is `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64`** — i.e.
/// exactly what [`stft`] returns — so `istft(&stft(s, ..)?, ..)` composes
/// directly. The reference instead documents a frequency-major
/// `(n_fft / 2 + 1, num_frames)` input and irffts along axis 0 then
/// transposes; here the frames are already on axis 0, so we irfft along
/// axis 1 and skip the transpose. This is a semantics-preserving adaptation,
/// not a behavior change (every sample of the reconstruction is identical to
/// the reference fed the transpose of `x`).
///
/// `n_fft` is an explicit `Option<usize>` defaulting to `(n_freqs - 1) * 2`
/// (the even-length irfft target). Passing it explicitly lets **odd** `n_fft`
/// round-trip (e.g. `n_fft = 9` ⇒ `n_freqs = 5`, where the default formula
/// would otherwise infer `8`); it is validated against the frequency axis
/// (`n_freqs == n_fft / 2 + 1`, a recoverable error otherwise) and used as
/// the irfft length. `win_length` defaults to `n_fft`. The reference's
/// documented default (`(n_fft - 1) * 2`) is computed as `(x.shape[1] - 1) *
/// 2`, which under its own documented frequency-major layout reads the
/// `num_frames` axis — an upstream axis bug; we derive from `n_freqs`
/// (`x.shape[1]` in our layout). `hop_length` defaults to `win_length / 4`.
///
/// **Both window-padding conventions are supported** via `window_pad`
/// ([`WindowPad`]), and `win_length <= n_fft` is allowed. The synthesis
/// window is built (Named periodic / explicit Array) at width `win_length`
/// and placed into the `n_fft` frame with the SAME `window_pad` as the
/// forward [`stft`]'s analysis window, then overlap-added full `n_fft` wide.
/// - [`WindowPad::Center`] (the default) is exactly invertible for every
///   `win_length <= n_fft`.
/// - [`WindowPad::Right`] is exactly invertible when `win_length > n_fft / 2`;
///   for `win_length <= n_fft / 2` a boundary sample has **no** window
///   coverage, so the coverage guard below returns a recoverable error
///   rather than emit a corrupt sample.
///
/// Normalization mirrors the reference: each output sample is divided by the
/// overlap-add sum of the (optionally squared) synthesis window. With
/// `normalized = false` the divisor is `Σ w` (simple window normalization);
/// with `normalized = true` it is `Σ w²` (COLA / `torch.istft` convention).
///
/// **Coverage guard (structural correctness invariant).** After the
/// overlap-add, every sample in the *requested output region* (the region
/// returned after the center-trim and `length` are applied) must have a
/// window-sum `> COVERAGE_EPS` (`1e-10`). If any requested sample's
/// window-sum is negligible, the un-normalized overlap-add value there is
/// meaningless, so [`istft`] returns a recoverable [`Error::Backend`] naming
/// the offending output index and parameters instead of silently emitting
/// that corrupt sample. [`WindowPad::Center`] always passes; [`WindowPad::Right`]
/// passes when coverage holds and errors cleanly otherwise. (The reference's
/// `mx.where(window_sum > 1e-10, ...)` merely leaves such samples
/// un-normalized — a silent corruption this guard converts into an error.)
///
/// Center / `length` ordering matches librosa / mlx-audio center semantics:
/// when `center = true` (the default), the `n_fft / 2` reflect-pad [`stft`]
/// added is removed FIRST (the centered signal begins at raw OLA index
/// `n_fft / 2`), and only then is `length` applied:
/// - `center = true,  length = None`    → the centered signal
///   `reconstructed[n_fft/2 .. t - n_fft/2]`.
/// - `center = true,  length = Some(n)` → `reconstructed[n_fft/2 .. n_fft/2 + n]`
///   (the first `n` real samples after dropping the reflected prefix).
/// - `center = false, length = Some(n)` → `reconstructed[0 .. n]`.
/// - `center = false, length = None`    → the full raw overlap-add.
///
/// Returns the reconstructed 1-D real signal (`Dtype::F32`).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `x` is not 2-D, or `n_freqs < 2` (need at least 2 bins to define
///     `n_fft`),
///   - an explicit `n_fft` is inconsistent with the frequency axis
///     (`n_freqs != n_fft / 2 + 1`),
///   - `num_frames == 0`,
///   - `hop_length == 0`, `win_length == 0`, or `win_length > n_fft`,
///   - an explicit [`Window::Array`] is not 1-D or its length is not exactly
///     `win_length`,
///   - **the coverage guard fires** — some requested output sample has a
///     window-sum `<= COVERAGE_EPS` (only possible with [`WindowPad::Right`]
///     and `win_length <= n_fft / 2`),
///   - any derived size overflows `usize`/`i32`, the OLA output length `t`
///     exceeds the
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap, or
///     the real scatter workload `num_frames * n_fft` exceeds an internal
///     work cap (`MAX_OLA_WORK`, checked before any allocation/broadcast —
///     guards against small-hop combinations whose work explodes far past
///     `t`),
///   - the `length` trim is out of range (with `center = true`,
///     `n_fft/2 + length > t`; with `center = false`, `length > t`).
/// - Propagates window-construction errors from [`window_from_name`].
#[allow(clippy::too_many_arguments)]
pub fn istft(
  x: &Array,
  n_fft: Option<usize>,
  hop_length: Option<usize>,
  win_length: Option<usize>,
  window: Window<'_>,
  window_pad: WindowPad,
  center: bool,
  length: Option<usize>,
  normalized: bool,
) -> Result<Array> {
  let shape = x.shape();
  if shape.len() != 2 {
    return Err(Error::Backend {
      message: format!(
        "istft: expected 2-D (num_frames, n_freqs) input, got {}-D",
        shape.len()
      ),
    });
  }
  let num_frames = shape[0];
  let n_freqs = shape[1];
  if n_freqs < 2 {
    return Err(Error::Backend {
      message: format!("istft: n_freqs {n_freqs} < 2 (need >= 2 bins for irfft)"),
    });
  }
  if num_frames == 0 {
    return Err(Error::Backend {
      message: "istft: num_frames must be > 0".into(),
    });
  }
  // The irfft target length. Defaults to `(n_freqs - 1) * 2` (the even
  // round-trip length), but may be passed explicitly so an ODD `n_fft`
  // round-trips (the default formula can only produce even values). Whichever
  // we use, it MUST satisfy `n_freqs == n_fft / 2 + 1` — the rfft/irfft bin
  // count for a length-`n_fft` real transform — else the irfft below would be
  // fed an inconsistent length. Validate the explicit value against the
  // frequency axis (recoverable error), and bound-check the default for
  // overflow.
  let n_fft = match n_fft {
    Some(n) => {
      if n == 0 {
        return Err(Error::Backend {
          message: "istft: n_fft must be > 0".into(),
        });
      }
      // `n / 2 + 1` cannot overflow (`n / 2 <= usize::MAX / 2`).
      if n_freqs != n / 2 + 1 {
        return Err(Error::Backend {
          message: format!(
            "istft: explicit n_fft {n} is inconsistent with n_freqs {n_freqs} \
             (require n_freqs == n_fft / 2 + 1 = {})",
            n / 2 + 1
          ),
        });
      }
      n
    }
    None => (n_freqs - 1).checked_mul(2).ok_or_else(|| Error::Backend {
      message: format!("istft: n_fft = (n_freqs - 1) * 2 overflows usize (n_freqs={n_freqs})"),
    })?,
  };
  let win_length = win_length.unwrap_or(n_fft);
  if win_length == 0 {
    return Err(Error::Backend {
      message: "istft: win_length must be > 0".into(),
    });
  }
  // The synthesis window (placed into the `n_fft` frame per `window_pad`)
  // cannot be wider than the irfft frame. `win_length < n_fft` IS supported
  // here (unlike earlier rounds): the synthesis window is placed identically
  // to the forward analysis window via `window_pad`, the overlap-add stays
  // full `n_fft` wide, and the coverage guard below converts any
  // zero-coverage boundary sample into a recoverable error rather than a
  // silent corruption.
  if win_length > n_fft {
    return Err(Error::Backend {
      message: format!(
        "istft: win_length {win_length} > n_fft {n_fft} (window cannot exceed the irfft frame)"
      ),
    });
  }
  let hop_length = hop_length.unwrap_or(win_length / 4);
  if hop_length == 0 {
    return Err(Error::Backend {
      message: "istft: hop_length must be > 0".into(),
    });
  }
  // Every frame is `n_fft` wide: the irfft output width AND the synthesis
  // window width (a `win_length`-wide window is placed into the `n_fft` frame
  // per `window_pad`, so the placed window is always exactly `n_fft` wide).
  // The overlap-add stride / frame width is therefore `n_fft`. Computed here,
  // before any window construction, so the OOM cap below precedes the
  // (potentially large) named-window allocation.
  let frame_width = n_fft;

  // Output / window-sum buffer length: `t = (num_frames - 1) * hop + n_fft`.
  let t = (num_frames - 1)
    .checked_mul(hop_length)
    .and_then(|v| v.checked_add(frame_width))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "istft: OLA length (num_frames-1)*hop + n_fft overflows usize \
         (num_frames={num_frames}, hop={hop_length}, n_fft={n_fft})"
      ),
    })?;
  if t > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "istft: OLA length {t} exceeds the {} cap",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }

  // OOM guard on the *real* scatter/update workload (`num_frames *
  // frame_width`), checked BEFORE any broadcast / flatten / `try_reserve`.
  // The `t` cap above bounds the *output* length, but with small hops the
  // scatter touches far more elements than `t` (e.g. num_frames=65536,
  // n_fft=65536, hop=1 → t≈131071 but idx_len≈4.29e9). Reject overflow,
  // `> i32::MAX`, and `> MAX_OLA_WORK` here so a shaped/lazy input can never
  // drive a multi-GB allocation downstream.
  let idx_len = num_frames
    .checked_mul(frame_width)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "istft: scatter work count num_frames * n_fft overflows usize \
         (num_frames={num_frames}, n_fft={n_fft})"
      ),
    })?;
  if idx_len > MAX_OLA_WORK {
    return Err(Error::Backend {
      message: format!(
        "istft: scatter work count {idx_len} (num_frames={num_frames} * n_fft={n_fft}) \
         exceeds the {MAX_OLA_WORK} work cap"
      ),
    });
  }
  let idx_len_i32 = i32::try_from(idx_len).map_err(|_| Error::Backend {
    message: format!("istft: scatter work count {idx_len} exceeds i32::MAX"),
  })?;

  let t_i32 = i32::try_from(t).map_err(|_| Error::Backend {
    message: format!("istft: OLA length {t} exceeds i32::MAX"),
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| Error::Backend {
    message: format!("istft: n_fft {n_fft} exceeds i32::MAX"),
  })?;

  // Synthesis window construction — built at width `win_length`, then placed
  // into the `n_fft` frame per `window_pad` (the SAME placement the forward
  // analysis window uses, so the inverse matches the forward exactly). Done
  // AFTER the OOM cap above so a pathological lazy huge-`n_fft` spectrum is
  // rejected before the named window (`window_from_name(win_length + 1)`, a
  // CPU `Vec` up to the work cap) is ever allocated. Named → periodic form
  // (symmetric window of `win_length + 1` with its trailing duplicate sample
  // dropped, the COLA-friendly periodic window the reference uses); Array →
  // used verbatim (length validated against `win_length` inside
  // `place_window`). The result is always exactly `n_fft` wide.
  let win_len_i32 = i32::try_from(win_length).map_err(|_| Error::Backend {
    message: format!("istft: win_length {win_length} exceeds i32::MAX"),
  })?;
  let base_window = match window {
    Window::Named(name) => {
      let win_p1 = win_length.checked_add(1).ok_or_else(|| Error::Backend {
        message: format!("istft: win_length {win_length} + 1 overflows usize"),
      })?;
      let full = window_from_name(name, win_p1)?;
      // Drop the trailing duplicate sample: full[0 .. win_length].
      ops::indexing::slice(&full, &[0], &[win_len_i32], &[1])?
    }
    Window::Array(w) => w.try_clone()?,
  };
  // Place the `win_length`-wide window into the `n_fft` frame per `window_pad`
  // (validates 1-D / `len == win_length` / `win_length <= n_fft`). No-op when
  // `win_length == n_fft`.
  let window = place_window("istft", &base_window, win_length, n_fft, window_pad)?;

  // Inverse FFT of every frame along the frequency axis (axis 1):
  // (num_frames, n_freqs) complex → (num_frames, n_fft) real. Frames are full
  // `n_fft` wide and the placed synthesis window is `n_fft` wide, so the
  // overlap-add multiply below is a straight element-wise product with no
  // slicing.
  let frames_time = fft::irfft(x, n_fft_i32, 1, FftNorm::Backward)?;

  // updates_reconstructed = (frames_time * w).flatten() — shape
  // (num_frames * n_fft,). `w` is (n_fft,) and broadcasts across the frame
  // axis.
  let windowed = ops::arithmetic::multiply(&frames_time, &window)?;
  let updates_reconstructed = ops::shape::flatten(&windowed, 0, -1)?;

  // window_norm = w*w if normalized else w; tiled across frames then flattened.
  let window_norm = if normalized {
    ops::arithmetic::multiply(&window, &window)?
  } else {
    window
  };
  // tile(window_norm, num_frames): (n_fft,) → (num_frames, n_fft).
  let window_norm_row = ops::shape::reshape(&window_norm, &(1usize, frame_width))?;
  let window_norm_tiled = ops::shape::broadcast_to(&window_norm_row, &(num_frames, frame_width))?;
  let updates_window = ops::shape::flatten(&window_norm_tiled, 0, -1)?;

  // Overlap-add destination indices:
  // indices[m, j] = m * hop + j, flattened to (num_frames * n_fft,).
  // Built CPU-side (bounded by the work cap above) as i32 — the reference
  // builds the same via arange broadcasts.
  let mut idx_buf: Vec<i32> = Vec::new();
  idx_buf
    .try_reserve_exact(idx_len)
    .map_err(|e| Error::Backend {
      message: format!("istft: index reservation for {idx_len} elements failed: {e}"),
    })?;
  let frame_width_i32 = i32::try_from(frame_width).map_err(|_| Error::Backend {
    message: format!("istft: n_fft {frame_width} exceeds i32::MAX"),
  })?;
  for m in 0..num_frames {
    // `m * hop_length < t <= i32::MAX` (t bounded above), and `+ j` stays
    // `< t`, so every index fits i32 without a per-element checked cast.
    let off = (m * hop_length) as i32;
    for j in 0..frame_width_i32 {
      idx_buf.push(off + j);
    }
  }
  let indices = Array::from_slice::<i32>(&idx_buf, &[idx_len_i32])?;

  // reconstructed / window_sum via scatter-add into zero buffers (axis 0).
  let zeros_recon = Array::zeros::<f32>(&[t_i32])?;
  let zeros_wsum = Array::zeros::<f32>(&[t_i32])?;
  let reconstructed =
    ops::indexing::scatter_add_axis(&zeros_recon, &indices, &updates_reconstructed, 0)?;
  let window_sum = ops::indexing::scatter_add_axis(&zeros_wsum, &indices, &updates_window, 0)?;

  // Requested-output region `[start_usize .. stop_usize)` within the raw OLA
  // buffer. The center reflect-pad `stft` added is `n_fft / 2` on EACH side
  // (`reflect_pad_1d(samples, n_fft / 2)`), so the centered signal begins at
  // raw OLA index `pad = n_fft / 2`, and the center pad must be removed BEFORE
  // `length` is applied (librosa / mlx-audio center semantics):
  //   * `center == true,  length = Some(n)` → `[pad .. pad + n]` (drop the
  //     reflected prefix, then keep `n` real samples).
  //   * `center == true,  length = None`    → `[pad .. t - pad]` (the centered
  //     signal; symmetric un-pad).
  //   * `center == false, length = Some(n)` → `[0 .. n]` (no pad was added).
  //   * `center == false, length = None`    → `[0 .. t]` (the full raw OLA).
  // Computing the bounds ONCE here lets the coverage guard and the final trim
  // operate on EXACTLY the same region, so the guard cannot disagree with what
  // is returned.
  let pad = n_fft / 2;
  let (start_usize, stop_usize) = match (center, length) {
    (true, Some(len)) => {
      let end = pad.checked_add(len).ok_or_else(|| Error::Backend {
        message: format!("istft: center offset {pad} + length {len} overflows usize"),
      })?;
      if end > t {
        return Err(Error::Backend {
          message: format!(
            "istft: center offset {pad} + length {len} = {end} exceeds reconstruction length {t}"
          ),
        });
      }
      (pad, end)
    }
    (true, None) => {
      // `t = (num_frames - 1) * hop + n_fft >= n_fft >= 2 * (n_fft / 2) =
      // 2 * pad`, so `t - pad >= pad` and the slice is non-empty / well-ordered.
      (pad, t - pad)
    }
    (false, Some(len)) => {
      if len > t {
        return Err(Error::Backend {
          message: format!("istft: requested length {len} exceeds reconstruction length {t}"),
        });
      }
      (0usize, len)
    }
    (false, None) => (0usize, t),
  };
  let start_i32 = i32::try_from(start_usize).map_err(|_| Error::Backend {
    message: format!("istft: trim start {start_usize} exceeds i32::MAX"),
  })?;
  let stop_i32 = i32::try_from(stop_usize).map_err(|_| Error::Backend {
    message: format!("istft: trim stop {stop_usize} exceeds i32::MAX"),
  })?;

  // COVERAGE GUARD (structural correctness invariant). Every sample in the
  // REQUESTED output region must have window-sum `> COVERAGE_EPS`; otherwise
  // its overlap-add value received negligible window energy and dividing by
  // that sum is meaningless — the reference's `mx.where` would silently emit
  // the un-normalized (corrupt) value there. We instead reduce the requested
  // slice of `window_sum` to its minimum (and the index of that minimum) and,
  // if the minimum is not strictly above the threshold, return a recoverable
  // error naming the offending GLOBAL output index and parameters. This is the
  // only place a scalar is read back (one explicit `eval` via `item`), and it
  // makes returning a corrupt sample structurally impossible. `WindowPad::Center`
  // always passes; `WindowPad::Right` passes iff coverage holds (e.g.
  // `win_length > n_fft / 2`).
  //
  // An empty requested region (possible only for a single centered even-`n_fft`
  // frame, where `[pad .. t - pad]` collapses) has no samples to corrupt, so
  // the guard is vacuously satisfied and the reduction (undefined over an empty
  // array) is skipped.
  if start_usize < stop_usize {
    let region_wsum = ops::indexing::slice(&window_sum, &[start_i32], &[stop_i32], &[1])?;
    let mut region_min = ops::reduction::min(&region_wsum, false)?;
    let min_wsum = region_min.item::<f32>()?;
    // Fire on `<= COVERAGE_EPS` AND on `NaN` (a NaN window-sum cannot normalize
    // either). Written explicitly rather than `!(min_wsum > eps)` for the
    // partial-ord lint, but with the same NaN-catching semantics.
    if min_wsum <= COVERAGE_EPS || min_wsum.is_nan() {
      let mut min_idx_arr = ops::misc::argmin(&region_wsum, None, false)?;
      let local_idx = min_idx_arr.item::<u32>()? as usize;
      let global_idx = start_usize + local_idx;
      return Err(Error::Backend {
        message: format!(
          "istft: requested output sample at index {global_idx} (region offset {local_idx}) \
           has window-sum {min_wsum:.3e} <= COVERAGE_EPS ({COVERAGE_EPS:.0e}) — it received \
           no window coverage in the forward transform and is not recoverable \
           (n_fft={n_fft}, win_length={win_length}, hop={hop_length}, window_pad={window_pad:?}, \
           normalized={normalized}); use WindowPad::Center or win_length > n_fft / 2"
        ),
      });
    }
  }

  // Normalize by the (squared) window-sum where it exceeds the coverage
  // threshold, else leave the raw overlap-add (matches the reference's
  // `mx.where` guard). The coverage guard above guarantees every REQUESTED
  // sample is on the normalized branch; the `where` only matters for the
  // trimmed-away region.
  let threshold = Array::full::<f32>(&[0i32; 0], COVERAGE_EPS)?;
  let mask = ops::comparison::greater(&window_sum, &threshold)?;
  let normalized_recon = ops::arithmetic::divide(&reconstructed, &window_sum)?;
  let reconstructed = ops::logical::select(&mask, &normalized_recon, &reconstructed)?;

  // Final trim to the requested region (same bounds the coverage guard used).
  ops::indexing::slice(&reconstructed, &[start_i32], &[stop_i32], &[1])
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
  // mel uses `win_length == n_fft` (the default), so `window_pad` is
  // immaterial here; pass the default `WindowPad::Center` for clarity.
  let spec = stft(samples, n_fft, hop_length, win_length, WindowPad::default())?;
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

#[cfg(test)]
mod tests {
  use super::*;

  /// Absolute tolerance for the closed-form window value checks. The
  /// formulas are evaluated in f32 here and in `mlx-audio` in f64 then cast
  /// to f32, so a few ULPs of slack is expected.
  const WIN_TOL: f32 = 1e-6;

  fn to_vec(a: &Array) -> Vec<f32> {
    // Tests own their arrays; clone so the accessor's `&mut self` (which
    // triggers the explicit eval) doesn't force a `mut` binding on callers.
    a.try_clone().unwrap().to_vec::<f32>().unwrap()
  }

  // ---- A2: window family closed-form parity (hand-derived) ----------------

  #[test]
  fn hamming_matches_closed_form_n5() {
    // 0.54 - 0.46 cos(2π k / 4) for k in 0..5:
    // k=0: 0.54-0.46 = 0.08; k=1: 0.54-0; wait cos(π/2)=0 → 0.54; k=2:
    // cos(π)=-1 → 1.0; k=3: 0.54; k=4: 0.08.
    let v = to_vec(&hamming(5).unwrap());
    let expected = [0.08_f32, 0.54, 1.0, 0.54, 0.08];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "hamming[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn hamming_endpoints_are_0_08() {
    // Distinguishing feature vs Hann: Hamming endpoints are 0.08, not 0.
    let v = to_vec(&hamming(8).unwrap());
    assert!((v[0] - 0.08).abs() < WIN_TOL, "first: {}", v[0]);
    assert!((v[7] - 0.08).abs() < WIN_TOL, "last: {}", v[7]);
  }

  #[test]
  fn blackman_matches_closed_form_n5() {
    // 0.42 - 0.5 cos(2π k/4) + 0.08 cos(4π k/4):
    // k=0: 0.42-0.5+0.08 = 0.0; k=1: 0.42-0+(-0.08)=0.34; k=2:
    // 0.42+0.5+0.08=1.0; k=3: 0.34; k=4: 0.0.
    let v = to_vec(&blackman(5).unwrap());
    let expected = [0.0_f32, 0.34, 1.0, 0.34, 0.0];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "blackman[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn bartlett_matches_closed_form_n5_and_n4() {
    // n=5 (odd): triangle peaking at 1.0 in the center, 0 at the ends.
    let v5 = to_vec(&bartlett(5).unwrap());
    let e5 = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
    for (i, (g, e)) in v5.iter().zip(e5.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett5[{i}]: got {g}, want {e}");
    }
    // n=4 (even): 1 - 2|k - 1.5|/3 → [0, 2/3, 2/3, 0].
    let v4 = to_vec(&bartlett(4).unwrap());
    let e4 = [0.0_f32, 2.0 / 3.0, 2.0 / 3.0, 0.0];
    for (i, (g, e)) in v4.iter().zip(e4.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett4[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn windows_reject_n_lt_2() {
    for r in [
      hamming(0),
      hamming(1),
      blackman(1),
      bartlett(0),
      bartlett(1),
    ] {
      assert!(matches!(r, Err(Error::Backend { .. })));
    }
  }

  #[test]
  fn window_from_name_dispatches_case_insensitively() {
    // "hann"/"hanning" → Hann (endpoints 0); "HAMMING" → Hamming
    // (endpoints 0.08); names are lowercased like the reference.
    let hann = to_vec(&window_from_name("HaNn", 8).unwrap());
    assert!(hann[0].abs() < WIN_TOL && hann[7].abs() < WIN_TOL);
    let hanning = to_vec(&window_from_name("hanning", 8).unwrap());
    assert_eq!(hann, hanning, "hann and hanning must be identical");
    let hamming = to_vec(&window_from_name("HAMMING", 8).unwrap());
    assert!((hamming[0] - 0.08).abs() < WIN_TOL);
    let bartlett = to_vec(&window_from_name("Bartlett", 5).unwrap());
    assert!((bartlett[2] - 1.0).abs() < WIN_TOL);
  }

  #[test]
  fn window_from_name_rejects_unknown() {
    assert!(matches!(
      window_from_name("kaiser", 8),
      Err(Error::Backend { .. })
    ));
  }

  // ---- A1: stft / istft WindowPad round-trips -----------------------------
  //
  // Every reconstruction test below asserts EVERY output sample value-for-
  // value against the ORIGINAL signal (no `.take`, no sub-range, no
  // "intrinsically zero" caveats). Expected values were cross-checked against
  // a self-contained f64 numpy mirror of stft/istft (`docs/istft_ref.py`,
  // local-only) implementing the same hann window, reflect-pad, OLA, and
  // window-sum normalization; that mirror reports max round-trip error
  // <= 4.5e-16 for every covered case here, so the f32 backend is asserted at
  // 1e-5. The coverage-guard test asserts the `Err` directly (it does NOT mask
  // the uncovered sample with a partial assertion).

  /// The 16-sample test signal used for the round-trips (arbitrary but fixed).
  fn signal_16() -> [f32; 16] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4,
    ]
  }

  /// A 19-sample fixed test signal for the non-hop-aligned / odd-`n_fft`
  /// round-trips.
  fn signal_19() -> [f32; 19] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4, 0.55,
      0.66, -0.77,
    ]
  }

  /// Round-trip `signal` through `stft`/`istft` with the SAME `win_length` and
  /// `window_pad`, feeding the matching symmetric-Hann analysis window via
  /// `Window::Array` with `normalized = true` (the exact `Σw²` inverse), and
  /// assert EVERY output sample equals the original. `len_override` is the
  /// `length` passed to `istft` (pass `Some(signal.len())` to recover the full
  /// input even when the centered region is shorter than the signal).
  fn assert_roundtrips_all_samples(
    signal: &[f32],
    n_fft: usize,
    win_length: usize,
    hop: usize,
    window_pad: WindowPad,
    len_override: Option<usize>,
  ) {
    let x = Array::from_slice::<f32>(signal, &[signal.len() as i32]).unwrap();
    let spec = stft(&x, n_fft, hop, Some(win_length), window_pad).unwrap();
    let w = hann_window(win_length).unwrap();
    let rec = istft(
      &spec,
      Some(n_fft), // explicit n_fft (required for odd n_fft to round-trip)
      Some(hop),
      Some(win_length),
      Window::Array(&w),
      window_pad,
      true, // center (undo stft's reflect pad)
      len_override,
      true, // normalized (Σw²)
    )
    .unwrap();
    let r = to_vec(&rec);
    let expected_len = len_override.unwrap_or(signal.len());
    assert_eq!(
      r.len(),
      expected_len,
      "round-trip length mismatch (n_fft={n_fft} win={win_length} hop={hop} {window_pad:?})"
    );
    // Assert ALL `expected_len` samples against the original signal.
    for (i, (g, e)) in r.iter().zip(signal.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "roundtrip[{i}] (n_fft={n_fft} win={win_length} hop={hop} {window_pad:?}): \
         got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  #[test]
  fn istft_win_eq_nfft_both_modes_identical_all_samples() {
    // win_length == n_fft ⇒ the two WindowPad variants place the window
    // identically (no padding), so BOTH must reconstruct every sample. n_fft=8,
    // hop=4 (50% overlap), 16 samples → centered region is exactly 16, so
    // length=None recovers all 16. Asserts all 16 samples for each mode.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    // Spectra are byte-identical across the two modes (no window padding).
    let spec_c = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let spec_r = stft(&x, 8, 4, Some(8), WindowPad::Right).unwrap();
    assert_eq!(spec_c.shape(), vec![5, 5]); // (num_frames, n_fft/2+1)
    for (c, r) in to_vec(&spec_c.abs().unwrap())
      .iter()
      .zip(to_vec(&spec_r.abs().unwrap()).iter())
    {
      assert!(
        (c - r).abs() < 1e-6,
        "win==nfft: spectra must match across modes"
      );
    }
    // length=None (centered region == 16) AND length=Some(16): both recover all.
    for mode in [WindowPad::Center, WindowPad::Right] {
      assert_roundtrips_all_samples(&buf, 8, 8, 4, mode, None);
      assert_roundtrips_all_samples(&buf, 8, 8, 4, mode, Some(16));
    }
  }

  #[test]
  fn istft_win_eq_nfft_non_hop_aligned_all_samples() {
    // Non-hop-aligned lengths (17, 19 are not multiples of hop=4): the centered
    // region is only 16 samples, so `length=None` would silently SHORTEN the
    // input — `length=Some(len)` recovers every sample (the center pad is
    // removed BEFORE length). Both modes (win==nfft ⇒ identical). Asserts ALL
    // `len` samples. Cross-checked vs numpy (max err 2.2e-16).
    for &len in &[17usize, 19usize] {
      let full = signal_19();
      let buf = &full[..len];
      for mode in [WindowPad::Center, WindowPad::Right] {
        assert_roundtrips_all_samples(buf, 8, 8, 4, mode, Some(len));
      }
    }
  }

  #[test]
  fn istft_center_short_window_all_samples() {
    // WindowPad::Center, win_length < n_fft: full COLA coverage, exactly
    // invertible. n_fft=16, hop=4, win=8 (min window-sum 0.41) and win=12 (min
    // window-sum 1.01) — both cover the centered 16-sample region, so
    // length=None recovers ALL 16 samples. Cross-checked vs numpy (max err
    // 1.1e-16). This is the correctness payoff of the Center convention: the
    // short-window inverse the Right convention cannot do safely.
    let buf = signal_16();
    for &win in &[8usize, 12usize] {
      assert_roundtrips_all_samples(&buf, 16, win, 4, WindowPad::Center, None);
    }
  }

  #[test]
  fn istft_right_large_window_all_samples() {
    // WindowPad::Right, win_length > n_fft / 2 (win=12 > 8 = 16/2): coverage
    // holds (min window-sum 0.68 over the centered region), so the Right
    // convention round-trips EXACTLY. n_fft=16, hop=4 → centered region 16,
    // length=None recovers ALL 16 samples. Cross-checked vs numpy (max err
    // 2.2e-16).
    let buf = signal_16();
    assert_roundtrips_all_samples(&buf, 16, 12, 4, WindowPad::Right, None);
  }

  #[test]
  fn istft_right_small_window_coverage_guard_errors() {
    // THE KEY FIX. WindowPad::Right with win_length <= n_fft / 2 (win=8 == 16/2)
    // places the SYNTHESIS window as `[hann(8), zeros(8)]`. We feed istft the
    // SAME symmetric Hann window the forward stft used (`Window::Array` +
    // normalized=true — the only configuration that WOULD reconstruct exactly
    // if coverage held). The symmetric Hann's last sample is 0, so the centered
    // output region's LAST sample has window-sum exactly 0 (numpy mirror:
    // wsum-region min == 0.0; that sample reconstructs to 0.0 instead of the
    // true -0.4 — a silent corruption). The coverage guard MUST turn this into
    // a recoverable Err rather than return that corrupt sample — asserted
    // DIRECTLY here (no partial / masked sample assertion), for both
    // length=None (the centered region) and length=Some(16) (both include the
    // uncovered last sample).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 16, 4, Some(8), WindowPad::Right).unwrap();
    assert_eq!(spec.shape(), vec![5, 9]); // (num_frames, n_fft/2+1), n_fft=16
    let w = hann_window(8).unwrap(); // SAME symmetric window stft used
    for len in [None, Some(16usize)] {
      let res = istft(
        &spec,
        Some(16),
        Some(4),
        Some(8), // win=8 <= n_fft/2=8 with Right → uncovered boundary sample
        Window::Array(&w),
        WindowPad::Right,
        true,
        len,
        true,
      );
      assert!(
        matches!(res, Err(Error::Backend { .. })),
        "Right + win=8 <= n_fft/2 (symmetric Array window, length={len:?}) must \
         hit the coverage guard, got {res:?}"
      );
    }
    // Contrast: the SAME short symmetric window (win=8 == n_fft/2) under
    // WindowPad::Center is fully covered and reconstructs EVERY sample — proving
    // it is the Right placement (not the short window per se) that triggers the
    // guard. (The Named periodic window is deliberately NOT probed here: its
    // nonzero endpoints give the boundary a small but positive window-sum, so
    // it numerically passes the guard while still NOT reconstructing — a
    // mismatched-window concern, distinct from the zero-coverage guard.)
    assert_roundtrips_all_samples(&buf, 16, 8, 4, WindowPad::Center, None);
  }

  #[test]
  fn istft_odd_nfft_center_all_samples() {
    // Odd n_fft must round-trip via the EXPLICIT `n_fft` argument (the default
    // `(n_freqs - 1) * 2` can only produce EVEN values). n_fft=9 (n_freqs=5,
    // hop=3) over signal_16 and n_fft=15 (n_freqs=8, hop=5) over signal_19,
    // Center mode. length=Some(signal.len()) recovers every sample. Cross-
    // checked vs numpy (max err 4.4e-16 / 2.2e-16).
    let s16 = signal_16();
    assert_roundtrips_all_samples(&s16, 9, 9, 3, WindowPad::Center, Some(16));
    let s19 = signal_19();
    assert_roundtrips_all_samples(&s19, 15, 15, 5, WindowPad::Center, Some(19));
  }

  #[test]
  fn istft_odd_nfft_validates_against_freq_axis() {
    // An explicit odd n_fft must satisfy `n_freqs == n_fft / 2 + 1`. A spec with
    // n_freqs=5 is consistent with n_fft ∈ {8, 9} (8/2+1 == 9/2+1 == 5) but NOT
    // with n_fft=11 (11/2+1 == 6) → must Err.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 9, 3, Some(9), WindowPad::Center).unwrap();
    assert_eq!(spec.shape()[1], 5); // n_freqs
    let w = hann_window(9).unwrap();
    let bad = istft(
      &spec,
      Some(11), // 11/2+1 = 6 != n_freqs (5)
      Some(3),
      Some(9),
      Window::Array(&w),
      WindowPad::Center,
      true,
      None,
      true,
    );
    assert!(
      matches!(bad, Err(Error::Backend { .. })),
      "explicit n_fft inconsistent with n_freqs must Err, got {bad:?}"
    );
  }

  #[test]
  fn istft_array_window_wrong_length_errors() {
    // Window::Array length must equal win_length (it is then placed per
    // window_pad). n_fft=8, win_length=8 but a length-6 array → Err.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let short = hann_window(6).unwrap(); // length 6 != win_length 8
    let res = istft(
      &spec,
      Some(8),
      Some(4),
      Some(8),
      Window::Array(&short),
      WindowPad::Center,
      true,
      None,
      true,
    );
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "Window::Array length != win_length must Err, got {res:?}"
    );
  }

  #[test]
  fn istft_center_length_removes_pad_before_truncating() {
    // With `center = true` and explicit `length`, the center reflect-pad
    // (`n_fft / 2 = 4`) is removed BEFORE the length cut, so the result is
    // `reconstructed[pad .. pad + length]` — the first `length` REAL samples —
    // NOT `reconstructed[0 .. length]` (which would start in the reflected
    // prefix). `assert_roundtrips_all_samples` with len_override=Some(10)
    // asserts all 10 returned samples equal the first 10 ORIGINAL samples; if
    // the pad were not removed first, element 0 would be the reflected prefix
    // and the value assertion would fail. (n_fft=8, hop=4, win=8.)
    let buf = signal_16();
    assert_roundtrips_all_samples(&buf, 8, 8, 4, WindowPad::Center, Some(10));
  }

  #[test]
  fn istft_center_false_uncovered_edge_errors() {
    // The coverage guard also protects the `center = false` path: the RAW OLA
    // index 0 is reached only by frame 0 at window position 0, and the
    // symmetric Hann window's first sample is 0, so OLA[0] has window-sum
    // exactly 0 (numpy mirror confirms `wsum[0] == 0`). Requesting that
    // un-centered head (which includes index 0) must therefore error rather
    // than return a corrupt sample — for both length=None (full raw OLA) and an
    // explicit length. (n_fft=8, hop=4, win=8 symmetric Hann.)
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let w = hann_window(8).unwrap();
    for len in [None, Some(10usize)] {
      let res = istft(
        &spec,
        Some(8),
        Some(4),
        Some(8),
        Window::Array(&w),
        WindowPad::Center,
        false, // center=false: requested region starts at the uncovered index 0
        len,
        true,
      );
      assert!(
        matches!(res, Err(Error::Backend { .. })),
        "center=false head (length={len:?}) includes the zero-coverage OLA \
         index 0 and must hit the coverage guard, got {res:?}"
      );
    }
  }

  #[test]
  fn istft_named_window_is_periodic_no_trailing_zero() {
    // Regression on the `window_fn(win_length + 1)[:-1]` periodic
    // construction: the synthesis window must have its trailing sample dropped
    // (so it is NOT the symmetric window with a zero at the end). periodic
    // hann(8) = hann(9)[:-1] = [0, .1464.., .5, .8535.., 1, .8535.., .5,
    // .1464..] — the LAST sample is 0.1464.., not 0.
    let full = hann_window(9).unwrap();
    let periodic = ops::indexing::slice(&full, &[0], &[8], &[1]).unwrap();
    let v = to_vec(&periodic);
    assert_eq!(v.len(), 8);
    assert!(
      v[0].abs() < WIN_TOL,
      "periodic[0] should be 0, got {}",
      v[0]
    );
    assert!(
      (v[7] - 0.146_447).abs() < 1e-4,
      "periodic[7] should be ~0.1464 (NOT 0), got {}",
      v[7]
    );
  }

  #[test]
  fn istft_rejects_bad_shapes_and_params() {
    // 1-D input (must be 2-D).
    let one_d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32]).unwrap();
    assert!(matches!(
      istft(
        &one_d,
        None,
        None,
        None,
        Window::Named("hann"),
        WindowPad::Center,
        true,
        None,
        false
      ),
      Err(Error::Backend { .. })
    ));

    // Valid spec for the remaining param checks.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let w = hann_window(8).unwrap();
    // hop_length == 0.
    assert!(matches!(
      istft(
        &spec,
        None,
        Some(0),
        Some(8),
        Window::Array(&w),
        WindowPad::Center,
        true,
        None,
        false
      ),
      Err(Error::Backend { .. })
    ));
    // win_length > n_fft (n_fft = 8 here; 16 > 8 → the window cannot exceed the
    // irfft frame width).
    assert!(matches!(
      istft(
        &spec,
        None,
        Some(4),
        Some(16),
        Window::Array(&w),
        WindowPad::Center,
        true,
        None,
        false
      ),
      Err(Error::Backend { .. })
    ));
    // length larger than the OLA length (t = 24).
    assert!(matches!(
      istft(
        &spec,
        None,
        Some(4),
        Some(8),
        Window::Array(&w),
        WindowPad::Center,
        true,
        Some(1000),
        true
      ),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn istft_rejects_pathological_scatter_work_before_window_alloc() {
    // Codex OOM finding (+ the medium "work cap runs after named-window alloc"
    // finding): the real scatter/update workload is `num_frames * n_fft`, which
    // can dwarf the OLA *output* length `t` for small hops. The
    // `t <= MAX_DECODED_SAMPLES` cap does NOT catch this; the dedicated
    // MAX_OLA_WORK guard must reject it BEFORE any window construction
    // (`window_from_name(win_length + 1)`, a CPU Vec up to the cap) and before
    // any broadcast/flatten/`try_reserve`/irfft.
    //
    // We use a LAZY mlx spectrum (`zeros(...).astype(Complex64)`) — nothing is
    // materialized — and `Window::Named("hann")` with the DEFAULT `win_length`
    // (= n_fft). If the cap ran after window construction, the Named path would
    // first allocate `hann_window(n_fft + 1)` ≈ 18 Mi f32s; because the cap
    // precedes window construction, that allocation never happens.
    //
    // num_frames=4, n_freqs=9 Mi+1 → n_fft=(n_freqs-1)*2=18 Mi, win_length=18 Mi.
    //   work = num_frames * n_fft = 4 * 18 Mi = 72 Mi  > MAX_OLA_WORK (64 Mi) ✓
    //   t    = (4-1)*hop + n_fft  = 6 + 18 Mi ≈ 18 Mi  < MAX_DECODED  (64 Mi)
    // so ONLY the work cap can reject this.
    let n_freqs: i32 = 9 * 1024 * 1024 + 1;
    let num_frames: i32 = 4;
    let spec = Array::zeros::<f32>(&[num_frames, n_freqs])
      .unwrap()
      .astype(crate::Dtype::Complex64)
      .unwrap();
    let res = istft(
      &spec,
      None,                  // n_fft defaults to (n_freqs-1)*2
      Some(2),               // small hop → t stays under the decoded cap
      None,                  // win_length defaults to n_fft
      Window::Named("hann"), // would alloc hann(win+1) AFTER the cap — never reached
      WindowPad::Center,
      true,
      None,
      false,
    );
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "pathological num_frames*n_fft must be rejected by the MAX_OLA_WORK cap \
       before the named-window allocation"
    );
  }
}
