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
//!   the first and last samples are zero for Hann/Blackman/Bartlett (Hamming
//!   is the exception — its endpoints are `0.08`, not zero). This matches
//!   scipy's `windows.hann(N, sym=True)` and the `mlx-audio` default for STFT.
//!   The string→window dispatch ([`window_from_name`]) mirrors `mlx-audio`'s
//!   `STR_TO_WINDOW_FN` table (`"hann"`/`"hanning"`/`"hamming"`/`"blackman"`/
//!   `"bartlett"`).
//! - STFT mirrors `mlx_audio.dsp.stft` defaults: `center=True`,
//!   `pad_mode="reflect"`. [`stft`] returns a typed [`Spectrum`] carrying the
//!   **`(num_frames, n_fft / 2 + 1)` complex** transform (mlx-c `rfft` yields
//!   `Complex64` natively, as in the reference) **plus the analysis metadata**
//!   (`n_fft`, `hop_length`, `win_length`, `window_pad`, `center`).
//! - [`istft`] inverts **even-`n_fft`** [`stft`] output by reading every
//!   parameter FROM the [`Spectrum`] (so `istft(&stft(x, ..)?, ..)` composes
//!   directly) — it **infers nothing**. This is a deliberate,
//!   semantics-preserving adaptation of `mlx_audio.dsp.istft`, which documents
//!   a frequency-major `(n_fft / 2 + 1, num_frames)` input and irffts along
//!   axis 0; here the frames are on axis 0 so we irfft along axis 1 (see
//!   [`istft`]). Carrying `n_fft` in the [`Spectrum`] makes the
//!   odd-vs-even-`n_fft` ambiguity structurally impossible: a one-sided
//!   spectrum cannot disambiguate odd `n_fft` from the adjacent even length
//!   from the bin count alone, so both [`stft`] and [`Spectrum::from_parts`]
//!   reject odd `n_fft` and a [`Spectrum`] can never carry it.
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

/// Hard ceiling on [`stft`]'s forward *work* — applied (with checked
/// arithmetic) to BOTH the strided-frame element count `num_frames * n_fft`
/// (the windowed-frame matrix the rfft consumes) AND the one-sided output
/// element count `num_frames * (n_fft / 2 + 1)`, BEFORE any frame view,
/// window multiply, or rfft is built. Mirrors [`MAX_OLA_WORK`] on the inverse
/// side: the public-input *sample* length is already capped at
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES), but a
/// *lazily-shaped* huge input (e.g. a 64 Mi-sample array with `n_fft = 1024,
/// hop = 1`) produces `num_frames ≈ 64 Mi` frames whose strided view is
/// `num_frames * n_fft ≈ 64 Gi` elements — orders of magnitude past the
/// sample count. We reject any `(num_frames, n_fft, hop)` combination whose
/// frame work or output element count overflows `usize` or exceeds this cap
/// *before* allocating, so a pathological / lazily-shaped input can never
/// drive a multi-GB framing/FFT allocation. The cap is intentionally in
/// elements, not bytes: at 64 Mi-elements the f32 frame matrix alone can be
/// ~256 MiB, while a `Dtype::Complex64` one-sided output of that size can be
/// ~512 MiB (plus any other intermediates). This is still a generous ceiling
/// that admits every realistic STFT.
const MAX_STFT_WORK: usize = 64 * 1024 * 1024;

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
/// - [`WindowPad::Right`] (the default) places the window right-aligned as
///   `[w, zeros(n_fft - win_length)]` — the convention `mlx_audio.dsp` (and
///   mlxrs's merged [`stft`]) use, so [`stft`]'s short-window output is
///   byte-identical to the reference. The forward [`stft`] supports it for any
///   `win_length`. **The inverse [`istft`], however, supports Right only for
///   `win_length == n_fft`**: right-pad short-window inversion is not a
///   faithful inverse (the forward transform discards / distorts boundary
///   information, so the reconstruction is wrong even where the window-sum is
///   nonzero), so [`istft`] rejects `win_length != n_fft` under Right with a
///   recoverable [`Error::Backend`]. Use [`WindowPad::Center`] for short-window
///   inversion.
/// - [`WindowPad::Center`] places the window as `[zeros(pad_low), w,
///   zeros(pad_high)]` with `pad_low = (n_fft - win_length) / 2` and
///   `pad_high = n_fft - win_length - pad_low` — the librosa `pad_center`
///   convention. This gives full COLA coverage of the centered output region,
///   so [`istft`]'s coverage guard always passes and the round-trip is exactly
///   invertible for **every** `win_length <= n_fft`. It is the placement
///   required for an invertible short-window round-trip.
///
/// The default is [`WindowPad::Right`] so the forward [`stft`] (and the
/// mel/log-mel front-ends built on it) stay byte-identical to `mlx_audio.dsp`
/// for short windows. Pass [`WindowPad::Center`] to [`stft`] when you need
/// `istft(&stft(x, .., Center)?, ..)?` to invert a short window — the
/// resulting [`Spectrum`] carries the placement, so [`istft`] re-applies it.
///
/// `mlx-audio-swift` has no `win_length` (the window is always `n_fft`), so
/// it corresponds to `win_length == n_fft`, which both variants handle the
/// same way (and both invert).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowPad {
  /// `[w, zeros]` right-padded in `n_fft` (mlx-audio / mlxrs `stft`). The
  /// default: keeps [`stft`]'s short-window output byte-identical to
  /// `mlx_audio.dsp`. Forward [`stft`] supports any `win_length`; the inverse
  /// [`istft`] supports Right only for `win_length == n_fft` (short-window
  /// right-pad inversion is not a faithful inverse and is rejected — use
  /// [`WindowPad::Center`]).
  #[default]
  Right,
  /// `[zeros, w, zeros]` centered in `n_fft` (librosa `pad_center`). Full COLA
  /// coverage, exactly invertible by [`istft`] for every `win_length <=
  /// n_fft`. Opt in to this for invertible short-window round-trips.
  Center,
}

/// A typed short-time spectrum: the `(num_frames, n_fft / 2 + 1)`
/// `Dtype::Complex64` transform data **plus all the metadata
/// [`istft`] needs to invert it exactly**.
///
/// This is the structural fix for the odd-vs-even-`n_fft` ambiguity that a
/// bare-array spectrum could not close. A one-sided real spectrum has
/// `n_freqs == n_fft / 2 + 1` for BOTH `n_fft = 2k` and `n_fft = 2k + 1`, so
/// the bin count alone cannot tell an odd `n_fft` from the adjacent even
/// length — any inverse that *infers* `n_fft` from a raw array can be made to
/// misdecode (an odd-`n_fft` external spectrum, in particular). By carrying
/// `n_fft` (and the rest of the analysis parameters) in the type, [`istft`]
/// reads them directly and **infers nothing**, and a `Spectrum` simply
/// *cannot exist* with inconsistent or odd metadata: every `Spectrum` is
/// produced either by [`stft`] (which rejects odd `n_fft`) or by the
/// validated [`Spectrum::from_parts`] constructor (which rejects odd `n_fft`,
/// a wrong bin count, `win_length > n_fft`, etc.). There is no way to hand
/// [`istft`] a raw array.
///
/// The fields are exactly the analysis parameters [`stft`] used, so the
/// inverse needs no inference:
/// - `data`: the `(num_frames, n_fft / 2 + 1)` `Complex64` transform.
/// - `n_fft`: the (even) FFT length — the irfft target width.
/// - `hop_length`: the analysis hop (overlap-add stride on the inverse).
/// - `win_length`: the analysis window length (`<= n_fft`).
/// - `window_pad`: the [`WindowPad`] placement of the `win_length` window in
///   the `n_fft` frame (so synthesis re-places it identically).
/// - `center`: whether [`stft`] reflect-padded the SIGNAL by `n_fft / 2` on
///   each side (`center = true, pad_mode = "reflect"`, the `mlx_audio.dsp`
///   default), which [`istft`] must undo before applying `length`.
///
/// (Not `Clone`: [`Array`] has only a fallible `try_clone`, so cloning a
/// [`Spectrum`] would have to be fallible too. Clone the `data` via
/// [`Array::try_clone`] and rebuild through [`Spectrum::from_parts`] if a copy
/// is needed.)
#[derive(Debug)]
pub struct Spectrum {
  /// `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64` transform data.
  data: Array,
  /// The (even) FFT length used by [`stft`]; the irfft target width.
  n_fft: usize,
  /// The analysis hop length (the inverse overlap-add stride).
  hop_length: usize,
  /// The analysis window length (`win_length <= n_fft`).
  win_length: usize,
  /// Placement of the `win_length` window inside the `n_fft` frame.
  window_pad: WindowPad,
  /// Whether [`stft`] reflect-padded the signal by `n_fft / 2` on each side.
  center: bool,
}

impl Spectrum {
  /// The `(num_frames, n_fft / 2 + 1)` `Dtype::Complex64` transform data.
  pub fn data(&self) -> &Array {
    &self.data
  }

  /// The (even) FFT length used to produce this spectrum (the irfft target
  /// width on the inverse).
  pub fn n_fft(&self) -> usize {
    self.n_fft
  }

  /// The analysis hop length (the overlap-add stride [`istft`] uses).
  pub fn hop_length(&self) -> usize {
    self.hop_length
  }

  /// The analysis window length (`win_length <= n_fft`).
  pub fn win_length(&self) -> usize {
    self.win_length
  }

  /// The [`WindowPad`] placement of the `win_length` window in the `n_fft`
  /// frame ([`istft`] re-places the synthesis window identically).
  pub fn window_pad(&self) -> WindowPad {
    self.window_pad
  }

  /// Whether [`stft`] reflect-padded the signal by `n_fft / 2` on each side
  /// (`center = true`). [`istft`] undoes this before applying `length`.
  pub fn center(&self) -> bool {
    self.center
  }

  /// The number of frames (`data`'s first dimension).
  pub fn num_frames(&self) -> usize {
    // `data` is validated 2-D at every construction site, so `shape()[0]`
    // is always present.
    self.data.shape()[0]
  }

  /// The number of one-sided frequency bins (`data`'s last dimension,
  /// `== n_fft / 2 + 1`).
  pub fn n_freqs(&self) -> usize {
    self.data.shape()[1]
  }

  /// Build a [`Spectrum`] from a raw `(num_frames, n_fft / 2 + 1)`
  /// `Complex64` array plus its analysis metadata, **validating that the
  /// metadata is self-consistent and invertible**.
  ///
  /// This is the only way to wrap an *external* / hand-built spectrum (one
  /// not produced by [`stft`]) for [`istft`], and it closes the
  /// external-odd-spectrum hole: the validation below makes it impossible to
  /// construct a `Spectrum` whose metadata [`istft`] would misdecode. Every
  /// check that [`stft`] enforces on the producer side is re-enforced here.
  ///
  /// # Errors
  /// Returns a recoverable [`Error::Backend`] when:
  /// - `data` is not 2-D (must be `(num_frames, n_freqs)`),
  /// - `n_fft == 0` or `n_fft` is **odd** (the one-sided spectrum cannot be
  ///   inverted unambiguously — see [`Spectrum`] / [`stft`]),
  /// - `data`'s last dimension `!= n_fft / 2 + 1` (the bin count does not
  ///   match the declared `n_fft`),
  /// - `hop_length == 0` or `win_length == 0`,
  /// - `win_length > n_fft` (the window cannot exceed the irfft frame),
  /// - `num_frames == 0`,
  /// - `data`'s dtype is not [`Dtype::Complex64`](crate::Dtype::Complex64).
  ///
  /// Note this does **not** itself reject [`WindowPad::Right`] with
  /// `win_length < n_fft`: such a spectrum is a perfectly valid forward
  /// transform (it is what [`stft`] emits for the mel front-end); it is only
  /// the *inverse* that is non-faithful, so [`istft`] rejects that
  /// combination when asked to invert (see [`istft`]).
  pub fn from_parts(
    data: Array,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window_pad: WindowPad,
    center: bool,
  ) -> Result<Spectrum> {
    let shape = data.shape();
    if shape.len() != 2 {
      return Err(Error::Backend {
        message: format!(
          "Spectrum::from_parts: data must be 2-D (num_frames, n_freqs), got {}-D",
          shape.len()
        ),
      });
    }
    if n_fft == 0 {
      return Err(Error::Backend {
        message: "Spectrum::from_parts: n_fft must be > 0".into(),
      });
    }
    // Reject odd `n_fft`: a one-sided spectrum has `n_freqs == n_fft / 2 + 1`
    // for both `n_fft = 2k` and `2k + 1`, so an odd-`n_fft` spectrum cannot be
    // inverted unambiguously. Closing it here means a `Spectrum` can never
    // carry odd metadata regardless of how it was constructed.
    if !n_fft.is_multiple_of(2) {
      return Err(Error::Backend {
        message: format!(
          "Spectrum::from_parts: n_fft must be even (got {n_fft}); odd n_fft is \
           unsupported because the one-sided spectrum cannot be inverted unambiguously"
        ),
      });
    }
    if hop_length == 0 {
      return Err(Error::Backend {
        message: "Spectrum::from_parts: hop_length must be > 0".into(),
      });
    }
    if win_length == 0 {
      return Err(Error::Backend {
        message: "Spectrum::from_parts: win_length must be > 0".into(),
      });
    }
    if win_length > n_fft {
      return Err(Error::Backend {
        message: format!(
          "Spectrum::from_parts: win_length {win_length} > n_fft {n_fft} \
           (the window cannot exceed the irfft frame)"
        ),
      });
    }
    let num_frames = shape[0];
    if num_frames == 0 {
      return Err(Error::Backend {
        message: "Spectrum::from_parts: num_frames must be > 0".into(),
      });
    }
    let n_freqs = shape[1];
    // `n_fft` is even, so `n_fft / 2 + 1` is the exact one-sided bin count.
    let expected_n_freqs = n_fft / 2 + 1;
    if n_freqs != expected_n_freqs {
      return Err(Error::Backend {
        message: format!(
          "Spectrum::from_parts: data last dim {n_freqs} != n_fft/2 + 1 = {expected_n_freqs} \
           (n_fft={n_fft}); the bin count does not match the declared n_fft"
        ),
      });
    }
    if data.dtype()? != crate::Dtype::Complex64 {
      return Err(Error::Backend {
        message: format!(
          "Spectrum::from_parts: data must be Dtype::Complex64, got {:?}",
          data.dtype()?
        ),
      });
    }
    Ok(Spectrum {
      data,
      n_fft,
      hop_length,
      win_length,
      window_pad,
      center,
    })
  }
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

/// The single source of truth for the `n_fft`-wide frame window shared by
/// [`stft`] (analysis) and [`istft`] (synthesis).
///
/// Builds the **symmetric** Hann window of length `win_length`
/// ([`hann_window`], `periodic=False`) and places it into the `n_fft`-wide
/// frame per `window_pad` ([`place_window`]). BOTH the forward and inverse
/// transforms construct their window through this one function with the same
/// `(win_length, n_fft, window_pad)` arguments, so the synthesis window is
/// **identical to the analysis window by construction** — a synthesis/analysis
/// mismatch (the historical source of silent round-trip corruption) is
/// structurally impossible. `istft` no longer takes any custom-window input
/// and `stft` has none to match against, so there is exactly one window per
/// `(win_length, n_fft, window_pad)` triple.
///
/// `caller` only flavors the error-message prefix (forwarded to
/// [`hann_window`] / [`place_window`]).
///
/// # Errors
/// Propagates [`hann_window`]'s constructor errors (`win_length < 2`, the
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap,
/// `i32::MAX`, allocation failure) and [`place_window`]'s placement errors
/// (`win_length > n_fft`, pad extent exceeding `i32::MAX`).
fn frame_window(win_length: usize, n_fft: usize, window_pad: WindowPad) -> Result<Array> {
  let w = hann_window(win_length)?;
  place_window("frame_window", &w, win_length, n_fft, window_pad)
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
pub fn hamming_window(n: usize) -> Result<Array> {
  symmetric_window("hamming_window", n, |k, denom| {
    let theta = 2.0 * PI * (k as f32) / denom;
    0.54 - 0.46 * theta.cos()
  })
}

/// Symmetric Blackman window:
/// `w[k] = 0.42 - 0.5 * cos(2π k / (n - 1)) + 0.08 * cos(4π k / (n - 1))`
/// for `k in 0..n`. Endpoints are zero (modulo f32 rounding of the
/// `0.42`/`0.5`/`0.08` literals, on the order of `1e-8`).
///
/// Matches `mlx_audio.dsp.blackman(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn blackman_window(n: usize) -> Result<Array> {
  symmetric_window("blackman_window", n, |k, denom| {
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
pub fn bartlett_window(n: usize) -> Result<Array> {
  symmetric_window("bartlett_window", n, |k, denom| {
    1.0 - 2.0 * (k as f32 - denom / 2.0).abs() / denom
  })
}

/// String → window dispatch, mirroring `mlx-audio`'s `STR_TO_WINDOW_FN`
/// table. The lookup is case-insensitive (matching the reference's
/// `window.lower()` in `stft`/`istft`):
/// - `"hann"` / `"hanning"` → [`hann_window`]
/// - `"hamming"` → [`hamming_window`]
/// - `"blackman"` → [`blackman_window`]
/// - `"bartlett"` → [`bartlett_window`]
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
    "hamming" => hamming_window(n),
    "blackman" => blackman_window(n),
    "bartlett" => bartlett_window(n),
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
/// window="hann", center=True, pad_mode="reflect")`. The analysis window is
/// built by the shared `frame_window` — the symmetric [`hann_window`] of
/// `win_length`, placed into the `n_fft` frame per `window_pad`.
/// When `win_length` (default = `n_fft`) is smaller than `n_fft`, the window
/// is zero-padded up to `n_fft` per `window_pad` ([`WindowPad::Right`] — the
/// `mlx_audio.dsp` convention and **the default**, so short-window output is
/// byte-identical to the reference — or [`WindowPad::Center`] — the librosa
/// `pad_center` convention, opt in for invertible short windows). For
/// `win_length == n_fft`, the two are identical (no padding). `win_length >
/// n_fft` is rejected — the reference would concatenate zeros, but a longer
/// window than the FFT length cannot occur in any documented `mlx-audio`
/// config.
///
/// **Pair `istft` with the SAME `window_pad` and `win_length`** to invert:
/// `istft` rebuilds the analysis window through the very same `frame_window`
/// call, so the synthesis window is identical to the analysis window by
/// construction (a mismatch is structurally impossible).
/// [`WindowPad::Center`] is exactly invertible for any `win_length <= n_fft`.
/// [`WindowPad::Right`] is invertible by [`istft`] only when
/// `win_length == n_fft`; short-window Right inversion is not faithful and
/// [`istft`] rejects it (use [`WindowPad::Center`] for `win_length < n_fft`).
/// This forward `stft` itself supports Right padding for any `win_length`.
///
/// Returns a typed [`Spectrum`] carrying the `(num_frames, n_fft / 2 + 1)`
/// `Dtype::Complex64` transform (`num_frames = 1 + (padded_len - n_fft) /
/// hop_length`, the reference layout) **plus all the metadata [`istft`] needs
/// to invert it** — `n_fft`, `hop_length`, `win_length`, `window_pad`, and the
/// signal-centering flag (always `center = true` here: `stft` hardcodes
/// `center = true, pad_mode = "reflect"`, the `mlx_audio.dsp` default). Because
/// the metadata travels in the type, [`istft`] reads it directly and infers
/// nothing — the odd-vs-even-`n_fft` ambiguity is structurally gone.
///
/// **Work cap.** Before building the strided frame view, the window multiply,
/// or the rfft, `stft` computes `num_frames`, the frame element count
/// `num_frames * n_fft`, and the output element count `num_frames * (n_fft/2 +
/// 1)` with checked arithmetic and rejects (recoverable [`Error::Backend`]) any
/// combination that overflows or exceeds `MAX_STFT_WORK`. A lazily-shaped huge
/// input (e.g. 64 Mi samples with `n_fft = 1024, hop = 1`) is rejected before
/// any allocation, so it cannot drive a multi-GB framing/FFT allocation.
///
/// **Input-length cap.** The reflect pad (`center = true`) is a lazy
/// slice+concatenate, but *evaluating* it materializes a signal proportional to
/// the INPUT length — independent of `num_frames`. Because the `MAX_STFT_WORK`
/// cap only bounds `num_frames * n_fft`, a lazily-shaped huge input with a LARGE
/// `hop_length` (few frames) would slip past it while the reflect-pad
/// concatenate still ballooned. So BEFORE the reflect pad, `stft` rejects any
/// input whose sample count — or padded length `samples_len + n_fft` (checked
/// arithmetic) — exceeds the per-call
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) budget,
/// bounding the reflect-pad allocation regardless of hop.
///
/// **`n_fft` must be even.** The one-sided spectrum has
/// `n_freqs == n_fft / 2 + 1` for BOTH `n_fft = 2k` and `n_fft = 2k + 1`, so
/// the bin count alone cannot disambiguate an odd `n_fft` from the adjacent
/// even length. Although [`istft`] now reads `n_fft` from the [`Spectrum`]
/// (rather than inferring it), keeping the producer even-only means a
/// [`Spectrum`] can never carry an odd `n_fft` at all, so this forward `stft`
/// rejects odd `n_fft` up front rather than emitting an un-invertible spectrum
/// ([`Spectrum::from_parts`] enforces the same on external spectra).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `samples` is not 1-D,
///   - `n_fft == 0`, `hop_length == 0`, or `win_length == 0`,
///   - `n_fft` is odd (the one-sided spectrum cannot be inverted
///     unambiguously; see above and [`istft`]),
///   - `win_length > n_fft`,
///   - the input sample count, or the padded length `samples_len + n_fft`,
///     exceeds the [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES)
///     budget (a lazily-shaped huge input is rejected before the reflect pad,
///     regardless of `hop_length`),
///   - the post-pad sample count is too short to fit a single frame
///     (matches the reference's `Input is too short` raise),
///   - the frame work `num_frames * n_fft` or output element count
///     `num_frames * (n_fft / 2 + 1)` overflows `usize` or exceeds
///     `MAX_STFT_WORK` (a lazily-shaped huge input is rejected before any
///     allocation),
///   - any size exceeds `i32::MAX`.
pub fn stft(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  window_pad: WindowPad,
) -> Result<Spectrum> {
  if n_fft == 0 {
    return Err(Error::Backend {
      message: "stft: n_fft must be > 0".into(),
    });
  }
  // Reject odd `n_fft` up front (before any framing/FFT work): a one-sided
  // real-FFT spectrum has `n_freqs == n_fft / 2 + 1` for both `n_fft = 2k` and
  // `2k + 1`, so the bin count alone cannot disambiguate odd from the adjacent
  // even length. Keeping the producer even-only means the `Spectrum` this
  // returns can never carry an odd `n_fft`, so its inverse is unambiguous.
  if !n_fft.is_multiple_of(2) {
    return Err(Error::Backend {
      message: format!(
        "stft: n_fft must be even (got {n_fft}); odd n_fft is unsupported \
         because the one-sided spectrum cannot be inverted unambiguously."
      ),
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

  // INPUT-LENGTH CAP (Codex OOM finding). The reflect pad below
  // (`reflect_pad_1d`) is a lazy slice+concatenate, but *evaluating* the graph
  // materializes a padded signal proportional to the INPUT length — independent
  // of `num_frames`. The post-framing `MAX_STFT_WORK` cap bounds
  // `num_frames * n_fft`, but a lazily-shaped huge 1-D input with a LARGE
  // `hop_length` yields few frames, so `frame_work` stays under that cap while
  // the reflect-pad concatenate still balloons proportional to the input. We
  // therefore reject any input whose sample count — or padded length
  // `samples_len + n_fft` (reflect pad adds `n_fft / 2` on each side) — exceeds
  // the per-call sample budget [`MAX_DECODED_SAMPLES`] BEFORE building the
  // padded signal or any frame view, bounding the reflect-pad allocation
  // regardless of hop. Checked arithmetic so the `+ n_fft` itself can't wrap.
  let samples_len = shape[0];
  if samples_len > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "stft: input sample count {samples_len} exceeds the {} sample budget \
         (would force a reflect-pad allocation proportional to the input)",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }
  let padded_len_budget = samples_len
    .checked_add(n_fft)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "stft: padded length samples_len + n_fft overflows usize \
         (samples_len={samples_len}, n_fft={n_fft})"
      ),
    })?;
  if padded_len_budget > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "stft: padded length {padded_len_budget} (samples_len={samples_len} + \
         n_fft={n_fft}) exceeds the {} sample budget (would force a reflect-pad \
         allocation proportional to the input)",
        crate::audio::io::MAX_DECODED_SAMPLES
      ),
    });
  }

  // Analysis window via the SHARED `frame_window` (symmetric hann of
  // `win_length`, placed into the `n_fft` frame per `window_pad`; no-op when
  // `win_length == n_fft`). Built AFTER the work cap below so a lazily-shaped
  // huge input is rejected before this CPU `Vec` (up to `win_length <= n_fft`
  // elements) is allocated. `istft` rebuilds its synthesis window through the
  // exact same call, so analysis and synthesis windows always match.

  // `center=True, pad_mode="reflect"` (reference default). The reflect pad is a
  // lazy slice+concatenate, but evaluating it materializes a signal
  // proportional to the input length, so the input/padded-length cap above
  // gates it; the post-framing `MAX_STFT_WORK` cap then gates the strided view /
  // window / rfft (frame work + FFT output).
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

  // WORK CAP (Codex finding). Mirror `istft`'s `MAX_OLA_WORK` guard on the
  // forward side: BEFORE building the strided frame view, the window, or the
  // rfft, reject any `(num_frames, n_fft, hop)` whose framing work or output
  // size is pathological. The public-input *sample* length is already capped
  // (`MAX_DECODED_SAMPLES`), but a LAZILY-shaped huge input (e.g. 64 Mi
  // samples, n_fft=1024, hop=1) yields `num_frames ≈ 64 Mi` frames and a
  // strided view of `num_frames * n_fft ≈ 64 Gi` elements — far past the
  // sample count. Both the frame element count `num_frames * n_fft` (the
  // windowed matrix the rfft consumes) and the one-sided output element count
  // `num_frames * (n_fft / 2 + 1)` are checked here so neither the framing
  // intermediate nor the FFT output can balloon past `MAX_STFT_WORK`. Checked
  // arithmetic + the cap precede every allocation below (including the
  // `frame_window` CPU `Vec`), so a shaped/lazy input never reaches them.
  let frame_work = num_frames
    .checked_mul(n_fft)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "stft: frame work count num_frames * n_fft overflows usize \
         (num_frames={num_frames}, n_fft={n_fft})"
      ),
    })?;
  if frame_work > MAX_STFT_WORK {
    return Err(Error::Backend {
      message: format!(
        "stft: frame work count {frame_work} (num_frames={num_frames} * n_fft={n_fft}) \
         exceeds the {MAX_STFT_WORK} work cap"
      ),
    });
  }
  // `n_fft` is even (checked above), so `n_fft / 2 + 1` cannot overflow.
  let out_elems = num_frames
    .checked_mul(n_fft / 2 + 1)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "stft: output element count num_frames * (n_fft/2 + 1) overflows usize \
         (num_frames={num_frames}, n_fft={n_fft})"
      ),
    })?;
  if out_elems > MAX_STFT_WORK {
    return Err(Error::Backend {
      message: format!(
        "stft: output element count {out_elems} (num_frames={num_frames} * \
         (n_fft/2 + 1)={}) exceeds the {MAX_STFT_WORK} work cap",
        n_fft / 2 + 1
      ),
    });
  }

  // Analysis window via the SHARED `frame_window` (symmetric hann of
  // `win_length`, placed into the `n_fft` frame per `window_pad`; no-op when
  // `win_length == n_fft`). Built AFTER the work cap so a lazily-shaped huge
  // input is rejected before this CPU `Vec` is allocated. `istft` rebuilds its
  // synthesis window through the EXACT same call, so analysis and synthesis
  // windows always match by construction.
  let window = frame_window(win_length, n_fft, window_pad)?;

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
  let data = fft::rfft(&windowed, n_fft_i32, 1, FftNorm::Backward)?;

  // Wrap the `(num_frames, n_fft / 2 + 1)` Complex64 transform together with
  // the analysis metadata `istft` needs to invert it exactly. `center` is
  // always `true` here (`stft` hardcodes `center = true, pad_mode = "reflect"`,
  // the `mlx_audio.dsp` default). All invariants `Spectrum::from_parts` would
  // re-check (even n_fft, `n_freqs == n_fft / 2 + 1`, `win_length <= n_fft`,
  // non-empty) hold by construction, so the data is wrapped directly.
  Ok(Spectrum {
    data,
    n_fft,
    hop_length,
    win_length,
    window_pad,
    center: true,
  })
}

/// Inverse Short-Time Fourier Transform — overlap-add reconstruction, the
/// inverse of [`stft`].
///
/// Faithful port of `mlx_audio.dsp.istft(x, hop_length, win_length, window,
/// center=True, length=None)`, adapted to mlxrs's STFT layout and restricted
/// to **even `n_fft`** (the universal case). The input is a typed [`Spectrum`]
/// — exactly what [`stft`] returns (or a validated [`Spectrum::from_parts`]) —
/// so `istft(&stft(s, ..)?, ..)` composes directly.
///
/// **All transform parameters are read FROM the [`Spectrum`] — nothing is
/// inferred.** `n_fft`, `hop_length`, `win_length`, `window_pad`, and the
/// signal-centering flag (`center`) all come straight off the [`Spectrum`]
/// (`spectrum.n_fft()`, …). There is **no** `n_fft`/`hop`/`win`/`pad`/`center`
/// parameter to mis-state, and crucially **no `n_fft` inference**: a
/// one-sided spectrum has `n_freqs == n_fft / 2 + 1` for BOTH `n_fft = 2k` and
/// `2k + 1`, so inferring `n_fft` from the bin count could misdecode an odd
/// transform — but [`Spectrum`] carries the exact even `n_fft` its producer
/// used, and a `Spectrum` cannot exist with odd `n_fft` (both [`stft`] and
/// [`Spectrum::from_parts`] reject it). The odd-vs-even ambiguity is therefore
/// **structurally impossible**, not merely guarded.
///
/// **There is no custom-window parameter.** The synthesis window is rebuilt
/// internally from the [`Spectrum`]'s `(win_length, n_fft, window_pad)`
/// through the very same `frame_window` the forward [`stft`] used for its
/// analysis window, so the synthesis window is **identical to the analysis
/// window by construction** — the historical synthesis/analysis mismatch (a
/// separately-specified periodic synthesis window silently differing from
/// `stft`'s symmetric analysis window) is structurally impossible. The
/// reference instead documents a frequency-major `(n_fft / 2 + 1, num_frames)`
/// input and irffts along axis 0 then transposes; here the frames are already
/// on axis 0, so we irfft along axis 1 and skip the transpose. This is a
/// semantics-preserving adaptation, not a behavior change (every sample of the
/// reconstruction is identical to the reference fed the transpose of the data).
///
/// The synthesis window is the symmetric Hann of the [`Spectrum`]'s
/// `win_length` placed into the `n_fft` frame with its `window_pad` (both via
/// `frame_window`), then overlap-added full `n_fft` wide.
/// - [`WindowPad::Center`] is exactly invertible for every `win_length <=
///   n_fft` — full short-window support.
/// - [`WindowPad::Right`] (the [`WindowPad`] default) is supported **only for
///   `win_length == n_fft`**. Right-pad short-window inversion (`win_length <
///   n_fft`) is not a faithful inverse: the forward transform discards /
///   distorts boundary information, so the reconstruction is wrong even where
///   the window-sum is nonzero. [`istft`] therefore **rejects** a [`Spectrum`]
///   whose `window_pad` is Right with `win_length != n_fft` up front with a
///   recoverable [`Error::Backend`] (such a [`Spectrum`] is a valid forward
///   transform — it is what the mel front-end produces — only its *inverse* is
///   non-faithful). Use [`WindowPad::Center`] for short-window inversion.
///
/// **Overlap-add normalization is always `Σ w²`** (the window-sum of
/// *squares*). [`stft`] emits FFTs of already-windowed frames, and [`istft`]
/// irffts and multiplies by the synthesis window *again*, so each output
/// sample carries a `w²` weight and the faithful inverse divides by the
/// overlap-add sum of `w²` (the COLA / `torch.istft` convention). There is no
/// `normalized` toggle: dividing by `Σ w` (the upstream `normalized=False`
/// branch) is a gain error against this windowed-twice forward transform, so
/// that path is removed and `Σ w²` is always used.
///
/// **Coverage guard (structural correctness invariant).** After the
/// overlap-add, every sample in the *requested output region* (the region
/// returned after the center-trim and `length` are applied) must have a
/// window-sum `> COVERAGE_EPS` (`1e-10`). If any requested sample's
/// window-sum is negligible, the un-normalized overlap-add value there is
/// meaningless, so [`istft`] returns a recoverable [`Error::Backend`] naming
/// the offending output index and parameters instead of silently emitting
/// that corrupt sample. For supported configurations ([`WindowPad::Center`]
/// any `win_length`, or [`WindowPad::Right`] with `win_length == n_fft`) the
/// guard should never fire — if it does, that is a real reconstruction bug,
/// not a masked one. (The reference's `mx.where(window_sum > 1e-10, ...)`
/// merely leaves such samples un-normalized — a silent corruption this guard
/// converts into an error. Right-pad short-window configs, where the guard is
/// insufficient to catch covered-but-wrong samples, are rejected up front.)
///
/// Center / `length` ordering matches librosa / mlx-audio center semantics.
/// `center` is read from the [`Spectrum`] (always `true` for an [`stft`]
/// output, since `stft` hardcodes `center = true`). When `center = true`, the
/// `n_fft / 2` reflect-pad [`stft`] added is removed FIRST (the centered
/// signal begins at raw OLA index `n_fft / 2`), and only then is `length`
/// applied. `length` (the desired output length) is the **only** inverse-side
/// parameter:
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
///   - the [`Spectrum`]'s `window_pad` is [`WindowPad::Right`] and
///     `win_length != n_fft` (short-window right-pad inversion is not a
///     faithful inverse — rejected up front; use [`WindowPad::Center`]),
///   - **the coverage guard fires** — some requested output sample has a
///     window-sum `<= COVERAGE_EPS` (a supported config should never trigger
///     this; if it does it is a real reconstruction bug),
///   - any derived size overflows `usize`/`i32`, the OLA output length `t`
///     exceeds the
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap, or
///     the real scatter workload `num_frames * n_fft` exceeds an internal
///     work cap (`MAX_OLA_WORK`, checked before any allocation/broadcast —
///     guards against small-hop combinations whose work explodes far past
///     `t`),
///   - the `length` trim is out of range (with `center = true`,
///     `n_fft/2 + length > t`; with `center = false`, `length > t`).
/// - Propagates window-construction errors from `frame_window` (the shared
///   symmetric-Hann builder).
///
/// (The [`Spectrum`]'s structural invariants — 2-D `Complex64` data, even
/// `n_fft`, `n_freqs == n_fft / 2 + 1`, `1 <= win_length <= n_fft`,
/// `hop_length >= 1`, `num_frames >= 1` — are guaranteed at construction by
/// [`stft`] / [`Spectrum::from_parts`], so [`istft`] does not re-validate
/// them.)
pub fn istft(spectrum: &Spectrum, length: Option<usize>) -> Result<Array> {
  // Every transform parameter is read straight off the typed `Spectrum` — no
  // inference, no ambiguity. The `Spectrum`'s invariants (even n_fft,
  // n_freqs == n_fft/2 + 1, 1 <= win_length <= n_fft, hop >= 1, num_frames >=
  // 1, Complex64 data) were enforced at construction by `stft` /
  // `Spectrum::from_parts`.
  let x = spectrum.data();
  let n_fft = spectrum.n_fft();
  let hop_length = spectrum.hop_length();
  let win_length = spectrum.win_length();
  let window_pad = spectrum.window_pad();
  let center = spectrum.center();
  let shape = x.shape();
  if shape.len() != 2 {
    return Err(Error::Backend {
      message: format!(
        "istft: expected 2-D (num_frames, n_freqs) spectrum data, got {}-D",
        shape.len()
      ),
    });
  }
  let num_frames = shape[0];
  // The irfft target length is the `Spectrum`'s OWN even `n_fft` — read from
  // the type, NOT inferred from the bin count. A `Spectrum` is guaranteed
  // (by `stft` / `Spectrum::from_parts`) to satisfy `n_freqs == n_fft / 2 + 1`
  // with `n_fft` even, `1 <= win_length <= n_fft`, `hop >= 1`, `num_frames >=
  // 1`, so there is no odd-vs-even ambiguity and no per-call re-derivation —
  // the historical `n_fft = (n_freqs - 1) * 2` inference (which could
  // misdecode an odd transform) is gone.

  // Right-pad short-window inversion is fundamentally NOT a faithful inverse.
  // For `WindowPad::Right` with `win_length < n_fft`, the right-pad geometry
  // combined with the `n_fft / 2` reflect-pad centering places the (symmetric
  // Hann) analysis window asymmetrically across the centered output region:
  // boundary samples are either zero-covered (caught by the coverage guard) OR
  // covered but reconstructed from distorted/discarded boundary energy (e.g. a
  // win > n_fft/2 short window reconstructs with large error while the coverage
  // guard does NOT fire). Because the forward transform discards / distorts
  // that boundary information, no partial guard can make it correct. Reject the
  // entire surface up front, before any reconstruction, so silent corruption is
  // structurally impossible. Short windows must use `WindowPad::Center`, which
  // is a faithful inverse. The forward `stft` keeps Right padding (byte-faithful
  // to mlx-audio); only the inverse restricts Right to `win_length == n_fft`.
  if matches!(window_pad, WindowPad::Right) && win_length != n_fft {
    return Err(Error::Backend {
      message: format!(
        "istft: WindowPad::Right supports only win_length == n_fft \
         (got win_length={win_length}, n_fft={n_fft}); right-pad short-window \
         inversion is not a faithful inverse — use WindowPad::Center for \
         short-window (win_length < n_fft) inversion"
      ),
    });
  }
  // Every frame is `n_fft` wide: the irfft output width AND the synthesis
  // window width (a `win_length`-wide window is placed into the `n_fft` frame
  // per `window_pad`, so the placed window is always exactly `n_fft` wide).
  // The overlap-add stride / frame width is therefore `n_fft`. Computed here,
  // before any window construction, so the OOM cap below precedes the
  // (potentially large) `frame_window` allocation.
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

  // Synthesis window via the SHARED `frame_window` — the symmetric Hann of
  // `win_length` placed into the `n_fft` frame per `window_pad`, the EXACT
  // same call the forward `stft` used for its analysis window, so the inverse
  // matches the forward by construction (no separately-specified window can
  // drift from `stft`'s). Built AFTER the OOM cap above so a pathological lazy
  // huge-`n_fft` spectrum is rejected before the window's CPU `Vec` (up to the
  // work cap) is ever allocated. The result is always exactly `n_fft` wide.
  let window = frame_window(win_length, n_fft, window_pad)?;

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

  // window_norm = w*w ALWAYS — the faithful OLA divisor. `stft` already
  // windowed each frame and `istft` multiplies by the synthesis window again
  // (in `windowed` above), so each sample carries a `w²` weight and the
  // overlap-add must be normalized by `Σ w²` (COLA / `torch.istft`). Dividing
  // by `Σ w` (the removed `normalized=false` branch) is a gain error against
  // this windowed-twice forward transform. Tiled across frames then flattened.
  let window_norm = ops::arithmetic::multiply(&window, &window)?;
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
  // makes returning a (divide-by-zero) corrupt sample structurally impossible.
  // It is a safety net for the supported configs — `WindowPad::Center` (any
  // `win_length`) and `WindowPad::Right` with `win_length == n_fft` — where it
  // should never fire; if it does, that is a real reconstruction-math bug to
  // fix, not to mask. (The non-invertible `WindowPad::Right` short-window
  // configs, where samples can be COVERED but still wrong, are rejected up
  // front above — the guard alone cannot catch those.)
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
           no window coverage in the overlap-add and is not recoverable \
           (n_fft={n_fft}, win_length={win_length}, hop={hop_length}, window_pad={window_pad:?}); \
           the requested region (e.g. a center=false head/tail) \
           includes a zero-coverage sample — adjust length/center or the window"
        ),
      });
    }
  }

  // Normalize by the squared-window-sum (`Σ w²`) where it exceeds the coverage
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
  // Mel front-end uses `WindowPad::Right` — the `mlx_audio.dsp` convention —
  // so a short `win_length < n_fft` produces byte-identical mel features to
  // the reference (and to mlxrs pre-#52). The forward `stft` supports Right
  // for any `win_length`; inversion is not needed here. Passed explicitly
  // (rather than relying on `WindowPad::default()`) so the mel placement is
  // pinned regardless of any future change to the enum default.
  let spec = stft(samples, n_fft, hop_length, win_length, WindowPad::Right)?;
  // `|stft|^2` — `abs` of the Complex64 spectrum data yields F32 magnitudes,
  // then square. `mel_spectrogram` only needs the magnitudes, so it reads the
  // transform array off the typed `Spectrum` (the metadata is irrelevant to
  // the forward magnitude path here).
  let mag = spec.data().abs()?;
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
  use crate::Dtype;

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
    let v = to_vec(&hamming_window(5).unwrap());
    let expected = [0.08_f32, 0.54, 1.0, 0.54, 0.08];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "hamming[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn hamming_endpoints_are_0_08() {
    // Distinguishing feature vs Hann: Hamming endpoints are 0.08, not 0.
    let v = to_vec(&hamming_window(8).unwrap());
    assert!((v[0] - 0.08).abs() < WIN_TOL, "first: {}", v[0]);
    assert!((v[7] - 0.08).abs() < WIN_TOL, "last: {}", v[7]);
  }

  #[test]
  fn blackman_matches_closed_form_n5() {
    // 0.42 - 0.5 cos(2π k/4) + 0.08 cos(4π k/4):
    // k=0: 0.42-0.5+0.08 = 0.0; k=1: 0.42-0+(-0.08)=0.34; k=2:
    // 0.42+0.5+0.08=1.0; k=3: 0.34; k=4: 0.0.
    let v = to_vec(&blackman_window(5).unwrap());
    let expected = [0.0_f32, 0.34, 1.0, 0.34, 0.0];
    for (i, (g, e)) in v.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "blackman[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn bartlett_matches_closed_form_n5_and_n4() {
    // n=5 (odd): triangle peaking at 1.0 in the center, 0 at the ends.
    let v5 = to_vec(&bartlett_window(5).unwrap());
    let e5 = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
    for (i, (g, e)) in v5.iter().zip(e5.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett5[{i}]: got {g}, want {e}");
    }
    // n=4 (even): 1 - 2|k - 1.5|/3 → [0, 2/3, 2/3, 0].
    let v4 = to_vec(&bartlett_window(4).unwrap());
    let e4 = [0.0_f32, 2.0 / 3.0, 2.0 / 3.0, 0.0];
    for (i, (g, e)) in v4.iter().zip(e4.iter()).enumerate() {
      assert!((g - e).abs() < WIN_TOL, "bartlett4[{i}]: got {g}, want {e}");
    }
  }

  #[test]
  fn windows_reject_n_lt_2() {
    for r in [
      hamming_window(0),
      hamming_window(1),
      blackman_window(1),
      bartlett_window(0),
      bartlett_window(1),
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
  // Every reconstruction test below drives the REAL public `stft` and feeds
  // its output straight into `istft` (`istft(&stft(signal, ..)?, ..)?`). The
  // private periodic-forward-helper pattern that hid the historical
  // synthesis/analysis window mismatch for 7 review rounds is BANNED — there
  // is no helper that builds spectra with its own window; the only forward is
  // `stft`, and `istft` rebuilds `stft`'s exact symmetric Hann via the shared
  // `frame_window`, so a window mismatch would surface here value-for-value.
  //
  // Each test asserts EVERY output sample value-for-value against the ORIGINAL
  // signal (no `.take`, no sub-range, no "intrinsically zero" caveats).
  // Expected values were cross-checked against a self-contained f64 numpy
  // mirror of stft/istft (`docs/istft_ref.py`, local-only) implementing the
  // same symmetric hann window, reflect-pad, OLA, and window-sum
  // normalization; that mirror reports max round-trip error <= 4.5e-16 for
  // every covered case here, so the f32 backend is asserted at 1e-5. The
  // coverage-guard / rejection tests assert the `Err` directly (they do NOT
  // mask a bad sample with a partial assertion).

  /// The 16-sample test signal used for the round-trips (arbitrary but fixed).
  fn signal_16() -> [f32; 16] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4,
    ]
  }

  /// A 19-sample fixed test signal for the non-hop-aligned round-trips.
  fn signal_19() -> [f32; 19] {
    [
      0.1, 0.5, -0.3, 0.8, -0.2, 0.6, 0.0, -0.7, 0.4, 0.9, -0.5, 0.2, 0.3, -0.1, 0.7, -0.4, 0.55,
      0.66, -0.77,
    ]
  }

  /// Round-trip `signal` through the REAL public [`stft`] then [`istft`] with
  /// the SAME `win_length` / `window_pad` (`istft` reads `n_fft` from the
  /// typed spectrum metadata and always uses the `Σw²` inverse), and assert
  /// EVERY output sample equals the original.
  ///
  /// This is the canary the previous review rounds were missing: it goes
  /// through `stft` itself (NOT a private periodic-forward helper), and the
  /// synthesis window `istft` rebuilds is the SAME symmetric Hann `stft`
  /// placed (both via `frame_window`) — so if the two ever drifted, this
  /// would fail value-for-value. `n_fft` is passed to `stft` only (even values
  /// only; `istft` reads it from the typed spectrum metadata, not the bin
  /// count). `len_override` is the `length` passed to `istft` (pass
  /// `Some(signal.len())` to recover the full original input length when
  /// `center=true`, including non-hop-aligned cases). The expected sample
  /// values were cross-checked against a self-contained f64 numpy mirror
  /// (`docs/istft_ref.py`, local-only) reporting max round-trip error
  /// <= 4.5e-16 for every covered case; the f32 backend is asserted at 1e-5.
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
    // `istft` reads n_fft / hop / win / pad / center FROM the typed `Spectrum`
    // (which `stft` built) — `length` is the ONLY inverse-side parameter, so
    // a synthesis/analysis mismatch is structurally impossible.
    let rec = istft(&spec, len_override).unwrap();
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
    assert_eq!(spec_c.data().shape(), vec![5, 5]); // (num_frames, n_fft/2+1)
    // Metadata is carried on the typed Spectrum (no inference downstream).
    assert_eq!(spec_c.n_fft(), 8);
    assert_eq!(spec_c.win_length(), 8);
    assert_eq!(spec_c.hop_length(), 4);
    assert_eq!(spec_c.window_pad(), WindowPad::Center);
    assert!(spec_c.center());
    for (c, r) in to_vec(&spec_c.data().abs().unwrap())
      .iter()
      .zip(to_vec(&spec_r.data().abs().unwrap()).iter())
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
    // invertible through the REAL public stft. `stft` places the symmetric Hann
    // of `win_length` center-padded into n_fft; `istft` rebuilds that EXACT
    // window via the shared `frame_window` and overlap-adds with the always-on
    // Σw² normalization. (min window-sum 0.41 for win=8, 1.01 for win=12; max err 1.1e-16 vs
    // the numpy mirror.) n_fft=16, hop=4, win=8 and win=12 — both cover the
    // centered 16-sample region. Asserts ALL 16 samples. This is the correctness
    // payoff of the Center convention (and of unifying the windows): the
    // short-window inverse the Right convention cannot do safely (Right
    // short-window inversion is rejected — see
    // `istft_right_short_window_rejected`).
    let buf = signal_16();
    for &win in &[8usize, 12usize] {
      assert_roundtrips_all_samples(&buf, 16, win, 4, WindowPad::Center, None);
    }
  }

  #[test]
  fn istft_right_short_window_rejected() {
    // THE FIX. WindowPad::Right inversion supports ONLY win_length == n_fft.
    // For win_length < n_fft the right-pad geometry is not a faithful inverse
    // (the forward transform discards/distorts boundary info), so istft REJECTS
    // it up front with a recoverable Err, BEFORE any reconstruction. The forward
    // stft (real, public) still produces the Right short-window spectrum — it is
    // the INVERSE that is rejected. We assert the Err DIRECTLY (no masked /
    // partial sample assertion).
    //
    // Probe win=8 (== n_fft/2; the symmetric Hann endpoints are zero, so this
    // boundary sample is ALSO zero-covered) AND win=12 (> n_fft/2; the boundary
    // sample is COVERED — window-sum well above COVERAGE_EPS — yet would still
    // mis-reconstruct, so the coverage guard alone would NOT catch it; this is
    // the heart of the fix and why the rejection is up-front, not guard-based).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    for &win in &[8usize, 12usize] {
      // The REAL public stft produces a valid Right short-window Spectrum
      // (carrying window_pad=Right, win<n_fft); it is the INVERSE that rejects
      // it, reading the placement off the typed Spectrum (no params passed).
      let spec = stft(&x, 16, 4, Some(win), WindowPad::Right).unwrap();
      assert_eq!(spec.data().shape(), vec![5, 9]); // (num_frames, n_fft/2+1), n_fft=16
      assert_eq!(spec.window_pad(), WindowPad::Right);
      assert_eq!(spec.win_length(), win);
      for len in [None, Some(16usize)] {
        let res = istft(&spec, len);
        assert!(
          matches!(res, Err(Error::Backend { .. })),
          "Right + win={win} < n_fft=16 (length={len:?}) must be rejected up front \
           (covered-but-wrong for win=12; the coverage guard does NOT catch it), \
           got {res:?}"
        );
      }
    }
    // Contrast: the SAME short window under WindowPad::Center is a faithful
    // inverse through the real stft and reconstructs EVERY sample — proving it is
    // the Right placement, not the short window per se, that is rejected.
    for &win in &[8usize, 12usize] {
      assert_roundtrips_all_samples(&buf, 16, win, 4, WindowPad::Center, None);
    }
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
    //
    // `center` is now carried on the Spectrum, so we build a `center = false`
    // Spectrum from the REAL stft's transform data via the validated
    // `Spectrum::from_parts` (stft itself always sets center = true). The
    // transform data is unchanged — only the carried `center` flag differs.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let spec_no_center = Spectrum::from_parts(
      spec.data().try_clone().unwrap(),
      8, // n_fft
      4, // hop_length
      8, // win_length
      WindowPad::Center,
      false, // center=false: requested region starts at the uncovered index 0
    )
    .unwrap();
    for len in [None, Some(10usize)] {
      let res = istft(&spec_no_center, len);
      assert!(
        matches!(res, Err(Error::Backend { .. })),
        "center=false head (length={len:?}) includes the zero-coverage OLA \
         index 0 and must hit the coverage guard, got {res:?}"
      );
    }
  }

  #[test]
  fn stft_rejects_odd_n_fft() {
    // Producer-side close of the odd-`n_fft` silent-misdecode path: a
    // one-sided spectrum has `n_freqs == n_fft / 2 + 1` for both `n_fft = 2k`
    // and `2k + 1`, so the bin count alone cannot disambiguate the parity.
    // `Spectrum` carries `n_fft` in the type (no inference), so keeping odd
    // `n_fft` off the producer means a `Spectrum` can never carry one: `stft`
    // must therefore reject odd `n_fft` up front rather than emit an
    // un-invertible spectrum. The signal
    // is long enough that an even `n_fft` of the same magnitude frames fine, so
    // the rejection is specifically about parity (not input length).
    let buf = signal_19();
    let x = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();
    for n_fft in [9usize, 15] {
      let res = stft(&x, n_fft, 4, None, WindowPad::Center);
      assert!(
        matches!(res, Err(Error::Backend { .. })),
        "odd n_fft={n_fft} must be rejected up front, got {res:?}"
      );
    }
    // Sanity: an even n_fft (8) of comparable magnitude still succeeds, proving
    // the rejection is parity-driven and not a length/shape failure.
    assert!(stft(&x, 8, 4, None, WindowPad::Center).is_ok());
  }

  #[test]
  fn istft_rejects_length_out_of_range() {
    // `length` (the desired output length) is the ONLY inverse-side parameter
    // now (n_fft/hop/win/pad/center are read off the Spectrum), so the only
    // istft-side numeric rejection is an out-of-range `length`. The structural
    // metadata rejections (bad shape, hop==0, win>n_fft, odd n_fft, wrong bin
    // count) are enforced at construction by `Spectrum::from_parts` — see
    // `spectrum_from_parts_rejects_inconsistent_metadata`.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    // length larger than the OLA length (t = (5-1)*4 + 8 = 24): center=true so
    // n_fft/2 + length = 4 + 1000 > 24 is out of range.
    assert!(matches!(
      istft(&spec, Some(1000)),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn spectrum_from_parts_rejects_inconsistent_metadata() {
    // `Spectrum::from_parts` is the validated constructor for EXTERNAL/raw
    // spectra: it must make it impossible to build a Spectrum whose metadata
    // istft would misdecode. This closes the external-odd-spectrum hole (the
    // bare-array path that allowed istft misdecodes for 11 review rounds) — a
    // Spectrum cannot exist with odd/inconsistent metadata.
    //
    // A valid `(num_frames=5, n_freqs=5)` Complex64 array for n_fft=8.
    let valid = Array::zeros::<f32>(&[5i32, 5i32])
      .unwrap()
      .astype(Dtype::Complex64)
      .unwrap();

    // Sanity: the consistent case constructs fine.
    assert!(
      Spectrum::from_parts(valid.try_clone().unwrap(), 8, 4, 8, WindowPad::Center, true).is_ok()
    );

    // Odd n_fft — THE external-odd-spectrum hole. (n_freqs=5 would match BOTH
    // n_fft=8 and the odd n_fft=9; the constructor must reject odd up front so
    // no Spectrum can ever carry it.)
    assert!(matches!(
      Spectrum::from_parts(valid.try_clone().unwrap(), 9, 4, 8, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));

    // Wrong n_freqs for the declared n_fft: n_fft=16 ⇒ n_freqs must be 9, but
    // the data has 5, so the bin count contradicts the metadata.
    assert!(matches!(
      Spectrum::from_parts(
        valid.try_clone().unwrap(),
        16,
        4,
        8,
        WindowPad::Center,
        true
      ),
      Err(Error::Backend { .. })
    ));

    // win_length > n_fft.
    assert!(matches!(
      Spectrum::from_parts(
        valid.try_clone().unwrap(),
        8,
        4,
        16,
        WindowPad::Center,
        true
      ),
      Err(Error::Backend { .. })
    ));

    // hop_length == 0 and win_length == 0.
    assert!(matches!(
      Spectrum::from_parts(valid.try_clone().unwrap(), 8, 0, 8, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));
    assert!(matches!(
      Spectrum::from_parts(valid.try_clone().unwrap(), 8, 4, 0, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));

    // n_fft == 0.
    assert!(matches!(
      Spectrum::from_parts(valid.try_clone().unwrap(), 0, 4, 0, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));

    // Non-2-D data (1-D).
    let one_d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32])
      .unwrap()
      .astype(Dtype::Complex64)
      .unwrap();
    assert!(matches!(
      Spectrum::from_parts(one_d, 8, 4, 8, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));

    // Non-Complex64 data (F32) with otherwise-consistent metadata.
    let real_data = Array::zeros::<f32>(&[5i32, 5i32]).unwrap();
    assert!(matches!(
      Spectrum::from_parts(real_data, 8, 4, 8, WindowPad::Center, true),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn spectrum_from_parts_then_istft_round_trips() {
    // An EXTERNAL Spectrum (rebuilt from raw stft data via `from_parts`, NOT
    // the stft-returned Spectrum) must invert exactly — proving the validated
    // constructor produces a faithfully-invertible Spectrum, not just a
    // well-formed one. n_fft=8, hop=4, win=8, Center, center=true.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let stft_spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let external = Spectrum::from_parts(
      stft_spec.data().try_clone().unwrap(),
      8,
      4,
      8,
      WindowPad::Center,
      true,
    )
    .unwrap();
    let rec = istft(&external, Some(16)).unwrap();
    let r = to_vec(&rec);
    assert_eq!(r.len(), 16);
    for (i, (g, e)) in r.iter().zip(buf.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "from_parts round-trip[{i}]: got {g}, want {e}"
      );
    }
  }

  #[test]
  fn istft_rejects_pathological_scatter_work_before_window_alloc() {
    // Codex OOM finding (+ the medium "work cap runs after window alloc"
    // finding): the real scatter/update workload is `num_frames * n_fft`, which
    // can dwarf the OLA *output* length `t` for small hops. The
    // `t <= MAX_DECODED_SAMPLES` cap does NOT catch this; the dedicated
    // MAX_OLA_WORK guard must reject it BEFORE the shared `frame_window`
    // (`hann_window(win_length)`, a CPU Vec up to the cap) and before any
    // broadcast/flatten/`try_reserve`/irfft.
    //
    // We use a LAZY mlx spectrum (`zeros(...).astype(Complex64)`) — nothing is
    // materialized — with the DEFAULT `win_length` (= n_fft). If the cap ran
    // after window construction, `frame_window` would first allocate
    // `hann_window(n_fft)` ≈ 18 Mi f32s; because the cap precedes window
    // construction, that allocation never happens.
    //
    // num_frames=4, n_freqs=9 Mi+1 → n_fft=(n_freqs-1)*2=18 Mi, win_length=18 Mi.
    //   work = num_frames * n_fft = 4 * 18 Mi = 72 Mi  > MAX_OLA_WORK (64 Mi) ✓
    //   t    = (4-1)*hop + n_fft  = 6 + 18 Mi ≈ 18 Mi  < MAX_DECODED  (64 Mi)
    // so ONLY the work cap can reject this.
    let n_freqs: i32 = 9 * 1024 * 1024 + 1;
    let num_frames: i32 = 4;
    let n_fft = (n_freqs as usize - 1) * 2; // 18 Mi (even; n_freqs == n_fft/2+1)
    let data = Array::zeros::<f32>(&[num_frames, n_freqs])
      .unwrap()
      .astype(crate::Dtype::Complex64)
      .unwrap();
    // `from_parts` accepts this (the shape/metadata are consistent: even n_fft,
    // n_freqs == n_fft/2+1, win<=n_fft) — it is a well-formed Spectrum. The
    // PATHOLOGY is the inverse work `num_frames * n_fft`, which only the
    // MAX_OLA_WORK guard inside istft can reject (and must, before frame_window
    // allocates `hann_window(n_fft)` ≈ 18 Mi f32s).
    let spec = Spectrum::from_parts(
      data,
      n_fft,
      2,     // small hop → t stays under the decoded cap
      n_fft, // win_length == n_fft (Right would also be valid here)
      WindowPad::Center,
      true,
    )
    .unwrap();
    let res = istft(&spec, None);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "pathological num_frames*n_fft must be rejected by the MAX_OLA_WORK cap \
       before the frame_window allocation"
    );
  }

  #[test]
  fn stft_rejects_pathological_work_before_alloc() {
    // Codex finding: cap stft's forward work. A LAZILY-shaped huge input (no
    // data materialized) with a small n_fft and hop=1 produces num_frames ≈
    // input length and a strided frame view of `num_frames * n_fft` elements —
    // orders of magnitude past the sample count. The MAX_STFT_WORK guard must
    // reject it BEFORE building the frame view / window / rfft (i.e. before any
    // allocation). The public sample cap (MAX_DECODED_SAMPLES = 64 Mi) does NOT
    // catch this: the input length is AT the sample cap, but the frame work is
    // ~64 Gi.
    //
    // We use a lazy `zeros` 1-D array of 64 Mi samples; with n_fft=1024, hop=1:
    //   padded_len ≈ 64 Mi (+ 1024), num_frames ≈ 64 Mi
    //   frame work = num_frames * n_fft ≈ 64 Mi * 1024 = 64 Gi  >> MAX_STFT_WORK
    // Nothing is materialized, so if the cap did NOT run first this would try a
    // multi-GB framing/FFT allocation. Asserting Err proves the cap fired early.
    let n_samples = 64 * 1024 * 1024i32;
    let lazy = Array::zeros::<f32>(&[n_samples]).unwrap();
    let res = stft(&lazy, 1024, 1, None, WindowPad::Right);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "pathological lazy huge-shape stft input (num_frames * n_fft) must be \
       rejected by the MAX_STFT_WORK cap before any framing/FFT allocation, got {res:?}"
    );
  }

  #[test]
  fn stft_rejects_oversized_input_before_reflect_pad_large_hop() {
    // Codex OOM finding: the reflect pad (`center=true`) is a lazy
    // slice+concatenate, but *evaluating* it materializes a signal proportional
    // to the INPUT length — independent of num_frames. The `MAX_STFT_WORK` cap
    // only bounds `num_frames * n_fft`, so a lazily-shaped huge input with a
    // LARGE hop (few frames) slips past it while the reflect-pad concatenate
    // still balloons. The input/padded-length cap must reject it BEFORE the
    // reflect pad.
    //
    // We use a lazy `zeros` 1-D array of MAX_DECODED_SAMPLES + 16 samples (just
    // ABOVE the budget) with n_fft=16 and a LARGE hop (= MAX_DECODED_SAMPLES):
    //   samples_len = 64 Mi + 16  > MAX_DECODED_SAMPLES (64 Mi)  → new cap fires
    // and, crucially, the OLD work cap would NOT catch this:
    //   padded_len ≈ 64 Mi + 32, num_frames = 1 + (64 Mi + 16)/64 Mi = 2,
    //   frame_work = num_frames * n_fft = 2 * 16 = 32  ≪ MAX_STFT_WORK (64 Mi).
    // So ONLY the input/padded-length cap (checked before the reflect-pad
    // concatenate) can reject this; asserting Err proves it fired first.
    let n_samples = (crate::audio::io::MAX_DECODED_SAMPLES + 16) as i32;
    let lazy = Array::zeros::<f32>(&[n_samples]).unwrap();
    let large_hop = crate::audio::io::MAX_DECODED_SAMPLES;
    let res = stft(&lazy, 16, large_hop, None, WindowPad::Right);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "oversized lazy stft input with a large hop (work cap would pass) must be \
       rejected by the input/padded-length cap before the reflect pad, got {res:?}"
    );
  }

  #[test]
  fn mel_spectrogram_short_window_uses_right_pad_unchanged() {
    // Pin that `mel_spectrogram` keeps its `mlx_audio.dsp` `WindowPad::Right`
    // placement for a SHORT `win_length < n_fft`, so its features are
    // byte-identical to building the mel by hand on the Right-padded stft (and
    // to mlxrs pre-#52). Making `WindowPad::Right` the stft default is exactly
    // so this front-end stays unchanged. n_fft=16, win=8 (< n_fft), hop=4.
    //
    // Expected = mel_filter_bank @ |stft(.., Right)|² (the canonical
    // mel-spectrogram pipeline), computed directly here. This both confirms the
    // value is unchanged AND pins the pad: the SAME mel built on the Center-
    // padded stft differs (asserted below), so a silent flip of mel's pad to
    // Center would fail this test.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let n_fft = 16usize;
    let win = 8usize;
    let hop = 4usize;
    let n_mels = 6usize;
    let sr = 16_000u32;

    let got = to_vec(&mel_spectrogram(&x, n_fft, hop, Some(win), n_mels, sr, 0.0, None).unwrap());

    // Hand-built reference on the Right-padded stft.
    let expected_mel = {
      let spec = stft(&x, n_fft, hop, Some(win), WindowPad::Right).unwrap();
      let power = spec.data().abs().unwrap().square().unwrap();
      let bank = mel_filter_bank(n_mels, n_fft, sr, 0.0, None).unwrap();
      let power_t = power.transpose().unwrap();
      to_vec(&ops::linalg_basic::matmul(&bank, &power_t).unwrap())
    };
    assert_eq!(got.len(), expected_mel.len(), "mel length mismatch");
    for (i, (g, e)) in got.iter().zip(expected_mel.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "mel_spectrogram[{i}] must match the Right-padded reference: got {g}, want {e}"
      );
    }

    // Pin the pad: the Center-padded stft gives a DIFFERENT mel, so if mel ever
    // silently switched to Center this test would catch it (the short window is
    // placed at a different offset, shifting the spectral energy).
    let center_mel = {
      let spec = stft(&x, n_fft, hop, Some(win), WindowPad::Center).unwrap();
      let power = spec.data().abs().unwrap().square().unwrap();
      let bank = mel_filter_bank(n_mels, n_fft, sr, 0.0, None).unwrap();
      let power_t = power.transpose().unwrap();
      to_vec(&ops::linalg_basic::matmul(&bank, &power_t).unwrap())
    };
    let max_diff = got
      .iter()
      .zip(center_mel.iter())
      .map(|(r, c)| (r - c).abs())
      .fold(0.0_f32, f32::max);
    assert!(
      max_diff > 1e-4,
      "Right- and Center-padded short-window mel must DIFFER (else the pad pin \
       is vacuous); max diff was {max_diff}"
    );
  }
}
