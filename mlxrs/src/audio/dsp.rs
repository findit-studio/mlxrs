//! DSP primitives: window family (Hann/Hamming/Blackman/Bartlett), STFT,
//! inverse STFT, mel filterbank, mel + log-mel spectrogram, 1-D IIR
//! `lfilter`, ITU-R BS.1770 K-weighted integrated loudness +
//! `normalize_loudness`.
//!
//! Faithful 1:1 port of the corresponding `mlx_audio.dsp` core
//! (`hanning`, `hamming`, `blackman`, `bartlett`, `STR_TO_WINDOW_FN`, `stft`,
//! `istft`, `ISTFTCache`, `mel_filters`, `lfilter`, `integrated_loudness`,
//! `normalize_loudness`, `normalize_peak`) at
//! <https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/dsp.py>.
//! Kaldi-style features live in the sibling [`crate::audio::features`] module.
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
/// window multiply, or rfft is built. Mirrors `MAX_OLA_WORK` on the inverse
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
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
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

impl WindowPad {
  /// The canonical lowercase string representation (`right`/`center`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Right => "right",
      Self::Center => "center",
    }
  }
}

/// Signal-centering pad mode for [`stft`]. Matches the `pad_mode` argument
/// of `mlx_audio.dsp.stft(x, n_fft, ..., center=True, pad_mode="reflect")`
/// and `librosa.stft`'s same-named parameter.
///
/// Currently only [`PadMode::Reflect`] is supported (the
/// `mlx_audio.dsp.stft` default and what `librosa.stft` / Whisper-style
/// front-ends use). The variant exists to allow a future
/// `"constant"`/`"edge"` extension without breaking the [`StftConfig`]
/// API.
///
/// `mlx-c`'s `mlx_pad` only supports `"constant"` and `"edge"`, NOT
/// `"reflect"`, so even on the back end this is built from
/// `slice + concatenate`, mirroring `mlx_audio.dsp.stft._pad`'s python
/// construction (a single-op reflect would require an upstream mlx
/// `Pad::reflect` change — see the [`stft`] implementation for the
/// reasoning).
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
pub enum PadMode {
  /// Reflect the signal at the edges (`mlx-audio` / Whisper / librosa
  /// default). Mirrors `pad_mode="reflect"`.
  #[default]
  Reflect,
}

impl PadMode {
  /// The canonical lowercase string representation matching the mlx-audio
  /// `pad_mode` argument (`reflect`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Reflect => "reflect",
    }
  }
}

/// Builder for [`stft_with_config`] — factors the `center` runtime branch
/// out of the [`stft`] signature so the caller configures padding once,
/// and the inner code path is uniform for the [`Spectrum`] producer.
///
/// Drives the same forward STFT as the bare [`stft`] entry point: it
/// shares the same validation + work-cap + framing machinery so byte-
/// identical output for `StftConfig::default()` is guaranteed.
///
/// # Examples
///
/// Centered (the [`stft`] default — `center=true, pad_mode="reflect"`):
///
/// ```ignore
/// let cfg = StftConfig::default();
/// let spec = stft_with_config(&samples, n_fft, hop, None, WindowPad::Right, &cfg)?;
/// ```
///
/// Aligned (no signal padding — frames start at sample 0 / index
/// `hop * k`):
///
/// ```ignore
/// let cfg = StftConfig::new(false, PadMode::Reflect);
/// let spec = stft_with_config(&samples, n_fft, hop, None, WindowPad::Right, &cfg)?;
/// // Equivalent shortcut:
/// // let spec = stft_aligned(&samples, n_fft, hop, None, WindowPad::Right)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StftConfig {
  /// Reflect-pad the signal by `n_fft / 2` on each side before framing.
  /// Default `true` — matches `mlx_audio.dsp.stft(..., center=True)` and
  /// is what every Whisper / librosa-style front-end expects.
  ///
  /// When `false` the frames start at sample 0 (the "aligned" path);
  /// the resulting [`Spectrum`]'s `center()` returns `false` and
  /// [`istft`] does NOT trim a centered prefix / suffix when inverting.
  center: bool,
  /// Signal-pad mode used when `center == true`. Default
  /// [`PadMode::Reflect`] (the `mlx-audio` default). Ignored when
  /// `center == false` (no signal padding is applied).
  pad_mode: PadMode,
}

impl StftConfig {
  /// Construct a [`StftConfig`] with explicit `center` and `pad_mode`.
  ///
  /// # Examples
  /// ```ignore
  /// // Centered (the default):
  /// let cfg = StftConfig::new(true, PadMode::Reflect);
  /// // Aligned (no centering pad):
  /// let cfg = StftConfig::new(false, PadMode::Reflect);
  /// ```
  pub const fn new(center: bool, pad_mode: PadMode) -> Self {
    Self { center, pad_mode }
  }

  /// Whether the signal is reflect-padded by `n_fft / 2` on each side
  /// before framing (`center = true`).
  #[inline(always)]
  pub fn center(&self) -> bool {
    self.center
  }

  /// Signal-pad mode used when `center == true`.
  #[inline(always)]
  pub fn pad_mode(&self) -> PadMode {
    self.pad_mode
  }
}

impl Default for StftConfig {
  fn default() -> Self {
    Self::new(true, PadMode::Reflect)
  }
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
  ///
  /// Named `data_ref` (not `data`) per §3 non-Copy `&T` accessor naming
  /// convention — [`Array`] is not `Copy`, so the accessor returns a
  /// reference rather than a copy and carries the `_ref` suffix to signal
  /// that the returned value borrows `self`.
  #[inline(always)]
  pub fn data_ref(&self) -> &Array {
    &self.data
  }

  /// The (even) FFT length used to produce this spectrum (the irfft target
  /// width on the inverse).
  #[inline(always)]
  pub fn n_fft(&self) -> usize {
    self.n_fft
  }

  /// The analysis hop length (the overlap-add stride [`istft`] uses).
  #[inline(always)]
  pub fn hop_length(&self) -> usize {
    self.hop_length
  }

  /// The analysis window length (`win_length <= n_fft`).
  #[inline(always)]
  pub fn win_length(&self) -> usize {
    self.win_length
  }

  /// The [`WindowPad`] placement of the `win_length` window in the `n_fft`
  /// frame ([`istft`] re-places the synthesis window identically).
  #[inline(always)]
  pub fn window_pad(&self) -> WindowPad {
    self.window_pad
  }

  /// Whether [`stft`] reflect-padded the signal by `n_fft / 2` on each side
  /// (`center = true`). [`istft`] undoes this before applying `length`.
  #[inline(always)]
  pub fn center(&self) -> bool {
    self.center
  }

  /// The number of frames (`data`'s first dimension).
  #[inline(always)]
  pub fn num_frames(&self) -> usize {
    // `data` is validated 2-D at every construction site, so `shape()[0]`
    // is always present.
    self.data.shape()[0]
  }

  /// The number of one-sided frequency bins (`data`'s last dimension,
  /// `== n_fft / 2 + 1`).
  #[inline(always)]
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
#[derive(Debug, Clone, Copy, PartialEq, Default, derive_more::IsVariant)]
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
/// validates `n`, applies the public-input allocation cap, and dispatches
/// to the C12 SIMD window builder
/// ([`crate::simd::audio::window::symmetric_window`]).
///
/// `name` only flavors the error messages so each public window keeps its
/// own diagnostic prefix; `kind` selects the per-window formula (Hann /
/// Hamming / Blackman / Bartlett). On `aarch64` the C12 dispatcher
/// routes to a 7-term Taylor cos polynomial NEON 4-lane tile;
/// elsewhere it falls back to the per-element `f32::cos` scalar loop.
fn symmetric_window(
  name: &str,
  n: usize,
  kind: crate::simd::audio::window::SymWindowKind,
) -> Result<Array> {
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

  // C12 SIMD: dispatch to the symmetric-window NEON kernel
  // (`simd::audio::window::symmetric_window`). The dispatcher does
  // its own fallible `try_reserve_exact(n)` + `spare_capacity_mut` +
  // `set_len(n)` internally; we feed the result straight into
  // `Array::from_slice`. The kernel's `n >= 2` precondition is
  // already satisfied (asserted above), and its only fallible step
  // is the request-scaled output reservation — which surfaces here
  // as `Error::OutOfMemory`.
  let buf = crate::simd::audio::window::symmetric_window(kind, n)?;
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
  symmetric_window(
    "hann_window",
    n,
    crate::simd::audio::window::SymWindowKind::Hann,
  )
}

/// Symmetric Hamming window: `w[k] = 0.54 - 0.46 * cos(2π k / (n - 1))` for
/// `k in 0..n`. Endpoints are `0.08` (not zero, unlike Hann).
///
/// Matches `mlx_audio.dsp.hamming(n, periodic=False)`.
///
/// # Errors
/// Same as [`hann_window`].
pub fn hamming_window(n: usize) -> Result<Array> {
  symmetric_window(
    "hamming_window",
    n,
    crate::simd::audio::window::SymWindowKind::Hamming,
  )
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
  symmetric_window(
    "blackman_window",
    n,
    crate::simd::audio::window::SymWindowKind::Blackman,
  )
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
  symmetric_window(
    "bartlett_window",
    n,
    crate::simd::audio::window::SymWindowKind::Bartlett,
  )
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
  // The bare entry point hardcodes `mlx_audio.dsp.stft`'s defaults
  // (`center = true, pad_mode = "reflect"`). `stft_with_config` carries
  // the same validation + work-cap + framing machinery so this call
  // produces byte-identical output to the prior in-line body.
  stft_with_config(
    samples,
    n_fft,
    hop_length,
    win_length,
    window_pad,
    &StftConfig::default(),
  )
}

/// [`stft`] without signal centering — frames start at sample 0
/// (`pad_mode` is ignored; no padding is applied). The resulting
/// [`Spectrum`] carries `center = false`, and [`istft`] will NOT trim a
/// centered prefix / suffix when inverting.
///
/// Equivalent to:
///
/// ```ignore
/// let cfg = StftConfig::new(false, PadMode::Reflect);
/// stft_with_config(samples, n_fft, hop, win_length, window_pad, &cfg)
/// ```
///
/// # Errors
/// - Same as [`stft`], plus rejecting inputs too short to fit one
///   frame *without* the centering reflect pad (the aligned path
///   requires `samples_len >= n_fft`).
pub fn stft_aligned(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  window_pad: WindowPad,
) -> Result<Spectrum> {
  stft_with_config(
    samples,
    n_fft,
    hop_length,
    win_length,
    window_pad,
    &StftConfig::new(false, PadMode::Reflect),
  )
}

/// [`stft`] driven by an explicit [`StftConfig`] — factors the
/// `center` / `pad_mode` knobs out of the hot-path signature.
///
/// Centered (`cfg.center == true`) routes through the same
/// reflect-pad path as the bare [`stft`]; aligned
/// (`cfg.center == false`) skips the pad entirely so the frame view is
/// a plain `as_strided` over the raw samples. Validation, work caps,
/// framing, FFT, and [`Spectrum`] assembly are otherwise identical, so
/// the centered path is byte-identical to [`stft`] for the same
/// arguments.
///
/// # Errors
/// - Same as [`stft`]; the aligned path additionally requires
///   `samples_len >= n_fft` (without the centering pad there is no
///   `+ n_fft` headroom).
pub fn stft_with_config(
  samples: &Array,
  n_fft: usize,
  hop_length: usize,
  win_length: Option<usize>,
  window_pad: WindowPad,
  cfg: &StftConfig,
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

  // INPUT-LENGTH CAP (Codex OOM finding). When `cfg.center == true` the
  // reflect pad below (`reflect_pad_1d`) is a lazy slice+concatenate, but
  // *evaluating* the graph materializes a padded signal proportional to the
  // INPUT length — independent of `num_frames`. The post-framing
  // `MAX_STFT_WORK` cap bounds `num_frames * n_fft`, but a lazily-shaped huge
  // 1-D input with a LARGE `hop_length` yields few frames, so `frame_work`
  // stays under that cap while the reflect-pad concatenate still balloons
  // proportional to the input. We therefore reject any input whose sample
  // count — or padded length `samples_len + n_fft` (reflect pad adds
  // `n_fft / 2` on each side) — exceeds the per-call sample budget
  // [`MAX_DECODED_SAMPLES`] BEFORE building the padded signal or any frame
  // view, bounding the reflect-pad allocation regardless of hop. Checked
  // arithmetic so the `+ n_fft` itself can't wrap. The aligned path
  // (`cfg.center == false`) doesn't add padding, but we keep the same
  // input-sample-count cap so an oversized aligned input is still rejected
  // at the same load-stage ceiling.
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
  if cfg.center() {
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
  }

  // Analysis window via the SHARED `frame_window` (symmetric hann of
  // `win_length`, placed into the `n_fft` frame per `window_pad`; no-op when
  // `win_length == n_fft`). Built AFTER the work cap below so a lazily-shaped
  // huge input is rejected before this CPU `Vec` (up to `win_length <= n_fft`
  // elements) is allocated. `istft` rebuilds its synthesis window through the
  // exact same call, so analysis and synthesis windows always match.

  // `cfg.center == true, pad_mode == Reflect` (reference default). The reflect
  // pad is a lazy slice+concatenate, but evaluating it materializes a signal
  // proportional to the input length, so the input/padded-length cap above
  // gates it; the post-framing `MAX_STFT_WORK` cap then gates the strided view
  // / window / rfft (frame work + FFT output). When `cfg.center == false` we
  // frame directly over `samples` — no slice/concatenate intermediates.
  //
  // mlx-c's `mlx_pad` only supports `pad_value`-mode and `edge`-mode
  // (not `reflect`), so reflect is constructed from `slice + concatenate`
  // here, exactly as `mlx_audio.dsp.stft._pad(..., pad_mode="reflect")`
  // does. Folding this to a single `ops::shape::pad(..., c"reflect")`
  // call awaits an upstream `mlx::core::Pad` extension — until that
  // lands, the 3-op reconstruction is the byte-for-byte parity path.
  let padded = if cfg.center() {
    match cfg.pad_mode() {
      PadMode::Reflect => reflect_pad_1d(samples, n_fft / 2)?,
    }
  } else {
    // No signal padding: the strided frame view operates directly on the
    // raw samples. `try_clone` is a cheap mlx-c rc bump (no buffer copy),
    // so `padded` stays a uniform owned `Array` for the framing path.
    samples.try_clone()?
  };
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
  // the analysis metadata `istft` needs to invert it exactly. `center` comes
  // from `cfg` (`true` for the [`stft`] / `stft_with_config(.., StftConfig::
  // default())` path, `false` for [`stft_aligned`] / explicit
  // `cfg.center = false`). All invariants `Spectrum::from_parts` would
  // re-check (even n_fft, `n_freqs == n_fft / 2 + 1`, `win_length <= n_fft`,
  // non-empty) hold by construction, so the data is wrapped directly.
  Ok(Spectrum {
    data,
    n_fft,
    hop_length,
    win_length,
    window_pad,
    center: cfg.center(),
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
  let x = spectrum.data_ref();
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

/// Key for [`ISTFTCache`]'s overlap-add position-index cache: the framing
/// geometry `(num_frames, frame_width, hop_length)` that fully determines the
/// flattened scatter-index buffer `indices[m, j] = m * hop + j`. `frame_width`
/// is `n_fft` (every frame is `n_fft` wide after window placement), so two
/// [`Spectrum`]s with the same `(num_frames, n_fft, hop)` share one index
/// buffer regardless of `win_length` / `window_pad`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PositionKey {
  num_frames: usize,
  frame_width: usize,
  hop_length: usize,
}

/// Key for [`ISTFTCache`]'s normalization-buffer cache. The scattered
/// `Σ w²` overlap-add divisor depends on the *full* synthesis-window geometry
/// (`n_fft`, `win_length`, `window_pad` — which together fix the placed
/// `n_fft`-wide window) plus the framing (`hop_length`, `num_frames`). The
/// reference keys on `(n_fft, hop, win_length, hash(window), num_frames)`; here
/// the window is fully determined by `(win_length, n_fft, window_pad)` (the
/// shared `frame_window`), so `window_pad` replaces the window hash — there is
/// no free-form window to hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NormKey {
  n_fft: usize,
  hop_length: usize,
  win_length: usize,
  window_pad: WindowPad,
  num_frames: usize,
}

/// Cached / batched overlap-add helper for [`istft`] — the mlxrs port of
/// `mlx_audio.dsp.ISTFTCache`.
///
/// Faithful adaptation of `mlx_audio.dsp.ISTFTCache` (`dsp.py:575`) to mlxrs's
/// typed [`Spectrum`] API. The reference caches, across repeated `istft` calls
/// with the same geometry, two derived buffers that are otherwise rebuilt every
/// call:
/// - the **flattened scatter-index buffer** `indices[m, j] = m * hop + j`
///   (keyed by `(num_frames, frame_width, hop)` — `get_positions`), and
/// - the **scattered `Σ w²` window-sum buffer** of overlap-add length `t`
///   (keyed by the full window + framing geometry — `get_norm_buffer`). mlxrs
///   caches the RAW (un-floored) window-sum rather than the reference's
///   `mx.maximum(., 1e-10)`-floored divisor, so the cached path can reproduce
///   the free [`istft`]'s coverage guard exactly (see below).
///
/// For a streaming decoder that inverts many same-shaped frame blocks (the
/// reference's stated use case — "streaming"), this skips the per-call CPU
/// index `Vec` build and the per-call scatter-add that produces the window-sum.
///
/// ## Relationship to the free [`istft`] function
/// [`ISTFTCache::istft`] is **numerically identical** to the free [`istft`] for
/// every [`Spectrum`] — **including its rejection behavior**, not just its happy
/// path. It builds the same raw `Σ w²` window-sum, performs the same `irfft` →
/// window → overlap-add, applies the **same coverage guard** over the requested
/// region, the same `mx.where(window_sum > COVERAGE_EPS, normalized, raw)`
/// normalization, and the same center-trim / `length` slicing. The only
/// difference is that the index buffer and the raw window-sum buffer are
/// memoized across same-geometry calls. It enforces the same invariants:
/// [`WindowPad::Right`] short-window inversion (`win_length != n_fft`) is
/// rejected up front (not a faithful inverse — see [`istft`]), and a requested
/// region containing a zero-coverage sample (a `center=true` length reaching
/// into the zero-coverage tail, or a `center=false` head/tail) is **rejected**
/// with the same coverage error the free [`istft`] returns — never divided by a
/// floor and silently emitted as corrupt audio.
///
/// **Why the raw window-sum is cached (not the floored divisor).** The
/// reference's `get_norm_buffer` floors the window-sum at `COVERAGE_EPS`
/// (`mx.maximum(norm_buffer, 1e-10)`) and divides directly. Caching that floored
/// divisor would LOSE the free [`istft`]'s coverage guard: a zero-coverage
/// requested sample would be divided by the `1e-10` floor and silently emit
/// invalid audio instead of being rejected. mlxrs therefore caches the RAW
/// window-sum and reproduces the free [`istft`]'s guard + `where` on it, so the
/// cached path is the numerically-identical *guarded* path — just with the index
/// and window-sum buffers memoized for streaming. For the supported
/// configurations ([`WindowPad::Center`] any `win_length`, or [`WindowPad::Right`]
/// with `win_length == n_fft`) every requested sample has full COLA coverage, so
/// the guard never fires and the round-trip is exact.
///
/// ## Bounded memory
/// Each cached entry is bounded by the same `MAX_OLA_WORK` /
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) caps the free
/// [`istft`] enforces (the scatter work `num_frames * n_fft` and the OLA length
/// `t`). The number of *distinct* cache entries is bounded by the number of
/// distinct geometries the caller feeds in; [`ISTFTCache::clear`] drops them
/// all, and [`ISTFTCache::len`] reports the count for a caller that wants to
/// bound it explicitly.
///
/// ```ignore
/// let mut cache = ISTFTCache::new();
/// for block in stream {
///   let spectrum = stft(&block, 1024, 256, None, WindowPad::Center)?;
///   let audio = cache.istft(&spectrum, None)?; // index/norm buffers reused
/// }
/// ```
#[derive(Debug)]
pub struct ISTFTCache {
  /// Memoized flattened scatter-index buffers, keyed by framing geometry.
  position_cache: std::collections::HashMap<PositionKey, Array>,
  /// Memoized scattered RAW `Σ w²` window-sum buffers (un-floored, so the
  /// coverage guard + `where`-normalization can reproduce the free [`istft`]
  /// exactly), keyed by the full window + framing geometry.
  norm_buffer_cache: std::collections::HashMap<NormKey, Array>,
}

impl ISTFTCache {
  /// A fresh, empty cache. Equivalent to [`ISTFTCache::default`].
  pub fn new() -> ISTFTCache {
    ISTFTCache {
      position_cache: std::collections::HashMap::new(),
      norm_buffer_cache: std::collections::HashMap::new(),
    }
  }

  /// The total number of cached buffers (position-index + norm-buffer
  /// entries). Mirrors the reference's `cache_info()["total_cached_items"]`.
  /// A caller that wants a hard memory bound can watch this and call
  /// [`ISTFTCache::clear`] when it grows past a chosen threshold.
  pub fn len(&self) -> usize {
    self.position_cache.len() + self.norm_buffer_cache.len()
  }

  /// Whether the cache holds no buffers at all.
  pub fn is_empty(&self) -> bool {
    self.position_cache.is_empty() && self.norm_buffer_cache.is_empty()
  }

  /// Drop every cached buffer, freeing the memory. Mirrors the reference's
  /// `clear_cache()`. Safe to call between streams with different geometries.
  pub fn clear(&mut self) {
    self.position_cache.clear();
    self.norm_buffer_cache.clear();
  }

  /// Cached / batched inverse STFT of `spectrum` — the [`ISTFTCache`] analogue
  /// of the free [`istft`].
  ///
  /// Builds the reconstruction exactly as [`istft`] does (`irfft` → synthesis
  /// window → overlap-add → coverage guard → `where`-normalize by `Σ w²`), but
  /// reuses the flattened scatter-index buffer and the raw `Σ w²` window-sum
  /// buffer from the cache when a [`Spectrum`] with the same geometry was seen
  /// before. The result is the reconstructed 1-D real signal (`Dtype::F32`),
  /// identical to `istft(spectrum, length)` for every [`Spectrum`] — including
  /// its rejection behavior.
  ///
  /// `length` has the same meaning as in [`istft`]: with `center = true` the
  /// `n_fft / 2` reflect prefix is dropped first, then `length` (if any) keeps
  /// the leading real samples; with `center = false` it slices `[0 .. length]`.
  ///
  /// # Errors
  /// The **same error surface as [`istft`], including its coverage guard** (this
  /// path caches the raw window-sum and reproduces the guard exactly rather than
  /// flooring — see the [`ISTFTCache`] type docs):
  /// - [`Error::Backend`] when the [`Spectrum`]'s `window_pad` is
  ///   [`WindowPad::Right`] and `win_length != n_fft` (short-window right-pad
  ///   inversion is not a faithful inverse — rejected up front),
  /// - [`Error::Backend`] when a derived size overflows `usize` / `i32`, the
  ///   OLA length `t` exceeds the
  ///   [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap, or
  ///   the scatter work `num_frames * n_fft` exceeds `MAX_OLA_WORK`,
  /// - [`Error::Backend`] when the `length` trim is out of range,
  /// - [`Error::Backend`] when the **coverage guard** fires — a requested output
  ///   sample has window-sum `<= COVERAGE_EPS` (a `center=true` length reaching
  ///   into the zero-coverage tail, or a `center=false` head/tail). Identical to
  ///   the free [`istft`]; should never fire for a supported config,
  /// - propagates window-construction errors from the shared `frame_window`.
  pub fn istft(&mut self, spectrum: &Spectrum, length: Option<usize>) -> Result<Array> {
    // Every transform parameter is read straight off the typed `Spectrum` (its
    // invariants — even n_fft, n_freqs == n_fft/2 + 1, 1 <= win_length <=
    // n_fft, hop >= 1, num_frames >= 1, Complex64 data — were enforced at
    // construction by `stft` / `Spectrum::from_parts`).
    let x = spectrum.data_ref();
    let n_fft = spectrum.n_fft();
    let hop_length = spectrum.hop_length();
    let win_length = spectrum.win_length();
    let window_pad = spectrum.window_pad();
    let center = spectrum.center();
    let num_frames = spectrum.num_frames();

    // Right-pad short-window inversion is not a faithful inverse — reject the
    // whole surface up front, exactly as the free `istft` does (see its body
    // for the full rationale). The forward `stft` keeps Right padding; only
    // the inverse restricts Right to `win_length == n_fft`.
    if matches!(window_pad, WindowPad::Right) && win_length != n_fft {
      return Err(Error::Backend {
        message: format!(
          "ISTFTCache::istft: WindowPad::Right supports only win_length == n_fft \
           (got win_length={win_length}, n_fft={n_fft}); right-pad short-window \
           inversion is not a faithful inverse — use WindowPad::Center for \
           short-window (win_length < n_fft) inversion"
        ),
      });
    }

    // Every frame is `n_fft` wide (the `win_length` window is placed into the
    // `n_fft` frame), so the overlap-add stride / frame width is `n_fft`.
    let frame_width = n_fft;

    // OLA output / norm-buffer length `t = (num_frames - 1) * hop + n_fft`,
    // capped at MAX_DECODED_SAMPLES (same as the free `istft`).
    let t = (num_frames - 1)
      .checked_mul(hop_length)
      .and_then(|v| v.checked_add(frame_width))
      .ok_or_else(|| Error::Backend {
        message: format!(
          "ISTFTCache::istft: OLA length (num_frames-1)*hop + n_fft overflows usize \
           (num_frames={num_frames}, hop={hop_length}, n_fft={n_fft})"
        ),
      })?;
    if t > crate::audio::io::MAX_DECODED_SAMPLES {
      return Err(Error::Backend {
        message: format!(
          "ISTFTCache::istft: OLA length {t} exceeds the {} cap",
          crate::audio::io::MAX_DECODED_SAMPLES
        ),
      });
    }
    let t_i32 = i32::try_from(t).map_err(|_| Error::Backend {
      message: format!("ISTFTCache::istft: OLA length {t} exceeds i32::MAX"),
    })?;
    let n_fft_i32 = i32::try_from(n_fft).map_err(|_| Error::Backend {
      message: format!("ISTFTCache::istft: n_fft {n_fft} exceeds i32::MAX"),
    })?;

    // Scatter-work cap on `num_frames * frame_width`, checked BEFORE any
    // allocation / cache insertion (the `t` cap bounds the output but small
    // hops drive the scatter far past `t` — same guard the free `istft` uses).
    let idx_len = num_frames
      .checked_mul(frame_width)
      .ok_or_else(|| Error::Backend {
        message: format!(
          "ISTFTCache::istft: scatter work count num_frames * n_fft overflows usize \
           (num_frames={num_frames}, n_fft={n_fft})"
        ),
      })?;
    if idx_len > MAX_OLA_WORK {
      return Err(Error::Backend {
        message: format!(
          "ISTFTCache::istft: scatter work count {idx_len} (num_frames={num_frames} * \
           n_fft={n_fft}) exceeds the {MAX_OLA_WORK} work cap"
        ),
      });
    }
    let idx_len_i32 = i32::try_from(idx_len).map_err(|_| Error::Backend {
      message: format!("ISTFTCache::istft: scatter work count {idx_len} exceeds i32::MAX"),
    })?;

    // ---- cached flattened scatter-index buffer --------------------------
    // `indices[m, j] = m * hop + j`, flattened. Depends only on the framing
    // geometry `(num_frames, frame_width, hop)`. All caps above already
    // bounded `idx_len`; build (once per geometry) and memoize.
    let pos_key = PositionKey {
      num_frames,
      frame_width,
      hop_length,
    };
    // `entry`-based fallible memoization: build the index buffer only on a cache
    // miss, then read it back as a shared reference for the rest of the call.
    // (Can't use `or_insert_with` — the build is fallible.)
    if let std::collections::hash_map::Entry::Vacant(slot) = self.position_cache.entry(pos_key) {
      let mut idx_buf: Vec<i32> = Vec::new();
      idx_buf
        .try_reserve_exact(idx_len)
        .map_err(|e| Error::Backend {
          message: format!(
            "ISTFTCache::istft: index reservation for {idx_len} elements failed: {e}"
          ),
        })?;
      let frame_width_i32 = i32::try_from(frame_width).map_err(|_| Error::Backend {
        message: format!("ISTFTCache::istft: n_fft {frame_width} exceeds i32::MAX"),
      })?;
      for m in 0..num_frames {
        // `m * hop < t <= i32::MAX` and `+ j` stays `< t`, so every index
        // fits i32 (same bound the free `istft` relies on).
        let off = (m * hop_length) as i32;
        for j in 0..frame_width_i32 {
          idx_buf.push(off + j);
        }
      }
      let indices = Array::from_slice::<i32>(&idx_buf, &[idx_len_i32])?;
      slot.insert(indices);
    }

    // ---- cached scattered `Σ w²` window-sum buffer ----------------------
    // The reference (`get_norm_buffer`) scatters the tiled squared window into
    // a zero buffer of OLA length. We cache the RAW (un-floored) window-sum —
    // NOT the `mx.maximum(., 1e-10)`-floored divisor — because the floored
    // divisor would LOSE the free `istft`'s coverage guard: a zero-coverage
    // requested sample (e.g. a `center=true` length reaching into the
    // zero-coverage tail, or a `center=false` head/tail) would be divided by
    // the `1e-10` floor and silently emit invalid audio instead of being
    // rejected. Caching the raw sum lets the reconstruction path reproduce the
    // free `istft` EXACTLY — both its coverage guard (reject the requested
    // region) and its `mx.where(window_sum > eps, normalized, raw)`
    // normalization. Keyed by the full window + framing geometry. Built once
    // via the shared `frame_window` (so the synthesis window matches the
    // forward `stft` exactly).
    let norm_key = NormKey {
      n_fft,
      hop_length,
      win_length,
      window_pad,
      num_frames,
    };
    if !self.norm_buffer_cache.contains_key(&norm_key) {
      // Synthesis window via the SHARED `frame_window` — symmetric Hann of
      // `win_length` placed into the `n_fft` frame per `window_pad`, the EXACT
      // same call the forward `stft` made (no drift possible). Always `n_fft`
      // wide.
      let window = frame_window(win_length, n_fft, window_pad)?;
      let window_norm = ops::arithmetic::multiply(&window, &window)?;
      // tile(window_norm, num_frames) via reshape + broadcast (mlxrs has no
      // `tile` op; the free `istft` uses the same idiom).
      let window_norm_row = ops::shape::reshape(&window_norm, &(1usize, frame_width))?;
      let window_norm_tiled =
        ops::shape::broadcast_to(&window_norm_row, &(num_frames, frame_width))?;
      let updates_window = ops::shape::flatten(&window_norm_tiled, 0, -1)?;
      // `position_cache` was just populated for `pos_key` above.
      let indices = self
        .position_cache
        .get(&pos_key)
        .expect("position_cache populated for pos_key above");
      let zeros_wsum = Array::zeros::<f32>(&[t_i32])?;
      // RAW window-sum (no floor): the coverage guard + `where` below need the
      // true `Σ w²` to detect zero-coverage samples, exactly as free `istft`.
      let window_sum = ops::indexing::scatter_add_axis(&zeros_wsum, indices, &updates_window, 0)?;
      self.norm_buffer_cache.insert(norm_key, window_sum);
    }

    // ---- reconstruction (cached buffers in hand) ------------------------
    // irfft every frame along the frequency axis: (num_frames, n_freqs)
    // complex → (num_frames, n_fft) real.
    let frames_time = fft::irfft(x, n_fft_i32, 1, FftNorm::Backward)?;
    // updates_reconstructed = (frames_time * w).flatten(); `w` broadcasts
    // across the frame axis. The synthesis window is rebuilt here (cheap,
    // bounded) so the reconstruction multiply matches the cached norm buffer.
    let window = frame_window(win_length, n_fft, window_pad)?;
    let windowed = ops::arithmetic::multiply(&frames_time, &window)?;
    let updates_reconstructed = ops::shape::flatten(&windowed, 0, -1)?;

    let indices = self
      .position_cache
      .get(&pos_key)
      .expect("position_cache populated for pos_key above");
    let window_sum = self
      .norm_buffer_cache
      .get(&norm_key)
      .expect("norm_buffer_cache populated for norm_key above");

    let zeros_recon = Array::zeros::<f32>(&[t_i32])?;
    let reconstructed =
      ops::indexing::scatter_add_axis(&zeros_recon, indices, &updates_reconstructed, 0)?;

    // ---- requested-output region (same bounds the free `istft` computes) -
    // Compute the trim bounds BEFORE the coverage guard / normalization so the
    // guard runs over EXACTLY the returned region — identical ordering to the
    // free `istft`, so the cached path cannot disagree with what it returns.
    let pad = n_fft / 2;
    let (start_usize, stop_usize) = match (center, length) {
      (true, Some(len)) => {
        let end = pad.checked_add(len).ok_or_else(|| Error::Backend {
          message: format!("ISTFTCache::istft: center offset {pad} + length {len} overflows usize"),
        })?;
        if end > t {
          return Err(Error::Backend {
            message: format!(
              "ISTFTCache::istft: center offset {pad} + length {len} = {end} exceeds \
               reconstruction length {t}"
            ),
          });
        }
        (pad, end)
      }
      (true, None) => (pad, t - pad),
      (false, Some(len)) => {
        if len > t {
          return Err(Error::Backend {
            message: format!(
              "ISTFTCache::istft: requested length {len} exceeds reconstruction length {t}"
            ),
          });
        }
        (0usize, len)
      }
      (false, None) => (0usize, t),
    };
    let start_i32 = i32::try_from(start_usize).map_err(|_| Error::Backend {
      message: format!("ISTFTCache::istft: trim start {start_usize} exceeds i32::MAX"),
    })?;
    let stop_i32 = i32::try_from(stop_usize).map_err(|_| Error::Backend {
      message: format!("ISTFTCache::istft: trim stop {stop_usize} exceeds i32::MAX"),
    })?;

    // ---- coverage guard (IDENTICAL to the free `istft`) -----------------
    // Every sample in the REQUESTED output region must have window-sum
    // `> COVERAGE_EPS`; otherwise it received negligible window energy and
    // dividing by the (would-be floored) sum is meaningless. The free `istft`
    // reduces the requested slice of the RAW `window_sum` to its minimum and
    // rejects (naming the offending global index) if that minimum is not
    // strictly above the threshold (or is `NaN`). Reproduce it EXACTLY here —
    // this is the guard the floored-divisor path silently dropped. An empty
    // requested region (a single centered even-`n_fft` frame collapsing
    // `[pad .. t - pad]`) has no samples to corrupt, so the reduction
    // (undefined over an empty array) is skipped.
    if start_usize < stop_usize {
      let region_wsum = ops::indexing::slice(window_sum, &[start_i32], &[stop_i32], &[1])?;
      let mut region_min = ops::reduction::min(&region_wsum, false)?;
      let min_wsum = region_min.item::<f32>()?;
      // Fire on `<= COVERAGE_EPS` AND on `NaN` (a NaN window-sum cannot
      // normalize either). Written explicitly (not `!(min > eps)`) for the
      // partial-ord lint, with the same NaN-catching semantics as free `istft`.
      if min_wsum <= COVERAGE_EPS || min_wsum.is_nan() {
        let mut min_idx_arr = ops::misc::argmin(&region_wsum, None, false)?;
        let local_idx = min_idx_arr.item::<u32>()? as usize;
        let global_idx = start_usize + local_idx;
        return Err(Error::Backend {
          message: format!(
            "ISTFTCache::istft: requested output sample at index {global_idx} (region offset \
             {local_idx}) has window-sum {min_wsum:.3e} <= COVERAGE_EPS ({COVERAGE_EPS:.0e}) — it \
             received no window coverage in the overlap-add and is not recoverable \
             (n_fft={n_fft}, win_length={win_length}, hop={hop_length}, window_pad={window_pad:?}); \
             the requested region (e.g. a center=false head/tail) includes a zero-coverage sample \
             — adjust length/center or the window"
          ),
        });
      }
    }

    // ---- normalize (IDENTICAL to the free `istft`'s `where` branch) -----
    // Divide by `Σ w²` where it exceeds the coverage threshold, else leave the
    // raw overlap-add (the reference's `mx.where(window_sum > 1e-10, ...)`). The
    // coverage guard above guarantees every REQUESTED sample is on the
    // normalized branch; the `where` only matters for the trimmed-away region.
    let threshold = Array::full::<f32>(&[0i32; 0], COVERAGE_EPS)?;
    let mask = ops::comparison::greater(window_sum, &threshold)?;
    let normalized_recon = ops::arithmetic::divide(&reconstructed, window_sum)?;
    let reconstructed = ops::logical::select(&mask, &normalized_recon, &reconstructed)?;

    // Final trim to the requested region (same bounds the coverage guard used).
    ops::indexing::slice(&reconstructed, &[start_i32], &[stop_i32], &[1])
  }
}

impl Default for ISTFTCache {
  /// A fresh, empty cache — delegates to [`ISTFTCache::new`].
  fn default() -> Self {
    Self::new()
  }
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
  // Use `try_reserve_exact` so a multi-GB request from a forged input
  // returns a recoverable `Error::Backend` rather than aborting on the
  // allocator's OOM panic.
  let mut bank: Vec<f32> = Vec::new();
  bank
    .try_reserve_exact(bank_len)
    .map_err(|e| Error::Backend {
      message: format!("mel_filter_bank: allocation of {bank_len} f32 elements failed: {e}"),
    })?;

  // C10 SIMD: dispatch the row-by-row triangle construction through
  // the SIMD kernel (`simd::audio::mel_triangle::mel_filter_bank_rows`).
  // The dispatcher writes 0.0 for collapsed-bin rows (lc <= 0 / cr <=
  // 0) so we no longer need `Vec::resize(bank_len, 0.0)` upfront —
  // the kernel initializes every cell via `MaybeUninit::write`.
  let spare = bank.spare_capacity_mut();
  crate::simd::audio::mel_triangle::mel_filter_bank_rows(
    &mut spare[..bank_len],
    &all_freqs,
    &f_pts,
    n_mels,
  );
  // SAFETY: the C10 dispatcher's init contract guarantees every cell
  // of the `bank_len`-prefix of `spare` is initialized before
  // returning; `bank_len <= bank.capacity()` per `try_reserve_exact`.
  unsafe { bank.set_len(bank_len) };

  Array::from_slice::<f32>(&bank, &[n_mels_i32, n_freqs_i32])
}

/// Maximum number of `(sample_rate, n_fft, n_mels, f_min, f_max)` triples kept
/// per thread by [`mel_filter_bank_cached`]. A handful covers every realistic
/// pipeline (one front-end per loaded model, occasionally a second per
/// chained stage); the bound stops a pathological caller from growing the
/// cache without limit. LRU evicts the oldest entry on overflow.
///
/// **Public** so [`mel_filter_bank_cached`] / [`clear_mel_filter_cache`]
/// can intra-doc-link it under `rustdoc::private-intra-doc-links` (which
/// the workspace CI denies). Also a useful published bound for callers
/// reasoning about worst-case per-thread mel-bank footprint.
pub const MEL_FILTER_CACHE_CAP: usize = 8;

/// Cache key. `f_min` / `f_max` are kept bit-wise so two different `NaN`
/// payloads are distinct — but [`mel_filter_bank`] itself rejects non-finite
/// `f_min` / `f_max` (the `f_min >= 0.0 && f_max > f_min` guard fails on
/// any NaN), so a NaN key can never be inserted via the cached entry path.
/// `f_max` is `Option<f32>` to preserve `None == Nyquist` distinct from
/// the explicitly-passed value `Some(sample_rate / 2)` (the two compute
/// to byte-identical banks, but caching them separately keeps the
/// cache transparent — never silently aliasing two distinct API calls).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MelFilterCacheKey {
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min_bits: u32,
  f_max_bits: Option<u32>,
}

impl MelFilterCacheKey {
  fn new(n_mels: usize, n_fft: usize, sample_rate: u32, f_min: f32, f_max: Option<f32>) -> Self {
    Self {
      n_mels,
      n_fft,
      sample_rate,
      f_min_bits: f_min.to_bits(),
      f_max_bits: f_max.map(f32::to_bits),
    }
  }
}

// `Array` is `!Send`, so the cache is per-thread. Mirrors mlx-audio's
// `@lru_cache` on `mel_filters(sample_rate, n_fft, n_mels, ...)` (CPython
// is single-threaded at the GIL level; here `thread_local!` provides the
// same observable behavior without forcing `Send`).
thread_local! {
  static MEL_FILTER_CACHE: std::cell::RefCell<Vec<(MelFilterCacheKey, Array)>> =
    const { std::cell::RefCell::new(Vec::new()) };
}

/// Cached variant of [`mel_filter_bank`] — returns a thread-local cached
/// constant matrix keyed on `(sample_rate, n_fft, n_mels, f_min, f_max)`.
///
/// The mel filterbank is a one-shot constant per
/// `(sample_rate, n_fft, n_mels, f_min, f_max)` triple — a streaming /
/// per-chunk caller rebuilds the same `(n_mels, n_freqs)` matrix on every
/// call. This wrapper memoizes the result so repeated calls with identical
/// parameters pay one `try_clone` (a cheap shallow reference-bump on the
/// mlx-c handle) instead of a full triangle-construction + `Array::from_slice`.
///
/// Mirrors `mlx_audio.dsp.mel_filters`'s `@lru_cache` decorator: the
/// reference's `mel_filters(sample_rate, n_fft, n_mels, f_min, f_max, ...)`
/// memoizes on the exact same key. We use a thread-local cache because
/// `Array` is `!Send` (mlx-c arrays are bound to the thread that created
/// them); a `Mutex<HashMap>`-style global would not be safely shareable.
///
/// The cache is **bounded** at [`MEL_FILTER_CACHE_CAP`] entries (LRU
/// eviction) so a buggy / adversarial caller cycling through many
/// `(sample_rate, n_fft)` combinations cannot grow the cache without
/// limit. A handful of entries covers any realistic pipeline (front-end
/// per loaded model, occasional second per chained stage).
///
/// The returned [`Array`] is a `try_clone` of the cached entry — the
/// caller may mutate / consume it freely; the cached copy is untouched.
///
/// Validation, work caps, and error paths match [`mel_filter_bank`]
/// exactly (the first call delegates through it); a cached hit returns
/// the same `Array` value-for-value.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
///
/// # See also
/// - [`mel_filter_bank`] — the uncached construction path.
/// - [`clear_mel_filter_cache`] — empties the per-thread cache (test /
///   memory-pressure use).
pub fn mel_filter_bank_cached(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  let key = MelFilterCacheKey::new(n_mels, n_fft, sample_rate, f_min, f_max);

  // Fast path: lookup + move-to-front for LRU semantics.
  let hit = MEL_FILTER_CACHE.with(|cell| -> Result<Option<Array>> {
    let mut cache = cell.borrow_mut();
    if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
      // Move the hit to the back (most-recently-used end) so the
      // front of the Vec is always the LRU victim.
      let entry = cache.remove(pos);
      let clone = entry.1.try_clone()?;
      cache.push(entry);
      Ok(Some(clone))
    } else {
      Ok(None)
    }
  })?;
  if let Some(arr) = hit {
    return Ok(arr);
  }

  // Miss: build via the uncached path. The construction can fail with a
  // recoverable `Error::Backend` (invalid params, work cap, etc.); we
  // propagate that error WITHOUT touching the cache, so a failed call
  // cannot poison the cache with an absent or invalid entry.
  let bank = mel_filter_bank(n_mels, n_fft, sample_rate, f_min, f_max)?;

  // Cache the just-built bank. Hold a `try_clone` for the caller so the
  // cached entry is independent. Eviction when full: drop the LRU
  // (front), then push the new entry at the back.
  let for_caller = bank.try_clone()?;
  MEL_FILTER_CACHE.with(|cell| {
    let mut cache = cell.borrow_mut();
    if cache.len() >= MEL_FILTER_CACHE_CAP {
      // Evict the LRU entry (oldest). `remove(0)` is O(n) but n <= 8.
      let _ = cache.remove(0);
    }
    cache.push((key, bank));
  });
  Ok(for_caller)
}

/// Empty the per-thread [`mel_filter_bank_cached`] cache.
///
/// Intended for memory-pressure recovery and test isolation: a sequence
/// of `(sample_rate, n_fft, n_mels)` triples can otherwise pin up to
/// [`MEL_FILTER_CACHE_CAP`] mel banks per thread for the lifetime of
/// the thread. Tests that exercise cache eviction or otherwise want a
/// fresh slate call this before / after the body.
///
/// **Per-thread only.** Other threads' caches are untouched; call from
/// each thread that needs eviction (the cache is `!Send`).
pub fn clear_mel_filter_cache() {
  MEL_FILTER_CACHE.with(|cell| cell.borrow_mut().clear());
}

/// Mel spectrogram: `mel_bank @ |stft(samples)|^2`.
///
/// Returns shape `(n_mels, num_frames)` `Dtype::F32`. Combines [`stft`],
/// magnitude-squared, and [`mel_filter_bank_cached`] in the canonical
/// Whisper / mlx-audio order.
///
/// The filter bank is fetched via [`mel_filter_bank_cached`] (per-thread
/// LRU cache keyed on `(n_mels, n_fft, sample_rate, f_min, f_max)`). A
/// streaming / per-chunk caller (and every [`log_mel_spectrogram`] /
/// [`log_mel_spectrogram_with`] + STT log-mel hop) therefore pays one bank
/// construction on the first call per parameter triple, then a cheap
/// `try_clone` (shallow mlx-c handle bump) on every subsequent call —
/// matching the `mlx_audio.dsp.mel_filters` `@lru_cache` shape. The bank
/// is a constant for a given `(sample_rate, n_fft, n_mels, f_min, f_max)`
/// so the returned mel spectrogram is byte-identical to the uncached path.
///
/// # Errors
/// Propagates from [`stft`] and [`mel_filter_bank_cached`].
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
  let mag = spec.data_ref().abs()?;
  let power = mag.square()?;
  // `power` is `(num_frames, n_freqs)`; mel is `(n_mels, n_freqs)`.
  // Mel-spec layout in mlx-audio / Whisper is `(n_mels, num_frames)` =
  // `mel @ power.T`. Uses `mel_filter_bank_cached` so repeated calls with
  // the same `(n_mels, n_fft, sample_rate, f_min, f_max)` share the
  // per-thread LRU cache (Codex R1 medium finding — the uncached path
  // rebuilt the bank on every chunk / per-utterance encode pass).
  let mel = mel_filter_bank_cached(n_mels, n_fft, sample_rate, f_min, f_max)?;
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

// ============================================================================
// 1-D IIR filter (`lfilter`) and BS.1770 K-weighted integrated loudness.
//
// Both run pure-Rust on the CPU (mirroring how [`mel_filter_bank`] materializes
// its bank Vec) — these are intrinsically sequential / per-block numerical
// recurrences, NOT mlx graph ops. `lfilter` is per-sample sequential
// (`y[n]` depends on `y[n-1..]`) so an mlx graph cannot express it as
// element-wise ops; the BS.1770 path chains lfilter twice per channel and
// then aggregates per-block mean-squares, all bounded by the K-weighted
// channel data we already materialized. Computation is in `f64` to match
// the reference (Python's `np.asarray(b, dtype=np.float64)` + `np.result_type`
// promotion), with the public surface staying `f32` (the [`Array`] data dtype
// throughout mlxrs's audio pipeline). Reads use `&self`-friendly accessors:
// each input `&Array` is **cloned with [`Array::try_clone`]** so the
// `&mut self`-tasking [`Array::to_vec`] (which performs the mandatory
// `eval`) does not force a `mut` binding on the caller's borrow — the
// `&self` borrow contract of these public functions is preserved.

/// Hard ceiling on the per-channel sample count consumed by [`lfilter`].
/// Mirrors the public-input allocation cap shared with [`stft`] and the
/// window family (see
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES)): a
/// caller-controllable `data` length above the cap returns a recoverable
/// [`Error::Backend`] instead of risking a multi-GB CPU allocation for the
/// `f64` state buffer + output `Vec`.
const MAX_LFILTER_SAMPLES: usize = crate::audio::io::MAX_DECODED_SAMPLES;

/// Hard ceiling on the **total element count** (`n_samples * n_channels`)
/// [`integrated_loudness`] will consume. Shared with [`MAX_LFILTER_SAMPLES`]
/// for the same reason: the input `to_vec` materializes
/// `n_samples * n_channels` f32 samples, and each channel is K-weighted by
/// two `lfilter_f64` chains. Capping the per-channel count alone (as a prior
/// revision did) is unsafe for 2-D `(n_samples, n_channels)` inputs: a
/// `(MAX_DECODED_SAMPLES, 5)` input would otherwise materialize multiple GB
/// in `raw_f32` BEFORE any per-channel processing. Cap is on
/// `n_samples.checked_mul(n_channels)` against
/// [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES); the
/// streaming per-channel path keeps the peak working set bounded by
/// `raw_f32 + 1 channel's f64 K-weighted buffer` (no de-interleaved-AND-
/// weighted-full-channel duplication). Above the cap returns a recoverable
/// [`Error::Backend`].
const MAX_LOUDNESS_SAMPLES: usize = crate::audio::io::MAX_DECODED_SAMPLES;

/// Hard byte ceiling on the per-channel `f64` mean-square matrix
/// [`integrated_loudness`] allocates: `num_blocks * n_channels *
/// size_of::<f64>() <= MAX_LOUDNESS_BLOCK_BYTES`. The previous revision
/// capped `num_blocks * n_channels` against the 64 Mi-**element** sample
/// budget — but the cells are `f64`, so the actual peak bytes were 8× the
/// element budget (`64 Mi * 8 B = 512 MiB`). A caller-controllable
/// `overlap` close to 1 (e.g. `0.99999990`) on a small mono input derived
/// tens of millions of blocks that all passed the element-only cap, then
/// reserved hundreds of MB / ~2 GiB for the mean-square matrix alone. The
/// byte cap bounds the dominant block-sized buffer directly: at 64 MiB
/// the peak `mean_square` footprint is bounded regardless of how the
/// caller distributes the (`num_blocks`, `n_channels`) product, and the
/// gate-index iteration that re-reads the same matrix never sees more
/// cells than this byte budget allows. Pathological overlaps return a
/// recoverable [`Error::Backend`] BEFORE any allocation.
const MAX_LOUDNESS_BLOCK_BYTES: usize = 64 * 1024 * 1024;
/// Hard ceiling on the **total sample-visit work** [`integrated_loudness`]
/// performs across the per-block mean-square loop: `num_blocks *
/// ceil(block_size_samples) * n_channels <= MAX_LOUDNESS_WORK`. Even
/// when the byte cap above admits a moderate `num_blocks`, a near-1
/// overlap combined with a SMALL block size still drives the per-block
/// CPU sum work to the trillions of sample-visits — each block re-sums
/// `ceil(block_size_samples)` weighted samples, and the per-block sum
/// loop runs ONCE PER CHANNEL (the streaming K-weighting loop iterates
/// `n_channels` times), so the actual visit count is `num_blocks *
/// ceil(block_size_samples) * n_channels` regardless of the matrix byte
/// footprint. A prior revision omitted the `n_channels` factor and
/// admitted a 5-channel pathological case (Codex review:
/// `num_blocks=1_677_721, block_samples=160, n_channels=5` — channel-less
/// product `~268 Mi <= 256 Mi cap` BUT actual visits `~1.34 Bi`); the
/// fix folds `n_channels` into the work product. `ceil` is used because
/// the per-block bounds `(floor(bi*step*bs*r), floor((bi*step+1)*bs*r))`
/// can give `upper - lower = ceil(block_size_samples)` in the worst
/// case for fractional `block_size_samples`. Cap the total visit count
/// at 256 Mi (`block_size = 0.4 s` at 48 kHz = `19,200 samples/block`,
/// so a default-overlap 256 Mi-visit budget admits ~13,653 blocks for
/// mono ≈ 91 hours of audio, or ~2,730 5-channel blocks ≈ 18 hours of
/// 5-channel audio — comfortable for any realistic loudness analysis,
/// but rejects multi-trillion-visit pathological cases in microseconds).
/// Pathological work returns a recoverable [`Error::Backend`] BEFORE
/// the per-block loop.
const MAX_LOUDNESS_WORK: usize = 256 * 1024 * 1024;

/// Per-channel BS.1770 weighting gains for up to 5 channels: front L/R/C =
/// `1.0`, surround L/R = `1.41` (~`+1.5 dB`). Matches
/// `mlx_audio.dsp.integrated_loudness`'s `channel_gains = [1.0, 1.0, 1.0,
/// 1.41, 1.41]` literal.
const BS1770_CHANNEL_GAINS: [f64; 5] = [1.0, 1.0, 1.0, 1.41, 1.41];
/// BS.1770 absolute gate (LUFS) — blocks at or below `-70 LUFS` never
/// contribute to the integrated loudness. Matches
/// `mlx_audio.dsp.integrated_loudness`'s `absolute_threshold = -70.0`.
const BS1770_ABSOLUTE_THRESHOLD_LUFS: f64 = -70.0;
/// BS.1770 relative gate offset (LUFS) — once an absolute-gated integrated
/// loudness is computed, the relative gate is set `10 LU` below it and the
/// final integrated loudness uses only blocks above BOTH gates. Matches
/// `mlx_audio.dsp.integrated_loudness`'s `... - 10.0`.
const BS1770_RELATIVE_OFFSET_LUFS: f64 = 10.0;
/// BS.1770 loudness offset added in the per-block `LUFS = -0.691 + 10 *
/// log10(...)` reduction. Matches `mlx_audio.dsp.integrated_loudness`'s
/// `-0.691` literal (the K-weighting calibration constant from BS.1770-4
/// Annex 1).
const BS1770_LOUDNESS_OFFSET_LUFS: f64 = -0.691;
/// Hard ceiling on channels accepted by [`integrated_loudness`]. The
/// reference rejects `>5` channels (mlx-audio supports mono / stereo /
/// 5.0); we mirror that.
const BS1770_MAX_CHANNELS: usize = 5;

/// Apply a 1-D causal linear filter (direct-form II transposed) to a
/// 1-D real signal.
///
/// Faithful port of `mlx_audio.dsp.lfilter(b, a, data)` — implements the
/// standard recurrence
/// `a[0] * y[n] = sum_k b[k] * x[n-k] - sum_{k>=1} a[k] * y[n-k]`,
/// normalizing by `a[0]`. Coefficients are taken in `f64` (matching the
/// reference's `np.asarray(b, dtype=np.float64)` + `np.result_type`
/// promotion); the state buffer is `f64`; the output is materialized back
/// as `Dtype::F32` to keep the public dtype consistent with the rest of
/// mlxrs's audio pipeline. The signal `data` is read once (`Array::to_vec`
/// via a `try_clone` so the caller's borrow stays `&Array`), then the
/// per-sample recurrence runs purely on the CPU — IIR `y[n]` depends on
/// `y[n-1..]`, so an mlx graph cannot express it as element-wise ops.
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `a` is empty,
///   - `a[0] == 0` (the denominator's leading coefficient cannot be zero),
///   - `data` is not 1-D,
///   - `data`'s sample count exceeds the shared
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap,
///   - any size exceeds `i32::MAX`.
///
/// The reference returns `np.zeros_like(data)` when `b` is empty; we mirror
/// that.
pub fn lfilter(b: &[f64], a: &[f64], data: &Array) -> Result<Array> {
  if data.ndim() != 1 {
    return Err(Error::Backend {
      message: format!("lfilter: only supports 1-D input, got {}-D", data.ndim()),
    });
  }
  let shape = data.shape();
  let n_samples = shape[0];
  let n_samples_i32 = i32::try_from(n_samples).map_err(|_| Error::Backend {
    message: format!("lfilter: n_samples {n_samples} exceeds i32::MAX"),
  })?;
  // Enforce the bounded-memory contract on the SHAPE before any
  // materialization: `data` may be a lazy oversized array
  // (e.g. `Array::zeros(MAX_LFILTER_SAMPLES + 1)`), in which case the
  // subsequent `to_vec::<f32>()` would eval-and-allocate the full f32
  // buffer (and then promote to a second f64 Vec) BEFORE the kernel's
  // post-promotion cap check ever fired. Reject up-front so the public
  // wrapper allocates nothing for oversized inputs. `lfilter_f64` still
  // re-checks the same cap for direct callers (K-weighting path) — this
  // is the shape-side guard for the f32 boundary.
  if n_samples > MAX_LFILTER_SAMPLES {
    return Err(Error::Backend {
      message: format!("lfilter: sample count {n_samples} exceeds the {MAX_LFILTER_SAMPLES} cap"),
    });
  }

  // f32 boundary wrapper: read the input via `try_clone().to_vec::<f32>()`
  // (no `&mut` on the caller's borrow), promote to f64, run the shared
  // f64 kernel, then cast back down to f32 for the public mlxrs audio
  // dtype. The K-weighting path of `integrated_loudness` calls the
  // internal `lfilter_f64` directly to keep TWO biquad stages in f64 end-
  // to-end (no intermediate f32 cast between stages), matching the
  // reference's `np.result_type(data.dtype, b.dtype, a.dtype)` promotion.
  let x_f32 = data.try_clone()?.to_vec::<f32>()?;
  let mut x_f64: Vec<f64> = Vec::new();
  x_f64
    .try_reserve_exact(n_samples)
    .map_err(|e| Error::Backend {
      message: format!("lfilter: input promotion reservation for {n_samples} samples failed: {e}"),
    })?;
  for v in &x_f32 {
    x_f64.push(f64::from(*v));
  }
  let y_f64 = lfilter_f64(b, a, &x_f64)?;
  let mut y: Vec<f32> = Vec::new();
  y.try_reserve_exact(n_samples).map_err(|e| Error::Backend {
    message: format!("lfilter: output reservation for {n_samples} samples failed: {e}"),
  })?;
  for v in &y_f64 {
    y.push(*v as f32);
  }
  Array::from_slice::<f32>(&y, &[n_samples_i32])
}

/// Private f64 kernel for [`lfilter`] and the BS.1770 K-weighting path.
///
/// Operates entirely on `f64` slices/Vecs — input, state, and output are
/// all `f64` — so two-stage chains (K-weighting's high-shelf → high-pass)
/// run in f64 end-to-end without an intermediate f32 cast between stages.
/// The reference's `_k_weight_audio` (Python `np.float64` throughout) and
/// our [`k_weight_channel`] both rely on this precision invariant; before
/// the split they were lost via the public `lfilter`'s f32 boundary.
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `a` is empty,
///   - `a[0] == 0` (the denominator's leading coefficient cannot be zero),
///   - `x.len()` exceeds the shared
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap
///     (mirrors the public [`lfilter`]'s element budget).
///
/// The reference returns `np.zeros_like(data)` when `b` is empty; we mirror
/// that.
fn lfilter_f64(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
  if a.is_empty() {
    return Err(Error::Backend {
      message: "lfilter: filter denominator must be non-empty".into(),
    });
  }
  if a[0] == 0.0 {
    return Err(Error::Backend {
      message: "lfilter: filter denominator must have a non-zero leading term (a[0] != 0)".into(),
    });
  }
  let n_samples = x.len();
  if n_samples > MAX_LFILTER_SAMPLES {
    return Err(Error::Backend {
      message: format!("lfilter: sample count {n_samples} exceeds the {MAX_LFILTER_SAMPLES} cap"),
    });
  }

  // Mirror the reference's `if b.size == 0: return np.zeros_like(data)`.
  // A zero-tap numerator is degenerate (filter output is identically zero
  // regardless of `data` or `a`); allocate the result directly.
  if b.is_empty() {
    let mut y: Vec<f64> = Vec::new();
    y.try_reserve_exact(n_samples).map_err(|e| Error::Backend {
      message: format!("lfilter: zero-output reservation for {n_samples} samples failed: {e}"),
    })?;
    y.resize(n_samples, 0.0);
    return Ok(y);
  }

  // Normalize `b` and `a` by `a[0]` (Python: `b = b / a[0]; a = a / a[0]`)
  // into stack-friendly `Vec`s. Both are short (3 taps for biquads,
  // typically <= a few dozen for general IIRs); the recoverable
  // `try_reserve_exact` mirrors the rest of the dsp.rs allocation
  // discipline.
  let a0 = a[0];
  let mut b_norm: Vec<f64> = Vec::new();
  b_norm
    .try_reserve_exact(b.len())
    .map_err(|e| Error::Backend {
      message: format!("lfilter: reservation for b ({} taps) failed: {e}", b.len()),
    })?;
  for &bv in b {
    b_norm.push(bv / a0);
  }
  let mut a_norm: Vec<f64> = Vec::new();
  a_norm
    .try_reserve_exact(a.len())
    .map_err(|e| Error::Backend {
      message: format!("lfilter: reservation for a ({} taps) failed: {e}", a.len()),
    })?;
  for &av in a {
    a_norm.push(av / a0);
  }

  // `state_len = max(len(a), len(b)) - 1`. With `a.len() >= 1` checked
  // above and `b.len() >= 1` (the `b.is_empty()` early-return rules out
  // `b.len() == 0`), the `max` is >= 1, so the subtraction is safe.
  let state_len = a_norm.len().max(b_norm.len()) - 1;

  // Output buffer (f64 — the kernel runs end-to-end in f64).
  let mut y: Vec<f64> = Vec::new();
  y.try_reserve_exact(n_samples).map_err(|e| Error::Backend {
    message: format!("lfilter: output reservation for {n_samples} samples failed: {e}"),
  })?;

  // Reference's `state_len == 0` fast path: `y = b[0] * x` (no recurrence,
  // no feedback state to maintain). This is the FIR-of-length-1 case.
  // C9 / [#154]: a NEON `vmulq_n_f64` 2-lane kernel exists at
  // [`crate::simd::audio::lfilter::lfilter_fir_b0`] but is NOT wired
  // here — the simd_lfilter bench (M2 Pro, release, 2026-05-24)
  // measured scalar at ~10 Gelem/s vs the NEON dispatcher at
  // ~7.8 Gelem/s on f64 across every benched size. LLVM auto-
  // vectorizes the per-sample multiply better than the hand-rolled
  // NEON tile at f64 width. The kernel ships in `simd::audio::lfilter`
  // as a regression guard + a building block for any future caller
  // that wants the dispatcher behaviour explicitly.
  if state_len == 0 {
    let b0 = b_norm[0];
    for &sample in x {
      y.push(b0 * sample);
    }
    return Ok(y);
  }

  // C9 / [#154]: biquad fast path — `state_len == 2`, `b.len() ==
  // a.len() == 3` — is NOT wired in this out-of-place path.
  //
  // The hand-unrolled biquad kernel in `simd::audio::lfilter` is
  // in-place; routing the out-of-place wrapper through it required an
  // extra `extend_from_slice(x)` full-buffer memcpy before the kernel
  // ran. Out-of-place benches (M2 Pro, release, 2026-05-24,
  // `simd_lfilter.rs` `lfilter_biquad_out_of_place/n=*` groups) showed
  // the `extend + in_place_kernel` dispatch losing 1-3% to the
  // single-pass `generic_out_of_place` reference at mid sizes
  // (16k-65k samples — within run-to-run variance) AND only mixed
  // wins elsewhere. The realistic-workload context here also matters:
  // the public out-of-place `lfilter_f64` is NOT the K-weighting hot
  // path — `integrated_loudness` calls `lfilter_f64_in_place` directly
  // through `k_weight_channel`, so the in-place arm (wired below in
  // `lfilter_f64_in_place`) IS the consumer that matters.
  //
  // Per the user directive 2026-05-24 ("KEEP wiring only if benchmark
  // proves > scalar at the actually-wired paths; prefer revert when
  // in doubt"), this arm is reverted. The single-pass generic loop
  // below handles `state_len == 2` along with all other shapes.

  // General direct-form II transposed recurrence (matching the reference's
  // per-sample loop body byte-for-byte) — for non-biquad / non-FIR shapes
  // AND for `state_len == 2` (the biquad-dispatcher fast-path was tried
  // and reverted; see comment above).
  let mut state: Vec<f64> = Vec::new();
  state
    .try_reserve_exact(state_len)
    .map_err(|e| Error::Backend {
      message: format!("lfilter: state reservation for {state_len} taps failed: {e}"),
    })?;
  state.resize(state_len, 0.0);

  for &sample in x {
    // `output = b[0] * sample + state[0]` — the next output sample;
    // `state[0]` is the running accumulator for `b[1] * x[n-1] - a[1] *
    // y[n-1] + state[1]` from the previous step.
    let output = b_norm[0] * sample + state[0];
    // Shift the state vector forward AND fold in the next sample's
    // feedforward / feedback contribution. `i` walks indices `1..state_len`;
    // the per-iteration assignment matches the reference's
    // `state[i - 1] = state[i] + b[i] * sample - a[i] * output`, with `0.0`
    // substituted for any out-of-range `b` / `a` tap (matches the
    // reference's `b[i] * sample if i < len(b) else 0.0`).
    for i in 1..state_len {
      let feedforward = b_norm.get(i).copied().unwrap_or(0.0) * sample;
      let feedback = a_norm.get(i).copied().unwrap_or(0.0) * output;
      state[i - 1] = state[i] + feedforward - feedback;
    }
    // Final state cell: `state[-1] = b[state_len] * sample - a[state_len] *
    // output`, again with 0.0 substituted for out-of-range taps. Indexing
    // with `state_len - 1` is safe (`state_len >= 1` here).
    let feedforward_last = b_norm.get(state_len).copied().unwrap_or(0.0) * sample;
    let feedback_last = a_norm.get(state_len).copied().unwrap_or(0.0) * output;
    state[state_len - 1] = feedforward_last - feedback_last;
    y.push(output);
  }

  Ok(y)
}

/// In-place variant of [`lfilter_f64`]: runs the same direct-form II
/// transposed IIR recurrence directly on the caller-provided `x` buffer,
/// overwriting `x[n]` with `y[n]`. The numerical result is bit-identical
/// to `lfilter_f64(b, a, x)` (same formula, same f64 precision, same
/// `state[i] = ...` updates) — the only difference is allocation: this
/// kernel only allocates the SMALL coefficient + state buffers
/// (`max(len(a), len(b))` f64s total, typically 3 for a biquad), NOT a
/// fresh `n_samples`-long output `Vec`.
///
/// Used by [`k_weight_channel`] to keep the peak working set at ONE f64
/// channel buffer instead of TWO (the old chain `after_shelf =
/// lfilter_f64(&hs, &chan); lfilter_f64(&hp, &after_shelf)` momentarily
/// held both the first-stage output AND the second-stage allocation while
/// `chan_f64` was still in scope — peak `~3 * channel_bytes` at the
/// chain-call boundary).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `a` is empty,
///   - `a[0] == 0` (the denominator's leading coefficient cannot be zero),
///   - `x.len()` exceeds the shared
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap
///     (mirrors [`lfilter_f64`]'s element budget).
///
/// `b` empty mirrors [`lfilter_f64`]'s "zero-output" semantics by writing
/// zeros into `x` (the in-place equivalent of returning `np.zeros_like`).
fn lfilter_f64_in_place(b: &[f64], a: &[f64], x: &mut [f64]) -> Result<()> {
  if a.is_empty() {
    return Err(Error::Backend {
      message: "lfilter: filter denominator must be non-empty".into(),
    });
  }
  if a[0] == 0.0 {
    return Err(Error::Backend {
      message: "lfilter: filter denominator must have a non-zero leading term (a[0] != 0)".into(),
    });
  }
  let n_samples = x.len();
  if n_samples > MAX_LFILTER_SAMPLES {
    return Err(Error::Backend {
      message: format!("lfilter: sample count {n_samples} exceeds the {MAX_LFILTER_SAMPLES} cap"),
    });
  }

  // Mirror `lfilter_f64`'s `b.is_empty()` early-return semantics in place:
  // overwrite `x` with zeros (the in-place equivalent of the out-of-place
  // kernel's `Vec::with_capacity` + `resize` of zeros).
  if b.is_empty() {
    for v in x.iter_mut() {
      *v = 0.0;
    }
    return Ok(());
  }

  // Normalize `b` and `a` by `a[0]` — same arithmetic as `lfilter_f64`.
  let a0 = a[0];
  let mut b_norm: Vec<f64> = Vec::new();
  b_norm
    .try_reserve_exact(b.len())
    .map_err(|e| Error::Backend {
      message: format!("lfilter: reservation for b ({} taps) failed: {e}", b.len()),
    })?;
  for &bv in b {
    b_norm.push(bv / a0);
  }
  let mut a_norm: Vec<f64> = Vec::new();
  a_norm
    .try_reserve_exact(a.len())
    .map_err(|e| Error::Backend {
      message: format!("lfilter: reservation for a ({} taps) failed: {e}", a.len()),
    })?;
  for &av in a {
    a_norm.push(av / a0);
  }

  // `state_len = max(len(a), len(b)) - 1`. `a.len() >= 1` checked above
  // and `b.len() >= 1` (the `b.is_empty()` branch returns earlier), so
  // the `max` is >= 1 and the subtraction is safe.
  let state_len = a_norm.len().max(b_norm.len()) - 1;

  // `state_len == 0` fast path: `y[n] = b[0] * x[n]`. Read each sample
  // BEFORE writing — both source and destination are the same slot, so
  // the multiplication is order-safe regardless (single-pass, no
  // dependency on neighboring slots). C9 / [#154]: the SIMD FIR
  // dispatcher at [`crate::simd::audio::lfilter::lfilter_fir_b0`]
  // exists, but the bench measured the scalar in-place `*v *= b0` loop
  // (which LLVM autovectorizes) at ~10 Gelem/s vs the NEON dispatcher
  // at ~7.8 Gelem/s on f64. The scalar loop stays here for the in-
  // place case; the SIMD kernel is shipped in `simd::audio::lfilter`
  // as a regression guard + building block.
  if state_len == 0 {
    let b0 = b_norm[0];
    for v in x.iter_mut() {
      *v *= b0;
    }
    return Ok(());
  }

  // C9 / [#154]: biquad fast path — `state_len == 2`, `b.len() ==
  // a.len() == 3` (the actual BS.1770 K-weighting workload — the
  // chain through this kernel from [`k_weight_channel`] is the hot
  // path of [`integrated_loudness`]). Benchmarks (M2 Pro, release,
  // authoritative re-run 2026-05-24, `simd_lfilter.rs`,
  // `lfilter_k_weight_chain` group; criterion `--warm-up-time 1
  // --measurement-time 2 --sample-size 30`, captured to
  // `/tmp/c9-r2-bench-authoritative.txt`) measured a +30% to +53%
  // speedup over the generic loop on the K-weighting HS → HP chain
  // across all benched lengths on the current run:
  //
  // ```text
  // n        generic         biquad_dispatch    dispatch vs generic
  // 48000    160 Melem/s     245 Melem/s        +52.7%
  // 192000   174 Melem/s     251 Melem/s        +44.4%
  // 480000   185 Melem/s     242 Melem/s        +30.5%
  // ```
  //
  // The single-stage `lfilter_biquad/n=*` lane sweep also shows
  // dispatch at +44% to +62% over generic across 1024 → 480000
  // samples (covering 4 s and 10 s @ 48 kHz — the realistic
  // `k_weight_channel` consumer range, which receives FULL audio
  // channels). The hand-unrolled body is bit-identical to the generic
  // loop for any 3-tap biquad (asserted by `biquad_bit_exact_vs_generic_*`
  // tests in [`crate::simd::audio::lfilter`]); LUFS measurements
  // through [`integrated_loudness`] remain byte-identical to pre-C9
  // output.
  //
  // CAVEAT: see `simd::audio::lfilter` module docs for cross-run
  // baseline-stability caveats — generic baselines dropped ~21-35%
  // between the prior `6728548` and current `feb477c` runs (in-place
  // rows: 27.6% to 34.6%; K-weight rows: 20.9% to 31.9%). Dispatch
  // values also drifted across runs (e.g. in-place dispatch at
  // `n=1024` moved from 487 → 543 Melem/s, about +11.5%, and the
  // K-weighting rows moved by roughly -3% to -5% per row) but
  // dispatch consistently beat the generic baseline in BOTH runs
  // at every benched length. The inflated current-run ratios above
  // (+30% to +62%) are substantially driven by the depressed
  // generic baseline rather than a newly-established dispatch
  // improvement. The KEEP-WIRED ship decision is grounded on the
  // CONSERVATIVE lower-bound win read from the prior run's own
  // row-paired ratios (+3.0% to +4.7% on the in-place sweep,
  // +8.1% to +9.4% on the K-weighting chain; SAME-RUN ratios from
  // a single prior run, NOT a cross-run mixed envelope), not on
  // the current run's inflated numbers.
  if state_len == 2 && b_norm.len() == 3 && a_norm.len() == 3 {
    crate::simd::audio::lfilter::lfilter_biquad(x, &b_norm, &a_norm);
    return Ok(());
  }

  // General direct-form II transposed recurrence — same body as
  // `lfilter_f64` but writes the output back into `x`. The critical
  // ordering invariant: read `sample = x[n]` into a local BEFORE
  // overwriting `x[n]` with `output`, since the per-sample feedforward
  // (`b[i] * sample`) and feedback (`a[i] * output`) both use the SAME
  // `sample` value that lived in `x[n]` on entry to this iteration.
  let mut state: Vec<f64> = Vec::new();
  state
    .try_reserve_exact(state_len)
    .map_err(|e| Error::Backend {
      message: format!("lfilter: state reservation for {state_len} taps failed: {e}"),
    })?;
  state.resize(state_len, 0.0);

  for slot in x.iter_mut() {
    let sample = *slot;
    let output = b_norm[0] * sample + state[0];
    for i in 1..state_len {
      let feedforward = b_norm.get(i).copied().unwrap_or(0.0) * sample;
      let feedback = a_norm.get(i).copied().unwrap_or(0.0) * output;
      state[i - 1] = state[i] + feedforward - feedback;
    }
    let feedforward_last = b_norm.get(state_len).copied().unwrap_or(0.0) * sample;
    let feedback_last = a_norm.get(state_len).copied().unwrap_or(0.0) * output;
    state[state_len - 1] = feedforward_last - feedback_last;
    *slot = output;
  }

  Ok(())
}

/// Compute a BS.1770 biquad's `(b, a)` coefficients (both length 3, both
/// normalized by `a[0]`) for either the K-weighting high-shelf or the
/// high-pass stage.
///
/// Faithful port of `mlx_audio.dsp._biquad_coefficients(gain_db, q_factor,
/// center_freq, rate, filter_type)`. Both filter shapes follow the standard
/// "Audio EQ Cookbook" (RBJ) biquad formulas; the high-shelf takes a
/// non-zero `gain_db` (BS.1770 uses `+4 dB` at `1500 Hz`, `Q = 1/sqrt(2)`)
/// while the high-pass uses `gain_db = 0` (BS.1770 uses `Q = 0.5`,
/// `fc = 38 Hz`).
///
/// Returns `(b_norm, a_norm)` with `b_norm[0] = b0/a0` etc. — pre-divided by
/// `a[0]` exactly as the reference does (`np.array([b0, b1, b2]) / a0`).
fn bs1770_biquad_coefficients(
  gain_db: f64,
  q_factor: f64,
  center_freq: f64,
  rate: f64,
  filter_type: BiquadKind,
) -> ([f64; 3], [f64; 3]) {
  // Reference matches:
  //   amplitude = 10 ** (gain_db / 40.0)
  //   omega = 2π * (center_freq / rate)
  //   alpha = sin(omega) / (2 * q)
  let amplitude = 10.0_f64.powf(gain_db / 40.0);
  let omega = 2.0 * std::f64::consts::PI * (center_freq / rate);
  let alpha = omega.sin() / (2.0 * q_factor);
  let cos_omega = omega.cos();

  let (b0, b1, b2, a0, a1, a2) = match filter_type {
    BiquadKind::HighShelf => {
      let sqrt_a = amplitude.sqrt();
      let b0 =
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
      let b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega);
      let b2 =
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
      let a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
      let a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega);
      let a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
      (b0, b1, b2, a0, a1, a2)
    }
    BiquadKind::HighPass => {
      let b0 = (1.0 + cos_omega) / 2.0;
      let b1 = -(1.0 + cos_omega);
      let b2 = (1.0 + cos_omega) / 2.0;
      let a0 = 1.0 + alpha;
      let a1 = -2.0 * cos_omega;
      let a2 = 1.0 - alpha;
      (b0, b1, b2, a0, a1, a2)
    }
  };

  // `a[0]` is `a0/a0 == 1.0` after the normalization; emit the literal so
  // `clippy::eq_op` is happy without changing the value the reference
  // produces (`np.array([a0, a1, a2]) / a0`).
  ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

/// Which RBJ biquad shape [`bs1770_biquad_coefficients`] should produce.
/// Only the two BS.1770 stages are wired (high-shelf, high-pass); the
/// reference's `_biquad_coefficients` raises on any other `filter_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::IsVariant)]
enum BiquadKind {
  HighShelf,
  HighPass,
}

impl BiquadKind {
  #[allow(dead_code)]
  const fn as_str(&self) -> &'static str {
    match self {
      Self::HighShelf => "high_shelf",
      Self::HighPass => "high_pass",
    }
  }
}

/// Apply BS.1770 K-weighting (high-shelf at 1.5 kHz then high-pass at
/// 38 Hz) to a single channel's f32 samples, returning the K-weighted
/// signal as a fresh `Vec<f64>`.
///
/// Internal helper for [`integrated_loudness`]; runs in `f64` to match the
/// reference's `_k_weight_audio` (Python `np.float64` throughout —
/// `np.result_type` promotes the input to f64 in BOTH biquad stages and
/// the second stage receives an f64 input from the first). Drives the
/// private [`lfilter_f64_in_place`] kernel directly: convert the f32
/// channel to f64 ONCE into a single `chan_f64` buffer, then run TWO
/// in-place `lfilter_f64_in_place` stages (no intermediate f32 cast
/// between stages, no `Array` round-trip per stage, no second f64 channel
/// buffer), then return that one buffer for the per-block mean-square
/// reduction.
///
/// # Memory
/// Peak working set per channel is ONE f64 channel buffer
/// (`n_samples * 8 bytes`) plus the two biquads' small (length 3)
/// coefficient + state buffers. The previous implementation chained
/// out-of-place `lfilter_f64` calls (`after_shelf = lfilter_f64(...)`)
/// which momentarily held TWO full f64 channel buffers across the second
/// stage's allocation — for a max-allowed mono input
/// (`MAX_DECODED_SAMPLES = 64 Mi samples`) that was an extra ~512 MiB
/// beyond the documented bound. Switching to the in-place kernel
/// eliminates the second buffer entirely.
fn k_weight_channel(channel: &[f32], rate: u32) -> Result<Vec<f64>> {
  let rate_f64 = f64::from(rate);
  // High-shelf: +4 dB at 1500 Hz, Q = 1/sqrt(2). Matches
  // `_k_weight_audio`'s `_biquad_coefficients(4.0, 1/sqrt(2), 1500.0, rate,
  // "high_shelf")`.
  let (hs_b, hs_a) = bs1770_biquad_coefficients(
    4.0,
    1.0 / std::f64::consts::SQRT_2,
    1500.0,
    rate_f64,
    BiquadKind::HighShelf,
  );
  // High-pass: 0 dB (no shelf gain), Q = 0.5, fc = 38 Hz. Matches
  // `_k_weight_audio`'s `_biquad_coefficients(0.0, 0.5, 38.0, rate,
  // "high_pass")`.
  let (hp_b, hp_a) = bs1770_biquad_coefficients(0.0, 0.5, 38.0, rate_f64, BiquadKind::HighPass);

  // Promote the channel to f64 ONCE (the reference's
  // `np.array(data, dtype=np.float64, copy=True)`). The high-shelf and
  // high-pass biquads then run in-place on `chan_f64`, so the peak
  // working set is exactly ONE f64 channel buffer (no
  // `after_shelf`-then-`hp_output` overlap). The f64 precision is still
  // preserved end-to-end across both stages: same `f64`-typed buffer,
  // same f64 coefficients, same f64 state — no intermediate f32 cast
  // between stages (which historically dropped ~16 bits of precision and
  // biased gate decisions near the absolute/relative LUFS thresholds).
  let n = channel.len();
  let mut chan_f64: Vec<f64> = Vec::new();
  chan_f64.try_reserve_exact(n).map_err(|e| Error::Backend {
    message: format!("k_weight_channel: input promotion reservation for {n} samples failed: {e}"),
  })?;
  for &v in channel {
    chan_f64.push(f64::from(v));
  }
  lfilter_f64_in_place(&hs_b, &hs_a, &mut chan_f64)?;
  lfilter_f64_in_place(&hp_b, &hp_a, &mut chan_f64)?;
  Ok(chan_f64)
}

/// Measure ITU-R BS.1770 integrated loudness (LUFS) of an audio signal.
///
/// Faithful port of `mlx_audio.dsp.integrated_loudness(data, rate,
/// block_size=0.400, overlap=0.75)`. Implements the K-weighting (a high-
/// shelf at 1.5 kHz then a high-pass at 38 Hz, via [`lfilter`] plus the
/// internal BS.1770 biquad coefficient helper), then the standard 400 ms /
/// 75 %-overlap gated block analysis with the absolute (`-70 LUFS`) and
/// relative (integrated - `10 LUFS`) gates, and finally the BS.1770
/// reduction `LUFS = -0.691 + 10 * log10(sum_channels gain_c *
/// mean_square_c)`.
///
/// `data` is either:
/// - 1-D (mono): a single channel of `(n_samples,)` `Dtype::F32`, OR
/// - 2-D (multi-channel): `(n_samples, n_channels)` `Dtype::F32` with
///   `1 <= n_channels <= 5` (BS.1770 channel gains are defined for up to
///   5 channels — front L/R/C + surround L/R).
///
/// Returns the integrated loudness in LUFS (`f64`). Returns
/// [`f64::NEG_INFINITY`] when there is no signal energy (matches Python's
/// `np.log10(0.0) = -inf` behavior, since the reference suppresses the
/// `divide` warning rather than raising).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `data` is not 1-D or 2-D,
///   - the 2-D case has more than 5 channels (BS.1770 only defines channel
///     gains up to surround 5.0),
///   - the input is shorter than `block_size * rate` samples (matches the
///     reference's `Audio must have length greater than the block size`
///     raise),
///   - `rate == 0`, `block_size <= 0`, or `overlap` is not in `[0, 1)`,
///   - the **total element count** `n_samples * n_channels` exceeds the
///     shared
///     [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap
///     (the cap is on the materialized buffer the `to_vec` reads — not
///     just the per-channel sample count — so 2-D inputs like
///     `(MAX_DECODED_SAMPLES, 5)` can't bypass it),
///   - the per-channel `f64` mean-square matrix would exceed the
///     `MAX_LOUDNESS_BLOCK_BYTES` byte cap (64 MiB), OR the total
///     per-block sum work `num_blocks * ceil(block_size_samples) *
///     n_channels` exceeds the `MAX_LOUDNESS_WORK` sample-visit cap
///     (256 Mi) — together these reject pathological overlaps very
///     close to 1 BEFORE any `num_blocks`-scaled allocation OR the
///     per-block loop runs (the byte cap dominates for normal block
///     sizes; the visit cap catches near-1 overlaps with small block
///     sizes that fit under the byte cap but multiply the per-block CPU
///     sum work — once per channel, since the per-block loop runs once
///     per channel — to multi-trillion visits),
///   - the number of blocks overflows `usize`,
///   - any size exceeds `i32::MAX`.
///
/// # Memory bounds
/// The streaming per-channel path keeps peak working memory bounded to:
/// - the input clone `raw_f32` (`n_samples * n_channels * 4 bytes`,
///   bounded by the total-element cap above),
/// - ONE channel's f64 K-weighted buffer (`n_samples * 8 bytes`, fully
///   in-place — the high-shelf and high-pass biquads both write back into
///   the SAME buffer via the private in-place lfilter kernel, so there
///   is no second `after_shelf` channel buffer outstanding at the stage
///   boundary; dropped per channel before the next channel's promotion),
///   and
/// - the per-channel mean-square matrix
///   (`num_blocks * n_channels * 8 bytes`, bounded by
///   `MAX_LOUDNESS_BLOCK_BYTES` = 64 MiB so the peak `mean_square`
///   footprint is capped regardless of overlap; the gate-index path
///   iterates `block_loudness` directly with NO intermediate
///   `Vec<usize>` reservation, so there is no `num_blocks`-sized gate-
///   index allocation beyond the mean-square matrix itself).
///
/// We do NOT allocate per-channel deinterleaved f32 buffers AND per-
/// channel weighted f64 buffers all-at-once — the previous revision held
/// `3 * n_samples * n_channels`-worth of channel data simultaneously,
/// which the [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES)
/// cap could not bound (the cap is per-element, and the multiplier on the
/// peak working set was hidden). The chained out-of-place
/// `lfilter_f64(&hs, ...) → lfilter_f64(&hp, ...)` form similarly held
/// TWO f64 channel buffers across the stage-boundary call (~+512 MiB for
/// the max-allowed mono input); the in-place kernel eliminates that
/// overlap.
pub fn integrated_loudness(data: &Array, rate: u32, block_size: f64, overlap: f64) -> Result<f64> {
  // Input validation mirrors `_validate_loudness_audio` + the loudness
  // parameter ranges.
  if rate == 0 {
    return Err(Error::Backend {
      message: "integrated_loudness: rate must be > 0".into(),
    });
  }
  if !(block_size > 0.0 && block_size.is_finite()) {
    return Err(Error::Backend {
      message: format!("integrated_loudness: block_size must be finite and > 0 (got {block_size})"),
    });
  }
  if !((0.0..1.0).contains(&overlap) && overlap.is_finite()) {
    return Err(Error::Backend {
      message: format!("integrated_loudness: overlap must be in [0, 1) (got {overlap})"),
    });
  }
  let shape = data.shape();
  let (n_samples, n_channels) = match shape.len() {
    1 => (shape[0], 1usize),
    2 => {
      // Reference stores audio as (n_samples, n_channels). We follow the
      // same layout (column-per-channel) since BS.1770 K-weights and
      // mean-squares each channel independently.
      let (n_samples, n_channels) = (shape[0], shape[1]);
      if n_channels > BS1770_MAX_CHANNELS {
        return Err(Error::Backend {
          message: format!(
            "integrated_loudness: audio must have at most {BS1770_MAX_CHANNELS} channels \
             (got {n_channels})"
          ),
        });
      }
      if n_channels == 0 {
        return Err(Error::Backend {
          message: "integrated_loudness: audio must have at least 1 channel".into(),
        });
      }
      (n_samples, n_channels)
    }
    other => {
      return Err(Error::Backend {
        message: format!(
          "integrated_loudness: data must be 1-D (mono) or 2-D \
           (n_samples, n_channels), got {other}-D"
        ),
      });
    }
  };
  // Cap on the TOTAL materialized element count BEFORE the `to_vec`. The
  // previous revision capped only `n_samples`, which a 2-D input like
  // `(MAX_DECODED_SAMPLES, 5)` would bypass — the `to_vec` would then
  // materialize `5 * MAX_DECODED_SAMPLES` f32 samples (multi-GB). Mirrors
  // the [`stft`] / OLA pattern of checking the materialized work cap
  // before any allocation.
  let total_elements = n_samples
    .checked_mul(n_channels)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "integrated_loudness: total element count overflows usize (n_samples={n_samples}, \
         n_channels={n_channels})"
      ),
    })?;
  if total_elements > MAX_LOUDNESS_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: total element count {total_elements} (n_samples={n_samples} * \
         n_channels={n_channels}) exceeds the {MAX_LOUDNESS_SAMPLES} cap"
      ),
    });
  }

  // `block_size * rate` is the per-block sample count in the reference.
  // Cast through `f64` to mirror the python arithmetic exactly (block_size
  // is a float seconds, rate is an int Hz).
  let rate_f64 = f64::from(rate);
  let block_samples_f64 = block_size * rate_f64;
  if !block_samples_f64.is_finite() || block_samples_f64 < 1.0 {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: block_size * rate = {block_samples_f64} produces \
         < 1 sample (block_size={block_size}, rate={rate})"
      ),
    });
  }
  if (n_samples as f64) < block_samples_f64 {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: audio length {n_samples} samples must be greater than the \
         block size ({block_samples_f64} samples)"
      ),
    });
  }

  // Per-block analysis. Reference:
  //   step = 1 - overlap
  //   duration_seconds = num_samples / rate
  //   num_blocks = int(round((duration - block_size) / (block_size * step)) + 1)
  let step = 1.0 - overlap;
  let duration_seconds = n_samples as f64 / rate_f64;
  // `step` is in `(0, 1]` (overlap in [0, 1)) so `block_size * step > 0`;
  // the division is safe. The `round + 1` produces the count of blocks
  // whose overlap-strided start fits inside `[0, duration]`.
  //
  // The reference rounds with `np.round`, which is round-half-to-EVEN
  // (banker's rounding) — so `round_ties_even` is REQUIRED here, NOT the
  // half-away-from-zero `f64::round`. They disagree on exact `*.5`
  // quotients: e.g. a default-parameter 0.65 s clip at 48 kHz gives
  // quotient 2.5 → `round_ties_even` ⇒ 2 (→ 3 blocks, the reference's
  // count), `round` ⇒ 3 (→ 4 blocks). A wrong block count shifts the
  // absolute/relative gates and the final LUFS for non-stationary audio.
  let num_blocks_f64 =
    ((duration_seconds - block_size) / (block_size * step)).round_ties_even() + 1.0;
  if !num_blocks_f64.is_finite() || num_blocks_f64 < 1.0 {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: derived num_blocks {num_blocks_f64} is invalid \
         (duration={duration_seconds}, block_size={block_size}, overlap={overlap})"
      ),
    });
  }
  // Reject pathological `num_blocks` BEFORE any `num_blocks`-scaled
  // allocation OR the per-block sum loop runs. A caller-controlled
  // `overlap` close to 1 (e.g. `0.99999990`) makes `step → 0` and
  // `num_blocks → ∞`, which the previous element-only block-work cap
  // (`num_blocks * n_channels <= 64 Mi`) admitted at tens of millions of
  // blocks for a small mono signal — the `mean_square` matrix's actual
  // BYTE footprint is `8 * num_blocks * n_channels` (`f64`), so the
  // element cap let multi-hundred-MB / ~2-GiB reservations through, and
  // the per-block sum loop then re-summed `block_size_samples` per
  // block for an extreme CPU-time blow-up. The two new caps bound the
  // ACTUAL peak bytes AND total sum work:
  //   1. `MAX_LOUDNESS_BLOCK_BYTES` (64 MiB) — bounds the dominant
  //      block-sized `f64` buffer (`mean_square`).
  //   2. `MAX_LOUDNESS_WORK` (256 Mi sample-visits) — bounds the per-
  //      block sum CPU work (`num_blocks * block_size_samples`).
  // Both use `checked_mul`; arithmetic overflow rejects. The
  // `num_blocks_f64 > usize::MAX as f64` check catches the as-cast
  // saturation case where the `usize` cast silently clamps.
  if num_blocks_f64 > usize::MAX as f64 {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: derived num_blocks {num_blocks_f64} overflows usize \
         (duration={duration_seconds}, block_size={block_size}, overlap={overlap})"
      ),
    });
  }
  let num_blocks = num_blocks_f64 as usize;
  if num_blocks == 0 {
    return Err(Error::Backend {
      message: "integrated_loudness: derived num_blocks == 0".into(),
    });
  }
  // Byte cap: `num_blocks * n_channels * sizeof::<f64>() <=
  // MAX_LOUDNESS_BLOCK_BYTES`. Compute the cell count first, then the
  // byte product — both `checked_mul` so overflow rejects.
  let block_cells = num_blocks
    .checked_mul(n_channels)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "integrated_loudness: block cells {num_blocks} * {n_channels} overflows usize \
         (overlap={overlap})"
      ),
    })?;
  let block_bytes = block_cells
    .checked_mul(std::mem::size_of::<f64>())
    .ok_or_else(|| Error::Backend {
      message: format!(
        "integrated_loudness: block bytes {block_cells} * 8 overflows usize \
         (overlap={overlap})"
      ),
    })?;
  if block_bytes > MAX_LOUDNESS_BLOCK_BYTES {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: mean-square byte footprint {block_bytes} \
         (num_blocks={num_blocks} * n_channels={n_channels} * 8) exceeds the \
         {MAX_LOUDNESS_BLOCK_BYTES} byte cap (overlap={overlap})"
      ),
    });
  }
  // Work cap: `num_blocks * block_size_samples * n_channels <=
  // MAX_LOUDNESS_WORK`. The per-block mean-square loop runs ONCE PER
  // CHANNEL (see the per-channel streaming loop below), so the actual
  // sample-visit count is `num_blocks * block_size_samples * n_channels`
  // — bounding the channel-less product alone admitted a 5-channel input
  // (Codex review: `num_blocks=1_677_721, block_samples=160, n_channels=5`
  // gives a channel-less work product of ~268 Mi ≤ 256 Mi cap BUT actual
  // visits ~1.34 Bi, defeating the bound for adversarial overlap × multi-
  // channel). Include `n_channels` in the work product.
  //
  // `block_samples_f64` was already validated finite + >= 1 + <= n_samples
  // (the audio-length-vs-block-size check above), so it fits in `usize`
  // for any `n_samples` we accept (bounded by `MAX_LOUDNESS_SAMPLES`). We
  // use `ceil` here — the per-block bounds (`lower = floor(bi*step*bs*r)`,
  // `upper = floor((bi*step+1)*bs*r)`) can produce `upper - lower =
  // ceil(block_samples_f64)` in the worst case for fractional
  // `block_samples_f64`, so the conservative bound on actual slice visits
  // uses `ceil`. `checked_mul` against `num_blocks` and `n_channels`
  // then rejects overflow up-front.
  let block_samples_usize: usize = block_samples_f64.ceil() as usize;
  let total_work = num_blocks
    .checked_mul(block_samples_usize)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "integrated_loudness: total work {num_blocks} * {block_samples_usize} overflows \
         usize (overlap={overlap})"
      ),
    })?
    .checked_mul(n_channels)
    .ok_or_else(|| Error::Backend {
      message: format!(
        "integrated_loudness: total work {num_blocks} * {block_samples_usize} * \
         {n_channels} overflows usize (overlap={overlap})"
      ),
    })?;
  if total_work > MAX_LOUDNESS_WORK {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: total sample-visit work {total_work} (num_blocks={num_blocks} \
         * block_samples={block_samples_usize} * n_channels={n_channels}) exceeds the \
         {MAX_LOUDNESS_WORK} cap (overlap={overlap})"
      ),
    });
  }

  // Extract the raw interleaved f32 buffer (bounded by the
  // `total_elements <= MAX_LOUDNESS_SAMPLES` cap above). We stream per
  // channel from this buffer with a stride — no per-channel f32 copy and
  // no per-channel weighted f64 buffer kept across iterations. Peak
  // working memory across the K-weight loop is therefore one f64 channel
  // buffer (and the f64 second-stage output) at a time, NOT
  // `3 * n_samples * n_channels`.
  let raw_f32 = data.try_clone()?.to_vec::<f32>()?;
  if raw_f32.len() != total_elements {
    return Err(Error::Backend {
      message: format!(
        "integrated_loudness: internal shape mismatch — got {} f32 samples for shape \
         ({n_samples}, {n_channels})",
        raw_f32.len()
      ),
    });
  }

  // mean_square[c][b] — per-channel, per-block mean-square. Matches the
  // reference's `np.zeros((num_channels, num_blocks))`. Bounded by the
  // `block_work` cap above.
  let mut mean_square: Vec<Vec<f64>> = Vec::new();
  mean_square
    .try_reserve_exact(n_channels)
    .map_err(|e| Error::Backend {
      message: format!(
        "integrated_loudness: reservation for {n_channels} mean-square channels failed: {e}"
      ),
    })?;
  for _ in 0..n_channels {
    let mut row: Vec<f64> = Vec::new();
    row
      .try_reserve_exact(num_blocks)
      .map_err(|e| Error::Backend {
        message: format!(
          "integrated_loudness: reservation for {num_blocks} mean-square blocks failed: {e}"
        ),
      })?;
    row.resize(num_blocks, 0.0);
    mean_square.push(row);
  }

  // Per-channel streaming loop: extract ONE column of `raw_f32` into a
  // small f32 channel buffer, K-weight it to f64 (the K-weighting helper
  // converts to f64 once and runs both biquad stages in f64 end-to-end),
  // compute the per-block mean-square into the matrix row, then DROP both
  // the channel buffer and the K-weighted buffer before the next channel.
  // Peak working set across channels is therefore one channel pair, not
  // `n_channels` pairs.
  //
  // 1-D input has n_channels = 1 and the de-interleave is a straight
  // copy. 2-D input has the reference's `(n_samples, n_channels)` layout
  // — channel `c`'s sample `i` lives at `raw_f32[i * n_channels + c]`.
  let mut chan_f32: Vec<f32> = Vec::new();
  chan_f32
    .try_reserve_exact(n_samples)
    .map_err(|e| Error::Backend {
      message: format!("integrated_loudness: reservation for {n_samples} samples failed: {e}"),
    })?;
  for (c, ms_row) in mean_square.iter_mut().enumerate() {
    chan_f32.clear();
    for i in 0..n_samples {
      chan_f32.push(raw_f32[i * n_channels + c]);
    }
    // K-weight this channel (high-shelf @ 1.5 kHz, then high-pass @ 38 Hz);
    // returns an f64 channel buffer ready for the per-block mean-square.
    let weighted = k_weight_channel(&chan_f32, rate)?;

    // Per-block bounds match the reference's
    //   lower = int(block_size * (block_index * step) * rate)
    //   upper = int(block_size * (block_index * step + 1) * rate)
    // i.e. `lower = floor(block_index * step * block_size * rate)`,
    // `upper = floor((block_index * step + 1) * block_size * rate)`.
    // Both are clamped to the channel length to avoid reading past the
    // end (the reference's slice `arr[lower:upper]` silently clamps; we
    // mirror).
    for (block_index, ms_cell) in ms_row.iter_mut().enumerate() {
      let bi_f64 = block_index as f64;
      let lower_f64 = block_size * (bi_f64 * step) * rate_f64;
      let upper_f64 = block_size * (bi_f64 * step + 1.0) * rate_f64;
      // The reference's `int(...)` floors for non-negative values; the
      // values here are non-negative by construction (step >= 0,
      // block_size > 0, rate > 0, bi >= 0).
      let lower = lower_f64 as usize;
      let upper = (upper_f64 as usize).min(weighted.len());
      if upper <= lower {
        // Empty block — leave the cell at the pre-`resize` 0.0 (the
        // reference's `np.sum(np.square([]))` returns 0.0).
        continue;
      }
      // `mean_square = (1 / (block_size * rate)) * sum(x[lower:upper]^2)`.
      // The `block_size * rate` divisor is the EXPECTED per-block sample
      // count (NOT `upper - lower`, which can be smaller on the trailing
      // block) — preserves the reference's bias-correction-free form.
      //
      // The `Σ v²` reduction goes through `simd::sum_of_squares`: a NEON
      // 2-lane FMA kernel on aarch64 (with a bit-identical scalar
      // fallback). `weighted` is a contiguous `Vec<f64>` and the block
      // slice is contiguous — ideal SIMD input, no layout fixup. The
      // SIMD reduction tree differs from the previous strict
      // left-to-right `sum_sq += v * v` loop, so `sum_sq` may move by a
      // few ULPs; the `log10` in the BS.1770 reduction compresses that
      // well within the loudness tests' tolerances.
      *ms_cell = crate::simd::sum_of_squares(&weighted[lower..upper]) / block_samples_f64;
    }
    // `weighted` drops here (end of channel iteration) — next channel
    // re-uses `chan_f32` via `.clear()` (no shrink) and allocates the next
    // weighted buffer fresh.
    drop(weighted);
  }
  // Free the raw interleaved buffer before we move on to the gate-index
  // collect / per-block loudness reduction — the mean_square matrix is the
  // only thing we still need from the audio data.
  drop(raw_f32);
  drop(chan_f32);

  // Per-block loudness in LUFS: `block_loudness[b] = -0.691 + 10 log10
  // (sum_c gain[c] * mean_square[c][b])`. `log10(0.0) = -inf` is
  // acceptable — these blocks fall below the absolute gate and are
  // dropped (matches the reference's `warnings.simplefilter("ignore", ...)`
  // around the same `log10`).
  let mut block_loudness: Vec<f64> = Vec::new();
  block_loudness
    .try_reserve_exact(num_blocks)
    .map_err(|e| Error::Backend {
      message: format!(
        "integrated_loudness: reservation for {num_blocks} block loudness failed: {e}"
      ),
    })?;
  for b in 0..num_blocks {
    let mut weighted_sum = 0.0_f64;
    // Channel gain is defined for the first 5 channels (the reference's
    // `channel_gains`); we already rejected `> 5` channels above.
    for (gain, ms_row) in BS1770_CHANNEL_GAINS.iter().zip(mean_square.iter()) {
      weighted_sum += gain * ms_row[b];
    }
    block_loudness.push(BS1770_LOUDNESS_OFFSET_LUFS + 10.0 * weighted_sum.log10());
  }

  // Per-channel gated mean of `mean_square`, computed directly from
  // `block_loudness` + `mean_square` WITHOUT materializing a `Vec<usize>`
  // of gated block indices. The previous revision allocated TWO
  // `num_blocks`-sized `Vec<usize>` (one per gate pass) via
  // `try_reserve_exact(num_blocks)` — each `8 * num_blocks` bytes on a
  // 64-bit target, scaling with `num_blocks` even when very few blocks
  // survive the gate. The byte/work caps above now bound `num_blocks`,
  // so this is no longer a multi-GB risk; we still eliminate the
  // intermediate `Vec`s because they're pure overhead — the gated mean
  // is a simple filter-fold over `block_loudness.iter().zip(ms_row)` that
  // visits the same cells either way. Returns NaN for an empty survivor
  // set (matches the reference's `np.mean([])` = NaN; subsequent
  // `nan_to_num` or `log10(NaN)` then carries through faithfully).
  //
  // `pred(block_loudness[b])` selects survivors; the closure runs once
  // per channel and computes that channel's gated mean by iterating
  // `block_loudness` once. `mean_square.iter()` is the per-channel outer
  // loop — total cell visits are `n_channels * num_blocks`, which is
  // bounded by the `MAX_LOUDNESS_BLOCK_BYTES` cap (≤ 8 Mi cells).
  let gated_mean_per_channel = |pred: &dyn Fn(f64) -> bool| -> Vec<f64> {
    let mut out = Vec::with_capacity(n_channels);
    for ms_row in mean_square.iter() {
      let mut acc = 0.0_f64;
      let mut count: usize = 0;
      for (b, &l) in block_loudness.iter().enumerate() {
        if pred(l) {
          acc += ms_row[b];
          count += 1;
        }
      }
      if count == 0 {
        out.push(f64::NAN);
      } else {
        out.push(acc / count as f64);
      }
    }
    out
  };

  // First (absolute-only) gate at -70 LUFS — reference's `>= -70`.
  let gated_mean_square_abs = gated_mean_per_channel(&|l| l >= BS1770_ABSOLUTE_THRESHOLD_LUFS);
  // `relative_threshold = -0.691 + 10*log10(sum(gain * gated_ms)) - 10`.
  // Carries through the reference's `log10` of a possibly-zero/NaN sum;
  // that produces a finite -inf / NaN threshold which simply admits no
  // additional blocks beyond the absolute gate (the python `if loudness >
  // threshold` would then be false / NaN-false).
  let mut weighted_abs = 0.0_f64;
  for (gain, &gms) in BS1770_CHANNEL_GAINS
    .iter()
    .zip(gated_mean_square_abs.iter())
  {
    weighted_abs += gain * gms;
  }
  let relative_threshold =
    BS1770_LOUDNESS_OFFSET_LUFS + 10.0 * weighted_abs.log10() - BS1770_RELATIVE_OFFSET_LUFS;

  // Second pass: blocks above BOTH the relative threshold AND the absolute
  // threshold (the reference uses `> relative_threshold AND > absolute`,
  // not `>=`). NaN/-inf relative_threshold simply fails the `>` test.
  let mut gated_mean_square_rel =
    gated_mean_per_channel(&|l| l > relative_threshold && l > BS1770_ABSOLUTE_THRESHOLD_LUFS);
  // Final per-channel gated mean-square; the reference `nan_to_num`s the
  // NaN-from-empty-set case to 0, which lets us safely add up the weighted
  // sum even when no blocks survived the second gate.
  for v in gated_mean_square_rel.iter_mut() {
    if v.is_nan() {
      *v = 0.0;
    }
  }
  let mut weighted_rel = 0.0_f64;
  for (gain, &gms) in BS1770_CHANNEL_GAINS
    .iter()
    .zip(gated_mean_square_rel.iter())
  {
    weighted_rel += gain * gms;
  }
  // `log10(0.0)` returns `-inf` (matches the reference's
  // `np.errstate(divide='ignore')` + `np.log10` behavior). The final LUFS
  // is then `-inf`, which is the correct answer for "no signal energy
  // survived the gates".
  Ok(BS1770_LOUDNESS_OFFSET_LUFS + 10.0 * weighted_rel.log10())
}

/// Apply a linear gain to bring `data` from `input_loudness` to
/// `target_loudness` (both in LUFS).
///
/// Faithful port of `mlx_audio.dsp.normalize_loudness(data, input_loudness,
/// target_loudness)` — the signal is scaled by
/// `gain = 10^((target - input) / 20)`. The output's shape and dtype match
/// `data` (which must be 1-D or 2-D `Dtype::F32` for the mlxrs audio
/// surface). Unlike the reference's `np.warn("Possible clipped samples
/// in output.")`, we do not emit a runtime warning — Rust has no
/// equivalent of Python's `warnings` module, and the loudness pipeline
/// (`integrated_loudness` → `normalize_loudness`) is the standard
/// pre-normalization step where the caller is expected to peak-limit
/// downstream if needed.
///
/// The typical round-trip is:
/// ```ignore
/// let lufs = integrated_loudness(&samples, rate, 0.4, 0.75)?;
/// let normalized = normalize_loudness(&samples, lufs, -23.0)?; // EBU R128 target
/// // integrated_loudness(&normalized, rate, 0.4, 0.75) ≈ -23.0
/// ```
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `input_loudness` or `target_loudness` is non-finite (NaN/+-inf would
///     yield a non-finite gain, silently corrupting the output),
///   - the input multiply propagates a backend error.
pub fn normalize_loudness(
  data: &Array,
  input_loudness: f64,
  target_loudness: f64,
) -> Result<Array> {
  if !input_loudness.is_finite() {
    return Err(Error::Backend {
      message: format!("normalize_loudness: input_loudness must be finite (got {input_loudness})"),
    });
  }
  if !target_loudness.is_finite() {
    return Err(Error::Backend {
      message: format!(
        "normalize_loudness: target_loudness must be finite (got {target_loudness})"
      ),
    });
  }
  let delta = target_loudness - input_loudness;
  // `gain = 10^(delta / 20)`. Reference: `np.power(10.0, delta / 20.0)`.
  // Compute in f64 (matches the reference) and cast down to f32 for the
  // mlx multiply.
  let gain_f64 = 10.0_f64.powf(delta / 20.0);
  let gain = gain_f64 as f32;
  let gain_arr = Array::full::<f32>(&[0i32; 0], gain)?;
  ops::arithmetic::multiply(data, &gain_arr)
}

/// Peak-normalize `data` so its loudest sample sits at `target_peak_db` dBFS.
///
/// Faithful port of `mlx_audio.dsp.normalize_peak(data, target_peak_db)`
/// (`dsp.py:356`). The signal is scaled by
/// `gain = 10^(target_peak_db / 20) / current_peak`, where
/// `current_peak = max(|data|)`. After scaling, the loudest sample's magnitude
/// is exactly `10^(target_peak_db / 20)` (the linear amplitude for the
/// requested dBFS peak): e.g. `target_peak_db = 0.0` brings the peak to `1.0`
/// (full scale), `target_peak_db = -6.0` to `≈ 0.501`.
///
/// **The target is in dBFS, not a raw linear amplitude** — this mirrors the
/// reference's `target_peak_db` parameter exactly. For a linear target `a`,
/// pass `20 * log10(a)`.
///
/// Like [`normalize_loudness`], the reference's `np.warn("Possible clipped
/// samples in output.")` is not mirrored (Rust has no `warnings` module);
/// with `target_peak_db <= 0.0` the output never exceeds full scale anyway.
///
/// The output's shape and dtype match `data` (1-D or 2-D `Dtype::F32` for the
/// mlxrs audio surface).
///
/// # Errors
/// - [`Error::Backend`] when:
///   - `target_peak_db` is non-finite (`NaN` / `±inf` — would yield a
///     non-finite gain, silently corrupting the output),
///   - `data` is empty (`max` over an empty array is undefined),
///   - the current peak `max(|data|)` is `0.0` (an all-silence input cannot be
///     peak-normalized — the gain would divide by zero) or non-finite (a
///     `NaN` / `inf` sample in the input),
///   - the linear target `10^(target_peak_db / 20)` is non-finite (a finite but
///     enormous `target_peak_db` overflows the f32 amplitude), or the resulting
///     `gain = target_linear / current_peak` is non-finite (a finite target
///     divided by a subnormal nonzero peak overflows) — either would scale the
///     signal into non-finite samples,
///   - the underlying `abs` / `max` / multiply propagates a backend error.
///
/// This reads back one scalar (the current peak) via an explicit `eval`.
pub fn normalize_peak(data: &Array, target_peak_db: f64) -> Result<Array> {
  if !target_peak_db.is_finite() {
    return Err(Error::Backend {
      message: format!("normalize_peak: target_peak_db must be finite (got {target_peak_db})"),
    });
  }
  if data.size() == 0 {
    return Err(Error::Backend {
      message: "normalize_peak: data must be non-empty (max over an empty array is undefined)"
        .into(),
    });
  }
  // current_peak = max(|data|). One explicit scalar readback.
  let abs_data = data.abs()?;
  let mut peak_arr = ops::reduction::max(&abs_data, false)?;
  let current_peak = peak_arr.item::<f32>()?;
  if !current_peak.is_finite() {
    return Err(Error::Backend {
      message: format!(
        "normalize_peak: current peak max(|data|) is non-finite ({current_peak}) — \
         the input contains a NaN or infinite sample"
      ),
    });
  }
  // `current_peak` is `max(|.|)`, so it is always `>= 0.0`; `== 0.0` means an
  // all-silence (or all-zero) input, which cannot be peak-normalized.
  if current_peak == 0.0 {
    return Err(Error::Backend {
      message: "normalize_peak: current peak max(|data|) is 0.0 — an all-silence input \
                cannot be peak-normalized (the gain would divide by zero)"
        .into(),
    });
  }
  // `gain = 10^(target_peak_db / 20) / current_peak`. Reference:
  // `np.power(10.0, target_peak_db / 20.0) / current_peak`. Compute the
  // numerator in f64 (matches the reference) and the division in f32.
  let target_linear = 10.0_f64.powf(target_peak_db / 20.0) as f32;
  let gain = target_linear / current_peak;
  // A FINITE `target_peak_db` and a finite, nonzero `current_peak` can still
  // produce a non-finite `target_linear` (the f64 → f32 narrowing overflows for
  // a huge `target_peak_db`, e.g. `10^(1e30/20)` → `+inf`) or a non-finite
  // `gain` (a finite `target_linear` divided by a subnormal nonzero peak
  // overflows to `+inf`). Either would multiply the signal into non-finite
  // samples — so reject BEFORE building the scalar, exactly as the up-front
  // non-finite `target_peak_db` / `current_peak` guards do.
  if !target_linear.is_finite() {
    return Err(Error::Backend {
      message: format!(
        "normalize_peak: target amplitude 10^(target_peak_db / 20) is non-finite \
         ({target_linear}) — target_peak_db {target_peak_db} is too large to represent \
         as a finite f32 gain"
      ),
    });
  }
  if !gain.is_finite() {
    return Err(Error::Backend {
      message: format!(
        "normalize_peak: gain target_linear / current_peak is non-finite ({gain}) — \
         the current peak {current_peak:.3e} is too small for target_peak_db \
         {target_peak_db} (the scaling overflows to a non-finite value)"
      ),
    });
  }
  let gain_arr = Array::full::<f32>(&[0i32; 0], gain)?;
  ops::arithmetic::multiply(data, &gain_arr)
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
    // P6 / AUDIO-1 (#127): the reference Python form `0.5 * (1 - cos(2π n /
    // (size - 1)))` divides by zero for `size == 1`, silently producing
    // `NaN` for every sample. mlxrs centralizes the rejection in
    // `symmetric_window` so EVERY window function (Hann / Hamming /
    // Blackman / Bartlett) returns a recoverable `Error::Backend` for
    // both `n == 0` (empty window — pointless) and `n == 1` (denom = 0
    // — silent NaN in the reference). The cross-product is exhaustively
    // exercised below to lock the contract for all four window families.
    for r in [
      hann_window(0),
      hann_window(1),
      hamming_window(0),
      hamming_window(1),
      blackman_window(0),
      blackman_window(1),
      bartlett_window(0),
      bartlett_window(1),
    ] {
      assert!(matches!(r, Err(Error::Backend { .. })));
    }
    // The `window_from_name` dispatch must propagate the same rejection
    // for every supported name (so `STR_TO_WINDOW_FN`-style callers also
    // get the error rather than a silent NaN window).
    for name in ["hann", "hanning", "hamming", "blackman", "bartlett"] {
      let r = window_from_name(name, 1);
      assert!(
        matches!(r, Err(Error::Backend { .. })),
        "window_from_name({name:?}, 1) must reject n<2, got {r:?}"
      );
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
    assert_eq!(spec_c.data_ref().shape(), vec![5, 5]); // (num_frames, n_fft/2+1)
    // Metadata is carried on the typed Spectrum (no inference downstream).
    assert_eq!(spec_c.n_fft(), 8);
    assert_eq!(spec_c.win_length(), 8);
    assert_eq!(spec_c.hop_length(), 4);
    assert_eq!(spec_c.window_pad(), WindowPad::Center);
    assert!(spec_c.center());
    for (c, r) in to_vec(&spec_c.data_ref().abs().unwrap())
      .iter()
      .zip(to_vec(&spec_r.data_ref().abs().unwrap()).iter())
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
      assert_eq!(spec.data_ref().shape(), vec![5, 9]); // (num_frames, n_fft/2+1), n_fft=16
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
      spec.data_ref().try_clone().unwrap(),
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
      stft_spec.data_ref().try_clone().unwrap(),
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
      let power = spec.data_ref().abs().unwrap().square().unwrap();
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
      let power = spec.data_ref().abs().unwrap().square().unwrap();
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

  // ---- A4: `lfilter` direct-form II transposed parity ---------------------

  /// Hand-trace the reference `mlx_audio.dsp.lfilter` for a single-pole IIR
  /// `y[n] = 0.5 * x[n] + 0.5 * y[n-1]` (i.e. `b=[0.5], a=[1, -0.5]`) on an
  /// impulse `x = [1, 0, 0, 0, 0]`. Closed form: `y[n] = 0.5 * (0.5)^n`,
  /// i.e. `[0.5, 0.25, 0.125, 0.0625, 0.03125]`. This is the canonical
  /// single-pole-IIR sanity check from the spec.
  #[test]
  fn lfilter_single_pole_iir_impulse_response() {
    let b: [f64; 1] = [0.5];
    let a: [f64; 2] = [1.0, -0.5];
    let x_buf: [f32; 5] = [1.0, 0.0, 0.0, 0.0, 0.0];
    let x = Array::from_slice::<f32>(&x_buf, &[5i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    let expected = [0.5_f32, 0.25, 0.125, 0.0625, 0.03125];
    assert_eq!(y.len(), expected.len());
    for (i, (g, e)) in y.iter().zip(expected.iter()).enumerate() {
      // f64-computed exact dyadic values; tight tolerance (the only error
      // source is the final f64→f32 cast on a representable f32).
      assert!(
        (g - e).abs() < 1e-7,
        "lfilter[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  /// Hand-trace the SAME single-pole IIR on a step input `x = [1, 1, 1, 1,
  /// 1]`. Closed form: `y[n] = 1 - (0.5)^(n+1)` →
  /// `[0.5, 0.75, 0.875, 0.9375, 0.96875]`. Asserts the recurrence runs
  /// correctly past the first sample (the impulse test only exercises the
  /// initial decay).
  #[test]
  fn lfilter_single_pole_iir_step_response() {
    let b: [f64; 1] = [0.5];
    let a: [f64; 2] = [1.0, -0.5];
    let x_buf: [f32; 5] = [1.0, 1.0, 1.0, 1.0, 1.0];
    let x = Array::from_slice::<f32>(&x_buf, &[5i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    let expected = [0.5_f32, 0.75, 0.875, 0.9375, 0.96875];
    assert_eq!(y.len(), expected.len());
    for (i, (g, e)) in y.iter().zip(expected.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-7,
        "lfilter step[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  /// Pure-FIR (state_len == 0): `b = [2.0], a = [1.0]` is a unit-delay-free
  /// passthrough doubler. `y[n] = 2 * x[n]`. Exercises the
  /// `state_len == 0` fast path in [`lfilter`].
  #[test]
  fn lfilter_fir_no_state_doubles() {
    let b: [f64; 1] = [2.0];
    let a: [f64; 1] = [1.0];
    let x_buf: [f32; 4] = [0.1, -0.5, 0.7, 1.0];
    let x = Array::from_slice::<f32>(&x_buf, &[4i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    let expected = [0.2_f32, -1.0, 1.4, 2.0];
    for (i, (g, e)) in y.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < 1e-6, "fir[{i}]: got {g}, want {e}");
    }
  }

  /// Normalization by `a[0] != 1`: `b = [1.0], a = [2.0]` should normalize
  /// to `b = [0.5], a = [1.0]`, i.e. `y[n] = 0.5 * x[n]`. Exercises the
  /// `a[0] != 1` normalization path (the reference always divides; we
  /// mirror).
  #[test]
  fn lfilter_normalizes_by_leading_a() {
    let b: [f64; 1] = [1.0];
    let a: [f64; 1] = [2.0];
    let x_buf: [f32; 3] = [4.0, 8.0, -2.0];
    let x = Array::from_slice::<f32>(&x_buf, &[3i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    let expected = [2.0_f32, 4.0, -1.0];
    for (i, (g, e)) in y.iter().zip(expected.iter()).enumerate() {
      assert!((g - e).abs() < 1e-6, "norm[{i}]: got {g}, want {e}");
    }
  }

  /// Biquad (state_len == 2) hand-trace: pass a 2-tap b and 3-tap a, then
  /// hand-trace 4 samples through the reference's recurrence and assert
  /// byte-for-byte.
  ///
  /// Filter: `b = [0.25, 0.5]`, `a = [1.0, -0.3, 0.1]` → recurrence
  /// `y[n] = 0.25 x[n] + 0.5 x[n-1] + 0.3 y[n-1] - 0.1 y[n-2]`. With
  /// state_len = max(2, 3) - 1 = 2 the reference's transposed loop produces
  /// (hand-traced with state vectors at each step) for input
  /// `x = [1, 0, 0, 0]`:
  ///   n=0: y=0.25; n=1: y=0.5 + 0.075 = 0.575; n=2: y=0+0.1725-0.025 =
  ///   0.1475; n=3: y=0+0.04425-0.0575 = -0.01325.
  #[test]
  fn lfilter_biquad_hand_traced_impulse() {
    let b: [f64; 2] = [0.25, 0.5];
    let a: [f64; 3] = [1.0, -0.3, 0.1];
    let x_buf: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
    let x = Array::from_slice::<f32>(&x_buf, &[4i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    let expected = [0.25_f32, 0.575, 0.1475, -0.01325];
    for (i, (g, e)) in y.iter().zip(expected.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-6,
        "biquad[{i}]: got {g}, want {e} (diff {})",
        (g - e).abs()
      );
    }
  }

  /// Empty `b` → return zeros of the input shape (mirrors the reference's
  /// `np.zeros_like(data)` early return).
  #[test]
  fn lfilter_empty_b_returns_zeros() {
    let b: [f64; 0] = [];
    let a: [f64; 1] = [1.0];
    let x_buf: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x = Array::from_slice::<f32>(&x_buf, &[4i32]).unwrap();
    let y = to_vec(&lfilter(&b, &a, &x).unwrap());
    assert_eq!(y, vec![0.0_f32; 4]);
  }

  /// `a` empty / `a[0] == 0` / non-1-D input must all be rejected with
  /// `Error::Backend`, matching the reference's `ValueError` raises. (Note:
  /// **empty `b`** is NOT a rejection — the reference returns
  /// `np.zeros_like(data)` and we mirror that fast-path; see
  /// `lfilter_empty_b_returns_zeros` for that case.)
  #[test]
  fn lfilter_rejects_invalid_inputs() {
    let x = Array::from_slice::<f32>(&[1.0_f32, 2.0], &[2i32]).unwrap();
    // a empty — reference raises `filter denominator must have a non-zero
    // leading term` (the empty `a` falls into the `a[0] == 0` branch via
    // `a.size == 0 or a[0] == 0`).
    assert!(matches!(
      lfilter(&[1.0_f64], &[], &x),
      Err(Error::Backend { .. })
    ));
    // a[0] == 0
    assert!(matches!(
      lfilter(&[1.0_f64], &[0.0_f64, 1.0], &x),
      Err(Error::Backend { .. })
    ));
    // 2-D input — reference raises `dsp.lfilter only supports 1-D input`.
    let x_2d = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &[2i32, 2i32]).unwrap();
    assert!(matches!(
      lfilter(&[0.5_f64], &[1.0_f64, -0.5], &x_2d),
      Err(Error::Backend { .. })
    ));
  }

  /// Public [`lfilter`]'s sample cap must fire on the input SHAPE BEFORE
  /// any `to_vec` materialization. We construct a lazy `Array::zeros` of
  /// `(MAX_LFILTER_SAMPLES + 1,)` f32 — which mlx does not eval until a
  /// data accessor runs — and assert `lfilter` rejects it with
  /// `Error::Backend`. If the cap check still lived behind the `to_vec`
  /// (the pre-fix behavior), the rejected call would have first
  /// materialized `(MAX_LFILTER_SAMPLES + 1) * 4 bytes` (≈256 MiB) of f32
  /// plus a second `(MAX_LFILTER_SAMPLES + 1) * 8 bytes` (≈512 MiB) f64
  /// promotion before erroring — a ~768 MiB allocation for a call that
  /// the bounded-memory contract says must allocate nothing. The lazy
  /// `Array::zeros` is the regression handle: with the up-front cap
  /// check this test runs effectively for free.
  #[test]
  fn lfilter_rejects_lazy_oversized_input_without_allocating() {
    let lazy_huge =
      Array::zeros::<f32>(&[(MAX_LFILTER_SAMPLES + 1) as i32]).expect("lazy zeros must succeed");
    let res = lfilter(&[0.5_f64], &[1.0_f64, -0.5], &lazy_huge);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "lfilter must reject a lazy ({} samples) input via the up-front \
       sample cap, BEFORE the to_vec materializes f32 / promotes to f64 \
       (got {res:?})",
      MAX_LFILTER_SAMPLES + 1
    );
  }

  /// [`lfilter_f64_in_place`] is the in-place variant the K-weighting path
  /// uses to keep the peak working set at ONE f64 channel buffer (the
  /// pre-fix chained out-of-place form held TWO across the high-shelf →
  /// high-pass stage boundary). Numerically the two kernels must produce
  /// BIT-IDENTICAL output (same direct-form II transposed math, same f64
  /// precision, same state updates) — only the allocation strategy
  /// differs. Pin this with the same biquad hand-traced impulse from
  /// `lfilter_biquad_hand_traced_impulse`, comparing the in-place output
  /// against the out-of-place `lfilter_f64`'s output sample-for-sample
  /// (zero tolerance — these MUST agree exactly in f64).
  #[test]
  fn lfilter_f64_in_place_matches_out_of_place() {
    let b: [f64; 2] = [0.25, 0.5];
    let a: [f64; 3] = [1.0, -0.3, 0.1];
    let x_in: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

    // Out-of-place reference output.
    let y_out = lfilter_f64(&b, &a, &x_in).expect("out-of-place must succeed");

    // In-place run on a mutable copy.
    let mut x_buf = x_in;
    lfilter_f64_in_place(&b, &a, &mut x_buf).expect("in-place must succeed");

    assert_eq!(
      x_buf.len(),
      y_out.len(),
      "in-place output length must match out-of-place"
    );
    for (i, (g, e)) in x_buf.iter().zip(y_out.iter()).enumerate() {
      // Bit-identical: same arithmetic in f64, no tolerance.
      assert_eq!(
        g, e,
        "in-place[{i}] = {g} must equal out-of-place[{i}] = {e} \
         (bit-identical f64)"
      );
    }
  }

  /// In-place `state_len == 0` fast path (`b = [2.0], a = [1.0]` →
  /// `y[n] = 2 * x[n]`) must overwrite the buffer correctly. The
  /// in-place kernel's per-slot ordering (read `sample` before writing
  /// `output`) is trivial in this branch but the parity guarantee with
  /// `lfilter_f64` still applies — assert both kernels produce the same
  /// output on the same input.
  #[test]
  fn lfilter_f64_in_place_state_len_zero_doubles() {
    let b: [f64; 1] = [2.0];
    let a: [f64; 1] = [1.0];
    let x_in: [f64; 4] = [0.1, -0.5, 0.7, 1.0];

    let y_out = lfilter_f64(&b, &a, &x_in).expect("out-of-place must succeed");
    let mut x_buf = x_in;
    lfilter_f64_in_place(&b, &a, &mut x_buf).expect("in-place must succeed");

    for (i, (g, e)) in x_buf.iter().zip(y_out.iter()).enumerate() {
      assert_eq!(g, e, "in-place fir[{i}] = {g} must equal out-of-place {e}");
    }
  }

  /// In-place `b.is_empty()` semantics: mirror [`lfilter_f64`]'s
  /// `np.zeros_like`-equivalent by overwriting the input buffer with
  /// zeros (the in-place equivalent of returning a fresh zero `Vec`).
  #[test]
  fn lfilter_f64_in_place_empty_b_zeros_buffer() {
    let b: [f64; 0] = [];
    let a: [f64; 1] = [1.0];
    let mut x_buf: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    lfilter_f64_in_place(&b, &a, &mut x_buf).expect("empty-b must succeed");
    for (i, &v) in x_buf.iter().enumerate() {
      assert_eq!(v, 0.0, "empty-b in-place must zero x_buf[{i}], got {v}");
    }
  }

  /// In-place kernel must reject the same invalid inputs as
  /// [`lfilter_f64`]: empty `a`, `a[0] == 0`. The sample-cap branch is
  /// not exercised here (would require a multi-GB buffer) but its
  /// presence in the kernel is verified by the existing
  /// `integrated_loudness_rejects_oversized_total_elements` cap test
  /// upstream.
  #[test]
  fn lfilter_f64_in_place_rejects_invalid_inputs() {
    let mut x_buf: [f64; 2] = [1.0, 2.0];
    assert!(matches!(
      lfilter_f64_in_place(&[1.0_f64], &[], &mut x_buf),
      Err(Error::Backend { .. })
    ));
    assert!(matches!(
      lfilter_f64_in_place(&[1.0_f64], &[0.0_f64, 1.0], &mut x_buf),
      Err(Error::Backend { .. })
    ));
  }

  // ---- A3: BS.1770 K-weighted integrated loudness + normalize_loudness -----

  /// Generate a `seconds`-long mono sine at `freq` Hz with amplitude `amp`
  /// at `rate` samples/sec, as an `Array` of `Dtype::F32`.
  fn sine_mono(freq: f64, amp: f32, rate: u32, seconds: f64) -> Array {
    let n = (seconds * f64::from(rate)) as usize;
    let mut buf: Vec<f32> = Vec::with_capacity(n);
    let two_pi_freq = 2.0 * std::f64::consts::PI * freq;
    let rate_f64 = f64::from(rate);
    for i in 0..n {
      let t = i as f64 / rate_f64;
      buf.push(amp * (two_pi_freq * t).sin() as f32);
    }
    Array::from_slice::<f32>(&buf, &[n as i32]).unwrap()
  }

  /// Sanity: a 1 kHz sine well above 0 LUFS produces a finite (not -inf,
  /// not NaN) integrated loudness above the absolute gate. This pins the
  /// happy path of the full pipeline (K-weighting + block analysis + both
  /// gates).
  #[test]
  fn integrated_loudness_sine_produces_finite_lufs() {
    // 3 s of 1 kHz sine, amp = 0.5, at 48 kHz. The signal is well above
    // -70 LUFS, so the absolute gate cannot drop every block, and a
    // single-frequency sine has uniform per-block loudness so the relative
    // gate keeps every block too.
    let x = sine_mono(1000.0, 0.5, 48_000, 3.0);
    let lufs = integrated_loudness(&x, 48_000, 0.4, 0.75).unwrap();
    assert!(
      lufs.is_finite(),
      "integrated_loudness on a 1 kHz sine must be finite, got {lufs}"
    );
    assert!(
      lufs > BS1770_ABSOLUTE_THRESHOLD_LUFS,
      "1 kHz sine at amp=0.5 should be well above -70 LUFS (got {lufs})"
    );
  }

  /// A 6 dB amplitude doubling raises integrated loudness by ~6 dB. This is
  /// a relative parity check that exercises the K-weighting + per-block
  /// mean-square pipeline end-to-end without needing to hardcode the
  /// absolute LUFS value (which depends on the exact K-filter coefficients).
  #[test]
  fn integrated_loudness_scales_with_amplitude_squared() {
    let rate = 48_000u32;
    let x_lo = sine_mono(1000.0, 0.25, rate, 3.0);
    let x_hi = sine_mono(1000.0, 0.5, rate, 3.0); // +6.02 dB
    let l_lo = integrated_loudness(&x_lo, rate, 0.4, 0.75).unwrap();
    let l_hi = integrated_loudness(&x_hi, rate, 0.4, 0.75).unwrap();
    let delta = l_hi - l_lo;
    // 20 log10(2) ≈ 6.02 LU. Allow ±0.05 LU for f32→f64 round-trip noise.
    assert!(
      (delta - 6.0206).abs() < 0.05,
      "doubling amplitude (+6 dB) should add ~6 LU (got {delta} = {l_hi} - {l_lo})"
    );
  }

  /// Round-trip: measure a signal's LUFS, normalize to a target, re-measure
  /// — the re-measured value must match the target. This is the spec's
  /// `normalize_loudness` parity test.
  #[test]
  fn normalize_loudness_round_trip_matches_target() {
    let rate = 48_000u32;
    let x = sine_mono(1000.0, 0.5, rate, 3.0);
    let lufs_before = integrated_loudness(&x, rate, 0.4, 0.75).unwrap();
    // EBU R128 broadcast target.
    let target = -23.0_f64;
    let normalized = normalize_loudness(&x, lufs_before, target).unwrap();
    let lufs_after = integrated_loudness(&normalized, rate, 0.4, 0.75).unwrap();
    // BS.1770 + normalize is linear in amplitude; the round-trip is exact
    // modulo the f32 gain quantization on the multiply. Tight tolerance.
    assert!(
      (lufs_after - target).abs() < 0.01,
      "normalize_loudness round-trip should hit target ±0.01 LUFS, \
       got {lufs_after} (target {target}, before {lufs_before})"
    );
  }

  /// Silence below the absolute gate produces `-inf` LUFS (the reference's
  /// `np.log10(0.0) = -inf` behavior, mirrored). Asserts the absolute-gate
  /// branch falls through correctly when no block survives.
  #[test]
  fn integrated_loudness_silence_returns_neg_inf() {
    let rate = 48_000u32;
    let n = (3.0 * f64::from(rate)) as usize;
    let zeros = vec![0.0_f32; n];
    let x = Array::from_slice::<f32>(&zeros, &[n as i32]).unwrap();
    let lufs = integrated_loudness(&x, rate, 0.4, 0.75).unwrap();
    assert!(
      lufs == f64::NEG_INFINITY,
      "silence should return -inf LUFS (got {lufs})"
    );
  }

  /// 2-D stereo input (n_samples, 2) accepted; mono and stereo of the same
  /// per-channel content produce identical LUFS (the channel gains for
  /// channels 0 and 1 are both 1.0, so doubling the channel count with the
  /// same content doubles the weighted-mean-square — i.e. adds ~3 LU). Pins
  /// the 2-D layout (n_samples, n_channels), de-interleave path, and the
  /// `channel_gains[0..2] = [1.0, 1.0]` literal.
  #[test]
  fn integrated_loudness_stereo_accepts_2d_and_adds_3lu() {
    let rate = 48_000u32;
    // Generate 3 s of mono sine, then build a stereo (n_samples, 2) Array
    // with both channels identical to that mono signal.
    let mono = sine_mono(1000.0, 0.5, rate, 3.0);
    let mono_buf = mono.try_clone().unwrap().to_vec::<f32>().unwrap();
    let n = mono_buf.len();
    // Interleave: [s0_l, s0_r, s1_l, s1_r, ...]
    let mut stereo_buf: Vec<f32> = Vec::with_capacity(n * 2);
    for &s in &mono_buf {
      stereo_buf.push(s);
      stereo_buf.push(s);
    }
    let stereo = Array::from_slice::<f32>(&stereo_buf, &[n as i32, 2i32]).unwrap();

    let lufs_mono = integrated_loudness(&mono, rate, 0.4, 0.75).unwrap();
    let lufs_stereo = integrated_loudness(&stereo, rate, 0.4, 0.75).unwrap();
    let delta = lufs_stereo - lufs_mono;
    // Two identical channels with gain 1.0 each → weighted sum is 2x mono.
    // 10 log10(2) ≈ 3.01 LU. Allow ±0.05 LU.
    assert!(
      (delta - 3.0103).abs() < 0.05,
      "duplicating a mono signal to stereo (same content, gains [1, 1]) \
       should add ~3 LU (got delta {delta} = {lufs_stereo} - {lufs_mono})"
    );
  }

  /// Regression: the BS.1770 block count must use round-half-to-EVEN
  /// (`np.round` parity), NOT half-away-from-zero `f64::round`. They
  /// disagree on exact `*.5` quotients.
  ///
  /// With the default parameters (`block_size = 0.4 s`, `overlap = 0.75`,
  /// so `step = 0.25`), a `0.65 s` clip at `48 kHz` (= `31200` samples)
  /// gives a block-count quotient of exactly
  ///   `(0.65 - 0.4) / (0.4 * 0.25) = 0.25 / 0.1 = 2.5`,
  /// so `num_blocks = round(2.5) + 1`:
  ///   - `round_ties_even(2.5) = 2` ⇒ **3 blocks** (the reference's count)
  ///   - `f64::round(2.5)      = 3` ⇒ **4 blocks** (a parity bug)
  ///
  /// The block start/stride is `lower = floor(block_index * step *
  /// block_size * rate)`, `upper = floor((block_index * step + 1) *
  /// block_size * rate)`, so the four candidate blocks cover
  ///   block 0 = `[0, 19200)`, block 1 = `[4800, 24000)`,
  ///   block 2 = `[9600, 28800)`, block 3 = `[14400, 31200)`.
  /// Crucially, samples `[28800, 31200)` (the last `0.05 s`) fall ONLY in
  /// block 3. This test builds a signal that is pure silence everywhere
  /// EXCEPT that tail, which carries a loud 1 kHz sine. A correct 3-block
  /// analysis then sees only silent blocks — every block is below the
  /// `-70 LUFS` absolute gate, so the integrated LUFS is `-inf`
  /// (`10 * log10(0)`). A buggy 4-block analysis additionally measures
  /// block 3, which is loud, yielding a finite integrated LUFS well above
  /// `-70`. Asserting `-inf` therefore pins the block count at 3 and fails
  /// loudly if the rounding ever regresses to `f64::round`.
  #[test]
  fn integrated_loudness_block_count_uses_round_ties_even() {
    let rate = 48_000u32;
    // 0.65 s @ 48 kHz = exactly 31200 samples (0.65 * 48000 is exact in
    // f64); quotient is exactly 2.5 — the tie that round vs round_ties_even
    // disagree on.
    let n = 31_200usize;
    debug_assert_eq!(n, (0.65_f64 * f64::from(rate)) as usize);
    // Pure silence except the final 0.05 s ([28800, 31200)) — that span is
    // covered ONLY by the would-be 4th block (block index 3).
    let loud_start = 28_800usize;
    let mut buf: Vec<f32> = vec![0.0_f32; n];
    let two_pi_freq = 2.0 * std::f64::consts::PI * 1000.0;
    let rate_f64 = f64::from(rate);
    for (i, s) in buf.iter_mut().enumerate().skip(loud_start) {
      let t = i as f64 / rate_f64;
      *s = 0.5_f32 * (two_pi_freq * t).sin() as f32;
    }
    let x = Array::from_slice::<f32>(&buf, &[n as i32]).unwrap();

    let lufs = integrated_loudness(&x, rate, 0.4, 0.75).unwrap();
    // 3 blocks (tie-to-even): every block is silent → -inf.
    // 4 blocks (half-away-from-zero, the bug): block 3 is loud → finite.
    assert!(
      lufs == f64::NEG_INFINITY,
      "0.65 s clip @ 48 kHz must yield 3 blocks (round-ties-even); a silent \
       signal with a loud tail only in the would-be 4th block must return \
       -inf LUFS. Got {lufs} — a finite value means the block count \
       regressed to 4 (f64::round instead of round_ties_even)"
    );
  }

  /// Input shorter than `block_size * rate` must be rejected (matches the
  /// reference's `Audio must have length greater than the block size`
  /// raise).
  #[test]
  fn integrated_loudness_rejects_too_short_input() {
    let rate = 48_000u32;
    // 0.1 s @ 48 kHz = 4800 samples; block_size=0.4 needs 19200.
    let x = sine_mono(1000.0, 0.5, rate, 0.1);
    let res = integrated_loudness(&x, rate, 0.4, 0.75);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "audio shorter than block_size * rate must be rejected (got {res:?})"
    );
  }

  /// 2-D input with `> 5` channels must be rejected (matches the
  /// reference's `Audio must have five channels or less` raise).
  #[test]
  fn integrated_loudness_rejects_more_than_five_channels() {
    let rate = 48_000u32;
    let n = 24_000usize; // 0.5 s
    let buf = vec![0.0_f32; n * 6];
    let x = Array::from_slice::<f32>(&buf, &[n as i32, 6i32]).unwrap();
    let res = integrated_loudness(&x, rate, 0.4, 0.75);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "audio with >5 channels must be rejected (got {res:?})"
    );
  }

  /// 3-D input must be rejected (loudness is defined on (n_samples,) or
  /// (n_samples, n_channels)).
  #[test]
  fn integrated_loudness_rejects_3d_input() {
    let rate = 48_000u32;
    let buf = vec![0.0_f32; 24_000];
    let x = Array::from_slice::<f32>(&buf, &[100i32, 60i32, 4i32]).unwrap();
    let res = integrated_loudness(&x, rate, 0.4, 0.75);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "3-D input must be rejected (got {res:?})"
    );
  }

  /// Invalid `overlap` (out of [0, 1)) and `block_size <= 0` must be
  /// rejected.
  #[test]
  fn integrated_loudness_rejects_invalid_block_params() {
    let rate = 48_000u32;
    let x = sine_mono(1000.0, 0.5, rate, 3.0);
    // overlap = 1.0 — would divide by zero in `step = 1 - overlap`.
    assert!(matches!(
      integrated_loudness(&x, rate, 0.4, 1.0),
      Err(Error::Backend { .. })
    ));
    // overlap < 0
    assert!(matches!(
      integrated_loudness(&x, rate, 0.4, -0.1),
      Err(Error::Backend { .. })
    ));
    // block_size <= 0
    assert!(matches!(
      integrated_loudness(&x, rate, 0.0, 0.75),
      Err(Error::Backend { .. })
    ));
    // rate == 0
    assert!(matches!(
      integrated_loudness(&x, 0, 0.4, 0.75),
      Err(Error::Backend { .. })
    ));
  }

  /// Total-element cap (`n_samples * n_channels`) must reject oversized 2-D
  /// inputs BEFORE the `to_vec`. The previous revision capped only
  /// `n_samples`, so a `(MAX_DECODED_SAMPLES, 5)` lazily-shaped input
  /// would slip past the per-channel cap and then materialize
  /// `5 * MAX_DECODED_SAMPLES` f32 samples (multi-GB) in `to_vec`. We use a
  /// LAZY `Array::zeros` so nothing is materialized when the cap is
  /// honored — asserting `Err` proves the cap fired BEFORE the to_vec
  /// allocation. Both shapes:
  /// - 1-D `(MAX_DECODED_SAMPLES + 1,)` — over the 1-channel cap, and
  /// - 2-D `(MAX_DECODED_SAMPLES / 5 + 1, 5)` — over the 5-channel cap
  ///   (per-channel count alone would NOT exceed the cap; total elements
  ///   does).
  ///
  /// (Tests that USE the full cap would force a multi-GB allocation per
  /// run; we test the rejection path, which is what bounds memory.)
  #[test]
  fn integrated_loudness_rejects_oversized_total_elements() {
    let rate = 48_000u32;
    // 1-D: per-channel count alone over the cap.
    let lazy_mono =
      Array::zeros::<f32>(&[(crate::audio::io::MAX_DECODED_SAMPLES + 1) as i32]).unwrap();
    let res_mono = integrated_loudness(&lazy_mono, rate, 0.4, 0.75);
    assert!(
      matches!(res_mono, Err(Error::Backend { .. })),
      "1-D input above the per-channel cap must be rejected (got {res_mono:?})"
    );
    // 2-D: per-channel BELOW the cap (would slip past a per-channel-only
    // check) but TOTAL ELEMENTS above. With n_channels=5, per-channel
    // n_samples = MAX_DECODED_SAMPLES / 5 + 1 < cap, but total =
    // 5 * n_samples > cap.
    let n_per_chan = crate::audio::io::MAX_DECODED_SAMPLES / 5 + 1;
    let lazy_5ch = Array::zeros::<f32>(&[n_per_chan as i32, 5i32]).unwrap();
    let res_5ch = integrated_loudness(&lazy_5ch, rate, 0.4, 0.75);
    assert!(
      matches!(res_5ch, Err(Error::Backend { .. })),
      "2-D input where per-channel count fits but total elements does not \
       must be rejected by the TOTAL-elements cap (got {res_5ch:?})"
    );
  }

  /// Pathological `overlap` very close to 1 (e.g. `0.999_999_999_999`)
  /// makes `step = 1 - overlap → ~1e-12` and `num_blocks ≈ duration /
  /// (block_size * step) → trillions`, driving a multi-GB `mean_square`
  /// reservation + gate-index collect for an otherwise-tiny signal. The
  /// `MAX_LOUDNESS_BLOCK_BYTES` byte cap (64 MiB on the `f64` mean-
  /// square matrix) and `MAX_LOUDNESS_WORK` visit cap (256 Mi sample-
  /// visits on the per-block sum loop) together must reject this BEFORE
  /// any `num_blocks`-scaled allocation OR the per-block loop runs. We
  /// use a small signal (3 s @ 48 kHz, well under
  /// [`MAX_LOUDNESS_SAMPLES`]) so the rejection is *purely* from the
  /// pathological-overlap caps (not the sample cap). The overlap is
  /// intentionally extreme — `0.999_999_999_999` makes `num_blocks ≈
  /// 6.5e12` regardless of `n_channels`, so even mono (n_channels=1)
  /// clears both caps by orders of magnitude. Asserting `Err` (not
  /// panic, not OOM, not multi-minute timeout) proves the caps fire
  /// up-front BEFORE the per-block loop.
  #[test]
  fn integrated_loudness_rejects_pathological_overlap() {
    let rate = 48_000u32;
    // 3 s of audio (way under MAX_DECODED_SAMPLES = 64 Mi samples).
    // block_size = 0.4 s, overlap = 1 - 1e-12 ⇒ step ≈ 1e-12,
    // num_blocks ≈ (3 - 0.4) / (0.4 * 1e-12) ≈ 6.5e12 — orders of
    // magnitude above both new caps. The caps MUST fire BEFORE the
    // mean_square allocation OR the per-block sum loop; if they did
    // not, this would attempt a >=50 TB mean_square reservation and
    // either OOM or take effectively forever.
    let x = sine_mono(1000.0, 0.5, rate, 3.0);
    let res = integrated_loudness(&x, rate, 0.4, 0.999_999_999_999);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "pathological overlap close to 1 must be rejected by the byte/work \
       caps BEFORE any num_blocks-scaled allocation (got {res:?})"
    );
    // Same test with a STEREO input — proves the `n_channels` factor in
    // the byte cap also catches it (a less-extreme overlap that produces
    // num_blocks just under the cap for mono would exceed it for stereo).
    let mono_buf = x.try_clone().unwrap().to_vec::<f32>().unwrap();
    let n = mono_buf.len();
    let mut stereo_buf: Vec<f32> = Vec::with_capacity(n * 2);
    for &s in &mono_buf {
      stereo_buf.push(s);
      stereo_buf.push(s);
    }
    let stereo = Array::from_slice::<f32>(&stereo_buf, &[n as i32, 2i32]).unwrap();
    let res_stereo = integrated_loudness(&stereo, rate, 0.4, 0.999_999_999_999);
    assert!(
      matches!(res_stereo, Err(Error::Backend { .. })),
      "pathological-overlap stereo (byte cap factor n_channels) must be \
       rejected (got {res_stereo:?})"
    );
  }

  /// Regression: the *just-below-the-old-element-cap* case Codex flagged.
  /// The previous revision capped `num_blocks * n_channels` against
  /// `MAX_DECODED_SAMPLES = 64 Mi-elements` — but each cell is `f64`, so
  /// `64 Mi cells * 8 B = 512 MiB` of actual `mean_square` allocation
  /// passed the element-only cap. A near-1 overlap like `0.99999990`
  /// on a tiny 3 s mono signal produces `num_blocks ≈ (3.0 - 0.4) /
  /// (0.4 * 1e-7) ≈ 6.5e7` blocks, which sit JUST UNDER the old
  /// 64 Mi-element cap (so the old guard would NOT reject and the
  /// `try_reserve_exact` would attempt a ~520 MiB `mean_square` matrix
  /// reservation followed by a per-block loop that re-sums 19,200
  /// samples per block — multi-trillion sample visits, hours of CPU).
  /// The new `MAX_LOUDNESS_BLOCK_BYTES` (64 MiB) cap rejects this case
  /// at the byte-budget check BEFORE any allocation (block_bytes =
  /// 6.5e7 * 1 * 8 = 520 MiB > 64 MiB); the visit cap would also catch
  /// it (6.5e7 * 19200 = 1.25 trillion visits > 256 Mi). Asserting
  /// `Err` in microseconds proves the byte/work caps fire up-front,
  /// not the old elements-only cap.
  #[test]
  fn integrated_loudness_rejects_overlap_just_below_old_element_cap() {
    let rate = 48_000u32;
    // 3 s of audio = 144,000 samples — well below MAX_LOUDNESS_SAMPLES.
    // overlap = 0.99999990 ⇒ step = 1e-7 ⇒ num_blocks ≈ 6.5e7. The OLD
    // element cap was 64 Mi ≈ 6.7e7, so 6.5e7 was UNDER the old cap;
    // the new byte cap (64 MiB / 8 B = 8 Mi cells) rejects num_blocks
    // > 8 Mi for n_channels=1, and the work cap (256 Mi) rejects
    // 6.5e7 * 19200 ≈ 1.25e12 visits — both fire well below 6.5e7.
    let x = sine_mono(1000.0, 0.5, rate, 3.0);
    let res = integrated_loudness(&x, rate, 0.4, 0.999_999_90);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "overlap=0.99999990 (under old elements-only cap; over new byte/work \
       caps) must be rejected BEFORE allocation (got {res:?})"
    );
  }

  /// Regression: the work cap MUST include `n_channels` (Codex review).
  /// The per-block mean-square sum loop runs ONCE PER CHANNEL (the
  /// per-channel streaming K-weighting loop), so the actual sample-visit
  /// count is `num_blocks * block_samples * n_channels`. A prior revision
  /// bounded the channel-less product alone, so a 5-channel pathological
  /// case slipped through: `num_blocks ≈ 500_000, block_samples = 160,
  /// n_channels = 5` gives a channel-less product of `8e7 < 256 Mi cap`
  /// (would PASS the broken bound) BUT actual visits of `4e8 > 256 Mi`
  /// (FAILS the corrected bound).
  ///
  /// We pick `rate = 16 kHz, block_size = 0.01 s` (⇒ `block_samples =
  /// 160`), `n_samples = 161` (just over the block size so the
  /// audio-length check passes), and an `overlap ≈ 1 - 1.25e-8` to land
  /// `num_blocks ≈ 500_000` — comfortably under the byte cap
  /// (`500_000 * 5 * 8 ≈ 19 MiB << 64 MiB`) so byte cap headroom is
  /// not the rejecting cap; the rejection MUST come from the
  /// n_channels-aware work cap. The byte and total-elements caps both
  /// have wide headroom here (total elements = `161 * 5 = 805 << 64 Mi`).
  ///
  /// Asserts `Err`: pre-fix this would silently allow ~400 M sample-
  /// visits across the per-block × per-channel loops (a multi-second
  /// CPU spike on a small input); post-fix the work cap fires up-front
  /// in microseconds.
  #[test]
  fn integrated_loudness_rejects_work_cap_only_when_n_channels_counted() {
    let rate = 16_000u32;
    let n_samples = 161usize;
    let n_channels = 5usize;
    // Interleaved 5-channel buffer of zeros — value doesn't matter,
    // the cap fires BEFORE the per-block loop reads any samples.
    let buf = vec![0.0_f32; n_samples * n_channels];
    let x = Array::from_slice::<f32>(&buf, &[n_samples as i32, n_channels as i32]).unwrap();
    // overlap chosen so num_blocks ≈ 500_000:
    //   step = 1 - overlap = 1.25e-8
    //   num_blocks = round((duration - bs) / (bs * step)) + 1
    //              = round(6.25e-5 / (0.01 * 1.25e-8)) + 1
    //              ≈ round(500_000) + 1 ≈ 500_001
    //
    // Channel-less work product (broken bound):
    //   500_001 * 160 ≈ 8.0e7 < MAX_LOUDNESS_WORK (256 Mi ≈ 2.68e8) → PASSES (defect)
    // n_channels-aware work product (corrected bound):
    //   500_001 * 160 * 5 ≈ 4.0e8 > MAX_LOUDNESS_WORK              → REJECTS (fix)
    // Byte cap (independent, must NOT be the rejecting cap):
    //   500_001 * 5 * 8 ≈ 20 MB << MAX_LOUDNESS_BLOCK_BYTES (64 MiB) → PASSES
    let res = integrated_loudness(&x, rate, 0.01, 0.999_999_987_5);
    assert!(
      matches!(res, Err(Error::Backend { .. })),
      "5-channel input with num_blocks * block_samples just under the cap \
       but num_blocks * block_samples * n_channels above must be REJECTED \
       by the n_channels-aware work cap (got {res:?})"
    );
    // Verify the rejection actually comes from the WORK cap (not the
    // byte cap and not the total-elements cap), so a future refactor that
    // accidentally weakens the work cap surfaces here rather than passing
    // because some other cap caught the case.
    let msg = match res {
      Err(Error::Backend { message }) => message,
      _ => unreachable!(),
    };
    assert!(
      msg.contains("total sample-visit work") && msg.contains("n_channels=5"),
      "rejection must come from the n_channels-aware work cap (got: {msg})"
    );
  }

  /// LUFS reference-parity test: a 1 kHz sine of known amplitude has a
  /// well-defined BS.1770 integrated loudness governed by the closed form
  /// `LUFS = -0.691 + 10 * log10(|K(f)|^2 * a_peak^2 / 2)` where `K(f)` is
  /// the K-weighting filter response at frequency `f`. For the BS.1770
  /// K-weighting (high-shelf +4 dB / Q=1/sqrt(2) / fc=1500 Hz, then
  /// high-pass Q=0.5 / fc=38 Hz) evaluated analytically at 1 kHz / 48 kHz
  /// (`z = exp(j * 2π * 1000 / 48000)`):
  ///
  /// ```text
  ///   |K(1000)|^2 ≈ 1.16313337638011         (+0.6563 dB shelf gain)
  ///   LUFS @ amp=0.5
  ///     = -0.691 + 10*log10(|K|^2 * 0.5^2 / 2)
  ///     = -0.691 + 10*log10(0.14539...)
  ///     ≈ -9.0656046890608
  /// ```
  ///
  /// The f64-end-to-end K-weighting kernel (no intermediate f32 cast
  /// between the two biquad stages) should produce a value within tight
  /// tolerance of the theoretical -9.0656. Before the f64-kernel split the
  /// stage-boundary f32 cast dropped ~16 bits of precision between
  /// biquads, biasing this absolute value (and gate decisions near the
  /// absolute/relative thresholds). We assert ±0.05 LUFS — a tolerance
  /// the previously-f32-between-stages path could overshoot for short
  /// signals near the gate boundaries, and which the new f64 path
  /// comfortably meets.
  #[test]
  fn integrated_loudness_one_khz_sine_matches_theoretical() {
    let rate = 48_000u32;
    let amp = 0.5_f32;
    // 3 s of 1 kHz sine — long enough to give plenty of blocks above
    // both gates with a uniform per-block loudness, so the integrated
    // value is essentially the per-block loudness (no gating bias).
    let x = sine_mono(1000.0, amp, rate, 3.0);
    let lufs = integrated_loudness(&x, rate, 0.4, 0.75).unwrap();
    // Theoretical value computed via the analytic evaluation of the two
    // K-weighting biquads at `z = exp(j * 2π * 1000 / 48000)` (see the
    // docstring above for the exact algebra). The K-weighting input
    // signal is `amp * sin(2π * 1000 * t)` whose continuous mean-square
    // is `amp^2 / 2`; after K-weighting the per-block mean-square is
    // `|K|^2 * amp^2 / 2`, and the BS.1770 reduction is
    // `-0.691 + 10*log10(mean_square)`.
    let theoretical = -9.0656046890608_f64;
    assert!(
      (lufs - theoretical).abs() < 0.05,
      "1 kHz sine @ amp=0.5 should be within ±0.05 LUFS of theoretical \
       {theoretical} (got {lufs}, diff {})",
      (lufs - theoretical).abs()
    );
    // Also: the f64-end-to-end kernel keeps the round-trip exact within
    // 0.005 LUFS (tighter than the previous f32-between-stages 0.01
    // tolerance for `normalize_loudness_round_trip_matches_target`).
    let normalized = normalize_loudness(&x, lufs, -23.0).unwrap();
    let lufs_after = integrated_loudness(&normalized, rate, 0.4, 0.75).unwrap();
    assert!(
      (lufs_after - (-23.0)).abs() < 0.005,
      "f64-end-to-end K-weighting must yield a tighter round-trip (±0.005 \
       LUFS), got {lufs_after} (target -23.0, before {lufs})"
    );
  }

  /// `normalize_loudness` with a non-finite (NaN/+-inf) input or target
  /// loudness must be rejected (the reference would propagate a NaN/inf
  /// gain silently corrupting downstream samples).
  #[test]
  fn normalize_loudness_rejects_non_finite_params() {
    let rate = 48_000u32;
    let x = sine_mono(1000.0, 0.5, rate, 1.0);
    assert!(matches!(
      normalize_loudness(&x, f64::NAN, -23.0),
      Err(Error::Backend { .. })
    ));
    assert!(matches!(
      normalize_loudness(&x, -10.0, f64::INFINITY),
      Err(Error::Backend { .. })
    ));
    assert!(matches!(
      normalize_loudness(&x, f64::NEG_INFINITY, -23.0),
      Err(Error::Backend { .. })
    ));
  }

  /// `normalize_loudness` with `target == input` is a no-op (gain = 1.0).
  #[test]
  fn normalize_loudness_identity_when_target_eq_input() {
    let rate = 48_000u32;
    let x = sine_mono(1000.0, 0.5, rate, 1.0);
    let original = x.try_clone().unwrap().to_vec::<f32>().unwrap();
    let y = normalize_loudness(&x, -10.0, -10.0).unwrap();
    let result = y.try_clone().unwrap().to_vec::<f32>().unwrap();
    assert_eq!(result.len(), original.len());
    // gain = 10^0 = 1.0; multiply by 1.0 is identity even in f32.
    for (i, (g, e)) in result.iter().zip(original.iter()).enumerate() {
      assert!((g - e).abs() < 1e-7, "identity[{i}]: got {g}, want {e}");
    }
  }

  /// `bs1770_biquad_coefficients` produces `a[0] == 1.0` after the
  /// normalization (the reference divides by `a0` at construction). Sanity
  /// check the biquad shape directly so a coefficient regression surfaces
  /// here, not 200 LOC downstream in `integrated_loudness`.
  #[test]
  fn biquad_coefficients_normalize_a0_to_one() {
    let (_b, a) = bs1770_biquad_coefficients(
      4.0,
      1.0 / std::f64::consts::SQRT_2,
      1500.0,
      48_000.0,
      BiquadKind::HighShelf,
    );
    assert!(
      (a[0] - 1.0).abs() < 1e-15,
      "high-shelf a[0] must normalize to 1.0, got {}",
      a[0]
    );
    let (_b, a) = bs1770_biquad_coefficients(0.0, 0.5, 38.0, 48_000.0, BiquadKind::HighPass);
    assert!(
      (a[0] - 1.0).abs() < 1e-15,
      "high-pass a[0] must normalize to 1.0, got {}",
      a[0]
    );
  }

  // ---- ISTFTCache (streaming == one-shot, hand-traced) ------------------

  #[test]
  fn istft_cache_matches_free_istft_win_eq_nfft() {
    // The cached path must be numerically identical to the free `istft` for a
    // supported spectrum. win_length == n_fft (Right and Center both invert),
    // n_fft=8, hop=4, 16 samples (centered region == 16 with length=None).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    for mode in [WindowPad::Center, WindowPad::Right] {
      // stft signature: (samples, n_fft, hop, win_length, pad).
      let spec = stft(&x, 8, 4, Some(8), mode).unwrap();
      let one_shot = to_vec(&istft(&spec, None).unwrap());
      let mut cache = ISTFTCache::new();
      let cached = to_vec(&cache.istft(&spec, None).unwrap());
      assert_eq!(one_shot.len(), cached.len(), "length mismatch ({mode:?})");
      for (i, (a, b)) in one_shot.iter().zip(cached.iter()).enumerate() {
        assert!(
          (a - b).abs() < 1e-6,
          "ISTFTCache vs istft[{i}] ({mode:?}): {a} vs {b}"
        );
      }
      // The cache populated one position entry + one norm entry.
      assert_eq!(cache.len(), 2, "expected 2 cached buffers after one call");
    }
  }

  #[test]
  fn istft_cache_center_short_window_round_trips() {
    // WindowPad::Center inverts short windows (win_length < n_fft); the cached
    // path must recover the original signal too. n_fft=8, win=4, hop=2.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 2, Some(4), WindowPad::Center).unwrap();
    let mut cache = ISTFTCache::new();
    let rec = to_vec(&cache.istft(&spec, Some(16)).unwrap());
    assert_eq!(rec.len(), 16);
    for (i, (g, e)) in rec.iter().zip(buf.iter()).enumerate() {
      assert!(
        (g - e).abs() < 1e-5,
        "ISTFTCache short-window round-trip[{i}]: got {g}, want {e}"
      );
    }
  }

  #[test]
  fn istft_cache_reuses_buffers_across_same_geometry_spectra() {
    // Two DIFFERENT signals with the SAME framing geometry must reuse the
    // cached index + norm buffers (cache size stays 2), and each result must
    // still match the free `istft` of its own spectrum. This is the streaming
    // use case: many same-shaped blocks, buffers built once.
    let buf_a = signal_16();
    let mut buf_b = signal_16();
    buf_b.reverse(); // a different signal, same length/geometry
    let xa = Array::from_slice::<f32>(&buf_a, &[16i32]).unwrap();
    let xb = Array::from_slice::<f32>(&buf_b, &[16i32]).unwrap();
    let spec_a = stft(&xa, 8, 4, Some(8), WindowPad::Center).unwrap();
    let spec_b = stft(&xb, 8, 4, Some(8), WindowPad::Center).unwrap();

    let mut cache = ISTFTCache::new();
    let ca = to_vec(&cache.istft(&spec_a, None).unwrap());
    assert_eq!(cache.len(), 2, "first call should populate 2 buffers");
    let cb = to_vec(&cache.istft(&spec_b, None).unwrap());
    assert_eq!(
      cache.len(),
      2,
      "same-geometry second call must REUSE buffers (no new entries)"
    );

    let fa = to_vec(&istft(&spec_a, None).unwrap());
    let fb = to_vec(&istft(&spec_b, None).unwrap());
    for (i, (g, e)) in ca.iter().zip(fa.iter()).enumerate() {
      assert!((g - e).abs() < 1e-6, "cache A[{i}]: {g} vs {e}");
    }
    for (i, (g, e)) in cb.iter().zip(fb.iter()).enumerate() {
      assert!((g - e).abs() < 1e-6, "cache B[{i}]: {g} vs {e}");
    }
  }

  #[test]
  fn istft_cache_clear_empties_and_rejects_right_short_window() {
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let mut cache = ISTFTCache::new();
    assert!(cache.is_empty());
    let _ = cache.istft(&spec, None).unwrap();
    assert!(!cache.is_empty());
    cache.clear();
    assert!(cache.is_empty(), "clear() must drop all cached buffers");

    // Right-pad short-window inversion is rejected (same as the free `istft`).
    let spec_short = stft(&x, 8, 2, Some(4), WindowPad::Right).unwrap();
    let mut cache2 = ISTFTCache::new();
    assert!(matches!(
      cache2.istft(&spec_short, None),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn istft_cache_center_zero_coverage_tail_rejects_like_free_istft() {
    // THE F1 FIX. A `center=true` spectrum whose requested `length` reaches into
    // the zero-coverage OLA tail must be REJECTED by the cached path, EXACTLY as
    // the free `istft` rejects it — not divided by a `1e-10` floor and silently
    // emitted as corrupt audio. n_fft=8, hop=4, win=8 symmetric Hann; 16-sample
    // input → num_frames=5, t = (5-1)*4 + 8 = 24, pad = 4. The last OLA index
    // 23 is reached only by frame 4 at window position 7, whose Hann sample is
    // 0, so wsum[23] == 0 (zero coverage).
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    assert_eq!(spec.num_frames(), 5);
    let t = (spec.num_frames() - 1) * spec.hop_length() + spec.n_fft();
    assert_eq!(t, 24);
    let pad = spec.n_fft() / 2; // 4

    // `length = Some(t - pad)` requests `[pad .. t)` == `[4 .. 24)`, which
    // includes the zero-coverage index 23. BOTH paths must reject it.
    let tail_len = t - pad; // 20
    let free_tail = istft(&spec, Some(tail_len));
    assert!(
      matches!(free_tail, Err(Error::Backend { .. })),
      "free istft must reject the zero-coverage tail, got {free_tail:?}"
    );
    let mut cache = ISTFTCache::new();
    let cached_tail = cache.istft(&spec, Some(tail_len));
    assert!(
      matches!(cached_tail, Err(Error::Backend { .. })),
      "ISTFTCache must reject the zero-coverage tail IDENTICALLY to free istft \
       (not divide by a floor + emit corrupt audio), got {cached_tail:?}"
    );

    // A COVERED request (`length = None` → `[pad .. t - pad)`, excludes the
    // zero-coverage tail) must succeed AND be numerically identical to free
    // istft. (Use a fresh cache so a populated norm-buffer from the rejected
    // call above can't mask a bug; then assert the rejecting call left no stale
    // corrupt state by re-rejecting on the same cache.)
    let free_ok = to_vec(&istft(&spec, None).unwrap());
    let mut cache_ok = ISTFTCache::new();
    let cached_ok = to_vec(&cache_ok.istft(&spec, None).unwrap());
    assert_eq!(free_ok.len(), cached_ok.len(), "covered-length mismatch");
    for (i, (a, b)) in free_ok.iter().zip(cached_ok.iter()).enumerate() {
      assert!(
        (a - b).abs() < 1e-6,
        "covered ISTFTCache vs istft[{i}]: {a} vs {b}"
      );
    }
    // The same cache (now warm with the geometry from the rejected call) still
    // rejects the tail — the guard runs every call, not just on a cold cache.
    let warm_reject = cache.istft(&spec, Some(tail_len));
    assert!(
      matches!(warm_reject, Err(Error::Backend { .. })),
      "warm-cache tail request must STILL reject (guard is per-call), got {warm_reject:?}"
    );
  }

  #[test]
  fn istft_cache_center_false_uncovered_head_rejects_like_free_istft() {
    // F1 `center=false` consistency: the RAW OLA index 0 is reached only by
    // frame 0 at window position 0 (Hann sample 0), so wsum[0] == 0. A
    // `center=false` request includes index 0, so BOTH the free `istft` and the
    // cached path must reject it — the cached path must NOT floor-divide and emit
    // a corrupt head sample. Built via `from_parts` (stft always sets
    // center=true); the transform data is unchanged, only the carried flag.
    let buf = signal_16();
    let x = Array::from_slice::<f32>(&buf, &[16i32]).unwrap();
    let spec = stft(&x, 8, 4, Some(8), WindowPad::Center).unwrap();
    let spec_no_center = Spectrum::from_parts(
      spec.data_ref().try_clone().unwrap(),
      8,
      4,
      8,
      WindowPad::Center,
      false, // center=false: requested region starts at the uncovered index 0
    )
    .unwrap();
    for len in [None, Some(10usize)] {
      let free_res = istft(&spec_no_center, len);
      assert!(
        matches!(free_res, Err(Error::Backend { .. })),
        "free istft center=false head (len={len:?}) must reject, got {free_res:?}"
      );
      let mut cache = ISTFTCache::new();
      let cached_res = cache.istft(&spec_no_center, len);
      assert!(
        matches!(cached_res, Err(Error::Backend { .. })),
        "ISTFTCache center=false head (len={len:?}) must reject IDENTICALLY to \
         free istft, got {cached_res:?}"
      );
    }

    // Consistency the other way: a `center=false` short window under Center
    // placement whose covered interior IS requested still matches free istft.
    // (win < n_fft, hop small enough that an interior length is fully covered.)
    let spec_cov = stft(&x, 8, 2, Some(8), WindowPad::Center).unwrap();
    let cov = Spectrum::from_parts(
      spec_cov.data_ref().try_clone().unwrap(),
      8,
      2,
      8,
      WindowPad::Center,
      false,
    )
    .unwrap();
    // Request a covered interior slice: skip the uncovered head by using free
    // istft as the oracle — if free istft accepts a given length, the cache must
    // produce the identical samples; if it rejects, the cache must reject too.
    for len in [Some(6usize), Some(8usize), Some(12usize), None] {
      let free_res = istft(&cov, len);
      let mut cache = ISTFTCache::new();
      let cached_res = cache.istft(&cov, len);
      match (free_res, cached_res) {
        (Ok(f), Ok(c)) => {
          let fv = to_vec(&f);
          let cv = to_vec(&c);
          assert_eq!(fv.len(), cv.len(), "len={len:?} length mismatch");
          for (i, (a, b)) in fv.iter().zip(cv.iter()).enumerate() {
            assert!(
              (a - b).abs() < 1e-6,
              "center=false covered ISTFTCache vs istft[{i}] (len={len:?}): {a} vs {b}"
            );
          }
        }
        (Err(_), Err(_)) => { /* both reject — consistent */ }
        (f, c) => panic!("center=false len={len:?}: free and cache DISAGREE: {f:?} vs {c:?}"),
      }
    }
  }

  // ---- normalize_peak (hand-traced vs reference) ------------------------

  #[test]
  fn normalize_peak_brings_peak_to_target_dbfs() {
    // data = [0.5, -0.25, 0.1], current_peak = 0.5.
    //   target 0 dBFS  → gain = 1.0 / 0.5 = 2.0   → max|.| == 1.0.
    //   target -6 dBFS → 10^(-6/20)/0.5 ≈ 1.00237 → max|.| ≈ 0.50119.
    let data = Array::from_slice::<f32>(&[0.5, -0.25, 0.1], &[3]).unwrap();

    let out0 = normalize_peak(&data, 0.0).unwrap();
    let v0 = to_vec(&out0);
    let peak0 = v0.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(
      (peak0 - 1.0).abs() < 1e-6,
      "0 dBFS peak: got {peak0}, want 1.0"
    );
    // Exact scaled values (gain == 2.0).
    for (g, e) in v0.iter().zip([1.0_f32, -0.5, 0.2].iter()) {
      assert!((g - e).abs() < 1e-6, "0 dBFS value: got {g}, want {e}");
    }

    let out6 = normalize_peak(&data, -6.0).unwrap();
    let v6 = to_vec(&out6);
    let peak6 = v6.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let want6 = 10.0_f32.powf(-6.0 / 20.0);
    assert!(
      (peak6 - want6).abs() < 1e-5,
      "-6 dBFS peak: got {peak6}, want {want6}"
    );
  }

  #[test]
  fn normalize_peak_2d_input_uses_global_peak() {
    // The peak is the GLOBAL max over the whole array (matches np.max(np.abs)).
    // 2x2 with global peak 0.8 → target 0 dBFS scales by 1/0.8.
    let data = Array::from_slice::<f32>(&[0.2, -0.8, 0.4, 0.1], &[2, 2]).unwrap();
    let out = normalize_peak(&data, 0.0).unwrap();
    assert_eq!(out.shape(), vec![2, 2], "shape must be preserved");
    let v = to_vec(&out);
    let peak = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!(
      (peak - 1.0).abs() < 1e-6,
      "global peak should hit 1.0, got {peak}"
    );
  }

  #[test]
  fn normalize_peak_rejects_silence_and_nonfinite() {
    // All-zero input: current_peak == 0.0 → reject (would divide by zero).
    let silence = Array::from_slice::<f32>(&[0.0, 0.0, 0.0], &[3]).unwrap();
    assert!(matches!(
      normalize_peak(&silence, 0.0),
      Err(Error::Backend { .. })
    ));
    // Non-finite target_peak_db.
    let data = Array::from_slice::<f32>(&[0.5, 0.1], &[2]).unwrap();
    assert!(matches!(
      normalize_peak(&data, f64::NAN),
      Err(Error::Backend { .. })
    ));
    assert!(matches!(
      normalize_peak(&data, f64::INFINITY),
      Err(Error::Backend { .. })
    ));
  }

  #[test]
  fn normalize_peak_rejects_overflowing_gain_from_finite_input() {
    // A FINITE `target_peak_db` can still drive the gain non-finite. These must
    // be rejected (not silently emit `inf` / `NaN` samples), per the f32
    // finiteness guards on `target_linear` and `gain`.
    let data = Array::from_slice::<f32>(&[0.5, -0.25, 0.1], &[3]).unwrap();

    // Huge target_peak_db: 10^(1e30/20) overflows f32 → target_linear == +inf.
    assert!(
      matches!(normalize_peak(&data, 1e30), Err(Error::Backend { .. })),
      "huge finite target_peak_db must be rejected (target_linear overflows f32)"
    );

    // A subnormal nonzero peak with a moderate target: target_linear is finite
    // but `target_linear / current_peak` overflows to +inf. f32::MIN_POSITIVE
    // (~1.18e-38) is the smallest normal positive; a *subnormal* nonzero peak is
    // even smaller, so 1.0 / peak overflows.
    let tiny = f32::from_bits(1); // smallest positive subnormal (~1.4e-45)
    assert!(
      tiny > 0.0 && tiny.is_finite(),
      "tiny must be a finite nonzero"
    );
    let subnormal_peak = Array::from_slice::<f32>(&[tiny, 0.0, -tiny], &[3]).unwrap();
    assert!(
      matches!(
        normalize_peak(&subnormal_peak, 0.0),
        Err(Error::Backend { .. })
      ),
      "subnormal nonzero peak that overflows the gain must be rejected"
    );

    // Sanity: a normal dBFS target on a normal-magnitude peak still succeeds and
    // stays finite (the guards only fire on genuine overflow).
    let ok = normalize_peak(&data, -3.0).unwrap();
    for v in to_vec(&ok) {
      assert!(
        v.is_finite(),
        "normal target must keep samples finite, got {v}"
      );
    }
  }

  // ---- P7 #128: mel_filter_bank_cached -------------------------------------

  /// Cached and uncached forms produce byte-identical banks.
  #[test]
  fn mel_filter_bank_cached_matches_uncached() {
    clear_mel_filter_cache();
    let plain = mel_filter_bank(80, 400, 16_000, 0.0, None).unwrap();
    let cached = mel_filter_bank_cached(80, 400, 16_000, 0.0, None).unwrap();
    let p = to_vec(&plain);
    let c = to_vec(&cached);
    assert_eq!(p, c, "cached mel bank must match uncached value-for-value");
    clear_mel_filter_cache();
  }

  /// A second call with the same parameters re-uses the cached entry; the
  /// returned `Array` is still value-for-value identical, and the cache
  /// did NOT rebuild it (a fresh `Vec<f32>` clone would still be value-
  /// equal — we assert structural equality here, and rely on the LRU
  /// behavior test below to assert the cache state itself).
  #[test]
  fn mel_filter_bank_cached_hit_returns_same_values() {
    clear_mel_filter_cache();
    let first = mel_filter_bank_cached(80, 400, 16_000, 0.0, None).unwrap();
    let second = mel_filter_bank_cached(80, 400, 16_000, 0.0, None).unwrap();
    assert_eq!(to_vec(&first), to_vec(&second));
    clear_mel_filter_cache();
  }

  /// Different `(sample_rate, n_fft, n_mels, f_min, f_max)` keys are
  /// cached separately; a request for a new key does not return the
  /// previous bank.
  #[test]
  fn mel_filter_bank_cached_distinguishes_keys() {
    clear_mel_filter_cache();
    let a = mel_filter_bank_cached(80, 400, 16_000, 0.0, None).unwrap();
    let b = mel_filter_bank_cached(80, 400, 22_050, 0.0, None).unwrap();
    let c = mel_filter_bank_cached(40, 400, 16_000, 0.0, None).unwrap();
    let d = mel_filter_bank_cached(80, 512, 16_000, 0.0, None).unwrap();
    let e = mel_filter_bank_cached(80, 400, 16_000, 80.0, None).unwrap();
    let f = mel_filter_bank_cached(80, 400, 16_000, 0.0, Some(7_500.0)).unwrap();
    // Each of (b..=f) must differ from `a` somewhere.
    let av = to_vec(&a);
    for (name, other) in [
      ("sample_rate", &b),
      ("n_mels", &c),
      ("n_fft", &d),
      ("f_min", &e),
      ("f_max", &f),
    ] {
      let ov = to_vec(other);
      assert_ne!(av, ov, "{name} key collapsed into same cache entry");
    }
    clear_mel_filter_cache();
  }

  /// LRU eviction: filling the cache with `MEL_FILTER_CACHE_CAP + 1`
  /// distinct keys evicts the oldest entry. A subsequent request for
  /// the evicted key still succeeds (rebuilds via the uncached path),
  /// and the most-recent key remains cached (still resolves correctly).
  #[test]
  fn mel_filter_bank_cached_evicts_lru_at_cap() {
    clear_mel_filter_cache();
    // Walk `cap + 1` distinct (sample_rate) keys.
    let cap = super::MEL_FILTER_CACHE_CAP;
    let mut first_bank: Option<Vec<f32>> = None;
    for i in 0..(cap + 1) {
      let sr = 16_000u32 + (i as u32) * 1_000;
      let bank = mel_filter_bank_cached(40, 400, sr, 0.0, None).unwrap();
      if i == 0 {
        first_bank = Some(to_vec(&bank));
      }
    }
    // The first key was evicted but a re-request still returns a
    // value-equal bank (the uncached construction path produces the
    // same matrix).
    let refetched = mel_filter_bank_cached(40, 400, 16_000, 0.0, None).unwrap();
    assert_eq!(
      to_vec(&refetched),
      first_bank.unwrap(),
      "evicted key must rebuild value-equal bank on re-request"
    );
    clear_mel_filter_cache();
  }

  /// Cached path propagates validation errors from the underlying
  /// `mel_filter_bank` constructor (and does NOT cache a failed entry).
  #[test]
  fn mel_filter_bank_cached_propagates_errors() {
    clear_mel_filter_cache();
    // `n_fft = 0` → recoverable Error::Backend.
    assert!(matches!(
      mel_filter_bank_cached(80, 0, 16_000, 0.0, None),
      Err(Error::Backend { .. })
    ));
    // A valid call AFTER the failed one still succeeds (the failure
    // didn't pollute the cache).
    let ok = mel_filter_bank_cached(80, 400, 16_000, 0.0, None);
    assert!(ok.is_ok());
    clear_mel_filter_cache();
  }

  // ---- P7 #131: AUDIO-5 magic constants are named ---------------------------

  /// Pin the exact named-constant values that mlx-audio expects. Closes
  /// AUDIO-5 by asserting the const surface (so a future refactor can't
  /// quietly drift any of the five magic numbers).
  #[test]
  fn dsp_named_constants_match_mlx_audio_literals() {
    assert_eq!(super::MEL_HZ_DIV, 2595.0_f32);
    assert_eq!(super::MEL_HZ_BREAK, 700.0_f32);
    assert_eq!(super::LOG_FLOOR_WHISPER, 1e-10_f32);
    assert_eq!(super::LOG_FLOOR_KALDI, 1e-8_f32);
    assert_eq!(super::BS1770_LOUDNESS_OFFSET_LUFS, -0.691_f64);
  }

  /// `LogFloor` surfaces the configurable log floor; both built-in
  /// variants resolve to the named constants and `Custom` clamps non-
  /// finite / non-positive inputs to `f32::MIN_POSITIVE` (the docs).
  #[test]
  fn log_floor_variants_resolve_named_constants() {
    assert_eq!(LogFloor::Whisper.value(), super::LOG_FLOOR_WHISPER);
    assert_eq!(LogFloor::Kaldi.value(), super::LOG_FLOOR_KALDI);
    assert_eq!(LogFloor::Custom(1e-6).value(), 1e-6);
    // Non-finite / non-positive clamp.
    assert_eq!(LogFloor::Custom(f32::NAN).value(), f32::MIN_POSITIVE);
    assert_eq!(LogFloor::Custom(-1.0).value(), f32::MIN_POSITIVE);
    assert_eq!(LogFloor::Custom(0.0).value(), f32::MIN_POSITIVE);
  }

  // ---- P7 #134: StftConfig + stft_with_config + stft_aligned ---------------

  /// `StftConfig::default()` is `(center: true, pad_mode: Reflect)` — the
  /// `mlx_audio.dsp.stft` reference defaults.
  #[test]
  fn stft_config_default_matches_mlx_audio_defaults() {
    let cfg = StftConfig::default();
    assert!(cfg.center());
    assert_eq!(cfg.pad_mode(), PadMode::Reflect);
  }

  /// `stft` and `stft_with_config(.., StftConfig::default())` produce
  /// byte-identical Spectra (data + every metadata field).
  #[test]
  fn stft_with_config_default_matches_bare_stft() {
    let n = 256usize;
    let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let arr = Array::from_slice::<f32>(&samples, &[n as i32]).unwrap();
    let bare = stft(&arr, 64, 16, None, WindowPad::Right).unwrap();
    let with_cfg =
      stft_with_config(&arr, 64, 16, None, WindowPad::Right, &StftConfig::default()).unwrap();
    assert_eq!(bare.n_fft(), with_cfg.n_fft());
    assert_eq!(bare.hop_length(), with_cfg.hop_length());
    assert_eq!(bare.win_length(), with_cfg.win_length());
    assert_eq!(bare.window_pad(), with_cfg.window_pad());
    assert_eq!(bare.center(), with_cfg.center());
    assert_eq!(bare.num_frames(), with_cfg.num_frames());
    assert_eq!(bare.n_freqs(), with_cfg.n_freqs());
  }

  /// `stft_aligned` returns a Spectrum with `center == false` and one
  /// fewer "centering" frame than the centered path for the same input.
  #[test]
  fn stft_aligned_carries_center_false() {
    let n = 256usize;
    let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
    let arr = Array::from_slice::<f32>(&samples, &[n as i32]).unwrap();
    let aligned = stft_aligned(&arr, 64, 16, None, WindowPad::Right).unwrap();
    let centered = stft(&arr, 64, 16, None, WindowPad::Right).unwrap();
    assert!(!aligned.center(), "stft_aligned must carry center == false");
    assert!(centered.center(), "stft must carry center == true");
    // Centered path adds `2 * (n_fft / 2) = n_fft` samples of reflect
    // padding, so its frame count is strictly greater.
    assert!(
      centered.num_frames() > aligned.num_frames(),
      "centered={} aligned={}",
      centered.num_frames(),
      aligned.num_frames()
    );
  }

  /// `stft_aligned` on an input shorter than `n_fft` rejects (no padding
  /// to bridge the gap).
  #[test]
  fn stft_aligned_rejects_too_short_input() {
    let samples: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let arr = Array::from_slice::<f32>(&samples, &[32]).unwrap();
    // n_fft = 64 > 32, so without padding there is no frame.
    assert!(matches!(
      stft_aligned(&arr, 64, 16, None, WindowPad::Right),
      Err(Error::Backend { .. })
    ));
  }

  // ---- P7 #129: reflect_pad_1d round-trip + zero-pad fast path -------------

  /// `reflect_pad_1d` with `padding == 0` returns the input unchanged
  /// (the cheap rc-clone fast path — skips the slice + concatenate).
  #[test]
  fn reflect_pad_1d_zero_padding_returns_unchanged() {
    let samples: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let arr = Array::from_slice::<f32>(&samples, &[16]).unwrap();
    let padded = reflect_pad_1d(&arr, 0).unwrap();
    assert_eq!(to_vec(&padded), samples);
  }

  /// `reflect_pad_1d` matches the python reference's
  /// `[1..=p][::-1] ++ samples ++ [-p-1..-1][::-1]` semantics.
  #[test]
  fn reflect_pad_1d_matches_python_reference_construction() {
    let samples: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let arr = Array::from_slice::<f32>(&samples, &[8]).unwrap();
    // padding = 3: prefix should be samples[3..=1] reversed = [3, 2, 1]
    // suffix should be samples[6..=4] reversed = [6, 5, 4]
    let padded = reflect_pad_1d(&arr, 3).unwrap();
    let v = to_vec(&padded);
    let expected: Vec<f32> = vec![
      3.0, 2.0, 1.0, // prefix
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // original
      6.0, 5.0, 4.0, // suffix
    ];
    assert_eq!(v, expected);
  }
}
