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
//! - Mel filterbank defaults to the HTK formula
//!   (`mel = 2595 * log10(1 + hz / 700)`) and returns shape
//!   **`(n_mels, n_fft / 2 + 1)`**. The Slaney scale + Slaney area
//!   normalization (the Whisper front-end) are available via
//!   [`mel_filter_bank_scaled`] / [`MelScale`].
//! - `log_mel_spectrogram` uses `log(max(mel, floor))` with `floor` chosen
//!   via the [`LogFloor`] enum (default [`LogFloor::Whisper`] = `1e-10`,
//!   matching the Whisper / mlx-audio front-end). [`LogFloor::Kaldi`] =
//!   `1e-8` matches the floor literal in `mlx-audio/mlx_audio/dsp.py:950`
//!   — floor-constant parity only; the upstream mel-filterbank
//!   `get_mel_banks_kaldi` path is out of scope (see the per-variant
//!   `LogFloor::Kaldi` docs). Tracks mlx-audio's literal, NOT the
//!   upstream kaldi-asr `FbankComputer` floor of `f32::EPSILON`.

use smol_str::format_smolstr;

use crate::{
  Array, Error, Result,
  error::{
    AllocFailurePayload, ArithmeticOverflowPayload, CapExceededPayload, DtypeMismatchPayload,
    EmptyInputPayload, InvariantViolationPayload, LengthMismatchPayload, NonFiniteScalarPayload,
    OutOfRangePayload, RankMismatchPayload, UnknownEnumValuePayload,
  },
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
///   recoverable [`Error::OutOfRange`]. Use [`WindowPad::Center`] for short-window
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
  /// Named `data_ref` (not `data`) per the non-Copy `&T` accessor naming
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
  /// Returns typed errors when:
  /// - `data` is not 2-D → [`Error::RankMismatch`],
  /// - `n_fft == 0`, `hop_length == 0`, `win_length == 0`, or `num_frames == 0`
  ///   → [`Error::InvariantViolation`],
  /// - `data`'s last dimension `!= n_fft / 2 + 1` → [`Error::LengthMismatch`],
  /// - `n_fft` is **odd** or `win_length > n_fft` → [`Error::OutOfRange`],
  /// - `data`'s dtype is not `Complex64` → [`Error::DtypeMismatch`].
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
      let rank = shape.len() as u32;
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "Spectrum::from_parts: data must be 2-D (num_frames, n_freqs)",
        rank,
        shape,
      )));
    }
    if n_fft == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Spectrum::from_parts: n_fft",
        "must be > 0",
      )));
    }
    // Reject odd `n_fft`: a one-sided spectrum has `n_freqs == n_fft / 2 + 1`
    // for both `n_fft = 2k` and `2k + 1`, so an odd-`n_fft` spectrum cannot be
    // inverted unambiguously. Closing it here means a `Spectrum` can never
    // carry odd metadata regardless of how it was constructed.
    if !n_fft.is_multiple_of(2) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Spectrum::from_parts: n_fft",
        "must be even (odd n_fft is unsupported because the one-sided spectrum \
           cannot be inverted unambiguously)",
        format_smolstr!("{n_fft}"),
      )));
    }
    if hop_length == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Spectrum::from_parts: hop_length",
        "must be > 0",
      )));
    }
    if win_length == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Spectrum::from_parts: win_length",
        "must be > 0",
      )));
    }
    if win_length > n_fft {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Spectrum::from_parts: win_length (the window cannot exceed the irfft frame)",
        "must be <= n_fft",
        format_smolstr!("win_length={win_length}, n_fft={n_fft}"),
      )));
    }
    let num_frames = shape[0];
    if num_frames == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Spectrum::from_parts: num_frames",
        "must be > 0",
      )));
    }
    let n_freqs = shape[1];
    // `n_fft` is even, so `n_fft / 2 + 1` is the exact one-sided bin count.
    let expected_n_freqs = n_fft / 2 + 1;
    if n_freqs != expected_n_freqs {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "Spectrum::from_parts: data last dim must equal n_fft/2 + 1 \
           (the bin count must match the declared n_fft)",
        expected_n_freqs,
        n_freqs,
      )));
    }
    if data.dtype()? != crate::Dtype::Complex64 {
      return Err(Error::DtypeMismatch(DtypeMismatchPayload::new(
        crate::Dtype::Complex64,
        data.dtype()?,
      )));
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
/// - [`Error::RankMismatch`] if `w` is not 1-D, [`Error::LengthMismatch`] if its
///   length is not exactly `win_length`, [`Error::OutOfRange`] if `win_length > n_fft`
///   or a pad extent exceeds `i32::MAX`.
fn place_window(
  caller: &'static str,
  w: &Array,
  win_length: usize,
  n_fft: usize,
  window_pad: WindowPad,
) -> Result<Array> {
  if w.ndim() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      caller,
      w.ndim() as u32,
      w.shape(),
    )));
  }
  let w_len = w.shape()[0];
  if w_len != win_length {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      caller, win_length, w_len,
    )));
  }
  if win_length > n_fft {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      caller,
      "win_length must be <= n_fft",
      format_smolstr!("win_length={win_length}, n_fft={n_fft}"),
    )));
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
  let low_i32 = i32::try_from(low).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      caller,
      "window pad-low must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{low}"),
    ))
  })?;
  let high_i32 = i32::try_from(high).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      caller,
      "window pad-high must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{high}"),
    ))
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
/// mismatch (a classic source of silent round-trip corruption) is
/// structurally impossible. `istft` takes no custom-window input
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
/// to the SIMD window builder
/// ([`crate::simd::audio::window::symmetric_window`]).
///
/// `name` only flavors the error messages so each public window keeps its
/// own diagnostic prefix; `kind` selects the per-window formula (Hann /
/// Hamming / Blackman / Bartlett). On `aarch64` the SIMD dispatcher
/// routes to a 7-term Taylor cos polynomial NEON 4-lane tile;
/// elsewhere it falls back to the per-element `f32::cos` scalar loop.
fn symmetric_window(
  name: &'static str,
  n: usize,
  kind: crate::simd::audio::window::SymWindowKind,
) -> Result<Array> {
  if n < 2 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      name,
      "n must be >= 2",
      format_smolstr!("{n}"),
    )));
  }
  // Cap on public-input-driven allocation — defends against an
  // adversarial / fuzzer-supplied `n = usize::MAX` that would otherwise
  // attempt a 16 EiB infallible allocation. Real-world windows are
  // typically <= a few thousand samples; 64 Mi-samples (256 MiB of f32)
  // is a generous ceiling that still excludes pathological inputs.
  if n > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      name,
      "MAX_DECODED_SAMPLES",
      crate::audio::io::MAX_DECODED_SAMPLES as u64,
      n as u64,
    )));
  }
  let n_i32 = i32::try_from(n).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      name,
      "n must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n}"),
    ))
  })?;

  // SIMD: dispatch to the symmetric-window NEON kernel
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
/// - Returns [`Error::OutOfRange`] when `n < 2`. The reference Python form
///   would divide by zero for `n == 1` (silently producing `NaN`); we
///   reject upfront. `n == 0` would produce an empty zero-length window
///   which is never useful for spectral analysis.
/// - Returns [`Error::CapExceeded`] when `n` exceeds the
///   [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap;
///   [`Error::OutOfRange`] if `n > i32::MAX`; or if the backing allocation fails.
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
/// - [`Error::UnknownEnumValue`] for an unknown window name (mirrors the reference's
///   `ValueError(f"Unknown window function: {window}")`).
/// - Propagates the constructor errors of the selected window (see
///   [`hann_window`]).
pub fn window_from_name(name: &str, n: usize) -> Result<Array> {
  match name.to_ascii_lowercase().as_str() {
    "hann" | "hanning" => hann_window(n),
    "hamming" => hamming_window(n),
    "blackman" => blackman_window(n),
    "bartlett" => bartlett_window(n),
    other => Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "window_from_name",
      other,
      &["hann", "hanning", "hamming", "blackman", "bartlett"],
    ))),
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
/// - [`Error::OutOfRange`] if `padding > samples_len - 1` (not enough samples
///   to reflect — would require `samples[len-padding-1]` which underflows
///   for `padding >= len`). The reference Python form would index out of
///   bounds and return a malformed array.
fn reflect_pad_1d(samples: &Array, padding: usize) -> Result<Array> {
  if padding == 0 {
    return samples.try_clone();
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    let rank = shape.len() as u32;
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "reflect_pad_1d: expected 1-D input",
      rank,
      shape,
    )));
  }
  let len = shape[0];
  // Need indices `samples[1..=padding]` AND `samples[len-padding-1..len-1]`
  // to exist — i.e. `len >= padding + 1`.
  if len < padding + 1 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "reflect_pad_1d: samples len for reflect padding",
      "must be >= padding + 1",
      format_smolstr!("len={len}, padding={padding}"),
    )));
  }

  let p_i32 = i32::try_from(padding).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "reflect_pad_1d: padding",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{padding}"),
    ))
  })?;
  let len_i32 = i32::try_from(len).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "reflect_pad_1d: samples len",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{len}"),
    ))
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
    // Overflow note: both `padding` and `len`
    // were checked to fit `i32` above via `i32::try_from`; combined with
    // `len == padding + 1` in this branch (`padding + 1 >= len` from the
    // else condition, and `len >= padding + 1` from the early check),
    // `len_i32` can be exactly `i32::MAX` (when `padding = i32::MAX - 1`,
    // `len = i32::MAX`). `len_i32 + 1` then overflows. Compute the
    // sentinel in `i64` and reject the (vanishingly rare) overflow as a
    // recoverable `Error::OutOfRange` rather than debug-panicking / wrapping.
    let sentinel_i64 = -(i64::from(len_i32) + 1);
    i32::try_from(sentinel_i64).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "reflect_pad_1d: reflect-pad sentinel `-(len + 1)` \
         (len == padding + 1, near i32::MAX boundary)",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("sentinel={sentinel_i64}, len={len}"),
      ))
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
/// 1)` with checked arithmetic and rejects (recoverable [`Error::CapExceeded`] /
/// [`Error::ArithmeticOverflow`]) any combination that overflows or exceeds
/// `MAX_STFT_WORK`. A lazily-shaped huge
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
/// - Typed errors: [`Error::RankMismatch`] if `samples` is not 1-D;
///   [`Error::InvariantViolation`] if `n_fft == 0`, `hop_length == 0`, or
///   `win_length == 0`; [`Error::OutOfRange`] if `n_fft` is odd, `win_length
///   > n_fft`, the post-pad count is too short, or any size exceeds `i32::MAX`;
///   [`Error::CapExceeded`] if the input/padded length or frame work exceeds
///   the cap; [`Error::ArithmeticOverflow`] if frame work overflows `usize`.
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
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stft: n_fft",
      "must be > 0",
    )));
  }
  // Reject odd `n_fft` up front (before any framing/FFT work): a one-sided
  // real-FFT spectrum has `n_freqs == n_fft / 2 + 1` for both `n_fft = 2k` and
  // `2k + 1`, so the bin count alone cannot disambiguate odd from the adjacent
  // even length. Keeping the producer even-only means the `Spectrum` this
  // returns can never carry an odd `n_fft`, so its inverse is unambiguous.
  if !n_fft.is_multiple_of(2) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stft: n_fft",
      "must be even (odd n_fft is unsupported because the one-sided spectrum \
         cannot be inverted unambiguously)",
      format_smolstr!("{n_fft}"),
    )));
  }
  if hop_length == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stft: hop_length",
      "must be > 0",
    )));
  }
  let win_length = win_length.unwrap_or(n_fft);
  if win_length == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stft: win_length",
      "must be > 0",
    )));
  }
  if win_length > n_fft {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stft: win_length",
      "must be <= n_fft (unsupported)",
      format_smolstr!("win_length={win_length}, n_fft={n_fft}"),
    )));
  }
  let shape = samples.shape();
  if shape.len() != 1 {
    let rank = shape.len() as u32;
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "stft: expected 1-D input",
      rank,
      shape,
    )));
  }

  // INPUT-LENGTH CAP (OOM guard). When `cfg.center == true` the
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
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "stft: input sample count exceeds sample budget \
         (would force a reflect-pad allocation proportional to the input)",
      "MAX_DECODED_SAMPLES",
      crate::audio::io::MAX_DECODED_SAMPLES as u64,
      samples_len as u64,
    )));
  }
  if cfg.center() {
    let padded_len_budget = samples_len.checked_add(n_fft).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "stft: padded length samples_len + n_fft",
        "usize",
        [("samples_len", samples_len as u64), ("n_fft", n_fft as u64)],
      ))
    })?;
    if padded_len_budget > crate::audio::io::MAX_DECODED_SAMPLES {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "stft: padded length (= samples_len + n_fft) exceeds sample budget \
           (would force a reflect-pad allocation proportional to the input)",
        "MAX_DECODED_SAMPLES",
        crate::audio::io::MAX_DECODED_SAMPLES as u64,
        padded_len_budget as u64,
      )));
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
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stft: padded_len (input is too short for n_fft)",
      "must be >= n_fft",
      format_smolstr!("padded_len={padded_len}, n_fft={n_fft}"),
    )));
  }
  let num_frames = 1 + (padded_len - n_fft) / hop_length;
  if num_frames == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stft: num_frames (input is too short for n_fft/hop_length)",
      "must be >= 1",
      format_smolstr!("num_frames=0, n_fft={n_fft}, hop_length={hop_length}"),
    )));
  }

  // SAFETY pre-condition: the reachable element range of the strided view
  // is `(num_frames - 1) * hop_length + n_fft - 1`. We assert this is
  // strictly less than `padded_len`, so every read is in-bounds.
  let last_element_index = (num_frames - 1)
    .checked_mul(hop_length)
    .and_then(|v| v.checked_add(n_fft))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "stft: reachable element range ((num_frames - 1) * hop_length + n_fft)",
        "usize",
        [
          ("num_frames", num_frames as u64),
          ("hop_length", hop_length as u64),
          ("n_fft", n_fft as u64),
        ],
      ))
    })?;
  if last_element_index > padded_len {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stft: derived frame reach (internal invariant violated)",
      "must be <= padded_len",
      format_smolstr!(
        "last_element_index={last_element_index}, padded_len={padded_len}, \
         num_frames={num_frames}, hop_length={hop_length}, n_fft={n_fft}"
      ),
    )));
  }

  // WORK CAP. Mirror `istft`'s `MAX_OLA_WORK` guard on the
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
  let frame_work = num_frames.checked_mul(n_fft).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "stft: frame work count num_frames * n_fft",
      "usize",
      [("num_frames", num_frames as u64), ("n_fft", n_fft as u64)],
    ))
  })?;
  if frame_work > MAX_STFT_WORK {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "stft: frame work count (= num_frames * n_fft) exceeds work cap",
      "MAX_STFT_WORK",
      MAX_STFT_WORK as u64,
      frame_work as u64,
    )));
  }
  // `n_fft` is even (checked above), so `n_fft / 2 + 1` cannot overflow.
  let out_elems = num_frames.checked_mul(n_fft / 2 + 1).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "stft: output element count num_frames * (n_fft/2 + 1)",
      "usize",
      [("num_frames", num_frames as u64), ("n_fft", n_fft as u64)],
    ))
  })?;
  if out_elems > MAX_STFT_WORK {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "stft: output element count (= num_frames * (n_fft/2 + 1)) exceeds work cap",
      "MAX_STFT_WORK",
      MAX_STFT_WORK as u64,
      out_elems as u64,
    )));
  }

  // Analysis window via the SHARED `frame_window` (symmetric hann of
  // `win_length`, placed into the `n_fft` frame per `window_pad`; no-op when
  // `win_length == n_fft`). Built AFTER the work cap so a lazily-shaped huge
  // input is rejected before this CPU `Vec` is allocated. `istft` rebuilds its
  // synthesis window through the EXACT same call, so analysis and synthesis
  // windows always match by construction.
  let window = frame_window(win_length, n_fft, window_pad)?;

  let num_frames_i32 = i32::try_from(num_frames).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "stft: num_frames",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{num_frames}"),
    ))
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "stft: n_fft",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n_fft}"),
    ))
  })?;
  let hop_i64 = i64::try_from(hop_length).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "stft: hop_length",
      "must fit in i64 (i64::MAX = 9223372036854775807)",
      format_smolstr!("{hop_length}"),
    ))
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
///   recoverable [`Error::OutOfRange`] (such a [`Spectrum`] is a valid forward
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
/// meaningless, so [`istft`] returns a recoverable [`Error::OutOfRange`] naming
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
/// - Typed errors: [`Error::OutOfRange`] if `window_pad` is Right with
///   `win_length != n_fft`, the coverage guard fires, sizes exceed `i32::MAX`,
///   or the `length` trim is out of range;
///   [`Error::CapExceeded`] if the OLA length or scatter work exceeds the cap;
///   [`Error::ArithmeticOverflow`] if derived sizes overflow `usize`.
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
    let rank = shape.len() as u32;
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "istft: expected 2-D (num_frames, n_freqs) spectrum data",
      rank,
      shape,
    )));
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
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "istft: win_length with WindowPad::Right \
         (right-pad short-window inversion is not a faithful inverse — \
         use WindowPad::Center for short-window inversion)",
      "must equal n_fft",
      format_smolstr!("win_length={win_length}, n_fft={n_fft}"),
    )));
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
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "istft: OLA length ((num_frames - 1) * hop + n_fft)",
        "usize",
        [
          ("num_frames", num_frames as u64),
          ("hop_length", hop_length as u64),
          ("n_fft", n_fft as u64),
        ],
      ))
    })?;
  if t > crate::audio::io::MAX_DECODED_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "istft: OLA length exceeds cap",
      "MAX_DECODED_SAMPLES",
      crate::audio::io::MAX_DECODED_SAMPLES as u64,
      t as u64,
    )));
  }

  // OOM guard on the *real* scatter/update workload (`num_frames *
  // frame_width`), checked BEFORE any broadcast / flatten / `try_reserve`.
  // The `t` cap above bounds the *output* length, but with small hops the
  // scatter touches far more elements than `t` (e.g. num_frames=65536,
  // n_fft=65536, hop=1 → t≈131071 but idx_len≈4.29e9). Reject overflow,
  // `> i32::MAX`, and `> MAX_OLA_WORK` here so a shaped/lazy input can never
  // drive a multi-GB allocation downstream.
  let idx_len = num_frames.checked_mul(frame_width).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "istft: scatter work count num_frames * n_fft",
      "usize",
      [("num_frames", num_frames as u64), ("n_fft", n_fft as u64)],
    ))
  })?;
  if idx_len > MAX_OLA_WORK {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "istft: scatter work count (= num_frames * n_fft) exceeds work cap",
      "MAX_OLA_WORK",
      MAX_OLA_WORK as u64,
      idx_len as u64,
    )));
  }
  let idx_len_i32 = i32::try_from(idx_len).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: scatter work count",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{idx_len}"),
    ))
  })?;

  let t_i32 = i32::try_from(t).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: OLA length",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{t}"),
    ))
  })?;
  let n_fft_i32 = i32::try_from(n_fft).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: n_fft",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n_fft}"),
    ))
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
  idx_buf.try_reserve_exact(idx_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "istft: index reservation",
      "i32 elements",
      idx_len as u64,
      e,
    ))
  })?;
  let frame_width_i32 = i32::try_from(frame_width).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: frame_width (n_fft)",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{frame_width}"),
    ))
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
      let end = pad.checked_add(len).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "istft: center offset pad + length",
          "usize",
          [("pad", pad as u64), ("len", len as u64)],
        ))
      })?;
      if end > t {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "istft: center offset pad + length",
          "must be <= reconstruction length t",
          format_smolstr!("pad={pad}, len={len}, end={end}, t={t}"),
        )));
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
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "istft: requested length",
          "must be <= reconstruction length t",
          format_smolstr!("len={len}, t={t}"),
        )));
      }
      (0usize, len)
    }
    (false, None) => (0usize, t),
  };
  let start_i32 = i32::try_from(start_usize).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: trim start",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{start_usize}"),
    ))
  })?;
  let stop_i32 = i32::try_from(stop_usize).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "istft: trim stop",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{stop_usize}"),
    ))
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
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "istft: requested output sample window-sum (received no window coverage \
           in the overlap-add and is not recoverable; \
           the requested region (e.g. center=false head/tail) includes a zero-coverage sample — \
           adjust length/center or the window)",
        "must be > COVERAGE_EPS (1e-10) and finite",
        format_smolstr!(
          "global_idx={global_idx}, local_idx={local_idx}, min_wsum={min_wsum:.3e}, \
           n_fft={n_fft}, win_length={win_length}, hop={hop_length}, window_pad={window_pad:?}"
        ),
      )));
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
  /// - [`Error::OutOfRange`] when the [`Spectrum`]'s `window_pad` is
  ///   [`WindowPad::Right`] and `win_length != n_fft` (short-window right-pad
  ///   inversion is not a faithful inverse — rejected up front),
  /// - [`Error::ArithmeticOverflow`] / [`Error::CapExceeded`] when a derived
  ///   size overflows `usize` / `i32`, the OLA length `t` exceeds the
  ///   [`MAX_DECODED_SAMPLES`](crate::audio::io::MAX_DECODED_SAMPLES) cap, or
  ///   the scatter work `num_frames * n_fft` exceeds `MAX_OLA_WORK`,
  /// - [`Error::OutOfRange`] when the `length` trim is out of range,
  /// - [`Error::OutOfRange`] when the **coverage guard** fires — a requested output
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
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: win_length with WindowPad::Right \
           (right-pad short-window inversion is not a faithful inverse — \
           use WindowPad::Center for short-window inversion)",
        "must equal n_fft",
        format_smolstr!("win_length={win_length}, n_fft={n_fft}"),
      )));
    }

    // Every frame is `n_fft` wide (the `win_length` window is placed into the
    // `n_fft` frame), so the overlap-add stride / frame width is `n_fft`.
    let frame_width = n_fft;

    // OLA output / norm-buffer length `t = (num_frames - 1) * hop + n_fft`,
    // capped at MAX_DECODED_SAMPLES (same as the free `istft`).
    let t = (num_frames - 1)
      .checked_mul(hop_length)
      .and_then(|v| v.checked_add(frame_width))
      .ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "ISTFTCache::istft: OLA length ((num_frames - 1) * hop + n_fft)",
          "usize",
          [
            ("num_frames", num_frames as u64),
            ("hop_length", hop_length as u64),
            ("n_fft", n_fft as u64),
          ],
        ))
      })?;
    if t > crate::audio::io::MAX_DECODED_SAMPLES {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "ISTFTCache::istft: OLA length exceeds cap",
        "MAX_DECODED_SAMPLES",
        crate::audio::io::MAX_DECODED_SAMPLES as u64,
        t as u64,
      )));
    }
    let t_i32 = i32::try_from(t).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: OLA length",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{t}"),
      ))
    })?;
    let n_fft_i32 = i32::try_from(n_fft).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: n_fft",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{n_fft}"),
      ))
    })?;

    // Scatter-work cap on `num_frames * frame_width`, checked BEFORE any
    // allocation / cache insertion (the `t` cap bounds the output but small
    // hops drive the scatter far past `t` — same guard the free `istft` uses).
    let idx_len = num_frames.checked_mul(frame_width).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "ISTFTCache::istft: scatter work count num_frames * n_fft",
        "usize",
        [("num_frames", num_frames as u64), ("n_fft", n_fft as u64)],
      ))
    })?;
    if idx_len > MAX_OLA_WORK {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "ISTFTCache::istft: scatter work count (= num_frames * n_fft) exceeds work cap",
        "MAX_OLA_WORK",
        MAX_OLA_WORK as u64,
        idx_len as u64,
      )));
    }
    let idx_len_i32 = i32::try_from(idx_len).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: scatter work count",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{idx_len}"),
      ))
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
      idx_buf.try_reserve_exact(idx_len).map_err(|e| {
        Error::AllocFailure(AllocFailurePayload::new(
          "ISTFTCache::istft: index reservation",
          "i32 elements",
          idx_len as u64,
          e,
        ))
      })?;
      let frame_width_i32 = i32::try_from(frame_width).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "ISTFTCache::istft: frame_width (n_fft)",
          "must fit in i32 (i32::MAX = 2147483647)",
          format_smolstr!("{frame_width}"),
        ))
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
        let end = pad.checked_add(len).ok_or_else(|| {
          Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
            "ISTFTCache::istft: center offset pad + length",
            "usize",
            [("pad", pad as u64), ("len", len as u64)],
          ))
        })?;
        if end > t {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "ISTFTCache::istft: center offset pad + length",
            "must be <= reconstruction length t",
            format_smolstr!("pad={pad}, len={len}, end={end}, t={t}"),
          )));
        }
        (pad, end)
      }
      (true, None) => (pad, t - pad),
      (false, Some(len)) => {
        if len > t {
          return Err(Error::OutOfRange(OutOfRangePayload::new(
            "ISTFTCache::istft: requested length",
            "must be <= reconstruction length t",
            format_smolstr!("len={len}, t={t}"),
          )));
        }
        (0usize, len)
      }
      (false, None) => (0usize, t),
    };
    let start_i32 = i32::try_from(start_usize).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: trim start",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{start_usize}"),
      ))
    })?;
    let stop_i32 = i32::try_from(stop_usize).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ISTFTCache::istft: trim stop",
        "must fit in i32 (i32::MAX = 2147483647)",
        format_smolstr!("{stop_usize}"),
      ))
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
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "ISTFTCache::istft: requested output sample window-sum (received no window coverage \
             in the overlap-add and is not recoverable; \
             the requested region (e.g. center=false head/tail) includes a zero-coverage sample — \
             adjust length/center or the window)",
          "must be > COVERAGE_EPS (1e-10) and finite",
          format_smolstr!(
            "global_idx={global_idx}, local_idx={local_idx}, min_wsum={min_wsum:.3e}, \
             n_fft={n_fft}, win_length={win_length}, hop={hop_length}, window_pad={window_pad:?}"
          ),
        )));
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

/// Slaney mel-scale break frequency (Hz): the scale is linear below this and
/// logarithmic at or above it. Matches `mlx-audio/mlx_audio/dsp.py:516`
/// (`min_log_hz = 1000.0`).
const SLANEY_MIN_LOG_HZ: f32 = 1000.0;
/// Slaney linear-region step: `f_sp = 200 / 3` Hz per mel below
/// [`SLANEY_MIN_LOG_HZ`]. Matches `mlx-audio/mlx_audio/dsp.py:514`
/// (`f_sp = 200.0 / 3`).
const SLANEY_F_SP: f32 = 200.0 / 3.0;
/// Slaney log-region break frequency over which the log warping is measured:
/// `6.4`. Matches the `math.log(6.4)` literal in
/// `mlx-audio/mlx_audio/dsp.py:518`. The reference computes
/// `logstep = math.log(6.4) / 27.0` in double precision; we keep the same
/// `f64` evaluation ([`slaney_logstep`]) and narrow only at the call sites so
/// the step matches the reference to the last f32 bit.
const SLANEY_LOGSTEP_HZ_RATIO: f64 = 6.4;
/// Slaney log-region divisor: `27.0`. Matches `mlx-audio/mlx_audio/dsp.py:518`
/// (`/ 27.0`).
const SLANEY_LOGSTEP_DIV: f64 = 27.0;

/// Slaney log-region step `logstep = ln(6.4) / 27` (f32), evaluated in `f64`
/// (matching the reference's `math.log` double-precision evaluation) and
/// narrowed once. `mel per natural-log Hz` at or above [`SLANEY_MIN_LOG_HZ`].
#[inline]
fn slaney_logstep() -> f32 {
  (SLANEY_LOGSTEP_HZ_RATIO.ln() / SLANEY_LOGSTEP_DIV) as f32
}

/// Which mel-frequency warping the filterbank uses.
///
/// Faithful port of the `mel_scale` argument of `mlx_audio.dsp.mel_filters`
/// (`mlx-audio/mlx_audio/dsp.py:507`). The reference accepts the string
/// `"htk"` (the default) or any other value (in practice `None`, passed by
/// the Whisper front-end) for the Slaney scale.
///
/// - [`MelScale::Htk`] — `mel = 2595 * log10(1 + hz / 700)` (the historic
///   mlxrs default; [`mel_filter_bank`] uses it).
/// - [`MelScale::Slaney`] — the Auditory-Toolbox / librosa "slaney" scale:
///   linear (`f_sp = 200/3` Hz/mel) below `1000 Hz`, log above. Required by
///   the Whisper mel front-end (`mlx-audio/.../whisper/audio.py:76` calls
///   `mel_filters(..., norm="slaney", mel_scale=None)`).
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
pub enum MelScale {
  /// HTK formula `mel = 2595 * log10(1 + hz / 700)` (the mlx-audio default
  /// and the historic mlxrs [`mel_filter_bank`] behavior).
  #[default]
  Htk,
  /// Slaney (Auditory-Toolbox / librosa) scale: linear below `1000 Hz`
  /// (`f_sp = 200/3` Hz per mel), logarithmic above. The Whisper front-end
  /// scale (`mel_scale=None` in the reference).
  Slaney,
}

impl MelScale {
  /// The canonical lowercase string representation (`htk`/`slaney`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Htk => "htk",
      Self::Slaney => "slaney",
    }
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

/// Slaney `hz_to_mel` (f32). Faithful port of the `mel_scale != "htk"` branch
/// of `mlx_audio.dsp.mel_filters.hz_to_mel` (`dsp.py:513-521`): linear
/// `mels = hz / f_sp` below `1000 Hz`, and
/// `min_log_mel + ln(hz / 1000) / logstep` at or above it. `f_min = 0` is the
/// reference's hard-coded inner constant, so `min_log_mel = 1000 / f_sp`.
#[inline]
fn hz_to_mel_slaney(hz: f32) -> f32 {
  let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
  if hz >= SLANEY_MIN_LOG_HZ {
    min_log_mel + (hz / SLANEY_MIN_LOG_HZ).ln() / slaney_logstep()
  } else {
    hz / SLANEY_F_SP
  }
}

/// Slaney `mel_to_hz` (f32). Faithful port of the `mel_scale != "htk"` branch
/// of `mlx_audio.dsp.mel_filters.mel_to_hz` (`dsp.py:527-538`): linear
/// `hz = f_sp * mels` below the break mel, `1000 * exp(logstep * (mels -
/// min_log_mel))` at or above it. (`f_min = 0`.)
#[inline]
fn mel_to_hz_slaney(mel: f32) -> f32 {
  let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
  if mel >= min_log_mel {
    SLANEY_MIN_LOG_HZ * (slaney_logstep() * (mel - min_log_mel)).exp()
  } else {
    SLANEY_F_SP * mel
  }
}

/// Dispatch `hz -> mel` by [`MelScale`] (f32).
#[inline]
fn hz_to_mel_scaled(hz: f32, scale: MelScale) -> f32 {
  match scale {
    MelScale::Htk => hz_to_mel(hz),
    MelScale::Slaney => hz_to_mel_slaney(hz),
  }
}

/// Dispatch `mel -> hz` by [`MelScale`] (f32).
#[inline]
fn mel_to_hz_scaled(mel: f32, scale: MelScale) -> f32 {
  match scale {
    MelScale::Htk => mel_to_hz(mel),
    MelScale::Slaney => mel_to_hz_slaney(mel),
  }
}

/// HTK mel scale in `f64` — the precise ([`MelPrecision::Precise`]) twin of
/// [`hz_to_mel`]. Identical formula, evaluated in double precision so the mel
/// grid matches a torchaudio float64 reference to ~1 ULP rather than drifting
/// ~5e-6 in the f32 path. The constants are the same literals widened to `f64`.
#[inline]
fn hz_to_mel_f64(hz: f64) -> f64 {
  f64::from(MEL_HZ_DIV) * (1.0 + hz / f64::from(MEL_HZ_BREAK)).log10()
}

/// Inverse HTK mel scale in `f64` — the precise ([`MelPrecision::Precise`])
/// twin of [`mel_to_hz`]. Identical formula, double precision (see
/// [`hz_to_mel_f64`]).
#[inline]
fn mel_to_hz_f64(mel: f64) -> f64 {
  f64::from(MEL_HZ_BREAK) * (f64::from(MEL_LOG_BASE).powf(mel / f64::from(MEL_HZ_DIV)) - 1.0)
}

/// Slaney `hz_to_mel` in `f64` — the precise twin of [`hz_to_mel_slaney`].
/// Same piecewise-linear/log formula, double precision.
#[inline]
fn hz_to_mel_slaney_f64(hz: f64) -> f64 {
  let min_log_hz = f64::from(SLANEY_MIN_LOG_HZ);
  let f_sp = f64::from(SLANEY_F_SP);
  let min_log_mel = min_log_hz / f_sp;
  if hz >= min_log_hz {
    min_log_mel + (hz / min_log_hz).ln() / (SLANEY_LOGSTEP_HZ_RATIO.ln() / SLANEY_LOGSTEP_DIV)
  } else {
    hz / f_sp
  }
}

/// Slaney `mel_to_hz` in `f64` — the precise twin of [`mel_to_hz_slaney`].
/// Same piecewise-linear/log formula, double precision.
#[inline]
fn mel_to_hz_slaney_f64(mel: f64) -> f64 {
  let min_log_hz = f64::from(SLANEY_MIN_LOG_HZ);
  let f_sp = f64::from(SLANEY_F_SP);
  let min_log_mel = min_log_hz / f_sp;
  let logstep = SLANEY_LOGSTEP_HZ_RATIO.ln() / SLANEY_LOGSTEP_DIV;
  if mel >= min_log_mel {
    min_log_hz * (logstep * (mel - min_log_mel)).exp()
  } else {
    f_sp * mel
  }
}

/// Dispatch `hz -> mel` by [`MelScale`] (f64).
#[inline]
fn hz_to_mel_scaled_f64(hz: f64, scale: MelScale) -> f64 {
  match scale {
    MelScale::Htk => hz_to_mel_f64(hz),
    MelScale::Slaney => hz_to_mel_slaney_f64(hz),
  }
}

/// Dispatch `mel -> hz` by [`MelScale`] (f64).
#[inline]
fn mel_to_hz_scaled_f64(mel: f64, scale: MelScale) -> f64 {
  match scale {
    MelScale::Htk => mel_to_hz_f64(mel),
    MelScale::Slaney => mel_to_hz_slaney_f64(mel),
  }
}

/// Computation precision for [`mel_filter_bank_with`] / the precise mel
/// filterbank path.
///
/// The default [`MelPrecision::Standard`] path builds the frequency grid, the
/// HTK mel grid, and the triangular filters in `f32` (the historic mlxrs
/// behavior; bit-for-bit unchanged and routed through the
/// [`crate::simd::audio::mel_triangle`] SIMD kernel). [`MelPrecision::Precise`]
/// performs the same arithmetic in `f64` on the CPU and casts the final
/// `(n_mels, n_freqs)` bank back to `f32` before returning the [`Array`].
///
/// # Why a precise mode
///
/// The default f32 filterbank drifts by roughly `5e-6` from a torchaudio
/// float64 reference (accumulated rounding in the `linspace` / mel-scale /
/// triangle divisions). For most pipelines that is negligible, but it is
/// enough to perturb the CTC decode of numerically-sensitive models. Mirrors
/// `mlx-audio`'s `mel_filters(..., precise=True)`: compute in float64, then
/// cast to f32. The cost is a one-time CPU build on a cache miss; the runtime
/// (the cached `mel_bank @ power` matmul) is unaffected.
///
/// `MelPrecision` is a typed flag rather than a `precise: bool` so the choice
/// is self-documenting at the call site and so the per-thread filterbank cache
/// can key on it directly (an f32 bank and an f64 bank for otherwise-identical
/// parameters are distinct entries — see [`mel_filter_bank_cached_with`]).
#[derive(
  Debug, Clone, Copy, PartialEq, Eq, Hash, Default, derive_more::Display, derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
pub enum MelPrecision {
  /// Build the filterbank in `f32` (historic mlxrs default; bit-identical to
  /// pre-#291 behavior, SIMD-accelerated triangle construction).
  #[default]
  Standard,
  /// Build the filterbank in `f64` on the CPU, then cast the final bank to
  /// `f32`. Closer to a torchaudio float64 reference (~1 ULP vs ~5e-6).
  Precise,
}

impl MelPrecision {
  /// The canonical lowercase string representation (`standard`/`precise`).
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Standard => "standard",
      Self::Precise => "precise",
    }
  }
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)` built
/// in `f32` ([`MelPrecision::Standard`]).
///
/// Faithful port of `mlx_audio.dsp.mel_filters(sample_rate, n_fft, n_mels,
/// f_min, f_max, norm=None, mel_scale="htk")` — the HTK formula, no
/// normalization. For the Slaney scale / Slaney area normalization (the
/// Whisper front-end), see [`mel_filter_bank_scaled`].
///
/// `f_max` defaults to `sample_rate / 2` (Nyquist) when `None`. The reference
/// builds frequency points via `mx.linspace(0, sample_rate // 2, n_freqs)`
/// which integer-divides the Nyquist — we mirror that exactly (using
/// `sample_rate as f32 / 2.0` would drift by 0.5 for odd sample rates).
///
/// This is the historic mlxrs default; the result is bit-for-bit identical to
/// pre-#291 behavior. For a float64 computation that tracks a torchaudio
/// reference more closely, see [`mel_filter_bank_with`] with
/// [`MelPrecision::Precise`].
///
/// # Errors
/// - Typed errors: [`Error::InvariantViolation`] if `n_fft == 0` or `n_mels == 0`;
///   [`Error::OutOfRange`] if `f_min < 0`, `f_max <= f_min`, or sizes exceed `i32::MAX`;
///   [`Error::AllocFailure`] if the filter-bank reservation fails.
pub fn mel_filter_bank(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  mel_filter_bank_with(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    MelPrecision::Standard,
  )
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)`,
/// choosing the computation precision via [`MelPrecision`].
///
/// [`MelPrecision::Standard`] is identical to [`mel_filter_bank`] (f32, SIMD
/// triangle construction). [`MelPrecision::Precise`] builds the frequency
/// grid, the HTK mel grid, and the triangular filters in `f64` on the CPU,
/// then casts the final `(n_mels, n_freqs)` bank back to `f32` — the same
/// float64-then-cast strategy used by [`lfilter`] and by
/// `mlx-audio`'s `mel_filters(..., precise=True)`. The default f32 path drifts
/// ~`5e-6` from a torchaudio float64 reference; the precise path closes that
/// to ~1 ULP (enough to stabilize CTC decode in numerically-sensitive
/// models). The precise build is a one-time CPU cost on a cache miss (see
/// [`mel_filter_bank_cached_with`]); the runtime matmul is unaffected.
///
/// The HTK formula, the integer-divided Nyquist (`sample_rate / 2`), the
/// validation, and the work/allocation caps are all identical across
/// precisions — only the per-element dtype of the intermediate frequency /
/// mel / triangle arithmetic differs.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
pub fn mel_filter_bank_with(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  precision: MelPrecision,
) -> Result<Array> {
  // HTK scale, no Slaney normalization — the historic mlxrs default.
  mel_filter_bank_core(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    precision,
    MelScale::Htk,
    false,
  )
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)`,
/// choosing the mel-frequency warping ([`MelScale`]) and whether to apply
/// Slaney area normalization, at [`MelPrecision::Standard`] (f32).
///
/// This is the entry point the **Whisper** front-end needs:
/// `mel_filter_bank_scaled(80, 400, 16_000, 0.0, None, MelScale::Slaney,
/// true)` reproduces `mlx_audio.dsp.mel_filters(16000, 400, 80,
/// norm="slaney", mel_scale=None)` (the call in
/// `mlx-audio/.../whisper/audio.py:76`).
///
/// - `scale` selects the [`MelScale`] (HTK vs Slaney warping).
/// - `slaney_norm` toggles the Slaney area normalization
///   `enorm[m] = 2 / (f_pts[m + 2] - f_pts[m])` applied per filter row
///   (`mlx_audio.dsp.mel_filters`'s `if norm == "slaney"` branch,
///   `dsp.py:567-569`). The HTK / no-norm default ([`mel_filter_bank`])
///   leaves the triangles un-normalized.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
pub fn mel_filter_bank_scaled(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  mel_filter_bank_scaled_with(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    MelPrecision::Standard,
    scale,
    slaney_norm,
  )
}

/// Triangular mel filterbank matrix of shape `(n_mels, n_fft / 2 + 1)`,
/// selecting [`MelPrecision`], [`MelScale`], and Slaney normalization — the
/// fully-general filterbank entry point.
///
/// [`mel_filter_bank`] (HTK, no norm, f32) and [`mel_filter_bank_scaled`]
/// (HTK/Slaney + optional norm, f32) are convenience shorthands over this. The
/// precision / scale / normalization are all orthogonal: the validation, the
/// integer-divided Nyquist frequency grid, the work/allocation caps, and the
/// triangular-filter construction are identical; only the per-element dtype
/// (`f32` vs `f64`), the `hz <-> mel` warping, and the optional per-row Slaney
/// scaling differ.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
#[allow(clippy::too_many_arguments)]
pub fn mel_filter_bank_scaled_with(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  precision: MelPrecision,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  mel_filter_bank_core(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    precision,
    scale,
    slaney_norm,
  )
}

/// Core filterbank construction shared by [`mel_filter_bank_with`] /
/// [`mel_filter_bank_scaled_with`]. Carries the [`MelScale`] warping and the
/// `slaney_norm` flag through the f32 SIMD path and the f64 precise path.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
#[allow(clippy::too_many_arguments)]
fn mel_filter_bank_core(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  precision: MelPrecision,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  if n_fft == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "mel_filter_bank: n_fft",
      "must be > 0",
    )));
  }
  if n_mels == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "mel_filter_bank: n_mels",
      "must be > 0",
    )));
  }
  if sample_rate == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "mel_filter_bank: sample_rate",
      "must be > 0",
    )));
  }
  let f_max = f_max.unwrap_or((sample_rate / 2) as f32);
  if !(f_min >= 0.0 && f_max > f_min) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "mel_filter_bank: f_min / f_max",
      "must satisfy f_min >= 0.0 and f_max > f_min",
      format_smolstr!("f_min={f_min}, f_max={f_max}"),
    )));
  }

  // `n_freqs = n_fft / 2 + 1`; `n_fft / 2 <= usize::MAX / 2`, so `+ 1`
  // cannot overflow `usize`. Bound on i32 happens after the multiplication
  // check below.
  let n_freqs = n_fft / 2 + 1;
  // `n_pts = n_mels + 2`; check for overflow on `n_mels = usize::MAX` /
  // `usize::MAX - 1` before we walk `0..n_pts`.
  let n_pts = n_mels.checked_add(2).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "mel_filter_bank: n_mels + 2",
      "usize",
      [("n_mels", n_mels as u64)],
    ))
  })?;
  // Bank size: `n_mels * n_freqs`. The reference uses an mlx broadcast
  // graph; we materialize one `Vec<f32>` of the same logical size, so we
  // must reject any combination that would attempt a multi-GB allocation
  // (the python form would silently swap or OOM-kill).
  let bank_len = n_mels.checked_mul(n_freqs).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "mel_filter_bank: n_mels * n_freqs",
      "usize",
      [("n_mels", n_mels as u64), ("n_freqs", n_freqs as u64)],
    ))
  })?;
  // i32 bounds on the final mlx shape go here, BEFORE any large allocation.
  let n_mels_i32 = i32::try_from(n_mels).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mel_filter_bank: n_mels",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n_mels}"),
    ))
  })?;
  let n_freqs_i32 = i32::try_from(n_freqs).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mel_filter_bank: n_freqs",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n_freqs}"),
    ))
  })?;

  // Precise (f64) path: build the frequency grid, the mel grid, and the
  // triangular filters entirely in `f64`, then cast the final
  // `(n_mels, n_freqs)` bank to `f32` before handing it to mlx — the same
  // float64-then-cast strategy as `lfilter`. Validation, the sizing caps,
  // and the i32 shape bounds above are shared with the f32 path; only the
  // intermediate dtype differs. The f32 path below stays bit-for-bit
  // unchanged (and SIMD-accelerated) so existing callers are unaffected.
  if precision.is_precise() {
    return mel_filter_bank_f64(
      n_mels,
      n_freqs,
      n_pts,
      bank_len,
      n_mels_i32,
      n_freqs_i32,
      sample_rate,
      f_min,
      f_max,
      scale,
      slaney_norm,
    );
  }

  // `all_freqs[i] = i * (sample_rate / 2) / (n_freqs - 1)` for the python
  // `mx.linspace(0, sample_rate // 2, n_freqs)` form. Build CPU-side;
  // n_freqs is small for any reasonable n_fft (e.g. 201 for n_fft=400).
  // Use `try_reserve_exact` for the same reason as `bank` below — a
  // crafted n_fft can drive n_freqs into multi-GB territory.
  let nyq = (sample_rate / 2) as f32;
  let denom = (n_freqs as f32 - 1.0).max(1.0);
  let mut all_freqs: Vec<f32> = Vec::new();
  all_freqs.try_reserve_exact(n_freqs).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: all_freqs reservation",
      "f32 elements",
      n_freqs as u64,
      e,
    ))
  })?;
  for i in 0..n_freqs {
    all_freqs.push(i as f32 * nyq / denom);
  }

  // Mel grid: `n_mels + 2` points (the +2 give the outer triangle edges).
  let m_min = hz_to_mel_scaled(f_min, scale);
  let m_max = hz_to_mel_scaled(f_max, scale);
  let m_denom = (n_pts as f32 - 1.0).max(1.0);
  let mut f_pts: Vec<f32> = Vec::new();
  f_pts.try_reserve_exact(n_pts).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: f_pts reservation",
      "f32 elements",
      n_pts as u64,
      e,
    ))
  })?;
  for i in 0..n_pts {
    let m = m_min + (m_max - m_min) * (i as f32) / m_denom;
    f_pts.push(mel_to_hz_scaled(m, scale));
  }

  // Build the filterbank directly on the CPU as `(n_mels, n_freqs)` to
  // avoid the reference's allocation chain (linspace + 4 broadcast ops);
  // this is the only place we elide an mlx-graph step — the
  // mel filter is a one-shot constant matrix per `(sample_rate, n_fft,
  // n_mels)` triple, and the on-device construction has no perf benefit.
  //
  // Use `try_reserve_exact` so a multi-GB request from a forged input
  // returns a recoverable `Error::AllocFailure` rather than aborting on the
  // allocator's OOM panic (Rust's default behavior is to abort, not
  // unwind, on allocation failure — `Vec::with_capacity` and `vec![]`
  // share that abort path).
  let mut bank: Vec<f32> = Vec::new();
  bank.try_reserve_exact(bank_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: bank reservation",
      "f32 elements",
      bank_len as u64,
      e,
    ))
  })?;

  // SIMD: dispatch the row-by-row triangle construction through
  // the SIMD kernel (`simd::audio::mel_triangle::mel_filter_bank_rows`).
  // The dispatcher writes 0.0 for collapsed-bin rows (lc <= 0 / cr <=
  // 0) so there is no need for `Vec::resize(bank_len, 0.0)` upfront —
  // the kernel initializes every cell via `MaybeUninit::write`.
  let spare = bank.spare_capacity_mut();
  crate::simd::audio::mel_triangle::mel_filter_bank_rows(
    &mut spare[..bank_len],
    &all_freqs,
    &f_pts,
    n_mels,
  );
  // SAFETY: the SIMD dispatcher's init contract guarantees every cell
  // of the `bank_len`-prefix of `spare` is initialized before
  // returning; `bank_len <= bank.capacity()` per `try_reserve_exact`.
  unsafe { bank.set_len(bank_len) };

  // Slaney area normalization (`mlx_audio.dsp.mel_filters`'s
  // `if norm == "slaney"` branch, `dsp.py:567-569`): scale filter row `m` by
  // `enorm[m] = 2 / (f_pts[m + 2] - f_pts[m])`. The bank is row-major
  // `(n_mels, n_freqs)`, so each row is a contiguous `n_freqs` slice.
  if slaney_norm {
    apply_slaney_norm_f32(&mut bank, &f_pts, n_mels, n_freqs);
  }

  Array::from_slice::<f32>(&bank, &[n_mels_i32, n_freqs_i32])
}

/// Apply the Slaney area normalization in-place to a row-major
/// `(n_mels, n_freqs)` f32 filterbank: row `m` is scaled by
/// `enorm[m] = 2 / (f_pts[m + 2] - f_pts[m])`. Faithful port of
/// `mlx_audio.dsp.mel_filters`'s `enorm = 2.0 / (f_pts[2 : n_mels + 2] -
/// f_pts[:n_mels]); filterbank *= enorm` (`dsp.py:567-569`).
///
/// `f_pts` is the `n_mels + 2`-point Hz grid; `bank.len() == n_mels *
/// n_freqs`. Both invariants hold at the single call site (the f32 builder).
#[inline]
fn apply_slaney_norm_f32(bank: &mut [f32], f_pts: &[f32], n_mels: usize, n_freqs: usize) {
  for m in 0..n_mels {
    let enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
    let row = &mut bank[m * n_freqs..(m + 1) * n_freqs];
    for v in row {
      *v *= enorm;
    }
  }
}

/// Precise (`f64`) filterbank construction for [`mel_filter_bank_with`] with
/// [`MelPrecision::Precise`]. Mirrors the f32 path's frequency / mel / triangle
/// math line-for-line in double precision, then casts the final
/// `(n_mels, n_freqs)` bank to `f32` (the `lfilter` float64-then-cast idiom).
///
/// All arguments are the already-validated, already-bounded values computed by
/// the public wrapper: `n_freqs == n_fft / 2 + 1`, `n_pts == n_mels + 2`,
/// `bank_len == n_mels * n_freqs` (checked, no overflow), and `n_mels_i32` /
/// `n_freqs_i32` the i32 shape; `f_max` is the resolved (`None` → Nyquist)
/// value. Re-validating here would duplicate the wrapper's guards, so this
/// helper is private and trusts its caller's invariants.
///
/// # Errors
/// - [`Error::AllocFailure`] if any of the three `f64` reservations or the
///   final `f32` cast reservation fails.
#[allow(clippy::too_many_arguments)]
fn mel_filter_bank_f64(
  n_mels: usize,
  n_freqs: usize,
  n_pts: usize,
  bank_len: usize,
  n_mels_i32: i32,
  n_freqs_i32: i32,
  sample_rate: u32,
  f_min: f32,
  f_max: f32,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  // `all_freqs[i] = i * (sample_rate / 2) / (n_freqs - 1)` in f64 — the
  // f32 path's `mx.linspace(0, sample_rate // 2, n_freqs)` form widened.
  // The integer-divided Nyquist (`sample_rate / 2`, NOT `/ 2.0`) is kept
  // identical to the f32 path so only the floating arithmetic precision
  // differs, never the grid definition.
  let nyq = f64::from(sample_rate / 2);
  let denom = (n_freqs as f64 - 1.0).max(1.0);
  let mut all_freqs: Vec<f64> = Vec::new();
  all_freqs.try_reserve_exact(n_freqs).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: all_freqs reservation (precise)",
      "f64 elements",
      n_freqs as u64,
      e,
    ))
  })?;
  for i in 0..n_freqs {
    all_freqs.push(i as f64 * nyq / denom);
  }

  // Mel grid: `n_mels + 2` points, evaluated in f64 (mirrors the f32 path).
  let m_min = hz_to_mel_scaled_f64(f64::from(f_min), scale);
  let m_max = hz_to_mel_scaled_f64(f64::from(f_max), scale);
  let m_denom = (n_pts as f64 - 1.0).max(1.0);
  let mut f_pts: Vec<f64> = Vec::new();
  f_pts.try_reserve_exact(n_pts).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: f_pts reservation (precise)",
      "f64 elements",
      n_pts as u64,
      e,
    ))
  })?;
  for i in 0..n_pts {
    let m = m_min + (m_max - m_min) * (i as f64) / m_denom;
    f_pts.push(mel_to_hz_scaled_f64(m, scale));
  }

  // Triangular filters in f64. Mirrors the SCALAR reference
  // (`simd::audio::mel_triangle::mel_filter_bank_rows_scalar`) exactly
  // modulo dtype: per row, `up = (freq - left) / lc`,
  // `down = (right - freq) / cr`, `v = up.min(down).max(0.0)`, and a
  // zero-width triangle (`lc <= 0.0` or `cr <= 0.0`) writes 0.0 across the
  // row. The direct `(freq - left) / lc` form (not the f32 SIMD kernel's
  // algebraically-equal `freq * inv_lc - left_over_lc` rearrangement) is
  // the canonical computation a torchaudio float64 reference performs.
  let mut bank: Vec<f64> = Vec::new();
  bank.try_reserve_exact(bank_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: bank reservation (precise)",
      "f64 elements",
      bank_len as u64,
      e,
    ))
  })?;
  for m in 0..n_mels {
    let left = f_pts[m];
    let center = f_pts[m + 1];
    let right = f_pts[m + 2];
    let lc = center - left;
    let cr = right - center;
    if lc <= 0.0 || cr <= 0.0 {
      // Zero-width triangle — the whole row is 0.0.
      bank.resize(bank.len() + n_freqs, 0.0);
      continue;
    }
    for &freq in &all_freqs {
      let up = (freq - left) / lc;
      let down = (right - freq) / cr;
      bank.push(up.min(down).max(0.0));
    }
  }

  // Slaney area normalization in f64 (mirrors the f32 path, `dsp.py:567-569`):
  // scale row `m` by `enorm[m] = 2 / (f_pts[m + 2] - f_pts[m])` before the
  // f32 cast so the normalization is carried in double precision too.
  if slaney_norm {
    for m in 0..n_mels {
      let enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
      let row = &mut bank[m * n_freqs..(m + 1) * n_freqs];
      for v in row {
        *v *= enorm;
      }
    }
  }

  // Cast the f64 bank down to f32 for the mlxrs audio dtype (mirrors
  // `lfilter`'s `for v in &y_f64 { y.push(*v as f32) }` boundary).
  let mut bank_f32: Vec<f32> = Vec::new();
  bank_f32.try_reserve_exact(bank_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "mel_filter_bank: bank f32 cast reservation (precise)",
      "f32 elements",
      bank_len as u64,
      e,
    ))
  })?;
  for &v in &bank {
    bank_f32.push(v as f32);
  }

  Array::from_slice::<f32>(&bank_f32, &[n_mels_i32, n_freqs_i32])
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
///
/// `precision` ([`MelPrecision`]) is part of the key so an f32
/// ([`MelPrecision::Standard`]) bank and an f64 ([`MelPrecision::Precise`])
/// bank built from otherwise-identical parameters never collide — the two are
/// numerically different matrices (the whole point of the precise mode), so
/// returning one for a request for the other would be a silent correctness
/// bug. `MelPrecision` is `Eq + Hash` (no float payload), so it participates
/// in the derived `PartialEq`/`Eq` directly.
///
/// `scale` ([`MelScale`]) and `slaney_norm` are likewise part of the key: a
/// Slaney-warped / Slaney-normalized bank is a numerically different matrix
/// from the HTK / un-normalized bank for otherwise-identical parameters, so
/// they must never collide. Both are `Eq + Hash` (no float payload) and
/// participate in the derived `PartialEq`/`Eq` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MelFilterCacheKey {
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min_bits: u32,
  f_max_bits: Option<u32>,
  precision: MelPrecision,
  scale: MelScale,
  slaney_norm: bool,
}

impl MelFilterCacheKey {
  #[allow(clippy::too_many_arguments)]
  fn new(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: Option<f32>,
    precision: MelPrecision,
    scale: MelScale,
    slaney_norm: bool,
  ) -> Self {
    Self {
      n_mels,
      n_fft,
      sample_rate,
      f_min_bits: f_min.to_bits(),
      f_max_bits: f_max.map(f32::to_bits),
      precision,
      scale,
      slaney_norm,
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
/// Caches the [`MelPrecision::Standard`] (f32) bank. For the precise (f64)
/// bank — cached under a distinct key so the two never collide — use
/// [`mel_filter_bank_cached_with`].
///
/// # Errors
/// - Same as [`mel_filter_bank`].
///
/// # See also
/// - [`mel_filter_bank`] — the uncached construction path.
/// - [`mel_filter_bank_cached_with`] — precision-selecting cached variant.
/// - [`clear_mel_filter_cache`] — empties the per-thread cache (test /
///   memory-pressure use).
pub fn mel_filter_bank_cached(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
) -> Result<Array> {
  mel_filter_bank_cached_with(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    MelPrecision::Standard,
  )
}

/// Cached variant of [`mel_filter_bank_with`] — the precision-selecting twin of
/// [`mel_filter_bank_cached`].
///
/// Identical caching semantics (per-thread LRU bounded at
/// [`MEL_FILTER_CACHE_CAP`], `try_clone` on hit, no caching of a failed build)
/// but the cache key includes the [`MelPrecision`], so an f32
/// ([`MelPrecision::Standard`]) bank and an f64 ([`MelPrecision::Precise`])
/// bank for otherwise-identical `(n_mels, n_fft, sample_rate, f_min, f_max)`
/// parameters occupy **distinct** cache slots and never alias. A miss builds
/// via [`mel_filter_bank_with`] (so the precise path's one-time f64 build cost
/// is paid once per parameter set, then memoized like the f32 path).
///
/// # Errors
/// - Same as [`mel_filter_bank`].
///
/// # See also
/// - [`mel_filter_bank_with`] — the uncached, precision-selecting path.
/// - [`mel_filter_bank_cached`] — the [`MelPrecision::Standard`] shorthand.
pub fn mel_filter_bank_cached_with(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  precision: MelPrecision,
) -> Result<Array> {
  // HTK scale, no Slaney normalization — matches `mel_filter_bank_with`.
  mel_filter_bank_cached_core(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    precision,
    MelScale::Htk,
    false,
  )
}

/// Cached variant of [`mel_filter_bank_scaled`] — the [`MelScale`] /
/// Slaney-normalization-aware cached filterbank (f32).
///
/// This is the **Whisper front-end's** cached bank entry point:
/// `mel_filter_bank_scaled_cached(80, 400, 16_000, 0.0, None,
/// MelScale::Slaney, true)` memoizes the Slaney bank Whisper uses on every
/// 30-second chunk. Identical per-thread LRU semantics to
/// [`mel_filter_bank_cached`]; the cache key additionally carries the
/// [`MelScale`] and the `slaney_norm` flag, so a Slaney bank never aliases the
/// HTK bank for the same `(n_mels, n_fft, sample_rate, f_min, f_max)`.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
pub fn mel_filter_bank_scaled_cached(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  mel_filter_bank_cached_core(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    MelPrecision::Standard,
    scale,
    slaney_norm,
  )
}

/// Core cached-filterbank lookup/build/store shared by every public cached
/// variant. The full `(n_mels, n_fft, sample_rate, f_min, f_max, precision,
/// scale, slaney_norm)` tuple is the cache key, so no two numerically-distinct
/// banks ever collide.
///
/// # Errors
/// - Same as [`mel_filter_bank`].
#[allow(clippy::too_many_arguments)]
fn mel_filter_bank_cached_core(
  n_mels: usize,
  n_fft: usize,
  sample_rate: u32,
  f_min: f32,
  f_max: Option<f32>,
  precision: MelPrecision,
  scale: MelScale,
  slaney_norm: bool,
) -> Result<Array> {
  let key = MelFilterCacheKey::new(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    precision,
    scale,
    slaney_norm,
  );

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
  // recoverable typed error (invalid params, work cap, etc.); we
  // propagate that error WITHOUT touching the cache, so a failed call
  // cannot poison the cache with an absent or invalid entry.
  let bank = mel_filter_bank_core(
    n_mels,
    n_fft,
    sample_rate,
    f_min,
    f_max,
    precision,
    scale,
    slaney_norm,
  )?;

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
  // per-thread LRU cache (the uncached path rebuilt the bank on every
  // chunk / per-utterance encode pass).
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
/// [`Error::CapExceeded`] instead of risking a multi-GB CPU allocation for the
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
/// [`Error::CapExceeded`].
const MAX_LOUDNESS_SAMPLES: usize = crate::audio::io::MAX_DECODED_SAMPLES;

/// Hard byte ceiling on the per-channel `f64` mean-square matrix
/// [`integrated_loudness`] allocates: `num_blocks * n_channels *
/// size_of::<f64>() <= MAX_LOUDNESS_BLOCK_BYTES`. Capping
/// `num_blocks * n_channels` against the 64 Mi-**element** sample
/// budget alone would be wrong — the cells are `f64`, so the actual peak
/// bytes would be 8× the element budget (`64 Mi * 8 B = 512 MiB`). A
/// caller-controllable
/// `overlap` close to 1 (e.g. `0.99999990`) on a small mono input derived
/// tens of millions of blocks that all passed the element-only cap, then
/// reserved hundreds of MB / ~2 GiB for the mean-square matrix alone. The
/// byte cap bounds the dominant block-sized buffer directly: at 64 MiB
/// the peak `mean_square` footprint is bounded regardless of how the
/// caller distributes the (`num_blocks`, `n_channels`) product, and the
/// gate-index iteration that re-reads the same matrix never sees more
/// cells than this byte budget allows. Pathological overlaps return a
/// recoverable [`Error::CapExceeded`] BEFORE any allocation.
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
/// footprint. Omitting the `n_channels` factor would
/// admit a 5-channel pathological case
/// (`num_blocks=1_677_721, block_samples=160, n_channels=5` — channel-less
/// product `~268 Mi <= 256 Mi cap` BUT actual visits `~1.34 Bi`), so
/// `n_channels` is folded into the work product. `ceil` is used because
/// the per-block bounds `(floor(bi*step*bs*r), floor((bi*step+1)*bs*r))`
/// can give `upper - lower = ceil(block_size_samples)` in the worst
/// case for fractional `block_size_samples`. Cap the total visit count
/// at 256 Mi (`block_size = 0.4 s` at 48 kHz = `19,200 samples/block`,
/// so a default-overlap 256 Mi-visit budget admits ~13,653 blocks for
/// mono ≈ 91 hours of audio, or ~2,730 5-channel blocks ≈ 18 hours of
/// 5-channel audio — comfortable for any realistic loudness analysis,
/// but rejects multi-trillion-visit pathological cases in microseconds).
/// Pathological work returns a recoverable [`Error::CapExceeded`] BEFORE
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
/// - [`Error::EmptyInput`] if `a` is empty,
///   [`Error::InvariantViolation`] if `a[0] == 0`,
///   [`Error::RankMismatch`] if `data` is not 1-D,
///   [`Error::CapExceeded`] if `data`'s sample count exceeds the cap,
///   [`Error::OutOfRange`] if sizes exceed `i32::MAX`.
///
/// The reference returns `np.zeros_like(data)` when `b` is empty; we mirror
/// that.
pub fn lfilter(b: &[f64], a: &[f64], data: &Array) -> Result<Array> {
  if data.ndim() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfilter: only supports 1-D input",
      data.ndim() as u32,
      data.shape(),
    )));
  }
  let shape = data.shape();
  let n_samples = shape[0];
  let n_samples_i32 = i32::try_from(n_samples).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "lfilter: n_samples",
      "must fit in i32 (i32::MAX = 2147483647)",
      format_smolstr!("{n_samples}"),
    ))
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
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "lfilter: sample count exceeds cap",
      "MAX_LFILTER_SAMPLES",
      MAX_LFILTER_SAMPLES as u64,
      n_samples as u64,
    )));
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
  x_f64.try_reserve_exact(n_samples).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: input promotion reservation",
      "f64 samples",
      n_samples as u64,
      e,
    ))
  })?;
  for v in &x_f32 {
    x_f64.push(f64::from(*v));
  }
  let y_f64 = lfilter_f64(b, a, &x_f64)?;
  let mut y: Vec<f32> = Vec::new();
  y.try_reserve_exact(n_samples).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: output reservation",
      "f32 samples",
      n_samples as u64,
      e,
    ))
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
/// - [`Error::EmptyInput`] if `a` is empty,
///   [`Error::InvariantViolation`] if `a[0] == 0`,
///   [`Error::CapExceeded`] if `x.len()` exceeds the cap
///   (mirrors the public [`lfilter`]'s element budget).
///
/// The reference returns `np.zeros_like(data)` when `b` is empty; we mirror
/// that.
fn lfilter_f64(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
  if a.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "lfilter: filter denominator (a)",
    )));
  }
  if a[0] == 0.0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfilter: filter denominator a[0]",
      "must be non-zero (a[0] != 0)",
    )));
  }
  let n_samples = x.len();
  if n_samples > MAX_LFILTER_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "lfilter: sample count exceeds cap",
      "MAX_LFILTER_SAMPLES",
      MAX_LFILTER_SAMPLES as u64,
      n_samples as u64,
    )));
  }

  // Mirror the reference's `if b.size == 0: return np.zeros_like(data)`.
  // A zero-tap numerator is degenerate (filter output is identically zero
  // regardless of `data` or `a`); allocate the result directly.
  if b.is_empty() {
    let mut y: Vec<f64> = Vec::new();
    y.try_reserve_exact(n_samples).map_err(|e| {
      Error::AllocFailure(AllocFailurePayload::new(
        "lfilter: zero-output reservation",
        "f64 samples",
        n_samples as u64,
        e,
      ))
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
  b_norm.try_reserve_exact(b.len()).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: numerator (b) normalize reservation",
      "f64 taps",
      b.len() as u64,
      e,
    ))
  })?;
  for &bv in b {
    b_norm.push(bv / a0);
  }
  let mut a_norm: Vec<f64> = Vec::new();
  a_norm.try_reserve_exact(a.len()).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: denominator (a) normalize reservation",
      "f64 taps",
      a.len() as u64,
      e,
    ))
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
  y.try_reserve_exact(n_samples).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: output reservation",
      "f64 samples",
      n_samples as u64,
      e,
    ))
  })?;

  // Reference's `state_len == 0` fast path: `y = b[0] * x` (no recurrence,
  // no feedback state to maintain). This is the FIR-of-length-1 case.
  // (#154): a NEON `vmulq_n_f64` 2-lane kernel exists at
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

  // (#154): biquad fast path — `state_len == 2`, `b.len() ==
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
  // This specialized arm is not separately wired: benchmarking did not
  // show it beating the generic loop on the actually-wired paths, so
  // the single-pass generic loop below handles `state_len == 2` along
  // with all other shapes.

  // General direct-form II transposed recurrence (matching the reference's
  // per-sample loop body byte-for-byte) — for non-biquad / non-FIR shapes
  // AND for `state_len == 2` (the biquad-dispatcher fast-path was tried
  // and reverted; see comment above).
  let mut state: Vec<f64> = Vec::new();
  state.try_reserve_exact(state_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: state reservation",
      "f64 taps",
      state_len as u64,
      e,
    ))
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
/// - [`Error::EmptyInput`] if `a` is empty,
///   [`Error::InvariantViolation`] if `a[0] == 0`,
///   [`Error::CapExceeded`] if `x.len()` exceeds the cap
///   (mirrors [`lfilter_f64`]'s element budget).
///
/// `b` empty mirrors [`lfilter_f64`]'s "zero-output" semantics by writing
/// zeros into `x` (the in-place equivalent of returning `np.zeros_like`).
fn lfilter_f64_in_place(b: &[f64], a: &[f64], x: &mut [f64]) -> Result<()> {
  if a.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "lfilter: filter denominator (a)",
    )));
  }
  if a[0] == 0.0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfilter: filter denominator a[0]",
      "must be non-zero (a[0] != 0)",
    )));
  }
  let n_samples = x.len();
  if n_samples > MAX_LFILTER_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "lfilter: sample count exceeds cap (in-place)",
      "MAX_LFILTER_SAMPLES",
      MAX_LFILTER_SAMPLES as u64,
      n_samples as u64,
    )));
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
  b_norm.try_reserve_exact(b.len()).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter (in-place): numerator (b) normalize reservation",
      "f64 taps",
      b.len() as u64,
      e,
    ))
  })?;
  for &bv in b {
    b_norm.push(bv / a0);
  }
  let mut a_norm: Vec<f64> = Vec::new();
  a_norm.try_reserve_exact(a.len()).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter (in-place): denominator (a) normalize reservation",
      "f64 taps",
      a.len() as u64,
      e,
    ))
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
  // dependency on neighboring slots). (#154): the SIMD FIR
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

  // (#154): biquad fast path — `state_len == 2`, `b.len() ==
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
  // through [`integrated_loudness`] remain byte-identical to pre-SIMD
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
  state.try_reserve_exact(state_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "lfilter: state reservation",
      "f64 taps",
      state_len as u64,
      e,
    ))
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
  chan_f64.try_reserve_exact(n).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "k_weight_channel: input promotion reservation",
      "f64 samples",
      n as u64,
      e,
    ))
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
/// - Typed errors: [`Error::RankMismatch`] if `data` is not 1-D or 2-D;
///   [`Error::OutOfRange`] if channels > 5, `rate == 0`, `block_size <= 0`,
///   `overlap` not in `[0, 1)`, or input shorter than one block;
///   [`Error::CapExceeded`] if the total element count, block-byte budget,
///   or sample-visit work exceeds the respective cap (these reject pathological
///   overlaps BEFORE any `num_blocks`-scaled allocation or the per-block loop
///   runs (the byte cap dominates for normal block
///   sizes; the visit cap catches near-1 overlaps with small block
///   sizes that fit under the byte cap but multiply the per-block CPU
///   sum work — once per channel, since the per-block loop runs once
///   per channel — to multi-trillion visits),
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
/// channel weighted f64 buffers all-at-once — doing so would hold
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
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: rate",
      "must be > 0",
      "0",
    )));
  }
  if !(block_size > 0.0 && block_size.is_finite()) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: block_size",
      "must be a finite value > 0",
      format!("{block_size}"),
    )));
  }
  if !((0.0..1.0).contains(&overlap) && overlap.is_finite()) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: overlap",
      "must be a finite value in [0, 1)",
      format!("{overlap}"),
    )));
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
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "integrated_loudness: n_channels",
          "must be at most 5 (BS.1770 standard limit)",
          format!("{n_channels}"),
        )));
      }
      if n_channels == 0 {
        return Err(Error::EmptyInput(EmptyInputPayload::new(
          "integrated_loudness: audio channels (must have at least 1 channel)",
        )));
      }
      (n_samples, n_channels)
    }
    other => {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "integrated_loudness: data must be 1-D (mono) or 2-D (n_samples, n_channels)",
        other as u32,
        data.shape(),
      )));
    }
  };
  // Cap on the TOTAL materialized element count BEFORE the `to_vec`.
  // Capping only `n_samples` would let a 2-D input like
  // `(MAX_DECODED_SAMPLES, 5)` bypass it — the `to_vec` would then
  // materialize `5 * MAX_DECODED_SAMPLES` f32 samples (multi-GB). Mirrors
  // the [`stft`] / OLA pattern of checking the materialized work cap
  // before any allocation.
  let total_elements = n_samples.checked_mul(n_channels).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "integrated_loudness: total element count n_samples * n_channels",
      "usize",
      [
        ("n_samples", n_samples as u64),
        ("n_channels", n_channels as u64),
      ],
    ))
  })?;
  if total_elements > MAX_LOUDNESS_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "integrated_loudness: total element count (= n_samples * n_channels) exceeds cap",
      "MAX_LOUDNESS_SAMPLES",
      MAX_LOUDNESS_SAMPLES as u64,
      total_elements as u64,
    )));
  }

  // `block_size * rate` is the per-block sample count in the reference.
  // Cast through `f64` to mirror the python arithmetic exactly (block_size
  // is a float seconds, rate is an int Hz).
  let rate_f64 = f64::from(rate);
  let block_samples_f64 = block_size * rate_f64;
  if !block_samples_f64.is_finite() || block_samples_f64 < 1.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: block_size * rate",
      "must be finite and >= 1 sample",
      format_smolstr!("block_samples={block_samples_f64}, block_size={block_size}, rate={rate}"),
    )));
  }
  if (n_samples as f64) < block_samples_f64 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: audio length (samples)",
      "must be greater than the block size",
      format!("{n_samples} samples (block_size*rate = {block_samples_f64:.1} samples)"),
    )));
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
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: derived num_blocks",
      "must be finite and >= 1",
      format_smolstr!(
        "num_blocks={num_blocks_f64}, duration={duration_seconds}, \
         block_size={block_size}, overlap={overlap}"
      ),
    )));
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
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: derived num_blocks",
      "must fit in usize",
      format_smolstr!(
        "num_blocks={num_blocks_f64}, duration={duration_seconds}, \
         block_size={block_size}, overlap={overlap}"
      ),
    )));
  }
  let num_blocks = num_blocks_f64 as usize;
  if num_blocks == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "integrated_loudness: derived num_blocks",
      "must be >= 1",
      "0",
    )));
  }
  // Byte cap: `num_blocks * n_channels * sizeof::<f64>() <=
  // MAX_LOUDNESS_BLOCK_BYTES`. Compute the cell count first, then the
  // byte product — both `checked_mul` so overflow rejects.
  let block_cells = num_blocks.checked_mul(n_channels).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "integrated_loudness: block cells num_blocks * n_channels",
      "usize",
      [
        ("num_blocks", num_blocks as u64),
        ("n_channels", n_channels as u64),
      ],
    ))
  })?;
  let block_bytes = block_cells
    .checked_mul(std::mem::size_of::<f64>())
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "integrated_loudness: block bytes block_cells * 8",
        "usize",
        [("block_cells", block_cells as u64)],
      ))
    })?;
  if block_bytes > MAX_LOUDNESS_BLOCK_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "integrated_loudness: mean-square byte footprint exceeds cap",
      "MAX_LOUDNESS_BLOCK_BYTES",
      MAX_LOUDNESS_BLOCK_BYTES as u64,
      block_bytes as u64,
    )));
  }
  // Work cap: `num_blocks * block_size_samples * n_channels <=
  // MAX_LOUDNESS_WORK`. The per-block mean-square loop runs ONCE PER
  // CHANNEL (see the per-channel streaming loop below), so the actual
  // sample-visit count is `num_blocks * block_size_samples * n_channels`
  // — bounding the channel-less product alone admitted a 5-channel input
  // (`num_blocks=1_677_721, block_samples=160, n_channels=5`
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
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "integrated_loudness: partial work num_blocks * block_samples",
        "usize",
        [
          ("num_blocks", num_blocks as u64),
          ("block_samples", block_samples_usize as u64),
        ],
      ))
    })?
    .checked_mul(n_channels)
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "integrated_loudness: total work num_blocks * block_samples * n_channels",
        "usize",
        [
          ("num_blocks", num_blocks as u64),
          ("block_samples", block_samples_usize as u64),
          ("n_channels", n_channels as u64),
        ],
      ))
    })?;
  if total_work > MAX_LOUDNESS_WORK {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "integrated_loudness: total sample-visit work \
         (= num_blocks * block_samples * n_channels) exceeds cap",
      "MAX_LOUDNESS_WORK",
      MAX_LOUDNESS_WORK as u64,
      total_work as u64,
    )));
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
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "integrated_loudness: internal shape mismatch — raw_f32 sample count",
      total_elements,
      raw_f32.len(),
    )));
  }

  // mean_square[c][b] — per-channel, per-block mean-square. Matches the
  // reference's `np.zeros((num_channels, num_blocks))`. Bounded by the
  // `block_work` cap above.
  let mut mean_square: Vec<Vec<f64>> = Vec::new();
  mean_square.try_reserve_exact(n_channels).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "integrated_loudness: mean_square channels reservation",
      "Vec<f64> rows",
      n_channels as u64,
      e,
    ))
  })?;
  for _ in 0..n_channels {
    let mut row: Vec<f64> = Vec::new();
    row.try_reserve_exact(num_blocks).map_err(|e| {
      Error::AllocFailure(AllocFailurePayload::new(
        "integrated_loudness: mean_square blocks reservation",
        "f64 blocks",
        num_blocks as u64,
        e,
      ))
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
  chan_f32.try_reserve_exact(n_samples).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "integrated_loudness: chan_f32 reservation",
      "f32 samples",
      n_samples as u64,
      e,
    ))
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
  block_loudness.try_reserve_exact(num_blocks).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "integrated_loudness: block_loudness reservation",
      "f64 blocks",
      num_blocks as u64,
      e,
    ))
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
  // of gated block indices. Materializing TWO
  // `num_blocks`-sized `Vec<usize>` (one per gate pass) via
  // `try_reserve_exact(num_blocks)` would cost `8 * num_blocks` bytes each
  // on a 64-bit target, scaling with `num_blocks` even when very few blocks
  // survive the gate. The byte/work caps above bound `num_blocks`,
  // so this is not a multi-GB risk; we still eliminate the
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
/// - [`Error::OutOfRange`] if `input_loudness` or `target_loudness` is
///   non-finite (NaN/±inf would yield a non-finite gain), or propagated
///   errors from the underlying multiply.
pub fn normalize_loudness(
  data: &Array,
  input_loudness: f64,
  target_loudness: f64,
) -> Result<Array> {
  if !input_loudness.is_finite() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "normalize_loudness: input_loudness",
      "must be finite (NaN/±inf would yield a non-finite gain)",
      format!("{input_loudness}"),
    )));
  }
  if !target_loudness.is_finite() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "normalize_loudness: target_loudness",
      "must be finite (NaN/±inf would yield a non-finite gain)",
      format!("{target_loudness}"),
    )));
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
/// - [`Error::OutOfRange`] if `target_peak_db` is non-finite or `current_peak
///   == 0.0`; [`Error::EmptyInput`] if `data` is empty;
///   [`Error::NonFiniteScalar`] if the current peak or computed gain is
///   non-finite; propagated errors from underlying `abs` / `max` / multiply.
///
/// This reads back one scalar (the current peak) via an explicit `eval`.
pub fn normalize_peak(data: &Array, target_peak_db: f64) -> Result<Array> {
  if !target_peak_db.is_finite() {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "normalize_peak: target_peak_db",
      "must be finite (NaN/±inf cannot represent a dB level)",
      format!("{target_peak_db}"),
    )));
  }
  if data.size() == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "normalize_peak: data must be non-empty (max over an empty array is undefined)",
    )));
  }
  // current_peak = max(|data|). One explicit scalar readback.
  let abs_data = data.abs()?;
  let mut peak_arr = ops::reduction::max(&abs_data, false)?;
  let current_peak = peak_arr.item::<f32>()?;
  if !current_peak.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "normalize_peak: current peak max(|data|) (the input contains a NaN or infinite sample)",
      current_peak as f64,
    )));
  }
  // `current_peak` is `max(|.|)`, so it is always `>= 0.0`; `== 0.0` means an
  // all-silence (or all-zero) input, which cannot be peak-normalized.
  if current_peak == 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "normalize_peak: current peak max(|data|)",
      "must be > 0 — an all-silence input cannot be peak-normalized (the gain would divide by zero)",
      "0",
    )));
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
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "normalize_peak: target amplitude 10^(target_peak_db / 20) \
         (target_peak_db is too large to represent as a finite f32 gain)",
      target_linear as f64,
    )));
  }
  if !gain.is_finite() {
    return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
      "normalize_peak: gain (= target_linear / current_peak) \
         (the current peak is too small for target_peak_db — scaling overflows to non-finite)",
      gain as f64,
    )));
  }
  let gain_arr = Array::full::<f32>(&[0i32; 0], gain)?;
  ops::arithmetic::multiply(data, &gain_arr)
}

#[cfg(test)]
mod tests;
