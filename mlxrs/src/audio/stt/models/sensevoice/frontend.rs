//! SenseVoice-Small front-end: Kaldi fbank + Low-Frame-Rate stacking + CMVN,
//! plus the HF/PyTorch → MLX weight `sanitize`.
//!
//! Faithful port of the module-level helpers in [`sensevoice.py`][sv]:
//! `_compute_fbank` (`:17-44`), `_apply_lfr` (`:47-72`), `_apply_cmvn`
//! (`:75-80`), `_parse_am_mvn` (`:83-103`), the `_extract_features` assembly
//! (`:378-395`), and `sanitize` (`:554-565`). The fbank step reuses
//! [`crate::audio::features::compute_fbank_kaldi`] verbatim — the reference's
//! `_compute_fbank` is exactly a `compute_fbank_kaldi` call with a fixed
//! parameter set (`win_type="hamming"`, `preemphasis=0.97`, `dither=0.0`,
//! `snip_edges=True`, `low_freq=20.0`, `high_freq=0.0`) preceded by a `2^15`
//! waveform pre-scale (`sensevoice.py:30`). The pre-emphasis first-sample
//! boundary is [`crate::audio::features::PreemphBoundary::Preserve`], matching
//! `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:913`, which keeps `x[0]`
//! unchanged).
//!
//! The LFR stacking and the `am.mvn` parse are the only genuinely new pieces;
//! both are model-private (FunASR/Paraformer-family idioms, not general DSP).
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

use std::collections::HashMap;

use regex::Regex;
use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::features::{KaldiWindow, PreemphBoundary, compute_fbank_kaldi},
  error::{
    Error, InvariantViolationPayload, MalformedDataPayload, OutOfRangePayload, RankMismatchPayload,
    Result,
  },
  model_validation::{checked_mul, reserve_or_error},
  ops,
};

use super::config::FrontendConfig;

/// The `2^15` waveform pre-scale the reference applies before the Kaldi fbank
/// (`sensevoice.py:30`: `waveform = waveform * (1 << 15)`). SenseVoice's `am.mvn`
/// statistics were estimated on `int16`-range fbank inputs, so the float
/// waveform is scaled into that range first.
const WAVEFORM_SCALE: f32 = (1u32 << 15) as f32;

/// Kaldi fbank `preemphasis` coefficient (`sensevoice.py:39`).
const PREEMPHASIS: f32 = 0.97;
/// Kaldi fbank `low_freq` mel floor in Hz (`sensevoice.py:42`).
///
/// `pub(super)` so [`FrontendConfig::validate`](super::config::FrontendConfig::validate)
/// can hoist the fbank Nyquist invariant to load against the SAME constant
/// [`compute_fbank`] passes — `get_mel_banks_kaldi` rejects `low_freq >= nyquist`
/// (`nyquist = fs / 2`, `dsp.py:826-831`), so a small `fs` whose Nyquist is
/// `<= LOW_FREQ` must fail at load rather than at the first transcribe.
pub(super) const LOW_FREQ: f32 = 20.0;
/// Kaldi fbank `high_freq` (`0.0` = Nyquist; `sensevoice.py:43`).
const HIGH_FREQ: f32 = 0.0;

/// Map the config `window` string to the reused [`KaldiWindow`]. The reference
/// passes the string straight to `compute_fbank_kaldi`, whose `win_type` only
/// accepts these four; an unrecognized window is rejected with a typed error
/// rather than silently falling back.
///
/// `pub(crate)` so [`FrontendConfig::validate`](super::config::FrontendConfig::validate)
/// can hoist the same window-enum check to load (the fbank-derived-invariant
/// load gate), sharing the *one* accepted set rather than restating it.
///
/// # Errors
/// [`Error::OutOfRange`] for a `window` value outside
/// `hamming` / `hanning` / `povey` / `rectangular`.
pub(crate) fn window_from_str(window: &str) -> Result<KaldiWindow> {
  match window {
    "hamming" => Ok(KaldiWindow::Hamming),
    "hanning" => Ok(KaldiWindow::Hanning),
    "povey" => Ok(KaldiWindow::Povey),
    "rectangular" => Ok(KaldiWindow::Rectangular),
    other => Err(Error::OutOfRange(OutOfRangePayload::new(
      "frontend window",
      "must be one of hamming / hanning / povey / rectangular",
      format_smolstr!("{other}"),
    ))),
  }
}

/// Compute the Kaldi fbank for a mono `waveform`, mirroring `_compute_fbank`
/// (`sensevoice.py:17-44`).
///
/// `waveform` is a 1-D float [`Array`]. The reference computes the analysis
/// window / hop sizes in samples from the config milliseconds
/// (`win_len = fs * frame_length / 1000`, `win_inc = fs * frame_shift / 1000`,
/// `sensevoice.py:27-28`) — at the defaults (`fs=16000`, `frame_length=25`,
/// `frame_shift=10`) this is `win_len=400`, `win_inc=160`. The deterministic
/// `dither=0.0` path passes `dither_key=None`.
///
/// # Errors
/// - [`Error::OutOfRange`] for an unrecognized window or for sample-size
///   computations exceeding `i32::MAX`;
/// - propagates [`compute_fbank_kaldi`]'s validation / op errors.
pub fn compute_fbank(waveform: &Array, fc: &FrontendConfig) -> Result<Array> {
  let win_type = window_from_str(fc.window())?;

  // `win_len = fs * frame_length / 1000`, `win_inc = fs * frame_shift / 1000`
  // (samples), exactly as `sensevoice.py:27-28`. `fs` / the frame sizes are
  // config-validated positive; compute in `u64` and narrow to `usize`.
  let fs = u64::from(fc.fs());
  let frame_length = fc.frame_length().max(0) as u64;
  let frame_shift = fc.frame_shift().max(0) as u64;
  let win_len = (fs * frame_length / 1000) as usize;
  let win_inc = (fs * frame_shift / 1000) as usize;
  let num_mels = fc.n_mels().max(0) as usize;

  // `waveform = waveform * (1 << 15)` (`sensevoice.py:30`).
  let scale = Array::full::<f32>(&[0i32; 0], WAVEFORM_SCALE)?;
  let scaled = waveform.multiply(&scale)?;

  compute_fbank_kaldi(
    &scaled,
    fc.fs(),
    win_len,
    win_inc,
    num_mels,
    win_type,
    PREEMPHASIS,
    0.0,
    true,
    LOW_FREQ,
    HIGH_FREQ,
    None,
    // `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:913`) keeps the first sample
    // of each frame unchanged under pre-emphasis (`first_col =
    // strided_input[:, 0:1]`), so SenseVoice's features match the reference.
    PreemphBoundary::Preserve,
  )
}

/// Stack `lfr_m` consecutive fbank frames into one and stride by `lfr_n`
/// (Low-Frame-Rate), mirroring `_apply_lfr` (`sensevoice.py:47-72`).
///
/// `feats` is the `(T, D)` fbank. The output is `(ceil(T / lfr_n), lfr_m * D)`:
/// each LFR frame concatenates `lfr_m` raw frames (`7 * 80 = 560` at the
/// defaults). The sequence is left-padded with `(lfr_m - 1) // 2` copies of the
/// first frame (`:51-55`); the final window, if it runs past the end, is
/// right-padded with copies of the last frame (`:63-70`).
///
/// # Errors
/// - [`Error::RankMismatch`] if `feats` is not rank-2;
/// - [`Error::OutOfRange`] if `lfr_m` / `lfr_n` is `<= 0`, or a dim exceeds
///   `i32::MAX`;
/// - propagates the slice / tile / concatenate / reshape / stack op errors.
pub fn apply_lfr(feats: &Array, lfr_m: i32, lfr_n: i32) -> Result<Array> {
  let shape = feats.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "apply_lfr: expected rank-2 (T, D) fbank",
      shape.len() as u32,
      shape,
    )));
  }
  if lfr_m <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "apply_lfr: lfr_m",
      "must be > 0",
      format_smolstr!("{lfr_m}"),
    )));
  }
  if lfr_n <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "apply_lfr: lfr_n",
      "must be > 0",
      format_smolstr!("{lfr_n}"),
    )));
  }

  let t = shape[0];
  let d = i32::try_from(shape[1]).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "apply_lfr: D",
      "must fit in i32",
      format_smolstr!("{}", shape[1]),
    ))
  })?;

  // The stacked LFR-frame width `lfr_m * D` (`sensevoice.py:62`). Computed once
  // through checked arithmetic so a huge `lfr_m` (or `D`) cannot wrap the `i32`
  // reshape extent into a small / negative value; the loader path additionally
  // pins `lfr_m * n_mels == input_size` in `Config::validate`, but this public
  // helper defends itself.
  let lfr_width = checked_mul("apply_lfr: lfr_m * D", "lfr_m", lfr_m, "D", d)?;

  // `T_lfr = ceil(T / lfr_n)` (`sensevoice.py:49`).
  let lfr_n_usize = lfr_n as usize;
  let t_lfr = t.div_ceil(lfr_n_usize);

  // Left-pad with `(lfr_m - 1) // 2` copies of the first frame
  // (`sensevoice.py:51-55`). `feats[:1]` tiled `(left_pad, 1)` then prepended.
  let left_pad = (lfr_m - 1) / 2;
  let first = ops::indexing::slice(feats, &[0, 0], &[1, d], &[1, 1])?;
  let padded = if left_pad > 0 {
    let head = ops::shape::tile(&first, &[left_pad, 1])?;
    ops::shape::concatenate(&[&head, feats], 0)?
  } else {
    feats.try_clone()?
  };
  let t_padded = padded.shape()[0];
  let t_padded_i32 = i32::try_from(t_padded).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "apply_lfr: padded T",
      "must fit in i32",
      format_smolstr!("{t_padded}"),
    ))
  })?;

  // The last frame of the padded sequence, used to right-pad a partial final
  // window (`feats[-1:]` in `sensevoice.py:68`).
  let last = ops::indexing::slice(&padded, &[t_padded_i32 - 1, 0], &[t_padded_i32, d], &[1, 1])?;

  // The LFR window count `t_lfr` is bounded by the fbank frame count `T`; reserve
  // it FALLIBLY (typed `AllocFailure`) rather than the abort `Vec::with_capacity`
  // would raise under allocator pressure on a long utterance.
  let lfr_m_usize = lfr_m as usize;
  let mut frames: Vec<Array> = Vec::new();
  reserve_or_error(&mut frames, "sensevoice apply_lfr: LFR frames", t_lfr)?;
  for i in 0..t_lfr {
    let start = i * lfr_n_usize;
    let end = start + lfr_m_usize;
    let start_i32 = i32::try_from(start).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "apply_lfr: window start",
        "must fit in i32",
        format_smolstr!("{start}"),
      ))
    })?;
    let stacked = if end <= t_padded {
      // Full window: `feats[start:end].reshape(-1)` (`sensevoice.py:62`).
      let end_i32 = start_i32 + lfr_m;
      let window = ops::indexing::slice(&padded, &[start_i32, 0], &[end_i32, d], &[1, 1])?;
      ops::shape::reshape(&window, &[lfr_width])?
    } else {
      // Partial final window: take what is available and right-pad with copies
      // of the last frame (`sensevoice.py:64-70`).
      let available = ops::indexing::slice(&padded, &[start_i32, 0], &[t_padded_i32, d], &[1, 1])?;
      let avail_rows = t_padded - start;
      let pad_count = i32::try_from(lfr_m_usize - avail_rows).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "apply_lfr: right-pad count",
          "must fit in i32",
          format_smolstr!("{}", lfr_m_usize - avail_rows),
        ))
      })?;
      let tail = ops::shape::tile(&last, &[pad_count, 1])?;
      let window = ops::shape::concatenate(&[&available, &tail], 0)?;
      ops::shape::reshape(&window, &[lfr_width])?
    };
    frames.push(stacked);
  }

  // Borrow each stacked frame for `stack`. The reference vector is `frames.len()`
  // long; reserve it FALLIBLY then fill (no infallible `collect` growth).
  let mut refs: Vec<&Array> = Vec::new();
  reserve_or_error(&mut refs, "sensevoice apply_lfr: frame refs", frames.len())?;
  for frame in &frames {
    refs.push(frame);
  }
  ops::shape::stack(&refs)
}

/// Apply global CMVN, mirroring `_apply_cmvn` (`sensevoice.py:75-80`):
/// `(feats + means) * istd` — an additive shift then a multiplicative scale.
///
/// `means` / `istd` are 1-D `(D,)` arrays broadcast across the `(T, D)`
/// features. The caller is responsible for matching `D` (the LFR width); a
/// shape mismatch surfaces as the underlying broadcast op error.
///
/// # Errors
/// Propagates the add / multiply op errors.
pub fn apply_cmvn(feats: &Array, means: &Array, istd: &Array) -> Result<Array> {
  feats.add(means)?.multiply(istd)
}

/// Parse the Kaldi MVN text statistics from an `am.mvn` file body, mirroring
/// `_parse_am_mvn` (`sensevoice.py:83-103`).
///
/// Returns `(means, istd)`: the `<AddShift>` vector (the additive shift, i.e.
/// negative per-dim means) and the `<Rescale>` vector (the inverse standard
/// deviations). Both are extracted by the same two regexes the reference uses,
/// in DOTALL mode, matching the bracketed float list that follows each tag's
/// `<LearnRateCoef> <n>` header.
///
/// # Errors
/// [`Error::MalformedData`] if either `<AddShift>` or `<Rescale>` is absent or
/// carries a non-float token (the reference raises `ValueError`).
pub fn parse_am_mvn(text: &str) -> Result<(Vec<f32>, Vec<f32>)> {
  // `<AddShift>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]` (DOTALL) — the additive
  // shift (`sensevoice.py:88-93`).
  let means = extract_bracketed_floats(text, r"(?s)<AddShift>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]")
    .ok_or_else(|| {
      Error::MalformedData(MalformedDataPayload::new(
        "parse_am_mvn",
        "could not parse <AddShift> means from am.mvn",
      ))
    })?;
  // `<Rescale>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]` (DOTALL) — the inverse
  // standard deviations (`sensevoice.py:96-101`).
  let istd = extract_bracketed_floats(text, r"(?s)<Rescale>.*?<LearnRateCoef>\s+\d+\s+\[(.*?)\]")
    .ok_or_else(|| {
    Error::MalformedData(MalformedDataPayload::new(
      "parse_am_mvn",
      "could not parse <Rescale> inverse-stddev from am.mvn",
    ))
  })?;
  Ok((means, istd))
}

/// Run `pattern` over `text` and parse the whitespace-separated float tokens in
/// its first capture group (the reference `[float(x) for x in
/// match.group(1).split()]`). Returns `None` if the pattern does not match;
/// returns `None` if any captured token fails to parse as a float (surfaced by
/// the caller as malformed `am.mvn`).
fn extract_bracketed_floats(text: &str, pattern: &str) -> Option<Vec<f32>> {
  let re = Regex::new(pattern).ok()?;
  let caps = re.captures(text)?;
  let body = caps.get(1)?.as_str();
  body
    .split_whitespace()
    .map(|tok| tok.parse::<f32>().ok())
    .collect()
}

/// Remap an HF/PyTorch checkpoint weight map to the MLX layout, mirroring
/// `sanitize` (`sensevoice.py:554-565`).
///
/// Two rules, applied in one pass:
/// 1. Strip the `ctc.ctc_lo.` prefix to `ctc_lo.` — the reference's CTC head
///    is nested under a `ctc.` module in the torch checkpoint
///    (`sensevoice.py:559`).
/// 2. For any `fsmn_block.weight` tensor of rank 3, transpose axes `(0, 2, 1)`:
///    torch stores a depthwise `Conv1d` weight as `(C_out, C_in/groups, K)`,
///    while MLX [`crate::ops::conv::conv1d`] expects `(C_out, K, C_in/groups)`
///    (`sensevoice.py:561-562`). This is the same conv-weight axis swap
///    wav2vec2 performs.
///
/// # Errors
/// Propagates the transpose op error (and a `dtype()` read on the candidate
/// FSMN weight).
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  let mut out = HashMap::with_capacity(weights.len());
  for (key, value) in weights {
    // Rule 1: `ctc.ctc_lo.` -> `ctc_lo.`.
    let new_key = key.replace("ctc.ctc_lo.", "ctc_lo.");

    // Rule 2: the depthwise FSMN conv weight `(C_out, C_in/groups, K)` ->
    // `(C_out, K, C_in/groups)` when rank-3.
    let new_value = if new_key.contains("fsmn_block.weight") && value.ndim() == 3 {
      ops::shape::transpose_axes(&value, &[0, 2, 1])?
    } else {
      value
    };

    // A duplicate key after the prefix strip would silently drop a tensor;
    // reject it rather than overwrite.
    if out.insert(new_key.clone(), new_value).is_some() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "sanitize: duplicate weight key after ctc.ctc_lo. -> ctc_lo. remap",
        "each remapped key must be unique",
      )));
    }
  }
  Ok(out)
}

#[cfg(test)]
mod tests;
