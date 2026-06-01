use super::*;
use crate::audio::dsp;

fn to_vec(a: &Array) -> Vec<f32> {
  a.try_clone().unwrap().to_vec::<f32>().unwrap()
}

/// A deterministic non-trivial test waveform: a sum of two sinusoids so the
/// mel energy is spread across several bins (a pure DC / single tone would
/// collapse the dynamic-range clamp). Length `1600` → with `center=true`
/// STFT (`pad n_fft/2` each side, `num_frames = 1 + n_samples / hop`) gives
/// `1 + 1600/160 = 11` frames, `10` after dropping the last.
fn sine_mix(n: usize) -> Vec<f32> {
  (0..n)
    .map(|i| {
      let t = i as f32;
      0.6 * (2.0 * std::f32::consts::PI * 440.0 * t / SAMPLE_RATE as f32).sin()
        + 0.3 * (2.0 * std::f32::consts::PI * 1200.0 * t / SAMPLE_RATE as f32).sin()
    })
    .collect()
}

/// The Whisper hyperparameters must match the reference literals exactly.
#[test]
fn whisper_hyperparams_match_reference() {
  assert_eq!(SAMPLE_RATE, 16_000);
  assert_eq!(N_FFT, 400);
  assert_eq!(HOP_LENGTH, 160);
  assert_eq!(CHUNK_LENGTH, 30);
  assert_eq!(N_SAMPLES, 480_000);
  assert_eq!(N_FRAMES, 3000);
  assert_eq!(N_SAMPLES_PER_TOKEN, 320);
  assert_eq!(FRAMES_PER_SECOND, 100);
  assert_eq!(TOKENS_PER_SECOND, 50);
}

/// Oracle test for [`log_mel_spectrogram_whisper`]: recompute the reference
/// equation (`audio.py:72-82`) independently from the lower-level public
/// primitives — `dsp::stft` → drop last frame → `|·|²` → Slaney
/// `mel_filter_bank_scaled` → matmul → `log10` → `max(., max-8)` →
/// `(x+4)/4` — and assert the helper reproduces it bit-for-bit (modulo f32).
///
/// This is an *independent* composition of the reference formula via separate
/// ops, NOT a call to the function under test, so it is a valid oracle.
#[test]
fn log_mel_spectrogram_whisper_matches_reference_equation() {
  let buf = sine_mix(1600);
  let x = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();
  let n_mels = 8usize;

  let got = to_vec(&log_mel_spectrogram_whisper(&x, n_mels, 0).unwrap());

  // Independent reference build.
  let expected = {
    let spec = dsp::stft(&x, N_FFT, HOP_LENGTH, None, dsp::WindowPad::Right).unwrap();
    let freqs = spec.data_ref();
    let num_frames = freqs.shape()[0] as i32;
    let n_freqs = freqs.shape()[1] as i32;
    // Drop last frame.
    let dropped =
      crate::ops::indexing::slice(freqs, &[0, 0], &[num_frames - 1, n_freqs], &[1, 1]).unwrap();
    let mag = dropped.abs().unwrap().square().unwrap();
    let filters = dsp::mel_filter_bank_scaled(
      n_mels,
      N_FFT,
      SAMPLE_RATE,
      0.0,
      None,
      dsp::MelScale::Slaney,
      true,
    )
    .unwrap();
    let mel_spec = crate::ops::linalg_basic::matmul(&mag, &filters.transpose().unwrap()).unwrap();
    let floor = Array::full::<f32>(&[0i32; 0], 1e-10).unwrap();
    let log_spec = crate::ops::arithmetic::maximum(&mel_spec, &floor)
      .unwrap()
      .log10()
      .unwrap();
    let peak = crate::ops::reduction::max(&log_spec, false).unwrap();
    let range = Array::full::<f32>(&[0i32; 0], 8.0).unwrap();
    let cf = crate::ops::arithmetic::subtract(&peak, &range).unwrap();
    let clamped = crate::ops::arithmetic::maximum(&log_spec, &cf).unwrap();
    let off = Array::full::<f32>(&[0i32; 0], 4.0).unwrap();
    let div = Array::full::<f32>(&[0i32; 0], 4.0).unwrap();
    let shifted = crate::ops::arithmetic::add(&clamped, &off).unwrap();
    to_vec(&crate::ops::arithmetic::divide(&shifted, &div).unwrap())
  };

  assert_eq!(got.len(), expected.len(), "log-mel length mismatch");
  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    assert!(
      (g - e).abs() < 1e-5,
      "log_mel_spectrogram_whisper[{i}] = {g} (want {e})"
    );
  }
}

/// Shape contract: a `1600`-sample input yields `10` frames (`11` STFT frames
/// minus the dropped last) and `n_mels` columns — frames on axis 0, mel bins
/// on axis 1 (the Whisper `(num_frames, n_mels)` layout).
#[test]
fn log_mel_spectrogram_whisper_shape_drops_last_frame() {
  let buf = sine_mix(1600);
  let x = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();
  let n_mels = 12usize;
  let mel = log_mel_spectrogram_whisper(&x, n_mels, 0).unwrap();
  let shape = mel.shape();
  assert_eq!(shape.len(), 2, "mel must be 2-D");
  // center=true: num_frames = 1 + n_samples/hop = 1 + 10 = 11; drop 1 → 10.
  assert_eq!(
    shape[0], 10,
    "frame axis (axis 0) after dropping last frame"
  );
  assert_eq!(shape[1], n_mels, "mel-bin axis (axis 1)");
}

/// The Whisper post-processing uses `log10` (decimal), NOT the natural log
/// the generic [`dsp::log_mel_spectrogram`] uses, and applies the
/// `max(., max-8)` + `(x+4)/4` tail. Pin the dynamic-range invariant: once
/// the `max(., max-8)` clamp is active, the output spans **exactly** 2.0
/// (`((max+4) - (max-8+4)) / 4 = 8/4 = 2`) — a closed-form property of the
/// affine+clamp independent of the input, which a `ln`-based or
/// clamp-omitting implementation would violate.
#[test]
fn log_mel_spectrogram_whisper_dynamic_range_is_two() {
  let buf = sine_mix(1600);
  let x = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();
  let mel = to_vec(&log_mel_spectrogram_whisper(&x, 16, 0).unwrap());
  let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let min = mel.iter().copied().fold(f32::INFINITY, f32::min);
  // Peak maps to (max+4)/4; the 8-decade floor maps to (max-8+4)/4; the gap
  // is exactly 2.0 when the clamp binds (it does for this multi-tone input).
  assert!(
    (max - min - 2.0).abs() < 1e-4,
    "dynamic range must be exactly 2.0 (got max={max}, min={min}, span={})",
    max - min
  );
  // The peak value equals (max_log + 4)/4; for normalized features this is
  // bounded above by ~0.2..0.3 in practice, but the structural assertion is
  // simply that the max is finite and the span is 2.
  assert!(max.is_finite() && min.is_finite(), "log-mel must be finite");
}

/// Right-padding (`padding > 0`) appends zero samples, increasing the frame
/// count by `padding / hop`. Pin that the `padding` argument is honored.
#[test]
fn log_mel_spectrogram_whisper_padding_extends_frames() {
  let buf = sine_mix(1600);
  let x = Array::from_slice::<f32>(&buf, &[buf.len() as i32]).unwrap();
  let unpadded = log_mel_spectrogram_whisper(&x, 8, 0).unwrap();
  // Pad by 320 samples = 2 hops → 2 more frames.
  let padded = log_mel_spectrogram_whisper(&x, 8, 320).unwrap();
  assert_eq!(
    padded.shape()[0],
    unpadded.shape()[0] + 2,
    "padding by 2 hops must add 2 frames"
  );
}

/// `log_mel_spectrogram_whisper` rejects a non-1-D input.
#[test]
fn log_mel_spectrogram_whisper_rejects_non_1d() {
  let x = Array::zeros::<f32>(&[2i32, 8]).unwrap();
  assert!(log_mel_spectrogram_whisper(&x, 8, 0).is_err());
}

// ---- pad_or_trim ----------------------------------------------------

/// `pad_or_trim` trims along axis 0 when the array is longer than `length`.
#[test]
fn pad_or_trim_trims_axis0() {
  // (5, 3) → trim to (2, 3).
  let buf: Vec<f32> = (0..15).map(|i| i as f32).collect();
  let a = Array::from_slice::<f32>(&buf, &[5i32, 3]).unwrap();
  let out = pad_or_trim(&a, 2, 0).unwrap();
  assert_eq!(out.shape(), vec![2, 3]);
  // First two rows preserved: [0,1,2, 3,4,5].
  assert_eq!(to_vec(&out), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// `pad_or_trim` zero-pads along axis 0 when the array is shorter.
#[test]
fn pad_or_trim_pads_axis0() {
  // (2, 3) → pad to (4, 3): two zero rows appended.
  let buf: Vec<f32> = (0..6).map(|i| i as f32).collect();
  let a = Array::from_slice::<f32>(&buf, &[2i32, 3]).unwrap();
  let out = pad_or_trim(&a, 4, 0).unwrap();
  assert_eq!(out.shape(), vec![4, 3]);
  assert_eq!(
    to_vec(&out),
    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  );
}

/// `pad_or_trim` is a no-op (a clone) when the length already matches.
#[test]
fn pad_or_trim_noop_when_equal() {
  let buf: Vec<f32> = (0..6).map(|i| i as f32).collect();
  let a = Array::from_slice::<f32>(&buf, &[3i32, 2]).unwrap();
  let out = pad_or_trim(&a, 3, 0).unwrap();
  assert_eq!(out.shape(), vec![3, 2]);
  assert_eq!(to_vec(&out), buf);
}

/// `pad_or_trim` on the canonical Whisper frame axis lands the mel on
/// exactly [`N_FRAMES`] rows — trim a too-long mel down to 3000.
#[test]
fn pad_or_trim_lands_on_n_frames() {
  // Build a (3005, 4) mel-like array, trim to N_FRAMES (3000) on axis 0.
  let rows = 3005usize;
  let cols = 4usize;
  let buf = vec![1.0_f32; rows * cols];
  let a = Array::from_slice::<f32>(&buf, &[rows as i32, cols as i32]).unwrap();
  let out = pad_or_trim(&a, N_FRAMES, 0).unwrap();
  assert_eq!(out.shape(), vec![N_FRAMES, cols]);
}

/// `pad_or_trim` rejects an out-of-bounds axis.
#[test]
fn pad_or_trim_rejects_bad_axis() {
  let a = Array::zeros::<f32>(&[3i32, 2]).unwrap();
  assert!(pad_or_trim(&a, 5, 2).is_err());
}
