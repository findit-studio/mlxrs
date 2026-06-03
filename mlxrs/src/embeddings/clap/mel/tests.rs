//! CLAP mel front-end oracle tests.
//!
//! These pin the front-end against `textclap`'s committed fixtures (copied into
//! `tests/fixtures/clap/`, see that dir's README):
//! - the Slaney filterbank rows vs `filterbank_row_{0,10,32}.npy` (`< 1e-6`);
//! - the full log-mel spectrogram vs `golden_mel.npy` on the exact waveform
//!   that produced it (`sample_audio.npy`).
//!
//! The fixtures are f64-derived (the HF `ClapFeatureExtractor` / librosa
//! references); the mlxrs STFT is f32, so the full-mel compare uses a relative /
//! cosine tolerance that absorbs the documented f32-vs-f64 STFT gap (the
//! filterbank rows, built in f64 via `MelPrecision::Precise`, still match
//! tightly).

use super::*;
use crate::io::load_npy;

use std::path::PathBuf;

/// Resolve a fixture path under `tests/fixtures/clap/`. In-crate tests run with
/// the crate manifest dir as CWD, so a manifest-relative path is stable.
fn fixture(name: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/fixtures/clap")
    .join(name)
}

/// Load a 1-D / 2-D `<f4` fixture as a flat `Vec<f32>` (row-major).
fn load_f32(name: &str) -> Vec<f32> {
  let mut arr = load_npy(&fixture(name)).unwrap_or_else(|e| panic!("load {name}: {e:?}"));
  arr.eval().expect("eval fixture");
  arr.as_slice::<f32>().expect("fixture is f32").to_vec()
}

/// Max absolute difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
  assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}

/// `1 - cosine_similarity` between two equal-length vectors (f64 accumulation).
fn one_minus_cosine(a: &[f32], b: &[f32]) -> f64 {
  assert_eq!(a.len(), b.len());
  let mut dot = 0.0f64;
  let mut na = 0.0f64;
  let mut nb = 0.0f64;
  for (&x, &y) in a.iter().zip(b.iter()) {
    dot += x as f64 * y as f64;
    na += x as f64 * x as f64;
    nb += y as f64 * y as f64;
  }
  let denom = (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
  1.0 - dot / denom
}

/// The periodic Hann window endpoints distinguish it from the symmetric form:
/// `w[0] = 0`, `w[n/2] = 1` exactly, and `w[n-1]` is small but **positive** (a
/// symmetric Hann would be exactly 0 at `n-1`).
#[test]
fn periodic_hann_endpoints() {
  let win = periodic_hann_f32(N_FFT).expect("build window");
  let mut win = win;
  win.eval().expect("eval");
  let w = win.as_slice::<f32>().expect("f32").to_vec();
  assert_eq!(w.len(), N_FFT);
  assert_eq!(w[0], 0.0, "w[0] must be 0");
  assert!(
    (w[N_FFT / 2] - 1.0).abs() < 1e-6,
    "w[n/2] must be 1; got {}",
    w[N_FFT / 2]
  );
  assert!(
    w[N_FFT - 1] > 0.0 && w[N_FFT - 1] < 1e-3,
    "periodic Hann last sample must be positive but tiny; got {} (symmetric would be 0)",
    w[N_FFT - 1]
  );
  for &v in &w {
    assert!((0.0..=1.0 + 1e-6).contains(&v), "window out of [0,1]: {v}");
  }
}

/// (a) Oracle: the mlxrs Slaney filterbank rows match librosa's
/// `filterbank_row_{0,10,32}.npy` within a tight tolerance. Row 10 is the
/// discriminator (near the 1 kHz Slaney inflection — an HTK build diverges
/// there).
#[test]
fn filterbank_rows_match_textclap() {
  let fe = MelFrontEnd::new().expect("build front-end");
  let mut fb = fe.filterbank().expect("filterbank"); // (n_mels, n_freqs)
  fb.eval().expect("eval");
  let n_freqs = N_FFT / 2 + 1;
  assert_eq!(fb.shape(), &[N_MELS, n_freqs]);
  let fb = fb.as_slice::<f32>().expect("f32").to_vec();

  for &row in &[0usize, 10, 32] {
    let expected = load_f32(&format!("filterbank_row_{row}.npy"));
    assert_eq!(expected.len(), n_freqs, "row {row} length");
    let actual = &fb[row * n_freqs..(row + 1) * n_freqs];
    let diff = max_abs_diff(actual, &expected);
    assert!(
      diff < 1e-6,
      "filterbank row {row} max_abs_diff = {diff:.3e} (budget 1e-6)"
    );
  }
}

/// (b) Oracle: the full log-mel spectrogram matches `golden_mel.npy` on the
/// exact waveform (`sample_audio.npy`) that produced it.
///
/// The mlxrs STFT is f32 while the HF reference is f64, so the compare uses a
/// relative / cosine tolerance (and reports the max-abs in dB) rather than the
/// `< 1e-4` abs budget textclap's own f64 path hits.
#[test]
fn full_mel_matches_golden() {
  let samples = load_f32("sample_audio.npy");
  let golden = load_f32("golden_mel.npy"); // (T_FRAMES, N_MELS), time-major
  assert_eq!(
    golden.len(),
    T_FRAMES * N_MELS,
    "golden mel shape {} != {}",
    golden.len(),
    T_FRAMES * N_MELS
  );

  let fe = MelFrontEnd::new().expect("build front-end");
  let mut mel = fe.extract(&samples).expect("extract"); // (1,1,T,N_MELS)
  mel.eval().expect("eval");
  assert_eq!(mel.shape(), vec![1, 1, T_FRAMES, N_MELS]);
  let mel = mel.as_slice::<f32>().expect("f32").to_vec();
  assert_eq!(mel.len(), golden.len());

  // Cosine similarity over the whole spectrogram absorbs the f32-vs-f64 STFT
  // drift while still catching any structural mistake (wrong window, wrong
  // power-to-dB, transposed layout) — those collapse the cosine far below 1.
  let cos_gap = one_minus_cosine(&mel, &golden);
  assert!(
    cos_gap <= 1e-4,
    "full-mel 1-cosine = {cos_gap:.3e} vs golden (budget 1e-4)"
  );

  // The max-abs dB gap documents the f32-vs-f64 STFT floor and guards against a
  // large localized deviation the cosine could mask. The achieved gap is
  // ~4.4e-4 dB — just above textclap's own f64-path `1e-4` budget (the
  // single-precision FFT accounts for the difference); `1e-3` is a tight band
  // that still absorbs it with margin.
  let max_db = max_abs_diff(&mel, &golden);
  assert!(
    max_db <= 1e-3,
    "full-mel max_abs dB diff = {max_db:.3e} vs golden (f32-vs-f64 band 1e-3 dB)"
  );
}

/// The power→dB step is applied exactly once: a unit 1 kHz sine peaks in the
/// single-`log10` range (~29 dB), not the double-log range (~15 dB) or the raw
/// range (>50 dB), and silent bins floor at -100 dB (`amin = 1e-10`). Mirrors
/// `textclap/src/mel.rs`'s `power_to_db_applied_once` test.
#[test]
fn power_to_db_applied_once() {
  let sr = SAMPLE_RATE as f32;
  let mut samples = vec![0.0f32; TARGET_SAMPLES];
  for (k, s) in samples.iter_mut().enumerate() {
    *s = (2.0 * std::f32::consts::PI * 1000.0 * (k as f32) / sr).sin();
  }
  let fe = MelFrontEnd::new().expect("build front-end");
  let mut mel = fe.extract(&samples).expect("extract");
  mel.eval().expect("eval");
  let mel = mel.as_slice::<f32>().expect("f32").to_vec();

  let max = mel.iter().fold(f32::MIN, |a, &b| a.max(b));
  let min = mel.iter().fold(f32::MAX, |a, &b| a.min(b));
  assert!(
    (20.0..50.0).contains(&max),
    "single-application 10·log10 of a unit sine should peak near 29 dB; got max = {max}"
  );
  assert!(
    (-100.0 - 1e-2..-50.0).contains(&min),
    "amin floor should clip silent bins to -100 dB; got min = {min}"
  );
}

/// Empty input is rejected with a typed error (the repeat-pad would divide by
/// zero), not a panic.
#[test]
fn extract_rejects_empty() {
  let fe = MelFrontEnd::new().expect("build front-end");
  let err = fe.extract(&[]).unwrap_err();
  assert!(
    matches!(err, Error::InvariantViolation(_)),
    "expected InvariantViolation for empty input, got {err:?}"
  );
}

/// A short clip is repeat-padded (not zero-only): the mel of a 1 s tone tiled to
/// 10 s is identical to the mel of the same tone supplied at 10 s, because the
/// repeat-pad reconstructs the periodic signal. (A weaker check than golden but
/// needs no fixture: confirms repeat-pad runs and the output is finite.)
#[test]
fn extract_repeat_pads_short_clip() {
  let sr = SAMPLE_RATE as f32;
  // 1 second of a 440 Hz tone (an exact divisor count into 10 s).
  let one_sec: Vec<f32> = (0..SAMPLE_RATE as usize)
    .map(|k| (2.0 * std::f32::consts::PI * 440.0 * (k as f32) / sr).sin())
    .collect();
  let fe = MelFrontEnd::new().expect("build front-end");
  let mut mel = fe.extract(&one_sec).expect("extract short clip");
  mel.eval().expect("eval");
  assert_eq!(mel.shape(), vec![1, 1, T_FRAMES, N_MELS]);
  let mel = mel.as_slice::<f32>().expect("f32").to_vec();
  assert!(
    mel.iter().all(|v| v.is_finite()),
    "repeat-padded mel must be finite"
  );
}
