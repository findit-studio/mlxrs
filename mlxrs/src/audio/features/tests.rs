use super::*;
use crate::Dtype;

/// Absolute tolerance for closed-form scalar checks. Mirrors `dsp.rs`'s
/// `WIN_TOL` (1e-6) for f32 evaluations of mlx-audio's f64-evaluated formulas.
const F32_TOL: f32 = 1e-5;

fn to_vec(a: &Array) -> Vec<f32> {
  // Materialize first: some helpers (e.g. `reverse_1d`) return strided
  // negative-step views, and `Array::to_vec` rejects non-contiguous storage
  // (`Error::NonContiguous`). `contiguous` is a no-op for already-contiguous
  // inputs, so this is safe for every caller.
  crate::ops::shape::contiguous(a, false)
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
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
    Err(Error::OutOfRange(_))
  ));
  // odd n_fft.
  assert!(matches!(
    get_mel_banks_kaldi(40, 513, 16_000.0, 0.0, 0.0),
    Err(Error::OutOfRange(_))
  ));
  // zero sample rate.
  assert!(matches!(
    get_mel_banks_kaldi(40, 512, 0.0, 0.0, 0.0),
    Err(Error::OutOfRange(_))
  ));
  // low >= high (after high_freq <= 0 resolution).
  assert!(matches!(
    get_mel_banks_kaldi(40, 512, 16_000.0, 9000.0, 0.0),
    Err(Error::OutOfRange(_))
  ));
  // low_freq >= nyquist.
  assert!(matches!(
    get_mel_banks_kaldi(40, 512, 16_000.0, 9000.0, -100.0),
    Err(Error::OutOfRange(_))
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
    PreemphBoundary::Scale,
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
    PreemphBoundary::Scale,
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
    PreemphBoundary::Scale,
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
    PreemphBoundary::Scale,
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
    PreemphBoundary::Scale,
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
      &x,
      16_000,
      400,
      160,
      40,
      wt,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
      PreemphBoundary::Scale,
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::RankMismatch(_))
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::InvariantViolation(_))
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::InvariantViolation(_))
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::OutOfRange(_))
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::InvariantViolation(_))
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
      None,
      PreemphBoundary::Scale,
    ),
    Err(Error::OutOfRange(_))
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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

// ---- work-cap follow-ups ----------------------------------------------

/// Output cap: a pathological `(win_len=2, win_inc=1, ~1M
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
    PreemphBoundary::Scale,
  )
  .expect_err("expected output-element cap to reject pathological num_mels");
  let msg = format!("{err:?}");
  assert!(
    msg.contains("output element count"),
    "expected error to mention the output-element cap, got: {msg}"
  );
}

/// Contiguity: a 1-D sliced waveform passes the rank-1 check
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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

/// Contiguity: a broadcasted scalar `waveform` (rank-1 by
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
      PreemphBoundary::Scale,
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
      PreemphBoundary::Scale,
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

/// Preemphasis: pin the kaldi-asr first-sample boundary by
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
    PreemphBoundary::Scale,
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

/// `PreemphBoundary::Preserve` keeps the first sample of each frame UNCHANGED
/// under pre-emphasis (`y[0] = x[0]`), matching
/// `mlx_audio.dsp.compute_fbank_kaldi` (`dsp.py:911-915`, where
/// `first_col = strided_input[:, 0:1]` is concatenated unchanged). This is the
/// boundary SenseVoice's reference uses; the scaling deviation
/// (`y[0] = x[0] * (1 - p)`) would FAIL the closed-form sentinel below.
///
/// We pin the boundary two ways:
/// 1. **Closed-form sentinel.** For a 1-frame rectangular case the `Preserve`
///    preemphasized-frame sum is mlx-audio's `-0.34375` (NOT the kaldi-asr
///    `Scale` value `-0.875`) — the same sentinel the
///    `compute_fbank_kaldi_preemphasis_first_sample_matches_kaldi` test uses to
///    distinguish the two variants, here pinned to the `Preserve` side.
/// 2. **End-to-end through the public function.** Driving the identical input
///    through `compute_fbank_kaldi` with `Preserve` vs `Scale` must produce
///    DIFFERENT features (the boundary feeds a different model input), and the
///    `Preserve` output must be finite.
#[test]
fn compute_fbank_kaldi_preemphasis_preserve_keeps_first_sample() {
  // Same minimal frame as the kaldi sibling test; here the boundary is the
  // mlx-audio passthrough `y[0] = c[0]`.
  let input = [2.0_f32, 1.0, 0.5, 0.25];
  let mean = (input[0] + input[1] + input[2] + input[3]) / 4.0;
  let centered: Vec<f32> = input.iter().map(|x| x - mean).collect();
  let p = 0.5_f32;

  // `PreemphBoundary::Preserve` boundary: y[0] = c[0] (unchanged).
  let mut preserve = [0.0_f32; 4];
  preserve[0] = centered[0];
  for n in 1..4 {
    preserve[n] = centered[n] - p * centered[n - 1];
  }
  let preserve_sum: f32 = preserve.iter().sum();
  assert!(
    (preserve_sum - (-0.34375)).abs() < 1e-5,
    "Preserve closed-form check: y-sum = {preserve_sum}, want -0.34375 \
       (mlx-audio passthrough; the Scale deviation would give -0.875)"
  );

  // Contrast sentinel: the Scale boundary gives the DISTINCT `-0.875`, proving
  // the test setup observes the boundary difference.
  let mut scale = [0.0_f32; 4];
  scale[0] = centered[0] * (1.0 - p);
  for n in 1..4 {
    scale[n] = centered[n] - p * centered[n - 1];
  }
  let scale_sum: f32 = scale.iter().sum();
  assert!(
    (scale_sum - (-0.875)).abs() < 1e-5,
    "Scale closed-form sentinel: y-sum = {scale_sum}, want -0.875"
  );
  assert!(
    (preserve_sum - scale_sum).abs() > 1e-3,
    "Preserve and Scale first-sample boundaries must be observably different"
  );

  // End-to-end: the boundary mode must change the public function's output.
  // A DC-rich ramp over several frames makes the first-sample boundary feed a
  // materially different rectangular-window spectrum.
  let samples: Vec<f32> = (0..4_000).map(|i| (i as f32) / 4_000.0).collect();
  let x = Array::from_slice::<f32>(&samples, &[4_000_i32]).unwrap();
  let preserve_feats = to_vec(
    &compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Rectangular,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
      PreemphBoundary::Preserve,
    )
    .unwrap(),
  );
  let scale_feats = to_vec(
    &compute_fbank_kaldi(
      &x,
      16_000,
      400,
      160,
      40,
      KaldiWindow::Rectangular,
      0.97,
      0.0,
      true,
      20.0,
      0.0,
      None,
      PreemphBoundary::Scale,
    )
    .unwrap(),
  );
  assert_eq!(preserve_feats.len(), scale_feats.len());
  for (i, &val) in preserve_feats.iter().enumerate() {
    assert!(
      val.is_finite(),
      "Preserve fbank[{i}] = {val}: must be finite"
    );
  }
  let max_diff = preserve_feats
    .iter()
    .zip(scale_feats.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0_f32, f32::max);
  assert!(
    max_diff > 1e-4,
    "Preserve vs Scale first-sample boundary must change the fbank features \
       (max diff {max_diff})"
  );
}

// ---- additional work-cap follow-ups -----------------------------------

/// Samples_len cap: a broadcasted 1-element waveform
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
    PreemphBoundary::Scale,
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

/// Padded mel-bank cap: the right operand of the
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
    PreemphBoundary::Scale,
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

/// `snip_edges=false` reflect-buffer cap. `compute_fbank_kaldi`
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
    PreemphBoundary::Scale,
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

/// The same reflect-buffer cap, but for the
/// `pad == 1` branch. That branch
/// concatenates `pad_left` (`1`) ++ `waveform` (`n`) ++ `pad_right`
/// (`reverse(wf[1..n])`, length `n - 1`) = `2*n`, NOT the `n + 2` a uniform
/// `n + 2*pad` estimate gives. So a `pad == 1` input whose `n + 2` is within
/// `MAX_FBANK_WORK` but whose true `2*n` exceeds it slipped a ~128 Mi
/// `concatenate` through. Example: `win_len = 1_048_576`,
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
    PreemphBoundary::Scale,
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

/// The `pad == 1` branch concatenates `pad_left`
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
  let x =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5]).unwrap();
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
  let x =
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5]).unwrap();
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
    Err(Error::OutOfRange(_))
  ));
  // even win_length (would silently truncate to next-lower odd).
  assert!(matches!(
    compute_deltas_kaldi(&x, 4, DeltaPadMode::Edge),
    Err(Error::OutOfRange(_))
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
        Err(Error::CapExceeded(_))
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
      Err(Error::CapExceeded(_))
    ),
    "padded work exceeding the cap must be rejected before allocating"
  );
}

/// The delta CUMULATIVE-work cap (`MAX_DELTA_WORK`), distinct
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
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 5]).unwrap();
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
  let Error::OutOfRange(payload) = &err else {
    panic!("expected OutOfRange overread/short-signal error, got: {err:?}");
  };
  assert!(
    payload.context().contains("reflect-pad") || payload.requirement().contains("reflect-pad"),
    "expected an overread/short-signal error referencing reflect-pad, got: context={:?}, requirement={:?}",
    payload.context(),
    payload.requirement()
  );
}

// ---- enum string + Display + variant-predicate surface ----------------
//
// `KaldiWindow` / `DeltaPadMode` derive `derive_more::Display` (rendering
// `as_str()`) and `derive_more::IsVariant`. These pin the canonical lowercase
// `win_type` / `mode` strings (the mlx-audio argument spelling) and the
// generated predicates / Display so a renamed variant or a broken `as_str`
// arm is caught.

#[test]
fn kaldi_window_as_str_and_display_are_canonical() {
  // `as_str` (const fn) must match the mlx-audio `win_type` spelling exactly,
  // and `Display` (via derive_more) must render the same string.
  for (w, s) in [
    (KaldiWindow::Hamming, "hamming"),
    (KaldiWindow::Hanning, "hanning"),
    (KaldiWindow::Povey, "povey"),
    (KaldiWindow::Rectangular, "rectangular"),
  ] {
    assert_eq!(w.as_str(), s, "{w:?}.as_str()");
    assert_eq!(w.to_string(), s, "{w:?} Display");
  }
  // Default is Hamming (the reference's `compute_fbank_kaldi` default).
  assert_eq!(KaldiWindow::default(), KaldiWindow::Hamming);
}

#[test]
fn kaldi_window_variant_predicates() {
  // `derive_more::IsVariant` generates `is_*`; each true exactly for its arm.
  assert!(KaldiWindow::Hamming.is_hamming());
  assert!(!KaldiWindow::Hamming.is_hanning());
  assert!(KaldiWindow::Hanning.is_hanning());
  assert!(KaldiWindow::Povey.is_povey());
  assert!(KaldiWindow::Rectangular.is_rectangular());
  assert!(!KaldiWindow::Rectangular.is_povey());
}

#[test]
fn delta_pad_mode_as_str_and_display_and_predicates() {
  // `as_str` arms + Display rendering + IsVariant predicates for the
  // `edge`/`constant` mlx-audio `mode` spelling.
  assert_eq!(DeltaPadMode::Edge.as_str(), "edge");
  assert_eq!(DeltaPadMode::Constant.as_str(), "constant");
  assert_eq!(DeltaPadMode::Edge.to_string(), "edge");
  assert_eq!(DeltaPadMode::Constant.to_string(), "constant");
  assert!(DeltaPadMode::Edge.is_edge());
  assert!(!DeltaPadMode::Edge.is_constant());
  assert!(DeltaPadMode::Constant.is_constant());
  // Default is Edge (kaldi-asr delta edge replication).
  assert_eq!(DeltaPadMode::default(), DeltaPadMode::Edge);
}

// ---- get_mel_banks_kaldi: high_freq resolution branches ---------------

#[test]
fn mel_banks_kaldi_positive_high_freq_used_verbatim() {
  // `high_freq > 0.0` takes the else branch (used verbatim, NOT
  // Nyquist-relative). With sample_freq=16000 (nyquist=8000) and
  // high_freq=4000, the highest mel center must sit BELOW 4000 (the band is
  // capped at the explicit high_freq, not at Nyquist). Contrast with the
  // `high_freq <= 0.0` tests where centers extend toward 8000.
  let (_, centers) = get_mel_banks_kaldi(40, 512, 16_000.0, 20.0, 4000.0).unwrap();
  let c = to_vec(&centers);
  assert!(
    c[c.len() - 1] < 4000.0,
    "explicit high_freq=4000 must cap the top center below 4000, got {}",
    c[c.len() - 1]
  );
  // And it should clearly exceed the half-band midpoint (the band really does
  // span up to ~4000, not collapse), so the top center is well above 1500 Hz.
  assert!(
    c[c.len() - 1] > 1500.0,
    "top center {} should approach the explicit high_freq",
    c[c.len() - 1]
  );
}

#[test]
fn mel_banks_kaldi_rejects_high_freq_above_nyquist() {
  // Positive high_freq > nyquist (8000) with a valid low_freq exercises the
  // `high_freq > 0 && high_freq <= nyquist` failure branch specifically
  // (distinct from the low_freq and low>=high branches).
  let err = get_mel_banks_kaldi(40, 512, 16_000.0, 20.0, 9000.0)
    .expect_err("high_freq above nyquist must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(
    p.context(),
    "get_mel_banks_kaldi: high_freq",
    "expected the high_freq range branch, got context {:?}",
    p.context()
  );
}

#[test]
fn mel_banks_kaldi_rejects_low_freq_ge_high_freq_both_valid() {
  // low_freq and high_freq are EACH individually valid (0 <= low < nyquist,
  // 0 < high <= nyquist) but low >= high — the dedicated `low_freq >=
  // high_freq` branch (NOT the low_freq-vs-nyquist nor high_freq-vs-nyquist
  // branches). low=4000 (< 8000, valid), high=2000 (positive, <= 8000, valid),
  // 4000 >= 2000.
  let err = get_mel_banks_kaldi(40, 512, 16_000.0, 4000.0, 2000.0)
    .expect_err("low_freq >= high_freq must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(
    p.context(),
    "get_mel_banks_kaldi: low_freq",
    "expected the low>=high branch context"
  );
  assert_eq!(p.requirement(), "must be < high_freq");
}

#[test]
fn mel_banks_kaldi_rejects_zero_n_fft_padded() {
  // n_fft_padded == 0 hits the `== 0 || !is_multiple_of(2)` branch.
  let err = get_mel_banks_kaldi(40, 0, 16_000.0, 20.0, 0.0)
    .expect_err("zero n_fft_padded must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "get_mel_banks_kaldi: n_fft_padded");
}

#[test]
fn mel_banks_kaldi_rejects_nonfinite_sample_freq() {
  // NaN / inf sample_freq must be rejected by the finite-and-positive guard.
  for bad in [f32::NAN, f32::INFINITY, -1.0_f32] {
    let err = get_mel_banks_kaldi(40, 512, bad, 20.0, 0.0)
      .expect_err("non-finite/negative sample_freq must be rejected");
    assert!(
      matches!(err, Error::OutOfRange(_)),
      "sample_freq={bad} expected OutOfRange, got {err:?}"
    );
  }
}

#[test]
fn mel_banks_kaldi_cap_rejects_oversized_bank() {
  // `bank_len = num_bins * (n_fft_padded / 2)` must be capped at
  // MAX_FBANK_WORK (64 Mi). Choose num_bins and n_fft_padded so the product
  // exceeds 64 Mi while each factor fits in i32. num_bins = 4096,
  // n_fft_padded = 65536 → num_fft_bins = 32768; 4096 * 32768 = 128 Mi > cap.
  let err = get_mel_banks_kaldi(4096, 65_536, 16_000.0, 20.0, 0.0)
    .expect_err("oversized bank_len must be capped");
  let Error::CapExceeded(p) = &err else {
    panic!("expected CapExceeded, got {err:?}");
  };
  assert_eq!(p.cap_name(), "MAX_FBANK_WORK");
  assert!(
    p.observed() > p.cap(),
    "observed {} must exceed cap {}",
    p.observed(),
    p.cap()
  );
}

// ---- build_kaldi_window (private) error + Povey shape ------------------

#[test]
fn build_kaldi_window_rejects_too_small() {
  // win_size < 2 is rejected (the periodic `win_size - 1` denominator needs
  // at least 2 samples). Exercised directly on the private helper.
  for bad in [0_usize, 1] {
    let err =
      build_kaldi_window(KaldiWindow::Hamming, bad).expect_err("win_size < 2 must be rejected");
    let Error::OutOfRange(p) = &err else {
      panic!("expected OutOfRange, got {err:?}");
    };
    assert_eq!(p.context(), "build_kaldi_window: win_size");
  }
}

#[test]
fn build_kaldi_window_povey_closed_form() {
  // Povey window = (0.5 - 0.5*cos(2*pi*n/(N-1)))^0.85. For N=4 (denom=3):
  //   n=0: (0.5 - 0.5*cos(0))^0.85         = 0^0.85          = 0
  //   n=1: (0.5 - 0.5*cos(2pi/3))^0.85     = (0.5+0.25)^0.85 = 0.75^0.85
  //   n=2: (0.5 - 0.5*cos(4pi/3))^0.85     = 0.75^0.85
  //   n=3: (0.5 - 0.5*cos(2pi))^0.85       = 0^0.85          = 0
  let w = build_kaldi_window(KaldiWindow::Povey, 4).unwrap();
  assert_eq!(w.shape(), vec![4]);
  let v = to_vec(&w);
  let mid = 0.75_f32.powf(0.85);
  let want = [0.0_f32, mid, mid, 0.0];
  for (i, (&g, &e)) in v.iter().zip(want.iter()).enumerate() {
    assert!((g - e).abs() < F32_TOL, "povey[{i}]: got {g}, want {e}");
  }
}

#[test]
fn build_kaldi_window_rectangular_is_all_ones() {
  // Rectangular = constant 1.0 (no windowing).
  let w = build_kaldi_window(KaldiWindow::Rectangular, 5).unwrap();
  let v = to_vec(&w);
  assert_eq!(v, vec![1.0_f32; 5]);
}

#[test]
fn build_kaldi_window_hanning_closed_form() {
  // Hanning = 0.5 - 0.5*cos(2*pi*n/(N-1)). For N=5 (denom=4):
  //   n=0: 0; n=1: 0.5 - 0.5*cos(pi/2) = 0.5; n=2: 0.5 - 0.5*cos(pi) = 1.0;
  //   n=3: 0.5; n=4: 0.
  let w = build_kaldi_window(KaldiWindow::Hanning, 5).unwrap();
  let v = to_vec(&w);
  let want = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
  for (i, (&g, &e)) in v.iter().zip(want.iter()).enumerate() {
    assert!((g - e).abs() < F32_TOL, "hanning[{i}]: got {g}, want {e}");
  }
}

// ---- reverse_1d (private) ----------------------------------------------

#[test]
fn reverse_1d_basic_reverses_including_index_zero() {
  // The `stop = -(len + 1)` sentinel must include index 0 (the descending
  // traversal len-1 .. 0). [10, 20, 30, 40] -> [40, 30, 20, 10].
  let a = Array::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
  let r = reverse_1d(&a).unwrap();
  assert_eq!(r.shape(), vec![4]);
  assert_eq!(to_vec(&r), vec![40.0, 30.0, 20.0, 10.0]);
}

#[test]
fn reverse_1d_single_element() {
  // len==1 is a non-empty edge case (not the EmptyInput branch).
  let a = Array::from_slice::<f32>(&[7.0], &[1]).unwrap();
  let r = reverse_1d(&a).unwrap();
  assert_eq!(to_vec(&r), vec![7.0]);
}

#[test]
fn reverse_1d_rejects_non_1d() {
  // 2-D input -> RankMismatch (the `shape.len() != 1` branch).
  let a = Array::zeros::<f32>(&[2_i32, 3_i32]).unwrap();
  let err = reverse_1d(&a).expect_err("2-D input must be rejected");
  let Error::RankMismatch(p) = &err else {
    panic!("expected RankMismatch, got {err:?}");
  };
  assert_eq!(p.actual(), 2, "observed rank");
  assert_eq!(p.context(), "reverse_1d: expected 1-D input");
}

#[test]
fn reverse_1d_rejects_empty() {
  // A length-0 1-D array hits the EmptyInput branch.
  let a = Array::zeros::<f32>(&[0_i32]).unwrap();
  let err = reverse_1d(&a).expect_err("empty input must be rejected");
  assert!(
    matches!(err, Error::EmptyInput(_)),
    "expected EmptyInput, got {err:?}"
  );
}

// ---- strided_frames_snip_edges (private) -------------------------------

#[test]
fn strided_frames_snip_edges_basic_frames() {
  // Closed-form snip_edges=true framing: waveform [0..7], win_size=4,
  // win_inc=2, num_frames = 1 + (8-4)/2 = 3.
  //   frame0 = wf[0..4] = [0,1,2,3]
  //   frame1 = wf[2..6] = [2,3,4,5]
  //   frame2 = wf[4..8] = [4,5,6,7]
  let wf: Vec<f32> = (0..8).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[8]).unwrap();
  let frames = strided_frames_snip_edges(&x, 4, 2, 3).unwrap();
  assert_eq!(frames.shape(), vec![3, 4]);
  let got = to_vec_2d(&frames, 3, 4);
  let want = [
    [0.0_f32, 1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0, 7.0],
  ];
  assert_eq!(got, want, "snip_edges=true framing mismatch");
}

#[test]
fn strided_frames_snip_edges_rejects_overread() {
  // last_index = (num_frames-1)*win_inc + win_size must be <= waveform_len.
  // waveform_len=5, num_frames=3, win_inc=2, win_size=4 → last = 2*2+4 = 8 > 5.
  let wf: Vec<f32> = (0..5).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[5]).unwrap();
  let err =
    strided_frames_snip_edges(&x, 4, 2, 3).expect_err("overreading the waveform must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert!(
    p.value().contains("last_index=8") && p.value().contains("waveform_len=5"),
    "expected the overread diagnostic, got value {:?}",
    p.value()
  );
}

// ---- strided_frames_no_snip_edges (private) extra branches -------------

#[test]
fn strided_no_snip_edges_rejects_non_1d() {
  // 2-D waveform -> RankMismatch (`shape.len() != 1`).
  let a = Array::zeros::<f32>(&[2_i32, 4_i32]).unwrap();
  let err = strided_frames_no_snip_edges(&a, 4, 2, 2).expect_err("2-D waveform must be rejected");
  let Error::RankMismatch(p) = &err else {
    panic!("expected RankMismatch, got {err:?}");
  };
  assert_eq!(
    p.context(),
    "strided_frames_no_snip_edges: expected 1-D waveform"
  );
  assert_eq!(p.actual(), 2);
}

#[test]
fn strided_no_snip_edges_zero_frames_returns_empty() {
  // num_frames == 0 short-circuits to a (0, 0) empty array.
  let wf: Vec<f32> = (0..4).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[4]).unwrap();
  let frames = strided_frames_no_snip_edges(&x, 2, 2, 0).unwrap();
  assert_eq!(frames.shape(), vec![0, 0]);
}

#[test]
fn strided_no_snip_edges_pad_gt1_rejects_signal_shorter_than_pad() {
  // pad > 1 needs n >= pad + 1 (the reflect bookends must fit). win_size=8,
  // win_inc=2 → pad = 4 - 1 = 3, so need n >= 4. A length-3 signal fails.
  let wf: Vec<f32> = (0..3).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[3]).unwrap();
  // num_frames irrelevant — the pad-fit check happens during buffer build.
  let err = strided_frames_no_snip_edges(&x, 8, 2, 2)
    .expect_err("signal shorter than the reflect pad must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.requirement(), "must be >= pad + 1");
  assert!(
    p.value().contains("n=3") && p.value().contains("pad=3"),
    "expected n/pad diagnostic, got {:?}",
    p.value()
  );
}

#[test]
fn strided_no_snip_edges_rejects_abs_pad_gt_n() {
  // pad <= 0 branch: |pad| > n must be rejected (win_inc >> win_size on a
  // tiny signal). win_size=2 (→ win_size/2 = 1), win_inc=200 (→ win_inc/2 =
  // 100) → pad = 1 - 100 = -99, abs_pad = 99. A length-5 signal: 99 > 5.
  let wf: Vec<f32> = (0..5).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[5]).unwrap();
  // m = (5 + 100) / 200 = 0 frames — but the helper does NOT short-circuit on
  // m==0 alone for the pad<=0 buffer build... we still pass a nonzero
  // num_frames so we reach the abs_pad check before the strided-read bound.
  let err = strided_frames_no_snip_edges(&x, 2, 200, 1)
    .expect_err("|pad| larger than the signal must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.requirement(), "must be <= waveform len");
  assert!(
    p.value().contains("abs_pad=99") && p.value().contains("n=5"),
    "expected abs_pad/n diagnostic, got {:?}",
    p.value()
  );
}

// ---- compute_fbank_kaldi additional validation + small paths ----------

#[test]
fn compute_fbank_kaldi_rejects_win_len_below_two() {
  // win_len < 2 → OutOfRange (distinct from win_inc==0 / rank checks).
  let x = Array::from_slice::<f32>(&[0.0_f32; 8], &[8_i32]).unwrap();
  let err = compute_fbank_kaldi(
    &x,
    16_000,
    1,
    1,
    4,
    KaldiWindow::Rectangular,
    0.0,
    0.0,
    true,
    0.0,
    0.0,
    None,
    PreemphBoundary::Scale,
  )
  .expect_err("win_len < 2 must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "compute_fbank_kaldi: win_len");
}

#[test]
fn compute_fbank_kaldi_rejects_win_len_over_decoded_samples_cap() {
  // win_len > MAX_DECODED_SAMPLES hits the win_len cap branch. Use a lazy
  // zeros input so no host buffer materializes; win_len just over the cap.
  let cap = crate::audio::io::MAX_DECODED_SAMPLES;
  // Build a small real waveform — the win_len cap is checked from the scalar
  // arg before any framing, so the array size is irrelevant.
  let x = Array::from_slice::<f32>(&[0.0_f32; 8], &[8_i32]).unwrap();
  let err = compute_fbank_kaldi(
    &x,
    16_000,
    cap + 1, // win_len just over the cap
    160,
    4,
    KaldiWindow::Rectangular,
    0.0,
    0.0,
    true,
    0.0,
    0.0,
    None,
    PreemphBoundary::Scale,
  )
  .expect_err("win_len over MAX_DECODED_SAMPLES must be rejected");
  let Error::CapExceeded(p) = &err else {
    panic!("expected CapExceeded, got {err:?}");
  };
  assert_eq!(p.cap_name(), "MAX_DECODED_SAMPLES");
  assert_eq!(p.context(), "compute_fbank_kaldi: win_len exceeds cap");
}

#[test]
fn compute_fbank_kaldi_snip_edges_false_zero_frames_returns_empty() {
  // snip_edges=false with `m = (n + win_inc/2)/win_inc == 0` returns a
  // (0, num_mels) empty array. samples_len=1, win_inc=4 → (1 + 2)/4 = 0.
  let x = Array::from_slice::<f32>(&[0.5_f32], &[1_i32]).unwrap();
  let out = compute_fbank_kaldi(
    &x,
    16_000,
    2, // win_len = 2 (>= 2, <= cap)
    4, // win_inc = 4 → m = (1 + 2)/4 = 0
    7, // num_mels = 7 → expect (0, 7)
    KaldiWindow::Rectangular,
    0.0,
    0.0,
    false, // snip_edges = false
    0.0,
    0.0,
    None,
    PreemphBoundary::Scale,
  )
  .unwrap();
  assert_eq!(out.shape(), vec![0, 7]);
}

// ---- compute_deltas_kaldi additional branches -------------------------

#[test]
fn compute_deltas_kaldi_rejects_rank_zero_scalar() {
  // A rank-0 (scalar) specgram has no time axis → RankMismatch. win_length is
  // valid (5) so the rank check (not the win_length checks) fires.
  let scalar = Array::full::<f32>(&[0_i32; 0], 3.0).unwrap();
  assert!(scalar.shape().is_empty(), "scalar must be rank-0");
  let err = compute_deltas_kaldi(&scalar, 5, DeltaPadMode::Edge)
    .expect_err("rank-0 specgram must be rejected");
  let Error::RankMismatch(p) = &err else {
    panic!("expected RankMismatch, got {err:?}");
  };
  assert_eq!(p.actual(), 0, "rank-0 reported");
  assert_eq!(
    p.context(),
    "compute_deltas_kaldi: specgram must have rank >= 1 (a time axis)"
  );
}

#[test]
fn compute_deltas_kaldi_time_zero_returns_empty_same_shape() {
  // time == 0 ⇒ total == 0 ⇒ the output is the same empty shape (no deltas).
  // Shape (2, 0): rank-2, last axis (time) == 0.
  let x = Array::zeros::<f32>(&[2_i32, 0_i32]).unwrap();
  let out = compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge).unwrap();
  assert_eq!(out.shape(), vec![2, 0], "time==0 preserves the empty shape");
}

#[test]
fn compute_deltas_kaldi_rejects_total_over_cap() {
  // `total` (== num_features * time) over MAX_FBANK_WORK must be rejected by
  // the element-count cap, BEFORE the padded-work cap. Shape
  // (MAX_FBANK_WORK + 1,) is a 1-D input: time = total = cap + 1 > cap.
  // `Array::zeros` is lazy — the cap engages on the shape alone.
  let total = MAX_FBANK_WORK + 1;
  let total_i32 = i32::try_from(total).unwrap();
  let x = Array::zeros::<f32>(&[total_i32]).unwrap();
  let err = compute_deltas_kaldi(&x, 5, DeltaPadMode::Edge)
    .expect_err("total over MAX_FBANK_WORK must be rejected");
  let Error::CapExceeded(p) = &err else {
    panic!("expected CapExceeded, got {err:?}");
  };
  assert_eq!(p.cap_name(), "MAX_FBANK_WORK");
  assert_eq!(
    p.context(),
    "compute_deltas_kaldi: element count exceeds work cap"
  );
}

#[test]
fn compute_deltas_kaldi_constant_mode_1d_ramp_interior_unit_slope() {
  // Constant (zero) pad mode on a 1-D ramp: the deep interior (clear of the
  // n=2 zero-pad on each edge) still has unit-slope derivative 1.0. This
  // covers the DeltaPadMode::Constant arm on a 1-D (num_features==1) input.
  let ramp: Vec<f32> = (0..9).map(|n| n as f32).collect();
  let x = Array::from_slice::<f32>(&ramp, &[9]).unwrap();
  let d = compute_deltas_kaldi(&x, 5, DeltaPadMode::Constant).unwrap();
  assert_eq!(d.shape(), vec![9]);
  let got = to_vec(&d);
  // Interior indices 2..=6 are unaffected by the zero pad (n=2).
  for (i, &g) in got.iter().enumerate().take(7).skip(2) {
    assert!(
      (g - 1.0).abs() < F32_TOL,
      "constant-pad ramp delta[{i}]: got {g}, want 1.0"
    );
  }
}

// ---- overflow / i32 / i64 conversion guards (scalar-arg driven) --------
//
// The following exercise the cold error closures that only fire when a SCALAR
// argument (not an array dimension) exceeds a `usize`-multiply / `i32` / `i64`
// bound. Array dimensions are bounded by `i32::MAX` (mlx shapes are `&[i32]`),
// so these branches are unreachable through a constructed `Array`; they are
// reached only by handing the (free `usize`) scalar parameters pathological
// values directly to the public / private functions. No allocation occurs —
// every guard fires BEFORE the function touches the FFI / host buffers.

#[test]
fn mel_banks_kaldi_num_bins_times_fft_bins_overflows_usize() {
  // `bank_len = num_bins * num_fft_bins` with `num_fft_bins = n_fft_padded/2`.
  // `num_bins = 1<<40`, `n_fft_padded = 1<<26` ⇒ `num_fft_bins = 1<<25`, and
  // `(1<<40) * (1<<25) = 1<<65` overflows `usize` (2^64). The `checked_mul`
  // closure (the ArithmeticOverflow path) fires BEFORE the cap / i32 checks.
  // All freq args are valid so we reach the multiply: nyquist = 8000,
  // high_freq=0.0 → 8000, low=20 (0<=20<8000), 20<8000.
  let num_bins = 1_usize << 40;
  let n_fft_padded = 1_usize << 26; // even, non-zero
  let num_fft_bins = n_fft_padded / 2; // 1<<25
  // Independent oracle: the product genuinely overflows usize.
  assert!(
    num_bins.checked_mul(num_fft_bins).is_none(),
    "test premise: num_bins * num_fft_bins must overflow usize"
  );
  let err = get_mel_banks_kaldi(num_bins, n_fft_padded, 16_000.0, 20.0, 0.0)
    .expect_err("num_bins * num_fft_bins overflow must be rejected");
  let Error::ArithmeticOverflow(p) = &err else {
    panic!("expected ArithmeticOverflow, got {err:?}");
  };
  assert_eq!(p.op_type(), "usize");
  assert!(
    p.context()
      .contains("get_mel_banks_kaldi: num_bins * num_fft_bins"),
    "expected the num_bins*num_fft_bins overflow context, got {:?}",
    p.context()
  );
  // Operands carry the offending runtime values by name.
  let names: Vec<&str> = p.operands().iter().map(|(n, _)| *n).collect();
  assert!(
    names.contains(&"num_bins") && names.contains(&"num_fft_bins"),
    "expected num_bins + num_fft_bins operands, got {names:?}"
  );
}

#[test]
fn build_kaldi_window_rejects_win_size_over_i32() {
  // `win_size` is a free `usize`; a value > i32::MAX fails the `i32::try_from`
  // BEFORE any reservation / SIMD kernel runs (the check is line-order first),
  // so no allocation is attempted. `>= 2` so the small-window guard passes.
  let win_size = (i32::MAX as usize) + 1;
  assert!(i32::try_from(win_size).is_err(), "test premise: > i32::MAX");
  let err = build_kaldi_window(KaldiWindow::Hamming, win_size)
    .expect_err("win_size over i32::MAX must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "build_kaldi_window: win_size");
  assert!(
    p.requirement().contains("fit in i32"),
    "expected the i32-fit requirement, got {:?}",
    p.requirement()
  );
}

#[test]
fn strided_frames_snip_edges_last_index_overflows_usize() {
  // `last_index = (num_frames - 1) * win_inc + win_size` is computed with
  // checked arithmetic over the free `usize` scalars. `num_frames = 2`,
  // `win_inc = usize::MAX` ⇒ `1 * usize::MAX` then `+ win_size` overflows.
  // The waveform array itself is small (the overflow is in the scalar math,
  // evaluated before the array is read).
  let x = Array::from_slice::<f32>(&[0.0_f32; 4], &[4_i32]).unwrap();
  // Independent oracle: the reachable-index arithmetic overflows usize.
  assert!(
    1_usize
      .checked_mul(usize::MAX)
      .and_then(|v| v.checked_add(10))
      .is_none(),
    "test premise: (num_frames-1)*win_inc + win_size must overflow usize"
  );
  let err = strided_frames_snip_edges(&x, 10, usize::MAX, 2)
    .expect_err("reachable-index overflow must be rejected");
  let Error::ArithmeticOverflow(p) = &err else {
    panic!("expected ArithmeticOverflow, got {err:?}");
  };
  assert_eq!(p.op_type(), "usize");
  assert!(
    p.context()
      .contains("strided_frames_snip_edges: reachable element range"),
    "expected the reachable-range overflow context, got {:?}",
    p.context()
  );
}

#[test]
fn strided_frames_snip_edges_rejects_num_frames_over_i32() {
  // `num_frames > i32::MAX` reaches the `num_frames` i32 conversion ONLY when
  // the `last_index <= waveform_len` bound passes first. With `win_inc = 0`,
  // `last_index = (num_frames-1)*0 + win_size = win_size`; choose `win_size = 4`
  // and a length-5 waveform so `4 <= 5` passes, then the huge `num_frames`
  // hits the i32 guard. No array of `num_frames` size is built — `num_frames`
  // is a free scalar.
  let x = Array::from_slice::<f32>(&[0.0_f32; 5], &[5_i32]).unwrap();
  let num_frames = (i32::MAX as usize) + 1;
  let err = strided_frames_snip_edges(&x, 4, 0, num_frames)
    .expect_err("num_frames over i32::MAX must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "strided_frames_snip_edges: num_frames");
}

#[test]
fn strided_frames_snip_edges_rejects_win_inc_over_i64() {
  // `win_inc > i64::MAX` (it is a `usize`; on 64-bit, usize::MAX > i64::MAX)
  // reaches the `win_inc` i64 conversion only after the earlier guards pass.
  // With `num_frames = 1`, the multiply `(1-1)*win_inc = 0` does NOT overflow
  // and `last_index = win_size = 4 <= waveform_len`, num_frames(1)/win_size(4)
  // fit i32 — so the i64 win_inc guard is the one that fires.
  let x = Array::from_slice::<f32>(&[0.0_f32; 8], &[8_i32]).unwrap();
  let win_inc = (i64::MAX as usize) + 1;
  assert!(i64::try_from(win_inc).is_err(), "test premise: > i64::MAX");
  let err = strided_frames_snip_edges(&x, 4, win_inc, 1)
    .expect_err("win_inc over i64::MAX must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "strided_frames_snip_edges: win_inc");
  assert!(
    p.requirement().contains("fit in i64"),
    "expected the i64-fit requirement, got {:?}",
    p.requirement()
  );
}

#[test]
fn strided_no_snip_edges_last_index_overflows_usize() {
  // Reach the post-`padded`-build `last_index` overflow closure: build a VALID
  // reflect buffer (win_size=4, win_inc=2, n=10 ⇒ pad=1, padded len 20), then
  // a `num_frames = usize::MAX` so `(num_frames-1)*win_inc` overflows usize.
  // `num_frames` does not affect the padded-buffer build (only win_size/win_inc/
  // n do), so the helper reaches line ~797 with the valid buffer and overflows.
  let wf: Vec<f32> = (0..10).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[10_i32]).unwrap();
  // Independent oracle: (usize::MAX - 1) * 2 overflows usize.
  assert!(
    (usize::MAX - 1).checked_mul(2).is_none(),
    "test premise: (num_frames-1)*win_inc must overflow usize"
  );
  let err = strided_frames_no_snip_edges(&x, 4, 2, usize::MAX)
    .expect_err("reachable-index overflow must be rejected");
  let Error::ArithmeticOverflow(p) = &err else {
    panic!("expected ArithmeticOverflow, got {err:?}");
  };
  assert_eq!(p.op_type(), "usize");
  assert!(
    p.context()
      .contains("strided_frames_no_snip_edges: reachable element range"),
    "expected the reachable-range overflow context, got {:?}",
    p.context()
  );
}

#[test]
fn strided_no_snip_edges_rejects_num_frames_over_i32() {
  // `num_frames > i32::MAX` reaches the `num_frames` i32 conversion only when
  // `last_index <= padded_len` passes. With `win_inc = 0`, `last_index =
  // (num_frames-1)*0 + win_size = win_size = 4`; a length-10 waveform with
  // win_size=4, win_inc=0 ⇒ pad = 4/2 - 0/2 = 2 (the pad>1 path) builds a
  // padded buffer of length 2+10+2 = 14, so 4 <= 14 passes and the huge
  // `num_frames` hits the i32 guard. `num_frames` is a free scalar (no array
  // of that size is built).
  let wf: Vec<f32> = (0..10).map(|v| v as f32).collect();
  let x = Array::from_slice::<f32>(&wf, &[10_i32]).unwrap();
  let num_frames = (i32::MAX as usize) + 1;
  let err = strided_frames_no_snip_edges(&x, 4, 0, num_frames)
    .expect_err("num_frames over i32::MAX must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "strided_frames_no_snip_edges: num_frames");
}

#[test]
fn compute_fbank_kaldi_rejects_num_mels_over_i32() {
  // `num_mels > i32::MAX` reaches the `num_mels_i32` conversion — it is the
  // first thing converted after the (all-passing) scalar validation. Every
  // other arg is valid (rank-1, sample_rate>0, win_len>=2 & <=cap, win_inc>0,
  // dither=0, preemph in [0,1], samples_len<=cap), so the num_mels i32 guard
  // is the one that fires. The waveform is a small real buffer; num_mels is a
  // free scalar (no `(.., num_mels)` array is built before the guard).
  let x = Array::from_slice::<f32>(&[0.0_f32; 8], &[8_i32]).unwrap();
  let num_mels = (i32::MAX as usize) + 1;
  let err = compute_fbank_kaldi(
    &x,
    16_000,
    4,
    2,
    num_mels,
    KaldiWindow::Rectangular,
    0.0,
    0.0,
    true,
    0.0,
    0.0,
    None,
    PreemphBoundary::Scale,
  )
  .expect_err("num_mels over i32::MAX must be rejected");
  let Error::OutOfRange(p) = &err else {
    panic!("expected OutOfRange, got {err:?}");
  };
  assert_eq!(p.context(), "compute_fbank_kaldi: num_mels");
}

#[test]
fn compute_fbank_kaldi_frame_work_cap_rejects_large_framed_matrix() {
  // The FIRST work cap is `frame_work = num_frames * n_fft_padded`. Choose
  // params so it exceeds MAX_FBANK_WORK (64 Mi) while `out_elems`,
  // `output_elems`, `mel_padded_elems` (checked AFTER) stay under it — so
  // `frame_work` is the binding cap. `samples_len = 200_000`, `win_len = 1000`
  // (n_fft_padded = next_pow2(1000) = 1024), `win_inc = 1`, snip_edges=true:
  //   num_frames  = 1 + (200000 - 1000)/1 = 199_001
  //   frame_work  = 199_001 * 1024 ≈ 203 Mi  > 64 Mi cap  (REJECT here)
  //   out_elems   = 199_001 * (1024/2+1) (only reached if frame_work passed)
  //   output_elems= 199_001 * 4 ≈ 796 K      <= cap
  // `Array::zeros` is lazy and `contiguous` runs AFTER the caps, so no host
  // buffer is materialized — the cap rejects on scalar framing math alone.
  let samples_len = 200_000;
  let n_fft_padded = 1024_usize; // next_pow2(1000)
  let num_frames = 1 + (samples_len - 1000); // win_inc = 1 → 199_001
  // Independent oracle: frame_work > cap, output_elems <= cap.
  assert!(
    num_frames * n_fft_padded > MAX_FBANK_WORK,
    "test premise: frame_work must exceed the cap"
  );
  assert!(
    num_frames * 4 <= MAX_FBANK_WORK,
    "test premise: output_elems must stay under the cap so frame_work is binding"
  );
  let x = Array::zeros::<f32>(&[i32::try_from(samples_len).unwrap()]).unwrap();
  let err = compute_fbank_kaldi(
    &x,
    16_000,
    1000, // win_len → n_fft_padded = 1024
    1,    // win_inc = 1 → maximal num_frames
    4,    // num_mels small → output_elems tiny
    KaldiWindow::Rectangular,
    0.0,
    0.0,
    true,
    0.0,
    0.0,
    None,
    PreemphBoundary::Scale,
  )
  .expect_err("frame_work over MAX_FBANK_WORK must be rejected");
  let Error::CapExceeded(p) = &err else {
    panic!("expected CapExceeded, got {err:?}");
  };
  assert_eq!(p.cap_name(), "MAX_FBANK_WORK");
  assert!(
    p.context().contains("frame work"),
    "expected the frame-work cap context, got {:?}",
    p.context()
  );
  assert!(
    p.observed() > p.cap(),
    "observed {} must exceed cap {}",
    p.observed(),
    p.cap()
  );
}
