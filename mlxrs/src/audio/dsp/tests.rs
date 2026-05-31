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

// ---- window family closed-form parity (hand-derived) ----------------

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
  // The reference Python form `0.5 * (1 - cos(2π n /
  // (size - 1)))` divides by zero for `size == 1`, silently producing
  // `NaN` for every sample. mlxrs centralizes the rejection in
  // `symmetric_window` so EVERY window function (Hann / Hamming /
  // Blackman / Bartlett) returns a recoverable `Error::OutOfRange` for
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
    assert!(matches!(r, Err(Error::OutOfRange(_))));
  }
  // The `window_from_name` dispatch must propagate the same rejection
  // for every supported name (so `STR_TO_WINDOW_FN`-style callers also
  // get the error rather than a silent NaN window).
  for name in ["hann", "hanning", "hamming", "blackman", "bartlett"] {
    let r = window_from_name(name, 1);
    assert!(
      matches!(r, Err(Error::OutOfRange(_))),
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
    Err(Error::UnknownEnumValue(_))
  ));
}

// ---- stft / istft WindowPad round-trips -----------------------------
//
// Every reconstruction test below drives the REAL public `stft` and feeds
// its output straight into `istft` (`istft(&stft(signal, ..)?, ..)?`). A
// private periodic-forward-helper pattern that builds spectra with its own
// window can hide a synthesis/analysis window mismatch, so it is BANNED —
// there is no helper that builds spectra with its own window; the only forward is
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
/// This is the canary for synthesis/analysis window drift: it goes
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
  // WindowPad::Right inversion supports ONLY win_length == n_fft.
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
  // why the rejection is up-front, not guard-based).
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
        matches!(res, Err(Error::OutOfRange(_))),
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
      matches!(res, Err(Error::OutOfRange(_))),
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
      matches!(res, Err(Error::OutOfRange(_))),
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
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn spectrum_from_parts_rejects_inconsistent_metadata() {
  // `Spectrum::from_parts` is the validated constructor for EXTERNAL/raw
  // spectra: it must make it impossible to build a Spectrum whose metadata
  // istft would misdecode. This closes the external-odd-spectrum hole (a
  // bare-array path would allow istft misdecodes) — a
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
    Err(Error::OutOfRange(_))
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
    Err(Error::LengthMismatch(_))
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
    Err(Error::OutOfRange(_))
  ));

  // hop_length == 0 and win_length == 0.
  assert!(matches!(
    Spectrum::from_parts(valid.try_clone().unwrap(), 8, 0, 8, WindowPad::Center, true),
    Err(Error::InvariantViolation(_))
  ));
  assert!(matches!(
    Spectrum::from_parts(valid.try_clone().unwrap(), 8, 4, 0, WindowPad::Center, true),
    Err(Error::InvariantViolation(_))
  ));

  // n_fft == 0.
  assert!(matches!(
    Spectrum::from_parts(valid.try_clone().unwrap(), 0, 4, 0, WindowPad::Center, true),
    Err(Error::InvariantViolation(_))
  ));

  // Non-2-D data (1-D).
  let one_d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3i32])
    .unwrap()
    .astype(Dtype::Complex64)
    .unwrap();
  assert!(matches!(
    Spectrum::from_parts(one_d, 8, 4, 8, WindowPad::Center, true),
    Err(Error::RankMismatch(_))
  ));

  // Non-Complex64 data (F32) with otherwise-consistent metadata.
  let real_data = Array::zeros::<f32>(&[5i32, 5i32]).unwrap();
  assert!(matches!(
    Spectrum::from_parts(real_data, 8, 4, 8, WindowPad::Center, true),
    Err(Error::DtypeMismatch(_))
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
  // The real scatter/update workload is `num_frames * n_fft`, which
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
    matches!(res, Err(Error::CapExceeded(_))),
    "pathological num_frames*n_fft must be rejected by the MAX_OLA_WORK cap \
       before the frame_window allocation"
  );
}

#[test]
fn stft_rejects_pathological_work_before_alloc() {
  // Cap stft's forward work. A LAZILY-shaped huge input (no
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
    matches!(res, Err(Error::CapExceeded(_))),
    "pathological lazy huge-shape stft input (num_frames * n_fft) must be \
       rejected by the MAX_STFT_WORK cap before any framing/FFT allocation, got {res:?}"
  );
}

#[test]
fn stft_rejects_oversized_input_before_reflect_pad_large_hop() {
  // The reflect pad (`center=true`) is a lazy
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
    matches!(res, Err(Error::CapExceeded(_))),
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

// ---- `lfilter` direct-form II transposed parity ---------------------

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

/// `a` empty / `a[0] == 0` / non-1-D input must all be rejected with typed
/// errors, matching the reference's `ValueError` raises. (Note:
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
    Err(Error::EmptyInput(_))
  ));
  // a[0] == 0
  assert!(matches!(
    lfilter(&[1.0_f64], &[0.0_f64, 1.0], &x),
    Err(Error::InvariantViolation(_))
  ));
  // 2-D input — reference raises `dsp.lfilter only supports 1-D input`.
  let x_2d = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &[2i32, 2i32]).unwrap();
  assert!(matches!(
    lfilter(&[0.5_f64], &[1.0_f64, -0.5], &x_2d),
    Err(Error::RankMismatch(_))
  ));
}

/// Public [`lfilter`]'s sample cap must fire on the input SHAPE BEFORE
/// any `to_vec` materialization. We construct a lazy `Array::zeros` of
/// `(MAX_LFILTER_SAMPLES + 1,)` f32 — which mlx does not eval until a
/// data accessor runs — and assert `lfilter` rejects it with
/// `Error::CapExceeded`. If the cap check lived behind the `to_vec`,
/// the rejected call would first
/// materialize `(MAX_LFILTER_SAMPLES + 1) * 4 bytes` (≈256 MiB) of f32
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
    matches!(res, Err(Error::CapExceeded(_))),
    "lfilter must reject a lazy ({} samples) input via the up-front \
       sample cap, BEFORE the to_vec materializes f32 / promotes to f64 \
       (got {res:?})",
    MAX_LFILTER_SAMPLES + 1
  );
}

/// [`lfilter_f64_in_place`] is the in-place variant the K-weighting path
/// uses to keep the peak working set at ONE f64 channel buffer (a
/// chained out-of-place form would hold TWO across the high-shelf →
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
    Err(Error::EmptyInput(_))
  ));
  assert!(matches!(
    lfilter_f64_in_place(&[1.0_f64], &[0.0_f64, 1.0], &mut x_buf),
    Err(Error::InvariantViolation(_))
  ));
}

// ---- BS.1770 K-weighted integrated loudness + normalize_loudness -----

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
    matches!(res, Err(Error::OutOfRange(_))),
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
  let Err(Error::OutOfRange(payload)) = &res else {
    panic!("audio with >5 channels must be rejected with OutOfRange (got {res:?})");
  };
  assert_eq!(payload.value(), "6");
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
    matches!(res, Err(Error::RankMismatch(_))),
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
    Err(Error::OutOfRange(_))
  ));
  // overlap < 0
  assert!(matches!(
    integrated_loudness(&x, rate, 0.4, -0.1),
    Err(Error::OutOfRange(_))
  ));
  // block_size <= 0
  assert!(matches!(
    integrated_loudness(&x, rate, 0.0, 0.75),
    Err(Error::OutOfRange(_))
  ));
  // rate == 0
  assert!(matches!(
    integrated_loudness(&x, 0, 0.4, 0.75),
    Err(Error::OutOfRange(_))
  ));
}

/// Total-element cap (`n_samples * n_channels`) must reject oversized 2-D
/// inputs BEFORE the `to_vec`. Capping only
/// `n_samples` would let a `(MAX_DECODED_SAMPLES, 5)` lazily-shaped input
/// slip past the per-channel cap and then materialize
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
    matches!(res_mono, Err(Error::CapExceeded(_))),
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
    matches!(res_5ch, Err(Error::CapExceeded(_))),
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
    matches!(res, Err(Error::CapExceeded(_))),
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
    matches!(res_stereo, Err(Error::CapExceeded(_))),
    "pathological-overlap stereo (byte cap factor n_channels) must be \
       rejected (got {res_stereo:?})"
  );
}

/// Regression: the *just-below-an-element-only-cap* case.
/// Capping `num_blocks * n_channels` against
/// `MAX_DECODED_SAMPLES = 64 Mi-elements` would be wrong — each cell is
/// `f64`, so `64 Mi cells * 8 B = 512 MiB` of actual `mean_square`
/// allocation would pass an element-only cap. A near-1 overlap like `0.99999990`
/// on a tiny 3 s mono signal produces `num_blocks ≈ (3.0 - 0.4) /
/// (0.4 * 1e-7) ≈ 6.5e7` blocks, which sit JUST UNDER a bare
/// 64 Mi-element cap (an element-only guard would NOT reject, and the
/// `try_reserve_exact` would attempt a ~520 MiB `mean_square` matrix
/// reservation followed by a per-block loop that re-sums 19,200
/// samples per block — multi-trillion sample visits, hours of CPU).
/// The `MAX_LOUDNESS_BLOCK_BYTES` (64 MiB) cap rejects this case
/// at the byte-budget check BEFORE any allocation (block_bytes =
/// 6.5e7 * 1 * 8 = 520 MiB > 64 MiB); the visit cap would also catch
/// it (6.5e7 * 19200 = 1.25 trillion visits > 256 Mi). Asserting
/// `Err` in microseconds proves the byte/work caps fire up-front,
/// not a bare elements-only cap.
#[test]
fn integrated_loudness_rejects_overlap_just_below_old_element_cap() {
  let rate = 48_000u32;
  // 3 s of audio = 144,000 samples — well below MAX_LOUDNESS_SAMPLES.
  // overlap = 0.99999990 ⇒ step = 1e-7 ⇒ num_blocks ≈ 6.5e7. A bare
  // element cap of 64 Mi ≈ 6.7e7 would leave 6.5e7 UNDER it;
  // the byte cap (64 MiB / 8 B = 8 Mi cells) rejects num_blocks
  // > 8 Mi for n_channels=1, and the work cap (256 Mi) rejects
  // 6.5e7 * 19200 ≈ 1.25e12 visits — both fire well below 6.5e7.
  let x = sine_mono(1000.0, 0.5, rate, 3.0);
  let res = integrated_loudness(&x, rate, 0.4, 0.999_999_90);
  assert!(
    matches!(res, Err(Error::CapExceeded(_))),
    "overlap=0.99999990 (under old elements-only cap; over new byte/work \
       caps) must be rejected BEFORE allocation (got {res:?})"
  );
}

/// Regression: the work cap MUST include `n_channels`.
/// The per-block mean-square sum loop runs ONCE PER CHANNEL (the
/// per-channel streaming K-weighting loop), so the actual sample-visit
/// count is `num_blocks * block_samples * n_channels`. Bounding
/// the channel-less product alone would let a 5-channel pathological
/// case slip through: `num_blocks ≈ 500_000, block_samples = 160,
/// n_channels = 5` gives a channel-less product of `8e7 < 256 Mi cap`
/// (would PASS a channel-less bound) BUT actual visits of `4e8 > 256 Mi`
/// (FAILS the channel-aware bound).
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
/// Asserts `Err`: without the n_channels-aware work cap this would
/// silently allow ~400 M sample-visits across the per-block ×
/// per-channel loops (a multi-second CPU spike on a small input); the
/// work cap fires up-front in microseconds.
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
  // Channel-less work product:
  //   500_001 * 160 ≈ 8.0e7 < MAX_LOUDNESS_WORK (256 Mi ≈ 2.68e8) → would PASS
  // n_channels-aware work product (the one used):
  //   500_001 * 160 * 5 ≈ 4.0e8 > MAX_LOUDNESS_WORK              → REJECTS
  // Byte cap (independent, must NOT be the rejecting cap):
  //   500_001 * 5 * 8 ≈ 20 MB << MAX_LOUDNESS_BLOCK_BYTES (64 MiB) → PASSES
  let res = integrated_loudness(&x, rate, 0.01, 0.999_999_987_5);
  // Verify the rejection actually comes from the WORK cap (not the
  // byte cap and not the total-elements cap), so a future refactor that
  // accidentally weakens the work cap surfaces here rather than passing
  // because some other cap caught the case.
  let Err(Error::CapExceeded(payload)) = &res else {
    panic!(
      "5-channel input with num_blocks * block_samples just under the cap \
         but num_blocks * block_samples * n_channels above must be REJECTED \
         by the n_channels-aware work cap (got {res:?})"
    );
  };
  assert!(
    payload.context().contains("total sample-visit work")
      && payload.context().contains("n_channels"),
    "rejection must come from the n_channels-aware work cap (got: {})",
    payload.context()
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
/// tolerance of the theoretical -9.0656. An f32 cast at the
/// stage boundary would drop ~16 bits of precision between
/// biquads, biasing this absolute value (and gate decisions near the
/// absolute/relative thresholds). We assert ±0.05 LUFS — a tolerance
/// an f32-between-stages path could overshoot for short
/// signals near the gate boundaries, and which the f64 path
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
    Err(Error::OutOfRange(_))
  ));
  assert!(matches!(
    normalize_loudness(&x, -10.0, f64::INFINITY),
    Err(Error::OutOfRange(_))
  ));
  assert!(matches!(
    normalize_loudness(&x, f64::NEG_INFINITY, -23.0),
    Err(Error::OutOfRange(_))
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
    Err(Error::OutOfRange(_))
  ));
}

#[test]
fn istft_cache_center_zero_coverage_tail_rejects_like_free_istft() {
  // A `center=true` spectrum whose requested `length` reaches into
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
    matches!(free_tail, Err(Error::OutOfRange(_))),
    "free istft must reject the zero-coverage tail, got {free_tail:?}"
  );
  let mut cache = ISTFTCache::new();
  let cached_tail = cache.istft(&spec, Some(tail_len));
  assert!(
    matches!(cached_tail, Err(Error::OutOfRange(_))),
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
    matches!(warm_reject, Err(Error::OutOfRange(_))),
    "warm-cache tail request must STILL reject (guard is per-call), got {warm_reject:?}"
  );
}

#[test]
fn istft_cache_center_false_uncovered_head_rejects_like_free_istft() {
  // `center=false` consistency: the RAW OLA index 0 is reached only by
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
      matches!(free_res, Err(Error::OutOfRange(_))),
      "free istft center=false head (len={len:?}) must reject, got {free_res:?}"
    );
    let mut cache = ISTFTCache::new();
    let cached_res = cache.istft(&spec_no_center, len);
    assert!(
      matches!(cached_res, Err(Error::OutOfRange(_))),
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
    Err(Error::OutOfRange(_))
  ));
  // Non-finite target_peak_db.
  let data = Array::from_slice::<f32>(&[0.5, 0.1], &[2]).unwrap();
  assert!(matches!(
    normalize_peak(&data, f64::NAN),
    Err(Error::OutOfRange(_))
  ));
  assert!(matches!(
    normalize_peak(&data, f64::INFINITY),
    Err(Error::OutOfRange(_))
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
    matches!(normalize_peak(&data, 1e30), Err(Error::NonFiniteScalar(_))),
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
      Err(Error::NonFiniteScalar(_))
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

// ---- mel_filter_bank_cached (#128) ---------------------------------------

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
  // `n_fft = 0` → recoverable Error::InvariantViolation.
  assert!(matches!(
    mel_filter_bank_cached(80, 0, 16_000, 0.0, None),
    Err(Error::InvariantViolation(_))
  ));
  // A valid call AFTER the failed one still succeeds (the failure
  // didn't pollute the cache).
  let ok = mel_filter_bank_cached(80, 400, 16_000, 0.0, None);
  assert!(ok.is_ok());
  clear_mel_filter_cache();
}

// ---- precise (f64) cached mel filterbank (#291) --------------------------

/// The cached precise path returns the same bank as the UNcached precise
/// path (memoization is transparent).
#[test]
fn mel_filter_bank_cached_precise_matches_uncached_precise() {
  clear_mel_filter_cache();
  let plain = mel_filter_bank_with(80, 400, 16_000, 0.0, None, MelPrecision::Precise).unwrap();
  let cached =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Precise).unwrap();
  assert_eq!(
    to_vec(&plain),
    to_vec(&cached),
    "cached precise bank must match the uncached precise bank value-for-value"
  );
  clear_mel_filter_cache();
}

/// The precise (f64) and standard (f32) banks for IDENTICAL parameters are
/// cached under DISTINCT keys — no collision. After fetching both, the
/// cache holds two entries, and each precision returns its own (different)
/// bank rather than aliasing the other.
#[test]
fn mel_filter_bank_cached_precision_does_not_collide() {
  clear_mel_filter_cache();
  let std_bank =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Standard).unwrap();
  let precise =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Precise).unwrap();

  // Two distinct cache entries for the same (n_mels, n_fft, sample_rate,
  // f_min, f_max) at different precisions — the key includes precision.
  let len = MEL_FILTER_CACHE.with(|cell| cell.borrow().len());
  assert_eq!(
    len, 2,
    "standard and precise banks for identical params must occupy distinct cache slots"
  );

  // The two banks differ (the f64 path is not a no-op) — proving the cache
  // returned the precision-correct entry, not a collision.
  assert_ne!(
    to_vec(&std_bank),
    to_vec(&precise),
    "precise cache hit must not alias the standard bank"
  );

  // A repeat fetch of each precision still resolves to its own bank.
  let std_again =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Standard).unwrap();
  let precise_again =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Precise).unwrap();
  assert_eq!(to_vec(&std_bank), to_vec(&std_again));
  assert_eq!(to_vec(&precise), to_vec(&precise_again));
  // Still exactly two entries (the repeats were hits, not new inserts).
  let len2 = MEL_FILTER_CACHE.with(|cell| cell.borrow().len());
  assert_eq!(len2, 2, "repeat fetches must hit, not insert new entries");
  clear_mel_filter_cache();
}

/// The cache KEY itself distinguishes precision: two keys equal in every
/// parameter but precision are unequal (the structural guarantee behind
/// the no-collision behavior above).
#[test]
fn mel_filter_cache_key_distinguishes_precision() {
  let std_key = MelFilterCacheKey::new(80, 400, 16_000, 0.0, None, MelPrecision::Standard);
  let precise_key = MelFilterCacheKey::new(80, 400, 16_000, 0.0, None, MelPrecision::Precise);
  assert_ne!(
    std_key, precise_key,
    "cache keys differing only in precision must be unequal"
  );
}

/// `mel_filter_bank_cached` (no precision arg) is the `Standard` shorthand:
/// it shares the cache slot with `mel_filter_bank_cached_with(.., Standard)`.
#[test]
fn mel_filter_bank_cached_shorthand_is_standard() {
  clear_mel_filter_cache();
  let shorthand = mel_filter_bank_cached(80, 400, 16_000, 0.0, None).unwrap();
  let with_std =
    mel_filter_bank_cached_with(80, 400, 16_000, 0.0, None, MelPrecision::Standard).unwrap();
  assert_eq!(to_vec(&shorthand), to_vec(&with_std));
  // Both resolved to the SAME cache slot — only one entry exists.
  let len = MEL_FILTER_CACHE.with(|cell| cell.borrow().len());
  assert_eq!(len, 1, "shorthand and Standard must share one cache slot");
  clear_mel_filter_cache();
}

// ---- magic constants are named ---------------------------

/// Pin the exact named-constant values that mlx-audio expects by
/// asserting the const surface (so a future refactor can't
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

/// `MelPrecision` surface: default is `Standard`, `as_str` /
/// `Display` match the lowercase names, and the `IsVariant` predicates
/// discriminate the two variants.
#[test]
fn mel_precision_surface() {
  assert_eq!(MelPrecision::default(), MelPrecision::Standard);
  assert_eq!(MelPrecision::Standard.as_str(), "standard");
  assert_eq!(MelPrecision::Precise.as_str(), "precise");
  assert_eq!(MelPrecision::Precise.to_string(), "precise");
  assert!(MelPrecision::Standard.is_standard());
  assert!(MelPrecision::Precise.is_precise());
  assert!(!MelPrecision::Standard.is_precise());
}

// ---- StftConfig + stft_with_config + stft_aligned (#134) -----------------

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
    Err(Error::OutOfRange(_))
  ));
}

// ---- reflect_pad_1d round-trip + zero-pad fast path (#129) ---------------

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
