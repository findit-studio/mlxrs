use super::{
  KaldiWindowKind, SymWindowKind, kaldi_window, kaldi_window_scalar, symmetric_window,
  symmetric_window_scalar,
};
use crate::simd::diff::assert_close_slice_over_lane_sweep;

/// Adapter for the slice-tolerance helper: input slice carries `n`
/// (the window size) as `xs.len()`; the kind is captured by the
/// closure that wraps the call.
#[allow(clippy::type_complexity)]
fn make_pair_sym(
  kind: SymWindowKind,
) -> (impl Fn(&[i32]) -> Vec<f64>, impl Fn(&[i32]) -> Vec<f64>) {
  let s = move |xs: &[i32]| {
    let n = xs.len();
    if n < 2 {
      return Vec::new();
    }
    symmetric_window_scalar(kind, n)
      .expect("test-sized window should not OOM")
      .into_iter()
      .map(|x| x as f64)
      .collect()
  };
  let d = move |xs: &[i32]| {
    let n = xs.len();
    if n < 2 {
      return Vec::new();
    }
    symmetric_window(kind, n)
      .expect("test-sized window should not OOM")
      .into_iter()
      .map(|x| x as f64)
      .collect()
  };
  (s, d)
}

#[allow(clippy::type_complexity)]
fn make_pair_kaldi(
  kind: KaldiWindowKind,
) -> (impl Fn(&[i32]) -> Vec<f64>, impl Fn(&[i32]) -> Vec<f64>) {
  let s = move |xs: &[i32]| {
    let n = xs.len();
    if n < 2 {
      return Vec::new();
    }
    kaldi_window_scalar(kind, n)
      .expect("test-sized window should not OOM")
      .into_iter()
      .map(|x| x as f64)
      .collect()
  };
  let d = move |xs: &[i32]| {
    let n = xs.len();
    if n < 2 {
      return Vec::new();
    }
    kaldi_window(kind, n)
      .expect("test-sized window should not OOM")
      .into_iter()
      .map(|x| x as f64)
      .collect()
  };
  (s, d)
}

/// Polynomial cos approximation tolerance: ~3 ULP worst case per
/// cos eval, scaled by the per-formula multiplier (max 0.5 for
/// Hann, 0.46 for Hamming, 0.58 for Blackman) — ≪ 1e-6 absolute
/// for f32. The window outputs are bounded scales, so 1e-6 abs +
/// 1e-5 rel covers all kinds.
const WINDOW_TOL_ABS: f64 = 1e-5;
const WINDOW_TOL_REL: f64 = 1e-5;

#[test]
fn symmetric_window_hann_scalar_matches_dispatcher() {
  let (s, d) = make_pair_sym(SymWindowKind::Hann);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], WINDOW_TOL_ABS, WINDOW_TOL_REL);
}

#[test]
fn symmetric_window_hamming_scalar_matches_dispatcher() {
  let (s, d) = make_pair_sym(SymWindowKind::Hamming);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], WINDOW_TOL_ABS, WINDOW_TOL_REL);
}

#[test]
fn symmetric_window_blackman_scalar_matches_dispatcher() {
  let (s, d) = make_pair_sym(SymWindowKind::Blackman);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], WINDOW_TOL_ABS, WINDOW_TOL_REL);
}

#[test]
fn symmetric_window_bartlett_scalar_matches_dispatcher() {
  // Bartlett is exact (no cos approx), but uses the same helper for
  // shape consistency.
  let (s, d) = make_pair_sym(SymWindowKind::Bartlett);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], 1e-7, 1e-7);
}

#[test]
fn kaldi_window_hamming_scalar_matches_dispatcher() {
  let (s, d) = make_pair_kaldi(KaldiWindowKind::Hamming);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], WINDOW_TOL_ABS, WINDOW_TOL_REL);
}

#[test]
fn kaldi_window_hanning_scalar_matches_dispatcher() {
  let (s, d) = make_pair_kaldi(KaldiWindowKind::Hanning);
  assert_close_slice_over_lane_sweep(4, s, d, |n| vec![0_i32; n], WINDOW_TOL_ABS, WINDOW_TOL_REL);
}

#[test]
fn kaldi_window_rectangular_scalar_matches_dispatcher_exact() {
  // Rectangular is constant 1.0 — bit-exact match.
  let s = kaldi_window_scalar(KaldiWindowKind::Rectangular, 17).expect("17 should not OOM");
  let d = kaldi_window(KaldiWindowKind::Rectangular, 17).expect("17 should not OOM");
  assert_eq!(s, d);
  assert!(s.iter().all(|&x| x == 1.0));
}

/// Specific-value pin: Hann at endpoints is 0.0; at the midpoint is
/// 1.0 (for odd `n`). Hamming at endpoints is 0.08 (= 0.54 - 0.46).
#[test]
fn symmetric_window_endpoint_pins() {
  let n = 17;
  let hann = symmetric_window(SymWindowKind::Hann, n).expect("n=17 should not OOM");
  assert!(
    hann[0].abs() < 1e-5,
    "Hann start should be ~0 (got {})",
    hann[0]
  );
  assert!(
    hann[n - 1].abs() < 1e-5,
    "Hann end should be ~0 (got {})",
    hann[n - 1]
  );
  assert!(
    (hann[(n - 1) / 2] - 1.0).abs() < 1e-5,
    "Hann mid should be ~1 (got {})",
    hann[(n - 1) / 2]
  );

  let ham = symmetric_window(SymWindowKind::Hamming, n).expect("n=17 should not OOM");
  assert!(
    (ham[0] - 0.08).abs() < 1e-5,
    "Hamming start should be ~0.08 (got {})",
    ham[0]
  );
  assert!(
    (ham[n - 1] - 0.08).abs() < 1e-5,
    "Hamming end should be ~0.08 (got {})",
    ham[n - 1]
  );
}

/// Length pin: every window is exactly `n` samples.
#[test]
fn window_length_pins() {
  for n in [2_usize, 3, 4, 8, 16, 17, 100, 1024] {
    assert_eq!(
      symmetric_window(SymWindowKind::Hann, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      symmetric_window(SymWindowKind::Hamming, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      symmetric_window(SymWindowKind::Blackman, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      symmetric_window(SymWindowKind::Bartlett, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      kaldi_window(KaldiWindowKind::Hamming, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      kaldi_window(KaldiWindowKind::Hanning, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
    assert_eq!(
      kaldi_window(KaldiWindowKind::Rectangular, n)
        .expect("small n should not OOM")
        .len(),
      n
    );
  }
}

/// Pre-cond panic — `n < 2`.
#[test]
#[should_panic(expected = "simd::audio::window::symmetric_window: n must be >= 2")]
fn symmetric_window_panics_on_n_lt_2() {
  let _ = symmetric_window(SymWindowKind::Hann, 1);
}

#[test]
#[should_panic(expected = "simd::audio::window::kaldi_window: n must be >= 2")]
fn kaldi_window_panics_on_n_lt_2() {
  let _ = kaldi_window(KaldiWindowKind::Hamming, 1);
}

/// Wrap-arith / OOM defence: requesting a window with `n` equal to
/// `usize::MAX` must fail with [`Error::OutOfMemory`] (the
/// fallible `try_reserve_exact` rejects the request) rather than
/// abort the process via `Vec::with_capacity`.
///
/// `usize::MAX` × 4 bytes (the per-element size of f32) is ~16 EiB
/// — far above any host allocator's ceiling. The dispatcher's
/// `try_reserve_exact(n)` is now the SOLE allocation site (the NEON
/// arm's transient full-size `Vec<f32>` in the Bartlett path +
/// cos-tile tail has been removed).
#[test]
fn symmetric_window_returns_err_on_extreme_size() {
  let r = symmetric_window(SymWindowKind::Hann, usize::MAX);
  assert!(
    matches!(r, Err(super::Error::OutOfMemory)),
    "symmetric_window(Hann, usize::MAX) must return Err(OutOfMemory), got {r:?}"
  );
  // Cover Blackman + Bartlett arms too — the dispatcher's
  // pre-kernel `try_reserve_exact(n)` is the gate for all kinds.
  let r = symmetric_window(SymWindowKind::Blackman, usize::MAX);
  assert!(matches!(r, Err(super::Error::OutOfMemory)));
  let r = symmetric_window(SymWindowKind::Bartlett, usize::MAX);
  assert!(matches!(r, Err(super::Error::OutOfMemory)));
}

/// Same as [`symmetric_window_returns_err_on_extreme_size`] but for
/// the Kaldi window dispatcher. Hamming + Hanning go through the
/// cos-tile path; Rectangular fills via `slot.write(1.0)` directly
/// from the spare capacity of the pre-allocated `Vec` (so the
/// `try_reserve_exact` is the gate).
#[test]
fn kaldi_window_returns_err_on_extreme_size() {
  for kind in [
    KaldiWindowKind::Hamming,
    KaldiWindowKind::Hanning,
    KaldiWindowKind::Rectangular,
  ] {
    let r = kaldi_window(kind, usize::MAX);
    assert!(
      matches!(r, Err(super::Error::OutOfMemory)),
      "kaldi_window({kind:?}, usize::MAX) must return Err(OutOfMemory), got {r:?}"
    );
  }
}

/// NEON arm of `symmetric_window`
/// MUST write tail samples (`n % 4` ≤ 3) DIRECTLY into the
/// pre-reserved output, not through a transient
/// `symmetric_window_scalar(kind, n)` call that allocates a second
/// full-size `Vec<f32>` (which would violate the dispatcher's
/// pre-kernel allocation gate and could OOM for adversarial `n`
/// that the one-buffer path would have succeeded on).
///
/// Structural assertion: we cannot deterministically synthesize a
/// host-specific allocation pressure that triggers the transient
/// OOM while the one-buffer path succeeds, but we CAN verify the
/// observable contract — for every `n` with a non-zero tail (`n %
/// 4 != 0`), the dispatcher output must match the scalar reference
/// element-by-element under the polynomial-cos tolerance. This
/// exercises the per-sample tail formula.
#[test]
fn symmetric_window_neon_does_not_double_allocate_on_non_multiple_of_4_tail() {
  // n = 5 → body_len = 4, tail = 1 sample.
  // n = 7 → body_len = 4, tail = 3 samples.
  // n = 9 → body_len = 8, tail = 1 sample.
  // n = 100 mod 4 == 0 → no tail (control case).
  for n in [5_usize, 7, 9, 100] {
    for kind in [
      SymWindowKind::Hann,
      SymWindowKind::Hamming,
      SymWindowKind::Blackman,
    ] {
      let s = symmetric_window_scalar(kind, n).expect("test-sized window should not OOM");
      let d = symmetric_window(kind, n).expect("test-sized window should not OOM");
      assert_eq!(s.len(), n, "scalar length pin (kind={kind:?}, n={n})");
      assert_eq!(d.len(), n, "dispatcher length pin (kind={kind:?}, n={n})");
      for (i, (a, b)) in s.iter().zip(d.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = 1e-5_f32.max(1e-5_f32 * a.abs());
        assert!(
          diff <= tol,
          "tail-handling mismatch at kind={kind:?} n={n} i={i}: \
             scalar={a} dispatcher={b} diff={diff} tol={tol}"
        );
      }
    }
  }
}

/// Bartlett-specific: the NEON
/// arm's Bartlett path MUST write its linear ramp directly into
/// the pre-reserved output, NOT delegate to
/// `symmetric_window_scalar(SymWindowKind::Bartlett, n)` (which
/// allocates a transient full-size `Vec<f32>` and would violate
/// the dispatcher's pre-kernel allocation gate).
///
/// Observable assertion: dispatcher matches scalar Bartlett
/// formula `1 - 2*|k - (n-1)/2| / (n-1)` exactly (no cos
/// approximation involved — Bartlett uses a closed-form linear
/// ramp on both paths).
#[test]
fn symmetric_window_bartlett_neon_writes_directly_to_out() {
  for n in [5_usize, 7, 8, 9, 100, 1024] {
    let s = symmetric_window_scalar(SymWindowKind::Bartlett, n)
      .expect("test-sized Bartlett should not OOM");
    let d =
      symmetric_window(SymWindowKind::Bartlett, n).expect("test-sized Bartlett should not OOM");
    assert_eq!(s.len(), n);
    assert_eq!(d.len(), n);
    // Bartlett has no cos approximation — must be bit-exact.
    for (i, (a, b)) in s.iter().zip(d.iter()).enumerate() {
      assert_eq!(
        a, b,
        "Bartlett scalar/dispatcher must match exactly at n={n} i={i}: \
           scalar={a} dispatcher={b}"
      );
    }
  }
}

/// Kaldi cos-kinds tail: mirrors
/// [`symmetric_window_neon_does_not_double_allocate_on_non_multiple_of_4_tail`]
/// for the Kaldi dispatcher. Hamming + Hanning go through the
/// cos-tile body + per-sample tail.
#[test]
fn kaldi_window_neon_does_not_double_allocate_on_non_multiple_of_4_tail() {
  for n in [5_usize, 7, 9, 100] {
    for kind in [KaldiWindowKind::Hamming, KaldiWindowKind::Hanning] {
      let s = kaldi_window_scalar(kind, n).expect("test-sized kaldi window should not OOM");
      let d = kaldi_window(kind, n).expect("test-sized kaldi window should not OOM");
      assert_eq!(s.len(), n);
      assert_eq!(d.len(), n);
      for (i, (a, b)) in s.iter().zip(d.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = 1e-5_f32.max(1e-5_f32 * a.abs());
        assert!(
          diff <= tol,
          "tail-handling mismatch at kind={kind:?} n={n} i={i}: \
             scalar={a} dispatcher={b} diff={diff} tol={tol}"
        );
      }
    }
  }
}
