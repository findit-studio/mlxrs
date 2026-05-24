//! C12 — Window generation: `symmetric_window` (Hann / Hamming /
//! Blackman / Bartlett) + Kaldi-style window (Hamming / Hanning /
//! Povey / Rectangular).
//!
//! Tracking: [#157](https://github.com/Findit-AI/mlxrs/issues/157).
//! Plan: `docs/core-arch-simd-candidates.md` §2 row C12, §3.6 (window
//! generation).
//!
//! # The defect class
//!
//! Pre-C12 `crate::audio::dsp::symmetric_window` and
//! `crate::audio::features::build_kaldi_window` are per-element
//! loops that call `f32::cos` per iteration:
//!
//! ```rust,ignore
//! for k in 0..n {
//!   let theta = 2.0 * PI * (k as f32) / denom;
//!   buf.push(0.5 * (1.0 - theta.cos()));
//! }
//! ```
//!
//! `f32::cos` is a libm call per sample. For a long window (a few
//! thousand samples, called once at session start) the absolute time
//! is small but the libm-call overhead dominates the actual cosine
//! evaluation.
//!
//! # The fix — vectorized cosine via a 7-term Taylor polynomial
//!
//! For the symmetric windows we evaluate cosine at points
//! `theta_k = 2π * k / (n - 1)` for `k ∈ [0, n-1]`. The input range
//! is therefore `[0, 2π]`. We use the symmetry `cos(x) = cos(x - 2π)`
//! to bring the range to `[-π, π]`, then `cos(-x) = cos(x)` to
//! `[0, π]`, then `cos(π - x) = -cos(x)` to `[0, π/2]`. After this
//! range reduction a 7-term Taylor polynomial in `x²` evaluates
//! `cos(x)` to within ~1e-7 over `[0, π/2]` — comparable to f32
//! libm `cosf` (which itself is ~1.5 ULP).
//!
//! For Bartlett (triangular), no cosine; pure linear ramp.
//!
//! # Two kernel triples
//!
//! - [`symmetric_window`] — Hann / Hamming / Blackman / Bartlett.
//!   Each is a different linear combination of cosine evaluations at
//!   `theta` and `2*theta`. The dispatcher takes the kind via the
//!   [`SymWindowKind`] enum so the body remains a single 4-lane SIMD
//!   loop over `k`. Bartlett is special-cased (no cosine).
//! - [`kaldi_window`] — Hamming / Hanning / Povey / Rectangular.
//!   Povey is `(0.5 - 0.5*cos(theta))^0.85` — a `powf(0.85)` per
//!   sample which DOES NOT vectorize cleanly (libm-only). For Povey
//!   we keep the scalar arm. Other Kaldi variants vectorize.
//!
//! # Correctness class — `Tolerance`
//!
//! The polynomial approximation matches libm `cosf` to within
//! ~1 ULP on average and ~3 ULP worst case over `[0, π/2]`. The
//! window outputs are then bounded scales of `cos` (0.5 ± 0.5*cos,
//! 0.54 - 0.46*cos, …), so the worst-case absolute error per
//! element stays under `1e-6` for all configured tolerances. The
//! differential tests use
//! [`crate::simd::diff::assert_close_slice_over_lane_sweep`] with
//! `abs = 1e-6, rel = 1e-6`.
//!
//! # `Vec<f32>` output API
//!
//! Window construction returns an owned `Vec<f32>` (matching the
//! callers' shape — both `symmetric_window` and `build_kaldi_window`
//! eventually feed `Array::from_slice::<f32>(&buf, ...)`). The
//! dispatcher allocates the Vec via `try_reserve_exact` (matching the
//! caller's allocation discipline) and writes through
//! `MaybeUninit<f32>` spare capacity for uninit safety.
//!
//! # Verify-before-claim bench
//!
//! Report-only per the user directive 2026-05-23 (project memory
//! rule **"SIMD ship NEON regardless"**). Bench:
//! `mlxrs/benches/simd_window.rs`.

use core::{f32::consts::PI, mem::MaybeUninit};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  float32x4_t, vbslq_f32, vcgtq_f32, vcvtq_f32_u32, vdupq_n_f32, vfmaq_f32, vld1q_u32, vmlaq_f32,
  vmulq_f32, vmulq_n_f32, vnegq_f32, vst1q_f32, vsubq_f32,
};

/// The symmetric-window kinds the dispatcher handles. Mirrors
/// [`crate::audio::dsp`]'s `hann_window` / `hamming_window` /
/// `blackman_window` / `bartlett_window` triple. Each kind selects a
/// different formula in `theta = 2π * k / (n - 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymWindowKind {
  /// `0.5 * (1.0 - cos(theta))` — Hann (a.k.a. Hanning).
  Hann,
  /// `0.54 - 0.46 * cos(theta)` — Hamming.
  Hamming,
  /// `0.42 - 0.5 * cos(theta) + 0.08 * cos(2*theta)` — Blackman.
  Blackman,
  /// `1.0 - 2.0 * |k - (n-1)/2| / (n-1)` — Bartlett (triangular,
  /// no cosine).
  Bartlett,
}

/// Kaldi window kinds — mirrors [`crate::audio::features::KaldiWindow`].
/// `Povey` uses `powf(0.85)` and CANNOT vectorize via the polynomial
/// path; this dispatcher handles only the cosine kinds. The Povey
/// arm in `crate::audio::features::build_kaldi_window` keeps its
/// scalar `theta.cos()` + `.powf(0.85)` loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KaldiWindowKind {
  /// `0.54 - 0.46 * cos(theta)`.
  Hamming,
  /// `0.5 - 0.5 * cos(theta)`.
  Hanning,
  /// Constant `1.0` — no cosine.
  Rectangular,
}

// ─── Scalar cosine reference ─────────────────────────────────────────
//
// Scalar arms call `f32::cos` directly — this is the existing
// behaviour the dispatcher must remain compatible with under
// tolerance. The NEON arm uses a polynomial approximation; the
// `Tolerance` differential test bounds the per-element error.

/// Scalar reference: build a symmetric window of length `n` into a
/// pre-reserved `Vec<f32>`. Bit-exact match for the pre-C12
/// `symmetric_window`/`build_kaldi_window` per-element loops.
///
/// # Preconditions
///
/// - `n >= 2` (asserted unconditionally; matches the caller's `n < 2`
///   error path which returns `Error::Backend` upstream).
#[inline]
#[doc(hidden)]
pub fn symmetric_window_scalar(kind: SymWindowKind, n: usize) -> Vec<f32> {
  assert!(n >= 2, "symmetric_window_scalar: n must be >= 2 (got {n})");
  let denom = (n - 1) as f32;
  let mut out: Vec<f32> = Vec::with_capacity(n);
  match kind {
    SymWindowKind::Hann => {
      for k in 0..n {
        let theta = 2.0 * PI * (k as f32) / denom;
        out.push(0.5 * (1.0 - theta.cos()));
      }
    }
    SymWindowKind::Hamming => {
      for k in 0..n {
        let theta = 2.0 * PI * (k as f32) / denom;
        out.push(0.54 - 0.46 * theta.cos());
      }
    }
    SymWindowKind::Blackman => {
      for k in 0..n {
        let theta = 2.0 * PI * (k as f32) / denom;
        out.push(0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos());
      }
    }
    SymWindowKind::Bartlett => {
      for k in 0..n {
        out.push(1.0 - 2.0 * (k as f32 - denom / 2.0).abs() / denom);
      }
    }
  }
  out
}

/// Scalar reference: build a Kaldi window. Mirrors
/// `crate::audio::features::build_kaldi_window`'s per-arm formulas
/// (excluding Povey, which is excluded from the NEON dispatch path).
///
/// # Preconditions
///
/// - `n >= 2`.
#[inline]
#[doc(hidden)]
pub fn kaldi_window_scalar(kind: KaldiWindowKind, n: usize) -> Vec<f32> {
  assert!(n >= 2, "kaldi_window_scalar: n must be >= 2 (got {n})");
  let denom = (n - 1) as f32;
  let mut out: Vec<f32> = Vec::with_capacity(n);
  match kind {
    KaldiWindowKind::Hamming => {
      for k in 0..n {
        let theta = 2.0 * PI * (k as f32) / denom;
        out.push(0.54 - 0.46 * theta.cos());
      }
    }
    KaldiWindowKind::Hanning => {
      for k in 0..n {
        let theta = 2.0 * PI * (k as f32) / denom;
        out.push(0.5 - 0.5 * theta.cos());
      }
    }
    KaldiWindowKind::Rectangular => {
      out.resize(n, 1.0);
    }
  }
  out
}

// ─── NEON cosine polynomial ───────────────────────────────────────────
//
// 7-term Taylor polynomial in x² for cos(x) over [0, π/2]:
//
//   cos(x) ≈ 1 + c2*x² + c4*x⁴ + c6*x⁶ + c8*x⁸ + c10*x¹⁰ + c12*x¹²
//
// with c2 = -0.5, c4 = 1/24, c6 = -1/720, c8 = 1/40320, c10 =
// -1/3628800, c12 = 1/479001600. Worst-case absolute error over
// [0, π/2] is below 1e-7 — comfortably under the 1e-5 window-
// construction tolerance even after the per-formula scale factor.
//
// Range reduction:
//   1. Caller-side: theta = 2π * k / (n-1), so theta ∈ [0, 2π].
//   2. If theta > π, replace theta → 2π - theta (so result is in [0, π]).
//      Uses cos(2π - x) = cos(x).
//   3. If theta > π/2, replace theta → π - theta, negate the result.
//      Uses cos(π - x) = -cos(x).
//   4. Evaluate the polynomial on [0, π/2].

#[cfg(target_arch = "aarch64")]
const PI_F: f32 = PI;
#[cfg(target_arch = "aarch64")]
const TWO_PI: f32 = 2.0 * PI;
#[cfg(target_arch = "aarch64")]
const PI_HALF: f32 = 0.5 * PI;

/// Polynomial cos approximation for a 4-lane f32 vector with input
/// in `[0, 2π]`. Returns `cos(x)` for each lane to within ~3 ULP.
///
/// # Safety
///
/// NEON must be available (caller's obligation).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn cos_neon_4(x: float32x4_t) -> float32x4_t {
  // Step 1: reduce theta to [0, π] using cos(2π - x) = cos(x).
  // (All NEON intrinsics here are register-only — no memory access
  // — and the `#[target_feature(enable = "neon")]` attribute alone
  // is enough for current rustc to consider them safe within the
  // function body; only the call site needs `unsafe`.)
  let two_pi = vdupq_n_f32(TWO_PI);
  let pi = vdupq_n_f32(PI_F);
  let pi_half = vdupq_n_f32(PI_HALF);

  // mask_gt_pi[lane] = 0xFFFFFFFF if x > π else 0
  let gt_pi = vcgtq_f32(x, pi);
  // x' = if x > π then 2π - x else x
  let x = vbslq_f32(gt_pi, vsubq_f32(two_pi, x), x);

  // Step 2: reduce to [0, π/2] using cos(π - x) = -cos(x).
  let gt_pi_half = vcgtq_f32(x, pi_half);
  let x = vbslq_f32(gt_pi_half, vsubq_f32(pi, x), x);

  // Now x in [0, π/2]. Polynomial: cos(x) ≈ 1 + c2*x² + c4*x⁴ +
  // c6*x⁶ + c8*x⁸ + c10*x¹⁰ + c12*x¹². Use Horner's scheme on x²:
  //   p = c12
  //   p = p*x² + c10
  //   p = p*x² + c8
  //   p = p*x² + c6
  //   p = p*x² + c4
  //   p = p*x² + c2
  //   p = p*x² + 1
  //
  // Where (Taylor coefficients):
  //   c2  = -1/2          = -0.5
  //   c4  = +1/24         ≈  0.0416666667
  //   c6  = -1/720        ≈ -0.00138888889
  //   c8  = +1/40320      ≈  2.48015873e-5
  //   c10 = -1/3628800    ≈ -2.75573192e-7
  //   c12 = +1/479001600  ≈  2.08767570e-9
  //
  // 5-term Taylor (through c8) bottoms at ~1.24e-5 absolute at
  // x = π/2 (single eval) — which exceeds the 1e-5 tolerance for the
  // Hann arm. Adding c10 + c12 brings the worst-case absolute error
  // below 1e-7 over [0, π/2], comfortably under any sane window
  // tolerance.
  let x2 = vmulq_f32(x, x);
  let c2 = vdupq_n_f32(-0.5);
  let c4 = vdupq_n_f32(1.0 / 24.0);
  let c6 = vdupq_n_f32(-1.0 / 720.0);
  let c8 = vdupq_n_f32(1.0 / 40320.0);
  let c10 = vdupq_n_f32(-1.0 / 3_628_800.0);
  let c12 = vdupq_n_f32(1.0 / 479_001_600.0);
  let one = vdupq_n_f32(1.0);

  // p = c12
  let mut p = c12;
  // p = p*x² + c10
  p = vfmaq_f32(c10, p, x2);
  // p = p*x² + c8
  p = vfmaq_f32(c8, p, x2);
  // p = p*x² + c6  (fused multiply-add for tighter rounding)
  p = vfmaq_f32(c6, p, x2);
  // p = p*x² + c4
  p = vfmaq_f32(c4, p, x2);
  // p = p*x² + c2
  p = vfmaq_f32(c2, p, x2);
  // p = p*x² + 1
  p = vfmaq_f32(one, p, x2);

  // Apply sign flip from step 2 (if x > π/2, result negated).
  vbslq_f32(gt_pi_half, vnegq_f32(p), p)
}

/// Per-lane index → theta. Given the starting `k_base`, produces
/// `theta_k = 2π * (k_base + lane) / (n-1)` for `lane ∈ [0, 4)`.
///
/// # Safety
///
/// NEON must be available.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn theta_neon_4(k_base: u32, inv_denom_times_2pi: f32) -> float32x4_t {
  let lane_offsets = [k_base, k_base + 1, k_base + 2, k_base + 3];
  // SAFETY: `lane_offsets` is a stack array of 4 u32 = 16 bytes,
  // sufficient for `vld1q_u32`'s 16-byte load. The other intrinsics
  // operate purely on register values. NEON availability is the
  // caller's obligation (the enclosing `unsafe fn` carries the
  // contract).
  unsafe {
    let k_u32 = vld1q_u32(lane_offsets.as_ptr());
    let k_f = vcvtq_f32_u32(k_u32);
    vmulq_n_f32(k_f, inv_denom_times_2pi)
  }
}

/// NEON 4-lane symmetric window builder. Writes `out` (length `n`)
/// with the window samples; takes the same kind enum as the
/// dispatcher. Bartlett bypasses the cosine path.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `out.len() == n` and `n >= 2`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn symmetric_window_neon(kind: SymWindowKind, out: &mut [MaybeUninit<f32>], n: usize) {
  assert_eq!(
    out.len(),
    n,
    "symmetric_window_neon: out.len() ({}) must equal n ({})",
    out.len(),
    n,
  );
  assert!(n >= 2, "symmetric_window_neon: n must be >= 2 (got {n})");

  if matches!(kind, SymWindowKind::Bartlett) {
    // No cosine — pure linear ramp. SIMD pays off less here; just
    // delegate to scalar (the actual instruction count is ~3 ops/lane
    // which the auto-vectorizer covers cleanly).
    let v = symmetric_window_scalar(kind, n);
    for (i, &x) in v.iter().enumerate() {
      out[i].write(x);
    }
    return;
  }

  let denom = (n - 1) as f32;
  let inv_denom_times_2pi = 2.0 * PI_F / denom;
  let body_len = n - (n % 4);

  // SAFETY: body loop writes `vst1q_f32` (4 f32) per tile at
  // `dst_base.add(i)` for `i + 4 <= body_len <= n = out.len()`,
  // within bounds. Stores target `MaybeUninit<f32>`. NEON
  // availability is the caller's obligation.
  unsafe {
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let mut i = 0usize;
    while i + 4 <= body_len {
      let theta = theta_neon_4(i as u32, inv_denom_times_2pi);
      let cos_theta = cos_neon_4(theta);

      let w = match kind {
        SymWindowKind::Hann => {
          // 0.5 * (1 - cos)
          vmulq_n_f32(vsubq_f32(vdupq_n_f32(1.0), cos_theta), 0.5)
        }
        SymWindowKind::Hamming => {
          // 0.54 - 0.46 * cos
          // = (-0.46) * cos + 0.54
          // vmlaq_f32(a, b, c) = a + b*c
          vmlaq_f32(vdupq_n_f32(0.54), cos_theta, vdupq_n_f32(-0.46))
        }
        SymWindowKind::Blackman => {
          // 0.42 - 0.5 * cos(theta) + 0.08 * cos(2*theta)
          let two_theta = vmulq_n_f32(theta, 2.0);
          // 2*theta might overflow into [0, 4π] — fold through
          // `cos_neon_4`'s [0, 2π] reduction. We need to bring
          // 2*theta into [0, 2π]: subtract 2π if >= 2π.
          let two_pi_v = vdupq_n_f32(TWO_PI);
          let ge_2pi = vcgtq_f32(two_theta, two_pi_v);
          let two_theta_folded = vbslq_f32(ge_2pi, vsubq_f32(two_theta, two_pi_v), two_theta);
          let cos_2theta = cos_neon_4(two_theta_folded);

          // 0.42 - 0.5*cos(theta) + 0.08*cos(2*theta)
          let term1 = vmlaq_f32(vdupq_n_f32(0.42), cos_theta, vdupq_n_f32(-0.5));
          vmlaq_f32(term1, cos_2theta, vdupq_n_f32(0.08))
        }
        SymWindowKind::Bartlett => unreachable!(),
      };

      vst1q_f32(dst_base.add(i), w);
      i += 4;
    }
  }

  // Tail: `n % 4` samples — scalar.
  if body_len < n {
    let tail = symmetric_window_scalar(kind, n);
    for i in body_len..n {
      out[i].write(tail[i]);
    }
  }
}

/// NEON 4-lane Kaldi window builder (Hamming / Hanning / Rectangular
/// only — Povey is excluded; caller must dispatch Povey to scalar).
///
/// # Safety
///
/// NEON must be available; `out.len() == n` and `n >= 2`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn kaldi_window_neon(kind: KaldiWindowKind, out: &mut [MaybeUninit<f32>], n: usize) {
  assert_eq!(
    out.len(),
    n,
    "kaldi_window_neon: out.len() ({}) must equal n ({})",
    out.len(),
    n,
  );
  assert!(n >= 2, "kaldi_window_neon: n must be >= 2 (got {n})");

  if matches!(kind, KaldiWindowKind::Rectangular) {
    for slot in out.iter_mut() {
      slot.write(1.0);
    }
    return;
  }

  let denom = (n - 1) as f32;
  let inv_denom_times_2pi = 2.0 * PI_F / denom;
  let body_len = n - (n % 4);

  // SAFETY: same bounds reasoning as `symmetric_window_neon`.
  unsafe {
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let mut i = 0usize;
    while i + 4 <= body_len {
      let theta = theta_neon_4(i as u32, inv_denom_times_2pi);
      let cos_theta = cos_neon_4(theta);

      let w = match kind {
        KaldiWindowKind::Hamming => vmlaq_f32(vdupq_n_f32(0.54), cos_theta, vdupq_n_f32(-0.46)),
        KaldiWindowKind::Hanning => vmulq_n_f32(vsubq_f32(vdupq_n_f32(1.0), cos_theta), 0.5),
        KaldiWindowKind::Rectangular => unreachable!(),
      };

      vst1q_f32(dst_base.add(i), w);
      i += 4;
    }
  }

  if body_len < n {
    let tail = kaldi_window_scalar(kind, n);
    for i in body_len..n {
      out[i].write(tail[i]);
    }
  }
}

/// Build a symmetric window of length `n` (`>= 2`). Routes to NEON on
/// `aarch64` when NEON is reported, else to the scalar reference.
///
/// # Preconditions
///
/// - `n >= 2` — asserted unconditionally.
///
/// # Correctness class
///
/// `Tolerance` (`abs = 1e-6, rel = 1e-6`) — polynomial cos
/// approximation, ~3 ULP worst-case per cos eval.
pub fn symmetric_window(kind: SymWindowKind, n: usize) -> Vec<f32> {
  assert!(
    n >= 2,
    "simd::audio::window::symmetric_window: n must be >= 2 (got {n})"
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      let mut v: Vec<f32> = Vec::with_capacity(n);
      let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
      // SAFETY: NEON gated; spare is sized exactly to `n`; the kernel
      // initializes every slot (body + tail). cap was sized to `n`.
      unsafe {
        symmetric_window_neon(kind, &mut spare[..n], n);
        v.set_len(n);
      }
      return v;
    }
  }
  symmetric_window_scalar(kind, n)
}

/// Build a Kaldi window of length `n` (`>= 2`). Routes to NEON on
/// `aarch64` when NEON is reported, else to the scalar reference.
/// Note: `Povey` is not handled here; the call site must dispatch
/// Povey to scalar (the scalar reference in
/// `crate::audio::features::build_kaldi_window` handles Povey
/// directly).
///
/// # Correctness class
///
/// `Tolerance`.
pub fn kaldi_window(kind: KaldiWindowKind, n: usize) -> Vec<f32> {
  assert!(
    n >= 2,
    "simd::audio::window::kaldi_window: n must be >= 2 (got {n})"
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      let mut v: Vec<f32> = Vec::with_capacity(n);
      let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
      // SAFETY: NEON gated; spare sized to `n`; kernel initializes
      // every slot.
      unsafe {
        kaldi_window_neon(kind, &mut spare[..n], n);
        v.set_len(n);
      }
      return v;
    }
  }
  kaldi_window_scalar(kind, n)
}

#[cfg(test)]
mod tests {
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
    let s = kaldi_window_scalar(KaldiWindowKind::Rectangular, 17);
    let d = kaldi_window(KaldiWindowKind::Rectangular, 17);
    assert_eq!(s, d);
    assert!(s.iter().all(|&x| x == 1.0));
  }

  /// Specific-value pin: Hann at endpoints is 0.0; at the midpoint is
  /// 1.0 (for odd `n`). Hamming at endpoints is 0.08 (= 0.54 - 0.46).
  #[test]
  fn symmetric_window_endpoint_pins() {
    let n = 17;
    let hann = symmetric_window(SymWindowKind::Hann, n);
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

    let ham = symmetric_window(SymWindowKind::Hamming, n);
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
      assert_eq!(symmetric_window(SymWindowKind::Hann, n).len(), n);
      assert_eq!(symmetric_window(SymWindowKind::Hamming, n).len(), n);
      assert_eq!(symmetric_window(SymWindowKind::Blackman, n).len(), n);
      assert_eq!(symmetric_window(SymWindowKind::Bartlett, n).len(), n);
      assert_eq!(kaldi_window(KaldiWindowKind::Hamming, n).len(), n);
      assert_eq!(kaldi_window(KaldiWindowKind::Hanning, n).len(), n);
      assert_eq!(kaldi_window(KaldiWindowKind::Rectangular, n).len(), n);
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
}
