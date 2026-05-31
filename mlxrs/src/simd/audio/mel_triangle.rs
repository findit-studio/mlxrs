//! `mel_filter_bank` triangle construction.
//!
//! Tracking: [#155](https://github.com/Findit-AI/mlxrs/issues/155).
//!
//! # The defect class
//!
//! The original `crate::audio::dsp::mel_filter_bank` has a per-row inner
//! loop over `all_freqs` that evaluates the triangle membership for
//! each `(m, f)` cell:
//!
//! ```rust,ignore
//! for m in 0..n_mels {
//!   let left   = f_pts[m];
//!   let center = f_pts[m + 1];
//!   let right  = f_pts[m + 2];
//!   let lc = center - left;
//!   let cr = right - center;
//!   if lc <= 0.0 || cr <= 0.0 { continue; }
//!   for (f, &freq) in all_freqs.iter().enumerate() {
//!     let up   = (freq - left) / lc;
//!     let down = (right - freq) / cr;
//!     let v    = up.min(down).max(0.0);
//!     bank[m * n_freqs + f] = v;
//!   }
//! }
//! ```
//!
//! Per cell: two subtracts, two divides, one `min`, one `max`. The
//! filterbank is `(n_mels, n_freqs)` and built once at session start,
//! so the absolute cost is small — but the SIMD shape is a clean
//! 4-lane f32 row, and shipping the NEON arm pins the contract against
//! a future scalar regression (LLVM heuristics + the per-row guard
//! `if lc <= 0.0 || cr <= 0.0` can de-vectorize the inner loop on a
//! rustc upgrade).
//!
//! # The fix — 4-lane row builder
//!
//! Per output row `m` (`n_freqs` cells):
//! 1. Pre-compute the per-row scalars `left, center, right, inv_lc,
//!    inv_cr, left_over_lc, right_over_cr`. Substituting
//!    `freq * inv_lc - left_over_lc` for `(freq - left) / lc` hoists
//!    the division out of the inner loop into a single reciprocal per
//!    row.
//! 2. 4-lane tile over `all_freqs`:
//!    - `vld1q_f32` 4 freqs.
//!    - `up   = freq * inv_lc - left_over_lc`.
//!    - `down = right_over_cr - freq * inv_cr`.
//!    - `vminq_f32(up, down)`.
//!    - `vmaxq_f32(_, vdupq_n_f32(0.0))`.
//!    - `vst1q_f32` 4 cells to the output row.
//! 3. Tail (`n_freqs % 4` ≤ 3) handled by the per-row scalar loop.
//!
//! Zero-width triangles (`lc <= 0.0` or `cr <= 0.0`) — the kernel
//! writes 0.0 to the whole row (matches the scalar reference, where
//! the caller pre-zeroed the bank and `continue` left the row at
//! 0.0; here we write 0.0 explicitly through `MaybeUninit::write` so
//! the dispatcher carries the `MaybeUninit` init contract instead of
//! relying on a caller `Vec::resize(_, 0.0)` pre-pass).
//!
//! # Correctness class — `Tolerance`
//!
//! The substitution `(freq - left) / lc → freq * (1/lc) - left * (1/lc)`
//! is **algebraically identical** in real arithmetic but differs in
//! f32 rounding (one reciprocal + one multiply + one subtract vs one
//! subtract + one divide). The per-cell error is at most 2 ULP
//! (~6e-7 absolute for the [0, 1] output range). Differential tests
//! pin the per-cell error to `abs = 1e-6, rel = 1e-6`.
//!
//! # `Vec<f32>` output API
//!
//! Matches `mel_filter_bank`'s allocation discipline — the dispatcher
//! writes into a pre-reserved `&mut [MaybeUninit<f32>]` (sized to
//! `n_mels * n_freqs`), and the caller wraps it with
//! `Vec::with_capacity` + `spare_capacity_mut` + `set_len`.

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  vdupq_n_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
};

/// Scalar reference: build a `(n_mels, n_freqs)` mel filterbank into
/// `out` from the precomputed `all_freqs` (length `n_freqs`) and
/// `f_pts` (length `n_mels + 2`). Bit-exact match for the original
/// `mel_filter_bank` inner two-loop **plus** explicit 0.0 writes for
/// the `lc <= 0` / `cr <= 0` zero-width-triangle rows (the scalar
/// reference relied on `Vec::resize(_, 0.0)` to pre-zero those rows;
/// here the `MaybeUninit` init contract is satisfied by writing 0.0
/// explicitly).
///
/// # Preconditions
///
/// - `out.len() == n_mels * n_freqs`.
/// - `f_pts.len() == n_mels + 2`.
/// - `n_mels * n_freqs` does not overflow `usize` — checked
///   explicitly via `checked_mul` rather than wrapping silently.
///
/// All asserted **unconditionally** (release-too) because `out` is
/// `MaybeUninit<f32>` — a short-iteration would leave slots
/// uninitialized and a caller `set_len` would expose them.
///
/// # Panics
///
/// Panics explicitly (not silently wraps) on `n_mels * n_freqs`
/// `usize` overflow — same defect class as `rotate_buf_u8`.
/// Silently wrapping would let an under-sized `out` satisfy the
/// size-equality assertion and reach the per-cell init loop.
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns.
#[inline]
#[doc(hidden)]
pub fn mel_filter_bank_rows_scalar(
  out: &mut [MaybeUninit<f32>],
  all_freqs: &[f32],
  f_pts: &[f32],
  n_mels: usize,
) {
  let n_freqs = all_freqs.len();
  let elements = n_mels.checked_mul(n_freqs).unwrap_or_else(|| {
    panic!("mel_filter_bank_rows_scalar: dimensions {n_mels}x{n_freqs} overflow usize")
  });
  assert_eq!(
    out.len(),
    elements,
    "mel_filter_bank_rows_scalar: out.len() ({}) must equal n_mels * n_freqs ({} * {} = {})",
    out.len(),
    n_mels,
    n_freqs,
    elements,
  );
  assert_eq!(
    f_pts.len(),
    n_mels + 2,
    "mel_filter_bank_rows_scalar: f_pts.len() ({}) must equal n_mels + 2 ({})",
    f_pts.len(),
    n_mels + 2,
  );

  for m in 0..n_mels {
    let left = f_pts[m];
    let center = f_pts[m + 1];
    let right = f_pts[m + 2];
    let lc = center - left;
    let cr = right - center;
    let row_off = m * n_freqs;
    if lc <= 0.0 || cr <= 0.0 {
      // Zero-width triangle — write 0.0 to the whole row.
      for k in 0..n_freqs {
        out[row_off + k].write(0.0);
      }
      continue;
    }
    for (f, &freq) in all_freqs.iter().enumerate() {
      let up = (freq - left) / lc;
      let down = (right - freq) / cr;
      let v = up.min(down).max(0.0);
      out[row_off + f].write(v);
    }
  }
}

/// NEON 4-lane mel filterbank row builder. See module-level "The fix"
/// for the per-tile structure.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    discharged by [`mel_filter_bank_rows`].
/// 2. `out.len() == n_mels * n_freqs` and `f_pts.len() == n_mels + 2`
///    — both asserted **unconditionally** here.
/// 3. `n_mels * n_freqs` does not overflow `usize` — checked via
///    `checked_mul` BEFORE the size-equality assertion, so a wrapped
///    product can never sneak past the size check and let the NEON
///    kernel compute per-row offsets `m * n_freqs` from unwrapped
///    loop dims (which would issue out-of-bounds `vst1q_f32` /
///    `vld1q_f32`).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn mel_filter_bank_rows_neon(
  out: &mut [MaybeUninit<f32>],
  all_freqs: &[f32],
  f_pts: &[f32],
  n_mels: usize,
) {
  let n_freqs = all_freqs.len();
  let elements = n_mels.checked_mul(n_freqs).unwrap_or_else(|| {
    panic!("mel_filter_bank_rows_neon: dimensions {n_mels}x{n_freqs} overflow usize")
  });
  assert_eq!(
    out.len(),
    elements,
    "mel_filter_bank_rows_neon: out.len() ({}) must equal n_mels * n_freqs ({} * {} = {})",
    out.len(),
    n_mels,
    n_freqs,
    elements,
  );
  assert_eq!(
    f_pts.len(),
    n_mels + 2,
    "mel_filter_bank_rows_neon: f_pts.len() ({}) must equal n_mels + 2 ({})",
    f_pts.len(),
    n_mels + 2,
  );

  let body_len = n_freqs - (n_freqs % 4);
  let zero = vdupq_n_f32(0.0);

  // SAFETY: each `vst1q_f32` writes 16 bytes (4 f32 slots of
  // `MaybeUninit<f32>`) per tile at `dst_base.add(row_off + k)` where
  // `row_off + k + 4 <= row_off + body_len <= row_off + n_freqs <=
  // n_mels * n_freqs = out.len()`. The `vld1q_f32` reads 4 freqs from
  // `freq_base.add(k)` where `k + 4 <= body_len <= n_freqs =
  // all_freqs.len()`. Both within bounds. Stores target
  // `MaybeUninit<f32>` backing memory, which has no validity
  // invariants beyond size + alignment. NEON availability is the
  // caller's obligation (precondition #1).
  unsafe {
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let freq_base = all_freqs.as_ptr();

    for m in 0..n_mels {
      let left = f_pts[m];
      let center = f_pts[m + 1];
      let right = f_pts[m + 2];
      let lc = center - left;
      let cr = right - center;
      let row_off = m * n_freqs;

      if lc <= 0.0 || cr <= 0.0 {
        // Zero the whole row via 4-lane stores + scalar tail.
        let mut k = 0usize;
        while k + 4 <= body_len {
          vst1q_f32(dst_base.add(row_off + k), zero);
          k += 4;
        }
        for kk in body_len..n_freqs {
          out[row_off + kk].write(0.0);
        }
        continue;
      }

      // Precompute reciprocals + scaled offsets per row.
      let inv_lc = 1.0_f32 / lc;
      let inv_cr = 1.0_f32 / cr;
      let left_over_lc = left * inv_lc;
      let right_over_cr = right * inv_cr;

      let inv_lc_v = vdupq_n_f32(inv_lc);
      let inv_cr_v = vdupq_n_f32(inv_cr);
      let left_over_lc_v = vdupq_n_f32(left_over_lc);
      let right_over_cr_v = vdupq_n_f32(right_over_cr);

      let mut k = 0usize;
      while k + 4 <= body_len {
        let f_v = vld1q_f32(freq_base.add(k));

        // up = freq * inv_lc - left_over_lc
        let up = vsubq_f32(vmulq_f32(f_v, inv_lc_v), left_over_lc_v);

        // down = right_over_cr - freq * inv_cr
        let prod = vmulq_f32(f_v, inv_cr_v);
        let down = vsubq_f32(right_over_cr_v, prod);

        // v = min(up, down).max(0.0)
        let mn = vminq_f32(up, down);
        let v = vmaxq_f32(mn, zero);

        vst1q_f32(dst_base.add(row_off + k), v);

        k += 4;
      }

      // Tail (`n_freqs % 4` cells, ≤ 3) — use the scalar-arithmetic
      // (subtract-then-divide) shape for bit-equality with the scalar
      // arm at the tail (the body uses the FMA-hoisted shape).
      for kk in body_len..n_freqs {
        let freq = all_freqs[kk];
        let up = (freq - left) / lc;
        let down = (right - freq) / cr;
        let v = up.min(down).max(0.0);
        out[row_off + kk].write(v);
      }
    }
  }
}

/// Public dispatcher: build a `(n_mels, n_freqs)` mel filterbank into
/// `out`. Routes to NEON on `aarch64` when NEON is reported, else to
/// the scalar reference.
///
/// Used by [`crate::audio::dsp::mel_filter_bank`] to fill the
/// pre-reserved bank buffer.
///
/// # Preconditions
///
/// - `out.len() == n_mels * all_freqs.len()` — asserted unconditionally.
/// - `f_pts.len() == n_mels + 2` — asserted unconditionally.
/// - `n_mels * all_freqs.len()` does not overflow `usize` — checked
///   via `checked_mul` BEFORE the size-equality assertion (same
///   defect class as `rotate_buf_u8` and the sibling
///   `get_mel_banks_kaldi_rows`).
///
/// # Panics
///
/// Panics explicitly (not silently wraps) on `n_mels * n_freqs`
/// `usize` overflow — the only correct response when caller dims
/// cannot fit a contiguous buffer, since silently wrapping would let
/// an under-sized buffer satisfy the size-equality assertion and reach
/// either inner kernel.
///
/// # Initialization contract
///
/// **Every f32 of `out` is written before this returns** (zero-width
/// rows write 0.0; non-zero rows write the per-cell triangle value).
///
/// # Correctness class
///
/// `Tolerance` (`abs = 1e-6, rel = 1e-6`) — the reciprocal-hoist
/// rearrangement of `(freq - left) / lc` introduces at most 2 ULP per
/// cell relative to the scalar arm's direct subtract-then-divide.
#[inline]
#[doc(hidden)]
pub fn mel_filter_bank_rows(
  out: &mut [MaybeUninit<f32>],
  all_freqs: &[f32],
  f_pts: &[f32],
  n_mels: usize,
) {
  let n_freqs = all_freqs.len();
  // Checked dimension math BEFORE the size-equality assertion: wrapping
  // `n_mels * n_freqs` in release mode could otherwise produce a small
  // `elements` that an under-sized `out` would satisfy, letting either
  // inner kernel compute per-row offsets from unwrapped loop dims
  // (same defect class as `rotate_buf_u8`).
  let elements = n_mels.checked_mul(n_freqs).unwrap_or_else(|| {
    panic!("simd::audio::mel_filter_bank_rows: dimensions {n_mels}x{n_freqs} overflow usize")
  });
  assert_eq!(
    out.len(),
    elements,
    "simd::audio::mel_filter_bank_rows: out.len() ({}) must equal n_mels * n_freqs ({} * {} = {})",
    out.len(),
    n_mels,
    n_freqs,
    elements,
  );
  assert_eq!(
    f_pts.len(),
    n_mels + 2,
    "simd::audio::mel_filter_bank_rows: f_pts.len() ({}) must equal n_mels + 2 ({})",
    f_pts.len(),
    n_mels + 2,
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: NEON gated; slice-length preconditions asserted above;
      // the kernel writes every cell of `out`.
      unsafe { mel_filter_bank_rows_neon(out, all_freqs, f_pts, n_mels) };
      return;
    }
  }
  mel_filter_bank_rows_scalar(out, all_freqs, f_pts, n_mels);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher Tolerance differential tests + edge coverage
  //! for the mel triangle.

  use super::{mel_filter_bank_rows, mel_filter_bank_rows_scalar};

  /// Build a synthetic `(all_freqs, f_pts)` pair of the requested
  /// shape.
  fn make_inputs(n_freqs: usize, n_mels: usize) -> (Vec<f32>, Vec<f32>) {
    // all_freqs: linearly spaced in [0, 8000) — same shape as
    // `mel_filter_bank` builds for sample_rate = 16000.
    let mut all_freqs: Vec<f32> = Vec::with_capacity(n_freqs);
    let denom = (n_freqs as f32 - 1.0).max(1.0);
    for k in 0..n_freqs {
      all_freqs.push(8000.0 * (k as f32) / denom);
    }
    // f_pts: `n_mels + 2` points spaced through [50, 7500].
    let n_pts = n_mels + 2;
    let mut f_pts: Vec<f32> = Vec::with_capacity(n_pts);
    let pts_denom = (n_pts as f32 - 1.0).max(1.0);
    for i in 0..n_pts {
      f_pts.push(50.0 + (7500.0 - 50.0) * (i as f32) / pts_denom);
    }
    (all_freqs, f_pts)
  }

  fn bank_via_scalar(n_freqs: usize, n_mels: usize) -> Vec<f32> {
    let (all_freqs, f_pts) = make_inputs(n_freqs, n_mels);
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    out
  }

  fn bank_via_dispatch(n_freqs: usize, n_mels: usize) -> Vec<f32> {
    let (all_freqs, f_pts) = make_inputs(n_freqs, n_mels);
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    out
  }

  #[test]
  fn mel_filter_bank_scalar_matches_dispatcher_tolerance() {
    let n_mels = 8usize;
    // Lane sweep over `n_freqs` (the inner-loop tile is 4-lane, so
    // exercise boundaries around multiples of 4).
    for &n_freqs in &[5usize, 8, 16, 17, 64, 201, 257] {
      let s = bank_via_scalar(n_freqs, n_mels);
      let d = bank_via_dispatch(n_freqs, n_mels);
      assert_eq!(s.len(), d.len(), "shape parity at n_freqs={n_freqs}");
      for (i, (a, b)) in s.iter().zip(d.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = 1e-6_f32.max(1e-6_f32 * a.abs());
        assert!(
          diff <= tol,
          "Tolerance mismatch at n_freqs={n_freqs} i={i}: scalar={a} dispatcher={b} \
           diff={diff} tol={tol}"
        );
      }
    }
  }

  #[test]
  fn mel_filter_bank_triangle_shape() {
    // Sanity: every cell is in [0, 1].
    let n_mels = 4usize;
    let n_freqs = 65;
    let bank = bank_via_dispatch(n_freqs, n_mels);
    for m in 0..n_mels {
      let row = &bank[m * n_freqs..(m + 1) * n_freqs];
      for (i, &v) in row.iter().enumerate() {
        assert!(
          (0.0..=1.0001).contains(&v),
          "cell out of [0, 1]: m={m} i={i} v={v}"
        );
      }
    }
  }

  #[test]
  fn mel_filter_bank_collapsed_row_is_zero() {
    // Force a row's `lc <= 0` by collapsing two adjacent f_pts.
    let n_mels = 3;
    let n_freqs = 16;
    let all_freqs: Vec<f32> = (0..n_freqs).map(|k| 100.0 * k as f32).collect();
    // f_pts[1] == f_pts[2] → row m=1 has lc = 0.
    let f_pts = vec![0.0, 500.0, 500.0, 1500.0, 2000.0];
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel writes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    for k in 0..n_freqs {
      assert_eq!(
        out[n_freqs + k],
        0.0,
        "collapsed row m=1 cell k={k} should be 0.0"
      );
    }
  }

  #[test]
  #[should_panic(
    expected = "simd::audio::mel_filter_bank_rows: out.len() (3) must equal n_mels * n_freqs"
  )]
  fn mel_filter_bank_panics_on_size_mismatch() {
    let all_freqs = vec![100.0_f32, 200.0, 300.0, 400.0];
    let f_pts = vec![0.0, 200.0, 400.0, 600.0]; // n_mels = 2 → 4 pts
    let mut out: Vec<f32> = Vec::with_capacity(3); // WRONG: should be 2*4 = 8
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(&mut spare[..3], &all_freqs, &f_pts, 2);
  }

  #[test]
  #[should_panic(expected = "f_pts.len() (5) must equal n_mels + 2 (4)")]
  fn mel_filter_bank_panics_on_f_pts_size_mismatch() {
    let all_freqs = vec![100.0_f32, 200.0, 300.0];
    let f_pts = vec![0.0, 200.0, 400.0, 600.0, 800.0]; // n_mels=2 expects 4
    let mut out: Vec<f32> = Vec::with_capacity(6);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(&mut spare[..6], &all_freqs, &f_pts, 2);
  }

  /// Wrap-arith defence (same defect class as `rotate_buf_u8` /
  /// the sibling `get_mel_banks_kaldi_rows`): `n_mels * n_freqs`
  /// MUST be evaluated via `checked_mul`. A wrapping multiply in
  /// release mode would otherwise let an under-sized (here
  /// zero-length) `out` satisfy the size-equality assertion,
  /// reaching either inner kernel where per-row offsets
  /// (`m * n_freqs`) would then be computed from the unwrapped loop
  /// dims and produce out-of-bounds NEON stores. Adversarial case:
  /// `n_mels = usize::MAX / 4 + 1, n_freqs = 4` wraps the product.
  ///
  /// `f_pts` here is `vec![0.0; n_mels + 2]` only nominally — we
  /// never reach the `f_pts.len()` assert because the `checked_mul`
  /// panic fires first. (Allocating `n_mels + 2` f32s for an
  /// adversarial `n_mels` is itself infeasible; we deliberately use
  /// the unwrapped `n_mels` only as a plain `usize` argument and
  /// skip building `f_pts` of the correct length — the
  /// `checked_mul` overflow guard runs strictly before the
  /// `f_pts.len()` assert in the new ordering.)
  #[test]
  #[should_panic(expected = "overflow usize")]
  fn mel_filter_bank_panics_on_dimension_overflow() {
    let n_mels = usize::MAX / 4 + 1;
    let all_freqs = vec![0.0_f32; 4]; // n_freqs = 4 → n_mels * n_freqs overflows
    // `f_pts` skipped — we never reach its assert because the
    // `checked_mul` panic fires first (asserted by `#[should_panic]`'s
    // message match).
    let f_pts = vec![0.0_f32; 4];
    let mut out: Vec<f32> = Vec::new();
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(spare, &all_freqs, &f_pts, n_mels);
  }

  // ── Scalar-reference direct coverage ──────────────────────────────
  //
  // The dispatcher routes to the NEON arm on a NEON-capable aarch64
  // host, so the scalar reference's own preconditions / zero-width
  // path are only exercised when the scalar kernel is called directly
  // (here) or when the dispatcher falls through to it (a
  // `--cfg mlxrs_force_scalar` build, or a non-aarch64 target). These
  // tests call `mel_filter_bank_rows_scalar` directly so the scalar
  // arm's lines are covered independent of the runtime NEON decision.

  /// Hand-computed triangle oracle (NOT a self-comparison): a known
  /// `(all_freqs, f_pts)` pair with closed-form rising/falling edges.
  /// Row m=0 has left=0, center=200, right=400 (lc=cr=200); the
  /// per-cell weight is `min((freq-left)/lc, (right-freq)/cr).max(0)`.
  #[test]
  fn mel_filter_bank_scalar_matches_handcomputed_triangle() {
    let n_mels = 1usize;
    let all_freqs = vec![0.0_f32, 100.0, 200.0, 300.0, 400.0, 500.0];
    let n_freqs = all_freqs.len();
    // n_mels + 2 == 3 points → one triangle [0, 200, 400].
    let f_pts = vec![0.0_f32, 200.0, 400.0];
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    // up   = (freq - 0) / 200, down = (400 - freq) / 200, v = min.max(0)
    //   0   -> min(0.0, 2.0) = 0.0
    //   100 -> min(0.5, 1.5) = 0.5
    //   200 -> min(1.0, 1.0) = 1.0
    //   300 -> min(1.5, 0.5) = 0.5
    //   400 -> min(2.0, 0.0) = 0.0
    //   500 -> min(2.5, -0.5).max(0) = 0.0
    let expected = [0.0_f32, 0.5, 1.0, 0.5, 0.0, 0.0];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
      assert!(
        (got - want).abs() <= 1e-6,
        "hand-computed triangle mismatch i={i}: got={got} want={want}"
      );
    }
  }

  /// Scalar zero-width row: collapse `f_pts[m+1] == f_pts[m]` so
  /// `lc <= 0` and the whole row is written 0.0 via the scalar
  /// zero-width loop. `n_freqs = 6` (not a multiple of 4) so the loop
  /// spans the full row length, not just lane-aligned cells.
  #[test]
  fn mel_filter_bank_scalar_collapsed_row_is_zero() {
    let n_mels = 3usize;
    let n_freqs = 6usize; // n_freqs % 4 == 2
    let all_freqs: Vec<f32> = (0..n_freqs).map(|k| 100.0 * k as f32).collect();
    // f_pts[1] == f_pts[2] → row m=1 has lc = 0 (zero-width triangle).
    let f_pts = vec![0.0_f32, 500.0, 500.0, 1500.0, 2000.0];
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel writes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    for k in 0..n_freqs {
      assert_eq!(
        out[n_freqs + k],
        0.0,
        "scalar collapsed row m=1 cell k={k} should be 0.0"
      );
    }
  }

  #[test]
  #[should_panic(
    expected = "mel_filter_bank_rows_scalar: out.len() (3) must equal n_mels * n_freqs"
  )]
  fn mel_filter_bank_scalar_panics_on_size_mismatch() {
    let all_freqs = vec![100.0_f32, 200.0, 300.0, 400.0]; // n_freqs = 4
    let f_pts = vec![0.0_f32, 200.0, 400.0, 600.0]; // n_mels = 2 → 4 pts
    let mut out: Vec<f32> = Vec::with_capacity(3); // WRONG: should be 2*4 = 8
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(&mut spare[..3], &all_freqs, &f_pts, 2);
  }

  #[test]
  #[should_panic(
    expected = "mel_filter_bank_rows_scalar: f_pts.len() (5) must equal n_mels + 2 (4)"
  )]
  fn mel_filter_bank_scalar_panics_on_f_pts_size_mismatch() {
    let all_freqs = vec![100.0_f32, 200.0, 300.0]; // n_freqs = 3
    let f_pts = vec![0.0_f32, 200.0, 400.0, 600.0, 800.0]; // n_mels=2 expects 4
    let mut out: Vec<f32> = Vec::with_capacity(6);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(&mut spare[..6], &all_freqs, &f_pts, 2);
  }

  /// Scalar arm's `checked_mul` overflow guard (same defect class as
  /// the dispatcher's). `n_mels = usize::MAX / 4 + 1, n_freqs = 4`
  /// wraps the product; the explicit panic must fire before the
  /// size-equality / `f_pts.len()` asserts (so the empty `out` and
  /// short `f_pts` never reach their checks).
  #[test]
  #[should_panic(expected = "overflow usize")]
  fn mel_filter_bank_scalar_panics_on_dimension_overflow() {
    let n_mels = usize::MAX / 4 + 1;
    let all_freqs = vec![0.0_f32; 4]; // n_freqs = 4 → product overflows
    let f_pts = vec![0.0_f32; 4];
    let mut out: Vec<f32> = Vec::new();
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows_scalar(spare, &all_freqs, &f_pts, n_mels);
  }

  // ── NEON-arm direct coverage (aarch64) ────────────────────────────
  //
  // The NEON kernel is `#[cfg(target_arch = "aarch64")]`, private, and
  // `#[target_feature(enable = "neon")]`. Its own precondition asserts
  // are defensive duplicates that the dispatcher's asserts shadow (the
  // dispatcher validates before dispatching), so they are only
  // reachable by calling `mel_filter_bank_rows_neon` directly. NEON is
  // baseline-present on every aarch64 host (Apple silicon included),
  // so the direct `unsafe` call is sound; the `is_neon_available()`
  // guard mirrors the sibling kernels' idiom and keeps these tests
  // green under `--cfg mlxrs_force_scalar` (where the dispatcher would
  // route to scalar but the CPU still physically has NEON).

  /// NEON zero-width row WITH a scalar tail: collapse row m=1 and use
  /// `n_freqs = 6` (`% 4 == 2`) so `body_len = 4` and the tail loop
  /// `for kk in body_len..n_freqs` writes the trailing 2 cells via the
  /// scalar `out[row_off + kk].write(0.0)` path. Covers both the
  /// 4-lane `vst1q_f32(_, zero)` body and the scalar zero tail.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn mel_filter_bank_neon_collapsed_row_with_tail_is_zero() {
    if !crate::simd::is_neon_available() {
      return;
    }
    let n_mels = 3usize;
    let n_freqs = 6usize; // body_len = 4, tail = 2
    let all_freqs: Vec<f32> = (0..n_freqs).map(|k| 100.0 * k as f32).collect();
    // f_pts[1] == f_pts[2] → row m=1 collapses (lc = 0).
    let f_pts = vec![0.0_f32, 500.0, 500.0, 1500.0, 2000.0];
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    // SAFETY: NEON checked above; out.len() == n_mels * n_freqs and
    // f_pts.len() == n_mels + 2; the kernel writes every slot.
    unsafe {
      super::mel_filter_bank_rows_neon(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
      out.set_len(n_mels * n_freqs);
    }
    for k in 0..n_freqs {
      assert_eq!(
        out[n_freqs + k],
        0.0,
        "NEON collapsed row m=1 cell k={k} should be 0.0 (incl. tail)"
      );
    }
  }

  /// NEON non-collapsed row through the direct kernel, with a tail
  /// (`n_freqs = 6`), checked against the hand-computed triangle so the
  /// 4-lane body AND scalar tail of the value path are exercised and
  /// pinned to the closed-form (not a self-comparison).
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn mel_filter_bank_neon_matches_handcomputed_triangle_with_tail() {
    if !crate::simd::is_neon_available() {
      return;
    }
    let n_mels = 1usize;
    let all_freqs = vec![0.0_f32, 100.0, 200.0, 300.0, 400.0, 500.0]; // 6 → tail of 2
    let n_freqs = all_freqs.len();
    let f_pts = vec![0.0_f32, 200.0, 400.0]; // triangle [0, 200, 400]
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    // SAFETY: NEON checked above; sizes match; kernel writes every slot.
    unsafe {
      super::mel_filter_bank_rows_neon(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
      out.set_len(n_mels * n_freqs);
    }
    let expected = [0.0_f32, 0.5, 1.0, 0.5, 0.0, 0.0];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
      // Reciprocal-hoist body vs subtract-then-divide tail differ by
      // <= 2 ULP from the closed form.
      assert!(
        (got - want).abs() <= 1e-6,
        "NEON hand-computed triangle mismatch i={i}: got={got} want={want}"
      );
    }
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "mel_filter_bank_rows_neon: out.len() (3) must equal n_mels * n_freqs")]
  fn mel_filter_bank_neon_panics_on_size_mismatch() {
    if !crate::simd::is_neon_available() {
      panic!(
        "mel_filter_bank_rows_neon: out.len() (3) must equal n_mels * n_freqs (skipped — NEON unavailable)"
      );
    }
    let all_freqs = vec![100.0_f32, 200.0, 300.0, 400.0]; // n_freqs = 4
    let f_pts = vec![0.0_f32, 200.0, 400.0, 600.0]; // n_mels = 2 → 4 pts
    let mut out: Vec<f32> = Vec::with_capacity(3); // WRONG: should be 8
    let spare = out.spare_capacity_mut();
    // SAFETY: NEON checked; expected-panic test on the intentional size
    // mismatch (precondition #2) before any pointer arithmetic.
    unsafe { super::mel_filter_bank_rows_neon(&mut spare[..3], &all_freqs, &f_pts, 2) };
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "mel_filter_bank_rows_neon: f_pts.len() (5) must equal n_mels + 2 (4)")]
  fn mel_filter_bank_neon_panics_on_f_pts_size_mismatch() {
    if !crate::simd::is_neon_available() {
      panic!(
        "mel_filter_bank_rows_neon: f_pts.len() (5) must equal n_mels + 2 (4) (skipped — NEON unavailable)"
      );
    }
    let all_freqs = vec![100.0_f32, 200.0, 300.0]; // n_freqs = 3
    let f_pts = vec![0.0_f32, 200.0, 400.0, 600.0, 800.0]; // n_mels = 2 expects 4
    let mut out: Vec<f32> = Vec::with_capacity(6);
    let spare = out.spare_capacity_mut();
    // SAFETY: NEON checked; expected-panic on the f_pts length
    // precondition, which fires before any pointer arithmetic.
    unsafe { super::mel_filter_bank_rows_neon(&mut spare[..6], &all_freqs, &f_pts, 2) };
  }

  /// NEON arm's `checked_mul` overflow guard. The explicit panic fires
  /// before the size-equality / `f_pts.len()` asserts, so the empty
  /// `out` and short `f_pts` never reach their checks.
  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "overflow usize")]
  fn mel_filter_bank_neon_panics_on_dimension_overflow() {
    if !crate::simd::is_neon_available() {
      panic!("dimensions overflow usize (skipped — NEON unavailable)");
    }
    let n_mels = usize::MAX / 4 + 1;
    let all_freqs = vec![0.0_f32; 4]; // n_freqs = 4 → product overflows
    let f_pts = vec![0.0_f32; 4];
    let mut out: Vec<f32> = Vec::new();
    let spare = out.spare_capacity_mut();
    // SAFETY: NEON checked; expected-panic on the checked_mul overflow,
    // which fires before any pointer arithmetic.
    unsafe { super::mel_filter_bank_rows_neon(spare, &all_freqs, &f_pts, n_mels) };
  }

  /// Dispatcher fall-through to the scalar arm with a collapsed row and
  /// a non-multiple-of-4 `n_freqs`. On a NEON host this routes through
  /// the NEON arm; under `--cfg mlxrs_force_scalar` (or a non-aarch64
  /// target) the dispatcher takes the scalar fall-through, exercising
  /// the dispatcher's final `mel_filter_bank_rows_scalar(...)` call and
  /// the scalar zero-width path through it. Asserted invariant (row is
  /// all-zero) holds on both arms.
  #[test]
  fn mel_filter_bank_dispatch_collapsed_row_with_tail_is_zero() {
    let n_mels = 3usize;
    let n_freqs = 6usize; // n_freqs % 4 == 2
    let all_freqs: Vec<f32> = (0..n_freqs).map(|k| 100.0 * k as f32).collect();
    let f_pts = vec![0.0_f32, 500.0, 500.0, 1500.0, 2000.0]; // row m=1 collapses
    let mut out: Vec<f32> = Vec::with_capacity(n_mels * n_freqs);
    let spare = out.spare_capacity_mut();
    mel_filter_bank_rows(&mut spare[..n_mels * n_freqs], &all_freqs, &f_pts, n_mels);
    // SAFETY: kernel writes every slot.
    unsafe { out.set_len(n_mels * n_freqs) };
    for k in 0..n_freqs {
      assert_eq!(
        out[n_freqs + k],
        0.0,
        "dispatch collapsed row m=1 cell k={k} should be 0.0"
      );
    }
  }
}
