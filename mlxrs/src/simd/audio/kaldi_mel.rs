//! `get_mel_banks_kaldi` triangle construction.
//!
//! Tracking: [#156](https://github.com/Findit-AI/mlxrs/issues/156).
//!
//! # The defect class
//!
//! The original `crate::audio::features::get_mel_banks_kaldi` has a per-row
//! inner loop that evaluates the Kaldi triangle membership for each
//! `(m, k)` cell — with an on-the-fly `mel_scale_kaldi` per cell:
//!
//! ```rust,ignore
//! for m in 0..num_bins {
//!   let left_mel   = mel_low + (m as f32) * mel_delta;
//!   let center_mel = mel_low + ((m + 1) as f32) * mel_delta;
//!   let right_mel  = mel_low + ((m + 2) as f32) * mel_delta;
//!   let lc = center_mel - left_mel;
//!   let cr = right_mel  - center_mel;
//!   if lc <= 0.0 || cr <= 0.0 { continue; }
//!   for k in 0..num_fft_bins {
//!     let mel = mel_scale_kaldi(fft_bin_width * k as f32);
//!     let up   = (mel - left_mel) / lc;
//!     let down = (right_mel - mel) / cr;
//!     let v    = up.min(down).max(0.0);
//!     bank[m * num_fft_bins + k] = v;
//!   }
//! }
//! ```
//!
//! `mel_scale_kaldi(hz) = 1127 * ln(1 + hz / 700)` — `ln` is the
//! per-cell hot spot. The mel values only depend on `k` (NOT on the
//! row `m`), so they can be **pre-computed once** outside the per-row
//! loop into a `Vec<f32>` of length `num_fft_bins`. After that
//! pre-pass, the per-row inner loop is the same 4-lane triangle shape
//! as the mel-triangle kernel (`crate::simd::audio::mel_triangle`) — `(mel - left_mel) /
//! lc` and `(right_mel - mel) / cr`.
//!
//! # The fix — two-stage kernel
//!
//! 1. **Pre-pass** — scalar `mel_values[k] = mel_scale_kaldi(fft_bin_width * k as f32)`
//!    for `k` in `[0, num_fft_bins)`. The `ln` cannot vectorize
//!    cleanly without a custom polynomial, and the pre-pass is
//!    `O(num_fft_bins)` — small (typically 200 cells) and runs once
//!    per `(sample_freq, n_fft_padded)` pair.
//! 2. **Per-row 4-lane tile** — identical to the mel-triangle kernel:
//!    - Pre-compute reciprocals (`inv_lc`, `inv_cr`, `left_over_lc`,
//!      `right_over_cr`) per row.
//!    - 4-lane NEON FMA over `mel_values`.
//!    - `vminq_f32` → `vmaxq_f32(_, 0)` → `vst1q_f32`.
//! 3. Tail (`num_fft_bins % 4` ≤ 3) handled by the scalar arm at the
//!    end of each row.
//!
//! Zero-width triangles (`lc <= 0.0` or `cr <= 0.0`) — the kernel
//! writes 0.0 to the whole row. (See `crate::simd::audio::mel_triangle`
//! for the matching rationale.)
//!
//! # Correctness class — `Tolerance`
//!
//! Two sources of divergence:
//!
//! 1. The reciprocal-hoist substitution `(mel - left_mel) / lc →
//!    mel * inv_lc - left_mel * inv_lc` adds ~2 ULP per cell vs the
//!    scalar arm's direct subtract-then-divide (same as the mel-triangle kernel).
//! 2. The pre-pass scalar `mel_values[k]` calls `f32::ln` (libm) which
//!    is the same call the scalar reference would inline per cell.
//!    Bit-equal up to the pre-pass + cell-evaluation ordering — `mel`
//!    is exactly the same value either way.
//!
//! Per-cell worst case empirically: ~30 ULP (~3e-6 absolute) in the
//! cell-by-cell comparison — wider than the mel-triangle kernel's ~2 ULP because the
//! per-row `inv_lc = 1.0 / lc` reciprocal compounds its own ULP error
//! into the subsequent `mel * inv_lc - left_mel * inv_lc` FMA chain
//! (whereas the scalar arm's `(mel - left_mel) / lc` is a single
//! subtract + single divide). Differential test tolerance: `abs =
//! 1e-5, rel = 1e-5` (well under any sane filterbank-output use).
//!
//! # `Vec<f32>` output API
//!
//! Matches `get_mel_banks_kaldi`'s allocation discipline — the
//! dispatcher writes into a pre-reserved `&mut [MaybeUninit<f32>]`
//! (sized to `num_bins * num_fft_bins`), and the caller wraps it
//! with `Vec::with_capacity` + `spare_capacity_mut` + `set_len`.

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  vdupq_n_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
};

use crate::error::{Error, Result};

/// Kaldi `mel_scale_kaldi(hz) = 1127 * ln(1 + hz / 700)` (mirrors
/// `crate::audio::features::mel_scale_kaldi` — kept local so this
/// module is self-contained without a public re-export).
#[inline]
fn mel_scale_kaldi(hz: f32) -> f32 {
  1127.0_f32 * (1.0_f32 + hz / 700.0_f32).ln()
}

/// Scalar reference: build a `(num_bins, num_fft_bins)` Kaldi mel
/// filterbank into `out` from `fft_bin_width`, `mel_low`, and
/// `mel_delta`. Bit-exact match for the original
/// `get_mel_banks_kaldi` inner two-loop **plus** explicit 0.0 writes
/// for the `lc <= 0` / `cr <= 0` zero-width-triangle rows.
///
/// `mel_low + i * mel_delta` is the mel-domain edge for row `i`;
/// `fft_bin_width = sample_freq / n_fft_padded` is the spacing of the
/// FFT bins in Hz.
///
/// # Preconditions
///
/// - `out.len() == num_bins * num_fft_bins`.
/// - `num_bins * num_fft_bins` does not overflow `usize` — checked
///   explicitly via `checked_mul` rather than wrapping silently.
///
/// All asserted **unconditionally** (release-too).
///
/// # Panics
///
/// Panics explicitly (not silently wraps) on `num_bins * num_fft_bins`
/// `usize` overflow — the only correct response when caller dims
/// cannot fit a contiguous buffer, since silently wrapping would let
/// an under-sized buffer satisfy the size-equality assertion and reach
/// the per-cell init loop.
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns.
#[inline]
#[doc(hidden)]
pub fn get_mel_banks_kaldi_scalar(
  out: &mut [MaybeUninit<f32>],
  num_bins: usize,
  num_fft_bins: usize,
  fft_bin_width: f32,
  mel_low: f32,
  mel_delta: f32,
) {
  let elements = num_bins.checked_mul(num_fft_bins).unwrap_or_else(|| {
    panic!("get_mel_banks_kaldi_scalar: dimensions {num_bins}x{num_fft_bins} overflow usize")
  });
  assert_eq!(
    out.len(),
    elements,
    "get_mel_banks_kaldi_scalar: out.len() ({}) must equal num_bins * num_fft_bins ({} * {} = {})",
    out.len(),
    num_bins,
    num_fft_bins,
    elements,
  );

  for m in 0..num_bins {
    let left_mel = mel_low + (m as f32) * mel_delta;
    let center_mel = mel_low + ((m + 1) as f32) * mel_delta;
    let right_mel = mel_low + ((m + 2) as f32) * mel_delta;
    let lc = center_mel - left_mel;
    let cr = right_mel - center_mel;
    let row_off = m * num_fft_bins;
    if lc <= 0.0 || cr <= 0.0 {
      for k in 0..num_fft_bins {
        out[row_off + k].write(0.0);
      }
      continue;
    }
    for k in 0..num_fft_bins {
      let mel = mel_scale_kaldi(fft_bin_width * k as f32);
      let up = (mel - left_mel) / lc;
      let down = (right_mel - mel) / cr;
      let v = up.min(down).max(0.0);
      out[row_off + k].write(v);
    }
  }
}

/// NEON 4-lane Kaldi mel filterbank row builder. Two-stage: scalar
/// pre-pass builds `mel_values[k] = mel_scale_kaldi(fft_bin_width *
/// k)`; then per-row 4-lane NEON tile over `mel_values`.
///
/// `mel_values` is heap-allocated once per call (length `num_fft_bins`)
/// via fallible `try_reserve_exact` — matches the public
/// [`get_mel_banks_kaldi_rows`] dispatcher's allocation discipline and
/// the wider crate convention that request-scaled allocations surface
/// as recoverable [`Error::OutOfMemory`] rather than aborting the
/// process. Without this, an adversarial / fuzzer-supplied
/// `num_fft_bins` could trigger an infallible `Vec::with_capacity`
/// abort in the NEON path while the scalar fallback would have
/// returned `Err`.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    discharged by [`get_mel_banks_kaldi_rows`].
/// 2. `out.len() == num_bins * num_fft_bins` — asserted
///    **unconditionally** here.
/// 3. `num_bins * num_fft_bins` does not overflow `usize` — checked
///    via `checked_mul` BEFORE the size-equality assertion, so a
///    wrapped product can never sneak past the size check and let the
///    NEON kernel compute per-row offsets `m * num_fft_bins` from
///    unwrapped loop dims (which would issue out-of-bounds
///    `vst1q_f32` / `vld1q_f32`).
///
/// # Errors
///
/// - [`Error::OutOfMemory`] if reserving the `num_fft_bins`-length
///   `mel_values` cache fails.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn get_mel_banks_kaldi_neon(
  out: &mut [MaybeUninit<f32>],
  num_bins: usize,
  num_fft_bins: usize,
  fft_bin_width: f32,
  mel_low: f32,
  mel_delta: f32,
) -> Result<()> {
  let elements = num_bins.checked_mul(num_fft_bins).unwrap_or_else(|| {
    panic!("get_mel_banks_kaldi_neon: dimensions {num_bins}x{num_fft_bins} overflow usize")
  });
  assert_eq!(
    out.len(),
    elements,
    "get_mel_banks_kaldi_neon: out.len() ({}) must equal num_bins * num_fft_bins ({} * {} = {})",
    out.len(),
    num_bins,
    num_fft_bins,
    elements,
  );

  // Stage 1: scalar pre-pass for `mel_values[k]`. Fallible allocation
  // (`try_reserve_exact`) matches the public dispatcher's contract —
  // the scalar arm of the wider `get_mel_banks_kaldi` path returns
  // `Err(Error::OutOfMemory)` for oversized inputs; this NEON path
  // would previously have aborted via `Vec::with_capacity`.
  let mut mel_values: Vec<f32> = Vec::new();
  mel_values
    .try_reserve_exact(num_fft_bins)
    .map_err(|_| Error::OutOfMemory)?;
  for k in 0..num_fft_bins {
    mel_values.push(mel_scale_kaldi(fft_bin_width * k as f32));
  }

  // Stage 2: 4-lane NEON row tile.
  let body_len = num_fft_bins - (num_fft_bins % 4);
  let zero = vdupq_n_f32(0.0);

  // SAFETY: each `vst1q_f32` writes 16 bytes (4 f32 slots of
  // `MaybeUninit<f32>`) per tile at `dst_base.add(row_off + k)` where
  // `row_off + k + 4 <= row_off + body_len <= row_off + num_fft_bins
  // <= num_bins * num_fft_bins = out.len()`. The `vld1q_f32` reads 4
  // mels from `mel_base.add(k)` where `k + 4 <= body_len <=
  // num_fft_bins = mel_values.len()`. Both within bounds. Stores
  // target `MaybeUninit<f32>` backing memory, which has no validity
  // invariants beyond size + alignment. NEON availability is the
  // caller's obligation (precondition #1).
  unsafe {
    let dst_base = out.as_mut_ptr().cast::<f32>();
    let mel_base = mel_values.as_ptr();

    for m in 0..num_bins {
      let left_mel = mel_low + (m as f32) * mel_delta;
      let center_mel = mel_low + ((m + 1) as f32) * mel_delta;
      let right_mel = mel_low + ((m + 2) as f32) * mel_delta;
      let lc = center_mel - left_mel;
      let cr = right_mel - center_mel;
      let row_off = m * num_fft_bins;

      if lc <= 0.0 || cr <= 0.0 {
        let mut k = 0usize;
        while k + 4 <= body_len {
          vst1q_f32(dst_base.add(row_off + k), zero);
          k += 4;
        }
        for kk in body_len..num_fft_bins {
          out[row_off + kk].write(0.0);
        }
        continue;
      }

      let inv_lc = 1.0_f32 / lc;
      let inv_cr = 1.0_f32 / cr;
      let left_over_lc = left_mel * inv_lc;
      let right_over_cr = right_mel * inv_cr;

      let inv_lc_v = vdupq_n_f32(inv_lc);
      let inv_cr_v = vdupq_n_f32(inv_cr);
      let left_over_lc_v = vdupq_n_f32(left_over_lc);
      let right_over_cr_v = vdupq_n_f32(right_over_cr);

      let mut k = 0usize;
      while k + 4 <= body_len {
        let mel_v = vld1q_f32(mel_base.add(k));

        // up = mel * inv_lc - left_over_lc
        let up = vsubq_f32(vmulq_f32(mel_v, inv_lc_v), left_over_lc_v);
        // down = right_over_cr - mel * inv_cr
        let prod = vmulq_f32(mel_v, inv_cr_v);
        let down = vsubq_f32(right_over_cr_v, prod);

        let mn = vminq_f32(up, down);
        let v = vmaxq_f32(mn, zero);

        vst1q_f32(dst_base.add(row_off + k), v);

        k += 4;
      }

      // Tail — scalar shape (subtract-then-divide) for bit-equality
      // with the scalar arm at boundaries.
      for kk in body_len..num_fft_bins {
        let mel = mel_values[kk];
        let up = (mel - left_mel) / lc;
        let down = (right_mel - mel) / cr;
        let v = up.min(down).max(0.0);
        out[row_off + kk].write(v);
      }
    }
  }
  Ok(())
}

/// Public dispatcher: build a `(num_bins, num_fft_bins)` Kaldi mel
/// filterbank into `out`. Routes to NEON on `aarch64` when NEON is
/// reported, else to the scalar reference.
///
/// Used by [`crate::audio::features::get_mel_banks_kaldi`] to fill
/// the pre-reserved bank buffer.
///
/// # Preconditions
///
/// - `out.len() == num_bins * num_fft_bins` — asserted unconditionally.
/// - `num_bins * num_fft_bins` does not overflow `usize` — checked
///   via `checked_mul` BEFORE the size-equality assertion, so a
///   wrapped product can never sneak past the size check and let the
///   inner kernels (scalar / NEON) compute per-row offsets from
///   unwrapped loop dims (same defect class as `rotate_buf_u8`).
///
/// # Panics
///
/// Panics explicitly (not silently wraps) on `num_bins * num_fft_bins`
/// `usize` overflow. Signature stays infallible-with-panic on
/// overflow (no `Result` lift) — the existing `Result<()>` only
/// carries the NEON arm's `Error::OutOfMemory` for the `mel_values`
/// cache.
///
/// # Initialization contract
///
/// **Every f32 of `out` is written before this returns** (zero-width
/// rows write 0.0; non-zero rows write the per-cell triangle value)
/// — provided this returns `Ok(())`. On `Err`, no init guarantee.
///
/// # Errors
///
/// - [`Error::OutOfMemory`] if the NEON arm's internal `mel_values`
///   cache (length `num_fft_bins`) cannot be reserved. The scalar arm
///   does not allocate, so it is infallible — wrapped in `Ok(())` for
///   signature parity.
///
/// # Correctness class
///
/// `Tolerance` (`abs = 1e-5, rel = 1e-5`) — the reciprocal-hoist
/// rearrangement and the pre-pass `mel_values` cache introduce at
/// most ~30 ULP per cell relative to the scalar arm's direct
/// subtract-then-divide (the compounding of `inv_lc = 1.0 / lc`'s
/// own ULP into the FMA chain).
#[inline]
#[doc(hidden)]
pub fn get_mel_banks_kaldi_rows(
  out: &mut [MaybeUninit<f32>],
  num_bins: usize,
  num_fft_bins: usize,
  fft_bin_width: f32,
  mel_low: f32,
  mel_delta: f32,
) -> Result<()> {
  // Checked dimension math BEFORE the size-equality assertion: wrapping
  // `num_bins * num_fft_bins` in release mode could otherwise produce a
  // small `elements` that an under-sized `out` would satisfy, letting
  // either inner kernel compute per-row offsets from unwrapped loop
  // dims (same defect class as `rotate_buf_u8`).
  let elements = num_bins.checked_mul(num_fft_bins).unwrap_or_else(|| {
    panic!(
      "simd::audio::get_mel_banks_kaldi_rows: dimensions {num_bins}x{num_fft_bins} overflow usize"
    )
  });
  assert_eq!(
    out.len(),
    elements,
    "simd::audio::get_mel_banks_kaldi_rows: out.len() ({}) must equal num_bins * num_fft_bins \
     ({} * {} = {})",
    out.len(),
    num_bins,
    num_fft_bins,
    elements,
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: NEON gated; slice-length precondition asserted above;
      // the kernel writes every cell of `out` on success.
      unsafe {
        return get_mel_banks_kaldi_neon(
          out,
          num_bins,
          num_fft_bins,
          fft_bin_width,
          mel_low,
          mel_delta,
        );
      }
    }
  }
  get_mel_banks_kaldi_scalar(
    out,
    num_bins,
    num_fft_bins,
    fft_bin_width,
    mel_low,
    mel_delta,
  );
  Ok(())
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher Tolerance differential tests + edge coverage
  //! for the kaldi mel triangle.

  use super::{get_mel_banks_kaldi_rows, get_mel_banks_kaldi_scalar};

  /// Realistic parameters: sample_freq = 16000 Hz, n_fft_padded = 512
  /// → num_fft_bins = 256, fft_bin_width = 31.25 Hz; mel range
  /// `[mel_scale_kaldi(20), mel_scale_kaldi(7800)]`.
  fn realistic_params(num_bins: usize) -> (usize, f32, f32, f32) {
    let n_fft_padded = 512usize;
    let num_fft_bins = n_fft_padded / 2;
    let sample_freq = 16_000.0_f32;
    let fft_bin_width = sample_freq / n_fft_padded as f32;
    // Kaldi: mel_low = mel_scale_kaldi(20), mel_high = mel_scale_kaldi(7800)
    let mel_low = 1127.0_f32 * (1.0_f32 + 20.0_f32 / 700.0_f32).ln();
    let mel_high = 1127.0_f32 * (1.0_f32 + 7800.0_f32 / 700.0_f32).ln();
    let mel_delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);
    (num_fft_bins, fft_bin_width, mel_low, mel_delta)
  }

  fn bank_via_scalar(num_bins: usize) -> Vec<f32> {
    let (num_fft_bins, w, mel_low, mel_delta) = realistic_params(num_bins);
    let mut out: Vec<f32> = Vec::with_capacity(num_bins * num_fft_bins);
    let spare = out.spare_capacity_mut();
    get_mel_banks_kaldi_scalar(
      &mut spare[..num_bins * num_fft_bins],
      num_bins,
      num_fft_bins,
      w,
      mel_low,
      mel_delta,
    );
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(num_bins * num_fft_bins) };
    out
  }

  fn bank_via_dispatch(num_bins: usize) -> Vec<f32> {
    let (num_fft_bins, w, mel_low, mel_delta) = realistic_params(num_bins);
    let mut out: Vec<f32> = Vec::with_capacity(num_bins * num_fft_bins);
    let spare = out.spare_capacity_mut();
    get_mel_banks_kaldi_rows(
      &mut spare[..num_bins * num_fft_bins],
      num_bins,
      num_fft_bins,
      w,
      mel_low,
      mel_delta,
    )
    .expect("realistic params should not OOM");
    // SAFETY: kernel contract initializes every slot.
    unsafe { out.set_len(num_bins * num_fft_bins) };
    out
  }

  #[test]
  fn kaldi_mel_scalar_matches_dispatcher_tolerance() {
    // Sweep over typical Kaldi num_bins values.
    for &num_bins in &[5usize, 10, 23, 40, 80] {
      let s = bank_via_scalar(num_bins);
      let d = bank_via_dispatch(num_bins);
      assert_eq!(s.len(), d.len(), "shape parity at num_bins={num_bins}");
      for (i, (a, b)) in s.iter().zip(d.iter()).enumerate() {
        let diff = (a - b).abs();
        let tol = 1e-5_f32.max(1e-5_f32 * a.abs());
        assert!(
          diff <= tol,
          "Tolerance mismatch at num_bins={num_bins} i={i}: scalar={a} dispatcher={b} \
           diff={diff} tol={tol}"
        );
      }
    }
  }

  #[test]
  fn kaldi_mel_triangle_shape() {
    // Sanity: every cell is in [0, 1].
    let num_bins = 23;
    let bank = bank_via_dispatch(num_bins);
    let (num_fft_bins, _, _, _) = realistic_params(num_bins);
    for m in 0..num_bins {
      let row = &bank[m * num_fft_bins..(m + 1) * num_fft_bins];
      for (i, &v) in row.iter().enumerate() {
        assert!(
          (0.0..=1.0001).contains(&v),
          "cell out of [0, 1]: m={m} i={i} v={v}"
        );
      }
    }
  }

  #[test]
  fn kaldi_mel_collapsed_row_is_zero() {
    // Force `lc <= 0` by setting `mel_delta = 0`. Every row collapses
    // (left_mel == center_mel == right_mel), so every cell is 0.0.
    let num_bins = 4;
    let num_fft_bins = 16;
    let fft_bin_width = 30.0_f32;
    let mel_low = 100.0_f32;
    let mel_delta = 0.0_f32;
    let mut out: Vec<f32> = Vec::with_capacity(num_bins * num_fft_bins);
    let spare = out.spare_capacity_mut();
    get_mel_banks_kaldi_rows(
      &mut spare[..num_bins * num_fft_bins],
      num_bins,
      num_fft_bins,
      fft_bin_width,
      mel_low,
      mel_delta,
    )
    .expect("small collapsed-row params should not OOM");
    // SAFETY: kernel writes every slot.
    unsafe { out.set_len(num_bins * num_fft_bins) };
    for (i, &v) in out.iter().enumerate() {
      assert_eq!(v, 0.0, "collapsed row cell i={i} should be 0.0");
    }
  }

  #[test]
  #[should_panic(
    expected = "simd::audio::get_mel_banks_kaldi_rows: out.len() (3) must equal num_bins"
  )]
  fn kaldi_mel_panics_on_size_mismatch() {
    let mut out: Vec<f32> = Vec::with_capacity(3); // WRONG
    let spare = out.spare_capacity_mut();
    // The pre-alloc size-equality assertion fires before any Result
    // can be returned; the `let _ =` keeps the test's intent clear
    // (we are asserting on the panic, not on the Result value).
    let _ = get_mel_banks_kaldi_rows(&mut spare[..3], 2, 4, 30.0, 100.0, 50.0);
  }

  /// Structural assertion: the dispatcher returns `Result<()>` so the
  /// NEON arm's request-scaled `mel_values` allocation can surface as
  /// recoverable [`Error::OutOfMemory`] instead of an infallible
  /// `Vec::with_capacity` abort. Without this signature shape the
  /// NEON path could abort while the scalar-arm path returned `Err`
  /// — an inconsistency that breaks the wider crate's
  /// allocation-discipline convention.
  ///
  /// (We cannot deterministically synthesize a `num_fft_bins` that
  /// OOMs on this host without overrunning the test process's address
  /// space; the contract is enforced at the type level — the call
  /// below MUST be a `Result` for the test to compile.)
  #[test]
  fn kaldi_mel_dispatcher_returns_result_for_fallible_allocation() {
    let num_bins = 2usize;
    let num_fft_bins = 4usize;
    let mut out: Vec<f32> = Vec::with_capacity(num_bins * num_fft_bins);
    let spare = out.spare_capacity_mut();
    let r: Result<(), super::Error> = get_mel_banks_kaldi_rows(
      &mut spare[..num_bins * num_fft_bins],
      num_bins,
      num_fft_bins,
      30.0,
      100.0,
      50.0,
    );
    assert!(r.is_ok(), "small input should not OOM");
  }

  /// Wrap-arith defence (same defect class as `rotate_buf_u8`):
  /// `num_bins * num_fft_bins` MUST be evaluated via `checked_mul` —
  /// a wrapping multiply in release mode would otherwise let an
  /// under-sized (here zero-length) `out` satisfy the size-equality
  /// assertion (`out.len() == wrapped_elements`), reaching either
  /// inner kernel where per-row offsets (`m * num_fft_bins`) would
  /// then be computed from the unwrapped loop dims and produce
  /// out-of-bounds NEON stores. The adversarial case:
  /// `num_bins = usize::MAX / 4 + 1, num_fft_bins = 4` wraps the
  /// product to `4` (one slot per bin would no longer be enough; the
  /// wrapped value is in fact `0` on 64-bit hosts where the product
  /// overflows the high bit).
  ///
  /// `checked_mul` is therefore the only correct gate; this test
  /// pins the explicit panic message so a future refactor cannot
  /// regress to a plain multiply.
  #[test]
  #[should_panic(expected = "overflow usize")]
  fn kaldi_mel_panics_on_dimension_overflow() {
    let num_bins = usize::MAX / 4 + 1;
    let num_fft_bins = 4usize;
    // `out.len() == 0` deliberately — if the wrapping multiply went
    // unchecked it would produce a small `elements` that this
    // zero-length slice could satisfy, letting the inner loop reach.
    let mut out: Vec<f32> = Vec::new();
    let spare = out.spare_capacity_mut();
    let _ = get_mel_banks_kaldi_rows(spare, num_bins, num_fft_bins, 30.0, 100.0, 50.0);
  }
}
