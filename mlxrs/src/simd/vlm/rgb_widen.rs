//! C3 — `image_to_array` RGB u8 → f32 widen (no swap): de-interleave
//! a `&[u8]` of packed RGB pixels into a `&mut [MaybeUninit<f32>]`
//! channel-last `[R, G, B]` f32 triples (same plane order as input).
//!
//! Tracking: [#148](https://github.com/Findit-AI/mlxrs/issues/148).
//! Plan: `docs/core-arch-simd-candidates.md` §2 row C3, §3.3 (C3/C4
//! image-to-array widen). The §5.5 doc originally deferred C3 pending
//! a disassembly check on the auto-vectorized `buf.extend(raw.iter().map(...))`
//! scalar shape; per the user directive 2026-05-23 (project memory
//! rule **"SIMD ship NEON regardless"**), the NEON kernel ships
//! unconditionally regardless of how the scalar bench compares.
//!
//! # The defect class
//!
//! The pre-C3 [`crate::vlm::image::image_to_array`] RGB arm is:
//!
//! ```rust,ignore
//! ColorOrder::Rgb => {
//!   buf.extend(raw.iter().map(|&b| f32::from(b)));
//! }
//! ```
//!
//! A `Vec::extend` over an iterator's `f32::from(u8)` per byte. LLVM
//! auto-vectorizes this shape on aarch64 (`vld1q_u8` + widen + `vst1q_f32`
//! ×4 per 16-byte tile), but the auto-vec path is compiler-version-
//! dependent and can regress silently on a rustc upgrade or a stylistic
//! refactor. Shipping a hand-rolled NEON arm pins the contract.
//!
//! # Two layered fixes — the scalar restructure + the NEON kernel
//!
//! 1. **Scalar restructure** — replace the `Vec::extend(map)` with a
//!    `chunks_exact_mut(1) + iter()` pair-iteration into pre-reserved
//!    spare capacity using `MaybeUninit::write`. Each iteration widens
//!    one byte to one f32. Same shape as C4's scalar restructure
//!    (sized destination, no per-iteration bounds check on `Vec` growth).
//! 2. **Hand-rolled NEON kernel** — `vld1q_u8` (16 bytes per load) +
//!    widen chain `vmovl_u8` (low/high) + `vmovl_u16` (low/high) +
//!    `vcvtq_f32_u32` → four `float32x4_t` outputs + four `vst1q_f32`
//!    stores per 16-byte tile. No de-interleave needed (the input
//!    layout matches the output). Tail (`len % 16` bytes ≤ 15) is
//!    handled by the scalar arm.
//!
//! Unlike C4 (BGR), there's no R↔B swap; the widen is a pure
//! u8 → f32 cast applied to every byte in source order. The
//! 16-bytes-per-tile NEON loop is simpler than C4's 16-pixels-per-tile
//! (which needed `vld3q_u8` + permuted `vst3q_f32`).
//!
//! # Correctness class — `Exact`
//!
//! Pure data movement plus a lossless u8 → f32 widen (every u8 is
//! exactly representable in f32). The scalar arm and the NEON arm
//! produce bit-identical output for every input — both perform the
//! same `f32::from(u8)` widen via `vcvtq_f32_u32` (lossless because
//! the source u8 is in `[0, 255]`, exactly representable in f32) and
//! write the same per-byte f32. Differential tests use
//! [`crate::simd::diff::assert_eq_over_lane_sweep`].
//!
//! # `MaybeUninit<f32>` API — type-encoded uninit safety
//!
//! Matches C4: takes `&mut [MaybeUninit<f32>]` so the
//! `image_to_array` call site can pass `Vec::spare_capacity_mut()`
//! directly and `set_len(total)` after every f32 has been written.
//! No `from_raw_parts_mut` cast over uninit memory.
//!
//! # Verify-before-claim bench
//!
//! Bench numbers are **report-only** per the user directive 2026-05-23
//! (project memory rule **"SIMD ship NEON regardless"**). The bench
//! (`mlxrs/benches/simd_rgb_widen.rs`) exists as a regression guard
//! against both a future scalar regression and a future NEON regression.
//!
//! # No new dependencies
//!
//! Pure `core::slice` + `core::arch::aarch64` + `core::mem::MaybeUninit`.

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  vcvtq_f32_u32, vget_low_u8, vget_low_u16, vld1q_u8, vmovl_high_u8, vmovl_high_u16, vmovl_u8,
  vmovl_u16, vst1q_f32,
};

/// Widen a packed RGB `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]` (no R↔B swap; one f32 per input byte).
/// Scalar reference — the bit-exact oracle for the NEON dispatcher
/// and the fallback path on every non-`aarch64` target.
///
/// **Always compiled** — independent of `target_arch`. Anchors the
/// math contract (each input byte `src[i]` produces `out[i] =
/// f32::from(src[i])`), is the differential-test oracle, and is the
/// fallback path.
///
/// # Preconditions
///
/// - `out.len() == src.len()` (one output f32 per input byte).
///
/// Asserted **unconditionally** (release-too). The function is `pub`,
/// reachable through `simd::vlm::rgb_widen`, and its initialization
/// contract is load-bearing for callers that then call `Vec::set_len`
/// over the covered region.
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns.
#[inline]
#[doc(hidden)]
pub fn rgb_widen_scalar(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  assert_eq!(
    out.len(),
    src.len(),
    "rgb_widen_scalar: out.len() ({}) must equal src.len() ({}) (one output f32 per input byte)",
    out.len(),
    src.len(),
  );
  // Per-byte widen into the pre-reserved slice. Sized-destination shape
  // matches C4's scalar arm (LLVM auto-vectorizes this on aarch64 once
  // the destination is a fixed-size slice rather than a `Vec` growing
  // through `extend`).
  for (slot, &b) in out.iter_mut().zip(src.iter()) {
    slot.write(f32::from(b));
  }
}

/// Widen a packed RGB `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]`. NEON 16-byte `vld1q_u8` + four
/// `vst1q_f32` tile.
///
/// # Algorithm
///
/// Per 16-byte tile:
/// 1. Load 16 bytes via `vld1q_u8` (no de-interleave — input order is
///    output order for the no-swap RGB path).
/// 2. Widen to four `float32x4_t` lanes via the chain
///    `vmovl_u8` (low/high) → `vmovl_u16` (low/high) → `vcvtq_f32_u32`
///    (lossless for u8 → f32).
/// 3. Four `vst1q_f32` stores per tile (16 f32 = 64 bytes output).
///
/// Tail (`src.len() % 16` bytes ≤ 15) is delegated to the scalar arm.
///
/// # Initialization contract
///
/// Every f32 of `out` is written before this returns.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    discharged by [`rgb_widen`]'s `is_neon_available()` gate.
/// 2. `out.len() == src.len()` — asserted **unconditionally** here.
///
/// `vld1q_u8`/`vst1q_f32` accept unaligned addresses at full throughput
/// on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb_widen_neon(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  assert_eq!(
    out.len(),
    src.len(),
    "rgb_widen_neon: out.len() ({}) must equal src.len() ({}) (one output f32 per input byte)",
    out.len(),
    src.len(),
  );

  let n = src.len();
  let body_len = n - (n % 16);

  // SAFETY: the body loop reads 16 bytes via `vld1q_u8` from
  // `src.as_ptr().add(i)` for `i + 16 <= body_len <= src.len()` —
  // within bounds. It writes four `vst1q_f32` (16 f32 = 64 bytes) per
  // tile to `out.as_mut_ptr().cast::<f32>().add(i)` for the same `i`
  // — i.e. `i + 16 f32 <= body_len <= out.len()` (slot count of
  // MaybeUninit<f32>), within bounds. Stores target `MaybeUninit<f32>`
  // backing memory, which has no validity invariants beyond size +
  // alignment and accepts any bit pattern — raw-pointer writes via
  // `vst1q_f32` are sound. NEON availability is the caller's
  // obligation (precondition #1 — discharged by the dispatcher).
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = out.as_mut_ptr().cast::<f32>();

    let mut i = 0usize;
    while i + 16 <= body_len {
      // Load 16 bytes (no de-interleave — RGB-source-to-RGB-output
      // is a 1:1 byte-for-byte widen).
      let v = vld1q_u8(src_base.add(i));

      // Widen u8x16 → two u16x8 (low/high) → four u32x4 → four
      // f32x4. Lossless because u8 ∈ [0, 255] is exactly
      // representable in f32 (mantissa has 24 bits).
      let v_lo16 = vmovl_u8(vget_low_u8(v));
      let v_hi16 = vmovl_high_u8(v);
      let v_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_lo16)));
      let v_f1 = vcvtq_f32_u32(vmovl_high_u16(v_lo16));
      let v_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_hi16)));
      let v_f3 = vcvtq_f32_u32(vmovl_high_u16(v_hi16));

      // Four contiguous 4-lane f32 stores = 16 f32 = 64 bytes per
      // tile, exactly the per-tile output budget.
      vst1q_f32(dst_base.add(i), v_f0);
      vst1q_f32(dst_base.add(i + 4), v_f1);
      vst1q_f32(dst_base.add(i + 8), v_f2);
      vst1q_f32(dst_base.add(i + 12), v_f3);

      i += 16;
    }
  }

  // Tail: `len % 16` bytes (≤ 15). Delegate to the scalar arm — both
  // arms produce bit-identical output.
  if body_len < n {
    rgb_widen_scalar(&mut out[body_len..], &src[body_len..]);
  }
}

/// Widen a packed RGB `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]` (no R↔B swap). Routes to NEON on
/// `aarch64` (when the CPU reports NEON), else to [`rgb_widen_scalar`].
///
/// Used by [`crate::vlm::image::image_to_array`] for the
/// [`crate::vlm::image::ColorOrder::Rgb`] arm of `as_rgb8()` sources.
///
/// # Preconditions
///
/// - `out.len() == src.len()` — asserted **unconditionally**.
///
/// # Initialization contract
///
/// **Every f32 of `out` is written before this returns.**
///
/// # Correctness class
///
/// `Exact` — bit-identical scalar vs NEON output (pure data movement +
/// lossless u8 → f32 widen). See module-level "Correctness class"
/// section.
#[inline]
#[doc(hidden)]
pub fn rgb_widen(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  assert_eq!(
    out.len(),
    src.len(),
    "simd::vlm::rgb_widen: out.len() ({}) must equal src.len() ({})",
    out.len(),
    src.len(),
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: `is_neon_available()` confirmed NEON is on this CPU
      // (precondition #1 of `rgb_widen_neon`). The slice-length
      // precondition (#2) was just asserted unconditionally above.
      // The kernel writes every f32 of `out` before returning per its
      // function-level contract.
      unsafe { rgb_widen_neon(out, src) };
      return;
    }
  }
  rgb_widen_scalar(out, src);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher + scalar vs NEON differential tests + edge
  //! coverage for C3.

  use core::mem::MaybeUninit;

  use super::{rgb_widen, rgb_widen_scalar};
  use crate::simd::diff::{assert_eq_over_lane_sweep, lane_sweep_lengths};

  /// Test adapter — call the scalar kernel on `src.len()` slots of
  /// uninit `Vec<f32>` spare capacity.
  fn rgb_widen_scalar_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    rgb_widen_scalar(&mut spare[..n], src);
    // SAFETY: kernel contract initializes every slot; `n <= capacity`.
    unsafe { v.set_len(n) };
    v
  }

  /// Test adapter — dispatcher version.
  fn rgb_widen_dispatch_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    rgb_widen(&mut spare[..n], src);
    // SAFETY: kernel contract initializes every slot; `n <= capacity`.
    unsafe { v.set_len(n) };
    v
  }

  /// Direct NEON-arm adapter, aarch64-only.
  #[cfg(target_arch = "aarch64")]
  fn rgb_widen_neon_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    // SAFETY: caller guards on `is_neon_available()`; size is `n`;
    // kernel initializes every slot.
    unsafe {
      super::rgb_widen_neon(&mut spare[..n], src);
      v.set_len(n);
    }
    v
  }

  /// Deterministic input generator — `(i * 7) % 256` per byte, so
  /// every byte is non-uniform across consecutive positions (any
  /// stride / off-by-one bug visible).
  fn gen_rgb_bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 7) % 256) as u8).collect()
  }

  /// `Exact` differential — scalar vs dispatcher over the lane sweep
  /// at `lanes = 16` (matches the NEON 16-byte tile width).
  #[test]
  fn rgb_widen_scalar_matches_dispatcher_exact() {
    assert_eq_over_lane_sweep(
      16,
      rgb_widen_scalar_init,
      rgb_widen_dispatch_init,
      gen_rgb_bytes,
    );
  }

  /// NEON-vs-scalar bit-identical assertion via direct kernel call.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn rgb_widen_neon_matches_scalar_bit_identical() {
    if !crate::simd::is_neon_available() {
      return;
    }
    for &n in &[
      0usize, 1, 15, 16, 17, 31, 32, 33, 48, 49, 64, 100, 1024, 4096,
    ] {
      let src = gen_rgb_bytes(n);
      let scalar = rgb_widen_scalar_init(&src);
      let neon = rgb_widen_neon_init(&src);
      assert_eq!(neon, scalar, "rgb_widen_neon vs scalar differ at n={n}");
    }
  }

  /// Lane-sweep covers C3-relevant boundary lengths.
  #[test]
  fn rgb_widen_lane_sweep_covers_tile_boundaries() {
    let sweep = lane_sweep_lengths(16);
    assert_eq!(sweep, [0, 1, 15, 16, 17, 31, 32, 48, 49]);
  }

  /// Edge: empty input is a no-op.
  #[test]
  fn rgb_widen_empty_is_noop() {
    assert!(rgb_widen_dispatch_init(&[]).is_empty());
    assert!(rgb_widen_scalar_init(&[]).is_empty());
  }

  /// Edge: 3-byte input (1 RGB pixel). Pins basic byte-for-byte
  /// widen: `[10, 20, 30]` → `[10.0, 20.0, 30.0]`.
  #[test]
  fn rgb_widen_one_pixel_no_swap() {
    let buf = rgb_widen_dispatch_init(&[10, 20, 30]);
    assert_eq!(buf, vec![10.0_f32, 20.0, 30.0]);
  }

  /// Edge: 16 bytes (1 full NEON tile, body=16, tail=0). Pins the
  /// body-loop's clean-exit behaviour.
  #[test]
  fn rgb_widen_sixteen_bytes_one_full_tile() {
    let src = gen_rgb_bytes(16);
    let buf = rgb_widen_dispatch_init(&src);
    let scalar = rgb_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 16);
  }

  /// Edge: 17 bytes (one full tile + 1 tail byte). Pins the
  /// body-then-tail handoff.
  #[test]
  fn rgb_widen_seventeen_bytes_tile_plus_one() {
    let src = gen_rgb_bytes(17);
    let buf = rgb_widen_dispatch_init(&src);
    let scalar = rgb_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 17);
  }

  /// Behavioural test — the dispatcher must produce byte-identical
  /// output to the OLD `buf.extend(raw.iter().map(|&b| f32::from(b)))`
  /// loop on a 512×512 RGB canvas (786432 bytes).
  #[test]
  fn image_to_array_rgb_matches_old_extend() {
    let n = 512usize * 512 * 3; // 786_432

    // Several patterns to stress the widen path. Boxed closures so
    // the array element type stays trivial; aliased to keep clippy's
    // type-complexity lint happy.
    type PatternFn<'a> = Box<dyn Fn() -> Vec<u8> + 'a>;
    let patterns: [(&str, PatternFn<'_>); 4] = [
      ("all_zero", Box::new(move || vec![0u8; n])),
      ("all_255", Box::new(move || vec![255u8; n])),
      (
        "asymmetric",
        Box::new(move || (0..n).map(|i| ((i * 13) % 256) as u8).collect()),
      ),
      (
        "gradient",
        Box::new(move || {
          let mut v = Vec::with_capacity(n);
          for i in 0..n {
            v.push((i % 256) as u8);
          }
          v
        }),
      ),
    ];

    for (name, make_pattern) in &patterns {
      let raw = make_pattern();
      assert_eq!(raw.len(), n, "pattern {name} length mismatch");

      // OLD path — pre-C3 idiom.
      let mut old: Vec<f32> = Vec::with_capacity(n);
      old.extend(raw.iter().map(|&b| f32::from(b)));
      assert_eq!(old.len(), n, "OLD extend length mismatch ({name})");

      // NEW path — the dispatcher, called on uninit spare capacity.
      let new = rgb_widen_dispatch_init(&raw);

      assert_eq!(
        new, old,
        "C3 dispatcher must produce byte-identical output to OLD extend (pattern={name})"
      );
    }
  }

  /// Release-mode precondition guards — scalar.
  #[test]
  #[should_panic(expected = "rgb_widen_scalar: out.len() (5) must equal src.len() (6)")]
  fn rgb_widen_scalar_panics_on_size_mismatch_in_release() {
    let src = [0u8; 6];
    let mut v: Vec<f32> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    rgb_widen_scalar(&mut spare[..5], &src);
  }

  /// Release-mode precondition guards — dispatcher.
  #[test]
  #[should_panic(expected = "simd::vlm::rgb_widen: out.len() (5) must equal src.len() (6)")]
  fn rgb_widen_dispatch_panics_on_size_mismatch_in_release() {
    let src = [0u8; 6];
    let mut v: Vec<f32> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    rgb_widen(&mut spare[..5], &src);
  }

  /// Release-mode precondition guards — NEON.
  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "rgb_widen_neon: out.len() (5) must equal src.len() (6)")]
  fn rgb_widen_neon_panics_on_size_mismatch_in_release() {
    if !crate::simd::is_neon_available() {
      panic!("rgb_widen_neon: out.len() (5) must equal src.len() (6) (skipped — NEON unavailable)");
    }
    let src = [0u8; 6];
    let mut v: Vec<f32> = Vec::with_capacity(5);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    // SAFETY: NEON checked; expected-panic on size-mismatch
    // precondition violation before any pointer arithmetic.
    unsafe { super::rgb_widen_neon(&mut spare[..5], &src) };
  }
}
