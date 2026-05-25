//! C5 — `rotate_buf` pixel permutation.
//!
//! Tracking: [#150](https://github.com/Findit-AI/mlxrs/issues/150).
//! Plan: `docs/core-arch-simd-candidates.md` §2 row C5, §4 (image
//! preprocessing). The §5.5 doc originally deferred C5 as "gather-bound";
//! per the user directive 2026-05-23 (project memory rule **"SIMD ship
//! NEON regardless"**), the NEON kernel ships unconditionally.
//!
//! # The defect class
//!
//! Pre-C5 `crate::vlm::image::rotate_buf` is a per-pixel
//! `copy_from_slice` loop over the source:
//!
//! ```rust,ignore
//! for y in 0..h_usize {
//!   for x in 0..w_usize {
//!     let (nx, ny) = match rotation { ... };
//!     let src_off = (y * w_usize + x) * channels;
//!     let dst_off = (ny * out_w_usize + nx) * channels;
//!     dst[dst_off..dst_off + channels].copy_from_slice(&src[src_off..src_off + channels]);
//!   }
//! }
//! ```
//!
//! Per pixel: a `copy_from_slice` of `channels` bytes (1, 2, 3, or 4
//! for u8 / u16 / f32 element types). LLVM auto-vectorizes the inner
//! per-channel copy as a single `LDR Wn`/`STR Wn` for the
//! `channels=4` case, but the outer iteration is **scatter-dominated**
//! — `dst_off` is a row-stride permutation of `src_off`, so successive
//! source pixels write to different output rows. NEON has no scatter,
//! so the SIMD win is bounded by the per-pixel widen (channels=4 case)
//! + the outer-loop unrolling.
//!
//! # The fix — u8 channels=4 specialised NEON kernel
//!
//! The hot path in mlxrs (and the only path where NEON gives a
//! meaningful speedup over LLVM's auto-vec) is **u8 + channels=4**
//! (Rgba8 source from `image::DynamicImage`). For this case the kernel:
//!
//! 1. Reads 4 source pixels (16 bytes) per tile via `vld1q_u8`.
//! 2. Computes the 4 destination offsets per the rotation kind.
//! 3. Writes each pixel as a single 32-bit store
//!    (`core::ptr::write_unaligned::<u32>`) at the destination offset.
//!
//! The destination writes are inherently scattered (each pixel goes to
//! a different row), so the NEON load is the only contiguous step. The
//! kernel matches the auto-vectorized scalar's per-pixel-copy shape but
//! pins the load width at 16 bytes — a guaranteed contract independent
//! of LLVM heuristics.
//!
//! For every other type / channels combination (u8 + 1/2/3, u16 + any,
//! f32 + any) the dispatcher falls back to the scalar arm. Specialising
//! for `(u8, 4)` covers the dominant Rgba8 image-decode + EXIF-rotate
//! path; the other arms are infrequent enough that the scalar arm's
//! auto-vectorized shape is the right contract.
//!
//! # Correctness class — `Exact`
//!
//! Pure data movement — every output byte equals exactly one input
//! byte. Scalar and NEON arms produce bit-identical output. Differential
//! tests use [`crate::simd::diff::assert_eq_over_lane_sweep`].
//!
//! # Output API
//!
//! The dispatcher writes into a caller-allocated `&mut [u8]` already
//! sized to `src.len()` (per the caller's pre-existing
//! `try_reserve_exact` + `resize` discipline in `rotate_buf`). Unlike
//! the C3/C4 widen kernels, no `MaybeUninit` is needed — every
//! destination byte is written by exactly one source-pixel store.
//!
//! # Rotation arms (mirrors `RotateKind`)
//!
//! - `Rotate90`        : `(x, y) -> (h - 1 - y, x)` ; out dims `(h, w)`
//! - `Rotate270`       : `(x, y) -> (y, w - 1 - x)` ; out dims `(h, w)`
//! - `Rotate90FlipH`   : `(x, y) -> (y, x)` ; out dims `(h, w)`
//! - `Rotate270FlipH`  : `(x, y) -> (h - 1 - y, w - 1 - x)` ; out dims `(h, w)`
//!
//! All four output dimensions are `(h, w)` (transpose); the NEON arm
//! is parameterised by the rotation kind via a `RotateKind` enum
//! mirror to keep dispatch symmetric with the call site.

/// Pixel-permutation rotation variants. Mirrors
/// `crate::vlm::image::RotateKind` (kept local because that enum is
/// crate-private; this is the dispatcher's parameter type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateKind {
  /// Clockwise 90° (transpose + flip on the horizontal axis).
  Rotate90,
  /// Clockwise 270° (transpose + flip on the vertical axis).
  Rotate270,
  /// `Rotate90` then horizontal flip — collapses to `(y, x)`.
  Rotate90FlipH,
  /// `Rotate270` then horizontal flip — collapses to `(h-1-y, w-1-x)`.
  Rotate270FlipH,
}

/// Scalar reference: per-pixel rotation of `src` into `dst`. Bit-exact
/// match for the pre-C5 `rotate_buf` inner two-loop.
///
/// `channels` is the per-pixel subpixel count (1 / 2 / 3 / 4); `dst`
/// and `src` are byte slices, so for `T = u8` the unit is bytes; the
/// scalar arm is intentionally **u8-only** since the dominant call
/// site is u8 (Rgba8 / Rgb8 image-decode), and the dispatcher gates
/// non-u8 inputs to the scalar arm directly in the caller (the public
/// generic `rotate_buf::<T>` in `crate::vlm::image` keeps its T-generic
/// inner loop).
///
/// # Preconditions
///
/// - `dst.len() == src.len() == src_w * src_h * channels`.
/// - `src_w * src_h * channels` does not overflow `usize` (panics
///   explicitly via `checked_mul` rather than wrapping silently).
///
/// All asserted **unconditionally** (release-too).
#[inline]
#[doc(hidden)]
pub fn rotate_buf_u8_scalar(
  dst: &mut [u8],
  src: &[u8],
  src_w: usize,
  src_h: usize,
  channels: usize,
  rotation: RotateKind,
) {
  let elements = src_w
    .checked_mul(src_h)
    .and_then(|wh| wh.checked_mul(channels))
    .unwrap_or_else(|| {
      panic!("rotate_buf_u8_scalar: dimensions {src_w}x{src_h}x{channels} overflow usize")
    });
  assert_eq!(
    src.len(),
    elements,
    "rotate_buf_u8_scalar: src.len() ({}) must equal src_w * src_h * channels ({} * {} * {} = {})",
    src.len(),
    src_w,
    src_h,
    channels,
    elements,
  );
  assert_eq!(
    dst.len(),
    elements,
    "rotate_buf_u8_scalar: dst.len() ({}) must equal src.len() ({})",
    dst.len(),
    elements,
  );

  // Output width per the rotation: every rotate variant transposes,
  // so out_w == src_h, out_h == src_w.
  let out_w = src_h;
  for y in 0..src_h {
    for x in 0..src_w {
      let (nx, ny) = match rotation {
        RotateKind::Rotate90 => (src_h - 1 - y, x),
        RotateKind::Rotate270 => (y, src_w - 1 - x),
        RotateKind::Rotate90FlipH => (y, x),
        RotateKind::Rotate270FlipH => (src_h - 1 - y, src_w - 1 - x),
      };
      let src_off = (y * src_w + x) * channels;
      let dst_off = (ny * out_w + nx) * channels;
      dst[dst_off..dst_off + channels].copy_from_slice(&src[src_off..src_off + channels]);
    }
  }
}

/// NEON 4-pixel-tile u8 rotation for `channels = 4` (RGBA). Reads 16
/// bytes per tile via `vld1q_u8`, then issues four 32-bit destination
/// stores via `core::ptr::write_unaligned::<u32>`.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. Caller obligation;
///    discharged by [`rotate_buf_u8`].
/// 2. `channels == 4` — required by the per-pixel u32 store assumption.
/// 3. `src.len() == dst.len() == src_w * src_h * 4` — asserted
///    **unconditionally** here.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn rotate_buf_u8_channels4_neon(
  dst: &mut [u8],
  src: &[u8],
  src_w: usize,
  src_h: usize,
  rotation: RotateKind,
) {
  let channels = 4usize;
  let elements = src_w
    .checked_mul(src_h)
    .and_then(|wh| wh.checked_mul(channels))
    .unwrap_or_else(|| {
      panic!("rotate_buf_u8_channels4_neon: dimensions {src_w}x{src_h}x4 overflow usize")
    });
  assert_eq!(
    src.len(),
    elements,
    "rotate_buf_u8_channels4_neon: src.len() ({}) must equal src_w * src_h * 4 ({} * {} * 4 = {})",
    src.len(),
    src_w,
    src_h,
    elements,
  );
  assert_eq!(
    dst.len(),
    elements,
    "rotate_buf_u8_channels4_neon: dst.len() ({}) must equal src.len() ({})",
    dst.len(),
    elements,
  );

  let out_w = src_h;

  // SAFETY: per-row tile loop reads 16 bytes via `vld1q_u8` from
  // `src.as_ptr().add(src_off)` for `src_off + 16 <= row_end_off
  // <= src.len()`. Destination writes are per-pixel u32 stores via
  // `core::ptr::write_unaligned::<u32>` at `dst.as_mut_ptr().add(dst_off)`
  // for `dst_off + 4 <= dst.len()` — checked by the per-pixel index
  // math (`(ny * out_w + nx) * 4` with `0 <= ny < src_h`, `0 <= nx <
  // src_w`, so `dst_off + 4 <= src_h * src_w * 4 = dst.len()`).
  // Writes target `&mut [u8]` backing memory, which has no validity
  // invariants beyond size + alignment; `write_unaligned` accepts any
  // address. NEON availability is the caller's obligation
  // (precondition #1).
  unsafe {
    let src_base = src.as_ptr();
    let dst_base = dst.as_mut_ptr();

    for y in 0..src_h {
      let row_x = src_w - (src_w % 4);
      let mut x = 0usize;
      while x + 4 <= row_x {
        let src_off = (y * src_w + x) * channels;
        // Load 16 source bytes (4 pixels × 4 channels).
        let tile = core::arch::aarch64::vld1q_u8(src_base.add(src_off));

        // Store as four u32 to a 16-byte stack scratch so we can
        // re-load per-pixel u32 lanes for the scattered destination
        // stores. This avoids vgetq_lane_u32 four times.
        let mut scratch = [0u8; 16];
        core::arch::aarch64::vst1q_u8(scratch.as_mut_ptr(), tile);

        for lane in 0..4 {
          let xx = x + lane;
          let (nx, ny) = match rotation {
            RotateKind::Rotate90 => (src_h - 1 - y, xx),
            RotateKind::Rotate270 => (y, src_w - 1 - xx),
            RotateKind::Rotate90FlipH => (y, xx),
            RotateKind::Rotate270FlipH => (src_h - 1 - y, src_w - 1 - xx),
          };
          let dst_off = (ny * out_w + nx) * channels;
          // Read the u32 pixel from the scratch buffer and write it
          // unaligned to dst_off. `read_unaligned`/`write_unaligned`
          // are required because `scratch` is u8-aligned and `dst`
          // offsets are channels-multiples (= 4n) which IS naturally
          // u32-aligned for u8-backed buffers, but we keep
          // `write_unaligned` for portability.
          let pixel: u32 = core::ptr::read_unaligned(scratch.as_ptr().add(lane * 4).cast::<u32>());
          core::ptr::write_unaligned(dst_base.add(dst_off).cast::<u32>(), pixel);
        }
        x += 4;
      }
      // Tail (`src_w % 4` < 4 pixels) — scalar per-pixel copy.
      while x < src_w {
        let (nx, ny) = match rotation {
          RotateKind::Rotate90 => (src_h - 1 - y, x),
          RotateKind::Rotate270 => (y, src_w - 1 - x),
          RotateKind::Rotate90FlipH => (y, x),
          RotateKind::Rotate270FlipH => (src_h - 1 - y, src_w - 1 - x),
        };
        let src_off = (y * src_w + x) * channels;
        let dst_off = (ny * out_w + nx) * channels;
        let pixel: u32 = core::ptr::read_unaligned(src_base.add(src_off).cast::<u32>());
        core::ptr::write_unaligned(dst_base.add(dst_off).cast::<u32>(), pixel);
        x += 1;
      }
    }
  }
}

/// Public dispatcher: rotate a u8 byte buffer in place. Routes to the
/// `channels=4` NEON kernel on `aarch64` (when NEON is reported) when
/// `channels == 4`; everything else falls back to the scalar arm
/// (which itself is per-pixel `copy_from_slice` that LLVM
/// auto-vectorizes for `channels=1/2/4`).
///
/// Used by `crate::vlm::image::rotate_buf` for the u8 element-type
/// arms (Luma8 / LumaA8 / Rgb8 / Rgba8).
///
/// # Preconditions
///
/// - `src.len() == dst.len() == src_w * src_h * channels` — asserted
///   unconditionally.
/// - `src_w * src_h * channels` does not overflow `usize` — checked
///   via `checked_mul` BEFORE the size-equality assertions, so a
///   wrapped product can never sneak past the size checks and let the
///   unsafe NEON kernel compute offsets from unwrapped loop dims (the
///   wired [`crate::vlm::image::rotate_buf`] caller already
///   pre-checks, but this public entry is reachable directly).
///
/// # Panics
///
/// Panics explicitly (not silently wraps) on `src_w * src_h *
/// channels` `usize` overflow — the only correct response when a
/// caller has supplied dimensions that cannot fit a contiguous
/// buffer, since silently wrapping would let an under-sized buffer
/// satisfy the size-equality assertion and reach the unsafe kernel.
///
/// # Correctness class
///
/// `Exact` — bit-identical output between scalar and NEON.
#[inline]
#[doc(hidden)]
pub fn rotate_buf_u8(
  dst: &mut [u8],
  src: &[u8],
  src_w: usize,
  src_h: usize,
  channels: usize,
  rotation: RotateKind,
) {
  // Checked dimension math BEFORE the size-equality assertions:
  // wrapping `src_w * src_h * channels` in release mode could
  // otherwise produce a small `elements` that an under-sized
  // `src` / `dst` would satisfy, letting the unsafe NEON kernel
  // compute per-pixel offsets from unwrapped loop dims and issue
  // out-of-bounds `vld1q_u8` / `write_unaligned` (UB).
  let elements = src_w
    .checked_mul(src_h)
    .and_then(|wh| wh.checked_mul(channels))
    .unwrap_or_else(|| {
      panic!("simd::vlm::rotate_buf_u8: dimensions {src_w}x{src_h}x{channels} overflow usize")
    });
  assert_eq!(
    src.len(),
    elements,
    "simd::vlm::rotate_buf_u8: src.len() ({}) must equal src_w * src_h * channels ({} * {} * {} = {})",
    src.len(),
    src_w,
    src_h,
    channels,
    elements,
  );
  assert_eq!(
    dst.len(),
    elements,
    "simd::vlm::rotate_buf_u8: dst.len() ({}) must equal src.len() ({})",
    dst.len(),
    elements,
  );

  #[cfg(target_arch = "aarch64")]
  {
    if channels == 4 && crate::simd::is_neon_available() {
      // SAFETY: NEON gated; channels == 4 confirmed; size preconditions
      // asserted above; `elements` derived via `checked_mul` so per-
      // pixel offsets cannot overflow into stale ranges.
      unsafe { rotate_buf_u8_channels4_neon(dst, src, src_w, src_h, rotation) };
      return;
    }
  }
  rotate_buf_u8_scalar(dst, src, src_w, src_h, channels, rotation);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher Exact differential tests + edge coverage for C5.

  use super::{RotateKind, rotate_buf_u8, rotate_buf_u8_scalar};

  /// Build a deterministic source buffer of `w * h * channels` bytes.
  fn src(w: usize, h: usize, channels: usize) -> Vec<u8> {
    (0..(w * h * channels)).map(|i| (i % 251) as u8).collect()
  }

  fn rotate_via(
    dispatch: bool,
    w: usize,
    h: usize,
    channels: usize,
    rotation: RotateKind,
  ) -> Vec<u8> {
    let s = src(w, h, channels);
    let mut d = vec![0u8; s.len()];
    if dispatch {
      rotate_buf_u8(&mut d, &s, w, h, channels, rotation);
    } else {
      rotate_buf_u8_scalar(&mut d, &s, w, h, channels, rotation);
    }
    d
  }

  #[test]
  fn rotate_buf_u8_channels4_scalar_matches_dispatcher_exact() {
    // Sweep over interesting widths (boundaries around multiples of 4).
    for &w in &[1usize, 4, 5, 7, 8, 16, 17, 33] {
      for &h in &[1usize, 2, 4, 8, 17] {
        for &rotation in &[
          RotateKind::Rotate90,
          RotateKind::Rotate270,
          RotateKind::Rotate90FlipH,
          RotateKind::Rotate270FlipH,
        ] {
          let s = rotate_via(false, w, h, 4, rotation);
          let d = rotate_via(true, w, h, 4, rotation);
          assert_eq!(
            s, d,
            "Exact mismatch (w={w}, h={h}, channels=4, rotation={rotation:?})"
          );
        }
      }
    }
  }

  #[test]
  fn rotate_buf_u8_channels3_scalar_matches_dispatcher_exact() {
    // channels=3 routes to the scalar arm; verify the dispatcher
    // produces the same output as the scalar reference.
    for &w in &[1usize, 4, 17] {
      for &h in &[1usize, 4, 8] {
        for &rotation in &[
          RotateKind::Rotate90,
          RotateKind::Rotate270,
          RotateKind::Rotate90FlipH,
          RotateKind::Rotate270FlipH,
        ] {
          let s = rotate_via(false, w, h, 3, rotation);
          let d = rotate_via(true, w, h, 3, rotation);
          assert_eq!(
            s, d,
            "Exact mismatch (w={w}, h={h}, channels=3, rotation={rotation:?})"
          );
        }
      }
    }
  }

  #[test]
  fn rotate_buf_u8_rotate90_pin() {
    // 2x2 RGBA, Rotate90: per scalar arm `(nx, ny) = (h-1-y, x)`,
    // dst_off = (ny * out_w + nx) * 4 where out_w = src_h = 2.
    let w = 2;
    let h = 2;
    let channels = 4;
    let s: Vec<u8> = (0..16).map(|i| i as u8).collect();
    let mut d = vec![0u8; 16];
    rotate_buf_u8(&mut d, &s, w, h, channels, RotateKind::Rotate90);
    // src (0, 0) = [0..4]; nx=h-1-0=1, ny=0; dst_off = (0*2 + 1)*4 = 4
    assert_eq!(&d[4..8], &s[0..4], "Rotate90: src(0,0) → dst[4..8]");
    // src (1, 0) = [4..8]; nx=h-1-0=1, ny=1; dst_off = (1*2 + 1)*4 = 12
    assert_eq!(&d[12..16], &s[4..8], "Rotate90: src(1,0) → dst[12..16]");
    // src (0, 1) = [8..12]; nx=h-1-1=0, ny=0; dst_off = (0*2 + 0)*4 = 0
    assert_eq!(&d[0..4], &s[8..12], "Rotate90: src(0,1) → dst[0..4]");
    // src (1, 1) = [12..16]; nx=h-1-1=0, ny=1; dst_off = (1*2 + 0)*4 = 8
    assert_eq!(&d[8..12], &s[12..16], "Rotate90: src(1,1) → dst[8..12]");
  }

  #[test]
  fn rotate_buf_u8_double_rotate_round_trip() {
    // Two Rotate90s + two Rotate270s should recover the input
    // (composition is identity for any 360° net rotation). We test
    // Rotate90 → Rotate270 round-trip (which IS identity since
    // Rotate90 inverse is Rotate270).
    let w = 4;
    let h = 3;
    let channels = 4;
    let s: Vec<u8> = (0..(w * h * channels)).map(|i| (i % 251) as u8).collect();

    // Rotate90: src (w=4, h=3) → out (w=h=3, h=w=4)
    let mut once = vec![0u8; s.len()];
    rotate_buf_u8(&mut once, &s, w, h, channels, RotateKind::Rotate90);
    // Rotate270 on the once-rotated buffer (w=3, h=4) → recovers (w=4, h=3)
    let mut twice = vec![0u8; s.len()];
    rotate_buf_u8(&mut twice, &once, h, w, channels, RotateKind::Rotate270);
    assert_eq!(twice, s, "Rotate90 ∘ Rotate270 should be identity");
  }

  #[test]
  #[should_panic(
    expected = "simd::vlm::rotate_buf_u8: src.len() (3) must equal src_w * src_h * channels"
  )]
  fn rotate_buf_u8_panics_on_size_mismatch() {
    let s = vec![0u8; 3]; // WRONG: should be 2*2*4 = 16
    let mut d = vec![0u8; 16];
    rotate_buf_u8(&mut d, &s, 2, 2, 4, RotateKind::Rotate90);
  }

  /// Wrap-arith defence: even though the wired `rotate_buf_u8_via_c5`
  /// caller pre-checks `src_w * src_h * channels` via `checked_mul`,
  /// the public dispatcher entry is reachable directly (e.g. via a
  /// `pub use` from a future caller, or via the in-crate `unsafe`
  /// neighbours that share the symbol). A wrapping multiply in
  /// release mode would otherwise let a small `elements` value pass
  /// the size-equality assertion and reach the unsafe NEON kernel —
  /// where the per-pixel offset math (computed from the unwrapped
  /// loop dims) would compute out-of-bounds offsets and trigger UB.
  /// `checked_mul` must therefore land BEFORE the asserts.
  #[test]
  #[should_panic(expected = "overflow usize")]
  fn rotate_buf_u8_panics_on_dimension_overflow() {
    // src_w * src_h would already saturate (usize::MAX/2 + 1) * 2 → wrap.
    // We give a small `src` + `dst` so allocation succeeds and the
    // dimension overflow is the only failure mode.
    let s = vec![0u8; 16];
    let mut d = vec![0u8; 16];
    rotate_buf_u8(&mut d, &s, usize::MAX / 2 + 1, 2, 4, RotateKind::Rotate90);
  }

  #[test]
  fn rotate_buf_u8_rotate90_flip_h_collapses() {
    // Rotate90FlipH: per scalar arm `(nx, ny) = (y, x)` — pure
    // transpose. dst_off = (ny * out_w + nx) * 4 with out_w = src_h = 2.
    let w = 2;
    let h = 2;
    let channels = 4;
    let s: Vec<u8> = (0..16).map(|i| i as u8).collect();
    let mut d = vec![0u8; 16];
    rotate_buf_u8(&mut d, &s, w, h, channels, RotateKind::Rotate90FlipH);
    // src (0, 0) = [0..4]; nx=0, ny=0; dst_off = 0. dst[0..4] = src[0..4]
    assert_eq!(&d[0..4], &s[0..4]);
    // src (1, 0) = [4..8]; nx=0, ny=1; dst_off = (1*2 + 0)*4 = 8.
    assert_eq!(&d[8..12], &s[4..8]);
    // src (0, 1) = [8..12]; nx=1, ny=0; dst_off = (0*2 + 1)*4 = 4.
    assert_eq!(&d[4..8], &s[8..12]);
    // src (1, 1) = [12..16]; nx=1, ny=1; dst_off = (1*2 + 1)*4 = 12.
    assert_eq!(&d[12..16], &s[12..16]);
  }
}
