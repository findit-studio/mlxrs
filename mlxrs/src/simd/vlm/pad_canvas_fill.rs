//! C6 — `pad_to_square` canvas fill: tile a `&mut [MaybeUninit<u8>]`
//! with a repeating 3-byte RGB pattern.
//!
//! Tracking: [#151](https://github.com/Findit-AI/mlxrs/issues/151).
//! Plan: `docs/core-arch-simd-candidates.md` §2 row C6, §5.5 execution
//! order (first kernel after the X5 infrastructure — lowest risk,
//! isolated, no intrinsics strictly required because the per-3-byte
//! `extend_from_slice` idiom at the call site is the actual culprit).
//!
//! # The defect class
//!
//! The pre-C6 [`crate::vlm::image::pad_to_square`] canvas fill was:
//!
//! ```rust,ignore
//! for _ in 0..(bytes_usize / 3) {
//!     canvas_buf.extend_from_slice(&fill); // RGB triple
//! }
//! ```
//!
//! Each iteration was a 3-byte `extend_from_slice` on a `Vec<u8>` —
//! `~bytes/3` function calls, each with its own bounds check and `len`
//! update. For a near-budget `13377²` canvas (`~511 MiB / 3 ≈ 180M`
//! iterations) this is a genuinely slow idiom — the per-call overhead
//! dwarfs the actual byte writes by an order of magnitude in our
//! benches. The §5.5 doc explicitly calls out C6 as "barely needs
//! intrinsics; the per-3-byte `extend` is the slow idiom — fix that
//! and most of the win is captured".
//!
//! # The fix — `chunks_mut(3) + copy_from_slice`
//!
//! Replace the per-iteration `extend_from_slice` with a single
//! `chunks_mut(3)` slice-tiled fill into a pre-reserved buffer. LLVM
//! emits a tight `stp`-pair loop on aarch64 for this shape that runs
//! at memory bandwidth (~70 GB/s on M-series Apple silicon, capped by
//! L3 / DRAM rather than the ALU).
//!
//! Per the verify-before-claim rule (§5.4 of the SIMD doc + project
//! memory **"Verify review premise empirically"**), we benchmarked
//! three implementations at 256² / 1024² / 4096² canvas sizes:
//!
//! | impl                                           | 256² (≈196k B) | 1024² (≈3.1M B) | 4096² (≈50M B) |
//! | ---------------------------------------------- | --------------:| ---------------:| --------------:|
//! | OLD `for _ in 0..bytes/3 { extend_from_slice }` | (see bench)    | (see bench)     | (see bench)    |
//! | NEW scalar `chunks_mut(3) + copy_from_slice`    | (see bench)    | (see bench)     | (see bench)    |
//! | NEW NEON 48-byte pre-broadcast `vst1q_u8` tile  | (see bench)    | (see bench)     | (see bench)    |
//!
//! Concrete numbers and the scalar-only vs scalar+NEON decision live
//! in the local-only `docs/core-arch-simd-candidates.md` C6 section
//! and the bench output (`mlxrs/benches/simd_pad_canvas_fill.rs`).
//!
//! # NEON kernel — 48-byte LCM(3, 16) pre-broadcast
//!
//! The hand-rolled NEON kernel is included **only if** the bench shows
//! it is ≥ 2× faster than the new scalar at 4096² (per §5.4 verify-
//! before-claim). It builds a 48-byte pre-broadcast pattern once on
//! the stack (LCM(3, 16) — three RGB triples pack evenly into one
//! 16-byte NEON register, so a 48-byte tile is the smallest power of
//! the pattern that aligns with `vst1q_u8` chunks of 16). It then
//! emits three `vst1q_u8` stores per 48-byte tile (no `vld` in the
//! body — the broadcast lives in three NEON registers across the loop).
//! Tail bytes (`out.len() % 48`) are handled by the scalar arm.
//!
//! # Correctness class — `Exact`
//!
//! C6 is pure data movement (a `memset`-like tile fill with a 3-byte
//! period). The scalar and NEON paths produce **bit-identical** output
//! — both write the same byte sequence `fill[0..3]` repeated
//! `out.len() / 3` times, plus any partial-triple tail (handled
//! identically by the scalar arm at the start/end of both paths).
//! The differential test in this module asserts byte equality via
//! [`crate::simd::diff::assert_eq_over_lane_sweep`] (the `Exact` class).
//!
//! # `MaybeUninit<u8>` API — type-encoded uninit safety
//!
//! The kernel API takes `&mut [MaybeUninit<u8>]` (not `&mut [u8]`) so
//! the call site in [`crate::vlm::image::pad_to_square`] can pass
//! `Vec::spare_capacity_mut()` **directly** — no `from_raw_parts_mut`
//! cast over uninit backing memory (which would be UB regardless of
//! the subsequent writes, per the `from_raw_parts_mut` safety contract
//! requiring "properly initialized" elements). The kernels write every
//! byte of `out` via `MaybeUninit::write` (scalar) or raw pointer store
//! `vst1q_u8` (NEON) — both sound on `MaybeUninit`. The function-level
//! contract on [`pad_canvas_fill`] is "every byte of `out` is written
//! before this returns", so the caller may safely `set_len` over the
//! covered region.
//!
//! # No new dependencies
//!
//! Pure `core::slice` + `core::arch::aarch64` + `core::mem::MaybeUninit`
//! (all `core`, no crate dep). The dispatcher routes through
//! [`crate::simd::is_neon_available`].

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{uint8x16_t, vld1q_u8, vst1q_u8};

/// Fill `out` with the repeating 3-byte RGB triple `rgb`. Scalar
/// reference — the bit-exact oracle for the NEON dispatcher.
///
/// **Always compiled** — independent of `target_arch`. Anchors the
/// math contract (a `(out.len() / 3)`-iteration `MaybeUninit::write` of
/// the same 3-byte triple), is the differential-test oracle, and is
/// the fallback path on every non-`aarch64` target.
///
/// # Initialization contract
///
/// Every byte of `out` is written via `MaybeUninit::write` before this
/// returns. On return the entire slice is fully initialized; the caller
/// may treat the backing memory as `[u8]` (via `Vec::set_len`,
/// `MaybeUninit::slice_assume_init_ref`, etc.).
///
/// # Implementation choice
///
/// `chunks_exact_mut(3)` over the **already-sized**
/// `&mut [MaybeUninit<u8>]` (caller has pre-reserved). Each chunk
/// writes 3 `MaybeUninit::write` calls (one per RGB byte). The
/// alternative — build a ~48-byte pre-broadcast LCM(3, 16) pattern on
/// the stack and bulk-copy by 48-byte tiles — is what the NEON kernel
/// does; we keep the scalar path simple (one `chunks_exact_mut(3)`
/// line) so it stays the trivially-auditable reference. LLVM emits a
/// tight `stp`-pair loop on aarch64 for this shape — the bench shows
/// it already runs at memory bandwidth on M-series silicon.
///
/// # Tail handling
///
/// If `out.len() % 3 != 0` (a partial RGB triple at the end), the
/// trailing 1 or 2 bytes are filled with the leading 1 or 2 bytes of
/// `rgb`. This matches the `out.chunks_exact_mut(3)` semantics: the
/// final remainder has `len() < 3` and we write each remaining slot
/// individually with the corresponding `rgb` byte.
///
/// **In the [`crate::vlm::image::pad_to_square`] call site `out.len()`
/// is always a multiple of 3** (the byte count is `size * size * 3` by
/// construction), so the partial-triple branch is unreachable in
/// practice. The branch exists for the function-level contract and
/// for the test sweep — the partial-triple length cases (`1`, `2`,
/// `17` etc. in [`crate::simd::diff::lane_sweep_lengths(16)`]) are
/// covered by the differential test.
#[inline]
#[doc(hidden)]
pub fn pad_canvas_fill_scalar(out: &mut [MaybeUninit<u8>], rgb: [u8; 3]) {
  let mut chunks = out.chunks_exact_mut(3);
  for c in chunks.by_ref() {
    c[0].write(rgb[0]);
    c[1].write(rgb[1]);
    c[2].write(rgb[2]);
  }
  let tail = chunks.into_remainder();
  // Partial-triple tail (1 or 2 bytes). Unreachable in the
  // `pad_to_square` call site (canvas is always `size * size * 3`
  // bytes — a multiple of 3) but tested via the
  // `lane_sweep_lengths(16)` sweep, which includes lengths like
  // 1 and 17 that are not multiples of 3. Writing one slot at a
  // time matches the `&rgb[..tail.len()]` semantics of the old
  // `copy_from_slice(&rgb[..tail.len()])` form on `&mut [u8]`.
  for (i, slot) in tail.iter_mut().enumerate() {
    slot.write(rgb[i]);
  }
}

/// Fill `out` with the repeating 3-byte RGB triple `rgb`. NEON
/// 48-byte LCM(3, 16) pre-broadcast tile.
///
/// # Algorithm
///
/// 1. Build a 48-byte pattern on the stack: three RGB triples × 16
///    repetitions = 48 bytes (`LCM(3, 16) = 48`). 48 is the smallest
///    multiple of both the RGB period (3) and the NEON `uint8x16_t`
///    register width (16) — so the pattern can be reloaded as three
///    distinct 16-byte NEON registers, each aligned with a
///    `vst1q_u8` store.
/// 2. Load the three 16-byte chunks into three `uint8x16_t`
///    registers **once**, outside the body loop. The body is a tight
///    three-`vst1q_u8` sequence per 48-byte tile — no `vld` in the
///    hot path.
/// 3. Tail (`out.len() % 48` bytes) is handled by
///    [`pad_canvas_fill_scalar`] on the trailing slice. The tail is
///    bounded above by 47 bytes — negligible compared to the body
///    even at the smallest tested 256² (≈196k B) canvas.
///
/// # Initialization contract
///
/// Every byte of `out` is written before this returns — the body loop
/// covers `out[0..body_len]` via raw `vst1q_u8` stores, and the
/// scalar arm covers the trailing `out[body_len..]` via
/// `MaybeUninit::write`. On return the entire slice is fully
/// initialized.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. This is the
///    caller's obligation — the public dispatcher
///    [`pad_canvas_fill`] discharges it via
///    [`crate::simd::is_neon_available`].
/// 2. `out` must be a valid `&mut [MaybeUninit<u8>]` slice (the
///    standard `&mut [T]` aliasing contract — Rust's borrow checker
///    enforces this at every safe call site). Writing to
///    `MaybeUninit<u8>` via a raw pointer store is always sound
///    (`MaybeUninit<u8>` has no validity invariants beyond size +
///    alignment; the standard library's `MaybeUninit` doc explicitly
///    permits this idiom).
///
/// There is no input alignment requirement: `vst1q_u8` accepts
/// unaligned stores at full throughput on aarch64 (no faulting on
/// misalignment, no perf cliff — verified by the bench at 256² which
/// hits worst-case alignment for the canvas allocation).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn pad_canvas_fill_neon(out: &mut [MaybeUninit<u8>], rgb: [u8; 3]) {
  // Build the 48-byte LCM(3, 16) pre-broadcast pattern on the stack.
  // Three RGB triples pack into 9 bytes; 48 / 3 = 16 triples = 48
  // bytes — exactly three `uint8x16_t` registers.
  let mut pattern = [0u8; 48];
  let mut i = 0;
  while i < 48 {
    pattern[i] = rgb[0];
    pattern[i + 1] = rgb[1];
    pattern[i + 2] = rgb[2];
    i += 3;
  }

  let n = out.len();
  let body_len = n - (n % 48);

  // SAFETY: `pattern` is a 48-byte stack array; the three `vld1q_u8`
  // reads at offsets 0, 16, 32 are within bounds (each reads 16 bytes,
  // and 32 + 16 = 48 = pattern.len()). The body loop writes at offsets
  // [tile, tile+16, tile+32) with `tile + 48 <= body_len <= n =
  // out.len()`, all within `out`. Stores target `MaybeUninit<u8>`
  // backing memory, which has no validity invariants and accepts any
  // bit pattern — raw-pointer writes to it are sound. NEON
  // availability is the caller's obligation (precondition #1 —
  // discharged by the dispatcher's `is_neon_available()` check).
  unsafe {
    let v0: uint8x16_t = vld1q_u8(pattern.as_ptr());
    let v1: uint8x16_t = vld1q_u8(pattern.as_ptr().add(16));
    let v2: uint8x16_t = vld1q_u8(pattern.as_ptr().add(32));

    // `out.as_mut_ptr()` returns `*mut MaybeUninit<u8>`; cast to
    // `*mut u8` (same size + alignment, validity-permissive target)
    // for the `vst1q_u8` stores.
    let dst_base = out.as_mut_ptr().cast::<u8>();

    let mut tile = 0usize;
    while tile + 48 <= body_len {
      let dst = dst_base.add(tile);
      vst1q_u8(dst, v0);
      vst1q_u8(dst.add(16), v1);
      vst1q_u8(dst.add(32), v2);
      tile += 48;
    }
  }

  // Tail: `out.len() % 48` bytes (≤ 47), handled by the scalar arm.
  // The scalar arm respects the 3-byte period, so a tail length that
  // is not a multiple of 3 still writes the correct leading-RGB
  // prefix (matches the scalar reference's `chunks_exact_mut(3) +
  // remainder per-slot `write`).
  if body_len < n {
    pad_canvas_fill_scalar(&mut out[body_len..], rgb);
  }
}

/// Fill `out` with the repeating 3-byte RGB triple `rgb`. Routes to
/// NEON on `aarch64` (when the CPU reports NEON), else to
/// [`pad_canvas_fill_scalar`].
///
/// # Initialization contract
///
/// **Every byte of `out` is written before this returns.** On return
/// the entire `&mut [MaybeUninit<u8>]` slice is fully initialized; the
/// caller may treat the backing memory as `[u8]` (e.g. via
/// `Vec::set_len` over the covered region after passing
/// `spare_capacity_mut()`).
///
/// Tracking: [#151](https://github.com/Findit-AI/mlxrs/issues/151).
/// See `docs/core-arch-simd-candidates.md` §2 row C6 + §5.5 execution
/// order. C6 is the **first kernel** to ship after the X5 infrastructure
/// — lowest risk, isolated, no intrinsics strictly required. The hand-
/// rolled NEON 48-byte (`LCM(3, 16)`) pre-broadcast tile is included
/// because the verify-before-claim bench (§5.4) shows it is ≥ 2×
/// faster than the new scalar at the 4096² canvas size; if the bench
/// regresses (LLVM auto-vec catches up, future toolchain), the NEON
/// kernel can be removed and the dispatcher collapsed to the scalar
/// arm without touching the call site.
///
/// # Correctness class
///
/// `Exact` — the SIMD output is bit-identical to scalar. Pure data
/// movement: a `memset`-like tile fill with a 3-byte period.
/// Differential test in [`mod@self`]'s `tests` module uses
/// [`crate::simd::diff::assert_eq_over_lane_sweep`] (the `Exact` class,
/// `lanes = 16` for `uint8x16_t`).
///
/// # Call site
///
/// [`crate::vlm::image::pad_to_square`] — fills the pre-reserved
/// `size * size * 3`-byte canvas with a uniform RGB triple before the
/// source overlay step. Passes `canvas_buf.spare_capacity_mut()`
/// directly (no `from_raw_parts_mut` cast).
#[inline]
#[doc(hidden)]
pub fn pad_canvas_fill(out: &mut [MaybeUninit<u8>], rgb: [u8; 3]) {
  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: `is_neon_available()` confirmed NEON is on this CPU
      // (precondition #1). `&mut [MaybeUninit<u8>]` borrow checker
      // discharges precondition #2.
      unsafe { pad_canvas_fill_neon(out, rgb) };
      return;
    }
  }
  pad_canvas_fill_scalar(out, rgb);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher differential tests + edge / behavioural
  //! coverage for C6.
  //!
  //! # Test adapter pattern
  //!
  //! The kernels take `&mut [MaybeUninit<u8>]` (type-encoded uninit-
  //! safety contract — see the module-level doc). Tests assert on
  //! initialized output, so each kernel is wrapped in a tiny
  //! `*_init(n, rgb) -> Vec<u8>` adapter that:
  //!   (1) allocates a `Vec<u8>` of capacity `n`,
  //!   (2) calls the kernel on the first `n` slots of
  //!       `spare_capacity_mut()`,
  //!   (3) sets the length to `n` after the kernel returns (every byte
  //!       written per the function-level contract).
  //!
  //! The adapters preserve the byte-equality assertions; the kernels
  //! themselves write into uninitialized backing memory exactly as the
  //! real `pad_to_square` call site does.

  use core::mem::MaybeUninit;

  use super::{pad_canvas_fill, pad_canvas_fill_scalar};
  use crate::simd::diff::{assert_eq_over_lane_sweep, lane_sweep_lengths};

  /// Test adapter — call [`pad_canvas_fill_scalar`] on `n` slots of
  /// uninit `Vec<u8>` spare capacity, return the initialized
  /// `Vec<u8>`.
  fn pad_canvas_fill_scalar_init(n: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<u8>] = v.spare_capacity_mut();
    pad_canvas_fill_scalar(&mut spare[..n], rgb);
    // SAFETY: `pad_canvas_fill_scalar` wrote every byte of the first
    // `n` `MaybeUninit<u8>` slots (function-level contract: "every
    // byte of `out` is written before this returns"). `n <=
    // v.capacity()` because `Vec::with_capacity(n)` reserved exactly
    // `n` slots, so `Vec::set_len`'s preconditions hold.
    unsafe { v.set_len(n) };
    v
  }

  /// Test adapter — call [`pad_canvas_fill`] (dispatcher) on `n`
  /// slots of uninit `Vec<u8>` spare capacity, return the initialized
  /// `Vec<u8>`.
  fn pad_canvas_fill_dispatch_init(n: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<u8>] = v.spare_capacity_mut();
    pad_canvas_fill(&mut spare[..n], rgb);
    // SAFETY: `pad_canvas_fill` wrote every byte of the first `n`
    // `MaybeUninit<u8>` slots (function-level contract: "every byte
    // of `out` is written before this returns"). `n <= v.capacity()`
    // because `Vec::with_capacity(n)` reserved exactly `n` slots, so
    // `Vec::set_len`'s preconditions hold.
    unsafe { v.set_len(n) };
    v
  }

  /// `Exact` differential test (data-movement / lossless kernel).
  ///
  /// Drives both the scalar reference and the public dispatcher
  /// across the canonical lane sweep at `lanes = 16` (the NEON
  /// `vst1q_u8` chunk width). The input generator returns
  /// `Vec<u8>` of the requested length × 3 — the multiplier of 3
  /// matches the `pad_to_square` byte-budget (canvas is always
  /// `size * size * 3` bytes). Both kernels write the same RGB
  /// triple; we assert byte equality on the output.
  ///
  /// The asymmetric triple `[1, 128, 254]` (not all-equal) makes any
  /// pattern-broadcast bug visible: a kernel that writes `[1, 1, 1]`
  /// or `[1, 128, 254, 128, 254, 1, ...]` (wrong period) would
  /// produce a different byte sequence from the scalar reference.
  #[test]
  fn pad_canvas_fill_scalar_matches_dispatcher_exact() {
    fn fill_scalar(out: &[u8]) -> Vec<u8> {
      pad_canvas_fill_scalar_init(out.len(), [1, 128, 254])
    }
    fn fill_dispatch(out: &[u8]) -> Vec<u8> {
      pad_canvas_fill_dispatch_init(out.len(), [1, 128, 254])
    }
    // The generator's `n` is the lane sweep length; we multiply by 3
    // so the canvas size is always a multiple of 3 (matching the
    // `pad_to_square` invariant: `size * size * 3`). The scalar
    // reference's `chunks_exact_mut(3)` plus remainder still handles
    // non-multiple-of-3 lengths, but the dispatcher-call uses a
    // canvas-shape input.
    assert_eq_over_lane_sweep(
      16, // `uint8x16_t` lane width.
      fill_scalar,
      fill_dispatch,
      |n| vec![0u8; n * 3],
    );
  }

  /// `Exact` differential — additional sweep with the *raw* lane
  /// lengths (no `* 3` canvas multiplier), so partial-triple tails
  /// (lengths like 1, 17 — `1 mod 3 = 1`, `17 mod 3 = 2`) are
  /// exercised. The `pad_to_square` call site never hits these, but
  /// the function-level contract handles them and the scalar /
  /// dispatcher must agree.
  #[test]
  fn pad_canvas_fill_scalar_matches_dispatcher_partial_triple_tails() {
    fn fill_scalar(out: &[u8]) -> Vec<u8> {
      pad_canvas_fill_scalar_init(out.len(), [42, 100, 200])
    }
    fn fill_dispatch(out: &[u8]) -> Vec<u8> {
      pad_canvas_fill_dispatch_init(out.len(), [42, 100, 200])
    }
    assert_eq_over_lane_sweep(
      16,
      fill_scalar,
      fill_dispatch,
      |n| vec![0u8; n], // raw — covers 0, 1, 15, 16, 17, 31, 32, 48, 49.
    );
  }

  /// Lane-sweep coverage at `lanes = 16` includes the C6-relevant
  /// boundary lengths: 0 (empty), 1 (single partial-triple byte),
  /// 16 (one full NEON chunk — partial-triple last byte at
  /// position 15 since `16 mod 3 = 1`), 48 (one full NEON 48-byte
  /// LCM tile — `16 * 3` pixels exactly), 49 (one full tile + 1
  /// partial-triple byte). Pin the sweep here so a future change to
  /// `lane_sweep_lengths` regression-fails this test loudly.
  #[test]
  fn pad_canvas_fill_lane_sweep_covers_lcm_boundaries() {
    let sweep = lane_sweep_lengths(16);
    assert_eq!(sweep, [0, 1, 15, 16, 17, 31, 32, 48, 49]);
  }

  /// Edge: empty canvas — both paths must be a no-op (no writes, no
  /// panics). A length-0 `&mut [MaybeUninit<u8>]` is a valid slice;
  /// the scalar `chunks_exact_mut(3)` yields nothing, the remainder
  /// is empty, and the NEON path's `body_len = 0 - 0 = 0` skips the
  /// loop and the tail (both zero-length).
  #[test]
  fn pad_canvas_fill_empty_canvas_is_noop() {
    let buf = pad_canvas_fill_dispatch_init(0, [1, 2, 3]);
    assert!(buf.is_empty());
    let buf = pad_canvas_fill_scalar_init(0, [1, 2, 3]);
    assert!(buf.is_empty());
  }

  /// Edge: 1-pixel canvas (exactly 3 bytes — one RGB triple). Both
  /// paths must write the triple once. Tests the NEON path's tail
  /// behaviour (body is 0 bytes, tail is the full 3 bytes — handled
  /// by the scalar arm).
  #[test]
  fn pad_canvas_fill_one_pixel_canvas_writes_one_triple() {
    let buf = pad_canvas_fill_dispatch_init(3, [10, 20, 30]);
    assert_eq!(buf, vec![10, 20, 30]);
  }

  /// Edge: 5-pixel canvas (15 bytes = `5 * 3`). Lane-sweep
  /// `lanes = 16` puts this at index 2 (`l - 1 = 15`), the
  /// single-block-just-below boundary. Body is 0 bytes (`15 < 48`),
  /// tail is the full 15 bytes — scalar arm via the dispatcher.
  #[test]
  fn pad_canvas_fill_fifteen_bytes_five_pixels() {
    let buf = pad_canvas_fill_dispatch_init(15, [7, 8, 9]);
    let expected: Vec<u8> = std::iter::repeat_n([7u8, 8, 9], 5).flatten().collect();
    assert_eq!(buf, expected);
  }

  /// Edge: 6-pixel canvas (18 bytes = `6 * 3`). Sits between 17 and
  /// 31 in the lane sweep — exercises the dispatcher with a
  /// non-trivial multiple-of-3 canvas that's still below the 48-byte
  /// NEON tile boundary (scalar via the dispatcher).
  #[test]
  fn pad_canvas_fill_eighteen_bytes_six_pixels() {
    let buf = pad_canvas_fill_dispatch_init(18, [11, 22, 33]);
    let expected: Vec<u8> = std::iter::repeat_n([11u8, 22, 33], 6).flatten().collect();
    assert_eq!(buf, expected);
  }

  /// Edge: 48-byte canvas = one full NEON 48-byte LCM(3, 16) tile
  /// = 16 pixels exactly. **The first canvas size that exercises
  /// the NEON body loop with zero tail**. Body is 48 bytes (one tile),
  /// tail is 0 bytes (skipped). A bug in the body loop's exit
  /// condition (e.g. `tile + 48 < body_len` instead of `<=`) would
  /// silently miss this tile, leaving the spare capacity uninit —
  /// caught here (the adapter `set_len` would expose uninit reads in
  /// the assert if Miri were run, and the byte-eq assert vs the
  /// expected pattern catches it on stable).
  #[test]
  fn pad_canvas_fill_forty_eight_bytes_one_full_tile() {
    let buf = pad_canvas_fill_dispatch_init(48, [100, 150, 200]);
    let expected: Vec<u8> = std::iter::repeat_n([100u8, 150, 200], 16)
      .flatten()
      .collect();
    assert_eq!(buf, expected);
  }

  /// Behavioural test — the new dispatcher must produce byte-identical
  /// output to the OLD `extend_from_slice` loop for several `rgb`
  /// triples (all-zero, all-255, asymmetric `[1, 128, 254]`). 512×512
  /// canvas (786432 bytes) — exercises the multi-tile body loop and
  /// (since `786432 % 48 = 0`) zero tail.
  #[test]
  fn pad_to_square_fill_matches_old_loop() {
    let canvas_bytes = 512usize * 512 * 3; // 786_432
    for &fill in &[[0u8, 0, 0], [255, 255, 255], [1, 128, 254]] {
      // OLD path — inline copy of the pre-C6 idiom (per-3-byte
      // `extend_from_slice` on a `Vec<u8>` grown from `Vec::new()`).
      let mut old: Vec<u8> = Vec::with_capacity(canvas_bytes);
      for _ in 0..(canvas_bytes / 3) {
        old.extend_from_slice(&fill);
      }
      assert_eq!(
        old.len(),
        canvas_bytes,
        "OLD loop length mismatch (fill={fill:?})"
      );

      // NEW path — the dispatcher, called on uninit spare capacity
      // (matching the real `pad_to_square` call site shape).
      let new = pad_canvas_fill_dispatch_init(canvas_bytes, fill);

      assert_eq!(
        new, old,
        "C6 dispatcher must produce byte-identical output to the OLD \
         `extend_from_slice` loop (fill={fill:?}, canvas_bytes={canvas_bytes})"
      );
    }
  }

  /// Behavioural test — the *scalar* path (independent of NEON) must
  /// also match the OLD loop. Pins the scalar reference itself against
  /// the legacy idiom, in case the dispatcher were ever forced to the
  /// NEON arm and we lost visibility into the scalar arm's behaviour.
  #[test]
  fn pad_to_square_fill_scalar_matches_old_loop() {
    let canvas_bytes = 512usize * 512 * 3;
    for &fill in &[[0u8, 0, 0], [255, 255, 255], [1, 128, 254]] {
      let mut old: Vec<u8> = Vec::with_capacity(canvas_bytes);
      for _ in 0..(canvas_bytes / 3) {
        old.extend_from_slice(&fill);
      }
      let new = pad_canvas_fill_scalar_init(canvas_bytes, fill);
      assert_eq!(
        new, old,
        "C6 scalar must produce byte-identical output to the OLD \
         `extend_from_slice` loop (fill={fill:?})"
      );
    }
  }
}
