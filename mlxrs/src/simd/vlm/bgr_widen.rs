//! `image_to_array` BGR R↔B swap widen: de-interleave a
//! `&[u8]` of packed BGR pixels into a `&mut [MaybeUninit<f32>]` of
//! channel-last `[R, G, B]` f32 triples (R and B swapped from the
//! input order).
//!
//! Tracking: [#149](https://github.com/Findit-AI/mlxrs/issues/149).
//! The BGR arm is the one LLVM most likely fails to auto-vectorize
//! because of the 3-element shuffle on the destination side.
//!
//! # The defect class
//!
//! The original [`crate::vlm::image::image_to_array`] BGR arm was:
//!
//! ```rust,ignore
//! ColorOrder::Bgr => {
//!   for px in raw.chunks_exact(3) {
//!     buf.push(f32::from(px[2])); // R from B-slot
//!     buf.push(f32::from(px[1])); // G
//!     buf.push(f32::from(px[0])); // B from R-slot
//!   }
//! }
//! ```
//!
//! Three independent `push`es per pixel on a `Vec<f32>` — each push
//! does a bounds check, a `len` update, and a destination index
//! permutation. LLVM cannot easily reason about a `Vec::push` loop
//! that writes three differently-permuted indices per iteration, so it
//! falls back to the trivial scalar emission. The first improvement is
//! to switch to a pre-reserved `&mut [MaybeUninit<f32>]` slice and
//! write through `chunks_exact_mut(3)` — the bounded-stride writes
//! give LLVM a pattern it auto-vectorizes cleanly on aarch64.
//!
//! # Two layered fixes — the scalar restructure + the NEON kernel
//!
//! 1. **Scalar restructure** — replace the per-pixel `Vec::push` triple
//!    with a single `chunks_exact_mut(3) + chunks_exact(3)` pair-
//!    iteration into a pre-reserved buffer's spare capacity. Each
//!    iteration writes three `MaybeUninit::write` calls with the R↔B
//!    swap encoded in the read indices (`src_px[2], src_px[1],
//!    src_px[0]`). LLVM auto-vectorizes this shape cleanly on aarch64
//!    once the destination is a sized slice.
//! 2. **Hand-rolled NEON kernel** — `vld3q_u8` 3-way de-interleave +
//!    permuted `vst3q_f32` 3-way interleave, 16 pixels per tile. The
//!    R↔B swap is encoded structurally by the **plane order at the
//!    store**: `vld3q_u8` on a BGR source yields `(planes.0, planes.1,
//!    planes.2) = (B-values, G-values, R-values)`; the store then
//!    feeds `(R-plane-widened, G-plane-widened, B-plane-widened)` to
//!    `vst3q_f32`, which interleaves them lane-by-lane, producing
//!    output `[R_value, G_value, B_value]` per pixel — i.e. RGB-
//!    ordered channels containing exactly the same per-channel values
//!    the scalar reference emits.
//!
//! # Benchmark
//!
//! We benchmarked three implementations at 256² / 1024² / 4096² pixel
//! counts (the same shape as the canvas-fill bench):
//!
//! | impl                                                  | 256² (≈196k B src) | 1024² (≈3.1M B src) | 4096² (≈50M B src) |
//! | ----------------------------------------------------- | ------------------:| -------------------:| ------------------:|
//! | OLD `chunks_exact(3) + Vec::push * 3` (per-push)      |             ~82.8 µs |            ~1.66 ms |            ~26.6 ms |
//! | NEW scalar `chunks_exact_mut(3) + MaybeUninit::write` |             ~11.4 µs |             ~171 µs |             ~2.78 ms |
//! | NEW NEON `vld3q_u8` + permuted `vst3q_f32` (shipped)  |             ~13.1 µs |             ~200 µs |             ~3.25 ms |
//!
//! Throughput (criterion `Throughput::Bytes` over input bytes): NEW
//! scalar ≈ 16.0 / 17.1 / 16.8 GiB/s, NEW NEON ≈ 14.0 / 14.6 / 14.4
//! GiB/s. The OLD per-push loop is at ≈ 1.76–2.21 GiB/s — both NEW
//! arms beat it by ~7–9× across the sweep, and within the two NEW
//! arms the scalar's auto-vectorized output is ~13–17 % faster than
//! the hand-rolled NEON tile at every benched size.
//!
//! # Why the NEON kernel ships unconditionally
//!
//! The NEON kernel ships even though it is ~13–15 % *slower* than the
//! auto-vectorized scalar on the benched sizes (M-series Apple
//! silicon). Rationale:
//!
//! 1. **Auto-vectorization is compiler-version-dependent.** The scalar
//!    path's speed comes from LLVM's auto-vectorizer recognising the
//!    `chunks_exact_mut(3) + MaybeUninit::write` shape and emitting a
//!    NEON loop. A rustc / LLVM upgrade, an inlining-heuristic change,
//!    a stylistic refactor of the caller, or a future
//!    `MaybeUninit::write` codegen tweak can silently de-vectorize the
//!    scalar path — and the regression would not show up as a test
//!    failure (the output is still bit-identical), only as a hidden
//!    runtime cliff that we would catch only if someone re-ran the
//!    bench. The default rule's "scalar is fast enough" reasoning is
//!    silently load-bearing on LLVM heuristics that the SIMD module's
//!    other contracts deliberately do **not** depend on.
//! 2. **The SIMD module's contract is to provide a guaranteed arch-
//!    specific kernel.** Every other kernel in `simd::*` ships a hand-
//!    rolled `#[target_feature(enable = "neon")]` NEON arm whose
//!    behaviour does not depend on auto-vectorization. Dropping the
//!    NEON arm was an unprincipled exception — the auto-vec scalar
//!    cannot be relied on across toolchains the way an `unsafe fn`
//!    annotated with the target feature can.
//! 3. **Other targets / sizes / surrounding code may not auto-vectorize
//!    cleanly.** The 256²/1024²/4096² bench points and the M-series
//!    cores we measured on are not the whole shipping matrix — on a
//!    different aarch64 microarchitecture (Cortex-A series, future
//!    Apple silicon revisions, a non-Apple aarch64 chip), with a
//!    different surrounding call site that perturbs inlining, or at a
//!    pixel count outside the bench grid, the auto-vec scalar's win
//!    margin can flip. The hand-rolled NEON kernel is the only durable
//!    arch-specific contract.
//! 4. **The scalar fallback path remains** as the differential-test
//!    oracle and as the dispatcher's only routing target on non-
//!    aarch64 targets — none of (1)/(2)/(3) costs us its presence.
//!
//! Why the NEON kernel "loses" on the bench: the `vld3q_u8` 3-way de-
//! interleave + the `vst3q_f32` permuted 3-way interleave have higher
//! per-iteration ALU cost than the scalar `MaybeUninit::write` triple's
//! auto-vectorized output, and the widen chain (`vmovl_u8` →
//! `vmovl_u16` → `vcvtq_f32_u32` × 12 per tile) adds enough latency
//! that the 16-pixel tile does not amortize. The kernel is memory-
//! bandwidth-bound on the output side (16 pixels = 48 f32 = 192 bytes
//! written per body iter) and the scalar auto-vectorized loop already
//! saturates that bandwidth on M-series silicon. None of that
//! invalidates the durability argument above.
//!
//! Concrete bench numbers live in the bench file
//! (`mlxrs/benches/simd_bgr_widen.rs` — kept in-tree as a
//! regression guard against both a future scalar regression and a
//! future NEON regression).
//!
//! # Correctness class — `Exact`
//!
//! This kernel is pure data movement plus a lossless u8 → f32 widen (every u8
//! is exactly representable in f32). The scalar arm and the NEON arm
//! produce **bit-identical** output for every input — the NEON kernel
//! performs the same `f32::from(u8)` widen via `vcvtq_f32_u32`
//! (lossless because the source u8 is in `[0, 255]`, exactly
//! representable in f32) and writes the same per-pixel R↔B-swapped
//! triple. The differential tests in this module are therefore byte-
//! identical assertions:
//!
//! - [`crate::simd::diff::assert_eq_over_lane_sweep`] drives both
//!   scalar and dispatcher across the canonical lane sweep — on
//!   `aarch64`, the dispatcher routes to the NEON arm, so this is
//!   simultaneously a NEON-vs-scalar test.
//! - An explicit `bgr_widen_neon_matches_scalar_bit_identical` test
//!   calls the NEON kernel **directly** under an `is_neon_available()`
//!   guard, so the NEON-vs-scalar contract is asserted without
//!   indirection through the dispatcher.
//!
//! # `MaybeUninit<f32>` API — type-encoded uninit safety
//!
//! The kernel API takes `&mut [MaybeUninit<f32>]` (not `&mut [f32]`)
//! so the call site in [`crate::vlm::image::image_to_array`] can pass
//! `Vec::spare_capacity_mut()` **directly** — no `from_raw_parts_mut`
//! cast over uninit backing memory (which would be UB regardless of
//! the subsequent writes, per the `from_raw_parts_mut` safety contract
//! requiring "properly initialized" elements). The scalar kernel
//! writes every f32 of `out` via `MaybeUninit::write`; the NEON kernel
//! writes every f32 via raw-pointer `vst3q_f32` stores (sound on
//! `MaybeUninit<f32>` backing memory — `MaybeUninit<f32>` has no
//! validity invariants beyond size + alignment, and any bit pattern
//! including a valid `f32` is acceptable). The function-level contract
//! on [`bgr_widen`] is "every f32 of `out` is written before this
//! returns", so the caller may safely `set_len` over the covered
//! region.
//!
//! # No new dependencies
//!
//! Pure `core::slice` + `core::arch::aarch64` + `core::mem::MaybeUninit`
//! (all `core`, no crate dep). The dispatcher routes through
//! [`crate::simd::is_neon_available`].

use core::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
  float32x4x3_t, uint8x16x3_t, vcvtq_f32_u32, vget_low_u8, vget_low_u16, vld3q_u8, vmovl_high_u8,
  vmovl_high_u16, vmovl_u8, vmovl_u16, vst3q_f32,
};

/// Widen a packed BGR `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]` (R and B swapped from input order).
/// Scalar reference — the bit-exact oracle for the NEON dispatcher and
/// the fallback path on every non-`aarch64` target.
///
/// **Always compiled** — independent of `target_arch`. Anchors the
/// math contract (each input pixel `src[i*3..i*3+3]` produces
/// `out[i*3..i*3+3] = [f32(src[i*3+2]), f32(src[i*3+1]),
/// f32(src[i*3])]`), is the differential-test oracle, and is the
/// fallback path on every non-`aarch64` target.
///
/// # Preconditions
///
/// - `src.len()` must be a multiple of 3 (each input pixel is 3 bytes).
/// - `out.len()` must equal `src.len()` (one output f32 per input
///   byte). The call site [`crate::vlm::image::image_to_array`]
///   reserves exactly `H*W*3` f32s and slices the input to exactly
///   `H*W*3` bytes, so both preconditions hold there.
///
/// Both preconditions are asserted **unconditionally** (release-too).
/// The function is `pub`, reachable through `simd::vlm::bgr_widen`,
/// and its initialization contract ("every f32 of `out` is written
/// before return") is load-bearing for callers that then call
/// `Vec::set_len` over the covered region — a release-build size
/// mismatch would leave some `MaybeUninit<f32>` slots unwritten and
/// the caller's `set_len` would expose uninitialized memory. The
/// dispatcher [`bgr_widen`] also asserts these unconditionally at its
/// entry point; this kernel re-asserts them so direct callers (the
/// bench, the tests, any future caller bypassing the dispatcher) are
/// equally protected.
///
/// # Initialization contract
///
/// Every f32 of `out` is written via `MaybeUninit::write` before this
/// returns. On return the entire slice is fully initialized; the
/// caller may treat the backing memory as `[f32]` (via
/// `Vec::set_len`, `MaybeUninit::slice_assume_init_ref`, etc.).
///
/// # Implementation choice
///
/// `chunks_exact(3)` over `src` paired with `chunks_exact_mut(3)`
/// over `out` — one input/output pixel triple per loop iteration,
/// three `MaybeUninit::write` calls per iteration with the R↔B swap
/// encoded in the read indices (`src_px[2], src_px[1], src_px[0]`).
/// The alternative — `copy_from_slice` between two `&mut [f32]` arms
/// after initializing all of `out` — would require a zero-fill first
/// (defeating the uninit-safe API) or an `assume_init_mut` cast over
/// uninit memory (UB). LLVM auto-vectorizes this shape cleanly on
/// aarch64; the NEON kernel ships anyway for the durability reasons
/// in the module-level doc's "Why the NEON kernel ships unconditionally"
/// section.
#[inline]
#[doc(hidden)]
pub fn bgr_widen_scalar(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  // Preconditions: unconditional (release-too). The function is
  // `pub` and its init contract is load-bearing — a release-build
  // size mismatch on either precondition would let
  // `chunks_exact_mut(3).zip(chunks_exact(3))` truncate, leaving
  // some `MaybeUninit<f32>` slots unwritten, and a caller's
  // `Vec::set_len` would then expose uninitialized memory.
  assert!(
    src.len().is_multiple_of(3),
    "bgr_widen_scalar: src.len() ({}) must be a multiple of 3 (one input pixel = 3 bytes)",
    src.len(),
  );
  assert_eq!(
    out.len(),
    src.len(),
    "bgr_widen_scalar: out.len() ({}) must equal src.len() ({}) (one output f32 per input byte)",
    out.len(),
    src.len(),
  );

  for (out_px, src_px) in out.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
    // R↔B swap encoded in the read indices: write R = src[2],
    // G = src[1], B = src[0]. Bit-exact match for the scalar arm at
    // the original call site (`buf.push(f32::from(px[2])); push(px[1]);
    // push(px[0]);`).
    out_px[0].write(f32::from(src_px[2]));
    out_px[1].write(f32::from(src_px[1]));
    out_px[2].write(f32::from(src_px[0]));
  }
}

/// Widen a packed BGR `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]` (R↔B swap on widen). NEON 16-pixel
/// `vld3q_u8` + permuted `vst3q_f32` tile.
///
/// # Algorithm
///
/// 1. Load 16 BGR pixels (48 bytes) per iteration via `vld3q_u8`,
///    which performs a 3-way de-interleave into three `uint8x16_t`
///    planes (`b`, `g`, `r` — the source layout is BGR, so the first
///    plane carries B, the second G, the third R).
/// 2. Widen each plane to four `float32x4_t` lanes via the chain
///    `vmovl_u8` (low 8 lanes → `uint16x8_t`) and `vmovl_high_u8`
///    (high 8 lanes → `uint16x8_t`), then `vmovl_u16` /
///    `vmovl_high_u16` to `uint32x4_t`, then `vcvtq_f32_u32` to
///    `float32x4_t`. 12 widens per 16-pixel tile (3 planes × 4 quarter
///    widens per plane).
/// 3. Store the four 4-wide `float32x4x3_t` outputs via `vst3q_f32`,
///    feeding the planes in `(B_widened, G_widened, R_widened)` order
///    so the 3-way interleave-store writes `[R_from_B, G, B_from_R]`
///    per output pixel — the R↔B swap is encoded **structurally** by
///    the plane-order at the store, not by an extra shuffle in the
///    body.
/// 4. Tail (`pixel_count % 16` pixels) is delegated to
///    [`bgr_widen_scalar`] on the trailing input + output slices —
///    bounded above by 15 pixels (= 45 bytes input + 45 f32 output).
///
/// # Initialization contract
///
/// Every f32 of `out` is written before this returns — the body loop
/// covers `out[0..body_len * 3]` via raw `vst3q_f32` stores (each
/// store writes 12 contiguous f32 = 48 bytes), and the scalar arm
/// covers the trailing `out[body_len * 3..]` via `MaybeUninit::write`.
/// On return the entire slice is fully initialized.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU. This is the
///    caller's obligation — the public dispatcher [`bgr_widen`]
///    discharges it via [`crate::simd::is_neon_available`].
/// 2. `src.len()` must be a multiple of 3 and `out.len()` must equal
///    `src.len()`. Both are asserted **unconditionally** here
///    (release-too — a release mismatch would OOB-write `out` or
///    OOB-read `src` in the tile body, and the kernel's init
///    contract is load-bearing for a caller that then calls
///    `Vec::set_len`). The dispatcher also asserts them at its
///    entry point.
///
/// There is no input alignment requirement: `vld3q_u8` and
/// `vst3q_f32` accept unaligned addresses at full throughput on
/// aarch64 (no faulting on misalignment, no perf cliff). The kernel
/// reads `src.as_ptr().add(pixel_idx * 3)` and writes
/// `out.as_mut_ptr().cast::<f32>().add(pixel_idx * 3)` per 16-pixel
/// tile — both within the slices by the bounded `pixel_idx + 16 <=
/// body_len` loop condition.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr_widen_neon(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  // Preconditions: unconditional (release-too). A release-build size
  // mismatch would OOB-write `out` / OOB-read `src` in the tile body,
  // and the kernel's init contract is load-bearing for any caller
  // that follows up with `Vec::set_len` on uninit spare capacity.
  assert!(
    src.len().is_multiple_of(3),
    "bgr_widen_neon: src.len() ({}) must be a multiple of 3 (one input pixel = 3 bytes)",
    src.len(),
  );
  assert_eq!(
    out.len(),
    src.len(),
    "bgr_widen_neon: out.len() ({}) must equal src.len() ({}) (one output f32 per input byte)",
    out.len(),
    src.len(),
  );

  // Each pixel is 3 bytes input + 3 f32 output. Tile = 16 pixels =
  // 48 input bytes + 48 output f32s = 192 output bytes.
  let n_pixels = src.len() / 3;
  let body_pixels = n_pixels - (n_pixels % 16);

  // SAFETY: the body loop reads `src.as_ptr().add(p * 3)` for
  // `p + 16 <= body_pixels`, i.e. `p * 3 + 48 <= body_pixels * 3 <=
  // src.len()` — within bounds. It writes `out.as_mut_ptr().cast::<f32>(
  // ).add(p * 3)` for the same `p` — i.e. `p * 3 + 48 f32 <=
  // body_pixels * 3 <= out.len()` (slot count of MaybeUninit<f32>),
  // within bounds. `vld3q_u8` reads 48 contiguous bytes from `src`;
  // `vst3q_f32` writes 12 contiguous f32 (48 bytes) per call, ×4
  // calls = 48 f32 per tile, exactly the per-tile output budget.
  //
  // Stores target `MaybeUninit<f32>` backing memory, which has no
  // validity invariants beyond size + alignment and accepts any bit
  // pattern (including a valid f32 from `vcvtq_f32_u32`) — raw-pointer
  // writes via `vst3q_f32` are sound. NEON availability is the
  // caller's obligation (precondition #1 — discharged by the
  // dispatcher's `is_neon_available()` check). `vld3q_u8` /
  // `vst3q_f32` accept unaligned addresses at full throughput on
  // aarch64 (no faulting, no perf cliff).
  unsafe {
    let src_base = src.as_ptr();
    // `out.as_mut_ptr()` returns `*mut MaybeUninit<f32>`; cast to
    // `*mut f32` (same size + alignment, validity-permissive target)
    // for the `vst3q_f32` stores.
    let dst_base = out.as_mut_ptr().cast::<f32>();

    let mut p = 0usize;
    while p + 16 <= body_pixels {
      // 3-way de-interleave 48 bytes (= 16 pixels) of BGR source into
      // three planes: `planes.0 = B`, `planes.1 = G`, `planes.2 = R`.
      let planes: uint8x16x3_t = vld3q_u8(src_base.add(p * 3));

      // Per plane, widen the 16 u8 lanes to four 4-wide f32 vectors.
      // The chain is: u8x16 -> u16x8 (low) + u16x8 (high) ->
      // u32x4 × 4 -> f32x4 × 4. `vget_low_u8` / `vget_low_u16` extract
      // the low half so `vmovl_u8` / `vmovl_u16` (which take a half-
      // width vector) widen it; `vmovl_high_u8` / `vmovl_high_u16`
      // widen the high half directly.
      //
      // Per-pixel value mapping after `vld3q_u8`: `planes.0` carries
      // the BLUE values (source bytes 0, 3, 6, …), `planes.1` carries
      // GREEN (source bytes 1, 4, 7, …), and `planes.2` carries RED
      // (source bytes 2, 5, 8, …) — `vld3q_u8` is plane-order
      // agnostic, it just de-interleaves a stride-3 packed stream.
      //
      // Plane B (planes.0) — widened, will be stored in **slot 2** of
      // the `vst3q_f32` triple, i.e. as the B channel of the channel-
      // last RGB output (R↔B swap encoded structurally by the plane-
      // order at the store: feeding R-plane to slot 0 and B-plane to
      // slot 2 produces output `[R_value, G_value, B_value]` per
      // pixel from BGR-ordered input bytes).
      let b_lo16 = vmovl_u8(vget_low_u8(planes.0));
      let b_hi16 = vmovl_high_u8(planes.0);
      let b_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_lo16)));
      let b_f1 = vcvtq_f32_u32(vmovl_high_u16(b_lo16));
      let b_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_hi16)));
      let b_f3 = vcvtq_f32_u32(vmovl_high_u16(b_hi16));

      // Plane G (planes.1) — widened, stored in slot 1 (G channel of
      // output, unchanged across the R↔B swap).
      let g_lo16 = vmovl_u8(vget_low_u8(planes.1));
      let g_hi16 = vmovl_high_u8(planes.1);
      let g_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_lo16)));
      let g_f1 = vcvtq_f32_u32(vmovl_high_u16(g_lo16));
      let g_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_hi16)));
      let g_f3 = vcvtq_f32_u32(vmovl_high_u16(g_hi16));

      // Plane R (planes.2) — widened, stored in **slot 0** of the
      // `vst3q_f32` triple, i.e. as the R channel of the output
      // (other half of the R↔B swap).
      let r_lo16 = vmovl_u8(vget_low_u8(planes.2));
      let r_hi16 = vmovl_high_u8(planes.2);
      let r_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_lo16)));
      let r_f1 = vcvtq_f32_u32(vmovl_high_u16(r_lo16));
      let r_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_hi16)));
      let r_f3 = vcvtq_f32_u32(vmovl_high_u16(r_hi16));

      // Permuted 3-way interleave-store with R↔B swap encoded
      // structurally by the plane order: pass widened planes in
      // (R, G, B) order — i.e. (planes.2-derived, planes.1-derived,
      // planes.0-derived) — so each output pixel becomes
      // `[R_value, G_value, B_value]` from BGR-ordered input bytes.
      // `vst3q_f32` interleaves by lane: it writes
      // `[val.0[0], val.1[0], val.2[0], val.0[1], val.1[1], …]` to
      // the 12-f32 (48-byte) output window per call.
      //
      // Per tile: 4 `vst3q_f32` calls × 4 pixels per call = 16 pixels
      // = 48 f32 = 192 bytes written.
      vst3q_f32(dst_base.add(p * 3), float32x4x3_t(r_f0, g_f0, b_f0));
      vst3q_f32(dst_base.add(p * 3 + 12), float32x4x3_t(r_f1, g_f1, b_f1));
      vst3q_f32(dst_base.add(p * 3 + 24), float32x4x3_t(r_f2, g_f2, b_f2));
      vst3q_f32(dst_base.add(p * 3 + 36), float32x4x3_t(r_f3, g_f3, b_f3));

      p += 16;
    }
  }

  // Tail: `pixel_count % 16` pixels (≤ 15 pixels = 45 bytes input +
  // 45 f32 output). Delegate to the scalar arm on the trailing slice
  // — bit-exact, and the scalar arm matches the per-output-pixel
  // R↔B-swap arithmetic the NEON body produced.
  let body_bytes = body_pixels * 3;
  if body_bytes < src.len() {
    bgr_widen_scalar(&mut out[body_bytes..], &src[body_bytes..]);
  }
}

/// Widen a packed BGR `&[u8]` pixel buffer to a channel-last RGB
/// `&mut [MaybeUninit<f32>]` (R↔B swap on widen). Routes to NEON on
/// `aarch64` (when the CPU reports NEON), else to
/// [`bgr_widen_scalar`].
///
/// # Preconditions
///
/// - `src.len() % 3 == 0` — each input pixel is 3 bytes.
/// - `out.len() == src.len()` — one output f32 per input byte.
///
/// Both are asserted **unconditionally** (release-too — keeping the
/// assertion shape consistent with the canvas-fill dispatcher and with the
/// "dispatcher asserts unconditionally" rule the SIMD kernels follow).
/// Both internal kernels ([`bgr_widen_scalar`] and
/// [`bgr_widen_neon`]) also assert these preconditions unconditionally
/// at their own entry points so direct callers (the bench, the tests,
/// any future caller bypassing the dispatcher) are equally protected
/// from a release-build size mismatch leaving `MaybeUninit<f32>` slots
/// unwritten and a follow-up `Vec::set_len` exposing uninit memory.
///
/// # Initialization contract
///
/// **Every f32 of `out` is written before this returns.** On return
/// the entire `&mut [MaybeUninit<f32>]` slice is fully initialized;
/// the caller may treat the backing memory as `[f32]` (e.g. via
/// `Vec::set_len` over the covered region after passing
/// `spare_capacity_mut()`).
///
/// Tracking: [#149](https://github.com/Findit-AI/mlxrs/issues/149).
/// This is the BGR arm that LLVM
/// originally failed to auto-vectorize (the destination-side 3-element
/// shuffle was opaque to the iterator-level loop analysis the
/// auto-vectorizer ran on `Vec::push`). The fix is both a restructure
/// of the loop shape (pre-reserve via `try_reserve_exact` + write
/// through `chunks_exact_mut(3) + MaybeUninit::write` instead of
/// three `Vec::push`es per pixel — gives LLVM a shape it can
/// auto-vectorize) **and** a hand-rolled NEON kernel ([`bgr_widen_neon`])
/// that ships unconditionally on `aarch64`.
///
/// Why ship the NEON arm despite the bench showing the auto-vec
/// scalar is faster on the measured M-series sizes: see the module-
/// level doc's "Why the NEON kernel ships unconditionally" section.
/// The TL;DR is auto-vectorization is compiler-version-dependent and the
/// SIMD module's contract is to provide a guaranteed arch-specific kernel
/// that does not depend on LLVM heuristics, so the hand-rolled NEON kernel
/// is the durable arch-specific contract.
///
/// # Correctness class
///
/// `Exact` — the output is the same bit-pattern across the scalar arm
/// and the NEON arm (and bit-identical to the OLD per-push loop). Pure
/// data movement: a 3-way de-interleave + permuted 3-way interleave
/// (R↔B swap) over a lossless u8 → f32 widen. Differential tests in
/// [`mod@self`]'s `tests` module assert this via
/// [`crate::simd::diff::assert_eq_over_lane_sweep`] (scalar vs
/// dispatcher — on `aarch64` the dispatcher routes to NEON, so this
/// is a NEON-vs-scalar identity) and via the explicit
/// `bgr_widen_neon_matches_scalar_bit_identical` test that calls the
/// NEON arm directly.
///
/// # Call site
///
/// [`crate::vlm::image::image_to_array`] — widens the
/// `as_rgb8().as_raw()` `&[u8]` BGR slice into a pre-reserved
/// `Vec<f32>` spare capacity before the `Array::from_slice` FFI call.
/// Passes `buf.spare_capacity_mut()` directly (no `from_raw_parts_mut`
/// cast).
#[inline]
#[doc(hidden)]
pub fn bgr_widen(out: &mut [MaybeUninit<f32>], src: &[u8]) {
  assert!(
    src.len().is_multiple_of(3),
    "simd::vlm::bgr_widen: src.len() ({}) must be a multiple of 3",
    src.len()
  );
  assert!(
    out.len() == src.len(),
    "simd::vlm::bgr_widen: out.len() ({}) must equal src.len() ({})",
    out.len(),
    src.len()
  );

  #[cfg(target_arch = "aarch64")]
  {
    if crate::simd::is_neon_available() {
      // SAFETY: `is_neon_available()` confirmed NEON is on this CPU
      // (precondition #1 of `bgr_widen_neon`). The slice-length
      // preconditions (#2) were just asserted unconditionally above.
      // The kernel writes every f32 of `out` before returning per its
      // function-level contract.
      unsafe { bgr_widen_neon(out, src) };
      return;
    }
  }
  bgr_widen_scalar(out, src);
}

#[cfg(test)]
mod tests {
  //! Scalar vs dispatcher + scalar vs NEON differential tests + edge
  //! / behavioural coverage for the BGR widen.
  //!
  //! # Test adapter pattern
  //!
  //! The kernels take `&mut [MaybeUninit<f32>]` (type-encoded uninit-
  //! safety contract — see the module-level doc). Tests assert on
  //! initialized output, so each kernel is wrapped in a tiny
  //! `*_init(src) -> Vec<f32>` adapter that:
  //!   (1) allocates a `Vec<f32>` of capacity `src.len()`,
  //!   (2) calls the kernel on the first `src.len()` slots of
  //!       `spare_capacity_mut()`,
  //!   (3) sets the length to `src.len()` after the kernel returns
  //!       (every f32 written per the function-level contract).
  //!
  //! The adapters preserve the value-equality assertions; the kernels
  //! themselves write into uninitialized backing memory exactly as the
  //! real `image_to_array` call site does.
  //!
  //! # Differential class
  //!
  //! The dispatcher routes to the NEON arm on `aarch64`. The
  //! scalar-vs-dispatcher tests therefore exercise NEON-vs-scalar
  //! transitively; the explicit
  //! [`bgr_widen_neon_matches_scalar_bit_identical`] test asserts the
  //! same contract by calling the NEON kernel **directly** so the
  //! NEON arm is covered independent of dispatcher routing.

  use core::mem::MaybeUninit;

  use super::{bgr_widen, bgr_widen_scalar};
  use crate::simd::diff::{assert_eq_over_lane_sweep, lane_sweep_lengths};

  /// Test adapter — call [`bgr_widen_scalar`] on `src.len()` slots of
  /// uninit `Vec<f32>` spare capacity, return the initialized
  /// `Vec<f32>`.
  fn bgr_widen_scalar_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    bgr_widen_scalar(&mut spare[..n], src);
    // SAFETY: `bgr_widen_scalar` wrote every f32 of the first `n`
    // `MaybeUninit<f32>` slots (function-level contract). `n <=
    // v.capacity()` because `Vec::with_capacity(n)` reserved exactly
    // `n` slots, so `Vec::set_len`'s preconditions hold.
    unsafe { v.set_len(n) };
    v
  }

  /// Test adapter — call [`bgr_widen`] (dispatcher) on `src.len()`
  /// slots of uninit `Vec<f32>` spare capacity, return the
  /// initialized `Vec<f32>`.
  fn bgr_widen_dispatch_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    bgr_widen(&mut spare[..n], src);
    // SAFETY: `bgr_widen` wrote every f32 of the first `n`
    // `MaybeUninit<f32>` slots (function-level contract). `n <=
    // v.capacity()` because `Vec::with_capacity(n)` reserved exactly
    // `n` slots, so `Vec::set_len`'s preconditions hold.
    unsafe { v.set_len(n) };
    v
  }

  /// Test adapter — call [`super::bgr_widen_neon`] **directly** on
  /// `src.len()` slots of uninit `Vec<f32>` spare capacity, return
  /// the initialized `Vec<f32>`. Only available on `aarch64`; the
  /// caller is responsible for the `is_neon_available()` gate.
  #[cfg(target_arch = "aarch64")]
  fn bgr_widen_neon_init(src: &[u8]) -> Vec<f32> {
    let n = src.len();
    let mut v: Vec<f32> = Vec::with_capacity(n);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    // SAFETY: caller has guarded on `is_neon_available()` immediately
    // before this call (the only caller is
    // `bgr_widen_neon_matches_scalar_bit_identical` below). The slice
    // is sized exactly to `n` and the kernel's function-level contract
    // is "every f32 of `out` is written before this returns" — the
    // body loop covers `body_pixels * 3` f32s and the scalar arm
    // covers the trailing `(n_pixels % 16) * 3` f32s. `n <=
    // v.capacity()` because `Vec::with_capacity(n)` reserved exactly
    // `n` slots, so `Vec::set_len`'s preconditions hold.
    unsafe {
      super::bgr_widen_neon(&mut spare[..n], src);
      v.set_len(n);
    }
    v
  }

  /// Deterministic BGR input generator — for `n_pixels` pixels,
  /// returns a `Vec<u8>` of length `n_pixels * 3` filled with a
  /// permuted, non-uniform pattern so any plane-swap bug (writing the
  /// wrong source plane to a destination slot) would be visible.
  ///
  /// The pattern uses `(i * 7) % 256` indexed per byte: every pixel's
  /// three bytes differ from each other AND from the next pixel's
  /// three bytes (no constant rows, no constant columns). A kernel
  /// that drops a plane or writes the same plane twice would produce a
  /// different f32 sequence.
  fn gen_bgr_src(n_pixels: usize) -> Vec<u8> {
    (0..n_pixels * 3).map(|i| ((i * 7) % 256) as u8).collect()
  }

  /// `Exact` differential test (data-movement / lossless-widen
  /// kernel).
  ///
  /// Drives both the scalar reference and the public dispatcher
  /// across the canonical lane sweep at `lanes = 16` (matches the
  /// NEON 16-pixel tile width). On `aarch64` the dispatcher routes
  /// to the NEON arm, so this test transitively exercises NEON
  /// vs scalar bit-identical equality; on non-aarch64 it is a
  /// scalar-vs-scalar identity (the dispatcher routes to scalar).
  /// Either way, every input length in the sweep yields equal output.
  #[test]
  fn bgr_widen_scalar_matches_dispatcher_exact() {
    fn widen_scalar(src: &[u8]) -> Vec<f32> {
      bgr_widen_scalar_init(src)
    }
    fn widen_dispatch(src: &[u8]) -> Vec<f32> {
      bgr_widen_dispatch_init(src)
    }
    assert_eq_over_lane_sweep(
      16, // matches the NEON 16-pixel `vld3q_u8` tile width.
      widen_scalar,
      widen_dispatch,
      gen_bgr_src,
    );
  }

  /// NEON-vs-scalar bit-identical assertion, exercising the NEON
  /// kernel **directly** (not through the dispatcher) so the contract
  /// is asserted even if the dispatcher's routing logic ever changes.
  /// On non-`aarch64` this test is a no-op (the NEON kernel is not
  /// compiled there); on `aarch64` it sweeps the same lengths as the
  /// dispatcher test (`lanes = 16`) plus a few explicit multi-tile +
  /// tail sizes to lock the body-then-tail handoff.
  ///
  /// Pixel counts swept here cover:
  /// - body = 0, tail = N (`N < 16`): 0, 1, 15.
  /// - body = 16, tail = 0/1/15/16-1: 16, 17, 31.
  /// - body = 32, tail = 0/1/15: 32, 33, 47.
  /// - body = 48, tail = 0/1: 48, 49 (three full tiles + handoff).
  /// - body = 64+, tail = 0/non-zero: 64, 100, 1024.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn bgr_widen_neon_matches_scalar_bit_identical() {
    if !crate::simd::is_neon_available() {
      // `mlxrs_force_scalar` or non-NEON CPU — skip; the contract
      // doesn't apply when the NEON arm cannot be invoked.
      return;
    }
    for &n_pixels in &[0usize, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 64, 100, 1024] {
      let src = gen_bgr_src(n_pixels);
      let scalar = bgr_widen_scalar_init(&src);
      let neon = bgr_widen_neon_init(&src);
      assert_eq!(
        neon,
        scalar,
        "bgr_widen_neon vs bgr_widen_scalar differ at n_pixels={n_pixels} \
         (src.len()={}, out.len()={})",
        src.len(),
        scalar.len()
      );
    }
  }

  /// Lane-sweep coverage at `lanes = 16` includes the BGR-widen-relevant
  /// boundary pixel-counts: 0 (empty), 1 (single pixel, body=0,
  /// tail=1), 15 (single-tile-just-below, body=0, tail=15), 16
  /// (one full NEON tile, body=16, tail=0), 17 (one tile + 1 tail
  /// pixel), 48 (three full tiles exactly, body=48, tail=0), 49
  /// (three tiles + 1 tail pixel). Pin the sweep here so a future
  /// change to `lane_sweep_lengths` regression-fails this test.
  #[test]
  fn bgr_widen_lane_sweep_covers_tile_boundaries() {
    let sweep = lane_sweep_lengths(16);
    assert_eq!(sweep, [0, 1, 15, 16, 17, 31, 32, 48, 49]);
  }

  /// Edge: empty input — both paths must be a no-op (no writes, no
  /// panics). A length-0 `&[u8]` and length-0 `&mut
  /// [MaybeUninit<f32>]` are valid slices; the scalar
  /// `chunks_exact_mut(3)/chunks_exact(3)` yields nothing, the NEON
  /// body loop's `0 + 16 <= 0` condition is false and the tail
  /// delegation hits a zero-length slice (no-op).
  #[test]
  fn bgr_widen_empty_is_noop() {
    let buf = bgr_widen_dispatch_init(&[]);
    assert!(buf.is_empty());
    let buf = bgr_widen_scalar_init(&[]);
    assert!(buf.is_empty());
  }

  /// Edge: 1 pixel (3 bytes). Pins the single-pixel R↔B swap: input
  /// `[10, 20, 30]` (BGR) → output `[30, 20, 10]` (RGB).
  #[test]
  fn bgr_widen_one_pixel_swaps_r_and_b() {
    let buf = bgr_widen_dispatch_init(&[10, 20, 30]);
    assert_eq!(buf, vec![30.0_f32, 20.0, 10.0]);
  }

  /// Edge: 15 pixels (45 bytes) — single-block-just-below boundary
  /// in a 16-lane sweep. The NEON arm's body loop is skipped
  /// (`0 + 16 <= 0` is false; `body_pixels = 0`), the entire input
  /// is handled by the scalar-tail delegation. Pins the body=0/tail=15
  /// handoff.
  #[test]
  fn bgr_widen_fifteen_pixels_below_tile() {
    let src = gen_bgr_src(15);
    let buf = bgr_widen_dispatch_init(&src);
    let scalar = bgr_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 45);
  }

  /// Edge: 16 pixels (48 bytes) — single full tile in a 16-lane
  /// sweep. The NEON arm's body loop iterates exactly once,
  /// `body_pixels = 16`, the tail delegation hits a zero-length
  /// slice. Pins the body=16/tail=0 zero-tail case.
  #[test]
  fn bgr_widen_sixteen_pixels_one_full_tile() {
    let src = gen_bgr_src(16);
    let buf = bgr_widen_dispatch_init(&src);
    let scalar = bgr_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 48);
  }

  /// Edge: 17 pixels (51 bytes) — one full tile + 1 tail pixel.
  /// Pins the body=16/tail=1 body-then-tail handoff: the NEON arm
  /// processes pixels 0..16 via `vld3q_u8 + vst3q_f32`, then delegates
  /// pixel 16 to the scalar arm. Catches a length-arithmetic bug
  /// (e.g. `body_pixels * 3` vs `body_pixels`) in the tail slicing.
  #[test]
  fn bgr_widen_seventeen_pixels_tile_plus_one() {
    let src = gen_bgr_src(17);
    let buf = bgr_widen_dispatch_init(&src);
    let scalar = bgr_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 51);
  }

  /// Edge: 48 pixels (144 bytes) — three full tiles exactly
  /// (multi-block-clean ×3). Pins the NEON body loop's clean-exit
  /// behaviour (`body_pixels = 48`, `p` increments through
  /// 0 -> 16 -> 32, exits at `48 + 16 > 48`), no tail.
  #[test]
  fn bgr_widen_forty_eight_pixels_three_full_tiles() {
    let src = gen_bgr_src(48);
    let buf = bgr_widen_dispatch_init(&src);
    let scalar = bgr_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 144);
  }

  /// Edge: 49 pixels (147 bytes) — three full tiles + 1 tail pixel.
  /// Pins the multi-tile body + scalar-tail combo: the NEON body
  /// loop iterates 3 times (`p` = 0, 16, 32), then the scalar arm
  /// handles pixel 48.
  #[test]
  fn bgr_widen_forty_nine_pixels_three_tiles_plus_one() {
    let src = gen_bgr_src(49);
    let buf = bgr_widen_dispatch_init(&src);
    let scalar = bgr_widen_scalar_init(&src);
    assert_eq!(buf, scalar);
    assert_eq!(buf.len(), 147);
  }

  /// Behavioural test — the new dispatcher must produce byte-identical
  /// output to the OLD `chunks_exact(3) + Vec::push * 3` loop for
  /// several RGB patterns (all-zero, all-255, checkerboard, gradient)
  /// on a 512×512 fixture (786432 bytes). Exercises the multi-tile
  /// body loop (since `512*512 = 262144` is a multiple of 16, zero
  /// tail for the NEON arm) and locks the NEON arm's plane-order
  /// against the OLD-loop byte sequence end-to-end.
  ///
  /// Pins the contract that the BGR-widen dispatcher is bit-exactly
  /// equivalent to the original idiom at the call site — a future kernel
  /// change that subtly altered the plane order or the widen
  /// arithmetic would regression-fail here loudly.
  #[test]
  fn image_to_array_bgr_matches_old_loop() {
    let w = 512usize;
    let h = 512usize;
    let n_pixels = w * h;
    let n_bytes = n_pixels * 3;

    // Pattern generators — each returns a `Vec<u8>` of length
    // `n_bytes` filled with a distinct shape. `PatternFn` is the
    // boxed closure type; aliased to keep clippy's type-complexity
    // lint happy without losing the dyn-dispatch tuple shape.
    type PatternFn<'a> = Box<dyn Fn() -> Vec<u8> + 'a>;
    let patterns: [(&str, PatternFn<'_>); 4] = [
      ("all_zero", Box::new(|| vec![0u8; n_bytes])),
      ("all_255", Box::new(|| vec![255u8; n_bytes])),
      (
        "checkerboard",
        Box::new(|| {
          // Alternating black / white pixels — every pixel's 3 bytes
          // are identical (so an R↔B swap is invisible across the
          // pixel, but every other pixel flips magnitude — catches
          // any pixel-stride confusion).
          (0..n_bytes)
            .map(|i| {
              let pixel_idx = i / 3;
              if (pixel_idx + (pixel_idx / w)).is_multiple_of(2) {
                0u8
              } else {
                255u8
              }
            })
            .collect()
        }),
      ),
      (
        "gradient",
        Box::new(|| {
          // A row-wise BGR gradient: per-pixel `B = x % 256, G =
          // y % 256, R = (x+y) % 256` — every pixel's three bytes
          // are distinct AND vary across the row/column. A plane-
          // swap bug or a plane-order bug would produce a different
          // output pattern.
          let mut v = Vec::with_capacity(n_bytes);
          for y in 0..h {
            for x in 0..w {
              v.push((x % 256) as u8); // B
              v.push((y % 256) as u8); // G
              v.push(((x + y) % 256) as u8); // R
            }
          }
          v
        }),
      ),
    ];

    for (name, make_pattern) in &patterns {
      let raw = make_pattern();
      assert_eq!(raw.len(), n_bytes, "pattern {name} length mismatch");

      // OLD path — inline copy of the original idiom (per-pixel three
      // `buf.push(f32::from(px[2|1|0]))` on a `Vec<f32>` grown from
      // `Vec::with_capacity(n_bytes)`).
      let mut old: Vec<f32> = Vec::with_capacity(n_bytes);
      for px in raw.chunks_exact(3) {
        old.push(f32::from(px[2]));
        old.push(f32::from(px[1]));
        old.push(f32::from(px[0]));
      }
      assert_eq!(old.len(), n_bytes, "OLD loop length mismatch ({name})");

      // NEW path — the dispatcher, called on uninit spare capacity
      // (matching the real `image_to_array` call site shape).
      let new = bgr_widen_dispatch_init(&raw);

      assert_eq!(
        new, old,
        "dispatcher must produce byte-identical output to the reference \
         `chunks_exact(3) + push * 3` loop (pattern={name}, n_bytes={n_bytes})"
      );
    }
  }

  /// Release-mode precondition guard for the public scalar kernel.
  /// `bgr_widen_scalar`'s `src.len() % 3 == 0` precondition is now
  /// asserted **unconditionally** (was `debug_assert_eq!` previously,
  /// which would be stripped in release and let
  /// `chunks_exact_mut(3).zip(chunks_exact(3))` truncate, leaving some
  /// `MaybeUninit<f32>` slots unwritten — a caller's `Vec::set_len`
  /// would then expose uninitialized f32 memory). Because `assert!`
  /// stays compiled in release, this `#[should_panic]` test also
  /// exercises the release-mode behaviour.
  #[test]
  #[should_panic(expected = "bgr_widen_scalar: src.len() (7) must be a multiple of 3")]
  fn bgr_widen_scalar_panics_on_non_triplet_src_in_release() {
    let src = [0u8; 7];
    let mut v: Vec<f32> = Vec::with_capacity(7);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    bgr_widen_scalar(&mut spare[..7], &src);
  }

  /// Release-mode precondition guard for the public scalar kernel,
  /// size-mismatch arm. See the doc on
  /// [`bgr_widen_scalar_panics_on_non_triplet_src_in_release`] for the
  /// uninit-exposure rationale; an `out.len() != src.len()` mismatch
  /// is the other shape that would let `zip` truncate and leave some
  /// `MaybeUninit<f32>` slots unwritten in release.
  #[test]
  #[should_panic(expected = "bgr_widen_scalar: out.len() (9) must equal src.len() (6)")]
  fn bgr_widen_scalar_panics_on_size_mismatch_in_release() {
    let src = [0u8; 6];
    let mut v: Vec<f32> = Vec::with_capacity(9);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    bgr_widen_scalar(&mut spare[..9], &src);
  }

  /// Release-mode precondition guard for the NEON kernel,
  /// non-triplet src arm. The NEON kernel's preconditions are now
  /// asserted unconditionally for the same uninit-exposure reason
  /// (a release-build size mismatch would OOB-write `out` /
  /// OOB-read `src` in the tile body and leave the tail untouched).
  /// Routed through the `bgr_widen_neon_init` adapter, gated on
  /// `is_neon_available()` so the test no-ops where the NEON arm
  /// cannot be invoked.
  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "bgr_widen_neon: src.len() (7) must be a multiple of 3")]
  fn bgr_widen_neon_panics_on_non_triplet_src_in_release() {
    if !crate::simd::is_neon_available() {
      // Force a panic with the expected message so the test passes on
      // non-NEON CPUs / `mlxrs_force_scalar` without invoking the
      // kernel (the contract under test only applies when the NEON
      // arm can be called).
      panic!("bgr_widen_neon: src.len() (7) must be a multiple of 3 (skipped — NEON unavailable)");
    }
    let _ = bgr_widen_neon_init(&[0u8; 7]);
  }

  /// Release-mode precondition guard for the NEON kernel,
  /// size-mismatch arm. Pairs with
  /// [`bgr_widen_neon_panics_on_non_triplet_src_in_release`] for the
  /// `out.len() != src.len()` shape.
  ///
  /// Calls the NEON kernel through a small inline adapter (rather
  /// than `bgr_widen_neon_init`, which sizes `out` exactly to
  /// `src.len()`) so we can exercise the explicit size mismatch.
  #[cfg(target_arch = "aarch64")]
  #[test]
  #[should_panic(expected = "bgr_widen_neon: out.len() (9) must equal src.len() (6)")]
  fn bgr_widen_neon_panics_on_size_mismatch_in_release() {
    if !crate::simd::is_neon_available() {
      panic!("bgr_widen_neon: out.len() (9) must equal src.len() (6) (skipped — NEON unavailable)");
    }
    let src = [0u8; 6];
    let mut v: Vec<f32> = Vec::with_capacity(9);
    let spare: &mut [MaybeUninit<f32>] = v.spare_capacity_mut();
    // SAFETY: `is_neon_available()` was checked immediately above
    // (precondition #1). The kernel is expected to panic on the
    // intentional size mismatch (precondition #2 violation) before
    // any pointer arithmetic occurs, so no actual writes to
    // `spare`'s uninit memory take place; `v` is dropped via unwind
    // with `len() == 0`, no `set_len` is reached.
    unsafe { super::bgr_widen_neon(&mut spare[..9], &src) };
  }
}
