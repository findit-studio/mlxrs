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
