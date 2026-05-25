//! C4 — `image_to_array` BGR R↔B swap widen micro-bench.
//!
//! Verify-before-claim (§5.4 of `docs/core-arch-simd-candidates.md` +
//! project memory **"Verify review premise empirically"**). Compares
//! the OLD per-pixel `chunks_exact(3) + Vec::push * 3` loop vs the NEW
//! `chunks_exact_mut(3) + MaybeUninit::write` scalar reference vs the
//! NEW NEON `vld3q_u8` + permuted `vst3q_f32` 16-pixel tile at three
//! pixel counts (256² / 1024² / 4096²).
//!
//! **NOTE — this bench's numbers do NOT drive the ship decision for
//! this kernel.** The §5.4 default rule ("ship NEON only if ≥ 2×
//! faster than scalar at 4096²") is **overridden** for C4 per
//! explicit user directive ("do not trust auto-vectorized, please
//! impl the NEON backend"): the NEON kernel ships unconditionally on
//! `aarch64` even though on the M-series sizes measured here it is
//! ~13–15 % *slower* than the auto-vectorized scalar. See the module-
//! level doc of `crate::simd::vlm::bgr_widen` ("Decision — RULE
//! OVERRIDE" paragraph) for the full rationale: auto-vectorization is
//! compiler-version-dependent and the SIMD module's contract is to
//! provide a guaranteed arch-specific kernel that does not depend on
//! LLVM heuristics; other targets / sizes / surrounding code may not
//! auto-vectorize as cleanly as the M-series shape benched here.
//!
//! This bench is **kept in-tree as a regression guard**: a future
//! scalar regression (LLVM heuristic change de-vectorizing the
//! `chunks_exact_mut(3) + MaybeUninit::write` pattern) or a future
//! NEON regression (codegen change in the `vld3q_u8 + vst3q_f32`
//! path) would show up here so we can size the cost. It is **not** a
//! ship gate.
//!
//! Cargo `[profile.bench]` already pins `opt-level = 3`,
//! `codegen-units = 1`, `lto = 'thin'` in the workspace root
//! `Cargo.toml` — no per-bench tuning needed.

use std::{hint::black_box, mem::MaybeUninit};

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::vlm::bgr_widen::{bgr_widen, bgr_widen_scalar};

/// Deterministic BGR input — `(i * 7) % 256` per byte. Same shape as
/// the differential-test generator: every pixel's 3 bytes differ from
/// each other AND from the next pixel's 3 bytes, so the optimizer
/// cannot constant-fold the widen output.
fn gen_bgr_src(n_pixels: usize) -> Vec<u8> {
  (0..n_pixels * 3).map(|i| ((i * 7) % 256) as u8).collect()
}

/// Pre-C4 idiom — `Vec::with_capacity` + per-pixel `chunks_exact(3) +
/// push * 3`. **Not** the dispatcher; this is the legacy code we
/// replaced. The `Vec::with_capacity(n_bytes)` matches the
/// `try_reserve_exact` cap discipline at the call site (one
/// allocation, no realloc inside the loop).
#[inline(never)]
fn old_chunks_push_loop(src: &[u8]) -> Vec<f32> {
  let n_bytes = src.len();
  let mut buf: Vec<f32> = Vec::with_capacity(n_bytes);
  for px in src.chunks_exact(3) {
    buf.push(f32::from(px[2]));
    buf.push(f32::from(px[1]));
    buf.push(f32::from(px[0]));
  }
  buf
}

/// NEW scalar reference — `chunks_exact_mut(3)` per-slot `write` on
/// `Vec<f32>::spare_capacity_mut()`. Matches the dispatcher's scalar
/// arm bit-for-bit. The allocation is `Vec::with_capacity(n_bytes)` +
/// kernel-write + `set_len` — the **exact** shape of the real
/// `image_to_array` call site (matches the type-encoded uninit-safe
/// API).
#[inline(never)]
fn new_scalar(src: &[u8]) -> Vec<f32> {
  let n_bytes = src.len();
  let mut buf: Vec<f32> = Vec::with_capacity(n_bytes);
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  bgr_widen_scalar(&mut spare[..n_bytes], src);
  // SAFETY: `bgr_widen_scalar` wrote every f32 of the first `n_bytes`
  // `MaybeUninit<f32>` slots (function-level contract). `n_bytes <=
  // buf.capacity()` because `Vec::with_capacity(n_bytes)` reserved
  // exactly `n_bytes` slots, so `Vec::set_len`'s preconditions hold.
  unsafe { buf.set_len(n_bytes) };
  buf
}

/// NEW dispatcher — on `aarch64` routes to the NEON 16-pixel
/// `vld3q_u8` + permuted `vst3q_f32` tile, elsewhere to
/// `bgr_widen_scalar`. Same allocation shape as `new_scalar` (and
/// the real call site) so the timing diff isolates the NEON body.
#[inline(never)]
fn new_dispatch(src: &[u8]) -> Vec<f32> {
  let n_bytes = src.len();
  let mut buf: Vec<f32> = Vec::with_capacity(n_bytes);
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  bgr_widen(&mut spare[..n_bytes], src);
  // SAFETY: `bgr_widen` wrote every f32 of the first `n_bytes`
  // `MaybeUninit<f32>` slots (function-level contract). `n_bytes <=
  // buf.capacity()` because `Vec::with_capacity(n_bytes)` reserved
  // exactly `n_bytes` slots, so `Vec::set_len`'s preconditions hold.
  unsafe { buf.set_len(n_bytes) };
  buf
}

fn bench_bgr_widen(c: &mut Criterion) {
  // 256² (≈65k pixels = ≈196 kB src, ≈786 kB dst f32),
  // 1024² (≈1M pixels = ≈3.1 MB src, ≈12.6 MB dst f32),
  // 4096² (≈16.7M pixels = ≈50 MB src, ≈201 MB dst f32) — three
  // orders of magnitude across L1 / L2 / DRAM working sets.
  for &side in &[256usize, 1024, 4096] {
    let n_pixels = side * side;
    let n_bytes = n_pixels * 3;
    let src = gen_bgr_src(n_pixels);
    let mut group = c.benchmark_group(format!("bgr_widen/{side}x{side}"));
    // Tell criterion the input-byte rate so it reports GB/s alongside
    // the wall-clock ns/iter. The output is 4× the input byte count
    // (u8 → f32), but we report input bytes for parity with C6 and
    // because the input is the natural fan-in.
    group.throughput(criterion::Throughput::Bytes(n_bytes as u64));

    group.bench_function("old_chunks_push_loop", |b| {
      b.iter(|| {
        let v = old_chunks_push_loop(black_box(&src));
        black_box(v);
      });
    });
    group.bench_function("new_scalar", |b| {
      b.iter(|| {
        let v = new_scalar(black_box(&src));
        black_box(v);
      });
    });
    group.bench_function("new_dispatch", |b| {
      b.iter(|| {
        let v = new_dispatch(black_box(&src));
        black_box(v);
      });
    });

    group.finish();
  }
}

criterion_group!(benches, bench_bgr_widen);
criterion_main!(benches);
