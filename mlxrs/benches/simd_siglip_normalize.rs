//! SigLIP2 NaFlex patchify-normalize fused RGBA → RGB widen + affine
//! micro-bench.
//!
//! Report-only: the NEON kernel ships unconditionally on aarch64
//! regardless of how it compares to the auto-vectorized scalar. This
//! bench exists as a regression guard against both a future scalar
//! regression and a future NEON regression.
//!
//! Three sizes — 256² / 1024² / 4096² pixel grids × 4 RGBA bytes =
//! ~256 kB / ~4.2 MB / ~67 MB input working sets (output is the RGB
//! f32 buffer, 3 floats / pixel), crossing L1 / L2 / DRAM boundaries
//! on M-series Apple silicon.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::vlm::rgba_to_rgb_affine::{rgba_to_rgb_affine, rgba_to_rgb_affine_scalar};

/// The SigLIP2 NaFlex normalize affine (`x / 127.5 - 1.0`).
const SCALE: f32 = 1.0 / 127.5;
const BIAS: f32 = -1.0;

/// Deterministic RGBA input — `(i * 13) % 256` per byte, non-uniform so
/// the optimizer cannot constant-fold the output.
fn gen_rgba(n_pixels: usize) -> Vec<u8> {
  (0..n_pixels * 4).map(|i| ((i * 13) % 256) as u8).collect()
}

/// Original per-pixel scalar idiom — `chunks_exact(4) + chunks_exact_mut(3)`
/// with the non-fused `x * SCALE - 1.0` (separate mul / sub). The
/// `normalize_row_rgba` shape, kept here as the regression baseline; the
/// kernel below is bit-for-bit equal to it.
#[inline(never)]
fn old_scalar_loop(src: &[u8]) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
    dst_px[0] = f32::from(src_px[0]) * SCALE - 1.0;
    dst_px[1] = f32::from(src_px[1]) * SCALE - 1.0;
    dst_px[2] = f32::from(src_px[2]) * SCALE - 1.0;
  }
  dst
}

/// Kernel scalar reference — matches the dispatcher's scalar arm
/// (non-fused `x * scale + bias`, two roundings).
#[inline(never)]
fn new_scalar(src: &[u8]) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  rgba_to_rgb_affine_scalar(src, &mut dst, SCALE, BIAS);
  dst
}

/// NEW dispatcher (NEON on aarch64).
#[inline(never)]
fn new_dispatch(src: &[u8]) -> Vec<f32> {
  let mut dst = vec![0.0f32; (src.len() / 4) * 3];
  rgba_to_rgb_affine(src, &mut dst, SCALE, BIAS);
  dst
}

fn bench_siglip_normalize(c: &mut Criterion) {
  for &side in &[256usize, 1024, 4096] {
    let n_pixels = side * side;
    let in_bytes = n_pixels * 4;
    let src = gen_rgba(n_pixels);
    let mut group = c.benchmark_group(format!("siglip_normalize/{side}x{side}"));
    group.throughput(criterion::Throughput::Bytes(in_bytes as u64));

    group.bench_function("old_scalar_loop", |b| {
      b.iter(|| {
        let v = old_scalar_loop(black_box(&src));
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

criterion_group!(benches, bench_siglip_normalize);
criterion_main!(benches);
