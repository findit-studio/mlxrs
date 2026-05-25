//! C3 — `image_to_array` RGB widen micro-bench.
//!
//! Report-only per the user directive 2026-05-23 (project memory rule
//! **"SIMD ship NEON regardless"**): the NEON kernel ships
//! unconditionally on aarch64 regardless of how it compares to the
//! auto-vectorized scalar. This bench exists as a regression guard.
//!
//! Three sizes — 256² / 1024² / 4096² pixel grids × 3 bytes = ~196 kB /
//! ~3.1 MB / ~50 MB working sets (cross L1 / L2 / DRAM boundaries on
//! M-series Apple silicon).

use std::{hint::black_box, mem::MaybeUninit};

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::vlm::rgb_widen::{rgb_widen, rgb_widen_scalar};

fn gen_rgb_bytes(n: usize) -> Vec<u8> {
  (0..n).map(|i| ((i * 13) % 256) as u8).collect()
}

/// Pre-C3 idiom — `Vec::extend(raw.iter().map(|&b| f32::from(b)))`
/// (LLVM auto-vectorizes this on aarch64).
#[inline(never)]
fn old_extend(src: &[u8]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  buf.extend(src.iter().map(|&b| f32::from(b)));
  buf
}

/// NEW scalar reference — matches dispatcher's scalar arm.
#[inline(never)]
fn new_scalar(src: &[u8]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  rgb_widen_scalar(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

/// NEW dispatcher (NEON on aarch64).
#[inline(never)]
fn new_dispatch(src: &[u8]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  rgb_widen(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

fn bench_rgb_widen(c: &mut Criterion) {
  for &side in &[256usize, 1024, 4096] {
    let bytes = side * side * 3;
    let src = gen_rgb_bytes(bytes);
    let mut group = c.benchmark_group(format!("rgb_widen/{side}x{side}"));
    group.throughput(criterion::Throughput::Bytes(bytes as u64));

    group.bench_function("old_extend", |b| {
      b.iter(|| {
        let v = old_extend(black_box(&src));
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

criterion_group!(benches, bench_rgb_widen);
criterion_main!(benches);
