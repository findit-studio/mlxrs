//! C7 — `save_wav` f32 → i16 quantize micro-bench.
//!
//! Report-only per the user directive 2026-05-23 (project memory rule
//! **"SIMD ship NEON regardless"**): the NEON kernel ships
//! unconditionally on aarch64 regardless of how it compares to the
//! auto-vectorized scalar at any benched size. This bench exists as a
//! regression guard against both a future scalar regression and a
//! future NEON regression.
//!
//! Three sizes — 16k / 256k / 4M f32 samples — span ~32 kB / ~512 kB /
//! ~8 MB working sets (cross L1 / L2 / DRAM boundaries on M-series
//! Apple silicon).

use std::{hint::black_box, mem::MaybeUninit};

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::audio::quantize::{f32_to_i16_quantize, f32_to_i16_quantize_scalar};

/// Deterministic sample generator — spans `[-1.2, 1.2]` so the clamp
/// path is exercised on both sides.
fn gen_samples(n: usize) -> Vec<f32> {
  (0..n)
    .map(|i| {
      let step = 0.0007_f32;
      let v = -1.2 + (i as f32) * step;
      ((v + 1.2).rem_euclid(2.4)) - 1.2
    })
    .collect()
}

/// Pre-C7 idiom — per-sample `clamp + round + as i16`, accumulating
/// into a `Vec<i16>` (the BufWriter cost is excluded — pure quantize).
#[inline(never)]
fn old_loop(src: &[f32]) -> Vec<i16> {
  let mut out: Vec<i16> = Vec::with_capacity(src.len());
  for &s in src {
    let clipped = s.clamp(-1.0, 1.0);
    out.push((clipped * 32_767.0).round() as i16);
  }
  out
}

/// NEW scalar reference — matches the dispatcher's scalar arm.
#[inline(never)]
fn new_scalar(src: &[f32]) -> Vec<i16> {
  let mut out: Vec<i16> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<i16>] = out.spare_capacity_mut();
  f32_to_i16_quantize_scalar(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { out.set_len(src.len()) };
  out
}

/// NEW dispatcher.
#[inline(never)]
fn new_dispatch(src: &[f32]) -> Vec<i16> {
  let mut out: Vec<i16> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<i16>] = out.spare_capacity_mut();
  f32_to_i16_quantize(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { out.set_len(src.len()) };
  out
}

fn bench_quantize(c: &mut Criterion) {
  for &n in &[16_384_usize, 262_144, 4_194_304] {
    let src = gen_samples(n);
    let mut group = c.benchmark_group(format!("quantize_f32_i16/{n}"));
    group.throughput(criterion::Throughput::Bytes((n * 4) as u64));

    group.bench_function("old_loop", |b| {
      b.iter(|| {
        let v = old_loop(black_box(&src));
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

criterion_group!(benches, bench_quantize);
criterion_main!(benches);
