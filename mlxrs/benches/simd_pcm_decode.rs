//! C1 — PCM decode → f32 normalize micro-bench.
//!
//! Report-only per the user directive 2026-05-23 (project memory rule
//! **"SIMD ship NEON regardless"**).

use std::{hint::black_box, mem::MaybeUninit};

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::audio::pcm_decode::{
  S16_INV_SCALE, S32_INV_SCALE, s16_to_f32_normalize, s16_to_f32_normalize_scalar,
  s32_to_f32_normalize, s32_to_f32_normalize_scalar,
};

fn gen_i16(n: usize) -> Vec<i16> {
  (0..n)
    .map(|i| {
      let base = (i as i32 * 257) & 0xFFFF;
      (base as i16).wrapping_sub(i16::MIN / 4)
    })
    .collect()
}

fn gen_i32(n: usize) -> Vec<i32> {
  (0..n)
    .map(|i| {
      let raw = i as i64 * 4_194_303;
      let bounded = (raw % (i32::MAX as i64)) as i32;
      if i % 2 == 0 { bounded } else { -bounded }
    })
    .collect()
}

#[inline(never)]
fn old_loop_s16(src: &[i16]) -> Vec<f32> {
  let mut out = Vec::with_capacity(src.len());
  for &s in src {
    out.push(f32::from(s) / 32768.0);
  }
  out
}

#[inline(never)]
fn new_scalar_s16(src: &[i16]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  s16_to_f32_normalize_scalar(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

#[inline(never)]
fn new_dispatch_s16(src: &[i16]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  s16_to_f32_normalize(&mut spare[..src.len()], src);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

#[inline(never)]
fn old_loop_s32(src: &[i32]) -> Vec<f32> {
  let mut out = Vec::with_capacity(src.len());
  for &s in src {
    out.push((s as f32) * S32_INV_SCALE);
  }
  out
}

#[inline(never)]
fn new_scalar_s32(src: &[i32]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  s32_to_f32_normalize_scalar(&mut spare[..src.len()], src, S32_INV_SCALE);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

#[inline(never)]
fn new_dispatch_s32(src: &[i32]) -> Vec<f32> {
  let mut buf: Vec<f32> = Vec::with_capacity(src.len());
  let spare: &mut [MaybeUninit<f32>] = buf.spare_capacity_mut();
  s32_to_f32_normalize(&mut spare[..src.len()], src, S32_INV_SCALE);
  // SAFETY: kernel writes every slot; cap was sized to `src.len()`.
  unsafe { buf.set_len(src.len()) };
  buf
}

fn bench_s16(c: &mut Criterion) {
  for &n in &[16_384_usize, 262_144, 4_194_304] {
    let src = gen_i16(n);
    let mut group = c.benchmark_group(format!("pcm_s16_to_f32/{n}"));
    group.throughput(criterion::Throughput::Bytes((n * 2) as u64));
    let _ = S16_INV_SCALE; // ensure const is referenced

    group.bench_function("old_loop", |b| {
      b.iter(|| black_box(old_loop_s16(black_box(&src))));
    });
    group.bench_function("new_scalar", |b| {
      b.iter(|| black_box(new_scalar_s16(black_box(&src))));
    });
    group.bench_function("new_dispatch", |b| {
      b.iter(|| black_box(new_dispatch_s16(black_box(&src))));
    });

    group.finish();
  }
}

fn bench_s32(c: &mut Criterion) {
  for &n in &[16_384_usize, 262_144, 4_194_304] {
    let src = gen_i32(n);
    let mut group = c.benchmark_group(format!("pcm_s32_to_f32/{n}"));
    group.throughput(criterion::Throughput::Bytes((n * 4) as u64));

    group.bench_function("old_loop", |b| {
      b.iter(|| black_box(old_loop_s32(black_box(&src))));
    });
    group.bench_function("new_scalar", |b| {
      b.iter(|| black_box(new_scalar_s32(black_box(&src))));
    });
    group.bench_function("new_dispatch", |b| {
      b.iter(|| black_box(new_dispatch_s32(black_box(&src))));
    });

    group.finish();
  }
}

criterion_group!(benches, bench_s16, bench_s32);
criterion_main!(benches);
