//! Shape ops benches — M2 representative set.
//!
//! - `reshape`: 2,048,576 → 1024x1024 (rank-1 to rank-2).
//! - `transpose`: 1024x1024 reverse-permutation (rank-2 swap).
//! - `concatenate`: 4× 256x1024 along axis 0 → 1024x1024.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_reshape(c: &mut Criterion) {
  // 2_048_576 = 2 * 1024 * 1024 / 2; we want 1024*1024 elements total
  let a = mlxrs::Array::ones::<f32>(&(1024usize * 1024,)).unwrap();
  let mut group = c.benchmark_group("shape");
  group.bench_function("reshape 1048576 -> 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::shape::reshape(black_box(&a), &(1024usize, 1024)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_transpose(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("shape");
  group.bench_function("transpose 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::shape::transpose(black_box(&a)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_concatenate(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(256usize, 1024)).unwrap();
  let b = mlxrs::Array::ones::<f32>(&(256usize, 1024)).unwrap();
  let cc = mlxrs::Array::ones::<f32>(&(256usize, 1024)).unwrap();
  let d = mlxrs::Array::ones::<f32>(&(256usize, 1024)).unwrap();
  let inputs = [&a, &b, &cc, &d];
  let mut group = c.benchmark_group("shape");
  group.bench_function("concatenate 4x 256x1024 axis=0 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::shape::concatenate(black_box(&inputs), 0).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

criterion_group!(benches, bench_reshape, bench_transpose, bench_concatenate);
criterion_main!(benches);
