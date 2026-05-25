//! Linalg ops benches — M2 representative set.
//!
//! `matmul` on 256x256 f32 (a single bench; bigger matmul benches are M2
//! follow-ups once we have a perf-floor regression budget).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_matmul(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(256usize, 256)).unwrap();
  let b = mlxrs::Array::ones::<f32>(&(256usize, 256)).unwrap();
  let mut group = c.benchmark_group("linalg");
  group.bench_function("matmul 256x256 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::linalg_basic::matmul(black_box(&a), black_box(&b)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
