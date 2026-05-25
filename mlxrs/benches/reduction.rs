//! Reduction ops benches — M2 representative set.
//!
//! Full reductions (no axes) over 1024x1024 f32, eval()'d inside the loop.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_sum(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("reduction");
  group.bench_function("sum 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::reduction::sum(black_box(&a), false).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_mean(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("reduction");
  group.bench_function("mean 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::reduction::mean(black_box(&a), false).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_max(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("reduction");
  group.bench_function("max 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::reduction::max(black_box(&a), false).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_prod(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("reduction");
  group.bench_function("prod 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::reduction::prod(black_box(&a), false).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

criterion_group!(benches, bench_sum, bench_mean, bench_max, bench_prod,);
criterion_main!(benches);
