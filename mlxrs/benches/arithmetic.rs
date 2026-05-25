//! Arithmetic ops benches — M2 representative set.
//!
//! Each bench pre-constructs the input arrays outside the timing loop and
//! `eval()`s the result inside, so we measure the op call + Metal eval rather
//! than FFI graph construction alone. Shapes are fixed at 1024x1024 f32.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_add(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let b = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("arithmetic");
  group.bench_function("add 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::arithmetic::add(black_box(&a), black_box(&b)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_multiply(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let b = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("arithmetic");
  group.bench_function("multiply 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::arithmetic::multiply(black_box(&a), black_box(&b)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_negative(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("arithmetic");
  group.bench_function("negative 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::arithmetic::negative(black_box(&a)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_sqrt(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("arithmetic");
  group.bench_function("sqrt 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::arithmetic::sqrt(black_box(&a)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

fn bench_exp(c: &mut Criterion) {
  let a = mlxrs::Array::ones::<f32>(&(1024usize, 1024)).unwrap();
  let mut group = c.benchmark_group("arithmetic");
  group.bench_function("exp 1024x1024 f32", |bencher| {
    bencher.iter(|| {
      let mut r = mlxrs::ops::arithmetic::exp(black_box(&a)).unwrap();
      r.eval().unwrap();
      black_box(r);
    });
  });
  group.finish();
}

criterion_group!(
  benches,
  bench_add,
  bench_multiply,
  bench_negative,
  bench_sqrt,
  bench_exp,
);
criterion_main!(benches);
