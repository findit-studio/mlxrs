//! M2 piece D — happy-path tests for random ops.
//!
//! Each sampler test fixes a seed via `key`, draws once, and asserts on shape
//! and dtype (and a sanity property such as "all values are bounded"); we
//! avoid direct value checks because the underlying counter-based PRNG is
//! mlx-internal and may shift across mlx-c versions.

use mlxrs::{Array, Dtype, ops::random};

#[test]
fn key_yields_u32_pair() {
  let k = random::key(0).unwrap();
  assert_eq!(k.shape(), vec![2]);
  assert_eq!(k.dtype().unwrap(), Dtype::U32);
}

#[test]
fn split_yields_two_keys() {
  let k = random::key(42).unwrap();
  let (a, b) = random::split(&k).unwrap();
  assert_eq!(a.shape(), vec![2]);
  assert_eq!(b.shape(), vec![2]);
}

#[test]
fn split_num_yields_n_keys() {
  let k = random::key(42).unwrap();
  let s = random::split_num(&k, 4).unwrap();
  assert_eq!(s.shape(), vec![4, 2]);
}

#[test]
fn split_method_form_matches_freefn() {
  let k = random::key(7).unwrap();
  let _ = k.split_key().unwrap();
  let s = k.split_key_num(3).unwrap();
  assert_eq!(s.shape(), vec![3, 2]);
}

#[test]
fn seed_succeeds() {
  random::seed(99).unwrap();
}

#[test]
fn uniform_draws_inside_range() {
  let key = random::key(0).unwrap();
  let low = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let high = Array::from_slice::<f32>(&[1.0], &[0i32; 0]).unwrap();
  let mut u = random::uniform(&low, &high, &(64usize,), Dtype::F32, &key).unwrap();
  assert_eq!(u.shape(), vec![64]);
  let v = u.to_vec::<f32>().unwrap();
  for x in v {
    assert!((0.0..1.0).contains(&x), "uniform out of [0,1): {x}");
  }
}

#[test]
fn normal_draws_have_correct_shape_and_dtype() {
  let key = random::key(1).unwrap();
  let mut n = random::normal(&(8usize, 8), Dtype::F32, 0.0, 1.0, &key).unwrap();
  assert_eq!(n.shape(), vec![8, 8]);
  assert_eq!(n.dtype().unwrap(), Dtype::F32);
  // Force eval — sanity check that materialization works for normal output.
  n.eval().unwrap();
}

#[test]
fn normal_broadcast_uses_array_loc_scale() {
  let key = random::key(2).unwrap();
  let loc = Array::from_slice::<f32>(&[0.0], &[0i32; 0]).unwrap();
  let scale = Array::from_slice::<f32>(&[1.0], &[0i32; 0]).unwrap();
  let n = random::normal_broadcast(&(4usize,), Dtype::F32, &loc, &scale, &key).unwrap();
  assert_eq!(n.shape(), vec![4]);
}

#[test]
fn randint_draws_inside_range() {
  let key = random::key(3).unwrap();
  let low = Array::from_slice::<i32>(&[0], &[0i32; 0]).unwrap();
  let high = Array::from_slice::<i32>(&[10], &[0i32; 0]).unwrap();
  let mut r = random::randint(&low, &high, &(32usize,), Dtype::I32, &key).unwrap();
  assert_eq!(r.shape(), vec![32]);
  let v = r.to_vec::<i32>().unwrap();
  for x in v {
    assert!((0..10).contains(&x), "randint out of [0,10): {x}");
  }
}

#[test]
fn bernoulli_draws_bool() {
  let key = random::key(4).unwrap();
  let p = Array::from_slice::<f32>(&[0.5], &[0i32; 0]).unwrap();
  let b = random::bernoulli(&p, &(8usize,), &key).unwrap();
  assert_eq!(b.shape(), vec![8]);
  assert_eq!(b.dtype().unwrap(), Dtype::Bool);
}

#[test]
fn categorical_yields_u32_indices() {
  let key = random::key(5).unwrap();
  let logits = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &(1, 4)).unwrap();
  let c = random::categorical(&logits, -1, &key).unwrap();
  assert_eq!(c.shape(), vec![1]);
  assert_eq!(c.dtype().unwrap(), Dtype::U32);
}

#[test]
fn categorical_num_samples_yields_multiple_indices() {
  let key = random::key(5).unwrap();
  let logits = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &(1, 4)).unwrap();
  let c = random::categorical_num_samples(&logits, -1, 5, &key).unwrap();
  assert_eq!(c.shape(), vec![1, 5]);
  assert_eq!(c.dtype().unwrap(), Dtype::U32);
}

#[test]
fn categorical_shape_yields_explicit_shape() {
  let key = random::key(6).unwrap();
  let logits = Array::from_slice::<f32>(&[0.1_f32, 0.2, 0.3, 0.4], &(1, 4)).unwrap();
  let c = random::categorical_shape(&logits, -1, &(2usize, 3), &key).unwrap();
  assert_eq!(c.shape(), vec![2, 3]);
  assert_eq!(c.dtype().unwrap(), Dtype::U32);
}

#[test]
fn gumbel_draws_have_correct_shape() {
  let key = random::key(7).unwrap();
  let g = random::gumbel(&(16usize,), Dtype::F32, &key).unwrap();
  assert_eq!(g.shape(), vec![16]);
  assert_eq!(g.dtype().unwrap(), Dtype::F32);
}

#[test]
fn truncated_normal_inside_bounds() {
  let key = random::key(8).unwrap();
  let lower = Array::from_slice::<f32>(&[-1.0], &[0i32; 0]).unwrap();
  let upper = Array::from_slice::<f32>(&[1.0], &[0i32; 0]).unwrap();
  let mut t = random::truncated_normal(&lower, &upper, &(32usize,), Dtype::F32, &key).unwrap();
  let v = t.to_vec::<f32>().unwrap();
  for x in v {
    assert!(
      (-1.0..=1.0).contains(&x),
      "truncated_normal out of bounds: {x}"
    );
  }
}

#[test]
fn multivariate_normal_yields_correct_event_dim() {
  let key = random::key(9).unwrap();
  // mean: shape [2], cov: shape [2,2] identity.
  let mean = Array::zeros::<f32>(&[2i32]).unwrap();
  let cov = Array::eye::<f32>(2, None, 0).unwrap();
  let r = random::multivariate_normal(&mean, &cov, &(8usize,), Dtype::F32, &key).unwrap();
  // Output shape is (8, 2) — sample shape with event dim appended.
  assert_eq!(r.shape(), vec![8, 2]);
}

#[test]
fn multivariate_normal_rejects_empty_covariance() {
  // mlx implements `multivariate_normal` via `linalg::svd(cov, ...)` and only
  // checks `cov.ndim() < 2` / squareness — a `0×0` (or `0×n` / `m×0`) cov would
  // reach the SVD kernel's `a.size() / (m * n)` (`0 / 0`, UB / SIGFPE). The
  // shared SVD-input guard rejects it with `Error::EmptyInput` via a cheap shape
  // check, so the call returns `Err` WITHOUT entering mlx (no `eval`).
  let key = random::key(9).unwrap();
  let mean = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  for dims in [[0i32, 0], [0, 3], [3, 0]] {
    let cov = Array::from_slice::<f32>(&[], &dims).unwrap();
    match random::multivariate_normal(&mean, &cov, &(8usize,), Dtype::F32, &key) {
      Err(mlxrs::Error::EmptyInput(p)) => assert_eq!(
        p.context(),
        "multivariate_normal: covariance matrix has a zero-length row or column dimension"
      ),
      other => panic!("expected EmptyInput for cov {dims:?}, got {other:?}"),
    }
  }
}

#[test]
fn laplace_draws_have_correct_shape() {
  let key = random::key(10).unwrap();
  let l = random::laplace(&(8usize,), Dtype::F32, 0.0, 1.0, &key).unwrap();
  assert_eq!(l.shape(), vec![8]);
}

#[test]
fn bits_yields_uint_array() {
  let key = random::key(11).unwrap();
  let b = random::bits(&(4usize,), 4, &key).unwrap();
  assert_eq!(b.shape(), vec![4]);
}

#[test]
fn permutation_arange_yields_n_unique_indices() {
  let key = random::key(12).unwrap();
  let mut p = random::permutation_arange(8, &key).unwrap();
  assert_eq!(p.shape(), vec![8]);
  // Output dtype is U32 (mlx index-output convention).
  let mut v = p.to_vec::<u32>().unwrap();
  v.sort();
  assert_eq!(v, (0..8u32).collect::<Vec<u32>>());
}

#[test]
fn permutation_method_form_works() {
  let key = random::key(13).unwrap();
  let a = Array::arange(0.0, 5.0, 1.0).unwrap();
  let p = a.permutation(0, &key).unwrap();
  assert_eq!(p.shape(), vec![5]);
}
