//! Arithmetic op happy-path tests.
//!
//! Each wrapper gets at least one assertion against a non-trivial scalar/vec
//! value. Method-form is exercised when convenient; free-fn parity is
//! identical so we don't double-test it everywhere.

use mlxrs::Array;

// ───────── Binary ops ─────────

#[test]
fn subtract_2_minus_3_yields_neg1() {
  let a = Array::full::<f32>(&(2, 2), 2.0).unwrap();
  let b = Array::full::<f32>(&(2, 2), 3.0).unwrap();
  let mut r = a.subtract(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0; 4]);
}

#[test]
fn multiply_2_times_3_yields_6() {
  let a = Array::full::<f32>(&(3,), 2.0).unwrap();
  let b = Array::full::<f32>(&(3,), 3.0).unwrap();
  let mut r = a.multiply(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![6.0; 3]);
}

#[test]
fn divide_6_over_2_yields_3() {
  let a = Array::full::<f32>(&(2, 2), 6.0).unwrap();
  let b = Array::full::<f32>(&(2, 2), 2.0).unwrap();
  let mut r = a.divide(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0; 4]);
}

#[test]
fn maximum_picks_larger_elementwise() {
  let a = Array::from_slice(&[1.0_f32, 5.0, 2.0, 4.0], &(4,)).unwrap();
  let b = Array::from_slice(&[3.0_f32, 2.0, 6.0, 1.0], &(4,)).unwrap();
  let mut r = a.maximum(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0, 5.0, 6.0, 4.0]);
}

#[test]
fn minimum_picks_smaller_elementwise() {
  let a = Array::from_slice(&[1.0_f32, 5.0, 2.0, 4.0], &(4,)).unwrap();
  let b = Array::from_slice(&[3.0_f32, 2.0, 6.0, 1.0], &(4,)).unwrap();
  let mut r = a.minimum(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 2.0, 1.0]);
}

#[test]
fn power_2_to_3_yields_8() {
  let a = Array::full::<f32>(&(2,), 2.0).unwrap();
  let b = Array::full::<f32>(&(2,), 3.0).unwrap();
  let mut r = a.power(&b).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![8.0, 8.0]);
}

#[test]
fn add_freefn_parity_with_method() {
  // Sanity check that the free-fn and method form produce identical scalar.
  let a = Array::full::<f32>(&(1,), 4.0).unwrap();
  let b = Array::full::<f32>(&(1,), 5.0).unwrap();
  let mut method = a.add(&b).unwrap();
  let mut freefn = mlxrs::ops::arithmetic::add(&a, &b).unwrap();
  assert_eq!(method.item::<f32>().unwrap(), freefn.item::<f32>().unwrap());
}

// ───────── Unary ops ─────────

#[test]
fn negative_flips_sign() {
  let a = Array::from_slice(&[1.0_f32, -2.0, 3.0], &(3,)).unwrap();
  let mut r = a.negative().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn abs_makes_positive() {
  let a = Array::from_slice(&[-1.0_f32, 2.0, -3.0], &(3,)).unwrap();
  let mut r = a.abs().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn sqrt_of_4_yields_2() {
  let a = Array::full::<f32>(&(2,), 4.0).unwrap();
  let mut r = a.sqrt().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
}

#[test]
fn square_of_3_yields_9() {
  let a = Array::full::<f32>(&(2,), 3.0).unwrap();
  let mut r = a.square().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![9.0, 9.0]);
}

#[test]
fn exp_of_0_yields_1() {
  let a = Array::full::<f32>(&(2,), 0.0).unwrap();
  let mut r = a.exp().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn log_of_1_yields_0() {
  let a = Array::full::<f32>(&(2,), 1.0).unwrap();
  let mut r = a.log().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![0.0, 0.0]);
}

#[test]
fn sin_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.sin().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn cos_of_0_yields_1() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.cos().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 1.0);
}

#[test]
fn tan_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.tan().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn tanh_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.tanh().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn tanh_of_large_saturates_near_1() {
  // tanh(10) ≈ 0.99999998; quick sanity that it's within 1e-6 of 1.
  let a = Array::full::<f32>(&(1,), 10.0).unwrap();
  let mut r = a.tanh().unwrap();
  let v = r.item::<f32>().unwrap();
  assert!((v - 1.0).abs() < 1e-6, "tanh(10) = {v}, expected ≈ 1.0");
}
