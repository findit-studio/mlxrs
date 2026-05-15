//! M2 piece E: by-value operator overload variants.
//!
//! Each binary op is exercised across all four arity combinations
//! (`&a op &b`, `a op &b`, `&a op b`, `a op b`); `Neg` covers both
//! `-&a` and `-a`. Asserts identical numeric results across forms.
#![cfg(feature = "unstable-ops-overload")]

use mlxrs::Array;

fn fives() -> Array {
  Array::full::<f32>(&(2, 2), 5.0).unwrap()
}

fn twos() -> Array {
  Array::full::<f32>(&(2, 2), 2.0).unwrap()
}

// ───────── Add ─────────

#[test]
fn add_ref_ref() {
  let a = fives();
  let b = twos();
  let mut r = &a + &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

#[test]
fn add_owned_ref() {
  let a = fives();
  let b = twos();
  let mut r = a + &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

#[test]
fn add_ref_owned() {
  let a = fives();
  let b = twos();
  let mut r = &a + b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

#[test]
fn add_owned_owned() {
  let a = fives();
  let b = twos();
  let mut r = a + b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![7.0; 4]);
}

// ───────── Sub ─────────

#[test]
fn sub_ref_ref() {
  let a = fives();
  let b = twos();
  let mut r = &a - &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0; 4]);
}

#[test]
fn sub_owned_ref() {
  let a = fives();
  let b = twos();
  let mut r = a - &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0; 4]);
}

#[test]
fn sub_ref_owned() {
  let a = fives();
  let b = twos();
  let mut r = &a - b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0; 4]);
}

#[test]
fn sub_owned_owned() {
  let a = fives();
  let b = twos();
  let mut r = a - b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![3.0; 4]);
}

// ───────── Mul ─────────

#[test]
fn mul_ref_ref() {
  let a = fives();
  let b = twos();
  let mut r = &a * &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![10.0; 4]);
}

#[test]
fn mul_owned_ref() {
  let a = fives();
  let b = twos();
  let mut r = a * &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![10.0; 4]);
}

#[test]
fn mul_ref_owned() {
  let a = fives();
  let b = twos();
  let mut r = &a * b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![10.0; 4]);
}

#[test]
fn mul_owned_owned() {
  let a = fives();
  let b = twos();
  let mut r = a * b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![10.0; 4]);
}

// ───────── Div ─────────

#[test]
fn div_ref_ref() {
  let a = fives();
  let b = twos();
  let mut r = &a / &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5; 4]);
}

#[test]
fn div_owned_ref() {
  let a = fives();
  let b = twos();
  let mut r = a / &b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5; 4]);
}

#[test]
fn div_ref_owned() {
  let a = fives();
  let b = twos();
  let mut r = &a / b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5; 4]);
}

#[test]
fn div_owned_owned() {
  let a = fives();
  let b = twos();
  let mut r = a / b;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![2.5; 4]);
}

// ───────── Neg ─────────

#[test]
fn neg_ref() {
  let a = Array::from_slice(&[1.0_f32, -2.0, 3.0], &(3,)).unwrap();
  let mut r = -&a;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn neg_owned() {
  let a = Array::from_slice(&[1.0_f32, -2.0, 3.0], &(3,)).unwrap();
  let mut r = -a;
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}
