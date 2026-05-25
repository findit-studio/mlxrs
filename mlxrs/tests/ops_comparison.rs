//! Happy-path tests for `ops::comparison` (Phase 4 Branch C).
//!
//! Element-wise comparisons return Bool arrays; query ops (`isfinite`/etc.)
//! exercise the unary template; `allclose`/`isclose` exercise the
//! tolerance-bearing trinary template (rtol/atol/equal_nan).

use mlxrs::Array;

#[test]
fn equal_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 9.0, 3.0, 0.0], &(4,)).unwrap();
  let mut r = a.equal(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, true, false]);
}

#[test]
fn not_equal_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 0.0, 3.0], &(3,)).unwrap();
  let mut r = a.not_equal(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true, false]);
}

#[test]
fn less_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[2.0, 2.0, 0.0], &(3,)).unwrap();
  let mut r = a.less(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, false]);
}

#[test]
fn less_equal_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[2.0, 2.0, 0.0], &(3,)).unwrap();
  let mut r = a.less_equal(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, true, false]);
}

#[test]
fn greater_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[2.0, 2.0, 0.0], &(3,)).unwrap();
  let mut r = a.greater(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, false, true]);
}

#[test]
fn greater_equal_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[2.0, 2.0, 0.0], &(3,)).unwrap();
  let mut r = a.greater_equal(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn allclose_within_tol_returns_true() {
  // Differences ~1e-5 are well within the default rtol=1e-5, atol=1e-8 envelope.
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.000001, 2.000001, 3.000001], &(3,)).unwrap();
  let mut r = a.allclose(&b, 1e-5, 1e-8, false).unwrap();
  assert!(r.item::<bool>().unwrap());
}

#[test]
fn allclose_outside_tol_returns_false() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 2.5, 3.0], &(3,)).unwrap();
  let mut r = a.allclose(&b, 1e-5, 1e-8, false).unwrap();
  assert!(!r.item::<bool>().unwrap());
}

#[test]
fn isclose_componentwise() {
  let a = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.000001, 9.0, 3.0], &(3,)).unwrap();
  let mut r = a.isclose(&b, 1e-5, 1e-8, false).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn isfinite_marks_inf_and_nan_as_false() {
  let a = Array::from_slice::<f32>(
    &[0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    &(5,),
  )
  .unwrap();
  let mut r = a.isfinite().unwrap();
  assert_eq!(
    r.to_vec::<bool>().unwrap(),
    vec![true, true, false, false, false]
  );
}

#[test]
fn isinf_marks_only_infinities() {
  let a = Array::from_slice::<f32>(
    &[0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    &(5,),
  )
  .unwrap();
  let mut r = a.isinf().unwrap();
  assert_eq!(
    r.to_vec::<bool>().unwrap(),
    vec![false, false, true, true, false]
  );
}

#[test]
fn isnan_marks_only_nan() {
  let a = Array::from_slice::<f32>(
    &[0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    &(5,),
  )
  .unwrap();
  let mut r = a.isnan().unwrap();
  assert_eq!(
    r.to_vec::<bool>().unwrap(),
    vec![false, false, false, false, true]
  );
}
