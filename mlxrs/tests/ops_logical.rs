//! Happy-path tests for `ops::logical`.

use mlxrs::Array;

#[test]
fn logical_and_componentwise() {
  let a = Array::from_slice::<bool>(&[true, true, false, false], &(4,)).unwrap();
  let b = Array::from_slice::<bool>(&[true, false, true, false], &(4,)).unwrap();
  let mut r = a.logical_and(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, false, false]);
}

#[test]
fn logical_or_componentwise() {
  let a = Array::from_slice::<bool>(&[true, true, false, false], &(4,)).unwrap();
  let b = Array::from_slice::<bool>(&[true, false, true, false], &(4,)).unwrap();
  let mut r = a.logical_or(&b).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, true, true, false]);
}

#[test]
fn logical_not_inverts_each_bit() {
  let a = Array::from_slice::<bool>(&[true, false, true, false], &(4,)).unwrap();
  let mut r = a.logical_not().unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true, false, true]);
}

#[test]
fn select_picks_x_where_condition_true() {
  // condition = [T, F, T, F], x = [10,20,30,40], y = [-1,-2,-3,-4]
  // result    = [10,-2,30,-4]
  let cond = Array::from_slice::<bool>(&[true, false, true, false], &(4,)).unwrap();
  let x = Array::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0], &(4,)).unwrap();
  let y = Array::from_slice::<f32>(&[-1.0, -2.0, -3.0, -4.0], &(4,)).unwrap();
  let mut r = cond.select(&x, &y).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![10.0, -2.0, 30.0, -4.0]);
}
