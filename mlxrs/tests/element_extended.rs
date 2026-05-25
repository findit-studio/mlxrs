//! M2a extended Element impls: round-trip each new dtype through
//! `from_slice` → `item` / `to_vec`.

use mlxrs::{Array, Dtype};

#[test]
fn round_trip_u8() {
  let data = [0_u8, 1, 2, 3, 255];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U8);
  assert_eq!(a.to_vec::<u8>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_u16() {
  let data = [0_u16, 1, 2, 3, 65_535];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U16);
  assert_eq!(a.to_vec::<u16>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_u64() {
  let data = [0_u64, 1, 2, 3, u64::MAX];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U64);
  assert_eq!(a.to_vec::<u64>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_i8() {
  let data = [-128_i8, -1, 0, 1, 127];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I8);
  assert_eq!(a.to_vec::<i8>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_i16() {
  let data = [i16::MIN, -1, 0, 1, i16::MAX];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I16);
  assert_eq!(a.to_vec::<i16>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_i64() {
  let data = [i64::MIN, -1, 0, 1, i64::MAX];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I64);
  assert_eq!(a.to_vec::<i64>().unwrap(), data.to_vec());
}

#[test]
fn round_trip_f64() {
  let data = [-1.0_f64, 0.0, 1.0, std::f64::consts::PI, 1e100];
  let mut a = Array::from_slice(&data, &(5,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::F64);
  let v = a.to_vec::<f64>().unwrap();
  assert_eq!(v.len(), data.len());
  for (got, want) in v.iter().zip(data.iter()) {
    assert_eq!(got, want, "f64 round-trip mismatch: got {got}, want {want}");
  }
}

#[test]
fn round_trip_bf16() {
  // bf16 has ~7-bit mantissa, so use values that are exactly representable.
  let data = [
    half::bf16::ZERO,
    half::bf16::ONE,
    half::bf16::from_f32(2.0),
    half::bf16::from_f32(-1.5),
  ];
  let mut a = Array::from_slice(&data, &(4,)).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::BF16);
  assert_eq!(a.to_vec::<half::bf16>().unwrap(), data.to_vec());
}

// ───────── scalar `item` round-trips ─────────

#[test]
fn item_u8_scalar() {
  let mut a = Array::from_slice(&[42_u8], &(1,)).unwrap();
  assert_eq!(a.item::<u8>().unwrap(), 42);
}

#[test]
fn item_u16_scalar() {
  let mut a = Array::from_slice(&[12345_u16], &(1,)).unwrap();
  assert_eq!(a.item::<u16>().unwrap(), 12345);
}

#[test]
fn item_u64_scalar() {
  let mut a = Array::from_slice(&[u64::MAX - 1], &(1,)).unwrap();
  assert_eq!(a.item::<u64>().unwrap(), u64::MAX - 1);
}

#[test]
fn item_i8_scalar() {
  let mut a = Array::from_slice(&[-7_i8], &(1,)).unwrap();
  assert_eq!(a.item::<i8>().unwrap(), -7);
}

#[test]
fn item_i16_scalar() {
  let mut a = Array::from_slice(&[-12345_i16], &(1,)).unwrap();
  assert_eq!(a.item::<i16>().unwrap(), -12345);
}

#[test]
fn item_i64_scalar() {
  let mut a = Array::from_slice(&[i64::MIN + 1], &(1,)).unwrap();
  assert_eq!(a.item::<i64>().unwrap(), i64::MIN + 1);
}

#[test]
fn item_f64_scalar() {
  let mut a = Array::from_slice(&[std::f64::consts::E], &(1,)).unwrap();
  let v = a.item::<f64>().unwrap();
  assert_eq!(v, std::f64::consts::E);
}

#[test]
fn item_bf16_scalar() {
  let val = half::bf16::from_f32(0.5);
  let mut a = Array::from_slice(&[val], &(1,)).unwrap();
  assert_eq!(a.item::<half::bf16>().unwrap(), val);
}

// ───────── dtype-mismatch error path (uses one of the new dtypes) ─────────

#[test]
fn item_dtype_mismatch_returns_err() {
  // Build an i64 array, ask for i32 → DtypeMismatch.
  let mut a = Array::from_slice(&[1_i64, 2, 3], &(3,)).unwrap();
  let r = a.item::<i32>();
  assert!(
    matches!(r, Err(mlxrs::Error::DtypeMismatch { .. })),
    "expected DtypeMismatch when reading i64 array as i32, got {r:?}",
  );
}
