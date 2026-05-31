use super::{check_axis_count, diagonal, tensordot, tensordot_axes, trace, tril, triu};
use crate::{array::Array, dtype::Dtype, error::Error};

// [[1,2,3],[4,5,6],[7,8,9]] — the shared 3x3 fixture for the structure ops.
fn mat3() -> Array {
  Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap()
}

// ---- direct-argument soundness guards (#259 issue-266 decision A) ----

// The real overflow path needs an axis list with > i32::MAX entries (~8GB+),
// impractical to allocate, so exercise the cap at its boundary with a
// synthetic `len` — mirrors `ops/shape.rs`'s `check_count_boundary`.
#[test]
fn check_axis_count_boundary() {
  assert!(check_axis_count("t", 0).is_ok());
  assert!(check_axis_count("t", i32::MAX as usize).is_ok());
  let over = i32::MAX as usize + 1;
  match check_axis_count("ctx", over) {
    Err(Error::CapExceeded(p)) => {
      assert_eq!(p.context(), "ctx");
      assert_eq!(p.cap(), i32::MAX as u64);
      assert_eq!(p.observed(), over as u64);
    }
    other => panic!("expected Err(CapExceeded) one past the cap, got {other:?}"),
  }
}

// core `diagonal` computes `std::max(-offset, 0)`; negating i32::MIN is UB, so
// the wrapper rejects it as a typed OutOfRange on a normal 2-D input (no FFI).
#[test]
fn diagonal_offset_i32_min_is_typed_error() {
  match diagonal(&mat3(), i32::MIN, 0, 1) {
    Err(Error::OutOfRange(p)) => assert_eq!(p.context(), "diagonal: offset"),
    other => panic!("expected OutOfRange for i32::MIN offset, got {other:?}"),
  }
  // A normal offset is unaffected (the guard does not over-reject).
  assert!(diagonal(&mat3(), 1, 0, 1).is_ok());
  assert!(diagonal(&mat3(), -1, 0, 1).is_ok());
}

// `trace` delegates to `diagonal`, so the same i32::MIN offset rejection holds.
#[test]
fn trace_offset_i32_min_is_typed_error() {
  match trace(&mat3(), i32::MIN, 0, 1, None) {
    Err(Error::OutOfRange(p)) => assert_eq!(p.context(), "trace: offset"),
    other => panic!("expected OutOfRange for i32::MIN offset, got {other:?}"),
  }
  assert!(trace(&mat3(), 0, 0, 1, None).is_ok());
}

// core `tril` -> `tri` builds `arange(-k, m - k)`: `-k` is UB at i32::MIN and
// `m - k` overflows i32 for very-negative k. Both are rejected as a typed
// ArithmeticOverflow on a normal 2-D input.
#[test]
fn tril_k_overflow_is_typed_error() {
  match tril(&mat3(), i32::MIN) {
    Err(Error::ArithmeticOverflow(p)) => assert_eq!(p.context(), "tril: k"),
    other => panic!("expected ArithmeticOverflow for i32::MIN k, got {other:?}"),
  }
  // k = i32::MIN + 1: `-k` is representable but `m - k` (3 - (i32::MIN+1))
  // overflows i32 -> still rejected (exercises the stop-endpoint guard).
  assert!(matches!(
    tril(&mat3(), i32::MIN + 1),
    Err(Error::ArithmeticOverflow(_))
  ));
  assert!(tril(&mat3(), 0).is_ok());
  assert!(tril(&mat3(), -1).is_ok());
}

// `triu` computes `k - 1` first (UB at i32::MIN), then the same `tri` endpoints.
#[test]
fn triu_k_overflow_is_typed_error() {
  match triu(&mat3(), i32::MIN) {
    Err(Error::ArithmeticOverflow(p)) => assert_eq!(p.context(), "triu: k - 1"),
    other => panic!("expected ArithmeticOverflow for i32::MIN k, got {other:?}"),
  }
  // k = i32::MIN + 1: `k - 1` == i32::MIN, whose negation in `tri` overflows ->
  // caught by the shared endpoint guard.
  assert!(matches!(
    triu(&mat3(), i32::MIN + 1),
    Err(Error::ArithmeticOverflow(_))
  ));
  assert!(triu(&mat3(), 0).is_ok());
  assert!(triu(&mat3(), 1).is_ok());
}

#[test]
fn tensordot_int_full_contraction() {
  // axis=2 contracts both axes of two 2x2 matrices: sum of the elementwise
  // product = 1*1 + 2*2 + 3*3 + 4*4 = 30.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let mut c = tensordot(&a, &b, 2).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
}

#[test]
fn tensordot_int_zero_axes_is_outer() {
  // axis=0 contracts nothing -> outer product, shape (2,)+(2,) = (2,2):
  // outer([1,2],[3,4]) = [[3,4],[6,8]].
  let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
  let b = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
  let mut c = tensordot(&a, &b, 0).unwrap();
  assert_eq!(c.shape(), vec![2, 2]);
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![3.0, 4.0, 6.0, 8.0]);
}

#[test]
fn tensordot_int_one_axis_is_matmul() {
  // For 2-D operands, axis=1 contracts a's last with b's first -> matmul.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
  let mut c = tensordot(&a, &b, 1).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensordot_int_negative_axis_errors() {
  // The C++ int form rejects axis < 0 (ops.cpp ~5371) -> typed Err, no panic.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  assert!(tensordot(&a, &b, -1).is_err());
}

#[test]
fn tensordot_axes_matmul_equivalent() {
  // Contract a's axis 1 with b's axis 0 -> standard matmul.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
  let mut c = tensordot_axes(&a, &b, &[1], &[0]).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensordot_axes_full_contraction() {
  // Contract both axes pairwise: 1*1+2*2+3*3+4*4 = 30.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let mut c = tensordot_axes(&a, &b, &[0, 1], &[0, 1]).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![30.0]);
}

#[test]
fn tensordot_axes_negative_axis_matches_matmul() {
  // a's axis -1 (== 1) contracted with b's axis 0 -> matmul, exercising the
  // C-side negative-axis normalization on the axes-lists path.
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
  let mut c = tensordot_axes(&a, &b, &[-1], &[0]).unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensordot_axes_length_mismatch_is_typed_error() {
  let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
  match tensordot_axes(&a, &b, &[0, 1], &[0]).unwrap_err() {
    Error::LengthMismatch(p) => {
      assert_eq!(p.expected(), 2);
      assert_eq!(p.actual(), 1);
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
}

#[test]
fn diagonal_main() {
  // main diagonal of the 3x3 fixture -> [1,5,9].
  let mut d = diagonal(&mat3(), 0, 0, 1).unwrap();
  assert_eq!(d.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn diagonal_positive_offset() {
  // offset=1 -> super-diagonal [2,6].
  let mut d = diagonal(&mat3(), 1, 0, 1).unwrap();
  assert_eq!(d.to_vec::<f32>().unwrap(), vec![2.0, 6.0]);
}

#[test]
fn diagonal_negative_offset() {
  // offset=-1 -> sub-diagonal [4,8].
  let mut d = diagonal(&mat3(), -1, 0, 1).unwrap();
  assert_eq!(d.to_vec::<f32>().unwrap(), vec![4.0, 8.0]);
}

#[test]
fn diagonal_negative_axes() {
  // axis1=-2, axis2=-1 on a 2-D array are the same as 0,1 -> [1,5,9].
  let mut d = diagonal(&mat3(), 0, -2, -1).unwrap();
  assert_eq!(d.to_vec::<f32>().unwrap(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn trace_main() {
  // trace of the 3x3 fixture = 1+5+9 = 15.
  let mut t = trace(&mat3(), 0, 0, 1, None).unwrap();
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![15.0]);
}

#[test]
fn trace_positive_offset() {
  // offset=1 -> sum of super-diagonal [2,6] = 8.
  let mut t = trace(&mat3(), 1, 0, 1, None).unwrap();
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![8.0]);
}

#[test]
fn trace_negative_offset() {
  // offset=-1 -> sum of sub-diagonal [4,8] = 12.
  let mut t = trace(&mat3(), -1, 0, 1, None).unwrap();
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![12.0]);
}

#[test]
fn trace_explicit_dtype_promotes() {
  // Integer input traced into Float32: 1+4 = 5.0, and the OUTPUT dtype is the
  // requested Float32 (not the input I32) — proving `dtype` is forwarded.
  let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
  let mut t = trace(&a, 0, 0, 1, Some(Dtype::F32)).unwrap();
  assert_eq!(t.dtype().unwrap(), Dtype::F32);
  assert_eq!(t.to_vec::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn trace_default_dtype_is_input_dtype() {
  // dtype=None infers the input dtype: an I32 input yields an I32 trace.
  let a = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]).unwrap();
  let mut t = trace(&a, 0, 0, 1, None).unwrap();
  assert_eq!(t.dtype().unwrap(), Dtype::I32);
  assert_eq!(t.to_vec::<i32>().unwrap(), vec![5]);
}

#[test]
fn tril_k_zero() {
  // Lower triangle incl. main diagonal: zeros strictly above.
  let mut l = tril(&mat3(), 0).unwrap();
  assert_eq!(
    l.to_vec::<f32>().unwrap(),
    vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
  );
}

#[test]
fn tril_k_positive() {
  // k=1 also keeps the first super-diagonal.
  let mut l = tril(&mat3(), 1).unwrap();
  assert_eq!(
    l.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  );
}

#[test]
fn tril_k_negative() {
  // k=-1 drops the main diagonal too.
  let mut l = tril(&mat3(), -1).unwrap();
  assert_eq!(
    l.to_vec::<f32>().unwrap(),
    vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
  );
}

#[test]
fn triu_k_zero() {
  // Upper triangle incl. main diagonal: zeros strictly below.
  let mut u = triu(&mat3(), 0).unwrap();
  assert_eq!(
    u.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
  );
}

#[test]
fn triu_k_positive() {
  // k=1 drops the main diagonal, keeps strictly-upper.
  let mut u = triu(&mat3(), 1).unwrap();
  assert_eq!(
    u.to_vec::<f32>().unwrap(),
    vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
  );
}

#[test]
fn triu_k_negative() {
  // k=-1 also keeps the first sub-diagonal.
  let mut u = triu(&mat3(), -1).unwrap();
  assert_eq!(
    u.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]
  );
}

#[test]
fn tril_requires_2d() {
  // 1-D input: the C++ op rejects ndim < 2 -> typed Err, no panic.
  let v = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
  assert!(tril(&v, 0).is_err());
}
