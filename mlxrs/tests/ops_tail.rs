//! M2a long-tail op happy-path tests.
//!
//! One assertion per new op against a non-trivial scalar/vec value. Method
//! and free-fn share the same impl so method form is preferred for coverage.

use mlxrs::Array;

// ───────── reduction tail: var / std / all / any / logsumexp ─────────

#[test]
fn var_of_constant_is_zero() {
  // Variance of a constant is 0 (regardless of ddof < n).
  let a = Array::full::<f32>(&(2, 2), 3.0).unwrap();
  let mut r = a.var(false, 0).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn var_axes_along_axis0() {
  // [[1, 2], [3, 4]] var(ddof=0) over axis 0 → mean of squared dev from col mean
  //   col0: mean=2, var=((1-2)^2 + (3-2)^2)/2 = 1
  //   col1: mean=3, var=((2-3)^2 + (4-3)^2)/2 = 1
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
  let mut r = a.var_axes(&[0], false, 0).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn var_axes_empty_promotes_int_to_float() {
  // Empty axes must route through MLX (not try_clone) so dtype promotion runs.
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  let r = a.var_axes(&[], false, 0).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::F32);
}

#[test]
fn std_of_constant_is_zero() {
  let a = Array::full::<f32>(&(2, 2), 5.0).unwrap();
  let mut r = a.std(false, 0).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn std_axes_along_axis1() {
  // std of [[1, 2, 3], [4, 5, 6]] across axis 1 with ddof=0:
  //   row0: mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3, std=sqrt(2/3)
  let a = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2, 3)).unwrap();
  let mut r = a.std_axes(&[1], false, 0).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  let expected = (2.0_f32 / 3.0).sqrt();
  assert!((v[0] - expected).abs() < 1e-5, "row0 std = {}", v[0]);
  assert!((v[1] - expected).abs() < 1e-5, "row1 std = {}", v[1]);
}

#[test]
fn std_axes_empty_promotes_int_to_float() {
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  let r = a.std_axes(&[], false, 0).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::F32);
}

#[test]
fn all_true_yields_true() {
  let a = Array::from_slice(&[true, true, true], &(3,)).unwrap();
  let mut r = a.all(false).unwrap();
  assert!(r.item::<bool>().unwrap());
}

#[test]
fn all_with_false_yields_false() {
  let a = Array::from_slice(&[true, false, true], &(3,)).unwrap();
  let mut r = a.all(false).unwrap();
  assert!(!r.item::<bool>().unwrap());
}

#[test]
fn all_axes_along_axis0() {
  // [[true, false], [true, true]] all over axis 0 → [true, false]
  let a = Array::from_slice(&[true, false, true, true], &(2, 2)).unwrap();
  let mut r = a.all_axes(&[0], false).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false]);
}

#[test]
fn any_with_one_true_yields_true() {
  let a = Array::from_slice(&[false, false, true], &(3,)).unwrap();
  let mut r = a.any(false).unwrap();
  assert!(r.item::<bool>().unwrap());
}

#[test]
fn any_axes_along_axis1() {
  // [[false, false], [false, true]] any over axis 1 → [false, true]
  let a = Array::from_slice(&[false, false, false, true], &(2, 2)).unwrap();
  let mut r = a.any_axes(&[1], false).unwrap();
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![false, true]);
}

// Regression: empty `axes` for all/any is NOT a dtype-preserving no-op.
// MLX returns `astype(a, bool)` for empty axes (numpy `all(a, axis=())` is
// bool too). A prior `try_clone` short-circuit silently returned the input's
// int dtype/values for a logical op.
#[test]
fn all_axes_empty_on_int_casts_to_bool() {
  let a = Array::from_slice(&[1i32, 0, 2], &(3,)).unwrap();
  let mut r = a.all_axes(&[], false).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::Bool);
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn any_axes_empty_on_int_casts_to_bool() {
  let a = Array::from_slice(&[1i32, 0, 2], &(3,)).unwrap();
  let mut r = a.any_axes(&[], false).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::Bool);
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn all_axes_empty_on_bool_is_identity() {
  // bool input + empty axes: still bool, values unchanged (no reduction).
  let a = Array::from_slice(&[true, false, true], &(3,)).unwrap();
  let mut r = a.all_axes(&[], false).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::Bool);
  assert_eq!(r.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn logsumexp_of_zeros_yields_log_n() {
  // logsumexp([0, 0, 0, 0]) = log(4*e^0) = log(4)
  let a = Array::zeros::<f32>(&(4,)).unwrap();
  let mut r = a.logsumexp(false).unwrap();
  let v = r.item::<f32>().unwrap();
  let expected = 4.0_f32.ln();
  assert!(
    (v - expected).abs() < 1e-5,
    "logsumexp = {v}, expected = {expected}"
  );
}

#[test]
fn logsumexp_axes_along_axis0() {
  // [[0, 0], [0, 0]] logsumexp over axis 0 → [log(2), log(2)]
  let a = Array::zeros::<f32>(&(2, 2)).unwrap();
  let mut r = a.logsumexp_axes(&[0], false).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  let expected = 2.0_f32.ln();
  assert!((v[0] - expected).abs() < 1e-5);
  assert!((v[1] - expected).abs() < 1e-5);
}

#[test]
fn logsumexp_axes_empty_promotes_int_to_float() {
  let a = Array::from_slice(&[1_i32, 2, 3, 4], &(2, 2)).unwrap();
  let r = a.logsumexp_axes(&[], false).unwrap();
  assert_eq!(r.dtype().unwrap(), mlxrs::Dtype::F32);
}

// ───────── arithmetic tail: unary ─────────

#[test]
fn log10_of_100_yields_2() {
  let a = Array::full::<f32>(&(1,), 100.0).unwrap();
  let mut r = a.log10().unwrap();
  assert!((r.item::<f32>().unwrap() - 2.0).abs() < 1e-5);
}

#[test]
fn log2_of_8_yields_3() {
  let a = Array::full::<f32>(&(1,), 8.0).unwrap();
  let mut r = a.log2().unwrap();
  assert!((r.item::<f32>().unwrap() - 3.0).abs() < 1e-5);
}

#[test]
fn log1p_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.log1p().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn expm1_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.expm1().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn erf_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.erf().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn erfinv_of_0_yields_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.erfinv().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.0);
}

#[test]
fn sigmoid_of_0_yields_half() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut r = a.sigmoid().unwrap();
  assert!((r.item::<f32>().unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn ceil_floor_round_on_fractional() {
  let a = Array::from_slice(&[1.7_f32, -1.7, 2.5, -2.5], &(4,)).unwrap();
  let mut c = a.ceil().unwrap();
  assert_eq!(c.to_vec::<f32>().unwrap(), vec![2.0, -1.0, 3.0, -2.0]);
  let mut f = a.floor().unwrap();
  assert_eq!(f.to_vec::<f32>().unwrap(), vec![1.0, -2.0, 2.0, -3.0]);
  // round(decimals=0) — mlx follows numpy's banker's rounding; for ±2.5 →
  // both round to even (±2.0). We only assert the obvious 1.7/-1.7 cases.
  let mut r = a.round(0).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v[0], 2.0);
  assert_eq!(v[1], -2.0);
}

#[test]
fn round_decimals_truncates_fraction() {
  // round([1.236], decimals=1) → [1.2]
  let a = Array::from_slice(&[1.236_f32], &(1,)).unwrap();
  let mut r = a.round(1).unwrap();
  assert!((r.item::<f32>().unwrap() - 1.2).abs() < 1e-5);
}

#[test]
fn sign_of_mixed_yields_negative_zero_positive() {
  let a = Array::from_slice(&[-2.0_f32, 0.0, 3.0], &(3,)).unwrap();
  let mut r = a.sign().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn reciprocal_of_2_yields_half() {
  let a = Array::full::<f32>(&(1,), 2.0).unwrap();
  let mut r = a.reciprocal().unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 0.5);
}

#[test]
fn rsqrt_of_4_yields_half() {
  let a = Array::full::<f32>(&(1,), 4.0).unwrap();
  let mut r = a.rsqrt().unwrap();
  assert!((r.item::<f32>().unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn conjugate_is_identity_on_real() {
  // Real input: conjugate is the identity (and dtype is preserved).
  let a = Array::from_slice(&[1.0_f32, -2.0, 3.0], &(3,)).unwrap();
  let mut r = a.conjugate().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, -2.0, 3.0]);
}

#[test]
fn real_is_identity_on_real() {
  let a = Array::from_slice(&[1.0_f32, 2.0], &(2,)).unwrap();
  let mut r = a.real().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn imag_of_real_yields_zeros() {
  let a = Array::from_slice(&[1.0_f32, 2.0], &(2,)).unwrap();
  let mut r = a.imag().unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![0.0, 0.0]);
}

#[test]
fn degrees_180_radians_yields_180_degrees() {
  let a = Array::full::<f32>(&(1,), std::f32::consts::PI).unwrap();
  let mut r = a.degrees().unwrap();
  assert!((r.item::<f32>().unwrap() - 180.0).abs() < 1e-3);
}

#[test]
fn radians_180_yields_pi() {
  let a = Array::full::<f32>(&(1,), 180.0).unwrap();
  let mut r = a.radians().unwrap();
  assert!((r.item::<f32>().unwrap() - std::f32::consts::PI).abs() < 1e-5);
}

#[test]
fn sinh_cosh_of_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut sh = a.sinh().unwrap();
  let mut ch = a.cosh().unwrap();
  assert_eq!(sh.item::<f32>().unwrap(), 0.0);
  assert_eq!(ch.item::<f32>().unwrap(), 1.0);
}

#[test]
fn arcsin_arccos_arctan_of_0() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut s = a.arcsin().unwrap();
  let mut c = a.arccos().unwrap();
  let mut t = a.arctan().unwrap();
  assert!((s.item::<f32>().unwrap() - 0.0).abs() < 1e-6);
  assert!((c.item::<f32>().unwrap() - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
  assert!((t.item::<f32>().unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn arcsinh_arccosh_arctanh_basic() {
  // arcsinh(0) = 0; arccosh(1) = 0; arctanh(0) = 0
  let zero = Array::full::<f32>(&(1,), 0.0).unwrap();
  let one = Array::full::<f32>(&(1,), 1.0).unwrap();
  let mut sh = zero.arcsinh().unwrap();
  let mut ch = one.arccosh().unwrap();
  let mut th = zero.arctanh().unwrap();
  assert!((sh.item::<f32>().unwrap() - 0.0).abs() < 1e-6);
  assert!((ch.item::<f32>().unwrap() - 0.0).abs() < 1e-6);
  assert!((th.item::<f32>().unwrap() - 0.0).abs() < 1e-6);
}

#[test]
fn nan_to_num_default_inf_substitutes_finite_extrema() {
  // Build [NaN, +inf, -inf, 1.0]; nan→0, posinf=None (mlx substitutes
  // dtype's finite max), neginf=None (mlx substitutes dtype's finite min).
  let a = Array::from_slice(
    &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0_f32],
    &(4,),
  )
  .unwrap();
  let mut r = a.nan_to_num(0.0, None, None).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  assert_eq!(v[0], 0.0, "NaN should be replaced with 0");
  assert!(
    v[1].is_finite() && v[1] > 0.0,
    "+inf substituted with finite max"
  );
  assert!(
    v[2].is_finite() && v[2] < 0.0,
    "-inf substituted with finite min"
  );
  assert_eq!(v[3], 1.0);
}

#[test]
fn nan_to_num_replaces_all() {
  let a = Array::from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &(3,)).unwrap();
  let mut r = a.nan_to_num(0.0, Some(99.0), Some(-99.0)).unwrap();
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![0.0, 99.0, -99.0]);
}

#[test]
fn bitwise_invert_of_zero_u32_yields_all_ones() {
  let a = Array::from_slice(&[0_u32, 1, 2], &(3,)).unwrap();
  let mut r = a.bitwise_invert().unwrap();
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![!0_u32, !1_u32, !2_u32]);
}

// ───────── arithmetic tail: binary ─────────

#[test]
fn arctan2_quadrants() {
  // arctan2(1, 1) = π/4; arctan2(1, -1) = 3π/4 (Q2); etc.
  let y = Array::from_slice(&[1.0_f32, 1.0], &(2,)).unwrap();
  let x = Array::from_slice(&[1.0_f32, -1.0], &(2,)).unwrap();
  let mut r = y.arctan2(&x).unwrap();
  let v = r.to_vec::<f32>().unwrap();
  let q1 = std::f32::consts::FRAC_PI_4;
  let q2 = 3.0 * std::f32::consts::FRAC_PI_4;
  assert!((v[0] - q1).abs() < 1e-5);
  assert!((v[1] - q2).abs() < 1e-5);
}

#[test]
fn floor_divide_7_over_2_yields_3() {
  let a = Array::full::<f32>(&(1,), 7.0).unwrap();
  let b = Array::full::<f32>(&(1,), 2.0).unwrap();
  let mut r = a.floor_divide(&b).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 3.0);
}

#[test]
fn remainder_7_mod_3_yields_1() {
  let a = Array::full::<f32>(&(1,), 7.0).unwrap();
  let b = Array::full::<f32>(&(1,), 3.0).unwrap();
  let mut r = a.remainder(&b).unwrap();
  assert_eq!(r.item::<f32>().unwrap(), 1.0);
}

#[test]
fn divmod_7_over_3_yields_quotient_and_remainder() {
  let a = Array::from_slice(&[7.0_f32, 8.0], &(2,)).unwrap();
  let b = Array::from_slice(&[3.0_f32, 3.0], &(2,)).unwrap();
  let (mut q, mut r) = a.divmod(&b).unwrap();
  assert_eq!(q.to_vec::<f32>().unwrap(), vec![2.0, 2.0]);
  assert_eq!(r.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn bitwise_and_or_xor_on_u32() {
  let a = Array::from_slice(&[0b1100_u32, 0b1010], &(2,)).unwrap();
  let b = Array::from_slice(&[0b1010_u32, 0b0110], &(2,)).unwrap();
  let mut and = a.bitwise_and(&b).unwrap();
  let mut or = a.bitwise_or(&b).unwrap();
  let mut xor = a.bitwise_xor(&b).unwrap();
  assert_eq!(and.to_vec::<u32>().unwrap(), vec![0b1000, 0b0010]);
  assert_eq!(or.to_vec::<u32>().unwrap(), vec![0b1110, 0b1110]);
  assert_eq!(xor.to_vec::<u32>().unwrap(), vec![0b0110, 0b1100]);
}

#[test]
fn left_shift_doubles_each_step() {
  let a = Array::from_slice(&[1_u32, 2, 3], &(3,)).unwrap();
  let n = Array::from_slice(&[1_u32, 1, 1], &(3,)).unwrap();
  let mut r = a.left_shift(&n).unwrap();
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![2, 4, 6]);
}

#[test]
fn right_shift_halves_each_step() {
  let a = Array::from_slice(&[8_u32, 4, 2], &(3,)).unwrap();
  let n = Array::from_slice(&[1_u32, 1, 1], &(3,)).unwrap();
  let mut r = a.right_shift(&n).unwrap();
  assert_eq!(r.to_vec::<u32>().unwrap(), vec![4, 2, 1]);
}

// ───────── free-fn parity sanity ─────────

#[test]
fn sigmoid_freefn_parity_with_method() {
  let a = Array::full::<f32>(&(1,), 0.0).unwrap();
  let mut method = a.sigmoid().unwrap();
  let mut freefn = mlxrs::ops::arithmetic::sigmoid(&a).unwrap();
  assert_eq!(method.item::<f32>().unwrap(), freefn.item::<f32>().unwrap());
}

#[test]
fn var_freefn_parity_with_method() {
  let a = Array::full::<f32>(&(2, 2), 3.0).unwrap();
  let mut method = a.var(false, 0).unwrap();
  let mut freefn = mlxrs::ops::reduction::var(&a, false, 0).unwrap();
  assert_eq!(method.item::<f32>().unwrap(), freefn.item::<f32>().unwrap());
}
