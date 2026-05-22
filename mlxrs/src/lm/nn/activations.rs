//! Element-wise activation primitives.
//!
//! 1:1 ports of the activation functions the mlx-lm Mixture-of-Experts
//! **Switch** blocks ([`SwitchGLU`](super::switch::SwitchGLU) /
//! [`SwitchMLP`](super::switch::SwitchMLP)) compose:
//!
//! - [`silu`] — the Sigmoid-Linear-Unit (a.k.a. Swish), `x · σ(x)`. The
//!   activation half of [`swiglu`], and `mlx-swift`'s default `SwitchGLU`
//!   activation (`MLXLMCommon/SwitchLayers.swift`).
//! - [`swiglu`] — the gated `silu(gate) · x`. mlx-lm's `SwiGLU` module wraps
//!   exactly this two-argument function (`mlx-lm/mlx_lm/models/activations.py`),
//!   and it is the default `SwitchGLU` activation in the python reference.
//! - [`gelu`] — the exact Gaussian-Error-Linear-Unit, `x · Φ(x)` with `Φ` the
//!   Gaussian CDF. The `approx="none"` arm of `mlx.nn.GELU`.
//! - [`gelu_approx`] — the `tanh` approximation of [`gelu`]
//!   (`mlx.nn.gelu_approx`); the `approx="precise"` / `"tanh"` arm of
//!   `mlx.nn.GELU`, and mlx-lm's default `SwitchMLP` activation
//!   (`SwitchMLP(..., activation=nn.GELU(approx="precise"))`).
//! - [`gelu_fast_approx`] — the cheaper sigmoid approximation of [`gelu`]
//!   (`mlx.nn.gelu_fast_approx`); the `approx="fast"` arm of `mlx.nn.GELU`.
//!
//! Each is a pure free function `&Array -> Result<Array>` (the gated
//! [`swiglu`] takes two operands): it builds a new lazy [`Array`] and never
//! evaluates — eval stays an explicit `&mut` step on the result, exactly like
//! every other `mlxrs` op. The reference (`python/mlx/nn/layers/activations.py`)
//! decorates the closed-form variants with `@mx.compile`; that is a graph-fusion
//! hint with no effect on the math, so it has no analogue here.
//!
//! Mirroring the references' forward expressions verbatim, the scalar
//! literals (`0.5`, `1 / √2`, `√(2/π)`, `0.044715`, `1.702`) are folded into
//! rank-1 single-element [`Array`]s that broadcast against `x`; no constant is
//! pre-multiplied away from the form the reference writes.

use crate::{array::Array, error::Result};

/// Build a broadcastable single-element `f32` constant array.
///
/// The activation expressions below mix `Array` operands with scalar
/// literals; mlx-c's element-wise ops take two `Array`s, so each literal is
/// lifted to a shape-`[1]` array that NumPy-broadcasts against `x` regardless
/// of `x`'s rank. Shape `[1]` (rather than rank-0) keeps the constructor on
/// the common [`Array::from_slice`] path.
#[inline]
fn scalar(value: f32) -> Result<Array> {
  Array::from_slice::<f32>(&[value], &(1usize,))
}

/// Sigmoid Linear Unit, a.k.a. Swish: `silu(x) = x · σ(x)`.
///
/// 1:1 port of `mlx.nn.silu` (`python/mlx/nn/layers/activations.py`):
/// `x * mx.sigmoid(x)`, where `σ` is the logistic sigmoid. This is the
/// activation half of [`swiglu`] and the default activation of mlx-swift's
/// `SwitchGLU`.
///
/// Returns a new lazy [`Array`] the same shape/dtype as `x` (no implicit
/// eval).
pub fn silu(x: &Array) -> Result<Array> {
  // `x * mx.sigmoid(x)` — verbatim from the reference.
  x.multiply(&x.sigmoid()?)
}

/// Gated SiLU: `swiglu(gate, x) = silu(gate) · x`.
///
/// 1:1 port of mlx-lm's `swiglu` (`mlx-lm/mlx_lm/models/activations.py`):
/// `nn.silu(gate) * x`. mlx-lm's `SwiGLU` module is a thin wrapper over this
/// two-argument function, and it is the default activation of the python
/// `SwitchGLU` block — there the `gate` and `x` operands are the `gate_proj`
/// and `up_proj` projections of the routed input.
///
/// The argument order matches the reference exactly: `gate` is squashed by
/// [`silu`], then multiplied by the (un-activated) `x`. Returns a new lazy
/// [`Array`] (no implicit eval).
pub fn swiglu(gate: &Array, x: &Array) -> Result<Array> {
  // `nn.silu(gate) * x` — verbatim from the reference.
  silu(gate)?.multiply(x)
}

/// Exact Gaussian Error Linear Unit: `gelu(x) = x · Φ(x)`, where `Φ` is the
/// standard-normal CDF.
///
/// 1:1 port of `mlx.nn.gelu` (`python/mlx/nn/layers/activations.py`):
/// `x * (1 + erf(x / √2)) / 2`. This is the `approx="none"` arm of the
/// `mlx.nn.GELU` module. See [`gelu_approx`] / [`gelu_fast_approx`] for the
/// cheaper approximations.
///
/// Returns a new lazy [`Array`] the same shape/dtype as `x` (no implicit
/// eval).
pub fn gelu(x: &Array) -> Result<Array> {
  // `x * (1 + mx.erf(x / math.sqrt(2))) / 2` — verbatim from the reference,
  // with the scalar literals (`√2`, `1`, `2`) lifted to broadcast arrays.
  let inv_sqrt2 = scalar(std::f32::consts::FRAC_1_SQRT_2)?;
  let one = scalar(1.0)?;
  let two = scalar(2.0)?;
  // erf(x / √2) — `x / math.sqrt(2)` is written here as `x * (1/√2)` so the
  // single-element constant is the (cheaper) multiplier; the value is exact.
  let erf_term = x.multiply(&inv_sqrt2)?.erf()?;
  // x * (1 + erf(...)) / 2
  x.multiply(&one.add(&erf_term)?)?.divide(&two)
}

/// `tanh` approximation of [`gelu`].
///
/// 1:1 port of `mlx.nn.gelu_approx` (`python/mlx/nn/layers/activations.py`):
/// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`. The reference
/// docstring bounds the absolute error below `0.0005` on `[-6, 6]`. This is
/// the `approx="precise"` / `"tanh"` arm of the `mlx.nn.GELU` module, and the
/// default activation of mlx-lm's `SwitchMLP` block.
///
/// Returns a new lazy [`Array`] the same shape/dtype as `x` (no implicit
/// eval).
pub fn gelu_approx(x: &Array) -> Result<Array> {
  // `0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))`
  // — verbatim from the reference, scalar literals lifted to broadcast arrays.
  let half = scalar(0.5)?;
  let one = scalar(1.0)?;
  let sqrt_2_over_pi = scalar((2.0 / std::f32::consts::PI).sqrt())?;
  let c = scalar(0.044715)?;
  // x + 0.044715 * x³  (x³ via `x.square() * x` — mlx's `x**3`).
  let x_cubed = x.square()?.multiply(x)?;
  let inner = x.add(&c.multiply(&x_cubed)?)?;
  // tanh(√(2/π) · inner)
  let tanh_term = sqrt_2_over_pi.multiply(&inner)?.tanh()?;
  // 0.5 * x * (1 + tanh(...))
  half.multiply(x)?.multiply(&one.add(&tanh_term)?)
}

/// Fast sigmoid approximation of [`gelu`].
///
/// 1:1 port of `mlx.nn.gelu_fast_approx` (`python/mlx/nn/layers/activations.py`):
/// `x · σ(1.702 · x)`, where `σ` is the logistic sigmoid. The reference
/// docstring bounds the absolute error below `0.015` on `[-6, 6]`. This is the
/// `approx="fast"` arm of the `mlx.nn.GELU` module.
///
/// Returns a new lazy [`Array`] the same shape/dtype as `x` (no implicit
/// eval).
pub fn gelu_fast_approx(x: &Array) -> Result<Array> {
  // `x * mx.sigmoid(1.702 * x)` — verbatim from the reference.
  let c = scalar(1.702)?;
  x.multiply(&c.multiply(x)?.sigmoid()?)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Logistic sigmoid `σ(v) = 1 / (1 + e^-v)` — the reference scalar formula
  /// the array ops are checked against.
  fn sigmoid_ref(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
  }

  /// Per-element near-equality with a tolerance generous enough for the f32
  /// op-graph vs the f64 reference arithmetic.
  fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (g, w) in got.iter().zip(want.iter()) {
      assert!(
        (g - w).abs() <= 1e-5 + 1e-5 * w.abs(),
        "activation mismatch: got {g}, want {w}"
      );
    }
  }

  /// A small spread of inputs covering negative, zero, and positive values.
  fn sample_input() -> Array {
    Array::from_slice::<f32>(&[-2.0, -0.5, 0.0, 0.5, 2.0], &(5usize,)).unwrap()
  }

  #[test]
  fn silu_matches_reference_formula() {
    let mut out = silu(&sample_input()).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // silu(x) = x · σ(x).
    let want: Vec<f32> = [-2.0f32, -0.5, 0.0, 0.5, 2.0]
      .iter()
      .map(|&x| x * sigmoid_ref(x))
      .collect();
    assert_close(&got, &want);
  }

  #[test]
  fn silu_zero_is_zero() {
    // σ(0) = 0.5, so silu(0) = 0 · 0.5 = 0 exactly.
    let zero = Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap();
    let mut out = silu(&zero).unwrap();
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.0]);
  }

  #[test]
  fn swiglu_matches_silu_gate_times_x() {
    // gate and x are distinct so a swapped-operand bug would be visible.
    let gate = Array::from_slice::<f32>(&[-1.0, 0.0, 1.0, 3.0], &(4usize,)).unwrap();
    let x = Array::from_slice::<f32>(&[2.0, 5.0, -4.0, 0.5], &(4usize,)).unwrap();
    let mut out = swiglu(&gate, &x).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // swiglu(gate, x) = silu(gate) · x = gate · σ(gate) · x.
    let g = [-1.0f32, 0.0, 1.0, 3.0];
    let xv = [2.0f32, 5.0, -4.0, 0.5];
    let want: Vec<f32> = g
      .iter()
      .zip(xv.iter())
      .map(|(&gi, &xi)| gi * sigmoid_ref(gi) * xi)
      .collect();
    assert_close(&got, &want);
  }

  #[test]
  fn swiglu_hand_traced_scalar() {
    // Hand-traced single value: gate = 1, x = 2.
    //   σ(1)        = 1/(1+e^-1) ≈ 0.7310585786
    //   silu(1)     = 1 · 0.7310585786 = 0.7310585786
    //   swiglu(1,2) = 0.7310585786 · 2 = 1.4621171572
    let gate = Array::from_slice::<f32>(&[1.0], &(1usize,)).unwrap();
    let x = Array::from_slice::<f32>(&[2.0], &(1usize,)).unwrap();
    let mut out = swiglu(&gate, &x).unwrap();
    assert_close(&out.to_vec::<f32>().unwrap(), &[1.462_117_2]);
  }

  #[test]
  fn gelu_matches_reference_formula() {
    let mut out = gelu(&sample_input()).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // gelu(x) = x · (1 + erf(x / √2)) / 2.  Reference uses f64 libm erf.
    let want: Vec<f32> = [-2.0f64, -0.5, 0.0, 0.5, 2.0]
      .iter()
      .map(|&x| (x * (1.0 + libm_erf(x / std::f64::consts::SQRT_2)) / 2.0) as f32)
      .collect();
    assert_close(&got, &want);
  }

  #[test]
  fn gelu_zero_is_zero() {
    // erf(0) = 0 ⇒ gelu(0) = 0 · (1 + 0) / 2 = 0.
    let zero = Array::from_slice::<f32>(&[0.0], &(1usize,)).unwrap();
    let mut out = gelu(&zero).unwrap();
    assert_close(&out.to_vec::<f32>().unwrap(), &[0.0]);
  }

  #[test]
  fn gelu_approx_matches_reference_formula() {
    let mut out = gelu_approx(&sample_input()).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // gelu_approx(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
    let want: Vec<f32> = [-2.0f64, -0.5, 0.0, 0.5, 2.0]
      .iter()
      .map(|&x| {
        let inner = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        (0.5 * x * (1.0 + inner.tanh())) as f32
      })
      .collect();
    assert_close(&got, &want);
  }

  #[test]
  fn gelu_approx_tracks_exact_gelu_within_error_bound() {
    // The reference docstring bounds |gelu_approx - gelu| < 5e-4 on [-6, 6].
    let x = Array::from_slice::<f32>(&[-6.0, -3.0, -1.0, 1.0, 3.0, 6.0], &(6usize,)).unwrap();
    let mut exact = gelu(&x).unwrap();
    let mut approx = gelu_approx(&x).unwrap();
    let e = exact.to_vec::<f32>().unwrap();
    let a = approx.to_vec::<f32>().unwrap();
    for (ev, av) in e.iter().zip(a.iter()) {
      assert!(
        (ev - av).abs() < 5e-4,
        "gelu_approx strayed beyond the documented 5e-4 bound: exact={ev}, approx={av}"
      );
    }
  }

  #[test]
  fn gelu_fast_approx_matches_reference_formula() {
    let mut out = gelu_fast_approx(&sample_input()).unwrap();
    let got = out.to_vec::<f32>().unwrap();
    // gelu_fast_approx(x) = x · σ(1.702 · x).
    let want: Vec<f32> = [-2.0f32, -0.5, 0.0, 0.5, 2.0]
      .iter()
      .map(|&x| x * sigmoid_ref(1.702 * x))
      .collect();
    assert_close(&got, &want);
  }

  #[test]
  fn gelu_fast_approx_tracks_exact_gelu_within_error_bound() {
    // The reference docstring bounds |gelu_fast_approx - gelu| < 1.5e-2 on
    // [-6, 6].
    let x = Array::from_slice::<f32>(&[-6.0, -3.0, -1.0, 1.0, 3.0, 6.0], &(6usize,)).unwrap();
    let mut exact = gelu(&x).unwrap();
    let mut fast = gelu_fast_approx(&x).unwrap();
    let e = exact.to_vec::<f32>().unwrap();
    let f = fast.to_vec::<f32>().unwrap();
    for (ev, fv) in e.iter().zip(f.iter()) {
      assert!(
        (ev - fv).abs() < 1.5e-2,
        "gelu_fast_approx strayed beyond the documented 1.5e-2 bound: exact={ev}, fast={fv}"
      );
    }
  }

  #[test]
  fn activations_preserve_shape() {
    // A non-trivial rank-3 shape: every activation returns the input shape.
    let x = Array::from_slice::<f32>(
      &(0..24).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
      &(2, 3, 4),
    )
    .unwrap();
    assert_eq!(silu(&x).unwrap().shape(), vec![2, 3, 4]);
    assert_eq!(gelu(&x).unwrap().shape(), vec![2, 3, 4]);
    assert_eq!(gelu_approx(&x).unwrap().shape(), vec![2, 3, 4]);
    assert_eq!(gelu_fast_approx(&x).unwrap().shape(), vec![2, 3, 4]);
    assert_eq!(swiglu(&x, &x).unwrap().shape(), vec![2, 3, 4]);
  }

  /// `erf` for the f64 reference path. The standard library has no `erf`, so
  /// this is the Abramowitz & Stegun 7.1.26 rational approximation (max
  /// absolute error ≈ 1.5e-7) — accurate enough to validate the f32 op graph,
  /// which itself only needs the `1e-5` test tolerance.
  fn libm_erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
      - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t
        * (-x * x).exp();
    sign * y
  }
}
