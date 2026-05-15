//! Arithmetic ops: element-wise binary and unary primitives.
//!
//! Phase 4 Branch A subset:
//! - Binary: `add` (template), `subtract`, `multiply`, `divide`, `maximum`,
//!   `minimum`, `power`.
//! - Unary: `negative`, `abs`, `sqrt`, `square`, `exp`, `log`, `sin`, `cos`,
//!   `tan`, `tanh`.

use crate::{
  array::Array,
  error::{Result, check},
  stream::default_stream,
};

/// Element-wise addition: `out[i] = a[i] + b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.add.html).
///
/// CANONICAL TEMPLATE — every output-pattern fn follows this shape:
///   1. Wrap mlx_array_new() in Array(...) FIRST so RAII covers failure.
///   2. Call the C fn with &mut out.0 + default_stream() trailing arg.
///   3. check(rc)? to surface backend errors.
///   4. Ok(out).
pub fn add(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_add(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.subtract.html).
pub fn subtract(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_subtract(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.multiply.html).
pub fn multiply(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_multiply(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise division: `out[i] = a[i] / b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.divide.html).
pub fn divide(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_divide(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise maximum: `out[i] = max(a[i], b[i])` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.maximum.html).
pub fn maximum(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_maximum(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise minimum: `out[i] = min(a[i], b[i])` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.minimum.html).
pub fn minimum(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_minimum(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise power: `out[i] = a[i] ** b[i]` (with broadcasting).
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.power.html).
pub fn power(a: &Array, b: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_power(&mut out.0, a.0, b.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise unary negation: `out[i] = -a[i]`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.negative.html).
pub fn negative(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_negative(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise absolute value: `out[i] = |a[i]|`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.abs.html).
pub fn abs(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_abs(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise square root: `out[i] = sqrt(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sqrt.html).
pub fn sqrt(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_sqrt(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise square: `out[i] = a[i] * a[i]`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.square.html).
pub fn square(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_square(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise natural exponential: `out[i] = exp(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.exp.html).
pub fn exp(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_exp(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise natural logarithm: `out[i] = ln(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.log.html).
pub fn log(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_log(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise sine: `out[i] = sin(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sin.html).
pub fn sin(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_sin(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise cosine: `out[i] = cos(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cos.html).
pub fn cos(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_cos(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise tangent: `out[i] = tan(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.tan.html).
pub fn tan(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_tan(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}

/// Element-wise hyperbolic tangent: `out[i] = tanh(a[i])`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.tanh.html).
pub fn tanh(a: &Array) -> Result<Array> {
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  check(unsafe { mlxrs_sys::mlx_tanh(&mut out.0, a.0, default_stream()) })?;
  Ok(out)
}
