//! Convolution ops — thin forwards over mlx-c `mlx_conv*`.
//!
//! MLX uses **channels-last** layout: a 1-D input is `(N, L, C_in)` with weight
//! `(C_out, K, C_in / groups)`; a 2-D input is `(N, H, W, C_in)` with weight
//! `(C_out, KH, KW, C_in / groups)`; 3-D analogously. Convolution is
//! **cross-correlation** (no kernel flip) except via [`conv_general`]'s `flip`.
//! The output spatial size per axis is
//! `(in + 2*padding - dilation*(k - 1) - 1) / stride + 1`.
//!
//! Mirrors `mlx.core.{conv1d, conv2d, conv3d, conv_transpose1d,
//! conv_transpose2d, conv_transpose3d, conv_general}`.
//!
//! Soundness: MLX divides by `groups` (`in.shape % groups`) and by `stride`
//! (forward output-shape), and computes the channel check and the per-axis
//! output shape in **int32 before it can raise an error**. So a value below 1,
//! a mismatched [`conv_general`] slice length, or extreme
//! `dilation`/`padding`/`stride`/`output_padding`/`groups` would otherwise be a
//! C++ division-by-zero, out-of-bounds read, or signed-overflow UB reachable
//! from safe Rust. Each wrapper rejects these with a typed error before the FFI.

use crate::{
  array::Array,
  error::{Error, OutOfRangePayload, Result, check},
  shape::dim_ptr,
  stream::default_stream,
};
use smol_str::format_smolstr;

/// Reject a `stride`/`groups` value below 1 (a C++ division-by-zero guard).
fn require_positive(context: &'static str, value: i32) -> Result<()> {
  if value < 1 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must be >= 1",
      format_smolstr!("{value}"),
    )));
  }
  Ok(())
}

/// Typed error for a convolution whose parameters would overflow MLX's internal
/// int32 shape arithmetic.
fn conv_overflow_err(context: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    context,
    "parameters overflow MLX int32 convolution shape arithmetic",
    "overflow",
  ))
}

/// Verify MLX cannot overflow its int32 convolution shape arithmetic for these
/// inputs. MLX builds that arithmetic — across every forward, broadcast,
/// negative-padding slice, transpose-prelude, and nested-`conv_general` path —
/// as sums of products of one spatial PARAMETER and one input/weight DIMENSION
/// (it never multiplies two dimensions; element counts are 64-bit). So if the
/// largest such product, times a margin covering the handful of summed terms in
/// the deepest (transposed) formula, still fits `i32`, none of those int32
/// intermediates can overflow — regardless of MLX's exact statement order. This
/// conservative bound is exact for every realistic convolution and only rejects
/// physically-impossible ones (a single axis near `i32::MAX` paired with a large
/// parameter, which could not be allocated anyway). Without it such a call would
/// be C++ signed-overflow UB instead of a typed error before the FFI.
#[allow(clippy::too_many_arguments)]
fn check_conv_no_overflow(
  context: &'static str,
  input: &Array,
  weight: &Array,
  stride: &[i32],
  pad_lo: &[i32],
  pad_hi: &[i32],
  kernel_dilation: &[i32],
  input_dilation: &[i32],
  output_padding: Option<&[i32]>,
  groups: i32,
) -> Result<()> {
  // MLX builds its int32 conv shape arithmetic as sums of products of one spatial
  // parameter and one input/weight dimension. The deepest running accumulation —
  // the transpose prelude's nested conv_general numerator
  // (id + padding_lo + padding_hi - kd) — reaches at most ~6 such products plus a
  // few parameter-linear and constant terms, so an affine term-count bound is
  // both complete and tight: a flat margin would needlessly reject large
  // unit-parameter convolutions whose products are only dimension * 1.
  let in_shape = input.shape();
  let wt_shape = weight.shape();
  let max_dim = in_shape
    .iter()
    .chain(wt_shape.iter())
    .map(|&d| d as i128)
    .max()
    .unwrap_or(0);
  let max_param = [
    stride,
    pad_lo,
    pad_hi,
    kernel_dilation,
    input_dilation,
    output_padding.unwrap_or(&[]),
  ]
  .into_iter()
  .flatten()
  .chain(std::iter::once(&groups))
  .map(|&p| i128::from(p).abs())
  .max()
  .unwrap_or(0);
  if 6 * max_param * max_dim + 32 * max_param + 64 > i128::from(i32::MAX) {
    return Err(conv_overflow_err(context));
  }
  Ok(())
}

/// 1-D convolution. `input` is `(N, L, C_in)`, `weight` is
/// `(C_out, K, C_in / groups)`; returns `(N, L_out, C_out)`.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv1d.html).
pub fn conv1d(
  input: &Array,
  weight: &Array,
  stride: i32,
  padding: i32,
  dilation: i32,
  groups: i32,
) -> Result<Array> {
  require_positive("conv1d stride", stride)?;
  require_positive("conv1d groups", groups)?;
  check_conv_no_overflow(
    "conv1d",
    input,
    weight,
    &[stride],
    &[padding],
    &[padding],
    &[dilation],
    &[],
    None,
    groups,
  )?;
  // SAFETY: fresh out-param handle, wrapped in the RAII newtype first so an
  // early return frees it.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: `input`/`weight` are valid borrowed handles live for the call;
  // `out.0` was just allocated and is written here; the rc is surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv1d(
      &mut out.0,
      input.0,
      weight.0,
      stride,
      padding,
      dilation,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D convolution. `input` is `(N, H, W, C_in)`, `weight` is
/// `(C_out, KH, KW, C_in / groups)`; returns `(N, H_out, W_out, C_out)`.
/// `stride` / `padding` / `dilation` are `(row, col)` pairs.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html).
pub fn conv2d(
  input: &Array,
  weight: &Array,
  stride: (i32, i32),
  padding: (i32, i32),
  dilation: (i32, i32),
  groups: i32,
) -> Result<Array> {
  require_positive("conv2d stride", stride.0)?;
  require_positive("conv2d stride", stride.1)?;
  require_positive("conv2d groups", groups)?;
  check_conv_no_overflow(
    "conv2d",
    input,
    weight,
    &[stride.0, stride.1],
    &[padding.0, padding.1],
    &[padding.0, padding.1],
    &[dilation.0, dilation.1],
    &[],
    None,
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: borrowed handles live for the call; `out.0` freshly allocated above
  // and written here; rc surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv2d(
      &mut out.0,
      input.0,
      weight.0,
      stride.0,
      stride.1,
      padding.0,
      padding.1,
      dilation.0,
      dilation.1,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 3-D convolution. `input` is `(N, D, H, W, C_in)`, `weight` is
/// `(C_out, KD, KH, KW, C_in / groups)`. Spatial params are `(d, h, w)` triples.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv3d.html).
pub fn conv3d(
  input: &Array,
  weight: &Array,
  stride: (i32, i32, i32),
  padding: (i32, i32, i32),
  dilation: (i32, i32, i32),
  groups: i32,
) -> Result<Array> {
  require_positive("conv3d stride", stride.0)?;
  require_positive("conv3d stride", stride.1)?;
  require_positive("conv3d stride", stride.2)?;
  require_positive("conv3d groups", groups)?;
  check_conv_no_overflow(
    "conv3d",
    input,
    weight,
    &[stride.0, stride.1, stride.2],
    &[padding.0, padding.1, padding.2],
    &[padding.0, padding.1, padding.2],
    &[dilation.0, dilation.1, dilation.2],
    &[],
    None,
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: borrowed handles live for the call; `out.0` freshly allocated above
  // and written here; rc surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv3d(
      &mut out.0,
      input.0,
      weight.0,
      stride.0,
      stride.1,
      stride.2,
      padding.0,
      padding.1,
      padding.2,
      dilation.0,
      dilation.1,
      dilation.2,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 1-D transposed convolution (a.k.a. deconvolution). `output_padding` resolves
/// the output-size ambiguity of a strided transpose.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_transpose1d.html).
pub fn conv_transpose1d(
  input: &Array,
  weight: &Array,
  stride: i32,
  padding: i32,
  dilation: i32,
  output_padding: i32,
  groups: i32,
) -> Result<Array> {
  // Transposed conv multiplies (does not divide) by stride, so only `groups`
  // is a division-by-zero risk here; the overflow check covers the rest.
  require_positive("conv_transpose1d groups", groups)?;
  check_conv_no_overflow(
    "conv_transpose1d",
    input,
    weight,
    &[stride],
    &[padding],
    &[padding],
    &[dilation],
    &[],
    Some(&[output_padding]),
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: borrowed handles live for the call; `out.0` freshly allocated above
  // and written here; rc surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv_transpose1d(
      &mut out.0,
      input.0,
      weight.0,
      stride,
      padding,
      dilation,
      output_padding,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 2-D transposed convolution. Spatial params are `(row, col)` pairs.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_transpose2d.html).
pub fn conv_transpose2d(
  input: &Array,
  weight: &Array,
  stride: (i32, i32),
  padding: (i32, i32),
  dilation: (i32, i32),
  output_padding: (i32, i32),
  groups: i32,
) -> Result<Array> {
  require_positive("conv_transpose2d groups", groups)?;
  check_conv_no_overflow(
    "conv_transpose2d",
    input,
    weight,
    &[stride.0, stride.1],
    &[padding.0, padding.1],
    &[padding.0, padding.1],
    &[dilation.0, dilation.1],
    &[],
    Some(&[output_padding.0, output_padding.1]),
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: borrowed handles live for the call; `out.0` freshly allocated above
  // and written here; rc surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv_transpose2d(
      &mut out.0,
      input.0,
      weight.0,
      stride.0,
      stride.1,
      padding.0,
      padding.1,
      dilation.0,
      dilation.1,
      output_padding.0,
      output_padding.1,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// 3-D transposed convolution. Spatial params are `(d, h, w)` triples.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_transpose3d.html).
pub fn conv_transpose3d(
  input: &Array,
  weight: &Array,
  stride: (i32, i32, i32),
  padding: (i32, i32, i32),
  dilation: (i32, i32, i32),
  output_padding: (i32, i32, i32),
  groups: i32,
) -> Result<Array> {
  require_positive("conv_transpose3d groups", groups)?;
  check_conv_no_overflow(
    "conv_transpose3d",
    input,
    weight,
    &[stride.0, stride.1, stride.2],
    &[padding.0, padding.1, padding.2],
    &[padding.0, padding.1, padding.2],
    &[dilation.0, dilation.1, dilation.2],
    &[],
    Some(&[output_padding.0, output_padding.1, output_padding.2]),
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: borrowed handles live for the call; `out.0` freshly allocated above
  // and written here; rc surfaced by `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv_transpose3d(
      &mut out.0,
      input.0,
      weight.0,
      stride.0,
      stride.1,
      stride.2,
      padding.0,
      padding.1,
      padding.2,
      dilation.0,
      dilation.1,
      dilation.2,
      output_padding.0,
      output_padding.1,
      output_padding.2,
      groups,
      default_stream(),
    )
  })?;
  Ok(out)
}

/// General N-D convolution — the primitive `conv1d`/`conv2d`/`conv3d` build on.
///
/// Each slice carries one value per spatial axis (`stride`, `kernel_dilation`,
/// `input_dilation`) or per side (`padding_lo`, `padding_hi`); a length of 0 or
/// 1 is broadcast across the spatial axes, otherwise the length must equal the
/// input spatial rank. `flip` selects a true convolution (kernel flipped) over
/// the default cross-correlation.
///
/// See [mlx docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv_general.html).
#[allow(clippy::too_many_arguments)]
pub fn conv_general(
  input: &Array,
  weight: &Array,
  stride: &[i32],
  padding_lo: &[i32],
  padding_hi: &[i32],
  kernel_dilation: &[i32],
  input_dilation: &[i32],
  groups: i32,
  flip: bool,
) -> Result<Array> {
  require_positive("conv_general groups", groups)?;
  // MLX broadcasts a length-0 or length-1 spatial slice, but uses a longer one
  // as-is and then indexes it in `[0, spatial_rank)`, so any other length reads
  // past the C++ vector. Spatial rank is `ndim - 2` (channels-last).
  let spatial_rank = input.shape().len().saturating_sub(2);
  for (context, slice) in [
    ("conv_general stride", stride),
    ("conv_general padding_lo", padding_lo),
    ("conv_general padding_hi", padding_hi),
    ("conv_general kernel_dilation", kernel_dilation),
    ("conv_general input_dilation", input_dilation),
  ] {
    if !(slice.is_empty() || slice.len() == 1 || slice.len() == spatial_rank) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        context,
        "length must be 0, 1, or the input spatial rank",
        format_smolstr!("{}", slice.len()),
      )));
    }
  }
  // stride is divided in the output-shape computation; reject non-positive.
  for &s in stride {
    require_positive("conv_general stride", s)?;
  }
  check_conv_no_overflow(
    "conv_general",
    input,
    weight,
    stride,
    padding_lo,
    padding_hi,
    kernel_dilation,
    input_dilation,
    None,
    groups,
  )?;
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: each `dim_ptr`/`len()` pair describes a slice valid for the whole
  // call (lengths validated above); `dim_ptr` routes an empty slice through a
  // non-singular sentinel so mlx-c's `std::vector<int>(p, p + 0)` never receives
  // a dangling iterator. mlx-c copies the spatial-param arrays rather than
  // retaining them. `out.0` was freshly allocated above and is written here; rc
  // via `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv_general(
      &mut out.0,
      input.0,
      weight.0,
      dim_ptr(stride),
      stride.len(),
      dim_ptr(padding_lo),
      padding_lo.len(),
      dim_ptr(padding_hi),
      padding_hi.len(),
      dim_ptr(kernel_dilation),
      kernel_dilation.len(),
      dim_ptr(input_dilation),
      input_dilation.len(),
      groups,
      flip,
      default_stream(),
    )
  })?;
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Array, Result, error::Error};

  #[test]
  fn conv1d_cross_correlation_closed_form() -> Result<()> {
    // input (N=1, L=4, C_in=1) = [1,2,3,4]; weight (C_out=1, K=3, C_in=1) = [1,0,-1].
    // Cross-correlation (no flip), stride 1, no pad ⇒ L_out = 4-3+1 = 2:
    //   out[0] = 1*1 + 2*0 + 3*(-1) = -2
    //   out[1] = 2*1 + 3*0 + 4*(-1) = -2
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1])?;
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1])?;
    let mut out = conv1d(&input, &weight, 1, 0, 1, 1)?;
    assert_eq!(out.shape(), vec![1, 2, 1]);
    assert_eq!(out.to_vec::<f32>()?, vec![-2.0, -2.0]);
    Ok(())
  }

  #[test]
  fn conv2d_cross_correlation_closed_form() -> Result<()> {
    // input (N=1, H=3, W=3, C_in=1) = 1..=9 row-major;
    // weight (C_out=1, KH=2, KW=2, C_in=1) = [[1,0],[0,-1]].
    // out[h,w] = in[h,w]*1 + in[h+1,w+1]*(-1); H_out = W_out = 2:
    //   [1-5, 2-6, 4-8, 5-9] = [-4,-4,-4,-4]
    let input = Array::from_slice::<f32>(
      &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      &[1, 3, 3, 1],
    )?;
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, -1.0], &[1, 2, 2, 1])?;
    let mut out = conv2d(&input, &weight, (1, 1), (0, 0), (1, 1), 1)?;
    assert_eq!(out.shape(), vec![1, 2, 2, 1]);
    assert_eq!(out.to_vec::<f32>()?, vec![-4.0, -4.0, -4.0, -4.0]);
    Ok(())
  }

  #[test]
  fn conv3d_all_ones_sums_kernel_volume() -> Result<()> {
    // 2x2x2 all-ones input, 2x2x2 all-ones kernel, single 1x1x1 output position:
    // the sum over the 8-element receptive field = 8.
    let input = Array::from_slice::<f32>(&[1.0; 8], &[1, 2, 2, 2, 1])?;
    let weight = Array::from_slice::<f32>(&[1.0; 8], &[1, 2, 2, 2, 1])?;
    let mut out = conv3d(&input, &weight, (1, 1, 1), (0, 0, 0), (1, 1, 1), 1)?;
    assert_eq!(out.shape(), vec![1, 1, 1, 1, 1]);
    assert_eq!(out.to_vec::<f32>()?, vec![8.0]);
    Ok(())
  }

  #[test]
  fn conv_general_matches_conv1d() -> Result<()> {
    // conv_general with one value per axis and flip=false reproduces conv1d.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1])?;
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1])?;
    let mut general = conv_general(&input, &weight, &[1], &[0], &[0], &[1], &[1], 1, false)?;
    assert_eq!(general.shape(), vec![1, 2, 1]);
    assert_eq!(general.to_vec::<f32>()?, vec![-2.0, -2.0]);
    Ok(())
  }

  #[test]
  fn conv_transpose1d_expands_length() -> Result<()> {
    // Transposed conv grows the length: L_out = (L-1)*stride - 2*pad
    // + dilation*(K-1) + output_padding + 1 = 1 + 1 + 0 + 1 = 3.
    let input = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2, 1])?;
    let weight = Array::from_slice::<f32>(&[1.0, 1.0], &[1, 2, 1])?;
    let out = conv_transpose1d(&input, &weight, 1, 0, 1, 0, 1)?;
    assert_eq!(out.shape(), vec![1, 3, 1]);
    Ok(())
  }

  #[test]
  fn conv1d_rejects_non_positive_groups() {
    // groups = 0 would reach C++ `in.shape % groups` (division by zero); negative
    // groups is equally invalid. Both must be a typed Err, never UB.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let zero = conv1d(&input, &weight, 1, 0, 1, 0).expect_err("groups=0 must be rejected");
    assert!(matches!(zero, Error::OutOfRange(_)), "got {zero:?}");
    let neg = conv1d(&input, &weight, 1, 0, 1, -1).expect_err("negative groups must be rejected");
    assert!(matches!(neg, Error::OutOfRange(_)), "got {neg:?}");
  }

  #[test]
  fn conv1d_rejects_zero_stride() {
    // stride = 0 would divide by zero in the output-shape computation.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err = conv1d(&input, &weight, 0, 0, 1, 1).expect_err("stride=0 must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_mismatched_slice_length() {
    // 1-D input (spatial rank 1); a length-2 stride slice would index past the
    // C++ vector. Must be rejected before the FFI.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err = conv_general(&input, &weight, &[1, 1], &[0], &[0], &[1], &[1], 1, false)
      .expect_err("len-2 spatial slice on a 1-D conv must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_non_positive_groups() {
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err = conv_general(&input, &weight, &[1], &[0], &[0], &[1], &[1], 0, false)
      .expect_err("groups=0 must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv1d_rejects_overflowing_dilation() {
    // dilation * (K - 1) = i32::MAX * 2 overflows MLX's int32 shape arithmetic.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err = conv1d(&input, &weight, 1, 0, i32::MAX, 1)
      .expect_err("overflowing dilation must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv1d_rejects_overflowing_padding() {
    // in + pad_lo + pad_hi overflows when padding is i32::MAX.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err =
      conv1d(&input, &weight, 1, i32::MAX, 1, 1).expect_err("overflowing padding must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv1d_rejects_overflowing_groups() {
    // weight C_in_per_group = 2; groups = i32::MAX ⇒ groups * 2 overflows the
    // channel-consistency check before MLX can raise the channel error.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 1.0], &[1, 1, 2]).unwrap();
    let err =
      conv1d(&input, &weight, 1, 0, 1, i32::MAX).expect_err("overflowing groups must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_transpose1d_rejects_overflowing_output_padding() {
    // (in-1)*stride + dk + output_padding + 1 overflows when output_padding is
    // i32::MAX.
    let input = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 1.0], &[1, 2, 1]).unwrap();
    let err = conv_transpose1d(&input, &weight, 1, 0, 1, i32::MAX, 1)
      .expect_err("overflowing output_padding must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_overflowing_dilation() {
    // kernel_dilation * (K - 1) overflows.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let err = conv_general(
      &input,
      &weight,
      &[1],
      &[0],
      &[0],
      &[i32::MAX],
      &[1],
      1,
      false,
    )
    .expect_err("overflowing kernel_dilation must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_canceling_padding() {
    // padding_lo = i32::MIN and padding_hi = i32::MAX cancel in the final output
    // size, but MLX normalizes negative padding via `0 - padding_lo` first, which
    // overflows. Must be rejected (Codex round-3 counterexample).
    let input = Array::from_slice::<f32>(&[1.0, 2.0], &[1, 2, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let err = conv_general(
      &input,
      &weight,
      &[1],
      &[i32::MIN],
      &[i32::MAX],
      &[1],
      &[1],
      1,
      false,
    )
    .expect_err("canceling i32::MIN / i32::MAX padding must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_transpose1d_rejects_overflowing_padding() {
    // MLX's transpose prelude computes `2 * padding`, which overflows at
    // padding = 1.5e9 even though the final output size stays in range (Codex
    // round-3 counterexample).
    let input = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let err = conv_transpose1d(&input, &weight, 1, 1_500_000_000, 1, 0, 1)
      .expect_err("transpose 2*padding overflow must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_all_empty_slices_uses_defaults() {
    // Empty spatial slices select MLX defaults (stride 1, no padding, dilation 1)
    // and must cross the FFI through dim_ptr, not a dangling empty-slice pointer
    // (Codex round-4 finding). The result matches conv1d with the same defaults.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 0.0, -1.0], &[1, 3, 1]).unwrap();
    let mut out = conv_general(&input, &weight, &[], &[], &[], &[], &[], 1, false)
      .expect("all-empty conv_general must use defaults");
    assert_eq!(out.shape(), vec![1, 2, 1]);
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![-2.0, -2.0]);
  }

  #[test]
  fn conv_general_allows_negative_padding_crop() {
    // A normal negative-padding crop (padding_hi = -1 drops one element) is valid
    // and must pass the bound — the conservative guard only rejects i32::MAX-class
    // parameters/dimensions, not ordinary cropping.
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let out = conv_general(&input, &weight, &[1], &[0], &[-1], &[1], &[1], 1, false)
      .expect("a normal negative-padding crop must not be rejected");
    assert_eq!(out.shape(), vec![1, 2, 1]);
  }

  #[test]
  fn conv_transpose1d_rejects_rank_invalid_prelude_overflow() {
    // Rank-2 input/weight is invalid for conv_transpose1d, but MLX runs its
    // prelude (1 + dilation*(weight.shape(1)-1)) over the wrapper arity BEFORE
    // rejecting the rank, so dilation=i32::MAX overflows first. The guard must
    // check the prelude regardless of rank (Codex round-5 counterexample).
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &[1, 3]).unwrap();
    let err = conv_transpose1d(&input, &weight, 1, 0, i32::MAX, 0, 1)
      .expect_err("rank-invalid transpose prelude overflow must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_slice_overflow_on_huge_dim() {
    // A lazy array can carry an i32::MAX-length axis without allocating. Negative
    // padding makes MLX slice EVERY dimension, computing `dim + 1`, which
    // overflows for the i32::MAX batch axis. The guard must reject before the FFI
    // (Codex round-5 counterexample).
    let input = Array::zeros::<f32>(&[i32::MAX, 2, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let err = conv_general(&input, &weight, &[1], &[0], &[-1], &[1], &[1], 1, false)
      .expect_err("i32::MAX dim + negative-padding slice overflow must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_rejects_huge_dim_conservatively() {
    // The conservative bound rejects any axis near i32::MAX paired with a
    // parameter — such a convolution is physically impossible to allocate (an
    // ~8 GB single axis). This is the accepted over-rejection of the bound.
    let input = Array::zeros::<f32>(&[1, i32::MAX, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let err = conv_general(&input, &weight, &[1], &[0], &[-1], &[1], &[1], 1, false)
      .expect_err("i32::MAX axis must be conservatively rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_transpose1d_rejects_4d_broadcast_overflow() {
    // A 4D conv_transpose1d broadcasts dilation to the extra spatial axis MLX
    // derives from the input. The bound rejects it because the parameter
    // magnitude alone is i32::MAX, regardless of which axis it lands on (Codex
    // round-7 counterexample, which the statement-mirror missed).
    let input = Array::zeros::<f32>(&[1, 1, 1, 1]).unwrap();
    let weight = Array::zeros::<f32>(&[1, 1, 3, 1]).unwrap();
    let err = conv_transpose1d(&input, &weight, 1, 0, i32::MAX, 0, 1)
      .expect_err("4D transpose broadcast overflow must be rejected");
    assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
  }

  #[test]
  fn conv_general_accepts_large_unit_parameter_dim() {
    // A 1-D unit-kernel conv on a ~134M-element axis (512 MiB f32, allocatable)
    // has no int32 shape overflow, so the affine bound must accept it — large
    // dimensions with default parameters are not rejected (Codex round-8
    // acceptance regression).
    let input = Array::zeros::<f32>(&[1, 134_217_728, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    assert!(
      conv_general(&input, &weight, &[1], &[0], &[0], &[1], &[1], 1, false).is_ok(),
      "a large unit-parameter convolution must not be falsely rejected"
    );
  }

  #[test]
  fn conv1d_handles_zero_length_axis_without_panic() {
    // A zero-length spatial axis must not panic the guard (the max-dimension scan
    // and i128 arithmetic stay valid); MLX surfaces the empty-conv shape error
    // through `check` (Codex round-8 zero-dimension regression).
    let input = Array::zeros::<f32>(&[1, 0, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let _ = conv1d(&input, &weight, 1, 0, 1, 1);
  }
}
