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
//! Soundness: MLX divides by `groups` (`in.shape % groups`) and, for the
//! forward convolutions, by `stride` (output-shape computation), so a value
//! below 1 from this safe API would be a C++ integer division by zero. The
//! wrappers reject non-positive `groups`/`stride` with a typed error first.
//! [`conv_general`] additionally validates each spatial-parameter slice length,
//! since MLX broadcasts a length-0/1 slice but indexes a longer one in
//! `[0, spatial_rank)` — any other length would read past the C++ vector.

use crate::{
  array::Array,
  error::{Error, OutOfRangePayload, Result, check},
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
  // is a division-by-zero risk here.
  require_positive("conv_transpose1d groups", groups)?;
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
  // SAFETY: fresh out-param handle, RAII-wrapped before the populating call.
  let mut out = Array(unsafe { mlxrs_sys::mlx_array_new() });
  // SAFETY: each `as_ptr()`/`len()` pair describes a slice valid for the whole
  // call (lengths validated above); mlx-c copies the spatial-param arrays rather
  // than retaining them. `out.0` was freshly allocated above and is written
  // here; rc via `check`.
  check(unsafe {
    mlxrs_sys::mlx_conv_general(
      &mut out.0,
      input.0,
      weight.0,
      stride.as_ptr(),
      stride.len(),
      padding_lo.as_ptr(),
      padding_lo.len(),
      padding_hi.as_ptr(),
      padding_hi.len(),
      kernel_dilation.as_ptr(),
      kernel_dilation.len(),
      input_dilation.as_ptr(),
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
}
