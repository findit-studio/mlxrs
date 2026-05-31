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

/// Verify MLX's int32 convolution shape arithmetic cannot overflow for these
/// inputs. MLX evaluates the channel check (`groups * C_in_per_group`), the
/// negative-padding normalization and slice, the per-axis output shape, and —
/// for the transposed convolutions — a `conv_transpose_general` prelude (which
/// feeds a nested forward `conv_general`), all in int32, **statement by
/// statement**, before it can return an error. So a safe call with extreme
/// `dilation`/`padding`/`stride`/`output_padding`/`groups` would be C++
/// signed-overflow UB. This recomputes each of MLX's int32 intermediates in
/// `i128` (which cannot overflow for `i32` inputs) and rejects any that does not
/// fit back into `i32`. Spatial parameter slices follow the MLX broadcast rule:
/// length 0 → the default, length 1 → all axes, else per-axis.
/// `output_padding = Some` selects the transposed-conv path; `None` is the
/// forward path (which divides by `stride` — callers ensure `stride >= 1` via
/// [`require_positive`]).
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
  fn param(slice: &[i32], axis: usize, default: i32) -> i32 {
    match slice.len() {
      0 => default,
      1 => slice[0],
      _ => slice[axis],
    }
  }
  // Every MLX int32 intermediate is recomputed in i128 (no overflow for i32
  // inputs) and must fit back into i32.
  fn fits(context: &'static str, value: i128) -> Result<()> {
    if (i32::MIN as i128..=i32::MAX as i128).contains(&value) {
      Ok(())
    } else {
      Err(conv_overflow_err(context))
    }
  }
  // One forward `conv_general` axis: the negative-padding normalization +
  // `slice` (numpy-style index wrap, then clamp to `[0, in]`), then
  // `conv_out_shape` on the sliced length. Reused by the transposed path for
  // the nested forward `conv_general` it dispatches.
  #[allow(clippy::too_many_arguments)]
  fn check_forward_axis(
    context: &'static str,
    in_d: i128,
    wt_d: i128,
    plo: i128,
    phi: i128,
    dil: i128,
    idil: i128,
    strd: i128,
  ) -> Result<()> {
    // Dilated kernel extent: dil * (wt - 1) + 1.
    let dwt = dil * (wt_d - 1);
    fits(context, dwt)?;
    let kd = dwt + 1;
    fits(context, kd)?;
    // Negative-padding normalization: start -= pad_lo, stop += pad_hi.
    let start = if plo < 0 { -plo } else { 0 };
    if plo < 0 {
      fits(context, start)?;
    }
    let stop = if phi < 0 { in_d + phi } else { in_d };
    if phi < 0 {
      fits(context, stop)?;
    }
    // slice normalize: wrap a negative index by + in, then clamp to [0, in].
    let s = if start < 0 {
      let wrapped = start + in_d;
      fits(context, wrapped)?;
      wrapped
    } else {
      start
    };
    let e = if stop < 0 {
      let wrapped = stop + in_d;
      fits(context, wrapped)?;
      wrapped
    } else {
      stop
    };
    let st = s.clamp(0, in_d);
    let ed = e.clamp(0, in_d).max(st);
    let sliced_in = ed - st;
    // conv_out_shape: id = idil * (sliced - 1) + 1;
    // out = (id + pad_lo + pad_hi - kd) / stride + 1, negatives normalized to 0.
    let din = idil * (sliced_in - 1);
    fits(context, din)?;
    let id = din + 1;
    fits(context, id)?;
    let pad_sum = plo.max(0) + phi.max(0);
    fits(context, pad_sum)?;
    let t = id + pad_sum;
    fits(context, t)?;
    let numer = t - kd;
    fits(context, numer)?;
    fits(context, numer / strd + 1)?;
    Ok(())
  }
  // conv_transpose_general prelude for one axis. Returns the (padding_lo,
  // padding_hi) it derives and feeds into the nested forward conv_general.
  fn check_transpose_prelude_axis(
    context: &'static str,
    in_d: i128,
    wt_d: i128,
    pad: i128,
    dil: i128,
    strd: i128,
    op: i128,
  ) -> Result<(i128, i128)> {
    let dwt = dil * (wt_d - 1);
    fits(context, dwt)?;
    let wt_size = dwt + 1;
    fits(context, wt_size)?;
    let t = wt_size - pad;
    fits(context, t)?;
    let padding_lo = t - 1;
    fits(context, padding_lo)?;
    let t = (in_d - 1) * strd;
    fits(context, t)?;
    let two_pad = 2 * pad;
    fits(context, two_pad)?;
    let t = t - two_pad;
    fits(context, t)?;
    let t = t + dwt; // + dil * (wt - 1)
    fits(context, t)?;
    let conv_output_shape = t + 1;
    fits(context, conv_output_shape)?;
    let t = strd * (in_d - 1);
    fits(context, t)?;
    let out_size = t + 1;
    fits(context, out_size)?;
    let t = conv_output_shape - out_size; // in_size == conv_output_shape
    fits(context, t)?;
    let t = t + pad;
    fits(context, t)?;
    let padding_hi = t + op;
    fits(context, padding_hi)?;
    Ok((padding_lo, padding_hi))
  }
  // When negative padding triggers MLX's `slice`, `normalize_slice` runs over
  // EVERY dimension (batch/channel too, not just spatial) computing
  // `(ed - st + stride - 1)`; for the conv slice (stride 1) the intermediate is
  // `dim + 1`, which overflows iff a dimension is i32::MAX. Reject that.
  fn check_slice_all_dims(context: &'static str, shape: &[usize]) -> Result<()> {
    for &d in shape {
      let d = i128::from(i32::try_from(d).map_err(|_| conv_overflow_err(context))?);
      fits(context, d + 1)?;
    }
    Ok(())
  }
  let in_shape = input.shape();
  let wt_shape = weight.shape();
  // run_conv_checks: groups * C_in_per_group.
  if let Some(&channels) = wt_shape.last() {
    let channels = i32::try_from(channels).map_err(|_| conv_overflow_err(context))?;
    fits(context, groups as i128 * channels as i128)?;
  }
  let nd = in_shape.len();
  match output_padding {
    None => {
      // MLX's forward conv_general rejects an invalid rank before any per-axis
      // arithmetic, so only a valid rank can overflow here.
      if nd < 2 || wt_shape.len() != nd {
        return Ok(());
      }
      // Negative padding on any spatial axis makes MLX slice every dimension.
      if (0..nd - 2).any(|i| param(pad_lo, i, 0) < 0 || param(pad_hi, i, 0) < 0) {
        check_slice_all_dims(context, &in_shape)?;
      }
      for axis in 0..nd - 2 {
        let in_d =
          i128::from(i32::try_from(in_shape[axis + 1]).map_err(|_| conv_overflow_err(context))?);
        let wt_d =
          i128::from(i32::try_from(wt_shape[axis + 1]).map_err(|_| conv_overflow_err(context))?);
        let plo = i128::from(param(pad_lo, axis, 0));
        let phi = i128::from(param(pad_hi, axis, 0));
        let dil = i128::from(param(kernel_dilation, axis, 1));
        let idil = i128::from(param(input_dilation, axis, 1));
        let strd = i128::from(param(stride, axis, 1));
        check_forward_axis(context, in_d, wt_d, plo, phi, dil, idil, strd)?;
      }
    }
    Some(out_pad) => {
      // MLX's conv_transpose_general runs its prelude over the wrapper arity
      // BEFORE the nested conv_general's rank check, so check the prelude over
      // every axis whose dimensions exist, regardless of rank validity.
      let arity = stride.len();
      let rank_valid = wt_shape.len() == nd && nd == arity + 2;
      let mut nested_has_neg = false;
      for i in 0..arity {
        if 1 + i >= in_shape.len() || 1 + i >= wt_shape.len() {
          continue; // dimension absent → MLX raises a rank error, not UB
        }
        let in_d =
          i128::from(i32::try_from(in_shape[1 + i]).map_err(|_| conv_overflow_err(context))?);
        let wt_d =
          i128::from(i32::try_from(wt_shape[1 + i]).map_err(|_| conv_overflow_err(context))?);
        let pad = i128::from(param(pad_lo, i, 0));
        let dil = i128::from(param(kernel_dilation, i, 1));
        let strd = i128::from(param(stride, i, 1));
        let op = i128::from(param(out_pad, i, 0));
        let (padding_lo, padding_hi) =
          check_transpose_prelude_axis(context, in_d, wt_d, pad, dil, strd, op)?;
        if padding_lo < 0 || padding_hi < 0 {
          nested_has_neg = true;
        }
        if rank_valid {
          // Nested conv_general (forward; stride 1, input_dilation = stride).
          check_forward_axis(context, in_d, wt_d, padding_lo, padding_hi, dil, strd, 1)?;
        }
      }
      if rank_valid && nested_has_neg {
        check_slice_all_dims(context, &in_shape)?;
      }
    }
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
    // Negative padding_hi crops the input before conv_out_shape, so MLX uses the
    // sliced (shorter) length. A huge input_dilation overflows on the original
    // length but not on the cropped length, so the guard must use the post-slice
    // length and NOT false-reject (Codex round-4 counterexample).
    let input = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
    let weight = Array::from_slice::<f32>(&[1.0], &[1, 1, 1]).unwrap();
    let out = conv_general(
      &input,
      &weight,
      &[1],
      &[0],
      &[-2],
      &[1],
      &[i32::MAX],
      1,
      false,
    )
    .expect("valid negative-padding crop must not be falsely rejected");
    assert_eq!(out.shape(), vec![1, 1, 1]);
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
}
