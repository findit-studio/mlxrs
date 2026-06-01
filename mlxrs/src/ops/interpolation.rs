//! Spatial interpolation ops.
//!
//! Currently the single primitive [`bicubic_interpolate`] — a separable
//! bicubic resize of a 2-D spatial grid, matching PyTorch
//! `torch.nn.functional.interpolate(mode="bicubic", align_corners=False)`
//! with the Keys cubic convolution coefficient **`a = -0.75`** (PyTorch's
//! default, which differs from PIL/`Image.BICUBIC`'s `a = -0.5`).
//!
//! This is the resize SigLIP2 NaFlex (and other native-resolution ViTs)
//! use to stretch a learned position-embedding grid from its trained
//! square shape (e.g. `16 x 16`) to a per-image patch grid
//! `(H_patch, W_patch)`. It is placed in [`crate::ops`] — not inside the
//! model code — so independent vision ports can share the one primitive
//! without coupling their model implementations.
//!
//! ## Why on-device (MLX), not a NEON/SIMD kernel
//!
//! The interpolation runs through MLX ([`matmul`] over precomputed
//! weight matrices), so it executes on the active MLX device (GPU on
//! Apple silicon) alongside the rest of the vision graph. A hand-rolled
//! CPU/NEON kernel is deliberately **not** used here: the position-embed
//! grid is already a device [`Array`], so a CPU path would force a
//! device→host→device round-trip purely to interpolate a tiny grid,
//! which would dominate the cost. Only the constant cubic-weight tables
//! are built on the host (a few hundred `f32`); the actual resampling is
//! two device matmuls. (Contrast the NaFlex *image* resize, which starts
//! from host RGB bytes and therefore correctly reuses the NEON
//! `vlm::resize`.)
//!
//! ## Algorithm (separable bicubic, `a = -0.75`, `align_corners=False`)
//!
//! Bicubic interpolation is separable: a 2-D resize equals a 1-D resize
//! along each axis. For an output axis of length `out` resampling an
//! input axis of length `in`, the source coordinate of output index `j`
//! is
//!
//! ```text
//! src = (j + 0.5) * (in / out) - 0.5
//! ```
//!
//! (the half-pixel-centered `align_corners=False` map). The four input
//! taps are `floor(src) - 1 ..= floor(src) + 2`, each weighted by the
//! Keys cubic kernel evaluated at its signed distance to `src`; source
//! indices are clamped to `[0, in - 1]` (edge replication — matching
//! PyTorch, which clamps the gather index rather than zero-padding). The
//! four weights are **not** renormalized (also matching PyTorch). This
//! produces a dense `(out, in)` weight matrix `W` per axis, and the
//! resize is `W_h · X · W_wᵀ` contracted over the two spatial axes via
//! [`matmul`].

use crate::{
  array::Array,
  dtype::Dtype,
  error::{ArithmeticOverflowPayload, CapExceededPayload, Error, OutOfRangePayload, Result},
  ops::{
    linalg_basic::matmul,
    misc::astype,
    shape::{contiguous, reshape, transpose_axes},
  },
};

/// Keys cubic convolution coefficient used by PyTorch
/// `F.interpolate(mode="bicubic")`. PIL/`Image.BICUBIC` uses `-0.5`;
/// SigLIP2 NaFlex resamples its position embeddings with the PyTorch
/// kernel, so this port pins `-0.75`.
const CUBIC_A: f64 = -0.75;

/// Upper bound on either output spatial dimension. The bicubic weight
/// matrices are dense (`out * in` `f32`), built on the host before the
/// device matmuls; this caps the host build (and the device matmul
/// shapes) so a hostile / mis-derived `(out_h, out_w)` cannot drive an
/// unbounded host allocation. A position-embed grid resize is at most a
/// few thousand patches per side in any realistic ViT, so `4096` is far
/// above any legitimate use while still bounding the `out * in` product
/// well within `i32` for the matmul.
const MAX_INTERP_DIM: usize = 4096;

/// The Keys cubic convolution kernel `k(t)` with coefficient `a`
/// ([`CUBIC_A`]), evaluated at signed distance `t`.
///
/// ```text
/// |t| <= 1:        (a + 2)|t|^3 - (a + 3)|t|^2 + 1
/// 1 <  |t| < 2:    a|t|^3 - 5a|t|^2 + 8a|t| - 4a
/// |t| >= 2:        0
/// ```
///
/// This is the standard piecewise cubic PyTorch's bicubic resampler
/// uses (`aten` `cubic_convolution1/2` with `A = -0.75`). Computed in
/// `f64` for the host-side weight build; the resulting weights are cast
/// to the input dtype before the device matmul.
fn cubic_kernel(t: f64) -> f64 {
  let x = t.abs();
  if x <= 1.0 {
    ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
  } else if x < 2.0 {
    (((x - 5.0) * x + 8.0) * x - 4.0) * CUBIC_A
  } else {
    0.0
  }
}

/// Build the dense `(out, in)` bicubic resampling weight matrix for one
/// axis, as a row-major `Vec<f32>` of length `out * in`.
///
/// Row `j` (output position) holds the four non-zero cubic weights at
/// the clamped source taps `floor(src) - 1 ..= floor(src) + 2`, where
/// `src = (j + 0.5) * scale - 0.5` and `scale = in / out`. Border taps
/// are clamped to `[0, in - 1]` (edge replication), so a tap that falls
/// off the edge **accumulates** its weight onto the boundary column
/// (the `+=`), exactly as PyTorch's clamped-index gather does. Weights
/// are not renormalized.
///
/// `in_dim` and `out_dim` are both `>= 1` (guaranteed by the caller).
fn build_axis_weights(in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
  // `out * in` bounds both the host weight buffer and the device matmul
  // operand; the per-axis caps already bound each factor by
  // `MAX_INTERP_DIM`, so this product cannot overflow `usize`, but keep
  // the checked form so the matmul `i32` shape stays honest.
  let total = out_dim.checked_mul(in_dim).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "bicubic_interpolate: weight-matrix size (out * in)",
      "usize",
      [("out", out_dim as u64), ("in", in_dim as u64)],
    ))
  })?;
  let mut w = vec![0.0f32; total];
  let scale = in_dim as f64 / out_dim as f64;
  let in_last = (in_dim - 1) as isize;
  for j in 0..out_dim {
    let src = (j as f64 + 0.5) * scale - 0.5;
    let base = src.floor();
    let frac = src - base; // in [0, 1)
    let base_i = base as isize;
    let row = j * in_dim;
    // Four taps at offsets -1, 0, 1, 2 from `base`. The signed distance
    // from the tap to `src` is `frac - offset`; the cubic kernel is
    // even, so the sign is immaterial to `cubic_kernel`.
    for offset in -1isize..=2 {
      let weight = cubic_kernel(frac - offset as f64) as f32;
      let tap = (base_i + offset).clamp(0, in_last) as usize;
      // `+=`: a clamped (off-edge) tap folds its weight onto the
      // boundary column, matching PyTorch's clamped-index gather.
      w[row + tap] += weight;
    }
  }
  Ok(w)
}

/// Reject a spatial dimension outside `[1, MAX_INTERP_DIM]`.
fn check_dim(context: &'static str, value: usize) -> Result<()> {
  if value == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must be a positive spatial dimension (>= 1)",
      "0",
    )));
  }
  if value > MAX_INTERP_DIM {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      context,
      "MAX_INTERP_DIM",
      MAX_INTERP_DIM as u64,
      value as u64,
    )));
  }
  Ok(())
}

/// Bicubic-resize a 2-D spatial grid from `(H_in, W_in, C)` to
/// `(out_h, out_w, C)`, matching PyTorch
/// `F.interpolate(mode="bicubic", align_corners=False)` with cubic
/// coefficient `a = -0.75`.
///
/// `grid` is a rank-3 `(H_in, W_in, C)` float array — the layout SigLIP2
/// reshapes its `(num_positions, embed_dim)` position-embedding table
/// into before resampling (`H_in == W_in == sqrt(num_positions)`,
/// `C == embed_dim`). The channel axis is interpolated independently
/// (the resize is purely spatial). The output is a fresh
/// `(out_h, out_w, C)` array in the **same dtype** as `grid` (the weight
/// matrices are cast to `grid.dtype()` before the matmuls, so an
/// f16/bf16 grid stays in its dtype — no silent f32 promotion).
///
/// The resize is computed as two [`matmul`]s by the separable identity
/// `out = W_h · grid · W_wᵀ`:
/// - `W_h` is the `(out_h, H_in)` row-resampling matrix,
/// - `W_w` is the `(out_w, W_in)` column-resampling matrix,
///
/// each built from the Keys cubic kernel (`a = -0.75`) at the
/// `align_corners=False` half-pixel source coordinates (see the module
/// docs). Both matmuls run on the active MLX device.
///
/// ## Errors
/// - `grid` is not rank-3 → [`Error::RankMismatch`].
/// - any of `H_in`, `W_in`, `out_h`, `out_w` is `0`, or `out_h` / `out_w`
///   exceeds the dense-weight-matrix cap (`MAX_INTERP_DIM`, 4096) →
///   [`Error::OutOfRange`] / [`Error::CapExceeded`].
/// - `grid`'s dtype is non-floating (the cubic kernel produces
///   fractional weights; an integer grid would truncate every sample) →
///   [`Error::UnsupportedDtype`].
/// - underlying [`matmul`] / [`reshape`] / [`transpose_axes`] /
///   [`astype`] errors propagate (e.g. a non-finite grid value flows
///   through unchanged — interpolation is linear in the samples).
pub fn bicubic_interpolate(grid: &Array, out_h: usize, out_w: usize) -> Result<Array> {
  let shape = grid.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
      "bicubic_interpolate: grid must be rank-3 (H, W, C)",
      shape.len() as u32,
      shape,
    )));
  }
  let (h_in, w_in, c) = (shape[0], shape[1], shape[2]);
  check_dim("bicubic_interpolate: H_in", h_in)?;
  check_dim("bicubic_interpolate: W_in", w_in)?;
  check_dim("bicubic_interpolate: out_h", out_h)?;
  check_dim("bicubic_interpolate: out_w", out_w)?;
  // The channel axis only needs to be representable in the reshape /
  // matmul `i32` shapes; it is bounded by the same cap so the
  // `(W_in * C)` and `(H_out * C)` flattened operands stay within `i32`.
  check_dim("bicubic_interpolate: C", c)?;

  // The cubic weights are fractional, so a non-floating grid would lose
  // every sample to truncation. Restrict to the float dtypes (the
  // position-embed table is always float).
  let dtype = grid.dtype()?;
  if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
    return Err(Error::UnsupportedDtype(
      crate::error::UnsupportedDtypePayload::new(
        "bicubic_interpolate: grid dtype",
        dtype,
        &[Dtype::F32, Dtype::F16, Dtype::BF16],
      ),
    ));
  }

  // Fast path: identical shape ⇒ the bicubic kernel reduces to identity
  // weights, but skip the matmuls entirely (and the f32 round-trip a
  // weight build + cast would introduce) so a no-op resize is exactly
  // the input. `try_clone` preserves the lazy graph reference.
  if h_in == out_h && w_in == out_w {
    return grid.try_clone();
  }

  // Build the two host-side weight matrices and move them onto the
  // device in the grid's dtype.
  let w_h_host = build_axis_weights(h_in, out_h)?; // (out_h, H_in)
  let w_w_host = build_axis_weights(w_in, out_w)?; // (out_w, W_in)
  let w_h = astype(&Array::from_slice::<f32>(&w_h_host, &(out_h, h_in))?, dtype)?;
  let w_w = astype(&Array::from_slice::<f32>(&w_w_host, &(out_w, w_in))?, dtype)?;

  // Row resample: (out_h, H_in) @ (H_in, W_in*C) -> (out_h, W_in*C).
  let grid_hw = reshape(grid, &(h_in, w_in * c))?;
  let rows = matmul(&w_h, &grid_hw)?; // (out_h, W_in*C)
  let rows = reshape(&rows, &(out_h, w_in, c))?; // (out_h, W_in, C)

  // Column resample: bring W_in to the front, flatten the rest, apply
  // W_w, then restore the (H_out, W_out, C) layout.
  //   (out_h, W_in, C) -> (W_in, out_h, C) -> (W_in, out_h*C)
  let cols_in = transpose_axes(&rows, &[1, 0, 2])?; // (W_in, out_h, C)
  let cols_in = reshape(&cols_in, &(w_in, out_h * c))?; // (W_in, out_h*C)
  let cols = matmul(&w_w, &cols_in)?; // (out_w, out_h*C)
  let cols = reshape(&cols, &(out_w, out_h, c))?; // (out_w, out_h, C)
  // (out_w, out_h, C) -> (out_h, out_w, C). The transpose leaves a
  // strided view; materialize a row-contiguous result so downstream
  // buffer reads (`as_slice` / `to_vec`) don't hit `NonContiguous`. The
  // grid is tiny (position-embed sized), so the copy is negligible.
  let out = transpose_axes(&cols, &[1, 0, 2])?;
  contiguous(&out, false)
}

#[cfg(test)]
mod tests;
