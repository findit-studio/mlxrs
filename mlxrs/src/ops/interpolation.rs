//! Spatial interpolation ops.
//!
//! Currently the single primitive [`bilinear_interpolate`] — a separable
//! **bilinear + antialias** resize of a 2-D spatial grid, matching PyTorch
//! `torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
//! antialias=True)` exactly (the antialias triangle-filter path of `aten`'s
//! `UpSampleKernel.cpp`, not the plain 2-tap align-corners=False kernel).
//!
//! This is the resize SigLIP2 NaFlex (and other native-resolution ViTs)
//! use to stretch a learned position-embedding grid from its trained
//! square shape (e.g. `16 x 16`) to a per-image patch grid
//! `(H_patch, W_patch)`. HF's
//! `Siglip2VisionEmbeddings.resize_positional_embeddings` resizes with
//! `F.interpolate(..., mode="bilinear", align_corners=False,
//! antialias=True)`, and the future LFM2.5-VL SigLIP2 vision tower uses the
//! same kernel, so the primitive lives in [`crate::ops`] — not inside the
//! model code — so independent vision ports can share it without coupling
//! their model implementations.
//!
//! ## Why on-device (MLX), not a NEON/SIMD kernel
//!
//! The interpolation runs through MLX ([`matmul`] over precomputed
//! weight matrices), so it executes on the active MLX device (GPU on
//! Apple silicon) alongside the rest of the vision graph. A hand-rolled
//! CPU/NEON kernel is deliberately **not** used here: the position-embed
//! grid is already a device [`Array`], so a CPU path would force a
//! device→host→device round-trip purely to interpolate a tiny grid,
//! which would dominate the cost. Only the constant triangle-weight tables
//! are built on the host (a few hundred `f32`); the actual resampling is
//! two device matmuls. (Contrast the NaFlex *image* resize, which starts
//! from host RGB bytes and therefore correctly reuses the NEON
//! `vlm::resize`.)
//!
//! ## Algorithm (separable bilinear + antialias, `align_corners=False`)
//!
//! Bilinear interpolation is separable: a 2-D resize equals a 1-D resize
//! along each axis. PyTorch's antialias path (the `aa_filter` triangle in
//! `UpSampleKernel.cpp`) builds, per output index `i` on an axis of length
//! `out` resampling an input axis of length `in`:
//!
//! ```text
//! scale    = in / out                              (align_corners=False)
//! center   = scale * (i + 0.5)                     (NO half-pixel -0.5 shift)
//! support  = scale >= 1 ? scale : 1                (interp_size = 2 ⇒ *0.5)
//! invscale = scale >= 1 ? 1 / scale : 1
//! xmin     = max(floor(center - support + 0.5), 0)
//! xmax     = min(floor(center + support + 0.5), in)
//! ```
//!
//! and for each input tap `t` in `xmin..xmax` the (unnormalized) weight is
//! the triangle filter `f((t - center + 0.5) * invscale)` with
//! `f(x) = max(0, 1 - |x|)`. The taps are then **renormalized** to sum to
//! `1` (PyTorch divides each by their total). This produces a dense
//! `(out, in)` weight matrix `W` per axis, and the resize is
//! `W_h · X · W_wᵀ` contracted over the two spatial axes via [`matmul`].
//!
//! - When **downsampling** (`scale > 1`) the support is stretched by
//!   `scale`, so the filter averages over a wider neighbourhood — this is
//!   the antialiasing, and where the result differs from a plain 2-tap
//!   bilinear.
//! - When **upsampling** (`scale <= 1`) `support = 1`, `invscale = 1`, and
//!   the two-tap weights already sum to `1`, so the renormalization is a
//!   no-op and the result is exactly plain bilinear `align_corners=False`.
//!
//! Off-edge taps are excluded by the `[0, in]` clamp on `xmin`/`xmax` (the
//! kernel narrows the window at the borders and renormalizes the surviving
//! taps), matching PyTorch. Byte-level agreement with the oracle is
//! confirmed numerically only by the gated SigLIP2 e2e parity test (the
//! hand-computed unit cases below pin the weight formula on small grids).

use crate::{
  array::Array,
  dtype::Dtype,
  error::{ArithmeticOverflowPayload, CapExceededPayload, Error, OutOfRangePayload, Result},
  model_validation::{Extent, elem_count},
  ops::{
    linalg_basic::matmul,
    misc::astype,
    shape::{contiguous, reshape, transpose_axes},
  },
};

/// Upper bound on either output spatial dimension. The bilinear weight
/// matrices are dense (`out * in` `f32`), built on the host before the
/// device matmuls; this caps the host build (and the device matmul
/// shapes) so a hostile / mis-derived `(out_h, out_w)` cannot drive an
/// unbounded host allocation. A position-embed grid resize is at most a
/// few thousand patches per side in any realistic ViT, so `4096` is far
/// above any legitimate use while still bounding the `out * in` product
/// well within `i32` for the matmul.
const MAX_INTERP_DIM: usize = 4096;

/// Upper bound on the element count of a single `(out, in)` weight table.
///
/// The per-axis [`MAX_INTERP_DIM`] cap alone permits an `out * in` product up
/// to `MAX_INTERP_DIM^2` (≈ 16 Mi `f32` ≈ 64 MiB) for one table — large enough
/// that an infallible `vec![0.0f32; total]` would abort on allocator pressure.
/// A real position-embed resize is a `16 x 16` grid stretched to a few-thousand
/// patch grid, so one weight table is at most a few thousand `f32`; `1 << 22`
/// (4 Mi elements) is far above any legitimate use yet keeps an over-product a
/// recoverable [`Error::CapExceeded`] *before* the allocation, tighter than the
/// `MAX_INTERP_DIM^2` the per-axis caps would otherwise allow.
const MAX_INTERP_WEIGHT_ELEMS: usize = 1 << 22;

/// Upper bound on the element count of any **resample intermediate / output**
/// (a `dim * dim * C` device tensor in the separable matmul chain).
///
/// The per-axis [`MAX_INTERP_DIM`] and per-table [`MAX_INTERP_WEIGHT_ELEMS`]
/// caps bound the *weight matrices* but not the resampled tensors that thread the
/// channel axis `C` through the two matmuls: with each axis individually within
/// `MAX_INTERP_DIM` (e.g. `H_in = W_in = 4096`, `out_h = out_w = 1024`,
/// `C = 4096`) the first matmul output `out_h * W_in * C` is already ≈ 17 G
/// elements — an unbounded device allocation. This bounds each of the three
/// `*_count` products (row-resample, column-resample, final output) before any
/// device graph is built. A real position-embed resize is a `16 x 16 x hidden`
/// grid stretched to a few-thousand-patch grid (a few million elements at most),
/// so `1 << 26` (64 Mi elements ≈ 256 MiB `f32`) is far above any legitimate use
/// yet rejects the adversarial product as a typed [`Error::CapExceeded`] before
/// the matmul.
const MAX_INTERP_RESAMPLE_ELEMS: usize = 1 << 26;

/// PyTorch's antialias triangle (tent) filter `f(x) = max(0, 1 - |x|)`
/// (`HelperInterpLinear::aa_filter` in `aten`'s `UpSampleKernel.cpp`).
/// Evaluated in `f64` for the host-side weight build; the resulting
/// per-axis weights are cast to the grid dtype before the device matmul.
fn triangle_filter(x: f64) -> f64 {
  let x = x.abs();
  if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Build the dense `(out, in)` bilinear+antialias resampling weight matrix
/// for one axis, as a row-major `Vec<f32>` of length `out * in`.
///
/// Implements PyTorch's `UpSampleKernel.cpp` antialias linear path exactly
/// (see the module docs): per output row `i`, the source `center =
/// scale * (i + 0.5)` (no `-0.5` shift — antialias bakes the half-pixel
/// into the `+0.5` inside the filter argument), a `support` stretched by
/// `scale` when downsampling, the `[0, in]`-clamped tap window
/// `xmin..xmax`, the triangle filter at `(t - center + 0.5) * invscale`,
/// and a per-row renormalization so the surviving taps sum to `1`.
///
/// `in_dim` and `out_dim` are both `>= 1` (guaranteed by the caller).
fn build_axis_weights(in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
  // `out * in` bounds both the host weight buffer and the device matmul
  // operand; the per-axis caps already bound each factor by
  // `MAX_INTERP_DIM`, so this product cannot overflow `usize`, but keep
  // the checked form so the matmul `i32` shape stays honest.
  let total = out_dim.checked_mul(in_dim).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "bilinear_interpolate: weight-matrix size (out * in)",
      "usize",
      [("out", out_dim as u64), ("in", in_dim as u64)],
    ))
  })?;
  // Bound the table element count tighter than the per-axis caps allow
  // (`MAX_INTERP_DIM^2`): a within-axis but adversarial `(out, in)` pair could
  // still request a ~16 Mi-element table. Reject an over-product here, before
  // the allocation, as a typed `CapExceeded`.
  if total > MAX_INTERP_WEIGHT_ELEMS {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "bilinear_interpolate: weight-matrix element count (out * in)",
      "MAX_INTERP_WEIGHT_ELEMS",
      MAX_INTERP_WEIGHT_ELEMS as u64,
      total as u64,
    )));
  }
  // Build the zero-initialized weight buffer fallibly: a `vec![0.0f32; total]`
  // aborts on allocator failure, but at the cap one table is ~16 MiB, so a
  // within-cap-but-heavyweight reservation that the allocator cannot satisfy
  // must surface as a typed `AllocFailure` instead. `reserve_or_error` reserves
  // exactly `total`; `resize` then zero-fills without reallocating.
  let mut w: Vec<f32> = Vec::new();
  crate::model_validation::reserve_or_error(&mut w, "f32 interpolation weights", total)?;
  w.resize(total, 0.0f32);
  let scale = in_dim as f64 / out_dim as f64;
  // interp_size = 2 for linear ⇒ base support = interp_size * 0.5 = 1.0,
  // stretched by `scale` only when downsampling (scale >= 1).
  let support = if scale >= 1.0 { scale } else { 1.0 };
  let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
  let in_i = in_dim as i64;
  for i in 0..out_dim {
    let center = scale * (i as f64 + 0.5);
    // `floor(... + 0.5)` via i64 truncation of the non-negative arguments
    // (xmin's argument is clamped to >= 0; center + support + 0.5 > 0).
    let xmin = ((center - support + 0.5).floor() as i64).max(0);
    let xmax = ((center + support + 0.5).floor() as i64).min(in_i);
    let row = i * in_dim;
    // First pass: unnormalized triangle weights + their total.
    let mut tot = 0.0f64;
    let mut t = xmin;
    while t < xmax {
      let weight = triangle_filter((t as f64 - center + 0.5) * invscale);
      w[row + t as usize] = weight as f32;
      tot += weight;
      t += 1;
    }
    // Second pass: renormalize so the surviving taps sum to 1 (PyTorch
    // divides each weight by their total). `tot` is > 0 for every valid
    // (in, out) pair since the center always lands within a tap's support.
    if tot != 0.0 {
      let inv_tot = (1.0 / tot) as f32;
      let mut t = xmin;
      while t < xmax {
        w[row + t as usize] *= inv_tot;
        t += 1;
      }
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

/// Bilinear-resize (with antialias) a 2-D spatial grid from `(H_in, W_in, C)`
/// to `(out_h, out_w, C)`, matching PyTorch
/// `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`.
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
/// each built from the antialias triangle filter at the half-pixel source
/// coordinates (see the module docs). Both matmuls run on the active MLX
/// device.
///
/// ## Errors
/// - `grid` is not rank-3 → [`Error::RankMismatch`].
/// - any of `H_in`, `W_in`, `out_h`, `out_w` is `0`, or `out_h` / `out_w`
///   exceeds the dense-weight-matrix cap (`MAX_INTERP_DIM`, 4096) →
///   [`Error::OutOfRange`] / [`Error::CapExceeded`].
/// - either axis's `out * in` weight-table element count exceeds the tighter
///   product cap (`MAX_INTERP_WEIGHT_ELEMS`) → [`Error::CapExceeded`]; or the
///   (fallible) weight-buffer reservation exceeds available memory →
///   [`Error::AllocFailure`].
/// - any resample tensor's element count — the row-resample `out_h * W_in * C`,
///   the column-resample `out_w * out_h * C`, or the output `out_h * out_w * C`
///   — exceeds the resample cap (`MAX_INTERP_RESAMPLE_ELEMS`), even though every
///   axis is within `MAX_INTERP_DIM` → [`Error::CapExceeded`] (rejected before
///   any device array / matmul is built).
/// - `grid`'s dtype is non-floating (the triangle weights are fractional;
///   an integer grid would truncate every sample) →
///   [`Error::UnsupportedDtype`].
/// - underlying [`matmul`] / [`reshape`] / [`transpose_axes`] /
///   [`astype`] errors propagate (e.g. a non-finite grid value flows
///   through unchanged — interpolation is linear in the samples).
pub fn bilinear_interpolate(grid: &Array, out_h: usize, out_w: usize) -> Result<Array> {
  let shape = grid.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
      "bilinear_interpolate: grid must be rank-3 (H, W, C)",
      shape.len() as u32,
      shape,
    )));
  }
  let (h_in, w_in, c) = (shape[0], shape[1], shape[2]);
  check_dim("bilinear_interpolate: H_in", h_in)?;
  check_dim("bilinear_interpolate: W_in", w_in)?;
  check_dim("bilinear_interpolate: out_h", out_h)?;
  check_dim("bilinear_interpolate: out_w", out_w)?;
  // The channel axis only needs to be representable in the reshape /
  // matmul `i32` shapes; it is bounded by the same cap so the
  // `(W_in * C)` and `(H_out * C)` flattened operands stay within `i32`.
  check_dim("bilinear_interpolate: C", c)?;

  // Bound the three resample TENSORS (not just the weight tables): the
  // per-axis caps leave `dim * dim * C` unbounded — e.g. each axis within
  // `MAX_INTERP_DIM` still makes the first matmul output `out_h * W_in * C`
  // ≈ 17 G elements. `check_dim` already proved every axis is `<= MAX_INTERP_DIM`,
  // so each is a valid `Extent`; `elem_count` is the checked product against the
  // resample cap. Reject an over-product as a typed `CapExceeded` BEFORE any
  // device array / matmul is constructed.
  let ext = |v: usize| Extent::new("bilinear_interpolate: spatial dim", v, MAX_INTERP_DIM);
  let (w_in_e, c_e) = (ext(w_in)?, ext(c)?);
  let (out_h_e, out_w_e) = (ext(out_h)?, ext(out_w)?);
  // Row resample output `(out_h, W_in * C)`.
  elem_count(
    "bilinear_interpolate: row-resample elements (out_h * W_in * C)",
    &[out_h_e, w_in_e, c_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;
  // Column resample output `(out_w, out_h * C)`.
  elem_count(
    "bilinear_interpolate: column-resample elements (out_w * out_h * C)",
    &[out_w_e, out_h_e, c_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;
  // Final output `(out_h, out_w, C)`.
  elem_count(
    "bilinear_interpolate: output elements (out_h * out_w * C)",
    &[out_h_e, out_w_e, c_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;

  // The triangle weights are fractional, so a non-floating grid would lose
  // every sample to truncation. Restrict to the float dtypes (the
  // position-embed table is always float).
  let dtype = grid.dtype()?;
  if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
    return Err(Error::UnsupportedDtype(
      crate::error::UnsupportedDtypePayload::new(
        "bilinear_interpolate: grid dtype",
        dtype,
        &[Dtype::F32, Dtype::F16, Dtype::BF16],
      ),
    ));
  }

  // Fast path: identical shape ⇒ the antialias weights reduce to identity
  // (scale == 1 ⇒ each row is one-hot), but skip the matmuls entirely (and
  // the f32 round-trip a weight build + cast would introduce) so a no-op
  // resize is exactly the input. `try_clone` preserves the lazy graph
  // reference.
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
