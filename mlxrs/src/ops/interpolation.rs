//! Spatial interpolation ops.
//!
//! Two primitives:
//!
//! - [`bilinear_interpolate`] — a separable **bilinear + antialias** resize of
//!   a 2-D spatial grid, matching PyTorch
//!   `torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
//!   antialias=True)` exactly (the antialias triangle-filter path of `aten`'s
//!   `UpSampleKernel.cpp`, not the plain 2-tap align-corners=False kernel).
//! - [`bicubic_interpolate`] — a separable **bicubic** (Keys' cubic, `a =
//!   -0.5`) resize of a `(B, C, H, W)` batched grid, the pure-MLX fallback
//!   `mlx-vlm` uses when no Metal kernel is available
//!   (`mlx-vlm/mlx_vlm/models/kernels.py`'s `_bicubic_interpolate_mlx`). The
//!   LFM2.5-VL SigLIP2 vision tower resizes its learned position-embedding grid
//!   per image with this (`vision.py`'s `resize_positional_embeddings`).
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
    indexing::take_axis,
    linalg_basic::matmul,
    misc::astype,
    reduction::sum_axes,
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

// ───────────────────────────── bicubic ─────────────────────────────

/// The bicubic tap count for one resampled axis: `int(2 * support + 1)` with
/// `support = 2.0`. A `support = 2` cubic spans the two input pixels on each side
/// of the source coordinate, so each output row reads `5` candidate taps. The
/// four taps `floor-1..floor+2` are the active window (PyTorch's bicubic uses
/// exactly these four; `mlx-vlm`'s `_bicubic_interpolate_mlx` uses the same
/// support-2 window); the fifth tap (`floor+3`, distance `∈ (2, 3]`) is always
/// zero-weighted by the cubic kernel, and off-grid taps are clamp-folded onto a
/// valid index. Both bicubic paths (mlx-vlm Keys' `a = -0.5` and PyTorch's
/// `A = -0.75`) share this 5-candidate layout.
const BICUBIC_TAPS: usize = 5;

/// Cubic-convolution coefficient for the `mlx-vlm` / SigLIP2 bicubic path
/// (`kernels.py`'s `_cubic_weight`): Keys' cubic with `a = -0.5`.
const CUBIC_A_KEYS: f64 = -0.5;

/// Cubic-convolution coefficient for PyTorch's bicubic path
/// (`torch.nn.functional.interpolate(mode="bicubic")`): `A = -0.75`, the value
/// `get_cubic_upsample_coefficients` hard-codes in
/// `aten/src/ATen/native/UpSample.h`. This is the kernel HF CLAP's
/// `reshape_mel2img` resize runs through (it calls PyTorch bicubic), so the CLAP
/// `align_corners=True` path must use this coefficient, not Keys' `-0.5`.
const CUBIC_A_PYTORCH: f64 = -0.75;

/// Two-piece cubic convolution kernel parameterized by the coefficient `A`,
/// evaluated in `f64` for the host-side weight build:
///
/// ```text
/// w(t) = (A+2)|t|^3 - (A+3)|t|^2 + 1            for |t| <= 1
///        A|t|^3 - 5A|t|^2 + 8A|t| - 4A          for 1 < |t| < 2
///        0                                      for |t| >= 2
/// ```
///
/// The `|t| <= 1` branch is PyTorch's `cubic_convolution1(|t|, A)` and the
/// `1 < |t| < 2` branch its `cubic_convolution2(|t|, A)`
/// (`aten/src/ATen/native/UpSample.h`); with `A = -0.5` it is Keys' cubic
/// (`kernels.py`'s `_cubic_weight`), with `A = -0.75` it is PyTorch's bicubic
/// kernel. Both forms are a partition of unity over the four `floor-1..floor+2`
/// taps, so the four in-window coefficients sum to `1` for any source phase.
fn cubic_weight(t: f64, a: f64) -> f64 {
  let at = t.abs();
  let at2 = at * at;
  let at3 = at2 * at;
  if at <= 1.0 {
    // PyTorch `cubic_convolution1(at, a)`.
    (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
  } else if at < 2.0 {
    // PyTorch `cubic_convolution2(at, a)`.
    a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
  } else {
    0.0
  }
}

/// How a bicubic axis builder handles taps that fall outside `[0, in_dim)`.
///
/// The two reference bicubic implementations differ ONLY here (the kernel form,
/// the 5-tap window, and the index clamp are shared):
#[derive(Clone, Copy, PartialEq, Eq)]
enum CubicBoundary {
  /// `mlx-vlm`'s `_bicubic_interpolate_mlx` (`kernels.py`'s `_weights_1d`):
  /// zero-weight every out-of-bounds tap, then renormalize the surviving
  /// in-bounds weights to sum to 1 (`w / (sum(w) + 1e-8)`).
  ZeroRenormalize,
  /// PyTorch's `interpolate(mode="bicubic")`
  /// (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`'s
  /// `upsample_get_value_bounded`): keep the full cubic coefficient on an
  /// out-of-bounds tap and read the **replicated edge** pixel
  /// (`data[clamp(x, 0, width-1)]`); do NOT renormalize — the four
  /// `get_cubic_upsample_coefficients` already sum to 1 by construction.
  ReplicateEdge,
}

/// Build the `(out_dim, BICUBIC_TAPS)` per-axis resampling tables for the
/// bicubic `align_corners=False`, `antialias=False` path — `kernels.py`'s
/// `_weights_1d` specialized to `support = 2.0`, `fs = 1.0`.
///
/// Returns `(pix, weights)` flattened row-major:
/// - `pix[i * TAPS + k]` is the (clamped-to-`[0, in_dim)`) source index of the
///   `k`-th tap of output row `i`;
/// - `weights[i * TAPS + k]` is its (renormalized) cubic weight.
///
/// The source coordinate is the half-pixel center `c = (i + 0.5) / out * in -
/// 0.5`; the first tap starts at `floor(c - support) + 1`; out-of-bounds taps
/// are zero-weighted (and their index clamped so the on-device gather only ever
/// reads a valid row), then the surviving weights are renormalized to sum to 1
/// (matching `_weights_1d`'s `w / (sum(w) + 1e-8)`). `in_dim`, `out_dim >= 1`.
fn build_bicubic_axis(in_dim: usize, out_dim: usize) -> (Vec<i32>, Vec<f32>) {
  let scale_in = in_dim as f64;
  let scale_out = out_dim as f64;
  // `(arange(out) + 0.5) / out * in - 0.5` — the half-pixel source center;
  // Keys' cubic (`a = -0.5`) with the mlx-vlm zero-weight + renormalize edge.
  build_bicubic_axis_with(
    in_dim,
    out_dim,
    CUBIC_A_KEYS,
    CubicBoundary::ZeroRenormalize,
    |i| (i as f64 + 0.5) / scale_out * scale_in - 0.5,
  )
}

/// Build the `(out_dim, BICUBIC_TAPS)` per-axis resampling tables for the
/// bicubic **`align_corners=True`**, `antialias=False` path — PyTorch
/// `nn.functional.interpolate(mode="bicubic", align_corners=True)` (the variant
/// HF CLAP's `reshape_mel2img` uses to stretch the mel spectrogram's time axis).
///
/// This is a faithful port of PyTorch's bicubic, NOT the mlx-vlm Keys' variant,
/// so it differs from [`build_bicubic_axis`] in two ways:
///
/// 1. **Coefficient**: the cubic uses `A = -0.75` ([`CUBIC_A_PYTORCH`], the value
///    `get_cubic_upsample_coefficients` hard-codes in
///    `aten/src/ATen/native/UpSample.h`), not Keys' `-0.5`.
/// 2. **Source-coordinate map**: `align_corners=True` aligns the input/output
///    endpoints exactly, so for an output axis of length `out > 1` the source
///    center is `c = i · (in − 1) / (out − 1)` (`i = 0 → 0`, `i = out − 1 →
///    in − 1`), with NO half-pixel `−0.5` shift. For `out == 1` PyTorch maps to
///    the single source coordinate `0` (it avoids the `out − 1 == 0` division).
///
/// The boundary handling is [`CubicBoundary::ReplicateEdge`] — PyTorch's
/// `upsample_get_value_bounded`
/// (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`): an out-of-range tap keeps its
/// full cubic coefficient and reads the **replicated edge** pixel
/// (`data[clamp(x, 0, width-1)]`), with NO renormalization (the four
/// coefficients already sum to 1). The 5-tap window starting at `floor(c) − 1`
/// and the `[0, in_dim)` index clamp are shared with [`build_bicubic_axis`].
/// `in_dim`, `out_dim >= 1`.
fn build_bicubic_axis_aligned(in_dim: usize, out_dim: usize) -> (Vec<i32>, Vec<f32>) {
  let scale = if out_dim > 1 {
    (in_dim as f64 - 1.0) / (out_dim as f64 - 1.0)
  } else {
    0.0
  };
  build_bicubic_axis_with(
    in_dim,
    out_dim,
    CUBIC_A_PYTORCH,
    CubicBoundary::ReplicateEdge,
    |i| i as f64 * scale,
  )
}

/// Shared core for the two bicubic axis builders: given the cubic coefficient
/// `a`, the out-of-bounds `boundary` policy, and a `center(i)` that maps an
/// output index to its (fractional) source coordinate, build the
/// `(out_dim, BICUBIC_TAPS)` clamped-index + cubic-weight tables. The callers
/// differ ONLY in those three parameters (the `align_corners` coordinate map,
/// `A`, and the edge handling); the 5-tap window, the kernel form, and the
/// `[0, in_dim)` index clamp are identical.
///
/// In both modes the index is clamped to `[0, in_dim)` so the on-device gather
/// only ever reads a valid row. The weight handling differs per [`CubicBoundary`]:
/// - [`CubicBoundary::ZeroRenormalize`]: out-of-bounds taps are zero-weighted,
///   then the surviving weights are renormalized to sum to 1 (`w / (sum(w) +
///   1e-8)`) — `kernels.py`'s `_weights_1d`.
/// - [`CubicBoundary::ReplicateEdge`]: every tap (in- or out-of-bounds) keeps its
///   full cubic coefficient, and the clamped index makes an out-of-bounds tap
///   read the replicated edge pixel; NO renormalization — PyTorch's
///   `upsample_get_value_bounded`. The four `floor-1..floor+2` cubic taps already
///   sum to 1, and the fifth (`floor+3`, distance `∈ (2, 3]`) is exactly
///   zero-weighted by the kernel, so the row sums to 1 without renormalizing.
fn build_bicubic_axis_with(
  in_dim: usize,
  out_dim: usize,
  a: f64,
  boundary: CubicBoundary,
  center: impl Fn(usize) -> f64,
) -> (Vec<i32>, Vec<f32>) {
  // `support = 2.0`, `fs = 1.0` (antialias off). Both factors are small, and
  // the caller has bounded `in_dim` / `out_dim` to representable spatial dims,
  // so `out_dim * TAPS` cannot overflow `usize` on any supported platform.
  let total = out_dim * BICUBIC_TAPS;
  let mut pix = vec![0i32; total];
  let mut weights = vec![0.0f32; total];
  let in_i = in_dim as i64;
  for i in 0..out_dim {
    let center = center(i);
    // `start = floor(center - support) + 1` (support = 2) = `floor(center) - 1`,
    // so the five candidate taps are `floor-1 .. floor+3`; the first four are
    // PyTorch's `floor-1..floor+2` window and the fifth (distance `∈ (2, 3]`) is
    // always zero-weighted by the cubic kernel.
    let start = (center - 2.0).floor() as i64 + 1;
    let row = i * BICUBIC_TAPS;
    // First pass: cubic weights over the TAPS candidate indices.
    let mut tot = 0.0f64;
    for k in 0..BICUBIC_TAPS {
      let p = start + k as i64;
      let in_bounds = p >= 0 && p < in_i;
      // `dist = center - p`; `fs = 1.0`, so `cubic_weight(dist / fs, a)`.
      // ReplicateEdge keeps the full coefficient on every tap (the clamped index
      // reads the edge pixel); ZeroRenormalize masks out-of-bounds taps to 0.
      let w = if in_bounds || boundary == CubicBoundary::ReplicateEdge {
        cubic_weight(center - p as f64, a)
      } else {
        0.0
      };
      weights[row + k] = w as f32;
      tot += w;
      // Clamp the (possibly out-of-bounds) index into `[0, in_dim)` so the
      // on-device gather always reads a valid row. Under ReplicateEdge a clamped
      // out-of-bounds tap reads the replicated edge pixel (PyTorch); under
      // ZeroRenormalize its zero weight makes the duplicate read inert
      // (`kernels.py`'s `mx.clip`).
      pix[row + k] = p.clamp(0, in_i - 1) as i32;
    }
    // Second pass: ZeroRenormalize divides by the surviving sum (`w / (sum(w) +
    // 1e-8)`). ReplicateEdge leaves the weights as-is: the four in-window cubic
    // coefficients already sum to 1 (PyTorch does not renormalize).
    if boundary == CubicBoundary::ZeroRenormalize {
      let inv = 1.0 / (tot + 1e-8);
      for k in 0..BICUBIC_TAPS {
        weights[row + k] = (weights[row + k] as f64 * inv) as f32;
      }
    }
  }
  (pix, weights)
}

/// Bicubic-resize a batched `(B, C, H_in, W_in)` grid to `(B, C, out_h,
/// out_w)`, matching `mlx-vlm`'s pure-MLX `_bicubic_interpolate_mlx`
/// (`align_corners=False`, `antialias=False`: Keys' cubic, `a = -0.5`).
///
/// This is the resize the LFM2.5-VL SigLIP2 vision tower applies to its learned
/// position-embedding grid per image (`vision.py`'s
/// `resize_positional_embeddings` calls `bicubic_interpolate(pos[None], size=(h,
/// w))`). Like `_bicubic_interpolate_mlx`, the resampling is separable: build a
/// `(out, TAPS)` cubic-weight table per axis (on the host), gather the candidate
/// taps along that axis, and contract over the taps. Height is resampled first,
/// then width — exactly the reference's two-stage `gather + sum`.
///
/// `x` is a rank-4 `(B, C, H_in, W_in)` float array; the output is
/// `(B, C, out_h, out_w)` in the **same dtype** as `x` (computed in `f32` and
/// cast back, mirroring the reference's `astype(float32)` / restore). The
/// channel and batch axes pass through untouched (the resize is purely spatial).
///
/// ## Errors
/// - `x` is not rank-4 → [`Error::RankMismatch`].
/// - any of `B`, `C`, `H_in`, `W_in`, `out_h`, `out_w` is `0` →
///   [`Error::OutOfRange`]; or exceeds the per-axis cap (`MAX_INTERP_DIM`, 4096)
///   → [`Error::CapExceeded`].
/// - a per-axis tap table (`out_dim * BICUBIC_TAPS`) exceeds
///   `MAX_INTERP_WEIGHT_ELEMS`, or any resample tensor — the height/width gather
///   (`B * C * out_h * TAPS * W_in`, `B * C * out_h * out_w * TAPS`), the
///   post-height intermediate, or the output — exceeds the resample cap
///   (`MAX_INTERP_RESAMPLE_ELEMS`), even with every axis within `MAX_INTERP_DIM`
///   → [`Error::CapExceeded`] (or [`Error::ArithmeticOverflow`] if the product
///   overflows `usize`), rejected before any host vector / device tensor is
///   built.
/// - `x`'s dtype is non-floating (the cubic weights are fractional; an integer
///   grid would truncate every sample) → [`Error::UnsupportedDtype`].
/// - underlying gather / reshape / reduce / cast op errors propagate (a
///   non-finite grid value flows through unchanged — interpolation is linear in
///   the samples).
pub fn bicubic_interpolate(x: &Array, out_h: usize, out_w: usize) -> Result<Array> {
  bicubic_resample(x, out_h, out_w, build_bicubic_axis)
}

/// Bicubic-resize a batched `(B, C, H_in, W_in)` grid to `(B, C, out_h, out_w)`
/// with **`align_corners=True`** — PyTorch
/// `nn.functional.interpolate(mode="bicubic", align_corners=True)`.
///
/// This is the resize HF CLAP's `ClapAudioEncoder.reshape_mel2img` applies to
/// the log-mel spectrogram before the patch-embed (it upsamples the time axis
/// from the mel frame count to `spec_size · freq_ratio` with `align_corners=True`
/// bicubic). It is a faithful port of PyTorch's bicubic, so it differs from the
/// mlx-vlm-flavored [`bicubic_interpolate`] in three ways (see the private
/// `build_bicubic_axis_aligned`): the cubic coefficient is PyTorch's `A = -0.75`
/// (not Keys' `-0.5`); the source-coordinate map aligns the endpoints with no
/// half-pixel shift; and out-of-range taps replicate the edge pixel with NO
/// renormalization (`upsample_get_value_bounded`) rather than zero-weight +
/// renormalize. The separable gather + cubic-weight contract, the dtype handling,
/// the bounds, and the errors are identical.
///
/// `x` is a rank-4 `(B, C, H_in, W_in)` float array; the output is
/// `(B, C, out_h, out_w)` in the **same dtype** as `x` (computed in `f32` and
/// cast back). The channel and batch axes pass through untouched.
///
/// ## Errors
/// Identical to [`bicubic_interpolate`]: rank ≠ 4 → [`Error::RankMismatch`]; a
/// zero spatial dim → [`Error::OutOfRange`]; an over-cap dimension or resample
/// tensor → [`Error::CapExceeded`] / [`Error::ArithmeticOverflow`]; a
/// non-floating dtype → [`Error::UnsupportedDtype`]; underlying gather / reshape
/// / reduce / cast errors propagate.
pub fn bicubic_interpolate_align_corners(x: &Array, out_h: usize, out_w: usize) -> Result<Array> {
  bicubic_resample(x, out_h, out_w, build_bicubic_axis_aligned)
}

/// The shared separable-bicubic resample, parameterized by the per-axis weight
/// builder (`align_corners` false vs true). Validates the rank / spatial dims /
/// dtype, then resamples height (axis 2) and width (axis 3) via a `gather +
/// weighted sum` over the `BICUBIC_TAPS` candidate taps — the two-stage
/// reference path mlx-vlm's `_bicubic_interpolate_mlx` uses.
fn bicubic_resample(
  x: &Array,
  out_h: usize,
  out_w: usize,
  build_axis: impl Fn(usize, usize) -> (Vec<i32>, Vec<f32>),
) -> Result<Array> {
  let shape = x.shape();
  if shape.len() != 4 {
    return Err(Error::RankMismatch(crate::error::RankMismatchPayload::new(
      "bicubic_interpolate: input must be rank-4 (B, C, H, W)",
      shape.len() as u32,
      shape,
    )));
  }
  let (b, c, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
  // Reject a zero (→ `OutOfRange`) or over-cap (→ `CapExceeded`) spatial
  // dimension on every axis — mirroring the bilinear path's `check_dim`. The
  // batch / channel axes are bounded the same way so the resample-tensor
  // products below have a valid per-factor cap (and stay within the matmul /
  // gather `i32` shapes); both real callers are well under the cap (CLAP audio
  // is `B`-small, `C = 1`; the SigLIP2 position-embed grid `C` is the embed
  // dim, a few thousand at most).
  check_dim("bicubic_interpolate: B", b)?;
  check_dim("bicubic_interpolate: C", c)?;
  check_dim("bicubic_interpolate: H_in", in_h)?;
  check_dim("bicubic_interpolate: W_in", in_w)?;
  check_dim("bicubic_interpolate: out_h", out_h)?;
  check_dim("bicubic_interpolate: out_w", out_w)?;

  // Bound the caller-controlled `out_h`/`out_w` intermediates BEFORE building
  // any host vector / gather index / device tensor — the same dimension + total
  // element caps the bilinear path applies. `check_dim` proved every axis is
  // `<= MAX_INTERP_DIM`, so each is a valid `Extent`; `elem_count` is the
  // checked product (overflow → `ArithmeticOverflow`, over-cap → `CapExceeded`)
  // against the resample cap. The gather index / weight host vectors are
  // `out_dim * BICUBIC_TAPS`, and the two gathered device tensors —
  // `(B, C, out_h * TAPS, W_in)` then `(B, C, out_h, out_w * TAPS)` — are the
  // largest buffers, so capping their products bounds the whole chain.
  let ext = |v: usize| Extent::new("bicubic_interpolate: spatial dim", v, MAX_INTERP_DIM);
  let taps_e = ext(BICUBIC_TAPS)?;
  let (b_e, c_e) = (ext(b)?, ext(c)?);
  // `in_h` is bounded by its `check_dim` above but is not a buffer-size factor:
  // the height resample gathers only `out_h * BICUBIC_TAPS` rows from `x` (the
  // tap indices, clamped into `[0, in_h)`) and casts *those* to f32 — it never
  // materializes a full-input f32 copy — so only `in_w` multiplies the gather /
  // intermediate tensors below.
  let in_w_e = ext(in_w)?;
  let (out_h_e, out_w_e) = (ext(out_h)?, ext(out_w)?);
  // Per-axis gather index / weight host vectors `out_dim * BICUBIC_TAPS`.
  elem_count(
    "bicubic_interpolate: height tap-table elements (out_h * TAPS)",
    &[out_h_e, taps_e],
    MAX_INTERP_WEIGHT_ELEMS,
  )?;
  elem_count(
    "bicubic_interpolate: width tap-table elements (out_w * TAPS)",
    &[out_w_e, taps_e],
    MAX_INTERP_WEIGHT_ELEMS,
  )?;
  // Height-gather tensor `(B, C, out_h * TAPS, W_in)`.
  elem_count(
    "bicubic_interpolate: height-gather elements (B * C * out_h * TAPS * W_in)",
    &[b_e, c_e, out_h_e, taps_e, in_w_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;
  // Post-height intermediate `(B, C, out_h, W_in)`.
  elem_count(
    "bicubic_interpolate: height-resample elements (B * C * out_h * W_in)",
    &[b_e, c_e, out_h_e, in_w_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;
  // Width-gather tensor `(B, C, out_h, out_w * TAPS)`.
  elem_count(
    "bicubic_interpolate: width-gather elements (B * C * out_h * out_w * TAPS)",
    &[b_e, c_e, out_h_e, out_w_e, taps_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;
  // Final output `(B, C, out_h, out_w)`.
  elem_count(
    "bicubic_interpolate: output elements (B * C * out_h * out_w)",
    &[b_e, c_e, out_h_e, out_w_e],
    MAX_INTERP_RESAMPLE_ELEMS,
  )?;

  // The cubic weights are fractional, so a non-floating grid would lose every
  // sample to truncation. Restrict to the float dtypes (the position-embed
  // table is always float). The reference casts to f32 internally; preserve the
  // input dtype on the way out so an f16/bf16 grid stays in its dtype.
  let in_dtype = x.dtype()?;
  if !matches!(in_dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
    return Err(Error::UnsupportedDtype(
      crate::error::UnsupportedDtypePayload::new(
        "bicubic_interpolate: input dtype",
        in_dtype,
        &[Dtype::F32, Dtype::F16, Dtype::BF16],
      ),
    ));
  }

  // ── height resample: gather TAPS rows along axis 2, contract over taps ──
  let (pix_y, wy) = build_axis(in_h, out_h);
  // `x[:, :, pix_y.reshape(-1), :]` → (B, C, out_h * TAPS, W_in).
  let pix_y_arr = Array::from_slice::<i32>(&pix_y, &(out_h * BICUBIC_TAPS,))?;
  // The reference casts the whole input to f32 up front (`x.astype(float32)`)
  // then gathers. We gather in the input dtype and cast the *gathered* rows
  // instead: the gather only copies elements, so it commutes with the
  // elementwise cast (bit-identical result), but the f32 buffer is now the
  // height-gather tensor — already bounded by its `elem_count` cap above —
  // rather than a full-input f32 copy whose size scales with the uncapped
  // `in_h`. The width resample then runs on `tmp`, which is already f32.
  let gathered_y = take_axis(x, &pix_y_arr, 2)?;
  let gathered_y = if in_dtype == Dtype::F32 {
    gathered_y
  } else {
    astype(&gathered_y, Dtype::F32)?
  };
  // → (B, C, out_h, TAPS, W_in).
  let gathered_y = reshape(&gathered_y, &(b, c, out_h, BICUBIC_TAPS, in_w))?;
  // weights (out_h, TAPS) → (1, 1, out_h, TAPS, 1) to broadcast, then
  // `sum(gathered * wy, axis=3)` → (B, C, out_h, W_in).
  let wy_arr = reshape(
    &Array::from_slice::<f32>(&wy, &(out_h, BICUBIC_TAPS))?,
    &(1usize, 1, out_h, BICUBIC_TAPS, 1),
  )?;
  let tmp = sum_axes(&gathered_y.multiply(&wy_arr)?, &[3], false)?;

  // ── width resample: gather TAPS columns along axis 3, contract over taps ──
  let (pix_x, wx) = build_axis(in_w, out_w);
  // `tmp[:, :, :, pix_x.reshape(-1)]` → (B, C, out_h, out_w * TAPS).
  let pix_x_arr = Array::from_slice::<i32>(&pix_x, &(out_w * BICUBIC_TAPS,))?;
  let gathered_x = take_axis(&tmp, &pix_x_arr, 3)?;
  // → (B, C, out_h, out_w, TAPS).
  let gathered_x = reshape(&gathered_x, &(b, c, out_h, out_w, BICUBIC_TAPS))?;
  // weights (out_w, TAPS) → (1, 1, 1, out_w, TAPS), then
  // `sum(gathered * wx, axis=4)` → (B, C, out_h, out_w).
  let wx_arr = reshape(
    &Array::from_slice::<f32>(&wx, &(out_w, BICUBIC_TAPS))?,
    &(1usize, 1, 1, out_w, BICUBIC_TAPS),
  )?;
  let result = sum_axes(&gathered_x.multiply(&wx_arr)?, &[4], false)?;

  // Restore the input dtype (the reference's `astype(input_dtype)`).
  if in_dtype == Dtype::F32 {
    Ok(result)
  } else {
    astype(&result, in_dtype)
  }
}

#[cfg(test)]
mod tests;
