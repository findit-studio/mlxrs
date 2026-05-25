//! Fully-fallible, PIL-matching RGBA8 image resize (own implementation).
//!
//! This module replaces the third-party `fast_image_resize` crate that
//! [`crate::vlm::image::resize`] previously delegated to. The motivation
//! is allocation safety, not performance parity: `resize`'s target
//! dimensions flow from an UNTRUSTED loaded `preprocessor_config.json`
//! (see [`crate::vlm::load`]), and `fast_image_resize` allocated internal
//! scratch (coefficient tables, per-row work buffers) *infallibly* inside
//! the crate — a hostile-but-under-cap target could `abort()` the process
//! despite our `Result` signature. Owning the whole resize lets EVERY
//! allocation route through `try_reserve_exact`, so `resize` returning
//! `Ok` guarantees no abort path for any (untrusted) target size.
//!
//! ## Correctness reference — PIL `Image.resize`
//! mlx-vlm preprocessing expects **PIL `Image.resize`** semantics (the
//! swift `MediaProcessing.resampleBicubic` mirrors PIL). The convolution
//! filters here reproduce PIL's `src/libImaging/Resample.c` *exactly*,
//! including its fixed-point integer accumulation, so the output is
//! **byte-for-byte identical to PIL** (verified against Pillow 12.2 over
//! bilinear/bicubic/lanczos, upscale + downscale, RGBA — see
//! `tests/vlm_image.rs`). No ±1 LSB tolerance is required for the scalar
//! path; it is bit-exact with PIL.
//!
//! ### Algorithm (matches `Resample.c`)
//! Separable two-pass convolution: a horizontal 1-D pass that emits an
//! 8-bit clamped intermediate image, then a vertical 1-D pass over that
//! intermediate. For each output coordinate the value is a weighted sum
//! of input pixels within the filter's support window, weights from the
//! filter kernel normalized to sum to 1.
//!
//! ### Premultiplied alpha (matches `Image.resize`)
//! PIL's `Image.resize` wrapper converts RGBA -> **premultiplied** `RGBa`
//! *before* any non-NEAREST resample and converts back after
//! (`Image.py`: `if self.mode in ["LA","RGBA"] and resample != NEAREST`).
//! Convolving straight (non-premultiplied) channels is NOT byte-exact for
//! an image with non-opaque alpha — it leaks the colour of
//! fully-transparent pixels into their neighbours. This module mirrors
//! that path exactly: it premultiplies the source colour channels
//! (`MULDIV255`), runs the separable convolution over `RGBa`, then
//! unpremultiplies the destination (`rgba2rgbA`'s `CLIP8(255*c/a)`). For
//! an all-opaque (`A == 255`) image both conversions are the identity, so
//! opaque inputs stay bit-identical to a straight-channel resize.
//! **NEAREST is exempt** (PIL does not premultiply for it — a pure
//! gather): `resize_nearest` keeps straight channels.
//!
//! ### Coordinate mapping + antialiasing (matches `precompute_coeffs`)
//! For output index `xx` along an axis resampled from `in_size` to
//! `out_size`:
//! - `scale = in_size / out_size`
//! - `center = (xx + 0.5) * scale`
//! - `filterscale = max(scale, 1.0)` — the **antialiasing filter-stretch**:
//!   when downscaling (`scale > 1`), the filter support widens by the
//!   scale factor so the kernel averages over the shrinking footprint.
//! - `support = filter_support * filterscale`
//! - window `[floor(center - support + 0.5), floor(center + support + 0.5))`
//!   clamped to `[0, in_size)`
//! - weight for input `x` in the window:
//!   `filter((x - center + 0.5) / filterscale)`, then all weights in the
//!   window normalized so they sum to 1.0.
//!
//! ### Fixed-point accumulation (matches `Resample.c` `clip8`)
//! PIL normalizes the f64 weights to fixed point with
//! `PRECISION_BITS = 22`: `coef_i = round(coef * (1 << 22))` (an `i32`).
//! The per-output accumulator is an `i32` seeded with the rounding bias
//! `1 << (PRECISION_BITS - 1)`, accumulates `pixel * coef_i`, then is
//! finished with an **arithmetic** `>> PRECISION_BITS` (sign-extending,
//! matching C's signed shift) and clamped to `[0, 255]`. The `i32`
//! accumulator does not overflow: the worst-case partial sum for these
//! kernels is `≈ 255 * 1.2 * (1 << 22) ≈ 1.28e9 < i32::MAX ≈ 2.15e9`
//! (the `sum(|coef|)` over each window is `< 1.2` for Keys-cubic a=-0.5
//! and Lanczos a=3; the filterscale spreads coefficients but shrinks each
//! so the bound holds at any scale).
//!
//! ### Nearest
//! PIL's `NEAREST` resize maps output index `o` to input
//! `min(floor((o + 0.5) * in_size / out_size), in_size - 1)` (verified
//! against Pillow). It is a pure pixel gather — no convolution, no
//! coefficient table.
//!
//! ## SIMD
//! The hot loop is the inner per-output-pixel weighted sum over the
//! support window, per channel. RGBA8 is `[u8; 4]` per pixel, so the NEON
//! kernel vectorizes **across the 4 channels**: widen the 4 source bytes
//! to `int32x4`, fused-multiply-accumulate by the (broadcast) `i32`
//! coefficient into an `int32x4` accumulator, then narrow back to 4 `u8`
//! with the same arithmetic shift + clamp. This produces output
//! bit-identical to the scalar path (same `i32` math, same rounding).
//! The coefficient precomputation (cold, once per resize) stays scalar.
//!
//! Per the project SIMD conventions: NEON is gated on
//! `#[cfg(target_arch = "aarch64")]` + a runtime
//! `is_aarch64_feature_detected!("neon")` check, the scalar fallback is
//! ALWAYS compiled, the `#[target_feature(enable = "neon")] unsafe fn`
//! kernels carry numbered `# Safety` clauses, slice-length preconditions
//! are `assert!`ed unconditionally, and the `--cfg mlxrs_force_scalar`
//! escape forces the scalar path even on aarch64. There is NO cargo
//! feature: the dispatch is always-on. (This is self-contained in `vlm`;
//! it can be refactored into a shared `mlxrs::simd` module later.)

use crate::error::{Error, Result, try_with_capacity};

/// Interpolation filter for [`resize_rgba8`], mirroring PIL's resampling
/// filters. The variants line up 1:1 with
/// [`crate::vlm::image::ResizeFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Filter {
  /// Nearest-neighbor pixel gather (no smoothing). PIL `Image.NEAREST`.
  Nearest,
  /// Triangle / linear kernel, support `1.0`. PIL `Image.BILINEAR`.
  Bilinear,
  /// Keys cubic with `a = -0.5`, support `2.0`. PIL `Image.BICUBIC`.
  Bicubic,
  /// Sinc-windowed sinc with `a = 3`, support `3.0`. PIL `Image.LANCZOS`.
  Lanczos3,
}

/// PIL fixed-point precision: `coef_int = round(coef * (1 << 22))`, and
/// the accumulator is finished with `>> 22`. Matches `Resample.c`'s
/// `#define PRECISION_BITS (32 - 8 - 2)`.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Rounding bias added to the fixed-point accumulator before the final
/// shift (`1 << (PRECISION_BITS - 1)`), matching `Resample.c`.
const ROUND_BIAS: i32 = 1 << (PRECISION_BITS - 1);

/// RGBA8 has 4 channels (the only pixel layout this module handles — the
/// caller materializes every source variant to RGBA8 first).
const CHANNELS: usize = 4;

/// Byte ceiling for EVERY allocation in the resize path — the same 512 MiB
/// budget [`crate::vlm::image::MAX_DECODED_IMAGE_BYTES`] caps the
/// RGBA-expanded source and final destination with. The public
/// [`crate::vlm::image::resize`] wrapper guards only those two end buffers;
/// the *internal* scratch this module allocates (the horizontal-pass
/// intermediate, the per-axis coefficient tables, the nearest-resize
/// x-index map) is sized from the SAME untrusted target dimensions and can
/// dwarf both ends — e.g. a `1×131072` source resized to `131072×1` has a
/// 0.5 MiB source and a 0.5 MiB destination but a `131072 * 131072 * 4`
/// ≈ 68 GiB horizontal intermediate. `try_reserve_exact` makes an allocator
/// *refusal* recoverable, but on an overcommitting allocator the reservation
/// succeeds and the subsequent zero-fill faults in all 68 GiB → process
/// death. So every scratch buffer is checked against this ceiling BEFORE its
/// `try_reserve_exact` (see [`checked_buffer_bytes`]).
///
/// Kept in sync with — and equal to — `image::MAX_DECODED_IMAGE_BYTES`
/// (`u64` there; `usize` here because these byte counts are compared
/// against `Vec` capacities). On a 32-bit host `usize` is 32-bit but
/// `512 * 1024 * 1024` still fits, so the `as usize` is lossless.
const MAX_DECODED_IMAGE_BYTES: usize = 512 * 1024 * 1024;

/// Compute `elems * elem_size` as a byte count, rejecting BOTH a `usize`
/// overflow and a product exceeding [`MAX_DECODED_IMAGE_BYTES`]. Every
/// `try_with_capacity` / `try_reserve_exact` in the resize path is preceded
/// by this check, so no resize allocation — source, horizontal
/// intermediate, coefficient table, x-index map, destination — can overflow
/// `usize` or exceed the 512 MiB budget.
///
/// `try_reserve_exact` already turns an *allocator refusal* into a
/// recoverable [`Error::OutOfMemory`], but it does not bound the request:
/// an overcommitting allocator hands back a 68 GiB reservation that only
/// faults (and kills the process) when the caller's zero-fill touches the
/// pages. This ceiling check makes the *request itself* recoverable.
///
/// `what` names the buffer (with its dimensions) for the error message.
///
/// # Errors
/// [`Error::ShapeMismatch`] if `elems * elem_size` overflows `usize` or
/// exceeds [`MAX_DECODED_IMAGE_BYTES`]; the message carries `what` and the
/// offending byte count.
fn checked_buffer_bytes(elems: usize, elem_size: usize, what: &str) -> Result<usize> {
  let bytes = elems
    .checked_mul(elem_size)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize: {what} size overflows usize ({elems} elems * {elem_size} B/elem)"),
    })?;
  if bytes > MAX_DECODED_IMAGE_BYTES {
    return Err(Error::ShapeMismatch {
      message: format!(
        "resize: {what} needs {bytes} bytes, exceeds \
         MAX_DECODED_IMAGE_BYTES={MAX_DECODED_IMAGE_BYTES} \
         ({elems} elems * {elem_size} B/elem)"
      ),
    });
  }
  Ok(bytes)
}

/// Continuous filter support radius (the half-width of the kernel before
/// the antialiasing filterscale stretch).
fn filter_support(f: Filter) -> f64 {
  match f {
    // Nearest has no continuous kernel; never queried (handled separately).
    Filter::Nearest => 0.0,
    Filter::Bilinear => 1.0,
    Filter::Bicubic => 2.0,
    Filter::Lanczos3 => 3.0,
  }
}

/// Evaluate the continuous filter kernel at `x` (already divided by the
/// filterscale by the caller). Each matches PIL's `Resample.c`:
/// - Bilinear: triangle `1 - |x|` on `[-1, 1]`.
/// - Bicubic: Keys cubic with `a = -0.5`.
/// - Lanczos3: `sinc(x) * sinc(x / 3)` on `[-3, 3]`.
fn filter_eval(f: Filter, x: f64) -> f64 {
  match f {
    Filter::Nearest => 0.0,
    Filter::Bilinear => {
      let x = x.abs();
      if x < 1.0 { 1.0 - x } else { 0.0 }
    }
    Filter::Bicubic => {
      // PIL Keys cubic, a = -0.5.
      const A: f64 = -0.5;
      let x = x.abs();
      if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
      } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
      } else {
        0.0
      }
    }
    Filter::Lanczos3 => {
      let x = x.abs();
      if x < 3.0 {
        sinc(x) * sinc(x / 3.0)
      } else {
        0.0
      }
    }
  }
}

/// Normalized sinc, `sin(pi x) / (pi x)`, with `sinc(0) = 1` — matching
/// PIL's `sinc_filter`.
fn sinc(x: f64) -> f64 {
  if x == 0.0 {
    1.0
  } else {
    let px = x * std::f64::consts::PI;
    px.sin() / px
  }
}

/// Precomputed per-output-index convolution coefficients for one axis.
///
/// `bounds[o] = (xmin, n)` gives the input window start and length for
/// output index `o`; `weights[o * ksize .. o * ksize + n]` are the
/// fixed-point `i32` taps for that output (the remaining `ksize - n`
/// slots in the row are zero-padded so every row has a uniform stride —
/// this keeps the convolution inner loop branch-free on row stride).
///
/// All three backing `Vec`s are reserved via `try_reserve_exact`; this
/// type is the "coefficient table" `fast_image_resize` allocated
/// infallibly.
struct Coeffs {
  /// `(xmin, n)` per output index.
  bounds: Vec<(usize, usize)>,
  /// Fixed-point taps, row-major with stride `ksize`.
  weights: Vec<i32>,
  /// Per-output row stride (`max` window length across outputs).
  ksize: usize,
}

/// Precompute the convolution coefficients for resampling one axis from
/// `in_size` to `out_size` with `filter` (PIL `precompute_coeffs` +
/// `normalize_coeffs_8bpc`).
///
/// Every buffer is `try_reserve_exact`-backed; an allocator refusal
/// surfaces as [`Error::OutOfMemory`]. A degenerate `in_size`/`out_size`
/// (zero), a `ksize` overflow, or a coefficient table exceeding
/// [`MAX_DECODED_IMAGE_BYTES`] surfaces as [`Error::ShapeMismatch`].
///
/// The coefficient table is `out_size * ksize` taps. `ksize` is small for
/// a sane resize (`ceil(filter_support * filterscale) * 2 + 1`, clamped to
/// `in_size`), but a `131072`-wide output combined with a stretched
/// downscale support could still size a multi-hundred-MiB table — so the
/// table's byte size, the bounds vector, and the per-row f64 scratch are
/// each capped against [`MAX_DECODED_IMAGE_BYTES`] via
/// [`checked_buffer_bytes`] BEFORE their `try_reserve_exact`.
fn precompute_coeffs(in_size: usize, out_size: usize, filter: Filter) -> Result<Coeffs> {
  // Caller guarantees non-zero, but guard defensively: a zero `out_size`
  // would divide by zero in `scale`, a zero `in_size` makes the window
  // empty.
  if in_size == 0 || out_size == 0 {
    return Err(Error::ShapeMismatch {
      message: format!("precompute_coeffs: in_size={in_size} out_size={out_size} must be non-zero"),
    });
  }
  let scale = in_size as f64 / out_size as f64;
  let filterscale = if scale < 1.0 { 1.0 } else { scale };
  let support = filter_support(filter) * filterscale;
  // `ksize` is the max number of taps any output index can reference:
  // `ceil(support) * 2 + 1`, exactly PIL's `ksize = (int)ceil(support) *
  // 2 + 1`. Bounded by `in_size` (a window can never exceed the input).
  let ksize_unclamped = (support.ceil() as usize)
    .checked_mul(2)
    .and_then(|v| v.checked_add(1))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("precompute_coeffs: ksize overflows for support={support}"),
    })?;
  let ksize = ksize_unclamped.min(in_size.max(1));

  // `bounds` is `out_size` `(usize, usize)` pairs; cap its byte size
  // against the 512 MiB budget before reserving — a `131072`-wide output
  // alone is tiny, but the same guard applies uniformly to every scratch
  // buffer so no resize allocation bypasses the ceiling.
  checked_buffer_bytes(
    out_size,
    std::mem::size_of::<(usize, usize)>(),
    "coefficient bounds table",
  )?;
  let mut bounds: Vec<(usize, usize)> = try_with_capacity(out_size)?;
  // `out_size * ksize` `i32` weights. `checked_mul` rejects a `usize`
  // overflow of the element count; `checked_buffer_bytes` then rejects a
  // table whose byte size exceeds `MAX_DECODED_IMAGE_BYTES` — a
  // `131072`-wide output with a stretched downscale support could
  // otherwise reserve a multi-GiB coefficient table that
  // `try_reserve_exact` cannot bound on an overcommitting allocator.
  let weight_len = out_size
    .checked_mul(ksize)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("precompute_coeffs: out_size*ksize overflows for {out_size}*{ksize}"),
    })?;
  checked_buffer_bytes(
    weight_len,
    std::mem::size_of::<i32>(),
    "coefficient weight table",
  )?;
  let mut weights: Vec<i32> = try_with_capacity(weight_len)?;
  weights.resize(weight_len, 0i32);

  // Scratch for one row of f64 weights before fixed-point conversion.
  // Bounded by `ksize`; capped against the budget before reserving (a
  // stretched downscale support can make `ksize` large).
  checked_buffer_bytes(ksize, std::mem::size_of::<f64>(), "coefficient row scratch")?;
  let mut row: Vec<f64> = try_with_capacity(ksize)?;

  let inv_filterscale = 1.0 / filterscale;
  for xx in 0..out_size {
    let center = (xx as f64 + 0.5) * scale;
    // Window `[xmin, xmax)` clamped to `[0, in_size)`. PIL adds 0.5 and
    // truncates toward zero; `center - support` is >= 0 here only after
    // the clamp, and the `+ 0.5` then `as usize`/`as i64` truncation
    // matches C's `(int)`.
    let xmin = {
      let v = (center - support + 0.5).floor();
      if v < 0.0 { 0 } else { v as usize }
    };
    let xmax = {
      let v = (center + support + 0.5).floor();
      let v = if v < 0.0 { 0usize } else { v as usize };
      v.min(in_size)
    };
    let n = xmax.saturating_sub(xmin);
    // Accumulate raw weights, then normalize to sum 1.0 (PIL divides
    // each tap by the window sum).
    row.clear();
    let mut wsum = 0.0f64;
    for i in 0..n {
      let w = filter_eval(
        filter,
        (xmin as f64 + i as f64 - center + 0.5) * inv_filterscale,
      );
      row.push(w);
      wsum += w;
    }
    let base = xx * ksize;
    if wsum != 0.0 {
      let inv = 1.0 / wsum;
      for (i, &w) in row.iter().enumerate() {
        // Fixed-point: round(coef * (1 << PRECISION_BITS)).
        let scaled = (w * inv) * f64::from(1i32 << PRECISION_BITS);
        weights[base + i] = scaled.round() as i32;
      }
    }
    // n is bounded by ksize by construction (window <= ceil(support)*2+1
    // and clamped to in_size). Assert to make the convolution's slice
    // access provably in-bounds.
    debug_assert!(
      n <= ksize,
      "precompute_coeffs: window n={n} exceeds ksize={ksize}"
    );
    bounds.push((xmin, n));
  }
  Ok(Coeffs {
    bounds,
    weights,
    ksize,
  })
}

/// Clamp a finished fixed-point accumulator to `u8` exactly as PIL's
/// `clip8`: arithmetic `>> PRECISION_BITS` (sign-extending) then clamp to
/// `[0, 255]`.
#[inline]
fn clip8(acc: i32) -> u8 {
  // Rust `>>` on `i32` is arithmetic (sign-preserving), matching C's
  // signed right shift used by `clip8`.
  let v = acc >> PRECISION_BITS;
  if v < 0 {
    0
  } else if v > 255 {
    255
  } else {
    v as u8
  }
}

/// Resize an RGBA8 image from `(src_w, src_h)` to `(dst_w, dst_h)` using
/// `filter`. `src` MUST be exactly `src_w * src_h * 4` bytes; the returned
/// `Vec<u8>` is exactly `dst_w * dst_h * 4` bytes (row-major RGBA8).
///
/// EVERY buffer (coefficient tables for both axes, the horizontal-pass
/// intermediate, the output) is `try_reserve_exact`-backed; an allocator
/// refusal surfaces as [`Error::OutOfMemory`], never a process abort. In
/// addition, every buffer is capped against [`MAX_DECODED_IMAGE_BYTES`]
/// (512 MiB) via [`checked_buffer_bytes`] BEFORE its reservation — the
/// public [`crate::vlm::image::resize`] wrapper only bounds the
/// RGBA-source and the destination, but the horizontal intermediate
/// (`src_h * dst_w * 4`) and the coefficient tables are sized from the
/// SAME untrusted target and can dwarf both ends (a `1×131072` →
/// `131072×1` resize has 0.5 MiB ends but a ~68 GiB intermediate). Capping
/// the request itself — not just relying on `try_reserve_exact` — closes
/// the overcommit zero-fill abort. So `resize_rgba8` is safe to call
/// directly, not only through the public wrapper.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if any dimension is `0`, if a byte/element
///   product overflows `usize`, if `src.len() != src_w * src_h * 4`, or if
///   ANY buffer in the resize path (source copy, coefficient tables,
///   horizontal intermediate, destination) would exceed
///   [`MAX_DECODED_IMAGE_BYTES`].
/// - [`Error::OutOfMemory`] if any `try_reserve_exact` fails.
///
/// # Panics
/// Does not panic on valid input: the only `assert!`s are slice-length
/// preconditions inside the SIMD/scalar kernels, which the dimension math
/// in this function makes structurally true.
pub(crate) fn resize_rgba8(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  dst_w: usize,
  dst_h: usize,
  filter: Filter,
) -> Result<Vec<u8>> {
  if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "resize_rgba8: dimensions must be non-zero, got src {src_w}x{src_h} dst {dst_w}x{dst_h}"
      ),
    });
  }
  let src_len = src_w
    .checked_mul(src_h)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: src_w*src_h*4 overflows usize for {src_w}x{src_h}"),
    })?;
  if src.len() != src_len {
    return Err(Error::ShapeMismatch {
      message: format!(
        "resize_rgba8: src buffer is {} bytes, expected src_w*src_h*4={src_len} for {src_w}x{src_h}",
        src.len()
      ),
    });
  }
  // Cap the source against the 512 MiB budget too: `src` is borrowed (not
  // allocated here), but the premultiplied copy below is `src.len()` bytes
  // — and a direct caller (not the public `resize` wrapper) has no other
  // guard. `src_len` already cleared the overflow check above.
  checked_buffer_bytes(src_len, 1, &format!("RGBA8 source ({src_w}x{src_h})"))?;
  let dst_len = dst_w
    .checked_mul(dst_h)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: dst_w*dst_h*4 overflows usize for {dst_w}x{dst_h}"),
    })?;
  // Cap the destination against the 512 MiB budget. The public `resize`
  // wrapper already bounds it, but `resize_rgba8` is `pub(crate)` and may
  // be called directly — every entry path is covered here.
  checked_buffer_bytes(dst_len, 1, &format!("destination ({dst_w}x{dst_h} RGBA8)"))?;

  if filter == Filter::Nearest {
    // PIL exempts `NEAREST` from premultiplication: it is a pure pixel
    // gather, so straight RGBA channels are already byte-exact (see the
    // `premultiply_rgba` doc + `Image.resize`'s `resample != NEAREST`
    // guard).
    return resize_nearest(src, src_w, src_h, dst_w, dst_h, dst_len);
  }

  // --- Premultiplied-alpha staging (PIL parity) ---
  // PIL's `Image.resize` converts RGBA -> premultiplied `RGBa` BEFORE any
  // non-NEAREST resample and converts back after (`Image.py`:
  // `if self.mode in ["LA", "RGBA"] and resample != NEAREST: ...
  // convert("RGBa") ... resize ... convert(self.mode)`). Straight-channel
  // convolution is NOT byte-exact for non-opaque alpha — it bleeds the
  // colour of fully-transparent pixels into their neighbours. We mirror
  // that exact path: premultiply the colour channels into an owned
  // fallible copy, run the existing separable convolution over the
  // premultiplied buffer, then unpremultiply the destination in place.
  // For an all-opaque (`A == 255`) image both passes are the identity
  // (`MULDIV255(c, 255) == c`, and unpremultiply special-cases
  // `alpha == 255`), so opaque inputs are bit-identical to the prior
  // behaviour. (`resize_rgba8` only ever sees RGBA8 — `vlm::image::resize`
  // projects every source variant, including `LumaA8`, to RGBA8 first —
  // so the single RGBA premultiply path also covers PIL's `LA -> La`.)
  let src_pm = premultiply_rgba(src)?;

  // --- Separable convolution ---
  // Horizontal pass: (src_h rows) x (dst_w cols) intermediate, RGBA8.
  // Vertical pass: (dst_h rows) x (dst_w cols) output.
  let hcoeffs = precompute_coeffs(src_w, dst_w, filter)?;
  let vcoeffs = precompute_coeffs(src_h, dst_h, filter)?;

  // Intermediate buffer: src_h * dst_w * 4 bytes, fallible. (PIL emits an
  // 8-bit clamped image between the two passes; the vertical pass reads
  // it back.) CRITICAL: this intermediate's dimensions are `src_h` (input)
  // by `dst_w` (untrusted target) — it is NOT bounded by either the
  // RGBA-source cap (`src_w*src_h*4`) or the destination cap
  // (`dst_w*dst_h*4`) the public `resize` wrapper enforces. A `1×131072`
  // source resized to `131072×1` has a 0.5 MiB source, a 0.5 MiB
  // destination, but a `131072 * 131072 * 4` ≈ 68 GiB intermediate. So
  // cap it explicitly against `MAX_DECODED_IMAGE_BYTES` (overflow OR
  // > 512 MiB -> ShapeMismatch) BEFORE the `try_reserve_exact` + zero-fill
  // — `try_reserve_exact` alone cannot stop an overcommitting allocator
  // from handing back 68 GiB that the `resize`/zero-fill then faults in.
  let inter_len = src_h
    .checked_mul(dst_w)
    .and_then(|v| v.checked_mul(CHANNELS))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("resize_rgba8: src_h*dst_w*4 overflows usize for {src_h}x{dst_w}"),
    })?;
  checked_buffer_bytes(
    inter_len,
    1,
    &format!("horizontal-pass intermediate ({src_h}x{dst_w} RGBA8)"),
  )?;
  let mut inter: Vec<u8> = try_with_capacity(inter_len)?;
  inter.resize(inter_len, 0u8);

  // Output buffer, fallible.
  let mut dst: Vec<u8> = try_with_capacity(dst_len)?;
  dst.resize(dst_len, 0u8);

  // Horizontal pass: for each src row, convolve along x into `inter`.
  // Operates on the PREMULTIPLIED source (`src_pm`).
  convolve_axis(&src_pm, src_w, src_h, &mut inter, dst_w, &hcoeffs);
  // Vertical pass: convolve `inter` along y into `dst`. We transpose the
  // access by treating columns: for each output row `oy`, gather input
  // rows `[ymin, ymin+n)` from `inter`. To reuse `convolve_axis` (which
  // convolves along the contiguous x-axis), the vertical pass is a
  // separate routine because its taps stride by a full row.
  convolve_vertical(&inter, dst_w, src_h, &mut dst, dst_h, &vcoeffs);

  // Convert the premultiplied `dst` back to straight RGBA8 in place (PIL's
  // post-resize `convert(self.mode)`).
  unpremultiply_rgba(&mut dst);

  Ok(dst)
}

/// PIL fixed-point `c * a / 255`, mirroring `libImaging`'s `MULDIV255`
/// macro exactly: `tmp = c * a + 128; ((tmp >> 8) + tmp) >> 8`. The `+128`
/// is PIL's rounding bias and the double-shift is its `/255`
/// approximation (`SHIFTFORDIV255`). Bit-exact with Pillow's premultiply.
#[inline]
fn muldiv255(c: u8, a: u8) -> u8 {
  // `c, a <= 255`, so `c * a + 128 <= 65153` — fits `u32` with room to
  // spare; the result is provably `<= 255`.
  let tmp = u32::from(c) * u32::from(a) + 128;
  (((tmp >> 8) + tmp) >> 8) as u8
}

/// Premultiply an RGBA8 buffer (PIL `rgbA2rgba` — the `RGBA -> RGBa`
/// mode conversion `Image.resize` applies before a non-NEAREST resample).
/// Each colour channel becomes `MULDIV255(c, A)`; alpha is unchanged. The
/// premultiplied buffer is an owned fallible copy (`src` is borrowed and
/// must stay intact); allocator refusal surfaces as
/// [`Error::OutOfMemory`].
///
/// `src.len()` must be a multiple of [`CHANNELS`] (guaranteed by
/// [`resize_rgba8`]'s `src.len() == src_w * src_h * 4` check).
fn premultiply_rgba(src: &[u8]) -> Result<Vec<u8>> {
  let mut out: Vec<u8> = try_with_capacity(src.len())?;
  for px in src.chunks_exact(CHANNELS) {
    let a = px[3];
    // PIL premultiplies the colour channels only; alpha passes through.
    out.push(muldiv255(px[0], a));
    out.push(muldiv255(px[1], a));
    out.push(muldiv255(px[2], a));
    out.push(a);
  }
  // `chunks_exact` drops a partial trailing chunk; the caller guarantees
  // `src.len()` is a whole number of RGBA pixels, so `out.len()` equals
  // `src.len()`. Assert it so a future caller violating that contract
  // fails loudly rather than silently truncating.
  debug_assert_eq!(
    out.len(),
    src.len(),
    "premultiply_rgba: src length must be a multiple of CHANNELS"
  );
  Ok(out)
}

/// Unpremultiply an RGBA8 buffer in place (PIL `rgba2rgbA` — the
/// `RGBa -> RGBA` conversion `Image.resize` applies after the resample).
/// Mirrors `libImaging` exactly: when `alpha` is `255` or `0` the colour
/// channels pass through unchanged, otherwise each is
/// `CLIP8((255 * c) / alpha)` (truncating integer division, clamped to
/// `[0, 255]`). Alpha is unchanged. No allocation — operates on the
/// destination buffer the convolution already produced.
///
/// The `alpha == 0` passthrough matches PIL: after premultiplication a
/// zero-alpha pixel already has colour channels `0` (`MULDIV255(c, 0)
/// == 0`), and the convolution of all-zero contributors keeps them `0`,
/// so the recovered straight colour is `0` regardless — PIL does not
/// special-case it to anything else.
///
/// `buf.len()` must be a multiple of [`CHANNELS`].
fn unpremultiply_rgba(buf: &mut [u8]) {
  for px in buf.chunks_exact_mut(CHANNELS) {
    let a = px[3];
    if a == 0 || a == 255 {
      // PIL passthrough: opaque needs no division, and a zero-alpha
      // pixel's premultiplied colour is already 0.
      continue;
    }
    // `CLIP8((255 * c) / a)`: `255 * c <= 65025` fits `u32`; integer
    // division truncates (matches C). `a` is in `1..=254` here, so the
    // quotient can exceed 255 (a premultiplied colour > alpha, possible
    // after convolution rounding) — `CLIP8` clamps it.
    let a32 = u32::from(a);
    px[0] = clip8_div(u32::from(px[0]), a32);
    px[1] = clip8_div(u32::from(px[1]), a32);
    px[2] = clip8_div(u32::from(px[2]), a32);
    // px[3] (alpha) unchanged.
  }
}

/// PIL `CLIP8((255 * c) / a)` for unpremultiply. `a` must be non-zero
/// (the caller special-cases `a == 0`). Truncating integer division then
/// clamp to `[0, 255]`.
#[inline]
fn clip8_div(c: u32, a: u32) -> u8 {
  let v = (255 * c) / a;
  if v > 255 { 255 } else { v as u8 }
}

/// Nearest-neighbor resize (pure pixel gather, PIL `Image.NEAREST`).
/// Output index `o` maps to input `min(floor((o+0.5)*in/out), in-1)`.
///
/// Both the per-column x-index map (`dst_w` `usize`s) and the destination
/// (`dst_len` bytes) are capped against [`MAX_DECODED_IMAGE_BYTES`] via
/// [`checked_buffer_bytes`] before their `try_reserve_exact`, so this
/// covers a direct caller as well as the dispatch from [`resize_rgba8`].
fn resize_nearest(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  dst_w: usize,
  dst_h: usize,
  dst_len: usize,
) -> Result<Vec<u8>> {
  // Precompute per-output-column source x indices. `dst_w` is an untrusted
  // target dimension; cap the x-index map's byte size against the 512 MiB
  // budget before reserving.
  checked_buffer_bytes(
    dst_w,
    std::mem::size_of::<usize>(),
    &format!("nearest x-index map ({dst_w} columns)"),
  )?;
  let mut xmap: Vec<usize> = try_with_capacity(dst_w)?;
  for ox in 0..dst_w {
    let sx = ((ox as f64 + 0.5) * src_w as f64 / dst_w as f64).floor() as usize;
    xmap.push(sx.min(src_w - 1));
  }
  // Cap the destination too — `resize_rgba8` already caps `dst_len` before
  // the dispatch, but a direct caller of `resize_nearest` has no other
  // guard.
  checked_buffer_bytes(
    dst_len,
    1,
    &format!("nearest destination ({dst_len} bytes)"),
  )?;
  let mut dst: Vec<u8> = try_with_capacity(dst_len)?;
  dst.resize(dst_len, 0u8);
  for oy in 0..dst_h {
    let sy = (((oy as f64 + 0.5) * src_h as f64 / dst_h as f64).floor() as usize).min(src_h - 1);
    let src_row = &src[sy * src_w * CHANNELS..(sy + 1) * src_w * CHANNELS];
    let dst_row = &mut dst[oy * dst_w * CHANNELS..(oy + 1) * dst_w * CHANNELS];
    for ox in 0..dst_w {
      let sx = xmap[ox];
      dst_row[ox * CHANNELS..ox * CHANNELS + CHANNELS]
        .copy_from_slice(&src_row[sx * CHANNELS..sx * CHANNELS + CHANNELS]);
    }
  }
  Ok(dst)
}

/// Horizontal convolution: for each of `rows` source rows, produce
/// `out_w` output pixels into `out` (RGBA8, `rows * out_w * 4` bytes).
/// Dispatches to the NEON kernel on aarch64 (unless `mlxrs_force_scalar`),
/// else the scalar kernel.
fn convolve_axis(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  // Slice-length preconditions (unconditional assert per SIMD conventions):
  // both kernels rely on these to keep their indexing in-bounds.
  assert_eq!(src.len(), src_w * rows * CHANNELS, "convolve_axis: src len");
  assert_eq!(out.len(), out_w * rows * CHANNELS, "convolve_axis: out len");
  assert_eq!(coeffs.bounds.len(), out_w, "convolve_axis: bounds len");

  #[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
  {
    if std::arch::is_aarch64_feature_detected!("neon") {
      // SAFETY: the `neon` target feature is confirmed available by the
      // runtime `is_aarch64_feature_detected!` check immediately above;
      // see `convolve_axis_neon`'s `# Safety` for the full contract.
      unsafe {
        convolve_axis_neon(src, src_w, rows, out, out_w, coeffs);
      }
      return;
    }
  }
  convolve_axis_scalar(src, src_w, rows, out, out_w, coeffs);
}

/// Vertical convolution: read the `src_h x out_w` intermediate `inter`
/// and produce `out_h` output rows into `out` (RGBA8). Taps stride by a
/// full intermediate row.
fn convolve_vertical(
  inter: &[u8],
  out_w: usize,
  src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  assert_eq!(
    inter.len(),
    out_w * src_h * CHANNELS,
    "convolve_vertical: inter len"
  );
  assert_eq!(
    out.len(),
    out_w * out_h * CHANNELS,
    "convolve_vertical: out len"
  );
  assert_eq!(coeffs.bounds.len(), out_h, "convolve_vertical: bounds len");

  #[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
  {
    if std::arch::is_aarch64_feature_detected!("neon") {
      // SAFETY: `neon` confirmed by the runtime check above; see
      // `convolve_vertical_neon`'s `# Safety`.
      unsafe {
        convolve_vertical_neon(inter, out_w, src_h, out, out_h, coeffs);
      }
      return;
    }
  }
  convolve_vertical_scalar(inter, out_w, src_h, out, out_h, coeffs);
}

/// Scalar horizontal convolution (always compiled). Bit-exact with PIL.
fn convolve_axis_scalar(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  let ksize = coeffs.ksize;
  for y in 0..rows {
    let src_row = &src[y * src_w * CHANNELS..(y + 1) * src_w * CHANNELS];
    let out_row = &mut out[y * out_w * CHANNELS..(y + 1) * out_w * CHANNELS];
    for ox in 0..out_w {
      let (xmin, n) = coeffs.bounds[ox];
      let taps = &coeffs.weights[ox * ksize..ox * ksize + n];
      let mut acc = [ROUND_BIAS; CHANNELS];
      for (i, &w) in taps.iter().enumerate() {
        let px = &src_row[(xmin + i) * CHANNELS..(xmin + i) * CHANNELS + CHANNELS];
        acc[0] += i32::from(px[0]) * w;
        acc[1] += i32::from(px[1]) * w;
        acc[2] += i32::from(px[2]) * w;
        acc[3] += i32::from(px[3]) * w;
      }
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      o[0] = clip8(acc[0]);
      o[1] = clip8(acc[1]);
      o[2] = clip8(acc[2]);
      o[3] = clip8(acc[3]);
    }
  }
}

/// Scalar vertical convolution (always compiled). Bit-exact with PIL.
fn convolve_vertical_scalar(
  inter: &[u8],
  out_w: usize,
  _src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  let ksize = coeffs.ksize;
  let row_stride = out_w * CHANNELS;
  for oy in 0..out_h {
    let (ymin, n) = coeffs.bounds[oy];
    let taps = &coeffs.weights[oy * ksize..oy * ksize + n];
    let out_row = &mut out[oy * row_stride..(oy + 1) * row_stride];
    for ox in 0..out_w {
      let mut acc = [ROUND_BIAS; CHANNELS];
      for (i, &w) in taps.iter().enumerate() {
        let base = (ymin + i) * row_stride + ox * CHANNELS;
        let px = &inter[base..base + CHANNELS];
        acc[0] += i32::from(px[0]) * w;
        acc[1] += i32::from(px[1]) * w;
        acc[2] += i32::from(px[2]) * w;
        acc[3] += i32::from(px[3]) * w;
      }
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      o[0] = clip8(acc[0]);
      o[1] = clip8(acc[1]);
      o[2] = clip8(acc[2]);
      o[3] = clip8(acc[3]);
    }
  }
}

/// NEON horizontal convolution. Vectorizes the per-output weighted sum
/// across the 4 RGBA channels: widen the 4 source bytes to `int32x4`,
/// multiply-accumulate by the broadcast `i32` coefficient, then narrow +
/// shift + clamp back to 4 `u8`. Output is bit-identical to
/// [`convolve_axis_scalar`] (identical `i32` arithmetic + rounding).
///
/// # Safety
/// 1. The `neon` target feature must be available at runtime. The sole
///    caller ([`convolve_axis`]) gates this on
///    `is_aarch64_feature_detected!("neon")`, so the `vld*`/`vmlaq`/etc.
///    intrinsics are legal on the executing CPU.
/// 2. `src.len() == src_w * rows * 4`, `out.len() == out_w * rows * 4`,
///    and `coeffs.bounds.len() == out_w` — all asserted unconditionally
///    by the caller before dispatch. Combined with the
///    [`precompute_coeffs`] invariant `xmin + n <= src_w` (window clamped
///    to the input), every byte slice accessed below is in-bounds.
/// 3. All loads/stores are 4-byte (one RGBA8 pixel) and operate on the
///    `&[u8]`/`&mut [u8]` slices directly (no raw pointer aliasing beyond
///    the borrow the references already grant).
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn convolve_axis_neon(
  src: &[u8],
  src_w: usize,
  rows: usize,
  out: &mut [u8],
  out_w: usize,
  coeffs: &Coeffs,
) {
  use std::arch::aarch64::*;
  let ksize = coeffs.ksize;
  for y in 0..rows {
    let src_row = &src[y * src_w * CHANNELS..(y + 1) * src_w * CHANNELS];
    let out_row = &mut out[y * out_w * CHANNELS..(y + 1) * out_w * CHANNELS];
    for ox in 0..out_w {
      let (xmin, n) = coeffs.bounds[ox];
      let taps = &coeffs.weights[ox * ksize..ox * ksize + n];
      // Seed all four lanes with the rounding bias. Value-only NEON
      // intrinsics need no `unsafe` block inside a `#[target_feature]`
      // fn — the feature gate discharges their safety; only the pointer
      // load/store below carry an `unsafe {}` (with a SAFETY note).
      let mut acc = vdupq_n_s32(ROUND_BIAS);
      for (i, &w) in taps.iter().enumerate() {
        let off = (xmin + i) * CHANNELS;
        // `off + 4 <= src_row.len()` by the window invariant
        // (`xmin + n <= src_w`, asserted via Safety clause 2).
        let px4 = [
          src_row[off],
          src_row[off + 1],
          src_row[off + 2],
          src_row[off + 3],
        ];
        // SAFETY: clauses 1+3 — `neon` confirmed by the dispatch gate;
        // `neon_load_rgba` zero-extends 4 RGBA bytes into a `uint8x8_t`
        // and only reads its own 8-byte stack array.
        let v8 = unsafe { neon_load_rgba(px4) };
        let v16 = vmovl_u8(v8); // u8x8 -> u16x8
        let v16lo = vget_low_u16(v16); // first 4 u16 (R,G,B,A)
        let v32 = vreinterpretq_s32_u32(vmovl_u16(v16lo)); // u16x4 -> s32x4
        let wv = vdupq_n_s32(w);
        acc = vmlaq_s32(acc, v32, wv);
      }
      // Arithmetic shift right by PRECISION_BITS (matches scalar `>>`),
      // then narrow with unsigned saturation to u8 (clamps to [0,255],
      // matching `clip8`): `vqmovun_s32` maps negatives to 0, the
      // subsequent `vqmovn_u16` saturates the > 255 case.
      let shifted = vshrq_n_s32::<{ PRECISION_BITS as i32 }>(acc);
      let u16x4 = vqmovun_s32(shifted); // s32x4 -> u16x4 (sat, >=0)
      let u16x8 = vcombine_u16(u16x4, vdup_n_u16(0));
      let u8x8 = vqmovn_u16(u16x8); // u16x8 -> u8x8 (sat to 255)
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      // SAFETY: clauses 1+3 — `neon` confirmed by the dispatch gate;
      // `neon_store_rgba` writes only its own 8-byte stack array and `o`
      // is exactly `CHANNELS` bytes (asserted inside the helper).
      unsafe { neon_store_rgba(u8x8, o) };
    }
  }
}

/// Load 4 RGBA bytes into the low half of a `uint8x8_t` (high 4 lanes
/// zero). Isolates the only pointer-based NEON `unsafe` in the kernels.
///
/// # Safety
/// 1. `neon` available at runtime (the kernels are reached only after the
///    dispatch gate's `is_aarch64_feature_detected!("neon")`).
/// 2. Reads exactly 8 bytes from an 8-byte stack array — fully in-bounds.
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn neon_load_rgba(px4: [u8; CHANNELS]) -> std::arch::aarch64::uint8x8_t {
  use std::arch::aarch64::*;
  // Widen to 8 bytes (low 4 = pixel, high 4 = 0) so the single 8-byte
  // `vld1_u8` reads only initialized stack memory.
  let buf = [px4[0], px4[1], px4[2], px4[3], 0, 0, 0, 0];
  // SAFETY: clauses 1+2 — `vld1_u8` reads 8 bytes from `buf` (`[u8; 8]`),
  // all initialized and in-bounds; `neon` confirmed by the dispatch gate.
  unsafe { vld1_u8(buf.as_ptr()) }
}

/// Store the low 4 lanes of a `uint8x8_t` into a 4-byte RGBA output slice.
///
/// # Safety
/// 1. `neon` available at runtime (see [`neon_load_rgba`]).
/// 2. `out.len() == 4` (one RGBA pixel) — the kernels slice exactly
///    `CHANNELS` bytes.
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn neon_store_rgba(v: std::arch::aarch64::uint8x8_t, out: &mut [u8]) {
  use std::arch::aarch64::*;
  assert_eq!(
    out.len(),
    CHANNELS,
    "neon_store_rgba: out must be one RGBA pixel"
  );
  let mut tmp = [0u8; 8];
  // SAFETY: clauses 1+2 — `vst1_u8` writes 8 bytes into `tmp` (`[u8; 8]`),
  // in-bounds; `neon` confirmed by the dispatch gate. Only the low 4
  // (the pixel) are copied out.
  unsafe { vst1_u8(tmp.as_mut_ptr(), v) };
  out.copy_from_slice(&tmp[..CHANNELS]);
}

/// NEON vertical convolution. Same per-channel vectorization as
/// [`convolve_axis_neon`] but taps stride by a full intermediate row.
/// Bit-identical to [`convolve_vertical_scalar`].
///
/// # Safety
/// 1. `neon` available at runtime — gated by the caller
///    ([`convolve_vertical`]) on `is_aarch64_feature_detected!("neon")`.
/// 2. `inter.len() == out_w * src_h * 4`, `out.len() == out_w * out_h *
///    4`, `coeffs.bounds.len() == out_h` — asserted by the caller.
///    Combined with `ymin + n <= src_h` from [`precompute_coeffs`], every
///    `inter[base..base+4]` access is in-bounds.
/// 3. Same 4-byte load/store contract as [`convolve_axis_neon`].
#[cfg(all(target_arch = "aarch64", not(mlxrs_force_scalar)))]
#[target_feature(enable = "neon")]
unsafe fn convolve_vertical_neon(
  inter: &[u8],
  out_w: usize,
  _src_h: usize,
  out: &mut [u8],
  out_h: usize,
  coeffs: &Coeffs,
) {
  use std::arch::aarch64::*;
  let ksize = coeffs.ksize;
  let row_stride = out_w * CHANNELS;
  for oy in 0..out_h {
    let (ymin, n) = coeffs.bounds[oy];
    let taps = &coeffs.weights[oy * ksize..oy * ksize + n];
    let out_row = &mut out[oy * row_stride..(oy + 1) * row_stride];
    for ox in 0..out_w {
      let mut acc = vdupq_n_s32(ROUND_BIAS);
      for (i, &w) in taps.iter().enumerate() {
        let base = (ymin + i) * row_stride + ox * CHANNELS;
        // `base + 4 <= inter.len()` by the window invariant
        // (`ymin + n <= src_h`, Safety clause 2).
        let px4 = [
          inter[base],
          inter[base + 1],
          inter[base + 2],
          inter[base + 3],
        ];
        // SAFETY: clauses 1+3 — see `neon_load_rgba`'s contract.
        let v8 = unsafe { neon_load_rgba(px4) };
        let v16 = vmovl_u8(v8);
        let v16lo = vget_low_u16(v16);
        let v32 = vreinterpretq_s32_u32(vmovl_u16(v16lo));
        let wv = vdupq_n_s32(w);
        acc = vmlaq_s32(acc, v32, wv);
      }
      let shifted = vshrq_n_s32::<{ PRECISION_BITS as i32 }>(acc);
      let u16x4 = vqmovun_s32(shifted);
      let u16x8 = vcombine_u16(u16x4, vdup_n_u16(0));
      let u8x8 = vqmovn_u16(u16x8);
      let o = &mut out_row[ox * CHANNELS..ox * CHANNELS + CHANNELS];
      // SAFETY: clauses 1+3 — see `neon_store_rgba`'s contract.
      unsafe { neon_store_rgba(u8x8, o) };
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Force-scalar variant of [`resize_rgba8`] (calls the `*_scalar`
  /// kernels directly, bypassing the NEON dispatch). Used only by the
  /// differential test to compare against the dispatched path. Mirrors
  /// [`resize_rgba8`]'s premultiplied-alpha staging exactly (premultiply
  /// the source, convolve, unpremultiply) so the differential test stays
  /// a faithful NEON-vs-scalar comparison of the WHOLE resize.
  fn resize_rgba8_scalar(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    filter: Filter,
  ) -> Vec<u8> {
    if filter == Filter::Nearest {
      let dst_len = dst_w * dst_h * CHANNELS;
      return resize_nearest(src, src_w, src_h, dst_w, dst_h, dst_len).unwrap();
    }
    let src_pm = premultiply_rgba(src).unwrap();
    let hc = precompute_coeffs(src_w, dst_w, filter).unwrap();
    let vc = precompute_coeffs(src_h, dst_h, filter).unwrap();
    let mut inter = vec![0u8; src_h * dst_w * CHANNELS];
    let mut dst = vec![0u8; dst_w * dst_h * CHANNELS];
    convolve_axis_scalar(&src_pm, src_w, src_h, &mut inter, dst_w, &hc);
    convolve_vertical_scalar(&inter, dst_w, src_h, &mut dst, dst_h, &vc);
    unpremultiply_rgba(&mut dst);
    dst
  }

  /// Deterministic pseudo-random RGBA8 source (LCG — no rand dependency).
  fn make_src(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_add(1);
    let mut v = Vec::with_capacity(w * h * CHANNELS);
    for _ in 0..w * h * CHANNELS {
      s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      v.push((s >> 24) as u8);
    }
    v
  }

  #[test]
  fn neon_matches_scalar_across_sizes_and_filters() {
    // Differential: the dispatched path (NEON on aarch64, scalar
    // elsewhere) must produce output BIT-IDENTICAL to the force-scalar
    // path, across sizes straddling the 4-channel vector boundary and
    // both up/down scaling. On a non-aarch64 host this is a scalar-vs-
    // scalar identity (still a useful determinism check); on aarch64 it
    // is the real NEON-vs-scalar guarantee.
    let filters = [
      Filter::Bilinear,
      Filter::Bicubic,
      Filter::Lanczos3,
      Filter::Nearest,
    ];
    // Sizes chosen to straddle odd/even widths + up/down + 1-px axes.
    let cases = [
      (4usize, 4usize, 2usize, 2usize),
      (3, 5, 7, 2),
      (5, 3, 2, 8),
      (8, 6, 4, 3),
      (2, 2, 9, 9),
      (5, 1, 2, 1),
      (1, 5, 1, 2),
      (7, 7, 7, 7),
      (16, 9, 5, 11),
    ];
    for (i, &(sw, sh, dw, dh)) in cases.iter().enumerate() {
      let src = make_src(sw, sh, i as u32 * 7 + 1);
      for &f in &filters {
        let dispatched = resize_rgba8(&src, sw, sh, dw, dh, f).unwrap();
        let scalar = resize_rgba8_scalar(&src, sw, sh, dw, dh, f);
        assert_eq!(
          dispatched, scalar,
          "NEON-vs-scalar differential mismatch for {f:?} {sw}x{sh}->{dw}x{dh}"
        );
      }
    }
  }

  #[test]
  fn rejects_zero_dimensions() {
    let src = [0u8; 4]; // 1x1 RGBA
    for (sw, sh, dw, dh) in [(0, 1, 2, 2), (1, 0, 2, 2), (1, 1, 0, 2), (1, 1, 2, 0)] {
      let r = resize_rgba8(
        &src[..sw.max(1) * sh.max(1) * CHANNELS],
        sw,
        sh,
        dw,
        dh,
        Filter::Bilinear,
      );
      assert!(
        matches!(r, Err(Error::ShapeMismatch { .. })),
        "zero dim {sw}x{sh}->{dw}x{dh} must be ShapeMismatch, got {r:?}"
      );
    }
  }

  #[test]
  fn rejects_src_buffer_length_mismatch() {
    // src buffer too short for the claimed dims -> ShapeMismatch (not a
    // panic / OOB read).
    let src = [0u8; 4]; // claims 4 bytes but we say 2x2 (needs 16)
    let r = resize_rgba8(&src, 2, 2, 1, 1, Filter::Bilinear);
    assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "got {r:?}");
  }

  #[test]
  fn rejects_overflowing_dst_product() {
    // dst_w * dst_h * 4 overflows usize -> ShapeMismatch (the structural
    // try_reserve guard's overflow branch). Use usize::MAX-ish dims.
    let src = [0u8; 4];
    let big = usize::MAX / 2 + 1;
    let r = resize_rgba8(&src, 1, 1, big, big, Filter::Bilinear);
    assert!(matches!(r, Err(Error::ShapeMismatch { .. })), "got {r:?}");
  }

  #[test]
  fn rejects_skinny_to_wide_oversized_intermediate() {
    // Codex adversarial case: a `1x131072` source resized to `131072x1`.
    // The RGBA source is `1*131072*4` = 512 KiB (under the 512 MiB cap)
    // and the destination is `131072*1*4` = 512 KiB (under the cap), but
    // the horizontal-pass intermediate is `src_h * dst_w * 4`
    // = `131072 * 131072 * 4` ≈ 68 GiB. `checked_buffer_bytes` must
    // reject the intermediate BEFORE any `try_reserve_exact` / zero-fill,
    // so this returns a recoverable `Err` — no 68 GiB allocation, no
    // overcommit zero-fill abort. (A convolution filter, not NEAREST:
    // NEAREST has no intermediate and a `1x131072`->`131072x1` NEAREST is
    // a legitimate small resize.)
    let src = vec![0u8; 131072 * CHANNELS];
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      let r = resize_rgba8(&src, 1, 131072, 131072, 1, f);
      assert!(
        matches!(r, Err(Error::ShapeMismatch { .. })),
        "{f:?}: 1x131072->131072x1 must reject the ~68 GiB intermediate, got {r:?}"
      );
      // The message must name the intermediate + its byte count so the
      // failure is diagnosable.
      if let Err(Error::ShapeMismatch { message }) = &r {
        assert!(
          message.contains("intermediate"),
          "{f:?}: error should name the intermediate buffer, got: {message}"
        );
      }
    }
  }

  #[test]
  fn wide_to_skinny_does_not_abort() {
    // The reverse orientation: a `131072x1` source resized to `1x131072`.
    // Unlike skinny->wide, this orientation has NO oversized buffer — the
    // intermediate is `src_h * dst_w * 4` = `1 * 1 * 4` = 4 bytes, the
    // destination is `1 * 131072 * 4` = 512 KiB, and both coefficient
    // tables are small (the `131072`-tall output axis upscales from
    // `in_size=1`, so `ksize=1` and the table is `131072 * 4` = 512 KiB).
    // So a correct implementation SUCCEEDS here with an exactly-sized
    // small output — the guarantee under test is simply "no abort, no
    // 68 GiB allocation": the asymmetry is the point (the 68 GiB scratch
    // needs a large `src_h` AND a large `dst_w`, see
    // `rejects_huge_intermediate_with_tiny_ends`).
    let src = vec![0u8; 131072 * CHANNELS];
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      let r = resize_rgba8(&src, 131072, 1, 1, 131072, f);
      match r {
        Ok(out) => assert_eq!(
          out.len(),
          131072 * CHANNELS,
          "{f:?}: wide->skinny output must be exactly dst_w*dst_h*4"
        ),
        Err(Error::ShapeMismatch { .. }) => {}
        Err(other) => panic!("{f:?}: unexpected error {other:?}"),
      }
    }
  }

  #[test]
  fn rejects_huge_intermediate_with_tiny_ends() {
    // The horizontal-pass intermediate is `src_h * dst_w * 4` — it blows
    // up only when BOTH `src_h` (input height) and `dst_w` (untrusted
    // target width) are large, which is exactly the gap the public
    // `resize` wrapper's source/destination caps miss. A `1x131072`
    // source (`src_h = 131072`) resized to a `131072`-WIDE, 1-tall target
    // gives a 512 KiB source, a 512 KiB destination, and an intermediate
    // of `131072 * 131072 * 4` ≈ 68 GiB. Use `dst_h = 1` so the
    // destination stays tiny and ONLY the intermediate trips the cap —
    // this proves the intermediate guard fires, not the destination
    // guard. (Same shape as `rejects_skinny_to_wide_oversized_intermediate`
    // but kept distinct to document the both-ends-tiny adversarial framing
    // explicitly.)
    let src = vec![7u8; 131072 * CHANNELS];
    let r = resize_rgba8(&src, 1, 131072, 131072, 1, Filter::Bicubic);
    assert!(
      matches!(r, Err(Error::ShapeMismatch { .. })),
      "huge intermediate with tiny source+dest must be ShapeMismatch, got {r:?}"
    );
  }

  #[test]
  fn rejects_oversized_coefficient_table() {
    // Coefficient-buffer adversarial case, exercised directly through
    // `precompute_coeffs`. The weight table is `out_size * ksize` `i32`s
    // and the `bounds` table is `out_size` `(usize, usize)` pairs — both
    // scale with the (untrusted) output dimension. With
    // `out_size = 200_000_000` and `in_size = 1` the weight table is
    // `200_000_000 * ksize(=1) * 4` = 800 MB and the `bounds` table is
    // `200_000_000 * 16` = 3.2 GB — both far over the 512 MiB cap, so
    // `checked_buffer_bytes` (whichever of the two is reached first)
    // rejects with `ShapeMismatch` rather than reserving + zero-filling
    // multiple GB.
    // `Coeffs` is not `Debug`; match the result rather than `{:?}`-ing it.
    let r = precompute_coeffs(1, 200_000_000, Filter::Bilinear);
    assert!(
      matches!(r, Err(Error::ShapeMismatch { .. })),
      "200M-wide coefficient table must exceed the 512 MiB cap (got Ok or wrong error)"
    );
    // And via the full resize: a `1x4` source upscaled to a
    // `200_000_000`-wide target must reject — recoverable, no abort. (The
    // destination `200_000_000*1*4` = 800 MB already trips the
    // destination cap; were it not for that, the h-axis coefficient table
    // and the horizontal intermediate would. Every one of these guards
    // yields `ShapeMismatch`.)
    let src = vec![0u8; 4 * CHANNELS];
    let r2 = resize_rgba8(&src, 1, 4, 200_000_000, 1, Filter::Bilinear);
    assert!(
      matches!(r2, Err(Error::ShapeMismatch { .. })),
      "resize to a 200M-wide target must be ShapeMismatch, got {r2:?}"
    );
  }

  #[test]
  fn checked_buffer_bytes_caps_and_overflows() {
    // Direct unit test of the helper. Under-cap passes and returns the
    // byte product; over-cap and overflow both yield ShapeMismatch.
    assert_eq!(
      checked_buffer_bytes(1024, 4, "ok").unwrap(),
      4096,
      "under-cap product must pass through"
    );
    // Exactly at the cap (512 MiB) is allowed; one byte over is not.
    assert_eq!(
      checked_buffer_bytes(MAX_DECODED_IMAGE_BYTES, 1, "at-cap").unwrap(),
      MAX_DECODED_IMAGE_BYTES,
      "a buffer exactly at the cap must be allowed"
    );
    assert!(
      matches!(
        checked_buffer_bytes(MAX_DECODED_IMAGE_BYTES + 1, 1, "over"),
        Err(Error::ShapeMismatch { .. })
      ),
      "one byte over the cap must be rejected"
    );
    assert!(
      matches!(
        checked_buffer_bytes(usize::MAX, 4, "overflow"),
        Err(Error::ShapeMismatch { .. })
      ),
      "a product overflowing usize must be rejected (not wrap)"
    );
  }

  #[test]
  fn output_length_is_exact() {
    // Every accepted resize returns exactly dst_w*dst_h*4 bytes — the
    // invariant `vlm::image::resize` relies on for `ImageBuffer::from_raw`.
    let src = make_src(8, 6, 3);
    for f in [
      Filter::Nearest,
      Filter::Bilinear,
      Filter::Bicubic,
      Filter::Lanczos3,
    ] {
      let out = resize_rgba8(&src, 8, 6, 5, 4, f).unwrap();
      assert_eq!(out.len(), 5 * 4 * CHANNELS, "filter {f:?} output length");
    }
  }

  #[test]
  fn constant_image_is_preserved() {
    // A constant-color image must reproduce the constant at every output
    // pixel for every convolution filter (kernel sums to 1.0). Exact for
    // the integer path (no rounding drift on a flat field).
    let mut src = Vec::with_capacity(6 * 6 * CHANNELS);
    for _ in 0..6 * 6 {
      src.extend_from_slice(&[123, 45, 200, 255]);
    }
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      for &(dw, dh) in &[(3usize, 3usize), (9, 9), (4, 7)] {
        let out = resize_rgba8(&src, 6, 6, dw, dh, f).unwrap();
        for px in out.chunks_exact(CHANNELS) {
          assert_eq!(
            px,
            &[123, 45, 200, 255],
            "constant must survive {f:?} -> {dw}x{dh}"
          );
        }
      }
    }
  }

  #[test]
  fn muldiv255_matches_pil_and_is_opaque_identity() {
    // MULDIV255(c, 255) must be the identity for EVERY c (PIL relies on
    // this so an opaque RGBA resize is bit-identical to a straight one).
    for c in 0u8..=255 {
      assert_eq!(
        muldiv255(c, 255),
        c,
        "MULDIV255({c}, 255) must equal {c} (opaque identity)"
      );
      // MULDIV255(c, 0) == 0 for every c (zero-alpha kills the colour).
      assert_eq!(muldiv255(c, 0), 0, "MULDIV255({c}, 0) must be 0");
    }
    // Spot-check PIL's exact rounding against a hand-computed value:
    // MULDIV255(255, 128) = ((32768>>8)+32768)>>8 = (128+32768)>>8 = 128.
    assert_eq!(muldiv255(255, 128), 128, "MULDIV255(255,128) hand-checked");
    // MULDIV255(200, 100) = ((20128>>8)+20128)>>8 = (78+20128)>>8 = 78.
    assert_eq!(muldiv255(200, 100), 78, "MULDIV255(200,100) hand-checked");
  }

  #[test]
  fn premultiply_unpremultiply_opaque_is_identity() {
    // For a fully-opaque buffer (A == 255) premultiply then unpremultiply
    // must round-trip to the exact input — this is why the opaque
    // PIL-reference resize tests are unaffected by the premultiply path.
    let src: Vec<u8> = (0u8..=255).flat_map(|c| [c, 255 - c, c / 2, 255]).collect();
    let pm = premultiply_rgba(&src).unwrap();
    assert_eq!(pm, src, "premultiply must be identity for opaque alpha");
    let mut round = pm;
    unpremultiply_rgba(&mut round);
    assert_eq!(
      round, src,
      "unpremultiply must be identity for opaque alpha"
    );
  }

  #[test]
  fn premultiply_transparent_pixel_zeros_colour() {
    // A fully-transparent pixel (A == 0): premultiply zeros every colour
    // channel (PIL `MULDIV255(c, 0) == 0`), and unpremultiply leaves the
    // already-zero colour at zero (PIL passthrough for A == 0).
    let src = vec![255u8, 128, 64, 0]; // transparent, arbitrary colour
    let pm = premultiply_rgba(&src).unwrap();
    assert_eq!(
      pm,
      vec![0, 0, 0, 0],
      "premultiply of a transparent pixel must zero the colour channels"
    );
    let mut round = pm;
    unpremultiply_rgba(&mut round);
    assert_eq!(
      round,
      vec![0, 0, 0, 0],
      "unpremultiply of a zero-alpha pixel keeps colour 0 (PIL passthrough)"
    );
  }

  #[test]
  fn unpremultiply_clips_and_divides_like_pil() {
    // Partial alpha: unpremultiply does CLIP8(255*c/a) (truncating
    // integer division, clamp [0,255]).
    // a=128: CLIP8(255*64/128) = 16320/128 = 127.
    let mut buf = vec![64u8, 0, 0, 128];
    unpremultiply_rgba(&mut buf);
    assert_eq!(
      buf[0], 127,
      "unpremultiply 64 over alpha 128: 255*64/128=127"
    );
    assert_eq!(buf[3], 128, "alpha unchanged");
    // Premultiplied colour > alpha (possible after convolution rounding):
    // CLIP8 must clamp to 255. c=200, a=100 -> 255*200/100=510 -> 255.
    let mut buf2 = vec![200u8, 0, 0, 100];
    unpremultiply_rgba(&mut buf2);
    assert_eq!(
      buf2[0], 255,
      "unpremultiply must clamp an over-alpha colour to 255"
    );
  }

  #[test]
  fn resize_premultiplied_transparent_red_opaque_blue() {
    // Codex example at the kernel level: transparent-red `(255,0,0,0)`
    // next to opaque-blue `(0,0,255,255)`, 2x1 -> 1x1. The premultiplied
    // path must yield pure blue with half alpha `(0,0,255,128)` for every
    // non-NEAREST filter — NOT the straight-channel purple
    // `(128,0,128,128)`. NEAREST is exempt (pure gather, no premultiply).
    let src = [255u8, 0, 0, 0, 0, 0, 255, 255]; // 2x1: t-red, o-blue
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      let out = resize_rgba8(&src, 2, 1, 1, 1, f).unwrap();
      assert_eq!(
        out,
        vec![0, 0, 255, 128],
        "{f:?}: premultiplied-alpha resize must give pure blue (0,0,255,128)"
      );
    }
    // NEAREST gathers the rightmost pixel (out 0 -> floor(0.5*2/1)=1):
    // straight opaque blue, no premultiply.
    let nn = resize_rgba8(&src, 2, 1, 1, 1, Filter::Nearest).unwrap();
    assert_eq!(
      nn,
      vec![0, 0, 255, 255],
      "NEAREST must not premultiply — gathers the opaque-blue pixel verbatim"
    );
  }

  #[test]
  fn precompute_coeffs_weights_sum_to_unity_fixedpoint() {
    // Each output index's normalized fixed-point taps should sum to
    // approximately 1<<PRECISION_BITS (the rounding may shift the sum by
    // at most `n` LSB across `n` taps). This guards the normalization.
    let one = 1i64 << PRECISION_BITS;
    for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
      for &(insz, outsz) in &[(8usize, 3usize), (3, 8), (5, 5), (16, 4)] {
        let c = precompute_coeffs(insz, outsz, f).unwrap();
        for o in 0..outsz {
          let (_, n) = c.bounds[o];
          let s: i64 = c.weights[o * c.ksize..o * c.ksize + n]
            .iter()
            .map(|&w| i64::from(w))
            .sum();
          let tol = n as i64 + 1;
          assert!(
            (s - one).abs() <= tol,
            "{f:?} {insz}->{outsz} out {o}: tap sum {s} not within {tol} of {one}"
          );
        }
      }
    }
  }
}
