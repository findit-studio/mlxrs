//! Core image preprocessing primitives for vision-language models.
//!
//! Ported 1:1 from
//! [`mlx-swift-lm/Libraries/MLXVLM/MediaProcessing.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift)
//! (the focused swift reference for VLM image preprocessing, 567 lines)
//! and cross-checked against
//! [`mlx-vlm/mlx_vlm/utils.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py)
//! (`load_image`, `resize_image`, `process_image`).
//!
//! ## Scope
//! - **In scope (this PR):** the *cross-model* preprocessing surface every
//!   ViT-class VLM encoder shares — image load, resize, channel layout,
//!   `[0, 255]` u8 → f32 conversion, `1/255` rescale, per-channel
//!   ImageNet-style normalization, uniform-grid patchify, and the
//!   end-to-end [`preprocess`] composer.
//! - **Out of scope:** per-model image processors (CLIP / SigLIP / Idefics
//!   / Qwen-VL / etc. specialized cropping, dynamic aspect-ratio patching,
//!   anyres tiling). Those are per-usecase per the project's
//!   no-per-model-arch rule; they live in user code that depends on these
//!   primitives. Video frame preprocessing is also out of scope (the
//!   `MediaProcessing.asProcessedSequence` family on lines 288-526 of the
//!   swift reference) — VLM video support is a sibling concern.
//!
//! ## Conventions
//! - **Channel layout (intentional divergence from swift):** outputs from
//!   [`image_to_array`] and [`preprocess`] are *channel-last*
//!   `[H, W, 3]`. Alpha is intentionally dropped — see
//!   [`image_to_array`]'s doc for the swift `array[..., :3]` parity
//!   citation. The swift
//!   reference's `MediaProcessing.asMLXArray` reshapes/transposes to
//!   *planar* `[1, C, H, W]` (`MediaProcessing.swift:190`) because the
//!   per-model encoders in `MLXVLM/Models/*.swift` consume that layout
//!   directly. We deliberately STOP at the channel-last step because:
//!     1. The `(3,)` mean/std broadcasts cleanly across the last axis
//!        without an extra transpose, so [`normalize_imagenet`] needs
//!        no layout-specific reshape.
//!     2. The planar `[1, C, H, W]` reshape is a per-model contract
//!        (some encoders want `[C, H, W]` without the batch axis;
//!        others want patchified `[N, P, P, C]` etc.), so emitting it
//!        from the cross-model primitive would force every caller to
//!        un-do or re-do the transpose for their encoder. Per the
//!        project's no-per-model-arch rule (see
//!        `feedback_no_per_model_arch_porting`), per-model layout is
//!        the per-model processor's job; the cross-model primitive
//!        owns the layout-agnostic ImageNet pipeline only.
//!     3. The per-model planar conversion is one mlx call
//!        (`reshape((1, h, w, 3))` + `transpose_axes(&[0, 3, 1, 2])`)
//!        on the lazy `Array` — zero memory cost beyond the metadata
//!        update. The end-to-end math is unchanged; only the trailing
//!        layout step moves to the call site.
//! - **Dtype:** [`image_to_array`] returns `f32` in `[0.0, 255.0]` *before*
//!   [`rescale`] — exactly mirroring the swift `CIFormat.RGBAf` render
//!   (`MediaProcessing.swift:171`) which produces f32 in `[0, 255]`.
//! - **No implicit eval:** every primitive composes lazily on `Array`;
//!   callers must `eval()` (or use a data accessor) to materialize.
//! - **No hot-path allocations beyond unavoidable decode/resize:** the
//!   `image` crate's `decode` / `imageops::resize` themselves allocate
//!   (CPU pixel buffers), and the f32 conversion of the resized buffer is
//!   one unavoidable `Vec<f32>` before [`Array::from_slice`] copies it
//!   into MLX. All other steps stay on the FFI-owned arrays.
//!
//! ## Pipeline
//! ```text
//! load_image → resize → image_to_array → rescale → normalize_imagenet
//! ```
//! [`preprocess`] composes the full chain off a decoded
//! [`image::DynamicImage`] + an [`ImageProcessorConfig`].

use crate::{
  Dtype,
  array::Array,
  error::{Error, Result},
  ops::{
    arithmetic::{divide, multiply, subtract},
    misc::astype,
    shape::{reshape, transpose_axes},
  },
};

/// Interpolation filter for [`resize`], mirroring swift
/// `MediaProcessing.swift`'s resampler choices (lines 81-132):
/// [`resampleLanczos`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L81-L103)
/// and
/// [`resampleBicubic`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L110-L132).
///
/// Backed by `fast_image_resize` (SIMD-accelerated, 5-15x faster than
/// `image::imageops::resize`). Filter names match both `fast_image_resize`'s
/// `FilterType` and (where the kernel is identical) the older
/// `image::imageops::FilterType` — so existing call-site usage of `Bilinear`,
/// `Bicubic`, `Lanczos3` is unchanged.
///
/// The swift reference exposes `bicubic` (default) and `lanczos`; we add
/// `Nearest` and `Bilinear` because they appear in the python VLM ecosystem
/// (PIL's `Image.resize` `resample=` argument that `mlx-vlm`'s
/// `resize_image` uses transitively at `mlx_vlm/utils.py:835-839`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
  /// Nearest-neighbor (no smoothing). Cheapest; rarely used for VLM.
  Nearest,
  /// Bilinear interpolation (triangle kernel — same as the older
  /// `image::imageops::FilterType::Triangle`).
  Bilinear,
  /// Bicubic interpolation (Catmull-Rom variant — matches
  /// `image::imageops::FilterType::CatmullRom` and PIL's `Image.BICUBIC`).
  /// Mirrors the swift `resampleBicubic` default
  /// (`MediaProcessing.swift:110-132`); the recommended choice for most
  /// ViT-class encoders.
  Bicubic,
  /// Lanczos3 interpolation (window=3 sinc-windowed sinc).
  /// Mirrors the swift `resampleLanczos` (`MediaProcessing.swift:81-103`).
  Lanczos3,
}

impl ResizeFilter {
  /// Map to `fast_image_resize`'s `ResizeAlg` enum. Kept private; the
  /// crate's types do not leak into our public surface. Filter parity:
  ///
  /// - `Nearest` → `ResizeAlg::Nearest` (no convolution).
  /// - `Bilinear` → `Convolution(FilterType::Bilinear)`. Same triangle
  ///   kernel that `image::FilterType::Triangle` implemented.
  /// - `Bicubic` → `Convolution(FilterType::CatmullRom)`. The
  ///   Catmull-Rom variant matches `image::FilterType::CatmullRom` and
  ///   PIL's `Image.BICUBIC`.
  /// - `Lanczos3` → `Convolution(FilterType::Lanczos3)` (window=3
  ///   sinc-windowed sinc).
  fn to_fir_alg(self) -> ::fast_image_resize::ResizeAlg {
    use ::fast_image_resize::{FilterType, ResizeAlg};
    match self {
      Self::Nearest => ResizeAlg::Nearest,
      Self::Bilinear => ResizeAlg::Convolution(FilterType::Bilinear),
      Self::Bicubic => ResizeAlg::Convolution(FilterType::CatmullRom),
      Self::Lanczos3 => ResizeAlg::Convolution(FilterType::Lanczos3),
    }
  }
}

/// Channel layout for [`image_to_array`]. `RGB` is the swift default
/// (`MediaProcessing.swift:171` — `CIFormat.RGBAf`'s RGBA channel order);
/// `BGR` is exposed for parity with python image-processor configs that
/// use OpenCV-style BGR (e.g. some older CLIP variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorOrder {
  /// Red-Green-Blue (the default; matches PIL / swift CoreImage).
  Rgb,
  /// Blue-Green-Red (OpenCV-style; swap R↔B).
  Bgr,
}

/// Image preprocessing config — the *union* of fields common across VLM
/// image processors.
///
/// Mirrors the swift `MediaProcessing` pipeline configuration (no single
/// struct in the swift source — the swift pipeline composes call-site
/// arguments at `MediaProcessing.swift:30-39` in the module-doc example,
/// and the python `BaseImageProcessor` HF subclasses expose the same
/// fields). [`Default`] is the ImageNet baseline that matches the values
/// hardcoded in nearly every HF image-processor JSON:
/// `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`,
/// `rescale_factor = 1/255`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageProcessorConfig {
  /// Target image size `(height, width)`.
  pub size: (u32, u32),
  /// Per-channel mean for [`normalize_imagenet`].
  pub mean: [f32; 3],
  /// Per-channel std for [`normalize_imagenet`].
  pub std: [f32; 3],
  /// Multiplier applied by [`rescale`] (typically `1.0 / 255.0`).
  pub rescale_factor: f32,
  /// Whether the [`preprocess`] composer applies [`resize`].
  pub do_resize: bool,
  /// Whether the [`preprocess`] composer applies [`rescale`].
  pub do_rescale: bool,
  /// Whether the [`preprocess`] composer applies [`normalize_imagenet`].
  pub do_normalize: bool,
  /// Interpolation filter forwarded to [`resize`].
  pub resample: ResizeFilter,
  /// Channel layout forwarded to [`image_to_array`].
  pub color_order: ColorOrder,
}

impl Default for ImageProcessorConfig {
  /// ImageNet defaults: `size = (224, 224)`, `mean = [0.485, 0.456,
  /// 0.406]`, `std = [0.229, 0.224, 0.225]`, `rescale_factor = 1/255`,
  /// `resample = Bicubic`, `color_order = Rgb`, all `do_*` flags `true`.
  /// These are the values nearly every CLIP / SigLIP / DINO / ViT
  /// preprocessing config ships with.
  fn default() -> Self {
    Self {
      size: (224, 224),
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      rescale_factor: 1.0 / 255.0,
      do_resize: true,
      do_rescale: true,
      do_normalize: true,
      resample: ResizeFilter::Bicubic,
      color_order: ColorOrder::Rgb,
    }
  }
}

/// Load and decode an image from disk, applying EXIF orientation.
///
/// Mirrors the swift `CIImage(contentsOf: url)` /
/// `CIImage(cgImage:)` entry points implied by
/// `MediaProcessing.swift:288-330` (the video frame loader uses
/// `CIImage(cgImage:)` per line 321-322 — Apple's `CIImage` honors
/// EXIF orientation transparently on macOS), and the python
/// [`load_image`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/utils.py#L801-L832)
/// which explicitly applies `ImageOps.exif_transpose(image)` before
/// returning. To match that parity, we route through `ImageReader`,
/// read the decoder's `orientation()` (which inspects EXIF metadata),
/// decode, then `DynamicImage::apply_orientation` so common
/// phone-camera JPEGs come back upright.
///
/// **Scope:** local files only. HTTP / data-URI / `BytesIO` sources are
/// handled by the python reference but are out of scope here; callers
/// that need them can construct an `image::DynamicImage` themselves and
/// hand it to [`preprocess`] / [`resize`] directly.
///
/// **Allocation guard:** this function honors the `image` crate's
/// default `Limits` (512 MiB `max_alloc`) — we explicitly call
/// `Limits::default().reserve(decoder.total_bytes())?` before consuming
/// the decoder, mirroring what `ImageReader::decode()` does internally
/// (image 0.25 `io::image_reader_type::ImageReader::decode`). Using
/// `into_decoder` so we can read EXIF orientation does NOT bypass this
/// check. Callers that need a different ceiling can pre-validate
/// dimensions with `image::ImageReader::open(path)?.into_dimensions()?`
/// (O(1) header probe) and either accept or build a custom reader with
/// `image::Limits` themselves before handing the result into
/// [`preprocess`]. Oversized images are rejected with
/// [`Error::Backend`] (the underlying `image::ImageError::Limits`).
pub fn load_image(path: &std::path::Path) -> Result<::image::DynamicImage> {
  // `ImageDecoder` is the trait that provides `.orientation()`; pull it
  // into local scope so the method resolves on the opaque decoder type
  // returned by `into_decoder`.
  use ::image::ImageDecoder as _;

  let backend_err = |e: ::image::ImageError| Error::Backend {
    message: format!("vlm::image::load_image({}): {e}", path.display()),
  };
  let io_err = |e: std::io::Error| Error::Backend {
    message: format!("vlm::image::load_image({}): {e}", path.display()),
  };
  // ImageReader::open guesses the format from the path extension; we
  // then call `with_guessed_format` to fall back to content sniffing
  // for extension-less paths (mirroring python `Image.open` which
  // sniffs the file header).
  let reader = ::image::ImageReader::open(path)
    .map_err(io_err)?
    .with_guessed_format()
    .map_err(io_err)?;
  let mut decoder = reader.into_decoder().map_err(backend_err)?;
  // NOTE (Codex review, combined-wave1-fu): an adversarial-review concern
  // observed that `into_decoder()` calls `JpegDecoder::new()` which does
  // `r.read_to_end(&mut input)?` *before* any `Limits` check fires
  // (image 0.25 `codecs/jpeg/decoder.rs:30-33`), so a very large JPEG
  // file allocates the compressed bytes uncapped before `total_bytes()`
  // gates the decoded buffer. Rejected on faithful-parity grounds: the
  // upstream canonical `ImageReader::decode()` flow has the *identical*
  // ordering (`image_reader_type.rs:311-322` — `make_decoder` runs the
  // jpeg `read_to_end` before `limits.reserve(decoder.total_bytes())`),
  // and python `PIL.Image.open` likewise does not cap compressed input
  // (only the post-decode `MAX_IMAGE_PIXELS` warning). The function's
  // documented scope is *local files only* — callers that need to
  // bound untrusted input should pre-validate with
  // `std::fs::metadata(path).len()` or use a `Take`-wrapped reader, the
  // same as for any `std::fs::read`. Per project rule
  // [[feedback_match_official_binding_design]] this primitive mirrors
  // the references' behavior and does not add divergent hardening.
  // `decoder.orientation()` returns `Orientation::NoTransforms` for
  // formats that don't carry orientation metadata (PNG), so this is
  // safe for every format we accept. Only JPEG photos here will incur
  // a real rotation (Copilot review #3272880155 — `mlxrs/Cargo.toml`
  // enables `image` with only the `png` + `jpeg` features; TIFF/WebP
  // are NOT in the build). We read orientation here while we still
  // have a `&mut` borrow on the decoder; once it's consumed by
  // `from_decoder` below, the metadata can no longer be queried.
  let orientation = decoder.orientation().map_err(backend_err)?;
  // Preserve the 512 MiB default allocation guard that
  // `ImageReader::decode()` enforces. Our use of `into_decoder` (so
  // we can read orientation) skips the `limits.reserve(total_bytes)`
  // check `decode()` does internally — see image 0.25
  // `io::image_reader_type::ImageReader::decode`. Mirror that check
  // explicitly so an oversized image is rejected with a clean
  // `Error::Backend` instead of running through the decoder and
  // panic-allocating downstream.
  let mut limits = ::image::Limits::default();
  limits.reserve(decoder.total_bytes()).map_err(backend_err)?;
  decoder.set_limits(limits).map_err(backend_err)?;
  let mut img = ::image::DynamicImage::from_decoder(decoder).map_err(backend_err)?;
  img.apply_orientation(orientation);
  Ok(img)
}

/// Resize `img` to `(height, width)` using `filter`.
///
/// Mirrors swift `MediaProcessing.resampleBicubic` /
/// `resampleLanczos` (`MediaProcessing.swift:110-132` / `81-103`).
///
/// The swift reference applies a separate y-scale + aspect-ratio adjust
/// (then a final crop) to "ensure exact dimensions" (lines 113-131); the
/// `image::imageops::resize` we forward to *also* produces exact
/// dimensions (the crate documents `resize(image, nwidth, nheight, filter)`
/// as scaling to exactly `nwidth × nheight`), so the trailing crop step
/// is unnecessary.
///
/// **Aspect-ratio preservation:** none. This is a *forced* resize to the
/// target dimensions, mirroring the swift `resampleBicubic` (which also
/// distorts to the requested size — it computes independent x- and
/// y-scale factors at lines 113-114 and applies them separately). The
/// python `resize_image` (`mlx_vlm/utils.py:835-839`) computes an
/// aspect-ratio-preserving thumbnail; that variant is a per-model
/// concern and is intentionally not exposed here. Callers that need it
/// can compute the target tuple themselves before calling `resize`.
///
/// **Return type — infallible by parity:** the function signature is
/// `-> DynamicImage` (not `Result<...>`) because both reference
/// implementations are infallible: swift `resampleBicubic(_, to:)
/// -> CIImage` (line 110-132; the `outputImage!` unwrap is an
/// explicit panic on bad input), and python `Image.resize(new_size)
/// -> Image` (no `Result`). `ImageProcessorConfig.size` is model
/// metadata loaded from a trusted JSON, not arbitrary user input;
/// callers that need to validate untrusted size values should clamp
/// or reject them before constructing the config. A pathological
/// target dimension (`u32::MAX` etc.) will panic-allocate inside
/// `image::imageops::resize` exactly as the swift `CIFilter.bicubic
/// ScaleTransform.outputImage!` and python `Image.resize` would; the
/// faithful-port contract preserves that behavior.
// NOTE (Codex finding, round 4): an adversarial-review concern asked
// for `resize` to return `Result<DynamicImage>` with explicit
// byte-budget validation. Rejected on faithful-parity grounds — both
// the swift and python references are infallible at this entry point
// and treat the size argument as trusted-config input. Wrapping in
// `Result` would diverge from both reference signatures and force a
// behavioral change on every per-model caller. Untrusted-size
// callers can bound the input before calling.
pub fn resize(
  img: &::image::DynamicImage,
  target: (u32, u32),
  filter: ResizeFilter,
) -> ::image::DynamicImage {
  let (height, width) = target;
  // SIMD-accelerated resize via `fast_image_resize` — 5-15x faster than
  // `image::imageops::resize` on the same algorithms (used by `wgpu`
  // examples, `kornia-rs`, etc.). Decode-side stays on `image` (above
  // in `load_image`); only the resize hot path switches. Public API of
  // this fn is unchanged.
  //
  // Pixel-type: RGBA8 for parity with the prior behavior (image-rs's
  // `imageops::resize` over a `DynamicImage` projects to `Rgba8`
  // unconditionally; downstream `image_to_array` drops alpha as before).
  // `img.to_rgba8()` is a borrow-or-convert (no-copy when the source is
  // already `ImageRgba8`).
  let src = img.to_rgba8();
  let src_view = ::fast_image_resize::images::ImageRef::new(
    src.width(),
    src.height(),
    src.as_raw(),
    ::fast_image_resize::PixelType::U8x4,
  )
  // `ImageRef::new` only errors on `width * height * channels` overflow
  // or buffer-length mismatch. The former is impossible because the
  // dimensions came from a successfully-decoded `DynamicImage` (which
  // image-rs validated in `from_decoder`). The latter is impossible
  // because `RgbaImage::as_raw().len() == width * height * 4` by
  // construction.
  .expect("ImageRef::new: source dims/buffer length validated by image-rs decoder");
  let mut dst =
    ::fast_image_resize::images::Image::new(width, height, ::fast_image_resize::PixelType::U8x4);
  ::fast_image_resize::Resizer::new()
    .resize(
      &src_view,
      &mut dst,
      &::fast_image_resize::ResizeOptions::new().resize_alg(filter.to_fir_alg()),
    )
    // Source + destination pixel types are identical (both `U8x4`) by
    // construction, so the only error class (`DifferentTypesOfPixelsError`)
    // is structurally excluded.
    .expect("Resizer::resize: src/dst PixelType match by construction (both U8x4)");
  ::image::DynamicImage::ImageRgba8(
    ::image::ImageBuffer::from_raw(width, height, dst.into_vec())
      // `Image::new(width, height, U8x4)` allocates exactly
      // `width * height * 4` bytes — the precise length
      // `ImageBuffer::from_raw` requires for a `width x height` RGBA buffer.
      .expect("ImageBuffer::from_raw: dst buffer length matches width * height * 4 by construction"),
  )
}

/// Convert a [`image::DynamicImage`] to an `Array` of shape `[H, W, 3]`,
/// dtype `f32`, value range `[0.0, 255.0]` (BEFORE [`rescale`]).
///
/// Mirrors swift `MediaProcessing.asMLXArray` (`MediaProcessing.swift:
/// 164-193`) up to channel layout: the swift reference renders RGBAf at
/// line 171, slices the 4th alpha channel at line 187 (`array[0..., 0...,
/// ..<3]`), and *additionally* reshapes to planar `[1, C, H, W]` at line
/// 190. We stop at the channel-last `[H, W, 3]` step — channel-last is
/// the natural layout for the subsequent [`normalize_imagenet`] +
/// [`rescale`] (the `(3,)` mean/std broadcast cleanly over the last
/// axis), and the per-model planar conversion (`transpose_axes(&[2, 0,
/// 1])` then optional batch axis) is a model-input detail the per-model
/// processor owns.
///
/// **Alpha:** dropped. The swift reference also drops it explicitly at
/// line 187. RGBA images are converted to RGB by discarding the alpha
/// channel (no compositing onto a background) to match the swift
/// behavior.
///
/// **`color_order`:** if [`ColorOrder::Bgr`], the per-pixel R and B
/// channels are swapped during the buffer build (no MLX transpose / no
/// extra Array allocation).
///
/// **Memory ceiling:** none at the `image_to_array` boundary itself.
/// [`load_image`] enforces the `image` crate's 512 MiB
/// `Limits::default().max_alloc` guard before any decoded buffer is
/// returned, so the `image_to_array` input is already size-bounded when
/// it comes through that path. The `h * w * 3` f32 buffer allocated
/// here is `4 * decoded_byte_count` (`Vec::with_capacity` does not yet
/// allocate eagerly until the first push, and we then push exactly
/// `total` f32s), so the upper bound is roughly `4 * 512 MiB = 2 GiB`
/// of f32s for a decoder-default `load_image` source. Callers that
/// hand a `DynamicImage` from a different source (raw `image::open`,
/// network decoders, etc.) inherit whatever limit that source imposed;
/// pre-validating `img.dimensions()` (a cheap O(1) field read) is the
/// standard escape hatch.
//
// NOTE (Codex finding, rounds 1 + 2 + 3): the OOM concern across three
// rounds. The decode-side bound is now restored at `load_image`
// (`Limits::default().reserve(total_bytes)?` mirrors what
// `ImageReader::decode()` does internally). The remaining `Vec<f32>`
// allocation here is the unavoidable `decoded -> f32` widening — the
// swift reference allocates the same buffer at `Data(count:
// w * h * bytesPerPixel)` in `MediaProcessing.swift:176`, and the
// python `np.asarray(image)` does too. A unilateral Rust gate on this
// specific `Vec<f32>` would have to come with a matching gate in both
// references; we surface the f32 allocation as the documented
// "unavoidable resized-buffer widening" already called out in the
// module `Conventions` block. This is a documented design choice plus
// a now-enforced decode-side guard, not an oversight.
pub fn image_to_array(img: &::image::DynamicImage, color_order: ColorOrder) -> Result<Array> {
  // Force decode to an RGB8 view; this dispatches to the most efficient
  // conversion in the `image` crate for each `DynamicImage` variant
  // (Luma8 broadcasts grey across all 3 channels, Rgba* drops alpha,
  // already-Rgb8 is a no-op clone). Alpha-aware variants are
  // intentionally NOT preserved per the swift reference (line 187).
  let rgb = img.to_rgb8();
  let (w, h) = rgb.dimensions();
  let w_usize = w as usize;
  let h_usize = h as usize;
  // FFI-bound shape product overflow guard. `Array::from_slice` validates
  // shape-product vs buffer length but does so in `usize` arithmetic
  // *after* our cast; on a 32-bit usize the multiplication
  // `h_usize * w_usize * 3` can wrap silently. Catch it here with a
  // recoverable `Error::ShapeMismatch` so callers see a clean error rather
  // than a panic downstream. This MUST run before `Vec::with_capacity`
  // so a hostile dimension product cannot abort in the allocator.
  let total = h_usize
    .checked_mul(w_usize)
    .and_then(|hw| hw.checked_mul(3))
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!("image_to_array: h*w*3 overflows usize for {h}x{w}"),
    })?;
  // Pre-sized `Vec<f32>` filled from `rgb.as_raw()` (`&[u8]` of length
  // `H*W*3`, RGB-ordered) — eliminates per-element capacity re-checks
  // versus the prior `rgb.pixels() + push` form AND lets LLVM
  // auto-vectorize the u8 → f32 widening (the RGB path is a straight
  // contiguous cast; the BGR path is a 3-element shuffle that
  // `chunks_exact(3)` keeps bounded enough for unrolling). See the
  // golden-standard `VLM-3` tracker entry for the rationale.
  //
  // `rgb.as_raw()` is documented to return `&[u8]` of length
  // `width * height * 3` (image 0.25 `ImageBuffer::as_raw` /
  // `ImageBuffer<P, Vec<P::Subpixel>>` with `P = Rgb<u8>`), so the
  // `chunks_exact(3)` iterator yields exactly `H*W` slices of length
  // 3 with no remainder — the BGR branch's indexed reads never panic.
  // `image::ImageBuffer::from_raw(w, h, vec)` accepts a backing buffer
  // whose length is AT LEAST `w * h * channels`; `as_raw()` returns the
  // full backing Vec (including any tail past the logical extent), while
  // the safer `pixels()` iterator slices to exactly `w * h` pixels.
  // Slice `as_raw()` down to the logical `total = H*W*3` bytes here so
  // the subsequent fill loop iterates the correct extent — without this
  // slice an overlong-backing-buffer image would grow the `buf` past
  // the `try_reserve_exact(total)` reservation via infallible allocation,
  // reintroducing the abort-on-OOM hazard the recoverable reservation
  // is supposed to remove. A standard `to_rgb8()` output is exactly
  // `total` long so this is a no-op for the common path.
  let raw = rgb
    .as_raw()
    .get(..total)
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "image_to_array: rgb backing buffer too short: {} bytes < H*W*3={total} for H={h} W={w}",
        rgb.as_raw().len()
      ),
    })?;
  // Recoverable OOM at the f32 widening boundary. `Vec::with_capacity`
  // would `abort()` on a hostile-but-non-overflowing image (the
  // `checked_mul` only proves `total` fits `usize`, not that the
  // `total * 4` byte alloc succeeds). `try_reserve_exact` surfaces an
  // allocator failure as a recoverable `Error::OutOfMemory` so callers
  // get a typed Err instead of process termination — matches the
  // allocation-discipline pattern `mlxrs::error::Error::OutOfMemory`
  // exists for.
  let mut buf: Vec<f32> = Vec::new();
  buf
    .try_reserve_exact(total)
    .map_err(|_| Error::OutOfMemory)?;
  match color_order {
    ColorOrder::Rgb => {
      // Contiguous u8 → f32 widening — LLVM auto-vectorizes this on
      // both NEON (aarch64) and AVX2 (x86_64) backends.
      buf.extend(raw.iter().map(|&b| f32::from(b)));
    }
    ColorOrder::Bgr => {
      // Per-pixel R↔B swap. `chunks_exact(3)` yields `&[u8]` of
      // guaranteed length 3 (verified by `as_raw().len() == total`
      // above), keeping the inner-loop indices bounded so LLVM can
      // unroll the 3-element shuffle.
      for px in raw.chunks_exact(3) {
        buf.push(f32::from(px[2]));
        buf.push(f32::from(px[1]));
        buf.push(f32::from(px[0]));
      }
    }
  }
  debug_assert_eq!(
    buf.len(),
    total,
    "buf fill length must equal pre-computed total"
  );
  Array::from_slice(&buf, &(h_usize, w_usize, 3))
}

/// Multiply `arr` by `scale` (typically `1.0 / 255.0`).
///
/// Mirrors the rescale step folded into the swift `MediaProcessing.
/// normalize` colorMatrix (`MediaProcessing.swift:145-156` — "input *
/// factor + bias", where `factor = 1/std` and the `1/255` rescale is
/// pre-applied by callers that pass `mean/255` and `std/255`). The
/// python image-processor surface (`mlx_vlm`) breaks rescale out as a
/// separate step (the HF `BaseImageProcessor.rescale` contract); we
/// expose it as its own primitive for that parity, and [`preprocess`]
/// composes it before [`normalize_imagenet`].
///
/// **Dtype requirement:** `arr` must be a floating-point dtype
/// (`F16` / `BF16` / `F32` / `F64`). The swift reference's CIFilter
/// colorMatrix only operates on float pixel buffers
/// (`MediaProcessing.swift:171` always renders `CIFormat.RGBAf`) and
/// the python `BaseImageProcessor.rescale` converts to f32 before
/// multiplying; rescaling a u8/i32 input by a sub-unit factor in the
/// input dtype would silently floor to zero (e.g.
/// `astype(1/255, U8) = 0`). Non-float inputs are rejected with
/// [`Error::ShapeMismatch`].
///
/// Returns a *new* array; the source is unchanged (mlx's standard
/// out-of-place op semantics).
pub fn rescale(arr: &Array, scale: f32) -> Result<Array> {
  let dtype = arr.dtype()?;
  require_float_dtype("rescale", dtype)?;
  // Build a `(1,)` f32 scalar in the input's dtype so an f16/bf16/f64
  // input is not silently promoted to f32 (same dtype-fidelity
  // discipline as `embeddings::scalar_like`). For an f32 input this is
  // a no-op cast. Non-float inputs are rejected above.
  let s = Array::full::<f32>(&(1,), scale)?;
  let s = astype(&s, dtype)?;
  multiply(arr, &s)
}

/// Per-channel ImageNet-style normalization: `(x - mean[c]) / std[c]`.
///
/// `arr` shape: `[..., 3]` (channel-last). The mean/std tuples are
/// broadcast across all leading dims by reshaping them to `[1, 1, 3]`
/// (when `arr` is `[H, W, 3]`) — generally `arr.ndim() - 1` leading
/// 1-dims so the broadcast applies cleanly regardless of batch axis.
///
/// Mirrors swift `MediaProcessing.normalize` (`MediaProcessing.swift:
/// 135-157`): the swift implementation expresses
/// `(x - mean) / std` via the CIFilter colorMatrix
/// "input * (1/std) + (-mean/std)" trick (algebraically equivalent — see
/// the swift comment block lines 142-148). We use the direct subtract +
/// divide for readability; the math is the same.
///
/// **Dtype requirement:** `arr` must be a floating-point dtype
/// (`F16` / `BF16` / `F32` / `F64`). ImageNet mean/std values are
/// sub-unit f32s; casting them to an integer dtype would floor every
/// channel to zero and division by `astype(0.229, U8) = 0` would be
/// undefined. Both reference implementations always normalize in
/// float — swift `MediaProcessing.normalize` operates on the f32
/// CIFormat.RGBAf buffer (`MediaProcessing.swift:135-156`) and the
/// HF python `BaseImageProcessor.normalize` converts to f32 before
/// the subtract / divide. Non-float inputs are rejected with
/// [`Error::ShapeMismatch`].
///
/// **Dtype fidelity (float inputs):** the mean/std arrays adopt the
/// input dtype (so an f16/bf16 input is not silently promoted to f32),
/// matching the embeddings crate's `scalar_like` discipline.
pub fn normalize_imagenet(arr: &Array, mean: &[f32; 3], std: &[f32; 3]) -> Result<Array> {
  let ndim = arr.ndim();
  if ndim == 0 {
    return Err(Error::ShapeMismatch {
      message: "normalize_imagenet: input must have at least 1 dimension".into(),
    });
  }
  // Validate trailing channel dim == 3 with a clear error before falling
  // through to mlx's less-friendly broadcast failure.
  let shape = arr.shape();
  let trailing = shape[ndim - 1];
  if trailing != 3 {
    return Err(Error::ShapeMismatch {
      message: format!("normalize_imagenet: trailing dim must be 3 (RGB), got shape {shape:?}"),
    });
  }
  let dtype = arr.dtype()?;
  require_float_dtype("normalize_imagenet", dtype)?;
  // Build (3,) mean and std arrays in the input dtype, then reshape to
  // [1, ..., 1, 3] so they broadcast over every leading axis of `arr`.
  let mean_arr = make_channel_broadcast(mean, ndim, dtype)?;
  let std_arr = make_channel_broadcast(std, ndim, dtype)?;
  let centered = subtract(arr, &mean_arr)?;
  divide(&centered, &std_arr)
}

/// Reject non-float dtypes for primitives that need fractional arithmetic.
///
/// The swift reference's CIFilter pipeline runs exclusively in float
/// space (CIFormat.RGBAf @ `MediaProcessing.swift:171`); the python
/// HF processors call `array.astype(np.float32)` before rescale /
/// normalize. We surface the dtype mismatch as a clean
/// `Error::ShapeMismatch` rather than letting the caller discover it
/// as silent zeros downstream.
fn require_float_dtype(op: &str, dtype: Dtype) -> Result<()> {
  match dtype {
    Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => Ok(()),
    other => Err(Error::ShapeMismatch {
      message: format!(
        "{op}: input must be a floating-point dtype (F16/BF16/F32/F64), got {other:?}; \
         convert with `astype(arr, Dtype::F32)?` before calling"
      ),
    }),
  }
}

/// Build a `[1, ..., 1, 3]`-shaped broadcast tensor from a length-3 f32
/// slice, cast to `dtype`. Helper for [`normalize_imagenet`].
fn make_channel_broadcast(vals: &[f32; 3], ndim: usize, dtype: Dtype) -> Result<Array> {
  // 1-D (3,) constant in f32, then astype to the input dtype.
  let a = Array::from_slice(vals, &(3usize,))?;
  let a = astype(&a, dtype)?;
  // Reshape to [1, ..., 1, 3] (ndim-1 leading 1-dims + the channel axis).
  // For ndim == 1 this is a no-op reshape back to (3,).
  if ndim <= 1 {
    return Ok(a);
  }
  // Build the target shape on the stack via a 16-dim ceiling: mlx
  // arrays are bounded well below 16 dims in practice (CLIP/SigLIP/
  // patchify all stay <= 5), and a stack buffer avoids the Vec
  // allocation per `feedback_allocation_discipline`. If a caller ever
  // hands us > 16 dims, the explicit guard below converts cleanly.
  const MAX_NDIM: usize = 16;
  if ndim > MAX_NDIM {
    return Err(Error::ShapeMismatch {
      message: format!(
        "normalize_imagenet: input ndim {ndim} exceeds supported maximum {MAX_NDIM}"
      ),
    });
  }
  let mut buf = [1usize; MAX_NDIM];
  buf[ndim - 1] = 3;
  reshape(&a, &&buf[..ndim])
}

/// Patchify `[H, W, C]` into `[H/p * W/p, p, p, C]` (ViT-style flat
/// patch sequence).
///
/// Mirrors the patchification step shared by every ViT-class VLM
/// encoder. The swift `MediaProcessing` module does not expose a
/// dedicated `patchify` helper (per-model image processors in
/// `MLXVLM/Models/*` perform their own patch extraction); we expose it
/// here as a *uniform-grid* primitive because every model that needs
/// patches needs at least this baseline transform, and exposing it as a
/// primitive keeps the per-model processor a thin caller. The
/// `transformers`-style `patchify` and `mlx-vlm`'s per-model patch
/// extractors are aspect-ratio-aware variants that are out of scope
/// (per-usecase per the no-per-model-arch rule).
///
/// Returns `Err(Error::ShapeMismatch)` if the input is not rank-3,
/// `patch_size == 0`, or `H % p != 0 || W % p != 0`.
///
/// Layout: input `[H, W, C]` → reshape `[H/p, p, W/p, p, C]` →
/// transpose `[H/p, W/p, p, p, C]` → reshape `[H/p * W/p, p, p, C]`.
pub fn patchify(arr: &Array, patch_size: usize) -> Result<Array> {
  if arr.ndim() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "patchify: input must be rank-3 [H, W, C], got ndim {}",
        arr.ndim()
      ),
    });
  }
  if patch_size == 0 {
    return Err(Error::ShapeMismatch {
      message: "patchify: patch_size must be > 0".into(),
    });
  }
  let shape = arr.shape();
  let h = shape[0];
  let w = shape[1];
  let c = shape[2];
  if !h.is_multiple_of(patch_size) || !w.is_multiple_of(patch_size) {
    return Err(Error::ShapeMismatch {
      message: format!(
        "patchify: H={h} and W={w} must both be divisible by patch_size {patch_size}"
      ),
    });
  }
  let hp = h / patch_size;
  let wp = w / patch_size;
  // Checked multiply for the final-stage shape — a hostile `(H, W,
  // patch_size)` could overflow `usize` on the `hp * wp` product on a
  // 32-bit target (or, with extreme inputs, on a 64-bit target via
  // genuinely-large images). Surface as recoverable
  // `Error::ShapeMismatch` rather than silently wrapping to a
  // smaller-than-expected first axis (which would later cause
  // reshape/broadcast misalignment) — Copilot review #3272880077.
  let n_patches = hp.checked_mul(wp).ok_or_else(|| Error::ShapeMismatch {
    message: format!(
      "patchify: hp ({hp}) * wp ({wp}) overflows usize \
       (H={h}, W={w}, patch_size={patch_size})"
    ),
  })?;
  // [H, W, C] → [hp, p, wp, p, C]   (stack `[usize; 5]` buffer; no Vec
  // alloc per `feedback_allocation_discipline`)
  let stage1: [usize; 5] = [hp, patch_size, wp, patch_size, c];
  let r1 = reshape(arr, &&stage1[..])?;
  // → [hp, wp, p, p, C]   (move axis 2 ahead of axis 1)
  let t = transpose_axes(&r1, &[0, 2, 1, 3, 4])?;
  // → [hp * wp, p, p, C]
  reshape(&t, &(n_patches, patch_size, patch_size, c))
}

/// End-to-end preprocessing: optional resize → channel-last
/// `[H, W, 3]` f32 → optional `1/255` rescale → optional ImageNet
/// normalization.
///
/// Mirrors the swift `MediaProcessing` pipeline composition documented
/// in the module example (`MediaProcessing.swift:25-39`):
/// ```text
/// resample → normalize → asMLXArray
/// ```
/// We re-order to `resample → asMLXArray → rescale → normalize` because
/// our [`image_to_array`] returns `[0, 255]` f32 (not the `[0, 1]`
/// f32 the swift CIFilter normalize expects on its input — swift's
/// version pre-bakes the `1/255` rescale into the colorMatrix factor),
/// which keeps each primitive single-purpose and lets callers swap
/// individual stages. The composite math is identical:
/// `(x/255 - mean) / std` = `(x - 255*mean) / (255*std)`.
///
/// The output is channel-last `[H, W, 3]`. Per-model processors that
/// need planar `[C, H, W]` or batched `[1, C, H, W]` apply that as a
/// post-step (one lazy `reshape` + `transpose_axes` — see the module
/// `Conventions > Channel layout` block for the rationale).
// NOTE (Codex finding, round 2): a review request asked `preprocess`
// to return swift's `[1, C, H, W]` directly. This is intentionally
// rejected to respect the project's no-per-model-arch boundary
// (`feedback_no_per_model_arch_porting`): the cross-model primitive
// owns the layout-agnostic ImageNet pipeline, and per-model encoders
// pick the trailing layout in their own processor. The module doc
// explains why `[H, W, 3]` is the canonical primitive output and how
// callers reach `[1, C, H, W]` in one cheap lazy step.
pub fn preprocess(img: &::image::DynamicImage, cfg: &ImageProcessorConfig) -> Result<Array> {
  let resized;
  let src = if cfg.do_resize {
    resized = resize(img, cfg.size, cfg.resample);
    &resized
  } else {
    img
  };
  let arr = image_to_array(src, cfg.color_order)?;
  let arr = if cfg.do_rescale {
    rescale(&arr, cfg.rescale_factor)?
  } else {
    arr
  };
  if cfg.do_normalize {
    normalize_imagenet(&arr, &cfg.mean, &cfg.std)
  } else {
    Ok(arr)
  }
}
