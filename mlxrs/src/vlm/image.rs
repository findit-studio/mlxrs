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
//!   The swift
//!   [`inSRGBToneCurveSpace`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L50-L54)
//!   sRGB gamma conversion is also deferred: it is a piecewise nonlinear
//!   transform (`x <= 0.0031308 ? 12.92*x : 1.055*x^(1/2.4) - 0.055`),
//!   not a color-matrix multiply, and requires a `where`/conditional-select
//!   op that is not yet exposed in `mlxrs::ops`. The swift reference uses
//!   CoreImage's `CIFilter.linearToSRGBToneCurve()` (an Apple-only
//!   primitive); python `mlx-vlm` processors operate on already-decoded
//!   sRGB-tagged inputs and do not perform an explicit linear→sRGB step.
//!   When sRGB gamma is needed it can be added as a follow-up once the
//!   `where_` op is exposed.
//!
//! ## Conventions
//! - **Channel layout (opt-in planar via [`ImageProcessorConfig::layout`]):**
//!   [`image_to_array`] always emits channel-last `[H, W, 3]` (so the
//!   `(3,)` mean/std broadcast cleanly across the trailing axis without
//!   a layout-specific reshape in [`normalize`] / [`rescale`]). Alpha is
//!   intentionally dropped — see [`image_to_array`]'s doc for the swift
//!   `array[..., :3]` parity citation.
//!
//!   [`preprocess`] then applies a **trailing layout post-step** picked
//!   via [`ImageProcessorConfig::layout`] (a [`Layout`] enum defaulting
//!   to `Hwc` — pre-existing callers see no change):
//!   - `Hwc` (default): identity. The historical `[H, W, 3]` output.
//!   - `Chw`: one `transpose_axes(&[2, 0, 1])` → `[3, H, W]`. The
//!     torchvision / timm / classical-CV planar layout without a batch
//!     axis.
//!   - `Bchw`: same transpose + one `expand_dims_axes(&[0])` →
//!     `[1, 3, H, W]`. Matches swift `MediaProcessing.asMLXArray`
//!     (`MediaProcessing.swift:190` —
//!     `array.reshape((1, h, w, 3)).transposed(0, 3, 1, 2)`), the layout
//!     every `MLXVLM/Models/*.swift` ViT-class encoder consumes
//!     directly.
//!
//!   Both `Chw` / `Bchw` compose lazily on the [`Array`] (metadata
//!   update only — `transpose_axes` and `expand_dims_axes` do not copy
//!   the underlying buffer), so the planar arms have zero memory cost
//!   beyond the shape/stride bookkeeping. Per-model encoders that need
//!   a layout NOT in the three-arm enum (e.g. patchified `[N, P, P, C]`)
//!   keep the default `Hwc` output and compose their own trailing step;
//!   the [`Layout`] enum only enumerates the layouts ≥1 per-model
//!   encoder consumes verbatim today.
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
//!
//! ## P8 polish bundle closure (#120 / #121 / #122 / #123 / #124 / #125 / #126)
//!
//! - **VLM-1 (#120)** — opt-in planar layout: addressed via
//!   [`ImageProcessorConfig::layout`] (a [`Layout`] enum defaulting to
//!   `Hwc`, with `Chw` / `Bchw` for planar). Pre-existing callers see
//!   no change (`Hwc` is the identity arm); per-model encoders that
//!   want `[1, C, H, W]` (the swift
//!   `MediaProcessing.asMLXArray` shape) request `Bchw` and the
//!   composer applies one lazy `transpose_axes` + `expand_dims_axes` at
//!   zero memory cost. See [`Layout`] for the per-arm rationale +
//!   parity citations.
//! - **VLM-3 (#121)** — bulk-fill widen in [`image_to_array`]:
//!   addressed via the C3 (`rgb_widen`) / C4 (`bgr_widen`) NEON
//!   dispatchers wired into BOTH the `as_rgb8()` fast path AND the
//!   non-`Rgb8` per-pixel-projection path (the latter builds a
//!   contiguous `Vec<u8>` of length `H*W*3` then hands it to the same
//!   SIMD dispatcher — the per-pixel `push` overhead the issue called
//!   out is gone). See [`image_to_array`] for the per-branch wiring.
//! - **VLM-5 (#122)** — SIMD resize: closed via the own
//!   `vlm::resize` kernel (not `fast_image_resize`).
//!   The issue's original ask was to adopt `fast_image_resize`; the
//!   subsequent allocation-discipline audit showed `fast_image_resize`
//!   allocates its internal scratch (coefficient tables, per-row work
//!   buffers) infallibly inside the crate — so a near-budget hostile
//!   target could abort the process despite our `Result`. The own
//!   kernel runs the same SIMD shape on aarch64 AND routes every
//!   internal buffer through `try_reserve_exact`, so the
//!   recoverable-OOM contract is honest end-to-end. See [`resize`] for
//!   the per-buffer fallibility breakdown.
//! - **VLM-6 (#123)** — hand-written SIMD u8→f32 + BGR-swap:
//!   addressed via the C3 (`rgb_widen` — 16-byte tile `vld1q_u8` + 4×
//!   `vst1q_f32`) and C4 (`bgr_widen` — 16-pixel tile `vld3q_u8` +
//!   permuted `vst3q_f32`) NEON kernels. Both ship unconditionally on
//!   aarch64 per the project memory rule "SIMD ship NEON regardless"
//!   (the user directive 2026-05-23 overriding §5.4 of the SIMD doc).
//! - **VLM-4 (#124)** — fallible-allocation discipline in
//!   [`crate::vlm::prompt`]: every production allocation in the splice
//!   / mask / placeholder paths uses the crate-private
//!   `error::try_with_capacity` helper (i.e. `try_reserve_exact` under
//!   the hood); allocator failure surfaces as
//!   [`crate::error::Error::OutOfMemory`] instead of aborting. Grep
//!   `Vec::with_capacity` in `vlm/prompt.rs` to verify: the only
//!   matches are inline comments documenting what the new shape
//!   replaces.
//! - **VLM-2 (#125)** — byte-budget validation in [`resize`]:
//!   addressed via the target-dimension guard (`height * width * 4 <=
//!   MAX_DECODED_IMAGE_BYTES`) that fires BEFORE any allocation. The
//!   signature is `-> Result<DynamicImage>` — over-budget / zero /
//!   overflow targets surface as a typed `Error::ShapeMismatch` with
//!   the offending dims and the cap. See [`resize`]'s "Return type"
//!   doc paragraph for the full rationale.
//! - **VLM-7 (#126)** — `to_rgb8()` clone elision in
//!   [`image_to_array`]: addressed via the `as_rgb8()` fast path
//!   (borrows the source's backing `&[u8]` directly) + a per-pixel
//!   `dynamic_image_rgb_pixel` projection on the non-`Rgb8` branch
//!   (NO intermediate source-sized RGB image allocation). Grep
//!   `to_rgb8\|to_rgba8` in `vlm/image.rs` to verify: every match is
//!   in an inline comment documenting what the new path replaces.
//!
//! ## Allocation-fallibility audit (Codex round-5 closure)
//!
//! Every source-pixel-scaled allocation in this module is classified
//! below — the table is exhaustive (a `grep` of `to_rgb*` / `to_rgba*`
//! / `to_luma*` / `clone` / `rotate*` / `flip*` / `apply_orientation`
//! / `crop*` / `ImageBuffer::new` / `RgbImage::*` / the own resize
//! kernel's buffers). The audit splits the guarantee into TWO honest
//! columns rather than collapsing them into one ambiguous "fallible"
//! flag (round-5 finding):
//!
//! - **Bounded-memory:** an upper bound on the allocation's byte
//!   count is enforced before it runs — by [`MAX_DECODED_IMAGE_BYTES`]
//!   (load-time decoder cap, 512 MiB), applied to BOTH source-scale
//!   allocations and the [`resize`] destination ([`ImageProcessorConfig::size`]
//!   is now LOADED from an untrusted on-disk config, so [`resize`]
//!   validates its target against the same 512 MiB ceiling before
//!   allocating — see [`resize`]'s target-dimension guard).
//!   "Y" guarantees the call cannot trigger quadratic / unbounded
//!   allocation from hostile input.
//! - **Recoverable-OOM:** an allocator failure surfaces as a typed
//!   [`Error::OutOfMemory`] (or [`Error::ShapeMismatch`] on the
//!   pre-allocation overflow gate) rather than aborting the process.
//!   "Y" requires every backing allocation under our direct control
//!   to route through `try_reserve_exact` (or an equivalent fallible
//!   primitive). "N" means image-crate-internal allocations
//!   (`rotate90` / `ImageBuffer::new`) may panic-abort on allocator
//!   pressure even though the byte count itself is bounded. (The
//!   former `fast_image_resize`-internal scratch — coefficient tables,
//!   per-row work buffers — was such an "N"; it is now an "Y" because
//!   the resize is owned by `vlm::resize` and every kernel
//!   buffer is `try_reserve_exact`-backed.)
//!
//! | Site                                          | Scale         | Caller fn                | Bounded-memory | Recoverable-OOM | Notes                                       |
//! |-----------------------------------------------|---------------|--------------------------|----------------|-----------------|---------------------------------------------|
//! | `apply_orientation_fallible` u8 variants (Rotate90/270/+FlipH on Luma8/LumaA8/Rgb8/Rgba8) | source pixels | `load_image` →`Result` | Y (`MAX_DECODED_IMAGE_BYTES`) | **Y** | **FIXED (R5):** manual rotate over `try_reserve_exact`-backed buffer — no second alloc, no probe race |
//! | `apply_orientation_fallible` non-u8 variants (Luma16/LumaA16/Rgb16/Rgba16/Rgb32F/Rgba32F + rotates) | source pixels | `load_image` →`Result` | Y (`MAX_DECODED_IMAGE_BYTES`) | **Y** | **FIXED (R6):** covered by manual generic rotate via `try_reserve_exact` over `T: Copy` — 16-bit PNGs (`Luma16`/`LumaA16`/`Rgb16`/`Rgba16` per image-rs PNG decoder) and 32-bit-float `DynamicImage` inputs (`Rgb32F`/`Rgba32F`) now route through the same fallible per-element-type buffer path as u8 |
//! | `apply_orientation_fallible` (NoTransforms/Flip/Rot180, all variants) | in-place      | `load_image` →`Result`   | Y (in-place)   | Y (no alloc)    | Upstream `*_in_place` path — zero allocation |
//! | `resize` source RGBA buffer (`try_reserve_exact` + `as_rgba8` fast path / per-pixel `dynamic_image_rgba_pixel`) | source pixels | `resize` →`Result<DynamicImage>` | Y (`MAX_DECODED_IMAGE_BYTES` via `load_image`) | **Y** | **FIXED (Codex review R2):** replaced the infallible `img.to_rgba8()` clone with a `try_reserve_exact`-backed buffer filled manually (borrowed RGBA8 fast path or per-pixel projection); handed to the own `vlm::resize` kernel; allocator failure → `Error::OutOfMemory` |
//! | own resize kernel buffers — h+v coefficient tables, inter-pass intermediate, destination (`vlm::resize::resize_rgba8`) | target pixels (untrusted loaded config) | `resize` →`Result<DynamicImage>` | Y (`MAX_DECODED_IMAGE_BYTES` via `resize`'s target guard) | **Y** | **FIXED (own NEON resize, drop `fast_image_resize`):** every kernel buffer routes through `try_reserve_exact`; the target guard rejects zero/overflow/>512 MiB BEFORE any reservation; allocator failure → `Error::OutOfMemory`. Replaces `fast_image_resize`'s infallible internal scratch which could abort despite our `Result`. Output bit-exact with PIL `Image.resize`. |
//! | `img.clone()` (early-return in `center_crop`) | source pixels | `center_crop` →`DynamicImage` | Y (via `load_image` cap) | N (image-rs infallible `Vec::clone`) | OUT-OF-SCOPE: `-> DynamicImage` by reference parity |
//! | `img.crop_imm(...)` (in `center_crop`)        | min(source, target) | `center_crop` →`DynamicImage` | Y (≤ source bound) | N | OUT-OF-SCOPE: same parity rationale |
//! | `Vec::<u8>::try_reserve_exact` canvas (in `pad_to_square`) | target square (bounded) | `pad_to_square` →`Result` | Y (`MAX_DECODED_IMAGE_BYTES`) | Y (`try_reserve_exact` + `Error::OutOfMemory`) | FALLIBLE (R3) |
//! | `Vec::<f32>::try_reserve_exact` buf (in `image_to_array`) | source pixels (bounded by `load_image` cap) | `image_to_array` →`Result` | Y (via `load_image` cap) | Y (`try_reserve_exact` + `Error::OutOfMemory`) | FALLIBLE (R3) |
//! | `dynamic_image_rgb_pixel` / `dynamic_image_rgba_pixel` per-pixel `get_pixel` | none (stack `Rgb`/`Rgba<u8>` only) | shared helpers | Y (zero alloc) | Y (no alloc) | OK — no full-image intermediate alloc |
//! | mlx `Array` ops (rescale/normalize/patchify/preprocess) | output array | each `-> Result<Array>` | Y (output-shape) | Y (mlx backend `Result`) | OK — mlx backend allocator errors surface via `Array::*` `Result` |
//!
//! **Class closure invariant.** This module guarantees
//! **bounded-memory** end-to-end — every source-scale allocation is
//! capped by [`MAX_DECODED_IMAGE_BYTES`] (512 MiB), and the [`resize`]
//! destination (driven by the now-untrusted loaded
//! [`ImageProcessorConfig::size`]) is capped against the same ceiling
//! by [`resize`]'s target-dimension guard; no quadratic or unbounded
//! growth is possible from hostile input OR hostile config.
//! **Recoverable-OOM** is now guaranteed for every allocation under our
//! direct control: [`pad_to_square`]'s canvas, [`image_to_array`]'s
//! f32 buffer, [`resize`]'s source RGBA buffer AND every buffer the own
//! resize kernel allocates (the horizontal + vertical coefficient
//! tables, the inter-pass intermediate, and the destination — see
//! `vlm::resize`), and the rotate buffer for every
//! `DynamicImage` element type (u8 / u16 / f32) in
//! `apply_orientation_fallible` (private helper called by [`load_image`])
//! — the generic-rotate path covers 16-bit-PNG-derived
//! `Luma16`/`LumaA16`/`Rgb16`/`Rgba16` (image-rs 0.25's PNG decoder
//! emits 16-bit variants for `BitDepth::Sixteen` PNG inputs) and
//! caller-supplied `Rgb32F`/`Rgba32F` inputs. [`resize`] also rejects an
//! over-budget / zero / overflowing target as a typed
//! [`Error::ShapeMismatch`] BEFORE allocating (its target now flows from
//! an untrusted on-disk config), then materializes its source RGBA
//! buffer (formerly an infallible `img.to_rgba8()` clone) via
//! `try_reserve_exact` and hands it to the own `vlm::resize`
//! kernel — which dropped the `fast_image_resize` dependency precisely
//! because that crate allocated its internal scratch infallibly and
//! could abort despite our `Result`. So a just-under-cap hostile target
//! can no longer force a ~512 MiB infallible alloc anywhere in the
//! resize path, and `resize`'s `Result` signature is honest. The
//! ONLY remaining sites that keep an infallible image-crate `Vec`
//! allocator are the `-> DynamicImage` by-reference-parity helpers
//! (`center_crop`'s early-return `img.clone()` / `crop_imm`), whose byte
//! counts are still bounded ≤512 MiB (source by the decoder cap) so the
//! only abort path is a genuine system-wide OOM at a ≤512 MiB request;
//! their public docstrings document this residual, and their signatures
//! stay `-> DynamicImage` per `feedback_match_official_binding_design`.

use crate::{
  Dtype,
  array::Array,
  error::{
    ArithmeticOverflowPayload, CapExceededPayload, DivisibilityConstraintPayload, Error,
    FileIoPayload, FileOp, LengthMismatchPayload, OutOfRangePayload, ParsePayload,
    RankMismatchPayload, Result, UnknownEnumValuePayload, UnsupportedDtypePayload,
  },
  ops::{
    arithmetic::{divide, multiply, subtract},
    misc::astype,
    shape::{reshape, transpose_axes},
  },
};

/// Upper bound on decoded RGB pixel-buffer size accepted by host-side
/// allocators in this module (e.g. [`pad_to_square`]'s `size × size × 3`
/// canvas). Matches `image::Limits::default().max_alloc = 512 * 1024 *
/// 1024` (the same 512 MiB ceiling [`load_image`] enforces via
/// `Limits::default().reserve(decoder.total_bytes())?` — see the
/// `Allocation guard` block in [`load_image`]'s doc). Exposing a single
/// shared constant here keeps the per-step caps consistent: a
/// `DynamicImage` that fit through `load_image` still has to clear this
/// gate before any quadratic-canvas builder allocates.
pub const MAX_DECODED_IMAGE_BYTES: u64 = 512 * 1024 * 1024;

/// Interpolation filter for [`resize`], mirroring swift
/// `MediaProcessing.swift`'s resampler choices (lines 81-132):
/// [`resampleLanczos`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L81-L103)
/// and
/// [`resampleBicubic`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L110-L132).
///
/// Backed by mlxrs's own fully-fallible, PIL-matching resize kernel
/// (`vlm::resize`) — bit-exact with PIL `Image.resize` (the
/// reference mlx-vlm preprocessing targets). Filter names match PIL's
/// `Image.NEAREST` / `BILINEAR` / `BICUBIC` / `LANCZOS` resampling
/// filters, so existing call-site usage of `Bilinear` / `Bicubic` /
/// `Lanczos3` / `Nearest` is unchanged.
///
/// The swift reference exposes `bicubic` (default) and `lanczos`; we add
/// `Nearest` and `Bilinear` because they appear in the python VLM ecosystem
/// (PIL's `Image.resize` `resample=` argument that `mlx-vlm`'s
/// `resize_image` uses transitively at `mlx_vlm/utils.py:835-839`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum ResizeFilter {
  /// Nearest-neighbor (no smoothing). Cheapest; rarely used for VLM.
  /// PIL `Image.NEAREST`.
  Nearest,
  /// Bilinear interpolation (triangle kernel, support 1.0). PIL
  /// `Image.BILINEAR`.
  Bilinear,
  /// Bicubic interpolation (Keys cubic `a = -0.5`, support 2.0) — matches
  /// PIL's `Image.BICUBIC`. Mirrors the swift `resampleBicubic` default
  /// (`MediaProcessing.swift:110-132`); the recommended choice for most
  /// ViT-class encoders.
  Bicubic,
  /// Lanczos3 interpolation (window=3 sinc-windowed sinc, support 3.0) —
  /// matches PIL's `Image.LANCZOS`. Mirrors the swift `resampleLanczos`
  /// (`MediaProcessing.swift:81-103`).
  Lanczos3,
}

impl ResizeFilter {
  /// Lowercase string tag matching PIL resampling filter names.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Nearest => "nearest",
      Self::Bilinear => "bilinear",
      Self::Bicubic => "bicubic",
      Self::Lanczos3 => "lanczos3",
    }
  }
}

impl ResizeFilter {
  /// Map to the crate-internal `vlm::resize::Filter` driven by
  /// the own fallible resize kernel. Kept private; the internal filter
  /// type does not leak into the public surface. The mapping is 1:1 and
  /// each variant matches the identically-named PIL resampling filter
  /// (see `vlm::resize` for the per-filter PIL parity notes).
  fn to_internal(self) -> crate::vlm::resize::Filter {
    use crate::vlm::resize::Filter;
    match self {
      Self::Nearest => Filter::Nearest,
      Self::Bilinear => Filter::Bilinear,
      Self::Bicubic => Filter::Bicubic,
      Self::Lanczos3 => Filter::Lanczos3,
    }
  }
}

/// Channel layout for [`image_to_array`]. `RGB` is the swift default
/// (`MediaProcessing.swift:171` — `CIFormat.RGBAf`'s RGBA channel order);
/// `BGR` is exposed for parity with python image-processor configs that
/// use OpenCV-style BGR (e.g. some older CLIP variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum ColorOrder {
  /// Red-Green-Blue (the default; matches PIL / swift CoreImage).
  Rgb,
  /// Blue-Green-Red (OpenCV-style; swap R↔B).
  Bgr,
}

impl ColorOrder {
  /// Lowercase string tag matching Python color-order convention.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Rgb => "rgb",
      Self::Bgr => "bgr",
    }
  }
}

/// Trailing tensor layout applied by [`preprocess`] AFTER the
/// channel-last `[H, W, 3]` ImageNet pipeline (resize → widen → rescale
/// → normalize) completes.
///
/// **Why an enum, not a per-model post-step.** The cross-model primitive
/// always emits `[H, W, 3]` from [`image_to_array`] (so the `(3,)`
/// mean/std broadcasts cleanly across the trailing axis without a
/// layout-specific reshape — see the module-doc `Conventions > Channel
/// layout` block for the rationale). The downstream per-model encoder
/// then needs one of three concrete layouts depending on its
/// architecture:
///
/// - `Hwc` (`[H, W, 3]`): identity. The historical [`preprocess`]
///   output; the natural ImageNet-pipeline layout. Default for source
///   compatibility — pre-existing callers see no change.
/// - `Chw` (`[3, H, W]`): the planar layout torchvision / timm /
///   classical-CV stack uses without a batch axis. One
///   `transpose_axes(&[2, 0, 1])` over the lazy `Array` — metadata
///   update only, no buffer copy.
/// - `Bchw` (`[1, 3, H, W]`): the planar+batch layout swift
///   `MediaProcessing.asMLXArray` produces (`MLXVLM/MediaProcessing.swift:190`,
///   `array.reshape(1, h, w, 3).transposed(0, 3, 1, 2)`) for direct
///   ViT-encoder feed. One `transpose_axes` plus one
///   `expand_dims_axes(&[0])` over the lazy `Array`.
///
/// The per-model encoder picks the layout via
/// [`ImageProcessorConfig::layout`]; the cross-model primitive owns the
/// trailing post-step so the per-model code does NOT manually compose
/// the transpose + expand at every call site (and so a future variant —
/// e.g. a `[H, W, 3, 1]` patch grid — only adds one new arm here).
///
/// **Cost.** The post-step composes lazily on the [`Array`] — both
/// [`transpose_axes`] and `expand_dims_axes` update strides/shape
/// metadata without copying the underlying buffer (mlx's standard
/// no-op view semantics). The `Bchw` and `Chw` arms therefore have
/// **zero memory cost** beyond the metadata update.
///
/// **Parity citations.**
/// - swift `MediaProcessing.asMLXArray` line 190 — planar `[1, C, H, W]`
///   via `reshape((1, h, w, 3))` + `transpose_axes(&[0, 3, 1, 2])`;
///   this is the `Bchw` arm verbatim.
/// - python `mlx_vlm` per-model processors (`siglip` /
///   `clip_image_processor`) call `np.transpose(arr, (2, 0, 1))` after
///   the ImageNet pipeline for the `Chw` arm.
///
/// Tracking issue: [#120](https://github.com/Findit-AI/mlxrs/issues/120)
/// (VLM-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
pub enum Layout {
  /// Channel-last `[H, W, 3]`. Identity post-step — the historical
  /// [`preprocess`] output, kept as default for source compatibility.
  Hwc,
  /// Planar `[3, H, W]`. One `transpose_axes(&[2, 0, 1])`; no batch axis.
  Chw,
  /// Planar batched `[1, 3, H, W]`. Matches swift's
  /// `MediaProcessing.asMLXArray` (`MediaProcessing.swift:190`).
  Bchw,
}

impl Layout {
  /// Lowercase string tag.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Hwc => "hwc",
      Self::Chw => "chw",
      Self::Bchw => "bchw",
    }
  }
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
  size: (u32, u32),
  /// Per-channel mean for [`normalize_imagenet`].
  mean: [f32; 3],
  /// Per-channel std for [`normalize_imagenet`].
  std: [f32; 3],
  /// Multiplier applied by [`rescale`] (typically `1.0 / 255.0`).
  rescale_factor: f32,
  /// Whether the [`preprocess`] composer applies [`resize`].
  do_resize: bool,
  /// Whether the [`preprocess`] composer applies [`rescale`].
  do_rescale: bool,
  /// Whether the [`preprocess`] composer applies [`normalize_imagenet`].
  do_normalize: bool,
  /// Interpolation filter forwarded to [`resize`].
  resample: ResizeFilter,
  /// Channel layout forwarded to [`image_to_array`].
  color_order: ColorOrder,
  /// Trailing tensor layout applied by [`preprocess`] after the
  /// ImageNet pipeline. Default [`Layout::Hwc`] preserves the
  /// historical channel-last `[H, W, 3]` output for source
  /// compatibility; per-model encoders that consume planar
  /// `[1, C, H, W]` (the swift `MediaProcessing.asMLXArray` shape) can
  /// opt in via [`Layout::Bchw`] without a manual call-site transpose.
  /// See [`Layout`] for the full per-arm rationale + cost analysis
  /// (zero copy — lazy `Array` metadata update only).
  ///
  /// Tracking issue: [#120](https://github.com/Findit-AI/mlxrs/issues/120)
  /// (VLM-1).
  layout: Layout,
}

impl Default for ImageProcessorConfig {
  fn default() -> Self {
    Self::new()
  }
}

impl ImageProcessorConfig {
  /// ImageNet defaults: `size = (224, 224)`, `mean = [0.485, 0.456,
  /// 0.406]`, `std = [0.229, 0.224, 0.225]`, `rescale_factor = 1/255`,
  /// `resample = Bicubic`, `color_order = Rgb`, `layout = Hwc`, all
  /// `do_*` flags `true`. These are the values nearly every CLIP /
  /// SigLIP / DINO / ViT preprocessing config ships with. `Hwc` is the
  /// historical [`preprocess`] output — pre-existing callers see no
  /// change; opt into [`Layout::Bchw`] / [`Layout::Chw`] explicitly when
  /// the per-model encoder wants planar layout.
  pub fn new() -> Self {
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
      layout: Layout::Hwc,
    }
  }

  // ── builders ──────────────────────────────────────────────────────────────

  /// Set the target image size `(height, width)`.
  #[must_use]
  pub fn with_size(mut self, v: (u32, u32)) -> Self {
    self.size = v;
    self
  }
  /// Set the per-channel mean for normalization.
  #[must_use]
  pub fn with_mean(mut self, v: [f32; 3]) -> Self {
    self.mean = v;
    self
  }
  /// Set the per-channel std for normalization.
  #[must_use]
  pub fn with_std(mut self, v: [f32; 3]) -> Self {
    self.std = v;
    self
  }
  /// Set the rescale factor (typically `1.0 / 255.0`).
  #[must_use]
  pub fn with_rescale_factor(mut self, v: f32) -> Self {
    self.rescale_factor = v;
    self
  }
  /// Set the `do_resize` flag.
  #[must_use]
  pub fn with_do_resize(mut self, v: bool) -> Self {
    self.do_resize = v;
    self
  }
  /// Set the `do_rescale` flag.
  #[must_use]
  pub fn with_do_rescale(mut self, v: bool) -> Self {
    self.do_rescale = v;
    self
  }
  /// Set the `do_normalize` flag.
  #[must_use]
  pub fn with_do_normalize(mut self, v: bool) -> Self {
    self.do_normalize = v;
    self
  }
  /// Set the interpolation filter.
  #[must_use]
  pub fn with_resample(mut self, v: ResizeFilter) -> Self {
    self.resample = v;
    self
  }
  /// Set the channel order.
  #[must_use]
  pub fn with_color_order(mut self, v: ColorOrder) -> Self {
    self.color_order = v;
    self
  }
  /// Set the trailing tensor layout.
  #[must_use]
  pub fn with_layout(mut self, v: Layout) -> Self {
    self.layout = v;
    self
  }

  // ── accessors ─────────────────────────────────────────────────────────────

  /// Target image size `(height, width)`.
  #[inline(always)]
  pub fn size(&self) -> (u32, u32) {
    self.size
  }
  /// Per-channel mean for normalization.
  #[inline(always)]
  pub fn mean(&self) -> [f32; 3] {
    self.mean
  }
  /// Per-channel std for normalization.
  #[inline(always)]
  pub fn std(&self) -> [f32; 3] {
    self.std
  }
  /// Rescale factor (typically `1.0 / 255.0`).
  #[inline(always)]
  pub fn rescale_factor(&self) -> f32 {
    self.rescale_factor
  }
  /// Whether [`preprocess`] applies [`resize`].
  #[inline(always)]
  pub fn do_resize(&self) -> bool {
    self.do_resize
  }
  /// Whether [`preprocess`] applies [`rescale`].
  #[inline(always)]
  pub fn do_rescale(&self) -> bool {
    self.do_rescale
  }
  /// Whether [`preprocess`] applies [`normalize_imagenet`].
  #[inline(always)]
  pub fn do_normalize(&self) -> bool {
    self.do_normalize
  }
  /// Interpolation filter for [`resize`].
  #[inline(always)]
  pub fn resample(&self) -> ResizeFilter {
    self.resample
  }
  /// Channel order.
  #[inline(always)]
  pub fn color_order(&self) -> ColorOrder {
    self.color_order
  }
  /// Trailing tensor layout.
  #[inline(always)]
  pub fn layout(&self) -> Layout {
    self.layout
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
///
/// **EXIF rotate gate (Codex rounds 5–6).** The EXIF orientation step
/// is routed through the private `apply_orientation_fallible` helper.
/// The decoder-side `set_limits` cap above does NOT protect the rotate
/// variants (`Rotate90` / `Rotate270` / `Rotate90FlipH` /
/// `Rotate270FlipH`) which each need a NEW source-sized buffer — the
/// decoder has already been consumed by `from_decoder` at that point.
/// The fallible helper allocates that buffer directly via
/// `Vec::try_reserve_exact` and performs a manual rotation into it via
/// the generic `rotate_buf<T: Copy + Default>` helper, which covers
/// every `image` 0.25 pixel variant: u8 (`Luma8` / `LumaA8` / `Rgb8` /
/// `Rgba8`), u16 (`Luma16` / `LumaA16` / `Rgb16` / `Rgba16` — image-rs
/// 0.25's PNG decoder emits these for 16-bit-per-channel PNGs per
/// `codecs/png.rs:64-71`, and PNG decoders expose EXIF orientation
/// via `ImageDecoder::orientation`), and f32 (`Rgb32F` / `Rgba32F`).
/// Allocator failure surfaces as [`Error::OutOfMemory`] before the
/// pixel-copy loop runs — no probe-then-delegate race, no infallible
/// `apply_orientation` fallback. In-place variants (`NoTransforms` /
/// `FlipHorizontal` / `FlipVertical` / `Rotate180`) pass through with
/// zero allocation. See the helper's doc for the full rationale and
/// the module-level audit table for the per-variant bounded-memory vs
/// recoverable-OOM contract.
pub fn load_image(path: &std::path::Path) -> Result<::image::DynamicImage> {
  // `ImageDecoder` is the trait that provides `.orientation()`; pull it
  // into local scope so the method resolves on the opaque decoder type
  // returned by `into_decoder`.
  use ::image::ImageDecoder as _;

  let parse_err =
    |e: ::image::ImageError| Error::Parse(ParsePayload::new("vlm::image::load_image", "image", e));
  let io_err = |e: std::io::Error| {
    Error::FileIo(FileIoPayload::new(
      "vlm::image::load_image",
      FileOp::Read,
      path.to_path_buf(),
      e,
    ))
  };
  // ImageReader::open guesses the format from the path extension; we
  // then call `with_guessed_format` to fall back to content sniffing
  // for extension-less paths (mirroring python `Image.open` which
  // sniffs the file header).
  let reader = ::image::ImageReader::open(path)
    .map_err(io_err)?
    .with_guessed_format()
    .map_err(io_err)?;
  let mut decoder = reader.into_decoder().map_err(parse_err)?;
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
  // formats that don't carry orientation metadata. With the current
  // `image` features (`png` + `jpeg`; TIFF/WebP NOT in the build,
  // `mlxrs/Cargo.toml`) BOTH formats may expose EXIF orientation:
  // JpegDecoder parses APP1/Exif, and image 0.25 PngDecoder exposes
  // `exif_metadata` which the default `ImageDecoder::orientation`
  // parses — so 16-bit PNGs with EXIF `Rotate90`/`Rotate270` reach
  // the rotate path here too (Codex review R8 — was previously
  // documented as JPEG-only, false for `image` 0.25.10). All rotate
  // orientations are handled by `apply_orientation_fallible` over
  // `rotate_buf<T>` covering u8/u16/f32 pixel variants. We read
  // orientation here while we still have a `&mut` borrow on the
  // decoder; once it's consumed by `from_decoder` below, the
  // metadata can no longer be queried.
  let orientation = decoder.orientation().map_err(parse_err)?;
  // Preserve the 512 MiB default allocation guard that
  // `ImageReader::decode()` enforces. Our use of `into_decoder` (so
  // we can read orientation) skips the `limits.reserve(total_bytes)`
  // check `decode()` does internally — see image 0.25
  // `io::image_reader_type::ImageReader::decode`. Mirror that check
  // explicitly so an oversized image is rejected with a clean
  // `Error::Backend` instead of running through the decoder and
  // panic-allocating downstream.
  let mut limits = ::image::Limits::default();
  limits.reserve(decoder.total_bytes()).map_err(parse_err)?;
  decoder.set_limits(limits).map_err(parse_err)?;
  let img = ::image::DynamicImage::from_decoder(decoder).map_err(parse_err)?;
  apply_orientation_fallible(img, orientation)
}

/// Apply EXIF `orientation` to `img` with a *truly recoverable*
/// allocator gate on the four allocating rotate variants
/// (`Rotate90` / `Rotate270` / `Rotate90FlipH` / `Rotate270FlipH`),
/// for **every** `DynamicImage` element type — `u8`
/// (Luma8 / LumaA8 / Rgb8 / Rgba8), `u16`
/// (Luma16 / LumaA16 / Rgb16 / Rgba16), and `f32`
/// (Rgb32F / Rgba32F).
///
/// **The defect class this closes (Codex round-5 → round-6 finding).**
/// Round 4 used a "probe-then-delegate" pattern (reserve a throwaway
/// `Vec<u8>`, drop it, then call image-rs's `apply_orientation`) which
/// is race-prone — allocator pressure between the probe drop and the
/// real `rotate90` / `rotate270` alloc can flip the result from "would
/// succeed" to "aborts". Round 5 replaced the u8 hot path with a
/// *manual* rotation that builds the rotated buffer INSIDE a
/// `try_reserve_exact`-backed `Vec<u8>` (no second alloc, no probe
/// drop, no race window). Round 5 left the non-u8 variants on the
/// infallible image-rs delegate, claiming they could not occur via
/// [`load_image`]; that claim was wrong — image-rs 0.25's PNG decoder
/// emits `Luma16` / `LumaA16` / `Rgb16` / `Rgba16` for 16-bit PNG
/// inputs (`codecs/png.rs:64-71`), so a 16-bit PNG carrying an EXIF
/// `Rotate90` / `Rotate270` orientation could reach the infallible
/// fallback from [`load_image`] and abort on allocator pressure.
/// Round 6 generalizes the manual rotate over the element type
/// `T: Copy` (see [`rotate_buf`]) so the u16 and f32 variants share
/// the same `try_reserve_exact` gate as u8. Allocator failure now
/// surfaces as [`Error::OutOfMemory`] exactly once for every
/// `DynamicImage` variant, before the pixel-copy loop runs.
///
/// **Dispatch.**
/// - **All rotate orientations × all element types** (u8 / u16 / f32):
///   manual per-pixel rotation into a `Vec<T>` whose backing buffer
///   was reserved via `try_reserve_exact`. Rebuild a `DynamicImage`
///   via `ImageBuffer::from_raw`. Recoverable-OOM guaranteed.
/// - **No-op / in-place variants** (NoTransforms, FlipHorizontal,
///   FlipVertical, Rotate180) on any variant: zero source-sized
///   alloc — upstream `apply_orientation` dispatches to
///   `*_in_place` helpers. Recoverable-OOM trivially (no alloc).
///
/// **Rotation conventions** (cross-checked against
/// `image-0.25.10/src/imageops/affine.rs`):
/// - `rotate90` (clockwise 90°): for each `(x, y)` in source dims
///   `(w0, h0)`, writes to `(h0 - 1 - y, x)` in output dims
///   `(h0, w0)`.
/// - `rotate270` (clockwise 270°): writes `(x, y)` → `(y, w0 - 1 - x)`
///   in output dims `(h0, w0)`.
/// - `Rotate90FlipH`: rotate90 then `fliph_in_place`. Composed
///   directly into the destination via `(h0 - 1 - y, x)` then
///   `(width - 1 - new_x, new_y)`, which collapses to writing
///   `(y, x)` directly.
/// - `Rotate270FlipH`: rotate270 then `fliph_in_place`. Collapses
///   to writing `(h0 - 1 - y, w0 - 1 - x)`.
fn apply_orientation_fallible(
  mut img: ::image::DynamicImage,
  orientation: ::image::metadata::Orientation,
) -> Result<::image::DynamicImage> {
  use ::image::metadata::Orientation;
  match orientation {
    // No-op / in-place variants: zero source-sized alloc on any
    // pixel variant — upstream `apply_orientation` dispatches to
    // `*_in_place` helpers or returns immediately. See
    // `image::DynamicImage::apply_orientation` arms at
    // `images/dynimage.rs:1163-1180`:
    //   - NoTransforms → `()` (no-op)
    //   - FlipHorizontal → `fliph_in_place`
    //   - FlipVertical  → `flipv_in_place`
    //   - Rotate180     → `rotate180_in_place`
    Orientation::NoTransforms
    | Orientation::FlipHorizontal
    | Orientation::FlipVertical
    | Orientation::Rotate180 => {
      img.apply_orientation(orientation);
      Ok(img)
    }
    // Allocating variants — every `DynamicImage` element type
    // (u8 / u16 / f32) routes through the generic-rotate path in
    // `apply_rotate_fallible`. No infallible fallback remains.
    Orientation::Rotate90
    | Orientation::Rotate270
    | Orientation::Rotate90FlipH
    | Orientation::Rotate270FlipH => {
      let kind = match orientation {
        Orientation::Rotate90 => RotateKind::Rotate90,
        Orientation::Rotate270 => RotateKind::Rotate270,
        Orientation::Rotate90FlipH => RotateKind::Rotate90FlipH,
        Orientation::Rotate270FlipH => RotateKind::Rotate270FlipH,
        _ => unreachable!("outer match restricts to the 4 rotate variants"),
      };
      apply_rotate_fallible(img, kind)
    }
  }
}

/// Internal kind tag for the four EXIF rotation orientations that
/// allocate a fresh source-sized buffer. Carried as a typed enum (not
/// re-derived from `image::Orientation`) so the manual-rotation
/// dispatch in [`apply_rotate_fallible`] can `match` on a closed set.
#[derive(Debug, Clone, Copy)]
enum RotateKind {
  /// Clockwise 90° (transpose + flip on the horizontal axis).
  Rotate90,
  /// Clockwise 270° (transpose + flip on the vertical axis).
  Rotate270,
  /// `Rotate90` then horizontal flip.
  Rotate90FlipH,
  /// `Rotate270` then horizontal flip.
  Rotate270FlipH,
}

/// Manual fallible rotation for every [`DynamicImage`] element type
/// (u8: Luma8 / LumaA8 / Rgb8 / Rgba8; u16: Luma16 / LumaA16 / Rgb16
/// / Rgba16; f32: Rgb32F / Rgba32F). The rotated buffer is built
/// INSIDE a `try_reserve_exact`-backed `Vec<T>` (via [`rotate_buf`])
/// so an allocator failure surfaces as [`Error::OutOfMemory`]
/// exactly once, before the pixel-copy loop runs (no race window
/// like the round-4 probe, no infallible fallback to image-rs).
///
/// **Round-6 closure.** Round 5 left non-u8 variants on an infallible
/// `apply_orientation` fallback under the assumption they could not
/// be reached from [`load_image`]; that was wrong (image-rs 0.25's
/// PNG decoder emits 16-bit variants for `BitDepth::Sixteen` PNG
/// inputs — `codecs/png.rs:64-71`). Round 6 generalizes the rotate
/// over `T: Copy` so the same fallible path covers u16 and f32.
/// `DynamicImage` is `#[non_exhaustive]` upstream, so a wildcard
/// arm is required by the compiler; we surface any future-added
/// variant as a typed [`Error::Backend`] rather than silently
/// falling back to image-rs's infallible `apply_orientation` — the
/// "no abort on allocator pressure" contract holds for every
/// variant the helper handles today (the ten listed above), and a
/// new variant produces a recoverable `Err` until its rotate arm
/// is wired in.
fn apply_rotate_fallible(
  img: ::image::DynamicImage,
  rotation: RotateKind,
) -> Result<::image::DynamicImage> {
  use ::image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
  let w = u64::from(img.width());
  let h = u64::from(img.height());
  let bytes_per_pixel = u64::from(img.color().bytes_per_pixel());
  // `width * height * bytes_per_pixel`. u64 to keep the overflow
  // check host-arch-independent; `u32::MAX^2 * 16 ≈ 2.95e20` fits
  // in u64, so the `checked_mul` only fires for hostile dimensions.
  // The byte budget is `bytes_per_pixel`-based (not subpixel-count)
  // so 16-bit and f32 variants are bounded by their true memory
  // footprint, not by the channel count alone.
  let bytes = w
    .checked_mul(h)
    .and_then(|wh| wh.checked_mul(bytes_per_pixel))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "apply_rotate_fallible: rotated buffer size (w * h * bytes_per_pixel)",
        "u64",
        [("w", w), ("h", h), ("bytes_per_pixel", bytes_per_pixel)],
      ))
    })?;
  if bytes > MAX_DECODED_IMAGE_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "apply_rotate_fallible: rotated buffer",
      "MAX_DECODED_IMAGE_BYTES",
      MAX_DECODED_IMAGE_BYTES,
      bytes,
    )));
  }
  let src_w = img.width();
  let src_h = img.height();
  let (out_w, out_h) = rotated_dims(src_w, src_h, rotation);
  // Exhaustive per-variant dispatch into the generic
  // `rotate_buf<T: Copy>` helper. `channels` is the per-pixel
  // *subpixel count* (number of `T` elements per pixel), NOT a
  // byte count — so `Rgb<u16>` is 3 (three u16 subpixels per
  // pixel), `Rgba<f32>` is 4 (four f32 subpixels per pixel), etc.
  // `ImageBuffer::from_raw` validates the container length is
  // `width * height * CHANNEL_COUNT` in subpixel units, which
  // exactly matches the buffer we hand it.
  match img {
    DynamicImage::ImageLuma8(buf) => {
      let dst = rotate_buf::<u8>(buf.as_raw(), src_w, src_h, 1, rotation)?;
      let out: ::image::GrayImage = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: dst buffer length matches w*h*1 by construction in rotate_buf",
      );
      Ok(DynamicImage::ImageLuma8(out))
    }
    DynamicImage::ImageLumaA8(buf) => {
      let dst = rotate_buf::<u8>(buf.as_raw(), src_w, src_h, 2, rotation)?;
      let out: ::image::GrayAlphaImage = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: dst buffer length matches w*h*2 by construction in rotate_buf",
      );
      Ok(DynamicImage::ImageLumaA8(out))
    }
    DynamicImage::ImageRgb8(buf) => {
      let dst = rotate_buf::<u8>(buf.as_raw(), src_w, src_h, 3, rotation)?;
      let out: ::image::RgbImage = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: dst buffer length matches w*h*3 by construction in rotate_buf",
      );
      Ok(DynamicImage::ImageRgb8(out))
    }
    DynamicImage::ImageRgba8(buf) => {
      // C5 SIMD: Rgba8 (u8 + channels=4) is the hot path — dispatch
      // through the C5 NEON kernel (`simd::vlm::rotate_buf::rotate_buf_u8`)
      // which uses a 4-pixel-tile `vld1q_u8` load + per-pixel u32
      // scattered store. The other u8 channel counts (1/2/3) and
      // every u16/f32 arm continue to use the generic `rotate_buf<T>`.
      let dst = rotate_buf_u8_via_c5(buf.as_raw(), src_w, src_h, 4, rotation)?;
      let out: ::image::RgbaImage = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: dst buffer length matches w*h*4 by construction in rotate_buf",
      );
      Ok(DynamicImage::ImageRgba8(out))
    }
    DynamicImage::ImageLuma16(buf) => {
      let dst = rotate_buf::<u16>(buf.as_raw(), src_w, src_h, 1, rotation)?;
      let out: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: Luma16 dst length matches w*h*1 u16 subpixels by construction",
      );
      Ok(DynamicImage::ImageLuma16(out))
    }
    DynamicImage::ImageLumaA16(buf) => {
      let dst = rotate_buf::<u16>(buf.as_raw(), src_w, src_h, 2, rotation)?;
      let out: ImageBuffer<LumaA<u16>, Vec<u16>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: LumaA16 dst length matches w*h*2 u16 subpixels by construction",
      );
      Ok(DynamicImage::ImageLumaA16(out))
    }
    DynamicImage::ImageRgb16(buf) => {
      let dst = rotate_buf::<u16>(buf.as_raw(), src_w, src_h, 3, rotation)?;
      let out: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: Rgb16 dst length matches w*h*3 u16 subpixels by construction",
      );
      Ok(DynamicImage::ImageRgb16(out))
    }
    DynamicImage::ImageRgba16(buf) => {
      let dst = rotate_buf::<u16>(buf.as_raw(), src_w, src_h, 4, rotation)?;
      let out: ImageBuffer<Rgba<u16>, Vec<u16>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: Rgba16 dst length matches w*h*4 u16 subpixels by construction",
      );
      Ok(DynamicImage::ImageRgba16(out))
    }
    DynamicImage::ImageRgb32F(buf) => {
      let dst = rotate_buf::<f32>(buf.as_raw(), src_w, src_h, 3, rotation)?;
      let out: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: Rgb32F dst length matches w*h*3 f32 subpixels by construction",
      );
      Ok(DynamicImage::ImageRgb32F(out))
    }
    DynamicImage::ImageRgba32F(buf) => {
      let dst = rotate_buf::<f32>(buf.as_raw(), src_w, src_h, 4, rotation)?;
      let out: ImageBuffer<Rgba<f32>, Vec<f32>> = ImageBuffer::from_raw(out_w, out_h, dst).expect(
        "ImageBuffer::from_raw: Rgba32F dst length matches w*h*4 f32 subpixels by construction",
      );
      Ok(DynamicImage::ImageRgba32F(out))
    }
    // `DynamicImage` is `#[non_exhaustive]` upstream — a wildcard
    // arm is required by the compiler even though the ten arms
    // above cover every variant defined in image-rs 0.25. Surface
    // any future addition as a typed `Backend` error rather than
    // silently calling the infallible `apply_orientation` (which
    // reintroduces the abort-on-OOM behavior round 5/6 closes).
    // Recoverable by definition (it's an `Err`, not a panic), and
    // upgrading to a proper rotate arm becomes a localized edit
    // here the day a new variant ships.
    other => Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "apply_rotate_fallible: DynamicImage color variant (image-rs added a new pixel type that \
         mlxrs has not yet wired into the fallible rotate path — please file an issue and extend \
         the match arms above)",
      format!("{:?}", other.color()),
      &[
        "L8", "La8", "Rgb8", "Rgba8", "L16", "La16", "Rgb16", "Rgba16", "Rgb32F", "Rgba32F",
      ],
    ))),
  }
}

/// Rotated output dimensions for [`RotateKind`]. All four rotate
/// variants swap width and height (90°/270° rotation, with or
/// without a subsequent in-place horizontal flip that does not
/// change extent).
fn rotated_dims(src_w: u32, src_h: u32, _rotation: RotateKind) -> (u32, u32) {
  (src_h, src_w)
}

/// Manual rotation of a `T`-typed pixel buffer for the four EXIF
/// rotate orientations, into a `Vec<T>` whose backing storage is
/// reserved via `try_reserve_exact` (recoverable-OOM).
///
/// **Generic over the element type** (`T: Copy + Default`) so the
/// same body covers every `DynamicImage` element width:
/// - u8: Luma8 / LumaA8 / Rgb8 / Rgba8
/// - u16: Luma16 / LumaA16 / Rgb16 / Rgba16
/// - f32: Rgb32F / Rgba32F
///
/// Rotation is a *permutation of pixel positions*, not a per-channel
/// projection — the index math is identical for every element type;
/// only the per-pixel memcpy length (`channels` subpixels of `T`)
/// scales with the variant. `Copy` is required for the per-pixel
/// `copy_from_slice`; `Default` is required to zero-init the
/// reservation before the permutation writes land (cheap: u8/u16
/// default to 0, f32 to 0.0).
///
/// **Pixel conventions** (cross-checked against image-rs
/// `imageops/affine.rs` in `image-0.25.10`):
/// - `Rotate90` (CW 90°): `image[y][x] -> dst[x][h-1-y]`, i.e.
///   `dst.put_pixel(h-1-y, x, src.get_pixel(x, y))` with output
///   dims `(h, w)`. (Source matches `rotate90_in` at
///   `imageops/affine.rs:65-71`.)
/// - `Rotate270` (CW 270°): `dst.put_pixel(y, w-1-x, src.get_pixel(x, y))`
///   with output dims `(h, w)`. (Matches `rotate270_in` at
///   `imageops/affine.rs:108-114`.)
/// - `Rotate90FlipH`: rotate90 then horizontal flip — the rotated
///   x-coordinate `x_rot = h-1-y` becomes `(out_w-1) - x_rot
///   = (h-1) - (h-1-y) = y`. Output: `dst.put_pixel(y, x, src.get_pixel(x, y))`
///   with dims `(h, w)`.
/// - `Rotate270FlipH`: rotate270 then horizontal flip — the rotated
///   x-coordinate `x_rot = y` becomes `(out_w-1) - y = (h-1) - y`.
///   Output: `dst.put_pixel(h-1-y, w-1-x, src.get_pixel(x, y))`
///   with dims `(h, w)`.
///
/// `channels` is the per-pixel *subpixel count* (1 for Luma, 2 for
/// LumaA, 3 for Rgb, 4 for Rgba) — NOT a byte count. The element
/// stride per pixel is `channels * size_of::<T>()` bytes, but the
/// allocator and copy operate in `T`-element units, so `bytes` here
/// is the subpixel-element count for `Vec::try_reserve_exact::<T>`.
///
/// `src.len()` MUST equal `src_w * src_h * channels` (in `T` units);
/// the caller (the ten per-variant arms in [`apply_rotate_fallible`])
/// guarantees this via the `ImageBuffer::as_raw()` contract. The
/// byte budget has already been validated by the caller against
/// [`MAX_DECODED_IMAGE_BYTES`] (using true `bytes_per_pixel`), so
/// the subpixel-element count is well under `usize::MAX`.
fn rotate_buf<T: Copy + Default>(
  src: &[T],
  src_w: u32,
  src_h: u32,
  channels: usize,
  rotation: RotateKind,
) -> Result<Vec<T>> {
  let w_usize = src_w as usize;
  let h_usize = src_h as usize;
  // `elements = w * h * channels` (in `T` subpixel units; total
  // bytes = elements * size_of::<T>() is already validated <=
  // MAX_DECODED_IMAGE_BYTES by the caller). Recompute here in
  // usize (the cast is lossless on any 64-bit host) so the buffer
  // reservation has a precise length. Defend against an unexpected
  // overflow regardless — `try_reserve_exact` on a pathologically
  // large `usize` would still return Err, but the explicit overflow
  // check yields a typed `ShapeMismatch` rather than the less
  // specific `OutOfMemory`.
  let elements = w_usize
    .checked_mul(h_usize)
    .and_then(|wh| wh.checked_mul(channels))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "rotate_buf: elements (w * h * channels)",
        "usize",
        [
          ("w", w_usize as u64),
          ("h", h_usize as u64),
          ("channels", channels as u64),
        ],
      ))
    })?;
  debug_assert_eq!(
    src.len(),
    elements,
    "rotate_buf: src.len() must equal w*h*channels by ImageBuffer::as_raw() contract"
  );
  // Allocate the destination buffer fallibly. `try_reserve_exact`
  // returns Err on allocator failure rather than aborting; convert
  // to `Error::OutOfMemory`. We then `resize(elements, T::default())`
  // so the length matches and the slice indices below are valid. No
  // second allocation occurs (`Vec::resize` does not reallocate when
  // `len + n <= capacity`, which is the case immediately after a
  // `try_reserve_exact(elements)` from `len == 0`).
  let mut dst: Vec<T> = Vec::new();
  dst
    .try_reserve_exact(elements)
    .map_err(|_| Error::OutOfMemory)?;
  dst.resize(elements, T::default());
  let (out_w, _out_h) = rotated_dims(src_w, src_h, rotation);
  let out_w_usize = out_w as usize;
  // Iterate the source once. For each source pixel at (x, y), copy
  // its `channels` `T` elements into the destination position
  // implied by `rotation`. The inner per-pixel `copy_from_slice` is
  // a contiguous memcpy of `channels * size_of::<T>()` bytes (LLVM
  // unrolls / converts to single loads/stores for small fixed sizes
  // — same code shape as the u8-only round-5 path, just parameterized
  // by `T`).
  //
  // Index math (verified against `imageops/affine.rs` upstream):
  //   Rotate90       : (x, y) -> (h_usize - 1 - y, x)        out dims (h, w)
  //   Rotate270      : (x, y) -> (y, w_usize - 1 - x)        out dims (h, w)
  //   Rotate90FlipH  : (x, y) -> (y, x)                       out dims (h, w)
  //   Rotate270FlipH : (x, y) -> (h_usize - 1 - y, w_usize - 1 - x)  out dims (h, w)
  for y in 0..h_usize {
    for x in 0..w_usize {
      let (nx, ny) = match rotation {
        RotateKind::Rotate90 => (h_usize - 1 - y, x),
        RotateKind::Rotate270 => (y, w_usize - 1 - x),
        RotateKind::Rotate90FlipH => (y, x),
        RotateKind::Rotate270FlipH => (h_usize - 1 - y, w_usize - 1 - x),
      };
      let src_off = (y * w_usize + x) * channels;
      let dst_off = (ny * out_w_usize + nx) * channels;
      dst[dst_off..dst_off + channels].copy_from_slice(&src[src_off..src_off + channels]);
    }
  }
  Ok(dst)
}

/// C5 SIMD-routed u8 rotate helper — shares the same allocation /
/// length contract as [`rotate_buf`] but dispatches the inner two-loop
/// through [`crate::simd::vlm::rotate_buf::rotate_buf_u8`], whose
/// aarch64 kernel uses a 4-pixel-tile `vld1q_u8` + per-pixel u32
/// scattered store on the `channels = 4` (Rgba8) hot path.
///
/// `channels = 1/2/3` fall through to the C5 dispatcher's scalar arm —
/// bit-identical to [`rotate_buf::<u8>`]'s per-pixel `copy_from_slice`
/// shape. Only `Rgba8` (the dominant call site post-image-decode) hits
/// the NEON tile in the C5 dispatcher.
fn rotate_buf_u8_via_c5(
  src: &[u8],
  src_w: u32,
  src_h: u32,
  channels: usize,
  rotation: RotateKind,
) -> Result<Vec<u8>> {
  let w_usize = src_w as usize;
  let h_usize = src_h as usize;
  let elements = w_usize
    .checked_mul(h_usize)
    .and_then(|wh| wh.checked_mul(channels))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "rotate_buf_u8_via_c5: elements (w * h * channels)",
        "usize",
        [
          ("w", w_usize as u64),
          ("h", h_usize as u64),
          ("channels", channels as u64),
        ],
      ))
    })?;
  debug_assert_eq!(
    src.len(),
    elements,
    "rotate_buf_u8_via_c5: src.len() must equal w*h*channels by ImageBuffer::as_raw() contract"
  );
  let mut dst: Vec<u8> = Vec::new();
  dst
    .try_reserve_exact(elements)
    .map_err(|_| Error::OutOfMemory)?;
  dst.resize(elements, 0u8);

  let simd_rotation = match rotation {
    RotateKind::Rotate90 => crate::simd::vlm::rotate_buf::RotateKind::Rotate90,
    RotateKind::Rotate270 => crate::simd::vlm::rotate_buf::RotateKind::Rotate270,
    RotateKind::Rotate90FlipH => crate::simd::vlm::rotate_buf::RotateKind::Rotate90FlipH,
    RotateKind::Rotate270FlipH => crate::simd::vlm::rotate_buf::RotateKind::Rotate270FlipH,
  };
  crate::simd::vlm::rotate_buf::rotate_buf_u8(
    &mut dst,
    src,
    w_usize,
    h_usize,
    channels,
    simd_rotation,
  );
  Ok(dst)
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
/// **Return type — fallible target-dimension guard (Codex review).**
/// The signature is `-> Result<DynamicImage>` rather than the
/// infallible `-> DynamicImage` of the swift `resampleBicubic(_,
/// to:) -> CIImage` (line 110-132) / python `Image.resize(new_size)
/// -> Image` references. The earlier faithful-port rationale assumed
/// `target` came from a TRUSTED `ImageProcessorConfig::size` (model
/// metadata baked into the binary or a trusted JSON), so a
/// pathological dimension was a caller bug. That assumption no longer
/// holds: [`ImageProcessorConfig::size`] is now populated from a
/// LOADED `preprocessor_config.json` / `processor_config.json` (see
/// [`crate::vlm::load`]), which is UNTRUSTED on-disk input. A
/// hostile/malformed config with an enormous `size` would otherwise
/// drive the source RGBA materialization and the destination resize
/// alloc to panic-abort the process — taking down image AND video
/// preprocessing on the first request. We therefore validate the
/// target dimensions BEFORE any allocation and surface an over-budget /
/// zero / overflow target as a recoverable `Err` instead of an abort.
///
/// **Allocation contract — fully fallible.** Byte counts are bounded on
/// both ends: the source is bounded by [`MAX_DECODED_IMAGE_BYTES`] via
/// [`load_image`]'s decoder cap, and the destination is bounded by the
/// same [`MAX_DECODED_IMAGE_BYTES`] ceiling enforced on
/// `height * width * 4` by the target guard below (mirroring
/// [`pad_to_square`]'s canvas gate). The cap bounds the SIZE; fallibility
/// is also guaranteed. EVERY allocation in the resize path routes through
/// `try_reserve_exact` and surfaces allocator failure as
/// [`Error::OutOfMemory`] instead of aborting:
/// - the source RGBA buffer (formerly an infallible `img.to_rgba8()`
///   clone) is reserved and filled manually — a borrowed `as_rgba8()`
///   fast path or a per-pixel `dynamic_image_rgba_pixel` projection;
/// - the resize kernel (`vlm::resize::resize_rgba8`) reserves
///   its horizontal + vertical coefficient tables, the inter-pass
///   intermediate, and the destination buffer ALL via `try_reserve_exact`
///   — replacing the dropped `fast_image_resize` dependency, whose
///   internal scratch (coefficient tables, per-row work buffers)
///   allocated infallibly inside the crate and could abort despite our
///   `Result`.
///
/// No infallible `vec!` / external-crate alloc / `to_rgba8` remains in
/// this path, so a just-under-cap hostile target (≈11585×11585 ≈
/// 512 MiB) can no longer force a ~512 MiB infallible alloc → the
/// `Result` signature is honest. The only residual abort is a genuine
/// allocator-internal failure (e.g. a panic inside `Vec::resize`'s growth
/// of an ALREADY-reserved buffer, which cannot fail), not the
/// reservation. The kernel's output is bit-exact with PIL `Image.resize`
/// (the reference mlx-vlm preprocessing expects). The [`preprocess`] and
/// [`crate::vlm::video::process_frames`] composers inherit this
/// transitively (both call `resize` via `?`).
///
/// # Errors
/// - [`Error::ShapeMismatch`] if either target dimension is `0`, if
///   `height * width * 4` overflows `u64`, if it exceeds
///   [`MAX_DECODED_IMAGE_BYTES`], if `source_width * source_height * 4`
///   overflows `usize`, or if that RGBA8-expanded source staging size
///   itself exceeds [`MAX_DECODED_IMAGE_BYTES`] — the message carries the
///   offending dims and the cap. (The over-cap cases use `ShapeMismatch`
///   rather than [`Error::OutOfMemory`] so they can name the dims +
///   ceiling, matching [`pad_to_square`]'s canvas gate.) The source guard
///   matters because `load_image`'s cap is on the *source* pixel format:
///   a `Luma8` (1 B/px) image accepted just under the cap would expand
///   4x as RGBA8 staging.
/// - [`Error::OutOfMemory`] if the allocator cannot satisfy the
///   `try_reserve_exact` for the source RGBA buffer or for any buffer the
///   resize kernel allocates (coefficient tables, inter-pass
///   intermediate, destination — all bounded ≤512 MiB by the caps above)
///   — a recoverable typed error instead of a process abort.
pub fn resize(
  img: &::image::DynamicImage,
  target: (u32, u32),
  filter: ResizeFilter,
) -> Result<::image::DynamicImage> {
  let (height, width) = target;
  // Target-dimension guard (Codex review): `target` now flows from an
  // UNTRUSTED loaded processor config, so validate it BEFORE the
  // source RGBA materialization and the `width * height * 4`
  // destination alloc. Mirrors `pad_to_square`'s canvas gate so the
  // per-step caps stay consistent.
  if width == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "resize: target width",
      "must be non-zero",
      format!("{width}"),
    )));
  }
  if height == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "resize: target height",
      "must be non-zero",
      format!("{height}"),
    )));
  }
  // `height * width * 4` (RGBA8 destination bytes). u64 throughout so the
  // check is host-width-independent; `u32::MAX^2 * 4 ≈ 7.4e19` fits in
  // u64, so the `checked_mul` chain only fires for a truly hostile
  // dimension product.
  let dst_bytes = u64::from(height)
    .checked_mul(u64::from(width))
    .and_then(|hw| hw.checked_mul(4))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "resize: destination bytes (height * width * 4)",
        "u64",
        [("height", u64::from(height)), ("width", u64::from(width))],
      ))
    })?;
  if dst_bytes > MAX_DECODED_IMAGE_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "resize: destination RGBA8",
      "MAX_DECODED_IMAGE_BYTES",
      MAX_DECODED_IMAGE_BYTES,
      dst_bytes,
    )));
  }
  // SIMD-accelerated, fully-fallible resize via mlxrs's OWN kernel
  // (`vlm::resize`) — `fast_image_resize` is dropped. NEON on
  // aarch64 (scalar fallback always compiled, `mlxrs_force_scalar`
  // honored), bit-exact with PIL `Image.resize`. Decode-side stays on
  // `image` (above in `load_image`); only the resize hot path is owned.
  // Public API of this fn is unchanged.
  //
  // Pixel-type: RGBA8 for parity with the prior behavior (image-rs's
  // `imageops::resize` over a `DynamicImage` projects to `Rgba8`
  // unconditionally; downstream `image_to_array` drops alpha as before).
  //
  // FALLIBLE source materialization (Codex review round 2): the prior
  // `img.to_rgba8()` was an OWNED source-sized copy/convert backed by an
  // INFALLIBLE `Vec` (image 0.25 returns a fresh `RgbaImage` and clones
  // even when the source is already `ImageRgba8`; only the consuming
  // `into_rgba8()` path avoids the clone, and we can't use it on a
  // `&DynamicImage`). Even though the source byte count is bounded by
  // [`MAX_DECODED_IMAGE_BYTES`] via `load_image`'s decoder cap, the
  // allocation itself would `abort()` under allocator pressure — making
  // the `Result` signature dishonest. We now reserve the source RGBA
  // buffer via `try_reserve_exact` (`error::try_with_capacity`) and fill
  // it ourselves, mirroring `image_to_array` / `pad_to_square`: a
  // borrowed `as_rgba8()` fast path when the source is already RGBA8,
  // else a per-pixel `dynamic_image_rgba_pixel` projection (the same
  // `dynamic_map!` color-space conversion `to_rgba8()` performed).
  // Allocator failure surfaces as [`Error::OutOfMemory`].
  let src_w = img.width();
  let src_h = img.height();
  // `src_w * src_h * 4` (RGBA8 source bytes) in usize. The source came
  // through `load_image`'s 512 MiB decoder cap so this product is bounded
  // in practice, but a `checked_mul` keeps the reservation honest on a
  // 32-bit `usize` (where the cap's byte count could still wrap) and
  // routes any pathological product to a recoverable `ShapeMismatch`
  // instead of a silent under-allocation.
  let src_bytes = (src_w as usize)
    .checked_mul(src_h as usize)
    .and_then(|wh| wh.checked_mul(4))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "resize: source bytes (src_w * src_h * 4)",
        "usize",
        [("src_w", src_w as u64), ("src_h", src_h as u64)],
      ))
    })?;
  // Source-staging cap (Codex review): `load_image`'s 512 MiB ceiling is
  // enforced on `decoder.total_bytes()` — the SOURCE pixel format. A
  // low-bytes-per-pixel source (e.g. `Luma8`, 1 B/px) accepted just under
  // that cap expands 4x when projected to the RGBA8 staging buffer below,
  // so an under-cap Luma8 image could drive a ~2 GiB RGBA staging
  // allocation and break the documented bounded-memory invariant. The
  // destination already has this guard (`dst_bytes` above); apply the
  // same `MAX_DECODED_IMAGE_BYTES` ceiling to the RGBA-expanded SOURCE
  // staging before its `try_reserve_exact`. `u64` so the comparison is
  // host-width-independent (matches the `dst_bytes` gate).
  if src_bytes as u64 > MAX_DECODED_IMAGE_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "resize: source RGBA8 staging (Luma8/Rgb8/etc. expanded to RGBA8)",
      "MAX_DECODED_IMAGE_BYTES",
      MAX_DECODED_IMAGE_BYTES,
      src_bytes as u64,
    )));
  }
  let mut src_buf = crate::error::try_with_capacity::<u8>(src_bytes)?;
  if let Some(rgba) = img.as_rgba8() {
    // Already RGBA8: borrow the backing `&[u8]` and copy exactly
    // `src_bytes` (slice to guard an over-long backing buffer, matching
    // `image_to_array`'s `.get(..total)` discipline so the fill cannot
    // grow `src_buf` past the `try_reserve_exact` reservation).
    let raw = rgba.as_raw().get(..src_bytes).ok_or_else(|| {
      Error::LengthMismatch(LengthMismatchPayload::new(
        "resize: rgba backing buffer bytes vs W*H*4",
        src_bytes,
        rgba.as_raw().len(),
      ))
    })?;
    src_buf.extend_from_slice(raw);
  } else {
    // Non-RGBA8 source (Luma8 / Rgb8 / Rgb16 / Rgb32F / …): per-pixel
    // `DynamicImage::get_pixel(x, y) -> Rgba<u8>` projection (alpha
    // preserved — this feeds the resize, downstream `image_to_array`
    // drops it). NO intermediate source-sized image allocation.
    for y in 0..src_h {
      for x in 0..src_w {
        src_buf.extend_from_slice(&dynamic_image_rgba_pixel(img, x, y));
      }
    }
  }
  debug_assert_eq!(
    src_buf.len(),
    src_bytes,
    "resize source RGBA fill length must equal pre-computed src_bytes",
  );
  // FALLIBLE resize via mlxrs's own kernel (`vlm::resize`) —
  // the previous `fast_image_resize` dependency is dropped. Every buffer
  // the kernel allocates (the horizontal + vertical coefficient tables,
  // the inter-pass intermediate, and the destination) routes through
  // `try_reserve_exact`, so allocator failure surfaces as
  // [`Error::OutOfMemory`] instead of aborting — closing the last abort
  // path that `fast_image_resize`'s infallible internal scratch left open
  // for an untrusted-config target. The kernel is bit-exact with PIL
  // `Image.resize` (the reference mlx-vlm preprocessing expects); the
  // public API of this fn is unchanged.
  //
  // `resize_rgba8` takes `(w, h)` in pixels (NOT bytes); `src_w`/`src_h`
  // are the source dims and `(width, height)` the validated target. The
  // returned `Vec<u8>` is exactly `width * height * 4` bytes (validated
  // ≤ `MAX_DECODED_IMAGE_BYTES` by the target guard above).
  let dst_buf = crate::vlm::resize::resize_rgba8(
    &src_buf,
    src_w as usize,
    src_h as usize,
    width as usize,
    height as usize,
    filter.to_internal(),
  )?;
  debug_assert_eq!(
    dst_buf.len(),
    dst_bytes as usize,
    "resize destination RGBA length must equal pre-computed dst_bytes",
  );
  Ok(::image::DynamicImage::ImageRgba8(
    ::image::ImageBuffer::from_raw(width, height, dst_buf)
      // `resize_rgba8` returns exactly `width * height * 4` bytes — the
      // precise length `ImageBuffer::from_raw` requires for a
      // `width x height` RGBA buffer.
      .expect("ImageBuffer::from_raw: dst buffer length matches width * height * 4 by construction"),
  ))
}

/// Resize `img` to `(target_h, target_w)` using Lanczos3 interpolation.
///
/// Convenience wrapper around [`resize`] that fixes the filter to
/// [`ResizeFilter::Lanczos3`]. Mirrors swift
/// [`MediaProcessing.resampleLanczos`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L81-L103)
/// and PIL `Image.resize(LANCZOS)`. Lanczos3 (`a = 3`, sinc-windowed sinc)
/// matches PIL's `Image.LANCZOS` kernel exactly (Pillow renamed the old
/// `ANTIALIAS` to `LANCZOS` in 9.1.0; both are the `a=3` Lanczos
/// convolution).
///
/// The argument order is `(target_h, target_w)` — the swift API takes a
/// `CGSize(width:, height:)` but we mirror the python image-processor
/// convention (`(height, width)`) that the rest of [`resize`] /
/// [`ImageProcessorConfig::size`] uses. Output dimensions are exact —
/// the own resize kernel (`vlm::resize`) produces exactly the
/// requested `(w, h)` (no trailing crop step needed, unlike the swift
/// implementation which has to `cropped(to: exactRect)` after
/// `lanczosScaleTransform` produces a near-target output).
///
/// # Errors
/// Propagates [`resize`]'s target-dimension guard verbatim
/// ([`Error::ShapeMismatch`] on zero/overflow/over-cap target dims).
pub fn resize_lanczos(
  img: &::image::DynamicImage,
  target_h: u32,
  target_w: u32,
) -> Result<::image::DynamicImage> {
  resize(img, (target_h, target_w), ResizeFilter::Lanczos3)
}

/// Center crop `img` to `(target_h, target_w)`.
///
/// Mirrors swift
/// [`MediaProcessing.centerCrop(_:size:)`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L213-L224)
/// and the python HF `BaseImageProcessor.center_crop` (`crop_size` field
/// at `mlx_vlm/models/base.py:140-153`).
///
/// **Early-return parity:** swift `rectSmallerOrEqual`
/// (`MediaProcessing.swift:196-198`) returns true only when *both* axes
/// already fit within the target (`source.width <= target.width &&
/// source.height <= target.height`); only then is the source returned
/// unchanged. When just one axis exceeds the target, swift's
/// `centerCrop` helper at lines 201-210 clamps each crop dim with
/// `min(source, target)` and computes centered offsets — the bigger
/// axis is cropped, the smaller axis is kept at the source extent
/// (centered offset of 0).
///
/// The geometric center is `(W - crop_w) / 2`, `(H - crop_h) / 2`
/// (integer division — for an even-sized source with odd-sized target
/// the crop is biased toward the top-left pixel by 0.5, matching
/// `crop_imm`'s unsigned-floor semantics and PIL `Image.crop` behavior).
///
/// Infallible by reference parity (swift signature returns `CIImage`
/// non-throwing; python `center_crop` returns `np.ndarray`).
pub fn center_crop(
  img: &::image::DynamicImage,
  target_h: u32,
  target_w: u32,
) -> ::image::DynamicImage {
  let w = img.width();
  let h = img.height();
  // Swift `rectSmallerOrEqual` early-return: BOTH axes must already
  // fit. If only one axis is larger than the target we still need to
  // crop that bigger axis (using `min(source, target)` for the smaller
  // axis), matching swift's `min(extent, target)` clamp at
  // `MediaProcessing.swift:201-210`.
  if w <= target_w && h <= target_h {
    return img.clone();
  }
  // Clamp each crop dimension to `min(source, target)` so a partial-fit
  // case crops only the bigger axis (the smaller axis is kept at the
  // source extent with a centered offset of 0). For the fully-bigger
  // case (both axes > target) this collapses to `(target_w, target_h)`.
  let crop_w = w.min(target_w);
  let crop_h = h.min(target_h);
  // Integer-floor center offsets; PIL `Image.crop` and the swift
  // `centerCrop` rect helper compute `(extent - crop) / 2` likewise.
  // When `crop_x == w` (smaller axis kept whole) this is `0`.
  let x = (w - crop_w) / 2;
  let y = (h - crop_h) / 2;
  img.crop_imm(x, y, crop_w, crop_h)
}

/// Pad `img` to a square by filling the shorter side with `fill`.
///
/// Mirrors python `expand2square` (`mlx_vlm/models/base.py:251-262`)
/// and `expand_to_square` (`mlx_vlm/models/fastvlm/processing.py:29-61`)
/// — the canonical pre-resize step that LLaVA-family processors apply
/// to preserve aspect ratio before a square encoder resize. The swift
/// `MediaProcessing` module does not expose a `padSquare` helper
/// directly; the per-model swift processors handle aspect-ratio
/// preservation in their own `preprocess` step.
///
/// **Padding policy:** the shorter side is symmetrically padded —
/// `(long - short) / 2` pixels on each edge (integer floor; for odd
/// differences the bottom / right edge gets the extra row / column,
/// matching python `Image.new(...).paste(img, ((width - height) // 2,
/// 0))` which centers the input). If `width == height`, the source is
/// returned **unchanged** (the input `DynamicImage` is moved out — no
/// allocation, no variant conversion).
///
/// **Ownership / signature:** `img` is taken **by value** so the
/// already-square fast path can hand the input back without a clone.
/// `DynamicImage::clone()` deep-copies the entire decoded pixel
/// buffer; for a near-budget input that buffer can itself be hundreds
/// of MiB, and Rust's infallible `Clone` would abort the process on
/// allocator failure — defeating the recoverable-OOM contract the
/// padded path enforces below. Callers that need to keep the source
/// alongside the output should clone upstream, where the failure mode
/// is the caller's to choose.
///
/// **Color order (padded path only):** `fill` is an `[R, G, B]` u8
/// triple regardless of the source dtype — the *padded* output is
/// always `Rgb8`. The already-square fast path returns the input
/// unchanged (any `DynamicImage` variant); callers that need a
/// uniform output dtype must convert post-hoc (e.g. `out.to_rgb8()`).
/// Callers needing `[0.0, 1.0]` float-space padding should pad after
/// the [`image_to_array`] + [`rescale`] steps in array space (one
/// `pad` op, not yet exposed here; per-model concern when needed).
///
/// **End-to-end fallible canvas (Rust safety, not parity):** the python
/// reference is infallible because `PIL.Image.new(size, size)` raises
/// `MemoryError` on OOM — an exception that propagates cleanly up the
/// processor stack. Rust's `RgbImage::from_pixel(size, size, ...)`,
/// `Vec::with_capacity`, and `DynamicImage::to_rgb8()` *abort* the
/// process on allocator failure (the standard `Vec` reallocation and
/// `image`'s infallible buffer constructors all panic). A
/// `100_000 x 1` source would otherwise drive a `100_000² × 3` = 30 GiB
/// canvas allocation. To preserve the exception-like recoverability
/// the python contract assumes, this function:
/// 1. Checks `size × size × 3` for `u64` overflow *and* against
///    [`MAX_DECODED_IMAGE_BYTES`] (the same 512 MiB ceiling
///    [`load_image`] enforces);
/// 2. Allocates the pixel buffer via `Vec::try_reserve_exact` so an
///    allocator failure surfaces as [`Error::OutOfMemory`] rather than
///    a panic-abort;
/// 3. Uses `image::ImageBuffer::from_raw` on a uniform-fill buffer to
///    keep the `RgbImage::from_pixel` semantics without its panicking
///    backing alloc;
/// 4. Writes source pixels into the *already-reserved* canvas slice
///    in-place — either row-wise `copy_from_slice` when the source is
///    already `ImageRgb8` (zero intermediate alloc), or via the
///    `DynamicImage::get_pixel` color-space-converting accessor for
///    non-`Rgb8` variants (one `Rgba<u8>` per pixel on the stack;
///    again no intermediate full-image alloc). The prior
///    `img.to_rgb8()` call materialized a fresh decoded-byte-sized
///    copy infallibly — a near-budget nonsquare input (e.g.
///    `13377×13376` RGB ≈ 511 MiB) would pass the canvas gate, then
///    panic-abort on the ~511 MiB `to_rgb8` clone. The
///    per-pixel-write path eliminates that second source-sized
///    allocation entirely.
///
/// The [`MAX_DECODED_IMAGE_BYTES`] budget bounds the canvas alone; the
/// source itself is already-decoded (its own allocation was bounded at
/// [`load_image`] time, or by the caller if constructed via a custom
/// path). The per-pixel iteration touches that already-resident memory
/// without spawning a second copy.
///
/// Oversized inputs return [`Error::ShapeMismatch`] with the requested
/// vs allowed byte count; allocator failures return
/// [`Error::OutOfMemory`].
pub fn pad_to_square(img: ::image::DynamicImage, fill: [u8; 3]) -> Result<::image::DynamicImage> {
  let w = img.width();
  let h = img.height();
  if w == h {
    // Square fast path: return the input unchanged. NOT `img.clone()`
    // — `DynamicImage::clone()` deep-copies the entire decoded buffer
    // via the infallible `Vec` clone, which `abort()`s on allocator
    // failure for near-budget inputs. By taking `img` by value we hand
    // the same allocation back to the caller; no second source-sized
    // copy ever happens here.
    return Ok(img);
  }
  let size = w.max(h);
  // `size * size * 3` byte budget. Use u64 throughout so the check is
  // identical on 32-bit and 64-bit hosts (and so `MAX_DECODED_IMAGE_BYTES`
  // can be compared without lossy casts). `u32::MAX^2 * 3 ≈ 5.5e19` fits
  // in u64, so the `checked_mul` chain only fires for a truly hostile
  // dimension product.
  let size_u64 = u64::from(size);
  let bytes = size_u64
    .checked_mul(size_u64)
    .and_then(|sq| sq.checked_mul(3))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "pad_to_square: canvas bytes (size * size * 3)",
        "u64",
        [("size", size_u64)],
      ))
    })?;
  if bytes > MAX_DECODED_IMAGE_BYTES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "pad_to_square: canvas (size x size x 3)",
      "MAX_DECODED_IMAGE_BYTES",
      MAX_DECODED_IMAGE_BYTES,
      bytes,
    )));
  }
  // `bytes <= MAX_DECODED_IMAGE_BYTES = 512 MiB`, well under `usize::MAX`
  // on every supported host (we ship aarch64-darwin and x86_64-linux —
  // both 64-bit; the 32-bit edge case is bounded by the u64 check
  // above). Cast is lossless given the prior gate.
  let bytes_usize = bytes as usize;
  // Recoverable OOM at the canvas allocation. `vec![value; n]` and
  // `Vec::with_capacity` would `abort()` on allocator failure;
  // `try_reserve_exact` surfaces it as `Error::OutOfMemory`.
  let mut canvas_buf: Vec<u8> = Vec::new();
  canvas_buf
    .try_reserve_exact(bytes_usize)
    .map_err(|_| Error::OutOfMemory)?;
  // Uniform RGB fill via the C6 `pad_canvas_fill` dispatcher
  // (`crate::simd::vlm::pad_canvas_fill`). On `aarch64` this routes to
  // a 48-byte LCM(3, 16) NEON pre-broadcast tile (~2-4× faster than
  // the new scalar at 4096²; ~10-50× faster than the pre-C6 per-3-byte
  // `extend_from_slice` idiom this replaces — verify-before-claim
  // bench in `mlxrs/benches/simd_pad_canvas_fill.rs`); elsewhere it
  // falls back to the scalar `chunks_exact_mut(3) + copy_from_slice`
  // path. See `docs/core-arch-simd-candidates.md` §2 row C6 + §5.5
  // for the rationale + tracking ([#151]).
  //
  // The dispatcher takes `&mut [MaybeUninit<u8>]` (type-encoded uninit
  // safety — see the kernel-module doc), so we pass the pre-reserved
  // spare capacity **directly** and `set_len` after every byte has
  // been written. No `from_raw_parts_mut` cast over uninit backing
  // memory (which would be UB regardless of subsequent writes, per
  // `from_raw_parts_mut`'s "properly initialized" precondition).
  {
    let spare = canvas_buf.spare_capacity_mut();
    // `bytes_usize <= spare.len()` because `try_reserve_exact` reserved
    // exactly `bytes_usize` extra capacity. Take the first `bytes_usize`
    // slots of spare (the canvas region).
    debug_assert!(
      spare.len() >= bytes_usize,
      "try_reserve_exact must have reserved at least bytes_usize"
    );
    crate::simd::vlm::pad_canvas_fill(&mut spare[..bytes_usize], fill);
  }
  // SAFETY: `pad_canvas_fill` wrote every byte in `0..bytes_usize` of
  // the spare capacity (its function-level contract: "every byte of
  // `out` is written before this returns" — both the scalar
  // `chunks_exact_mut(3)` path and the NEON 48-byte tile path cover
  // the full slice). `Vec::set_len`'s preconditions:
  //   (1) `bytes_usize <= canvas_buf.capacity()` — `try_reserve_exact`
  //       above reserved exactly that much capacity (a
  //       `try_reserve_exact` failure would have early-returned
  //       `Error::OutOfMemory`);
  //   (2) elements at `[old_len..new_len]` are initialized — the
  //       kernel-contract above guarantees this.
  // Both hold.
  unsafe { canvas_buf.set_len(bytes_usize) };
  debug_assert_eq!(
    canvas_buf.len(),
    bytes_usize,
    "canvas fill length must equal pre-computed bytes",
  );
  // Symmetric center offset on the shorter axis; longer axis stays at 0.
  let (x_off, y_off) = if w > h {
    (0u32, (w - h) / 2)
  } else {
    ((h - w) / 2, 0u32)
  };
  // `size` was bounded above (`bytes = size * size * 3 <= 512 MiB`),
  // so `size <= 13_377` — well within `u32`/`usize` on any 64-bit host.
  // Cast to `usize` for slice indexing.
  let size_usize = size as usize;
  let w_usize = w as usize;
  let h_usize = h as usize;
  let x_off_usize = x_off as usize;
  let y_off_usize = y_off as usize;
  // Write source pixels directly into the already-reserved canvas
  // buffer — NO intermediate `to_rgb8` allocation. Two paths:
  //   - source is `ImageRgb8`: row-wise `copy_from_slice` (one memcpy
  //     of `w*3` bytes per source row). Zero per-pixel overhead, zero
  //     intermediate alloc.
  //   - other variant (Luma8 / Rgba8 / Rgb16 / …): per-pixel
  //     `DynamicImage::get_pixel(x, y) -> Rgba<u8>` (color-space
  //     conversion handled by `image`'s `dynamic_map!` dispatch). We
  //     drop alpha and keep the RGB channels — identical projection
  //     to `image_to_array` and the prior `to_rgb8()` did.
  if let Some(src_rgb) = img.as_rgb8() {
    let src_raw = src_rgb.as_raw();
    // `src_raw.len() == w * h * 3` by `ImageBuffer<Rgb<u8>>::as_raw`
    // contract (image 0.25 `ImageBuffer::as_raw` for `P = Rgb<u8>`).
    // The `dst_stride * src_h` and `src_stride * src_h` bounds below
    // are therefore both within their respective buffers.
    let src_stride = w_usize * 3;
    let dst_stride = size_usize * 3;
    let dst_x_byte = x_off_usize * 3;
    for y_src in 0..h_usize {
      let dst_row_off = (y_off_usize + y_src) * dst_stride + dst_x_byte;
      let src_row_off = y_src * src_stride;
      canvas_buf[dst_row_off..dst_row_off + src_stride]
        .copy_from_slice(&src_raw[src_row_off..src_row_off + src_stride]);
    }
  } else {
    // Per-pixel path for non-`Rgb8` sources. Reuses
    // [`dynamic_image_rgb_pixel`] (the shared `to_rgba() -> drop alpha`
    // projection) so the non-Rgb8 branch and `image_to_array`'s
    // non-Rgb8 branch produce byte-identical RGB triples — the
    // structural unification kills the defect class (any future
    // tweak to the projection lives in one place).
    let dst_stride = size_usize * 3;
    for y_src in 0..h_usize {
      let dst_row_off = (y_off_usize + y_src) * dst_stride + x_off_usize * 3;
      for x_src in 0..w_usize {
        let rgb = dynamic_image_rgb_pixel(&img, x_src as u32, y_src as u32);
        let off = dst_row_off + x_src * 3;
        canvas_buf[off] = rgb[0];
        canvas_buf[off + 1] = rgb[1];
        canvas_buf[off + 2] = rgb[2];
      }
    }
  }
  // `from_raw` only returns `None` when `buf.len() < width * height *
  // channels`. By construction `canvas_buf.len() == size * size * 3`
  // (the uniform-fill loop pushed exactly `bytes_usize / 3` × 3 bytes;
  // the source overlay above writes in place via index assignment and
  // does not change the buffer length).
  let canvas: ::image::RgbImage = ::image::ImageBuffer::from_raw(size, size, canvas_buf)
    .expect("ImageBuffer::from_raw: canvas buffer length matches size * size * 3 by construction");
  Ok(::image::DynamicImage::ImageRgb8(canvas))
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
/// here is the unavoidable `decoded -> f32` widening — the swift
/// reference allocates the same buffer at `Data(count: w * h *
/// bytesPerPixel)` in `MediaProcessing.swift:176`, and python
/// `np.asarray(image)` does too. The upper bound is roughly
/// `4 * 512 MiB = 2 GiB` of f32s for a decoder-default `load_image`
/// source. Callers that hand a `DynamicImage` from a different source
/// (raw `image::open`, network decoders, etc.) inherit whatever limit
/// that source imposed; pre-validating `img.dimensions()` (a cheap
/// O(1) field read) is the standard escape hatch.
///
/// **No infallible source clone (Codex review):** the prior
/// implementation called `img.to_rgb8()` unconditionally as its first
/// step. `DynamicImage::to_rgb8()` is documented as "Returns a copy
/// of this image as an RGB image" (image 0.25
/// `DynamicImage::to_rgb8`) — it clones the backing buffer for *every*
/// variant including the already-`Rgb8` case (the buffer is cloned
/// via the infallible `Vec::clone` because the underlying `RgbImage`
/// is `Clone`). For a near-budget input (e.g. an `ImageRgb8` whose
/// decoded buffer is ~512 MiB) this materialized a second source-sized
/// allocation infallibly before the recoverable `try_reserve_exact`
/// gate ever ran — `Vec::clone` aborts on allocator failure.
/// The current implementation eliminates that second source-sized
/// clone:
/// 1. Reserve the output f32 buffer first via `try_reserve_exact`.
/// 2. `as_rgb8()` fast path: read directly from the source's backing
///    `&[u8]` (no clone) and widen to f32.
/// 3. Non-`Rgb8` (`Luma8`/`Rgba8`/`Rgb16`/`Rgb32F`/…): per-pixel
///    `dynamic_image_rgb_pixel` projection (shared private helper) —
///    one `Rgba<u8>` on the stack per pixel, no intermediate
///    full-image alloc. The same projection [`pad_to_square`]'s
///    non-`Rgb8` branch uses, so any future tweak to the per-pixel
///    RGB extraction lives in one place.
pub fn image_to_array(img: &::image::DynamicImage, color_order: ColorOrder) -> Result<Array> {
  let w = img.width();
  let h = img.height();
  let w_usize = w as usize;
  let h_usize = h as usize;
  // FFI-bound shape product overflow guard. `Array::from_slice` validates
  // shape-product vs buffer length but does so in `usize` arithmetic
  // *after* our cast; on a 32-bit usize the multiplication
  // `h_usize * w_usize * 3` can wrap silently. Catch it here with a
  // recoverable `Error::ShapeMismatch` so callers see a clean error rather
  // than a panic downstream. This MUST run before any allocation so a
  // hostile dimension product cannot abort in the allocator.
  let total = h_usize
    .checked_mul(w_usize)
    .and_then(|hw| hw.checked_mul(3))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "image_to_array: total elements (h * w * 3)",
        "usize",
        [("h", h_usize as u64), ("w", w_usize as u64)],
      ))
    })?;
  // Recoverable OOM at the f32 widening boundary. `Vec::with_capacity`
  // would `abort()` on a hostile-but-non-overflowing image (the
  // `checked_mul` only proves `total` fits `usize`, not that the
  // `total * 4` byte alloc succeeds). `try_reserve_exact` surfaces an
  // allocator failure as a recoverable `Error::OutOfMemory` so callers
  // get a typed Err instead of process termination — matches the
  // allocation-discipline pattern `mlxrs::error::Error::OutOfMemory`
  // exists for. NO `to_rgb8()` clone runs before this gate.
  let mut buf: Vec<f32> = Vec::new();
  buf
    .try_reserve_exact(total)
    .map_err(|_| Error::OutOfMemory)?;
  // Fast path: source is already `ImageRgb8`. Read its backing `&[u8]`
  // directly (no clone, no per-pixel dispatch) and widen to f32.
  //
  // `as_rgb8()` returns `Option<&RgbImage>` (borrow, not clone). When
  // `Some`, `rgb.as_raw()` is `&[u8]` with length AT LEAST
  // `width * height * 3` — the `ImageBuffer::as_raw()` contract allows
  // a backing buffer longer than the logical extent (callers can
  // construct via `from_raw` with an oversized Vec). Slice to exactly
  // `total = H*W*3` bytes via `.get(..total)` so the fill loop iterates
  // the correct extent — without this slice an overlong-backing-buffer
  // source would grow `buf` past the `try_reserve_exact(total)`
  // reservation via infallible allocation, reintroducing the
  // abort-on-OOM hazard.
  if let Some(rgb) = img.as_rgb8() {
    let raw = rgb.as_raw().get(..total).ok_or_else(|| {
      Error::LengthMismatch(LengthMismatchPayload::new(
        "image_to_array: rgb backing buffer bytes vs H*W*3",
        total,
        rgb.as_raw().len(),
      ))
    })?;
    match color_order {
      ColorOrder::Rgb => {
        // C3 SIMD dispatcher (`crate::simd::vlm::rgb_widen`). On
        // `aarch64` this routes to a `vld1q_u8` + four `vst1q_f32`
        // 16-byte-per-tile NEON kernel (pure byte-for-byte widen, no
        // de-interleave needed for RGB-in-RGB-out); elsewhere it
        // falls back to the scalar `MaybeUninit::write` path. The
        // pre-C3 `buf.extend(raw.iter().map(|&b| f32::from(b)))`
        // shape LLVM auto-vectorized cleanly, but the hand-rolled
        // NEON arm pins the contract against compiler-version drift
        // (per the user directive 2026-05-23 / project memory rule
        // **"SIMD ship NEON regardless"**). See
        // `docs/core-arch-simd-candidates.md` §2 row C3 + §3.3 +
        // tracking [#148].
        //
        // The dispatcher takes `&mut [MaybeUninit<f32>]` (type-
        // encoded uninit safety, matching C4/C6); we pass the
        // pre-reserved spare capacity directly and `set_len(total)`
        // after every f32 has been written.
        {
          let spare = buf.spare_capacity_mut();
          debug_assert!(
            spare.len() >= total,
            "try_reserve_exact must have reserved at least total f32s"
          );
          crate::simd::vlm::rgb_widen(&mut spare[..total], raw);
        }
        // SAFETY: `rgb_widen` wrote every f32 in `0..total` of the
        // spare capacity (function-level contract: "every f32 of
        // `out` is written before this returns" — both scalar and
        // NEON arms cover the full slice). `Vec::set_len`'s
        // preconditions:
        //   (1) `total <= buf.capacity()` — `try_reserve_exact`
        //       above reserved exactly that much;
        //   (2) elements at `[0..total]` are initialized — by
        //       kernel contract.
        unsafe { buf.set_len(total) };
      }
      ColorOrder::Bgr => {
        // Per-pixel R↔B swap via the C4 `bgr_widen` dispatcher
        // (`crate::simd::vlm::bgr_widen`). On `aarch64` this routes
        // to a `vld3q_u8` + permuted `vst3q_f32` 16-pixel-per-tile
        // NEON kernel (the R↔B swap is encoded by feeding the
        // de-interleaved planes to the interleave-store in reversed
        // R/B order); elsewhere it falls back to the scalar
        // `chunks_exact_mut(3) + MaybeUninit::write` path. See
        // `docs/core-arch-simd-candidates.md` §2 row C4 + §3.3 + §5.5
        // for the rationale + tracking ([#149]).
        //
        // The dispatcher takes `&mut [MaybeUninit<f32>]` (type-encoded
        // uninit safety — see the kernel-module doc), so we pass the
        // pre-reserved spare capacity **directly** and `set_len` after
        // every f32 has been written. No `from_raw_parts_mut` cast
        // over uninit backing memory (which would be UB regardless of
        // subsequent writes, per `from_raw_parts_mut`'s "properly
        // initialized" precondition).
        {
          let spare = buf.spare_capacity_mut();
          // `total <= spare.len()` because `try_reserve_exact(total)`
          // above reserved exactly `total` extra capacity. Take the
          // first `total` slots of spare (one f32 per input byte).
          debug_assert!(
            spare.len() >= total,
            "try_reserve_exact must have reserved at least total f32s"
          );
          crate::simd::vlm::bgr_widen(&mut spare[..total], raw);
        }
        // SAFETY: `bgr_widen` wrote every f32 in `0..total` of the
        // spare capacity (its function-level contract: "every f32 of
        // `out` is written before this returns" — both the scalar
        // `chunks_exact_mut(3)` path and the NEON 16-pixel tile path
        // cover the full slice). `Vec::set_len`'s preconditions:
        //   (1) `total <= buf.capacity()` — `try_reserve_exact` above
        //       reserved exactly that much capacity (a
        //       `try_reserve_exact` failure would have early-returned
        //       `Error::OutOfMemory`);
        //   (2) elements at `[0..total]` are initialized — the
        //       kernel-contract above guarantees this.
        // Both hold.
        unsafe { buf.set_len(total) };
      }
    }
  } else {
    // Non-`Rgb8` source (Luma8 / Rgba8 / Rgb16 / Rgb32F / …):
    // build a contiguous `Vec<u8>` of length `H*W*3` via the shared
    // [`dynamic_image_rgb_pixel`] helper (one `Rgba<u8>` on the stack
    // per pixel — NO intermediate source-sized RGB image allocation),
    // THEN hand the bulk buffer to the same C3/C4 SIMD dispatcher
    // the `Rgb8` fast path uses above. The earlier per-pixel
    // `buf.push(f32::from(...))` shape paid an iterator + per-element
    // capacity-check overhead even though `try_reserve_exact` had
    // already reserved exact capacity — Copilot review (VLM-3, #121).
    //
    // The intermediate `Vec<u8>` is itself fallibly reserved through
    // the crate-private `error::try_with_capacity` helper so allocator
    // failure surfaces as [`Error::OutOfMemory`] identically to the f32 buf
    // (no second abort-on-OOM seam). The intermediate is bounded by
    // the same `total` overflow-checked shape product above (its
    // byte count == 1×total, vs the f32 buf's 4×total), so the
    // byte-budget audit table's "Bounded-memory" Y still holds
    // end-to-end — `MAX_DECODED_IMAGE_BYTES` (512 MiB) at
    // [`load_image`] caps the SOURCE byte count, the f32 widening
    // here is at most 4× of that, and this intermediate adds at most
    // 1× more (5× total = ~2.5 GiB peak for a near-cap source —
    // unchanged from the prior `push` shape which materialized the
    // same f32 buf and merely paid per-element overhead).
    let mut rgb_buf = crate::error::try_with_capacity::<u8>(total)?;
    for y in 0..h {
      for x in 0..w {
        let rgb = dynamic_image_rgb_pixel(img, x, y);
        rgb_buf.push(rgb[0]);
        rgb_buf.push(rgb[1]);
        rgb_buf.push(rgb[2]);
      }
    }
    debug_assert_eq!(
      rgb_buf.len(),
      total,
      "non-Rgb8 RGB intermediate fill length must equal pre-computed total"
    );
    // Dispatch through the same C3/C4 SIMD widener the Rgb8 fast path
    // uses. NEON kernel on aarch64; auto-vectorized scalar elsewhere.
    // Identical kernel-contract (every f32 in `0..total` of spare is
    // written before return), identical post-call `set_len` safety
    // chain — fold this branch into the same code shape as the fast
    // path so the SIMD ship-decision applies uniformly to every input
    // variant, not just `ImageRgb8`.
    match color_order {
      ColorOrder::Rgb => {
        let spare = buf.spare_capacity_mut();
        debug_assert!(
          spare.len() >= total,
          "try_reserve_exact must have reserved at least total f32s"
        );
        crate::simd::vlm::rgb_widen(&mut spare[..total], &rgb_buf);
      }
      ColorOrder::Bgr => {
        let spare = buf.spare_capacity_mut();
        debug_assert!(
          spare.len() >= total,
          "try_reserve_exact must have reserved at least total f32s"
        );
        crate::simd::vlm::bgr_widen(&mut spare[..total], &rgb_buf);
      }
    }
    // SAFETY: identical to the `Rgb8` fast path above — `rgb_widen` /
    // `bgr_widen` write every f32 in `0..total` per their function-level
    // contract; `try_reserve_exact(total)` reserved exactly that much
    // capacity; both `Vec::set_len` preconditions hold.
    unsafe { buf.set_len(total) };
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

/// Per-channel normalization: `(x - mean[c]) / std[c]`.
///
/// `arr` shape: `[..., 3]` (channel-last). The mean/std tuples are
/// broadcast across all leading dims by reshaping them to `[1, 1, 3]`
/// (when `arr` is `[H, W, 3]`) — generally `arr.ndim() - 1` leading
/// 1-dims so the broadcast applies cleanly regardless of batch axis.
///
/// Mirrors swift
/// [`MediaProcessing.normalize`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/MediaProcessing.swift#L135-L157)
/// and torchvision
/// [`Normalize`](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)
/// (`output[c] = (input[c] - mean[c]) / std[c]`). The swift
/// implementation expresses `(x - mean) / std` via the CIFilter
/// colorMatrix "input * (1/std) + (-mean/std)" trick (algebraically
/// equivalent — see the swift comment block lines 142-148). We use the
/// direct subtract + divide for readability; the math is the same.
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
///
/// **Layout note:** the swift / torchvision references both operate on
/// the layout natural to their stack (CIFilter on `[H, W, C]`-rendered
/// CIFormat.RGBAf; torchvision on planar `[C, H, W]` with a `[C, 1, 1]`
/// broadcast). We chose channel-last `[..., 3]` because [`image_to_array`]
/// emits that layout and the `(3,)` mean/std broadcasts over the trailing
/// axis without an extra transpose. Per-model processors that operate
/// post planar-conversion can adapt by adding leading singleton axes to
/// the mean/std tensors themselves before calling [`subtract`] /
/// [`divide`] directly.
pub fn normalize(arr: &Array, mean: &[f32; 3], std: &[f32; 3]) -> Result<Array> {
  let ndim = arr.ndim();
  if ndim == 0 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "normalize: input must be rank >= 1 (at least 1 dimension)",
      ndim as u32,
      arr.shape(),
    )));
  }
  // Validate trailing channel dim == 3 with a clear error before falling
  // through to mlx's less-friendly broadcast failure.
  let shape = arr.shape();
  let trailing = shape[ndim - 1];
  if trailing != 3 {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "normalize: trailing channel dim (must be 3 for RGB)",
      3,
      trailing,
    )));
  }
  let dtype = arr.dtype()?;
  require_float_dtype("normalize", dtype)?;
  // Build (3,) mean and std arrays in the input dtype, then reshape to
  // [1, ..., 1, 3] so they broadcast over every leading axis of `arr`.
  let mean_arr = make_channel_broadcast(mean, ndim, dtype)?;
  let std_arr = make_channel_broadcast(std, ndim, dtype)?;
  let centered = subtract(arr, &mean_arr)?;
  divide(&centered, &std_arr)
}

/// ImageNet-named alias for [`normalize`] — same `(x - mean) / std`
/// per-channel semantics. Retained for source-compatibility; new code
/// should prefer [`normalize`] (matches swift `MediaProcessing.normalize`
/// and torchvision `Normalize` naming).
pub fn normalize_imagenet(arr: &Array, mean: &[f32; 3], std: &[f32; 3]) -> Result<Array> {
  normalize(arr, mean, std)
}

/// Reject non-float dtypes for primitives that need fractional arithmetic.
///
/// The swift reference's CIFilter pipeline runs exclusively in float
/// space (CIFormat.RGBAf @ `MediaProcessing.swift:171`); the python
/// HF processors call `array.astype(np.float32)` before rescale /
/// normalize. We surface the dtype mismatch as a clean
/// `Error::ShapeMismatch` rather than letting the caller discover it
/// as silent zeros downstream.
fn require_float_dtype(op: &'static str, dtype: Dtype) -> Result<()> {
  match dtype {
    Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => Ok(()),
    _ => Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
      op,
      dtype,
      &[Dtype::F16, Dtype::BF16, Dtype::F32, Dtype::F64],
    ))),
  }
}

/// Build a `[1, ..., 1, 3]`-shaped broadcast tensor from a length-3 f32
/// slice, cast to `dtype`. Helper for [`normalize`].
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
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "normalize: input ndim",
      "MAX_NDIM",
      MAX_NDIM as u64,
      ndim as u64,
    )));
  }
  let mut buf = [1usize; MAX_NDIM];
  buf[ndim - 1] = 3;
  reshape(&a, &&buf[..ndim])
}

/// Read a single `[R, G, B]` u8 triple at `(x, y)` from a [`DynamicImage`]
/// without materializing an intermediate full-image `RgbImage`.
///
/// Shared per-pixel projection for the non-`Rgb8` branches of
/// [`pad_to_square`] and [`image_to_array`]. Both callers used to embed
/// the same `get_pixel().0[..3]` projection inline; lifting it into one
/// helper structurally unifies them so any future tweak (alpha
/// premultiplication, gamma handling, etc.) lives in one place.
///
/// **Why not `img.to_rgb8()` once?** `DynamicImage::to_rgb8()` is
/// documented as "Returns a copy of this image as an RGB image"
/// (image 0.25 `DynamicImage::to_rgb8`) — it clones the backing buffer
/// for *every* variant including the already-`Rgb8` case, via the
/// infallible `Vec::clone`. For a near-budget input that buffer is
/// itself hundreds of MiB, so the clone aborts the process on allocator
/// failure — defeating the recoverable-OOM contract the two callers
/// enforce on their *output* allocations. The per-pixel path here
/// touches the source's already-resident memory in place (one
/// `Rgba<u8>` on the stack per pixel) and never spawns a second
/// source-sized copy.
///
/// **Color projection:** `DynamicImage::get_pixel(x, y)` returns
/// `Rgba<u8>` regardless of the underlying variant — see image 0.25
/// `dynimage.rs:1499-1501` for the `dynamic_map!(*self, ref p,
/// p.get_pixel(x, y).to_rgba().into_color())` dispatch. The
/// per-variant projections this composes through:
///   - `ImageLuma8`: grey → broadcast to `(L, L, L, 255)`.
///   - `ImageLumaA8`: grey + alpha → `(L, L, L, A)`.
///   - `ImageRgb8`: `(R, G, B)` → `(R, G, B, 255)` (we don't take this
///     path here — see `as_rgb8()` fast path in the callers).
///   - `ImageRgba8`: identity.
///   - `ImageRgb16` / `ImageRgba16`: 16-bit → 8-bit via the standard
///     `Subpixel: ColorConvert` shift-down.
///   - `ImageRgb32F` / `ImageRgba32F`: float → 8-bit via the standard
///     clamp + scale.
///
/// We drop the alpha channel and return `[R, G, B]` — identical
/// projection to the prior `to_rgb8()` call, just without the
/// intermediate full-image allocation.
///
/// **Bounds:** caller must guarantee `x < img.width()` and
/// `y < img.height()` — `DynamicImage::get_pixel` panics on
/// out-of-bounds indices (the `image` crate documents this; the
/// `dynamic_map!` dispatch goes through the per-variant
/// `ImageBuffer::get_pixel` which `panics` rather than returns
/// `Option`). Both callers iterate `0..h` × `0..w` so this is
/// trivially satisfied.
fn dynamic_image_rgb_pixel(img: &::image::DynamicImage, x: u32, y: u32) -> [u8; 3] {
  // `GenericImageView` is brought into scope locally so `get_pixel`
  // resolves on the opaque `DynamicImage` type without polluting the
  // module-level imports.
  use ::image::GenericImageView as _;
  let p = img.get_pixel(x, y);
  [p.0[0], p.0[1], p.0[2]]
}

/// Project a single pixel of any [`::image::DynamicImage`] variant to an
/// 8-bit RGBA quad on the stack (no allocation).
///
/// The RGBA sibling of [`dynamic_image_rgb_pixel`]: keeps the alpha
/// channel instead of dropping it, for the [`resize`] source-buffer
/// materialization (which resizes in `U8x4` for parity with image-rs's
/// `imageops::resize` over a `DynamicImage`). `DynamicImage::get_pixel`
/// returns an `Rgba<u8>` regardless of the backing variant — image's
/// `dynamic_map!` dispatch performs the Luma-broadcast / 16-bit→8-bit /
/// float→u8 / opaque-alpha conversions, exactly what `to_rgba8()` did
/// per-pixel, so the borrowed-`as_rgba8()` fast path and this per-pixel
/// path produce byte-identical RGBA for the same source.
fn dynamic_image_rgba_pixel(img: &::image::DynamicImage, x: u32, y: u32) -> [u8; 4] {
  use ::image::GenericImageView as _;
  let p = img.get_pixel(x, y);
  [p.0[0], p.0[1], p.0[2], p.0[3]]
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
  let shape = arr.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "patchify: input must be rank-3 [H, W, C]",
      shape.len() as u32,
      shape,
    )));
  }
  if patch_size == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "patchify: patch_size",
      "must be > 0",
      "0",
    )));
  }
  let h = shape[0];
  let w = shape[1];
  let c = shape[2];
  if !h.is_multiple_of(patch_size) {
    return Err(Error::DivisibilityConstraint(
      DivisibilityConstraintPayload::new(
        "patchify: H by patch_size",
        "H",
        h as u64,
        "patch_size",
        patch_size as u64,
      ),
    ));
  }
  if !w.is_multiple_of(patch_size) {
    return Err(Error::DivisibilityConstraint(
      DivisibilityConstraintPayload::new(
        "patchify: W by patch_size",
        "W",
        w as u64,
        "patch_size",
        patch_size as u64,
      ),
    ));
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
  let n_patches = hp.checked_mul(wp).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "patchify: n_patches (hp * wp)",
      "usize",
      [("hp", hp as u64), ("wp", wp as u64)],
    ))
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
///
/// **Allocation contract.** This composer returns `Result<Array>` and
/// is fully **bounded-memory**: the source is capped by
/// [`MAX_DECODED_IMAGE_BYTES`] (512 MiB) via [`load_image`], and the
/// [`resize`] destination is capped against the same ceiling by
/// [`resize`]'s target-dimension guard (Codex review: `cfg.size` now
/// flows from an UNTRUSTED loaded processor config, so an over-budget /
/// zero / overflowing `size` surfaces as a typed
/// [`Error::ShapeMismatch`] from `resize` via `?` — NOT a process
/// abort). The remaining stages — [`image_to_array`] (f32 widening),
/// [`rescale`], and [`normalize_imagenet`] — are end-to-end recoverable
/// ([`Error::OutOfMemory`] / mlx backend `Result`). The only residual
/// abort path is `resize`'s two image-crate-internal `Vec` allocs
/// (`to_rgba8` clone, `Image::new` destination) under a genuine
/// system-wide OOM at a now-bounded ≤512 MiB request.
///
/// See the module-level audit table for the per-site breakdown.
// NOTE (Codex finding, round 2): a review request asked `preprocess`
// to return swift's `[1, C, H, W]` directly. The cross-model primitive
// always runs the ImageNet pipeline on `[H, W, 3]` (so the `(3,)`
// mean/std broadcast cleanly across the trailing axis — see the
// module-doc `Conventions > Channel layout` block). The trailing
// **planar layout** is now opt-in via [`ImageProcessorConfig::layout`]
// (a `Layout` enum defaulting to `Hwc` — pre-existing callers see no
// change; per-model encoders that want `[1, C, H, W]` request
// `Layout::Bchw` and the composer applies one lazy transpose +
// expand_dims at zero memory cost). See [`Layout`] for the per-arm
// rationale; tracking issue [#120](https://github.com/Findit-AI/mlxrs/issues/120).
pub fn preprocess(img: &::image::DynamicImage, cfg: &ImageProcessorConfig) -> Result<Array> {
  let resized;
  let src = if cfg.do_resize {
    resized = resize(img, cfg.size, cfg.resample)?;
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
  let arr = if cfg.do_normalize {
    normalize(&arr, &cfg.mean, &cfg.std)?
  } else {
    arr
  };
  apply_layout(arr, cfg.layout)
}

/// Apply the trailing tensor layout post-step to a channel-last
/// `[H, W, 3]` array (the canonical [`preprocess`] output).
///
/// **Rank precondition.** The post-step targets the cross-model
/// preprocessor's rank-3 channel-last output specifically — `Hwc`,
/// `Chw`, and `Bchw` are all defined as `H W 3` permutations / batch
/// expansions of that exact shape. Non-rank-3 / non-trailing-3-channel
/// inputs are rejected with [`Error::ShapeMismatch`] before any FFI
/// call; per-model processors that produce different ranks (patchified
/// `[N, P, P, 3]`, batched `[B, H, W, 3]`, etc.) compose their own
/// trailing layout via `transpose_axes` / `expand_dims_axes` directly.
///
/// **Ownership.** Takes `arr` by **value** so the [`Layout::Hwc`] arm
/// is a literal identity (returns the input as-is) without going
/// through [`Array::try_clone`] — `Array` is intentionally non-`Clone`
/// (every duplication is a fallible mlx FFI bump), and the
/// pre-existing `Hwc` default must not pay a `try_clone` cost just to
/// pass through.
///
/// **Lazy / zero-copy on the non-identity arms.** Both [`transpose_axes`]
/// and `expand_dims_axes` update the [`Array`]'s shape/stride metadata
/// without copying the underlying buffer (mlx's standard no-op view
/// semantics). The `Chw` arm is one metadata-only `transpose_axes`; the
/// `Bchw` arm is the same transpose + one `expand_dims_axes(&[0])`
/// (which inserts a leading unit axis, again metadata-only).
///
/// **Axis convention.** `transpose_axes(&[2, 0, 1])` permutes the
/// rank-3 input `[H, W, 3]` → `[3, H, W]`, matching the swift
/// `MediaProcessing.asMLXArray` line 190
/// (`array.reshape((1, h, w, 3)).transposed(0, 3, 1, 2)`) modulo the
/// batch axis that `expand_dims_axes(&[0])` adds for the `Bchw` arm.
///
/// # Errors
/// - [`Error::ShapeMismatch`] if the input is not rank-3 with a
///   trailing channel dim of 3.
///
/// Tracking issue: [#120](https://github.com/Findit-AI/mlxrs/issues/120)
/// (VLM-1).
pub fn apply_layout(arr: Array, layout: Layout) -> Result<Array> {
  use crate::ops::shape::expand_dims_axes;
  // Validate rank-3 `[H, W, 3]` shape before any FFI call — surfaces a
  // typed `ShapeMismatch` for batched / patchified / non-RGB inputs
  // rather than letting mlx's permutation-validation error message
  // surface (which would be less actionable).
  let shape = arr.shape();
  if shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "apply_layout: expected rank-3 [H, W, 3] input (per-model processors that produce \
         non-rank-3 layouts should compose transpose_axes / expand_dims_axes directly)",
      shape.len() as u32,
      shape,
    )));
  }
  if shape[2] != 3 {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "apply_layout: trailing channel dim (must be 3 for RGB)",
      3,
      shape[2],
    )));
  }
  match layout {
    Layout::Hwc => {
      // Identity — historical channel-last output. Pass through by
      // value: no `try_clone`, no allocation.
      Ok(arr)
    }
    Layout::Chw => {
      // [H, W, 3] → [3, H, W]. One metadata-only transpose.
      transpose_axes(&arr, &[2, 0, 1])
    }
    Layout::Bchw => {
      // [H, W, 3] → [3, H, W] → [1, 3, H, W]. Matches swift
      // `MediaProcessing.asMLXArray` (`MediaProcessing.swift:190` —
      // `array.reshape((1, h, w, 3)).transposed(0, 3, 1, 2)`); we
      // compose the equivalent via transpose + expand_dims since the
      // input is already `[H, W, 3]` (no reshape needed). Both ops
      // are metadata-only on the lazy `Array`.
      let chw = transpose_axes(&arr, &[2, 0, 1])?;
      expand_dims_axes(&chw, &[0])
    }
  }
}

/// Private regression tests for [`apply_orientation_fallible`] and
/// the round-5 truly-fallible [`rotate_buf`] helper.
///
/// **Round-5 coverage.** The four EXIF rotate orientations
/// (Rotate90 / Rotate270 / Rotate90FlipH / Rotate270FlipH) are
/// exercised on all four u8 [`DynamicImage`] variants
/// (Luma8 / LumaA8 / Rgb8 / Rgba8) and compared byte-for-byte
/// against the corresponding `image::imageops` reference output.
/// The non-square 4x3 test image makes 90° dimension swaps visible
/// at the test boundary (output dims must be 3x4, not 4x3).
///
/// These live inline (not in the `tests/vlm_image.rs` integration
/// suite) because [`apply_orientation_fallible`] / [`rotate_buf`]
/// are private — exposing them just to test would widen the public
/// surface for no caller benefit.
#[cfg(test)]
mod apply_orientation_tests {
  use super::*;
  use ::image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, RgbImage, Rgba,
    RgbaImage, metadata::Orientation,
  };

  /// Build a `width × height` Rgb8 image whose pixel values encode
  /// `(x, y)` so any rotation/flip is checkable byte-for-byte against
  /// the upstream `image::imageops` reference. Default test sizes use
  /// 4x3 (non-square) so 90° rotates flip dimensions visibly.
  fn xy_encoded_rgb8(width: u32, height: u32) -> DynamicImage {
    let mut buf = RgbImage::new(width, height);
    for y in 0..height {
      for x in 0..width {
        // (10*x, 10*y, 200) is unique-per-pixel for x,y < 25 and stays
        // well below 256, so rotate/flip orderings are visible at the
        // byte level.
        buf.put_pixel(x, y, Rgb([(x * 10) as u8, (y * 10) as u8, 200]));
      }
    }
    DynamicImage::ImageRgb8(buf)
  }

  /// Luma8 (single-channel u8) variant of [`xy_encoded_rgb8`].
  /// Encodes the row-major pixel index so rotation invariants are
  /// visible at the byte level — for a 4x3 image, source pixel
  /// `(x, y)` carries value `y * 4 + x` (0..12).
  fn xy_encoded_luma8(width: u32, height: u32) -> DynamicImage {
    let mut buf = GrayImage::new(width, height);
    for y in 0..height {
      for x in 0..width {
        buf.put_pixel(x, y, Luma([(y * width + x) as u8]));
      }
    }
    DynamicImage::ImageLuma8(buf)
  }

  /// LumaA8 (grayscale + alpha) variant: luma encodes `y*w + x`,
  /// alpha encodes `255 - (y*w + x)` so each byte position is
  /// individually traceable through the rotation.
  fn xy_encoded_luma_alpha8(width: u32, height: u32) -> DynamicImage {
    let mut buf = GrayAlphaImage::new(width, height);
    for y in 0..height {
      for x in 0..width {
        let v = (y * width + x) as u8;
        buf.put_pixel(x, y, LumaA([v, 255 - v]));
      }
    }
    DynamicImage::ImageLumaA8(buf)
  }

  /// Rgba8 variant: each channel is independently traceable
  /// (R = 10*x, G = 10*y, B = 200, A = 255 - 10*y) so the
  /// 4-byte stride is visible end-to-end.
  fn xy_encoded_rgba8(width: u32, height: u32) -> DynamicImage {
    let mut buf = RgbaImage::new(width, height);
    for y in 0..height {
      for x in 0..width {
        buf.put_pixel(
          x,
          y,
          Rgba([(x * 10) as u8, (y * 10) as u8, 200, 255 - (y * 10) as u8]),
        );
      }
    }
    DynamicImage::ImageRgba8(buf)
  }

  #[test]
  fn no_transforms_passes_through_unchanged() {
    // Identity-orientation path: no rotation overhead, no clone. The
    // returned image's raw bytes must match the source exactly
    // (rules out an accidental rotate dispatch).
    let img = xy_encoded_rgb8(4, 3);
    let original_bytes = img.as_rgb8().expect("rgb8 source").as_raw().clone();
    let out = apply_orientation_fallible(img, Orientation::NoTransforms).expect("infallible path");
    assert_eq!(out.width(), 4);
    assert_eq!(out.height(), 3);
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      &original_bytes
    );
  }

  #[test]
  fn rotate180_in_place_path_matches_reference() {
    // Rotate180 goes through `rotate180_in_place` upstream (no source-
    // sized alloc). Verify the dispatcher routes it through the
    // no-allocation arm AND that the pixel transform matches
    // `image::imageops::rotate180`.
    let img = xy_encoded_rgb8(4, 3);
    let reference: ImageBuffer<Rgb<u8>, Vec<u8>> =
      ::image::imageops::rotate180(img.as_rgb8().expect("rgb8 source"));
    let out =
      apply_orientation_fallible(img, Orientation::Rotate180).expect("in-place path infallible");
    assert_eq!(out.width(), 4, "Rotate180 preserves width");
    assert_eq!(out.height(), 3, "Rotate180 preserves height");
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
      "Rotate180 pixel bytes must match image::imageops::rotate180"
    );
  }

  #[test]
  fn flip_horizontal_in_place_path_matches_reference() {
    // FlipHorizontal goes through `fliph_in_place` upstream — verify
    // pixel-level parity with `image::imageops::flip_horizontal`.
    let img = xy_encoded_rgb8(4, 3);
    let reference = ::image::imageops::flip_horizontal(img.as_rgb8().expect("rgb8 source"));
    let out = apply_orientation_fallible(img, Orientation::FlipHorizontal)
      .expect("in-place path infallible");
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
      "FlipHorizontal pixel bytes must match image::imageops::flip_horizontal"
    );
  }

  // -- Rgb8 manual rotation parity (round-5) ---------------------------

  #[test]
  fn rotate90_rgb8_manual_matches_reference() {
    // ROUND-5: Rotate90 on Rgb8 now goes through the truly-fallible
    // manual `rotate_buf` path. Must:
    //   (1) succeed for a small input under `MAX_DECODED_IMAGE_BYTES`,
    //   (2) swap dimensions (4x3 → 3x4),
    //   (3) produce byte-identical pixels to `image::imageops::rotate90`.
    let img = xy_encoded_rgb8(4, 3);
    let reference = ::image::imageops::rotate90(img.as_rgb8().expect("rgb8 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate90)
      .expect("manual rotate90 should succeed for a 4x3 image well under the 512 MiB ceiling");
    assert_eq!(out.width(), 3, "Rotate90 swaps width <- height");
    assert_eq!(out.height(), 4, "Rotate90 swaps height <- width");
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
      "manual rotate must yield byte-identical pixels to image::imageops::rotate90"
    );
  }

  #[test]
  fn rotate270_rgb8_manual_matches_reference() {
    let img = xy_encoded_rgb8(4, 3);
    let reference = ::image::imageops::rotate270(img.as_rgb8().expect("rgb8 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate270).expect("rotate270 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
    );
  }

  #[test]
  fn rotate90_fliph_rgb8_composite_matches_reference() {
    // Rotate90FlipH = rotate90 + fliph (the composite collapses to
    // `dst.put_pixel(y, x, src.get_pixel(x, y))` in `rotate_buf`).
    // Verify the composite output is byte-identical to the
    // image-rs dispatch.
    let img = xy_encoded_rgb8(4, 3);
    let rotated = ::image::imageops::rotate90(img.as_rgb8().expect("rgb8 source"));
    let reference = ::image::imageops::flip_horizontal(&rotated);
    let out = apply_orientation_fallible(img, Orientation::Rotate90FlipH)
      .expect("Rotate90FlipH composite ok");
    assert_eq!(out.width(), reference.width());
    assert_eq!(out.height(), reference.height());
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
    );
  }

  #[test]
  fn rotate270_fliph_rgb8_composite_matches_reference() {
    let img = xy_encoded_rgb8(4, 3);
    let rotated = ::image::imageops::rotate270(img.as_rgb8().expect("rgb8 source"));
    let reference = ::image::imageops::flip_horizontal(&rotated);
    let out = apply_orientation_fallible(img, Orientation::Rotate270FlipH)
      .expect("Rotate270FlipH composite ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_rgb8().expect("rgb8 output").as_raw(),
      reference.as_raw(),
    );
  }

  // -- Rgba8 manual rotation parity (round-5) --------------------------

  #[test]
  fn rotate90_rgba8_manual_matches_reference() {
    // Rgba8 has 4 channels (R, G, B, A) — verify the per-pixel
    // 4-byte memcpy in `rotate_buf` preserves the alpha channel
    // alongside RGB.
    let img = xy_encoded_rgba8(4, 3);
    let reference = ::image::imageops::rotate90(img.as_rgba8().expect("rgba8 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("rgba8 rotate90 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_rgba8().expect("rgba8 output").as_raw(),
      reference.as_raw(),
      "manual Rgba8 rotate90 must preserve all 4 channels byte-identical to image::imageops::rotate90"
    );
  }

  #[test]
  fn rotate270_fliph_rgba8_composite_matches_reference() {
    // Composite path × alpha channel — adversarial: verify the
    // collapsed `dst[h-1-y, w-1-x] = src[x, y]` index math handles
    // the 4-byte stride correctly.
    let img = xy_encoded_rgba8(4, 3);
    let rotated = ::image::imageops::rotate270(img.as_rgba8().expect("rgba8 source"));
    let reference = ::image::imageops::flip_horizontal(&rotated);
    let out = apply_orientation_fallible(img, Orientation::Rotate270FlipH)
      .expect("rgba8 Rotate270FlipH ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_rgba8().expect("rgba8 output").as_raw(),
      reference.as_raw(),
    );
  }

  // -- Luma8 manual rotation parity (round-5) --------------------------

  #[test]
  fn rotate90_luma8_manual_matches_reference() {
    // Luma8 has 1 channel — verify the per-pixel 1-byte memcpy in
    // `rotate_buf` produces the same byte order as image-rs.
    let img = xy_encoded_luma8(4, 3);
    let reference = ::image::imageops::rotate90(img.as_luma8().expect("luma8 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("luma8 rotate90 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_luma8().expect("luma8 output").as_raw(),
      reference.as_raw(),
      "manual Luma8 rotate90 must match image::imageops::rotate90 byte-identical"
    );
  }

  #[test]
  fn rotate270_luma8_manual_matches_reference() {
    let img = xy_encoded_luma8(4, 3);
    let reference = ::image::imageops::rotate270(img.as_luma8().expect("luma8 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate270).expect("luma8 rotate270 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_luma8().expect("luma8 output").as_raw(),
      reference.as_raw(),
    );
  }

  // -- LumaA8 manual rotation parity (round-5) -------------------------

  #[test]
  fn rotate90_luma_alpha8_manual_matches_reference() {
    // LumaA8 has 2 channels — verify the 2-byte stride.
    let img = xy_encoded_luma_alpha8(4, 3);
    let reference = ::image::imageops::rotate90(img.as_luma_alpha8().expect("luma_alpha8 source"));
    let out =
      apply_orientation_fallible(img, Orientation::Rotate90).expect("luma_alpha8 rotate90 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_luma_alpha8().expect("luma_alpha8 output").as_raw(),
      reference.as_raw(),
      "manual LumaA8 rotate90 must match image::imageops::rotate90 byte-identical"
    );
  }

  // -- Round-6 16-bit-PNG / float-pixel rotate parity ------------------
  //
  // image-rs 0.25's PNG decoder emits `Luma16`/`LumaA16`/`Rgb16`/
  // `Rgba16` for `BitDepth::Sixteen` PNG inputs (`codecs/png.rs:64-71`),
  // and caller-supplied `DynamicImage::ImageRgb32F`/`ImageRgba32F`
  // round-trip through the same fallible rotate path. Round 5
  // missed these and left them on an infallible image-rs delegate
  // that could abort on allocator pressure for a 16-bit PNG with
  // EXIF Rotate90/270 (the load_image entry-point closure was
  // broken). Round 6 generalizes the manual rotate over `T: Copy`
  // so the same `try_reserve_exact` gate covers u16 and f32 — the
  // tests below verify byte-identical parity with `image::imageops`
  // for the new element widths.

  /// Build a non-square `Rgb16` image whose subpixels encode pixel
  /// position so any rotation is checkable byte-for-byte against
  /// `image::imageops::rotate*`. Uses the high byte of each u16 so
  /// the values survive any accidental u8-truncation regression and
  /// remain unique-per-pixel for small sizes.
  fn xy_encoded_rgb16(width: u32, height: u32) -> DynamicImage {
    let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(width, height);
    for y in 0..height {
      for x in 0..width {
        // (256 * (10*x + 1), 256 * (10*y + 1), 0xBEEF) — three
        // independent u16 channels with non-zero high bytes so a
        // hypothetical truncation to u8 would zero out the channel
        // distinctly.
        buf.put_pixel(
          x,
          y,
          Rgb([
            (((x * 10) as u16) + 1) << 8,
            (((y * 10) as u16) + 1) << 8,
            0xBEEF,
          ]),
        );
      }
    }
    DynamicImage::ImageRgb16(buf)
  }

  /// `Luma16` (single-channel u16) test image: encodes the
  /// row-major pixel index in the high byte plus a fixed low-byte
  /// pattern so any rotation is byte-traceable.
  fn xy_encoded_luma16(width: u32, height: u32) -> DynamicImage {
    let mut buf: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(width, height);
    for y in 0..height {
      for x in 0..width {
        let idx = (y * width + x) as u16;
        // (idx << 8) | 0x5A — high byte = row-major index, low byte
        // = constant 0x5A so a stride-1 u8-truncation would lose
        // identity.
        buf.put_pixel(x, y, Luma([(idx << 8) | 0x005A]));
      }
    }
    DynamicImage::ImageLuma16(buf)
  }

  /// `Rgb32F` test image: encodes pixel position in three f32
  /// channels with deliberately fractional values so any 4-byte
  /// stride bug surfaces as a visible bit-pattern shift.
  fn xy_encoded_rgb32f(width: u32, height: u32) -> DynamicImage {
    let mut buf: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(width, height);
    for y in 0..height {
      for x in 0..width {
        // (x + 0.25, y + 0.5, 0.875) — distinct fractions in each
        // channel; 0.875 is exactly representable so equality
        // comparison against the rotated buffer is exact.
        buf.put_pixel(x, y, Rgb([(x as f32) + 0.25, (y as f32) + 0.5, 0.875]));
      }
    }
    DynamicImage::ImageRgb32F(buf)
  }

  #[test]
  fn rotate90_rgb16_manual_matches_reference() {
    // ROUND-6: Rotate90 on Rgb16 now routes through the same
    // truly-fallible `rotate_buf<u16>` path the u8 variants use.
    // A non-square 4x3 image must rotate to 3x4 with byte-identical
    // pixels to `image::imageops::rotate90`. This is the regression
    // test for the 16-bit-PNG-EXIF-rotate `load_image` gap the audit
    // missed in round 5.
    let img = xy_encoded_rgb16(4, 3);
    let reference = ::image::imageops::rotate90(img.as_rgb16().expect("rgb16 source"));
    let out = apply_orientation_fallible(img, Orientation::Rotate90)
      .expect("manual Rgb16 rotate90 should succeed under MAX_DECODED_IMAGE_BYTES");
    assert_eq!(out.width(), 3, "Rotate90 swaps width <- height");
    assert_eq!(out.height(), 4, "Rotate90 swaps height <- width");
    assert_eq!(
      out.as_rgb16().expect("rgb16 output").as_raw(),
      reference.as_raw(),
      "manual Rgb16 rotate90 must yield byte-identical u16 subpixels to image::imageops::rotate90"
    );
  }

  #[test]
  fn rotate270_luma16_manual_matches_reference() {
    // Rotate270 on Luma16 — single-channel u16 path. Verifies the
    // 1-subpixel-per-pixel stride is honored for the u16 element
    // type. Non-square 4x3 → 3x4.
    let img = xy_encoded_luma16(4, 3);
    let reference = ::image::imageops::rotate270(img.as_luma16().expect("luma16 source"));
    let out =
      apply_orientation_fallible(img, Orientation::Rotate270).expect("manual Luma16 rotate270 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_luma16().expect("luma16 output").as_raw(),
      reference.as_raw(),
      "manual Luma16 rotate270 must yield byte-identical u16 subpixels to image::imageops::rotate270"
    );
  }

  #[test]
  fn rotate90_rgb32f_manual_matches_reference() {
    // Rotate90 on Rgb32F — verifies the `rotate_buf<f32>` arm.
    // Comparison is bit-exact via `Vec<f32>` equality because the
    // operation is a pure permutation (no arithmetic, no FP drift).
    let img = xy_encoded_rgb32f(4, 3);
    let reference = ::image::imageops::rotate90(img.as_rgb32f().expect("rgb32f source"));
    let out =
      apply_orientation_fallible(img, Orientation::Rotate90).expect("manual Rgb32F rotate90 ok");
    assert_eq!(out.width(), 3);
    assert_eq!(out.height(), 4);
    assert_eq!(
      out.as_rgb32f().expect("rgb32f output").as_raw(),
      reference.as_raw(),
      "manual Rgb32F rotate90 must yield bit-identical f32 subpixels to image::imageops::rotate90 \
       (permutation only — no FP arithmetic)"
    );
  }

  #[test]
  fn synthetic_16bit_rotate_succeeds_via_orientation_entry_point() {
    // End-to-end check that mimics the broken-by-round-5 load_image
    // closure: build a 16-bit `DynamicImage` directly (no on-disk
    // PNG needed; a unit test does not need a real decoder to
    // exercise the post-decode rotate path) and hand it to
    // `apply_orientation_fallible` with EXIF Rotate90 / Rotate270 /
    // composite orientations. All four allocating-rotate
    // orientations must now succeed via the generic `rotate_buf`
    // path — round 5 would have hit the infallible fallback for
    // these. (The "real 16-bit PNG decode" path is exercised
    // implicitly through `load_image` once a 16-bit PNG with EXIF
    // orientation is supplied; we test the rotate-shaped subgoal
    // here without coupling to disk I/O.)
    for rotation in [
      Orientation::Rotate90,
      Orientation::Rotate270,
      Orientation::Rotate90FlipH,
      Orientation::Rotate270FlipH,
    ] {
      let img = xy_encoded_rgb16(4, 3);
      let out = apply_orientation_fallible(img, rotation)
        .expect("16-bit Rgb16 rotate must succeed via the round-6 fallible u16 path");
      assert_eq!(out.width(), 3, "all four rotate orientations swap w<->h");
      assert_eq!(out.height(), 4);
      assert!(
        matches!(out, DynamicImage::ImageRgb16(_)),
        "output must preserve the source DynamicImage variant (Rgb16 in, Rgb16 out)"
      );
    }
  }

  // -- Byte-budget gate -----------------------------------------------

  #[test]
  fn rotate90_accepts_small_image_within_budget() {
    // The byte-budget gate against `MAX_DECODED_IMAGE_BYTES` must
    // accept inputs that fit. Sanity check: a 1x1 Rgb8 image is
    // 3 bytes, well under the 512 MiB cap. The negative
    // (overflow → ShapeMismatch) path uses the identical
    // `checked_mul` chain that `pad_to_square`'s overflow tests
    // already cover; we can't construct an image at hostile
    // dimensions without OOM-ing the test process itself.
    let img = xy_encoded_rgb8(1, 1);
    let bytes_per_pixel = u64::from(img.color().bytes_per_pixel());
    let bytes = u64::from(img.width()) * u64::from(img.height()) * bytes_per_pixel;
    assert!(
      bytes <= MAX_DECODED_IMAGE_BYTES,
      "1x1 Rgb8 = {bytes} bytes must be well under MAX_DECODED_IMAGE_BYTES={MAX_DECODED_IMAGE_BYTES}"
    );
    let out = apply_orientation_fallible(img, Orientation::Rotate90).expect("1x1 ok");
    assert_eq!(out.width(), 1);
    assert_eq!(out.height(), 1);
  }
}
