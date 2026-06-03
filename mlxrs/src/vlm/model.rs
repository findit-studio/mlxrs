//! The architecture-agnostic [`Model`] seam for `mlxrs::vlm` multimodal
//! generation, extending [`crate::lm::model::Model`] with the
//! image-embedding entry points every VLM forward needs (vision encode +
//! token embed + image-into-text embed splice).
//!
//! Mirrors mlx-vlm's `VisionLanguageModel` per-model protocol
//! (`mlx-vlm/mlx_vlm/models/base.py` + each model's `get_input_embeddings`
//! and `merge_input_ids_with_image_features` — see e.g. `pixtral.py:41-153`)
//! and mlx-swift-lm's
//! [`VLMModel`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/VLMModel.swift)
//! marker protocol (`VLMModel: LanguageModel, LoRAModel`). Per the project's
//! no-per-model-arch rule (`feedback_no_per_model_arch_porting`), mlxrs does
//! NOT ship concrete VLM model implementations — this trait is the *shape*
//! per-model code (Qwen3-VL / LFM2-VL / LLaVA / etc.) must conform to so the
//! [`crate::vlm::generate::vlm_generate`] loop can drive any architecture
//! uniformly.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, EmptyInputPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result, try_with_capacity,
  },
  ops,
  vlm::image::{ImageProcessorConfig, preprocess},
};

/// Turn a raw decoded image into the model's native vision input plus the
/// number of image-token positions that image expands to.
///
/// A loaded preprocessing artifact, mirroring mlx-vlm's per-model
/// `processing_<model>.py` (the processor object `generate(model, processor,
/// …)` carries alongside the model). A fixed-grid CLIP/SigLIP/LLaVA model uses
/// the default [`Model::image_processor`] — a [`FixedGridProcessor`] that runs
/// the cross-model [`crate::vlm::image::preprocess`] pipeline and reports a
/// constant per-image grid count. A native-resolution model (LFM2.5-VL NaFlex,
/// the future Qwen3.5-VL line) returns its own [`ImageProcessor`] whose
/// [`process`](Self::process) builds the variable patch tensor + its
/// per-image token count.
///
/// Object-safe (the [`process`](Self::process) method takes `&self`, borrows
/// the decoded image, and returns an owned [`ProcessedImage`]) so a model can
/// hand back a `Box<dyn ImageProcessor>` and the
/// [`crate::vlm::generate::vlm_generate`] loop can drive it behind a `&dyn
/// ImageProcessor`.
///
/// **Why the decoded `image::DynamicImage`, not a `[H, W, 3]` [`Array`].** The
/// raw decoded image is the natural processor input here, exactly as mlx-vlm's
/// `processing_<model>.py` receives PIL images. The fixed-grid default runs the
/// cross-model [`crate::vlm::image::preprocess`] pipeline whose resize is the
/// PIL-bit-exact image-crate kernel (`vlm::resize`) operating on a
/// `DynamicImage` — so the default processor reproduces the historical
/// fixed-grid result byte-for-byte by handing the same `DynamicImage` straight
/// to `preprocess`; an `[H, W, 3]` Array could not be PIL-resized without
/// diverging. The native-resolution processor likewise reads the decoded
/// image's interleaved RGB bytes + dimensions for its NaFlex smart-resize. The
/// resulting model-native tensor IS an [`Array`]
/// ([`ProcessedImage::pixels`]).
pub trait ImageProcessor {
  /// Process one decoded image (the output of
  /// [`crate::vlm::image::load_image`]) into a [`ProcessedImage`] carrying the
  /// model's native pixel/patch tensor, any native-resolution companions, and
  /// the image-token count this image occupies.
  ///
  /// # Errors
  /// Propagates the per-model preprocessing failure (resize / patchify /
  /// normalize / overflow / allocation) as a typed [`crate::Error`].
  fn process(&self, image: &::image::DynamicImage) -> Result<ProcessedImage>;
}

/// One image's preprocessed vision input, as produced by an
/// [`ImageProcessor`] and consumed by [`Model::encode_image`].
///
/// Carries the model's native pixel/patch tensor ([`pixels`](Self::pixels)),
/// the optional native-resolution companions ([`native`](Self::native), `None`
/// for a fixed-grid model), and the number of image-token positions this image
/// expands to ([`num_tokens`](Self::num_tokens) — variable for a
/// native-resolution model, constant for a fixed grid).
///
/// Constructed by the [processor](ImageProcessor); read by the model's
/// [`encode_image`](Model::encode_image) (`pixels` always; `native` when the
/// model requires the native-resolution companions) and by
/// [`crate::vlm::generate::vlm_generate`] (`num_tokens` to size the prompt's
/// placeholder run for this image).
#[non_exhaustive]
pub struct ProcessedImage {
  /// The model's native pixel/patch tensor: a fixed-grid `[1, H, W, C]`
  /// preprocessed image for a CLIP/SigLIP-style model, or a native-resolution
  /// `[1, num_patches, patch_feat]` flattened patch tensor for a NaFlex model.
  pixels: Array,
  /// The native-resolution companions (`spatial_shape` + `patch_mask`); `None`
  /// for a fixed-grid model whose `pixels` is fully self-describing.
  native: Option<NativeResolution>,
  /// The number of image-token positions this image expands to (variable for a
  /// native-resolution model, a constant per-image grid count for a fixed
  /// grid). [`crate::vlm::generate::vlm_generate`] splices exactly this many
  /// placeholder ids for this image, and [`Model::encode_image`] must return
  /// exactly this many feature rows.
  num_tokens: usize,
}

impl ProcessedImage {
  /// Assemble a [`ProcessedImage`] from its parts — the native pixel/patch
  /// tensor, the optional native-resolution companions (`None` for a
  /// fixed-grid model), and the per-image token count.
  #[inline(always)]
  pub const fn new(pixels: Array, native: Option<NativeResolution>, num_tokens: usize) -> Self {
    Self {
      pixels,
      native,
      num_tokens,
    }
  }

  /// The model's native pixel/patch tensor.
  #[inline(always)]
  pub const fn pixels(&self) -> &Array {
    &self.pixels
  }

  /// The native-resolution companions, or `None` for a fixed-grid model.
  #[inline(always)]
  pub const fn native(&self) -> Option<&NativeResolution> {
    self.native.as_ref()
  }

  /// The number of image-token positions this image expands to.
  #[inline(always)]
  pub const fn num_tokens(&self) -> usize {
    self.num_tokens
  }
}

/// The native-resolution companions an [`ImageProcessor`] attaches to a
/// [`ProcessedImage`] for a NaFlex-style model — the `spatial_shape` and
/// `patch_mask` the vision tower consumes alongside the flattened patch
/// tensor.
///
/// `None` on a fixed-grid [`ProcessedImage`]; `Some` for a native-resolution
/// model (LFM2.5-VL, the future Qwen3.5-VL line). A model that requires these
/// reads them in [`Model::encode_image`] via
/// `image.native().ok_or(...)?` — a missing companion is a typed
/// [`Error::InvariantViolation`], never a panic.
#[non_exhaustive]
pub struct NativeResolution {
  /// The patch grid dimensions the vision tower reshapes the active rows to —
  /// e.g. LFM2.5-VL's `(2,)` i32 `[H_p, W_p]` `spatial_shapes`.
  spatial_shape: Array,
  /// The per-patch attention mask (`1` for active patch rows, `0` for padding)
  /// — e.g. LFM2.5-VL's `(max_num_patches,)` i32 `pixel_attention_mask`.
  patch_mask: Array,
}

impl NativeResolution {
  /// Assemble a [`NativeResolution`] from the `spatial_shape` + `patch_mask`
  /// companions.
  #[inline(always)]
  pub const fn new(spatial_shape: Array, patch_mask: Array) -> Self {
    Self {
      spatial_shape,
      patch_mask,
    }
  }

  /// The patch grid dimensions (`[H_p, W_p]`).
  #[inline(always)]
  pub const fn spatial_shape(&self) -> &Array {
    &self.spatial_shape
  }

  /// The per-patch attention mask.
  #[inline(always)]
  pub const fn patch_mask(&self) -> &Array {
    &self.patch_mask
  }
}

/// A vision-language model: a [`crate::lm::model::Model`] augmented with
/// image-embedding inputs.
///
/// Per-model concrete impls (Qwen-VL / LLaVA / Idefics / …) own:
/// - the vision tower forward in [`Self::encode_image`],
/// - the token-embedding lookup in [`Self::embed_tokens`],
/// - the model-input pixel layout in [`Self::image_processor_config`].
///
/// They typically inherit [`Self::merge_embeddings`]'s default span-replace
/// splice and only override it when the splice is embedding-space-specific
/// (e.g. a learned projection between text and image embeds — Pixtral does
/// a per-batch cumsum/gather variant at `pixtral.py:104-153`; the default
/// here mirrors the simpler `mlx-vlm`-side post-projector splice every
/// Qwen-VL-family model uses).
pub trait Model: crate::lm::model::Model {
  /// Token-id → text-embedding lookup (the LM's `embed_tokens` layer).
  ///
  /// `tokens` is the assembled prompt as a `[1, T]` integer `Array` (the
  /// same shape [`crate::lm::generate`]'s loop feeds
  /// [`crate::lm::model::Model::forward`]).
  /// Returns the LM's text embeddings `[1, T, D]` where `D` is the LM's
  /// hidden dim.
  ///
  /// Mirrors `language_model.model.embed_tokens(input_ids)` in
  /// `mlx-vlm/mlx_vlm/models/*/get_input_embeddings` (e.g.
  /// `pixtral.py:54`). Required because [`crate::vlm::generate`] needs
  /// the text embeddings as a separate value to splice image embeds INTO
  /// before dispatching through [`crate::lm::model::Model::forward_embeddings`].
  fn embed_tokens(&self, tokens: &Array) -> Result<Array>;

  /// Encode one preprocessed image (the [`ProcessedImage`] an
  /// [`ImageProcessor`] produced) into vision-encoder embeddings, shape
  /// `[N, D]` where `N` is the image-token count this image expands to
  /// (`image.num_tokens()` — variable for a native-resolution model,
  /// constant for a fixed grid) and `D` is the LM's hidden dim.
  ///
  /// Per-model encoders (CLIP / SigLIP / Qwen-VL ViT / NaFlex ViT / etc.)
  /// implement this. The model always reads [`image.pixels()`](ProcessedImage::pixels)
  /// — most commonly `[1, H, W, 3]` for a fixed-grid model, or the flattened
  /// `[1, num_patches, patch_feat]` patch tensor for a native-resolution one.
  /// A native-resolution model additionally reads
  /// [`image.native()`](ProcessedImage::native) for the `spatial_shape` /
  /// `patch_mask` companions; it requires them via
  /// `image.native().ok_or(Error::InvariantViolation(...))?` (a typed error,
  /// never a panic, when a caller hands it a fixed-grid [`ProcessedImage`]).
  /// Per-model code converts the pixel layout inside its own `encode_image`
  /// (e.g. `transpose_axes(&[2, 0, 1])` + add batch) so the cross-model
  /// surface stays layout-agnostic.
  ///
  /// Mirrors `vision_tower(pixel_values.transpose(0, 2, 3, 1), …)` +
  /// `multi_modal_projector(selected_image_feature)` in
  /// `mlx-vlm/mlx_vlm/models/pixtral/pixtral.py:60-77` (and the per-image
  /// NaFlex `vision_tower(pixel_values, spatial_shapes)` body in
  /// `mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py:141-153`).
  fn encode_image(&self, image: &ProcessedImage) -> Result<Array>;

  /// Splice `image_embeds` into the LM's `text_embeds` at the positions
  /// identified by `image_spans` (the spans returned by
  /// [`crate::vlm::prompt::assemble_multimodal_prompt`]). Returns the
  /// merged embedding sequence `[1, T, D]` ready to feed to
  /// [`crate::lm::model::Model::forward_embeddings`].
  ///
  /// **Default**: a direct slice-replace splice — for each `(start, end)`
  /// span, the matching `(end - start, D)` slab of `image_embeds` (taken
  /// contiguously in span order) replaces `text_embeds[:, start..end, :]`.
  /// The output is assembled by `concatenate`-ing the alternating
  /// text/image slices along the sequence axis (no in-place mutation,
  /// faithful to mlx's lazy-graph contract). This mirrors the
  /// `mlx-vlm`-side splice every Qwen-VL-family `get_input_embeddings`
  /// uses post-projector (after the vision-tower features have already
  /// been mapped to the LM's hidden dim).
  ///
  /// **Override** when the splice is embedding-space-specific — e.g.
  /// Pixtral's per-batch cumsum/gather variant (`pixtral.py:104-153`) or a
  /// model that fuses a learned projection into the merge step. Per-model
  /// code calls back into [`Self::encode_image`] /
  /// [`Self::embed_tokens`] and composes its own merge using the
  /// `mlxrs::ops` primitives.
  ///
  /// # Errors
  ///
  /// - `Error::RankMismatch` if `text_embeds` is not rank-3 or `image_embeds`
  ///   is not rank-2; `Error::LengthMismatch` if the batch dim is not 1 or
  ///   the hidden dims `D` differ.
  /// - `Error::LengthMismatch` if the sum of all span widths
  ///   `Σ(end - start)` differs from `image_embeds`' first axis `N`
  ///   (one image-feature per placeholder position is the splice
  ///   contract).
  /// - `Error::InvariantViolation` if any span is empty or overlaps the
  ///   previous span; `Error::OutOfRange` if a span end exceeds `T`
  ///   (mirrors [`crate::vlm::prompt::build_multimodal_mask`]'s validation).
  /// - `Error::EmptyInput` if `image_spans` is empty — there are no
  ///   positions to merge into and the caller should use the
  ///   no-image text path (`forward(tokens)`) instead.
  fn merge_embeddings(
    &self,
    text_embeds: &Array,
    image_embeds: &Array,
    image_spans: &[(usize, usize)],
  ) -> Result<Array> {
    default_merge_embeddings(text_embeds, image_embeds, image_spans)
  }

  /// Run the LM prefill over one chunk of merged multimodal embeddings,
  /// with access to the chunk's per-image span layout AND the cache
  /// offset for mask-requiring models.
  ///
  /// `embeddings` is the chunk's merged sequence `[1, chunk_len, D]`
  /// (output of [`Self::merge_embeddings`] for this chunk).
  /// `image_spans` are the **chunk-local** half-open `(start, end)`
  /// ranges in `[0, chunk_len)` — the caller (`vlm_generate`) shifts
  /// absolute spans by the chunk's start offset and guarantees no span
  /// is split across a chunk boundary. `cache_offset` is the number of
  /// tokens ALREADY in `cache` before this chunk (the chunk's absolute
  /// start position), so a mask-building override can size the
  /// attention mask `[chunk_len × (cache_offset + chunk_len)]` over
  /// past + current keys. `cache` is the LM's per-layer KV cache,
  /// mutated in place.
  ///
  /// **Default**: dispatches to [`crate::lm::model::Model::forward_embeddings`]
  /// — IGNORING `image_spans` and `cache_offset`. The vast majority of
  /// VLMs (Qwen-VL family, LLaVA, Idefics, etc.) consume merged
  /// embeddings under a pure causal attention mask, exactly like
  /// text-only generation: the image-span identity is already baked into
  /// the merged embeddings before the LM sees them, the cache's own
  /// position bookkeeping supplies the causal offset, and no further
  /// mask work is needed.
  ///
  /// **Override** when the model needs the multimodal mask — e.g.
  /// falcon-ocr-style models that require bidirectional attention
  /// WITHIN each image span (see
  /// [`crate::vlm::prompt::build_multimodal_mask_with_past`] for the
  /// chunked formula). The override builds the
  /// `[chunk_len × (cache_offset + chunk_len)]` mask from the
  /// chunk-local `image_spans` + `cache_offset` and threads it into its
  /// per-model attention layer through the model's own internal API. It
  /// does NOT store the spans/offset on `&self` (which would create
  /// cross-request / interleaved-iterator hazards: two `vlm_generate`
  /// iterators constructed against the same model, polled out of order,
  /// would otherwise swap each other's mask state) — every per-chunk
  /// value arrives by argument.
  ///
  /// Mirrors the per-model `__call__` that consumes
  /// `inputs_embeds=...` plus per-model mask kwargs in mlx-vlm (e.g.
  /// `pixtral.py:170-184` — `language_model(input_ids, cache=cache,
  /// inputs_embeds=...)` plus optional `mask=...` kwarg routing).
  fn forward_embeddings_multimodal(
    &self,
    embeddings: &Array,
    _image_spans: &[(usize, usize)],
    _cache_offset: usize,
    cache: &mut [Box<dyn crate::lm::cache::KvCache>],
  ) -> Result<Array> {
    crate::lm::model::Model::forward_embeddings(self, embeddings, cache)
  }

  /// The image-processor config this model expects (mean / std / size /
  /// resize-filter / channel order).
  ///
  /// Default returns the [`ImageProcessorConfig::default`] — ImageNet
  /// baseline at 224×224 RGB Bicubic — which matches nearly every
  /// CLIP/SigLIP/DINO/ViT preprocessing config. Per-model overrides for
  /// non-standard configs (Qwen-VL uses 448×448 Bilinear, some HF models
  /// override the mean/std, etc.); per-model code returns its own value
  /// loaded from the model's `preprocessor_config.json`.
  ///
  /// Still used by the default [`image_processor`](Self::image_processor)
  /// (the [`FixedGridProcessor`] drives [`crate::vlm::image::preprocess`] off
  /// this config); a model that overrides `image_processor` with its own
  /// native-resolution processor may leave this at a nominal value.
  fn image_processor_config(&self) -> ImageProcessorConfig {
    ImageProcessorConfig::default()
  }

  /// The per-model [`ImageProcessor`] [`crate::vlm::generate::vlm_generate`]
  /// runs each raw loaded image through to obtain its native vision input +
  /// per-image token count, mirroring mlx-vlm's per-model
  /// `processing_<model>.py`.
  ///
  /// **Default**: a [`FixedGridProcessor`] that reproduces the historical
  /// fixed-grid behavior exactly — it runs the cross-model
  /// [`crate::vlm::image::preprocess`] pipeline (driven by
  /// [`image_processor_config`](Self::image_processor_config)) to get the
  /// resized `[1, H, W, C]` (well, `[H, W, C]` — the `Hwc`/`Chw`/`Bchw`
  /// layout the config selects) and wraps it as a [`ProcessedImage`] with
  /// `native: None` and a constant per-image token count `num_tokens`. A
  /// fixed-grid VLM that does not override this gets behavior identical to the
  /// previous single-`num_tokens_per_image` path.
  ///
  /// **Override** for a native-resolution model (LFM2.5-VL NaFlex, the future
  /// Qwen3.5-VL line) whose per-image patch tensor + token count are computed
  /// from the source dimensions: return a model-specific `Box<dyn
  /// ImageProcessor>` whose [`process`](ImageProcessor::process) builds the
  /// variable patch tensor, attaches the
  /// [`NativeResolution`] companions, and reports the per-image
  /// `num_tokens`.
  ///
  /// `num_tokens` is the per-image image-token count the default fixed-grid
  /// processor reports for EVERY image — the constant grid size a
  /// CLIP/SigLIP/LLaVA encoder emits (e.g. `(image_size / patch_size)^2`,
  /// the value the previous `VlmGenConfig::num_tokens_per_image` carried).
  fn image_processor(&self, num_tokens: usize) -> Box<dyn ImageProcessor> {
    Box::new(FixedGridProcessor::new(
      self.image_processor_config(),
      num_tokens,
    ))
  }
}

/// The default fixed-grid [`ImageProcessor`] — runs the cross-model
/// [`crate::vlm::image::preprocess`] pipeline over a loaded image and reports a
/// constant per-image grid token count.
///
/// This is what [`Model::image_processor`] returns by default, so a fixed-grid
/// CLIP/SigLIP/LLaVA VLM that does not override `image_processor` gets behavior
/// identical to the historical single-`num_tokens_per_image`
/// [`crate::vlm::generate::vlm_generate`] path: each raw `[H, W, 3]` image is
/// run through [`crate::vlm::image::preprocess`] (driven by the model's
/// [`ImageProcessorConfig`]) and wrapped as a [`ProcessedImage`] with
/// `native: None` and the constant [`num_tokens`](Self::num_tokens).
pub struct FixedGridProcessor {
  config: ImageProcessorConfig,
  num_tokens: usize,
}

impl FixedGridProcessor {
  /// Build a fixed-grid processor from the model's [`ImageProcessorConfig`]
  /// (the preprocessing params) and the constant per-image token count.
  #[inline(always)]
  pub const fn new(config: ImageProcessorConfig, num_tokens: usize) -> Self {
    Self { config, num_tokens }
  }

  /// The preprocessing config driving [`crate::vlm::image::preprocess`].
  #[inline(always)]
  pub const fn config(&self) -> &ImageProcessorConfig {
    &self.config
  }

  /// The constant per-image grid token count this processor reports.
  #[inline(always)]
  pub const fn num_tokens(&self) -> usize {
    self.num_tokens
  }
}

impl ImageProcessor for FixedGridProcessor {
  /// Run the cross-model [`crate::vlm::image::preprocess`] pipeline over the
  /// decoded image and wrap the resized tensor as a fixed-grid
  /// [`ProcessedImage`] (`native: None`, the constant
  /// [`num_tokens`](Self::num_tokens)).
  ///
  /// Byte-identical to the historical [`crate::vlm::generate::vlm_generate`]
  /// fixed-grid step: `preprocess(&img, self.config())` reproduces the same
  /// resize / rescale / normalize / layout result the previous code computed
  /// from the loaded `DynamicImage` and the model's [`ImageProcessorConfig`].
  fn process(&self, image: &::image::DynamicImage) -> Result<ProcessedImage> {
    let pixels = preprocess(image, &self.config)?;
    Ok(ProcessedImage::new(pixels, None, self.num_tokens))
  }
}

/// Default span-replace splice for [`Model::merge_embeddings`].
///
/// Validates the rank/shape contract, then assembles the merged sequence
/// by alternating slices of `text_embeds` (for text positions) with slices
/// of `image_embeds` (for image-span positions), `concatenate`-ing the
/// pieces along the sequence axis. No in-place mutation; lazy on the
/// returned `Array`.
fn default_merge_embeddings(
  text_embeds: &Array,
  image_embeds: &Array,
  image_spans: &[(usize, usize)],
) -> Result<Array> {
  // Rank guards — text embeds must be [1, T, D] (the LM's standard embed
  // output shape; `vlm_generate` always builds a single-batch prompt) and
  // image embeds must be [N, D] (post-`encode_image`/-projector). A wrong
  // rank is a programmer error in the per-model impl rather than user
  // data; surface as a recoverable typed error per the rest of
  // the crate's error discipline.
  let text_shape = text_embeds.shape();
  let text_rank = text_shape.len() as u32;
  if text_shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "merge_embeddings: text_embeds must be rank-3 [1, T, D]",
      text_rank,
      text_shape,
    )));
  }
  if text_shape[0] != 1 {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "merge_embeddings: text_embeds batch dim must be 1 (single-batch prompt)",
      1,
      text_shape[0],
    )));
  }
  let image_shape = image_embeds.shape();
  let image_rank = image_shape.len() as u32;
  if image_shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "merge_embeddings: image_embeds must be rank-2 [N, D]",
      image_rank,
      image_shape,
    )));
  }
  let t = text_shape[1];
  let d_text = text_shape[2];
  let n_total = image_shape[0];
  let d_image = image_shape[1];
  if d_text != d_image {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "merge_embeddings: hidden-dim D (text_embeds vs image_embeds)",
      d_text,
      d_image,
    )));
  }

  // Empty spans + empty image embeds is the no-image text path; the
  // caller should use `forward(tokens)` instead of `forward_embeddings`.
  // Reject loudly so a buggy caller can't silently produce a text-only
  // forward through the embed path (which would still work but masks the
  // upstream defect of building an `image_embeds` with zero rows).
  if image_spans.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "merge_embeddings: image_spans (use forward(tokens) for the text-only path)",
    )));
  }

  // Validate spans (start<end, in-bounds, non-overlapping, monotone) and
  // accumulate the total width — `Σ(end - start)` must match
  // `image_embeds.shape[0]`. Bound by a single forward walk over the
  // input slice; we deliberately do NOT sort here because the merge
  // **order matters** for the per-image slab assignment (spans[i]
  // consumes image_embeds rows [Σwidths[..i] .. Σwidths[..=i]]). The
  // upstream `assemble_multimodal_prompt` already emits spans in
  // monotone order; enforce that here as a contract rather than silently
  // re-sorting and assigning out of order.
  let mut total_width: usize = 0;
  let mut prev_end: usize = 0;
  for &(s, e) in image_spans.iter() {
    if s >= e {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "merge_embeddings: image span (start, end)",
        "start must be strictly less than end (empty spans not allowed)",
      )));
    }
    if e > t {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "merge_embeddings: image span end vs text seq_len T",
        "must be <= T",
        format!("end={e}, T={t}"),
      )));
    }
    if s < prev_end {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "merge_embeddings: image span order (s vs prev_end)",
        "spans must be monotone non-overlapping (assemble_multimodal_prompt emits them in order)",
      )));
    }
    // Checked add — a hostile span ((0, usize::MAX), …) is impossible
    // for any real prompt but we keep the discipline consistent with the
    // splice in `prompt.rs`.
    total_width = total_width.checked_add(e - s).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "merge_embeddings: cumulative span width (total_width + (e - s))",
        "usize",
        [
          ("total_width", total_width as u64),
          ("span_width", (e - s) as u64),
        ],
      ))
    })?;
    prev_end = e;
  }
  if total_width != n_total {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "merge_embeddings: sum of caller-supplied placeholder span widths vs image_embeds row \
         count N (expected = total_width, actual = n_total)",
      total_width,
      n_total,
    )));
  }

  // Assemble: text[:, 0..s1, :], image[0..w1, :].reshape([1, w1, D]),
  // text[:, e1..s2, :], image[w1..w1+w2, :].reshape([1, w2, D]), …,
  // text[:, eN.., :]. We slice with mlx `ops::indexing::slice` (lazy,
  // no host materialization) and reshape each `[w, D]` image slab into
  // `[1, w, D]` so the final `concatenate` along axis=1 is well-defined.
  //
  // Dimension widths are bounded by the LM's seq_len (well below
  // i32::MAX in any realistic prompt; the upstream
  // `assemble_multimodal_prompt` already enforces T <= i32::MAX before
  // this point) so the i32 cast for `slice` is safe.
  // Capacity = up to 2 pieces per span (leading text + image) + 1 trailing
  // text slice. `checked_mul`/`checked_add` so a pathological span count
  // can't overflow the capacity arithmetic before the recoverable
  // `try_with_capacity` (request-scaled in the image count).
  let pieces_cap = image_spans
    .len()
    .checked_mul(2)
    .and_then(|n| n.checked_add(1))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "merge_embeddings: piece-count capacity (image_spans.len() * 2 + 1)",
        "usize",
        [("image_spans.len()", image_spans.len() as u64)],
      ))
    })?;
  let mut pieces: Vec<Array> = try_with_capacity(pieces_cap)?;
  let d_i32 = d_text as i32;
  let t_i32 = t as i32;
  let mut text_cursor: usize = 0;
  let mut image_cursor: usize = 0;
  for &(s, e) in image_spans {
    // Leading text slice (may be empty when s == text_cursor).
    if s > text_cursor {
      let start = [0_i32, text_cursor as i32, 0_i32];
      let stop = [1_i32, s as i32, d_i32];
      let strides = [1_i32, 1_i32, 1_i32];
      pieces.push(ops::indexing::slice(text_embeds, &start, &stop, &strides)?);
    }
    // Image slab: image_embeds[image_cursor..image_cursor+w, :] reshaped
    // to [1, w, D].
    let width = e - s;
    let img_start = [image_cursor as i32, 0_i32];
    let img_stop = [(image_cursor + width) as i32, d_i32];
    let img_strides = [1_i32, 1_i32];
    let img_slab = ops::indexing::slice(image_embeds, &img_start, &img_stop, &img_strides)?;
    let img_slab = ops::shape::reshape(&img_slab, &(1_usize, width, d_text))?;
    pieces.push(img_slab);
    text_cursor = e;
    image_cursor += width;
  }
  // Trailing text slice (may be empty when text_cursor == t).
  if text_cursor < t {
    let start = [0_i32, text_cursor as i32, 0_i32];
    let stop = [1_i32, t_i32, d_i32];
    let strides = [1_i32, 1_i32, 1_i32];
    pieces.push(ops::indexing::slice(text_embeds, &start, &stop, &strides)?);
  }

  // Concatenate along the sequence axis. `pieces` is guaranteed non-empty
  // because `image_spans` is non-empty (guarded above). Recoverable
  // reservation for the `&Array` ref vec (request-scaled in piece count).
  let mut refs: Vec<&Array> = try_with_capacity(pieces.len())?;
  refs.extend(pieces.iter());
  ops::shape::concatenate(&refs, 1)
}

#[cfg(test)]
mod tests {
  //! Coverage for the per-model [`ImageProcessor`] seam: the
  //! [`ProcessedImage`] / [`NativeResolution`] accessors and the default
  //! [`FixedGridProcessor`] (which must reproduce the historical
  //! `vlm::image::preprocess` result, report `native = None`, and carry the
  //! constant fixed-grid count).

  use super::*;
  use crate::vlm::image::{ImageProcessorConfig, Layout};

  /// `ProcessedImage::new` + accessors round-trip: `pixels` projects to the
  /// stored tensor, `native` to the optional companions, `num_tokens` to the
  /// count. The fixed-grid case has `native == None`.
  #[test]
  fn processed_image_accessors_fixed_grid() {
    let pixels = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1_usize, 1, 3)).unwrap();
    let p = ProcessedImage::new(pixels, None, 7);
    assert_eq!(p.pixels().shape(), vec![1, 1, 3]);
    assert!(
      p.native().is_none(),
      "fixed-grid carries no native companions"
    );
    assert_eq!(p.num_tokens(), 7);
  }

  /// The native-resolution case: `native()` is `Some`, and the
  /// [`NativeResolution`] accessors project the `spatial_shape` + `patch_mask`
  /// arrays.
  #[test]
  fn processed_image_accessors_native_resolution() {
    let pixels = Array::from_slice::<f32>(&[0.0; 6], &(2_usize, 3)).unwrap();
    let spatial = Array::from_slice::<i32>(&[2, 3], &(2_usize,)).unwrap();
    let mask = Array::from_slice::<i32>(&[1, 1, 1, 0], &(4_usize,)).unwrap();
    let native = NativeResolution::new(spatial, mask);
    assert_eq!(native.spatial_shape().shape(), vec![2]);
    assert_eq!(native.patch_mask().shape(), vec![4]);
    let p = ProcessedImage::new(pixels, Some(native), 6);
    let got = p.native().expect("native present");
    assert_eq!(got.spatial_shape().shape(), vec![2]);
    assert_eq!(got.patch_mask().shape(), vec![4]);
    assert_eq!(p.num_tokens(), 6);
  }

  /// [`FixedGridProcessor`] accessors expose the config + count it was built
  /// with.
  #[test]
  fn fixed_grid_processor_accessors() {
    let cfg = ImageProcessorConfig::new().with_size((32, 48));
    let proc = FixedGridProcessor::new(cfg, 5);
    assert_eq!(proc.config().size(), (32, 48));
    assert_eq!(proc.num_tokens(), 5);
  }

  /// The default [`FixedGridProcessor`] reproduces the historical
  /// [`crate::vlm::image::preprocess`] result EXACTLY: `process(&img)` equals
  /// `preprocess(&img, cfg)` element-for-element, carries `native == None`,
  /// and reports the constant `num_tokens`. This is the contract that keeps a
  /// fixed-grid VLM's behavior unchanged through the new seam.
  #[test]
  fn fixed_grid_processor_matches_preprocess_and_is_fixed() {
    // A synthetic 6×4 RGB image decoded as a DynamicImage (the processor's
    // input). The pixel content is arbitrary but deterministic.
    let mut buf = ::image::RgbImage::new(6, 4);
    for y in 0..4u32 {
      for x in 0..6u32 {
        buf.put_pixel(x, y, ::image::Rgb([(x * 10) as u8, (y * 20) as u8, 90]));
      }
    }
    let img = ::image::DynamicImage::ImageRgb8(buf);
    // A non-default config (so the test exercises a real resize/normalize), in
    // the Hwc layout the historical path uses.
    let cfg = ImageProcessorConfig::new()
      .with_size((8, 8))
      .with_layout(Layout::Hwc);

    let proc = FixedGridProcessor::new(cfg, 4);
    let processed = proc.process(&img).expect("fixed-grid process succeeds");
    assert!(
      processed.native().is_none(),
      "fixed-grid processor attaches no native companions"
    );
    assert_eq!(
      processed.num_tokens(),
      4,
      "fixed-grid processor reports the constant count"
    );

    // The pixels MUST equal the direct preprocess result (byte-identical
    // fixed-grid behavior).
    let mut want = crate::vlm::image::preprocess(&img, &cfg).expect("direct preprocess");
    let mut got = processed.pixels().try_clone().expect("clone pixels");
    assert_eq!(got.shape(), want.shape(), "preprocessed shape matches");
    assert_eq!(
      got.to_vec::<f32>().unwrap(),
      want.to_vec::<f32>().unwrap(),
      "fixed-grid process reproduces preprocess element-for-element"
    );
  }
}
