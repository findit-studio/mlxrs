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
  error::{Error, Result, try_with_capacity},
  ops,
  vlm::image::ImageProcessorConfig,
};

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

  /// Encode a preprocessed image (post-[`crate::vlm::image::preprocess`])
  /// into vision-encoder embeddings, shape `[N, D]` where `N` is the
  /// image-token count this model expects per image (Qwen-VL is variable,
  /// LLaVA fixed-grid, etc.) and `D` is the LM's hidden dim.
  ///
  /// Per-model encoders (CLIP / SigLIP / Qwen-VL ViT / etc.) implement
  /// this. The input layout is the encoder's expected layout — most
  /// commonly `[1, H, W, 3]` after the per-model post-step from the
  /// channel-last `[H, W, 3]` that [`crate::vlm::image::preprocess`]
  /// returns. Per-model code can convert the layout inside its own
  /// `encode_image` (e.g. `transpose_axes(&[2, 0, 1])` + add batch) so
  /// the cross-model surface stays layout-agnostic, matching the same
  /// boundary [`crate::vlm::image`] documents in its `Channel layout`
  /// conventions section.
  ///
  /// Mirrors `vision_tower(pixel_values.transpose(0, 2, 3, 1), …)` +
  /// `multi_modal_projector(selected_image_feature)` in
  /// `mlx-vlm/mlx_vlm/models/pixtral/pixtral.py:60-77`.
  fn encode_image(&self, image: &Array) -> Result<Array>;

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
  /// - `Error::ShapeMismatch` if `text_embeds` is not rank-3 `[1, T, D]`,
  ///   `image_embeds` is not rank-2 `[N, D]`, or the `D` dims differ.
  /// - `Error::ShapeMismatch` if the sum of all span widths
  ///   `Σ(end - start)` differs from `image_embeds`' first axis `N`
  ///   (one image-feature per placeholder position is the splice
  ///   contract).
  /// - `Error::ShapeMismatch` if any span is out of bounds, overlaps the
  ///   previous span, or is empty (mirrors
  ///   [`crate::vlm::prompt::build_multimodal_mask`]'s validation).
  /// - `Error::ShapeMismatch` if `image_spans` is empty — there are no
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
  fn image_processor_config(&self) -> ImageProcessorConfig {
    ImageProcessorConfig::default()
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
  // data; surface as a recoverable `Err(ShapeMismatch)` per the rest of
  // the crate's error discipline.
  let text_shape = text_embeds.shape();
  if text_shape.len() != 3 || text_shape[0] != 1 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "merge_embeddings: text_embeds must be rank-3 [1, T, D], got {text_shape:?}"
      ),
    });
  }
  let image_shape = image_embeds.shape();
  if image_shape.len() != 2 {
    return Err(Error::ShapeMismatch {
      message: format!("merge_embeddings: image_embeds must be rank-2 [N, D], got {image_shape:?}"),
    });
  }
  let t = text_shape[1];
  let d_text = text_shape[2];
  let n_total = image_shape[0];
  let d_image = image_shape[1];
  if d_text != d_image {
    return Err(Error::ShapeMismatch {
      message: format!(
        "merge_embeddings: hidden-dim mismatch (text_embeds D={d_text}, image_embeds D={d_image})"
      ),
    });
  }

  // Empty spans + empty image embeds is the no-image text path; the
  // caller should use `forward(tokens)` instead of `forward_embeddings`.
  // Reject loudly so a buggy caller can't silently produce a text-only
  // forward through the embed path (which would still work but masks the
  // upstream defect of building an `image_embeds` with zero rows).
  if image_spans.is_empty() {
    return Err(Error::ShapeMismatch {
      message: "merge_embeddings: image_spans is empty; use forward(tokens) for the text-only path"
        .into(),
    });
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
  for (idx, &(s, e)) in image_spans.iter().enumerate() {
    if s >= e {
      return Err(Error::ShapeMismatch {
        message: format!("merge_embeddings: image span #{idx} ({s}, {e}) is empty (start>=end)"),
      });
    }
    if e > t {
      return Err(Error::ShapeMismatch {
        message: format!(
          "merge_embeddings: image span #{idx} ({s}, {e}) end exceeds text seq_len T={t}"
        ),
      });
    }
    if s < prev_end {
      return Err(Error::ShapeMismatch {
        message: format!(
          "merge_embeddings: image span #{idx} ({s}, {e}) overlaps or precedes previous \
           span ending at {prev_end} (spans must be monotone non-overlapping; \
           assemble_multimodal_prompt emits them in order)"
        ),
      });
    }
    // Checked add — a hostile span ((0, usize::MAX), …) is impossible
    // for any real prompt but we keep the discipline consistent with the
    // splice in `prompt.rs`.
    total_width = total_width
      .checked_add(e - s)
      .ok_or_else(|| Error::ShapeMismatch {
        message: format!(
          "merge_embeddings: cumulative span width overflows usize at span #{idx} ({s}, {e})"
        ),
      })?;
    prev_end = e;
  }
  if total_width != n_total {
    return Err(Error::ShapeMismatch {
      message: format!(
        "merge_embeddings: sum of span widths ({total_width}) must match image_embeds row \
         count N ({n_total})"
      ),
    });
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
    .ok_or_else(|| Error::ShapeMismatch {
      message: format!(
        "merge_embeddings: piece-count capacity (image_spans.len() * 2 + 1) overflows usize \
         (image_spans.len()={})",
        image_spans.len()
      ),
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
