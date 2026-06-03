//! The architecture-agnostic multimodal generation Iterator, ported from
//! [`mlx_vlm.generate.generate_step`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/generate.py)
//! (lines ~700–963: `get_input_embeddings(input_ids, pixel_values, …)` →
//! `_step(input_ids, inputs_embeds=…)` → `while True: _step(y[None])`)
//! and cross-checked against
//! [`mlx-swift-lm/Libraries/MLXVLM/VLMModel.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXVLM/VLMModel.swift)
//! (the `VLMModel: LanguageModel, LoRAModel` marker that swift uses to
//! drive the same `_step`-style decode loop).
//!
//! Everything is generic over the [`Model`] trait
//! ([`crate::vlm::model::Model`], which itself extends
//! [`crate::lm::model::Model`]): the loop calls `encode_image` per image,
//! then prefills in span-aware chunks — per chunk `embed_tokens` (chunk
//! text), `merge_embeddings` (chunk image slabs), and
//! `forward_embeddings_multimodal` (with the chunk's `cache_offset`) —
//! then `forward` per decode step.
//!
//! **Exact per-step order** (faithful to mlx-vlm `generate_step`, lines
//! 864–963):
//!
//! 1. *Validate the non-count prompt shape FIRST* (cheap, deterministic;
//!    precedes any image load). `validate_marker_run` checks marker
//!    presence (under [`crate::vlm::prompt::MarkerPolicy`]) and the
//!    image/marker pairing (the first contiguous marker run is exactly
//!    `image_count` long, no stray markers after it) and returns the
//!    splice base offset. A template-drift request (missing marker under
//!    `Required`, wrong run length) errors here BEFORE any image is
//!    loaded.
//! 2. *Preprocess each image via the model's [`crate::vlm::model::ImageProcessor`],
//!    then assemble + encode.* For each path in `images`,
//!    `vlm::image::load_image(path) → processor.process(&img)` yields a
//!    [`crate::vlm::model::ProcessedImage`] carrying the model's native
//!    pixel/patch tensor, any native-resolution companions, and the
//!    per-image image-token count. The processor is passed explicitly
//!    (mirroring mlx-vlm `generate(model, processor, …)`): a fixed-grid
//!    model's default
//!    [`FixedGridProcessor`](crate::vlm::model::FixedGridProcessor)
//!    reproduces the historical `preprocess` `[H, W, 3]` result and a
//!    constant grid count; a native-resolution model returns its own
//!    processor with a variable per-image count. Then
//!    `assemble_prompt_with_counts` replaces the marker run with each
//!    image's `num_tokens` placeholder ids in order (accumulating, NOT a
//!    uniform stride — a native-res model's per-image widths differ) and
//!    computes the matching per-image spans, and
//!    `model.encode_image(&processed_image)` lifts each image into
//!    `[N_i, D]` vision-encoder embeddings, validated to return EXACTLY
//!    `[image.num_tokens(), D]` rows (the cross-model splice contract). We
//!    deliberately build the spans inline (NOT
//!    [`crate::vlm::prompt::assemble_multimodal_prompt`], which also
//!    builds an O(T*T) bidirectional-within-image attention mask) because
//!    the trait's `forward_embeddings(embeds, cache)` signature has no way
//!    to thread that mask through to a per-model attention layer; instead,
//!    [`crate::vlm::model::Model::forward_embeddings_multimodal`] receives
//!    chunk-local `image_spans` + a `cache_offset` BY VALUE on every
//!    prefill chunk so a mask-requiring per-model override builds its
//!    `[chunk × (past + chunk)]` mask without any `&self` state. The
//!    per-image slabs are kept SEPARATE (one `[N_i, D]` Array per image),
//!    NOT pre-concatenated — the prefill gathers only the slabs whose span
//!    falls in the current chunk (step 4).
//! 3. *(no global embed/merge).* Text embedding + image merge are
//!    NOT done over the full sequence here — they happen INCREMENTALLY
//!    per chunk in step 4, so peak memory is bounded by `prefill_step_size
//!    · D` plus the (inherent) vision-feature slabs, not the full
//!    `T · D` merged sequence (which the image-token expansion inflates).
//! 4. *Offset-aware, span-aware chunked prefill.* The assembled
//!    prompt is walked in chunks of `cfg.lm.prefill_step_size` that NEVER
//!    split an image span (a boundary landing inside a span extends to
//!    the span end). For each chunk: embed only the chunk's tokens,
//!    merge only the image slabs whose span falls in the chunk
//!    (chunk-local coords), then
//!    `model.forward_embeddings_multimodal(chunk_merged, chunk_spans,
//!    cache_offset, &mut cache)` — `cache_offset` lets a mask-requiring
//!    override size its `[chunk × (past + chunk)]` mask correctly (see
//!    [`crate::vlm::prompt::build_multimodal_mask_with_past`]). The FINAL
//!    chunk's last-position logits drive the first sampler call (mlx-vlm's
//!    `_step(input_ids, inputs_embeds=inputs_embeds)` at `generate.py:903`).
//! 5. *Decode loop.* From token #2 onwards the loop is the standard
//!    text-only decode — `forward(last_token[1, 1], &mut cache)` → sample →
//!    yield — exactly the per-step order documented in
//!    [`crate::lm::generate`] (steps 1–6) and ported byte-identically
//!    here so a future shared `_step` factor can drop in without
//!    behavior change.
//!
//! **Why Option B (this file owns the loop) and not Option A
//! (`lm::generate` learns an embeds-prefill mode):** the in-line decode
//! loop is ~30 lines of mlx-vlm-faithful step composition (sampler,
//! processors, logsumexp, sample, eos check). Extending
//! [`crate::lm::generate::GenConfig`] with an `embeds_prefill: Option<Array>`
//! field would push a VLM-specific concept into the text-only surface
//! (`lm::generate` is consumed by every audio / pure-LM use too); keeping
//! the seam at the model-trait level (the two `forward*` methods are the
//! only LM-side primitives the VLM loop needs) preserves the cleaner
//! abstraction boundary. The duplication is bounded — both loops share
//! [`crate::lm::generate::make_sampler`] and
//! [`crate::lm::generate::make_logits_processors`], the exact normalization
//! formula (`logits - logsumexp(logits, keepdims=true)`), and the
//! [`crate::lm::generate::GenStep`] item shape, so any future refactor to
//! a shared `_step` is a pure code-movement, not a semantic change.
//!
//! **Error model:** matches [`crate::lm::generate`] — every fallible op
//! returns [`crate::Result`]; the returned iterator yields a step error
//! once as `Err` and then ends (fuses; no panic, no poison, never
//! re-entered). Preprocessing / encode / merge errors surface as the
//! `Err` of the returned `Result` from [`vlm_generate`] BEFORE the
//! iterator even runs — they happen synchronously at construction.

use std::{cell::RefCell, path::PathBuf};

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, EmptyInputPayload, Error, InvariantViolationPayload,
    LengthMismatchPayload, MissingFieldPayload, OutOfRangePayload, RankMismatchPayload, Result,
    try_extend_from_slice, try_with_capacity,
  },
  lm::{
    cache::KvCache,
    generate::{
      FinishReason, GenConfig, GenStep, LogitsProcessor, Sampler, make_logits_processors,
      make_sampler,
    },
  },
  ops,
  vlm::{
    image::load_image,
    model::{ImageProcessor, Model, ProcessedImage},
    prompt::MarkerPolicy,
  },
};

/// The assembled multimodal prompt + its per-image placeholder spans: the
/// `(assembled_tokens, image_spans)` pair [`assemble_prompt_with_counts`]
/// returns. `image_spans[i]` is the half-open `(start, end)` range image `i`'s
/// placeholder run occupies in `assembled_tokens`.
type AssembledPrompt = (Vec<u32>, Vec<(usize, usize)>);

/// Multimodal generation config — wraps [`GenConfig`] with the
/// image-specific knobs the multimodal pipeline needs.
///
/// Mirrors the surface mlx-vlm's `generate_step` exposes for its
/// `image_token_index` / `num_tokens_per_image` / `image_marker_id` knobs
/// (those live on the per-model config in the python reference and are
/// passed through the multimodal `_step`; mlxrs surfaces them explicitly
/// here because the per-model arch is user-owned per
/// `feedback_no_per_model_arch_porting`).
#[derive(Debug, Clone)]
pub struct VlmGenConfig {
  /// All text-generation knobs (sampler / processors / `max_tokens` /
  /// `prefill_step_size` / `eos` / `seed`). Reused 1:1 from the LM
  /// surface — the multimodal loop adds NO new sampler / processor
  /// concepts.
  lm: GenConfig,
  /// Token id of the image placeholder the splice emits (per-model — e.g.
  /// `<image>` or `<|image_pad|>`'s ID after tokenization). The merged
  /// embed sequence places `image_embeds` at every run of this id.
  image_token_id: u32,
  /// Token id the chat template emits where images go. Often the same as
  /// [`Self::image_token_id`] (single-token marker that BOTH delimits the
  /// splice site AND occupies the placeholder positions), but some
  /// models use distinct ids — e.g. `<|image|>` (marker) vs `<|image_pad|>`
  /// (placeholder). When `None`, defaults to [`Self::image_token_id`]
  /// (the common case).
  image_marker_id: Option<u32>,
  /// The fixed-grid per-image token count fed to the default
  /// [`Model::image_processor`] (a
  /// [`FixedGridProcessor`](crate::vlm::model::FixedGridProcessor)) when the
  /// caller does not pass an explicit processor — the constant grid size a
  /// CLIP/SigLIP/LLaVA encoder emits per image. A native-resolution model
  /// (LFM2.5-VL, Qwen3.5-VL) gets its variable per-image counts from its OWN
  /// [`ImageProcessor`] instead, and this field is unused for that path. The
  /// per-image counts the prompt is actually built from come from each
  /// image's [`ProcessedImage::num_tokens`](crate::vlm::model::ProcessedImage::num_tokens),
  /// and [`Model::encode_image`] must emit exactly that many rows per image.
  num_tokens_per_image: usize,
  /// Marker-vs-prepend policy. See
  /// [`crate::vlm::prompt::MarkerPolicy`].
  marker_policy: MarkerPolicy,
}

impl VlmGenConfig {
  /// Construct a [`VlmGenConfig`].
  ///
  /// `image_marker_id` defaults to `None` (marker == placeholder — the
  /// common single-token case). Use [`with_image_marker_id`] to set a
  /// distinct marker id when the chat template uses separate tokens for the
  /// splice site and the placeholder positions.
  ///
  /// [`with_image_marker_id`]: Self::with_image_marker_id
  pub fn new(
    lm: GenConfig,
    image_token_id: u32,
    num_tokens_per_image: usize,
    marker_policy: MarkerPolicy,
  ) -> Self {
    Self {
      lm,
      image_token_id,
      image_marker_id: None,
      num_tokens_per_image,
      marker_policy,
    }
  }

  /// Set a distinct `image_marker_id` when the chat template uses separate
  /// tokens for the splice site vs. the placeholder positions (e.g.
  /// `<|image|>` marker vs. `<|image_pad|>` placeholder).
  #[must_use]
  pub fn with_image_marker_id(mut self, v: Option<u32>) -> Self {
    self.image_marker_id = v;
    self
  }

  // ── accessors ──────────────────────────────────────────────────────────────

  /// All LM generation knobs.
  #[inline(always)]
  pub fn lm_ref(&self) -> &GenConfig {
    &self.lm
  }
  /// Mutable borrow of the LM generation knobs for in-place mutation.
  #[inline(always)]
  pub fn lm_mut(&mut self) -> &mut GenConfig {
    &mut self.lm
  }
  /// Image placeholder token id.
  #[inline(always)]
  pub fn image_token_id(&self) -> u32 {
    self.image_token_id
  }
  /// Optional distinct image marker token id (`None` = use
  /// [`image_token_id`](Self::image_token_id)).
  #[inline(always)]
  pub fn image_marker_id(&self) -> Option<u32> {
    self.image_marker_id
  }
  /// Number of image feature tokens per image.
  #[inline(always)]
  pub fn num_tokens_per_image(&self) -> usize {
    self.num_tokens_per_image
  }
  /// Marker vs. prepend policy.
  #[inline(always)]
  pub fn marker_policy(&self) -> MarkerPolicy {
    self.marker_policy
  }
}

/// End-to-end multimodal generation Iterator.
///
/// Loads each image, preprocesses it via the caller-supplied
/// `image_processor_config`, encodes via [`Model::encode_image`],
/// embeds the prompt via [`Model::embed_tokens`], splices the image
/// features into the text embeds via [`Model::merge_embeddings`], runs
/// the prefill via [`crate::lm::model::Model::forward_embeddings`], then
/// dispatches the per-token decode (same per-step order as
/// [`crate::lm::generate`]) via [`crate::lm::model::Model::forward`].
///
/// **The image processor is an explicit parameter, not derived from the
/// model.** mlx-vlm's `generate` / `generate_step` take the `processor`
/// separately from the `model` (`generate.py:1183`, `:966` — `generate(model,
/// processor, …)`). Pass the [`ImageProcessor`] the loaded model carries: a
/// fixed-grid model's default is
/// [`model.image_processor(num_tokens)`](Model::image_processor) (a
/// [`FixedGridProcessor`](crate::vlm::model::FixedGridProcessor) reproducing
/// the historical [`crate::vlm::image::preprocess`] path), while a
/// native-resolution model (LFM2.5-VL NaFlex, the future Qwen3.5-VL line)
/// returns its own model-specific processor whose
/// [`process`](ImageProcessor::process) builds the variable patch tensor + its
/// per-image token count. `vlm_generate` runs each raw loaded image through
/// `processor.process(&img)` to get a [`ProcessedImage`], uses each image's
/// [`num_tokens`](ProcessedImage::num_tokens) to size that image's placeholder
/// run in the prompt, and feeds the [`ProcessedImage`] to
/// [`Model::encode_image`]. Passing the processor explicitly mirrors the
/// loaded-processor flow and keeps a fixed-grid caller able to specify its
/// constant count via the default processor.
///
/// Returns `impl Iterator<Item = Result<GenStep>> + 'a` — `impl` keeps the
/// concrete iterator type unnamed (matching the LM-side
/// [`crate::lm::generate::stream_generate`] shape so a future text-only
/// fallback can drop in without an API break). Borrows `&'a M` plus owns
/// the cache, so no aliasing of the model across the borrow.
///
/// `M: Model + ?Sized` — the loop only ever touches the model behind the
/// `&'a M` borrow (`model.embed_tokens(...)`, `model.encode_image(...)`,
/// `model.forward*(...)`), never by value and never via a
/// `Sized`-requiring associated item. `M` may therefore be an unsized
/// trait object: a deref-coerced `Box<dyn VlmModel>` — the exact handle
/// the load factory returns ([`crate::vlm::load::LoadedVlmContext::model`])
/// — drives generation directly without a forwarding shim. The
/// zero-image passthrough hands the same `&'a M` to
/// [`crate::lm::generate::generate_step`], which is likewise
/// `?Sized`-generic (and accepts it because `VlmModel: Model`).
///
/// **Zero-image passthrough**: when `images.is_empty()`, the function
/// dispatches directly to [`crate::lm::generate::generate_step`] (the
/// merge/encode steps are skipped entirely) — the iterator's per-step
/// behavior is byte-identical to the LM-only path. This makes
/// `vlm_generate` a strict superset, safe to use from a higher-level
/// dispatch that doesn't know whether the prompt has images.
///
/// # Errors
///
/// Surface (as `Err` of the returned `Result` — synchronous):
///
/// - `Error::Backend` on image load / preprocess / encode failures (the
///   path's I/O / decode error propagates).
/// - `Error::RankMismatch` / `Error::LengthMismatch` / `Error::EmptyInput`
///   on a span/embed/dim contract violation in [`Model::merge_embeddings`].
/// - `Error::RankMismatch` (wrong ndim) or `Error::LengthMismatch` (wrong
///   row count) on a per-image encoder output that is not
///   `[image.num_tokens(), D]` — every image MUST emit exactly the feature
///   rows its [`ProcessedImage::num_tokens`] reports, enforced per-image
///   BEFORE the slabs are concatenated (the cross-model splice contract; a
///   variable-per-image model reports each image's own count from its
///   processor and emits that many rows).
///
/// Surface (as the iterator's first `Err`, exactly like
/// [`crate::lm::generate::generate_step`]):
///
/// - sampler / logits-processor construction failure
/// - any per-step forward / sample failure
pub fn vlm_generate<'a, M: Model + ?Sized>(
  model: &'a M,
  processor: &dyn ImageProcessor,
  text_tokens: &[u32],
  images: &[PathBuf],
  cache: Vec<Box<dyn KvCache>>,
  cfg: VlmGenConfig,
) -> Result<impl Iterator<Item = Result<GenStep>> + 'a> {
  // ── EAGER `cfg.lm.validate()` ────────────────────────────────────────
  // #136 — mirror single-seq [`crate::lm::generate::generate_step`]'s
  // eager validation gate ACROSS BOTH VLM branches. The zero-image branch
  // delegates to `generate_step` (which validates internally), but the
  // multimodal branch builds its own sampler / logits-processors below
  // and would otherwise burn the entire vision pipeline (load /
  // preprocess / encode_image, possibly multi-image) before the first
  // decode step surfaced an invalid bound — or silently NaN-poisoned
  // logits via a NaN `logit_bias` / `*_penalty` that the per-primitive
  // path does not finite-check. Validating HERE — synchronously, before
  // the `max_tokens == 0` / zero-image / multimodal split — also gives
  // the `max_tokens == 0` short-circuit identical "invalid config is
  // always Err, never silent" semantics it has on the LM side, so
  // `vlm_generate` is uniformly fail-fast on a bad cfg regardless of
  // image count or zero-budget.
  cfg.lm.validate()?;

  // ── max_tokens == 0 SHORT-CIRCUIT ────────────────────────────────────
  // Mirror the LM-side contract: `lm::generate`'s iterator checks
  // `produced >= max_tokens` BEFORE running prefill (generate.rs:598),
  // so a `max_tokens == 0` request yields nothing and runs no model
  // call. The VLM multimodal path does its vision work (load /
  // preprocess / encode_image / merge) EAGERLY at construction, so
  // without this guard a zero-output request would still trigger image
  // I/O + vision compute + potential decode/OOM errors.
  // Short-circuit to an empty iterator BEFORE any
  // vision work — and before the zero-image split — so both paths are
  // identically free of work when nothing will be produced.
  if cfg.lm.max_tokens == 0 {
    return Ok(Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Result<GenStep>> + 'a>);
  }

  // ── ZERO-IMAGE PASSTHROUGH ───────────────────────────────────────────
  // Faithful to `mlx-vlm`'s `get_input_embeddings`'s `if pixel_values is
  // None: return InputEmbeddingsFeatures(inputs_embeds=embed_tokens(input_ids))`
  // branch (e.g. `pixtral.py:48-51`): no images ⇒ the LM-only path. We
  // skip the encode / merge / `forward_embeddings` stages entirely and
  // hand off to `lm::generate::generate_step`. The returned iterator is
  // strictly the LM-side iterator (boxed into the `impl Iterator` via
  // `Box<dyn>` so both branches return the same opaque type).
  //
  // The marker isn't relevant when image_count == 0 (`insert_image_tokens`
  // already short-circuits to a copy in that case) — but we DELIBERATELY
  // do NOT touch the text tokens here. The chat template may have emitted
  // an image marker that the caller intends to remain literal text in a
  // no-image run; `lm::generate::generate_step` consumes raw token ids
  // exactly as supplied.
  //
  // **`collect_logprobs` override**: the multimodal-path decode loop in
  // this file ALWAYS emits `Some(logprobs)` (the comment at the
  // post-sampler squeeze documents the unconditional yield), so the
  // cross-crate VLM-surface contract is "every `GenStep.logprobs` is
  // `Some`". The zero-image branch delegates to `lm::generate_step`,
  // which honors `cfg.lm.collect_logprobs` — and that field's `Default`
  // is `false`, so a default-cfg zero-image VLM run would otherwise
  // silently flip to `None`-logprobs and break the documented surface
  // (contract drift between the two branches). Force
  // the LM-level opt-in here so the zero-image branch yields the same
  // `Some(logprobs)` shape the multimodal branch does — the caller still
  // controls `collect_logprobs` end-to-end via [`VlmGenConfig::lm`], but
  // the zero-image fallback can never undershoot the documented VLM
  // contract.
  if images.is_empty() {
    // `cfg` is consumed by this branch (the multimodal path below
    // re-borrows `cfg.*` fields directly), so move `cfg.lm` out instead
    // of cloning — `clippy::redundant-clone` flags the avoidable extra
    // owned `GenConfig` heap-walk of the eos / xtc_special_tokens /
    // logit_bias vectors.
    let mut lm = cfg.lm;
    lm.collect_logprobs = true;
    let iter = crate::lm::generate::generate_step(model, text_tokens, cache, lm);
    // Box so the two branches share an opaque return type. Allocation
    // here is one-shot at construction (not per step), and the
    // alternative — duplicating the iterator-state struct across both
    // paths — would dwarf the cost in code volume.
    return Ok(Box::new(iter) as Box<dyn Iterator<Item = Result<GenStep>> + 'a>);
  }

  // ── MULTIMODAL PATH ──────────────────────────────────────────────────
  //
  // ORDER: deterministic non-count prompt-shape validation FIRST (marker
  // presence under MarkerPolicy::Required, image/marker pairing), THEN the
  // per-image processor run (whose output carries each image's variable
  // `num_tokens`), THEN the count-dependent prompt assembly + spans, THEN
  // the per-image encode. The non-count validation runs before any image
  // is even loaded so a template-drift request (missing marker, wrong
  // marker-run length) errors cheaply; the count is per-image and comes
  // from the processor (native-resolution models produce a variable count
  // per image), so the placeholder-run sizing necessarily follows
  // processing.
  let marker_id = cfg.image_marker_id.unwrap_or(cfg.image_token_id);
  let base = validate_marker_run(text_tokens, images.len(), marker_id, cfg.marker_policy)?;

  // Per-image preprocess via the model's [`ImageProcessor`]. Each image is
  // loaded + decoded, then run through `processor.process(&img)` to its
  // [`ProcessedImage`] — the model's native pixel/patch tensor, optional
  // native-resolution companions, and the per-image image-token count. A
  // fixed-grid model's default processor reproduces the historical
  // `vlm::image::preprocess` result and reports the constant grid count; a
  // native-resolution model returns its own variable count per image. The
  // processor is passed explicitly (mlx-vlm's `generate(model, processor,
  // …)`), so the loaded preprocessor config flows in rather than being
  // silently re-derived from the model.
  let mut processed: Vec<ProcessedImage> = try_with_capacity(images.len())?;
  for path in images.iter() {
    let img = load_image(path)?;
    processed.push(processor.process(&img)?);
  }

  // Per-image token counts → the count-dependent prompt assembly + spans.
  // The marker run (one marker per image, validated above) is replaced by
  // each image's `num_tokens` placeholders in order; spans are the
  // half-open ranges those placeholder runs occupy. A native-resolution
  // model's per-image counts differ, so the spans are NOT a uniform stride
  // — they accumulate each image's own width.
  let mut counts: Vec<usize> = try_with_capacity(processed.len())?;
  for p in processed.iter() {
    counts.push(p.num_tokens());
  }
  let (assembled_tokens, image_spans) =
    assemble_prompt_with_counts(text_tokens, base, marker_id, cfg.image_token_id, &counts)?;

  // Now the per-image encode. We deliberately encode one image at a time
  // and concatenate the resulting `[N_i, D]` slabs along axis 0: some
  // models' `encode_image` accepts a batch and some don't (the per-model
  // encoder owns the input layout / batch contract per
  // [`Model::encode_image`]'s doc), so the cross-model surface stays at
  // "one image at a time" — the simplest contract that every encoder
  // satisfies.
  //
  // PER-IMAGE SHAPE VALIDATION: every `encode_image` MUST return exactly
  // `[image.num_tokens(), D]`. The cross-model splice contract is "one
  // image emits exactly its reported `num_tokens` features"; a
  // variable-per-image model reports each image's own count from its
  // processor and emits that many rows. Without this per-slab check, a
  // model returning e.g. `[2, D]` for image 1 and `[4, D]` for image 2
  // whose reported counts are 3 / 3 would pass the merge layer's "total
  // widths == total rows" check (both = 6) but cause silent
  // marker-to-image misalignment (the first prompt span would consume 2
  // rows from image 1 plus 1 row from image 2). Surface as
  // `Error::LengthMismatch` instead.
  let mut image_slabs: Vec<Array> = try_with_capacity(processed.len())?;
  for (p, &want) in processed.iter().zip(counts.iter()) {
    let encoded = model.encode_image(p)?;
    let enc_shape = encoded.shape();
    let (rows, _d) = match enc_shape.as_slice() {
      [n, d] => (*n, *d),
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "vlm_generate: encode_image must return rank-2 [N, D]",
          enc_shape.len() as u32,
          enc_shape.clone(),
        )));
      }
    };
    if rows != want {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "vlm_generate: encode_image feature rows vs ProcessedImage::num_tokens (cross-model \
           splice contract requires exactly the processor-reported per-image token count per \
           image)",
        want,
        rows,
      )));
    }
    image_slabs.push(encoded);
  }
  // Offset-aware chunked multimodal prefill: we DELIBERATELY do
  // NOT concat the slabs / embed the text / merge the full sequence here.
  // The embed + merge happen INCREMENTALLY per chunk inside
  // `prefill_step`, so peak memory is bounded by `prefill_step_size · D`
  // plus the (inherent) vision-feature slabs — not by the full
  // `T · D` merged sequence (which the image-token expansion inflates
  // substantially). The per-image `image_slabs` are kept as-is (one
  // `[N_i, D]` Array per image) and gathered per-chunk via the
  // span→image-index correspondence (image `i`'s features occupy
  // `image_spans[i]`). `encode_image` itself runs once per image above
  // (vision encoding cannot be chunked), so the slabs' Σ N_i · D is the
  // inherent floor; everything else is now chunk-bounded.

  // Build sampler + processors up front (mlx-vlm `generate_step` does
  // this the same way — see `generate.py:786-796`). Defer any error to
  // the iterator's first `next()` exactly like
  // `lm::generate::generate_step` does, so the public surface stays a
  // pure `Iterator`. Construction failures are wrapped in the deferred
  // `pending_err` slot of `VlmDecode`.
  let built = (|| -> Result<(Sampler, Vec<LogitsProcessor>)> {
    let sampler = make_sampler(
      cfg.lm.temp,
      cfg.lm.top_p,
      cfg.lm.min_p,
      cfg.lm.min_tokens_to_keep,
      cfg.lm.top_k,
      cfg.lm.xtc_probability,
      cfg.lm.xtc_threshold,
      &cfg.lm.xtc_special_tokens,
      cfg.lm.seed,
    )?;
    let processors = make_logits_processors(
      &cfg.lm.logit_bias,
      cfg.lm.repetition_penalty,
      cfg.lm.repetition_context_size,
      cfg.lm.presence_penalty,
      cfg.lm.presence_context_size,
      cfg.lm.frequency_penalty,
      cfg.lm.frequency_context_size,
    )?;
    Ok((sampler, processors))
  })();

  // VlmDecode owns the merged embeds (consumed once at prefill), the
  // cache, sampler, processors, history, and the per-step state. The
  // per-iteration `next()` is shaped exactly like
  // `lm::generate::Generator::next` (1:1 byte-equivalent step order
  // sans the prefill chunking, which is replaced by the one-shot
  // embed-based prefill).
  match built {
    Ok((sampler, processors)) => Ok(Box::new(VlmDecode {
      model,
      cache: RefCell::new(cache),
      sampler: RefCell::new(sampler),
      processors,
      history: RefCell::new(Vec::new()),
      eos: cfg.lm.eos,
      max_tokens: cfg.lm.max_tokens,
      produced: 0,
      prefill_step_size: cfg.lm.prefill_step_size.max(1),
      last: None,
      prefilled: false,
      image_slabs: Some(image_slabs),
      // Stash the per-image spans so the prefill `_step` can thread
      // them into [`Model::forward_embeddings_multimodal`] WITHOUT
      // touching `&self` — every iterator owns its own spans, so two
      // iterators constructed against the same model with different
      // spans never share state and a model's
      // `forward_embeddings_multimodal` override receives the correct
      // per-request spans (avoid the cross-request hazard of model-side
      // mask state).
      image_spans: Some(image_spans),
      // Stash the assembled prompt ids for the prefill `_step`'s
      // processor-history seeding — mirrors mlx-vlm `generate.py:845`
      // (`tokens = mx.concat([tokens, y.flatten()])` where `y` is
      // `input_ids` for the prefill `_step` and the subsequent `y[None]`
      // for each decode step). The history is consumed once on the
      // first poll (drained via `take` so the `Vec<u32>` storage is
      // released after the single use).
      prompt_history: Some(assembled_tokens),
      pending_err: None,
      done: false,
    }) as Box<dyn Iterator<Item = Result<GenStep>> + 'a>),
    Err(e) => Ok(Box::new(VlmDecode {
      model,
      cache: RefCell::new(cache),
      // Cheapest no-allocation placeholder ([`Sampler::Argmax`]); never
      // invoked because `pending_err` short-circuits the first `next()`.
      sampler: RefCell::new(Sampler::Argmax),
      processors: Vec::new(),
      history: RefCell::new(Vec::new()),
      eos: Vec::new(),
      max_tokens: cfg.lm.max_tokens,
      produced: 0,
      prefill_step_size: 1,
      last: None,
      prefilled: true, // skip the prefill — pending_err ends iteration first
      image_slabs: None,
      image_spans: None,
      prompt_history: None,
      pending_err: Some(e),
      done: false,
    }) as Box<dyn Iterator<Item = Result<GenStep>> + 'a>),
  }
}

/// The architecture-agnostic VLM decode iterator. Owns the cache, the
/// sampler, the logits processors, and the merged-embed prefill payload;
/// borrows `&'a M`. Yields one [`GenStep`] per call until eos or
/// `max_tokens`.
///
/// `RefCell`'d cache / sampler / history so the iterator's `next()` can
/// take `&mut self` while the internal step helper takes `&self` — this
/// matches the [`crate::lm::generate::Generator`] interior-mutability
/// pattern (its `step` is `&mut self`; we use `&self + RefCell` because
/// the prefill / decode branches share the same step body and one borrow
/// scope keeps the code linear).
struct VlmDecode<'a, M: Model + ?Sized> {
  model: &'a M,
  cache: RefCell<Vec<Box<dyn KvCache>>>,
  sampler: RefCell<Sampler>,
  processors: Vec<LogitsProcessor>,
  history: RefCell<Vec<u32>>,
  eos: Vec<u32>,
  max_tokens: usize,
  produced: usize,
  /// Prefill chunk size — the merged-embed prefill is processed in
  /// `[1, k, D]` slices of this width along axis 1 to bound peak
  /// memory (mirrors mlx-vlm `generate.py:881-901` chunked prefill).
  /// Must be `>= 1`; the iterator already clamps `0` to `1` at
  /// construction (matching `lm::generate::Generator::prefill_step_size`'s
  /// `cfg.prefill_step_size.max(1)` discipline).
  prefill_step_size: usize,
  /// Most-recently sampled token (mlx-vlm's `y` fed into the next
  /// `_step`); `None` before the first decode step.
  last: Option<u32>,
  /// `true` once the embed-based prefill has run (which yields the FIRST
  /// token via the prefill's last-position logits — exactly mlx-vlm
  /// `_step(input_ids, inputs_embeds=inputs_embeds)`).
  prefilled: bool,
  /// Per-image vision-feature slabs (`[N_i, D]` each, output of
  /// `encode_image`) consumed once at prefill; `take()`n so the storage
  /// is released after the single use. Kept per-image (NOT pre-concatenated)
  /// so [`Self::prefill_step`] can gather only the slabs whose span falls
  /// in the current chunk and merge them incrementally.
  image_slabs: Option<Vec<Array>>,
  /// Per-image `(start, end)` ABSOLUTE spans (in the assembled prompt's
  /// position axis) the prefill threads — shifted to chunk-local
  /// coordinates per chunk — into
  /// [`crate::vlm::model::Model::forward_embeddings_multimodal`] so a
  /// mask-requiring model can recompute its own multimodal mask from
  /// this iterator's spans (NOT from any per-model `&self` state — that
  /// would mix masks across concurrent / interleaved iterators).
  /// `image_spans[i]` is image `i`'s span and corresponds to
  /// `image_slabs[i]`. Owned by the iterator; consumed once at prefill.
  image_spans: Option<Vec<(usize, usize)>>,
  /// The assembled prompt token ids — fed into the prefill `_step`'s
  /// processor history (mlx-vlm `generate.py:845` accumulates
  /// `y.flatten()` where `y` is the prefill `input_ids`; we mirror that
  /// exactly so the FIRST multimodal token is subject to configured
  /// logits processors with the prompt in history, just like the
  /// LM-only loop and mlx-vlm itself). `take()`n on the first poll so
  /// the storage is freed once the prefill `_step` runs.
  prompt_history: Option<Vec<u32>>,
  /// A deferred sampler / processor construction error, yielded as the
  /// iterator's first (and only) `Err` before any step runs.
  pending_err: Option<Error>,
  /// Fused: set after a yielded `Err` or a finish so the iterator never
  /// re-enters mlx-c / re-runs the model.
  done: bool,
}

impl<M: Model + ?Sized> VlmDecode<'_, M> {
  /// Sample one token from `logits` (`[1, V]`) using the sampler and the
  /// configured logits processors. Mirrors the post-forward portion of
  /// [`crate::lm::generate::Generator::step`] (steps 3–6 in that file's
  /// doc), kept in sync with that loop's exact normalization /
  /// processor-history accumulation order.
  ///
  /// `step_inputs` are the token ids that drove the just-completed
  /// forward — appended to history when processors are present and the
  /// input is non-empty (faithful to the
  /// `if logits_processors and len(input_tokens) > 0` guard at
  /// `mlx-vlm/mlx_vlm/generate.py:844-848` and `mlx_lm/generate.py:409-414`).
  fn sample_from_logits(&self, logits: &Array, step_inputs: &[u32]) -> Result<GenStep> {
    // 1. `logits[:, -1, :]` — keep only the final sequence position,
    //    drop that axis ⇒ `[1, V]`. Same routine as
    //    `lm::generate::last_position` (kept private there; replicated
    //    here as a guard-pinned helper to avoid widening the LM public
    //    surface for this single shared concern).
    let logits = last_position(logits)?;
    // 2. logits processors over RAW logits, history-accumulated when
    //    present + input non-empty.
    let mut logits = logits;
    if !self.processors.is_empty() && !step_inputs.is_empty() {
      let mut history = self.history.borrow_mut();
      try_extend_from_slice(&mut history, step_inputs)?;
      for p in &self.processors {
        logits = p.apply(&history, &logits)?;
      }
    }
    // 3. `logprobs = logits - mx.logsumexp(logits, keepdims=True)` —
    //    exact mlx-vlm / mlx-lm normalization (all-axes logsumexp,
    //    `[1, 1]`, broadcast).
    let lse = ops::reduction::logsumexp(&logits, true)?;
    let logprobs = ops::arithmetic::subtract(&logits, &lse)?;
    // 4. sampler — argmax (temp=0) or the make_sampler chain.
    let mut sampled = self.sampler.borrow_mut().sample(&logprobs)?;
    // 5. token boundary — the ONLY materialization
    //    (`y.item()` in mlx-vlm / mlx-lm).
    let token: u32 = sampled.item::<u32>()?;
    // mlx-vlm/mlx-lm `logprobs.squeeze(0)` ⇒ a `[V]` vector. Kept lazy.
    // L3 `GenStep.logprobs` is `Option<Array>`: VLM has not adopted the
    // [`crate::lm::generate::GenConfig::collect_logprobs`] opt-in yet, so
    // we always emit `Some` to preserve the prior unconditional yield
    // (callers' field access shape changes from `step.logprobs` to
    // `step.logprobs.unwrap()` / `.as_ref()` — the same source-break the
    // LM crate accepts).
    let logprobs = ops::shape::squeeze_axes(&logprobs, &[0])?;
    // #114: provisional `step_index`/`finish_reason` — the iterator
    // overrides `finish_reason` to `Some("stop")` on the EOS-token step
    // (mirrors `lm::generate::Generator::step` + its `Iterator::next`).
    Ok(GenStep {
      token,
      logprobs: Some(logprobs),
      step_index: self.produced,
      finish_reason: None,
    })
  }

  /// The embed-based prefill (offset-aware chunked design): walk
  /// the assembled prompt in span-aware chunks, embedding then merging
  /// then forwarding ONE chunk at a time. Populate the cache to position
  /// T, then sample the FIRST token from the FINAL chunk's last-position
  /// logits. Mirrors `_step(input_ids, inputs_embeds=inputs_embeds)` at
  /// `mlx-vlm/mlx_vlm/generate.py:903` extended with the chunked-prefill
  /// loop at `mlx-vlm/mlx_vlm/generate.py:881-901`.
  ///
  /// **Per-chunk peak memory** is `max(prefill_step_size, W_max) · D`
  /// for the chunk's text-embed + merged buffer, where `W_max` is the
  /// widest single image span (`= num_tokens_per_image` for the
  /// fixed-grid case): invariant 1 keeps each image span whole, so a
  /// span wider than `prefill_step_size` forces a chunk that wide
  /// (the bound is NOT `prefill_step_size · D` alone).
  /// This is still bounded by a model constant, never the full expanded
  /// `T`; and `W_max · D <= Σ N_i · D`, the vision-feature slab floor
  /// that is resident regardless (vision encoding can't be chunked). So
  /// the total is `Σ N_i · D` (inherent image features) plus
  /// `max(prefill_step_size, W_max) · D` (one chunk) — independent of the
  /// text length and of the image COUNT beyond the per-image width.
  ///
  /// **Two invariants make chunking correct for mask-requiring VLMs**
  /// (the structural fix):
  ///
  /// 1. **Never split an image span.** When the natural
  ///    `cursor + prefill_step_size` boundary lands strictly inside a
  ///    span, the chunk extends to that span's end. Each span's
  ///    bidirectional-within-image attention therefore stays in one
  ///    forward; cross-span attention is causal (a later image's query
  ///    attends to an earlier image whose keys are already cached).
  /// 2. **Pass `cache_offset` + chunk-local spans.** Each chunk's
  ///    `forward_embeddings_multimodal` receives the cache offset
  ///    (`cursor` = tokens already cached) plus spans shifted to
  ///    chunk-local `(s - cursor, e - cursor)` coordinates, so a
  ///    mask-building override sizes the attention mask
  ///    `[chunk_len × (cache_offset + chunk_len)]` over past + current
  ///    keys (see [`crate::vlm::prompt::build_multimodal_mask_with_past`]).
  ///
  /// **Incremental embed/merge:** for each chunk only the chunk's text
  /// tokens are embedded and only the image slabs whose span falls in
  /// the chunk are merged — the full merged sequence is never
  /// materialized. `image_slabs[i]` is image `i`'s `[N_i, D]` features
  /// and corresponds to `image_spans[i]`.
  ///
  /// `prompt_tokens` is the assembled prompt id sequence — fed as the
  /// `step_inputs` for the prefill `_step`'s processor-history
  /// accumulation (mlx-vlm `generate.py:845`).
  fn prefill_step(
    &self,
    prompt_tokens: &[u32],
    image_spans: &[(usize, usize)],
    image_slabs: &[Array],
  ) -> Result<GenStep> {
    let t = prompt_tokens.len();
    if t == 0 {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "vlm_generate: assembled prompt (T=0); prefill cannot produce logits",
      )));
    }
    // The cache may already hold tokens (a restored / pre-populated
    // prompt cache, or a model that pre-seeds the cache). Read the
    // starting offset so each chunk's `cache_offset` is ABSOLUTE
    // (`initial_offset + cursor`), not just the in-prefill `cursor` —
    // otherwise a mask-requiring override would size its mask too short
    // against a non-empty cache.
    //
    // All layers advance in lockstep during generation and a faithfully
    // saved/restored prompt cache loads every layer from the same state,
    // so they MUST share one offset — but the cache API does not enforce
    // that, and a corrupt/hand-built cache could differ per layer. The
    // override receives a single scalar `cache_offset`, so a per-layer
    // mismatch would silently size the mask wrong for some layers.
    // Validate equality up front and fail closed.
    let initial_offset = {
      let cache = self.cache.borrow();
      let mut iter = cache.iter();
      match iter.next() {
        None => 0, // no layers (degenerate); treated as offset 0
        Some(first) => {
          let off = first.offset();
          for layer in iter {
            if layer.offset() != off {
              return Err(Error::LengthMismatch(LengthMismatchPayload::new(
                "vlm_generate: KV cache layer offsets must agree (layer 0 vs layer i; \
                   chunked-multimodal prefill needs one consistent cache offset to size per-chunk \
                   attention masks — a faithfully restored prompt cache has all layers at the \
                   same offset)",
                off,
                layer.offset(),
              )));
            }
          }
          off
        }
      }
    };
    let step = self.prefill_step_size.max(1);
    let mut cursor: usize = 0;
    let mut last_logits: Option<Array> = None;
    while cursor < t {
      // Invariant 1: never split a span. If the natural boundary lands
      // strictly inside a span `(s, e)` (s < end < e), extend `end` to
      // `e`. image_spans is small (one per image), so the scan is cheap.
      let mut end = (cursor + step).min(t);
      for &(s, e) in image_spans {
        if s < end && end < e {
          end = end.max(e);
        }
      }
      let end = end.min(t);
      let chunk_len = end - cursor;

      // Embed ONLY this chunk's tokens — `[1, chunk_len, D]`. (Image
      // placeholder ids embed to throwaway vectors that the per-chunk
      // merge overwrites at the chunk-local span positions.)
      let chunk_window = {
        let mut row: Vec<i32> = try_with_capacity(chunk_len)?;
        row.extend(prompt_tokens[cursor..end].iter().map(|&x| x as i32));
        Array::from_slice::<i32>(&row, &(1_usize, chunk_len))?
      };
      let chunk_text_embeds = self.model.embed_tokens(&chunk_window)?;

      // Invariant 2: chunk-local spans (and the matching slabs). Image
      // `i` is in this chunk iff `image_spans[i] ⊆ [cursor, end)`
      // (guaranteed whole by invariant 1). Collect both in index order.
      // Pre-reserve to the image count (upper bound on spans in one chunk).
      let mut chunk_spans: Vec<(usize, usize)> = try_with_capacity(image_spans.len())?;
      let mut chunk_slab_refs: Vec<&Array> = try_with_capacity(image_spans.len())?;
      for (i, &(s, e)) in image_spans.iter().enumerate() {
        if cursor <= s && e <= end {
          chunk_spans.push((s - cursor, e - cursor));
          chunk_slab_refs.push(&image_slabs[i]);
        }
      }

      // Merge ONLY when this chunk carries image features; a pure-text
      // chunk's text embeds ARE its merged embeds (the default
      // `merge_embeddings` rejects empty spans by contract).
      let chunk_merged = if chunk_spans.is_empty() {
        chunk_text_embeds
      } else {
        let chunk_image_embeds = if chunk_slab_refs.len() == 1 {
          chunk_slab_refs[0].try_clone()?
        } else {
          ops::shape::concatenate(&chunk_slab_refs, 0)?
        };
        self
          .model
          .merge_embeddings(&chunk_text_embeds, &chunk_image_embeds, &chunk_spans)?
      };

      // Forward with the ABSOLUTE cache offset (initial + cursor) so a
      // mask-requiring override sizes its mask over the true past +
      // current keys, and chunk-local spans so its coordinates line up
      // with `chunk_merged`. `checked_add` guards a near-`usize::MAX`
      // restored offset (recoverable error, never a debug-panic /
      // release-wrap before the mask builder could reject it).
      let chunk_offset = initial_offset.checked_add(cursor).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
          "vlm_generate: initial cache offset + chunk cursor",
          "usize",
          [
            ("initial_offset", initial_offset as u64),
            ("cursor", cursor as u64),
          ],
        ))
      })?;
      let logits = self.model.forward_embeddings_multimodal(
        &chunk_merged,
        &chunk_spans,
        chunk_offset,
        &mut self.cache.borrow_mut(),
      )?;
      // Retain only the final chunk's logits — earlier chunks just fill
      // the cache. `Option::replace` drops the prior chunk's logits
      // immediately so peak host/GPU memory stays at one chunk's worth.
      last_logits = Some(logits);
      cursor = end;
    }
    // `t > 0` is guarded above so at least one chunk ran.
    let logits = last_logits.expect("at least one prefill chunk ran (t > 0 guarded above)");
    self.sample_from_logits(&logits, prompt_tokens)
  }

  /// One decode step — `forward([last_token], cache)` + sample. The
  /// per-step `tokens` arg appended to history is the single decode
  /// token, exactly mirroring `_step(y[None])` at
  /// `mlx-vlm/mlx_vlm/generate.py:949` (and the analogous `_step` at
  /// `mlx_lm/generate.py:396-422`).
  fn decode_step(&self, last_token: u32) -> Result<GenStep> {
    let tokens = Array::from_slice::<i32>(&[last_token as i32], &(1_usize, 1_usize))?;
    let logits = self.model.forward(&tokens, &mut self.cache.borrow_mut())?;
    self.sample_from_logits(&logits, &[last_token])
  }
}

impl<M: Model + ?Sized> Iterator for VlmDecode<'_, M> {
  type Item = Result<GenStep>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.done {
      return None;
    }
    if let Some(e) = self.pending_err.take() {
      self.done = true;
      return Some(Err(e));
    }
    if self.produced >= self.max_tokens {
      self.done = true;
      return None;
    }

    let step_result = if !self.prefilled {
      self.prefilled = true;
      // Take the prompt-history payload (consumed by the prefill `_step`'s
      // processor-history accumulation); freed after this single use to
      // match the per-step state's small-footprint invariant. Same
      // discipline for `image_spans` — the per-model
      // `forward_embeddings_multimodal` override receives the spans by
      // reference for this single call, then the iterator-local storage
      // is dropped.
      let prompt_tokens = self.prompt_history.take().unwrap_or_default();
      let spans = self.image_spans.take().unwrap_or_default();
      let slabs = self.image_slabs.take().unwrap_or_default();
      // Free the per-image slabs AFTER the prefill runs (whether it
      // succeeded or failed) — `slabs` is moved in and dropped at the
      // end of this block, releasing the vision-feature Arrays' mlx-c
      // refcounts once the prefill has consumed them.
      self.prefill_step(&prompt_tokens, &spans, &slabs)
    } else {
      match self.last {
        Some(t) => self.decode_step(t),
        None => {
          // Unreachable: `last` is `Some` after the first step. Defend
          // by ending the iterator rather than feeding an empty window.
          self.done = true;
          return None;
        }
      }
    };

    match step_result {
      Ok(mut step) => {
        self.produced += 1;
        self.last = Some(step.token);
        // Same eos discipline as `lm::generate`: the eos token IS
        // yielded (faithful to mlx-vlm `_step` semantics), then the
        // iterator fuses.
        if self.eos.contains(&step.token) {
          self.done = true;
          // #114: surface "stop" on the EOS-token step (matches
          // `lm::generate::Generator::next`).
          step.finish_reason = Some(FinishReason::Eos);
        }
        Some(Ok(step))
      }
      Err(e) => {
        self.done = true;
        Some(Err(e))
      }
    }
  }
}

/// Validate the marker run for the multimodal prompt WITHOUT the per-image
/// token count, returning the splice base offset (the leading edge where the
/// image placeholders go).
///
/// Mirrors the non-count validation [`crate::vlm::prompt::insert_image_tokens`]
/// performs — but factored out so the count-dependent assembly can follow the
/// per-image processor run (a native-resolution model's per-image
/// `num_tokens` is only known post-processing). Checks:
/// - the first contiguous run of `marker_id` has length exactly `image_count`
///   (the chat-template producer emits `marker * image_count`); a mismatch is
///   [`Error::LengthMismatch`];
/// - no further `marker_id` occurs after that run
///   ([`Error::InvariantViolation`] — the splice supports one marker run,
///   mirroring python `prompt_utils`' `prompt.split("<image>")` 2-chunk
///   contract);
/// - when no marker is present, [`MarkerPolicy::Required`] is
///   [`Error::MissingField`] and [`MarkerPolicy::PrependIfAbsent`] yields
///   base `0` (prepend).
///
/// `image_count` is `> 0` here (the caller's zero-image passthrough handles the
/// empty case). Returns the base offset: the position of the first marker (when
/// present) or `0` (the PrependIfAbsent path).
fn validate_marker_run(
  text_tokens: &[u32],
  image_count: usize,
  marker_id: u32,
  policy: MarkerPolicy,
) -> Result<usize> {
  match text_tokens.iter().position(|&t| t == marker_id) {
    Some(run_start) => {
      let run_end = text_tokens[run_start..]
        .iter()
        .position(|&t| t != marker_id)
        .map_or(text_tokens.len(), |off| run_start + off);
      let run_len = run_end - run_start;
      // Reject extra markers after the consumed run.
      if text_tokens[run_end..].contains(&marker_id) {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "vlm_generate: image marker occurrences (after the first contiguous run)",
          "must be 0 — the splice supports at most one contiguous marker run (mirrors python \
           prompt_utils' `prompt.split(\"<image>\")` 2-chunk contract)",
        )));
      }
      // Reject contiguous-run length mismatch with `image_count` (the producer
      // emits exactly `marker * image_count` adjacent markers).
      if run_len != image_count {
        return Err(Error::LengthMismatch(LengthMismatchPayload::new(
          "vlm_generate: contiguous marker run length vs image_count (the chat-template producer \
             should emit exactly `marker * image_count` adjacent markers; mismatch suggests \
             caller/template skew)",
          image_count,
          run_len,
        )));
      }
      Ok(run_start)
    }
    None => {
      if policy == MarkerPolicy::Required {
        return Err(Error::MissingField(MissingFieldPayload::new(
          "vlm_generate (MarkerPolicy::Required, image_count > 0; chat-template / tokenizer drift \
             detected — pass MarkerPolicy::PrependIfAbsent if the model uses the \
             PROMPT_WITH_IMAGE_TOKEN-family formatter)",
          "image_marker_id token in text_tokens",
        )));
      }
      // PrependIfAbsent → the placeholders go at the front.
      Ok(0)
    }
  }
}

/// Build the assembled prompt + per-image spans from the per-image token
/// `counts`, replacing the validated marker run (one marker per image) with
/// each image's `counts[i]` placeholder ids in order.
///
/// This is the per-image-count generalization of
/// [`crate::vlm::prompt::insert_image_tokens`] + the span computation
/// [`crate::vlm::prompt::assemble_multimodal_prompt`] does internally: a
/// fixed-grid model passes a uniform `counts` (every entry the constant grid
/// size) and gets the historical uniform-stride spans; a native-resolution
/// model passes its variable per-image counts and gets accumulating spans.
///
/// `base` is the splice leading edge from [`validate_marker_run`] — the marker
/// run's start (marker present) or `0` (PrependIfAbsent). When the marker is
/// present the run spans `base..base + counts.len()` (one marker per image,
/// already validated); when absent the placeholders are prepended at `base ==
/// 0`. The placeholder block has total width `Σ counts`; the surrounding text
/// (before `base` and after the marker run, or the whole text after a
/// prepend) is copied verbatim.
///
/// Returns `(assembled_tokens, image_spans)` where `image_spans[i]` is the
/// half-open `(start, end)` range image `i`'s `counts[i]` placeholders occupy
/// in `assembled_tokens`.
///
/// # Errors
/// - [`Error::InvariantViolation`] if any `counts[i] == 0` (an image that
///   expands to zero placeholders would silently drop — degenerate
///   model/config state, fail closed, mirroring `insert_image_tokens`);
/// - [`Error::ArithmeticOverflow`] if the cumulative placeholder total or the
///   output capacity overflows `usize`;
/// - [`Error::OutOfMemory`] if the output reservation fails.
fn assemble_prompt_with_counts(
  text_tokens: &[u32],
  base: usize,
  marker_id: u32,
  image_token_id: u32,
  counts: &[usize],
) -> Result<AssembledPrompt> {
  // The marker run is present iff the text actually contains the marker; the
  // base from `validate_marker_run` is the run start in that case, else 0.
  let marker_present = text_tokens.contains(&marker_id);
  // The run length is `counts.len()` (one marker per image, validated). With no
  // marker (PrependIfAbsent) the run length is 0 — the placeholders are a pure
  // prepend.
  let run_len = if marker_present { counts.len() } else { 0 };

  // Total placeholder width Σ counts, checked. A zero per-image count would
  // silently drop that image — reject it.
  let mut placeholder_total: usize = 0;
  for (i, &c) in counts.iter().enumerate() {
    if c == 0 {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "vlm_generate: per-image token count (with image_count > 0)",
        "must be > 0 — an image expanding to zero placeholders would silently drop; the \
         processor reported a degenerate count",
      )));
    }
    placeholder_total = placeholder_total.checked_add(c).ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "vlm_generate: cumulative placeholder total (Σ per-image counts)",
        "usize",
        [
          ("placeholder_total", placeholder_total as u64),
          ("count_i", c as u64),
          ("i", i as u64),
        ],
      ))
    })?;
  }

  // Output capacity = text.len() + placeholder_total - run_len (the run is
  // replaced by the placeholder block). Checked to surface overflow as a
  // recoverable error.
  let cap = text_tokens
    .len()
    .checked_add(placeholder_total)
    .and_then(|n| n.checked_sub(run_len))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
        "vlm_generate: assembled prompt capacity (text_len + placeholder_total - run_len)",
        "usize",
        [
          ("text_len", text_tokens.len() as u64),
          ("placeholder_total", placeholder_total as u64),
          ("run_len", run_len as u64),
        ],
      ))
    })?;
  let mut out: Vec<u32> = try_with_capacity(cap)?;
  // Leading text (before the splice base). For the prepend path base == 0 so
  // this is empty.
  out.extend_from_slice(&text_tokens[..base]);
  // Placeholder block (Σ counts copies of image_token_id) + per-image spans.
  let mut spans: Vec<(usize, usize)> = try_with_capacity(counts.len())?;
  let mut cursor = base;
  for &c in counts.iter() {
    let start = cursor;
    let end = start + c; // `start + c <= base + placeholder_total <= cap`; no overflow.
    out.extend(std::iter::repeat_n(image_token_id, c));
    spans.push((start, end));
    cursor = end;
  }
  // Trailing text. With a marker present, skip the consumed run
  // (`base..base + run_len`); with a prepend, copy the whole text.
  if marker_present {
    out.extend_from_slice(&text_tokens[base + run_len..]);
  } else {
    out.extend_from_slice(text_tokens);
  }
  Ok((out, spans))
}

/// `logits[:, -1, :]` — slice the final sequence position of a `[B, S, V]`
/// logits tensor and drop the (now size-1) sequence axis ⇒ `[B, V]`.
///
/// Replicates `lm::generate::last_position` (kept private there) because
/// the two loops share the exact same final-position contract — a wrong
/// rank or a zero-length S/V axis is a recoverable `Err`, never a panic.
/// A future refactor can hoist this into a shared helper without changing
/// behavior.
fn last_position(logits: &Array) -> Result<Array> {
  let shape = logits.shape();
  let rank = shape.len() as u32;
  if shape.len() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "vlm_generate: expected [B, S, V] logits from forward (rank 3)",
      rank,
      shape,
    )));
  }
  if shape[1] == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "vlm_generate: forward logits S axis (logits[:, -1, :] requires S >= 1)",
      "must be >= 1",
      format!("{} (full shape {:?})", shape[1], shape),
    )));
  }
  if shape[2] == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "vlm_generate: forward logits V axis (logits[:, -1, :] requires V >= 1)",
      "must be >= 1",
      format!("{} (full shape {:?})", shape[2], shape),
    )));
  }
  let (b, s, v) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
  let sliced = ops::indexing::slice(logits, &[0, s - 1, 0], &[b, s, v], &[1, 1, 1])?;
  ops::shape::squeeze_axes(&sliced, &[1])
}

#[cfg(test)]
mod tests {
  //! Closed-form coverage for the VLM generation glue:
  //!
  //! - [`VlmGenConfig`] accessors (pure getters; no mlx).
  //! - the [`last_position`] rank / S-axis / V-axis guards (crafted shapes,
  //!   typed-error matching) plus its happy `[B, S, V] -> [B, V]` contract.
  //! - the [`VlmDecode`] iterator's deferred-`pending_err` channel and the
  //!   `last == None` defensive end-of-iteration arm (constructed directly,
  //!   since this module can reach the private struct + fields).
  //! - [`VlmDecode::prefill_step`]'s `T == 0` empty-prompt guard, the
  //!   KV-cache layer-offset-disagreement guard, and the absolute-offset
  //!   `checked_add` overflow guard — driven through a deterministic
  //!   in-crate VLM mock + a fixed-offset mock cache so each typed error is
  //!   exercised against an independent oracle, never the fn under test.
  //! - the full [`vlm_generate`] vision path's per-image encode-shape
  //!   contract (a rank-1 `encode_image` output -> `Error::RankMismatch`),
  //!   driven through a real on-disk PNG so `load_image` / `preprocess`
  //!   run end-to-end before the shape check fires.

  use super::*;
  use crate::lm::cache::{CacheConfig, KvCache, MaskMode, make_prompt_cache};

  // ── deterministic in-crate VLM mock ────────────────────────────────────
  //
  // Implements both `lm::model::Model` and `vlm::model::Model`. `forward` /
  // `forward_embeddings` return a fixed `[B, S, V]` (resp. `[1, S, V]`)
  // logits tile (argmax == last vocab index); `embed_tokens` returns
  // `[1, T, hidden]` zeros; `encode_image` returns a controllable shape so
  // the cross-model encode-shape contract can be exercised. The default
  // `merge_embeddings` + default `forward_embeddings_multimodal` (which
  // dispatches to `forward_embeddings`) are inherited.

  /// What shape `encode_image` should fabricate for a given preprocessed
  /// image — drives the per-image shape-contract branches in `vlm_generate`.
  #[derive(Clone, Copy)]
  enum EncodeShape {
    /// A well-formed `[rows, hidden]` feature slab.
    Rank2 { rows: usize, hidden: usize },
    /// A malformed rank-1 `[n]` output (violates the `[N, D]` contract).
    Rank1 { n: usize },
  }

  struct VlmMock {
    vocab: usize,
    hidden: usize,
    encode: EncodeShape,
  }

  impl VlmMock {
    fn new(vocab: usize, hidden: usize) -> Self {
      Self {
        vocab,
        hidden,
        encode: EncodeShape::Rank2 { rows: 1, hidden },
      }
    }

    fn with_encode(mut self, encode: EncodeShape) -> Self {
      self.encode = encode;
      self
    }

    /// `[batch, seq, vocab]` tile of `0..vocab` (argmax == vocab - 1).
    fn logits(&self, batch: usize, seq: usize) -> Result<Array> {
      let mut data: Vec<f32> = Vec::with_capacity(batch * seq * self.vocab);
      for _ in 0..batch * seq {
        for v in 0..self.vocab {
          data.push(v as f32);
        }
      }
      Array::from_slice::<f32>(&data, &(batch, seq, self.vocab))
    }
  }

  impl crate::lm::model::Model for VlmMock {
    fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
      let shape = tokens.shape();
      let (b, s) = match shape.as_slice() {
        [b, s] => (*b, *s),
        [s] => (1, *s),
        _ => (1, 1),
      };
      self.logits(b, s)
    }

    fn forward_embeddings(
      &self,
      embeddings: &Array,
      _cache: &mut [Box<dyn KvCache>],
    ) -> Result<Array> {
      // embeddings is [1, S, D]; emit [1, S, V].
      let shape = embeddings.shape();
      let s = if shape.len() == 3 { shape[1] } else { 1 };
      self.logits(1, s)
    }

    fn supports_input_embeddings(&self) -> bool {
      true
    }
  }

  impl Model for VlmMock {
    fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
      let shape = tokens.shape();
      let t = match shape.as_slice() {
        [_b, t] => *t,
        [t] => *t,
        _ => 1,
      };
      let data = vec![0.0_f32; t * self.hidden];
      Array::from_slice::<f32>(&data, &(1_usize, t, self.hidden))
    }

    fn encode_image(&self, _image: &ProcessedImage) -> Result<Array> {
      match self.encode {
        EncodeShape::Rank2 { rows, hidden } => {
          let data = vec![1.0_f32; rows * hidden];
          Array::from_slice::<f32>(&data, &(rows, hidden))
        }
        EncodeShape::Rank1 { n } => {
          let data = vec![1.0_f32; n];
          Array::from_slice::<f32>(&data, &(n,))
        }
      }
    }
  }

  // ── fixed-offset mock cache ─────────────────────────────────────────────
  //
  // Only `offset()` is exercised by `prefill_step`'s initial-offset read; the
  // rest of the `KvCache` surface is inert (the prefill paths under test
  // either fail before any `update`, or the model mock ignores the cache).

  struct FixedOffsetCache {
    offset: usize,
  }

  impl KvCache for FixedOffsetCache {
    fn offset(&self) -> usize {
      self.offset
    }
    fn update(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
      Ok((keys.try_clone()?, values.try_clone()?))
    }
    fn state(&self) -> Result<Vec<Array>> {
      Ok(Vec::new())
    }
    fn set_state(&mut self, _state: Vec<Array>) -> Result<()> {
      Ok(())
    }
    fn materialize(&mut self) -> Result<()> {
      Ok(())
    }
    fn make_mask(
      &self,
      _n: usize,
      _window_size: Option<usize>,
      _return_array: bool,
    ) -> Result<MaskMode> {
      Ok(MaskMode::None)
    }
    fn nbytes(&self) -> usize {
      0
    }
    fn is_empty(&self) -> bool {
      self.offset == 0
    }
    fn copy(&self) -> Result<Box<dyn KvCache>> {
      Ok(Box::new(FixedOffsetCache {
        offset: self.offset,
      }))
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
      self
    }
    fn reference_class_name(&self) -> &'static str {
      "FixedOffsetCache"
    }
  }

  /// Build a bare `VlmDecode` against `model`/`cache` with prefill payload
  /// fields supplied — the rest defaulted to a benign "ready to poll" state.
  /// Lets the offset / overflow / empty-prompt prefill branches be reached
  /// without the (image-file-dependent) `vlm_generate` construction path.
  fn decode_with<'a>(
    model: &'a VlmMock,
    cache: Vec<Box<dyn KvCache>>,
    prefill_step_size: usize,
    prompt: Vec<u32>,
    spans: Vec<(usize, usize)>,
    slabs: Vec<Array>,
  ) -> VlmDecode<'a, VlmMock> {
    VlmDecode {
      model,
      cache: RefCell::new(cache),
      sampler: RefCell::new(Sampler::Argmax),
      processors: Vec::new(),
      history: RefCell::new(Vec::new()),
      eos: Vec::new(),
      max_tokens: 8,
      produced: 0,
      prefill_step_size,
      last: None,
      prefilled: false,
      image_slabs: Some(slabs),
      image_spans: Some(spans),
      prompt_history: Some(prompt),
      pending_err: None,
      done: false,
    }
  }

  // ── VlmGenConfig accessors (lines 198-225) ──────────────────────────────

  /// Every `VlmGenConfig` getter returns exactly what was constructed; the
  /// `image_marker_id` builder flips the default `None` to `Some`.
  #[test]
  fn vlm_gen_config_accessors_roundtrip() {
    let lm = GenConfig::default().with_max_tokens(7);
    let cfg = VlmGenConfig::new(lm, 99, 3, MarkerPolicy::Required);

    // lm_ref / lm_mut expose the wrapped GenConfig (198-205).
    assert_eq!(cfg.lm_ref().max_tokens, 7);
    assert_eq!(cfg.image_token_id(), 99); // 208-210
    assert_eq!(cfg.image_marker_id(), None); // 213-216 (default None)
    assert_eq!(cfg.num_tokens_per_image(), 3); // 218-221
    assert!(cfg.marker_policy().is_required()); // 223-226

    // with_image_marker_id sets a distinct marker; lm_mut mutates in place.
    let mut cfg = cfg.with_image_marker_id(Some(42));
    assert_eq!(cfg.image_marker_id(), Some(42));
    cfg.lm_mut().max_tokens = 11;
    assert_eq!(cfg.lm_ref().max_tokens, 11);
    // Unchanged fields survive the builder.
    assert_eq!(cfg.image_token_id(), 99);
    assert_eq!(cfg.num_tokens_per_image(), 3);
    assert!(cfg.marker_policy().is_required());
  }

  // ── last_position guards + happy path (lines 988-1015) ──────────────────

  /// Rank != 3 -> `RankMismatch` naming `[B, S, V]`, carrying the observed
  /// rank + shape (992-997).
  #[test]
  fn last_position_rejects_non_rank3() {
    let two_d = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2_usize, 2)).unwrap();
    let err = last_position(&two_d).unwrap_err();
    match err {
      Error::RankMismatch(p) => {
        assert!(p.context().contains("rank 3"), "ctx: {}", p.context());
        assert_eq!(p.actual(), 2, "observed rank carried");
        assert_eq!(p.actual_shape(), &[2, 2], "observed shape carried");
      }
      other => panic!("expected RankMismatch, got {other:?}"),
    }
  }

  /// A zero-length S axis -> `OutOfRange` on the S axis (998-1003). Closed
  /// form: `[1, 0, 4]` has S == 0.
  #[test]
  fn last_position_rejects_zero_s_axis() {
    let data: Vec<f32> = Vec::new();
    let z = Array::from_slice::<f32>(&data, &(1_usize, 0, 4)).unwrap();
    let err = last_position(&z).unwrap_err();
    match err {
      Error::OutOfRange(p) => {
        assert!(p.context().contains("S axis"), "ctx: {}", p.context());
        assert!(
          p.value().starts_with('0'),
          "value reports S=0: {}",
          p.value()
        );
      }
      other => panic!("expected OutOfRange(S), got {other:?}"),
    }
  }

  /// A zero-length V axis -> `OutOfRange` on the V axis (1005-1010). Closed
  /// form: `[1, 2, 0]` has V == 0 (and S == 2 > 0 so the S guard passes).
  #[test]
  fn last_position_rejects_zero_v_axis() {
    let data: Vec<f32> = Vec::new();
    let z = Array::from_slice::<f32>(&data, &(1_usize, 2, 0)).unwrap();
    let err = last_position(&z).unwrap_err();
    match err {
      Error::OutOfRange(p) => {
        assert!(p.context().contains("V axis"), "ctx: {}", p.context());
        assert!(
          p.value().starts_with('0'),
          "value reports V=0: {}",
          p.value()
        );
      }
      other => panic!("expected OutOfRange(V), got {other:?}"),
    }
  }

  /// Happy path: `[B, S, V]` -> `[B, V]` keeping the FINAL position.
  /// Oracle: build `[1, 3, 2]` with per-position rows `[0,1], [2,3], [4,5]`;
  /// the slice + squeeze must yield exactly the LAST row `[4, 5]` at shape
  /// `[1, 2]`.
  #[test]
  fn last_position_slices_final_position() {
    // positions 0..3 along S; values per (s): [s*2, s*2+1]. Last = [4,5].
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let logits = Array::from_slice::<f32>(&data, &(1_usize, 3, 2)).unwrap();
    let mut out = last_position(&logits).unwrap();
    assert_eq!(out.shape(), vec![1, 2], "S axis dropped");
    assert_eq!(
      out.to_vec::<f32>().unwrap(),
      vec![4.0, 5.0],
      "final position kept"
    );
  }

  // ── VlmDecode deferred-err + last==None defense (lines 919-921, 948-952) ─

  /// A `pending_err` is yielded as the FIRST `next()` (919-921) and the
  /// iterator fuses afterwards (the `done` short-circuit at 916-918).
  #[test]
  fn vlm_decode_pending_err_is_first_then_fuses() {
    let model = VlmMock::new(4, 2);
    let cache: Vec<Box<dyn KvCache>> = Vec::new();
    let mut it = VlmDecode {
      model: &model,
      cache: RefCell::new(cache),
      sampler: RefCell::new(Sampler::Argmax),
      processors: Vec::new(),
      history: RefCell::new(Vec::new()),
      eos: Vec::new(),
      max_tokens: 5,
      produced: 0,
      prefill_step_size: 1,
      last: None,
      prefilled: true, // pending_err short-circuits before any prefill
      image_slabs: None,
      image_spans: None,
      prompt_history: None,
      pending_err: Some(Error::EmptyInput(EmptyInputPayload::new(
        "sentinel pending error",
      ))),
      done: false,
    };
    let err = it.next().expect("yields the pending err").unwrap_err();
    assert!(
      matches!(err, Error::EmptyInput(ref p) if p.context().contains("sentinel")),
      "deferred pending_err surfaced, got {err:?}"
    );
    assert!(it.next().is_none(), "fuses after the single deferred Err");
  }

  /// The defensive `last == None` arm after prefill (948-952): with
  /// `prefilled == true` but `last == None`, `next()` ends the iterator
  /// rather than feeding an empty decode window.
  #[test]
  fn vlm_decode_prefilled_but_no_last_ends() {
    let model = VlmMock::new(4, 2);
    let cache: Vec<Box<dyn KvCache>> = Vec::new();
    let mut it = VlmDecode {
      model: &model,
      cache: RefCell::new(cache),
      sampler: RefCell::new(Sampler::Argmax),
      processors: Vec::new(),
      history: RefCell::new(Vec::new()),
      eos: Vec::new(),
      max_tokens: 5,
      produced: 0,
      prefill_step_size: 1,
      last: None,
      prefilled: true,
      image_slabs: None,
      image_spans: None,
      prompt_history: None,
      pending_err: None,
      done: false,
    };
    assert!(
      it.next().is_none(),
      "prefilled + last==None ends the iterator"
    );
    // And it stays fused.
    assert!(it.next().is_none());
  }

  // ── prefill_step empty prompt (lines 770-774) ───────────────────────────

  /// `prefill_step` with an empty assembled prompt (T == 0) -> `EmptyInput`
  /// (no chunk can produce logits).
  #[test]
  fn prefill_step_empty_prompt_is_empty_input() {
    let model = VlmMock::new(4, 2);
    let cache: Vec<Box<dyn KvCache>> = Vec::new();
    let it = decode_with(&model, cache, 4, Vec::new(), Vec::new(), Vec::new());
    let err = it.prefill_step(&[], &[], &[]).unwrap_err();
    match err {
      Error::EmptyInput(p) => assert!(
        p.context().contains("T=0") || p.context().contains("prefill"),
        "ctx names the empty-prompt prefill: {}",
        p.context()
      ),
      other => panic!("expected EmptyInput, got {other:?}"),
    }
  }

  // ── prefill_step layer-offset disagreement (lines 789-811, 798-804) ─────

  /// Two cache layers at DIFFERENT offsets -> `LengthMismatch` BEFORE any
  /// embed/forward. Oracle: layer 0 fresh (offset 0), layer 1 forced to
  /// offset 5 — the guard compares layer i to layer 0 and reports
  /// `expected = layer0_off (0)`, `actual = layer1_off (5)`.
  #[test]
  fn prefill_step_rejects_mismatched_layer_offsets() {
    let model = VlmMock::new(4, 2);
    let cache: Vec<Box<dyn KvCache>> = vec![
      Box::new(FixedOffsetCache { offset: 0 }),
      Box::new(FixedOffsetCache { offset: 5 }),
    ];
    let it = decode_with(&model, cache, 4, vec![1, 2, 3], Vec::new(), Vec::new());
    let err = it.prefill_step(&[1, 2, 3], &[], &[]).unwrap_err();
    match err {
      Error::LengthMismatch(p) => {
        assert!(
          p.context().contains("offset"),
          "ctx names the offset disagreement: {}",
          p.context()
        );
        assert_eq!(p.expected(), 0, "layer-0 offset is the reference");
        assert_eq!(p.actual(), 5, "the disagreeing layer's offset");
      }
      other => panic!("expected LengthMismatch(offsets), got {other:?}"),
    }
  }

  // ── prefill_step absolute-offset checked_add overflow (lines 873-882) ───

  /// A near-`usize::MAX` restored cache offset overflows once the chunk
  /// cursor advances past chunk 0. Oracle: single-layer cache at
  /// `usize::MAX - 1`, `prefill_step_size == 1`, a pure-text 3-token prompt
  /// (no image spans, so the mock's `forward_embeddings` runs each chunk).
  /// Chunk 0 cursor 0 -> (MAX-1)+0 ok; chunk 1 cursor 1 -> (MAX-1)+1 ok;
  /// chunk 2 cursor 2 -> (MAX-1)+2 overflows -> `ArithmeticOverflow`.
  #[test]
  fn prefill_step_offset_overflow_is_arithmetic_overflow() {
    let model = VlmMock::new(4, 2);
    let cache: Vec<Box<dyn KvCache>> = vec![Box::new(FixedOffsetCache {
      offset: usize::MAX - 1,
    })];
    // Pure-text prompt (no markers/spans) so no merge runs and the mock's
    // forward_embeddings produces the per-chunk logits.
    let it = decode_with(&model, cache, 1, vec![10, 11, 12], Vec::new(), Vec::new());
    let err = it.prefill_step(&[10, 11, 12], &[], &[]).unwrap_err();
    match err {
      Error::ArithmeticOverflow(p) => assert!(
        p.context().contains("cache offset") || p.context().contains("cursor"),
        "ctx names the offset+cursor add: {}",
        p.context()
      ),
      other => panic!("expected ArithmeticOverflow, got {other:?}"),
    }
  }

  // ── vlm_generate per-image encode-shape contract (lines 464-474) ────────

  /// Write a tiny valid PNG to a temp path, then drive the full
  /// `vlm_generate` vision path with a mock whose `encode_image` returns a
  /// malformed rank-1 array. The per-image shape check rejects it with
  /// `Error::RankMismatch` naming the rank-2 `[N, D]` requirement (467-471).
  /// `load_image` + `preprocess` run end-to-end first (so this is the real
  /// construction path, not a direct prefill call).
  #[test]
  fn vlm_generate_rejects_rank1_encode_output() {
    // A 2x2 RGB PNG is a valid decode + preprocess input.
    let dir =
      std::env::temp_dir().join(format!("mlxrs-vlm-generate-encode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("tiny.png");
    let mut buf = ::image::RgbImage::new(2, 2);
    for y in 0..2 {
      for x in 0..2 {
        buf.put_pixel(x, y, ::image::Rgb([128, 64, 200]));
      }
    }
    ::image::DynamicImage::ImageRgb8(buf)
      .save_with_format(&path, ::image::ImageFormat::Png)
      .expect("encode tiny PNG");

    let model = VlmMock::new(4, 2).with_encode(EncodeShape::Rank1 { n: 1 });
    // The default fixed-grid processor (num_tokens = 1) preprocesses the real
    // PNG; the rank-1 encode output then fails the per-image shape check.
    let proc = model.image_processor(1);
    // Prompt: a single marker token (id 7) so the marker run validates with
    // one image, per-image count = 1.
    let cfg = VlmGenConfig::new(
      GenConfig::default().with_max_tokens(4),
      7, // image_token_id == marker_id (single-token marker)
      1, // num_tokens_per_image (the fixed-grid default count)
      MarkerPolicy::Required,
    );
    let cache = make_prompt_cache(&CacheConfig {
      num_hidden_layers: 1,
      sliding_window: None,
    });
    let res = vlm_generate(
      &model,
      proc.as_ref(),
      &[7u32],
      std::slice::from_ref(&path),
      cache,
      cfg,
    );

    // Best-effort cleanup before asserting.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);

    let err = res.err().expect("rank-1 encode_image output must error");
    match err {
      Error::RankMismatch(p) => {
        assert!(
          p.context().contains("encode_image") && p.context().contains("[N, D]"),
          "ctx names the encode_image rank-2 contract: {}",
          p.context()
        );
        assert_eq!(p.actual(), 1, "observed rank-1 carried");
      }
      other => panic!("expected RankMismatch from the encode-shape check, got {other:?}"),
    }
  }

  // ── max_tokens == 0 short-circuit (lines 336-338) ───────────────────────

  /// `max_tokens == 0` returns an empty iterator with NO vision work — even
  /// with a (nonexistent) image path, because the short-circuit precedes the
  /// image pipeline. Oracle: zero yielded items, and the bogus path is never
  /// opened (no error surfaces at construction).
  #[test]
  fn vlm_generate_zero_max_tokens_is_empty_no_vision() {
    let model = VlmMock::new(4, 2);
    let proc = model.image_processor(1);
    let cfg = VlmGenConfig::new(
      GenConfig::default().with_max_tokens(0),
      7,
      1,
      MarkerPolicy::Required,
    );
    let cache = make_prompt_cache(&CacheConfig {
      num_hidden_layers: 1,
      sliding_window: None,
    });
    // A path that does not exist: if vision work ran, load_image would error.
    let bogus = PathBuf::from("/nonexistent/mlxrs-vlm-no-such-image.png");
    let mut it = vlm_generate(&model, proc.as_ref(), &[7u32], &[bogus], cache, cfg)
      .expect("max_tokens==0 short-circuits to Ok(empty) before any vision work");
    assert!(it.next().is_none(), "zero-budget run yields nothing");
  }

  // ── invalid cfg is eager Err (lines 323) ────────────────────────────────

  /// An invalid `cfg.lm` (negative temp) is an EAGER `Err` from
  /// `vlm_generate` (the `cfg.lm.validate()?` gate at construction), before
  /// the max_tokens / zero-image / multimodal split.
  #[test]
  fn vlm_generate_invalid_cfg_is_eager_err() {
    let model = VlmMock::new(4, 2);
    let proc = model.image_processor(1);
    let cfg = VlmGenConfig::new(
      GenConfig::default().with_temp(-1.0),
      7,
      1,
      MarkerPolicy::Required,
    );
    let cache: Vec<Box<dyn KvCache>> = Vec::new();
    let res = vlm_generate(&model, proc.as_ref(), &[7u32], &[], cache, cfg);
    match res.err().expect("invalid temp must be an eager Err") {
      Error::OutOfRange(p) => assert!(
        p.context().contains("temp"),
        "eager validate() surfaced temp range error: {}",
        p.context()
      ),
      other => panic!("expected eager OutOfRange(temp), got {other:?}"),
    }
  }
}
