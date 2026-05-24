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
//! 1. *Assemble + validate prompt FIRST* (Codex round-3 finding-2:
//!    deterministic validation must precede expensive vision work).
//!    [`crate::vlm::prompt::insert_image_tokens`] splices
//!    `num_tokens_per_image` placeholders per image into `text_tokens`
//!    at the marker run; per-image spans are computed inline as
//!    `[(base + i*N, base + (i+1)*N)]`. We deliberately invoke the
//!    splice primitive directly (NOT
//!    [`crate::vlm::prompt::assemble_multimodal_prompt`], which also
//!    builds an O(T*T) bidirectional-within-image attention mask)
//!    because the trait's `forward_embeddings(embeds, cache)` signature
//!    has no way to thread that mask through to a per-model attention
//!    layer; instead,
//!    [`crate::vlm::model::Model::forward_embeddings_multimodal`]
//!    receives chunk-local `image_spans` + a `cache_offset` BY VALUE on
//!    every prefill chunk so a mask-requiring per-model override builds
//!    its `[chunk × (past + chunk)]` mask without any `&self` state. A
//!    malformed prompt (missing marker, marker count mismatch,
//!    `num_tokens_per_image == 0`) errors here BEFORE any image is
//!    loaded / preprocessed / encoded.
//! 2. *Preprocess + encode images.* For each path in `images`,
//!    `vlm::image::load_image(path) → preprocess(…, image_processor_config)`
//!    — using the caller-supplied [`crate::vlm::image::ImageProcessorConfig`]
//!    (the loaded processor's config, mirroring mlx-vlm `generate(model,
//!    processor, …)`), NOT one re-derived from the model — yields
//!    `[H, W, 3]` f32; `model.encode_image(image)` lifts each into
//!    `[N_i, D]` vision-encoder embeddings. Each `encode_image` call
//!    is validated to return EXACTLY `[num_tokens_per_image, D]` rows
//!    — a model with variable-per-image counts must pad/truncate
//!    inside its own `encode_image` to satisfy this cross-model splice
//!    contract. The per-image slabs are kept SEPARATE (one `[N_i, D]`
//!    Array per image), NOT pre-concatenated — the prefill gathers only
//!    the slabs whose span falls in the current chunk (step 4).
//! 3. *(no global embed/merge).* VLM-8: text embedding + image merge are
//!    NOT done over the full sequence here — they happen INCREMENTALLY
//!    per chunk in step 4, so peak memory is bounded by `prefill_step_size
//!    · D` plus the (inherent) vision-feature slabs, not the full
//!    `T · D` merged sequence (which the image-token expansion inflates).
//! 4. *Offset-aware, span-aware chunked prefill (VLM-8).* The assembled
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
  error::{Error, Result, try_extend_from_slice, try_with_capacity},
  lm::{
    cache::KvCache,
    generate::{
      GenConfig, GenStep, LogitsProcessor, Sampler, make_logits_processors, make_sampler,
    },
  },
  ops,
  vlm::{
    image::{ImageProcessorConfig, load_image, preprocess},
    model::Model,
    prompt::{MarkerPolicy, insert_image_tokens},
  },
};

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
  pub lm: GenConfig,
  /// Token id of the image placeholder the splice emits (per-model — e.g.
  /// `<image>` or `<|image_pad|>`'s ID after tokenization). The merged
  /// embed sequence places `image_embeds` at every run of this id.
  pub image_token_id: u32,
  /// Token id the chat template emits where images go. Often the same as
  /// [`Self::image_token_id`] (single-token marker that BOTH delimits the
  /// splice site AND occupies the placeholder positions), but some
  /// models use distinct ids — e.g. `<|image|>` (marker) vs `<|image_pad|>`
  /// (placeholder). When `None`, defaults to [`Self::image_token_id`]
  /// (the common case).
  pub image_marker_id: Option<u32>,
  /// Number of image tokens per image — the per-model
  /// `num_tokens_per_image` (Qwen-VL variable, LLaVA fixed-grid, etc.).
  /// MUST match what [`Model::encode_image`] emits (`N_i` per image), or
  /// the splice will fail the `Σ widths == N_total` contract in
  /// [`Model::merge_embeddings`].
  pub num_tokens_per_image: usize,
  /// Marker-vs-prepend policy. See
  /// [`crate::vlm::prompt::MarkerPolicy`].
  pub marker_policy: MarkerPolicy,
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
/// **The image-preprocessing config is an explicit parameter, not derived
/// from the model.** mlx-vlm's `generate` / `generate_step` take the
/// `processor` separately from the `model` (`generate.py:1183`, `:966` —
/// `generate(model, processor, …)`): a VLM loaded via
/// [`crate::vlm::load::load`] returns a [`crate::vlm::load::LoadedVlmContext`]
/// whose [`processor`](crate::vlm::load::LoadedVlmContext::processor)
/// carries the parsed `preprocessor_config.json` /
/// `processor_config.json`. Pass that
/// processor's [`image_processor_config`](crate::vlm::load::Processor::image_processor_config)
/// here so real image prompts are preprocessed with the *loaded* config —
/// `vlm_generate` deliberately does NOT call
/// [`Model::image_processor_config`] itself (that would silently fall back
/// to the trait default / a stale baked-in config when a loaded processor
/// exists). A caller that only has a model and no separate processor can
/// still pass `&model.image_processor_config()` explicitly.
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
/// - `Error::ShapeMismatch` on a span/embed/dim contract violation in
///   [`Model::merge_embeddings`].
/// - `Error::ShapeMismatch` on a per-image encoder output that is not
///   `[cfg.num_tokens_per_image, D]` — every image MUST emit exactly
///   `num_tokens_per_image` feature rows, enforced per-image BEFORE
///   the slabs are concatenated (the cross-model splice contract; a
///   model with variable-per-image counts must pad/truncate inside its
///   own `encode_image`, or override the entry point).
///
/// Surface (as the iterator's first `Err`, exactly like
/// [`crate::lm::generate::generate_step`]):
///
/// - sampler / logits-processor construction failure
/// - any per-step forward / sample failure
pub fn vlm_generate<'a, M: Model + ?Sized>(
  model: &'a M,
  image_processor_config: &ImageProcessorConfig,
  text_tokens: &[u32],
  images: &[PathBuf],
  cache: Vec<Box<dyn KvCache>>,
  cfg: VlmGenConfig,
) -> Result<impl Iterator<Item = Result<GenStep>> + 'a> {
  // ── max_tokens == 0 SHORT-CIRCUIT ────────────────────────────────────
  // Mirror the LM-side contract: `lm::generate`'s iterator checks
  // `produced >= max_tokens` BEFORE running prefill (generate.rs:598),
  // so a `max_tokens == 0` request yields nothing and runs no model
  // call. The VLM multimodal path does its vision work (load /
  // preprocess / encode_image / merge) EAGERLY at construction, so
  // without this guard a zero-output request would still trigger image
  // I/O + vision compute + potential decode/OOM errors (Codex bundle-#62
  // round-2 finding). Short-circuit to an empty iterator BEFORE any
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
  // (Codex finding-2: contract drift between the two branches). Force
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
  // ORDER: deterministic prompt-shape validation FIRST (marker
  // presence, num_tokens_per_image, splice overflow), THEN the
  // expensive vision pipeline. A malformed request — missing marker
  // under MarkerPolicy::Required, marker-count mismatch,
  // num_tokens_per_image == 0 — must NOT load/preprocess/encode any
  // images before erroring; otherwise a service accepting a multi-
  // image request burns the full vision cost before surfacing the
  // template-drift error.
  let marker_id = cfg.image_marker_id.unwrap_or(cfg.image_token_id);
  let assembled_tokens = insert_image_tokens(
    text_tokens,
    images.len(),
    marker_id,
    cfg.image_token_id,
    cfg.num_tokens_per_image,
    cfg.marker_policy,
  )?;
  // Compute per-image spans directly from the splice base offset and
  // the per-image `num_tokens_per_image` width — byte-identical to what
  // [`crate::vlm::prompt::assemble_multimodal_prompt`] computes
  // internally, but without the trailing mask construction. The base
  // offset is either the position of the first marker (when present)
  // or 0 (PrependIfAbsent path). `insert_image_tokens` has already
  // validated the marker policy + run length, so the same lookup here
  // is just locating the splice's leading edge.
  let base: usize = text_tokens
    .iter()
    .position(|&t| t == marker_id)
    .unwrap_or_default();
  let mut image_spans: Vec<(usize, usize)> = try_with_capacity(images.len())?;
  for i in 0..images.len() {
    let start = base + i * cfg.num_tokens_per_image;
    let end = start + cfg.num_tokens_per_image;
    image_spans.push((start, end));
  }

  // Now the expensive vision path. Per-image preprocess + encode. We
  // deliberately encode one image at a time and concatenate the
  // resulting `[N_i, D]` slabs along axis 0: some models' `encode_image`
  // accepts a batch and some don't (the per-model encoder owns the
  // input layout / batch contract per [`Model::encode_image`]'s doc),
  // so the cross-model surface stays at "one image at a time" — the
  // simplest contract that every encoder satisfies. `Vec::with_capacity`
  // so the per-image push is amortized O(1) without reallocation.
  //
  // PER-IMAGE SHAPE VALIDATION: every `encode_image` MUST return exactly
  // `[cfg.num_tokens_per_image, D]`. The cross-model splice contract is
  // "one image emits exactly `num_tokens_per_image` features"; a model
  // with variable-per-image counts (some Qwen-VL configurations) MUST
  // pad / truncate / repeat inside its own `encode_image` to satisfy this
  // contract, or override the whole `vlm_generate` entry point with its
  // own variable-span loop. Without this per-slab check, a model
  // returning e.g. `[2, D]` for image 1 and `[4, D]` for image 2 with
  // `num_tokens_per_image = 3` would pass the merge layer's "total
  // widths == total rows" check (both = 6) but cause silent
  // marker-to-image misalignment (the first prompt span would consume
  // 2 rows from image 1 plus 1 row from image 2). Surface as
  // `Error::ShapeMismatch` instead.
  //
  // Image preprocessing uses the caller-supplied `image_processor_config`
  // — NOT `model.image_processor_config()`. mlx-vlm's `generate` /
  // `generate_step` receive the `processor` separately from the `model`
  // (`generate.py:1183`, `:966` — `generate(model, processor, …)`); a VLM
  // loaded via [`crate::vlm::load::load`] carries a `Box<dyn Processor>`
  // whose [`crate::vlm::load::Processor::image_processor_config`] reflects
  // the parsed `preprocessor_config.json` / `processor_config.json`. The
  // generation entry point therefore must NOT silently re-derive the
  // preprocessing params from the model (which would fall back to the
  // trait default / a stale baked-in config); the caller passes the
  // processor's config explicitly. A caller that only has a model can
  // still pass `&model.image_processor_config()`.
  let mut image_slabs: Vec<Array> = try_with_capacity(images.len())?;
  for (idx, path) in images.iter().enumerate() {
    let img = load_image(path)?;
    let pre = preprocess(&img, image_processor_config)?;
    let encoded = model.encode_image(&pre)?;
    let enc_shape = encoded.shape();
    let (rows, _d) = match enc_shape.as_slice() {
      [n, d] => (*n, *d),
      _ => {
        return Err(Error::ShapeMismatch {
          message: format!(
            "vlm_generate: encode_image for image #{idx} must return rank-2 [N, D], got {enc_shape:?}"
          ),
        });
      }
    };
    if rows != cfg.num_tokens_per_image {
      return Err(Error::ShapeMismatch {
        message: format!(
          "vlm_generate: encode_image for image #{idx} returned {rows} feature rows; expected \
           exactly cfg.num_tokens_per_image ({}). The cross-model splice contract requires one \
           image to emit exactly `num_tokens_per_image` features — a model with \
           variable-per-image counts must pad/truncate inside its own `encode_image` or override \
           the `vlm_generate` entry point",
          cfg.num_tokens_per_image,
        ),
      });
    }
    image_slabs.push(encoded);
  }
  // VLM-8 (offset-aware chunked multimodal prefill): we DELIBERATELY do
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
      // per-request spans (Codex round-2 finding: avoid the
      // cross-request hazard of model-side mask state).
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
  /// in the current chunk and merge them incrementally (VLM-8).
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
    Ok(GenStep {
      token,
      logprobs: Some(logprobs),
    })
  }

  /// The embed-based prefill (VLM-8 offset-aware chunked design): walk
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
  /// (Codex VLM-8 R1F1 — the bound is NOT `prefill_step_size · D` alone).
  /// This is still bounded by a model constant, never the full expanded
  /// `T`; and `W_max · D <= Σ N_i · D`, the vision-feature slab floor
  /// that is resident regardless (vision encoding can't be chunked). So
  /// the total is `Σ N_i · D` (inherent image features) plus
  /// `max(prefill_step_size, W_max) · D` (one chunk) — independent of the
  /// text length and of the image COUNT beyond the per-image width.
  ///
  /// **Two invariants make chunking correct for mask-requiring VLMs**
  /// (the structural fix VLM-8 escalated to — Codex bundle rounds 1-3):
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
      return Err(Error::ShapeMismatch {
        message: "vlm_generate: assembled prompt is empty (T=0); prefill cannot produce logits"
          .into(),
      });
    }
    // The cache may already hold tokens (a restored / pre-populated
    // prompt cache, or a model that pre-seeds the cache). Read the
    // starting offset so each chunk's `cache_offset` is ABSOLUTE
    // (`initial_offset + cursor`), not just the in-prefill `cursor` —
    // otherwise a mask-requiring override would size its mask too short
    // against a non-empty cache (Codex VLM-8 R1F3).
    //
    // All layers advance in lockstep during generation and a faithfully
    // saved/restored prompt cache loads every layer from the same state,
    // so they MUST share one offset — but the cache API does not enforce
    // that, and a corrupt/hand-built cache could differ per layer. The
    // override receives a single scalar `cache_offset`, so a per-layer
    // mismatch would silently size the mask wrong for some layers.
    // Validate equality up front and fail closed (Codex VLM-8 R2F1).
    let initial_offset = {
      let cache = self.cache.borrow();
      let mut iter = cache.iter();
      match iter.next() {
        None => 0, // no layers (degenerate); treated as offset 0
        Some(first) => {
          let off = first.offset();
          for (i, layer) in iter.enumerate() {
            if layer.offset() != off {
              return Err(Error::ShapeMismatch {
                message: format!(
                  "vlm_generate: KV cache layers disagree on offset (layer 0 = {off}, layer \
                   {} = {}); chunked-multimodal prefill needs one consistent cache offset to \
                   size per-chunk attention masks (a faithfully restored prompt cache has all \
                   layers at the same offset)",
                  i + 1,
                  layer.offset()
                ),
              });
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
      // release-wrap before the mask builder could reject it) — Codex
      // VLM-8 R2F1.
      let chunk_offset =
        initial_offset
          .checked_add(cursor)
          .ok_or_else(|| Error::ShapeMismatch {
            message: format!(
              "vlm_generate: cache offset {initial_offset} + chunk cursor {cursor} overflows usize"
            ),
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
      Ok(step) => {
        self.produced += 1;
        self.last = Some(step.token);
        // Same eos discipline as `lm::generate`: the eos token IS
        // yielded (faithful to mlx-vlm `_step` semantics), then the
        // iterator fuses.
        if self.eos.contains(&step.token) {
          self.done = true;
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
  if shape.len() != 3 {
    return Err(Error::ShapeMismatch {
      message: format!("vlm_generate: expected [B, S, V] logits from forward, got {shape:?}"),
    });
  }
  if shape[1] == 0 || shape[2] == 0 {
    return Err(Error::ShapeMismatch {
      message: format!(
        "vlm_generate: forward returned logits with a zero-length axis (got [B, S, V] \
         {shape:?}); logits[:, -1, :] requires S >= 1 and V >= 1"
      ),
    });
  }
  let (b, s, v) = (shape[0] as i32, shape[1] as i32, shape[2] as i32);
  let sliced = ops::indexing::slice(logits, &[0, s - 1, 0], &[b, s, v], &[1, 1, 1])?;
  ops::shape::squeeze_axes(&sliced, &[1])
}
