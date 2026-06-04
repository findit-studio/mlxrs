//! The top-level LFM2.5-VL model + VLM-factory registration — faithful 1:1 port
//! of `mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py`'s `Model`.
//!
//! [`Lfm2Vl`] assembles the four ported building blocks into the multimodal
//! forward:
//!
//! 1. the [native-resolution SigLIP2 vision tower](super::vision::VisionModel)
//!    runs each image's pre-flattened patches → per-image hidden states;
//! 2. each image's active patch rows (`pixel_attention_mask.sum()`) are reshaped
//!    to its `(H_p, W_p)` grid, [pixel-unshuffled](super::projector::PixelUnshuffleBlock)
//!    (downsample by `downsample_factor`), and [projected](super::projector::Lfm2VlMultiModalProjector)
//!    into the LM token space → `image_features (ΣN, D)`;
//! 3. the [language adapter](super::language::LanguageModel) embeds the prompt
//!    token ids, the projected features are
//!    [spliced](super::projector::merge_input_ids_with_image_features) into the
//!    embeddings at the `<image>`-token positions (the mask-driven
//!    `masked_scatter`), and
//! 4. the merged embeddings run through the LFM2 LM → logits.
//!
//! Every `nn.Linear` / `nn.Embedding` in the assembly is routed through the
//! shared quantize-aware
//! [`MaybeQuantizedLinear`](crate::nn::MaybeQuantizedLinear) /
//! `QuantizedEmbedding`, so the 8-bit `LiquidAI/LFM2.5-VL-450M-MLX-8bit`
//! checkpoint loads through the same code path as a dense one — the per-layer
//! `.scales` sibling is the "this layer is quantized" signal, and the
//! `quantization` block is resolved by
//! [`crate::lm::models::lfm2::resolve_quantization`] at load time.
//!
//! ## Native entry point vs. the cross-model trait
//!
//! The load-bearing faithful surface is the inherent
//! [`Lfm2Vl::get_input_embeddings`] (the 1:1 port of
//! `lfm2_vl.py:115-160`, taking the per-image
//! [`Lfm2VlImageInputs`] the [processor](super::processor) produces) and
//! [`Lfm2Vl::forward_multimodal`] (`lfm2_vl.py:188-205`). LFM2.5-VL's
//! native-resolution NaFlex encoding is inherently a function of
//! `(pixel_values, spatial_shapes)` per image, so it is expressed through these
//! triple-carrying methods rather than the cross-model
//! [`crate::vlm::model::Model::encode_image`] single-`Array` `[H, W, 3]`
//! contract (which a fixed-resolution CLIP/SigLIP model uses). The
//! [`crate::vlm::model::Model`] trait is still implemented in full so [`Lfm2Vl`]
//! registers in the VLM [`crate::vlm::load`] factory and the cross-model
//! `embed_tokens` / `image_processor_config` work; its
//! [`encode_image`](crate::vlm::model::Model::encode_image) handles a single
//! image's NaFlex patch rows (see that method's contract).

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{Error, InvariantViolationPayload, RankMismatchPayload, Result},
  lm::{
    cache::KvCache,
    model::Model as LmModel,
    models::lfm2::{Lfm2, resolve_quantization},
    quant::PerLayerQuantization,
  },
  ops::{
    self,
    shape::{concatenate, expand_dims_axes, reshape},
  },
  vlm::{
    image::{ImageProcessorConfig, Layout},
    load::{LoadedVlmModel, VlmModelConstructor, VlmTypeRegistry},
    model::{ImageProcessor, Model as VlmModel, NativeResolution, ProcessedImage},
    models::lfm2_vl::{
      config::ModelConfig,
      language::LanguageModel,
      processor::{
        Lfm2VlImageInputs, Lfm2VlProcessorConfig, num_image_tokens_from_patch_grid,
        preprocess_image,
      },
      projector::{
        Lfm2VlMultiModalProjector, PixelUnshuffleBlock, merge_input_ids_with_image_features,
      },
      vision::{VisionModel, validate_active_grid},
    },
  },
};

/// The pixel-unshuffle stage — either the real [`PixelUnshuffleBlock`] (when
/// `downsample_factor > 1`) or an identity no-op (when `downsample_factor ==
/// 1`), mirroring `lfm2_vl.py:107-110`'s `pixel_unshuffle =
/// PixelUnshuffleBlock(...) if downsample_factor > 1 else nn.Identity()`.
#[cfg(feature = "lfm2-vl")]
#[derive(Debug)]
enum PixelUnshuffle {
  /// `downsample_factor > 1` — the real fold.
  Block(PixelUnshuffleBlock),
  /// `downsample_factor == 1` — a no-op (the reference's `nn.Identity()`).
  Identity,
}

#[cfg(feature = "lfm2-vl")]
impl PixelUnshuffle {
  /// `(N, W, H, C)` → `(N, W/f, H/f, C·f²)` (the block) or the input unchanged
  /// (identity).
  fn forward(&self, x: &Array) -> Result<Array> {
    match self {
      PixelUnshuffle::Block(b) => b.forward(x),
      PixelUnshuffle::Identity => x.try_clone(),
    }
  }
}

/// The LFM2.5-VL vision-language model (`lfm2_vl.py`'s `Model`).
///
/// Owns the vision tower, the pixel-unshuffle + projector, and the LFM2
/// language adapter. Construct from a model directory's parsed
/// [`ModelConfig`] + (sanitized) weight map via [`Lfm2Vl::from_weights`] (the
/// VLM-factory [`constructor`] wraps it); drive the multimodal forward with the
/// inherent [`get_input_embeddings`](Self::get_input_embeddings) /
/// [`forward_multimodal`](Self::forward_multimodal), or the cross-model
/// [`crate::vlm::model::Model`] trait.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct Lfm2Vl {
  config: ModelConfig,
  vision_tower: VisionModel,
  pixel_unshuffle: PixelUnshuffle,
  multi_modal_projector: Lfm2VlMultiModalProjector,
  language_model: LanguageModel,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl Lfm2Vl {
  /// Build the LFM2.5-VL model from a parsed [`ModelConfig`], the
  /// (already-[`sanitize`](Self::sanitize)d) weight map, and an optional parsed
  /// quantization config.
  ///
  /// The weight keys follow `lfm2_vl.py`'s post-sanitize module tree:
  /// `vision_tower.{embeddings,encoder,post_layernorm}.*`,
  /// `multi_modal_projector.{layer_norm,linear_1,linear_2}.*`, and
  /// `language_model.model.*` (the LFM2 LM tree). Each sub-builder drains its
  /// own prefix off `weights`.
  ///
  /// `quantization` resolves per-layer `(group_size, bits, mode)` from the
  /// config's `quantization` block (`None` ⇒ a dense checkpoint); a quantized
  /// layer is detected by its `<prefix>.scales` sibling, so a dense checkpoint
  /// loads unchanged whether or not a config is threaded.
  ///
  /// # Errors
  /// - propagates [`ModelConfig::validate`];
  /// - [`Error::MissingKey`] for an absent required weight;
  /// - propagates the vision / projector / LM sub-builder validation (including
  ///   the quantized-triple checks).
  pub fn from_weights(
    mut config: ModelConfig,
    mut weights: HashMap<String, Array>,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Re-apply `__post_init__`'s RoPE-base precedence (`lfm2.py:40-42`:
    // `text_config.rope_parameters["rope_theta"]` wins over the top-level
    // `text_config.rope_theta`) on the nested text config BEFORE it is validated,
    // handed to the LM builder, and stored. `TextConfig`'s `Deserialize` already
    // normalizes a freshly parsed config, so this is a no-op there; it is
    // corrective for a `ModelConfig` materialized then mutated post-deserialize
    // (the `text_config` field is public) into a stale `rope_parameters` /
    // `rope_theta` pair. Normalizing the STORED config here — not just the clone
    // passed to the LM builder below — keeps `config().text_config.rope_theta` in
    // agreement with the RoPE the built LM actually uses.
    config.text_config.apply_rope_parameters_override();
    config.validate()?;

    // Per-layer scheme resolver over the FULL VL module path namespace — the
    // same `quant_for` shape the LFM2 LM / vision tower use. `None` ⇒ that layer
    // is dense (no global default or an explicit per-layer `Skip`).
    let quant_for = |path: &str| -> Option<(i32, i32, &str)> {
      quantization
        .and_then(|q| q.quantization_for(path))
        .map(|q| (q.group_size, q.bits, q.mode.as_str()))
    };

    // ── vision tower (`vision_tower.*`) ──
    let layers_kept = config.vision_feature_layers_kept()?;
    let vision_weights_quant = |path: &str| quant_for(&format!("vision_tower.{path}"));
    let mut vision_weights = take_prefixed(&mut weights, "vision_tower.")?;
    VisionModel::sanitize(&mut vision_weights);
    let vision_tower = VisionModel::from_weights(
      &config.vision_config,
      layers_kept,
      &mut vision_weights,
      &vision_weights_quant,
    )?;
    reject_leftover(&vision_weights, "vision_tower")?;

    // ── pixel unshuffle (`downsample_factor > 1` ⇒ block, else identity) ──
    let pixel_unshuffle = if config.downsample_factor > 1 {
      PixelUnshuffle::Block(PixelUnshuffleBlock::new(config.downsample_factor)?)
    } else {
      PixelUnshuffle::Identity
    };

    // ── multimodal projector (`multi_modal_projector.*`) ──
    let proj_quant = |path: &str| quant_for(&format!("multi_modal_projector.{path}"));
    let mut proj_weights = take_prefixed(&mut weights, "multi_modal_projector.")?;
    let multi_modal_projector =
      Lfm2VlMultiModalProjector::from_weights(&config, &mut proj_weights, &proj_quant)?;
    reject_leftover(&proj_weights, "multi_modal_projector")?;

    // ── language model (`language_model.model.*` → the LFM2 LM `model.*`) ──
    // `lfm2_vl.py`'s VL sanitize maps `model.language_model` →
    // `language_model.model`; the LFM2 LM's own weight tree is `model.*`. Strip
    // the `language_model.` prefix so the LM sees its native `model.*` keys.
    let mut lm_weights = take_prefixed(&mut weights, "language_model.")?;
    // The conv-weight `(C, 1, K) → (C, K, 1)` sanitize is the LFM2 LM's own
    // (`language.py`'s `sanitize`); apply it before constructing the LM.
    Lfm2::sanitize(&mut lm_weights)?;
    // The LFM2 LM resolves its OWN per-layer scheme off the parsed
    // `PerLayerQuantization`. Its quantizable layers live in the `model.*`
    // namespace (post-prefix-strip), and the 8-bit checkpoint's quantization is
    // a uniform global default that applies to every quantizable LM Linear /
    // Embedding — so the same `PerLayerQuantization` resolves the LM exactly as
    // it does the LFM2 LM port's own loader. (A hypothetical per-layer override
    // keyed under the VL `language_model.` prefix would be an edge case the
    // checkpoint does not use; the global default — what the checkpoint carries
    // — applies regardless.)
    // `config.text_config` was already RoPE-normalized above, so this clone — and
    // the model the LM builder constructs from it — share the same effective
    // `rope_theta` as the `config` stored on `self` below. (The LM builder also
    // re-applies the override defensively; for this already-normalized clone that
    // is a no-op.)
    let lm = Lfm2::from_weights_quantized(config.text_config.clone(), lm_weights, quantization)?;
    let language_model = LanguageModel::new(lm);

    // Any weight not claimed by a sub-builder is a checkpoint/key mismatch — a
    // typed error rather than a silently-ignored tensor.
    reject_leftover(&weights, "lfm2_vl (top-level)")?;

    Ok(Self {
      config,
      vision_tower,
      pixel_unshuffle,
      multi_modal_projector,
      language_model,
    })
  }

  /// `lfm2_vl.py`'s `sanitize` (`:207-227`) — the VL-level key remap applied
  /// before construction:
  /// - vision keys: strip a leading `model.`, then
  ///   `vision_encoder → encoder`, `vision_embeddings → embeddings`,
  ///   `vision_post_layernorm → post_layernorm`;
  /// - `model.language_model → language_model.model`;
  /// - `model.multi_modal_projector → multi_modal_projector`.
  ///
  /// Operates on a name → [`Array`] map, returning the remapped map. The
  /// per-tower sanitizes (the vision `position_ids` drop, the LM `conv.weight`
  /// transpose) are applied by [`from_weights`](Self::from_weights) on the
  /// already-remapped sub-maps.
  pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut out: HashMap<String, Array> = HashMap::new();
    crate::model_validation::reserve_or_error(
      &mut out,
      "lfm2_vl sanitize: destination map",
      weights.len(),
    )?;
    for (k, v) in weights {
      let key = transform_key(&k);
      crate::model_validation::insert_unique(&mut out, key, v, "lfm2_vl sanitize")?;
    }
    Ok(out)
  }

  /// Read-only view of the parsed configuration.
  #[inline(always)]
  pub fn config(&self) -> &ModelConfig {
    &self.config
  }

  /// Build the heterogeneous per-layer LM cache (`lfm2_vl.py` delegates to the
  /// LM's `make_cache`).
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    self.language_model.make_cache()
  }

  /// Encode ONE image's NaFlex inputs into its projected `(N_i, D)` feature rows
  /// — the per-image loop body of `lfm2_vl.py`'s `get_input_embeddings`
  /// (`:141-153`).
  ///
  /// Runs the vision tower on the image's `pixel_values` `(1, num_patches,
  /// patch_feat)` + `spatial_shapes` `(1, 2)`, slices the active
  /// `H_p * W_p` patch rows (`pixel_attention_mask.sum()` is exactly
  /// `H_p * W_p`, carried losslessly in [`Lfm2VlImageInputs::grid`]), reshapes
  /// to the `(1, H_p, W_p, hidden)` grid, pixel-unshuffles, projects, and
  /// flattens to `(N_i, D)` where `N_i = ceil(H_p/f) * ceil(W_p/f)`.
  ///
  /// # Errors
  /// Propagates the vision / reshape / unshuffle / projector op errors; an
  /// active grid larger than the patch budget surfaces from the vision tower.
  pub fn encode_image_inputs(&self, inputs: &Lfm2VlImageInputs) -> Result<Array> {
    // The active grid `(H_p, W_p)` is read from `inputs.spatial_shapes` (the
    // single source of truth) via `grid()` — there is no separate host-int grid
    // field, so the value driving the active-row slice + the PixelUnshuffle
    // reshape below is the SAME one the vision tower derives its attention mask +
    // position resize from. A mask/slice disagreement is unrepresentable.
    let (h_p, w_p) = inputs.grid()?;
    // Add the leading per-image batch axis the vision tower expects:
    // pixel_values (1, num_patches, patch_feat), spatial_shapes (1, 2),
    // pixel_attention_mask (1, num_patches).
    let pv = expand_dims_axes(&inputs.pixel_values, &[0])?;
    let ss = expand_dims_axes(&inputs.spatial_shapes, &[0])?;
    let pam = expand_dims_axes(&inputs.pixel_attention_mask, &[0])?;
    // `vision_tower(pixel_values, spatial_shapes, pixel_attention_mask)` →
    // (1, num_patches, hidden). Passing `Some(&pam)` opts this native-resolution
    // input into key masking; the tower DERIVES the additive key mask from
    // `spatial_shapes` (the source of truth) rather than trusting the companion's
    // content, so it always matches the `H_p * W_p` active-row slice taken below
    // (a malformed companion cannot corrupt it). The mask excludes the padded
    // patch rows from every encoder layer's attention, so the active rows are
    // uncontaminated by the padding (the HF LFM2-VL reference threads
    // `pixel_attention_mask` into the vision tower).
    let hidden_states = self.vision_tower.forward(&pv, &ss, Some(&pam))?;
    // Slice the active `H_p * W_p` patch rows (the reference's
    // `feature[: img_feature_lengths[i], :]`). `spatial_shapes` is the
    // authoritative active count (`pixel_attention_mask.sum() == H_p * W_p`),
    // and the vision tower's mask above was derived from the same source.
    let hs_shape = hidden_states.shape();
    let num_patches = dim_i32(&hs_shape, 1, "lfm2_vl encode_image: num_patches")?;
    let hidden = dim_i32(&hs_shape, 2, "lfm2_vl encode_image: hidden")?;
    // The SINGLE active-grid validation point (shared with the vision tower's
    // mask + position resize): positive dims + `H_p * W_p <= num_patches`,
    // returning the active patch count. Reusing it (over an inline multiply)
    // guarantees the slice bound matches the tower's masked active prefix, and
    // rejects an out-of-budget grid before the slice indexes out of bounds.
    let active = validate_active_grid(h_p, w_p, num_patches)?;
    // (1, num_patches, hidden) → (num_patches, hidden) → [: active, :]. mlxrs's
    // `reshape` validates dims (no `-1` inference), so the explicit dims are
    // computed from the hidden-state shape.
    let hs2 = reshape(&hidden_states, &[num_patches, hidden])?;
    let active_rows = ops::indexing::slice(&hs2, &[0, 0], &[active, hidden], &[1, 1])?;
    // `feature.reshape(1, H_p, W_p, -1)`.
    let grid = reshape(&active_rows, &[1, h_p, w_p, hidden])?;
    // pixel-unshuffle → (1, H_p/f, W_p/f, hidden*f^2).
    let unshuffled = self.pixel_unshuffle.forward(&grid)?;
    // projector → (1, H_p/f, W_p/f, D).
    let projected = self.multi_modal_projector.forward(&unshuffled)?;
    // `img_embedding.reshape(-1, D)` → (N_i, D). The flattened row count is the
    // product of the projected tensor's leading axes (`N * H_p/f * W_p/f`,
    // computed explicitly since `reshape` takes no `-1`).
    let proj_shape = projected.shape();
    let d = dim_i32(&proj_shape, 3, "lfm2_vl encode_image: projector out width")?;
    let rows = leading_product(&proj_shape, "lfm2_vl encode_image: projected rows")?;
    reshape(&projected, &[rows, d])
  }

  /// `lfm2_vl.py`'s `get_input_embeddings` (`:115-160`): embed the prompt token
  /// ids, then (if images are present) encode each image, concatenate the
  /// projected features `(ΣN, D)`, and splice them into the embeddings at the
  /// `<image>`-token positions (the mask-driven `masked_scatter`).
  ///
  /// `input_ids` is the `(B, T)` integer token ids (the `<image>` placeholders
  /// already expanded to the per-image token runs by
  /// [`expand_image_tokens`](super::processor::expand_image_tokens)).
  /// `images` is the per-image [`Lfm2VlImageInputs`] (empty ⇒ the text-only
  /// path, which returns `embed_tokens(input_ids)` unchanged — the reference's
  /// `if pixel_values is None`).
  ///
  /// Returns the merged `(B, T, D)` input embeddings ready for the LM.
  ///
  /// # Errors
  /// - [`Error::LengthMismatch`] if the spliced feature row count disagrees with
  ///   the `<image>`-token count (the [`merge_input_ids_with_image_features`]
  ///   count contract);
  /// - propagates the embed / vision / projector / merge op errors.
  pub fn get_input_embeddings(
    &self,
    input_ids: &Array,
    images: &[Lfm2VlImageInputs],
  ) -> Result<Array> {
    let inputs_embeds = self.language_model.embed_tokens(input_ids)?;
    if images.is_empty() {
      // `if pixel_values is None: return inputs_embeds`.
      return Ok(inputs_embeds);
    }

    // Encode each image and concatenate the projected feature rows along axis 0
    // (`image_features = mx.concatenate(image_features, axis=0)`).
    let mut feats: Vec<Array> = Vec::new();
    crate::model_validation::reserve_or_error(
      &mut feats,
      "lfm2_vl get_input_embeddings: per-image features",
      images.len(),
    )?;
    for img in images {
      feats.push(self.encode_image_inputs(img)?);
    }
    let refs: Vec<&Array> = feats.iter().collect();
    let image_features = concatenate(&refs, 0)?;

    // `merge_input_ids_with_image_features` — the mask-driven splice + count
    // check (`lfm2_vl.py:157-160` / `:162-182`).
    merge_input_ids_with_image_features(
      &image_features,
      &inputs_embeds,
      input_ids,
      self.config.image_token_index,
    )
  }

  /// Build a tiling-enabled [`Lfm2VlProcessorConfig`] from this model's
  /// [`ModelConfig`] — the SigLIP NaFlex knobs plus the HF
  /// `Lfm2VlImageProcessor` image-splitting parameters
  /// (`do_image_splitting`, `min_tiles`, `max_tiles`, `use_thumbnail`,
  /// `min_image_tokens`, `max_image_tokens`, `encoder_patch_size`, `tile_size`,
  /// `max_pixels_tolerance`) carried on [`ModelConfig`]. Drives the tiled
  /// [`split_image`](Self::split_image) path; the SigLIP2 native-resolution
  /// [`crate::vlm::models::lfm2_vl::preprocess_image`] path ignores the tiling
  /// knobs.
  ///
  /// # Errors
  /// Propagates [`Lfm2VlProcessorConfig::new`] / `with_tiling` validation (a
  /// non-positive dimension, an inverted tile / token band, a non-finite
  /// tolerance, or a `tile_size` not divisible by `encoder_patch_size`).
  pub fn processor_config(&self) -> Result<Lfm2VlProcessorConfig> {
    let c = &self.config;
    let factor = c.downsample_factor.max(1) as u32;
    let patch_size = c.vision_config.patch_size.max(1) as u32;
    let max_num_patches = c.max_num_patches.max(1) as u32;
    Lfm2VlProcessorConfig::new(c.image_token_index, factor, patch_size, max_num_patches)?
      .with_tiling(
        c.do_image_splitting,
        c.min_tiles.max(1) as u32,
        c.max_tiles.max(1) as u32,
        c.use_thumbnail,
        c.min_image_tokens.max(1) as u32,
        c.max_image_tokens.max(1) as u32,
        c.encoder_patch_size.max(1) as u32,
        c.tile_size.max(1) as u32,
        c.max_pixels_tolerance,
      )
  }

  /// Split ONE decoded interleaved-RGB image into its HF-tiling sub-image
  /// NaFlex inputs — the
  /// [`tile_image`](crate::vlm::models::lfm2_vl::tile_image) port, gated on
  /// [`do_image_splitting`](ModelConfig::do_image_splitting). When splitting is
  /// enabled and the image is over the size threshold this returns one
  /// [`Lfm2VlImageInputs`] per tile (+ an optional thumbnail); otherwise a
  /// single native-resolution sub-image. The returned sub-images flatten
  /// directly into the `images` slice
  /// [`get_input_embeddings`](Self::get_input_embeddings) consumes, and their
  /// per-tile grids ([`Lfm2VlImageInputs::grid`]) drive
  /// [`expand_image_tokens`](crate::vlm::models::lfm2_vl::expand_image_tokens)
  /// for the matching `<image>`-token run (each sub-image bracketed +
  /// concatenated).
  ///
  /// `rgb` is `width * height * 3` row-major interleaved RGB bytes. The
  /// processor config is built by [`processor_config`](Self::processor_config).
  ///
  /// # Errors
  /// Propagates [`processor_config`](Self::processor_config) and
  /// [`tile_image`](crate::vlm::models::lfm2_vl::tile_image) (the grid math,
  /// resize, patchify, and budget checks).
  pub fn split_image(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<Lfm2VlImageInputs>> {
    let cfg = self.processor_config()?;
    crate::vlm::models::lfm2_vl::processor::tile_image(rgb, width, height, &cfg)
  }

  /// `lfm2_vl.py`'s `__call__` (`:188-205`): the full multimodal forward —
  /// [`get_input_embeddings`](Self::get_input_embeddings) then the LFM2 LM over
  /// the merged embeddings → `(B, T, vocab_size)` logits.
  ///
  /// `cache` is the heterogeneous per-layer LM cache (from
  /// [`make_cache`](Self::make_cache)), mutated in place.
  ///
  /// # Errors
  /// Propagates [`get_input_embeddings`](Self::get_input_embeddings) and the LM
  /// forward errors (including the per-layer cache count check).
  pub fn forward_multimodal(
    &self,
    input_ids: &Array,
    images: &[Lfm2VlImageInputs],
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    let embeds = self.get_input_embeddings(input_ids, images)?;
    self.language_model.forward_embeddings(&embeds, cache)
  }
}

// ───────────────────────── lm::model::Model ─────────────────────────

#[cfg(feature = "lfm2-vl")]
impl LmModel for Lfm2Vl {
  /// The text-only forward (no images) — `language.py`'s `__call__` without
  /// `inputs_embeds`. Delegates to the LFM2 LM through the language adapter.
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    self.language_model.forward(tokens, cache)
  }

  /// Run the LM over pre-computed (merged) input embeddings — the
  /// `language_model(..., inputs_embeds=...)` path. Used by
  /// [`forward_multimodal`](Lfm2Vl::forward_multimodal) and the cross-model VLM
  /// generate loop.
  fn forward_embeddings(
    &self,
    embeddings: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    self.language_model.forward_embeddings(embeddings, cache)
  }

  fn supports_input_embeddings(&self) -> bool {
    true
  }
}

// ───────────────────────── vlm::model::Model ─────────────────────────

#[cfg(feature = "lfm2-vl")]
impl VlmModel for Lfm2Vl {
  /// Token-id → text-embedding lookup (`language_model.model.embed_tokens`),
  /// range-guarded by the language adapter.
  fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
    self.language_model.embed_tokens(tokens)
  }

  /// Encode one image's NaFlex inputs into its projected `(N_i, D)` feature
  /// rows — the cross-model [`crate::vlm::model::Model::encode_image`] entry,
  /// consuming the [`ProcessedImage`] the [`Lfm2VlImageProcessor`] produced.
  ///
  /// The native-resolution NaFlex path is inherently a function of
  /// `(pixel_values, spatial_shapes)` per image, so this reads
  /// [`image.pixels()`](ProcessedImage::pixels) (the flattened patch tensor
  /// `(max_num_patches, patch_feat)`) AND the required
  /// [`image.native()`](ProcessedImage::native) companions
  /// (`spatial_shape = [H_p, W_p]`, `patch_mask`), reconstructs the per-image
  /// [`Lfm2VlImageInputs`] triple, and runs the inherent
  /// [`Lfm2Vl::encode_image_inputs`] body (`lfm2_vl.py:141-153`). The active
  /// grid `(H_p, W_p)` is carried solely by the `spatial_shape` companion (the
  /// single source of truth); [`encode_image_inputs`](Lfm2Vl::encode_image_inputs)
  /// reads + validates it from there, so no separate grid is threaded.
  ///
  /// # Errors
  /// - [`Error::InvariantViolation`] if `image.native()` is `None` (a
  ///   native-resolution model requires the NaFlex companions; a fixed-grid
  ///   [`ProcessedImage`] is rejected here, never a panic);
  /// - [`Error::RankMismatch`] if the `spatial_shape` companion is not a `(2,)`
  ///   array of `[H_p, W_p]`, and [`Error::OutOfRange`] if its active grid
  ///   `H_p * W_p` exceeds the patch budget — both surfaced from
  ///   [`encode_image_inputs`](Lfm2Vl::encode_image_inputs) (the grid read +
  ///   the shared active-grid validation);
  /// - propagates the vision / reshape / unshuffle / projector op errors.
  fn encode_image(&self, image: &ProcessedImage) -> Result<Array> {
    let native = image.native().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "lfm2_vl encode_image: ProcessedImage::native (the NaFlex companions)",
        "LFM2.5-VL requires the native-resolution spatial_shape + patch_mask companions; this \
         ProcessedImage carries none (a fixed-grid input) — use the Lfm2VlImageProcessor",
      ))
    })?;
    // Reconstruct the NaFlex triple for the inherent per-image body. The
    // pixel/patch tensor + mask + spatial_shape are shared (refcount-only
    // `try_clone`). The active grid is read from `spatial_shape` alone by
    // `encode_image_inputs` (which validates the `(2,)` shape), so there is no
    // separate grid to thread — a disagreeing grid cannot be constructed.
    let inputs = Lfm2VlImageInputs::from_parts(
      image.pixels().try_clone()?,
      native.patch_mask().try_clone()?,
      native.spatial_shape().try_clone()?,
    );
    self.encode_image_inputs(&inputs)
  }

  /// The image-processor config: SigLIP NaFlex defaults — `image_mean =
  /// image_std = 0.5` (the `x/127.5 - 1.0` rescale), the vision patch size,
  /// Bilinear resample (PIL `Image.BILINEAR`), channel-last layout. The
  /// native-resolution patch grid is computed by the [processor](super::processor)
  /// per image (not the fixed `size` field), so the returned config's `size` is
  /// the nominal vision `image_size` and is not load-bearing for the NaFlex
  /// path.
  fn image_processor_config(&self) -> ImageProcessorConfig {
    let p = self.config.vision_config.patch_size.max(1) as u32;
    let nominal = (self.config.vision_config.image_size.max(p as i32)) as u32;
    ImageProcessorConfig::new()
      .with_size((nominal, nominal))
      .with_mean([0.5, 0.5, 0.5])
      .with_std([0.5, 0.5, 0.5])
      .with_rescale_factor(1.0 / 255.0)
      .with_resample(crate::vlm::image::ResizeFilter::Bilinear)
      .with_layout(Layout::Hwc)
  }

  /// The LFM2.5-VL native-resolution [`ImageProcessor`] — a
  /// [`Lfm2VlImageProcessor`] wrapping the [SigLIP2 NaFlex
  /// processor](super::processor::preprocess_image). Overrides the default
  /// fixed-grid processor: each image is smart-resized to its native patch
  /// grid within the budget, and its per-image token count
  /// (`ceil(H_p / f) * ceil(W_p / f)`) is reported on the
  /// [`ProcessedImage`] for the prompt's placeholder run. The `num_tokens`
  /// argument the cross-model surface passes for the fixed-grid default is
  /// unused (the count is per-image and computed by the processor).
  ///
  /// # Panics
  /// Does not panic. The processor config is built from the already-validated
  /// [`ModelConfig`], whose `image_token_index` / `downsample_factor` /
  /// `patch_size` / `max_num_patches` cleared
  /// [`Lfm2VlProcessorConfig::new`]'s positivity checks at construction; an
  /// unexpected degenerate config surfaces as a typed error from
  /// [`process`](ImageProcessor::process), not here.
  fn image_processor(&self, _num_tokens: usize) -> Box<dyn ImageProcessor> {
    Box::new(Lfm2VlImageProcessor::new(self.config()))
  }
}

// ───────────────────────── native-resolution processor ─────────────────────────

/// The LFM2.5-VL native-resolution [`ImageProcessor`] — wraps the SigLIP2
/// NaFlex [`preprocess_image`] so the cross-model
/// [`crate::vlm::generate::vlm_generate`] loop drives LFM2.5-VL end-to-end.
///
/// [`Model::image_processor`](crate::vlm::model::Model::image_processor) on
/// [`Lfm2Vl`] returns this in place of the default fixed-grid processor. Its
/// [`process`](ImageProcessor::process) smart-resizes each decoded image to its
/// native patch grid, normalizes + patchifies into the
/// `(max_num_patches, patch_feat)` flattened patch tensor, and returns a
/// [`ProcessedImage`] carrying that tensor as `pixels`, the `spatial_shape` +
/// `patch_mask` as the [`NativeResolution`] companions, and the per-image
/// `ceil(H_p / f) * ceil(W_p / f)` image-token count.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct Lfm2VlImageProcessor {
  cfg: Lfm2VlProcessorConfig,
  downsample_factor: i32,
}

#[cfg(feature = "lfm2-vl")]
impl Lfm2VlImageProcessor {
  /// Build the processor from the model's parsed [`ModelConfig`] — the
  /// `image_token_index` / `downsample_factor` / vision `patch_size` /
  /// `max_num_patches` drive the SigLIP2 NaFlex
  /// [`Lfm2VlProcessorConfig`], with the SigLIP `mean = std = 0.5` defaults.
  ///
  /// On the (already-validated) config the `Lfm2VlProcessorConfig::new`
  /// positivity checks always pass; a degenerate config is clamped to the
  /// SigLIP defaults so construction is infallible. Any genuine preprocessing
  /// failure surfaces from [`process`](ImageProcessor::process).
  pub fn new(config: &ModelConfig) -> Self {
    let factor = config.downsample_factor.max(1);
    let patch_size = config.vision_config.patch_size.max(1) as u32;
    let max_num_patches = config.max_num_patches.max(1) as u32;
    // `Lfm2VlProcessorConfig::new` only fails on a non-positive
    // patch_size / max_num_patches / downsample_factor or a negative
    // image_token; all are clamped positive above and the token index is a
    // valid id on a constructed model. Fall back to the SigLIP defaults if a
    // hand-built config ever violates that, so the processor is infallible to
    // build (preprocessing errors surface per-image from `process`).
    let cfg = Lfm2VlProcessorConfig::new(
      config.image_token_index,
      factor as u32,
      patch_size,
      max_num_patches,
    )
    .unwrap_or_else(|_| {
      Lfm2VlProcessorConfig::new(0, 1, 1, 1).expect("SigLIP default processor config is valid")
    });
    Self {
      cfg,
      downsample_factor: factor,
    }
  }
}

#[cfg(feature = "lfm2-vl")]
impl ImageProcessor for Lfm2VlImageProcessor {
  /// Smart-resize → normalize → patchify the decoded image into its NaFlex
  /// triple, then package it as a [`ProcessedImage`] with the per-image token
  /// count.
  ///
  /// Extracts the decoded image's interleaved RGB bytes + dimensions (the
  /// SigLIP2 slow processor's `np.array(image)` input), runs
  /// [`preprocess_image`] (the shared SigLIP2 NaFlex patchify), reads the
  /// resulting patch grid `(H_p, W_p)`, and reports
  /// `ceil(H_p / f) * ceil(W_p / f)` image tokens — the
  /// [`num_image_tokens_from_patch_grid`] count that mirrors the
  /// [`PixelUnshuffleBlock`] pad.
  ///
  /// # Errors
  /// Propagates [`preprocess_image`] (resize / patchify / overflow /
  /// allocation) and [`num_image_tokens_from_patch_grid`] errors as typed
  /// [`crate::Error`]s.
  fn process(&self, image: &::image::DynamicImage) -> Result<ProcessedImage> {
    // The SigLIP2 slow processor operates on the decoded RGB image
    // (`np.array(image.convert("RGB"))`) — an interleaved `width * height * 3`
    // RGB buffer (alpha dropped, RGB-ordered). Use the FALLIBLE
    // [`crate::vlm::image::decode_rgb`] (a borrowed `as_rgb8` fast path, else a
    // per-pixel projection, both `try_reserve_exact`-backed) rather than
    // `to_rgb8()`, whose infallible global allocation can `abort()` under
    // memory pressure / a near-cap decoded image — surfacing allocator failure
    // as a typed [`Error::OutOfMemory`] through this seam's `Result`.
    let (rgb, width, height) = crate::vlm::image::decode_rgb(image)?;
    let inputs = preprocess_image(&rgb, width, height, &self.cfg)?;
    // The grid `(H_p, W_p)` is read from the produced `spatial_shapes` (the
    // single source of truth) — the same value the vision tower + the active-row
    // slice consume downstream, so the reported image-token count cannot
    // disagree with them.
    let (h_p, w_p) = inputs.grid()?;
    let num_tokens = num_image_tokens_from_patch_grid(h_p, w_p, self.downsample_factor)? as usize;
    let Lfm2VlImageInputs {
      pixel_values,
      pixel_attention_mask,
      spatial_shapes,
    } = inputs;
    Ok(ProcessedImage::new(
      pixel_values,
      Some(NativeResolution::new(spatial_shapes, pixel_attention_mask)),
      num_tokens,
    ))
  }
}

// ───────────────────────── factory registration ─────────────────────────

/// The VLM-factory [`constructor`] for LFM2.5-VL — assembles a boxed
/// [`Lfm2Vl`] from an already-resolved [`LoadedVlmModel`] (parsed base config +
/// raw `config.json` + weights).
///
/// Parses the full [`ModelConfig`] off the verbatim `config.json` (the typed
/// [`crate::vlm::load::VlmBaseConfig`] only carries the dispatch subset),
/// resolves the optional quantization config (accepting both `quantization` and
/// the HF `quantization_config` key — the LFM2 LM's
/// [`resolve_quantization`]), [`sanitize`](Lfm2Vl::sanitize)s the weights, and
/// constructs the model.
///
/// Mirrors the SigLIP2 dual-tower embeddings constructor shape
/// ([`crate::embeddings::siglip2_naflex::constructor`]) and the LFM2 LM loader.
///
/// # Errors
/// - propagates [`ModelConfig::from_json`] / [`ModelConfig::validate`];
/// - propagates [`resolve_quantization`];
/// - propagates [`Lfm2Vl::from_weights`].
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn constructor() -> VlmModelConstructor {
  Box::new(|loaded: &LoadedVlmModel| -> Result<Box<dyn VlmModel>> {
    let config = ModelConfig::from_json(loaded.config_json_ref())?;
    let quantization = resolve_quantization(loaded.config_json_ref())?;
    // The constructor receives the weights by reference (no implicit eval). mlx
    // `Array` is a cheap refcounted handle; `try_clone` shares the device buffer
    // (no copy). Clone the loaded map into an owned, sanitizable + drainable map,
    // reserving FALLIBLY (typed `AllocFailure`) since the key count is
    // checkpoint-controlled.
    let mut raw: HashMap<String, Array> = HashMap::new();
    crate::model_validation::reserve_or_error(
      &mut raw,
      "lfm2_vl constructor: sanitizable weight-map clone",
      loaded.weights_ref().len(),
    )?;
    for (k, v) in loaded.weights_ref() {
      raw.insert(k.clone(), v.try_clone()?);
    }
    let weights = Lfm2Vl::sanitize(raw)?;
    let model = Lfm2Vl::from_weights(config, weights, quantization.as_ref())?;
    Ok(Box::new(model))
  })
}

/// Register the LFM2.5-VL constructor into a [`VlmTypeRegistry`] under its
/// canonical `model_type` (`"lfm2_vl"`; the `config.json` `"lfm2-vl"` alias is
/// canonicalized by [`crate::vlm::load::remap_vlm_model_type`] on both
/// registration and lookup). Returns any displaced constructor
/// (last-writer-wins).
///
/// Mirrors [`crate::embeddings::siglip2_naflex::register`].
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
pub fn register(registry: &mut VlmTypeRegistry) -> Option<VlmModelConstructor> {
  registry.register("lfm2-vl", constructor())
}

// ───────────────────────── helpers ─────────────────────────

/// `lfm2_vl.py`'s `sanitize.transform_key` (`:208-225`): the VL-level key remap.
#[cfg(feature = "lfm2-vl")]
fn transform_key(key: &str) -> String {
  let mut k = key.to_string();
  if k.contains("vision_tower") {
    k = k
      .replace("model.", "")
      .replace("vision_encoder", "encoder")
      .replace("vision_embeddings", "embeddings")
      .replace("vision_post_layernorm", "post_layernorm");
  }
  if k.contains("language_model") {
    k = k.replace("model.language_model", "language_model.model");
  }
  if k.contains("multi_modal_projector") {
    k = k.replace("model.multi_modal_projector", "multi_modal_projector");
  }
  k
}

/// Drain every `weights` entry whose key starts with `prefix`, returning a new
/// map with the prefix stripped (so a sub-builder sees its native keys). The
/// drained entries are removed from `weights`.
#[cfg(feature = "lfm2-vl")]
fn take_prefixed(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
) -> Result<HashMap<String, Array>> {
  let keys: Vec<String> = weights
    .keys()
    .filter(|k| k.starts_with(prefix))
    .cloned()
    .collect();
  let mut out: HashMap<String, Array> = HashMap::new();
  crate::model_validation::reserve_or_error(&mut out, "lfm2_vl take_prefixed", keys.len())?;
  for k in keys {
    let v = weights.remove(&k).expect("key just enumerated");
    let stripped = k
      .strip_prefix(prefix)
      .expect("key starts_with prefix")
      .to_string();
    crate::model_validation::insert_unique(&mut out, stripped, v, "lfm2_vl take_prefixed")?;
  }
  Ok(out)
}

/// Reject a non-empty leftover weight map — a tensor the sub-builder did not
/// claim (a checkpoint / key mismatch) is a typed error rather than a silently
/// ignored weight.
#[cfg(feature = "lfm2-vl")]
fn reject_leftover(weights: &HashMap<String, Array>, context: &'static str) -> Result<()> {
  if !weights.is_empty() {
    // Name one leftover key (deterministic: the lexicographically smallest) so
    // the error is reproducible.
    let mut keys: Vec<&String> = weights.keys().collect();
    keys.sort();
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "lfm2_vl: unconsumed checkpoint weight(s) after construction",
      match context {
        "vision_tower" => {
          "vision_tower has leftover weights (unexpected key / wrong feature layer)"
        }
        "multi_modal_projector" => {
          "multi_modal_projector has leftover weights (unexpected projector key)"
        }
        _ => "top-level has leftover weights (unexpected key / unmapped prefix)",
      },
    )));
  }
  Ok(())
}

/// The product of every axis of `shape` except the last, as a checked `i32` —
/// the explicit row count for a `(.., D) → (rows, D)` flatten (mlxrs's `reshape`
/// takes no `-1` inference). Requires rank `>= 2`.
#[cfg(feature = "lfm2-vl")]
fn leading_product(shape: &[usize], context: &'static str) -> Result<i32> {
  if shape.len() < 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    )));
  }
  let mut rows: i32 = 1;
  for (axis, _) in shape.iter().enumerate().take(shape.len() - 1) {
    let d = dim_i32(shape, axis, context)?;
    rows = crate::model_validation::checked_mul(context, "rows", rows, "dim", d)?;
  }
  Ok(rows)
}

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "lfm2-vl")]
fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(crate::error::OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      smol_str::format_smolstr!("{d}"),
    ))
  })
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
