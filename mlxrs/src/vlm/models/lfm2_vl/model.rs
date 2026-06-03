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
    model::Model as VlmModel,
    models::lfm2_vl::{
      config::ModelConfig,
      language::LanguageModel,
      processor::Lfm2VlImageInputs,
      projector::{
        Lfm2VlMultiModalProjector, PixelUnshuffleBlock, merge_input_ids_with_image_features,
      },
      vision::VisionModel,
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
    config: ModelConfig,
    mut weights: HashMap<String, Array>,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
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
    let (h_p, w_p) = inputs.grid();
    // Add the leading per-image batch axis the vision tower expects:
    // pixel_values (1, num_patches, patch_feat), spatial_shapes (1, 2).
    let pv = expand_dims_axes(&inputs.pixel_values, &[0])?;
    let ss = expand_dims_axes(&inputs.spatial_shapes, &[0])?;
    // `vision_tower(pixel_values, spatial_shapes)` → (1, num_patches, hidden).
    let hidden_states = self.vision_tower.forward(&pv, &ss)?;
    // Slice the active `H_p * W_p` patch rows (the reference's
    // `feature[: img_feature_lengths[i], :]`). The grid is the authoritative
    // active count (`pixel_attention_mask.sum() == H_p * W_p`).
    let hs_shape = hidden_states.shape();
    let num_patches = dim_i32(&hs_shape, 1, "lfm2_vl encode_image: num_patches")?;
    let hidden = dim_i32(&hs_shape, 2, "lfm2_vl encode_image: hidden")?;
    let active = crate::model_validation::checked_mul(
      "lfm2_vl encode_image: active patches (H_p * W_p)",
      "H_p",
      h_p,
      "W_p",
      w_p,
    )?;
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

  /// Encode one image's NaFlex patch rows into its projected `(N_i, D)` feature
  /// rows.
  ///
  /// **LFM2.5-VL caveat.** The native-resolution NaFlex path is inherently a
  /// function of `(pixel_values, spatial_shapes)` per image and is driven
  /// through the inherent
  /// [`Lfm2Vl::encode_image_inputs`] / [`Lfm2Vl::get_input_embeddings`]
  /// (which carry the per-image [`Lfm2VlImageInputs`] triple). This cross-model
  /// trait entry takes a single `Array` and therefore only realizes the
  /// **square fully-active** sub-case: `image` is the flattened patches
  /// `(1, num_patches, patch_feat)` of an image whose active grid is the full
  /// square `sqrt(num_patches) x sqrt(num_patches)` (i.e. the whole patch
  /// budget is active, the common square-image case). The grid is reconstructed
  /// as that square and the per-image body runs. A non-square / partially-active
  /// image must use [`Lfm2Vl::encode_image_inputs`] with its real
  /// `spatial_shapes`; here a `num_patches` that is not a perfect square is a
  /// typed [`Error::InvariantViolation`] directing the caller to the native
  /// method.
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `image` is not rank-3 `(1, num_patches,
  ///   patch_feat)`;
  /// - [`Error::InvariantViolation`] if `num_patches` is not a perfect square
  ///   (use [`Lfm2Vl::encode_image_inputs`] for the general grid);
  /// - propagates the vision / unshuffle / projector op errors.
  fn encode_image(&self, image: &Array) -> Result<Array> {
    let shape = image.shape();
    if shape.len() != 3 || shape[0] != 1 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl encode_image: image must be rank-3 (1, num_patches, patch_feat) — for the \
           general native-resolution path use Lfm2Vl::encode_image_inputs with spatial_shapes",
        shape.len() as u32,
        shape,
      )));
    }
    let num_patches = shape[1];
    let side = (num_patches as f64).sqrt().round() as usize;
    if side.saturating_mul(side) != num_patches {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "lfm2_vl encode_image: num_patches is not a perfect square (the cross-model single-Array \
           entry only realizes the square fully-active grid)",
        "use Lfm2Vl::encode_image_inputs with the image's real spatial_shapes for a non-square \
           or partially-active grid",
      )));
    }
    // Reconstruct the square fully-active NaFlex inputs for this image: a mask
    // of all-ones over the `num_patches` rows, `spatial_shapes = [side, side]`.
    let side_i = side as i32;
    let pv = reshape(
      image,
      &[num_patches as i32, dim_i32(&shape, 2, "patch_feat")?],
    )?;
    let spatial = Array::from_slice::<i32>(&[side_i, side_i], &(2usize,))?;
    let mask = Array::from_slice::<i32>(&vec![1i32; num_patches], &(num_patches,))?;
    let inputs = Lfm2VlImageInputs::from_parts(pv, mask, spatial, side_i, side_i);
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
