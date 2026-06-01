//! SigLIP2 NaFlex — a standalone dual-tower image+text **embeddings**
//! model (`google/siglip2-base-patch16-naflex`).
//!
//! Ported from
//! [`mlx-embeddings`'s `models/siglip.py`](https://github.com/Blaizzy/mlx-embeddings/blob/main/mlx_embeddings/models/siglip.py),
//! with native-resolution (NaFlex) position-embedding interpolation —
//! the piece `siglip.py` leaves as `NotImplementedError` — added here.
//! The NaFlex preprocessing + sizing is pinned against the user's
//! PyTorch-validated `siglip2-naflex` crate.
//!
//! This is a self-contained embeddings model under
//! [`crate::embeddings`]; it does **not** depend on the LFM2.5-VL vision
//! tower. The only shared low-level primitive is
//! [`crate::ops::interpolation::bicubic_interpolate`] (the per-image
//! position-embedding resize), which lives in [`crate::ops`] precisely so
//! it can be reused by an independent vision port without coupling the
//! model code.
//!
//! ## Modules
//!
//! - [`config`] — the [`config::TextConfig`] / [`config::VisionConfig`] /
//!   [`config::Siglip2NaflexConfig`] dataclasses (serde parse +
//!   architecture-pinning validation).
//! - [`processing`] — the NaFlex image preprocessing
//!   ([`processing::patch_grid`] sizing + aspect-preserving resize +
//!   normalize/patchify into the flat `(num_patches, P^2 * C)` tensor,
//!   plus `spatial_shapes` + `pixel_attention_mask`).
//! - [`vision`] — the NaFlex ViT ([`vision::VisionTower`]): patch-embed
//!   (Conv2d-as-Linear) + per-image bicubic position resize + masked
//!   pre-norm encoder + optional attention-pool head.
//! - [`text`] — the text tower ([`text::TextTower`]): token + position
//!   embedding + pre-norm encoder + final LayerNorm + sticky-EOS pooled
//!   projection.
//! - `shared` (private) — the `Linear` / `MLP` / `Attention` /
//!   `EncoderLayer` blocks both towers compose, plus the weight-fetch +
//!   shape-pinning helpers.
//!
//! ## Public surface ([`Siglip2NaflexModel`])
//!
//! Mirrors the [`crate::embeddings::colvision`] surface: a struct built via
//! [`Siglip2NaflexModel::from_weights`] (sanitize + shape-pinned load) with
//! [`encode_image`](Siglip2NaflexModel::encode_image) /
//! [`encode_text`](Siglip2NaflexModel::encode_text) producing L2-normalized
//! embeddings and [`logits`](Siglip2NaflexModel::logits) computing the
//! `logit_scale` / `logit_bias` contrastive similarity. The model also
//! implements [`crate::embeddings::EmbeddingModel`] (its `forward` runs the
//! text tower) so it registers into the
//! [`crate::embeddings::EmbeddingModelTypeRegistry`] via [`register`].

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod config;

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod processing;

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod text;

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub mod vision;

#[cfg(feature = "siglip2-naflex")]
mod shared;

#[cfg(feature = "siglip2-naflex")]
use std::collections::HashMap;

#[cfg(feature = "siglip2-naflex")]
use crate::{
  array::Array,
  embeddings::{
    EmbeddingModel, EmbeddingModelConstructor, EmbeddingModelOutput, EmbeddingModelTypeRegistry,
    LoadedEmbeddingModel, SWIFT_L2_EPS, l2_normalize_eps,
    siglip2_naflex::{
      config::Siglip2NaflexConfig, processing::NaflexInputs, shared::take_shaped, text::TextTower,
      vision::VisionTower,
    },
  },
  error::{Error, InvariantViolationPayload, Result},
  ops,
};

/// The top-level architecture id this model registers under.
#[cfg(feature = "siglip2-naflex")]
pub const MODEL_TYPE: &str = "siglip";

/// SigLIP2 NaFlex dual-tower embeddings model
/// (`google/siglip2-base-patch16-naflex`).
///
/// See the [module docs](self) for the architecture and public API. Built via
/// [`from_weights`](Self::from_weights); encode with
/// [`encode_image`](Self::encode_image) / [`encode_text`](Self::encode_text)
/// and score with [`logits`](Self::logits).
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub struct Siglip2NaflexModel {
  config: Siglip2NaflexConfig,
  vision: VisionTower,
  text: TextTower,
  /// `logit_scale` `(1,)` — the contrastive temperature (`exp`-d at use).
  logit_scale: Array,
  /// `logit_bias` `(1,)` — the contrastive bias.
  logit_bias: Array,
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl Siglip2NaflexModel {
  /// L2-normalization eps for the image/text embeddings. SigLIP normalizes
  /// with the swift `1e-12` floor (`MLXEmbedders` `l2Normalized`); the
  /// embeddings are f32 here so the choice is immaterial to the result, but
  /// it pins the intent.
  const NORMALIZE_EPS: f32 = SWIFT_L2_EPS;

  /// Build a model from a parsed [`Siglip2NaflexConfig`] and the **sanitized**
  /// weight map (run [`sanitize`] first).
  ///
  /// The dual-tower (contrastive) arm is the only supported path —
  /// `config.num_labels` must be `0` (a classifier checkpoint is rejected).
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions before it is stored or fed to any op (typed
  /// [`Error::ShapePairMismatch`] wrapped in [`Error::LayerKeyed`]), exactly
  /// like the merged Wav2Vec2 / LFM2 ports.
  pub fn from_weights(
    config: Siglip2NaflexConfig,
    mut weights: HashMap<String, Array>,
  ) -> Result<Self> {
    config.validate()?;
    if config.num_labels != 0 {
      // `siglip.py`'s `num_labels > 0` builds a vision classifier head
      // instead of the dual-tower contrastive path; this embeddings port
      // targets the `0` (contrastive) arm only.
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Siglip2NaflexConfig: num_labels",
        "must be 0 — the classifier-head arm (num_labels > 0) is not ported (embeddings target the contrastive dual-tower path)",
      )));
    }

    // Strip the tower-namespace prefixes the sanitized map carries
    // (`vision_model.vision_model.` / `text_model.text_model.`) and build each
    // tower from its own sub-map, then read the top-level contrastive params.
    let mut vision_weights = strip_prefix(&mut weights, "vision_model.vision_model.");
    let mut text_weights = strip_prefix(&mut weights, "text_model.text_model.");

    let vision = VisionTower::from_weights(&config.vision_config, &mut vision_weights)?;
    let text = TextTower::from_weights(&config.text_config, &mut text_weights)?;

    // logit_scale / logit_bias are top-level `(1,)` tensors.
    let logit_scale = take_shaped(&mut weights, "logit_scale", "logit_scale (1,)", &[1])?;
    let logit_bias = take_shaped(&mut weights, "logit_bias", "logit_bias (1,)", &[1])?;

    Ok(Self {
      config,
      vision,
      text,
      logit_scale,
      logit_bias,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &Siglip2NaflexConfig {
    &self.config
  }

  /// Encode one preprocessed image ([`NaflexInputs`]) to an L2-normalized
  /// image embedding `(1, hidden)`.
  ///
  /// Runs the vision tower and takes its attention-pooled output (the
  /// `pooler_output` of `siglip.py`'s `get_image_features`), then L2-normalizes
  /// along the last axis — `siglip.py`'s `image_embeds =
  /// normalize_embeddings(image_embeds)`. The vision config must enable the
  /// attention-pool head (`vision_use_head = true`, the base checkpoint
  /// default); a head-less config is rejected (no pooled image embedding
  /// exists).
  pub fn encode_image(&self, inputs: &NaflexInputs) -> Result<Array> {
    let (_last_hidden, pooled) = self.vision.forward(inputs)?;
    let pooled = pooled.ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "Siglip2NaflexModel::encode_image",
        "vision_use_head is false — no pooled image embedding (the contrastive image feature is the attention-pool head output)",
      ))
    })?;
    l2_normalize_eps(&pooled, Self::NORMALIZE_EPS)
  }

  /// Encode a `(batch, seq_len)` i32 token-id batch to L2-normalized text
  /// embeddings `(batch, projection_size)`.
  ///
  /// Runs the text tower's sticky-EOS pooled projection (the `pooled_output`
  /// of `siglip.py`'s `get_text_features`) and L2-normalizes along the last
  /// axis — `siglip.py`'s `text_embeds = normalize_embeddings(text_embeds)`.
  /// `input_ids` is padded/truncated to a fixed `seq_len` by the caller.
  pub fn encode_text(&self, input_ids: &Array) -> Result<Array> {
    let pooled = self.text.forward(input_ids)?;
    l2_normalize_eps(&pooled, Self::NORMALIZE_EPS)
  }

  /// Contrastive logits between text and image embeddings.
  ///
  /// Mirrors `siglip.py`'s `Model.__call__` contrastive tail:
  /// `logits_per_text = (text_embeds @ image_embeds.T) * exp(logit_scale) +
  /// logit_bias`. Both inputs are the **already-L2-normalized** embeddings
  /// (the outputs of [`encode_text`](Self::encode_text) /
  /// [`encode_image`](Self::encode_image)); `text_embeds` is
  /// `(n_text, dim)`, `image_embeds` is `(n_image, dim)`, and the result is
  /// `(n_text, n_image)`.
  pub fn logits(&self, text_embeds: &Array, image_embeds: &Array) -> Result<Array> {
    // text_embeds @ image_embeds.T → (n_text, n_image).
    let image_t = ops::shape::swapaxes(image_embeds, -1, -2)?;
    let sim = ops::linalg_basic::matmul(text_embeds, &image_t)?;
    // * exp(logit_scale) + logit_bias (both (1,), broadcast).
    let scale = ops::arithmetic::exp(&self.logit_scale)?;
    let scaled = sim.multiply(&scale)?;
    scaled.add(&self.logit_bias)
  }
}

/// Strip every key with `prefix` from `weights` into a new map (with the prefix
/// removed), leaving the non-matching keys in place. Used to split the
/// sanitized dual-tower map into per-tower sub-maps.
#[cfg(feature = "siglip2-naflex")]
fn strip_prefix(weights: &mut HashMap<String, Array>, prefix: &str) -> HashMap<String, Array> {
  let keys: Vec<String> = weights
    .keys()
    .filter(|k| k.starts_with(prefix))
    .cloned()
    .collect();
  let mut out = HashMap::with_capacity(keys.len());
  for k in keys {
    if let Some(v) = weights.remove(&k) {
      out.insert(k[prefix.len()..].to_string(), v);
    }
  }
  out
}

/// Rewrite a raw `google/siglip2-base-patch16-naflex` checkpoint into the
/// layout [`Siglip2NaflexModel::from_weights`] loads — the Rust analogue of
/// `siglip.py`'s `Model.sanitize` (combined with each tower's own `sanitize`).
///
/// Rules (applied per `(key, value)`):
/// 1. **Namespace each tower.** A `text_model.*` key that is not already
///    `text_model.text_model.*` is re-prefixed to `text_model.text_model.*`
///    (HF stores the text transformer one level shallower than `siglip.py`'s
///    `SiglipTextModel.text_model` nesting); likewise `vision_model.*` →
///    `vision_model.vision_model.*`.
/// 2. **Rename the attention-pool head's combined projection.** The HF
///    `…head.attention.in_proj_weight` / `…in_proj_bias` (PyTorch
///    `nn.MultiheadAttention`'s combined-QKV parameter names) become
///    `…head.attention.in_proj.weight` / `…in_proj.bias` (the `siglip.py`
///    `MHA.in_proj` `nn.Linear` form this port loads).
/// 3. **Drop unused `position_ids`** buffers (a non-parameter index buffer HF
///    stores; both towers' `sanitize` skip it).
/// 4. **Reject a duplicate destination key** with [`Error::KeyCollision`] (via
///    [`crate::model_validation::insert_unique`]) rather than letting an
///    arbitrary (per-run nondeterministic) survivor silently overwrite the
///    other.
///
/// The patch-embed Conv2d→channels-last transpose is **not** done here: the
/// NaFlex patch embed is a [`vision::VisionTower`] Linear over pre-flattened
/// patches, and [`vision`]'s `reshape_patch_weight` accepts both the
/// channels-last `(hidden, P, P, C)` and the flattened `(hidden, P^2 * C)`
/// forms, pinning the shape either way.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  let mut out = HashMap::with_capacity(weights.len());
  for (mut k, v) in weights {
    // 3. Drop the non-parameter position_ids buffers.
    if k.contains("position_ids") {
      continue;
    }

    // 1. Namespace each tower (one level deeper) if not already nested.
    if k.starts_with("text_model.") && !k.starts_with("text_model.text_model.") {
      k = format!("text_model.{k}");
    } else if k.starts_with("vision_model.") && !k.starts_with("vision_model.vision_model.") {
      k = format!("vision_model.{k}");
    }

    // 2. The MultiheadAttention combined-QKV parameter rename.
    if k.contains("in_proj_weight") {
      k = k.replace("in_proj_weight", "in_proj.weight");
    } else if k.contains("in_proj_bias") {
      k = k.replace("in_proj_bias", "in_proj.bias");
    }

    // 4. Insert, rejecting a duplicate destination key with a typed error.
    crate::model_validation::insert_unique(&mut out, k, v, "Siglip2 sanitize")?;
  }
  Ok(out)
}

/// The [`EmbeddingModelConstructor`] that builds a [`Siglip2NaflexModel`] from
/// a loaded model directory, for registration into an
/// [`EmbeddingModelTypeRegistry`].
///
/// The constructor parses the raw `config.json`, [`sanitize`]s a *cheap clone*
/// of the loaded weight map (mlx [`Array`] is a refcounted handle, so the
/// clone shares the device buffers — no data copy), and builds the model. The
/// `config.json` parse uses the `siglip2-naflex` feature's `serde_json` (the
/// `embeddings` base feature is otherwise `serde_json`-free; this model gates
/// it on).
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub fn constructor() -> EmbeddingModelConstructor {
  Box::new(
    |loaded: &LoadedEmbeddingModel| -> Result<Box<dyn EmbeddingModel>> {
      let config = Siglip2NaflexConfig::from_json(loaded.config_json())?;
      // mlx `Array` is a cheap refcounted handle; `try_clone` shares the device
      // buffer (no copy). Clone the loaded map into an owned, sanitizable map.
      let mut raw = HashMap::with_capacity(loaded.weights_ref().len());
      for (k, v) in loaded.weights_ref() {
        raw.insert(k.clone(), v.try_clone()?);
      }
      let weights = sanitize(raw)?;
      let model = Siglip2NaflexModel::from_weights(config, weights)?;
      Ok(Box::new(model))
    },
  )
}

/// Register [`Siglip2NaflexModel`] under [`MODEL_TYPE`] (`"siglip"`) into
/// `registry`, returning any constructor it displaced.
///
/// The registry is the documented architecture extension point (per-model
/// architectures are not auto-registered); call this to enable loading a
/// SigLIP2 NaFlex checkpoint through [`crate::embeddings::load`].
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub fn register(registry: &mut EmbeddingModelTypeRegistry) -> Option<EmbeddingModelConstructor> {
  registry.register(MODEL_TYPE, constructor())
}

/// The text-tower forward seam: SigLIP2's text encoder maps a `(batch,
/// seq_len)` token-id batch to its sticky-EOS pooled projection.
///
/// `forward`'s `last_hidden_state` is the projected pooled embedding lifted to
/// `(batch, 1, projection_size)` (the `EmbeddingModelOutput` contract requires
/// a rank-3 hidden state), with the same vector also exposed as
/// `pooled_output`. `attention_mask` is ignored: SigLIP's sticky-EOS pooling
/// reads the last position regardless (it pads with a real token and does not
/// mask), matching `siglip.py`'s `attention_mask=None` text path.
#[cfg(feature = "siglip2-naflex")]
impl EmbeddingModel for Siglip2NaflexModel {
  fn forward(&self, input_ids: &Array, _attention_mask: &Array) -> Result<EmbeddingModelOutput> {
    // The text tower already returns the pooled (batch, projection_size)
    // embedding. Expose it as both the pooled output and (lifted to rank-3)
    // the last_hidden_state the EmbeddingModelOutput contract requires.
    let pooled = self.text.forward(input_ids)?;
    let last_hidden = ops::shape::expand_dims_axes(&pooled, &[1])?; // (B, 1, dim)
    Ok(EmbeddingModelOutput::new(last_hidden, Some(pooled)))
  }
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
