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
//! [`crate::ops::interpolation::bilinear_interpolate`] (the per-image
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
//!   (Conv2d-as-Linear) + per-image bilinear+antialias position resize +
//!   masked pre-norm encoder + optional attention-pool head.
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
//! `logit_scale` / `logit_bias` contrastive similarity. It implements the golden
//! embedding seams — the model-implemented [`TextEmbedder`] (owning its
//! fixed-length tokenization + sticky-EOS pooling), `Embed<ImageInput>`, and
//! [`Contrastive`] — and answers the load factory's
//! [`crate::embeddings::EmbeddingModel`] umbrella (text + contrastive accessors;
//! the image tower is reached by downcast), so it registers into the
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
    Contrastive, Embed, Embedding, EmbeddingModel, EmbeddingModelConstructor,
    EmbeddingModelTypeRegistry, LoadedEmbeddingModel, Padding, SWIFT_L2_EPS, StPoolingConfig,
    TextEmbedder, TextEncoding, l2_normalize_eps,
    siglip2_naflex::{
      config::Siglip2NaflexConfig, processing::NaflexInputs, shared::take_shaped, text::TextTower,
      vision::VisionTower,
    },
  },
  error::{AllocFailurePayload, Error, InvariantViolationPayload, OutOfRangePayload, Result},
  model_validation::reserve_or_error,
  ops,
};

/// The top-level architecture id this model registers under.
#[cfg(feature = "siglip2-naflex")]
pub const MODEL_TYPE: &str = "siglip2";

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

  /// The token id the SigLIP2 processor right-pads text to its fixed sequence
  /// length with — the canonical SigLIP sentencepiece pad/EOS id (`1`). The
  /// sticky-EOS pooling reads a fixed last position regardless of the mask, so
  /// a shorter prompt's trailing positions hold this pad id (which is exactly
  /// what the reference processor produces); it is a real, *unmasked* position.
  const TEXT_PAD_TOKEN_ID: u32 = 1;

  /// The SigLIP2 sentencepiece `<eos>` id (`1`). The text tower pools the
  /// **last** position under the sticky-EOS invariant ("last token is always
  /// EOS"), so an **overlength** prompt must keep the EOS at its final position
  /// rather than a content token. The HF SigLIP processor enforces this by
  /// head-truncating to `max_length - n_added_tokens` and then appending the
  /// EOS via its post-processor template; this is the id the generic
  /// fixed-length builder forces into the last slot on truncation
  /// ([`Padding::FixedLength::eos_token_id`]). It coincides with
  /// [`TEXT_PAD_TOKEN_ID`](Self::TEXT_PAD_TOKEN_ID) for SigLIP (both `1`) but is
  /// conceptually the EOS, not the pad fill.
  const TEXT_EOS_TOKEN_ID: u32 = 1;

  /// The fixed text sequence length the SigLIP2 processor pads / truncates every
  /// prompt to — the text tower's `max_position_embeddings` (the sticky-EOS
  /// sequence length whose last position the tower pools). Read from the parsed
  /// config so it tracks the checkpoint rather than a hard-coded literal.
  fn text_seq_len(&self) -> Result<usize> {
    usize::try_from(self.config.text_config.max_position_embeddings).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Siglip2NaflexModel::text_seq_len",
        "text_config.max_position_embeddings must be a non-negative sequence length",
        smol_str::format_smolstr!("{}", self.config.text_config.max_position_embeddings),
      ))
    })
  }

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
    let mut vision_weights = strip_prefix(&mut weights, "vision_model.vision_model.")?;
    let mut text_weights = strip_prefix(&mut weights, "text_model.text_model.")?;

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

/// Fallibly build the concatenation of `parts` into a freshly-reserved
/// [`String`], turning an allocator failure into a typed [`Error::AllocFailure`]
/// instead of the abort `String::push_str` / `format!` would raise on growth.
///
/// Every per-key rewrite here (tower namespacing, the `in_proj` rename, prefix
/// stripping, the constructor's key clone) is sized by a **checkpoint-controlled**
/// key, so each new key `String` is built through this one fallible path rather
/// than an infallible `format!` / `to_string` / `clone` — a hostile checkpoint
/// with enormous keys surfaces a recoverable error instead of aborting. The
/// reservation is exact (the total of `parts`' byte lengths), so the pushes
/// cannot reallocate. An overflowing total length is itself an allocation the
/// reservation rejects (a `String` cannot exceed `isize::MAX` bytes).
#[cfg(feature = "siglip2-naflex")]
fn fallible_concat(context: &'static str, parts: &[&str]) -> Result<String> {
  let total = parts
    .iter()
    .fold(0usize, |acc, p| acc.saturating_add(p.len()));
  let mut out = String::new();
  out.try_reserve_exact(total).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "Siglip2 key rewrite",
      context,
      total as u64,
      e,
    ))
  })?;
  for p in parts {
    out.push_str(p);
  }
  Ok(out)
}

/// Fallibly clone `s` into an owned [`String`] (the single-part `fallible_concat`),
/// surfacing a typed [`Error::AllocFailure`] on allocator failure rather than the
/// abort `str::to_string` / `String::clone` would raise.
#[cfg(feature = "siglip2-naflex")]
fn fallible_clone_str(context: &'static str, s: &str) -> Result<String> {
  fallible_concat(context, &[s])
}

/// Fallibly replace every non-overlapping occurrence of `from` in `s` with `to`
/// (the [`str::replace`] semantics), building the result through the fallible
/// allocation path ([`Error::AllocFailure`] on allocator failure) rather than the
/// infallible `str::replace`. Used for the checkpoint-controlled `in_proj` key
/// rename so a hostile key cannot abort on the rewrite allocation.
#[cfg(feature = "siglip2-naflex")]
fn fallible_replace(context: &'static str, s: &str, from: &str, to: &str) -> Result<String> {
  let count = s.matches(from).count();
  if count == 0 {
    // No occurrence: the result is `s` unchanged (still fallibly owned).
    return fallible_clone_str(context, s);
  }
  // `str::split(from)` yields `count + 1` pieces joined by `to`; the exact output
  // length is `s.len() - count*from.len() + count*to.len()`. `count*from.len() <=
  // s.len()` (non-overlapping matches lie within `s`), so the subtraction cannot
  // underflow; the additions saturate defensively.
  let removed = count.saturating_mul(from.len());
  let added = count.saturating_mul(to.len());
  let new_len = s.len().saturating_sub(removed).saturating_add(added);
  let mut out = String::new();
  out.try_reserve_exact(new_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "Siglip2 key rewrite",
      context,
      new_len as u64,
      e,
    ))
  })?;
  for (i, piece) in s.split(from).enumerate() {
    if i != 0 {
      out.push_str(to);
    }
    out.push_str(piece);
  }
  Ok(out)
}

/// Strip every key with `prefix` from `weights` into a new map (with the prefix
/// removed), leaving the non-matching keys in place. Used to split the
/// sanitized dual-tower map into per-tower sub-maps.
///
/// Both the matched-key `Vec` and the destination `HashMap` are sized by the
/// (checkpoint-controlled) matching-key count, so each is reserved **fallibly**
/// via [`crate::model_validation::reserve_or_error`] — a within-cap but
/// heavyweight reservation surfaces as a typed [`Error::AllocFailure`] rather
/// than the abort `Vec::with_capacity` / `HashMap::with_capacity` would raise on
/// a hostile checkpoint with an enormous key set. Both the full matched key (the
/// owned key needed to `remove` the entry) and the prefix-stripped destination
/// key are built through the fallible `fallible_clone_str` path (not an
/// infallible `String::clone` / `to_string`), so every owned key allocation
/// sized by a checkpoint key surfaces a typed [`Error::AllocFailure`].
#[cfg(feature = "siglip2-naflex")]
fn strip_prefix(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
) -> Result<HashMap<String, Array>> {
  let matching = weights.keys().filter(|k| k.starts_with(prefix)).count();
  let mut keys: Vec<String> = Vec::new();
  reserve_or_error(
    &mut keys,
    "Siglip2 strip_prefix: matched key list",
    matching,
  )?;
  // Build the matched-key list through the fallible String path: the full key
  // clone (needed to `remove` the entry below) is sized by the
  // checkpoint-controlled source key, so an enormous key surfaces a typed
  // `AllocFailure` instead of the abort `String::clone` would raise. The Vec is
  // already reserved to `matching`, so the push cannot reallocate.
  for k in weights.keys().filter(|k| k.starts_with(prefix)) {
    keys.push(fallible_clone_str("strip_prefix: matched key", k)?);
  }
  let mut out: HashMap<String, Array> = HashMap::new();
  reserve_or_error(
    &mut out,
    "Siglip2 strip_prefix: per-tower sub-map",
    keys.len(),
  )?;
  for k in keys {
    if let Some(v) = weights.remove(&k) {
      // The stripped key is sized by the (checkpoint-controlled) source key, so
      // build it through the fallible String path (typed `AllocFailure`) rather
      // than an infallible `to_string`.
      let stripped = fallible_clone_str("strip_prefix: stripped key", &k[prefix.len()..])?;
      out.insert(stripped, v);
    }
  }
  Ok(out)
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
/// The rewritten keys (rules 1 and 2) are each built through the fallible
/// `fallible_concat` / `fallible_replace` path (typed [`Error::AllocFailure`]),
/// not an infallible `format!` / `str::replace`, so a hostile checkpoint with
/// enormous keys cannot abort on a per-key rewrite allocation; the destination
/// map reserve and each `insert_unique` slot are fallible too.
///
/// The patch-embed Conv2d→channels-last transpose is **not** done here (the key
/// names need no rewrite): the NaFlex patch embed is a [`vision::VisionTower`]
/// Linear over pre-flattened patches, and [`vision`]'s `reshape_patch_weight`
/// handles every checkpoint layout — the MLX channels-last `(hidden, P, P, C)`,
/// the raw PyTorch / HF `(hidden, C, P, P)` (`nn.Conv2d`'s `(out, in, kH, kW)`,
/// transposed `[0, 2, 3, 1]` to channels-last there), and the already-flattened
/// `(hidden, P^2 * C)` — pinning the shape in every case.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  // The destination map is sized by the (checkpoint-controlled) source key
  // count; reserve it FALLIBLY (typed `AllocFailure`) rather than via
  // `HashMap::with_capacity`, which aborts on a hostile checkpoint whose key set
  // is large enough to exhaust the allocator. (`insert_unique` below reserves
  // each new slot fallibly too, but pre-sizing once avoids repeated growth.)
  let mut out: HashMap<String, Array> = HashMap::new();
  reserve_or_error(&mut out, "Siglip2 sanitize: destination map", weights.len())?;
  for (mut k, v) in weights {
    // 3. Drop the non-parameter position_ids buffers.
    if k.contains("position_ids") {
      continue;
    }

    // 1. Namespace each tower (one level deeper) if not already nested. The
    //    re-prefixed key is sized by the (checkpoint-controlled) source key, so
    //    build it through the fallible String path (typed `AllocFailure`) rather
    //    than an infallible `format!`.
    if k.starts_with("text_model.") && !k.starts_with("text_model.text_model.") {
      k = fallible_concat("sanitize: text_model namespace", &["text_model.", &k])?;
    } else if k.starts_with("vision_model.") && !k.starts_with("vision_model.vision_model.") {
      k = fallible_concat("sanitize: vision_model namespace", &["vision_model.", &k])?;
    }

    // 2. The MultiheadAttention combined-QKV parameter rename (fallible replace).
    if k.contains("in_proj_weight") {
      k = fallible_replace(
        "sanitize: in_proj.weight rename",
        &k,
        "in_proj_weight",
        "in_proj.weight",
      )?;
    } else if k.contains("in_proj_bias") {
      k = fallible_replace(
        "sanitize: in_proj.bias rename",
        &k,
        "in_proj_bias",
        "in_proj.bias",
      )?;
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
/// map reserve and each cloned key `String` are built through the fallible path
/// (typed [`Error::AllocFailure`] via [`reserve_or_error`] / `fallible_clone_str`),
/// not an infallible `with_capacity` / `clone`, so a hostile loaded map cannot
/// abort during the clone. The `config.json` parse uses the `siglip2-naflex`
/// feature's `serde_json` (the `embeddings` base feature is otherwise
/// `serde_json`-free; this model gates it on).
///
/// The parsed `1_Pooling/config.json` (`_pooling`) is **ignored**: SigLIP is a
/// dual-tower contrastive model that bakes its own fixed sticky-EOS text pooling
/// and fixed-length padding (see [`TextEmbedder`]); it does not consume a
/// sentence-encoder pooling config.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub fn constructor() -> EmbeddingModelConstructor {
  Box::new(
    |loaded: &LoadedEmbeddingModel,
     _pooling: Option<&StPoolingConfig>|
     -> Result<Box<dyn EmbeddingModel>> {
      let config = Siglip2NaflexConfig::from_json(loaded.config_json())?;
      // mlx `Array` is a cheap refcounted handle; `try_clone` shares the device
      // buffer (no copy). Clone the loaded map into an owned, sanitizable map.
      // The map is sized by the loaded (checkpoint-controlled) key count, so
      // reserve it FALLIBLY (typed `AllocFailure`) rather than via
      // `HashMap::with_capacity`, which aborts under allocator pressure.
      let mut raw: HashMap<String, Array> = HashMap::new();
      reserve_or_error(
        &mut raw,
        "Siglip2 constructor: sanitizable weight-map clone",
        loaded.weights_ref().len(),
      )?;
      for (k, v) in loaded.weights_ref() {
        // The cloned key is sized by the (checkpoint-controlled) loaded key, so
        // build it through the fallible String path (typed `AllocFailure`)
        // rather than an infallible `clone`.
        let key = fallible_clone_str("constructor: weight-map key clone", k)?;
        raw.insert(key, v.try_clone()?);
      }
      let weights = sanitize(raw)?;
      let model = Siglip2NaflexModel::from_weights(config, weights)?;
      Ok(Box::new(model))
    },
  )
}

/// Register [`Siglip2NaflexModel`] under [`MODEL_TYPE`] (`"siglip2"`) into
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

/// SigLIP2's image input modality for [`Embed`]: a preprocessed NaFlex image
/// ([`NaflexInputs`]).
///
/// Image inputs are model-defined — a fixed-resolution model would use a
/// different type — so SigLIP owns this newtype; there is no universal
/// `dyn ImageEmbedder`. Reach the image tower from a loaded
/// [`crate::embeddings::EmbeddingModel`] by downcasting to [`Siglip2NaflexModel`]
/// and calling [`encode_image`](Siglip2NaflexModel::encode_image) (or this
/// `Embed` impl).
#[cfg(feature = "siglip2-naflex")]
pub struct ImageInput<'a>(pub &'a NaflexInputs);

/// Text tower as the universal text seam ([`TextEmbedder`]). SigLIP owns both
/// text stages a generic pipeline cannot standardize:
///
/// - [`text_encoding`](TextEmbedder::text_encoding) declares the SigLIP2
///   processor's **fixed-length** input scheme ([`Padding::FixedLength`]):
///   tokenize with special tokens, then pad/truncate every prompt to the text
///   tower's `max_position_embeddings` with the SigLIP pad id, with an all-`1`
///   mask, and an `eos_token_id` so an overlength prompt keeps the EOS at its
///   final position on truncation (HF truncate-then-append-EOS — the sticky-EOS
///   pooler must never see a content token in its last slot). The generic
///   [`encode`](crate::embeddings::encode()) pipeline applies exactly this — so
///   a mixed-length batch never pools a foreign dynamic-pad position (the bug a
///   generic mask-aware right-pad would introduce against a fixed-position
///   pooler), and an overlength prompt is byte-identical to the SigLIP
///   processor's truncated ids.
/// - [`embed_text`](TextEmbedder::embed_text) runs the sticky-EOS pooled
///   projection (SigLIP's real text feature, not a generic pool) via
///   [`encode_text`](Self::encode_text). The mask is unused — sticky-EOS reads a
///   fixed last position regardless, matching `siglip.py`'s `attention_mask=None`
///   text path.
#[cfg(feature = "siglip2-naflex")]
impl TextEmbedder for Siglip2NaflexModel {
  fn text_encoding(&self) -> TextEncoding {
    // The fixed sequence length is the text tower's `max_position_embeddings`
    // (the sticky-EOS length). `text_seq_len()` is fallible only on a
    // (config-validation-rejected) negative value; fall back to the conversion's
    // saturated value so this infallible accessor never panics — a real
    // checkpoint always yields the true positive length.
    let length = self
      .text_seq_len()
      .unwrap_or_else(|_| self.config.text_config.max_position_embeddings.max(0) as usize);
    // The tokenizer's per-text truncation cap is NOT set here: the
    // `Padding::FixedLength` scheme is itself the truncation cap, and the generic
    // pipeline derives the effective cap centrally from it (the fixed `length`,
    // `+ 1` for the sticky-EOS slot so the trailing EOS survives a genuine
    // truncation). Leaving `max_length = None` means the cap is the padding mode's
    // own contract, not an optional field this model must remember to set; the
    // central derivation yields exactly `length + 1` for this sticky-EOS scheme,
    // so the behaviour is identical to setting it explicitly here.
    TextEncoding::new(
      // SigLIP2's processor encodes with special tokens, then pads/truncates to
      // the fixed length. The fixed `length` is the effective output cap and the
      // generic pipeline derives the slightly-larger tokenizer truncation cap
      // (`length + 1`) from it centrally (the final fixed-length truncation still
      // runs after it).
      true,
      None,
      Padding::FixedLength {
        length,
        pad_token_id: Self::TEXT_PAD_TOKEN_ID,
        // Preserve the sticky-EOS invariant on overlength prompts: a truncated
        // row keeps its head to `length - 1` and the EOS is forced into the
        // last position (HF truncate-then-append-EOS), so the pooled last slot
        // is never a content token. A within-length prompt is unaffected.
        eos_token_id: Some(Self::TEXT_EOS_TOKEN_ID),
      },
    )
  }

  fn embed_text(&self, input_ids: &Array, _attention_mask: &Array) -> Result<Embedding> {
    Ok(Embedding::new(self.encode_text(input_ids)?))
  }
}

/// Vision tower: a preprocessed NaFlex image → its L2-normalized image embedding.
#[cfg(feature = "siglip2-naflex")]
impl<'a> Embed<ImageInput<'a>> for Siglip2NaflexModel {
  type Output = Embedding;
  fn embed(&self, input: ImageInput<'a>) -> Result<Embedding> {
    Ok(Embedding::new(self.encode_image(input.0)?))
  }
}

/// Contrastive similarity — `logits_per_text` over the two already-L2-normalized
/// embeddings (`exp(logit_scale) * text @ image.T + logit_bias`).
#[cfg(feature = "siglip2-naflex")]
impl Contrastive for Siglip2NaflexModel {
  fn similarity(&self, text: &Embedding, image: &Embedding) -> Result<Array> {
    self.logits(text.array(), image.array())
  }
}

/// SigLIP2 answers the load factory umbrella's universal text + contrastive
/// capabilities; its image tower (a model-defined input) is reached by downcast
/// via [`as_any`](crate::embeddings::EmbeddingModel::as_any).
#[cfg(feature = "siglip2-naflex")]
impl EmbeddingModel for Siglip2NaflexModel {
  fn as_text_embedder(&self) -> Option<&dyn TextEmbedder> {
    Some(self)
  }
  fn as_contrastive(&self) -> Option<&dyn Contrastive> {
    Some(self)
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
