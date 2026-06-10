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
  error::{
    AllocFailurePayload, Error, InvariantViolationPayload, KeyCollisionPayload, OutOfRangePayload,
    ParsePayload, Result,
  },
  lm::quant::PerLayerQuantization,
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
  /// The pad id the fixed-length text scheme fills with. Defaults to the
  /// SigLIP2 Gemma `<pad>` id
  /// ([`DEFAULT_TEXT_PAD_TOKEN_ID`](Self::DEFAULT_TEXT_PAD_TOKEN_ID), `0`);
  /// the load-factory [`constructor`] refines it from the checkpoint's
  /// tokenizer metadata via [`set_text_pad_token_id`](Self::set_text_pad_token_id).
  text_pad_token_id: u32,
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl Siglip2NaflexModel {
  /// L2-normalization eps for the image/text embeddings. SigLIP normalizes
  /// with the swift `1e-12` floor (`MLXEmbedders` `l2Normalized`); the
  /// embeddings are f32 here so the choice is immaterial to the result, but
  /// it pins the intent.
  const NORMALIZE_EPS: f32 = SWIFT_L2_EPS;

  /// The default token id the SigLIP2 processor right-pads text to its fixed
  /// sequence length with — the SigLIP2 **Gemma** tokenizer's `<pad>` id (`0`).
  ///
  /// SigLIP2 ships a Gemma sentencepiece tokenizer
  /// (`google/siglip2-base-patch16-naflex`'s `tokenizer_config.json`:
  /// `tokenizer_class = "GemmaTokenizer"`, `added_tokens_decoder` binding
  /// `"0" → <pad>` and `"1" → <eos>`, `add_eos_token = true`), and the HF
  /// `Siglip2Processor` pads with `padding="max_length"` — i.e. with
  /// `tokenizer.pad_token_id == 0`. This is **not** SigLIP1's convention: the
  /// original SigLIP sentencepiece used pad == EOS == `1`, and padding a
  /// SigLIP2 prompt with `1` fills the tail with `<eos>` tokens instead of
  /// `<pad>`. Because the sticky-EOS pooling reads a fixed **last** position
  /// regardless of the mask — a *pad* slot for every shorter-than-`seq_len`
  /// prompt — a wrong pad fill changes the pooled token embedding and shifts
  /// every short prompt's embedding away from the reference processor's.
  /// (HF `Siglip2TextConfig` still *declares* `pad_token_id = 1` as a class
  /// default inherited from SigLIP1's CLIP-vocab defaults; that field is
  /// unused by the tokenization path and is not what the processor pads with.)
  ///
  /// The load-factory [`constructor`] refines this default from the loaded
  /// **tokenizer directory**'s metadata (`tokenizer_config.json` /
  /// `special_tokens_map.json` — see [`read_text_pad_token_id`]); `0` is the
  /// correct fallback when that metadata is unavailable (every published
  /// SigLIP2 checkpoint uses the Gemma tokenizer's `<pad> = 0`).
  const DEFAULT_TEXT_PAD_TOKEN_ID: u32 = 0;

  /// The SigLIP2 sentencepiece `<eos>` id (`1` — the Gemma tokenizer's
  /// `added_tokens_decoder` `"1" → <eos>`). The text tower pools the
  /// **last** position under the sticky-EOS invariant ("last token is always
  /// EOS"), so an **overlength** prompt must keep the EOS at its final position
  /// rather than a content token. The HF SigLIP processor enforces this by
  /// head-truncating to `max_length - n_added_tokens` and then appending the
  /// EOS via its post-processor template; this is the id the generic
  /// fixed-length builder forces into the last slot on truncation
  /// ([`Padding::FixedLength::eos_token_id`]). It is distinct from the pad
  /// fill ([`DEFAULT_TEXT_PAD_TOKEN_ID`](Self::DEFAULT_TEXT_PAD_TOKEN_ID),
  /// `<pad> = 0`): only a *truncated* (overlength) prompt ends in this EOS at
  /// the fixed last slot; a shorter prompt's last slot is a pad token, exactly
  /// as the reference processor produces.
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
    weights: HashMap<String, Array>,
  ) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build a model from a parsed [`Siglip2NaflexConfig`], the **sanitized**
  /// weight map, and an optional parsed quantization config — the
  /// quantization-aware analogue of [`from_weights`](Self::from_weights) (which
  /// is just this with `quantization == None`).
  ///
  /// When a layer's `<prefix>.weight` carries the sibling `<prefix>.scales`
  /// tensor, that `nn.Linear` / `nn.Embedding` is built quantized — the
  /// weight-map analogue of mlx-embeddings' `get_class_predicate`'s
  /// `f"{p}.scales" in weights` signal (`convert.py`), with the
  /// `(group_size, bits, mode)` resolved per layer from `quantization`. The
  /// `.scales` sibling ALONE selects the quantized path, exactly as the shared
  /// [`crate::nn::MaybeQuantizedLinear`] / [`crate::nn::MaybeQuantizedEmbedding`]
  /// loaders do: a `.scales`-bearing layer that resolves no scheme params (a
  /// missing or non-resolving `quantization`) is the typed
  /// [`Error::InvariantViolation`] below, never a silent fall-through to the
  /// dense loader that could accept a malformed packed weight as a dense one. A
  /// dense layer (no `.scales` sibling) builds exactly as before, so a
  /// non-quantized checkpoint loads identically whether or not a `quantization`
  /// config is threaded. An mlx-community 8-bit SigLIP2 checkpoint
  /// (`skip_vision=True` by default) loads its quantized text-tower projections +
  /// token embedding through this entry; the [`sanitize`] key-remap carries the
  /// `.scales` / `.biases` siblings through unchanged.
  ///
  /// # Errors
  /// The [`from_weights`](Self::from_weights) errors, plus
  /// [`Error::InvariantViolation`] if a `<prefix>.scales` sibling is present but
  /// `quantization` resolved no scheme parameters for that layer, and the shared
  /// [`crate::nn::QuantizedLinear::from_parts`] /
  /// [`crate::nn::MaybeQuantizedEmbedding::from_parts`] structural-validation
  /// errors for a malformed quantized triple.
  pub fn from_weights_quantized(
    config: Siglip2NaflexConfig,
    mut weights: HashMap<String, Array>,
    quantization: Option<&PerLayerQuantization>,
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

    // Normalize the per-layer quantization keys into the SAME namespace as the
    // per-tower lookup prefixes the towers resolve against. A SigLIP2 config can
    // carry per-layer override keys under the HF-shallow tower path
    // (`text_model.encoder.layers.0.self_attn.q_proj`), but each tower calls
    // `quantization_for` with its prefix-stripped path
    // (`encoder.layers.0.self_attn.q_proj`); reproject the config keys through
    // the same `sanitize` + strip the tower namespace so a per-layer override
    // matches. (The common case — a global-only `{ group_size, bits, mode }`
    // with no per-layer keys — is unaffected.)
    let vision_quant = quantization
      .map(|q| reproject_quant_keys(q, "vision_model.vision_model."))
      .transpose()?;
    let text_quant = quantization
      .map(|q| reproject_quant_keys(q, "text_model.text_model."))
      .transpose()?;

    // Strip the tower-namespace prefixes the sanitized map carries
    // (`vision_model.vision_model.` / `text_model.text_model.`) and build each
    // tower from its own sub-map, then read the top-level contrastive params.
    let mut vision_weights = strip_prefix(&mut weights, "vision_model.vision_model.")?;
    let mut text_weights = strip_prefix(&mut weights, "text_model.text_model.")?;

    let vision = VisionTower::from_weights_quantized(
      &config.vision_config,
      &mut vision_weights,
      vision_quant.as_ref(),
    )?;
    let text = TextTower::from_weights_quantized(
      &config.text_config,
      &mut text_weights,
      text_quant.as_ref(),
    )?;

    // logit_scale / logit_bias are top-level `(1,)` tensors.
    let logit_scale = take_shaped(&mut weights, "logit_scale", "logit_scale (1,)", &[1])?;
    let logit_bias = take_shaped(&mut weights, "logit_bias", "logit_bias (1,)", &[1])?;

    Ok(Self {
      config,
      vision,
      text,
      logit_scale,
      logit_bias,
      text_pad_token_id: Self::DEFAULT_TEXT_PAD_TOKEN_ID,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &Siglip2NaflexConfig {
    &self.config
  }

  /// The pad id the fixed-length text scheme
  /// ([`TextEmbedder::text_encoding`]) fills with.
  #[inline(always)]
  pub fn text_pad_token_id(&self) -> u32 {
    self.text_pad_token_id
  }

  /// Override the pad id the fixed-length text scheme fills with.
  ///
  /// Defaults to the SigLIP2 Gemma `<pad>` id (`DEFAULT_TEXT_PAD_TOKEN_ID`,
  /// `0`). The load-factory [`constructor`] calls this with the id resolved
  /// from the checkpoint's own tokenizer metadata (`read_text_pad_token_id`)
  /// so the pad fill always tracks the tokenizer that produced the checkpoint;
  /// a caller building the model directly via
  /// [`from_weights`](Self::from_weights) can do the same.
  pub fn set_text_pad_token_id(&mut self, id: u32) {
    self.text_pad_token_id = id;
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

/// Reproject a parsed [`PerLayerQuantization`]'s per-layer override keys into a
/// single tower's prefix-stripped namespace, keeping the global default — so a
/// per-layer override matches the tower-relative path the tower's
/// `quantization_for` lookups use.
///
/// Each tower's `from_weights` receives a weight sub-map with its
/// `vision_model.vision_model.` / `text_model.text_model.` namespace stripped
/// (see [`strip_prefix`]) and resolves the per-layer scheme against the
/// tower-relative path (e.g. `encoder.layers.0.self_attn.q_proj`). A SigLIP2
/// quantized `config.json` keys any per-layer override under the full HF path,
/// which appears either already-nested (`<prefix>encoder.layers…`) or one level
/// shallower (`text_model.encoder.layers…` / `vision_model.encoder.layers…`, the
/// HF-shallow form [`sanitize`] re-nests). This strips whichever of those two
/// forms a key carries to the tower-relative path, and drops keys belonging to
/// the *other* tower (or to neither). The global default — the common case, the
/// only thing an mlx-community SigLIP2 checkpoint's `quantization` block carries
/// — is preserved verbatim.
///
/// Both accepted source forms (the nested and the shallow tower prefix) strip to
/// the same tower-relative key, so two source keys can reproject onto one
/// destination. An IDENTICAL duplicate is fine (one is kept); two with DIFFERENT
/// `(group_size, bits, mode)` for the same layer are a config contradiction and a
/// typed [`Error::KeyCollision`] — a plain `insert` would otherwise let an
/// arbitrary (source-`HashMap`-iteration-order, thus nondeterministic) survivor
/// win, silently running a layer with the wrong scheme.
///
/// Each reprojected per-layer key is a config-controlled string; the destination
/// map is reserved fallibly (typed [`Error::AllocFailure`]) and each kept key is
/// cloned through the fallible String path, mirroring [`strip_prefix`].
#[cfg(feature = "siglip2-naflex")]
fn reproject_quant_keys(
  quant: &PerLayerQuantization,
  nested_prefix: &str,
) -> Result<PerLayerQuantization> {
  // The HF-shallow form `sanitize` re-nests: `text_model.text_model.` ⇒
  // `text_model.`, `vision_model.vision_model.` ⇒ `vision_model.`. A per-layer
  // key may already be nested OR still shallow; accept both. The shallow form is
  // the namespace up to and including the FIRST `.` (one tower segment).
  let shallow_prefix: &str = nested_prefix
    .split_inclusive('.')
    .next()
    .unwrap_or(nested_prefix);
  let src = quant.per_layer_ref();
  let mut per_layer: HashMap<String, crate::lm::quant::QuantizationOption> = HashMap::new();
  reserve_or_error(
    &mut per_layer,
    "Siglip2 reprojected per-layer quantization keys",
    src.len(),
  )?;
  for (path, opt) in src {
    // Strip whichever tower namespace this key carries; skip a key for the other
    // tower (it does not begin with this tower's namespace in either form).
    let stripped = if let Some(rest) = path.strip_prefix(nested_prefix) {
      rest
    } else if let Some(rest) = path.strip_prefix(shallow_prefix) {
      rest
    } else {
      continue;
    };
    // Both the nested and the shallow source forms can reproject to the SAME
    // tower-relative key. A plain `insert` would let an arbitrary survivor win
    // (the source is a `HashMap`, so iteration order — and thus which override a
    // layer ends up with — is nondeterministic). Detect a collision: an IDENTICAL
    // duplicate is fine (keep one); a DIFFERENT one is a config contradiction (the
    // same layer with two conflicting schemes) and a typed `KeyCollision`. The
    // lookup is by `&str`, so the identical-skip path allocates no owned key.
    match per_layer.get(stripped) {
      Some(existing) if *existing == *opt => continue,
      Some(_) => {
        let key = fallible_clone_str("reproject_quant_keys: conflicting per-layer key", stripped)?;
        return Err(Error::KeyCollision(KeyCollisionPayload::new(
          "siglip2 reproject_quant_keys: a per-layer quantization override is supplied twice (the nested and shallow tower-prefix forms) with conflicting (group_size, bits, mode) for the same layer",
          key,
        )));
      }
      None => {
        let key = fallible_clone_str("reproject_quant_keys: per-layer key", stripped)?;
        per_layer.insert(key, *opt);
      }
    }
  }
  Ok(PerLayerQuantization::new(quant.quantization, per_layer))
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
      // Parse the optional `config.json` `quantization` block (the mlx-community
      // native key); a dense checkpoint has none and loads dense. An mlx-community
      // 8-bit SigLIP2 checkpoint loads its quantized projections through here.
      let quantization = parse_quantization(loaded.config_json())?;
      let mut model =
        Siglip2NaflexModel::from_weights_quantized(config, weights, quantization.as_ref())?;
      // Refine the text pad id from the checkpoint's OWN tokenizer metadata
      // (`tokenizer_config.json` / `special_tokens_map.json` in the TOKENIZER
      // directory — the same directory `embeddings::load` builds the
      // `Tokenizer` from, respecting a split `tokenizer_source`), so the
      // fixed-length pad fill tracks the tokenizer that actually encodes the
      // prompts — never a stale copy in the model directory. Best-effort: an
      // absent directory (a hand-built `LoadedEmbeddingModel`), missing files,
      // or malformed metadata keep the Gemma `<pad> = 0` default — correct for
      // every published SigLIP2 checkpoint (see `DEFAULT_TEXT_PAD_TOKEN_ID`).
      if let Some(dir) = loaded.tokenizer_dir()
        && let Some(pad_id) = read_text_pad_token_id(dir)
      {
        model.set_text_pad_token_id(pad_id);
      }
      Ok(Box::new(model))
    },
  )
}

/// Upper bound on a tokenizer-metadata JSON file read into memory by
/// [`read_text_pad_token_id`]. A real `tokenizer_config.json` is tens of KB
/// (`google/siglip2-base-patch16-naflex`'s is ~40 KB; the largest Gemma-family
/// ones with embedded chat templates stay low-MB) and `special_tokens_map.json`
/// is under 1 KB; the cap keeps a planted multi-GB file from being slurped
/// (the same bounded-read discipline as the factory's `config.json` reader).
#[cfg(feature = "siglip2-naflex")]
const MAX_TOKENIZER_METADATA_BYTES: u64 = 4 << 20;

/// Best-effort read of the text **pad token id** from the tokenizer metadata
/// in the loaded **tokenizer directory** (the directory `embeddings::load`
/// builds the [`crate::tokenizer::Tokenizer`] from — the separate
/// `tokenizer_source` when configured, else the model directory) — the id the
/// SigLIP2 processor right-pads with.
///
/// Resolution (mirroring how HF's `Siglip2Processor` derives the pad fill —
/// `tokenizer.pad_token_id`, i.e. the configured `pad_token` string resolved
/// through the tokenizer's added-token table):
///
/// 1. The pad token **string**: `tokenizer_config.json`'s `pad_token`, else
///    `special_tokens_map.json`'s `pad_token` — each accepted in both HF
///    shapes, a plain string (`"<pad>"`) or an `AddedToken`-style object
///    (`{"content": "<pad>", …}`).
/// 2. The string → **id**: the entry of `tokenizer_config.json`'s
///    `added_tokens_decoder` (an `"<id>" → {"content": …}` map) whose
///    `content` equals the pad token AND whose key parses as a `u32` id.
///    (`google/siglip2-base-patch16-naflex` binds `"0" → <pad>`.) A
///    content-matching entry under a corrupt non-numeric key is **skipped**,
///    not a scan abort — a planted junk entry must not shadow the legitimate
///    binding elsewhere in the table.
///
/// Returns `None` — the caller keeps the
/// [`Siglip2NaflexModel::DEFAULT_TEXT_PAD_TOKEN_ID`] Gemma `<pad> = 0` default
/// — on **any** miss: an absent or unreadable file, an over-cap file
/// ([`MAX_TOKENIZER_METADATA_BYTES`]), malformed JSON, a missing `pad_token`
/// field, or a pad token no numeric-keyed `added_tokens_decoder` entry
/// resolves. The pad id is a refinement of an already-correct default, so a
/// metadata problem must not turn a loadable checkpoint into a load error (the
/// tokenizer load itself — which `embeddings::load` runs separately — still
/// surfaces a malformed `tokenizer_config.json` as its own typed error). The
/// deliberately **unconsulted** source is `config.json`'s
/// `text_config.pad_token_id`: HF's `Siglip2TextConfig` declares
/// `pad_token_id = 1` as a class default inherited from SigLIP1, which the
/// real tokenization path never uses — the exact trap this resolution avoids.
#[cfg(feature = "siglip2-naflex")]
fn read_text_pad_token_id(dir: &std::path::Path) -> Option<u32> {
  let tokenizer_config = read_bounded_json(&dir.join("tokenizer_config.json"));

  // 1. The pad token string: tokenizer_config.json first, else
  //    special_tokens_map.json (read only when needed).
  let pad_token = tokenizer_config
    .as_ref()
    .and_then(|cfg| token_content(cfg.get("pad_token")?).map(str::to_owned))
    .or_else(|| {
      let special = read_bounded_json(&dir.join("special_tokens_map.json"))?;
      token_content(special.get("pad_token")?).map(str::to_owned)
    })?;

  // 2. Resolve the string to its id via `added_tokens_decoder` ("<id>" →
  //    {"content": …}). The pad token is always an added special token, so
  //    this table carries it in every real HF checkpoint layout. A
  //    content-matching entry whose key is not a valid u32 id is SKIPPED
  //    (continue) rather than aborting the scan: in this best-effort reader a
  //    corrupt extra entry must not shadow a legitimate numeric binding later
  //    in the table (real HF tables have unique contents under numeric keys,
  //    so the skip only ever matters for pathological metadata).
  let decoder = tokenizer_config.as_ref()?.get("added_tokens_decoder")?;
  let entries = decoder.as_object()?;
  for (id_str, entry) in entries {
    if token_content(entry) == Some(pad_token.as_str())
      && let Ok(id) = id_str.parse::<u32>()
    {
      return Some(id);
    }
  }
  None
}

/// Read `path` as JSON with the [`MAX_TOKENIZER_METADATA_BYTES`] bounded-read
/// discipline (post-open regular-file check + `Read::take` cap). Any failure —
/// absent file, not a regular file, I/O error, over-cap size, malformed JSON —
/// is `None`: this feeds the best-effort [`read_text_pad_token_id`] only.
#[cfg(feature = "siglip2-naflex")]
fn read_bounded_json(path: &std::path::Path) -> Option<serde_json::Value> {
  use std::io::Read;

  // `O_NONBLOCK | O_CLOEXEC` so a planted FIFO cannot hang the open (the same
  // open discipline as the factory's `config.json` reader).
  #[cfg(unix)]
  let file = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(path)
      .ok()?
  };
  #[cfg(not(unix))]
  let file = std::fs::File::open(path).ok()?;

  // Reject a non-regular file (FIFO/device/directory) after the open.
  if !file.metadata().ok()?.is_file() {
    return None;
  }
  let mut bytes = Vec::new();
  file
    .take(MAX_TOKENIZER_METADATA_BYTES + 1)
    .read_to_end(&mut bytes)
    .ok()?;
  if bytes.len() as u64 > MAX_TOKENIZER_METADATA_BYTES {
    // Over-cap: a real tokenizer metadata file is far smaller; treat a planted
    // oversized file as absent rather than parsing a truncated prefix.
    return None;
  }
  serde_json::from_slice(&bytes).ok()
}

/// The token string of an HF special-token JSON value, accepting both shapes
/// the HF tokenizer files use: a plain string (`"<pad>"`) or an
/// `AddedToken`-style object (`{"content": "<pad>", …}`) — the same two shapes
/// the tokenizer wrapper's `cfg_str` handles. Anything else is `None`.
#[cfg(feature = "siglip2-naflex")]
fn token_content(value: &serde_json::Value) -> Option<&str> {
  match value {
    serde_json::Value::String(s) => Some(s.as_str()),
    serde_json::Value::Object(o) => o.get("content")?.as_str(),
    _ => None,
  }
}

/// The reserved (non-layer) keys of a `quantization` block — the three
/// [`crate::lm::quant::Quantization`] scalars plus the legacy HF/MLX-community
/// interop tags. Mirrors the [`PerLayerQuantization`] deserializer's `RESERVED`
/// list (`quant.rs`) so the two agree on which entries are per-layer overrides
/// (everything NOT in this set) versus the global spec's own scalars.
#[cfg(feature = "siglip2-naflex")]
const QUANT_RESERVED_KEYS: &[&str] = &[
  "group_size",
  "bits",
  "mode",
  "quant_method",
  "linear_class",
  "quantization_mode",
];

/// ONE quantization-spec object (`{ mode?, group_size?, bits? }`), deserialized
/// **strictly** by serde so the integer-range and type validation is serde's,
/// not a hand-rolled `serde_json::Value` walk.
///
/// All three fields are optional (a missing key is `None`); the per-mode
/// resolution ([`resolve_quant_spec`]) fills an absent / falsy `group_size` /
/// `bits` from the [`crate::lm::convert::defaults_for_mode`] table. Each field is
/// read through its own `Deserialize` impl ([`strict_field`]) so a failure can be
/// mapped to a field-named typed [`Error`]:
///
/// - `group_size` / `bits` are `Option<i32>` — **serde's** `i32` deserialization
///   rejects any JSON integer outside `i32` range whether the checkpoint wrote it
///   as an `i64` (`2147483648`) OR a `u64` past `i64::MAX`
///   (`9223372036854775808`), and rejects a float / string / bool. A JSON `null`
///   (or an absent key) deserializes to `None`. This is the edge the old
///   `Value::as_i64` walk missed: `as_i64` silently returns `None` for a `u64`
///   past `i64::MAX`, collapsing a corrupt oversized literal to the per-mode
///   default instead of rejecting it.
/// - `mode` is `Option<QuantMode>` — serde rejects an unrecognized scheme tag
///   (mapped to [`Error::UnknownEnumValue`]); a missing tag is `None`, resolved to
///   [`QuantMode::Affine`] (swift's `_mode ?? .affine`).
///
/// Any other key in the object (a per-layer path mistakenly nested, etc.) is
/// ignored by the struct deserialize — only the three scheme fields are read.
#[cfg(feature = "siglip2-naflex")]
#[derive(Debug)]
struct QuantSpec {
  mode: Option<crate::lm::quant::QuantMode>,
  group_size: Option<i32>,
  bits: Option<i32>,
}

/// Strictly deserialize ONE field's [`serde_json::Value`] through `T`'s own
/// `Deserialize` impl, mapping a serde failure to a field-named typed [`Error`].
///
/// This is the single place serde owns the integer-range + type validation: `T`
/// is `Option<i32>` for `group_size` / `bits` and `Option<QuantMode>` for `mode`,
/// so serde rejects an out-of-range integer, a wrong JSON type, or an unrecognized
/// enum tag — there is no manual `as_i64` / `as_u64`. An absent key short-circuits
/// to the type's [`Default`] (`None` for the `Option` fields). The failure is
/// classified WITHOUT matching serde's message text:
///
/// - an unrecognized enum tag (the value is a string but `T` rejected it) →
///   [`Error::UnknownEnumValue`] listing the recognized scheme tags;
/// - an out-of-range integer (the value IS a JSON integer but `T` rejected it) →
///   [`Error::OutOfRange`] naming the field and the offending magnitude;
/// - any other rejected type (a float, a string for an integer field, a bool, an
///   array) → [`Error::Parse`] naming the field.
///
/// The classification reads the rejected `Value`'s JSON kind (integer vs string
/// vs other), not serde's error string, so it is robust to serde's wording.
#[cfg(feature = "siglip2-naflex")]
fn strict_field<T>(
  obj: &serde_json::Map<String, serde_json::Value>,
  field: &'static str,
  is_enum: bool,
) -> Result<T>
where
  T: serde::de::DeserializeOwned + Default,
{
  use serde::de::IntoDeserializer;

  use crate::{error::UnknownEnumValuePayload, lm::quant::QuantMode};

  /// The recognized scheme tags (`QuantMode::as_str`), for the typed
  /// unknown-mode error's suggestion list.
  const KNOWN_MODES: &[&str] = &[
    QuantMode::Affine.as_str(),
    QuantMode::Mxfp4.as_str(),
    QuantMode::Mxfp8.as_str(),
    QuantMode::Nvfp4.as_str(),
  ];

  let Some(value) = obj.get(field) else {
    // An absent key is the type's default (`None` for the `Option` fields) — the
    // per-mode resolution fills it, exactly like a JSON `null`.
    return Ok(T::default());
  };
  // Deserialize through `T::deserialize` so serde performs the integer-range +
  // type + enum-tag validation. `into_deserializer` is infallible (it clones the
  // value into an owned deserializer) and never errors itself, so any error here
  // is `T` rejecting the value.
  T::deserialize(value.clone().into_deserializer()).map_err(|e: serde_json::Error| {
    if is_enum && value.is_string() {
      // A string the enum did not recognize — a present-but-unknown scheme tag.
      // Guessing (e.g. silently `affine`) would resolve the WRONG per-mode
      // `(group_size, bits)`, so reject.
      Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        "siglip2 parse_quantization: quantization `mode`",
        smol_str::format_smolstr!("{value}"),
        KNOWN_MODES,
      ))
    } else if value.is_i64() || value.is_u64() {
      // The value IS a JSON integer that serde rejected — it does not fit the
      // target integer range. (`is_u64` catches a magnitude past `i64::MAX` that
      // a plain `as_i64` walk would silently drop to the default.) A corrupt /
      // hostile oversized literal must surface, not collapse to the per-mode
      // default (masking the declared scheme) nor truncate.
      Error::OutOfRange(OutOfRangePayload::new(
        field,
        "siglip2 parse_quantization: quantization value out of range, must fit i32",
        smol_str::format_smolstr!("{value}"),
      ))
    } else {
      // Any other rejected JSON type (a float, a string for an integer field, a
      // bool, an array) — a malformed value for this field.
      Error::Parse(ParsePayload::new(
        field,
        "siglip2 parse_quantization: quantization scalar",
        e,
      ))
    }
  })
}

/// Strictly deserialize ONE quant-spec object and resolve its `(group_size, bits,
/// mode)` through the shared per-mode default table — the single typed path
/// applied uniformly to the top-level global spec AND every per-layer override.
///
/// Each of `mode` / `group_size` / `bits` is read by [`strict_field`] (serde owns
/// the range + type + enum validation; an absent key is `None`). The resolved
/// values then route through [`crate::lm::convert::defaults_for_mode`] for the
/// falsy fallback — an absent key, an explicit `null`, or a `0` all fall back to
/// the per-mode default (mlx-lm's `value or default`, `utils.py:808`: `0` is
/// falsy), while a present positive value is preserved. A present NEGATIVE
/// `group_size` / `bits` is the one departure from python's truthiness: python
/// keeps a negative (it is truthy), but a negative `group_size` / `bits` is
/// invalid for the `quantized_matmul` kernel (mlx's `quantize` asserts positive),
/// so it would let a checkpoint load under different quant params than declared —
/// a silent malformed-numeric load. It is rejected here as a typed
/// [`Error::OutOfRange`] before the falsy resolution runs, so the net contract is:
/// negative → error; `{absent, null, 0}` → per-mode default; positive → override.
/// The `mode` is resolved FIRST (a missing tag → [`QuantMode::Affine`], swift's
/// `_mode ?? .affine`) so the injected defaults are the per-mode ones — a blanket
/// `group_size = 64` would mis-resolve an `mxfp4` / `nvfp4` / `mxfp8` spec that
/// relies on its mode default.
///
/// # Errors
/// - [`Error::UnknownEnumValue`] if `mode` is present but is not a recognized
///   scheme tag;
/// - [`Error::OutOfRange`] if `group_size` or `bits` is a present JSON integer
///   that does not fit `i32`, or is a present negative value (invalid for the
///   quantization kernel);
/// - [`Error::Parse`] if `group_size` or `bits` is a present non-integer JSON
///   scalar (a float, a string, a bool, an array).
#[cfg(feature = "siglip2-naflex")]
fn resolve_quant_spec(
  obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<crate::lm::quant::Quantization> {
  use crate::lm::{
    convert::defaults_for_mode,
    quant::{QuantMode, Quantization},
  };

  let spec = QuantSpec {
    mode: strict_field(obj, "mode", true)?,
    group_size: strict_field(obj, "group_size", false)?,
    bits: strict_field(obj, "bits", false)?,
  };
  // A PRESENT negative value would survive python's `value or default` (a
  // negative is truthy) but is invalid for the quantization kernel, so it must
  // not reach `defaults_for_mode` (whose `> 0` filter would silently rewrite it
  // to the per-mode default — a malformed-numeric load under a quant param the
  // config never declared). Reject it as a typed error before resolution. The
  // remaining falsy cases (`absent`, `null`, `0`) still fall back below.
  for (field, value) in [("group_size", spec.group_size), ("bits", spec.bits)] {
    if let Some(v) = value
      && v < 0
    {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        field,
        "siglip2 parse_quantization: quantization value must be non-negative",
        smol_str::format_smolstr!("{v}"),
      )));
    }
  }
  // A missing `mode` defaults to affine (swift's `_mode ?? .affine`) so the
  // per-mode `(group_size, bits)` defaults below are the affine ones.
  let mode = spec.mode.unwrap_or(QuantMode::Affine);
  let (group_size, bits) = defaults_for_mode(mode, spec.group_size, spec.bits);
  Ok(Quantization {
    group_size,
    bits,
    mode,
  })
}

/// Parse the optional `quantization` block from a SigLIP2 `config.json` into a
/// [`PerLayerQuantization`] — the embeddings analogue of mlx-embeddings'
/// `utils.py` `config.get("quantization")` (with the HF `quantization_config`
/// fallback), and the structural twin of `crate::audio::load::apply_quantization`
/// (kept local here because `siglip2-naflex` does not enable the `audio`
/// feature).
///
/// Prefers a non-null top-level `"quantization"` block; falls back to a non-null
/// `"quantization_config"` (the HF post-quantize artifact); returns `Ok(None)`
/// for a dense checkpoint (neither key present, or both null) — the dense-model
/// no-op. Each quant-spec object (the global spec AND every per-layer override)
/// is resolved by the single strict [`resolve_quant_spec`] path: serde owns the
/// `group_size` / `bits` integer-range + type validation, then the shared
/// [`crate::lm::convert::defaults_for_mode`] table — `affine` → `(64, 4)`,
/// `mxfp4` → `(32, 4)`, `nvfp4` → `(16, 4)`, `mxfp8` → `(32, 8)` — resolves an
/// absent / falsy value. A blanket `group_size = 64` would mis-resolve an
/// `mxfp4` / `nvfp4` / `mxfp8` spec that relies on its mode default, so the mode
/// gates the default. Resolution mirrors mlx-lm's `value or default` truthiness
/// (`utils.py:808`): an absent key, an explicit `null`, **and** a present `0` all
/// fall back to the per-mode default (`0` is falsy), while a present positive
/// value is preserved. A present NEGATIVE `group_size` / `bits` (invalid for the
/// quantization kernel) is the lone departure from python's truthiness — it is a
/// typed [`Error::OutOfRange`] rather than a silent collapse to the per-mode
/// default, so a checkpoint never loads under a quant param the config never
/// declared.
///
/// That ONE strict path is applied UNIFORMLY: to the top-level global spec AND to
/// every per-layer override object (each non-reserved object-valued entry — the
/// nested `{ mode?, group_size?, bits? }` schemes). There is no
/// top-level-vs-per-layer gap: a per-layer override's falsy / absent
/// `group_size` / `bits` resolves through the same per-mode contract as the
/// global spec's, and a per-layer negative is rejected by the same guard. A
/// per-layer `false` is the [`QuantizationOption::Skip`]
/// sentinel; a per-layer `true` is ignored (swift `if !f`); any other per-layer
/// scalar (a number / string / null / array) is a typed [`Error::OutOfRange`] —
/// matching the [`PerLayerQuantization`] deserializer's contract.
///
/// # Errors
/// - [`Error::Parse`] if the config JSON does not parse, or a spec object's
///   `group_size` / `bits` is a present non-integer JSON scalar (a float, a
///   string, a bool, an array);
/// - [`Error::OutOfRange`] if the `quantization` value is present but is not a
///   JSON object, if a per-layer override is an unrecognized non-object scalar, if
///   any spec object's `group_size` / `bits` is a present JSON integer that does
///   not fit `i32` (an oversized literal — including a `u64` past `i64::MAX` — is
///   rejected rather than silently collapsed to the per-mode default or
///   truncated), or if any spec object's `group_size` / `bits` is a present
///   negative value (invalid for the quantization kernel, rejected rather than
///   silently collapsed to the per-mode default);
/// - [`Error::UnknownEnumValue`] if a spec object's `mode` is present but is not a
///   recognized scheme tag (`affine` / `mxfp4` / `mxfp8` / `nvfp4`) — the per-mode
///   default cannot be resolved, so this is rejected rather than guessed.
#[cfg(feature = "siglip2-naflex")]
fn parse_quantization(config_json: &str) -> Result<Option<PerLayerQuantization>> {
  use serde_json::Value;

  use crate::lm::quant::QuantizationOption;

  let value: Value = serde_json::from_str(config_json).map_err(|e| {
    Error::Parse(ParsePayload::new(
      "siglip2 parse_quantization: config",
      "JSON",
      e,
    ))
  })?;

  // Prefer a non-null top-level `"quantization"`, else a non-null
  // `"quantization_config"` (the HF artifact key), else dense (no-op).
  let block = match value.get("quantization") {
    Some(b) if !b.is_null() => b,
    _ => match value.get("quantization_config") {
      Some(b) if !b.is_null() => b,
      _ => return Ok(None),
    },
  };

  let Value::Object(map) = block else {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "siglip2 parse_quantization: quantization block",
      "must be a JSON object",
      smol_str::format_smolstr!("{block:?}"),
    )));
  };

  // (1) The top-level global spec — this object's own `mode` / `group_size` /
  //     `bits`, resolved through the single strict path (per-layer keys in the
  //     same object are ignored by the spec deserialize).
  let global = resolve_quant_spec(map)?;

  // (2) Each per-layer override — every non-reserved entry. The map is reserved
  //     fallibly (typed `AllocFailure`) and each kept key is cloned through the
  //     fallible String path, mirroring `reproject_quant_keys` / `strip_prefix`.
  let mut per_layer: HashMap<String, QuantizationOption> = HashMap::new();
  reserve_or_error(
    &mut per_layer,
    "Siglip2 parse_quantization: per-layer overrides",
    map.len(),
  )?;
  for (key, slot) in map {
    if QUANT_RESERVED_KEYS.contains(&key.as_str()) {
      continue;
    }
    // The same per-layer value contract the `PerLayerQuantization` deserializer
    // enforces: `false` is the skip sentinel, a `true` is ignored (swift's
    // `if !f` falls through), an object is a resolved override, and any other
    // scalar is a typed error.
    let opt = match slot {
      Value::Bool(false) => QuantizationOption::Skip,
      Value::Bool(true) => continue,
      Value::Object(spec) => QuantizationOption::Quantize(resolve_quant_spec(spec)?),
      other => {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "siglip2 parse_quantization: per-layer override",
          "must be `false` or a quantization object",
          smol_str::format_smolstr!("{key}: {other}"),
        )));
      }
    };
    let owned = fallible_clone_str("parse_quantization: per-layer key", key)?;
    per_layer.insert(owned, opt);
  }

  Ok(Some(PerLayerQuantization::new(Some(global), per_layer)))
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
        // The checkpoint tokenizer's `<pad>` id (Gemma `<pad> = 0` by default;
        // refined from the loaded tokenizer directory's metadata by the
        // factory `constructor`). NOT the EOS: a short prompt's pooled last
        // slot is a pad position, and the reference processor fills it with
        // `<pad>`, not `<eos>`.
        pad_token_id: self.text_pad_token_id,
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
