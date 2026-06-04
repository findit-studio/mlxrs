//! CLAP-HTSAT-unfused — dual-tower audio+text embeddings model
//! (`laion/clap-htsat-unfused`).
//!
//! CLAP is a contrastive dual-tower model: an HTSAT Swin-Transformer audio
//! encoder and a RoBERTa text encoder each map their input to a 512-dim vector
//! in one shared, L2-normalized space, so `cosine(audio, text)` ranks
//! audio↔text relevance (the zero-shot-classification primitive). This port
//! targets the **unfused** checkpoint (`enable_fusion=False`): a single fixed
//! 10 s mel window, no feature-fusion branch.
//!
//! Sources:
//! - HF `transformers` `ClapModel` (`laion/clap-htsat-unfused`) — the
//!   authoritative architecture (`ClapAudioModel` + `ClapTextModel` + two
//!   projection MLPs + L2-normalize).
//! - The Findit-AI `textclap` crate — owns the mel front-end + the I/O
//!   contract verbatim, and its committed `golden_mel.npy` /
//!   `filterbank_row_*.npy` fixtures pin the mel front-end numerically (the
//!   [`mel`] oracle).
//!
//! ## Public surface ([`ClapModel`])
//! The full dual-tower model wires the configuration
//! ([`config::ClapConfig`] = [`config::ClapAudioConfig`] +
//! [`config::ClapTextConfig`] + projection dims), the mel / spectrogram
//! front-end ([`mel`]), the RoBERTa **text tower** ([`text::ClapTextModel`] —
//! embeddings + post-norm encoder + CLS pooling + the `pooler` (dense+tanh) +
//! the text projection + L2-normalize), and the HTSAT Swin **audio tower**
//! ([`audio::HtsatAudioTower`] —
//! the `reshape_mel2img` mel→image fold, the patch-embed stem, the four Swin
//! stages, and the token-semantic mean-pool producing the `(B, 768)` pooled
//! audio feature) into [`ClapModel`], adding the **audio projection** (the CLAP
//! `ClapProjectionLayer`: `Linear(768 → 512) → ReLU → Linear(512 → 512)`) over
//! the pooled audio feature.
//! [`ClapModel::embed_audio`] runs `extract_mel → audio tower → audio_projection
//! → L2-normalize`, [`ClapModel::embed_text`] reuses the text tower's
//! projected+normalized embedding, and [`ClapModel::classify`] computes the
//! zero-shot cosine top-k (mirroring `textclap`'s `Clap::classify_all`: cosine
//! per label, sorted descending, stable input-order tie-break). It implements the
//! golden embedding seams — the model-implemented
//! [`crate::embeddings::TextEmbedder`] (the text tower's), [`Embed<AudioInput>`],
//! and the [`crate::embeddings::EmbeddingModel`] umbrella — and registers into the
//! [`crate::embeddings::EmbeddingModelTypeRegistry`] via [`register`] under
//! `model_type = "clap"`. [`sanitize`] rewrites a raw `laion/clap-htsat-unfused`
//! checkpoint into the per-tower layout; the end-to-end checkpoint-parity test
//! (against `textclap`'s ONNX outputs) lands in a separate PR.
//!
//! CLAP's contrastive inference path is **plain cosine** — there is no
//! `logit_scale` applied at classify time (HF's `logit_scale_a` / `logit_scale_t`
//! are train-time parameters; `ClapModel.get_audio_features` /
//! `get_text_features` only L2-normalize, and zero-shot classification is the
//! cosine of those unit embeddings), unlike SigLIP's `logit_scale` / `logit_bias`.
//!
//! ## Reuse
//! The mel front-end reuses [`crate::audio::dsp`] —
//! [`mel_filter_bank_scaled`](crate::audio::dsp::mel_filter_bank_scaled)
//! (Slaney scale + Slaney normalization), [`stft`](crate::audio::dsp::stft),
//! and the framing / rfft machinery — rather than re-implementing the DSP. The
//! configuration mirrors the [`crate::embeddings::siglip2_naflex`] dual-tower
//! config precedent (serde `#[serde(default)]` + a `validate()` that pins
//! every architecture-defining field before any tensor is built).

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod audio;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod config;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod mel;

#[cfg(feature = "clap")]
mod shared;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub mod text;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use audio::HtsatAudioTower;

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use config::{ClapAudioConfig, ClapConfig, ClapTextConfig};

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use mel::{MelFrontEnd, N_MELS, T_FRAMES};

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub use text::ClapTextModel;

#[cfg(feature = "clap")]
use std::collections::HashMap;

#[cfg(feature = "clap")]
use crate::{
  array::Array,
  embeddings::{
    Embed, Embedding, EmbeddingModel, EmbeddingModelConstructor, EmbeddingModelTypeRegistry,
    LoadedEmbeddingModel, SWIFT_L2_EPS, StPoolingConfig, TextEmbedder,
    clap::{
      audio::HtsatAudioTower as AudioTower,
      mel::MelFrontEnd as Mel,
      shared::{ClapProjectionLayer, fallible_clone_str, reshape_patch_weight},
      text::ClapTextModel as TextModel,
    },
    cosine_similarity, l2_normalize_eps,
  },
  error::{Error, OutOfRangePayload, Result},
  lm::quant::PerLayerQuantization,
  model_validation::{insert_unique, reserve_or_error},
  ops,
};

#[cfg(feature = "clap")]
use smol_str::format_smolstr;

/// The top-level architecture id this model registers under (`config.json`
/// `model_type`).
#[cfg(feature = "clap")]
pub const MODEL_TYPE: &str = "clap";

/// HF prefix for the audio tower's encoder sub-tree
/// (`ClapModel.audio_model.audio_encoder.*`). `sanitize` keeps it; the assembly
/// strips it to reach the [`HtsatAudioTower`] keys (`batch_norm.*` /
/// `patch_embed.*` / `layers.{stage}.*` / `norm.*`).
#[cfg(feature = "clap")]
const AUDIO_ENCODER_PREFIX: &str = "audio_model.audio_encoder.";

/// HF prefix for the text tower (`ClapModel.text_model.*`). The assembly strips
/// it to reach the [`ClapTextModel`] embeddings + encoder keys (`embeddings.*` /
/// `encoder.layer.{i}.*`).
#[cfg(feature = "clap")]
const TEXT_MODEL_PREFIX: &str = "text_model.";

/// HF prefix for the audio projection head (`ClapModel.audio_projection.*`).
#[cfg(feature = "clap")]
const AUDIO_PROJECTION_PREFIX: &str = "audio_projection";

/// HF prefix for the text projection head (`ClapModel.text_projection.*`). The
/// [`ClapTextModel`] owns the text projection, so this stays in the text sub-map
/// verbatim (it is a sibling of `text_model.*`, NOT nested under it).
#[cfg(feature = "clap")]
const TEXT_PROJECTION_PREFIX: &str = "text_projection";

/// CLAP-HTSAT-unfused dual-tower audio+text embeddings model
/// (`laion/clap-htsat-unfused`).
///
/// See the [module docs](self) for the architecture and public API. Built via
/// [`from_weights`](Self::from_weights) over a [`sanitize`]d checkpoint; embed
/// with [`embed_audio`](Self::embed_audio) / [`embed_text`](Self::embed_text)
/// and zero-shot classify with [`classify`](Self::classify).
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub struct ClapModel {
  /// The parsed, validated configuration (the SigLIP2 precedent of keeping the
  /// config on the model — exposes it via [`config`](Self::config) and keeps the
  /// dual-tower dims with the model).
  config: ClapConfig,
  /// The shared mel / spectrogram front-end (precomputed window + filterbank).
  mel: Mel,
  /// The HTSAT Swin audio tower → the `(B, 768)` pooled audio feature.
  audio: AudioTower,
  /// The RoBERTa text tower (owns its CLS pooling + `pooler` (dense+tanh) +
  /// `text_projection` + L2-normalize, the [`TextEmbedder`] impl).
  text: TextModel,
  /// The audio projection head (`Linear(768 → 512) → ReLU → Linear(512 → 512)`)
  /// over the pooled audio feature.
  audio_projection: ClapProjectionLayer,
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl ClapModel {
  /// L2-normalization eps for the audio/text embeddings. CLAP's reference uses
  /// PyTorch `F.normalize` (a `1e-12` floor); the swift `1e-12`
  /// ([`SWIFT_L2_EPS`]) matches it. The embeddings are f32 here so the choice is
  /// immaterial to the result, but it pins the intent (mirroring the SigLIP2
  /// `NORMALIZE_EPS` precedent).
  const NORMALIZE_EPS: f32 = SWIFT_L2_EPS;

  /// Build a model from a parsed [`ClapConfig`] and the **sanitized** weight map
  /// (run [`sanitize`] first).
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions before it is stored or fed to any op (the shared
  /// `take_shaped` / `check_quantized_shape` discipline).
  pub fn from_weights(config: ClapConfig, weights: HashMap<String, Array>) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build a model from a parsed [`ClapConfig`], the **sanitized** weight map,
  /// and an optional parsed quantization config — the quantization-aware
  /// analogue of [`from_weights`](Self::from_weights) (which is this with
  /// `quantization == None`).
  ///
  /// When a layer's `<prefix>.weight` carries the sibling `<prefix>.scales`
  /// tensor, that `nn.Linear` / `nn.Embedding` is built quantized (the
  /// `.scales`-sibling `class_predicate` signal), with `(group_size, bits, mode)`
  /// resolved per layer from `quantization`. A `.scales`-bearing layer that
  /// resolves no scheme is a typed [`Error::InvariantViolation`] (raised by the
  /// shared `QuantLinear`), never a silent dense fall-through. A dense checkpoint
  /// (no `.scales`) builds byte-identically whether or not a `quantization`
  /// config is threaded.
  ///
  /// The weight map is split per tower: the `audio_model.audio_encoder.*` keys
  /// are stripped to the [`HtsatAudioTower`] namespace, the `text_model.*` keys
  /// to the [`ClapTextModel`] namespace (with the sibling `text_projection.*`
  /// kept verbatim, since the text tower owns its projection), and the
  /// `audio_projection.*` head is built directly.
  pub fn from_weights_quantized(
    config: ClapConfig,
    mut weights: HashMap<String, Array>,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation (the towers re-validate too): bounds every dim
    // and pins the unfused contract before any tensor is built.
    config.validate()?;

    // The audio-tower sub-map: strip `audio_model.audio_encoder.`.
    let mut audio_weights = strip_prefix(&mut weights, AUDIO_ENCODER_PREFIX)?;
    // The text-tower sub-map: strip `text_model.`, then fold in the verbatim
    // top-level `text_projection.*` keys (the text tower owns its projection,
    // which is a sibling of `text_model`, not nested under it).
    let mut text_weights = strip_prefix(&mut weights, TEXT_MODEL_PREFIX)?;
    move_prefixed(&mut weights, &mut text_weights, TEXT_PROJECTION_PREFIX)?;

    let audio = AudioTower::from_weights_quantized(&config, &mut audio_weights, quantization)?;
    let text = TextModel::from_weights_quantized(&config, &mut text_weights, quantization)?;

    // The audio projection head: `Linear(audio_hidden → projection_dim) → ReLU →
    // Linear(projection_dim → projection_dim)`. Its in-dim is the pooled audio
    // feature width (`audio_config.hidden_size`), its out-dim `projection_dim`.
    let audio_projection = ClapProjectionLayer::from_weights(
      AUDIO_PROJECTION_PREFIX,
      &mut weights,
      config.audio_config.hidden_size,
      config.projection_dim,
      quantization,
    )?;

    Ok(Self {
      config,
      mel: Mel::new()?,
      audio,
      text,
      audio_projection,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &ClapConfig {
    &self.config
  }

  /// Encode a 48 kHz mono `&[f32]` waveform to its L2-normalized audio embedding
  /// `(1, projection_dim)`.
  ///
  /// `extract_mel(samples) → HTSAT tower → audio_projection → L2-normalize`,
  /// mirroring `ClapModel.get_audio_features` (the pooled audio feature, the
  /// audio projection, then `F.normalize`). The waveform is repeat-padded /
  /// head-truncated to the fixed 10 s window by the front-end.
  pub fn embed_audio(&self, samples: &[f32]) -> Result<Embedding> {
    let mel = self.mel.extract(samples)?; // (1, 1, T_FRAMES, N_MELS)
    self.embed_mel(&mel)
  }

  /// Encode a **preprocessed** `(B, 1, T_FRAMES, N_MELS)` log-mel spectrogram (an
  /// [`MelFrontEnd::extract`] output) to its L2-normalized audio embedding
  /// `(B, projection_dim)`.
  ///
  /// The mel-preprocessed boundary of [`embed_audio`](Self::embed_audio) (its
  /// `extract_mel` tail): HTSAT tower → audio projection → L2-normalize. The
  /// [`Embed<AudioInput>`] impl routes through here.
  pub fn embed_mel(&self, mel: &Array) -> Result<Embedding> {
    let feature = self.audio.forward(mel)?; // (B, audio_hidden=768)
    let projected = self.audio_projection.forward(&feature)?; // (B, projection_dim)
    Ok(Embedding::new(l2_normalize_eps(
      &projected,
      Self::NORMALIZE_EPS,
    )?))
  }

  /// Encode a `(batch, seq_len)` i32 token-id batch (+ its `(batch, seq_len)`
  /// `{0,1}` attention mask) to L2-normalized text embeddings
  /// `(batch, projection_dim)`.
  ///
  /// Delegates to the text tower's [`encode_text`](ClapTextModel::encode_text)
  /// (RoBERTa → CLS → `pooler` (dense+tanh) → `text_projection` → L2-normalize),
  /// which already returns the projected, normalized embedding
  /// (`ClapModel.get_text_features`).
  pub fn embed_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Embedding> {
    Ok(Embedding::new(
      self.text.encode_text(input_ids, attention_mask)?,
    ))
  }

  /// The shared mel / spectrogram front-end.
  #[inline(always)]
  pub fn mel_front_end(&self) -> &MelFrontEnd {
    &self.mel
  }

  /// Cosine similarity between two already-L2-normalized embedding vectors. For
  /// unit vectors cosine == dot (`textclap`'s `Embedding::cosine == dot`), so
  /// this is the per-label score [`classify`](Self::classify) ranks. Each input
  /// is a single `(dim,)` row (the embeddings are `(1, dim)`; pass `audio.array()`
  /// / `text.array()` reshaped to rank-1, which [`classify`](Self::classify) does).
  pub fn cosine(&self, a: &Array, b: &Array) -> Result<f32> {
    cosine_similarity(a, b)
  }

  /// Top-k zero-shot classification — the cosine of the audio embedding against
  /// each label's text embedding, sorted descending, top-k.
  ///
  /// Ports `textclap`'s `Clap::classify` / `classify_all` semantics: an empty
  /// `labels` or `k == 0` returns an empty `Vec`; otherwise the audio is embedded
  /// once, each label is embedded and scored by cosine
  /// ([`cosine`](Self::cosine) — for the unit embeddings this equals the dot
  /// `textclap` uses), the `(index, score)` pairs are sorted **descending by
  /// score with a stable input-order tie-break** (Rust's `sort_by` is stable, so
  /// equal scores keep their input order — exactly `textclap`'s stable sort), and
  /// the top `k.min(labels.len())` are returned as `(label_index, score)`.
  ///
  /// `samples` is a 48 kHz mono waveform; `label_ids` / `label_masks` are the
  /// already-tokenized `(num_labels, seq_len)` id batch + `{0,1}` mask the caller
  /// builds from the labels via the [`TextEmbedder`] encode pipeline (the model
  /// owns tokenization but not the label→ids step, which the generic `encode`
  /// path drives). Returns `(index_into_labels, cosine_score)` so the caller maps
  /// back to its own label strings (the borrow `textclap`'s `LabeledScore` holds).
  pub fn classify(
    &self,
    samples: &[f32],
    label_ids: &Array,
    label_masks: &Array,
    k: usize,
  ) -> Result<Vec<(usize, f32)>> {
    let num_labels = classify_label_count(label_ids)?;
    // textclap: `if k == 0 || labels.is_empty() { return Vec::new() }`.
    if k == 0 || num_labels == 0 {
      return Ok(Vec::new());
    }

    // Embed the audio once, then each label's text, scoring by cosine. The
    // embeddings are `(1, dim)` / `(num_labels, dim)`; cosine is over rank-1
    // rows, so reshape the single audio embedding to `(dim,)` and take each
    // text row.
    let audio_embed = self.embed_audio(samples)?;
    let audio_row = flatten_single_row(audio_embed.array())?; // (dim,)
    let text_embeds = self.embed_text(label_ids, label_masks)?; // (num_labels, dim)

    let mut scores: Vec<(usize, f32)> = Vec::new();
    reserve_or_error(&mut scores, "clap classify: per-label scores", num_labels)?;
    for i in 0..num_labels {
      let text_row = take_row(text_embeds.array(), i as i32)?; // (dim,)
      scores.push((i, self.cosine(&audio_row, &text_row)?));
    }

    // Sort descending by score; a stable sort keeps the input order on ties
    // (textclap's `sort_by(|a, b| b.score.partial_cmp(&a.score)...)`, stable).
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k.min(num_labels));
    Ok(scores)
  }

  /// `true` if every quantizable `Linear` across both towers + the audio
  /// projection loaded the quantized variant (test-only introspection).
  #[cfg(test)]
  pub(crate) fn all_linears_quantized(&self) -> bool {
    self.audio.all_swin_linears_quantized()
      && self.text.all_projections_quantized()
      && self.audio_projection.all_quantized()
  }
}

/// CLAP's audio input modality for [`Embed`]: a **preprocessed** `(B, 1,
/// T_FRAMES, N_MELS)` log-mel spectrogram (a [`MelFrontEnd::extract`] output).
///
/// Audio inputs are model-defined (a different audio model would use a different
/// preprocessed type), so CLAP owns this newtype — there is intentionally no
/// universal `dyn AudioEmbedder`. Reach the audio tower from a loaded
/// [`crate::embeddings::EmbeddingModel`] by downcasting to [`ClapModel`] and
/// calling [`embed_audio`](ClapModel::embed_audio) /
/// [`embed_mel`](ClapModel::embed_mel) (or this `Embed` impl), exactly as
/// SigLIP2's `Embed<ImageInput>` is reached.
#[cfg(feature = "clap")]
pub struct AudioInput<'a>(pub &'a Array);

/// Audio tower: a preprocessed `(B, 1, T_FRAMES, N_MELS)` mel → its
/// L2-normalized audio embedding.
#[cfg(feature = "clap")]
impl<'a> Embed<AudioInput<'a>> for ClapModel {
  type Output = Embedding;
  fn embed(&self, input: AudioInput<'a>) -> Result<Embedding> {
    self.embed_mel(input.0)
  }
}

/// CLAP answers the load factory umbrella's universal text capability (the
/// RoBERTa text tower). Its audio tower (a model-defined input) + `classify` are
/// reached by concrete downcast via
/// [`as_any`](crate::embeddings::EmbeddingModel::as_any). CLAP does **not** claim
/// [`Contrastive`](crate::embeddings::Contrastive): its public contrastive
/// surface is the zero-shot [`classify`](ClapModel::classify) (plain cosine, no
/// `logit_scale`), not a SigLIP-style `logit_scale` / `logit_bias` similarity.
#[cfg(feature = "clap")]
impl EmbeddingModel for ClapModel {
  fn as_text_embedder(&self) -> Option<&dyn TextEmbedder> {
    // The text tower implements `TextEmbedder`; expose it as the model's text
    // seam so the generic `encode` pipeline drives label/query embedding.
    Some(&self.text)
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

/// Rewrite a raw `laion/clap-htsat-unfused` checkpoint into the layout
/// [`ClapModel::from_weights`] loads — the analogue of HF `ClapModel.sanitize`
/// (combined with the towers' buffer drops), mirroring the SigLIP2 `sanitize`
/// precedent.
///
/// Rules (applied per `(key, value)`):
/// 1. **Drop non-parameter buffers.** A `position_ids` buffer (HF
///    `ClapTextEmbeddings.register_buffer("position_ids", …)`) and a
///    `relative_position_index` buffer (HF
///    `ClapAudioSelfAttention.register_buffer("relative_position_index", …)`) are
///    recomputed deterministically in Rust (`position_ids_from_ids` /
///    `relative_position_index`), so they are skipped rather than loaded.
/// 2. **Transpose the patch-embed Conv2d weight to channels-last.** mlxrs
///    `conv2d` is NHWC (weight `(C_out, KH, KW, C_in)`) while HF's `Conv2d`
///    weight is `(C_out, C_in, KH, KW)`; the
///    `audio_model.audio_encoder.patch_embed.proj.weight` is transposed
///    `[0, 2, 3, 1]` here (the SigLIP2 `reshape_patch_weight` precedent), and the
///    image is fed `(B, H, W, 1)` by the tower. A weight already in NHWC layout is
///    left unchanged (the transpose is keyed on the rank-4 NCHW shape).
/// 3. **Reject a duplicate destination key** with [`Error::KeyCollision`] (via
///    [`insert_unique`]) rather than letting an arbitrary (per-run
///    nondeterministic) survivor silently overwrite the other.
///
/// The towers' own prefixes (`audio_model.audio_encoder.*` / `text_model.*` /
/// `audio_projection.*` / `text_projection.*`, plus the unused `logit_scale_a` /
/// `logit_scale_t`) are otherwise kept verbatim; [`ClapModel::from_weights`]
/// strips them per tower. Every rewritten key is built through the fallible
/// `fallible_concat` path (typed [`Error::AllocFailure`]), and the destination
/// map reserve + each `insert_unique` slot are fallible too.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  // The destination map is sized by the (checkpoint-controlled) source key
  // count; reserve it fallibly (typed `AllocFailure`) rather than via
  // `HashMap::with_capacity`, which aborts under allocator pressure.
  let mut out: HashMap<String, Array> = HashMap::new();
  reserve_or_error(&mut out, "clap sanitize: destination map", weights.len())?;
  for (k, v) in weights {
    // 1. Drop the non-parameter buffers (recomputed in Rust).
    if k.contains("position_ids") || k.contains("relative_position_index") {
      continue;
    }

    // 2. Transpose the patch-embed Conv2d weight to channels-last NHWC. The key
    //    is unchanged; only the tensor layout is rewritten (and only if it is the
    //    rank-4 NCHW `(C_out, C_in, KH, KW)` HF shape — an already-NHWC weight is
    //    left as-is by `reshape_patch_weight`).
    let v = if k.ends_with("patch_embed.proj.weight") {
      reshape_patch_weight(&v, &k)?
    } else {
      v
    };

    // 3. Insert, rejecting a duplicate destination key with a typed error.
    insert_unique(&mut out, k, v, "clap sanitize")?;
  }
  Ok(out)
}

/// The [`EmbeddingModelConstructor`] that builds a [`ClapModel`] from a loaded
/// model directory, for registration into an [`EmbeddingModelTypeRegistry`].
///
/// Parses the raw `config.json`, [`sanitize`]s a *cheap clone* of the loaded
/// weight map (mlx [`Array`] is a refcounted handle, so the clone shares the
/// device buffers — no data copy), parses the optional `quantization` block, and
/// builds the model. The map reserve + each cloned key are fallible (typed
/// [`Error::AllocFailure`]).
///
/// The parsed `1_Pooling/config.json` (`_pooling`) is **ignored**: CLAP is a
/// dual-tower contrastive model that bakes its own fixed text pooling (CLS +
/// the RoBERTa `pooler` dense+tanh) + dynamic-right-pad scheme; it does not
/// consume a sentence-encoder pooling config.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub fn constructor() -> EmbeddingModelConstructor {
  Box::new(
    |loaded: &LoadedEmbeddingModel,
     _pooling: Option<&StPoolingConfig>|
     -> Result<Box<dyn EmbeddingModel>> {
      let config = ClapConfig::from_json(loaded.config_json())?;
      // mlx `Array` is a cheap refcounted handle; `try_clone` shares the device
      // buffer (no copy). Clone the loaded map into an owned, sanitizable map,
      // reserving fallibly (typed `AllocFailure`).
      let mut raw: HashMap<String, Array> = HashMap::new();
      reserve_or_error(
        &mut raw,
        "clap constructor: sanitizable weight-map clone",
        loaded.weights_ref().len(),
      )?;
      for (k, v) in loaded.weights_ref() {
        let key = fallible_clone_str("clap constructor: weight-map key clone", k)?;
        raw.insert(key, v.try_clone()?);
      }
      let weights = sanitize(raw)?;
      // Parse the optional `config.json` `quantization` block (the mlx-community
      // native key); a dense checkpoint has none and loads dense.
      let quantization = crate::audio::load::apply_quantization(loaded.config_json())?;
      let model = ClapModel::from_weights_quantized(config, weights, quantization.as_ref())?;
      Ok(Box::new(model))
    },
  )
}

/// Register [`ClapModel`] under [`MODEL_TYPE`] (`"clap"`) into `registry`,
/// returning any constructor it displaced.
///
/// The registry is the documented architecture extension point (per-model
/// architectures are not auto-registered); call this to enable loading a
/// `laion/clap-htsat-unfused` checkpoint through [`crate::embeddings::load`].
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub fn register(registry: &mut EmbeddingModelTypeRegistry) -> Option<EmbeddingModelConstructor> {
  registry.register(MODEL_TYPE, constructor())
}

// ═══════════════════════════════ free functions ════════════════════════════

/// Strip every key with `prefix` from `weights` into a new map (with the prefix
/// removed), leaving the non-matching keys in place — the SigLIP2 `strip_prefix`
/// precedent, used to split the sanitized dual-tower map into per-tower sub-maps.
///
/// Both the matched-key `Vec` and the destination `HashMap` are sized by the
/// (checkpoint-controlled) matching-key count, so each is reserved fallibly
/// (typed [`Error::AllocFailure`] via [`reserve_or_error`]). Both the full
/// matched key (needed to `remove` the entry) and the prefix-stripped
/// destination key are built through the fallible String path.
#[cfg(feature = "clap")]
fn strip_prefix(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
) -> Result<HashMap<String, Array>> {
  let matching = weights.keys().filter(|k| k.starts_with(prefix)).count();
  let mut keys: Vec<String> = Vec::new();
  reserve_or_error(&mut keys, "clap strip_prefix: matched key list", matching)?;
  for k in weights.keys().filter(|k| k.starts_with(prefix)) {
    keys.push(fallible_clone_str("clap strip_prefix: matched key", k)?);
  }
  let mut out: HashMap<String, Array> = HashMap::new();
  reserve_or_error(&mut out, "clap strip_prefix: per-tower sub-map", keys.len())?;
  for k in keys {
    if let Some(v) = weights.remove(&k) {
      let stripped = fallible_clone_str("clap strip_prefix: stripped key", &k[prefix.len()..])?;
      out.insert(stripped, v);
    }
  }
  Ok(out)
}

/// Move every key with `prefix` from `weights` into `dest` **verbatim** (the key
/// is kept, not stripped), leaving the non-matching keys in place. Used to fold
/// the top-level `text_projection.*` head into the text-tower sub-map (the text
/// tower owns its projection, and consumes `text_projection.*` unchanged).
///
/// Sized fallibly like [`strip_prefix`]; the moved keys are cloned through the
/// fallible String path. A key already present in `dest` is rejected with a typed
/// [`Error::KeyCollision`] (via [`insert_unique`]) — `sanitize` already
/// de-duplicates the source, so this is a defense-in-depth guard against a
/// `text_model.*` and a `text_projection.*` collision (they share no prefix, so
/// this never fires for a real checkpoint).
#[cfg(feature = "clap")]
fn move_prefixed(
  weights: &mut HashMap<String, Array>,
  dest: &mut HashMap<String, Array>,
  prefix: &str,
) -> Result<()> {
  let matching = weights.keys().filter(|k| k.starts_with(prefix)).count();
  let mut keys: Vec<String> = Vec::new();
  reserve_or_error(&mut keys, "clap move_prefixed: matched key list", matching)?;
  for k in weights.keys().filter(|k| k.starts_with(prefix)) {
    keys.push(fallible_clone_str("clap move_prefixed: matched key", k)?);
  }
  reserve_or_error(dest, "clap move_prefixed: dest growth", keys.len())?;
  for k in keys {
    if let Some(v) = weights.remove(&k) {
      insert_unique(dest, k, v, "clap move_prefixed")?;
    }
  }
  Ok(())
}

/// The number of label rows in a `(num_labels, seq_len)` token-id batch, erroring
/// if it is not rank-2.
#[cfg(feature = "clap")]
fn classify_label_count(label_ids: &Array) -> Result<usize> {
  let shape = label_ids.shape();
  if shape.len() != 2 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap classify: label_ids",
      "must be rank-2 (num_labels, seq_len)",
      format_smolstr!("rank {}", shape.len()),
    )));
  }
  Ok(shape[0])
}

/// Flatten a single-row `(1, dim)` embedding to the rank-1 `(dim,)` vector
/// [`cosine_similarity`] takes, erroring if the leading axis is not exactly `1`.
#[cfg(feature = "clap")]
fn flatten_single_row(embedding: &Array) -> Result<Array> {
  let shape = embedding.shape();
  if shape.len() != 2 || shape[0] != 1 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap classify: audio embedding",
      "must be (1, dim) — a single-clip audio embedding",
      format_smolstr!("{shape:?}"),
    )));
  }
  let dim = crate::embeddings::clap::shared::dim_i32(&shape, 1, "clap classify: embedding dim")?;
  ops::shape::reshape(embedding, &[dim])
}

/// Take row `i` of a `(rows, dim)` embedding batch as the rank-1 `(dim,)` vector
/// [`cosine_similarity`] takes.
#[cfg(feature = "clap")]
fn take_row(embeddings: &Array, i: i32) -> Result<Array> {
  let idx = Array::from_slice::<i32>(&[i], &(1usize,))?;
  let row = ops::indexing::take_axis(embeddings, &idx, 0)?; // (1, dim)
  let dim = crate::embeddings::clap::shared::dim_i32(&row.shape(), 1, "clap classify: row dim")?;
  ops::shape::reshape(&row, &[dim])
}

#[cfg(all(test, feature = "clap"))]
mod tests;
