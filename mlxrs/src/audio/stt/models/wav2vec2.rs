//! Wav2Vec2 CTC speech recognizer (the `wav2vec2` / `hubert` CTC family).
//!
//! Port of mlx-audio's `Wav2Vec2ForCTC` — the backbone in
//! [`stt/models/wav2vec/wav2vec.py`][wav2vec] (feature encoder + feature
//! projection + transformer encoder) composed with the CTC head + greedy
//! decode + waveform normalization in [`stt/models/mms/mms.py`][mms]. MMS
//! *is* Wav2Vec2ForCTC plus a per-attention-block language adapter; the MMS
//! per-language adapter overlay (`Model.post_load_hook`) is ported here too —
//! see [`Model::load_with_target_lang`].
//!
//! The port is generic over the family the way the reference's `ModelConfig`
//! is: the builders read every width / count / conv-stack field from the
//! config, so the `base`, `large`, and `large-960h-lv60-self`-style variants
//! (and the `hubert` checkpoints, which share the plain self-attention
//! transformer) all load. Both encoder arms the reference ships are mirrored:
//!
//! - **post-norm** ([`Wav2Vec2Encoder`][enc] / [`Wav2Vec2EncoderLayer`][el],
//!   `do_stable_layer_norm = false`) — the encoder `LayerNorm` is applied
//!   *before* the layer stack, and each layer is
//!   `h = layer_norm(h + attn(h)); h = final_layer_norm(h + ff(h))`;
//! - **stable-layer-norm** (pre-norm) ([`Wav2Vec2EncoderStableLayerNorm`][senc]
//!   / [`Wav2Vec2EncoderLayerStableLayerNorm`][sel],
//!   `do_stable_layer_norm = true`) — the encoder `LayerNorm` is applied
//!   *after* the layer stack, and each layer is
//!   `h = h_in + attn(layer_norm(h_in)); h = h + ff(final_layer_norm(h))`.
//!
//! The model is **not** autoregressive, so it does not implement the
//! [`crate::audio::stt::model::AutoregressiveStt`] trait (encoder + per-token
//! cross-attention `decode_step` + KV cache). Inference is a single forward
//! over the raw 16 kHz mono waveform producing per-frame logits `(B, T', V)`,
//! followed by a greedy CTC collapse over a character vocabulary. The public
//! surface is therefore inherent:
//!
//! - [`Model::forward`] — waveform `(B, T)` → logits `(B, T', V)`.
//! - [`Model::transcribe`] — waveform → decoded `String`
//!   (normalize → forward → greedy CTC collapse → vocabulary map).
//!
//! The config and model types are named [`Config`] and [`Model`] (not
//! `Wav2Vec2Config` / `Wav2Vec2Ctc`): inside the `wav2vec2` module the
//! `Wav2Vec2`-prefixed forms stutter (the no-stutter convention), so the
//! un-prefixed names are intentional and there are deliberately no prefixed
//! compatibility aliases.
//!
//! ## Activation
//!
//! mlx-audio hardcodes `nn.GELU()` (the exact, `approx="none"` GELU) at every
//! block and ignores the config activation fields. This port instead dispatches
//! on `hidden_act` (the transformer feed-forward) and `feat_extract_activation`
//! (the feature-encoder convs), mapping the HF activation names — `gelu` to the
//! exact GELU (so a `gelu` checkpoint is identical to the reference),
//! `gelu_new` / `gelu_pytorch_tanh` to the tanh-approx GELU, and `silu` /
//! `swish` to SiLU. See [`Activation`].
//!
//! ## Variant coverage
//!
//! Every wav2vec2 CTC variant mlx-audio serves is wired:
//!
//! - **both feature-encoder norm arms** — `feat_extract_norm = "group"` (the
//!   `base` / `large` default) and `"layer"` (the all-LayerNorm extractor used
//!   by `large-960h-lv60-self`), branched in the feature-encoder builder;
//! - **the MMS per-attention-block language adapter** (`adapter_attn_dim`) —
//!   the attention adapter on every stable-LN encoder layer, plus the
//!   per-language adapter overlay + per-language vocabulary
//!   ([`Model::load_with_target_lang`]), unlocking `facebook/mms-1b-all` /
//!   `mms-1b-fl102`;
//! - **the HuBERT no-LayerNorm feature projection** (`feat_proj_layer_norm =
//!   false`) — the conditional projection LayerNorm in the feature projection.
//!   This is **HuBERT-only**: HF's `Wav2Vec2FeatureProjection` always applies the
//!   projection LayerNorm, so the no-LayerNorm arm is honored only for
//!   `model_type == "hubert"` (a wav2vec2 config with `feat_proj_layer_norm =
//!   false` is rejected by [`Config::validate`]).
//!
//! ## Out of scope
//!
//! The post-encoder conv adapter (`add_adapter = true`, a different output
//! dimension), the HuBERT-only batch-norm positional conv
//! (`conv_pos_batch_norm = true`) arm, Conformer relative-position attention,
//! sharded-checkpoint loading, and a configurable CTC blank are **not** wired;
//! a config that needs one of them is rejected by [`Config::validate`] with a
//! typed error. (The HuBERT `conv_pos_batch_norm = false` default matches the
//! wired graph, so a default HuBERT checkpoint is faithfully supported; only
//! the non-default arm is rejected.)
//!
//! **WavLM** is likewise deferred: its defining feature is gated
//! relative-position-bias attention, which this phase does not implement, and
//! it has no plain-attention variant. So a `wavlm` checkpoint cannot be run
//! faithfully through the plain self-attention path here (its relative-position
//! tensors would go unconsumed = silent corruption); `validate` therefore
//! **rejects** `model_type == "wavlm"` rather than silently accepting a model
//! this phase cannot serve. WavLM is unlocked once gated relative-position-bias
//! attention is wired (a later phase).
//!
//! [wav2vec]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py
//! [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py
//! [enc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L511-L574
//! [el]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L436-L465
//! [senc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L577-L644
//! [sel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L468-L508

use std::collections::HashMap;

use derive_more::{Display, IsVariant};
use serde::de::DeserializeOwned;
use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::stt::{
    generate::greedy_ctc_transcribe,
    model::{CtcModel, Transcribe, TranscribeOptions, Transcription},
  },
  error::{
    Error, FileIoPayload, FileOp, InvariantViolationPayload, KeyCollisionPayload,
    LayerKeyedPayload, LengthMismatchPayload, MalformedDataPayload, MissingKeyPayload,
    NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload, UnknownEnumValuePayload,
  },
  lm::{
    nn::{
      attention::{Mask, scaled_dot_product_attention},
      norm::{GroupNorm, LayerNorm},
    },
    quant::{PerLayerQuantization, QuantizationOption},
  },
  model_validation::{
    checked_mul, insert_unique, pin_bool, pin_str, require_cardinality, require_divisible,
    require_positive, reserve_or_error,
  },
  nn::{MaybeQuantizedLinear, QuantizedLinear},
  ops,
};

// ───────────────────────────── family traits ─────────────────────────────

/// The variant's transformer block stack — the locus of family variation
/// across the wav2vec2 speech-encoder family (wav2vec2 / HuBERT share one
/// [`StandardEncoder`]; WavLM / Conformer diverge here in a later phase).
///
/// The shared [`Model`] is generic over the [`Family`] and stores the family's
/// [`Family::Encoder`], dispatching the per-utterance forward through this
/// trait. `attention_mask` is the optional additive mask over the time axis;
/// the CTC path runs a single un-padded utterance, so it passes `None` (the
/// `Mask::None` self-attention the concrete arms apply).
#[cfg(feature = "wav2vec2")]
pub trait Encoder {
  /// Run the encoder over the projected hidden states `(B, T', hidden)`,
  /// returning the encoded states of the same shape. `attention_mask`, when
  /// present, is the additive mask the self-attention adds before the softmax.
  fn forward(&self, hidden: &Array, attention_mask: Option<&Array>) -> Result<Array>;
}

/// The shared config surface every family config exposes — the base view the
/// shared scaffolding reads, plus the variant's structural validation.
///
/// [`Self::validate`] returns the crate's typed [`Error`] (never a stringly
/// error). [`Self::base`] projects the shared base config; in this single-
/// dialect (Standard) phase the dialect config [`Config`] *is* the shared base,
/// so it returns `&self`. A later phase that adds a second dialect carrying its
/// own extra fields extracts a distinct shared base struct this projects to.
#[cfg(feature = "wav2vec2")]
pub trait FamilyConfig: DeserializeOwned {
  /// The shared base config the generic scaffolding reads.
  fn base(&self) -> &Config;

  /// Reject any config the variant cannot run, with a typed [`Error`], before
  /// any tensor is allocated.
  fn validate(&self) -> Result<()>;
}

/// One member of the wav2vec2 speech-encoder family ("dialect"). Captures
/// everything that varies across the family; the shared [`Model`] is generic
/// over it. Mirrors HF transformers' separate encoder classes
/// (`Wav2Vec2Encoder`, `WavLMEncoder`, `Wav2Vec2ConformerEncoder`) rather than
/// one config-branched class.
///
/// This phase ships a single dialect, [`Standard`] (wav2vec2 + HuBERT, the
/// plain self-attention transformer). WavLM (gated relative-position-bias
/// attention) and Conformer (the Conformer block + relative position) each add
/// one `impl Family` in a later phase.
#[cfg(feature = "wav2vec2")]
pub trait Family {
  /// The dialect's config — its [`FamilyConfig`] surface.
  type Config: FamilyConfig;
  /// The dialect's encoder — its [`Encoder`] block stack.
  type Encoder: Encoder;

  /// The HF `model_type` strings this dialect claims (e.g. the Standard
  /// dialect's `["wav2vec2", "hubert"]`).
  const MODEL_TYPES: &'static [&'static str];

  /// Build the dialect's [`Self::Encoder`] from the validated config and the
  /// sanitized weight map, resolving each Linear's dense-vs-quantized scheme
  /// from `quant` (the per-layer quantization config), exactly as the dense
  /// builders do. `weights` keys are consumed by exact (sanitized) key.
  fn build_encoder(
    config: &Self::Config,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self::Encoder>;
}

/// Feature-extractor normalization scheme: group-norm (the wav2vec2 / HuBERT
/// `base` / `large` default) vs the layer-norm arm (used by
/// `large-960h-lv60-self`).
///
/// Both arms are wired, mirroring `Wav2Vec2FeatureEncoder`'s
/// `feat_extract_norm` branch ([wav2vec.py:254-267][fe]): [`Self::Group`] is an
/// affine L0 `Wav2Vec2GroupNormConvLayer` then plain `Wav2Vec2NoLayerNormConvLayer`s;
/// [`Self::Layer`] is a `Wav2Vec2LayerNormConvLayer` (conv → LayerNorm →
/// activation) at every layer. Resolve via [`Config::feat_extract_norm_scheme`].
/// Unit-only enum → mandatory `as_str` projection; `Display` derives through it.
///
/// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L254-L267
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, IsVariant)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum FeatExtractNorm {
  /// Group-norm feature extractor: an affine L0 GroupNorm then plain conv
  /// layers. HF `feat_extract_norm == "group"`.
  Group,
  /// Layer-norm feature extractor: an affine LayerNorm at every conv layer
  /// (conv → LayerNorm → activation). HF `feat_extract_norm == "layer"`.
  Layer,
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl FeatExtractNorm {
  /// The scheme's stable HF string name (`config.json` `feat_extract_norm`).
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Group => "group",
      Self::Layer => "layer",
    }
  }
}

// ───────────────────────────── config ─────────────────────────────

/// Wav2Vec2 model configuration — the typed subset of HF `config.json`
/// mlx-audio's `ModelConfig` ([wav2vec.py][cfg]) reads, restricted to the
/// fields the inference forward pass actually consumes.
///
/// Defaults match `facebook/wav2vec2-base-960h`. Like mlx-audio's
/// `BaseModelArgs.from_dict` (and the rest of mlxrs's `#[serde(default)]`
/// configs), unmodeled keys parse cleanly and absent keys fall back to the
/// default — a forward-compatible read, not `deny_unknown_fields`.
///
/// [cfg]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L27-L74
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
  /// Architecture id (`config.json` `model_type`). Accepted: `"wav2vec2"` and
  /// `"hubert"` — the family that shares the plain self-attention transformer
  /// this port wires (HuBERT reuses the wav2vec2 encoder architecture). WavLM
  /// is **rejected** by [`Config::validate`]: it needs gated
  /// relative-position-bias attention, deferred to a later phase.
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Vocabulary size — the CTC head's output width / logits last axis.
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// Transformer hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of transformer encoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Number of attention heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Feed-forward intermediate dimension.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// `eps` shared by every `LayerNorm` and the L0 `GroupNorm`.
  #[serde(default = "default_layer_norm_eps")]
  pub layer_norm_eps: f32,
  /// Feature-encoder normalization scheme. Both arms are wired (see
  /// [`Config::feat_extract_norm_scheme`]): `"group"` (the `base`/`large`
  /// default — an affine L0 GroupNorm then plain conv layers) and `"layer"`
  /// (used by `large-960h-lv60-self` — an affine LayerNorm at every conv
  /// layer). Any other value is rejected by [`Config::validate`].
  #[serde(default = "default_feat_extract_norm")]
  pub feat_extract_norm: String,
  /// Hidden (transformer feed-forward) activation. Dispatched on by
  /// [`Activation::resolve`]: `"gelu"` (the exact GELU, matching the
  /// reference), `"gelu_new"` / `"gelu_pytorch_tanh"` (tanh-approx GELU), or
  /// `"silu"` / `"swish"`. An unsupported name is rejected by
  /// [`Config::validate`].
  #[serde(default = "default_hidden_act")]
  pub hidden_act: String,
  /// Feature-encoder conv activation. Dispatched on by [`Activation::resolve`]
  /// exactly as [`Config::hidden_act`]; an unsupported name is rejected
  /// by [`Config::validate`].
  #[serde(default = "default_feat_extract_activation")]
  pub feat_extract_activation: String,
  /// Per-conv-layer output channel widths (length `num_feat_extract_layers`).
  #[serde(default = "default_conv_dim")]
  pub conv_dim: Vec<i32>,
  /// Per-conv-layer stride.
  #[serde(default = "default_conv_stride")]
  pub conv_stride: Vec<i32>,
  /// Per-conv-layer kernel size.
  #[serde(default = "default_conv_kernel")]
  pub conv_kernel: Vec<i32>,
  /// Whether the feature-encoder convolutions carry a bias (`base`/`large`:
  /// `false`). When `true`, every feature-encoder conv layer loads and adds its
  /// `conv.bias`.
  #[serde(default)]
  pub conv_bias: bool,
  /// Positional-conv-embedding kernel size (even → one trailing frame is
  /// cropped by the SamePad step).
  #[serde(default = "default_num_conv_pos_embeddings")]
  pub num_conv_pos_embeddings: i32,
  /// Positional-conv-embedding group count (grouped depthwise-ish conv).
  #[serde(default = "default_num_conv_pos_embedding_groups")]
  pub num_conv_pos_embedding_groups: i32,
  /// Number of feature-extractor conv layers.
  #[serde(default = "default_num_feat_extract_layers")]
  pub num_feat_extract_layers: i32,
  /// Whether the encoder uses the stable-layer-norm (pre-norm) arm.
  /// `base`/`large`: `false` (post-norm); `large-960h-lv60-self`-style
  /// checkpoints: `true`. Both arms are wired (see the module docs).
  #[serde(default)]
  pub do_stable_layer_norm: bool,
  /// Whether a convolutional adapter network is stacked on top of the encoder.
  /// `base-960h`: `false`. When `true`, HF inserts a `Wav2Vec2Adapter` stack
  /// that re-shapes the encoder output to `output_hidden_size` and the CTC head
  /// reads from *that* dimension — a graph this port does not wire — so a
  /// `true` checkpoint would load and run silently wrong. Rejected by
  /// [`Config::validate`].
  #[serde(default)]
  pub add_adapter: bool,
  /// Per-attention-block adapter bottleneck dimension — the **MMS** language
  /// adapter (`facebook/mms-1b-all` / `mms-1b-fl102`). `base-960h`: absent /
  /// `null`. When set, the stable-layer-norm encoder layer adds a
  /// `Wav2Vec2AttnAdapterLayer` (`LayerNorm → Linear(hidden → adapter_attn_dim)
  /// → ReLU → Linear(adapter_attn_dim → hidden)`) whose output is **added** to
  /// the hidden states ([wav2vec.py:484-487,503-504][ad]), specializing the
  /// language-agnostic backbone to one language. Modeled as `Option<i32>`
  /// (absent ⇒ `None`); a non-positive value is rejected by
  /// [`Config::validate`]. The reference attaches the adapter only to the
  /// stable-LN layer, so a post-norm checkpoint carrying it runs no adapter.
  ///
  /// [ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L484-L504
  #[serde(default)]
  pub adapter_attn_dim: Option<i32>,
  /// CTC blank-token id. `base-960h`: `0`. Greedy CTC decoding drops exactly
  /// this id (the port hardcodes `CTC_BLANK = 0`), so a checkpoint declaring a
  /// different blank would collapse the per-frame argmax against the wrong
  /// token and decode silently wrong. Pinned to `0` by
  /// [`Config::validate`].
  #[serde(default = "default_pad_token_id")]
  pub pad_token_id: i32,
  /// Whether the feature projection applies a `LayerNorm` to the feature
  /// encoder's output before the linear projection. A **HuBERT** config field
  /// (`HubertConfig.feat_proj_layer_norm`, default `true`); the wav2vec2 config
  /// has no such field and always applies the LayerNorm. Both arms are wired:
  /// the feature projection applies the LayerNorm iff this flag is `true`
  /// (always so for a wav2vec2 checkpoint) and skips it when `false` (HuBERT's
  /// no-LayerNorm projection arm, feeding the linear the un-normalized
  /// feature-encoder output directly).
  #[serde(default = "default_feat_proj_layer_norm")]
  pub feat_proj_layer_norm: bool,
  /// Whether the positional conv embedding uses batch-norm instead of the
  /// weight-norm reparametrization. A **HuBERT** config field
  /// (`HubertConfig.conv_pos_batch_norm`, default `false`); the wav2vec2 config
  /// has no such field and always uses weight-norm. This port wires the `false`
  /// (weight-norm) arm only — the positional conv embedding reconstructs the
  /// fused kernel from the `weight_g` / `weight_v` pair — so a `true` value
  /// selects a different module (a `BatchNorm1d` over a plain conv, whose
  /// checkpoint carries no `weight_g` / `weight_v`): the builder would fail to
  /// find those keys, or, worse, run the wrong graph. Pinned to `false` by
  /// [`Config::validate`]; the batch-norm arm is out of scope.
  #[serde(default)]
  pub conv_pos_batch_norm: bool,
}

#[cfg(feature = "wav2vec2")]
fn default_model_type() -> String {
  "wav2vec2".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_vocab_size() -> i32 {
  32
}
#[cfg(feature = "wav2vec2")]
fn default_hidden_size() -> i32 {
  768
}
#[cfg(feature = "wav2vec2")]
fn default_num_hidden_layers() -> i32 {
  12
}
#[cfg(feature = "wav2vec2")]
fn default_num_attention_heads() -> i32 {
  12
}
#[cfg(feature = "wav2vec2")]
fn default_intermediate_size() -> i32 {
  3072
}
#[cfg(feature = "wav2vec2")]
fn default_layer_norm_eps() -> f32 {
  1e-5
}
#[cfg(feature = "wav2vec2")]
fn default_feat_extract_norm() -> String {
  "group".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_hidden_act() -> String {
  "gelu".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_feat_extract_activation() -> String {
  "gelu".to_string()
}
#[cfg(feature = "wav2vec2")]
fn default_conv_dim() -> Vec<i32> {
  vec![512, 512, 512, 512, 512, 512, 512]
}
#[cfg(feature = "wav2vec2")]
fn default_conv_stride() -> Vec<i32> {
  vec![5, 2, 2, 2, 2, 2, 2]
}
#[cfg(feature = "wav2vec2")]
fn default_conv_kernel() -> Vec<i32> {
  vec![10, 3, 3, 3, 3, 2, 2]
}
#[cfg(feature = "wav2vec2")]
fn default_num_conv_pos_embeddings() -> i32 {
  128
}
#[cfg(feature = "wav2vec2")]
fn default_num_conv_pos_embedding_groups() -> i32 {
  16
}
#[cfg(feature = "wav2vec2")]
fn default_num_feat_extract_layers() -> i32 {
  7
}
#[cfg(feature = "wav2vec2")]
fn default_pad_token_id() -> i32 {
  0
}
#[cfg(feature = "wav2vec2")]
fn default_feat_proj_layer_norm() -> bool {
  // HuBERT default; the wav2vec2 graph always applies the projection LayerNorm.
  true
}

// ── architecture invariants ──
//
// The port is generic over the family: the builders read every width / count /
// conv-stack field from the (validated) config and each layer's shape from the
// checkpoint tensor. `validate` therefore no longer pins the *dimensions* to a
// single variant — it enforces only the structural invariants the wired graph
// genuinely requires (positivity, divisibility, conv-array length consistency,
// a supported activation / feature-norm scheme, the absent adapters, and the
// CTC blank id) — so the `base` / `large` / stable-LN variants, and the
// `hubert` checkpoints sharing the plain self-attention transformer, all load.

/// `model_type` values this port accepts — the family that shares the plain
/// self-attention transformer graph wired here: `wav2vec2` and `hubert`
/// (HuBERT reuses the wav2vec2 encoder architecture). WavLM is **not** accepted:
/// its defining feature is gated relative-position-bias attention, which this
/// phase does not implement, and it has no plain-attention variant — so a wavlm
/// checkpoint would run its relative-position tensors through the plain
/// self-attention path unconsumed (silent transcription corruption). WavLM is
/// deferred until gated relative-position-bias attention is wired.
#[cfg(feature = "wav2vec2")]
const SUPPORTED_MODEL_TYPES: &[&str] = &["wav2vec2", "hubert"];

/// `feat_extract_norm` values this port wires — the two feature-encoder
/// normalization schemes the reference's `Wav2Vec2FeatureEncoder` builds
/// ([wav2vec.py:254-267][fe]): `"group"` (an affine L0 GroupNorm then plain
/// conv layers, the `base` / `large` default) and `"layer"` (an all-LayerNorm
/// extractor, used by `large-960h-lv60-self`). Any other value is rejected.
///
/// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L254-L267
#[cfg(feature = "wav2vec2")]
const SUPPORTED_FEAT_EXTRACT_NORMS: &[&str] = &["group", "layer"];

/// Backbone key prefixes [`sanitize`] strips — one per supported `*ForCTC`
/// checkpoint family. `Wav2Vec2ForCTC` nests the backbone under `wav2vec2.`,
/// `HubertForCTC` under `hubert.` (HuBERT reuses the wav2vec2 encoder, so the
/// post-strip keys are identical); `lm_head` stays top-level in both. Kept in
/// lock-step with [`SUPPORTED_MODEL_TYPES`] (no `wavlm.`, since WavLM is not
/// served this phase).
#[cfg(feature = "wav2vec2")]
const SUPPORTED_BACKBONE_PREFIXES: &[&str] = &["wav2vec2.", "hubert."];

/// CTC blank-token id this port hardcodes (`pad_token_id`). Greedy decoding
/// drops exactly this id, so `validate` pins the config's `pad_token_id` to it.
#[cfg(feature = "wav2vec2")]
const PAD_TOKEN_ID: i32 = 0;

/// Cap on the cardinality fields that size an **eager per-layer allocation** —
/// `num_hidden_layers` (the encoder-layer `Vec`, and the adapter-key `Vec` /
/// allowed-key set the MMS overlay builds, all sized at `O(num_hidden_layers)`)
/// and `num_feat_extract_layers` (the feature-encoder conv `Vec`). Unlike the
/// width fields these size *up-front allocations*, so the overflow-safe width
/// cap is far too loose: a `2^24`-layer config would request a multi-gigabyte
/// `Vec` (and, on the MMS path, an `O(layers)` key `Vec` + set) before the first
/// missing-key error. The largest real wav2vec2 / HuBERT / MMS checkpoint has
/// tens of layers; `4096` is generous headroom yet keeps a malformed cardinality
/// a recoverable [`Error::CapExceeded`] (or, if a within-cap count still exceeds
/// memory, a recoverable [`Error::AllocFailure`] from the `try_reserve`d `Vec` /
/// set) rather than an allocator abort. This is the project's config-cardinality
/// bound (matching `qwen3` / `lfm2` / whisper's `MAX_LAYERS`), **not** a DoS cap
/// on otherwise-valid input — a transformer with billions of layers is not a
/// valid checkpoint.
#[cfg(feature = "wav2vec2")]
const MAX_CONFIG_CARDINALITY: i64 = 4096;

/// The default MMS target language — `"eng"` (English), the language
/// mlx-audio's `Model.post_load_hook` reaches for first
/// (`adapter.eng.safetensors`, `vocab.get("eng", …)`,
/// [mms.py:134,151][mms]). [`Model::load`] uses it when no explicit
/// `target_lang` is requested.
///
/// [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L134-L151
#[cfg(feature = "wav2vec2")]
const DEFAULT_TARGET_LANG: &str = "eng";

/// The MMS per-language adapter filename prefix / suffix
/// (`adapter.{lang}.safetensors`, [mms.py:134-138][mms]). [`adapter_file_for`]
/// discovers `adapter.{target_lang}.safetensors` or falls back to the first
/// `adapter.*.safetensors`.
///
/// [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L134-L138
#[cfg(feature = "wav2vec2")]
const ADAPTER_FILE_PREFIX: &str = "adapter.";
#[cfg(feature = "wav2vec2")]
const ADAPTER_FILE_SUFFIX: &str = ".safetensors";

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Config {
  /// Parse a [`Config`] from an in-memory `config.json` string.
  ///
  /// Mirrors mlx-audio's `ModelConfig.from_dict` (a `json.load` restricted to
  /// the known keys). A malformed-JSON failure maps to [`Error::Parse`]; an
  /// unmodeled key is ignored (forward-compatible) and an absent key takes
  /// the `base-960h` default.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json)
      .map_err(|e| Error::Parse(ParsePayload::new("Config::from_json", "config JSON", e)))
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// `true` when `model_type == "hubert"` — the HuBERT dialect (which shares
  /// the plain self-attention transformer but adds the `feat_proj_layer_norm`
  /// flag gating its no-LayerNorm feature projection). HF's
  /// `Wav2Vec2FeatureProjection` has **no** such flag (it always applies the
  /// projection LayerNorm); only `HubertFeatureProjection` honors it. So the
  /// no-LayerNorm projection arm (`feat_proj_layer_norm = false`) is HuBERT-only
  /// — see [`Config::validate`] and the feature-projection builder.
  #[inline(always)]
  pub fn is_hubert(&self) -> bool {
    self.model_type == "hubert"
  }

  /// `true` when `feat_extract_norm == "group"` — the group-norm feature
  /// encoder (the `base` / `large` default: an affine L0 GroupNorm then plain
  /// conv layers). The `"layer"` variant used by `large-960h-lv60-self` is the
  /// other wired arm (see [`Config::feat_extract_norm_scheme`]).
  #[inline(always)]
  pub fn is_group_norm(&self) -> bool {
    self.feat_extract_norm == "group"
  }

  /// Resolve `feat_extract_norm` to the typed [`FeatExtractNorm`] scheme, or
  /// reject an unsupported name with [`Error::UnknownEnumValue`].
  ///
  /// `"group"` → [`FeatExtractNorm::Group`] (the `base` / `large` default: an
  /// affine L0 GroupNorm then plain conv layers); `"layer"` →
  /// [`FeatExtractNorm::Layer`] (the all-LayerNorm extractor used by
  /// `large-960h-lv60-self`). Any other value is rejected (matching the
  /// reference's `else: raise ValueError(...)`, [wav2vec.py:264-266][fe]).
  ///
  /// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L254-L267
  pub fn feat_extract_norm_scheme(&self) -> Result<FeatExtractNorm> {
    match self.feat_extract_norm.as_str() {
      "group" => Ok(FeatExtractNorm::Group),
      "layer" => Ok(FeatExtractNorm::Layer),
      other => Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        "Config: feat_extract_norm",
        other,
        SUPPORTED_FEAT_EXTRACT_NORMS,
      ))),
    }
  }

  /// Reject any config this port cannot run, with a typed error, **before**
  /// any tensor is allocated or any weight is read.
  ///
  /// The port is generic over the family: the builders read every width /
  /// count / conv-stack field from this config and each layer's shape from the
  /// checkpoint tensor. `validate` therefore enforces only the **structural
  /// invariants** the wired graph genuinely requires — not a single variant's
  /// magnitudes — so the `base` / `large` / stable-LN variants and the
  /// `hubert` checkpoints that share the plain self-attention transformer all
  /// pass. It is the single gate (called at the top of
  /// [`Model::from_weights`], and eagerly in [`Model::load`] so a
  /// mismatch fails before the weights file is even read).
  ///
  /// Rejected (each a typed error, never a panic or a silent mis-load):
  /// - `model_type` not one of the supported family ids (`wav2vec2`,
  ///   `hubert`) ([`Error::UnknownEnumValue`]) — `wavlm` is **rejected** here
  ///   (its gated relative-position-bias attention is not wired this phase);
  /// - `feat_extract_norm` not one of `"group"` / `"layer"`
  ///   ([`Error::UnknownEnumValue`]) — both feature-encoder arms are wired,
  ///   any other value is rejected;
  /// - `hidden_act` / `feat_extract_activation` not a supported activation
  ///   ([`Error::UnknownEnumValue`], via [`Activation::resolve`]);
  /// - `add_adapter == true` ([`Error::InvariantViolation`]) — the
  ///   post-encoder conv adapter stack (a different output dimension) is out of
  ///   scope; (`adapter_attn_dim`, the MMS per-attention-block adapter, is now
  ///   wired — only a non-positive bottleneck width is rejected,
  ///   [`Error::OutOfRange`]);
  /// - `conv_pos_batch_norm == true` ([`Error::InvariantViolation`]) — the
  ///   HuBERT-only batch-norm positional-conv arm is out of scope (the wired
  ///   graph reconstructs a weight-normalized positional conv); the HF default
  ///   `false` matches the wired graph, so default HuBERT and every wav2vec2
  ///   checkpoint pass;
  /// - `feat_proj_layer_norm == false` for a non-`hubert` `model_type`
  ///   ([`Error::InvariantViolation`]) — the no-LayerNorm projection is a
  ///   HuBERT-only arm (HF's `Wav2Vec2FeatureProjection` always applies the
  ///   projection LayerNorm; only `HubertFeatureProjection` gates it), so honoring
  ///   `false` on a wav2vec2 `model_type` would build a silently-wrong graph;
  ///   `true` / absent is accepted for every `model_type`, and `false` only for
  ///   `model_type == "hubert"`;
  /// - `pad_token_id != 0` ([`Error::OutOfRange`]) — greedy CTC decoding drops
  ///   exactly id `0` (the hardcoded CTC blank), so a different declared blank
  ///   would collapse the argmax against the wrong token;
  /// - a non-positive `hidden_size` / `num_attention_heads` /
  ///   `intermediate_size` / `vocab_size` / `num_conv_pos_embeddings`
  ///   ([`Error::OutOfRange`]); `hidden_size` not divisible by
  ///   `num_attention_heads` or by `num_conv_pos_embedding_groups`
  ///   ([`Error::DivisibilityConstraint`]);
  /// - a non-positive or over-cap (`MAX_CONFIG_CARDINALITY`) `num_hidden_layers`
  ///   / `num_feat_extract_layers` ([`Error::OutOfRange`] /
  ///   [`Error::CapExceeded`]) — each sizes an eager per-layer `Vec` (and, on the
  ///   MMS path, an `O(num_hidden_layers)` adapter-key `Vec` + set), reserved
  ///   fallibly by the builder / overlay;
  /// - a non-finite or non-positive `layer_norm_eps`
  ///   ([`Error::NonFiniteScalar`] / [`Error::OutOfRange`]);
  /// - a `conv_dim` / `conv_stride` / `conv_kernel` array whose length is not
  ///   **exactly** `num_feat_extract_layers` ([`Error::LengthMismatch`]) — a
  ///   too-long array would desync the feature encoder (`n` layers, output
  ///   width `conv_dim[n-1]`) from the projection (which reads `conv_dim.last()`)
  ///   — or carrying a non-positive element ([`Error::OutOfRange`]).
  pub fn validate(&self) -> Result<()> {
    // model_type: the family sharing the plain self-attention transformer
    // (wav2vec2 + hubert). `wavlm` is rejected — its gated
    // relative-position-bias attention is not wired this phase.
    pin_str(
      "Config: model_type",
      self.model_type.as_str(),
      SUPPORTED_MODEL_TYPES,
    )?;
    // Feature-encoder normalization scheme: both the group-norm and the
    // layer-norm arms are wired (the reference's `Wav2Vec2FeatureEncoder`
    // branches on `"group"` / `"layer"`). Resolving rejects any other value
    // with the same typed error the reference's `else: raise` would.
    self.feat_extract_norm_scheme()?;
    // Activations: resolve both names against the supported set. The resolved
    // values are recomputed at build time; resolving here makes an unsupported
    // activation fail fast (before any tensor) with the same typed error.
    Activation::resolve(&self.hidden_act, "Config: hidden_act")?;
    Activation::resolve(
      &self.feat_extract_activation,
      "Config: feat_extract_activation",
    )?;
    // The post-encoder convolutional adapter stack is out of scope: when
    // `true`, HF re-shapes the encoder output (to `output_hidden_size`) before
    // the CTC head, a graph this port does not build.
    pin_bool("Config: add_adapter", self.add_adapter, false)?;
    // The per-attention-block adapter (MMS) is now wired: when
    // `adapter_attn_dim` is set, the stable-LN encoder layer adds a
    // `Wav2Vec2AttnAdapterLayer` whose output is added to the hidden states
    // ([wav2vec.py:484-487,503-504]). It must be a positive bottleneck width
    // (it sizes the adapter Linears); a non-positive value is malformed. The
    // reference attaches the adapter only to the stable-LN layer, so a post-norm
    // checkpoint carrying `adapter_attn_dim` simply runs no adapter (faithful) —
    // the value is still validated here.
    if let Some(d) = self.adapter_attn_dim {
      require_positive("Config: adapter_attn_dim", d)?;
    }
    // CTC blank id: greedy decode hardcodes `CTC_BLANK = 0`, so a checkpoint
    // declaring a different blank would collapse the per-frame argmax against
    // the wrong token. Pin it to the wired value (`0`); a configurable blank is
    // out of scope.
    if self.pad_token_id != PAD_TOKEN_ID {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Config: pad_token_id",
        "must equal the hardcoded CTC blank id (0)",
        format_smolstr!("{} (expected {})", self.pad_token_id, PAD_TOKEN_ID),
      )));
    }
    // `feat_proj_layer_norm` is a HuBERT-ONLY flag: HF's
    // `Wav2Vec2FeatureProjection` has no such field and ALWAYS applies the
    // projection LayerNorm, while `HubertFeatureProjection` gates it on this flag
    // (HuBERT default `true`). So the no-LayerNorm projection arm
    // (`feat_proj_layer_norm = false`) is only valid for `model_type == "hubert"`:
    // honoring `false` for a wav2vec2 `model_type` would build a silently-wrong
    // graph (a wav2vec2 model with its projection LayerNorm dropped). A wav2vec2
    // config that nonetheless declares `feat_proj_layer_norm = false` is therefore
    // rejected here (fail-closed, before any tensor is read); `true` / absent is
    // accepted for every `model_type` (the LayerNorm arm), and `false` is honored
    // only for HuBERT (its no-LayerNorm arm — see `build_feature_projection`).
    if !self.feat_proj_layer_norm && !self.is_hubert() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Config: feat_proj_layer_norm",
        "feat_proj_layer_norm = false is a HuBERT-only arm (HF wav2vec2 always applies the \
         projection LayerNorm); it must not be set for a non-hubert model_type",
      )));
    }
    //
    // `conv_pos_batch_norm` (HuBERT default `false`): the wired
    // `PositionalConvEmbedding` reconstructs the fused kernel from the
    // weight-norm `weight_g` / `weight_v` pair, so the batch-norm (`true`) arm
    // selects a different module (a `BatchNorm1d` over a plain conv whose
    // checkpoint carries no `weight_g` / `weight_v`).
    pin_bool(
      "Config: conv_pos_batch_norm",
      self.conv_pos_batch_norm,
      false,
    )?;
    // Width dimensions: positivity + the divisibility the wired graph needs
    // (per-head split; grouped positional conv). These bound every width a
    // later step divides by or uses to size work.
    require_positive("Config: hidden_size", self.hidden_size)?;
    require_positive("Config: num_attention_heads", self.num_attention_heads)?;
    require_positive("Config: intermediate_size", self.intermediate_size)?;
    require_positive("Config: vocab_size", self.vocab_size)?;
    require_positive(
      "Config: num_conv_pos_embeddings",
      self.num_conv_pos_embeddings,
    )?;
    require_positive(
      "Config: num_conv_pos_embedding_groups",
      self.num_conv_pos_embedding_groups,
    )?;
    require_divisible(
      "Config: hidden_size",
      self.hidden_size,
      "Config: num_attention_heads",
      self.num_attention_heads,
    )?;
    require_divisible(
      "Config: hidden_size",
      self.hidden_size,
      "Config: num_conv_pos_embedding_groups",
      self.num_conv_pos_embedding_groups,
    )?;
    // Layer counts size eager per-layer `Vec`s (the encoder-layer `Vec`, the
    // feature-encoder conv `Vec`, and — on the MMS path — an `O(num_hidden_layers)`
    // adapter-key `Vec` + allowed-key set), so each is bounded by the
    // config-cardinality cap, not merely checked positive: a non-positive count
    // is [`Error::OutOfRange`] and an over-cap one is [`Error::CapExceeded`]
    // (matching `qwen3` / `lfm2`). The within-cap reservations are still made
    // fallibly by the builders / overlay, so a within-cap-but-heavyweight count
    // surfaces as a typed [`Error::AllocFailure`] rather than an abort.
    require_cardinality(
      "Config: num_hidden_layers",
      i64::from(self.num_hidden_layers),
      MAX_CONFIG_CARDINALITY as u64,
    )?;
    require_cardinality(
      "Config: num_feat_extract_layers",
      i64::from(self.num_feat_extract_layers),
      MAX_CONFIG_CARDINALITY as u64,
    )?;
    // `eps` shared by every LayerNorm and the L0 GroupNorm: must be a finite,
    // positive scalar (it varies across the family, so it is not pinned to a
    // magnitude). A non-finite value would drive a non-finite denominator.
    if !self.layer_norm_eps.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Config: layer_norm_eps",
        f64::from(self.layer_norm_eps),
      )));
    }
    if self.layer_norm_eps <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Config: layer_norm_eps",
        "must be a positive finite scalar (> 0)",
        format_smolstr!("{}", self.layer_norm_eps),
      )));
    }
    // Conv-stack arrays: each length must EQUAL `num_feat_extract_layers`
    // exactly (a longer array would desync the feature encoder from the
    // projection, which reads `conv_dim.last()`) and carry only positive
    // entries (widths, kernels, strides the conv stack consumes).
    // `num_feat_extract_layers` is positive (checked above), so the cast fits.
    let n = self.num_feat_extract_layers as usize;
    check_conv_array("Config: conv_dim", &self.conv_dim, n)?;
    check_conv_array("Config: conv_stride", &self.conv_stride, n)?;
    check_conv_array("Config: conv_kernel", &self.conv_kernel, n)?;
    Ok(())
  }

  /// Per-head dimension `hidden_size / num_attention_heads`.
  fn head_dim(&self) -> Result<i32> {
    require_positive("Config: num_attention_heads", self.num_attention_heads)?;
    require_divisible(
      "Config: hidden_size",
      self.hidden_size,
      "Config: num_attention_heads",
      self.num_attention_heads,
    )?;
    Ok(self.hidden_size / self.num_attention_heads)
  }
}

/// Validate a `conv_dim` / `conv_stride` / `conv_kernel` config array: its
/// length must **equal** `n` (`num_feat_extract_layers`) exactly, and every
/// entry must be positive.
///
/// The exact-length requirement is the faithful invariant — the reference's
/// `conv_dim` / `conv_stride` / `conv_kernel` arrays each have one entry per
/// feature-extractor layer (their length *is* `num_feat_extract_layers`). A
/// too-short array would index past the end; a too-long one would silently
/// desync the feature encoder from the projection — the encoder builds `n`
/// layers (so its output width is `conv_dim[n-1]`) while
/// [`build_feature_projection`] reads `conv_dim.last()` (a *later* trailing
/// entry), so they would expect different channel counts. A length mismatch in
/// either direction is therefore [`Error::LengthMismatch`]; a non-positive
/// element is [`Error::OutOfRange`] naming the index and value. (The builder
/// consumes `conv_dim[i]` as the conv output width, `conv_dim[i-1]` as the
/// input width, `conv_stride[i]` as the stride, and `conv_kernel[i]` for the
/// pinned weight shape — each must be a positive integer.)
#[cfg(feature = "wav2vec2")]
fn check_conv_array(field: &'static str, array: &[i32], n: usize) -> Result<()> {
  if array.len() != n {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      field,
      n,
      array.len(),
    )));
  }
  for (i, &v) in array.iter().enumerate() {
    if v <= 0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        field,
        "every conv-stack entry must be a positive integer (> 0)",
        format_smolstr!("element {i} = {v}"),
      )));
    }
  }
  Ok(())
}

// ───────────────────────────── activation ─────────────────────────────

/// The element-wise activation a [`Config`] selects, resolved from an
/// HF activation name.
///
/// mlx-audio hardcodes `nn.GELU()` (the exact GELU) at every block and ignores
/// the config's `hidden_act` / `feat_extract_activation`; this port instead
/// honours them, mapping the HF names exactly. `"gelu"` resolves to the exact
/// GELU, so a `gelu` checkpoint (every `base`/`large` wav2vec2/hubert CTC
/// model) is bit-for-bit the reference's behaviour.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
  /// The exact GELU (`mlx.nn.GELU(approx="none")`) — HF `"gelu"`.
  Gelu,
  /// The `tanh` approximation of GELU (`mlx.nn.gelu_approx`) — HF `"gelu_new"`
  /// / `"gelu_pytorch_tanh"`.
  GeluApprox,
  /// SiLU / Swish (`x · σ(x)`) — HF `"silu"` / `"swish"`.
  Silu,
}

/// HF activation names this port supports, for the [`Error::UnknownEnumValue`]
/// `supported` list. Mirrors the [`Activation::resolve`] match arms.
#[cfg(feature = "wav2vec2")]
const SUPPORTED_ACTIVATIONS: &[&str] = &["gelu", "gelu_new", "gelu_pytorch_tanh", "silu", "swish"];

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Activation {
  /// Resolve an HF activation name to an [`Activation`], or reject an
  /// unsupported name with [`Error::UnknownEnumValue`] (carrying `field`, the
  /// offending value, and the supported-activation list).
  ///
  /// The name mapping follows HF transformers' `ACT2FN`: `"gelu"` is the exact
  /// GELU; `"gelu_new"` and `"gelu_pytorch_tanh"` are the tanh approximation;
  /// `"silu"` and `"swish"` are SiLU.
  pub fn resolve(name: &str, field: &'static str) -> Result<Self> {
    match name {
      "gelu" => Ok(Self::Gelu),
      "gelu_new" | "gelu_pytorch_tanh" => Ok(Self::GeluApprox),
      "silu" | "swish" => Ok(Self::Silu),
      other => Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        field,
        other,
        SUPPORTED_ACTIVATIONS,
      ))),
    }
  }

  /// Apply the activation to `x`, dispatching to the dtype-preserving
  /// primitives in [`crate::lm::nn::activations`]. Returns a new lazy
  /// [`Array`] (no implicit eval).
  fn forward(&self, x: &Array) -> Result<Array> {
    use crate::lm::nn::activations;
    match self {
      Self::Gelu => activations::gelu(x),
      Self::GeluApprox => activations::gelu_approx(x),
      Self::Silu => activations::silu(x),
    }
  }
}

// ───────────────────────────── sanitize ─────────────────────────────

/// Rewrite an HF `*ForCTC` checkpoint (`Wav2Vec2ForCTC` / `HubertForCTC`) into
/// the layout this port loads — the Rust analogue of mlx-audio's
/// `Model.sanitize` ([mms.py:107-128][san], plus the backbone's prefix strip
/// from [wav2vec.py:723][san-bb]). Pure key/axis bookkeeping, no MLX evaluation
/// beyond the lazy `swapaxes`.
///
/// Rules (applied per `(key, value)`):
/// 1. Strip a leading supported backbone prefix (`wav2vec2.` for
///    Wav2Vec2ForCTC, `hubert.` for HubertForCTC) — each `*ForCTC` nests the
///    encoder under its backbone name while `lm_head` stays top-level. A key
///    carries at most one, so whichever leads is stripped.
/// 2. `*.conv.weight` and `*.conv.weight_v` / `*.conv.weight_g`:
///    `swapaxes(1, 2)` (HF conv weight `(out, in, k)` → MLX channels-last
///    `(out, k, in)`).
/// 3. `*.parametrizations.weight.original0` → `*.weight_g`,
///    `*.parametrizations.weight.original1` → `*.weight_v` (the PyTorch
///    weight-norm reparametrization rename), each also `swapaxes(1, 2)`.
/// 4. **Drop** training-only keys: `quantizer.*`, `project_*`,
///    `masked_spec_embed`.
/// 5. **Keep** `lm_head.*` — the CTC head (the backbone's own `sanitize`
///    drops it; this composed model needs it).
/// 6. **Reject a duplicate** destination key with [`Error::KeyCollision`]
///    (via [`crate::model_validation::insert_unique`]) rather than silently
///    overwriting — e.g. a checkpoint carrying both the prefixed and
///    unprefixed form of a key (rule 1, e.g. `hubert.<x>` and `<x>`), or both
///    `parametrizations.weight.original0` and a legacy `weight_g` (rule 3),
///    which would otherwise let an arbitrary (per-run nondeterministic)
///    survivor win since the source is a `HashMap`.
///
/// [san]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L107-L128
/// [san-bb]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L723
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn sanitize(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
  let mut out = HashMap::with_capacity(weights.len());
  for (mut k, mut v) in weights {
    // 1. Backbone prefix: each `*ForCTC` nests the encoder under its backbone
    //    name (`wav2vec2.` for Wav2Vec2ForCTC, `hubert.` for HubertForCTC —
    //    HuBERT reuses the wav2vec2 encoder) while `lm_head` stays top-level.
    //    Strip whichever supported prefix leads (a key cannot carry two), so
    //    the unprefixed `feature_extractor.*` / `encoder.*` keys the builders
    //    expect are produced regardless of the backbone the checkpoint used.
    for prefix in SUPPORTED_BACKBONE_PREFIXES {
      if let Some(stripped) = k.strip_prefix(prefix) {
        k = stripped.to_string();
        break;
      }
    }

    // 2 + 3. Conv weight axis swaps and weight-norm parametrization renames.
    if k.ends_with(".conv.weight") || k.ends_with(".conv.weight_v") || k.ends_with(".conv.weight_g")
    {
      v = ops::shape::swapaxes(&v, 1, 2)?;
    } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original0") {
      k = format!("{base}.weight_g");
      v = ops::shape::swapaxes(&v, 1, 2)?;
    } else if let Some(base) = k.strip_suffix(".parametrizations.weight.original1") {
      k = format!("{base}.weight_v");
      v = ops::shape::swapaxes(&v, 1, 2)?;
    }

    // 4. Drop training-only keys. `lm_head.*` is deliberately NOT dropped
    //    here (unlike the backbone's own sanitize) — it is the CTC head.
    if k.starts_with("quantizer.") || k.starts_with("project_") || k == "masked_spec_embed" {
      continue;
    }

    // 6. Insert, rejecting a duplicate destination key with a typed error
    //    rather than letting an arbitrary survivor silently overwrite the
    //    other. Two source keys can collide here — e.g. a checkpoint carrying
    //    both a prefixed `wav2vec2.<x>` / `hubert.<x>` and the unprefixed `<x>`
    //    form (rule 1), or both `parametrizations.weight.original0` and the
    //    legacy `weight_g` (rule 3) — and the source is a `HashMap`, so the
    //    surviving tensor would otherwise be per-run nondeterministic.
    insert_unique(&mut out, k, v, "Wav2Vec2 sanitize")?;
  }
  Ok(out)
}

/// Reproject the per-layer [`PerLayerQuantization`] override keys into the same
/// sanitized namespace the builders resolve a layer's scheme in.
///
/// [`sanitize`] strips a leading supported backbone prefix (`wav2vec2.` /
/// `hubert.`, [`SUPPORTED_BACKBONE_PREFIXES`]) from every weight key, and the
/// builders then look a layer's scheme up by its **sanitized** prefix (e.g.
/// `encoder.layers.0.attention.q_proj`) via
/// [`PerLayerQuantization::quantization_for`]. An on-disk `config.json`, by
/// contrast, keys its per-layer overrides in the HF form that carries the
/// backbone prefix (`wav2vec2.encoder.layers.0.attention.q_proj`). Without this
/// rewrite an override would never match the sanitized lookup and the layer
/// would silently fall back to the global scheme — loading with the wrong
/// `(group_size, bits)` or being rejected by the shape gate. Mirrors the
/// per-tower key reprojection the embeddings towers apply for the same reason.
///
/// Each override key has the leading backbone prefix stripped (a key already in
/// the sanitized form, e.g. the top-level `lm_head`, is left unchanged). The
/// global default ([`PerLayerQuantization::quantization`]) is carried through
/// untouched. Collisions are handled **deterministically**, never by an
/// arbitrary `HashMap` survivor: two source keys that reproject to the same
/// sanitized key are deduplicated when their [`QuantizationOption`] is identical
/// and rejected with [`Error::KeyCollision`] when it conflicts (a checkpoint
/// carrying both `wav2vec2.<x>` and the unprefixed `<x>` with different schemes
/// is a genuine config contradiction).
///
/// This is **idempotent**: an already-sanitized key carries no leading backbone
/// prefix, so re-stripping it is a no-op and a key unchanged by the first pass
/// is unchanged by a second. Re-normalizing an already-normalized
/// [`PerLayerQuantization`] therefore returns an equivalent map — which lets
/// [`Model::from_weights_quantized`] normalize unconditionally at its
/// single boundary without double-strip hazards.
#[cfg(feature = "wav2vec2")]
fn reproject_quant_keys(quant: &PerLayerQuantization) -> Result<PerLayerQuantization> {
  let per_layer = quant.per_layer_ref();
  let mut out: HashMap<String, QuantizationOption> = HashMap::new();
  // The reprojected map has at most as many entries as the source (a strip can
  // only collapse two keys onto one, never add); reserve fallibly so a hostile
  // config's huge per-layer set surfaces a typed `AllocFailure`.
  reserve_or_error(&mut out, "Wav2Vec2 reproject_quant_keys", per_layer.len())?;
  for (key, &opt) in per_layer {
    // Strip whichever supported backbone prefix leads (a key carries at most
    // one); a key already unprefixed (e.g. the top-level `lm_head`) is kept.
    let stripped = SUPPORTED_BACKBONE_PREFIXES
      .iter()
      .find_map(|p| key.strip_prefix(p))
      .unwrap_or(key.as_str());
    match out.get(stripped) {
      // Same destination, identical scheme: a benign duplicate (e.g. a
      // prefixed + unprefixed form of one layer that agree). Keep the single
      // entry without growing the map.
      Some(existing) if *existing == opt => {}
      // Same destination, conflicting scheme: a genuine config contradiction —
      // a typed error, never an arbitrary `HashMap` survivor.
      Some(_) => {
        return Err(Error::KeyCollision(KeyCollisionPayload::new(
          "Wav2Vec2 reproject_quant_keys: two per-layer overrides reproject to the same sanitized layer with conflicting schemes",
          format_smolstr!("{stripped}"),
        )));
      }
      // First occurrence of this sanitized key: insert it (the map is already
      // reserved to the source length, so this cannot reallocate).
      None => {
        out.insert(stripped.to_string(), opt);
      }
    }
  }
  Ok(PerLayerQuantization::new(quant.quantization, out))
}

// ───────────────────────── MMS per-language adapter ─────────────────────────

/// The MMS adapter file discovered for a requested language: the path **and**
/// the language code actually selected (which differs from the requested one
/// when discovery falls back to an available adapter).
///
/// The caller selects the per-language `vocab.json` map from [`Self::lang`] —
/// **not** the originally-requested language — so the overlaid adapter, the
/// per-language `lm_head`, and the vocabulary always describe the **same**
/// language (a French fallback adapter is decoded with the French token table,
/// never the requested English one).
#[cfg(feature = "wav2vec2")]
struct SelectedAdapter {
  /// The **canonicalized** on-disk `adapter.{lang}.safetensors` path to overlay
  /// — the exact path that passed the under-`dir` no-escape check, *not* the
  /// raw directory entry. The loader opens **this** path, so the path that was
  /// validated to stay under the model directory is the path that is read: a
  /// mutable model directory cannot swap a checked symlink/file between the
  /// check and the open (a TOCTOU re-resolution of the unchecked original is
  /// impossible because the original is never reopened after validation).
  path: std::path::PathBuf,
  /// The ISO-639-3 code of the adapter that was actually selected — the exact
  /// requested language when its file exists, else the fallback's language.
  lang: String,
}

/// Discover the MMS per-language adapter safetensors file in `dir` for
/// `target_lang` — the Rust analogue of mlx-audio's
/// `Model.post_load_hook` adapter-file discovery ([mms.py:134-138][mms]).
///
/// Prefers the `adapter.{target_lang}.safetensors` whose `<lang>` segment equals
/// `target_lang`; if no such file is present, falls back to the
/// lexicographically-smallest `adapter.*.safetensors` in the directory (a
/// deterministic substitute for the reference's `list(glob(...))[0]`, whose
/// order is filesystem-dependent). Returns `None` when the directory holds no
/// `adapter.*.safetensors` at all (a plain wav2vec2 / HuBERT checkpoint, which
/// has no language adapter) — that is not an error, only the absence of an
/// overlay.
///
/// **The selected path is never built by interpolating `target_lang` into a
/// filename.** It always comes from a real directory entry discovered by
/// `read_dir`: the candidate list is enumerated from the on-disk
/// `adapter.*.safetensors` files, and `target_lang` is matched against each
/// file's **extracted** `<lang>` segment (a single path component, so it can
/// carry no `/`, `\`, or `..`). A caller-controlled `target_lang` containing
/// path separators or `..` therefore cannot select a file outside `dir` — it
/// simply matches nothing and the fallback (an in-`dir` file) is used. As
/// defense in depth the selected path is canonicalized and required to stay
/// under the canonicalized `dir`, so a symlinked `adapter.*.safetensors`
/// pointing outside the directory is rejected with a typed error rather than
/// overlaid. The canonical basename's `<lang>` is **also** required to equal the
/// discovered entry's `<lang>`, so an in-`dir` symlink retargeting to **another**
/// language's file (`adapter.eng.safetensors -> adapter.fra.safetensors`) is a
/// typed mismatch rather than a silent open of French weights under the English
/// vocabulary.
///
/// Returns the **canonicalized** selected path (the exact path that passed the
/// under-`dir` check, so the loader opens that validated path — not the
/// unchecked raw directory entry — closing the swap-between-check-and-open
/// TOCTOU) **with** the language code that was selected (the requested
/// `target_lang` on an exact hit, else the fallback file's `<lang>` segment) —
/// and that language is guaranteed to match the canonical file the loader opens.
/// The caller aligns the per-language `vocab.json` selection to this returned
/// code, so the adapter, the `lm_head`, and the vocab never diverge (the
/// divergence the reference's `glob[0]` fallback would otherwise cause).
///
/// [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L134-L138
#[cfg(feature = "wav2vec2")]
fn adapter_file_for(dir: &std::path::Path, target_lang: &str) -> Result<Option<SelectedAdapter>> {
  // Enumerate the on-disk `adapter.*.safetensors` files. The selected path is
  // taken from a real directory entry — NEVER built by interpolating
  // `target_lang` into a filename — so a caller-controlled `target_lang` with
  // path separators or `..` cannot escape `dir`: it can only match (or fail to
  // match) an extracted in-`dir` `<lang>`. Read the dir entries; a read error (a
  // non-existent / unreadable dir) is a typed error.
  let read = std::fs::read_dir(dir).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "Model::load: reading the model directory for MMS adapters",
      FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  // The exact hit (a file whose extracted `<lang>` equals `target_lang`), and —
  // independently — the lexicographically-smallest filename as the fallback. The
  // exact hit wins when present; otherwise the smallest is used.
  let mut exact: Option<(std::path::PathBuf, String)> = None;
  let mut smallest: Option<(std::path::PathBuf, String)> = None;
  for entry in read {
    let entry = entry.map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "Model::load: reading a directory entry for MMS adapters",
        FileOp::Read,
        dir.to_path_buf(),
        e,
      ))
    })?;
    let name = entry.file_name();
    let Some(name) = name.to_str() else { continue };
    // `adapter.<lang>.safetensors` with a non-empty `<lang>` segment.
    if name.starts_with(ADAPTER_FILE_PREFIX)
      && name.ends_with(ADAPTER_FILE_SUFFIX)
      && name.len() > ADAPTER_FILE_PREFIX.len() + ADAPTER_FILE_SUFFIX.len()
    {
      // The `<lang>` is the filename with the `adapter.` prefix and
      // `.safetensors` suffix stripped (the length check above guarantees a
      // non-empty middle segment). It is a single path component of a real
      // directory entry, so it can never contain a path separator or `..`.
      let lang = &name[ADAPTER_FILE_PREFIX.len()..name.len() - ADAPTER_FILE_SUFFIX.len()];
      // An exact match on the extracted `<lang>` is the preferred selection —
      // matched by VALUE against the real filename, not by reconstructing a path
      // from `target_lang`.
      if lang == target_lang && exact.is_none() {
        exact = Some((entry.path(), lang.to_string()));
      }
      // Keep the lexicographically-smallest filename for the deterministic
      // fallback.
      let take = match &smallest {
        Some((existing, _)) => Some(name) < existing.file_name().and_then(|n| n.to_str()),
        None => true,
      };
      if take {
        smallest = Some((entry.path(), lang.to_string()));
      }
    }
  }
  // Prefer the exact-language hit; else the smallest fallback.
  let Some((path, lang)) = exact.or(smallest) else {
    return Ok(None);
  };
  // Defense in depth: the selected path is a real directory entry, so it is
  // already in `dir` by construction — but a symlinked `adapter.*.safetensors`
  // could resolve elsewhere. Canonicalize the selection and require it to stay
  // under the canonicalized `dir`; a path that escapes (a symlink out of the
  // model directory) is a typed load failure, never overlaid.
  let canon_dir = dir.canonicalize().map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "Model::load: canonicalizing the model directory for MMS adapters",
      FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  let canon_path = path.canonicalize().map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "Model::load: canonicalizing the selected MMS adapter file",
      FileOp::Read,
      path.clone(),
      e,
    ))
  })?;
  if !canon_path.starts_with(&canon_dir) {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "Model::load: the selected MMS adapter file resolves outside the model directory",
      "an adapter.{lang}.safetensors must stay under the model directory (a symlink escaping it is \
       rejected)",
    )));
  }
  // The `lang` above is the `<lang>` extracted from the directory ENTRY name. A
  // symlinked `adapter.{lang}.safetensors` could resolve (still under `dir`) to a
  // file named for a DIFFERENT language (`adapter.eng.safetensors -> \
  // adapter.fra.safetensors`): the loader would then open the canonical (French)
  // weights while the caller selected the entry's (English) vocab — silently
  // decoding French logits with an English token table. Re-derive the language
  // from the CANONICAL basename and require it to equal the entry-extracted one,
  // so the opened weights and the language the vocab is keyed on always agree. A
  // canonical name with no `adapter.<lang>.safetensors` shape, or one naming a
  // different language, is a typed mismatch (never a silent hybrid). (Resolving
  // through the symlink to the same language — `real/adapter.fra.safetensors` —
  // still agrees and is accepted.)
  let canon_name = canon_path.file_name().and_then(|n| n.to_str());
  let canon_lang = canon_name.and_then(|name| {
    (name.starts_with(ADAPTER_FILE_PREFIX)
      && name.ends_with(ADAPTER_FILE_SUFFIX)
      && name.len() > ADAPTER_FILE_PREFIX.len() + ADAPTER_FILE_SUFFIX.len())
    .then(|| &name[ADAPTER_FILE_PREFIX.len()..name.len() - ADAPTER_FILE_SUFFIX.len()])
  });
  if canon_lang != Some(lang.as_str()) {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "Model::load: the selected MMS adapter resolves to a file whose language differs from the \
       discovered adapter name (a symlink retargeting to another language's weights would decode \
       its logits with the wrong vocabulary)",
      "an adapter.{lang}.safetensors must resolve to a file named for the SAME {lang} so the \
       overlaid weights and the selected vocabulary describe one language",
    )));
  }
  // Return the CANONICALIZED path that just passed the under-`dir` check — never
  // the unchecked original `entry.path()`. The loader opens this exact path, so
  // the path validated to stay inside the model directory is the path read: a
  // mutable model dir cannot swap a checked symlink/file between this check and
  // the later open, because the original is never re-resolved after validation
  // (closing the TOCTOU the no-escape invariant would otherwise leave open).
  Ok(Some(SelectedAdapter {
    path: canon_path,
    lang,
  }))
}

/// `true` when `config` is an **MMS** checkpoint that *requires* a per-language
/// adapter: `adapter_attn_dim` is set **and** the stable-layer-norm arm is
/// selected — the exact condition under which [`Standard::build_encoder`]
/// attaches a `Wav2Vec2AttnAdapterLayer` to every encoder layer
/// ([wav2vec.py:484-487][sel-ad]). `facebook/mms-1b-all` / `mms-1b-fl102` ship
/// this shape and cannot transcribe without an `adapter.{lang}.safetensors`
/// (the base `model.safetensors` carries only the language-agnostic init), so
/// the loader treats a missing/incomplete adapter as a typed load failure
/// rather than silently building from the base init.
///
/// A post-norm checkpoint that merely carries `adapter_attn_dim` builds **no**
/// adapter (the reference attaches the adapter only to the stable-LN layer), so
/// it is **not** MMS by this predicate: an absent adapter there stays a correct
/// no-op (matching the no adapter layers it builds).
///
/// [sel-ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L484-L487
#[cfg(feature = "wav2vec2")]
fn config_requires_adapter(config: &Config) -> bool {
  config.adapter_attn_dim.is_some() && config.do_stable_layer_norm
}

/// The exact post-[`sanitize`] key set an MMS adapter overlay **must** supply,
/// derived from `config` — the per-layer adapter-layer tensors for every
/// encoder layer plus the per-language CTC head, matching the keys
/// [`Standard::build_encoder`] / [`Model::from_weights_quantized`] consume when
/// `config_requires_adapter(config)` holds.
///
/// Each of the `num_hidden_layers` layers contributes its
/// `encoder.layers.{i}.adapter_layer.{norm.weight, norm.bias, linear_1.weight,
/// linear_1.bias, linear_2.weight, linear_2.bias}` (the `LayerNorm → Linear →
/// ReLU → Linear` bottleneck), and the head contributes `lm_head.weight` /
/// `lm_head.bias`. Built from the config (layer count + the fixed key scheme),
/// never a hardcoded list, so it tracks the model's actual structure. For a
/// **quantized** adapter the `linear_{1,2}` / `lm_head` `.weight` is the packed
/// `uint32` triple's weight; the `.scales`/`.biases` sidecars are validated
/// structurally by [`build_linear`], so this required-key floor (the `.weight`
/// alongside the bias) is what a dense **and** a quantized overlay both carry.
///
/// The `O(num_hidden_layers)` key `Vec` is reserved **fallibly**: although
/// [`Config::validate`] bounds `num_hidden_layers` by [`MAX_CONFIG_CARDINALITY`],
/// the reservation goes through [`reserve_or_error`] so a within-cap-but-large
/// count that exhausts memory surfaces as a typed [`Error::AllocFailure`] rather
/// than the abort `Vec::with_capacity` would raise (the crate's allocation
/// discipline — no infallible heap allocation sized from a config field).
/// `true` when `prefix` names an adapter weight loaded through [`build_linear`]
/// (so a quantized overlay may carry its `.scales` / `.biases` affine sidecars):
/// the per-layer adapter linear projections (`…adapter_layer.linear_1` /
/// `…adapter_layer.linear_2`) and the CTC `lm_head`.
///
/// The adapter `LayerNorm` (`…adapter_layer.norm`) is loaded by `take_shaped`,
/// **not** `build_linear`, so it is never quantized and carries no sidecars — a
/// `…adapter_layer.norm.scales` is a key `build_linear` would never read, so the
/// allowed-key set must **not** admit it merely because `norm.weight` is present.
#[cfg(feature = "wav2vec2")]
fn is_quantizable_linear_prefix(prefix: &str) -> bool {
  prefix == "lm_head"
    || prefix.ends_with(".adapter_layer.linear_1")
    || prefix.ends_with(".adapter_layer.linear_2")
}

#[cfg(feature = "wav2vec2")]
fn expected_adapter_keys(config: &Config) -> Result<Vec<String>> {
  let n = config.num_hidden_layers.max(0) as usize;
  // 6 adapter-layer tensors per layer + the 2 lm_head tensors. Reserved fallibly
  // (the count is config-derived) — a `try_reserve_exact` failure becomes a typed
  // `AllocFailure`, never an abort.
  let cap = n.saturating_mul(6).saturating_add(2);
  let mut keys: Vec<String> = Vec::new();
  reserve_or_error(&mut keys, "Wav2Vec2 expected_adapter_keys", cap)?;
  for i in 0..n {
    let p = format!("encoder.layers.{i}.adapter_layer");
    keys.push(format!("{p}.norm.weight"));
    keys.push(format!("{p}.norm.bias"));
    keys.push(format!("{p}.linear_1.weight"));
    keys.push(format!("{p}.linear_1.bias"));
    keys.push(format!("{p}.linear_2.weight"));
    keys.push(format!("{p}.linear_2.bias"));
  }
  keys.push("lm_head.weight".to_string());
  keys.push("lm_head.bias".to_string());
  Ok(keys)
}

/// Overlay an MMS per-language adapter checkpoint onto the (sanitized) base
/// weight map — the Rust analogue of mlx-audio's
/// `model.load_weights(sanitized, strict=False)` ([mms.py:140-143][mms]).
///
/// The base `model.safetensors` carries the language-agnostic adapter init
/// (and the language-agnostic `lm_head`); the per-language
/// `adapter.{lang}.safetensors` carries the trained adapter-layer weights and
/// the per-language `lm_head` for that language. Loading the adapter on top
/// **replaces** exactly those keys in the base map (an
/// `strict=False`-equivalent partial overlay: the adapter file is a subset of
/// the model's keys), so the single subsequent `from_weights` build produces a
/// model specialized to the loaded language — identical to the reference
/// loading the base module then overlaying the adapter params.
///
/// The adapter file is loaded and run through the same [`sanitize`] the base
/// checkpoint uses (its keys are the backbone-relative
/// `encoder.layers.{i}.adapter_layer.*` / `lm_head.*` forms — no conv axis
/// swaps apply, `lm_head` is kept). Each sanitized adapter key is inserted into
/// `base` (replacing the base init).
///
/// **Exact-key validation (every overlay) + completeness floor
/// (`require_complete`).** The key-restriction — the overlay carries **only**
/// allowed keys, with no orphan sidecar — runs for **every** overlay regardless
/// of `require_complete`; only the required-key **floor** (all expected keys
/// present) is gated on `require_complete`. So the overlay is validated against
/// the **exact** allowed key set **before** any merge (no partial overlay is
/// applied):
/// - it must carry **no** key *outside* the allowed set — the expected `.weight`
///   / `.bias` keys plus, for each **`build_linear`-loaded** weight prefix (the
///   adapter `linear_1` / `linear_2` projections and `lm_head`), its **optional**
///   quant sidecars `<prefix>.scales` / `<prefix>.biases`. The adapter `LayerNorm`
///   (`adapter_layer.norm`) is loaded by `take_shaped`, not `build_linear`, so it
///   is never quantized: a `adapter_layer.norm.scales` is **not** admitted (a
///   sidecar `build_linear` would never read). An extra/foreign key — a stray
///   `encoder.layers.0.attention.q_proj.scales` with no matching `q_proj.weight`,
///   or a `adapter_layer.norm.scales` on a non-quantizable prefix — would
///   otherwise clobber a base tensor / smuggle a malformed key, so it is
///   [`Error::KeyCollision`] (the key conflicts with the exact adapter contract)
///   — checked for **any** overlay;
/// - every quant sidecar must travel **with** its `.weight`: an orphan
///   `<prefix>.scales` / `<prefix>.biases` whose `<prefix>.weight` is **not** in
///   the overlay is [`Error::MissingKey`] (the absent companion weight); and a
///   `<prefix>.biases` (the affine half of a quantized triple) with **no**
///   `<prefix>.scales` sibling is likewise [`Error::MissingKey`] (a `.biases` is
///   read only alongside its `.scales`) — both checked for **any** overlay;
/// - when `require_complete` (an MMS config), it must **also** carry **every**
///   key in [`expected_adapter_keys`] (all per-layer adapter tensors +
///   `lm_head.weight` / `lm_head.bias`); a missing key is [`Error::MissingKey`].
///   An MMS model cannot run on a base that is only language-agnostically
///   initialized — a missing/truncated adapter would silently build the wrong
///   (hybrid/base) model — so its overlay is required to be complete.
///
/// Net: **any** adapter overlay must carry **only** the allowed adapter-layer +
/// `lm_head` weights/biases (+ optional matching sidecars) with no foreign key
/// and no orphan sidecar, or it is rejected with a typed error naming the
/// offending key — structurally closing the malformed-adapter-key class (a
/// sidecar-only quant hybrid can no longer be smuggled in via **any** overlay
/// path). An MMS (`require_complete`) overlay must **additionally** be complete.
/// (In the loader, discovery + overlay run only for an MMS config — a non-MMS
/// checkpoint never overlays an adapter file at all, faithful to mms.py; this
/// per-overlay key-restriction is the in-function defense in depth.)
///
/// **Quant identity is fully self-contained from the adapter.** [`build_linear`]
/// keys the quantized path on a `<prefix>.scales` sibling and pops a matching
/// `<prefix>.biases` for the affine triple. So for **every** `<prefix>.weight`
/// the overlay supplies, **both** stale base sidecars (`<prefix>.scales` and
/// `<prefix>.biases`) are removed from `base` **before** the overlay's own
/// tensors are inserted; the prefix then carries exactly — and only — the
/// sidecars the **adapter file** itself supplies. A **dense** overlay (a
/// `.weight` with no `.scales`) loads the prefix dense (both stale base sidecars
/// gone, so [`build_linear`] cannot mis-read the dense F32 weight as a packed
/// `uint32` triple). A **quantized** overlay (`.weight` + `.scales`
/// [+ `.biases`]) loads quantized from its **own** sidecars — never the base's.
/// In particular a quantized overlay that supplies `.scales` but is **missing**
/// `.biases` no longer inherits the base `.biases`: it surfaces an
/// incomplete-quant failure downstream ([`build_linear`] →
/// [`crate::nn::QuantizedLinear::from_parts`] rejects an affine triple with no
/// biases as a typed [`Error::InvariantViolation`]) rather than silently
/// building a hybrid (adapter weight + scales over **base** biases).
///
/// [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L140-L143
#[cfg(feature = "wav2vec2")]
fn overlay_adapter_weights(
  base: &mut HashMap<String, Array>,
  adapter_path: &std::path::Path,
  config: &Config,
  require_complete: bool,
) -> Result<()> {
  let raw = crate::io::load_safetensors(adapter_path)?;
  let sanitized = sanitize(raw)?;
  // ANY adapter overlay — MMS or a stray `adapter.*.safetensors` on a non-MMS
  // checkpoint — must match the model's adapter surface EXACTLY: it must carry
  // NO key the model does not consume, and every quant sidecar it carries must
  // travel WITH its `.weight`. A merely-required-keys check (gated on
  // `require_complete`) would let a malformed/stray adapter smuggle in an
  // EXTRA/ORPHAN key (e.g. an `encoder.layers.0.attention.q_proj.scales` with no
  // matching `q_proj.weight`): the blind insert below would then overwrite the
  // base packed weight's `.scales` while leaving the base `.weight` + base
  // `.biases`, silently building a hybrid quantized layer from mismatched parts.
  // So the EXACT allowed key set — the expected adapter/lm_head `.weight` /
  // `.bias` keys plus, for each `build_linear`-loaded weight prefix (the adapter
  // `linear_1` / `linear_2` projections and `lm_head`, NOT the `take_shaped`
  // LayerNorm `norm`), its OPTIONAL quant sidecars `<prefix>.scales` /
  // `<prefix>.biases` — is enforced for EVERY overlay regardless of
  // `require_complete` (the foreign-key + orphan-sidecar checks below), and ONLY
  // the required-key FLOOR (every expected key present) is gated on
  // `require_complete` (an MMS config requires a COMPLETE adapter; a non-MMS
  // checkpoint with a partial/stray adapter is rejected by the foreign/orphan
  // checks rather than silently overlaid). All checks run BEFORE any merge (no
  // partial overlay).
  //
  // The exact set of keys an adapter overlay may carry: every required `.weight`
  // / `.bias` key, plus (optionally) the two quant sidecars for each QUANTIZABLE
  // (`build_linear`-loaded) weight prefix. A LayerNorm `<…norm>.scales` is NOT
  // admitted — `norm` is loaded by `take_shaped`, never quantized, so such a
  // sidecar would be a key `build_linear` never reads, and admitting it merely
  // because `norm.weight` is present would let a malformed key pass the
  // foreign-key check. Built from `expected_adapter_keys` so it tracks the
  // config's layer count + key scheme — never a hardcoded list. Both the
  // required-key `Vec` and this allowed-key set are `O(num_hidden_layers)`
  // (config-derived), so both are reserved FALLIBLY (a `try_reserve` failure
  // becomes a typed `AllocFailure`, never an abort — the crate's allocation
  // discipline).
  let required = expected_adapter_keys(config)?;
  let mut allowed: std::collections::HashSet<String> = std::collections::HashSet::new();
  reserve_or_error(
    &mut allowed,
    "Wav2Vec2 adapter allowed-key set",
    required.len().saturating_mul(2),
  )?;
  for key in &required {
    // Admit the optional affine sidecars (`.scales`/`.biases`) ONLY for the
    // weight prefixes actually loaded through `build_linear` — the adapter linear
    // projections (`adapter_layer.linear_1` / `adapter_layer.linear_2`) and the
    // CTC `lm_head`. The adapter `LayerNorm` (`adapter_layer.norm`) is loaded by
    // `take_shaped`, NOT `build_linear`, so it is never quantized: admitting
    // `adapter_layer.norm.scales` would let a malformed key pass the foreign-key
    // check merely because `norm.weight` is present (a sidecar `build_linear`
    // would never read). A `.biases` is only valid ALONGSIDE a `.scales` (the
    // affine triple), so it is admitted only with its `.scales` — never alone.
    if let Some(prefix) = key
      .strip_suffix(".weight")
      .filter(|p| is_quantizable_linear_prefix(p))
    {
      allowed.insert(format!("{prefix}.scales"));
      allowed.insert(format!("{prefix}.biases"));
    }
    allowed.insert(key.clone());
  }
  // (b) + (c) — validate every key the overlay actually carries, for ANY overlay
  // (not just an MMS `require_complete` one), so a stray sidecar-only adapter on
  // a non-MMS checkpoint can no longer clobber a base tensor. Run before the
  // required-key floor so the structural defects (a foreign key / an orphan
  // sidecar) are surfaced at their precise offending key.
  for key in sanitized.keys() {
    // (b) The overlay must carry NO key outside the allowed set — an EXTRA/foreign
    // key (e.g. a stray `encoder.layers.0.attention.q_proj.scales` for a layer
    // the adapter does not overlay) would clobber a base tensor and build a
    // silent hybrid. Reject it naming the offending key (a `KeyCollision`: the
    // key conflicts with the exact adapter contract). Checked first so a FOREIGN
    // sidecar (whose prefix is not even an expected weight) is diagnosed as the
    // foreign key it is, not as an orphan of a never-expected weight.
    if !allowed.contains(key) {
      return Err(Error::KeyCollision(KeyCollisionPayload::new(
        "Model::load: the adapter file carries an unexpected key (an adapter overlay \
         must supply ONLY the adapter-layer + lm_head weights/biases, plus optional matching \
         .scales/.biases sidecars — and nothing else)",
        key.clone(),
      )));
    }
    // (c) An allowed quant sidecar (`.scales`/`.biases`, on a quantizable linear
    // prefix) must still travel WITH its `.weight`: an orphan sidecar (its
    // `<prefix>.weight` absent from the overlay) would define a quant identity
    // for a layer the overlay does not actually replace. Reject it as a
    // MissingKey naming the absent companion weight. (Reached only for allowed
    // sidecars — a foreign sidecar was already rejected by (b) — and run before
    // the required-key floor so the orphan is reported as the orphan it is.)
    for suffix in [".scales", ".biases"] {
      if let Some(prefix) = key.strip_suffix(suffix) {
        let weight_key = format!("{prefix}.weight");
        if !sanitized.contains_key(&weight_key) {
          return Err(Error::MissingKey(MissingKeyPayload::new(
            "Model::load: the adapter file carries a quant sidecar \
             (.scales/.biases) with no matching .weight in the overlay (an orphan sidecar — the \
             overlaid layer's .weight must accompany its sidecars)",
            weight_key,
          )));
        }
      }
    }
    // (c′) A `.biases` defines the affine half of a quantized triple, which
    // `build_linear` reads only when a `.scales` sibling marks the prefix
    // quantized. A `.biases` WITHOUT its `.scales` in the overlay (even with a
    // `.weight` present) is not a valid quant prefix — it would be silently
    // ignored as a dense weight while carrying a stale affine bias. Require the
    // `.scales` alongside any `.biases`, naming the absent `.scales`. (The prefix
    // is already an allowed quantizable-linear prefix here — a foreign or
    // LayerNorm `.biases` was rejected by (b) — so `.weight` presence was
    // confirmed by the orphan check above.)
    if let Some(prefix) = key.strip_suffix(".biases") {
      let scales_key = format!("{prefix}.scales");
      if !sanitized.contains_key(&scales_key) {
        return Err(Error::MissingKey(MissingKeyPayload::new(
          "Model::load: the adapter file carries a quant affine `.biases` with no `.scales` sibling \
           in the overlay (a `.biases` is the affine half of a quantized triple and is read only \
           alongside its `.scales`)",
          scales_key,
        )));
      }
    }
  }
  // (a) The required-key FLOOR is the ONLY check gated on `require_complete`: an
  // MMS config cannot run on a base that is only language-agnostically
  // initialized, so every REQUIRED key must be present — a missing adapter
  // tensor or `lm_head` half would otherwise leave the base init in place and
  // silently build the WRONG model. A non-MMS overlay is NOT required to be
  // complete (it has already passed the foreign/orphan checks above).
  if require_complete {
    for key in required {
      if !sanitized.contains_key(&key) {
        return Err(Error::MissingKey(MissingKeyPayload::new(
          "Model::load: the MMS per-language adapter file is missing a required tensor (it must \
           supply every adapter-layer weight + lm_head.weight/bias for this config)",
          key,
        )));
      }
    }
  }
  // Make every overlaid layer's quant-vs-dense identity come ENTIRELY from the
  // adapter file: for each `<prefix>.weight` the overlay supplies, drop BOTH
  // base quant sidecars (`<prefix>.scales` and `<prefix>.biases`) BEFORE the
  // insert below, then let the insert bring in whatever sidecars the adapter
  // file itself carries for that prefix. So a DENSE overlay (a `.weight` with no
  // `.scales`) loads the prefix dense (both stale base sidecars gone, so
  // `build_linear` cannot mistake the dense F32 weight for a packed triple); a
  // QUANTIZED overlay (`.weight` + `.scales` [+ `.biases`]) uses ONLY its own
  // sidecars (no base leakage). Critically, a quantized overlay that supplies
  // `.scales` but is MISSING `.biases` no longer inherits the base `.biases`: it
  // surfaces an incomplete-quant error downstream (`build_linear` →
  // `QuantizedLinear::from_parts` rejects an affine triple with no biases as a
  // typed `Error::InvariantViolation`) instead of silently building a hybrid
  // (adapter weight+scales plus BASE biases).
  for k in sanitized.keys() {
    if let Some(prefix) = k.strip_suffix(".weight") {
      base.remove(&format!("{prefix}.scales"));
      base.remove(&format!("{prefix}.biases"));
    }
  }
  // Replace the base init for each adapter key. The adapter file is a strict
  // subset of the model's parameters (`strict=False` in the reference), so an
  // overlay key is expected to already exist in `base` (the language-agnostic
  // init it specializes); inserting unconditionally mirrors `load_weights`. The
  // reservation is sized from the ADAPTER FILE's key count (an untrusted,
  // file-derived count), so it is made FALLIBLY — a `try_reserve` failure on a
  // hostile/oversized adapter surfaces as a typed `AllocFailure`, never an abort.
  reserve_or_error(base, "Wav2Vec2 adapter overlay into base", sanitized.len())?;
  for (k, v) in sanitized {
    base.insert(k, v);
  }
  Ok(())
}

// ───────────────────────────── CTC decode ─────────────────────────────

/// CTC blank token id (`pad_token_id = 0` for `base-960h`). Greedy decoding
/// drops this token and collapses runs.
#[cfg(feature = "wav2vec2")]
const CTC_BLANK: u32 = 0;

// The greedy decoder hardcodes `CTC_BLANK` while `validate` pins the config's
// `pad_token_id` to `PAD_TOKEN_ID`; the two encode the same blank id and must
// agree. This const assertion makes a future edit to either fail to compile
// rather than let a pinned-but-unused blank silently drift.
#[cfg(feature = "wav2vec2")]
const _: () = assert!(CTC_BLANK == PAD_TOKEN_ID as u32);

/// Greedy CTC collapse of a single per-frame argmax sequence — pure Rust over
/// `&[u32]`, the inner loop of mlx-audio's `Model._ctc_decode`
/// ([mms.py:33-45][dec]).
///
/// Emits a token only when it differs from the immediately preceding frame's
/// token **and** is not the blank token (`CTC_BLANK`): `token != prev && token != 0`.
/// `prev` is updated on **every** frame (including blanks), so a blank breaks
/// a run — `[5, 5, 0, 5]` decodes to `[5, 5]` (the blank separates the two
/// runs of `5`), while `[5, 5, 5]` collapses to `[5]`.
///
/// [dec]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L33-L45
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn ctc_greedy_collapse(predictions: &[u32]) -> Vec<u32> {
  let mut tokens = Vec::new();
  // `prev` starts at a sentinel that cannot equal any vocab id, matching the
  // reference's `prev = -1` (so the first non-blank frame always emits).
  let mut prev: Option<u32> = None;
  for &token in predictions {
    if Some(token) != prev && token != CTC_BLANK {
      tokens.push(token);
    }
    prev = Some(token);
  }
  tokens
}

// ───────────────────────────── vocabulary ─────────────────────────────

/// The character-level CTC vocabulary — the inverse of HF `vocab.json`'s
/// `{token_string: id}` map, used to render decoded token ids back to text.
///
/// mlx-audio loads `vocab.json` and inverts it to `{id: token}`
/// ([mms.py:145-155][voc]); the `base-960h` vocabulary is 32 single-character
/// tokens (`<pad>=0, <s>, </s>, <unk>, |, A-Z, '`) — **not** BPE, so no
/// tokenizer crate is needed. `_tokens_to_text` joins the per-id strings and
/// maps the word-delimiter `|` to a space ([mms.py:47-52][voc-text]).
///
/// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L145-L155
/// [voc-text]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L47-L52
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
#[derive(Debug, Clone, Default)]
pub struct Vocab {
  /// `id → token string`. A `Vec` indexed by id keeps the common path
  /// (`tokens_to_text`) allocation-free of a hash lookup; ids past the end
  /// (or with a `None` slot, if `vocab.json` were sparse) render as the empty
  /// string, matching the reference's `self._vocab.get(t, "")`.
  id_to_token: Vec<Option<String>>,
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Vocab {
  /// Parse a `vocab.json` body — the HF `{token: id}` object — and invert it
  /// to `id → token`.
  ///
  /// Mirrors mlx-audio's `model._vocab = {v: k for k, v in vocab.items()}`
  /// ([mms.py:155][voc]). The nested-`{lang: {...}}` MMS multilingual form is
  /// **not** handled (`base-960h` is monolingual); a nested object is a parse
  /// error.
  ///
  /// Malformed ids are rejected with a typed error rather than panicking or
  /// silently corrupting the table:
  /// - a **negative** id is rejected ([`Error::OutOfRange`]); a non-empty map
  ///   whose ids are *all* negative is [`Error::MalformedData`] (distinct from
  ///   the legitimately empty `{}` map, which yields an empty [`Vocab`]);
  /// - the `id → token` table is a dense `Vec` sized by the largest id, so its
  ///   one allocation is **fallible**: a within-range id whose slot count
  ///   exceeds available memory surfaces as [`Error::AllocFailure`] rather than
  ///   the abort `vec![None; len]` would raise. Bounding a legitimately large
  ///   vocabulary is the caller's concern, not the library's;
  /// - two distinct tokens claiming the **same** id is rejected
  ///   ([`Error::MalformedData`]); the source is a `HashMap`, so silently
  ///   overwriting one slot would pick an arbitrary (per-run nondeterministic)
  ///   survivor and corrupt the vocabulary.
  ///
  /// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L155
  pub fn from_json(json: &str) -> Result<Self> {
    let map: HashMap<String, i64> = serde_json::from_str(json)
      .map_err(|e| Error::Parse(ParsePayload::new("Vocab::from_json", "vocab.json", e)))?;
    Self::from_token_id_map(map)
  }

  /// Parse an MMS-style `vocab.json`, selecting the per-language `{token: id}`
  /// map for `target_lang` when the file is the nested multilingual form, then
  /// inverting it to `id → token`.
  ///
  /// Mirrors mlx-audio's `Model.post_load_hook` ([mms.py:145-155][voc]): when
  /// `vocab.json`'s values are themselves objects (the nested `{lang: {token:
  /// id}}` MMS form), it selects `vocab.get(target_lang, vocab.get("eng",
  /// vocab.get("en", first)))` — the requested language, then English (`eng` /
  /// `en`), then an arbitrary first language; otherwise it treats the file as a
  /// flat monolingual `{token: id}` map (identical to [`Vocab::from_json`]). The
  /// "arbitrary first" tie-break reads the smallest language key (deterministic,
  /// not a per-run `HashMap` survivor — the reference's `next(iter(...))` is
  /// insertion-order, which mlxrs cannot replicate, so a stable key order is
  /// substituted).
  ///
  /// This **lenient** selection (the eng/en/smallest fallback) is correct for
  /// the **non-adapter** path — a plain checkpoint with no per-language adapter,
  /// where the `vocab.json` is the model's whole token table and a flat file is
  /// language-agnostic. The **adapter** path (an MMS per-language adapter was
  /// selected) must instead require an exact entry for the adapter's language so
  /// the adapter / `lm_head` / vocab never describe different languages — it uses
  /// [`Vocab::from_json_for_lang_strict`].
  ///
  /// Reuses the same malformed-id rejections as [`Vocab::from_json`] (negative
  /// / duplicate ids, fallible dense-table allocation).
  ///
  /// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L145-L155
  pub fn from_json_for_lang(json: &str, target_lang: &str) -> Result<Self> {
    Self::from_json_for_lang_inner(json, target_lang, false)
  }

  /// Parse an MMS-style `vocab.json` for the **adapter path**, requiring an
  /// **exact** nested entry for `target_lang` (no silent fallback to another
  /// language's token table).
  ///
  /// Identical to [`Vocab::from_json_for_lang`] for a **flat** monolingual
  /// `{token: id}` vocab (a flat file is language-agnostic and is used as-is
  /// regardless of `target_lang`). But when the file is the **nested**
  /// `{lang: {token: id}}` MMS form, this requires `target_lang` to be present:
  /// if it is absent the selection does **not** fall back to `eng` / `en` / the
  /// smallest key (which the lenient variant does) — it returns a typed
  /// [`Error::MissingKey`] naming the missing language. This is the loader's
  /// guard against the divergence where a per-language adapter (e.g. `fra`) is
  /// overlaid but its logits would be decoded with another language's table
  /// because the nested vocab lacks a `fra` entry.
  ///
  /// Reuses the same malformed-id rejections as [`Vocab::from_json`].
  pub fn from_json_for_lang_strict(json: &str, target_lang: &str) -> Result<Self> {
    Self::from_json_for_lang_inner(json, target_lang, true)
  }

  /// Shared core of [`Vocab::from_json_for_lang`] (`strict = false`) and
  /// [`Vocab::from_json_for_lang_strict`] (`strict = true`).
  ///
  /// A **flat** `{token: id}` vocab is language-agnostic and used as-is in both
  /// modes. For a **nested** `{lang: {token: id}}` vocab the language map is
  /// selected as the requested `target_lang`, then — only when `strict` is
  /// `false` — English (`eng` / `en`), then the lexicographically-smallest
  /// language key (a deterministic substitute for the reference's
  /// insertion-order `next(iter(...))`, which a `HashMap` cannot replicate).
  /// When `strict` is `true` an absent `target_lang` is a typed
  /// [`Error::MissingKey`] (the adapter path forbids a fallback token table); an
  /// empty nested object likewise lacks the requested language. The empty-nested
  /// lenient case keeps yielding an empty [`Vocab`] (the prior behavior).
  fn from_json_for_lang_inner(json: &str, target_lang: &str, strict: bool) -> Result<Self> {
    // First try the flat monolingual `{token: id}` form (the `base-960h` shape).
    // A flat vocab is language-agnostic, so it is used as-is in both modes.
    if let Ok(flat) = serde_json::from_str::<HashMap<String, i64>>(json) {
      return Self::from_token_id_map(flat);
    }
    // Otherwise it is the nested multilingual `{lang: {token: id}}` form.
    let mut nested: HashMap<String, HashMap<String, i64>> =
      serde_json::from_str(json).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "Vocab::from_json_for_lang",
          "vocab.json (flat {token:id} or nested {lang:{token:id}})",
          e,
        ))
      })?;
    // The adapter (strict) path requires an exact `target_lang` entry — never a
    // fallback to another language's table (an empty nested object also lacks
    // it). Take the selected language map by OWNERSHIP (`remove`) rather than
    // cloning it — the whole token map is moved out of `nested`, so a large
    // per-language vocab is not duplicated (allocation discipline: no implicit
    // clone of a file-derived map). An absent language is a typed missing-key
    // error naming it.
    if strict {
      let lang_map = nested.remove(target_lang).ok_or_else(|| {
        Error::MissingKey(MissingKeyPayload::new(
          "Vocab::from_json_for_lang_strict: the selected MMS adapter language has no entry in the \
           nested vocab.json (the adapter / lm_head / vocab must all be the same language; no \
           fallback to another language's token table is allowed)",
          target_lang,
        ))
      })?;
      return Self::from_token_id_map(lang_map);
    }
    // Lenient (non-adapter) path: an empty nested object yields an empty Vocab,
    // and the language map is the requested `target_lang`, then English
    // (`eng` / `en`), then the lexicographically-smallest language key.
    if nested.is_empty() {
      return Ok(Self {
        id_to_token: Vec::new(),
      });
    }
    // Resolve the selected language KEY under immutable borrows (the fallback
    // chain), cloning only the small key `String` — then take the selected token
    // map out of `nested` by OWNERSHIP (`remove`), so the (potentially large)
    // per-language map is moved, never cloned.
    let selected_lang = if nested.contains_key(target_lang) {
      target_lang.to_string()
    } else if nested.contains_key("eng") {
      "eng".to_string()
    } else if nested.contains_key("en") {
      "en".to_string()
    } else {
      // Lexicographically-smallest language key — a deterministic tie-break
      // (not an arbitrary `HashMap` survivor).
      nested
        .keys()
        .min()
        .ok_or_else(|| {
          Error::MalformedData(MalformedDataPayload::new(
            "Vocab::from_json_for_lang",
            "nested vocab.json has no selectable language map",
          ))
        })?
        .clone()
    };
    // `selected_lang` was just resolved against `nested`, so it is present.
    let lang_map = nested.remove(&selected_lang).ok_or_else(|| {
      Error::MalformedData(MalformedDataPayload::new(
        "Vocab::from_json_for_lang",
        "nested vocab.json has no selectable language map",
      ))
    })?;
    Self::from_token_id_map(lang_map)
  }

  /// Invert a `{token: id}` map to the dense `id → token` table — the shared
  /// core of [`Vocab::from_json`] and [`Vocab::from_json_for_lang`].
  ///
  /// Mirrors mlx-audio's `model._vocab = {v: k for k, v in vocab.items()}`
  /// ([mms.py:155][voc]) with the same typed rejections:
  /// - a **negative** id is rejected ([`Error::OutOfRange`]); a non-empty map
  ///   whose ids are *all* negative is [`Error::MalformedData`] (distinct from
  ///   the legitimately empty `{}` map, which yields an empty [`Vocab`]);
  /// - the dense table is sized by the largest id and allocated **fallibly**
  ///   (a within-range id whose slot count exceeds memory is
  ///   [`Error::AllocFailure`], not an abort);
  /// - two distinct tokens claiming the **same** id is rejected
  ///   ([`Error::MalformedData`]).
  ///
  /// [voc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L155
  fn from_token_id_map(map: HashMap<String, i64>) -> Result<Self> {
    if map.is_empty() {
      // Legitimately empty vocabulary → no slots (forward still works,
      // transcribe errors with a clear message).
      return Ok(Self {
        id_to_token: Vec::new(),
      });
    }
    let max_id = map.values().copied().max().unwrap_or(-1);
    if max_id < 0 {
      // A non-empty map with no non-negative id is malformed (every entry's
      // id is negative) — reject rather than silently inverting to an empty
      // table, which would lose the whole vocabulary.
      return Err(Error::MalformedData(MalformedDataPayload::new(
        "Vocab::from_json",
        "vocab.json has entries but every token id is negative",
      )));
    }
    // `max_id >= 0` here. Slot count is `max_id + 1`. The dense table is the
    // only allocation sized from untrusted input, so it is made fallible rather
    // than capped: an id too large to index a `Vec` on this target, or a slot
    // count that exhausts memory, surfaces as a typed `AllocFailure` instead of
    // the abort `vec![None; len]` would raise. (Bounding a legitimately large
    // vocabulary is the caller's concern.)
    let len = usize::try_from(max_id)
      .ok()
      .and_then(|m| m.checked_add(1))
      .ok_or_else(|| {
        Error::OutOfRange(OutOfRangePayload::new(
          "Vocab::from_json: token id",
          "must be representable as a Vec index on this target",
          format_smolstr!("{max_id}"),
        ))
      })?;
    let mut id_to_token: Vec<Option<String>> = Vec::new();
    reserve_or_error(&mut id_to_token, "id_to_token slots", len)?;
    id_to_token.resize(len, None);
    for (token, id) in map {
      if id < 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Vocab::from_json: token id",
          "must be non-negative",
          format_smolstr!("{id}"),
        )));
      }
      // `0 <= id <= max_id`, so the index is in bounds.
      let slot = &mut id_to_token[id as usize];
      if slot.is_some() {
        // Two distinct tokens claim the same id. The source is a `HashMap`, so
        // a bare `slot = Some(token)` would let an arbitrary survivor (per-run
        // nondeterministic) silently overwrite the other — corrupting the
        // vocabulary. Reject it instead.
        return Err(Error::MalformedData(MalformedDataPayload::new(
          "Vocab::from_json",
          "vocab.json maps two distinct tokens to the same id",
        )));
      }
      *slot = Some(token);
    }
    Ok(Self { id_to_token })
  }

  /// Number of id slots (one past the maximum id).
  #[inline(always)]
  pub fn len(&self) -> usize {
    self.id_to_token.len()
  }

  /// Whether the vocabulary is empty.
  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.id_to_token.is_empty()
  }

  /// The token string for `id`, or `None` if out of range / absent.
  #[inline]
  pub fn token(&self, id: u32) -> Option<&str> {
    self
      .id_to_token
      .get(id as usize)
      .and_then(|slot| slot.as_deref())
  }

  /// Render a decoded token-id sequence to text — the reference's
  /// `"".join(self._vocab.get(t, "")).replace("|", " ")`
  /// ([mms.py:52][voc-text]). Unknown ids contribute nothing; the
  /// word-delimiter `|` becomes a space.
  ///
  /// [voc-text]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L52
  pub fn tokens_to_text(&self, tokens: &[u32]) -> String {
    let mut text = String::new();
    for &t in tokens {
      if let Some(tok) = self.token(t) {
        text.push_str(tok);
      }
    }
    text.replace('|', " ")
  }
}

// ───────────────────────────── linear helper ─────────────────────────────

/// A wav2vec2 linear projection `y = x @ Wᵀ (+ b)` — quantize-aware.
///
/// Mirrors `mlx.nn.Linear` (`weight` stored `(out, in)`, the forward
/// transposes it) for a dense checkpoint, and `mlx.nn.QuantizedLinear` for an
/// mlx-community quantized (e.g. 8-bit) wav2vec2 checkpoint. The two cases
/// share one [`forward`](Self::forward) call site via the shared
/// [`MaybeQuantizedLinear`], so the attention / feed-forward / feature
/// projection / CTC-head code is unchanged whether the weights are dense or
/// quantized.
///
/// This is the wav2vec2 adoption of the shared quantize-aware layer, mirroring
/// the Whisper model's `Linear` wrapper exactly. Only the Linear / projection
/// layers (the encoder attention `q/k/v/out` projections, the feed-forward
/// `intermediate` / `output` projections, the feature projection, and the CTC
/// `lm_head`) use it — the convolutional feature extractor / positional conv
/// stay dense, matching what mlx-audio / MLX quantizes (`nn.Linear` only).
#[cfg(feature = "wav2vec2")]
struct Linear {
  /// The dense-or-quantized projection. Built by the model builders (either
  /// directly from a shape-validated dense `(weight, bias)` via [`Self::new`],
  /// or from the checkpoint's quantized `(weight, scales, biases)` triple via
  /// [`Self::quantized`]).
  inner: MaybeQuantizedLinear,
}

#[cfg(feature = "wav2vec2")]
impl Linear {
  /// Construct a **dense** projection from a `(out, in)` `weight` and an
  /// optional `(out,)` `bias`.
  fn new(weight: Array, bias: Option<Array>) -> Self {
    Self {
      inner: MaybeQuantizedLinear::Dense(crate::nn::Linear::new(weight, bias)),
    }
  }

  /// Wrap an already-built [`MaybeQuantizedLinear`] (the quantized path: the
  /// model builders construct it from the checkpoint's quantized
  /// `(weight, scales, biases)` triple).
  fn quantized(inner: MaybeQuantizedLinear) -> Self {
    Self { inner }
  }

  /// `y = x @ weightᵀ (+ bias)` (dense) or `quantized_matmul(...) (+ bias)`
  /// (quantized). `x` is `(..., in)`; the result is `(..., out)`.
  ///
  /// # Errors
  /// Propagates the transpose / matmul / quantized-matmul / add op errors.
  fn forward(&self, x: &Array) -> Result<Array> {
    self.inner.forward(x)
  }

  /// `true` if this projection was loaded from a quantized checkpoint.
  #[cfg(test)]
  fn is_quantized(&self) -> bool {
    self.inner.is_quantized()
  }
}

// ───────────────────────── weight-fetch helper ─────────────────────────

/// Pull a weight by exact key from the (sanitized) checkpoint map, erroring
/// with the key if absent.
#[cfg(feature = "wav2vec2")]
fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights
    .remove(key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("Model::from_weights", key)))
}

/// Assert a checkpoint tensor's shape (rank + every dimension) equals the
/// `expected` shape the (validated) config implies, before it is stored or fed
/// to any op.
///
/// `from_weights` reads each layer's shape from the checkpoint, not from the
/// config, so a corrupt / hostile tensor that survives the config gate could
/// otherwise run a *different* graph (a conv weight with a different kernel
/// axis → a different receptive field) or drive an oversized allocation (an
/// `lm_head.weight` with a huge output dim → huge logits). Pinning every
/// consumed tensor to the shape the config implies closes that whole
/// dimension: a mismatch fails fast with a typed error here, before any
/// forward op.
///
/// The expected dims are computed from the (already-`validate`d)
/// [`Config`]'s width / count / conv-stack fields and are non-negative
/// by construction. The rank is checked by the length comparison, so this
/// single helper covers both the rank and the exact-shape requirements. On
/// mismatch returns an [`Error::ShapePairMismatch`] carrying both full shapes,
/// wrapped in an [`Error::LayerKeyed`] naming the offending tensor `key` (the
/// dynamic per-layer key the `&'static` `descriptor` cannot carry).
#[cfg(feature = "wav2vec2")]
fn expect_shape(
  tensor: &Array,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<()> {
  let actual = tensor.shape();
  // Compare in i64 so the usize dims and the i32 expectations both widen
  // losslessly (real MLX dims are i32-bounded); the length check also pins the
  // rank. A negative expected dim is a builder bug and never matches.
  let matches = actual.len() == expected.len()
    && actual
      .iter()
      .zip(expected.iter())
      .all(|(&a, &e)| e >= 0 && a as i64 == i64::from(e));
  if !matches {
    let expected_usize: Vec<usize> = expected.iter().map(|&e| e.max(0) as usize).collect();
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        expected_usize,
        actual,
      )),
    )));
  }
  Ok(())
}

/// [`take`] a weight by key, then assert its shape equals `expected` via
/// [`expect_shape`] — the fused fetch-and-shape-check the builders use for
/// every tensor stored verbatim, so a consumed tensor can never skip the gate.
#[cfg(feature = "wav2vec2")]
fn take_shaped(
  weights: &mut HashMap<String, Array>,
  key: &str,
  descriptor: &'static str,
  expected: &[i32],
) -> Result<Array> {
  let tensor = take(weights, key)?;
  expect_shape(&tensor, key, descriptor, expected)?;
  Ok(tensor)
}

// ───────────────────────── quantize-aware projection ─────────────────────────

/// Validate a quantized layer's packed `<prefix>.weight` + `<prefix>.scales`
/// against the config-derived `(out, in_features)` BEFORE the quantized layer
/// is constructed — the quantized analogue of [`expect_shape`].
///
/// The dense path pins every consumed tensor to its exact config shape via
/// [`take_shaped`]; the quantized path must reach the same load-time gate,
/// because a corrupt quantized checkpoint could otherwise ship a packed weight
/// whose *logical* output or input dimension disagrees with the config, and the
/// first forward would then size projections / logits from the checkpoint
/// tensors instead of the validated config. The packed `uint32` weight has a
/// different shape than the dense `(out, in)`, so the recovery mirrors mlx's
/// quantized layout (`mlx/ops.cpp:107,131,4790-4792`):
///
/// - the weight is rank-2 `uint32` `(out, in * bits / 32)`; its *logical*
///   output dim is the leading axis and must equal `out`, and its *logical*
///   input width — mlx's `w_inner_dims = w.shape(-1) * 32 / bits`, the dimension
///   `quantized_matmul` contracts against — must equal `in_features`;
/// - the `scales` are rank-2 `(out, in / group_size)`; the leading axis must
///   equal `out`, and `scales.shape(-1) * group_size` must equal `in_features`
///   (mlx's `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
///   invariant).
///
/// `group_size` / `bits` are the per-layer-resolved scheme parameters; both are
/// checked `> 0` before they divide (a non-positive value is a malformed config
/// and an [`Error::OutOfRange`], never a panic). The per-mode value tables
/// remain mlx-c's; this only pins the structural relationship to the config the
/// dense gate also enforces. Reads only `shape()` / `dtype()` metadata (no
/// materialization), so it is bounded regardless of the declared dims.
///
/// On mismatch returns an [`Error::ShapePairMismatch`] (or [`Error::RankMismatch`]
/// / [`Error::InvariantViolation`] for a wrong rank / dtype) wrapped in an
/// [`Error::LayerKeyed`] naming the offending `<prefix>.weight` / `<prefix>.scales`
/// key.
#[cfg(feature = "wav2vec2")]
fn check_quantized_shape(
  weights: &HashMap<String, Array>,
  prefix: &str,
  descriptor: &'static str,
  out: i32,
  in_features: i32,
  group_size: i32,
  bits: i32,
) -> Result<()> {
  // The scheme parameters divide the recovered widths; a non-positive value is
  // a malformed config (`from_parts` also rejects it, but guard here so the
  // divisions below cannot trap).
  if bits <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Model: quantized layer bits",
      "must be > 0",
      format_smolstr!("{bits}"),
    )));
  }
  if group_size <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Model: quantized layer group_size",
      "must be > 0",
      format_smolstr!("{group_size}"),
    )));
  }

  // Packed weight `(out, in * bits / 32)`, `uint32`.
  let weight_key = format!("{prefix}.weight");
  let weight = weights.get(&weight_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Model: quantized weight not found in checkpoint",
      weight_key.clone(),
    ))
  })?;
  let w_shape = weight.shape();
  if w_shape.len() != 2 {
    let rank = w_shape.len() as u32;
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::RankMismatch(RankMismatchPayload::new(
        "quantized weight must be rank-2 (out, in * bits / 32)",
        rank,
        w_shape,
      )),
    )));
  }
  if weight.dtype()? != crate::dtype::Dtype::U32 {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::InvariantViolation(InvariantViolationPayload::new(
        "quantized weight dtype",
        "must be `uint32` (the packed-quantized-weight dtype)",
      )),
    )));
  }
  // Logical output dim is the leading axis; logical input width is mlx's
  // `w_inner_dims = w.shape(-1) * 32 / bits` (the contraction dim). Compare in
  // i64 so the recovery cannot overflow on a corrupt huge packed width.
  let logical_in = (w_shape[1] as i64) * 32 / i64::from(bits);
  if w_shape[0] as i64 != i64::from(out) || logical_in != i64::from(in_features) {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &weight_key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        descriptor,
        vec![out.max(0) as usize, in_features.max(0) as usize],
        vec![w_shape[0], logical_in.max(0) as usize],
      )),
    )));
  }

  // Scales `(out, in / group_size)`: leading axis is `out`, and the per-group
  // count recovers the same logical input width as the packed weight.
  let scales_key = format!("{prefix}.scales");
  let scales = weights.get(&scales_key).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Model: quantized scales not found in checkpoint",
      scales_key.clone(),
    ))
  })?;
  let s_shape = scales.shape();
  if s_shape.len() != 2 {
    let rank = s_shape.len() as u32;
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &scales_key,
      Error::RankMismatch(RankMismatchPayload::new(
        "quantized scales must be rank-2 (out, in / group_size)",
        rank,
        s_shape,
      )),
    )));
  }
  let scales_in = (s_shape[1] as i64) * i64::from(group_size);
  if s_shape[0] as i64 != i64::from(out) || scales_in != i64::from(in_features) {
    return Err(Error::LayerKeyed(LayerKeyedPayload::new(
      &scales_key,
      Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "quantized scales (out, in / group_size) must match the config",
        vec![out.max(0) as usize, in_features.max(0) as usize],
        vec![s_shape[0], scales_in.max(0) as usize],
      )),
    )));
  }

  Ok(())
}

/// Build a quantize-aware [`Linear`] from `<prefix>.weight` (+ the dense
/// `<prefix>.bias` when the architecture `bias` flag is `true`) — the wav2vec2
/// analogue of the Whisper `Builder::linear`.
///
/// **Quantized path** — when `quant` is `Some` AND the checkpoint carries a
/// `<prefix>.scales` sibling: the projection is built quantized via
/// [`crate::nn::QuantizedLinear::from_parts`], running
/// [`crate::ops::quantized::quantized_matmul`] over the packed
/// `(weight, scales, biases)` triple with the per-layer-resolved
/// `(group_size, bits, mode)`. The packed `uint32` weight has shape
/// `(out, in * bits / 32)`, NOT the dense `(out, in)`, so it does **not** go
/// through the dense [`take_shaped`] config-shape check — its structural
/// consistency is validated by [`check_quantized_shape`] (against the config)
/// and [`crate::nn::QuantizedLinear::from_parts`] (the triple). This mirrors
/// mlx-audio's whisper `class_predicate` (`f"{p}.scales" in weights`), with the
/// global `quantization` block's scheme parameters.
///
/// The dense output `<prefix>.bias` is loaded with the **same arity** the dense
/// path enforces: when `bias` is `true` this path **requires** `<prefix>.bias`,
/// validates it as `(out,)`, and passes it as the explicit dense bias to
/// [`crate::nn::QuantizedLinear::from_parts`] (NOT via the optional-bias
/// `from_weights` convenience), so a quantized checkpoint missing a required
/// bias fails fast with the same typed error the dense path returns.
///
/// **Dense path** — otherwise: pops `<prefix>.weight` `(out, in)` (+ the
/// `<prefix>.bias` `(out,)` when `bias` is `true`), both shape-validated against
/// the config-derived extents via [`take_shaped`] BEFORE materialization,
/// exactly as before — so a dense (no `.scales`) checkpoint loads identically
/// whether or not a `quantization` config is threaded.
///
/// # Errors
/// - [`Error::InvariantViolation`] if `<prefix>.scales` is present but `quant`
///   resolved no scheme parameters for this layer;
/// - [`Error::MissingKey`] / [`Error::ShapePairMismatch`] / [`Error::RankMismatch`]
///   for an absent / mis-shaped tensor;
/// - propagates [`crate::nn::QuantizedLinear::from_parts`]'s structural
///   validation of the quantized triple.
#[cfg(feature = "wav2vec2")]
#[allow(clippy::too_many_arguments)]
fn build_linear(
  weights: &mut HashMap<String, Array>,
  quant: Option<&PerLayerQuantization>,
  prefix: &str,
  weight_descriptor: &'static str,
  bias_descriptor: &'static str,
  out: i32,
  in_features: i32,
  bias: bool,
) -> Result<Linear> {
  // `<prefix>.scales` is the load-bearing "this layer is quantized" signal
  // (mlx-audio whisper `class_predicate`). Only when it is present AND a
  // quantization config is threaded do we take the quantized path.
  let scales_key = format!("{prefix}.scales");
  if quant.is_some() && weights.contains_key(&scales_key) {
    // Resolve the per-layer `(group_size, bits, mode)` from the config. A
    // `quantization_for` of `None` (an explicit per-layer `Skip`, or no global
    // default) next to a present `.scales` is a config/checkpoint inconsistency
    // — a typed error, never a guessed scheme.
    let Some(q) = quant.and_then(|q| q.quantization_for(prefix)) else {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Model: Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
        "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
      )));
    };
    // The quantized triple must reach the same config-shape gate the dense
    // `take_shaped` enforces: the packed weight's logical `(out, in)` (and the
    // scales' recovery) must equal the config-derived extents BEFORE
    // construction.
    check_quantized_shape(
      weights,
      prefix,
      weight_descriptor,
      out,
      in_features,
      q.group_size,
      q.bits,
    )?;
    // Load the dense output bias with the SAME arity as the dense branch: when
    // `bias` is `true`, `take_shaped` REQUIRES `<prefix>.bias` and validates it
    // `(out,)`; when `false`, any stray `<prefix>.bias` is dropped unused. The
    // loaded bias is passed as the explicit dense bias to `from_parts`.
    let dense_bias = if bias {
      Some(take_shaped(
        weights,
        &format!("{prefix}.bias"),
        bias_descriptor,
        &[out],
      )?)
    } else {
      weights.remove(&format!("{prefix}.bias"));
      None
    };
    // Pop the packed triple by key: the `uint32` weight, the `.scales`, and the
    // per-group affine `.biases` (present iff `mode == "affine"`; `from_parts`
    // enforces the mode/arity contract).
    let weight = take(weights, &format!("{prefix}.weight"))?;
    let scales = take(weights, &format!("{prefix}.scales"))?;
    let quant_biases = weights.remove(&format!("{prefix}.biases"));
    let q = QuantizedLinear::from_parts(
      weight,
      scales,
      quant_biases,
      dense_bias,
      q.group_size,
      q.bits,
      q.mode.as_str(),
    )?;
    return Ok(Linear::quantized(MaybeQuantizedLinear::Quantized(q)));
  }

  // Dense path (unchanged): shape-validate against the config-derived
  // `(out, in)` before materialization.
  let weight = take_shaped(
    weights,
    &format!("{prefix}.weight"),
    weight_descriptor,
    &[out, in_features],
  )?;
  let b = if bias {
    Some(take_shaped(
      weights,
      &format!("{prefix}.bias"),
      bias_descriptor,
      &[out],
    )?)
  } else {
    None
  };
  Ok(Linear::new(weight, b))
}

// ───────────────────────── feature encoder ─────────────────────────

/// A single feature-encoder convolution layer, covering all three reference
/// layer kinds. Conv weight is channels-last `(out, k, in)` (post-sanitize);
/// the optional `bias` `(out,)` is present iff `config.conv_bias`. At most one
/// of the two normalizations is present, never both:
///
/// - the L0 `Wav2Vec2GroupNormConvLayer` (`feat_extract_norm == "group"`)
///   carries an affine [`GroupNorm`] (`num_groups == dims`, pytorch-compatible)
///   in [`Self::group_norm`];
/// - every layer of the `Wav2Vec2LayerNormConvLayer` extractor
///   (`feat_extract_norm == "layer"`) carries an affine [`LayerNorm`] over the
///   conv output width in [`Self::layer_norm`];
/// - the `Wav2Vec2NoLayerNormConvLayer`s (the non-L0 layers of the `"group"`
///   extractor) carry neither — conv → activation only.
#[cfg(feature = "wav2vec2")]
struct ConvLayer {
  weight: Array,
  /// `Some(bias)` iff `config.conv_bias` — `nn.Conv1d(bias=config.conv_bias)`.
  bias: Option<Array>,
  stride: i32,
  /// `Some` for the L0 `Wav2Vec2GroupNormConvLayer` (`"group"` extractor),
  /// `None` otherwise. Mutually exclusive with [`Self::layer_norm`].
  group_norm: Option<GroupNorm>,
  /// `Some` for every `Wav2Vec2LayerNormConvLayer` (`"layer"` extractor),
  /// `None` otherwise. Mutually exclusive with [`Self::group_norm`].
  layer_norm: Option<LayerNorm>,
  /// The `feat_extract_activation` (resolved once at build).
  activation: Activation,
}

#[cfg(feature = "wav2vec2")]
impl ConvLayer {
  /// `conv(x.swapaxes(-2,-1)) (+ bias) → [group_norm | layer_norm] →
  /// swapaxes(-2,-1) → act`.
  ///
  /// Input/output are `(B, C, L)` (channels-second). MLX conv1d is
  /// channels-last, so the layer transposes to `(B, L, C)` around the conv
  /// (and around the GroupNorm / LayerNorm, which normalize over the
  /// last/feature axis), exactly as the reference's
  /// `hidden_states.swapaxes(-2, -1)` bracketing. The bias is added in the
  /// channels-last `(B, L', C_out)` layout (the `(out,)` bias broadcasts over
  /// the last axis), before the norm — matching `nn.Conv1d`'s fused bias
  /// followed by the layer norm.
  ///
  /// The LayerNorm placement mirrors `Wav2Vec2LayerNormConvLayer.__call__`
  /// ([wav2vec.py:116-122][ln]): conv → LayerNorm (channels-last) → swap back →
  /// activation. The GroupNorm placement mirrors
  /// `Wav2Vec2GroupNormConvLayer.__call__` ([wav2vec.py:148-155][gn]). At most
  /// one of the two is present.
  ///
  /// [ln]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L116-L122
  /// [gn]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L148-L155
  fn forward(&self, x: &Array) -> Result<Array> {
    // (B, C_in, L) → (B, L, C_in)
    let xt = ops::shape::swapaxes(x, -2, -1)?;
    // channels-last conv → (B, L', C_out)
    let mut h = ops::conv::conv1d(&xt, &self.weight, self.stride, 0, 1, 1)?;
    // `nn.Conv1d(bias=True)` folds the bias into the conv; here it is an
    // explicit add over the channels-last feature (last) axis.
    if let Some(bias) = &self.bias {
      h = h.add(bias)?;
    }
    // GroupNorm (L0 of the `"group"` extractor) or LayerNorm (every layer of
    // the `"layer"` extractor) runs in channels-last layout (feature = last
    // axis); at most one is present.
    if let Some(gn) = &self.group_norm {
      h = gn.forward(&h)?;
    } else if let Some(ln) = &self.layer_norm {
      h = ln.forward(&h)?;
    }
    // back to (B, C_out, L')
    let h = ops::shape::swapaxes(&h, -2, -1)?;
    self.activation.forward(&h)
  }
}

/// The feature encoder: `input[:, None]` → `num_feat_extract_layers` conv
/// layers → `(B, conv_dim[-1], T')`. Ports the `feat_extract_norm == "group"`
/// arm of `Wav2Vec2FeatureEncoder` ([wav2vec.py:250-276][fe]) — the L0
/// `Wav2Vec2GroupNormConvLayer` followed by `Wav2Vec2NoLayerNormConvLayer`s.
///
/// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L250-L276
#[cfg(feature = "wav2vec2")]
struct FeatureEncoder {
  conv_layers: Vec<ConvLayer>,
}

#[cfg(feature = "wav2vec2")]
impl FeatureEncoder {
  fn forward(&self, input_values: &Array) -> Result<Array> {
    // input_values (B, T) → (B, 1, T): insert a channel axis at position 1.
    let mut h = ops::shape::expand_dims_axes(input_values, &[1])?;
    for layer in &self.conv_layers {
      h = layer.forward(&h)?;
    }
    Ok(h)
  }
}

// ───────────────────────── feature projection ─────────────────────────

/// `[LayerNorm(512) →] Linear(512 → 768)` over `(B, T', 512)`.
/// Ports `Wav2Vec2FeatureProjection` ([wav2vec.py:279-290][fp]).
///
/// The LayerNorm is **conditional**, but HuBERT-only: HF's
/// `Wav2Vec2FeatureProjection` always applies it, while `HubertFeatureProjection`
/// gates it on `config.feat_proj_layer_norm` (default `true`). So
/// [`Self::layer_norm`] is `None` **only** for a HuBERT checkpoint that sets
/// `feat_proj_layer_norm = false` (its no-LayerNorm arm, feeding the linear the
/// un-normalized feature-encoder output directly); for a wav2vec2 `model_type`
/// the LayerNorm is always present (and [`Config::validate`] rejects
/// `feat_proj_layer_norm = false` there).
///
/// [fp]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L279-L290
#[cfg(feature = "wav2vec2")]
struct FeatureProjection {
  /// `LayerNorm(conv_dim[-1])` — `Some` whenever the projection LayerNorm is
  /// applied (always for a wav2vec2 `model_type`; for HuBERT iff
  /// `feat_proj_layer_norm`). `None` only for a HuBERT checkpoint that sets
  /// `feat_proj_layer_norm = false` (its no-LayerNorm arm).
  layer_norm: Option<LayerNorm>,
  /// `Linear(conv_dim[-1] → hidden_size)` — quantize-aware.
  projection: Linear,
}

#[cfg(feature = "wav2vec2")]
impl FeatureProjection {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    // The LayerNorm is applied only when present (HuBERT's
    // `feat_proj_layer_norm = false` arm skips it, feeding the linear the
    // un-normalized feature-encoder output directly).
    match &self.layer_norm {
      Some(ln) => {
        let normed = ln.forward(hidden_states)?;
        self.projection.forward(&normed)
      }
      None => self.projection.forward(hidden_states),
    }
  }
}

// ───────────────────────── positional conv embedding ─────────────────────────

/// The weight-normalized positional conv embedding
/// (`Wav2Vec2PositionalConvEmbedding`, [wav2vec.py:218-247][pos]).
///
/// HF stores the weight-normalized conv as a magnitude `weight_g` and a
/// direction `weight_v`; the effective kernel is
/// `weight = weight_g * weight_v / ‖weight_v‖`, the norm reduced over every
/// axis except the kernel axis (axis 1 in MLX's channels-last
/// `(out, k, in)` layout, because the reference's `swapaxes(1, 2)` made
/// `except_dim=1` the kernel axis — see [`reconstruct_wn_weight`]). The fused
/// kernel is **reconstructed once at load**, so the forward pass is a plain
/// grouped conv. SamePad then crops the single trailing frame produced by the
/// even (128) kernel, and the `feat_extract_activation` follows.
///
/// HF's `Wav2Vec2PositionalConvEmbedding` applies
/// `ACT2FN[config.feat_extract_activation]` after the conv + SamePad — the same
/// activation the feature-encoder convs use — so the activation is resolved
/// from `feat_extract_activation` (not hardcoded GELU): a checkpoint declaring,
/// say, `silu` / `gelu_new` runs a consistent graph rather than a hybrid one.
///
/// [pos]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L218-L247
#[cfg(feature = "wav2vec2")]
struct PositionalConvEmbedding {
  /// The reconstructed fused conv kernel `(out, k, in/groups)`.
  weight: Array,
  bias: Array,
  groups: i32,
  padding: i32,
  /// Frames to crop from the end (SamePad): `1` for an even kernel, else `0`.
  num_pad_remove: i32,
  /// The `feat_extract_activation` (resolved once at build) — HF applies
  /// `ACT2FN[config.feat_extract_activation]` here, matching the feature
  /// encoder. Not hardcoded GELU.
  activation: Activation,
}

#[cfg(feature = "wav2vec2")]
impl PositionalConvEmbedding {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    // hidden_states is (B, T, C) (channels-last already); plain grouped conv.
    let mut h = ops::conv::conv1d(hidden_states, &self.weight, 1, self.padding, 1, self.groups)?;
    h = h.add(&self.bias)?;
    // SamePad: drop the trailing `num_pad_remove` frames along the time axis
    // (axis 1 in channels-last (B, T, C)). The reference slices
    // `hidden_states[:, :-num_pad_remove, :]`.
    if self.num_pad_remove > 0 {
      let shape = h.shape();
      let rank = shape.len();
      let mut start = vec![0i32; rank];
      let mut stop: Vec<i32> = shape
        .iter()
        .map(|&d| {
          i32::try_from(d).map_err(|_| {
            Error::OutOfRange(OutOfRangePayload::new(
              "PositionalConvEmbedding: dim",
              "exceeds i32::MAX",
              format_smolstr!("{d}"),
            ))
          })
        })
        .collect::<Result<Vec<_>>>()?;
      let strides = vec![1i32; rank];
      // Crop the time axis (axis 1).
      start[1] = 0;
      stop[1] -= self.num_pad_remove;
      h = ops::indexing::slice(&h, &start, &stop, &strides)?;
    }
    // HF applies `ACT2FN[config.feat_extract_activation]` here (the same
    // activation the feature-encoder convs use), not an unconditional GELU.
    self.activation.forward(&h)
  }
}

/// Reconstruct a weight-normalized conv kernel from its `(weight_g, weight_v)`
/// reparametrization: `weight = weight_g * weight_v / ‖weight_v‖`, the L2 norm
/// of `weight_v` taken over every axis **except** the kernel axis (axis 1),
/// with `keepdims` so it broadcasts.
///
/// This matches mlx-audio's `WNConv1d.__call__`
/// (`weight_g * weight_v / normalize_weight(weight_v, except_dim=1)`,
/// [wav2vec.py:206-211][wn]) where `normalize_weight(x, except_dim=1)` reduces
/// `sqrt(sum(x**2))` over `axes = (0, 2)` (every axis but `except_dim`). In
/// MLX's channels-last `(out, k, in)` layout that `except_dim=1` is the kernel
/// axis (the reference applied `swapaxes(1, 2)` to the stored `(out, in, k)`
/// tensor at load, turning the original PyTorch `dim=0` output-channel
/// reduction into this `except_dim=1` form). Reconstructed **once** at load —
/// numerically identical to recomputing it per forward.
///
/// [wn]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L206-L211
#[cfg(feature = "wav2vec2")]
fn reconstruct_wn_weight(weight_g: &Array, weight_v: &Array) -> Result<Array> {
  if weight_v.ndim() != 3 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "reconstruct_wn_weight: weight_v must be rank-3 (out, k, in)",
      weight_v.ndim() as u32,
      weight_v.shape(),
    )));
  }
  // ‖weight_v‖ over axes (0, 2), keepdims → shape (1, k, 1).
  let sq = weight_v.square()?;
  let sum_sq = ops::reduction::sum_axes(&sq, &[0, 2], true)?;
  let norm_v = sum_sq.sqrt()?;
  // weight_g * weight_v / norm_v.
  let scaled = weight_g.multiply(weight_v)?;
  scaled.divide(&norm_v)
}

// ───────────────────────── attention ─────────────────────────

/// Multi-head self-attention (`Wav2Vec2Attention`, [wav2vec.py:293-393][att]).
///
/// `q/k/v/out` are biased `Linear(hidden, hidden)`. The query is **pre-scaled**
/// by `head_dim**-0.5` and SDPA is then called with `scale = 1.0` (matching the
/// reference's `query_states = self.q_proj(h) * self.scaling` followed by
/// `scaled_dot_product_attention(..., scale=1.0)`). The CTC inherent path runs
/// the encoder over a single (un-padded) utterance, so no attention mask is
/// applied.
///
/// [att]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L293-L393
#[cfg(feature = "wav2vec2")]
struct Attention {
  /// `Linear(hidden → hidden)` query projection — quantize-aware.
  q_proj: Linear,
  /// `Linear(hidden → hidden)` key projection — quantize-aware.
  k_proj: Linear,
  /// `Linear(hidden → hidden)` value projection — quantize-aware.
  v_proj: Linear,
  /// `Linear(hidden → hidden)` output projection — quantize-aware.
  out_proj: Linear,
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, pre-multiplied into the query.
  scaling: f32,
}

#[cfg(feature = "wav2vec2")]
impl Attention {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let shape = hidden_states.shape();
    // (B, T, C)
    let bsz = dim_i32(&shape, 0, "Attention: batch")?;
    let tgt_len = dim_i32(&shape, 1, "Attention: seq")?;

    // q pre-scaled by head_dim**-0.5; SDPA scale is then 1.0. The scale
    // constant is cast to q's dtype (mirroring the activation helpers'
    // dtype-matched scalars) so an F16/BF16 checkpoint keeps F16/BF16 through
    // attention — a bare F32 scalar would promote half-precision q to F32 under
    // MLX type promotion, breaking mixed-precision numerics and inflating
    // memory / the KV-cache.
    let q = self.q_proj.forward(hidden_states)?;
    let scale = scalar_like(self.scaling, &q)?;
    let q = q.multiply(&scale)?;
    let k = self.k_proj.forward(hidden_states)?;
    let v = self.v_proj.forward(hidden_states)?;

    // (B, T, C) → (B, n_heads, T, head_dim): reshape then transpose(0,2,1,3).
    let q = self.split_heads(&q, bsz, tgt_len)?;
    let k = self.split_heads(&k, bsz, tgt_len)?;
    let v = self.split_heads(&v, bsz, tgt_len)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, 1.0, Mask::None)?;

    // (B, n_heads, T, head_dim) → (B, T, C).
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = checked_mul(
      "Attention: num_heads * head_dim",
      "num_heads",
      self.num_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[bsz, tgt_len, embed_dim])?;
    self.out_proj.forward(&attn)
  }

  /// `(B, T, C) → (B, n_heads, T, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

// ───────────────────────── feed-forward ─────────────────────────

/// `Linear(hidden → intermediate) → act → Linear(intermediate → hidden)`
/// (`Wav2Vec2FeedForward`, [wav2vec.py:396-417][ff]), where `act` is the
/// resolved `hidden_act`.
///
/// [ff]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L396-L417
#[cfg(feature = "wav2vec2")]
struct FeedForward {
  /// `Linear(hidden → intermediate)` — quantize-aware.
  intermediate: Linear,
  /// `Linear(intermediate → hidden)` — quantize-aware.
  output: Linear,
  /// The `hidden_act` (resolved once at build).
  activation: Activation,
}

#[cfg(feature = "wav2vec2")]
impl FeedForward {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let h = self.intermediate.forward(hidden_states)?;
    let h = self.activation.forward(&h)?;
    self.output.forward(&h)
  }
}

// ───────────────────────── attention adapter (MMS) ─────────────────────────

/// The per-attention-block adapter `Wav2Vec2AttnAdapterLayer`
/// ([wav2vec.py:420-433][ad]) — the bottleneck MMS stacks on every encoder
/// layer to specialize the language-agnostic backbone to one language.
///
/// `LayerNorm(hidden) → Linear(hidden → adapter_attn_dim) → ReLU →
/// Linear(adapter_attn_dim → hidden)`, all dense in the reference (the adapter
/// weights ship per-language in `adapter.{lang}.safetensors`). The output is
/// **added** to the hidden states by the enclosing layer
/// ([wav2vec.py:503-504][ad-add]); this type computes only the adapter branch.
///
/// Present only when `config.adapter_attn_dim` is `Some` — i.e. an MMS
/// checkpoint (`facebook/mms-1b-all` / `mms-1b-fl102`). The two Linears are
/// quantize-aware ([`Linear`]) so a quantized MMS checkpoint loads its adapter
/// through the same `<prefix>.scales` dispatch the rest of the model uses,
/// though the reference's adapter is dense.
///
/// [ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L420-L433
/// [ad-add]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L503-L504
#[cfg(feature = "wav2vec2")]
struct AttnAdapterLayer {
  /// `LayerNorm(hidden_size)` — the adapter's own pre-norm (`self.norm`).
  norm: LayerNorm,
  /// `Linear(hidden_size → adapter_attn_dim)` — quantize-aware.
  linear_1: Linear,
  /// `Linear(adapter_attn_dim → hidden_size)` — quantize-aware.
  linear_2: Linear,
}

#[cfg(feature = "wav2vec2")]
impl AttnAdapterLayer {
  /// `norm(h) → linear_1 → relu → linear_2` — the adapter branch
  /// (`Wav2Vec2AttnAdapterLayer.__call__`, [wav2vec.py:428-433][ad]). Returns
  /// the branch output; the caller adds it to the hidden states.
  ///
  /// [ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L428-L433
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let h = self.norm.forward(hidden_states)?;
    let h = self.linear_1.forward(&h)?;
    let h = relu(&h)?;
    self.linear_2.forward(&h)
  }
}

// ───────────────────────── encoder layer ─────────────────────────

/// The per-layer transformer block, common to both encoder arms (their weights
/// are identical; only the block ordering and the encoder-level `LayerNorm`
/// placement differ — see [`EncoderLayer::forward`] / [`EncoderLayer::forward_stable`]).
///
/// The optional [`Self::adapter_layer`] is the MMS per-attention-block adapter
/// (`Wav2Vec2AttnAdapterLayer`): the reference attaches it **only** to the
/// stable-layer-norm layer (`Wav2Vec2EncoderLayerStableLayerNorm`, when
/// `config.adapter_attn_dim is not None`, [wav2vec.py:484-487][sel-ad]) and
/// adds its output to the hidden states after the feed-forward
/// ([wav2vec.py:503-504][ad-add]). The post-norm layer has no adapter, so it is
/// `None` there; [`EncoderLayer::forward`] never consults it.
///
/// [sel-ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L484-L487
/// [ad-add]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L503-L504
#[cfg(feature = "wav2vec2")]
struct EncoderLayer {
  attention: Attention,
  layer_norm: LayerNorm,
  feed_forward: FeedForward,
  final_layer_norm: LayerNorm,
  /// `Some` iff `config.adapter_attn_dim` is set AND this is a stable-LN layer
  /// (the MMS adapter is attached only to `Wav2Vec2EncoderLayerStableLayerNorm`).
  adapter_layer: Option<AttnAdapterLayer>,
}

#[cfg(feature = "wav2vec2")]
impl EncoderLayer {
  /// **Post-norm** ordering (`Wav2Vec2EncoderLayer`, [wav2vec.py:453-465][el]):
  /// `h = layer_norm(h_in + attn(h_in)); h = final_layer_norm(h + ff(h))` — the
  /// LayerNorms follow the residual adds.
  ///
  /// [el]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L453-L465
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let attn = self.attention.forward(hidden_states)?;
    let h = hidden_states.add(&attn)?;
    let h = self.layer_norm.forward(&h)?;
    let ff = self.feed_forward.forward(&h)?;
    let h = h.add(&ff)?;
    self.final_layer_norm.forward(&h)
  }

  /// **Stable-layer-norm** (pre-norm) ordering
  /// (`Wav2Vec2EncoderLayerStableLayerNorm`, [wav2vec.py:489-508][sel]):
  /// `h = h_in + attn(layer_norm(h_in)); h = h + ff(final_layer_norm(h))` — the
  /// LayerNorms precede their sub-layers, inside the residual.
  ///
  /// When the MMS adapter is present (`adapter_attn_dim` set), its branch
  /// output is added after the feed-forward residual —
  /// `h = h + adapter_layer(h)` ([wav2vec.py:503-504][sel-ad]) — specializing
  /// the language-agnostic backbone to the loaded language.
  ///
  /// [sel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L489-L508
  /// [sel-ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L503-L504
  fn forward_stable(&self, hidden_states: &Array) -> Result<Array> {
    let normed = self.layer_norm.forward(hidden_states)?;
    let attn = self.attention.forward(&normed)?;
    let h = hidden_states.add(&attn)?;
    let ff = self
      .feed_forward
      .forward(&self.final_layer_norm.forward(&h)?)?;
    let h = h.add(&ff)?;
    // MMS per-attention-block adapter (present only when `adapter_attn_dim` is
    // set): `h = h + adapter_layer(h)`.
    match &self.adapter_layer {
      Some(adapter) => {
        let a = adapter.forward(&h)?;
        h.add(&a)
      }
      None => Ok(h),
    }
  }
}

/// ReLU `max(x, 0)` — the MMS attention adapter's activation (`nn.ReLU()`,
/// [wav2vec.py:425][ad]). Implemented as an element-wise `maximum` against a
/// dtype-matched rank-0 zero (so a half-precision adapter stays half-precision
/// rather than being promoted to F32 under MLX type promotion).
///
/// [ad]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L425
#[cfg(feature = "wav2vec2")]
fn relu(x: &Array) -> Result<Array> {
  let zero = scalar_like(0.0, x)?;
  ops::arithmetic::maximum(x, &zero)
}

// ───────────────────────── encoder ─────────────────────────

/// The Standard dialect's transformer encoder (wav2vec2 + HuBERT). Mirrors
/// mlx-audio's two distinct encoder classes — the post-norm `Wav2Vec2Encoder`
/// ([wav2vec.py:511-574][enc]) and the pre-norm `Wav2Vec2EncoderStableLayerNorm`
/// ([wav2vec.py:577-644][senc]) — as two arms selected by
/// `config.do_stable_layer_norm` (`Wav2Vec2Model.__init__`,
/// [wav2vec.py:663-666][sel]).
///
/// Both arms add the positional conv embedding to the hidden states and run the
/// same stack of `EncoderLayer`s; they differ in **where the encoder-level
/// `LayerNorm` sits** and in the per-layer block ordering:
/// - [`StandardEncoder::PostNorm`]: `LayerNorm` is applied **before** the layer
///   stack, and each layer uses `EncoderLayer::forward` (post-norm);
/// - [`StandardEncoder::StableLayerNorm`]: `LayerNorm` is applied **after** the
///   layer stack, and each layer uses `EncoderLayer::forward_stable`
///   (pre-norm).
///
/// It is the Standard dialect's [`Family::Encoder`], implementing the [`Encoder`]
/// trait the generic [`Model`] dispatches through. The CTC path runs a single
/// un-padded utterance, so the trait's `attention_mask` is ignored (the arms
/// apply `Mask::None` self-attention).
///
/// [enc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L511-L574
/// [senc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L577-L644
/// [sel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L663-L666
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub enum StandardEncoder {
  /// `do_stable_layer_norm == false` — `Wav2Vec2Encoder`.
  PostNorm(EncoderInner),
  /// `do_stable_layer_norm == true` — `Wav2Vec2EncoderStableLayerNorm`.
  StableLayerNorm(EncoderInner),
}

/// The fields shared by both encoder arms (their weight layout is identical).
/// Public only because it is the payload of the public [`StandardEncoder`]
/// variants (reachable via [`Family::Encoder`]); its fields stay private.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub struct EncoderInner {
  pos_conv_embed: PositionalConvEmbedding,
  layer_norm: LayerNorm,
  layers: Vec<EncoderLayer>,
}

#[cfg(feature = "wav2vec2")]
impl Encoder for StandardEncoder {
  /// Run the post-norm or stable-layer-norm arm over the projected hidden
  /// states. `attention_mask` is ignored — the CTC path runs a single un-padded
  /// utterance, so the per-layer self-attention applies `Mask::None`.
  fn forward(&self, hidden_states: &Array, _attention_mask: Option<&Array>) -> Result<Array> {
    match self {
      // Post-norm (`Wav2Vec2Encoder.__call__`): pos-embed add → LayerNorm →
      // layer stack. The encoder LayerNorm runs BEFORE the layers.
      Self::PostNorm(inner) => {
        let pos = inner.pos_conv_embed.forward(hidden_states)?;
        let mut h = hidden_states.add(&pos)?;
        h = inner.layer_norm.forward(&h)?;
        for layer in &inner.layers {
          h = layer.forward(&h)?;
        }
        Ok(h)
      }
      // Stable-LN (`Wav2Vec2EncoderStableLayerNorm.__call__`): pos-embed add →
      // layer stack → LayerNorm. The encoder LayerNorm runs AFTER the layers.
      Self::StableLayerNorm(inner) => {
        let pos = inner.pos_conv_embed.forward(hidden_states)?;
        let mut h = hidden_states.add(&pos)?;
        for layer in &inner.layers {
          h = layer.forward_stable(&h)?;
        }
        inner.layer_norm.forward(&h)
      }
    }
  }
}

// ───────────────────────── standard dialect ─────────────────────────

/// The **Standard** dialect — wav2vec2 + HuBERT, the plain self-attention
/// transformer (HuBERT reuses the wav2vec2 encoder architecture). The first and,
/// this phase, only member of the [`Family`]; WavLM (gated relative-position-bias
/// attention) and Conformer (the Conformer block) each add their own dialect in
/// a later phase.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub struct Standard;

#[cfg(feature = "wav2vec2")]
impl Family for Standard {
  type Config = Config;
  type Encoder = StandardEncoder;
  const MODEL_TYPES: &'static [&'static str] = SUPPORTED_MODEL_TYPES;

  /// Build the Standard dialect's [`StandardEncoder`] — selecting the post-norm
  /// or stable-layer-norm arm by `config.do_stable_layer_norm`, with each
  /// projection dense-or-quantized per `quant`. Delegates to `build_encoder`.
  fn build_encoder(
    config: &Self::Config,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self::Encoder> {
    build_encoder(config, weights, quant)
  }
}

#[cfg(feature = "wav2vec2")]
impl FamilyConfig for Config {
  /// In the single-dialect (Standard) phase the dialect config *is* the shared
  /// base config, so the base view is `self`.
  #[inline(always)]
  fn base(&self) -> &Config {
    self
  }

  fn validate(&self) -> Result<()> {
    Config::validate(self)
  }
}

// ───────────────────────── model ─────────────────────────

/// Wav2Vec2 CTC speech recognizer, generic over the [`Family`] dialect (the
/// `wav2vec2` / `hubert` CTC family this phase ships as [`Standard`]).
///
/// The shared scaffolding — the convolutional feature extractor, the feature
/// projection, the CTC head, the vocabulary, and the per-utterance forward /
/// transcribe / CTC contract — lives here once; only the transformer
/// [`Family::Encoder`] varies per dialect. See the [module docs](self) for the
/// architecture and public API.
///
/// The `F: Family` bound on the struct is the documented **storage-shape
/// exception** to the method-local-bounds rule (rust-type-conventions §8): the
/// struct stores `F::Encoder` / `F::Config`, which cannot be *named* without it.
/// All fields are private; reads go through the projected accessors.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub struct Model<F>
where
  F: Family,
{
  config: F::Config,
  feature_encoder: FeatureEncoder,
  feature_projection: FeatureProjection,
  encoder: F::Encoder,
  /// CTC head `Linear(hidden → vocab)` — quantize-aware.
  lm_head: Linear,
  vocab: Vocab,
}

// §8: the `F: Family` bound lives on the impl block (where-form), shared by
// every method here; the struct itself stays bound only by its storage-shape
// exception. These are the dialect-agnostic methods — the per-utterance forward
// / transcribe / CTC contract + the shared accessors — that work for any family
// (they read only the shared fields and dispatch the encoder through the
// [`Encoder`] trait). Construction (`from_weights` / `load`) is dialect-specific
// and lives in the `impl Model<Standard>` block below.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl<F> Model<F>
where
  F: Family,
{
  /// The fixed input sample rate (16 kHz mono) — mlx-audio's
  /// `Model.sample_rate`.
  pub const SAMPLE_RATE: u32 = 16_000;

  /// Conservative upper bound on the TOTAL input element count (batch x time)
  /// for the inherent [`Model::forward`] / [`Model::transcribe`]
  /// path: 60 s at [`Model::SAMPLE_RATE`] (960 000 samples) for the common
  /// mono single-utterance case. The inherent path has no STT-pipeline
  /// `max_audio_seconds` cap, so an over-large waveform — a long single sequence
  /// OR a large batch — would otherwise drive the O(N) convolutional feature
  /// maps and, after the ~320x feature-encoder downsampling, the transformer's
  /// quadratic self-attention without bound (a process-level OOM / DoS). Inputs
  /// whose total element count exceeds this are rejected up front with a
  /// recoverable [`Error::OutOfRange`]; process longer / larger audio in chunks.
  pub const MAX_INPUT_SAMPLES: usize = Self::SAMPLE_RATE as usize * 60;

  /// Reject an over-cap input waveform before any allocation (see
  /// [`Model::MAX_INPUT_SAMPLES`]). `waveform` is `(T,)` or `(B, T)`; the
  /// last axis is the sample count.
  fn ensure_input_within_cap(waveform: &Array) -> Result<()> {
    // Total element count across all axes (batch x time). A checked product so a
    // pathological shape that would overflow usize saturates to usize::MAX and
    // takes the rejected path rather than wrapping to a small value.
    let total = waveform
      .shape()
      .iter()
      .copied()
      .try_fold(1usize, usize::checked_mul)
      .unwrap_or(usize::MAX);
    if total > Self::MAX_INPUT_SAMPLES {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Model inherent path: total input samples (batch x time)",
        "must not exceed MAX_INPUT_SAMPLES (60 s at 16 kHz); process longer or larger audio in chunks",
        total.to_string(),
      )));
    }
    Ok(())
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config_ref(&self) -> &F::Config {
    &self.config
  }

  /// The decoding vocabulary.
  #[inline(always)]
  pub fn vocab_ref(&self) -> &Vocab {
    &self.vocab
  }

  /// The dialect's transformer encoder ([`Family::Encoder`]).
  #[inline(always)]
  pub fn encoder_ref(&self) -> &F::Encoder {
    &self.encoder
  }

  /// The CTC blank class id this model collapses against (the hardcoded
  /// `CTC_BLANK`, pinned by [`Config::validate`]).
  #[inline(always)]
  pub const fn blank_id(&self) -> u32 {
    CTC_BLANK
  }

  /// Run the full forward pass: raw waveform `(B, T)` (or `(T,)` — promoted to
  /// `(1, T)`) → per-frame CTC logits `(B, T', vocab)`.
  ///
  /// This does **not** normalize the waveform (the reference's
  /// zero-mean-unit-variance step happens in `generate`); call
  /// [`Model::transcribe`] for the full normalize → forward → decode
  /// path. Returns a lazy [`Array`] (no implicit eval).
  pub fn forward(&self, waveform: &Array) -> Result<Array> {
    Self::ensure_input_within_cap(waveform)?;
    let input_values = match waveform.ndim() {
      1 => ops::shape::expand_dims_axes(waveform, &[0])?,
      _ => waveform.try_clone()?,
    };
    // (B, 512, T')
    let extract_features = self.feature_encoder.forward(&input_values)?;
    // (B, T', 512)
    let extract_features = ops::shape::transpose_axes(&extract_features, &[0, 2, 1])?;
    // (B, T', 768)
    let hidden_states = self.feature_projection.forward(&extract_features)?;
    let hidden_states = self.encoder.forward(&hidden_states, None)?;
    // (B, T', vocab)
    self.lm_head.forward(&hidden_states)
  }

  /// Transcribe a raw 16 kHz mono waveform to text.
  ///
  /// Mirrors mlx-audio's `Model.generate`: zero-mean / unit-variance normalize
  /// the waveform (`(x - mean) / sqrt(var + 1e-7)`, HF's
  /// `zero_mean_unit_var_norm`), forward to logits, greedy CTC collapse, then
  /// map the token ids through the vocabulary. Requires the model to have been
  /// built with a non-empty [`Vocab`]; otherwise returns an error.
  ///
  /// `waveform` is `(T,)` or `(B, T)`; only the first batch element's
  /// transcript is returned (matching the reference's `decoded[0]`).
  pub fn transcribe(&self, waveform: &Array) -> Result<String> {
    Self::ensure_input_within_cap(waveform)?;
    self.ensure_decodable()?;
    let normed = normalize_waveform(waveform)?;
    let logits = self.forward(&normed)?;
    // argmax over the vocab axis → (B, T') u32 ids.
    let mut predictions = ops::misc::argmax(&logits, Some(-1), false)?;
    let shape = predictions.shape();
    let seq_len = match shape.as_slice() {
      [_, t] => *t,
      [t] => *t,
      _ => {
        return Err(Error::RankMismatch(RankMismatchPayload::new(
          "Model::transcribe: predictions must be rank-1 or rank-2",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    let all_ids = predictions.to_vec::<u32>()?;
    // First batch row (the reference decodes `decoded[0]`).
    let first_row = &all_ids[..seq_len.min(all_ids.len())];
    let tokens = ctc_greedy_collapse(first_row);
    // The shared decode seam: map ids → text via the vocab and trim the edges,
    // identical to the `Box<dyn Transcribe>` path (which reaches the same
    // [`CtcModel::decode_ids`] through the greedy driver).
    Ok(self.decode_ids(&tokens))
  }
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Model<Standard> {
  /// Build a [`Standard`]-dialect model from a parsed [`Config`], the
  /// **sanitized** weight map (run [`sanitize`] first), and an optional [`Vocab`]
  /// (for [`Model::transcribe`]).
  ///
  /// The config is gated by [`Config::validate`] first (so an
  /// unsupported scheme / out-of-scope adapter / malformed dimension is
  /// rejected before any tensor is built); then the matching encoder arm
  /// (post-norm or stable-layer-norm) is wired from the family's width / count
  /// / conv-stack fields. Every weight key the architecture needs must be
  /// present, else [`Error::MissingKey`].
  ///
  /// Beyond the config gate, **every consumed tensor's shape is pinned to the
  /// dimensions the validated config implies** before it is stored or fed to
  /// any op: a checkpoint that passes config validation but carries a
  /// wrong-shaped tensor — a conv weight with a different kernel axis
  /// (a different receptive field), or an `lm_head.weight` with a huge output
  /// dim (a huge-logits allocation) — is rejected here with a typed
  /// [`Error::ShapePairMismatch`] (wrapped in [`Error::LayerKeyed`] naming the
  /// tensor), before any forward pass.
  pub fn from_weights(
    config: Config,
    weights: HashMap<String, Array>,
    vocab: Vocab,
  ) -> Result<Self> {
    Self::from_weights_quantized(config, weights, vocab, None)
  }

  /// Build a model from a parsed [`Config`], the **sanitized** weight
  /// map, an optional [`Vocab`], and an optional parsed quantization config —
  /// the quantization-aware analogue of [`Model::from_weights`] (which is
  /// just this with `quantization = None`).
  ///
  /// When `quantization` is `Some` AND a Linear / projection layer's
  /// `<prefix>.weight` carries the sibling `<prefix>.scales` tensor in the
  /// (sanitized) checkpoint, that projection is built as a quantized layer
  /// ([`crate::nn::MaybeQuantizedLinear::Quantized`]) running
  /// [`crate::ops::quantized::quantized_matmul`] — the weight-map analogue of
  /// mlx-audio's whisper `class_predicate`
  /// (`isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights`,
  /// `mlx_audio/stt/models/whisper/whisper.py:674-676`), with the
  /// `(group_size, bits, mode)` resolved per layer from `quantization`. A dense
  /// projection (no `.scales` sibling) builds exactly as before, so a
  /// non-quantized checkpoint loads identically whether or not a `quantization`
  /// config is supplied.
  ///
  /// Only the Linear / projection layers quantize — the encoder attention
  /// `q/k/v/out` projections, the feed-forward `intermediate` / `output`
  /// projections, the feature projection, and the CTC `lm_head`. The
  /// convolutional feature extractor and the weight-normalized positional conv
  /// embedding stay dense, matching what mlx-audio / MLX quantizes (`nn.Linear`
  /// only).
  ///
  /// `quantization` is the parsed
  /// [`crate::lm::quant::PerLayerQuantization`] (the `config.json` quantization
  /// block parsed by the shared audio resolver
  /// [`crate::audio::load::apply_quantization`], which accepts either a
  /// top-level `quantization` block or the HF `quantization_config` key). A
  /// model whose config carries neither passes `None`.
  ///
  /// The supplied `quantization` is taken **as parsed**: its per-layer override
  /// keys may still carry the on-disk HF backbone prefix
  /// (`wav2vec2.encoder.layers.0.attention.q_proj`). This constructor normalizes
  /// them into the sanitized namespace the builders resolve a layer's scheme in
  /// (`encoder.layers.0.attention.q_proj`, matching the [`sanitize`]d `weights`)
  /// internally, so a per-layer override is applied regardless of whether the
  /// caller pre-normalized it — this is the single reprojection boundary every
  /// caller (the [`Model::load`] reader and any direct public-API caller)
  /// crosses. The normalization is idempotent: passing an already-sanitized
  /// per-layer map is a no-op. Two overrides that reproject to the same
  /// sanitized layer are deduplicated when identical and rejected with
  /// [`Error::KeyCollision`] when they conflict.
  ///
  /// # Errors
  /// The [`Model::from_weights`] errors, plus:
  /// - [`Error::KeyCollision`] if two per-layer overrides reproject to the same
  ///   sanitized layer with conflicting schemes (a genuine config
  ///   contradiction — see the normalization note above);
  /// - [`Error::InvariantViolation`] if a `<prefix>.scales` sibling is present
  ///   but `quantization` resolved no scheme parameters for that layer (the
  ///   weights say quantized, the config says dense);
  /// - [`Error::MissingKey`] / [`Error::ShapePairMismatch`] if a quantized
  ///   layer's packed weight / scales is the wrong shape;
  /// - propagates [`crate::nn::QuantizedLinear::from_parts`]'s structural
  ///   validation of the quantized triple.
  pub fn from_weights_quantized(
    config: Config,
    mut weights: HashMap<String, Array>,
    vocab: Vocab,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Single config-validation gate: reject any unsupported / out-of-scope arm
    // and any malformed dimension BEFORE any tensor is built.
    config.validate()?;

    // Normalize the (possibly HF-prefixed) per-layer override keys into the
    // sanitized namespace the builders resolve a layer's scheme in — the single
    // boundary every caller (the `load()` reader AND any direct public-API
    // caller) crosses. The supplied `weights` are already sanitized (their
    // backbone prefix stripped), but the parsed config's per-layer overrides may
    // still be keyed in the on-disk HF form (`wav2vec2.` / `hubert.`), which
    // would never match the sanitized lookup; without this rewrite the layer
    // would silently fall back to the global scheme. `reproject_quant_keys` is
    // idempotent (an already-sanitized key carries no backbone prefix, so the
    // strip is a no-op), so re-normalizing an already-normalized config is safe.
    // A config with no per-layer overrides (the common case — only a global
    // `{ group_size, bits }`) needs no rewrite and is threaded through unchanged.
    let reprojected = match quantization {
      Some(q) if !q.per_layer_ref().is_empty() => Some(reproject_quant_keys(q)?),
      _ => None,
    };
    let quantization = match &reprojected {
      Some(q) => Some(q),
      None => quantization,
    };

    let feature_encoder = build_feature_encoder(&config, &mut weights)?;
    let feature_projection = build_feature_projection(&config, &mut weights, quantization)?;
    let encoder = Standard::build_encoder(&config, &mut weights, quantization)?;
    // CTC head Linear (out, in) = (vocab_size, hidden_size); bias (vocab_size,).
    // Pinning the exact shape (dense path) is the key allocation guard: an
    // oversized output dim would otherwise drive a huge logits tensor at forward
    // time. The quantized path reaches the same gate via `check_quantized_shape`.
    let lm_head = build_linear(
      &mut weights,
      quantization,
      "lm_head",
      "CTC head weight (vocab_size, hidden_size)",
      "CTC head bias (vocab_size)",
      config.vocab_size,
      config.hidden_size,
      true,
    )?;

    Ok(Self {
      config,
      feature_encoder,
      feature_projection,
      encoder,
      lm_head,
      vocab,
    })
  }

  /// Load a [`Standard`]-dialect model from a local on-disk directory — the
  /// convenience entry point mirroring mlx-audio's `stt.load` for this
  /// architecture.
  ///
  /// Resolves `path` via [`crate::audio::load::get_model_path`] (local-only;
  /// a Hub id is rejected per the project's no-network policy), reads and
  /// parses `config.json`, loads + [`sanitize`]s the single un-sharded
  /// `model.safetensors`, and reads the character `vocab.json` (so
  /// [`Model::transcribe`] works). `vocab.json` is optional — if absent
  /// the model still loads and [`Model::forward`] works, but
  /// `transcribe` then errors.
  ///
  /// Only the single-file `model.safetensors` layout is handled here; sharded
  /// checkpoints are out of scope (a missing file is a clear
  /// [`Error::MissingKey`]).
  ///
  /// A quantized (e.g. 8-bit) checkpoint loads transparently: the
  /// `config.json` quantization block is parsed by the shared audio resolver
  /// [`crate::audio::load::apply_quantization`] — which mirrors mlx-audio by
  /// accepting either a top-level `quantization` block or the HF
  /// `quantization_config` key and defaulting a missing `group_size` to 64 —
  /// and threaded into [`Model::from_weights_quantized`], so each Linear /
  /// projection layer carrying a `<prefix>.scales` sibling is built quantized
  /// while the dense layers (and a checkpoint with no quantization block) load
  /// unchanged.
  ///
  /// For an **MMS** checkpoint (one carrying `adapter_attn_dim` +
  /// `adapter.{lang}.safetensors` files) this loads the default-language
  /// (`"eng"`) adapter — see [`Model::load_with_target_lang`] for a specific
  /// language.
  pub fn load(path: &str) -> Result<Self> {
    Self::load_with_target_lang(path, None)
  }

  /// Load a model, selecting an explicit MMS `target_lang` for the per-language
  /// adapter + vocabulary.
  ///
  /// Identical to [`Model::load`] for a plain wav2vec2 / HuBERT checkpoint
  /// (which has no language adapter — `target_lang` is then only used to pick
  /// the per-language `vocab.json` map, falling back to English / the first
  /// language if the file is monolingual). For an **MMS** checkpoint
  /// (`facebook/mms-1b-all` / `mms-1b-fl102`), this is the Rust analogue of
  /// mlx-audio's `Model.post_load_hook` ([mms.py:130-163][mms]): the base
  /// `model.safetensors` (the language-agnostic backbone) is loaded, then the
  /// `adapter.{target_lang}.safetensors` overlay (the trained adapter-layer
  /// weights + the per-language `lm_head`) is applied on top, and the
  /// per-language `vocab.json` map is selected — specializing the model to the
  /// requested language.
  ///
  /// `target_lang` is the ISO-639-3 code (e.g. `"eng"`, `"fra"`); `None`
  /// requests the default `"eng"` (the language the reference reaches for
  /// first). The adapter file is discovered by preferring the exact
  /// `adapter.{target_lang}.safetensors`, else the lexicographically-smallest
  /// `adapter.*.safetensors`; the per-language `vocab.json` map then follows the
  /// language that was actually selected (so a fallback adapter is decoded with
  /// its own token table, never the requested language's). When an adapter is
  /// selected, that selected language is required to have an **exact** entry in
  /// a nested `{lang: {token: id}}` `vocab.json` — a missing entry is a typed
  /// [`Error::MissingKey`], never a silent fallback to another language's table
  /// ([`Vocab::from_json_for_lang_strict`]). Only the **no-adapter** /
  /// plain-checkpoint path keeps the lenient eng/en/smallest fallback
  /// ([`Vocab::from_json_for_lang`]); a flat language-agnostic `vocab.json` is
  /// used as-is in either case.
  ///
  /// An **MMS** config (`adapter_attn_dim` on the stable-LN arm) *requires* a
  /// per-language adapter — the base `model.safetensors` carries only the
  /// language-agnostic init, so a missing or truncated adapter would silently
  /// build the wrong model. The loader therefore returns a typed
  /// [`Error::MissingKey`] when an MMS checkpoint ships **no** adapter file, or
  /// when the discovered adapter is missing any required tensor (every
  /// per-layer adapter weight together with `lm_head.weight` and
  /// `lm_head.bias`). A plain wav2vec2 / HuBERT checkpoint (no
  /// `adapter_attn_dim` on the stable-LN arm) has no adapter requirement, so
  /// absent `adapter.*` files load without an overlay, unchanged.
  ///
  /// [mms]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/mms/mms.py#L130-L163
  pub fn load_with_target_lang(path: &str, target_lang: Option<&str>) -> Result<Self> {
    let lang = target_lang.unwrap_or(DEFAULT_TARGET_LANG);
    let dir = crate::audio::load::get_model_path(path)?;
    let config_json = crate::audio::load::load_config(&dir)?;
    let config = Config::from_json(&config_json)?;
    // Reject an unsupported architecture arm before reading the (large)
    // weights file — `from_weights` re-checks, but failing here avoids the
    // safetensors read on a config this port cannot serve.
    config.validate()?;
    // Parse the optional quantization block (the per-layer scheme params for a
    // quantized checkpoint) through the shared audio resolver, so wav2vec2
    // resolves quantization exactly the way the rest of the STT subsystem does:
    // it prefers a non-null top-level `quantization`, falls back to the HF
    // `quantization_config` key, and defaults a missing `group_size` to 64
    // (mlx-audio's convention). A config carrying neither key resolves to
    // `None`, so a dense checkpoint loads exactly as before. The per-layer
    // override keys are normalized into the sanitized namespace by
    // [`Model::from_weights_quantized`] itself (the single reprojection
    // boundary every caller crosses), so this raw parsed config is threaded
    // straight through.
    let quantization = crate::audio::load::apply_quantization(&config_json)?;

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "Model::load: model.safetensors not found (sharded checkpoints unsupported)",
        format_smolstr!("{}", weights_path.display()),
      )));
    }
    let raw = crate::io::load_safetensors(&weights_path)?;
    let mut weights = sanitize(raw)?;

    // MMS per-language adapter overlay (mms.py `post_load_hook`): when an MMS
    // checkpoint ships an `adapter.{lang}.safetensors` (or any `adapter.*`),
    // load it and overlay its sanitized keys onto the base map — replacing the
    // language-agnostic adapter-layer init and `lm_head` with the trained
    // per-language ones BEFORE the single build.
    //
    // Discovery + overlay are gated on `requires_adapter` (an MMS config —
    // `adapter_attn_dim` on the stable-LN arm): faithful to the reference, where
    // ONLY the MMS `Model` class defines a `post_load_hook` adapter overlay
    // ([mms.py:130-163][mms]) while the plain `Wav2Vec2ForCTC` loader
    // ([wav2vec.py:766-775]) just loads `model.safetensors` with no adapter
    // discovery at all. So a NON-MMS checkpoint never overlays an adapter file —
    // a stray `adapter.*.safetensors` next to a plain wav2vec2 / HuBERT (or a
    // post-norm) checkpoint is ignored, not blindly merged into the base (which
    // would let a sidecar-only adapter clobber a base packed weight's sidecars
    // and build a silent quantized hybrid). For an MMS config the overlay is
    // ADDITIONALLY key-restricted in `overlay_adapter_weights` (defense in depth:
    // only the allowed adapter-layer + lm_head keys, no foreign key, no orphan
    // sidecar) and required to be complete.
    //
    // Discovery returns the language ACTUALLY selected (the requested one on an
    // exact hit, else the fallback file's language); the per-language `vocab.json`
    // selection below follows THAT language, so the overlaid adapter, the
    // per-language `lm_head`, and the vocab always describe the same language.
    let requires_adapter = config_requires_adapter(&config);
    let selected = if requires_adapter {
      adapter_file_for(&dir, lang)?
    } else {
      None
    };
    // An MMS config REQUIRES a per-language adapter: the base `model.safetensors`
    // is only language-agnostically initialized, so an absent adapter file would
    // silently build the WRONG (base) model. Reject it as a typed load failure
    // rather than transcribing with the untrained adapter init.
    if requires_adapter && selected.is_none() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "Model::load: this MMS checkpoint (config sets adapter_attn_dim on the stable-LN arm) \
         requires a per-language adapter file, but no adapter.{lang}.safetensors was found",
        format_smolstr!("adapter.{lang}.safetensors"),
      )));
    }
    // The vocab follows the SELECTED adapter's language (not the originally
    // requested `lang`), so a fallback adapter is decoded with its own token
    // table. Absent any adapter (a plain checkpoint), the vocab keeps the
    // requested language.
    let vocab_lang = selected.as_ref().map_or(lang, |s| s.lang.as_str());
    if let Some(selected) = &selected {
      overlay_adapter_weights(&mut weights, &selected.path, &config, requires_adapter)?;
    }

    // vocab.json is optional; an absent file leaves an empty Vocab (forward
    // still works, transcribe then errors with a clear message). Reuse the
    // shared bounded reader so a hostile directory can't OOM the loader. The
    // language-aware parser selects the `vocab_lang` map for an MMS multilingual
    // `{lang: {token: id}}` vocab, and reads a flat `{token: id}` vocab
    // unchanged (so a `base-960h` vocab loads exactly as before).
    //
    // When an adapter was actually selected (the MMS per-language path), the
    // nested vocab MUST carry an exact `vocab_lang` entry — the STRICT parser
    // rejects a missing one with a typed error rather than silently falling back
    // to another language's token table (which would decode the overlaid
    // adapter's logits with the wrong language). The lenient eng/en/smallest
    // fallback is kept ONLY for the no-adapter / plain-checkpoint path (a flat
    // vocab, or no overlay), and a flat language-agnostic vocab is used as-is in
    // both cases.
    let vocab_path = dir.join("vocab.json");
    let vocab = match crate::lm::load::read_bounded_config_file(&vocab_path, "wav2vec2 vocab.json")?
    {
      Some(body) if selected.is_some() => Vocab::from_json_for_lang_strict(&body, vocab_lang)?,
      Some(body) => Vocab::from_json_for_lang(&body, vocab_lang)?,
      None => Vocab::default(),
    };

    Self::from_weights_quantized(config, weights, vocab, quantization.as_ref())
  }
}

// §8: the `F: Family` bound is on the impl (where-form). The CTC contract is
// dialect-agnostic — it reads the shared vocabulary + blank id and runs the
// shared normalize → forward path — so every family's `Model<F>` satisfies it.
#[cfg(feature = "wav2vec2")]
impl<F> CtcModel for Model<F>
where
  F: Family,
{
  /// Per-frame CTC logits `(T', vocab)` for the mono `waveform` — the
  /// non-autoregressive frontend the greedy-collapse driver
  /// ([`greedy_ctc_transcribe`]) reads.
  ///
  /// Normalizes the waveform (HF's `zero_mean_unit_var_norm`, as [`Model::transcribe`]
  /// does), runs the shared forward to `(1, T', vocab)`, and squeezes the
  /// single-utterance batch axis to the rank-2 `(T', vocab)` the driver expects.
  fn logits(&self, waveform: &Array) -> Result<Array> {
    let normed = normalize_waveform(waveform)?;
    let logits = self.forward(&normed)?;
    // forward returns (1, T', vocab) for a single-utterance (T,)/(1, T) input;
    // drop the leading batch axis to the rank-2 (T', vocab) the driver reads.
    ops::shape::squeeze_axes(&logits, &[0])
  }

  /// The CTC blank class id collapsed out of the greedy decode (the hardcoded
  /// `CTC_BLANK`, pinned by [`Config::validate`]).
  #[inline(always)]
  fn blank_id(&self) -> u32 {
    CTC_BLANK
  }

  /// Map a collapsed id sequence to text via the model's [`Vocab`], then trim
  /// surrounding whitespace — the reference's
  /// `"".join(...).replace("|", " ").strip()` (the word-delimiter `|` becomes a
  /// space; unknown ids contribute nothing; the final `.strip()` drops the
  /// leading / trailing spaces the delimiter mapping leaves at the utterance
  /// edges, [wav2vec.py:101-103][gen]).
  ///
  /// This is the single decode seam every transcription path runs: both the
  /// greedy-collapse driver ([`greedy_ctc_transcribe`]) and the inherent
  /// [`Model::transcribe`] map collapsed ids to final text through here, so the
  /// `|`→space mapping and the edge trim are applied identically regardless of
  /// which path produced the ids.
  ///
  /// [gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L101-L103
  fn decode_ids(&self, ids: &[u32]) -> String {
    self.vocab.tokens_to_text(ids).trim().to_string()
  }

  /// Reject transcription on a model built without a vocabulary — the
  /// empty-vocabulary guard, enforced once at the shared chokepoint.
  ///
  /// A model loaded without a `vocab.json` carries an empty [`Vocab`]: the
  /// forward still runs (the CTC head is architecture-sized), but there is no
  /// `id → token` map to render the decoded ids, so [`Self::decode_ids`] would
  /// map every id to nothing and the text would be silently empty. Overriding
  /// the [`CtcModel::ensure_decodable`] default makes the empty-vocabulary case
  /// a typed [`Error::InvariantViolation`] at the one chokepoint every
  /// text-producing route passes through: [`greedy_ctc_transcribe`] calls it at
  /// its start (covering both the [`Transcribe`] impl and a direct
  /// `greedy_ctc_transcribe(&model, …)`), and the inherent [`Model::transcribe`]
  /// — which decodes without the driver — calls it directly. So all three
  /// routes reject the empty vocabulary identically rather than succeeding with
  /// empty text.
  fn ensure_decodable(&self) -> Result<()> {
    if self.vocab.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Model::transcribe",
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)",
      )));
    }
    Ok(())
  }
}

// CTC models opt into [`Transcribe`] by delegating to the shared
// [`greedy_ctc_transcribe`] driver from their own impl (the documented CTC
// opt-in on [`CtcModel`]), so a loaded `Model<F>` is usable as `Box<dyn
// Transcribe>`.
#[cfg(feature = "wav2vec2")]
impl<F> Transcribe for Model<F>
where
  F: Family,
{
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription> {
    // The empty-vocabulary guard is enforced once at the shared chokepoint:
    // `greedy_ctc_transcribe` calls `self.ensure_decodable()` at its start, so a
    // model loaded without a `vocab.json` is rejected here too — with the same
    // typed error the inherent [`Model::transcribe`] returns — rather than the
    // greedy driver silently succeeding with an empty transcription. No separate
    // per-route call is needed; the driver subsumes it.
    greedy_ctc_transcribe(self, audio, opts)
  }
}

/// Load a wav2vec2-family CTC model from a local on-disk directory, erasing the
/// dialect at the loader boundary — the one `dyn` point. Reads `config.json`,
/// dispatches on its `model_type` to the matching [`Family`] dialect, and
/// returns the loaded model as a `Box<dyn Transcribe>`.
///
/// This phase serves the [`Standard`] dialect (wav2vec2 + HuBERT + MMS); a
/// `model_type` no dialect claims is rejected with a typed
/// [`Error::UnknownEnumValue`]. For the concrete (non-erased)
/// [`Model<Standard>`] — with its inherent [`Model::forward`] /
/// [`Model::transcribe`] CTC API — call [`Model::<Standard>::load`] directly.
///
/// Loads the default MMS language (`"eng"`) for an MMS checkpoint; use
/// [`load_with_target_lang`] to select a specific language.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn load(path: &str) -> Result<Box<dyn Transcribe>> {
  load_with_target_lang(path, None)
}

/// Load a wav2vec2-family CTC model, selecting an explicit MMS `target_lang` —
/// the dialect-erased analogue of [`Model::<Standard>::load_with_target_lang`].
///
/// `target_lang` selects the MMS per-language adapter + vocabulary (`None`
/// requests the default `"eng"`); it is inert for a plain wav2vec2 / HuBERT
/// checkpoint (no language adapter). Dispatches on `model_type` to the matching
/// [`Family`] dialect exactly as [`load`]; a `model_type` no dialect claims is
/// rejected with [`Error::UnknownEnumValue`].
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub fn load_with_target_lang(path: &str, target_lang: Option<&str>) -> Result<Box<dyn Transcribe>> {
  let dir = crate::audio::load::get_model_path(path)?;
  let config_json = crate::audio::load::load_config(&dir)?;
  let config = Config::from_json(&config_json)?;
  let model_type = config.model_type();
  if Standard::MODEL_TYPES.contains(&model_type) {
    Ok(Box::new(Model::<Standard>::load_with_target_lang(
      path,
      target_lang,
    )?))
  } else {
    Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
      "wav2vec2 load: model_type",
      model_type,
      SUPPORTED_MODEL_TYPES,
    )))
  }
}

/// Zero-mean / unit-variance waveform normalization, HF's
/// `zero_mean_unit_var_norm`: `(x - mean(x, -1)) / sqrt(var(x, -1) + 1e-7)`
/// (the `sqrt(var + eps)` form HF's feature extractor uses, not mms.py's
/// `std + eps`). Operates per row over the last axis. Promotes a `(T,)` input
/// to `(1, T)` first.
#[cfg(feature = "wav2vec2")]
fn normalize_waveform(waveform: &Array) -> Result<Array> {
  let x = match waveform.ndim() {
    1 => ops::shape::expand_dims_axes(waveform, &[0])?,
    _ => waveform.try_clone()?,
  };
  let mean = ops::reduction::mean_axes(&x, &[-1], true)?;
  let var = ops::reduction::var_axes(&x, &[-1], true, 0)?;
  // Cast the eps to the waveform's dtype so a half-precision input is not
  // silently promoted to F32 here (a bare F32 scalar would, under MLX type
  // promotion); the reference's `var + 1e-7` keeps the operand dtype.
  let eps = scalar_like(1e-7, &var)?;
  let denom = var.add(&eps)?.sqrt()?;
  let centered = x.subtract(&mean)?;
  centered.divide(&denom)
}

// ───────────────────────── builders ─────────────────────────

/// Read `feature_extractor.conv_layers.{i}.*` into the
/// `num_feat_extract_layers`-layer [`FeatureEncoder`], branching on
/// `feat_extract_norm` exactly as `Wav2Vec2FeatureEncoder.__init__`
/// ([wav2vec.py:254-267][fe]):
///
/// - `"group"` → an L0 `Wav2Vec2GroupNormConvLayer` (an affine GroupNorm) then
///   `Wav2Vec2NoLayerNormConvLayer`s (no norm);
/// - `"layer"` → a `Wav2Vec2LayerNormConvLayer` at every layer (an affine
///   LayerNorm over the conv output width).
///
/// Each layer carries the resolved `feat_extract_activation` and, when
/// `config.conv_bias`, its `conv.bias`. An unsupported `feat_extract_norm` is
/// rejected by [`Config::feat_extract_norm_scheme`] (matching the reference's
/// `else: raise`).
///
/// [fe]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L254-L267
#[cfg(feature = "wav2vec2")]
fn build_feature_encoder(
  config: &Config,
  weights: &mut HashMap<String, Array>,
) -> Result<FeatureEncoder> {
  // `validate` (run by `from_weights` before any builder) already pinned
  // `num_feat_extract_layers` positive and the conv arrays to exactly that
  // length with positive entries; re-derive the count and re-check the arrays
  // cover `n` here defensively (the builder must never index past the end).
  let n = config.num_feat_extract_layers;
  if n <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Config: num_feat_extract_layers",
      "must be positive (> 0)",
      format_smolstr!("{n}"),
    )));
  }
  let n_usize = n as usize;
  if config.conv_dim.len() < n_usize
    || config.conv_stride.len() < n_usize
    || config.conv_kernel.len() < n_usize
  {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "Config: conv_dim/stride/kernel length",
      n_usize,
      config
        .conv_dim
        .len()
        .min(config.conv_stride.len())
        .min(config.conv_kernel.len()),
    )));
  }
  let activation = Activation::resolve(
    &config.feat_extract_activation,
    "Config: feat_extract_activation",
  )?;
  // The feature-extractor normalization scheme — `"group"` (L0 GroupNorm only)
  // vs `"layer"` (LayerNorm at every layer). `validate` already accepted it; an
  // unsupported value is rejected again here (the builder never runs an
  // unresolved scheme).
  let norm_scheme = config.feat_extract_norm_scheme()?;
  let mut conv_layers = Vec::new();
  reserve_or_error(&mut conv_layers, "feature-encoder conv layers", n_usize)?;
  for i in 0..n_usize {
    let prefix = format!("feature_extractor.conv_layers.{i}");
    // Post-sanitize channels-last conv weight is (out, k, in): out = conv_dim[i],
    // k = conv_kernel[i], in = 1 for L0 else conv_dim[i - 1]. A deviating kernel
    // axis would silently change the receptive field, so pin the exact shape.
    let in_ch = if i == 0 { 1 } else { config.conv_dim[i - 1] };
    let weight = take_shaped(
      weights,
      &format!("{prefix}.conv.weight"),
      "feature-encoder conv weight (out, k, in)",
      &[config.conv_dim[i], config.conv_kernel[i], in_ch],
    )?;
    // `nn.Conv1d(bias=config.conv_bias)`: the bias `(out,)` exists iff
    // `conv_bias`. Pin its shape to the conv output width.
    let bias = if config.conv_bias {
      Some(take_shaped(
        weights,
        &format!("{prefix}.conv.bias"),
        "feature-encoder conv bias (conv_dim[i])",
        &[config.conv_dim[i]],
      )?)
    } else {
      None
    };
    let stride = config.conv_stride[i];
    // Per-layer normalization depends on the feature-extractor scheme:
    //   - "group" (`Wav2Vec2GroupNormConvLayer` at L0, then
    //     `Wav2Vec2NoLayerNormConvLayer`s): ONLY L0 carries an affine
    //     pytorch-compatible GroupNorm (num_groups == dims == conv_dim[0]); the
    //     rest have no norm;
    //   - "layer" (`Wav2Vec2LayerNormConvLayer` at every layer): EVERY layer
    //     carries an affine LayerNorm over its conv output width (conv_dim[i]),
    //     and there is no GroupNorm.
    // Both store their affine under the `.layer_norm.{weight,bias}` key (the HF
    // submodule is named `layer_norm` even for the GroupNorm variant).
    let dims = config.conv_dim[i];
    let (group_norm, layer_norm) = match norm_scheme {
      FeatExtractNorm::Group if i == 0 => {
        let gn_weight = take_shaped(
          weights,
          &format!("{prefix}.layer_norm.weight"),
          "feature-encoder L0 GroupNorm weight (conv_dim[0])",
          &[dims],
        )?;
        let gn_bias = take_shaped(
          weights,
          &format!("{prefix}.layer_norm.bias"),
          "feature-encoder L0 GroupNorm bias (conv_dim[0])",
          &[dims],
        )?;
        let gn = GroupNorm::with_affine(
          dims,
          dims,
          config.layer_norm_eps,
          Some((gn_weight, gn_bias)),
          true,
        )?;
        (Some(gn), None)
      }
      FeatExtractNorm::Group => (None, None),
      FeatExtractNorm::Layer => {
        let ln_weight = take_shaped(
          weights,
          &format!("{prefix}.layer_norm.weight"),
          "feature-encoder LayerNorm weight (conv_dim[i])",
          &[dims],
        )?;
        let ln_bias = take_shaped(
          weights,
          &format!("{prefix}.layer_norm.bias"),
          "feature-encoder LayerNorm bias (conv_dim[i])",
          &[dims],
        )?;
        let ln = LayerNorm::new(Some(ln_weight), Some(ln_bias), config.layer_norm_eps);
        (None, Some(ln))
      }
    };
    conv_layers.push(ConvLayer {
      weight,
      bias,
      stride,
      group_norm,
      layer_norm,
      activation,
    });
  }
  Ok(FeatureEncoder { conv_layers })
}

/// Read `feature_projection.*` into the [`FeatureProjection`]. The LayerNorm
/// stays dense; the `projection` Linear is quantize-aware (the dense / quantized
/// dispatch keyed on `<prefix>.scales`, via [`build_linear`]).
#[cfg(feature = "wav2vec2")]
fn build_feature_projection(
  config: &Config,
  weights: &mut HashMap<String, Array>,
  quant: Option<&PerLayerQuantization>,
) -> Result<FeatureProjection> {
  // The feature encoder's last conv width (conv_dim[-1]) is the projection's
  // input dim and the pre-norm's normalized dim.
  let conv_dim_last = *config.conv_dim.last().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "Config: conv_dim",
      "must be non-empty (the projection reads conv_dim[-1])",
    ))
  })?;
  // The projection LayerNorm is conditional on `feat_proj_layer_norm`, but that
  // flag is HuBERT-only: HF's `Wav2Vec2FeatureProjection` ALWAYS applies the
  // LayerNorm, while only `HubertFeatureProjection` gates it. So the no-LayerNorm
  // arm is taken only when the flag is `false` AND `model_type == "hubert"`; a
  // wav2vec2 `model_type` always applies the LayerNorm (and `Config::validate`
  // already rejects `feat_proj_layer_norm = false` for a non-hubert model_type,
  // so this `is_hubert()` guard is the matching graph-construction half — a
  // faithful mirror of `HubertFeatureProjection`'s `if config.feat_proj_layer_norm`
  // guard, never building the wav2vec2 graph with its LayerNorm dropped). When the
  // LayerNorm is skipped the projection feeds the linear the un-normalized
  // feature-encoder output directly, so neither the affine weight nor bias is
  // consumed.
  let apply_layer_norm = config.feat_proj_layer_norm || !config.is_hubert();
  let layer_norm = if apply_layer_norm {
    let ln_weight = take_shaped(
      weights,
      "feature_projection.layer_norm.weight",
      "feature_projection LayerNorm weight (conv_dim[-1])",
      &[conv_dim_last],
    )?;
    let ln_bias = take_shaped(
      weights,
      "feature_projection.layer_norm.bias",
      "feature_projection LayerNorm bias (conv_dim[-1])",
      &[conv_dim_last],
    )?;
    Some(LayerNorm::new(
      Some(ln_weight),
      Some(ln_bias),
      config.layer_norm_eps,
    ))
  } else {
    None
  };
  // HF Linear weight is (out, in) = (hidden_size, conv_dim[-1]); bias (hidden_size,).
  let projection = build_linear(
    weights,
    quant,
    "feature_projection.projection",
    "feature_projection projection weight (hidden_size, conv_dim[-1])",
    "feature_projection projection bias (hidden_size)",
    config.hidden_size,
    conv_dim_last,
    true,
  )?;
  Ok(FeatureProjection {
    layer_norm,
    projection,
  })
}

/// Read `encoder.*` into the [`StandardEncoder`]. The positional conv embedding
/// and the LayerNorms stay dense; each encoder layer's attention `q/k/v/out` and
/// feed-forward `intermediate` / `output` projections are quantize-aware (the
/// dense / quantized dispatch keyed on `<prefix>.scales`, via [`build_linear`]).
#[cfg(feature = "wav2vec2")]
fn build_encoder(
  config: &Config,
  weights: &mut HashMap<String, Array>,
  quant: Option<&PerLayerQuantization>,
) -> Result<StandardEncoder> {
  // Positional conv embedding: reconstruct the fused weight-normalized kernel
  // once, then a plain grouped conv at forward time. Post-sanitize MLX layout
  // (out, k, in/groups): weight_v is (hidden_size, num_conv_pos_embeddings,
  // hidden_size / num_conv_pos_embedding_groups); weight_g is the per-kernel
  // magnitude (1, num_conv_pos_embeddings, 1); bias is (hidden_size,). A
  // deviating kernel axis here would silently change the positional receptive
  // field, so pin every exact shape.
  let kernel = config.num_conv_pos_embeddings;
  require_divisible(
    "Config: hidden_size",
    config.hidden_size,
    "Config: num_conv_pos_embedding_groups",
    config.num_conv_pos_embedding_groups,
  )?;
  let pos_in_per_group = config.hidden_size / config.num_conv_pos_embedding_groups;
  let weight_g = take_shaped(
    weights,
    "encoder.pos_conv_embed.conv.weight_g",
    "positional conv weight_g (1, num_conv_pos_embeddings, 1)",
    &[1, kernel, 1],
  )?;
  let weight_v = take_shaped(
    weights,
    "encoder.pos_conv_embed.conv.weight_v",
    "positional conv weight_v (hidden_size, num_conv_pos_embeddings, hidden_size / groups)",
    &[config.hidden_size, kernel, pos_in_per_group],
  )?;
  let pos_weight = reconstruct_wn_weight(&weight_g, &weight_v)?;
  let pos_bias = take_shaped(
    weights,
    "encoder.pos_conv_embed.conv.bias",
    "positional conv bias (hidden_size)",
    &[config.hidden_size],
  )?;
  // HF's Wav2Vec2PositionalConvEmbedding applies
  // `ACT2FN[config.feat_extract_activation]` — the same activation the feature
  // encoder uses (resolved by the same dispatch ConvLayer/FeedForward use), not
  // a hardcoded GELU.
  let pos_activation = Activation::resolve(
    &config.feat_extract_activation,
    "Config: feat_extract_activation",
  )?;
  let pos_conv_embed = PositionalConvEmbedding {
    weight: pos_weight,
    bias: pos_bias,
    groups: config.num_conv_pos_embedding_groups,
    padding: kernel / 2,
    num_pad_remove: if kernel % 2 == 0 { 1 } else { 0 },
    activation: pos_activation,
  };

  let ln_weight = take_shaped(
    weights,
    "encoder.layer_norm.weight",
    "encoder LayerNorm weight (hidden_size)",
    &[config.hidden_size],
  )?;
  let ln_bias = take_shaped(
    weights,
    "encoder.layer_norm.bias",
    "encoder LayerNorm bias (hidden_size)",
    &[config.hidden_size],
  )?;
  let layer_norm = LayerNorm::new(Some(ln_weight), Some(ln_bias), config.layer_norm_eps);

  let num_layers = config.num_hidden_layers;
  if num_layers < 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Config: num_hidden_layers",
      "must be non-negative",
      format_smolstr!("{num_layers}"),
    )));
  }
  let head_dim = config.head_dim()?;
  let scaling = (head_dim as f32).powf(-0.5);
  let activation = Activation::resolve(&config.hidden_act, "Config: hidden_act")?;
  // The MMS per-attention-block adapter is attached to every encoder layer ONLY
  // when `adapter_attn_dim` is set AND this is the stable-layer-norm arm — the
  // reference puts `Wav2Vec2AttnAdapterLayer` only on
  // `Wav2Vec2EncoderLayerStableLayerNorm` ([wav2vec.py:484-487]), never on the
  // post-norm `Wav2Vec2EncoderLayer`. So a post-norm checkpoint (or one without
  // `adapter_attn_dim`) builds no adapter. A non-positive `adapter_attn_dim` is
  // a malformed config (it sizes the adapter bottleneck Linears).
  let adapter_dim = match config.adapter_attn_dim {
    Some(d) if config.do_stable_layer_norm => {
      require_positive("Config: adapter_attn_dim", d)?;
      Some(d)
    }
    _ => None,
  };
  // Per-layer expected shapes (derived from the validated config): every
  // attention projection is a square (hidden_size, hidden_size) Linear with a
  // (hidden_size,) bias; the LayerNorms are (hidden_size,); the feed-forward is
  // (intermediate_size, hidden_size) + (hidden_size, intermediate_size) Linears
  // with their respective biases.
  let hs = config.hidden_size;
  let inter = config.intermediate_size;
  let mut layers = Vec::new();
  reserve_or_error(&mut layers, "encoder layers", num_layers as usize)?;
  for i in 0..num_layers {
    let prefix = format!("encoder.layers.{i}");
    // Each attention projection is a square (hidden, hidden) Linear with a
    // (hidden,) bias — quantize-aware (the dense / quantized dispatch keyed on
    // `<prefix>.scales`). Built before the struct literal because `build_linear`
    // borrows `weights` mutably.
    let q_proj = build_linear(
      weights,
      quant,
      &format!("{prefix}.attention.q_proj"),
      "attention q_proj weight (hidden_size, hidden_size)",
      "attention q_proj bias (hidden_size)",
      hs,
      hs,
      true,
    )?;
    let k_proj = build_linear(
      weights,
      quant,
      &format!("{prefix}.attention.k_proj"),
      "attention k_proj weight (hidden_size, hidden_size)",
      "attention k_proj bias (hidden_size)",
      hs,
      hs,
      true,
    )?;
    let v_proj = build_linear(
      weights,
      quant,
      &format!("{prefix}.attention.v_proj"),
      "attention v_proj weight (hidden_size, hidden_size)",
      "attention v_proj bias (hidden_size)",
      hs,
      hs,
      true,
    )?;
    let out_proj = build_linear(
      weights,
      quant,
      &format!("{prefix}.attention.out_proj"),
      "attention out_proj weight (hidden_size, hidden_size)",
      "attention out_proj bias (hidden_size)",
      hs,
      hs,
      true,
    )?;
    let attention = Attention {
      q_proj,
      k_proj,
      v_proj,
      out_proj,
      num_heads: config.num_attention_heads,
      head_dim,
      scaling,
    };
    let el_ln_weight = take_shaped(
      weights,
      &format!("{prefix}.layer_norm.weight"),
      "encoder-layer LayerNorm weight (hidden_size)",
      &[hs],
    )?;
    let el_ln_bias = take_shaped(
      weights,
      &format!("{prefix}.layer_norm.bias"),
      "encoder-layer LayerNorm bias (hidden_size)",
      &[hs],
    )?;
    let layer_norm_l = LayerNorm::new(Some(el_ln_weight), Some(el_ln_bias), config.layer_norm_eps);
    // Feed-forward `intermediate` (hidden → intermediate) and `output`
    // (intermediate → hidden) projections — quantize-aware.
    let intermediate = build_linear(
      weights,
      quant,
      &format!("{prefix}.feed_forward.intermediate_dense"),
      "feed_forward intermediate weight (intermediate_size, hidden_size)",
      "feed_forward intermediate bias (intermediate_size)",
      inter,
      hs,
      true,
    )?;
    let output = build_linear(
      weights,
      quant,
      &format!("{prefix}.feed_forward.output_dense"),
      "feed_forward output weight (hidden_size, intermediate_size)",
      "feed_forward output bias (hidden_size)",
      hs,
      inter,
      true,
    )?;
    let feed_forward = FeedForward {
      intermediate,
      output,
      activation,
    };
    let fln_weight = take_shaped(
      weights,
      &format!("{prefix}.final_layer_norm.weight"),
      "encoder-layer final LayerNorm weight (hidden_size)",
      &[hs],
    )?;
    let fln_bias = take_shaped(
      weights,
      &format!("{prefix}.final_layer_norm.bias"),
      "encoder-layer final LayerNorm bias (hidden_size)",
      &[hs],
    )?;
    let final_layer_norm = LayerNorm::new(Some(fln_weight), Some(fln_bias), config.layer_norm_eps);
    // MMS per-attention-block adapter (`encoder.layers.{i}.adapter_layer.*`),
    // built only when `adapter_dim` is set (an MMS stable-LN checkpoint):
    // `LayerNorm(hidden) → Linear(hidden → adapter_dim) → ReLU →
    // Linear(adapter_dim → hidden)`. The adapter weights ship per-language in
    // `adapter.{lang}.safetensors`; the base checkpoint's `model.safetensors`
    // carries them too (the language-agnostic init), so this consumes them at
    // base load and the per-language overlay later replaces them by key.
    let adapter_layer = match adapter_dim {
      Some(d) => {
        let ad_prefix = format!("{prefix}.adapter_layer");
        let an_weight = take_shaped(
          weights,
          &format!("{ad_prefix}.norm.weight"),
          "adapter LayerNorm weight (hidden_size)",
          &[hs],
        )?;
        let an_bias = take_shaped(
          weights,
          &format!("{ad_prefix}.norm.bias"),
          "adapter LayerNorm bias (hidden_size)",
          &[hs],
        )?;
        let norm = LayerNorm::new(Some(an_weight), Some(an_bias), config.layer_norm_eps);
        // `Linear(hidden → adapter_dim)` then `Linear(adapter_dim → hidden)`,
        // both with the `nn.Linear` default bias — quantize-aware.
        let linear_1 = build_linear(
          weights,
          quant,
          &format!("{ad_prefix}.linear_1"),
          "adapter linear_1 weight (adapter_attn_dim, hidden_size)",
          "adapter linear_1 bias (adapter_attn_dim)",
          d,
          hs,
          true,
        )?;
        let linear_2 = build_linear(
          weights,
          quant,
          &format!("{ad_prefix}.linear_2"),
          "adapter linear_2 weight (hidden_size, adapter_attn_dim)",
          "adapter linear_2 bias (hidden_size)",
          hs,
          d,
          true,
        )?;
        Some(AttnAdapterLayer {
          norm,
          linear_1,
          linear_2,
        })
      }
      None => None,
    };
    layers.push(EncoderLayer {
      attention,
      layer_norm: layer_norm_l,
      feed_forward,
      final_layer_norm,
      adapter_layer,
    });
  }

  // Select the encoder arm by `do_stable_layer_norm` (the reference's
  // `Wav2Vec2Model.__init__` choice between `Wav2Vec2Encoder` and
  // `Wav2Vec2EncoderStableLayerNorm`). The weights are identical; only the
  // forward ordering / encoder-LayerNorm placement differ.
  let inner = EncoderInner {
    pos_conv_embed,
    layer_norm,
    layers,
  };
  Ok(if config.do_stable_layer_norm {
    StandardEncoder::StableLayerNorm(inner)
  } else {
    StandardEncoder::PostNorm(inner)
  })
}

// ───────────────────────── small helpers ─────────────────────────

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "wav2vec2")]
fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      format_smolstr!("{d}"),
    ))
  })
}

/// Build a rank-0 constant of `value`, cast to `like`'s dtype — the
/// dtype-preserving scalar the activation helpers in
/// [`crate::lm::nn::activations`] use.
///
/// A scale / constant built as a bare F32 promotes an F16/BF16 operand to F32
/// under MLX type promotion; casting it to the operand dtype first keeps a
/// half-precision activation half-precision (preserving the checkpoint's
/// mixed-precision numerics and not inflating memory). Rank-0 (empty shape) so
/// it NumPy-broadcasts against any operand rank without lifting it.
#[cfg(feature = "wav2vec2")]
fn scalar_like(value: f32, like: &Array) -> Result<Array> {
  let dtype = like.dtype()?;
  ops::misc::astype(&Array::full::<f32>(&[0i32; 0], value)?, dtype)
}

#[cfg(all(test, feature = "wav2vec2"))]
mod tests;
