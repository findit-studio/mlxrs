//! Wav2Vec2 CTC speech recognizer (the `wav2vec2` / `hubert` CTC family).
//!
//! Port of mlx-audio's `Wav2Vec2ForCTC` — the backbone in
//! [`stt/models/wav2vec/wav2vec.py`][wav2vec] (feature encoder + feature
//! projection + transformer encoder) composed with the CTC head + greedy
//! decode + waveform normalization in [`stt/models/mms/mms.py`][mms]. MMS
//! *is* Wav2Vec2ForCTC; the MMS language-adapter logic is dropped here.
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
//! - [`Wav2Vec2Ctc::forward`] — waveform `(B, T)` → logits `(B, T', V)`.
//! - [`Wav2Vec2Ctc::transcribe`] — waveform → decoded `String`
//!   (normalize → forward → greedy CTC collapse → vocabulary map).
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
//! ## Out of scope
//!
//! The `feat_extract_norm = "layer"` feature-encoder arm, the MMS language
//! adapters, the post-encoder conv adapter / per-layer attention adapter
//! (`add_adapter` / `adapter_attn_dim`), the HuBERT-only no-LayerNorm feature
//! projection (`feat_proj_layer_norm = false`) and batch-norm positional conv
//! (`conv_pos_batch_norm = true`) arms, Conformer relative-position attention,
//! sharded-checkpoint loading, and a configurable CTC blank are **not** wired;
//! a config that needs one of them is rejected by
//! [`Wav2Vec2Config::validate`] with a typed error. (Both HuBERT flag defaults
//! — `feat_proj_layer_norm = true`, `conv_pos_batch_norm = false` — match the
//! wired graph, so a default HuBERT checkpoint is faithfully supported; only the
//! non-default arm is rejected.)
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

use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{
    Error, InvariantViolationPayload, LayerKeyedPayload, LengthMismatchPayload,
    MalformedDataPayload, MissingKeyPayload, NonFiniteScalarPayload, OutOfRangePayload,
    ParsePayload, RankMismatchPayload, Result, ShapePairMismatchPayload, UnknownEnumValuePayload,
  },
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::{GroupNorm, LayerNorm},
  },
  model_validation::{
    checked_mul, insert_unique, pin_bool, pin_str, require_divisible, require_positive,
    reserve_or_error,
  },
  ops,
};

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
pub struct Wav2Vec2Config {
  /// Architecture id (`config.json` `model_type`). Accepted: `"wav2vec2"` and
  /// `"hubert"` — the family that shares the plain self-attention transformer
  /// this port wires (HuBERT reuses the wav2vec2 encoder architecture). WavLM
  /// is **rejected** by [`Wav2Vec2Config::validate`]: it needs gated
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
  /// Feature-encoder normalization scheme. `"group"` (the `base`/`large`
  /// default) is the only scheme this port wires — see
  /// [`Wav2Vec2Config::is_group_norm`]; the `"layer"` arm (used by
  /// `large-960h-lv60-self`) is out of scope and rejected by
  /// [`Wav2Vec2Config::validate`].
  #[serde(default = "default_feat_extract_norm")]
  pub feat_extract_norm: String,
  /// Hidden (transformer feed-forward) activation. Dispatched on by
  /// [`Activation::resolve`]: `"gelu"` (the exact GELU, matching the
  /// reference), `"gelu_new"` / `"gelu_pytorch_tanh"` (tanh-approx GELU), or
  /// `"silu"` / `"swish"`. An unsupported name is rejected by
  /// [`Wav2Vec2Config::validate`].
  #[serde(default = "default_hidden_act")]
  pub hidden_act: String,
  /// Feature-encoder conv activation. Dispatched on by [`Activation::resolve`]
  /// exactly as [`Wav2Vec2Config::hidden_act`]; an unsupported name is rejected
  /// by [`Wav2Vec2Config::validate`].
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
  /// [`Wav2Vec2Config::validate`].
  #[serde(default)]
  pub add_adapter: bool,
  /// Per-attention-block adapter dimension. `base-960h`: absent / `null`. When
  /// set, HF adds a `Wav2Vec2AttnAdapterLayer` to every encoder layer whose
  /// output is **added** to the hidden states — an extra graph term this port
  /// does not wire — so a non-`null` value would load and run silently wrong.
  /// Modeled as `Option<i32>` (absent ⇒ `None`) and rejected unless `None` by
  /// [`Wav2Vec2Config::validate`].
  #[serde(default)]
  pub adapter_attn_dim: Option<i32>,
  /// CTC blank-token id. `base-960h`: `0`. Greedy CTC decoding drops exactly
  /// this id (the port hardcodes `CTC_BLANK = 0`), so a checkpoint declaring a
  /// different blank would collapse the per-frame argmax against the wrong
  /// token and decode silently wrong. Pinned to `0` by
  /// [`Wav2Vec2Config::validate`].
  #[serde(default = "default_pad_token_id")]
  pub pad_token_id: i32,
  /// Whether the feature projection applies a `LayerNorm` to the feature
  /// encoder's output before the linear projection. A **HuBERT** config field
  /// (`HubertConfig.feat_proj_layer_norm`, default `true`); the wav2vec2 config
  /// has no such field and always applies the LayerNorm. This port wires the
  /// `true` arm only — the feature projection unconditionally applies the
  /// LayerNorm — so a `false` value (HuBERT's no-LayerNorm projection arm) would
  /// load and feed the projection an un-normalized input through a graph that
  /// still normalizes: a silently wrong forward. Pinned to `true` by
  /// [`Wav2Vec2Config::validate`]; the no-LayerNorm arm is out of scope.
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
  /// [`Wav2Vec2Config::validate`]; the batch-norm arm is out of scope.
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

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Wav2Vec2Config {
  /// Parse a [`Wav2Vec2Config`] from an in-memory `config.json` string.
  ///
  /// Mirrors mlx-audio's `ModelConfig.from_dict` (a `json.load` restricted to
  /// the known keys). A malformed-JSON failure maps to [`Error::Parse`]; an
  /// unmodeled key is ignored (forward-compatible) and an absent key takes
  /// the `base-960h` default.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Wav2Vec2Config::from_json",
        "config JSON",
        e,
      ))
    })
  }

  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// `true` when `feat_extract_norm == "group"` — the only feature-encoder
  /// normalization scheme this port wires. The `"layer"` variant used by
  /// `large-960h-lv60-self` is out of scope (see the module docs).
  #[inline(always)]
  pub fn is_group_norm(&self) -> bool {
    self.feat_extract_norm == "group"
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
  /// [`Wav2Vec2Ctc::from_weights`], and eagerly in [`Wav2Vec2Ctc::load`] so a
  /// mismatch fails before the weights file is even read).
  ///
  /// Rejected (each a typed error, never a panic or a silent mis-load):
  /// - `model_type` not one of the supported family ids (`wav2vec2`,
  ///   `hubert`) ([`Error::UnknownEnumValue`]) — `wavlm` is **rejected** here
  ///   (its gated relative-position-bias attention is not wired this phase);
  /// - `feat_extract_norm != "group"` ([`Error::UnknownEnumValue`]) — the
  ///   `"layer"` feature-encoder arm is out of scope;
  /// - `hidden_act` / `feat_extract_activation` not a supported activation
  ///   ([`Error::UnknownEnumValue`], via [`Activation::resolve`]);
  /// - `add_adapter == true` / `adapter_attn_dim` set
  ///   ([`Error::InvariantViolation`]) — the post-encoder conv adapter stack
  ///   and the per-layer attention adapter are out of scope;
  /// - `feat_proj_layer_norm == false` / `conv_pos_batch_norm == true`
  ///   ([`Error::InvariantViolation`]) — the HuBERT-only no-LayerNorm
  ///   feature-projection arm and batch-norm positional-conv arm are out of
  ///   scope (the wired graph applies the projection LayerNorm and reconstructs
  ///   a weight-normalized positional conv); both HF defaults match the wired
  ///   graph, so default HuBERT and every wav2vec2 checkpoint pass;
  /// - `pad_token_id != 0` ([`Error::OutOfRange`]) — greedy CTC decoding drops
  ///   exactly id `0` (the hardcoded CTC blank), so a different declared blank
  ///   would collapse the argmax against the wrong token;
  /// - a non-positive `hidden_size` / `num_attention_heads` /
  ///   `intermediate_size` / `vocab_size` / `num_conv_pos_embeddings`
  ///   ([`Error::OutOfRange`]); `hidden_size` not divisible by
  ///   `num_attention_heads` or by `num_conv_pos_embedding_groups`
  ///   ([`Error::DivisibilityConstraint`]);
  /// - a non-positive `num_hidden_layers` / `num_feat_extract_layers`
  ///   ([`Error::OutOfRange`]) — each sizes an eager per-layer `Vec`, reserved
  ///   fallibly by the builder;
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
      "Wav2Vec2Config: model_type",
      self.model_type.as_str(),
      SUPPORTED_MODEL_TYPES,
    )?;
    // Feature-encoder normalization scheme: only the group-norm arm is wired
    // (the "layer" arm is out of scope for this phase).
    pin_str(
      "Wav2Vec2Config: feat_extract_norm",
      self.feat_extract_norm.as_str(),
      &["group"],
    )?;
    // Activations: resolve both names against the supported set. The resolved
    // values are recomputed at build time; resolving here makes an unsupported
    // activation fail fast (before any tensor) with the same typed error.
    Activation::resolve(&self.hidden_act, "Wav2Vec2Config: hidden_act")?;
    Activation::resolve(
      &self.feat_extract_activation,
      "Wav2Vec2Config: feat_extract_activation",
    )?;
    // The post-encoder convolutional adapter stack is out of scope: when
    // `true`, HF re-shapes the encoder output (to `output_hidden_size`) before
    // the CTC head, a graph this port does not build.
    pin_bool("Wav2Vec2Config: add_adapter", self.add_adapter, false)?;
    // The per-attention-block adapter is out of scope: when `adapter_attn_dim`
    // is set, HF adds a `Wav2Vec2AttnAdapterLayer` to every encoder layer whose
    // output is *added* to the hidden states. Only the absent (`None`) form is
    // supported.
    if self.adapter_attn_dim.is_some() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Wav2Vec2Config: adapter_attn_dim",
        "must be absent (null) — the per-layer attention adapter is not wired",
      )));
    }
    // CTC blank id: greedy decode hardcodes `CTC_BLANK = 0`, so a checkpoint
    // declaring a different blank would collapse the per-frame argmax against
    // the wrong token. Pin it to the wired value (`0`); a configurable blank is
    // out of scope.
    if self.pad_token_id != PAD_TOKEN_ID {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Wav2Vec2Config: pad_token_id",
        "must equal the hardcoded CTC blank id (0)",
        format_smolstr!("{} (expected {})", self.pad_token_id, PAD_TOKEN_ID),
      )));
    }
    // HuBERT-only feature-projection / positional-conv graph arms. These two
    // flags exist on `HubertConfig` (the wav2vec2 config has neither and always
    // takes the wired arm); both HF defaults match the wired graph, so a
    // default HuBERT (and every wav2vec2) checkpoint passes. Only the
    // non-default arm — not implemented this phase — is rejected.
    //
    // `feat_proj_layer_norm` (HuBERT default `true`): the wired
    // `FeatureProjection` unconditionally applies the LayerNorm, so the
    // no-LayerNorm (`false`) arm would feed the projection an un-normalized
    // input through a graph that still normalizes — a silently wrong forward.
    pin_bool(
      "Wav2Vec2Config: feat_proj_layer_norm",
      self.feat_proj_layer_norm,
      true,
    )?;
    // `conv_pos_batch_norm` (HuBERT default `false`): the wired
    // `PositionalConvEmbedding` reconstructs the fused kernel from the
    // weight-norm `weight_g` / `weight_v` pair, so the batch-norm (`true`) arm
    // selects a different module (a `BatchNorm1d` over a plain conv whose
    // checkpoint carries no `weight_g` / `weight_v`).
    pin_bool(
      "Wav2Vec2Config: conv_pos_batch_norm",
      self.conv_pos_batch_norm,
      false,
    )?;
    // Width dimensions: positivity + the divisibility the wired graph needs
    // (per-head split; grouped positional conv). These bound every width a
    // later step divides by or uses to size work.
    require_positive("Wav2Vec2Config: hidden_size", self.hidden_size)?;
    require_positive(
      "Wav2Vec2Config: num_attention_heads",
      self.num_attention_heads,
    )?;
    require_positive("Wav2Vec2Config: intermediate_size", self.intermediate_size)?;
    require_positive("Wav2Vec2Config: vocab_size", self.vocab_size)?;
    require_positive(
      "Wav2Vec2Config: num_conv_pos_embeddings",
      self.num_conv_pos_embeddings,
    )?;
    require_positive(
      "Wav2Vec2Config: num_conv_pos_embedding_groups",
      self.num_conv_pos_embedding_groups,
    )?;
    require_divisible(
      "Wav2Vec2Config: hidden_size",
      self.hidden_size,
      "Wav2Vec2Config: num_attention_heads",
      self.num_attention_heads,
    )?;
    require_divisible(
      "Wav2Vec2Config: hidden_size",
      self.hidden_size,
      "Wav2Vec2Config: num_conv_pos_embedding_groups",
      self.num_conv_pos_embedding_groups,
    )?;
    // Layer counts size eager per-layer `Vec`s: each must be positive (a zero
    // or negative count is malformed). The builder reserves the `Vec`s fallibly,
    // so a large positive count surfaces as a typed allocation error, not a cap.
    require_positive("Wav2Vec2Config: num_hidden_layers", self.num_hidden_layers)?;
    require_positive(
      "Wav2Vec2Config: num_feat_extract_layers",
      self.num_feat_extract_layers,
    )?;
    // `eps` shared by every LayerNorm and the L0 GroupNorm: must be a finite,
    // positive scalar (it varies across the family, so it is not pinned to a
    // magnitude). A non-finite value would drive a non-finite denominator.
    if !self.layer_norm_eps.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Wav2Vec2Config: layer_norm_eps",
        f64::from(self.layer_norm_eps),
      )));
    }
    if self.layer_norm_eps <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Wav2Vec2Config: layer_norm_eps",
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
    check_conv_array("Wav2Vec2Config: conv_dim", &self.conv_dim, n)?;
    check_conv_array("Wav2Vec2Config: conv_stride", &self.conv_stride, n)?;
    check_conv_array("Wav2Vec2Config: conv_kernel", &self.conv_kernel, n)?;
    Ok(())
  }

  /// Per-head dimension `hidden_size / num_attention_heads`.
  fn head_dim(&self) -> Result<i32> {
    require_positive(
      "Wav2Vec2Config: num_attention_heads",
      self.num_attention_heads,
    )?;
    require_divisible(
      "Wav2Vec2Config: hidden_size",
      self.hidden_size,
      "Wav2Vec2Config: num_attention_heads",
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

/// The element-wise activation a [`Wav2Vec2Config`] selects, resolved from an
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

/// `y = x @ wᵀ (+ bias)` — a private `nn.Linear` forward.
///
/// mlxrs ships no public `nn::Linear` (GAP by design); each transformer
/// projection composes this. HF stores a `Linear` weight as `(out, in)`, so
/// the matmul is `x @ wᵀ`. With a bias this is the fused
/// `addmm(bias, x, wᵀ, 1, 1)`; without, a plain `matmul(x, wᵀ)`.
#[cfg(feature = "wav2vec2")]
fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
  let wt = ops::shape::swapaxes(weight, 0, 1)?;
  match bias {
    Some(b) => ops::linalg_basic::addmm(b, x, &wt, 1.0, 1.0),
    None => ops::linalg_basic::matmul(x, &wt),
  }
}

// ───────────────────────── weight-fetch helper ─────────────────────────

/// Pull a weight by exact key from the (sanitized) checkpoint map, erroring
/// with the key if absent.
#[cfg(feature = "wav2vec2")]
fn take(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights
    .remove(key)
    .ok_or_else(|| Error::MissingKey(MissingKeyPayload::new("Wav2Vec2Ctc::from_weights", key)))
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
/// [`Wav2Vec2Config`]'s width / count / conv-stack fields and are non-negative
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

// ───────────────────────── feature encoder ─────────────────────────

/// A single feature-encoder convolution layer. Conv weight is channels-last
/// `(out, k, in)` (post-sanitize); the optional `bias` `(out,)` is present iff
/// `config.conv_bias`. The L0 `Wav2Vec2GroupNormConvLayer` additionally carries
/// an affine [`GroupNorm`] (`num_groups == dims`, pytorch-compatible); the
/// remaining `Wav2Vec2NoLayerNormConvLayer`s are conv → activation only.
#[cfg(feature = "wav2vec2")]
struct ConvLayer {
  weight: Array,
  /// `Some(bias)` iff `config.conv_bias` — `nn.Conv1d(bias=config.conv_bias)`.
  bias: Option<Array>,
  stride: i32,
  /// `Some` for L0 (`Wav2Vec2GroupNormConvLayer`), `None` for the
  /// `Wav2Vec2NoLayerNormConvLayer`s.
  group_norm: Option<GroupNorm>,
  /// The `feat_extract_activation` (resolved once at build).
  activation: Activation,
}

#[cfg(feature = "wav2vec2")]
impl ConvLayer {
  /// `conv(x.swapaxes(-2,-1)) (+ bias) → [group_norm] → swapaxes(-2,-1) → act`.
  ///
  /// Input/output are `(B, C, L)` (channels-second). MLX conv1d is
  /// channels-last, so the layer transposes to `(B, L, C)` around the conv
  /// (and around the GroupNorm, which normalizes over the last/feature axis),
  /// exactly as the reference's `hidden_states.swapaxes(-2, -1)` bracketing.
  /// The bias is added in the channels-last `(B, L', C_out)` layout (the
  /// `(out,)` bias broadcasts over the last axis), before the GroupNorm —
  /// matching `nn.Conv1d`'s fused bias followed by the layer norm.
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
    // GroupNorm (L0 only) runs in channels-last layout (feature = last axis).
    if let Some(gn) = &self.group_norm {
      h = gn.forward(&h)?;
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

/// `LayerNorm(512) → Linear(512 → 768)` over `(B, T', 512)`.
/// Ports `Wav2Vec2FeatureProjection` ([wav2vec.py:279-290][fp]).
///
/// [fp]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L279-L290
#[cfg(feature = "wav2vec2")]
struct FeatureProjection {
  layer_norm: LayerNorm,
  proj_weight: Array,
  proj_bias: Array,
}

#[cfg(feature = "wav2vec2")]
impl FeatureProjection {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let normed = self.layer_norm.forward(hidden_states)?;
    linear(&normed, &self.proj_weight, Some(&self.proj_bias))
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
  q_weight: Array,
  q_bias: Array,
  k_weight: Array,
  k_bias: Array,
  v_weight: Array,
  v_bias: Array,
  out_weight: Array,
  out_bias: Array,
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
    let q = linear(hidden_states, &self.q_weight, Some(&self.q_bias))?;
    let scale = scalar_like(self.scaling, &q)?;
    let q = q.multiply(&scale)?;
    let k = linear(hidden_states, &self.k_weight, Some(&self.k_bias))?;
    let v = linear(hidden_states, &self.v_weight, Some(&self.v_bias))?;

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
    linear(&attn, &self.out_weight, Some(&self.out_bias))
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
  intermediate_weight: Array,
  intermediate_bias: Array,
  output_weight: Array,
  output_bias: Array,
  /// The `hidden_act` (resolved once at build).
  activation: Activation,
}

#[cfg(feature = "wav2vec2")]
impl FeedForward {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
    let h = linear(
      hidden_states,
      &self.intermediate_weight,
      Some(&self.intermediate_bias),
    )?;
    let h = self.activation.forward(&h)?;
    linear(&h, &self.output_weight, Some(&self.output_bias))
  }
}

// ───────────────────────── encoder layer ─────────────────────────

/// The per-layer transformer block, common to both encoder arms (their weights
/// are identical; only the block ordering and the encoder-level `LayerNorm`
/// placement differ — see [`EncoderLayer::forward`] / [`EncoderLayer::forward_stable`]).
#[cfg(feature = "wav2vec2")]
struct EncoderLayer {
  attention: Attention,
  layer_norm: LayerNorm,
  feed_forward: FeedForward,
  final_layer_norm: LayerNorm,
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
  /// [sel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L489-L508
  fn forward_stable(&self, hidden_states: &Array) -> Result<Array> {
    let normed = self.layer_norm.forward(hidden_states)?;
    let attn = self.attention.forward(&normed)?;
    let h = hidden_states.add(&attn)?;
    let ff = self
      .feed_forward
      .forward(&self.final_layer_norm.forward(&h)?)?;
    h.add(&ff)
  }
}

// ───────────────────────── encoder ─────────────────────────

/// The transformer encoder. Mirrors mlx-audio's two distinct encoder classes —
/// the post-norm `Wav2Vec2Encoder` ([wav2vec.py:511-574][enc]) and the pre-norm
/// `Wav2Vec2EncoderStableLayerNorm` ([wav2vec.py:577-644][senc]) — as two arms
/// selected by `config.do_stable_layer_norm` (`Wav2Vec2Model.__init__`,
/// [wav2vec.py:663-666][sel]).
///
/// Both arms add the positional conv embedding to the hidden states and run the
/// same stack of [`EncoderLayer`]s; they differ in **where the encoder-level
/// `LayerNorm` sits** and in the per-layer block ordering:
/// - [`Encoder::PostNorm`]: `LayerNorm` is applied **before** the layer stack,
///   and each layer uses [`EncoderLayer::forward`] (post-norm);
/// - [`Encoder::StableLayerNorm`]: `LayerNorm` is applied **after** the layer
///   stack, and each layer uses [`EncoderLayer::forward_stable`] (pre-norm).
///
/// [enc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L511-L574
/// [senc]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L577-L644
/// [sel]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/wav2vec/wav2vec.py#L663-L666
#[cfg(feature = "wav2vec2")]
enum Encoder {
  /// `do_stable_layer_norm == false` — `Wav2Vec2Encoder`.
  PostNorm(EncoderInner),
  /// `do_stable_layer_norm == true` — `Wav2Vec2EncoderStableLayerNorm`.
  StableLayerNorm(EncoderInner),
}

/// The fields shared by both encoder arms (their weight layout is identical).
#[cfg(feature = "wav2vec2")]
struct EncoderInner {
  pos_conv_embed: PositionalConvEmbedding,
  layer_norm: LayerNorm,
  layers: Vec<EncoderLayer>,
}

#[cfg(feature = "wav2vec2")]
impl Encoder {
  fn forward(&self, hidden_states: &Array) -> Result<Array> {
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

// ───────────────────────── model ─────────────────────────

/// Wav2Vec2 CTC speech recognizer (the `wav2vec2` / `hubert` CTC family).
///
/// See the [module docs](self) for the architecture and public API.
#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
pub struct Wav2Vec2Ctc {
  config: Wav2Vec2Config,
  feature_encoder: FeatureEncoder,
  feature_projection: FeatureProjection,
  encoder: Encoder,
  /// CTC head `Linear(hidden → vocab)`.
  lm_head_weight: Array,
  lm_head_bias: Array,
  vocab: Vocab,
}

#[cfg(feature = "wav2vec2")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav2vec2")))]
impl Wav2Vec2Ctc {
  /// The fixed input sample rate (16 kHz mono) — mlx-audio's
  /// `Model.sample_rate`.
  pub const SAMPLE_RATE: u32 = 16_000;

  /// Conservative upper bound on the TOTAL input element count (batch x time)
  /// for the inherent [`Wav2Vec2Ctc::forward`] / [`Wav2Vec2Ctc::transcribe`]
  /// path: 60 s at [`Wav2Vec2Ctc::SAMPLE_RATE`] (960 000 samples) for the common
  /// mono single-utterance case. The inherent path has no STT-pipeline
  /// `max_audio_seconds` cap, so an over-large waveform — a long single sequence
  /// OR a large batch — would otherwise drive the O(N) convolutional feature
  /// maps and, after the ~320x feature-encoder downsampling, the transformer's
  /// quadratic self-attention without bound (a process-level OOM / DoS). Inputs
  /// whose total element count exceeds this are rejected up front with a
  /// recoverable [`Error::OutOfRange`]; process longer / larger audio in chunks.
  pub const MAX_INPUT_SAMPLES: usize = Self::SAMPLE_RATE as usize * 60;

  /// Reject an over-cap input waveform before any allocation (see
  /// [`Wav2Vec2Ctc::MAX_INPUT_SAMPLES`]). `waveform` is `(T,)` or `(B, T)`; the
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
        "Wav2Vec2Ctc inherent path: total input samples (batch x time)",
        "must not exceed MAX_INPUT_SAMPLES (60 s at 16 kHz); process longer or larger audio in chunks",
        total.to_string(),
      )));
    }
    Ok(())
  }

  /// Build a model from a parsed [`Wav2Vec2Config`], the **sanitized** weight
  /// map (run [`sanitize`] first), and an optional [`Vocab`] (for
  /// [`Wav2Vec2Ctc::transcribe`]).
  ///
  /// The config is gated by [`Wav2Vec2Config::validate`] first (so an
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
    config: Wav2Vec2Config,
    mut weights: HashMap<String, Array>,
    vocab: Vocab,
  ) -> Result<Self> {
    // Single config-validation gate: reject any unsupported / out-of-scope arm
    // and any malformed dimension BEFORE any tensor is built.
    config.validate()?;

    let feature_encoder = build_feature_encoder(&config, &mut weights)?;
    let feature_projection = build_feature_projection(&config, &mut weights)?;
    let encoder = build_encoder(&config, &mut weights)?;
    // CTC head Linear (out, in) = (vocab_size, hidden_size); bias (vocab_size,).
    // Pinning the exact shape is the key allocation guard: an oversized output
    // dim here would otherwise drive a huge logits tensor at forward time.
    let lm_head_weight = take_shaped(
      &mut weights,
      "lm_head.weight",
      "CTC head weight (vocab_size, hidden_size)",
      &[config.vocab_size, config.hidden_size],
    )?;
    let lm_head_bias = take_shaped(
      &mut weights,
      "lm_head.bias",
      "CTC head bias (vocab_size)",
      &[config.vocab_size],
    )?;

    Ok(Self {
      config,
      feature_encoder,
      feature_projection,
      encoder,
      lm_head_weight,
      lm_head_bias,
      vocab,
    })
  }

  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &Wav2Vec2Config {
    &self.config
  }

  /// The decoding vocabulary.
  #[inline(always)]
  pub fn vocab(&self) -> &Vocab {
    &self.vocab
  }

  /// Run the full forward pass: raw waveform `(B, T)` (or `(T,)` — promoted to
  /// `(1, T)`) → per-frame CTC logits `(B, T', vocab)`.
  ///
  /// This does **not** normalize the waveform (the reference's
  /// zero-mean-unit-variance step happens in `generate`); call
  /// [`Wav2Vec2Ctc::transcribe`] for the full normalize → forward → decode
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
    let hidden_states = self.encoder.forward(&hidden_states)?;
    // (B, T', vocab)
    linear(
      &hidden_states,
      &self.lm_head_weight,
      Some(&self.lm_head_bias),
    )
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
    if self.vocab.is_empty() {
      return Err(Error::InvariantViolation(InvariantViolationPayload::new(
        "Wav2Vec2Ctc::transcribe",
        "model was built without a vocabulary (use forward + ctc_greedy_collapse)",
      )));
    }
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
          "Wav2Vec2Ctc::transcribe: predictions must be rank-1 or rank-2",
          shape.len() as u32,
          shape.clone(),
        )));
      }
    };
    let all_ids = predictions.to_vec::<u32>()?;
    // First batch row (the reference decodes `decoded[0]`).
    let first_row = &all_ids[..seq_len.min(all_ids.len())];
    let tokens = ctc_greedy_collapse(first_row);
    Ok(self.vocab.tokens_to_text(&tokens).trim().to_string())
  }

  /// Load a model from a local on-disk directory — the convenience entry
  /// point mirroring mlx-audio's `stt.load` for this architecture.
  ///
  /// Resolves `path` via [`crate::audio::load::get_model_path`] (local-only;
  /// a Hub id is rejected per the project's no-network policy), reads and
  /// parses `config.json`, loads + [`sanitize`]s the single un-sharded
  /// `model.safetensors`, and reads the character `vocab.json` (so
  /// [`Wav2Vec2Ctc::transcribe`] works). `vocab.json` is optional — if absent
  /// the model still loads and [`Wav2Vec2Ctc::forward`] works, but
  /// `transcribe` then errors.
  ///
  /// Only the single-file `model.safetensors` layout is handled here; sharded
  /// checkpoints are out of scope (a missing file is a clear
  /// [`Error::MissingKey`]).
  pub fn load(path: &str) -> Result<Self> {
    let dir = crate::audio::load::get_model_path(path)?;
    let config_json = crate::audio::load::load_config(&dir)?;
    let config = Wav2Vec2Config::from_json(&config_json)?;
    // Reject an unsupported architecture arm before reading the (large)
    // weights file — `from_weights` re-checks, but failing here avoids the
    // safetensors read on a config this port cannot serve.
    config.validate()?;

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
      return Err(Error::MissingKey(MissingKeyPayload::new(
        "Wav2Vec2Ctc::load: model.safetensors not found (sharded checkpoints unsupported)",
        format_smolstr!("{}", weights_path.display()),
      )));
    }
    let raw = crate::io::load_safetensors(&weights_path)?;
    let weights = sanitize(raw)?;

    // vocab.json is optional; an absent file leaves an empty Vocab (forward
    // still works, transcribe then errors with a clear message). Reuse the
    // shared bounded reader so a hostile directory can't OOM the loader.
    let vocab_path = dir.join("vocab.json");
    let vocab = match crate::lm::load::read_bounded_config_file(&vocab_path, "wav2vec2 vocab.json")?
    {
      Some(body) => Vocab::from_json(&body)?,
      None => Vocab::default(),
    };

    Self::from_weights(config, weights, vocab)
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
/// `num_feat_extract_layers`-layer [`FeatureEncoder`] (the `"group"` arm: an L0
/// `Wav2Vec2GroupNormConvLayer` then `Wav2Vec2NoLayerNormConvLayer`s). Each
/// layer carries the resolved `feat_extract_activation` and, when
/// `config.conv_bias`, its `conv.bias`.
#[cfg(feature = "wav2vec2")]
fn build_feature_encoder(
  config: &Wav2Vec2Config,
  weights: &mut HashMap<String, Array>,
) -> Result<FeatureEncoder> {
  // `validate` (run by `from_weights` before any builder) already pinned
  // `num_feat_extract_layers` positive and the conv arrays to exactly that
  // length with positive entries; re-derive the count and re-check the arrays
  // cover `n` here defensively (the builder must never index past the end).
  let n = config.num_feat_extract_layers;
  if n <= 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "Wav2Vec2Config: num_feat_extract_layers",
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
      "Wav2Vec2Config: conv_dim/stride/kernel length",
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
    "Wav2Vec2Config: feat_extract_activation",
  )?;
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
    // L0 carries an affine pytorch-compatible GroupNorm with
    // num_groups == dims == conv_dim[0]; the rest have no norm.
    let group_norm = if i == 0 {
      let dims = config.conv_dim[0];
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
      Some(GroupNorm::with_affine(
        dims,
        dims,
        config.layer_norm_eps,
        Some((gn_weight, gn_bias)),
        true,
      )?)
    } else {
      None
    };
    conv_layers.push(ConvLayer {
      weight,
      bias,
      stride,
      group_norm,
      activation,
    });
  }
  Ok(FeatureEncoder { conv_layers })
}

/// Read `feature_projection.*` into the [`FeatureProjection`].
#[cfg(feature = "wav2vec2")]
fn build_feature_projection(
  config: &Wav2Vec2Config,
  weights: &mut HashMap<String, Array>,
) -> Result<FeatureProjection> {
  // The feature encoder's last conv width (conv_dim[-1]) is the projection's
  // input dim and the pre-norm's normalized dim.
  let conv_dim_last = *config.conv_dim.last().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "Wav2Vec2Config: conv_dim",
      "must be non-empty (the projection reads conv_dim[-1])",
    ))
  })?;
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
  let layer_norm = LayerNorm::new(Some(ln_weight), Some(ln_bias), config.layer_norm_eps);
  // HF Linear weight is (out, in) = (hidden_size, conv_dim[-1]); bias (hidden_size,).
  let proj_weight = take_shaped(
    weights,
    "feature_projection.projection.weight",
    "feature_projection projection weight (hidden_size, conv_dim[-1])",
    &[config.hidden_size, conv_dim_last],
  )?;
  let proj_bias = take_shaped(
    weights,
    "feature_projection.projection.bias",
    "feature_projection projection bias (hidden_size)",
    &[config.hidden_size],
  )?;
  Ok(FeatureProjection {
    layer_norm,
    proj_weight,
    proj_bias,
  })
}

/// Read `encoder.*` into the [`Encoder`].
#[cfg(feature = "wav2vec2")]
fn build_encoder(config: &Wav2Vec2Config, weights: &mut HashMap<String, Array>) -> Result<Encoder> {
  // Positional conv embedding: reconstruct the fused weight-normalized kernel
  // once, then a plain grouped conv at forward time. Post-sanitize MLX layout
  // (out, k, in/groups): weight_v is (hidden_size, num_conv_pos_embeddings,
  // hidden_size / num_conv_pos_embedding_groups); weight_g is the per-kernel
  // magnitude (1, num_conv_pos_embeddings, 1); bias is (hidden_size,). A
  // deviating kernel axis here would silently change the positional receptive
  // field, so pin every exact shape.
  let kernel = config.num_conv_pos_embeddings;
  require_divisible(
    "Wav2Vec2Config: hidden_size",
    config.hidden_size,
    "Wav2Vec2Config: num_conv_pos_embedding_groups",
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
    "Wav2Vec2Config: feat_extract_activation",
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
      "Wav2Vec2Config: num_hidden_layers",
      "must be non-negative",
      format_smolstr!("{num_layers}"),
    )));
  }
  let head_dim = config.head_dim()?;
  let scaling = (head_dim as f32).powf(-0.5);
  let activation = Activation::resolve(&config.hidden_act, "Wav2Vec2Config: hidden_act")?;
  // Per-layer expected shapes (derived from the validated config): every
  // attention projection is a square (hidden_size, hidden_size) Linear with a
  // (hidden_size,) bias; the LayerNorms are (hidden_size,); the feed-forward is
  // (intermediate_size, hidden_size) + (hidden_size, intermediate_size) Linears
  // with their respective biases.
  let hs = config.hidden_size;
  let inter = config.intermediate_size;
  let attn_proj = [hs, hs];
  let mut layers = Vec::new();
  reserve_or_error(&mut layers, "encoder layers", num_layers as usize)?;
  for i in 0..num_layers {
    let prefix = format!("encoder.layers.{i}");
    let attention = Attention {
      q_weight: take_shaped(
        weights,
        &format!("{prefix}.attention.q_proj.weight"),
        "attention q_proj weight (hidden_size, hidden_size)",
        &attn_proj,
      )?,
      q_bias: take_shaped(
        weights,
        &format!("{prefix}.attention.q_proj.bias"),
        "attention q_proj bias (hidden_size)",
        &[hs],
      )?,
      k_weight: take_shaped(
        weights,
        &format!("{prefix}.attention.k_proj.weight"),
        "attention k_proj weight (hidden_size, hidden_size)",
        &attn_proj,
      )?,
      k_bias: take_shaped(
        weights,
        &format!("{prefix}.attention.k_proj.bias"),
        "attention k_proj bias (hidden_size)",
        &[hs],
      )?,
      v_weight: take_shaped(
        weights,
        &format!("{prefix}.attention.v_proj.weight"),
        "attention v_proj weight (hidden_size, hidden_size)",
        &attn_proj,
      )?,
      v_bias: take_shaped(
        weights,
        &format!("{prefix}.attention.v_proj.bias"),
        "attention v_proj bias (hidden_size)",
        &[hs],
      )?,
      out_weight: take_shaped(
        weights,
        &format!("{prefix}.attention.out_proj.weight"),
        "attention out_proj weight (hidden_size, hidden_size)",
        &attn_proj,
      )?,
      out_bias: take_shaped(
        weights,
        &format!("{prefix}.attention.out_proj.bias"),
        "attention out_proj bias (hidden_size)",
        &[hs],
      )?,
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
    let feed_forward = FeedForward {
      intermediate_weight: take_shaped(
        weights,
        &format!("{prefix}.feed_forward.intermediate_dense.weight"),
        "feed_forward intermediate weight (intermediate_size, hidden_size)",
        &[inter, hs],
      )?,
      intermediate_bias: take_shaped(
        weights,
        &format!("{prefix}.feed_forward.intermediate_dense.bias"),
        "feed_forward intermediate bias (intermediate_size)",
        &[inter],
      )?,
      output_weight: take_shaped(
        weights,
        &format!("{prefix}.feed_forward.output_dense.weight"),
        "feed_forward output weight (hidden_size, intermediate_size)",
        &[hs, inter],
      )?,
      output_bias: take_shaped(
        weights,
        &format!("{prefix}.feed_forward.output_dense.bias"),
        "feed_forward output bias (hidden_size)",
        &[hs],
      )?,
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
    layers.push(EncoderLayer {
      attention,
      layer_norm: layer_norm_l,
      feed_forward,
      final_layer_norm,
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
    Encoder::StableLayerNorm(inner)
  } else {
    Encoder::PostNorm(inner)
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
