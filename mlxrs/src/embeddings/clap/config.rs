//! CLAP-HTSAT-unfused dual-tower configuration.
//!
//! Ports the `ClapAudioConfig` / `ClapTextConfig` / `ClapConfig` fields of HF
//! `transformers`' `modeling_clap.py` (`configuration_clap.py`), with defaults
//! pinned to the `laion/clap-htsat-unfused` checkpoint. The mel / spectrogram
//! parameters on [`ClapAudioConfig`] additionally match the proven `textclap`
//! front-end (`textclap/src/mel.rs:14-24`) and its committed
//! `golden_params.json` — the same constants the [`super::mel`] front-end uses.
//!
//! As elsewhere in the crate, parsing is forward-compatible: an unmodeled key
//! parses cleanly and an absent key falls back to its default
//! (`#[serde(default)]`, not `deny_unknown_fields`) — matching HF's
//! `PretrainedConfig.from_dict`, which retains only the known signature
//! parameters.
//!
//! Each config exposes a `validate()` that pins every architecture-defining
//! field onto the shared [`crate::model_validation`] toolkit **before** any
//! tensor is allocated, so a corrupt / hostile / wrong-architecture
//! `config.json` fails fast with a typed [`crate::Error`] instead of building
//! the wrong graph. The port hard-codes the `laion/clap-htsat-unfused`
//! contract — the [`super::mel`] front-end is fixed at 48 kHz / 64 mels, and
//! the HTSAT / RoBERTa / projection towers are wired for the fixed stage
//! depths, head counts, widths, activations, and norm epsilons — so a
//! deviating-but-positive field would silently run the wrong model (or drive an
//! oversized allocation) rather than fail. Every such fixed field is therefore
//! pinned to its exact checkpoint value; the structural per-head split is
//! additionally asserted to divide (defense in depth). This is a
//! correctness / fail-fast gate, not a DoS cap.

use crate::{
  error::{Error, OutOfRangePayload, ParsePayload, Result},
  model_validation::{pin_f64, pin_i32, pin_i32_slice, pin_str, require_divisible},
};

use smol_str::format_smolstr;

// ════════════════════════════ ClapAudioConfig ══════════════════════════════

/// CLAP HTSAT audio-tower configuration. Defaults match
/// `laion/clap-htsat-unfused`'s `audio_config` (HF `ClapAudioConfig`); the mel
/// / spectrogram fields additionally match `textclap/src/mel.rs` +
/// `golden_params.json`.
///
/// The HTSAT encoder is a Swin-Transformer V1 over the mel spectrogram:
/// hierarchical 4 stages with per-stage [`depths`](Self::depths) and
/// [`num_attention_heads`](Self::num_attention_heads), `patch_size`-strided
/// patch embedding, and a token-semantic pooling head producing the
/// `(batch, 768)` audio feature that feeds the audio projection.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClapAudioConfig {
  /// Architecture id (`"clap_audio_model"`).
  #[serde(default = "default_audio_model_type")]
  model_type: String,

  // ── HTSAT Swin architecture ──
  /// Patch-embed stem hidden dimension (the Swin stage-0 width; doubles per
  /// stage `96 → 192 → 384 → 768`). HF `ClapAudioConfig.patch_embeds_hidden_size`
  /// is `96` for the base checkpoint.
  #[serde(default = "default_patch_embeds_hidden_size")]
  pub patch_embeds_hidden_size: i32,
  /// Per-stage Swin block counts (`[2, 2, 6, 2]` for HTSAT-base). HF
  /// `ClapAudioConfig.depths`. HTSAT is a fixed 4-stage hierarchy, so this is a
  /// fixed-length `[i32; 4]`: a `config.json` whose `depths` is not exactly four
  /// elements is an invalid HTSAT config and is rejected at parse (serde's
  /// fixed-array deserializer fails on a length mismatch) before any allocation.
  #[serde(default = "default_depths")]
  pub depths: [i32; 4],
  /// Per-stage attention-head counts (`[4, 8, 16, 32]` for HTSAT-base). HF
  /// `ClapAudioConfig.num_attention_heads`. Fixed-length `[i32; 4]` for the same
  /// reason as [`depths`](Self::depths): a non-4-element array is an invalid
  /// HTSAT config, rejected at parse rather than allocated then validated.
  #[serde(default = "default_audio_num_attention_heads")]
  pub num_attention_heads: [i32; 4],
  /// Swin local-attention window side (`8`). HF `ClapAudioConfig.window_size`.
  #[serde(default = "default_window_size")]
  pub window_size: i32,
  /// Patch-embed Conv2d kernel / stride side (`4`). HF
  /// `ClapAudioConfig.patch_size`.
  #[serde(default = "default_patch_size")]
  pub patch_size: i32,
  /// Patch-embed input channel count (`1`, a single-channel mel image). HF
  /// `ClapAudioConfig.patch_embed_input_channels`.
  #[serde(default = "default_patch_embed_input_channels")]
  pub patch_embed_input_channels: i32,
  /// Swin MLP expansion ratio (`4.0`). HF `ClapAudioConfig.mlp_ratio`.
  #[serde(default = "default_mlp_ratio")]
  pub mlp_ratio: f64,
  /// The square mel-image side the `reshape_mel2img` step targets (`256`). HF
  /// `ClapAudioConfig.spec_size`.
  #[serde(default = "default_spec_size")]
  pub spec_size: i32,
  /// The time↔freq fold ratio `reshape_mel2img` uses (`4`). HF
  /// `ClapAudioConfig.freq_ratio`.
  #[serde(default = "default_freq_ratio")]
  pub freq_ratio: i32,
  /// The pooled audio feature width that feeds the audio projection (`768`,
  /// `= patch_embeds_hidden_size << (len(depths) - 1)`). HF
  /// `ClapAudioConfig.hidden_size`.
  #[serde(default = "default_audio_hidden_size")]
  pub hidden_size: i32,
  /// `eps` shared by the encoder `LayerNorm`s (`1e-5`). HF
  /// `ClapAudioConfig.layer_norm_eps`.
  #[serde(default = "default_audio_layer_norm_eps")]
  pub layer_norm_eps: f64,
  /// Whether the audio front-end uses the feature-fusion branch. The unfused
  /// checkpoint sets this `false`; the port only supports the unfused path.
  /// HF `ClapAudioConfig.enable_fusion`.
  #[serde(default)]
  pub enable_fusion: bool,

  // ── mel / spectrogram front-end (also pinned by textclap/src/mel.rs) ──
  /// Audio sample rate (`48_000` Hz). HF `ClapAudioConfig.sampling_rate` /
  /// `golden_params.json["sampling_rate"]`.
  #[serde(default = "default_sampling_rate")]
  pub sampling_rate: i32,
  /// Mel-bin count (`64`). HF `ClapAudioConfig.num_mel_bins` /
  /// `golden_params.json["feature_size"]`.
  #[serde(default = "default_num_mel_bins")]
  pub num_mel_bins: i32,
}

// ════════════════════════════ ClapTextConfig ═══════════════════════════════

/// CLAP RoBERTa text-tower configuration. Defaults match
/// `laion/clap-htsat-unfused`'s `text_config` (HF `ClapTextConfig`, a RoBERTa
/// encoder).
///
/// `ClapTextModel` is a standard RoBERTa (BERT-family, post-norm) encoder; the
/// CLAP text path takes the first token (`<s>`, position 0) of the last hidden
/// state as the sequence representation that feeds the text projection.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClapTextConfig {
  /// Architecture id (`"clap_text_model"`).
  #[serde(default = "default_text_model_type")]
  model_type: String,
  /// Token-embedding table size / the RoBERTa vocabulary (`50265`). HF
  /// `ClapTextConfig.vocab_size`.
  #[serde(default = "default_text_vocab_size")]
  pub vocab_size: i32,
  /// Transformer hidden / embedding dimension (`768`). HF
  /// `ClapTextConfig.hidden_size`.
  #[serde(default = "default_text_hidden_size")]
  pub hidden_size: i32,
  /// Number of transformer encoder layers (`12`). HF
  /// `ClapTextConfig.num_hidden_layers`.
  #[serde(default = "default_text_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// Number of attention heads (`12`). HF
  /// `ClapTextConfig.num_attention_heads`.
  #[serde(default = "default_text_num_attention_heads")]
  pub num_attention_heads: i32,
  /// Feed-forward intermediate dimension (`3072`). HF
  /// `ClapTextConfig.intermediate_size`.
  #[serde(default = "default_text_intermediate_size")]
  pub intermediate_size: i32,
  /// Maximum position-embedding length (`514` — RoBERTa reserves the first two
  /// for the `padding_idx` offset). HF `ClapTextConfig.max_position_embeddings`.
  #[serde(default = "default_text_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// Token-type-embedding table size (`1`; token_type_ids are all-zero). HF
  /// `ClapTextConfig.type_vocab_size`.
  #[serde(default = "default_text_type_vocab_size")]
  pub type_vocab_size: i32,
  /// Padding token id (`1`). RoBERTa offsets real positions by `pad_token_id +
  /// 1` (`create_position_ids_from_input_ids`). HF
  /// `ClapTextConfig.pad_token_id`.
  #[serde(default = "default_text_pad_token_id")]
  pub pad_token_id: i32,
  /// `eps` shared by every `LayerNorm` (`1e-12`, the RoBERTa default). HF
  /// `ClapTextConfig.layer_norm_eps`.
  #[serde(default = "default_text_layer_norm_eps")]
  pub layer_norm_eps: f64,
  /// Hidden activation (`"gelu"`, exact). HF `ClapTextConfig.hidden_act`.
  #[serde(default = "default_text_hidden_act")]
  hidden_act: String,
}

// ══════════════════════════════ ClapConfig ═════════════════════════════════

/// CLAP top-level configuration: the two tower configs plus the shared
/// projection width. Ports HF `ClapConfig` (`audio_config` + `text_config` +
/// `projection_dim`).
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClapConfig {
  /// Audio-tower (HTSAT) config (`audio_config`).
  #[serde(default)]
  pub audio_config: ClapAudioConfig,
  /// Text-tower (RoBERTa) config (`text_config`).
  #[serde(default)]
  pub text_config: ClapTextConfig,
  /// Top-level architecture id (`"clap"`).
  #[serde(default = "default_model_type")]
  model_type: String,
  /// Shared contrastive-projection output width (`512`). Each
  /// `ClapProjectionLayer` is `Linear(hidden_size → projection_dim)` → ReLU →
  /// `Linear(projection_dim → projection_dim)`, so this is both the projection
  /// output and its MLP intermediate width. HF `ClapConfig.projection_dim`.
  #[serde(default = "default_projection_dim")]
  pub projection_dim: i32,
  /// Projection-input hidden width (`768`). The top-level HF `ClapConfig`
  /// serializes `hidden_size = text_config.hidden_size` (see HF
  /// `ClapConfig.__init__`), and each `ClapProjectionLayer` reads its
  /// `linear1` in-dim from this tower hidden. Modeled under the real
  /// `hidden_size` key (not silently dropped) and pinned to the tower hidden.
  #[serde(rename = "hidden_size", default = "default_clap_hidden_size")]
  pub hidden_size: i32,
  /// Projection-MLP activation (`"relu"`, exact). The `ClapProjectionLayer`
  /// activation between `linear1` and `linear2`; CLAP uses ReLU (NOT the
  /// towers' GELU). HF `ClapConfig.projection_hidden_act`.
  #[serde(default = "default_projection_hidden_act")]
  projection_hidden_act: String,
}

// ── defaults (single source of truth; the `defaults_match_*` tests pin these
//    against the named architecture constants) ──

#[cfg(feature = "clap")]
fn default_model_type() -> String {
  "clap".to_string()
}
#[cfg(feature = "clap")]
fn default_audio_model_type() -> String {
  "clap_audio_model".to_string()
}
#[cfg(feature = "clap")]
fn default_text_model_type() -> String {
  "clap_text_model".to_string()
}
#[cfg(feature = "clap")]
fn default_patch_embeds_hidden_size() -> i32 {
  96
}
#[cfg(feature = "clap")]
fn default_depths() -> [i32; 4] {
  [2, 2, 6, 2]
}
#[cfg(feature = "clap")]
fn default_audio_num_attention_heads() -> [i32; 4] {
  [4, 8, 16, 32]
}
#[cfg(feature = "clap")]
fn default_window_size() -> i32 {
  8
}
#[cfg(feature = "clap")]
fn default_patch_size() -> i32 {
  4
}
#[cfg(feature = "clap")]
fn default_patch_embed_input_channels() -> i32 {
  1
}
#[cfg(feature = "clap")]
fn default_mlp_ratio() -> f64 {
  4.0
}
#[cfg(feature = "clap")]
fn default_spec_size() -> i32 {
  256
}
#[cfg(feature = "clap")]
fn default_freq_ratio() -> i32 {
  4
}
#[cfg(feature = "clap")]
fn default_audio_hidden_size() -> i32 {
  768
}
#[cfg(feature = "clap")]
fn default_audio_layer_norm_eps() -> f64 {
  1e-5
}
#[cfg(feature = "clap")]
fn default_sampling_rate() -> i32 {
  48_000
}
#[cfg(feature = "clap")]
fn default_num_mel_bins() -> i32 {
  64
}
#[cfg(feature = "clap")]
fn default_text_vocab_size() -> i32 {
  50265
}
#[cfg(feature = "clap")]
fn default_text_hidden_size() -> i32 {
  768
}
#[cfg(feature = "clap")]
fn default_text_num_hidden_layers() -> i32 {
  12
}
#[cfg(feature = "clap")]
fn default_text_num_attention_heads() -> i32 {
  12
}
#[cfg(feature = "clap")]
fn default_text_intermediate_size() -> i32 {
  3072
}
#[cfg(feature = "clap")]
fn default_text_max_position_embeddings() -> i32 {
  514
}
#[cfg(feature = "clap")]
fn default_text_type_vocab_size() -> i32 {
  1
}
#[cfg(feature = "clap")]
fn default_text_pad_token_id() -> i32 {
  1
}
#[cfg(feature = "clap")]
fn default_text_layer_norm_eps() -> f64 {
  1e-12
}
#[cfg(feature = "clap")]
fn default_text_hidden_act() -> String {
  "gelu".to_string()
}
#[cfg(feature = "clap")]
fn default_projection_dim() -> i32 {
  512
}
#[cfg(feature = "clap")]
fn default_clap_hidden_size() -> i32 {
  768
}
#[cfg(feature = "clap")]
fn default_projection_hidden_act() -> String {
  "relu".to_string()
}

#[cfg(feature = "clap")]
impl Default for ClapAudioConfig {
  fn default() -> Self {
    Self {
      model_type: default_audio_model_type(),
      patch_embeds_hidden_size: default_patch_embeds_hidden_size(),
      depths: default_depths(),
      num_attention_heads: default_audio_num_attention_heads(),
      window_size: default_window_size(),
      patch_size: default_patch_size(),
      patch_embed_input_channels: default_patch_embed_input_channels(),
      mlp_ratio: default_mlp_ratio(),
      spec_size: default_spec_size(),
      freq_ratio: default_freq_ratio(),
      hidden_size: default_audio_hidden_size(),
      layer_norm_eps: default_audio_layer_norm_eps(),
      enable_fusion: false,
      sampling_rate: default_sampling_rate(),
      num_mel_bins: default_num_mel_bins(),
    }
  }
}

#[cfg(feature = "clap")]
impl Default for ClapTextConfig {
  fn default() -> Self {
    Self {
      model_type: default_text_model_type(),
      vocab_size: default_text_vocab_size(),
      hidden_size: default_text_hidden_size(),
      num_hidden_layers: default_text_num_hidden_layers(),
      num_attention_heads: default_text_num_attention_heads(),
      intermediate_size: default_text_intermediate_size(),
      max_position_embeddings: default_text_max_position_embeddings(),
      type_vocab_size: default_text_type_vocab_size(),
      pad_token_id: default_text_pad_token_id(),
      layer_norm_eps: default_text_layer_norm_eps(),
      hidden_act: default_text_hidden_act(),
    }
  }
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl ClapAudioConfig {
  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Reject any config that deviates from the `laion/clap-htsat-unfused`
  /// audio contract with a typed error before any tensor is built.
  ///
  /// The HTSAT tower and the [`super::mel`] front-end are hard-wired for the
  /// fixed unfused checkpoint, so every architecture-defining field is pinned
  /// to its exact checkpoint value: `model_type`; the per-stage
  /// `depths` / `num_attention_heads` lists; the stem / window / patch / grid
  /// dimensions; the pooled `hidden_size`; the Swin `mlp_ratio` and
  /// `layer_norm_eps`; and the mel front-end `sampling_rate` (`48_000`, pinned
  /// to [`super::mel::SAMPLE_RATE`]) and `num_mel_bins` (`64`, pinned to
  /// [`super::mel::N_MELS`]). `enable_fusion` must be `false` (the port only
  /// implements the unfused single-window path). The deepest-stage per-head
  /// split is additionally asserted to divide (defense in depth — the pins
  /// already fix both operands).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "ClapAudioConfig: model_type",
      self.model_type.as_str(),
      &["clap_audio_model"],
    )?;
    // The unfused checkpoint has no feature-fusion branch; the port only
    // implements the unfused single-window path, so reject `enable_fusion`.
    if self.enable_fusion {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "ClapAudioConfig: enable_fusion",
        "must be false (only the unfused single-window path is supported)",
        format_smolstr!("{}", self.enable_fusion),
      )));
    }
    // Pin the per-stage layout: depths and head counts define the Swin stack
    // and must match the HTSAT-base architecture the port implements.
    pin_i32_slice("ClapAudioConfig: depths", &self.depths, &[2, 2, 6, 2])?;
    pin_i32_slice(
      "ClapAudioConfig: num_attention_heads",
      &self.num_attention_heads,
      &[4, 8, 16, 32],
    )?;
    // Pin every fixed scalar dimension to its checkpoint value. The HTSAT
    // patch-embed / Swin stack and the mel front-end hard-code each of these,
    // so a deviating-but-positive value would silently build / run the wrong
    // graph (or, for the mel sizes, mismatch the front-end's fixed
    // `(1, 1, 1001, 64)` output) instead of failing fast.
    for (name, actual, expected) in [
      (
        "ClapAudioConfig: patch_embeds_hidden_size",
        self.patch_embeds_hidden_size,
        96,
      ),
      ("ClapAudioConfig: window_size", self.window_size, 8),
      ("ClapAudioConfig: patch_size", self.patch_size, 4),
      (
        "ClapAudioConfig: patch_embed_input_channels",
        self.patch_embed_input_channels,
        1,
      ),
      ("ClapAudioConfig: spec_size", self.spec_size, 256),
      ("ClapAudioConfig: freq_ratio", self.freq_ratio, 4),
      ("ClapAudioConfig: hidden_size", self.hidden_size, 768),
      (
        "ClapAudioConfig: sampling_rate",
        self.sampling_rate,
        super::mel::SAMPLE_RATE as i32,
      ),
      (
        "ClapAudioConfig: num_mel_bins",
        self.num_mel_bins,
        super::mel::N_MELS as i32,
      ),
    ] {
      pin_i32(name, actual, expected)?;
    }
    // Pin the float fields the Swin stack hard-codes (the MLP expansion ratio
    // and the encoder LayerNorm epsilon).
    pin_f64("ClapAudioConfig: mlp_ratio", self.mlp_ratio, 4.0)?;
    pin_f64("ClapAudioConfig: layer_norm_eps", self.layer_norm_eps, 1e-5)?;
    // The deepest Swin stage (index 3 of the fixed 4-stage hierarchy) runs
    // window attention with the last head count; the pooled `hidden_size` must
    // split evenly across those heads. Both operands are pinned above, so this
    // is a structural defense-in-depth assertion.
    require_divisible(
      "ClapAudioConfig: hidden_size",
      self.hidden_size,
      "num_attention_heads[last]",
      self.num_attention_heads[3],
    )?;
    Ok(())
  }
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl ClapTextConfig {
  /// Architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Hidden activation (`config.json` `hidden_act`; `"gelu"`, exact).
  #[inline(always)]
  pub fn hidden_act(&self) -> &str {
    &self.hidden_act
  }

  /// Reject any config that deviates from the `laion/clap-htsat-unfused`
  /// text (RoBERTa) contract with a typed error before any tensor is built.
  ///
  /// The RoBERTa encoder is wired for the fixed checkpoint, so every
  /// architecture-defining field is pinned to its exact value: `model_type`;
  /// `vocab_size`, `hidden_size`, the layer / head counts, `intermediate_size`,
  /// `max_position_embeddings`, `type_vocab_size`, the `pad_token_id`
  /// position-offset, `layer_norm_eps`, and the `hidden_act` GELU. The per-head
  /// split (`hidden_size` divisible by `num_attention_heads`) is additionally
  /// asserted (defense in depth — both operands are pinned above).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "ClapTextConfig: model_type",
      self.model_type.as_str(),
      &["clap_text_model"],
    )?;
    // The RoBERTa embeddings / encoder hard-code each of these widths, counts,
    // and table sizes; a deviating-but-positive value would silently build /
    // run the wrong graph.
    for (name, actual, expected) in [
      ("ClapTextConfig: vocab_size", self.vocab_size, 50265),
      ("ClapTextConfig: hidden_size", self.hidden_size, 768),
      (
        "ClapTextConfig: num_hidden_layers",
        self.num_hidden_layers,
        12,
      ),
      (
        "ClapTextConfig: num_attention_heads",
        self.num_attention_heads,
        12,
      ),
      (
        "ClapTextConfig: intermediate_size",
        self.intermediate_size,
        3072,
      ),
      (
        "ClapTextConfig: max_position_embeddings",
        self.max_position_embeddings,
        514,
      ),
      ("ClapTextConfig: type_vocab_size", self.type_vocab_size, 1),
      // `pad_token_id` (`1`) drives the RoBERTa position-id offset
      // (`pad_id + 1 + cumsum(non_pad)`); a different value shifts every
      // position embedding.
      ("ClapTextConfig: pad_token_id", self.pad_token_id, 1),
    ] {
      pin_i32(name, actual, expected)?;
    }
    // The shared LayerNorm epsilon is fixed for the checkpoint (`1e-12`, the
    // RoBERTa default — distinct from the audio tower's `1e-5`).
    pin_f64("ClapTextConfig: layer_norm_eps", self.layer_norm_eps, 1e-12)?;
    // The encoder + FFN use exact GELU; pin the activation id.
    pin_str(
      "ClapTextConfig: hidden_act",
      self.hidden_act.as_str(),
      &["gelu"],
    )?;
    require_divisible(
      "ClapTextConfig: hidden_size",
      self.hidden_size,
      "num_attention_heads",
      self.num_attention_heads,
    )?;
    Ok(())
  }
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl ClapConfig {
  /// Parse a [`ClapConfig`] from an in-memory `config.json` string. A
  /// malformed-JSON failure maps to [`Error::Parse`]; absent keys take their
  /// checkpoint defaults; unmodeled keys are ignored.
  pub fn from_json(json: &str) -> Result<Self> {
    serde_json::from_str(json)
      .map_err(|e| Error::Parse(ParsePayload::new("ClapConfig::from_json", "config JSON", e)))
  }

  /// Top-level architecture id (`config.json` `model_type`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// Projection-MLP activation (`config.json` `projection_hidden_act`;
  /// `"relu"`, exact). The `ClapProjectionLayer` activation between `linear1`
  /// and `linear2`.
  #[inline(always)]
  pub fn projection_hidden_act(&self) -> &str {
    &self.projection_hidden_act
  }

  /// Validate both towers and the top-level fields.
  ///
  /// Pins the top-level `model_type` to `"clap"`, the shared `projection_dim`
  /// to the fixed `512`-wide contrastive projection, the projection-input
  /// `hidden_size` to the `768`-wide tower hidden, and the
  /// `projection_hidden_act` to ReLU (the projection MLPs are wired for these
  /// widths and activation), then validates each tower config (see
  /// [`ClapAudioConfig::validate`] / [`ClapTextConfig::validate`]).
  pub fn validate(&self) -> Result<()> {
    pin_str(
      "ClapConfig: model_type",
      self.model_type.as_str(),
      &["clap"],
    )?;
    // The contrastive projection is a fixed MLP for the unfused checkpoint:
    // `Linear(hidden_size=768 → projection_dim=512)` → ReLU →
    // `Linear(512 → 512)`. Pin the output width, the projection-input hidden
    // (the top-level `hidden_size`, which HF sets to `text_config.hidden_size`),
    // and the ReLU activation. The serialized `hidden_size` / `projection_dim`
    // / `projection_hidden_act` are each read (not silently dropped) so a
    // deviating value fails fast here instead of mismatching the projection
    // weight shapes / running the wrong activation later.
    pin_i32("ClapConfig: projection_dim", self.projection_dim, 512)?;
    pin_i32("ClapConfig: hidden_size", self.hidden_size, 768)?;
    pin_str(
      "ClapConfig: projection_hidden_act",
      self.projection_hidden_act.as_str(),
      &["relu"],
    )?;
    self.audio_config.validate()?;
    self.text_config.validate()?;
    Ok(())
  }
}

#[cfg(test)]
mod tests;
