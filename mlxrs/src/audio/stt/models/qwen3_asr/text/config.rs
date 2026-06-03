//! Qwen3-ASR text-decoder configuration.
//!
//! Mirrors mlx-audio's `qwen3_asr.config.TextConfig` (the `text_config` block of
//! the Qwen3-ASR / Qwen3-ForcedAligner `config.json`). Structurally a Qwen3
//! decoder config, but — unlike the dense [`Qwen3Config`](crate::lm::models::qwen3::Qwen3Config)
//! — it carries the released checkpoints' **non-null MRoPE `rope_scaling`**
//! (`mrope_section` + the interleaved flag). The dense Qwen3 config rejects a
//! non-null `rope_scaling`; this one parses and validates it, because the ASR
//! text decoder ([`Qwen3AsrTextModel`](super::Qwen3AsrTextModel)) applies
//! multimodal rotary position embeddings.
//!
//! Parsed via [`Qwen3AsrTextConfig::from_json`]; like the reference's
//! `from_dict`/`inspect.signature` filter, unmodeled keys are ignored (serde
//! does not `deny_unknown_fields`) and absent keys take the reference default.
//! The parsed fields are then [`validate`]d so a malformed config (zero /
//! negative / non-divisible / oversized dimension, or a malformed MRoPE section)
//! is a recoverable error here rather than a panic in the forward pass.
//!
//! Only the default base-theta MRoPE rotary is implemented; the decoder always
//! builds plain `base^(-2d/dim)` inverse frequencies with unit attention
//! scaling. A `rope_scaling` whose `rope_type` / `type` names a different
//! RoPE-init formula (`linear` / `llama3` / `yarn` / `longrope` / …) would
//! require a path this port does not have, so it is rejected with a typed error
//! rather than loaded and silently mis-rotated.
//!
//! [`validate`]: Qwen3AsrTextConfig::validate

use smol_str::format_smolstr;

use crate::{
  error::{
    Error, NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, Result, UnknownEnumValuePayload,
  },
  model_validation::{
    checked_mul, require_divisible, require_even, require_in_range, require_positive,
  },
};

/// Inclusive upper bound for every *width*-like config field. `2^24` is far
/// above any real Qwen3-ASR text checkpoint yet small enough that the
/// downstream `i32` shape arithmetic cannot overflow `i32`.
const MAX_CONFIG_DIM: i32 = 1 << 24;

/// The number of MRoPE position axes (temporal, height, width) — Qwen2-VL /
/// Qwen3-Omni-family multimodal RoPE always splits the rotary half-dimension
/// across exactly three sections.
pub(super) const MROPE_AXES: usize = 3;

/// The parsed, validated MRoPE configuration extracted from `rope_scaling`.
///
/// The released Qwen3-ASR `text_config.rope_scaling` carries a `mrope_section`
/// (a 3-tuple summing to `head_dim / 2`) and an `interleaved` /
/// `mrope_interleaved` flag selecting the Qwen3-Omni interleaved frequency
/// layout (vs the Qwen2.5-VL chunked layout). Both are surfaced as typed fields
/// so the decoder applies the correct multimodal rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MRopeConfig {
  /// `(temporal, height, width)` rotary half-dimension split. Sums to
  /// `head_dim / 2`.
  pub section: [i32; MROPE_AXES],
  /// Whether the interleaved frequency layout (`[THTHWHTHW...TT]`, Qwen3-Omni)
  /// is used. `false` selects the chunked layout (`[TTT...HHH...WWW]`,
  /// Qwen2.5-VL).
  pub interleaved: bool,
}

/// The set of `rope_scaling` RoPE types this port implements: the default
/// multimodal layout (`mrope_section`, plain base-theta inverse frequencies,
/// unit attention scaling). The HF / mlx-lm `initialize_rope` dispatch treats
/// both `"default"` and `"mrope"` as this same base-theta path; the remaining
/// types (`"linear"`, `"llama3"`, `"yarn"`, `"longrope"`, `"proportional"`, …)
/// select a different inverse-frequency / attention-scaling formula this decoder
/// does not build, so a config naming one of them is rejected rather than
/// silently mis-rotated.
const SUPPORTED_ROPE_TYPES: &[&str] = &["default", "mrope"];

/// The default RoPE type when `rope_scaling` carries neither `type` nor
/// `rope_type` — matches the reference `initialize_rope`, which falls back to
/// `"default"`.
const DEFAULT_ROPE_TYPE: &str = "default";

/// The raw `rope_scaling` object as it appears in the checkpoint JSON — only the
/// keys this port acts on are modeled (the RoPE type, the MRoPE section, and the
/// interleave flag); the rest (`original_max_*`, scaling factors for unimplemented
/// types, …) are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
struct RawRopeScaling {
  /// `rope_type` — the HF key naming the RoPE-init formula. Only the
  /// base-theta MRoPE path is implemented; see [`SUPPORTED_ROPE_TYPES`].
  #[serde(default)]
  rope_type: Option<String>,
  /// `type` — the alias the Qwen VL / Omni configs use for the same field. The
  /// reference `initialize_rope` reads `type` first, then `rope_type`.
  #[serde(rename = "type", default)]
  type_alias: Option<String>,
  /// The `(temporal, height, width)` rotary half-dimension split. Required for
  /// a non-null `rope_scaling`.
  #[serde(default)]
  mrope_section: Option<Vec<i32>>,
  /// `interleaved` (Qwen3-Omni) — the primary key.
  #[serde(default)]
  interleaved: Option<bool>,
  /// `mrope_interleaved` — the HF alias some checkpoints use.
  #[serde(default)]
  mrope_interleaved: Option<bool>,
}

impl RawRopeScaling {
  /// The effective RoPE type, resolved like the reference `initialize_rope`:
  /// `type` takes precedence over `rope_type`, falling back to
  /// [`DEFAULT_ROPE_TYPE`] when neither is present.
  fn rope_type(&self) -> &str {
    self
      .type_alias
      .as_deref()
      .or(self.rope_type.as_deref())
      .unwrap_or(DEFAULT_ROPE_TYPE)
  }
}

/// Qwen3-ASR text-decoder configuration — a serde-parsed mirror of mlx-audio's
/// `qwen3_asr.config.TextConfig`.
///
/// Carries the same dimensions / head counts / `rope_theta` / `head_dim` /
/// `tie_word_embeddings` as a dense Qwen3 config, plus the non-null MRoPE
/// `rope_scaling` the released checkpoints set.
#[derive(Debug, Clone, serde::Deserialize)]
#[non_exhaustive]
pub struct Qwen3AsrTextConfig {
  /// Architecture id (`"qwen3"`).
  #[serde(default = "default_model_type")]
  pub model_type: String,
  /// Hidden / embedding dimension.
  #[serde(default = "default_hidden_size")]
  pub hidden_size: i32,
  /// Number of decoder layers.
  #[serde(default = "default_num_hidden_layers")]
  pub num_hidden_layers: i32,
  /// MLP feed-forward (gate/up) width.
  #[serde(default = "default_intermediate_size")]
  pub intermediate_size: i32,
  /// Number of attention (query) heads.
  #[serde(default = "default_num_attention_heads")]
  pub num_attention_heads: i32,
  /// RMSNorm variance floor.
  #[serde(default = "default_rms_norm_eps")]
  pub rms_norm_eps: f32,
  /// Vocabulary size (token-embedding rows).
  #[serde(default = "default_vocab_size")]
  pub vocab_size: i32,
  /// Number of key/value heads (GQA).
  #[serde(default = "default_num_key_value_heads")]
  pub num_key_value_heads: i32,
  /// Maximum positional context (carried; not used by the forward pass).
  #[serde(default = "default_max_position_embeddings")]
  pub max_position_embeddings: i32,
  /// RoPE base frequency (`theta`).
  #[serde(default = "default_rope_theta")]
  pub rope_theta: f32,
  /// Per-head dimension (carried explicitly, not derived from `hidden_size /
  /// num_attention_heads`).
  #[serde(default = "default_head_dim")]
  pub head_dim: i32,
  /// Whether the output projection reuses the input embedding table.
  #[serde(default = "default_tie_word_embeddings")]
  pub tie_word_embeddings: bool,
  /// Raw MRoPE scaling object (opaque until [`validate`](Self::validate) parses
  /// it into [`mrope`](Self::mrope)). The released Qwen3-ASR checkpoints set
  /// this to a non-null `{mrope_section, interleaved}` object.
  #[serde(default)]
  rope_scaling: Option<serde_json::Value>,
}

fn default_model_type() -> String {
  "qwen3".to_string()
}
fn default_hidden_size() -> i32 {
  2048
}
fn default_num_hidden_layers() -> i32 {
  28
}
fn default_intermediate_size() -> i32 {
  6144
}
fn default_num_attention_heads() -> i32 {
  16
}
fn default_rms_norm_eps() -> f32 {
  1e-6
}
fn default_vocab_size() -> i32 {
  151936
}
fn default_num_key_value_heads() -> i32 {
  8
}
fn default_max_position_embeddings() -> i32 {
  65536
}
fn default_rope_theta() -> f32 {
  1_000_000.0
}
fn default_head_dim() -> i32 {
  128
}
fn default_tie_word_embeddings() -> bool {
  true
}

impl Default for Qwen3AsrTextConfig {
  /// The reference `TextConfig` defaults (the same per-field defaults serde
  /// applies to an absent key). `rope_scaling` defaults to `None`, which
  /// [`validate`](Self::validate) treats as the standard-RoPE
  /// (full-section, non-interleaved) fallback. Known-valid.
  fn default() -> Self {
    Self {
      model_type: default_model_type(),
      hidden_size: default_hidden_size(),
      num_hidden_layers: default_num_hidden_layers(),
      intermediate_size: default_intermediate_size(),
      num_attention_heads: default_num_attention_heads(),
      rms_norm_eps: default_rms_norm_eps(),
      vocab_size: default_vocab_size(),
      num_key_value_heads: default_num_key_value_heads(),
      max_position_embeddings: default_max_position_embeddings(),
      rope_theta: default_rope_theta(),
      head_dim: default_head_dim(),
      tie_word_embeddings: default_tie_word_embeddings(),
      rope_scaling: None,
    }
  }
}

impl Qwen3AsrTextConfig {
  /// Parse a [`Qwen3AsrTextConfig`] from a `text_config` JSON string.
  ///
  /// A serde failure (malformed JSON) maps to [`Error::Parse`]; missing keys
  /// fall back to the reference defaults. The parsed fields are then
  /// [`validate`]d.
  ///
  /// [`validate`]: Qwen3AsrTextConfig::validate
  pub fn from_json(json: &str) -> Result<Qwen3AsrTextConfig> {
    let cfg: Qwen3AsrTextConfig = serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Qwen3AsrTextConfig::from_json",
        "Qwen3-ASR text config JSON",
        e,
      ))
    })?;
    cfg.validate()?;
    Ok(cfg)
  }

  /// The validated MRoPE configuration: `head_dim / 2` split across the three
  /// axes, with the interleaved flag.
  ///
  /// When `rope_scaling` is absent / `null`, the standard-RoPE fallback puts the
  /// entire rotary half-dimension on the temporal axis (`[head_dim/2, 0, 0]`,
  /// non-interleaved) — for the text-only positions the aligner feeds, that is
  /// numerically identical to standard RoPE. A non-null `rope_scaling` must name
  /// a supported `rope_type` (`default` / `mrope` — the only base-theta MRoPE
  /// layout implemented) and carry a 3-section `mrope_section` summing to
  /// `head_dim / 2`; an unimplemented `rope_type` is an
  /// [`Error::UnknownEnumValue`].
  ///
  /// Assumes [`validate`](Self::validate) has confirmed the section is
  /// well-formed; it re-validates here so the typed value is always sound.
  pub fn mrope(&self) -> Result<MRopeConfig> {
    let half = self.head_dim / 2;
    let Some(scaling) = &self.rope_scaling else {
      return Ok(MRopeConfig {
        section: [half, 0, 0],
        interleaved: false,
      });
    };
    if scaling.is_null() {
      return Ok(MRopeConfig {
        section: [half, 0, 0],
        interleaved: false,
      });
    }

    let raw: RawRopeScaling = serde_json::from_value(scaling.clone()).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "Qwen3AsrTextConfig::mrope",
        "Qwen3-ASR text config rope_scaling",
        e,
      ))
    })?;

    // Only the default base-theta MRoPE path is implemented. A non-default
    // rope_type (linear / llama3 / yarn / longrope / …) selects a different
    // inverse-frequency or attention-scaling formula in the reference
    // `initialize_rope`; loading it here would silently produce wrong decoder
    // logits, so reject it with a typed error until that path exists.
    let rope_type = raw.rope_type();
    if !SUPPORTED_ROPE_TYPES.contains(&rope_type) {
      return Err(Error::UnknownEnumValue(UnknownEnumValuePayload::new(
        "Qwen3AsrTextConfig: rope_scaling.rope_type",
        rope_type,
        SUPPORTED_ROPE_TYPES,
      )));
    }

    let section_vec = raw.mrope_section.ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3AsrTextConfig: rope_scaling.mrope_section",
        "a non-null rope_scaling must carry mrope_section",
        format_smolstr!("{scaling}"),
      ))
    })?;
    if section_vec.len() != MROPE_AXES {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3AsrTextConfig: rope_scaling.mrope_section length",
        "must have exactly 3 entries (temporal, height, width)",
        format_smolstr!("len={}", section_vec.len()),
      )));
    }
    let mut section = [0i32; MROPE_AXES];
    let mut sum: i64 = 0;
    for (slot, &v) in section.iter_mut().zip(section_vec.iter()) {
      if v < 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "Qwen3AsrTextConfig: rope_scaling.mrope_section entry",
          "each section must be non-negative",
          format_smolstr!("{v}"),
        )));
      }
      *slot = v;
      sum += i64::from(v);
    }
    if sum != i64::from(half) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3AsrTextConfig: rope_scaling.mrope_section sum vs head_dim/2",
        "the three sections must sum to head_dim / 2",
        format_smolstr!("sum={sum}, head_dim/2={half}"),
      )));
    }
    let interleaved = raw.interleaved.or(raw.mrope_interleaved).unwrap_or(false);
    Ok(MRopeConfig {
      section,
      interleaved,
    })
  }

  /// Reject a structurally invalid configuration before it can panic the
  /// forward pass.
  ///
  /// Every width-like field must be a positive integer no larger than `2^24`;
  /// the cardinality fields (`num_hidden_layers`, the head counts) must be
  /// positive. Beyond that:
  /// `num_attention_heads` must be divisible by `num_key_value_heads`;
  /// `head_dim` must be even (RoPE rotates `k` with `k + head_dim/2`); the
  /// query-projection width `num_attention_heads * head_dim` must stay within
  /// the width cap; `rope_theta` must be finite and positive; `rms_norm_eps`
  /// must be finite; and a non-null `rope_scaling` must name a supported
  /// `rope_type` and carry a well-formed 3-section `mrope_section` summing to
  /// `head_dim / 2`.
  pub fn validate(&self) -> Result<()> {
    for (name, value) in [
      ("hidden_size", self.hidden_size),
      ("intermediate_size", self.intermediate_size),
      ("vocab_size", self.vocab_size),
      ("head_dim", self.head_dim),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Cardinality-like fields (layer / head counts): a non-positive value is
    // malformed (it sizes the decoder-layer `Vec` and is used as a divisor).
    for (name, value) in [
      ("num_hidden_layers", self.num_hidden_layers),
      ("num_attention_heads", self.num_attention_heads),
      ("num_key_value_heads", self.num_key_value_heads),
    ] {
      require_positive(name, value)?;
    }
    require_divisible(
      "num_attention_heads",
      self.num_attention_heads,
      "num_key_value_heads",
      self.num_key_value_heads,
    )?;
    require_even("head_dim", self.head_dim)?;
    let q_width = checked_mul(
      "Qwen3AsrTextConfig::validate: num_attention_heads * head_dim",
      "num_attention_heads",
      self.num_attention_heads,
      "head_dim",
      self.head_dim,
    )?;
    require_in_range("num_attention_heads * head_dim", q_width, 1, MAX_CONFIG_DIM)?;
    if !self.rope_theta.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Qwen3AsrTextConfig::validate (rope_theta)",
        f64::from(self.rope_theta),
      )));
    }
    if self.rope_theta <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3AsrTextConfig::validate (rope_theta)",
        "must be a positive float",
        format_smolstr!("rope_theta={}", self.rope_theta),
      )));
    }
    if !self.rms_norm_eps.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "Qwen3AsrTextConfig::validate (rms_norm_eps)",
        f64::from(self.rms_norm_eps),
      )));
    }
    // Parse + validate the MRoPE section (a non-null rope_scaling must be
    // well-formed; null/absent is the standard-RoPE fallback).
    self.mrope()?;
    Ok(())
  }
}
