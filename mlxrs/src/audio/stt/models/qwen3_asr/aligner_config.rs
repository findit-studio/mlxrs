//! Qwen3 forced-aligner configuration.
//!
//! Mirrors mlx-audio's `qwen3_forced_aligner.ForcedAlignerConfig`: the audio
//! encoder ([`AudioEncoderConfig`]) and text decoder ([`Qwen3AsrTextConfig`])
//! sub-configs plus the timestamp-classification head width and the special
//! token ids / segment quantum used to turn a predicted class id at each
//! `<timestamp>` position into a wall-clock time.
//!
//! The released Qwen3-ForcedAligner `config.json` nests `audio_config`,
//! `text_config`, the audio marker ids, `classify_num`, and (optionally) the
//! timestamp parameters under a top-level `thinker_config` object (the HF
//! "thinker" wrapper), with `timestamp_token_id` / `timestamp_segment_time`
//! also accepted at the root as a fallback. [`from_json`](ForcedAlignerConfig::from_json)
//! parses that nested shape: it requires the `thinker_config` object and errors
//! clearly if it is absent.
//!
//! The aligner is the Qwen3-ASR decoder with the vocab LM head **replaced** by
//! a `Linear(hidden_size -> classify_num, bias=False)` timestamp head: each
//! `<timestamp>` input position predicts a class id whose product with
//! [`timestamp_segment_time`](ForcedAlignerConfig::timestamp_segment_time)
//! (milliseconds) is the time of that boundary.
//!
//! Parsed via [`ForcedAlignerConfig::from_json`]; like the reference's
//! `from_dict`/`inspect.signature` filter, unmodeled keys are ignored (serde
//! does not `deny_unknown_fields`) and absent keys take the reference default.
//! The parsed config is then [`validate`]d so a malformed one (oversized
//! `classify_num`, a negative token id, a non-finite / non-positive segment
//! quantum) is a recoverable error here rather than a panic in the decode.
//!
//! [`validate`]: ForcedAlignerConfig::validate

use crate::{
  error::{
    Error, MissingKeyPayload, NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, Result,
  },
  model_validation::require_in_range,
};

use super::{config::AudioEncoderConfig, text::Qwen3AsrTextConfig};

/// Inclusive upper bound on the timestamp-classification head width
/// `classify_num`. The head is `Linear(hidden_size -> classify_num)`, so
/// `classify_num` is a width-like field sizing the `(classify_num, hidden)`
/// projection and the `(B, L, classify_num)` logits; `2^24` is the same
/// overflow-safe width cap the text decoder config uses (far above the real
/// ~5000) yet small enough that the downstream `i32` shape arithmetic cannot
/// overflow.
const MAX_CLASSIFY_NUM: i32 = 1 << 24;

/// The default number of timestamp classes (the reference dataclass default).
fn default_classify_num() -> i32 {
  5000
}
/// The default `<audio_pad>` token id (the reference dataclass default).
fn default_audio_token_id() -> i64 {
  151676
}
/// The default `<|audio_start|>` token id.
fn default_audio_start_token_id() -> i64 {
  151669
}
/// The default `<|audio_end|>` token id.
fn default_audio_end_token_id() -> i64 {
  151670
}
/// The default `<timestamp>` token id.
fn default_timestamp_token_id() -> i64 {
  151705
}
/// The default segment quantum in milliseconds (a class id times this is the
/// boundary time).
fn default_timestamp_segment_time() -> f64 {
  80.0
}

/// The top level of a Qwen3-ForcedAligner `config.json`: the (required)
/// `thinker_config` object plus the root-level timestamp fallbacks. Everything
/// the aligner actually configures lives inside `thinker_config`.
#[derive(Debug, serde::Deserialize)]
struct RootConfig {
  /// The HF "thinker" wrapper carrying the aligner sub-configs and ids.
  #[serde(default)]
  thinker_config: Option<serde_json::Value>,
  /// Root-level `<timestamp>` token id — a fallback when the thinker object
  /// omits it.
  #[serde(default)]
  timestamp_token_id: Option<i64>,
  /// Root-level per-class segment quantum (ms) — a fallback when the thinker
  /// object omits it.
  #[serde(default)]
  timestamp_segment_time: Option<f64>,
}

/// Qwen3 forced-aligner configuration — a serde-parsed mirror of mlx-audio's
/// `ForcedAlignerConfig`.
///
/// The audio / text sub-configs carry their own validation
/// ([`AudioEncoderConfig::validate`] / [`Qwen3AsrTextConfig::validate`]); the
/// aligner-specific fields are the timestamp-head width
/// ([`classify_num`](Self::classify_num)) and the special token ids / segment
/// quantum the decode reads.
#[derive(Debug, Clone, serde::Deserialize)]
#[non_exhaustive]
pub struct ForcedAlignerConfig {
  /// The audio-encoder (audio tower) sub-config.
  #[serde(default)]
  pub audio_config: AudioEncoderConfig,
  /// The Qwen3-ASR text-decoder sub-config (carries the MRoPE `rope_scaling`).
  #[serde(default)]
  pub text_config: Qwen3AsrTextConfig,
  /// Number of timestamp classes — the width of the
  /// `Linear(hidden -> classify_num)` head and the logits' last axis.
  #[serde(default = "default_classify_num")]
  pub classify_num: i32,
  /// The `<audio_pad>` token id: positions equal to it in `input_ids` are
  /// where the audio-encoder features are spliced in.
  #[serde(default = "default_audio_token_id")]
  pub audio_token_id: i64,
  /// The `<|audio_start|>` token id (carried; bounds the audio span).
  #[serde(default = "default_audio_start_token_id")]
  pub audio_start_token_id: i64,
  /// The `<|audio_end|>` token id (carried; bounds the audio span).
  #[serde(default = "default_audio_end_token_id")]
  pub audio_end_token_id: i64,
  /// The `<timestamp>` token id: the argmax class at each such input position
  /// times [`timestamp_segment_time`](Self::timestamp_segment_time) is a
  /// boundary time. Words sit between paired `<timestamp><timestamp>` markers
  /// (even index = start, odd index = end).
  #[serde(default = "default_timestamp_token_id")]
  pub timestamp_token_id: i64,
  /// The per-class time quantum in **milliseconds**: a predicted class id `c`
  /// at a `<timestamp>` position decodes to `c * timestamp_segment_time` ms.
  #[serde(default = "default_timestamp_segment_time")]
  pub timestamp_segment_time: f64,
}

impl Default for ForcedAlignerConfig {
  fn default() -> Self {
    Self {
      audio_config: AudioEncoderConfig::default(),
      text_config: Qwen3AsrTextConfig::default(),
      classify_num: default_classify_num(),
      audio_token_id: default_audio_token_id(),
      audio_start_token_id: default_audio_start_token_id(),
      audio_end_token_id: default_audio_end_token_id(),
      timestamp_token_id: default_timestamp_token_id(),
      timestamp_segment_time: default_timestamp_segment_time(),
    }
  }
}

impl ForcedAlignerConfig {
  /// Parse a [`ForcedAlignerConfig`] from a Qwen3-ForcedAligner `config.json`
  /// string.
  ///
  /// The released checkpoint nests the aligner fields (`audio_config`,
  /// `text_config`, `audio_token_id`, `audio_start_token_id`,
  /// `audio_end_token_id`, `classify_num`, and optionally `timestamp_token_id` /
  /// `timestamp_segment_time`) under a top-level `thinker_config` object; this
  /// parser **requires** that object and returns a clear [`Error::MissingKey`]
  /// when it is absent (an unwrapped flat config would otherwise be parsed as
  /// all-defaults). `timestamp_token_id` / `timestamp_segment_time` are also
  /// accepted at the root as a fallback when the thinker object omits them.
  ///
  /// A serde failure (malformed JSON) maps to [`Error::Parse`]; unmodeled keys
  /// are ignored and absent modeled keys take the reference default. The parsed
  /// config is then [`validate`]d so a malformed one is a recoverable error here
  /// rather than a panic downstream.
  ///
  /// [`validate`]: ForcedAlignerConfig::validate
  pub fn from_json(json: &str) -> Result<ForcedAlignerConfig> {
    // Parse the root, capturing the (required) thinker_config object and the
    // root-level timestamp fallbacks.
    let root: RootConfig = serde_json::from_str(json).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "ForcedAlignerConfig::from_json",
        "Qwen3 forced-aligner config JSON",
        e,
      ))
    })?;

    let thinker = root.thinker_config.ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "Qwen3-ForcedAligner config.json",
        "thinker_config",
      ))
    })?;
    let serde_json::Value::Object(thinker_map) = thinker else {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Qwen3-ForcedAligner config.json: thinker_config",
        "must be a JSON object",
        thinker.to_string(),
      )));
    };

    // Whether the thinker object explicitly carried the root-fallback keys
    // (checked before the map is consumed by deserialization).
    let thinker_has_token_id = thinker_map.contains_key("timestamp_token_id");
    let thinker_has_segment_time = thinker_map.contains_key("timestamp_segment_time");

    // Deserialize the aligner fields from inside the thinker object (the field
    // names match the nested keys). Unmodeled keys are ignored; absent ones take
    // the field default.
    let mut cfg: ForcedAlignerConfig =
      serde_json::from_value(serde_json::Value::Object(thinker_map)).map_err(|e| {
        Error::Parse(ParsePayload::new(
          "ForcedAlignerConfig::from_json",
          "Qwen3 forced-aligner thinker_config object",
          e,
        ))
      })?;

    // Root-level timestamp fallbacks apply only when the thinker object did not
    // carry the field itself.
    if let Some(t) = root.timestamp_token_id
      && !thinker_has_token_id
    {
      cfg.timestamp_token_id = t;
    }
    if let Some(t) = root.timestamp_segment_time
      && !thinker_has_segment_time
    {
      cfg.timestamp_segment_time = t;
    }

    cfg.validate()?;
    Ok(cfg)
  }

  /// Reject a structurally invalid configuration before it can panic the
  /// forward pass or the decode.
  ///
  /// Validates both sub-configs ([`AudioEncoderConfig::validate`] /
  /// [`Qwen3AsrTextConfig::validate`]) and the aligner-specific fields:
  ///
  /// - `classify_num` must be a positive integer no larger than
  ///   `MAX_CLASSIFY_NUM` (the overflow-safe `2^24` width cap), so the head
  ///   projection and `(B, L, classify_num)` logits cannot overflow `i32`.
  /// - every special token id (`audio_token_id`, `audio_start_token_id`,
  ///   `audio_end_token_id`, `timestamp_token_id`) must be non-negative (a
  ///   token id is an embedding-row index).
  /// - `timestamp_segment_time` must be **finite and positive** — it scales a
  ///   class id into a time; a non-finite or non-positive quantum would
  ///   produce NaN / non-increasing boundary times.
  ///
  /// Returns the first violation; `Ok(())` when every field is sound.
  pub fn validate(&self) -> Result<()> {
    self.audio_config.validate()?;
    self.text_config.validate()?;

    require_in_range("classify_num", self.classify_num, 1, MAX_CLASSIFY_NUM)?;

    for (name, id) in [
      ("audio_token_id", self.audio_token_id),
      ("audio_start_token_id", self.audio_start_token_id),
      ("audio_end_token_id", self.audio_end_token_id),
      ("timestamp_token_id", self.timestamp_token_id),
    ] {
      if id < 0 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          name,
          "token id must be non-negative (an embedding-row index)",
          id.to_string(),
        )));
      }
    }

    if !self.timestamp_segment_time.is_finite() {
      return Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "ForcedAlignerConfig::validate (timestamp_segment_time)",
        self.timestamp_segment_time,
      )));
    }
    if self.timestamp_segment_time <= 0.0 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAlignerConfig::validate (timestamp_segment_time)",
        "must be a positive number of milliseconds",
        self.timestamp_segment_time.to_string(),
      )));
    }
    Ok(())
  }
}
