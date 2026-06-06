//! Silero VAD configuration — port of the two reference dataclasses in
//! [`vad/models/silero_vad/config.py`][config]: `BranchConfig` (`:9-19`,
//! one per sample-rate branch) and the top-level `ModelConfig` (`:22-63`).
//!
//! The reference dataclasses carry plain public fields with `default`s; this
//! port mirrors the same defaults but keeps the fields private behind
//! accessors (the crate's struct conventions). [`ModelConfig::from_json`]
//! reproduces the reference's `from_dict` precedence: an absent `branch_16k` /
//! `branch_8k` falls back to the per-rate default branch (`config.py:35-52`),
//! and a present sub-object is parsed key-by-key over that default (a missing
//! sub-key keeps the branch default, exactly like a Python `@dataclass` field
//! left unset). An eager `validate` rejects non-positive dims
//! with typed errors — the reference performs no such validation, so a
//! malformed `config.json` would otherwise surface as a downstream reshape
//! error deep in the forward.
//!
//! [config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/config.py

use crate::{
  dtype::Dtype,
  error::{Error, ParsePayload, Result},
  model_validation::require_positive,
};

/// The reference `model_type` tag (`config.py:25`).
pub const MODEL_TYPE: &str = "silero_vad";

/// Configuration for one Silero VAD sample-rate branch — port of
/// `BranchConfig` ([config.py:9-19][config]).
///
/// One branch processes audio at a single sample rate (16 kHz or 8 kHz). The
/// [`Self::chunk_size`] is the fixed number of *new* samples consumed per
/// frame (512 at 16 kHz, 256 at 8 kHz), and [`Self::context_size`] is the
/// count of trailing samples from the previous frame prepended as left
/// context.
///
/// [config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/config.py#L9-L19
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchConfig {
  sample_rate: u32,
  filter_length: i32,
  hop_length: i32,
  pad: i32,
  cutoff: i32,
  context_size: i32,
  chunk_size: i32,
}

impl BranchConfig {
  /// The 16 kHz branch defaults (`config.py:12-19` — the `BranchConfig()`
  /// dataclass defaults).
  pub const fn default_16k() -> Self {
    Self {
      sample_rate: 16_000,
      filter_length: 256,
      hop_length: 128,
      pad: 64,
      cutoff: 129,
      context_size: 64,
      chunk_size: 512,
    }
  }

  /// The 8 kHz branch defaults (`config.py:43-52` — the explicit
  /// `BranchConfig(sample_rate=8000, …)` the `ModelConfig.__post_init__`
  /// installs when `branch_8k` is absent).
  pub const fn default_8k() -> Self {
    Self {
      sample_rate: 8_000,
      filter_length: 128,
      hop_length: 64,
      pad: 32,
      cutoff: 65,
      context_size: 32,
      chunk_size: 256,
    }
  }

  /// Construct a [`BranchConfig`] from explicit fields. Mainly for tests; the
  /// loader builds branches through [`Self::default_16k`] / [`Self::default_8k`]
  /// overlaid by [`ModelConfig::from_json`].
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    sample_rate: u32,
    filter_length: i32,
    hop_length: i32,
    pad: i32,
    cutoff: i32,
    context_size: i32,
    chunk_size: i32,
  ) -> Self {
    Self {
      sample_rate,
      filter_length,
      hop_length,
      pad,
      cutoff,
      context_size,
      chunk_size,
    }
  }

  /// The branch sample rate in Hz.
  #[inline(always)]
  pub const fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// The STFT analysis window length (the `stft_conv` kernel size).
  #[inline(always)]
  pub const fn filter_length(&self) -> i32 {
    self.filter_length
  }

  /// The STFT hop (the `stft_conv` stride).
  #[inline(always)]
  pub const fn hop_length(&self) -> i32 {
    self.hop_length
  }

  /// The reflect-padding applied to the right before the STFT conv.
  #[inline(always)]
  pub const fn pad(&self) -> i32 {
    self.pad
  }

  /// The number of magnitude-spectrum bins kept (the real/imag split point;
  /// `stft_conv` emits `2 * cutoff` channels).
  #[inline(always)]
  pub const fn cutoff(&self) -> i32 {
    self.cutoff
  }

  /// The count of trailing samples carried as left context between frames.
  #[inline(always)]
  pub const fn context_size(&self) -> i32 {
    self.context_size
  }

  /// The fixed count of *new* samples consumed per frame.
  #[inline(always)]
  pub const fn chunk_size(&self) -> i32 {
    self.chunk_size
  }

  /// Overlay the present keys of a JSON sub-object onto this branch — the
  /// per-key `from_dict` of `BranchConfig` (`config.py`'s
  /// `BranchConfig.from_dict`). A missing key keeps the existing (default)
  /// value, matching a Python `@dataclass` field left unset.
  fn overlay_json(mut self, obj: &serde_json::Map<String, serde_json::Value>) -> Result<Self> {
    if let Some(v) = obj.get("sample_rate") {
      self.sample_rate = parse_u32("BranchConfig.sample_rate", v)?;
    }
    if let Some(v) = obj.get("filter_length") {
      self.filter_length = parse_i32("BranchConfig.filter_length", v)?;
    }
    if let Some(v) = obj.get("hop_length") {
      self.hop_length = parse_i32("BranchConfig.hop_length", v)?;
    }
    if let Some(v) = obj.get("pad") {
      self.pad = parse_i32("BranchConfig.pad", v)?;
    }
    if let Some(v) = obj.get("cutoff") {
      self.cutoff = parse_i32("BranchConfig.cutoff", v)?;
    }
    if let Some(v) = obj.get("context_size") {
      self.context_size = parse_i32("BranchConfig.context_size", v)?;
    }
    if let Some(v) = obj.get("chunk_size") {
      self.chunk_size = parse_i32("BranchConfig.chunk_size", v)?;
    }
    Ok(self)
  }

  /// Reject non-positive dims that would otherwise surface as a downstream
  /// reshape / conv error. The reference performs no validation; this guard
  /// is the audio-config convention.
  fn validate(&self) -> Result<()> {
    require_positive("BranchConfig.filter_length", self.filter_length)?;
    require_positive("BranchConfig.hop_length", self.hop_length)?;
    // `pad` may legitimately be 0 (the reflect-pad no-op path), so it is only
    // required to be non-negative.
    if self.pad < 0 {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "BranchConfig.pad",
        "must be >= 0",
        smol_str::format_smolstr!("{}", self.pad),
      )));
    }
    require_positive("BranchConfig.cutoff", self.cutoff)?;
    require_positive("BranchConfig.context_size", self.context_size)?;
    require_positive("BranchConfig.chunk_size", self.chunk_size)?;
    Ok(())
  }
}

/// Silero voice activity detector configuration — port of `ModelConfig`
/// ([config.py:22-63][config]).
///
/// Carries the two per-rate [`BranchConfig`]s plus the post-processing
/// hyper-parameters ([`Self::threshold`], the min-speech / min-silence
/// durations, and the speech-pad) the speech-segment extractor reads. The
/// [`Self::dtype`] selects the model's activation precision (`float16` or
/// `float32`).
///
/// [config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/config.py#L22-L63
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
  dtype: Dtype,
  threshold: f64,
  min_speech_duration_ms: i32,
  min_silence_duration_ms: i32,
  speech_pad_ms: i32,
  branch_16k: BranchConfig,
  branch_8k: BranchConfig,
}

impl Default for ModelConfig {
  /// The reference `ModelConfig()` dataclass defaults (`config.py:25-52`).
  fn default() -> Self {
    Self {
      dtype: Dtype::F32,
      threshold: 0.5,
      min_speech_duration_ms: 250,
      min_silence_duration_ms: 100,
      speech_pad_ms: 30,
      branch_16k: BranchConfig::default_16k(),
      branch_8k: BranchConfig::default_8k(),
    }
  }
}

impl ModelConfig {
  /// The model activation dtype (`mx.float16` if `dtype == "float16"`, else
  /// `mx.float32` — `silero_vad.py:117`).
  #[inline(always)]
  pub const fn dtype(&self) -> Dtype {
    self.dtype
  }

  /// The speech-probability decision threshold.
  #[inline(always)]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }

  /// Minimum speech-segment duration (ms); shorter detections are dropped.
  #[inline(always)]
  pub const fn min_speech_duration_ms(&self) -> i32 {
    self.min_speech_duration_ms
  }

  /// Minimum silence duration (ms) required to close a speech segment.
  #[inline(always)]
  pub const fn min_silence_duration_ms(&self) -> i32 {
    self.min_silence_duration_ms
  }

  /// Padding (ms) added to each side of an emitted speech segment.
  #[inline(always)]
  pub const fn speech_pad_ms(&self) -> i32 {
    self.speech_pad_ms
  }

  /// The 16 kHz branch config.
  #[inline(always)]
  pub const fn branch_16k(&self) -> &BranchConfig {
    &self.branch_16k
  }

  /// The 8 kHz branch config.
  #[inline(always)]
  pub const fn branch_8k(&self) -> &BranchConfig {
    &self.branch_8k
  }

  /// Parse a [`ModelConfig`] from a verbatim `config.json` body — the analog
  /// of the reference's `ModelConfig.from_dict` (`config.py:54-63`) plus the
  /// `__post_init__` branch defaulting (`config.py:35-52`).
  ///
  /// Precedence, faithful to the reference:
  /// - a top-level scalar absent from the JSON keeps its dataclass default;
  /// - `dtype` is the string `"float16"` → [`Dtype::F16`], anything else →
  ///   [`Dtype::F32`] (`silero_vad.py:117` reads it identically);
  /// - `branch_16k` / `branch_8k` absent → the per-rate default branch
  ///   ([`BranchConfig::default_16k`] / [`BranchConfig::default_8k`]);
  ///   present → that default overlaid by the sub-object's keys.
  ///
  /// # Errors
  /// - [`Error::Parse`] if the body is not a JSON object, or a numeric field
  ///   is the wrong JSON type / out of `i32`/`u32` range;
  /// - [`Error::OutOfRange`] from the eager branch `validate` for a
  ///   non-positive branch dim.
  pub fn from_json(config_json: &str) -> Result<Self> {
    use serde_json::Value;

    let value: Value = serde_json::from_str(config_json)
      .map_err(|e| Error::Parse(ParsePayload::new("silero_vad config", "JSON", e)))?;
    let Value::Object(map) = value else {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "silero_vad config",
        "must be a JSON object",
        "non-object",
      )));
    };

    let mut cfg = Self::default();

    if let Some(v) = map.get("dtype") {
      // `mx.float16 if config.dtype == "float16" else mx.float32`
      // (silero_vad.py:117) — the reference supports ONLY float16-or-float32 for
      // Silero: the exact string "float16" selects half, and EVERY other string
      // (including "bfloat16") maps to F32, exactly as the reference does. This
      // is intentional 1:1 parity, not a dtype-preservation gap — Silero ships
      // float32 / float16 checkpoints, never bf16.
      cfg.dtype = match v.as_str() {
        Some("float16") => Dtype::F16,
        _ => Dtype::F32,
      };
    }
    if let Some(v) = map.get("threshold") {
      cfg.threshold = parse_f64("ModelConfig.threshold", v)?;
    }
    if let Some(v) = map.get("min_speech_duration_ms") {
      cfg.min_speech_duration_ms = parse_i32("ModelConfig.min_speech_duration_ms", v)?;
    }
    if let Some(v) = map.get("min_silence_duration_ms") {
      cfg.min_silence_duration_ms = parse_i32("ModelConfig.min_silence_duration_ms", v)?;
    }
    if let Some(v) = map.get("speech_pad_ms") {
      cfg.speech_pad_ms = parse_i32("ModelConfig.speech_pad_ms", v)?;
    }

    // Tri-state branch handling, faithful to the reference `__post_init__`
    // (`config.py:36-52`) which only special-cases a `dict` (→ `from_dict`) vs
    // `None` (→ the per-rate default): a JSON `null` (serde `Value::Null`) maps
    // to Python `None`, so absent OR null keeps the default; a JSON object
    // overlays; any OTHER present type (array / string / number / bool) is a
    // malformed branch — the reference would leave the raw value and crash
    // downstream, so here we fail CLOSED with a typed error instead of silently
    // using the default.
    match map.get("branch_16k") {
      None | Some(Value::Null) => {}
      Some(Value::Object(obj)) => cfg.branch_16k = cfg.branch_16k.overlay_json(obj)?,
      Some(_) => return Err(branch_type_error("branch_16k")),
    }
    match map.get("branch_8k") {
      None | Some(Value::Null) => {}
      // A PRESENT (object) `branch_8k`, even partial, fills omitted fields from
      // the 16 kHz dataclass defaults (`BranchConfig.from_dict`, config.py:41-42)
      // — NOT the 8 kHz overrides, which apply only when it is absent/null (the
      // `Self::default()` 8 kHz branch kept above).
      Some(Value::Object(obj)) => cfg.branch_8k = BranchConfig::default_16k().overlay_json(obj)?,
      Some(_) => return Err(branch_type_error("branch_8k")),
    }

    cfg.validate()?;
    Ok(cfg)
  }

  /// Eager validation of both branches plus the non-negative post-processing
  /// durations.
  fn validate(&self) -> Result<()> {
    self.branch_16k.validate()?;
    self.branch_8k.validate()?;
    for (field, v) in [
      (
        "ModelConfig.min_speech_duration_ms",
        self.min_speech_duration_ms,
      ),
      (
        "ModelConfig.min_silence_duration_ms",
        self.min_silence_duration_ms,
      ),
      ("ModelConfig.speech_pad_ms", self.speech_pad_ms),
    ] {
      if v < 0 {
        return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
          field,
          "must be >= 0",
          smol_str::format_smolstr!("{v}"),
        )));
      }
    }
    Ok(())
  }
}

/// A present `branch_16k` / `branch_8k` that is neither a JSON object nor
/// `null` is a malformed config — reject it with a typed error (fail closed)
/// rather than silently falling back to the per-rate default.
fn branch_type_error(field: &'static str) -> Error {
  Error::OutOfRange(crate::error::OutOfRangePayload::new(
    field,
    "must be a JSON object or null",
    "non-object",
  ))
}

fn parse_i32(field: &'static str, v: &serde_json::Value) -> Result<i32> {
  v.as_i64()
    .and_then(|n| i32::try_from(n).ok())
    .ok_or_else(|| {
      Error::OutOfRange(crate::error::OutOfRangePayload::new(
        field,
        "must be an i32 integer",
        smol_str::format_smolstr!("{v}"),
      ))
    })
}

fn parse_u32(field: &'static str, v: &serde_json::Value) -> Result<u32> {
  v.as_u64()
    .and_then(|n| u32::try_from(n).ok())
    .ok_or_else(|| {
      Error::OutOfRange(crate::error::OutOfRangePayload::new(
        field,
        "must be a u32 integer",
        smol_str::format_smolstr!("{v}"),
      ))
    })
}

fn parse_f64(field: &'static str, v: &serde_json::Value) -> Result<f64> {
  v.as_f64().ok_or_else(|| {
    Error::OutOfRange(crate::error::OutOfRangePayload::new(
      field,
      "must be a number",
      smol_str::format_smolstr!("{v}"),
    ))
  })
}
