//! SenseVoice-Small model configuration.
//!
//! Faithful port of the three reference dataclasses in
//! [`config.py`][config] — `EncoderConfig` (`:5-29`), `FrontendConfig`
//! (`:32-49`), and the top-level `ModelConfig` (`:52-95`) — plus their
//! `from_dict` constructors. The reference dataclasses carry plain public
//! fields with `default`s; this port mirrors the same defaults but keeps the
//! fields private behind accessors (the crate's struct conventions) and adds an
//! eager [`Config::validate`] that rejects non-positive / non-divisible dims
//! with typed errors — the reference performs no such validation, so a
//! malformed `config.json` would otherwise surface as a downstream reshape
//! error deep in the forward.
//!
//! The two nested configs are deserialized from the `encoder_conf` /
//! `frontend_conf` sub-objects of `config.json` (mirroring the reference's
//! `__post_init__` / `from_dict`, which build `EncoderConfig.from_dict` /
//! `FrontendConfig.from_dict` from those sub-dicts, `config.py:65-95`). The
//! upstream `config.json` carries the misspelled key `sanm_shfit`; the reference
//! `EncoderConfig.from_dict` maps it to `sanm_shift` only when the correct key
//! is absent (`config.py:26-28`). serde's `alias` reproduces exactly that
//! precedence: the canonical `sanm_shift` wins when both keys are present,
//! otherwise the `sanm_shfit` alias fills the field.
//!
//! [config]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/config.py

use crate::{
  error::Result,
  model_validation::{require_divisible, require_positive},
};

/// The reference `model_type` tag (`config.py:54`, swift
/// `SenseVoiceConfig.swift:163`, the `MODEL_REMAPPING["sensevoice"]` table key
/// in `stt/utils.py:13`).
pub const MODEL_TYPE: &str = "sensevoice";

// ─────────────────────────────── defaults ───────────────────────────────
// Each mirrors the corresponding reference dataclass field default verbatim,
// so a `config.json` omitting a key resolves to the same value the Python
// `@dataclass` would.

const fn default_output_size() -> i32 {
  512
}
const fn default_attention_heads() -> i32 {
  4
}
const fn default_linear_units() -> i32 {
  2048
}
const fn default_num_blocks() -> i32 {
  50
}
const fn default_tp_blocks() -> i32 {
  20
}
const fn default_kernel_size() -> i32 {
  11
}
const fn default_normalize_before() -> bool {
  true
}

const fn default_fs() -> u32 {
  16000
}
fn default_window() -> String {
  "hamming".to_string()
}
const fn default_n_mels() -> i32 {
  80
}
const fn default_frame_length() -> i32 {
  25
}
const fn default_frame_shift() -> i32 {
  10
}
const fn default_lfr_m() -> i32 {
  7
}
const fn default_lfr_n() -> i32 {
  6
}

const fn default_vocab_size() -> i32 {
  25055
}
const fn default_input_size() -> i32 {
  560
}

fn default_model_type() -> String {
  MODEL_TYPE.to_string()
}

/// The SANM encoder hyper-parameters (`config.py:5-29`, `EncoderConfig`).
///
/// The fields the encoder graph actually consumes are `output_size`,
/// `attention_heads`, `linear_units`, `num_blocks`, `tp_blocks`,
/// `kernel_size`, `sanm_shift`, and `normalize_before`. The four `*_dropout_*`
/// fields the reference carries are inference no-ops (mlx evaluates with
/// dropout disabled), so they are intentionally not modeled here.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EncoderConfig {
  #[serde(default = "default_output_size")]
  output_size: i32,
  #[serde(default = "default_attention_heads")]
  attention_heads: i32,
  #[serde(default = "default_linear_units")]
  linear_units: i32,
  #[serde(default = "default_num_blocks")]
  num_blocks: i32,
  #[serde(default = "default_tp_blocks")]
  tp_blocks: i32,
  #[serde(default = "default_kernel_size")]
  kernel_size: i32,
  // The upstream `config.json` misspells this key `sanm_shfit`. The reference
  // `from_dict` resolves the precedence "use `sanm_shift` if present, else the
  // `sanm_shfit` typo, else the dataclass default `0`" (`config.py:26-28`). A
  // serde `alias` cannot express this — with both keys present serde raises a
  // duplicate-field error — so the canonical key and the typo are captured as
  // two separate optional fields and the precedence is resolved in
  // [`EncoderConfig::sanm_shift`].
  #[serde(default)]
  sanm_shift: Option<i32>,
  #[serde(default)]
  sanm_shfit: Option<i32>,
  #[serde(default = "default_normalize_before")]
  normalize_before: bool,
}

impl Default for EncoderConfig {
  fn default() -> Self {
    // Mirrors the `EncoderConfig` dataclass field defaults (`config.py:7-17`).
    Self {
      output_size: default_output_size(),
      attention_heads: default_attention_heads(),
      linear_units: default_linear_units(),
      num_blocks: default_num_blocks(),
      tp_blocks: default_tp_blocks(),
      kernel_size: default_kernel_size(),
      sanm_shift: None,
      sanm_shfit: None,
      normalize_before: default_normalize_before(),
    }
  }
}

impl EncoderConfig {
  /// Encoder hidden width (`output_size`; `512`). Also the SANM attention's
  /// `n_feat` and the CTC head's input width.
  #[inline(always)]
  pub const fn output_size(&self) -> i32 {
    self.output_size
  }

  /// SANM self-attention head count (`attention_heads`; `4`).
  #[inline(always)]
  pub const fn attention_heads(&self) -> i32 {
    self.attention_heads
  }

  /// Position-wise feed-forward hidden width (`linear_units`; `2048`).
  #[inline(always)]
  pub const fn linear_units(&self) -> i32 {
    self.linear_units
  }

  /// Total `encoders0` + `encoders` block count (`num_blocks`; `50`). The
  /// width-changing first block lives in `encoders0`; the remaining
  /// `num_blocks - 1` constant-width blocks live in `encoders`.
  #[inline(always)]
  pub const fn num_blocks(&self) -> i32 {
    self.num_blocks
  }

  /// Second-stage (`tp_encoders`) block count (`tp_blocks`; `20`).
  #[inline(always)]
  pub const fn tp_blocks(&self) -> i32 {
    self.tp_blocks
  }

  /// Depthwise FSMN convolution kernel size (`kernel_size`; `11`).
  #[inline(always)]
  pub const fn kernel_size(&self) -> i32 {
    self.kernel_size
  }

  /// FSMN asymmetric-pad shift (`sanm_shift`; `0`). Added to the left pad when
  /// positive (`sensevoice.py:164-168`).
  ///
  /// Resolves the canonical-key / typo-key precedence the reference `from_dict`
  /// implements (`config.py:26-28`): the correct `sanm_shift` wins; the
  /// misspelled `sanm_shfit` fills in only when the correct key is absent;
  /// neither present falls back to the dataclass default `0`.
  #[inline(always)]
  pub const fn sanm_shift(&self) -> i32 {
    match (self.sanm_shift, self.sanm_shfit) {
      (Some(v), _) => v,
      (None, Some(v)) => v,
      (None, None) => 0,
    }
  }

  /// Pre-norm flag (`normalize_before`; `true`). Every released checkpoint is
  /// pre-norm; the post-norm arm is intentionally not wired.
  #[inline(always)]
  pub const fn normalize_before(&self) -> bool {
    self.normalize_before
  }

  /// Validate the encoder dims: every width / count is positive, the kernel is
  /// positive (so the FSMN pad split is non-negative), and the hidden width is
  /// divisible by the head count (so the SANM head reshape is exact).
  ///
  /// # Errors
  /// - [`crate::error::Error::OutOfRange`] if any dim is `<= 0`;
  /// - [`crate::error::Error::DivisibilityConstraint`] if `output_size %
  ///   attention_heads != 0`.
  pub fn validate(&self) -> Result<()> {
    require_positive("encoder_conf.output_size", self.output_size)?;
    require_positive("encoder_conf.attention_heads", self.attention_heads)?;
    require_positive("encoder_conf.linear_units", self.linear_units)?;
    require_positive("encoder_conf.num_blocks", self.num_blocks)?;
    // `tp_blocks` may be `0` (a checkpoint with no second stage), so it is
    // only required to be non-negative.
    if self.tp_blocks < 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "encoder_conf.tp_blocks",
          "must be >= 0",
          smol_str::format_smolstr!("{}", self.tp_blocks),
        ),
      ));
    }
    require_positive("encoder_conf.kernel_size", self.kernel_size)?;
    let sanm_shift = self.sanm_shift();
    if sanm_shift < 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "encoder_conf.sanm_shift",
          "must be >= 0",
          smol_str::format_smolstr!("{sanm_shift}"),
        ),
      ));
    }
    require_divisible(
      "encoder_conf.output_size",
      self.output_size,
      "encoder_conf.attention_heads",
      self.attention_heads,
    )?;
    Ok(())
  }
}

/// The front-end (fbank / LFR / CMVN) parameters (`config.py:32-49`,
/// `FrontendConfig`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FrontendConfig {
  #[serde(default = "default_fs")]
  fs: u32,
  #[serde(default = "default_window")]
  window: String,
  #[serde(default = "default_n_mels")]
  n_mels: i32,
  #[serde(default = "default_frame_length")]
  frame_length: i32,
  #[serde(default = "default_frame_shift")]
  frame_shift: i32,
  #[serde(default = "default_lfr_m")]
  lfr_m: i32,
  #[serde(default = "default_lfr_n")]
  lfr_n: i32,
}

impl Default for FrontendConfig {
  fn default() -> Self {
    // Mirrors the `FrontendConfig` dataclass field defaults (`config.py:34-40`).
    Self {
      fs: default_fs(),
      window: default_window(),
      n_mels: default_n_mels(),
      frame_length: default_frame_length(),
      frame_shift: default_frame_shift(),
      lfr_m: default_lfr_m(),
      lfr_n: default_lfr_n(),
    }
  }
}

impl FrontendConfig {
  /// Sample rate the model expects (`fs`; `16000` Hz).
  #[inline(always)]
  pub const fn fs(&self) -> u32 {
    self.fs
  }

  /// Kaldi analysis window kind (`window`; `"hamming"`).
  #[inline(always)]
  pub fn window(&self) -> &str {
    &self.window
  }

  /// Mel filterbank bin count before LFR stacking (`n_mels`; `80`).
  #[inline(always)]
  pub const fn n_mels(&self) -> i32 {
    self.n_mels
  }

  /// Analysis-window length in milliseconds (`frame_length`; `25` ms = 400
  /// samples at 16 kHz). The reference passes this to `_compute_fbank` as
  /// `frame_length_ms` (`sensevoice.py:385`).
  #[inline(always)]
  pub const fn frame_length(&self) -> i32 {
    self.frame_length
  }

  /// Frame hop in milliseconds (`frame_shift`; `10` ms = 160 samples at
  /// 16 kHz).
  #[inline(always)]
  pub const fn frame_shift(&self) -> i32 {
    self.frame_shift
  }

  /// Low-Frame-Rate stacking factor (`lfr_m`; `7`) — the number of consecutive
  /// fbank frames stacked into one LFR frame.
  #[inline(always)]
  pub const fn lfr_m(&self) -> i32 {
    self.lfr_m
  }

  /// Low-Frame-Rate stride (`lfr_n`; `6`) — the hop between consecutive LFR
  /// windows.
  #[inline(always)]
  pub const fn lfr_n(&self) -> i32 {
    self.lfr_n
  }

  /// Validate the front-end params: the sample rate, mel count, frame sizes,
  /// and LFR factors are positive (so the fbank framing and the LFR stacking
  /// are well-formed).
  ///
  /// # Errors
  /// [`crate::error::Error::OutOfRange`] if any of `fs` / `n_mels` /
  /// `frame_length` / `frame_shift` / `lfr_m` / `lfr_n` is `<= 0`.
  pub fn validate(&self) -> Result<()> {
    if self.fs == 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new("frontend_conf.fs", "must be > 0", "0"),
      ));
    }
    require_positive("frontend_conf.n_mels", self.n_mels)?;
    require_positive("frontend_conf.frame_length", self.frame_length)?;
    require_positive("frontend_conf.frame_shift", self.frame_shift)?;
    require_positive("frontend_conf.lfr_m", self.lfr_m)?;
    require_positive("frontend_conf.lfr_n", self.lfr_n)?;
    Ok(())
  }
}

/// The top-level SenseVoice-Small configuration (`config.py:52-95`,
/// `ModelConfig`).
///
/// The two nested configs default when absent, matching the reference
/// `__post_init__` (`config.py:65-73`). The optional `cmvn_means` /
/// `cmvn_istd` carry the global CMVN statistics when a checkpoint embeds them
/// in `config.json` instead of shipping an `am.mvn` file (`config.py:59-61`,
/// consumed by the reference `post_load_hook` fallback `sensevoice.py:577-579`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
  #[serde(default = "default_model_type")]
  model_type: String,
  #[serde(default = "default_vocab_size")]
  vocab_size: i32,
  #[serde(default = "default_input_size")]
  input_size: i32,
  #[serde(default)]
  encoder_conf: EncoderConfig,
  #[serde(default)]
  frontend_conf: FrontendConfig,
  #[serde(default)]
  cmvn_means: Option<Vec<f32>>,
  #[serde(default)]
  cmvn_istd: Option<Vec<f32>>,
}

impl Default for Config {
  fn default() -> Self {
    // Mirrors the `ModelConfig` dataclass defaults (`config.py:54-63`), with the
    // two nested configs defaulted as `__post_init__` does (`:66-72`).
    Self {
      model_type: default_model_type(),
      vocab_size: default_vocab_size(),
      input_size: default_input_size(),
      encoder_conf: EncoderConfig::default(),
      frontend_conf: FrontendConfig::default(),
      cmvn_means: None,
      cmvn_istd: None,
    }
  }
}

impl Config {
  /// The `model_type` tag (`"sensevoice"`).
  #[inline(always)]
  pub fn model_type(&self) -> &str {
    &self.model_type
  }

  /// CTC vocabulary size (`vocab_size`; `25055`) — the `ctc_lo` output width.
  #[inline(always)]
  pub const fn vocab_size(&self) -> i32 {
    self.vocab_size
  }

  /// Encoder input width after LFR stacking (`input_size`; `560 = lfr_m *
  /// n_mels = 7 * 80`). Also the width of the 16-row prompt-embedding table.
  #[inline(always)]
  pub const fn input_size(&self) -> i32 {
    self.input_size
  }

  /// The nested encoder configuration.
  #[inline(always)]
  pub const fn encoder_conf(&self) -> &EncoderConfig {
    &self.encoder_conf
  }

  /// The nested front-end configuration.
  #[inline(always)]
  pub const fn frontend_conf(&self) -> &FrontendConfig {
    &self.frontend_conf
  }

  /// The optional in-`config.json` CMVN means (the `am.mvn` `<AddShift>`
  /// fallback, `config.py:59`).
  #[inline(always)]
  pub fn cmvn_means(&self) -> Option<&[f32]> {
    self.cmvn_means.as_deref()
  }

  /// The optional in-`config.json` CMVN inverse-stddev (the `am.mvn`
  /// `<Rescale>` fallback, `config.py:60`).
  #[inline(always)]
  pub fn cmvn_istd(&self) -> Option<&[f32]> {
    self.cmvn_istd.as_deref()
  }

  /// Validate the whole configuration: `vocab_size` / `input_size` are
  /// positive, the nested configs validate, and (when both are present) the
  /// in-config CMVN means / inverse-stddev have a consistent, `input_size`-wide
  /// length — the CMVN stats are applied element-wise to the `(T', input_size)`
  /// LFR features.
  ///
  /// # Errors
  /// - [`crate::error::Error::OutOfRange`] if `vocab_size` / `input_size` is
  ///   `<= 0`, or from the nested validators;
  /// - [`crate::error::Error::DivisibilityConstraint`] from
  ///   [`EncoderConfig::validate`];
  /// - [`crate::error::Error::LengthMismatch`] if exactly one of `cmvn_means` /
  ///   `cmvn_istd` is present, or if both are present but their lengths differ
  ///   or do not equal `input_size`.
  pub fn validate(&self) -> Result<()> {
    require_positive("config.vocab_size", self.vocab_size)?;
    require_positive("config.input_size", self.input_size)?;
    self.encoder_conf.validate()?;
    self.frontend_conf.validate()?;

    // The in-config CMVN fallback (`config.py:59-61`) is consumed as a pair —
    // means AND inverse-stddev — by `post_load_hook` (`sensevoice.py:577-579`).
    // Either both or neither must be present, and (when present) both must be
    // `input_size`-wide for the element-wise `(feats + means) * istd`.
    match (&self.cmvn_means, &self.cmvn_istd) {
      (None, None) => {}
      (Some(means), Some(istd)) => {
        let input = self.input_size.max(0) as usize;
        if means.len() != input {
          return Err(crate::error::Error::LengthMismatch(
            crate::error::LengthMismatchPayload::new("config.cmvn_means", input, means.len()),
          ));
        }
        if istd.len() != input {
          return Err(crate::error::Error::LengthMismatch(
            crate::error::LengthMismatchPayload::new("config.cmvn_istd", input, istd.len()),
          ));
        }
      }
      (Some(_), None) => {
        return Err(crate::error::Error::LengthMismatch(
          crate::error::LengthMismatchPayload::new(
            "config.cmvn_istd (required when cmvn_means is set)",
            self.cmvn_means.as_ref().map_or(0, Vec::len),
            0,
          ),
        ));
      }
      (None, Some(_)) => {
        return Err(crate::error::Error::LengthMismatch(
          crate::error::LengthMismatchPayload::new(
            "config.cmvn_means (required when cmvn_istd is set)",
            self.cmvn_istd.as_ref().map_or(0, Vec::len),
            0,
          ),
        ));
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests;
