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
  model_validation::{
    checked_add, checked_mul, require_cardinality, require_divisible, require_positive,
  },
};

/// The reference `model_type` tag (`config.py:54`, swift
/// `SenseVoiceConfig.swift:163`, the `MODEL_REMAPPING["sensevoice"]` table key
/// in `stt/utils.py:13`).
pub const MODEL_TYPE: &str = "sensevoice";

/// Inclusive upper bound for the encoder *block counts* (`num_blocks` /
/// `tp_blocks`). Each sizes an eager `Vec` of heavyweight SANM blocks reserved
/// up front in [`super::encoder::Encoder::from_weights`], so a corrupt
/// `config.json` with a huge count would request a multi-gigabyte allocation
/// before the first missing-key error. The released SenseVoice-Small has
/// `num_blocks = 50` / `tp_blocks = 20` (`config.py:14-15`); `4096` is generous
/// headroom for any variant yet keeps a malformed cardinality a recoverable
/// [`crate::error::Error::CapExceeded`] (or, if it still slips through, a
/// recoverable [`crate::error::Error::AllocFailure`] from the fallibly-reserved
/// block `Vec`) rather than an allocator abort. Matches the cardinality cap the
/// qwen3 / lfm2 / wav2vec2 configs use.
const MAX_CONFIG_CARDINALITY: i32 = 4096;

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

  /// Pre-norm flag (`normalize_before`; `true`). SenseVoice-Small is pre-norm
  /// (`config.py:17` defaults it `true`), and the encoder wires the pre-norm
  /// order unconditionally (`norm1` → attention → `norm2` → feed-forward,
  /// `sensevoice.py:220-237`); a `false` value is rejected by [`Self::validate`]
  /// as an unsupported configuration rather than silently running a different
  /// network.
  #[inline(always)]
  pub const fn normalize_before(&self) -> bool {
    self.normalize_before
  }

  /// Validate the encoder dims: every width / count is positive, the block
  /// counts are within `MAX_CONFIG_CARDINALITY` (they size eager per-block
  /// `Vec`s), the kernel is positive, the FSMN pad split stays non-negative (the
  /// `sanm_shift` cannot push `right_padding = kernel_size - 1 - left_padding`
  /// below zero, `sensevoice.py:164-168`), the hidden width is divisible by the
  /// head count (so the SANM head reshape is exact), and `normalize_before` is
  /// `true` (the only order the encoder wires).
  ///
  /// # Errors
  /// - [`crate::error::Error::OutOfRange`] if any dim is `<= 0`;
  /// - [`crate::error::Error::CapExceeded`] if `num_blocks` or `tp_blocks`
  ///   exceeds `MAX_CONFIG_CARDINALITY`;
  /// - [`crate::error::Error::DivisibilityConstraint`] if `output_size %
  ///   attention_heads != 0`;
  /// - [`crate::error::Error::InvariantViolation`] if `normalize_before` is
  ///   `false` (post-norm is not a supported SenseVoice-Small configuration).
  pub fn validate(&self) -> Result<()> {
    require_positive("encoder_conf.output_size", self.output_size)?;
    require_positive("encoder_conf.attention_heads", self.attention_heads)?;
    require_positive("encoder_conf.linear_units", self.linear_units)?;
    // `num_blocks` sizes the eager `encoders0` + `encoders` block `Vec`s, so it
    // takes the cardinality cap: positive AND within `MAX_CONFIG_CARDINALITY`.
    require_cardinality(
      "encoder_conf.num_blocks",
      i64::from(self.num_blocks),
      MAX_CONFIG_CARDINALITY as u64,
    )?;
    // `tp_blocks` may be `0` (a checkpoint with no second stage), so it is only
    // required to be non-negative — but it also sizes an eager `tp_encoders`
    // block `Vec`, so an over-cap positive count is rejected like `num_blocks`.
    if self.tp_blocks < 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "encoder_conf.tp_blocks",
          "must be >= 0",
          smol_str::format_smolstr!("{}", self.tp_blocks),
        ),
      ));
    }
    if i64::from(self.tp_blocks) > i64::from(MAX_CONFIG_CARDINALITY) {
      return Err(crate::error::Error::CapExceeded(
        crate::error::CapExceededPayload::new(
          "encoder_conf.tp_blocks",
          "encoder_conf.tp_blocks",
          MAX_CONFIG_CARDINALITY as u64,
          self.tp_blocks as u64,
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
    // The FSMN pad split `left_padding = (kernel_size - 1) / 2 (+ sanm_shift)`,
    // `right_padding = kernel_size - 1 - left_padding` (`sensevoice.py:164-168`).
    // A `sanm_shift` larger than the available right slack drives `right_padding`
    // negative, which `mx.pad` rejects only at the first forward; pin the
    // invariant at load instead. `kernel_size > 0` here so `(kernel_size - 1) / 2`
    // cannot wrap; the `+ sanm_shift` is checked so an adversarial pair cannot
    // overflow `i32`.
    let base_left = (self.kernel_size - 1) / 2;
    let left_padding = checked_add(
      "encoder_conf: FSMN left_padding = (kernel_size - 1) / 2 + sanm_shift",
      "(kernel_size - 1) / 2",
      base_left,
      "encoder_conf.sanm_shift",
      sanm_shift,
    )?;
    if self.kernel_size - 1 - left_padding < 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "encoder_conf.sanm_shift",
          "must keep the FSMN right pad (kernel_size - 1 - left_padding) >= 0",
          smol_str::format_smolstr!("{sanm_shift} (kernel_size {})", self.kernel_size),
        ),
      ));
    }
    require_divisible(
      "encoder_conf.output_size",
      self.output_size,
      "encoder_conf.attention_heads",
      self.attention_heads,
    )?;
    // SenseVoice-Small is pre-norm (`config.py:17`), and the encoder wires the
    // pre-norm order unconditionally (`sensevoice.py:220-237`). A `false` value
    // would otherwise load and run a different (unsupported) network — reject it
    // loudly here rather than mis-transcribe.
    if !self.normalize_before {
      return Err(crate::error::Error::InvariantViolation(
        crate::error::InvariantViolationPayload::new(
          "encoder_conf.normalize_before",
          "must be true (SenseVoice-Small is pre-norm; post-norm is unsupported)",
        ),
      ));
    }
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

  /// Validate the front-end params, including the fbank-derived invariants
  /// [`compute_fbank`](super::frontend::compute_fbank) /
  /// [`compute_fbank_kaldi`](crate::audio::features::compute_fbank_kaldi) would
  /// otherwise reject only at the first transcription — so a malformed
  /// `frontend_conf` fails at LOAD, not at first transcribe:
  ///
  /// - the sample rate, mel count, and frame sizes are positive (so the fbank
  ///   framing is well-formed);
  /// - the `window` string is one of the four `compute_fbank_kaldi` accepts
  ///   (`hamming` / `hanning` / `povey` / `rectangular`), via the shared
  ///   `window_from_str` typed set (`dsp.py:918-929`);
  /// - `n_mels > 3` — the reference `get_mel_banks_kaldi` asserts `num_bins > 3`
  ///   (`dsp.py:822`), the smallest count its `(num_bins + 1)` mel-delta math is
  ///   well-defined for;
  /// - the fbank Nyquist invariant `fs / 2 > LOW_FREQ` —
  ///   [`compute_fbank`](super::frontend::compute_fbank) always passes the fixed
  ///   `low_freq = LOW_FREQ` (20 Hz), and `get_mel_banks_kaldi` rejects
  ///   `low_freq >= nyquist` (`nyquist = fs / 2`, `dsp.py:826-831`), so a small
  ///   `fs` whose Nyquist is `<= LOW_FREQ` (e.g. `fs = 40`) passes `fs > 0` but
  ///   would fail only at the first transcribe;
  /// - the derived analysis sizes `win_len = fs * frame_length / 1000` and
  ///   `win_inc = fs * frame_shift / 1000` (samples, `sensevoice.py:27-28`)
  ///   satisfy `win_len >= 2` (the window functions divide by `win_len - 1`,
  ///   `dsp.py:920` etc., and the `next_power_of_2` padded size must be even,
  ///   `dsp.py:823`) and `win_inc > 0` (the strided-framing hop, `dsp.py:890`).
  ///   Both are computed with CHECKED arithmetic so a corrupt `fs` /
  ///   `frame_length` / `frame_shift` cannot overflow;
  /// - the LFR factors are positive AND within `MAX_CONFIG_CARDINALITY`.
  ///   `lfr_m` sizes the per-frame stack `lfr_m * n_mels` (the LFR output width,
  ///   `sensevoice.py:62`) and `lfr_n` is the stride; an oversized factor would
  ///   drive an enormous tile / reshape, so both take the same cardinality cap
  ///   the block counts do.
  ///
  /// # Errors
  /// - [`crate::error::Error::OutOfRange`] if any of `fs` / `n_mels` /
  ///   `frame_length` / `frame_shift` / `lfr_m` / `lfr_n` is `<= 0`, if `n_mels
  ///   <= 3`, if `fs / 2 <= LOW_FREQ` (the fbank Nyquist invariant), if the
  ///   `window` string is unrecognized, or if the derived `win_len < 2`;
  /// - [`crate::error::Error::InvariantViolation`] if the derived `win_inc == 0`;
  /// - [`crate::error::Error::ArithmeticOverflow`] if `fs * frame_length` or
  ///   `fs * frame_shift` overflows `i64`;
  /// - [`crate::error::Error::CapExceeded`] if the derived `win_len` exceeds the
  ///   fbank's `MAX_DECODED_SAMPLES` window budget, or if `lfr_m` / `lfr_n`
  ///   exceeds `MAX_CONFIG_CARDINALITY`.
  pub fn validate(&self) -> Result<()> {
    if self.fs == 0 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new("frontend_conf.fs", "must be > 0", "0"),
      ));
    }
    require_positive("frontend_conf.n_mels", self.n_mels)?;
    require_positive("frontend_conf.frame_length", self.frame_length)?;
    require_positive("frontend_conf.frame_shift", self.frame_shift)?;

    // The `window` string must be one of the four `compute_fbank_kaldi` accepts
    // (`dsp.py:918-929`); reuse the *one* typed accepted set so load and the
    // fbank step agree. The mapped `KaldiWindow` is discarded — only validity
    // matters here.
    super::frontend::window_from_str(&self.window)?;

    // `get_mel_banks_kaldi` asserts `num_bins > 3` (`dsp.py:822`); a mel count
    // `<= 3` fails only at the first fbank otherwise. Hoist it to load.
    if self.n_mels <= 3 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "frontend_conf.n_mels",
          "must be > 3 (get_mel_banks_kaldi requires at least 4 mel bins)",
          smol_str::format_smolstr!("{}", self.n_mels),
        ),
      ));
    }

    // The fbank Nyquist invariant. SenseVoice's `compute_fbank` always passes the
    // FIXED `low_freq = LOW_FREQ` (20 Hz) into `compute_fbank_kaldi`
    // (`frontend.rs` `compute_fbank`), and `get_mel_banks_kaldi` rejects
    // `low_freq >= nyquist` with `nyquist = fs / 2`
    // (`features.rs` `get_mel_banks_kaldi`, the `0.0 <= low_freq < nyquist`
    // check; `dsp.py:826-831`). A small `fs` (e.g. `40`, whose Nyquist is exactly
    // `LOW_FREQ`) passes the `fs > 0` / `win_len` checks but fails only at the
    // first transcribe — hoist the same invariant here, tied to the SAME
    // `LOW_FREQ` constant the fbank uses (referenced, not a divergent literal).
    // `high_freq = HIGH_FREQ = 0.0` resolves to Nyquist, which is `> 0` and
    // `<= nyquist` for any `fs > 0`, so it adds no constraint beyond this one.
    // `fs` is a `u32`, so `nyquist` is always finite (never NaN); the direct
    // `<=` is exact and avoids the partial-ord negation lint.
    let nyquist = self.fs as f32 * 0.5;
    if nyquist <= super::frontend::LOW_FREQ {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "frontend_conf.fs",
          "must keep the fbank Nyquist (fs / 2) above the fixed low_freq mel floor (20 Hz)",
          smol_str::format_smolstr!(
            "fs={} (nyquist={nyquist}, low_freq={})",
            self.fs,
            super::frontend::LOW_FREQ
          ),
        ),
      ));
    }

    // The derived analysis-window / hop sizes `win_len = fs * frame_length /
    // 1000`, `win_inc = fs * frame_shift / 1000` (samples, `sensevoice.py:27-28`).
    // `fs` / the frame sizes are validated positive above; compute the products
    // through CHECKED i64 arithmetic so a corrupt huge `fs` / frame size cannot
    // overflow, then narrow. `compute_fbank_kaldi` rejects `win_len < 2`
    // (`features.rs`, the `window_size - 1` window denominator / even padded
    // size) and `win_inc == 0` (the strided-framing hop) only at the first
    // transcribe; pin both at load.
    let win_len = derived_samples(
      "frontend_conf: win_len = fs * frame_length / 1000",
      self.fs,
      self.frame_length,
    )?;
    let win_inc = derived_samples(
      "frontend_conf: win_inc = fs * frame_shift / 1000",
      self.fs,
      self.frame_shift,
    )?;
    if win_len < 2 {
      return Err(crate::error::Error::OutOfRange(
        crate::error::OutOfRangePayload::new(
          "frontend_conf: derived win_len = fs * frame_length / 1000",
          "must be >= 2 (the fbank window denominator is win_len - 1)",
          smol_str::format_smolstr!(
            "{win_len} (fs={}, frame_length={})",
            self.fs,
            self.frame_length
          ),
        ),
      ));
    }
    // `compute_fbank_kaldi` rejects `win_len > MAX_DECODED_SAMPLES`
    // (`features.rs`, the analysis window cannot exceed the audio-IO sample
    // budget); hoist that same upper bound to load so a pathological `fs` /
    // `frame_length` whose derived window is enormous fails here, not at the
    // first transcribe. This bounds the config-derived window size against the
    // fbank's own contract — not a valid-input magnitude cap.
    if win_len > crate::audio::io::MAX_DECODED_SAMPLES as i64 {
      return Err(crate::error::Error::CapExceeded(
        crate::error::CapExceededPayload::new(
          "frontend_conf: derived win_len = fs * frame_length / 1000",
          "MAX_DECODED_SAMPLES",
          crate::audio::io::MAX_DECODED_SAMPLES as u64,
          win_len.max(0) as u64,
        ),
      ));
    }
    if win_inc == 0 {
      return Err(crate::error::Error::InvariantViolation(
        crate::error::InvariantViolationPayload::new(
          "frontend_conf: derived win_inc = fs * frame_shift / 1000",
          "must be > 0 (the fbank framing hop)",
        ),
      ));
    }

    // `lfr_m` / `lfr_n` size the LFR stack width / stride; bound them like the
    // encoder block counts so a corrupt value cannot request an enormous tile.
    require_cardinality(
      "frontend_conf.lfr_m",
      i64::from(self.lfr_m),
      MAX_CONFIG_CARDINALITY as u64,
    )?;
    require_cardinality(
      "frontend_conf.lfr_n",
      i64::from(self.lfr_n),
      MAX_CONFIG_CARDINALITY as u64,
    )?;
    Ok(())
  }
}

/// The reference's derived analysis-window / hop size in samples:
/// `int(sample_rate * ms / 1000)` (`sensevoice.py:27-28`), computed through
/// CHECKED i64 arithmetic so a corrupt huge `fs` / `ms` cannot overflow.
///
/// `fs` (a `u32`) and `ms` (validated positive by the caller) are widened to
/// `i64`; their product is at most `~9.2e18`, comfortably within `i64::MAX`, but
/// the multiply is checked so the bound holds for any input. The integer divide
/// by `1000` mirrors the reference's `int(...)` truncation. The result is
/// non-negative (both operands are) and is returned as `i64` so the caller can
/// compare against the `win_len >= 2` / `win_inc > 0` thresholds without a
/// narrowing cast.
///
/// # Errors
/// [`crate::error::Error::ArithmeticOverflow`] if `fs * ms` overflows `i64`.
fn derived_samples(context: &'static str, fs: u32, ms: i32) -> Result<i64> {
  let product = i64::from(fs).checked_mul(i64::from(ms)).ok_or_else(|| {
    crate::error::Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::with_operands(
      context,
      "i64",
      [("fs", u64::from(fs)), ("ms", i64::from(ms) as u64)],
    ))
  })?;
  Ok(product / 1000)
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
  /// positive, the nested configs validate, the LFR output width matches the
  /// encoder input width (`lfr_m * n_mels == input_size`), and (when both are
  /// present) the in-config CMVN means / inverse-stddev have a consistent,
  /// `input_size`-wide length — the CMVN stats are applied element-wise to the
  /// `(T', input_size)` LFR features.
  ///
  /// The LFR relation pins the front-end's output width to the encoder's input
  /// width: `_apply_lfr` stacks `lfr_m` consecutive `n_mels`-wide fbank frames
  /// into one `lfr_m * n_mels`-wide LFR frame (`sensevoice.py:62`), and that is
  /// exactly the `input_size` the encoder's first block consumes
  /// (`sensevoice.py:244-254`, `input_size = config.input_size`). At the
  /// SenseVoice-Small defaults this is `7 * 80 == 560`. A `config.json` whose
  /// `input_size` disagrees with `lfr_m * n_mels` would build an encoder that
  /// the front-end's features cannot feed; the product is computed with checked
  /// arithmetic so a corrupt `lfr_m` / `n_mels` cannot overflow `i32`.
  ///
  /// # Errors
  /// - [`crate::error::Error::OutOfRange`] if `vocab_size` / `input_size` is
  ///   `<= 0`, or from the nested validators;
  /// - [`crate::error::Error::DivisibilityConstraint`] from
  ///   [`EncoderConfig::validate`];
  /// - [`crate::error::Error::ArithmeticOverflow`] if `lfr_m * n_mels`
  ///   overflows `i32`;
  /// - [`crate::error::Error::LengthMismatch`] if `lfr_m * n_mels != input_size`,
  ///   if exactly one of `cmvn_means` / `cmvn_istd` is present, or if both are
  ///   present but their lengths differ or do not equal `input_size`.
  pub fn validate(&self) -> Result<()> {
    require_positive("config.vocab_size", self.vocab_size)?;
    require_positive("config.input_size", self.input_size)?;
    self.encoder_conf.validate()?;
    self.frontend_conf.validate()?;

    // The LFR output width must equal the encoder input width:
    // `lfr_m * n_mels == input_size` (`sensevoice.py:62`, `:244-254`). Both
    // factors are validated positive + cardinality-capped above, so the checked
    // product cannot overflow in practice — but compute it through `checked_mul`
    // so a corrupt config can never wrap into a spuriously-matching width.
    let lfr_width = checked_mul(
      "frontend_conf.lfr_m * frontend_conf.n_mels",
      "frontend_conf.lfr_m",
      self.frontend_conf.lfr_m(),
      "frontend_conf.n_mels",
      self.frontend_conf.n_mels(),
    )?;
    if lfr_width != self.input_size {
      return Err(crate::error::Error::LengthMismatch(
        crate::error::LengthMismatchPayload::new(
          "config.input_size vs frontend_conf.lfr_m * frontend_conf.n_mels",
          self.input_size.max(0) as usize,
          lfr_width.max(0) as usize,
        ),
      ));
    }

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
