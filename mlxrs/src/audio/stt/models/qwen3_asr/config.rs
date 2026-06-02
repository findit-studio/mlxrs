//! Qwen3-ASR audio-encoder configuration.
//!
//! Mirrors mlx-audio's `qwen3_asr.config.AudioEncoderConfig` (the `audio_config`
//! block of the model's `config.json`). Parsed via
//! [`AudioEncoderConfig::from_json`]; like the reference's
//! `from_dict`/`inspect.signature` filter, unmodeled keys are ignored (serde
//! does not `deny_unknown_fields`) and absent keys take the reference default,
//! so a partial config still parses. The parsed fields are then [`validate`]d
//! so a malformed config (zero / negative / non-divisible dimension, or one
//! whose downstream `i32` shape arithmetic would overflow) is a recoverable
//! error here rather than a panic deep in the forward pass.
//!
//! [`validate`]: AudioEncoderConfig::validate

use smol_str::format_smolstr;

use crate::{
  error::{Error, OutOfRangePayload, Result},
  model_validation::{
    checked_mul, require_divisible, require_even, require_in_range, require_positive,
  },
};

/// Inclusive upper bound for every *width*-like config field (`d_model`,
/// `encoder_ffn_dim`, `output_dim`, `num_mel_bins`, `downsample_hidden_size`,
/// `max_source_positions`). `2^20` is far above any real Qwen3-ASR checkpoint
/// (the largest, `encoder_ffn_dim`, is 4096) yet small enough that the
/// downstream `i32` shape arithmetic — the per-head reshapes, the
/// `downsample_hidden_size * freq_after_conv` conv-out width, the
/// `d_model * output_dim` head — cannot overflow `i32`. Rejecting an oversized
/// field keeps a malformed config a recoverable [`Error::OutOfRange`] instead
/// of a wrapping multiply downstream.
const MAX_CONFIG_DIM: i32 = 1 << 20;

/// Qwen3-ASR audio-encoder configuration — a serde-parsed mirror of mlx-audio's
/// `AudioEncoderConfig`.
///
/// Defaults match the reference dataclass. The forward pass uses the mel-bin
/// count, the model / FFN / downsample / output dimensions, the layer and head
/// counts, and the window sizes; `dropout` / `attention_dropout` /
/// `activation_dropout` / `initializer_range` are carried for parse
/// completeness (inference is dropout-free). `activation_function` is pinned to
/// `"gelu"` by [`validate`](AudioEncoderConfig::validate) (every block hardcodes
/// GELU, so a deviating value would silently run a different graph).
#[derive(Debug, Clone, serde::Deserialize)]
#[non_exhaustive]
pub struct AudioEncoderConfig {
  /// Number of mel-spectrogram frequency bins — the conv stem's freq (height)
  /// axis before downsampling.
  #[serde(default = "default_num_mel_bins")]
  pub num_mel_bins: i32,
  /// Number of transformer encoder layers.
  #[serde(default = "default_encoder_layers")]
  pub encoder_layers: i32,
  /// Number of self-attention heads per encoder layer.
  #[serde(default = "default_encoder_attention_heads")]
  pub encoder_attention_heads: i32,
  /// Feed-forward (fc1) intermediate dimension.
  #[serde(default = "default_encoder_ffn_dim")]
  pub encoder_ffn_dim: i32,
  /// Encoder hidden / embedding dimension.
  #[serde(default = "default_d_model")]
  pub d_model: i32,
  /// Attention dropout (carried; inference is dropout-free).
  #[serde(default)]
  pub dropout: f32,
  /// Attention-probability dropout (carried).
  #[serde(default)]
  pub attention_dropout: f32,
  /// Activation function. Pinned to `"gelu"` by
  /// [`validate`](AudioEncoderConfig::validate).
  #[serde(default = "default_activation_function")]
  pub activation_function: String,
  /// Activation dropout (carried).
  #[serde(default)]
  pub activation_dropout: f32,
  /// Whether the conv-out embedding is scaled by `sqrt(d_model)`.
  #[serde(default)]
  pub scale_embedding: bool,
  /// Weight-init std (carried; unused at inference).
  #[serde(default = "default_initializer_range")]
  pub initializer_range: f32,
  /// Maximum positional context — the length of the precomputed sinusoidal
  /// position-embedding table (and the dense-attention worst case).
  #[serde(default = "default_max_source_positions")]
  pub max_source_positions: i32,
  /// Per-window mel-frame count (the chunked-attention window is `n_window *
  /// 2` mel frames). See the
  /// [`AudioEncoder`](crate::audio::stt::models::qwen3_asr::AudioEncoder)
  /// block-attention notes.
  #[serde(default = "default_n_window")]
  pub n_window: i32,
  /// Output projection (proj2) dimension — the audio-embedding width fed to
  /// the text decoder.
  #[serde(default = "default_output_dim")]
  pub output_dim: i32,
  /// Inference window stride (post-CNN block grouping is
  /// `n_window_infer / (n_window * 2)` windows wide).
  #[serde(default = "default_n_window_infer")]
  pub n_window_infer: i32,
  /// Conv chunk size (carried; chunked-conv tiling parameter).
  #[serde(default = "default_conv_chunksize")]
  pub conv_chunksize: i32,
  /// Conv2d stem hidden channel count (each of the three convs).
  #[serde(default = "default_downsample_hidden_size")]
  pub downsample_hidden_size: i32,
}

fn default_num_mel_bins() -> i32 {
  128
}
fn default_encoder_layers() -> i32 {
  24
}
fn default_encoder_attention_heads() -> i32 {
  16
}
fn default_encoder_ffn_dim() -> i32 {
  4096
}
fn default_d_model() -> i32 {
  1024
}
fn default_activation_function() -> String {
  "gelu".to_string()
}
fn default_initializer_range() -> f32 {
  0.02
}
fn default_max_source_positions() -> i32 {
  1500
}
fn default_n_window() -> i32 {
  50
}
fn default_output_dim() -> i32 {
  2048
}
fn default_n_window_infer() -> i32 {
  800
}
fn default_conv_chunksize() -> i32 {
  500
}
fn default_downsample_hidden_size() -> i32 {
  480
}

impl Default for AudioEncoderConfig {
  /// The reference `AudioEncoderConfig` dataclass defaults (the same per-field
  /// defaults serde applies to an absent key). Known-valid —
  /// [`validate`](Self::validate) passes on it.
  fn default() -> Self {
    Self {
      num_mel_bins: default_num_mel_bins(),
      encoder_layers: default_encoder_layers(),
      encoder_attention_heads: default_encoder_attention_heads(),
      encoder_ffn_dim: default_encoder_ffn_dim(),
      d_model: default_d_model(),
      dropout: 0.0,
      attention_dropout: 0.0,
      activation_function: default_activation_function(),
      activation_dropout: 0.0,
      scale_embedding: false,
      initializer_range: default_initializer_range(),
      max_source_positions: default_max_source_positions(),
      n_window: default_n_window(),
      output_dim: default_output_dim(),
      n_window_infer: default_n_window_infer(),
      conv_chunksize: default_conv_chunksize(),
      downsample_hidden_size: default_downsample_hidden_size(),
    }
  }
}

/// The output length of one `kernel = 3, stride = 2, padding = 1` convolution
/// over an axis of length `n`: `(n + 2*1 - 3) / 2 + 1 = (n - 1) / 2 + 1`
/// (floor), equivalently `(n + 1) / 2` for `n >= 1`. Saturates at `0` for an
/// empty / degenerate axis. Pure integer arithmetic over `i64` so it never
/// overflows for any realistic mel/time length.
const fn conv3x3_stride2_out(n: i64) -> i64 {
  if n <= 0 {
    return 0;
  }
  (n + 1) / 2
}

impl AudioEncoderConfig {
  /// Parse an [`AudioEncoderConfig`] from a `config.json` (`audio_config`)
  /// string.
  ///
  /// A serde failure (malformed JSON) maps to [`Error::Parse`]; missing keys
  /// fall back to the reference defaults rather than erroring. The parsed
  /// fields are then [`validate`]d so a malformed config is a recoverable error
  /// here rather than a panic downstream.
  ///
  /// [`validate`]: AudioEncoderConfig::validate
  pub fn from_json(json: &str) -> Result<AudioEncoderConfig> {
    let cfg: AudioEncoderConfig = serde_json::from_str(json).map_err(|e| {
      Error::Parse(crate::error::ParsePayload::new(
        "AudioEncoderConfig::from_json",
        "Qwen3-ASR audio config JSON",
        e,
      ))
    })?;
    cfg.validate()?;
    Ok(cfg)
  }

  /// Per-head attention dimension `d_model / encoder_attention_heads`.
  /// Assumes [`validate`](Self::validate) has confirmed divisibility.
  #[inline]
  pub fn head_dim(&self) -> i32 {
    self.d_model / self.encoder_attention_heads
  }

  /// The mel-frequency axis length after the three stride-2 convs:
  /// `f3 = conv(conv(conv(num_mel_bins)))`. This is the reference's
  /// `((((num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2`, expressed via the shared
  /// `conv3x3_stride2_out` recurrence. Computed in `i64` (never overflows for
  /// a within-cap `num_mel_bins`) and returned as `i32`.
  #[inline]
  pub fn freq_after_conv(&self) -> i32 {
    let f1 = conv3x3_stride2_out(i64::from(self.num_mel_bins));
    let f2 = conv3x3_stride2_out(f1);
    let f3 = conv3x3_stride2_out(f2);
    // f3 <= num_mel_bins <= MAX_CONFIG_DIM, so the cast is lossless.
    f3 as i32
  }

  /// The time axis length after the three stride-2 convs for an input of
  /// `time` mel frames: `conv(conv(conv(time)))`, the same recurrence as
  /// [`freq_after_conv`](Self::freq_after_conv) applied to the time axis. Used
  /// to predict the downsampled `[batch, time', d_model]` shape. `i64`
  /// throughout (overflow-safe for any realistic frame count).
  #[inline]
  pub fn time_after_conv(time: i64) -> i64 {
    let t1 = conv3x3_stride2_out(time);
    let t2 = conv3x3_stride2_out(t1);
    conv3x3_stride2_out(t2)
  }

  /// The conv-out linear's input width `downsample_hidden_size *
  /// freq_after_conv` — the flattened `(channel, freq)` feature fed to the
  /// `conv_out` projection. A checked product (kept within `MAX_CONFIG_DIM`)
  /// so the flatten reshape cannot overflow `i32`.
  #[inline]
  pub fn conv_out_in_features(&self) -> Result<i32> {
    checked_mul(
      "AudioEncoderConfig: downsample_hidden_size * freq_after_conv",
      "downsample_hidden_size",
      self.downsample_hidden_size,
      "freq_after_conv",
      self.freq_after_conv(),
    )
  }

  /// Reject a structurally invalid configuration before it can panic the
  /// forward pass.
  ///
  /// Every width-like field (`num_mel_bins`, `d_model`, `encoder_ffn_dim`,
  /// `output_dim`, `downsample_hidden_size`, `max_source_positions`) must be a
  /// positive integer no larger than `MAX_CONFIG_DIM` (so the downstream
  /// `i32` shape arithmetic cannot overflow); the cardinality fields
  /// (`encoder_layers`, `encoder_attention_heads`) must be positive. Beyond the
  /// per-field bounds:
  ///
  /// - `activation_function` must be `"gelu"` ([`Error::UnknownEnumValue`]) —
  ///   every encoder block hardcodes GELU, so a deviating value would run a
  ///   different graph silently.
  /// - `d_model` must be divisible by `encoder_attention_heads`
  ///   ([`Error::DivisibilityConstraint`]) — the per-head reshape and the
  ///   [`scaled_dot_product_attention`](crate::lm::nn::attention::scaled_dot_product_attention)
  ///   kernel require it.
  /// - `d_model` must be **even** ([`Error::OutOfRange`]) — the sinusoidal
  ///   position embedding splits the channels into `sin`/`cos` halves and the
  ///   reference raises `"needs even channels"` for an odd width.
  /// - the conv-out width `downsample_hidden_size * freq_after_conv` must stay
  ///   within `MAX_CONFIG_DIM` (a checked product), so the flatten reshape
  ///   cannot overflow `i32`.
  /// - `max_source_positions` (the sinusoidal positional-table row count) must
  ///   cover the longest per-chunk post-CNN sequence the windowed encoder can
  ///   produce — `time_after_conv(n_window * 2)`, the conv output length of a
  ///   full `n_window * 2`-frame conv chunk. The windowed path adds
  ///   `pos_emb[:t']` to every chunk, so a `max_source_positions` smaller than
  ///   that would clamp/over-slice the table for a valid full-chunk utterance.
  ///   This makes the windowed audio path self-consistent at construction (the
  ///   per-call positional-slice guard still rejects an over-long caller mel on
  ///   the plain forward).
  ///
  /// Returns the first violation; `Ok(())` when every field is sound.
  pub fn validate(&self) -> Result<()> {
    // Width-like fields: positive and within the overflow-safe cap.
    for (name, value) in [
      ("num_mel_bins", self.num_mel_bins),
      ("d_model", self.d_model),
      ("encoder_ffn_dim", self.encoder_ffn_dim),
      ("output_dim", self.output_dim),
      ("downsample_hidden_size", self.downsample_hidden_size),
      ("max_source_positions", self.max_source_positions),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Window sizes: positive (used as divisors / stride lengths). Not eager
    // allocations, so the width cap is fine.
    for (name, value) in [
      ("n_window", self.n_window),
      ("n_window_infer", self.n_window_infer),
      ("conv_chunksize", self.conv_chunksize),
    ] {
      require_in_range(name, value, 1, MAX_CONFIG_DIM)?;
    }
    // Cardinality-like fields (layer / head counts): a non-positive value is
    // malformed (it sizes the encoder-layer `Vec` and is used as a divisor).
    for (name, value) in [
      ("encoder_layers", self.encoder_layers),
      ("encoder_attention_heads", self.encoder_attention_heads),
    ] {
      require_positive(name, value)?;
    }
    // Every block hardcodes GELU; a deviating activation would run a different
    // (unsupported) graph silently.
    crate::model_validation::pin_str(
      "AudioEncoderConfig: activation_function",
      self.activation_function.as_str(),
      &["gelu"],
    )?;
    // Attention head grouping: d_model must split evenly across heads.
    require_divisible(
      "d_model",
      self.d_model,
      "encoder_attention_heads",
      self.encoder_attention_heads,
    )?;
    // Sinusoidal position embedding needs even channels (sin/cos halves); the
    // reference raises "needs even channels input" otherwise.
    require_even("d_model", self.d_model)?;
    // The conv-out flatten width must stay within the overflow-safe cap.
    let conv_out_in = self.conv_out_in_features()?;
    require_in_range(
      "downsample_hidden_size * freq_after_conv",
      conv_out_in,
      1,
      MAX_CONFIG_DIM,
    )?;
    // The positional table must be long enough for the windowed encoder's
    // longest per-chunk post-CNN sequence: a full `n_window * 2`-frame conv
    // chunk produces `time_after_conv(n_window * 2)` rows, and the windowed path
    // adds `pos_emb[:t']` to every chunk. A `max_source_positions` smaller than
    // that would (absent the runtime guard) clamp the slice and reuse positions;
    // reject it here so the aligner's audio path is self-consistent for any
    // valid full-chunk utterance. `n_window` is positive (checked above), so the
    // conv chunk is `>= 2`; `time_after_conv` is `i64` and `<= n_window * 2 <=
    // MAX_CONFIG_DIM`, so the cast is lossless.
    let max_chunk_after_cnn = Self::time_after_conv(i64::from(self.n_window).saturating_mul(2));
    let needed = i32::try_from(max_chunk_after_cnn).unwrap_or(i32::MAX);
    if self.max_source_positions < needed {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "AudioEncoderConfig: max_source_positions vs time_after_conv(n_window * 2)",
        "max_source_positions must cover the post-CNN length of a full conv chunk",
        format_smolstr!(
          "max_source_positions={}, needed={needed}",
          self.max_source_positions
        ),
      )));
    }
    Ok(())
  }
}
