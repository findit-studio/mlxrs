//! Whisper model configuration — [`ModelDimensions`] (`whisper.py:269-312`).
//!
//! Faithful port of the `ModelDimensions` dataclass + its `from_dict`
//! classmethod, which accepts **both** the native MLX field names
//! (`n_mels`, `n_audio_ctx`, …) and the HuggingFace transformers field names
//! (`d_model`, `encoder_layers`, …). On top of the reference, this adds an
//! eager [`ModelDimensions::validate`] that rejects zero / oversized /
//! non-divisible (`n_state % n_head`) values with the crate's typed errors —
//! the reference performs no such validation, so a malformed `config.json`
//! would otherwise surface as a downstream reshape panic.

use serde_json::Value;
use smol_str::format_smolstr;

use crate::{
  Error, Result,
  error::{EmptyInputPayload, MissingFieldPayload, OutOfRangePayload, ParsePayload},
  model_validation::{
    Extent, elem_count, pin_usize, require_cardinality, require_divisible, require_even,
    require_in_range, require_positive, reserve_or_error,
  },
};

use super::audio::N_FRAMES;

/// The `n_vocab` threshold at/above which a Whisper checkpoint is
/// multilingual (`Model.is_multilingual`, `whisper.py:621-623`): the
/// English-only `*.en` models top out at `51864`, the multilingual ones add
/// the language + task tokens to reach `>= 51865`.
const MULTILINGUAL_VOCAB_THRESHOLD: usize = 51865;

/// An upper bound on every dimension, enforced **at construction** as each
/// field's [`Extent`] cap, rejecting absurd / malicious config values before
/// they are used to size allocations. Whisper large-v3 is the biggest official
/// checkpoint (`n_audio_state = 1280`, `n_vocab = 51866`, 32 layers, 20 heads,
/// `n_text_ctx = 448`); this cap (`1 << 22`, ~4.2M) clears every real model by
/// orders of magnitude while keeping the products that follow (e.g.
/// `n_layer * n_head`) far from `usize` overflow.
const MAX_DIM: usize = 1 << 22;

/// Element-count cap on every dense 2-D tensor [`ModelDimensions`] materializes
/// whose extent is the *product* of two dimensions: the encoder positional
/// embedding (`n_audio_ctx * n_audio_state`), the decoder positional embedding
/// (`n_text_ctx * n_text_state`), the decoder causal mask (`n_text_ctx *
/// n_text_ctx`), and the encoder `conv1` pre-downsample activation (`N_FRAMES *
/// n_audio_state`, on the fixed padded frame count). Each field is individually
/// `<= MAX_DIM`, but their product is not, so [`elem_count`] caps the
/// materialized extent directly. `1 << 26` (~67M elements, ~268 MB at `f32`)
/// clears the largest real product — large-v3's `n_audio_ctx * n_audio_state =
/// 1500 * 1280 = 1_920_000` — by ~35x while bounding a hostile but
/// per-field-valid config far below an out-of-memory allocation.
const DENSE_2D_ELEM_CAP: usize = 1 << 26;

/// Element-count cap on the mel filterbank (`n_mels * n_freqs`). `n_freqs` is
/// the one-sided STFT width [`N_FREQS`] (a constant), so this is effectively a
/// tighter `n_mels` bound on the filterbank `magnitudes @ filters.T` matmul.
/// `1 << 20` (~1M elements) clears the largest real bank — large-v3's
/// `n_mels * n_freqs = 128 * 201 = 25_728` — by ~40x.
const MEL_FILTER_ELEM_CAP: usize = 1 << 20;

/// Element-count cap on every dense attention-score tensor the forward pass
/// materializes — the `(batch=1, n_head, T_q, T_kv)` `q @ kᵀ` product, whose
/// dense extent is `n_head * T_q * T_kv`. Three sites reach it: encoder
/// self-attention (`n_audio_head * n_audio_ctx * n_audio_ctx`), decoder
/// self-attention (`n_text_head * n_text_ctx * n_text_ctx`), and cross-attention
/// (`n_text_head * n_text_ctx * n_audio_ctx`). Each factor is individually
/// `<= MAX_DIM`, but a per-field-valid config (e.g. `n_text_ctx = n_text_head =
/// 8192`) can still multiply to a multi-gigabyte score buffer, so [`elem_count`]
/// bounds the three-way product directly. `1 << 29` (~537M elements, ~2.1 GB at
/// `f32`) clears the largest real score buffer — large-v3's encoder
/// `20 * 1500 * 1500 = 45_000_000` — by ~12x.
const ATTN_SCORE_ELEM_CAP: usize = 1 << 29;

/// Element-count cap on every transformer-block MLP hidden activation — the
/// `(batch=1, T, 4 * n_state)` first-projection output, dense extent
/// `T * MLP_RATIO * n_state`. Two sites reach it: the encoder MLP
/// (`n_audio_ctx * 4 * n_audio_state`) and the decoder MLP
/// (`n_text_ctx * 4 * n_text_state`). `1 << 27` (~134M elements) clears the
/// largest real hidden — large-v3's encoder `1500 * 4 * 1280 = 7_680_000` — by
/// ~17x.
const MLP_HIDDEN_ELEM_CAP: usize = 1 << 27;

/// The transformer MLP expansion ratio — the hidden width is `4 * n_state`
/// (`ResidualAttentionBlock` builds `mlp1: n_state -> 4 * n_state`). A small
/// fixed constant (`<= MAX_DIM`), so it is a valid [`Extent`] axis in the
/// MLP-hidden product.
const MLP_RATIO: usize = 4;

/// Element-count cap on the vocabulary-projection tensors — the weight-tied
/// logit head. Two extents share this magnitude class: the per-step logits
/// `(batch=1, n_text_ctx, n_vocab)` (extent `n_text_ctx * n_vocab`) and the
/// token-embedding / tied-logit table `(n_vocab, n_text_state)` (extent
/// `n_vocab * n_text_state`). `1 << 28` (~268M elements) clears the larger real
/// extent — large-v3's table `51866 * 1280 = 66_388_480` — by ~4x.
const VOCAB_PROJ_ELEM_CAP: usize = 1 << 28;

/// Element-count cap on each per-attention-kind cumulative KV cache — the
/// decode cache grows one `(batch=1, T, n_state)` key (and an equal value)
/// per block up to the full context, so the cumulative key extent is
/// `n_layer * T * n_state`. Two extents reach it: the decoder self-attention
/// cache (`n_text_layer * n_text_ctx * n_text_state`) and the cross-attention
/// cache (`n_text_layer * n_audio_ctx * n_audio_state`). `1 << 29` (~537M
/// elements) clears the larger real cache — large-v3's cross cache
/// `32 * 1500 * 1280 = 61_440_000` — by ~8x.
const KV_CACHE_ELEM_CAP: usize = 1 << 29;

/// The smallest non-degenerate encoder hidden width for the sinusoidal
/// positional embedding. `AudioEncoder::new` builds `sinusoids(n_audio_ctx,
/// n_audio_state)`, whose `inv_timescales = exp(-log(max_timescale) /
/// (n_audio_state/2 - 1) * arange(n_audio_state/2))` divides by
/// `n_audio_state/2 - 1`. So `n_audio_state` must be even (the `concat([sin,
/// cos])` halves the width) AND `n_audio_state/2 >= 2`, i.e. `n_audio_state >=
/// 4`: width 1 / any odd value is not even, and width 2 makes the divisor
/// `n_audio_state/2 - 1 == 0`, producing a `+inf` increment and a `0 * inf`
/// `NaN` row. Every released Whisper checkpoint has `n_audio_state >= 384`, so
/// this clears them all; it rejects only the degenerate small widths a malformed
/// config could carry. (The `concat` halving also requires `n_audio_state` even,
/// already implied by `>= 4` together with [`require_even`].)
const MIN_AUDIO_STATE_FOR_SINUSOID: i32 = 4;

/// The encoder convolution downsample factor — `conv2` runs at stride 2, so the
/// pre-downsample `conv1` activation spans [`N_FRAMES`] frames before the time
/// axis is halved to `n_audio_ctx` (`N_FRAMES / CONV_DOWNSAMPLE`). A small fixed
/// constant (`<= MAX_DIM`), so it is a valid [`Extent`] axis where the conv
/// downsample relates frame counts.
const CONV_DOWNSAMPLE: usize = 2;

/// The architecturally fixed encoder context length — the audio-frame count
/// after the stride-2 conv downsample. Whisper's front-end pads **every**
/// segment to the constant [`N_FRAMES`] (`3000`) mel frames before the encoder
/// runs (the 30-second chunk), and `conv2`'s stride 2 halves that to
/// `N_FRAMES / CONV_DOWNSAMPLE` (`1500`) regardless of the checkpoint. So
/// `n_audio_ctx` is not a free hyperparameter: it must equal this value, and
/// [`ModelDimensions::build`] pins it (a deviating config is a different,
/// unsupported architecture). Pinning it also makes the encoder `conv1`
/// pre-downsample activation cap provably equal the **runtime** extent
/// `N_FRAMES * n_audio_state` — not a config-derived `2 * n_audio_ctx *
/// n_audio_state` that a mismatched `n_audio_ctx` could undercount.
const N_AUDIO_CTX_FIXED: usize = N_FRAMES / CONV_DOWNSAMPLE;

/// Upper bound on the encoder / decoder transformer **layer counts**
/// (`n_audio_layer` / `n_text_layer`), enforced at construction by
/// [`require_cardinality`]. Unlike the width dimensions, a layer count sizes an
/// eager per-layer allocation — the encoder / decoder block `Vec`s and the
/// decoder KV-cache `Vec`s — so even a tiny-width config with millions of layers
/// would drive those reservations toward an out-of-memory abort before any
/// product cap fires. The largest released Whisper checkpoint (large-v3) has 32
/// layers; this cap (`1 << 12 = 4096`) clears it by two orders of magnitude
/// (covering any plausible future scale-up) while keeping the per-layer
/// reservation count far below an allocation that could not be served, and well
/// under the `MAX_DIM` magnitude the field's [`Extent`] already enforces.
const MAX_LAYERS: u64 = 1 << 12;

/// The one-sided STFT spectrum width fed to the mel filterbank — `N_FFT / 2 + 1`
/// for Whisper's fixed `N_FFT = 400` (see [`super::audio::N_FFT`]). The mel
/// filterbank the encoder front-end materializes is `(n_mels, N_FREQS)`, so its
/// element count scales with `n_mels` against this constant.
const N_FREQS: usize = super::audio::N_FFT / 2 + 1;

/// Whisper model hyperparameters — `ModelDimensions` (`whisper.py:269-280`).
///
/// The ten fields mirror the reference dataclass exactly. Each is stored as an
/// [`Extent`] — a dimension capped at construction (`<= MAX_DIM`) — so an
/// over-cap dimension cannot be represented by a built [`ModelDimensions`], and
/// every tensor extent the model materializes is the [`elem_count`] product of
/// these already-bounded axes against a per-extent element cap. Construct via
/// [`ModelDimensions::from_dict`] (which validates eagerly) or
/// [`ModelDimensions::new`] (which also validates); the fields are private so
/// no unvalidated instance can be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDimensions {
  /// Number of mel filterbank bins fed to the encoder (`80` for tiny..large-v2,
  /// `128` for large-v3 / turbo).
  n_mels: Extent,
  /// Encoder context length — the number of audio frames after the stride-2
  /// conv downsample. Architecturally fixed at [`N_AUDIO_CTX_FIXED`] (`1500`,
  /// = `N_FRAMES / CONV_DOWNSAMPLE`) and pinned at construction.
  n_audio_ctx: Extent,
  /// Encoder hidden width (`d_model`; `384` tiny .. `1280` large).
  n_audio_state: Extent,
  /// Encoder attention head count.
  n_audio_head: Extent,
  /// Number of encoder transformer blocks.
  n_audio_layer: Extent,
  /// Vocabulary size (`51864` for `*.en`, `>= 51865` multilingual).
  n_vocab: Extent,
  /// Decoder context length — the maximum decoded token count (`448`).
  n_text_ctx: Extent,
  /// Decoder hidden width (equal to [`Self::n_audio_state`] in every Whisper
  /// checkpoint).
  n_text_state: Extent,
  /// Decoder attention head count.
  n_text_head: Extent,
  /// Number of decoder transformer blocks.
  n_text_layer: Extent,
}

impl ModelDimensions {
  /// Construct from all ten fields, validating eagerly
  /// ([`Self::validate`]).
  ///
  /// # Errors
  /// Any of the [`Self::validate`] constraints (zero / oversized field,
  /// `n_state` not divisible by `n_head`).
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
  ) -> Result<Self> {
    Self::build(RawDims {
      n_mels,
      n_audio_ctx,
      n_audio_state,
      n_audio_head,
      n_audio_layer,
      n_vocab,
      n_text_ctx,
      n_text_state,
      n_text_head,
      n_text_layer,
    })
  }

  /// Create [`ModelDimensions`] from a parsed `config.json`, validating
  /// eagerly. Faithful port of `ModelDimensions.from_dict`
  /// (`whisper.py:282-312`): a config carrying `d_model` or `encoder_layers`
  /// is treated as the HuggingFace transformers layout and mapped field-by-
  /// field (with the reference's defaults for absent keys); otherwise it is
  /// the native MLX layout (the `model_type` / `quantization` keys are
  /// ignored, every other recognized key is read).
  ///
  /// # Errors
  /// - [`Error::MissingField`] if a required MLX-format field is absent (the
  ///   reference would raise `TypeError` from the `cls(**filtered)` call);
  /// - the [`Self::validate`] constraints on the resolved dimensions.
  pub fn from_dict(config: &Value) -> Result<Self> {
    let obj = config.as_object().ok_or_else(|| {
      Error::Parse(ParsePayload::new(
        "ModelDimensions::from_dict",
        "JSON object",
        "config root must be a JSON object",
      ))
    })?;

    // HuggingFace format iff `d_model` or `encoder_layers` is present
    // (`whisper.py:292`).
    let is_hf = obj.contains_key("d_model") || obj.contains_key("encoder_layers");

    let raw = if is_hf {
      // Map HF → MLX, using the reference's defaults for absent keys
      // (`whisper.py:294-305`). `d_model` backs both the audio and text
      // state width.
      let d_model = hf_field(obj, "d_model", 1280)?;
      RawDims {
        n_mels: hf_field(obj, "num_mel_bins", 128)?,
        n_audio_ctx: hf_field(obj, "max_source_positions", 1500)?,
        n_audio_state: d_model,
        n_audio_head: hf_field(obj, "encoder_attention_heads", 20)?,
        n_audio_layer: hf_field(obj, "encoder_layers", 32)?,
        n_vocab: hf_field(obj, "vocab_size", 51866)?,
        n_text_ctx: hf_field(obj, "max_target_positions", 448)?,
        n_text_state: d_model,
        n_text_head: hf_field(obj, "decoder_attention_heads", 20)?,
        n_text_layer: hf_field(obj, "decoder_layers", 32)?,
      }
    } else {
      // Native MLX format: every field is required (the reference's
      // `cls(**filtered)` raises if one is missing). `model_type` /
      // `quantization` are ignored.
      RawDims {
        n_mels: mlx_field(obj, "n_mels")?,
        n_audio_ctx: mlx_field(obj, "n_audio_ctx")?,
        n_audio_state: mlx_field(obj, "n_audio_state")?,
        n_audio_head: mlx_field(obj, "n_audio_head")?,
        n_audio_layer: mlx_field(obj, "n_audio_layer")?,
        n_vocab: mlx_field(obj, "n_vocab")?,
        n_text_ctx: mlx_field(obj, "n_text_ctx")?,
        n_text_state: mlx_field(obj, "n_text_state")?,
        n_text_head: mlx_field(obj, "n_text_head")?,
        n_text_layer: mlx_field(obj, "n_text_layer")?,
      }
    };

    Self::build(raw)
  }

  /// Build a validated [`ModelDimensions`] from ten raw `usize` fields.
  ///
  /// The single validating constructor behind [`Self::new`] and
  /// [`Self::from_dict`]. Each field becomes an [`Extent`] (capped at
  /// `MAX_DIM` at construction, so an over-cap dimension is rejected as
  /// [`Error::CapExceeded`] and an over-cap value can never be stored), then is
  /// additionally required strictly positive ([`require_positive`], a zero
  /// dimension is [`Error::OutOfRange`]) — together the per-field cardinality
  /// guard. The architecturally fixed `n_audio_ctx` is then pinned to
  /// [`N_AUDIO_CTX_FIXED`], and the two layer counts are bounded by
  /// [`MAX_LAYERS`] (the [`require_cardinality`] guard that keeps an eager
  /// per-layer allocation from over-reserving). The state/head divisibility and
  /// every config-derived product extent are checked by [`Self::validate`] on
  /// the assembled value. Whisper spans many checkpoints (`tiny`..`large-v3`),
  /// so the remaining dims are bounded by cap, not pinned to a single value.
  ///
  /// # Errors
  /// - [`Error::CapExceeded`] / [`Error::OutOfRange`] from the per-field
  ///   [`Extent`] / positivity guard;
  /// - [`Error::OutOfRange`] if `n_audio_ctx` does not equal
  ///   [`N_AUDIO_CTX_FIXED`];
  /// - [`Error::CapExceeded`] if `n_audio_layer` or `n_text_layer` exceeds
  ///   [`MAX_LAYERS`];
  /// - the [`Self::validate`] divisibility / product-extent constraints.
  fn build(raw: RawDims) -> Result<Self> {
    let dims = Self {
      n_mels: dim("n_mels", raw.n_mels)?,
      n_audio_ctx: dim("n_audio_ctx", raw.n_audio_ctx)?,
      n_audio_state: dim("n_audio_state", raw.n_audio_state)?,
      n_audio_head: dim("n_audio_head", raw.n_audio_head)?,
      n_audio_layer: dim("n_audio_layer", raw.n_audio_layer)?,
      n_vocab: dim("n_vocab", raw.n_vocab)?,
      n_text_ctx: dim("n_text_ctx", raw.n_text_ctx)?,
      n_text_state: dim("n_text_state", raw.n_text_state)?,
      n_text_head: dim("n_text_head", raw.n_text_head)?,
      n_text_layer: dim("n_text_layer", raw.n_text_layer)?,
    };

    // Pin the architecturally fixed encoder context. Whisper pads every segment
    // to N_FRAMES before the encoder and conv2 halves that to
    // N_FRAMES / CONV_DOWNSAMPLE, so a config whose n_audio_ctx differs is an
    // unsupported architecture — and, left unpinned, would let the conv1
    // activation cap (computed below from the real N_FRAMES extent) bound a
    // tensor the encoder never builds while the encoder materializes a
    // differently-sized one. Reject the mismatch naming the field.
    pin_usize("n_audio_ctx", dims.n_audio_ctx.get(), N_AUDIO_CTX_FIXED)?;

    // Bound the layer counts before they size the eager per-layer block / KV
    // cache reservations (a tiny-width, millions-of-layers config would
    // otherwise over-reserve toward an out-of-memory abort). Each is already
    // strictly positive (the `dim` guard) and `<= MAX_DIM`; this additionally
    // caps it at the tighter MAX_LAYERS as CapExceeded.
    require_cardinality("n_audio_layer", dims.n_audio_layer.get() as i64, MAX_LAYERS)?;
    require_cardinality("n_text_layer", dims.n_text_layer.get() as i64, MAX_LAYERS)?;

    dims.validate()?;
    Ok(dims)
  }

  /// Re-check the divisibility and product-extent constraints on an assembled
  /// [`ModelDimensions`].
  ///
  /// The per-field `MAX_DIM` cap and the strict-positive guard are already
  /// enforced when each field's [`Extent`] is built in `Self::build`, so a
  /// stored [`ModelDimensions`] cannot hold an over-cap or zero dimension. This
  /// re-checks the remaining classes idempotently: each state width must be
  /// divisible by its head count (otherwise the per-head reshape
  /// `n_state -> (n_head, head_dim)` inside attention is ill-defined); the
  /// encoder and decoder hidden widths must be EQUAL (`n_audio_state ==
  /// n_text_state`, the cross-attention bridge the crate has no adapter for); and
  /// every tensor extent the model materializes — a *product* of two or three
  /// already-bounded axes — must stay within its element cap (two individually
  /// `<= MAX_DIM` axes can still multiply to an out-of-memory extent). The
  /// reference performs no validation; this turns a malformed `config.json` into
  /// a typed error at construction rather than a downstream reshape / OOM.
  ///
  /// Each product flows through [`elem_count`] over the stored [`Extent`] axes:
  /// the running product is computed overflow-checked in `usize` (a genuinely
  /// `usize`-overflowing product is [`Error::ArithmeticOverflow`]) and rejected
  /// against the per-extent element cap as [`Error::CapExceeded`] before any
  /// tensor is built. The complete set of config-derived extents (no model
  /// allocation is left uncapped):
  ///
  /// | extent (site) | product | cap |
  /// |---|---|---|
  /// | encoder positional embedding | `n_audio_ctx * n_audio_state` | `DENSE_2D_ELEM_CAP` |
  /// | decoder positional embedding | `n_text_ctx * n_text_state` | `DENSE_2D_ELEM_CAP` |
  /// | decoder causal mask | `n_text_ctx * n_text_ctx` | `DENSE_2D_ELEM_CAP` |
  /// | encoder conv1 pre-downsample activation | `N_FRAMES * n_audio_state` | `DENSE_2D_ELEM_CAP` |
  /// | mel filterbank | `n_mels * n_freqs` | `MEL_FILTER_ELEM_CAP` |
  /// | encoder self-attention scores | `n_audio_head * n_audio_ctx * n_audio_ctx` | `ATTN_SCORE_ELEM_CAP` |
  /// | decoder self-attention scores | `n_text_head * n_text_ctx * n_text_ctx` | `ATTN_SCORE_ELEM_CAP` |
  /// | cross-attention scores | `n_text_head * n_text_ctx * n_audio_ctx` | `ATTN_SCORE_ELEM_CAP` |
  /// | encoder MLP hidden | `n_audio_ctx * 4 * n_audio_state` | `MLP_HIDDEN_ELEM_CAP` |
  /// | decoder MLP hidden | `n_text_ctx * 4 * n_text_state` | `MLP_HIDDEN_ELEM_CAP` |
  /// | token-embedding / tied-logit table | `n_vocab * n_text_state` | `VOCAB_PROJ_ELEM_CAP` |
  /// | decoder logits | `n_text_ctx * n_vocab` | `VOCAB_PROJ_ELEM_CAP` |
  /// | decoder self-attention KV cache | `n_text_layer * n_text_ctx * n_text_state` | `KV_CACHE_ELEM_CAP` |
  /// | cross-attention KV cache | `n_text_layer * n_audio_ctx * n_audio_state` | `KV_CACHE_ELEM_CAP` |
  ///
  /// # Errors
  /// - [`Error::DivisibilityConstraint`] if `n_audio_state % n_audio_head != 0`
  ///   or `n_text_state % n_text_head != 0`;
  /// - [`Error::OutOfRange`] if `n_audio_state` is odd or `< 4` (the encoder
  ///   sinusoidal positional-embedding precondition — even, and
  ///   `n_audio_state/2 >= 2`), or if `n_text_state != n_audio_state` (the
  ///   encoder / decoder hidden widths must be equal for cross-attention);
  /// - [`Error::CapExceeded`] if a derived tensor's element count exceeds its
  ///   product cap;
  /// - [`Error::ArithmeticOverflow`] if a derived product overflows `usize`
  ///   (a per-field-valid but extreme config).
  pub fn validate(&self) -> Result<()> {
    // Each state/head pair is bounded to `1..=MAX_DIM` (<= i32::MAX), so the
    // `as i32` conversions are lossless. `require_divisible` guards the divisor
    // and returns `DivisibilityConstraint` on a non-multiple.
    require_divisible(
      "n_audio_state",
      self.n_audio_state.get() as i32,
      "n_audio_head",
      self.n_audio_head.get() as i32,
    )?;
    require_divisible(
      "n_text_state",
      self.n_text_state.get() as i32,
      "n_text_head",
      self.n_text_head.get() as i32,
    )?;

    // The encoder positional embedding is `sinusoids(n_audio_ctx,
    // n_audio_state)`, whose `inv_timescales` divides by `n_audio_state/2 - 1`.
    // So `n_audio_state` must be even (the `concat([sin, cos])` halves the width)
    // and at least MIN_AUDIO_STATE_FOR_SINUSOID (= 4, so `n_audio_state/2 - 1 >=
    // 1`): an odd width never halves cleanly, and width 2 makes the divisor 0,
    // producing a `+inf` increment and a `0 * inf` NaN positional row that would
    // surface only AFTER `AudioEncoder::new` consumes weights. Pinning the
    // precondition here turns those degenerate small widths into a typed error at
    // config time. `n_audio_state.get() <= MAX_DIM` (`1 << 22`), so the `as i32`
    // is lossless and non-negative.
    let n_audio_state_i32 = self.n_audio_state.get() as i32;
    require_even("n_audio_state", n_audio_state_i32)?;
    require_in_range(
      "n_audio_state",
      n_audio_state_i32,
      MIN_AUDIO_STATE_FOR_SINUSOID,
      MAX_DIM as i32,
    )?;

    // A small fixed multiplier axis (`<= MAX_DIM`) reused across product checks:
    // the MLP expansion ratio and the fixed pre-downsample frame count. Built as
    // an `Extent` so it composes with the config axes inside `elem_count`. The
    // conv1 activation runs on the constant `N_FRAMES` frames (the padded
    // segment), not a config-derived count, so its axis is `N_FRAMES` directly.
    let mlp_ratio = Extent::new("MLP_RATIO", MLP_RATIO, MAX_DIM)?;
    let n_frames = Extent::new("N_FRAMES", N_FRAMES, MAX_DIM)?;
    let n_freqs = Extent::new("n_freqs", N_FREQS, MAX_DIM)?;

    // Bound every dense tensor whose extent is a PRODUCT of two stored axes.
    // `elem_count` computes the running product overflow-checked in `usize` and
    // rejects an out-of-memory extent against the cap as CapExceeded.
    elem_count(
      "n_audio_ctx * n_audio_state (encoder positional embedding)",
      &[self.n_audio_ctx, self.n_audio_state],
      DENSE_2D_ELEM_CAP,
    )?;
    elem_count(
      "n_text_ctx * n_text_state (decoder positional embedding)",
      &[self.n_text_ctx, self.n_text_state],
      DENSE_2D_ELEM_CAP,
    )?;
    elem_count(
      "n_text_ctx * n_text_ctx (decoder causal mask)",
      &[self.n_text_ctx, self.n_text_ctx],
      DENSE_2D_ELEM_CAP,
    )?;
    elem_count(
      "n_mels * n_freqs (mel filterbank)",
      &[self.n_mels, n_freqs],
      MEL_FILTER_ELEM_CAP,
    )?;

    // The encoder `conv1` runs on the FIXED padded frame count `N_FRAMES`
    // (every segment is padded to it before the encoder; `n_audio_ctx` is pinned
    // to `N_FRAMES / CONV_DOWNSAMPLE`, so this is the real runtime extent, not a
    // config-derived `2 * n_audio_ctx` a mismatched `n_audio_ctx` could
    // undercount), before `conv2`'s stride-2 halves it. Its activation is
    // `N_FRAMES * n_audio_state`.
    elem_count(
      "N_FRAMES * n_audio_state (encoder conv1 pre-downsample activation)",
      &[n_frames, self.n_audio_state],
      DENSE_2D_ELEM_CAP,
    )?;

    // Bound every 3-D attention-score tensor `(1, n_head, T_q, T_kv)` the
    // forward pass materializes: the dense `q @ kᵀ` extent is the THREE-way
    // product `n_head * T_q * T_kv`.
    elem_count(
      "n_audio_head * n_audio_ctx * n_audio_ctx (encoder self-attention scores)",
      &[self.n_audio_head, self.n_audio_ctx, self.n_audio_ctx],
      ATTN_SCORE_ELEM_CAP,
    )?;
    elem_count(
      "n_text_head * n_text_ctx * n_text_ctx (decoder self-attention scores)",
      &[self.n_text_head, self.n_text_ctx, self.n_text_ctx],
      ATTN_SCORE_ELEM_CAP,
    )?;
    elem_count(
      "n_text_head * n_text_ctx * n_audio_ctx (cross-attention scores)",
      &[self.n_text_head, self.n_text_ctx, self.n_audio_ctx],
      ATTN_SCORE_ELEM_CAP,
    )?;

    // Bound each transformer-block MLP hidden activation `(1, T, 4 * n_state)`:
    // the dense extent is `T * MLP_RATIO * n_state`.
    elem_count(
      "n_audio_ctx * 4 * n_audio_state (encoder MLP hidden)",
      &[self.n_audio_ctx, mlp_ratio, self.n_audio_state],
      MLP_HIDDEN_ELEM_CAP,
    )?;
    elem_count(
      "n_text_ctx * 4 * n_text_state (decoder MLP hidden)",
      &[self.n_text_ctx, mlp_ratio, self.n_text_state],
      MLP_HIDDEN_ELEM_CAP,
    )?;

    // Bound the weight-tied logit head: the token-embedding / tied-logit table
    // `(n_vocab, n_text_state)` and the per-step logits `(1, n_text_ctx,
    // n_vocab)`.
    elem_count(
      "n_vocab * n_text_state (token-embedding / tied-logit table)",
      &[self.n_vocab, self.n_text_state],
      VOCAB_PROJ_ELEM_CAP,
    )?;
    elem_count(
      "n_text_ctx * n_vocab (decoder logits)",
      &[self.n_text_ctx, self.n_vocab],
      VOCAB_PROJ_ELEM_CAP,
    )?;

    // Bound the cumulative KV cache each decode grows up to the full context:
    // one `(1, T, n_state)` key (and an equal value) per block, so the
    // cumulative key extent is `n_layer * T * n_state`. The decoder
    // self-attention cache and the cross-attention cache each get a bound.
    elem_count(
      "n_text_layer * n_text_ctx * n_text_state (decoder self-attention KV cache)",
      &[self.n_text_layer, self.n_text_ctx, self.n_text_state],
      KV_CACHE_ELEM_CAP,
    )?;
    elem_count(
      "n_text_layer * n_audio_ctx * n_audio_state (cross-attention KV cache)",
      &[self.n_text_layer, self.n_audio_ctx, self.n_audio_state],
      KV_CACHE_ELEM_CAP,
    )?;

    // The encoder and decoder hidden widths must be EQUAL. The decoder's
    // cross-attention projects the encoder states `(1, n_audio_ctx,
    // n_audio_state)` through square `(n_text_state, n_text_state)` query/key/
    // value/out weights, so a config whose `n_audio_state != n_text_state` builds
    // a shape-valid encoder and decoder that nonetheless cannot bridge: the
    // cross-attention `key`/`value` matmul `xa @ Wᵀ` would contract a
    // `n_audio_state`-wide last axis against a `n_text_state`-wide weight. The
    // crate carries no unequal-width adapter (every released Whisper checkpoint
    // has `n_audio_state == n_text_state`), and the cross-attention KV cache cap
    // above is stated with `n_audio_state` while the projected width would be the
    // decoder width — so reject the mismatch (a typed `OutOfRange` naming the
    // field) rather than let it surface as a downstream matmul shape error.
    // `n_audio_state` is the reference width the decoder must match. Checked
    // after the product caps so a config that is BOTH unequal-width and over-cap
    // still reports the (more specific) extent it exceeds first; a config valid
    // in every magnitude but unequal-width is caught here. No allocation occurs
    // anywhere in `validate`, so this ordering is a diagnostics choice, not a
    // resource-safety one.
    pin_usize(
      "n_text_state",
      self.n_text_state.get(),
      self.n_audio_state.get(),
    )?;
    Ok(())
  }

  /// `true` iff the vocabulary is multilingual (`n_vocab >= 51865`) —
  /// `Model.is_multilingual` (`whisper.py:621-623`).
  #[inline(always)]
  pub fn is_multilingual(&self) -> bool {
    self.n_vocab.get() >= MULTILINGUAL_VOCAB_THRESHOLD
  }

  /// Number of language tokens — `Model.num_languages`
  /// (`whisper.py:625-627`): `n_vocab - 51765 - (is_multilingual as usize)`.
  /// Saturates at `0` for a (non-multilingual) `n_vocab <= 51765`.
  #[inline(always)]
  pub fn num_languages(&self) -> usize {
    self
      .n_vocab
      .get()
      .saturating_sub(51765)
      .saturating_sub(self.is_multilingual() as usize)
  }

  /// Number of mel filterbank bins.
  #[inline(always)]
  pub fn n_mels(&self) -> usize {
    self.n_mels.get()
  }

  /// Encoder context length (audio frames after downsampling).
  #[inline(always)]
  pub fn n_audio_ctx(&self) -> usize {
    self.n_audio_ctx.get()
  }

  /// Encoder hidden width.
  #[inline(always)]
  pub fn n_audio_state(&self) -> usize {
    self.n_audio_state.get()
  }

  /// Encoder attention head count.
  #[inline(always)]
  pub fn n_audio_head(&self) -> usize {
    self.n_audio_head.get()
  }

  /// Number of encoder transformer blocks.
  #[inline(always)]
  pub fn n_audio_layer(&self) -> usize {
    self.n_audio_layer.get()
  }

  /// Vocabulary size.
  #[inline(always)]
  pub fn n_vocab(&self) -> usize {
    self.n_vocab.get()
  }

  /// Decoder context length (maximum decoded token count).
  #[inline(always)]
  pub fn n_text_ctx(&self) -> usize {
    self.n_text_ctx.get()
  }

  /// Decoder hidden width.
  #[inline(always)]
  pub fn n_text_state(&self) -> usize {
    self.n_text_state.get()
  }

  /// Decoder attention head count.
  #[inline(always)]
  pub fn n_text_head(&self) -> usize {
    self.n_text_head.get()
  }

  /// Number of decoder transformer blocks.
  #[inline(always)]
  pub fn n_text_layer(&self) -> usize {
    self.n_text_layer.get()
  }

  /// The default word-timing alignment heads — every `(layer, head)` in the
  /// **last half** of the decoder layers (`whisper.py:510-516`):
  /// `all_heads[n_text_layer // 2 :] = True`, enumerated in
  /// `(layer, head)` row-major order (numpy `nonzero().T` over a
  /// `(n_text_layer, n_text_head)` boolean mask).
  ///
  /// These are used by the word-timestamp DTW when a checkpoint ships no
  /// explicit `alignment_heads` in its `generation_config.json` (the override
  /// is parsed by [`AlignmentHeads::from_generation_config`]).
  pub fn default_alignment_heads(&self) -> Vec<(usize, usize)> {
    let layers = self.n_text_layer();
    let heads = self.n_text_head();
    // `n_text_layer // 2` — the first layer index in the last half.
    let first = layers / 2;
    let mut out = Vec::with_capacity((layers - first) * heads);
    for l in first..layers {
      for h in 0..heads {
        out.push((l, h));
      }
    }
    out
  }
}

/// The word-timing alignment heads — the `(layer, head)` cross-attention heads
/// the word-timestamp DTW averages to align text tokens to audio frames
/// (`whisper.py:_alignment_heads`).
///
/// The reference stores these as an `mx.array` of shape `(num_heads, 2)` (each
/// row `[layer, head]`); this carries the equivalent `Vec<(layer, head)>`. The
/// default is the **last half** of the decoder layers
/// ([`ModelDimensions::default_alignment_heads`]); a checkpoint can override it
/// through its `generation_config.json` (`set_alignment_heads`,
/// `whisper.py:522-537`), parsed by [`Self::from_generation_config`].
///
/// Alongside the pairs this carries the `(n_layer, n_head)` decoder grid the
/// list was validated against, so the value has a dimension identity. The pairs
/// alone cannot say *which* model they are in-grid for; a set validated against
/// a larger model's grid would carry indices out of a smaller model's grid.
/// [`WhisperModel::with_alignment_heads`](super::model::WhisperModel::with_alignment_heads)
/// checks the carried `(n_layer, n_head)` equals the target model's
/// `(n_text_layer, n_text_head)`, so a set validated for one model cannot be
/// installed on another and reach the word-timestamp DTW gather out of grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentHeads {
  heads: Vec<(usize, usize)>,
  n_layer: usize,
  n_head: usize,
}

impl AlignmentHeads {
  /// Wrap an explicit `(layer, head)` list **without validation**, tagged with
  /// the `(n_layer, n_head)` grid it belongs to — for the crate's internal
  /// default construction only ([`Self::default_for`], which derives the list
  /// from validated `dims` so it is in-grid + distinct by construction). The
  /// public, validating constructor is [`Self::try_new`].
  #[inline(always)]
  pub(crate) fn new(heads: Vec<(usize, usize)>, n_layer: usize, n_head: usize) -> Self {
    Self {
      heads,
      n_layer,
      n_head,
    }
  }

  /// Wrap an explicit `(layer, head)` list, validated against `dims` — the only
  /// public way to build a custom [`AlignmentHeads`]. Each pair must lie in the
  /// decoder grid (`layer < n_text_layer`, `head < n_text_head`) and the list
  /// must contain no duplicate pair; on any violation the shared in-grid +
  /// no-duplicate check's typed error is returned. The validated `dims`'
  /// `(n_text_layer, n_text_head)` is recorded on the value, so the returned
  /// [`AlignmentHeads`] carries the grid it is in-grid for. The list can only be
  /// installed on a [`WhisperModel`](super::model::WhisperModel) whose dims match
  /// that grid (the install rechecks), so no out-of-grid or duplicate head set —
  /// nor a set validated for a *different* model — can ever reach the
  /// word-timestamp DTW.
  ///
  /// # Errors
  /// - [`Error::EmptyInput`] if `heads` is empty;
  /// - [`Error::OutOfRange`] if a pair is outside the model's
  ///   `(n_text_layer, n_text_head)` grid, or if the same pair appears twice;
  /// - [`Error::AllocFailure`] if reserving the dedup bitset fails.
  pub fn try_new(heads: Vec<(usize, usize)>, dims: &ModelDimensions) -> Result<Self> {
    let n_layer = dims.n_text_layer();
    let n_head = dims.n_text_head();
    Ok(Self::new(
      validate_alignment_heads(heads, n_layer, n_head)?,
      n_layer,
      n_head,
    ))
  }

  /// The default alignment heads for `dims` — the last half of the decoder
  /// layers ([`ModelDimensions::default_alignment_heads`]). The list is in-grid
  /// and distinct by construction (derived from the validated `dims`), so it is
  /// wrapped through the unchecked `Self::new`, tagged with the same `dims` grid.
  #[inline(always)]
  pub fn default_for(dims: &ModelDimensions) -> Self {
    Self::new(
      dims.default_alignment_heads(),
      dims.n_text_layer(),
      dims.n_text_head(),
    )
  }

  /// The `(layer, head)` pairs.
  #[inline(always)]
  pub fn heads(&self) -> &[(usize, usize)] {
    &self.heads
  }

  /// The decoder layer count (`n_text_layer`) the heads were validated against.
  #[inline(always)]
  pub fn n_layer(&self) -> usize {
    self.n_layer
  }

  /// The decoder head count (`n_text_head`) the heads were validated against.
  #[inline(always)]
  pub fn n_head(&self) -> usize {
    self.n_head
  }

  /// Parse the alignment heads from a parsed `generation_config.json` body —
  /// the reference's `set_alignment_heads(gen_config["alignment_heads"])`
  /// (`whisper.py:704-715`, `:522-537`). Returns `None` when the config carries
  /// no `alignment_heads` key (the caller then keeps the
  /// [`Self::default_for`] last-half default).
  ///
  /// The JSON-reachable form is a **list of `[layer, head]` pairs** (the
  /// `isinstance(dump, list)` branch — a `generation_config.json` cannot embed
  /// the raw-bytes bitmask). Each pair is parsed **as `usize`** (no intermediate
  /// `i32` cast, so a pathological index `> i32::MAX` cannot wrap into an
  /// in-range `i32`), then the whole list is run through the shared in-grid +
  /// no-duplicate validation (`validate_alignment_heads`), so neither an
  /// out-of-grid value nor a repeated pair can survive as an out-of-bounds index
  /// at DTW time.
  ///
  /// # Errors
  /// - [`Error::Parse`] if `alignment_heads` is present but not a list of
  ///   two-element non-negative-integer pairs;
  /// - [`Error::EmptyInput`] if `alignment_heads` is present but an empty list;
  /// - [`Error::OutOfRange`] if a pair's layer / head is outside the model's
  ///   `(n_text_layer, n_text_head)` grid, if the same pair appears twice, or if
  ///   the list has more entries than the grid (a valid in-grid, distinct list
  ///   cannot exceed `n_text_layer * n_text_head`);
  /// - [`Error::AllocFailure`] if reserving the head `Vec` fails.
  pub fn from_generation_config(config: &Value, dims: &ModelDimensions) -> Result<Option<Self>> {
    let Some(value) = config.get("alignment_heads") else {
      return Ok(None);
    };
    let list = value.as_array().ok_or_else(|| {
      Error::Parse(ParsePayload::new(
        "Whisper alignment_heads",
        "generation_config.json",
        "alignment_heads must be a list of [layer, head] pairs",
      ))
    })?;

    let n_layer = dims.n_text_layer();
    let n_head = dims.n_text_head();
    // In-grid + no-duplicate makes the valid list bounded by the decoder grid
    // (`n_layer * n_head`); both are validated dims (>= 1), so the product is the
    // logical maximum number of distinct heads. Reserve to that bound through the
    // fallible path — no arbitrary magnitude cap beyond the grid.
    let grid = n_layer * n_head;
    // A valid list is in-grid AND has no duplicate, so it cannot be longer than
    // the grid; reject an over-long list up front so the parse loop's pushes stay
    // bounded by the reserved grid capacity (no growth past it via `Vec`'s
    // infallible path). This is the logical grid bound, not a magnitude cap.
    if list.len() > grid {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Whisper alignment_heads",
        "list cannot have more entries than the decoder grid (n_text_layer * n_text_head)",
        format_smolstr!("{} (grid {grid})", list.len()),
      )));
    }
    let mut parsed: Vec<(usize, usize)> = Vec::new();
    reserve_or_error(&mut parsed, "Whisper alignment_heads", grid)?;
    for pair in list {
      let pair = pair.as_array().filter(|p| p.len() == 2).ok_or_else(|| {
        Error::Parse(ParsePayload::new(
          "Whisper alignment_heads",
          "generation_config.json",
          "each alignment_heads entry must be a [layer, head] pair",
        ))
      })?;
      // Parse each index as `usize` (no `i32` cast). The in-grid + no-duplicate
      // validation is deferred to the shared helper below, the single source of
      // truth for a valid alignment-head list.
      parsed.push((alignment_index(&pair[0])?, alignment_index(&pair[1])?));
    }
    Ok(Some(Self::new(
      validate_alignment_heads(parsed, n_layer, n_head)?,
      n_layer,
      n_head,
    )))
  }
}

/// Validate an alignment-head `(layer, head)` list against the decoder grid —
/// the single source of truth for "a valid alignment-head list", shared by
/// [`AlignmentHeads::try_new`] and [`AlignmentHeads::from_generation_config`].
///
/// The list must be non-empty (the DTW averages over the heads; an empty set
/// has no analogue in `whisper.py`, whose default is the last-half of the
/// layers). Each pair must lie in the half-open grid (`layer < n_layer`,
/// `head < n_head`), compared in `usize` so a pathological value `> i32::MAX`
/// cannot wrap into an in-range `i32` (the bug an `as i32` cast would
/// introduce). Duplicates are rejected through an `O(1)` per-pair lookup into a
/// grid-sized seen bitset (no `O(n^2)` linear scan): with both dims validated
/// (`>= 1`), `grid = n_layer * n_head` is the logical maximum number of distinct
/// heads, and a duplicate maps to an already-set slot. The bitset is reserved
/// through the fallible [`reserve_or_error`] path — no magnitude cap beyond the
/// logical grid. Returns the validated list unchanged.
///
/// # Errors
/// - [`Error::EmptyInput`] if `heads` is empty;
/// - [`Error::OutOfRange`] if a pair is outside the `(n_layer, n_head)` grid, or
///   if the same pair appears twice;
/// - [`Error::AllocFailure`] if reserving the dedup bitset fails.
fn validate_alignment_heads(
  heads: Vec<(usize, usize)>,
  n_layer: usize,
  n_head: usize,
) -> Result<Vec<(usize, usize)>> {
  if heads.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "Whisper alignment_heads",
    )));
  }
  let grid = n_layer * n_head;
  // Grid-sized seen bitset for O(1) duplicate detection (replaces an O(n^2)
  // `contains` scan). Reserved through the fallible path against the logical
  // grid bound — no arbitrary magnitude cap.
  let mut seen: Vec<bool> = Vec::new();
  reserve_or_error(&mut seen, "Whisper alignment_heads dedup", grid)?;
  seen.resize(grid, false);
  for &(layer, head) in &heads {
    // Validate against the decoder grid BEFORE indexing the bitset, so the DTW's
    // `cross_qk[layer][0, head]` gather can never index out of bounds. The grid
    // is `[0, n)` exclusive.
    require_index_below("Whisper alignment_heads layer", layer, n_layer)?;
    require_index_below("Whisper alignment_heads head", head, n_head)?;
    // `layer < n_layer` and `head < n_head` ⇒ `idx < n_layer * n_head == grid`,
    // so the bitset index is always in bounds.
    let idx = layer * n_head + head;
    if seen[idx] {
      // Reject a repeated pair (the reference's heads are a distinct set); a
      // duplicate would inflate the alignment-head stack with no analogue in
      // `whisper.py`.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "Whisper alignment_heads",
        "each [layer, head] pair must be unique",
        format_smolstr!("({layer}, {head})"),
      )));
    }
    seen[idx] = true;
  }
  Ok(heads)
}

/// Parse one non-negative-integer alignment-head index from a JSON value.
fn alignment_index(v: &Value) -> Result<usize> {
  v.as_u64()
    .and_then(|n| usize::try_from(n).ok())
    .ok_or_else(|| {
      Error::Parse(ParsePayload::new(
        "Whisper alignment_heads",
        "generation_config.json",
        "layer / head must be a non-negative integer",
      ))
    })
}

/// Require a `usize` index to lie in the half-open grid `[0, bound)` — the
/// alignment-head bound check.
///
/// Compared in `usize` so a pathological value `> i32::MAX` cannot wrap into an
/// in-range `i32` (the bug an `as i32` cast would introduce). `bound` is a
/// validated dimension (`>= 1`). On violation returns [`Error::OutOfRange`]
/// naming `field`, the bound, and the offending index.
fn require_index_below(field: &'static str, index: usize, bound: usize) -> Result<()> {
  if index >= bound {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      field,
      "must be in the half-open range [0, bound)",
      format_smolstr!("{index} (allowed [0, {bound}))"),
    )));
  }
  Ok(())
}

/// The ten raw `usize` Whisper dimensions, before each is lifted into a capped
/// [`Extent`]. The plain carrier both [`ModelDimensions::new`] and
/// [`ModelDimensions::from_dict`] assemble and hand to
/// [`ModelDimensions::build`], so the per-field cap + positivity guard and the
/// product-extent checks are applied in exactly one place.
#[derive(Clone, Copy)]
struct RawDims {
  n_mels: usize,
  n_audio_ctx: usize,
  n_audio_state: usize,
  n_audio_head: usize,
  n_audio_layer: usize,
  n_vocab: usize,
  n_text_ctx: usize,
  n_text_state: usize,
  n_text_head: usize,
  n_text_layer: usize,
}

/// Lift one raw config dimension into a capped [`Extent`] that is also strictly
/// positive — the per-field cardinality guard.
///
/// [`Extent::new`] enforces the `MAX_DIM` magnitude cap (rejecting an over-cap
/// value as [`Error::CapExceeded`], so it can never be stored), and
/// [`require_positive`] then rejects a zero dimension as [`Error::OutOfRange`].
/// A pathological `usize > i32::MAX` is caught by the `MAX_DIM` cap before the
/// `as i32` positivity check, so the cast is non-wrapping.
fn dim(field: &'static str, value: usize) -> Result<Extent> {
  let extent = Extent::new(field, value, MAX_DIM)?;
  require_positive(field, extent.get() as i32)?;
  Ok(extent)
}

/// Read a required native-MLX-format unsigned field. A missing key is the
/// reference's `cls(**filtered)` `TypeError`; a present-but-non-integer value
/// is a [`Error::Parse`].
fn mlx_field(obj: &serde_json::Map<String, Value>, key: &'static str) -> Result<usize> {
  match obj.get(key) {
    None => Err(Error::MissingField(MissingFieldPayload::new(
      "ModelDimensions (MLX format)",
      key,
    ))),
    Some(v) => parse_usize(v, key),
  }
}

/// Read an optional HuggingFace-format unsigned field, falling back to
/// `default` when the key is absent (mirroring the reference's
/// `config.get(key, default)`). A present-but-non-integer value is a
/// [`Error::Parse`].
fn hf_field(
  obj: &serde_json::Map<String, Value>,
  key: &'static str,
  default: usize,
) -> Result<usize> {
  match obj.get(key) {
    None => Ok(default),
    Some(Value::Null) => Ok(default),
    Some(v) => parse_usize(v, key),
  }
}

/// Coerce a JSON number to `usize`, rejecting non-integers / negatives /
/// out-of-range values with a typed [`Error::Parse`].
fn parse_usize(v: &Value, key: &'static str) -> Result<usize> {
  let n = v.as_u64().ok_or_else(|| {
    Error::Parse(ParsePayload::new(
      "ModelDimensions field",
      "non-negative integer",
      format!("field `{key}` is not a non-negative integer: {v}"),
    ))
  })?;
  usize::try_from(n).map_err(|_| {
    Error::Parse(ParsePayload::new(
      "ModelDimensions field",
      "value fits in usize",
      format!("field `{key}` = {n} overflows usize"),
    ))
  })
}

#[cfg(test)]
mod tests;
