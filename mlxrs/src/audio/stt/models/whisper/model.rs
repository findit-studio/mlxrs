//! The loadable Whisper model — `WhisperModel` (the `Model` class,
//! `whisper.py:489-630`) plus the pure-Rust `sanitize` weight remap
//! (`whisper.py:539-606`).
//!
//! Ties the [`super::config::ModelDimensions`], the `encoder` `AudioEncoder`,
//! the `decoder` `TextDecoder`, and the `layers` building blocks (all
//! crate-private) into one struct that implements the STT trait architecture:
//!
//! - [`AutoregressiveStt`] — the encoder/decoder family hooks (`encode` /
//!   `decode_step` / `new_cache` / `initial_tokens` / `eot` / `max_context` /
//!   `log_mel`), with the decode KV cache carried as the caller-owned
//!   associated [`WhisperDecodeCache`]; and
//! - [`Transcribe`] — the universal "audio in, text out" contract, run by
//!   Whisper's own [`super::decoding::transcribe`] decoding task (greedy
//!   decode, logit filters, temperature fallback, the 30-second seek loop, and
//!   language detection) rather than the generic
//!   [`greedy_transcribe`](crate::audio::stt::generate::greedy_transcribe) loop.
//!
//! The model holds NO mutable decode state: the per-block `(self_kv, cross_kv)`
//! cache is a value each generation mints fresh via
//! [`AutoregressiveStt::new_cache`] and threads by `&mut`, so one loaded model
//! backs any number of concurrent or sequential decodes without cross-utterance
//! aliasing.
//!
//! [`WhisperModel::from_weights`] runs the HF→MLX [`sanitize`] remap, then
//! [`ModelDimensions::validate`], then builds every sub-module from the
//! sanitized weight map. [`WhisperModel::load`] reads the `*.safetensors` from
//! a model directory and forwards to `from_weights`.
//!
//! ## Threat model / trust boundary
//!
//! This states precisely what the Whisper port's validation layer defends
//! against, so the surface is reviewed against a stated boundary rather than an
//! open-ended "harden everything".
//!
//! ### Trusted flow (the intended use)
//!
//! The normal transcription path — the [`Transcribe`] `transcribe` (or the
//! lower-level [`super::decoding::transcribe`]) — feeds validated,
//! in-distribution inputs through the guarded entry points: a mono waveform is
//! turned into a log-mel by [`super::audio::log_mel_spectrogram_whisper`], each
//! 30-second window is padded to the fixed `N_FRAMES` by
//! [`super::audio::pad_or_trim`], the window is encoded once (the
//! [`AutoregressiveStt`] `encode`), and the decoder is driven one token window
//! at a time (the [`AutoregressiveStt`] `decode_step`). Inputs on this path are
//! well-formed by construction; the guards below are not aimed at it.
//!
//! ### Defended (the validation layer hardens these)
//!
//! These three classes are explicitly defended — a caller hitting one gets a
//! typed error, never undefined behaviour, a silent mis-shape, or an
//! unbounded-allocation abort:
//!
//! 1. **A corrupt-but-loadable checkpoint** with mismatched tensor shapes. The
//!    [`sanitize`] step is **key-remap only** (no tensor data is transposed or
//!    cast), then the private weight `Builder` **shape-validates every consumed
//!    tensor against the config-derived extent before it is transposed / cast /
//!    materialized** (`take_shaped` / `take_conv_weight` / `check_shape`), so an
//!    oversized tensor under a consumed key is rejected with a typed
//!    [`Error::ShapePairMismatch`] (keyed by [`Error::LayerKeyed`]) ahead of any
//!    allocation it would size. Two source keys colliding onto one sanitized key
//!    is an [`Error::KeyCollision`]; a missing required weight is an
//!    [`Error::MissingKey`]. Unconsumed tensors are never materialized.
//! 2. **A malformed native config** with invalid dimension relationships.
//!    [`ModelDimensions::validate`] enforces the cross-invariants the reference
//!    does not: per-field positivity and a `MAX_DIM` cap, the architecturally
//!    fixed `n_audio_ctx` pin, the `MAX_LAYERS` cap on each layer count (which
//!    sizes an eager per-layer allocation), `n_state % n_head == 0` for both
//!    stacks, the encoder sinusoidal positional-embedding precondition
//!    (`n_audio_state` even and `>= 4`, so `sinusoids` is non-degenerate),
//!    `n_audio_state == n_text_state` (the cross-attention bridge), and an
//!    element cap on **every** config-derived tensor extent (positional
//!    embeddings, causal mask, conv1 activation, mel filterbank, attention
//!    scores, MLP hidden, vocab projection, and KV caches) — each a typed
//!    [`Error::CapExceeded`] / [`Error::OutOfRange`] /
//!    [`Error::DivisibilityConstraint`] / [`Error::ArithmeticOverflow`].
//! 3. **Direct misuse of the public `decode_step` / `encode` / `from_weights`
//!    APIs** with oversized, batched, or wrong-shape inputs. Every public entry
//!    carries an extent guard that fires before any allocation it would size:
//!    - The [`AutoregressiveStt`] `encode` (the `AudioEncoder` forward) rejects
//!      a non-`1` batch, a mel frame count other than `conv2.stride *
//!      n_audio_ctx` (= `N_FRAMES`), a mel channel width other than the
//!      configured `n_mels` (the `conv1` input-channel dimension — checked
//!      before `conv1` contracts that axis), and a post-conv shape that
//!      disagrees with the positional embedding — each a typed
//!      [`Error::ShapePairMismatch`].
//!    - The [`AutoregressiveStt`] `decode_step` (and the internal `decode_tokens`
//!      primitive) validate the encoder-states extent to exactly
//!      `(1, n_audio_ctx, n_audio_state)` (`validate_encoder_states`) **before**
//!      the decoder cross-attention projects it, reject a token prefix longer
//!      than `max_context` (`n_text_ctx`) before the token array is built, and
//!      the decoder additionally bounds `offset + seq_len` against the
//!      positional table before the embedding gather (`check_context`). The
//!      self-attention causal mask is sliced **offset-aware**, so a warm-cache
//!      multi-token step masks each new token against exactly the keys at or
//!      before its absolute position.
//!    - [`WhisperModel::from_weights`] is the checkpoint path covered by (1)
//!      above.
//!
//! ### Out of scope (semi-trusted — NOT defended here)
//!
//! The following are outside the port's threat model; a caller supplying them is
//! responsible for their trustworthiness, and the port does not attempt to
//! defend against them:
//!
//! - **A maliciously-crafted tokenizer.** The tokenizer's vocabulary and special
//!   tokens are taken as trusted configuration. (As a courtesy,
//!   [`super::decoding::detect_language`] rejects a language token id `>=
//!   n_vocab` rather than producing a `NaN` distribution, but a fully
//!   adversarial tokenizer is the caller's trust to establish.)
//! - **Adversarial audio content.** The transcription path feeds the decoder the
//!   safe, shape-checked output of the [`AutoregressiveStt`] `encode`; the
//!   *content* of the audio (what it makes the model decode) is not a resource-
//!   or memory-safety concern and is not constrained.
//! - **Resource consumption within the documented element caps.** A config /
//!   input that is valid under every cap above may still legitimately allocate up
//!   to those caps; bounding usage below the caps is the caller's policy, not the
//!   port's.
//! - **Decode-cache reuse across utterances.** This is a library, not end
//!   software: correct API usage is the consuming application's responsibility.
//!   The [`WhisperDecodeCache`] projects the cross-attention K/V from the first
//!   step's encoder states and reuses it verbatim on every warm step (ignoring
//!   each later step's `xa`), matching the reference decoder, which trusts the
//!   cross-attention states it is handed and does not store-and-compare them
//!   across steps. The library therefore does **not** detect or defend against a
//!   caller that reuses one cache across two different utterances (passing a
//!   different-content `enc` on a warm step) — doing so would silently decode the
//!   second utterance against the first's audio features. The consumer must mint
//!   a fresh cache per utterance ([`AutoregressiveStt::new_cache`]); the only
//!   pin on `enc` at the decode entry is the O(1) shape/rank check above, not a
//!   content scan.

use std::{collections::HashMap, fmt};

use crate::{
  Array, Dtype, Error, Result,
  audio::stt::model::{
    AutoregressiveStt, MelConfig, Segment, Task, Transcribe, TranscribeOptions, Transcription,
  },
  error::{
    InvariantViolationPayload, KeyCollisionPayload, LayerKeyedPayload, RankMismatchPayload,
    ShapePairMismatchPayload,
  },
  lm::{nn::norm::LayerNorm, quant::PerLayerQuantization},
  nn::{MaybeQuantizedLinear, QuantizedLinear},
  tokenizer::Tokenizer,
};

use super::{
  audio::{N_FRAMES, N_SAMPLES, log_mel_spectrogram_whisper},
  backend::WhisperBackend,
  config::{AlignmentHeads, ModelDimensions},
  decoder::{DecoderKvCache, TextDecoder},
  decoding::{self, DecodingOptions, SuppressSpec, TranscribeOptions as WhisperTranscribeOptions},
  encoder::AudioEncoder,
  layers::{Embedding, Linear, MultiHeadAttention, ResidualAttentionBlock},
  tokenizer::{HFTokenizerWrapper, Task as WhisperTask},
};

/// `nn.LayerNorm`'s default epsilon (`mlx.nn.LayerNorm(dims, eps=1e-5)`).
const LAYER_NORM_EPS: f32 = 1e-5;

/// The caller-owned Whisper decode cache — the
/// [`AutoregressiveStt::Cache`]
/// associated type.
///
/// Wraps the decoder's per-block `(self_kv, cross_kv)` cache (the crate-private
/// `DecoderKvCache`); `None` before the first step (the reference's
/// `kv_cache = [None] * len(self.blocks)`). Each generation mints a fresh one
/// via [`WhisperModel::new_cache`] and threads it by `&mut` through
/// [`WhisperModel::decode_step`], so the model itself holds no in-flight decode
/// state — the stale-cache hazard a model-stored `RefCell` would carry is
/// retired by construction.
///
/// The cross-attention K/V is projected from the **first** step's encoder states
/// and reused verbatim on every later step (ignoring each warm step's `xa`),
/// matching the reference decoder. The library does not detect a cache reused
/// across two different utterances, so a caller that threads a different-content
/// `enc` on a warm step would silently decode against the first utterance's audio
/// features — the consumer mints a fresh cache per utterance
/// ([`WhisperModel::new_cache`]). See the module threat-model note.
#[derive(Debug, Default)]
pub struct WhisperDecodeCache {
  /// The per-block KV cache accumulated so far; `None` until the first
  /// [`WhisperModel::decode_step`] forwards the initial prefix.
  inner: Option<DecoderKvCache>,
}

impl WhisperDecodeCache {
  /// A fresh, empty cache (no decoded positions).
  #[inline(always)]
  pub fn new() -> Self {
    Self { inner: None }
  }

  /// The number of decoder positions cached so far (the self-attention key
  /// time dimension), `0` for a fresh cache.
  #[inline(always)]
  pub fn len(&self) -> usize {
    match self.inner.as_ref().and_then(|c| c.first()) {
      // `block[0].0` is the self-attention `(k, v)`; `k.shape[1]` is its
      // accumulated time dimension.
      Some((Some((k, _)), _)) => k.shape().get(1).copied().unwrap_or(0),
      _ => 0,
    }
  }

  /// Whether the cache holds no decoded positions yet.
  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

/// A loaded Whisper model: the audio encoder, the text decoder, the
/// dimensions, and an optional tokenizer for the high-level
/// [`Transcribe`] entry point.
///
/// Mirrors the reference `Model` (`whisper.py:489`): `encode` runs the encoder
/// once per utterance, `decode_step` runs the decoder one token window at a
/// time against a caller-owned [`WhisperDecodeCache`]. The model holds NO
/// mutable decode state — the per-block `(self_kv, cross_kv)` cache is the
/// caller's [`WhisperDecodeCache`] value, so a single loaded model backs any
/// number of concurrent / sequential decodes safely.
pub struct WhisperModel {
  dims: ModelDimensions,
  dtype: Dtype,
  encoder: AudioEncoder,
  decoder: TextDecoder,
  /// The Whisper tokenizer, when one has been attached
  /// ([`WhisperModel::with_tokenizer`]). Required by the high-level
  /// [`Transcribe`] impl (which builds
  /// the [`HFTokenizerWrapper`] and runs the decoding task); the lower-level
  /// [`super::decoding`] entry points take an explicit tokenizer instead, so a
  /// model loaded without one can still be driven there.
  tokenizer: Option<Tokenizer>,
  /// The `<|endoftext|>` id resolved from the attached tokenizer, when known.
  /// `None` ⇒
  /// [`AutoregressiveStt::eot`]
  /// falls back to the canonical [`EOT_TOKEN_ID`]. Set by
  /// [`WhisperModel::with_tokenizer`] / [`WhisperModel::with_eot_token`].
  eot_token: Option<u32>,
  /// The word-timing alignment heads — the `(layer, head)` cross-attention
  /// heads the word-timestamp DTW averages (`whisper.py:_alignment_heads`).
  /// Defaults to the last half of the decoder layers
  /// ([`AlignmentHeads::default_for`]); overridable from a checkpoint's
  /// `generation_config.json` ([`WhisperModel::with_alignment_heads`], loaded by
  /// [`WhisperModel::load`]).
  alignment_heads: AlignmentHeads,
  /// The CoreML / Neural-Engine sibling backend, when the loaded checkpoint
  /// directory carried a `.mlmodelc` bundle (Apple Silicon only) — see
  /// [`super::coreml::CoreMlWhisper`]. When present, the high-level
  /// [`Transcribe`] entry point drives transcription on the Neural Engine
  /// (selected by [`Self::backend`]); the MLX weights this struct also holds
  /// remain available and unchanged. `None` ⇒ the MLX path, byte-identical to
  /// before this field existed. Boxed to keep [`WhisperModel`] small (the
  /// loaded `MLModel` handles are large relative to the rest of the struct).
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  coreml: Option<Box<super::coreml::CoreMlWhisper>>,
}

impl fmt::Debug for WhisperModel {
  /// Manual [`Debug`] — the borrowed-vocabulary [`Tokenizer`] is not `Debug`,
  /// so only the scalar model state and whether a tokenizer is attached are
  /// printed.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("WhisperModel")
      .field("dims", &self.dims)
      .field("dtype", &self.dtype)
      .field("encoder", &self.encoder)
      .field("decoder", &self.decoder)
      .field("has_tokenizer", &self.tokenizer.is_some())
      .field("eot_token", &self.eot_token)
      .field("alignment_heads", &self.alignment_heads.heads().len())
      .finish()
  }
}

impl WhisperModel {
  /// Build a [`WhisperModel`] from a **raw** (pre-sanitize) weight map and the
  /// dimensions.
  ///
  /// The order is allocation-disciplined: validate the config
  /// ([`ModelDimensions::validate`]), then [`sanitize`] the checkpoint by
  /// **key-remap only** (strip the `model.` prefix, apply the HF→MLX `KEY_MAP`,
  /// drop the recomputed positional key, reject colliding destinations) — no
  /// tensor data is transposed or cast yet — and finally build every sub-module,
  /// where each consumed tensor is **shape-validated against the config-derived
  /// extents BEFORE it is transposed / cast / materialized**. So a corrupt-but-
  /// loadable checkpoint that ships an oversized tensor under a consumed key is
  /// rejected with a typed shape error *before* any transpose / `astype` sized by
  /// that tensor allocates (the oversized-conv-weight OOM the eager-cast order
  /// risked). Unneeded tensors are never materialized — they are dropped with the
  /// builder.
  ///
  /// `dtype` is the model compute dtype every consumed weight is cast to (the
  /// reference default is `float16`).
  ///
  /// # Errors
  /// - [`ModelDimensions::validate`] constraints;
  /// - [`Error::KeyCollision`] if two source keys remap onto one sanitized key;
  /// - [`Error::LayerKeyed`] wrapping [`Error::ShapePairMismatch`] if a consumed
  ///   tensor's shape disagrees with the config-derived expectation;
  /// - [`Error::MissingKey`] if a weight required to build a sub-module is
  ///   absent from the sanitized map;
  /// - propagates the sub-module construction op errors (causal-mask /
  ///   sinusoids).
  pub fn from_weights(
    dims: ModelDimensions,
    weights: HashMap<String, Array>,
    dtype: Dtype,
  ) -> Result<Self> {
    Self::from_weights_quantized(dims, weights, dtype, None)
  }

  /// Build a [`WhisperModel`] from a **raw** (pre-sanitize) weight map, the
  /// dimensions, and an optional parsed quantization config — the
  /// quantization-aware analogue of [`WhisperModel::from_weights`] (which is
  /// just this with `quantization = None`).
  ///
  /// When `quantization` is `Some` AND a layer's `<prefix>.weight` carries the
  /// sibling `<prefix>.scales` tensor in the (sanitized) checkpoint, that
  /// projection is built as a quantized layer
  /// ([`MaybeQuantizedLinear::Quantized`]) running
  /// [`crate::ops::quantized::quantized_matmul`] — the weight-map analogue of
  /// mlx-audio's whisper `class_predicate`
  /// (`isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights`,
  /// `mlx_audio/stt/models/whisper/whisper.py:674-676`), with the
  /// `(group_size, bits, mode)` resolved per layer from `quantization`. A dense
  /// projection (no `.scales` sibling) builds exactly as before, so a
  /// non-quantized checkpoint loads identically whether or not a `quantization`
  /// config is threaded.
  ///
  /// The [`sanitize`] HF→MLX key-remap carries the `.scales` / `.biases`
  /// sibling tensors through unchanged (the rename patterns key on the
  /// `<module>.` prefix, not the `.weight` leaf), so a HuggingFace-format
  /// quantized checkpoint lands its quantized triples on the right layer.
  ///
  /// `quantization` is the parsed
  /// [`crate::audio::load::LoadedAudioModel::quantization`] (the
  /// `config.json` `quantization` block parsed by
  /// [`crate::audio::load::apply_quantization`]). A model whose config has no
  /// `quantization` block passes `None` here.
  ///
  /// # Errors
  /// The [`WhisperModel::from_weights`] errors, plus:
  /// - [`Error::InvariantViolation`] if a `<prefix>.scales` sibling is present
  ///   but `quantization` resolved no scheme parameters for that layer (the
  ///   weights say quantized, the config says dense);
  /// - [`Error::MissingKey`] / [`Error::ShapePairMismatch`] if a quantized
  ///   biasful projection is missing its required dense `<prefix>.bias` or
  ///   carries one of the wrong shape (the quantized path enforces the same
  ///   dense-bias arity the dense path does);
  /// - propagates [`crate::nn::QuantizedLinear::from_parts`]'s structural
  ///   validation of the quantized triple.
  pub fn from_weights_quantized(
    dims: ModelDimensions,
    weights: HashMap<String, Array>,
    dtype: Dtype,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    dims.validate()?;
    // Key-remap ONLY (no transpose / cast): renames are O(1) per key and never
    // touch tensor data, so an oversized tensor is not materialized here. The
    // remap carries `.scales` / `.biases` siblings through alongside `.weight`.
    let (weights, is_hf_format) = sanitize(weights)?;
    // Normalize the per-layer quantization keys into the SAME namespace as the
    // sanitized weight-lookup prefixes the builder resolves against: a config
    // can carry raw HF paths (`model.encoder.layers.0.self_attn.q_proj`,
    // `decoder.embed_tokens`) regardless of how the weights are named, but the
    // builder calls `quantization_for` with the sanitized MLX prefix
    // (`encoder.blocks.0.attn.query`, `decoder.token_embedding`). Without this a
    // per-layer override would silently miss and fall back to the global scheme.
    // Run unconditionally (independent of the weight-map format): `remap_hf_key`
    // is idempotent, so MLX-named config keys pass through unchanged while HF
    // keys become MLX names (mlx-lm / mlx-audio resolve the per-layer map
    // against post-sanitize module paths, `mlx_lm/utils.py:349-352`).
    let normalized_quant = quantization.map(normalize_quant_keys).transpose()?;
    // The builder shape-validates each consumed dense tensor against the config
    // BEFORE transposing / casting it, so the materialization runs only on
    // tensors already proven within the config caps. A quantized layer's packed
    // weight has a different (packed) shape, so it bypasses the dense shape
    // check and is validated structurally by `QuantizedLinear::from_parts`.
    let mut builder = Builder {
      weights,
      is_hf_format,
      dtype,
      quantization: normalized_quant.as_ref(),
    };

    let encoder = builder.build_encoder(&dims)?;
    let decoder = builder.build_decoder(&dims, dtype)?;

    // Default the alignment heads to the last half of the decoder layers
    // (`whisper.py:510-516`); a checkpoint's `generation_config.json` can
    // override this via `WhisperModel::with_alignment_heads`. Computed before
    // `dims` is moved into the struct.
    let alignment_heads = AlignmentHeads::default_for(&dims);

    Ok(Self {
      dims,
      dtype,
      encoder,
      decoder,
      tokenizer: None,
      eot_token: None,
      alignment_heads,
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      coreml: None,
    })
  }

  /// Attach a Whisper [`Tokenizer`] so the high-level
  /// [`Transcribe`] entry point can build
  /// the [`HFTokenizerWrapper`] and run the decoding task. Also records the
  /// tokenizer's resolved `<|endoftext|>` id, so
  /// [`AutoregressiveStt::eot`]
  /// reflects the actual checkpoint vocabulary rather than the canonical
  /// default.
  ///
  /// Returns `self` for chaining after [`WhisperModel::from_weights`] /
  /// [`WhisperModel::load`].
  ///
  /// # Errors
  /// [`Error::MissingKey`] if the tokenizer is missing a required Whisper
  /// special token (via [`HFTokenizerWrapper::new`]) when resolving the eot id.
  pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Result<Self> {
    // Resolve eot from the tokenizer (a one-shot wrapper, dropped after).
    let eot = {
      let wrapper = HFTokenizerWrapper::new(
        &tokenizer,
        self.dims.is_multilingual(),
        self.dims.num_languages(),
        None,
        WhisperTask::Transcribe,
      )?;
      wrapper.eot()
    };
    self.eot_token = Some(eot);
    self.tokenizer = Some(tokenizer);
    Ok(self)
  }

  /// Record the `<|endoftext|>` id resolved from the loaded tokenizer, so
  /// [`AutoregressiveStt::eot`]
  /// reflects the actual checkpoint vocabulary rather than the canonical
  /// multilingual default.
  ///
  /// Whisper's English-only (`*.en`) and multilingual vocabularies place the
  /// special tokens at different offsets; the id is normally resolved by string
  /// lookup through [`HFTokenizerWrapper`]. A caller that has built the wrapper
  /// passes its
  /// [`HFTokenizerWrapper::eot`](super::tokenizer::HFTokenizerWrapper::eot)
  /// here (or use [`WhisperModel::with_tokenizer`] to attach the full tokenizer
  /// for the high-level [`Transcribe`]
  /// entry point). Returns `self` for chaining after
  /// [`WhisperModel::from_weights`] / [`WhisperModel::load`].
  pub fn with_eot_token(mut self, eot: u32) -> Self {
    self.eot_token = Some(eot);
    self
  }

  /// Override the word-timing [`AlignmentHeads`] — the `set_alignment_heads`
  /// analogue (`whisper.py:522-537`). A checkpoint that ships an explicit head
  /// set in its `generation_config.json` uses it (instead of the last-half
  /// default) for the word-timestamp DTW. Returns `self` for chaining.
  ///
  /// Takes an already-validated [`AlignmentHeads`]: the only public ways to
  /// build a custom one are [`AlignmentHeads::try_new`] and
  /// [`AlignmentHeads::from_generation_config`], both of which run the
  /// `validate_alignment_heads` in-grid + no-duplicate check against the dims
  /// passed to them (the unchecked `AlignmentHeads::new` is crate-private). An
  /// [`AlignmentHeads`] carries the `(n_layer, n_head)` grid it was validated
  /// against; this rechecks it equals this model's `(n_text_layer, n_text_head)`
  /// and rejects a mismatch, so a set validated against a *different* (e.g.
  /// larger) model cannot be installed here and reach the DTW gather out of this
  /// model's grid. Build the override with
  /// `AlignmentHeads::try_new(heads, model.dims())?`.
  ///
  /// [`WhisperModel::load`] applies this automatically when a
  /// `generation_config.json` with an `alignment_heads` key is present
  /// alongside the weights.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `heads`'s validated `(n_layer, n_head)` grid does
  ///   not equal this model's `(n_text_layer, n_text_head)` — the heads were
  ///   validated for a different model.
  pub fn with_alignment_heads(mut self, heads: AlignmentHeads) -> Result<Self> {
    let (n_layer, n_head) = (self.dims.n_text_layer(), self.dims.n_text_head());
    if heads.n_layer() != n_layer || heads.n_head() != n_head {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "Whisper alignment_heads",
        "heads were validated for a different model's (n_text_layer, n_text_head) grid",
        smol_str::format_smolstr!(
          "validated ({}, {}), model ({n_layer}, {n_head})",
          heads.n_layer(),
          heads.n_head()
        ),
      )));
    }
    self.alignment_heads = heads;
    Ok(self)
  }

  /// The word-timing [`AlignmentHeads`] (the last-half default unless
  /// overridden). Consumed by the word-timestamp DTW
  /// ([`super::timing::find_alignment`]).
  #[inline(always)]
  pub fn alignment_heads(&self) -> &AlignmentHeads {
    &self.alignment_heads
  }

  /// Whether the CoreML / Neural-Engine backend should drive a decode: iff a
  /// CoreML sibling is loaded (`has_coreml`) AND neither word timestamps NOR
  /// best-of-N (`best_of > 1`) is requested.
  ///
  /// CoreML is skipped for a word-timestamp request because its
  /// `alignment_heads_weights` cross-attention does not expose the per-head
  /// `cross_qk` the word-timing DTW ([`super::timing::find_alignment`]) consumes;
  /// and for a best-of-N request because the WhisperKit explicit-cache decoder
  /// advances a single token per step and cannot batch the candidate dimension.
  /// Either falls back to the MLX backend until the CoreML path supports it.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  fn prefers_coreml(has_coreml: bool, word_timestamps: bool, best_of: Option<usize>) -> bool {
    has_coreml && !word_timestamps && !best_of.is_some_and(|n| n > 1)
  }

  /// The concrete inference [`WhisperBackend`] the decode pipeline drives for a
  /// request — the CoreML / Neural-Engine sibling when one was loaded alongside
  /// this checkpoint (Apple Silicon) AND neither word timestamps nor best-of-N
  /// (`best_of > 1`) is requested (see `prefers_coreml`), else the MLX path.
  ///
  /// The high-level [`Transcribe`] entry points build the backend through this
  /// one chokepoint and hand `&backend` to the [`super::decoding`] free
  /// functions, so backend selection lives in a single place. Off Apple Silicon
  /// (and whenever no `.mlmodelc` bundle was present) this is always
  /// [`WhisperBackend::Mlx`], byte-identical to the pre-backend pipeline.
  #[inline]
  pub fn backend(&self, word_timestamps: bool, best_of: Option<usize>) -> WhisperBackend<'_> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if Self::prefers_coreml(self.coreml.is_some(), word_timestamps, best_of) {
      // `prefers_coreml` returned true ⇒ a CoreML sibling IS loaded; bind it.
      if let Some(coreml) = self.coreml.as_deref() {
        return WhisperBackend::CoreMl(coreml);
      }
    }
    WhisperBackend::Mlx(self)
  }

  /// Attach a loaded CoreML / Neural-Engine sibling backend, so the high-level
  /// [`Transcribe`] entry point drives transcription on the ANE
  /// ([`Self::backend`]). Apple Silicon only; applied automatically by
  /// [`WhisperModel::load`] when the model directory carries a `.mlmodelc`
  /// bundle. Returns `self` for chaining.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[inline]
  pub fn with_coreml(mut self, coreml: super::coreml::CoreMlWhisper) -> Self {
    self.coreml = Some(Box::new(coreml));
    self
  }

  /// Load a Whisper model from a local model directory: read the
  /// `*.safetensors`, then forward to [`WhisperModel::from_weights`]. `dims`
  /// is parsed by the caller via [`ModelDimensions::from_dict`] (the
  /// `config.json` body is on the [`crate::audio::load::LoadedAudioModel`]
  /// bundle).
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if no `*.safetensors` is found under `dir`;
  /// - propagates [`crate::io::load_safetensors`] / [`Self::from_weights`]
  ///   errors.
  pub fn load(dir: &std::path::Path, dims: ModelDimensions, dtype: Dtype) -> Result<Self> {
    let weights = load_all_safetensors(dir)?;
    let model = Self::from_weights(dims, weights, dtype)?;
    let model = apply_dir_alignment_heads(model, dir)?;
    maybe_attach_coreml(model, dir)
  }

  /// Load a Whisper model from a local model directory **with** an optional
  /// parsed quantization config — the quantization-aware analogue of
  /// [`WhisperModel::load`].
  ///
  /// Reads the `*.safetensors` under `dir`, then forwards to
  /// [`WhisperModel::from_weights_quantized`]. The caller parses `dims` via
  /// [`ModelDimensions::from_dict`] and `quantization` via
  /// [`crate::audio::load::apply_quantization`] from the same `config.json`
  /// body (both available on the [`crate::audio::load::LoadedAudioModel`]
  /// bundle the STT [`crate::audio::stt::load::load`] factory hands the
  /// constructor). An mlx-community 8-bit checkpoint (e.g.
  /// `whisper-large-v3-turbo-8bit`) loads its quantized projections through
  /// this entry; a dense checkpoint with `quantization = None` loads exactly
  /// as [`WhisperModel::load`].
  ///
  /// # Errors
  /// - [`Error::MissingKey`] if no `*.safetensors` is found under `dir`;
  /// - propagates [`crate::io::load_safetensors`] /
  ///   [`Self::from_weights_quantized`] errors.
  pub fn load_quantized(
    dir: &std::path::Path,
    dims: ModelDimensions,
    dtype: Dtype,
    quantization: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let weights = load_all_safetensors(dir)?;
    let model = Self::from_weights_quantized(dims, weights, dtype, quantization)?;
    let model = apply_dir_alignment_heads(model, dir)?;
    maybe_attach_coreml(model, dir)
  }

  /// The model dimensions.
  #[inline(always)]
  pub fn dims(&self) -> &ModelDimensions {
    &self.dims
  }

  /// The model compute dtype.
  #[inline(always)]
  pub fn dtype(&self) -> Dtype {
    self.dtype
  }

  /// Build an [`HFTokenizerWrapper`] from the model's **attached** tokenizer for
  /// the given `language` / `task` — the wrapper-construction the high-level
  /// [`Transcribe`] impl runs internally, exposed so the streaming session
  /// ([`super::streaming::WhisperStreaming::with_model_tokenizer`]) can build the
  /// same wrapper from a model loaded with [`Self::with_tokenizer`].
  ///
  /// # Errors
  /// - [`Error::InvariantViolation`] if no tokenizer is attached (use
  ///   [`Self::with_tokenizer`], or build the wrapper explicitly and call
  ///   [`super::streaming::WhisperStreaming::new`]);
  /// - propagates [`HFTokenizerWrapper::new`] (a missing Whisper special token).
  pub(crate) fn streaming_tokenizer(
    &self,
    language: Option<&str>,
    task: WhisperTask,
  ) -> Result<HFTokenizerWrapper<'_>> {
    let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "WhisperModel::streaming_tokenizer",
        "requires an attached tokenizer (use WhisperModel::with_tokenizer, or build \
         the HFTokenizerWrapper explicitly and call WhisperStreaming::new)",
      ))
    })?;
    HFTokenizerWrapper::new(
      tokenizer,
      self.dims.is_multilingual(),
      self.dims.num_languages(),
      language,
      task,
    )
  }

  /// Validate that an encoder-states tensor is exactly `(expected_group,
  /// n_audio_ctx, n_audio_state)` — one Whisper segment (the single-sequence
  /// `expected_group == 1`) or that segment broadcast across the best-of-N
  /// candidate rows (`expected_group == n_group`) — before it is fed to the
  /// decoder's cross-attention.
  ///
  /// The decoder's cross-attention projects `xa` and forms its scores from
  /// `xa.shape()[1]` (the key time axis), so a caller that supplies a longer
  /// encoder extent would drive the cross-attention KV / score buffers past the
  /// caps the config states for a single `[·, n_audio_ctx, n_audio_state]`
  /// segment (the cross-attention score cap is `n_text_head * n_text_ctx *
  /// n_audio_ctx`, the KV cache cap `n_text_layer * n_audio_ctx * n_audio_state`
  /// — both assume the encoder extent is exactly `n_audio_ctx`). Pinning the two
  /// inner extents here makes those caps provably bound the runtime
  /// cross-attention tensors regardless of caller. The batch axis is pinned to
  /// `expected_group` so it always equals the decoder-input batch: the single-
  /// sequence path (`expected_group == 1`) rejects any `[B != 1, …]` tensor (its
  /// queries are `(1, T)`, so a wider K/V would silently broadcast-mismatch), and
  /// the batched path pins it to `n_group`. The encoder's own `forward` produces
  /// the `(1, …)` shape, which the decode task broadcasts to `(n_group, …)`
  /// before a batched decode ([`Self::broadcast_encoder_states`]); a direct
  /// caller of [`Self::decode_tokens`] / [`AutoregressiveStt::decode_step`] (e.g.
  /// [`super::decoding::detect_language`], which takes `audio_features` straight
  /// from the caller) could pass any tensor — so the guard lives at the decoder
  /// entry, not only at the encoder exit.
  ///
  /// # Errors
  /// [`Error::ShapePairMismatch`] if `enc.shape() != [expected_group,
  /// n_audio_ctx, n_audio_state]`.
  fn validate_encoder_states(&self, enc: &Array, expected_group: usize) -> Result<()> {
    let expected = [
      expected_group,
      self.dims.n_audio_ctx(),
      self.dims.n_audio_state(),
    ];
    let actual = enc.shape();
    if actual.len() != expected.len() || actual.iter().zip(expected.iter()).any(|(a, e)| a != e) {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "WhisperModel: encoder states must be (expected_group, n_audio_ctx, n_audio_state) — a single Whisper segment (single-sequence) or broadcast across candidate rows (best-of-N)",
        expected.to_vec(),
        actual,
      )));
    }
    Ok(())
  }

  /// Broadcast `(1, n_audio_ctx, n_audio_state)` encoder states to `(n_group,
  /// n_audio_ctx, n_audio_state)` for a batched best-of-N decode — the analogue
  /// of the reference reusing one audio feature tensor across the `n_group`
  /// candidate rows (`decoding.py:640` / the `audio_features[::n_group]`
  /// regrouping at `:670`).
  ///
  /// The encoder runs once per utterance, so every candidate shares the same
  /// audio features; broadcasting (rather than re-encoding) keeps the cross-
  /// attention K/V batch equal to the self-attention K/V batch (`n_group`) so the
  /// whole decoder forward batches without any per-row encoder work. `n_group ==
  /// 1` returns the states unchanged (a clone), so the single-sequence path does
  /// no broadcast.
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `enc` is not `(1, n_audio_ctx,
  ///   n_audio_state)`;
  /// - [`Error::OutOfRange`] if `n_group` is `0` or overflows `i32`;
  /// - propagates the broadcast op error.
  pub(crate) fn broadcast_encoder_states(&self, enc: &Array, n_group: usize) -> Result<Array> {
    // The source must be the single-segment `(1, …)` encoder output; reject a
    // pre-batched tensor here so the broadcast target is unambiguous.
    let expected = [1usize, self.dims.n_audio_ctx(), self.dims.n_audio_state()];
    let actual = enc.shape();
    if actual.len() != expected.len() || actual.iter().zip(expected.iter()).any(|(a, e)| a != e) {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "WhisperModel::broadcast_encoder_states: source must be (1, n_audio_ctx, n_audio_state)",
        expected.to_vec(),
        actual,
      )));
    }
    if n_group == 0 {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::broadcast_encoder_states: n_group",
        "must be >= 1",
        smol_str::format_smolstr!("n_group={n_group}"),
      )));
    }
    if n_group == 1 {
      return enc.try_clone();
    }
    let g = i32::try_from(n_group).map_err(|_| {
      Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::broadcast_encoder_states: n_group",
        "must fit in i32",
        smol_str::format_smolstr!("{n_group}"),
      ))
    })?;
    let ctx =
      i32::try_from(self.dims.n_audio_ctx()).map_err(|_| dim_i32_overflow("n_audio_ctx"))?;
    let state =
      i32::try_from(self.dims.n_audio_state()).map_err(|_| dim_i32_overflow("n_audio_state"))?;
    crate::ops::shape::broadcast_to(enc, &[g, ctx, state])
  }

  /// Run the decoder over a token sequence `tokens` `(1, T)` against the
  /// encoder states, with an **explicit caller-owned** KV cache — the
  /// `Inference.logits` analogue (`decoding.py:170-175`).
  ///
  /// The cache is threaded by value, so a caller — the
  /// [`super::decoding::DecodingTask`] — owns the decode trajectory's cache and
  /// runs multi-token prefill + single-token steps. This is the multi-position
  /// primitive underneath both the
  /// [`AutoregressiveStt::decode_step`]
  /// family hook (single-position) and the decoding task's prefill;
  /// `detect_language` likewise runs a throwaway single-token forward through
  /// here.
  ///
  /// The `encoder_states` extent is validated to exactly `(1, n_audio_ctx,
  /// n_audio_state)` ([`Self::validate_encoder_states`]) BEFORE the decoder runs,
  /// so a longer / batched `enc` from any caller (including a direct caller that
  /// bypasses the encoder) cannot drive the cross-attention past its config caps.
  /// That is an O(1) shape/rank pin only — the decoder trusts the *content* of
  /// the encoder states it is handed (matching the reference), so a cache reused
  /// against a same-shaped but different-content `enc` is **not** detected; the
  /// consumer mints a fresh cache per utterance (see the module threat-model
  /// note).
  ///
  /// Returns `(logits, updated_cache)` — logits `(1, T, n_vocab)` (cast to
  /// `f32`, matching `Inference.logits`'s `.astype(mx.float32)`) and the
  /// per-block cache to thread into the next call.
  ///
  /// `tokens` is the new-segment token slice, passed straight to the decoder,
  /// which is the single crate-visible chokepoint every gather path funnels
  /// through: it builds the `(1, T)` `u32` decoder-input array **internally**
  /// (so the rank/batch and dtype classes are true by construction) and
  /// value-checks every id `< n_vocab` BEFORE the token-embedding gather, so an
  /// id `>= n_vocab` — which would index out of bounds in the `(n_vocab,
  /// n_text_state)` embedding — is rejected at the gather root regardless of
  /// caller ([`AutoregressiveStt::decode_step`] and the decoding task's prefill /
  /// step / language-detection forwards all reach the decoder through here or
  /// [`Self::forward_with_cross_qk`]).
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `encoder_states` is not `(1, n_audio_ctx,
  ///   n_audio_state)`;
  /// - [`Error::OutOfRange`] if `tokens` is empty, longer than `max_context`
  ///   (`n_text_ctx`), overflows `i32`, or contains an id `>= n_vocab`;
  /// - propagates the decoder forward op errors (embedding / block / LayerNorm /
  ///   positional-slice).
  pub(crate) fn decode_tokens(
    &self,
    tokens: &[u32],
    encoder_states: &Array,
    cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache)> {
    // Bound the encoder extent BEFORE the decoder cross-attention projects it.
    // This single-sequence path builds `(1, T)` queries, so the encoder batch
    // must be exactly 1 (a wider K/V would silently broadcast-mismatch).
    self.validate_encoder_states(encoder_states, 1)?;
    // Pass the validated `&[u32]` straight to the decoder: it builds the
    // `(1, T)` `u32` array itself and enforces the empty / `id < n_vocab` /
    // `i32` / `offset + T <= n_text_ctx` guards at the single gather chokepoint —
    // so the value-range / rank / dtype classes are structurally closed there for
    // every caller, with no double Array-build at this layer.
    let (logits, new_cache) = self.decoder.forward(tokens, encoder_states, cache)?;
    let logits = logits.astype(Dtype::F32)?;
    Ok((logits, new_cache))
  }

  /// Warm-step decode from a token already on-device — the lazy-input analogue
  /// of [`Self::decode_tokens`] (#369). `token` is a `(1, 1)` `u32` array (the
  /// previous step's argmax, kept lazy), so the decode loop never round-trips the
  /// token through a host `&[u32]`.
  pub(crate) fn decode_token_lazy(
    &self,
    token: &Array,
    encoder_states: &Array,
    cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache)> {
    self.validate_encoder_states(encoder_states, 1)?;
    let (logits, new_cache) = self.decoder.forward_array(token, encoder_states, cache)?;
    let logits = logits.astype(Dtype::F32)?;
    Ok((logits, new_cache))
  }

  /// Run the decoder over `n_group` parallel candidate rows — the batched
  /// (`n_group > 1`) analogue of [`Self::decode_tokens`] underneath best-of-N
  /// sampling (the reference's `(n_audio * n_group, T)` decode,
  /// `decoding.py:657-667`).
  ///
  /// `tokens` is the `(n_group, T)` new-window ids in **row-major** order, its
  /// length an exact multiple of `n_group`. `encoder_states` is the audio
  /// features broadcast to `(n_group, n_audio_ctx, n_audio_state)` (via
  /// [`Self::broadcast_encoder_states`]), so every candidate's cross-attention
  /// K/V batches the same width as its self-attention K/V — the rows decode
  /// independently. `encoder_states` is extent-validated (its batch axis must
  /// equal `n_group`); the decoder re-checks the per-row token invariants at its
  /// single gather chokepoint. Returns `(logits, updated_cache)` with logits
  /// `(n_group, T, n_vocab)` (cast to `f32`, matching `Inference.logits`).
  ///
  /// `n_group == 1` forwards through the same decoder path as
  /// [`Self::decode_tokens`] (a single `(1, T)` array), so the single-sequence
  /// numerics are unchanged.
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `encoder_states` is not `(n_group,
  ///   n_audio_ctx, n_audio_state)`;
  /// - [`Error::OutOfRange`] if `n_group` is `0`, `tokens.len()` is not a
  ///   multiple of `n_group`, the per-row window is empty / over-context, or an
  ///   id is `>= n_vocab`;
  /// - propagates the decoder forward op errors.
  pub(crate) fn decode_tokens_batched(
    &self,
    tokens: &[u32],
    n_group: usize,
    encoder_states: &Array,
    cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache)> {
    // Bound the encoder extent AND pin its batch axis to `n_group` BEFORE the
    // decoder cross-attention projects it, so the broadcast K/V line up with the
    // self-attention rows (a `(1, …)` or wrongly-sized `enc` would otherwise
    // silently broadcast under MLX rules against the `(n_group, …)` queries,
    // decoding every row against the wrong feature batch).
    self.validate_encoder_states(encoder_states, n_group)?;
    let (logits, new_cache) =
      self
        .decoder
        .forward_batched(tokens, n_group, encoder_states, cache)?;
    let logits = logits.astype(Dtype::F32)?;
    Ok((logits, new_cache))
  }

  /// Run one decode step like [`AutoregressiveStt::decode_step`], additionally
  /// returning the per-layer cross-attention weights — the
  /// `Model.forward_with_cross_qk` / `Inference.logits_with_cross_qk` analogue
  /// (`whisper.py:614-616`, `decoding.py:177-189`).
  ///
  /// This is the incremental (cache-threaded) extraction point for the
  /// cross-attention pattern. The word-timestamp DTW alignment uses the
  /// cacheless full-sequence variant [`Self::forward_with_cross_qk`] instead;
  /// this per-step form lets a caller (e.g. a streaming AlignAtt monitor)
  /// retrieve the per-decoder-layer cross-attention as the decode advances.
  ///
  /// `enc` is the encoder states `(1, n_audio_ctx, n_audio_state)` (extent-
  /// guarded exactly as [`AutoregressiveStt::decode_step`]); `cache` is the
  /// caller-owned [`WhisperDecodeCache`], threaded the same way (a fresh cache
  /// prefills the whole prefix, a warm cache forwards only the new tail).
  ///
  /// Returns `(logits, cross_qk)`:
  /// - `logits` — the full `(1, T, n_vocab)` decoder output (cast to `f32`,
  ///   matching `Inference.logits`'s `.astype(mx.float32)`); the caller slices
  ///   the position(s) it needs;
  /// - `cross_qk` — one cross-attention weight tensor `(1, n_text_head, T,
  ///   n_audio_ctx)` per decoder layer (`Some` for every block, since every
  ///   Whisper decoder block carries cross-attention). The `Option` mirrors the
  ///   reference's `cross_qk = [None] * len(blocks)` per-layer list.
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `enc` is not `(1, n_audio_ctx,
  ///   n_audio_state)`;
  /// - [`Error::EmptyInput`] if `tokens` is shorter than the cache;
  /// - [`Error::OutOfRange`] if the token prefix exceeds `max_context` or a
  ///   dimension overflows `i32`;
  /// - propagates the decoder forward op errors.
  pub fn decode_step_with_cross_qk(
    &self,
    cache: &mut WhisperDecodeCache,
    enc: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    // Bound the encoder-states extent and the token prefix BEFORE the decoder
    // allocates — the same guards as `decode_step`, factored into
    // `prepare_decode_tail`, which returns the new (uncached) `&[u32]` tail. The
    // decoder builds the `(1, T_new)` array itself and re-checks the value range
    // at the gather chokepoint, so there is no double Array-build at this layer.
    let new_tokens = self.prepare_decode_tail(cache, enc, tokens)?;
    let (logits, new_cache, cross_qk) =
      self
        .decoder
        .forward_with_cross_qk(new_tokens, enc, cache.inner.as_ref())?;
    // Run EVERY fallible step (the `f32` cast — `Inference.logits`'s
    // `.astype(mx.float32)`) BEFORE committing the cache, then make the cache
    // mutation the LAST step. If the cast (an allocation / backend op) failed
    // after `cache.inner` had already advanced, a retry with the same prefix
    // would see no new tokens to decode (`prepare_decode_tail` would treat the
    // mutated cache as caught up) and the decode trajectory would be corrupted.
    // Mirrors the `decode_tokens` path, which casts before returning the cache.
    let logits = logits.astype(Dtype::F32)?;
    cache.inner = Some(new_cache);
    Ok((logits, cross_qk))
  }

  /// Encode `mel` and run the decoder over the **full** `tokens` sequence in one
  /// (cacheless) forward, returning the logits and per-layer cross-attention
  /// weights — `Model.forward_with_cross_qk` (`whisper.py:614-616`).
  ///
  /// This is the entry the word-timestamp DTW ([`super::timing::find_alignment`])
  /// drives: unlike [`Self::decode_step_with_cross_qk`] (the incremental,
  /// cache-threaded decode path), it forwards the whole token prefix at once
  /// (`kv_cache = None`), so the returned `cross_qk[layer]` is the full
  /// `(1, n_text_head, T, n_audio_ctx)` attention over every token position —
  /// exactly what the alignment needs.
  ///
  /// `tokens` is the token-id sequence (the sot sequence + `no_timestamps` +
  /// text + eot the DTW assembles), taken as a `&[u32]` to mirror the
  /// incremental [`Self::decode_step_with_cross_qk`] contract: the `(1, T)`
  /// decoder-input array is built **internally** from the slice, so the
  /// rank/batch (always `(1, T)`) and dtype (always `u32`) classes are guaranteed
  /// by construction. The cheap host-side `tokens` checks (non-empty, `<=
  /// max_context`, `i32` fit, every id `< n_vocab`) run **before** `mel` is
  /// encoded, so an invalid-token call fails fast with its typed token error
  /// rather than after a wasted full encoder pass (the decoder re-checks the same
  /// invariants at its gather chokepoint as defense-in-depth). `mel` is a
  /// `(N_FRAMES, n_mels)` (or already-encoded
  /// `(1, n_audio_ctx, n_audio_state)`) log-mel segment. Returns
  /// `(logits, cross_qk)`:
  /// - `logits` — the full `(1, T, n_vocab)` decoder output, cast to `f32`
  ///   (`Inference.logits`'s `.astype(mx.float32)`);
  /// - `cross_qk` — one `(1, n_text_head, T, n_audio_ctx)` weight tensor per
  ///   decoder layer (`Some` for every block).
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `tokens` is empty, longer than `max_context`
  ///   (`n_text_ctx`), overflows `i32`, or contains an id `>= n_vocab` (which
  ///   would gather out of bounds in the decoder token embedding);
  /// - [`Error::RankMismatch`] / [`Error::ShapePairMismatch`] if `mel` is not a
  ///   valid mel / encoder-states tensor (via [`Self::encode`]);
  /// - propagates the encoder / decoder forward op errors.
  pub fn forward_with_cross_qk(
    &self,
    mel: &Array,
    tokens: &[u32],
  ) -> Result<(Array, Vec<Option<Array>>)> {
    // Fail-fast on the cheap host-side `tokens` input BEFORE the expensive
    // `self.encode(mel)` (which drives the FULL encoder + its device
    // allocations). A caller can supply an empty slice, an over-context prefix,
    // or an id `>= n_vocab`; each is a typed token error the decoder already
    // raises, but with `encode` first that error only surfaces AFTER a wasted
    // encoder pass — so this re-uses the same host-side fail-fast boundary the
    // incremental path establishes in `prepare_decode_tail`, ordered the same
    // way (non-empty → `<= max_context` → `i32` fit → `id < n_vocab`). The
    // decoder re-checks every one of these at its single gather chokepoint
    // (`TextDecoder::run`), so this is a (cheap, whole-prefix) defense-in-depth
    // fail-fast that makes the token error precede encoder work, NOT the gather's
    // only guard. This cacheless full-sequence path forwards the whole prefix at
    // `offset == 0`, so `offset + T <= n_text_ctx` reduces to `T <= max_context`.
    if tokens.is_empty() {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::forward_with_cross_qk: token sequence length",
        "must be non-empty",
        smol_str::format_smolstr!("len={}", tokens.len()),
      )));
    }
    let max_context = self.max_context();
    if tokens.len() > max_context {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::forward_with_cross_qk: token sequence length",
        "must not exceed max_context (n_text_ctx)",
        smol_str::format_smolstr!("len={}, max_context={max_context}", tokens.len()),
      )));
    }
    if i32::try_from(tokens.len()).is_err() {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::forward_with_cross_qk: token sequence length",
        "must fit in i32",
        smol_str::format_smolstr!("{}", tokens.len()),
      )));
    }
    self.validate_token_ids("WhisperModel::forward_with_cross_qk", tokens)?;

    let enc = self.encode(mel)?;
    // Cacheless full-sequence decode (`kv_cache = None`): the whole prefix is
    // forwarded, so every position's cross-attention is surfaced. The caller's
    // `&[u32]` is passed straight to the decoder, which builds the `(1, T)` `u32`
    // array itself and re-enforces the empty / `id < n_vocab` / `i32` /
    // `T <= n_text_ctx` guards at the single gather chokepoint — so the value-
    // range / rank / dtype classes are structurally closed there, with no
    // double Array-build at this layer.
    let (logits, _cache, cross_qk) = self.decoder.forward_with_cross_qk(tokens, &enc, None)?;
    let logits = logits.astype(Dtype::F32)?;
    Ok((logits, cross_qk))
  }

  /// Reject any token id `>= n_vocab` BEFORE it reaches the decoder token-
  /// embedding gather — a `(n_vocab, n_text_state)` table, so an out-of-range id
  /// would index out of bounds. This is the model-layer early fail-fast: the
  /// decoding task ([`DecodingTask::new`]) uses it to reject a caller-supplied
  /// `prompt` / `prefix` id at construction (before any forward / encode work),
  /// and the incremental decode path ([`Self::prepare_decode_tail`]) uses it to
  /// validate the WHOLE running prefix (including already-cached positions) up
  /// front. The decoder's [`super::decoder::TextDecoder::run`] independently
  /// re-checks the same `id < n_vocab` invariant on the new window at the gather
  /// root — the single structural chokepoint — so this model-layer check is a
  /// (cheap, whole-prefix) defense-in-depth fail-fast, not the gather's only
  /// guard. `context` names the calling entry for the typed error.
  ///
  /// # Errors
  /// [`Error::OutOfRange`] on the first id `>= n_vocab`.
  pub(crate) fn validate_token_ids(&self, context: &'static str, tokens: &[u32]) -> Result<()> {
    let n_vocab = self.dims.n_vocab();
    if let Some(&id) = tokens.iter().find(|&&id| id as usize >= n_vocab) {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        context,
        "token id must be < n_vocab (the decoder token-embedding rows)",
        smol_str::format_smolstr!("id={id}, n_vocab={n_vocab}"),
      )));
    }
    Ok(())
  }

  /// Validate the encoder-states extent + the token-prefix length / ids and
  /// return the **new** (uncached) token tail — the shared front of
  /// [`AutoregressiveStt::decode_step`] and [`Self::decode_step_with_cross_qk`].
  /// Returns the `[cached ..]` subslice of `tokens` (the positions not yet in the
  /// cache), validated so every id is `< n_vocab` before any embedding gather.
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `enc` is not `(1, n_audio_ctx,
  ///   n_audio_state)`;
  /// - [`Error::EmptyInput`] if `tokens` is shorter than the cache;
  /// - [`Error::OutOfRange`] if the token prefix exceeds `max_context`, contains
  ///   an id `>= n_vocab`, or the new-token window overflows `i32`.
  fn prepare_decode_tail<'t>(
    &self,
    cache: &WhisperDecodeCache,
    enc: &Array,
    tokens: &'t [u32],
  ) -> Result<&'t [u32]> {
    // Bound the encoder-states extent FIRST — before the token array or the
    // decoder allocates. A caller can supply any `enc`; the cross-attention
    // forms its scores from `enc.shape()[1]`, so a longer / batched segment must
    // be rejected before it can size the cross-attention buffers past the config
    // caps (which assume a single `[1, n_audio_ctx, n_audio_state]` segment).
    // This incremental path builds `(1, T_new)` queries, so the encoder batch
    // must be exactly 1.
    self.validate_encoder_states(enc, 1)?;

    let cached = cache.len();
    // Reject a token prefix longer than the decoder context BEFORE building the
    // token array: the decoder positions `tokens` at `[cached .. tokens.len())`
    // against the learned `n_text_ctx` positional table, so a prefix exceeding
    // `max_context` is out of range. Checking the length here makes the typed
    // error precede the `(1, T)` token allocation and the downstream
    // `(1, T, n_text_state)` embedding the decoder would otherwise materialize.
    let max_context = self.max_context();
    if tokens.len() > max_context {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::decode_step: token prefix length",
        "must not exceed max_context (n_text_ctx)",
        smol_str::format_smolstr!("len={}, max_context={max_context}", tokens.len()),
      )));
    }
    // Reject any token id `>= n_vocab` before the decoder token-embedding gather —
    // the same value-range guard the full-sequence `forward_with_cross_qk` path
    // applies, shared so an out-of-range caller-owned id cannot index out of
    // bounds on either path. Checks the whole prefix: on a warm cache the new
    // tail is gathered, on a fresh cache the entire prefix is.
    self.validate_token_ids("WhisperModel::decode_step", tokens)?;
    // Forward only the tokens not yet in the cache: the whole prefix on a fresh
    // cache, else just the new tail.
    let new_tokens = tokens
      .get(cached..)
      .filter(|t| !t.is_empty())
      .ok_or_else(|| {
        Error::EmptyInput(crate::error::EmptyInputPayload::new(
          "WhisperModel::decode_step: tokens prefix must extend the cache (no new \
           tokens to decode)",
        ))
      })?;
    Ok(new_tokens)
  }

  /// Map the universal [`TranscribeOptions`] onto the lower-level
  /// [`WhisperTranscribeOptions`], wiring EVERY surfaced knob: the task /
  /// language / timestamp flags, the generated-token cap (`sample_len`), the
  /// temperature schedule (a positive temperature pins a single-temperature
  /// decode; greedy keeps the default fallback schedule), the
  /// compression-ratio / logprob / no-speech thresholds, the
  /// `condition_on_previous_text` / `initial_prompt` conditioning, the
  /// word-timestamp flag, and the `clip_timestamps` restriction.
  ///
  /// Shared by [`Transcribe::transcribe`] and [`Self::transcribe_detailed`] so
  /// the two entries honor the options identically.
  fn whisper_transcribe_options(&self, opts: &TranscribeOptions) -> WhisperTranscribeOptions {
    let decode = DecodingOptions {
      task: task_to_whisper(opts.task()),
      language: opts.language().map(str::to_owned),
      temperature: opts.temperature(),
      // The universal transcribe options expose no best-of / beam knobs; the
      // high-level path runs the single-sequence decode.
      best_of: None,
      beam_size: None,
      length_penalty: None,
      sample_len: opts.max_new_tokens(),
      prompt: Vec::new(),
      prefix: Vec::new(),
      suppress_tokens: SuppressSpec::NonSpeech,
      suppress_blank: true,
      without_timestamps: opts.no_timestamps(),
      max_initial_timestamp: Some(1.0),
    };
    let temperatures = if opts.temperature() > 0.0 {
      vec![opts.temperature()]
    } else {
      decoding::DEFAULT_TEMPERATURES.to_vec()
    };
    WhisperTranscribeOptions {
      decode,
      temperatures,
      compression_ratio_threshold: opts.compression_ratio_threshold(),
      logprob_threshold: opts.logprob_threshold(),
      no_speech_threshold: opts.no_speech_threshold(),
      word_timestamps: opts.word_timestamps(),
      condition_on_previous_text: opts.condition_on_previous_text(),
      initial_prompt: opts.initial_prompt().map(str::to_owned),
      clip_timestamps: opts.clip_timestamps().to_vec(),
      ..WhisperTranscribeOptions::default()
    }
  }

  /// Transcribe a mono waveform and return the **rich** Whisper result — the
  /// per-segment token ids, seek-derived time offsets, the decode statistics
  /// (avg-logprob / no-speech / compression-ratio / temperature), and (when
  /// [`TranscribeOptions::word_timestamps`] is set) the per-word cross-attention
  /// timings — fields the universal [`Transcription`] cannot hold.
  ///
  /// This is the inherent analogue of
  /// [`SenseVoiceModel::transcribe_rich`](super::super::sensevoice::model::SenseVoiceModel::transcribe_rich):
  /// the universal [`Transcribe::transcribe`] returns the standard
  /// [`Transcription`] (text + language + segment spans), and a caller wanting
  /// the richer per-segment data calls this instead. It honors EVERY universal
  /// option (the same internal option mapping the universal path uses), so the
  /// two entries decode identically and only differ in the result richness.
  ///
  /// Requires an attached tokenizer ([`Self::with_tokenizer`]); the lower-level
  /// [`super::decoding::transcribe`] takes an explicit tokenizer for a model
  /// loaded without one.
  ///
  /// # Errors
  /// - [`Error::InvariantViolation`] if no tokenizer is attached;
  /// - propagates the waveform validation, front-end, encoder, decoder, and
  ///   decoding-task op errors (the same set as [`Transcribe::transcribe`]).
  pub fn transcribe_detailed(
    &self,
    audio: &Array,
    opts: &TranscribeOptions,
  ) -> Result<WhisperTranscription> {
    let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "WhisperModel::transcribe_detailed",
        "requires an attached tokenizer (use WhisperModel::with_tokenizer, or the \
           lower-level audio::stt::models::whisper::decoding::transcribe with an \
           explicit HFTokenizerWrapper)",
      ))
    })?;
    let wrapper = HFTokenizerWrapper::new(
      tokenizer,
      self.dims.is_multilingual(),
      self.dims.num_languages(),
      opts.language(),
      task_to_whisper(opts.task()),
    )?;

    let mel = log_mel_spectrogram_whisper(audio, self.dims.n_mels(), N_SAMPLES)?;
    let content_frames = mel.shape()[0].saturating_sub(N_FRAMES);

    let whisper_opts = self.whisper_transcribe_options(opts);
    let result = decoding::transcribe(
      &self.backend(whisper_opts.word_timestamps, whisper_opts.decode.best_of),
      &wrapper,
      &mel,
      content_frames,
      &whisper_opts,
    )?;
    Ok(WhisperTranscription::from_result(result))
  }
}

// ───────────────────── rich transcription result (PR-E) ────────────────────

/// One word's cross-attention timing on a [`WhisperSegment`] — the public
/// analogue of the decoding-layer [`super::decoding::Word`], surfaced only when
/// [`TranscribeOptions::word_timestamps`] is set.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperWord {
  /// The word text (including any leading space / merged punctuation).
  word: String,
  /// The word start time in seconds (absolute, including the segment offset).
  start: f64,
  /// The word end time in seconds (absolute).
  end: f64,
  /// The mean per-token probability of the word's tokens.
  probability: f64,
}

impl WhisperWord {
  /// The word text.
  #[inline(always)]
  pub fn word(&self) -> &str {
    &self.word
  }

  /// The word start time in seconds (absolute).
  #[inline(always)]
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// The word end time in seconds (absolute).
  #[inline(always)]
  pub const fn end(&self) -> f64 {
    self.end
  }

  /// The mean per-token probability of the word's tokens.
  #[inline(always)]
  pub const fn probability(&self) -> f64 {
    self.probability
  }
}

/// One rich Whisper segment — the per-window decode result with the fields the
/// universal [`Segment`] cannot hold: the sampled **token ids**, the
/// seek-derived **time offsets** (`start` / `end`), the decode statistics
/// (temperature / avg-logprob / no-speech-prob / compression-ratio), and the
/// per-word timings (when word timestamps were requested).
#[derive(Debug, Clone)]
pub struct WhisperSegment {
  start: f64,
  end: f64,
  text: String,
  tokens: Vec<u32>,
  temperature: f32,
  avg_logprob: f64,
  no_speech_prob: f64,
  compression_ratio: f64,
  words: Vec<WhisperWord>,
}

impl WhisperSegment {
  /// The segment start time in seconds (the seek-derived window offset).
  #[inline(always)]
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// The segment end time in seconds.
  #[inline(always)]
  pub const fn end(&self) -> f64 {
    self.end
  }

  /// The decoded text for this segment.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The sampled token ids for this segment.
  #[inline(always)]
  pub fn tokens(&self) -> &[u32] {
    &self.tokens
  }

  /// The temperature this segment was decoded at.
  #[inline(always)]
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  /// The mean token log-probability for this segment.
  #[inline(always)]
  pub const fn avg_logprob(&self) -> f64 {
    self.avg_logprob
  }

  /// The no-speech probability for this segment.
  #[inline(always)]
  pub const fn no_speech_prob(&self) -> f64 {
    self.no_speech_prob
  }

  /// The compression ratio of this segment's text.
  #[inline(always)]
  pub const fn compression_ratio(&self) -> f64 {
    self.compression_ratio
  }

  /// The per-word timings — empty unless [`TranscribeOptions::word_timestamps`]
  /// was set.
  #[inline(always)]
  pub fn words(&self) -> &[WhisperWord] {
    &self.words
  }
}

/// The rich Whisper transcription returned by [`WhisperModel::transcribe_detailed`]
/// — the full text, the detected/configured language, and the per-window
/// [`WhisperSegment`]s carrying the token ids, seek offsets, decode statistics,
/// and per-word timings the universal [`Transcription`] omits.
///
/// The universal [`Transcribe::transcribe`] returns the standard
/// [`Transcription`]; this is the model-local rich result (mirroring
/// [`SenseVoiceResult`](super::super::sensevoice::model::SenseVoiceResult)).
#[derive(Debug, Clone)]
pub struct WhisperTranscription {
  text: String,
  language: String,
  segments: Vec<WhisperSegment>,
}

impl WhisperTranscription {
  /// Convert a decoding-layer [`super::decoding::TranscribeResult`] into the
  /// public rich result.
  fn from_result(result: decoding::TranscribeResult) -> Self {
    let segments = result
      .segments
      .into_iter()
      .map(|s| WhisperSegment {
        start: s.start,
        end: s.end,
        text: s.text,
        tokens: s.tokens,
        temperature: s.temperature,
        avg_logprob: s.avg_logprob,
        no_speech_prob: s.no_speech_prob,
        compression_ratio: s.compression_ratio,
        words: s
          .words
          .into_iter()
          .map(|w| WhisperWord {
            word: w.word,
            start: w.start,
            end: w.end,
            probability: w.probability,
          })
          .collect(),
      })
      .collect();
    Self {
      text: result.text,
      language: result.language,
      segments,
    }
  }

  /// The full transcribed text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The detected / configured language code.
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }

  /// The per-window rich segments.
  #[inline(always)]
  pub fn segments(&self) -> &[WhisperSegment] {
    &self.segments
  }
}

/// Read and merge every `*.safetensors` in `dir`. `MissingKey` if there is
/// none.
///
/// Shards are merged with [`crate::model_validation::insert_unique`] rather than
/// a last-wins `HashMap::extend`: a malformed / accidentally-duplicated sharded
/// checkpoint that defines the SAME tensor key in two shard files would
/// otherwise silently keep the later-sorted shard's tensor and decode with
/// shadowed parameters. The duplicate is surfaced as a typed error instead.
///
/// # Errors
/// - [`Error::FileIo`] if the model directory cannot be opened, or if any
///   directory entry cannot be read while listing it (a per-entry walk error is
///   propagated, not silently skipped, so an incomplete shard set fails closed
///   rather than loading a partial model);
/// - [`Error::MissingKey`] if `dir` holds no `*.safetensors`;
/// - [`Error::LayerKeyed`] (`layer` = the offending shard file name) wrapping an
///   [`Error::KeyCollision`] (naming the duplicated tensor key) if two shards
///   define the same key;
/// - [`Error::AllocFailure`] if the merged map's reservation cannot be served;
/// - propagates [`crate::io::load_safetensors`] read errors.
fn load_all_safetensors(dir: &std::path::Path) -> Result<HashMap<String, Array>> {
  use crate::error::MissingKeyPayload;
  // Collect EVERY directory entry, failing closed on the FIRST per-entry read
  // error (a flaky / network filesystem, or a directory changed mid-walk) rather
  // than dropping the failed entry — silently skipping it could load the model
  // from an incomplete shard set. The per-entry error reuses the same
  // `FileIoPayload` / `FileOp::Read` idiom as the `read_dir` open above. The
  // `.safetensors` extension filter runs only after the whole listing succeeds.
  let entries = std::fs::read_dir(dir).map_err(|e| {
    Error::FileIo(crate::error::FileIoPayload::new(
      "WhisperModel::load: read model directory",
      crate::error::FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  let mut files: Vec<std::path::PathBuf> = entries
    .map(|entry| {
      entry.map(|e| e.path()).map_err(|e| {
        Error::FileIo(crate::error::FileIoPayload::new(
          "WhisperModel::load: read model directory entry",
          crate::error::FileOp::Read,
          dir.to_path_buf(),
          e,
        ))
      })
    })
    .collect::<Result<Vec<_>>>()?;
  files.retain(|p| p.extension().is_some_and(|ext| ext == "safetensors"));
  files.sort();
  if files.is_empty() {
    return Err(Error::MissingKey(MissingKeyPayload::new(
      "WhisperModel::load: no *.safetensors in model directory",
      dir.to_string_lossy().into_owned(),
    )));
  }
  let mut all = HashMap::new();
  for f in &files {
    let shard = crate::io::load_safetensors(f)?;
    for (key, value) in shard {
      // Fail closed on a cross-shard duplicate key instead of letting the
      // later-sorted shard silently overwrite the earlier tensor. On a
      // `KeyCollision` the shard file name is attached as the `LayerKeyed`
      // context so the error names BOTH the duplicated key and its source shard.
      crate::model_validation::insert_unique(
        &mut all,
        key,
        value,
        "WhisperModel::load: duplicate tensor key across shards",
      )
      .map_err(|e| match e {
        Error::KeyCollision(_) => {
          Error::LayerKeyed(LayerKeyedPayload::new(f.to_string_lossy().into_owned(), e))
        }
        other => other,
      })?;
    }
  }
  Ok(all)
}

/// Apply the word-timing alignment heads from `<dir>/generation_config.json`,
/// if present — the model-load half of `set_alignment_heads`
/// (`whisper.py:704-715`). An absent `generation_config.json` (or one with no
/// `alignment_heads` key) leaves `model`'s last-half default in place; a present
/// override replaces it.
///
/// The `generation_config.json` is read with the shared bounded config reader
/// (capped at the `lm::load` `MAX_CONFIG_BYTES` ceiling), so a hostile model
/// directory cannot OOM the loader by planting a huge file.
///
/// # Errors
/// - propagates the bounded reader's typed errors (`FileIo`, `CapExceeded`,
///   non-UTF-8 `Parse`);
/// - [`Error::Parse`] for a malformed `generation_config.json` body or a
///   malformed `alignment_heads` value;
/// - [`Error::OutOfRange`] for an alignment head outside the decoder grid.
fn apply_dir_alignment_heads(model: WhisperModel, dir: &std::path::Path) -> Result<WhisperModel> {
  let path = dir.join("generation_config.json");
  let Some(text) = crate::lm::load::read_bounded_config_file(&path, "Whisper generation config")?
  else {
    return Ok(model);
  };
  let config: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
    Error::Parse(crate::error::ParsePayload::new(
      "Whisper generation_config.json",
      "generation_config.json",
      e.to_string(),
    ))
  })?;
  match AlignmentHeads::from_generation_config(&config, model.dims())? {
    // `from_generation_config` validated against `model.dims()`, so the carried
    // grid matches and the install-time dims recheck always passes here; `?`
    // surfaces it as a typed error rather than panicking if that ever changes.
    Some(heads) => model.with_alignment_heads(heads),
    None => Ok(model),
  }
}

/// Auto-attach a CoreML / Neural-Engine sibling backend if `dir` carries a
/// `.mlmodelc` bundle (Apple Silicon) — the load-time arm of the backend
/// auto-selection. Off Apple Silicon (and whenever no bundle is present) this is
/// an identity pass-through, so the MLX path is byte-identical.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn maybe_attach_coreml(model: WhisperModel, dir: &std::path::Path) -> Result<WhisperModel> {
  if super::coreml::CoreMlWhisper::is_present(dir) {
    let coreml = super::coreml::CoreMlWhisper::load(dir)?;
    Ok(model.with_coreml(coreml))
  } else {
    Ok(model)
  }
}

/// Off-Apple-Silicon stub: no CoreML backend exists, so the model is returned
/// unchanged (the MLX path).
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[inline]
fn maybe_attach_coreml(model: WhisperModel, _dir: &std::path::Path) -> Result<WhisperModel> {
  Ok(model)
}

// ───────────────────────── sanitize (whisper.py:539-606) ──────────────────

/// The HF→MLX key-remap table (`whisper.py:550-572`). Order matters: more
/// specific patterns precede generic ones. A `None` target marks a key to
/// drop (the encoder positional embedding is recomputed via sinusoids).
const KEY_MAP: &[(&str, Option<&str>)] = &[
  ("encoder.embed_positions.weight", None),
  (
    "decoder.embed_positions.weight",
    Some("decoder.positional_embedding"),
  ),
  ("encoder.layer_norm.", Some("encoder.ln_post.")),
  ("decoder.layer_norm.", Some("decoder.ln.")),
  ("encoder.layers.", Some("encoder.blocks.")),
  ("decoder.layers.", Some("decoder.blocks.")),
  (".self_attn_layer_norm.", Some(".attn_ln.")),
  (".final_layer_norm.", Some(".mlp_ln.")),
  (".encoder_attn_layer_norm.", Some(".cross_attn_ln.")),
  (".fc1.", Some(".mlp1.")),
  (".fc2.", Some(".mlp2.")),
  (".self_attn.q_proj.", Some(".attn.query.")),
  (".self_attn.k_proj.", Some(".attn.key.")),
  (".self_attn.v_proj.", Some(".attn.value.")),
  (".self_attn.out_proj.", Some(".attn.out.")),
  (".encoder_attn.q_proj.", Some(".cross_attn.query.")),
  (".encoder_attn.k_proj.", Some(".cross_attn.key.")),
  (".encoder_attn.v_proj.", Some(".cross_attn.value.")),
  (".encoder_attn.out_proj.", Some(".cross_attn.out.")),
  ("decoder.embed_tokens.", Some("decoder.token_embedding.")),
];

/// The `.weight` suffix mlx stores dense Linear / Embedding matrices under.
/// Appended to a per-layer quantization **path** so it remaps through the same
/// `.<leaf>.`-boundary [`KEY_MAP`] patterns a `<path>.weight` key does, then
/// stripped back off ([`normalize_quant_keys`]).
const WEIGHT_SUFFIX: &str = ".weight";

/// Apply the HuggingFace→MLX key remap to a single weight key — the per-key
/// core of [`sanitize`], factored out so the per-layer quantization config can
/// be renamed through the **exact same** transform its weights are
/// ([`normalize_quant_keys`]).
///
/// Strips the leading `model.` prefix (present on HF checkpoints) and then
/// applies the [`KEY_MAP`] substitutions in table order. Returns `None` for a
/// key whose [`KEY_MAP`] target is `None` (dropped — e.g. the encoder
/// positional embedding, recomputed via sinusoids); `Some(remapped)`
/// otherwise.
///
/// Idempotent on an already-MLX key: a key with no `model.` prefix that matches
/// no HF-form `KEY_MAP` pattern (the `.self_attn.q_proj.`-style left-hand
/// sides) is returned unchanged, so running it twice — or running it over a
/// config that already carries MLX-native paths — does not double-remap.
fn remap_hf_key(mut key: String) -> Option<String> {
  if let Some(stripped) = key.strip_prefix("model.") {
    key = stripped.to_string();
  }
  for (old, new) in KEY_MAP {
    if key.contains(old) {
      // A `None` target drops the key entirely (the encoder positional
      // embedding); `?` propagates that as this function's `None`.
      let replacement = (*new)?;
      key = key.replace(old, replacement);
    }
  }
  Some(key)
}

/// Normalize a [`PerLayerQuantization`] config's per-layer override keys into
/// the **same namespace** as the [`sanitize`]d weight-lookup prefixes, so an
/// HF-format quantized checkpoint's per-layer overrides match the sanitized
/// module paths the builder resolves against (mlx-lm / mlx-audio resolve the
/// per-layer `quantization` map against the post-`sanitize` module-tree path
/// `p`, `mlx_lm/utils.py:349-352`, `mlx_audio/utils.py:243-244`).
///
/// The builder calls [`PerLayerQuantization::quantization_for`] with the
/// **sanitized** MLX prefix (e.g. `encoder.blocks.0.attn.query`,
/// `decoder.token_embedding`), but the parsed config keys an HF checkpoint
/// ships are raw HF paths (e.g.
/// `model.encoder.layers.0.self_attn.q_proj`, `decoder.embed_tokens`). Without
/// this remap a per-layer override (a different `group_size`/`bits` for one
/// `Linear` or for the token embedding) silently misses and the builder falls
/// back to the global scheme — rejecting a valid mixed-quant checkpoint at
/// [`Builder::check_quantized_shape`], or loading it with the wrong scheme.
///
/// Each per-layer key is renamed through [`remap_hf_key`] by appending the
/// [`WEIGHT_SUFFIX`] sentinel (the config key is a module **path**, but the
/// `KEY_MAP` left-hand sides match on the `.<leaf>.` boundary, exactly as a
/// `<path>.weight` key presents), remapping, then stripping the sentinel back
/// off — yielding byte-identical naming to the weight remap. The global default
/// and the `Skip`/`Quantize` override value are preserved verbatim.
///
/// Normalization is **unconditional** — it runs for every config regardless of
/// the checkpoint's weight-map format, because the config carries HF-named or
/// MLX-named keys independently of how the weights are named (an MLX-native
/// weight map can still ship a config whose per-layer keys are raw HF paths).
/// [`remap_hf_key`] is idempotent: an already-sanitized MLX key maps to itself
/// (no `model.` prefix to strip, no `KEY_MAP` left-hand side to match) and an HF
/// key maps to its MLX form. So normalizing every per-layer key is correct in
/// both namespaces — HF keys become the MLX names the builder's lookups use, and
/// MLX keys pass through unchanged — and the collision check below runs in every
/// case, closing the key-namespace mismatch through every path rather than only
/// the HF-weight path.
///
/// Two source paths can remap onto the **same** sanitized path (e.g. a mixed
/// config carrying both the raw HF key `model.encoder.layers.0.self_attn.q_proj`
/// and its already-sanitized MLX alias `encoder.blocks.0.attn.query` for one
/// layer). The collision is resolved deterministically — never by an arbitrary
/// `HashMap`-iteration-order survivor:
/// - if the two overrides are **identical** (same `Skip`, or same
///   `group_size`/`bits`/`mode` `Quantize`) the duplicate is harmless and one is
///   kept;
/// - if they **conflict** (`Skip` vs `Quantize`, or differing
///   `group_size`/`bits`/`mode`) the config is ambiguous for that layer — a
///   single coherent per-layer scheme is required — and an [`Error::KeyCollision`]
///   naming the sanitized layer key is returned (fail fast, rather than load with
///   an arbitrary scheme and trip `check_quantized_shape` unpredictably).
///
/// Because identical-value collisions converge and conflicting-value collisions
/// always error, the result does not depend on the source-map iteration order.
fn normalize_quant_keys(quant: &PerLayerQuantization) -> Result<PerLayerQuantization> {
  let src = quant.per_layer_ref();
  let mut per_layer: HashMap<String, crate::lm::quant::QuantizationOption> = HashMap::new();
  crate::model_validation::reserve_or_error(
    &mut per_layer,
    "normalized per-layer quantization keys",
    src.len(),
  )?;
  for (path, opt) in src {
    // The config key is a module PATH (no `.weight`); append the weight
    // sentinel so it remaps through the SAME `.<leaf>.`-boundary `KEY_MAP`
    // patterns the weights do, then strip the sentinel back off. A `None`
    // remap (a dropped key) cannot occur for a per-layer override path
    // (`KEY_MAP`'s only `None` target is the encoder positional embedding,
    // never a quantizable Linear/Embedding), but is handled defensively by
    // skipping the entry.
    let with_suffix = format!("{path}{WEIGHT_SUFFIX}");
    let Some(remapped) = remap_hf_key(with_suffix) else {
      continue;
    };
    let normalized = remapped
      .strip_suffix(WEIGHT_SUFFIX)
      .map(str::to_string)
      .unwrap_or(remapped);
    // Collision-aware, order-independent insert: if two source keys normalize
    // to the same sanitized path, the surviving value must not depend on the
    // source-map iteration order. An identical override is a harmless duplicate
    // (keep the existing entry); a conflicting one makes the per-layer scheme
    // for that layer ambiguous, so reject it with a typed error.
    match per_layer.get(&normalized) {
      Some(existing) if *existing == *opt => {}
      Some(_) => {
        return Err(Error::KeyCollision(KeyCollisionPayload::new(
          "whisper per-layer quantization config",
          normalized,
        )));
      }
      None => {
        per_layer.insert(normalized, *opt);
      }
    }
  }
  Ok(PerLayerQuantization::new(quant.quantization, per_layer))
}

/// Sanitize a raw checkpoint for the mlxrs Whisper modules by **key-remap
/// only** — the renaming half of `Model.sanitize` (`whisper.py:539-606`),
/// deliberately split from the tensor-data half (conv transpose + dtype cast):
///
/// 1. detect HuggingFace format (any key starts with `model.`);
/// 2. for HF checkpoints: strip the `model.` prefix and apply the `KEY_MAP`
///    remap, dropping `None`-mapped keys (the encoder positional embedding is
///    recomputed via sinusoids).
///
/// The conv `(out, in, k) -> (out, k, in)` transpose and the `astype` cast are
/// **not** done here: they are deferred to the builder, which performs each only
/// **after** the tensor's shape has been validated against the config-derived
/// extents — so an oversized tensor under a consumed key cannot force a
/// transpose / cast allocation ahead of the shape check. This function therefore
/// never touches tensor data (renames are O(1) and copy no buffer), and returns
/// the renamed map plus `is_hf_format` so the builder knows whether the conv
/// weights still carry the HF layout that needs transposing.
///
/// Each rewritten key is inserted via [`crate::model_validation::insert_unique`],
/// which rejects a duplicate destination key (two source keys collapsing onto
/// one sanitized name) instead of letting a nondeterministic survivor win.
///
/// # Errors
/// - [`Error::AllocFailure`] if the sanitized-map reservation cannot be served;
/// - [`Error::KeyCollision`] if two source keys remap onto the same sanitized
///   key.
pub fn sanitize(weights: HashMap<String, Array>) -> Result<(HashMap<String, Array>, bool)> {
  let is_hf_format = weights.keys().any(|k| k.starts_with("model."));

  // Pre-size the destination fallibly (a typed `AllocFailure` instead of the
  // abort `HashMap::with_capacity` would raise) sized by the checkpoint key
  // count; the per-insert `insert_unique` reservation below then never has to
  // grow it.
  let mut sanitized: HashMap<String, Array> = HashMap::new();
  crate::model_validation::reserve_or_error(&mut sanitized, "sanitized weights", weights.len())?;
  for (mut key, value) in weights {
    if is_hf_format {
      // Strip `model.` and apply `KEY_MAP`; a `None` target drops the key
      // entirely (the same per-key transform `normalize_quant_keys` runs over
      // the per-layer quantization config, so weights and config stay in one
      // namespace).
      let Some(remapped) = remap_hf_key(key) else {
        continue;
      };
      key = remapped;
    }

    // Reject a duplicate destination key instead of silently overwriting: the
    // HF `KEY_MAP` remap can collapse two distinct source keys onto the same
    // sanitized key (a corrupt checkpoint), which would otherwise let an
    // arbitrary survivor win nondeterministically. The tensor is moved in
    // verbatim (still at its checkpoint layout / dtype) — the builder validates
    // its shape and then transposes / casts it.
    crate::model_validation::insert_unique(&mut sanitized, key, value, "WhisperModel::sanitize")?;
  }
  Ok((sanitized, is_hf_format))
}

// ───────────────────────── sub-module builder ─────────────────────────────

/// The Whisper convolution kernel width — both `conv1` and `conv2` are kernel-3
/// `nn.Conv1d`s (`whisper.py:411-412`). Used to pin the post-sanitize MLX
/// channels-last `(C_out, K, C_in)` conv weight shape at build.
const CONV_KERNEL: i32 = 3;

/// The transformer MLP expansion ratio — the `ResidualAttentionBlock` builds
/// `mlp1: n_state -> 4 * n_state` (`whisper.py:380`), so the consumed `mlp1` /
/// `mlp2` weights are sized against this `4 * n_state` hidden width.
const MLP_RATIO: i32 = 4;

/// The `4 * n_state` MLP hidden width, computed overflow-safe.
///
/// `n_state` is a validated [`ModelDimensions`] field (`<= MAX_DIM`, `1 << 22`),
/// so `4 * n_state <= 1 << 24` fits `i32` and the `checked_mul` never trips for
/// a real config; the guard is defense-in-depth so a future cap relaxation
/// surfaces as a typed [`Error::ArithmeticOverflow`] rather than a wrapping
/// multiply that could mis-size an expected-shape check.
fn mlp_hidden_dim(n_state: usize) -> Result<i32> {
  i32::try_from(n_state)
    .ok()
    .and_then(|n| n.checked_mul(MLP_RATIO))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::with_operands(
        "WhisperModel builder: 4 * n_state (MLP hidden width)",
        "i32",
        [("n_state", n_state as u64), ("MLP_RATIO", MLP_RATIO as u64)],
      ))
    })
}

/// Consumes a (key-remapped, not-yet-materialized) weight map, popping tensors
/// by key to build the encoder / decoder sub-modules. Popping (rather than
/// borrowing) ensures each weight is used once and frees the map as it goes.
///
/// Each consumed tensor is **shape-validated against the config before it is
/// transposed / cast**, so the materialization (the conv transpose + the
/// `astype` to [`Self::dtype`]) runs only on tensors already proven within the
/// config caps — an oversized tensor under a consumed key is rejected by the
/// shape check ahead of any allocation it would size. Unconsumed tensors are
/// never materialized; they are dropped with the builder.
struct Builder<'q> {
  weights: HashMap<String, Array>,
  /// Whether the source checkpoint was HuggingFace format — the `conv1`/`conv2`
  /// weights then still carry the HF `(out, in, k)` layout and must be
  /// transposed to the MLX `(out, k, in)` layout after their shape is validated.
  /// An already-MLX checkpoint's conv weights are not transposed.
  is_hf_format: bool,
  /// The model compute dtype each consumed tensor is cast to (after its shape is
  /// validated). A `uint32` tensor (quantized / indices) is left integer, as in
  /// the reference.
  dtype: Dtype,
  /// The parsed per-layer quantization config, if the checkpoint is quantized.
  /// Consulted by [`Self::linear`]: when `Some` AND the layer carries a
  /// `<prefix>.scales` sibling, the projection is built quantized (the
  /// `(group_size, bits, mode)` resolved per layer via
  /// [`PerLayerQuantization::quantization_for`]); otherwise dense.
  quantization: Option<&'q PerLayerQuantization>,
}

impl Builder<'_> {
  /// Pop a required weight by key, **still at its checkpoint dtype / layout**
  /// (un-cast, un-transposed). [`Error::MissingKey`] if absent (`MissingKey`
  /// carries the runtime key string, unlike the `&'static str` `MissingField`).
  fn take(&mut self, key: &str) -> Result<Array> {
    self.weights.remove(key).ok_or_else(|| {
      Error::MissingKey(crate::error::MissingKeyPayload::new(
        "WhisperModel: weight not found in checkpoint",
        key.to_string(),
      ))
    })
  }

  /// Cast a **shape-validated** tensor to the model [`Self::dtype`], the
  /// deferred materialization step. The cast is skipped when the tensor is
  /// already in `dtype` or is `uint32` (quantized packed weights / indices stay
  /// integer, as in the reference). Run only after the caller has shape-
  /// validated the tensor, so the `astype` allocation is bounded by the config.
  ///
  /// # Errors
  /// Propagates the dtype-query / `astype` op errors.
  fn cast_to_dtype(&self, value: Array) -> Result<Array> {
    if value.dtype()? != self.dtype && value.dtype()? != Dtype::U32 {
      value.astype(self.dtype)
    } else {
      Ok(value)
    }
  }

  /// Pop a required weight by key AND assert its shape (rank + every
  /// dimension) equals the `expected` shape the validated [`ModelDimensions`]
  /// require, before it is stored or fed to any op.
  ///
  /// The builder reads each module's tensor by key and wires it in verbatim,
  /// while the [`ModelDimensions`] product caps bound only extents *derived
  /// from the config*. A corrupt-but-loadable checkpoint that passes the config
  /// gate could otherwise ship a tensor whose actual shape disagrees with the
  /// config — e.g. a `conv1.weight` declaring a huge output-channel count while
  /// `n_audio_state` stays small — and the forward pass would then materialize
  /// an activation sized by the *tensor* (`N_FRAMES * actual_out_channels`),
  /// not by the capped config product. Pinning every consumed tensor to its
  /// exact config-derived shape closes that gap: the validated caps then
  /// provably equal the actual runtime tensor extents, and a mismatch fails
  /// fast here with a typed error before any module is built.
  ///
  /// The expected dims are computed from the already-`validate`d
  /// [`ModelDimensions`] (each `<= MAX_DIM`, so the `as i32` widening the
  /// callers use is lossless and non-negative). The length comparison pins the
  /// rank, so this single helper covers both the rank and the exact-shape
  /// requirements. On mismatch returns an [`Error::ShapePairMismatch`] carrying
  /// both full shapes, wrapped in an [`Error::LayerKeyed`] naming the offending
  /// tensor `key` (the dynamic per-layer key the `&'static` `descriptor` cannot
  /// carry).
  ///
  /// The validation runs on the **raw** tensor (still at its checkpoint dtype);
  /// only after the shape passes is the tensor cast to the model dtype, so the
  /// `astype` allocation is bounded by the config. Every consumed non-conv
  /// tensor's raw shape already equals its MLX shape (no transpose), so a single
  /// expected shape covers it. The conv weights, whose raw HF layout differs from
  /// the MLX layout, go through [`Self::take_conv_weight`] instead.
  fn take_shaped(
    &mut self,
    key: &str,
    descriptor: &'static str,
    expected: &[i32],
  ) -> Result<Array> {
    let tensor = self.take(key)?;
    // Shape-validate the RAW tensor before any materialization.
    Self::check_shape(key, descriptor, &tensor, expected)?;
    // Now bounded by the config: cast to the model dtype.
    self.cast_to_dtype(tensor)
  }

  /// Pop and shape-validate a `conv1`/`conv2` weight, then materialize it into
  /// the MLX channels-last `(out, kernel, in)` layout the [`crate::ops::conv`]
  /// path expects.
  ///
  /// The validation runs on the **raw checkpoint layout**, which differs by
  /// source format: a HuggingFace conv weight is `(out, in, kernel)` and a
  /// native-MLX one is already `(out, kernel, in)`. Validating the raw shape
  /// (against the format-appropriate expected permutation) BEFORE the transpose
  /// closes the oversized-conv-weight gap — a tensor declaring a huge
  /// out-channel count is rejected here, before the `(out, in, k) -> (out, k,
  /// in)` transpose and the `astype` it would otherwise size allocate. Only after
  /// the shape passes is the (HF) tensor transposed and contiguous-materialized,
  /// then cast to the model dtype.
  ///
  /// `out` / `kernel` / `in_c` are the config-derived MLX-layout dimensions
  /// (`(out, kernel, in_c)`); for an HF checkpoint the raw expectation is the
  /// `(out, in_c, kernel)` permutation.
  fn take_conv_weight(
    &mut self,
    key: &str,
    descriptor: &'static str,
    out: i32,
    kernel: i32,
    in_c: i32,
  ) -> Result<Array> {
    let tensor = self.take(key)?;
    if self.is_hf_format {
      // HF raw layout is `(out, in, kernel)`. Validate that, THEN transpose to
      // the MLX `(out, kernel, in)` and materialize the strided view contiguous.
      Self::check_shape(key, descriptor, &tensor, &[out, in_c, kernel])?;
      let transposed = crate::ops::shape::contiguous(&tensor.transpose_axes(&[0, 2, 1])?, false)?;
      self.cast_to_dtype(transposed)
    } else {
      // Already-MLX layout `(out, kernel, in)`: no transpose, validate directly.
      Self::check_shape(key, descriptor, &tensor, &[out, kernel, in_c])?;
      self.cast_to_dtype(tensor)
    }
  }

  /// Assert a tensor's shape (rank + every dimension) equals the `expected`
  /// shape, returning a keyed [`Error::ShapePairMismatch`] otherwise. The
  /// pure check shared by [`Self::take_shaped`] and [`Self::take_conv_weight`],
  /// run on the raw tensor before any materialization.
  fn check_shape(
    key: &str,
    descriptor: &'static str,
    tensor: &Array,
    expected: &[i32],
  ) -> Result<()> {
    let actual = tensor.shape();
    // Compare in i64 so the usize dims and the i32 expectations both widen
    // losslessly (real MLX dims are i32-bounded); the length check also pins
    // the rank. A negative expected dim is a builder bug and never matches.
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

  /// Validate a quantized layer's packed `<prefix>.weight` + `<prefix>.scales`
  /// against the config-derived `(out, in_features)` BEFORE the quantized layer
  /// is constructed — the quantized analogue of [`Self::check_shape`].
  ///
  /// The dense path pins every consumed tensor to its exact config shape via
  /// [`Self::take_shaped`]; the quantized path must reach the same load-time
  /// gate, because a corrupt quantized checkpoint could otherwise ship a packed
  /// weight whose *logical* output or vocab dimension disagrees with the config,
  /// and the first forward would then size projections / logits from the
  /// checkpoint tensors instead of the validated config. The packed `uint32`
  /// weight has a different shape than the dense `(out, in)`, so the recovery
  /// mirrors mlx's quantized layout (`mlx/ops.cpp:107,131,4790-4792`):
  ///
  /// - the weight is rank-2 `uint32` `(out, in * bits / 32)`; its *logical*
  ///   output dim is the leading axis and must equal `out`, and its *logical*
  ///   input width — mlx's `w_inner_dims = w.shape(-1) * 32 / bits`, the
  ///   dimension `quantized_matmul` contracts against — must equal `in_features`;
  /// - the `scales` are rank-2 `(out, in / group_size)`; the leading axis must
  ///   equal `out`, and `scales.shape(-1) * group_size` must equal `in_features`
  ///   (mlx's `w.shape(-1) * 32 / bits == scales.shape(-1) * group_size`
  ///   invariant). Validating both the packed-weight and the scales recovery
  ///   here turns a malformed quantized checkpoint into a typed error at load
  ///   time rather than an opaque mlx-c matmul failure on the first forward.
  ///
  /// `group_size` / `bits` are the per-layer-resolved scheme parameters; both
  /// are checked `> 0` before they divide (a non-positive value is a malformed
  /// config and a [`Error::OutOfRange`], never a panic). The per-mode value
  /// tables remain mlx-c's; this only pins the structural relationship to the
  /// config the dense gate also enforces.
  ///
  /// On mismatch returns an [`Error::ShapePairMismatch`] (or [`Error::RankMismatch`]
  /// / [`Error::InvariantViolation`] for a wrong rank / dtype) wrapped in an
  /// [`Error::LayerKeyed`] naming the offending `<prefix>.weight` /
  /// `<prefix>.scales` key. Reads only `shape()` / `dtype()` metadata (no
  /// materialization), so it is bounded regardless of the declared dims.
  fn check_quantized_shape(
    &self,
    prefix: &str,
    descriptor: &'static str,
    out: i32,
    in_features: i32,
    group_size: i32,
    bits: i32,
  ) -> Result<()> {
    // The scheme parameters divide the recovered widths; a non-positive value
    // is a malformed config (`from_parts` also rejects it, but guard here so
    // the divisions below cannot trap).
    if bits <= 0 {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel: quantized layer bits",
        "must be > 0",
        smol_str::format_smolstr!("{bits}"),
      )));
    }
    if group_size <= 0 {
      return Err(Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel: quantized layer group_size",
        "must be > 0",
        smol_str::format_smolstr!("{group_size}"),
      )));
    }

    // Packed weight `(out, in * bits / 32)`, `uint32`.
    let weight_key = format!("{prefix}.weight");
    let weight = self.weights.get(&weight_key).ok_or_else(|| {
      Error::MissingKey(crate::error::MissingKeyPayload::new(
        "WhisperModel: quantized weight not found in checkpoint",
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
    if weight.dtype()? != Dtype::U32 {
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
    let scales = self.weights.get(&scales_key).ok_or_else(|| {
      Error::MissingKey(crate::error::MissingKeyPayload::new(
        "WhisperModel: quantized scales not found in checkpoint",
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

  /// Build a [`Linear`] from `<prefix>.weight` (+ the dense `<prefix>.bias`
  /// when the architecture `bias` flag is `true`).
  ///
  /// **Quantized path** — when the checkpoint carries a `<prefix>.scales`
  /// sibling (PRESENCE ALONE, the same `.scales` discriminator the shared
  /// [`MaybeQuantizedLinear::from_weights`] uses): the projection is built
  /// quantized via [`crate::nn::QuantizedLinear::from_parts`], running
  /// [`crate::ops::quantized::quantized_matmul`] over the packed
  /// `(weight, scales, biases)` triple with the per-layer-resolved
  /// `(group_size, bits, mode)`. The packed `uint32` weight has shape
  /// `(out, in * bits / 32)`, NOT the dense `(out, in)`, so it does **not** go
  /// through the dense `take_shaped` config-shape check — its structural
  /// consistency is validated by [`crate::nn::QuantizedLinear::from_parts`]
  /// instead. This mirrors mlx-audio's whisper `class_predicate`
  /// (`f"{p}.scales" in weights`), with the global `quantization` block's
  /// scheme parameters (`whisper.py:674-676`).
  ///
  /// The dense output `<prefix>.bias` is loaded with the **same arity** the
  /// dense path enforces: mlx's `QuantizedLinear.from_linear` preserves the
  /// source `Linear.bias`, so a faithful quantized Whisper checkpoint carries
  /// `query.bias` / `value.bias` / `out.bias` / `mlp1.bias` / `mlp2.bias`. When
  /// the architecture `bias` flag is `true` this path therefore **requires**
  /// `<prefix>.bias`, validates it as `(out,)`, casts it through the same
  /// builder dtype path as the dense bias, and passes it as the explicit dense
  /// bias to [`crate::nn::QuantizedLinear::from_parts`] — NOT via the
  /// optional-bias [`MaybeQuantizedLinear::from_weights`] convenience, so a
  /// quantized checkpoint missing a required bias fails fast with the same
  /// typed [`Error::MissingKey`] / [`Error::ShapePairMismatch`] the dense path
  /// returns, instead of silently loading a biasless projection.
  ///
  /// **Dense path** — otherwise: pops `<prefix>.weight` `(out, in)` (+ the
  /// `<prefix>.bias` `(out,)` when `bias` is `true`), both shape-validated
  /// against the config-derived extents BEFORE materialization, exactly as
  /// before.
  ///
  /// `bias = false` for the Whisper attention `key` projection
  /// (`whisper.py:333`); on both paths that projection carries no dense output
  /// bias and any stray `<prefix>.bias` is dropped unused.
  fn linear(&mut self, prefix: &str, out: i32, in_features: i32, bias: bool) -> Result<Linear> {
    // `<prefix>.scales` PRESENCE ALONE is the load-bearing "this layer is
    // quantized" signal (mlx-audio whisper `class_predicate`) — the same
    // discriminator the shared `MaybeQuantizedLinear::from_weights` uses. A
    // layer carrying `.scales` takes the quantized path even when no quant
    // config is threaded; the `let-else` below then surfaces the missing scheme
    // as a typed error rather than silently reinterpreting the packed weight as
    // dense.
    let scales_key = format!("{prefix}.scales");
    if self.weights.contains_key(&scales_key) {
      // Resolve the per-layer `(group_size, bits, mode)` from the config. No
      // resolvable scheme — `self.quantization == None` (no config threaded at
      // all), an explicit per-layer `Skip`, or no global default — next to a
      // present `.scales` is a config/checkpoint inconsistency: a typed error,
      // never a guessed scheme nor a silent dense reinterpret of the packed
      // weight.
      let Some(q) = self.quantization.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "WhisperModel: Linear carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized Linear requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      // The quantized triple must reach the same config-shape gate the dense
      // `take_shaped` enforces: the packed weight's logical `(out, in)` (and
      // the scales' recovery) must equal the config-derived extents BEFORE
      // construction, so a corrupt quantized checkpoint cannot size the
      // projection from the tensor instead of the config.
      self.check_quantized_shape(
        prefix,
        "quantized Linear weight (out, in)",
        out,
        in_features,
        q.group_size,
        q.bits,
      )?;
      // Load the dense output bias with the SAME arity as the dense branch: when
      // the architecture `bias` flag is `true`, `take_shaped` REQUIRES
      // `<prefix>.bias`, validates it `(out,)`, and casts it through the builder
      // dtype path (a missing key → `Error::MissingKey`, a wrong shape →
      // `Error::ShapePairMismatch`); when `false` (Whisper's attention `key`),
      // any stray `<prefix>.bias` is dropped unused, exactly as the dense path
      // leaves it. The loaded bias is passed as the explicit dense bias to
      // `QuantizedLinear::from_parts` (not auto-detected by `from_weights`), so
      // dense and quantized are arity-identical.
      let dense_bias = if bias {
        Some(self.take_shaped(&format!("{prefix}.bias"), "Linear bias (out,)", &[out])?)
      } else {
        self.weights.remove(&format!("{prefix}.bias"));
        None
      };
      // Pop the packed triple by key (the same key-remap-free consume the dense
      // path uses): the `uint32` weight, the `.scales`, and the per-group affine
      // `.biases` (present iff `mode == "affine"`; `from_parts` enforces the
      // mode/arity contract). `take` returns a typed `MissingKey` if absent.
      let weight = self.take(&format!("{prefix}.weight"))?;
      let scales = self.take(&format!("{prefix}.scales"))?;
      let quant_biases = self.weights.remove(&format!("{prefix}.biases"));
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
    let weight = self.take_shaped(
      &format!("{prefix}.weight"),
      "Linear weight (out, in)",
      &[out, in_features],
    )?;
    let b = if bias {
      Some(self.take_shaped(&format!("{prefix}.bias"), "Linear bias (out,)", &[out])?)
    } else {
      None
    };
    Ok(Linear::new(weight, b))
  }

  /// Build an [`Embedding`] from `<prefix>.weight`.
  ///
  /// **Quantized path** — when the checkpoint carries a `<prefix>.scales`
  /// sibling (PRESENCE ALONE, the same `.scales` discriminator the shared
  /// [`MaybeQuantizedLinear::from_weights`] uses): the table is built
  /// quantized ([`Embedding::quantized`], `mlx.nn.QuantizedEmbedding`), with
  /// the packed `(weight, scales, biases)` triple popped from the map and the
  /// `(group_size, bits, mode)` resolved per layer. The packed `uint32` weight
  /// is `(n_vocab, n_state * bits / 32)`, NOT the dense `(n_vocab, n_state)`,
  /// so it bypasses the dense `take_shaped` config-shape check. This mirrors
  /// mlx-audio's whisper `class_predicate`, which quantizes `nn.Embedding`
  /// (the weight-tied logit head) alongside `nn.Linear` (`whisper.py:674-676`).
  ///
  /// **Dense path** — otherwise: pops the `<prefix>.weight` `(n_vocab,
  /// n_state)`, shape-validated against the config-derived extents.
  fn embedding(
    &mut self,
    prefix: &str,
    descriptor: &'static str,
    n_vocab: i32,
    n_state: i32,
  ) -> Result<Embedding> {
    let scales_key = format!("{prefix}.scales");
    if self.weights.contains_key(&scales_key) {
      // Quantized embedding: `.scales` PRESENCE ALONE routes here (the same
      // discriminator the shared `MaybeQuantizedLinear::from_weights` uses).
      // Resolve the per-layer scheme params, then pop the packed triple. A
      // present `.scales` with no resolvable params — `self.quantization ==
      // None` (no config threaded), an explicit `Skip`, or no global default —
      // is a config/checkpoint inconsistency surfaced as a typed error below,
      // never a silent dense reinterpret of the packed table.
      let Some(q) = self.quantization.and_then(|q| q.quantization_for(prefix)) else {
        return Err(Error::InvariantViolation(InvariantViolationPayload::new(
          "WhisperModel: embedding carries a `.scales` sibling but the quantization config resolved no scheme parameters for this layer",
          "a quantized embedding requires (group_size, bits, mode) from the config `quantization` block",
        )));
      };
      // Same load-time gate as the dense path: `check_quantized_shape` pins the
      // packed table's logical `(n_vocab, n_state)` (and the scales' recovery)
      // to the config-derived extents, and `Embedding::quantized` then validates
      // the triple's structural consistency (mode/bias arity, the biases/scales
      // shape match, the rank/dtype invariants) — the embedding analogue of the
      // `QuantizedLinear::from_parts` gate the dense / quantized Linear path
      // reaches. So a corrupt quantized embedding (wrong logical shape, OR a
      // missing/stale/mis-shaped `.biases`) is a typed load-time error here,
      // never an opaque mlx-c failure on the first gather / logit projection.
      self.check_quantized_shape(prefix, descriptor, n_vocab, n_state, q.group_size, q.bits)?;
      let weight = self.take(&format!("{prefix}.weight"))?;
      let scales = self.take(&format!("{prefix}.scales"))?;
      let biases = self.weights.remove(&format!("{prefix}.biases"));
      return Embedding::quantized(
        weight,
        scales,
        biases,
        q.group_size,
        q.bits,
        q.mode.as_str(),
      );
    }

    // Dense path: shape-validate against the config-derived (n_vocab, n_state).
    let weight = self.take_shaped(&format!("{prefix}.weight"), descriptor, &[n_vocab, n_state])?;
    Ok(Embedding::new(weight))
  }

  /// Build a [`LayerNorm`] from `<prefix>.{weight,bias}` (full affine, the
  /// `nn.LayerNorm` default). Both tensors are shape-validated against the
  /// config-derived `(n_state,)`.
  fn layer_norm(&mut self, prefix: &str, n_state: i32) -> Result<LayerNorm> {
    let weight = self.take_shaped(
      &format!("{prefix}.weight"),
      "LayerNorm weight (n_state,)",
      &[n_state],
    )?;
    let bias = self.take_shaped(
      &format!("{prefix}.bias"),
      "LayerNorm bias (n_state,)",
      &[n_state],
    )?;
    Ok(LayerNorm::new(Some(weight), Some(bias), LAYER_NORM_EPS))
  }

  /// Build one [`MultiHeadAttention`] from `<prefix>.{query,key,value,out}`.
  /// Every projection is a square `(n_state, n_state)` [`Linear`]; the `key`
  /// projection has no bias. Each tensor is shape-validated.
  fn attention(&mut self, prefix: &str, n_head: usize, n_state: i32) -> Result<MultiHeadAttention> {
    let query = self.linear(&format!("{prefix}.query"), n_state, n_state, true)?;
    let key = self.linear(&format!("{prefix}.key"), n_state, n_state, false)?;
    let value = self.linear(&format!("{prefix}.value"), n_state, n_state, true)?;
    let out = self.linear(&format!("{prefix}.out"), n_state, n_state, true)?;
    Ok(MultiHeadAttention::new(n_head, query, key, value, out))
  }

  /// Build one [`ResidualAttentionBlock`]. `cross_attention = true` adds the
  /// `<prefix>.cross_attn` + `<prefix>.cross_attn_ln` sub-modules (decoder
  /// blocks only). Every consumed tensor is shape-validated against the
  /// config-derived widths (`n_state`, the `4 * n_state` MLP hidden).
  fn block(
    &mut self,
    prefix: &str,
    n_head: usize,
    n_state: i32,
    mlp_hidden: i32,
    cross_attention: bool,
  ) -> Result<ResidualAttentionBlock> {
    let attn = self.attention(&format!("{prefix}.attn"), n_head, n_state)?;
    let attn_ln = self.layer_norm(&format!("{prefix}.attn_ln"), n_state)?;
    let cross = if cross_attention {
      let ca = self.attention(&format!("{prefix}.cross_attn"), n_head, n_state)?;
      let cln = self.layer_norm(&format!("{prefix}.cross_attn_ln"), n_state)?;
      Some((ca, cln))
    } else {
      None
    };
    // mlp1: (4 * n_state, n_state); mlp2: (n_state, 4 * n_state).
    let mlp1 = self.linear(&format!("{prefix}.mlp1"), mlp_hidden, n_state, true)?;
    let mlp2 = self.linear(&format!("{prefix}.mlp2"), n_state, mlp_hidden, true)?;
    let mlp_ln = self.layer_norm(&format!("{prefix}.mlp_ln"), n_state)?;
    Ok(ResidualAttentionBlock::new(
      attn, attn_ln, cross, mlp1, mlp2, mlp_ln,
    ))
  }

  /// Build the [`AudioEncoder`] (conv front-end + self-attention blocks +
  /// `ln_post`). Every consumed tensor is shape-validated against the
  /// config-derived encoder widths before the encoder is built.
  fn build_encoder(&mut self, dims: &ModelDimensions) -> Result<AudioEncoder> {
    // Encoder widths (each `<= MAX_DIM`, so `as i32` is lossless).
    let n_mels = dims.n_mels() as i32;
    let n_state = dims.n_audio_state() as i32;
    let mlp_hidden = mlp_hidden_dim(dims.n_audio_state())?;
    // conv1: (n_state, kernel=3, n_mels), conv2: (n_state, kernel=3, n_state)
    // in the MLX channels-last `(C_out, K, C_in)` layout — the conv1 out-channel
    // count is exactly the cap axis (`N_FRAMES * n_state`), so validating it
    // makes the config cap provably bound the runtime conv1 activation. The conv
    // weights are shape-validated against the config-derived (format-appropriate)
    // raw layout BEFORE the HF `(out, in, k) -> (out, k, in)` transpose runs, so
    // an oversized conv weight cannot force that transpose to allocate.
    let conv1_weight = self.take_conv_weight(
      "encoder.conv1.weight",
      "encoder conv1 weight (n_audio_state, 3, n_mels)",
      n_state,
      CONV_KERNEL,
      n_mels,
    )?;
    let conv1_bias = self.take_shaped(
      "encoder.conv1.bias",
      "encoder conv1 bias (n_audio_state,)",
      &[n_state],
    )?;
    let conv2_weight = self.take_conv_weight(
      "encoder.conv2.weight",
      "encoder conv2 weight (n_audio_state, 3, n_audio_state)",
      n_state,
      CONV_KERNEL,
      n_state,
    )?;
    let conv2_bias = self.take_shaped(
      "encoder.conv2.bias",
      "encoder conv2 bias (n_audio_state,)",
      &[n_state],
    )?;

    // `n_audio_layer` is bounded by `MAX_LAYERS` at config construction; reserve
    // the block vector fallibly (an allocator failure is a typed `AllocFailure`,
    // not the abort `Vec::with_capacity` would raise), then push into the
    // reserved capacity so the loop never reallocates.
    let mut blocks = Vec::new();
    crate::model_validation::reserve_or_error(&mut blocks, "encoder blocks", dims.n_audio_layer())?;
    for i in 0..dims.n_audio_layer() {
      blocks.push(self.block(
        &format!("encoder.blocks.{i}"),
        dims.n_audio_head(),
        n_state,
        mlp_hidden,
        false,
      )?);
    }
    let ln_post = self.layer_norm("encoder.ln_post", n_state)?;

    AudioEncoder::new(
      conv1_weight,
      conv1_bias,
      conv2_weight,
      conv2_bias,
      dims.n_audio_ctx(),
      dims.n_audio_state(),
      blocks,
      ln_post,
      self.dtype,
    )
  }

  /// Build the [`TextDecoder`] (token + learned positional embedding,
  /// cross-attention blocks, final `ln`, weight-tied logits). Every consumed
  /// tensor is shape-validated against the config-derived decoder widths — in
  /// particular the token-embedding `(n_vocab, n_text_state)` and the
  /// positional `(n_text_ctx, n_text_state)`, so the weight-tied logit head and
  /// the positional slice provably stay within their config caps.
  fn build_decoder(&mut self, dims: &ModelDimensions, dtype: Dtype) -> Result<TextDecoder> {
    // Decoder widths (each `<= MAX_DIM`, so `as i32` is lossless).
    let n_state = dims.n_text_state() as i32;
    let n_vocab = dims.n_vocab() as i32;
    let n_ctx = dims.n_text_ctx() as i32;
    let mlp_hidden = mlp_hidden_dim(dims.n_text_state())?;

    let token_embedding = self.embedding(
      "decoder.token_embedding",
      "decoder token embedding (n_vocab, n_text_state)",
      n_vocab,
      n_state,
    )?;
    let positional_embedding = self.take_shaped(
      "decoder.positional_embedding",
      "decoder positional embedding (n_text_ctx, n_text_state)",
      &[n_ctx, n_state],
    )?;

    // `n_text_layer` is bounded by `MAX_LAYERS` at config construction; reserve
    // the block vector fallibly (a typed `AllocFailure` instead of an abort),
    // then push into the reserved capacity.
    let mut blocks = Vec::new();
    crate::model_validation::reserve_or_error(&mut blocks, "decoder blocks", dims.n_text_layer())?;
    for i in 0..dims.n_text_layer() {
      blocks.push(self.block(
        &format!("decoder.blocks.{i}"),
        dims.n_text_head(),
        n_state,
        mlp_hidden,
        true,
      )?);
    }
    let ln = self.layer_norm("decoder.ln", n_state)?;

    TextDecoder::new(
      token_embedding,
      positional_embedding,
      blocks,
      ln,
      dims.n_text_ctx(),
      dims.n_vocab(),
      dtype,
    )
  }
}

// ───────────────────────── trait impls ────────────────────────────────────

impl AutoregressiveStt for WhisperModel {
  /// The caller-owned per-block decode KV cache — no model-stored state.
  type Cache = WhisperDecodeCache;

  /// Extract the Whisper log-mel features `(num_frames, n_mels)` from a 1-D
  /// `16 kHz` mono waveform — Whisper's own front-end
  /// ([`log_mel_spectrogram_whisper`]) rather than the generic
  /// [`default_log_mel`](crate::audio::stt::generate::default_log_mel): Slaney
  /// filterbank, `log10`, the dynamic-range clamp, the `(x + 4) / 4` renorm,
  /// and the `(num_frames, n_mels)` layout [`Self::encode`] consumes. `n_mels`
  /// is the checkpoint's value (`80` or `128`); no extra padding is applied
  /// (the 30-second segment framing is the decoding task's responsibility,
  /// not this per-utterance feature extraction).
  ///
  /// # Errors
  /// Propagates [`log_mel_spectrogram_whisper`] (STFT / mel-filterbank /
  /// matmul / reduction op errors; rejects a non-1-D input or a waveform too
  /// short to yield ≥ 1 frame).
  fn log_mel(&self, audio: &Array) -> Result<Array> {
    log_mel_spectrogram_whisper(audio, self.dims.n_mels(), 0)
  }

  /// Encode a Whisper mel `(num_frames, n_mels)` into encoder states
  /// `(1, n_audio_ctx, n_audio_state)`. Forwards to the encoder's `forward`.
  fn encode(&self, mel: &Array) -> Result<Array> {
    self.encoder.forward(mel)
  }

  /// Mint a fresh, empty decode cache (no decoded positions). The model itself
  /// holds no decode state — this value is the caller's.
  fn new_cache(&self) -> Self::Cache {
    WhisperDecodeCache::new()
  }

  /// One decode step over the running token window. `tokens` is the complete
  /// prefix decoded so far; the `cache` carries the per-block KV from the
  /// positions already processed, so only the **new** tail
  /// (`tokens[cache.len()..]`) is forwarded — a fresh cache prefills the whole
  /// prefix, a warm cache forwards just the last token (the reference's
  /// prefill-then-single-step shape, `decoding.py:608`/`:618-631`). Returns the
  /// last-position next-token logits as a rank-1 `(n_vocab,)` row (the
  /// [`greedy_transcribe`](crate::audio::stt::generate::greedy_transcribe)
  /// contract).
  ///
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `enc` is not `(1, n_audio_ctx,
  ///   n_audio_state)` (a longer / batched encoder segment is rejected before
  ///   any allocation);
  /// - [`Error::EmptyInput`] if `tokens` is shorter than the cache (no new
  ///   tokens to decode — a misuse the driver never triggers);
  /// - [`Error::RankMismatch`] if the decoder returns a non-`[1, T, V]` logits
  ///   tensor;
  /// - [`Error::OutOfRange`] if the vocab dimension exceeds `i32::MAX`;
  /// - propagates the embedding / block / LayerNorm / logit op errors.
  fn decode_step(&self, cache: &mut Self::Cache, enc: &Array, tokens: &[u32]) -> Result<Array> {
    // Validate the encoder extent + token-prefix length / ids and take the
    // new-tail slice (the guards fire before any allocation they size).
    // `decode_tokens` builds the `(1, T)` array from this slice and re-checks the
    // value range at the gather chokepoint.
    let new_tokens = self.prepare_decode_tail(cache, enc, tokens)?;

    let (logits, new_cache) = self.decode_tokens(new_tokens, enc, cache.inner.as_ref())?;

    // The decoder returns `[B=1, T, V]`; slice the LAST position and reshape to
    // the rank-1 `[V]` row the greedy driver reads `argmax` over. Run every
    // fallible step (the shape check, the slice, the reshape) BEFORE committing
    // the cache, so a failure here cannot leave `cache.inner` advanced while an
    // `Err` is returned (a retry with the same prefix would then see no new
    // tokens to decode and corrupt the trajectory). The cache mutation is the
    // LAST step, after the row has been built.
    let shape = logits.shape();
    if shape.len() != 3 || shape[0] != 1 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "WhisperModel::decode_step: decoder logits must be [1, T, V]",
        shape.len() as u32,
        shape,
      )));
    }
    let (seq, v) = (shape[1], shape[2]);
    let last = i32::try_from(seq.saturating_sub(1)).map_err(|_| vocab_overflow("T"))?;
    let vi = i32::try_from(v).map_err(|_| vocab_overflow("n_vocab"))?;
    // `logits[0, T-1, :]` → `(V,)`.
    let row = crate::ops::indexing::slice(&logits, &[0, last, 0], &[1, last + 1, vi], &[1, 1, 1])?
      .reshape(&[vi])?;
    cache.inner = Some(new_cache);
    Ok(row)
  }

  /// The full start-of-transcript prompt prefix — the Whisper `sot_sequence`
  /// (`<|startoftranscript|>` + language + task, plus `<|notimestamps|>` when
  /// [`TranscribeOptions::no_timestamps`] is set) for the
  /// [`greedy_transcribe`](crate::audio::stt::generate::greedy_transcribe)
  /// driver to seed from.
  ///
  /// Resolving the real ids requires the attached tokenizer
  /// ([`WhisperModel::with_tokenizer`]); without one this is a typed
  /// [`Error::InvariantViolation`] (the canonical sot id alone cannot stand in
  /// for the full language/task sequence). The high-level
  /// [`Transcribe`] impl runs its own
  /// decoding task and does not route through this hook.
  ///
  /// # Errors
  /// [`Error::InvariantViolation`] if no tokenizer is attached, or the
  /// [`HFTokenizerWrapper`] construction error.
  fn initial_tokens(&self, opts: &TranscribeOptions) -> Result<Vec<u32>> {
    let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "WhisperModel::initial_tokens",
        "requires an attached tokenizer (use WhisperModel::with_tokenizer) to \
           build the start-of-transcript language/task sequence",
      ))
    })?;
    let wrapper = HFTokenizerWrapper::new(
      tokenizer,
      self.dims.is_multilingual(),
      self.dims.num_languages(),
      opts.language(),
      task_to_whisper(opts.task()),
    )?;
    let seq = if opts.no_timestamps() {
      wrapper.sot_sequence_including_notimestamps()
    } else {
      wrapper.sot_sequence()
    };
    Ok(seq)
  }

  /// The end-of-transcript token — `<|endoftext|>`. Returns the attached
  /// tokenizer's id when recorded (via [`WhisperModel::with_tokenizer`] /
  /// [`WhisperModel::with_eot_token`]), else the canonical `50257`.
  fn eot(&self) -> u32 {
    self.eot_token.unwrap_or(EOT_TOKEN_ID)
  }

  /// Whisper's text-decoder context — `n_text_ctx` (the checkpoint's value,
  /// `448` for every released model). Bounds `prompt + generated` in the
  /// greedy driver to the decoder's learned positional table.
  fn max_context(&self) -> usize {
    self.dims.n_text_ctx()
  }

  /// The Whisper mel front-end config: the [`MelConfig::whisper_default`]
  /// (Slaney-scale mel, `n_fft=400`, `hop=160`, `sample_rate=16000`) with
  /// `n_mels` overridden to the checkpoint's value (`80` or `128`). Only used
  /// by the [`Self::log_mel`] default; Whisper overrides `log_mel` wholesale,
  /// so this reports the config for completeness.
  fn mel_config(&self) -> MelConfig {
    MelConfig::whisper_default().with_n_mels(self.dims.n_mels())
  }
}

impl Transcribe for WhisperModel {
  /// Transcribe a mono waveform via Whisper's full decoding task
  /// ([`super::decoding::transcribe`]): the 30-second seek loop, the greedy
  /// decode with the three logit filters, the temperature-fallback schedule,
  /// and (for a multilingual checkpoint with no requested language) language
  /// detection. Reuses the [`AutoregressiveStt`] encoder / decoder internally
  /// while running its own loop rather than the generic
  /// [`greedy_transcribe`](crate::audio::stt::generate::greedy_transcribe).
  ///
  /// `audio` is a mono waveform [`Array`] taken to be at Whisper's `16 kHz`
  /// already — the trait input carries no sample rate, so a caller at another
  /// rate resamples (e.g. via
  /// [`resample_waveform`](crate::audio::stt::generate::resample_waveform))
  /// first. The waveform is framed into the full padded log-mel (a trailing
  /// 30-second pad so the last real window has context), and `content_frames`
  /// excludes that pad.
  ///
  /// # Errors
  /// - [`Error::InvariantViolation`] if no tokenizer is attached (use
  ///   [`WhisperModel::with_tokenizer`], or the lower-level
  ///   [`super::decoding::transcribe`] with an explicit tokenizer);
  /// - propagates the waveform validation, front-end, encoder, decoder, and
  ///   decoding-task op errors.
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription> {
    let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
      Error::InvariantViolation(InvariantViolationPayload::new(
        "WhisperModel::transcribe",
        "requires an attached tokenizer (use WhisperModel::with_tokenizer, or \
           the lower-level audio::stt::models::whisper::decoding::transcribe \
           with an explicit HFTokenizerWrapper)",
      ))
    })?;
    let wrapper = HFTokenizerWrapper::new(
      tokenizer,
      self.dims.is_multilingual(),
      self.dims.num_languages(),
      opts.language(),
      task_to_whisper(opts.task()),
    )?;

    // Frame the FULL waveform into a padded log-mel: a trailing 30-second pad
    // (`log_mel_spectrogram`'s default `padding = N_SAMPLES`) gives the final
    // real window full context; `content_frames` is the real (non-pad) frame
    // count the seek loop is bounded by (`whisper.py:763`).
    let mel = log_mel_spectrogram_whisper(audio, self.dims.n_mels(), N_SAMPLES)?;
    let total_frames = mel.shape()[0];
    let content_frames = total_frames.saturating_sub(N_FRAMES);

    let whisper_opts = self.whisper_transcribe_options(opts);
    let result = decoding::transcribe(
      &self.backend(whisper_opts.word_timestamps, whisper_opts.decode.best_of),
      &wrapper,
      &mel,
      content_frames,
      &whisper_opts,
    )?;

    // Convert the Whisper transcribe result into the universal `Transcription`
    // (text + language + segment spans). The richer per-segment fields (token
    // ids, word timings, seek-derived offsets) are surfaced through the
    // inherent [`Self::transcribe_detailed`] / [`WhisperTranscription`] instead,
    // mirroring how SenseVoice keeps its rich tags model-local.
    let segments = result
      .segments
      .iter()
      .map(|s| Segment::new(s.text.clone(), s.start, s.end))
      .collect();
    Ok(Transcription::new(
      result.text,
      Some(result.language),
      segments,
    ))
  }
}

/// Convert a universal [`Task`] into the Whisper-internal [`WhisperTask`]
/// (the tokenizer's task slug); the two enums carry the same
/// transcribe/translate distinction.
#[inline(always)]
fn task_to_whisper(task: Task) -> WhisperTask {
  match task {
    Task::Transcribe => WhisperTask::Transcribe,
    Task::Translate => WhisperTask::Translate,
  }
}

/// A decoder vocab / sequence dimension exceeding `i32::MAX`.
fn vocab_overflow(which: &'static str) -> Error {
  Error::OutOfRange(crate::error::OutOfRangePayload::new(
    "WhisperModel::decode_step: dimension",
    "must fit in i32",
    smol_str::format_smolstr!("{which} exceeds i32::MAX"),
  ))
}

/// A model dimension exceeding `i32::MAX` (the broadcast / batched-decode path).
fn dim_i32_overflow(which: &'static str) -> Error {
  Error::OutOfRange(crate::error::OutOfRangePayload::new(
    "WhisperModel: dimension",
    "must fit in i32",
    smol_str::format_smolstr!("{which} exceeds i32::MAX"),
  ))
}

/// Canonical `<|endoftext|>` id (the multilingual + English-only vocabularies
/// agree on this offset).
const EOT_TOKEN_ID: u32 = 50257;

#[cfg(test)]
mod tests;
