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
    InvariantViolationPayload, LayerKeyedPayload, RankMismatchPayload, ShapePairMismatchPayload,
  },
  lm::nn::norm::LayerNorm,
  tokenizer::Tokenizer,
};

use super::{
  audio::{N_FRAMES, N_SAMPLES, log_mel_spectrogram_whisper},
  config::ModelDimensions,
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
    dims.validate()?;
    // Key-remap ONLY (no transpose / cast): renames are O(1) per key and never
    // touch tensor data, so an oversized tensor is not materialized here.
    let (weights, is_hf_format) = sanitize(weights)?;
    // The builder shape-validates each consumed tensor against the config BEFORE
    // transposing / casting it, so the materialization runs only on tensors
    // already proven within the config caps.
    let mut builder = Builder {
      weights,
      is_hf_format,
      dtype,
    };

    let encoder = builder.build_encoder(&dims)?;
    let decoder = builder.build_decoder(&dims, dtype)?;

    Ok(Self {
      dims,
      dtype,
      encoder,
      decoder,
      tokenizer: None,
      eot_token: None,
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
    Self::from_weights(dims, weights, dtype)
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

  /// Validate that an encoder-states tensor is exactly one Whisper segment —
  /// `(1, n_audio_ctx, n_audio_state)` — before it is fed to the decoder's
  /// cross-attention.
  ///
  /// The decoder's cross-attention projects `xa` and forms its scores from
  /// `xa.shape()[1]` (the key time axis), so a caller that supplies a longer or
  /// batched `enc` would drive the cross-attention KV / score buffers past the
  /// caps the config states for a single `[1, n_audio_ctx, n_audio_state]`
  /// segment (the cross-attention score cap is `n_text_head * n_text_ctx *
  /// n_audio_ctx`, the KV cache cap `n_text_layer * n_audio_ctx * n_audio_state`
  /// — both assume the encoder extent is exactly `n_audio_ctx`). Pinning the
  /// extent here makes those caps provably bound the runtime cross-attention
  /// tensors regardless of caller. The encoder's own `forward` already produces
  /// this shape, but a direct caller of [`Self::decode_tokens`] /
  /// [`AutoregressiveStt::decode_step`] (e.g.
  /// [`super::decoding::detect_language`], which takes `audio_features`
  /// straight from the caller) could pass any tensor — so the guard lives at the
  /// decoder entry, not only at the encoder exit.
  ///
  /// # Errors
  /// [`Error::ShapePairMismatch`] if `enc.shape() != [1, n_audio_ctx,
  /// n_audio_state]`.
  fn validate_encoder_states(&self, enc: &Array) -> Result<()> {
    let expected = [1usize, self.dims.n_audio_ctx(), self.dims.n_audio_state()];
    let actual = enc.shape();
    if actual.len() != expected.len() || actual.iter().zip(expected.iter()).any(|(a, e)| a != e) {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "WhisperModel: encoder states must be (1, n_audio_ctx, n_audio_state) — a single Whisper segment",
        expected.to_vec(),
        actual,
      )));
    }
    Ok(())
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
  /// # Errors
  /// - [`Error::ShapePairMismatch`] if `encoder_states` is not `(1, n_audio_ctx,
  ///   n_audio_state)`;
  /// - propagates the decoder forward op errors (embedding / block / LayerNorm /
  ///   positional-slice).
  pub(crate) fn decode_tokens(
    &self,
    tokens: &Array,
    encoder_states: &Array,
    cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache)> {
    // Bound the encoder extent BEFORE the decoder cross-attention projects it.
    self.validate_encoder_states(encoder_states)?;
    let (logits, new_cache) = self.decoder.forward(tokens, encoder_states, cache)?;
    let logits = logits.astype(Dtype::F32)?;
    Ok((logits, new_cache))
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
      if let Some(stripped) = key.strip_prefix("model.") {
        key = stripped.to_string();
      }

      // Apply the key remap; a `None` target drops the key entirely.
      let mut skip = false;
      for (old, new) in KEY_MAP {
        if key.contains(old) {
          match new {
            None => {
              skip = true;
              break;
            }
            Some(replacement) => key = key.replace(old, replacement),
          }
        }
      }
      if skip {
        continue;
      }
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
struct Builder {
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
}

impl Builder {
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

  /// Build a [`Linear`] from `<prefix>.weight` `(out, in)` (+ optional
  /// `<prefix>.bias` `(out,)`). `bias = false` for the Whisper attention `key`
  /// projection. Both tensors are shape-validated against the config-derived
  /// `(out, in)` before the [`Linear`] is built.
  fn linear(&mut self, prefix: &str, out: i32, in_features: i32, bias: bool) -> Result<Linear> {
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

    let token_embedding = Embedding::new(self.take_shaped(
      "decoder.token_embedding.weight",
      "decoder token embedding (n_vocab, n_text_state)",
      &[n_vocab, n_state],
    )?);
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
    // Bound the encoder-states extent FIRST — before the token array or the
    // decoder allocates. A caller can supply any `enc`; the cross-attention
    // forms its scores from `enc.shape()[1]`, so a longer / batched segment must
    // be rejected before it can size the cross-attention buffers past the config
    // caps (which assume a single `[1, n_audio_ctx, n_audio_state]` segment).
    self.validate_encoder_states(enc)?;

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
    let t = i32::try_from(new_tokens.len()).map_err(|_| {
      Error::OutOfRange(crate::error::OutOfRangePayload::new(
        "WhisperModel::decode_step: new-token window length",
        "must fit in i32",
        smol_str::format_smolstr!("{}", new_tokens.len()),
      ))
    })?;
    let tok = Array::from_slice::<u32>(new_tokens, &[1, t])?;

    let (logits, new_cache) = self.decode_tokens(&tok, enc, cache.inner.as_ref())?;
    cache.inner = Some(new_cache);

    // The decoder returns `[B=1, T, V]`; slice the LAST position and reshape to
    // the rank-1 `[V]` row the greedy driver reads `argmax` over.
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
    crate::ops::indexing::slice(&logits, &[0, last, 0], &[1, last + 1, vi], &[1, 1, 1])?
      .reshape(&[vi])
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

    // Map the golden options onto the decoding task's options: greedy
    // (`temperature == 0`) keeps the default fallback schedule; an explicit
    // positive temperature pins a single-temperature decode. The generated-
    // token cap maps onto `sample_len` (clamped by the decoder context inside
    // the task).
    let decode = DecodingOptions {
      task: task_to_whisper(opts.task()),
      language: opts.language().map(str::to_owned),
      temperature: opts.temperature(),
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
    let whisper_opts = WhisperTranscribeOptions {
      decode,
      temperatures,
      compression_ratio_threshold: Some(decoding::DEFAULT_COMPRESSION_RATIO_THRESHOLD),
      logprob_threshold: Some(decoding::DEFAULT_LOGPROB_THRESHOLD),
      no_speech_threshold: Some(decoding::DEFAULT_NO_SPEECH_THRESHOLD),
    };

    let result = decoding::transcribe(self, &wrapper, &mel, content_frames, &whisper_opts)?;

    // Convert the Whisper transcribe result into the universal `Transcription`.
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

/// Canonical `<|endoftext|>` id (the multilingual + English-only vocabularies
/// agree on this offset).
const EOT_TOKEN_ID: u32 = 50257;

#[cfg(test)]
mod tests;
