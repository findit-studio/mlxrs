//! The golden embedding trait architecture for `mlxrs::embeddings`.
//!
//! A single lowest-common-denominator `forward(ids, mask) -> {hidden, pooled}`
//! trait fits only BERT-style sentence-encoders; contrastive dual-tower models
//! (SigLIP2) and multi-vector / late-interaction models (ColVision) do not. This
//! module replaces it with a generic core plus object-safe handles and
//! composable capability traits, so each shape implements exactly the contract
//! it has:
//!
//! - [`Embed<Input>`] — the universal core, generic over the **input** modality
//!   with the **output** associated to `(Self, Input)` (the `Add<Rhs>{Output}`
//!   pattern). A model implements it once per modality it supports. Image inputs
//!   are model-defined (a model owns the preprocessed-image type), so a
//!   dual-tower model reaches its image tower through its own
//!   `Embed<ImageInput>` and inherent `encode_image`.
//! - [`TextEmbedder`] — the object-safe text-family seam, **implemented by the
//!   model**. It owns the two stages a one-size pipeline cannot standardize: how
//!   the model tokenizes + pads a batch ([`text_encoding`](TextEmbedder::text_encoding))
//!   and how it forwards + pools token ids into one [`Embedding`]
//!   ([`embed_text`](TextEmbedder::embed_text)). `dyn TextEmbedder` is the handle
//!   the generic [`encode`](crate::embeddings::encode()) pipeline and the load
//!   factory dispatch on, so a sentence-encoder and a dual-tower text tower are
//!   both reached object-safely — each routing its own padding scheme and pooling.
//! - [`Contrastive`] / [`LateInteraction`] — capability traits a model composes
//!   on top (cross-tower similarity; MaxSim scoring).
//! - [`TokenEncoder`] + [`pool_embed`] — the sentence-encoder family: a model
//!   produces per-token hidden states and bakes its deployment
//!   [`PoolingConfig`] at load (from `1_Pooling/config.json`); the
//!   free-function driver turns those into an [`Embedding`] by applying the full
//!   config (strategy + normalize + matryoshka dimension + layer/rms-norm) via
//!   the [`pool`] dispatcher. A sentence-encoder's
//!   [`embed_text`](TextEmbedder::embed_text) delegates to it. It is a free
//!   function, **not** an `impl<M: TokenEncoder> TextEmbedder for M` blanket, so
//!   it never coherence-conflicts with a dual-tower model's own text impl (the
//!   same constraint the STT `Transcribe` drivers resolve with free functions).

use crate::{
  array::Array,
  embeddings::{PoolingStrategy, pool},
  error::Result,
};

/// A single embedding vector, `(batch, dim)` — conventionally L2-normalized.
///
/// The output of a [`TextEmbedder`] / an image tower / a sentence-encoder. A
/// newtype over [`Array`] so an embedding cannot be confused with an arbitrary
/// tensor (and so the L2-normalized contract has a name to hang on). No implicit
/// eval: the inner array is a lazy graph node.
#[derive(Debug)]
pub struct Embedding(Array);

impl Embedding {
  /// Wrap an array as an [`Embedding`] (the caller asserts it is the final,
  /// conventionally L2-normalized embedding vector).
  #[inline(always)]
  pub fn new(array: Array) -> Self {
    Self(array)
  }

  /// The embedding tensor `(batch, dim)`.
  #[inline(always)]
  pub fn array(&self) -> &Array {
    &self.0
  }

  /// Consume into the inner array.
  #[inline(always)]
  pub fn into_array(self) -> Array {
    self.0
  }
}

/// Multiple per-token vectors, `(batch, n_tokens, dim)` — the output of a
/// multi-vector / late-interaction model (ColBERT / ColVision), scored by
/// [`LateInteraction::score`] (MaxSim) rather than reduced to one vector.
#[derive(Debug)]
pub struct MultiVector(Array);

impl MultiVector {
  /// Wrap an array as a [`MultiVector`].
  #[inline(always)]
  pub fn new(array: Array) -> Self {
    Self(array)
  }

  /// The multi-vector tensor `(batch, n_tokens, dim)`.
  #[inline(always)]
  pub fn array(&self) -> &Array {
    &self.0
  }

  /// Consume into the inner array.
  #[inline(always)]
  pub fn into_array(self) -> Array {
    self.0
  }
}

/// How a model pads a tokenized batch — the half of [`TextEncoding`] a generic
/// `encode` pipeline cannot standardize across architectures.
///
/// A sentence-encoder pads to the batch maximum and masks the pad cells
/// ([`DynamicRightPad`](Self::DynamicRightPad)); a fixed-length contrastive text
/// tower (SigLIP's sticky-EOS pooling) pads/truncates every row to its own fixed
/// sequence length with an all-`1` mask ([`FixedLength`](Self::FixedLength)).
/// The model names which one it needs; the pipeline applies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
  /// Right-pad every id row to the **batch maximum** length with `pad_token_id`,
  /// masking `0` over the pad cells (and `1` over real tokens). The
  /// sentence-encoder default — mask-aware pooling reads only the real tokens,
  /// so the per-row length and the pad id are immaterial to the result. Matches
  /// the HF tokenizer's `padding_side="right"` for encoders.
  ///
  /// This scheme has no intrinsic token cap, so its effective tokenizer
  /// truncation cap is exactly [`TextEncoding::max_length`] — `None` tokenizes
  /// each text at full length. The caller's input is tokenized unmodified;
  /// bounding an oversized / untrusted prompt is the consuming application's
  /// responsibility.
  DynamicRightPad {
    /// Token id written into pad cells. The mask is `0` there, so the value
    /// never reaches the embedding; it exists only so the `(batch, seq)` id
    /// tensor is well-formed (HF/swift pad with `0`).
    pad_token_id: u32,
  },
  /// Pad **or truncate** every id row to a model-fixed `length` with
  /// `pad_token_id`, and build an **all-`1`** mask of that length. The fixed
  /// processor scheme of a model whose pooling reads an absolute position
  /// regardless of the mask (SigLIP's sticky-EOS text tower pools a fixed last
  /// position, so every row must reach exactly `length` and the pad cells are
  /// *not* masked out — the reference processor right-pads with the pad id and
  /// pools the last position whatever it holds).
  ///
  /// `eos_token_id` carries the EOS-preserving truncation contract of a
  /// sticky-EOS pooler. The HF processor adds the EOS *after* a head-truncation
  /// (`max_length - n_added_tokens`, then the template appends EOS), so for an
  /// **overlength** prompt the last real position is the EOS, never a content
  /// token. A naive `ids[..length]` head-keep would instead leave a content
  /// token in the pooled last slot — wrong for a sticky-EOS tower and not
  /// byte-identical to the reference. When `eos_token_id` is `Some(eos)` and a
  /// row exceeds `length`, the head is kept to `length - 1` and `eos` is forced
  /// into the final position (mirroring HF's truncate-then-append-EOS). A
  /// shorter row is unaffected (the tokenizer's own special-token pass already
  /// placed the trailing EOS, and the pad cells fill the rest). `None` keeps the
  /// plain head-truncation (a fixed-position pooler without a sticky-EOS
  /// invariant).
  ///
  /// **Sticky-EOS guarantee.** For `eos_token_id: Some(_)`, a *truncated*
  /// prompt's final pooled slot is **always** the EOS, *regardless of any
  /// explicit* [`max_length`](TextEncoding::max_length). The `encode` pipeline
  /// derives the tokenizer truncation cap with a floor of `length + 1` for a
  /// sticky-EOS scheme, so an explicit `max_length` (even `Some(length)`) can
  /// never reduce the cap to `length` and thereby skip the EOS forcing; an
  /// explicit cap is only ever *additive* for a sticky-EOS model and cannot leave
  /// a content token in the pooled last position.
  FixedLength {
    /// The fixed sequence length every row is padded or truncated to. This is
    /// the model's declared sequence length (e.g. SigLIP2's fixed `64`); the
    /// `length == 0` edge yields the empty `(batch, 0)` batch. The `encode`
    /// pipeline allocates the `(batch, length)` buffers fallibly, so an absurd
    /// `length` from a malformed config yields a typed
    /// [`Error::AllocFailure`](crate::Error::AllocFailure) /
    /// [`Error::ArithmeticOverflow`](crate::Error::ArithmeticOverflow) rather than
    /// a panic.
    length: usize,
    /// Token id written into pad cells (here a real, unmasked position the
    /// model's fixed-position pooling may read).
    pad_token_id: u32,
    /// The EOS id forced into the **final** position when an overlength row is
    /// truncated, so a sticky-EOS pooler's last slot is the EOS (not a content
    /// token). `None` disables EOS preservation (plain head-truncation). See the
    /// variant docs for the HF truncate-then-append-EOS parity rationale.
    eos_token_id: Option<u32>,
  },
}

/// How a model tokenizes and pads a batch of texts — the deployment input
/// contract a model bakes and the generic [`encode`](crate::embeddings::encode())
/// pipeline reads back.
///
/// `encode` is thin: it reads this off the model, tokenizes + pads per it, and
/// hands the `(batch, seq)` ids + mask to [`embed_text`](TextEmbedder::embed_text).
/// The tokenization knobs (special tokens, truncation cap, padding) live here on
/// the model rather than on a pipeline-side config the caller guesses, so the
/// *model* owns its input scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEncoding {
  /// Add the tokenizer's special tokens (BOS/EOS/sep) when encoding, as in
  /// python `processor(..., add_special_tokens=True)` (the transformers default)
  /// and swift `tokenizer.encode(text:, addSpecialTokens: true)`.
  pub add_special_tokens: bool,
  /// Optional **explicit** per-sequence hard token cap (python `truncation=True`,
  /// `max_length=512`): each text is right-truncated (keep the head, drop the
  /// tail) to at most this many ids *before* batch padding. `None` disables this
  /// explicit cap.
  ///
  /// This is not the only truncation source. The [`padding`](Self::padding)
  /// scheme can carry its own intrinsic cap — a [`Padding::FixedLength`] truncates
  /// every row to its fixed `length`, so the generic
  /// [`encode`](crate::embeddings::encode()) pipeline derives the effective
  /// tokenizer truncation cap from the padding mode itself (the fixed `length`,
  /// `+ 1` for a sticky-EOS slot) and combines it with this field by the
  /// **tighter** bound. So a `FixedLength` encoding always truncates *regardless*
  /// of this field, and leaving this `None` there is the norm (the cap comes from
  /// the padding mode, not this optional field). Set it only to impose an
  /// *additional*, tighter explicit cap.
  ///
  /// The caller's input is tokenized UNMODIFIED and head-truncated to the derived
  /// cap; the library imposes no input-size limit, so bounding an oversized /
  /// untrusted prompt is the **consuming application's** responsibility. Every
  /// buffer the pipeline allocates is fallible, so an oversized prompt or
  /// configuration yields a typed
  /// [`Error::AllocFailure`](crate::Error::AllocFailure) / arithmetic-overflow
  /// error rather than a panic.
  ///
  /// **Sticky-EOS preservation.** For a sticky-EOS [`Padding::FixedLength`]
  /// (`eos_token_id: Some(_)`), a truncated prompt's final pooled slot is always
  /// the EOS. An explicit `max_length` set here (even `Some(length)`) cannot
  /// suppress it — the derived cap is floored at `length + 1`. See the
  /// [`Padding::FixedLength`] variant docs.
  pub max_length: Option<usize>,
  /// How the tokenized rows are padded into the `(batch, seq)` tensor + mask.
  pub padding: Padding,
}

impl TextEncoding {
  /// Construct a [`TextEncoding`] from its three parts.
  #[inline]
  pub fn new(add_special_tokens: bool, max_length: Option<usize>, padding: Padding) -> Self {
    Self {
      add_special_tokens,
      max_length,
      padding,
    }
  }
}

/// The deployment pooling configuration a sentence-encoder bakes at load (from
/// `1_Pooling/config.json`) and [`pool_embed`] applies in full.
///
/// This is the resolved superset of [`crate::embeddings::StPoolingConfig`] (the
/// parsed ST config) plus the fused post-pool norms the [`pool`] dispatcher
/// surfaces. A model converts the parsed config into this once at construction
/// and returns it from [`TokenEncoder::pooling`], so the encode path never
/// guesses normalize / dimension / layer-norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolingConfig {
  /// Which pooling reduction to apply (swift `Pooling.Strategy` / python
  /// `pool_by_config` modes).
  pub strategy: PoolingStrategy,
  /// L2-normalize the pooled vector (python `normalize_embeddings`). ST configs
  /// always normalize; surfaced so a model can override.
  pub normalize: bool,
  /// Matryoshka output dimension — truncate the pooled vector's last axis to
  /// this width (swift `Pooling.dimension` / ST `word_embedding_dimension`).
  /// `None` keeps the full width.
  pub dimension: Option<usize>,
  /// Apply a fused `LayerNorm` to the pooled vector before truncation /
  /// normalize (swift `Pooling.applyLayerNorm`). Takes precedence over
  /// [`rms_norm`](Self::rms_norm) if both are set.
  pub layer_norm: bool,
  /// Apply a fused `RMSNorm` to the pooled vector before truncation / normalize
  /// (the mlx-c-surfaced variant). Ignored if [`layer_norm`](Self::layer_norm)
  /// is also set.
  pub rms_norm: bool,
}

impl PoolingConfig {
  /// Construct a [`PoolingConfig`] from its parts.
  #[inline]
  pub fn new(
    strategy: PoolingStrategy,
    normalize: bool,
    dimension: Option<usize>,
    layer_norm: bool,
    rms_norm: bool,
  ) -> Self {
    Self {
      strategy,
      normalize,
      dimension,
      layer_norm,
      rms_norm,
    }
  }
}

/// The universal embedding contract: map a model-defined `Input` to its
/// associated [`Output`](Self::Output).
///
/// Generic over the input modality with the output associated to `(Self, Input)`
/// — the `core::ops::Add<Rhs>{type Output}` shape. A model implements it once
/// per modality it supports. Text is exposed object-safely through
/// [`TextEmbedder`] (not this trait); a model-defined image input flows through
/// `Embed<ImageInput>` (with the model owning the `ImageInput` type), each
/// fixing its own `Output` ([`Embedding`] for a single-vector tower,
/// [`MultiVector`] for a late-interaction model).
pub trait Embed<Input> {
  /// What this `(model, input-modality)` pair produces — [`Embedding`] for a
  /// single-vector model, [`MultiVector`] for a late-interaction model.
  type Output;

  /// Embed one input into [`Output`](Self::Output). No implicit eval — the
  /// result composes lazily; the caller materializes via [`Array`] accessors.
  fn embed(&self, input: Input) -> Result<Self::Output>;
}

/// The object-safe text-family seam, **implemented by the model**.
///
/// A model owns the two text stages a one-size pipeline cannot standardize:
///
/// - [`text_encoding`](Self::text_encoding) — how it tokenizes + pads a batch
///   (the [`TextEncoding`] the generic [`encode`](crate::embeddings::encode())
///   pipeline reads back and applies).
/// - [`embed_text`](Self::embed_text) — how it forwards + pools a padded
///   `(batch, seq)` id batch (and its mask) into one [`Embedding`].
///
/// `dyn TextEmbedder` is the handle the encode pipeline and the load factory
/// dispatch on, so a sentence-encoder (pooling via [`pool_embed`]) and a
/// dual-tower text tower (its own sticky-EOS projection) are both reached
/// object-safely — each routing its own padding scheme and pooling, with **no**
/// model-specific branch in the pipeline.
pub trait TextEmbedder {
  /// How this model tokenizes + pads a batch of texts. The
  /// [`encode`](crate::embeddings::encode()) pipeline reads this and produces
  /// the `(batch, seq)` ids + mask `embed_text` expects.
  fn text_encoding(&self) -> TextEncoding;

  /// Embed a `(batch, seq_len)` token-id batch (and its `(batch, seq_len)`
  /// attention mask, `1` real / `0` pad) into a `(batch, dim)` [`Embedding`].
  /// The mask matches the scheme [`text_encoding`](Self::text_encoding) declared
  /// (an all-`1` mask for [`Padding::FixedLength`]); a model whose pooling reads
  /// an absolute position may ignore it.
  fn embed_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Embedding>;
}

/// Contrastive dual-tower capability: compare an already-computed text and image
/// [`Embedding`] (SigLIP `logits_per_text`).
///
/// Object-safe — it operates on the [`Embedding`]s the model's text/image
/// embedding already produced, so it carries none of the model-specific image
/// input.
pub trait Contrastive {
  /// Similarity logits between L2-normalized text and image embeddings. SigLIP:
  /// `(text @ image.T) * exp(logit_scale) + logit_bias`.
  fn similarity(&self, text: &Embedding, image: &Embedding) -> Result<Array>;
}

/// Late-interaction (MaxSim / ColBERT) capability: score multi-vector queries
/// against multi-vector passages.
pub trait LateInteraction {
  /// The `(n_queries, n_passages)` MaxSim score matrix. `batch_size` tiles the
  /// quadratic interaction (the ColVision `score_multi_vector` knob).
  fn score(
    &self,
    queries: &[MultiVector],
    passages: &[MultiVector],
    batch_size: usize,
  ) -> Result<Array>;
}

/// The sentence-encoder family: a model that produces per-token hidden states
/// and bakes the [`PoolingConfig`] those states are reduced with.
///
/// A model implements this plus a one-line [`TextEmbedder`] whose
/// [`embed_text`](TextEmbedder::embed_text) delegates to [`pool_embed`] — it does
/// **not** get `TextEmbedder` from a blanket (that would conflict with a
/// dual-tower model's own text impl under coherence).
pub trait TokenEncoder {
  /// Per-token hidden states `(batch, seq_len, hidden)` for a token-id batch and
  /// its padding mask.
  fn token_states(&self, token_ids: &Array, attention_mask: &Array) -> Result<Array>;

  /// The full pooling configuration these states are reduced with — baked at
  /// load (`1_Pooling/config.json`), so the model owns strategy + normalize +
  /// matryoshka dimension + layer/rms-norm and [`pool_embed`] applies all of it.
  fn pooling(&self) -> PoolingConfig;
}

/// Drive a [`TokenEncoder`] to a pooled [`Embedding`]: run the encoder, then
/// apply its full baked [`PoolingConfig`](TokenEncoder::pooling) (strategy +
/// fused layer/rms-norm + matryoshka dimension + L2-normalize) via the [`pool`]
/// dispatcher.
///
/// The free-function bridge a sentence-encoder's
/// [`embed_text`](TextEmbedder::embed_text) delegates to. The mask is required:
/// mask-aware pooling needs the padding mask, and the generic
/// [`encode`](crate::embeddings::encode()) pipeline always supplies one (built
/// from the tokenizer per the model's [`TextEncoding`]).
pub fn pool_embed<M: TokenEncoder + ?Sized>(
  model: &M,
  token_ids: &Array,
  attention_mask: &Array,
) -> Result<Embedding> {
  let states = model.token_states(token_ids, attention_mask)?;
  let cfg = model.pooling();
  let pooled = pool(
    &states,
    attention_mask,
    cfg.strategy,
    cfg.normalize,
    cfg.dimension,
    cfg.layer_norm,
    cfg.rms_norm,
  )?;
  Ok(Embedding::new(pooled))
}

/// The object-safe handle the load factory hands back for **any** embedding
/// model, regardless of its capability set.
///
/// A model declares the capabilities it has by overriding the relevant accessor
/// (each defaults to `None`); a consumer queries them. This replaces a
/// lowest-common-denominator `forward` contract — a text-only encoder, a
/// dual-tower contrastive model, and a multi-vector model are all
/// `EmbeddingModel`s, each answering only the accessors it supports. It mirrors
/// mlx-swift's single `EmbeddingModel` protocol existential, but is
/// capability-typed rather than forcing one forward shape.
///
/// Universal text embedding is exposed object-safely
/// ([`as_text_embedder`](Self::as_text_embedder)); a model-specific input (an
/// image tower whose input type the model defines) is reached by concrete
/// downcast through [`as_any`](Self::as_any).
pub trait EmbeddingModel: std::any::Any {
  /// This model's universal text embedding, if it embeds text. A SigLIP text
  /// tower and a sentence-encoder both answer `Some`; an image-only model
  /// answers `None`.
  fn as_text_embedder(&self) -> Option<&dyn TextEmbedder> {
    None
  }

  /// This model's contrastive dual-tower capability, if any (SigLIP).
  fn as_contrastive(&self) -> Option<&dyn Contrastive> {
    None
  }

  /// This model's late-interaction / multi-vector capability, if any.
  fn as_late_interaction(&self) -> Option<&dyn LateInteraction> {
    None
  }

  /// Upcast to [`Any`](std::any::Any) for a concrete downcast — the escape hatch
  /// for a model-specific API whose input type cannot be erased into an
  /// object-safe modality handle (e.g. an image tower taking a model-defined
  /// preprocessed-image type). A consumer that loaded a known architecture
  /// downcasts to it via `model.as_any().downcast_ref::<ConcreteModel>()`.
  fn as_any(&self) -> &dyn std::any::Any;
}
