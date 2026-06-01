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
//!   pattern). A model implements it once per modality it supports — a
//!   dual-tower model implements [`Embed<TextInput>`] **and** `Embed<ImageInput>`
//!   (two distinct traits, no conflict), each with its own [`Output`](Embed::Output).
//! - [`TextEmbedder`] — the object-safe projection of the one *universal* input
//!   modality (token ids). `dyn TextEmbedder` is the handle the generic text
//!   pipeline and the load factory dispatch on. Image inputs are model-defined,
//!   so there is no universal `dyn ImageEmbedder`; image embedding stays on the
//!   model's [`Embed<ImageInput>`] and its inherent `encode_image`.
//! - [`Contrastive`] / [`LateInteraction`] — capability traits a model composes
//!   on top (cross-tower similarity; MaxSim scoring).
//! - [`TokenEncoder`] + [`pool_embed`] — the sentence-encoder family: a model
//!   produces per-token hidden states and names a [`PoolingStrategy`]; the
//!   free-function driver turns those into an [`Embedding`]. It is a free
//!   function, **not** a `impl<M: TokenEncoder> Embed<TextInput> for M` blanket,
//!   because such a blanket would coherence-conflict with a dual-tower model's
//!   own `Embed<TextInput>` impl (the same constraint the STT `Transcribe`
//!   drivers resolve with free functions).

use crate::{
  array::Array,
  embeddings::{PoolingStrategy, pool},
  error::{Error, InvariantViolationPayload, Result},
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

/// The one **universal** input modality: a token-id batch (and an optional
/// padding mask). Token ids are the same shape across every text model, so this
/// is the modality [`TextEmbedder`] erases to.
///
/// - `token_ids` — `(batch, seq_len)` `I32` ids, padded to the batch max by the
///   caller (the generic `encode` pipeline builds it).
/// - `attention_mask` — `(batch, seq_len)`, `1` for real tokens and `0` for
///   padding; `None` for a model that does not mask (SigLIP's sticky-EOS text
///   pooling reads the last position regardless). Sentence-encoder pooling
///   ([`pool_embed`]) requires it.
#[derive(Clone, Copy)]
pub struct TextInput<'a> {
  token_ids: &'a Array,
  attention_mask: Option<&'a Array>,
}

impl<'a> TextInput<'a> {
  /// A text input from a token-id batch and an optional padding mask.
  #[inline(always)]
  pub fn new(token_ids: &'a Array, attention_mask: Option<&'a Array>) -> Self {
    Self {
      token_ids,
      attention_mask,
    }
  }

  /// The `(batch, seq_len)` `I32` token ids.
  #[inline(always)]
  pub fn token_ids(&self) -> &'a Array {
    self.token_ids
  }

  /// The optional `(batch, seq_len)` padding mask.
  #[inline(always)]
  pub fn attention_mask(&self) -> Option<&'a Array> {
    self.attention_mask
  }
}

/// The universal embedding contract: map a model-defined `Input` to its
/// associated [`Output`](Self::Output).
///
/// Generic over the input modality with the output associated to `(Self, Input)`
/// — the `core::ops::Add<Rhs>{type Output}` shape. A model implements it once
/// per modality it supports; a dual-tower model implements both
/// `Embed<TextInput>` and `Embed<ImageInput>` (distinct traits, no overlap),
/// each fixing its own `Output` (here always [`Embedding`]; a multi-vector model
/// fixes [`MultiVector`]).
pub trait Embed<Input> {
  /// What this `(model, input-modality)` pair produces — [`Embedding`] for a
  /// single-vector model, [`MultiVector`] for a late-interaction model.
  type Output;

  /// Embed one input into [`Output`](Self::Output). No implicit eval — the
  /// result composes lazily; the caller materializes via [`Array`] accessors.
  fn embed(&self, input: Input) -> Result<Self::Output>;
}

/// The object-safe handle for the universal text modality: token ids → one
/// [`Embedding`].
///
/// Every text-capable [`Embed`] model is a `TextEmbedder` for free via the
/// blanket projection below, so `dyn TextEmbedder` (what the generic `encode`
/// pipeline and the load factory hold) covers a sentence-encoder and a
/// dual-tower model's text tower alike. This is a *projection* blanket, not a
/// behavior one: nothing implements `TextEmbedder` other than through its
/// `Embed<TextInput>`, so — unlike the STT `Transcribe` blanket — there is no
/// model-supplied override for it to conflict with.
pub trait TextEmbedder {
  /// Embed a `(batch, seq_len)` token-id batch (with an optional padding mask)
  /// into a `(batch, dim)` [`Embedding`].
  fn embed_text(&self, token_ids: &Array, attention_mask: Option<&Array>) -> Result<Embedding>;
}

impl<M> TextEmbedder for M
where
  M: for<'a> Embed<TextInput<'a>, Output = Embedding>,
{
  #[inline]
  fn embed_text(&self, token_ids: &Array, attention_mask: Option<&Array>) -> Result<Embedding> {
    self.embed(TextInput::new(token_ids, attention_mask))
  }
}

/// Contrastive dual-tower capability: compare an already-computed text and image
/// [`Embedding`] (SigLIP `logits_per_text`).
///
/// Object-safe — it operates on the [`Embedding`]s the model's [`Embed`] impls
/// already produced, so it carries none of the model-specific image input.
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
/// and names the [`PoolingStrategy`] those states are reduced with.
///
/// A model implements this plus a one-line [`Embed<TextInput>`] that delegates to
/// [`pool_embed`] — it does **not** get `Embed<TextInput>` from a blanket (that
/// would conflict with a dual-tower model's own text impl under coherence).
pub trait TokenEncoder {
  /// Per-token hidden states `(batch, seq_len, hidden)` for a token-id batch and
  /// its padding mask.
  fn token_states(&self, token_ids: &Array, attention_mask: &Array) -> Result<Array>;

  /// The pooling strategy these states are reduced with — resolved at load
  /// (`1_Pooling/config.json`), so the model owns it and `encode` stays thin.
  fn pooling(&self) -> PoolingStrategy;
}

/// Drive a [`TokenEncoder`] to a pooled, L2-normalized [`Embedding`]: run the
/// encoder, reduce by its [`pooling`](TokenEncoder::pooling), L2-normalize.
///
/// The free-function bridge a sentence-encoder's [`Embed<TextInput>`] delegates
/// to. Requires `input.attention_mask()` to be `Some`: mask-aware pooling needs
/// the padding mask, and the generic `encode` pipeline always supplies one (it
/// builds it from the tokenizer). A model that genuinely has no mask is a
/// dual-tower text tower, which embeds via its own `Embed<TextInput>`, not this.
pub fn pool_embed<M: TokenEncoder + ?Sized>(model: &M, input: TextInput<'_>) -> Result<Embedding> {
  let mask = input.attention_mask().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "pool_embed: attention_mask",
      "must be Some — mask-aware sentence-encoder pooling requires the padding mask \
       (the encode pipeline builds it from the tokenizer)",
    ))
  })?;
  let states = model.token_states(input.token_ids(), mask)?;
  // `pool` with `normalize = true` applies the L2-normalize tail; no matryoshka
  // dimension truncation and no fused layer/rms-norm here (those are encode-level
  // config, applied by the caller when a pooling config requests them).
  let pooled = pool(&states, mask, model.pooling(), true, None, false, false)?;
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
