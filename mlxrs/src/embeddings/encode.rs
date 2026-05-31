//! The `encode` entry — tokenize a batch of texts, pad to the batch's max
//! length, run an [`EmbeddingModel`], pool, and optionally L2-normalize into a
//! `(batch, dim)` embedding matrix.
//!
//! Ports the orchestration of:
//! - python `mlx-embeddings` `utils.py::generate` (tokenize via the processor
//!   with `padding` / `truncation` / `max_length`, run the model, return the
//!   embeddings) cross-referenced with `models/pooling.py::pool_by_config` and
//!   `models/base.py::normalize_embeddings`;
//! - swift `MLXEmbedders` `EmbedderModelContainer.perform` (encode each text →
//!   pad to the batch max → build the mask → `model(padded, …, attentionMask:
//!   mask)` → `pooling(output, normalize: …)` → `eval`).
//!
//! Unlike python, where the per-architecture model returns an already pooled +
//! normalized `text_embeds`, mlxrs pools *externally* with the existing
//! [`pool`] dispatcher (the no-model-arch rule keeps per-model heads out of
//! scope), exactly as swift's container does. Tokenization is local-only via
//! the existing [`Tokenizer`]; pooling and normalization reuse
//! [`crate::embeddings::pool`] — nothing here re-implements them.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload, try_with_capacity,
  },
  tokenizer::{EncodeOptions, Tokenizer},
};

use super::{PoolingStrategy, model::EmbeddingModel, pool, pool_post};

/// Configuration for [`encode`].
///
/// Defaults mirror python `generate` (`max_length = 512`, padding +
/// truncation on, special tokens added) composed with swift's
/// `pooling(output, normalize: true)`: [`mean`](PoolingStrategy::Mean)
/// pooling, L2-normalized output.
///
/// Build via [`EncodeConfig::new`] and chain `with_*` setters:
///
/// ```rust,ignore
/// let cfg = EncodeConfig::new()
///   .with_strategy(PoolingStrategy::Cls)
///   .with_normalize(false);
/// ```
#[derive(Debug, Clone)]
pub struct EncodeConfig {
  /// Pooling strategy applied to the model's `(batch, seq_len, hidden)`
  /// hidden states (the existing [`PoolingStrategy`] / [`pool`] dispatcher).
  /// Default [`PoolingStrategy::Mean`] (python `generate`'s `text_embeds` is
  /// "mean pooled and normalized"; swift container default).
  strategy: PoolingStrategy,
  /// L2-normalize the pooled vectors (python `normalize_embeddings`, swift
  /// `pooling(_, normalize: true)`). Default `true`.
  normalize: bool,
  /// Add the tokenizer's special tokens (BOS/EOS/sep) when encoding, as in
  /// python `processor(..., add_special_tokens=True)` (the transformers
  /// default) and swift `tokenizer.encode(text:, addSpecialTokens: true)`.
  /// Default `true`.
  add_special_tokens: bool,
  /// Per-sequence hard token cap (python `truncation=True`,
  /// `max_length=512`): each text is right-truncated (keep the head, drop the
  /// tail) to at most this many ids *before* batch padding. `None` disables
  /// truncation. Default `Some(512)`.
  max_length: Option<usize>,
  /// Token id written into padding positions. The attention mask is `0`
  /// there, so this value never reaches the pooled output — it exists only so
  /// the padded `(batch, seq_len)` id tensor is well-formed (swift pads with
  /// `0`). Default `0`.
  pad_token_id: u32,
  /// Optional matryoshka last-dim truncation forwarded to [`pool`] (swift
  /// `Pooling.dimension`). `None` keeps the model's full hidden width.
  /// Default `None`.
  dimension: Option<usize>,
  /// Apply a fused LayerNorm to the pooled vector before truncation /
  /// normalization (swift `applyLayerNorm:`), forwarded to [`pool`]. Default
  /// `false`.
  apply_layer_norm: bool,
  /// Apply a fused RMSNorm to the pooled vector (mlx-c-surfaced variant;
  /// ignored if `apply_layer_norm` is also set), forwarded to [`pool`].
  /// Default `false`.
  apply_rms_norm: bool,
}

impl Default for EncodeConfig {
  fn default() -> Self {
    Self {
      strategy: PoolingStrategy::Mean,
      normalize: true,
      add_special_tokens: true,
      max_length: Some(512),
      pad_token_id: 0,
      dimension: None,
      apply_layer_norm: false,
      apply_rms_norm: false,
    }
  }
}

impl EncodeConfig {
  /// Fluent builder constructor; equivalent to [`EncodeConfig::default`].
  pub fn new() -> Self {
    Self::default()
  }

  // ── builders ──────────────────────────────────────────────────────────

  /// Set the pooling [`strategy`](Self::strategy).
  #[must_use]
  pub fn with_strategy(mut self, v: PoolingStrategy) -> Self {
    self.strategy = v;
    self
  }
  /// Set the [`normalize`](Self::normalize) flag.
  #[must_use]
  pub fn with_normalize(mut self, v: bool) -> Self {
    self.normalize = v;
    self
  }
  /// Set the [`add_special_tokens`](Self::add_special_tokens) flag.
  #[must_use]
  pub fn with_add_special_tokens(mut self, v: bool) -> Self {
    self.add_special_tokens = v;
    self
  }
  /// Set the per-sequence [`max_length`](Self::max_length) cap.
  #[must_use]
  pub fn with_max_length(mut self, v: Option<usize>) -> Self {
    self.max_length = v;
    self
  }
  /// Set the [`pad_token_id`](Self::pad_token_id).
  #[must_use]
  pub fn with_pad_token_id(mut self, v: u32) -> Self {
    self.pad_token_id = v;
    self
  }
  /// Set the matryoshka [`dimension`](Self::dimension) truncation.
  #[must_use]
  pub fn with_dimension(mut self, v: Option<usize>) -> Self {
    self.dimension = v;
    self
  }
  /// Set the [`apply_layer_norm`](Self::apply_layer_norm) flag.
  #[must_use]
  pub fn with_apply_layer_norm(mut self, v: bool) -> Self {
    self.apply_layer_norm = v;
    self
  }
  /// Set the [`apply_rms_norm`](Self::apply_rms_norm) flag.
  #[must_use]
  pub fn with_apply_rms_norm(mut self, v: bool) -> Self {
    self.apply_rms_norm = v;
    self
  }

  // ── accessors ─────────────────────────────────────────────────────────

  /// The pooling strategy.
  #[inline(always)]
  pub fn strategy(&self) -> PoolingStrategy {
    self.strategy
  }
  /// Whether the pooled vectors are L2-normalized.
  #[inline(always)]
  pub fn normalize(&self) -> bool {
    self.normalize
  }
  /// Whether special tokens are added when encoding.
  #[inline(always)]
  pub fn add_special_tokens(&self) -> bool {
    self.add_special_tokens
  }
  /// Per-sequence token cap (truncation limit). `None` means no truncation.
  #[inline(always)]
  pub fn max_length(&self) -> Option<usize> {
    self.max_length
  }
  /// Token id written into padding positions.
  #[inline(always)]
  pub fn pad_token_id(&self) -> u32 {
    self.pad_token_id
  }
  /// Matryoshka output dimension cap. `None` keeps the model's full width.
  #[inline(always)]
  pub fn dimension(&self) -> Option<usize> {
    self.dimension
  }
  /// Whether a fused LayerNorm is applied to the pooled vector.
  #[inline(always)]
  pub fn apply_layer_norm(&self) -> bool {
    self.apply_layer_norm
  }
  /// Whether a fused RMSNorm is applied to the pooled vector.
  #[inline(always)]
  pub fn apply_rms_norm(&self) -> bool {
    self.apply_rms_norm
  }
}

/// Tokenize `texts`, right-pad each id row to the batch's max length with
/// `pad_token_id`, and build the matching `(batch, seq_len)` attention mask
/// (`1` for real tokens, `0` for padding).
///
/// Returns `(input_ids, attention_mask, seq_len)`:
/// - `input_ids` — `(batch, seq_len)` `i32` array (right-padded). `I32` is
///   MLX's default index dtype for the embedding `take` / gather an
///   [`EmbeddingModel`] performs (matching `lm/generate.rs::token_window`),
///   so the lookup never has to cast. Each `u32` id is converted with a
///   CHECKED `i32::try_from` (a token id `> i32::MAX` — realistically never —
///   yields a recoverable [`Error::OutOfRange`] rather than silently
///   wrapping negative);
/// - `attention_mask` — `(batch, seq_len)` `f32` array (`1.0` / `0.0`);
/// - `seq_len` — the batch max length (after per-text truncation).
///
/// `seq_len` is the longest *truncated* row, so it never exceeds
/// `max_length`. An empty `texts` slice, or a batch whose every row is empty
/// (e.g. `max_length = Some(0)`), produces `seq_len = 0` and correspondingly
/// shaped `(batch, 0)` arrays (an all-padding batch — the mask is all-`0`,
/// which the mean / max poolers floor / guard).
///
/// **Tokenizer-applied padding is not treated as real tokens.** Each text is
/// encoded via [`Tokenizer::encode_with`] with
/// [`return_attention_mask`](EncodeOptions::return_attention_mask), which
/// strips any HF padding cells (e.g. when the loaded `tokenizer.json` has
/// padding enabled) and returns only the *attended* ids with an all-`1` mask.
/// The per-text `(ids, mask)` therefore describe real tokens only; the manual
/// batch padding below is the **sole** source of `0` mask cells. This makes
/// the result correct whether the tokenizer has padding enabled or disabled —
/// without it, HF pad ids would be marked `1.0` and pollute mask-aware
/// pooling, yielding batch-dependent embeddings.
///
/// Right-padding (and the resulting trailing-`0` mask) matches the HF
/// tokenizer's default `padding_side="right"` for encoders and swift's
/// container, so the existing mask-aware poolers behave as in the references.
fn tokenize_and_pad(
  tokenizer: &Tokenizer,
  texts: &[&str],
  add_special_tokens: bool,
  max_length: Option<usize>,
  pad_token_id: u32,
) -> Result<(Array, Array, usize)> {
  let batch = texts.len();

  // Encode each text via the pad-stripping path: `encode_with` drops every
  // HF `mask == 0` cell (any tokenizer-applied padding) and returns the
  // attended ids plus a synthesized all-`1` mask of equal length, applying
  // the per-sequence right-truncation cap (`truncate_to`). The result is
  // real tokens only — independent of the tokenizer's padding setting.
  let opts = EncodeOptions::new()
    .with_add_special(add_special_tokens)
    .with_truncate_to(max_length)
    .with_return_attention_mask(true);
  let mut rows: Vec<(Vec<u32>, Vec<u8>)> = try_with_capacity(batch)?;
  for &text in texts {
    let enc = tokenizer.encode_with(text, &opts)?;
    // Validate the length-equality invariant on the borrowed slices BEFORE
    // moving — `with_return_attention_mask(true)` guarantees `mask.len()
    // == ids.len()` including the legitimate `(0, 0)` case. After the
    // check passes, `into_parts()` moves both `Vec`s into the row without
    // cloning (avoids the O(tokens) per-row copy that borrowed-accessor +
    // owned-flatten would otherwise pay).
    if enc.attention_mask().len() != enc.ids().len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "encode: encode_with(return_attention_mask=true) mask.len() must match ids.len()",
        enc.ids().len(),
        enc.attention_mask().len(),
      )));
    }
    rows.push(enc.into_parts());
  }

  let seq_len = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);

  // Flatten into right-padded (batch, seq_len) id + mask buffers. Each row's
  // own mask is all-`1` (pad-stripped above); the only `0` cells come from
  // the manual padding appended here to reach the batch max length. Ids are
  // emitted as `i32` (MLX's index dtype) via a CHECKED `u32 -> i32`
  // conversion so a token id `> i32::MAX` is a recoverable error, not a
  // silent wrap to a negative index.
  let total = batch.checked_mul(seq_len).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "encode: batch * seq_len",
      "usize",
      [("batch", batch as u64), ("seq_len", seq_len as u64)],
    ))
  })?;
  // The padding id is written into every padded cell, so range-check it once
  // up front rather than per cell.
  let pad_id = i32::try_from(pad_token_id).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "encode: pad_token_id",
      "must fit in i32 (the MLX index dtype)",
      smol_str::format_smolstr!("{pad_token_id}"),
    ))
  })?;
  let mut id_data: Vec<i32> = try_with_capacity(total)?;
  let mut mask_data: Vec<f32> = try_with_capacity(total)?;
  for (ids, mask) in &rows {
    let real = ids.len();
    for &id in ids {
      let id = i32::try_from(id).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "encode: token id",
          "must fit in i32 (the MLX index dtype)",
          smol_str::format_smolstr!("{id}"),
        ))
      })?;
      id_data.push(id);
    }
    mask_data.extend(mask.iter().map(|&m| f32::from(m)));
    let pad = seq_len - real;
    id_data.extend(std::iter::repeat_n(pad_id, pad));
    mask_data.extend(std::iter::repeat_n(0.0_f32, pad));
  }

  let input_ids = Array::from_slice::<i32>(&id_data, &(batch, seq_len))?;
  let attention_mask = Array::from_slice::<f32>(&mask_data, &(batch, seq_len))?;
  Ok((input_ids, attention_mask, seq_len))
}

/// Encode a batch of texts into a `(batch, dim)` embedding matrix.
///
/// Pipeline (python `generate` ∘ swift `EmbedderModelContainer.perform`):
/// 1. tokenize each text (special tokens per `cfg.add_special_tokens`),
///    right-truncate to `cfg.max_length`;
/// 2. right-pad every id row to the batch's max length and build the matching
///    `(batch, seq_len)` attention mask (`1` real, `0` pad);
/// 3. run `model.forward(input_ids, attention_mask)` → hidden states (and an
///    optional model-provided `pooled_output`);
/// 4. pool with `cfg.strategy` and apply `cfg.{apply_layer_norm,
///    apply_rms_norm, dimension, normalize}` via the existing [`pool`]
///    dispatcher. For [`PoolingStrategy::Cls`] / [`PoolingStrategy::None`],
///    if the model returned a `pooled_output` (a trained BERT-style pooler
///    head) it is used directly — the configured normalize / dimension /
///    layer-norm tail still applies via [`pool_post`] — matching swift's
///    `inputs.pooledOutput ?? hiddenStates…`; otherwise the hidden-states
///    pooling path is taken as before.
///
/// The returned array is usually `(batch, dim)`. If `cfg.strategy` is
/// [`PoolingStrategy::None`] and the model does not provide a
/// `pooled_output`, the hidden states are passed through and the result is
/// `(batch, seq_len, dim)` instead; if a `pooled_output` is present, that
/// fast-path still returns a rank-2 `(batch, dim)` array. **No implicit eval**:
/// the result is a lazy graph node; the caller evaluates (or reads it) when
/// ready.
///
/// An empty `texts` slice returns a `(0, …)` array (zero-row batch). The
/// pooling stage receives the model's hidden states unchanged from the
/// reference behavior — mask-aware poolers exclude the padded tail.
///
/// - `model` — any [`EmbeddingModel`] (trait object: one call site, many
///   architectures);
/// - `tokenizer` — the loaded [`Tokenizer`] (local-only; no network);
/// - `texts` — the batch to encode;
/// - `cfg` — pooling / normalization / tokenization knobs ([`EncodeConfig`]).
pub fn encode(
  model: &dyn EmbeddingModel,
  tokenizer: &Tokenizer,
  texts: &[&str],
  cfg: &EncodeConfig,
) -> Result<Array> {
  // Fail fast on a cleared/poisoned worker thread (and install the mlx-c error
  // handler) before any work, since `model.forward` + the pooling ops touch
  // per-thread stream/TLS state. Mirrors the crate's other safe entry points
  // (e.g. `stream::default_stream`), which install the handler before asserting.
  crate::error::ensure_handler_installed();
  crate::stream::assert_streams_not_cleared();
  let (input_ids, attention_mask, _seq_len) = tokenize_and_pad(
    tokenizer,
    texts,
    cfg.add_special_tokens,
    cfg.max_length,
    cfg.pad_token_id,
  )?;

  let output = model.forward(&input_ids, &attention_mask)?;

  // swift `Pooling.callAsFunction`: the `.cls` and `.none` strategies use the
  // model's own `pooledOutput` when present (a trained BERT-style pooler head)
  // — `inputs.pooledOutput ?? hiddenStates…` — falling back to hidden-states
  // pooling only when the model emits none. The other strategies always pool
  // hidden states. Either way the normalize / dimension / layer-norm tail is
  // applied identically (here via `pool_post`, the shared tail of `pool`).
  let (last_hidden_state, pooled_output) = output.into_parts();
  if matches!(cfg.strategy, PoolingStrategy::Cls | PoolingStrategy::None)
    && let Some(pooled) = pooled_output
  {
    // This path bypasses the hidden-state poolers' rank/mask guards
    // (`validate_token_embeddings_*` in `pooling.rs`), so validate the
    // model-provided pooler here with the same panic→`Err` discipline: it
    // must be exactly rank-2 `(batch, hidden)` whose batch covers the
    // request. A squeezed `[hidden]` or a stale `[1, hidden]` pooler for a
    // multi-text batch would otherwise be normalized / truncated and
    // returned as if it covered every text — silent wrong/missing
    // embeddings rather than a recoverable shape error.
    let pooled_shape = pooled.shape();
    if pooled_shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "encode: model pooled_output must be rank-2 (batch, hidden)",
        pooled_shape.len() as u32,
        pooled_shape,
      )));
    }
    if pooled_shape[0] != texts.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "encode: model pooled_output batch must match texts",
        texts.len(),
        pooled_shape[0],
      )));
    }
    // The pooler's hidden width must match the hidden states' (`(batch,
    // seq_len, hidden)`): a pooler emitting a different width than the model's
    // hidden dim would otherwise be normalized / truncated and returned as
    // embeddings of an unexpected dimension. This fast-path bypasses the
    // hidden-state poolers' rank-3 guard, so confirm `last_hidden_state` is
    // rank-3 before indexing its hidden axis (same panic→`Err` discipline).
    let hidden_shape = last_hidden_state.shape();
    if hidden_shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "encode: model last_hidden_state must be rank-3 (batch, seq_len, hidden)",
        hidden_shape.len() as u32,
        hidden_shape,
      )));
    }
    if pooled_shape[1] != hidden_shape[2] {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "encode: model pooled_output hidden width must match last_hidden_state hidden",
        pooled_shape,
        hidden_shape,
      )));
    }
    return pool_post(
      pooled,
      cfg.normalize,
      cfg.dimension,
      cfg.apply_layer_norm,
      cfg.apply_rms_norm,
    );
  }

  pool(
    &last_hidden_state,
    &attention_mask,
    cfg.strategy,
    cfg.normalize,
    cfg.dimension,
    cfg.apply_layer_norm,
    cfg.apply_rms_norm,
  )
}

#[cfg(test)]
mod tests;
