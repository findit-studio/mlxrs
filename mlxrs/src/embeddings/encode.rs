//! The `encode` entry — tokenize a batch of texts, pad to the batch's max
//! length, and run them through a model's universal text embedding into a
//! `(batch, dim)` embedding matrix.
//!
//! Ports the orchestration of python `mlx-embeddings` `utils.py::generate`
//! (tokenize with padding / truncation / `max_length`, run the model, return the
//! embeddings) and swift `MLXEmbedders` `EmbedderModelContainer.perform` (encode
//! each text → pad to the batch max → build the mask → run the model → `eval`).
//!
//! Pooling and normalization are the **model's** concern now: a sentence-encoder
//! owns its [`PoolingStrategy`](crate::embeddings::PoolingStrategy) (resolved at
//! load from `1_Pooling/config.json`) and applies it inside its
//! [`TextEmbedder::embed_text`] via
//! [`pool_embed`](crate::embeddings::pool_embed); a dual-tower model's text tower
//! embeds directly. So `encode` is thin — it tokenizes a batch and calls
//! `embed_text`. The pooling / normalization / similarity helpers remain public
//! drivers a model composes; `encode` itself no longer pools.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, OutOfRangePayload, Result,
    try_with_capacity,
  },
  tokenizer::{EncodeOptions, Tokenizer},
};

use super::embed::TextEmbedder;

/// Configuration for [`encode`] — the tokenization knobs.
///
/// Defaults mirror python `generate` (`max_length = 512`, padding + truncation
/// on, special tokens added). Pooling / normalization are no longer here: the
/// model owns them (see the module docs).
///
/// Build via [`EncodeConfig::new`] and chain `with_*` setters:
///
/// ```rust,ignore
/// let cfg = EncodeConfig::new().with_max_length(Some(256));
/// ```
#[derive(Debug, Clone)]
pub struct EncodeConfig {
  /// Add the tokenizer's special tokens (BOS/EOS/sep) when encoding, as in
  /// python `processor(..., add_special_tokens=True)` (the transformers
  /// default) and swift `tokenizer.encode(text:, addSpecialTokens: true)`.
  /// Default `true`.
  add_special_tokens: bool,
  /// Per-sequence hard token cap (python `truncation=True`, `max_length=512`):
  /// each text is right-truncated (keep the head, drop the tail) to at most
  /// this many ids *before* batch padding. `None` disables truncation. Default
  /// `Some(512)`.
  max_length: Option<usize>,
  /// Token id written into padding positions. The attention mask is `0` there,
  /// so this value never reaches the embedding — it exists only so the padded
  /// `(batch, seq_len)` id tensor is well-formed (swift pads with `0`). Default
  /// `0`.
  pad_token_id: u32,
}

impl Default for EncodeConfig {
  fn default() -> Self {
    Self {
      add_special_tokens: true,
      max_length: Some(512),
      pad_token_id: 0,
    }
  }
}

impl EncodeConfig {
  /// Fluent builder constructor; equivalent to [`EncodeConfig::default`].
  pub fn new() -> Self {
    Self::default()
  }

  // ── builders ──────────────────────────────────────────────────────────

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

  // ── accessors ─────────────────────────────────────────────────────────

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
}

/// Tokenize `texts`, right-pad each id row to the batch's max length with
/// `pad_token_id`, and build the matching `(batch, seq_len)` attention mask
/// (`1` for real tokens, `0` for padding).
///
/// Returns `(input_ids, attention_mask, seq_len)`:
/// - `input_ids` — `(batch, seq_len)` `i32` array (right-padded). `I32` is
///   MLX's default index dtype for the embedding `take` / gather a model
///   performs (matching `lm/generate.rs::token_window`), so the lookup never
///   has to cast. Each `u32` id is converted with a CHECKED `i32::try_from`
///   (a token id `> i32::MAX` — realistically never — yields a recoverable
///   [`Error::OutOfRange`] rather than silently wrapping negative);
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
/// 3. call `model.embed_text(input_ids, Some(attention_mask))` — the model
///    applies its own pooling / normalization (a sentence-encoder via its
///    configured [`PoolingStrategy`](crate::embeddings::PoolingStrategy) inside
///    [`pool_embed`](crate::embeddings::pool_embed); a dual-tower text tower
///    directly).
///
/// The returned array is the model's text embedding, conventionally
/// `(batch, dim)` and L2-normalized. **No implicit eval**: the result is a lazy
/// graph node; the caller evaluates (or reads it) when ready.
///
/// An empty `texts` slice produces a `(0, seq_len)` id batch; the model's
/// `embed_text` returns the correspondingly zero-row embedding.
///
/// - `model` — any [`TextEmbedder`] (every text-capable model is one via the
///   [`Embed<TextInput>`](crate::embeddings::Embed) projection; reach it from a
///   loaded [`EmbeddingModel`](crate::embeddings::EmbeddingModel) via
///   [`as_text_embedder`](crate::embeddings::EmbeddingModel::as_text_embedder));
/// - `tokenizer` — the loaded [`Tokenizer`] (local-only; no network);
/// - `texts` — the batch to encode;
/// - `cfg` — tokenization knobs ([`EncodeConfig`]).
pub fn encode(
  model: &dyn TextEmbedder,
  tokenizer: &Tokenizer,
  texts: &[&str],
  cfg: &EncodeConfig,
) -> Result<Array> {
  // Fail fast on a cleared/poisoned worker thread (and install the mlx-c error
  // handler) before any work, since `embed_text` touches per-thread stream/TLS
  // state. Mirrors the crate's other safe entry points (e.g.
  // `stream::default_stream`), which install the handler before asserting.
  crate::error::ensure_handler_installed();
  crate::stream::assert_streams_not_cleared();
  let (input_ids, attention_mask, _seq_len) = tokenize_and_pad(
    tokenizer,
    texts,
    cfg.add_special_tokens,
    cfg.max_length,
    cfg.pad_token_id,
  )?;
  let embedding = model.embed_text(&input_ids, Some(&attention_mask))?;
  Ok(embedding.into_array())
}

#[cfg(test)]
mod tests;
