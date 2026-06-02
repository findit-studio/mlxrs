//! The Whisper text decoder — `TextDecoder` (`whisper.py:440-486`).
//!
//! Faithful port of the autoregressive decoder that maps text tokens (+ the
//! encoder states) to vocabulary logits:
//!
//! 1. `token_embedding(tokens)` (a `(n_vocab, n_state)` gather) **plus** the
//!    **learned** `positional_embedding[offset : offset + T]` — the positional
//!    table is sliced by the KV-cache offset so an incremental decode step
//!    indexes the right absolute positions;
//! 2. `n_text_layer` × [`ResidualAttentionBlock`] with `cross_attention=True`:
//!    masked self-attention → cross-attention over the encoder states → MLP;
//! 3. a final `ln` LayerNorm;
//! 4. weight-tied logits via [`Embedding::as_linear`].
//!
//! The self-attention causal mask is precomputed once (`(n_text_ctx,
//! n_text_ctx)`) and sliced offset-aware to the new query window inside the
//! attention (`mask[offset : offset + T, 0 : offset + T]`), so an incremental
//! warm-cache step masks each new token against the keys at or before its
//! absolute position.

use crate::{
  Array, Result,
  error::{Error, OutOfRangePayload},
  lm::nn::norm::LayerNorm,
  ops::{self, indexing::slice},
};
use smol_str::format_smolstr;

use super::layers::{BlockKvCache, Embedding, ResidualAttentionBlock};

/// The decoder KV cache: one [`BlockKvCache`] per transformer block.
///
/// Mirrors the reference's `kv_cache` list (`whisper.py:477-483`), one
/// `(self_kv, cross_kv)` entry per block. `None` on the first call (the
/// reference's `kv_cache = [None] * len(self.blocks)`).
pub(crate) type DecoderKvCache = Vec<BlockKvCache>;

/// The per-layer cross-attention weights of one decoder forward — the
/// reference's `cross_qk` list (`whisper.py:479-483`), one entry per
/// transformer block.
///
/// Each entry is the block's cross-attention `qk` tensor `(B, H, T, n_audio_ctx)`
/// (the pre-softmax scaled scores; the alignment signal the later
/// word-timestamp DTW consumes). Every decoder block carries cross-attention,
/// so in practice every entry is `Some`; the `Option` mirrors the reference's
/// `cross_qk = [None] * len(self.blocks)` initialization (a block without
/// cross-attention would leave its slot `None`).
pub(crate) type DecoderCrossQk = Vec<Option<Array>>;

/// The Whisper text decoder (`whisper.py:440-486`).
#[derive(Debug)]
pub(crate) struct TextDecoder {
  token_embedding: Embedding,
  /// Learned `(n_text_ctx, n_state)` positional embedding (kept from the
  /// checkpoint — unlike the encoder's computed sinusoids).
  positional_embedding: Array,
  blocks: Vec<ResidualAttentionBlock>,
  ln: LayerNorm,
  /// Precomputed additive causal mask `(n_text_ctx, n_text_ctx)`, sliced to the
  /// query length inside each block's self-attention.
  mask: Array,
}

impl TextDecoder {
  /// Construct from the loaded sub-modules. `n_ctx` (= `n_text_ctx`, 448) sizes
  /// the precomputed causal mask; the mask is built in `dtype` (the model
  /// dtype) so the `qk + mask` add inside attention stays in that dtype.
  ///
  /// # Errors
  /// Propagates the causal-mask construction op errors.
  pub(crate) fn new(
    token_embedding: Embedding,
    positional_embedding: Array,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    n_ctx: usize,
    dtype: crate::Dtype,
  ) -> Result<Self> {
    let mask = create_additive_causal_mask(n_ctx, dtype)?;
    Ok(Self {
      token_embedding,
      positional_embedding,
      blocks,
      ln,
      mask,
    })
  }

  /// The number of transformer blocks (`n_text_layer`).
  #[cfg(test)]
  pub(crate) fn num_blocks(&self) -> usize {
    self.blocks.len()
  }

  /// Read-only reference to the precomputed additive causal mask.
  #[cfg(test)]
  pub(crate) fn mask_ref(&self) -> &Array {
    &self.mask
  }

  /// The KV offset for a forward call — the self-attention key time dimension
  /// already stored in the first block's cache, or `0` for a fresh cache.
  /// Mirrors `offset = kv_cache[0][0][0].shape[1] if kv_cache else 0`
  /// (`whisper.py:471`).
  fn cache_offset(kv_cache: Option<&DecoderKvCache>) -> usize {
    match kv_cache {
      Some(cache) => match cache.first() {
        // `block[0].0` is the self-attention `(k, v)`; `k.shape[1]` is its
        // accumulated time dimension.
        Some((Some((k, _)), _)) => k.shape().get(1).copied().unwrap_or(0),
        _ => 0,
      },
      None => 0,
    }
  }

  /// Run the decoder. Faithful port of `TextDecoder.__call__`
  /// (`whisper.py:464-486`).
  ///
  /// - `tokens`: the text token ids `(B, T)`.
  /// - `xa`: the encoder states `(B, n_audio_ctx, n_audio_state)`.
  /// - `kv_cache`: the incoming per-block cache; `None` on the first step.
  ///
  /// Returns `(logits, updated_cache)` — the vocabulary logits `(B, T,
  /// n_vocab)` and the per-block cache to thread into the next decode step. The
  /// reference's third `cross_qk` return is dropped on this path;
  /// [`Self::forward_with_cross_qk`] is the variant that surfaces it.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if the positional slice `offset + T` exceeds the
  ///   positional table or `i32::MAX`;
  /// - [`Error::AllocFailure`] if the fresh per-block cache vector cannot be
  ///   reserved;
  /// - propagates the embedding / block / LayerNorm / logit op errors.
  pub(crate) fn forward(
    &self,
    tokens: &Array,
    xa: &Array,
    kv_cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache)> {
    let (logits, cache, _cross_qk) = self.run(tokens, xa, kv_cache, false)?;
    Ok((logits, cache))
  }

  /// Run the decoder and also surface the per-layer cross-attention weights —
  /// the full three-tuple return of `TextDecoder.__call__`
  /// (`whisper.py:464-486`), mirroring the reference's
  /// `logits_with_cross_qk` (`decoding.py:177-189`).
  ///
  /// Returns `(logits, updated_cache, cross_qk)` where `cross_qk` is the
  /// [`DecoderCrossQk`] list — one cross-attention `qk` tensor `(B, H, T,
  /// n_audio_ctx)` per decoder block. The weights are the alignment signal the
  /// later word-timestamp DTW consumes; this method only extracts and exposes
  /// them.
  ///
  /// # Errors
  /// Same as [`Self::forward`].
  pub(crate) fn forward_with_cross_qk(
    &self,
    tokens: &Array,
    xa: &Array,
    kv_cache: Option<&DecoderKvCache>,
  ) -> Result<(Array, DecoderKvCache, DecoderCrossQk)> {
    self.run(tokens, xa, kv_cache, true)
  }

  /// The shared decoder forward, optionally collecting the per-layer cross-
  /// attention weights. `collect_cross_qk` gates whether each block's
  /// cross-attention `qk` is retained (it is dropped when `false`, so the
  /// normal decode path keeps the weights' lifetime to a single block).
  ///
  /// # Errors
  /// See [`Self::forward`].
  fn run(
    &self,
    tokens: &Array,
    xa: &Array,
    kv_cache: Option<&DecoderKvCache>,
    collect_cross_qk: bool,
  ) -> Result<(Array, DecoderKvCache, DecoderCrossQk)> {
    let offset = Self::cache_offset(kv_cache);
    let seq_len = *tokens.shape().last().unwrap_or(&0);

    // Bound the decode context BEFORE the token-embedding gather: a direct
    // caller can pass a prefix longer than the validated `n_text_ctx`, and the
    // gather would otherwise materialize `(1, T, n_text_state)` before the
    // positional slice's bound check fires. Reject `offset + seq_len >
    // n_text_ctx` here, so no allocation precedes the typed error.
    self.check_context(offset, seq_len)?;

    // `x = token_embedding(tokens) + positional_embedding[offset:offset+T]`.
    let token_emb = self.token_embedding.forward(tokens)?;
    let pe_slice = self.positional_slice(offset, seq_len)?;
    let mut x = token_emb.add(&pe_slice)?;

    // `n_text_layer` × ResidualAttentionBlock(cross_attention=True): masked
    // self-attn → cross-attn over `xa` → MLP, threading the per-block cache. The
    // block count is bounded by `MAX_LAYERS` at config construction; reserve the
    // fresh cache vector fallibly (a typed `AllocFailure` instead of the abort
    // `Vec::with_capacity` would raise), then push into the reserved capacity.
    let mut new_cache: DecoderKvCache = Vec::new();
    crate::model_validation::reserve_or_error(
      &mut new_cache,
      "decoder KV cache",
      self.blocks.len(),
    )?;
    // Only reserve the cross-qk collector when the caller wants the weights.
    let mut cross_qk: DecoderCrossQk = Vec::new();
    if collect_cross_qk {
      crate::model_validation::reserve_or_error(
        &mut cross_qk,
        "decoder cross-attention weights",
        self.blocks.len(),
      )?;
    }
    for (i, block) in self.blocks.iter().enumerate() {
      let block_cache = kv_cache.and_then(|c| c.get(i));
      // Branch on `collect_cross_qk` so an ordinary decode takes the plain
      // `forward` — which never materializes or returns the cross-attention
      // score tensor past the attention computation — instead of carrying that
      // `(B, H, T, n_audio_ctx)` tensor live through the residual / MLP only to
      // drop it. The qk-collecting path keeps the per-layer weights.
      if collect_cross_qk {
        let (out, updated, qk) =
          block.forward_with_cross_qk(&x, Some(xa), Some(&self.mask), block_cache)?;
        x = out;
        new_cache.push(updated);
        cross_qk.push(qk);
      } else {
        let (out, updated) = block.forward(&x, Some(xa), Some(&self.mask), block_cache)?;
        x = out;
        new_cache.push(updated);
      }
    }

    // Final LayerNorm, then weight-tied logits `token_embedding.as_linear(x)`.
    let x = self.ln.forward(&x)?;
    let logits = self.token_embedding.as_linear(&x)?;
    Ok((logits, new_cache, cross_qk))
  }

  /// Reject a decode window whose absolute span `offset + seq_len` would
  /// exceed the learned positional table (`n_text_ctx` rows) — the decode
  /// context bound, factored out so [`Self::forward`] can enforce it BEFORE the
  /// token-embedding gather allocates, and [`Self::positional_slice`] re-checks
  /// it (defense-in-depth) at the actual slice. Returns the validated
  /// exclusive end `offset + seq_len`.
  ///
  /// # Errors
  /// [`Error::OutOfRange`] if `offset + seq_len` overflows `usize` or exceeds
  /// the positional table's row count.
  fn check_context(&self, offset: usize, seq_len: usize) -> Result<usize> {
    let n_ctx = self.positional_embedding.shape()[0];
    let end = offset.checked_add(seq_len).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "TextDecoder: positional slice offset + seq_len",
        "must not overflow usize",
        format_smolstr!("offset={offset}, seq_len={seq_len}"),
      ))
    })?;
    if end > n_ctx {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "TextDecoder: positional slice end exceeds n_text_ctx",
        "offset + seq_len must be <= positional_embedding rows",
        format_smolstr!("end={end}, n_text_ctx={n_ctx}"),
      )));
    }
    Ok(end)
  }

  /// `positional_embedding[offset : offset + seq_len]` — slice the learned
  /// positional table to the current absolute positions.
  fn positional_slice(&self, offset: usize, seq_len: usize) -> Result<Array> {
    let pe_shape = self.positional_embedding.shape();
    let end = self.check_context(offset, seq_len)?;
    let n_state = i32::try_from(pe_shape[1]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "TextDecoder: n_state",
        "must fit in i32",
        format_smolstr!("{}", pe_shape[1]),
      ))
    })?;
    let start_i32 = i32::try_from(offset).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "TextDecoder: positional slice start",
        "must fit in i32",
        format_smolstr!("{offset}"),
      ))
    })?;
    let end_i32 = i32::try_from(end).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "TextDecoder: positional slice end",
        "must fit in i32",
        format_smolstr!("{end}"),
      ))
    })?;
    slice(
      &self.positional_embedding,
      &[start_i32, 0],
      &[end_i32, n_state],
      &[1, 1],
    )
  }
}

/// Build Whisper's additive causal self-attention mask — a port of
/// `nn.MultiHeadAttention.create_additive_causal_mask(N)`
/// (`mlx/python/mlx/nn/layers/transformer.py`).
///
/// The reference is
/// ```text
/// indices = arange(N)
/// mask = (indices[:, None] < indices[None]).astype(dtype) * finfo(dtype).min
/// ```
/// i.e. `0` on/below the diagonal, the dtype's most-negative value strictly
/// above it. This port uses `-inf` (cast to `dtype`) for the above-diagonal
/// entries instead of `finfo(dtype).min`: post-softmax the two are identical
/// for a causal mask (every row keeps its on-diagonal entry, so no row is fully
/// masked → no `NaN`), `-inf` is the established masking idiom across this crate
/// (`lm::structured` / `lm::sample`), and it avoids the f16 `qk + finfo.min`
/// underflow the literal reference risks. The mask is sliced offset-aware to the
/// new query window (`mask[offset : offset + T, 0 : offset + T]`) inside the
/// attention, so a warm-cache step over `T` new tokens masks correctly.
fn create_additive_causal_mask(n: usize, dtype: crate::Dtype) -> Result<Array> {
  let n_i32 = i32::try_from(n).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "create_additive_causal_mask: N",
      "must fit in i32",
      format_smolstr!("{n}"),
    ))
  })?;
  // indices[:, None] < indices[None] — a strictly-upper-triangular bool mask.
  let indices = Array::arange::<i32>(0.0, n as f64, 1.0)?;
  let col = ops::shape::expand_dims_axes(&indices, &[1])?; // (N, 1)
  let row = ops::shape::expand_dims_axes(&indices, &[0])?; // (1, N)
  let upper = ops::comparison::less(&col, &row)?; // bool (N, N), true above diag
  // where(upper, -inf, 0), cast to the model dtype.
  let neg_inf = ops::misc::astype(&Array::full::<f32>(&[0i32; 0], f32::NEG_INFINITY)?, dtype)?;
  let zero = ops::misc::astype(&Array::full::<f32>(&[0i32; 0], 0.0)?, dtype)?;
  let mask = ops::logical::select(&upper, &neg_inf, &zero)?;
  // Materialize the `(N, N)` shape (select broadcasts the rank-0 scalars).
  ops::shape::broadcast_to(&mask, &[n_i32, n_i32])
}

#[cfg(test)]
mod tests;
