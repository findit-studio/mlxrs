//! LFM2.5-VL language adapter — faithful 1:1 port of
//! `mlx-vlm/mlx_vlm/models/lfm2_vl/language.py`'s `LanguageModel`.
//!
//! `language.py`'s `LanguageModel` is a thin wrapper around the LFM2 LM
//! (`mlx_lm.models.lfm2.Lfm2Model`): it forwards token ids (or pre-computed
//! `inputs_embeds`) through the decoder and applies the tied logit head
//! (`self.model.embed_tokens.as_linear(out)`). This adapter wraps the ported
//! [`Lfm2`] the same way, adding the embed-from-ids
//! ([`LanguageModel::embed_tokens`]) and forward-from-embeddings
//! ([`LanguageModel::forward_embeddings`]) entry points the VL model needs to
//! splice image features between the token embed and the decoder.
//!
//! ## Guarded forward-from-embeddings
//!
//! The reference passes `inputs_embeds` straight into `Lfm2Model.__call__`,
//! which indexes `inputs_embeds.shape[1]` (and feeds the per-layer mixers a
//! `(B, T, hidden)` tensor) with no shape validation — a rank-2 / wrong-width
//! input would fail deep inside the decoder (a Python `IndexError` /
//! shape-mismatch) rather than at the boundary. This adapter adds a **preflight**
//! that pins `inputs_embeds` to rank-3 `(batch, seq, hidden)` and the hidden
//! width to the LM's `hidden_size` BEFORE the decoder runs, surfacing a typed
//! [`Error`] instead — the same boundary discipline the rest of the crate's
//! model-input seams use. [`LanguageModel::embed_tokens`] likewise range-guards
//! the token ids against the LM vocabulary before the embedding gather (mlx's
//! gather
//! silently clamps an out-of-range id; the guard surfaces a typed error so a
//! malformed prompt is caught at the boundary, not silently mis-embedded).

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  lm::{cache::KvCache, model::Model as LmModel, models::lfm2::Lfm2},
  ops,
};

/// The LFM2.5-VL language adapter — a thin wrapper over the ported [`Lfm2`] LM
/// (`language.py`'s `LanguageModel`).
///
/// Owns the LFM2 LM and exposes the three entry points the VL
/// [`Model`](crate::vlm::models::lfm2_vl) drives: [`Self::embed_tokens`] (the
/// token embed the image splice writes into), [`Self::forward_embeddings`] (the
/// guarded forward-from-merged-embeddings), and [`Self::forward`] (the
/// text-only forward-from-ids). The tied logit head is applied inside the LM's
/// forward (`embed_tokens.as_linear`), exactly as `language.py` does.
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct LanguageModel {
  model: Lfm2,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl LanguageModel {
  /// Wrap a constructed [`Lfm2`] LM.
  pub fn new(model: Lfm2) -> Self {
    Self { model }
  }

  /// Read-only view of the wrapped LFM2 LM.
  #[inline(always)]
  pub fn inner(&self) -> &Lfm2 {
    &self.model
  }

  /// Build the heterogeneous per-layer cache (`language.py`'s `make_cache`),
  /// delegating to [`Lfm2::make_cache`].
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    self.model.make_cache()
  }

  /// Gather the token embeddings `embed_tokens(input_ids)` — the
  /// `language_model.model.embed_tokens(input_ids)` entry point
  /// (`lfm2_vl.py:124`), range-guarded.
  ///
  /// `input_ids` is an integer `(B, T)` (or `(T,)`) array; the result is
  /// `input_ids.shape ++ (hidden,)`. The ids are validated to lie in
  /// `[0, vocab_size)` before the gather — mlx's embedding gather silently
  /// clamps an out-of-range id, so this surfaces a malformed prompt as a typed
  /// [`Error::OutOfRange`] at the boundary instead of mis-embedding it. The
  /// validation reads the (small) ids host-side (an explicit eval on a clone);
  /// the returned embedding stays lazy.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if any id is `< 0` or `>= vocab_size`;
  /// - propagates the dtype-cast / eval / gather op errors.
  pub fn embed_tokens(&self, input_ids: &Array) -> Result<Array> {
    self.check_token_id_range(input_ids)?;
    self.model.embed_tokens(input_ids)
  }

  /// Run the LM over pre-computed input embeddings (the merged
  /// text+image embeddings), with a rank/width preflight — the
  /// `language_model(..., inputs_embeds=...)` path (`lfm2_vl.py:202-204`).
  ///
  /// `inputs_embeds` must be rank-3 `(batch, seq, hidden)` with `hidden` equal
  /// to the LM's `hidden_size`; a wrong rank / width is a typed [`Error`] here
  /// rather than a shape failure deep inside the decoder. Returns the
  /// `(batch, seq, vocab_size)` logits (the tied head is applied inside the LM
  /// forward).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `inputs_embeds` is not rank-3;
  /// - [`Error::OutOfRange`] if the trailing hidden width `!= hidden_size`;
  /// - propagates the decoder forward errors (including the per-layer cache
  ///   count check).
  pub fn forward_embeddings(
    &self,
    inputs_embeds: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    let shape = inputs_embeds.shape();
    if shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl language: inputs_embeds must be rank-3 (batch, seq, hidden)",
        shape.len() as u32,
        shape,
      )));
    }
    let hidden = self.model.config().hidden_size;
    let width = shape[2];
    if width != hidden as usize {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl language: inputs_embeds hidden width vs LM hidden_size",
        "trailing dim must equal the LM hidden_size",
        smol_str::format_smolstr!("hidden={width}, hidden_size={hidden}"),
      )));
    }
    // The LM's `forward_embeddings` (the `lm::model::Model` trait method) runs
    // the decoder over the embeddings and applies the tied logit head.
    LmModel::forward_embeddings(&self.model, inputs_embeds, cache)
  }

  /// Run the LM over token ids (the text-only path) — `language.py`'s
  /// `__call__(inputs, ...)` without `inputs_embeds`.
  ///
  /// Range-guards the ids (as [`Self::embed_tokens`] does), then delegates to
  /// [`Lfm2`]'s forward-from-ids. Returns the `(B, T, vocab_size)` logits.
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if any id is out of `[0, vocab_size)`;
  /// - propagates the decoder forward errors.
  pub fn forward(&self, input_ids: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    self.check_token_id_range(input_ids)?;
    LmModel::forward(&self.model, input_ids, cache)
  }

  /// Validate that every token id lies in `[0, vocab_size)` — the embedding
  /// gather's valid row range. Reads the (small) ids host-side (an explicit
  /// eval on a clone cast to i32) so a malformed id is a typed
  /// [`Error::OutOfRange`] rather than a silent mlx gather clamp.
  fn check_token_id_range(&self, input_ids: &Array) -> Result<()> {
    let vocab = self.model.config().vocab_size;
    let mut ids = ops::misc::astype(input_ids, Dtype::I32)?;
    ids.eval()?;
    let flat = ids.to_vec::<i32>()?;
    for &id in &flat {
      if id < 0 || id >= vocab {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "lfm2_vl language: token id vs vocabulary range [0, vocab_size)",
          "every input id must be a valid embedding row",
          smol_str::format_smolstr!("id={id}, vocab_size={vocab}"),
        )));
      }
    }
    Ok(())
  }
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
