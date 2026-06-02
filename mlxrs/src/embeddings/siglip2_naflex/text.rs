//! SigLIP2 text tower.
//!
//! Ports `siglip.py`'s `SiglipTextEmbeddings` / `SiglipTextTransformer`: a
//! token embedding + a learned absolute position embedding, the shared
//! pre-norm encoder stack (the private `shared::EncoderLayer`,
//! `approx="precise"`), a final `LayerNorm`, and the pooled contrastive
//! projection.
//!
//! ## Pooling head (sticky-EOS last token)
//!
//! `siglip.py`'s `SiglipTextTransformer.__call__` pools the **last** sequence
//! position — `pooled_output = x[:, -1, :]` — under the comment "Assuming
//! 'sticky' EOS tokenization, last token is always EOS." The pooled vector is
//! then projected by `self.head = nn.Linear(hidden, projection_size)`. This
//! port mirrors that exactly: the caller pads/truncates each input to a fixed
//! length and the last position is taken, then the (biased) `head` Linear maps
//! `hidden → projection_size`. (The SigLIP text head is a *biased* `nn.Linear`
//! — `nn.Linear(embed_dim, config.projection_size)` with the default
//! `bias=True` — unlike the encoder's bias-free convention, so the bias tensor
//! is consumed.)
//!
//! ## Attention mask
//!
//! `Model.get_text_features` passes `attention_mask=None` into the text tower
//! (the contrastive path), so the text encoder runs **full** (non-causal)
//! bidirectional attention with no mask — [`Mask::None`]. SigLIP pads to a
//! fixed length with a real pad token and does not mask it (the sticky-EOS
//! pooling reads the last position regardless), matching the reference.

use std::collections::HashMap;

use crate::{
  array::Array,
  embeddings::siglip2_naflex::{
    config::TextConfig,
    shared::{
      EncoderLayer, LayerDims, QuantLinear, build_layer_norm, dim_i32, expect_logical_shape,
      expect_shape, resolve_quant,
    },
  },
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  lm::{
    nn::{attention::Mask, norm::LayerNorm},
    quant::PerLayerQuantization,
  },
  model_validation::reserve_or_error,
  nn::MaybeQuantizedEmbedding,
  ops,
};

/// The SigLIP2 text transformer: token + position embedding → pre-norm encoder
/// → final LayerNorm → last-token pooled projection.
///
/// Ports `siglip.py`'s `SiglipTextTransformer`. The public
/// [`Siglip2NaflexModel`](super::Siglip2NaflexModel) drives this with a
/// `(batch, seq_len)` i32 token-id batch.
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub struct TextTower {
  /// Token-embedding table `(vocab, hidden)` — quantize-aware: a token lookup is
  /// an axis-0 row gather, which the [`MaybeQuantizedEmbedding`] dequantize-gather
  /// handles directly for a quantized checkpoint.
  token_embedding: MaybeQuantizedEmbedding,
  /// Position-embedding table `(max_position_embeddings, hidden)`, materialized
  /// dense at load (dequantized once if the checkpoint quantized it): the
  /// position lookup is a contiguous row slice (`arange(seq)`), which a packed
  /// quantized table cannot be sliced for, so the dense table keeps the slice
  /// path unchanged.
  position_embedding: Array,
  layers: Vec<EncoderLayer>,
  final_layer_norm: LayerNorm,
  /// Pooled projection `head` (biased `Linear(hidden, projection_size)`) —
  /// quantize-aware ([`QuantLinear`]).
  head: QuantLinear,
  max_position_embeddings: i32,
  hidden: i32,
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl TextTower {
  /// Build the text tower from a validated [`TextConfig`] and the (sanitized)
  /// weight map with the `text_model.text_model.` prefix already stripped (so
  /// keys are e.g. `embeddings.token_embedding.weight`,
  /// `encoder.layers.0.self_attn.q_proj.weight`, `final_layer_norm.weight`,
  /// `head.weight`, matching `siglip.py`'s module tree).
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions (typed [`crate::Error::ShapePairMismatch`] wrapped in
  /// [`crate::Error::LayerKeyed`]).
  pub fn from_weights(config: &TextConfig, weights: &mut HashMap<String, Array>) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build the text tower with an optional parsed quantization config — the
  /// quantization-aware analogue of [`from_weights`](Self::from_weights) (which
  /// is just this with `quant == None`).
  ///
  /// Each `nn.Linear` (the attention q/k/v/out, the MLP fc1/fc2, and the pooled
  /// `head`) auto-picks the dense or quantized variant per layer by its
  /// `<prefix>.scales` sibling (the `class_predicate`'s `f"{p}.scales" in
  /// weights` signal). The `token_embedding` is quantize-aware too (a quantized
  /// row gather dequantizes the gathered rows); the `position_embedding` is
  /// materialized dense at load (dequantized once if quantized) because the
  /// position lookup is a contiguous row slice a packed table cannot serve.
  pub fn from_weights_quantized(
    config: &TextConfig,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: `from_weights` is public, so a caller may build
    // a tower from a directly-constructed (unvalidated) config. This bounds
    // `num_hidden_layers` (and every other dim) before the per-layer
    // reservation/loop below.
    config.validate()?;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let vocab = config.vocab_size;
    let max_pos = config.max_position_embeddings;
    let proj = config.projection_size();
    // Per-layer shape constants (validates num_heads positive + divides hidden,
    // and computes the head split / SDPA scale once).
    let dims = LayerDims::new(hidden, inter, num_heads, config.layer_norm_eps as f32)?;
    let eps = dims.eps;

    // Token embedding — quantize-aware (an axis-0 row gather; dequantize-gather
    // for a quantized table). Pin the LOGICAL `(vocab, hidden)` to the config for
    // BOTH arms via `logical_shape`: the dense variant from the table's own shape,
    // the quantized variant from the dequantized `(num_embeddings, dim)` recovered
    // from the validated triple (no whole-table dequantization). A quantized table
    // whose logical shape disagrees would otherwise mis-gather at the first
    // forward instead of failing at load.
    let token_embedding = MaybeQuantizedEmbedding::from_weights(
      weights,
      "embeddings.token_embedding",
      resolve_quant(quant, "embeddings.token_embedding"),
    )?;
    expect_logical_shape(
      &token_embedding,
      "embeddings.token_embedding.weight",
      "text token-embedding table (vocab, hidden)",
      vocab,
      hidden,
    )?;

    // Position embedding — materialized dense (dequantized once if quantized):
    // the lookup is a contiguous row slice a packed table cannot serve. Pin the
    // materialized table to the config `(max_position_embeddings, hidden)`.
    let position_embedding = {
      let pos_quant = MaybeQuantizedEmbedding::from_weights(
        weights,
        "embeddings.position_embedding",
        resolve_quant(quant, "embeddings.position_embedding"),
      )?;
      let table = pos_quant.dense_table(None)?;
      expect_shape(
        &table,
        "embeddings.position_embedding.weight",
        "text position-embedding table (max_position_embeddings, hidden)",
        &[max_pos, hidden],
      )?;
      table
    };

    // `num_hidden_layers` is required positive in `validate`, but reserve
    // fallibly so even a heavyweight per-layer `Vec` the allocator cannot
    // satisfy is a recoverable [`Error::AllocFailure`] rather than
    // `with_capacity`'s abort (the merged LFM2 / Wav2Vec2 pattern).
    let mut layers: Vec<EncoderLayer> = Vec::new();
    reserve_or_error(
      &mut layers,
      "EncoderLayer",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      layers.push(EncoderLayer::from_weights(
        weights, "encoder", i, dims, quant,
      )?);
    }

    let final_layer_norm = build_layer_norm(weights, "final_layer_norm", hidden, eps)?;

    // The text head is a BIASED Linear(hidden, projection_size) — quantize-aware.
    let head = QuantLinear::from_weights(weights, "head", proj, hidden, true, quant)?;

    Ok(Self {
      token_embedding,
      position_embedding,
      layers,
      final_layer_norm,
      head,
      max_position_embeddings: max_pos,
      hidden,
    })
  }

  /// Forward a `(batch, seq_len)` i32 token-id batch through the tower.
  ///
  /// Returns the pooled projected text embedding `(batch, projection_size)` —
  /// `siglip.py`'s `pooled_output = head(x[:, -1, :])` (the sticky-EOS last
  /// token). `seq_len` must be in `1..=max_position_embeddings`.
  pub fn forward(&self, input_ids: &Array) -> Result<Array> {
    let shape = input_ids.shape();
    // Pin `input_ids` to EXACTLY rank-2 `(batch, seq_len)` before any op. The
    // public `encode_text` / `embed_text` accept an untrusted array, and the
    // sticky-EOS pooling + the position-row slice are only defined for a rank-2
    // batch: a rank-3+ input would otherwise pass the `shape[1]` read and gather
    // a different-rank graph (`(B, L, X, hidden)`), and a rank-<2 input would
    // index past its own rank. Reject anything but rank-2 up front, mirroring
    // the vision tower's runtime-input shape gate.
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "siglip2 text: input_ids must be rank-2 (batch, seq_len)",
        shape.len() as u32,
        shape,
      )));
    }
    let seq = dim_i32(&shape, 1, "siglip2 text: seq_len")?;
    if seq < 1 {
      // An empty sequence axis has no last token to pool: `index_last(0)` would
      // build the `[-1]` index and `take_axis` would run on the empty axis (a
      // backend / negative-index path). Reject it up front, before any
      // embedding lookup, as a typed error. The sticky-EOS pooling
      // (`x[:, -1, :]`) is only defined for a non-empty sequence.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "siglip2 text: seq_len",
        "must be a positive sequence length (>= 1)",
        smol_str::format_smolstr!("{seq}"),
      )));
    }
    if seq > self.max_position_embeddings {
      // `siglip.py`'s `SiglipTextEmbeddings.__call__` raises when seq_len >
      // max_position_embeddings; surface the same as a recoverable error.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "siglip2 text: seq_len",
        "must not exceed max_position_embeddings",
        smol_str::format_smolstr!("{seq} > {}", self.max_position_embeddings),
      )));
    }

    // token_embedding(ids): (B, L) → (B, L, hidden) via axis-0 gather (a
    // dequantize-gather for a quantized table).
    let tok = self.token_embedding.gather(input_ids)?;
    // position_embedding(arange(L)): (L, hidden), sliced from the table's
    // first L rows (position_ids = arange(seq_len)), then broadcast-added.
    let pos = self.position_rows(seq)?; // (L, hidden)
    // (B, L, hidden) + (L, hidden) broadcasts over the batch axis.
    let mut h = tok.add(&pos)?;

    let mask = Mask::None;
    for layer in &self.layers {
      h = layer.forward(&h, mask)?;
    }
    let h = self.final_layer_norm.forward(&h)?;

    // pooled = x[:, -1, :] → (B, hidden): take the last sequence position.
    let last = ops::indexing::take_axis(&h, &index_last(seq)?, 1)?; // (B, 1, hidden)
    let last = ops::shape::squeeze_axes(&last, &[1])?; // (B, hidden)
    // head(pooled): (B, hidden) → (B, projection_size).
    self.head.forward(&last)
  }

  /// The first `seq` rows of the position-embedding table — the lazy analogue
  /// of `position_embedding(arange(seq))`. Returns `(seq, hidden)`.
  fn position_rows(&self, seq: i32) -> Result<Array> {
    let lo = [0i32, 0];
    let hi = [seq, self.hidden];
    let strides = [1i32, 1];
    ops::indexing::slice(&self.position_embedding, &lo, &hi, &strides)
  }
}

/// The `(1,)` i32 index array `[seq - 1]`, for `take_axis(h, axis=1)` —
/// selecting the last sequence position (`x[:, -1, :]`).
#[cfg(feature = "siglip2-naflex")]
fn index_last(seq: i32) -> Result<Array> {
  Array::from_slice::<i32>(&[seq - 1], &(1usize,))
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
