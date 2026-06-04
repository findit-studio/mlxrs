//! CLAP RoBERTa text tower (`ClapTextModel`) + the CLAP text projection.
//!
//! Ports HF `transformers`' `ClapTextModel` (a RoBERTa encoder) and the CLAP
//! text-feature path (`ClapModel.get_text_features`): the RoBERTa embeddings
//! (word + position with the `padding_idx` offset + token_type + post-norm
//! `LayerNorm`), the `num_hidden_layers` BERT **post-norm** encoder layers (the
//! private `shared` blocks), the CLS pooling (the first token of the last
//! hidden state), and the two-layer `ClapProjectionLayer` (Linear → ReLU →
//! Linear) that maps the `(B, hidden)` text feature into the shared
//! contrastive space, followed by L2-normalize.
//!
//! The strong Rust reference is the Findit-AI `textclap` crate's `src/text.rs`
//! (the CLAP RoBERTa text side, ONNX-backed): it pins the **position-id offset**
//! (`pad_id + 1 + cumsum(non_pad_mask)`, its design spec §7.4), the
//! `512`-real-token cap (`TEXT_MAX_TOKENS = 512`, `text.rs:31`), and the final
//! L2-normalize of the `512`-d embedding (`text.rs:24,273` —
//! `TEXT_OUTPUT_IS_UNIT_NORM = false`, so the tower normalizes itself).
//!
//! ## RoBERTa position-id offset (the critical detail)
//!
//! RoBERTa positions are **not** `0..seq`. HF
//! `create_position_ids_from_input_ids` computes
//! `mask = (input_ids != pad_token_id)`, then
//! `position_ids = cumsum(mask) * mask + pad_token_id` — so a real token gets
//! position `pad_id + 1, pad_id + 2, …` (with `pad_id = 1`, the first real token
//! is position **2**) and a pad token gets `pad_id`. Getting this wrong shifts
//! every position embedding (silent wrong embeddings). It is pinned by the
//! closed-form `position_ids_from_ids` and its oracle test (ids
//! `[0, 5, 9, 1, 1]`, `pad = 1` → positions `[2, 3, 4, 1, 1]`).
//!
//! ## Pooling → text feature + projection
//!
//! CLAP's text path takes the **first** sequence position (`<s>` / CLS,
//! position 0) of the last hidden state — `ClapModel.get_text_features` uses
//! `text_outputs[0][:, 0, :]` (the `last_hidden_state` CLS), NOT a BERT
//! `pooler` dense+tanh — and feeds it to the text projection
//! (`ClapProjectionLayer`: `linear1 → ReLU → linear2`, `projection_dim = 512`,
//! activation `relu`). The projected `(B, 512)` vector is L2-normalized into the
//! shared contrastive space (mirroring `siglip.py`'s `normalize_embeddings`).
//!
//! ## Quant + dtype
//!
//! Every `nn.Linear` (the q/k/v/output, the intermediate / output dense, the two
//! projection layers) and every `nn.Embedding` (word / position / token_type)
//! is quantize-aware via the shared `MaybeQuantizedLinear` /
//! `MaybeQuantizedEmbedding` (`.scales`-sibling `class_predicate`), so a
//! quantized CLAP checkpoint loads with a byte-identical dense path. The
//! position-id buffer, the additive padding mask, and any scalar floor that
//! meets the activations are cast back to the activation dtype before they are
//! combined, so an f16/bf16 checkpoint is not silently promoted to f32 (the
//! recurring activation-dtype-preservation faithfulness bug).
//!
//! ## Scope
//!
//! This is **phase 2** of the CLAP port (the text tower). The HTSAT Swin audio
//! tower (phase 3), the full dual-tower `ClapModel` assembly + `classify` +
//! the factory registration (phase 4), and the end-to-end checkpoint-parity test
//! (phase 5) are out of scope; [`ClapTextModel`] exposes a clean
//! [`embed_text`](ClapTextModel::embed_text) the assembly layer consumes.

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  embeddings::{
    Embedding, Padding, TextEmbedder, TextEncoding,
    clap::{
      config::ClapConfig,
      shared::{
        LayerDims, RobertaLayer, build_layer_norm, dim_i32, expect_logical_shape, expect_shape,
        resolve_quant,
      },
    },
    l2_normalize,
  },
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result, ShapePairMismatchPayload},
  lm::{
    nn::{attention::Mask, norm::LayerNorm},
    quant::PerLayerQuantization,
  },
  model_validation::reserve_or_error,
  nn::MaybeQuantizedEmbedding,
  ops,
};

use smol_str::format_smolstr;

/// The RoBERTa BPE pad token id (`pad_token_id = 1`). Drives the position-id
/// offset and the [`Padding::DynamicRightPad`] pad cells. Pinned by
/// [`ClapTextConfig::validate`](super::config::ClapTextConfig); hard-coded here
/// only as the `u32` form the [`TextEncoding`] needs (the config field is the
/// source of truth and is asserted against this at load).
#[cfg(feature = "clap")]
const PAD_TOKEN_ID: u32 = 1;

/// The maximum real-token sequence length CLAP's RoBERTa accepts
/// (`max_position_embeddings - 2 = 512`, the two reserved for the `padding_idx`
/// offset). Matches `textclap`'s `TEXT_MAX_TOKENS = 512` (`text.rs:31`).
#[cfg(feature = "clap")]
const TEXT_MAX_TOKENS: usize = 512;

// ═══════════════════════════ ClapTextEmbeddings ════════════════════════════

/// RoBERTa input embeddings (HF `ClapTextEmbeddings`, a `RobertaEmbeddings`):
/// `word_embeddings(ids) + position_embeddings(position_ids) +
/// token_type_embeddings(0)`, then a post-norm `LayerNorm` (dropout is identity
/// at inference).
///
/// The **position ids** carry the RoBERTa `padding_idx` offset (see
/// [`position_ids_from_ids`] and the module docs). The token-type ids are all
/// zero (`type_vocab_size = 1`), so the token-type contribution is the single
/// row-0 of the token-type table broadcast across the batch.
#[cfg(feature = "clap")]
struct ClapTextEmbeddings {
  /// `word_embeddings` table `(vocab, hidden)` — quantize-aware (a token lookup
  /// is an axis-0 row gather, which the dequantize-gather handles for a
  /// quantized table).
  word_embeddings: MaybeQuantizedEmbedding,
  /// `position_embeddings` table `(max_position_embeddings, hidden)`,
  /// materialized dense at load (dequantized once if quantized): the position
  /// lookup gathers arbitrary rows (the offset ids), which a packed table cannot
  /// serve directly, so the dense table keeps the gather path simple.
  position_embeddings: Array,
  /// `token_type_embeddings` table `(type_vocab_size, hidden)`, materialized
  /// dense (only row 0 is ever read — token_type_ids are all zero).
  token_type_embeddings: Array,
  /// `LayerNorm` over the summed embeddings (post-norm, the BERT convention).
  layer_norm: LayerNorm,
  /// `pad_token_id` (`1`) — the `padding_idx` for the position-id offset.
  pad_token_id: i32,
  hidden: i32,
}

#[cfg(feature = "clap")]
impl ClapTextEmbeddings {
  /// Build from the `embeddings.*` sub-tree of the (sanitized) weight map:
  /// `embeddings.word_embeddings.weight` `(vocab, hidden)`,
  /// `embeddings.position_embeddings.weight`
  /// `(max_position_embeddings, hidden)`,
  /// `embeddings.token_type_embeddings.weight` `(type_vocab_size, hidden)`, and
  /// `embeddings.LayerNorm.{weight,bias}` `(hidden,)`. Every table's logical
  /// shape is pinned to the config.
  fn from_weights(
    config: &ClapConfig,
    weights: &mut HashMap<String, Array>,
    eps: f32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = config.text_config.hidden_size;
    let vocab = config.text_config.vocab_size;
    let max_pos = config.text_config.max_position_embeddings;
    let type_vocab = config.text_config.type_vocab_size;
    let pad_token_id = config.text_config.pad_token_id;

    // Word embedding — quantize-aware (an axis-0 row gather; dequantize-gather
    // for a quantized table). Pin the logical `(vocab, hidden)` for both arms.
    let word_embeddings = MaybeQuantizedEmbedding::from_weights(
      weights,
      "embeddings.word_embeddings",
      resolve_quant(quant, "embeddings.word_embeddings"),
    )?;
    expect_logical_shape(
      &word_embeddings,
      "embeddings.word_embeddings.weight",
      "clap RoBERTa word-embedding table (vocab, hidden)",
      vocab,
      hidden,
    )?;

    // Position embedding — materialized dense (dequantized once if quantized):
    // the lookup gathers arbitrary offset rows a packed table cannot serve. Pin
    // the materialized table to `(max_position_embeddings, hidden)`.
    let position_embeddings = {
      let pos = MaybeQuantizedEmbedding::from_weights(
        weights,
        "embeddings.position_embeddings",
        resolve_quant(quant, "embeddings.position_embeddings"),
      )?;
      let table = pos.dense_table(None)?;
      expect_shape(
        &table,
        "embeddings.position_embeddings.weight",
        "clap RoBERTa position-embedding table (max_position_embeddings, hidden)",
        &[max_pos, hidden],
      )?;
      table
    };

    // Token-type embedding — materialized dense (`type_vocab_size` rows, only
    // row 0 is read). Pin to `(type_vocab_size, hidden)`.
    let token_type_embeddings = {
      let tt = MaybeQuantizedEmbedding::from_weights(
        weights,
        "embeddings.token_type_embeddings",
        resolve_quant(quant, "embeddings.token_type_embeddings"),
      )?;
      let table = tt.dense_table(None)?;
      expect_shape(
        &table,
        "embeddings.token_type_embeddings.weight",
        "clap RoBERTa token-type-embedding table (type_vocab_size, hidden)",
        &[type_vocab, hidden],
      )?;
      table
    };

    let layer_norm = build_layer_norm(weights, "embeddings.LayerNorm", hidden, eps)?;

    Ok(Self {
      word_embeddings,
      position_embeddings,
      token_type_embeddings,
      layer_norm,
      pad_token_id,
      hidden,
    })
  }

  /// `(B, L) i32 ids → (B, L, hidden)` summed + post-norm embeddings.
  ///
  /// `word(ids) + position(position_ids_from_ids(ids)) + token_type(0)`, then
  /// `LayerNorm`. The position-id offset is derived from `ids` (HF
  /// `create_position_ids_from_input_ids`); the token-type contribution is the
  /// single row-0 broadcast across `(B, L)` (token_type_ids are all zero).
  fn forward(&self, input_ids: &Array) -> Result<Array> {
    let shape = input_ids.shape();
    let seq = dim_i32(&shape, 1, "clap RoBERTa embeddings: seq_len")?;

    // word(ids): (B, L) → (B, L, hidden) via axis-0 gather.
    let words = self.word_embeddings.gather(input_ids)?;

    // RoBERTa position ids with the padding_idx offset, gathered from the
    // position table → (B, L, hidden). Cast the gathered position rows back to
    // the word-embedding dtype so an f16/bf16 checkpoint is not promoted to f32.
    let position_ids = position_ids_from_ids(input_ids, self.pad_token_id)?;
    let positions = ops::indexing::take_axis(&self.position_embeddings, &position_ids, 0)?;
    let positions = cast_like(&positions, &words)?;

    // token_type row 0 → (1, hidden), broadcast-added across (B, L, hidden).
    let token_type = self.token_type_row0(seq)?;
    let token_type = cast_like(&token_type, &words)?;

    let summed = words.add(&positions)?.add(&token_type)?;
    self.layer_norm.forward(&summed)
  }

  /// Row 0 of the token-type table reshaped to `(1, 1, hidden)` for a
  /// broadcast-add across `(B, L, hidden)` (token_type_ids are all `0`).
  /// `seq` is accepted for symmetry / a future per-position token-type path but
  /// is not needed — row 0 broadcasts over both batch and sequence.
  fn token_type_row0(&self, _seq: i32) -> Result<Array> {
    let lo = [0i32, 0];
    let hi = [1, self.hidden];
    let strides = [1i32, 1];
    let row = ops::indexing::slice(&self.token_type_embeddings, &lo, &hi, &strides)?; // (1, hidden)
    // (1, hidden) → (1, 1, hidden) so it broadcasts over the (B, L) axes.
    ops::shape::expand_dims_axes(&row, &[0])
  }
}

// ════════════════════════════ RobertaEncoder ═══════════════════════════════

/// The RoBERTa encoder stack: `num_hidden_layers` BERT post-norm layers (HF
/// `RobertaEncoder`), each the private [`RobertaLayer`].
#[cfg(feature = "clap")]
struct RobertaEncoder {
  layers: Vec<RobertaLayer>,
}

#[cfg(feature = "clap")]
impl RobertaEncoder {
  /// Build the `num_hidden_layers` layers from `encoder.layer.{i}.*`.
  fn from_weights(
    config: &ClapConfig,
    weights: &mut HashMap<String, Array>,
    dims: LayerDims,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let n = config.text_config.num_hidden_layers;
    // `num_hidden_layers` is required positive in `validate`, but reserve
    // fallibly so a heavyweight per-layer `Vec` the allocator cannot satisfy is a
    // recoverable [`Error::AllocFailure`] rather than `with_capacity`'s abort.
    let mut layers: Vec<RobertaLayer> = Vec::new();
    reserve_or_error(&mut layers, "clap RobertaLayer", n.max(0) as usize)?;
    for i in 0..n {
      layers.push(RobertaLayer::from_weights(
        weights, "encoder", i, dims, quant,
      )?);
    }
    Ok(Self { layers })
  }

  /// Run every layer over `(B, L, hidden)` with the additive key `mask`.
  fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let mut h = x.try_clone()?;
    for layer in &self.layers {
      h = layer.forward(&h, mask)?;
    }
    Ok(h)
  }

  /// `true` if every layer's projections loaded quantized (test-only). An empty
  /// stack vacuously returns `true`.
  #[cfg(test)]
  fn all_quantized(&self) -> bool {
    self.layers.iter().all(|l| l.all_quantized())
  }
}

// ════════════════════════════ ClapProjectionLayer ══════════════════════════

/// The CLAP text projection (HF `ClapProjectionLayer`):
/// `linear2(relu(linear1(x)))`, with biased `Linear(hidden → projection_dim)`
/// then `Linear(projection_dim → projection_dim)` and a **ReLU** between (CLAP
/// uses ReLU, not the towers' GELU — `projection_hidden_act = "relu"`).
#[cfg(feature = "clap")]
struct ClapProjectionLayer {
  linear1: super::shared::QuantLinear,
  linear2: super::shared::QuantLinear,
}

#[cfg(feature = "clap")]
impl ClapProjectionLayer {
  /// Build from `text_projection.linear1.*` + `text_projection.linear2.*`,
  /// pinning `linear1` to `(projection_dim, hidden)` and `linear2` to
  /// `(projection_dim, projection_dim)` (both biased).
  fn from_weights(
    prefix: &str,
    weights: &mut HashMap<String, Array>,
    hidden: i32,
    projection_dim: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let linear1 = super::shared::QuantLinear::from_weights(
      weights,
      &format!("{prefix}.linear1"),
      projection_dim,
      hidden,
      true,
      quant,
    )?;
    let linear2 = super::shared::QuantLinear::from_weights(
      weights,
      &format!("{prefix}.linear2"),
      projection_dim,
      projection_dim,
      true,
      quant,
    )?;
    Ok(Self { linear1, linear2 })
  }

  /// `linear2(relu(linear1(x)))`.
  fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.linear1.forward(x)?;
    let h = relu(&h)?;
    self.linear2.forward(&h)
  }

  /// `true` if both projection layers loaded quantized (test-only).
  #[cfg(test)]
  fn all_quantized(&self) -> bool {
    self.linear1.is_quantized() && self.linear2.is_quantized()
  }
}

// ═══════════════════════════════ ClapTextModel ═════════════════════════════

/// The CLAP RoBERTa text tower (`ClapTextModel`) + the CLAP text projection.
///
/// Maps a `(batch, seq_len)` token-id batch (and its `(batch, seq_len)` `{0,1}`
/// attention mask) to the L2-normalized `(batch, 512)` text embedding in the
/// shared contrastive space:
///
/// 1. `ClapTextEmbeddings` — word + position (offset) + token_type + LayerNorm.
/// 2. `RobertaEncoder` — `num_hidden_layers` post-norm layers with the additive
///    padding mask.
/// 3. CLS pooling — the first sequence position of the last hidden state.
/// 4. `ClapProjectionLayer` — `linear1 → ReLU → linear2` → L2-normalize.
///
/// Built via [`from_weights`](Self::from_weights) /
/// [`from_weights_quantized`](Self::from_weights_quantized). It implements the
/// golden object-safe [`TextEmbedder`] (owning its tokenization + its forward →
/// pool → project → normalize), so the generic
/// [`crate::embeddings::encode()`] pipeline drives it; the phase-4 `ClapModel`
/// assembly wraps it as the text tower.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub struct ClapTextModel {
  embeddings: ClapTextEmbeddings,
  encoder: RobertaEncoder,
  text_projection: ClapProjectionLayer,
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl ClapTextModel {
  /// Build the text tower from a validated [`ClapConfig`] and the (sanitized)
  /// weight map whose keys follow HF's `ClapTextModel` + projection tree:
  /// `embeddings.{word,position,token_type}_embeddings.weight`,
  /// `embeddings.LayerNorm.{weight,bias}`,
  /// `encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}`,
  /// `encoder.layer.{i}.attention.output.{dense.{weight,bias},LayerNorm.{weight,bias}}`,
  /// `encoder.layer.{i}.{intermediate.dense,output.dense}.{weight,bias}`,
  /// `encoder.layer.{i}.output.LayerNorm.{weight,bias}`, and
  /// `text_projection.{linear1,linear2}.{weight,bias}`.
  pub fn from_weights(config: &ClapConfig, weights: &mut HashMap<String, Array>) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build the text tower with an optional parsed quantization config — the
  /// quantization-aware analogue of [`from_weights`](Self::from_weights).
  ///
  /// Each `nn.Linear` (the attention q/k/v/output, the intermediate / output
  /// dense, the two projection layers) and each `nn.Embedding` (word / position /
  /// token_type) auto-picks the dense or quantized variant per layer by its
  /// `<prefix>.scales` sibling.
  pub fn from_weights_quantized(
    config: &ClapConfig,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: `from_weights*` is public, so a caller may build
    // from a directly-constructed (unvalidated) config. This bounds
    // `num_hidden_layers` (and every dim, and pins `pad_token_id = 1`) before the
    // per-layer reservation/loop.
    config.validate()?;
    let hidden = config.text_config.hidden_size;
    let inter = config.text_config.intermediate_size;
    let num_heads = config.text_config.num_attention_heads;
    let eps = config.text_config.layer_norm_eps as f32;
    let projection_dim = config.projection_dim;

    // Per-layer shape constants (validates num_heads positive + divides hidden,
    // and computes the head split / SDPA scale once).
    let dims = LayerDims::new(hidden, inter, num_heads, eps)?;

    let embeddings = ClapTextEmbeddings::from_weights(config, weights, eps, quant)?;
    let encoder = RobertaEncoder::from_weights(config, weights, dims, quant)?;
    let text_projection =
      ClapProjectionLayer::from_weights("text_projection", weights, hidden, projection_dim, quant)?;

    Ok(Self {
      embeddings,
      encoder,
      text_projection,
    })
  }

  /// Forward a `(batch, seq_len)` i32 token-id batch (+ its `(batch, seq_len)`
  /// `{0,1}` attention mask) to the L2-normalized `(batch, 512)` text embedding.
  ///
  /// Mirrors `ClapModel.get_text_features`: embeddings → RoBERTa encoder (with
  /// the additive padding mask) → CLS (`last_hidden_state[:, 0, :]`) → text
  /// projection → L2-normalize. `input_ids` is pinned to exactly rank-2 (the
  /// public [`embed_text`](Self::embed_text) accepts an untrusted array; the
  /// embedding gather + CLS slice are only defined for a rank-2 batch).
  pub fn encode_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
    let shape = input_ids.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "clap RoBERTa: input_ids must be rank-2 (batch, seq_len)",
        shape.len() as u32,
        shape,
      )));
    }
    let seq = dim_i32(&shape, 1, "clap RoBERTa: seq_len")?;
    if seq < 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap RoBERTa: seq_len",
        "must be a positive sequence length (>= 1)",
        format_smolstr!("{seq}"),
      )));
    }

    // HF RoBERTa contract: the attention mask is element-wise over the tokens, so
    // it must share the `(batch, seq_len)` shape of `input_ids`. The fused SDPA
    // accepts broadcastable masks, so a `(1, L)` or `(B, 1)` mask would silently
    // apply the wrong padding pattern instead of failing — pin it to an exact match.
    let mask_shape = attention_mask.shape();
    if mask_shape != shape {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "clap RoBERTa: attention_mask shape must equal input_ids (batch, seq_len)",
        shape,
        mask_shape,
      )));
    }

    // RoBERTa input embeddings (word + position offset + token_type + LN).
    let hidden = self.embeddings.forward(input_ids)?; // (B, L, hidden)

    // Additive padding mask in the activation dtype so the fused SDPA sees a
    // matching-dtype mask (dtype preservation: read it off the hidden states).
    let dtype = hidden.dtype()?;
    let mask = build_additive_mask(attention_mask, dtype)?; // (B, 1, 1, L)

    // RoBERTa encoder → (B, L, hidden).
    let hidden = self.encoder.forward(&hidden, Mask::Array(&mask))?;

    // CLS pooling: last_hidden_state[:, 0, :] → (B, hidden).
    let cls = pool_cls(&hidden)?;

    // Text projection (linear1 → ReLU → linear2) → (B, 512).
    let projected = self.text_projection.forward(&cls)?;

    // L2-normalize into the shared contrastive space (textclap normalizes the
    // tower output itself — `TEXT_OUTPUT_IS_UNIT_NORM = false`).
    l2_normalize(&projected)
  }

  /// `true` if every encoder projection loaded quantized (test-only).
  #[cfg(test)]
  pub(crate) fn all_projections_quantized(&self) -> bool {
    self.encoder.all_quantized() && self.text_projection.all_quantized()
  }

  /// `true` if the word embedding loaded from a quantized checkpoint (test-only).
  #[cfg(test)]
  pub(crate) fn word_embedding_is_quantized(&self) -> bool {
    self.embeddings.word_embeddings.is_quantized()
  }
}

/// CLAP's text tower owns the RoBERTa tokenization + the forward → CLS → project
/// → normalize. The mask is mask-aware (RoBERTa masks pad keys), so the model
/// declares a [`Padding::DynamicRightPad`] (a batch-invariant right-pad with a
/// proper attention mask — UNLIKE SigLIP2's sticky-EOS `FixedLength`).
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl TextEmbedder for ClapTextModel {
  fn text_encoding(&self) -> TextEncoding {
    TextEncoding::new(
      // RoBERTa adds its `<s>` / `</s>` special tokens.
      true,
      // Cap at `max_position_embeddings - 2 = 512` real tokens (textclap
      // `TEXT_MAX_TOKENS = 512`); tokens beyond would index past the position
      // table. The DynamicRightPad scheme carries no intrinsic cap, so this
      // explicit cap is the effective truncation length.
      Some(TEXT_MAX_TOKENS),
      Padding::DynamicRightPad {
        pad_token_id: PAD_TOKEN_ID,
      },
    )
  }

  fn embed_text(&self, input_ids: &Array, attention_mask: &Array) -> Result<Embedding> {
    Ok(Embedding::new(self.encode_text(input_ids, attention_mask)?))
  }
}

// ═══════════════════════════════ free functions ════════════════════════════

/// The RoBERTa `padding_idx` position-id offset (HF
/// `create_position_ids_from_input_ids`):
///
/// ```text
/// mask          = (input_ids != pad_token_id)           # {0, 1}
/// position_ids  = cumsum(mask, axis=1) * mask + pad_token_id
/// ```
///
/// So a real token at running index `k` (1-based over the non-pad tokens of its
/// row) gets position `pad_token_id + k`, and a pad token gets `pad_token_id`.
/// With `pad_token_id = 1`, ids `[0, 5, 9, 1, 1]` → mask `[1, 1, 1, 0, 0]` →
/// cumsum `[1, 2, 3, 3, 3]` → `* mask` `[1, 2, 3, 0, 0]` → `+ 1`
/// `[2, 3, 4, 1, 1]`. Returns a `(batch, seq_len)` i32 index array for the
/// position-table gather.
///
/// The whole computation runs in **i32** (the index dtype): the `!= pad` mask is
/// reduced to an i32 `{0, 1}` and the cumsum is exact, so there is no float
/// rounding to corrupt a position index.
#[cfg(feature = "clap")]
fn position_ids_from_ids(input_ids: &Array, pad_token_id: i32) -> Result<Array> {
  // `mask = (input_ids != pad) as i32` — the non-pad indicator.
  let pad = Array::from_slice::<i32>(&[pad_token_id], &(1usize,))?;
  let not_pad = ops::comparison::not_equal(input_ids, &pad)?; // bool
  let mask = ops::misc::astype(&not_pad, Dtype::I32)?; // {0, 1} i32

  // `incremental = cumsum(mask, axis=1) * mask` (inclusive forward scan).
  let cumsum = ops::misc::cumsum(&mask, 1, false, true)?;
  let incremental = cumsum.multiply(&mask)?;

  // `position_ids = incremental + pad_token_id`.
  let pad_add = Array::from_slice::<i32>(&[pad_token_id], &(1usize,))?;
  incremental.add(&pad_add)
}

/// Build the `(batch, 1, 1, seq_len)` additive attention mask from a
/// `(batch, seq_len)` `{0, 1}` padding mask: `0.0` where a token is attended,
/// `-inf` where it is padding — broadcastable to the SDPA `[B, N_q, T_q, T_kv]`
/// key axis (RoBERTa masks pad **keys**; every query attends to every real key,
/// bidirectional).
///
/// The result is cast to `dtype` (the activation dtype) so the fused SDPA sees a
/// matching-dtype additive mask (dtype preservation). Mirrors the
/// [`crate::embeddings::embeddinggemma`] `build_additive_mask` (the
/// `[:, None, None, :]` reshape + the `where(mask != 0, 0.0, -inf)` step).
#[cfg(feature = "clap")]
fn build_additive_mask(attention_mask: &Array, dtype: Dtype) -> Result<Array> {
  let shape = attention_mask.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "clap RoBERTa: attention_mask must be rank-2 (batch, seq_len)",
      shape.len() as u32,
      shape,
    )));
  }
  // (B, S) → (B, 1, 1, S): one broadcastable key-axis mask per batch row.
  let expanded = ops::shape::expand_dims_axes(attention_mask, &[1, 2])?;
  // `where(mask != 0, 0.0, -inf)`. The encode pipeline builds an f32 `{0, 1}`
  // mask; compare against 0 to a boolean, then select 0.0 / -inf.
  let zero = Array::full::<f32>(&(1,), 0.0)?;
  let keep = ops::comparison::not_equal(&expanded, &zero)?; // bool: real tokens
  let additive_zero = Array::full::<f32>(&(1,), 0.0)?;
  let neg_inf = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
  let mask = ops::logical::select(&keep, &additive_zero, &neg_inf)?;
  // Cast to the activation dtype so SDPA sees a matching additive mask dtype.
  ops::misc::astype(&mask, dtype)
}

/// CLS pooling: take the first sequence position of `(B, L, hidden)` →
/// `(B, hidden)` (`last_hidden_state[:, 0, :]`, the CLAP text feature).
#[cfg(feature = "clap")]
fn pool_cls(hidden: &Array) -> Result<Array> {
  // index 0 along the sequence axis (axis 1) → (B, 1, hidden), then squeeze.
  let idx0 = Array::from_slice::<i32>(&[0], &(1usize,))?;
  let first = ops::indexing::take_axis(hidden, &idx0, 1)?; // (B, 1, hidden)
  ops::shape::squeeze_axes(&first, &[1]) // (B, hidden)
}

/// ReLU (`max(x, 0)`), the CLAP projection activation
/// (`projection_hidden_act = "relu"`). Built with a dtype-matched rank-0 `0`
/// constant so an f16/bf16 activation is not promoted to f32.
#[cfg(feature = "clap")]
fn relu(x: &Array) -> Result<Array> {
  let zero = cast_like(&Array::full::<f32>(&[0i32; 0], 0.0)?, x)?;
  ops::arithmetic::maximum(x, &zero)
}

/// Cast `a` to `like`'s dtype (a no-op when they already match) — the uniform
/// stand-in for MLX weak-scalar / `astype(x.dtype)` semantics, so a tensor built
/// in f32 (a position-row gather, a scalar floor) that meets an f16/bf16
/// activation is cast back rather than promoting the activation to f32.
#[cfg(feature = "clap")]
fn cast_like(a: &Array, like: &Array) -> Result<Array> {
  ops::misc::astype(a, like.dtype()?)
}

#[cfg(all(test, feature = "clap"))]
mod tests;
