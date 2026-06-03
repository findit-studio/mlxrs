//! Qwen3 — the dense Qwen3 text transformer.
//!
//! Faithful 1:1 port of `mlx-lm/mlx_lm/models/qwen3.py` (the dense LM; the MoE
//! variant `qwen3_moe.py` is out of scope). Qwen3 is a standard pre-norm
//! decoder transformer with three Qwen3-specific touches:
//!
//! - **Grouped-query attention with per-head Q/K-norm** (`qwen3.py:32-89`) —
//!   the `{q,k,v}_proj` outputs are reshaped to per-head `(B, L, n_heads,
//!   head_dim)`, then a per-head
//!   [`RMSNorm`](crate::lm::nn::norm::RMSNorm) over `head_dim` is applied to
//!   the queries and keys (**before** RoPE), RoPE with `traditional=false` and
//!   `rope_theta` is applied at the cache offset, the running K/V is appended
//!   via a [`StandardKvCache`](crate::lm::cache::StandardKvCache), and the
//!   fused
//!   [`scaled_dot_product_attention`](crate::lm::nn::attention::scaled_dot_product_attention)
//!   handles the GQA repeat against a causal mask. Unlike LFM2, `head_dim` is
//!   an explicit config field (not `hidden_size / n_heads`), so the query
//!   projection is `Linear(hidden, n_heads * head_dim)`.
//! - **SwiGLU MLP** (`qwen3.py:92-100`) — `down_proj(silu(gate_proj(x)) *
//!   up_proj(x))`.
//! - **Tied / untied LM head** (`qwen3.py:163-188`) — when
//!   `tie_word_embeddings` is set the output projection reuses the embedding
//!   table (`embed_tokens.as_linear`, i.e. `out @ embed_tokens.T`); otherwise a
//!   separate `lm_head` linear is used.
//!
//! Each decoder block is pre-norm: `h = x + attention(input_layernorm(x))`
//! then `out = h + mlp(post_attention_layernorm(h))` (`qwen3.py:103-126`). The
//! decoder embeds the tokens, runs the blocks with a single shared causal mask
//! sourced from the first layer's cache, and applies the final
//! [`RMSNorm`](crate::lm::nn::norm::RMSNorm) (`qwen3.py:129-160`).
//!
//! The per-layer cache is homogeneous: a
//! [`StandardKvCache`](crate::lm::cache::StandardKvCache) for every layer.
//! [`Qwen3::make_cache`](crate::lm::models::qwen3::Qwen3::make_cache) builds
//! it; the public forward
//! ([`Qwen3`](crate::lm::models::qwen3::Qwen3)'s
//! [`Model::forward`](crate::lm::model::Model::forward)) takes token ids plus
//! that cache and returns `[B, L, vocab]` logits.
//!
//! The head-less decoder is its own public type
//! [`Qwen3Model`](crate::lm::models::qwen3::Qwen3Model) (token embedding +
//! blocks + final norm, returning `(B, L, hidden)` hidden states with **no**
//! output projection), mirroring the reference's `TextModel`.
//! [`Qwen3`](crate::lm::models::qwen3::Qwen3) is the thin wrapper that adds the
//! (tied or untied) vocab head. This dense decoder is standard-RoPE only; the
//! Qwen3-ASR forced aligner uses its own MRoPE text decoder
//! (`crate::audio::stt::models::qwen3_asr`), because the released Qwen3-ASR
//! `text_config` carries a non-null MRoPE `rope_scaling` the dense Qwen3 config
//! rejects.

mod config;
mod linear;

use std::collections::HashMap;

pub use config::Qwen3Config;
use linear::{Embedding, Linear, SCALES_SUFFIX, take_shaped};
use smol_str::format_smolstr;

use crate::{
  Dtype,
  array::Array,
  error::{
    Error, LengthMismatchPayload, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload,
  },
  lm::{
    cache::{KvCache, MaskMode, StandardKvCache},
    model::Model as LmModel,
    nn::{
      activations::swiglu,
      attention::{Mask, scaled_dot_product_attention},
      norm::RMSNorm,
      rope::Rope,
    },
    quant::PerLayerQuantization,
  },
  model_validation::reserve_or_error,
  ops::shape::{reshape, transpose_axes},
};

// ───────────────────────── attention ─────────────────────────

/// Grouped-query attention with per-head Q/K-norm (`qwen3.py:32-89`).
#[derive(Debug)]
struct Attention {
  n_heads: i32,
  n_kv_heads: i32,
  /// Per-head dimension (Qwen3's explicit `head_dim`). Used for the per-head
  /// reshapes and the `o_proj`-input width; the crate's `reshape` requires
  /// concrete dims (no `-1` inference), so this is carried explicitly.
  head_dim: i32,
  scale: f32,
  q_proj: Linear,
  k_proj: Linear,
  v_proj: Linear,
  o_proj: Linear,
  q_norm: RMSNorm,
  k_norm: RMSNorm,
  rope: Rope,
}

impl Attention {
  /// `queries/keys/values = {q,k,v}_proj(x)`, per-head reshape `(B, L, n,
  /// head_dim)`, q/k RMSNorm over `head_dim` (**before** RoPE), RoPE at the
  /// cache offset, `cache.update`, then the fused
  /// [`scaled_dot_product_attention`] and `o_proj`.
  ///
  /// `cache` is the layer's [`StandardKvCache`]; `mask` is the attention mask
  /// mode for this forward pass.
  fn forward(&self, x: &Array, mask: &MaskMode, cache: &mut StandardKvCache) -> Result<Array> {
    let shape = x.shape();
    let (b, l) = (shape[0] as i32, shape[1] as i32);

    let queries = self.q_proj.forward(x)?;
    let keys = self.k_proj.forward(x)?;
    let values = self.v_proj.forward(x)?;

    // Per-head reshape `(B, L, n, head_dim)`, q/k RMSNorm over the last axis
    // (head_dim), then transpose to `(B, n, L, head_dim)`. The norm runs on the
    // `(B, L, n, head_dim)` layout — RMSNorm normalizes the last axis, which is
    // `head_dim`, matching `qwen3.py:69-74`. `head_dim` is spelled out (the
    // crate's `reshape` rejects a `-1` infer dim).
    let hd = self.head_dim;
    let queries = reshape(&queries, &[b, l, self.n_heads, hd])?;
    let queries = self.q_norm.forward(&queries)?;
    let queries = transpose_axes(&queries, &[0, 2, 1, 3])?;

    let keys = reshape(&keys, &[b, l, self.n_kv_heads, hd])?;
    let keys = self.k_norm.forward(&keys)?;
    let keys = transpose_axes(&keys, &[0, 2, 1, 3])?;

    let values = reshape(&values, &[b, l, self.n_kv_heads, hd])?;
    let values = transpose_axes(&values, &[0, 2, 1, 3])?;

    // RoPE at the cache offset, then append+fetch the running K/V.
    let offset = cache.offset() as i32;
    let queries = self.rope.apply(&queries, offset)?;
    let keys = self.rope.apply(&keys, offset)?;
    let (keys, values) = cache.update(&keys, &values)?;

    let attn_mask = mask_mode_to_mask(mask);
    let output = scaled_dot_product_attention(&queries, &keys, &values, self.scale, attn_mask)?;
    // `(B, n, L, head_dim)` -> `(B, L, n*head_dim)`.
    let output = transpose_axes(&output, &[0, 2, 1, 3])?;
    let output = reshape(&output, &[b, l, self.n_heads * hd])?;
    self.o_proj.forward(&output)
  }
}

/// Map a [`MaskMode`] to the attention [`Mask`] selector. `Array` borrows the
/// mode's owned array for the call's duration.
fn mask_mode_to_mask(mode: &MaskMode) -> Mask<'_> {
  match mode {
    MaskMode::None => Mask::None,
    MaskMode::Causal => Mask::Causal,
    MaskMode::Array(a) => Mask::Array(a),
  }
}

// ───────────────────────── MLP ─────────────────────────

/// Dense SwiGLU feed-forward (`qwen3.py:92-100`):
/// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug)]
struct Mlp {
  gate_proj: Linear,
  up_proj: Linear,
  down_proj: Linear,
}

impl Mlp {
  fn forward(&self, x: &Array) -> Result<Array> {
    let gate = self.gate_proj.forward(x)?;
    let up = self.up_proj.forward(x)?;
    let act = swiglu(&gate, &up)?;
    self.down_proj.forward(&act)
  }
}

// ───────────────────────── decoder block ─────────────────────────

/// A pre-norm Qwen3 transformer block (`qwen3.py:103-126`).
#[derive(Debug)]
struct TransformerBlock {
  self_attn: Attention,
  mlp: Mlp,
  input_layernorm: RMSNorm,
  post_attention_layernorm: RMSNorm,
}

impl TransformerBlock {
  /// `h = x + self_attn(input_layernorm(x), mask, cache)` then
  /// `out = h + mlp(post_attention_layernorm(h))`.
  fn forward(&self, x: &Array, mask: &MaskMode, cache: &mut StandardKvCache) -> Result<Array> {
    let r = self
      .self_attn
      .forward(&self.input_layernorm.forward(x)?, mask, cache)?;
    let hidden = x.add(&r)?;
    let ffn = self
      .mlp
      .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
    hidden.add(&ffn)
  }
}

// ───────────────────────── model ─────────────────────────

/// The Qwen3 head-less decoder stack (`Qwen3Model`, `qwen3.py:129-160`): token
/// embedding, the per-layer transformer blocks with a shared causal mask, and
/// the final RMSNorm. Its forward returns the **normalized hidden states**
/// `norm(h)` of shape `(B, L, hidden)` — **no** output projection.
///
/// This is the standard-RoPE decoder under the causal LM [`Qwen3`] (which
/// applies a vocab head over these hidden states), mirroring the reference's
/// `TextModel`. The Qwen3-ASR forced aligner uses its own MRoPE variant
/// (`crate::audio::stt::models::qwen3_asr`) rather than this dense decoder.
#[derive(Debug)]
pub struct Qwen3Model {
  /// `(vocab, hidden)` token-embedding table (quantize-aware); also the tied
  /// output head when reused by [`Qwen3`] (via [`Embedding::as_linear`]).
  embed_tokens: Embedding,
  /// The hidden width (`config.hidden_size`), carried explicitly so the input
  /// embeddings' width can be validated against it. The quantized embedding's
  /// packed weight has axis-1 `hidden * bits / 32` (not `hidden`), so the width
  /// cannot be recovered from the embedding table on a quantized checkpoint.
  hidden_size: i32,
  layers: Vec<TransformerBlock>,
  norm: RMSNorm,
}

impl Qwen3Model {
  /// Embed `tokens` (`(B, L)` integer ids) to `(B, L, hidden)` via the token
  /// embedding table — the row-gather front of the decoder, exposed so a
  /// caller can splice input features into chosen positions of the embeddings
  /// before running [`forward_hidden`](Self::forward_hidden). No implicit eval of
  /// a *lazy graph* — the returned [`Array`] is lazy; reading the (data-backed)
  /// `tokens` to range-check them is an explicit materialization of an input, not
  /// a hidden eval.
  ///
  /// Every id must be a valid embedding row index in `[0, row_count)` (the
  /// embedding table's leading dimension). MLX `take` (the gather) does **not**
  /// bound-check its indices — a negative id (read as `id + row_count`, which for
  /// `id < -row_count` stays negative) or an id `>= row_count` is an
  /// out-of-bounds embedding-table read, i.e. UB — so each id is range-checked
  /// here with a typed [`Error::OutOfRange`] before the gather. The values are
  /// read in their native integer dtype (widened to `i64`, so a `u32`/`i32` id
  /// tensor is validated without truncation, mirroring the aligner's
  /// `input_ids` guard).
  pub fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
    let row_count = self.embed_tokens.row_count();
    check_token_ids_in_rows(tokens, row_count, "Qwen3Model::embed_tokens: token id")?;
    self.embed_tokens.forward(tokens)
  }

  /// The number of decoder layers — also the per-layer cache cardinality
  /// [`make_cache`](Self::make_cache) builds.
  #[inline(always)]
  pub fn num_layers(&self) -> usize {
    self.layers.len()
  }

  /// `true` if the token embedding loaded from a quantized checkpoint
  /// (test-only introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn embedding_is_quantized(&self) -> bool {
    self.embed_tokens.is_quantized()
  }

  /// `true` if every decoder attention + MLP projection loaded quantized
  /// (test-only introspection).
  #[cfg(test)]
  pub(crate) fn all_projections_quantized(&self) -> bool {
    self.layers.iter().all(|l| {
      let a = &l.self_attn;
      let m = &l.mlp;
      a.q_proj.is_quantized()
        && a.k_proj.is_quantized()
        && a.v_proj.is_quantized()
        && a.o_proj.is_quantized()
        && m.gate_proj.is_quantized()
        && m.up_proj.is_quantized()
        && m.down_proj.is_quantized()
    })
  }

  /// Build the homogeneous per-layer cache: a [`StandardKvCache`] for every
  /// layer, in layer order (`make_prompt_cache` for a full-attention model).
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    (0..self.layers.len())
      .map(|_| -> Box<dyn KvCache> { Box::new(StandardKvCache::new()) })
      .collect()
  }

  /// Run the decoder over precomputed `h` (`(B, L, hidden)`), updating each
  /// layer's cache in place; returns the final-normed hidden states.
  ///
  /// The per-layer cache must hold exactly one entry per decoder layer (as
  /// [`make_cache`](Self::make_cache) builds it). A mismatched count is a
  /// recoverable [`Error::LengthMismatch`] rather than an out-of-bounds index
  /// panic on the mask-source lookup / a silently truncated `zip` over the
  /// layers.
  ///
  /// Each per-layer cache is the model's fixed [`StandardKvCache`]; per the
  /// crate's per-layer fast-path convention it is downcast once before the
  /// per-layer loop (the only `Box<dyn KvCache>` vtable hop), then every
  /// `offset` / `update` / `make_mask` dispatches statically.
  pub fn forward_hidden(&self, h: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    if cache.len() != self.layers.len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "Qwen3Model::forward: per-layer cache count vs decoder layers",
        self.layers.len(),
        cache.len(),
      )));
    }
    // `forward_hidden` is public (and reachable from `Qwen3::forward_embeddings`,
    // whose `supports_input_embeddings` is `true`), so a caller can hand in an
    // arbitrary-rank `Array`. Require rank-3 `[batch, seq, hidden]` and validate
    // the hidden width against the token-embedding table BEFORE reading
    // `shape[1]`: a rank-0 / rank-1 input would otherwise panic on the `shape[1]`
    // index, and a rank-2 (or wrong-width rank-3) input would proceed with a
    // misinterpreted shape into a downstream matmul.
    let shape = h.shape();
    if shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "Qwen3Model::forward: hidden states must be rank-3 [batch, seq, hidden]",
        shape.len() as u32,
        shape,
      )));
    }
    // The hidden width (`config.hidden_size`) is the width every attention/MLP
    // projection expects; reject a mismatched hidden axis before the layers index
    // it. Carried explicitly (not read off the embedding table) because a
    // quantized embedding's packed weight has axis-1 `hidden * bits / 32`.
    // `hidden_size` is a `validate`d positive `i32`, so the `usize` cast is exact.
    let hidden = self.hidden_size as usize;
    if shape[2] != hidden {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "Qwen3Model::forward: hidden-states width vs model hidden width",
        vec![hidden],
        vec![shape[2]],
      )));
    }
    let n = shape[1];

    // Build the single shared causal mask once from the first layer's cache
    // (mlx-lm `create_attention_mask(h, cache[0])`). An empty model has no
    // layers and thus needs no mask.
    let mask = match cache.first_mut() {
      Some(c) => downcast_cache(c.as_mut())?.make_mask(n, None, false)?,
      None => MaskMode::None,
    };

    let mut h = h.try_clone()?;
    for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
      let kv = downcast_cache(c.as_mut())?;
      h = layer.forward(&h, &mask, kv)?;
    }
    self.norm.forward(&h)
  }

  /// Embed `tokens` (`(B, L)` integer ids) via the embedding table, then run
  /// the decoder — the convenience composition of [`embed_tokens`] and
  /// [`forward_hidden`].
  ///
  /// [`embed_tokens`]: Self::embed_tokens
  /// [`forward_hidden`]: Self::forward_hidden
  pub fn forward_tokens(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let h = self.embed_tokens(tokens)?;
    self.forward_hidden(&h, cache)
  }

  /// Build the head-less decoder from a parsed [`Qwen3Config`] and a flat name
  /// → [`Array`] weight map, draining the `model.*` keys it consumes:
  /// `model.embed_tokens.weight`, `model.norm.weight`, and per-layer
  /// `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
  /// `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
  /// `model.layers.{i}.self_attn.{q,k}_norm.weight`,
  /// `model.layers.{i}.mlp.{gate,up,down}_proj.weight`. A missing required
  /// weight is an [`Error::MissingKey`]. `config` is [`validate`]d first (it is
  /// public and its fields are public, so this constructor cannot trust a caller
  /// to have done so); the caller builds the output head that sits on top (the LM
  /// vocab head in [`Qwen3::from_weights`]).
  ///
  /// [`validate`]: Qwen3Config::validate
  ///
  /// Every loaded tensor is shape-checked against the config-derived expectation
  /// (the same arithmetic the forward realizes) on load, so a
  /// malformed decoder checkpoint — a wrong-rank or wrong-width projection /
  /// norm — is rejected here with a typed [`Error::LayerKeyed`] rather than
  /// deferring to an opaque MLX `matmul` / `reshape` / `RMSNorm` failure (or, for
  /// a broadcastable mismatch, silently running a different graph). This mirrors
  /// the Qwen3-ASR text decoder
  /// (`crate::audio::stt::models::qwen3_asr`), which validates every decoder
  /// weight the same way.
  ///
  /// A **quantized** checkpoint loads through the same path: each projection /
  /// the token embedding is built quantized via the shared
  /// [`crate::nn::MaybeQuantizedLinear`] / a quantized table when the checkpoint
  /// carries that layer's `.scales` sibling ALONE (the per-layer auto-detect
  /// Whisper / EmbeddingGemma use), with the per-layer `(group_size, bits,
  /// mode)` resolved from `quant`. The quantized path pins the packed triple's
  /// logical shape to the same config-derived extents the dense path checks.
  /// `quant` is `None` for a dense checkpoint; a dense checkpoint (no `.scales`)
  /// loads dense even if `quant` is `Some`, but a layer that DOES carry
  /// `.scales` with no resolvable scheme is a typed `Error::InvariantViolation`,
  /// never reinterpreted as dense.
  pub fn from_weights(
    config: &Qwen3Config,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Qwen3Model> {
    // Validate the (public-field) config BEFORE deriving any head count or
    // projection width from it, or reserving/iterating `num_hidden_layers`. This
    // constructor is public and the fields are public, so a caller can hand in a
    // structurally invalid config (e.g. `num_hidden_layers == 0`, a zero
    // attention- or kv-head count, a non-divisible GQA grouping) even after the
    // parsed config was valid. Without this gate `num_hidden_layers == 0` would
    // load a norm-only decoder skipping every required per-layer weight, and a
    // zero head count would derive zero-width projections that fail later in
    // `reshape` / SDPA rather than at load. `validate` rejects every such value
    // with a typed error here, so the downstream shape derivation below operates
    // only on a validated config.
    config.validate()?;
    let eps = config.rms_norm_eps;
    let head_dim = config.head_dim;

    // Config-derived expected projection widths (the same arithmetic the forward
    // realizes), so every loaded tensor can be shape-checked against them.
    // `validate()` (run just above) bounds `num_attention_heads * head_dim`
    // within the i32 width cap; the kv product is no larger, so neither
    // `saturating_mul` can overflow i32.
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let q_out = config.num_attention_heads.saturating_mul(head_dim);
    let kv_out = config.num_key_value_heads.saturating_mul(head_dim);

    // Validate the embedding table's shape on load: its logical shape must be
    // `(vocab_size, hidden_size)` from `config`. The embedding `take` gather
    // indexes its leading dimension with caller token ids and does not
    // bound-check, so a checkpoint whose table has fewer rows than
    // `config.vocab_size` would let an id in `[rows, vocab_size)` — which
    // `embed_tokens` admits against `config.vocab_size` — reach the gather out of
    // bounds (UB). On the quantized path the packed weight's logical `(vocab,
    // hidden)` is pinned identically. `model.embed_tokens` quantizes alongside
    // the projections (mlx-lm's `class_predicate` covers `nn.Embedding`).
    let embed_tokens = Embedding::from_weights(
      weights,
      "model.embed_tokens",
      config.vocab_size,
      hidden,
      "embed_tokens weight (vocab_size, hidden_size)",
      quant,
    )?;
    let norm = RMSNorm::new(
      take_shaped(
        weights,
        "model.norm.weight",
        "final norm weight (hidden_size)",
        &[hidden],
      )?,
      eps,
    );

    // `num_hidden_layers` is bounded by `MAX_CONFIG_CARDINALITY` in `validate`,
    // but reserve fallibly so even a within-cap heavyweight per-layer `Vec` the
    // allocator cannot satisfy is a recoverable [`Error::AllocFailure`] rather
    // than `with_capacity`'s abort.
    let mut layers: Vec<TransformerBlock> = Vec::new();
    reserve_or_error(
      &mut layers,
      "Qwen3 TransformerBlock",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      let p = format!("model.layers.{i}");
      let q = format!("{p}.self_attn");

      let self_attn = Attention {
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
        q_proj: Linear::from_weights(
          weights,
          &format!("{q}.q_proj"),
          q_out,
          hidden,
          "q_proj weight (n_heads * head_dim, hidden_size)",
          quant,
        )?,
        k_proj: Linear::from_weights(
          weights,
          &format!("{q}.k_proj"),
          kv_out,
          hidden,
          "k_proj weight (n_kv_heads * head_dim, hidden_size)",
          quant,
        )?,
        v_proj: Linear::from_weights(
          weights,
          &format!("{q}.v_proj"),
          kv_out,
          hidden,
          "v_proj weight (n_kv_heads * head_dim, hidden_size)",
          quant,
        )?,
        o_proj: Linear::from_weights(
          weights,
          &format!("{q}.o_proj"),
          hidden,
          q_out,
          "o_proj weight (hidden_size, n_heads * head_dim)",
          quant,
        )?,
        q_norm: RMSNorm::new(
          take_shaped(
            weights,
            &format!("{q}.q_norm.weight"),
            "q_norm weight (head_dim)",
            &[head_dim],
          )?,
          eps,
        ),
        k_norm: RMSNorm::new(
          take_shaped(
            weights,
            &format!("{q}.k_norm.weight"),
            "k_norm weight (head_dim)",
            &[head_dim],
          )?,
          eps,
        ),
        rope: Rope::new(head_dim, false, config.rope_theta, 1.0),
      };

      let mlp = Mlp {
        gate_proj: Linear::from_weights(
          weights,
          &format!("{p}.mlp.gate_proj"),
          inter,
          hidden,
          "mlp gate_proj weight (intermediate_size, hidden_size)",
          quant,
        )?,
        up_proj: Linear::from_weights(
          weights,
          &format!("{p}.mlp.up_proj"),
          inter,
          hidden,
          "mlp up_proj weight (intermediate_size, hidden_size)",
          quant,
        )?,
        down_proj: Linear::from_weights(
          weights,
          &format!("{p}.mlp.down_proj"),
          hidden,
          inter,
          "mlp down_proj weight (hidden_size, intermediate_size)",
          quant,
        )?,
      };

      let input_layernorm = RMSNorm::new(
        take_shaped(
          weights,
          &format!("{p}.input_layernorm.weight"),
          "input_layernorm weight (hidden_size)",
          &[hidden],
        )?,
        eps,
      );
      let post_attention_layernorm = RMSNorm::new(
        take_shaped(
          weights,
          &format!("{p}.post_attention_layernorm.weight"),
          "post_attention_layernorm weight (hidden_size)",
          &[hidden],
        )?,
        eps,
      );

      layers.push(TransformerBlock {
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
      });
    }

    Ok(Qwen3Model {
      embed_tokens,
      hidden_size: hidden,
      layers,
      norm,
    })
  }
}

/// Downcast a per-layer `Box<dyn KvCache>` to the model's fixed
/// [`StandardKvCache`], erroring with a typed [`Error::InvariantViolation`]
/// when the cache kind does not match (the per-layer fast-path convention).
fn downcast_cache(cache: &mut dyn KvCache) -> Result<&mut StandardKvCache> {
  cache
    .as_any_mut()
    .downcast_mut::<StandardKvCache>()
    .ok_or_else(|| {
      Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
        "Qwen3 per-layer cache kind",
        "every Qwen3 layer expects a StandardKvCache",
      ))
    })
}

/// The optional output projection (`qwen3.py:163-188`): either the tied
/// embedding head (`embed_tokens.as_linear`) or a separate `lm_head` linear.
#[derive(Debug)]
enum LmHead {
  /// `tie_word_embeddings = true`: reuse the embedding table as the output
  /// projection (`out @ embed_tokens.T`).
  Tied,
  /// `tie_word_embeddings = false`: a dedicated `(vocab, hidden)` projection.
  Untied(Linear),
}

/// The Qwen3 causal language model (`Model`, `qwen3.py:163-188`): the head-less
/// decoder ([`Qwen3Model`]) plus the (tied or untied) LM head.
#[derive(Debug)]
pub struct Qwen3 {
  config: Qwen3Config,
  model: Qwen3Model,
  lm_head: LmHead,
}

impl Qwen3 {
  /// Read-only view of the parsed configuration.
  pub fn config(&self) -> &Qwen3Config {
    &self.config
  }

  /// Read-only view of the underlying head-less [`Qwen3Model`] decoder.
  #[inline(always)]
  pub fn model(&self) -> &Qwen3Model {
    &self.model
  }

  /// `true` if the untied `lm_head` loaded quantized. `false` for the tied head
  /// (its quantization is the embedding's — see
  /// [`Qwen3Model::embedding_is_quantized`]). Test-only introspection.
  #[cfg(test)]
  pub(crate) fn untied_lm_head_is_quantized(&self) -> bool {
    match &self.lm_head {
      LmHead::Untied(head) => head.is_quantized(),
      LmHead::Tied => false,
    }
  }

  /// Project final hidden states to vocab logits via the configured head.
  ///
  /// Tied (`qwen3.py:180-181`): `embed_tokens.as_linear(out)` = `out @
  /// embed_tokens.T` (dense) or `quantized_matmul` (quantized embedding).
  /// Untied: the dedicated `lm_head` linear.
  fn project_logits(&self, hidden: &Array) -> Result<Array> {
    match &self.lm_head {
      LmHead::Tied => self.model.embed_tokens.as_linear(hidden),
      LmHead::Untied(head) => head.forward(hidden),
    }
  }

  /// Build the homogeneous per-layer cache: a [`StandardKvCache`] for every
  /// layer, in layer order (`make_prompt_cache` for a full-attention model).
  pub fn make_cache(&self) -> Vec<Box<dyn KvCache>> {
    self.model.make_cache()
  }
}

impl LmModel for Qwen3 {
  fn forward(&self, tokens: &Array, cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let hidden = self.model.forward_tokens(tokens, cache)?;
    self.project_logits(&hidden)
  }

  fn forward_embeddings(
    &self,
    embeddings: &Array,
    cache: &mut [Box<dyn KvCache>],
  ) -> Result<Array> {
    let hidden = self.model.forward_hidden(embeddings, cache)?;
    self.project_logits(&hidden)
  }

  fn supports_input_embeddings(&self) -> bool {
    true
  }
}

// ───────────────────────── weight loading ─────────────────────────

/// Reject any token id in `tokens` outside `[0, row_count)` — the valid
/// embedding-row range (the table's leading dimension).
///
/// MLX `take` (the `embed_tokens` gather) does not bound-check its indices, so a
/// negative id (read as `id + row_count`, and for `id < -row_count` still
/// negative) or an id `>= row_count` is an out-of-bounds embedding-table read
/// (UB). This fails fast with a typed [`Error::OutOfRange`] before the gather.
///
/// The ids are read in their native integer dtype, widened to `i64` so the value
/// is captured without truncation for any integer id tensor (`i32` here, `u32`
/// in other model paths), then compared in `i64` (a negative `row_count`, which
/// cannot occur for a validated config, still admits nothing). Reading the
/// data-backed `tokens` is an explicit materialization of an input, not a hidden
/// eval of a lazy graph.
fn check_token_ids_in_rows(tokens: &Array, row_count: usize, context: &'static str) -> Result<()> {
  let mut widened = tokens.astype(Dtype::I64)?;
  let ids: Vec<i64> = widened.to_vec::<i64>()?;
  let rows = row_count as i64;
  for &id in &ids {
    if id < 0 || id >= rows {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        context,
        "token id must be in [0, embedding-table rows)",
        format_smolstr!("id={id}, rows={row_count}"),
      )));
    }
  }
  Ok(())
}

/// `true` if `weights` carries a `<prefix>.scales` sibling for some layer the
/// Qwen3 loaders **actually consume** under this exact `config` — i.e. the
/// checkpoint is (at least partly) a quantized one.
///
/// This is the load-time half of the `.scales`-presence discriminator the
/// per-layer [`Linear::from_weights`] / [`Embedding::from_weights`] use: those
/// gate on the exact `<prefix>.scales` key for one layer; this pre-scans the
/// whole map once for the same signal across every layer the loaders consume, so
/// [`Qwen3::from_weights`] resolves the quantization scheme
/// ([`Qwen3Config::quantization`]) ONLY when a scale actually needs
/// interpreting. A dense checkpoint (no relevant `.scales`) then loads through
/// the unchanged dense path regardless of any stale / foreign / partial
/// quantization block the config may still carry — the scheme is irrelevant when
/// no consumed layer is quantized.
///
/// The match is **exact and config-aware** (it runs after [`config.validate`]):
/// the relevant keys are precisely the `<prefix>.scales` siblings the
/// [`Linear`] / [`Embedding`] loaders build a `scales_key` for, with the SAME
/// `<prefix>` strings —
///
/// - `model.embed_tokens.scales` (the token embedding);
/// - for each `i` in `0..config.num_hidden_layers` (the actual loaded layer
///   count): the attention `model.layers.{i}.self_attn.{q,k,v,o}_proj.scales`
///   and the MLP `model.layers.{i}.mlp.{gate,up,down}_proj.scales`;
/// - `lm_head.scales` ONLY when the head is untied
///   (`config.tie_word_embeddings == false`), mirroring exactly how
///   [`Qwen3::from_weights`] decides tied-vs-untied: a tied head reuses the
///   embedding table and never consumes `lm_head.scales`, so a stale one is
///   ignored.
///
/// Building the loaders' exact `<prefix>.scales` strings and probing
/// `weights.contains_key` (not a suffix / `ends_with` match) means a foreign key
/// (`foreign.q_proj.scales`), an out-of-range layer index
/// (`model.layers.{N}.…` for `N >= num_hidden_layers`), a never-quantized
/// layer's stale `.scales` (`model.norm.scales`), or a tied `lm_head.scales` is
/// correctly IGNORED — exactly the keys no loader ever consults. Reads only the
/// map's keys (cheap string lookups); no `Array` is touched.
///
/// [`config.validate`]: Qwen3Config::validate
fn has_relevant_scales(config: &Qwen3Config, weights: &HashMap<String, Array>) -> bool {
  // Probe the exact `<prefix>.scales` key the matching loader would build, with
  // the SAME `<prefix>` format the `Linear` / `Embedding` loaders use.
  let has_scales = |prefix: &str| weights.contains_key(&format!("{prefix}{SCALES_SUFFIX}"));

  // The token embedding.
  if has_scales("model.embed_tokens") {
    return true;
  }
  // The untied vocab head consumes `lm_head.scales`; a tied head reuses the
  // embedding table and never does (mirror `from_weights`' tied/untied split).
  if !config.tie_word_embeddings && has_scales("lm_head") {
    return true;
  }
  // Every per-layer projection, for the ACTUAL loaded layer count — an
  // out-of-range index `N >= num_hidden_layers` is never built, so its `.scales`
  // is irrelevant. `num_hidden_layers` is a `validate`d positive `i32`.
  (0..config.num_hidden_layers).any(|i| {
    let q = format!("model.layers.{i}.self_attn");
    let m = format!("model.layers.{i}.mlp");
    has_scales(&format!("{q}.q_proj"))
      || has_scales(&format!("{q}.k_proj"))
      || has_scales(&format!("{q}.v_proj"))
      || has_scales(&format!("{q}.o_proj"))
      || has_scales(&format!("{m}.gate_proj"))
      || has_scales(&format!("{m}.up_proj"))
      || has_scales(&format!("{m}.down_proj"))
  })
}

impl Qwen3 {
  /// `mlx-lm`'s `Model.sanitize` (`qwen3.py:185-188`): when
  /// `tie_word_embeddings` is set, drop a stray `lm_head.weight` (the tied head
  /// reuses the embedding table, so a separate `lm_head` weight is unused).
  /// Operates on a name → [`Array`] map in place; the load path applies this
  /// before constructing the model.
  pub fn sanitize(config: &Qwen3Config, weights: &mut HashMap<String, Array>) {
    if config.tie_word_embeddings {
      weights.remove("lm_head.weight");
    }
  }

  /// Construct a Qwen3 model from a parsed [`Qwen3Config`] and a flat name →
  /// [`Array`] weight map.
  ///
  /// Weight keys follow mlx-lm's `model.*` tree: `model.embed_tokens.weight`,
  /// `model.norm.weight`, and per-layer
  /// `model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight`,
  /// `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
  /// `model.layers.{i}.self_attn.{q,k}_norm.weight`, and
  /// `model.layers.{i}.mlp.{gate,up,down}_proj.weight`. When
  /// `tie_word_embeddings` is false, a top-level `lm_head.weight` is also
  /// required. The map is drained (weights are moved out); a missing required
  /// weight is an [`Error::MissingKey`].
  ///
  /// Callers should run [`sanitize`](Qwen3::sanitize) first (it drops a stray
  /// tied `lm_head.weight`); `from_weights` itself only consumes the keys the
  /// configured head needs.
  ///
  /// A **quantized** checkpoint (e.g. a 4-bit / 8-bit `mlx-community` Qwen3)
  /// loads through the same path: the per-layer scheme parameters are resolved
  /// from the config's `quantization` block (via
  /// [`Qwen3Config::quantization`]) and each `nn.Linear` / the token embedding
  /// is built quantized via the shared [`crate::nn::MaybeQuantizedLinear`] when
  /// the checkpoint carries that layer's `.scales` sibling (the per-layer
  /// auto-detect Whisper / EmbeddingGemma use). The scheme is resolved ONLY when
  /// a relevant `.scales` is present (a one-pass pre-scan over the weight keys,
  /// the load-time half of that same discriminator), so a **dense** checkpoint (no
  /// `.scales`) loads dense regardless of any stale / foreign / partial
  /// `quantization` block the config may still carry — the scheme is only needed
  /// to interpret a scale that is actually present. The non-quant
  /// [`Qwen3Config::validate`] always runs first.
  pub fn from_weights(config: Qwen3Config, mut weights: HashMap<String, Array>) -> Result<Qwen3> {
    config.validate()?;

    // Resolve the per-layer quantization scheme ONLY when the checkpoint
    // actually carries a `.scales` sibling for some layer the model loads (the
    // `.scales`-presence discriminator the per-layer `Linear` / `Embedding`
    // loaders use, hoisted to the whole map). When no layer is quantized the
    // scheme is irrelevant, so a DENSE checkpoint loads through the unchanged
    // dense path regardless of any stale / foreign / partial `quantization`
    // block the config may still carry — only a present `.scales` makes an
    // unresolvable scheme fatal (the per-layer typed `InvariantViolation`). The
    // non-quant config validation above always runs. The result is threaded to
    // the decoder + the untied head, which pick quantized-vs-dense per layer by
    // the same `.scales` sibling.
    let quant = if has_relevant_scales(&config, &weights) {
      config.quantization()?
    } else {
      None
    };

    // The head-less decoder (drains the `model.*` keys).
    let model = Qwen3Model::from_weights(&config, &mut weights, quant.as_ref())?;

    // The vocab head: tied reuses the embedding table (via
    // `Embedding::as_linear`); untied reads `lm_head.weight` from the remaining
    // map. The untied weight is shape-checked on load to `(vocab_size,
    // hidden_size)` — the same pin the embedding table gets — so `forward`
    // produces the `(B, S, vocab_size)` logits the `Model` contract promises. A
    // wrong row count would emit token ids outside the configured vocab; a wrong
    // hidden width would defer the failure into the matmul rather than report a
    // typed load-time error here. The untied head auto-detects dense-vs-quantized
    // from its `lm_head.scales` sibling like every other projection.
    let lm_head = if config.tie_word_embeddings {
      LmHead::Tied
    } else {
      LmHead::Untied(Linear::from_weights(
        &mut weights,
        "lm_head",
        config.vocab_size,
        config.hidden_size,
        "lm_head weight (vocab_size, hidden_size)",
        quant.as_ref(),
      )?)
    };

    Ok(Qwen3 {
      config,
      model,
      lm_head,
    })
  }
}

#[cfg(test)]
mod tests;
