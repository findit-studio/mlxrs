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
use linear::Linear;
use smol_str::format_smolstr;

use crate::{
  array::Array,
  error::{Error, LengthMismatchPayload, MissingKeyPayload, Result},
  lm::{
    cache::{KvCache, MaskMode, StandardKvCache},
    model::Model as LmModel,
    nn::{
      activations::swiglu,
      attention::{Mask, scaled_dot_product_attention},
      norm::RMSNorm,
      rope::Rope,
    },
  },
  model_validation::reserve_or_error,
  ops::{
    indexing::take_axis,
    linalg_basic::matmul,
    shape::{reshape, swapaxes, transpose_axes},
  },
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
  /// `(vocab, hidden)` token-embedding table; also the tied output head when
  /// reused by [`Qwen3`].
  embed_tokens: Array,
  layers: Vec<TransformerBlock>,
  norm: RMSNorm,
}

impl Qwen3Model {
  /// Embed `tokens` (`(B, L)` integer ids) to `(B, L, hidden)` via the token
  /// embedding table — the row-gather front of the decoder, exposed so a
  /// caller can splice input features into chosen positions of the embeddings
  /// before running [`forward_hidden`](Self::forward_hidden). No implicit eval —
  /// the returned [`Array`] is lazy.
  pub fn embed_tokens(&self, tokens: &Array) -> Result<Array> {
    take_axis(&self.embed_tokens, tokens, 0)
  }

  /// The number of decoder layers — also the per-layer cache cardinality
  /// [`make_cache`](Self::make_cache) builds.
  #[inline(always)]
  pub fn num_layers(&self) -> usize {
    self.layers.len()
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
    let n = h.shape()[1];

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
  /// weight is an [`Error::MissingKey`]. The caller validates `config` and
  /// builds the output head that sits on top (the LM vocab head in
  /// [`Qwen3::from_weights`]).
  pub fn from_weights(
    config: &Qwen3Config,
    weights: &mut HashMap<String, Array>,
  ) -> Result<Qwen3Model> {
    let eps = config.rms_norm_eps;
    let head_dim = config.head_dim;

    let embed_tokens = take_weight(weights, "model.embed_tokens.weight")?;
    let norm = RMSNorm::new(take_weight(weights, "model.norm.weight")?, eps);

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
        q_proj: Linear::new(take_weight(weights, &format!("{q}.q_proj.weight"))?, None),
        k_proj: Linear::new(take_weight(weights, &format!("{q}.k_proj.weight"))?, None),
        v_proj: Linear::new(take_weight(weights, &format!("{q}.v_proj.weight"))?, None),
        o_proj: Linear::new(take_weight(weights, &format!("{q}.o_proj.weight"))?, None),
        q_norm: RMSNorm::new(take_weight(weights, &format!("{q}.q_norm.weight"))?, eps),
        k_norm: RMSNorm::new(take_weight(weights, &format!("{q}.k_norm.weight"))?, eps),
        rope: Rope::new(head_dim, false, config.rope_theta, 1.0),
      };

      let mlp = Mlp {
        gate_proj: Linear::new(
          take_weight(weights, &format!("{p}.mlp.gate_proj.weight"))?,
          None,
        ),
        up_proj: Linear::new(
          take_weight(weights, &format!("{p}.mlp.up_proj.weight"))?,
          None,
        ),
        down_proj: Linear::new(
          take_weight(weights, &format!("{p}.mlp.down_proj.weight"))?,
          None,
        ),
      };

      let input_layernorm = RMSNorm::new(
        take_weight(weights, &format!("{p}.input_layernorm.weight"))?,
        eps,
      );
      let post_attention_layernorm = RMSNorm::new(
        take_weight(weights, &format!("{p}.post_attention_layernorm.weight"))?,
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

  /// Project final hidden states to vocab logits via the configured head.
  ///
  /// Tied (`qwen3.py:180-181`): `embed_tokens.as_linear(out)` = `out @
  /// embed_tokens.T`. Untied: the dedicated `lm_head` linear.
  fn project_logits(&self, hidden: &Array) -> Result<Array> {
    match &self.lm_head {
      LmHead::Tied => {
        let wt = swapaxes(&self.model.embed_tokens, -1, -2)?;
        matmul(hidden, &wt)
      }
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

/// Pull a required weight out of the map by `name`, erroring with the key on
/// absence (mlx's `model.update(tree_unflatten(weights))` would raise).
fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> Result<Array> {
  weights.remove(name).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "Qwen3 weight map",
      format_smolstr!("{name}"),
    ))
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
  pub fn from_weights(config: Qwen3Config, mut weights: HashMap<String, Array>) -> Result<Qwen3> {
    config.validate()?;

    // The head-less decoder (drains the `model.*` keys).
    let model = Qwen3Model::from_weights(&config, &mut weights)?;

    // The vocab head: tied reuses the embedding table; untied reads
    // `lm_head.weight` from the remaining map.
    let lm_head = if config.tie_word_embeddings {
      LmHead::Tied
    } else {
      LmHead::Untied(Linear::new(
        take_weight(&mut weights, "lm_head.weight")?,
        None,
      ))
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
