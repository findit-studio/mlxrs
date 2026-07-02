//! EmbeddingGemma's Gemma3 text backbone.
//!
//! Ports `mlx-embeddings`'s `models/gemma3_text.py` `Gemma3Model`, which reuses
//! the `ModelArgs` / `RMSNorm` / `TransformerBlock` of `mlx-lm`'s
//! `models/gemma3_text.py`. The backbone is a Gemma3 text transformer driven as
//! a **bidirectional encoder**: the public model (see [`super`]) builds an
//! additive padding mask (`0` on real tokens, `-inf` on pad) from the `{0,1}`
//! attention mask, and the backbone derives each layer's mask from it
//! (non-causal throughout):
//!
//! - **Global** (full-attention) layers — [`Gemma3Config::is_global_layer`],
//!   every `sliding_window_pattern`-th layer — attend over **all** real tokens
//!   through the padding-only mask.
//! - **Local** (sliding-window) layers additionally attend through the
//!   **bidirectional sliding window**: query `i` sees key `j` only when
//!   `|i - j| < sliding_window` (strict). The band overlay is materialized only
//!   when `seq_len > sliding_window` (`build_local_layer_mask`); a shorter
//!   sequence skips it entirely, which is *exact* — every distance is then
//!   `<= seq_len - 1 < sliding_window`, the overlay would be identically zero,
//!   and `softmax(x + 0) = softmax(x)` — so `<= 512`-token inputs are
//!   bit-for-bit unaffected.
//!
//! **Reference choice**: the window masking follows the google/HF reference —
//! `transformers`' `modeling_gemma3.py`, whose `_bidirectional_window_overlay`
//! (`abs(q_idx - kv_idx) < sliding_window`) is OR-composed onto the
//! `sliding_attention` layers' mask when the checkpoint sets
//! `use_bidirectional_attention: true` (EmbeddingGemma's `config.json` does) —
//! and **deliberately deviates from the declared upstream** `mlx-embeddings`
//! (`models/gemma3_text.py`), whose `Model.__call__` feeds the padding-only
//! `extended_attention_mask` to every layer (bypassing the imported mlx-lm
//! window path, which only activates when *no* explicit mask is given). For
//! `> sliding_window`-token inputs that upstream silently diverges from the
//! google reference, so window-faithful masking wins here; the per-layer RoPE
//! base split (below) is unchanged and shared by both references.
//!
//! ## Per-layer architecture (`gemma3_text.py` `TransformerBlock`)
//!
//! Each layer is the Gemma3 sandwich-norm block:
//!
//! ```text
//! r   = self_attn(input_layernorm(x))
//! h   = x + post_attention_layernorm(r)
//! r   = mlp(pre_feedforward_layernorm(h))
//! out = h + post_feedforward_layernorm(r)
//! ```
//!
//! - **Attention**: grouped-query (`num_attention_heads` query heads,
//!   `num_key_value_heads` kv heads), per-head `head_dim` (independent of
//!   `hidden / heads`), **query and key RMSNorm** over the head dimension,
//!   RoPE, and the fused SDPA with `scale = query_pre_attn_scalar ** -0.5`. The
//!   RoPE base alternates per layer: the **global** layers (`is_global_layer`)
//!   use `rope_theta`, the local layers use `rope_local_base_freq`. The same
//!   global/local split selects each layer's mask (padding-only vs banded, see
//!   above) — both read [`Gemma3Config::is_global_layer`], so the RoPE base and
//!   the window can never disagree.
//! - **MLP**: the Gemma gated feed-forward `down(gelu_approx(gate(x)) * up(x))`
//!   (`gelu_approx` = the `tanh` GELU, `mlx.nn.gelu_approx`).
//! - **Norms**: four `RMSNorm`s per layer plus a final backbone `RMSNorm`,
//!   each with Gemma's `1.0 + weight` reparameterization (folded in at load by
//!   the private `shared::build_gemma_rms_norm`).
//!
//! The token embedding is scaled by `sqrt(hidden_size)` before the first layer
//! (`gemma3_text.py`'s `h *= hidden_size ** 0.5`).

use std::collections::HashMap;

use crate::{
  array::Array,
  error::{Error, RankMismatchPayload, Result},
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::RMSNorm,
    rope::Rope,
  },
  model_validation::reserve_or_error,
  ops,
};

use super::{
  config::Gemma3Config,
  shared::{DtypeAlias, Embedding, Linear, build_gemma_rms_norm, dim_i32, embedding_scale_like},
};
use crate::lm::quant::PerLayerQuantization;

/// Gemma3 grouped-query attention with per-head q/k RMSNorm (`gemma3_text.py`'s
/// `Attention`).
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct Attention {
  n_heads: i32,
  n_kv_heads: i32,
  head_dim: i32,
  /// SDPA scale (`query_pre_attn_scalar ** -0.5`).
  scale: f32,
  q_proj: Linear,
  k_proj: Linear,
  v_proj: Linear,
  o_proj: Linear,
  q_norm: RMSNorm,
  k_norm: RMSNorm,
  rope: Rope,
}

#[cfg(feature = "embeddinggemma")]
impl Attention {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    config: &Gemma3Config,
    layer_idx: i32,
    eps: f32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim;
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    // q is `(n_heads * head_dim, hidden)`; k/v are `(n_kv_heads * head_dim,
    // hidden)`; o is `(hidden, n_heads * head_dim)`. The products are
    // overflow-checked (config caps keep them small in practice).
    let q_out = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_heads * head_dim",
      "n_heads",
      n_heads,
      "head_dim",
      head_dim,
    )?;
    let kv_out = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_kv_heads * head_dim",
      "n_kv_heads",
      n_kv_heads,
      "head_dim",
      head_dim,
    )?;
    // Each projection auto-detects dense-vs-quantized from its `.scales` sibling
    // (mlx-embeddings' `class_predicate` quantizes every `nn.Linear`); the dense
    // path pins the `(out, in)` weight to the config exactly as before.
    let q_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.q_proj"),
      q_out,
      hidden,
      "attn q_proj weight (n_heads*head_dim, hidden)",
      quant,
    )?;
    let k_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.k_proj"),
      kv_out,
      hidden,
      "attn k_proj weight (n_kv_heads*head_dim, hidden)",
      quant,
    )?;
    let v_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.v_proj"),
      kv_out,
      hidden,
      "attn v_proj weight (n_kv_heads*head_dim, hidden)",
      quant,
    )?;
    let o_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.o_proj"),
      hidden,
      q_out,
      "attn o_proj weight (hidden, n_heads*head_dim)",
      quant,
    )?;
    let q_norm = build_gemma_rms_norm(weights, &format!("{prefix}.q_norm"), head_dim, eps)?;
    let k_norm = build_gemma_rms_norm(weights, &format!("{prefix}.k_norm"), head_dim, eps)?;

    // RoPE base alternates: global layers use `rope_theta`, local (sliding)
    // layers use `rope_local_base_freq`. Non-traditional layout, scale 1.0,
    // matching `gemma3_text.py`'s `initialize_rope(..., traditional=False)`.
    let base = if config.is_global_layer(layer_idx) {
      config.rope_theta as f32
    } else {
      config.rope_local_base_freq as f32
    };
    let rope = Rope::new(head_dim, false, base, 1.0);

    Ok(Self {
      n_heads,
      n_kv_heads,
      head_dim,
      scale: (config.query_pre_attn_scalar as f32).powf(-0.5),
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      q_norm,
      k_norm,
      rope,
    })
  }

  /// `(B, L, hidden) → (B, L, hidden)` bidirectional attention with the
  /// additive `mask` — padding-only `(B, 1, 1, L)` on global layers, padding +
  /// sliding-window band `(B, 1, L, L)` on local layers once `L` exceeds the
  /// window (both broadcast over heads). No KV cache (encoder), so RoPE is
  /// applied at offset 0.
  fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
    let shape = x.shape();
    let b = dim_i32(&shape, 0, "Gemma3 Attention: batch")?;
    let l = dim_i32(&shape, 1, "Gemma3 Attention: seq")?;

    let queries = self.q_proj.forward(x)?;
    let keys = self.k_proj.forward(x)?;
    let values = self.v_proj.forward(x)?;

    // Per-head reshape `(B, L, heads, head_dim)`, then transpose to
    // `(B, heads, L, head_dim)`. q/k RMSNorm over the head dimension is applied
    // **before** the transpose (the reference norms the `(B, L, heads,
    // head_dim)` layout, whose last axis is `head_dim`).
    let queries = ops::shape::reshape(&queries, &[b, l, self.n_heads, self.head_dim])?;
    let queries = self.q_norm.forward(&queries)?;
    let queries = ops::shape::transpose_axes(&queries, &[0, 2, 1, 3])?;

    let keys = ops::shape::reshape(&keys, &[b, l, self.n_kv_heads, self.head_dim])?;
    let keys = self.k_norm.forward(&keys)?;
    let keys = ops::shape::transpose_axes(&keys, &[0, 2, 1, 3])?;

    let values = ops::shape::reshape(&values, &[b, l, self.n_kv_heads, self.head_dim])?;
    let values = ops::shape::transpose_axes(&values, &[0, 2, 1, 3])?;

    // Encoder: no cache, rotate at absolute position 0.
    let queries = self.rope.apply(&queries, 0)?;
    let keys = self.rope.apply(&keys, 0)?;

    let attn =
      scaled_dot_product_attention(&queries, &keys, &values, self.scale, Mask::Array(mask))?;
    // `(B, heads, L, head_dim)` → `(B, L, heads*head_dim)`.
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = crate::model_validation::checked_mul(
      "Gemma3 Attention: n_heads * head_dim",
      "n_heads",
      self.n_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[b, l, embed_dim])?;
    self.o_proj.forward(&attn)
  }
}

/// Gemma gated feed-forward (`gemma3_text.py`'s `MLP`):
/// `down(gelu_approx(gate(x)) * up(x))`, all three projections bias-free.
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct Mlp {
  gate_proj: Linear,
  up_proj: Linear,
  down_proj: Linear,
}

#[cfg(feature = "embeddinggemma")]
impl Mlp {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    hidden: i32,
    intermediate: i32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let gate_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.gate_proj"),
      intermediate,
      hidden,
      "MLP gate_proj weight (intermediate, hidden)",
      quant,
    )?;
    let up_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.up_proj"),
      intermediate,
      hidden,
      "MLP up_proj weight (intermediate, hidden)",
      quant,
    )?;
    let down_proj = Linear::from_weights(
      weights,
      &format!("{prefix}.down_proj"),
      hidden,
      intermediate,
      "MLP down_proj weight (hidden, intermediate)",
      quant,
    )?;
    Ok(Self {
      gate_proj,
      up_proj,
      down_proj,
    })
  }

  /// `down(gelu_approx(gate(x)) * up(x))`.
  fn forward(&self, x: &Array) -> Result<Array> {
    let gate = self.gate_proj.forward(x)?;
    let gate = crate::lm::nn::activations::gelu_approx(&gate)?;
    let up = self.up_proj.forward(x)?;
    let h = gate.multiply(&up)?;
    self.down_proj.forward(&h)
  }
}

/// The Gemma3 residual add with f16 saturation (`gemma3_text.py`'s
/// `clip_residual`).
///
/// For any non-f16 dtype this is a plain `x + y`. For f16 — whose finite range
/// tops out at `f16::MAX` (`65504`) — a residual sum can overflow to `inf` and
/// then poison the rest of the network with `NaN`s, so the reference instead
/// adds in f32, clamps to the finite f16 range `[-f16::MAX, f16::MAX]`, and
/// casts back to f16, yielding the saturated finite value rather than `inf`.
/// Both residual additions in the block route through this.
#[cfg(feature = "embeddinggemma")]
fn clip_residual(x: &Array, y: &Array) -> Result<Array> {
  if x.dtype()? != crate::dtype::Dtype::F16 {
    return x.add(y);
  }
  // The f16 saturating add below is `gemma3_text.py`'s
  // `@partial(mx.compile, shapeless=True) def clip_residual` — five ops
  // (astype, astype, add, clip, astype) the reference fuses into ONE kernel.
  // Route it through a process-lifetime compiled graph (traced once on first
  // use, reused for every later call) so this block dispatches one fused kernel
  // instead of five; it is applied twice per layer across every block, so the
  // un-fused form is a measurable share of a short f16 forward.
  // Fall back to the uncompiled body if the graph cannot be built or a call
  // fails — the math is identical, only un-fused.
  //
  // The `Compiled` is **leaked** to `'static` rather than stored by value in the
  // `thread_local`. That is deliberate: a `thread_local` `Compiled` would be
  // dropped at *thread teardown*, running `mlx_closure_free` after mlx's own
  // static state may already be gone — a use-after-free observed as a SIGSEGV at
  // process exit. Leaking mirrors the reference's module-level `@mx.compile`
  // (likewise never freed) and mlx's process-global compile cache: one tiny
  // handle per thread, alive for the process.
  thread_local! {
    static CLIP_F16: std::cell::OnceCell<Option<&'static crate::transforms::compile::Compiled>> =
      const { std::cell::OnceCell::new() };
  }
  let compiled: Option<&'static crate::transforms::compile::Compiled> = CLIP_F16.with(|cell| {
    *cell.get_or_init(|| {
      crate::transforms::compile::compile(
        |ins: &[Array]| clip_residual_f16(&ins[0], &ins[1]).map(|out| vec![out]),
        true,
      )
      .ok()
      .map(|c| &*Box::leak(Box::new(c)))
    })
  });
  match compiled {
    Some(c) => {
      // Marshal the two inputs for the compiled call. `try_clone` is a fallible
      // FFI handle-retain; a clone failure falls back to the uncompiled body
      // rather than propagating — the contract is that ANY compiled-path failure
      // degrades to the identical uncompiled math, so a recoverable alloc/FFI
      // error never surfaces as an error out of a saturating residual add.
      let inputs = match (x.try_clone(), y.try_clone()) {
        (Ok(a), Ok(b)) => [a, b],
        _ => return clip_residual_f16(x, y),
      };
      c.call(&inputs)
        .ok()
        .and_then(|mut out| out.pop())
        .map_or_else(|| clip_residual_f16(x, y), Ok)
    }
    None => clip_residual_f16(x, y),
  }
}

/// The f16 saturating residual add (`gemma3_text.py`'s `clip_residual` body):
/// add in f32, clamp to the finite f16 range `[-f16::MAX, f16::MAX]`, and cast
/// back to f16 — so an f16 backbone saturates rather than overflowing to
/// `inf`/`NaN`. Extracted so [`clip_residual`] can hand it to `mx.compile` and
/// reuse it verbatim as the uncompiled fallback.
#[cfg(feature = "embeddinggemma")]
fn clip_residual_f16(x: &Array, y: &Array) -> Result<Array> {
  // `half::f16::MAX` is mlx's `finfo(float16).max` (`65504`).
  let bound = f32::from(half::f16::MAX);
  let xf = ops::misc::astype(x, crate::dtype::Dtype::F32)?;
  let yf = ops::misc::astype(y, crate::dtype::Dtype::F32)?;
  let sum = xf.add(&yf)?;
  let clipped = ops::misc::clip_with_scalar(&sum, -bound, bound)?;
  ops::misc::astype(&clipped, crate::dtype::Dtype::F16)
}

/// A Gemma3 sandwich-norm transformer block (`gemma3_text.py`'s
/// `TransformerBlock`).
#[cfg(feature = "embeddinggemma")]
#[derive(Debug)]
struct TransformerBlock {
  self_attn: Attention,
  mlp: Mlp,
  input_layernorm: RMSNorm,
  post_attention_layernorm: RMSNorm,
  pre_feedforward_layernorm: RMSNorm,
  post_feedforward_layernorm: RMSNorm,
  /// `true` for a **global** (full-attention) layer: it attends through the
  /// padding-only mask, while a local layer gets the banded mask once the
  /// sequence exceeds the sliding window ([`Gemma3Backbone::forward`]).
  /// Derived from the SAME [`Gemma3Config::is_global_layer`] that selects the
  /// RoPE base in [`Attention::from_weights`], so mask and base cannot
  /// disagree.
  is_global: bool,
}

#[cfg(feature = "embeddinggemma")]
impl TransformerBlock {
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    config: &Gemma3Config,
    layer_idx: i32,
    eps: f32,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    let prefix = format!("model.layers.{layer_idx}");
    let hidden = config.hidden_size;
    let self_attn = Attention::from_weights(
      weights,
      &format!("{prefix}.self_attn"),
      config,
      layer_idx,
      eps,
      quant,
    )?;
    let mlp = Mlp::from_weights(
      weights,
      &format!("{prefix}.mlp"),
      hidden,
      config.intermediate_size,
      quant,
    )?;
    let input_layernorm =
      build_gemma_rms_norm(weights, &format!("{prefix}.input_layernorm"), hidden, eps)?;
    let post_attention_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.post_attention_layernorm"),
      hidden,
      eps,
    )?;
    let pre_feedforward_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.pre_feedforward_layernorm"),
      hidden,
      eps,
    )?;
    let post_feedforward_layernorm = build_gemma_rms_norm(
      weights,
      &format!("{prefix}.post_feedforward_layernorm"),
      hidden,
      eps,
    )?;
    Ok(Self {
      self_attn,
      mlp,
      input_layernorm,
      post_attention_layernorm,
      pre_feedforward_layernorm,
      post_feedforward_layernorm,
      is_global: config.is_global_layer(layer_idx),
    })
  }

  /// `r = attn(ln_in(x)); h = clip_residual(x, ln_post_attn(r)); r =
  /// mlp(ln_pre_ff(h)); out = clip_residual(h, ln_post_ff(r))`. Both residual
  /// adds go through [`clip_residual`] so an f16 backbone saturates to the
  /// finite f16 range instead of overflowing to `inf`/`NaN` (the reference's
  /// `clip_residual`).
  fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
    let r = self
      .self_attn
      .forward(&self.input_layernorm.forward(x)?, mask)?;
    let h = clip_residual(x, &self.post_attention_layernorm.forward(&r)?)?;
    let r = self
      .mlp
      .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
    clip_residual(&h, &self.post_feedforward_layernorm.forward(&r)?)
  }
}

/// The Gemma3 text backbone (`gemma3_text.py`'s `Gemma3Model`): token embedding
/// (scaled by `sqrt(hidden)`) → sandwich-norm transformer stack → final
/// `RMSNorm`. Driven as a bidirectional encoder by the public model.
#[cfg(feature = "embeddinggemma")]
#[cfg_attr(docsrs, doc(cfg(feature = "embeddinggemma")))]
#[derive(Debug)]
pub(crate) struct Gemma3Backbone {
  embed_tokens: Embedding,
  layers: Vec<TransformerBlock>,
  norm: RMSNorm,
  hidden_size: i32,
  /// The local layers' bidirectional band half-width
  /// ([`Gemma3Config::sliding_window`], validated `>= 1`): once
  /// `seq_len > sliding_window`, [`forward`](Self::forward) builds the banded
  /// mask the local layers attend through.
  sliding_window: i32,
}

#[cfg(feature = "embeddinggemma")]
impl Gemma3Backbone {
  /// Build the backbone from a validated [`Gemma3Config`] and the (sanitized)
  /// weight map, whose keys follow `mlx-lm`'s `model.*` tree
  /// (`model.embed_tokens.weight`, `model.layers.{i}.*`, `model.norm.weight`).
  ///
  /// `quant` is the parsed per-layer quantization config (from
  /// [`Gemma3Config::quantization`]); each `nn.Linear` / the token embedding
  /// loads quantized when the checkpoint carries that layer's `.scales` sibling
  /// (the mlx-embeddings `class_predicate` auto-detect), else dense.
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions (typed [`crate::Error::ShapePairMismatch`] wrapped in
  /// [`crate::Error::LayerKeyed`]); the quantized path pins the packed triple's
  /// logical shape identically.
  pub(crate) fn from_weights(
    config: &Gemma3Config,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: a caller may build directly from an unvalidated
    // config. This bounds `num_hidden_layers` (and every dim) before the
    // per-layer reservation/loop.
    config.validate()?;
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps as f32;

    // The token embedding is an `nn.Embedding`, also quantize-aware (the
    // weight-tied projection mlx-embeddings' `class_predicate` quantizes).
    let embed_tokens = Embedding::from_weights(
      weights,
      "model.embed_tokens",
      config.vocab_size,
      hidden,
      "token-embedding table (vocab, hidden)",
      quant,
    )?;

    // `num_hidden_layers` is bounded by `MAX_CARDINALITY` in `validate`, but
    // reserve fallibly so even a within-cap heavyweight per-layer `Vec` the
    // allocator cannot satisfy is a recoverable [`Error::AllocFailure`] rather
    // than `with_capacity`'s abort (the merged LFM2 / SigLIP2 pattern).
    let mut layers: Vec<TransformerBlock> = Vec::new();
    reserve_or_error(
      &mut layers,
      "Gemma3 TransformerBlock",
      config.num_hidden_layers as usize,
    )?;
    for i in 0..config.num_hidden_layers {
      layers.push(TransformerBlock::from_weights(
        weights, config, i, eps, quant,
      )?);
    }

    let norm = build_gemma_rms_norm(weights, "model.norm", hidden, eps)?;

    Ok(Self {
      embed_tokens,
      layers,
      norm,
      hidden_size: hidden,
      sliding_window: config.sliding_window,
    })
  }

  /// Run the backbone over a `(batch, seq_len)` i32 token-id batch and the
  /// `(batch, 1, 1, seq_len)` additive padding `mask`, returning the final
  /// `(batch, seq_len, hidden)` hidden states (post final `RMSNorm`).
  ///
  /// Global layers attend through `mask` as-is; once
  /// `seq_len > sliding_window` the local layers attend through the banded
  /// `(batch, 1, seq_len, seq_len)` mask derived from it
  /// ([`build_local_layer_mask`], built once and shared by every local layer).
  /// At `seq_len <= sliding_window` the band is skipped — exactly equivalent
  /// (the overlay would be identically zero) and bit-for-bit the pre-window
  /// behavior (every layer gets the same `mask` object as before).
  ///
  /// `input_ids` is pinned to exactly rank-2 before any op (the public
  /// `embed_text` accepts an untrusted array; the embedding gather + the
  /// per-head reshape are only defined for a rank-2 batch).
  pub(crate) fn forward(&self, input_ids: &Array, mask: &Array) -> Result<Array> {
    let shape = input_ids.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "embeddinggemma backbone: input_ids must be rank-2 (batch, seq_len)",
        shape.len() as u32,
        shape,
      )));
    }
    let seq_len = dim_i32(&shape, 1, "Gemma3 backbone: seq_len")?;

    // token_embedding(ids): (B, L) → (B, L, hidden) via axis-0 gather (the
    // quantized table dequantizes the gathered rows).
    let mut h = self.embed_tokens.forward(input_ids)?;
    // Scale by sqrt(hidden_size). The reference builds this scalar in the
    // `embed_tokens` weight dtype, then casts to the hidden dtype, with no bf16
    // rounding (mlx-embeddings `gemma3_text.py`'s
    // `h *= mx.array(hidden_size ** 0.5, embed_tokens.weight.dtype).astype(h.dtype)`).
    // Here `h` is gathered straight from `embed_tokens`, so its dtype is the
    // weight (dense) / dequantized-activation (quantized) dtype and the scalar is
    // built directly in it.
    let scale = embedding_scale_like(self.hidden_size as f32, &h)?;
    h = h.multiply(&scale)?;

    // The local layers' banded mask, built ONCE per forward (shared by every
    // local layer) and only when the sequence actually exceeds the window —
    // at `seq_len <= sliding_window` the band overlay would be identically
    // zero (`|i - j| <= seq_len - 1 < window` everywhere), so skipping it is
    // exact and keeps short inputs on the unchanged padding-only path.
    let local_mask = if seq_len > self.sliding_window {
      Some(build_local_layer_mask(mask, seq_len, self.sliding_window)?)
    } else {
      None
    };

    for layer in &self.layers {
      let layer_mask = if layer.is_global {
        mask
      } else {
        local_mask.as_ref().unwrap_or(mask)
      };
      h = layer.forward(&h, layer_mask)?;
    }
    self.norm.forward(&h)
  }

  /// The token-embedding table's dtype — the dtype the additive attention mask
  /// must be cast to so the fused SDPA sees a matching-dtype mask. A cheap
  /// handle query (no eval).
  #[inline]
  pub(crate) fn embed_dtype(&self) -> Result<crate::dtype::Dtype> {
    self.embed_tokens.dtype()
  }

  /// `true` if the token embedding was loaded from a quantized checkpoint
  /// (test-only introspection for the quantized-load test).
  #[cfg(test)]
  pub(crate) fn embedding_is_quantized(&self) -> bool {
    self.embed_tokens.is_quantized()
  }

  /// `true` if every layer's attention + MLP projections were loaded quantized
  /// (test-only introspection). Empty layer stacks vacuously return `true`.
  #[cfg(test)]
  pub(crate) fn all_projections_quantized(&self) -> bool {
    self.layers.iter().all(|l| {
      l.self_attn.q_proj.is_quantized()
        && l.self_attn.k_proj.is_quantized()
        && l.self_attn.v_proj.is_quantized()
        && l.self_attn.o_proj.is_quantized()
        && l.mlp.gate_proj.is_quantized()
        && l.mlp.up_proj.is_quantized()
        && l.mlp.down_proj.is_quantized()
    })
  }
}

/// Build the `(batch, 1, 1, seq_len)` additive attention mask from a
/// `(batch, seq_len)` `{0,1}` padding mask: `0.0` where a token is attended,
/// `-inf` where it is padding — broadcastable to the SDPA `[B, N_q, T_q, T_kv]`
/// key axis.
///
/// Mirrors `gemma3_text.py` (the mlx-embeddings `Model`)
/// `get_extended_attention_mask` + the `where(mask, 0.0, -inf)` step: a rank-2
/// `attention_mask` becomes `[:, None, None, :]`, then the boolean is mapped to
/// the additive `{0, -inf}` form and cast to `dtype`. The result masks **keys**
/// only (every query attends to every real key — bidirectional), which is the
/// encoder contract for the **global** layers; the local layers' banded mask is
/// derived from it by [`build_local_layer_mask`] once the sequence exceeds the
/// sliding window.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn build_additive_mask(attention_mask: &Array, dtype: DtypeAlias) -> Result<Array> {
  let shape = attention_mask.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "embeddinggemma: attention_mask must be rank-2 (batch, seq_len)",
      shape.len() as u32,
      shape,
    )));
  }
  // (B, S) → (B, 1, 1, S): one broadcastable key-axis mask per batch row.
  let expanded = ops::shape::expand_dims_axes(attention_mask, &[1, 2])?;
  // `where(mask != 0, 0.0, -inf)`. The padding mask is `{0.0, 1.0}` f32 here
  // (the encode pipeline builds an f32 mask); compare against 0 to a boolean,
  // then select 0.0 / -inf.
  let zero = Array::full::<f32>(&(1,), 0.0)?;
  let keep = ops::comparison::not_equal(&expanded, &zero)?; // bool: real tokens
  let additive_zero = Array::full::<f32>(&(1,), 0.0)?;
  let neg_inf = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
  let mask = ops::logical::select(&keep, &additive_zero, &neg_inf)?;
  // Cast to the hidden dtype so SDPA sees a matching additive mask dtype.
  ops::misc::astype(&mask, dtype)
}

/// Build the `(1, 1, seq_len, seq_len)` additive **bidirectional
/// sliding-window overlay**: `0` where `|i - j| < window` (query `i` may see
/// key `j`), `-inf` elsewhere — the HF Gemma3 `_bidirectional_window_overlay`
/// (`modeling_gemma3.py`: `abs(q_idx - kv_idx) < sliding_window`, strictly
/// exclusive at `|i - j| == window`).
///
/// The position/distance grid is computed in **f32** and only the finished
/// `{0, -inf}` overlay is cast to `dtype`: small position integers are exact in
/// f32, whereas bf16 cannot represent every integer near a real 512 window
/// (e.g. `511` rounds to `512`), which would corrupt the strict `< window`
/// comparison at the band edge. The cast itself is exact (`0` and `-inf` are
/// representable in every float dtype), so SDPA sees a matching-dtype additive
/// mask — the same dtype discipline as [`build_additive_mask`].
#[cfg(feature = "embeddinggemma")]
pub(crate) fn build_sliding_window_overlay(
  seq_len: i32,
  window: i32,
  dtype: DtypeAlias,
) -> Result<Array> {
  // Positions 0..seq_len as an f32 ramp, split into a (S, 1) query column and a
  // (1, S) key row so `q - k` broadcasts to the (S, S) signed-distance grid.
  let positions = Array::arange::<f32>(0.0, f64::from(seq_len), 1.0)?;
  let q = ops::shape::reshape(&positions, &[seq_len, 1])?;
  let k = ops::shape::reshape(&positions, &[1, seq_len])?;
  let dist = ops::arithmetic::abs(&ops::arithmetic::subtract(&q, &k)?)?;
  // `|i - j| < window` (strict — HF's exclusive band edge).
  let window_arr = Array::full::<f32>(&(1,), window as f32)?;
  let inside = ops::comparison::less(&dist, &window_arr)?;
  let zero = Array::full::<f32>(&(1,), 0.0)?;
  let neg_inf = Array::full::<f32>(&(1,), f32::NEG_INFINITY)?;
  let overlay = ops::logical::select(&inside, &zero, &neg_inf)?;
  let overlay = ops::misc::astype(&overlay, dtype)?;
  // (S, S) → (1, 1, S, S): broadcastable over batch and heads.
  ops::shape::reshape(&overlay, &[1, 1, seq_len, seq_len])
}

/// Combine the `(batch, 1, 1, seq_len)` additive padding `mask` with the
/// [`build_sliding_window_overlay`] band into the
/// `(batch, 1, seq_len, seq_len)` additive mask the **local** layers attend
/// through: key `j` is visible to query `i` iff it is a real token AND
/// `|i - j| < window` — the HF Gemma3 bidirectional sliding-window semantics
/// (window AND padding; in additive form the two `{0, -inf}` masks simply add,
/// and `-inf + -inf = -inf`, never a `NaN` — there is no `+inf` operand).
///
/// A query row left with **no** visible key — only possible at *padded* query
/// positions, once the right-padding run reaches the window (every real key is
/// then outside the band; a real query always sees itself, `|i - i| = 0 <
/// window`) — is reset to fully-visible (all-`0`), mirroring HF
/// `masking_utils.sdpa_mask`'s fully-masked-row fix (`attention_mask |
/// torch.all(~attention_mask, dim=-1, keepdim=True)`). Without the reset such a
/// row softmaxes all-`-inf` logits, and a `NaN` there poisons every REAL token
/// at the next layer: a real query weights the padded position's `NaN` *value*
/// by `exp(-inf) = 0`, and `0 × NaN = NaN`. The reset only changes hidden
/// states at padded query positions, which mean pooling masks out, so the
/// pooled embedding is unaffected.
///
/// Every piece is built in (or cast to) `mask`'s dtype, so SDPA sees one
/// matching-dtype additive mask — the fast-SDPA mask dtype rule
/// [`build_additive_mask`] already follows.
#[cfg(feature = "embeddinggemma")]
pub(crate) fn build_local_layer_mask(mask: &Array, seq_len: i32, window: i32) -> Result<Array> {
  let dtype = mask.dtype()?;
  let overlay = build_sliding_window_overlay(seq_len, window, dtype)?;
  // (B, 1, 1, S) + (1, 1, S, S) → (B, 1, S, S): the per-batch-row padding
  // broadcasts down the query axis, the shared band broadcasts over the batch.
  let combined = mask.add(&overlay)?;
  // The HF fully-masked-row fix: a row whose max is still -inf has no visible
  // key — reset exactly those rows to fully-visible.
  let row_max = ops::reduction::max_axes(&combined, &[-1], true)?;
  let fully_masked = ops::comparison::isneginf(&row_max)?;
  let zero = ops::misc::astype(&Array::full::<f32>(&(1,), 0.0)?, dtype)?;
  ops::logical::select(&fully_masked, &zero, &combined)
}

#[cfg(all(test, feature = "embeddinggemma"))]
mod tests {
  use super::*;
  use crate::dtype::Dtype;

  /// Materialize `values` as an f16 `Array` (built in f32, cast to f16).
  fn f16_arr(values: &[f32]) -> Array {
    let f32_arr = Array::from_slice::<f32>(values, &(values.len(),)).unwrap();
    ops::misc::astype(&f32_arr, Dtype::F16).unwrap()
  }

  /// Cast `a` to f32, eval, and read it back (`to_vec` is dtype-strict, so an
  /// f16 array must be cast before it can be read as `f32`).
  fn to_f32(a: &Array) -> Vec<f32> {
    let mut a = ops::misc::astype(a, Dtype::F32).unwrap();
    a.eval().unwrap();
    a.to_vec::<f32>().unwrap()
  }

  #[test]
  fn clip_residual_saturates_f16_overflow_to_finite() {
    // Each operand is well within the f16 range (`f16::MAX` = 65504), but their
    // sum (120000) overflows f16. A bare f16 add would round to `inf`; the
    // reference `clip_residual` instead adds in f32 and clamps to `f16::MAX`.
    let bound = f32::from(half::f16::MAX);
    let x = f16_arr(&[60_000.0, -60_000.0, 1.0]);
    let y = f16_arr(&[60_000.0, -60_000.0, 2.0]);
    let out = clip_residual(&x, &y).expect("clip_residual");
    assert_eq!(out.dtype().unwrap(), Dtype::F16, "stays f16");
    let v = to_f32(&out);
    assert!(
      v.iter().all(|x| x.is_finite()),
      "f16 residual overflow must saturate to finite, got {v:?}"
    );
    // +overflow saturates to +f16::MAX, -overflow to -f16::MAX, the in-range
    // sum (3.0) is preserved (exactly representable in f16).
    assert_eq!(v[0], bound, "positive overflow clamps to +f16::MAX");
    assert_eq!(v[1], -bound, "negative overflow clamps to -f16::MAX");
    assert_eq!(v[2], 3.0, "in-range residual is unchanged");
  }

  #[test]
  fn clip_residual_f16_bare_add_would_overflow_to_infinity() {
    // Guard the test's own premise: a *bare* f16 add of the same operands does
    // round to infinity, so the saturation above is doing real work.
    let x = f16_arr(&[60_000.0]);
    let y = f16_arr(&[60_000.0]);
    let bare = x.add(&y).expect("bare f16 add");
    assert_eq!(bare.dtype().unwrap(), Dtype::F16);
    assert!(
      to_f32(&bare)[0].is_infinite(),
      "a bare f16 add of 60000+60000 overflows to inf"
    );
  }

  #[test]
  fn clip_residual_non_f16_is_a_plain_add() {
    // For f32 the helper must NOT clip — a sum past the f16 range stays as the
    // true f32 value (the reference returns `x + y` unchanged off the f16 path).
    let x = Array::from_slice::<f32>(&[60_000.0, 1.5], &(2usize,)).unwrap();
    let y = Array::from_slice::<f32>(&[60_000.0, 2.5], &(2usize,)).unwrap();
    let out = clip_residual(&x, &y).expect("clip_residual");
    assert_eq!(out.dtype().unwrap(), Dtype::F32, "f32 stays f32");
    let v = to_f32(&out);
    assert_eq!(v[0], 120_000.0, "f32 add is not clamped to the f16 range");
    assert_eq!(v[1], 4.0);
  }

  // ─────────────── bidirectional sliding-window mask helpers ───────────────

  #[test]
  fn sliding_window_overlay_matches_hf_band_semantics() {
    // S = 5, window = 2: visible (0) iff |i - j| < 2 — the strict HF
    // `_bidirectional_window_overlay` — and -inf everywhere else; (1,1,S,S).
    let overlay = build_sliding_window_overlay(5, 2, Dtype::F32).expect("overlay");
    assert_eq!(overlay.shape(), vec![1, 1, 5, 5]);
    let v = to_f32(&overlay);
    for i in 0..5usize {
      for j in 0..5usize {
        let got = v[i * 5 + j];
        if (i as i32 - j as i32).abs() < 2 {
          assert_eq!(got, 0.0, "({i},{j}) inside the band must be visible");
        } else {
          assert_eq!(
            got,
            f32::NEG_INFINITY,
            "({i},{j}) outside the band must be -inf"
          );
        }
      }
    }
  }

  #[test]
  fn sliding_window_overlay_boundary_is_exclusive() {
    // |i - j| == window is OUTSIDE the band (HF: `abs(q - kv) < sliding_window`
    // — strict). Pin both sides of the edge.
    let overlay = build_sliding_window_overlay(4, 3, Dtype::F32).expect("overlay");
    let v = to_f32(&overlay);
    assert_eq!(v[2], 0.0, "|0-2| = 2 < 3 is visible");
    assert_eq!(
      v[3],
      f32::NEG_INFINITY,
      "|0-3| = 3 == window is masked (exclusive edge)"
    );
  }

  #[test]
  fn sliding_window_overlay_is_identically_zero_when_seq_le_window() {
    // The mathematical basis for skipping the band at seq <= window: every
    // distance is `|i - j| <= seq - 1 < window`, so the overlay is the additive
    // identity (`softmax(x + 0) = softmax(x)`) — skipping is exact.
    for (s, w) in [(4, 4), (3, 4), (1, 1)] {
      let overlay = build_sliding_window_overlay(s, w, Dtype::F32).expect("overlay");
      let v = to_f32(&overlay);
      assert!(
        v.iter().all(|x| *x == 0.0),
        "seq {s} <= window {w} must be an all-zero overlay"
      );
    }
  }

  #[test]
  fn sliding_window_overlay_computes_band_in_f32_before_cast() {
    // bf16 cannot represent 511 (it rounds to 512): had the distance grid been
    // built in bf16, |i - j| = 511 would land on the window edge and the strict
    // `< 512` comparison would mask a visible position. The grid is computed in
    // f32 and only the finished {0, -inf} overlay is cast, so the real 512
    // window is exact in bf16 — and the cast preserves the requested dtype.
    let s = 515usize;
    let overlay = build_sliding_window_overlay(s as i32, 512, Dtype::BF16).expect("overlay");
    assert_eq!(overlay.dtype().unwrap(), Dtype::BF16);
    let v = to_f32(&overlay);
    assert_eq!(v[511], 0.0, "|0-511| = 511 < 512 must stay visible in bf16");
    assert_eq!(v[512], f32::NEG_INFINITY, "|0-512| = 512 is masked");
    assert_eq!(v[514 * s + 3], 0.0, "|514-3| = 511 < 512 must stay visible");
    assert_eq!(v[514 * s + 2], f32::NEG_INFINITY, "|514-2| = 512 is masked");
  }

  #[test]
  fn local_layer_mask_adds_band_to_padding_and_broadcasts_batch() {
    // Padding (2,1,1,4) — row 0 unpadded, row 1 pads the last key — + window 2
    // → (2,1,4,4): each row's padding must combine with the shared band.
    let pad =
      Array::from_slice::<f32>(&[1., 1., 1., 1., 1., 1., 1., 0.], &(2usize, 4usize)).unwrap();
    let additive = build_additive_mask(&pad, Dtype::F32).expect("padding mask");
    let local = build_local_layer_mask(&additive, 4, 2).expect("local mask");
    assert_eq!(local.shape(), vec![2, 1, 4, 4]);
    assert_eq!(local.dtype().unwrap(), Dtype::F32);
    let v = to_f32(&local);
    let at = |b: usize, q: usize, k: usize| v[b * 16 + q * 4 + k];
    // Batch row 0: pure band.
    assert_eq!(at(0, 0, 1), 0.0, "in-band real key visible");
    assert_eq!(
      at(0, 0, 2),
      f32::NEG_INFINITY,
      "out-of-band real key masked"
    );
    assert_eq!(at(0, 3, 3), 0.0, "diagonal always visible");
    // Batch row 1: key 3 is padding → masked even inside the band.
    assert_eq!(
      at(1, 3, 3),
      f32::NEG_INFINITY,
      "padded key masked inside the band"
    );
    assert_eq!(at(1, 3, 2), 0.0, "real in-band key visible on padded row");
    assert_eq!(at(1, 0, 1), 0.0);
    assert_eq!(at(1, 0, 3), f32::NEG_INFINITY, "padding + band both mask");
  }

  #[test]
  fn local_layer_mask_resets_fully_masked_padded_query_rows() {
    // real_len 1, window 2, seq 4: padded queries 2 and 3 have no real key
    // inside the band (distance to key 0 >= window) → without the HF
    // fully-masked-row reset they would softmax all-`-inf` (NaN). The reset
    // turns exactly those rows fully visible; rows with any visible key are
    // untouched.
    let pad = Array::from_slice::<f32>(&[1., 0., 0., 0.], &(1usize, 4usize)).unwrap();
    let additive = build_additive_mask(&pad, Dtype::F32).expect("padding mask");
    let local = build_local_layer_mask(&additive, 4, 2).expect("local mask");
    let v = to_f32(&local);
    let at = |q: usize, k: usize| v[q * 4 + k];
    // q0 (real): sees itself; other keys are padding → untouched row.
    assert_eq!(at(0, 0), 0.0);
    assert_eq!(at(0, 1), f32::NEG_INFINITY);
    // q1 (padded) still sees the real in-band key 0 (|1-0| = 1 < 2) → NOT reset.
    assert_eq!(at(1, 0), 0.0);
    assert_eq!(at(1, 3), f32::NEG_INFINITY);
    // q2 / q3 (padded, no visible key) → reset to all-visible.
    for q in 2..4usize {
      for k in 0..4usize {
        assert_eq!(
          at(q, k),
          0.0,
          "fully-masked padded query row must be reset to visible at ({q},{k})"
        );
      }
    }
  }

  #[test]
  fn local_layer_mask_keeps_model_dtype() {
    // The banded mask must stay in the model dtype end to end (the fast-SDPA
    // mask dtype rule): no silent f32 promotion from the band constants, the
    // row-reset zero, or the combine add.
    for dt in [Dtype::F16, Dtype::BF16] {
      let pad = Array::from_slice::<f32>(&[1.0; 6], &(1usize, 6usize)).unwrap();
      let additive = build_additive_mask(&pad, dt).expect("padding mask");
      let local = build_local_layer_mask(&additive, 6, 2).expect("local mask");
      assert_eq!(
        local.dtype().unwrap(),
        dt,
        "banded mask must stay {dt:?} (matching-dtype SDPA mask)"
      );
      // The {0, -inf} values survive the half-precision cast exactly.
      let v = to_f32(&local);
      assert_eq!(v[1], 0.0, "in-band visible");
      assert_eq!(v[2], f32::NEG_INFINITY, "out-of-band masked");
    }
  }
}
