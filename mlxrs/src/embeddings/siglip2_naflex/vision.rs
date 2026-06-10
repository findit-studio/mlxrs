//! SigLIP2 NaFlex vision tower.
//!
//! Ports `siglip.py`'s `VisionEmbeddings` / `SiglipVisionTransformer` /
//! `SiglipMultiheadAttentionPoolingHead`, specialized to the NaFlex
//! (native-resolution) variant and wiring the per-image position-embedding
//! interpolation `siglip.py` leaves as `NotImplementedError`. The shared
//! pre-norm encoder blocks (`Attention` / `EncoderLayer` / `MLP`) live in the
//! private `shared` module.
//!
//! ## Patch embedding — Linear, not Conv2d
//!
//! `siglip.py`'s `VisionEmbeddings.patch_embedding` is an
//! `nn.Conv2d(num_channels, hidden, kernel=patch, stride=patch)` applied to a
//! dense `(B, C, H, W)` image, then flattened to `(B, num_patches, hidden)`. A
//! convolution whose kernel **equals** its stride over a grid of
//! non-overlapping patches is exactly a per-patch
//! [`crate::ops::linalg_basic::matmul`]: each output patch is
//! `dot(flatten(patch), W) + b`. The NaFlex
//! [preprocessing][crate::embeddings::siglip2_naflex::processing] already emits
//! the *pre-flattened* `pixel_values (num_patches, P^2 * C)` tensor, so the
//! patch embed here is `pixel_values @ W_flat^T + bias` — a single matmul, no
//! Conv2d.
//!
//! The checkpoint weight key is `…embeddings.patch_embedding.weight`. A PyTorch
//! Conv2d weight is `(out, in, kH, kW) = (hidden, C, P, P)`; MLX is channels-last
//! `(out, kH, kW, in) = (hidden, P, P, C)`. The NaFlex preprocessing flattens
//! each patch in `(row, col, channel-innermost)` order — i.e. `(kH, kW, C)` — so
//! the matmul weight is that `(out, kH, kW, in)` tensor reshaped to
//! `(out, kH * kW * in) = (hidden, P^2 * C)`, row-for-row aligned with the
//! preprocessing's flattened patch rows. `reshape_patch_weight` accepts every
//! layout a real checkpoint ships — the MLX channels-last `(hidden, P, P, C)`,
//! the **raw PyTorch / HF** `(hidden, C, P, P)` (transposed `[0, 2, 3, 1]` to
//! channels-last, the way the crate's other Conv loaders convert PyTorch
//! kernels), and the already-flattened `(hidden, P^2 * C)` — pinning the consumed
//! tensor's exact dims in every case so a mis-shaped checkpoint fails fast.
//!
//! ## Position embedding — bilinear+antialias-resized per image
//!
//! `position_embedding` is an `nn.Embedding(num_positions, hidden)` whose
//! `num_positions = num_patches = 256` ⇒ a trained `16 x 16` grid. For NaFlex
//! each image has its own patch grid `(H_p, W_p)` (from `spatial_shapes`), so
//! the trained grid is resized to `(H_p, W_p)` via
//! [`bilinear_interpolate`](crate::ops::interpolation::bilinear_interpolate)
//! (PyTorch `F.interpolate(mode="bilinear", align_corners=False,
//! antialias=True)`) and added to the active patch embeds. This matches HF's
//! `Siglip2VisionEmbeddings.resize_positional_embeddings`, the source of the
//! parity oracle's fixtures (`siglip.py` stubs the interpolation out). Padded
//! patch rows (past `H_p * W_p`) are filled with the first resized position
//! (HF's `resized_positional_embeddings[0]`) and masked out of attention via
//! `pixel_attention_mask`, so their value is immaterial to the output.
//!
//! ## Attention masking (NaFlex divergence from `siglip.py`)
//!
//! `siglip.py`'s `SiglipVisionTransformer.__call__` hardcodes
//! `attention_mask = None` (it accepts `pixel_attention_mask` but explicitly
//! does **not** apply it — a documented gap). For NaFlex that is a correctness
//! bug: the padded patch rows would attend and be attended to, contaminating
//! every real patch's representation (and the attention-pool probe). This port
//! builds a **boolean** `(1, 1, 1, num_patches)` key mask from
//! `pixel_attention_mask` (`true` for real, `false` for padded) and passes it
//! to the bidirectional
//! [`crate::lm::nn::attention::scaled_dot_product_attention`], so padded
//! positions contribute nothing — matching the HF native-resolution
//! implementation the reference fixtures are generated from. Boolean rather
//! than additive float (see `build_attention_mask` for the dtype rationale):
//! the fused SDPA accepts a bool mask for any q/k/v float dtype, whereas an
//! additive mask must match the computation dtype — load-bearing now that
//! [`VisionTower::forward`] casts `pixel_values` to the model dtype.
//!
//! ## Input dtype (`pixel_values` cast at the forward entry)
//!
//! The NaFlex [preprocessing][crate::embeddings::siglip2_naflex::processing]
//! always emits an `F32` `pixel_values` (the image-processor contract, like
//! HF's), and MLX type promotion only **widens** (`f16 op f32 → f32`) — so
//! without a cast an f16/bf16 checkpoint would silently run the whole tower in
//! `F32`, `astype`-ing every reduced-precision weight up on every forward.
//! Both references cast at the vision entry — HF `modeling_siglip2.py`
//! (`Siglip2VisionEmbeddings.forward`: `target_dtype =
//! patch_embedding.weight.dtype; pixel_values.to(dtype=target_dtype)`) and
//! mlx-embeddings `siglip.py` (`get_image_features`:
//! `pixel_values.astype(patch_embedding.weight.dtype)`) — and
//! [`VisionTower::forward`] restores that cast (the patch-embed weight dtype;
//! an **affine**-quantized patch embed resolves its float `scales` dtype). A
//! same-dtype `astype` is a no-op, so an `F32` checkpoint pays nothing. The
//! resolution is **float-gated**: an fp-mode (`mxfp4` / `mxfp8` / `nvfp4`)
//! quantized patch embed carries packed e8m0 `uint8` scales — not a compute
//! precision — so no cast is applied there (the non-affine `quantized_matmul`
//! rejects an integer input; the F32 pixels pass through unchanged, the
//! pre-cast behavior).

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  embeddings::siglip2_naflex::{
    config::VisionConfig,
    processing::NaflexInputs,
    shared::{
      EncoderLayer, LayerDims, Mlp, QuantLinear, build_layer_norm, dim_i32, expect_shape, linear,
      resolve_quant, take, take_shaped,
    },
  },
  error::{
    CapExceededPayload, Error, OutOfRangePayload, RankMismatchPayload, Result,
    ShapePairMismatchPayload,
  },
  lm::{
    nn::{
      attention::{Mask, scaled_dot_product_attention},
      norm::LayerNorm,
    },
    quant::PerLayerQuantization,
  },
  model_validation::{checked_mul, require_positive, reserve_or_error},
  nn::MaybeQuantizedEmbedding,
  ops,
};

/// The flattened patch-embedding `Linear`: `pixel_values @ W^T + bias` —
/// quantize-aware.
///
/// The dense weight is the (reshaped) Conv2d kernel `(hidden, P^2 * C)`; the
/// quantized weight is the packed equivalent whose logical shape is the same
/// `(hidden, P^2 * C)`. See the [module docs](self) for why this is a matmul.
#[cfg(feature = "siglip2-naflex")]
struct PatchEmbed {
  proj: QuantLinear,
}

#[cfg(feature = "siglip2-naflex")]
impl PatchEmbed {
  /// `patch_embeds = pixel_values @ weight^T + bias`, over
  /// `(B, num_patches, P^2 * C) → (B, num_patches, hidden)`.
  fn forward(&self, pixel_values: &Array) -> Result<Array> {
    self.proj.forward(pixel_values)
  }
}

/// The multihead attention-pooling head (`siglip.py`'s
/// `SiglipMultiheadAttentionPoolingHead`): a learned `probe` query attends over
/// the encoded patch tokens (with the same padded-key mask), a residual MLP
/// refines it, and the single probe row is the pooled image embedding.
///
/// The probe→patch attention uses the **combined-`in_proj`** MHA form
/// (`siglip.py`'s `MHA`), where a single `in_proj.weight (3*hidden, hidden)`
/// holds the stacked `[q; k; v]` projections. The query (probe) uses the first
/// `hidden` rows, the key/value (patches) the remaining `2*hidden`.
#[cfg(feature = "siglip2-naflex")]
struct AttentionPoolHead {
  /// Learned probe `(1, 1, hidden)`.
  probe: Array,
  /// Combined QKV input projection weight `(3*hidden, hidden)` — held **dense**:
  /// `MHA.__call__` slices it by logical row (`in_proj.weight[:dims]` /
  /// `[dims:]`) to project the probe-query and patch-key/value separately, a
  /// weight-row-slice a packed quantized matrix cannot represent (so the
  /// reference never quantizes it — the attention-pool head is vision-only, and
  /// `quantize_model`'s default `skip_vision=True` leaves the whole vision tower
  /// dense). This field is therefore always the dense `(3*hidden, hidden)`
  /// matrix; a fully-quantized checkpoint that nonetheless carries the
  /// `in_proj.scales` sibling is dequantized to this dense matrix at load
  /// (`build_attention_pool_head`), so the runtime slice path is identical.
  in_proj_weight: Array,
  /// Combined QKV input projection bias `(3*hidden,)`.
  in_proj_bias: Array,
  /// Output projection — quantize-aware ([`QuantLinear`]).
  out_proj: QuantLinear,
  layernorm: LayerNorm,
  mlp: Mlp,
  num_heads: i32,
  head_dim: i32,
  hidden: i32,
  scale: f32,
}

#[cfg(feature = "siglip2-naflex")]
impl AttentionPoolHead {
  /// `hidden_state (B, L, C) → pooled (B, C)`, masking padded keys.
  ///
  /// Mirrors `SiglipMultiheadAttentionPoolingHead.__call__`:
  /// `probe = repeat(self.probe, B)`; `h = attention(probe, x, x)`;
  /// `h = h + mlp(layernorm(h))`; `return h[:, 0]`.
  fn forward(&self, hidden_state: &Array, mask: Mask<'_>) -> Result<Array> {
    let shape = hidden_state.shape();
    let bsz = dim_i32(&shape, 0, "siglip2 attn-pool: batch")?;
    let seq = dim_i32(&shape, 1, "siglip2 attn-pool: seq")?;

    // Repeat the probe across the batch: (1,1,C) → (B,1,C).
    let probe = ops::shape::broadcast_to(&self.probe, &[bsz, 1, self.hidden])?;

    // Split the combined in_proj into the q half (first `hidden` rows) and the
    // kv half (remaining `2*hidden` rows), mirroring `MHA.__call__`'s
    // `q_weight = in_proj.weight[:dims]`, `kv_weight = in_proj.weight[dims:]`.
    // `linear` computes `x @ W^T + b`, so a row-slice of the stacked weight is
    // the same as three separate projections.
    let q = self.proj_slice(&probe, 0)?; // (B, 1, C)
    let k = self.proj_slice(hidden_state, self.hidden)?; // (B, L, C)
    let v = self.proj_slice(hidden_state, self.hidden.saturating_mul(2))?; // (B, L, C)

    // (B, 1, C) → (B, n_heads, 1, head_dim); patches (B, L, C) → (B, n_heads, L, head_dim).
    let q = self.split_heads(&q, bsz, 1)?;
    let k = self.split_heads(&k, bsz, seq)?;
    let v = self.split_heads(&v, bsz, seq)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
    // (B, n_heads, 1, head_dim) → (B, 1, C).
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = checked_mul(
      "siglip2 attn-pool: num_heads * head_dim",
      "num_heads",
      self.num_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[bsz, 1, embed_dim])?;
    let h = self.out_proj.forward(&attn)?; // (B, 1, C)

    // residual = h; h = h + mlp(layernorm(h)).
    let normed = self.layernorm.forward(&h)?;
    let h = h.add(&self.mlp.forward(&normed)?)?; // (B, 1, C)

    // return h[:, 0] → (B, C).
    let pooled = ops::indexing::take_axis(&h, &index_zero()?, 1)?; // (B, 1, C)
    ops::shape::squeeze_axes(&pooled, &[1]) // (B, C)
  }

  /// Project `x` with the `hidden`-row slice of the combined `in_proj`
  /// starting at row `row_start` (q at 0, k at hidden, v at 2*hidden).
  fn proj_slice(&self, x: &Array, row_start: i32) -> Result<Array> {
    let w = slice_rows(&self.in_proj_weight, row_start, self.hidden)?;
    let b = slice_rows(&self.in_proj_bias, row_start, self.hidden)?;
    linear(x, &w, Some(&b))
  }

  /// `(B, L, C) → (B, n_heads, L, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

/// The NaFlex vision transformer: patch embed → per-image positional resize →
/// pre-norm encoder → post-LayerNorm → optional attention-pool head.
///
/// Ports `siglip.py`'s `SiglipVisionTransformer` (specialized to NaFlex). The
/// public [`Siglip2NaflexModel`](super::Siglip2NaflexModel) drives this with a
/// single image's [`NaflexInputs`].
#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
pub struct VisionTower {
  patch_embed: PatchEmbed,
  /// The trained position-embedding table `(num_positions, hidden)`.
  position_embedding: Array,
  /// `sqrt(num_positions)` — the square side of the trained position grid.
  pos_grid_side: i32,
  layers: Vec<EncoderLayer>,
  post_layernorm: LayerNorm,
  /// The attention-pool head, present iff `config.vision_use_head`.
  head: Option<AttentionPoolHead>,
  hidden: i32,
  /// The loaded config's runtime patch budget (`max_num_patches`).
  /// [`forward`](Self::forward) pins the runtime input's patch count to this
  /// before any embed / interp / attention (untrusted `NaflexInputs` has no
  /// validating constructor), so a hand-built input cannot exceed the loaded
  /// budget. This is a structural runtime-vs-config consistency check, not a
  /// magnitude cap.
  max_num_patches: usize,
  /// The loaded config's per-patch feature width (`P^2 * C`); `forward` pins
  /// the runtime `pixel_values` last axis to it.
  patch_feature_dim: usize,
  /// The model compute dtype — the patch-embed projection's **float**
  /// parameter dtype (the dense weight's dtype, or the `scales` dtype for an
  /// affine-quantized patch embed), resolved once at load.
  /// [`forward`](Self::forward) casts the (always-`F32`) runtime
  /// `pixel_values` to it so a reduced-precision checkpoint runs in its own
  /// precision (see the module docs). `None` when no float parameter dtype
  /// exists — an **fp-mode** (`mxfp4` / `mxfp8` / `nvfp4`) quantized patch
  /// embed carries packed e8m0 `uint8` scales, which are not a compute
  /// precision; `forward` then leaves the input dtype untouched (casting to
  /// `uint8` would be rejected by the non-affine `quantized_matmul`).
  dtype: Option<Dtype>,
}

#[cfg(feature = "siglip2-naflex")]
#[cfg_attr(docsrs, doc(cfg(feature = "siglip2-naflex")))]
impl VisionTower {
  /// Build the vision tower from a validated [`VisionConfig`] and the
  /// (sanitized) weight map with the `vision_model.vision_model.` prefix
  /// already stripped (so keys are e.g. `embeddings.patch_embedding.weight`,
  /// `encoder.layers.0.self_attn.q_proj.weight`, `post_layernorm.weight`,
  /// `head.probe`, …, matching `siglip.py`'s module tree).
  ///
  /// Every consumed tensor's shape is pinned to its exact config-derived
  /// dimensions (typed [`Error::ShapePairMismatch`] wrapped in
  /// [`Error::LayerKeyed`]) so a corrupt / hostile checkpoint that passes
  /// config validation cannot run a different graph or drive an oversized
  /// allocation.
  pub fn from_weights(config: &VisionConfig, weights: &mut HashMap<String, Array>) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build the vision tower with an optional parsed quantization config — the
  /// quantization-aware analogue of [`from_weights`](Self::from_weights) (which
  /// is just this with `quant == None`).
  ///
  /// Each `nn.Linear` (the attention q/k/v/out, the MLP fc1/fc2, the patch-embed
  /// matmul, and the attention-pool out_proj) auto-picks the dense or quantized
  /// variant per layer by the presence of a `<prefix>.scales` sibling (the
  /// `class_predicate`'s `f"{p}.scales" in weights` signal). The position
  /// embedding is materialized to a dense table at load (dequantized once if
  /// quantized) because the per-image positional resize reshapes it to a grid and
  /// bilinear-interpolates it — a packed quantized table cannot be reshaped /
  /// interpolated, so the dense materialization keeps the resize path unchanged.
  ///
  /// A standard mlx-community quantized SigLIP2 checkpoint is produced with
  /// `skip_vision=True` (`quantize_model`'s default), so in practice the vision
  /// tower's tensors carry no `.scales` siblings and every layer loads dense; the
  /// per-layer auto-detect nonetheless loads quantized whichever vision layers a
  /// checkpoint did quantize.
  pub fn from_weights_quantized(
    config: &VisionConfig,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: `from_weights` is public, so a caller may
    // build a tower from a directly-constructed (unvalidated) config. This
    // bounds `num_hidden_layers` (and every other dim) before the per-layer
    // reservation/loop below.
    config.validate()?;
    let hidden = config.hidden_size;
    let patch = config.patch_size;
    let channels = config.num_channels;
    let inter = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    // Per-layer shape constants (validates num_heads positive + divides hidden,
    // and computes the head split / SDPA scale once).
    let dims = LayerDims::new(hidden, inter, num_heads, config.layer_norm_eps as f32)?;
    let eps = dims.eps;
    let num_positions = config.num_patches()?;
    let patch_feat = config.patch_feature_dim()?; // P^2 * C
    // The loaded config's runtime budget — `forward` pins the untrusted
    // runtime `NaflexInputs` against these. `config.validate` above already
    // required both positive (and overflow-checked `patch_feat`), so these are
    // positive dimensions the runtime input is checked against.
    let max_num_patches = config.max_num_patches() as usize;
    let patch_feature_dim = patch_feat as usize;

    // ── patch embedding (the Conv2d-as-Linear kernel) ──
    // Quantized path (the `.scales` sibling alone is the `class_predicate`
    // signal, as in the shared loaders): the packed weight is already logical-2D
    // `(hidden, P^2*C)` — no Conv2d reshape — so it takes the auto-detecting
    // `QuantLinear::from_weights`, which resolves the scheme (or returns the
    // typed `.scales`-without-params error) and structurally gates to the
    // config-derived `(hidden, P^2*C)`. Dense path: reshape the Conv2d kernel to
    // `(hidden, P^2*C)` (accepting every checkpoint layout) and wrap dense.
    let patch_embed = if weights.contains_key("embeddings.patch_embedding.scales") {
      PatchEmbed {
        proj: QuantLinear::from_weights(
          weights,
          "embeddings.patch_embedding",
          hidden,
          patch_feat,
          true,
          quant,
        )?,
      }
    } else {
      let patch_weight_raw = take(weights, "embeddings.patch_embedding.weight")?;
      let patch_weight =
        reshape_patch_weight(&patch_weight_raw, hidden, patch, channels, patch_feat)?;
      let patch_bias = take_shaped(
        weights,
        "embeddings.patch_embedding.bias",
        "vision patch-embed bias (hidden,)",
        &[hidden],
      )?;
      PatchEmbed {
        proj: QuantLinear::dense(patch_weight, Some(patch_bias)),
      }
    };
    // The model compute dtype: the patch-embed projection's FLOAT parameter
    // dtype (dense weight dtype / affine-quantized scales dtype) — the same
    // resolution the references use for the pixel-values cast (HF
    // `modeling_siglip2.py`'s `target_dtype = patch_embedding.weight.dtype`;
    // mlx-embeddings `siglip.py`'s `patch_embedding.weight.dtype`). `None`
    // for an fp-mode quantized patch embed (uint8 e8m0 scales — not a compute
    // precision; `forward` then skips the cast). Resolved once here so
    // `forward` pays no per-call metadata lookup.
    let dtype = patch_embed.proj.param_dtype()?;

    // ── position embedding table ──
    // Built quantize-aware then materialized to a dense `(num_positions, hidden)`
    // table (dequantized once if quantized): the per-image positional resize
    // reshapes + bilinear-interpolates this table, which a packed quantized table
    // cannot represent. The materialized table is shape-pinned to the config
    // (the dense `expect_shape` gate the original load enforced), covering both
    // the dense and the dequantized-quantized cases.
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
        "vision position-embedding table (num_positions, hidden)",
        &[num_positions, hidden],
      )?;
      table
    };
    let pos_grid_side = isqrt_exact(num_positions, "VisionConfig: num_patches")?;

    // ── encoder layers ──
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

    // ── post-LayerNorm ──
    let post_layernorm = build_layer_norm(weights, "post_layernorm", hidden, eps)?;

    // ── optional attention-pool head ──
    let head = if config.vision_use_head {
      Some(build_attention_pool_head(weights, dims, quant)?)
    } else {
      None
    };

    Ok(Self {
      patch_embed,
      position_embedding,
      pos_grid_side,
      layers,
      post_layernorm,
      head,
      hidden,
      max_num_patches,
      patch_feature_dim,
      dtype,
    })
  }

  /// Whether the attention-pool head is present (`config.vision_use_head`).
  #[inline(always)]
  pub fn has_head(&self) -> bool {
    self.head.is_some()
  }

  /// Forward one image's [`NaflexInputs`] through the tower.
  ///
  /// Returns `(last_hidden, pooled)`:
  /// - `last_hidden` — the post-LayerNorm patch hidden states
  ///   `(1, num_patches, hidden)`.
  /// - `pooled` — `Some((1, hidden))` from the attention-pool head when
  ///   present, else `None` (mirroring `siglip.py`'s
  ///   `pooler_output = head(x) if use_head else None`).
  ///
  /// The leading batch dim is `1` (one image per call); the foundation's
  /// `NaflexInputs` carries a single image's `(max_num_patches, …)` tensors,
  /// unsqueezed to `(1, max_num_patches, …)` here.
  pub fn forward(&self, inputs: &NaflexInputs) -> Result<(Array, Option<Array>)> {
    // Pin EVERY runtime input to its EXACT shape before any op. `NaflexInputs`
    // is a public, all-pub-fields struct with no validating constructor, so a
    // hand-built input (or an over-budget `preprocess` call) can carry not just
    // a too-large patch dim but a wrong *rank* (a rank-3 `pixel_values`) or a
    // trailing junk axis (a `(patch_dim, huge)` mask) that a leading-dim-only
    // check waves through into the position-embed scatter / the O(patch^2)
    // SDPA. Checking the full runtime shapes here (rank + every dim) — against
    // the stored config budget — rejects all of those as crisp typed errors
    // up front, rather than relying on a downstream op's incidental broadcast.
    let pv_shape = inputs.pixel_values.shape();
    // `pixel_values` is rank-2 `(patch_dim, patch_feature_dim)`. Read the patch
    // dim from a *verified* rank-2 shape, so a rank-3 input is rejected (not
    // silently treated as `shape[0]`).
    if pv_shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "siglip2 vision: pixel_values must be rank-2 (patch_dim, patch_feature_dim)",
        pv_shape.len() as u32,
        pv_shape,
      )));
    }
    let patch_dim = pv_shape[0];

    // Cap the runtime patch count against the loaded budget BEFORE pinning the
    // exact shape, so an over-cap patch dim surfaces as the dedicated
    // `CapExceeded` (the DoS boundary) rather than a generic shape mismatch.
    if patch_dim > self.max_num_patches {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "siglip2 vision: runtime patch count vs loaded max_num_patches",
        "max_num_patches",
        self.max_num_patches as u64,
        patch_dim as u64,
      )));
    }

    // `pixel_values` last axis must equal the loaded `patch_feature_dim` (the
    // patch-embed Linear's input width). Pins the full `(patch_dim, P^2*C)`.
    expect_runtime_shape(
      &pv_shape,
      "siglip2 vision: pixel_values shape (patch_dim, patch_feature_dim)",
      &[patch_dim, self.patch_feature_dim],
    )?;
    // `pixel_attention_mask` is exactly rank-1 `(patch_dim,)` — a trailing junk
    // axis (`(patch_dim, huge)`) or a length mismatch is rejected here, not by
    // the SDPA broadcast on the resulting `(1,1,1,n)` mask.
    expect_runtime_shape(
      &inputs.pixel_attention_mask.shape(),
      "siglip2 vision: pixel_attention_mask shape (patch_dim,)",
      &[patch_dim],
    )?;
    // `spatial_shapes` is exactly rank-1 `(2,)` — `[H_p, W_p]`. Pinned here
    // before any op (the host-side `read_spatial_shape` re-checks defensively).
    expect_runtime_shape(
      &inputs.spatial_shapes.shape(),
      "siglip2 vision: spatial_shapes shape (2,)",
      &[2],
    )?;

    // Cast the runtime pixel_values to the model dtype FIRST — the reference
    // does this at the vision entry (HF `modeling_siglip2.py`:
    // `pixel_values.to(dtype=patch_embedding.weight.dtype)`; mlx-embeddings
    // `siglip.py`: `pixel_values.astype(...weight.dtype)`). The preprocessing
    // always emits `F32`, and MLX promotion only widens (`f16 op f32 → f32`),
    // so without this cast an f16/bf16 checkpoint silently runs the WHOLE
    // tower — patch embed, encoder, attention pool — in `F32`. A same-dtype
    // astype is a no-op, so an `F32` checkpoint (or a pre-cast input) pays
    // nothing. The key mask below is BOOLEAN precisely so it stays valid for
    // any model dtype (see `build_attention_mask`).
    let pixel_values = match self.dtype {
      Some(dt) => inputs.pixel_values.astype(dt)?,
      // No float parameter dtype to pin to (an fp-mode quantized patch embed —
      // uint8 e8m0 scales): leave the input dtype untouched, the pre-cast
      // behavior. The non-affine `quantized_matmul` itself requires a float
      // `x`, so casting to the scales' integer dtype would error; `try_clone`
      // is a cheap refcounted handle copy (no data copy).
      None => inputs.pixel_values.try_clone()?,
    };
    // (max_num_patches, P^2*C) → (1, max_num_patches, P^2*C).
    let pixel_values = ops::shape::expand_dims_axes(&pixel_values, &[0])?;
    // (1, max_num_patches, hidden).
    let patch_embeds = self.patch_embed.forward(&pixel_values)?;

    // Per-image position embedding: read this image's (H_p, W_p), bilinear-
    // resize the trained square grid to it, scatter into the active patch rows
    // (the first H_p*W_p), and add. Padded rows take the first resized position
    // (HF) and are masked out of attention anyway.
    let (h_p, w_p) = read_spatial_shape(&inputs.spatial_shapes)?;
    let pos = self.resized_position_embedding(h_p, w_p, &patch_embeds.shape())?;
    let mut h = patch_embeds.add(&pos)?;

    // Boolean key mask from pixel_attention_mask: true for real, false for
    // padded, broadcast to (1, 1, 1, num_patches) for SDPA — dtype-agnostic,
    // so it pairs with the model-dtype hidden states from the cast above.
    let attn_mask = build_attention_mask(&inputs.pixel_attention_mask)?;
    let mask = Mask::Array(&attn_mask);

    for layer in &self.layers {
      h = layer.forward(&h, mask)?;
    }
    let last_hidden = self.post_layernorm.forward(&h)?;

    let pooled = match &self.head {
      Some(head) => Some(head.forward(&last_hidden, mask)?),
      None => None,
    };
    Ok((last_hidden, pooled))
  }

  /// Build the per-image positional-embedding term `(1, num_patches, hidden)`:
  /// bilinear+antialias-resize the trained `(pos_grid_side, pos_grid_side,
  /// hidden)` grid to `(H_p, W_p, hidden)`, flatten to `(H_p*W_p, hidden)`, and
  /// right-pad to `num_patches` by repeating the first resized position
  /// (matching HF's `resized_positional_embeddings[0]` padding). The padded
  /// patches' attention is masked out, so the padding value is immaterial to
  /// the output; it is filled faithfully nonetheless.
  fn resized_position_embedding(
    &self,
    h_p: i32,
    w_p: i32,
    patch_embeds_shape: &[usize],
  ) -> Result<Array> {
    let num_patches = dim_i32(patch_embeds_shape, 1, "siglip2 vision: num_patches")?;
    require_positive("siglip2 vision: H_patch", h_p)?;
    require_positive("siglip2 vision: W_patch", w_p)?;
    let active = checked_mul(
      "siglip2 vision: H_patch * W_patch",
      "H_patch",
      h_p,
      "W_patch",
      w_p,
    )?;
    if active > num_patches {
      // The active grid cannot exceed the patch budget — the preprocessing
      // guarantees H_p*W_p <= max_num_patches. A violation means a mismatched
      // spatial_shapes / pixel_values pair.
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "siglip2 vision: active patches (H_p * W_p)",
        "must not exceed num_patches (the pixel_values leading patch dim)",
        smol_str::format_smolstr!("{active} > {num_patches}"),
      )));
    }

    // (num_positions, hidden) → (side, side, hidden) → bilinear+antialias
    // (H_p, W_p, hidden).
    let side = self.pos_grid_side as usize;
    let hidden = self.hidden as usize;
    let grid = ops::shape::reshape(&self.position_embedding, &(side, side, hidden))?;
    let resized =
      crate::ops::interpolation::bilinear_interpolate(&grid, h_p as usize, w_p as usize)?;
    // (H_p, W_p, hidden) → (H_p*W_p, hidden).
    let active_pos = ops::shape::reshape(&resized, &[active, self.hidden])?;

    // Right-pad to (num_patches, hidden) by repeating the FIRST resized
    // position (`resized_embeddings[0]`), matching HF's
    // `resize_positional_embeddings`. The padded rows' attention is masked out
    // by `pixel_attention_mask`, so this value is immaterial to the output; it
    // is filled faithfully nonetheless (rather than with zeros).
    let padded = if active < num_patches {
      let pad_rows = num_patches - active;
      // active_pos[0:1] → broadcast to (pad_rows, hidden).
      let first = ops::indexing::take_axis(&active_pos, &index_zero()?, 0)?; // (1, hidden)
      let pad_block = ops::shape::broadcast_to(&first, &[pad_rows, self.hidden])?;
      ops::shape::concatenate(&[&active_pos, &pad_block], 0)?
    } else {
      active_pos
    };
    // (num_patches, hidden) → (1, num_patches, hidden).
    ops::shape::expand_dims_axes(&padded, &[0])
  }
}

// ───────────────────────── builders / helpers ─────────────────────────

/// Assert a runtime input's shape (rank + every dimension) equals `expected`
/// exactly, before any op consumes it. A wrong rank is [`Error::RankMismatch`];
/// a right-rank shape whose dims disagree is [`Error::ShapePairMismatch`] (both
/// full shapes). This is the runtime-input analogue of the load-time
/// [`expect_shape`] (which pins consumed *weights*): the untrusted
/// [`NaflexInputs`] has no validating constructor, so [`VisionTower::forward`]
/// pins each tensor's exact shape here rather than trusting a leading-dim check.
#[cfg(feature = "siglip2-naflex")]
fn expect_runtime_shape(actual: &[usize], context: &'static str, expected: &[usize]) -> Result<()> {
  if actual.len() != expected.len() {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      context,
      actual.len() as u32,
      actual.to_vec(),
    )));
  }
  if actual != expected {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      context,
      expected.to_vec(),
      actual.to_vec(),
    )));
  }
  Ok(())
}

/// Build a **boolean** attention key-mask `(1, 1, 1, num_patches)` from the
/// `(max_num_patches,)` `pixel_attention_mask` (`1` real, `0` padded): real →
/// `true` (attend), padded → `false` (masked out), so SDPA's softmax zeroes
/// padded keys.
///
/// Boolean rather than the previous additive `0 / -inf` float mask, because
/// the mask must stay valid for **any** model dtype now that
/// [`VisionTower::forward`] casts `pixel_values` to the checkpoint precision:
///
/// - the fused SDPA REJECTS an additive mask whose dtype does not promote to
///   the computation dtype (`mlx/fast.cpp`'s
///   `promote_types(mask, final_type) != final_type` → "Mask type must promote
///   to output type") — the old `F32` additive mask would error against
///   f16/bf16 hidden states, and avoiding that with an additive mask would
///   couple the mask construction (its `-inf` constant and dtype) to the model
///   dtype;
/// - a bool mask is exempt from that check and natively supported on every
///   path: the fused kernel takes it directly when
///   `supports_bool_mask()` (else mlx converts it to the `0 / -inf` form **in
///   the computation dtype** itself), and the unfused fallback applies
///   `where(mask, scores, finfo(scores.dtype).min)` — numerically identical
///   for the softmax (a masked key's weight is exactly `0` either way, as with
///   the old additive mask).
#[cfg(feature = "siglip2-naflex")]
fn build_attention_mask(pixel_attention_mask: &Array) -> Result<Array> {
  // (num_patches,) i32 mask → bool (nonzero = real / attend).
  let mask_bool = ops::misc::astype(pixel_attention_mask, Dtype::Bool)?;
  let shape = pixel_attention_mask.shape();
  let n = dim_i32(&shape, 0, "siglip2 vision: mask length")?;
  // (num_patches,) → (1, 1, 1, num_patches) to broadcast over (B, heads, L_q, L_kv).
  ops::shape::reshape(&mask_bool, &[1, 1, 1, n])
}

/// Reshape the patch-embed weight into the flattened Linear kernel
/// `(hidden, P^2 * C)`. Accepts three checkpoint layouts (rank disambiguated, and
/// the rank-4 layout disambiguated by its exact dims against the config):
///
/// 1. the MLX channels-last Conv2d form `(hidden, P, P, C)` — flatten the
///    trailing three axes directly (the form the e2e fixtures carry);
/// 2. the **raw PyTorch / HF** Conv2d form `(hidden, C, P, P)` (`nn.Conv2d`'s
///    `(out, in, kH, kW)`) — transpose `[0, 2, 3, 1]` to channels-last
///    `(hidden, P, P, C)`, then flatten. This mirrors how the crate's other
///    Conv loaders convert PyTorch's channels-first kernels to MLX channels-last
///    (the wav2vec2 audio encoder's conv1d axis-swap, and the vlm
///    `reshape(1,h,w,3).transpose(0,3,1,2)` image path);
/// 3. an already-flattened `(hidden, P^2 * C)` (rank-2).
///
/// Any other shape is a typed error. When `P == C` the two rank-4 layouts are
/// shape-identical; the channels-last interpretation (no transpose) is chosen,
/// matching the MLX-native / fixture layout (a square `P == C` kernel is not a
/// real SigLIP2 configuration — base is `P = 16`, `C = 3`).
#[cfg(feature = "siglip2-naflex")]
fn reshape_patch_weight(
  raw: &Array,
  hidden: i32,
  patch: i32,
  channels: i32,
  patch_feat: i32,
) -> Result<Array> {
  let shape = raw.shape();
  match shape.len() {
    4 => {
      // Disambiguate the two rank-4 layouts by exact dims against the config (no
      // heuristic): channels-last `(hidden, P, P, C)` is the MLX-native form and
      // is preferred; the raw PyTorch `(hidden, C, P, P)` is transposed to it.
      let channels_last = [hidden, patch, patch, channels];
      let pytorch = [hidden, channels, patch, patch];
      let dims_eq = |expected: &[i32; 4]| {
        shape.len() == 4
          && shape
            .iter()
            .zip(expected.iter())
            .all(|(&a, &e)| e >= 0 && a as i64 == i64::from(e))
      };
      if dims_eq(&channels_last) {
        // (hidden, P, P, C) — flatten the trailing three axes directly.
        ops::shape::reshape(raw, &[hidden, patch_feat])
      } else if dims_eq(&pytorch) {
        // Raw PyTorch `(hidden, C, P, P)` -> channels-last `(hidden, P, P, C)`
        // (`transpose(0, 2, 3, 1)`), then flatten to the Linear kernel. The
        // transpose makes the flattened rows align with the preprocessing's
        // `(row, col, channel-innermost)` patch flattening.
        let channels_last = ops::shape::transpose_axes(raw, &[0, 2, 3, 1])?;
        ops::shape::reshape(&channels_last, &[hidden, patch_feat])
      } else {
        // Rank-4 but neither layout's dims match — pin against the channels-last
        // form for a crisp `(hidden, P, P, C)` shape error (both full shapes).
        expect_shape(
          raw,
          "embeddings.patch_embedding.weight",
          "vision patch-embed conv weight (hidden, P, P, C) or raw PyTorch (hidden, C, P, P)",
          &channels_last,
        )?;
        // `expect_shape` always errors here (the dims did not match); the flatten
        // is unreachable but keeps the arm's type as the flattened kernel.
        ops::shape::reshape(raw, &[hidden, patch_feat])
      }
    }
    2 => {
      // Already flattened (hidden, P^2 * C).
      expect_shape(
        raw,
        "embeddings.patch_embedding.weight",
        "vision patch-embed flattened weight (hidden, P^2 * C)",
        &[hidden, patch_feat],
      )?;
      raw.try_clone()
    }
    _ => Err(Error::RankMismatch(RankMismatchPayload::new(
      "embeddings.patch_embedding.weight must be rank-4 (hidden, P, P, C) / (hidden, C, P, P) or rank-2 (hidden, P^2*C)",
      shape.len() as u32,
      shape,
    ))),
  }
}

/// Build the attention-pool head (`head.*`). The `out_proj` and the residual
/// `mlp` are quantize-aware; the combined-QKV `in_proj` is consumed dense (it is
/// sliced by logical row — see [`AttentionPoolHead::in_proj_weight`]), but a
/// quantized checkpoint that ships its `.scales` sibling is dequantized to that
/// dense `(3*hidden, hidden)` matrix at load (never run dense with the siblings
/// ignored).
#[cfg(feature = "siglip2-naflex")]
fn build_attention_pool_head(
  weights: &mut HashMap<String, Array>,
  dims: LayerDims,
  quant: Option<&PerLayerQuantization>,
) -> Result<AttentionPoolHead> {
  let hidden = dims.hidden;
  let three_h = checked_mul(
    "siglip2 attn-pool: 3 * hidden",
    "three",
    3,
    "hidden",
    hidden,
  )?;
  let probe = take_shaped(
    weights,
    "head.probe",
    "attn-pool probe (1, 1, hidden)",
    &[1, 1, hidden],
  )?;
  // The combined-QKV `in_proj` weight is consumed DENSE (sliced by logical row;
  // a packed quantized matrix cannot serve a row slice). A dense checkpoint
  // ships it as the dense `(3*hidden, hidden)` matrix; a fully-quantized
  // checkpoint instead ships the packed `(weight, scales, biases)` triple under
  // the SAME `.scales`-sibling signal every other quantized weight uses. When
  // that sibling is present, dequantize the triple to the dense `(3*hidden,
  // hidden)` matrix through the shared quantize-aware path (the same materialize
  // -then-slice strategy the position-embedding table uses above), so a
  // quantized `in_proj` loads correctly while staying logically dense; the
  // separate dense `in_proj.bias` (the MHA additive bias, distinct from the
  // per-group quantization `biases`) is loaded the same way in both cases.
  let in_proj_weight = if weights.contains_key("head.attention.in_proj.scales") {
    let packed = MaybeQuantizedEmbedding::from_weights(
      weights,
      "head.attention.in_proj",
      resolve_quant(quant, "head.attention.in_proj"),
    )?;
    let table = packed.dense_table(None)?;
    expect_shape(
      &table,
      "head.attention.in_proj.weight",
      "attn-pool in_proj weight (3*hidden, hidden)",
      &[three_h, hidden],
    )?;
    table
  } else {
    take_shaped(
      weights,
      "head.attention.in_proj.weight",
      "attn-pool in_proj weight (3*hidden, hidden)",
      &[three_h, hidden],
    )?
  };
  let in_proj_bias = take_shaped(
    weights,
    "head.attention.in_proj.bias",
    "attn-pool in_proj bias (3*hidden,)",
    &[three_h],
  )?;
  let out_proj = QuantLinear::from_weights(
    weights,
    "head.attention.out_proj",
    hidden,
    hidden,
    true,
    quant,
  )?;
  let layernorm = build_layer_norm(weights, "head.layernorm", hidden, dims.eps)?;
  let mlp = Mlp::from_weights(weights, "head.mlp", hidden, dims.intermediate, quant)?;
  Ok(AttentionPoolHead {
    probe,
    in_proj_weight,
    in_proj_bias,
    out_proj,
    layernorm,
    mlp,
    num_heads: dims.num_heads,
    head_dim: dims.head_dim,
    hidden,
    scale: dims.scale,
  })
}

/// Slice `count` contiguous rows from `mat` starting at row `start` (axis 0).
/// `mat` is rank-1 (a bias vector) or rank-2 (a weight `(rows, cols)`); the
/// trailing axes pass through.
#[cfg(feature = "siglip2-naflex")]
fn slice_rows(mat: &Array, start: i32, count: i32) -> Result<Array> {
  let shape = mat.shape();
  let rank = shape.len();
  let mut lo = vec![0i32; rank];
  let mut hi: Vec<i32> = shape
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "siglip2 attn-pool: in_proj dim",
          "exceeds i32::MAX",
          smol_str::format_smolstr!("{d}"),
        ))
      })
    })
    .collect::<Result<Vec<_>>>()?;
  lo[0] = start;
  hi[0] = start.saturating_add(count);
  let strides = vec![1i32; rank];
  ops::indexing::slice(mat, &lo, &hi, &strides)
}

/// The `(1,)` i32 index array `[0]`, for `take_axis(h, axis=1)` (selecting the
/// probe row) — the lazy analogue of `h[:, 0]`.
#[cfg(feature = "siglip2-naflex")]
fn index_zero() -> Result<Array> {
  Array::from_slice::<i32>(&[0], &(1usize,))
}

/// Read this image's `(H_p, W_p)` from the `(2,)` i32 `spatial_shapes`.
/// Evaluates the tiny 2-element array to host integers (the resize geometry is
/// host-side: the bilinear weight matrices are built on the host).
#[cfg(feature = "siglip2-naflex")]
fn read_spatial_shape(spatial_shapes: &Array) -> Result<(i32, i32)> {
  let shape = spatial_shapes.shape();
  if shape.as_slice() != [2] {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "siglip2 vision: spatial_shapes must be (2,) [H_p, W_p]",
      shape.len() as u32,
      shape,
    )));
  }
  let mut s = spatial_shapes.try_clone()?;
  s.eval()?;
  let v = s.to_vec::<i32>()?;
  Ok((v[0], v[1]))
}

/// Exact integer square root: returns `r` with `r*r == n`, else a typed error.
/// Used to derive the trained position grid's square side from `num_positions`.
#[cfg(feature = "siglip2-naflex")]
fn isqrt_exact(n: i32, context: &'static str) -> Result<i32> {
  require_positive(context, n)?;
  let r = (n as f64).sqrt().round() as i32;
  if r.saturating_mul(r) != n {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "num_positions must be a perfect square (the trained position grid is square)",
      smol_str::format_smolstr!("{n}"),
    )));
  }
  Ok(r)
}

#[cfg(all(test, feature = "siglip2-naflex"))]
mod tests;
