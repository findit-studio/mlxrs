//! LFM2.5-VL SigLIP2-style vision tower.
//!
//! Faithful 1:1 port of `mlx-vlm/mlx_vlm/models/lfm2_vl/vision.py` — a
//! native-resolution SigLIP2 ViT specialized to the LFM2.5-VL checkpoint, with
//! every `nn.Linear` routed through the shared quantize-aware
//! [`MaybeQuantizedLinear`] so the 8-bit
//! `LiquidAI/LFM2.5-VL-450M-MLX-8bit` checkpoint loads through the same code
//! path as a dense one (the `<prefix>.scales` sibling is the load-bearing
//! "this layer is quantized" signal).
//!
//! ## Patch embedding — Linear, not Conv2d
//!
//! `vision.py`'s `VisionEmbeddings.patch_embedding` is an
//! `nn.Linear(num_channels * patch_size^2, hidden)` applied to the processor's
//! **pre-flattened** `(N, num_patches, num_channels * patch_size^2)` patches —
//! NOT a `Conv2d` over a dense image. The patch embed here is therefore a
//! single (dense-or-quantized) Linear: `pixel_values @ W^T (+ b)`.
//!
//! ## Position embedding — bicubic-resized per image
//!
//! `position_embedding` is an `nn.Embedding(num_patches, hidden)` whose
//! `num_patches = 256` ⇒ a trained `16 x 16` grid. For each image the trained
//! grid is reshaped to `(side, side, hidden)`, transposed to the
//! `(1, hidden, side, side)` layout [`bicubic_interpolate`] expects, resized to
//! that image's `(H_patch, W_patch)`, flattened back to
//! `(H_patch * W_patch, hidden)`, and added to the patch embeds — mirroring
//! `vision.py`'s `resize_positional_embeddings`. Rows past `H_patch * W_patch`
//! (padding) take the first resized position, exactly as the reference does
//! (`resulted_positional_embeddings[i, h*w:] = resized[0]`).
//!
//! ## Attention — bidirectional, no RoPE, no causal mask
//!
//! `vision.py`'s `Attention` is the bias=True `q/k/v/out_proj` form feeding
//! `mx.fast.scaled_dot_product_attention(..., scale=head_dim**-0.5, mask=None)`
//! — full bidirectional attention with no rotary embedding and no causal mask
//! (`VisionModel.__call__` passes `mask=None`). The pre-norm encoder layer is
//! `h = x + attn(ln1(x)); out = h + mlp(ln2(h))` with `LayerNorm(eps =
//! layer_norm_eps)` and an `fc1 -> GELU(precise) -> fc2` MLP (`fc*` biased).
//!
//! ## Encoder truncation
//!
//! `VisionModel` truncates `encoder.layers` to `vision_feature_layer + 1`
//! (`-1` ⇒ keep all). The caller resolves the kept count from the
//! [`ModelConfig`](super::config::ModelConfig); this module's
//! [`VisionModel::from_weights`] takes the already-resolved count and only
//! builds (and only requires the weights for) that many layers.
//!
//! ## Sanitize
//!
//! `vision.py`'s `sanitize` drops any `position_ids` key (a non-parameter
//! buffer). The VL-level key renames (`vision_encoder -> encoder`, …) belong to
//! the later VL-model phase; this module loads from its own already-sanitized
//! prefixes ([`VisionModel::sanitize`] applies the `position_ids` drop).

use std::collections::HashMap;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  lm::nn::{
    attention::{Mask, scaled_dot_product_attention},
    norm::LayerNorm,
  },
  model_validation::{checked_mul, require_positive, reserve_or_error},
  nn::MaybeQuantizedLinear,
  ops::{self, interpolation::bicubic_interpolate},
  vlm::models::lfm2_vl::config::VisionConfig,
};

/// A per-layer quantization resolver: maps a layer's module path to its
/// `(group_size, bits, mode)` scheme (or `None` when that layer is dense). The
/// mode string's lifetime `'q` is the parsed config's, **independent** of the
/// queried path's lifetime (the resolver borrows the mode from the config, not
/// from the path) — the same shape as the LFM2 LM's `quant_for` closure. A
/// resolver returning `None` everywhere loads a dense checkpoint unchanged.
type QuantResolver<'q> = dyn Fn(&str) -> Option<(i32, i32, &'q str)> + 'q;

/// mlx's `nn.LayerNorm` default `eps` (`1e-5`). `vision.py`'s `post_layernorm`
/// is built as `nn.LayerNorm(hidden)` with NO explicit `eps`, so it uses this
/// default — distinct from the encoder layers' `layer_norm1/2`, which pass
/// `eps = config.layer_norm_eps` (`1e-6`).
const LAYERNORM_DEFAULT_EPS: f32 = 1e-5;

/// The per-layer constants the encoder stack shares — the head split, the SDPA
/// scale, and the LayerNorm eps. Bundled so each layer builder takes one arg,
/// and so the head split / scale are computed once. (The transformer / FF
/// widths are not carried here: the quantize-aware
/// [`MaybeQuantizedLinear::from_weights`] reads each projection's shape from the
/// checkpoint, so the layer builders need no per-axis width.)
#[derive(Clone, Copy)]
struct LayerDims {
  num_heads: i32,
  head_dim: i32,
  /// `head_dim**-0.5`, the SDPA scale.
  scale: f32,
  /// The encoder LayerNorm eps (`config.layer_norm_eps`).
  eps: f32,
}

impl LayerDims {
  /// Derive the per-layer dims from `(hidden, num_heads, eps)`, computing the
  /// head split + SDPA scale once. `num_heads` must be positive and divide
  /// `hidden` (the caller validates this against the config).
  fn new(hidden: i32, num_heads: i32, eps: f32) -> Result<Self> {
    require_positive("lfm2_vl vision: num_attention_heads", num_heads)?;
    crate::model_validation::require_divisible(
      "lfm2_vl vision: hidden_size",
      hidden,
      "lfm2_vl vision: num_attention_heads",
      num_heads,
    )?;
    let head_dim = hidden / num_heads;
    Ok(Self {
      num_heads,
      head_dim,
      scale: (head_dim as f32).powf(-0.5),
      eps,
    })
  }
}

/// SigLIP2 self-attention (`vision.py`'s `Attention`): bias=True `q/k/v/out`
/// projections feeding the bidirectional `scaled_dot_product_attention` with no
/// RoPE and `mask=None`. Each projection is quantize-aware
/// ([`MaybeQuantizedLinear`]).
#[derive(Debug)]
struct Attention {
  q_proj: MaybeQuantizedLinear,
  k_proj: MaybeQuantizedLinear,
  v_proj: MaybeQuantizedLinear,
  out_proj: MaybeQuantizedLinear,
  num_heads: i32,
  head_dim: i32,
  scale: f32,
}

impl Attention {
  /// Build from `{prefix}.{q,k,v,out}_proj` (bias=True), routing each
  /// projection through the quantize-aware
  /// [`MaybeQuantizedLinear::from_weights`] (quantized iff its `.scales`
  /// sibling is present and `quant` resolved scheme params).
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    dims: LayerDims,
    quant: &QuantResolver<'_>,
  ) -> Result<Self> {
    let q = format!("{prefix}.q_proj");
    let k = format!("{prefix}.k_proj");
    let v = format!("{prefix}.v_proj");
    let o = format!("{prefix}.out_proj");
    Ok(Self {
      q_proj: MaybeQuantizedLinear::from_weights(weights, &q, quant(&q))?,
      k_proj: MaybeQuantizedLinear::from_weights(weights, &k, quant(&k))?,
      v_proj: MaybeQuantizedLinear::from_weights(weights, &v, quant(&v))?,
      out_proj: MaybeQuantizedLinear::from_weights(weights, &o, quant(&o))?,
      num_heads: dims.num_heads,
      head_dim: dims.head_dim,
      scale: dims.scale,
    })
  }

  /// `(B, L, C) -> (B, L, C)` bidirectional attention with the given key
  /// `mask` (the native patch key-padding mask, or [`Mask::None`]).
  fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let shape = x.shape();
    let bsz = dim_i32(&shape, 0, "lfm2_vl vision Attention: batch")?;
    let seq = dim_i32(&shape, 1, "lfm2_vl vision Attention: seq")?;

    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    let q = self.split_heads(&q, bsz, seq)?;
    let k = self.split_heads(&k, bsz, seq)?;
    let v = self.split_heads(&v, bsz, seq)?;

    let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;
    // (B, n_heads, L, head_dim) -> (B, L, n_heads*head_dim).
    let attn = ops::shape::transpose_axes(&attn, &[0, 2, 1, 3])?;
    let embed_dim = checked_mul(
      "lfm2_vl vision Attention: num_heads * head_dim",
      "num_heads",
      self.num_heads,
      "head_dim",
      self.head_dim,
    )?;
    let attn = ops::shape::reshape(&attn, &[bsz, seq, embed_dim])?;
    self.out_proj.forward(&attn)
  }

  /// `(B, L, C) -> (B, n_heads, L, head_dim)`.
  fn split_heads(&self, x: &Array, bsz: i32, seq: i32) -> Result<Array> {
    let reshaped = ops::shape::reshape(x, &[bsz, seq, self.num_heads, self.head_dim])?;
    ops::shape::transpose_axes(&reshaped, &[0, 2, 1, 3])
  }
}

/// The two-layer GELU feed-forward (`vision.py`'s `MLP`):
/// `fc2(gelu_precise(fc1(x)))`, both `Linear`s biased and quantize-aware.
#[derive(Debug)]
struct Mlp {
  fc1: MaybeQuantizedLinear,
  fc2: MaybeQuantizedLinear,
}

impl Mlp {
  /// Build from `{prefix}.fc1` / `{prefix}.fc2` (bias=True), quantize-aware.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    prefix: &str,
    quant: &QuantResolver<'_>,
  ) -> Result<Self> {
    let fc1 = format!("{prefix}.fc1");
    let fc2 = format!("{prefix}.fc2");
    Ok(Self {
      fc1: MaybeQuantizedLinear::from_weights(weights, &fc1, quant(&fc1))?,
      fc2: MaybeQuantizedLinear::from_weights(weights, &fc2, quant(&fc2))?,
    })
  }

  /// `fc2(gelu_precise(fc1(x)))` — `nn.GELU(approx="precise")` is the tanh GELU
  /// ([`crate::lm::nn::activations::gelu_approx`]).
  fn forward(&self, x: &Array) -> Result<Array> {
    let h = self.fc1.forward(x)?;
    let h = crate::lm::nn::activations::gelu_approx(&h)?;
    self.fc2.forward(&h)
  }
}

/// A SigLIP2 pre-norm encoder layer (`vision.py`'s `EncoderLayer`):
/// `h = x + attn(ln1(x)); out = h + mlp(ln2(h))`.
#[derive(Debug)]
struct EncoderLayer {
  layer_norm1: LayerNorm,
  self_attn: Attention,
  layer_norm2: LayerNorm,
  mlp: Mlp,
}

impl EncoderLayer {
  /// Build the `i`-th layer from `{encoder_prefix}.layers.{i}.*`.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    encoder_prefix: &str,
    i: i32,
    dims: LayerDims,
    quant: &QuantResolver<'_>,
  ) -> Result<Self> {
    let prefix = format!("{encoder_prefix}.layers.{i}");
    let layer_norm1 = build_layer_norm(weights, &format!("{prefix}.layer_norm1"), dims.eps)?;
    let self_attn = Attention::from_weights(weights, &format!("{prefix}.self_attn"), dims, quant)?;
    let layer_norm2 = build_layer_norm(weights, &format!("{prefix}.layer_norm2"), dims.eps)?;
    let mlp = Mlp::from_weights(weights, &format!("{prefix}.mlp"), quant)?;
    Ok(Self {
      layer_norm1,
      self_attn,
      layer_norm2,
      mlp,
    })
  }

  /// `h = x + attn(ln1(x), mask); out = h + mlp(ln2(h))`.
  fn forward(&self, x: &Array, mask: Mask<'_>) -> Result<Array> {
    let r = self
      .self_attn
      .forward(&self.layer_norm1.forward(x)?, mask)?;
    let h = x.add(&r)?;
    let r = self.mlp.forward(&self.layer_norm2.forward(&h)?)?;
    h.add(&r)
  }
}

/// The LFM2.5-VL SigLIP2 vision transformer (`vision.py`'s `VisionModel`):
/// per-image position-resized patch embeds → pre-norm encoder (truncated to
/// the feature layer) → post-LayerNorm.
///
/// Every `nn.Linear` (patch embed, q/k/v/out, fc1/fc2) is routed through the
/// shared quantize-aware [`MaybeQuantizedLinear`], so the 8-bit checkpoint
/// loads through the same path as a dense one (per-layer auto-detect by the
/// `.scales` sibling). The public [`crate::vlm::models::lfm2_vl`] VL model (a
/// later phase) drives this; for now it is exercised directly through
/// [`VisionModel::from_weights`] + [`VisionModel::forward`].
#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
#[derive(Debug)]
pub struct VisionModel {
  /// The patch-embedding `Linear` (`(num_channels*patch^2) -> hidden`).
  patch_embedding: MaybeQuantizedLinear,
  /// The trained position-embedding table `(num_patches, hidden)`.
  position_embedding: Array,
  /// `sqrt(num_patches)` — the square side of the trained position grid.
  pos_grid_side: i32,
  layers: Vec<EncoderLayer>,
  post_layernorm: LayerNorm,
  hidden: i32,
}

#[cfg(feature = "lfm2-vl")]
#[cfg_attr(docsrs, doc(cfg(feature = "lfm2-vl")))]
impl VisionModel {
  /// Build the vision tower from a validated [`VisionConfig`], the resolved
  /// kept-layer count `layers_kept` (`vision_feature_layer + 1`; the caller
  /// resolves it from the [`ModelConfig`](super::config::ModelConfig) — `-1`
  /// keeps all `num_hidden_layers`), the (already-[`sanitize`](Self::sanitize)d)
  /// weight map, and the optional per-layer quantization resolver.
  ///
  /// `weights` keys follow `vision.py`'s module tree with the VL-level prefix
  /// already stripped: `embeddings.patch_embedding.{weight,…}`,
  /// `embeddings.position_embedding.weight`,
  /// `encoder.layers.{i}.{layer_norm1,layer_norm2}.{weight,bias}`,
  /// `encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias,…}`,
  /// `encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias,…}`,
  /// `post_layernorm.{weight,bias}`. Only `layers_kept` encoder layers are
  /// built (and only their weights are required).
  ///
  /// `quant` resolves a layer path's `(group_size, bits, mode)` from the parsed
  /// quantization config (`None` for a dense layer); a quantized layer is
  /// detected by the presence of its `<prefix>.scales` sibling, so passing a
  /// resolver that returns `None` everywhere (or `&|_| None`) loads a dense
  /// checkpoint unchanged.
  ///
  /// # Errors
  /// - [`Error::MissingKey`] for an absent required weight;
  /// - [`Error::OutOfRange`] if `layers_kept` is not in
  ///   `[1, num_hidden_layers]`;
  /// - propagates [`VisionConfig::validate`], the
  ///   [`MaybeQuantizedLinear`] quantized-triple validation, and any op error.
  pub fn from_weights(
    config: &VisionConfig,
    layers_kept: i32,
    weights: &mut HashMap<String, Array>,
    quant: &QuantResolver<'_>,
  ) -> Result<Self> {
    // Idempotent re-validation (the constructor is public, so a caller may pass
    // a directly-built config): bounds num_hidden_layers + every dim before the
    // per-layer reservation / loop below.
    config.validate()?;
    let hidden = config.hidden_size;
    let dims = LayerDims::new(
      hidden,
      config.num_attention_heads,
      config.layer_norm_eps as f32,
    )?;
    // The kept-layer count must be in `[1, num_hidden_layers]` (the caller
    // resolves it, but pin it here too since `from_weights` is public).
    if layers_kept < 1 || layers_kept > config.num_hidden_layers {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "lfm2_vl vision: layers_kept (vision_feature_layer + 1)",
        "must be in [1, num_hidden_layers]",
        smol_str::format_smolstr!(
          "layers_kept={layers_kept}, num_hidden_layers={}",
          config.num_hidden_layers
        ),
      )));
    }

    // ── patch embedding (the flattened Linear, quantize-aware) ──
    let patch_embedding = MaybeQuantizedLinear::from_weights(
      weights,
      "embeddings.patch_embedding",
      quant("embeddings.patch_embedding"),
    )?;

    // ── position embedding table (kept dense — it is an nn.Embedding gathered
    //    then resized; the LFM2.5-VL checkpoint stores it unquantized) ──
    let num_positions = config.num_patches;
    let position_embedding = take_weight(weights, "embeddings.position_embedding.weight")?;
    let pos_grid_side = isqrt_exact(num_positions, "lfm2_vl::VisionConfig: num_patches")?;

    // ── encoder layers (only the kept prefix) ──
    let mut layers: Vec<EncoderLayer> = Vec::new();
    reserve_or_error(&mut layers, "lfm2_vl EncoderLayer", layers_kept as usize)?;
    for i in 0..layers_kept {
      layers.push(EncoderLayer::from_weights(
        weights, "encoder", i, dims, quant,
      )?);
    }

    // ── post-LayerNorm (mlx default eps 1e-5, NOT layer_norm_eps) ──
    let post_layernorm = build_layer_norm(weights, "post_layernorm", LAYERNORM_DEFAULT_EPS)?;

    Ok(Self {
      patch_embedding,
      position_embedding,
      pos_grid_side,
      layers,
      post_layernorm,
      hidden,
    })
  }

  /// `vision.py`'s `sanitize`: drop any `position_ids` key (a non-parameter
  /// buffer some checkpoints carry). Applied in place before construction.
  pub fn sanitize(weights: &mut HashMap<String, Array>) {
    weights.retain(|k, _| !k.contains("position_ids"));
  }

  /// Forward one image's pre-flattened patches through the tower.
  ///
  /// - `pixel_values` — `(N, num_patches, num_channels * patch_size^2)` (the
  ///   processor's flattened patches; `N` is the per-call image/batch dim).
  /// - `spatial_shapes` — `(N, 2)` i32 `[H_patch, W_patch]` per image (the
  ///   active patch grid each image's position embedding is resized to).
  /// - `pixel_attention_mask` — optional `(N, num_patches)` i32 companion. When
  ///   `Some`, an additive key mask excludes the padded patches from every
  ///   encoder layer's attention (so active patches cannot attend to the padded
  ///   zero rows); when `None`, the tower runs full bidirectional attention over
  ///   all `num_patches` rows. **The mask's *content* is not trusted**: it is
  ///   shape-validated (a wrong-shape companion is a typed error) but the
  ///   additive key mask is *derived from `spatial_shapes`*, the single source
  ///   of truth for the active grid (see below). The companion's `Some`/`None`
  ///   only selects whether key masking is applied at all.
  ///
  /// ## Why the mask is load-bearing, and why it is derived from `spatial_shapes`
  ///
  /// The processor pads every image to `num_patches`, so an image whose active
  /// grid `H_p * W_p` is smaller than the patch budget carries padded zero
  /// rows (with the first-resized-position positional embedding). Without the
  /// mask those padded keys attend and are attended to, contaminating every
  /// active patch's representation before the active-row slice — the HF
  /// LFM2-VL reference threads `pixel_attention_mask` into the vision tower
  /// (`modeling_lfm2_vl.py`) for exactly this reason.
  ///
  /// `spatial_shapes` is the authoritative active grid: the processor lays the
  /// active patches out FIRST in row-major order and zero-pads the suffix
  /// (`processing_lfm2_vl.py` / [`super::processor::preprocess_image`]), and the
  /// active-row slice downstream
  /// ([`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs))
  /// reads its active count as `H_p * W_p` from `spatial_shapes`. So the
  /// additive key mask is built from `spatial_shapes` (the first `H_p * W_p`
  /// keys active → `0.0`, the rest padded → `-inf`) rather than from the
  /// companion mask's content — guaranteeing the mask can never disagree with
  /// the active-row slice, even if a caller fed a malformed companion through
  /// the public [`NativeResolution`](crate::vlm::model::NativeResolution) seam.
  /// The construction mirrors the SigLIP2 NaFlex vision tower's
  /// [`crate::embeddings::siglip2_naflex`] additive padded-key mask (active →
  /// `0.0`, padded → `-inf`, broadcast over heads + query positions by SDPA).
  ///
  /// Returns the post-LayerNorm hidden states `(N, num_patches, hidden)`
  /// (`vision.py`'s `last_hidden_state`).
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `pixel_values` is not rank-3, or
  ///   `spatial_shapes` is not exactly `(N, 2)`, or `pixel_attention_mask`
  ///   (when `Some`) is not exactly `(N, num_patches)` (so a per-image-count or
  ///   trailing-axis mismatch is rejected here, not by a downstream broadcast);
  /// - [`Error::OutOfRange`] if an image's active grid `H_p * W_p` is
  ///   non-positive or exceeds the patch budget;
  /// - propagates the embed / interp / attention op errors.
  pub fn forward(
    &self,
    pixel_values: &Array,
    spatial_shapes: &Array,
    pixel_attention_mask: Option<&Array>,
  ) -> Result<Array> {
    let pv_shape = pixel_values.shape();
    if pv_shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl vision: pixel_values must be rank-3 (N, num_patches, patch_feature_dim)",
        pv_shape.len() as u32,
        pv_shape,
      )));
    }
    let n = pv_shape[0];
    let num_patches = dim_i32(&pv_shape, 1, "lfm2_vl vision: num_patches")?;
    let ss_shape = spatial_shapes.shape();
    if ss_shape.as_slice() != [n, 2] {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "lfm2_vl vision: spatial_shapes must be (N, 2) [H_patch, W_patch] per image",
        ss_shape.len() as u32,
        ss_shape,
      )));
    }

    // Read every image's active grid `(H_p, W_p)` from `spatial_shapes` — the
    // single source of truth for the active patch count — once, and validate it
    // (positive dims, `H_p * W_p <= num_patches`) before it drives BOTH the
    // position resize and the attention mask. Reused so the two stay consistent.
    let shapes = read_spatial_shapes(spatial_shapes, n)?;
    for &(h_p, w_p) in &shapes {
      validate_active_grid(h_p, w_p, num_patches)?;
    }

    // `patch_embeds = patch_embedding(pixel_values)` → (N, num_patches, hidden).
    let patch_embeds = self.patch_embedding.forward(pixel_values)?;

    // Per-image position embedding: resize the trained square grid to each
    // image's (H_p, W_p) and scatter into the active patch rows.
    let pos = self.resize_positional_embeddings(&shapes, num_patches)?;
    let mut h = patch_embeds.add(&pos)?;

    // Build the additive key-padding mask. When the companion
    // `pixel_attention_mask` is present, shape-validate it against the verified
    // `(n, num_patches)` (a mismatched shape is a typed error, not a silent SDPA
    // broadcast), but DERIVE the mask content from `spatial_shapes` (the source
    // of truth): the first `H_p * W_p` keys per image are active (`0.0`), the
    // rest padded (`-inf`). The companion's content is intentionally NOT trusted
    // — a malformed-but-right-shaped companion (passed through the public
    // `NativeResolution` seam) therefore cannot drop real keys, re-admit padded
    // keys, or all-mask a row. Held in an `Option<Array>` so the `Mask` borrow
    // below outlives every layer's use.
    //
    // Skip the mask entirely when NO image is padded (every grid fills the
    // patch budget): the derived mask would be all-`0.0`, numerically identical
    // to no mask, but building it costs a per-call host buffer + device copy and
    // routes SDPA onto the masked-kernel path every forward. `vision.py` always
    // passes `mask=None`; match that for the unpadded case and only pay for the
    // mask when padding actually requires excluding keys.
    let attn_mask = match pixel_attention_mask {
      Some(m) => {
        validate_mask_shape(m, n, num_patches)?;
        // `validate_active_grid` above proved every `H_p * W_p <= num_patches`,
        // so the product cannot overflow; `saturating_mul` is belt-and-braces.
        let has_padding = shapes
          .iter()
          .any(|&(h_p, w_p)| h_p.saturating_mul(w_p) < num_patches);
        if has_padding {
          Some(build_attention_mask(&shapes, n, num_patches)?)
        } else {
          None
        }
      }
      None => None,
    };
    let mask = match &attn_mask {
      Some(m) => Mask::Array(m),
      None => Mask::None,
    };

    for layer in &self.layers {
      h = layer.forward(&h, mask)?;
    }
    self.post_layernorm.forward(&h)
  }

  /// Build the per-image positional-embedding term `(N, num_patches, hidden)`,
  /// mirroring `vision.py`'s `resize_positional_embeddings`: reshape the trained
  /// `(num_positions, hidden)` table to `(side, side, hidden)`, and per image
  /// bicubic-resize it to that image's `(H_p, W_p)`, flatten to
  /// `(H_p * W_p, hidden)`, place into the active rows, and pad the remainder
  /// with the first resized position.
  ///
  /// `shapes` is the host-side per-image `(H_p, W_p)` grid the caller already
  /// read + validated from `spatial_shapes` (so the position resize and the
  /// attention mask are driven by the same source-of-truth values).
  fn resize_positional_embeddings(&self, shapes: &[(i32, i32)], num_patches: i32) -> Result<Array> {
    let side = self.pos_grid_side as usize;
    let hidden = self.hidden as usize;
    // (num_positions, hidden) -> (side, side, hidden).
    let grid = ops::shape::reshape(&self.position_embedding, &(side, side, hidden))?;
    // -> (hidden, side, side) -> (1, hidden, side, side) for bicubic_interpolate
    // (the reference's `transpose(2, 0, 1)[None, :]`).
    let grid_chw = ops::shape::transpose_axes(&grid, &[2, 0, 1])?;
    let grid_bchw = ops::shape::expand_dims_axes(&grid_chw, &[0])?;

    // Build one (1, num_patches, hidden) term per image and concatenate along N.
    let mut per_image: Vec<Array> = Vec::new();
    reserve_or_error(&mut per_image, "lfm2_vl vision position term", shapes.len())?;
    for &(h_p, w_p) in shapes {
      per_image.push(self.resize_one_image(&grid_bchw, h_p, w_p, num_patches)?);
    }
    let refs: Vec<&Array> = per_image.iter().collect();
    ops::shape::concatenate(&refs, 0)
  }

  /// One image's `(1, num_patches, hidden)` position term: bicubic-resize the
  /// `(1, hidden, side, side)` grid to `(H_p, W_p)`, reshape to
  /// `(H_p * W_p, hidden)`, and right-pad to `num_patches` with the first
  /// resized position.
  fn resize_one_image(
    &self,
    grid_bchw: &Array,
    h_p: i32,
    w_p: i32,
    num_patches: i32,
  ) -> Result<Array> {
    // Validate + compute the active patch count from the (source-of-truth)
    // grid (positive dims, `H_p * W_p <= num_patches`). `forward` already
    // validates every image's grid up front; this keeps the private helper
    // independently sound.
    let active = validate_active_grid(h_p, w_p, num_patches)?;
    // (1, hidden, side, side) -> bicubic -> (1, hidden, H_p, W_p).
    let resized = bicubic_interpolate(grid_bchw, h_p as usize, w_p as usize)?;
    // `resized.reshape(hidden, h*w).transpose(1, 0)` -> (H_p*W_p, hidden).
    let resized = ops::shape::reshape(&resized, &[self.hidden, active])?;
    let active_pos = ops::shape::transpose_axes(&resized, &[1, 0])?;

    // Right-pad to (num_patches, hidden) with the FIRST resized position
    // (the reference's `resulted_positional_embeddings[i, h*w:] = resized[0]`).
    let padded = if active < num_patches {
      let pad_rows = num_patches - active;
      let first = ops::indexing::take_axis(&active_pos, &index_zero()?, 0)?; // (1, hidden)
      let pad_block = ops::shape::broadcast_to(&first, &[pad_rows, self.hidden])?;
      ops::shape::concatenate(&[&active_pos, &pad_block], 0)?
    } else {
      active_pos
    };
    // (num_patches, hidden) -> (1, num_patches, hidden).
    ops::shape::expand_dims_axes(&padded, &[0])
  }
}

// ───────────────────────── builders / helpers ─────────────────────────

/// Build an additive attention key-mask `(N, 1, 1, num_patches)` *derived from
/// `spatial_shapes`* (the active-grid source of truth): for image `i` the first
/// `H_p * W_p` keys are active → `0.0`, the remaining `num_patches - H_p * W_p`
/// are padding → `-inf`, so SDPA's softmax zeroes the padded keys for every
/// active query (the padded patches contribute nothing).
///
/// Deriving from `spatial_shapes` (rather than from the companion
/// `pixel_attention_mask`'s content) guarantees the mask cannot disagree with
/// the `H_p * W_p` active-row slice
/// [`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs)
/// takes from the same `spatial_shapes`: the NaFlex processor lays the active
/// patches out FIRST and zero-pads the suffix
/// ([`super::processor::preprocess_image`]), so `spatial_shapes` fully
/// determines which keys are active. A malformed-but-right-shaped companion
/// (fed through the public [`NativeResolution`](crate::vlm::model::NativeResolution)
/// seam — zeros in the active prefix, ones in the padding, or all-zero) is
/// therefore inert: it cannot drop a real key, re-admit a padded key, or
/// all-mask a row.
///
/// Mirrors the SigLIP2 NaFlex padded-key additive form
/// ([`crate::embeddings::siglip2_naflex`]'s `build_attention_mask`: active →
/// `0.0`, padded → `-inf`, reshaped to broadcast over `(B, heads, L_q, L_kv)`),
/// generalized to the LFM2-VL batched leading axis `N`. The active counts in
/// `shapes` are assumed already validated by [`validate_active_grid`] (positive
/// dims, `H_p * W_p <= num_patches`); each is recomputed here with the same
/// checked multiply for independent soundness.
#[cfg(feature = "lfm2-vl")]
fn build_attention_mask(shapes: &[(i32, i32)], n: usize, num_patches: i32) -> Result<Array> {
  debug_assert_eq!(shapes.len(), n, "shapes already read for n images");
  let np = num_patches as usize;
  // Host-side additive mask: `n * num_patches` f32, each image's active prefix
  // `0.0` and padded suffix `-inf`. Reserved fallibly (the element count is
  // bounded by the already-validated patch budget).
  let total = checked_mul(
    "lfm2_vl vision: attention mask elements (N * num_patches)",
    "N",
    n as i32,
    "num_patches",
    num_patches,
  )? as usize;
  let mut data: Vec<f32> = Vec::new();
  reserve_or_error(&mut data, "lfm2_vl vision attention mask", total)?;
  data.resize(total, f32::NEG_INFINITY);
  for (i, &(h_p, w_p)) in shapes.iter().enumerate() {
    // Source of truth for image `i`'s active count (same checked multiply +
    // budget bound as the position resize / the active-row slice).
    let active = validate_active_grid(h_p, w_p, num_patches)? as usize;
    let base = i * np;
    for slot in data[base..base + active].iter_mut() {
      *slot = 0.0;
    }
  }
  // (N * num_patches,) host buffer → (N, 1, 1, num_patches) to broadcast over
  // (B, heads, L_q, L_kv).
  Array::from_slice::<f32>(&data, &(n, 1usize, 1usize, np))
}

/// Validate one image's active grid `(H_p, W_p)` against the patch budget and
/// return the active patch count `H_p * W_p`: both dims must be positive and
/// the product must not exceed `num_patches` (the `pixel_values` patch dim). A
/// degenerate grid is a typed [`Error::OutOfRange`] / [`Error::ArithmeticOverflow`]
/// — `spatial_shapes` is the source of truth for the active grid, so a grid that
/// disagrees with the patch budget is rejected here, before it drives the
/// position resize, the active-row slice, or the attention mask.
///
/// Shared as the single active-grid validation point: the vision tower's
/// position resize + attention mask call it, and
/// [`Lfm2Vl::encode_image_inputs`](super::model::Lfm2Vl::encode_image_inputs)
/// calls it for the active-row slice + PixelUnshuffle reshape — all from the
/// same `spatial_shapes`-derived `(H_p, W_p)`.
#[cfg(feature = "lfm2-vl")]
pub(crate) fn validate_active_grid(h_p: i32, w_p: i32, num_patches: i32) -> Result<i32> {
  require_positive("lfm2_vl vision: H_patch", h_p)?;
  require_positive("lfm2_vl vision: W_patch", w_p)?;
  let active = checked_mul(
    "lfm2_vl vision: H_patch * W_patch",
    "H_patch",
    h_p,
    "W_patch",
    w_p,
  )?;
  if active > num_patches {
    // The active grid cannot exceed the patch budget — the processor guarantees
    // H_p*W_p <= num_patches; an out-of-range spatial_shapes entry is rejected.
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "lfm2_vl vision: active patches (H_p * W_p)",
      "must not exceed num_patches (the pixel_values patch dim)",
      smol_str::format_smolstr!("{active} > {num_patches}"),
    )));
  }
  Ok(active)
}

/// Shape-validate the companion `pixel_attention_mask` against the
/// `(n, num_patches)` the caller already verified from `pixel_values`. The
/// companion's *content* is not trusted (the additive mask is derived from
/// `spatial_shapes` in [`build_attention_mask`]); this only rejects a
/// wrong-shaped companion as a typed [`Error::RankMismatch`] (no panic, no
/// silent downstream broadcast), keeping the shape contract faithful.
#[cfg(feature = "lfm2-vl")]
fn validate_mask_shape(pixel_attention_mask: &Array, n: usize, num_patches: i32) -> Result<()> {
  let shape = pixel_attention_mask.shape();
  if shape.as_slice() != [n, num_patches as usize] {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfm2_vl vision: pixel_attention_mask must be (N, num_patches) matching pixel_values",
      shape.len() as u32,
      shape,
    )));
  }
  Ok(())
}

/// Build a `LayerNorm` from `{prefix}.weight` + `{prefix}.bias` with the given
/// `eps`. Both affine params are required (SigLIP2 LayerNorms are full-affine).
#[cfg(feature = "lfm2-vl")]
fn build_layer_norm(
  weights: &mut HashMap<String, Array>,
  prefix: &str,
  eps: f32,
) -> Result<LayerNorm> {
  let weight = take_weight(weights, &format!("{prefix}.weight"))?;
  let bias = take_weight(weights, &format!("{prefix}.bias"))?;
  Ok(LayerNorm::new(Some(weight), Some(bias), eps))
}

/// Pop a required weight by exact key from the (sanitized) checkpoint map,
/// erroring with the key if absent.
#[cfg(feature = "lfm2-vl")]
fn take_weight(weights: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
  weights.remove(key).ok_or_else(|| {
    Error::MissingKey(crate::error::MissingKeyPayload::new(
      "lfm2_vl::VisionModel::from_weights",
      key,
    ))
  })
}

/// Pull `shape[axis]` as `i32`, erroring on rank underflow or `i32::MAX`
/// overflow.
#[cfg(feature = "lfm2-vl")]
fn dim_i32(shape: &[usize], axis: usize, context: &'static str) -> Result<i32> {
  let d = *shape.get(axis).ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      context,
      shape.len() as u32,
      shape.to_vec(),
    ))
  })?;
  i32::try_from(d).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "dim exceeds i32::MAX",
      smol_str::format_smolstr!("{d}"),
    ))
  })
}

/// The `(1,)` i32 index array `[0]`, for `take_axis(_, axis=0)` (the first
/// resized position row) — the lazy analogue of `resized[0]`.
#[cfg(feature = "lfm2-vl")]
fn index_zero() -> Result<Array> {
  Array::from_slice::<i32>(&[0], &(1usize,))
}

/// Read every image's `(H_p, W_p)` from the `(N, 2)` i32 `spatial_shapes`.
/// Evaluates the tiny `2N`-element array to host integers (the resize geometry
/// — the per-image bicubic target dims — is host-side).
#[cfg(feature = "lfm2-vl")]
fn read_spatial_shapes(spatial_shapes: &Array, n: usize) -> Result<Vec<(i32, i32)>> {
  let shape = spatial_shapes.shape();
  if shape.as_slice() != [n, 2] {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "lfm2_vl vision: spatial_shapes must be (N, 2) [H_p, W_p]",
      shape.len() as u32,
      shape,
    )));
  }
  let mut s = ops::misc::astype(spatial_shapes, Dtype::I32)?;
  s.eval()?;
  let v = s.to_vec::<i32>()?;
  let mut out: Vec<(i32, i32)> = Vec::new();
  reserve_or_error(&mut out, "lfm2_vl vision spatial shape", n)?;
  for i in 0..n {
    out.push((v[2 * i], v[2 * i + 1]));
  }
  Ok(out)
}

/// Exact integer square root: returns `r` with `r*r == n`, else a typed error.
/// Derives the trained position grid's square side from `num_positions`.
#[cfg(feature = "lfm2-vl")]
fn isqrt_exact(n: i32, context: &'static str) -> Result<i32> {
  require_positive(context, n)?;
  let r = (n as f64).sqrt().round() as i32;
  if r.saturating_mul(r) != n {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "num_patches must be a perfect square (the trained position grid is square)",
      smol_str::format_smolstr!("{n}"),
    )));
  }
  Ok(r)
}

#[cfg(all(test, feature = "lfm2-vl"))]
mod tests;
