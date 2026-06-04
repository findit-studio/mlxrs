//! CLAP HTSAT Swin-Transformer audio tower (`ClapAudioModel`).
//!
//! Ports HF `transformers`' `ClapAudioModel` / `ClapAudioEncoder`
//! (`src/transformers/models/clap/modeling_clap.py`): the mel-spectrogram → 2-D
//! "image" fold (`reshape_mel2img`, `ClapAudioEncoder.reshape_mel2img`,
//! ~L1370-1410), the patch-embed stem (`ClapAudioPatchEmbed`, ~L615-705), the
//! four hierarchical Swin stages (`ClapAudioStage` ~L1235-1285 of
//! `ClapAudioLayer` blocks + `ClapAudioPatchMerging` downsamples), the encoder
//! batch-norm + final `LayerNorm`, and the token-semantic mean-pool that
//! produces the clip-level `(batch, hidden=768)` feature `ClapAudioModel.forward`
//! returns to the audio projection (the `pooler_output` / `latent_output`, NOT
//! the AudioSet classifier logits — pinned below).
//!
//! The genuinely-new Swin blocks (window partition/reverse, the shifted-window
//! roll + SW-MSA mask, the relative-position-bias gather, the window SDPA, the
//! Swin MLP, the `2×2` patch-merging) live in the private `super::shared` `swin`
//! module (the step-1 shared blocks); this file **assembles** them into the
//! tower. The patch-embed `Conv2d` reuses [`crate::ops::conv::conv2d`] (NHWC),
//! and the mel→image upsample reuses
//! [`crate::ops::interpolation::bicubic_interpolate_align_corners`] (the
//! `align_corners=True` bicubic HF `reshape_mel2img` uses).
//!
//! ## `reshape_mel2img` (the #1 fidelity risk — HF-only, no textclap cross-check)
//!
//! HF `ClapAudioEncoder.reshape_mel2img` (`modeling_clap.py` ~L1370-1410) folds a
//! `(batch, 1, time = T_FRAMES, freq = num_mel_bins)` log-mel into the
//! `(spec_size, spec_size)` square the Swin stem expects. With the unfused
//! `laion/clap-htsat-unfused` constants (`spec_size = 256`, `num_mel_bins = 64`,
//! so the encoder's `freq_ratio = spec_size // num_mel_bins = 4`, matching the
//! config `freq_ratio = 4`):
//!
//! ```text
//! spec_width  = spec_size * freq_ratio = 1024
//! spec_height = spec_size / freq_ratio = 64
//! # time = 1001 < spec_width = 1024  → bicubic(align_corners=True) to (1024, 64)
//! # freq = 64    == spec_height = 64 → no second interpolation
//! reshape (B, 1*freq_ratio=4, time//freq_ratio=256, freq=64)
//! permute (0, 1, 3, 2)                         → (B, 4, 64, 256)
//! reshape (B, 1, freq*freq_ratio=256, time//freq_ratio=256)  → (B, 1, 256, 256)
//! ```
//!
//! So the `1001`-frame time axis is bicubically stretched to `1024` then folded
//! into four `256`-row freq-blocks stacked along the (new) `256`-tall frequency
//! axis. The fold reshape/permute is ported verbatim and pinned by a hand-checked
//! small-grid oracle; the `align_corners=True` bicubic is the
//! [`bicubic_interpolate_align_corners`](crate::ops::interpolation::bicubic_interpolate_align_corners)
//! primitive (HF uses `nn.functional.interpolate(mode="bicubic",
//! align_corners=True)`).
//!
//! ## Pooling → the audio feature (pinned)
//!
//! After the four stages + the final `LayerNorm`, HF
//! `ClapAudioEncoder.forward` (~L1470-1495) permutes/reshapes the
//! `(B, L, 768)` tokens into a `(B, 768, freq, temporal)` grid, re-folds the
//! freq-blocks, and applies `nn.AdaptiveAvgPool1d(1)` over the flattened
//! spatial axis — i.e. it **averages every spatial token per channel**. A mean
//! is invariant to that rearrangement, so the pooled `(B, 768)`
//! `latent_output` equals the plain mean of the post-norm `(B, L, 768)` tokens
//! over the token axis; this port computes it directly as that mean (and pins
//! the token count against HF's `freq · c_freq_bin · temporal`). This is the
//! `BaseModelOutputWithPooling.pooler_output` that `ClapAudioModel.forward`
//! (~L1540-1585) returns to the audio projection — **not** the
//! token-semantic classifier (HTSAT's `tscam`/AudioSet head is absent from the
//! contrastive `ClapAudioModel` path), so the port has no classifier conv.
//!
//! ## Quant + dtype
//!
//! Every `nn.Linear` (the window q/k/v/output, the Swin MLP fc1/fc2, the
//! patch-merge reductions) is quantize-aware via the shared blocks'
//! `QuantLinear` (the `.scales`-sibling `class_predicate`), so a quantized CLAP
//! checkpoint loads
//! with a byte-identical dense path. The mel→image fold, the batch-norm affine,
//! and the relative-position bias are cast back to the activation dtype before
//! they meet the activations, so an f16/bf16 checkpoint is not silently promoted
//! to f32 (the recurring activation-dtype-preservation faithfulness bug).
//!
//! ## Scope
//!
//! This is **phase 3** of the CLAP port (the audio tower). The audio projection,
//! the full dual-tower `ClapModel` assembly + `classify` + the factory
//! registration (phase 4), and the end-to-end checkpoint-parity test (phase 5)
//! are out of scope; [`HtsatAudioTower`] exposes a clean
//! [`forward`](HtsatAudioTower::forward) the assembly layer consumes as the audio
//! tower (mirroring [`super::text::ClapTextModel`]'s `embed_text`).

use std::collections::HashMap;

use crate::{
  array::Array,
  embeddings::clap::{
    config::ClapConfig,
    shared::{PatchMerging, SwinBlock, build_layer_norm, dim_i32, take_shaped},
  },
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  lm::{nn::norm::LayerNorm, quant::PerLayerQuantization},
  model_validation::{checked_mul, require_divisible, require_positive, reserve_or_error},
  ops,
};

use smol_str::format_smolstr;

/// The PyTorch `nn.BatchNorm2d` default epsilon (`1e-5`). HF
/// `ClapAudioEncoder.batch_norm = nn.BatchNorm2d(num_mel_bins)` is constructed
/// with the default eps, which is not serialized in `config.json`.
#[cfg(feature = "clap")]
const BATCH_NORM_EPS: f32 = 1e-5;

// ════════════════════════════ EncoderBatchNorm ═════════════════════════════

/// HF `ClapAudioEncoder.batch_norm` — `nn.BatchNorm2d(num_mel_bins)` applied (at
/// inference) as a per-mel-bin affine over the running statistics.
///
/// HF transposes the mel to put the `num_mel_bins` on the channel axis
/// (`input_features.transpose(1, 3)`), batch-norms, then transposes back
/// (`modeling_clap.py` ~L1430-1432). Since `BatchNorm2d` normalizes the channel
/// axis and the transpose only moves the `num_mel_bins` between axis 1 and the
/// last axis, the equivalent in the native `(B, 1, time, freq)` layout is a
/// per-`freq` affine over the **last** axis — computed directly from the four
/// `(num_mel_bins,)` buffers (`weight`, `bias`, `running_mean`, `running_var`):
///
/// ```text
/// y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
/// ```
///
/// This avoids the transpose round-trip while being numerically identical.
#[cfg(feature = "clap")]
struct EncoderBatchNorm {
  /// `weight * rsqrt(running_var + eps)` precombined `(num_mel_bins,)` scale.
  scale: Array,
  /// `bias - running_mean * scale` precombined `(num_mel_bins,)` shift.
  shift: Array,
}

#[cfg(feature = "clap")]
impl EncoderBatchNorm {
  /// Build from `batch_norm.{weight,bias,running_mean,running_var}`, each pinned
  /// to `(num_mel_bins,)`, precombining the inference affine
  /// (`scale = weight * rsqrt(var + eps)`, `shift = bias - mean * scale`).
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    num_mel_bins: i32,
    eps: f32,
  ) -> Result<Self> {
    let weight = take_shaped(
      weights,
      "batch_norm.weight",
      "clap HTSAT batch_norm weight (num_mel_bins,)",
      &[num_mel_bins],
    )?;
    let bias = take_shaped(
      weights,
      "batch_norm.bias",
      "clap HTSAT batch_norm bias (num_mel_bins,)",
      &[num_mel_bins],
    )?;
    let running_mean = take_shaped(
      weights,
      "batch_norm.running_mean",
      "clap HTSAT batch_norm running_mean (num_mel_bins,)",
      &[num_mel_bins],
    )?;
    let running_var = take_shaped(
      weights,
      "batch_norm.running_var",
      "clap HTSAT batch_norm running_var (num_mel_bins,)",
      &[num_mel_bins],
    )?;
    // scale = weight / sqrt(var + eps); shift = bias - mean * scale. Built in the
    // buffers' dtype; the forward casts the result back to the activation dtype.
    let eps_arr = Array::full::<f32>(&(1usize,), eps)?;
    let eps_arr = ops::misc::astype(&eps_arr, running_var.dtype()?)?;
    let denom = ops::arithmetic::rsqrt(&running_var.add(&eps_arr)?)?;
    let scale = weight.multiply(&denom)?;
    let shift = bias.subtract(&running_mean.multiply(&scale)?)?;
    Ok(Self { scale, shift })
  }

  /// `(B, 1, time, freq) → (B, 1, time, freq)`: `x * scale + shift` broadcast
  /// over the last (`freq`) axis, in the activation dtype.
  fn forward(&self, x: &Array) -> Result<Array> {
    let dtype = x.dtype()?;
    let scale = ops::misc::astype(&self.scale, dtype)?;
    let shift = ops::misc::astype(&self.shift, dtype)?;
    x.multiply(&scale)?.add(&shift)
  }
}

// ══════════════════════════════ PatchEmbed ═════════════════════════════════

/// HF `ClapAudioPatchEmbed` (`modeling_clap.py` ~L615-705) for the unfused path:
/// a `Conv2d(in = patch_embed_input_channels, out = patch_embeds_hidden_size,
/// kernel = patch_size, stride = patch_size)` over the `(spec_size, spec_size)`
/// mel image, flattened to patch tokens, then a patch-embed `LayerNorm`.
///
/// mlxrs's [`conv2d`](crate::ops::conv::conv2d) is **NHWC** (input
/// `(N, H, W, C_in)`, weight `(C_out, KH, KW, C_in)`), while HF's `Conv2d` weight
/// is `(C_out, C_in, KH, KW)`; the step-4 `sanitize` transposes the patch-embed
/// weight to channels-last `[0, 2, 3, 1]` (the SigLIP2 `reshape_patch_weight`
/// precedent), and the image is fed `(B, H, W, 1)`. The HF `padding =
/// ((patch_size - patch_stride) // 2, …)` is `0` for the `patch_size ==
/// patch_stride == 4` checkpoint, so the conv is a non-overlapping `4×4`/stride-4
/// patchify producing a `(B, H/4, W/4, hidden)` grid, flattened to
/// `(B, (H/4)·(W/4), hidden)` (HF `flatten(2).transpose(1, 2)`).
#[cfg(feature = "clap")]
struct PatchEmbed {
  /// `proj.weight` `(hidden, KH, KW, in_channels)` (NHWC, sanitize-transposed).
  proj_weight: Array,
  /// `proj.bias` `(hidden,)`.
  proj_bias: Array,
  /// The patch-embed `LayerNorm(hidden)` (HF `enable_patch_layer_norm = True`).
  norm: LayerNorm,
  /// Conv kernel / stride side (`patch_size = patch_stride = 4`).
  patch_size: i32,
  /// Output channel count (`patch_embeds_hidden_size = 96`).
  hidden: i32,
  /// Input channel count (`patch_embed_input_channels = 1`).
  in_channels: i32,
}

#[cfg(feature = "clap")]
impl PatchEmbed {
  /// Build from `patch_embed.proj.{weight,bias}` + `patch_embed.norm.{weight,
  /// bias}`. The conv weight is pinned to the NHWC `(hidden, patch_size,
  /// patch_size, in_channels)` the step-4 sanitize produces; the bias to
  /// `(hidden,)`; the norm to `(hidden,)`.
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    in_channels: i32,
    hidden: i32,
    patch_size: i32,
    eps: f32,
  ) -> Result<Self> {
    let proj_weight = take_shaped(
      weights,
      "patch_embed.proj.weight",
      "clap HTSAT patch_embed conv weight (hidden, KH, KW, C_in) NHWC",
      &[hidden, patch_size, patch_size, in_channels],
    )?;
    let proj_bias = take_shaped(
      weights,
      "patch_embed.proj.bias",
      "clap HTSAT patch_embed conv bias (hidden,)",
      &[hidden],
    )?;
    let norm = build_layer_norm(weights, "patch_embed.norm", hidden, eps)?;
    Ok(Self {
      proj_weight,
      proj_bias,
      norm,
      patch_size,
      hidden,
      in_channels,
    })
  }

  /// `(B, 1, H, W)` NCHW mel image → `(B, (H/patch)·(W/patch), hidden)` patch
  /// tokens. Transposes the image to NHWC, runs the strided conv, flattens the
  /// `(B, H', W', hidden)` grid to `(B, H'·W', hidden)`, adds the conv bias, and
  /// applies the patch-embed `LayerNorm`. Returns the tokens and the `(H', W')`
  /// patch-grid resolution (the stage-0 input resolution).
  fn forward(&self, image: &Array) -> Result<(Array, i32, i32)> {
    let shape = image.shape();
    let b = dim_i32(&shape, 0, "clap HTSAT patch_embed: batch")?;
    let c_in = dim_i32(&shape, 1, "clap HTSAT patch_embed: channels")?;
    let height = dim_i32(&shape, 2, "clap HTSAT patch_embed: height")?;
    let width = dim_i32(&shape, 3, "clap HTSAT patch_embed: width")?;
    if c_in != self.in_channels {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap HTSAT patch_embed: input channels",
        "must equal patch_embed_input_channels",
        format_smolstr!("{c_in}"),
      )));
    }
    // (B, C_in, H, W) → (B, H, W, C_in) for mlxrs NHWC conv2d.
    let image = ops::shape::transpose_axes(image, &[0, 2, 3, 1])?;
    let conv = ops::conv::conv2d(
      &image,
      &self.proj_weight,
      (self.patch_size, self.patch_size),
      (0, 0),
      (1, 1),
      1,
    )?; // (B, H', W', hidden)
    let h_grid = height / self.patch_size;
    let w_grid = width / self.patch_size;
    let tokens = checked_mul("clap HTSAT patch_embed: H'·W'", "H'", h_grid, "W'", w_grid)?;
    // (B, H', W', hidden) → (B, H'·W', hidden), then + conv bias (HF folds the
    // bias into the conv; mlxrs conv2d takes no bias, so add it here).
    let flat = ops::shape::reshape(&conv, &[b, tokens, self.hidden])?;
    let bias = ops::misc::astype(&self.proj_bias, flat.dtype()?)?;
    let flat = flat.add(&bias)?;
    let normed = self.norm.forward(&flat)?;
    Ok((normed, h_grid, w_grid))
  }
}

// ═══════════════════════════════ AudioStage ════════════════════════════════

/// One HTSAT Swin stage — HF `ClapAudioStage` (`modeling_clap.py` ~L1235-1285):
/// `depth` [`SwinBlock`]s followed (except the deepest stage) by a
/// [`PatchMerging`] `2×2` downsample.
///
/// Even blocks use `shift = 0` (W-MSA), odd blocks `shift = window/2` (SW-MSA) —
/// BUT HF `ClapAudioLayer.set_shift_and_window_size` (~L1320-1328) zeroes the
/// shift (and shrinks the window to `min(H, W)`) when the stage resolution is
/// `<= window`. For every HTSAT stage resolution (`64, 32, 16, 8`) the effective
/// window stays the configured `8` (the smallest resolution, the deepest stage's
/// `8`, equals the window), so only the **shift** is affected: the deepest stage
/// (resolution `8 == window`) runs both blocks unshifted. Each block is therefore
/// built with the configured `window` and the resolved effective shift.
#[cfg(feature = "clap")]
struct AudioStage {
  blocks: Vec<SwinBlock>,
  /// The `PatchMerging` downsample, present on every stage except the last.
  downsample: Option<PatchMerging>,
  /// This stage's input (pre-downsample) resolution `(height, width)`.
  input_height: i32,
  input_width: i32,
}

#[cfg(feature = "clap")]
impl AudioStage {
  /// Build the `stage`-th stage from `layers.{stage}.*`: `depth` blocks
  /// (`layers.{stage}.blocks.{i}`) + an optional `layers.{stage}.downsample`
  /// `PatchMerging`. `dim` is this stage's channel width, `num_heads` its head
  /// count, `(input_height, input_width)` its resolution, `window` / `mlp_ratio`
  /// / `eps` the shared Swin constants.
  #[allow(clippy::too_many_arguments)]
  fn from_weights(
    weights: &mut HashMap<String, Array>,
    stage: i32,
    dim: i32,
    num_heads: i32,
    depth: i32,
    input_height: i32,
    input_width: i32,
    window: i32,
    mlp_ratio: f64,
    eps: f32,
    has_downsample: bool,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    require_positive("clap HTSAT stage: depth", depth)?;
    // mlp_ratio is pinned to 4.0 by config.validate; compute the Swin MLP hidden
    // width as round(dim * mlp_ratio) — exact for the integer ratio.
    let hidden = mlp_hidden(dim, mlp_ratio)?;
    // Effective window/shift per HF set_shift_and_window_size: at a resolution
    // <= window the shift is zeroed (and the window shrinks to min(H, W)). For
    // every HTSAT resolution the effective window equals the configured window,
    // so only the shift is gated.
    let min_res = input_height.min(input_width);
    let partitions_full = min_res > window;

    let mut blocks: Vec<SwinBlock> = Vec::new();
    reserve_or_error(&mut blocks, "clap HTSAT SwinBlock", depth.max(0) as usize)?;
    for i in 0..depth {
      let shift = if partitions_full && (i % 2 != 0) {
        window / 2
      } else {
        0
      };
      let prefix = format!("layers.{stage}.blocks.{i}");
      blocks.push(SwinBlock::from_weights(
        weights, &prefix, dim, num_heads, window, shift, hidden, eps, quant,
      )?);
    }

    let downsample = if has_downsample {
      Some(PatchMerging::from_weights(
        weights,
        &format!("layers.{stage}.downsample"),
        dim,
        eps,
        quant,
      )?)
    } else {
      None
    };

    Ok(Self {
      blocks,
      downsample,
      input_height,
      input_width,
    })
  }

  /// Run every block over `(B, L, C)` at this stage's resolution, then (if
  /// present) the `2×2` patch-merge. Returns the stage output and its
  /// `(out_height, out_width)` (halved when merged, unchanged otherwise).
  fn forward(&self, x: &Array) -> Result<(Array, i32, i32)> {
    let mut h = x.try_clone()?;
    for block in &self.blocks {
      h = block.forward(&h, self.input_height, self.input_width)?;
    }
    match &self.downsample {
      Some(merge) => {
        let merged = merge.forward(&h, self.input_height, self.input_width)?;
        Ok((merged, self.input_height / 2, self.input_width / 2))
      }
      None => Ok((h, self.input_height, self.input_width)),
    }
  }

  /// `true` if every block (and the downsample, if present) loaded quantized
  /// (test-only).
  #[cfg(test)]
  fn all_quantized(&self) -> bool {
    self.blocks.iter().all(|b| b.all_quantized())
      && self.downsample.as_ref().is_none_or(|d| d.is_quantized())
  }
}

// ══════════════════════════════ HtsatAudioTower ════════════════════════════

/// The CLAP HTSAT Swin-Transformer audio tower (`ClapAudioModel` /
/// `ClapAudioEncoder`).
///
/// Maps a `(batch, 1, time = T_FRAMES, freq = num_mel_bins)` log-mel
/// spectrogram (the [`super::mel`] front-end's output) to the clip-level
/// `(batch, hidden = 768)` pooled audio feature `ClapAudioModel.forward` returns
/// to the audio projection:
///
/// 1. `batch_norm` — the encoder per-mel-bin affine.
/// 2. `reshape_mel2img` — fold the mel into the `(spec_size, spec_size)` image.
/// 3. the patch-embed stem — the strided `Conv2d` patchify + `LayerNorm`.
/// 4. the four Swin stages — `depths` Swin blocks + `2×2` merges.
/// 5. the final `LayerNorm` + the token-semantic mean-pool.
///
/// Built via [`from_weights`](Self::from_weights) /
/// [`from_weights_quantized`](Self::from_weights_quantized). The phase-4
/// `ClapModel` assembly wraps it as the audio tower and feeds the pooled feature
/// to the audio projection + L2-normalize.
#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
pub struct HtsatAudioTower {
  batch_norm: EncoderBatchNorm,
  patch_embed: PatchEmbed,
  stages: Vec<AudioStage>,
  /// The final `LayerNorm(num_features = 768)` over the deepest-stage tokens.
  norm: LayerNorm,
  /// `spec_size` — the square mel-image side `reshape_mel2img` targets.
  spec_size: i32,
  /// `freq_ratio` — the time↔freq fold ratio (`spec_size / num_mel_bins`).
  freq_ratio: i32,
  /// `num_mel_bins` — the mel front-end's freq-bin count (the fold's input freq).
  num_mel_bins: i32,
  /// `patch_size` — the patch-embed stride (drives the pooled token count check).
  patch_size: i32,
  /// `len(depths)` — the stage count (drives the `2^(stages-1)` pool fold).
  num_stages: i32,
}

#[cfg(feature = "clap")]
#[cfg_attr(docsrs, doc(cfg(feature = "clap")))]
impl HtsatAudioTower {
  /// Build the audio tower from a validated [`ClapConfig`] and the (sanitized)
  /// weight map whose keys follow HF's `ClapAudioEncoder` tree:
  /// `batch_norm.{weight,bias,running_mean,running_var}`,
  /// `patch_embed.proj.{weight,bias}`, `patch_embed.norm.{weight,bias}`,
  /// `layers.{stage}.blocks.{i}.*` (the Swin block sub-tree), `layers.{stage}.
  /// downsample.{norm,reduction}.*` (the patch-merge, stages `0..len-1`), and
  /// `norm.{weight,bias}` (the final norm).
  pub fn from_weights(config: &ClapConfig, weights: &mut HashMap<String, Array>) -> Result<Self> {
    Self::from_weights_quantized(config, weights, None)
  }

  /// Build the audio tower with an optional parsed quantization config — the
  /// quantize-aware analogue of [`from_weights`](Self::from_weights).
  ///
  /// Each `nn.Linear` (the window q/k/v/output, the Swin MLP fc1/fc2, the
  /// patch-merge reductions) auto-picks the dense or quantized variant per layer
  /// by its `<prefix>.scales` sibling. The patch-embed `Conv2d` weight and the
  /// batch-norm / `LayerNorm` parameters always load dense (a conv / norm is not
  /// a quantized `Linear`).
  pub fn from_weights_quantized(
    config: &ClapConfig,
    weights: &mut HashMap<String, Array>,
    quant: Option<&PerLayerQuantization>,
  ) -> Result<Self> {
    // Idempotent re-validation: `from_weights*` is public, so a caller may build
    // from a directly-constructed (unvalidated) config. This pins every audio
    // dim (depths, heads, spec_size, freq_ratio, window, hidden, …) before the
    // per-stage build / reshape arithmetic.
    config.validate()?;
    let audio = &config.audio_config;
    let eps = audio.layer_norm_eps as f32;
    let window = audio.window_size;
    let mlp_ratio = audio.mlp_ratio;
    let patch_hidden = audio.patch_embeds_hidden_size;
    let num_stages = audio.depths.len() as i32;

    let batch_norm = EncoderBatchNorm::from_weights(weights, audio.num_mel_bins, BATCH_NORM_EPS)?;
    let patch_embed = PatchEmbed::from_weights(
      weights,
      audio.patch_embed_input_channels,
      patch_hidden,
      audio.patch_size,
      eps,
    )?;

    // The stage-0 patch grid side: spec_size / patch_size (the patch-embed
    // produces a (grid, grid) token map). Each stage halves the resolution and
    // doubles the channel width (96 → 192 → 384 → 768).
    require_positive("clap HTSAT: patch_size", audio.patch_size)?;
    require_divisible(
      "clap HTSAT: spec_size",
      audio.spec_size,
      "clap HTSAT: patch_size",
      audio.patch_size,
    )?;
    let grid0 = audio.spec_size / audio.patch_size;

    let mut stages: Vec<AudioStage> = Vec::new();
    reserve_or_error(&mut stages, "clap HTSAT AudioStage", audio.depths.len())?;
    for (i, (&depth, &num_heads)) in audio
      .depths
      .iter()
      .zip(audio.num_attention_heads.iter())
      .enumerate()
    {
      let stage = i as i32;
      // dim = patch_hidden << i ; resolution = grid0 >> i (both bounded — the
      // 4-stage hierarchy on a 64-wide grid stays well within i32).
      let dim = checked_mul(
        "clap HTSAT: stage channel width",
        "patch_embeds_hidden_size",
        patch_hidden,
        "2^stage",
        1i32 << stage,
      )?;
      let resolution = grid0 >> stage;
      if resolution < 1 {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "clap HTSAT: stage resolution",
          "must stay >= 1 across every stage (spec_size / patch_size too small)",
          format_smolstr!("{resolution}"),
        )));
      }
      let has_downsample = stage < num_stages - 1;
      stages.push(AudioStage::from_weights(
        weights,
        stage,
        dim,
        num_heads,
        depth,
        resolution,
        resolution,
        window,
        mlp_ratio,
        eps,
        has_downsample,
        quant,
      )?);
    }

    // The final norm is over num_features = patch_hidden << (num_stages - 1).
    let num_features = checked_mul(
      "clap HTSAT: num_features",
      "patch_embeds_hidden_size",
      patch_hidden,
      "2^(stages-1)",
      1i32 << (num_stages - 1),
    )?;
    let norm = build_layer_norm(weights, "norm", num_features, eps)?;

    Ok(Self {
      batch_norm,
      patch_embed,
      stages,
      norm,
      spec_size: audio.spec_size,
      freq_ratio: audio.freq_ratio,
      num_mel_bins: audio.num_mel_bins,
      patch_size: audio.patch_size,
      num_stages,
    })
  }

  /// Forward a `(batch, 1, time = T_FRAMES, freq = num_mel_bins)` log-mel
  /// spectrogram to the clip-level `(batch, hidden = 768)` pooled audio feature.
  ///
  /// Mirrors `ClapAudioEncoder.forward`: batch-norm → `reshape_mel2img` →
  /// patch-embed → the four Swin stages → final `LayerNorm` → the token-semantic
  /// mean-pool. `input_features` is pinned to rank-4 `(B, 1, T, F)` (the public
  /// forward accepts an untrusted array; every downstream reshape assumes that
  /// shape).
  pub fn forward(&self, input_features: &Array) -> Result<Array> {
    let shape = input_features.shape();
    if shape.len() != 4 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "clap HTSAT: input_features must be rank-4 (batch, 1, time, freq)",
        shape.len() as u32,
        shape,
      )));
    }
    let channels = dim_i32(&shape, 1, "clap HTSAT: input channels")?;
    let freq = dim_i32(&shape, 3, "clap HTSAT: input freq")?;
    if channels != 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap HTSAT: input channel axis",
        "must be 1 (a single-channel mel image)",
        format_smolstr!("{channels}"),
      )));
    }
    if freq != self.num_mel_bins {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap HTSAT: input freq axis",
        "must equal num_mel_bins",
        format_smolstr!("{freq}"),
      )));
    }

    // 1. encoder batch-norm (per-mel-bin affine over the running statistics).
    let normalized = self.batch_norm.forward(input_features)?;

    // 2. reshape_mel2img: fold the mel into the (spec_size, spec_size) image.
    let image = reshape_mel2img(&normalized, self.spec_size, self.freq_ratio)?;
    // frames_num = the image height (HF reads hidden_states.shape[2] here) — the
    // pooled token-fold factor depends on it.
    let frames_num = dim_i32(&image.shape(), 2, "clap HTSAT: reshape_mel2img height")?;

    // 3. patch-embed → (B, grid·grid, hidden) tokens + the (H', W') resolution.
    let (mut hidden, _h0, _w0) = self.patch_embed.forward(&image)?;

    // 4. the four Swin stages.
    for stage in &self.stages {
      let (out, _h, _w) = stage.forward(&hidden)?;
      hidden = out;
    }

    // 5. final norm + token-semantic mean-pool → (B, hidden).
    let normed = self.norm.forward(&hidden)?;
    self.token_semantic_pool(&normed, frames_num)
  }

  /// The token-semantic mean-pool (HF `ClapAudioEncoder.forward` ~L1470-1495):
  /// average every spatial token per channel to the clip-level `(B, hidden)`
  /// feature.
  ///
  /// HF permutes the post-norm `(B, L, hidden)` tokens to a
  /// `(B, hidden, freq, temporal)` grid, re-folds the freq-blocks, and applies
  /// `AdaptiveAvgPool1d(1)` over the flattened spatial axis. The result is the
  /// mean of all `L` tokens per channel (a mean is invariant to that
  /// rearrangement), so this computes it directly as the token-axis mean — after
  /// pinning `L` against HF's expected `freq · c_freq_bin · temporal` token count
  /// (a guard that the upstream stages produced the resolution the pool fold
  /// assumes).
  fn token_semantic_pool(&self, normed: &Array, frames_num: i32) -> Result<Array> {
    let shape = normed.shape();
    let tokens = dim_i32(&shape, 1, "clap HTSAT pool: token count")?;
    // HF: freq_shape = temporal_shape = frames_num // 2^(stages-1) // patch_stride.
    let fold = checked_mul(
      "clap HTSAT pool: 2^(stages-1) · patch_stride",
      "2^(stages-1)",
      1i32 << (self.num_stages - 1),
      "patch_stride",
      self.patch_size,
    )?;
    require_positive("clap HTSAT pool: fold factor", fold)?;
    let freq_shape = frames_num / fold;
    let temporal_shape = frames_num / fold;
    let expected_tokens = checked_mul(
      "clap HTSAT pool: freq_shape · temporal_shape",
      "freq_shape",
      freq_shape,
      "temporal_shape",
      temporal_shape,
    )?;
    if tokens != expected_tokens {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "clap HTSAT pool: post-norm token count",
        "must equal HF freq_shape · temporal_shape (the pooled spatial grid)",
        format_smolstr!("{tokens} != {expected_tokens}"),
      )));
    }
    // AdaptiveAvgPool1d(1) over the flattened spatial axis == mean over the token
    // axis (rearrangement-invariant). keepdims=false → (B, hidden).
    ops::reduction::mean_axes(normed, &[1], false)
  }

  /// `true` if every stage's Swin Linears (and the patch-merge reductions)
  /// loaded the quantized variant (test-only).
  #[cfg(test)]
  pub(crate) fn all_swin_linears_quantized(&self) -> bool {
    self.stages.iter().all(|s| s.all_quantized())
  }
}

// ═══════════════════════════════ free functions ════════════════════════════

/// The Swin MLP hidden width `round(dim · mlp_ratio)` (HF
/// `ClapAudioStage` → `ClapAudioLayer` → `ClapAudioIntermediate`'s
/// `int(config.mlp_ratio * dim)`). `mlp_ratio` is pinned to `4.0`, so this is the
/// exact `4 · dim`; the rounded form keeps it correct if the ratio were ever a
/// non-integer. Erroring on overflow.
#[cfg(feature = "clap")]
fn mlp_hidden(dim: i32, mlp_ratio: f64) -> Result<i32> {
  require_positive("clap HTSAT: stage dim", dim)?;
  // (mlp_ratio · dim) rounded to nearest; mlp_ratio is a small positive constant
  // (4.0) and dim <= 768, so the product is exact in f64 and fits i32.
  let hidden = (mlp_ratio * dim as f64).round();
  if !(hidden.is_finite() && (1.0..=(i32::MAX as f64)).contains(&hidden)) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap HTSAT: Swin MLP hidden width",
      "mlp_ratio · dim must be a positive in-range integer",
      format_smolstr!("{hidden}"),
    )));
  }
  Ok(hidden as i32)
}

/// HF `ClapAudioEncoder.reshape_mel2img` (`modeling_clap.py` ~L1370-1410): fold a
/// `(batch, 1, time, freq)` log-mel into the `(batch, 1, spec_size, spec_size)`
/// image the Swin stem expects.
///
/// Ported verbatim (the encoder's `freq_ratio = spec_size // num_mel_bins`
/// equals the config `freq_ratio` for the unfused checkpoint):
///
/// ```text
/// spec_width  = spec_size * freq_ratio
/// spec_height = spec_size / freq_ratio
/// if time < spec_width:  bicubic(align_corners=True) time → spec_width
/// if freq < spec_height: bicubic(align_corners=True) freq → spec_height
/// (time, freq are now spec_width, spec_height)
/// reshape (B, channels * freq_ratio, time // freq_ratio, freq)
/// permute (0, 1, 3, 2)
/// reshape (B, channels, freq * freq_ratio, time // freq_ratio)
/// ```
///
/// `channels` is `1` for the unfused single-mel path. The two interpolations use
/// the `align_corners=True` bicubic HF's `nn.functional.interpolate(...,
/// mode="bicubic", align_corners=True)` uses
/// ([`bicubic_interpolate_align_corners`]); a `time`/`freq` already at the target
/// extent skips its interpolation (HF's `if … <` guard). A `time > spec_width`
/// or `freq > spec_height` (HF's `ValueError`) is a typed [`Error::OutOfRange`].
#[cfg(feature = "clap")]
pub(crate) fn reshape_mel2img(mel: &Array, spec_size: i32, freq_ratio: i32) -> Result<Array> {
  require_positive("clap HTSAT reshape_mel2img: spec_size", spec_size)?;
  require_positive("clap HTSAT reshape_mel2img: freq_ratio", freq_ratio)?;
  let shape = mel.shape();
  if shape.len() != 4 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "clap HTSAT reshape_mel2img: input must be rank-4 (batch, channels, time, freq)",
      shape.len() as u32,
      shape,
    )));
  }
  let batch = dim_i32(&shape, 0, "clap HTSAT reshape_mel2img: batch")?;
  let channels = dim_i32(&shape, 1, "clap HTSAT reshape_mel2img: channels")?;
  let time = dim_i32(&shape, 2, "clap HTSAT reshape_mel2img: time")?;
  let freq = dim_i32(&shape, 3, "clap HTSAT reshape_mel2img: freq")?;

  // spec_width = spec_size * freq_ratio ; spec_height = spec_size / freq_ratio.
  let spec_width = checked_mul(
    "clap HTSAT reshape_mel2img: spec_width",
    "spec_size",
    spec_size,
    "freq_ratio",
    freq_ratio,
  )?;
  require_divisible(
    "clap HTSAT reshape_mel2img: spec_size",
    spec_size,
    "clap HTSAT reshape_mel2img: freq_ratio",
    freq_ratio,
  )?;
  let spec_height = spec_size / freq_ratio;

  // HF raises if the wav is larger than the swin input on either axis.
  if time > spec_width || freq > spec_height {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "clap HTSAT reshape_mel2img: input size",
      "time must be <= spec_size·freq_ratio and freq <= spec_size/freq_ratio",
      format_smolstr!("time {time} / freq {freq}"),
    )));
  }

  // Upsample the time axis to spec_width (and the freq axis to spec_height) with
  // align_corners=True bicubic — HF's two `if … < …: interpolate(...)` steps.
  // bicubic_interpolate_align_corners is rank-4 (B, C, H, W) with the spatial
  // axes last two — exactly the mel's (batch, channels, time, freq) layout.
  let mut x = mel.try_clone()?;
  if time < spec_width {
    x = ops::interpolation::bicubic_interpolate_align_corners(
      &x,
      spec_width as usize,
      freq as usize,
    )?;
  }
  if freq < spec_height {
    let cur_time = dim_i32(
      &x.shape(),
      2,
      "clap HTSAT reshape_mel2img: time (post-interp)",
    )?;
    x = ops::interpolation::bicubic_interpolate_align_corners(
      &x,
      cur_time as usize,
      spec_height as usize,
    )?;
  }

  // Re-read the (possibly interpolated) time/freq extents.
  let time = dim_i32(&x.shape(), 2, "clap HTSAT reshape_mel2img: time (folded)")?;
  let freq = dim_i32(&x.shape(), 3, "clap HTSAT reshape_mel2img: freq (folded)")?;
  // reshape (B, channels * freq_ratio, time // freq_ratio, freq).
  require_divisible(
    "clap HTSAT reshape_mel2img: time",
    time,
    "clap HTSAT reshape_mel2img: freq_ratio",
    freq_ratio,
  )?;
  let channels_x = checked_mul(
    "clap HTSAT reshape_mel2img: channels · freq_ratio",
    "channels",
    channels,
    "freq_ratio",
    freq_ratio,
  )?;
  let time_div = time / freq_ratio;
  let x = ops::shape::reshape(&x, &[batch, channels_x, time_div, freq])?;
  // permute (0, 1, 3, 2) → (B, channels·freq_ratio, freq, time // freq_ratio).
  let x = ops::shape::transpose_axes(&x, &[0, 1, 3, 2])?;
  // reshape (B, channels, freq · freq_ratio, time // freq_ratio).
  let freq_x = checked_mul(
    "clap HTSAT reshape_mel2img: freq · freq_ratio",
    "freq",
    freq,
    "freq_ratio",
    freq_ratio,
  )?;
  ops::shape::reshape(&x, &[batch, channels, freq_x, time_div])
}

#[cfg(all(test, feature = "clap"))]
mod tests;
