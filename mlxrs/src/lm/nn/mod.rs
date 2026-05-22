//! Neural-network primitives ported from `mlx.nn`
//! (python `python/mlx/nn/layers/`) and the mlx-swift `MLXNN` / `MLXLMCommon`
//! layers, scoped to what the `lm` inference stack composes.
//!
//! M-N1 lands the base **Rotary Position Embedding**
//! ([`mod@rope`]) — the standard / "traditional" RoPE that backs every
//! attention layer's positional encoding (mlx-lm's `nn.RoPE`, swift's
//! `RoPE` + `MLXFast.RoPE`).
//!
//! The scaled RoPE variants (Llama3 / Su-scaled (longrope) / YaRN — swift
//! `Llama3RoPE` / `SuScaledRoPE` / `YarnRoPE` in `MLXLMCommon/RoPEUtils.swift`,
//! python `mlx_lm/models/rope_utils.py`) land in
//! [`mod@rope_scaling`]. Each precomputes a per-dimension `freqs` array and
//! forwards it through the same `mlx_fast_rope` primitive with `base = None`,
//! via the freqs-path entry points [`rope::rope_with_freqs`] /
//! [`rope::rope_dynamic_with_freqs`] now exposed alongside the base
//! [`rope::rope`] (`base`) path.
//!
//! M-N2 adds the fast scaled-dot-product **attention** primitive
//! ([`mod@attention`]) — a 1:1 wrap of mlx's
//! `mx.fast.scaled_dot_product_attention` /
//! `MLXFast.scaledDotProductAttention` (`mlx_fast_scaled_dot_product_attention`),
//! covering Multi-Head, Grouped Query, and Multi-Query attention with
//! `None` / `Causal` / explicit-array masks.
//!
//! The cache-aware quantized routing variant of attention
//! (swift `attentionWithCacheUpdate`'s `QuantizedKVCacheProtocol` branch
//! dispatching to `quantizedScaledDotProductAttention`) and the attention
//! `sinks` argument are likewise deliberately out of scope here — both are
//! follow-ups layered on top of the base [`attention::scaled_dot_product_attention`].
//!
//! M-N3 lands the **Mixture-of-Experts Switch** primitives ([`mod@switch`]):
//! `SwitchLinear` / `QuantizedSwitchLinear` — the per-token expert-routed
//! linear layers backing mlx-lm's `SwitchMLP` / `SwitchGLU` (and every MoE
//! model in `mlx-lm/mlx_lm/models/`). The higher-level `SwitchMLP` /
//! `SwitchGLU` blocks depend on activation primitives (`gelu` / `silu` /
//! `swiglu`) not yet ported and are intentionally deferred.
//!
//! M-N4 adds the **normalization** primitives ([`mod@norm`]):
//! [`RMSNorm`] / [`LayerNorm`] (both wrapping the fused `mlx_fast_*`
//! kernels — same primitives mlx-lm's `nn.RMSNorm` / `nn.LayerNorm` and
//! swift `RMSNorm` / `LayerNorm` delegate to) and [`GroupNorm`]
//! (no fused kernel; reproduced via [`crate::ops`]). The
//! `BatchNorm` / `InstanceNorm` siblings are deferred — these three
//! cover ~all transformer LM/VLM use.

pub mod attention;
pub mod norm;
pub mod rope;
pub mod rope_scaling;
pub mod switch;

pub use attention::{Mask, scaled_dot_product_attention};
pub use norm::{GroupNorm, LayerNorm, RMSNorm};
pub use rope::{
  Rope, RopeOffsetRef, rope, rope_dynamic, rope_dynamic_with_freqs, rope_with_freqs,
  rope_with_freqs_offset, rope_with_offset,
};
pub use rope_scaling::{Llama3Rope, Llama3ScalingConfig, SuScaledRope, YarnConfig, YarnRope};
pub use switch::{QuantizedSwitchLinear, SwitchLinear};
