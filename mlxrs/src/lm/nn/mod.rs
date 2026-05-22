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

pub mod rope;
pub mod rope_scaling;

pub use rope::{
  Rope, RopeOffsetRef, rope, rope_dynamic, rope_dynamic_with_freqs, rope_with_freqs,
  rope_with_freqs_offset, rope_with_offset,
};
pub use rope_scaling::{Llama3Rope, Llama3ScalingConfig, SuScaledRope, YarnConfig, YarnRope};
