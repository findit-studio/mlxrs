//! `mlx.fast.*` — JIT-compiled custom Metal kernels and other "fast" surface
//! ops fused at the mlx-c level (per-tensor primitives like `rms_norm` /
//! `rope` / `scaled_dot_product_attention` live in their own headers and are
//! wrapped separately; this module currently houses the
//! [`metal_kernel`] subset that exposes user-authored Metal Shading
//! Language kernels).
//!
//! Mirrors `mlx-swift`'s `Source/MLX/MLXFastKernel.swift` (the `MLXFast.MLXFastKernel`
//! container + `MLXFast.metalKernel` factory) and the python `mlx.fast.metal_kernel`
//! callable. The Rust surface follows the upstream split: a compiled
//! `metal_kernel::MetalKernel` holds the long-lived kernel handle, and each
//! invocation supplies an [`metal_kernel::MetalKernelApplyConfig`] describing
//! the per-call grid / thread-group / output shapes / template arguments.
//!
//! Custom Metal kernels require a real Metal device at apply time. Tests
//! exercising the apply path are gated behind `#[cfg(target_os = "macos")]`
//! plus `#[ignore]` so they only run with `cargo test -- --ignored` on a host
//! that actually has a Metal-capable GPU.

pub mod metal_kernel;
