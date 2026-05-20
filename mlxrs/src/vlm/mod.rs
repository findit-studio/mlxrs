//! Vision-Language Model (VLM) — multimodal support layer.
//!
//! This module hosts the shared *support* surface for VLM inference:
//! image preprocessing primitives (this PR), VLM chat-template / prompt
//! assembly, and the multimodal generate loop (subsequent PRs). Per
//! [`crate::vlm`]'s module doc on the crate root, **per-model
//! architectures** (Qwen-VL / LLaVA / etc.) are added per-usecase and are
//! NOT bulk-ported from `mlx-vlm/models/` — only the cross-model support
//! surface lives here.
//!
//! ## Submodules
//! - [`image`] — core image preprocessing primitives ported 1:1 from
//!   `mlx-swift-lm`'s `Libraries/MLXVLM/MediaProcessing.swift`
//!   (load / resize / channel-extract / rescale / normalize / patchify
//!   / `preprocess` pipeline composer). Decoupled from any model
//!   architecture; the per-model image-processor layer
//!   (`clip_image_processor`, `siglip_image_processor`, …) is per-usecase.

pub mod image;
