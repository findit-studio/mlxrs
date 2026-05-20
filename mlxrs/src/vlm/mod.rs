//! Vision-Language Model (VLM) — multimodal support layer.
//!
//! Hosts the shared *support* surface for VLM inference: image
//! preprocessing primitives, prompt-assembly primitives, and the
//! multimodal generate loop (later PRs). Per the project's
//! no-per-model-arch rule, **per-model architectures** (Qwen-VL /
//! LLaVA / etc.) are added per-usecase and are NOT bulk-ported from
//! `mlx-vlm/models/` — only the cross-model support surface lives here.
//!
//! ## Submodules
//! - [`image`] — core image preprocessing primitives ported 1:1 from
//!   `mlx-swift-lm`'s `Libraries/MLXVLM/MediaProcessing.swift`
//!   (load / resize / channel-extract / rescale / normalize / patchify
//!   / `preprocess` pipeline composer). Decoupled from any model
//!   architecture; the per-model image-processor layer
//!   (`clip_image_processor`, `siglip_image_processor`, …) is per-usecase.
//! - [`crate::vlm::prompt`] — model-agnostic multimodal prompt-assembly
//!   primitives ported 1:1 from `mlx-vlm/mlx_vlm/prompt_utils.py`
//!   (image-token splicing, image-span location, multimodal attention-mask
//!   construction). Per-model chat templates are out of scope.

pub mod image;
pub mod prompt;
