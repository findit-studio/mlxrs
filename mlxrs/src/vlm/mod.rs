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
//! - [`crate::vlm::model`] — the `vlm::Model` trait extending
//!   `lm::Model` with the image-embedding entry points every VLM forward
//!   needs (vision encode + token embed + image-into-text embed splice).
//!   Mirrors mlx-vlm's `VisionLanguageModel` protocol and mlx-swift-lm's
//!   `VLMModel` marker.
//! - [`crate::vlm::generate`] — the architecture-agnostic multimodal
//!   generation Iterator, ported from `mlx-vlm/mlx_vlm/generate.py::
//!   generate_step` (preprocess images → vision encode → embed merge →
//!   prefill via `forward_embeddings` → per-token decode via `forward`).

pub mod generate;
pub mod image;
pub mod model;
pub mod prompt;
