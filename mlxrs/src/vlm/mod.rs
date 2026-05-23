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
//! - [`crate::vlm::feature_cache`] — an LRU cache of vision-encoder output
//!   features keyed by image identity, ported 1:1 from
//!   `mlx-vlm/mlx_vlm/vision_cache.py::VisionFeatureCache`. Lets a VLM
//!   discussing the same image across multiple turns re-use the cached
//!   embeddings instead of re-running the (expensive) vision encoder.
//! - [`crate::vlm::load`] — local VLM **load factory** + a
//!   (`model_type`, `processor_class`) → constructor registry pair,
//!   the VLM analog of [`crate::lm::factory`]. Reads the model's
//!   `config.json` once, the VLM processor config
//!   (`preprocessor_config.json` preferred, `processor_config.json`
//!   fallback) once, validates both registries early, then loads
//!   weights / tokenizer / constructs model + processor. Local-only;
//!   no Hugging Face Hub download.
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
//! - [`crate::vlm::inputs`] — multimodal **input assembly** (V4): the
//!   branch-dispatch + padding-side core of
//!   `mlx-vlm/mlx_vlm/utils.py::prepare_inputs` (lines 1173–1449),
//!   plus the VLM-side audio/video glue wrappers
//!   ([`crate::vlm::inputs::read_audio`],
//!   [`crate::vlm::inputs::load_audio_vlm`],
//!   [`crate::vlm::inputs::normalize_audio_features`],
//!   [`crate::vlm::inputs::load_video`]). The audio glue is gated on
//!   both `vlm` AND `audio` (it bridges the two subsystems).
//! - [`crate::vlm::video`] — model-agnostic video *preprocessing* math
//!   ported from `mlx-vlm/mlx_vlm/video_generate.py` (`smart_resize`,
//!   `smart_nframes`, frame-index sampling, and a `process_frames` that
//!   reuses [`image::preprocess`](crate::vlm::image::preprocess) per
//!   frame + stacks). **Container
//!   decoding (mp4 → frames) is intentionally NOT here** — it needs a
//!   codec dependency and is a documented follow-up; this module ports
//!   the portable sampling + resize + prep arithmetic and takes
//!   caller-decoded frames.

pub mod feature_cache;
pub mod generate;
pub mod image;
pub mod inputs;
pub mod load;
pub mod model;
pub mod prompt;
pub(crate) mod resize;
pub mod video;
