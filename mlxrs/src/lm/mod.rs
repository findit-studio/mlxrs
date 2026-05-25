//! Language Model (LM) support — text-only inference building blocks ported
//! from [mlx-lm](https://github.com/ml-explore/mlx-lm).
//!
//! M3 lands the sampling utilities ([`crate::lm::sample`]), the
//! architecture-agnostic generation foundation — the
//! [`Model`](crate::lm::model::Model) trait, the KV
//! [`cache`](crate::lm::cache), the model-load support surface
//! ([`crate::lm::load`]), the local load [`factory`](crate::lm::factory)
//! (`ModelConfiguration` + `model_type` registry + `load`), and the
//! [`generate`](crate::lm::generate) loop (`generate_step` /
//! `stream_generate` / `generate` + `make_sampler` /
//! `make_logits_processors`), the [`perplexity`](crate::lm::perplexity)
//! evaluation (`perplexity` / `make_windows` / `cross_entropy_none`), and the
//! prompt-cache fill+save driver ([`crate::lm::cache_prompt`] — `cache_prompt`,
//! the support-surface port of `mlx_lm.cache_prompt`).
//!
//! M3 also lands inference-time **LoRA/DoRA adapter loading**
//! ([`crate::lm::lora`]) — `LoRALinear` / `DoRALinear` (+ their quantized
//! `QLoRA` / `QDoRA` bases), `fuse`, `linear_to_lora_layers`, and
//! `load_adapters` — the runtime surface that applies a pre-trained adapter
//! (`adapter_config.json` + `adapters.safetensors`) to a base model's weight
//! map (mlx-lm `tuner/{lora,dora,utils}.py`, swift `Adapters/LoRA/`); the
//! [`crate::lm::fuse::fuse`] driver wires `load_adapters` + the per-layer
//! [`fuse`](crate::lm::lora::LoraLayer::fuse) + the F6 save into a one-call
//! "fold the adapter into the base model and write the result as a
//! standalone checkpoint" pipeline (mlx-lm `fuse.py`).
//!
//! M3 also lands the stateful multi-turn chat
//! [`session`](crate::lm::session) ([`ChatSession`] — the port of
//! mlx-swift-lm's `ChatSession`: a type owning the model, tokenizer, KV
//! cache and conversation history that reuses the cache across `respond`
//! turns).
//!
//! [`ChatSession`]: crate::lm::session::ChatSession

pub mod cache;
pub mod cache_prompt;
pub mod convert;
pub mod factory;
pub mod fuse;
pub mod generate;
#[cfg(feature = "gguf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gguf")))]
pub mod gguf;
pub mod load;
pub mod lora;
pub mod model;
pub mod nn;
pub mod perplexity;
pub mod quant;
pub mod sample;
pub mod session;
pub mod speculative;
pub mod stop;
/// Grammar-constrained decoding — port of `mlx_vlm/structured.py` (V6,
/// issue #180). [`LLGuidanceLogitsProcessor`] +
/// [`build_json_schema_logits_processor`] mask each step's logits down
/// to the tokens that keep the next sequence valid against a JSON
/// schema / regex / Lark grammar. Backed by the
/// [`llguidance`](https://crates.io/crates/llguidance) crate; gated on
/// the `llguidance` cargo feature so the default `lm` build doesn't pay
/// the grammar-engine compile cost.
///
/// [`LLGuidanceLogitsProcessor`]: crate::lm::structured::LLGuidanceLogitsProcessor
/// [`build_json_schema_logits_processor`]: crate::lm::structured::build_json_schema_logits_processor
#[cfg(feature = "llguidance")]
#[cfg_attr(docsrs, doc(cfg(feature = "llguidance")))]
pub mod structured;
/// Tool-call format parsers — Python `mlx_lm.tool_parsers.*`.
///
/// Surface re-export of [`crate::tokenizer::tools`] under the canonical
/// `lm::tool_parsers` path; the parser logic and per-format documentation
/// live in the module that owns the [`crate::tokenizer::wrapper::Tokenizer`]
/// consumer. Gated on the `tokenizer-tools` capability feature, which the
/// `lm` umbrella always pulls in.
#[cfg(feature = "tokenizer-tools")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
pub mod tool_parsers;
pub mod tuner;
