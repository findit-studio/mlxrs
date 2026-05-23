//! Tool-call format parsers (Python `mlx_lm/tool_parsers/`).
//!
//! Surface façade re-exporting the concrete parser types and the
//! dispatcher / `_infer_tool_parser` selector that already live under
//! [`crate::tokenizer::tools`]. The parser *logic* is owned by the
//! `tokenizer-tools` capability feature (consumed by
//! [`crate::tokenizer::Tokenizer`] for in-line tool-call decoding
//! during generation); this `lm::tool_parsers` module is the canonical
//! caller-visible entry point under the `lm` umbrella, matching Python's
//! `mlx_lm.tool_parsers.<format>` import path.
//!
//! ## Reference mapping
//!
//! Each parser mirrors one Python module in `mlx-lm`'s
//! `mlx_lm/tool_parsers/` directory (mlx-lm df1d3f3):
//!
//! - [`crate::tokenizer::tools::JsonTools`] — `json_tools.py`
//!   (`<tool_call>{json}</tool_call>` body).
//! - [`crate::tokenizer::tools::Pythonic`] — `pythonic.py`
//!   (`[name(a="x", n=2)]`).
//! - [`crate::tokenizer::tools::Mistral`] — `mistral.py`
//!   (`name[ARGS]{json}`).
//! - [`crate::tokenizer::tools::Qwen3Coder`] — `qwen3_coder.py`
//!   (`<function=name><parameter=p>v</parameter></function>`).
//! - [`crate::tokenizer::tools::Glm47`] — `glm47.py`
//!   (`name<arg_key>k</arg_key><arg_value>v</arg_value>` plus JSON /
//!   plain-text fallbacks).
//! - [`crate::tokenizer::tools::KimiK2`] — `kimi_k2.py`
//!   (`functions.name:0<|tool_call_argument_begin|>{json}` per call, multi
//!   split by `<|tool_call_begin|>...<|tool_call_end|>`).
//! - [`crate::tokenizer::tools::Longcat`] — `longcat.py`
//!   (`name<longcat_arg_key>k</longcat_arg_key><longcat_arg_value>v</longcat_arg_value>`).
//! - [`crate::tokenizer::tools::MinimaxM2`] — `minimax_m2.py`
//!   (`<invoke name="n"><parameter name="p">v</parameter></invoke>`).
//! - [`crate::tokenizer::tools::FunctionGemma`] — `function_gemma.py`
//!   (`call:name{k:v,...}` with `<escape>`-delimited strings).
//! - [`crate::tokenizer::tools::Gemma4`] — `gemma4.py`
//!   (`call:name{bare_key: ...}` with balanced braces and `<|"|>...<|"|>`
//!   string literals).
//!
//! ## Public surface
//!
//! - [`crate::tokenizer::tools::ToolCall`] — parsed `{name, arguments, id?}`
//!   shape.
//! - [`crate::tokenizer::tools::ToolParser`] —
//!   `parse(text, tools) -> Result<Vec<ToolCall>>` trait, with `name()` +
//!   `tool_call_start()` / `tool_call_end()` markers.
//! - [`crate::tokenizer::tools::ToolCallProcessor`] — the streaming
//!   state-machine that detects and extracts tool calls mid-generation,
//!   fed text chunk-by-chunk (port of `mlx-swift-lm`'s `ToolCallProcessor`).
//! - [`crate::tokenizer::tools::parser_by_name`] — dispatcher mirroring
//!   Python's `importlib.import_module("mlx_lm.tool_parsers.<name>")`.
//! - [`crate::tokenizer::tools::infer_tool_parser`] —
//!   chat-template-driven auto-selection mirroring
//!   `mlx_lm.tokenizer_utils._infer_tool_parser`.
//!
//! All re-exports are stable proxies; the underlying definitions and tests
//! live alongside the consumer ([`crate::tokenizer`]) under the
//! `tokenizer-tools` feature, which the `lm` umbrella pulls.

pub use crate::tokenizer::tools::{
  FunctionGemma, Gemma4, Glm47, JsonTools, KimiK2, Longcat, MinimaxM2, Mistral, Pythonic,
  Qwen3Coder, ToolCall, ToolCallProcessor, ToolParser, infer_tool_parser, parser_by_name,
};
