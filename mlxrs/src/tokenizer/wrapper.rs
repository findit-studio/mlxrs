//! `Tokenizer` — the HF-tokenizer + detokenizer wrapper.
//!
//! Ports Python `mlx_lm/tokenizer_utils.py` `TokenizerWrapper` (line 287+) +
//! `_infer_thinking` + `load`, cross-referenced against `mlx-swift-lm`'s
//! `MLXLMCommon/Tokenizer.swift` `Tokenizer` protocol (`encode` /
//! `decode` / `convertTokenToId` / `convertIdToToken` / `bosToken` /
//! `eosToken` / `unknownToken` / `applyChatTemplate`).
//!
//! Loads strictly from local paths (`tokenizer.json` + `tokenizer_config.json`
//! in `model_path`). No Hugging Face Hub network download — that is the
//! caller's responsibility, matching the spec constraint.
//!
//! **serde_json-free core.** With only the bare `tokenizer` feature the
//! wrapper is built purely from [`HfTokenizer::from_file`]: `encode`/`decode`,
//! `convert_token_to_id`/`convert_id_to_token`, the
//! `tokenizer.json`-derived thinking inference and (with `tokenizer-stream`)
//! the detokenizer factory all work with **no `serde_json` on the code
//! path**. The `tokenizer_config.json` read, the parsed-config field and
//! every config-derived accessor (bos/eos/unk/pad, `chat_template`,
//! `has_chat_template`) are `#[cfg(feature = "tokenizer-config")]`-gated and
//! *absent* (not merely empty) without it.

use std::path::Path;

#[cfg(any(
  feature = "tokenizer-config",
  feature = "tokenizer-spm",
  feature = "tokenizer-bpe"
))]
use serde_json::Value;
use tokenizers::Tokenizer as HfTokenizer;

use crate::Error;

#[cfg(feature = "tokenizer-chat")]
use super::chat;
#[cfg(feature = "tokenizer-deepseek-v32")]
use super::chat::ChatTemplateOverride;
#[cfg(feature = "tokenizer-bpe")]
use super::stream::BpeStreamingDetokenizer;
#[cfg(feature = "tokenizer-spm")]
use super::stream::SpmStreamingDetokenizer;
#[cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))]
use super::stream::infer_detokenizer_class;
#[cfg(feature = "tokenizer-stream")]
use super::stream::{DetokenizerClass, NaiveStreamingDetokenizer, StreamingDetokenizer};
#[cfg(feature = "tokenizer-tools")]
use super::tools::{self, ToolParser};

/// A boxed streaming detokenizer (the `detokenizer()` factory return).
#[cfg(feature = "tokenizer-stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
pub type BoxedDetokenizer = Box<dyn StreamingDetokenizer>;

/// Result of `_infer_thinking`: start/end marker strings and their token ids.
#[derive(Debug, Clone, Default)]
struct Thinking {
  start: Option<String>,
  end: Option<String>,
  start_tokens: Option<Vec<u32>>,
  end_tokens: Option<Vec<u32>>,
}

/// HF tokenizer + detokenizer wrapper (Python `TokenizerWrapper`,
/// Swift `Tokenizer`).
pub struct Tokenizer {
  hf: HfTokenizer,
  /// The parsed `tokenizer_config.json`. Absent without `tokenizer-config`
  /// (serde_json-free core).
  #[cfg(feature = "tokenizer-config")]
  config: Value,
  /// The inferred streaming-detokenizer class. Without `tokenizer-spm` /
  /// `tokenizer-bpe` the `decoder` node is never parsed so this is always
  /// [`DetokenizerClass::Naive`].
  #[cfg(feature = "tokenizer-stream")]
  detok_class: DetokenizerClass,
  /// `clean_up_tokenization_spaces` (from config; only consumed by the
  /// SPM/BPE/naive streaming-detokenizer factory).
  #[cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream"))]
  clean_up_spaces: bool,
  eos_token_ids: std::collections::BTreeSet<u32>,
  /// The jinja `chat_template` string. Only consumed by
  /// `apply_chat_template` (so gated on `tokenizer-chat`).
  #[cfg(feature = "tokenizer-chat")]
  chat_template: Option<String>,
  #[cfg(feature = "tokenizer-config")]
  has_chat_template: bool,
  #[cfg(feature = "tokenizer-deepseek-v32")]
  chat_override: Option<Box<dyn ChatTemplateOverride>>,
  #[cfg(feature = "tokenizer-tools")]
  tool_parser: Option<Box<dyn ToolParser>>,
  #[cfg(feature = "tokenizer-tools")]
  tool_call_start: Option<String>,
  #[cfg(feature = "tokenizer-tools")]
  tool_call_end: Option<String>,
  thinking: Thinking,
  #[cfg(feature = "tokenizer-config")]
  bos_token: Option<String>,
  #[cfg(feature = "tokenizer-config")]
  eos_token: Option<String>,
  #[cfg(feature = "tokenizer-config")]
  unk_token: Option<String>,
  #[cfg(feature = "tokenizer-config")]
  pad_token: Option<String>,
}

#[cfg(feature = "tokenizer-config")]
fn cfg_str(cfg: &Value, key: &str) -> Option<String> {
  match cfg.get(key) {
    Some(Value::String(s)) => Some(s.clone()),
    Some(Value::Object(o)) => o.get("content").and_then(Value::as_str).map(str::to_owned),
    _ => None,
  }
}

impl Tokenizer {
  /// Load from a local model directory. Mirrors Python `load`:
  /// reads `tokenizer.json`, then (with `tokenizer-config`)
  /// `tokenizer_config.json` (chat template, tool parser, special tokens).
  ///
  /// `extra_eos_token_ids` augments the eos set (Python `eos_token_ids`
  /// argument).
  pub fn from_path(
    model_path: impl AsRef<Path>,
    extra_eos_token_ids: Option<&[u32]>,
  ) -> Result<Self, Error> {
    let dir = model_path.as_ref();
    let tok_file = dir.join("tokenizer.json");
    let hf = HfTokenizer::from_file(&tok_file)
      .map_err(|e| Error::tokenizer(format!("load tokenizer.json: {e}")))?;

    // Detokenizer class inference reads the raw `decoder` node — that needs
    // `serde_json`, so it only happens with `tokenizer-spm`/`tokenizer-bpe`.
    // Otherwise (incl. bare `tokenizer`/`tokenizer-stream`) the class is the
    // naive re-decode detokenizer (no JSON parse on this path).
    #[cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))]
    let detok_class = {
      let bytes = std::fs::read(&tok_file)
        .map_err(|e| Error::tokenizer(format!("read tokenizer.json: {e}")))?;
      let raw: Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::tokenizer(format!("parse tokenizer.json: {e}")))?;
      infer_detokenizer_class(raw.get("decoder"))
    };
    #[cfg(all(
      feature = "tokenizer-stream",
      not(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))
    ))]
    let detok_class = DetokenizerClass::Naive;

    // tokenizer_config.json (optional; only parsed with `tokenizer-config`).
    #[cfg(feature = "tokenizer-config")]
    let config: Value = {
      let cfg_file = dir.join("tokenizer_config.json");
      if cfg_file.exists() {
        let bytes = std::fs::read(&cfg_file)
          .map_err(|e| Error::tokenizer(format!("read tokenizer_config.json: {e}")))?;
        serde_json::from_slice(&bytes)
          .map_err(|e| Error::tokenizer(format!("parse tokenizer_config.json: {e}")))?
      } else {
        Value::Object(Default::default())
      }
    };

    Self::from_loaded(
      hf,
      #[cfg(feature = "tokenizer-config")]
      config,
      #[cfg(feature = "tokenizer-stream")]
      detok_class,
      extra_eos_token_ids,
    )
  }

  /// Build from an already-loaded `HfTokenizer` (+ parsed config / inferred
  /// detokenizer class when those features are on). Used by `from_path` and
  /// tests.
  pub fn from_loaded(
    hf: HfTokenizer,
    #[cfg(feature = "tokenizer-config")] config: Value,
    #[cfg(feature = "tokenizer-stream")] detok_class: DetokenizerClass,
    extra_eos_token_ids: Option<&[u32]>,
  ) -> Result<Self, Error> {
    #[cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream"))]
    let clean_up_spaces = config
      .get("clean_up_tokenization_spaces")
      .and_then(Value::as_bool)
      .unwrap_or(true);

    #[cfg(feature = "tokenizer-config")]
    let bos_token = cfg_str(&config, "bos_token");
    #[cfg(feature = "tokenizer-config")]
    let eos_token = cfg_str(&config, "eos_token");
    #[cfg(feature = "tokenizer-config")]
    let unk_token = cfg_str(&config, "unk_token");
    #[cfg(feature = "tokenizer-config")]
    let pad_token = cfg_str(&config, "pad_token");

    let mut eos_token_ids = std::collections::BTreeSet::new();
    #[cfg(feature = "tokenizer-config")]
    if let Some(ref e) = eos_token
      && let Some(id) = hf.token_to_id(e)
    {
      eos_token_ids.insert(id);
    }
    if let Some(extra) = extra_eos_token_ids {
      eos_token_ids.extend(extra.iter().copied());
    }

    #[cfg(feature = "tokenizer-config")]
    let chat_template = match config.get("chat_template") {
      Some(Value::String(s)) => Some(s.clone()),
      _ => None,
    };
    #[cfg(feature = "tokenizer-deepseek-v32")]
    let chat_override = config
      .get("chat_template_type")
      .and_then(Value::as_str)
      .and_then(chat::override_by_name);
    #[cfg(all(feature = "tokenizer-config", feature = "tokenizer-deepseek-v32"))]
    let has_chat_template = chat_template.is_some() || chat_override.is_some();
    #[cfg(all(feature = "tokenizer-config", not(feature = "tokenizer-deepseek-v32")))]
    let has_chat_template = chat_template.is_some();

    #[cfg(feature = "tokenizer-tools")]
    let (tool_parser, tool_call_start, tool_call_end) = {
      let parser_name = config
        .get("tool_parser_type")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| tools::infer_tool_parser(chat_template.as_deref()).map(str::to_owned));
      let tool_parser = parser_name.as_deref().and_then(tools::parser_by_name);
      let (s, e) = match &tool_parser {
        Some(p) => (
          Some(p.tool_call_start().to_owned()),
          Some(p.tool_call_end().to_owned()),
        ),
        None => (None, None),
      };
      (tool_parser, s, e)
    };

    let thinking = infer_thinking(&hf);

    Ok(Self {
      hf,
      #[cfg(feature = "tokenizer-config")]
      config,
      #[cfg(feature = "tokenizer-stream")]
      detok_class,
      #[cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream"))]
      clean_up_spaces,
      eos_token_ids,
      #[cfg(feature = "tokenizer-chat")]
      chat_template,
      #[cfg(feature = "tokenizer-config")]
      has_chat_template,
      #[cfg(feature = "tokenizer-deepseek-v32")]
      chat_override,
      #[cfg(feature = "tokenizer-tools")]
      tool_parser,
      #[cfg(feature = "tokenizer-tools")]
      tool_call_start,
      #[cfg(feature = "tokenizer-tools")]
      tool_call_end,
      thinking,
      #[cfg(feature = "tokenizer-config")]
      bos_token,
      #[cfg(feature = "tokenizer-config")]
      eos_token,
      #[cfg(feature = "tokenizer-config")]
      unk_token,
      #[cfg(feature = "tokenizer-config")]
      pad_token,
    })
  }

  /// Build from already-parsed parts (legacy signature, kept API-stable for
  /// the `lm` configuration). The `_raw` value is the parsed `tokenizer.json`
  /// (only `decoder` matters, already folded into `detok_class`); `config` is
  /// the parsed `tokenizer_config.json`.
  #[cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream"))]
  #[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "tokenizer-config", feature = "tokenizer-stream")))
  )]
  pub fn from_parts(
    hf: HfTokenizer,
    _raw: Value,
    config: Value,
    detok_class: DetokenizerClass,
    extra_eos_token_ids: Option<&[u32]>,
  ) -> Result<Self, Error> {
    Self::from_loaded(hf, config, detok_class, extra_eos_token_ids)
  }

  // --- encode / decode (Swift `Tokenizer` protocol) ----------------------

  /// Encode text to token ids. `add_special_tokens` mirrors the Swift /
  /// transformers flag.
  pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
    let enc = self
      .hf
      .encode(text, add_special_tokens)
      .map_err(|e| Error::tokenizer(format!("encode: {e}")))?;
    Ok(enc.get_ids().to_vec())
  }

  /// Encode a batch of texts.
  pub fn encode_batch(
    &self,
    texts: Vec<String>,
    add_special_tokens: bool,
  ) -> Result<Vec<Vec<u32>>, Error> {
    let encs = self
      .hf
      .encode_batch(texts, add_special_tokens)
      .map_err(|e| Error::tokenizer(format!("encode_batch: {e}")))?;
    Ok(encs.iter().map(|e| e.get_ids().to_vec()).collect())
  }

  /// Decode token ids to text. `skip_special_tokens` mirrors the Swift /
  /// transformers flag.
  pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
    self
      .hf
      .decode(ids, skip_special_tokens)
      .map_err(|e| Error::tokenizer(format!("decode: {e}")))
  }

  /// Decode a batch of id sequences.
  pub fn decode_batch(
    &self,
    sequences: &[&[u32]],
    skip_special_tokens: bool,
  ) -> Result<Vec<String>, Error> {
    self
      .hf
      .decode_batch(sequences, skip_special_tokens)
      .map_err(|e| Error::tokenizer(format!("decode_batch: {e}")))
  }

  /// `convert_token_to_id` (Swift `convertTokenToId`).
  pub fn convert_token_to_id(&self, token: &str) -> Option<u32> {
    self.hf.token_to_id(token)
  }

  /// `convert_id_to_token` (Swift `convertIdToToken`).
  pub fn convert_id_to_token(&self, id: u32) -> Option<String> {
    self.hf.id_to_token(id)
  }

  // --- special-token property set (config-derived) -----------------------

  /// `bos_token` (from `tokenizer_config.json`).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn bos_token(&self) -> Option<&str> {
    self.bos_token.as_deref()
  }
  /// `eos_token` (from `tokenizer_config.json`).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn eos_token(&self) -> Option<&str> {
    self.eos_token.as_deref()
  }
  /// `unk_token` (from `tokenizer_config.json`).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn unk_token(&self) -> Option<&str> {
    self.unk_token.as_deref()
  }
  /// `pad_token` (from `tokenizer_config.json`).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn pad_token(&self) -> Option<&str> {
    self.pad_token.as_deref()
  }
  /// `bos_token_id`.
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn bos_token_id(&self) -> Option<u32> {
    self
      .bos_token
      .as_deref()
      .and_then(|t| self.hf.token_to_id(t))
  }
  /// `eos_token_id` (primary).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn eos_token_id(&self) -> Option<u32> {
    self
      .eos_token
      .as_deref()
      .and_then(|t| self.hf.token_to_id(t))
  }
  /// `unk_token_id`.
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn unk_token_id(&self) -> Option<u32> {
    self
      .unk_token
      .as_deref()
      .and_then(|t| self.hf.token_to_id(t))
  }
  /// `pad_token_id`.
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn pad_token_id(&self) -> Option<u32> {
    self
      .pad_token
      .as_deref()
      .and_then(|t| self.hf.token_to_id(t))
  }
  /// The full eos-token-id set (Python `eos_token_ids`).
  pub fn eos_token_ids(&self) -> &std::collections::BTreeSet<u32> {
    &self.eos_token_ids
  }
  /// Add an eos token by string or numeric-string id (Python `add_eos_token`).
  pub fn add_eos_token(&mut self, token: &str) -> Result<(), Error> {
    let id = match token.parse::<u32>() {
      Ok(i) => Some(i),
      Err(_) => self.hf.token_to_id(token),
    };
    let id = id.ok_or_else(|| Error::tokenizer(format!("'{token}' is not a token")))?;
    self.eos_token_ids.insert(id);
    Ok(())
  }
  /// Whether a chat template (jinja or override) is available.
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn has_chat_template(&self) -> bool {
    self.has_chat_template
  }
  /// `tool_call_start` delimiter, if a tool parser was selected.
  #[cfg(feature = "tokenizer-tools")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
  pub fn tool_call_start(&self) -> Option<&str> {
    self.tool_call_start.as_deref()
  }
  /// `tool_call_end` delimiter, if a tool parser was selected.
  #[cfg(feature = "tokenizer-tools")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
  pub fn tool_call_end(&self) -> Option<&str> {
    self.tool_call_end.as_deref()
  }
  /// Whether tool calling is configured.
  #[cfg(feature = "tokenizer-tools")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
  pub fn has_tool_calling(&self) -> bool {
    self.tool_parser.is_some()
  }
  /// The selected tool parser, if any.
  #[cfg(feature = "tokenizer-tools")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
  pub fn tool_parser(&self) -> Option<&dyn ToolParser> {
    self.tool_parser.as_deref()
  }
  /// Parse an assistant tool-call payload with the selected parser.
  #[cfg(feature = "tokenizer-tools")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-tools")))]
  pub fn parse_tool_call(
    &self,
    text: &str,
    tools: Option<&Value>,
  ) -> Result<Vec<tools::ToolCall>, Error> {
    let p = self
      .tool_parser
      .as_ref()
      .ok_or_else(|| Error::tokenizer("no tool parser configured"))?;
    p.parse(text, tools)
  }

  // --- thinking (Python `_infer_thinking` + accessors) -------------------

  /// Whether the model exposes a thinking mode.
  pub fn has_thinking(&self) -> bool {
    self.thinking.start.is_some()
  }
  /// Thinking start marker string.
  pub fn think_start(&self) -> Option<&str> {
    self.thinking.start.as_deref()
  }
  /// Thinking end marker string.
  pub fn think_end(&self) -> Option<&str> {
    self.thinking.end.as_deref()
  }
  /// Thinking start token ids.
  pub fn think_start_tokens(&self) -> Option<&[u32]> {
    self.thinking.start_tokens.as_deref()
  }
  /// Thinking end token ids.
  pub fn think_end_tokens(&self) -> Option<&[u32]> {
    self.thinking.end_tokens.as_deref()
  }

  // --- detokenizer factory (Python `detokenizer` property) ---------------

  /// Build a fresh streaming detokenizer of the inferred class.
  ///
  /// **Graceful fallback:** if the model's decoder selects the SPM or BPE
  /// detokenizer but `tokenizer-spm` / `tokenizer-bpe` is disabled, this
  /// falls back to the naive re-decode detokenizer and emits a one-time
  /// warning. It never panics or hard-errors.
  #[cfg(feature = "tokenizer-stream")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
  pub fn detokenizer(&self) -> BoxedDetokenizer {
    #[cfg(feature = "tokenizer-config")]
    let clean = self.clean_up_spaces;
    // Without `tokenizer-config` there is no `clean_up_tokenization_spaces`
    // signal; mirror mlx-swift-lm (never strip trailing spaces).
    #[cfg(not(feature = "tokenizer-config"))]
    let clean = false;

    match self.detok_class {
      #[cfg(feature = "tokenizer-spm")]
      DetokenizerClass::Spm | DetokenizerClass::SpmNoSpace => {
        let vocab = self.hf.get_vocab(true);
        let trim = self.detok_class == DetokenizerClass::Spm;
        Box::new(SpmStreamingDetokenizer::new(vocab, trim))
      }
      #[cfg(feature = "tokenizer-bpe")]
      DetokenizerClass::Bpe => {
        let vocab = self.hf.get_vocab(true);
        Box::new(BpeStreamingDetokenizer::new(vocab, clean))
      }
      #[cfg(not(feature = "tokenizer-spm"))]
      DetokenizerClass::Spm | DetokenizerClass::SpmNoSpace => {
        warn_detok_fallback("spm");
        self.naive_detokenizer(clean)
      }
      #[cfg(not(feature = "tokenizer-bpe"))]
      DetokenizerClass::Bpe => {
        warn_detok_fallback("bpe");
        self.naive_detokenizer(clean)
      }
      DetokenizerClass::Naive => self.naive_detokenizer(clean),
    }
  }

  /// The naive re-decode detokenizer over a cloned HF tokenizer.
  #[cfg(feature = "tokenizer-stream")]
  fn naive_detokenizer(&self, clean: bool) -> BoxedDetokenizer {
    let hf = self.hf.clone();
    let decode = move |ids: &[u32]| hf.decode(ids, false).unwrap_or_default();
    Box::new(NaiveStreamingDetokenizer::new(decode, clean))
  }

  /// The inferred detokenizer class.
  #[cfg(feature = "tokenizer-stream")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
  pub fn detokenizer_class(&self) -> DetokenizerClass {
    self.detok_class
  }

  // --- chat template (Python `apply_chat_template`) ----------------------

  /// Render the chat template to a prompt string. A registered override
  /// (e.g. `deepseek_v32`) takes precedence over the jinja `chat_template`,
  /// mirroring Python `TokenizerWrapper.apply_chat_template`.
  ///
  /// `messages` / `tools` are JSON values; `additional_context` adds extra
  /// template variables.
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template(
    &self,
    messages: &Value,
    tools: Option<&Value>,
    add_generation_prompt: bool,
    additional_context: Option<&Value>,
  ) -> Result<String, Error> {
    let enable_thinking = additional_context
      .and_then(|c| c.get("enable_thinking"))
      .and_then(Value::as_bool)
      .unwrap_or(self.has_thinking());

    #[cfg(feature = "tokenizer-deepseek-v32")]
    if let Some(ovr) = &self.chat_override {
      let msgs = messages
        .as_array()
        .cloned()
        .ok_or_else(|| Error::tokenizer("messages must be a list"))?;
      let continue_final = additional_context
        .and_then(|c| c.get("continue_final_message"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
      return ovr.apply(
        &msgs,
        tools,
        add_generation_prompt,
        continue_final,
        enable_thinking,
      );
    }

    let template = self
      .chat_template
      .as_deref()
      .ok_or_else(|| Error::tokenizer("this tokenizer does not have a chat template"))?;
    let extra = additional_context.cloned().unwrap_or(Value::Null);
    chat::render_jinja(
      template,
      messages,
      tools,
      add_generation_prompt,
      self.bos_token.as_deref(),
      self.eos_token.as_deref(),
      enable_thinking,
      &extra,
    )
  }

  /// Render the chat template and tokenize the result (Python
  /// `apply_chat_template(tokenize=True)`).
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template_ids(
    &self,
    messages: &Value,
    tools: Option<&Value>,
    add_generation_prompt: bool,
    additional_context: Option<&Value>,
  ) -> Result<Vec<u32>, Error> {
    let text =
      self.apply_chat_template(messages, tools, add_generation_prompt, additional_context)?;
    self.encode(&text, false)
  }

  /// Access the underlying parsed `tokenizer_config.json`.
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn config(&self) -> &Value {
    &self.config
  }

  /// Access the underlying HF tokenizer.
  pub fn hf(&self) -> &HfTokenizer {
    &self.hf
  }
}

/// One-time `eprintln!` warning when the model wants a precise streaming
/// detokenizer whose feature is disabled (we fall back to naive). Never
/// panics; emits at most once per `kind` per process. Only compiled when a
/// fallback arm is actually reachable (i.e. `tokenizer-spm` and/or
/// `tokenizer-bpe` is off while `tokenizer-stream` is on).
#[cfg(all(
  feature = "tokenizer-stream",
  not(all(feature = "tokenizer-spm", feature = "tokenizer-bpe"))
))]
fn warn_detok_fallback(kind: &'static str) {
  use std::sync::Once;
  static SPM_ONCE: Once = Once::new();
  static BPE_ONCE: Once = Once::new();
  let once = if kind == "spm" { &SPM_ONCE } else { &BPE_ONCE };
  once.call_once(|| {
    eprintln!(
      "mlxrs: model wants the {kind} streaming detokenizer but the \
       `tokenizer-{kind}` feature is disabled; falling back to naive \
       (less precise streaming)"
    );
  });
}

/// Port of Python `_infer_thinking`.
fn infer_thinking(hf: &HfTokenizer) -> Thinking {
  let vocab = hf.get_vocab(true);
  let pairs = [
    ("<think>", "</think>"),
    ("<longcat_think>", "</longcat_think>"),
  ];
  for (ts, te) in pairs {
    if let (Some(&sid), Some(&eid)) = (vocab.get(ts), vocab.get(te)) {
      return Thinking {
        start: Some(ts.to_owned()),
        end: Some(te.to_owned()),
        start_tokens: Some(vec![sid]),
        end_tokens: Some(vec![eid]),
      };
    }
  }
  if vocab.contains_key("<|channel>") && vocab.contains_key("<channel|>") {
    let ts = "<|channel>thought";
    let te = "<channel|>";
    let st = hf
      .encode(ts, false)
      .map(|e| e.get_ids().to_vec())
      .unwrap_or_default();
    let et = hf
      .encode(te, false)
      .map(|e| e.get_ids().to_vec())
      .unwrap_or_default();
    return Thinking {
      start: Some(ts.to_owned()),
      end: Some(te.to_owned()),
      start_tokens: Some(st),
      end_tokens: Some(et),
    };
  }
  Thinking::default()
}

/// No-bos-or-eos helper (Python `no_bos_or_eos`).
pub fn no_bos_or_eos(sequence: &[u32], bos: u32, eos: u32) -> Vec<u32> {
  let start = if sequence.first() == Some(&bos) { 1 } else { 0 };
  let mut s = sequence[start..].to_vec();
  if s.last() == Some(&eos) {
    s.pop();
  }
  s
}
