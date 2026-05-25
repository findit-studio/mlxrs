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

use super::encode_options::{EncodeOptions, Encoded};
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
use super::stream::{Detokenizer, DetokenizerClass, NaiveHfDetokenizer};
#[cfg(feature = "tokenizer-tools")]
use super::tools::{self, ToolParser};

/// The `detokenizer()` factory return — the enum-unified
/// [`Detokenizer`] (P1 #111).
///
/// # Breaking change (P1 #111)
///
/// Previously `pub type BoxedDetokenizer = Box<dyn StreamingDetokenizer>`
/// — one vtable indirection per emitted token. This is now an alias for
/// the unified [`Detokenizer`] enum (`Naive` / `Spm` / `Bpe` /
/// `Custom`), dispatching the per-token `add_token` / `text` /
/// `last_segment` calls via `match` so the canonical variants inline.
/// Callers passing `Box<dyn StreamingDetokenizer>` directly must wrap
/// in [`Detokenizer::Custom`] (one indirection per call — the same
/// cost as the prior alias).
#[cfg(feature = "tokenizer-stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
pub type BoxedDetokenizer = Detokenizer;

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
  /// The PRIMARY EOS id — the one to APPEND when a caller asks for one
  /// EOS (rather than the full stop-id SET). For caller-supplied
  /// `eos_token_ids`, this is the first slice element (preserving input
  /// order, which `BTreeSet` would have sorted away); for the
  /// `tokenizer-config` fallback this is the `eos_token` resolved to its
  /// id. `None` when there is no configured primary EOS, including when
  /// both sources are absent or the caller explicitly supplies an empty
  /// `eos_token_ids` slice (which suppresses the fallback and leaves the
  /// set empty; used to error on `encode_with(add_eos=true)`).
  primary_eos: Option<u32>,
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
  /// `eos_token_ids` is the **complete** eos set, mirroring Python
  /// `TokenizerWrapper`: `set(eos_token_ids) if eos_token_ids is not None
  /// else {tokenizer.eos_token_id}`. `Some(ids)` REPLACES the
  /// tokenizer-config default with exactly `ids`; `None` falls back to the
  /// tokenizer's own `eos_token`. (The loader resolves the precedence
  /// generation_config-truthy → config.json → `None`.)
  pub fn from_path(
    model_path: impl AsRef<Path>,
    eos_token_ids: Option<&[u32]>,
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
      eos_token_ids,
    )
  }

  /// Build from an already-loaded `HfTokenizer` (+ parsed config / inferred
  /// detokenizer class when those features are on). Used by `from_path` and
  /// tests.
  pub fn from_loaded(
    hf: HfTokenizer,
    #[cfg(feature = "tokenizer-config")] config: Value,
    #[cfg(feature = "tokenizer-stream")] detok_class: DetokenizerClass,
    eos_token_ids: Option<&[u32]>,
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

    // Python `TokenizerWrapper`: `self._eos_token_ids = set(eos_token_ids)
    // if eos_token_ids is not None else {tokenizer.eos_token_id}`. A
    // supplied set REPLACES the tokenizer-config default entirely (it is
    // NOT unioned); `None` falls back to the tokenizer's own `eos_token`.
    // (`if let` rather than `match` — the `None` arm is empty without the
    // `tokenizer-config` feature, which would trip `clippy::single_match`.)
    let mut eos_set = std::collections::BTreeSet::new();
    // Track the PRIMARY eos id (the one to APPEND for `add_eos`) separately
    // from the full stop set, since `BTreeSet::iter().next()` returns the
    // numerically smallest entry — wrong when a multi-id stop list contains
    // a non-EOS pad/unk with a smaller id than the actual EOS.
    let mut primary_eos: Option<u32> = None;
    if let Some(ids) = eos_token_ids {
      if let Some(&first) = ids.first() {
        primary_eos = Some(first);
      }
      eos_set.extend(ids.iter().copied());
    }
    #[cfg(feature = "tokenizer-config")]
    if eos_token_ids.is_none()
      && let Some(ref e) = eos_token
      && let Some(id) = hf.token_to_id(e)
    {
      primary_eos = Some(id);
      eos_set.insert(id);
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
      eos_token_ids: eos_set,
      primary_eos,
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
    eos_token_ids: Option<&[u32]>,
  ) -> Result<Self, Error> {
    Self::from_loaded(hf, config, detok_class, eos_token_ids)
  }

  // --- encode / decode (Swift `Tokenizer` protocol) ----------------------

  /// Encode text to token ids. `add_special_tokens` mirrors the Swift /
  /// transformers flag.
  ///
  /// This is the short positional form preserved for back-compat: it
  /// returns the **raw** HF `Encoding` ids verbatim (including any
  /// HF-applied padding cells when the tokenizer has padding enabled).
  /// For explicit control over EOS appending, truncation, attention-mask
  /// emission — and for pad-stripping — use
  /// [`Tokenizer::encode_with`] with an [`EncodeOptions`] builder.
  pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, Error> {
    let enc = self
      .hf
      .encode(text, add_special_tokens)
      .map_err(|e| Error::tokenizer(format!("encode: {e}")))?;
    Ok(enc.get_ids().to_vec())
  }

  /// Encode `text` with explicit options.
  ///
  /// Exposes the richer surface of the underlying HF `tokenizers` crate that
  /// the short [`encode`](Self::encode) hides — explicit EOS appending,
  /// truncation, attention-mask emission, and pad-stripping. Mask contract on
  /// the returned [`Encoded::attention_mask`]:
  /// - if [`EncodeOptions::return_attention_mask`] was `false`: empty slice
  ///   (no allocation);
  /// - if `true`: `mask.len() == ids.len()`, **including the legitimate
  ///   `(0, 0)` zero-length encoding** (e.g.
  ///   [`EncodeOptions::with_truncate_to`]`(Some(0))` or empty `text`).
  ///
  /// Presence of the mask is therefore a property of the caller's
  /// [`EncodeOptions`], not of the result — do not test
  /// `attention_mask().is_empty()` as a "not requested" sentinel.
  ///
  /// **Padding stripping.** If the HF tokenizer has padding enabled,
  /// `encode_with` drops all `mask == 0` cells regardless of position
  /// (right-pad, left-pad, or sparse). The returned `ids` and
  /// `attention_mask` describe only the real attended tokens — every
  /// cell of the returned mask is `1`. This diverges from the legacy
  /// [`encode`](Self::encode), which preserves HF's raw padded layout.
  ///
  /// **EOS placement.** `add_eos: true` appends the **primary EOS** id
  /// **after** the real attended tokens. The primary EOS is the first
  /// caller-supplied EOS id (else the `tokenizer-config` `eos_token`)
  /// tracked at load — NOT `eos_token_ids.iter().next()`, which would be
  /// the numerically smallest id in the sorted eos-id set
  /// (possibly a non-EOS stop token). If no primary EOS is
  /// configured it returns an error rather than silently no-op-ing; the
  /// precondition is validated **before** the underlying `hf.encode`
  /// call so a configuration gap fails fast.
  ///
  /// **EOS + truncation interaction.** When `add_eos` is combined with
  /// `truncate_to(n)` for `n >= 1`, the EOS is **guaranteed to be the
  /// last id** in the returned vector: the head is sliced to `n - 1`
  /// attended ids and the EOS is appended. This matches the typical
  /// LM-training expectation that "truncate-to-N with EOS" still ends in
  /// EOS. The `n == 0` edge case is the sole exception — the output must
  /// be empty, so no EOS is appended (an empty cap dominates `add_eos`).
  ///
  /// **Truncation.** `truncate_to: Some(n)` caps the **returned** vectors
  /// with a bounded slice; HF `Encoding::truncate` is intentionally not
  /// used because (as of `tokenizers` 0.23) it retains the discarded tail
  /// in `Encoding::overflowing` and would defeat the cap on long inputs.
  /// Note that `truncate_to` caps the **output** length only — the
  /// underlying HF `tokenizer.encode` is still called on the full input,
  /// so it is not an input-allocation bound. Callers needing an input
  /// cap should pre-trim `text` themselves.
  pub fn encode_with(&self, text: &str, opts: &EncodeOptions) -> Result<Encoded, Error> {
    // Resolve the eos id BEFORE calling `hf.encode`: if the caller asked
    // for `add_eos` but no primary eos was configured, fail fast on the
    // configuration error rather than spending tokenizer cost on a doomed
    // call. Uses `self.primary_eos` (the first user-supplied id, or the
    // `tokenizer-config` `eos_token`), NOT `eos_token_ids.iter().next()` —
    // the latter returns the numerically smallest entry in the sorted
    // set, which can be a non-EOS pad/unk in a multi-id stop list.
    let eos = Self::resolve_eos(opts.add_eos(), self.primary_eos)?;

    let enc = self
      .hf
      .encode(text, opts.add_special())
      .map_err(|e| Error::tokenizer(format!("hf.encode: {e}")))?;

    finalize_encoding(&enc, opts, eos)
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

  /// Encode a batch of texts with explicit options (LM-2).
  ///
  /// Batch analogue of [`Self::encode_with`]: applies the SAME
  /// [`EncodeOptions`] (add_special, add_eos, truncate_to,
  /// return_attention_mask) to every input. Returns one [`Encoded`] per
  /// input, in the same order. The per-item post-processing — pad
  /// stripping, EOS append, head-truncation, optional all-1s mask — is
  /// byte-for-byte the same as [`Self::encode_with`] applied
  /// independently to each text, so callers can switch from a hand-rolled
  /// `for text in texts { tok.encode_with(text, opts) }` loop without
  /// observable result change while letting HF's `encode_batch` exploit
  /// its internal parallelism.
  ///
  /// **EOS pre-validation.** When `opts.add_eos` is `true`, the primary
  /// EOS is resolved BEFORE the HF `encode_batch` call — a missing
  /// primary EOS fails fast and skips the entire (potentially
  /// large-batch) tokenizer pass, mirroring [`Self::encode_with`]'s
  /// fast-fail.
  pub fn encode_batch_with(
    &self,
    texts: Vec<String>,
    opts: &EncodeOptions,
  ) -> Result<Vec<Encoded>, Error> {
    // Same fast-fail-on-missing-eos contract as `encode_with` — resolve
    // BEFORE the (potentially expensive) batch tokenizer call.
    let eos = Self::resolve_eos(opts.add_eos(), self.primary_eos)?;

    let encs = self
      .hf
      .encode_batch(texts, opts.add_special())
      .map_err(|e| Error::tokenizer(format!("hf.encode_batch: {e}")))?;

    let mut out = Vec::with_capacity(encs.len());
    for enc in &encs {
      out.push(finalize_encoding(enc, opts, eos)?);
    }
    Ok(out)
  }

  /// Resolve the primary-EOS id once for the (batch-shared) `add_eos`
  /// precondition. Extracted so [`Self::encode_with`] and
  /// [`Self::encode_batch_with`] share a single fast-fail path.
  fn resolve_eos(add_eos: bool, primary_eos: Option<u32>) -> Result<Option<u32>, Error> {
    if add_eos {
      Ok(Some(primary_eos.ok_or_else(|| {
        Error::tokenizer("encode_with(add_eos=true) requires a configured eos token id")
      })?))
    } else {
      Ok(None)
    }
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
  /// `additional_special_tokens` resolved to ids (from
  /// `tokenizer_config.json`). Mirrors the HF
  /// `PreTrainedTokenizerBase.additional_special_tokens_ids` accessor.
  ///
  /// Each entry in the `additional_special_tokens` array may be either a
  /// plain string (`"<extra>"`) or an `AddedToken`-style object
  /// (`{"content": "<extra>", ...}`) — the same two shapes the private
  /// `cfg_str` helper handles for the singular `bos_token`/`eos_token`/
  /// etc. fields. An entry that does not resolve to a known vocab id is
  /// silently skipped (matching HF behavior — the underlying
  /// `convert_tokens_to_ids` returns `None`/`unk_token_id` for unknown
  /// entries, but the GGUF-export caller only needs the IDs that exist
  /// in the vocab to flag them as `Control` tokens).
  #[cfg(feature = "tokenizer-config")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-config")))]
  pub fn additional_special_token_ids(&self) -> Vec<u32> {
    let Some(arr) = self.config.get("additional_special_tokens") else {
      return Vec::new();
    };
    let Some(items) = arr.as_array() else {
      return Vec::new();
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
      let token: Option<&str> = match item {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("content").and_then(Value::as_str),
        _ => None,
      };
      if let Some(tok) = token
        && let Some(id) = self.hf.token_to_id(tok)
      {
        out.push(id);
      }
    }
    out
  }
  /// Iterate over all eos-token ids (Python `eos_token_ids`).
  ///
  /// Returns a `Copy`-element iterator over the sorted set; callers that
  /// need a `Vec<u32>` can do `.eos_token_ids_iter().collect()`.
  pub fn eos_token_ids_iter(&self) -> impl Iterator<Item = u32> + '_ {
    self.eos_token_ids.iter().copied()
  }

  /// Returns `true` if `id` is in the eos-token-id set.
  pub fn contains_eos_id(&self, id: u32) -> bool {
    self.eos_token_ids.contains(&id)
  }
  /// Add an eos token by string or numeric-string id (Python `add_eos_token`).
  /// If no primary EOS was established at construction time (no
  /// `tokenizer-config` eos and no caller-supplied set), the first id added
  /// via this method becomes the primary used by [`Self::encode_with`]
  /// when `add_eos = true`.
  pub fn add_eos_token(&mut self, token: &str) -> Result<(), Error> {
    let id = match token.parse::<u32>() {
      Ok(i) => Some(i),
      Err(_) => self.hf.token_to_id(token),
    };
    let id = id.ok_or_else(|| Error::tokenizer(format!("'{token}' is not a token")))?;
    self.eos_token_ids.insert(id);
    if self.primary_eos.is_none() {
      self.primary_eos = Some(id);
    }
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
        Detokenizer::Spm(SpmStreamingDetokenizer::new(vocab, trim))
      }
      #[cfg(feature = "tokenizer-bpe")]
      DetokenizerClass::Bpe => {
        let vocab = self.hf.get_vocab(true);
        Detokenizer::Bpe(BpeStreamingDetokenizer::new(vocab, clean))
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

  /// The naive re-decode detokenizer over a cloned HF tokenizer —
  /// returns the [`Detokenizer::Naive`] variant (the non-generic
  /// concrete [`NaiveHfDetokenizer`] so the enum unification holds).
  #[cfg(feature = "tokenizer-stream")]
  fn naive_detokenizer(&self, clean: bool) -> BoxedDetokenizer {
    Detokenizer::Naive(Box::new(NaiveHfDetokenizer::new(self.hf.clone(), clean)))
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
  ///
  /// `continue_final_message` ports HF Transformers' flag of the same name:
  /// when `true`, the rendered prompt is trimmed so it ends exactly at the
  /// final message's content — the model *continues* that message instead of
  /// starting a fresh turn (HF strips the trailing end-of-turn / EOS tokens
  /// the template appended after it; see [`chat::render_jinja`]). It is
  /// **mutually exclusive** with `add_generation_prompt`: HF raises a
  /// `ValueError` if both are set, and this method returns an `Err` likewise.
  /// Existing callers that do not continue the final message pass `false`
  /// (unchanged behavior).
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template(
    &self,
    messages: &Value,
    tools: Option<&Value>,
    add_generation_prompt: bool,
    continue_final_message: bool,
    additional_context: Option<&Value>,
  ) -> Result<String, Error> {
    // HF rejects `add_generation_prompt` + `continue_final_message` together
    // (`apply_chat_template`: "continue_final_message is not compatible with
    // add_generation_prompt"). Reject up front, before any rendering, so both
    // the jinja and override paths share the guard.
    if add_generation_prompt && continue_final_message {
      return Err(Error::tokenizer(
        "continue_final_message is not compatible with add_generation_prompt \
         (only one may be set)",
      ));
    }

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
      return ovr.apply(
        &msgs,
        tools,
        add_generation_prompt,
        continue_final_message,
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
      continue_final_message,
      self.bos_token.as_deref(),
      self.eos_token.as_deref(),
      enable_thinking,
      &extra,
    )
  }

  /// Render the chat template and tokenize the result (Python
  /// `apply_chat_template(tokenize=True)`).
  ///
  /// `continue_final_message` is forwarded to [`Self::apply_chat_template`] —
  /// see that method for the semantics (and the mutual exclusivity with
  /// `add_generation_prompt`).
  #[cfg(feature = "tokenizer-chat")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-chat")))]
  pub fn apply_chat_template_ids(
    &self,
    messages: &Value,
    tools: Option<&Value>,
    add_generation_prompt: bool,
    continue_final_message: bool,
    additional_context: Option<&Value>,
  ) -> Result<Vec<u32>, Error> {
    let text = self.apply_chat_template(
      messages,
      tools,
      add_generation_prompt,
      continue_final_message,
      additional_context,
    )?;
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

/// Shared per-`Encoding` post-processor for
/// [`Tokenizer::encode_with`] and [`Tokenizer::encode_batch_with`] (LM-2).
///
/// Applies (in order):
///   1. shape-skew guard on `ids.len() == attention_mask.len()`;
///   2. `mask == 0` cell drop (HF pad cells, left/right/sparse all dropped);
///   3. head-truncation to `truncate_to` ids (HF
///      `TruncationDirection::Right` semantics — keep the head); with
///      `add_eos`, the head is sliced to `n - 1` so the appended EOS is
///      guaranteed to be the last id (the `n == 0` edge case is the doc'd
///      exception — an empty cap dominates `add_eos`, no EOS appended);
///   4. constant-1 attention mask synthesis when
///      `return_attention_mask` is set (every returned cell is real /
///      attended, including the appended EOS).
fn finalize_encoding(
  enc: &tokenizers::Encoding,
  opts: &EncodeOptions,
  eos: Option<u32>,
) -> Result<Encoded, Error> {
  let hf_ids = enc.get_ids();
  let hf_mask = enc.get_attention_mask();
  // HF contract: `attention_mask.len() == ids.len()`. Surface any future
  // shape skew as a clean error rather than panicking on indexed access.
  if hf_ids.len() != hf_mask.len() {
    return Err(Error::tokenizer(format!(
      "HF Encoding shape mismatch: ids.len()={} attention_mask.len()={}",
      hf_ids.len(),
      hf_mask.len(),
    )));
  }

  // Real attended length = count of all `mask != 0` cells, regardless
  // of where they sit. This keeps left-padded (`[0,0,1,1]`) and any
  // sparse-zero masks correct: every attended cell becomes a real
  // token, every pad cell is dropped (not just the trailing ones).
  let real_len: usize = hf_mask.iter().filter(|&&m| m != 0).count();

  // Bounded allocation: one Vec sized to the FINAL output length.
  let extra = usize::from(eos.is_some());
  let pre_trunc_len = real_len + extra;
  let final_len = opts
    .truncate_to()
    .map_or(pre_trunc_len, |n| n.min(pre_trunc_len));

  let mut ids: Vec<u32> = Vec::with_capacity(final_len);
  // Copy at most `head_cap` attended ids, in HF order, leaving room
  // for the eos slot when it survives truncation.
  let head_cap = final_len.saturating_sub(extra).min(real_len);
  if head_cap > 0 {
    let mut emitted = 0usize;
    for (&id, &m) in hf_ids.iter().zip(hf_mask.iter()) {
      if m == 0 {
        continue;
      }
      ids.push(id);
      emitted += 1;
      if emitted == head_cap {
        break;
      }
    }
  }
  if let Some(e) = eos
    && ids.len() < final_len
  {
    // Append eos if it still fits after truncation.
    ids.push(e);
  }

  // Mask is constant-1 for the returned cells (all attended, including
  // the appended eos). Single bounded allocation matching `ids.len()`.
  // §1 EMPTY MEANS ABSENT: empty Vec when mask was not requested.
  let attention_mask = if opts.return_attention_mask() {
    vec![1u8; ids.len()]
  } else {
    Vec::new()
  };

  Ok(Encoded::new(ids, attention_mask))
}
