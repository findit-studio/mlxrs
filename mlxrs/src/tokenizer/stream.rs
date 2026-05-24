//! Streaming detokenizers.
//!
//! Ported from `mlx-lm`'s `mlx_lm/tokenizer_utils.py` lines 11-255
//! (`StreamingDetokenizer`, `NaiveStreamingDetokenizer`,
//! `SPMStreamingDetokenizer`, `BPEStreamingDetokenizer`) and cross-referenced
//! against `mlx-swift-lm`'s `MLXLMCommon/Tokenizer.swift`
//! `StreamingDetokenizer` protocol + `NaiveStreamingDetokenizer`.
//!
//! A streaming detokenizer turns a stream of token ids into text one token at
//! a time. After `reset`, callers `add_token` per generated id and read
//! `text()` (whole text so far) / `last_segment()` (newly printable text since
//! the last read) / `tokens()`. `finalize()` flushes any buffered bytes so
//! `text()` matches a full `decode(tokens())`.

/// The streaming detokenizer interface (Python `StreamingDetokenizer`,
/// Swift `StreamingDetokenizer`).
///
/// Mirrors the Python contract: `reset` → repeated `add_token` → `finalize`,
/// with `text` / `last_segment` / `tokens` observers.
pub trait StreamingDetokenizer {
  /// Reset all streaming state to empty.
  fn reset(&mut self);

  /// Feed one token id into the stream.
  fn add_token(&mut self, token: u32);

  /// Flush any buffered (incomplete) bytes into `text`.
  fn finalize(&mut self);

  /// The whole decoded text so far, including any not-yet-committed
  /// in-progress segment (Python `text` property = `_text + _current_text`).
  /// Some trailing tokens may still be withheld until a word/line boundary,
  /// exactly as in the Python implementation. Returns
  /// [`Cow::Borrowed`](std::borrow::Cow::Borrowed) when the implementation
  /// already accumulates the full text (SPM/BPE: zero alloc) and
  /// [`Cow::Owned`](std::borrow::Cow::Owned) only when an in-progress segment
  /// must be stitched on (Naive).
  fn text(&self) -> std::borrow::Cow<'_, str>;

  /// All token ids added since the last `reset`.
  fn tokens(&self) -> &[u32];

  /// The current readable-text offset (Python `self.offset`). Used by
  /// [`StreamingDetokenizer::last_segment`].
  fn offset(&self) -> usize;

  /// Set the readable-text offset.
  fn set_offset(&mut self, offset: usize);

  /// Return the last segment of readable text since this was last called,
  /// advancing the offset. Mirrors the Python `last_segment` property.
  fn last_segment(&mut self) -> String {
    // `text()` borrows `self.text` for SPM/BPE (`Cow::Borrowed`, zero alloc)
    // and is already `Cow::Owned` for Naive. Binding it to a local keeps the
    // borrowed/owned backing alive for the whole slice + `to_owned` below,
    // so we allocate ONLY the returned segment — never a copy of the full
    // accumulated output buffer (the old `into_owned()` was O(total_output)
    // per call ⇒ O(total_output²) over a generation).
    let text = self.text();
    let s: &str = text.as_ref();
    let len = s.len();
    let off = self.offset().min(len);
    // Advance to a char boundary so slicing never panics on multi-byte text.
    let mut start = off;
    while start < len && !s.is_char_boundary(start) {
      start += 1;
    }
    // `end` is the full text length: `start` (already char-boundary-clamped
    // and `<= len`) can never exceed it, so `len.max(start) == len`, matching
    // the prior `end.max(start)` semantics exactly.
    let end = len;
    let seg = s[start..].to_owned();
    // End the borrow of `self` before the `&mut self` call.
    drop(text);
    self.set_offset(end);
    seg
  }
}

/// `NaiveStreamingDetokenizer` — relies on a full `decode` callback and works
/// with every tokenizer. O(T^2) over the longest line, matching Python.
///
/// The decode callback decodes a token-id slice into a string and reports the
/// tokenizer's `clean_up_tokenization_spaces` flag (the Python field of the
/// same name).
pub struct NaiveStreamingDetokenizer<F>
where
  F: Fn(&[u32]) -> String,
{
  decode: F,
  clean_up_spaces: bool,
  tokens: Vec<u32>,
  offset: usize,
  text: String,
  current_tokens: Vec<u32>,
  current_text: String,
}

impl<F> NaiveStreamingDetokenizer<F>
where
  F: Fn(&[u32]) -> String,
{
  /// Build a naive detokenizer from a decode callback and the tokenizer's
  /// `clean_up_tokenization_spaces` flag.
  pub fn new(decode: F, clean_up_spaces: bool) -> Self {
    let mut s = Self {
      decode,
      clean_up_spaces,
      tokens: Vec::new(),
      offset: 0,
      text: String::new(),
      current_tokens: Vec::new(),
      current_text: String::new(),
    };
    s.reset();
    s
  }

  fn recompute_text(&mut self) {
    if !self.current_tokens.is_empty() {
      let mut ct = (self.decode)(&self.current_tokens);
      let ends_replacement = ct.ends_with('\u{fffd}');
      let trailing_space = self.clean_up_spaces && !ct.is_empty() && ct.ends_with(' ');
      if ends_replacement || trailing_space {
        ct.pop();
      }
      self.current_text = ct;
    }
    if self.current_text.ends_with('\n') {
      self.text.push_str(&self.current_text);
      self.current_tokens.clear();
      self.current_text.clear();
    }
  }
}

impl<F> StreamingDetokenizer for NaiveStreamingDetokenizer<F>
where
  F: Fn(&[u32]) -> String,
{
  fn reset(&mut self) {
    self.offset = 0;
    self.tokens.clear();
    self.text.clear();
    self.current_tokens.clear();
    self.current_text.clear();
  }

  fn add_token(&mut self, token: u32) {
    self.current_tokens.push(token);
    self.tokens.push(token);
    self.recompute_text();
  }

  fn finalize(&mut self) {
    let decoded = (self.decode)(&self.current_tokens);
    self.text.push_str(&decoded);
    self.current_tokens.clear();
    self.current_text.clear();
  }

  fn text(&self) -> std::borrow::Cow<'_, str> {
    // Python's `text` property returns `_text + _current_text`: the committed
    // text plus the not-yet-flushed in-progress segment. Naive defers
    // mid-line output to `current_text`, so returning only `self.text` would
    // hide all incremental text until a newline/`finalize()` — breaking
    // token-by-token streaming via `last_segment`. Borrow when there is no
    // pending segment (zero alloc); only clone+stitch when one exists.
    if self.current_text.is_empty() {
      std::borrow::Cow::Borrowed(&self.text)
    } else {
      let mut s = String::with_capacity(self.text.len() + self.current_text.len());
      s.push_str(&self.text);
      s.push_str(&self.current_text);
      std::borrow::Cow::Owned(s)
    }
  }

  fn tokens(&self) -> &[u32] {
    &self.tokens
  }

  fn offset(&self) -> usize {
    self.offset
  }

  fn set_offset(&mut self, offset: usize) {
    self.offset = offset;
  }
}

impl<F> NaiveStreamingDetokenizer<F>
where
  F: Fn(&[u32]) -> String,
{
  /// The Python `text` property value: committed text plus the not-yet-flushed
  /// current segment. Identical to [`StreamingDetokenizer::text`] (kept for
  /// source compatibility); allocates only when a pending segment exists.
  pub fn combined_text(&self) -> String {
    use crate::tokenizer::stream::StreamingDetokenizer;
    self.text().into_owned()
  }
}

/// GPT-2 byte decoder (unicode char -> raw byte), regenerated by
/// `cargo xtask-codegen` into the committed
/// [`crate::tokenizer::generated`] module. Replaces the old per-detokenizer
/// runtime `HashMap`: zero runtime allocation, binary search.
#[cfg(feature = "tokenizer-gpt2")]
mod byte_decoder {
  use crate::tokenizer::generated::BYTE_DECODER;

  /// The codegen'd sorted table (exposed for the parity test).
  #[cfg(test)]
  pub(super) const TABLE: &[(char, u8)] = BYTE_DECODER;

  /// Look up the raw byte for a GPT-2 unicode char via binary search over the
  /// build-time-sorted static table. Behaviour-identical to the previous
  /// `HashMap<char, u8>` lookup.
  #[inline]
  pub(super) fn decode_char(c: char) -> Option<u8> {
    BYTE_DECODER
      .binary_search_by(|&(k, _)| k.cmp(&c))
      .ok()
      .map(|i| BYTE_DECODER[i].1)
  }
}

/// `SPMStreamingDetokenizer` — for SentencePiece models. Adds tokens to the
/// text when the next chunk starts with the SPM `▁` underscore, giving linear
/// complexity. `trim_space` drops a single leading space (Python
/// `SPMStreamingDetokenizer(trim_space=...)`).
#[cfg(feature = "tokenizer-spm")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-spm")))]
pub struct SpmStreamingDetokenizer {
  trim_space: bool,
  // Keyed by token id. A `HashMap` (not an id-indexed `Vec`) so a sparse or
  // corrupt vocab — e.g. a single huge id like `u32::MAX` from an untrusted
  // `tokenizer.json` — cannot trigger a multi-GB allocation / overflow panic.
  // O(1) lookup, behaviour-identical to the dense map for valid vocabs.
  tokenmap: std::collections::HashMap<u32, Vec<u8>>,
  tokens: Vec<u32>,
  offset: usize,
  text: String,
  unflushed: Vec<u8>,
}

#[cfg(feature = "tokenizer-spm")]
impl SpmStreamingDetokenizer {
  /// Build from `(token_string, id)` vocab pairs. `<0xHH>` byte tokens are
  /// decoded to their raw byte, matching Python.
  pub fn new<I, S>(vocab: I, trim_space: bool) -> Self
  where
    I: IntoIterator<Item = (S, u32)>,
    S: AsRef<str>,
  {
    let iter = vocab.into_iter();
    let mut tokenmap: std::collections::HashMap<u32, Vec<u8>> =
      std::collections::HashMap::with_capacity(iter.size_hint().0);
    for (value, id) in iter {
      let value = value.as_ref();
      // Python: `<0xHH>` → `bytes([int(value[3:5], 16)])`. Read the two hex
      // digit *bytes* directly (valid hex is always ASCII) instead of
      // `value[3..5]`: a token like `<0x€` would otherwise slice through a
      // multi-byte char (byte 5 is not a char boundary) and panic at
      // construction on an untrusted vocab. Lenient fallback (keep raw
      // bytes) preserved.
      let vb = value.as_bytes();
      let bytes = if vb.len() >= 5 && &vb[..3] == b"<0x" {
        std::str::from_utf8(&vb[3..5])
          .ok()
          .and_then(|h| u8::from_str_radix(h, 16).ok())
          .map(|b| vec![b])
          .unwrap_or_else(|| vb.to_vec())
      } else {
        vb.to_vec()
      };
      tokenmap.insert(id, bytes);
    }
    let mut s = Self {
      trim_space,
      tokenmap,
      tokens: Vec::new(),
      offset: 0,
      text: String::new(),
      unflushed: Vec::new(),
    };
    s.reset();
    s
  }

  fn try_flush(&mut self, force: bool) {
    // Replace the SPM separator (U+2581, bytes E2 96 81) with a space.
    let mut replaced: Vec<u8> = Vec::with_capacity(self.unflushed.len());
    let sep = "\u{2581}".as_bytes();
    let mut i = 0;
    while i < self.unflushed.len() {
      if self.unflushed[i..].starts_with(sep) {
        replaced.push(b' ');
        i += sep.len();
      } else {
        replaced.push(self.unflushed[i]);
        i += 1;
      }
    }
    let text = String::from_utf8_lossy(&replaced).into_owned();
    if !force && text.ends_with('\u{fffd}') {
      return;
    }
    let text = if self.text.is_empty() && self.trim_space && text.starts_with(' ') {
      text[1..].to_owned()
    } else {
      text
    };
    self.text.push_str(&text);
    self.unflushed.clear();
  }
}

#[cfg(feature = "tokenizer-spm")]
impl StreamingDetokenizer for SpmStreamingDetokenizer {
  fn reset(&mut self) {
    self.offset = 0;
    self.unflushed.clear();
    self.text.clear();
    self.tokens.clear();
  }

  fn add_token(&mut self, token: u32) {
    self.tokens.push(token);
    if let Some(v) = self.tokenmap.get(&token) {
      self.unflushed.extend_from_slice(v);
    }
    self.try_flush(false);
  }

  fn finalize(&mut self) {
    self.try_flush(true);
    self.unflushed.clear();
  }

  fn text(&self) -> std::borrow::Cow<'_, str> {
    // SPM accumulates the full decoded text into `self.text` (Python keeps
    // `self.text` authoritative, no separate `_current_text`): borrow it,
    // zero alloc, behaviour unchanged.
    std::borrow::Cow::Borrowed(&self.text)
  }

  fn tokens(&self) -> &[u32] {
    &self.tokens
  }

  fn offset(&self) -> usize {
    self.offset
  }

  fn set_offset(&mut self, offset: usize) {
    self.offset = offset;
  }
}

/// `BPEStreamingDetokenizer` — for OpenAI-style byte-level BPE models. Uses the
/// GPT-2 byte<->unicode table to decode and withholds single-space tokens for
/// one step so they can be cleaned (Python `BPEStreamingDetokenizer`).
#[cfg(feature = "tokenizer-bpe")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-bpe")))]
pub struct BpeStreamingDetokenizer {
  clean_spaces: bool,
  // Keyed by token id (see `SpmStreamingDetokenizer::tokenmap`): a `HashMap`
  // rather than an id-indexed `Vec` so a corrupt/sparse vocab with a huge id
  // cannot OOM/overflow at construction.
  tokenmap: std::collections::HashMap<u32, String>,
  // Python builds `tokenmap = [None] * len(vocab)`, so an absent id `< len`
  // (an in-range hole) yields `None` and an id `>= len` yields `"!"` via
  // `token < len(self.tokenmap)`. For a dense vocab `len == max_id + 1`, so
  // the boundary is exactly `token <= max_id`. Stored as the max vocab id
  // (computed O(1) at construction, no allocation) to reproduce that split
  // without the dense `Vec`.
  max_id: u32,
  tokens: Vec<u32>,
  offset: usize,
  text: String,
  unflushed: String,
}

// `BPE_SPACE_MATCHES` is regenerated by `cargo xtask-codegen` from
// `mlxrs/data/tokenizer/bpe_space_matches.toml` (single source of truth)
// into the committed `crate::tokenizer::generated` module.
#[cfg(feature = "tokenizer-bpe")]
use crate::tokenizer::generated::BPE_SPACE_MATCHES;

#[cfg(feature = "tokenizer-bpe")]
impl BpeStreamingDetokenizer {
  /// Build from `(token_string, id)` vocab pairs.
  pub fn new<I, S>(vocab: I, clean_spaces: bool) -> Self
  where
    I: IntoIterator<Item = (S, u32)>,
    S: AsRef<str>,
  {
    let iter = vocab.into_iter();
    let mut tokenmap: std::collections::HashMap<u32, String> =
      std::collections::HashMap::with_capacity(iter.size_hint().0);
    // Max vocab id == `len(tokenmap) - 1` for the dense Python list; tracked
    // here so the in-range/out-of-range `"!"` boundary needs no allocation.
    let mut max_id: u32 = 0;
    for (value, id) in iter {
      max_id = max_id.max(id);
      tokenmap.insert(id, value.as_ref().to_owned());
    }
    let mut s = Self {
      clean_spaces,
      tokenmap,
      max_id,
      tokens: Vec::new(),
      offset: 0,
      text: String::new(),
      unflushed: String::new(),
    };
    s.reset();
    s
  }

  fn decode_bytes(&self, seq: &str) -> String {
    let mut barr: Vec<u8> = Vec::with_capacity(seq.len());
    for c in seq.chars() {
      // `decode_char` returns `Option<u8>`: `None` is the ONLY miss (the
      // char is not a GPT-2 byte-level char). `Some(0)` is the *valid* byte
      // for U+0100 (GPT-2 byte 0x00) — it must be pushed, not treated as a
      // miss. The previous `Some(b) if b != 0` arm corrupted byte 0x00 into
      // U+0100 utf-8 text; `None => fallback` matches the byte-decoder table
      // and keeps streaming state-independent with `finalize` (which already
      // pushes `Some(0)`).
      match byte_decoder::decode_char(c) {
        Some(b) => barr.push(b),
        None => {
          let mut buf = [0u8; 4];
          barr.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
      }
    }
    String::from_utf8_lossy(&barr).into_owned()
  }

  fn maybe_trim_space(&self, current_text: &str) -> String {
    if current_text.is_empty() {
      return current_text.to_owned();
    }
    if !current_text.starts_with(' ') {
      return current_text.to_owned();
    }
    if self.text.is_empty() {
      return current_text[1..].to_owned();
    }
    if self.clean_spaces {
      let rest = &current_text[1..];
      if BPE_SPACE_MATCHES.iter().any(|m| rest.starts_with(m)) {
        return rest.to_owned();
      }
    }
    current_text.to_owned()
  }
}

#[cfg(feature = "tokenizer-bpe")]
impl StreamingDetokenizer for BpeStreamingDetokenizer {
  fn reset(&mut self) {
    self.offset = 0;
    self.unflushed.clear();
    self.text.clear();
    self.tokens.clear();
  }

  fn add_token(&mut self, token: u32) {
    self.tokens.push(token);
    // Python: `v = tokenmap[token] if token < len(tokenmap) else "!"`.
    // `tokenmap` is `[None] * len(vocab)`, so an in-range id with no entry is
    // `None` (yields `""` here, matching `str(None)`-free dense decode of an
    // unset slot), and only an id `>= len` (i.e. `token > max_id`, since for
    // a dense vocab `len == max_id + 1`) falls back to `"!"`. This reproduces
    // Python's `< len(tokenmap)` boundary exactly with O(1) memory.
    let v: &str = match self.tokenmap.get(&token) {
      Some(s) => s.as_str(),
      None if token <= self.max_id => "",
      None => "!",
    };
    self.unflushed.push_str(v);
    let text = self.decode_bytes(&self.unflushed);

    // Single bare-space tokens are held for one step so they can be cleaned.
    let single_space =
      v.chars().count() == 1 && v.chars().next().and_then(byte_decoder::decode_char) == Some(32);
    if !text.ends_with('\u{fffd}') && !single_space {
      let trimmed = self.maybe_trim_space(&text);
      self.text.push_str(&trimmed);
      self.unflushed.clear();
    }
  }

  fn finalize(&mut self) {
    let mut barr: Vec<u8> = Vec::new();
    for c in self.unflushed.chars() {
      if let Some(b) = byte_decoder::decode_char(c) {
        barr.push(b);
      }
    }
    let current_text = String::from_utf8_lossy(&barr).into_owned();
    let trimmed = self.maybe_trim_space(&current_text);
    self.text.push_str(&trimmed);
    self.unflushed.clear();
  }

  fn text(&self) -> std::borrow::Cow<'_, str> {
    // BPE accumulates the full decoded text into `self.text` (Python keeps
    // `self.text` authoritative, no separate `_current_text`): borrow it,
    // zero alloc, behaviour unchanged.
    std::borrow::Cow::Borrowed(&self.text)
  }

  fn tokens(&self) -> &[u32] {
    &self.tokens
  }

  fn offset(&self) -> usize {
    self.offset
  }

  fn set_offset(&mut self, offset: usize) {
    self.offset = offset;
  }
}

/// Which detokenizer class to instantiate, inferred from `tokenizer.json`'s
/// `decoder` field. Mirrors Python `load`'s `detokenizer_class` selection via
/// `_is_spm_decoder` / `_is_spm_decoder_no_space` / `_is_bpe_decoder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetokenizerClass {
  /// `NaiveStreamingDetokenizer` (default / unknown decoder).
  Naive,
  /// `SPMStreamingDetokenizer` with `trim_space=true`.
  Spm,
  /// `SPMStreamingDetokenizer` with `trim_space=false` (no-space SPM decoder).
  SpmNoSpace,
  /// `BPEStreamingDetokenizer`.
  Bpe,
}

#[cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))]
fn json_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
  use serde_json::Value;
  match (a, b) {
    (Value::Object(x), Value::Object(y)) => {
      x.len() == y.len()
        && x
          .iter()
          .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
    }
    (Value::Array(x), Value::Array(y)) => {
      x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| json_eq(p, q))
    }
    _ => a == b,
  }
}

#[cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))]
fn spm_decoder_target(with_strip: bool) -> serde_json::Value {
  let mut decoders = vec![
    serde_json::json!({"type": "Replace", "pattern": {"String": "▁"}, "content": " "}),
    serde_json::json!({"type": "ByteFallback"}),
    serde_json::json!({"type": "Fuse"}),
  ];
  if with_strip {
    decoders.push(serde_json::json!({"type": "Strip", "content": " ", "start": 1, "stop": 0}));
  }
  serde_json::json!({"type": "Sequence", "decoders": decoders})
}

/// Infer the detokenizer class from a parsed `tokenizer.json` `decoder` node.
/// Returns [`DetokenizerClass::Naive`] when the decoder is absent/unknown.
///
/// Available only with `tokenizer-spm` or `tokenizer-bpe` (it parses the
/// `decoder` JSON node, which needs `serde_json`).
#[cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe"))]
#[cfg_attr(
  docsrs,
  doc(cfg(any(feature = "tokenizer-spm", feature = "tokenizer-bpe")))
)]
pub fn infer_detokenizer_class(decoder: Option<&serde_json::Value>) -> DetokenizerClass {
  let Some(decoder) = decoder else {
    return DetokenizerClass::Naive;
  };
  if json_eq(&spm_decoder_target(true), decoder) {
    DetokenizerClass::Spm
  } else if json_eq(&spm_decoder_target(false), decoder) {
    DetokenizerClass::SpmNoSpace
  } else if decoder.get("type").and_then(|t| t.as_str()) == Some("ByteLevel") {
    DetokenizerClass::Bpe
  } else {
    DetokenizerClass::Naive
  }
}

/// A streaming naive detokenizer over a HuggingFace tokenizer — the
/// non-generic concrete implementation that backs
/// [`Detokenizer::Naive`].
///
/// Compared to [`NaiveStreamingDetokenizer<F>`], this avoids the
/// generic `F` parameter so the [`Detokenizer`] enum can hold the
/// Naive variant as a sized field (an enum cannot contain a closure
/// type directly without monomorphizing the whole enum, which would
/// defeat the unification). Behavior is byte-identical to
/// `NaiveStreamingDetokenizer::new(|ids| hf.decode(ids, false), clean)`
/// — the only difference is that the [`tokenizers::Tokenizer`] is
/// stored inline rather than captured by a moved closure.
#[cfg(feature = "tokenizer-stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
pub struct NaiveHfDetokenizer {
  hf: tokenizers::Tokenizer,
  clean_up_spaces: bool,
  tokens: Vec<u32>,
  offset: usize,
  text: String,
  current_tokens: Vec<u32>,
  current_text: String,
}

#[cfg(feature = "tokenizer-stream")]
impl NaiveHfDetokenizer {
  /// Build from a (cloned) [`tokenizers::Tokenizer`] and the
  /// tokenizer's `clean_up_tokenization_spaces` flag.
  pub fn new(hf: tokenizers::Tokenizer, clean_up_spaces: bool) -> Self {
    Self {
      hf,
      clean_up_spaces,
      tokens: Vec::new(),
      offset: 0,
      text: String::new(),
      current_tokens: Vec::new(),
      current_text: String::new(),
    }
  }

  fn decode(&self, ids: &[u32]) -> String {
    self.hf.decode(ids, false).unwrap_or_default()
  }

  fn recompute_text(&mut self) {
    if !self.current_tokens.is_empty() {
      let mut ct = self.decode(&self.current_tokens);
      let ends_replacement = ct.ends_with('\u{fffd}');
      let trailing_space = self.clean_up_spaces && !ct.is_empty() && ct.ends_with(' ');
      if ends_replacement || trailing_space {
        ct.pop();
      }
      self.current_text = ct;
    }
    if self.current_text.ends_with('\n') {
      self.text.push_str(&self.current_text);
      self.current_tokens.clear();
      self.current_text.clear();
    }
  }
}

#[cfg(feature = "tokenizer-stream")]
impl StreamingDetokenizer for NaiveHfDetokenizer {
  fn reset(&mut self) {
    self.offset = 0;
    self.tokens.clear();
    self.text.clear();
    self.current_tokens.clear();
    self.current_text.clear();
  }

  fn add_token(&mut self, token: u32) {
    self.current_tokens.push(token);
    self.tokens.push(token);
    self.recompute_text();
  }

  fn finalize(&mut self) {
    let decoded = self.decode(&self.current_tokens);
    self.text.push_str(&decoded);
    self.current_tokens.clear();
    self.current_text.clear();
  }

  fn text(&self) -> std::borrow::Cow<'_, str> {
    if self.current_text.is_empty() {
      std::borrow::Cow::Borrowed(&self.text)
    } else {
      let mut s = String::with_capacity(self.text.len() + self.current_text.len());
      s.push_str(&self.text);
      s.push_str(&self.current_text);
      std::borrow::Cow::Owned(s)
    }
  }

  fn tokens(&self) -> &[u32] {
    &self.tokens
  }

  fn offset(&self) -> usize {
    self.offset
  }

  fn set_offset(&mut self, offset: usize) {
    self.offset = offset;
  }
}

/// A streaming detokenizer — the enum-unified replacement for the
/// prior `Box<dyn StreamingDetokenizer>` trait-object alias.
///
/// # Breaking change (P1 #111)
///
/// Previously [`crate::tokenizer::wrapper::BoxedDetokenizer`] aliased
/// `Box<dyn StreamingDetokenizer>` — one vtable indirection per
/// emitted token (the `detok.add_token(token)` call in the
/// `stream_generate` hot loop). The enum dispatches via `match`,
/// inlining the canonical Naive / SPM / BPE variants and reserving
/// [`Self::Custom`] for out-of-tree detokenizers.
///
/// The lower per-token dispatch cost of LM-1d (one indirection per
/// token vs LM-1a/b's ~5 per token) is partly mitigated by
/// branch-prediction warming on the consistent variant — a single
/// generation run hits the same variant every token. The enum
/// unification still wins on:
/// - **inlining**: each variant's `add_token` body inlines through
///   the `match`, so the per-variant logic (SPM's `try_flush`, BPE's
///   `decode_bytes`) can vectorize / unroll.
/// - **monomorphization**: no per-call vtable lookup, even cold.
///
/// Construct via [`crate::tokenizer::Tokenizer::detokenizer`] (the
/// canonical class is inferred from `tokenizer.json`'s `decoder` node);
/// out-of-tree detokenizers plug in through [`Self::Custom`].
#[cfg(feature = "tokenizer-stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "tokenizer-stream")))]
pub enum Detokenizer {
  /// Naive re-decode detokenizer — the default / unknown-decoder
  /// fallback. O(T²) over the longest line, matching Python. Boxed
  /// because the wrapped [`tokenizers::Tokenizer`] is `~1KB` (dwarfs
  /// the other variants); boxing keeps the enum's discriminant +
  /// inline payload to a pointer, so per-token `match` dispatch
  /// reads one pointer's worth of memory.
  Naive(Box<NaiveHfDetokenizer>),
  /// SentencePiece streaming detokenizer (`tokenizer-spm`-gated). Linear-time.
  #[cfg(feature = "tokenizer-spm")]
  Spm(SpmStreamingDetokenizer),
  /// Byte-level BPE streaming detokenizer (`tokenizer-bpe`-gated).
  #[cfg(feature = "tokenizer-bpe")]
  Bpe(BpeStreamingDetokenizer),
  /// Custom out-of-tree detokenizer (escape hatch). One indirection
  /// per call.
  Custom(Box<dyn StreamingDetokenizer>),
}

#[cfg(feature = "tokenizer-stream")]
impl StreamingDetokenizer for Detokenizer {
  fn reset(&mut self) {
    match self {
      Self::Naive(d) => d.reset(),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.reset(),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.reset(),
      Self::Custom(d) => d.reset(),
    }
  }

  fn add_token(&mut self, token: u32) {
    match self {
      Self::Naive(d) => d.add_token(token),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.add_token(token),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.add_token(token),
      Self::Custom(d) => d.add_token(token),
    }
  }

  fn finalize(&mut self) {
    match self {
      Self::Naive(d) => d.finalize(),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.finalize(),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.finalize(),
      Self::Custom(d) => d.finalize(),
    }
  }

  fn text(&self) -> std::borrow::Cow<'_, str> {
    match self {
      Self::Naive(d) => d.text(),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.text(),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.text(),
      Self::Custom(d) => d.text(),
    }
  }

  fn tokens(&self) -> &[u32] {
    match self {
      Self::Naive(d) => d.tokens(),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.tokens(),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.tokens(),
      Self::Custom(d) => d.tokens(),
    }
  }

  fn offset(&self) -> usize {
    match self {
      Self::Naive(d) => d.offset(),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.offset(),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.offset(),
      Self::Custom(d) => d.offset(),
    }
  }

  fn set_offset(&mut self, offset: usize) {
    match self {
      Self::Naive(d) => d.set_offset(offset),
      #[cfg(feature = "tokenizer-spm")]
      Self::Spm(d) => d.set_offset(offset),
      #[cfg(feature = "tokenizer-bpe")]
      Self::Bpe(d) => d.set_offset(offset),
      Self::Custom(d) => d.set_offset(offset),
    }
  }
}

#[cfg(all(test, feature = "tokenizer-gpt2"))]
mod byte_decoder_tests {
  /// The previous *runtime* `make_byte_decoder()` algorithm, reconstructed
  /// verbatim. The `cargo xtask-codegen`-generated `BYTE_DECODER` must
  /// round-trip byte-identically to this for every key.
  fn legacy_make_byte_decoder() -> std::collections::HashMap<char, u8> {
    let limits: [u32; 7] = [
      0,
      '!' as u32,
      '~' as u32 + 1,
      '¡' as u32,
      '¬' as u32 + 1,
      '®' as u32,
      'ÿ' as u32 + 1,
    ];
    let mut map = std::collections::HashMap::new();
    let mut n: u32 = 0;
    for (i, w) in limits.windows(2).enumerate() {
      let (start, stop) = (w[0], w[1]);
      if i % 2 == 0 {
        for b in start..stop {
          let c = char::from_u32(256 + n).unwrap();
          map.insert(c, b as u8);
          n += 1;
        }
      } else {
        for b in start..stop {
          let c = char::from_u32(b).unwrap();
          map.insert(c, b as u8);
        }
      }
    }
    map
  }

  #[test]
  fn generated_byte_decoder_matches_legacy_algorithm() {
    let legacy = legacy_make_byte_decoder();
    // Same cardinality.
    assert_eq!(super::byte_decoder::TABLE.len(), legacy.len());
    // Every legacy (char -> byte) entry is reproduced by the generated slice.
    for (&c, &b) in &legacy {
      assert_eq!(
        super::byte_decoder::decode_char(c),
        Some(b),
        "mismatch for char {c:?}"
      );
    }
    // Sorted-by-char invariant (binary search precondition).
    assert!(
      super::byte_decoder::TABLE
        .windows(2)
        .all(|w| w[0].0 < w[1].0)
    );
    // No key outside the legacy map.
    for &(c, b) in super::byte_decoder::TABLE {
      assert_eq!(legacy.get(&c), Some(&b));
    }
  }
}
