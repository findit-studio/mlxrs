//! The SenseVoice-Small detokenizer — the two-tier SentencePiece / `tokens.json`
//! decode the reference uses to render the collapsed CTC speech ids to text.
//!
//! Faithful port of the reference `_decode_tokens` (`sensevoice.py:439-448`)
//! and the swift `SenseVoiceTokenizer.decode` (`SenseVoiceTokenizer.swift:36-48`),
//! which resolve a decode strategy in this precedence:
//!
//! 1. **SentencePiece** — the `chn_jpn_yue_eng_ko_spectok.bpe.model` SPM, loaded
//!    via the shared [`crate::tokenizer::sentencepiece::SentencePieceTokenizer`]
//!    (the `tokenizer-spm` feature). `decode` routes the collapsed speech ids
//!    through its byte-fallback-aware [`SentencePieceTokenizer::decode`], which
//!    reassembles `<0xHH>` pieces, strips the `▁` metaspace marker, and trims —
//!    exactly the python `sp.decode` path (`sensevoice.py:441`).
//! 2. **`tokens.json` list** — a `List[str]` of pieces; decode indexes each id,
//!    joins, replaces `"▁"` with a space, and trims (`sensevoice.py:443-447`,
//!    swift `:40-46`). The pure-Rust no-SentencePiece path.
//! 3. **id join** — the degenerate no-asset fallback: the space-joined decimal
//!    ids (`sensevoice.py:448`, swift `:47`).
//!
//! The tokenizer holds NONE of the rich-transcription query / tag ids: those
//! live in the vocab id space and are read by argmax position
//! ([`super::model::SenseVoiceModel::rich_info`]), never decoded as text. This
//! detokenizer only ever sees the post-prefix, blank-collapsed SPEECH ids.
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

use std::{fmt::Write as _, path::Path};

use crate::{error::Result, tokenizer::sentencepiece::SentencePieceTokenizer};

/// The SentencePiece model file the reference loads
/// (`sensevoice.py:581`, swift `SenseVoiceTokenizer.swift` prefers any `*.model`).
pub const SPM_MODEL_FILE: &str = "chn_jpn_yue_eng_ko_spectok.bpe.model";

/// The `tokens.json` piece-list file the reference falls back to
/// (`sensevoice.py:582`, swift `:18`).
pub const TOKENS_JSON_FILE: &str = "tokens.json";

/// The `▁` (U+2581) metaspace marker the `tokens.json` path strips to a space
/// (`sensevoice.py:446`, swift `:45`).
const METASPACE: char = '\u{2581}';

/// The SenseVoice detokenizer — the two-tier SentencePiece / `tokens.json`
/// decode (`sensevoice.py:439-448`).
///
/// Built once at load (the Phase 4 `post_load_hook` equivalent) from the model
/// directory's assets and consulted by [`super::model::SenseVoiceModel`]'s CTC
/// decode. The three variants mirror the reference's three decode strategies in
/// their precedence order; [`Self::decode`] dispatches on which assets were
/// present.
#[derive(Debug)]
pub enum SenseVoiceTokenizer {
  /// The SentencePiece path — the `.bpe.model` SPM was loaded
  /// (`sensevoice.py:584-590`). [`Box`]ed because the
  /// [`SentencePieceTokenizer`] (its vocabulary + trie + lattice arena) is far
  /// larger than the other variants, so an unboxed variant would inflate every
  /// `SenseVoiceTokenizer` value.
  SentencePiece(Box<SentencePieceTokenizer>),
  /// The `tokens.json` piece-list path — no SPM, but a `tokens.json` was
  /// present (`sensevoice.py:594-596`).
  TokenList(Vec<String>),
  /// The degenerate no-asset path — neither a `.bpe.model` nor a `tokens.json`
  /// was present; ids render as their space-joined decimals
  /// (`sensevoice.py:448`).
  IdJoin,
}

impl SenseVoiceTokenizer {
  /// Build directly from a loaded [`SentencePieceTokenizer`] (the
  /// `sensevoice.py:584-590` SPM path).
  #[inline(always)]
  pub fn from_sentencepiece(tokenizer: SentencePieceTokenizer) -> Self {
    Self::SentencePiece(Box::new(tokenizer))
  }

  /// Build from a `tokens.json` piece list (the `sensevoice.py:594-596`
  /// fallback path).
  #[inline(always)]
  pub fn from_token_list(tokens: Vec<String>) -> Self {
    Self::TokenList(tokens)
  }

  /// The degenerate no-asset detokenizer (the `sensevoice.py:448` id-join path).
  #[inline(always)]
  pub const fn id_join() -> Self {
    Self::IdJoin
  }

  /// Load the SentencePiece variant from a `.bpe.model` file on disk
  /// (`sensevoice.py:588-590`: `sp.Load(str(bpe_path))`).
  ///
  /// # Errors
  /// Propagates [`SentencePieceTokenizer::from_model_file`]'s
  /// [`crate::error::Error::FileIo`] (read failure) / parse errors.
  pub fn from_spm_file(path: &Path) -> Result<Self> {
    Ok(Self::SentencePiece(Box::new(
      SentencePieceTokenizer::from_model_file(path)?,
    )))
  }

  /// `true` if this is the degenerate id-join variant (neither asset present).
  #[inline(always)]
  pub const fn is_id_join(&self) -> bool {
    matches!(self, Self::IdJoin)
  }

  /// Decode a collapsed speech-id sequence to text, mirroring `_decode_tokens`
  /// (`sensevoice.py:439-448`) / swift `decode` (`:36-48`).
  ///
  /// - **SentencePiece**: routes through [`SentencePieceTokenizer::decode`] —
  ///   the byte-fallback reassembly + metaspace strip + trim it performs IS the
  ///   reference `sp.decode` behavior (`sensevoice.py:441`). The driver hands
  ///   `u32` ids; SentencePiece indexes by `usize`, so each is widened (a CTC
  ///   class id is a small non-negative index, well within `usize`).
  /// - **`tokens.json`**: indexes each in-range id into the piece list, joins,
  ///   replaces `▁` with a space, and trims (`sensevoice.py:443-447`).
  ///   Out-of-range ids contribute nothing (the reference's `if 0 <= t <
  ///   len(...)` guard, `:444`).
  /// - **id join**: the space-joined decimal ids (`sensevoice.py:448`).
  ///
  /// Total over every id slice (no panic / no error path) — the
  /// [`super::model::SenseVoiceModel`] CTC `decode_ids` is infallible by
  /// contract.
  pub fn decode(&self, ids: &[u32]) -> String {
    match self {
      Self::SentencePiece(tokenizer) => {
        let usize_ids: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        tokenizer.decode(&usize_ids)
      }
      Self::TokenList(tokens) => decode_token_list(tokens, ids),
      Self::IdJoin => join_ids(ids),
    }
  }
}

/// Index `ids` into the `tokens.json` piece list, join, strip the metaspace
/// marker, and trim — the reference `tokens.json` decode
/// (`sensevoice.py:443-447`): `"".join(pieces).replace("▁", " ").strip()` over
/// the in-range pieces.
fn decode_token_list(tokens: &[String], ids: &[u32]) -> String {
  // Pre-size to the summed piece lengths so the join does not reallocate.
  let capacity: usize = ids
    .iter()
    .filter_map(|&id| tokens.get(id as usize))
    .map(String::len)
    .sum();
  let mut joined = String::with_capacity(capacity);
  for &id in ids {
    // The reference indexes `self._token_list[t] for t in token_ids if
    // 0 <= t < len(...)` (`sensevoice.py:444`): an out-of-range id is skipped.
    if let Some(piece) = tokens.get(id as usize) {
      joined.push_str(piece);
    }
  }
  joined.replace(METASPACE, " ").trim().to_string()
}

/// Render `ids` as their space-joined decimals — the degenerate no-asset
/// fallback (`sensevoice.py:448`: `" ".join(str(t) for t in token_ids)`).
fn join_ids(ids: &[u32]) -> String {
  let mut out = String::new();
  for (i, id) in ids.iter().enumerate() {
    if i != 0 {
      out.push(' ');
    }
    // Writing a decimal integer into a `String` is infallible.
    let _ = write!(out, "{id}");
  }
  out
}

#[cfg(test)]
mod tests;
