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

use crate::{
  error::{AllocFailurePayload, ArithmeticOverflowPayload, Error, Result},
  model_validation::reserve_or_error,
  tokenizer::sentencepiece::SentencePieceTokenizer,
};

/// The SentencePiece model file the reference loads
/// (`sensevoice.py:581`, swift `SenseVoiceTokenizer.swift` prefers any `*.model`).
pub const SPM_MODEL_FILE: &str = "chn_jpn_yue_eng_ko_spectok.bpe.model";

/// Generous upper bound on the SentencePiece `.model` we read into memory — a
/// bounded-read **soundness** guard against a hostile model directory planting a
/// huge `.model`, NOT a valid-input cap. Real SentencePiece models are typically
/// a few MiB, but a very large multilingual vocabulary can be larger; `64 MiB`
/// is generous headroom so no real `.model` is rejected, while still bounding
/// the read so a planted multi-GB file cannot OOM the loader. Deliberately far
/// above the 1 MiB `config.json` cap (a binary protobuf, not a small JSON).
pub(crate) const MAX_SPM_MODEL_BYTES: u64 = 64 << 20;

/// The `tokens.json` piece-list file the reference falls back to
/// (`sensevoice.py:582`, swift `:18`).
pub const TOKENS_JSON_FILE: &str = "tokens.json";

/// The `▁` (U+2581) metaspace marker the `tokens.json` path strips to a space
/// (`sensevoice.py:446`, swift `:45`).
const METASPACE: char = '\u{2581}';

/// The SenseVoice detokenizer — the two-tier SentencePiece / `tokens.json`
/// decode (`sensevoice.py:439-448`).
///
/// Built once at load (the `post_load_hook` equivalent) from the model
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

  /// Build the SentencePiece variant from already-read `.model` protobuf bytes
  /// (`sensevoice.py:588-590`: `sp.Load(...)`), via
  /// [`SentencePieceTokenizer::from_model_bytes`]. The caller is responsible for
  /// reading the bytes through a bounded reader (see [`Self::from_spm_file`]).
  ///
  /// # Errors
  /// Propagates [`SentencePieceTokenizer::from_model_bytes`]'s protobuf-parse
  /// errors (malformed / truncated / empty-vocabulary input).
  pub fn from_spm_bytes(data: &[u8]) -> Result<Self> {
    Ok(Self::SentencePiece(Box::new(
      SentencePieceTokenizer::from_model_bytes(data)?,
    )))
  }

  /// Load the SentencePiece variant from a `.bpe.model` file on disk
  /// (`sensevoice.py:588-590`: `sp.Load(str(bpe_path))`).
  ///
  /// The `.model` is read through the shared bounded `read_bounded_bytes_file`
  /// reader with the generous `MAX_SPM_MODEL_BYTES` cap — an open-once,
  /// TOCTOU-closed, non-regular-file-rejecting, size-capped read — so a hostile
  /// model directory cannot OOM the loader by planting a huge `.model` (the
  /// bounded-read soundness convention the `config.json` / `tokens.json` /
  /// `am.mvn` reads use, with a cap sized for a binary tokenizer model rather
  /// than a small JSON), then parsed via [`Self::from_spm_bytes`].
  ///
  /// Returns `Ok(None)` if the file is absent (the caller's "fall through to
  /// `tokens.json`" signal — this also closes the stat-then-read TOCTOU window
  /// where the file vanishes after a presence check).
  ///
  /// # Errors
  /// - [`crate::error::Error::FileIo`] (open / read failure, non-regular file) or
  ///   [`crate::error::Error::CapExceeded`] from the bounded read;
  /// - propagates [`Self::from_spm_bytes`]'s protobuf-parse errors.
  pub fn from_spm_file(path: &Path) -> Result<Option<Self>> {
    match crate::lm::load::read_bounded_bytes_file(
      path,
      "sensevoice spm model",
      MAX_SPM_MODEL_BYTES,
    )? {
      Some(bytes) => Ok(Some(Self::from_spm_bytes(&bytes)?)),
      None => Ok(None),
    }
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
  /// contract (the [`crate::audio::stt::model::CtcModel`] trait and the shared
  /// `greedy_ctc_transcribe` driver require an infallible signature). The rich
  /// [`Transcribe`](crate::audio::stt::model::Transcribe) path uses the fallible
  /// [`Self::try_decode`] companion instead.
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

  /// The fallible analogue of [`Self::decode`] — the same two-tier
  /// SentencePiece / `tokens.json` / id-join decode (`sensevoice.py:439-448`),
  /// but every id / output buffer this detokenizer owns is reserved FALLIBLY
  /// (typed [`Error::AllocFailure`]) instead of the abort the infallible
  /// `decode`'s `collect` / `String::with_capacity` growth would raise.
  ///
  /// Used by the rich [`SenseVoiceModel::transcribe_rich`](super::model::SenseVoiceModel::transcribe_rich)
  /// /
  /// [`Transcribe`](crate::audio::stt::model::Transcribe) path. The infallible
  /// [`Self::decode`] is retained for the
  /// [`CtcModel::decode_ids`](crate::audio::stt::model::CtcModel::decode_ids)
  /// seam the shared `greedy_ctc_transcribe` driver requires.
  ///
  /// The decode result is byte-identical to [`Self::decode`]; only the
  /// allocation behavior differs.
  ///
  /// # Errors
  /// [`Error::AllocFailure`] if reserving the id buffer (bounded by the id
  /// count) or the joined-output buffer (bounded by the summed piece lengths)
  /// exhausts the allocator. For the SentencePiece variant, only the
  /// SenseVoice-owned `usize` id buffer is reserved fallibly; the shared
  /// [`SentencePieceTokenizer::decode`] owns its internal output buffer.
  pub fn try_decode(&self, ids: &[u32]) -> Result<String> {
    match self {
      Self::SentencePiece(tokenizer) => {
        // Reserve the `usize` id buffer fallibly (bounded by the id count),
        // then route through the shared byte-fallback decode.
        let mut usize_ids: Vec<usize> = Vec::new();
        reserve_or_error(
          &mut usize_ids,
          "sensevoice tokenizer try_decode: spm id buffer",
          ids.len(),
        )?;
        for &id in ids {
          usize_ids.push(id as usize);
        }
        Ok(tokenizer.decode(&usize_ids))
      }
      Self::TokenList(tokens) => try_decode_token_list(tokens, ids),
      Self::IdJoin => try_join_ids(ids),
    }
  }
}

/// Wrap a `String::try_reserve_exact` failure into a typed
/// [`Error::AllocFailure`] (the [`crate::model_validation::reserve_or_error`]
/// analogue for `String`, which the `TryReserve` trait does not cover).
fn reserve_string_or_error(s: &mut String, item: &'static str, additional: usize) -> Result<()> {
  s.try_reserve_exact(additional).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "sensevoice tokenizer try_decode",
      item,
      additional as u64,
      e,
    ))
  })
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

/// The fallible analogue of [`decode_token_list`]: index `ids` into the piece
/// list, join + strip the metaspace marker + trim, but reserve the output buffer
/// FALLIBLY and write the metaspace-normalized text directly (so the extra
/// `replace` allocation the infallible path incurs is avoided).
///
/// Byte-identical to [`decode_token_list`]: each `▁` is substituted with a
/// single space inline (== `replace(METASPACE, " ")`), then the result is
/// trimmed in place (no reallocation). The reserved capacity is the summed
/// in-range piece byte-length — an upper bound, since substituting a 3-byte `▁`
/// for a 1-byte space only shrinks the output. The sum is accumulated with
/// `checked_add` so it cannot wrap to an undersized capacity.
///
/// # Errors
/// - [`Error::ArithmeticOverflow`] if the summed piece byte-length overflows
///   `usize` (before any reservation);
/// - [`Error::AllocFailure`] if reserving the output buffer exhausts the
///   allocator.
fn try_decode_token_list(tokens: &[String], ids: &[u32]) -> Result<String> {
  // The summed in-range piece lengths bound the (pre-trim) output; substituting
  // the 3-byte metaspace for a 1-byte space never grows past it. Accumulate with
  // `checked_add` so a pathological id/piece set cannot WRAP the `usize` sum to a
  // smaller capacity (in release) — a wrapped reservation would then let the
  // subsequent infallible `out.push` grow unbounded, defeating the reserve guard.
  let mut capacity: usize = 0;
  for &id in ids {
    if let Some(piece) = tokens.get(id as usize) {
      capacity = capacity.checked_add(piece.len()).ok_or_else(|| {
        Error::ArithmeticOverflow(ArithmeticOverflowPayload::new(
          "sensevoice try_decode_token_list: summed piece byte-length",
          "usize",
        ))
      })?;
    }
  }
  let mut out = String::new();
  reserve_string_or_error(&mut out, "token-list joined output", capacity)?;

  for &id in ids {
    // The reference indexes `self._token_list[t] for t in token_ids if
    // 0 <= t < len(...)` (`sensevoice.py:444`): an out-of-range id is skipped.
    if let Some(piece) = tokens.get(id as usize) {
      // Substitute the metaspace marker inline (== `replace(METASPACE, " ")`,
      // `sensevoice.py:446`) so no separate `replace` pass allocates.
      for ch in piece.chars() {
        out.push(if ch == METASPACE { ' ' } else { ch });
      }
    }
  }

  // `.trim()` in place (`sensevoice.py:446`): drop the trailing then the leading
  // whitespace, both shrinking `out` without a fresh allocation. The byte offsets
  // come from `trim_end` / `trim_start` lengths (char boundaries), so no raw
  // pointer arithmetic is needed.
  let end = out.trim_end().len();
  out.truncate(end);
  let start = out.len() - out.trim_start().len();
  out.drain(..start);
  Ok(out)
}

/// The fallible analogue of [`join_ids`]: render `ids` as their space-joined
/// decimals (`sensevoice.py:448`), reserving the output buffer FALLIBLY.
///
/// Byte-identical to [`join_ids`]. The reserved capacity bounds the decimal
/// digits (≤ 10 per `u32`) plus the `n - 1` separators.
///
/// # Errors
/// [`Error::AllocFailure`] if reserving the output buffer exhausts the
/// allocator.
fn try_join_ids(ids: &[u32]) -> Result<String> {
  // Each `u32` is at most 10 decimal digits; with `n - 1` single-space
  // separators the join is at most `11 * n` bytes.
  let capacity = ids.len().saturating_mul(11);
  let mut out = String::new();
  reserve_string_or_error(&mut out, "id-join output", capacity)?;
  for (i, id) in ids.iter().enumerate() {
    if i != 0 {
      out.push(' ');
    }
    // Writing a decimal integer into a `String` is infallible (the buffer is
    // pre-reserved to the digit + separator bound above).
    let _ = write!(out, "{id}");
  }
  Ok(out)
}

#[cfg(test)]
mod tests;
