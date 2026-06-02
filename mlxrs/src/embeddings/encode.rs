//! The `encode` entry — tokenize a batch of texts per the **model's** declared
//! [`TextEncoding`], pad them into a `(batch, seq)` id + mask tensor, and run
//! them through the model's universal text embedding into a `(batch, dim)`
//! embedding matrix.
//!
//! Ports the orchestration of python `mlx-embeddings` `utils.py::generate`
//! (tokenize with padding / truncation / `max_length`, run the model, return the
//! embeddings) and swift `MLXEmbedders` `EmbedderModelContainer.perform` (encode
//! each text → pad → build the mask → run the model → `eval`).
//!
//! Tokenization, padding, pooling, and normalization are all the **model's**
//! concern now: a model bakes its [`TextEncoding`] (how it tokenizes + pads) and
//! applies its own pooling inside [`TextEmbedder::embed_text`] (a sentence-encoder
//! via its baked [`PoolingConfig`](crate::embeddings::PoolingConfig) inside
//! [`pool_embed`](crate::embeddings::pool_embed); a dual-tower text tower
//! directly). So `encode` is thin — it reads the model's [`TextEncoding`],
//! tokenizes + pads per it, and calls `embed_text`. The pooling / normalization
//! / similarity helpers remain public drivers a model composes; `encode` itself
//! no longer pools.
//!
//! **Input contract.** `encode` faithfully tokenizes the caller's input
//! UNMODIFIED and truncates a [`Padding::FixedLength`] encoding to its `length`
//! (head-truncation, sticky-EOS into the final pooled slot when the encoding
//! declares an `eos_token_id`), exactly as the reference SigLIP processor does.
//! It imposes no input-size limit: bounding an oversized / untrusted prompt is
//! the **consuming application's** responsibility, not the library's. Every
//! buffer this path allocates is fallible (`try_with_capacity` /
//! [`alloc_filled`](crate::model_validation::alloc_filled)), so a pathologically
//! large prompt or configuration yields a typed [`Error::AllocFailure`] /
//! arithmetic-overflow error rather than a panic.

use crate::{
  array::Array,
  error::{
    ArithmeticOverflowPayload, Error, LengthMismatchPayload, OutOfRangePayload, Result,
    try_with_capacity,
  },
  tokenizer::{EncodeOptions, Tokenizer},
};

use super::embed::{Padding, TextEmbedder, TextEncoding};

/// Range-check a `u32` token / pad id against the `i32` index dtype, mapping an
/// id `> i32::MAX` (realistically never) to a recoverable [`Error::OutOfRange`]
/// rather than a silent wrap to a negative index. `context` names which id path
/// the value came from.
fn id_to_i32(id: u32, context: &'static str) -> Result<i32> {
  i32::try_from(id).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      context,
      "must fit in i32 (the MLX index dtype)",
      smol_str::format_smolstr!("{id}"),
    ))
  })
}

/// Encode each text via the pad-stripping path and return one `(ids, mask)` row
/// per text, with real tokens only (the tokenizer's own padding stripped) and
/// the per-sequence truncation cap applied.
///
/// Each text is encoded via [`Tokenizer::encode_with`] with
/// [`return_attention_mask`](EncodeOptions::return_attention_mask), which strips
/// any HF padding cells (e.g. when the loaded `tokenizer.json` has padding
/// enabled) and returns only the *attended* ids with an all-`1` mask — so the
/// per-text `(ids, mask)` describe real tokens only, independent of the
/// tokenizer's padding setting.
///
/// The caller's input is tokenized UNMODIFIED. When `max_length` is `Some(cap)`,
/// each encoding is head-truncated to `cap` ids ([`return_attention_mask`]'s
/// `truncate_to`, HF `TruncationDirection::Right` semantics); `None` leaves the
/// encoding at its full length. Bounding an oversized / untrusted prompt is the
/// consuming application's responsibility — this path imposes no input-byte
/// limit.
fn encode_rows(
  tokenizer: &Tokenizer,
  texts: &[&str],
  add_special_tokens: bool,
  max_length: Option<usize>,
) -> Result<Vec<(Vec<u32>, Vec<u8>)>> {
  let opts = EncodeOptions::new()
    .with_add_special(add_special_tokens)
    .with_truncate_to(max_length)
    .with_return_attention_mask(true);
  let mut rows: Vec<(Vec<u32>, Vec<u8>)> = try_with_capacity(texts.len())?;
  for &text in texts {
    let enc = tokenizer.encode_with(text, &opts)?;
    // Validate the length-equality invariant on the borrowed slices BEFORE
    // moving — `with_return_attention_mask(true)` guarantees `mask.len()
    // == ids.len()` including the legitimate `(0, 0)` case. After the check
    // passes, `into_parts()` moves both `Vec`s into the row without cloning.
    if enc.attention_mask().len() != enc.ids().len() {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "encode: encode_with(return_attention_mask=true) mask.len() must match ids.len()",
        enc.ids().len(),
        enc.attention_mask().len(),
      )));
    }
    rows.push(enc.into_parts());
  }
  Ok(rows)
}

/// Build the `(batch, seq_len)` `i32` id + `f32` mask arrays from per-text rows,
/// for the [`Padding::DynamicRightPad`] scheme: right-pad each row to the batch
/// max length with `pad_token_id` and mask `0` over the pad cells.
///
/// Returns `(input_ids, attention_mask, seq_len)`:
/// - `input_ids` — `(batch, seq_len)` `i32` (right-padded). `I32` is MLX's
///   default index dtype for the embedding `take` / gather a model performs, so
///   the lookup never has to cast. Each `u32` id is converted with a CHECKED
///   `i32::try_from` (an id `> i32::MAX` — realistically never — yields a
///   recoverable [`Error::OutOfRange`] rather than silently wrapping negative);
/// - `attention_mask` — `(batch, seq_len)` `f32` (`1.0` / `0.0`);
/// - `seq_len` — the batch max length (after per-text truncation).
///
/// Each row's own mask is all-`1` (pad-stripped in [`encode_rows`]); the only
/// `0` mask cells come from the manual padding appended here. This makes the
/// result correct whether the tokenizer has padding enabled or disabled.
fn build_dynamic_right_pad(
  rows: &[(Vec<u32>, Vec<u8>)],
  pad_token_id: u32,
) -> Result<(Array, Array, usize)> {
  let batch = rows.len();
  let seq_len = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
  let total = batch.checked_mul(seq_len).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "encode: batch * seq_len",
      "usize",
      [("batch", batch as u64), ("seq_len", seq_len as u64)],
    ))
  })?;
  // The padding id is written into every padded cell, so range-check it once.
  let pad_id = id_to_i32(pad_token_id, "encode: pad_token_id")?;
  let mut id_data: Vec<i32> = try_with_capacity(total)?;
  let mut mask_data: Vec<f32> = try_with_capacity(total)?;
  for (ids, mask) in rows {
    let real = ids.len();
    for &id in ids {
      id_data.push(id_to_i32(id, "encode: token id")?);
    }
    mask_data.extend(mask.iter().map(|&m| f32::from(m)));
    let pad = seq_len - real;
    id_data.extend(std::iter::repeat_n(pad_id, pad));
    mask_data.extend(std::iter::repeat_n(0.0_f32, pad));
  }
  let input_ids = Array::from_slice::<i32>(&id_data, &(batch, seq_len))?;
  let attention_mask = Array::from_slice::<f32>(&mask_data, &(batch, seq_len))?;
  Ok((input_ids, attention_mask, seq_len))
}

/// Build the `(batch, length)` `i32` id + `f32` mask arrays from per-text rows,
/// for the [`Padding::FixedLength`] scheme: pad/truncate every row to the fixed
/// `length` with `pad_token_id` and build an **all-`1`** mask of that length.
///
/// Unlike [`build_dynamic_right_pad`], the pad cells are **not** masked out —
/// the mask is all-`1`. This matches a fixed-processor model whose pooling reads
/// an absolute position regardless of the mask (SigLIP's sticky-EOS text tower):
/// the reference processor right-pads each row to a fixed sequence length and
/// pools the last position whatever it holds, so the pad id is a real, unmasked
/// position.
///
/// **Overlength truncation.** A row longer than `length` is right-truncated
/// (head kept). When `eos_token_id` is `Some(eos)` (a sticky-EOS pooler), the
/// head is kept to `length - 1` and `eos` is forced into the **final** position,
/// so the pooled last slot is the EOS — never a content token. This mirrors the
/// HF SigLIP processor, which head-truncates to `max_length - n_added_tokens`
/// and *then* appends the EOS via its post-processor template (so an overlength
/// prompt always ends in EOS, then right-pads). `None` keeps the plain
/// head-truncation. A row that already fits is unaffected either way (the
/// tokenizer's special-token pass already placed the trailing EOS; the pad cells
/// fill the rest), so a within-length batch is byte-identical regardless of
/// `eos_token_id`. The `length == 0` edge case yields the empty `(batch, 0)`
/// batch and never writes an EOS.
fn build_fixed_length(
  rows: &[(Vec<u32>, Vec<u8>)],
  length: usize,
  pad_token_id: u32,
  eos_token_id: Option<u32>,
) -> Result<(Array, Array, usize)> {
  let batch = rows.len();
  let total = batch.checked_mul(length).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "encode: batch * length",
      "usize",
      [("batch", batch as u64), ("length", length as u64)],
    ))
  })?;
  let pad_id = id_to_i32(pad_token_id, "encode: pad_token_id")?;
  // Range-check the EOS id once if EOS-preserving truncation is enabled (it is
  // written into the final position of every truncated row).
  let eos_id = eos_token_id
    .map(|e| id_to_i32(e, "encode: eos_token_id"))
    .transpose()?;
  let mut id_data: Vec<i32> = try_with_capacity(total)?;
  for (ids, _mask) in rows {
    // Pad or truncate this row to exactly `length`. The per-text encoding may
    // already be capped by `max_length`, but a model can leave that `None` and
    // rely on this fixed truncation, so truncate defensively here too.
    let keep = ids.len().min(length);
    // EOS-preserving truncation: an overlength row whose sticky-EOS pooler must
    // see the EOS at the last position keeps the head to `length - 1` so the
    // forced EOS occupies the final slot (matching HF's truncate-then-append).
    // Only a genuine truncation (`ids.len() > length`) into a non-empty row
    // forces the EOS — a row that already fits is untouched, and `length == 0`
    // writes nothing.
    let force_eos = eos_id.filter(|_| ids.len() > length && length > 0);
    // When forcing the EOS, the head is one shorter so EOS lands at the last
    // slot; the total written is always exactly `length` (head + optional EOS +
    // pad), so the row reaches the fixed length on every path.
    let head = keep - usize::from(force_eos.is_some());
    for &id in &ids[..head] {
      id_data.push(id_to_i32(id, "encode: token id")?);
    }
    if let Some(eos) = force_eos {
      id_data.push(eos);
    }
    let written = head + usize::from(force_eos.is_some());
    id_data.extend(std::iter::repeat_n(pad_id, length - written));
  }
  let input_ids = Array::from_slice::<i32>(&id_data, &(batch, length))?;
  // All-`1` mask: every position (including pad cells) is a real, unmasked
  // position for a fixed-position pooler. Allocate it FALLIBLY (typed
  // [`Error::AllocFailure`] via [`alloc_filled`]) — `length` is model-declared
  // and `batch` is caller-controlled, so a bare `vec![1.0; total]` would abort
  // on allocation pressure instead of returning a recoverable error (an
  // oversized configuration yields a typed error, not a panic), unlike the
  // fallibly-reserved id buffer above.
  let mask_data =
    crate::model_validation::alloc_filled("encode: fixed-length attention mask", 1.0_f32, total)?;
  let attention_mask = Array::from_slice::<f32>(&mask_data, &(batch, length))?;
  Ok((input_ids, attention_mask, length))
}

/// Derive the effective tokenizer truncation cap an `encoding` admits, combining
/// its [`Padding`] scheme's intrinsic cap with any explicit
/// [`max_length`](TextEncoding::max_length).
///
/// This is faithful tokenizer behaviour, not a resource bound: it reduces the
/// encoding's two truncation sources (the padding scheme and the explicit
/// `max_length`) to the single `Option<usize>` cap [`encode_rows`] applies via
/// the tokenizer's own head-truncation. `None` (an uncapped
/// [`Padding::DynamicRightPad`] with no `max_length`) tokenizes at full length.
/// Bounding an oversized / untrusted prompt is the consuming application's
/// responsibility.
///
/// A [`Padding::FixedLength`] scheme is itself a token cap — every row is
/// truncated to `length` — so the tokenizer is capped by the fixed length even
/// when a model leaves `max_length = None`. The intrinsic cap is:
/// - `length + 1` for a **sticky-EOS** fixed length (`eos_token_id` is `Some`):
///   the tokenizer head-truncation keeps `length + 1` content ids (dropping the
///   processor's trailing EOS), then [`build_fixed_length`] sees a row longer
///   than `length` and forces the EOS into the final slot — byte-identical to
///   HF's truncate-then-append-EOS;
/// - `length` for a plain fixed length (`eos_token_id` is `None`): plain
///   head-truncation, no extra slot needed.
///
/// [`Padding::DynamicRightPad`] has no intrinsic token cap, so its effective cap
/// is exactly `encoding.max_length`.
///
/// The intrinsic cap is then combined with any explicit `encoding.max_length` by
/// taking the **tighter** (minimum) of the two — an explicit per-text cap can
/// only ever shrink the cap, never loosen the padding scheme's own truncation.
/// So a [`Padding::FixedLength`] model truncates whether or not it also sets
/// `max_length`, and a model that redundantly sets `Some(length + 1)` derives the
/// same value centrally (behaviour identical).
///
/// **Sticky-EOS floor.** For a sticky-EOS fixed length the combined cap is held
/// at a floor of `length + 1`: an explicit `max_length` may not pull it below
/// that. The `+ 1` slot is exactly what lets the tokenizer keep one id past the
/// fixed `length` so [`build_fixed_length`] sees a row longer than `length` and
/// forces the trailing EOS into the final (pooled) slot. Were the cap allowed to
/// drop to `length` (e.g. an explicit `max_length == Some(length)`), an overlength
/// prompt would be head-truncated to exactly `length` content ids —
/// [`build_fixed_length`] would then see `ids.len() == length` (not `> length`),
/// skip the EOS forcing, and leave a **content** token in the pooled final slot,
/// silently corrupting the embedding. The floor keeps the sticky-EOS guarantee
/// (the truncated prompt's last pooled position is always the EOS) for every
/// configuration. An explicit `max_length` larger than `length + 1` is already
/// absorbed by the `min` (the fixed length is the binding cap); an explicit cap is
/// therefore only ever *additive* for a sticky-EOS model and cannot break EOS
/// preservation.
fn effective_token_cap(encoding: &TextEncoding) -> Option<usize> {
  let intrinsic = match encoding.padding {
    Padding::DynamicRightPad { .. } => None,
    Padding::FixedLength {
      length,
      eos_token_id,
      ..
    } => Some(match eos_token_id {
      // `saturating_add` keeps `length + 1` from overflowing for a pathological
      // `length == usize::MAX`; `build_fixed_length` then handles the absurd
      // `length` with a fallible allocation (a typed error, never a panic).
      Some(_) => length.saturating_add(1),
      None => length,
    }),
  };
  // Combine with the explicit cap by taking the tighter bound. `min` of two
  // `Some`s is the smaller; otherwise whichever side is `Some` (a present cap
  // always wins over an absent one), and `None` only if both are absent.
  let combined = match (intrinsic, encoding.max_length) {
    (Some(a), Some(b)) => Some(a.min(b)),
    (Some(a), None) => Some(a),
    (None, b) => b,
  };
  // Sticky-EOS floor: never let an explicit `max_length` reduce the cap below
  // `length + 1`, so the tokenizer always keeps one id past the fixed length and
  // the trailing EOS is forced into the pooled final slot on a genuine truncation.
  match encoding.padding {
    Padding::FixedLength {
      length,
      eos_token_id: Some(_),
      ..
    } => combined.map(|c| c.max(length.saturating_add(1))),
    _ => combined,
  }
}

/// Tokenize `texts` per `encoding`, then pad them into a `(batch, seq_len)`
/// `i32` id tensor + matching `(batch, seq_len)` `f32` attention mask.
///
/// The tokenizer truncation cap is derived CENTRALLY from the encoding via
/// [`effective_token_cap`] — a `FixedLength` scheme truncates to its own fixed
/// `length` (+1 for sticky-EOS), combined with any explicit
/// [`max_length`](TextEncoding::max_length) by the tighter bound — so a
/// fixed-length model truncates correctly whether or not it also sets
/// `max_length`. The caller's input is tokenized UNMODIFIED and head-truncated to
/// that cap in [`encode_rows`]; bounding an oversized / untrusted prompt is the
/// consuming application's responsibility.
///
/// Dispatches on [`TextEncoding::padding`]:
/// - [`Padding::DynamicRightPad`] — right-pad each row to the batch max length,
///   mask `0` over pad cells (the sentence-encoder default; see
///   [`build_dynamic_right_pad`]).
/// - [`Padding::FixedLength`] — pad/truncate every row to the model-fixed
///   length, all-`1` mask, EOS-preserving overlength truncation when the
///   encoding declares an `eos_token_id` (a fixed-position sticky-EOS pooler
///   like SigLIP; see [`build_fixed_length`]).
///
/// `seq_len` is the resulting sequence length (the batch max for
/// `DynamicRightPad`, the fixed `length` for `FixedLength`). An empty `texts`
/// slice produces correspondingly shaped `(0, seq_len)` arrays.
fn tokenize_and_pad(
  tokenizer: &Tokenizer,
  texts: &[&str],
  encoding: &TextEncoding,
) -> Result<(Array, Array, usize)> {
  // Derive the tokenizer truncation cap from the encoding (the FixedLength scheme,
  // an explicit max_length, or their tighter combination) and apply it as the
  // tokenizer's own head-truncation. The caller's input is tokenized unmodified.
  let cap = effective_token_cap(encoding);
  let rows = encode_rows(tokenizer, texts, encoding.add_special_tokens, cap)?;
  match encoding.padding {
    Padding::DynamicRightPad { pad_token_id } => build_dynamic_right_pad(&rows, pad_token_id),
    Padding::FixedLength {
      length,
      pad_token_id,
      eos_token_id,
    } => build_fixed_length(&rows, length, pad_token_id, eos_token_id),
  }
}

/// Encode a batch of texts into a `(batch, dim)` embedding matrix.
///
/// Pipeline (python `generate` ∘ swift `EmbedderModelContainer.perform`):
/// 1. read the model's [`TextEncoding`](TextEmbedder::text_encoding) — how it
///    tokenizes + pads;
/// 2. tokenize each text (special tokens + truncation per the encoding) and pad
///    them into the `(batch, seq_len)` id tensor + matching attention mask, per
///    the encoding's [`Padding`] scheme;
/// 3. call `model.embed_text(&input_ids, &attention_mask)` — the model applies
///    its own pooling / normalization (a sentence-encoder via its baked
///    [`PoolingConfig`](crate::embeddings::PoolingConfig) inside
///    [`pool_embed`](crate::embeddings::pool_embed); a dual-tower text tower
///    directly).
///
/// The returned array is the model's text embedding, conventionally
/// `(batch, dim)` and L2-normalized. **No implicit eval**: the result is a lazy
/// graph node; the caller evaluates (or reads it) when ready.
///
/// An empty `texts` slice produces a `(0, seq_len)` id batch; the model's
/// `embed_text` returns the correspondingly zero-row embedding.
///
/// - `model` — any [`TextEmbedder`] (reach it from a loaded
///   [`EmbeddingModel`](crate::embeddings::EmbeddingModel) via
///   [`as_text_embedder`](crate::embeddings::EmbeddingModel::as_text_embedder));
/// - `tokenizer` — the loaded [`Tokenizer`] (local-only; no network);
/// - `texts` — the batch to encode.
pub fn encode(model: &dyn TextEmbedder, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Array> {
  // Fail fast on a cleared/poisoned worker thread (and install the mlx-c error
  // handler) before any work, since `embed_text` touches per-thread stream/TLS
  // state. Mirrors the crate's other safe entry points (e.g.
  // `stream::default_stream`), which install the handler before asserting.
  crate::error::ensure_handler_installed();
  crate::stream::assert_streams_not_cleared();
  let encoding = model.text_encoding();
  let (input_ids, attention_mask, _seq_len) = tokenize_and_pad(tokenizer, texts, &encoding)?;
  let embedding = model.embed_text(&input_ids, &attention_mask)?;
  Ok(embedding.into_array())
}

#[cfg(test)]
mod tests;
