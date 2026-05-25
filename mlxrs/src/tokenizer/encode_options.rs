//! `EncodeOptions` builder + `Encoded` result for [`Tokenizer::encode_with`].
//!
//! Exposes a richer encoding surface than the short positional
//! [`Tokenizer::encode`]: explicit EOS appending, right-truncation (keep
//! the head, drop the tail), and attention-mask emission. Per the mlxrs
//! LM-2 tracker entry this is an additive Rust-idiom upgrade over the
//! python reference (whose `tokenizer.encode(text, add_special)` is a
//! single positional arg) — the old [`Tokenizer::encode`] API remains
//! unchanged for back-compat.
//!
//! See [`Tokenizer::encode_with`] for usage.
//!
//! [`Tokenizer::encode`]: super::Tokenizer::encode
//! [`Tokenizer::encode_with`]: super::Tokenizer::encode_with

/// Encoding options for [`Tokenizer::encode_with`].
///
/// Only [`add_special`](Self::add_special) maps directly to an HF
/// `tokenizers` flag (`add_special_tokens`). [`add_eos`](Self::add_eos)
/// and [`truncate_to`](Self::truncate_to) are mlxrs-side
/// post-processing on the HF output, and
/// [`return_attention_mask`](Self::return_attention_mask) returns a
/// mask synthesized for the post-strip / post-truncation ids (NOT HF's
/// raw mask — see that field). The mlxrs [`Tokenizer::encode`]
/// (positional `add_special: bool`) remains available unchanged for
/// back-compat; new callers should prefer [`Tokenizer::encode_with`].
///
/// Construct via [`EncodeOptions::default`] / [`EncodeOptions::new`] and
/// chain the fluent setters.
///
/// ```ignore
/// use mlxrs::tokenizer::{EncodeOptions, Tokenizer};
///
/// let tok = Tokenizer::from_path("model/", None)?;
/// let out = tok.encode_with(
///     "hello world",
///     &EncodeOptions::new()
///         .with_add_eos(true)
///         .with_truncate_to(Some(512))
///         .with_return_attention_mask(true),
/// )?;
/// ```
///
/// [`Tokenizer::encode`]: super::Tokenizer::encode
/// [`Tokenizer::encode_with`]: super::Tokenizer::encode_with
#[derive(Debug, Clone)]
pub struct EncodeOptions {
  /// Add the tokenizer's special tokens (BOS/EOS/sep) when encoding, exactly
  /// as the HF `tokenizers` crate's `add_special_tokens` flag does. Defaults
  /// to `true` — matches the existing `encode(..., true)` call path.
  add_special: bool,
  /// Append the tokenizer's **primary EOS** id unconditionally after
  /// encoding (some LM training-time tokenizers don't include EOS in
  /// `add_special` but consumers want it appended). The primary EOS is
  /// the first caller-supplied EOS id (else the config eos) tracked at
  /// load — NOT the numeric-min of the [`Tokenizer::eos_token_ids_iter`]
  /// set (which is sorted, so its first element would be the
  /// smallest id, not the intended one). When no primary EOS exists,
  /// [`Tokenizer::encode_with`] returns an error. Defaults to `false`.
  ///
  /// [`Tokenizer::eos_token_ids_iter`]: super::Tokenizer::eos_token_ids_iter
  /// [`Tokenizer::encode_with`]: super::Tokenizer::encode_with
  add_eos: bool,
  /// Hard maximum token count. If `Some(n)`, truncate output from the right
  /// to keep at most `n` tokens (matching the HF `tokenizers`
  /// `TruncationDirection::Right` default — keep the head, drop the tail).
  /// `Some(0)` produces an empty result. Defaults to `None` (no truncation).
  truncate_to: Option<usize>,
  /// If `true`, also return an attention mask (`Vec<u8>` of `1`s)
  /// alongside the token ids. `encode_with` strips HF padding cells
  /// (`mask == 0`) from the output, so EVERY returned id is a real
  /// attended token — the mask is therefore an all-`1`s vector of length
  /// `ids.len()` synthesized for the returned ids, NOT a downcast of
  /// HF's raw `Encoding::get_attention_mask` (whose `0`s, if any, were
  /// the padding cells already dropped). It exists so callers that
  /// re-pad a batch downstream get a uniform `(ids, mask)` shape per
  /// item. Defaults to `false` — the [`Encoded::attention_mask`] slice
  /// is then empty.
  return_attention_mask: bool,
}

impl Default for EncodeOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl EncodeOptions {
  /// Construct with defaults: `add_special=true`, `add_eos=false`,
  /// `truncate_to=None`, `return_attention_mask=false`.
  pub const fn new() -> Self {
    Self {
      add_special: true,
      add_eos: false,
      truncate_to: None,
      return_attention_mask: false,
    }
  }

  // --- accessors ---

  /// Whether to add special tokens when encoding.
  #[inline(always)]
  pub fn add_special(&self) -> bool {
    self.add_special
  }

  /// Whether to append the primary EOS id after encoding.
  #[inline(always)]
  pub fn add_eos(&self) -> bool {
    self.add_eos
  }

  /// Hard token-count cap (`None` = no truncation).
  #[inline(always)]
  pub fn truncate_to(&self) -> Option<usize> {
    self.truncate_to
  }

  /// Whether to synthesize an all-`1`s attention mask alongside the ids.
  #[inline(always)]
  pub fn return_attention_mask(&self) -> bool {
    self.return_attention_mask
  }

  // --- chaining builders ---

  /// Set whether to add special tokens. Returns `self` for chaining.
  #[must_use]
  pub fn with_add_special(mut self, v: bool) -> Self {
    self.add_special = v;
    self
  }

  /// Set whether to append the primary EOS id. Returns `self` for chaining.
  #[must_use]
  pub fn with_add_eos(mut self, v: bool) -> Self {
    self.add_eos = v;
    self
  }

  /// Set the hard token-count cap. Returns `self` for chaining.
  #[must_use]
  pub fn with_truncate_to(mut self, v: Option<usize>) -> Self {
    self.truncate_to = v;
    self
  }

  /// Set whether to return an attention mask. Returns `self` for chaining.
  #[must_use]
  pub fn with_return_attention_mask(mut self, v: bool) -> Self {
    self.return_attention_mask = v;
    self
  }
}

/// Result of [`Tokenizer::encode_with`].
///
/// **Length-equality invariant** (the runtime contract):
/// - If [`EncodeOptions::return_attention_mask`] was `false`, `attention_mask`
///   is empty (callers that did not request a mask get no allocation).
/// - If `return_attention_mask` was `true`, `attention_mask.len() ==
///   ids.len()` — including the legitimate `(0, 0)` case for zero-length
///   encodings (`with_truncate_to(Some(0))`, empty input, etc.). The mask is
///   an all-`1`s slice; `encode_with` drops HF padding cells from `ids`, so
///   every returned id is attended (it is NOT a slice of HF's original mask;
///   see [`EncodeOptions::return_attention_mask`]).
///
/// **Do not use `attention_mask.is_empty()` as a "mask not requested"
/// sentinel.** An empty `attention_mask` is ambiguous: it can mean either
/// "not requested" OR "requested but the encoding was zero-length".
/// Consumers that requested a mask must validate `attention_mask().len()
/// == ids().len()` (which accepts `(0, 0)`) rather than test `is_empty()`.
/// The presence of the request itself is known from the caller's
/// [`EncodeOptions`], not from the result.
///
/// [`Tokenizer::encode_with`]: super::Tokenizer::encode_with
#[derive(Debug, Clone)]
pub struct Encoded {
  /// Token ids — the same surface as [`Tokenizer::encode`]'s `Vec<u32>` return.
  ///
  /// [`Tokenizer::encode`]: super::Tokenizer::encode
  ids: Vec<u32>,
  /// Attention mask (0/1). Length contract:
  /// - empty if `return_attention_mask` was `false`
  /// - `len() == ids.len()` if `return_attention_mask` was `true`
  ///   (including the legitimate `(0, 0)` zero-length encoding)
  ///
  /// `is_empty()` is NOT a reliable "not requested" check (see struct
  /// docs); use [`EncodeOptions`] presence + length-equality instead.
  attention_mask: Vec<u8>,
}

impl Encoded {
  /// Construct from pre-built ids and attention mask.
  pub fn new(ids: Vec<u32>, attention_mask: Vec<u8>) -> Self {
    Self {
      ids,
      attention_mask,
    }
  }

  /// Token ids (same surface as [`Tokenizer::encode`]).
  ///
  /// [`Tokenizer::encode`]: super::Tokenizer::encode
  #[inline(always)]
  pub fn ids(&self) -> &[u32] {
    &self.ids
  }

  /// Attention mask (0/1). Length contract:
  /// - empty slice if [`EncodeOptions::return_attention_mask`] was `false`;
  /// - `len() == ids().len()` if `return_attention_mask` was `true` — which
  ///   **may itself be empty** for zero-length encodings
  ///   (`with_truncate_to(Some(0))`, empty input, etc.).
  ///
  /// `is_empty()` is therefore ambiguous and is NOT a reliable "not
  /// requested" check. Consumers that requested a mask should compare
  /// `attention_mask().len() == ids().len()` (which accepts `(0, 0)`);
  /// presence of the request itself lives on the caller's
  /// [`EncodeOptions`], not on the result.
  #[inline(always)]
  pub fn attention_mask(&self) -> &[u8] {
    &self.attention_mask
  }

  /// Move out the owned `(ids, attention_mask)` vectors. Use this on hot
  /// paths (e.g. batched tokenization → MLX arrays) to avoid the
  /// `to_vec()` clone of two `Vec<_>` per row that borrowed-accessor +
  /// owned-flatten would otherwise pay. `std::io::Error`-like
  /// owned-move-out applies: the `Encoded` is consumed.
  #[inline(always)]
  pub fn into_parts(self) -> (Vec<u32>, Vec<u8>) {
    (self.ids, self.attention_mask)
  }
}
