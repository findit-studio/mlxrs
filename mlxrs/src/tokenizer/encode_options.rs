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
///         .add_eos(true)
///         .truncate_to(Some(512))
///         .return_attention_mask(true),
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
  pub add_special: bool,
  /// Append the tokenizer's **primary EOS** id unconditionally after
  /// encoding (some LM training-time tokenizers don't include EOS in
  /// `add_special` but consumers want it appended). The primary EOS is
  /// the first caller-supplied EOS id (else the config eos) tracked at
  /// load — NOT the numeric-min of the [`Tokenizer::eos_token_ids`]
  /// `BTreeSet` (which is sorted, so its first element would be the
  /// smallest id, not the intended one). When no primary EOS exists,
  /// [`Tokenizer::encode_with`] returns an error. Defaults to `false`.
  ///
  /// [`Tokenizer::eos_token_ids`]: super::Tokenizer::eos_token_ids
  /// [`Tokenizer::encode_with`]: super::Tokenizer::encode_with
  pub add_eos: bool,
  /// Hard maximum token count. If `Some(n)`, truncate output from the right
  /// to keep at most `n` tokens (matching the HF `tokenizers`
  /// `TruncationDirection::Right` default — keep the head, drop the tail).
  /// `Some(0)` produces an empty result. Defaults to `None` (no truncation).
  pub truncate_to: Option<usize>,
  /// If `true`, also return an attention mask (`Vec<u8>` of `1`s)
  /// alongside the token ids. `encode_with` strips HF padding cells
  /// (`mask == 0`) from the output, so EVERY returned id is a real
  /// attended token — the mask is therefore an all-`1`s vector of length
  /// `ids.len()` synthesized for the returned ids, NOT a downcast of
  /// HF's raw `Encoding::get_attention_mask` (whose `0`s, if any, were
  /// the padding cells already dropped). It exists so callers that
  /// re-pad a batch downstream get a uniform `(ids, mask)` shape per
  /// item. Defaults to `false` — the [`Encoded::attention_mask`] field
  /// is then `None`.
  pub return_attention_mask: bool,
}

impl Default for EncodeOptions {
  fn default() -> Self {
    Self {
      add_special: true,
      add_eos: false,
      truncate_to: None,
      return_attention_mask: false,
    }
  }
}

impl EncodeOptions {
  /// Fluent builder constructor; equivalent to [`EncodeOptions::default`].
  pub fn new() -> Self {
    Self::default()
  }
  /// Set [`EncodeOptions::add_special`].
  pub fn add_special(mut self, v: bool) -> Self {
    self.add_special = v;
    self
  }
  /// Set [`EncodeOptions::add_eos`].
  pub fn add_eos(mut self, v: bool) -> Self {
    self.add_eos = v;
    self
  }
  /// Set [`EncodeOptions::truncate_to`].
  pub fn truncate_to(mut self, v: Option<usize>) -> Self {
    self.truncate_to = v;
    self
  }
  /// Set [`EncodeOptions::return_attention_mask`].
  pub fn return_attention_mask(mut self, v: bool) -> Self {
    self.return_attention_mask = v;
    self
  }
}

/// Result of [`Tokenizer::encode_with`].
///
/// `attention_mask` is `Some` exactly when [`EncodeOptions::return_attention_mask`]
/// was `true`; otherwise `None`. When `Some`, it is an all-`1`s vector
/// with `attention_mask.len() == ids.len()` — `encode_with` drops HF
/// padding cells from `ids`, so every returned id is attended (it is
/// NOT a slice of HF's original mask; see
/// [`EncodeOptions::return_attention_mask`]).
///
/// [`Tokenizer::encode_with`]: super::Tokenizer::encode_with
#[derive(Debug, Clone)]
pub struct Encoded {
  /// Token ids — the same surface as [`Tokenizer::encode`]'s `Vec<u32>` return.
  ///
  /// [`Tokenizer::encode`]: super::Tokenizer::encode
  pub ids: Vec<u32>,
  /// Attention mask (0/1), present iff
  /// [`EncodeOptions::return_attention_mask`] was `true`.
  pub attention_mask: Option<Vec<u8>>,
}
