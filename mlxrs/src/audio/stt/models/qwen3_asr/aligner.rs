//! The Qwen3 forced aligner (`qwen3_forced_aligner.ForcedAlignerModel`).
//!
//! Timestamp **classification**, not CTC: the Qwen3-ASR decoder with the vocab
//! LM head replaced by a `Linear(hidden -> classify_num, bias=False)`
//! timestamp head. The forward embeds the token ids, splices the audio-encoder
//! features into the `<audio_pad>` positions, runs the head-less
//! [`Qwen3AsrTextModel`](super::Qwen3AsrTextModel) decoder, and applies the
//! timestamp head to the normalized hidden states, producing
//! `(batch, seq, classify_num)` logits.
//!
//! The decode reads, at every `<timestamp>` input position, the argmax class
//! id and multiplies it by
//! [`timestamp_segment_time`](ForcedAlignerConfig::timestamp_segment_time)
//! (milliseconds) to get a boundary time. Words sit between paired
//! `<timestamp><timestamp>` markers (even index = start, odd index = end); a
//! Longest-Increasing-Subsequence repair ([`fix_timestamp`]) fixes any
//! non-monotone run before the per-word `(text, start_s, end_s)` spans are
//! emitted.
//!
//! The aligner reuses the shared [`AudioEncoder`](super::AudioEncoder) audio
//! tower (the same one the Qwen3-ASR transcriber uses) and the
//! [`Qwen3AsrTextModel`](super::Qwen3AsrTextModel) decoder, mirroring the
//! reference's `AudioEncoder` + `TextModel` reuse across `Qwen3ASRModel` and
//! `ForcedAlignerModel`. The ASR text decoder is its **own** type (rather than
//! the dense Qwen3 LM decoder) because the released Qwen3-ASR `text_config`
//! carries a non-null MRoPE `rope_scaling` the dense Qwen3 config rejects.

use std::collections::HashMap;

use crate::{
  array::Array,
  audio::stt::model::{
    AlignOptions, AlignedSpan, ForcedAligner as ForcedAlignerTrait, ForcedAlignment,
  },
  error::{
    AllocFailurePayload, Error, LengthMismatchPayload, MissingKeyPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload,
  },
  lm::cache::KvCache,
  model_validation::{alloc_filled, reserve_or_error},
  ops::{
    indexing::take_axis,
    linalg_basic::matmul,
    misc::argmax,
    shape::{concatenate, reshape, swapaxes},
  },
  tokenizer::Tokenizer,
};

use super::{aligner_config::ForcedAlignerConfig, audio::AudioEncoder, text::Qwen3AsrTextModel};

/// The `<|audio_start|>` marker string the aligner prepends to the assembled
/// transcript before tokenization (the reference's literal in
/// `ForceAlignProcessor.encode_timestamp`).
const AUDIO_START_MARKER: &str = "<|audio_start|>";
/// The `<|audio_pad|>` placeholder string, emitted once per audio token (the
/// reference expands a single `<|audio_pad|>` to `num_audio_tokens` copies).
const AUDIO_PAD_MARKER: &str = "<|audio_pad|>";
/// The `<|audio_end|>` marker string closing the audio span.
const AUDIO_END_MARKER: &str = "<|audio_end|>";
/// The `<timestamp>` marker string; a pair is appended after each word so the
/// tokenized sequence carries two `timestamp_token_id` positions per word.
const TIMESTAMP_MARKER: &str = "<timestamp>";

/// The **primary** forced-alignment input: a raw transcript `text` plus the
/// `language` it is in.
///
/// The [`RawTranscript`] [`ForcedAligner`](ForcedAlignerTrait) impl owns the
/// tokenization — it splits `text` into words per `language` and encodes the
/// assembled marker string with the model's own tokenizer (faithful to
/// `qwen3_forced_aligner.ForceAlignProcessor.encode_timestamp` +
/// `self._tokenizer.encode`). The caller passes only raw text, so there is no
/// caller-side, possibly-wrong tokenization. `language` selects the word
/// splitter (matched case-insensitively, mirroring the reference's
/// `language.lower()`).
///
/// This raw-text path is reference-faithful for the languages the reference
/// segments **inline**: whitespace-segmented languages and CJK-per-character
/// languages (Chinese included) split on whitespace with embedded CJK broken out
/// per character. It is **not** faithful for Japanese and Korean, which the
/// reference delegates to **optional external** morphological segmenters
/// (`nagisa` / `soynlp`) this port does not bundle; a Japanese or Korean
/// [`RawTranscript`] raises a typed [`Error::Tokenizer`] pointing to the
/// pre-tokenized path. Align Japanese/Korean transcripts via
/// [`PreTokenizedTranscript`], where the caller supplies the tokenization (the
/// scope decision is tracked in issue #322).
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTranscript {
  /// The transcript text to align.
  text: String,
  /// The transcript language (selects the word splitter; matched
  /// case-insensitively).
  language: String,
}

impl RawTranscript {
  /// Construct a [`RawTranscript`] from its text and language.
  #[inline]
  pub fn new(text: impl Into<String>, language: impl Into<String>) -> Self {
    Self {
      text: text.into(),
      language: language.into(),
    }
  }

  /// Construct an English [`RawTranscript`] (the reference's default language),
  /// using whitespace word splitting.
  #[inline]
  pub fn english(text: impl Into<String>) -> Self {
    Self::new(text, "English")
  }

  /// The transcript text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The transcript language.
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }
}

/// One transcript word for the **pre-tokenized** alignment path: its display
/// `text` (used to label the returned span) and its decoder token ids
/// (`token_ids`).
///
/// This is the input unit for [`PreTokenizedTranscript`]. The caller has already
/// performed the language-specific word splitting and subword encoding; the
/// aligner interleaves the `<timestamp><timestamp>` boundary markers between
/// these words and wraps the sequence with the audio span markers. Prefer
/// [`RawTranscript`] (the aligner tokenizes internally) unless the caller
/// genuinely already holds the model's subword ids.
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignWord {
  /// The word's display text — copied verbatim into the output span's text.
  text: String,
  /// The word's decoder token ids (the caller's subword tokenization).
  token_ids: Vec<i32>,
}

impl AlignWord {
  /// Construct an [`AlignWord`] from its display text and token ids.
  #[inline(always)]
  pub fn new(text: impl Into<String>, token_ids: Vec<i32>) -> Self {
    Self {
      text: text.into(),
      token_ids,
    }
  }

  /// The word's display text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The word's decoder token ids.
  #[inline(always)]
  pub fn token_ids(&self) -> &[i32] {
    &self.token_ids
  }
}

/// The **pre-tokenized** forced-alignment input: a borrowed slice of
/// already-tokenized [`AlignWord`]s, in transcript order.
///
/// The secondary [`ForcedAligner`](ForcedAlignerTrait) input — used when the
/// caller already holds the model's per-word subword ids and wants to skip the
/// internal tokenization (and the tokenizer dependency). A borrowing newtype, so
/// it carries no allocation of its own. Most callers should prefer
/// [`RawTranscript`].
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
#[derive(Debug, Clone, Copy)]
pub struct PreTokenizedTranscript<'a> {
  words: &'a [AlignWord],
}

impl<'a> PreTokenizedTranscript<'a> {
  /// Wrap a slice of pre-tokenized words as a [`PreTokenizedTranscript`].
  #[inline(always)]
  pub fn new(words: &'a [AlignWord]) -> Self {
    Self { words }
  }

  /// The pre-tokenized words, in transcript order.
  #[inline(always)]
  pub fn words(&self) -> &'a [AlignWord] {
    self.words
  }
}

impl<'a> From<&'a [AlignWord]> for PreTokenizedTranscript<'a> {
  #[inline(always)]
  fn from(words: &'a [AlignWord]) -> Self {
    Self::new(words)
  }
}

/// The Qwen3 forced aligner: the audio tower, the head-less decoder, and the
/// timestamp-classification head.
///
/// It implements [`ForcedAligner`](ForcedAlignerTrait) for two transcript
/// inputs: [`RawTranscript`] (the primary, reference-faithful path — the aligner
/// splits the raw text into words and tokenizes the assembled marker string
/// with its own [`Tokenizer`] internally) and [`PreTokenizedTranscript`] (an
/// already-tokenized word sequence, which needs no tokenizer). A tokenizer is
/// supplied via [`from_weights_with_tokenizer`](Self::from_weights_with_tokenizer)
/// or [`with_tokenizer`](Self::with_tokenizer) and is required only for the
/// [`RawTranscript`] path.
#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
pub struct ForcedAligner {
  config: ForcedAlignerConfig,
  audio_tower: AudioEncoder,
  model: Qwen3AsrTextModel,
  /// Bias-free timestamp head weight `(classify_num, hidden)`. Applied as
  /// `hidden @ weightᵀ` (an `nn.Linear(hidden -> classify_num)` forward).
  timestamp_head_weight: Array,
  /// The model's own tokenizer, used by the [`RawTranscript`] align path to
  /// encode the assembled `<|audio_start|>…<timestamp><timestamp>` marker
  /// string (the reference's `self._tokenizer.encode(..,
  /// add_special_tokens=False)`). `None` until supplied; the raw-text path then
  /// errors with a typed [`Error::Tokenizer`], while the
  /// [`PreTokenizedTranscript`] path never consults it.
  tokenizer: Option<Tokenizer>,
}

// `Tokenizer` does not implement `Debug`, so the derive cannot apply; report the
// tokenizer's presence (not its contents) and forward the rest.
impl std::fmt::Debug for ForcedAligner {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ForcedAligner")
      .field("config", &self.config)
      .field("audio_tower", &self.audio_tower)
      .field("model", &self.model)
      .field("timestamp_head_weight", &self.timestamp_head_weight)
      .field("tokenizer", &self.tokenizer.is_some())
      .finish()
  }
}

#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
impl ForcedAligner {
  /// The model configuration.
  #[inline(always)]
  pub fn config(&self) -> &ForcedAlignerConfig {
    &self.config
  }

  /// Read-only view of the audio tower.
  #[inline(always)]
  pub fn audio_tower(&self) -> &AudioEncoder {
    &self.audio_tower
  }

  /// Read-only view of the head-less decoder.
  #[inline(always)]
  pub fn decoder(&self) -> &Qwen3AsrTextModel {
    &self.model
  }

  /// Build a [`ForcedAligner`] from a parsed [`ForcedAlignerConfig`] and a flat
  /// name → [`Array`] weight map.
  ///
  /// The map carries three groups, each drained by its sub-builder:
  /// `audio_tower.*` (the audio tower — its keys are pre-stripped to the
  /// submodule layout by [`super::sanitize`]), `model.*` (the Qwen3 decoder),
  /// and `lm_head.weight` (the `(classify_num, hidden)` timestamp head). A
  /// missing required weight is an [`Error::MissingKey`]; a head of the wrong
  /// shape is an [`Error::ShapePairMismatch`].
  ///
  /// `audio_weights` is the already-[`sanitize`](super::sanitize)d audio-tower
  /// map (conv weights transposed to channels-last, the `audio_tower.` prefix
  /// stripped); `decoder_weights` carries the decoder `model.*` keys plus
  /// `lm_head.weight`. They are kept separate because the audio sanitize drops
  /// every non-audio key, so a single combined map cannot feed both builders.
  pub fn from_weights(
    config: ForcedAlignerConfig,
    audio_weights: HashMap<String, Array>,
    mut decoder_weights: HashMap<String, Array>,
  ) -> Result<Self> {
    config.validate()?;

    let audio_tower = AudioEncoder::from_weights(config.audio_config.clone(), audio_weights)?;
    let model = Qwen3AsrTextModel::from_weights(&config.text_config, &mut decoder_weights)?;

    // The timestamp head is `Linear(hidden -> classify_num, bias=False)`, so
    // the weight is `(classify_num, hidden)`.
    let hidden = config.text_config.hidden_size;
    let classify_num = config.classify_num;
    let timestamp_head_weight = decoder_weights.remove("lm_head.weight").ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "ForcedAligner weight map",
        smol_str::format_smolstr!("lm_head.weight"),
      ))
    })?;
    let want = vec![classify_num.max(0) as usize, hidden.max(0) as usize];
    let got = timestamp_head_weight.shape();
    if got != want {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "ForcedAligner: lm_head.weight (classify_num, hidden)",
        want,
        got,
      )));
    }

    Ok(Self {
      config,
      audio_tower,
      model,
      timestamp_head_weight,
      tokenizer: None,
    })
  }

  /// Build a [`ForcedAligner`] as [`from_weights`](Self::from_weights), with the
  /// model's [`Tokenizer`] attached so the [`RawTranscript`] align path can
  /// split + encode raw transcript text internally.
  ///
  /// The tokenizer must be the **model's own** tokenizer (the one whose vocab
  /// carries the `<|audio_start|>` / `<|audio_pad|>` / `<|audio_end|>` /
  /// `<timestamp>` markers as single ids matching the config token ids);
  /// supplying it here is what removes the caller-side, wrong-tokenizer hazard.
  pub fn from_weights_with_tokenizer(
    config: ForcedAlignerConfig,
    audio_weights: HashMap<String, Array>,
    decoder_weights: HashMap<String, Array>,
    tokenizer: Tokenizer,
  ) -> Result<Self> {
    Ok(Self::from_weights(config, audio_weights, decoder_weights)?.with_tokenizer(tokenizer))
  }

  /// Attach (or replace) the model tokenizer used by the [`RawTranscript`] align
  /// path, returning `self` for chaining.
  #[must_use]
  #[inline]
  pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
    self.tokenizer = Some(tokenizer);
    self
  }

  /// The attached model tokenizer, or `None` when only the
  /// [`PreTokenizedTranscript`] path is available.
  #[inline(always)]
  pub fn tokenizer(&self) -> Option<&Tokenizer> {
    self.tokenizer.as_ref()
  }

  /// Apply the bias-free timestamp head: `(B, L, hidden) @ weightᵀ` →
  /// `(B, L, classify_num)`.
  fn timestamp_head(&self, hidden: &Array) -> Result<Array> {
    let wt = swapaxes(&self.timestamp_head_weight, -1, -2)?;
    matmul(hidden, &wt)
  }

  /// Forward pass: token ids + audio mel features → timestamp logits
  /// `(batch, seq, classify_num)`, treating the whole `time` axis of
  /// `input_features` as valid (an unpadded utterance).
  ///
  /// `input_ids` is `(batch, seq)` integer ids (built so that exactly the
  /// audio-encoder output length of positions equal `audio_token_id`);
  /// `input_features` is the encoder's mel input `(batch, n_mels, time)`. The
  /// audio-encoder output rows are spliced into the `<audio_pad>` positions of
  /// the token embeddings (one row per placeholder, in flat order), then the
  /// head-less decoder and the timestamp head run. The `<audio_pad>` count must
  /// equal the audio-encoder row count, else an [`Error::LengthMismatch`]. No
  /// implicit eval — the returned [`Array`] is lazy.
  ///
  /// For a padded mel input, supply the valid mel-frame count via
  /// [`forward_with_feature_length`](Self::forward_with_feature_length) so the
  /// padding is trimmed before encoding.
  pub fn forward(&self, input_ids: &Array, input_features: &Array) -> Result<Array> {
    self.forward_with_feature_length(input_ids, input_features, None)
  }

  /// As [`forward`](Self::forward), but with the per-utterance valid mel-frame
  /// count `feature_length` (the reference's `feature_attention_mask.sum(-1)`).
  ///
  /// The audio tower trims the padded trailing mel frames to `feature_length`
  /// before encoding, so padded positions contribute neither audio tokens nor
  /// attention. `None` treats the whole `time` axis as valid.
  ///
  /// The aligner is a single-utterance (batch 1) path. An `input_ids` batch
  /// other than 1 is rejected here with [`Error::OutOfRange`]; a multi-window
  /// utterance (valid frames exceeding one conv chunk) or a batched
  /// `input_features` is rejected by the audio tower, until the chunked windowed
  /// attention path is ported.
  pub fn forward_with_feature_length(
    &self,
    input_ids: &Array,
    input_features: &Array,
    feature_length: Option<i64>,
  ) -> Result<Array> {
    let (b, l) = self.input_ids_shape(input_ids)?;
    // Single-utterance path only. The audio tower rejects an `input_features`
    // batch other than 1; reject an `input_ids` batch other than 1 here too, so
    // a `(B > 1, L)` token tensor paired with a single audio feature tensor
    // cannot reach the splice — where the audio pads are matched in flat order
    // across all batch rows — and silently produce mis-spliced batched logits.
    if b != 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::forward_with_feature_length: input_ids batch size",
        "only single-utterance (batch == 1) is supported; the windowed batch path is not yet ported",
        b.to_string(),
      )));
    }
    // The flat position count `batch * seq` sizes the host-side splice gather
    // index (fallibly reserved below). `b` and `l` are the dims of the existing
    // `input_ids` tensor, so the product already fits `usize`; compute it with
    // `checked_mul` so a hypothetical wrap surfaces as a typed error rather than
    // a panic.
    let seq_elems = b.checked_mul(l).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::forward: batch * seq",
        "overflows usize",
        format!("{b} * {l}"),
      ))
    })?;

    // Reject any `input_ids` value outside `[0, vocab_size)` BEFORE `embed_tokens`
    // gathers the embedding rows. MLX `take` does not bound-check its indices:
    // a negative id selects `id + vocab` (a wrong row, or a further out-of-bounds
    // read) and an id `>= vocab` reads past the embedding table — an out-of-bounds
    // read, i.e. UB. Reading `input_ids` is an explicit materialization of an
    // input tensor (already backed by data), not a hidden eval of a lazy graph.
    self.validate_input_ids_range(input_ids)?;

    let embeds = self.model.embed_tokens(input_ids)?;
    let hidden_dim = self.decoder_hidden_dim(&embeds)?;

    let lengths = feature_length.map(|len| [len]);
    let audio_features = self
      .audio_tower
      .forward_single_window(input_features, lengths.as_ref().map(|l| l.as_slice()))?;
    let spliced = self.splice_audio(input_ids, &embeds, &audio_features, seq_elems, hidden_dim)?;

    let mut cache: Vec<Box<dyn KvCache>> = self.model.make_cache();
    let hidden = self.model.forward_hidden(&spliced, &mut cache)?;
    self.timestamp_head(&hidden)
  }

  /// Splice the audio-encoder feature rows into the `<audio_pad>` positions of
  /// the token embeddings, returning the modified `(B, L, hidden)` embeddings.
  ///
  /// Flattens the embeddings to `(B*L, hidden)`, finds the flat positions equal
  /// to `audio_token_id`, and overwrites them (in flat order) with successive
  /// audio rows — one row per placeholder. The placeholder count must equal the
  /// audio-encoder row count ([`Error::LengthMismatch`] otherwise), so the whole
  /// audio span is spliced rather than partially merged. Implemented as a single
  /// gather: `combined = concat([flat embeds, audio rows])`, then a host-built
  /// index selects, per flat position, either its own embedding row or its audio
  /// row.
  fn splice_audio(
    &self,
    input_ids: &Array,
    embeds: &Array,
    audio_features: &Array,
    seq_elems: usize,
    hidden_dim: usize,
  ) -> Result<Array> {
    // Flatten audio features to `(num_audio_rows, hidden)`. The encoder output
    // is `(batch, time', output_dim)`; its `output_dim` must match the decoder
    // hidden width (the splice replaces a hidden-width embedding row).
    let audio_rows_total = audio_features.size();
    let audio_hidden = *audio_features.shape().last().unwrap_or(&0);
    if audio_hidden != hidden_dim {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "ForcedAligner: audio encoder output_dim vs decoder hidden_size",
        vec![hidden_dim],
        vec![audio_hidden],
      )));
    }
    let num_audio_rows = audio_rows_total.checked_div(audio_hidden).unwrap_or(0);
    let audio_hidden_i = i32::try_from(audio_hidden).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner: audio hidden",
        "exceeds i32::MAX",
        audio_hidden.to_string(),
      ))
    })?;
    // The gather concatenates the `seq_elems` embedding rows and the
    // `num_audio_rows` audio rows into one `combined` of `seq_elems +
    // num_audio_rows` rows, then indexes it with `i32` positions. MLX `take`
    // does NOT bound-check its indices, so the combined row count must fit in
    // `i32` for every index (up to `seq_elems + num_audio_rows - 1`) to be a
    // valid in-range row — checked before the gather is built so an overflow is
    // a typed error rather than a wrapped (negative / out-of-range) index that
    // would feed an out-of-bounds row read (UB) to `take_axis`.
    let combined_rows = seq_elems.checked_add(num_audio_rows).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::splice_audio: batch*seq + audio rows",
        "overflows usize",
        format!("seq_elems={seq_elems}, audio_rows={num_audio_rows}"),
      ))
    })?;
    i32::try_from(combined_rows).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::splice_audio: batch*seq + audio rows",
        "exceeds i32::MAX",
        combined_rows.to_string(),
      ))
    })?;
    let num_audio_rows_i = i32::try_from(num_audio_rows).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner: num audio rows",
        "exceeds i32::MAX",
        num_audio_rows.to_string(),
      ))
    })?;
    let flat_audio = reshape(audio_features, &[num_audio_rows_i, audio_hidden_i])?;

    // Flat embeddings `(B*L, hidden)`.
    let seq_elems_i = i32::try_from(seq_elems).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner: batch*seq",
        "exceeds i32::MAX",
        seq_elems.to_string(),
      ))
    })?;
    let hidden_i = audio_hidden_i;
    let flat_embeds = reshape(embeds, &[seq_elems_i, hidden_i])?;

    // Build the gather index host-side from the audio-pad mask. Reading
    // `input_ids` is an explicit materialization of an input tensor (already
    // backed by data), not a hidden eval of a lazy compute graph.
    let mut ids = input_ids.try_clone()?;
    let ids_flat: Vec<i32> = ids.to_vec::<i32>()?;
    debug_assert_eq!(ids_flat.len(), seq_elems);

    // The splice merges exactly one audio feature row per `<audio_pad>`
    // placeholder, so the count of `audio_token_id` flat positions must equal
    // the audio-encoder row count. A mismatch (a wrong feature length, or a
    // caller-built `input_ids` whose pad run does not match the encoder output)
    // would otherwise leave part of the audio span unspliced — plausible-looking
    // logits with a silently wrong alignment — so it is rejected here rather
    // than truncated to `min(num_pad, num_audio_rows)`.
    let audio_token_id = self.config.audio_token_id;
    let num_pad = ids_flat
      .iter()
      .filter(|&&id| i64::from(id) == audio_token_id)
      .count();
    if num_pad != num_audio_rows {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "ForcedAligner::splice_audio: <audio_pad> placeholder count vs audio encoder rows",
        num_audio_rows,
        num_pad,
      )));
    }

    // `gather[i]` defaults to `i` (its own embed row); the k-th audio-pad flat
    // position maps to `seq_elems + k` (the k-th appended audio row). The pad
    // count equals `num_audio_rows` (checked above), so every pad is mapped.
    // Every index is built with checked arithmetic so none can wrap past
    // `i32::MAX` into a negative / out-of-range row.
    let gather = splice_gather_index(&ids_flat, audio_token_id, seq_elems_i)?;

    let combined = concatenate(&[&flat_embeds, &flat_audio], 0)?;
    let idx = Array::from_slice::<i32>(&gather, &(seq_elems,))?;
    let spliced_flat = take_axis(&combined, &idx, 0)?;

    let (b, l) = self.input_ids_shape(input_ids)?;
    reshape(
      &spliced_flat,
      &[
        i32::try_from(b).unwrap_or(i32::MAX),
        i32::try_from(l).unwrap_or(i32::MAX),
        hidden_i,
      ],
    )
  }

  /// Decode timestamp logits into per-word spans for a pre-tokenized
  /// transcript.
  ///
  /// `input_ids` is the `(batch, seq)` (or `(seq,)`) token id tensor the
  /// forward ran on; `logits` is its `(batch, seq, classify_num)` (or `(seq,
  /// classify_num)`) output; `transcript` is the words in order. Reads, at
  /// every `<timestamp>` input position (batch row 0), the argmax class and
  /// converts `class * timestamp_segment_time` (ms) to a boundary time; the
  /// LIS repair fixes non-monotone runs; word `i` takes `(fixed[2i],
  /// fixed[2i+1])` as `(start, end)` (ms → seconds, rounded to ms). A
  /// `<timestamp>`-marker count other than `2 * transcript.len()` is an
  /// [`Error::LengthMismatch`].
  pub fn decode_alignment(
    &self,
    input_ids: &Array,
    logits: &Array,
    transcript: &[AlignWord],
    language: Option<String>,
  ) -> Result<ForcedAlignment> {
    let times = self.decode_marker_times(input_ids, logits, transcript.len())?;
    let mut spans: Vec<AlignedSpan> = Vec::new();
    reserve_or_error(&mut spans, "ForcedAligner: aligned spans", transcript.len())?;
    for (word, (start_s, end_s)) in transcript.iter().zip(times) {
      spans.push(AlignedSpan::new(word.text(), start_s, end_s));
    }
    Ok(ForcedAlignment::new(spans, language))
  }

  /// The per-word `(start_seconds, end_seconds)` boundary pairs decoded from the
  /// timestamp logits — the label-agnostic core shared by both align paths.
  ///
  /// Reads the argmax class at every `<timestamp>` input position (batch row 0),
  /// scales each by [`timestamp_segment_time`](ForcedAlignerConfig::timestamp_segment_time)
  /// (ms), repairs the marker sequence with the LIS monotonicity fix, and
  /// returns word `i`'s `(fixed[2i], fixed[2i+1])` as seconds (rounded to ms).
  /// A `<timestamp>`-marker count other than `2 * num_words` is an
  /// [`Error::LengthMismatch`]; the caller pairs each returned span time with
  /// its own display label.
  fn decode_marker_times(
    &self,
    input_ids: &Array,
    logits: &Array,
    num_words: usize,
  ) -> Result<Vec<(f64, f64)>> {
    // Validate the input_ids/logits rank + exact shape relation BEFORE decoding,
    // so a rank-2 `input_ids` whose row 0 is shorter than the `logits` sequence
    // cannot make the row-0 slice cross into another batch row (a plausible-but-
    // wrong span). The forward path enforces batch == 1; the decode does too.
    //
    // `input_ids` is rank-1 `(seq,)` or rank-2 `(batch, seq)`; `logits` is
    // rank-2 `(seq, classify_num)` or rank-3 `(batch, seq, classify_num)`. The
    // batch must be 1, the `input_ids` row (`seq`) length must equal the
    // `logits` sequence length, and the `logits` last dim must equal
    // `classify_num`.
    let (ids_batch, ids_seq) = self.input_ids_shape(input_ids)?;
    let (logits_batch, logits_seq, logits_classes) = self.decode_logits_shape(logits)?;
    if ids_batch != 1 || logits_batch != 1 {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::decode: batch size",
        "only single-utterance (batch == 1) is supported",
        format!("input_ids batch {ids_batch}, logits batch {logits_batch}"),
      )));
    }
    if ids_seq != logits_seq {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "ForcedAligner::decode: input_ids row length vs logits seq",
        vec![logits_seq],
        vec![ids_seq],
      )));
    }
    let classify_num = usize::try_from(self.config.classify_num.max(0)).unwrap_or(usize::MAX);
    if logits_classes != classify_num {
      return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
        "ForcedAligner::decode: logits last dim vs classify_num",
        vec![classify_num],
        vec![logits_classes],
      )));
    }

    // argmax over the last (classify_num) axis → class id per position.
    let class_axis = (logits.ndim() as i32) - 1;
    let mut output_ids = argmax(logits, Some(class_axis), false)?;
    // Take batch row 0 when batched: `(B, L)` → `(L,)` via flat read of row 0.
    let out_classes = self.row0_u32(&mut output_ids)?;

    let mut ids = input_ids.try_clone()?;
    let ids_all: Vec<i32> = ids.to_vec::<i32>()?;
    // Slice row 0 of the ids by the input_ids ROW length (`ids_seq`), NOT the
    // flattened total — for a (validated batch == 1) tensor these coincide, but
    // slicing by the row length is what structurally prevents a cross-row read.
    // The shape validation above pins `ids_seq == logits_seq == out_classes.len()`.
    if ids_all.len() < ids_seq {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "ForcedAligner::decode: flattened input_ids vs row length",
        ids_seq,
        ids_all.len(),
      )));
    }
    let ids_row = &ids_all[..ids_seq];

    // Each word needs a (start, end) pair: exactly `2 * num_words` markers.
    let want_markers = num_words.checked_mul(2).ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner::decode: 2 * num_words",
        "overflows usize",
        num_words.to_string(),
      ))
    })?;

    // Collect the predicted class at each `<timestamp>` input position.
    let timestamp_token_id = self.config.timestamp_token_id;
    let mut classes: Vec<i64> = Vec::new();
    reserve_or_error(
      &mut classes,
      "ForcedAligner: timestamp classes",
      want_markers,
    )?;
    for (pos, &id) in ids_row.iter().enumerate() {
      if i64::from(id) == timestamp_token_id {
        classes.push(i64::from(out_classes[pos]));
      }
    }

    if classes.len() != want_markers {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "ForcedAligner::decode: <timestamp> marker count vs 2 * num_words",
        want_markers,
        classes.len(),
      )));
    }

    // class → milliseconds, then the LIS monotonicity repair.
    let segment_time = self.config.timestamp_segment_time;
    let ms: Vec<f64> = classes.iter().map(|&c| c as f64 * segment_time).collect();
    let fixed = fix_timestamp(&ms);

    let mut times: Vec<(f64, f64)> = Vec::new();
    reserve_or_error(&mut times, "ForcedAligner: marker times", num_words)?;
    for i in 0..num_words {
      let start_ms = fixed[i * 2] as f64;
      let end_ms = fixed[i * 2 + 1] as f64;
      times.push((round_ms_to_seconds(start_ms), round_ms_to_seconds(end_ms)));
    }
    Ok(times)
  }

  /// Build the aligner's `(1, seq)` input id tensor for `transcript` and a
  /// given number of audio tokens: `[audio_start, audio_pad * num_audio_tokens,
  /// audio_end, word0_ids..., ts, ts, word1_ids..., ts, ts, ...]`.
  ///
  /// Mirrors the reference's `encode_timestamp` (`<|audio_start|><|audio_pad|>
  /// <|audio_end|>` then each word followed by `<timestamp><timestamp>`), with
  /// the single `<audio_pad>` expanded to `num_audio_tokens` copies. Returns
  /// the `(1, seq)` id tensor.
  pub fn build_input_ids(
    &self,
    transcript: &[AlignWord],
    num_audio_tokens: usize,
  ) -> Result<Array> {
    // Total length = 1 (audio_start) + num_audio_tokens + 1 (audio_end)
    // + sum(word token counts) + 2 per word (the timestamp pair).
    let mut total: usize = 2usize
      .checked_add(num_audio_tokens)
      .ok_or_else(|| seq_overflow("audio markers + audio tokens"))?;
    for word in transcript {
      total = total
        .checked_add(word.token_ids().len())
        .and_then(|t| t.checked_add(2))
        .ok_or_else(|| seq_overflow("word tokens + timestamp pair"))?;
    }

    let mut ids: Vec<i32> = Vec::new();
    reserve_or_error(&mut ids, "ForcedAligner: built input ids", total)?;
    let audio_start = id_to_i32(self.config.audio_start_token_id, "audio_start_token_id")?;
    let audio_pad = id_to_i32(self.config.audio_token_id, "audio_token_id")?;
    let audio_end = id_to_i32(self.config.audio_end_token_id, "audio_end_token_id")?;
    let ts = id_to_i32(self.config.timestamp_token_id, "timestamp_token_id")?;

    // Reject pre-tokenized ids outside `[0, vocab_size)`: a caller's AlignWord
    // token id is copied straight into the model input and reaches MLX `take`
    // in `embed_tokens`, whose gather kernel does NOT bound-check — a negative
    // id is read as `id + vocab` (a wrong, possibly still-out-of-bounds row) and
    // an id `>= vocab` reads past the embedding table (an out-of-bounds read,
    // i.e. UB). Validating here fails fast with a typed error before the splice.
    let vocab = self.config.text_config.vocab_size;
    ids.push(audio_start);
    for _ in 0..num_audio_tokens {
      ids.push(audio_pad);
    }
    ids.push(audio_end);
    for word in transcript {
      for &id in word.token_ids() {
        check_token_id_in_vocab(
          id,
          vocab,
          "ForcedAligner::build_input_ids: pre-tokenized token id",
        )?;
        ids.push(id);
      }
      ids.push(ts);
      ids.push(ts);
    }
    debug_assert_eq!(ids.len(), total);
    Array::from_slice::<i32>(&ids, &(1usize, total))
  }

  /// The number of audio tokens the audio tower emits for an utterance with
  /// `valid_mel_len` non-padded mel frames — the **exact** windowed conv
  /// recurrence ([`AudioEncoder::windowed_output_length`] over the
  /// `chunk = n_window * 2`-frame conv chunks), used by [`align`](Self::align)
  /// to size the `<audio_pad>` run so the splice replaces exactly the audio
  /// rows.
  ///
  /// This is the same exact recurrence the windowed audio tower uses for its
  /// per-chunk keep / `seq_len`, so the `<audio_pad>` count equals the encoder's
  /// actual row count for **any** `n_window` (the reference's
  /// `_get_feat_extract_output_lengths` closed form would over-count for a
  /// non-100-frame chunk; the two agree at the standard `n_window = 50`). It is
  /// computed from the valid length directly (no audio encode), so a padded mel
  /// input is never counted by its padded `time` axis.
  fn num_audio_tokens(&self, valid_mel_len: i64) -> Result<usize> {
    let chunk = i64::from(self.audio_tower.config().n_window).saturating_mul(2);
    let tokens = AudioEncoder::windowed_output_length(valid_mel_len, chunk);
    usize::try_from(tokens.max(0)).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "ForcedAligner: audio token count",
        "exceeds usize::MAX",
        tokens.to_string(),
      ))
    })
  }

  /// The valid mel-frame count of `input_features` `(1, n_mels, time)`: the full
  /// `time` axis (the single-utterance aligner treats its input as unpadded).
  fn mel_time(input_features: &Array) -> Result<i64> {
    let s = input_features.shape();
    let t = *s.get(2).ok_or_else(|| {
      Error::RankMismatch(RankMismatchPayload::new(
        "ForcedAligner: input_features must be rank-3 (batch, n_mels, time)",
        s.len() as u32,
        s.clone(),
      ))
    })?;
    Ok(i64::try_from(t).unwrap_or(i64::MAX))
  }

  /// `(batch, seq)` shape of `input_ids`, accepting a rank-1 `(seq,)` as batch
  /// 1.
  fn input_ids_shape(&self, input_ids: &Array) -> Result<(usize, usize)> {
    let s = input_ids.shape();
    match s.as_slice() {
      [l] => Ok((1, *l)),
      [b, l] => Ok((*b, *l)),
      _ => Err(Error::RankMismatch(RankMismatchPayload::new(
        "ForcedAligner: input_ids must be rank-1 (seq) or rank-2 (batch, seq)",
        s.len() as u32,
        s,
      ))),
    }
  }

  /// `(batch, seq, classify_num)` shape of the decode `logits`, accepting a
  /// rank-2 `(seq, classify_num)` as batch 1.
  ///
  /// The decode reads, per `<timestamp>` position, the argmax class over the
  /// last axis; a too-low rank would index a missing dim, and a mismatched seq
  /// would let the row-0 slice cross batch rows, so the rank is pinned here and
  /// the dims are returned for the caller's exact-shape check.
  fn decode_logits_shape(&self, logits: &Array) -> Result<(usize, usize, usize)> {
    let s = logits.shape();
    match s.as_slice() {
      [l, c] => Ok((1, *l, *c)),
      [b, l, c] => Ok((*b, *l, *c)),
      _ => Err(Error::RankMismatch(RankMismatchPayload::new(
        "ForcedAligner: logits must be rank-2 (seq, classify_num) or rank-3 (batch, seq, classify_num)",
        s.len() as u32,
        s,
      ))),
    }
  }

  /// Test-only: the number of audio tokens the encoder emits for the full
  /// `time` axis of `input_features` — lets a shape test size the `<audio_pad>`
  /// run to the encoder output so the splice count matches.
  #[cfg(test)]
  pub(crate) fn num_audio_tokens_for_test(&self, input_features: &Array) -> usize {
    let mel_len = Self::mel_time(input_features).unwrap();
    self.num_audio_tokens(mel_len).unwrap()
  }

  /// Reject any `input_ids` value outside `[0, text_config.vocab_size)`.
  ///
  /// `embed_tokens` gathers the embedding rows with MLX `take`, whose kernel
  /// does not bound-check the indices, so an out-of-range id is an out-of-bounds
  /// embedding-table read (UB) — guarded here with a typed [`Error::OutOfRange`]
  /// before the gather. Reading the (data-backed) `input_ids` tensor is an
  /// explicit materialization of an input, not a hidden eval of a lazy graph.
  fn validate_input_ids_range(&self, input_ids: &Array) -> Result<()> {
    let vocab = self.config.text_config.vocab_size;
    let mut ids = input_ids.try_clone()?;
    let ids_flat: Vec<i32> = ids.to_vec::<i32>()?;
    for &id in &ids_flat {
      check_token_id_in_vocab(id, vocab, "ForcedAligner::forward: input_ids token id")?;
    }
    Ok(())
  }

  /// The decoder hidden width from the `(B, L, hidden)` embeddings.
  fn decoder_hidden_dim(&self, embeds: &Array) -> Result<usize> {
    embeds.shape().last().copied().ok_or_else(|| {
      Error::RankMismatch(RankMismatchPayload::new(
        "ForcedAligner: embeddings must be rank-3 (batch, seq, hidden)",
        embeds.ndim() as u32,
        embeds.shape(),
      ))
    })
  }

  /// Flat row-0 of a `(B, L)` (or `(L,)`) U32 array as a `Vec<u32>`.
  fn row0_u32(&self, a: &mut Array) -> Result<Vec<u32>> {
    let s = a.shape();
    let all = a.to_vec::<u32>()?;
    match s.as_slice() {
      [_l] => Ok(all),
      [_b, l] => Ok(all[..(*l).min(all.len())].to_vec()),
      _ => Err(Error::RankMismatch(RankMismatchPayload::new(
        "ForcedAligner: argmax output must be rank-1 or rank-2",
        s.len() as u32,
        s,
      ))),
    }
  }
}

impl ForcedAligner {
  /// Localize each word of the raw transcript `text` (in `language`) in `audio`,
  /// owning the word splitting + tokenization internally.
  ///
  /// The reference-faithful path: splits `text` into words per `language` (the
  /// `ForceAlignProcessor.encode_timestamp` word logic), assembles the
  /// `<|audio_start|>`…`<timestamp><timestamp>` marker string with the
  /// `<|audio_pad|>` placeholder expanded to the audio-token count, and encodes
  /// it with the model's own [`Tokenizer`] (the reference's
  /// `self._tokenizer.encode(.., add_special_tokens=False)`). It then forwards
  /// and decodes the per-word spans. Requires an attached tokenizer (build with
  /// [`from_weights_with_tokenizer`](Self::from_weights_with_tokenizer) /
  /// [`with_tokenizer`](Self::with_tokenizer)), else a typed
  /// [`Error::Tokenizer`].
  ///
  /// `result_language` labels the returned [`ForcedAlignment`].
  fn align_raw_text(
    &self,
    audio: &Array,
    text: &str,
    language: &str,
    result_language: Option<String>,
  ) -> Result<ForcedAlignment> {
    let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
      Error::tokenizer(
        "ForcedAligner: the RawTranscript align path requires the model tokenizer; \
         build with from_weights_with_tokenizer or with_tokenizer",
      )
    })?;

    let mel_len = Self::mel_time(audio)?;
    let num_audio_tokens = self.num_audio_tokens(mel_len)?;

    // Split the transcript into words and assemble the marker string, then
    // tokenize the whole string with the model tokenizer (the reference encodes
    // the assembled string in one pass with add_special_tokens=False).
    let word_list = split_words(text, language)?;
    let input_text = assemble_input_text(&word_list, num_audio_tokens)?;
    let ids_u32 = tokenizer.encode(&input_text, false)?;

    let mut ids: Vec<i32> = Vec::new();
    reserve_or_error(&mut ids, "ForcedAligner: raw-text input ids", ids_u32.len())?;
    for id in ids_u32 {
      ids.push(i32::try_from(id).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "ForcedAligner: encoded token id exceeds i32::MAX",
          "tokenizer id",
          id.to_string(),
        ))
      })?);
    }
    let input_ids = Array::from_slice::<i32>(&ids, &(1usize, ids.len()))?;

    let logits = self.forward_with_feature_length(&input_ids, audio, Some(mel_len))?;
    let times = self.decode_marker_times(&input_ids, &logits, word_list.len())?;

    let mut spans: Vec<AlignedSpan> = Vec::new();
    reserve_or_error(&mut spans, "ForcedAligner: aligned spans", word_list.len())?;
    for (word, (start_s, end_s)) in word_list.iter().zip(times) {
      spans.push(AlignedSpan::new(word.as_str(), start_s, end_s));
    }
    Ok(ForcedAlignment::new(spans, result_language))
  }
}

#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
impl ForcedAlignerTrait<RawTranscript> for ForcedAligner {
  /// Localize each word of a raw [`RawTranscript`] in `audio` (the precomputed
  /// mel features `(1, n_mels, time)` of a single unpadded utterance), splitting
  /// + tokenizing the transcript internally.
  ///
  /// The primary path: the aligner splits the transcript text into words per its
  /// language and tokenizes the assembled marker string with its own
  /// [`Tokenizer`], so the caller passes raw text and never a (possibly wrong)
  /// tokenization. It is reference-faithful for whitespace-segmented and
  /// CJK-per-character languages (Chinese included); Japanese and Korean — which
  /// the reference delegates to optional external segmenters this port does not
  /// bundle — return a typed [`Error::Tokenizer`] and must instead use the
  /// pre-tokenized path ([`PreTokenizedTranscript`]); see issue #322. The result
  /// language label is the transcript's language unless
  /// [`AlignOptions::language`] overrides it. Requires an attached tokenizer,
  /// else a typed [`Error::Tokenizer`].
  fn align(
    &self,
    audio: &Array,
    input: RawTranscript,
    opts: &AlignOptions,
  ) -> Result<ForcedAlignment> {
    let result_language = opts
      .language()
      .map(str::to_string)
      .or_else(|| Some(input.language().to_string()));
    self.align_raw_text(audio, input.text(), input.language(), result_language)
  }
}

#[cfg_attr(docsrs, doc(cfg(feature = "qwen3-asr-aligner")))]
impl ForcedAlignerTrait<PreTokenizedTranscript<'_>> for ForcedAligner {
  /// Localize each word of an already-tokenized [`PreTokenizedTranscript`] in
  /// `audio` (the precomputed mel features `(1, n_mels, time)` of a single
  /// unpadded utterance).
  ///
  /// Computes the audio-token count from the valid mel-frame count (the full
  /// `time` axis), builds the audio-marker-wrapped, `<timestamp>`-interleaved
  /// input ids directly from the supplied per-word token ids, forwards (trimming
  /// nothing — the utterance is unpadded), and decodes the per-word spans. The
  /// audio token count is derived from the length formula directly (no extra
  /// encode); only the forward encodes the audio, lazily until the hidden states
  /// are read in the decode. This path needs no tokenizer.
  fn align(
    &self,
    audio: &Array,
    input: PreTokenizedTranscript<'_>,
    opts: &AlignOptions,
  ) -> Result<ForcedAlignment> {
    let transcript = input.words();
    let mel_len = Self::mel_time(audio)?;
    let num_audio_tokens = self.num_audio_tokens(mel_len)?;
    let input_ids = self.build_input_ids(transcript, num_audio_tokens)?;
    let logits = self.forward_with_feature_length(&input_ids, audio, Some(mel_len))?;
    self.decode_alignment(
      &input_ids,
      &logits,
      transcript,
      opts.language().map(str::to_string),
    )
  }
}

/// Round a millisecond value to seconds at millisecond resolution
/// (`round(ms / 1000, 3)`), matching the reference's
/// `round(start_time / 1000.0, 3)`.
fn round_ms_to_seconds(ms: f64) -> f64 {
  (ms / 1000.0 * 1000.0).round() / 1000.0
}

/// `OutOfRange` for a sequence-length `usize` overflow.
fn seq_overflow(what: &'static str) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "ForcedAligner::build_input_ids: sequence length",
    "overflows usize",
    what.to_string(),
  ))
}

/// Reject a token `id` outside `[0, vocab)` — the valid embedding-row range.
///
/// MLX `take` (the `embed_tokens` gather) does not bound-check its indices, so a
/// negative id (read as `id + vocab`) or an id `>= vocab` is an out-of-bounds
/// embedding-table read (UB). This fails fast with a typed [`Error::OutOfRange`]
/// before the gather. `vocab` is the validated `text_config.vocab_size` (a
/// positive `i32`); the comparison is done in `i64` so a negative `vocab` (which
/// `validate` already rejects) still cannot admit anything.
fn check_token_id_in_vocab(id: i32, vocab: i32, context: &'static str) -> Result<()> {
  if i64::from(id) < 0 || i64::from(id) >= i64::from(vocab) {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      context,
      "token id must be in [0, vocab_size)",
      format!("id={id}, vocab_size={vocab}"),
    )));
  }
  Ok(())
}

/// Convert a config token id (`i64`) to the `i32` an id tensor stores, erroring
/// on overflow (a token id is a small non-negative embedding-row index).
fn id_to_i32(id: i64, name: &'static str) -> Result<i32> {
  i32::try_from(id).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "ForcedAligner: token id exceeds i32::MAX",
      name,
      id.to_string(),
    ))
  })
}

/// Build the host-side splice gather index for [`ForcedAligner::splice_audio`].
///
/// Returns one `i32` per flat token position: a non-pad position maps to its own
/// flat embedding row `i`, and the k-th `audio_token_id` (pad) position maps to
/// the k-th appended audio row `seq_elems_i + k` (the audio rows are
/// concatenated after the `seq_elems_i` embedding rows). Every index is produced
/// with checked arithmetic ([`i32::try_from`] / [`i32::checked_add`]) and an
/// overflow is a typed [`Error::OutOfRange`] — never a panic and never an `as`
/// truncation that would wrap past `i32::MAX` into a negative / out-of-range
/// row. The gather feeds MLX `take`, whose kernel does not bound-check its
/// indices, so a wrapped index would be an out-of-bounds row read (UB); the
/// checked arithmetic here is the soundness boundary before that unchecked op.
fn splice_gather_index(
  ids_flat: &[i32],
  audio_token_id: i64,
  seq_elems_i: i32,
) -> Result<Vec<i32>> {
  let mut gather = alloc_filled::<i32>("ForcedAligner: splice gather index", 0, ids_flat.len())?;
  let mut k: usize = 0;
  for (i, slot) in gather.iter_mut().enumerate() {
    if i64::from(ids_flat[i]) == audio_token_id {
      let k_i = i32::try_from(k).map_err(|_| splice_index_overflow("audio row offset", k))?;
      *slot = seq_elems_i
        .checked_add(k_i)
        .ok_or_else(|| splice_index_overflow("batch*seq + audio row offset", k))?;
      k += 1;
    } else {
      *slot = i32::try_from(i).map_err(|_| splice_index_overflow("embedding row index", i))?;
    }
  }
  Ok(gather)
}

/// `OutOfRange` for a [`splice_gather_index`] index that does not fit in `i32`.
fn splice_index_overflow(what: &'static str, value: usize) -> Error {
  Error::OutOfRange(OutOfRangePayload::new(
    "ForcedAligner::splice_audio: gather index",
    what,
    value.to_string(),
  ))
}

/// Split a raw transcript `text` into the ordered word units, dispatching on
/// `language` as `ForceAlignProcessor.encode_timestamp` does (matched
/// case-insensitively, mirroring the reference's `language.lower()`).
///
/// This raw-text splitter is faithful for the languages the reference handles
/// **inline**: whitespace-segmented languages and CJK-per-character languages
/// (Chinese included) go through the space-separated path
/// ([`tokenize_space_lang`]), which cleans each whitespace segment and then
/// breaks out any embedded CJK characters per character.
///
/// Japanese and Korean are the exception: the reference delegates them to
/// **optional external** morphological segmenters (`nagisa` / `soynlp`) imported
/// lazily, which this port does not bundle. They therefore return a typed
/// [`Error::Tokenizer`] (rather than silently mis-splitting — the reference
/// itself raises `ImportError` when the dependency is absent) directing the
/// caller to the pre-tokenized path ([`PreTokenizedTranscript`]), where the
/// caller supplies the Japanese/Korean tokenization. Bundling those segmenters
/// vs. a pluggable hook vs. pre-tokenized-only is the open scope decision in
/// issue #322; this raw path is therefore **not** reference-faithful for every
/// language.
fn split_words(text: &str, language: &str) -> Result<Vec<String>> {
  let lang = language.to_ascii_lowercase();
  match lang.as_str() {
    "japanese" | "korean" => Err(Error::tokenizer(
      "ForcedAligner: raw-text alignment for Japanese and Korean requires an external \
       morphological word segmenter (nagisa for Japanese, soynlp for Korean) that this \
       port does not bundle; align Japanese/Korean transcripts via the pre-tokenized path \
       — ForcedAligner over a PreTokenizedTranscript, where the caller supplies the \
       per-word tokenization (see issue #322)",
    )),
    // Every other language (Chinese included) — the reference's default branch.
    _ => Ok(tokenize_space_lang(text)),
  }
}

/// Assemble the aligner's input string for the tokenizer: the audio-span markers
/// (with the `<|audio_pad|>` placeholder repeated `num_audio_tokens` times)
/// followed by each word trailed by a `<timestamp><timestamp>` pair.
///
/// Mirrors `encode_timestamp` (words joined by `<timestamp><timestamp>` with a
/// trailing pair, then prefixed with `<|audio_start|><|audio_pad|><|audio_end|>`)
/// combined with the reference's `replace("<|audio_pad|>", "<|audio_pad|>" *
/// num_audio_tokens)` expansion — the equivalent final string built directly.
fn assemble_input_text(words: &[String], num_audio_tokens: usize) -> Result<String> {
  // Approximate the final length to reserve once (markers + audio pads + words +
  // a timestamp pair per word), bounding the reservation fallibly.
  let pad_len = AUDIO_PAD_MARKER
    .len()
    .checked_mul(num_audio_tokens)
    .ok_or_else(|| seq_overflow("audio pad markers"))?;
  let words_len: usize = words.iter().map(|w| w.len()).sum();
  let ts_pair_len = TIMESTAMP_MARKER.len().saturating_mul(2);
  let cap = AUDIO_START_MARKER
    .len()
    .checked_add(pad_len)
    .and_then(|n| n.checked_add(AUDIO_END_MARKER.len()))
    .and_then(|n| n.checked_add(words_len))
    .and_then(|n| n.checked_add(ts_pair_len.saturating_mul(words.len())))
    .ok_or_else(|| seq_overflow("assembled input text"))?;

  let mut s = String::new();
  s.try_reserve_exact(cap).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "ForcedAligner::assemble_input_text",
      "input text bytes",
      cap as u64,
      e,
    ))
  })?;
  s.push_str(AUDIO_START_MARKER);
  for _ in 0..num_audio_tokens {
    s.push_str(AUDIO_PAD_MARKER);
  }
  s.push_str(AUDIO_END_MARKER);
  for word in words {
    s.push_str(word);
    s.push_str(TIMESTAMP_MARKER);
    s.push_str(TIMESTAMP_MARKER);
  }
  Ok(s)
}

/// Whether a character is kept in a cleaned token: an apostrophe, or a character
/// whose Unicode General_Category is a Letter (`Lu`/`Ll`/`Lt`/`Lm`/`Lo`) or a
/// Number (`Nd`/`Nl`/`No`) — mirroring the reference's
/// `unicodedata.category(ch)` starting with `"L"` or `"N"` (plus `'`) exactly.
///
/// The General_Category test is required for parity: it drops combining marks
/// (`Mn`/`Mc`/`Me`, e.g. U+064E ARABIC FATHA), which the broader
/// [`char::is_alphabetic`] / [`char::is_numeric`] derived-property predicates
/// would keep, shifting timestamp-marker positions on diacritized transcripts.
fn is_kept_char(ch: char) -> bool {
  use unicode_general_category::{GeneralCategory, get_general_category};
  if ch == '\'' {
    return true;
  }
  matches!(
    get_general_category(ch),
    GeneralCategory::UppercaseLetter
      | GeneralCategory::LowercaseLetter
      | GeneralCategory::TitlecaseLetter
      | GeneralCategory::ModifierLetter
      | GeneralCategory::OtherLetter
      | GeneralCategory::DecimalNumber
      | GeneralCategory::LetterNumber
      | GeneralCategory::OtherNumber
  )
}

/// Drop the non-kept characters from `token` (the reference's `clean_token`).
fn clean_token(token: &str) -> String {
  token.chars().filter(|&c| is_kept_char(c)).collect()
}

/// Whether `ch` is a CJK ideograph — the verbatim codepoint ranges of the
/// reference's `is_cjk_char`.
fn is_cjk_char(ch: char) -> bool {
  let code = ch as u32;
  (0x4E00..=0x9FFF).contains(&code)      // CJK Unified Ideographs
    || (0x3400..=0x4DBF).contains(&code) // Extension A
    || (0x20000..=0x2A6DF).contains(&code) // Extension B
    || (0x2A700..=0x2B73F).contains(&code) // Extension C
    || (0x2B740..=0x2B81F).contains(&code) // Extension D
    || (0x2B820..=0x2CEAF).contains(&code) // Extension E
    || (0xF900..=0xFAFF).contains(&code) // Compatibility Ideographs
}

/// Split a whitespace segment, breaking out any embedded CJK characters as their
/// own tokens (the reference's `split_segment_with_chinese`).
fn split_segment_with_chinese(seg: &str) -> Vec<String> {
  let mut tokens: Vec<String> = Vec::new();
  let mut buf = String::new();
  for ch in seg.chars() {
    if is_cjk_char(ch) {
      if !buf.is_empty() {
        tokens.push(std::mem::take(&mut buf));
      }
      tokens.push(ch.to_string());
    } else {
      buf.push(ch);
    }
  }
  if !buf.is_empty() {
    tokens.push(buf);
  }
  tokens
}

/// Split space-separated text (English and the reference's default branch): each
/// whitespace segment is cleaned, dropped if empty, and any embedded CJK is split
/// out (the reference's `tokenize_space_lang`).
fn tokenize_space_lang(text: &str) -> Vec<String> {
  let mut tokens: Vec<String> = Vec::new();
  for seg in text.split_whitespace() {
    let cleaned = clean_token(seg);
    if !cleaned.is_empty() {
      tokens.extend(split_segment_with_chinese(&cleaned));
    }
  }
  tokens
}

/// Repair a non-monotone timestamp sequence using its Longest Increasing
/// Subsequence (`ForceAlignProcessor.fix_timestamp`).
///
/// The LIS (non-strict: `a[j] <= a[i]`) positions are treated as the "normal"
/// anchors; each maximal run of non-normal positions between anchors is
/// repaired: a short run (`<= 2`) is filled from the nearer valid neighbor,
/// a longer run is linearly interpolated between the bracketing anchors (or
/// flat-filled when only one side exists). Operates on the millisecond floats
/// and returns the per-position integer (truncated) milliseconds, exactly as
/// the reference does (`[int(res) for res in result]`).
fn fix_timestamp(data: &[f64]) -> Vec<i64> {
  let n = data.len();
  if n == 0 {
    return Vec::new();
  }

  // Per-position longest non-decreasing-subsequence length ending at `i`,
  // `dp[i] = 1 + max{ dp[j] : j < i, data[j] <= data[i] }`. The reference
  // computes this with an O(n^2) double loop; this is the same recurrence
  // evaluated in O(n log n) via a value-indexed Fenwick prefix-max
  // ([`lis_lengths`]), which yields a byte-identical `dp`.
  let dp = lis_lengths(data);

  // The first index achieving the maximum length (`dp.index(max)`), and the
  // per-length index buckets used to walk the parent chain.
  let max_length = dp.iter().copied().max().unwrap_or(1);
  let buckets = LengthBuckets::new(&dp, max_length);
  let max_idx = buckets.first(max_length).unwrap_or(0);

  // Reconstruct the LIS "normal" anchors. The reference's parent of `cur` is the
  // *smallest* `j < cur` with `data[j] <= data[cur]` and `dp[j] == dp[cur] - 1`
  // (the first index the ascending inner loop found at the maximal length).
  // Walking the chain consults each length bucket at most once (the chain's
  // lengths strictly decrease), so the reconstruction is linear overall.
  let mut is_normal = vec![false; n];
  let mut cur = max_idx;
  is_normal[cur] = true;
  while dp[cur] > 1 {
    let target = dp[cur] - 1;
    let Some(parent) = buckets.first_before(target, cur, data) else {
      break;
    };
    cur = parent;
    is_normal[cur] = true;
  }

  let mut result: Vec<f64> = data.to_vec();
  let mut i = 0;
  while i < n {
    if !is_normal[i] {
      // The maximal non-normal run [i, j).
      let mut j = i;
      while j < n && !is_normal[j] {
        j += 1;
      }
      let anomaly_count = j - i;

      // Nearest valid neighbor on each side (the reference scans outward from
      // the run for the closest `is_normal` position).
      let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
      let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

      if anomaly_count <= 2 {
        for (offset, slot) in result[i..j].iter_mut().enumerate() {
          let k = i + offset;
          *slot = match (left_val, right_val) {
            (None, Some(r)) => r,
            (Some(l), None) => l,
            (Some(l), Some(r)) => {
              // The reference's tie-break: pick left when the distance to the
              // left anchor (`k - (i - 1)`) is `<=` the distance to the right
              // anchor (`j - k`), else right. `i - 1` and `j` are the anchor
              // positions bracketing the run.
              let dist_left = (k + 1) - i; // k - (i - 1)
              let dist_right = j - k;
              if dist_left <= dist_right { l } else { r }
            }
            (None, None) => *slot,
          };
        }
      } else {
        match (left_val, right_val) {
          (Some(l), Some(r)) => {
            let step = (r - l) / (anomaly_count as f64 + 1.0);
            for (offset, slot) in result[i..j].iter_mut().enumerate() {
              *slot = l + step * (offset as f64 + 1.0);
            }
          }
          (Some(l), None) => {
            result[i..j].fill(l);
          }
          (None, Some(r)) => {
            result[i..j].fill(r);
          }
          (None, None) => {}
        }
      }
      i = j;
    } else {
      i += 1;
    }
  }

  result.into_iter().map(|v| v as i64).collect()
}

/// The per-position longest non-decreasing-subsequence length ending at each
/// index — `dp[i] = 1 + max{ dp[j] : j < i, data[j] <= data[i] }`, the same
/// recurrence the reference evaluates with an O(n^2) double loop, computed here
/// in O(n log n) via a Fenwick tree of prefix maxima keyed by the
/// rank-compressed values.
///
/// The values are coordinate-compressed to dense ranks (ties share a rank, so
/// the prefix query `rank <= rank(data[i])` admits exactly the `data[j] <=
/// data[i]` predecessors). Processing indices in order and querying the running
/// prefix-max before inserting `dp[i]` reproduces the reference `dp` exactly.
fn lis_lengths(data: &[f64]) -> Vec<usize> {
  let n = data.len();
  if n == 0 {
    return Vec::new();
  }

  // Coordinate-compress the values to 1-based dense ranks. NaN never appears
  // (the caller's milliseconds are `class * segment_time` with a validated
  // finite, positive quantum), so `total_cmp` gives a total order; using it
  // keeps the sort well-defined regardless.
  let mut order: Vec<usize> = (0..n).collect();
  order.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
  let mut rank = vec![0usize; n];
  let mut next_rank = 0usize;
  for (pos, &idx) in order.iter().enumerate() {
    if pos > 0 && data[idx].total_cmp(&data[order[pos - 1]]) != std::cmp::Ordering::Equal {
      next_rank += 1;
    }
    rank[idx] = next_rank + 1; // 1-based for the Fenwick index space.
  }
  let ranks = next_rank + 1;

  // Fenwick (binary-indexed) tree of prefix maxima over `dp` keyed by rank.
  let mut tree = vec![0usize; ranks + 1];
  let mut dp = vec![1usize; n];
  for i in 0..n {
    // Prefix-max over ranks `[1, rank[i]]` = best `dp[j]` with `data[j] <=
    // data[i]` among already-inserted j (all have index < i).
    let mut best = 0usize;
    let mut r = rank[i];
    while r > 0 {
      best = best.max(tree[r]);
      r -= r & r.wrapping_neg();
    }
    dp[i] = best + 1;
    // Insert `dp[i]` at `rank[i]`, keeping the tree a prefix-max.
    let mut r = rank[i];
    while r <= ranks {
      tree[r] = tree[r].max(dp[i]);
      r += r & r.wrapping_neg();
    }
  }
  dp
}

/// Indices grouped by their LIS length (`dp[i]`), each bucket in ascending
/// index order — the structure that replaces the reference's parent-link walk.
///
/// The reference reconstructs the LIS by following, from the max-length index,
/// the smallest earlier index at each one-shorter length whose value does not
/// exceed the current value. [`first_before`](Self::first_before) answers that
/// query directly from these buckets; because the reconstructed chain's lengths
/// strictly decrease, each bucket is consulted at most once, so the whole walk
/// is linear.
struct LengthBuckets {
  /// `by_len[L]` holds the indices `i` with `dp[i] == L`, in ascending order.
  /// Index 0 is unused (lengths are 1-based).
  by_len: Vec<Vec<usize>>,
}

impl LengthBuckets {
  /// Bucket the `dp` lengths (`1..=max_length`) by value.
  fn new(dp: &[usize], max_length: usize) -> Self {
    let mut by_len: Vec<Vec<usize>> = vec![Vec::new(); max_length + 1];
    for (i, &len) in dp.iter().enumerate() {
      by_len[len].push(i);
    }
    Self { by_len }
  }

  /// The first (smallest) index with `dp == length`, or `None` when no position
  /// reaches that length.
  fn first(&self, length: usize) -> Option<usize> {
    self.by_len.get(length).and_then(|b| b.first().copied())
  }

  /// The smallest index `j` with `dp[j] == length`, `j < before`, and
  /// `data[j] <= data[before]` — the reference's parent of `before` at the
  /// one-shorter length. The bucket is in ascending index order, so the first
  /// matching entry is the smallest such index.
  fn first_before(&self, length: usize, before: usize, data: &[f64]) -> Option<usize> {
    let bucket = self.by_len.get(length)?;
    bucket
      .iter()
      .copied()
      .take_while(|&j| j < before)
      .find(|&j| data[j] <= data[before])
  }
}

#[cfg(all(test, feature = "qwen3-asr-aligner"))]
mod tests;
