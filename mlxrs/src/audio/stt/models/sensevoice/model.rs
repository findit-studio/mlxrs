//! The SenseVoice-Small model: the CTC head, the prepended rich-transcription
//! query rows, the greedy blank-collapse decode, the language / emotion / event
//! rich-info extraction, and the golden [`CtcModel`] / [`Transcribe`] wiring.
//!
//! Faithful port of the `SenseVoiceSmall` class (`sensevoice.py:341-552`) and
//! the swift `SenseVoiceModel` (`SenseVoiceModel.swift`). The model owns:
//!
//! - the [`Encoder`] tower producing the `(B, T+4, output_size)`
//!   hidden states;
//! - `ctc_lo` — the quantize-aware CTC projection `output_size -> vocab`
//!   (`sensevoice.py:347`), routed through [`MaybeQuantizedLinear`];
//! - `embed` — the 16-row prompt-embedding table (`sensevoice.py:348`), routed
//!   through the quantize-aware [`MaybeQuantizedEmbedding`], from which the four
//!   query rows are gathered;
//! - the [`SenseVoiceTokenizer`] detokenizer + the optional CMVN statistics
//!   (loaded by the [`super::loader`] factory).
//!
//! ## The forward (`__call__`, `sensevoice.py:426-437`)
//!
//! `feats (B, T, input_size)` -> prepend the 4 query rows (`build_query`) ->
//! [`Encoder`] -> `ctc_lo` ->
//! `log_softmax(axis=-1)` -> `(B, T+4, vocab)` per-frame log-probabilities. The
//! first 4 frames are the rich-info query heads; frame index `>= 4` is the
//! speech.
//!
//! ## The golden-trait fit (plan §5(A))
//!
//! SenseVoice is a CTC recognizer, so it implements [`CtcModel`]: [`CtcModel::logits`]
//! returns the **speech-only** `(T', vocab)` frames (the encoder forward minus
//! the 4 prepended query rows), [`CtcModel::blank_id`] is `0`
//! (`sensevoice.py:368`), and [`CtcModel::decode_ids`] is the
//! [`SenseVoiceTokenizer`] decode. It gets [`Transcribe`] by an inherent impl
//! that runs the encoder ONCE, reads the rich tags off frames 0-2
//! ([`SenseVoiceModel::rich_info`]), and collapses the speech frames — the rich
//! tags (language / emotion / event), which do not fit the universal
//! [`Segment`], are exposed through the model-local [`SenseVoiceResult`] (and
//! the inherent [`SenseVoiceModel::transcribe_rich`]) rather than widening the
//! shared contract.
//!
//! [sv]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sensevoice/sensevoice.py

use std::collections::HashMap;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  audio::stt::model::{CtcModel, Segment, Transcribe, TranscribeOptions, Transcription},
  error::{Error, OutOfRangePayload, RankMismatchPayload, Result},
  nn::{MaybeQuantizedEmbedding, MaybeQuantizedLinear},
  ops,
};

use super::{
  config::Config,
  encoder::Encoder,
  frontend::{apply_cmvn, apply_lfr, compute_fbank},
  tokenizer::SenseVoiceTokenizer,
};

/// The CTC blank class id collapsed out of the greedy decode
/// (`sensevoice.py:368`: `self.blank_id = 0`).
pub const BLANK_ID: u32 = 0;

/// The number of prepended query rows: `[language, event, emotion, textnorm]`
/// (`sensevoice.py:432-433`). The encoder output's first [`QUERY_FRAMES`] frames
/// are the rich-info heads; frame index `>= QUERY_FRAMES` is the speech.
pub const QUERY_FRAMES: i32 = 4;

/// The fixed `event` query embedding row id (`sensevoice.py:410`:
/// `event_emo_query = self.embed([[1, 2]])`, the first of the pair).
const EVENT_QUERY_ROW: i32 = 1;
/// The fixed `emotion` query embedding row id (`sensevoice.py:410`, the second
/// of the pair).
const EMOTION_QUERY_ROW: i32 = 2;

/// Resolve the language-id query embedding row from a language code, mirroring
/// the reference `lid_dict` (`sensevoice.py:350-358`).
///
/// An unrecognized language falls back to `auto` (`0`), matching the reference
/// `self.lid_dict.get(language, 0)` (`sensevoice.py:403`).
fn lid_query_row(language: &str) -> i32 {
  match language {
    "auto" => 0,
    "zh" => 3,
    "en" => 4,
    "yue" => 7,
    "ja" => 11,
    "ko" => 12,
    "nospeech" => 13,
    _ => 0,
  }
}

/// Resolve the text-normalization query embedding row from the ITN flag,
/// mirroring the reference `textnorm_dict` (`sensevoice.py:359, 406-407`):
/// `withitn = 14` when `use_itn`, else `woitn = 15`.
const fn textnorm_query_row(use_itn: bool) -> i32 {
  if use_itn { 14 } else { 15 }
}

/// Map a language-id frame argmax to its label, mirroring the reference
/// `lid_map` (`sensevoice.py:469-477`): an unrecognized id is `"unknown"`.
fn lid_label(id: u32) -> String {
  match id {
    24884 => "zh",
    24885 => "en",
    24888 => "yue",
    24892 => "ja",
    24896 => "ko",
    24992 => "nospeech",
    _ => "unknown",
  }
  .to_string()
}

/// Map an emotion frame argmax to its label, mirroring the reference `emo_map`
/// (`sensevoice.py:480-491`): an unrecognized id is `"token_<id>"`.
fn emotion_label(id: u32) -> String {
  match id {
    25001 => "happy".to_string(),
    25002 => "sad".to_string(),
    25003 => "angry".to_string(),
    25004 => "neutral".to_string(),
    25005 => "fearful".to_string(),
    25006 => "disgusted".to_string(),
    25007 => "surprised".to_string(),
    25008 => "other".to_string(),
    25009 => "unk".to_string(),
    other => format!("token_{other}"),
  }
}

/// Map an event frame argmax to its label, mirroring the reference `event_map`
/// (`sensevoice.py:494-500`): an unrecognized id is `"token_<id>"`.
fn event_label(id: u32) -> String {
  match id {
    24993 => "Speech".to_string(),
    24995 => "BGM".to_string(),
    24997 => "Laughter".to_string(),
    24999 => "Applause".to_string(),
    other => format!("token_{other}"),
  }
}

/// The rich-transcription tags SenseVoice predicts alongside the text — the
/// language / emotion / event argmax heads off the first 3 prepended query
/// frames (`_extract_rich_info`, `sensevoice.py:465-502`).
///
/// These do NOT fit the universal [`crate::audio::stt::model::Segment`] (which
/// carries only text + time bounds), so they are surfaced through this
/// model-local type and the inherent [`SenseVoiceModel::transcribe_rich`] rather
/// than widening the shared STT contract (plan §5 Q3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichInfo {
  /// The detected language label (`lid_map`, `sensevoice.py:469-477`);
  /// `"unknown"` for an unrecognized id.
  language: String,
  /// The detected emotion label (`emo_map`, `sensevoice.py:480-491`);
  /// `"token_<id>"` for an unrecognized id.
  emotion: String,
  /// The detected acoustic-event label (`event_map`, `sensevoice.py:494-500`);
  /// `"token_<id>"` for an unrecognized id.
  event: String,
}

impl RichInfo {
  /// Construct from the three resolved labels.
  #[inline(always)]
  pub fn new(
    language: impl Into<String>,
    emotion: impl Into<String>,
    event: impl Into<String>,
  ) -> Self {
    Self {
      language: language.into(),
      emotion: emotion.into(),
      event: event.into(),
    }
  }

  /// The detected language label (`"unknown"` if unrecognized).
  #[inline(always)]
  pub fn language(&self) -> &str {
    &self.language
  }

  /// The detected emotion label (`"token_<id>"` if unrecognized).
  #[inline(always)]
  pub fn emotion(&self) -> &str {
    &self.emotion
  }

  /// The detected acoustic-event label (`"token_<id>"` if unrecognized).
  #[inline(always)]
  pub fn event(&self) -> &str {
    &self.event
  }
}

/// The full SenseVoice transcription result: the text plus the rich tags — the
/// model-local analogue of the reference `STTOutput` `segments[0]`
/// (`sensevoice.py:541-552`), which carries `{text, language, emotion, event}`.
///
/// The inherent [`SenseVoiceModel::transcribe_rich`] returns this; the universal
/// [`Transcribe::transcribe`] returns the standard [`Transcription`] (text +
/// language + a single full-utterance [`Segment`], minus the emotion / event
/// keys the golden [`Segment`] lacks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenseVoiceResult {
  /// The decoded text (the greedy CTC collapse over the speech frames).
  text: String,
  /// The rich-transcription tags (language / emotion / event).
  rich: RichInfo,
}

impl SenseVoiceResult {
  /// Construct from the decoded text and the rich tags.
  #[inline(always)]
  pub fn new(text: impl Into<String>, rich: RichInfo) -> Self {
    Self {
      text: text.into(),
      rich,
    }
  }

  /// The decoded text.
  #[inline(always)]
  pub fn text(&self) -> &str {
    &self.text
  }

  /// The rich-transcription tags.
  #[inline(always)]
  pub const fn rich(&self) -> &RichInfo {
    &self.rich
  }
}

/// The SenseVoice-Small model (`SenseVoiceSmall`, `sensevoice.py:341-552`).
///
/// Holds the [`Encoder`] tower, the quantize-aware `ctc_lo` CTC head and the
/// 16-row `embed` query table, the [`SenseVoiceTokenizer`] detokenizer, and the
/// optional global CMVN statistics. The [`Config`] is retained for the front-end
/// parameters ([`SenseVoiceModel::extract_features`]) and the `input_size` /
/// `vocab_size` the loader pins.
#[derive(Debug)]
pub struct SenseVoiceModel {
  config: Config,
  encoder: Encoder,
  /// The CTC projection `output_size -> vocab` (`sensevoice.py:347`).
  ctc_lo: MaybeQuantizedLinear,
  /// The 16-row prompt-embedding query table (`sensevoice.py:348`).
  embed: MaybeQuantizedEmbedding,
  /// The detokenizer (SentencePiece / `tokens.json` / id-join).
  tokenizer: SenseVoiceTokenizer,
  /// The global CMVN means (the `am.mvn` `<AddShift>` or the in-config
  /// fallback), or `None` when the checkpoint ships no CMVN statistics.
  cmvn_means: Option<Array>,
  /// The global CMVN inverse standard deviations (the `am.mvn` `<Rescale>` or
  /// the in-config fallback).
  cmvn_istd: Option<Array>,
}

impl SenseVoiceModel {
  /// Assemble a model from its already-built components.
  ///
  /// The factory ([`SenseVoiceModel::from_weights`]) builds the [`Encoder`],
  /// `ctc_lo`, and `embed` from a weight map and the [`SenseVoiceTokenizer`] +
  /// CMVN statistics from the model directory, then calls this. The CMVN pair is
  /// `Some` together
  /// or `None` together (the reference loads `means`/`istd` as a pair,
  /// `sensevoice.py:573-579`); a half-present pair is a loader bug, so the
  /// constructor is total and trusts its inputs.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    config: Config,
    encoder: Encoder,
    ctc_lo: MaybeQuantizedLinear,
    embed: MaybeQuantizedEmbedding,
    tokenizer: SenseVoiceTokenizer,
    cmvn_means: Option<Array>,
    cmvn_istd: Option<Array>,
  ) -> Self {
    Self {
      config,
      encoder,
      ctc_lo,
      embed,
      tokenizer,
      cmvn_means,
      cmvn_istd,
    }
  }

  /// The model configuration.
  #[inline(always)]
  pub const fn config_ref(&self) -> &Config {
    &self.config
  }

  /// The detokenizer.
  #[inline(always)]
  pub const fn tokenizer_ref(&self) -> &SenseVoiceTokenizer {
    &self.tokenizer
  }

  /// Turn a mono waveform into the `(T', input_size)` LFR features the forward
  /// consumes, mirroring `_extract_features` (`sensevoice.py:378-395`): Kaldi
  /// fbank -> LFR stacking -> optional CMVN.
  ///
  /// `audio` is a 1-D float [`Array`] at the config sample rate. The CMVN step
  /// runs only when the model carries statistics (`am.mvn` or the in-config
  /// fallback), exactly as the reference's `if self._cmvn_means is not None`
  /// guard (`sensevoice.py:392`).
  ///
  /// # Errors
  /// Propagates the fbank / LFR / CMVN op errors (an unrecognized window, a
  /// rank-mismatched waveform, a CMVN length mismatch).
  pub fn extract_features(&self, audio: &Array) -> Result<Array> {
    let fc = self.config.frontend_conf();
    let fbank = compute_fbank(audio, fc)?;
    let feats = apply_lfr(&fbank, fc.lfr_m(), fc.lfr_n())?;
    match (&self.cmvn_means, &self.cmvn_istd) {
      (Some(means), Some(istd)) => apply_cmvn(&feats, means, istd),
      _ => Ok(feats),
    }
  }

  /// Build the four prepended query rows, mirroring `_build_query`
  /// (`sensevoice.py:397-424`).
  ///
  /// Returns `(textnorm_query, input_query)` where `input_query =
  /// concat([language_query, event_emo_query], axis=1)` is the
  /// `[language, event, emotion]` 3-row prefix and `textnorm_query` is the
  /// single text-norm row (`sensevoice.py:423-424`). Each is `(B, n, input_size)`
  /// (gathered rows broadcast over the batch, `sensevoice.py:412-421`).
  ///
  /// The embedding rows: `language_query = embed[[lid]]` (the `lid_dict` row,
  /// `sensevoice.py:404`), `event_emo_query = embed[[1, 2]]` (the fixed event +
  /// emotion rows, `sensevoice.py:410`), `textnorm_query = embed[[14 or 15]]`
  /// (the `textnorm_dict` row, `sensevoice.py:407-408`).
  ///
  /// # Errors
  /// - [`Error::OutOfRange`] if `batch_size` does not fit in `i32`;
  /// - propagates the embedding gather / broadcast op errors.
  fn build_query(
    &self,
    batch_size: usize,
    language: &str,
    use_itn: bool,
  ) -> Result<(Array, Array)> {
    let b = i32::try_from(batch_size).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::build_query: batch_size",
        "must fit in i32",
        format_smolstr!("{batch_size}"),
      ))
    })?;

    // `language_query = embed([[lid]])` — gather row `lid` with a (1, 1) id grid
    // so the gathered shape is (1, 1, input_size) (`sensevoice.py:403-404`).
    let lid = lid_query_row(language);
    let lid_ids = Array::from_slice::<i32>(&[lid], &[1, 1])?;
    let language_query = self.embed.gather(&lid_ids)?;

    // `event_emo_query = embed([[1, 2]])` — the fixed event + emotion rows, a
    // (1, 2) id grid -> (1, 2, input_size) (`sensevoice.py:410`).
    let event_emo_ids = Array::from_slice::<i32>(&[EVENT_QUERY_ROW, EMOTION_QUERY_ROW], &[1, 2])?;
    let event_emo_query = self.embed.gather(&event_emo_ids)?;

    // `textnorm_query = embed([[14 or 15]])` (`sensevoice.py:406-408`).
    let textnorm = textnorm_query_row(use_itn);
    let textnorm_ids = Array::from_slice::<i32>(&[textnorm], &[1, 1])?;
    let textnorm_query = self.embed.gather(&textnorm_ids)?;

    // Broadcast each query to the batch when B > 1 (`sensevoice.py:412-421`):
    // `broadcast_to(q, (B,) + q.shape[1:])`.
    let (language_query, event_emo_query, textnorm_query) = if b > 1 {
      (
        Self::broadcast_query(&language_query, b)?,
        Self::broadcast_query(&event_emo_query, b)?,
        Self::broadcast_query(&textnorm_query, b)?,
      )
    } else {
      (language_query, event_emo_query, textnorm_query)
    };

    // `input_query = concat([language_query, event_emo_query], axis=1)`
    // (`sensevoice.py:423`) -> (B, 3, input_size): [language, event, emotion].
    let input_query = ops::shape::concatenate(&[&language_query, &event_emo_query], 1)?;
    Ok((textnorm_query, input_query))
  }

  /// Broadcast a `(1, n, input_size)` query to `(B, n, input_size)`
  /// (`sensevoice.py:413-415`: `broadcast_to(q, (B,) + q.shape[1:])`).
  fn broadcast_query(query: &Array, b: i32) -> Result<Array> {
    let shape = query.shape();
    let n = i32::try_from(shape[1]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::broadcast_query: query rows",
        "must fit in i32",
        format_smolstr!("{}", shape[1]),
      ))
    })?;
    let dim = i32::try_from(shape[2]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::broadcast_query: query dim",
        "must fit in i32",
        format_smolstr!("{}", shape[2]),
      ))
    })?;
    ops::shape::broadcast_to(query, &[b, n, dim])
  }

  /// Run the full forward, mirroring `__call__` (`sensevoice.py:426-437`):
  /// prepend the 4 query rows, run the [`Encoder`], project with `ctc_lo`, and
  /// take `log_softmax(axis=-1)` -> `(B, T+4, vocab)` per-frame log-probs.
  ///
  /// The query assembly (`sensevoice.py:432-433`): `speech =
  /// concat([textnorm_query, feats], axis=1)`, then `speech =
  /// concat([input_query, speech], axis=1)`, so the final time order is
  /// `[language, event, emotion, textnorm, <feats…>]`.
  ///
  /// `feats` is `(B, T, input_size)`.
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `feats` is not rank-3;
  /// - propagates the query / concat / encoder / projection / log-softmax op
  ///   errors.
  pub fn forward(&self, feats: &Array, language: &str, use_itn: bool) -> Result<Array> {
    let shape = feats.shape();
    if shape.len() != 3 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "SenseVoiceModel::forward: feats must be rank-3 (B, T, input_size)",
        shape.len() as u32,
        shape,
      )));
    }
    let batch_size = shape[0];
    let (textnorm_query, input_query) = self.build_query(batch_size, language, use_itn)?;

    // `speech = concat([textnorm_query, feats], axis=1)` then
    // `concat([input_query, speech], axis=1)` (`sensevoice.py:432-433`).
    let speech = ops::shape::concatenate(&[&textnorm_query, feats], 1)?;
    let speech = ops::shape::concatenate(&[&input_query, &speech], 1)?;

    let encoder_out = self.encoder.forward(&speech)?;
    let logits = self.ctc_lo.forward(&encoder_out)?;
    log_softmax_last_axis(&logits)
  }

  /// Run the encoder forward for a single mono utterance and return the rank-2
  /// `(T+4, vocab)` log-probs (the batch axis squeezed) — the shared chokepoint
  /// for both [`CtcModel::logits`] (which slices off the query frames) and the
  /// inherent [`SenseVoiceModel::transcribe_rich`] (which reads the query frames
  /// then collapses the speech frames).
  ///
  /// Mirrors the reference `generate` head (`sensevoice.py:526-530`):
  /// `feats = extract_features(audio)[None]; log_probs = self(feats)[0]`.
  ///
  /// # Errors
  /// Propagates [`Self::extract_features`] / [`Self::forward`] / squeeze errors.
  fn utterance_log_probs(&self, audio: &Array, language: &str, use_itn: bool) -> Result<Array> {
    // `feats[None, :, :]` — add the leading batch axis (`sensevoice.py:527`).
    let feats = self.extract_features(audio)?;
    let feats = ops::shape::expand_dims_axes(&feats, &[0])?;
    let log_probs = self.forward(&feats, language, use_itn)?;
    // `self(feats)[0]` — drop the single-utterance batch axis (`sensevoice.py:530`).
    ops::shape::squeeze_axes(&log_probs, &[0])
  }

  /// Extract the rich-transcription tags from the first 3 query frames of a
  /// `(T+4, vocab)` log-prob grid, mirroring `_extract_rich_info`
  /// (`sensevoice.py:465-502`): argmax frame 0 -> language, frame 1 -> emotion,
  /// frame 2 -> event (frame 3, the text-norm slot, is read for nothing — plan
  /// §9 Q8).
  ///
  /// `log_probs` is the rank-2 utterance grid; the caller passes the FULL grid
  /// (this reads only rows `[0, 3)`). Each argmax is taken over the vocab axis of
  /// a single frame row.
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `log_probs` is not rank-2;
  /// - [`Error::OutOfRange`] if the grid has fewer than [`QUERY_FRAMES`] frames
  ///   (the query rows must be present);
  /// - propagates the slice / argmax / item op errors.
  pub fn rich_info(&self, log_probs: &Array) -> Result<RichInfo> {
    let shape = log_probs.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "SenseVoiceModel::rich_info: log_probs must be rank-2 (T+4, vocab)",
        shape.len() as u32,
        shape,
      )));
    }
    let frames = shape[0];
    if (frames as i64) < i64::from(QUERY_FRAMES) {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::rich_info: frame count",
        "must have at least QUERY_FRAMES (4) prepended query frames",
        format_smolstr!("{frames}"),
      )));
    }

    // Argmax over the vocab axis of frames 0 / 1 / 2 (`sensevoice.py:468/479/493`).
    let language = lid_label(Self::frame_argmax(log_probs, 0)?);
    let emotion = emotion_label(Self::frame_argmax(log_probs, 1)?);
    let event = event_label(Self::frame_argmax(log_probs, 2)?);
    Ok(RichInfo::new(language, emotion, event))
  }

  /// Argmax over the vocab axis of a single frame row `frame` of a rank-2
  /// `(frames, vocab)` grid (`mx.argmax(log_probs[frame]).item()`,
  /// `sensevoice.py:468`).
  fn frame_argmax(log_probs: &Array, frame: i32) -> Result<u32> {
    let vocab = i32::try_from(log_probs.shape()[1]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::frame_argmax: vocab",
        "must fit in i32",
        format_smolstr!("{}", log_probs.shape()[1]),
      ))
    })?;
    // `log_probs[frame]` — slice the single frame row -> (1, vocab) -> (vocab,).
    let row = ops::indexing::slice(log_probs, &[frame, 0], &[frame + 1, vocab], &[1, 1])?;
    let row = ops::shape::reshape(&row, &[vocab])?;
    let mut arg = ops::misc::argmax(&row, None, false)?;
    arg.item::<u32>()
  }

  /// The speech-only `(T', vocab)` log-probs — the full utterance grid with the
  /// first [`QUERY_FRAMES`] query rows sliced off (`log_probs[4:]`,
  /// `sensevoice.py:533`).
  ///
  /// This is what [`CtcModel::logits`] returns: the post-prefix frames the
  /// greedy collapse runs over. A grid with exactly [`QUERY_FRAMES`] frames (a
  /// zero-length input) yields an empty `(0, vocab)` speech grid — a well-defined
  /// empty transcription, not an error.
  ///
  /// # Errors
  /// - [`Error::RankMismatch`] if `log_probs` is not rank-2;
  /// - [`Error::OutOfRange`] if the grid has fewer than [`QUERY_FRAMES`] frames;
  /// - propagates the slice op error.
  fn speech_frames(log_probs: &Array) -> Result<Array> {
    let shape = log_probs.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "SenseVoiceModel::speech_frames: log_probs must be rank-2 (T+4, vocab)",
        shape.len() as u32,
        shape,
      )));
    }
    let frames = i32::try_from(shape[0]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::speech_frames: frame count",
        "must fit in i32",
        format_smolstr!("{}", shape[0]),
      ))
    })?;
    if frames < QUERY_FRAMES {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::speech_frames: frame count",
        "must have at least QUERY_FRAMES (4) prepended query frames",
        format_smolstr!("{frames}"),
      )));
    }
    let vocab = i32::try_from(shape[1]).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        "SenseVoiceModel::speech_frames: vocab",
        "must fit in i32",
        format_smolstr!("{}", shape[1]),
      ))
    })?;
    // `log_probs[4:]` — drop the 4 query frames (`sensevoice.py:533`).
    ops::indexing::slice(log_probs, &[QUERY_FRAMES, 0], &[frames, vocab], &[1, 1])
  }

  /// Greedily collapse a speech-frame `(T', vocab)` log-prob grid to a token-id
  /// sequence, mirroring `_greedy_ctc_decode` (`sensevoice.py:450-463`): per-frame
  /// argmax -> run-length dedup -> drop the blank id (`0`).
  ///
  /// This is the same collapse the shared
  /// [`greedy_ctc_transcribe`](crate::audio::stt::generate::greedy_ctc_transcribe)
  /// driver runs; it is replicated here (a faithful port of
  /// `sensevoice.py:451-461`)
  /// so the inherent rich-transcription path can run ONE encoder forward and
  /// decode both the rich tags and the text from it, rather than forwarding
  /// twice (once for the rich heads, once inside the driver). The
  /// [`CtcModel`] route still uses the shared driver verbatim.
  ///
  /// # Errors
  /// Propagates the argmax / `to_vec` op errors.
  fn greedy_collapse(speech_frames: &Array) -> Result<Vec<u32>> {
    // `pred = argmax(log_probs, axis=-1)` -> (T',) (`sensevoice.py:451`).
    let mut pred = ops::misc::argmax(speech_frames, Some(1), false)?;
    let ids = pred.to_vec::<u32>()?;

    // Run-length dedup then drop the blank (`sensevoice.py:454-461`).
    let mut collapsed: Vec<u32> = Vec::new();
    let mut prev: Option<u32> = None;
    for &id in &ids {
      if prev != Some(id) {
        if id != BLANK_ID {
          collapsed.push(id);
        }
        prev = Some(id);
      }
    }
    Ok(collapsed)
  }

  /// Transcribe a mono `audio` waveform to text + rich tags, mirroring the
  /// reference `generate` (`sensevoice.py:504-552`).
  ///
  /// Runs the encoder ONCE (`utterance_log_probs`), reads the rich tags
  /// off frames 0-2 ([`Self::rich_info`]), and greedily collapses + decodes the
  /// speech frames (`log_probs[4:]`). The returned [`SenseVoiceResult`] carries
  /// the text and the language / emotion / event tags — the model-local rich
  /// result the universal [`Transcribe`] cannot express.
  ///
  /// `language` is the conditioning language code (`"auto"` to let the LID head
  /// decide, the reference default `sensevoice.py:508`); `use_itn` toggles the
  /// inverse-text-normalization query row (`sensevoice.py:509`).
  ///
  /// # Errors
  /// Propagates the feature-extraction / forward / rich-info / collapse op
  /// errors.
  pub fn transcribe_rich(
    &self,
    audio: &Array,
    language: &str,
    use_itn: bool,
  ) -> Result<SenseVoiceResult> {
    let log_probs = self.utterance_log_probs(audio, language, use_itn)?;
    // Rich tags off frames 0-2 (`sensevoice.py:532`).
    let rich = self.rich_info(&log_probs)?;
    // Greedy collapse over the speech frames `log_probs[4:]` (`sensevoice.py:533`).
    let speech = Self::speech_frames(&log_probs)?;
    let ids = Self::greedy_collapse(&speech)?;
    let text = self.tokenizer.decode(&ids);
    Ok(SenseVoiceResult::new(text, rich))
  }
}

/// `log_softmax(x, axis=-1)` — `x - logsumexp(x, axis=-1, keepdims=True)`,
/// the numerically-stable form mlx's `nn.log_softmax` computes
/// (`sensevoice.py:437`). `mlxrs` has no standalone `log_softmax` op, so it is
/// composed from [`crate::ops::reduction::logsumexp_axes`] + subtract here.
///
/// # Errors
/// - [`Error::RankMismatch`] if `x` is rank-0 (no last axis to reduce);
/// - propagates the logsumexp / subtract op errors.
fn log_softmax_last_axis(x: &Array) -> Result<Array> {
  let rank = x.shape().len();
  if rank == 0 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "log_softmax_last_axis: input must have a last axis (rank >= 1)",
      0,
      x.shape(),
    )));
  }
  let last_axis = (rank - 1) as i32;
  let lse = ops::reduction::logsumexp_axes(x, &[last_axis], true)?;
  x.subtract(&lse)
}

// ───────────────────────────── golden traits ─────────────────────────────

// SenseVoice is a CTC recognizer (plan §5(A)): `CtcModel` supplies the shared
// greedy-collapse pieces — the speech-only `logits`, the blank id, and the
// detokenizer — so a direct `greedy_ctc_transcribe(&model, …)` works, and the
// inherent `Transcribe` reuses the same `decode_ids` seam.
#[cfg(feature = "sensevoice")]
impl CtcModel for SenseVoiceModel {
  /// The **speech-only** per-frame logits `(T', vocab)` — the encoder forward
  /// minus the 4 prepended rich-info query rows (`log_probs[4:]`,
  /// `sensevoice.py:533`).
  ///
  /// Runs the front-end + encoder + `ctc_lo` + `log_softmax` for the single
  /// mono utterance, squeezes the batch axis, and slices off the query frames so
  /// the shared
  /// [`greedy_ctc_transcribe`](crate::audio::stt::generate::greedy_ctc_transcribe)
  /// driver collapses ONLY the speech
  /// frames (the first 4 frames are the rich heads, not speech). Conditioning
  /// uses `language = "auto"` + `use_itn = false` (the reference defaults,
  /// `sensevoice.py:508-509`) — the [`Transcribe`] route reads the rich tags off
  /// the same frames separately.
  fn logits(&self, waveform: &Array) -> Result<Array> {
    let log_probs = self.utterance_log_probs(waveform, "auto", false)?;
    Self::speech_frames(&log_probs)
  }

  /// The CTC blank class id collapsed out of the greedy decode
  /// (`sensevoice.py:368`: `self.blank_id = 0`).
  #[inline(always)]
  fn blank_id(&self) -> u32 {
    BLANK_ID
  }

  /// Map a collapsed speech-id sequence to text via the [`SenseVoiceTokenizer`]
  /// — the SentencePiece / `tokens.json` / id-join decode (`_decode_tokens`,
  /// `sensevoice.py:439-448`). This is the single decode seam both the shared
  /// driver and the inherent [`SenseVoiceModel::transcribe_rich`] run.
  fn decode_ids(&self, ids: &[u32]) -> String {
    self.tokenizer.decode(ids)
  }
}

// SenseVoice's `Transcribe` runs ONE encoder forward and reads BOTH the rich
// tags (off frames 0-2) and the text (the collapsed speech frames) from it —
// faithful to the reference `generate`, which forwards once
// (`sensevoice.py:529`) and extracts both. It therefore does NOT delegate to
// `greedy_ctc_transcribe` (which would forward a second time inside its own
// `logits` call), but runs the same greedy collapse (`Self::greedy_collapse`,
// a faithful port of the driver's collapse + `sensevoice.py:454-461`) over the
// speech frames. The `CtcModel` impl is still fully present, so a text-only
// caller can drive the model through the shared `greedy_ctc_transcribe`
// verbatim. The universal `Transcription` carries the text + the LID-head
// language + a single full-utterance `Segment`; the emotion / event tags, which
// `Segment` cannot express, are exposed through the inherent `transcribe_rich`
// + `SenseVoiceResult` rather than widening the shared contract (plan §5 Q3).
#[cfg(feature = "sensevoice")]
impl Transcribe for SenseVoiceModel {
  /// Transcribe `audio` to a [`Transcription`].
  ///
  /// Runs the full rich transcription ([`SenseVoiceModel::transcribe_rich`] with
  /// `use_itn = false`, the reference default `sensevoice.py:509`) and maps it
  /// onto the universal result: the detected language fills
  /// [`Transcription::language`] (when not `"unknown"`), and the text is carried
  /// as a single full-utterance [`Segment`] — faithful to the reference's
  /// single-`segments[0]` shape (`sensevoice.py:544-551`), minus the emotion /
  /// event keys the golden [`Segment`] lacks (those stay on
  /// [`SenseVoiceResult`]).
  ///
  /// `opts` conditions the decode where SenseVoice supports it: an explicit
  /// [`TranscribeOptions::language`] sets the LID query row (`None` ⇒ the
  /// reference `"auto"`); the task / temperature / timestamp knobs do not apply
  /// to a non-autoregressive CTC model and are ignored.
  fn transcribe(&self, audio: &Array, opts: &TranscribeOptions) -> Result<Transcription> {
    let language = opts.language().unwrap_or("auto");
    let result = self.transcribe_rich(audio, language, false)?;
    let detected = result.rich().language();
    // The reference reports the LID-head language at the top level
    // (`sensevoice.py:543`); surface it unless it is the "unknown" sentinel.
    let language = if detected == "unknown" {
      None
    } else {
      Some(detected.to_string())
    };
    let segment = Segment::new(result.text(), 0.0, 0.0);
    Ok(Transcription::new(result.text(), language, vec![segment]))
  }
}

/// Build the `ctc_lo` CTC head + the `embed` query table from a checkpoint
/// weight map — the quantize-aware head the [`super::loader`] factory composes
/// with the [`Encoder`]. `ctc_lo` projects `output_size -> vocab`
/// (`sensevoice.py:347`); `embed` is the 16-row prompt table
/// (`sensevoice.py:348`). Both auto-detect a quantized checkpoint via their
/// sibling `.scales` (the `class_predicate` analogue), exactly as the encoder's
/// linears do.
///
/// `quant` carries the resolved `(group_size, bits, mode)` scheme (a dense
/// checkpoint passes `None`); quant resolution is the loader's concern.
///
/// # Errors
/// - [`Error::MissingKey`] for an absent `ctc_lo.weight` / `embed.weight`;
/// - propagates the [`MaybeQuantizedLinear`] / [`MaybeQuantizedEmbedding`] build
///   errors.
pub fn build_head(
  weights: &mut HashMap<String, Array>,
  quant: Option<(i32, i32, &str)>,
) -> Result<(MaybeQuantizedLinear, MaybeQuantizedEmbedding)> {
  let ctc_lo = MaybeQuantizedLinear::from_weights(weights, "ctc_lo", quant)?;
  let embed = MaybeQuantizedEmbedding::from_weights(weights, "embed", quant)?;
  Ok((ctc_lo, embed))
}

#[cfg(test)]
mod tests;
