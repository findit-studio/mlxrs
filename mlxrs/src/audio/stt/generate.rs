//! Shared STT decoding drivers for the [`super::model`] trait architecture.
//!
//! Two families share one decoding seam each, both producing a
//! [`Transcription`] through the universal [`Transcribe`] contract:
//!
//! - **CTC** — the blanket `impl<M: CtcModel> Transcribe for M`: one encoder
//!   forward, a per-frame `argmax`, a greedy blank-collapse, and the model's
//!   vocabulary map. Safe as a blanket because no CTC model needs to override
//!   it.
//! - **Autoregressive** — the [`greedy_transcribe`] free function: the model's
//!   frontend + encoder, a fresh per-call cache, and a token-by-token greedy
//!   `argmax` loop. NOT a blanket impl (that would overlap-conflict with a
//!   model's own [`Transcribe`] impl, e.g. Whisper's, which Rust coherence
//!   forbids without specialization); each autoregressive model calls this
//!   from inside its own [`Transcribe`] impl, or supplies its own procedure.
//!
//! The shared waveform helpers ([`default_log_mel`], [`resample_waveform`],
//! and the non-empty-waveform validation both drivers run) live here so the
//! load/validation/feature-extraction logic is implemented once.
//!
//! No implicit eval: every `Array` op is a pure [`crate::ops`] composition
//! returning `Result`; the only materializations are the token boundaries (the
//! `argmax` read-outs) the drivers perform explicitly.

use crate::{
  array::Array,
  audio::{dsp, io as audio_io},
  error::{EmptyInputPayload, Error, OutOfRangePayload, RankMismatchPayload, Result},
  ops,
};

use super::model::{
  AutoregressiveStt, CtcModel, MelConfig, Segment, Transcribe, TranscribeOptions, Transcription,
};

/// Maximum number of decode steps [`greedy_transcribe`] runs before stopping
/// with a length finish when the model never emits its end-of-transcript
/// token — `448`, Whisper's text-decoder context size. Bounds the greedy loop
/// so a model that never terminates cannot drive an unbounded decode.
pub const DEFAULT_MAX_DECODE_STEPS: usize = 448;

/// Read the mono waveform samples out of an `audio` [`Array`], rejecting an
/// empty waveform.
///
/// The shared non-empty-waveform validation both decode families run: a
/// zero-sample waveform would fabricate a zero-frame feature map that concrete
/// encoders can reasonably assume is non-empty and fail deep in per-model
/// code, so it is surfaced here as a clear recoverable [`Error::EmptyInput`].
///
/// Reads through [`Array::try_clone`] so the caller's shared `&Array` borrow
/// is preserved (the `to_vec` eval needs `&mut`).
fn waveform_samples(audio: &Array) -> Result<Vec<f32>> {
  let samples = audio.try_clone()?.to_vec::<f32>()?;
  if samples.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stt: audio waveform (0 samples; the model frontend requires at least \
         one sample — provide a non-empty waveform)",
    )));
  }
  Ok(samples)
}

/// Resample a mono waveform [`Array`] from `from_rate` to `to_rate`.
///
/// A shared helper for models whose source audio rate differs from their
/// [`MelConfig::sample_rate`]: the trait input to [`Transcribe::transcribe`]
/// is a bare waveform [`Array`] carrying no sample rate, so a model that wants
/// the standard Whisper-style resample-on-mismatch runs this inside its
/// [`AutoregressiveStt::log_mel`] (or before calling [`Transcribe::transcribe`])
/// once it knows the source rate.
///
/// `from_rate == to_rate` is a verbatim copy (no FP drift), matching
/// [`crate::audio::io::resample_linear`]. The non-empty-waveform validation is
/// applied first.
pub fn resample_waveform(audio: &Array, from_rate: u32, to_rate: u32) -> Result<Array> {
  let samples = waveform_samples(audio)?;
  let resampled = audio_io::resample_linear(&samples, from_rate, to_rate)?;
  let n = i32::try_from(resampled.len()).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "stt resample_waveform: resampled length",
      "must fit in i32 (i32::MAX = 2147483647)",
      resampled.len().to_string(),
    ))
  })?;
  Array::from_slice::<f32>(&resampled, &[n])
}

/// The default log-mel frontend for [`AutoregressiveStt::log_mel`]: validate a
/// non-empty waveform, then run [`crate::audio::dsp::log_mel_spectrogram_with`]
/// with `cfg`'s parameters (including its [`crate::audio::dsp::LogFloor`]).
///
/// Output shape `(n_mels, T)` — the mlx-audio / Whisper canonical layout fed
/// straight into [`AutoregressiveStt::encode`]. Assumes `audio` is already at
/// `cfg`'s [`MelConfig::sample_rate`]; a model resampling a different source
/// rate does so (e.g. via [`resample_waveform`]) before calling this.
pub fn default_log_mel(cfg: &MelConfig, audio: &Array) -> Result<Array> {
  let samples = waveform_samples(audio)?;
  let n = i32::try_from(samples.len()).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "stt default_log_mel: samples.len()",
      "must fit in i32 (i32::MAX = 2147483647)",
      samples.len().to_string(),
    ))
  })?;
  let samples_arr = Array::from_slice::<f32>(&samples, &[n])?;
  dsp::log_mel_spectrogram_with(
    &samples_arr,
    cfg.n_fft(),
    cfg.hop_length(),
    cfg.win_length(),
    cfg.n_mels(),
    cfg.sample_rate(),
    cfg.f_min(),
    cfg.f_max(),
    cfg.log_floor(),
  )
}

/// The CTC greedy-collapse driver: every [`CtcModel`] is [`Transcribe`].
///
/// One encoder forward ([`CtcModel::logits`]) produces `(T', vocab)` per-frame
/// logits; the driver takes a per-frame `argmax`, collapses consecutive
/// duplicate ids and drops the blank id ([`CtcModel::blank_id`]) — the
/// standard CTC greedy decode — then maps the surviving ids to text via
/// [`CtcModel::decode_ids`]. The result is a single untimed [`Segment`]
/// spanning the whole utterance (CTC carries no per-frame time bounds through
/// this trait).
///
/// Safe as a blanket impl: no CTC model needs a different `transcribe`, so
/// there is no coherence conflict with a model-specific [`Transcribe`].
impl<M: CtcModel> Transcribe for M {
  fn transcribe(&self, audio: &Array, _opts: &TranscribeOptions) -> Result<Transcription> {
    // Per-frame logits `(T', vocab)`; validate the rank so a malformed encoder
    // output surfaces as a typed error rather than a confusing `argmax` shape.
    let logits = self.logits(audio)?;
    let shape = logits.shape();
    if shape.len() != 2 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "stt CtcModel::logits must be rank 2 (shape [T', vocab])",
        shape.len() as u32,
        shape,
      )));
    }

    // Per-frame argmax over the vocab axis → `(T',)` class ids.
    let mut frame_ids = ops::misc::argmax(&logits, Some(1), false)?;
    let ids = frame_ids.to_vec::<u32>()?;

    // Greedy CTC collapse: drop consecutive duplicates, then drop the blank.
    let blank = self.blank_id();
    let mut collapsed: Vec<u32> = Vec::new();
    let mut prev: Option<u32> = None;
    for &id in &ids {
      if prev != Some(id) {
        if id != blank {
          collapsed.push(id);
        }
        prev = Some(id);
      }
    }

    let text = self.decode_ids(&collapsed);
    let segments = vec![Segment::new(text.clone(), 0.0, 0.0)];
    Ok(Transcription::new(text, None, segments))
  }
}

/// The generic autoregressive greedy decode loop, callable from a model's own
/// [`Transcribe`] impl.
///
/// Procedure (using only the [`AutoregressiveStt`] hooks):
/// 1. [`AutoregressiveStt::log_mel`] — the model's frontend → log-mel features.
/// 2. [`AutoregressiveStt::encode`] — one encoder pass; states reused below.
/// 3. [`AutoregressiveStt::new_cache`] — a fresh, owned decode cache.
/// 4. [`AutoregressiveStt::initial_tokens`] — the prompt prefix to seed from.
/// 5. Greedy loop: [`AutoregressiveStt::decode_step`] → `(vocab,)` next-token
///    logits, take `argmax`, stop at [`AutoregressiveStt::eot`], else append
///    and continue — bounded by [`DEFAULT_MAX_DECODE_STEPS`].
///
/// Because the [`AutoregressiveStt`] surface carries no detokenizer, the
/// returned [`Transcription`]'s text is the decoded token-id sequence (the
/// tokens produced *after* the prompt prefix) rendered as a space-separated
/// decimal string — the deterministic, model-agnostic output the loop itself
/// controls. A model that detokenizes to natural text implements its own
/// [`Transcribe`] (Whisper does), reusing these hooks internally.
///
/// NOT a blanket impl: a blanket `impl<M: AutoregressiveStt> Transcribe` would
/// overlap-conflict with such model-specific impls, which Rust coherence
/// forbids without specialization.
pub fn greedy_transcribe<M: AutoregressiveStt>(
  m: &M,
  audio: &Array,
  opts: &TranscribeOptions,
) -> Result<Transcription> {
  let mel = m.log_mel(audio)?;
  let enc = m.encode(&mel)?;
  let mut cache = m.new_cache();

  let mut tokens = m.initial_tokens(opts)?;
  let prompt_len = tokens.len();
  let eot = m.eot();

  for _ in 0..DEFAULT_MAX_DECODE_STEPS {
    let logits = m.decode_step(&mut cache, &enc, &tokens)?;

    // Validate the `(vocab,)` next-token logits shape so a malformed
    // `decode_step` surfaces as a typed error rather than a confusing
    // `argmax`/`item` failure downstream.
    let shape = logits.shape();
    if shape.len() != 1 {
      return Err(Error::RankMismatch(RankMismatchPayload::new(
        "stt AutoregressiveStt::decode_step must return rank-1 next-token logits (shape [vocab])",
        shape.len() as u32,
        shape,
      )));
    }
    if shape[0] == 0 {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "stt AutoregressiveStt::decode_step returned an empty vocab axis (vocab == 0)",
      )));
    }

    // Greedy argmax over the vocab axis; the only materialization.
    let mut next_arr = ops::misc::argmax(&logits, None, false)?;
    let next: u32 = next_arr.item::<u32>()?;

    if next == eot {
      break;
    }
    tokens.push(next);
  }

  // The decoded ids are everything after the prompt prefix.
  let decoded = &tokens[prompt_len..];
  let text = render_token_ids(decoded);
  let language = opts.language().map(str::to_owned);
  let segments = vec![Segment::new(text.clone(), 0.0, 0.0)];
  Ok(Transcription::new(text, language, segments))
}

/// Render a decoded token-id sequence as a space-separated decimal string —
/// the model-agnostic text [`greedy_transcribe`] produces when the
/// [`AutoregressiveStt`] surface carries no detokenizer (real models override
/// [`Transcribe`] to emit natural text).
fn render_token_ids(ids: &[u32]) -> String {
  use std::fmt::Write as _;
  let mut out = String::new();
  for (i, id) in ids.iter().enumerate() {
    if i != 0 {
      out.push(' ');
    }
    // `write!` formats the integer in place (no per-element `String` alloc);
    // writing to a `String` is infallible, so the `Result` is discarded.
    let _ = write!(out, "{id}");
  }
  out
}

#[cfg(test)]
mod tests;
