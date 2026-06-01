//! Shared STT decoding drivers for the [`super::model`] trait architecture.
//!
//! Two families share one decoding seam each, both producing a
//! [`Transcription`] through the universal [`Transcribe`](super::model::Transcribe) contract. Each is a
//! free function a model calls from inside its own [`Transcribe`](super::model::Transcribe) impl — never
//! a blanket impl (a blanket would occupy the [`Transcribe`](super::model::Transcribe) slot for *every*
//! model in the family, so a model could never supply a custom decode, which
//! Rust coherence forbids without specialization):
//!
//! - **CTC** — [`greedy_ctc_transcribe`]: one encoder forward, a per-frame
//!   `argmax`, a greedy blank-collapse, and the model's vocabulary map.
//! - **Autoregressive** — [`greedy_transcribe`]: the model's frontend +
//!   encoder, a fresh per-call cache, and a token-by-token greedy `argmax`
//!   loop bounded by the decoder's [`AutoregressiveStt::max_context`].
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
  AutoregressiveStt, CtcModel, MelConfig, Segment, TranscribeOptions, Transcription,
};

/// The default TOTAL decoder context — the [`AutoregressiveStt::max_context`]
/// default — `448`, Whisper's text-decoder context size. [`greedy_transcribe`]
/// bounds `prompt + generated` by the model's `max_context`, so a model that
/// never emits its end-of-transcript token cannot drive an unbounded decode
/// and the decoder is never fed past its positional context.
pub const DEFAULT_MAX_DECODE_STEPS: usize = 448;

/// Validate a mono `audio` waveform's METADATA — rank, length, cap — WITHOUT
/// materializing it, returning its sample count.
///
/// The shared waveform gate both decode families run. Inspecting
/// [`Array::shape`] (a `Vec<usize>` read off the lazy array's metadata, no
/// `eval`) lets a malformed waveform be rejected with a typed error *before*
/// any unbounded `to_vec` allocation or graph evaluation:
///
/// - rank `!= 1` ⇒ [`Error::RankMismatch`] (a 2-D input is NOT silently
///   flattened to mono; the model decides how to lay out multi-channel audio).
/// - `0` samples ⇒ [`Error::EmptyInput`] (a zero-sample waveform fabricates a
///   zero-frame feature map concrete encoders assume is non-empty).
/// - `> MAX_DECODED_SAMPLES` ⇒ [`Error::OutOfRange`] (the same load-stage cap
///   [`crate::audio::io::load_audio`] enforces, so an oversized lazily-shaped
///   array can't drive a multi-GB materialization here).
fn validate_waveform(audio: &Array) -> Result<usize> {
  let shape = audio.shape();
  if shape.len() != 1 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "stt: audio waveform must be rank 1 (a mono [samples] array; \
         multi-channel audio is the model frontend's concern)",
      shape.len() as u32,
      shape,
    )));
  }
  let len = shape[0];
  if len == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stt: audio waveform (0 samples; the model frontend requires at least \
         one sample — provide a non-empty waveform)",
    )));
  }
  if len > audio_io::MAX_DECODED_SAMPLES {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt: audio waveform length",
      "must not exceed MAX_DECODED_SAMPLES (64 Mi samples)",
      len.to_string(),
    )));
  }
  Ok(len)
}

/// Read the mono waveform samples out of an `audio` [`Array`] after validating
/// its metadata ([`validate_waveform`]).
///
/// The metadata gate runs FIRST, so an empty / oversized / multi-rank waveform
/// is rejected with a typed error before the `to_vec` forces an `eval` + an
/// (otherwise unbounded) `Vec` allocation. Reads through [`Array::try_clone`]
/// so the caller's shared `&Array` borrow is preserved (the `to_vec` eval
/// needs `&mut`).
fn waveform_samples(audio: &Array) -> Result<Vec<f32>> {
  validate_waveform(audio)?;
  let samples = audio.try_clone()?.to_vec::<f32>()?;
  Ok(samples)
}

/// Resample a mono waveform [`Array`] from `from_rate` to `to_rate`.
///
/// A shared helper for models whose source audio rate differs from their
/// [`MelConfig::sample_rate`]: the trait input to [`Transcribe::transcribe`](super::model::Transcribe::transcribe)
/// is a bare waveform [`Array`] carrying no sample rate, so a model that wants
/// the standard Whisper-style resample-on-mismatch runs this inside its
/// [`AutoregressiveStt::log_mel`] (or before calling [`Transcribe::transcribe`](super::model::Transcribe::transcribe))
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
  // Validate the waveform metadata first (rank/non-empty/cap), then hand the
  // ORIGINAL lazy `&Array` straight to `log_mel_spectrogram_with` — it accepts
  // an `Array` and runs its own pre-eval guards, so there is no `to_vec` +
  // rebuild round-trip (no forced materialization, no extra `Vec`).
  validate_waveform(audio)?;
  dsp::log_mel_spectrogram_with(
    audio,
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

/// The CTC greedy-collapse driver, callable from a model's own [`Transcribe`](super::model::Transcribe)
/// impl.
///
/// One encoder forward ([`CtcModel::logits`]) produces `(T', vocab)` per-frame
/// logits; the driver takes a per-frame `argmax`, collapses consecutive
/// duplicate ids and drops the blank id ([`CtcModel::blank_id`]) — the
/// standard CTC greedy decode — then maps the surviving ids to text via
/// [`CtcModel::decode_ids`]. The result is a single untimed [`Segment`]
/// spanning the whole utterance (CTC carries no per-frame time bounds through
/// this trait).
///
/// NOT a blanket impl: a blanket `impl<M: CtcModel> Transcribe` would occupy
/// the [`Transcribe`](super::model::Transcribe) slot for every CTC model, so a model could never supply
/// its own [`Transcribe`](super::model::Transcribe) (a conflicting impl Rust coherence rejects). Each
/// CTC model instead calls this from inside its own [`Transcribe`](super::model::Transcribe) impl —
/// symmetric with [`greedy_transcribe`] for [`AutoregressiveStt`].
///
/// Validates the input waveform metadata (rank / length / cap) before the
/// encoder forward, and the returned logits' shape `(T', vocab)`: a malformed
/// rank is a typed [`Error::RankMismatch`]; an empty vocab axis (`vocab == 0`)
/// a typed [`Error::EmptyInput`] (mirroring the autoregressive guard). An empty
/// TIME axis (`T' == 0`) is well-defined — an empty [`Transcription`] (no ids
/// survive to collapse) — not an error.
pub fn greedy_ctc_transcribe<M: CtcModel + ?Sized>(
  model: &M,
  audio: &Array,
  _opts: &TranscribeOptions,
) -> Result<Transcription> {
  // Reject an empty / oversized / multi-rank waveform before the encoder
  // forward (the CTC frontend can reasonably assume a non-empty mono input).
  validate_waveform(audio)?;

  // Per-frame logits `(T', vocab)`; validate the rank so a malformed encoder
  // output surfaces as a typed error rather than a confusing `argmax` shape.
  let logits = model.logits(audio)?;
  let shape = logits.shape();
  if shape.len() != 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "stt CtcModel::logits must be rank 2 (shape [T', vocab])",
      shape.len() as u32,
      shape,
    )));
  }
  // Empty vocab axis: argmax over an empty axis is undefined — typed error
  // (mirrors the `AutoregressiveStt::decode_step` empty-vocab guard).
  if shape[1] == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stt CtcModel::logits returned an empty vocab axis (vocab == 0)",
    )));
  }
  // Empty time axis `(0, vocab)`: no frames to collapse → an empty
  // transcription (an explicit, non-panicking definition).
  if shape[0] == 0 {
    return Ok(Transcription::new(
      String::new(),
      None,
      vec![Segment::new(String::new(), 0.0, 0.0)],
    ));
  }

  // Per-frame argmax over the vocab axis → `(T',)` class ids.
  let mut frame_ids = ops::misc::argmax(&logits, Some(1), false)?;
  let ids = frame_ids.to_vec::<u32>()?;

  // Greedy CTC collapse: drop consecutive duplicates, then drop the blank.
  let blank = model.blank_id();
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

  let text = model.decode_ids(&collapsed);
  let segments = vec![Segment::new(text.clone(), 0.0, 0.0)];
  Ok(Transcription::new(text, None, segments))
}

/// The generic autoregressive greedy decode loop, callable from a model's own
/// [`Transcribe`](super::model::Transcribe) impl.
///
/// Procedure (using only the [`AutoregressiveStt`] hooks):
/// 1. [`AutoregressiveStt::log_mel`] — the model's frontend → log-mel features.
/// 2. [`AutoregressiveStt::encode`] — one encoder pass; states reused below.
/// 3. [`AutoregressiveStt::new_cache`] — a fresh, owned decode cache.
/// 4. [`AutoregressiveStt::initial_tokens`] — the prompt prefix to seed from.
/// 5. Greedy loop: [`AutoregressiveStt::decode_step`] → `(vocab,)` next-token
///    logits, take `argmax`, stop at [`AutoregressiveStt::eot`], else append
///    and continue.
///
/// The loop is bounded by the model's [`AutoregressiveStt::max_context`]: it
/// generates AT MOST `max_context - prompt.len()` new tokens, so
/// `prompt + generated` never exceeds the decoder's total positional context
/// (a prompt that already meets or exceeds `max_context` is a typed
/// [`Error::OutOfRange`] — there is no room left to decode).
///
/// Because the [`AutoregressiveStt`] surface carries no detokenizer, the
/// returned [`Transcription`]'s text is the decoded token-id sequence (the
/// tokens produced *after* the prompt prefix) rendered as a space-separated
/// decimal string — the deterministic, model-agnostic output the loop itself
/// controls. A model that detokenizes to natural text implements its own
/// [`Transcribe`](super::model::Transcribe) (Whisper does), reusing these hooks internally.
///
/// NOT a blanket impl: a blanket `impl<M: AutoregressiveStt> Transcribe` would
/// overlap-conflict with such model-specific impls, which Rust coherence
/// forbids without specialization.
pub fn greedy_transcribe<M: AutoregressiveStt + ?Sized>(
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

  // Bound `prompt + generated` by the decoder's TOTAL context: a prompt that
  // already fills (or overflows) `max_context` leaves no room to decode.
  let max_ctx = m.max_context();
  if prompt_len >= max_ctx {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt greedy_transcribe: initial_tokens prompt length",
      "must be < the decoder's max_context (prompt exceeds decoder context, \
         leaving no room to generate)",
      prompt_len.to_string(),
    )));
  }
  // At most this many new tokens, so the total never exceeds `max_ctx`.
  let max_new = max_ctx - prompt_len;

  for _ in 0..max_new {
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
/// [`Transcribe`](super::model::Transcribe) to emit natural text).
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
