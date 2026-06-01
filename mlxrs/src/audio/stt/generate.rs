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
  error::{
    ArithmeticOverflowPayload, EmptyInputPayload, Error, InvariantViolationPayload,
    OutOfRangePayload, RankMismatchPayload, Result,
  },
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

/// A generous-but-bounded sanity cap on a model-provided logits VOCAB axis,
/// `256 Ki` (`262_144`).
///
/// Both drivers take a per-position `argmax` over the vocab axis of logits the
/// model returns; the axis length is read off the lazy array's [`Array::shape`]
/// (no `eval`) so an absurd width can be rejected with a typed
/// [`Error::OutOfRange`] *before* the `argmax`/materialization. Real speech
/// vocabularies are at most ~256 K tokens, so `256 Ki` admits every real model
/// yet rejects a denial-of-service shape (a lazily-shaped `(T, huge_vocab)` that
/// would otherwise materialize multi-GB of intermediate logits before any
/// error).
pub const MAX_LOGITS_VOCAB: usize = 256 * 1024;

/// A bound on the TOTAL element count of a model-provided CTC logits grid,
/// `64 Mi` (`67_108_864`) — the materialization budget for the whole
/// `(T', vocab)` tensor [`greedy_ctc_transcribe`] reads.
///
/// The per-axis caps ([`MAX_LOGITS_VOCAB`] and the time axis against
/// [`crate::audio::io::MAX_DECODED_SAMPLES`]) bound each dimension
/// INDEPENDENTLY, but not their product: a lazily-shaped `(T', vocab)` whose
/// dimensions are each individually under their cap can still have a product of
/// terabytes of elements (cheap O(1) lazy metadata via broadcast / outer
/// product), which the `argmax` + `to_vec::<u32>()` would then force `eval` to
/// materialize — a driver-triggered OOM. So [`greedy_ctc_transcribe`] also
/// checks `T' * vocab` (a `checked_mul` off the lazy [`Array::shape`], before
/// any `eval`) against this budget. A real CTC logits is tiny — ~`T'`(~1500 for
/// 30 s of audio) x `vocab`(~32) ≈ 48 K elements — so `64 Mi` is a wide margin
/// that admits every real model yet rejects the product-overflow shape.
pub const MAX_LOGITS_ELEMENTS: usize = 64 * 1024 * 1024;

/// A generous-but-bounded sanity cap on a model-provided
/// [`AutoregressiveStt::max_context`], `128 Ki` (`131_072`).
///
/// [`greedy_transcribe`] derives its decode-loop bound (`max_context -
/// prompt_len`) and the generated-token `Vec`'s growth from `max_context`. A
/// model is free to report any `max_context`, so an absurd value would make the
/// loop effectively unbounded and drive an infallible-`push` `Vec` toward OOM
/// (a never-eot model). Real decoder contexts are at most ~128 K positions, so
/// `128 Ki` admits every real decoder yet rejects an absurd value — and, once
/// capped, the generated-token `Vec` is itself bounded by `max_context`, so its
/// bounded infallible `push` is acceptable.
pub const MAX_DECODE_CONTEXT: usize = 128 * 1024;

/// An absolute backstop on the CUMULATIVE driver `argmax` work across a whole
/// autoregressive decode — the sum of every step's vocab-axis length —
/// `256 Mi` (`268_435_456`) logits-vocab elements.
///
/// The other [`greedy_transcribe`] guards bound each step INDIVIDUALLY (the
/// per-step vocab cap [`MAX_LOGITS_VOCAB`]) and the trip count
/// (`max_context - prompt_len`, with `max_context` capped against
/// [`MAX_DECODE_CONTEXT`]); their product is the worst-case cumulative work.
/// That product (`128 Ki` steps x `256 Ki` vocab = `32 Gi` elements) is a far
/// larger argmax budget than any real decode needs, so even with a caller that
/// sets no [`TranscribeOptions::max_new_tokens`] and a model that never emits
/// [`AutoregressiveStt::eot`], the loop can still spend the full context worth
/// of large-vocab argmaxes. This cap bounds that cumulative work directly: the
/// loop maintains a running sum of the per-step vocab lengths (in the
/// `accumulate_decode_work` helper) and aborts with a typed
/// [`Error::OutOfRange`] the step the sum would exceed this budget, BEFORE that
/// step's `argmax`.
///
/// `256 Mi` is generous for any real STT model: Whisper's worst case is its
/// `448`-slot context x a ~`52 K` vocab ≈ `23 Mi` cumulative elements, so this
/// budget is ~`11x` that headroom — a real decoder stays well under it — while
/// still bounding the never-eot, no-caller-limit denial-of-service.
pub const MAX_AR_DECODE_WORK: usize = 256 * 1024 * 1024;

/// A generous-but-bounded sanity cap on a model-provided
/// [`MelConfig::n_mels`], `4 Ki` (`4096`) filterbank bins.
///
/// [`default_log_mel`] forwards the model's [`MelConfig`] straight into the
/// DSP mel front end, where `n_mels` sizes BOTH the `(n_mels, n_freqs)`
/// filterbank `Vec` and the materialized `(n_mels, num_frames)` mel-output
/// tensor. `n_mels` is model-provided (read off [`MelConfig::n_mels`]), so a
/// hostile model could report an absurd value and force a multi-GB filterbank
/// (and a far larger output tensor) before any of the decode-stage budgets
/// apply. Real mel front ends are small — Whisper uses `80`, canary `128` — so
/// `4 Ki` admits every real model yet rejects a denial-of-service width. This
/// is the per-axis cap; the two product caps ([`MAX_MEL_ELEMENTS`]) bound the
/// derived tensors whose axes each individually pass.
pub const MAX_MEL_BINS: usize = 4 * 1024;

/// A bound on the element count of EITHER materialized mel tensor a
/// [`default_log_mel`] call produces — the `(n_mels, n_freqs)` filterbank or
/// the `(n_mels, num_frames)` mel output — `64 Mi` (`67_108_864`) elements.
/// Mirrors [`MAX_LOGITS_ELEMENTS`] (the CTC product budget).
///
/// The per-axis [`MAX_MEL_BINS`] cap bounds `n_mels` alone, but NOT its
/// products: a model whose `n_mels` and `n_fft` each individually pass can
/// still drive a filterbank (`n_mels * n_freqs`, with `n_freqs = n_fft/2 + 1`)
/// or a mel output (`n_mels * num_frames`) of terabytes of elements — both of
/// which the DSP would `eval`/materialize, a driver-triggered OOM. So
/// [`default_log_mel`] checks BOTH products (via `checked_mul` off the
/// model-provided [`MelConfig`] fields and a safe upper bound on the STFT frame
/// count, before any DSP work) against this budget. A real mel front end is
/// tiny — Whisper's filterbank is `80 * 201 ≈ 16 K` and its 30 s output
/// `80 * ~3000 ≈ 240 K` elements — so `64 Mi` is a wide margin that admits every
/// real model yet rejects the product-overflow shape.
pub const MAX_MEL_ELEMENTS: usize = 64 * 1024 * 1024;

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
///
/// The model-provided [`MelConfig`] magnitudes are capped BEFORE the DSP call —
/// off the plain config fields, so a denial-of-service config is rejected with a
/// typed error with NO mel materialization (the model's `n_mels` sizes both the
/// filterbank and the mel output, neither of which the decode-stage budgets
/// bound):
///
/// - `n_mels == 0` ⇒ [`Error::InvariantViolation`] (a degenerate filterbank);
///   `n_mels > MAX_MEL_BINS` ⇒ [`Error::OutOfRange`];
/// - `n_fft == 0` / `hop_length == 0` ⇒ [`Error::InvariantViolation`] (so the
///   derived frame bounds are well-defined);
/// - the filterbank PRODUCT `n_mels * n_freqs` (with `n_freqs = n_fft/2 + 1`)
///   and the mel-output PRODUCT `n_mels * num_frames` (with a SAFE upper bound
///   on the STFT frame count derived from the validated sample count) each
///   against [`MAX_MEL_ELEMENTS`] via `checked_mul` — so a config whose axes
///   each individually pass but whose product is terabytes of elements is still
///   rejected with no allocation.
pub fn default_log_mel(cfg: &MelConfig, audio: &Array) -> Result<Array> {
  // Validate the waveform metadata first (rank/non-empty/cap); the returned
  // sample count bounds the mel-output frame count below.
  let samples = validate_waveform(audio)?;

  // Cap the model-provided `MelConfig` magnitudes BEFORE forwarding them to the
  // DSP. `default_log_mel` hands `cfg`'s fields straight into the mel front end,
  // where `n_mels` sizes BOTH the `(n_mels, n_freqs)` filterbank `Vec` and the
  // materialized `(n_mels, num_frames)` mel output. The decode-stage budgets
  // (vocab/context/decode-work) only apply AFTER the encoder consumes this mel,
  // so without a driver-owned cap here a hostile model could report an absurd
  // `n_mels` (or an `n_mels`/`n_fft` pair whose products are huge) and force a
  // multi-GB filterbank / terabyte-scale mel output before any later guard runs
  // — the same per-axis-bounded-but-product-unbounded mode the CTC
  // `MAX_LOGITS_ELEMENTS` guard already defends. All reads are off the plain
  // `MelConfig` values (no `eval`); arithmetic is `checked_*`.
  let n_mels = cfg.n_mels();
  let n_fft = cfg.n_fft();
  let hop_length = cfg.hop_length();

  // (1) A zero-bin filterbank is a degenerate config (no mel bands); reject it
  // up front so the products below are well-defined.
  if n_mels == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stt default_log_mel: MelConfig n_mels",
      "must be > 0 (a zero-bin mel filterbank is degenerate)",
    )));
  }
  // (1) Per-axis cap on the filterbank bin count.
  if n_mels > MAX_MEL_BINS {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt default_log_mel: MelConfig n_mels",
      "must not exceed MAX_MEL_BINS (4 Ki)",
      n_mels.to_string(),
    )));
  }
  // (2) `n_fft == 0` / `hop_length == 0` make the derived frame bounds below
  // undefined (a div-by-zero hop / a degenerate FFT). Reject them here — ahead
  // of the DSP, which rejects them too — so the product bounds are well-defined.
  if n_fft == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stt default_log_mel: MelConfig n_fft",
      "must be > 0",
    )));
  }
  if hop_length == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "stt default_log_mel: MelConfig hop_length",
      "must be > 0",
    )));
  }
  // One-sided real-FFT bin count: `n_freqs == n_fft / 2 + 1` (the filterbank's
  // free axis). `n_fft / 2 + 1` cannot overflow `usize` (`n_fft / 2 <= n_fft`).
  let n_freqs = n_fft / 2 + 1;

  // (3) FILTERBANK-product guard: bound the `(n_mels, n_freqs)` filterbank Vec
  // the DSP materializes. The per-axis `n_mels` cap above does NOT bound this
  // product — a legal `n_mels` paired with a large (legal) `n_fft` still yields
  // a multi-Gi filterbank. `checked_mul` off the plain config values rejects an
  // overflow or an over-budget product BEFORE the DSP builds the bank.
  let filterbank_elems = n_mels.checked_mul(n_freqs).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "stt default_log_mel: mel filterbank element count (n_mels * n_freqs)",
      "usize",
      [("n_mels", n_mels as u64), ("n_freqs", n_freqs as u64)],
    ))
  })?;
  if filterbank_elems > MAX_MEL_ELEMENTS {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt default_log_mel: mel filterbank element count (n_mels * n_freqs)",
      "must not exceed MAX_MEL_ELEMENTS (64 Mi)",
      filterbank_elems.to_string(),
    )));
  }

  // (4) MEL-OUTPUT-product guard: bound the `(n_mels, num_frames)` mel tensor
  // the DSP materializes. The DSP's centered STFT (`mlx_audio.dsp.stft`
  // default, see `dsp::stft_with_config`) reflect-pads the signal by `n_fft / 2`
  // each side, so its actual frame count is `num_frames = 1 + samples /
  // hop_length`. We compute a SAFE upper bound from the driver's own values —
  // `num_frames_ub = 1 + (samples + n_fft) / hop_length` — which is `>= ` that
  // actual count (the `+ n_fft` slack only inflates it), so the product bound is
  // never an under-estimate. `checked_*` throughout: an overflow is a typed
  // error, never a wrap.
  let padded_ub = samples.checked_add(n_fft).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "stt default_log_mel: padded length upper bound (samples + n_fft)",
      "usize",
      [("samples", samples as u64), ("n_fft", n_fft as u64)],
    ))
  })?;
  // `+ 1` cannot overflow: `padded_ub / hop_length <= padded_ub < usize::MAX`
  // once `samples + n_fft` did not overflow above.
  let num_frames_ub = padded_ub / hop_length + 1;
  let mel_output_elems = n_mels.checked_mul(num_frames_ub).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "stt default_log_mel: mel output element count (n_mels * num_frames)",
      "usize",
      [
        ("n_mels", n_mels as u64),
        ("num_frames_ub", num_frames_ub as u64),
      ],
    ))
  })?;
  if mel_output_elems > MAX_MEL_ELEMENTS {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt default_log_mel: mel output element count (n_mels * num_frames)",
      "must not exceed MAX_MEL_ELEMENTS (64 Mi)",
      mel_output_elems.to_string(),
    )));
  }

  // Gates passed: hand the ORIGINAL lazy `&Array` straight to
  // `log_mel_spectrogram_with` — it accepts an `Array` and runs its own
  // pre-eval guards, so there is no `to_vec` + rebuild round-trip (no forced
  // materialization, no extra `Vec`).
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
///
/// The model-provided logits magnitudes are also capped — off the lazy
/// [`Array::shape`], so a denial-of-service shape is rejected BEFORE any
/// `argmax`/`to_vec` materialization, each a typed [`Error::OutOfRange`]:
///
/// - the time axis `T'` against BOTH the validated input sample count (a valid
///   feature extractor emits no more frames than it had input samples) AND the
///   absolute [`crate::audio::io::MAX_DECODED_SAMPLES`] cap;
/// - the vocab axis against [`MAX_LOGITS_VOCAB`];
/// - the PRODUCT `T' * vocab` (the total elements the `argmax`/`to_vec` would
///   materialize) against [`MAX_LOGITS_ELEMENTS`], via a `checked_mul` off the
///   lazy shape — so a `(T', vocab)` whose axes each individually pass their cap
///   but whose product is terabytes of elements is still rejected with no
///   allocation.
pub fn greedy_ctc_transcribe<M: CtcModel + ?Sized>(
  model: &M,
  audio: &Array,
  _opts: &TranscribeOptions,
) -> Result<Transcription> {
  // Reject an empty / oversized / multi-rank waveform before the encoder
  // forward (the CTC frontend can reasonably assume a non-empty mono input).
  // Keep the validated sample count: a valid feature extractor emits no more
  // frames `T'` than it had input samples, so `samples` is the natural per-input
  // bound on the model-provided time axis below.
  let samples = validate_waveform(audio)?;

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
  // Cap the model-provided TIME axis `T'` (frame count) BEFORE the `argmax` +
  // `to_vec::<u32>()` below. `shape` is read off the lazy array's metadata (no
  // `eval`), so a model returning a lazily-shaped `(huge_T', vocab)` is rejected
  // here with NO allocation — rather than materializing one `u32` per frame
  // (an OOM). Two bounds apply: the validated input sample count (a valid
  // feature extractor emits no more frames than it had input samples, so a
  // normal-length input cannot be amplified into a huge frame axis) and the
  // absolute `MAX_DECODED_SAMPLES` cap (a hard ceiling regardless of input).
  if shape[0] > samples {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt CtcModel::logits time axis (T', frame count)",
      "must not exceed the input sample count (a valid feature extractor emits \
         no more frames than input samples)",
      shape[0].to_string(),
    )));
  }
  if shape[0] > audio_io::MAX_DECODED_SAMPLES {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt CtcModel::logits time axis (T', frame count)",
      "must not exceed MAX_DECODED_SAMPLES (64 Mi — a valid model emits no more \
         frames than input samples)",
      shape[0].to_string(),
    )));
  }
  // Cap the model-provided VOCAB axis BEFORE the `argmax`, so a lazily-shaped
  // `(T', huge_vocab)` is rejected with no materialization of the argmax over a
  // multi-GB-wide axis.
  if shape[1] > MAX_LOGITS_VOCAB {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt CtcModel::logits vocab axis",
      "must not exceed MAX_LOGITS_VOCAB (256 Ki)",
      shape[1].to_string(),
    )));
  }
  // Empty vocab axis: argmax over an empty axis is undefined — typed error
  // (mirrors the `AutoregressiveStt::decode_step` empty-vocab guard).
  if shape[1] == 0 {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "stt CtcModel::logits returned an empty vocab axis (vocab == 0)",
    )));
  }
  // Cap the PRODUCT `T' * vocab` BEFORE the `argmax` + `to_vec::<u32>()`. The
  // two per-axis caps above bound each dimension independently but NOT their
  // product: a lazily-shaped `(T', vocab)` whose axes each individually pass
  // (e.g. via broadcast / outer-product, O(1) lazy metadata) can still total
  // terabytes of elements, which the `argmax`/`to_vec` would force `eval` to
  // materialize — an OOM. `checked_mul` off the lazy `shape` (no `eval`) rejects
  // an overflow (`None`) or an over-budget product with a typed error and NO
  // allocation, so this is the last shape gate before the materialization below.
  match shape[0].checked_mul(shape[1]) {
    Some(elements) if elements <= MAX_LOGITS_ELEMENTS => {}
    _ => {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "stt CtcModel::logits element count (T' * vocab)",
        "must not overflow and must not exceed MAX_LOGITS_ELEMENTS (64 Mi)",
        format!("{} * {}", shape[0], shape[1]),
      )));
    }
  }
  // Read the model-provided blank id EXACTLY ONCE: `blank_id` is `&self`, so an
  // interior-mutability model could otherwise return one value to the range
  // check below and a different one to the collapse — a TOCTOU that smuggles an
  // unvalidated blank into the decode. The single cached `blank` is the value
  // both validated and used.
  let blank = model.blank_id();
  // The blank id must index into the vocab axis: a `blank` outside `[0, vocab)`
  // can never equal a per-frame `argmax` (which is always in range), so its
  // blank frames would survive the collapse and be fed to the infallible
  // `decode_ids` — silent bad text (or an index panic in a real vocab map).
  // Reject it up front with a typed error carrying the cached value.
  if (blank as usize) >= shape[1] {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt CtcModel::blank_id",
      "must be < the logits vocab size (a blank id outside the vocab axis \
         leaves blank frames un-collapsed)",
      blank.to_string(),
    )));
  }
  // Empty time axis `(0, vocab)`: no frames to collapse → an empty
  // transcription (an explicit, non-panicking definition). Route the empty
  // collapse through `decode_ids(&[])` — exactly as the all-blank path below
  // does — so the two semantically-identical "no surviving ids" inputs produce
  // identical text (a model whose `decode_ids(&[])` emits a sentinel must not
  // disagree between an empty-time and an all-blank input). The blank-id range
  // check above already precedes this branch, so reaching `decode_ids` here is
  // sound.
  if shape[0] == 0 {
    let text = model.decode_ids(&[]);
    let segments = vec![Segment::new(text.clone(), 0.0, 0.0)];
    return Ok(Transcription::new(text, None, segments));
  }

  // Per-frame argmax over the vocab axis → `(T',)` class ids.
  let mut frame_ids = ops::misc::argmax(&logits, Some(1), false)?;
  let ids = frame_ids.to_vec::<u32>()?;

  // Greedy CTC collapse: drop consecutive duplicates, then drop the blank.
  // Reuses the `blank` validated above — never re-calls `model.blank_id()`, so
  // the collapsed-out id is exactly the one that passed the range check.
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
///    and continue. The [`AutoregressiveStt::eot`] id is read once and
///    range-checked against EVERY step's actual vocab axis (an `eot` outside
///    `[0, vocab)` can never be produced by `argmax`, so it is a typed
///    [`Error::OutOfRange`] rather than a never-stopping loop). The check runs
///    per step — not once — so an interior-mutable model that shrinks its vocab
///    on a later step (making `eot` out of range there) is still rejected
///    rather than looping to `max_context`.
///
/// The loop is bounded by the model's [`AutoregressiveStt::max_context`]: it
/// generates AT MOST `max_context - prompt.len()` new tokens, so
/// `prompt + generated` never exceeds the decoder's total positional context
/// (a prompt that already meets or exceeds `max_context` is a typed
/// [`Error::OutOfRange`] — there is no room left to decode). A caller may lower
/// that bound further via [`TranscribeOptions::max_new_tokens`]: the loop then
/// generates at most `min(max_context - prompt.len(), max_new_tokens)` (a
/// caller limit larger than the remaining context is harmlessly clamped to the
/// context; `None` uses the full context). The model-provided `max_context` is
/// itself capped against [`MAX_DECODE_CONTEXT`] BEFORE the loop (a typed
/// [`Error::OutOfRange`]), so an absurd value can neither drive an
/// effectively-unbounded loop nor an unbounded generated-token `Vec` (its
/// growth is then bounded by `max_context`, so the per-step `push` is a bounded
/// infallible append). Each step's `(vocab,)` logits width is likewise capped
/// against [`MAX_LOGITS_VOCAB`] before its `argmax`.
///
/// As an absolute backstop independent of the caller, the CUMULATIVE driver
/// `argmax` work — the running sum of every step's vocab-axis length — is
/// bounded by [`MAX_AR_DECODE_WORK`] (via the `accumulate_decode_work` helper),
/// checked each step BEFORE that step's `argmax`. This bounds the total argmax
/// work a
/// never-eot model can drive even when the caller sets no `max_new_tokens`
/// (the per-step vocab cap x the `max_context` trip count is otherwise a large
/// cumulative budget); over-budget (or a `usize` overflow of the accumulator)
/// is a typed [`Error::OutOfRange`].
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
  // Driver-owned preflight, BEFORE any model frontend call. `log_mel` is
  // overrideable, so a custom frontend that skips its own waveform validation
  // could otherwise eval/copy a rank-2 / empty / oversized `Array` before this
  // shared gate; and an over-context prompt must be rejected before the full
  // frontend + encode is spent. So validate the waveform metadata and the
  // prompt length here — ahead of `log_mel`, `encode`, and `new_cache`.
  validate_waveform(audio)?;

  let mut tokens = m.initial_tokens(opts)?;
  let prompt_len = tokens.len();

  // Bound `prompt + generated` by the decoder's TOTAL context: a prompt that
  // already fills (or overflows) `max_context` leaves no room to decode.
  let max_ctx = m.max_context();
  // Cap the model-provided `max_context` BEFORE deriving `max_new` / entering
  // the loop. `max_context` is model-provided and otherwise only compared to
  // `prompt_len`, so an absurd value makes `max_new = max_ctx - prompt_len` an
  // effectively unbounded decode with infallible `tokens.push` growth — an OOM
  // (a never-eot model). Rejecting it here with a typed error bounds both the
  // loop trip count and the generated-token `Vec` by `MAX_DECODE_CONTEXT`.
  if max_ctx > MAX_DECODE_CONTEXT {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt AutoregressiveStt::max_context",
      "must not exceed MAX_DECODE_CONTEXT (128 Ki)",
      max_ctx.to_string(),
    )));
  }
  if prompt_len >= max_ctx {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt greedy_transcribe: initial_tokens prompt length",
      "must be < the decoder's max_context (prompt exceeds decoder context, \
         leaving no room to generate)",
      prompt_len.to_string(),
    )));
  }

  // Gates passed: run the model frontend → encode → fresh cache, then decode.
  let mel = m.log_mel(audio)?;
  let enc = m.encode(&mel)?;
  let mut cache = m.new_cache();
  // Read the model-provided end-of-transcript id EXACTLY ONCE. `eot` is `&self`
  // and the loop compares every `argmax` against it; capturing it in a local
  // means a later mutation of the model can't swap the stop token mid-decode.
  // The VALUE is read once here; its range-check (below) runs per step against
  // that step's own vocab axis.
  let eot = m.eot();

  // The decoder-context bound on new tokens, so the total never exceeds
  // `max_ctx`. A caller-supplied `max_new_tokens` further lowers it (clamped to
  // this bound, so a caller limit larger than the remaining context is
  // harmlessly capped to the context).
  let cap_new = max_ctx - prompt_len;
  let max_new = opts.max_new_tokens().map_or(cap_new, |n| n.min(cap_new));

  // Cumulative driver-`argmax` work (sum of every step's vocab length). The
  // per-step vocab cap + the `max_new` trip count already bound this, but their
  // PRODUCT can still be a huge cumulative argmax budget for a never-eot model
  // with no caller limit; this accumulator caps the cumulative work directly
  // (checked each step BEFORE that step's argmax, see `accumulate_decode_work`).
  let mut decode_work: usize = 0;

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
    let vocab_len = shape[0];
    if vocab_len == 0 {
      return Err(Error::EmptyInput(EmptyInputPayload::new(
        "stt AutoregressiveStt::decode_step returned an empty vocab axis (vocab == 0)",
      )));
    }
    // Cap THIS step's model-provided vocab axis BEFORE the `argmax` below, so a
    // lazily-shaped `(huge_vocab,)` row is rejected with no materialization of
    // an argmax over a multi-GB-wide axis. Re-checked per step (like the `eot`
    // range check) so an interior-mutable model that returns an absurd width on
    // a later step is still caught.
    if vocab_len > MAX_LOGITS_VOCAB {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "stt AutoregressiveStt::decode_step vocab axis",
        "must not exceed MAX_LOGITS_VOCAB (256 Ki)",
        vocab_len.to_string(),
      )));
    }

    // Range-check the cached `eot` against THIS step's vocab axis, every step.
    // An `eot >= vocab_len` could never be produced by `argmax`, so the loop's
    // `next == eot` stop condition would never fire — reject it with a typed
    // error instead of running to `max_context` and returning bogus output.
    // Re-checking per step (not once via a latch) defends against an
    // interior-mutable model that returns a large vocab on an early step
    // (passing) then a SMALLER vocab on a later step where `eot` is out of
    // range. It is a single `usize` compare per step (negligible); a
    // consistent-vocab model never triggers it.
    if (eot as usize) >= vocab_len {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        "stt AutoregressiveStt::eot",
        "must be < the decode_step logits vocab size (an eot outside the \
           vocab axis can never be produced by argmax, so the greedy loop \
           would never stop)",
        eot.to_string(),
      )));
    }

    // Accumulate this step's `argmax` cost into the cumulative decode-work
    // budget BEFORE the `argmax` below. Even with every per-step + trip-count
    // guard above, a never-eot model and a caller that set no `max_new_tokens`
    // could otherwise spend `max_context` steps x a large per-step vocab of
    // cumulative argmax work; `accumulate_decode_work` caps that running sum
    // against `MAX_AR_DECODE_WORK` and returns a typed error (on over-budget or
    // `usize` overflow) so the loop aborts here rather than at `max_context`.
    decode_work = accumulate_decode_work(decode_work, vocab_len)?;

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

/// Add one decode step's vocab-axis length `step_vocab` to the running
/// cumulative-`argmax`-work accumulator `work`, enforcing the
/// [`MAX_AR_DECODE_WORK`] backstop.
///
/// Returns the new accumulated total on success, or a typed
/// [`Error::OutOfRange`] when the addition overflows `usize` OR the new total
/// would exceed [`MAX_AR_DECODE_WORK`]. [`greedy_transcribe`] calls this each
/// step with that step's vocab length (read off the lazy [`Array::shape`]) and
/// aborts on the error BEFORE the step's `argmax`, so the cumulative driver
/// argmax work is bounded even when the caller sets no
/// [`TranscribeOptions::max_new_tokens`] and the model never emits
/// [`AutoregressiveStt::eot`].
///
/// Factored out as a pure function so the budget arithmetic is unit-testable
/// without running a real (giant) decode loop.
fn accumulate_decode_work(work: usize, step_vocab: usize) -> Result<usize> {
  match work.checked_add(step_vocab) {
    Some(total) if total <= MAX_AR_DECODE_WORK => Ok(total),
    _ => Err(Error::OutOfRange(OutOfRangePayload::new(
      "stt greedy_transcribe: cumulative decode work (sum of per-step vocab lengths)",
      "must not overflow and must not exceed MAX_AR_DECODE_WORK (256 Mi)",
      work.to_string(),
    ))),
  }
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
