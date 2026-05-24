//! End-to-end STT generation: load audio → optional resample → log-mel
//! spectrogram → encoder → token-by-token cross-attention decode loop.
//!
//! Ported in shape from mlx-audio's [`stt/generate.py`][stt-gen] (the model-
//! agnostic top-level entry point) and the per-model decode loops
//! ([`whisper/whisper.py`][whisper], [`parakeet/parakeet.py`][parakeet] —
//! consulted for the cross-attention shape, NOT for the per-model
//! algorithm: greedy / beam / RNN-T expansion / segment alignment etc. live
//! in per-model code per the
//! [`project_no_per_model_arch_porting`][noarch] rule).
//!
//! [`stt_generate`] composes [`crate::audio::io::load_audio`],
//! [`crate::audio::io::resample_linear`],
//! [`crate::audio::dsp::log_mel_spectrogram`], the
//! [`super::model::Model`] trait, and the LM's sampler / logits-processor
//! chain ([`crate::lm::generate::make_sampler`] /
//! [`crate::lm::generate::make_logits_processors`]) into one
//! [`Iterator<Item = Result<crate::lm::generate::GenStep>>`][iter] — the
//! same step-by-step contract the [LM loop][crate::lm::generate::generate_step]
//! exposes, so callers familiar with the LM loop see no new step shape.
//!
//! No implicit eval: every `Array` op is a pure [`crate::ops`] composition
//! returning `Result`; the only materialization is the token boundary
//! `.item::<u32>()` the inner LM-side generator handles —
//! [`stt_generate`] never materializes the encoder states or the logits
//! itself.
//!
//! ## `wired_limit` / generation-stats parity (audit, AUDIO-A13)
//!
//! mlx-audio's `generate_transcription` (`stt/generate.py:272-413`) wraps
//! per-model decoding in a `wired_limit(model, [generation_stream])` context
//! manager and emits per-run `STTOutput` timing fields (`prompt_tokens`,
//! `generation_tokens`, `prompt_tps`, `generation_tps`, `total_time`).
//! mlxrs's [`stt_generate`] is the **iterator-shaped** analogue (mirroring
//! [`crate::lm::generate::generate_step`], NOT the higher-level
//! [`crate::lm::generate::stream_generate`] that aggregates into
//! [`crate::lm::generate::GenerationResponse`]); both `wired_limit`
//! integration and per-run [`crate::lm::generate::GenerationStats`]-shaped
//! aggregation are intentionally deferred to a coordinated LM/STT follow-up
//! (mlxrs-safe has no `set_wired_limit` wrapper yet — `mlxrs_sys::
//! mlx_set_wired_limit` exists but no `crate::memory::set_wired_limit`
//! surface fn does, and `mlx_device_info_get` for
//! `max_recommended_working_set_size` is also un-wrapped — both gaps are
//! shared with the LM side and live with the LM L6 follow-up). The
//! detailed audit + rationale is in [`super::serializers`] — that module's
//! header carries the canonical write-up so this loop's doc stays focused
//! on the decode-step contract.
//!
//! [stt-gen]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/generate.py
//! [whisper]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/whisper/whisper.py
//! [parakeet]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/parakeet.py
//! [noarch]: https://github.com/uqio/mlxrs/blob/mlx/docs/superpowers/conventions/no-per-model-arch-porting.md
//! [iter]: core::iter::Iterator

use std::path::Path;

use crate::{
  array::Array,
  audio::{dsp, io as audio_io},
  error::{Error, Result, try_extend_from_slice},
  lm::{
    cache::KvCache,
    generate::{
      GenConfig, GenStep, LogitsProcessor, Sampler, make_logits_processors, make_sampler,
    },
  },
  ops,
};

/// Default maximum audio duration accepted by [`stt_generate`] when no
/// override is supplied — `30.0` seconds, mlx-audio whisper's per-segment
/// context size.
pub const DEFAULT_MAX_AUDIO_SECONDS: f32 = 30.0;

/// STT generation config — wraps [`crate::lm::generate::GenConfig`] with
/// audio-specific knobs.
///
/// Composition (not inheritance) lets the LM loop's sampler / penalty /
/// max-tokens knobs be reused verbatim without re-declaring every field; the
/// audio-specific fields are pre-allocation safety knobs:
///
/// - `auto_resample` — if the WAV's source sample rate differs from
///   [`super::model::Model::mel_config`]'s `sample_rate`, run a
///   [`crate::audio::io::resample_linear`] pass before mel-spec extraction.
///   Default `true` — matches the standard mlx-audio whisper preprocessing
///   path (which assumes a 16 kHz input and resamples otherwise).
/// - `max_audio_seconds` — reject inputs longer than this (recoverable
///   [`Error::Backend`]). Default [`DEFAULT_MAX_AUDIO_SECONDS`] = 30 s. The
///   check runs against the **source** duration immediately after
///   `load_audio`, BEFORE the resample, mel-spectrogram, and encoder passes
///   allocate — so a crafted / fuzz input claiming long audio cannot drive
///   a multi-GB allocation through the STT pipeline. The **load-stage
///   ceiling** is a separate cap inside `audio::io::load_audio`
///   (`MAX_DECODED_SAMPLES` = 64 Mi samples ≈ 256 MiB, ~17 min of 16 kHz
///   mono) — `max_audio_seconds` is the layered STT-pipeline cap on top
///   of that, NOT a replacement for it.
pub struct SttGenConfig {
  /// LM loop config (sampler, max tokens, prefill chunk size, …).
  pub lm: GenConfig,
  /// Resample the input WAV to [`super::model::Model::mel_config`]'s
  /// `sample_rate` when the source rate differs. Default `true`.
  pub auto_resample: bool,
  /// Maximum accepted audio duration in seconds; inputs longer than this
  /// return [`Error::Backend`] **before** mel-spectrogram allocation.
  /// Default [`DEFAULT_MAX_AUDIO_SECONDS`] (30 s, mlx-audio whisper segment).
  pub max_audio_seconds: f32,
}

impl Default for SttGenConfig {
  fn default() -> Self {
    Self {
      lm: GenConfig::default(),
      auto_resample: true,
      max_audio_seconds: DEFAULT_MAX_AUDIO_SECONDS,
    }
  }
}

/// Build the `Array` mel-spectrogram from a WAV file path, validating the
/// duration cap **before** the mel-spectrogram allocation. Used by both
/// [`stt_generate`] and [`encode_audio_file`] so the load → resample →
/// max-seconds-check → log-mel pipeline is implemented once.
fn audio_path_to_mel<M: super::model::Model>(
  model: &M,
  audio_path: &Path,
  cfg: &SttGenConfig,
) -> Result<Array> {
  // 0. Validate `max_audio_seconds` UP FRONT — before any filesystem or
  //    decode work — so a malformed cap (NaN / ±inf / zero / negative)
  //    surfaces as the recoverable `Err` the public docs promise and a
  //    `samples.len() / sample_rate <= 0.0` comparison can't silently
  //    reject every input.
  //
  //    Two positive guards (`is_finite() && > 0.0`) instead of the
  //    NaN-catching `!(x > 0.0)`: clippy's `neg_cmp_op_on_partial_ord`
  //    forbids the negated-comparison shorthand on `f32` because NaN
  //    makes the negated form non-trivially equivalent. `is_finite()`
  //    covers both NaN and ±inf, `> 0.0` covers zero/negative.
  if !cfg.max_audio_seconds.is_finite() || cfg.max_audio_seconds <= 0.0 {
    return Err(Error::Backend {
      message: format!(
        "stt_generate: `max_audio_seconds` must be a finite value > 0 (got {})",
        cfg.max_audio_seconds
      ),
    });
  }

  // 1. Load. `load_audio` decodes WAV / MP3 / FLAC / OGG-Vorbis (the
  //    format is auto-detected from the file content). The mlxrs
  //    pipeline uses a **layered resource cap**:
  //    - `load_audio` rejects upfront when a container's declared frame
  //      count exceeds `MAX_DECODED_SAMPLES` = 64 Mi samples ≈ 256 MiB
  //      at 4 B / f32 (~17 minutes of 16 kHz mono, ~25 minutes of
  //      44.1 kHz mono), and caps the running decoded length at that
  //      same ceiling for compressed inputs that omit / under-estimate
  //      their length. That is the absolute load-stage ceiling.
  //    - `max_audio_seconds` (default 30 s, the whisper segment size)
  //      is the STT-pipeline cap; it rejects audio whose source
  //      duration exceeds the per-utterance limit before the resample
  //      + mel + encode passes allocate.
  //
  //    The layered cap is applied at the LOAD stage via
  //    `load_audio_with_max_seconds`: it probes the container's source
  //    sample rate FIRST, then enforces the load-stage cap as
  //    `src_sr * max_audio_seconds` (clamped to
  //    `MAX_DECODED_SAMPLES`). For exact-count formats (WAV / FLAC-with-
  //    STREAMINFO) whose container header declares a sample-exact total
  //    the rejection fires BEFORE allocating the sample buffer; for
  //    lossy formats (MP3 / OGG-Vorbis / FLAC-without-STREAMINFO) the
  //    cap also bounds the per-decoded-buffer push, so the wall-time
  //    cost of partial decode is bounded by `max_audio_seconds *
  //    src_sr` worth of decoded f32 frames, not the full 256 MiB
  //    load-stage ceiling.
  //
  //    Source-rate cap (NOT target-rate): a Codex R1 high finding
  //    flagged that deriving the cap from the model's target sample
  //    rate spuriously rejected valid auto-resample inputs whose
  //    `src_sr > target_sr` (e.g. a 1.0 s 44.1 kHz WAV at a 16 kHz
  //    model with `max_audio_seconds = 1.0` — declared 44 100 source
  //    samples vs `target_sr * 1.0 = 16 000` cap). Probing the source
  //    rate first and capping by `src_sr * max_audio_seconds` keeps
  //    every input whose source duration is `<= max_audio_seconds`
  //    decodable regardless of the model's resample target.
  //    Closes the AUDIO-11 layered-cap gap.
  //
  // Snapshot the model's mel config ONCE (Codex #64 finding): the same
  // `mc` drives the resample target rate (step 3) and the log-mel
  // parameters (step 6). Calling `model.mel_config()` multiple times
  // risks subtle skew if a model computes it dynamically. The load-
  // stage cap is now source-rate-driven (handled inside
  // `load_audio_with_max_seconds`), so `mc.sample_rate` no longer
  // appears in the load-stage budget here.
  let mc = model.mel_config();
  let (samples, src_sr) = audio_io::load_audio_with_max_seconds(audio_path, cfg.max_audio_seconds)?;

  // 2. Duration cap — checked against the **source** duration (load_audio's
  //    `samples.len() / src_sr`) BEFORE resampling allocates a second
  //    buffer. The source duration is the ground truth: resampling can
  //    only refactor the same audio span into a different sample count, so
  //    a long-source over-cap input MUST reject here, before the resample
  //    pass. Avoids the post-resample-only check Codex flagged: a source
  //    just-over-cap could be truncated by `resample_linear`'s
  //    `floor(in * to / from)` and silently pass.
  //
  //    f64 arithmetic for the comparison (cap is `f32`, but the
  //    `samples_len * sr` product can reach `~10^14` at the load_audio cap;
  //    `f64` mantissa carries it exactly, `f32` would round it). The
  //    `> cfg.max_audio_seconds as f64` lift keeps both sides in f64 for
  //    the comparison.
  let cap_f64 = f64::from(cfg.max_audio_seconds);
  let src_duration = samples.len() as f64 / f64::from(src_sr);
  if src_duration > cap_f64 {
    return Err(Error::Backend {
      message: format!(
        "stt_generate: audio duration {src_duration:.3}s (source sample_rate \
         {src_sr}) exceeds `max_audio_seconds` cap {:.3}s (samples={}); reject \
         before resample / mel-spec allocation",
        cfg.max_audio_seconds,
        samples.len()
      ),
    });
  }

  // `mc` was snapshotted ONCE above (Codex #64 finding: calling
  // `model.mel_config()` twice risks subtle skew if a model computes it
  // dynamically, and duplicates the work). It drives the resample target
  // rate (step 3) and the log-mel parameters (step 6). The load-stage
  // cap is now source-rate-driven and lives inside
  // `audio_io::load_audio_with_max_seconds`, so `mc.sample_rate` no
  // longer participates in the load budget.

  // 3. Resample. `mc.sample_rate` is what the model's feature extractor
  //    was trained on; `resample_linear` is a verbatim copy when the
  //    rates match (no FP drift) and a naive linear pass otherwise (the
  //    `mlx-audio` default for Whisper-style models). `auto_resample` off
  //    + mismatched rates surfaces as a recoverable `Error::Backend` so a
  //    misconfigured pipeline cannot silently feed wrong-rate mels to the
  //    model.
  let target_sr = mc.sample_rate;
  let samples: Vec<f32> = if src_sr == target_sr {
    samples
  } else if cfg.auto_resample {
    audio_io::resample_linear(&samples, src_sr, target_sr)?
  } else {
    return Err(Error::Backend {
      message: format!(
        "stt_generate: audio sample rate {src_sr} != model.mel_config().sample_rate \
         {target_sr} but `cfg.auto_resample` is false; enable auto_resample or pre-resample"
      ),
    });
  };

  // 4. Reject empty (or otherwise too-short-to-frame) audio. Fabricating
  //    an `[n_mels, 0]` zero-frame mel would silently feed an invalid
  //    shape into `model.encode_audio` — concrete encoders can reasonably
  //    assume at least one frame and panic / fail deep in per-model code
  //    on a zero-T input. Surface the empty-WAV case as a clear pipeline
  //    `Error::Backend` here (Codex round-1 medium); too-short-but-non-
  //    empty inputs are caught downstream by `log_mel_spectrogram`'s own
  //    reflect-pad guards, which already return a recoverable `Err` with
  //    a descriptive message.
  if samples.is_empty() {
    return Err(Error::Backend {
      message: format!(
        "stt_generate: audio input is empty (0 samples after load{}) — \
         `model.encode_audio` requires at least one mel frame; provide a \
         non-empty WAV",
        if src_sr == target_sr {
          ""
        } else {
          " + resample"
        }
      ),
    });
  }

  // 5. Build an Array from the f32 buffer. `samples.len()` fits i32 because
  //    `load_audio`'s `MAX_DECODED_SAMPLES = 64 Mi` and `resample_linear`'s
  //    `MAX_RESAMPLED_SAMPLES = 64 Mi` are both well below `i32::MAX`.
  let n_samples = i32::try_from(samples.len()).map_err(|_| Error::Backend {
    message: format!(
      "stt_generate: samples.len() {} exceeds i32::MAX",
      samples.len()
    ),
  })?;
  let samples_arr = Array::from_slice::<f32>(&samples, &[n_samples])?;

  // 6. log-mel spectrogram. Output shape `(n_mels, T)` per the
  //    mlx-audio / Whisper canonical layout — fed straight into
  //    `model.encode_audio`. Threads `mc.log_floor` through the
  //    `_with` variant so a Kaldi/Custom-floor model (AUDIO-5 LogFloor)
  //    is encoded with its own floor instead of the hard-coded Whisper
  //    `1e-10` (Codex bundle-#64 finding). Reuses the `mc` snapshot
  //    taken once at the top of this function.
  dsp::log_mel_spectrogram_with(
    &samples_arr,
    mc.n_fft,
    mc.hop_length,
    mc.win_length,
    mc.n_mels,
    mc.sample_rate,
    mc.f_min,
    mc.f_max,
    mc.log_floor,
  )
}

/// The [`Iterator`] returned by [`stt_generate`]: borrows the model and
/// owns the encoder states, the per-layer KV cache, the sampler, the
/// logits processors, and the running token / step counters. Yields one
/// [`GenStep`] per decode step (the same step type the [LM
/// loop][crate::lm::generate] yields, so callers familiar with the LM loop
/// see no new step shape).
///
/// Lifetime `'a` ties to the borrowed model (same pattern as the LM-side
/// generator returned by [`crate::lm::generate::generate_step`]); the
/// cache is owned so the iterator fully owns the in-flight decode state.
///
/// The iterator **fuses**: after it yields `Err` (a step failed) or
/// finishes (eos / `max_tokens`) every further `next()` is `None` — never a
/// panic, never a re-entry into the model (the same `done` flag pattern the
/// LM loop uses).
pub struct SttGenerator<'a, M: super::model::Model> {
  model: &'a M,
  /// The output of [`super::model::Model::encode_audio`] — passed
  /// unchanged into every [`super::model::Model::decode_step`] call (one
  /// encode pass per utterance, faithful to mlx-audio whisper's
  /// `audio_features = self.encoder(mel)` once before the decoding loop).
  encoder_states: Array,
  /// One [`KvCache`] per decoder layer (typically the LM
  /// [`crate::lm::cache::make_prompt_cache`] output for a whisper-style
  /// self-attention decoder; per-model code may pre-populate cross-attn
  /// caches here too — the trait is opaque to the cache list's shape).
  cache: Vec<Box<dyn KvCache>>,
  sampler: Sampler,
  processors: Vec<LogitsProcessor>,
  /// The running token history fed to the logits processors (mirrors the
  /// LM loop's `history` Vec — extended with each step's input token before
  /// processors run, so penalty processors see the same history shape).
  history: Vec<u32>,
  /// The most-recently sampled token; seeded with
  /// [`super::model::Model::bos_token`] on the first step (mlx-audio
  /// whisper's `tokens[0] == sot`).
  last: u32,
  /// Tokens yielded so far (LM-loop equivalent of `n`); generation ends at
  /// `max_tokens`.
  produced: usize,
  max_tokens: usize,
  /// Stop-token set: per-model `eos_token()` plus any
  /// [`GenConfig::eos`][crate::lm::generate::GenConfig::eos] caller override
  /// (the union, so a caller can add task-specific stop tokens — e.g.
  /// whisper's `<|endoftext|>` plus a custom timestamp-end marker — without
  /// dropping the model's own EOS).
  eos: Vec<u32>,
  /// Fused: set after a yielded `Err` or a finish so the iterator never
  /// re-enters the model.
  done: bool,
}

impl<M: super::model::Model> SttGenerator<'_, M> {
  /// One decode step — mirrors the LM loop's per-step shape (forward then
  /// last-position slice then history-accumulate then logits processors
  /// then `logits - logsumexp` then sampler then `GenStep`) but routes
  /// through [`super::model::Model::decode_step`] which already returns the
  /// `[1, V]` last-position logits directly (so no last-position-slice step
  /// is needed).
  fn step(&mut self) -> Result<GenStep> {
    // 1. decode step. The model returns `[1, V]` logits directly — STT
    //    decoders are autoregressive token-at-a-time, so there is no
    //    "prefill chunk" / "[B, S, V]" intermediate the LM loop has.
    let logits = self
      .model
      .decode_step(self.last, &self.encoder_states, &mut self.cache)?;

    // Validate the returned logits shape `[1, V]` (mirrors the LM loop's
    // `last_position` rank/zero-axis check). A `decode_step` impl returning
    // anything else is a per-model defect; surface it as a recoverable
    // `Err` rather than letting `apply_logit_bias` / `logsumexp` produce a
    // confusing downstream error.
    let shape = logits.shape();
    if shape.len() != 2 || shape[0] != 1 || shape[1] == 0 {
      return Err(Error::ShapeMismatch {
        message: format!(
          "stt_generate: `decode_step` must return [1, V] logits with V >= 1, got {shape:?}"
        ),
      });
    }

    // 2. accumulate the step's *input* token (the previously-sampled
    //    token, i.e. `self.last`) into the running history before running
    //    processors — same shape as the LM loop's
    //    `history.extend_from_slice(input)` + per-processor application
    //    over the FULL history on RAW logits.
    let mut logits = logits;
    if !self.processors.is_empty() {
      try_extend_from_slice(&mut self.history, &[self.last])?;
      for p in &self.processors {
        logits = p.apply(&self.history, &logits)?;
      }
    }

    // 3. `logprobs = logits - logsumexp(logits, keepdims=True)` — the exact
    //    LM-loop normalization (mlx-lm's `generate_step` line 416).
    let lse = ops::reduction::logsumexp(&logits, true)?;
    let logprobs = ops::arithmetic::subtract(&logits, &lse)?;

    // 4. sample. The sampler chain is built by `make_sampler` from
    //    `self.lm` (the LM loop's exact sampler composition); the default
    //    `temp == 0` resolves to the deterministic argmax sampler.
    let mut sampled = self.sampler.sample(&logprobs)?;

    // 5. token boundary — the only materialization (LM loop's `y.item()`).
    //    `argmax` / `categorical` both yield `U32`.
    let token: u32 = sampled.item::<u32>()?;

    // mlx-lm returns `logprobs.squeeze(0)` ⇒ `[V]` vector. Kept lazy.
    // L3 `GenStep.logprobs` is `Option<Array>`: audio STT has not adopted
    // the [`crate::lm::generate::GenConfig::collect_logprobs`] opt-in,
    // so we always emit `Some` to preserve the prior unconditional yield.
    let logprobs = ops::shape::squeeze_axes(&logprobs, &[0])?;
    Ok(GenStep {
      token,
      logprobs: Some(logprobs),
    })
  }
}

impl<M: super::model::Model> Iterator for SttGenerator<'_, M> {
  type Item = Result<GenStep>;

  fn next(&mut self) -> Option<Self::Item> {
    // Fused: a prior Err or a finish ends iteration permanently — no
    // panic, no re-entry into the model.
    if self.done {
      return None;
    }

    // Exactly `max_tokens` tokens (LM loop's "length" finish):
    // `if n == max_tokens: break` BEFORE the yield.
    if self.produced >= self.max_tokens {
      self.done = true;
      return None;
    }

    match self.step() {
      Ok(step) => {
        self.produced += 1;
        let token = step.token;
        self.last = token;
        // Same "yield EOS then fuse" pattern the LM loop uses — the EOS
        // token IS yielded so callers can decode it through their own
        // detokenizer; iteration ends after.
        if self.eos.contains(&token) {
          self.done = true;
        }
        Some(Ok(step))
      }
      Err(e) => {
        // A step error is yielded once, then the iterator ends.
        self.done = true;
        Some(Err(e))
      }
    }
  }
}

/// Start an end-to-end STT generation run.
///
/// Pipeline (mlx-audio whisper / parakeet shape):
/// 1. [`crate::audio::io::load_audio`] (mono `Vec<f32>` in `[-1, 1]`).
/// 2. Optional [`crate::audio::io::resample_linear`] when the source sample
///    rate differs from [`super::model::Model::mel_config`]'s `sample_rate`
///    (gated by [`SttGenConfig::auto_resample`]).
/// 3. [`SttGenConfig::max_audio_seconds`] cap (checked BEFORE the mel-
///    spectrogram allocation; rejects crafted long-duration inputs).
/// 4. [`crate::audio::dsp::log_mel_spectrogram`] using the model's
///    [`super::model::Model::mel_config`]; output shape `(n_mels, T)`.
/// 5. [`super::model::Model::encode_audio`] — one pass, cached on the
///    returned [`SttGenerator`].
/// 6. Token-by-token decode via [`super::model::Model::decode_step`] (seeded
///    with [`super::model::Model::bos_token`]); sampled via the LM loop's
///    [`make_sampler`] / [`make_logits_processors`] chain — so every LM
///    sampler / penalty knob is available verbatim through
///    [`SttGenConfig::lm`].
///
/// Returns an [`Iterator`]`<Item = Result<GenStep>>` — the same per-step
/// contract the LM loop returns. Iteration ends on the EOS token (the
/// union of [`super::model::Model::eos_token`] and the
/// [`GenConfig::eos`][crate::lm::generate::GenConfig::eos] override; the
/// EOS token IS yielded as the final step) or after
/// [`GenConfig::max_tokens`][crate::lm::generate::GenConfig::max_tokens]
/// tokens have been produced.
///
/// Any step error is yielded once as `Err`, after which the iterator ends
/// (no panic, no re-entry into the model — the same fused-iterator
/// contract the LM loop guarantees).
///
/// `cache` is the per-layer KV cache (typically
/// [`crate::lm::cache::make_prompt_cache`] for self-attention-only
/// decoders; per-model code that pre-populates cross-attention caches
/// passes them here). The `'a` lifetime on the model borrow + the owned
/// cache means no aliasing.
pub fn stt_generate<'a, M: super::model::Model>(
  model: &'a M,
  audio_path: &Path,
  cache: Vec<Box<dyn KvCache>>,
  cfg: SttGenConfig,
) -> Result<SttGenerator<'a, M>> {
  // Build the sampler / logits-processor chain FIRST — BEFORE the
  // expensive audio load + resample + mel + `encode_audio` pipeline, so a
  // gen config that `make_sampler` / `make_logits_processors` rejects at
  // BUILD time (e.g. a logit_bias index/value-arity mismatch) fails fast
  // from the constructor without paying the audio/encode cost (Codex #64
  // finding). NOTE: `make_sampler` validates only some constraints
  // eagerly; purely-scalar bounds (`temp < 0`, `min_p > 1`,
  // `xtc_probability` out of range, a negative penalty) are checked
  // INSIDE the sampler/processor closure when it first runs against
  // logits — so those still surface on the iterator's first decode step,
  // exactly as in `lm::generate::generate_step` (the STT loop mirrors the
  // LM loop's deferred-runtime-validation behavior, NOT a divergent
  // audio-only eager pass). A fully-eager `GenConfig::validate()` shared
  // by both loops is tracked as a coordinated follow-up (AUDIO-12). Built
  // by reference from `cfg.lm` so `cfg` stays intact for
  // `audio_path_to_mel`; the owned fields are moved out afterward.
  let (sampler, processors) = {
    let lm = &cfg.lm;
    let sampler = make_sampler(
      lm.temp,
      lm.top_p,
      lm.min_p,
      lm.min_tokens_to_keep,
      lm.top_k,
      lm.xtc_probability,
      lm.xtc_threshold,
      &lm.xtc_special_tokens,
      lm.seed,
    )?;
    let processors = make_logits_processors(
      &lm.logit_bias,
      lm.repetition_penalty,
      lm.repetition_context_size,
      lm.presence_penalty,
      lm.presence_context_size,
      lm.frequency_penalty,
      lm.frequency_context_size,
    )?;
    (sampler, processors)
  };

  // Now the (potentially expensive) audio pipeline — steps 1-6 of the doc
  // above. Eager (not deferred) so audio errors surface as the
  // constructor's `Result` ("WAV file not found" / "audio too long")
  // without the caller having to poll `.next()`.
  let mel = audio_path_to_mel(model, audio_path, &cfg)?;
  let encoder_states = model.encode_audio(&mel)?;

  // Move the owned LM fields out of `cfg` for the iterator (the eos
  // override `Vec<u32>` consumed without a clone — by-value-consume style
  // matching the LM `generate_step`). The earlier `&cfg.lm` borrow + the
  // `&cfg` audio borrow have both ended, so this partial move is sound.
  let max_tokens = cfg.lm.max_tokens;
  let cfg_eos = cfg.lm.eos;

  // EOS union: model.eos_token() ∪ cfg.lm.eos. The model's EOS is always
  // a stop token (no way for the caller to opt out — the model's own
  // identity); the LM `eos` override adds any extras (custom timestamp /
  // task tokens) without dropping the model's identity. `cfg_eos` is the
  // moved-out `Vec<u32>` (no clone, no per-call alloc).
  let model_eos = model.eos_token();
  let mut eos: Vec<u32> = cfg_eos;
  if !eos.contains(&model_eos) {
    eos.push(model_eos);
  }

  Ok(SttGenerator {
    model,
    encoder_states,
    cache,
    sampler,
    processors,
    history: Vec::new(),
    last: model.bos_token(),
    produced: 0,
    max_tokens,
    eos,
    done: false,
  })
}

/// Encode `audio_path` into the model's encoder hidden states.
///
/// Runs steps 1-5 of [`stt_generate`]'s pipeline (load → optional resample →
/// duration cap → log-mel spectrogram → [`super::model::Model::encode_audio`])
/// and returns the resulting encoder states. Useful for callers that want
/// to run multi-pass decoding (e.g. beam search across multiple hypotheses)
/// without re-encoding the same audio for every pass, or that want to cache
/// the encoder output across multiple [`stt_generate`] runs sharing the same
/// input.
pub fn encode_audio_file<M: super::model::Model>(
  model: &M,
  audio_path: &Path,
  cfg: &SttGenConfig,
) -> Result<Array> {
  let mel = audio_path_to_mel(model, audio_path, cfg)?;
  model.encode_audio(&mel)
}
