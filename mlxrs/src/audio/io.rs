//! Multi-format audio file IO + naive linear resampling.
//!
//! Faithful port of the [`mlx_audio.audio_io.read`] /
//! [`mlx_audio.audio_io.write`] core. [`load_audio`] decodes **WAV, MP3,
//! FLAC, and OGG/Vorbis** — the exact set `mlx_audio.audio_io.read`
//! decodes through its in-process `miniaudio` path (WAV / MP3 / FLAC /
//! Vorbis). Resampling is a naive linear-interpolation pass mirroring
//! the `mlx-audio` utilities used by Whisper-style models; high-quality
//! polyphase / sinc resampling is a planned follow-up.
//!
//! *Decoding* goes through the pure-Rust [`symphonia`] crate (active
//! multi-format library). The format is auto-detected by symphonia's
//! probe — [`load_audio`] does not branch on the file extension to pick
//! a codec, it dispatches on the probed container. The decode →
//! `f32`-samples path (`push_samples`) is fully format-agnostic: every
//! codec hands symphonia's PCM decoder a typed [`GenericAudioBufferRef`]
//! and the same per-variant normalization applies regardless of source
//! format. Output is always normalized to `f32` in `[-1.0, 1.0]`.
//!
//! **Formats scoped out of the in-process decoder:** M4A/AAC, Opus, and
//! WebM. `mlx_audio.audio_io.read` decodes those by shelling out to an
//! external `ffmpeg` subprocess (not an in-process codec). Opus in
//! particular has **no pure-Rust symphonia codec** in `symphonia 0.6`
//! (there is no `opus` feature and no `symphonia-codec-opus` crate — it
//! is an open upstream issue), and every alternative needs a heavy
//! `libopus` / `ffmpeg` C dependency. Per the minimal-deps project
//! rule, mlxrs decodes the four formats symphonia covers natively and
//! leaves M4A/AAC/Opus/WebM to a future PR (which would have to add the
//! external-process or C-FFI dependency mlx-audio uses).
//!
//! *Encoding* is roll-our-own pure-Rust 16-bit PCM mono WAV — symphonia
//! exposes no encoder API and the entire RIFF/WAVE/fmt/data spec fits in
//! ~80 LOC, letting us control eval discipline and add atomic-rename
//! without a crate-side change. **Save stays WAV-only**: `mlx-audio`
//! encodes MP3/FLAC/OGG/Opus by shelling out to `ffmpeg`, and adding an
//! MP3/FLAC encoder crate (or the `ffmpeg` process dependency) is a
//! heavy dep mlxrs does not take on for the save path — callers needing
//! a compressed output format must post-process the WAV. This is the
//! same "scope out the heavy-dep formats" decision as the decode side.
//!
//! Both load and save are **mono-only**: multi-channel input is rejected
//! with [`Error::OutOfRange`] (the `Vec<f32>` return shape cannot
//! faithfully carry the reference's 2-D `(samples, channels)` layout). A
//! multi-channel variant returning `(Vec<f32>, u32, u16)` is a planned
//! follow-up.
//!
//! [`mlx_audio.audio_io.read`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/audio_io.py
//! [`mlx_audio.audio_io.write`]: https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/audio_io.py

use std::{
  fs::{self, File},
  io::{BufWriter, Write},
  path::{Path, PathBuf},
};

use symphonia::core::{
  audio::{Audio, GenericAudioBufferRef},
  codecs::audio::AudioDecoderOptions,
  errors::Error as SymphoniaError,
  formats::{
    FormatOptions, TrackType,
    probe::Hint,
    well_known::{FORMAT_ID_FLAC, FORMAT_ID_WAVE},
  },
  io::MediaSourceStream,
  meta::MetadataOptions,
};

use smol_str::format_smolstr;

use crate::error::{
  AllocFailurePayload, ArithmeticOverflowPayload, BoundedDecodePayload, CapExceededPayload, Error,
  FileIoPayload, FileOp, InvariantViolationPayload, LayerKeyedPayload, LengthMismatchPayload,
  MissingKeyPayload, NonFiniteScalarPayload, OutOfRangePayload, ParsePayload, Result,
};

/// PCM-16 full-scale on `read` (see #130). mlxrs uses the
/// **symmetric** `32768.0` convention on both sides — `read` divides by
/// `I16_DIV = 32768.0` and `write` multiplies by `I16_MUL = 32768.0`,
/// matching `torchaudio.save`'s default. The asymmetric `read=32768.0`
/// / `write=32767` split that `mlx_audio.audio_io` inherits from scipy
/// loses one LSB per `read → write → read` cycle; the symmetric form is
/// round-trip-exact within `[-1.0, 1.0)` (the `+1.0` extreme alone
/// saturates by one LSB on the positive boundary — see the `I16_MUL`
/// constant in [`crate::simd::audio::quantize`] for the cast contract).
///
/// On `write`, the SIMD quantizer in
/// [`crate::simd::audio::quantize`] carries the matching `32768`
/// multiplier; that constant lives in the SIMD module so the NEON
/// kernel and scalar reference share one source of truth.
const I16_DIV: f32 = 32768.0;

/// Public-input-driven Vec allocation cap. Hard ceiling on the number of
/// f32 samples [`load_audio`] / [`resample_linear`] will materialize from
/// caller-controllable parameters. 64 Mi-samples ≈ 256 MiB at 4 B / f32
/// — about 30 min of mono 44.1 kHz audio — well above any realistic
/// "load entire file into memory" use case, and well below "abort the
/// host" territory. Crafted-attacker / fuzzer inputs above the cap get
/// a recoverable typed error instead of a Rust allocator abort.
///
/// For **lossy** formats (MP3 / OGG-Vorbis) this is the *primary* memory
/// bound: a container's declared frame count is treated only as a
/// capacity hint and an upfront over-cap rejection, never as a hard
/// per-decode ceiling, because lossy encoders routinely declare an
/// estimate (MP3 Xing/Info header) that differs from the true decoded
/// length. For **exact-count** formats — uncompressed WAV (header frame
/// count is sample-exact) and FLAC carrying a STREAMINFO total
/// (`track_num_frames`, also sample-exact) — the declared count is the
/// hard cap and additionally drives a strict post-decode count
/// cross-check, so a truncated WAV/FLAC surfaces as an error rather than
/// silently returning short audio.
///
/// The cap is enforced *per decoded buffer*, not per individual sample:
/// before a packet's samples are appended, [`load_audio`] rejects the
/// buffer if it would push `out.len()` past the cap and then
/// `try_reserve`s exactly the buffer's sample count under the cap — so
/// `out` never grows through Rust's infallible (abort-on-OOM) allocator
/// path and never exceeds the cap, even when a compressed header
/// under-estimates the true decoded length.
pub const MAX_DECODED_SAMPLES: usize = 64 * 1024 * 1024;
/// See [`MAX_DECODED_SAMPLES`] — same cap, applied to [`resample_linear`].
pub const MAX_RESAMPLED_SAMPLES: usize = MAX_DECODED_SAMPLES;

/// Load a mono audio file and return `(samples, sample_rate)`.
///
/// Decodes **WAV, MP3, FLAC, and OGG/Vorbis** — the format is
/// auto-detected by [`symphonia`]'s probe (the file extension is only a
/// *hint* that speeds detection; the actual codec is chosen from the
/// container magic, so a `.bin` containing a valid MP3 still decodes).
/// This is the exact format set `mlx_audio.audio_io.read` decodes via
/// its in-process `miniaudio` path; M4A/AAC/Opus/WebM (which mlx-audio
/// routes through an external `ffmpeg` process — symphonia has no
/// pure-Rust Opus codec) are out of scope. See the module doc.
///
/// Returns `f32` samples in `[-1.0, 1.0]`. Integer-PCM inputs are divided
/// by `2^(bits-1)` (16-bit → 32768.0, matching `mlx_audio.audio_io.read`'s
/// `dtype="float32"` arm exactly). Float / already-decoded samples are
/// passed through unchanged. The per-format decoder (MP3 / FLAC / Vorbis)
/// produces a typed [`GenericAudioBufferRef`] that the shared
/// `push_samples` normalizer handles identically — there is no
/// per-format sample path.
///
/// **Multi-channel files are rejected** with [`Error::OutOfRange`] — the
/// reference `mlx_audio.audio_io.read` returns a 2-D `(samples, channels)`
/// array, but the `Vec<f32>` return shape here cannot faithfully carry that
/// information. Callers needing stereo / 5.1 / etc. must either preprocess
/// (downmix or split channels) before calling this fn, or use the planned
/// multi-channel follow-up.
///
/// # Errors
/// - Typed errors: [`Error::FileIo`] if the file cannot be opened,
///   [`Error::Parse`] if the format is unsupported or codec fails,
///   [`Error::OutOfRange`] if `channels != 1` or a chained stream,
///   [`Error::MissingKey`] if no audio track or codec params,
///   [`Error::BoundedDecode`] if decoded samples exceed the cap,
///   [`Error::LengthMismatch`] for WAV/FLAC frame-count mismatch,
///   [`Error::NonFiniteScalar`] for NaN/inf samples.
pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
  load_audio_with_cap(path, MAX_DECODED_SAMPLES)
}

/// Decode `path` into an owned `Vec<f32>` like [`load_audio`], but reuse
/// `out`'s pre-allocated capacity instead of allocating a fresh buffer.
///
/// **Buffer reuse for streaming / batch loaders.** A typical TTS dataset
/// pre-allocates one `Vec<f32>` sized for the longest expected utterance
/// and calls [`load_audio_into`] for each file — the destination's
/// capacity is reused, so per-file `load_audio` allocations collapse to
/// (at most) one growth when the cap is exceeded. `out` is cleared
/// (`out.clear()`) before decoding; existing samples are discarded but
/// the underlying capacity is retained. On any decode error `out`'s
/// length is unspecified (it may carry partial decoded samples that
/// should not be observed) — call `out.clear()` before retrying with a
/// different file.
///
/// Mirrors [`load_audio`]'s validation and the
/// [`MAX_DECODED_SAMPLES`] cap exactly; the only difference is the
/// caller-owned destination buffer. The sample-rate return is the
/// container-declared rate, same as [`load_audio`].
///
/// # Errors
/// - Same as [`load_audio`].
pub fn load_audio_into(path: &Path, out: &mut Vec<f32>) -> Result<u32> {
  load_audio_into_with_cap(path, out, MAX_DECODED_SAMPLES)
}

/// Load a mono audio file enforcing `max_samples` BEFORE allocating the
/// decoded sample buffer.
///
/// Layered-cap variant of [`load_audio`] for callers (e.g. an STT
/// preprocessing front-end) that need to reject an oversized input strictly
/// **before** the load-stage allocation rather than after. The behavior is
/// identical to [`load_audio`] except:
///
/// - The container-declared frame count (`Track::num_frames` for WAV /
///   FLAC-with-STREAMINFO — the formats whose header carries an exact
///   total) is rejected against `min(max_samples, MAX_DECODED_SAMPLES)`
///   BEFORE the sample `Vec` is allocated. A 30-minute WAV with a
///   30-second cap therefore never touches the 256 MiB load-stage
///   ceiling — the rejection fires at the header-parse stage with one
///   recoverable [`Error::CapExceeded`].
/// - The per-decoded-buffer cap (`reserve_under_cap` / `push_samples`)
///   uses the same `min(max_samples, MAX_DECODED_SAMPLES)` so a
///   compressed file lacking a declared frame count — MP3 Xing/Info
///   under-estimate, OGG-Vorbis, FLAC without STREAMINFO — is still
///   rejected mid-decode the moment `out.len()` would exceed
///   `max_samples`. The wall-time cost of partial decode is bounded by
///   `max_samples` worth of decoded f32 frames.
///
/// Passing `max_samples == usize::MAX` (or `>= MAX_DECODED_SAMPLES`) is
/// equivalent to calling [`load_audio`] directly — the
/// [`MAX_DECODED_SAMPLES`] global cap still applies.
///
/// # Errors
/// - Same set as [`load_audio`], plus the early header-cap rejection
///   above (an [`Error::CapExceeded`] naming `effective_cap` and the
///   container-declared `header_len`).
pub fn load_audio_with_cap(path: &Path, max_samples: usize) -> Result<(Vec<f32>, u32)> {
  let mut out: Vec<f32> = Vec::new();
  let sample_rate = load_audio_into_with_cap(path, &mut out, max_samples)?;
  Ok((out, sample_rate))
}

/// Duration-aware variant of [`load_audio_with_cap`] that derives the
/// load-stage sample cap from the **source** file's own sample rate.
///
/// Mirrors [`load_audio_with_cap`] except the caller supplies a
/// `max_seconds` budget instead of a raw sample count: the function
/// probes the container's codec parameters to read the source sample
/// rate, then enforces `max_samples = src_sr * max_seconds` (clamped to
/// [`MAX_DECODED_SAMPLES`]) as the load-stage cap.
///
/// **Why source-rate, not target-rate.** A pipeline that downsamples
/// (e.g. STT at a 16 kHz model with a 44.1 kHz source) cannot use
/// `target_sr * max_seconds` as the load cap: a perfectly valid 1.0 s
/// source carries `src_sr * 1.0 = 44 100` samples, but
/// `target_sr * 1.0 = 16 000` would reject the header at the load
/// stage. Deriving the cap from the source rate after probing keeps
/// the cap consistent with the file's declared duration, so any input
/// whose source duration is `<= max_seconds` decodes successfully
/// regardless of the caller's downstream resample target rate.
///
/// Returns the same `(samples, source_sample_rate)` pair as
/// [`load_audio_with_cap`] — the caller still resamples to its target
/// rate (or asserts equality) after this returns.
///
/// `max_seconds` must be a positive finite `f32`; NaN, ±∞, zero, and
/// negatives surface as [`Error::OutOfRange`] before any file IO. The
/// `src_sr * max_seconds` product is computed in `f64` (the load-stage
/// `MAX_DECODED_SAMPLES` ceiling fits comfortably in the `f64`
/// mantissa) and any out-of-range product saturates to `usize::MAX`
/// per the Rust `as usize` spec, which is then clamped to
/// [`MAX_DECODED_SAMPLES`].
///
/// **TOCTOU-free unified probe + decode.** The function opens the file
/// EXACTLY ONCE: the probe pass that reads
/// `src_sr` and the decode pass that produces samples share the same
/// `File` handle / `FormatReader` / `Track`, so a path that is
/// mutated, replaced, or symlink-swapped between the two phases
/// cannot drift the cap from the decoded stream. Re-opening the file
/// for a probe-only `File::open` and then handing off to
/// [`load_audio_with_cap`] (which re-opens a second time) would let a
/// high-rate probe authorize a much larger cap for a subsequent
/// low-rate decode of a different file. The implementation instead
/// funnels through a single internal worker that
/// derives `effective_cap` from the SAME `FormatReader` that drives
/// the decode loop.
///
/// # Errors
/// - [`Error::OutOfRange`] if `max_seconds` is not a finite value > 0
///   (NaN, ±∞, zero, negative).
/// - Same set as [`load_audio_with_cap`] for the open / probe / decode
///   pass (open / probe / no audio track / no codec params /
///   multi-channel / header > cap / decode failure).
pub fn load_audio_with_max_seconds(path: &Path, max_seconds: f32) -> Result<(Vec<f32>, u32)> {
  // Validate the caller's duration cap up front (mirrors the STT
  // pipeline's `max_audio_seconds` guard). `is_finite()` covers NaN
  // and ±∞; `> 0.0` covers zero and negatives. Two positive guards
  // avoid the `neg_cmp_op_on_partial_ord` clippy lint forbidding the
  // negated-comparison shorthand on `f32`.
  if !max_seconds.is_finite() || max_seconds <= 0.0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "load_audio_with_max_seconds: max_seconds",
      "must be a finite value > 0",
      format!("{max_seconds}"),
    )));
  }

  // One open, one probe, one decode — the unified worker resolves the
  // cap strategy to `effective_cap` AFTER reading `src_sr` from the same
  // `FormatReader` it then decodes through.
  let mut out: Vec<f32> = Vec::new();
  let sr = load_audio_into_unified(path, &mut out, CapStrategy::SrcRateMaxSeconds(max_seconds))?;
  Ok((out, sr))
}

/// Resolution strategy for the load-stage sample cap, threaded through
/// [`load_audio_into_unified`] so a single open/probe pass serves both
/// the explicit `max_samples` and the duration-derived
/// `src_sr * max_seconds` paths.
///
/// Both variants are clamped to [`MAX_DECODED_SAMPLES`] inside the
/// unified worker — the global ceiling is the hard upper bound and no
/// caller-supplied strategy can raise it.
#[derive(Debug, Clone, Copy)]
enum CapStrategy {
  /// Explicit caller-supplied sample cap. Used by [`load_audio_with_cap`]
  /// and [`load_audio_into_with_cap`].
  MaxSamples(usize),
  /// Duration cap (in seconds) resolved AFTER probing the source sample
  /// rate from the same `FormatReader` used for the decode loop. The
  /// computed cap is `src_sr * max_seconds` (in `f64`, saturated to
  /// `usize::MAX` per the `as usize` spec). Used by
  /// [`load_audio_with_max_seconds`]. The `f32` payload must be a finite
  /// value > 0 — the public entry point validates this before calling.
  SrcRateMaxSeconds(f32),
}

/// Buffer-reusing + layered-cap variant of [`load_audio_into`].
///
/// Combines [`load_audio_into`] (reuse caller's `Vec`) with
/// [`load_audio_with_cap`] (reject oversized inputs against
/// `max_samples` BEFORE allocation). Used by streaming pipelines that
/// need both a tight per-utterance cap AND a reused decode buffer
/// across files — the worker behind [`load_audio_into`] and
/// [`load_audio_with_cap`].
///
/// `out` is cleared before decoding; existing samples are discarded
/// but the underlying capacity is retained. On any decode error
/// `out`'s length is unspecified.
///
/// # Errors
/// - Same as [`load_audio_with_cap`].
pub fn load_audio_into_with_cap(
  path: &Path,
  out: &mut Vec<f32>,
  max_samples: usize,
) -> Result<u32> {
  load_audio_into_unified(path, out, CapStrategy::MaxSamples(max_samples))
}

/// Unified open + probe + decode worker shared by every load-audio entry
/// point.
///
/// **One `File::open` per call (TOCTOU fix).** Both the
/// explicit-sample-cap path ([`CapStrategy::MaxSamples`]) and the
/// duration-derived path ([`CapStrategy::SrcRateMaxSeconds`]) flow
/// through this function: it opens `path` ONCE, probes the container
/// ONCE, and runs the decode loop against the SAME `FormatReader` —
/// no second `File::open` ever appears between the probe that reads
/// `src_sr` and the decode that consumes the audio. Re-opening the path
/// for a separate probe-only pass and then handing off to
/// `load_audio_with_cap` (which opens it a third time) would let a
/// high-rate probe of `foo.wav` authorize a much larger cap for a
/// low-rate decode of a different `foo.wav` whose contents had been
/// swapped between opens. Funneling every entry through one open + one
/// probe + one decode eliminates the TOCTOU window structurally.
///
/// **Cap resolution.** The strategy is resolved to `effective_cap`
/// AFTER reading `sample_rate` from the same `FormatReader` that drives
/// the decode loop:
/// - `MaxSamples(n)` → `effective_cap = min(n, MAX_DECODED_SAMPLES)`.
/// - `SrcRateMaxSeconds(s)` → `effective_cap = min(src_sr * s as
///   usize, MAX_DECODED_SAMPLES)`, with the product computed in `f64`
///   and saturated to `usize::MAX` per the `as usize` spec.
///
/// **Header cap policy (lossy-overestimate fix).** The
/// container-declared `num_frames` is treated differently depending on
/// `exact_count`:
/// - **Exact-count formats** (WAV always, FLAC with STREAMINFO total):
///   a `header_len > effective_cap` declaration is a hard upfront
///   rejection — the count is sample-exact, so an over-cap declaration
///   IS a file outside the budget.
/// - **Estimate formats** (MP3 Xing/Info, OGG-Vorbis granule rounding):
///   the header is an estimate that routinely overstates the true
///   length, so an upfront rejection of `header_len > effective_cap`
///   would reject perfectly valid in-cap audio. Instead, the
///   reservation hint is clamped to `effective_cap` and the actual cap
///   enforcement is left to the `push_samples` per-buffer guard, which
///   decides based on real decoded sample count.
///
/// `out` is cleared before decoding (capacity retained); on any decode
/// error `out`'s length is unspecified.
fn load_audio_into_unified(path: &Path, out: &mut Vec<f32>, strategy: CapStrategy) -> Result<u32> {
  // Open + wrap in symphonia's MediaSourceStream. Box<File> is required by
  // the MediaSource trait; the allocation is one-per-load_audio (fine —
  // file IO dominates the cost). THIS IS THE ONLY `File::open` IN THE
  // WHOLE LOAD PATH — every cap strategy reads `src_sr` from this same
  // handle, eliminating the probe-vs-decode TOCTOU window.
  let file = File::open(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "load_audio",
      FileOp::Open,
      ::std::path::PathBuf::from(path),
      e,
    ))
  })?;
  let mss = MediaSourceStream::new(Box::new(file), Default::default());

  // Hint with the file extension if present — lets the probe skip
  // format-detection backtracking for known extensions. The hint is only
  // an optimization: symphonia still verifies the container magic, so a
  // mislabelled / extensionless file is detected by content, not name.
  let mut hint = Hint::new();
  if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
    hint.with_extension(ext);
  }

  let mut format = symphonia::default::get_probe()
    .probe(
      &hint,
      mss,
      FormatOptions::default(),
      MetadataOptions::default(),
    )
    .map_err(|e| {
      Error::Parse(ParsePayload::new(
        "load_audio: container probe (unsupported or corrupt format; \
         WAV/MP3/FLAC/OGG-Vorbis are supported, M4A/AAC/Opus/WebM are not)",
        "audio container",
        e,
      ))
    })?;

  // Probed container id. Two formats carry a *sample-exact* declared
  // frame count and so qualify for the strict post-decode count
  // cross-check + the header-count-as-hard-cap path:
  //   - WAV: the header `num_frames` is sample-exact by construction.
  //   - FLAC: STREAMINFO carries an EXACT total sample count, which
  //     symphonia surfaces as `Track::num_frames` (combined with the
  //     `track_num_frames.is_some()` check below — a FLAC without a
  //     declared total stays relaxed).
  // Genuinely-estimated lossy formats (MP3 Xing/Info, Vorbis granule
  // rounding) use `MAX_DECODED_SAMPLES` as the hard cap and skip the
  // equality check. Captured BEFORE the mutable `format.next_packet`
  // borrows below.
  let format_id = format.format_info().format;
  let is_wav = format_id == FORMAT_ID_WAVE;
  let is_flac = format_id == FORMAT_ID_FLAC;

  let track = format.default_track(TrackType::Audio).ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "load_audio: container has no audio track",
      format_smolstr!("{}", path.display()),
    ))
  })?;
  let track_id = track.id;
  // `Track::num_frames` is the per-channel frame count from the
  // container header (`None` if the container didn't declare one).
  // Captured BEFORE we borrow `codec_params` because we'll move out
  // of the immutable `track` borrow below.
  let track_num_frames = track.num_frames;

  // Does this track carry a sample-EXACT declared frame count? True for
  // WAV (always exact) and for FLAC *when* STREAMINFO declared a total
  // (`track_num_frames` present) — symphonia surfaces the STREAMINFO
  // total as `num_frames` for FLAC, and that value is exact, so a
  // truncated FLAC that decodes fewer frames is real corruption. Lossy
  // codecs (MP3 Xing/Info, Vorbis) are NOT exact: their declared count
  // is an estimate, so they stay relaxed (hint-only) regardless. Drives
  // (1) the upfront `header_len > effective_cap` REJECTION (only on
  // exact-count — otherwise an over-estimate would spuriously fail a
  // valid in-cap file), (2) the hard per-buffer `cap` ceiling, and
  // (3) the strict post-decode equality cross-check.
  let exact_count = is_wav || (is_flac && track_num_frames.is_some());

  let codec_params = track.codec_params.as_ref().ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "load_audio: track has no codec parameters",
      format_smolstr!("{}", path.display()),
    ))
  })?;
  let audio_params = codec_params.audio().ok_or_else(|| {
    Error::InvariantViolation(InvariantViolationPayload::new(
      "load_audio: default track",
      "must be an audio track (codec_params.audio() returned None)",
    ))
  })?;

  // Channel count: reject non-mono UPFRONT before any decode work. Using
  // the codec_params channel count avoids waiting until the first packet
  // is decoded to discover stereo.
  let nchannels = audio_params
    .channels
    .as_ref()
    .map(|c| c.count())
    .ok_or_else(|| {
      Error::MissingKey(MissingKeyPayload::new(
        "load_audio: container has no channel layout",
        format_smolstr!("{}", path.display()),
      ))
    })?;
  if nchannels == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "load_audio: container channel count",
      "must be exactly 1 (mono input required)",
      "0",
    )));
  }
  if nchannels != 1 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "load_audio: container channel count (multi-channel input not supported; \
         this API returns mono Vec<f32>; downmix or split channels before calling)",
      "must be exactly 1 (mono)",
      format_smolstr!("{nchannels}"),
    )));
  }

  let sample_rate = audio_params.sample_rate.ok_or_else(|| {
    Error::MissingKey(MissingKeyPayload::new(
      "load_audio: container has no sample_rate",
      format_smolstr!("{}", path.display()),
    ))
  })?;

  // Resolve the cap strategy to `effective_cap` USING THE SAME
  // `sample_rate` JUST READ from the SAME `FormatReader` that drives
  // the decode loop below. This closes the TOCTOU window: the
  // duration-cap path does not derive `src_sr` from a separate probe
  // open whose decode could be a different file.
  let effective_cap = strategy.resolve(sample_rate);

  let mut decoder = symphonia::default::get_codecs()
    .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
    .map_err(|e| {
      Error::Parse(ParsePayload::new(
        "load_audio: make_audio_decoder failed",
        "audio codec",
        e,
      ))
    })?;

  // Capped allocation. `num_frames` is the container-declared per-channel
  // frame count; for mono that equals the total f32 output length.
  // `Vec::with_capacity` is infallible (aborts on allocator OOM), so we
  // fall back to `try_reserve_exact` against `MAX_DECODED_SAMPLES`.
  //
  // Header-cap policy (lossy-overestimate fix):
  // - **Exact-count formats** (WAV always, FLAC-with-STREAMINFO): a
  //   `header_len > effective_cap` declaration is rejected upfront. The
  //   count is sample-exact by construction, so an over-cap declaration
  //   IS the file being outside the budget — no reason to start the
  //   decode.
  // - **Estimate formats** (MP3 Xing/Info, OGG-Vorbis granule rounding,
  //   FLAC without STREAMINFO): the header is a *capacity hint*, not a
  //   bound. MP3 Xing/Info routinely *overstates* the true decoded
  //   length by encoder-delay/padding frames, so an upfront
  //   `header_len > effective_cap` rejection would spuriously fail
  //   perfectly valid in-cap audio. Instead the reservation hint below
  //   is clamped to `effective_cap`, and the actual cap is enforced per
  //   decoded buffer by `push_samples` using the REAL decoded count —
  //   so an estimate-format file that genuinely exceeds the cap still
  //   fails (mid-decode), but one that overstates an in-cap length
  //   decodes successfully.
  let header_len_opt = track_num_frames.and_then(|n| usize::try_from(n).ok());
  if exact_count
    && let Some(header_len) = header_len_opt
    && header_len > effective_cap
  {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "load_audio: container-declared sample count exceeds cap (refuse to allocate; \
         oversized / crafted inputs require a streaming decoder API, planned follow-up; \
         exact-count format: WAV or FLAC-with-STREAMINFO)",
      "effective_cap (= min(max_samples, MAX_DECODED_SAMPLES))",
      effective_cap as u64,
      header_len as u64,
    )));
  }
  // Reservation hint — used by `try_reserve_exact` below to avoid
  // reallocation churn during the decode loop. For exact-count formats
  // this is the (already cap-checked) header length; for estimate
  // formats this is the header's overestimate CLAMPED to `effective_cap`
  // so a 10-minute MP3 Xing estimate paired with a 30-second cap
  // reserves at most 30 seconds of samples upfront and lets the
  // `push_samples` per-buffer cap reject mid-decode if the real length
  // actually overshoots.
  let reserve_len = header_len_opt.unwrap_or(0).min(effective_cap);
  // Buffer-reuse path (`load_audio_into` / `load_audio_into_with_cap`):
  // discard any prior contents but keep the caller's capacity. The
  // subsequent `try_reserve_exact(reserve_len)` requests *additional*
  // capacity beyond what `out` already has, so a generously-sized reused
  // buffer skips the reservation entirely (the `try_reserve_exact` op
  // is a no-op when `additional <= remaining capacity`).
  out.clear();
  out.try_reserve_exact(reserve_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "load_audio: reservation",
      "samples",
      reserve_len as u64,
      e,
    ))
  })?;

  // `cap` is the hard upper bound on `out.len()` enforced per decoded
  // buffer by `push_samples`.
  // - Exact-count formats (WAV, FLAC-with-STREAMINFO-total): the header
  //   `num_frames` is sample-exact, so the cap is that count —
  //   over-running it means a malformed/corrupt file. Still clamped to
  //   `effective_cap` so a caller-supplied tighter cap (e.g. the STT
  //   pipeline's `max_audio_seconds * sample_rate`) does not raise the
  //   effective ceiling.
  // - Lossy MP3 / OGG-Vorbis (and a FLAC without a declared total): the
  //   header count (if any) is only an estimate, so the cap is the
  //   `effective_cap` (= `min(max_samples, MAX_DECODED_SAMPLES)`); using
  //   the estimate as a hard cap would spuriously fail a valid file whose
  //   true length slightly exceeds an under-estimating Xing/Info header.
  // Reaching `cap` makes `push_samples` return `Error::BoundedDecode` rather
  // than re-grow into the infallible-alloc path.
  let cap = if exact_count {
    header_len_opt.unwrap_or(effective_cap).min(effective_cap)
  } else {
    effective_cap
  };

  // Decode loop. End-of-stream is signalled by `Ok(None)` from
  // `next_packet`. Some symphonia format readers surface end-of-stream
  // as `UnexpectedEof` instead, and that case is also treated as a clean
  // EOF. All other packet/decode errors are reported to the caller
  // rather than being skipped.
  loop {
    let packet = match format.next_packet() {
      Ok(Some(p)) => p,
      Ok(None) => break,
      Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
        // Some symphonia format readers surface end-of-stream as an
        // UnexpectedEof error rather than `Ok(None)` — treat both as
        // clean end-of-stream so a properly-terminated file decodes
        // identically regardless of which path the reader takes.
        break;
      }
      Err(SymphoniaError::ResetRequired) => {
        // A chained / multi-segment stream (most commonly a chained
        // OGG file) whose track list changes mid-stream. Faithfully
        // handling it would require re-examining tracks and rebuilding
        // the decoder; this mono single-stream loader does not support
        // that, so fail loud rather than silently truncating at the
        // segment boundary.
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "load_audio: container kind (only single-stream audio is supported; \
             chained/multi-segment streams surfaced as ResetRequired)",
          "must be a single-stream container",
          format_smolstr!("{}", path.display()),
        )));
      }
      Err(e) => {
        return Err(Error::Parse(ParsePayload::new(
          "load_audio: next_packet failed",
          "audio packet",
          e,
        )));
      }
    };
    if packet.track_id != track_id {
      continue;
    }
    // Fail-loud on ANY decode error — a silent "skip the bad packet"
    // path would let a truncated or malformed file come back as `Ok`
    // with missing audio, exactly the silent-corruption surface the
    // load_audio contract excludes.
    let audio_buf = decoder.decode(&packet).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "load_audio: decode failed",
        "audio packet",
        e,
      ))
    })?;

    // Push interleaved f32 samples into `out`. We match on the typed
    // GenericAudioBufferRef variant + apply our own `/2^(bits-1)`
    // divisor for integer-PCM variants (symphonia's built-in
    // `Sample::to_sample::<f32>` for i16 divides by `i16::MAX = 32767`,
    // not `32768.0` — a 1-LSB drift from `mlx_audio.audio_io.read`'s
    // `int16 / 32768.0` convention that we avoid by going through the
    // raw integer samples). MP3 / Vorbis decoders hand us an already-
    // decoded f32 buffer; that arm is pass-through.
    // `out` is already `&mut Vec<f32>`; auto-reborrow handles the
    // call-site type without an explicit `&mut *out`.
    push_samples(&audio_buf, out, cap)?;
  }

  // Cross-check decoded sample count against the header-declared length
  // — for **exact-count formats only** (WAV always, FLAC when STREAMINFO
  // declared a total). Their declared `num_frames` is sample-EXACT, so a
  // short read (truncated WAV / truncated FLAC that hit a clean EOF after
  // partial frames) would otherwise return `Ok` with missing audio — the
  // same silent-corruption surface the per-packet fail-loud above closes,
  // but at the stream level. `out.len() > header_len` cannot be reached
  // here for these formats because the per-buffer cap is exactly
  // `header_len` when known.
  //
  // For lossy MP3 / OGG-Vorbis (and a FLAC without a declared total) this
  // check is deliberately SKIPPED: a lossy encoder's declared frame count
  // is an estimate (MP3 Xing/Info header) and granule/decoder-delay
  // rounding means even an "accurate" count routinely differs from
  // `out.len()` by a handful of samples. Enforcing equality there would
  // reject perfectly valid compressed files. The `MAX_DECODED_SAMPLES`
  // cap (enforced per buffer above) remains the bound that protects
  // against an unbounded/oversized compressed stream.
  if exact_count
    && let Some(header_len) = header_len_opt
    && out.len() != header_len
  {
    return Err(Error::LengthMismatch(LengthMismatchPayload::new(
      "load_audio: decoded vs container-header sample count \
         (truncated or malformed exact-count file: WAV or FLAC)",
      header_len,
      out.len(),
    )));
  }

  Ok(sample_rate)
}

impl CapStrategy {
  /// Resolve the strategy to a concrete `effective_cap` (in samples)
  /// AFTER `sample_rate` has been read from the same `FormatReader`
  /// that drives the decode loop. Both arms clamp at
  /// [`MAX_DECODED_SAMPLES`] — no caller-supplied strategy can raise
  /// the global ceiling.
  fn resolve(self, sample_rate: u32) -> usize {
    match self {
      // The effective cap is the smaller of the caller's cap and the
      // load-stage global ceiling. A caller can never raise the
      // ceiling above `MAX_DECODED_SAMPLES`; passing `usize::MAX`
      // simply degrades to the bare `load_audio` behavior.
      Self::MaxSamples(n) => n.min(MAX_DECODED_SAMPLES),
      Self::SrcRateMaxSeconds(max_seconds) => {
        // Derive the load-stage sample cap from the SOURCE rate, so a
        // valid `max_seconds`-long input always decodes regardless of
        // the caller's downstream resample target. `f64` arithmetic
        // avoids any overflow on a very large `src_sr * max_seconds`
        // product; the result saturates to `usize::MAX` per the
        // `as usize` spec when the product exceeds `usize::MAX as f64`,
        // and we then clamp to `MAX_DECODED_SAMPLES`. The public
        // `load_audio_with_max_seconds` entry point validated
        // `max_seconds` is a finite value > 0 so the product is finite
        // by construction; the defensive `is_finite()` guard here makes
        // the saturation explicit if a future caller bypasses the
        // public entry point.
        let raw = f64::from(sample_rate) * f64::from(max_seconds);
        let max_samples = if raw.is_finite() && raw >= 0.0 {
          raw.min(usize::MAX as f64) as usize
        } else {
          MAX_DECODED_SAMPLES
        };
        max_samples.min(MAX_DECODED_SAMPLES)
      }
    }
  }
}

/// Bound + reserve room for `n` more samples in `out` BEFORE appending
/// them, so the subsequent appends cannot grow the `Vec` through Rust's
/// infallible (abort-on-OOM) allocator path.
///
/// Two guards, in order:
/// 1. **Cap.** Reject if `n` would push `out.len()` past `cap` — the hard
///    ceiling, either an exact-count header's frame count (WAV /
///    FLAC-with-STREAMINFO) or [`MAX_DECODED_SAMPLES`] (lossy /
///    no-declared-count path). Computed as `n > cap - out.len()` with the
///    subtraction order chosen so it cannot underflow (`out.len() <= cap`
///    is the invariant every caller maintains by routing each decoded
///    buffer through this guard). Returns [`Error::BoundedDecode`] (the existing
///    over-cap error).
/// 2. **Cap-limited fallible reserve.** Grow capacity to fit `n` more
///    samples while STILL under the cap, mapping a
///    [`std::collections::TryReserveError`] to [`Error::OutOfMemory`].
///    After this returns `Ok`, the caller's `out.push` calls for these `n`
///    samples land in reserved capacity and cannot trigger an infallible
///    (abort-on-OOM) `Vec` regrowth.
///
/// The reserve keeps an amortized-doubling growth strategy (avoids
/// quadratic reallocation when a compressed header under-estimates the true
/// length) BUT clamps the *target capacity* at `cap`. A plain
/// `Vec::try_reserve(n)` is amortized too, but it can grow capacity to
/// ~2× the *current* capacity even when only a few samples remain under the
/// cap — so a near-cap header hint (reserved upfront by `try_reserve_exact`
/// in [`load_audio`]) plus a final in-cap packet would make the reservation
/// itself attempt FAR more than [`MAX_DECODED_SAMPLES`], defeating the
/// advertised memory ceiling (and spuriously failing an in-cap decode under
/// memory pressure, since the oversized reserve fails). Clamping `target`
/// at `cap` keeps the doubling shape while guaranteeing the reservation
/// never asks for more than the cap.
fn reserve_under_cap(out: &mut Vec<f32>, n: usize, cap: usize) -> Result<()> {
  // `out.len() <= cap` always holds on entry (every prior buffer went
  // through this guard), so `cap - out.len()` does not underflow. This is
  // the hard cap enforcement and stays BEFORE the growth logic below.
  if n > cap - out.len() {
    // `out.len() + n` would panic in debug builds (or wrap in release) on a
    // pathological `n` (e.g. `usize::MAX`) from a corrupt/hostile decoder
    // buffer count. The observed value is for diagnostics only (no real
    // allocation rides on it), so saturate at `u64::MAX` to keep the
    // recoverable BoundedDecode error path intact. The cap-violation
    // semantics are unchanged: any saturated count is also `> cap`.
    let observed = (out.len() as u64).saturating_add(n as u64);
    return Err(Error::BoundedDecode(BoundedDecodePayload::new(
      "load_audio: stream produced more than the sample cap",
      cap as u64,
      observed,
    )));
  }
  // The over-cap check above guarantees `out.len() + n <= cap`, so `needed`
  // does not overflow and is itself `<= cap`.
  let needed = out.len() + n;
  if out.capacity() < needed {
    // Amortized doubling, but clamped at the cap: a plain `try_reserve(n)`
    // could grow to ~2× the current capacity (potentially past `cap`),
    // whereas this never targets more than `cap`. `needed <= cap`, so the
    // `max` keeps `target <= cap` while still allowing the doubling shape
    // when there is headroom under the cap.
    let target = needed.max(out.capacity().saturating_mul(2).min(cap));
    out
      .try_reserve_exact(target - out.len())
      .map_err(|_| Error::OutOfMemory)?;
  }
  Ok(())
}

/// Append the interleaved samples in `buf` to `out`, normalizing to
/// `[-1, 1]` f32 via `/2^(bits-1)` for integer variants (matching
/// `mlx_audio.audio_io.read`'s `int16 / 32768.0` convention exactly) and
/// pass-through for f32 / f64. The intrinsic-bit-width per typed variant
/// is what drives the divisor (NOT `codec_params.bits_per_coded_sample`)
/// because the symphonia PCM decoder has already shifted any narrower
/// coded samples up to the buffer's intrinsic width — see the per-arm
/// comments and `read_pcm_*!` in `symphonia-codec-pcm`.
///
/// This normalizer is **format-agnostic** — it is the single decode →
/// `f32` path for WAV, MP3, FLAC, and OGG/Vorbis alike. Compressed
/// codecs (MP3 / Vorbis) decode to the f32 / f64 arms (pass-through);
/// PCM/WAV and FLAC decode to the integer arms (`/2^(bits-1)`).
///
/// Enforces a hard upper bound `cap` on the running `out` length and,
/// critically, does so *before* any append can grow the `Vec`. For each
/// decoded buffer the count `n` of interleaved samples is known upfront
/// (`Audio::samples_interleaved`), so we (1) reject the buffer if it would
/// push `out.len()` past `cap` and (2) `try_reserve(n)` the room while
/// still under the cap — mapping a reservation failure to
/// [`Error::OutOfMemory`]. The subsequent per-sample `out.push` calls then
/// land in already-reserved capacity and CANNOT trigger Rust's infallible
/// (abort-on-OOM) `Vec` growth. This is the bound that protects against an
/// attacker-controlled compressed file whose header under-estimates the
/// true decoded length (the header is only a HINT for lossy formats), so
/// `out` never grows infallibly and never exceeds the cap.
fn push_samples(buf: &GenericAudioBufferRef<'_>, out: &mut Vec<f32>, cap: usize) -> Result<()> {
  /// Reject a non-finite f32 (NaN/inf) before pushing — a NaN/inf in the
  /// decoded PCM would silently poison every downstream DSP stage.
  fn check_finite(s: f32) -> Result<f32> {
    if s.is_finite() {
      Ok(s)
    } else {
      Err(Error::NonFiniteScalar(NonFiniteScalarPayload::new(
        "load_audio: non-finite f32 PCM sample",
        s as f64,
      )))
    }
  }
  /// Resolve the integer-PCM divisor `2^(bits-1)` for one of the
  /// fixed widths the typed match arms below pass in (8 / 16 / 24 /
  /// 32). 16-bit is fast-pathed through `I16_DIV` so the byte-for-byte
  /// mlx-audio match is exact. Returns `Err` only on a future-arms
  /// programmer error.
  fn int_divisor(bits: u32) -> Result<f32> {
    Ok(match bits {
      8 => 128.0,
      16 => I16_DIV,
      24 => 8_388_608.0,
      32 => 2_147_483_648.0,
      n => {
        return Err(Error::OutOfRange(OutOfRangePayload::new(
          "load_audio: int_divisor bit-width (programmer error)",
          "must be one of {8, 16, 24, 32}",
          format_smolstr!("{n}"),
        )));
      }
    })
  }
  // Float buffers (f32/f64 — MP3/Vorbis decoders, float WAVs) are
  // pass-through (already `[-1, 1]`). Integer-PCM buffers (PCM WAV,
  // FLAC) are normalized by `2^(bits-1)` against the variant's intrinsic
  // bit-width, NOT `bits_per_coded_sample` — the symphonia PCM decoder
  // has already up-shifted any coded-sample bits below the intrinsic
  // width via `shift = decoded_width - coded_width` in `read_pcm_*!`,
  // so the typed buffer's range is always the full intrinsic-width
  // range for that variant (i8 in ±2^7, i16 in ±2^15, i24-wrapper
  // `.inner()` in ±2^23, i32 in ±2^31, and the unsigned/offset-binary
  // variants in their corresponding [0, 2^N) ranges).
  //
  // 16-bit i16 is fast-pathed through the `I16_DIV = 32768.0` const so
  // the byte-for-byte mlx-audio match is exact. Symphonia's own
  // built-in `Sample::to_sample::<f32>` for i16 divides by `i16::MAX =
  // 32767` (a 1-LSB drift from mlx-audio's 32768.0), so we deliberately
  // bypass it and apply our own divisor.
  //
  // Each arm bounds + reserves the buffer's interleaved sample count
  // (`b.samples_interleaved() == num_planes * frames`, exactly what
  // `iter_interleaved` yields) ONCE via `reserve_under_cap` before the
  // push loop. After that reservation the per-sample `out.push` calls
  // land in reserved capacity and cannot trigger an infallible `Vec`
  // regrowth — so the cap is a true bound even when a compressed header
  // under-estimated the decoded length.
  match buf {
    GenericAudioBufferRef::F32(b) => {
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        out.push(check_finite(s)?);
      }
    }
    GenericAudioBufferRef::F64(b) => {
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        out.push(check_finite(s as f32)?);
      }
    }
    GenericAudioBufferRef::U8(b) => {
      // Offset-binary unsigned 8-bit (range [0, 256), midpoint 128).
      // Recenter via `i16::from(s) - 128` to land in `[-128, 127]`,
      // then divide by `2^7 = 128.0`.
      let divisor = int_divisor(8)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        let signed = i16::from(s) - 128;
        out.push(f32::from(signed) / divisor);
      }
    }
    GenericAudioBufferRef::U16(b) => {
      // Offset-binary unsigned 16-bit (range [0, 65536), midpoint 32768).
      let divisor = int_divisor(16)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        let signed = i32::from(s) - 32_768;
        out.push(signed as f32 / divisor);
      }
    }
    GenericAudioBufferRef::U24(b) => {
      // Offset-binary unsigned 24-bit (range [0, 2^24), midpoint 2^23).
      // `u24.inner()` returns the `[0, 2^24)` value as `u32`.
      let divisor = int_divisor(24)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        let signed = s.inner() as i32 - 0x80_0000;
        out.push(signed as f32 / divisor);
      }
    }
    GenericAudioBufferRef::U32(b) => {
      // Offset-binary unsigned 32-bit (range [0, 2^32), midpoint 2^31).
      // Use `wrapping_sub` to stay in i32 wraparound semantics — the
      // result is the i32 reinterpretation of `(s - 2^31)`.
      let divisor = int_divisor(32)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        let signed = s.wrapping_sub(0x8000_0000) as i32;
        out.push(signed as f32 / divisor);
      }
    }
    GenericAudioBufferRef::S8(b) => {
      let divisor = int_divisor(8)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      for s in b.iter_interleaved() {
        out.push(f32::from(s) / divisor);
      }
    }
    GenericAudioBufferRef::S16(b) => {
      let _ = int_divisor(16)?; // validate the bits arm (for parity + error shape)
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      // SIMD widen — collect the symphonia interleaved iterator
      // into a contiguous `Vec<i16>` once, then dispatch the NEON
      // s16-to-f32 normalizer over it in one shot. This keeps the
      // (1) cap discipline above and (2) the per-iteration sample
      // count contract intact, but removes the per-sample push +
      // divide loop.
      //
      // The materialization step (`buf_i16`) is bounded by
      // `b.samples_interleaved()` (already cap-checked) and uses
      // `try_reserve_exact` so an adversarial decoder cannot trigger
      // an infallible abort here. The subsequent dispatcher writes
      // into the pre-reserved spare capacity of `out` (the f32 sample
      // buffer) — no second copy.
      //
      // Multi-channel interleaved layout is preserved: `iter_interleaved`
      // yields samples in `[ch0_t0, ch1_t0, ..., ch0_t1, ch1_t1, ...]`
      // order, exactly what the f32 output buffer is contracted to
      // carry.
      let n = b.samples_interleaved();
      let mut buf_i16: Vec<i16> = Vec::new();
      buf_i16
        .try_reserve_exact(n)
        .map_err(|_| Error::OutOfMemory)?;
      buf_i16.extend(b.iter_interleaved());
      let spare: &mut [core::mem::MaybeUninit<f32>] = out.spare_capacity_mut();
      crate::simd::audio::pcm_decode::s16_to_f32_normalize(&mut spare[..n], &buf_i16);
      // SAFETY: dispatcher's function-level contract initializes
      // every f32 of `spare[..n]`; `out.len() + n <= cap` was
      // guaranteed by `reserve_under_cap`; capacity for `n` more
      // slots was reserved there too.
      unsafe { out.set_len(out.len() + n) };
    }
    GenericAudioBufferRef::S24(b) => {
      // `i24.inner()` returns the `[-2^23, 2^23)` value as `i32`.
      let divisor = int_divisor(24)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      // SIMD widen — collect the symphonia i24 iterator
      // (`.inner()` → i32 in `[-2^23, 2^23)`) into a contiguous
      // Vec<i32> once, then dispatch the NEON s32-to-f32 normalizer
      // with the 24-bit divisor (`1.0 / 2^23`). Same shape as the
      // S16 arm.
      let n = b.samples_interleaved();
      let mut buf_i32: Vec<i32> = Vec::new();
      buf_i32
        .try_reserve_exact(n)
        .map_err(|_| Error::OutOfMemory)?;
      buf_i32.extend(b.iter_interleaved().map(|s| s.inner()));
      let inv_scale = 1.0_f32 / divisor;
      let spare: &mut [core::mem::MaybeUninit<f32>] = out.spare_capacity_mut();
      crate::simd::audio::pcm_decode::s32_to_f32_normalize(&mut spare[..n], &buf_i32, inv_scale);
      // SAFETY: dispatcher's contract initializes every f32 in
      // `spare[..n]`; cap + capacity discharged by
      // `reserve_under_cap`.
      unsafe { out.set_len(out.len() + n) };
    }
    GenericAudioBufferRef::S32(b) => {
      let divisor = int_divisor(32)?;
      reserve_under_cap(out, b.samples_interleaved(), cap)?;
      // SIMD widen — collect symphonia's i32 iterator into a
      // contiguous Vec<i32>, then dispatch the NEON s32-to-f32
      // normalizer with the 32-bit divisor (`1.0 / 2^31`).
      let n = b.samples_interleaved();
      let mut buf_i32: Vec<i32> = Vec::new();
      buf_i32
        .try_reserve_exact(n)
        .map_err(|_| Error::OutOfMemory)?;
      buf_i32.extend(b.iter_interleaved());
      let inv_scale = 1.0_f32 / divisor;
      let spare: &mut [core::mem::MaybeUninit<f32>] = out.spare_capacity_mut();
      crate::simd::audio::pcm_decode::s32_to_f32_normalize(&mut spare[..n], &buf_i32, inv_scale);
      // SAFETY: dispatcher's contract initializes every f32 in
      // `spare[..n]`; cap + capacity discharged by
      // `reserve_under_cap`.
      unsafe { out.set_len(out.len() + n) };
    }
  }
  Ok(())
}

// Failure-injection hook for the
// post-metadata `meta_file.sync_all()` site inside `save_wav`. Only
// compiled under `cfg(test)`; the production binary never sees this
// flag. See `set_force_meta_fsync_failure` below for the full
// contract.
#[cfg(test)]
thread_local! {
  static FORCE_META_FSYNC_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only setter for the post-metadata fsync failure-injection
/// hook used by `save_wav`. When set to `true`, the next `save_wav`
/// invocation on the SAME THREAD that reaches the post-metadata fsync
/// site returns an injected `io::Error` instead of calling the real
/// `sync_all` — exercising the failure-propagation arm (cleanup
/// tempfile, NOT renaming, original bytes preserved). `pub(crate)` so
/// only the in-crate tests inside the [`tests`] module can flip the
/// flag.
///
/// **Thread-local scoping.** A process-global
/// `AtomicBool` would not work here: `SeqCst`
/// ordering prevents a data race but does NOT scope the injected
/// failure to the calling test — under the Rust test harness's
/// default parallel execution, any unrelated audio test that reached
/// the post-metadata fsync site while the flag was true would receive
/// the injected `sync_all` error instead of exercising real IO. The
/// flag is a `thread_local!` `Cell<bool>` so the injection is
/// scoped to the thread that called `set_force_meta_fsync_failure` —
/// concurrent tests on other threads observe the default `false` and
/// run their real-IO paths unaffected. The injecting test still uses
/// the `Drop`-guard reset pattern so a panic on the injecting thread
/// cannot leak `true` into a subsequent test that the harness happens
/// to schedule on the same worker thread (worker threads are reused
/// across tests within a binary).
#[cfg(test)]
pub(crate) fn set_force_meta_fsync_failure(b: bool) {
  FORCE_META_FSYNC_FAILURE.with(|cell| cell.set(b));
}

/// Post-metadata fsync helper. Called from [`save_wav`] AFTER the
/// permission and xattr restoration block and BEFORE the publishing
/// rename. Returns `Err` if the sync fails; the caller is responsible
/// for tempfile cleanup and `Err` propagation.
///
/// Factoring this site into a
/// named helper eliminates source-substring fragility: rather than
/// asserting that `meta_file.sync_all(`
/// appears in BOTH cfg-branches of an inline `let sync_result = ...`
/// binding (which a regression could satisfy with a stale comment or
/// string-literal mention of the token), the structural guard asserts a
/// single distinctive call to this helper appears between
/// `restore_xattrs(` and `fs::rename(` in the `save_wav` body. The
/// test-only failure-injection branch lives HERE, not at the call site —
/// so the call site is a single uncommented function call that cannot be
/// satisfied by a string-literal collision.
///
/// In a production build (`cfg(not(test))`) the function collapses to a
/// direct `meta_file.sync_all()` call with no flag-load overhead. The
/// `#[inline]` hint lets the optimiser fold the helper into the call
/// site as if it were an inline `let sync_result = ...` binding.
#[inline]
fn save_wav_post_metadata_fsync(meta_file: &File) -> std::io::Result<()> {
  #[cfg(test)]
  if FORCE_META_FSYNC_FAILURE.with(|cell| cell.get()) {
    return Err(std::io::Error::other(
      "test-injected meta fsync failure (FORCE_META_FSYNC_FAILURE)",
    ));
  }
  meta_file.sync_all()
}

/// Write `samples` to `path` as a 16-bit mono WAV at `sample_rate`.
///
/// Samples outside `[-1.0, 1.0]` are clipped (matches `mlx_audio.audio_io.write`'s
/// `np.clip(data, -1.0, 1.0)` pre-quantization), then multiplied by
/// `32768` (symmetric `* 32768` convention matching
/// `torchaudio.save` — `read` divides by `32768.0`, `write` multiplies
/// by `32768.0`, so the round-trip is bit-exact for in-range samples)
/// and converted to `i16` via Rust's saturating-cast (`+1.0 * 32768 =
/// +32768` saturates to `i16::MAX = +32767`; `-1.0 * 32768 = -32768 =
/// i16::MIN` is exact).
///
/// **All samples are validated finite UPFRONT**, before any tempfile is
/// opened, so a non-finite sample never leaves a partially-written WAV
/// on disk. This is stricter than `mlx_audio.audio_io.write` (which
/// would silently corrupt the WAV by casting `NaN → i16` to 0).
///
/// **Mid-write IO failure does not leave a partial WAV** — the
/// destination is updated atomically via tempfile + rename. The bytes
/// are first written to a tempfile in the SAME directory as `path`
/// (suffix `.<pid>.<rand>.tmp`), opened with `OpenOptions::create_new`
/// (= `O_CREAT|O_EXCL` on POSIX, `CREATE_NEW` on Windows) so an
/// attacker cannot pre-create the predictable temp path as a symlink
/// or replace it with an unrelated file — the open fails with
/// `AlreadyExists` and we retry with a fresh random name. Once the
/// tempfile is fully written, we `flush` the in-process buffer AND
/// call `File::sync_all` (fsync) so the data is durable on disk
/// before the rename — late-allocation / NFS / quota writeback errors
/// surface here, not after `Ok` has been returned. Pre-existing
/// destination permissions are captured up-front and re-applied to
/// the tempfile (via `fs::set_permissions`) before the rename, so a
/// 0600 audio file does not silently widen to the process umask.
/// Finally, `std::fs::rename` atomically substitutes the tempfile for
/// `path`. On POSIX `rename(2)` is atomic-within-fs; on Windows
/// `MoveFileEx` provides the same guarantee. Mid-write IO failures
/// (disk full, signal interruption, etc.) clean up the tempfile and
/// return [`Error::FileIo`] with the destination untouched — a
/// partial WAV cannot be observed at `path`. Note: the tempfile lives
/// in the same directory as `path` so the rename is single-fs
/// (cross-fs rename would silently fall back to copy+unlink and lose
/// the atomicity guarantee). **Extended attributes** (see #138):
/// (Linux user xattrs, POSIX-1.e ACLs stored as
/// `system.posix_acl_access`, SELinux contexts in `security.selinux`,
/// macOS xattrs, etc.) are captured from the existing destination
/// before the write and re-applied to the tempfile before the rename
/// (best-effort — per-xattr failures are silently dropped so a
/// security-namespace EPERM does not poison the entire save).
/// **Parent-dir fsync** (see #135): after the rename the parent directory is
/// `fsync(dirfd)`'d so the directory-entry update is durable on disk
/// — without this, a crash between rename and writeback can leave the
/// FS with no entry for the new file on ext4/xfs/APFS. Windows skips
/// the dirfd-fsync (no equivalent primitive); the pre-rename data
/// fsync we already issue covers that case.
/// **Original-handle fsync**: the writable tempfile handle is kept
/// alive across `set_permissions` + `restore_xattrs` and the
/// post-metadata `sync_all` runs on that ORIGINAL handle (not a
/// reopened one), so a read-only captured destination mode (e.g.
/// 0444) cannot cause the metadata fsync to be silently skipped via a
/// reopen EACCES. A failed post-metadata sync_all is propagated as
/// `Error::FileIo` and the rename is NOT attempted — the destination
/// stays at its pre-call contents.
///
/// # Errors
/// - [`Error::InvariantViolation`] if `sample_rate == 0`,
///   [`Error::CapExceeded`] if `samples.len()` exceeds the 16-bit-WAV limit
///   (`(u32::MAX - 36) / 2`), [`Error::OutOfRange`] if `sample_rate` exceeds
///   the byte-rate u32 ceiling (`u32::MAX / 2`),
///   [`Error::LayerKeyed`]/[`Error::NonFiniteScalar`] if any sample is
///   non-finite (NaN/inf), or [`Error::FileIo`] if the destination directory
///   has no `file_name` component, all tempfile retries (16) collide on
///   `AlreadyExists`, or the tempfile cannot be created / written /
///   flushed / renamed. On upfront validation failure the destination is
///   untouched; on mid-write failure the destination is still untouched
///   (tempfile removed, original `path` contents — if any — remain).
///
/// **Test-only failure injection** (`set_force_meta_fsync_failure`):
/// under `cfg(test)`, the `save_wav_post_metadata_fsync` helper
/// honors a thread-local flag that forces
/// an injected `io::Error` in place of the real fsync on the calling
/// thread only. This exists so the regression test
/// `save_wav_post_metadata_fsync_helper_is_called_before_rename_runtime`
/// can prove the error path (cleanup tempfile, NOT renaming, original
/// bytes preserved) — a behavioral guarantee userspace cannot
/// otherwise observe via real I/O on a writable destination. The hook
/// is completely absent from non-test builds (gated on `cfg(test)`),
/// and concurrent tests on other threads observe the default `false`
/// so unrelated audio tests cannot be poisoned by an injecting test.
pub fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
  // Allocate a fresh i16 scratch buffer; for the buffer-reuse path used
  // by streaming / batch writers, see [`save_wav_into`].
  let mut scratch: Vec<i16> = Vec::new();
  save_wav_into(path, samples, sample_rate, &mut scratch)
}

/// Quantize `src` into the `MaybeUninit<i16>` slice via the SIMD
/// dispatcher (`simd::audio::quantize::f32_to_i16_quantize`): clip to
/// `[-1, 1]`, multiply by `I16_MUL = 32767`, round-half-away-from-zero,
/// cast to `i16`. The single-source-of-truth quantizer used by
/// [`save_wav`] / [`save_wav_into`] and by future encoder writers
/// (e.g. flac/opus, when those land per the module doc) so every
/// destination format shares the same f32 → i16 conversion.
///
/// **Trait, not free function**, so a future encoder can keep its own
/// `Quantizer<f32, i16>` (or `<f32, i24>`, `<f32, u8>`) instance
/// without rewriting [`save_wav_into`]; the trait is the extension
/// point.
///
/// # Contract
/// - `dst.len() == src.len()` — one quantized output per input
///   sample. The implementation MUST initialize every cell of `dst`
///   before returning; callers may then `Vec::set_len(src.len())`.
///   Panics on length mismatch (debug + release — the misuse is a
///   caller bug, not a runtime input).
pub trait Quantizer<Source, Target> {
  /// Quantize `src` into `dst`. `dst.len() == src.len()`; the
  /// implementation initializes every cell of `dst` before returning.
  fn quantize_into(&self, dst: &mut [core::mem::MaybeUninit<Target>], src: &[Source]);
}

/// `f32` → `i16` quantizer used by [`save_wav`] / [`save_wav_into`] and
/// the shared SIMD path. Stateless newtype so the buffer-reuse
/// helper's signature is `&dyn Quantizer<f32, i16>` (vs. a free fn
/// taking a `&mut [MaybeUninit<i16>]`).
///
/// Wraps `simd::audio::quantize::f32_to_i16_quantize` (NEON 8-lane on
/// aarch64, scalar fallback elsewhere); see that module's docs for the
/// rounding mode (FCVTAS, ties away from zero — bit-exact match for
/// `f32::round`).
#[derive(Debug, Default, Clone, Copy)]
pub struct I16Quantizer;

impl Quantizer<f32, i16> for I16Quantizer {
  fn quantize_into(&self, dst: &mut [core::mem::MaybeUninit<i16>], src: &[f32]) {
    crate::simd::audio::quantize::f32_to_i16_quantize(dst, src);
  }
}

/// Write `samples` to `path` as a 16-bit mono WAV at `sample_rate`,
/// reusing `scratch`'s pre-allocated capacity for the f32 → i16
/// quantize buffer.
///
/// Mirrors [`save_wav`] (same atomic-rename + fsync + permission
/// preservation + 256 MiB cap + NaN/inf rejection), but a streaming /
/// batch writer can amortize the i16 scratch allocation across many
/// `save_wav_into` calls by passing the same `&mut Vec<i16>` each
/// time. The scratch is cleared (`scratch.clear()`) before writing;
/// existing contents are discarded but the underlying capacity is
/// retained. On any write error `scratch`'s length is unspecified;
/// the destination is unaffected (the tempfile-then-rename invariant
/// from [`save_wav`] holds here too).
///
/// `scratch` may be empty on the first call; subsequent calls reuse
/// whatever capacity the previous call left (clamped to `samples.len()`
/// for the next quantize).
///
/// # Errors
/// - Same as [`save_wav`].
pub fn save_wav_into(
  path: &Path,
  samples: &[f32],
  sample_rate: u32,
  scratch: &mut Vec<i16>,
) -> Result<()> {
  if sample_rate == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "save_wav: sample_rate",
      "must be > 0",
    )));
  }
  // Classic 16-bit WAV (the format we emit) has a 32-bit data-chunk size
  // field in bytes, AND a 32-bit RIFF chunk size field whose value is
  // `36 + data_size`. For mono i16 (`bytes_per_sample = 2`) the strict
  // upper bound on `samples.len()` is therefore `(u32::MAX - 36) / 2`
  // (so both `data_size = samples.len() * 2` and `file_size_minus_8 =
  // 36 + data_size` fit in u32 without wrap). Reject upfront — otherwise
  // a debug build would panic on the implicit add, and a release build
  // would silently wrap and produce an invalid RIFF header.
  // (RF64 / W64 large-WAV variants are a planned follow-up.)
  const MAX_MONO_I16_SAMPLES: usize = ((u32::MAX - 36) as usize) / 2;
  if samples.len() > MAX_MONO_I16_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "save_wav: sample count exceeds the 16-bit WAV total-file-size limit \
         (split into multiple files or use a large-WAV variant; \
         RF64 / W64 planned follow-up)",
      "MAX_MONO_I16_SAMPLES (= (u32::MAX - 36) bytes / 2-bytes-per-sample)",
      MAX_MONO_I16_SAMPLES as u64,
      samples.len() as u64,
    )));
  }
  // byte_rate = sample_rate * channels * bytes_per_sample (= sample_rate * 2
  // for mono i16). Must fit in u32 — reject sample_rate values whose
  // byte_rate would wrap. `u32::MAX / 2 = 2147483647` (~2.1 GHz, well
  // above any real audio sample rate).
  const MAX_SAMPLE_RATE_FOR_MONO_I16: u32 = u32::MAX / 2;
  if sample_rate > MAX_SAMPLE_RATE_FOR_MONO_I16 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "save_wav: sample_rate (must keep byte_rate = sample_rate * 2 within u32)",
      "must be <= MAX_SAMPLE_RATE_FOR_MONO_I16 (= u32::MAX / 2 = 2147483647)",
      format_smolstr!("{sample_rate}"),
    )));
  }
  // Pre-validate ALL samples before any filesystem mutation, so a NaN/inf
  // in the buffer cannot leave a partially-written WAV on disk. This
  // departs from `mlx_audio.audio_io.write` (which would silently cast
  // NaN to 0 via `astype(int16)`), but the cost is one extra scan and
  // the gain is "destination integrity is preserved on input error".
  for (i, &s) in samples.iter().enumerate() {
    if !s.is_finite() {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        format_smolstr!("save_wav: sample index {i}"),
        Error::NonFiniteScalar(NonFiniteScalarPayload::new(
          "save_wav: sample (cannot quantize)",
          s as f64,
        )),
      )));
    }
  }

  // Capture the existing destination's permissions (if it exists) so
  // the post-rename file keeps the user's chosen mode/ACL — otherwise
  // a private 0600 audio file silently widens to whatever the process
  // umask grants on the fresh tempfile inode. `None` means the
  // destination doesn't exist yet; in that case the tempfile keeps
  // its umask-granted mode.
  let existing_perms = fs::metadata(path).ok().map(|m| m.permissions());

  // Capture the existing destination's extended
  // attributes (Linux user xattrs, macOS xattrs, SELinux contexts, etc.)
  // so the post-rename file inherits the user's pre-existing xattr set.
  // Pre-C performance: the read happens ONCE upfront before any
  // filesystem mutation, mirroring `existing_perms`. Failures during
  // the read are silently dropped (best-effort) — a destination that
  // doesn't exist returns `None`; a filesystem that doesn't support
  // xattrs (or returns ENOTSUP for one specific xattr) skips that
  // namespace without failing the save. The `capture_xattrs` helper has
  // a `#[cfg(not(unix))]` no-op stub returning `None` for non-Unix
  // platforms (the `xattr` crate is declared under
  // `[target.'cfg(unix)'.dependencies]` because it does NOT compile on
  // Windows — see Cargo.toml).
  // Note: POSIX-1.e ACLs (stored in `system.posix_acl_access`) and
  // SELinux contexts (`security.selinux`) are explicitly probed by
  // `capture_xattrs` because `listxattr` is allowed to omit them — the
  // generic `xattr::list` walk alone is NOT sufficient.
  let existing_xattrs: Option<Vec<(std::ffi::OsString, Vec<u8>)>> = capture_xattrs(path);

  // Open a tempfile in the SAME directory as `path` so the subsequent
  // `fs::rename` stays single-fs (cross-fs rename silently falls back
  // to copy+unlink and loses the atomicity guarantee). Use
  // `OpenOptions::create_new(true)` (= O_CREAT|O_EXCL on POSIX,
  // CREATE_NEW on Windows) so we cannot follow an attacker-precreated
  // symlink or truncate an unrelated file at the predictable temp path
  // — the open fails with `AlreadyExists` and we retry with a fresh
  // randomized name. A small bounded retry budget (16) is overkill in
  // practice (the random suffix is 64 bits + a per-call atomic counter)
  // but guards against pathological precreation campaigns.
  const MAX_TEMPFILE_OPEN_RETRIES: u32 = 16;
  let (tmp_path, file) = open_excl_tempfile(path, MAX_TEMPFILE_OPEN_RETRIES)?;

  // Inner write closure: returns the writable `File` handle on success
  // so we can KEEP IT ALIVE past `set_permissions` and `restore_xattrs`
  // and call the post-metadata `sync_all` ON THE ORIGINAL handle (see
  // the post-metadata-fsync block below). On
  // failure we clean up the tempfile (best-effort) before the rename.
  let write_result = (|| -> Result<File> {
    let mut writer = BufWriter::new(file);

    // Build the 44-byte RIFF/WAVE/fmt/data header. Layout:
    //   +0  "RIFF"           (4 bytes, ASCII)
    //   +4  u32 LE           file_size - 8  (= 36 + data_size)
    //   +8  "WAVE"           (4 bytes)
    //   +12 "fmt "           (4 bytes, trailing space)
    //   +16 u32 LE = 16      (fmt chunk size for PCM)
    //   +20 u16 LE = 1       (audio format: PCM = 1)
    //   +22 u16 LE = 1       (channels)
    //   +24 u32 LE           (sample_rate)
    //   +28 u32 LE           (byte_rate = sample_rate * channels * bits/8)
    //   +32 u16 LE = 2       (block_align = channels * bits/8)
    //   +34 u16 LE = 16      (bits_per_sample)
    //   +36 "data"           (4 bytes)
    //   +40 u32 LE           (data_size = samples.len() * 2 for mono i16)
    //   +44 ... samples_i16_le ...
    //
    // All multi-byte fields are little-endian per the WAV spec.
    // `samples.len() <= MAX_MONO_I16_SAMPLES = (u32::MAX - 36) / 2`
    // and `sample_rate <= MAX_SAMPLE_RATE_FOR_MONO_I16 = u32::MAX / 2`
    // were both enforced above, so `data_size`, `file_size_minus_8`,
    // and `byte_rate` all fit in u32 without wrap. We still use checked
    // arithmetic + `.expect` against the cap so a future cap bug
    // produces a deterministic panic message instead of a wrapped
    // header field; in the current code paths the expects are
    // unreachable.
    const BITS_PER_SAMPLE: u16 = 16;
    const CHANNELS: u16 = 1;
    const BLOCK_ALIGN: u16 = CHANNELS * (BITS_PER_SAMPLE / 8);
    let data_size: u32 = u32::try_from(samples.len())
      .ok()
      .and_then(|n| n.checked_mul(u32::from(BLOCK_ALIGN)))
      .expect("save_wav: data_size overflow despite MAX_MONO_I16_SAMPLES cap");
    let file_size_minus_8: u32 = 36u32
      .checked_add(data_size)
      .expect("save_wav: file_size overflow despite MAX_MONO_I16_SAMPLES cap");
    let byte_rate: u32 = sample_rate
      .checked_mul(u32::from(CHANNELS))
      .and_then(|n| n.checked_mul(u32::from(BITS_PER_SAMPLE / 8)))
      .expect("save_wav: byte_rate overflow despite MAX_SAMPLE_RATE_FOR_MONO_I16 cap");

    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&file_size_minus_8.to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes());
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&BLOCK_ALIGN.to_le_bytes());
    header[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_size.to_le_bytes());
    writer.write_all(&header).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "save_wav: header write failed",
        FileOp::Write,
        tmp_path.clone(),
        e,
      ))
    })?;

    // Quantize all samples first via the SIMD dispatcher
    // (`simd::audio::quantize::f32_to_i16_quantize`) — clip to `[-1, 1]`,
    // multiply by `I16_MUL = 32768` (symmetric `* 32768`
    // convention; the +1.0 extreme saturates to i16::MAX via the cast),
    // round-half-away-from-zero, cast to i16. On `aarch64` this routes
    // to an 8-lane NEON tile (vminq/vmaxq clamp → vmulq_n scale →
    // vcvtaq_s32 round (FCVTAS, ties away from zero — bit-exact match
    // for `f32::round`) → vqmovn+vcombine narrow → vst1q_s16 store);
    // elsewhere it falls back to the scalar `clamp + round + cast` loop.
    // Tracking [#152].
    //
    // The dispatcher takes `&mut [MaybeUninit<i16>]` (type-encoded
    // uninit safety), so we pre-reserve via `try_reserve_exact` (so a
    // multi-GB sample buffer cannot trigger an infallible abort here),
    // pass the spare capacity directly, and `set_len` after every i16
    // has been written. Then a SINGLE `write_all` writes the entire
    // i16 byte view in one syscall — replacing the per-sample
    // BufWriter pushes.
    // Reuse the caller-provided `scratch` for the i16 quantize buffer.
    // `clear()` drops the prior contents but retains the allocated
    // capacity, so a streaming caller (one `Vec<i16>` shared across
    // many `save_wav_into` calls) pays one growth at the high-water
    // mark and skips per-call allocations afterward. The shared
    // quantizer dispatcher is invoked through the `Quantizer` trait
    // so future encoder writers (flac/opus, when those land) reuse
    // the same fused f32 → i16 path.
    scratch.clear();
    scratch
      .try_reserve_exact(samples.len())
      .map_err(|_| Error::OutOfMemory)?;
    {
      let spare: &mut [core::mem::MaybeUninit<i16>] = scratch.spare_capacity_mut();
      // `samples.len() <= spare.len()` because `try_reserve_exact(samples.len())`
      // above reserved exactly that much additional capacity.
      debug_assert!(spare.len() >= samples.len());
      I16Quantizer.quantize_into(&mut spare[..samples.len()], samples);
    }
    // Rebind `quantized` to a local immutable view of the scratch (so
    // the subsequent byte-view / write code reads naturally as
    // "quantized samples", matching the prior `save_wav` body).
    // SAFETY: `f32_to_i16_quantize` (via the `Quantizer` impl) wrote
    // every i16 in `0..samples.len()` of the spare capacity (function-
    // level contract). `Vec::set_len`'s preconditions: (1)
    // `samples.len() <= scratch.capacity()` — the `try_reserve_exact`
    // succeeded; (2) elements at `[0..samples.len()]` are initialized —
    // kernel contract guarantees this.
    unsafe { scratch.set_len(samples.len()) };
    let quantized: &Vec<i16> = scratch;

    // Single bulk write of the entire i16 buffer as little-endian bytes.
    // `to_le` on i16 is a no-op on LE hosts (the common case) and a
    // bswap on BE hosts; per-element. After the conversion, the
    // `&[i16]` is reinterpreted as `&[u8]` of `2 * samples.len()` bytes
    // and written in one shot. The wav format stores i16 samples in
    // little-endian order regardless of host endianness.
    if cfg!(target_endian = "little") {
      // SAFETY: `quantized` is `Vec<i16>` (len = samples.len(),
      // cap = samples.len()), all initialized above. On a
      // little-endian host the i16 in-memory representation IS the
      // LE byte order required by the WAV format — reinterpret the
      // contiguous slice as `&[u8]` of double the length for a
      // single zero-copy `write_all`. `i16` and `u8` have well-
      // defined layouts (no padding, no validity invariants), so a
      // borrow of one as the other via `from_raw_parts` is sound.
      // The borrow lives only for the `write_all` call below.
      let byte_view: &[u8] = unsafe {
        core::slice::from_raw_parts(
          quantized.as_ptr().cast::<u8>(),
          quantized.len().checked_mul(2).expect(
            "save_wav: byte-view length overflow (cap was MAX_MONO_I16_SAMPLES, * 2 ≤ u32::MAX - 36)",
          ),
        )
      };
      writer.write_all(byte_view).map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "save_wav: bulk sample write failed",
          FileOp::Write,
          tmp_path.clone(),
          e,
        ))
      })?;
    } else {
      // Big-endian host: byteswap each i16 into a small stack buffer
      // and write in chunks. Not benchmarked; rare path.
      const CHUNK: usize = 1024;
      let mut buf = [0u8; CHUNK * 2];
      let mut idx = 0;
      while idx < quantized.len() {
        let n = (quantized.len() - idx).min(CHUNK);
        for (i, &q) in quantized[idx..idx + n].iter().enumerate() {
          buf[i * 2..i * 2 + 2].copy_from_slice(&q.to_le_bytes());
        }
        writer.write_all(&buf[..n * 2]).map_err(|e| {
          Error::FileIo(FileIoPayload::new(
            "save_wav: bulk sample write failed",
            FileOp::Write,
            tmp_path.clone(),
            e,
          ))
        })?;
        idx += n;
      }
    }

    // BufWriter does NOT auto-flush on drop into a Result, and a missed
    // flush would leave us renaming an incomplete tempfile into place.
    writer.flush().map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "save_wav: flush failed",
        FileOp::Flush,
        tmp_path.clone(),
        e,
      ))
    })?;
    // `flush()` only drains the in-process buffer to the OS; on delayed-
    // allocation filesystems, NFS, quotas, etc. a writeback / late-ENOSPC
    // failure would otherwise be observed only on close (whose result Drop
    // discards) — meaning we could rename in a tempfile whose contents
    // never actually hit the disk. `sync_all` (fsync) forces the data +
    // metadata to durable storage, and we propagate its error so we
    // never rename a not-yet-durable tempfile into place.
    let inner = writer.into_inner().map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "save_wav: BufWriter::into_inner failed (final flush did not reach underlying File)",
        FileOp::Flush,
        tmp_path.clone(),
        e.into_error(),
      ))
    })?;
    inner.sync_all().map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "save_wav: sync_all failed",
        FileOp::Fsync,
        tmp_path.clone(),
        e,
      ))
    })?;
    // RETURN the writable `File`
    // handle to the outer scope instead of dropping it here. The outer
    // code re-uses this exact handle for the post-metadata `sync_all`
    // (after `set_permissions` + `restore_xattrs`), which avoids a
    // reopen-with-OpenOptions-write-true hazard: if the captured
    // destination perms were read-only (e.g. 0444) we already restored
    // those perms onto the tempfile via `fs::set_permissions`, so a
    // subsequent reopen-for-write would correctly fail with `EACCES`
    // — and that EACCES would be silently
    // swallowed by `if let Ok(..)` and the metadata fsync
    // skipped, letting the rename publish a file whose chmod/xattrs
    // weren't yet on stable storage. Keeping the original RW handle
    // alive sidesteps that whole reopen — `sync_all` on an already-
    // open writable handle is unaffected by mode bits on the inode.
    // The outer code drops the handle EXPLICITLY before
    // `fs::rename` (Windows in particular dislikes renaming an open
    // file handle).
    Ok(inner)
  })();

  let meta_file = match write_result {
    Ok(f) => f,
    Err(err) => {
      // Best-effort tempfile cleanup. Don't fail the call on cleanup
      // failure — the original `err` is what the caller needs to see.
      let _ = fs::remove_file(&tmp_path);
      return Err(err);
    }
  };

  // Restore the destination's prior permissions BEFORE the rename so
  // the post-rename file matches the user's pre-existing mode bits.
  // Skipped when the destination didn't previously exist (the
  // tempfile's umask-granted mode is the natural default for new files).
  // Failure here is treated like any other write-path failure: clean up
  // the tempfile and propagate. The xattr block immediately below
  // covers ACLs / SELinux contexts / macOS extended attributes — which
  // are stored as ordinary xattrs on every Unix that exposes them.
  if let Some(perms) = existing_perms.clone()
    && let Err(e) = fs::set_permissions(&tmp_path, perms)
  {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::FileIo(FileIoPayload::new(
      "save_wav: set_permissions on tempfile failed",
      FileOp::Other("set_permissions"),
      tmp_path,
      e,
    )));
  }

  // Restore the destination's prior xattrs onto
  // the tempfile BEFORE the rename so the published file inherits the
  // pre-existing extended-attribute set (Linux user xattrs, POSIX-1.e
  // ACLs as `system.posix_acl_access`, SELinux contexts in
  // `security.selinux`, macOS xattrs, etc.). Best-effort: per-xattr
  // write failures (ENOTSUP for a namespace the tempfile's filesystem
  // doesn't accept, EPERM for a restricted `security.*` xattr the
  // current user can't set on a fresh inode) are silently dropped —
  // we cannot make atomic-rename + xattr-preservation strictly atomic
  // across kernels, and a per-xattr write failure should NOT cause the
  // entire save to roll back (the WAV bytes are still valid; the user's
  // bigger pain is "the save failed altogether"). On Windows the
  // `xattr` crate isn't linked (target-specific dep — see Cargo.toml),
  // and the local `restore_xattrs` stub is a no-op there.
  if let Some(xattrs) = &existing_xattrs {
    restore_xattrs(&tmp_path, xattrs);
  }

  // Post-metadata fsync via the
  // ORIGINAL writable handle kept alive from tempfile creation. The
  // `inner.sync_all()` (inside the closure, above the
  // `set_permissions` / `restore_xattrs` block) made the WAV BYTES
  // durable, but the subsequent `set_permissions` and per-xattr `set`
  // calls mutate inode metadata (mode bits, ACL/security xattrs) which
  // are NOT yet on stable storage. The follow-up `fsync_parent_dir`
  // after the rename only makes the new directory entry durable — it
  // does not flush the tempfile inode's metadata. So a crash AFTER the
  // rename but BEFORE background writeback could leave the published
  // file with the new bytes but STALE permissions/xattrs (mode
  // reverted to umask, ACLs lost, SELinux label gone).
  //
  // Reopening the tempfile with `OpenOptions::write(true)` would be unsafe
  // here: after `set_permissions(tmp_path, captured_perms)` restores a
  // read-only captured mode (e.g. 0444 from the destination), the
  // reopen could fail with EACCES (the process owns the inode so the
  // chmod succeeded, but a subsequent write-open is gated by the inode
  // mode), AND that failure would be silently swallowed by `if let Ok(..)`
  // — letting the rename publish a file whose chmod/xattrs weren't on
  // stable storage. Use the ORIGINAL handle we kept open from tempfile
  // creation; `sync_all` on an existing writable File handle is
  // unaffected by inode mode bits. AND TREAT FAILURE AS AN ERROR
  // whenever we restored metadata — silently proceeding past a failed
  // metadata fsync would defeat the entire point of this block. On a
  // fresh destination (no metadata to restore) the post-write
  // `sync_all` already inside the closure covers data + metadata, so
  // we skip the redundant second sync.
  // The post-metadata fsync lives
  // in the [`save_wav_post_metadata_fsync`] helper, which folds the
  // test-only failure-injection branch inside the helper body. The call
  // site is a single uncommented function call, so the source-structural
  // guard `save_wav_calls_post_metadata_fsync_helper_before_rename`
  // can assert the distinctive helper name (rather than chasing the
  // `meta_file.sync_all(` token across two cfg-branches and risking
  // string-literal false positives). In `cfg(not(test))` the helper
  // collapses to a direct `meta_file.sync_all()` call with no
  // flag-load overhead — see [`FORCE_META_FSYNC_FAILURE`] for the
  // test-only setter contract.
  let sync_result = save_wav_post_metadata_fsync(&meta_file);
  if (existing_perms.is_some() || existing_xattrs.is_some())
    && let Err(e) = sync_result
  {
    // Best-effort tempfile cleanup before propagating — the published
    // destination at `path` is still untouched (we never started the
    // rename), and a leftover tempfile under `<path>.<pid>.<rand>.tmp`
    // would be visible to operators.
    drop(meta_file);
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::FileIo(FileIoPayload::new(
      "save_wav: post-metadata sync_all on tempfile failed \
         (perms/xattrs were restored but metadata is not yet durable; \
          NOT renaming to final destination — see #138)",
      FileOp::Fsync,
      tmp_path,
      e,
    )));
  }
  // Drop the writable handle BEFORE the rename. Windows in particular
  // dislikes renaming an open file handle, and on POSIX the rename
  // itself doesn't care but closing first matches the cross-platform
  // contract.
  drop(meta_file);

  // Atomic-within-fs rename. POSIX `rename(2)` and Windows `MoveFileEx`
  // both make the destination point at the new bytes atomically — no
  // observer can see a half-written WAV at `path`.
  if let Err(e) = fs::rename(&tmp_path, path) {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::FileIo(FileIoPayload::new(
      "save_wav: rename tempfile -> destination failed",
      FileOp::Rename,
      tmp_path,
      e,
    )));
  }

  // Fsync the PARENT directory after the rename.
  // On most filesystems (ext4 default, xfs, APFS) the rename itself is
  // atomic, but the directory-entry update isn't durable until the
  // parent inode is fsync'd. A crash between rename and parent-fsync
  // can leave the filesystem with the OLD entry (no observation of the
  // new file) or — on some FS configurations — the tempfile entry
  // persisting under its random name. Open the parent in read mode
  // (POSIX requires read for `fsync(dirfd)`) and propagate sync_all's
  // error. Best-effort on Windows (and on any platform where the
  // parent open fails) — Windows has no equivalent durability primitive
  // for the directory entry, so we silently skip and rely on the
  // post-rename data fsync we already issued before the rename.
  fsync_parent_dir(path);
  Ok(())
}

/// Helper (see #138): read every extended attribute from
/// `path`, returning `Some(vec)` on success (including the empty case
/// when the file exists but has no xattrs) and `None` when the read
/// cannot be attempted (path doesn't exist, the platform doesn't
/// support xattrs at all, or any of the listing/getting syscalls
/// fails for an unrecoverable reason). Per-xattr `get` failures are
/// silently dropped — the resulting vec carries the subset we could
/// read, and the caller's `restore_xattrs` is best-effort.
///
/// Unix arm. In addition to walking `xattr::list(path)`, we EXPLICITLY
/// probe a fixed set of known ACL/security xattr names (see #138). The
/// `xattr` crate documents that `list` may omit
/// `system.*` names entirely on some kernels (POSIX-1.e ACLs are
/// commonly hidden from `listxattr`) and that `trusted.*` is only
/// listed when the caller is root; SELinux's `security.selinux` and
/// IMA's `security.ima`/`security.evm` can likewise be filtered out of
/// `list` depending on kernel config. Without the explicit probe a
/// destination with a POSIX ACL or an SELinux label would silently
/// lose those attributes on overwrite even though the namespace was
/// captured-by-name. The explicit-probe path uses `xattr::get(path,
/// name)` which talks to `getxattr(2)` directly and ignores any
/// listxattr-side filtering.
///
/// Duplicate names (a name that appears in both `list` and the
/// explicit-probe set) are deduplicated — the explicit-probe value
/// wins, since the two reads can race against a concurrent writer
/// and the more-recent read is closer to ground truth.
#[cfg(unix)]
fn capture_xattrs(path: &Path) -> Option<Vec<(std::ffi::OsString, Vec<u8>)>> {
  if !path.exists() {
    return None;
  }
  let names = xattr::list(path).ok()?;
  // Use an insertion-ordered Vec keyed by name; later writes overwrite
  // earlier ones so the explicit-probe block (which runs after the
  // list walk) takes precedence on conflict.
  let mut out: Vec<(std::ffi::OsString, Vec<u8>)> = Vec::new();
  let upsert =
    |out: &mut Vec<(std::ffi::OsString, Vec<u8>)>, name: std::ffi::OsString, value: Vec<u8>| {
      if let Some(slot) = out.iter_mut().find(|(n, _)| n == &name) {
        slot.1 = value;
      } else {
        out.push((name, value));
      }
    };
  for name in names {
    if let Ok(Some(value)) = xattr::get(path, &name) {
      upsert(&mut out, name, value);
    }
  }
  // Explicit-probe set for ACL / security xattrs that `listxattr` is
  // documented to potentially omit. The names are byte-stable across
  // every Linux/Android/macOS/BSD kernel that exposes them; absent
  // names simply yield `Ok(None)` and are skipped.
  //   - system.posix_acl_access  — POSIX-1.e access ACL (Linux/Hurd)
  //   - system.posix_acl_default — POSIX-1.e default ACL (rarely set
  //                                on files, but present on dirs the
  //                                file was inherited from; cheap to
  //                                probe regardless)
  //   - security.selinux         — SELinux process/file context
  //   - security.capability      — `getcap`/`setcap` file capabilities
  //   - security.ima             — IMA file-integrity hash
  //   - security.evm             — EVM metadata signature
  const EXPLICIT_PROBES: &[&str] = &[
    "system.posix_acl_access",
    "system.posix_acl_default",
    "security.selinux",
    "security.capability",
    "security.ima",
    "security.evm",
  ];
  for &probe in EXPLICIT_PROBES {
    if let Ok(Some(value)) = xattr::get(path, probe) {
      upsert(&mut out, std::ffi::OsString::from(probe), value);
    }
  }
  Some(out)
}

/// Helper (see #138), Windows / non-Unix arm. The `xattr`
/// crate isn't linked on non-Unix targets (see Cargo.toml — `xattr` is
/// declared under `[target.'cfg(unix)'.dependencies]` because the
/// crate imports `std::os::unix::io` unconditionally and does not
/// compile on Windows). Return `None` so the call site's
/// `restore_xattrs` arm is also a no-op — the published WAV simply
/// gets the tempfile's default (umask-granted) metadata on Windows,
/// matching that platform's umask-default behavior.
#[cfg(not(unix))]
fn capture_xattrs(_path: &Path) -> Option<Vec<(std::ffi::OsString, Vec<u8>)>> {
  None
}

/// Helper (see #138): re-apply the xattrs captured by
/// [`capture_xattrs`] onto `path`. Per-xattr write failures are
/// silently dropped (see the call-site comment in `save_wav` for the
/// rationale).
#[cfg(unix)]
fn restore_xattrs(path: &Path, xattrs: &[(std::ffi::OsString, Vec<u8>)]) {
  for (name, value) in xattrs {
    let _ = xattr::set(path, name, value);
  }
}

/// Windows / non-Unix arm for `restore_xattrs`. No-op stub — the
/// matching `capture_xattrs` returns `None` on this platform, so the
/// caller's `if let Some(...)` branch never invokes this function. The
/// stub exists so the call site does not need a `#[cfg(unix)]`
/// guard of its own.
#[cfg(not(unix))]
fn restore_xattrs(_path: &Path, _xattrs: &[(std::ffi::OsString, Vec<u8>)]) {}

/// Helper (see #135): open the parent directory of `path` in
/// read mode and `fsync` it, so the directory entry produced by the
/// preceding `fs::rename` is durable on the filesystem. Best-effort —
/// any failure (no parent, parent-open failure, sync failure) is
/// silently dropped; the caller has already fsync'd the data bytes
/// before the rename, so a parent-sync skip only weakens the rename
/// durability claim (which is the platform default anyway), it does
/// NOT compromise the renamed file's contents.
///
/// POSIX-only in practice — Windows has no `fsync(dirfd)` equivalent
/// (`MoveFileEx` provides a separate durability flag instead), and the
/// `OpenOptions::read(true).open(<dir>)` call returns `Err(EISDIR)` /
/// `ERROR_ACCESS_DENIED` on Windows, which we silently drop.
fn fsync_parent_dir(path: &Path) {
  let Some(parent) = path.parent() else {
    return;
  };
  // An empty parent path (the relative-path-with-no-slashes case)
  // means "the current directory" — open `"."` explicitly so the open
  // does not fail with ENOENT.
  let parent_path: &Path = if parent.as_os_str().is_empty() {
    Path::new(".")
  } else {
    parent
  };
  if let Ok(dir) = std::fs::OpenOptions::new().read(true).open(parent_path) {
    let _ = dir.sync_all();
  }
}

/// Build a randomized tempfile path in the same directory as `final_path`,
/// of the form `<file_name>.<pid>.<rand>.tmp`, and open it with
/// `OpenOptions::create_new(true)` (= `O_CREAT|O_EXCL` on POSIX,
/// `CREATE_NEW` on Windows). The exclusive-create guarantee means we
/// cannot follow an attacker-precreated symlink or truncate an unrelated
/// file at the predictable temp path — `AlreadyExists` triggers a retry
/// with a fresh random name. The same-directory invariant keeps the
/// subsequent `fs::rename` single-fs (atomic on POSIX/Windows).
///
/// Bounded retries (`max_retries`) prevent a pathological precreation
/// campaign from looping indefinitely; the 64-bit random suffix mixed
/// with a per-call atomic counter and a nanos timestamp makes a
/// collision under normal operation vanishingly unlikely.
fn open_excl_tempfile(final_path: &Path, max_retries: u32) -> Result<(PathBuf, File)> {
  use std::{
    fs::OpenOptions,
    io::ErrorKind,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
  };
  static COUNTER: AtomicU64 = AtomicU64::new(0);

  let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
  let file_name = final_path
    .file_name()
    .ok_or_else(|| {
      Error::OutOfRange(OutOfRangePayload::new(
        "save_wav: destination path",
        "must have a file_name component (not a bare directory or `..`)",
        format_smolstr!("{}", final_path.display()),
      ))
    })?
    .to_string_lossy()
    .into_owned();
  let pid = std::process::id();
  let mut last_err: Option<std::io::Error> = None;
  for _ in 0..max_retries {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let rand = nanos ^ counter.rotate_left(17);
    let candidate = parent.join(format!("{file_name}.{pid}.{rand:016x}.tmp"));
    match OpenOptions::new()
      .write(true)
      .create_new(true)
      .open(&candidate)
    {
      Ok(file) => return Ok((candidate, file)),
      Err(e) if e.kind() == ErrorKind::AlreadyExists => {
        last_err = Some(e);
        continue;
      }
      Err(e) => {
        return Err(Error::FileIo(FileIoPayload::new(
          "save_wav: tempfile create_new failed",
          FileOp::Create,
          candidate,
          e,
        )));
      }
    }
  }
  Err(Error::FileIo(FileIoPayload::new(
    "save_wav: exhausted tempfile retry budget (all candidate names collided \
       with AlreadyExists)",
    FileOp::Create,
    final_path.to_path_buf(),
    last_err.unwrap_or_else(|| {
      std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "tempfile retries exhausted",
      )
    }),
  )))
}

/// Naive linear-interpolation resample from `from_rate` to `to_rate`.
///
/// For each output index `i` (`0 <= i < out_len`), the source position is
/// `x = i * from_rate / to_rate`, and the output is the linear blend of
/// `samples[floor(x)]` and `samples[ceil(x)]`. This matches the simplest
/// `mlx-audio` resampling utility (used by Whisper preprocessing) and is
/// faster but lower fidelity than `scipy.signal.resample_poly` /
/// `libsamplerate`. A sinc/polyphase resampler is a planned follow-up.
///
/// Returns an empty `Vec` when `samples` is empty. When `from_rate == to_rate`
/// the input is returned as a verbatim copy (no interpolation, no
/// floating-point rounding drift) — still subject to the same
/// [`MAX_RESAMPLED_SAMPLES`] cap and fallible reservation as the resampling
/// path, so an over-cap input yields [`Error::CapExceeded`] rather than an
/// infallible allocation that could abort.
///
/// # Errors
/// - [`Error::InvariantViolation`] if `from_rate == 0` or `to_rate == 0`,
///   [`Error::ArithmeticOverflow`] if the output length overflows `usize`,
///   [`Error::OutOfRange`] if the output length doesn't fit `usize`,
///   [`Error::CapExceeded`] if it exceeds [`MAX_RESAMPLED_SAMPLES`] (64 Mi-samples),
///   or [`Error::AllocFailure`] if the Vec reservation fails.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
  if from_rate == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "resample_linear: from_rate",
      "must be > 0",
    )));
  }
  if to_rate == 0 {
    return Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "resample_linear: to_rate",
      "must be > 0",
    )));
  }
  if samples.is_empty() {
    return Ok(Vec::new());
  }
  if from_rate == to_rate {
    // Verbatim copy — avoids any FP rounding drift on a no-op resample. The
    // output length equals the input length, so it is bounded by the same cap
    // and built through the same fallible reservation as the resampling path
    // below — never an infallible `to_vec` that could abort on OOM, and never
    // a silent bypass of `MAX_RESAMPLED_SAMPLES`.
    if samples.len() > MAX_RESAMPLED_SAMPLES {
      return Err(Error::CapExceeded(CapExceededPayload::new(
        "resample_linear: output length exceeds cap \
           (use chunked resampling or raise the cap manually)",
        "MAX_RESAMPLED_SAMPLES",
        MAX_RESAMPLED_SAMPLES as u64,
        samples.len() as u64,
      )));
    }
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(samples.len()).map_err(|e| {
      Error::AllocFailure(AllocFailurePayload::new(
        "resample_linear: reservation",
        "samples",
        samples.len() as u64,
        e,
      ))
    })?;
    out.extend_from_slice(samples);
    return Ok(out);
  }

  // Output length: `samples.len() * to_rate / from_rate`. Use u64 to avoid
  // overflow in the intermediate product; then check the final fits in
  // usize so we don't silently truncate on 32-bit targets.
  let in_len = samples.len() as u64;
  let out_len_u64 = in_len.checked_mul(u64::from(to_rate)).ok_or_else(|| {
    Error::ArithmeticOverflow(ArithmeticOverflowPayload::with_operands(
      "resample_linear: in_len * to_rate",
      "u64",
      [("in_len", in_len), ("to_rate", u64::from(to_rate))],
    ))
  })?
    / u64::from(from_rate);
  let out_len = usize::try_from(out_len_u64).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "resample_linear: output length",
      "must fit in usize",
      format_smolstr!("{out_len_u64}"),
    ))
  })?;
  if out_len == 0 {
    return Ok(Vec::new());
  }

  // Hard cap on the output buffer — defends against `from_rate=1,
  // to_rate=u32::MAX` (or similar adversarial ratios) that would attempt
  // a tens-of-GB allocation. The cap matches `load_audio`'s
  // `MAX_DECODED_SAMPLES` (64 Mi-samples ≈ 256 MiB of f32).
  if out_len > MAX_RESAMPLED_SAMPLES {
    return Err(Error::CapExceeded(CapExceededPayload::new(
      "resample_linear: output length exceeds cap \
         (use chunked resampling or raise the cap manually)",
      "MAX_RESAMPLED_SAMPLES",
      MAX_RESAMPLED_SAMPLES as u64,
      out_len as u64,
    )));
  }

  let mut out: Vec<f32> = Vec::new();
  out.try_reserve_exact(out_len).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "resample_linear: reservation",
      "samples",
      out_len as u64,
      e,
    ))
  })?;
  let ratio = f64::from(from_rate) / f64::from(to_rate);

  // SIMD: dispatch to the resample-linear NEON kernel
  // (`crate::simd::audio::resample`). On `aarch64` this routes to a
  // 4-lane NEON tile (`vmulq_f64` index math + `vfmaq_f32` FMA + scalar
  // gather for `samples[lo_idx]` / `samples[hi_idx]`); elsewhere it
  // falls back to the scalar shape preserved in
  // `simd::audio::resample::resample_linear_scalar`.
  //
  // The dispatcher takes a `&mut [MaybeUninit<f32>]` sized to `out_len`
  // and writes every slot via `MaybeUninit::write` before returning, so
  // `set_len(out_len)` after the kernel returns is sound.
  let spare = out.spare_capacity_mut();
  crate::simd::audio::resample::resample_linear(&mut spare[..out_len], samples, ratio);
  // SAFETY: the SIMD dispatcher's init contract guarantees every f32 of
  // the `out_len`-prefix of `spare` is initialized before returning;
  // `out_len <= out.capacity()` per the `try_reserve_exact(out_len)`
  // above.
  unsafe { out.set_len(out_len) };
  Ok(out)
}

#[cfg(test)]
mod tests;
