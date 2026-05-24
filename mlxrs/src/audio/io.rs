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
//! with [`Error::Backend`] (the `Vec<f32>` return shape cannot
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

use crate::error::{Error, Result};

/// PCM-16 full-scale on `read`; matches `mlx-audio`'s `int16 → float32`
/// divisor on `read` (the reference uses `32768.0` on read and `32767`
/// on write — both conventions are common and mlx-audio inherits scipy's
/// split; we follow `read=32768.0` / `write=32767` to match
/// `mlx_audio.audio_io.read` and `.write` exactly).
///
/// On `write`, the C7 SIMD quantizer in
/// [`crate::simd::audio::quantize`] carries the matching `32767`
/// multiplier; that constant lives in the SIMD module so the NEON
/// kernel and scalar reference share one source of truth.
const I16_DIV: f32 = 32768.0;

/// Public-input-driven Vec allocation cap. Hard ceiling on the number of
/// f32 samples [`load_audio`] / [`resample_linear`] will materialize from
/// caller-controllable parameters. 64 Mi-samples ≈ 256 MiB at 4 B / f32
/// — about 30 min of mono 44.1 kHz audio — well above any realistic
/// "load entire file into memory" use case, and well below "abort the
/// host" territory. Crafted-attacker / fuzzer inputs above the cap get
/// a recoverable [`Error::Backend`] instead of a Rust allocator abort.
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
/// **Multi-channel files are rejected** with [`Error::Backend`] — the
/// reference `mlx_audio.audio_io.read` returns a 2-D `(samples, channels)`
/// array, but the `Vec<f32>` return shape here cannot faithfully carry that
/// information. Callers needing stereo / 5.1 / etc. must either preprocess
/// (downmix or split channels) before calling this fn, or use the planned
/// multi-channel follow-up.
///
/// # Errors
/// - [`Error::Backend`] if the file cannot be opened, the format cannot be
///   probed / is unsupported (e.g. an Opus or M4A file — scoped out, see
///   the module doc), the input has `channels != 1`, the codec is
///   unsupported, the declared frame count exceeds the
///   [`MAX_DECODED_SAMPLES`] cap, decoding produces more than the cap, an
///   uncompressed WAV's decoded sample count disagrees with its header,
///   or any sample is non-finite (NaN/inf in the decoded PCM).
pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
  // Open + wrap in symphonia's MediaSourceStream. Box<File> is required by
  // the MediaSource trait; the allocation is one-per-load_audio (fine —
  // file IO dominates the cost).
  let file = File::open(path).map_err(|e| Error::Backend {
    message: format!("load_audio: open {} failed: {e}", path.display()),
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
    .map_err(|e| Error::Backend {
      message: format!(
        "load_audio: probe {} failed: {e} (unsupported or corrupt format; \
         WAV/MP3/FLAC/OGG-Vorbis are supported, M4A/AAC/Opus/WebM are not)",
        path.display()
      ),
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

  let track = format
    .default_track(TrackType::Audio)
    .ok_or_else(|| Error::Backend {
      message: format!("load_audio: {} has no audio track", path.display()),
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
  // both the hard cap (`cap` below) and the strict post-decode equality
  // cross-check.
  let exact_count = is_wav || (is_flac && track_num_frames.is_some());

  let codec_params = track.codec_params.as_ref().ok_or_else(|| Error::Backend {
    message: format!("load_audio: {} has no codec parameters", path.display()),
  })?;
  let audio_params = codec_params.audio().ok_or_else(|| Error::Backend {
    message: format!("load_audio: {} default track is not audio", path.display()),
  })?;

  // Channel count: reject non-mono UPFRONT before any decode work. Using
  // the codec_params channel count avoids waiting until the first packet
  // is decoded to discover stereo.
  let nchannels = audio_params
    .channels
    .as_ref()
    .map(|c| c.count())
    .ok_or_else(|| Error::Backend {
      message: format!("load_audio: {} has no channel layout", path.display()),
    })?;
  if nchannels == 0 {
    return Err(Error::Backend {
      message: format!("load_audio: {} reports 0 channels", path.display()),
    });
  }
  if nchannels != 1 {
    return Err(Error::Backend {
      message: format!(
        "load_audio: multi-channel input not supported (got {nchannels} channels); \
         this API returns mono Vec<f32>. Downmix or split channels before calling."
      ),
    });
  }

  let sample_rate = audio_params.sample_rate.ok_or_else(|| Error::Backend {
    message: format!("load_audio: {} has no sample_rate", path.display()),
  })?;

  let mut decoder = symphonia::default::get_codecs()
    .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
    .map_err(|e| Error::Backend {
      message: format!("load_audio: make_audio_decoder failed: {e}"),
    })?;

  // Capped allocation. `num_frames` is the container-declared per-channel
  // frame count; for mono that equals the total f32 output length.
  // `Vec::with_capacity` is infallible (aborts on allocator OOM), so we
  // fall back to `try_reserve_exact` against `MAX_DECODED_SAMPLES`.
  //
  // - If the container declares `num_frames`, reject upfront if it
  //   exceeds `MAX_DECODED_SAMPLES` — that rejection is format-agnostic
  //   (a 10-GB-claiming MP3 header is refused exactly like a 10-GB WAV
  //   header), since it bounds the *reservation* and a wildly oversized
  //   declared length is never a valid file.
  // - WAV: `num_frames` is sample-EXACT, so pre-reserve exactly that
  //   much (one allocation per call; matches the prior hound-based path).
  // - MP3 / FLAC / OGG-Vorbis: `num_frames` is an ESTIMATE (MP3
  //   Xing/Info header) or subject to decoder delay/granule rounding.
  //   It is used only as a capacity HINT — pre-reserve it (clamped to
  //   the cap) to avoid reallocation churn, but the *hard* per-push
  //   ceiling below is `MAX_DECODED_SAMPLES`, not the header value, so
  //   a slight under-estimate does not spuriously fail a valid decode.
  // - If the container omits `num_frames`, reserve 0 and grow lazily;
  //   the per-push cap (`MAX_DECODED_SAMPLES` checked in `push_one`
  //   below) bounds the allocation so an unbounded stream cannot exceed
  //   the cap.
  let header_len_opt = track_num_frames.and_then(|n| usize::try_from(n).ok());
  if let Some(header_len) = header_len_opt
    && header_len > MAX_DECODED_SAMPLES
  {
    return Err(Error::Backend {
      message: format!(
        "load_audio: container declares {header_len} samples (>{MAX_DECODED_SAMPLES} cap); \
         refuse to allocate. Crafted/oversized inputs require a streaming \
         decoder API (planned follow-up)."
      ),
    });
  }
  // For WAV the header count is the exact length; for compressed formats
  // it is a hint clamped to the cap (it was already rejected above if it
  // exceeded the cap, so this clamp is a belt-and-braces no-op there).
  let reserve_len = header_len_opt.unwrap_or(0).min(MAX_DECODED_SAMPLES);
  let mut out: Vec<f32> = Vec::new();
  out
    .try_reserve_exact(reserve_len)
    .map_err(|e| Error::Backend {
      message: format!("load_audio: reservation for {reserve_len} samples failed: {e}"),
    })?;

  // `cap` is the hard upper bound on `out.len()` enforced per decoded
  // buffer by `push_samples`.
  // - Exact-count formats (WAV, FLAC-with-STREAMINFO-total): the header
  //   `num_frames` is sample-exact, so the cap is that count —
  //   over-running it means a malformed/corrupt file.
  // - Lossy MP3 / OGG-Vorbis (and a FLAC without a declared total): the
  //   header count (if any) is only an estimate, so the cap is the
  //   global `MAX_DECODED_SAMPLES` ceiling; using the estimate as a hard
  //   cap would spuriously fail a valid file whose true length slightly
  //   exceeds an under-estimating Xing/Info header.
  // Reaching `cap` makes `push_samples` return `Error::Backend` rather
  // than re-grow into the infallible-alloc path.
  let cap = if exact_count {
    header_len_opt.unwrap_or(MAX_DECODED_SAMPLES)
  } else {
    MAX_DECODED_SAMPLES
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
        return Err(Error::Backend {
          message: format!(
            "load_audio: {} is a chained/multi-segment stream (ResetRequired); \
             only single-stream audio is supported",
            path.display()
          ),
        });
      }
      Err(e) => {
        return Err(Error::Backend {
          message: format!("load_audio: next_packet failed: {e}"),
        });
      }
    };
    if packet.track_id != track_id {
      continue;
    }
    // Fail-loud on ANY decode error — a silent "skip the bad packet"
    // path would let a truncated or malformed file come back as `Ok`
    // with missing audio, exactly the silent-corruption surface the
    // load_audio contract excludes.
    let audio_buf = decoder.decode(&packet).map_err(|e| Error::Backend {
      message: format!("load_audio: decode failed: {e}"),
    })?;

    // Push interleaved f32 samples into `out`. We match on the typed
    // GenericAudioBufferRef variant + apply our own `/2^(bits-1)`
    // divisor for integer-PCM variants (symphonia's built-in
    // `Sample::to_sample::<f32>` for i16 divides by `i16::MAX = 32767`,
    // not `32768.0` — a 1-LSB drift from `mlx_audio.audio_io.read`'s
    // `int16 / 32768.0` convention that we avoid by going through the
    // raw integer samples). MP3 / Vorbis decoders hand us an already-
    // decoded f32 buffer; that arm is pass-through.
    push_samples(&audio_buf, &mut out, cap)?;
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
    return Err(Error::Backend {
      message: format!(
        "load_audio: decoded {} samples but container header declared {header_len} \
         (truncated or malformed exact-count file: WAV or FLAC)",
        out.len()
      ),
    });
  }

  Ok((out, sample_rate))
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
///    buffer through this guard). Returns [`Error::Backend`] (the existing
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
    return Err(Error::Backend {
      message: format!("load_audio: stream produced more than the {cap}-sample cap"),
    });
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
      Err(Error::Backend {
        message: "load_audio: non-finite f32 PCM sample".into(),
      })
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
        return Err(Error::Backend {
          message: format!("load_audio: int_divisor: unexpected bits={n} (programmer error)"),
        });
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
      // C1 SIMD widen — collect the symphonia interleaved iterator
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
      // C1 SIMD widen — collect the symphonia i24 iterator
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
      // C1 SIMD widen — collect symphonia's i32 iterator into a
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

/// Write `samples` to `path` as a 16-bit mono WAV at `sample_rate`.
///
/// Samples outside `[-1.0, 1.0]` are clipped (matches `mlx_audio.audio_io.write`'s
/// `np.clip(data, -1.0, 1.0)` pre-quantization), then multiplied by `32767`
/// and converted to `i16`.
///
/// **All samples are validated finite UPFRONT**, before any tempfile is
/// opened, so a non-finite sample never leaves a partially-written WAV
/// on disk. This is stricter than `mlx_audio.audio_io.write` (which
/// would silently corrupt the WAV by casting `NaN → i16` to 0).
///
/// **Mid-write IO failure no longer leaves a partial WAV** — the
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
/// return [`Error::Backend`] with the destination untouched — a
/// partial WAV cannot be observed at `path`. Note: the tempfile lives
/// in the same directory as `path` so the rename is single-fs
/// (cross-fs rename would silently fall back to copy+unlink and lose
/// the atomicity guarantee). On POSIX `set_permissions` only restores
/// the mode bits, NOT ownership/ACLs/xattrs — preserving those
/// requires platform-specific code and is a planned follow-up.
///
/// # Errors
/// - [`Error::Backend`] if any sample is non-finite (NaN/inf), `sample_rate`
///   is 0 or exceeds the byte-rate u32 ceiling (`u32::MAX / 2`),
///   `samples.len()` exceeds the 16-bit-WAV total-file-size limit
///   (`(u32::MAX - 36) / 2`), the destination directory has no
///   `file_name` component, all tempfile retries (16) collide on
///   `AlreadyExists`, or the tempfile cannot be created/written/
///   flushed/renamed. On UPFRONT validation failure (NaN/inf, zero or
///   oversized sample-rate, oversized buffer) the destination is
///   untouched; on mid-write failure the destination is still
///   untouched (tempfile path is removed, original `path` contents
///   — if any — remain).
pub fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
  if sample_rate == 0 {
    return Err(Error::Backend {
      message: "save_wav: sample_rate must be > 0".into(),
    });
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
    return Err(Error::Backend {
      message: format!(
        "save_wav: sample count {} exceeds the 16-bit WAV total-file-size limit \
         ({MAX_MONO_I16_SAMPLES} samples = (u32::MAX - 36) bytes / \
         2-bytes-per-sample); split into multiple files or use a large-WAV variant",
        samples.len()
      ),
    });
  }
  // byte_rate = sample_rate * channels * bytes_per_sample (= sample_rate * 2
  // for mono i16). Must fit in u32 — reject sample_rate values whose
  // byte_rate would wrap. `u32::MAX / 2 = 2147483647` (~2.1 GHz, well
  // above any real audio sample rate).
  const MAX_SAMPLE_RATE_FOR_MONO_I16: u32 = u32::MAX / 2;
  if sample_rate > MAX_SAMPLE_RATE_FOR_MONO_I16 {
    return Err(Error::Backend {
      message: format!(
        "save_wav: sample_rate {sample_rate} exceeds the {MAX_SAMPLE_RATE_FOR_MONO_I16} \
         ceiling at which byte_rate (sample_rate * 2) fits in u32"
      ),
    });
  }
  // Pre-validate ALL samples before any filesystem mutation, so a NaN/inf
  // in the buffer cannot leave a partially-written WAV on disk. This
  // departs from `mlx_audio.audio_io.write` (which would silently cast
  // NaN to 0 via `astype(int16)`), but the cost is one extra scan and
  // the gain is "destination integrity is preserved on input error".
  for (i, &s) in samples.iter().enumerate() {
    if !s.is_finite() {
      return Err(Error::Backend {
        message: format!("save_wav: non-finite sample at index {i} (NaN/inf) — cannot quantize"),
      });
    }
  }

  // Capture the existing destination's permissions (if it exists) so
  // the post-rename file keeps the user's chosen mode/ACL — otherwise
  // a private 0600 audio file silently widens to whatever the process
  // umask grants on the fresh tempfile inode. `None` means the
  // destination doesn't exist yet; in that case the tempfile keeps
  // its umask-granted mode.
  let existing_perms = fs::metadata(path).ok().map(|m| m.permissions());

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

  // Inner write closure: returns Result so we can clean up the tempfile
  // (best-effort) on any failure before the rename.
  let write_result = (|| -> Result<()> {
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
    writer.write_all(&header).map_err(|e| Error::Backend {
      message: format!("save_wav: header write failed: {e}"),
    })?;

    // Quantize all samples first via the C7 SIMD dispatcher
    // (`simd::audio::quantize::f32_to_i16_quantize`) — clip to `[-1, 1]`,
    // multiply by `I16_MUL = 32767`, round-half-away-from-zero, cast to
    // i16. On `aarch64` this routes to an 8-lane NEON tile (vminq/vmaxq
    // clamp → vmulq_n scale → vcvtaq_s32 round (FCVTAS, ties away from
    // zero — bit-exact match for `f32::round`) → vqmovn+vcombine narrow
    // → vst1q_s16 store); elsewhere it falls back to the scalar
    // `clamp + round + cast` loop. See `docs/core-arch-simd-candidates.md`
    // §2 row C7 + §3.5 + tracking [#152].
    //
    // The dispatcher takes `&mut [MaybeUninit<i16>]` (type-encoded
    // uninit safety), so we pre-reserve via `try_reserve_exact` (so a
    // multi-GB sample buffer cannot trigger an infallible abort here),
    // pass the spare capacity directly, and `set_len` after every i16
    // has been written. Then a SINGLE `write_all` writes the entire
    // i16 byte view in one syscall — replacing the per-sample
    // BufWriter pushes.
    let mut quantized: Vec<i16> = Vec::new();
    quantized
      .try_reserve_exact(samples.len())
      .map_err(|_| Error::OutOfMemory)?;
    {
      let spare: &mut [core::mem::MaybeUninit<i16>] = quantized.spare_capacity_mut();
      // `samples.len() <= spare.len()` because `try_reserve_exact(samples.len())`
      // above reserved exactly that much capacity.
      debug_assert!(spare.len() >= samples.len());
      crate::simd::audio::quantize::f32_to_i16_quantize(&mut spare[..samples.len()], samples);
    }
    // SAFETY: `f32_to_i16_quantize` wrote every i16 in `0..samples.len()`
    // of the spare capacity (function-level contract). `Vec::set_len`'s
    // preconditions: (1) `samples.len() <= quantized.capacity()` — the
    // `try_reserve_exact` succeeded; (2) elements at `[0..samples.len()]`
    // are initialized — kernel contract guarantees this.
    unsafe { quantized.set_len(samples.len()) };

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
      writer.write_all(byte_view).map_err(|e| Error::Backend {
        message: format!("save_wav: bulk sample write failed: {e}"),
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
        writer
          .write_all(&buf[..n * 2])
          .map_err(|e| Error::Backend {
            message: format!("save_wav: bulk sample write failed: {e}"),
          })?;
        idx += n;
      }
    }

    // BufWriter does NOT auto-flush on drop into a Result, and a missed
    // flush would leave us renaming an incomplete tempfile into place.
    writer.flush().map_err(|e| Error::Backend {
      message: format!("save_wav: flush failed: {e}"),
    })?;
    // `flush()` only drains the in-process buffer to the OS; on delayed-
    // allocation filesystems, NFS, quotas, etc. a writeback / late-ENOSPC
    // failure would otherwise be observed only on close (whose result Drop
    // discards) — meaning we could rename in a tempfile whose contents
    // never actually hit the disk. `sync_all` (fsync) forces the data +
    // metadata to durable storage, and we propagate its error so we
    // never rename a not-yet-durable tempfile into place.
    let inner = writer.into_inner().map_err(|e| Error::Backend {
      message: format!("save_wav: BufWriter::into_inner failed: {e}"),
    })?;
    inner.sync_all().map_err(|e| Error::Backend {
      message: format!("save_wav: sync_all failed: {e}"),
    })?;
    // Close the inner File before rename — Windows in particular
    // dislikes renaming an open file handle.
    drop(inner);
    Ok(())
  })();

  if let Err(err) = write_result {
    // Best-effort tempfile cleanup. Don't fail the call on cleanup
    // failure — the original `err` is what the caller needs to see.
    let _ = fs::remove_file(&tmp_path);
    return Err(err);
  }

  // Restore the destination's prior permissions BEFORE the rename so
  // the post-rename file matches the user's pre-existing mode/ACL.
  // Skipped when the destination didn't previously exist (the
  // tempfile's umask-granted mode is the natural default for new files).
  // Failure here is treated like any other write-path failure: clean up
  // the tempfile and propagate. Note: on POSIX `set_permissions` only
  // sets the mode bits, NOT ownership/ACLs/xattrs — preserving those
  // requires platform-specific code and is a planned follow-up.
  if let Some(perms) = existing_perms
    && let Err(e) = fs::set_permissions(&tmp_path, perms)
  {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::Backend {
      message: format!(
        "save_wav: set_permissions on tempfile {} failed: {e}",
        tmp_path.display()
      ),
    });
  }

  // Atomic-within-fs rename. POSIX `rename(2)` and Windows `MoveFileEx`
  // both make the destination point at the new bytes atomically — no
  // observer can see a half-written WAV at `path`.
  if let Err(e) = fs::rename(&tmp_path, path) {
    let _ = fs::remove_file(&tmp_path);
    return Err(Error::Backend {
      message: format!(
        "save_wav: rename {} -> {} failed: {e}",
        tmp_path.display(),
        path.display()
      ),
    });
  }
  Ok(())
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
    .ok_or_else(|| Error::Backend {
      message: format!(
        "save_wav: destination {} has no file_name component",
        final_path.display()
      ),
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
        return Err(Error::Backend {
          message: format!("save_wav: create_new {} failed: {e}", candidate.display()),
        });
      }
    }
  }
  Err(Error::Backend {
    message: format!(
      "save_wav: exhausted {max_retries} tempfile retries (last error: {})",
      last_err
        .map(|e| e.to_string())
        .unwrap_or_else(|| "<none>".into())
    ),
  })
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
/// floating-point rounding drift).
///
/// # Errors
/// - [`Error::Backend`] if `from_rate == 0` (would divide by zero),
///   `to_rate == 0`, the computed output length overflows `usize`,
///   exceeds the [`MAX_RESAMPLED_SAMPLES`] cap (64 Mi-samples), or the
///   recoverable Vec reservation fails.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
  if from_rate == 0 {
    return Err(Error::Backend {
      message: "resample_linear: from_rate must be > 0".into(),
    });
  }
  if to_rate == 0 {
    return Err(Error::Backend {
      message: "resample_linear: to_rate must be > 0".into(),
    });
  }
  if samples.is_empty() {
    return Ok(Vec::new());
  }
  if from_rate == to_rate {
    // Verbatim copy — avoids any FP rounding drift on a no-op resample.
    return Ok(samples.to_vec());
  }

  // Output length: `samples.len() * to_rate / from_rate`. Use u64 to avoid
  // overflow in the intermediate product; then check the final fits in
  // usize so we don't silently truncate on 32-bit targets.
  let in_len = samples.len() as u64;
  let out_len_u64 = in_len
    .checked_mul(u64::from(to_rate))
    .ok_or_else(|| Error::Backend {
      message: format!(
        "resample_linear: in_len * to_rate overflows u64 (in_len={in_len}, to_rate={to_rate})"
      ),
    })?
    / u64::from(from_rate);
  let out_len = usize::try_from(out_len_u64).map_err(|_| Error::Backend {
    message: format!("resample_linear: output length {out_len_u64} exceeds usize::MAX"),
  })?;
  if out_len == 0 {
    return Ok(Vec::new());
  }

  // Hard cap on the output buffer — defends against `from_rate=1,
  // to_rate=u32::MAX` (or similar adversarial ratios) that would attempt
  // a tens-of-GB allocation. The cap matches `load_audio`'s
  // `MAX_DECODED_SAMPLES` (64 Mi-samples ≈ 256 MiB of f32).
  if out_len > MAX_RESAMPLED_SAMPLES {
    return Err(Error::Backend {
      message: format!(
        "resample_linear: output length {out_len} exceeds the {MAX_RESAMPLED_SAMPLES} cap; \
         use chunked resampling or raise the cap manually"
      ),
    });
  }

  let mut out: Vec<f32> = Vec::new();
  out.try_reserve_exact(out_len).map_err(|e| Error::Backend {
    message: format!("resample_linear: reservation for {out_len} samples failed: {e}"),
  })?;
  let ratio = f64::from(from_rate) / f64::from(to_rate);

  // C8 SIMD: dispatch to the resample-linear NEON kernel
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
  // SAFETY: the C8 dispatcher's init contract guarantees every f32 of
  // the `out_len`-prefix of `spare` is initialized before returning;
  // `out_len <= out.capacity()` per the `try_reserve_exact(out_len)`
  // above.
  unsafe { out.set_len(out_len) };
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Fix 1 (bounded compressed decode): `reserve_under_cap` is the single
  /// gate every decoded buffer passes through before its samples are
  /// appended. A buffer that would push `out` past `cap` — the exact
  /// scenario a compressed header under-estimating the cap creates (header
  /// reserves `cap`, the decoder yields one more valid sample) — must
  /// return a recoverable `Error::Backend`, and must NOT have grown `out`
  /// past `cap` (no infallible `Vec` regrowth, no allocator abort).
  #[test]
  fn reserve_under_cap_rejects_buffer_that_would_exceed_cap() {
    // Small synthetic cap so the test allocates nothing large. `out` is
    // pre-filled to exactly one below the cap, mirroring "header reserved
    // up to `cap`, decoder produced one more sample than fits".
    let cap = 8usize;
    let mut out: Vec<f32> = vec![0.0; cap - 1];
    let cap_before = out.capacity();

    // A single extra sample exactly fills the cap — allowed.
    assert!(reserve_under_cap(&mut out, 1, cap).is_ok());
    // (We do not push here; we only assert the reservation/cap math.)

    // From the now-full-capacity position, any further buffer (even one
    // sample) must be rejected, NOT pushed through an infallible regrowth.
    out.resize(cap, 0.0); // out.len() == cap now (the "decoded up to cap" state)
    let r = reserve_under_cap(&mut out, 1, cap);
    assert!(
      matches!(r, Err(Error::Backend { .. })),
      "over-cap buffer must return a recoverable Backend error, got {r:?}"
    );
    // The Vec must not have been grown past the cap by the rejected call.
    assert!(
      out.len() <= cap,
      "out grew past cap on the rejected path: len={} cap={cap}",
      out.len()
    );
    // Capacity sanity: the rejection happened before any reallocation that
    // would push capacity wildly above the cap (it may be >= cap from the
    // earlier successful reserve, but the reject path added nothing).
    assert!(
      out.capacity() >= cap_before,
      "capacity unexpectedly shrank: {} < {cap_before}",
      out.capacity()
    );
  }

  /// A buffer larger than the entire remaining room (multi-sample
  /// over-cap, the realistic decoded-packet case) is rejected up front
  /// with no growth.
  #[test]
  fn reserve_under_cap_rejects_oversized_buffer_against_empty_out() {
    let cap = 16usize;
    let mut out: Vec<f32> = Vec::new();
    // A packet claiming more samples than the whole cap.
    let r = reserve_under_cap(&mut out, cap + 1, cap);
    assert!(
      matches!(r, Err(Error::Backend { .. })),
      "buffer larger than the cap must be rejected, got {r:?}"
    );
    assert_eq!(
      out.len(),
      0,
      "rejected reservation must not append anything"
    );
    assert!(
      out.capacity() <= cap,
      "rejected reservation must not allocate past the cap: capacity={}",
      out.capacity()
    );
  }

  /// An exactly-fitting buffer is accepted and reserves the room (so the
  /// caller's subsequent pushes cannot regrow).
  #[test]
  fn reserve_under_cap_accepts_and_reserves_up_to_cap() {
    let cap = 32usize;
    let mut out: Vec<f32> = Vec::new();
    reserve_under_cap(&mut out, cap, cap).expect("filling exactly to cap must succeed");
    assert!(
      out.capacity() >= cap,
      "reservation did not provide cap room: capacity={} cap={cap}",
      out.capacity()
    );
    // Pushing the reserved `cap` samples cannot reallocate (capacity was
    // reserved), so this loop never hits the infallible-growth path.
    for i in 0..cap {
      out.push(i as f32);
    }
    assert_eq!(out.len(), cap);
  }

  /// Codex review (cap-limited reserve growth): a plain amortized
  /// `Vec::try_reserve(n)` can grow capacity to ~2× the *current* capacity
  /// even when only a few samples remain under `cap`. With a near-cap
  /// header hint (capacity already reserved up to `cap` by
  /// `try_reserve_exact` in `load_audio`) plus a final in-cap packet, that
  /// doubling would attempt an allocation FAR larger than the cap —
  /// defeating the [`MAX_DECODED_SAMPLES`] memory ceiling (and, under
  /// memory pressure, spuriously failing an in-cap decode because the
  /// oversized reserve fails). The cap-limited growth must (a) accept the
  /// in-cap packet and (b) NOT grow capacity past `cap`.
  #[test]
  fn reserve_under_cap_growth_does_not_exceed_cap() {
    let cap = 64usize;
    // `out` already holds capacity == cap (the near-cap header-hint state)
    // and is filled to one below the cap, so a 1-sample packet still fits
    // under the cap but the *needed* capacity (cap) equals current
    // capacity — the case where a plain `try_reserve` would otherwise
    // double to ~2*cap.
    let mut out: Vec<f32> = Vec::with_capacity(cap);
    out.resize(cap - 1, 0.0);
    assert_eq!(out.capacity(), cap, "precondition: capacity == cap");

    // A final packet that still fits under the cap is accepted.
    reserve_under_cap(&mut out, 1, cap).expect("in-cap final packet must be accepted");

    // The reservation must NOT have doubled capacity past the cap.
    assert!(
      out.capacity() <= cap,
      "reserve grew capacity past the cap: capacity={} cap={cap}",
      out.capacity()
    );

    // Also exercise the headroom case: capacity == cap - packet_len with
    // plenty of slack below the cap. Growth is still clamped at the cap.
    let packet = 8usize;
    let mut out2: Vec<f32> = Vec::with_capacity(cap - packet);
    out2.resize(cap - packet, 0.0);
    let cap2_before = out2.capacity();
    reserve_under_cap(&mut out2, packet, cap).expect("packet filling exactly to cap must succeed");
    assert!(
      out2.capacity() <= cap,
      "reserve grew capacity past the cap: capacity={} cap={cap}",
      out2.capacity()
    );
    assert!(
      out2.capacity() >= cap2_before + packet,
      "reserve did not provide room for the packet: capacity={} need>={}",
      out2.capacity(),
      cap2_before + packet
    );
    // The reserved room is real: pushing the packet cannot reallocate.
    for i in 0..packet {
      out2.push(i as f32);
    }
    assert_eq!(out2.len(), cap);
  }
}
