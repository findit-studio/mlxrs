//! WAV file IO + naive linear resampling.
//!
//! Faithful 1:1 port of the [`mlx_audio.audio_io.read`] / [`mlx_audio.audio_io.write`]
//! core (WAV-only — MP3/FLAC/OGG/Opus opt-in via additional `symphonia`
//! features in future PRs). Resampling is a naive linear-interpolation
//! pass mirroring the `mlx-audio` utilities used by Whisper-style models;
//! high-quality polyphase / sinc resampling is a planned follow-up.
//!
//! WAV *decoding* goes through the pure-Rust [`symphonia`] crate (active
//! multi-format library, `wav` + `pcm` features only for now). WAV
//! *encoding* is roll-our-own pure-Rust 16-bit PCM mono — symphonia
//! exposes no encoder API and the entire RIFF/WAVE/fmt/data spec fits in
//! ~80 LOC, letting us control eval discipline and add atomic-rename
//! without a crate-side change. This first cut is **mono-only**:
//! multi-channel input is rejected with [`Error::Backend`] (the
//! `Vec<f32>` return shape cannot faithfully carry the reference's 2-D
//! `(samples, channels)` layout). A `load_wav_multichannel` variant
//! returning `(Vec<f32>, u32, u16)` is a planned follow-up. Output is
//! always normalized to `f32` in `[-1.0, 1.0]`.
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
  formats::{FormatOptions, TrackType, probe::Hint},
  io::MediaSourceStream,
  meta::MetadataOptions,
};

use crate::error::{Error, Result};

/// PCM-16 full-scale; matches `mlx-audio`'s `int16 → float32` divisor on
/// `read` and `float32 → int16` multiplier on `write` (the reference uses
/// `32768.0` on read and `32767` on write — both conventions are common and
/// mlx-audio inherits scipy's split; we follow `read=32768.0` /
/// `write=32767` to match `mlx_audio.audio_io.read` and `.write` exactly).
const I16_DIV: f32 = 32768.0;
const I16_MUL: f32 = 32767.0;

/// Public-input-driven Vec allocation cap. Hard ceiling on the number of
/// f32 samples [`load_wav`] / [`resample_linear`] will materialize from
/// caller-controllable parameters. 64 Mi-samples ≈ 256 MiB at 4 B / f32
/// — about 30 min of mono 44.1 kHz audio — well above any realistic
/// "load entire file into memory" use case, and well below "abort the
/// host" territory. Crafted-attacker / fuzzer inputs above the cap get
/// a recoverable [`Error::Backend`] instead of a Rust allocator abort.
pub const MAX_DECODED_SAMPLES: usize = 64 * 1024 * 1024;
/// See [`MAX_DECODED_SAMPLES`] — same cap, applied to [`resample_linear`].
pub const MAX_RESAMPLED_SAMPLES: usize = MAX_DECODED_SAMPLES;

/// Load a mono WAV file and return `(samples, sample_rate)`.
///
/// Returns `f32` samples in `[-1.0, 1.0]`. Integer-PCM inputs are divided
/// by `2^(bits-1)` (16-bit → 32768.0, matching `mlx_audio.audio_io.read`'s
/// `dtype="float32"` arm exactly). Float WAVs are passed through unchanged.
///
/// **Multi-channel WAVs are rejected** with [`Error::Backend`] — the
/// reference `mlx_audio.audio_io.read` returns a 2-D `(samples, channels)`
/// array, but the `Vec<f32>` return shape here cannot faithfully carry that
/// information. Callers needing stereo / 5.1 / etc. must either preprocess
/// (downmix or split channels) before calling this fn, or use the planned
/// `load_wav_multichannel` follow-up.
///
/// # Errors
/// - [`Error::Backend`] if the file cannot be opened, the format is invalid,
///   the input has `channels != 1`, the codec is unsupported, or any
///   sample is non-finite (NaN/inf in the source f32 PCM).
pub fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
  // Open + wrap in symphonia's MediaSourceStream. Box<File> is required by
  // the MediaSource trait; the allocation is one-per-load_wav (fine — file
  // IO dominates the cost).
  let file = File::open(path).map_err(|e| Error::Backend {
    message: format!("load_wav: open {} failed: {e}", path.display()),
  })?;
  let mss = MediaSourceStream::new(Box::new(file), Default::default());

  // Hint with the file extension if present — lets the probe skip
  // format-detection backtracking for known extensions.
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
      message: format!("load_wav: probe {} failed: {e}", path.display()),
    })?;

  let track = format
    .default_track(TrackType::Audio)
    .ok_or_else(|| Error::Backend {
      message: format!("load_wav: {} has no audio track", path.display()),
    })?;
  let track_id = track.id;
  // `Track::num_frames` is the per-channel frame count from the
  // container header (`None` if the container didn't declare one).
  // Captured BEFORE we borrow `codec_params` because we'll move out
  // of the immutable `track` borrow below.
  let track_num_frames = track.num_frames;

  let codec_params = track.codec_params.as_ref().ok_or_else(|| Error::Backend {
    message: format!("load_wav: {} has no codec parameters", path.display()),
  })?;
  let audio_params = codec_params.audio().ok_or_else(|| Error::Backend {
    message: format!("load_wav: {} default track is not audio", path.display()),
  })?;

  // Channel count: reject non-mono UPFRONT before any decode work. Using
  // the codec_params channel count avoids waiting until the first packet
  // is decoded to discover stereo.
  let nchannels = audio_params
    .channels
    .as_ref()
    .map(|c| c.count())
    .ok_or_else(|| Error::Backend {
      message: format!("load_wav: {} has no channel layout", path.display()),
    })?;
  if nchannels == 0 {
    return Err(Error::Backend {
      message: "load_wav: WAV header reports 0 channels".into(),
    });
  }
  if nchannels != 1 {
    return Err(Error::Backend {
      message: format!(
        "load_wav: multi-channel input not supported (got {nchannels} channels); \
         this API returns mono Vec<f32>. Downmix or split channels before calling."
      ),
    });
  }

  let sample_rate = audio_params.sample_rate.ok_or_else(|| Error::Backend {
    message: format!("load_wav: {} has no sample_rate", path.display()),
  })?;

  let mut decoder = symphonia::default::get_codecs()
    .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
    .map_err(|e| Error::Backend {
      message: format!("load_wav: make_audio_decoder failed: {e}"),
    })?;

  // Capped allocation. `num_frames` is the WAV-header-declared sample
  // count per channel; for mono that equals the total f32 output length.
  // `Vec::with_capacity` is infallible (aborts on allocator OOM), so we
  // fall back to `try_reserve_exact` against `MAX_DECODED_SAMPLES`.
  //
  // - If the container declares `num_frames`, reject upfront if it
  //   exceeds `MAX_DECODED_SAMPLES` and pre-reserve exactly that much
  //   (matches the prior hound-based path: one allocation per call,
  //   rejection before any decode work).
  // - If the container omits `num_frames`, reserve 0 and grow lazily.
  //   In that mode the per-push cap (`MAX_DECODED_SAMPLES` checked in
  //   `push_one` below) is what bounds the allocation; an unbounded
  //   stream therefore cannot exceed the cap.
  let header_len_opt = track_num_frames.and_then(|n| usize::try_from(n).ok());
  if let Some(header_len) = header_len_opt
    && header_len > MAX_DECODED_SAMPLES
  {
    return Err(Error::Backend {
      message: format!(
        "load_wav: header declares {header_len} samples (>{MAX_DECODED_SAMPLES} cap); \
         refuse to allocate. Crafted/oversized WAV inputs require a streaming \
         decoder API (planned follow-up)."
      ),
    });
  }
  let reserve_len = header_len_opt.unwrap_or(0);
  let mut out: Vec<f32> = Vec::new();
  out
    .try_reserve_exact(reserve_len)
    .map_err(|e| Error::Backend {
      message: format!("load_wav: reservation for {reserve_len} samples failed: {e}"),
    })?;

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
        // clean end-of-stream so a properly-terminated WAV decodes
        // identically regardless of which path the reader takes.
        break;
      }
      Err(e) => {
        return Err(Error::Backend {
          message: format!("load_wav: next_packet failed: {e}"),
        });
      }
    };
    if packet.track_id != track_id {
      continue;
    }
    // Fail-loud on ANY decode error — the prior hound-based path
    // returned `Error::Backend` for every per-sample decode failure,
    // and a silent "skip the bad packet" path would let a truncated
    // or malformed WAV come back as `Ok` with missing audio. That
    // mismatch with the strict-integer-PCM contract is exactly the
    // silent-corruption surface the load_wav contract excludes.
    let audio_buf = decoder.decode(&packet).map_err(|e| Error::Backend {
      message: format!("load_wav: decode failed: {e}"),
    })?;

    // Push interleaved f32 samples into `out`. We match on the typed
    // GenericAudioBufferRef variant + apply our own `/2^(bits-1)`
    // divisor so the f32 values are byte-identical to the prior
    // hound-based path (symphonia's built-in `Sample::to_sample::<f32>`
    // for i16 divides by `i16::MAX = 32767`, not `32768.0` — a 1-LSB
    // drift that the round-trip tolerance would absorb but that we
    // can avoid entirely by going through the raw integer samples).
    //
    // `cap` is the hard upper bound on `out.len()`: the
    // header-declared count if the container provided one, else the
    // global `MAX_DECODED_SAMPLES` ceiling. Reaching `cap` causes
    // `push_one` to return `Error::Backend` rather than re-grow into
    // the infallible-alloc path.
    let cap = header_len_opt.unwrap_or(MAX_DECODED_SAMPLES);
    push_samples(&audio_buf, &mut out, cap)?;
  }

  // Cross-check decoded sample count against the header-declared length
  // when the container provided one. A short read (truncated WAV) would
  // otherwise return `Ok` with missing audio — that's the same silent
  // corruption surface the per-packet fail-loud above closes, but at
  // the stream-level. We tolerate `out.len() > header_len` (over-declared
  // headers; that path was already cap-rejected above) only because the
  // `push_one` cap is exactly `header_len` when known, which prevents
  // this branch from ever being reached with `out.len() > header_len`.
  if let Some(header_len) = header_len_opt
    && out.len() != header_len
  {
    return Err(Error::Backend {
      message: format!(
        "load_wav: decoded {} samples but header declared {header_len} \
         (truncated or malformed WAV)",
        out.len()
      ),
    });
  }

  Ok((out, sample_rate))
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
/// Enforces a hard upper bound `cap` on the running `out` length so the
/// stream-vs-header / unbounded-stream paths are rejected before any
/// further allocation.
fn push_samples(buf: &GenericAudioBufferRef<'_>, out: &mut Vec<f32>, cap: usize) -> Result<()> {
  /// Reject a non-finite f32 (NaN/inf) before pushing — matches
  /// `load_wav`'s prior hound-based Float arm.
  fn check_finite(s: f32) -> Result<f32> {
    if s.is_finite() {
      Ok(s)
    } else {
      Err(Error::Backend {
        message: "load_wav: non-finite f32 PCM sample".into(),
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
          message: format!("load_wav: int_divisor: unexpected bits={n} (programmer error)"),
        });
      }
    })
  }
  /// Push a single sample after enforcing the cap. The cap is the
  /// hard ceiling — either the header-declared length (one-allocation
  /// pre-reserve path) or `MAX_DECODED_SAMPLES` (lazy-grow path).
  /// Exceeding the cap returns `Error::Backend` rather than letting
  /// the Vec re-grow into the infallible-alloc path.
  fn push_one(out: &mut Vec<f32>, sample: f32, cap: usize) -> Result<()> {
    if out.len() >= cap {
      return Err(Error::Backend {
        message: format!("load_wav: stream produced more than the {cap}-sample cap"),
      });
    }
    out.push(sample);
    Ok(())
  }

  // Float WAVs are pass-through (already `[-1, 1]` f32). Integer-PCM
  // WAVs are normalized by `2^(bits-1)` against the variant's intrinsic
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
  match buf {
    GenericAudioBufferRef::F32(b) => {
      for s in b.iter_interleaved() {
        push_one(out, check_finite(s)?, cap)?;
      }
    }
    GenericAudioBufferRef::F64(b) => {
      for s in b.iter_interleaved() {
        push_one(out, check_finite(s as f32)?, cap)?;
      }
    }
    GenericAudioBufferRef::U8(b) => {
      // Offset-binary unsigned 8-bit (range [0, 256), midpoint 128).
      // Recenter via `i16::from(s) - 128` to land in `[-128, 127]`,
      // then divide by `2^7 = 128.0`.
      let divisor = int_divisor(8)?;
      for s in b.iter_interleaved() {
        let signed = i16::from(s) - 128;
        push_one(out, f32::from(signed) / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::U16(b) => {
      // Offset-binary unsigned 16-bit (range [0, 65536), midpoint 32768).
      let divisor = int_divisor(16)?;
      for s in b.iter_interleaved() {
        let signed = i32::from(s) - 32_768;
        push_one(out, signed as f32 / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::U24(b) => {
      // Offset-binary unsigned 24-bit (range [0, 2^24), midpoint 2^23).
      // `u24.inner()` returns the `[0, 2^24)` value as `u32`.
      let divisor = int_divisor(24)?;
      for s in b.iter_interleaved() {
        let signed = s.inner() as i32 - 0x80_0000;
        push_one(out, signed as f32 / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::U32(b) => {
      // Offset-binary unsigned 32-bit (range [0, 2^32), midpoint 2^31).
      // Use `wrapping_sub` to stay in i32 wraparound semantics — the
      // result is the i32 reinterpretation of `(s - 2^31)`.
      let divisor = int_divisor(32)?;
      for s in b.iter_interleaved() {
        let signed = s.wrapping_sub(0x8000_0000) as i32;
        push_one(out, signed as f32 / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::S8(b) => {
      let divisor = int_divisor(8)?;
      for s in b.iter_interleaved() {
        push_one(out, f32::from(s) / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::S16(b) => {
      let divisor = int_divisor(16)?;
      for s in b.iter_interleaved() {
        push_one(out, f32::from(s) / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::S24(b) => {
      // `i24.inner()` returns the `[-2^23, 2^23)` value as `i32`.
      let divisor = int_divisor(24)?;
      for s in b.iter_interleaved() {
        push_one(out, s.inner() as f32 / divisor, cap)?;
      }
    }
    GenericAudioBufferRef::S32(b) => {
      let divisor = int_divisor(32)?;
      for s in b.iter_interleaved() {
        push_one(out, s as f32 / divisor, cap)?;
      }
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

    // Stream the samples. Quantization: clip to `[-1, 1]`, multiply by
    // `I16_MUL = 32767`, round to nearest, cast to i16, write LE.
    for &s in samples {
      let clipped = s.clamp(-1.0, 1.0);
      let q = (clipped * I16_MUL).round() as i16;
      writer
        .write_all(&q.to_le_bytes())
        .map_err(|e| Error::Backend {
          message: format!("save_wav: sample write failed: {e}"),
        })?;
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
  // a tens-of-GB allocation. The cap matches `load_wav`'s
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
  // `last_in` is the largest valid source index; saturating-sub guards the
  // `samples.len() == 1` case (we already returned Ok early for `is_empty`,
  // but a 1-element input still has `last_in == 0`, in which case all
  // interpolation degenerates to copying `samples[0]`).
  let last_in = samples.len() - 1;
  for i in 0..out_len {
    let x = i as f64 * ratio;
    let lo = x.floor();
    let frac = x - lo;
    // `lo as usize` is well-defined because `out_len = in_len * to_rate /
    // from_rate`, so `x_max = (out_len-1) * from_rate/to_rate ≤ in_len - 1`
    // (with strict inequality unless `(out_len-1)*from_rate` divides
    // `to_rate` exactly), keeping `lo` in `[0, in_len-1]`. We still saturate
    // to `last_in` defensively to absorb any FP rounding-up at the boundary.
    let lo_idx = (lo as usize).min(last_in);
    let hi_idx = (lo_idx + 1).min(last_in);
    let a = samples[lo_idx];
    let b = samples[hi_idx];
    out.push(a + (b - a) * frac as f32);
  }
  Ok(out)
}
