//! Happy-path + edge-case tests for `mlxrs::audio::io` (WAV/MP3/FLAC/
//! OGG-Vorbis load + WAV save + naive linear resample).
//!
//! NO disk I/O outside `std::env::temp_dir() + process::id()` (matches the
//! existing `tests/io.rs` convention; tempfile is not a workspace dep yet).
//!
//! Compressed-format tests decode tiny real fixtures committed under
//! `tests/fixtures/audio_tone.{mp3,flac,ogg}` — each a ~0.25 s 8 kHz mono
//! 440 Hz tone (<5 KB) generated once via ffmpeg / libsndfile. The bytes
//! are `include_bytes!`-embedded and written to a temp file per test, so
//! the suite needs no encoder dependency and no ffmpeg at test time —
//! only `mlxrs`'s symphonia *decoder*. Sample-count asserts use a
//! tolerance band (not an exact count) because lossy decoders differ in
//! how they handle encoder delay / padding.

#![cfg(feature = "audio")]

use std::{
  fs::{self, File},
  io::Write,
  path::PathBuf,
  process,
};

use mlxrs::audio::io::{load_audio, resample_linear, save_wav};

/// Tiny committed fixtures: ~0.25 s 8 kHz mono 440 Hz tone, one per
/// newly-enabled compressed format. See the module doc for provenance.
const FIXTURE_MP3: &[u8] = include_bytes!("fixtures/audio_tone.mp3");
const FIXTURE_FLAC: &[u8] = include_bytes!("fixtures/audio_tone.flac");
const FIXTURE_OGG_VORBIS: &[u8] = include_bytes!("fixtures/audio_tone.ogg");

/// Expected source duration of every fixture (0.25 s @ 8 kHz = 2000
/// frames). Decoders disagree by a few hundred samples on encoder
/// delay/padding, so callers assert a band around this, never equality.
const FIXTURE_SR: u32 = 8000;
const FIXTURE_NOMINAL_SAMPLES: usize = 2000;

fn temp_wav(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_audio_io_{}_{}.wav", process::id(), name));
  p
}

/// A temp path with an arbitrary `ext` (used to prove `load_audio`
/// dispatches on file *content*, not the extension).
fn temp_path(name: &str, ext: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_audio_io_{}_{}.{}", process::id(), name, ext));
  p
}

/// Write `bytes` to a fresh temp file at `path`, returning a guard-free
/// path (callers `fs::remove_file` in the test body, matching the
/// existing convention).
fn write_fixture(path: &std::path::Path, bytes: &[u8]) {
  let mut f = File::create(path).unwrap();
  f.write_all(bytes).unwrap();
  f.flush().unwrap();
}

/// Shared assertions for a decoded compressed fixture: correct sample
/// rate, a sane sample-count band around the 2000-frame nominal length
/// (tolerant of per-decoder encoder-delay handling), all-finite, and
/// not silently all-zero (proves real audio came through, not an empty
/// buffer).
fn assert_decoded_tone(samples: &[f32], sr: u32, fmt: &str) {
  assert_eq!(sr, FIXTURE_SR, "{fmt}: sample rate mismatch");
  // Tolerance band: at least half the nominal length, at most ~2.5x
  // (covers MP3 decoders that prepend ~1000+ samples of encoder delay
  // and a final partial frame).
  assert!(
    samples.len() >= FIXTURE_NOMINAL_SAMPLES / 2
      && samples.len() <= FIXTURE_NOMINAL_SAMPLES * 5 / 2,
    "{fmt}: decoded {} samples, expected ~{FIXTURE_NOMINAL_SAMPLES}",
    samples.len()
  );
  assert!(
    samples.iter().all(|s| s.is_finite()),
    "{fmt}: decoded a non-finite sample"
  );
  assert!(
    samples.iter().any(|&s| s.abs() > 1e-4),
    "{fmt}: decoded an all-(near-)zero buffer (codec likely not wired)"
  );
  // Normalized PCM is always in [-1, 1].
  assert!(
    samples.iter().all(|&s| (-1.0..=1.0).contains(&s)),
    "{fmt}: decoded a sample outside [-1, 1]"
  );
}

/// Roll-own stereo i16 PCM WAV writer for the `load_wav_rejects_multichannel`
/// test. Mirrors the exact 44-byte RIFF/WAVE/fmt/data layout that
/// `save_wav` emits, parameterized by `channels` so the test can craft a
/// stereo file without re-introducing a heavyweight WAV dep into
/// `[dev-dependencies]`.
fn write_pcm16_wav(path: &std::path::Path, samples: &[i16], sample_rate: u32, channels: u16) {
  let bits_per_sample: u16 = 16;
  let block_align: u16 = channels * (bits_per_sample / 8);
  let data_size: u32 = (samples.len() as u32) * u32::from(bits_per_sample / 8);
  let file_size_minus_8: u32 = 36u32 + data_size;
  let byte_rate: u32 = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
  let mut header = [0u8; 44];
  header[0..4].copy_from_slice(b"RIFF");
  header[4..8].copy_from_slice(&file_size_minus_8.to_le_bytes());
  header[8..12].copy_from_slice(b"WAVE");
  header[12..16].copy_from_slice(b"fmt ");
  header[16..20].copy_from_slice(&16u32.to_le_bytes());
  header[20..22].copy_from_slice(&1u16.to_le_bytes());
  header[22..24].copy_from_slice(&channels.to_le_bytes());
  header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
  header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
  header[32..34].copy_from_slice(&block_align.to_le_bytes());
  header[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
  header[36..40].copy_from_slice(b"data");
  header[40..44].copy_from_slice(&data_size.to_le_bytes());
  let mut f = File::create(path).unwrap();
  f.write_all(&header).unwrap();
  for s in samples {
    f.write_all(&s.to_le_bytes()).unwrap();
  }
  f.flush().unwrap();
}

#[test]
fn wav_round_trip_preserves_samples_within_quantization() {
  let path = temp_wav("round_trip");
  let samples: Vec<f32> = (0..32)
    .map(|i| ((i as f32) / 16.0 - 1.0).clamp(-1.0, 1.0))
    .collect();
  save_wav(&path, &samples, 16_000).unwrap();
  let (got, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, 16_000);
  assert_eq!(got.len(), samples.len());
  // 16-bit PCM round-trip quantization step is 1 / 32768; allow a slightly
  // larger tolerance to absorb rounding-direction differences in
  // `(x * 32767).round() → / 32768` (asymmetric scale + symmetric divide).
  for (g, w) in got.iter().zip(samples.iter()) {
    assert!(
      (g - w).abs() <= 1.0 / 16_384.0,
      "round-trip diff too large: got={g} want={w}"
    );
  }
  let _ = fs::remove_file(&path);
}

#[test]
fn wav_round_trip_preserves_sample_rate_44100() {
  let path = temp_wav("sr44100");
  let samples = vec![0.0_f32; 8];
  save_wav(&path, &samples, 44_100).unwrap();
  let (_, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, 44_100);
  let _ = fs::remove_file(&path);
}

#[test]
fn save_clips_out_of_range_samples() {
  // Samples > 1.0 should be clipped to +1 (which quantizes to 32767 → ≈ +0.99997).
  let path = temp_wav("clip");
  save_wav(&path, &[2.0_f32, -3.0, 0.0], 8_000).unwrap();
  let (got, _) = load_audio(&path).unwrap();
  assert_eq!(got.len(), 3);
  assert!(got[0] > 0.999, "+2 should clip to ~+1, got {}", got[0]);
  assert!(got[1] < -0.999, "-3 should clip to ~-1, got {}", got[1]);
  assert!(
    got[2].abs() < 1.0 / 16_384.0,
    "0.0 should stay ~0, got {}",
    got[2]
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn save_rejects_nan_before_touching_destination() {
  // Per save_wav's contract: NaN is rejected UPFRONT, before any tempfile
  // is created. Pre-stage a known-good marker file at the target path
  // and verify save_wav's error path does NOT replace/truncate it.
  let path = temp_wav("nan");
  fs::write(&path, b"PRESERVED").unwrap();
  let r = save_wav(&path, &[0.0_f32, f32::NAN, 0.0], 8_000);
  // NaN sample is rejected as LayerKeyed(NonFiniteScalar(_)) — see save_wav.
  assert!(matches!(r, Err(mlxrs::Error::LayerKeyed(_))));
  // The destination must be untouched (validation runs before any
  // filesystem mutation, including the tempfile create).
  let stored = fs::read(&path).unwrap();
  assert_eq!(stored, b"PRESERVED");
  let _ = fs::remove_file(&path);
}

#[test]
fn save_rejects_zero_sample_rate() {
  let path = temp_wav("zero_sr");
  fs::write(&path, b"PRESERVED").unwrap();
  let r = save_wav(&path, &[0.0_f32, 0.5, -0.5], 0);
  // sample_rate == 0 surfaces as InvariantViolation.
  assert!(matches!(r, Err(mlxrs::Error::InvariantViolation(_))));
  // Same guarantee: destination preserved on header validation failure.
  let stored = fs::read(&path).unwrap();
  assert_eq!(stored, b"PRESERVED");
  let _ = fs::remove_file(&path);
}

#[test]
fn load_wav_rejects_multichannel() {
  // Write a stereo WAV via the roll-own helper above, then assert
  // load_wav errors out — the mono-only signature of load_wav cannot
  // faithfully represent a stereo input per the doc contract.
  let path = temp_wav("stereo");
  // 4 interleaved L/R sample pairs = 8 i16 samples.
  let interleaved: &[i16] = &[100, -100, 200, -200, 300, -300, 400, -400];
  write_pcm16_wav(&path, interleaved, 16_000, 2);
  let r = load_audio(&path);
  // Multi-channel input is rejected via OutOfRange (channel count must be 1).
  assert!(matches!(r, Err(mlxrs::Error::OutOfRange(_))));
  let _ = fs::remove_file(&path);
}

#[test]
fn resample_passthrough_at_equal_rates() {
  let samples = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];
  let got = resample_linear(&samples, 16_000, 16_000).unwrap();
  // Verbatim copy on `from_rate == to_rate`.
  assert_eq!(got, samples);
}

#[test]
fn resample_upsample_doubles_length() {
  let samples = vec![0.0_f32, 1.0, 0.0, 1.0];
  let got = resample_linear(&samples, 8_000, 16_000).unwrap();
  // Output length = 4 * 16000 / 8000 = 8.
  assert_eq!(got.len(), 8);
  // The first sample is `samples[0]` (frac=0); the second is at source
  // position 0.5 = midpoint of (0.0, 1.0) = 0.5.
  assert!((got[0] - 0.0).abs() < 1e-6, "got[0]={}", got[0]);
  assert!((got[1] - 0.5).abs() < 1e-6, "got[1]={}", got[1]);
}

#[test]
fn resample_downsample_halves_length() {
  let samples = vec![0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25];
  let got = resample_linear(&samples, 16_000, 8_000).unwrap();
  // Output length = 8 * 8000 / 16000 = 4.
  assert_eq!(got.len(), 4);
}

#[test]
fn resample_rejects_zero_from_rate() {
  let r = resample_linear(&[0.0_f32, 1.0], 0, 16_000);
  // from_rate == 0 surfaces as InvariantViolation (pre-existing).
  assert!(matches!(r, Err(mlxrs::Error::InvariantViolation(_))));
}

#[test]
fn resample_rejects_zero_to_rate() {
  let r = resample_linear(&[0.0_f32, 1.0], 16_000, 0);
  assert!(matches!(r, Err(mlxrs::Error::InvariantViolation(_))));
}

#[test]
fn resample_empty_input_returns_empty() {
  let got = resample_linear(&[], 8_000, 16_000).unwrap();
  assert!(got.is_empty());
}

#[test]
fn resample_rejects_oversized_output_cap() {
  // Adversarial ratio: 2 samples * u32::MAX / 1 = ~8.6B output samples,
  // which exceeds MAX_RESAMPLED_SAMPLES (64 Mi). Must error BEFORE any
  // allocation attempt.
  let r = resample_linear(&[0.5_f32, -0.5], 1, u32::MAX);
  // Oversized output surfaces as CapExceeded against MAX_RESAMPLED_SAMPLES.
  assert!(matches!(r, Err(mlxrs::Error::CapExceeded(_))));
}

#[test]
fn load_wav_missing_file_returns_backend_error() {
  // Use `temp_wav` (pid-suffixed) so concurrent test processes don't
  // collide on a shared filename — same convention the rest of this
  // module follows.
  let path = temp_wav("missing");
  // Make absolutely sure it doesn't exist (a stale file from a prior run
  // would mask the test).
  let _ = fs::remove_file(&path);
  let r = load_audio(&path);
  // Missing file produces a FileIo error from the `File::open` step (the
  // io::Error is wrapped in FileIoPayload with FileOp::Open and the path).
  assert!(matches!(r, Err(mlxrs::Error::FileIo(_))));
}

// -------- New tests covering the atomic-rename addition --------

#[test]
fn save_wav_atomic_no_tempfile_remains_after_successful_save() {
  // After a successful save_wav, there must be no stray `*.tmp` siblings
  // of the destination — the tempfile must have been atomically renamed
  // into place. Sweep the temp dir for `*.<pid>.<rand>.tmp` files
  // matching the destination's basename prefix.
  let path = temp_wav("no_temp_remains");
  let samples: Vec<f32> = vec![0.1_f32, -0.1, 0.2, -0.2, 0.3];
  save_wav(&path, &samples, 8_000).unwrap();
  let final_name = path.file_name().unwrap().to_string_lossy().into_owned();
  let parent = path.parent().unwrap();
  let mut stray_tempfiles: Vec<PathBuf> = Vec::new();
  for entry in fs::read_dir(parent).unwrap().flatten() {
    let name = entry.file_name().to_string_lossy().into_owned();
    if name.starts_with(&final_name) && name.ends_with(".tmp") {
      stray_tempfiles.push(entry.path());
    }
  }
  assert!(
    stray_tempfiles.is_empty(),
    "found stray tempfile(s) after successful save: {stray_tempfiles:?}"
  );
  // Sanity: the destination is the new file (load works).
  let (got, _) = load_audio(&path).unwrap();
  assert_eq!(got.len(), samples.len());
  let _ = fs::remove_file(&path);
}

#[test]
fn save_wav_atomically_replaces_existing_file() {
  // Pre-stage a marker file at the destination, then save_wav over it
  // with real audio. The destination must contain the new WAV bytes
  // (load_wav returns the saved samples) — never the marker bytes —
  // and the marker bytes must not persist anywhere else.
  let path = temp_wav("replaces_existing");
  fs::write(&path, b"OLD_MARKER_DATA_THAT_IS_NOT_A_WAV").unwrap();
  let samples: Vec<f32> = vec![0.0_f32, 0.5, -0.5, 0.25];
  save_wav(&path, &samples, 16_000).unwrap();
  // Loading via symphonia must succeed (would fail if the marker bytes
  // were still at the destination, since they aren't a valid WAV).
  let (got, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, 16_000);
  assert_eq!(got.len(), samples.len());
  // Confirm the raw bytes start with "RIFF" — proving it's the new
  // WAV file, not the marker.
  let raw = fs::read(&path).unwrap();
  assert_eq!(&raw[0..4], b"RIFF", "destination is not a fresh WAV file");
  let _ = fs::remove_file(&path);
}

#[test]
fn load_wav_rejects_truncated_wav() {
  // Regression: silently skipping a decoder IoError/DecodeError would
  // return Ok with fewer samples than the header declared. The load
  // path either (a) fails the decode step with Error::Backend, or (b)
  // reaches the post-loop header_len-vs-out.len mismatch check and
  // fails there.
  //
  // Build a valid 16-bit mono WAV header that declares 32 samples,
  // but only write 8 samples of data. Either failure mode (decode
  // error, or the count cross-check) must surface as
  // Error::Backend — load_wav must NOT return Ok with a truncated
  // sample buffer.
  let path = temp_wav("truncated");
  let bits_per_sample: u16 = 16;
  let channels: u16 = 1;
  let sample_rate: u32 = 16_000;
  let block_align: u16 = channels * (bits_per_sample / 8);
  let declared_samples: u32 = 32;
  let actual_samples: u32 = 8;
  // Header declares 32 samples (32 * 2 = 64 data bytes) but we only
  // emit `actual_samples * 2` bytes of data.
  let data_size_declared: u32 = declared_samples * u32::from(bits_per_sample / 8);
  let file_size_minus_8: u32 = 36u32 + data_size_declared;
  let byte_rate: u32 = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
  let mut header = [0u8; 44];
  header[0..4].copy_from_slice(b"RIFF");
  header[4..8].copy_from_slice(&file_size_minus_8.to_le_bytes());
  header[8..12].copy_from_slice(b"WAVE");
  header[12..16].copy_from_slice(b"fmt ");
  header[16..20].copy_from_slice(&16u32.to_le_bytes());
  header[20..22].copy_from_slice(&1u16.to_le_bytes());
  header[22..24].copy_from_slice(&channels.to_le_bytes());
  header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
  header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
  header[32..34].copy_from_slice(&block_align.to_le_bytes());
  header[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
  header[36..40].copy_from_slice(b"data");
  header[40..44].copy_from_slice(&data_size_declared.to_le_bytes());
  let mut f = File::create(&path).unwrap();
  f.write_all(&header).unwrap();
  for i in 0..actual_samples as i16 {
    f.write_all(&i.to_le_bytes()).unwrap();
  }
  f.flush().unwrap();
  drop(f);
  let r = load_audio(&path);
  // A truncated WAV either fails the symphonia decode (Parse) or the
  // post-loop sample-count cross-check (LengthMismatch).
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::Parse(_) | mlxrs::Error::LengthMismatch(_))
    ),
    "load_wav must reject a truncated WAV; got {r:?}"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn save_wav_rejects_sample_rate_exceeding_byte_rate_u32_ceiling() {
  // Cap: save_wav rejects sample_rate values whose
  // byte_rate = sample_rate * 2 would wrap u32. Upper bound is
  // u32::MAX / 2 = 2147483647; anything above that must error UPFRONT
  // before any tempfile is created.
  let path = temp_wav("sr_overflow");
  fs::write(&path, b"PRESERVED").unwrap();
  let r = save_wav(&path, &[0.0_f32], u32::MAX);
  // sample_rate > MAX_SAMPLE_RATE_FOR_MONO_I16 surfaces as OutOfRange.
  assert!(matches!(r, Err(mlxrs::Error::OutOfRange(_))));
  let stored = fs::read(&path).unwrap();
  assert_eq!(stored, b"PRESERVED");
  let _ = fs::remove_file(&path);
}

#[cfg(unix)]
#[test]
fn save_wav_preserves_existing_destination_mode_bits() {
  // Regression: when save_wav overwrites an
  // existing file, the post-rename mode bits must match the prior
  // mode bits — the fresh-tempfile-inode path would otherwise drop
  // back to the process umask. Pre-stage a 0600 file at the
  // destination, save_wav over it, and assert the post-save mode is
  // still 0600.
  use std::os::unix::fs::PermissionsExt;
  let path = temp_wav("preserve_perms");
  fs::write(&path, b"prior content (does not need to be a valid WAV)").unwrap();
  fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
  let pre_mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
  assert_eq!(pre_mode, 0o600, "test precondition: pre-set mode is 0600");
  save_wav(&path, &[0.0_f32, 0.5, -0.5], 16_000).unwrap();
  let post_mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
  assert_eq!(
    post_mode, 0o600,
    "post-save mode bits drifted: pre={pre_mode:o} post={post_mode:o}"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn load_wav_decodes_24bit_pcm_mono_wav() {
  // Regression: load_wav must accept 24-bit mono
  // PCM WAVs (not just 16-bit). Craft one by hand and assert load_wav
  // returns f32 samples in [-1, 1] with the expected per-sample value.
  // Three 24-bit samples: +max (0x7fffff = 8388607), 0, -max (0x800001 = -8388607).
  let path = temp_wav("pcm24");
  let bits_per_sample: u16 = 24;
  let channels: u16 = 1;
  let sample_rate: u32 = 16_000;
  let block_align: u16 = channels * (bits_per_sample / 8);
  let n_samples: u32 = 3;
  let data_size: u32 = n_samples * u32::from(bits_per_sample / 8);
  let file_size_minus_8: u32 = 36u32 + data_size;
  let byte_rate: u32 = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
  let mut header = [0u8; 44];
  header[0..4].copy_from_slice(b"RIFF");
  header[4..8].copy_from_slice(&file_size_minus_8.to_le_bytes());
  header[8..12].copy_from_slice(b"WAVE");
  header[12..16].copy_from_slice(b"fmt ");
  header[16..20].copy_from_slice(&16u32.to_le_bytes());
  header[20..22].copy_from_slice(&1u16.to_le_bytes());
  header[22..24].copy_from_slice(&channels.to_le_bytes());
  header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
  header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
  header[32..34].copy_from_slice(&block_align.to_le_bytes());
  header[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
  header[36..40].copy_from_slice(b"data");
  header[40..44].copy_from_slice(&data_size.to_le_bytes());
  let mut f = File::create(&path).unwrap();
  f.write_all(&header).unwrap();
  // +0x7fffff (max positive 24-bit signed)
  f.write_all(&[0xff, 0xff, 0x7f]).unwrap();
  // 0
  f.write_all(&[0x00, 0x00, 0x00]).unwrap();
  // -0x7fffff (min nonzero negative 24-bit signed = 2-complement of 0x7fffff = 0x800001)
  f.write_all(&[0x01, 0x00, 0x80]).unwrap();
  f.flush().unwrap();
  drop(f);
  let (got, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, 16_000);
  assert_eq!(got.len(), 3);
  // 24-bit divisor = 2^23 = 8388608. Expected: 8388607/8388608, 0, -8388607/8388608.
  let expected = [8_388_607.0 / 8_388_608.0, 0.0, -8_388_607.0 / 8_388_608.0];
  for (g, w) in got.iter().zip(expected.iter()) {
    assert!(
      (g - w).abs() < 1.0 / (1u64 << 22) as f32,
      "24-bit decode mismatch: got={g} want={w}"
    );
  }
  let _ = fs::remove_file(&path);
}

#[test]
fn load_wav_via_symphonia_roundtrip_matches_save_wav_output() {
  // End-to-end "roll-own encoder feeds symphonia decoder" check.
  // Pretty much the existing `wav_round_trip_preserves_samples_within_quantization`
  // test, kept as a named regression to catch any future drift between
  // our encoder's emitted byte layout and what symphonia expects.
  let path = temp_wav("symphonia_roundtrip");
  let samples: Vec<f32> = (0..64)
    .map(|i| ((i as f32) / 32.0 - 1.0).clamp(-1.0, 1.0))
    .collect();
  save_wav(&path, &samples, 22_050).unwrap();
  let (got, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, 22_050);
  assert_eq!(got.len(), samples.len());
  for (g, w) in got.iter().zip(samples.iter()) {
    assert!(
      (g - w).abs() <= 1.0 / 16_384.0,
      "symphonia round-trip diff too large: got={g} want={w}"
    );
  }
  let _ = fs::remove_file(&path);
}

// -------- New tests: MP3 / FLAC / OGG-Vorbis decode --------

#[test]
fn load_audio_decodes_mp3() {
  // Decode a real (tiny) MP3 fixture through the public `load_audio` API
  // — proves the `mp3` symphonia feature is enabled and the lossy-decode
  // → f32 pass-through arm works end to end.
  let path = temp_path("mp3_decode", "mp3");
  write_fixture(&path, FIXTURE_MP3);
  let (samples, sr) = load_audio(&path).unwrap();
  assert_decoded_tone(&samples, sr, "mp3");
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_decodes_flac() {
  // Decode a real FLAC fixture — proves the `flac` symphonia feature is
  // enabled. FLAC decodes to the integer-PCM arm (lossless), so this
  // also exercises the `/2^(bits-1)` normalization on a non-WAV source.
  let path = temp_path("flac_decode", "flac");
  write_fixture(&path, FIXTURE_FLAC);
  let (samples, sr) = load_audio(&path).unwrap();
  assert_decoded_tone(&samples, sr, "flac");
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_flac_decodes_exact_streaminfo_sample_count() {
  // FLAC's STREAMINFO carries an EXACT total sample count, which
  // symphonia surfaces as the track's `num_frames`. `load_audio` treats
  // FLAC-with-a-declared-total like WAV (an exact-count format) and
  // applies the strict post-decode equality cross-check — so the intact
  // fixture must decode to EXACTLY its declared 2000 samples (not merely
  // a tolerance band). If this drifts off 2000, either the fixture or
  // the exact-count gate changed.
  let path = temp_path("flac_exact", "flac");
  write_fixture(&path, FIXTURE_FLAC);
  let (samples, sr) = load_audio(&path).unwrap();
  assert_eq!(sr, FIXTURE_SR, "flac: sample rate mismatch");
  assert_eq!(
    samples.len(),
    FIXTURE_NOMINAL_SAMPLES,
    "flac: exact STREAMINFO total must decode to exactly {FIXTURE_NOMINAL_SAMPLES} samples"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_rejects_truncated_flac() {
  // A FLAC truncated mid-stream declares (via STREAMINFO) more samples
  // than survive in the truncated byte buffer. Symphonia can hit a clean
  // EOF after the partial frames; a WAV-only gate would then return `Ok`
  // with missing audio (silent corruption). With FLAC treated as an
  // exact-count format, the post-decode `out.len() == declared`
  // cross-check (or an earlier decode error) must surface `Error::Backend`
  // instead.
  //
  // We truncate to the first half of the fixture bytes, which keeps the
  // STREAMINFO header (so `num_frames` = 2000 is known) but drops the
  // tail audio frames.
  let path = temp_path("flac_truncated", "flac");
  let cut = FIXTURE_FLAC.len() / 2;
  write_fixture(&path, &FIXTURE_FLAC[..cut]);
  let r = load_audio(&path);
  // A truncated FLAC either fails the symphonia decode (Parse) or the
  // post-loop sample-count cross-check (LengthMismatch).
  assert!(
    matches!(
      r,
      Err(mlxrs::Error::Parse(_) | mlxrs::Error::LengthMismatch(_))
    ),
    "truncated FLAC must be rejected (count mismatch / corruption), got {r:?}"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_decodes_ogg_vorbis() {
  // Decode a real mono OGG/Vorbis fixture — proves BOTH the `ogg`
  // (container demux) AND `vorbis` (audio codec) symphonia features are
  // enabled and that Vorbis audio packets decode to finite f32 samples.
  let path = temp_path("ogg_decode", "ogg");
  write_fixture(&path, FIXTURE_OGG_VORBIS);
  let (samples, sr) = load_audio(&path).unwrap();
  assert_decoded_tone(&samples, sr, "ogg/vorbis");
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_autodetects_format_ignoring_extension() {
  // Format dispatch is by CONTENT, not file extension: write the MP3
  // fixture bytes to a file with a misleading `.flac` extension and a
  // FLAC fixture to a `.wav` extension; both must still decode correctly
  // via symphonia's probe (the extension is only a probe *hint*).
  let mp3_as_flac = temp_path("mislabeled_mp3", "flac");
  write_fixture(&mp3_as_flac, FIXTURE_MP3);
  let (samples, sr) = load_audio(&mp3_as_flac).unwrap();
  assert_decoded_tone(&samples, sr, "mp3-labeled-flac");
  let _ = fs::remove_file(&mp3_as_flac);

  let flac_as_wav = temp_path("mislabeled_flac", "wav");
  write_fixture(&flac_as_wav, FIXTURE_FLAC);
  let (samples, sr) = load_audio(&flac_as_wav).unwrap();
  assert_decoded_tone(&samples, sr, "flac-labeled-wav");
  let _ = fs::remove_file(&flac_as_wav);
}

#[test]
fn load_audio_autodetects_format_with_no_extension() {
  // No extension at all (no hint) — symphonia must still detect the
  // container from its magic bytes. Exercises each newly-enabled format.
  for (name, bytes, fmt) in [
    ("noext_mp3", FIXTURE_MP3, "mp3"),
    ("noext_flac", FIXTURE_FLAC, "flac"),
    ("noext_ogg", FIXTURE_OGG_VORBIS, "ogg/vorbis"),
  ] {
    let mut p = std::env::temp_dir();
    p.push(format!("mlxrs_audio_io_{}_{}", process::id(), name));
    write_fixture(&p, bytes);
    let (samples, sr) = load_audio(&p).unwrap();
    assert_decoded_tone(&samples, sr, fmt);
    let _ = fs::remove_file(&p);
  }
}

#[test]
fn load_audio_rejects_unsupported_opus_like_garbage() {
  // A buffer that is not any supported container must surface a
  // recoverable Err (probe failure), never a panic. This stands in for
  // the scoped-out formats (Opus/M4A/AAC/WebM): mlxrs has no in-process
  // decoder for them, so they fail the probe like any other unknown
  // container. Use a non-trivial byte pattern so the probe actually runs
  // its format detection rather than short-circuiting on an empty file.
  let path = temp_path("unsupported", "opus");
  let garbage: Vec<u8> = (0..512u32).map(|i| (i % 251) as u8).collect();
  write_fixture(&path, &garbage);
  let r = load_audio(&path);
  // Unsupported / garbage input fails the symphonia probe as Parse.
  assert!(
    matches!(r, Err(mlxrs::Error::Parse(_))),
    "unsupported/garbage input must return Parse error, got {r:?}"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn load_audio_truncated_compressed_is_bounded_and_recoverable() {
  // Bounded-memory / robustness: a compressed stream truncated mid-file
  // must NEVER panic, hang, or allocate unboundedly. It returns either a
  // recoverable Err (the common case — decode error / failed page CRC /
  // short read) OR `Ok` with a sample count bounded by
  // `MAX_DECODED_SAMPLES`. Both are acceptable; the invariant under test
  // is "completes with a bounded Result". (We deliberately do NOT
  // require `Err`: MP3 is frame-resync tolerant and a cut on a frame
  // boundary can legitimately decode a shorter clean clip.)
  //
  // The strict-Err truncation guarantee is covered for the sample-exact
  // WAV path by `load_wav_rejects_truncated_wav`; the cap mechanism is
  // covered by `resample_rejects_oversized_output_cap`.
  for (name, bytes, ext) in [
    ("trunc_mp3", FIXTURE_MP3, "mp3"),
    ("trunc_flac", FIXTURE_FLAC, "flac"),
    ("trunc_ogg", FIXTURE_OGG_VORBIS, "ogg"),
  ] {
    let path = temp_path(name, ext);
    let cut = (bytes.len() * 2) / 5; // ~40%
    write_fixture(&path, &bytes[..cut]);
    match load_audio(&path) {
      // Recoverable error: Parse (codec/probe) or LengthMismatch (count
      // cross-check for exact-count formats).
      Err(mlxrs::Error::Parse(_) | mlxrs::Error::LengthMismatch(_)) => { /* ok */ }
      Ok((samples, _)) => {
        assert!(
          samples.len() <= mlxrs::audio::io::MAX_DECODED_SAMPLES,
          "{ext}: truncated decode returned {} samples (> cap)",
          samples.len()
        );
        assert!(
          samples.iter().all(|s| s.is_finite()),
          "{ext}: truncated decode returned a non-finite sample"
        );
      }
      Err(other) => panic!("{ext}: unexpected error variant: {other:?}"),
    }
    let _ = fs::remove_file(&path);
  }
}

// ---- #132: load_audio_into buffer reuse ----------------------------------

/// `load_audio_into` decodes a WAV into the caller's existing `Vec<f32>`
/// and reuses its pre-allocated capacity across calls. The returned
/// sample rate matches `load_audio`, and the decoded samples are
/// value-for-value identical.
#[test]
fn load_audio_into_reuses_buffer_across_calls() {
  use mlxrs::audio::io::load_audio_into;

  let path1 = temp_wav("load_into_reuse_1");
  let path2 = temp_wav("load_into_reuse_2");
  let s1: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0).sin() * 0.5).collect();
  let s2: Vec<f32> = (0..500).map(|i| (i as f32 / 100.0).cos() * 0.25).collect();
  save_wav(&path1, &s1, 16_000).unwrap();
  save_wav(&path2, &s2, 16_000).unwrap();

  let mut scratch: Vec<f32> = Vec::with_capacity(2000);
  let cap_before = scratch.capacity();

  let sr1 = load_audio_into(&path1, &mut scratch).unwrap();
  assert_eq!(sr1, 16_000);
  assert_eq!(scratch.len(), s1.len());
  // f32 round-trip via i16 has up to 1/32767 quantization error.
  for (i, (g, e)) in scratch.iter().zip(s1.iter()).enumerate() {
    assert!(
      (g - e).abs() < 1.1 / 32767.0,
      "load_into[{i}]: got {g}, want {e}"
    );
  }

  // Second load reuses the capacity (>= cap_before, since we already
  // had room for the first/second file's samples).
  let sr2 = load_audio_into(&path2, &mut scratch).unwrap();
  assert_eq!(sr2, 16_000);
  assert_eq!(scratch.len(), s2.len());
  assert!(
    scratch.capacity() >= cap_before,
    "buffer reuse must not shrink capacity: {} < {cap_before}",
    scratch.capacity()
  );
  // Samples from path1 are gone (cleared), only path2 samples remain.
  for (i, (g, e)) in scratch.iter().zip(s2.iter()).enumerate() {
    assert!(
      (g - e).abs() < 1.1 / 32767.0,
      "load_into[{i}]: got {g}, want {e}"
    );
  }
  let _ = fs::remove_file(&path1);
  let _ = fs::remove_file(&path2);
}

// ---- #137: load_audio_with_cap rejects oversized before alloc ------------

/// `load_audio_with_cap` rejects a WAV declaring more samples than the
/// caller's `max_samples` cap BEFORE allocating the sample buffer.
/// Uses a 100-sample WAV and a 50-sample cap; the rejection fires at
/// the header-parse stage with a recoverable `Error::Backend`.
#[test]
fn load_audio_with_cap_rejects_oversized_wav_at_header_stage() {
  use mlxrs::{audio::io::load_audio_with_cap, error::Error};

  let path = temp_wav("with_cap_oversized");
  let s: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0).sin() * 0.3).collect();
  save_wav(&path, &s, 16_000).unwrap();

  // Cap at 50 samples — strictly below the header's 100.
  let r = load_audio_with_cap(&path, 50);
  // Exact-count header rejection surfaces as CapExceeded.
  assert!(
    matches!(r, Err(Error::CapExceeded(_))),
    "over-cap WAV header must reject with CapExceeded, got {r:?}"
  );

  let _ = fs::remove_file(&path);
}

/// Under-cap (cap >= header sample count) decodes normally; the
/// returned `Vec<f32>` equals what `load_audio` would have returned.
#[test]
fn load_audio_with_cap_undersized_decodes_identically() {
  use mlxrs::audio::io::load_audio_with_cap;

  let path = temp_wav("with_cap_undersized");
  let s: Vec<f32> = (0..200).map(|i| (i as f32 / 100.0).sin() * 0.25).collect();
  save_wav(&path, &s, 16_000).unwrap();

  // Cap at 1000 (well above the WAV's 200).
  let (got_cap, sr_cap) = load_audio_with_cap(&path, 1000).unwrap();
  let (got_plain, sr_plain) = load_audio(&path).unwrap();
  assert_eq!(sr_cap, sr_plain);
  assert_eq!(got_cap, got_plain);

  let _ = fs::remove_file(&path);
}

/// `load_audio` (no cap) is equivalent to `load_audio_with_cap` with
/// `MAX_DECODED_SAMPLES`. Tests the delegating wrapper.
#[test]
fn load_audio_equivalent_to_with_cap_at_max() {
  use mlxrs::audio::io::{MAX_DECODED_SAMPLES, load_audio_with_cap};

  let path = temp_wav("with_cap_at_max");
  let s: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0).cos() * 0.2).collect();
  save_wav(&path, &s, 16_000).unwrap();

  let (got_plain, _) = load_audio(&path).unwrap();
  let (got_at_max, _) = load_audio_with_cap(&path, MAX_DECODED_SAMPLES).unwrap();
  let (got_at_usize_max, _) = load_audio_with_cap(&path, usize::MAX).unwrap();
  assert_eq!(got_plain, got_at_max);
  assert_eq!(got_plain, got_at_usize_max);

  let _ = fs::remove_file(&path);
}

// ---- #133: save_wav_into scratch-buffer reuse + Quantizer trait ----------

/// `save_wav_into` writes a WAV identical to `save_wav` while reusing a
/// caller-provided `Vec<i16>` scratch buffer. Two consecutive calls
/// share the same scratch (capacity is preserved across calls).
#[test]
fn save_wav_into_reuses_scratch_buffer() {
  use mlxrs::audio::io::save_wav_into;

  let path_into = temp_wav("save_into_reuse");
  let path_plain = temp_wav("save_plain_reuse");

  let s1: Vec<f32> = (0..800).map(|i| (i as f32 / 50.0).sin() * 0.6).collect();
  let s2: Vec<f32> = (0..400).map(|i| (i as f32 / 25.0).cos() * 0.3).collect();

  let mut scratch: Vec<i16> = Vec::new();
  save_wav_into(&path_into, &s1, 16_000, &mut scratch).unwrap();
  let cap_after_first = scratch.capacity();
  assert!(
    cap_after_first >= s1.len(),
    "scratch did not retain capacity"
  );

  save_wav_into(&path_into, &s2, 16_000, &mut scratch).unwrap();
  // Smaller second write — capacity stays at the high-water mark.
  assert!(
    scratch.capacity() >= cap_after_first,
    "scratch shrank on smaller write: {} < {cap_after_first}",
    scratch.capacity()
  );

  // Cross-check value parity vs `save_wav`.
  save_wav(&path_plain, &s2, 16_000).unwrap();
  let into_bytes = fs::read(&path_into).unwrap();
  let plain_bytes = fs::read(&path_plain).unwrap();
  assert_eq!(
    into_bytes, plain_bytes,
    "save_wav_into must produce byte-identical WAV vs save_wav"
  );

  let _ = fs::remove_file(&path_into);
  let _ = fs::remove_file(&path_plain);
}

/// `I16Quantizer` is the `Quantizer<f32, i16>` impl used by `save_wav` /
/// `save_wav_into` — clip + scale + cast in one pass. Wires through the
/// SIMD dispatcher; results are bit-identical regardless of caller.
#[test]
fn i16_quantizer_matches_simd_dispatcher() {
  use core::mem::MaybeUninit;
  use mlxrs::audio::io::{I16Quantizer, Quantizer};

  let src: Vec<f32> = vec![
    -1.5, // clip down to -1.0 → -1.0 * 32768 = -32768 (i16::MIN, no saturation)
    -1.0, // -1.0 * 32768 = -32768 (i16::MIN, no saturation)
    -0.5, // -16384 (round-half-away)
    0.0, 0.5,   // 16384
    1.0,   // +1.0 * 32768 = +32768 → saturates to i16::MAX = 32767
    1.5,   // clip up to +1.0 → same as above → 32767
    0.001, // tiny positive
  ];
  let mut dst_a: Vec<i16> = Vec::with_capacity(src.len());
  let mut dst_b: Vec<i16> = Vec::with_capacity(src.len());
  let spare_a: &mut [MaybeUninit<i16>] = dst_a.spare_capacity_mut();
  I16Quantizer.quantize_into(&mut spare_a[..src.len()], &src);
  // SAFETY: Quantizer contract initializes every cell of `dst` for `0..src.len()`.
  unsafe { dst_a.set_len(src.len()) };

  let spare_b: &mut [MaybeUninit<i16>] = dst_b.spare_capacity_mut();
  mlxrs::simd::audio::quantize::f32_to_i16_quantize(&mut spare_b[..src.len()], &src);
  // SAFETY: SIMD dispatcher contract initializes every cell of `dst`.
  unsafe { dst_b.set_len(src.len()) };

  assert_eq!(
    dst_a, dst_b,
    "I16Quantizer must produce identical output to the SIMD dispatcher"
  );
  // Clip bounds. The SIMD convention is `* 32768` (NOT `* 32767`); the
  // post-clip values land at the i16 extremes:
  //   `-1.5` → clip to `-1.0` → `-1.0 * 32768 = -32768` → in-range as `i16::MIN`.
  //   `+1.5` → clip to `+1.0` → `+1.0 * 32768 = +32768` → SATURATES to `i16::MAX = 32767`.
  // The asymmetry comes from `i16`'s range `[-32768, 32767]` and the
  // saturating-narrow on the positive boundary only.
  assert_eq!(dst_a[0], -32768, "-1.5 should clip to -32768 (i16::MIN)");
  assert_eq!(
    dst_a[6], 32767,
    "+1.5 should clip to +32767 (i16::MAX via saturating narrow)"
  );
}

// ---- unified probe+decode (no TOCTOU) + lossy overestimate -----------------

/// `load_audio_with_max_seconds` does
/// ONE `File::open` / probe, not two: the cap derivation and the decode
/// pass share the same `FormatReader`, so the cap is structurally
/// guaranteed to match the file actually being decoded.
///
/// A probe-then-decode design would re-open the path for a probe-only
/// `File::open` (to read `src_sr`) and then hand off to
/// `load_audio_with_cap` which re-opens a SECOND time for the decode.
/// Between those two opens a path could be replaced / symlink-swapped,
/// letting a high-rate probe authorize a much larger cap for a low-rate
/// decode of a different file. Unifying probe + decode against the same
/// handle closes that TOCTOU window.
///
/// Structural-via-behavior test: we write a WAV at the EXACT capacity
/// boundary (sample_rate * max_seconds == decoded_samples) and call
/// `load_audio_with_max_seconds`. The decode must succeed because the
/// cap was derived from the SAME file's `src_sr`. A stale probe at a
/// DIFFERENT sample rate (the bug pattern) would compute a different
/// cap, and at this exact-boundary configuration any cap drift would
/// either fail the load (cap one sample short) or succeed for the
/// wrong reason (cap loosened from a high-rate-probe of a stale
/// handle). The function returns `src_sr` from the SAME stream it
/// decoded — we assert that, plus the exact decoded sample count.
#[test]
fn load_audio_with_max_seconds_unified_probe_no_toctou() {
  use mlxrs::audio::io::load_audio_with_max_seconds;

  let path = temp_wav("max_seconds_unified_no_toctou");
  // 8000 samples at 8 kHz = 1.0 s exactly.
  let sr = 8000_u32;
  let samples: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin() * 0.4).collect();
  save_wav(&path, &samples, sr).unwrap();

  // max_seconds = 1.0 → cap = src_sr * 1.0 = 8000 samples — EXACTLY the
  // WAV's declared header length. With the unified probe+decode path the
  // `src_sr` driving the cap derivation IS the `src_sr` of the file being
  // decoded, so the boundary-fit succeeds. A TOCTOU stale-probe would
  // compute the cap from a DIFFERENT file's `src_sr` and the boundary
  // would no longer hold.
  let (got_samples, got_sr) = load_audio_with_max_seconds(&path, 1.0).unwrap();
  assert_eq!(
    got_sr, sr,
    "returned sr must equal the file's actual sr (not a stale probe's)"
  );
  assert_eq!(
    got_samples.len(),
    samples.len(),
    "exact-boundary decode (cap == header_len) must include every sample"
  );

  // Negative half of the structural assertion: a max_seconds budget
  // ONE FRAME below the file's actual duration must reject — the cap
  // is derived from the SAME file's `src_sr`, so the rejection fires
  // for the right reason (header length > cap). A stale probe at a
  // DIFFERENT rate would compute a different cap and the rejection
  // boundary would shift.
  use mlxrs::error::Error;
  // max_seconds = 0.999875 → cap = 8000 * 0.999875 = 7999 samples →
  // header_len = 8000 > 7999 → reject.
  let max_seconds_just_below = 7999.0 / 8000.0;
  let r = load_audio_with_max_seconds(&path, max_seconds_just_below);
  // Exact-count header beats cap → CapExceeded.
  assert!(
    matches!(r, Err(Error::CapExceeded(_))),
    "cap one frame below header must reject; got {r:?}"
  );

  let _ = fs::remove_file(&path);
}

/// Lossy formats (MP3 / OGG-Vorbis)
/// whose header can OVERESTIMATE the true decoded frame count must NOT
/// be rejected upfront by `header_len > effective_cap`. The upfront
/// rejection is reserved for exact-count formats (WAV / FLAC with
/// STREAMINFO); estimate formats clamp the reservation hint to the cap
/// and let `push_samples` enforce the actual cap during decode using
/// the REAL decoded sample count.
///
/// This covers the MP3 whose TRUE decoded length actually exceeds the
/// cap: it must STILL reject — the cap is enforced mid-decode by
/// `push_samples` for estimate formats (not upfront), so an over-cap
/// MP3 fails after partial decode rather than passing silently.
///
/// The two cap paths surface distinct error variants. An upfront
/// `header_len > cap` rejection on an exact-count format surfaces
/// `Error::CapExceeded`; an estimate-format MP3 over-cap rejection is
/// mid-decode via `push_samples`, which surfaces `Error::BoundedDecode`.
/// This test asserts the BoundedDecode variant — a regression that
/// routed MP3 through the upfront path would produce CapExceeded
/// instead, which a bare `matches!` on `Err(_)` alone would not catch.
#[test]
fn load_audio_with_max_seconds_mp3_genuinely_over_cap_rejects() {
  use mlxrs::{audio::io::load_audio_with_max_seconds, error::Error};

  let path = temp_path("mp3_over_cap_rejects", "mp3");
  write_fixture(&path, FIXTURE_MP3);

  // 0.01 s @ 8 kHz = 80-sample cap. The MP3's true decoded length is
  // ~2000 samples — well above the 80-sample cap. The per-buffer
  // `push_samples` cap MUST reject this even though the upfront
  // header check is skipped for estimate formats.
  let r = load_audio_with_max_seconds(&path, 0.01);
  // For an estimate-count format the rejection is mid-decode via
  // `push_samples`, which returns `Error::BoundedDecode(_)`. An upfront
  // `header_len > cap` rejection would return `Error::CapExceeded(_)`
  // instead — variant-typed so this distinction is machine-checked
  // (no string-substring brittleness).
  match &r {
    Err(Error::BoundedDecode(_)) => { /* mid-decode reject — ok */ }
    Err(Error::CapExceeded(p)) => panic!(
      "MP3 over-cap rejection came from the upfront \
       `header_len > cap` path (CapExceeded against `{}`); estimate-count formats \
       must reject mid-decode via BoundedDecode",
      p.cap_name()
    ),
    other => panic!("MP3 genuinely over cap must reject with Error::BoundedDecode; got {other:?}"),
  }

  let _ = fs::remove_file(&path);
}

/// STRUCTURAL guard for the TOCTOU fix: read the
/// `mlxrs/src/audio/io.rs` source and assert the `load_audio_into_unified`
/// worker function (the SOLE entry point for both `load_audio_with_cap`
/// and `load_audio_with_max_seconds`) does at most ONE `File::open`
/// and never reaches for any removed probe helper.
///
/// The companion behavioral test
/// (`load_audio_with_max_seconds_unified_probe_no_toctou`) verifies
/// exact-boundary cap behavior; this structural test pins the property
/// the boundary depends on — a SINGLE open per call — directly to the
/// source text so a regression that reintroduces a probe-then-decode
/// double-open is caught even if the behavioral fixture happens to
/// coincidentally pass (e.g. probe + decode both happen to read the
/// SAME sample rate from the SAME path in test).
///
/// A probe-then-decode design would open-then-close a `File::open` JUST
/// to derive `src_sr`, then have `load_audio_with_cap` reopen a SECOND
/// `File::open` to decode. That is the structural defect this test
/// forbids.
#[test]
fn load_audio_into_unified_has_single_file_open() {
  // Source of `mlxrs/src/audio/io.rs` — `include_str!` resolves
  // relative to THIS test file at `mlxrs/tests/audio_io.rs`.
  let src = include_str!("../src/audio/io.rs");

  // Locate the `load_audio_into_unified` definition and extract its
  // body via brace-matching from the opening `{` of the signature
  // through the matching `}` at depth zero. The function is a
  // top-level free `fn` (no enclosing `impl`/`mod`), so the source
  // text's leading-`fn` anchor uniquely identifies it.
  let sig = "fn load_audio_into_unified(";
  let sig_pos = src
    .find(sig)
    .expect("io.rs must define `fn load_audio_into_unified(`");
  let body_start_rel = src[sig_pos..]
    .find('{')
    .expect("`load_audio_into_unified` signature must be followed by `{`");
  let body_start = sig_pos + body_start_rel;

  // Scan forward from the opening `{`, tracking brace depth, and stop
  // at the matching `}`. This handles nested `{`/`}` from closures,
  // match arms, struct literals, and string literals are tolerated
  // because the function does not contain unescaped `{`/`}` inside
  // raw-string blocks (verified by the parse succeeding below — a
  // mismatched depth would either run off the end of the file or
  // include unrelated trailing items, and we assert the body length
  // is plausible).
  let bytes = src.as_bytes();
  let mut depth: i32 = 0;
  let mut body_end = body_start;
  for (i, &b) in bytes.iter().enumerate().skip(body_start) {
    if b == b'{' {
      depth += 1;
    } else if b == b'}' {
      depth -= 1;
      if depth == 0 {
        body_end = i + 1;
        break;
      }
    }
  }
  assert!(
    depth == 0 && body_end > body_start,
    "structural test could not locate the closing `}}` of \
     `load_audio_into_unified` — io.rs source layout may have changed; \
     update the brace-matcher in this test."
  );
  let body = &src[body_start..body_end];

  // Sanity: the body must be substantial — the unified worker is the
  // entire load path and runs into the hundreds of lines. A tiny body
  // would mean the brace-matcher latched onto a different `{...}`.
  assert!(
    body.len() > 1000,
    "load_audio_into_unified body looks suspiciously short ({} bytes); \
     structural test extracted the wrong region",
    body.len()
  );

  // Strip `//` line comments and `/* */` block comments from the body
  // before pattern-counting — the in-source comment block on line
  // ~377 explicitly mentions ``File::open`` in prose ("THIS IS THE
  // ONLY `File::open` IN THE WHOLE LOAD PATH"), which would cause a
  // naive `body.contains(...)` count to over-report by one. A simple
  // state machine handles `//` to end-of-line and `/* */` balanced
  // blocks. String literals are NOT stripped (they cannot legally
  // contain the bare identifier `File::open` without a `"..."` quote
  // around it, and the function body is verified by the size-sanity
  // assertion above to be the real worker, not a synthetic test
  // fixture). The body never uses raw `r#"..."#` strings, so the
  // simple state machine is correct here.
  let body_no_comments = strip_comments(body);

  // Count `File::open` occurrences in the comment-stripped body. Use
  // identifier-boundary matching so an unrelated `XFile::open` would
  // not match. The body is the canonical unified worker, and the
  // contract is exactly ONE `File::open` in it. A reintroduced
  // second open is the exact regression this test exists to catch.
  let file_open_count = body_no_comments
    .match_indices("File::open")
    .filter(|(idx, _)| {
      if *idx == 0 {
        return true;
      }
      let prev = body_no_comments.as_bytes()[*idx - 1];
      // Reject matches embedded in a longer identifier
      // (`SomeFile::open`, `fs::File::open` would still pass because
      // `:` is not an identifier-continuation char — and that fully
      // qualified form is exactly the same open).
      !(prev.is_ascii_alphanumeric() || prev == b'_')
    })
    .count();
  assert_eq!(
    file_open_count, 1,
    "STRUCTURAL regression: \
     `load_audio_into_unified` body must contain EXACTLY ONE \
     `File::open` (the unified probe+decode handle); found \
     {file_open_count} in the comment-stripped body. A reintroduced \
     second open is the TOCTOU defect this test guards against. \
     Comment-stripped body was:\n{body_no_comments}"
  );

  // A probe-then-decode design routes cap-derivation through a separate
  // probe helper (`probe_source_sample_rate`) that does its own
  // `File::open`. The unified worker has no such helper; if any
  // reference to it appears in the worker body (in code, not just in
  // comments), a probe-then-decode double-open has crept in.
  assert!(
    !body_no_comments.contains("probe_source_sample_rate"),
    "STRUCTURAL regression: \
     `load_audio_into_unified` body must NOT reference a \
     `probe_source_sample_rate` helper — such a helper performs a \
     SECOND `File::open` and reintroduces the TOCTOU double-open. \
     Comment-stripped body was:\n{body_no_comments}"
  );

  // Belt-and-braces: also assert the public-entry
  // `load_audio_with_max_seconds` body has ZERO `File::open` calls
  // (it delegates to the unified worker — any `File::open` here is
  // a probe-then-delegate two-open regression at the caller level).
  let pub_sig = "pub fn load_audio_with_max_seconds(";
  let pub_pos = src
    .find(pub_sig)
    .expect("io.rs must define `pub fn load_audio_with_max_seconds(`");
  let pub_body_start_rel = src[pub_pos..]
    .find('{')
    .expect("`load_audio_with_max_seconds` signature must be followed by `{`");
  let pub_body_start = pub_pos + pub_body_start_rel;
  let mut pub_depth: i32 = 0;
  let mut pub_body_end = pub_body_start;
  for (i, &b) in bytes.iter().enumerate().skip(pub_body_start) {
    if b == b'{' {
      pub_depth += 1;
    } else if b == b'}' {
      pub_depth -= 1;
      if pub_depth == 0 {
        pub_body_end = i + 1;
        break;
      }
    }
  }
  let pub_body = &src[pub_body_start..pub_body_end];
  let pub_body_no_comments = strip_comments(pub_body);
  let pub_file_open_count = pub_body_no_comments
    .match_indices("File::open")
    .filter(|(idx, _)| {
      if *idx == 0 {
        return true;
      }
      let prev = pub_body_no_comments.as_bytes()[*idx - 1];
      !(prev.is_ascii_alphanumeric() || prev == b'_')
    })
    .count();
  assert_eq!(
    pub_file_open_count, 0,
    "STRUCTURAL regression: \
     `load_audio_with_max_seconds` must delegate to the unified \
     worker without performing its own `File::open` (found \
     {pub_file_open_count} direct opens). The unified worker is the \
     SOLE owner of the load-path file handle. \
     Comment-stripped body was:\n{pub_body_no_comments}"
  );
}

/// Strip Rust `//` line comments (to end-of-line) and `/* ... */`
/// block comments (with nesting) from `src`, returning a `String`
/// with comment regions elided. String literals are passed through
/// verbatim — Rust string literals cannot embed an unescaped `//` or
/// `/*` that would be misclassified, and the source region this
/// is applied to (the audio-io worker function body) does not
/// currently use raw string literals (`r"..."` / `r#"..."#`).
///
/// Operates on `char_indices()` so multi-byte UTF-8 sequences (e.g.
/// em-dashes in source comments) are emitted intact when they fall
/// outside stripped regions — though in practice the audio/io.rs
/// worker confines all non-ASCII chars to comments, which are
/// stripped, so the output is ASCII-only.
///
/// Test-internal helper only — lexer-faithful enough for the
/// audio/io.rs function bodies the structural test inspects.
/// NOT a general Rust source preprocessor.
fn strip_comments(src: &str) -> String {
  let bytes = src.as_bytes();
  let mut out = String::with_capacity(src.len());
  let mut i = 0;
  while i < bytes.len() {
    let b = bytes[i];
    // `//` line comment: consume to end-of-line (newline preserved
    // by falling through on the next iteration, where `\n` is
    // emitted by the default arm).
    if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
      i += 2;
      while i < bytes.len() && bytes[i] != b'\n' {
        i += 1;
      }
      continue;
    }
    // `/* ... */` block comment with nesting (Rust permits nesting).
    if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
      i += 2;
      let mut depth: u32 = 1;
      while i < bytes.len() && depth > 0 {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
          depth += 1;
          i += 2;
        } else if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
          depth -= 1;
          i += 2;
        } else {
          i += 1;
        }
      }
      continue;
    }
    // Double-quoted string literal: copy verbatim up to and including
    // the matching unescaped `"`. (Substring scans for `File::open`
    // are unaffected — the worker body's string literals never
    // embed that identifier-shaped substring.)
    if b == b'"' {
      out.push('"');
      i += 1;
      while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' && i + 1 < bytes.len() {
          // Escape sequence — copy both bytes verbatim. Escapes are
          // single-byte each (`\\`, `\"`, `\n`, etc. are all ASCII
          // up to and including their lead-in `\`).
          out.push(c as char);
          out.push(bytes[i + 1] as char);
          i += 2;
          continue;
        }
        if c == b'"' {
          out.push('"');
          i += 1;
          break;
        }
        // For non-ASCII bytes inside a string literal, find the
        // char boundary and append the whole char as a slice.
        let ch_end = utf8_char_end(bytes, i);
        out.push_str(&src[i..ch_end]);
        i = ch_end;
      }
      continue;
    }
    // Default arm: copy one whole UTF-8 char.
    let ch_end = utf8_char_end(bytes, i);
    out.push_str(&src[i..ch_end]);
    i = ch_end;
  }
  out
}

/// Returns the byte index of the end of the UTF-8 character starting
/// at `bytes[i]`. Assumes `i < bytes.len()` and `bytes` holds valid
/// UTF-8 (the caller is reading directly from an `&str`).
fn utf8_char_end(bytes: &[u8], i: usize) -> usize {
  let b = bytes[i];
  let width = if b < 0x80 {
    1
  } else if b < 0xc0 {
    // Continuation byte mid-char — should not be reached from a
    // char-start scan; defensively advance by 1.
    1
  } else if b < 0xe0 {
    2
  } else if b < 0xf0 {
    3
  } else {
    4
  };
  (i + width).min(bytes.len())
}
