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
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
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
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
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
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
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
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
}

#[test]
fn resample_rejects_zero_to_rate() {
  let r = resample_linear(&[0.0_f32, 1.0], 16_000, 0);
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
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
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
}

#[test]
fn load_wav_missing_file_returns_backend_error() {
  // Use `temp_wav` (pid-suffixed) so concurrent test processes don't
  // collide on a shared filename — same convention the rest of this
  // module follows (Copilot review #3273868515).
  let path = temp_wav("missing");
  // Make absolutely sure it doesn't exist (a stale file from a prior run
  // would mask the test).
  let _ = fs::remove_file(&path);
  let r = load_audio(&path);
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
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
  // Pre-codex-round-1 regression: silently skipping a decoder
  // IoError/DecodeError would return Ok with fewer samples than the
  // header declared. The post-fix path either (a) fails the decode
  // step with Error::Backend, or (b) reaches the post-loop
  // header_len-vs-out.len mismatch check and fails there.
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
  assert!(
    matches!(r, Err(mlxrs::Error::Backend { .. })),
    "load_wav must reject a truncated WAV; got {r:?}"
  );
  let _ = fs::remove_file(&path);
}

#[test]
fn save_wav_rejects_sample_rate_exceeding_byte_rate_u32_ceiling() {
  // Post-codex-round-1 cap: save_wav rejects sample_rate values whose
  // byte_rate = sample_rate * 2 would wrap u32. Upper bound is
  // u32::MAX / 2 = 2147483647; anything above that must error UPFRONT
  // before any tempfile is created.
  let path = temp_wav("sr_overflow");
  fs::write(&path, b"PRESERVED").unwrap();
  let r = save_wav(&path, &[0.0_f32], u32::MAX);
  assert!(matches!(r, Err(mlxrs::Error::Backend { .. })));
  let stored = fs::read(&path).unwrap();
  assert_eq!(stored, b"PRESERVED");
  let _ = fs::remove_file(&path);
}

#[cfg(unix)]
#[test]
fn save_wav_preserves_existing_destination_mode_bits() {
  // Post-codex-round-3 regression: when save_wav overwrites an
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
  // Post-codex-round-2 regression: load_wav must accept 24-bit mono
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

// -------- New tests: MP3 / FLAC / OGG-Vorbis decode (A6) --------

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
  // Fix 2 (FLAC exact-count): FLAC's STREAMINFO carries an EXACT total
  // sample count, which symphonia surfaces as the track's `num_frames`.
  // `load_audio` now treats FLAC-with-a-declared-total like WAV (an
  // exact-count format) and applies the strict post-decode equality
  // cross-check — so the intact fixture must decode to EXACTLY its
  // declared 2000 samples (not merely a tolerance band). If this drifts
  // off 2000, either the fixture or the exact-count gate changed.
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
  // Fix 2 (FLAC exact-count): a FLAC truncated mid-stream declares (via
  // STREAMINFO) more samples than survive in the truncated byte buffer.
  // Symphonia can hit a clean EOF after the partial frames; the old
  // WAV-only gate would then return `Ok` with missing audio (silent
  // corruption). With FLAC promoted to an exact-count format, the
  // post-decode `out.len() == declared` cross-check (or an earlier decode
  // error) must surface `Error::Backend` instead.
  //
  // We truncate to the first half of the fixture bytes, which keeps the
  // STREAMINFO header (so `num_frames` = 2000 is known) but drops the
  // tail audio frames.
  let path = temp_path("flac_truncated", "flac");
  let cut = FIXTURE_FLAC.len() / 2;
  write_fixture(&path, &FIXTURE_FLAC[..cut]);
  let r = load_audio(&path);
  assert!(
    matches!(r, Err(mlxrs::Error::Backend { .. })),
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
  assert!(
    matches!(r, Err(mlxrs::Error::Backend { .. })),
    "unsupported/garbage input must return Backend error, got {r:?}"
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
      Err(mlxrs::Error::Backend { .. }) => { /* recoverable error: ok */ }
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
