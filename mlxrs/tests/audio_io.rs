//! Happy-path + edge-case tests for `mlxrs::audio::io` (WAV load/save +
//! naive linear resample).
//!
//! NO disk I/O outside `std::env::temp_dir() + process::id()` (matches the
//! existing `tests/io.rs` convention; tempfile is not a workspace dep yet).

#![cfg(feature = "audio")]

use std::{
  fs::{self, File},
  io::Write,
  path::PathBuf,
  process,
};

use mlxrs::audio::io::{load_wav, resample_linear, save_wav};

fn temp_wav(name: &str) -> PathBuf {
  let mut p = std::env::temp_dir();
  p.push(format!("mlxrs_audio_io_{}_{}.wav", process::id(), name));
  p
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
  let (got, sr) = load_wav(&path).unwrap();
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
  let (_, sr) = load_wav(&path).unwrap();
  assert_eq!(sr, 44_100);
  let _ = fs::remove_file(&path);
}

#[test]
fn save_clips_out_of_range_samples() {
  // Samples > 1.0 should be clipped to +1 (which quantizes to 32767 → ≈ +0.99997).
  let path = temp_wav("clip");
  save_wav(&path, &[2.0_f32, -3.0, 0.0], 8_000).unwrap();
  let (got, _) = load_wav(&path).unwrap();
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
  let r = load_wav(&path);
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
  let r = load_wav(&path);
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
  let (got, _) = load_wav(&path).unwrap();
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
  let (got, sr) = load_wav(&path).unwrap();
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
  let r = load_wav(&path);
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
  let (got, sr) = load_wav(&path).unwrap();
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
  let (got, sr) = load_wav(&path).unwrap();
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
