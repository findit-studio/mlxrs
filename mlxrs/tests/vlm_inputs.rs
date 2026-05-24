//! V4 VLM input-assembly tests — `prepare_inputs` branch dispatch +
//! padding-side handling, plus the VLM-side audio/video glue.
//!
//! Reference: `mlx-vlm/mlx_vlm/utils.py` lines 1173–1449
//! (`prepare_inputs`), 852–994 (`read_audio`), 997–1029 (`load_audio`),
//! 1032–1034 (`normalize_audio_features`), 1037–1099 (`load_video`).
#![cfg(feature = "vlm")]

use mlxrs::{
  Array, Error,
  vlm::inputs::{PaddingSide, PrepareInputsOpts, prepare_inputs},
};

// ──────────────────────── prepare_inputs: branch dispatch ────────────────

#[test]
fn prepare_inputs_text_only_no_payloads() {
  // Text-only input (no images/audio/video) → input_ids + attention_mask
  // only; payload Options are None.
  let batch_a = [10_u32, 20, 30];
  let batches: &[&[u32]] = &[&batch_a];
  let out = prepare_inputs(batches, None, None, None, &PrepareInputsOpts::default()).unwrap();
  assert_eq!(out.input_ids.shape(), vec![1, 3]);
  assert_eq!(out.attention_mask.shape(), vec![1, 3]);
  assert!(out.pixel_values.is_none());
  assert!(out.input_features.is_none());
  assert!(out.pixel_values_videos.is_none());
}

#[test]
fn prepare_inputs_image_only_dispatch() {
  // Single batch with an image payload → input_ids/attention_mask are
  // built; pixel_values is passed through.
  let batch_a = [10_u32, 99, 20]; // 99 = image-token placeholder
  let batches: &[&[u32]] = &[&batch_a];
  // Synthetic pixel_values: [1, 3, 4, 4] f32 — caller-supplied (the
  // per-model image processor lives elsewhere).
  let pixels = Array::full::<f32>(&(1usize, 3usize, 4usize, 4usize), 0.5).unwrap();
  let out = prepare_inputs(
    batches,
    Some(pixels),
    None,
    None,
    &PrepareInputsOpts::default(),
  )
  .unwrap();
  assert_eq!(out.input_ids.shape(), vec![1, 3]);
  assert_eq!(out.attention_mask.shape(), vec![1, 3]);
  let pv = out.pixel_values.expect("pixel_values present");
  assert_eq!(pv.shape(), vec![1, 3, 4, 4]);
  assert!(out.input_features.is_none());
  assert!(out.pixel_values_videos.is_none());
}

#[test]
fn prepare_inputs_audio_only_dispatch() {
  // Audio-only input → input_features is passed through; pixel_values
  // is None.
  let batch_a = [10_u32, 88, 20]; // 88 = audio-token placeholder
  let batches: &[&[u32]] = &[&batch_a];
  // Synthetic input_features: [1, 80, 100] (n_mels=80, time=100).
  let features = Array::full::<f32>(&(1usize, 80usize, 100usize), 0.25).unwrap();
  let out = prepare_inputs(
    batches,
    None,
    Some(features),
    None,
    &PrepareInputsOpts::default(),
  )
  .unwrap();
  let f = out.input_features.expect("input_features present");
  assert_eq!(f.shape(), vec![1, 80, 100]);
  assert!(out.pixel_values.is_none());
  assert!(out.pixel_values_videos.is_none());
}

#[test]
fn prepare_inputs_combined_image_text_audio() {
  // Combined multimodal: text + image + audio → all branches populated.
  let batch_a = [10_u32, 99, 88, 20];
  let batches: &[&[u32]] = &[&batch_a];
  let pixels = Array::full::<f32>(&(1usize, 3usize, 4usize, 4usize), 0.5).unwrap();
  let features = Array::full::<f32>(&(1usize, 80usize, 50usize), 0.25).unwrap();
  let out = prepare_inputs(
    batches,
    Some(pixels),
    Some(features),
    None,
    &PrepareInputsOpts::default(),
  )
  .unwrap();
  assert_eq!(out.input_ids.shape(), vec![1, 4]);
  assert!(out.pixel_values.is_some());
  assert!(out.input_features.is_some());
  assert!(out.pixel_values_videos.is_none());
}

#[test]
fn prepare_inputs_video_dispatch() {
  // Video-only payload pass-through.
  let batch_a = [10_u32, 77, 20]; // 77 = video-token placeholder
  let batches: &[&[u32]] = &[&batch_a];
  let frames = Array::full::<f32>(&(8usize, 224usize, 224usize, 3usize), 0.5).unwrap();
  let out = prepare_inputs(
    batches,
    None,
    None,
    Some(frames),
    &PrepareInputsOpts::default(),
  )
  .unwrap();
  let v = out
    .pixel_values_videos
    .expect("pixel_values_videos present");
  assert_eq!(v.shape(), vec![8, 224, 224, 3]);
  assert!(out.pixel_values.is_none());
  assert!(out.input_features.is_none());
}

// ──────────────────────── prepare_inputs: padding-side ───────────────────

#[test]
fn prepare_inputs_padding_side_left_default() {
  // Default is LEFT-pad: shorter sequences get pad tokens BEFORE the
  // content. Python line 1183 default = "left".
  let a = [10_u32, 20]; // len 2
  let b = [30_u32, 40, 50, 60]; // len 4 (max)
  let batches: &[&[u32]] = &[&a, &b];
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Left,
    attention_mask: None,
  };
  let mut out = prepare_inputs(batches, None, None, None, &opts).unwrap();
  assert_eq!(out.input_ids.shape(), vec![2, 4]);
  let ids = out.input_ids.to_vec::<i32>().unwrap();
  // Row 0 (left-padded): [0, 0, 10, 20]
  assert_eq!(&ids[0..4], &[0, 0, 10, 20]);
  // Row 1: [30, 40, 50, 60] (no padding needed)
  assert_eq!(&ids[4..8], &[30, 40, 50, 60]);
  // Attention mask: false at left-pads.
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  assert_eq!(&mask[0..4], &[false, false, true, true]);
  assert_eq!(&mask[4..8], &[true, true, true, true]);
}

#[test]
fn prepare_inputs_padding_side_right_vs_left() {
  // RIGHT-pad: shorter sequences get pad tokens AFTER the content.
  let a = [10_u32, 20]; // len 2
  let b = [30_u32, 40, 50, 60]; // len 4 (max)
  let batches: &[&[u32]] = &[&a, &b];
  let opts = PrepareInputsOpts {
    pad_token_id: 7,
    padding: true,
    padding_side: PaddingSide::Right,
    attention_mask: None,
  };
  let mut out = prepare_inputs(batches, None, None, None, &opts).unwrap();
  let ids = out.input_ids.to_vec::<i32>().unwrap();
  // Row 0 (right-padded): [10, 20, 7, 7]
  assert_eq!(&ids[0..4], &[10, 20, 7, 7]);
  // Row 1: unchanged.
  assert_eq!(&ids[4..8], &[30, 40, 50, 60]);
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  assert_eq!(&mask[0..4], &[true, true, false, false]);
}

#[test]
fn prepare_inputs_padding_disabled_requires_uniform_length() {
  // padding=false + varying lengths → error.
  let a = [10_u32, 20];
  let b = [30_u32, 40, 50];
  let batches: &[&[u32]] = &[&a, &b];
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: false,
    padding_side: PaddingSide::Left,
    attention_mask: None,
  };
  let err = prepare_inputs(batches, None, None, None, &opts).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("padding=false"), "unexpected error: {msg}");
}

#[test]
fn prepare_inputs_empty_batches_errors() {
  let batches: &[&[u32]] = &[];
  let err = prepare_inputs(batches, None, None, None, &PrepareInputsOpts::default()).unwrap_err();
  let msg = format!("{err}");
  assert!(msg.contains("empty"), "unexpected error: {msg}");
}

#[test]
fn prepare_inputs_uniform_no_padding_needed() {
  // padding=true with uniform lengths → no padding applied.
  let a = [10_u32, 20, 30];
  let b = [40_u32, 50, 60];
  let batches: &[&[u32]] = &[&a, &b];
  let mut out = prepare_inputs(batches, None, None, None, &PrepareInputsOpts::default()).unwrap();
  assert_eq!(out.input_ids.shape(), vec![2, 3]);
  let ids = out.input_ids.to_vec::<i32>().unwrap();
  assert_eq!(ids, vec![10, 20, 30, 40, 50, 60]);
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  assert!(mask.iter().all(|&b| b));
}

// ──────────────────────── V4 R1: attention_mask threading ─────────────────
//
// Finding 3 regressions — caller-supplied attention_mask must override
// the internal "every-token-true" computation so pre-padded uniform
// batches survive into the output with their padding positions marked
// `false`.

#[test]
fn prepare_inputs_caller_supplied_attention_mask_overrides_default() {
  // Pre-padded uniform batches (length 4) with the LAST 2 positions of
  // batch a being pre-pads (caller's mask: [true,true,false,false]).
  // Pre-fix: the internal step marked every position true. Post-fix:
  // the caller's mask is threaded through.
  let a = [10_u32, 20, 0, 0]; // last 2 = caller's pre-pads
  let b = [30_u32, 40, 50, 60]; // all real tokens
  let batches: &[&[u32]] = &[&a, &b];
  let caller_mask = vec![vec![true, true, false, false], vec![true, true, true, true]];
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Left,
    attention_mask: Some(caller_mask),
  };
  let mut out = prepare_inputs(batches, None, None, None, &opts).unwrap();
  assert_eq!(out.attention_mask.shape(), vec![2, 4]);
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  // Row 0 (uniform-length → no extra padding step) → caller's mask
  // directly.
  assert_eq!(&mask[0..4], &[true, true, false, false]);
  // Row 1 → caller's mask directly (all true).
  assert_eq!(&mask[4..8], &[true, true, true, true]);
}

#[test]
fn prepare_inputs_caller_mask_left_pads_with_false() {
  // With padding_side=Left and varying-length batches, the caller's
  // mask is applied AT the token positions (post-pad), and the leading
  // pad positions are marked `false` per the contract.
  let a = [10_u32, 20]; // len 2 — caller's mask: [true, false] (the second token is a pre-pad they want masked)
  let b = [30_u32, 40, 50, 60]; // len 4
  let batches: &[&[u32]] = &[&a, &b];
  let caller_mask = vec![vec![true, false], vec![true, true, true, true]];
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Left,
    attention_mask: Some(caller_mask),
  };
  let mut out = prepare_inputs(batches, None, None, None, &opts).unwrap();
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  // Row 0 left-padded to length 4:
  //   leading pad position (1) → false
  //   leading pad position (2) → false
  //   caller's mask[0] = true
  //   caller's mask[1] = false (caller marked this token as a pre-pad)
  assert_eq!(&mask[0..4], &[false, false, true, false]);
  assert_eq!(&mask[4..8], &[true, true, true, true]);
}

#[test]
fn prepare_inputs_caller_mask_right_pads_with_false() {
  // padding_side=Right symmetric counterpart.
  let a = [10_u32, 20]; // len 2
  let b = [30_u32, 40, 50, 60]; // len 4
  let batches: &[&[u32]] = &[&a, &b];
  let caller_mask = vec![vec![true, false], vec![true, true, true, true]];
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Right,
    attention_mask: Some(caller_mask),
  };
  let mut out = prepare_inputs(batches, None, None, None, &opts).unwrap();
  let mask = out.attention_mask.to_vec::<bool>().unwrap();
  // Row 0 right-padded: caller's mask [true, false] then 2 trailing
  // pad positions (false).
  assert_eq!(&mask[0..4], &[true, false, false, false]);
  assert_eq!(&mask[4..8], &[true, true, true, true]);
}

#[test]
fn prepare_inputs_caller_mask_dimension_mismatch_errors() {
  // Outer-dim mismatch: caller supplies 1 mask, batch has 2 entries.
  let a = [10_u32, 20];
  let b = [30_u32, 40];
  let batches: &[&[u32]] = &[&a, &b];
  let bad_mask = vec![vec![true, true]]; // outer len 1 != 2
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Left,
    attention_mask: Some(bad_mask),
  };
  let err = prepare_inputs(batches, None, None, None, &opts).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "expected ShapeMismatch, got: {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("attention_mask outer length") && msg.contains("1") && msg.contains("2"),
    "expected outer-length mismatch text, got: {msg}"
  );
}

#[test]
fn prepare_inputs_caller_mask_inner_dimension_mismatch_errors() {
  // Inner-dim mismatch: caller's mask[1] has the wrong length.
  let a = [10_u32, 20];
  let b = [30_u32, 40, 50];
  let batches: &[&[u32]] = &[&a, &b];
  let bad_mask = vec![vec![true, true], vec![true, true]]; // mask[1] len 2 != 3
  let opts = PrepareInputsOpts {
    pad_token_id: 0,
    padding: true,
    padding_side: PaddingSide::Left,
    attention_mask: Some(bad_mask),
  };
  let err = prepare_inputs(batches, None, None, None, &opts).unwrap_err();
  assert!(
    matches!(err, Error::ShapeMismatch { .. }),
    "expected ShapeMismatch, got: {err:?}"
  );
  let msg = format!("{err}");
  assert!(
    msg.contains("attention_mask[1]"),
    "expected attention_mask[1] index in error message, got: {msg}"
  );
}

// ──────────────────────── audio/video glue (cfg-gated) ───────────────────

/// `load_video` is the model-agnostic preprocessing composer; it
/// requires only the `vlm` feature (it wraps `vlm/video::process_frames`).
#[test]
fn load_video_wraps_vlm_video() {
  use mlxrs::vlm::{
    image::{ColorOrder, ImageProcessorConfig, ResizeFilter},
    inputs::load_video,
  };

  // Synth 2 small RGB frames (4x4) — caller-decoded.
  let mk_frame = || {
    let buf = ::image::ImageBuffer::<::image::Rgb<u8>, Vec<u8>>::from_fn(4, 4, |x, y| {
      ::image::Rgb([x as u8 * 50, y as u8 * 50, 128])
    });
    ::image::DynamicImage::ImageRgb8(buf)
  };
  let frames = vec![mk_frame(), mk_frame()];

  // Minimal ImageProcessorConfig — keep input frame size, just rescale.
  let cfg = ImageProcessorConfig {
    size: (4, 4),
    mean: [0.0, 0.0, 0.0],
    std: [1.0, 1.0, 1.0],
    rescale_factor: 1.0 / 255.0,
    do_resize: false,
    do_rescale: true,
    do_normalize: false,
    resample: ResizeFilter::Bilinear,
    color_order: ColorOrder::Rgb,
    ..ImageProcessorConfig::default()
  };
  let out = load_video(&frames, &cfg).unwrap();
  // stack of 2 frames, 4x4 RGB, channel-last → [2, 4, 4, 3]
  assert_eq!(out.shape(), vec![2, 4, 4, 3]);
}

// Audio glue tests gated on `feature = "audio"` (which is NOT enabled by
// `--features vlm`). They live behind the additional gate so the
// vlm-only test build excludes them; a `cargo test -p mlxrs --features
// "vlm audio"` run picks them up.

#[cfg(feature = "audio")]
mod audio_glue {
  use mlxrs::{
    Array,
    vlm::inputs::{load_audio_vlm, normalize_audio_features, read_audio},
  };
  use std::io::Write;

  /// Helper: write a synthetic 16-bit PCM mono WAV file to a temp path.
  /// 1000 Hz tone at 16000 Hz sample rate, 0.1 sec duration → 1600 samples.
  fn write_wav(path: &std::path::Path, samples: &[i16], sr: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    // RIFF/WAVE header for 16-bit mono PCM.
    let data_bytes = (samples.len() * 2) as u32;
    let chunk_size = 36 + data_bytes;
    f.write_all(b"RIFF").unwrap();
    f.write_all(&chunk_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap(); // channels=1
    f.write_all(&sr.to_le_bytes()).unwrap();
    f.write_all(&(sr * 2).to_le_bytes()).unwrap(); // byte_rate
    f.write_all(&2u16.to_le_bytes()).unwrap(); // block_align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits_per_sample
    f.write_all(b"data").unwrap();
    f.write_all(&data_bytes.to_le_bytes()).unwrap();
    for &s in samples {
      f.write_all(&s.to_le_bytes()).unwrap();
    }
  }

  #[test]
  fn read_audio_wraps_load_audio() {
    // Synth WAV → assert read_audio returns (samples, sr) matching
    // crate::audio::io::load_audio.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("mlxrs_v4_read_audio_{}.wav", std::process::id()));
    let samples_i16: Vec<i16> = (0..1600)
      .map(|i| ((i as f32 * 0.1).sin() * 8000.0) as i16)
      .collect();
    write_wav(&path, &samples_i16, 16000);

    let (samples, sr) = read_audio(&path).unwrap();
    assert_eq!(sr, 16000);
    assert_eq!(samples.len(), 1600);
    // Spot check first sample is finite.
    assert!(samples[0].is_finite());

    // Compare to direct load_audio call.
    let (samples2, sr2) = mlxrs::audio::io::load_audio(&path).unwrap();
    assert_eq!(sr, sr2);
    assert_eq!(samples, samples2);

    let _ = std::fs::remove_file(&path);
  }

  #[test]
  fn load_audio_vlm_no_resample_when_sr_matches() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
      "mlxrs_v4_load_audio_same_{}.wav",
      std::process::id()
    ));
    let samples_i16: Vec<i16> = (0..1600)
      .map(|i| ((i as f32 * 0.1).sin() * 8000.0) as i16)
      .collect();
    write_wav(&path, &samples_i16, 16000);

    let out = load_audio_vlm(&path, 16000).unwrap();
    assert_eq!(out.len(), 1600); // identical length when SR matches

    let _ = std::fs::remove_file(&path);
  }

  #[test]
  fn load_audio_vlm_resamples_when_sr_differs() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
      "mlxrs_v4_load_audio_resamp_{}.wav",
      std::process::id()
    ));
    // 0.1 sec at 16000 Hz = 1600 samples
    let samples_i16: Vec<i16> = (0..1600)
      .map(|i| ((i as f32 * 0.1).sin() * 8000.0) as i16)
      .collect();
    write_wav(&path, &samples_i16, 16000);

    // Resample 16000 → 8000 should yield ~800 samples (half).
    let out = load_audio_vlm(&path, 8000).unwrap();
    assert!(
      (out.len() as i64 - 800).abs() <= 2,
      "expected ~800 samples after downsampling, got {}",
      out.len()
    );

    let _ = std::fs::remove_file(&path);
  }

  #[test]
  fn normalize_audio_features_matches_python_reference() {
    // python: (features - mx.mean(features)) / (mx.std(features) + 1e-6).
    // Use a known table: features = [[1, 2, 3], [4, 5, 6]] f32.
    // mean = 21/6 = 3.5; std = sqrt(((1-3.5)^2 + ... + (6-3.5)^2) / 6)
    //      = sqrt((6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6)
    //      = sqrt(17.5 / 6) = sqrt(2.91667) ≈ 1.707825127659933
    // expected[0,0] = (1 - 3.5) / (1.707825... + 1e-6) ≈ -1.4638475
    let features =
      Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3usize)).unwrap();
    let mut normalized = normalize_audio_features(&features).unwrap();
    assert_eq!(normalized.shape(), vec![2, 3]);
    let vals = normalized.to_vec::<f32>().unwrap();
    // mean = 3.5
    let mean = 3.5_f32;
    // std with ddof=0
    let sq_diff: f32 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
      .iter()
      .map(|x| (x - mean).powi(2))
      .sum();
    let std = (sq_diff / 6.0).sqrt();
    let denom = std + 1e-6_f32;
    for (i, x) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().enumerate() {
      let expected = (x - mean) / denom;
      assert!(
        (vals[i] - expected).abs() < 1e-5,
        "vals[{i}]={} expected≈{expected}",
        vals[i]
      );
    }
  }
}
