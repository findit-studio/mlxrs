//! V1 VLM video-preprocessing math tests.
//!
//! Reference basis:
//! - python `mlx-vlm/mlx_vlm/video_generate.py` (`round_by_factor`,
//!   `ceil_by_factor`, `floor_by_factor`, `smart_resize`, `smart_nframes`,
//!   the `np.linspace(0, total-1, n).round()` frame-index pick, and the
//!   per-frame resize+`np.stack` body of `fetch_video`).
//! - swift `MLXVLM/Models/QwenVL.swift` `QwenVL.targetSize` (the swift
//!   mirror of `smart_resize`).
//!
//! All expectations are HAND-TRACED in the comments above each assert so a
//! reviewer can re-derive them from the python math without running the
//! reference.

#![cfg(feature = "vlm")]

use mlxrs::vlm::{
  image::{ColorOrder, ImageProcessorConfig, ResizeFilter},
  video::{
    FrameSampling, MAX_PIXELS, MIN_PIXELS, ceil_by_factor, floor_by_factor, process_frames,
    round_by_factor, sample_frame_indices, smart_nframes, smart_resize,
  },
};

// ---------- *_by_factor (banker's rounding parity) ----------

#[test]
fn round_by_factor_banker_rounding_edges() {
  // python `round_by_factor(n, f) = round(n / f) * f` with round-HALF-TO-
  // EVEN (python built-in `round`). Half-quotient cases must pick the even
  // neighbor, NOT round-away-from-zero. (Now `Result`-returning for
  // overflow safety; sane inputs always `Ok`.)
  assert_eq!(round_by_factor(14, 28).unwrap(), 0); // 0.5  -> 0 (even)
  assert_eq!(round_by_factor(42, 28).unwrap(), 56); // 1.5 -> 2 (even) -> 56
  assert_eq!(round_by_factor(70, 28).unwrap(), 56); // 2.5 -> 2 (even) -> 56
  assert_eq!(round_by_factor(98, 28).unwrap(), 112); // 3.5 -> 4 (even) -> 112
  assert_eq!(round_by_factor(800, 28).unwrap(), 812); // 28.571 -> 29 -> 812
  assert_eq!(round_by_factor(1280, 28).unwrap(), 1288); // 45.714 -> 46 -> 1288
}

#[test]
fn ceil_floor_by_factor_edges() {
  // ceil_by_factor(n, f) = ceil(n / f) * f ; floor_by_factor = floor(...) * f
  assert_eq!(ceil_by_factor(10, 2).unwrap(), 10); // ceil(5.0) * 2
  assert_eq!(ceil_by_factor(9, 2).unwrap(), 10); // ceil(4.5) = 5 -> 10
  assert_eq!(floor_by_factor(9, 2).unwrap(), 8); // floor(4.5) = 4 -> 8
  assert_eq!(floor_by_factor(10, 2).unwrap(), 10); // floor(5.0) -> 10
  assert_eq!(ceil_by_factor(4, 2).unwrap(), 4); // ceil(2.0) -> 4
}

// ---------- *_by_factor / smart_resize / smart_nframes overflow safety ----------

#[test]
fn factor_helpers_reject_overflow_instead_of_panic_or_wrap() {
  // Regression for the Codex "factor rounding can overflow before
  // validation" finding. `round/ceil/floor_by_factor(i64::MAX, 28)` would
  // compute `quotient * 28`, which overflows i64: debug builds PANIC,
  // release WRAPS negative. All three must instead return a recoverable
  // Err (NOT a panic, NOT a wrapped/garbage value).
  assert!(
    round_by_factor(i64::MAX, 28).is_err(),
    "round_by_factor near i64::MAX must Err, not panic/wrap"
  );
  assert!(
    ceil_by_factor(i64::MAX, 28).is_err(),
    "ceil_by_factor near i64::MAX must Err, not panic/wrap"
  );
  assert!(
    floor_by_factor(i64::MAX, 28).is_err(),
    "floor_by_factor near i64::MAX must Err, not panic/wrap"
  );
  // An oversized factor against a large-but-not-max number also overflows
  // the product (quotient ~= i64::MAX/2, times the same factor).
  assert!(
    round_by_factor(i64::MAX / 2, i64::MAX).is_err(),
    "oversized factor must Err on product overflow"
  );
  // Sanity: a normal input still rounds correctly through the new path.
  assert_eq!(round_by_factor(800, 28).unwrap(), 812);
}

#[test]
fn smart_resize_rejects_overflow_dimension() {
  // The Codex finding's exact silent-corruption scenario: a positive
  // near-i64::MAX height/width with factor=28. Before the fix this
  // overflowed inside `round_by_factor` (panic debug / wrap release), and
  // on release the wrapped-negative bar was promoted by `factor.max(..)` to
  // a small valid-looking size so `smart_resize` returned a bogus small
  // (h, w) instead of erroring. It must now Err recoverably (NOT panic, NOT
  // a small wrapped size). Square dims keep aspect ratio = 1 so the failure
  // is the overflow, not the ratio guard.
  let r = smart_resize(i64::MAX, i64::MAX, 28, MIN_PIXELS, MAX_PIXELS);
  assert!(
    r.is_err(),
    "near-i64::MAX dims must Err on overflow, got {r:?}"
  );
  // Asymmetric near-i64::MAX dims (ratio still ~1 < MAX_RATIO so the failure
  // is the factor-product overflow, not the ratio guard) must also Err
  // rather than silently corrupt. Both bars overflow `round_by_factor`; the
  // first `?` is enough — we only assert the recoverable Err.
  let r2 = smart_resize(i64::MAX - 56, i64::MAX, 28, MIN_PIXELS, MAX_PIXELS);
  assert!(r2.is_err(), "overflowing dims must Err, got {r2:?}");
}

#[test]
fn smart_nframes_rejects_overflow() {
  // Fixed{i64::MAX}: round_by_factor(i64::MAX, FRAME_FACTOR=2) overflows the
  // `quotient * 2` product -> must Err (not panic/wrap into a small count
  // that could then pass the [FRAME_FACTOR, total_frames] check).
  let r = smart_nframes(FrameSampling::Fixed { nframes: i64::MAX }, 100, 30.0);
  assert!(
    r.is_err(),
    "Fixed near-i64::MAX nframes must Err, got {r:?}"
  );
  // Fps with a near-i64::MAX min_frames: ceil_by_factor(min_frames, 2)
  // overflows -> Err (caught before the clamp / final range check).
  let r2 = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: i64::MAX,
      max_frames: None,
    },
    100,
    30.0,
  );
  assert!(
    r2.is_err(),
    "Fps near-i64::MAX min_frames must Err, got {r2:?}"
  );
  // Fps with a near-i64::MAX explicit max_frames: floor_by_factor(max, 2)
  // overflows -> Err.
  let r3 = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: Some(i64::MAX),
    },
    100,
    30.0,
  );
  assert!(
    r3.is_err(),
    "Fps near-i64::MAX max_frames must Err, got {r3:?}"
  );
}

// ---------- smart_resize ----------

#[test]
fn smart_resize_within_budget_rounds_to_factor() {
  // height=800, width=1280, factor=28, default min/max pixels.
  //   h_bar = max(28, round_by_factor(800, 28))  = max(28, 812)  = 812
  //   w_bar = max(28, round_by_factor(1280, 28)) = max(28, 1288) = 1288
  //   812 * 1288 = 1_045_856  (in [MIN_PIXELS, MAX_PIXELS]) -> no scaling
  let (h, w) = smart_resize(800, 1280, 28, MIN_PIXELS, MAX_PIXELS).unwrap();
  assert_eq!((h, w), (812, 1288));
}

#[test]
fn smart_resize_scales_down_for_max_pixels() {
  // height=420, width=420, factor=28, min=MIN_PIXELS, max=50_000.
  //   h_bar = w_bar = round_by_factor(420, 28) = round(15.0) * 28 = 420
  //   420 * 420 = 176_400 > 50_000  -> scale DOWN
  //   beta = sqrt(176_400 / 50_000) = sqrt(3.528) = 1.878297...
  //   height / beta = 420 / 1.878297 = 223.610...
  //   floor_by_factor(223.610, 28) = floor(223.610 / 28) * 28
  //                                = floor(7.986) * 28 = 7 * 28 = 196
  let (h, w) = smart_resize(420, 420, 28, MIN_PIXELS, 50_000).unwrap();
  assert_eq!((h, w), (196, 196));
  assert!(h * w <= 50_000, "downscaled pixel count within max_pixels");
}

#[test]
fn smart_resize_scales_up_for_min_pixels() {
  // height=28, width=28, factor=28, min=MIN_PIXELS(=3136), max=MAX_PIXELS.
  //   h_bar = w_bar = max(28, round_by_factor(28, 28)) = 28
  //   28 * 28 = 784 < 3136  -> scale UP
  //   beta = sqrt(3136 / (28*28)) = sqrt(4.0) = 2.0
  //   height * beta = 56 ; ceil_by_factor(56, 28) = ceil(2.0) * 28 = 56
  let (h, w) = smart_resize(28, 28, 28, MIN_PIXELS, MAX_PIXELS).unwrap();
  assert_eq!((h, w), (56, 56));
  assert_eq!(h * w, MIN_PIXELS, "upscaled to exactly min_pixels here");
}

#[test]
fn smart_resize_factor_floor_via_max_guard() {
  // height=width=14, factor=28: round_by_factor(14, 28) = round(0.5) = 0
  // (banker's), so the `max(factor, 0)` guard pins both bars to 28.
  //   28 * 28 = 784 < MIN_PIXELS -> scale up
  //   beta = sqrt(3136 / 196) = sqrt(16) = 4.0 ; 14 * 4 = 56 -> (56, 56)
  let (h, w) = smart_resize(14, 14, 28, MIN_PIXELS, MAX_PIXELS).unwrap();
  assert_eq!((h, w), (56, 56));
}

#[test]
fn smart_resize_rejects_extreme_aspect_ratio() {
  // max/min = 1000/1 = 1000 > MAX_RATIO(200) -> Err.
  assert!(smart_resize(1, 1000, 28, MIN_PIXELS, MAX_PIXELS).is_err());
}

#[test]
fn smart_resize_rejects_nonpositive_and_bad_budget() {
  assert!(smart_resize(0, 100, 28, MIN_PIXELS, MAX_PIXELS).is_err());
  assert!(smart_resize(100, 0, 28, MIN_PIXELS, MAX_PIXELS).is_err());
  assert!(smart_resize(100, 100, 0, MIN_PIXELS, MAX_PIXELS).is_err());
  // min_pixels > max_pixels is an empty interval.
  assert!(smart_resize(100, 100, 28, 5000, 4000).is_err());
}

#[test]
fn smart_resize_rejects_impossible_budget() {
  // factor=28 -> the smallest legal output is a 28x28 = 784-pixel square.
  // A max_pixels of 1 cannot contain it, so there is no positive
  // factor-aligned solution. The python reference silently scales to
  // (0, 0); we reject. (Regression for the zero-dim Codex finding.)
  assert!(
    smart_resize(28, 28, 28, 1, 1).is_err(),
    "max_pixels=1 < 28*28 -> no positive factor-aligned size -> Err"
  );
}

#[test]
fn smart_resize_rejects_extreme_aspect_floor_to_zero() {
  // An extreme (but < MAX_RATIO) aspect with a tight max_pixels can pass the
  // `max_pixels >= factor*factor` guard yet still floor the SHORT side to 0
  // in the scale-down branch:
  //   height=537, width=4962, factor=28, max=1646  (>= 784, ratio 9.24 < 200)
  //   h_bar=max(28, round_by_factor(537,28)=532)=532,
  //   w_bar=max(28, round_by_factor(4962,28)=4956)=4956
  //   532*4956 = 2_636_592 > 1646 -> scale down
  //   beta = sqrt(537*4962 / 1646) = sqrt(1618.7...) = 40.23...
  //   floor_by_factor(537/40.23 = 13.34, 28) = floor(0.476)*28 = 0  -> zero dim
  // The reference would emit (0, 112); we reject the non-positive dimension.
  assert!(
    smart_resize(537, 4962, 28, 1, 1646).is_err(),
    "scale-down floors the short side to 0 -> Err, not a zero dim"
  );
}

#[test]
fn smart_resize_narrow_but_possible_budget_succeeds() {
  // A budget pinned to exactly the minimal factor-aligned cell still has a
  // solution and must succeed (not be over-rejected by the new guards).
  //   height=width=28, factor=28, min=max=784.
  //   h_bar=w_bar=max(28, round_by_factor(28,28)=28)=28 ; 28*28 = 784.
  //   784 is within [784, 784] and not > max -> no scaling -> (28, 28).
  let (h, w) = smart_resize(28, 28, 28, 784, 784).unwrap();
  assert_eq!((h, w), (28, 28));
  assert_eq!(h * w, 784, "exactly fills the single-cell budget");

  // A downscale into the same single-cell ceiling is also possible:
  //   height=width=420, factor=28, min=1, max=784.
  //   h_bar=w_bar=420 ; 420*420=176_400 > 784 -> scale down
  //   beta = sqrt(176_400 / 784) = sqrt(225) = 15.0
  //   floor_by_factor(420/15 = 28.0, 28) = floor(1.0)*28 = 28 -> (28, 28)
  let (h2, w2) = smart_resize(420, 420, 28, 1, 784).unwrap();
  assert_eq!((h2, w2), (28, 28));
}

#[test]
fn smart_resize_keeps_positive_result_just_outside_band() {
  // Faithfulness guard: the reference's floor_by_factor/ceil_by_factor do
  // NOT re-clamp, so a positive output whose area lands one factor-step
  // outside [min, max] must be KEPT (matching python), not rejected.
  //   height=420, width=420, factor=28, min=MIN_PIXELS, max=50_000.
  //   -> (196, 196), area 38_416 (already covered above) is within budget;
  //   here we assert a scale-up overshoot is preserved:
  //   height=width=28, factor=28, min=3137 (one over 4*28*28=3136), max=MAX.
  //   28*28=784 < 3137 -> scale up
  //   beta = sqrt(3137 / 784) = 2.0003... ; 28*beta = 56.009...
  //   ceil_by_factor(56.009, 28) = ceil(2.0003)*28 = 3*28 = 84 -> (84, 84)
  //   84*84 = 7056 (> the 3137 floor, well within MAX) -> kept.
  let (h, w) = smart_resize(28, 28, 28, 3137, MAX_PIXELS).unwrap();
  assert_eq!((h, w), (84, 84));
  assert!(
    h > 0 && w > 0,
    "positive size preserved, not re-clamped to error"
  );
}

// ---------- smart_nframes ----------

#[test]
fn smart_nframes_fixed_rounds_to_frame_factor() {
  // Fixed{7}: round_by_factor(7, FRAME_FACTOR=2) = round(3.5) = 4 -> 8.
  let n = smart_nframes(FrameSampling::Fixed { nframes: 7 }, 100, 30.0).unwrap();
  assert_eq!(n, 8);
}

#[test]
fn smart_nframes_fps_default_path() {
  // Fps{fps=2, min=4, max=None}, total=100, video_fps=30.
  //   min_frames = ceil_by_factor(4, 2) = 4
  //   max_frames = floor_by_factor(min(768, 100), 2) = 100
  //   raw = 100 / 30 * 2 = 6.6667
  //   clamp: max(6.6667, 4)=6.6667; min(_,100)=6.6667; min(_,100)=6.6667
  //   floor_by_factor(floor(6.6667)=6, 2) = 6
  let n = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: None,
    },
    100,
    30.0,
  )
  .unwrap();
  assert_eq!(n, 6);
}

#[test]
fn smart_nframes_fps_clamps_to_min() {
  // total=10, video_fps=30, fps=2 -> raw = 0.6667, clamped UP to min_frames.
  //   min_frames = ceil_by_factor(4, 2) = 4 ; result = 4
  let n = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: None,
    },
    10,
    30.0,
  )
  .unwrap();
  assert_eq!(n, 4);
}

#[test]
fn smart_nframes_fps_clamps_to_max_default() {
  // total=1000, video_fps=1, fps=2 -> raw = 2000.
  //   max_frames = floor_by_factor(min(768, 1000), 2) = 768
  //   clamp: max(2000,4)=2000; min(2000,768)=768; min(768,1000)=768 -> 768
  let n = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: None,
    },
    1000,
    1.0,
  )
  .unwrap();
  assert_eq!(n, 768);
}

#[test]
fn smart_nframes_fps_clamps_to_total() {
  // total=10, video_fps=1, fps=2 -> raw = 20 > total.
  //   max_frames = floor_by_factor(min(768,10),2)=10
  //   clamp: max(20,4)=20; min(20,10)=10; min(10,10)=10 -> 10 (== total)
  let n = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: None,
    },
    10,
    1.0,
  )
  .unwrap();
  assert_eq!(n, 10);
}

#[test]
fn smart_nframes_fps_custom_max_frames() {
  // Fps{fps=2, min=4, max=Some(6)}, total=100, video_fps=10.
  //   raw = 100 / 10 * 2 = 20 ; max_frames = floor_by_factor(6,2)=6
  //   clamp: max(20,4)=20; min(20,6)=6; min(6,100)=6 -> 6
  let n = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: Some(6),
    },
    100,
    10.0,
  )
  .unwrap();
  assert_eq!(n, 6);
}

#[test]
fn smart_nframes_default_sampling_matches_explicit() {
  // FrameSampling::default() == Fps{FPS, FPS_MIN_FRAMES, None}.
  let d = smart_nframes(FrameSampling::default(), 100, 30.0).unwrap();
  let e = smart_nframes(
    FrameSampling::Fps {
      fps: 2.0,
      min_frames: 4,
      max_frames: None,
    },
    100,
    30.0,
  )
  .unwrap();
  assert_eq!(d, e);
  assert_eq!(d, 6);
}

#[test]
fn smart_nframes_rejects_bad_inputs() {
  assert!(smart_nframes(FrameSampling::default(), 0, 30.0).is_err()); // total<=0
  assert!(
    smart_nframes(
      FrameSampling::Fps {
        fps: 2.0,
        min_frames: 4,
        max_frames: None
      },
      100,
      0.0
    )
    .is_err()
  ); // video_fps<=0
  // Fixed{1} -> round_by_factor(1,2)=round(0.5)=0 < FRAME_FACTOR -> Err.
  assert!(smart_nframes(FrameSampling::Fixed { nframes: 1 }, 100, 30.0).is_err());
}

// ---------- sample_frame_indices ----------

#[test]
fn sample_frame_indices_linspace_round_even() {
  // linspace(0, 9, 5) = [0, 2.25, 4.5, 6.75, 9]
  //   round-half-to-even -> [0, 2, 4 (4.5->even), 7 (6.75->7), 9]
  assert_eq!(sample_frame_indices(10, 5).unwrap(), vec![0, 2, 4, 7, 9]);
}

#[test]
fn sample_frame_indices_more_cases() {
  // linspace(0, 7, 4) = [0, 2.333, 4.667, 7] -> [0, 2, 5, 7]
  assert_eq!(sample_frame_indices(8, 4).unwrap(), vec![0, 2, 5, 7]);
  // num=1 -> [start] = [0]
  assert_eq!(sample_frame_indices(10, 1).unwrap(), vec![0]);
  // linspace(0, 3, 2) = [0, 3]
  assert_eq!(sample_frame_indices(4, 2).unwrap(), vec![0, 3]);
  // Endpoints are exactly [0, total-1].
  let idx = sample_frame_indices(100, 8).unwrap();
  assert_eq!(idx.first(), Some(&0));
  assert_eq!(idx.last(), Some(&99));
  assert_eq!(idx.len(), 8);
}

#[test]
fn sample_frame_indices_numpy_op_order_26_23() {
  // Regression for the float-tie divergence between numpy's
  // `linspace(0, total-1, n)` and the fused `(total-1)*i/(n-1)` form.
  //
  // numpy computes `step = (total-1)/(n-1)` FIRST, then point `i = step*i`,
  // then forces the last sample to `total-1` (endpoint=True). For
  // total=26, n=23 the midpoint i=11 is `12.500000000000002` the numpy way
  // (banker's-rounds UP to 13), but EXACTLY `12.5` the fused way
  // (banker's-rounds DOWN to 12) — a silently wrong frame.
  //
  // Expected vector is the full
  //   np.linspace(0, 25, 23).round().astype(int)
  // computed with numpy (raw midpoint 12.500000000000002 -> 13):
  let expected: Vec<i64> = vec![
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25,
  ];
  let got = sample_frame_indices(26, 23).unwrap();
  assert_eq!(got, expected, "must match np.linspace operation order");
  // The whole point: index 11 is 13 (numpy), NOT 12 (fused form).
  assert_eq!(got[11], 13, "midpoint rounds to 13 the numpy way, not 12");
  // endpoint=True: last sample is exactly total-1.
  assert_eq!(got.last(), Some(&25));
  assert_eq!(got.len(), 23);
}

#[test]
fn sample_frame_indices_rejects_bad_inputs() {
  assert!(sample_frame_indices(0, 4).is_err());
  assert!(sample_frame_indices(10, 0).is_err());
}

// ---------- process_frames ----------

/// Solid-color `width x height` RGB frame.
fn solid_frame(width: u32, height: u32, rgb: [u8; 3]) -> ::image::DynamicImage {
  let mut buf = ::image::RgbImage::new(width, height);
  for y in 0..height {
    for x in 0..width {
      buf.put_pixel(x, y, ::image::Rgb(rgb));
    }
  }
  ::image::DynamicImage::ImageRgb8(buf)
}

/// No-op-resize, no-rescale, no-normalize config: `process_frames` then
/// equals "image_to_array per frame, stacked", which is fully
/// deterministic (no interpolation) for hand-tracing.
fn passthrough_cfg(size: (u32, u32)) -> ImageProcessorConfig {
  ImageProcessorConfig {
    size,
    mean: [0.0, 0.0, 0.0],
    std: [1.0, 1.0, 1.0],
    rescale_factor: 1.0,
    do_resize: false,
    do_rescale: false,
    do_normalize: false,
    resample: ResizeFilter::Bicubic,
    color_order: ColorOrder::Rgb,
  }
}

#[test]
fn process_frames_stacks_channel_last_t_h_w_3() {
  // 3 solid 2x2 frames; do_resize=false so each frame -> image_to_array
  // [2,2,3] of the solid color; stacked along a new leading T axis ->
  // [3, 2, 2, 3].
  let frames = [
    solid_frame(2, 2, [10, 20, 30]),
    solid_frame(2, 2, [40, 50, 60]),
    solid_frame(2, 2, [70, 80, 90]),
  ];
  let cfg = passthrough_cfg((2, 2));
  let mut out = process_frames(&frames, &cfg).unwrap();
  assert_eq!(out.shape(), vec![3, 2, 2, 3], "stacked layout [T, H, W, 3]");

  let v: Vec<f32> = out.to_vec().unwrap();
  // Build the expected channel-last stack by hand: per frame, 4 pixels
  // (row-major) each = the solid color.
  let mut expected = Vec::new();
  for color in [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]] {
    for _pixel in 0..4 {
      expected.extend_from_slice(&color);
    }
  }
  assert_eq!(v, expected);
}

#[test]
fn process_frames_matches_per_frame_preprocess_with_rescale() {
  // With do_rescale=true (1/255) the stacked output must equal the
  // per-frame preprocess values. Single frame keeps the hand-trace tiny.
  let frame = solid_frame(1, 1, [255, 0, 128]);
  let cfg = ImageProcessorConfig {
    size: (1, 1),
    mean: [0.0, 0.0, 0.0],
    std: [1.0, 1.0, 1.0],
    rescale_factor: 1.0 / 255.0,
    do_resize: false,
    do_rescale: true,
    do_normalize: false,
    resample: ResizeFilter::Bicubic,
    color_order: ColorOrder::Rgb,
  };
  let frames = [frame];
  let mut out = process_frames(&frames, &cfg).unwrap();
  assert_eq!(out.shape(), vec![1, 1, 1, 3]);
  let v: Vec<f32> = out.to_vec().unwrap();
  // 255/255=1.0, 0/255=0.0, 128/255=0.50196...
  let expected = [1.0_f32, 0.0, 128.0 / 255.0];
  assert!(
    v.iter().zip(expected).all(|(a, b)| (a - b).abs() <= 1e-5),
    "got {v:?}"
  );
}

#[test]
fn process_frames_single_frame_shape() {
  let frames = [solid_frame(2, 3, [5, 5, 5])];
  let cfg = passthrough_cfg((2, 3));
  let out = process_frames(&frames, &cfg).unwrap();
  // image_to_array yields [H=3, W=2, 3]; stacked -> [1, 3, 2, 3].
  assert_eq!(out.shape(), vec![1, 3, 2, 3]);
}

#[test]
fn process_frames_empty_is_err() {
  let frames: [::image::DynamicImage; 0] = [];
  let cfg = passthrough_cfg((2, 2));
  assert!(process_frames(&frames, &cfg).is_err());
}
