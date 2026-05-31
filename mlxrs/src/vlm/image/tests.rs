//! Private unit tests for `vlm::image` helpers and enum string tags
//! that the public-API integration suite (`tests/vlm_image.rs`) cannot
//! reach: the `as_str` / `Display` surface of [`ResizeFilter`],
//! [`ColorOrder`], and [`Layout`]; the private [`make_channel_broadcast`]
//! rank arms; the rank-0 and W-divisibility validation branches; and the
//! `load_image` parse-error closure.
//!
//! These live inline (not in the integration suite) because
//! [`make_channel_broadcast`] is private and because the `as_str`
//! coverage is cheapest to assert directly against the const-fn tag.

use super::*;

// -- ResizeFilter / ColorOrder / Layout string tags ------------------
//
// `as_str` (and the `derive_more::Display` impl that forwards to it via
// `#[display("{}", self.as_str())]`) is otherwise never exercised. Assert
// every variant's tag explicitly so the per-variant match arms are
// covered, and round-trip through `Display` so the forwarding holds.

#[test]
fn resize_filter_as_str_all_variants() {
  assert_eq!(ResizeFilter::Nearest.as_str(), "nearest");
  assert_eq!(ResizeFilter::Bilinear.as_str(), "bilinear");
  assert_eq!(ResizeFilter::Bicubic.as_str(), "bicubic");
  assert_eq!(ResizeFilter::Lanczos3.as_str(), "lanczos3");
  // Display forwards to as_str.
  for f in [
    ResizeFilter::Nearest,
    ResizeFilter::Bilinear,
    ResizeFilter::Bicubic,
    ResizeFilter::Lanczos3,
  ] {
    assert_eq!(format!("{f}"), f.as_str());
  }
}

#[test]
fn color_order_as_str_all_variants() {
  assert_eq!(ColorOrder::Rgb.as_str(), "rgb");
  assert_eq!(ColorOrder::Bgr.as_str(), "bgr");
  for c in [ColorOrder::Rgb, ColorOrder::Bgr] {
    assert_eq!(format!("{c}"), c.as_str());
  }
}

#[test]
fn layout_as_str_all_variants() {
  assert_eq!(Layout::Hwc.as_str(), "hwc");
  assert_eq!(Layout::Chw.as_str(), "chw");
  assert_eq!(Layout::Bchw.as_str(), "bchw");
  for l in [Layout::Hwc, Layout::Chw, Layout::Bchw] {
    assert_eq!(format!("{l}"), l.as_str());
  }
}

// -- make_channel_broadcast rank arms --------------------------------

#[test]
fn make_channel_broadcast_rank1_returns_unreshaped_channel_vector() {
  // ndim <= 1: the early-return arm hands back the plain `(3,)` array
  // without reshaping to `[1, ..., 1, 3]`. Verify the shape is `[3]`
  // and the values survive the f32 round-trip.
  let mut a = make_channel_broadcast(&[0.485, 0.456, 0.406], 1, Dtype::F32)
    .expect("rank-1 channel broadcast must succeed");
  assert_eq!(
    a.shape(),
    vec![3],
    "ndim<=1 returns the unreshaped (3,) array"
  );
  let v: Vec<f32> = a.to_vec().expect("materialize (3,) channel vector");
  assert_eq!(v.len(), 3);
  assert!((v[0] - 0.485).abs() < 1e-6);
  assert!((v[1] - 0.456).abs() < 1e-6);
  assert!((v[2] - 0.406).abs() < 1e-6);
}

#[test]
fn make_channel_broadcast_rank2_reshapes_to_leading_singleton() {
  // ndim == 2: reshape to `[1, 3]`. Confirms the > 1 path builds the
  // stack `[1; MAX_NDIM]` buffer, sets the trailing dim to 3, and
  // reshapes the (3,) constant accordingly.
  let a = make_channel_broadcast(&[1.0, 2.0, 3.0], 2, Dtype::F32)
    .expect("rank-2 channel broadcast must succeed");
  assert_eq!(a.shape(), vec![1, 3], "ndim==2 reshapes to [1, 3]");
}

#[test]
fn make_channel_broadcast_rejects_ndim_over_max() {
  // ndim > MAX_NDIM (16): the explicit guard converts to a typed
  // `CapExceeded` naming the cap (rather than indexing a 16-slot stack
  // buffer out of bounds). Called directly with `ndim = 17` so no
  // 17-dim mlx array is constructed — only the guard arithmetic runs.
  let err = make_channel_broadcast(&[0.0, 0.0, 0.0], 17, Dtype::F32)
    .expect_err("ndim > MAX_NDIM (16) must be rejected");
  match err {
    Error::CapExceeded(p) => {
      assert_eq!(p.cap_name(), "MAX_NDIM");
      assert_eq!(p.cap(), 16, "cap is MAX_NDIM = 16");
      assert_eq!(p.observed(), 17, "offending ndim is 17");
    }
    other => panic!("expected CapExceeded(MAX_NDIM), got {other:?}"),
  }
}

// -- normalize rank-0 validation -------------------------------------

#[test]
fn normalize_rejects_rank0_scalar_input() {
  // A 0-D (scalar) array trips the `ndim == 0` guard with a typed
  // `RankMismatch` BEFORE the `shape[ndim - 1]` trailing-channel read
  // (which would otherwise underflow). Build the scalar via the same
  // empty-shape `from_slice` idiom `Array::full` uses internally.
  let scalar = Array::from_slice(&[42.0_f32], &[0i32; 0]).expect("0-d scalar array");
  assert_eq!(scalar.ndim(), 0, "from_slice with an empty shape is rank-0");
  let err = normalize(&scalar, &[0.0; 3], &[1.0; 3])
    .expect_err("rank-0 scalar input must be rejected before the trailing-dim read");
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 0, "observed rank is 0");
      assert!(
        p.context().contains("normalize"),
        "RankMismatch must name normalize; got: {}",
        p.context()
      );
    }
    other => panic!("expected RankMismatch on rank-0 input, got {other:?}"),
  }
}

// -- patchify W-divisibility validation ------------------------------

#[test]
fn patchify_w_not_divisible_errors_on_width_axis() {
  // The H-divisibility check fires first; to reach the W check the
  // input must have H divisible by patch_size but W not. `[4, 5, 3]`
  // with patch_size 2: H=4 is divisible, W=5 is not -> the W-axis
  // `DivisibilityConstraint` arm. The existing integration test uses
  // `[5, 4, 3]` (H not divisible), which exercises the H arm instead.
  let arr = Array::from_slice(&[0.0_f32; 4 * 5 * 3], &(4usize, 5, 3)).expect("[4,5,3] array");
  let err = patchify(&arr, 2).expect_err("W=5 not divisible by patch_size=2 must error");
  match err {
    Error::DivisibilityConstraint(p) => {
      assert_eq!(p.name_dividend(), "W", "must be the W-axis arm, not H");
      assert_eq!(p.dividend(), 5);
      assert_eq!(p.divisor(), 2);
      assert!(
        p.context().contains("W by patch_size"),
        "context must name the W divisibility constraint; got: {}",
        p.context()
      );
    }
    other => panic!("expected DivisibilityConstraint on the W axis, got {other:?}"),
  }
}

// -- load_image parse-error path -------------------------------------

#[test]
fn load_image_corrupt_png_returns_parse_error() {
  // A file with a recognized `.png` extension whose CONTENT is not a
  // valid PNG: `ImageReader::open` + `with_guessed_format` succeed (the
  // extension pins the PNG format guess; the garbage bytes do not match
  // another decoder's magic), but `into_decoder()` fails to construct a
  // PngDecoder on the bad signature -> the `parse_err` closure ->
  // `Error::Parse`. (`load_image_nonexistent_path_returns_err` covers
  // the sibling `io_err`/open-failure closure instead.)
  let dir = std::env::temp_dir().join(format!("mlxrs-vlm-image-parse-{}", std::process::id()));
  std::fs::create_dir_all(&dir).expect("create temp dir");
  let path = dir.join("corrupt.png");
  // Not a PNG: no 8-byte PNG signature, no other recognizable header.
  std::fs::write(&path, b"this is definitely not a png file at all").expect("write garbage");
  let err = load_image(&path).expect_err("a corrupt PNG must fail to decode");
  // Best-effort cleanup before asserting (so a failed assert still GCs).
  let _ = std::fs::remove_file(&path);
  let _ = std::fs::remove_dir(&dir);
  match err {
    Error::Parse(p) => assert!(
      p.context().contains("load_image"),
      "Parse error must name load_image; got context: {}",
      p.context()
    ),
    other => panic!("expected Error::Parse from the decode path, got {other:?}"),
  }
}

// -- rotate_buf / rotate_buf_u8 element-count overflow gate -----------
//
// `rotate_buf<T>` and `rotate_buf_u8` each compute the destination
// element count `w * h * channels` in `usize` via a `checked_mul`
// chain and surface a wrap as a typed `ArithmeticOverflow` (naming the
// w/h/channels operands) BEFORE the `try_reserve_exact` allocation and
// the per-pixel copy loop. The sibling `apply_orientation_tests.rs`
// notes (see `rotate90_accepts_small_image_within_budget`) that this
// negative path "can't be constructed without OOM-ing the test
// process" via a real `DynamicImage` — but the helpers take the
// dimensions as plain `u32` arguments decoupled from any backing
// buffer, so we can drive the overflow directly with hostile dims and
// an EMPTY source slice: the `checked_mul` fires and returns `Err`
// before the source slice is ever indexed (the `debug_assert_eq!` on
// `src.len()` and the copy loop are both downstream of the early
// return), so no allocation and no out-of-bounds read occurs.
//
// `u32::MAX as usize` * `u32::MAX as usize` = 18_446_744_065_119_617_025
// which still FITS in a 64-bit `usize` (< usize::MAX), so the wrap
// happens at the trailing `* channels` step (channels >= 2) — exercising
// the full `and_then(|wh| wh.checked_mul(channels))` arm rather than the
// first multiply. The project ships only 64-bit targets; on a (hypothetical)
// 32-bit usize the wrap would instead fire at the `w * h` step but still
// yield the same `ArithmeticOverflow` variant with all three operands, so
// the assertions below hold regardless of pointer width.

#[test]
fn rotate_buf_element_count_overflow_is_typed_arithmetic_error() {
  // `rotate_buf::<u8>` with `channels = 2` (the LumaA8 stride). Empty
  // source slice — the overflow check precedes any `src` access.
  let err = rotate_buf::<u8>(&[], u32::MAX, u32::MAX, 2, RotateKind::Rotate90)
    .expect_err("w * h * channels must overflow usize for u32::MAX dims and channels=2");
  match err {
    Error::ArithmeticOverflow(p) => {
      assert_eq!(
        p.context(),
        "rotate_buf: elements (w * h * channels)",
        "context must name the rotate_buf element-count expression"
      );
      assert_eq!(p.op_type(), "usize", "the overflowing result type is usize");
      // All three runtime operands are recorded for the diagnostic.
      let ops = p.operands();
      assert_eq!(ops.len(), 3, "w, h, and channels operands are all carried");
      assert_eq!(ops[0], ("w", u64::from(u32::MAX)));
      assert_eq!(ops[1], ("h", u64::from(u32::MAX)));
      assert_eq!(ops[2], ("channels", 2));
    }
    other => panic!("expected ArithmeticOverflow from rotate_buf, got {other:?}"),
  }
}

#[test]
fn rotate_buf_element_count_overflow_independent_of_rotation_kind() {
  // The overflow gate runs before the `match rotation` index math, so
  // every `RotateKind` reaches it identically. Sweep all four with a
  // wider channel stride (4 = Rgba) to cover the same closure from the
  // other rotate variants without constructing a hostile image.
  for kind in [
    RotateKind::Rotate90,
    RotateKind::Rotate270,
    RotateKind::Rotate90FlipH,
    RotateKind::Rotate270FlipH,
  ] {
    let err = rotate_buf::<f32>(&[], u32::MAX, u32::MAX, 4, kind)
      .expect_err("w * h * 4 must overflow usize for u32::MAX dims regardless of rotation");
    match err {
      Error::ArithmeticOverflow(p) => {
        assert_eq!(p.context(), "rotate_buf: elements (w * h * channels)");
        assert_eq!(p.operands().last(), Some(&("channels", 4)));
      }
      other => panic!("expected ArithmeticOverflow for {kind:?}, got {other:?}"),
    }
  }
}

#[test]
fn rotate_buf_u8_element_count_overflow_is_typed_arithmetic_error() {
  // The SIMD-routed u8 helper shares the same `checked_mul` gate and
  // the same early-return-before-alloc/SIMD ordering, with its own
  // distinct context label. `channels = 4` is the Rgba8 hot-path
  // stride. Empty source slice — the overflow returns before the
  // `try_reserve_exact`, the `set_len`, and the SIMD dispatch.
  let err = rotate_buf_u8(&[], u32::MAX, u32::MAX, 4, RotateKind::Rotate90)
    .expect_err("w * h * 4 must overflow usize for u32::MAX dims");
  match err {
    Error::ArithmeticOverflow(p) => {
      assert_eq!(
        p.context(),
        "rotate_buf_u8: elements (w * h * channels)",
        "context must name the rotate_buf_u8 element-count expression"
      );
      assert_eq!(p.op_type(), "usize");
      let ops = p.operands();
      assert_eq!(ops.len(), 3);
      assert_eq!(ops[0], ("w", u64::from(u32::MAX)));
      assert_eq!(ops[1], ("h", u64::from(u32::MAX)));
      assert_eq!(ops[2], ("channels", 4));
    }
    other => panic!("expected ArithmeticOverflow from rotate_buf_u8, got {other:?}"),
  }
}

// -- rotate_buf success path on a non-overflowing tiny input ---------
//
// A direct (rather than via-`apply_orientation_fallible`) call to the
// generic helper on a 2x1 buffer confirms the post-gate path — the
// `try_reserve_exact` + `resize` + permutation loop — returns the
// rotated buffer with the swapped-extent length, closing the bridge
// between the overflow-gate tests above (which return before the loop)
// and the byte-identical image-rs parity tests in
// `apply_orientation_tests.rs` (which go through the public entry).

#[test]
fn rotate_buf_rotate90_tiny_luma_permutes_into_swapped_extent() {
  // 2x1 Luma8 (channels = 1): source pixels [a, b] at (0,0) and (1,0).
  // Rotate90 maps (x, y) -> (h-1-y, x) with output dims (h, w) = (1, 2),
  // i.e. a 1-wide, 2-tall buffer. With h=1: (0,0)->(0,0), (1,0)->(0,1).
  // Output buffer (row-major over out_w=1) is therefore [a, b] — the
  // same two values, now stacked vertically. Length must be 2 (1*2*1).
  let src: [u8; 2] = [7, 9];
  let dst =
    rotate_buf::<u8>(&src, 2, 1, 1, RotateKind::Rotate90).expect("2x1 rotate90 well under cap");
  assert_eq!(dst.len(), 2, "1*2*channels(1) destination element count");
  // out_w = rotated_dims(2,1).0 = src_h = 1. dst_off for (nx=h-1-y, ny=x):
  //   src (0,0) -> (nx=0, ny=0) -> off 0
  //   src (1,0) -> (nx=0, ny=1) -> off 1
  assert_eq!(
    dst,
    vec![7, 9],
    "Rotate90 of a single row stacks the pixels"
  );
}
