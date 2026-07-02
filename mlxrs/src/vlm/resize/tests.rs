use super::*;

/// Force-scalar variant of [`resize_rgba8`] (calls the `*_scalar`
/// kernels directly, bypassing the NEON dispatch). Used only by the
/// differential test to compare against the dispatched path. Mirrors
/// [`resize_rgba8`]'s premultiplied-alpha staging exactly (premultiply
/// the source, convolve, unpremultiply) so the differential test stays
/// a faithful NEON-vs-scalar comparison of the WHOLE resize.
fn resize_rgba8_scalar(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  dst_w: usize,
  dst_h: usize,
  filter: Filter,
) -> Vec<u8> {
  if filter == Filter::Nearest {
    let dst_len = dst_w * dst_h * CHANNELS;
    return resize_nearest(src, src_w, src_h, dst_w, dst_h, dst_len).unwrap();
  }
  let src_pm = premultiply_rgba(src).unwrap();
  let hc = precompute_coeffs(src_w, dst_w, filter).unwrap();
  let vc = precompute_coeffs(src_h, dst_h, filter).unwrap();
  let mut inter = vec![0u8; src_h * dst_w * CHANNELS];
  let mut dst = vec![0u8; dst_w * dst_h * CHANNELS];
  convolve_axis_scalar(&src_pm, src_w, src_h, &mut inter, dst_w, &hc);
  convolve_vertical_scalar(&inter, dst_w, src_h, &mut dst, dst_h, &vc);
  unpremultiply_rgba(&mut dst);
  dst
}

/// Deterministic pseudo-random RGBA8 source (LCG — no rand dependency).
fn make_src(w: usize, h: usize, seed: u32) -> Vec<u8> {
  let mut s = seed.wrapping_add(1);
  let mut v = Vec::with_capacity(w * h * CHANNELS);
  for _ in 0..w * h * CHANNELS {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    v.push((s >> 24) as u8);
  }
  v
}

#[test]
fn neon_matches_scalar_across_sizes_and_filters() {
  // Differential: the dispatched path (NEON on aarch64, scalar
  // elsewhere) must produce output BIT-IDENTICAL to the force-scalar
  // path, across sizes straddling the 4-channel vector boundary and
  // both up/down scaling. On a non-aarch64 host this is a scalar-vs-
  // scalar identity (still a useful determinism check); on aarch64 it
  // is the real NEON-vs-scalar guarantee.
  let filters = [
    Filter::Bilinear,
    Filter::Bicubic,
    Filter::Lanczos3,
    Filter::Nearest,
  ];
  // Sizes chosen to straddle odd/even widths + up/down + 1-px axes.
  let cases = [
    (4usize, 4usize, 2usize, 2usize),
    (3, 5, 7, 2),
    (5, 3, 2, 8),
    (8, 6, 4, 3),
    (2, 2, 9, 9),
    (5, 1, 2, 1),
    (1, 5, 1, 2),
    (7, 7, 7, 7),
    (16, 9, 5, 11),
  ];
  for (i, &(sw, sh, dw, dh)) in cases.iter().enumerate() {
    let src = make_src(sw, sh, i as u32 * 7 + 1);
    for &f in &filters {
      let dispatched = resize_rgba8(&src, sw, sh, dw, dh, f).unwrap();
      let scalar = resize_rgba8_scalar(&src, sw, sh, dw, dh, f);
      assert_eq!(
        dispatched, scalar,
        "NEON-vs-scalar differential mismatch for {f:?} {sw}x{sh}->{dw}x{dh}"
      );
    }
  }
}

#[test]
fn rejects_zero_dimensions() {
  let src = [0u8; 4]; // 1x1 RGBA
  for (sw, sh, dw, dh) in [(0, 1, 2, 2), (1, 0, 2, 2), (1, 1, 0, 2), (1, 1, 2, 0)] {
    let r = resize_rgba8(
      &src[..sw.max(1) * sh.max(1) * CHANNELS],
      sw,
      sh,
      dw,
      dh,
      Filter::Bilinear,
    );
    assert!(
      matches!(r, Err(Error::OutOfRange(_))),
      "zero dim {sw}x{sh}->{dw}x{dh} must be OutOfRange, got {r:?}"
    );
  }
}

#[test]
fn rejects_src_buffer_length_mismatch() {
  // src buffer too short for the claimed dims -> LengthMismatch (not a
  // panic / OOB read).
  let src = [0u8; 4]; // claims 4 bytes but we say 2x2 (needs 16)
  let r = resize_rgba8(&src, 2, 2, 1, 1, Filter::Bilinear);
  assert!(matches!(r, Err(Error::LengthMismatch(_))), "got {r:?}");
}

#[test]
fn rejects_overflowing_dst_product() {
  // dst_w * dst_h * 4 overflows usize -> ArithmeticOverflow (the
  // structural try_reserve guard's overflow branch). Use usize::MAX-ish
  // dims.
  let src = [0u8; 4];
  let big = usize::MAX / 2 + 1;
  let r = resize_rgba8(&src, 1, 1, big, big, Filter::Bilinear);
  assert!(matches!(r, Err(Error::ArithmeticOverflow(_))), "got {r:?}");
}

#[test]
fn rejects_skinny_to_wide_oversized_intermediate() {
  // Adversarial case: a `1x131072` source resized to `131072x1`.
  // The RGBA source is `1*131072*4` = 512 KiB (under the 512 MiB cap)
  // and the destination is `131072*1*4` = 512 KiB (under the cap), but
  // the horizontal-pass intermediate is `src_h * dst_w * 4`
  // = `131072 * 131072 * 4` ≈ 68 GiB. `checked_buffer_bytes` must
  // reject the intermediate BEFORE any `try_reserve_exact` / zero-fill,
  // so this returns a recoverable `Err` — no 68 GiB allocation, no
  // overcommit zero-fill abort. (A convolution filter, not NEAREST:
  // NEAREST has no intermediate and a `1x131072`->`131072x1` NEAREST is
  // a legitimate small resize.)
  let src = vec![0u8; 131072 * CHANNELS];
  for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
    let r = resize_rgba8(&src, 1, 131072, 131072, 1, f);
    match &r {
      Err(Error::CapExceeded(p)) => {
        assert_eq!(p.cap_name(), "MAX_DECODED_IMAGE_BYTES");
        assert!(
          p.context().contains("intermediate"),
          "{f:?}: CapExceeded context should name the intermediate buffer, got: {}",
          p.context()
        );
      }
      _ => panic!("{f:?}: 1x131072->131072x1 must reject the ~68 GiB intermediate, got {r:?}"),
    }
  }
}

#[test]
fn wide_to_skinny_does_not_abort() {
  // The reverse orientation: a `131072x1` source resized to `1x131072`.
  // Unlike skinny->wide, this orientation has NO oversized buffer — the
  // intermediate is `src_h * dst_w * 4` = `1 * 1 * 4` = 4 bytes, the
  // destination is `1 * 131072 * 4` = 512 KiB, and both coefficient
  // tables are small (the `131072`-tall output axis upscales from
  // `in_size=1`, so `ksize=1` and the table is `131072 * 4` = 512 KiB).
  // So a correct implementation SUCCEEDS here with an exactly-sized
  // small output — the guarantee under test is simply "no abort, no
  // 68 GiB allocation": the asymmetry is the point (the 68 GiB scratch
  // needs a large `src_h` AND a large `dst_w`, see
  // `rejects_huge_intermediate_with_tiny_ends`).
  let src = vec![0u8; 131072 * CHANNELS];
  for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
    let r = resize_rgba8(&src, 131072, 1, 1, 131072, f);
    match r {
      Ok(out) => assert_eq!(
        out.len(),
        131072 * CHANNELS,
        "{f:?}: wide->skinny output must be exactly dst_w*dst_h*4"
      ),
      Err(Error::CapExceeded(_)) | Err(Error::OutOfMemory) => {}
      Err(other) => panic!("{f:?}: unexpected error {other:?}"),
    }
  }
}

#[test]
fn rejects_huge_intermediate_with_tiny_ends() {
  // The horizontal-pass intermediate is `src_h * dst_w * 4` — it blows
  // up only when BOTH `src_h` (input height) and `dst_w` (untrusted
  // target width) are large, which is exactly the gap the public
  // `resize` wrapper's source/destination caps miss.
  let src = vec![7u8; 131072 * CHANNELS];
  let r = resize_rgba8(&src, 1, 131072, 131072, 1, Filter::Bicubic);
  assert!(
    matches!(r, Err(Error::CapExceeded(_))),
    "huge intermediate with tiny source+dest must be CapExceeded, got {r:?}"
  );
}

#[test]
fn rejects_oversized_coefficient_table() {
  // Coefficient-buffer adversarial case via `precompute_coeffs`.
  let r = precompute_coeffs(1, 200_000_000, Filter::Bilinear);
  assert!(
    matches!(r, Err(Error::CapExceeded(_))),
    "200M-wide coefficient table must exceed the 512 MiB cap (got Ok or wrong error)"
  );
  // And via the full resize: every guard yields a typed cap/overflow Err.
  let src = vec![0u8; 4 * CHANNELS];
  let r2 = resize_rgba8(&src, 1, 4, 200_000_000, 1, Filter::Bilinear);
  assert!(
    matches!(r2, Err(Error::CapExceeded(_))),
    "resize to a 200M-wide target must be CapExceeded, got {r2:?}"
  );
}

#[test]
fn checked_buffer_bytes_caps_and_overflows() {
  // Direct unit test of the helper. Under-cap passes and returns the
  // byte product; over-cap yields CapExceeded and overflow yields
  // ArithmeticOverflow.
  assert_eq!(
    checked_buffer_bytes(1024, 4, "ok").unwrap(),
    4096,
    "under-cap product must pass through"
  );
  // Exactly at the cap (512 MiB) is allowed; one byte over is not.
  assert_eq!(
    checked_buffer_bytes(MAX_DECODED_IMAGE_BYTES, 1, "at-cap").unwrap(),
    MAX_DECODED_IMAGE_BYTES,
    "a buffer exactly at the cap must be allowed"
  );
  assert!(
    matches!(
      checked_buffer_bytes(MAX_DECODED_IMAGE_BYTES + 1, 1, "over"),
      Err(Error::CapExceeded(_))
    ),
    "one byte over the cap must be rejected"
  );
  assert!(
    matches!(
      checked_buffer_bytes(usize::MAX, 4, "overflow"),
      Err(Error::ArithmeticOverflow(_))
    ),
    "a product overflowing usize must be rejected (not wrap)"
  );
}

#[test]
fn output_length_is_exact() {
  // Every accepted resize returns exactly dst_w*dst_h*4 bytes — the
  // invariant `vlm::image::resize` relies on for `ImageBuffer::from_raw`.
  let src = make_src(8, 6, 3);
  for f in [
    Filter::Nearest,
    Filter::Bilinear,
    Filter::Bicubic,
    Filter::Lanczos3,
  ] {
    let out = resize_rgba8(&src, 8, 6, 5, 4, f).unwrap();
    assert_eq!(out.len(), 5 * 4 * CHANNELS, "filter {f:?} output length");
  }
}

#[test]
fn constant_image_is_preserved() {
  // A constant-color image must reproduce the constant at every output
  // pixel for every convolution filter (kernel sums to 1.0). Exact for
  // the integer path (no rounding drift on a flat field).
  let mut src = Vec::with_capacity(6 * 6 * CHANNELS);
  for _ in 0..6 * 6 {
    src.extend_from_slice(&[123, 45, 200, 255]);
  }
  for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
    for &(dw, dh) in &[(3usize, 3usize), (9, 9), (4, 7)] {
      let out = resize_rgba8(&src, 6, 6, dw, dh, f).unwrap();
      for px in out.as_chunks::<CHANNELS>().0 {
        assert_eq!(
          px,
          &[123, 45, 200, 255],
          "constant must survive {f:?} -> {dw}x{dh}"
        );
      }
    }
  }
}

#[test]
fn muldiv255_matches_pil_and_is_opaque_identity() {
  // MULDIV255(c, 255) must be the identity for EVERY c (PIL relies on
  // this so an opaque RGBA resize is bit-identical to a straight one).
  for c in 0u8..=255 {
    assert_eq!(
      muldiv255(c, 255),
      c,
      "MULDIV255({c}, 255) must equal {c} (opaque identity)"
    );
    // MULDIV255(c, 0) == 0 for every c (zero-alpha kills the colour).
    assert_eq!(muldiv255(c, 0), 0, "MULDIV255({c}, 0) must be 0");
  }
  // Spot-check PIL's exact rounding against a hand-computed value:
  // MULDIV255(255, 128) = ((32768>>8)+32768)>>8 = (128+32768)>>8 = 128.
  assert_eq!(muldiv255(255, 128), 128, "MULDIV255(255,128) hand-checked");
  // MULDIV255(200, 100) = ((20128>>8)+20128)>>8 = (78+20128)>>8 = 78.
  assert_eq!(muldiv255(200, 100), 78, "MULDIV255(200,100) hand-checked");
}

#[test]
fn premultiply_unpremultiply_opaque_is_identity() {
  // For a fully-opaque buffer (A == 255) premultiply then unpremultiply
  // must round-trip to the exact input — this is why the opaque
  // PIL-reference resize tests are unaffected by the premultiply path.
  let src: Vec<u8> = (0u8..=255).flat_map(|c| [c, 255 - c, c / 2, 255]).collect();
  let pm = premultiply_rgba(&src).unwrap();
  assert_eq!(pm, src, "premultiply must be identity for opaque alpha");
  let mut round = pm;
  unpremultiply_rgba(&mut round);
  assert_eq!(
    round, src,
    "unpremultiply must be identity for opaque alpha"
  );
}

#[test]
fn premultiply_transparent_pixel_zeros_colour() {
  // A fully-transparent pixel (A == 0): premultiply zeros every colour
  // channel (PIL `MULDIV255(c, 0) == 0`), and unpremultiply leaves the
  // already-zero colour at zero (PIL passthrough for A == 0).
  let src = vec![255u8, 128, 64, 0]; // transparent, arbitrary colour
  let pm = premultiply_rgba(&src).unwrap();
  assert_eq!(
    pm,
    vec![0, 0, 0, 0],
    "premultiply of a transparent pixel must zero the colour channels"
  );
  let mut round = pm;
  unpremultiply_rgba(&mut round);
  assert_eq!(
    round,
    vec![0, 0, 0, 0],
    "unpremultiply of a zero-alpha pixel keeps colour 0 (PIL passthrough)"
  );
}

#[test]
fn unpremultiply_clips_and_divides_like_pil() {
  // Partial alpha: unpremultiply does CLIP8(255*c/a) (truncating
  // integer division, clamp [0,255]).
  // a=128: CLIP8(255*64/128) = 16320/128 = 127.
  let mut buf = vec![64u8, 0, 0, 128];
  unpremultiply_rgba(&mut buf);
  assert_eq!(
    buf[0], 127,
    "unpremultiply 64 over alpha 128: 255*64/128=127"
  );
  assert_eq!(buf[3], 128, "alpha unchanged");
  // Premultiplied colour > alpha (possible after convolution rounding):
  // CLIP8 must clamp to 255. c=200, a=100 -> 255*200/100=510 -> 255.
  let mut buf2 = vec![200u8, 0, 0, 100];
  unpremultiply_rgba(&mut buf2);
  assert_eq!(
    buf2[0], 255,
    "unpremultiply must clamp an over-alpha colour to 255"
  );
}

#[test]
fn resize_premultiplied_transparent_red_opaque_blue() {
  // Example at the kernel level: transparent-red `(255,0,0,0)`
  // next to opaque-blue `(0,0,255,255)`, 2x1 -> 1x1. The premultiplied
  // path must yield pure blue with half alpha `(0,0,255,128)` for every
  // non-NEAREST filter — NOT the straight-channel purple
  // `(128,0,128,128)`. NEAREST is exempt (pure gather, no premultiply).
  let src = [255u8, 0, 0, 0, 0, 0, 255, 255]; // 2x1: t-red, o-blue
  for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
    let out = resize_rgba8(&src, 2, 1, 1, 1, f).unwrap();
    assert_eq!(
      out,
      vec![0, 0, 255, 128],
      "{f:?}: premultiplied-alpha resize must give pure blue (0,0,255,128)"
    );
  }
  // NEAREST gathers the rightmost pixel (out 0 -> floor(0.5*2/1)=1):
  // straight opaque blue, no premultiply.
  let nn = resize_rgba8(&src, 2, 1, 1, 1, Filter::Nearest).unwrap();
  assert_eq!(
    nn,
    vec![0, 0, 255, 255],
    "NEAREST must not premultiply — gathers the opaque-blue pixel verbatim"
  );
}

#[test]
fn precompute_coeffs_weights_sum_to_unity_fixedpoint() {
  // Each output index's normalized fixed-point taps should sum to
  // approximately 1<<PRECISION_BITS (the rounding may shift the sum by
  // at most `n` LSB across `n` taps). This guards the normalization.
  let one = 1i64 << PRECISION_BITS;
  for f in [Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
    for &(insz, outsz) in &[(8usize, 3usize), (3, 8), (5, 5), (16, 4)] {
      let c = precompute_coeffs(insz, outsz, f).unwrap();
      for o in 0..outsz {
        let (_, n) = c.bounds[o];
        let s: i64 = c.weights[o * c.ksize..o * c.ksize + n]
          .iter()
          .map(|&w| i64::from(w))
          .sum();
        let tol = n as i64 + 1;
        assert!(
          (s - one).abs() <= tol,
          "{f:?} {insz}->{outsz} out {o}: tap sum {s} not within {tol} of {one}"
        );
      }
    }
  }
}
