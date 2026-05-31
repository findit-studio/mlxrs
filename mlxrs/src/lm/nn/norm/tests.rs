use super::*;

/// `1e-4` matches the existing `lm::nn::rope` golden tolerance for the
/// f32-vs-f64 fused-kernel rounding gap, with extra slack because
/// LayerNorm/GroupNorm fold a sqrt/rsqrt + a division on top of the
/// mean/var reduce.
const TOL: f32 = 1e-4;

fn vclose(got: &[f32], want: &[f32]) -> bool {
  if got.len() != want.len() {
    return false;
  }
  got
    .iter()
    .zip(want)
    .all(|(g, w)| (g - w).abs() <= TOL && g.is_finite() && w.is_finite())
}

// ─── RMSNorm ───

#[test]
fn rms_norm_hand_traced() {
  // RMSNorm of [1, 2, 3] with weight=[1, 1, 1], eps=1e-6:
  //   rms = sqrt(mean(x*x) + eps) = sqrt((1+4+9)/3 + eps) ≈ sqrt(14/3)
  //   out = x / rms * w = [1, 2, 3] / sqrt(14/3)
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let w = Array::from_slice::<f32>(&[1.0, 1.0, 1.0], &(3,)).unwrap();
  let rn = RMSNorm::new(w, 1e-6);
  let mut y = rn.forward(&x).unwrap();
  let rms = (14.0_f32 / 3.0).sqrt();
  assert!(vclose(
    &y.to_vec::<f32>().unwrap(),
    &[1.0 / rms, 2.0 / rms, 3.0 / rms]
  ));
}

#[test]
fn rms_norm_zero_input_is_finite() {
  // Zero input + eps in the rsqrt ⇒ output is finite (not NaN/Inf).
  let x = Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0], &(1, 4)).unwrap();
  let w = Array::ones::<f32>(&(4,)).unwrap();
  let rn = RMSNorm::new(w, 1e-5);
  let mut y = rn.forward(&x).unwrap();
  let v = y.to_vec::<f32>().unwrap();
  assert!(
    v.iter().all(|x| x.is_finite()),
    "expected finite, got {v:?}"
  );
}

#[test]
fn rms_norm_preserves_rank3_shape() {
  // Rank-3 [2, 3, 4] in → same shape out.
  let x =
    Array::from_slice::<f32>(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 4)).unwrap();
  let w = Array::ones::<f32>(&(4,)).unwrap();
  let rn = RMSNorm::new(w, 1e-5);
  let y = rn.forward(&x).unwrap();
  assert_eq!(y.shape(), vec![2, 3, 4]);
}

#[test]
fn rms_norm_matches_manual_fallback() {
  // The fused `mlx_fast_rms_norm` kernel must match the manual
  // `x / sqrt(mean(x*x, -1, keepdims) + eps) * weight` composition.
  let x = Array::from_slice::<f32>(&[0.5, -1.5, 2.0, 3.0, 4.0, 5.0], &(1, 2, 3)).unwrap();
  let w = Array::from_slice::<f32>(&[0.5, 1.0, 1.5], &(3,)).unwrap();
  let eps = 1e-5_f32;

  let mut via_kernel = RMSNorm::new(w.try_clone().unwrap(), eps)
    .forward(&x)
    .unwrap();

  // Manual fallback path: `x / sqrt(mean(x*x, -1, keepdims) + eps) * weight`.
  let xx = ops::arithmetic::square(&x).unwrap();
  let m = ops::reduction::mean_axes(&xx, &[-1], true).unwrap();
  let eps_arr = scalar_like(eps, &m).unwrap();
  let denom = ops::arithmetic::rsqrt(&ops::arithmetic::add(&m, &eps_arr).unwrap()).unwrap();
  let scaled = ops::arithmetic::multiply(&x, &denom).unwrap();
  let mut via_manual = ops::arithmetic::multiply(&scaled, &w).unwrap();

  assert!(vclose(
    &via_kernel.to_vec::<f32>().unwrap(),
    &via_manual.to_vec::<f32>().unwrap()
  ));
}

// ─── LayerNorm ───

#[test]
fn layer_norm_hand_traced() {
  // LayerNorm of [1, 2, 3, 4] with no affine, eps=1e-5:
  //   mean = 2.5, var = ((1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2)/4 = 1.25
  //   out = (x - 2.5) / sqrt(1.25 + 1e-5)
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let ln = LayerNorm::new(None, None, 1e-5);
  let mut y = ln.forward(&x).unwrap();
  let denom = (1.25_f32 + 1e-5).sqrt();
  let want = [
    (1.0 - 2.5) / denom,
    (2.0 - 2.5) / denom,
    (3.0 - 2.5) / denom,
    (4.0 - 2.5) / denom,
  ];
  assert!(vclose(&y.to_vec::<f32>().unwrap(), &want));
}

#[test]
fn layer_norm_zero_input_is_finite() {
  // Zero input ⇒ mean=0, var=0; eps prevents the div-by-zero. Output
  // is the all-zero array (numerator is 0 too), which is finite.
  let x = Array::from_slice::<f32>(&[0.0; 6], &(1, 6)).unwrap();
  let ln = LayerNorm::new(None, None, 1e-5);
  let mut y = ln.forward(&x).unwrap();
  let v = y.to_vec::<f32>().unwrap();
  assert!(
    v.iter().all(|x| x.is_finite()),
    "expected finite, got {v:?}"
  );
}

#[test]
fn layer_norm_preserves_rank3_shape() {
  let x =
    Array::from_slice::<f32>(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 4)).unwrap();
  let ln = LayerNorm::new(None, None, 1e-5);
  let y = ln.forward(&x).unwrap();
  assert_eq!(y.shape(), vec![2, 3, 4]);
}

#[test]
fn layer_norm_affine_applies_weight_and_bias() {
  // LayerNorm with full affine: weight=[2,2,2,2], bias=[1,1,1,1]
  // should produce 2*unaffine + 1 element-wise.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let w = Array::full::<f32>(&(4,), 2.0).unwrap();
  let b = Array::ones::<f32>(&(4,)).unwrap();
  let plain = LayerNorm::new(None, None, 1e-5);
  let affine = LayerNorm::new(Some(w), Some(b), 1e-5);
  let mut p = plain.forward(&x).unwrap();
  let mut a = affine.forward(&x).unwrap();
  let pv = p.to_vec::<f32>().unwrap();
  let av = a.to_vec::<f32>().unwrap();
  let want: Vec<f32> = pv.iter().map(|v| 2.0 * v + 1.0).collect();
  assert!(vclose(&av, &want));
}

#[test]
fn layer_norm_matches_manual_fallback() {
  // The fused `mlx_fast_layer_norm` (no affine) must match
  // `(x - mean) / sqrt(var + eps)` over the last axis.
  let x = Array::from_slice::<f32>(&[0.5, -1.5, 2.0, 3.0, 4.0, 5.0], &(1, 2, 3)).unwrap();
  let eps = 1e-5_f32;
  let mut via_kernel = LayerNorm::new(None, None, eps).forward(&x).unwrap();

  let m = ops::reduction::mean_axes(&x, &[-1], true).unwrap();
  let v = ops::reduction::var_axes(&x, &[-1], true, 0).unwrap();
  let eps_arr = scalar_like(eps, &v).unwrap();
  let denom = ops::arithmetic::rsqrt(&ops::arithmetic::add(&v, &eps_arr).unwrap()).unwrap();
  let centered = ops::arithmetic::subtract(&x, &m).unwrap();
  let mut via_manual = ops::arithmetic::multiply(&centered, &denom).unwrap();

  assert!(vclose(
    &via_kernel.to_vec::<f32>().unwrap(),
    &via_manual.to_vec::<f32>().unwrap()
  ));
}

// ─── GroupNorm ───

#[test]
fn group_norm_hand_traced_one_group_matches_layer_norm() {
  // GroupNorm with num_groups=1 is equivalent to per-token LayerNorm
  // across the (spatial + dims) features. For a [1, dims] input, that
  // is exactly LayerNorm — the hand-traced reference.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let gn = GroupNorm::new(1, 4, 1e-5, false, false).unwrap();
  let mut y = gn.forward(&x).unwrap();
  // Reference: rank-2 [1, 4] with 1 group ⇒ mean=2.5, var=1.25.
  let denom = (1.25_f32 + 1e-5).sqrt();
  let want = [
    (1.0 - 2.5) / denom,
    (2.0 - 2.5) / denom,
    (3.0 - 2.5) / denom,
    (4.0 - 2.5) / denom,
  ];
  assert!(vclose(&y.to_vec::<f32>().unwrap(), &want));
}

#[test]
fn group_norm_zero_input_is_finite() {
  // Zero rank-3 input ⇒ output is finite (eps in the rsqrt).
  let x = Array::from_slice::<f32>(&[0.0; 12], &(1, 3, 4)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let mut y = gn.forward(&x).unwrap();
  let v = y.to_vec::<f32>().unwrap();
  assert!(
    v.iter().all(|x| x.is_finite()),
    "expected finite, got {v:?}"
  );
}

#[test]
fn group_norm_preserves_rank4_shape() {
  // Rank-4 [B=2, H=3, W=3, C=4] in → same shape out.
  let n = 2 * 3 * 3 * 4;
  let x =
    Array::from_slice::<f32>(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), &(2, 3, 3, 4)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
  let y = gn.forward(&x).unwrap();
  assert_eq!(y.shape(), vec![2, 3, 3, 4]);
}

#[test]
fn group_norm_pytorch_compat_preserves_shape() {
  // The pytorch_compatible path follows the same input/output shape
  // contract, and must produce a finite result.
  let n = 2 * 3 * 3 * 4;
  let x = Array::from_slice::<f32>(
    &(0..n).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
    &(2, 3, 3, 4),
  )
  .unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, true).unwrap();
  let mut y = gn.forward(&x).unwrap();
  assert_eq!(y.shape(), vec![2, 3, 3, 4]);
  let v = y.to_vec::<f32>().unwrap();
  assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn group_norm_affine_true_applies_scale_and_shift() {
  // Regression: affine=true output == normalized * weight + bias, where
  // `normalized` is the affine=false (pure-normalization) result and
  // `(weight, bias)` is the pair the `affine()` accessor exposes. The
  // constructor materializes the references' `(ones, zeros)`, so this
  // also pins that the default affine is the identity on `normalized`.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let plain = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let affine = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
  let mut normalized = plain.forward(&x).unwrap();
  let mut a = affine.forward(&x).unwrap();
  let normalized_v = normalized.to_vec::<f32>().unwrap();
  let av = a.to_vec::<f32>().unwrap();
  // affine=true output == normalized * weight + bias.
  let (w, b) = affine.affine().expect("affine=true ⇒ Some");
  let scaled = ops::arithmetic::multiply(w, &normalized).unwrap();
  let mut want = ops::arithmetic::add(&scaled, b).unwrap();
  assert!(vclose(&av, &want.to_vec::<f32>().unwrap()));
  // default (ones, zeros) ⇒ the affine is the identity on `normalized`.
  assert!(vclose(&av, &normalized_v));
}

#[test]
fn group_norm_affine_false_is_pure_normalization() {
  // affine=false ⇒ `affine()` is None ⇒ `forward` takes the `None` arm
  // and returns the pure normalized result (no scale, no shift). Pinned
  // against `group_norm_affine_true_applies_scale_and_shift`, which
  // asserts the affine=false output is exactly the `normalized` term.
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  assert!(gn.affine().is_none());
  let mut y = gn.forward(&x).unwrap();
  let v = y.to_vec::<f32>().unwrap();
  assert_eq!(v.len(), 4);
  assert!(v.iter().all(|x| x.is_finite()));
}

/// affine is a single both-or-none `Option<(weight, bias)>`:
/// `affine=true` ⇒ `affine()` is `Some`, `affine=false` ⇒ `None`. A
/// partial affine (lone weight or lone bias) is a compile-time
/// impossibility — the field is private and holds the pair, not two
/// independent `Option`s, so `(Some, None)` / `(None, Some)` cannot be
/// constructed or mutated into (the old code had a silent
/// `_ => Ok(normalized)` drop for exactly those states).
#[test]
fn group_norm_affine_is_both_or_none_by_construction() {
  let with_affine = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
  assert!(with_affine.affine().is_some());
  let no_affine = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  assert!(no_affine.affine().is_none());
  // (compile-fail) `with_affine.affine = Some((w, ...))` with only a
  // weight is impossible: the field is private AND its type is
  // `Option<(Array, Array)>`, so a lone parameter has no representation.
}

#[test]
fn group_norm_default_constructor_no_affine() {
  // affine=false ⇒ the affine field is None (no allocation).
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  assert!(gn.affine().is_none());
  assert!(!gn.pytorch_compatible);
  assert_eq!(gn.num_groups(), 2);
}

#[test]
fn group_norm_default_constructor_affine_allocates() {
  let gn = GroupNorm::new(2, 4, 1e-5, true, false).unwrap();
  let (w, b) = gn.affine().expect("affine=true ⇒ Some");
  assert_eq!(w.shape(), vec![4]);
  assert_eq!(b.shape(), vec![4]);
  let mut w = w.try_clone().unwrap();
  let mut b = b.try_clone().unwrap();
  assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.0; 4]);
  assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.0; 4]);
}

// ─── GroupNorm::with_affine checkpoint-tensor regressions ───

/// `with_affine` installs a checkpoint's LEARNED (non-identity)
/// `(weight, bias)` — the gap `new`'s `affine: bool` couldn't fill
/// (it can only build the default `(ones, zeros)`). `affine()` must
/// return those exact tensors back.
#[test]
fn group_norm_with_affine_accepts_checkpoint_tensors() {
  let w = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &(4,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(4,)).unwrap();
  let gn = GroupNorm::with_affine(2, 4, 1e-5, Some((w, b)), false).unwrap();
  let (gw, gb) = gn.affine().expect("with_affine(Some(_)) ⇒ Some");
  let mut gw = gw.try_clone().unwrap();
  let mut gb = gb.try_clone().unwrap();
  assert_eq!(gw.to_vec::<f32>().unwrap(), vec![2.0; 4]);
  assert_eq!(gb.to_vec::<f32>().unwrap(), vec![1.0; 4]);
}

/// Coverage gap the finding flagged: the prior affine test only used
/// `(ones, zeros)` (an identity affine), so a broken scale/shift would
/// not be caught. Construct via `with_affine` with NON-identity
/// `weight`/`bias` and assert `forward` output is exactly
/// `normalized * weight + bias` (and NOT the identity `normalized`).
#[test]
fn group_norm_with_affine_non_identity_forward_applies_scale_shift() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let w = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &(4,)).unwrap();
  let b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(4,)).unwrap();
  // `normalized` = the pure (affine=None) normalization of `x`.
  let plain = GroupNorm::with_affine(2, 4, 1e-5, None, false).unwrap();
  let mut normalized = plain.forward(&x).unwrap();
  let normalized_v = normalized.to_vec::<f32>().unwrap();
  // `forward` with the non-identity affine.
  let affine = GroupNorm::with_affine(
    2,
    4,
    1e-5,
    Some((w.try_clone().unwrap(), b.try_clone().unwrap())),
    false,
  )
  .unwrap();
  let mut got = affine.forward(&x).unwrap();
  // Expected: `normalized * weight + bias`, computed independently.
  let scaled = ops::arithmetic::multiply(&w, &normalized).unwrap();
  let mut want = ops::arithmetic::add(&scaled, &b).unwrap();
  assert!(vclose(
    &got.to_vec::<f32>().unwrap(),
    &want.to_vec::<f32>().unwrap()
  ));
  // Sanity: the non-identity affine actually moved the result off
  // `normalized` (weight=2/bias=1 cannot be the identity here).
  assert!(
    !vclose(&got.to_vec::<f32>().unwrap(), &normalized_v),
    "non-identity affine must change the output"
  );
}

/// `with_affine` rejects a `weight` that is not exactly rank-1
/// `[dims]` — wrong length (`[dims + 1]`, rank-1) ⇒ `LengthMismatch`,
/// wrong rank (`[1, dims]`, rank-2) ⇒ `RankMismatch`. We split the two
/// to avoid the prior bug where both collapsed to `ShapePairMismatch`
/// despite being distinct violation classes (one rank, one length).
#[test]
fn group_norm_with_affine_rejects_wrong_shape_weight() {
  let bias = Array::zeros::<f32>(&(4,)).unwrap();
  // Wrong length: `[dims + 1]` — rank-1, single dim differs ⇒ LengthMismatch.
  let long_w = Array::ones::<f32>(&(5,)).unwrap();
  let err = GroupNorm::with_affine(2, 4, 1e-5, Some((long_w, bias.try_clone().unwrap())), false)
    .unwrap_err();
  match err {
    crate::error::Error::LengthMismatch(payload) => {
      assert!(
        payload.context().contains("weight"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.expected(), 4, "expected length 4 (dims)");
      assert_eq!(payload.actual(), 5, "actual length 5");
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
  // Wrong rank: `[1, dims]` — rank-2 ⇒ RankMismatch.
  let rank2_w = Array::ones::<f32>(&(1, 4)).unwrap();
  let err = GroupNorm::with_affine(2, 4, 1e-5, Some((rank2_w, bias)), false).unwrap_err();
  match err {
    crate::error::Error::RankMismatch(payload) => {
      assert!(
        payload.context().contains("weight"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.actual(), 2, "expected observed rank 2");
      assert_eq!(
        payload.actual_shape(),
        &[1usize, 4],
        "expected actual shape [1, 4]"
      );
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

/// `with_affine` rejects a `bias` that is not exactly rank-1 `[dims]`
/// — same split as for `weight`: wrong length (`[dims + 1]`) ⇒
/// `LengthMismatch`, wrong rank (`[1, dims]`) ⇒ `RankMismatch`.
#[test]
fn group_norm_with_affine_rejects_wrong_shape_bias() {
  let weight = Array::ones::<f32>(&(4,)).unwrap();
  // Wrong length: `[dims + 1]` — rank-1 ⇒ LengthMismatch.
  let long_b = Array::zeros::<f32>(&(5,)).unwrap();
  let err = GroupNorm::with_affine(
    2,
    4,
    1e-5,
    Some((weight.try_clone().unwrap(), long_b)),
    false,
  )
  .unwrap_err();
  match err {
    crate::error::Error::LengthMismatch(payload) => {
      assert!(
        payload.context().contains("bias"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.expected(), 4, "expected length 4 (dims)");
      assert_eq!(payload.actual(), 5, "actual length 5");
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
  // Wrong rank: `[1, dims]` — rank-2 ⇒ RankMismatch.
  let rank2_b = Array::zeros::<f32>(&(1, 4)).unwrap();
  let err = GroupNorm::with_affine(2, 4, 1e-5, Some((weight, rank2_b)), false).unwrap_err();
  match err {
    crate::error::Error::RankMismatch(payload) => {
      assert!(
        payload.context().contains("bias"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.actual(), 2, "expected observed rank 2");
      assert_eq!(
        payload.actual_shape(),
        &[1usize, 4],
        "expected actual shape [1, 4]"
      );
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

/// `with_affine(.., None, ..)` ⇒ `affine()` is `None` and `forward`
/// returns the pure normalized result (no scale, no shift).
#[test]
fn group_norm_with_affine_none_is_pure_normalization() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let gn = GroupNorm::with_affine(2, 4, 1e-5, None, false).unwrap();
  assert!(gn.affine().is_none());
  // `forward` must equal the default-path `group_norm` normalization.
  let mut got = gn.forward(&x).unwrap();
  let mut want = gn.group_norm(&x).unwrap();
  assert!(vclose(
    &got.to_vec::<f32>().unwrap(),
    &want.to_vec::<f32>().unwrap()
  ));
}

// ─── GroupNorm shape-invariant regressions ───

/// Rank-1 `[C]` with `num_groups=1` used to silently corrupt
/// activations (passed as `[C, 1, 1]` and normalized singleton groups
/// to zero); now an explicit `Err(RankMismatch)`.
#[test]
fn group_norm_rank1_input_errors() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let gn = GroupNorm::new(1, 4, 1e-5, false, false).unwrap();
  let err = gn.forward(&x).unwrap_err();
  match err {
    crate::error::Error::RankMismatch(payload) => {
      assert!(
        payload.context().contains("rank"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.actual(), 1);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

/// Rank-2 `[1, 3]` with `num_groups=2` used to silently pass (element
/// count is divisible — 6/2 = 3 — but the 3-wide feature axis isn't
/// splittable). The new constructor catches `dims % num_groups != 0`
/// before construction; constructing with valid `dims=4` and then
/// forwarding `[1, 3]` (whose last-axis 3 != configured 4) exercises
/// the dims-equality enforcement in the forward.
#[test]
fn group_norm_feature_dim_mismatch_errors() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let err = gn.forward(&x).unwrap_err();
  match err {
    crate::error::Error::LengthMismatch(payload) => {
      assert!(
        payload.context().contains("last-axis"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.expected(), 4);
      assert_eq!(payload.actual(), 3);
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
}

/// Same invariants must hold on the `pytorch_compatible` path: rank-1
/// input rejected.
#[test]
fn group_norm_pytorch_compat_rank1_input_errors() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(4,)).unwrap();
  let gn = GroupNorm::new(1, 4, 1e-5, false, true).unwrap();
  let err = gn.forward(&x).unwrap_err();
  match err {
    crate::error::Error::RankMismatch(payload) => {
      assert!(
        payload.context().contains("rank"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.actual(), 1);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

/// Same dims-equality invariant on the `pytorch_compatible` path:
/// mismatch between configured `dims=4` and input last-axis 3 is
/// rejected.
#[test]
fn group_norm_pytorch_compat_feature_dim_mismatch_errors() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(1, 3)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, true).unwrap();
  let err = gn.forward(&x).unwrap_err();
  match err {
    crate::error::Error::LengthMismatch(payload) => {
      assert!(
        payload.context().contains("last-axis"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.expected(), 4);
      assert_eq!(payload.actual(), 3);
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
}

/// Regression: the valid rank-2 case (`[1, 4]` with `num_groups=2`)
/// must continue to work after the new validation guards.
#[test]
fn group_norm_valid_rank2_still_works() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 4)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let mut y = gn.forward(&x).unwrap();
  let v = y.to_vec::<f32>().unwrap();
  assert_eq!(v.len(), 4);
  assert!(v.iter().all(|x| x.is_finite()));
}

// ─── GroupNorm constructor validation regressions ───

/// Constructor must reject negative `dims` on BOTH affine paths.
/// Previously only the `affine=true` branch ran `usize::try_from`; the
/// `affine=false` branch silently accepted any `dims` (including
/// nonsense) and the forward derived the feature width from
/// `x.shape().last()`.
#[test]
fn group_norm_constructor_rejects_negative_dims() {
  let err = GroupNorm::new(2, -1, 1e-5, false, false).unwrap_err();
  match err {
    crate::error::Error::OutOfRange(payload) => {
      assert!(
        payload.context().contains("dims"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert!(
        payload.requirement().contains("positive"),
        "unexpected requirement: {:?}",
        payload.requirement()
      );
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
}

/// Constructor must reject `dims` not divisible by `num_groups` on
/// BOTH affine paths. Previously the divisibility was only checked at
/// forward-time against `x.shape().last()`, so an `affine=false`
/// GroupNorm could be constructed with `dims=3, num_groups=2` and
/// later normalize a `[1, 4]` input (whose last axis happens to
/// divide 2) — silent config/checkpoint mismatch.
#[test]
fn group_norm_constructor_rejects_non_divisible_dims() {
  let err = GroupNorm::new(2, 3, 1e-5, false, false).unwrap_err();
  match err {
    crate::error::Error::DivisibilityConstraint(payload) => {
      assert_eq!(payload.name_dividend(), "dims");
      assert_eq!(payload.name_divisor(), "num_groups");
    }
    other => panic!("expected DivisibilityConstraint, got {other:?}"),
  }
}

/// `dims == 0` is rejected (`positive` ⇒ `> 0`, not `>= 0`). A
/// zero-dim GroupNorm has no feature axis to normalize and the
/// downstream `dims / num_groups` would yield `group_size = 0`.
#[test]
fn group_norm_constructor_rejects_zero_dims() {
  let err = GroupNorm::new(2, 0, 1e-5, false, false).unwrap_err();
  match err {
    crate::error::Error::OutOfRange(payload) => {
      assert!(
        payload.context().contains("dims"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert!(
        payload.requirement().contains("positive"),
        "unexpected requirement: {:?}",
        payload.requirement()
      );
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
}

/// Regression: a valid `affine=false` GroupNorm still constructs Ok
/// after the new constructor checks. (Belt for the suspenders — the
/// existing `group_norm_default_constructor_no_affine` test already
/// covers this; keep an explicit one named for the constructor-spec
/// item.)
#[test]
fn group_norm_constructor_accepts_valid_non_affine() {
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  assert_eq!(gn.dims(), 4);
  assert_eq!(gn.num_groups(), 2);
  assert!(gn.affine().is_none());
}

/// Forward rejects a config/checkpoint dim mismatch: construct with
/// `dims=4` and call forward on `[1, 8]`. The 8-wide input is
/// divisible by `num_groups=2` (would have silently normalized
/// previously), but doesn't match the configured `dims=4` ⇒
/// `Err(LengthMismatch)` naming both expected (4) and actual (8).
#[test]
fn group_norm_forward_rejects_dim_mismatch() {
  let x = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &(1, 8)).unwrap();
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let err = gn.forward(&x).unwrap_err();
  match err {
    crate::error::Error::LengthMismatch(payload) => {
      assert!(
        payload.context().contains("last-axis"),
        "expected context to name last-axis: {:?}",
        payload.context()
      );
      assert_eq!(payload.expected(), 4);
      assert_eq!(payload.actual(), 8);
    }
    other => panic!("expected LengthMismatch, got {other:?}"),
  }
}

/// `GroupNorm::new(.., affine = true, ..)` must reject a malformed
/// `(num_groups, dims)` config on the cheap integer validation
/// (`validate_group_params`) BEFORE materializing the default
/// `(ones, zeros)` affine tensors. Previously `new` checked only
/// `dims > 0`, built the two MLX arrays, and only THEN ran the full
/// `num_groups`/divisibility validation inside `with_affine` — so a
/// known-bad config paid for two allocations before erroring. Both a
/// non-positive `num_groups` and a non-divisible `dims` must `Err`.
///
/// The no-allocation property is structural: `validate_group_params`
/// is called before the `Array::ones`/`Array::zeros` lines in `new`,
/// so an `Err` here is returned without ever reaching them.
#[test]
fn group_norm_new_affine_true_invalid_config_rejects_before_alloc() {
  // `num_groups = 0` (non-positive) with `affine = true`.
  let err = GroupNorm::new(0, 4, 1e-5, true, false).unwrap_err();
  assert!(
    matches!(err, crate::error::Error::OutOfRange(_)),
    "expected OutOfRange for num_groups=0, got {err:?}"
  );
  // `dims = 8` not divisible by `num_groups = 3`, with `affine = true`.
  let err = GroupNorm::new(3, 8, 1e-5, true, false).unwrap_err();
  assert!(
    matches!(err, crate::error::Error::DivisibilityConstraint(_)),
    "expected DivisibilityConstraint for non-divisible dims, got {err:?}"
  );
}

// ─── GroupNorm field-visibility regressions ───

/// `num_groups` and `dims` are PRIVATE fields with read-only public
/// accessors. This test demonstrates the accessors return the
/// constructor-validated values and — by virtue of compiling without
/// reaching for the field — confirms the read path goes through the
/// accessor. Direct field access from outside `super::` would fail to
/// compile (the field's visibility is module-private). External code
/// previously could write `gn.num_groups = 0` and then `gn.forward(_)`
/// would PANIC inside `validate_input_shape`'s `dims_i32 % 0`; with
/// the field private, that mutation path is statically impossible.
#[test]
fn group_norm_num_groups_dims_are_read_only_via_accessors() {
  let gn = GroupNorm::new(4, 16, 1e-5, false, false).unwrap();
  assert_eq!(gn.num_groups(), 4);
  assert_eq!(gn.dims(), 16);
  // (compile-fail) external `gn.num_groups = 0` and `gn.dims = 0` are
  // both private-field errors; trying them here from inside `super::`
  // would compile (same module), so we don't try — the visibility
  // guarantee is what the regression turns on, not a runtime check.
}

// ─── inferred_dim overflow regression ───

/// `inferred_dim` used to compute `total = shape.iter().product()`
/// unchecked, so a shape whose `usize` product wraps would yield the
/// wrong inferred dim (and either a reshape boundary failure or a
/// passing reshape on a corrupted layout). Now an `Err(ArithmeticOverflow)`
/// before we ever reach the divisibility check.
///
/// `usize::MAX` on its own already wraps on the `* 2` step.
#[test]
fn inferred_dim_overflow_errors() {
  let shape: [usize; 2] = [usize::MAX, 2];
  let err = inferred_dim(&shape, &[1, 1]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("overflow"),
        "unexpected context: {:?}",
        payload.context()
      );
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

// ─── accessor coverage (weight_ref / bias_ref) ───

/// `RMSNorm::weight_ref` returns a borrow of the installed weight (lazy,
/// no eval). Pins the read path goes through the accessor (the field is
/// private). The `Array` is already materialized — `shape()` does not
/// evaluate.
#[test]
fn rms_norm_weight_ref_returns_installed_weight() {
  let w = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &(3,)).unwrap();
  let rn = RMSNorm::new(w, 1e-5);
  let got = rn.weight_ref();
  assert_eq!(got.shape(), vec![3]);
  // The borrowed handle is the same weight we can read back element-wise.
  let mut got = got.try_clone().unwrap();
  assert_eq!(got.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

/// `LayerNorm::weight_ref` is `Some(&weight)` when affine weight is
/// installed and `None` when it is absent (the `affine=false` arm).
#[test]
fn layer_norm_weight_ref_some_and_none() {
  // Present: weight installed ⇒ Some(&weight), borrow exposes the values.
  let w = Array::from_slice::<f32>(&[2.0, 2.0, 2.0, 2.0], &(4,)).unwrap();
  let ln = LayerNorm::new(Some(w), None, 1e-5);
  let got = ln.weight_ref().expect("weight installed ⇒ Some");
  assert_eq!(got.shape(), vec![4]);
  let mut got = got.try_clone().unwrap();
  assert_eq!(got.to_vec::<f32>().unwrap(), vec![2.0; 4]);
  // Absent: no weight ⇒ None.
  let plain = LayerNorm::new(None, None, 1e-5);
  assert!(plain.weight_ref().is_none());
}

/// `LayerNorm::bias_ref` is `Some(&bias)` when affine bias is installed
/// and `None` when it is absent.
#[test]
fn layer_norm_bias_ref_some_and_none() {
  // Present: bias installed ⇒ Some(&bias).
  let b = Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(4,)).unwrap();
  let ln = LayerNorm::new(None, Some(b), 1e-5);
  let got = ln.bias_ref().expect("bias installed ⇒ Some");
  assert_eq!(got.shape(), vec![4]);
  let mut got = got.try_clone().unwrap();
  assert_eq!(got.to_vec::<f32>().unwrap(), vec![1.0; 4]);
  // Absent: no bias ⇒ None.
  let plain = LayerNorm::new(None, None, 1e-5);
  assert!(plain.bias_ref().is_none());
}

// ─── validate_input_shape branch coverage ───

/// `validate_input_shape` rejects a feature (last) axis past `i32::MAX`
/// with `ArithmeticOverflow` BEFORE the dims-equality comparison (the
/// `i32::try_from(dims)` map_err). Pure integer logic — exercised by
/// feeding the private method a synthetic shape directly (no MLX array,
/// which could never allocate `i32::MAX + 1` elements anyway). Built on
/// a validly-constructed GroupNorm so the call site is the real method.
#[test]
fn validate_input_shape_feature_dim_overflow_errors() {
  let gn = GroupNorm::new(2, 4, 1e-5, false, false).unwrap();
  let big = (i32::MAX as usize) + 1;
  let err = gn.validate_input_shape(&[2, big]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("feature dim"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.op_type(), "i32");
      assert_eq!(payload.operands(), &[("dim", big as u64)]);
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

/// `validate_input_shape`'s divisibility check (`dims_i32 %
/// self.num_groups != 0`) is documented as belt-and-suspenders —
/// unreachable through the public constructors (which enforce `dims %
/// num_groups == 0`). To exercise it we build a `GroupNorm` via a
/// struct literal with a deliberately broken invariant (`dims = 4` NOT
/// divisible by `num_groups = 3`) — possible only because the test
/// module is a child of `norm` and so can name the private fields. The
/// input's last axis (4) equals the configured `dims` (4), so the
/// dims-equality check passes and control reaches the divisibility arm:
/// `4 % 3 != 0` ⇒ `DivisibilityConstraint`.
#[test]
fn validate_input_shape_divisibility_belt_and_suspenders() {
  let gn = GroupNorm {
    num_groups: 3,
    dims: 4,
    affine: None,
    eps: 1e-5,
    pytorch_compatible: false,
  };
  let err = gn.validate_input_shape(&[1, 4]).unwrap_err();
  match err {
    crate::error::Error::DivisibilityConstraint(payload) => {
      assert_eq!(payload.name_dividend(), "feature_dim");
      assert_eq!(payload.dividend(), 4);
      assert_eq!(payload.name_divisor(), "num_groups");
      assert_eq!(payload.divisor(), 3);
    }
    other => panic!("expected DivisibilityConstraint, got {other:?}"),
  }
}

// ─── shape_to_i32 overflow ───

/// `shape_to_i32` errors with `ArithmeticOverflow` on a `usize` dim past
/// `i32::MAX`. Pure function over a slice — no MLX. Operand carries the
/// offending dim value.
#[test]
fn shape_to_i32_overflow_errors() {
  let big = (i32::MAX as usize) + 1;
  let err = shape_to_i32(&[2, big]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("exceeds i32::MAX"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.op_type(), "i32");
      assert_eq!(payload.operands(), &[("dim", big as u64)]);
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

/// A well-formed shape round-trips through `shape_to_i32` unchanged
/// (the success arm of the `try_from` collect).
#[test]
fn shape_to_i32_ok_roundtrip() {
  assert_eq!(shape_to_i32(&[2, 3, 4]).unwrap(), vec![2i32, 3, 4]);
}

// ─── batch_dim branches ───

/// `batch_dim` of a rank-0 (empty) shape errors with `RankMismatch`
/// (`first()` is `None`). Pure function — no MLX.
#[test]
fn batch_dim_rank0_errors() {
  let err = batch_dim(&[]).unwrap_err();
  match err {
    crate::error::Error::RankMismatch(payload) => {
      assert!(
        payload.context().contains("rank >= 1"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.actual(), 0);
      assert_eq!(payload.actual_shape(), &[] as &[usize]);
    }
    other => panic!("expected RankMismatch, got {other:?}"),
  }
}

/// `batch_dim` errors with `ArithmeticOverflow` when the leading dim is
/// past `i32::MAX` (the `i32::try_from(b)` map_err). Pure function.
#[test]
fn batch_dim_overflow_errors() {
  let big = (i32::MAX as usize) + 1;
  let err = batch_dim(&[big, 3]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("batch dim exceeds i32::MAX"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.op_type(), "i32");
      assert_eq!(payload.operands(), &[("batch_dim", big as u64)]);
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

/// `batch_dim` of a well-formed shape returns the leading dim as `i32`
/// (the success arm).
#[test]
fn batch_dim_ok() {
  assert_eq!(batch_dim(&[7, 2, 5]).unwrap(), 7i32);
}

// ─── inferred_dim remaining branches ───

/// `inferred_dim` rejects a negative `known_dims` entry with `OutOfRange`
/// (the `usize::try_from(d)` map_err). Pure function — `[4]` reshaped
/// with a `-1` literal known dim (distinct from mlx's `-1` sentinel,
/// which the safe layer resolves numerically, never passing it here).
#[test]
fn inferred_dim_negative_known_dim_errors() {
  let err = inferred_dim(&[4], &[-1]).unwrap_err();
  match err {
    crate::error::Error::OutOfRange(payload) => {
      assert!(
        payload.context().contains("known reshape dim"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert!(
        payload.requirement().contains("non-negative"),
        "unexpected requirement: {:?}",
        payload.requirement()
      );
      assert_eq!(payload.value(), "-1");
    }
    other => panic!("expected OutOfRange, got {other:?}"),
  }
}

/// `inferred_dim` errors with `ArithmeticOverflow` when the PRODUCT of
/// `known_dims` overflows `usize` (the divisor `checked_mul`). The
/// `shape` product (total = 1) is kept tiny so it does not overflow
/// first; three `i32::MAX` factors overflow on the third multiply
/// (`i32::MAX^2 ≈ 4.6e18 < usize::MAX`, `* i32::MAX` overflows). Pure
/// function — distinct from the existing `inferred_dim_overflow_errors`,
/// which exercises the `shape`-product (total) overflow instead.
#[test]
fn inferred_dim_divisor_product_overflow_errors() {
  let err = inferred_dim(&[1], &[i32::MAX, i32::MAX, i32::MAX]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("divisor product"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.op_type(), "usize");
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

/// `inferred_dim` errors with `InvariantViolation` when a `known_dims`
/// entry is `0` (the divisor collapses to 0). Pure function — `[4]`
/// reshaped against a known dim of `0`.
#[test]
fn inferred_dim_zero_divisor_errors() {
  let err = inferred_dim(&[4], &[0]).unwrap_err();
  match err {
    crate::error::Error::InvariantViolation(payload) => {
      assert!(
        payload.context().contains("reshape divisor"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert!(
        payload.requirement().contains("non-zero"),
        "unexpected requirement: {:?}",
        payload.requirement()
      );
    }
    other => panic!("expected InvariantViolation, got {other:?}"),
  }
}

/// `inferred_dim` errors with `DivisibilityConstraint` when `total` is
/// not an exact multiple of the divisor (`5 % 2 != 0`). Pure function.
#[test]
fn inferred_dim_not_multiple_errors() {
  let err = inferred_dim(&[5], &[2]).unwrap_err();
  match err {
    crate::error::Error::DivisibilityConstraint(payload) => {
      assert_eq!(payload.name_dividend(), "total_elements");
      assert_eq!(payload.dividend(), 5);
      assert_eq!(payload.name_divisor(), "divisor_per_slot");
      assert_eq!(payload.divisor(), 2);
    }
    other => panic!("expected DivisibilityConstraint, got {other:?}"),
  }
}

/// `inferred_dim` errors with `ArithmeticOverflow` when the resolved
/// quotient (`total / divisor`) exceeds `i32::MAX` (the final
/// `i32::try_from(inferred)` map_err). `total = i32::MAX + 1` with a
/// divisor of `1` ⇒ the quotient is one past `i32::MAX`. Pure function.
#[test]
fn inferred_dim_result_overflow_errors() {
  let big = (i32::MAX as usize) + 1;
  let err = inferred_dim(&[big], &[1]).unwrap_err();
  match err {
    crate::error::Error::ArithmeticOverflow(payload) => {
      assert!(
        payload.context().contains("inferred dim exceeds i32::MAX"),
        "unexpected context: {:?}",
        payload.context()
      );
      assert_eq!(payload.op_type(), "i32");
      assert_eq!(payload.operands(), &[("inferred_dim", big as u64)]);
    }
    other => panic!("expected ArithmeticOverflow, got {other:?}"),
  }
}

/// `inferred_dim` happy path: `total = product(shape)` divided by
/// `product(known_dims)` yields the residual `-1`-replacement dim. For
/// `shape = [2, 12]` (total 24) and `known_dims = [2, 3]` (divisor 6),
/// the inferred middle dim is `24 / 6 = 4`. Pure function — pins the
/// success arm independently of any reshape.
#[test]
fn inferred_dim_happy_path() {
  assert_eq!(inferred_dim(&[2, 12], &[2, 3]).unwrap(), 4i32);
}
