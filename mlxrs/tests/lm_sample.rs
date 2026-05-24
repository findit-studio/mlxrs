//! Deterministic sanity tests for the M3 sampling utilities
//! (`mlxrs::lm::sample`), ported from `mlx_lm.sample_utils`.
//!
//! RNG-flaky assertions are avoided: the transforms are exercised on the
//! deterministic mask/argmax paths; the categorical draw only asserts
//! shape/dtype + index bounds under a fixed key.

#![cfg(feature = "lm")]

use mlxrs::{Array, Dtype, lm::sample};

/// `[1, 4]` log-probs whose argmax is index 3.
fn logprobs() -> Array {
  Array::from_slice::<f32>(&[-3.0, -2.0, -1.0, 0.0], &[1, 4]).unwrap()
}

#[test]
fn argmax_sample_picks_max_index() {
  let lp = logprobs();
  let mut tok = sample::argmax_sample(&lp).unwrap();
  assert_eq!(tok.shape(), vec![1]);
  assert_eq!(tok.to_vec::<u32>().unwrap(), vec![3]);
}

#[test]
fn top_k_1_keeps_only_the_max() {
  let lp = logprobs();
  let mut out = sample::apply_top_k(&lp, 1).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  // Only the top token (index 3) survives; the rest are -inf.
  assert_eq!(v[3], 0.0);
  assert!(v[0].is_infinite() && v[0] < 0.0);
  assert!(v[1].is_infinite() && v[1] < 0.0);
  assert!(v[2].is_infinite() && v[2] < 0.0);
}

#[test]
fn top_k_out_of_range_errors() {
  let lp = logprobs();
  assert!(sample::apply_top_k(&lp, 0).is_err());
  assert!(sample::apply_top_k(&lp, 4).is_err());
}

#[test]
fn top_p_full_mass_keeps_all() {
  let lp = logprobs();
  // top_p just under 1 keeps the full distribution (cumprob > 1 - top_p
  // for every token); shape/dtype preserved.
  let mut out = sample::apply_top_p(&lp, 0.999).unwrap();
  assert_eq!(out.shape(), vec![1, 4]);
  let v = out.to_vec::<f32>().unwrap();
  assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn top_p_aggressive_keeps_at_least_the_top() {
  let lp = logprobs();
  // Very small top_p prunes the tail; the most-likely token (index 3) must
  // always remain finite.
  let mut out = sample::apply_top_p(&lp, 0.05).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  assert!(v[3].is_finite());
}

#[test]
fn min_p_keeps_top_and_prunes_tail() {
  let lp = logprobs();
  let mut out = sample::apply_min_p(&lp, 0.5, 1).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  // Threshold = max + log(0.5) = 0 + (-0.693) ≈ -0.693, so only index 3
  // (logprob 0.0) is kept.
  assert!(v[3].is_finite());
  assert!(v[0].is_infinite() && v[0] < 0.0);
}

#[test]
fn min_p_invalid_params_error() {
  let lp = logprobs();
  assert!(sample::apply_min_p(&lp, 1.5, 1).is_err());
  assert!(sample::apply_min_p(&lp, 0.1, 0).is_err());
  // vocab = 4 (logprobs() is [1,4]); min_tokens_to_keep > vocab_size must
  // error (mlx-lm rejects the out-of-range kth) rather than silently
  // over-pruning to a single token.
  assert!(sample::apply_min_p(&lp, 0.9, 5).is_err());
}

#[test]
fn categorical_sampling_shape_and_bounds() {
  let lp = logprobs();
  let key = mlxrs::ops::random::key(0).unwrap();
  let mut tok = sample::categorical_sampling(&lp, 0.8, &key).unwrap();
  assert_eq!(tok.shape(), vec![1]);
  let idx = tok.to_vec::<u32>().unwrap();
  assert_eq!(idx.len(), 1);
  assert!(idx[0] < 4);
}

/// Regression: `apply_top_p` must scatter the cumulative mass back to the
/// ORIGINAL token order via an integer-exact inverse permutation
/// (`argsort(sorted_indices)`), not an `arange as f32` build (which aliased
/// indices above 2^24). Uses logprobs whose ascending sort is a NON-identity
/// permutation `[1, 2, 0, 3]`, so a wrong inverse keeps the wrong originals.
#[test]
fn top_p_inverse_permutation_preserves_original_order() {
  // exp() (unnormalized): [0.368, 0.0498, 0.135, 1.0]; sorted ascending by
  // logprob → original indices [1, 2, 0, 3]; inclusive cumsum
  // [0.0498, 0.185, 0.553, 1.553]; mapped back to original order →
  // [0.553, 0.0498, 0.185, 1.553]. threshold = 1 - 0.7 = 0.3 ⇒ keep {0, 3},
  // drop {1, 2} (well-separated from 0.3 — f32-robust).
  let lp = Array::from_slice::<f32>(&[-1.0, -3.0, -2.0, 0.0], &[1, 4]).unwrap();
  let mut out = sample::apply_top_p(&lp, 0.7).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  assert_eq!(v[0], -1.0, "orig idx 0 kept with its logprob");
  assert!(v[1].is_infinite() && v[1] < 0.0, "orig idx 1 pruned");
  assert!(v[2].is_infinite() && v[2] < 0.0, "orig idx 2 pruned");
  assert_eq!(v[3], 0.0, "orig idx 3 kept with its logprob");
}

/// Regression (Codex f16/bf16 dtype fidelity): the masking transforms must
/// PRESERVE the input dtype and keep threshold/mask arithmetic *in that
/// dtype* — mirroring mlx-lm's weak Python scalars (`scalar_like`), not
/// promoting f16/bf16 to f32 via a concrete f32 scalar. Asserts output
/// dtype == input dtype AND the mask stays correct in half precision (the
/// dtype change must not alter behavior); the temperature path must still
/// draw a valid index from half input.
#[test]
fn half_and_bfloat_preserve_dtype_and_mask() {
  for dt in [Dtype::F16, Dtype::BF16] {
    // argmax of [-3,-2,-1,0] is index 3; threshold separations below are
    // wide enough to be robust to f16/bf16 rounding.
    let lp = Array::from_slice::<f32>(&[-3.0, -2.0, -1.0, 0.0], &[1, 4])
      .unwrap()
      .astype(dt)
      .unwrap();

    let tk = sample::apply_top_k(&lp, 1).unwrap();
    assert_eq!(tk.dtype().unwrap(), dt, "top_k preserves {dt:?}");
    let mut tkf = tk.astype(Dtype::F32).unwrap();
    let v = tkf.to_vec::<f32>().unwrap();
    assert_eq!(v[3], 0.0, "{dt:?} top_k keeps argmax with its logprob");
    assert!(v[0].is_infinite() && v[0] < 0.0, "{dt:?} top_k prunes tail");

    let tp = sample::apply_top_p(&lp, 0.7).unwrap();
    assert_eq!(tp.dtype().unwrap(), dt, "top_p preserves {dt:?}");
    let mut tpf = tp.astype(Dtype::F32).unwrap();
    let vp = tpf.to_vec::<f32>().unwrap();
    assert!(vp[3].is_finite(), "{dt:?} top_p keeps argmax");

    let mp = sample::apply_min_p(&lp, 0.5, 1).unwrap();
    assert_eq!(mp.dtype().unwrap(), dt, "min_p preserves {dt:?}");
    let mut mpf = mp.astype(Dtype::F32).unwrap();
    let vm = mpf.to_vec::<f32>().unwrap();
    assert!(vm[3].is_finite(), "{dt:?} min_p keeps top");
    assert!(
      vm[0].is_infinite() && vm[0] < 0.0,
      "{dt:?} min_p prunes tail"
    );

    let key = mlxrs::ops::random::key(0).unwrap();
    let mut tok = sample::categorical_sampling(&lp, 0.8, &key).unwrap();
    let idx = tok.to_vec::<u32>().unwrap();
    assert_eq!(idx.len(), 1);
    assert!(idx[0] < 4, "{dt:?} categorical draws an in-range index");
  }
}

/// Copilot 4309130407: `apply_top_p` must reject a non-finite or
/// out-of-`(0, 1]` `top_p` (consistent with `apply_top_k`/`apply_min_p`)
/// instead of silently no-op'ing or masking everything. `1.0` is a valid
/// no-op (mlx-lm's effective gate is the open interval).
#[test]
fn top_p_out_of_range_errors() {
  let lp = logprobs();
  assert!(sample::apply_top_p(&lp, 0.0).is_err());
  assert!(sample::apply_top_p(&lp, -0.1).is_err());
  assert!(sample::apply_top_p(&lp, 1.5).is_err());
  assert!(sample::apply_top_p(&lp, f32::NAN).is_err());
  assert!(sample::apply_top_p(&lp, f32::INFINITY).is_err());
  assert!(
    sample::apply_top_p(&lp, 1.0).is_ok(),
    "top_p == 1.0 is a valid no-op"
  );
}

/// Copilot 4309130407: `categorical_sampling` must reject a non-finite or
/// non-positive `temp` (mlx-lm relies on `make_sampler` guaranteeing
/// `temp > 0` — `temp == 0` → argmax; that dispatch is deferred here) rather
/// than producing `inf`/`NaN` logits.
#[test]
fn categorical_sampling_invalid_temp_errors() {
  let lp = logprobs();
  let key = mlxrs::ops::random::key(0).unwrap();
  assert!(sample::categorical_sampling(&lp, 0.0, &key).is_err());
  assert!(sample::categorical_sampling(&lp, -1.0, &key).is_err());
  assert!(sample::categorical_sampling(&lp, f32::NAN, &key).is_err());
  assert!(sample::categorical_sampling(&lp, f32::INFINITY, &key).is_err());
}

/// LM-6 (#116) NaN-safety regression: a tiny `temp` whose `1/temp` would
/// overflow either (a) the logits dtype on the `scalar_like` astype cast
/// (f16 + `temp < 1/f16::MAX ≈ 1.526e-5`) or (b) the upstream f32
/// reciprocal itself (any dtype + subnormal positive `temp <
/// 1/f32::MAX ≈ 2.94e-39`) must still produce a draw within
/// `[0, vocab)` — NOT silently degrade to a NaN distribution that
/// `random::categorical` would draw a degenerate index from. The fix
/// uses `divide(logits, scalar_like(temp))` rather than
/// `multiply(logits, scalar_like(1/temp))` so the reciprocal is never
/// materialized. Covers all three call sites (LM, VLM, STT) through the
/// shared primitive.
#[test]
fn categorical_sampling_tiny_and_subnormal_temp_stays_finite() {
  let key = mlxrs::ops::random::key(0).unwrap();

  // Path (a): f16 logits + tiny temp. 1/temp ≈ 1e7 overflows f16::MAX
  // (~65504); the multiply path would clamp to +Inf inside `scalar_like`,
  // and `0 * +Inf = NaN` propagates over the L3-R2-max-shifted row.
  let lp_f16 = Array::from_slice::<f32>(&[-3.0, -2.0, -1.0, 0.0], &[1, 4])
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let mut tok = sample::categorical_sampling(&lp_f16, 1e-7_f32, &key).unwrap();
  let idx = tok.to_vec::<u32>().unwrap();
  assert_eq!(idx.len(), 1, "f16 tiny-temp shape");
  assert!(
    idx[0] < 4,
    "f16 tiny-temp must draw within [0, vocab); got {}",
    idx[0]
  );

  // Path (b): subnormal positive f32 temp. 1/temp = +Inf in f32 BEFORE
  // any astype, regardless of logits dtype. The validator passes
  // (`temp.is_finite() && > 0`), so a NaN row would have escaped the old
  // guard. Exercise on every float dtype that `make_sampler` allows.
  let subnormal: f32 = 1e-40; // < 1/f32::MAX ≈ 2.94e-39; subnormal positive
  assert!(subnormal.is_finite() && subnormal > 0.0);
  assert!(
    (1.0_f32 / subnormal).is_infinite(),
    "test premise: 1/temp overflows f32 reciprocal"
  );
  for dt in [Dtype::F32, Dtype::F16, Dtype::BF16] {
    let lp = Array::from_slice::<f32>(&[-3.0, -2.0, -1.0, 0.0], &[1, 4])
      .unwrap()
      .astype(dt)
      .unwrap();
    let mut tok = sample::categorical_sampling(&lp, subnormal, &key).unwrap();
    let idx = tok.to_vec::<u32>().unwrap();
    assert_eq!(idx.len(), 1, "{dt:?} subnormal-temp shape");
    assert!(
      idx[0] < 4,
      "{dt:?} subnormal-temp must draw within [0, vocab); got {}",
      idx[0]
    );
  }
}

/// LM-6 R1 follow-up (Codex adversarial review): the previous fix moved
/// from `multiply(logits, scalar_like(1/temp))` to
/// `divide(logits, scalar_like(temp, logits))`, killing the materialized-
/// `1/temp` overflow but leaving the dtype-cast leg open — `scalar_like`
/// still casts `temp` to the logits dtype BEFORE the divide, and any
/// positive `temp` below the dtype's minimum subnormal rounds to 0 in
/// f16/bf16. A max-shifted row then yields `0 / 0 = NaN`, which the
/// older index-bounds-only test
/// (`categorical_sampling_tiny_and_subnormal_temp_stays_finite`) could
/// not catch (`random::categorical` returns *some* in-range index on a
/// NaN distribution). This test asserts the SCALED LOGITS themselves
/// are all finite for sub-min-subnormal `temp`s across F32/F16/BF16,
/// reaching into the new `scale_logits_by_temp` helper so the assertion
/// is on the divide output itself — not the post-softmax categorical
/// draw, which is uninformative under NaN. The fix in
/// `scale_logits_by_temp` has two parts: (a) the f32-denominator path
/// (upcast logits to f32, divide in f32, downcast back) so `temp` never
/// gets cast down to f16/bf16; (b) a `temp.max(f32::MIN_POSITIVE)`
/// clamp so MLX's internal multiply-by-reciprocal in `divide` (the
/// f32-divisor reciprocal it materializes inside the kernel on Apple
/// Silicon) does not overflow for sub-`1/f32::MAX` temps and produce
/// `0 * +Inf = NaN` on the max-shifted zero entries (this is what the
/// `temp = 1e-40` regression below would trip without the clamp; bf16
/// shares an exponent range with f32, so any `temp` below bf16's min
/// subnormal is also below `1/f32::MAX` — the clamp is unavoidable for
/// that specific Codex-finding sub-case).
#[test]
fn categorical_sampling_tiny_temp_produces_finite_scaled_logits() {
  // `temp = 1e-40_f32` — sub-min-subnormal for f16 (its min subnormal
  // ~5.96e-8) AND for bf16 (its min subnormal ~9.18e-41, since bf16
  // shares f32's exponent range); also below `1/f32::MAX ≈ 2.94e-39`,
  // so the divide kernel's internal multiply-by-reciprocal would
  // overflow for the f32 path too without the `f32::MIN_POSITIVE`
  // clamp — the test exercises BOTH the dtype-cast leg (f16) AND the
  // f32-reciprocal-overflow leg (bf16 + f32) the LM-6 R1 fix closes.
  let temp: f32 = 1e-40;
  assert!(
    temp.is_finite() && temp > 0.0,
    "test premise: temp is finite +ve"
  );

  // Construct logits with a single max position (idx 3) and the rest at
  // zero, so a max-shift (or just the raw row) contains explicit zeros —
  // the entries that turn into 0/0 NaNs under the dtype-cast bug and
  // 0*+Inf NaNs under the divide-reciprocal-overflow bug.
  for dt in [Dtype::F32, Dtype::F16, Dtype::BF16] {
    let lp = Array::from_slice::<f32>(
      &[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      &[1, 10],
    )
    .unwrap()
    .astype(dt)
    .unwrap();

    // (1) Direct assertion on the scaled logits themselves — the strong
    // check the R1 finding mandates (not just sampled-index range). All
    // entries must be finite (the `0` entries must NOT have become NaN
    // via either of the two failure legs).
    let scaled = sample::scale_logits_by_temp(&lp, temp).unwrap();
    assert_eq!(scaled.dtype().unwrap(), dt, "{dt:?} preserves dtype");
    let mut sf = scaled.astype(Dtype::F32).unwrap();
    let sv = sf.to_vec::<f32>().unwrap();
    assert_eq!(sv.len(), 10, "{dt:?} scaled shape preserved");
    for (i, x) in sv.iter().enumerate() {
      assert!(
        !x.is_nan(),
        "{dt:?} scaled[{i}] must NOT be NaN under sub-min-subnormal temp; got {x} in {sv:?}"
      );
    }
    // The max position (idx 3) divides a finite positive by the clamped
    // `temp` (= `f32::MIN_POSITIVE` ~ 1.18e-38), so `5 / 1.18e-38 ≈
    // 4.2e38` overflows f32::MAX to +Inf legitimately —
    // `random::categorical`'s internal softmax shifts +Inf rows
    // correctly (one-hot at the max). The zero positions must be
    // exactly finite zero: `0 / temp = 0` with the clamp in place.
    for (i, x) in sv.iter().enumerate() {
      if i == 3 {
        assert!(!x.is_nan(), "{dt:?} max position must NOT be NaN");
      } else {
        assert!(
          x.is_finite() && *x == 0.0,
          "{dt:?} zero position {i} must stay finite zero, got {x}"
        );
      }
    }

    // (2) Cross-check via the public categorical_sampling path — its
    // softmax on the scaled logits must NOT produce a NaN distribution.
    // The index-only check below is necessary but not sufficient; the
    // finite-scaled assertion above is the load-bearing one.
    let key = mlxrs::ops::random::key(0).unwrap();
    let mut tok = sample::categorical_sampling(&lp, temp, &key).unwrap();
    let idx = tok.to_vec::<u32>().unwrap();
    assert_eq!(idx.len(), 1, "{dt:?} categorical shape");
    assert!(
      idx[0] < 10,
      "{dt:?} categorical draws in-range index; got {}",
      idx[0]
    );
  }
}

// ---------------------------------------------------------------------------
// M3 follow-up: XTC + penalty/bias logits-processor parity
// ---------------------------------------------------------------------------

/// `[1, 4]` raw logits with mixed signs for the penalty/bias transforms.
fn penalty_logits() -> Array {
  Array::from_slice::<f32>(&[2.0, -4.0, 1.0, -1.0], &[1, 4]).unwrap()
}

/// `f32` view of an array (after an `astype` for half-precision cases).
fn vals(a: &Array, dt: Dtype) -> Vec<f32> {
  a.astype(dt)
    .unwrap()
    .astype(Dtype::F32)
    .unwrap()
    .to_vec::<f32>()
    .unwrap()
}

/// XTC with `xtc_probability == 1.0`: `uniform(0,1) > 1.0` is *always* false,
/// so the mask is applied on every call (key-independent, deterministic).
/// Logits are `ln([0.5, 0.3, 0.15, 0.05])` ⇒ softmax probs exactly
/// `[0.5, 0.3, 0.15, 0.05]`. threshold 0.1 ⇒ probs above = {0.5,0.3,0.15};
/// cutoff = min = 0.15; `mask = probs > 0.15` ⇒ idx0 (0.5) & idx1 (0.3) → -inf;
/// idx2 (0.15, NOT > 0.15) & idx3 kept (the boundary token always survives).
#[test]
fn xtc_excludes_top_choices_above_cutoff() {
  let lp = Array::from_slice::<f32>(
    &[0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()],
    &[1, 4],
  )
  .unwrap();
  let key = mlxrs::ops::random::key(0).unwrap();
  let mut out = sample::apply_xtc(&lp, 1.0, 0.1, &[], &key).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  assert!(v[0].is_infinite() && v[0] < 0.0, "top prob 0.5 excluded");
  assert!(v[1].is_infinite() && v[1] < 0.0, "prob 0.3 excluded");
  assert_eq!(v[2], 0.15f32.ln(), "boundary prob 0.15 kept (strict >)");
  assert_eq!(v[3], 0.05f32.ln(), "tail prob 0.05 kept");
}

/// `xtc_special_tokens` are force-kept even when they would be masked: idx0
/// (prob 0.5, normally excluded with prob=1.0) stays finite when special.
#[test]
fn xtc_special_tokens_are_preserved() {
  let lp = Array::from_slice::<f32>(
    &[0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()],
    &[1, 4],
  )
  .unwrap();
  let key = mlxrs::ops::random::key(0).unwrap();
  let mut out = sample::apply_xtc(&lp, 1.0, 0.1, &[0], &key).unwrap();
  let v = out.to_vec::<f32>().unwrap();
  assert_eq!(v[0], 0.5f32.ln(), "special idx0 kept despite mask");
  assert!(
    v[1].is_infinite() && v[1] < 0.0,
    "non-special idx1 excluded"
  );
}

/// No prob exceeds the threshold ⇒ `where(probs>thr, probs, +inf).min()` is
/// `+inf` ⇒ `probs > +inf` is all-false ⇒ identity (uniform logits, thr 0.4
/// > the uniform prob 0.25), even with the gate forced on (prob=1.0).
#[test]
fn xtc_no_token_above_threshold_is_identity() {
  let lp = Array::from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
  let key = mlxrs::ops::random::key(0).unwrap();
  let mut out = sample::apply_xtc(&lp, 1.0, 0.4, &[], &key).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
}

/// mlx-lm's `apply_xtc` `ValueError` bounds: threshold ∈ [0, 0.5],
/// probability ∈ [0, 1]; non-finite rejected too.
#[test]
fn xtc_invalid_params_error() {
  let lp = logprobs();
  let key = mlxrs::ops::random::key(0).unwrap();
  assert!(
    sample::apply_xtc(&lp, 0.5, 0.6, &[], &key).is_err(),
    "thr>0.5"
  );
  assert!(
    sample::apply_xtc(&lp, 0.5, -0.1, &[], &key).is_err(),
    "thr<0"
  );
  assert!(
    sample::apply_xtc(&lp, 1.5, 0.3, &[], &key).is_err(),
    "prob>1"
  );
  assert!(
    sample::apply_xtc(&lp, f32::NAN, 0.3, &[], &key).is_err(),
    "prob NaN"
  );
  assert!(
    sample::apply_xtc(&lp, 0.5, f32::NAN, &[], &key).is_err(),
    "thr NaN"
  );
  assert!(
    sample::apply_xtc(&lp, 0.5, 0.5, &[], &key).is_ok(),
    "thr==0.5 / prob==0.5 valid"
  );
}

/// `apply_repetition_penalty` (mlx-lm `make_repetition_penalty` closure):
/// `logit<0 → logit*penalty` else `logit/penalty`, only on the given ids.
/// `[2,-4,1,-1]`, ids {0,1}, penalty 2 ⇒ idx0 2/2=1; idx1 (-4)*2=-8;
/// idx2,3 untouched.
#[test]
fn repetition_penalty_sign_aware() {
  let lg = penalty_logits();
  let mut out = sample::apply_repetition_penalty(&lg, &[0, 1], 2.0).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![1.0, -8.0, 1.0, -1.0]);
}

#[test]
fn repetition_penalty_empty_tokens_is_identity_and_bad_penalty_errors() {
  let lg = penalty_logits();
  let mut out = sample::apply_repetition_penalty(&lg, &[], 2.0).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -4.0, 1.0, -1.0]);
  assert!(sample::apply_repetition_penalty(&lg, &[0], -1.0).is_err());
  assert!(sample::apply_repetition_penalty(&lg, &[0], f32::NAN).is_err());
}

/// `apply_presence_penalty` (mlx-lm `make_presence_penalty`): subtract once
/// per *present* id. `[2,-4,1,-1]`, ids {0,2}, p 1.5 ⇒ idx0 0.5, idx2 -0.5.
#[test]
fn presence_penalty_subtracts_once() {
  let lg = penalty_logits();
  let mut out = sample::apply_presence_penalty(&lg, &[0, 2], 1.5).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.5, -4.0, -0.5, -1.0]);
}

/// Duplicate ids ⇒ assignment semantics (mlx-lm `logits[:,tokens]-=p`):
/// penalized exactly *once* (-4 - 1.5 = -5.5), NOT per occurrence.
#[test]
fn presence_penalty_duplicate_ids_penalized_once() {
  let lg = penalty_logits();
  let mut out = sample::apply_presence_penalty(&lg, &[1, 1], 1.5).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -5.5, 1.0, -1.0]);
}

/// `apply_frequency_penalty` (mlx-lm `make_frequency_penalty`; mlx-swift
/// histogram form): subtract `penalty * occurrence_count`. ids {1,1,2},
/// p 0.5 ⇒ idx1 -4 - 0.5*2 = -5; idx2 1 - 0.5*1 = 0.5; idx0,3 untouched.
#[test]
fn frequency_penalty_scales_with_count() {
  let lg = penalty_logits();
  let mut out = sample::apply_frequency_penalty(&lg, &[1, 1, 2], 0.5).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -5.0, 0.5, -1.0]);
}

#[test]
fn frequency_penalty_empty_tokens_is_identity() {
  let lg = penalty_logits();
  let mut out = sample::apply_frequency_penalty(&lg, &[], 0.5).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -4.0, 1.0, -1.0]);
}

/// `apply_logit_bias` (mlx-lm inline `logit_bias_processor`): additive at the
/// given indices. `[2,-4,1,-1]` + {0:+1, 3:-2} ⇒ idx0 3, idx3 -3.
#[test]
fn logit_bias_adds_at_indices() {
  let lg = penalty_logits();
  let bv = Array::from_slice::<f32>(&[1.0, -2.0], &[2]).unwrap();
  let mut out = sample::apply_logit_bias(&lg, &[0, 3], &bv).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![3.0, -4.0, 1.0, -3.0]);
}

/// Duplicate bias indices **accumulate** (mlx `.at[].add` semantics, the
/// key difference vs the assignment-based penalties): idx2 1 + (1.0+0.5).
#[test]
fn logit_bias_duplicate_indices_accumulate() {
  let lg = penalty_logits();
  let bv = Array::from_slice::<f32>(&[1.0, 0.5], &[2]).unwrap();
  let mut out = sample::apply_logit_bias(&lg, &[2, 2], &bv).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -4.0, 2.5, -1.0]);
}

#[test]
fn logit_bias_length_mismatch_errors_and_empty_is_identity() {
  let lg = penalty_logits();
  let bv = Array::from_slice::<f32>(&[1.0, -2.0], &[2]).unwrap();
  assert!(
    sample::apply_logit_bias(&lg, &[0, 1, 2], &bv).is_err(),
    "3 indices vs 2 values"
  );
  // Empty indices with NON-empty values is a length mismatch, not a no-op:
  // the empty short-circuit must not mask it (Copilot #29 review).
  assert!(
    sample::apply_logit_bias(&lg, &[], &bv).is_err(),
    "0 indices vs 2 values must error, not silently drop the bias"
  );
  // Genuinely empty (0 indices, 0 values) is the identity.
  let empty = Array::from_slice::<f32>(&[], &[0i32]).unwrap();
  let mut out = sample::apply_logit_bias(&lg, &[], &empty).unwrap();
  assert_eq!(out.to_vec::<f32>().unwrap(), vec![2.0, -4.0, 1.0, -1.0]);
}

/// f16/bf16 dtype + numeric fidelity for every new transform (mirrors
/// `half_and_bfloat_preserve_dtype_and_mask`): the output dtype must equal
/// the input dtype AND the in-dtype math must still produce the f32 result
/// (separations chosen wide enough to be exact in half precision).
#[test]
fn new_transforms_preserve_half_and_bfloat_dtype() {
  for dt in [Dtype::F16, Dtype::BF16] {
    let lg = penalty_logits().astype(dt).unwrap();

    let rep = sample::apply_repetition_penalty(&lg, &[0, 1], 2.0).unwrap();
    assert_eq!(rep.dtype().unwrap(), dt, "rep preserves {dt:?}");
    assert_eq!(vals(&rep, dt), vec![1.0, -8.0, 1.0, -1.0], "{dt:?} rep");

    let pres = sample::apply_presence_penalty(&lg, &[0, 2], 1.5).unwrap();
    assert_eq!(pres.dtype().unwrap(), dt, "pres preserves {dt:?}");
    assert_eq!(vals(&pres, dt), vec![0.5, -4.0, -0.5, -1.0], "{dt:?} pres");

    let freq = sample::apply_frequency_penalty(&lg, &[1, 1, 2], 0.5).unwrap();
    assert_eq!(freq.dtype().unwrap(), dt, "freq preserves {dt:?}");
    assert_eq!(vals(&freq, dt), vec![2.0, -5.0, 0.5, -1.0], "{dt:?} freq");

    let bv = Array::from_slice::<f32>(&[1.0, -2.0], &[2])
      .unwrap()
      .astype(dt)
      .unwrap();
    let bias = sample::apply_logit_bias(&lg, &[0, 3], &bv).unwrap();
    assert_eq!(bias.dtype().unwrap(), dt, "bias preserves {dt:?}");
    assert_eq!(vals(&bias, dt), vec![3.0, -4.0, 1.0, -3.0], "{dt:?} bias");

    // XTC in half precision: probs [0.5,0.3,0.15,0.05] separations are wide
    // enough that the f16/bf16 cutoff still excludes idx0/idx1 only.
    let xl = Array::from_slice::<f32>(
      &[0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()],
      &[1, 4],
    )
    .unwrap()
    .astype(dt)
    .unwrap();
    let key = mlxrs::ops::random::key(0).unwrap();
    let xtc = sample::apply_xtc(&xl, 1.0, 0.1, &[], &key).unwrap();
    assert_eq!(xtc.dtype().unwrap(), dt, "xtc preserves {dt:?}");
    let xv = vals(&xtc, dt);
    assert!(xv[0].is_infinite() && xv[0] < 0.0, "{dt:?} xtc excl idx0");
    assert!(xv[1].is_infinite() && xv[1] < 0.0, "{dt:?} xtc excl idx1");
    assert!(
      xv[2].is_finite() && xv[3].is_finite(),
      "{dt:?} xtc keeps tail"
    );
  }
}

/// Codex review (#29): `apply_frequency_penalty` must NOT corrupt untouched
/// logits when a large `penalty` overflows a low-precision dtype. The prior
/// `histogram * scalar` form computed `0 * +inf = NaN` on every unmentioned
/// token for f16; mlx-lm's `.at[:, tokens].subtract(penalty)` only updates
/// the selected tokens. f16 `[2,-4,1,-1]`, ids=[1], penalty=70000 (overflows
/// f16) → expect `[2, -inf, 1, -1]`: untouched cols bit-exact, NOT NaN.
#[test]
fn apply_frequency_penalty_f16_large_penalty_no_nan_bleed() {
  let lp = Array::from_slice::<f32>(&[2.0, -4.0, 1.0, -1.0], &(1, 4))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let out = sample::apply_frequency_penalty(&lp, &[1], 70000.0).unwrap();
  assert_eq!(out.dtype().unwrap(), Dtype::F16, "dtype preserved");
  let v = vals(&out, Dtype::F16);
  assert_eq!(v[0], 2.0, "untouched col 0 bit-exact, not NaN: {v:?}");
  assert!(
    v[1].is_infinite() && v[1] < 0.0,
    "selected col 1 suppressed: {v:?}"
  );
  assert_eq!(v[2], 1.0, "untouched col 2 bit-exact, not NaN: {v:?}");
  assert_eq!(v[3], -1.0, "untouched col 3 bit-exact, not NaN: {v:?}");
  // f32 sanity: a duplicate id accumulates `-penalty * count` (matches
  // mlx-lm `.at[].subtract` repeated-index accumulation); others unchanged.
  let f = Array::from_slice::<f32>(&[10.0, 20.0, 30.0], &(1, 3)).unwrap();
  let of = sample::apply_frequency_penalty(&f, &[2, 2], 5.0).unwrap();
  assert_eq!(
    vals(&of, Dtype::F32),
    vec![10.0, 20.0, 20.0],
    "id 2 twice → -10; others exact"
  );
  // Codex #29 round-2: untouched columns must be the BITWISE-identical input
  // (direct indexed scatter-add performs no arithmetic on them) — IEEE
  // signed zero is the witness: an untouched `-0.0` must NOT become `+0.0`
  // (a global `logits + delta` would canonicalize it). f16 `[-0.0, 5.0]`,
  // id=[1], penalty=70000 → col 0 stays raw `-0.0`, col 1 suppressed.
  let sz = Array::from_slice::<f32>(&[-0.0, 5.0], &(1, 2))
    .unwrap()
    .astype(Dtype::F16)
    .unwrap();
  let so = sample::apply_frequency_penalty(&sz, &[1], 70000.0).unwrap();
  let sv = vals(&so, Dtype::F16);
  assert_eq!(
    sv[0].to_bits(),
    (-0.0_f32).to_bits(),
    "untouched -0.0 must stay raw -0.0 (no signed-zero canonicalization): {sv:?}"
  );
  assert!(
    sv[1].is_infinite() && sv[1] < 0.0,
    "selected col suppressed: {sv:?}"
  );
}

// ---------------------------------------------------------------------------
// LM-6 R2 (Codex adversarial review) — `scale_logits_by_temp` follow-ups
// ---------------------------------------------------------------------------

/// LM-6 R2 (high): `scale_logits_by_temp` MUST install the error handler
/// BEFORE any `Array::full` ctor — `Array::full::<f32>` runs the fallible
/// `mlx_array_new_float32` ctor before its `mlx_full(default_stream())`
/// would lazily install the handler. With the eager `#[ctor]` stripped, a
/// ctor-stripped first sampling call could otherwise reach mlx-c with no
/// handler installed → its default `printf + exit(-1)` instead of a
/// recoverable `Err`. This is a STRUCTURAL test (reads sample.rs at
/// compile time via `CARGO_MANIFEST_DIR` and asserts the install call is
/// the first executable line of the function body), because the failure
/// mode (process termination on scalar allocation failure) is impossible
/// to exercise from a runtime test without a real allocation failure.
#[test]
fn scale_logits_by_temp_ensures_handler_installed_r2_structural() {
  let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lm/sample.rs");
  let src = std::fs::read_to_string(&path).expect("read sample.rs");

  // Locate the function body. The signature line is unique in the file.
  let sig = "pub fn scale_logits_by_temp(logits: &Array, temp: f32) -> Result<Array> {";
  let sig_idx = src
    .find(sig)
    .expect("scale_logits_by_temp signature must exist in sample.rs");
  // Body starts after the `{` of the signature line.
  let body_start = sig_idx + sig.len();

  // Slice from body start, then collect non-blank, non-comment lines until
  // we have the first 3 executable statements.
  let body = &src[body_start..];
  let mut executable: Vec<&str> = Vec::new();
  for raw in body.lines() {
    if executable.len() >= 3 {
      break;
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
      continue;
    }
    if trimmed.starts_with("//") {
      continue;
    }
    executable.push(trimmed);
  }
  assert!(
    !executable.is_empty(),
    "scale_logits_by_temp body must contain at least one executable line"
  );
  // The very first executable line must be the handler install.
  assert_eq!(
    executable[0], "crate::error::ensure_handler_installed();",
    "scale_logits_by_temp MUST call ensure_handler_installed() as the FIRST \
     executable statement (LM-6 R2 high finding); first 3 executable lines were: {executable:?}"
  );
}

/// LM-6 R2 (medium): F64 logits MUST be rejected — the prior non-F32
/// branch silently funneled them through `astype(F32) → divide →
/// astype(F64)`, losing precision on near-tied f64 logits before the
/// Gumbel draw while still returning an F64 array downstream. A "native
/// F64 divide" alternative is not available here because MLX's GPU
/// stream (which `ops::arithmetic::divide` routes through) does not
/// implement float64 — eval would error with `"float64 is not supported
/// on the GPU"`. Rejecting up front is bit-honest about the backend's
/// actual F64 capability AND surfaces the precision-loss bug the prior
/// implicit-roundtrip path masked. This test verifies (a) the error
/// path fires (b) the message mentions F64 + tells the caller to cast,
/// and (c) the input that would have silently lost precision under
/// the old path is the exact one this rejection protects.
#[test]
fn scale_logits_by_temp_rejects_f64() {
  // 10 logits separated by exactly 1e-9 at the f64 level — well below
  // f32's ~1.19e-7 epsilon at magnitude 1.0, so an f32 roundtrip would
  // collide consecutive entries to the same value and destroy the
  // strict-monotonic input ordering. Use base = 1.0 (exactly
  // representable in both f32 and f64) so the only precision loss
  // would be the per-step delta. This is the exact construction the
  // prior implicit f32-roundtrip path corrupted; rejection protects
  // the caller from that silent corruption.
  let base: f64 = 1.0;
  let step: f64 = 1e-9;
  let input: Vec<f64> = (0..10).map(|i| base + (i as f64) * step).collect();
  // Sanity: the input is strictly monotonic at the f64 level, but its
  // f32 cast collapses adjacent values (demonstrating the precision
  // loss the prior implementation hid).
  for w in input.windows(2) {
    assert!(w[0] < w[1], "test premise: input strictly monotonic in f64");
  }
  let f32_roundtripped: Vec<f64> = input.iter().map(|x| *x as f32 as f64).collect();
  assert!(
    f32_roundtripped.windows(2).any(|w| w[0] == w[1]),
    "test premise: f32 roundtrip collapses some adjacent f64 values to equal — \
     this is the silent precision loss the F64 rejection protects against; \
     got {f32_roundtripped:?}"
  );

  let lp = Array::from_slice::<f64>(&input, &[1, 10]).unwrap();
  assert_eq!(lp.dtype().unwrap(), Dtype::F64);
  let err = sample::scale_logits_by_temp(&lp, 0.5).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("F64"),
    "F64 rejection message must mention F64 explicitly: {msg}"
  );
  assert!(
    msg.contains("astype") || msg.contains("Cast"),
    "F64 rejection message must tell the caller to cast: {msg}"
  );
  // And the public categorical_sampling entry surfaces the same error
  // — the rejection is the entire categorical-sampling surface, not
  // just the helper.
  let key = mlxrs::ops::random::key(0).unwrap();
  assert!(
    sample::categorical_sampling(&lp, 0.5, &key).is_err(),
    "categorical_sampling on F64 must error via scale_logits_by_temp"
  );
}

/// LM-6 R2 (medium): integer / boolean logits must be REJECTED (the prior
/// branch treated every non-F32 dtype as a half type and would silently
/// astype through f32). Mirrors the dtype-rejection pattern in
/// `kl_div_loss` — the message must mention floating-point and name the
/// rejected dtype so the caller knows what to cast to.
#[test]
fn scale_logits_by_temp_rejects_integer_dtype() {
  let lp = Array::from_slice::<i32>(&[1, 2, 3, 4], &[1, 4]).unwrap();
  assert_eq!(lp.dtype().unwrap(), Dtype::I32);
  let err = sample::scale_logits_by_temp(&lp, 0.8).unwrap_err();
  let msg = format!("{err}");
  assert!(
    msg.contains("floating-point"),
    "i32 rejection message must mention floating-point: {msg}"
  );
  assert!(
    msg.contains("I32"),
    "i32 rejection message must name the rejected dtype: {msg}"
  );
  // And via the public categorical_sampling entry — the same rejection
  // propagates.
  let key = mlxrs::ops::random::key(0).unwrap();
  assert!(
    sample::categorical_sampling(&lp, 0.8, &key).is_err(),
    "categorical_sampling on i32 logits must error via scale_logits_by_temp"
  );
}
