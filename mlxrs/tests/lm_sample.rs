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
