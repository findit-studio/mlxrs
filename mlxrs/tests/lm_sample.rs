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
