//! Real-device integration tests for `mlxrs::lm::tuner::losses`.
//!
//! These exercise the full Metal kernel + custom_vjp pipeline for the two
//! distillation losses (`kl_div_loss` / `js_div_loss`). Each test below
//! requires a Metal-capable GPU at apply time, so they are gated behind
//! `#[cfg(target_os = "macos")] #[ignore]`. Run locally with:
//!
//!     CARGO_TARGET_DIR=/tmp/mlxrs-a5-iso \
//!     cargo +nightly test -p mlxrs --features lm --test losses \
//!         -- --ignored --test-threads=1
//!
//! Tests:
//!
//! - `kl_div_loss_forward_known_input_matches_python_ref` — `[2, 4]`
//!   logits_q + logits_p, computes a hand-derived KL reference value
//!   (`sum_i p_i * (logp_i - logq_i)`) and asserts the kernel output
//!   matches within an f32 epsilon.
//! - `kl_div_loss_backward_via_vjp_matches_python_ref` — wraps
//!   `kl_div_loss` in `transforms::vjp`, computes the gradient w.r.t.
//!   `logits_q` with a unit cotangent, asserts it equals the analytic
//!   gradient `q_i - p_i`.
//! - `js_div_loss_forward_known_input_matches_python_ref` — `[2, 4]`
//!   logits, hand-derives the JS divergence
//!   `0.5*(KL(p||m) + KL(q||m))`, asserts kernel output matches.
//! - `js_div_loss_backward_via_vjp_matches_python_ref` — wraps
//!   `js_div_loss` in `transforms::vjp` and compares the gradient to a
//!   reference numerical-difference computation.
//! - `kl_div_loss_handles_zero_logits_without_nan` — all-zero `[1, 4]`
//!   logits should yield a finite KL of `0` (both distributions are
//!   uniform), no NaN.
//! - `kl_forward_kernel_lazy_init_compiles_once` — calls `kl_div_loss`
//!   twice and asserts both return finite, equal values; the actual "compiled
//!   once" property is structural (the thread_local OnceCell guarantees it),
//!   the test pins it by exercising the path through the cache.
//! - `kl_div_loss_real_device_shape_mismatch` — sanity-check that the
//!   shape-mismatch error path is reachable via the public API even on
//!   a Metal device (no kernel launch).
//!
//! All numeric assertions use a generous f32 tolerance because the Metal
//! kernel uses `metal::fast::exp` / `metal::fast::log` which trade a few
//! ULPs of accuracy for speed (the python reference uses the same).

#![cfg(all(target_os = "macos", feature = "lm"))]

use mlxrs::{
  Array, Error,
  lm::tuner::losses::{js_div_loss, kl_div_loss},
  transforms::{grad, vjp},
};

// ─────────────────── Contract rejection tests (no GPU needed) ───────────────────
//
// These tests assert wrapper-layer rejections that happen BEFORE any FFI
// kernel allocation, so they are non-ignored and run in routine CI. They
// pin the public API contract documented on `kl_div_loss` / `js_div_loss`:
// rank must be `>= 2` and dtype must be one of `F32` / `F16` / `BF16`.

/// F1 contract: rank-1 logits `[V]` are rejected with a clear contract
/// message. The shared `MetalKernel` wrapper would otherwise reject the
/// empty `leading_shape` output (`custom Metal kernel outputs must have
/// rank >= 1`); we surface a more useful message at the boundary.
#[test]
fn kl_div_loss_rejects_rank_1_input() {
  let logits_q = Array::ones::<f32>(&[4]).unwrap();
  let logits_p = Array::ones::<f32>(&[4]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 1);
      assert_eq!(p.actual_shape(), &[4]);
      assert!(
        p.context().contains("kl_div_loss"),
        "context names the loss site: {}",
        p.context()
      );
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

/// F1 contract for `js_div_loss` (mirrors `kl_div_loss_rejects_rank_1_input`).
#[test]
fn js_div_loss_rejects_rank_1_input() {
  let logits_q = Array::ones::<f32>(&[4]).unwrap();
  let logits_p = Array::ones::<f32>(&[4]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 1);
      assert_eq!(p.actual_shape(), &[4]);
      assert!(
        p.context().contains("js_div_loss"),
        "context names the loss site: {}",
        p.context()
      );
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

/// F2 contract: integer dtype (`I32`) logits are rejected. Same-dtype
/// integer logits would otherwise be admitted by the pre-fix dtype-equality
/// check and the kernel would silently truncate KL / JS to integer
/// arithmetic. This must reject explicitly at the wrapper boundary.
#[test]
fn kl_div_loss_rejects_integer_dtype() {
  let logits_q = Array::ones::<i32>(&[1, 4]).unwrap();
  let logits_p = Array::ones::<i32>(&[1, 4]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::UnsupportedDtype(p) => {
      use mlxrs::Dtype;
      assert_eq!(p.dtype(), Dtype::I32);
      assert_eq!(p.supported(), &[Dtype::F32, Dtype::F16, Dtype::BF16]);
    }
    other => panic!("expected Error::UnsupportedDtype, got: {other:?}"),
  }
}

/// F2 contract: boolean dtype logits are rejected.
#[test]
fn kl_div_loss_rejects_bool_dtype() {
  let logits_q = Array::ones::<bool>(&[1, 4]).unwrap();
  let logits_p = Array::ones::<bool>(&[1, 4]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::UnsupportedDtype(p) => {
      use mlxrs::Dtype;
      assert_eq!(p.dtype(), Dtype::Bool);
      assert_eq!(p.supported(), &[Dtype::F32, Dtype::F16, Dtype::BF16]);
    }
    other => panic!("expected Error::UnsupportedDtype, got: {other:?}"),
  }
}

/// F2 contract for `js_div_loss`: integer dtype rejection.
#[test]
fn js_div_loss_rejects_integer_dtype() {
  let logits_q = Array::ones::<i32>(&[1, 4]).unwrap();
  let logits_p = Array::ones::<i32>(&[1, 4]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::UnsupportedDtype(p) => {
      use mlxrs::Dtype;
      assert_eq!(p.dtype(), Dtype::I32);
      assert_eq!(p.supported(), &[Dtype::F32, Dtype::F16, Dtype::BF16]);
    }
    other => panic!("expected Error::UnsupportedDtype, got: {other:?}"),
  }
}

/// F2 contract for `js_div_loss`: boolean dtype rejection.
#[test]
fn js_div_loss_rejects_bool_dtype() {
  let logits_q = Array::ones::<bool>(&[1, 4]).unwrap();
  let logits_p = Array::ones::<bool>(&[1, 4]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::UnsupportedDtype(p) => {
      use mlxrs::Dtype;
      assert_eq!(p.dtype(), Dtype::Bool);
      assert_eq!(p.supported(), &[Dtype::F32, Dtype::F16, Dtype::BF16]);
    }
    other => panic!("expected Error::UnsupportedDtype, got: {other:?}"),
  }
}

// ─── R2 contract: rank-first precedence (rank check runs BEFORE shape) ───
//
// `validate_inputs` enforces `ndim() >= 2` on BOTH inputs BEFORE comparing
// shapes, so a rank-0 pair and a mismatched-rank pair both surface the
// rank-rejection guidance (Error::Backend, "rank >= 2") instead of a
// typed RankMismatch. These tests pin that precedence rule for both
// losses (4 tests = 2 cases × 2 losses).

/// R2: rank-0 (scalar) inputs are rejected with the rank message — NOT
/// silently accepted, NOT routed through the shape-mismatch path.
#[test]
fn kl_div_loss_rejects_rank_0_input() {
  let logits_q = Array::ones::<f32>(&[]).unwrap();
  let logits_p = Array::ones::<f32>(&[]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 0);
      assert!(
        p.context().contains("kl_div_loss"),
        "context names the loss site: {}",
        p.context()
      );
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

/// R2: rank-1 `logits_q` paired with rank-2 `logits_p` is mismatched in
/// BOTH rank and shape. Asserts rank-first precedence: the returned error
/// is the rank-rejection (`Error::RankMismatch`), NOT a shape compare —
/// proving the rank check fires before shape comparison.
#[test]
fn kl_div_loss_rejects_rank_1_vs_rank_2_mismatch() {
  let logits_q = Array::ones::<f32>(&[4]).unwrap();
  let logits_p = Array::ones::<f32>(&[1, 4]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(
        p.actual(),
        1,
        "rank-first precedence: q (rank 1) reported first"
      );
    }
    Error::ShapePairMismatch(p) => panic!(
      "rank-first precedence violated: got ShapePairMismatch (expected={:?}, actual={:?}) \
       instead of rank rejection",
      p.expected(),
      p.actual()
    ),
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

/// R2 mirror for `js_div_loss`: rank-0 inputs rejected with the rank message.
#[test]
fn js_div_loss_rejects_rank_0_input() {
  let logits_q = Array::ones::<f32>(&[]).unwrap();
  let logits_p = Array::ones::<f32>(&[]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 0);
      assert!(
        p.context().contains("js_div_loss"),
        "context names the loss site: {}",
        p.context()
      );
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

/// R2 mirror for `js_div_loss`: rank-1 vs rank-2 surfaces the rank error,
/// not shape comparison (proves rank-first precedence).
#[test]
fn js_div_loss_rejects_rank_1_vs_rank_2_mismatch() {
  let logits_q = Array::ones::<f32>(&[4]).unwrap();
  let logits_p = Array::ones::<f32>(&[1, 4]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(
        p.actual(),
        1,
        "rank-first precedence: q (rank 1) reported first"
      );
    }
    Error::ShapePairMismatch(p) => panic!(
      "rank-first precedence violated: got ShapePairMismatch (expected={:?}, actual={:?}) \
       instead of rank rejection",
      p.expected(),
      p.actual()
    ),
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
}

// ─── R3 contract: shape-class errors precede dtype errors ───
//
// The documented `# Errors` precedence on `kl_div_loss` / `js_div_loss`
// promises step 2 (typed shape error — incl. zero-last-dim + i32-overflow)
// fires BEFORE step 3 (DtypeMismatch) and step 4 (dtype-admissibility
// Backend). Pre-fix, the zero-last-dim rejection only fired inside
// `n_outs_of` AFTER `validate_inputs` returned successfully, so dtype
// errors won the precedence race for shape `[1, 0]` inputs. These 4
// tests pin the corrected precedence: zero-last-dim is reported as
// OutOfRange even when the dtype is unsupported (would otherwise
// yield Backend) or mismatched between q and p (would otherwise yield
// DtypeMismatch). Zero-sized rank-2 arrays are valid mlx constructs
// (see `tests/error_paths.rs::from_slice_zero_element_uses_sentinel`),
// so this path is reachable from the safe API.

/// R3: shape `[1, 0]` with same unsupported dtype (`i32`) — the
/// pre-fix code would return `Error::Backend` (dtype admissibility,
/// step 4). After moving the zero-last-dim check into `validate_inputs`
/// before the dtype checks, the result is `Error::OutOfRange`
/// (step 2), matching the documented contract.
#[test]
fn kl_div_loss_rejects_zero_last_dim_before_unsupported_dtype() {
  let logits_q = Array::from_slice::<i32>(&[], &[1i32, 0]).unwrap();
  let logits_p = Array::from_slice::<i32>(&[], &[1i32, 0]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert!(
        p.context().contains("last dimension"),
        "context names the last-dim check: {}",
        p.context()
      );
      assert_eq!(p.value(), "0");
      assert_eq!(p.requirement(), "must be > 0");
    }
    Error::UnsupportedDtype(p) => panic!(
      "precedence violated: got dtype-admissibility UnsupportedDtype({:?}) instead of \
       OutOfRange for zero-last-dim `i32` input",
      p.dtype()
    ),
    other => panic!("expected Error::OutOfRange, got: {other:?}"),
  }
}

/// R3: shape `[1, 0]` with mismatched dtypes (`f32` vs `f16`) — the
/// pre-fix code would return `Error::DtypeMismatch` (step 3). After
/// the precedence move, the result is `Error::OutOfRange` (step 2).
#[test]
fn kl_div_loss_rejects_zero_last_dim_before_mixed_dtype() {
  let logits_q = Array::from_slice::<f32>(&[], &[1i32, 0]).unwrap();
  let logits_p = Array::from_slice::<half::f16>(&[], &[1i32, 0]).unwrap();
  let err = kl_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert!(
        p.context().contains("last dimension"),
        "context names the last-dim check: {}",
        p.context()
      );
      assert_eq!(p.value(), "0");
    }
    Error::DtypeMismatch(p) => panic!(
      "precedence violated: got DtypeMismatch (expected {:?}, got {:?}) \
       instead of OutOfRange for zero-last-dim mixed-dtype input",
      p.expected(),
      p.got()
    ),
    other => panic!("expected Error::OutOfRange, got: {other:?}"),
  }
}

/// R3 mirror for `js_div_loss`: unsupported-dtype + zero-last-dim — the
/// pre-fix code returned `Error::Backend` (dtype admissibility); after
/// the precedence move it returns `Error::OutOfRange`.
#[test]
fn js_div_loss_rejects_zero_last_dim_before_unsupported_dtype() {
  let logits_q = Array::from_slice::<i32>(&[], &[1i32, 0]).unwrap();
  let logits_p = Array::from_slice::<i32>(&[], &[1i32, 0]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert!(
        p.context().contains("last dimension"),
        "context names the last-dim check: {}",
        p.context()
      );
      assert_eq!(p.value(), "0");
    }
    Error::UnsupportedDtype(p) => panic!(
      "precedence violated: got dtype-admissibility UnsupportedDtype({:?}) instead of \
       OutOfRange for zero-last-dim `i32` input",
      p.dtype()
    ),
    other => panic!("expected Error::OutOfRange, got: {other:?}"),
  }
}

/// R3 mirror for `js_div_loss`: mixed-dtype + zero-last-dim — the
/// pre-fix code returned `Error::DtypeMismatch`; after the precedence
/// move it returns `Error::OutOfRange`.
#[test]
fn js_div_loss_rejects_zero_last_dim_before_mixed_dtype() {
  let logits_q = Array::from_slice::<f32>(&[], &[1i32, 0]).unwrap();
  let logits_p = Array::from_slice::<half::f16>(&[], &[1i32, 0]).unwrap();
  let err = js_div_loss(&logits_q, &logits_p).unwrap_err();
  match err {
    Error::OutOfRange(p) => {
      assert!(
        p.context().contains("last dimension"),
        "context names the last-dim check: {}",
        p.context()
      );
      assert_eq!(p.value(), "0");
    }
    Error::DtypeMismatch(p) => panic!(
      "precedence violated: got DtypeMismatch (expected {:?}, got {:?}) \
       instead of OutOfRange for zero-last-dim mixed-dtype input",
      p.expected(),
      p.got()
    ),
    other => panic!("expected Error::OutOfRange, got: {other:?}"),
  }
}

/// Hand-derive `KL(p || q) = sum_i softmax(logp)_i * (logsoftmax(logp)_i - logsoftmax(logq)_i)`
/// over a single row (rank-1 input slice).
fn reference_kl_row(logp_row: &[f32], logq_row: &[f32]) -> f32 {
  fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse: f32 = m + row.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    row.iter().map(|&x| x - lse).collect()
  }
  let lp = log_softmax(logp_row);
  let lq = log_softmax(logq_row);
  let p: Vec<f32> = lp.iter().map(|&x| x.exp()).collect();
  p.iter()
    .zip(lp.iter())
    .zip(lq.iter())
    .map(|((&pi, &lpi), &lqi)| pi * (lpi - lqi))
    .sum()
}

/// Hand-derive `JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)` where `m`
/// is the average of the two softmax distributions. Numerically stable form:
/// uses log-domain throughout.
fn reference_js_row(logp_row: &[f32], logq_row: &[f32]) -> f32 {
  fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse: f32 = m + row.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    row.iter().map(|&x| x - lse).collect()
  }
  let lp = log_softmax(logp_row);
  let lq = log_softmax(logq_row);
  // logm_i = log(0.5 * (exp(lp_i) + exp(lq_i)))
  //        = logsumexp(lp_i, lq_i) - ln(2)
  let log2 = 2f32.ln();
  let logm: Vec<f32> = lp
    .iter()
    .zip(lq.iter())
    .map(|(&a, &b)| {
      let m = a.max(b);
      m + ((a - m).exp() + (b - m).exp()).ln() - log2
    })
    .collect();
  let p: Vec<f32> = lp.iter().map(|&x| x.exp()).collect();
  let q: Vec<f32> = lq.iter().map(|&x| x.exp()).collect();
  let kl_pm: f32 = p
    .iter()
    .zip(lp.iter())
    .zip(logm.iter())
    .map(|((&pi, &lpi), &lmi)| pi * (lpi - lmi))
    .sum();
  let kl_qm: f32 = q
    .iter()
    .zip(lq.iter())
    .zip(logm.iter())
    .map(|((&qi, &lqi), &lmi)| qi * (lqi - lmi))
    .sum();
  0.5 * kl_pm + 0.5 * kl_qm
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_div_loss_forward_known_input_matches_python_ref() {
  // [2, 4] logits: two batched rows.
  let logits_q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.0, -0.5, -1.0];
  let logits_p_data: Vec<f32> = vec![2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
  let logits_q = Array::from_slice::<f32>(&logits_q_data, &[2, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&logits_p_data, &[2, 4]).unwrap();

  let mut loss = kl_div_loss(&logits_q, &logits_p).unwrap();
  // Output shape is logits_q.shape[:-1] = [2].
  assert_eq!(loss.shape(), vec![2]);
  let got: Vec<f32> = loss.to_vec().unwrap();

  let expected: Vec<f32> = vec![
    reference_kl_row(&logits_p_data[0..4], &logits_q_data[0..4]),
    reference_kl_row(&logits_p_data[4..8], &logits_q_data[4..8]),
  ];

  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    let abs_err = (g - e).abs();
    let rel = abs_err / e.abs().max(1e-6);
    assert!(
      abs_err < 1e-4 || rel < 1e-4,
      "row {i}: got {g}, expected {e}, abs_err={abs_err}, rel_err={rel}"
    );
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_div_loss_backward_via_vjp_matches_python_ref() {
  // Same fixture; analytic gradient w.r.t. logits_q is `softmax(q) - softmax(p)`
  // multiplied by the cotangent (we pass a unit cotangent so it's just
  // `q_i - p_i` in probability space).
  let logits_q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.0, -0.5, -1.0];
  let logits_p_data: Vec<f32> = vec![2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
  let logits_q = Array::from_slice::<f32>(&logits_q_data, &[2, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&logits_p_data, &[2, 4]).unwrap();
  // cotangent matches the [2] output shape.
  let cotangent = Array::from_slice::<f32>(&[1.0, 1.0], &[2]).unwrap();

  let (mut values, mut grads) = vjp(
    |xs| Ok(vec![kl_div_loss(&xs[0], &xs[1])?]),
    &[logits_q, logits_p],
    &[cotangent],
  )
  .unwrap();
  assert_eq!(values.len(), 1);
  assert_eq!(grads.len(), 2);
  // grads[0] is dq, shape [2, 4]; grads[1] is dp, shape [2, 4] but all zeros.
  assert_eq!(grads[0].shape(), vec![2, 4]);
  assert_eq!(grads[1].shape(), vec![2, 4]);

  let got_dq: Vec<f32> = grads[0].to_vec().unwrap();
  let got_dp: Vec<f32> = grads[1].to_vec().unwrap();

  fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
  }
  let expected_dq_r0: Vec<f32> = softmax(&logits_q_data[0..4])
    .iter()
    .zip(softmax(&logits_p_data[0..4]).iter())
    .map(|(&q, &p)| q - p)
    .collect();
  let expected_dq_r1: Vec<f32> = softmax(&logits_q_data[4..8])
    .iter()
    .zip(softmax(&logits_p_data[4..8]).iter())
    .map(|(&q, &p)| q - p)
    .collect();
  let expected: Vec<f32> = [&expected_dq_r0[..], &expected_dq_r1[..]].concat();

  for (i, (g, e)) in got_dq.iter().zip(expected.iter()).enumerate() {
    let abs_err = (g - e).abs();
    assert!(
      abs_err < 1e-4,
      "dq[{i}]: got {g}, expected {e}, err {abs_err}"
    );
  }

  // dp must be identically zero (python ref uses mx.zeros_like).
  for (i, &v) in got_dp.iter().enumerate() {
    assert_eq!(v, 0.0, "dp[{i}] should be zero, got {v}");
  }

  // Also verify the value half matches the forward.
  let v_got: Vec<f32> = values[0].to_vec().unwrap();
  let expected_loss: Vec<f32> = vec![
    reference_kl_row(&logits_p_data[0..4], &logits_q_data[0..4]),
    reference_kl_row(&logits_p_data[4..8], &logits_q_data[4..8]),
  ];
  for (i, (g, e)) in v_got.iter().zip(expected_loss.iter()).enumerate() {
    let abs_err = (g - e).abs();
    let rel = abs_err / e.abs().max(1e-6);
    assert!(
      abs_err < 1e-4 || rel < 1e-4,
      "value[{i}]: got {g}, expected {e}"
    );
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn js_div_loss_forward_known_input_matches_python_ref() {
  let logits_q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.0, -0.5, -1.0];
  let logits_p_data: Vec<f32> = vec![2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
  let logits_q = Array::from_slice::<f32>(&logits_q_data, &[2, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&logits_p_data, &[2, 4]).unwrap();

  let mut loss = js_div_loss(&logits_q, &logits_p).unwrap();
  assert_eq!(loss.shape(), vec![2]);
  let got: Vec<f32> = loss.to_vec().unwrap();

  let expected: Vec<f32> = vec![
    reference_js_row(&logits_p_data[0..4], &logits_q_data[0..4]),
    reference_js_row(&logits_p_data[4..8], &logits_q_data[4..8]),
  ];

  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    let abs_err = (g - e).abs();
    let rel = abs_err / e.abs().max(1e-6);
    assert!(
      abs_err < 1e-3 || rel < 1e-3,
      "row {i}: got {g}, expected {e}, abs_err={abs_err}, rel_err={rel}"
    );
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn js_div_loss_backward_via_vjp_matches_python_ref() {
  // Verify the backward pass via finite-differences against the forward —
  // the kernel uses a specialized analytic gradient (q_j * (log(2) -
  // log(1 + exp(logp-logq)) - kl_q)) that's tricky to re-derive symbolically,
  // so we numerically approximate `d JS / d logits_q` and check the kernel
  // result matches within the noise of the central-difference scheme.
  let logits_q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
  let logits_p_data: Vec<f32> = vec![2.0, 1.0, 0.0, -1.0];
  let logits_q = Array::from_slice::<f32>(&logits_q_data, &[1, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&logits_p_data, &[1, 4]).unwrap();
  let cotangent = Array::from_slice::<f32>(&[1.0], &[1]).unwrap();

  let (_values, mut grads) = vjp(
    |xs| Ok(vec![js_div_loss(&xs[0], &xs[1])?]),
    &[logits_q, logits_p],
    &[cotangent],
  )
  .unwrap();
  let got_dq: Vec<f32> = grads[0].to_vec().unwrap();
  let got_dp: Vec<f32> = grads[1].to_vec().unwrap();

  // Central differences against the scalar Rust reference.
  let h = 1e-3f32;
  let mut expected_dq = Vec::with_capacity(4);
  for i in 0..4 {
    let mut qp = logits_q_data.clone();
    let mut qm = logits_q_data.clone();
    qp[i] += h;
    qm[i] -= h;
    let fp = reference_js_row(&logits_p_data, &qp);
    let fm = reference_js_row(&logits_p_data, &qm);
    expected_dq.push((fp - fm) / (2.0 * h));
  }
  for (i, (g, e)) in got_dq.iter().zip(expected_dq.iter()).enumerate() {
    let abs_err = (g - e).abs();
    assert!(
      abs_err < 5e-3,
      "dq[{i}]: got {g}, expected {e}, abs_err={abs_err}"
    );
  }
  // dp must be identically zero.
  for (i, &v) in got_dp.iter().enumerate() {
    assert_eq!(v, 0.0, "dp[{i}] should be zero, got {v}");
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_div_loss_handles_zero_logits_without_nan() {
  // All-zero logits → both distributions are uniform → KL = 0 exactly.
  let logits_q = Array::from_slice::<f32>(&[0.0; 4], &[1, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&[0.0; 4], &[1, 4]).unwrap();

  let mut loss = kl_div_loss(&logits_q, &logits_p).unwrap();
  let got: Vec<f32> = loss.to_vec().unwrap();
  assert_eq!(got.len(), 1);
  assert!(got[0].is_finite(), "loss must be finite, got {}", got[0]);
  assert!(got[0].abs() < 1e-5, "loss must be ~0, got {}", got[0]);
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_forward_kernel_lazy_init_compiles_once() {
  // Call kl_div_loss twice on the same thread; both must succeed and return
  // equal results. The structural guarantee that the kernel is compiled
  // exactly once per thread is provided by the thread_local! + OnceCell —
  // this test pins that the cache path is exercised (a regression to a
  // per-call construct would still succeed numerically but lose the caching
  // property the docs promise).
  let logits_q = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&[0.5, 0.5, 0.0, 0.0], &[1, 4]).unwrap();

  let mut a = kl_div_loss(&logits_q, &logits_p).unwrap();
  let mut b = kl_div_loss(&logits_q, &logits_p).unwrap();
  let va: Vec<f32> = a.to_vec().unwrap();
  let vb: Vec<f32> = b.to_vec().unwrap();
  assert_eq!(va, vb);
  assert!(va[0].is_finite());
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_div_loss_real_device_shape_mismatch() {
  // Shape mismatch is checked in the wrapper BEFORE any FFI call — but
  // verify that even with a real device present, the same error surfaces.
  let a = Array::ones::<f32>(&[2, 4]).unwrap();
  let b = Array::ones::<f32>(&[2, 8]).unwrap();
  let err = kl_div_loss(&a, &b).unwrap_err();
  match err {
    Error::ShapePairMismatch(payload) => {
      assert!(
        payload.context().contains("kl_div_loss"),
        "got: {:?}",
        payload.context()
      );
    }
    other => panic!("expected ShapePairMismatch, got: {other:?}"),
  }
}

#[test]
#[ignore = "requires a Metal-capable GPU"]
fn kl_div_loss_grad_via_transforms_grad() {
  // Smoke-test that `transforms::grad` chains cleanly over kl_div_loss,
  // not just `vjp`. `grad` differentiates only argnums[0] (logits_q here),
  // returning just the gradient w.r.t. that primal.
  let logits_q = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
  let logits_p = Array::from_slice::<f32>(&[2.0, 1.0, 0.0, -1.0], &[1, 4]).unwrap();

  // grad expects a scalar output, so sum the loss along the batch axis.
  let g = grad(
    |xs| {
      let loss = kl_div_loss(&xs[0], &xs[1])?;
      let scalar = mlxrs::ops::reduction::sum(&loss, false)?;
      Ok(vec![scalar])
    },
    &[0],
  )
  .unwrap();
  let mut grads = g(&[logits_q, logits_p]).unwrap();
  assert_eq!(grads.len(), 1);
  assert_eq!(grads[0].shape(), vec![1, 4]);
  let got: Vec<f32> = grads[0].to_vec().unwrap();
  // Analytic: softmax(q) - softmax(p) (the cotangent is `1` from the sum).
  fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
  }
  let logits_q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
  let logits_p_data: Vec<f32> = vec![2.0, 1.0, 0.0, -1.0];
  let expected: Vec<f32> = softmax(&logits_q_data)
    .iter()
    .zip(softmax(&logits_p_data).iter())
    .map(|(&q, &p)| q - p)
    .collect();
  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    let abs_err = (g - e).abs();
    assert!(
      abs_err < 1e-4,
      "g[{i}]: got {g}, expected {e}, err {abs_err}"
    );
  }
}
