use super::*;

/// Read a 1-element `Array` as `f32`. `item` is `&mut self`, so clone first.
fn read_scalar(a: &Array) -> Result<f32> {
  let mut clone = a.try_clone()?;
  clone.item::<f32>()
}

// ── Validation rejection tests ────────────────────────────────────────────

#[test]
fn muon_new_rejects_non_finite_momentum() {
  assert!(Muon::new(0.01, f32::NAN, 0.01, true, 5).is_err());
  assert!(Muon::new(0.01, f32::INFINITY, 0.01, true, 5).is_err());
}

#[test]
fn muon_new_rejects_negative_weight_decay() {
  assert!(Muon::new(0.01, 0.95, -0.1, true, 5).is_err());
}

#[test]
fn muon_new_rejects_non_finite_weight_decay() {
  assert!(Muon::new(0.01, 0.95, f32::NAN, true, 5).is_err());
}

#[test]
fn muon_with_momentum_rejects_non_finite() {
  let res = Muon::default_with_lr(0.01).and_then(|m| m.with_momentum(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn muon_with_weight_decay_rejects_negative() {
  let res = Muon::default_with_lr(0.01).and_then(|m| m.with_weight_decay(-0.1));
  assert!(res.is_err());
}

#[test]
fn muon_with_weight_decay_rejects_non_finite() {
  let res = Muon::default_with_lr(0.01).and_then(|m| m.with_weight_decay(f32::INFINITY));
  assert!(res.is_err());
}

#[test]
fn muon_with_learning_rate_rejects_fixed_nan() {
  let res =
    Muon::default_with_lr(0.01).and_then(|m| m.with_learning_rate(LearningRate::Fixed(f32::NAN)));
  assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
}

#[test]
fn muon_1d_param_runs_without_newton_schulz() -> Result<()> {
  // 1D params skip the Newton-Schulz branch (ndim < 2) and reduce to
  // plain momentum SGD (with weight decay).
  let mut muon = Muon::default_with_lr(0.01)?;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3])?,
  );
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3], &[3])?,
  );
  muon.apply_gradients(&grads, &mut params)?;
  let mut got = params["w"].try_clone()?;
  let v: Vec<f32> = got.to_vec()?;
  // First step with default momentum=0.95, nesterov=true, weight_decay=0.01:
  //   g_eff[i] = g[i] + 0.01·w[i]
  //   v = 0.05·g_eff
  //   update = g_eff·0.05 + v·0.95 = 0.05·g_eff + 0.95·(0.05·g_eff)
  //          = 0.05·g_eff·(1 + 0.95) = 0.0975·g_eff
  //   w_new = w - lr·update = w - 0.01·0.0975·g_eff
  //         = w - 0.000975·g_eff
  // For w[0]=1.0, g_eff[0] = 0.1 + 0.01 = 0.11
  //   w_new[0] = 1.0 - 0.000975·0.11 = 0.99989275
  assert!(
    (v[0] - 0.999_892_8).abs() < 1e-5,
    "expected ~0.9998928, got {}",
    v[0]
  );
  Ok(())
}

#[test]
fn muon_2d_param_invokes_newton_schulz_branch() -> Result<()> {
  let mut muon = Muon::new(0.01, 0.0, 0.0, false, 5)?;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2))?,
  );
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.5, 0.0, 0.0, 0.5], &(2, 2))?,
  );
  muon.apply_gradients(&grads, &mut params)?;
  let mut got = params["w"].try_clone()?;
  let v: Vec<f32> = got.to_vec()?;
  // Just verify the update moved (no exact match — Newton-Schulz output
  // depends on the polynomial iteration; the test confirms the branch
  // ran and produced a finite, distinct value).
  assert!(v[0].is_finite() && v[3].is_finite());
  assert!((v[0] - 1.0).abs() > 1e-6 || (v[3] - 1.0).abs() > 1e-6);
  Ok(())
}

// ── Config getters echo constructor inputs ────────────────────────────────

#[test]
fn muon_getters_echo_inputs() -> Result<()> {
  // Distinct non-default values so every getter is observably distinct.
  // Covers `learning_rate_ref`, `momentum`, `weight_decay`, `nesterov`,
  // `ns_steps`, and the trait `step()` / `learning_rate()` on a fresh
  // optimizer (step 0, fixed LR echoed).
  let muon = Muon::new(LearningRate::Fixed(0.25), 0.8, 0.02, false, 7)?;
  assert!(
    muon.learning_rate_ref().is_fixed(),
    "learning_rate_ref must echo the Fixed schedule"
  );
  assert_eq!(muon.momentum(), 0.8);
  assert_eq!(muon.weight_decay(), 0.02);
  assert!(
    !muon.nesterov(),
    "nesterov getter must echo the `false` arm"
  );
  assert_eq!(muon.ns_steps(), 7);
  // Trait methods: pre-increment step is 0; fixed LR resolved at step 0.
  assert_eq!(muon.step(), 0);
  assert_eq!(muon.learning_rate(), 0.25);
  Ok(())
}

#[test]
fn muon_default_with_lr_getters() -> Result<()> {
  // Exercises the Python-default getter values + the `true` `nesterov` arm.
  let muon = Muon::default_with_lr(0.01)?;
  assert_eq!(muon.momentum(), 0.95);
  assert_eq!(muon.weight_decay(), 0.01);
  assert!(muon.nesterov(), "default nesterov is true");
  assert_eq!(muon.ns_steps(), 5);
  assert!(muon.learning_rate_ref().is_fixed());
  Ok(())
}

// ── Builder success paths (echo the set value) ────────────────────────────

#[test]
fn muon_builder_success_paths_echo() -> Result<()> {
  // Each `with_*` success arm must echo its input. `with_learning_rate`'s
  // success arm resolves the fixed value at step 0; `with_nesterov` and
  // `with_ns_steps` are infallible.
  let muon = Muon::default_with_lr(0.01)?
    .with_learning_rate(LearningRate::Fixed(0.05))?
    .with_momentum(0.7)?
    .with_weight_decay(0.2)?
    .with_nesterov(false)
    .with_ns_steps(3);
  assert_eq!(muon.learning_rate(), 0.05);
  assert!(muon.learning_rate_ref().is_fixed());
  assert_eq!(muon.momentum(), 0.7);
  assert_eq!(muon.weight_decay(), 0.2);
  assert!(!muon.nesterov());
  assert_eq!(muon.ns_steps(), 3);
  Ok(())
}

#[test]
fn muon_with_nesterov_toggles_on() -> Result<()> {
  // The infallible setter must also flip false → true.
  let muon = Muon::new(0.01, 0.95, 0.01, false, 5)?.with_nesterov(true);
  assert!(muon.nesterov());
  Ok(())
}

// ── newton_schulz5 rank guard (typed-variant oracle) ──────────────────────

#[test]
fn muon_newton_schulz5_rejects_non_2d() -> Result<()> {
  // The private 2D-only kernel must reject any non-rank-2 input with a
  // typed `RankMismatch` BEFORE the polynomial iteration. Reachable only
  // by calling the method directly (the `apply_gradients` path reshapes
  // >2D down to 2D first); a 1D input is the simplest violation.
  let muon = Muon::default_with_lr(0.01)?;
  let x1 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3])?;
  let err1 = muon.newton_schulz5(&x1, 5).unwrap_err();
  assert!(
    err1.is_rank_mismatch(),
    "1D input must yield RankMismatch, got {err1:?}"
  );
  // A 3D input is also rejected (rank != 2).
  let x3 = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(1, 2, 2))?;
  let err3 = muon.newton_schulz5(&x3, 5).unwrap_err();
  assert!(
    err3.is_rank_mismatch(),
    "3D input must yield RankMismatch, got {err3:?}"
  );
  Ok(())
}

#[test]
fn muon_newton_schulz5_tall_matrix_transposes_and_preserves_shape() -> Result<()> {
  // A "tall" matrix (rows > cols) takes the `transpose_needed` branch: the
  // kernel transposes to a wide matrix, iterates, then transposes back. This
  // is the only test that covers both the transpose-in and transpose-back arms.
  let muon = Muon::default_with_lr(0.01)?;
  let tall = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &(3, 2))?;
  let out = muon.newton_schulz5(&tall, 5)?;
  // The transpose-in and transpose-back arms are exercised by the call itself;
  // the restored (3, 2) shape is the structural oracle. The Newton-Schulz value
  // is not cheaply closed-form and its conditioning is input-sensitive, so this
  // test pins the shape contract of the transpose round-trip, not the values.
  assert_eq!(
    out.shape(),
    vec![3, 2],
    "tall-matrix output shape preserved"
  );
  Ok(())
}

#[test]
fn muon_newton_schulz5_wide_matrix_no_transpose_preserves_shape() -> Result<()> {
  // A "wide" matrix (cols >= rows) skips the transpose branch (the `else`
  // `try_clone` arm). Pairs with the tall-matrix test to pin both sides of
  // the `transpose_needed` conditional. Structural oracle: shape (2, 3)
  // preserved, entries finite.
  let muon = Muon::default_with_lr(0.01)?;
  let wide = Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &(2, 3))?;
  let mut out = muon.newton_schulz5(&wide, 5)?;
  assert_eq!(
    out.shape(),
    vec![2, 3],
    "wide-matrix output shape preserved"
  );
  let v: Vec<f32> = out.to_vec()?;
  assert!(v.iter().all(|x| x.is_finite()), "all entries finite: {v:?}");
  Ok(())
}

// ── preflight re-resolution across two steps ──────────────────────────────

#[test]
fn muon_two_steps_preflight_re_resolves() -> Result<()> {
  // First apply hits preflight's step-0 cache (stamped by `new`); the second
  // apply lands at step_count=1 with the stamp still Some(0), so `preflight`
  // re-resolves the fixed LR via its non-cache body (the `try_current` +
  // re-stamp lines). 1D params keep the math to plain momentum-SGD so the
  // weight strictly decreases on a positive gradient. Also exercises the
  // trait `init`, `step()`, and `learning_rate()`.
  let mut muon = Muon::default_with_lr(0.01)?;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3])?,
  );
  muon.init(&params)?;
  assert_eq!(muon.step(), 0);
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3], &[3])?,
  );
  muon.apply_gradients(&grads, &mut params)?;
  let mut w1 = params["w"].try_clone()?;
  let after_one: f32 = w1.to_vec::<f32>()?[0];
  assert_eq!(muon.step(), 1);
  muon.apply_gradients(&grads, &mut params)?;
  let mut w2 = params["w"].try_clone()?;
  let after_two: f32 = w2.to_vec::<f32>()?[0];
  assert_eq!(muon.step(), 2);
  assert_eq!(muon.learning_rate(), 0.01);
  assert!(after_two < after_one, "weight should keep decreasing");
  Ok(())
}

// ── apply_gradients per-key branches ──────────────────────────────────────

#[test]
fn muon_skips_grad_key_absent_from_params() -> Result<()> {
  // A gradient whose key has no matching parameter must be skipped (the
  // `let Some(param) = params.get(key) else { continue }` guard), leaving
  // the present parameter updated and the absent one never materialized.
  // 1D param ⇒ plain momentum-SGD closed form (default μ=0.95, wd=0.01,
  // nesterov=true):
  //   g_eff = 0.5 + 0.01·1.0 = 0.51
  //   v = 0.05·0.51 = 0.0255
  //   update = 0.51·0.05 + 0.0255·0.95 = 0.0255 + 0.024225 = 0.049725
  //   w_new = 1.0 - 0.01·0.049725 = 0.99950275
  let mut muon = Muon::default_with_lr(0.01)?;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), Array::from_slice::<f32>(&[1.0], &[1])?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), Array::from_slice::<f32>(&[0.5], &[1])?);
  grads.insert("absent".into(), Array::from_slice::<f32>(&[0.5], &[1])?);
  muon.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.999_503).abs() < 1e-5, "present got {got}");
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added to params"
  );
  Ok(())
}

#[test]
fn muon_step_none_state_arm_via_uninit_grad_key() -> Result<()> {
  // The per-key `None` velocity arm (`None => zeros_like(param)`) is only
  // reachable when a gradient key was NOT pre-initialized: explicitly `init`
  // with a SUBSET of params, then `apply_gradients` with an extra grad key
  // that IS present in params. Because state is already non-empty,
  // `apply_gradients` skips its lazy re-init, so the extra key "b" falls
  // through to `None => zeros_like(param)` — equivalent to a fresh velocity,
  // so the 1D momentum-SGD closed form applies (same as a first step):
  //   w=1.0, g=0.5 ⇒ w_new ≈ 0.99950275 (see the skip-key oracle).
  let mut muon = Muon::default_with_lr(0.01)?;
  // init only "a".
  let mut init_params: Weights = HashMap::new();
  init_params.insert("a".into(), Array::from_slice::<f32>(&[1.0], &[1])?);
  muon.init(&init_params)?;
  assert!(
    !muon.state.is_empty(),
    "explicit init populated state for 'a'"
  );
  // params + grads both carry "a" (initialized) and "b" (NOT initialized).
  let mut params: Weights = HashMap::new();
  params.insert("a".into(), Array::from_slice::<f32>(&[1.0], &[1])?);
  params.insert("b".into(), Array::from_slice::<f32>(&[1.0], &[1])?);
  let mut grads: Weights = HashMap::new();
  grads.insert("a".into(), Array::from_slice::<f32>(&[0.5], &[1])?);
  grads.insert("b".into(), Array::from_slice::<f32>(&[0.5], &[1])?);
  muon.apply_gradients(&grads, &mut params)?;
  // "b" took the None-arm fresh-velocity step and matches the closed form.
  let got_b = read_scalar(&params["b"])?;
  assert!((got_b - 0.999_503).abs() < 1e-5, "b got {got_b}");
  // "a" (pre-initialized to zero velocity) took the same first step.
  let got_a = read_scalar(&params["a"])?;
  assert!((got_a - 0.999_503).abs() < 1e-5, "a got {got_a}");
  Ok(())
}

#[test]
fn muon_3d_param_invokes_reshape_branch() -> Result<()> {
  // A param with ndim > 2 takes the reshape-to-(M, prod(rest)) branch, runs
  // Newton-Schulz on the flattened 2D matrix, then reshapes back to the
  // original 3D shape. Structural oracle (Newton-Schulz value not cheaply
  // closed-form): the output shape must equal the (2, 2, 1) input shape, all
  // entries finite, and the param must have moved off its initial value.
  // Use μ=0, wd=0, nesterov=false so update == newton_schulz5(grad) exactly.
  let mut muon = Muon::new(0.01, 0.0, 0.0, false, 5)?;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 0.0, 0.0, 1.0], &(2, 2, 1))?,
  );
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.5, 0.0, 0.0, 0.5], &(2, 2, 1))?,
  );
  muon.apply_gradients(&grads, &mut params)?;
  let mut out = params["w"].try_clone()?;
  assert_eq!(out.shape(), vec![2, 2, 1], "3D output shape preserved");
  let v: Vec<f32> = out.to_vec()?;
  assert!(v.iter().all(|x| x.is_finite()), "all entries finite: {v:?}");
  // The update moved at least one entry off its initial value.
  assert!(
    (v[0] - 1.0).abs() > 1e-6 || (v[3] - 1.0).abs() > 1e-6,
    "reshape+newton-schulz update must move the param: {v:?}"
  );
  Ok(())
}
