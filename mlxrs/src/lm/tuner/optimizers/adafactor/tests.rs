use super::*;

// ── Validation rejection tests ────────────────────────────────────────────

#[test]
fn adafactor_new_rejects_negative_eps() {
  // eps.0 < 0
  assert!(
    Adafactor::new(
      None,
      (-1e-30, 1e-3),
      1.0,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
  // eps.1 < 0
  assert!(
    Adafactor::new(
      None,
      (1e-30, -1e-3),
      1.0,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_non_finite_eps() {
  assert!(
    Adafactor::new(
      None,
      (f32::NAN, 1e-3),
      1.0,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
  assert!(
    Adafactor::new(
      None,
      (1e-30, f32::INFINITY),
      1.0,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_non_positive_clip_threshold() {
  assert!(Adafactor::new(None, (1e-30, 1e-3), 0.0, -0.8, None, 0.0, true, true, false).is_err());
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      -1.0,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_non_finite_clip_threshold() {
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      f32::NAN,
      -0.8,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_non_finite_decay_rate() {
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      1.0,
      f32::NAN,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      1.0,
      f32::INFINITY,
      None,
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_negative_weight_decay() {
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      1.0,
      -0.8,
      None,
      -0.1,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_new_rejects_non_finite_weight_decay() {
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      1.0,
      -0.8,
      None,
      f32::NAN,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_with_eps_rejects_negative() {
  let res = Adafactor::default_python().and_then(|a| a.with_eps((-1e-30, 1e-3)));
  assert!(res.is_err());
}

#[test]
fn adafactor_with_eps_rejects_non_finite() {
  let res = Adafactor::default_python().and_then(|a| a.with_eps((f32::NAN, 1e-3)));
  assert!(res.is_err());
}

#[test]
fn adafactor_with_clip_threshold_rejects_non_positive() {
  let res = Adafactor::default_python().and_then(|a| a.with_clip_threshold(0.0));
  assert!(res.is_err());
  let res2 = Adafactor::default_python().and_then(|a| a.with_clip_threshold(-1.0));
  assert!(res2.is_err());
}

#[test]
fn adafactor_with_clip_threshold_rejects_non_finite() {
  let res = Adafactor::default_python().and_then(|a| a.with_clip_threshold(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn adafactor_with_decay_rate_rejects_non_finite() {
  let res = Adafactor::default_python().and_then(|a| a.with_decay_rate(f32::NAN));
  assert!(res.is_err());
  let res2 = Adafactor::default_python().and_then(|a| a.with_decay_rate(f32::INFINITY));
  assert!(res2.is_err());
}

#[test]
fn adafactor_with_weight_decay_rejects_negative() {
  let res = Adafactor::default_python().and_then(|a| a.with_weight_decay(-0.1));
  assert!(res.is_err());
}

#[test]
fn adafactor_with_weight_decay_rejects_non_finite() {
  let res = Adafactor::default_python().and_then(|a| a.with_weight_decay(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn adafactor_1d_param_runs_one_step_without_error() -> Result<()> {
  // 1D tensors take the NonFactored branch. Verify the step completes
  // and produces a different weight (no Python ref number — Adafactor's
  // relative-step + RMS clip + lr scaling chain is not easily
  // reduced to a closed-form scalar in two lines).
  let mut adafactor = Adafactor::default_python()?;
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
  adafactor.apply_gradients(&grads, &mut params)?;
  let mut got = params["w"].try_clone()?;
  let v: Vec<f32> = got.to_vec()?;
  // Step must move (Adafactor's relative step is non-zero by default).
  assert!(
    (v[0] - 1.0).abs() > 1e-8,
    "expected w[0] to move, got {}",
    v[0]
  );
  Ok(())
}

#[test]
fn adafactor_2d_param_runs_one_step_without_error() -> Result<()> {
  // 2D tensors take the Factored branch.
  let mut adafactor = Adafactor::default_python()?;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
  );
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
  );
  adafactor.apply_gradients(&grads, &mut params)?;
  let mut got = params["w"].try_clone()?;
  let _: Vec<f32> = got.to_vec()?;
  Ok(())
}

#[test]
fn adafactor_new_rejects_nan_beta_1() {
  assert!(
    Adafactor::new(
      None,
      (1e-30, 1e-3),
      1.0,
      -0.8,
      Some(f32::NAN),
      0.0,
      true,
      true,
      false
    )
    .is_err()
  );
}

#[test]
fn adafactor_with_beta_1_rejects_nan_some() {
  let res = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(f32::NAN)));
  assert!(res.is_err());
}

#[test]
fn adafactor_with_beta_1_rejects_above_one_some() {
  // beta_1 >= 1.0 makes the EMA non-contractive.
  let res = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(1.0)));
  assert!(res.is_err());
  let res2 = Adafactor::default_python().and_then(|a| a.with_beta_1(Some(1.5)));
  assert!(res2.is_err());
}

#[test]
fn adafactor_with_beta_1_accepts_none() -> Result<()> {
  // None is always valid (disables first-moment branch).
  let _a = Adafactor::default_python()?.with_beta_1(None)?;
  Ok(())
}

#[test]
fn adafactor_try_set_beta_1_rejects_nan_pre_init() {
  let mut adafactor = Adafactor::default_python().unwrap();
  // Pre-init: validation should still fire on NaN.
  assert!(adafactor.try_set_beta_1(Some(f32::NAN)).is_err());
}

#[test]
fn adafactor_with_beta_1_rejects_post_init() -> Result<()> {
  // `with_beta_1` controls per-parameter exp_avg state shape; toggling
  // it after any parameter has been initialized would silently produce
  // existing-vs-new parameter shape mismatch. Must error post-init.
  let adafactor = Adafactor::default_python()?;
  let mut params = HashMap::from([(
    "w".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
  )]);
  let grads = HashMap::from([(
    "w".to_string(),
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
  )]);
  let mut adafactor = adafactor;
  adafactor.apply_gradients(&grads, &mut params)?;
  // state is now non-empty — toggling beta_1 must error.
  assert!(adafactor.with_beta_1(Some(0.9)).is_err());
  Ok(())
}

#[test]
fn adafactor_with_learning_rate_rejects_fixed_nan() {
  let res = Adafactor::default_python()
    .and_then(|a| a.with_learning_rate(Some(LearningRate::Fixed(f32::NAN))));
  assert!(
    res.is_err(),
    "with_learning_rate must reject Some(Fixed(NaN))"
  );
}

#[test]
fn adafactor_try_set_beta_1_preserves_state_on_error() -> Result<()> {
  // Non-consuming setter must reject post-init AND leave the optimizer
  // usable (state preserved). Caller still has the populated `adafactor`
  // after the failed setter call.
  let mut adafactor = Adafactor::default_python()?;
  let original_beta_1 = adafactor.beta_1;
  let mut params = HashMap::from([(
    "w".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2))?,
  )]);
  let grads = HashMap::from([(
    "w".to_string(),
    Array::from_slice::<f32>(&[0.1, 0.2, 0.3, 0.4], &(2, 2))?,
  )]);
  adafactor.apply_gradients(&grads, &mut params)?;
  // try_set_beta_1 errors post-init…
  assert!(adafactor.try_set_beta_1(Some(0.9)).is_err());
  // …and beta_1 + state are untouched, so training can continue.
  assert_eq!(adafactor.beta_1, original_beta_1);
  assert!(!adafactor.state.is_empty(), "state preserved on error");
  // Prove the optimizer still works after the rejected setter call.
  adafactor.apply_gradients(&grads, &mut params)?;
  Ok(())
}
