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

// ── Closed-form step helpers ──────────────────────────────────────────────

/// Read a 0D array as an `f32`. `item` is `&mut self`, so clone first
/// (Array has no `Clone`; `try_clone` is the explicit form).
fn read_scalar(a: &Array) -> Result<f32> {
  let mut clone = a.try_clone()?;
  clone.item::<f32>()
}

/// Read an array's elements as a `Vec<f32>`. `to_vec` is `&mut self`.
fn read_vec(a: &Array) -> Result<Vec<f32>> {
  let mut clone = a.try_clone()?;
  clone.to_vec::<f32>()
}

/// Build `(params, grads)` each holding a single 0D scalar under key `"w"`.
fn scalar_p_g(p: f32, g: f32) -> Result<(Weights, Weights)> {
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], p)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), Array::full::<f32>(&[0i32; 0], g)?);
  Ok((params, grads))
}

// ── decay_rate > 0.0 rejection (typed) ────────────────────────────────────

#[test]
fn adafactor_new_rejects_positive_decay_rate() {
  // decay_rate > 0 ⇒ β₂ = 1 - step^decay_rate goes negative after step 1,
  // poisoning the squared-gradient average. Must be a typed OutOfRange.
  let err = Adafactor::new(None, (1e-30, 1e-3), 1.0, 0.5, None, 0.0, true, true, false)
    .err()
    .expect("decay_rate > 0 must be rejected");
  assert!(
    matches!(err, Error::OutOfRange(_)),
    "expected Error::OutOfRange, got {err:?}"
  );
}

#[test]
fn adafactor_with_decay_rate_rejects_positive() {
  let err = Adafactor::default_python()
    .and_then(|a| a.with_decay_rate(0.5))
    .err()
    .expect("with_decay_rate(0.5) must be rejected");
  assert!(matches!(err, Error::OutOfRange(_)), "got {err:?}");
}

// ── Constructor with a Some(learning_rate) ────────────────────────────────

#[test]
fn adafactor_new_with_fixed_lr_echoes_step0() -> Result<()> {
  // Exercises the `Some(lr) => lr.try_current(0)` constructor arm: a fixed
  // LR must resolve at step 0 and be reported by `learning_rate()`.
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (1e-30, 1e-3),
    1.0,
    -0.8,
    None,
    0.0,
    true,
    true,
    false,
  )?;
  assert_eq!(af.learning_rate(), 0.1);
  assert!(af.learning_rate_ref().is_some());
  Ok(())
}

#[test]
fn adafactor_new_rejects_fixed_nan_lr() {
  // The constructor's `try_current(0)` must reject a non-finite fixed LR.
  let res = Adafactor::new(
    Some(LearningRate::Fixed(f32::NAN)),
    (1e-30, 1e-3),
    1.0,
    -0.8,
    None,
    0.0,
    true,
    true,
    false,
  );
  assert!(res.is_err(), "Some(Fixed(NaN)) must be rejected at step 0");
}

// ── Config getters echo constructor inputs ────────────────────────────────

#[test]
fn adafactor_getters_echo_inputs() -> Result<()> {
  // Distinct non-default values so every getter is observably distinct.
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.25)),
    (3e-30, 4e-3),
    2.5,
    -0.6,
    Some(0.7),
    0.125,
    false,
    false,
    true,
  )?;
  assert!(af.learning_rate_ref().is_some());
  assert_eq!(af.eps(), (3e-30, 4e-3));
  assert_eq!(af.clip_threshold(), 2.5);
  assert_eq!(af.decay_rate(), -0.6);
  assert_eq!(af.beta_1(), Some(0.7));
  assert_eq!(af.weight_decay(), 0.125);
  assert!(!af.scale_parameter());
  assert!(!af.relative_step());
  assert!(af.warmup_init());
  Ok(())
}

#[test]
fn adafactor_default_python_getters() -> Result<()> {
  // The None-LR arm of `learning_rate_ref` + the Python defaults.
  let af = Adafactor::default_python()?;
  assert!(af.learning_rate_ref().is_none());
  assert_eq!(af.eps(), (1e-30, 1e-3));
  assert_eq!(af.clip_threshold(), 1.0);
  assert_eq!(af.decay_rate(), -0.8);
  assert_eq!(af.beta_1(), None);
  assert_eq!(af.weight_decay(), 0.0);
  assert!(af.scale_parameter());
  assert!(af.relative_step());
  assert!(!af.warmup_init());
  Ok(())
}

// ── Builder success paths (echo the set value) ────────────────────────────

#[test]
fn adafactor_builder_success_paths_echo() -> Result<()> {
  let af = Adafactor::default_python()?
    .with_learning_rate(Some(LearningRate::Fixed(0.05)))?
    .with_eps((2e-30, 5e-3))?
    .with_clip_threshold(3.0)?
    .with_decay_rate(-0.5)?
    .with_weight_decay(0.2)?
    .with_scale_parameter(false)
    .with_relative_step(false)
    .with_warmup_init(true);
  // with_learning_rate's success arm resolves the fixed value at step 0.
  assert_eq!(af.learning_rate(), 0.05);
  assert!(af.learning_rate_ref().is_some());
  assert_eq!(af.eps(), (2e-30, 5e-3));
  assert_eq!(af.clip_threshold(), 3.0);
  assert_eq!(af.decay_rate(), -0.5);
  assert_eq!(af.weight_decay(), 0.2);
  assert!(!af.scale_parameter());
  assert!(!af.relative_step());
  assert!(af.warmup_init());
  Ok(())
}

#[test]
fn adafactor_with_learning_rate_none_sets_zero() -> Result<()> {
  // The `None => 0.0` arm of `with_learning_rate`.
  let af = Adafactor::default_python()?
    .with_learning_rate(Some(LearningRate::Fixed(0.3)))?
    .with_learning_rate(None)?;
  assert_eq!(af.learning_rate(), 0.0);
  assert!(af.learning_rate_ref().is_none());
  Ok(())
}

#[test]
fn adafactor_try_set_beta_1_pre_init_succeeds() -> Result<()> {
  // Pre-init (empty state): `try_set_beta_1` must succeed and echo.
  let mut af = Adafactor::default_python()?;
  af.try_set_beta_1(Some(0.9))?;
  assert_eq!(af.beta_1(), Some(0.9));
  // Setting back to None is also valid.
  af.try_set_beta_1(None)?;
  assert_eq!(af.beta_1(), None);
  Ok(())
}

// ── Closed-form single-step oracles ───────────────────────────────────────
//
// All step oracles below disable both `relative_step` and `scale_parameter`
// and use a fixed LR, so `compute_learning_rate` returns exactly the fixed
// `current_lr`. At step 1, `step^decay_rate = 1` for any decay_rate, so
// β₂ = 1 - 1 = 0 and (1-β₂) = 1 — the running averages collapse to the
// current update, which makes the factored / non-factored arithmetic
// hand-computable. ε₁ = 0 and clip_threshold = 1 keep the divisor exact.

#[test]
fn adafactor_nonfactored_scalar_no_beta1_no_wd_step1() -> Result<()> {
  // 0D param ⇒ NonFactored branch, β₁=None ⇒ `other => other`, wd=0.
  // w=1, g=0.5, lr=0.1, ε₁=0, clip=1, scale_param=false, relative_step=false:
  //   update   = g² = 0.25
  //   new_eas  = 0·0 + 1·0.25 = 0.25
  //   upd      = rsqrt(0.25)·g = 2·0.5 = 1.0
  //   clip     = 1.0 / max(1, 1.0/1) = 1.0
  //   upd·lr   = 0.1
  //   w_new    = 1.0 - 0.1 = 0.9
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    None,
    0.0,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let (mut params, grads) = scalar_p_g(1.0, 0.5)?;
  af.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.9).abs() < 1e-4, "got {got}");
  Ok(())
}

#[test]
fn adafactor_nonfactored_scalar_with_beta1_step1() -> Result<()> {
  // 0D param + β₁=0.5 ⇒ exercises the NonFactored first-moment branch.
  // Same as above through `upd·lr = 0.1`, then:
  //   new_ea  = β₁·0 + (1-β₁)·0.1 = 0.5·0.1 = 0.05
  //   update  = new_ea = 0.05
  //   w_new   = 1.0 - 0.05 = 0.95
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    Some(0.5),
    0.0,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let (mut params, grads) = scalar_p_g(1.0, 0.5)?;
  af.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.95).abs() < 1e-4, "got {got}");
  Ok(())
}

#[test]
fn adafactor_nonfactored_scalar_with_weight_decay_step1() -> Result<()> {
  // 0D param + wd=0.5, β₁=None ⇒ exercises the weight-decay branch.
  // Through `upd·lr = 0.1` as before, then:
  //   param_after_decay = w + w·(-wd·lr) = 1·(1 - 0.5·0.1) = 0.95
  //   w_new = 0.95 - 0.1 = 0.85
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    None,
    0.5,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let (mut params, grads) = scalar_p_g(1.0, 0.5)?;
  af.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.85).abs() < 1e-4, "got {got}");
  Ok(())
}

#[test]
fn adafactor_factored_2d_with_beta1_step1() -> Result<()> {
  // 2x2 param with a constant grad ⇒ the row/col factored averages are
  // symmetric, so the rsqrt + outer-product `approx` reduces to 1/g and
  // `approx·g = 1` for every element. Exercises the Factored first-moment
  // branch.
  // g=0.5 everywhere, lr=0.1, ε₁=0, clip=1, β₁=0.5, scale_param=false:
  //   update         = g² = 0.25 (all)
  //   row_new=col_new= 0 + 1·0.25 = 0.25
  //   row_mean       = 0.25 ⇒ row_norm = 1 ⇒ r_factor = 1
  //   c_factor       = rsqrt(0.25) = 2 = 1/g
  //   approx         = r ⊗ c = 1/g (all) ⇒ approx·g = 1 (all)
  //   clip           = 1 ⇒ upd·lr = 0.1 (all)
  //   new_ea         = (1-β₁)·0.1 = 0.05 ⇒ update = 0.05 (all)
  //   w_new          = 1.0 - 0.05 = 0.95 (all)
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    Some(0.5),
    0.0,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let mut params: Weights = HashMap::new();
  params.insert(
    "w".into(),
    Array::from_slice::<f32>(&[1.0, 1.0, 1.0, 1.0], &(2, 2))?,
  );
  let mut grads: Weights = HashMap::new();
  grads.insert(
    "w".into(),
    Array::from_slice::<f32>(&[0.5, 0.5, 0.5, 0.5], &(2, 2))?,
  );
  af.apply_gradients(&grads, &mut params)?;
  let v = read_vec(&params["w"])?;
  for (i, x) in v.iter().enumerate() {
    assert!((x - 0.95).abs() < 1e-4, "w[{i}] = {x}, expected 0.95");
  }
  Ok(())
}

// ── relative_step rsqrt path + state-init beta_1 allocation ───────────────

#[test]
fn adafactor_relative_step_scaled_runs_and_moves() -> Result<()> {
  // scale_parameter=true + relative_step=true (Python defaults) + β₁ set
  // ⇒ exercises (a) the `init_state_for` β₁ `exp_avg` allocation, (b) the
  // relative-step rsqrt branch, and (c) the scale_parameter `param_scale`
  // multiply. No closed-form number (relative-step + RMS chain), so assert
  // the step moves the weight.
  let mut af = Adafactor::default_python()?.with_beta_1(Some(0.9))?;
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
  af.apply_gradients(&grads, &mut params)?;
  let v = read_vec(&params["w"])?;
  assert!(
    (v[0] - 1.0).abs() > 1e-8,
    "expected w[0] to move, got {}",
    v[0]
  );
  Ok(())
}

// ── preflight Some(lr) arm + trait step()/learning_rate() across 2 steps ──

#[test]
fn adafactor_two_steps_preflight_some_lr_arm() -> Result<()> {
  // First apply hits preflight's step-0 cache (stamped by `new`); the
  // second apply lands at step_count=1 with the stamp still at Some(0),
  // so `preflight` re-resolves via the `Some(lr)` arm. Also covers the
  // `step()` and `learning_rate()` trait methods.
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    None,
    0.0,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let (mut params, grads) = scalar_p_g(1.0, 0.5)?;
  assert_eq!(af.step(), 0);
  af.apply_gradients(&grads, &mut params)?;
  assert_eq!(af.step(), 1);
  af.apply_gradients(&grads, &mut params)?;
  assert_eq!(af.step(), 2);
  assert_eq!(af.learning_rate(), 0.1);
  Ok(())
}

#[test]
fn adafactor_two_steps_preflight_none_arm() -> Result<()> {
  // Same two-step shape but with `learning_rate=None`, so the second
  // apply's `preflight` re-resolves via the `None => 0.0` arm.
  let mut af = Adafactor::default_python()?;
  let (mut params, grads) = scalar_p_g(1.0, 0.5)?;
  af.apply_gradients(&grads, &mut params)?;
  af.apply_gradients(&grads, &mut params)?;
  assert_eq!(af.step(), 2);
  assert_eq!(af.learning_rate(), 0.0);
  Ok(())
}

// ── grad key absent from params ⇒ skipped ─────────────────────────────────

#[test]
fn adafactor_skips_grad_key_absent_from_params() -> Result<()> {
  // A gradient whose key has no matching parameter must be skipped (the
  // `let Some(param) = params.get(key) else { continue }` guard), leaving
  // the present parameter updated and the absent one never inserted.
  let af = Adafactor::new(
    Some(LearningRate::Fixed(0.1)),
    (0.0, 1e-3),
    1.0,
    -0.8,
    None,
    0.0,
    false,
    false,
    false,
  )?;
  let mut af = af;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), Array::full::<f32>(&[0i32; 0], 0.5)?);
  grads.insert("absent".into(), Array::full::<f32>(&[0i32; 0], 0.5)?);
  af.apply_gradients(&grads, &mut params)?;
  // The present param took the closed-form step (= 0.9, see scalar oracle).
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.9).abs() < 1e-4, "got {got}");
  // The absent key was never materialized into params.
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added"
  );
  Ok(())
}
