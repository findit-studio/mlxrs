use super::*;

fn scalar(v: f32) -> Result<Array> {
  Array::full::<f32>(&[0i32; 0], v)
}

fn read_scalar(a: &Array) -> Result<f32> {
  let mut clone = a.try_clone()?;
  clone.item::<f32>()
}

#[test]
fn vanilla_sgd_single_step_matches_python_ref() -> Result<()> {
  // Python: w_new = w - lr * g; w=1.0, g=0.5, lr=0.1 → w_new = 0.95.
  let mut sgd = SGD::vanilla(0.1)?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.95).abs() < 1e-6, "expected 0.95, got {got}");
  assert_eq!(sgd.step(), 1);
  assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
  Ok(())
}

#[test]
fn sgd_with_momentum_single_step_matches_python_ref() -> Result<()> {
  // Python (first step, v=0): v = 0 + g; w_new = w - lr * v.
  // w=1.0, g=0.5, lr=0.1, momentum=0.9 → v=0.5, w_new=0.95.
  let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.0, false)?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  assert!((read_scalar(&params["w"])? - 0.95).abs() < 1e-6);
  let v = read_scalar(&sgd.state["w"])?;
  assert!((v - 0.5).abs() < 1e-6, "expected v=0.5, got {v}");
  Ok(())
}

#[test]
fn sgd_with_weight_decay_matches_python_ref() -> Result<()> {
  // Python: g_eff = g + wd*w; w_new = w - lr * g_eff (vanilla; momentum=0).
  // w=2.0, g=1.0, lr=0.1, wd=0.5 → g_eff=2.0, w_new=1.8.
  let mut sgd = SGD::new(0.1, 0.0, 0.5, 0.0, false)?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(2.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(1.0)?);
  sgd.apply_gradients(&grads, &mut params)?;
  assert!((read_scalar(&params["w"])? - 1.8).abs() < 1e-6);
  Ok(())
}

#[test]
fn sgd_nesterov_precondition_rejects_zero_momentum() {
  // `SGD` does not implement `Debug`, so `expect_err` is unavailable; match
  // the `Result` directly to surface the typed payload.
  match SGD::new(0.1, 0.0, 0.0, 0.0, true) {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(payload.context(), "SGD: Nesterov momentum");
      assert_eq!(
        payload.requirement(),
        "requires momentum > 0 (finite) and dampening == 0"
      );
    }
    Err(other) => panic!("expected InvariantViolation, got: {other:?}"),
    Ok(_) => panic!("nesterov with momentum=0 must be rejected"),
  }
}

#[test]
fn sgd_builder_with_nesterov_rejects_zero_momentum() {
  // vanilla has momentum=0; enabling nesterov should fail
  let res = SGD::vanilla(0.1).and_then(|s| s.with_nesterov(true));
  assert!(res.is_err());
}

#[test]
fn sgd_builder_with_momentum_zero_rejects_nesterov() {
  // Start with valid nesterov=true (momentum > 0), then set momentum to 0
  let res = SGD::new(0.1, 0.9, 0.0, 0.0, true).and_then(|s| s.with_momentum(0.0));
  assert!(res.is_err());
}

#[test]
fn sgd_builder_with_dampening_rejects_nesterov() {
  // Start with valid nesterov=true, then add dampening
  let res = SGD::new(0.1, 0.9, 0.0, 0.0, true).and_then(|s| s.with_dampening(0.1));
  assert!(res.is_err());
}

#[test]
fn sgd_schedule_advances_lr_each_step() -> Result<()> {
  // Schedule: lr(step) = 0.1 / max(step, 1). The Python
  // `apply_gradients` resolves scheduled parameters at the PRE-increment
  // step (`optimizers.py:102..=106`), so the first call sees step 0
  // (lr = 0.1 / max(0,1) = 0.1), the second call sees step 1 (also
  // 0.1), and the third call sees step 2 (lr = 0.05).
  let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| 0.1 / (step as f32).max(1.0));
  let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(1.0)?);
  sgd.apply_gradients(&grads, &mut params)?;
  assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
  sgd.apply_gradients(&grads, &mut params)?;
  // step==1 → lr(1) = 0.1
  assert!((sgd.learning_rate() - 0.1).abs() < 1e-6);
  sgd.apply_gradients(&grads, &mut params)?;
  // step==2 → lr(2) = 0.05
  assert!((sgd.learning_rate() - 0.05).abs() < 1e-6);
  Ok(())
}

#[test]
fn optimizer_lr_schedule_resolves_at_pre_increment_step() -> Result<()> {
  // Identity-of-step schedule: returns the step value as the LR. Verifies
  // each apply_gradients resolves at the PRE-increment counter (Python
  // semantics: scheduler is called with `self.step` BEFORE the
  // increment, `optimizers.py:102..=106`).
  let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| step as f32);
  let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.0)?); // zero grads keep params unchanged
  sgd.apply_gradients(&grads, &mut params)?;
  assert!(
    (sgd.learning_rate() - 0.0).abs() < 1e-6,
    "first call must see step 0, got {}",
    sgd.learning_rate()
  );
  sgd.apply_gradients(&grads, &mut params)?;
  assert!(
    (sgd.learning_rate() - 1.0).abs() < 1e-6,
    "second call must see step 1, got {}",
    sgd.learning_rate()
  );
  sgd.apply_gradients(&grads, &mut params)?;
  assert!(
    (sgd.learning_rate() - 2.0).abs() < 1e-6,
    "third call must see step 2, got {}",
    sgd.learning_rate()
  );
  // step counter must still post-increment to N after N calls.
  assert_eq!(sgd.step(), 3);
  Ok(())
}

#[test]
fn sgd_new_rejects_nan_weight_decay() {
  assert!(SGD::new(0.1, 0.0, f32::NAN, 0.0, false).is_err());
}

#[test]
fn sgd_new_rejects_inf_weight_decay() {
  assert!(SGD::new(0.1, 0.0, f32::INFINITY, 0.0, false).is_err());
}

#[test]
fn sgd_new_rejects_negative_weight_decay() {
  assert!(SGD::new(0.1, 0.0, -0.1, 0.0, false).is_err());
}

#[test]
fn sgd_with_weight_decay_rejects_nan() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn sgd_with_weight_decay_rejects_inf() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(f32::INFINITY));
  assert!(res.is_err());
}

#[test]
fn sgd_with_weight_decay_rejects_negative() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_weight_decay(-0.1));
  assert!(res.is_err());
}

#[test]
fn sgd_new_rejects_nan_dampening() {
  assert!(SGD::new(0.1, 0.0, 0.0, f32::NAN, false).is_err());
}

#[test]
fn sgd_new_rejects_inf_dampening() {
  assert!(SGD::new(0.1, 0.0, 0.0, f32::INFINITY, false).is_err());
}

#[test]
fn sgd_new_rejects_negative_dampening() {
  assert!(SGD::new(0.1, 0.0, 0.0, -0.1, false).is_err());
}

#[test]
fn sgd_with_dampening_rejects_nan() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn sgd_with_dampening_rejects_inf() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(f32::INFINITY));
  assert!(res.is_err());
}

#[test]
fn sgd_with_dampening_rejects_negative() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_dampening(-0.1));
  assert!(res.is_err());
}

#[test]
fn sgd_validate_nesterov_rejects_nan_momentum() {
  // `validate_nesterov` must reject `f32::NAN` momentum under the
  // Nesterov branch — `momentum <= 0` is false for NaN under IEEE-754,
  // so `!is_finite() || momentum <= 0.0` is the correct guard.
  assert!(SGD::new(0.1, f32::NAN, 0.0, 0.0, true).is_err());
  // The corresponding builder path must also reject.
  let with_path = SGD::new(0.1, 0.9, 0.0, 0.0, false)
    .unwrap()
    .with_momentum(f32::NAN);
  assert!(with_path.is_err());
}

#[test]
fn sgd_rejects_nan_momentum_even_without_nesterov() {
  // Non-finite momentum must be rejected unconditionally — without this
  // guard `SGD::new(..., NaN, ..., false)` would succeed and the first
  // `apply_gradients` would propagate NaN into velocity (Nesterov-path
  // -agnostic).
  assert!(SGD::new(0.1, f32::NAN, 0.0, 0.0, false).is_err());
  assert!(SGD::new(0.1, f32::INFINITY, 0.0, 0.0, false).is_err());
  // Builder path: vanilla SGD has momentum=0; toggling to NaN must reject.
  let with_path = SGD::vanilla(0.1).unwrap().with_momentum(f32::NAN);
  assert!(with_path.is_err());
}

#[test]
fn sgd_with_learning_rate_rejects_fixed_nan() {
  let res = SGD::vanilla(0.1).and_then(|s| s.with_learning_rate(LearningRate::Fixed(f32::NAN)));
  assert!(res.is_err(), "with_learning_rate must reject Fixed(NaN)");
}

// ── Config getters echo constructor inputs ────────────────────────────────

#[test]
fn sgd_getters_echo_inputs() -> Result<()> {
  // Distinct non-default values so every getter is observably distinct.
  // Covers `learning_rate_ref`, `momentum`, `weight_decay`, `dampening`,
  // `nesterov`, plus the `step()` / `learning_rate()` trait methods on a
  // fresh optimizer. Nesterov stays false so the (momentum>0, dampening>0)
  // combo is admissible.
  let sgd = SGD::new(LearningRate::Fixed(0.25), 0.8, 0.05, 0.1, false)?;
  assert!(
    sgd.learning_rate_ref().is_fixed(),
    "learning_rate_ref must echo the Fixed schedule"
  );
  assert_eq!(sgd.momentum(), 0.8);
  assert_eq!(sgd.weight_decay(), 0.05);
  assert_eq!(sgd.dampening(), 0.1);
  assert!(!sgd.nesterov());
  // Resolved LR at step 0 echoes the fixed value; step is 0 pre-apply.
  assert_eq!(sgd.learning_rate(), 0.25);
  assert_eq!(sgd.step(), 0);
  Ok(())
}

#[test]
fn sgd_nesterov_getter_reflects_true() -> Result<()> {
  // The `nesterov()` getter's `true` arm requires a valid Nesterov config
  // (momentum > 0, dampening == 0).
  let sgd = SGD::new(0.1, 0.9, 0.0, 0.0, true)?;
  assert!(sgd.nesterov());
  Ok(())
}

// ── Builder success paths (echo the set value) ────────────────────────────

#[test]
fn sgd_builder_success_paths_echo() -> Result<()> {
  // Each `with_*` success arm must echo its input. `with_learning_rate`'s
  // success arm resolves the fixed value at step 0. Start vanilla (all
  // zeros, nesterov=false) so every setter's success branch is reachable.
  let sgd = SGD::vanilla(0.1)?
    .with_learning_rate(LearningRate::Fixed(0.05))?
    .with_momentum(0.7)?
    .with_weight_decay(0.2)?
    .with_dampening(0.3)?;
  assert_eq!(sgd.learning_rate(), 0.05);
  assert!(sgd.learning_rate_ref().is_fixed());
  assert_eq!(sgd.momentum(), 0.7);
  assert_eq!(sgd.weight_decay(), 0.2);
  assert_eq!(sgd.dampening(), 0.3);
  Ok(())
}

#[test]
fn sgd_with_nesterov_success_enables_flag() -> Result<()> {
  // `with_nesterov(true)` success arm: starting from a valid momentum>0,
  // dampening==0 config, toggling nesterov on must succeed and echo.
  let sgd = SGD::new(0.1, 0.9, 0.0, 0.0, false)?.with_nesterov(true)?;
  assert!(sgd.nesterov());
  Ok(())
}

// ── Closed-form step oracles: dampening + nesterov + None-state arm ───────

#[test]
fn sgd_with_dampening_single_step_matches_python_ref() -> Result<()> {
  // First step (v=0), momentum branch with dampening τ>0:
  //   v_new  = µ·0 + (1-τ)·g = (1-τ)·g
  //   update = v_new                (non-nesterov)
  //   w_new  = w - lr·update
  // w=1.0, g=0.5, lr=0.1, µ=0.9, τ=0.5 → v_new = 0.5·0.5 = 0.25;
  //   w_new = 1.0 - 0.1·0.25 = 0.975.
  let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.5, false)?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.975).abs() < 1e-6, "expected 0.975, got {got}");
  // velocity state holds (1-τ)·g = 0.25.
  let v = read_scalar(&sgd.state["w"])?;
  assert!((v - 0.25).abs() < 1e-6, "expected v=0.25, got {v}");
  Ok(())
}

#[test]
fn sgd_nesterov_single_step_matches_python_ref() -> Result<()> {
  // First step (v=0), Nesterov branch (dampening=0):
  //   v_new  = µ·0 + g = g
  //   update = g + µ·v_new = g·(1 + µ)
  //   w_new  = w - lr·update
  // w=1.0, g=0.5, lr=0.1, µ=0.9 → update = 0.5·1.9 = 0.95;
  //   w_new = 1.0 - 0.1·0.95 = 0.905.
  let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.0, true)?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.905).abs() < 1e-6, "expected 0.905, got {got}");
  Ok(())
}

#[test]
fn sgd_momentum_step_none_state_arm_via_uninit_grad_key() -> Result<()> {
  // The momentum branch's per-key `None` velocity arm (no prior `v` for the
  // key) is reached when a grad key was NOT pre-initialized: explicit `init`
  // with a SUBSET of params, then `apply_gradients` with an extra grad key
  // that IS present in params. State is already non-empty, so the lazy
  // re-init is skipped and the extra key hits `None => zeros_like(param)`.
  //   w=1.0, g=0.5, lr=0.1, µ=0.9 (first momentum step, v=0):
  //   v_new=g=0.5; w_new = 1.0 - 0.1·0.5 = 0.95.
  let mut sgd = SGD::new(0.1, 0.9, 0.0, 0.0, false)?;
  let mut init_params: Weights = HashMap::new();
  init_params.insert("a".into(), scalar(1.0)?);
  sgd.init(&init_params)?;
  assert!(
    !sgd.state.is_empty(),
    "explicit init populated state for 'a'"
  );
  let mut params: Weights = HashMap::new();
  params.insert("a".into(), scalar(1.0)?);
  params.insert("b".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("a".into(), scalar(0.5)?);
  grads.insert("b".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  // "b" took the None-arm fresh-velocity step.
  let got_b = read_scalar(&params["b"])?;
  assert!((got_b - 0.95).abs() < 1e-6, "b got {got_b}");
  Ok(())
}

#[test]
fn sgd_skips_grad_key_absent_from_params() -> Result<()> {
  // A gradient whose key has no matching parameter must be skipped (the
  // `let Some(param) = params.get(key) else { continue }` guard), leaving
  // the present parameter updated and the absent one never materialized.
  // Vanilla SGD: w=1.0, g=0.5, lr=0.1 → w_new = 0.95.
  let mut sgd = SGD::vanilla(0.1)?;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), scalar(0.5)?);
  grads.insert("absent".into(), scalar(0.5)?);
  sgd.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.95).abs() < 1e-6, "present got {got}");
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added to params"
  );
  Ok(())
}

#[test]
fn sgd_apply_gradients_rejects_schedule_that_goes_nan() -> Result<()> {
  // Step 0 → 0.1 (finite, new() accepts). Step 1 → NaN (must error
  // before any param mutation).
  let sched: Box<dyn Fn(usize) -> f32> = Box::new(|step| if step == 0 { 0.1 } else { f32::NAN });
  let mut sgd = SGD::vanilla(LearningRate::Schedule(sched))?;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(0.5)?);
  // First call (step 0, lr=0.1) must succeed.
  sgd.apply_gradients(&grads, &mut params)?;
  // Second call (step 1, lr=NaN) must error before mutating params.
  let w_before = {
    let mut c = params["w"].try_clone()?;
    c.item::<f32>()?
  };
  let result = sgd.apply_gradients(&grads, &mut params);
  assert!(
    result.is_err(),
    "apply_gradients must reject schedule NaN at step 1"
  );
  // Params must not have been mutated.
  let w_after = {
    let mut c = params["w"].try_clone()?;
    c.item::<f32>()?
  };
  assert!(
    (w_before - w_after).abs() < 1e-9,
    "params must not be mutated when LR goes NaN: before={w_before}, after={w_after}"
  );
  Ok(())
}
