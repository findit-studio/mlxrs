use super::*;

fn read_scalar(a: &Array) -> Result<f32> {
  let mut clone = a.try_clone()?;
  clone.item::<f32>()
}

fn p_g(p: f32, g: f32) -> Result<(Weights, Weights)> {
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), scalar(p)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("w".into(), scalar(g)?);
  Ok((params, grads))
}

#[test]
fn adam_single_step_no_bias_correction_matches_python_ref() -> Result<()> {
  // Python (no bias correction): m=(1-β₁)g, v=(1-β₂)g²;
  // w_new = w - lr·m / (sqrt(v) + eps).
  // w=1.0, g=0.5, lr=0.001, β=(0.9,0.999), eps=1e-8
  //   m = 0.1 * 0.5 = 0.05
  //   v = 0.001 * 0.25 = 0.00025
  //   step = 0.001 * 0.05 / (sqrt(0.00025) + 1e-8)
  //        = 0.00005 / 0.01581138...
  //        ≈ 0.0031623
  //   w_new ≈ 1.0 - 0.0031623 = 0.9968377
  let mut adam = Adam::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adam.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.996_837_7).abs() < 1e-5, "got {got}");
  Ok(())
}

#[test]
fn adam_bias_correction_step1_matches_python_ref() -> Result<()> {
  // Python with bias correction at t=1:
  //   m=(1-β₁)g, v=(1-β₂)g²
  //   c₁ = lr / (1 - β₁); c₂ = rsqrt(1 - β₂)
  //   numerator = c₁·m
  //   denominator = sqrt(v)·c₂ + eps
  //   step = numerator / denominator
  // At t=1 this reduces to:
  //   c₁·m = (lr / 0.1) * 0.1*g = lr * g
  //   sqrt(v)*c₂ = |g| * sqrt(1-β₂) * rsqrt(1-β₂) = |g|
  //   step ≈ lr*g / (|g| + eps) = lr * sign(g) (for g != 0)
  // w=1.0, g=0.5, lr=0.001 → step ≈ 0.001 → w_new ≈ 0.999
  let mut adam = Adam::new(0.001, (0.9, 0.999), 1e-8, true)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adam.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.999).abs() < 1e-4, "got {got}");
  Ok(())
}

#[test]
fn adamw_decoupled_weight_decay_applies_before_step() -> Result<()> {
  // Python AdamW first step: w_decoupled = w·(1 - lr·wd) then Adam step.
  // w=1.0, lr=0.001, wd=0.01 → w_decoupled = 1.0·(1 - 1e-5) = 0.99999
  // Then Adam step with g=0.5 ≈ 0.0031623 → w_new ≈ 0.99999 - 0.0031623
  //                                              ≈ 0.99683
  let mut adamw = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01, false)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adamw.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.996_827_7).abs() < 1e-5, "got {got}");
  Ok(())
}

#[test]
fn adamax_single_step_matches_python_ref() -> Result<()> {
  // Python Adamax first step:
  //   m = (1-β₁)·g = 0.1·0.5 = 0.05
  //   v = max(β₂·0, |g|) = 0.5
  //   step = lr·m / (v + eps) = 0.001·0.05 / 0.5 = 1e-4
  //   w_new = 1.0 - 1e-4 = 0.9999
  let mut adamax = Adamax::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adamax.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.9999).abs() < 1e-5, "got {got}");
  Ok(())
}

#[test]
fn adam_two_consecutive_steps_advance_state() -> Result<()> {
  let mut adam = Adam::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adam.apply_gradients(&grads, &mut params)?;
  let after_one = read_scalar(&params["w"])?;
  adam.apply_gradients(&grads, &mut params)?;
  let after_two = read_scalar(&params["w"])?;
  assert!(after_two < after_one, "weight should keep decreasing");
  assert_eq!(adam.step(), 2);
  Ok(())
}

#[test]
fn adamax_builder_with_eps_rejects_negative() {
  let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_eps(-1e-8));
  assert!(res.is_err());
}

// ── Adam betas validation ────────────────────────────────────────────────

#[test]
fn adam_new_rejects_betas_above_one() {
  // b2 >= 1.0 → sqrt(negative) at bias-correction → NaN weights
  assert!(Adam::new(0.001, (0.9, 1.1), 1e-8, false).is_err());
  assert!(Adam::new(0.001, (1.0, 0.999), 1e-8, false).is_err());
}

#[test]
fn adam_new_rejects_betas_negative() {
  assert!(Adam::new(0.001, (-0.1, 0.999), 1e-8, false).is_err());
  assert!(Adam::new(0.001, (0.9, -0.1), 1e-8, false).is_err());
}

#[test]
fn adam_new_rejects_non_finite_betas() {
  assert!(Adam::new(0.001, (f32::NAN, 0.999), 1e-8, false).is_err());
  assert!(Adam::new(0.001, (0.9, f32::INFINITY), 1e-8, false).is_err());
}

#[test]
fn adam_with_betas_rejects_above_one() {
  let res = Adam::default_with_lr(0.001).and_then(|a| a.with_betas((0.9, 1.1)));
  assert!(res.is_err());
}

#[test]
fn adam_with_betas_rejects_non_finite() {
  let res = Adam::default_with_lr(0.001).and_then(|a| a.with_betas((f32::NAN, 0.999)));
  assert!(res.is_err());
}

#[test]
fn adam_with_eps_rejects_negative() {
  let res = Adam::default_with_lr(0.001).and_then(|a| a.with_eps(-1e-8));
  assert!(res.is_err());
}

#[test]
fn adam_with_eps_rejects_non_finite() {
  let res = Adam::default_with_lr(0.001).and_then(|a| a.with_eps(f32::NAN));
  assert!(res.is_err());
}

// ── AdamW weight_decay validation ────────────────────────────────────────

#[test]
fn adamw_new_rejects_negative_weight_decay() {
  assert!(AdamW::new(0.001, (0.9, 0.999), 1e-8, -0.01, false).is_err());
}

#[test]
fn adamw_new_rejects_non_finite_weight_decay() {
  assert!(AdamW::new(0.001, (0.9, 0.999), 1e-8, f32::NAN, false).is_err());
}

#[test]
fn adamw_with_weight_decay_rejects_negative() {
  let res = AdamW::default_with_lr(0.001).and_then(|a| a.with_weight_decay(-0.1));
  assert!(res.is_err());
}

#[test]
fn adamw_with_weight_decay_rejects_non_finite() {
  let res = AdamW::default_with_lr(0.001).and_then(|a| a.with_weight_decay(f32::INFINITY));
  assert!(res.is_err());
}

// ── Adamax betas validation ───────────────────────────────────────────────

#[test]
fn adamax_new_rejects_betas_above_one() {
  assert!(Adamax::new(0.001, (0.9, 1.1), 1e-8).is_err());
  assert!(Adamax::new(0.001, (1.0, 0.999), 1e-8).is_err());
}

#[test]
fn adamax_new_rejects_non_finite_betas() {
  assert!(Adamax::new(0.001, (f32::NAN, 0.999), 1e-8).is_err());
  assert!(Adamax::new(0.001, (0.9, f32::INFINITY), 1e-8).is_err());
}

#[test]
fn adamax_with_betas_rejects_above_one() {
  let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_betas((0.9, 1.1)));
  assert!(res.is_err());
}

#[test]
fn adamax_with_betas_rejects_non_finite() {
  let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_betas((f32::NAN, 0.999)));
  assert!(res.is_err());
}

#[test]
fn adamax_with_eps_rejects_non_finite() {
  let res = Adamax::default_with_lr(0.001).and_then(|a| a.with_eps(f32::NAN));
  assert!(res.is_err());
}

#[test]
fn adam_with_learning_rate_rejects_fixed_nan() {
  let res =
    Adam::default_with_lr(0.001).and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
  assert!(
    res.is_err(),
    "Adam::with_learning_rate must reject Fixed(NaN)"
  );
}

#[test]
fn adamw_with_learning_rate_rejects_fixed_nan() {
  let res =
    AdamW::default_with_lr(0.001).and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
  assert!(
    res.is_err(),
    "AdamW::with_learning_rate must reject Fixed(NaN)"
  );
}

#[test]
fn adamax_with_learning_rate_rejects_fixed_nan() {
  let res = Adamax::default_with_lr(0.001)
    .and_then(|a| a.with_learning_rate(LearningRate::Fixed(f32::NAN)));
  assert!(
    res.is_err(),
    "Adamax::with_learning_rate must reject Fixed(NaN)"
  );
}

// ── Adam config getters echo constructor inputs ───────────────────────────

#[test]
fn adam_getters_echo_inputs() -> Result<()> {
  // Distinct non-default values so every getter is observably distinct.
  // Covers `learning_rate_ref`, `betas`, `eps`, `bias_correction`.
  let adam = Adam::new(LearningRate::Fixed(0.25), (0.8, 0.99), 1e-6, true)?;
  assert!(
    adam.learning_rate_ref().is_fixed(),
    "learning_rate_ref must echo the Fixed schedule"
  );
  assert_eq!(adam.betas(), (0.8, 0.99));
  assert_eq!(adam.eps(), 1e-6);
  assert!(adam.bias_correction());
  // The resolved learning rate at step 0 echoes the fixed value.
  assert_eq!(adam.learning_rate(), 0.25);
  assert_eq!(adam.step(), 0);
  Ok(())
}

#[test]
fn adam_default_with_lr_getters() -> Result<()> {
  // Exercises the Python-default getter values + the `false`
  // `bias_correction` arm.
  let adam = Adam::default_with_lr(0.001)?;
  assert_eq!(adam.betas(), (0.9, 0.999));
  assert_eq!(adam.eps(), 1e-8);
  assert!(!adam.bias_correction());
  assert!(adam.learning_rate_ref().is_fixed());
  Ok(())
}

// ── Adam builder success paths (echo the set value) ───────────────────────

#[test]
fn adam_builder_success_paths_echo() -> Result<()> {
  // Each `with_*` success arm must echo its input. `with_learning_rate`'s
  // success arm resolves the fixed value at step 0; `with_bias_correction`
  // is infallible.
  let adam = Adam::default_with_lr(0.001)?
    .with_learning_rate(LearningRate::Fixed(0.05))?
    .with_betas((0.7, 0.95))?
    .with_eps(2e-7)?
    .with_bias_correction(true);
  assert_eq!(adam.learning_rate(), 0.05);
  assert!(adam.learning_rate_ref().is_fixed());
  assert_eq!(adam.betas(), (0.7, 0.95));
  assert_eq!(adam.eps(), 2e-7);
  assert!(adam.bias_correction());
  Ok(())
}

#[test]
fn adam_with_bias_correction_toggles_off() -> Result<()> {
  // The infallible setter must also flip true → false.
  let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, true)?.with_bias_correction(false);
  assert!(!adam.bias_correction());
  Ok(())
}

// ── Adam None-state arm + skip-absent-grad-key ────────────────────────────

#[test]
fn adam_step_none_state_arm_via_uninit_grad_key() -> Result<()> {
  // `adam_step`'s `None` match arm (no prior `(m, v)` for the key) is only
  // reachable when a gradient key was NOT pre-initialized: explicitly
  // `init` with a SUBSET of params, then `apply_gradients` with an extra
  // grad key that IS present in params. Because state is already non-empty,
  // `apply_gradients` skips its lazy re-init, so the extra key falls through
  // to the `None => (zeros_like, zeros_like)` arm — equivalent to a fresh
  // moment, so the closed-form no-bias-correction step applies to both.
  //   w=1.0, g=0.5, lr=0.001 ⇒ w_new ≈ 0.9968377 (see the Python-ref oracle).
  let mut adam = Adam::default_with_lr(0.001)?;
  // init only "a".
  let mut init_params: Weights = HashMap::new();
  init_params.insert("a".into(), scalar(1.0)?);
  adam.init(&init_params)?;
  assert!(
    !adam.state.is_empty(),
    "explicit init populated state for 'a'"
  );
  // params + grads both carry "a" (initialized) and "b" (NOT initialized).
  let mut params: Weights = HashMap::new();
  params.insert("a".into(), scalar(1.0)?);
  params.insert("b".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("a".into(), scalar(0.5)?);
  grads.insert("b".into(), scalar(0.5)?);
  adam.apply_gradients(&grads, &mut params)?;
  // "b" took the None-arm fresh-moment step and matches the closed form.
  let got_b = read_scalar(&params["b"])?;
  assert!((got_b - 0.996_837_7).abs() < 1e-5, "b got {got_b}");
  // "a" (pre-initialized to zero moments) took the same first step.
  let got_a = read_scalar(&params["a"])?;
  assert!((got_a - 0.996_837_7).abs() < 1e-5, "a got {got_a}");
  Ok(())
}

#[test]
fn adam_skips_grad_key_absent_from_params() -> Result<()> {
  // A gradient whose key has no matching parameter must be skipped (the
  // `let Some(param) = params.get(key) else { continue }` guard), leaving
  // the present parameter updated and the absent one never materialized.
  let mut adam = Adam::default_with_lr(0.001)?;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), scalar(0.5)?);
  grads.insert("absent".into(), scalar(0.5)?);
  adam.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.996_837_7).abs() < 1e-5, "present got {got}");
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added to params"
  );
  Ok(())
}

// ── AdamW constructor / getters / builders / trait methods ────────────────

#[test]
fn adamw_getters_echo_inputs() -> Result<()> {
  // Exercises the `AdamW::new` Ok body + `weight_decay` getter + the
  // `step()` / `learning_rate()` trait methods on a fresh optimizer.
  let adamw = AdamW::new(LearningRate::Fixed(0.02), (0.8, 0.99), 1e-7, 0.05, true)?;
  assert_eq!(adamw.weight_decay(), 0.05);
  assert_eq!(adamw.learning_rate(), 0.02);
  assert_eq!(adamw.step(), 0);
  Ok(())
}

#[test]
fn adamw_default_with_lr_weight_decay_default() -> Result<()> {
  let adamw = AdamW::default_with_lr(0.001)?;
  assert_eq!(adamw.weight_decay(), 0.01);
  Ok(())
}

#[test]
fn adamw_builder_success_paths_echo() -> Result<()> {
  // `with_learning_rate` (success arm, resolves at step 0) +
  // `with_weight_decay` (success arm echoes the set value).
  let adamw = AdamW::default_with_lr(0.001)?
    .with_learning_rate(LearningRate::Fixed(0.03))?
    .with_weight_decay(0.2)?;
  assert_eq!(adamw.learning_rate(), 0.03);
  assert_eq!(adamw.weight_decay(), 0.2);
  Ok(())
}

#[test]
fn adamw_init_then_step_advances_trait_methods() -> Result<()> {
  // Exercises AdamW's `init` + `preflight` delegation and the trait
  // `step()` / `learning_rate()` across a real step. The closed-form first
  // step (wd=0.01) matches `adamw_decoupled_weight_decay_applies_before_step`.
  let mut adamw = AdamW::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  // Explicit init populates inner state (covers the trait `init` forward).
  adamw.init(&params)?;
  assert_eq!(adamw.step(), 0);
  adamw.apply_gradients(&grads, &mut params)?;
  assert_eq!(adamw.step(), 1);
  assert_eq!(adamw.learning_rate(), 0.001);
  let got = read_scalar(&params["w"])?;
  assert!((got - 0.996_827_7).abs() < 1e-5, "got {got}");
  Ok(())
}

#[test]
fn adamw_skips_grad_key_absent_from_params() -> Result<()> {
  // AdamW's own `apply_gradients` guard (`params.get(key) else continue`).
  let mut adamw = AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01, false)?;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), scalar(0.5)?);
  grads.insert("absent".into(), scalar(0.5)?);
  adamw.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.996_827_7).abs() < 1e-5, "present got {got}");
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added to params"
  );
  Ok(())
}

#[test]
fn adamw_two_steps_advance_state() -> Result<()> {
  // Two consecutive applies: covers AdamW's preflight cache re-resolution
  // path (stamp at Some(0) → re-resolve at step 1) and confirms monotone
  // decrease of the weight.
  let mut adamw = AdamW::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adamw.apply_gradients(&grads, &mut params)?;
  let after_one = read_scalar(&params["w"])?;
  adamw.apply_gradients(&grads, &mut params)?;
  let after_two = read_scalar(&params["w"])?;
  assert!(after_two < after_one, "weight should keep decreasing");
  assert_eq!(adamw.step(), 2);
  Ok(())
}

// ── Adamax config getters / builders / trait methods ──────────────────────

#[test]
fn adamax_getters_echo_inputs() -> Result<()> {
  // Covers `learning_rate_ref`, `betas`, `eps`, `step()`, `learning_rate()`.
  let adamax = Adamax::new(LearningRate::Fixed(0.02), (0.8, 0.99), 1e-7)?;
  assert!(adamax.learning_rate_ref().is_fixed());
  assert_eq!(adamax.betas(), (0.8, 0.99));
  assert_eq!(adamax.eps(), 1e-7);
  assert_eq!(adamax.learning_rate(), 0.02);
  assert_eq!(adamax.step(), 0);
  Ok(())
}

#[test]
fn adamax_default_with_lr_getters() -> Result<()> {
  let adamax = Adamax::default_with_lr(0.001)?;
  assert_eq!(adamax.betas(), (0.9, 0.999));
  assert_eq!(adamax.eps(), 1e-8);
  assert!(adamax.learning_rate_ref().is_fixed());
  Ok(())
}

#[test]
fn adamax_builder_success_paths_echo() -> Result<()> {
  // `with_learning_rate` (success arm) + `with_betas` (success arm) +
  // `with_eps` (success arm) all echo their inputs.
  let adamax = Adamax::default_with_lr(0.001)?
    .with_learning_rate(LearningRate::Fixed(0.05))?
    .with_betas((0.7, 0.95))?
    .with_eps(2e-7)?;
  assert_eq!(adamax.learning_rate(), 0.05);
  assert!(adamax.learning_rate_ref().is_fixed());
  assert_eq!(adamax.betas(), (0.7, 0.95));
  assert_eq!(adamax.eps(), 2e-7);
  Ok(())
}

#[test]
fn adamax_init_then_two_steps_preflight_re_resolves() -> Result<()> {
  // First apply hits preflight's step-0 cache (stamped by `new`); the second
  // apply lands at step_count=1 with the stamp still Some(0), so `preflight`
  // re-resolves the fixed LR via its non-cache body. Also covers the trait
  // `init`, `step()`, and `learning_rate()`.
  let mut adamax = Adamax::default_with_lr(0.001)?;
  let (mut params, grads) = p_g(1.0, 0.5)?;
  adamax.init(&params)?;
  assert_eq!(adamax.step(), 0);
  adamax.apply_gradients(&grads, &mut params)?;
  let after_one = read_scalar(&params["w"])?;
  assert_eq!(adamax.step(), 1);
  adamax.apply_gradients(&grads, &mut params)?;
  let after_two = read_scalar(&params["w"])?;
  assert_eq!(adamax.step(), 2);
  assert_eq!(adamax.learning_rate(), 0.001);
  assert!(after_two < after_one, "weight should keep decreasing");
  Ok(())
}

#[test]
fn adamax_step_none_state_arm_via_uninit_grad_key() -> Result<()> {
  // Adamax's per-key `None` state arm (line in `apply_gradients`) — same
  // technique as Adam: explicit init of a SUBSET, then an extra present
  // grad key falls through to `None => (zeros_like, zeros_like)`.
  //   w=1.0, g=0.5, lr=0.001:
  //     m = (1-β₁)·g = 0.05; v = max(β₂·0, |g|) = 0.5
  //     step = lr·m / (v + eps) = 1e-4 ⇒ w_new ≈ 0.9999
  let mut adamax = Adamax::default_with_lr(0.001)?;
  let mut init_params: Weights = HashMap::new();
  init_params.insert("a".into(), scalar(1.0)?);
  adamax.init(&init_params)?;
  let mut params: Weights = HashMap::new();
  params.insert("a".into(), scalar(1.0)?);
  params.insert("b".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("a".into(), scalar(0.5)?);
  grads.insert("b".into(), scalar(0.5)?);
  adamax.apply_gradients(&grads, &mut params)?;
  let got_b = read_scalar(&params["b"])?;
  assert!((got_b - 0.9999).abs() < 1e-5, "b got {got_b}");
  Ok(())
}

#[test]
fn adamax_skips_grad_key_absent_from_params() -> Result<()> {
  // Adamax's `apply_gradients` skip guard for a grad key absent from params.
  let mut adamax = Adamax::default_with_lr(0.001)?;
  let mut params: Weights = HashMap::new();
  params.insert("present".into(), scalar(1.0)?);
  let mut grads: Weights = HashMap::new();
  grads.insert("present".into(), scalar(0.5)?);
  grads.insert("absent".into(), scalar(0.5)?);
  adamax.apply_gradients(&grads, &mut params)?;
  let got = read_scalar(&params["present"])?;
  assert!((got - 0.9999).abs() < 1e-5, "present got {got}");
  assert!(
    !params.contains_key("absent"),
    "absent grad must not be added to params"
  );
  Ok(())
}
