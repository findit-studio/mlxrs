use super::*;
use crate::lm::tuner::optimizers::sgd::SGD;

// Tiny in-memory dataset for trainer tests.
struct FakeDataset {
  samples: Vec<Example>,
}
impl FakeDataset {
  fn new(n: usize, len: usize) -> Self {
    let samples = (0..n)
      .map(|i| ((0..len).map(|k| ((i + k) as u32) % 32).collect(), 0_usize))
      .collect();
    Self { samples }
  }
}
impl Dataset for FakeDataset {
  fn len(&self) -> usize {
    self.samples.len()
  }
  fn get(&self, _idx: usize) -> Result<&serde_json::Value> {
    Err(Error::InvariantViolation(InvariantViolationPayload::new(
      "FakeDataset::get",
      "is not used by the trainer iterator",
    )))
  }
  fn process(&self, idx: usize) -> Result<Example> {
    Ok((self.samples[idx].0.clone(), self.samples[idx].1))
  }
}

// Tiny model: returns vocab=8 uniform logits, ignores cache.
struct FakeModel;
impl Model for FakeModel {
  fn forward(&self, tokens: &Array, _cache: &mut [Box<dyn KvCache>]) -> Result<Array> {
    let shape = tokens.shape();
    let (b, s) = (shape[0], shape[1]);
    // Uniform logits over vocab=8.
    let vocab = 8;
    let n = b * s * vocab;
    let buf = vec![0.1_f32; n];
    Array::from_slice::<f32>(&buf, &(b, s, vocab))
  }
}

#[test]
fn training_args_default_matches_python() {
  let a = TrainingArgs::default();
  assert_eq!(a.batch_size(), 4);
  assert_eq!(a.iters(), 100);
  assert_eq!(a.val_batches(), Some(25));
  assert_eq!(a.steps_per_report(), 10);
  assert_eq!(a.steps_per_eval(), 200);
  assert_eq!(a.steps_per_save(), 100);
  assert_eq!(a.max_seq_length(), 2048);
  assert!(!a.grad_checkpoint());
  assert_eq!(a.grad_accumulation_steps(), 1);
  // mlxrs-specific: defaults to false so a fresh `TrainingArgs` cannot
  // accidentally run the v1 mechanics-only stub.
  assert!(!a.acknowledge_no_real_gradients());
}

#[test]
fn default_loss_matches_masked_cross_entropy() -> Result<()> {
  // FakeModel returns uniform vocab=8 logits regardless of input. We
  // construct a small [B=1, S=3] batch with lengths=(1,3): mask is at
  // positions {step : step >= 1 && step < 3} = {1, 2} of the [S-1=2]-
  // element target → 2 tokens contribute. The exclusive upper bound
  // (`<`) doesn't change this case because the target range stops at
  // T=2 (steps never reach step==length=3); see
  // `default_loss_excludes_padded_target_at_length_boundary` for the
  // regression that exercises the boundary.
  let model = FakeModel;
  // batch [B=1, S=3]: tokens [1, 2, 3]
  let batch = Array::from_slice::<i32>(&[1, 2, 3], &(1, 3))?;
  // lengths [B=1, 2]: (offset=1, length=3)
  let lengths = Array::from_slice::<i32>(&[1, 3], &(1, 2))?;
  let (mut loss, mut ntoks) = default_loss(&model, &batch, &lengths)?;
  let loss_v = loss.item::<f32>()?;
  let ntoks_v = ntoks.item::<f32>()?;
  // Uniform logits → cross-entropy per token = log(8) ≈ 2.0794
  assert!((loss_v - 8.0_f32.ln()).abs() < 1e-4, "got loss {loss_v}");
  // mask at positions {1,2} of the 2-element target → ntoks=2.
  assert!((ntoks_v - 2.0).abs() < 1e-6, "got ntoks {ntoks_v}");
  Ok(())
}

#[test]
fn default_loss_excludes_padded_target_at_length_boundary() -> Result<()> {
  // Construct a [B=1, S=4] batch with lengths=(0, 2). Valid tokens are
  // at positions [0, 2): batch[0], batch[1]. Positions batch[2..4) are
  // padding (zeros). After shifting, targets has [S-1=3] positions:
  //   target[0] = batch[1] (valid)
  //   target[1] = batch[2] (PAD — boundary)
  //   target[2] = batch[3] (PAD)
  // The mask `steps >= offset && steps < length` with offset=0 and
  // length=2 keeps steps ∈ {1} (since arange runs over [1, 4)):
  //   step 1: 1 >= 0 AND 1 < 2 → ✓ (target[0] = batch[1] = valid)
  //   step 2: 2 >= 0 AND 2 < 2 → ✗ (would be target[1] = batch[2] = PAD)
  //   step 3: 3 >= 0 AND 3 < 2 → ✗
  // An inclusive `<=` upper bound would INCLUDE step 2 (the boundary
  // pad), counting batch[2] = 0 as a supervised target and skewing
  // training toward predicting the pad token. ntoks must be 1, not 2.
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[1, 2, 0, 0], &(1, 4))?;
  let lengths = Array::from_slice::<i32>(&[0, 2], &(1, 2))?;
  let (mut loss, mut ntoks) = default_loss(&model, &batch, &lengths)?;
  let loss_v = loss.item::<f32>()?;
  let ntoks_v = ntoks.item::<f32>()?;
  assert!(
    (ntoks_v - 1.0).abs() < 1e-6,
    "expected ntoks=1 (boundary pad excluded by `<` upper bound), got {ntoks_v}",
  );
  // Single supervised token, uniform logits over vocab=8 → loss = log(8).
  assert!(
    (loss_v - 8.0_f32.ln()).abs() < 1e-4,
    "expected loss=log(8) for single supervised token, got {loss_v}",
  );
  Ok(())
}

#[test]
fn iterate_batches_emits_expected_shape_for_known_dataset_size() -> Result<()> {
  let dataset = FakeDataset::new(8, 4); // 8 examples × len 4
  let iter = iterate_batches(&dataset, 4, 64, false, None)?;
  let mut count = 0;
  for b in iter {
    let b = b?;
    assert_eq!(b.tokens_ref().shape()[0], 4);
    assert_eq!(b.lengths_ref().shape(), &[4, 2]);
    count += 1;
  }
  assert_eq!(count, 2, "8/4=2 batches expected");
  Ok(())
}

#[test]
fn iterate_batches_rejects_too_small_dataset() {
  let dataset = FakeDataset::new(2, 4);
  let res = iterate_batches(&dataset, 4, 64, false, None);
  assert!(res.is_err());
}

#[test]
fn iterate_batches_loop_forever_yields_more_batches_than_dataset_size() -> Result<()> {
  let dataset = FakeDataset::new(4, 4); // 1 batch per pass
  let mut iter = iterate_batches(&dataset, 4, 64, true, Some(0xCAFE))?;
  // Take 5 batches (way more than 1 per pass) — must not exhaust.
  for _ in 0..5 {
    assert!(iter.next().is_some());
  }
  Ok(())
}

#[test]
fn evaluate_returns_correct_loss_for_known_eval_set() -> Result<()> {
  let dataset = FakeDataset::new(4, 6); // 1 batch of 4 examples
  let model = FakeModel;
  let loss = evaluate(&model, &dataset, 4, Some(1), 64, |m, b, l| {
    default_loss(m, b, l)
  })?;
  // Uniform logits over vocab=8: cross-entropy per token = log(8) ≈ 2.0794.
  assert!((loss - 8.0_f32.ln()).abs() < 1e-4, "got {loss}");
  Ok(())
}

struct CountingCallback {
  train_reports: usize,
  val_reports: usize,
  saves: usize,
}
impl TrainingCallback for CountingCallback {
  fn on_train_loss_report(&mut self, _info: &TrainInfo) {
    self.train_reports += 1;
  }
  fn on_val_loss_report(&mut self, _info: &ValInfo) {
    self.val_reports += 1;
  }
  fn on_save(&mut self, _it: usize, _adapter_file: &str) -> Result<()> {
    self.saves += 1;
    Ok(())
  }
}

#[test]
fn train_completes_n_iters_with_progress_callback() -> Result<()> {
  let dataset = FakeDataset::new(4, 6); // 1 batch per pass
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut sgd = SGD::vanilla(0.01)?;
  let mut cb = CountingCallback {
    train_reports: 0,
    val_reports: 0,
    saves: 0,
  };
  let args = TrainingArgs::new()
    .with_iters(6)
    .with_steps_per_report(2)
    .with_steps_per_eval(4)
    .with_steps_per_save(3)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut sgd,
    &mut params,
    &dataset,
    Some(&dataset),
    &args,
    default_loss,
    &mut cb,
  )?;
  // 6 iters @ steps_per_report=2 → 3 windows (it=2,4,6).
  assert_eq!(cb.train_reports, 3);
  // val: it=1, it=4 (multiple of 4), it=6 (final). = 3 vals
  assert_eq!(cb.val_reports, 3);
  // save: it=3, it=6 plus final → 3 saves
  assert_eq!(cb.saves, 3);
  Ok(())
}

#[test]
fn grad_checkpoint_wraps_layer_without_changing_output() -> Result<()> {
  // x → x² wrapped in checkpoint produces the same forward value.
  let plain = |xs: &[Array]| Ok(vec![crate::ops::arithmetic::square(&xs[0])?]);
  let wrapped = grad_checkpoint(plain)?;
  let x = Array::full::<f32>(&[0i32; 0], 3.0)?;
  let mut out = wrapped(&[x])?;
  assert_eq!(out[0].item::<f32>()?, 9.0);
  Ok(())
}

// ─────────── acknowledge_no_real_gradients gate ───────────

#[test]
fn train_rejects_when_acknowledge_no_real_gradients_is_false() -> Result<()> {
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut sgd = SGD::vanilla(0.01)?;
  let mut cb = NoopCallback;
  // Default args leaves `acknowledge_no_real_gradients` = false.
  let args = TrainingArgs::new()
    .with_iters(1)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1));
  assert!(!args.acknowledge_no_real_gradients());
  let res = train(
    &model,
    &mut sgd,
    &mut params,
    &dataset,
    None,
    &args,
    default_loss,
    &mut cb,
  );
  match res {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(
        payload.context(),
        "train: TrainingArgs::acknowledge_no_real_gradients"
      );
      assert_eq!(
        payload.requirement(),
        "must be set to `true` to run the v1 mechanics-only training path"
      );
    }
    other => panic!("expected Err(InvariantViolation), got {other:?}"),
  }
  Ok(())
}

#[test]
fn train_runs_when_acknowledge_no_real_gradients_is_true() -> Result<()> {
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut sgd = SGD::vanilla(0.01)?;
  let mut cb = NoopCallback;
  let args = TrainingArgs::new()
    .with_iters(1)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true);
  let res = train(
    &model,
    &mut sgd,
    &mut params,
    &dataset,
    None,
    &args,
    default_loss,
    &mut cb,
  );
  assert!(
    res.is_ok(),
    "train should run when opt-in is set; got {res:?}"
  );
  Ok(())
}

// ─────────── zero-interval rejection ───────────

fn args_for_zero_interval_tests() -> TrainingArgs {
  TrainingArgs::new()
    .with_iters(1)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true)
}

fn run_train_with_args(args: &TrainingArgs) -> crate::Result<()> {
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut sgd = SGD::vanilla(0.01)?;
  let mut cb = NoopCallback;
  train(
    &model,
    &mut sgd,
    &mut params,
    &dataset,
    None,
    args,
    default_loss,
    &mut cb,
  )
}

#[test]
fn train_rejects_zero_steps_per_report() {
  let args = args_for_zero_interval_tests().with_steps_per_report(0);
  let res = run_train_with_args(&args);
  match res {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(payload.context(), "train: steps_per_report");
      assert_eq!(payload.requirement(), "must be >= 1");
    }
    other => panic!("expected Err(InvariantViolation) for steps_per_report=0; got {other:?}"),
  }
}

#[test]
fn train_rejects_zero_steps_per_eval() {
  let args = args_for_zero_interval_tests().with_steps_per_eval(0);
  let res = run_train_with_args(&args);
  match res {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(payload.context(), "train: steps_per_eval");
      assert_eq!(payload.requirement(), "must be >= 1");
    }
    other => panic!("expected Err(InvariantViolation) for steps_per_eval=0; got {other:?}"),
  }
}

#[test]
fn train_rejects_zero_steps_per_save() {
  let args = args_for_zero_interval_tests().with_steps_per_save(0);
  let res = run_train_with_args(&args);
  match res {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(payload.context(), "train: steps_per_save");
      assert_eq!(payload.requirement(), "must be >= 1");
    }
    other => panic!("expected Err(InvariantViolation) for steps_per_save=0; got {other:?}"),
  }
}

#[test]
fn train_rejects_zero_grad_accumulation_steps() {
  let args = args_for_zero_interval_tests().with_grad_accumulation_steps(0);
  let res = run_train_with_args(&args);
  match res {
    Err(Error::InvariantViolation(payload)) => {
      assert_eq!(payload.context(), "train: grad_accumulation_steps");
      assert_eq!(payload.requirement(), "must be >= 1");
    }
    other => {
      panic!("expected Err(InvariantViolation) for grad_accumulation_steps=0; got {other:?}")
    }
  }
}

// ─────────── grad accumulation respects window cadence ───────────

/// Counting optimizer wrapper: counts `apply_gradients` invocations
/// without modifying params. Used to assert the train loop fires the
/// optimizer at the OPTIMIZER STEP cadence (one call per accumulation
/// window completion) rather than per microbatch.
struct CountingOptimizer {
  apply_calls: usize,
  step_count: usize,
  lr: f32,
}
impl crate::lm::tuner::optimizers::Optimizer for CountingOptimizer {
  fn init(&mut self, _params: &Weights) -> Result<()> {
    Ok(())
  }
  fn apply_gradients(&mut self, _gradients: &Weights, _params: &mut Weights) -> Result<()> {
    self.apply_calls += 1;
    self.step_count += 1;
    Ok(())
  }
  fn step(&self) -> usize {
    self.step_count
  }
  fn learning_rate(&self) -> f32 {
    self.lr
  }
}

fn build_train_fixture() -> Result<(
  FakeModel,
  FakeDataset,
  Weights,
  NoopCallback,
  CountingOptimizer,
)> {
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let cb = NoopCallback;
  let opt = CountingOptimizer {
    apply_calls: 0,
    step_count: 0,
    lr: 0.0,
  };
  Ok((model, dataset, params, cb, opt))
}

#[test]
fn grad_accumulation_steps_2_calls_optimizer_every_other_iter() -> Result<()> {
  // iters=10, grad_accumulation_steps=2 → 5 optimizer calls.
  let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
  let args = TrainingArgs::new()
      .with_iters(10)
      .with_grad_accumulation_steps(2)
      // steps_per_* large enough to avoid firing during this test (we
      // only care about optimizer call count here).
      .with_steps_per_report(100)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut opt,
    &mut params,
    &dataset,
    None,
    &args,
    default_loss,
    &mut cb,
  )?;
  assert_eq!(
    opt.apply_calls, 5,
    "iters=10 + grad_accumulation_steps=2 must produce 5 optimizer steps; got {}",
    opt.apply_calls,
  );
  Ok(())
}

#[test]
fn grad_accumulation_steps_partial_window_at_end_drops() -> Result<()> {
  // iters=11, grad_accumulation_steps=4 → only 11/4 = 2 complete
  // windows (microbatches 1..=4 → step 1, 5..=8 → step 2). The final
  // 3 microbatches (9, 10, 11) form a partial window which is DROPPED
  // (no third optimizer call).
  let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
  let args = TrainingArgs::new()
    .with_iters(11)
    .with_grad_accumulation_steps(4)
    .with_steps_per_report(100)
    .with_steps_per_eval(100)
    .with_steps_per_save(100)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut opt,
    &mut params,
    &dataset,
    None,
    &args,
    default_loss,
    &mut cb,
  )?;
  assert_eq!(
    opt.apply_calls, 2,
    "iters=11 + grad_accumulation_steps=4 must drop the final partial \
       window of 3 microbatches; expected 2 optimizer calls, got {}",
    opt.apply_calls,
  );
  Ok(())
}

#[test]
fn grad_accumulation_steps_1_is_identity_to_microbatch_count() -> Result<()> {
  // The grad_accumulation_steps=1 case must NOT regress — every
  // microbatch is its own optimizer step.
  let (model, dataset, mut params, mut cb, mut opt) = build_train_fixture()?;
  let args = TrainingArgs::new()
    .with_iters(7)
    .with_grad_accumulation_steps(1)
    .with_steps_per_report(100)
    .with_steps_per_eval(100)
    .with_steps_per_save(100)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut opt,
    &mut params,
    &dataset,
    None,
    &args,
    default_loss,
    &mut cb,
  )?;
  assert_eq!(opt.apply_calls, 7);
  Ok(())
}

// ─────────── report-loss denominator parity with mlx-lm ───────────

/// Recording callback: captures `train_loss` from every
/// `on_train_loss_report` call. Used to prove the report denominator
/// matches the per-microbatch loss (mlx-lm parity) instead of the
/// optimizer-step count (which would inflate every reported loss by
/// `grad_accumulation_steps×`).
struct LossRecordingCallback {
  losses: Vec<f32>,
}
impl TrainingCallback for LossRecordingCallback {
  fn on_train_loss_report(&mut self, info: &TrainInfo) {
    self.losses.push(info.train_loss());
  }
}

#[test]
fn grad_accumulation_steps_4_reports_constant_loss_at_2_not_8() -> Result<()> {
  // Regression guard against loss inflation: when each microbatch's loss
  // is constant 2.0 and `grad_accumulation_steps = 4`, summing one term
  // per microbatch into `window_loss` and then dividing by `window_steps`
  // (which only increments per completed accumulation window) would
  // report an 8.0 loss — every callback / log line / early-stop monitor
  // would see the per-microbatch loss multiplied by 4×.
  //
  // The denominator is the completed-microbatch count, so every report
  // fires at the true per-microbatch loss 2.0.
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut opt = CountingOptimizer {
    apply_calls: 0,
    step_count: 0,
    lr: 0.0,
  };
  let mut cb = LossRecordingCallback { losses: Vec::new() };
  // Constant-loss mock: returns (loss=2.0, ntoks=1.0) for every
  // microbatch, regardless of inputs. Drives `window_loss` to grow by
  // exactly +2.0 per microbatch so the inflation factor is unambiguous.
  let const_loss_fn =
    |_m: &FakeModel, _batch: &Array, _lengths: &Array| -> Result<(Array, Array)> {
      let loss = Array::full::<f32>(&[0i32; 0], 2.0)?;
      let ntoks = Array::full::<f32>(&[0i32; 0], 1.0)?;
      Ok((loss, ntoks))
    };
  let args = TrainingArgs::new()
      .with_iters(12)
      .with_grad_accumulation_steps(4)
      // 12 microbatches / 4 = 3 optimizer steps. steps_per_report=1 fires
      // a report on EVERY optimizer step, so we get 3 callbacks
      // (windows of 4 microbatches each, all with constant per-microbatch
      // loss = 2.0).
      .with_steps_per_report(1)
      .with_steps_per_eval(100)
      .with_steps_per_save(100)
      .with_batch_size(4)
      .with_max_seq_length(64)
      .with_val_batches(Some(1))
      .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut opt,
    &mut params,
    &dataset,
    None,
    &args,
    const_loss_fn,
    &mut cb,
  )?;
  assert_eq!(
    cb.losses.len(),
    3,
    "iters=12 + grad_accumulation_steps=4 + steps_per_report=1 must fire 3 train-loss reports; got {}",
    cb.losses.len(),
  );
  for (i, &loss) in cb.losses.iter().enumerate() {
    assert!(
      (loss - 2.0).abs() < 1e-6,
      "report #{i} train_loss = {loss}, expected 2.0 (per-microbatch loss); dividing \
         `window_loss / window_steps` (4×constant-2.0 by 1 optimizer-step) would wrongly \
         report 8.0",
    );
  }
  Ok(())
}

// ─────────── default_loss rejects zero-supervised-token batches ───────────

#[test]
fn default_loss_rejects_zero_token_batch_after_mask() -> Result<()> {
  // Construct a [B=2, S=2] batch where BOTH rows have lengths=(0, 1).
  // - Shifted targets has T=S-1=1 position; arange runs over [1, 2).
  // - Mask is `steps >= 0 && steps < 1` over step ∈ {1}: never true.
  // With the exclusive `<` upper bound, mask.sum() == 0 → without the
  // zero-token guard, `ce_sum / ntoks` would produce NaN/Inf and poison
  // every downstream accumulator (`train`'s `window_loss`, `evaluate`'s
  // `total_loss`) silently. The guard returns an explicit `Backend`
  // error before the divide so the caller filters the offending rows.
  let model = FakeModel;
  // Two rows, two tokens each; padding is fine since the mask zeros
  // every position out anyway.
  let batch = Array::from_slice::<i32>(&[0, 0, 0, 0], &(2, 2))?;
  // Both rows: offset=0, length=1.
  let lengths = Array::from_slice::<i32>(&[0, 1, 0, 1], &(2, 2))?;
  let err = default_loss(&model, &batch, &lengths)
    .expect_err("expected default_loss to reject zero-token batch");
  match err {
    Error::EmptyInput(p) => {
      assert!(
        p.context().contains("0 supervised tokens"),
        "expected context to mention '0 supervised tokens', got: {}",
        p.context(),
      );
    }
    other => panic!("expected Error::EmptyInput, got: {other:?}"),
  }
  Ok(())
}

#[test]
fn default_loss_rejects_lengths_with_extra_batch_row() -> Result<()> {
  // batch is [B=2, S=2] but lengths is [B+1=3, 2] — a rank-only guard
  // would accept this and silently slice only the first 2 rows,
  // building masks from mismatched metadata. The full-shape guard must
  // reject up-front.
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2))?;
  let lengths = Array::from_slice::<i32>(&[0, 2, 0, 2, 0, 2], &(3, 2))?;
  let err = default_loss(&model, &batch, &lengths)
    .expect_err("expected ShapePairMismatch for extra length row");
  match err {
    Error::ShapePairMismatch(p) => {
      assert_eq!(p.expected(), &[2_usize, 2_usize][..]);
      assert_eq!(p.actual(), &[3_usize, 2_usize][..]);
    }
    other => panic!("expected Error::ShapePairMismatch, got: {other:?}"),
  }
  Ok(())
}

#[test]
fn default_loss_rejects_lengths_with_missing_batch_row() -> Result<()> {
  // batch is [B=2, S=2] but lengths is [B-1=1, 2] — a rank-only guard
  // would accept this too (rank=2 and dim[1]=2 both held), then the
  // per-row slice would either OOB or silently truncate metadata. The
  // full-shape guard must reject up-front.
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[1, 2, 3, 4], &(2, 2))?;
  let lengths = Array::from_slice::<i32>(&[0, 2], &(1, 2))?;
  let err = default_loss(&model, &batch, &lengths)
    .expect_err("expected ShapePairMismatch for missing length row");
  match err {
    Error::ShapePairMismatch(p) => {
      assert_eq!(p.expected(), &[2_usize, 2_usize][..]);
      assert_eq!(p.actual(), &[1_usize, 2_usize][..]);
    }
    other => panic!("expected Error::ShapePairMismatch, got: {other:?}"),
  }
  Ok(())
}

// ─────────── TrainingArgs: remaining getters + builders ───────────

#[test]
fn training_args_clear_cache_threshold_getter_and_builder() {
  // Default is 0 (disabled); the builder overwrites the field and the
  // getter reads it back verbatim. Closed-form: input == output.
  let a = TrainingArgs::new();
  assert_eq!(a.clear_cache_threshold(), 0);
  let a = a.with_clear_cache_threshold(4096);
  assert_eq!(a.clear_cache_threshold(), 4096);
}

#[test]
fn training_args_adapter_file_builder_accepts_str_and_string() {
  // `with_adapter_file(impl Into<String>)` — exercise both a &str and a
  // String argument; getter returns the exact stored path.
  let a = TrainingArgs::new();
  // Python default path.
  assert_eq!(a.adapter_file(), "adapters.safetensors");
  let a = a.with_adapter_file("custom/path.safetensors");
  assert_eq!(a.adapter_file(), "custom/path.safetensors");
  let owned = String::from("owned/adapters.safetensors");
  let a = a.with_adapter_file(owned);
  assert_eq!(a.adapter_file(), "owned/adapters.safetensors");
}

#[test]
fn training_args_grad_checkpoint_builder_flips_flag() {
  // Default false → builder sets true → getter observes true.
  let a = TrainingArgs::new();
  assert!(!a.grad_checkpoint());
  let a = a.with_grad_checkpoint(true);
  assert!(a.grad_checkpoint());
}

// ─────────── default_loss: shape-guard branches ───────────

#[test]
fn default_loss_rejects_non_rank_2_batch() -> Result<()> {
  // A rank-1 batch must be rejected with RankMismatch BEFORE any forward
  // pass. Closed-form: actual rank = 1, actual_shape = [3].
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[1, 2, 3], &(3usize,))?;
  // `lengths` shape is irrelevant — the rank guard fires first.
  let lengths = Array::from_slice::<i32>(&[0, 3], &(1, 2))?;
  let err =
    default_loss(&model, &batch, &lengths).expect_err("expected RankMismatch for rank-1 batch");
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.context(), "default_loss: batch must be rank-2 [B, S]");
      assert_eq!(p.actual(), 1);
      assert_eq!(p.actual_shape(), &[3_usize][..]);
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
  Ok(())
}

#[test]
fn default_loss_rejects_rank_3_batch() -> Result<()> {
  // A rank-3 batch also fails the `[b, s]` match arm. Closed-form: actual
  // rank = 3, actual_shape = [1, 1, 2].
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[1, 2], &(1usize, 1usize, 2usize))?;
  let lengths = Array::from_slice::<i32>(&[0, 2], &(1, 2))?;
  let err =
    default_loss(&model, &batch, &lengths).expect_err("expected RankMismatch for rank-3 batch");
  match err {
    Error::RankMismatch(p) => {
      assert_eq!(p.actual(), 3);
      assert_eq!(p.actual_shape(), &[1_usize, 1, 2][..]);
    }
    other => panic!("expected Error::RankMismatch, got: {other:?}"),
  }
  Ok(())
}

#[test]
fn default_loss_rejects_seq_len_below_2() -> Result<()> {
  // S=1 cannot form a next-token (input, target) pair, so default_loss must
  // reject with OutOfRange before forwarding. Closed-form: value = "1".
  let model = FakeModel;
  let batch = Array::from_slice::<i32>(&[5], &(1usize, 1usize))?;
  let lengths = Array::from_slice::<i32>(&[0, 1], &(1, 2))?;
  let err = default_loss(&model, &batch, &lengths).expect_err("expected OutOfRange for S < 2");
  match err {
    Error::OutOfRange(p) => {
      assert_eq!(p.context(), "default_loss: batch S");
      assert_eq!(p.requirement(), "must be >= 2 for next-token prediction");
      assert_eq!(p.value(), "1");
    }
    other => panic!("expected Error::OutOfRange, got: {other:?}"),
  }
  Ok(())
}

// ─────────── TrainInfo / ValInfo accessors ───────────

#[test]
fn train_info_accessors_return_constructor_inputs() {
  // Closed-form: every getter must echo the exact value passed to `new`.
  let info = TrainInfo::new(7, 1.5, 0.001, 12.0, 480.0, 3_200);
  assert_eq!(info.iteration(), 7);
  assert_eq!(info.train_loss(), 1.5);
  assert_eq!(info.learning_rate(), 0.001);
  assert_eq!(info.iterations_per_second(), 12.0);
  assert_eq!(info.tokens_per_second(), 480.0);
  assert_eq!(info.trained_tokens(), 3_200);
}

#[test]
fn val_info_accessors_return_constructor_inputs() {
  // Closed-form: getters echo the constructor inputs verbatim.
  let info = ValInfo::new(42, 2.25, 0.75);
  assert_eq!(info.iteration(), 42);
  assert_eq!(info.val_loss(), 2.25);
  assert_eq!(info.val_time(), 0.75);
}

#[test]
fn noop_callback_default_methods_are_no_ops() -> Result<()> {
  // The default trait impls (train/val report + save) must be reachable and
  // side-effect-free for `NoopCallback`. on_save returns Ok(()).
  let mut cb = NoopCallback;
  cb.on_train_loss_report(&TrainInfo::new(1, 0.0, 0.0, 0.0, 0.0, 0));
  cb.on_val_loss_report(&ValInfo::new(0, 0.0, 0.0));
  cb.on_save(3, "adapters.safetensors")?;
  Ok(())
}

// ─────────── iterate_batches / build_batch / shuffle internals ───────────

#[test]
fn build_batch_clamps_padded_length_to_max_seq_length() -> Result<()> {
  // Examples are length 4; the pad heuristic wants
  //   1 + 32 * ceil(4/32) = 1 + 32 = 33
  // columns, but max_seq_length = 8 forces the clamp to 8. The yielded
  // token tensor must therefore be [B=4, 8], proving line 816's clamp ran.
  let dataset = FakeDataset::new(4, 4);
  let iter = iterate_batches(&dataset, 4, 8, false, None)?;
  let mut saw = false;
  for b in iter {
    let b = b?;
    assert_eq!(
      b.tokens_ref().shape(),
      &[4, 8],
      "padded width must be clamped to max_seq_length=8 (un-clamped would be 33)",
    );
    saw = true;
  }
  assert!(saw, "expected at least one batch");
  Ok(())
}

#[test]
fn fisher_yates_shuffle_is_a_deterministic_permutation() {
  // Independent oracle: a Fisher-Yates shuffle is a PERMUTATION, so the
  // multiset of elements is invariant; and it is seeded-deterministic, so
  // the same seed yields the identical ordering on a repeat. Both checks
  // avoid re-deriving the SplitMix64 stream by hand.
  let mut a: Vec<usize> = (0..16).collect();
  let mut b: Vec<usize> = (0..16).collect();
  fisher_yates_shuffle(&mut a, 0xDEAD_BEEF);
  fisher_yates_shuffle(&mut b, 0xDEAD_BEEF);
  assert_eq!(a, b, "same seed must produce the same permutation");
  let mut sorted = a.clone();
  sorted.sort_unstable();
  assert_eq!(
    sorted,
    (0..16).collect::<Vec<_>>(),
    "shuffle must be a permutation (no lost/duplicated elements)",
  );
}

#[test]
fn fisher_yates_shuffle_different_seeds_can_differ() {
  // Two distinct seeds over a 16-element slice should not collapse to the
  // identical ordering (guards against the swap body being a no-op).
  let mut a: Vec<usize> = (0..16).collect();
  let mut b: Vec<usize> = (0..16).collect();
  fisher_yates_shuffle(&mut a, 1);
  fisher_yates_shuffle(&mut b, 999_999);
  assert_ne!(a, b, "distinct seeds should yield distinct permutations");
}

#[test]
fn iterate_batches_shuffle_over_multiple_batches_runs_shuffle_body() -> Result<()> {
  // 8 examples / batch_size 4 = 2 batch groups. With a shuffle seed the
  // BatchIter refreshes `order` = [0, 1] and runs fisher_yates_shuffle
  // (loop body executes for a 2-element order). We only assert the
  // iterator yields valid batches across more than one pass (loop_forever)
  // — the shuffle internals are covered by the unit tests above.
  let dataset = FakeDataset::new(8, 4);
  let mut iter = iterate_batches(&dataset, 4, 64, true, Some(0x1234))?;
  for _ in 0..4 {
    let b = iter.next().expect("loop_forever must not exhaust")?;
    assert_eq!(b.tokens_ref().shape()[0], 4);
  }
  Ok(())
}

#[test]
fn batch_iter_with_empty_batch_idx_yields_none() {
  // Directly drive a BatchIter whose `batch_idx` is empty (a state
  // `iterate_batches` rejects up front, but the iterator must still
  // terminate cleanly): the first `next()` refreshes `order` to empty and
  // hits the `order.is_empty()` early-None. loop_forever=true ensures the
  // first-pass guard does not short-circuit before reaching that branch.
  let dataset = FakeDataset::new(4, 4);
  let mut iter = BatchIter {
    dataset: &dataset,
    batch_idx: Vec::new(),
    max_seq_length: 64,
    cursor: 0,
    order: Vec::new(),
    loop_forever: true,
    shuffle_seed: None,
    rng_state: None,
    first_pass: true,
  };
  assert!(
    iter.next().is_none(),
    "empty batch_idx must yield None even with loop_forever=true",
  );
}

// ─────────── evaluate: cap + zero-token branches ───────────

#[test]
fn evaluate_stops_after_num_batches_cap() -> Result<()> {
  // 8 examples / batch_size 4 = 2 available batches, but num_batches=Some(1)
  // caps the loop at one batch (the `i >= cap` break fires at i=1). The
  // returned loss is the single batch's per-token loss = log(8) for uniform
  // vocab=8 logits (closed-form, independent of the second batch).
  let dataset = FakeDataset::new(8, 6);
  let model = FakeModel;
  let loss = evaluate(&model, &dataset, 4, Some(1), 64, |m, b, l| {
    default_loss(m, b, l)
  })?;
  assert!(
    (loss - 8.0_f32.ln()).abs() < 1e-4,
    "capped eval over 1 batch must report log(8); got {loss}",
  );
  Ok(())
}

#[test]
fn evaluate_rejects_eval_set_that_produces_no_tokens() {
  // A loss closure that always reports ntoks=0 keeps total_tokens at 0,
  // tripping the post-loop EmptyInput guard. Uses a custom closure (NOT
  // default_loss, which rejects zero-token batches earlier) so the loop
  // body runs to completion and the guard at the end is what fires.
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let zero_tok = |_m: &FakeModel, _b: &Array, _l: &Array| -> Result<(Array, Array)> {
    let loss = Array::full::<f32>(&[0i32; 0], 1.0)?;
    let ntoks = Array::full::<f32>(&[0i32; 0], 0.0)?;
    Ok((loss, ntoks))
  };
  let err = evaluate(&model, &dataset, 4, Some(1), 64, zero_tok)
    .expect_err("expected EmptyInput when eval produces no tokens");
  match err {
    Error::EmptyInput(p) => {
      assert!(
        p.context().contains("produced no batches with tokens"),
        "unexpected context: {}",
        p.context(),
      );
    }
    other => panic!("expected Error::EmptyInput, got: {other:?}"),
  }
}

// ─────────── train: iters == 0 fast path ───────────

#[test]
fn train_with_zero_iters_returns_ok_without_firing_callbacks() -> Result<()> {
  // iters=0 hits the early `return Ok(())` (before the interval-zero checks
  // and the loop), so NO save / report / eval callback ever fires —
  // including the final save hook, which lives after the loop.
  let dataset = FakeDataset::new(4, 6);
  let model = FakeModel;
  let mut params: Weights = HashMap::new();
  params.insert("w".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut sgd = SGD::vanilla(0.01)?;
  let mut cb = CountingCallback {
    train_reports: 0,
    val_reports: 0,
    saves: 0,
  };
  let args = TrainingArgs::new()
    .with_iters(0)
    .with_batch_size(4)
    .with_max_seq_length(64)
    .with_val_batches(Some(1))
    .with_acknowledge_no_real_gradients(true);
  train(
    &model,
    &mut sgd,
    &mut params,
    &dataset,
    Some(&dataset),
    &args,
    default_loss,
    &mut cb,
  )?;
  assert_eq!(cb.train_reports, 0, "no train reports for iters=0");
  assert_eq!(cb.val_reports, 0, "no eval for iters=0");
  assert_eq!(
    cb.saves, 0,
    "iters=0 returns before the final save hook, so no save fires",
  );
  Ok(())
}

// ─────────── add_weights: error branches ───────────

#[test]
fn add_weights_rejects_mismatched_key_counts() -> Result<()> {
  // lhs has 2 keys, rhs has 1 → LengthMismatch. Closed-form: expected=2
  // (lhs len), actual=1 (rhs len). Drives the private helper directly.
  let mut a: Weights = HashMap::new();
  a.insert("x".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  a.insert("y".into(), Array::full::<f32>(&[0i32; 0], 2.0)?);
  let mut b: Weights = HashMap::new();
  b.insert("x".into(), Array::full::<f32>(&[0i32; 0], 3.0)?);
  let err = add_weights(&a, &b).expect_err("expected LengthMismatch for key-count skew");
  match err {
    Error::LengthMismatch(p) => {
      assert_eq!(p.context(), "trainer::add_weights: lhs vs rhs key counts");
      assert_eq!(p.expected(), 2);
      assert_eq!(p.actual(), 1);
    }
    other => panic!("expected Error::LengthMismatch, got: {other:?}"),
  }
  Ok(())
}

#[test]
fn add_weights_rejects_key_present_in_lhs_but_missing_from_rhs() -> Result<()> {
  // Equal key counts but disjoint key sets: lhs={"x"}, rhs={"y"}. The loop
  // over lhs looks up "x" in rhs, fails, and returns MissingKey naming "x".
  let mut a: Weights = HashMap::new();
  a.insert("x".into(), Array::full::<f32>(&[0i32; 0], 1.0)?);
  let mut b: Weights = HashMap::new();
  b.insert("y".into(), Array::full::<f32>(&[0i32; 0], 2.0)?);
  let err = add_weights(&a, &b).expect_err("expected MissingKey for disjoint key sets");
  match err {
    Error::MissingKey(p) => {
      assert_eq!(p.context(), "trainer::add_weights: key missing from rhs");
      assert_eq!(p.key(), "x");
    }
    other => panic!("expected Error::MissingKey, got: {other:?}"),
  }
  Ok(())
}
