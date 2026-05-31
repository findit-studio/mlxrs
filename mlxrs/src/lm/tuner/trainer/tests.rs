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
