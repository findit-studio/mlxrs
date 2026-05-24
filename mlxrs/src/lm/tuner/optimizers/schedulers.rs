//! Step-driven learning-rate schedules.
//!
//! Mirrors Python `mlx.optimizers.schedulers`
//! (`mlx/python/mlx/optimizers/schedulers.py`).
//!
//! Each schedule returns a `Box<dyn Fn(usize) -> f32>` matching the
//! [`super::LearningRate::Schedule`] variant — wrap the returned closure in
//! `LearningRate::Schedule(...)` to plug it into any [`super::Optimizer`].
//!
//! Schedules:
//!
//! - [`exponential_decay`] — `init · decay_rate^step`
//! - [`step_decay`] — staircase: `init · decay_rate^(step // step_size)`
//! - [`cosine_decay`] — half-cosine from `init` to `end` over
//!   `decay_steps` (constant `end` beyond)
//! - [`linear_schedule`] — straight line from `init` to `end` over
//!   `steps` (clamped at `end` beyond)
//! - [`join_schedules`] — piecewise composition by integer boundaries
//!
//! `Box<dyn Fn>` (not `impl Fn`) so the return type erases the closure's
//! capture set — enables runtime composition (e.g. building a vec of
//! schedules from a config file).

use crate::{Result, error::Error};

/// Schedule closure shape: `Fn(step) -> learning_rate`.
pub type Schedule = Box<dyn Fn(usize) -> f32>;

/// `lr(step) = init · decay_rate^step`. Mirrors Python `exponential_decay`
/// (`schedulers.py:9..=31`).
pub fn exponential_decay(init: f32, decay_rate: f32) -> Schedule {
  Box::new(move |step| init * decay_rate.powi(step as i32))
}

/// `lr(step) = init · decay_rate^(step // step_size)` — staircase decay.
/// Mirrors Python `step_decay` (`schedulers.py:34..=58`).
pub fn step_decay(init: f32, decay_rate: f32, step_size: usize) -> Result<Schedule> {
  if step_size == 0 {
    return Err(Error::Backend {
      message: "step_decay: step_size must be > 0".into(),
    });
  }
  Ok(Box::new(move |step| {
    init * decay_rate.powi((step / step_size) as i32)
  }))
}

/// Half-cosine decay from `init` to `end` over `decay_steps`; constant
/// `end` beyond. Mirrors Python `cosine_decay` (`schedulers.py:61..=88`).
///
/// ```text
/// s = min(step, decay_steps)
/// decay = 0.5 · (1 + cos(π · s / decay_steps))
/// lr = end + decay · (init - end)
/// ```
pub fn cosine_decay(init: f32, decay_steps: usize, end: f32) -> Result<Schedule> {
  if decay_steps == 0 {
    return Err(Error::Backend {
      message: "cosine_decay: decay_steps must be > 0".into(),
    });
  }
  let pi = std::f32::consts::PI;
  let decay_steps_f = decay_steps as f32;
  Ok(Box::new(move |step| {
    let s = (step as f32).min(decay_steps_f);
    let decay = 0.5 * (1.0 + (pi * s / decay_steps_f).cos());
    end + decay * (init - end)
  }))
}

/// Linear interpolation from `init` to `end` over `steps`; constant `end`
/// beyond. Mirrors Python `linear_schedule` (`schedulers.py:131..=158`).
pub fn linear_schedule(init: f32, end: f32, steps: usize) -> Result<Schedule> {
  if steps == 0 {
    return Err(Error::Backend {
      message: "linear_schedule: steps must be > 0".into(),
    });
  }
  let steps_f = steps as f32;
  let slope = (end - init) / steps_f;
  Ok(Box::new(move |step| {
    let s = (step as f32).min(steps_f);
    s * slope + init
  }))
}

/// Piecewise composition of schedules. Mirrors Python `join_schedules`
/// (`schedulers.py:91..=128`).
///
/// `schedules.len()` MUST equal `boundaries.len() + 1`. Up to boundary `b₀`
/// the first schedule is consulted with `step`; between boundaries `b₀` and
/// `b₁` the second schedule is consulted with `step - b₀`, etc.
pub fn join_schedules(schedules: Vec<Schedule>, boundaries: Vec<usize>) -> Result<Schedule> {
  if schedules.is_empty() {
    return Err(Error::Backend {
      message: "join_schedules: must provide at least one schedule".into(),
    });
  }
  if schedules.len() != boundaries.len() + 1 {
    return Err(Error::Backend {
      message: format!(
        "join_schedules: expected {} boundaries (schedules - 1), got {}",
        schedules.len() - 1,
        boundaries.len(),
      ),
    });
  }
  Ok(Box::new(move |step| {
    let mut output = schedules[0](step);
    for (i, &boundary) in boundaries.iter().enumerate() {
      if step >= boundary {
        output = schedules[i + 1](step - boundary);
      }
    }
    output
  }))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn exponential_decay_at_step_0_returns_init() {
    let s = exponential_decay(0.1, 0.9);
    assert!((s(0) - 0.1).abs() < 1e-6);
  }

  #[test]
  fn exponential_decay_at_step_5_matches_formula() {
    // 0.1 · 0.9^5 ≈ 0.059049
    let s = exponential_decay(0.1, 0.9);
    assert!((s(5) - 0.059_049).abs() < 1e-6, "got {}", s(5));
  }

  #[test]
  fn step_decay_holds_within_one_size_then_drops() -> Result<()> {
    let s = step_decay(0.1, 0.5, 10)?;
    assert!((s(0) - 0.1).abs() < 1e-6);
    assert!((s(9) - 0.1).abs() < 1e-6);
    assert!((s(10) - 0.05).abs() < 1e-6);
    assert!((s(19) - 0.05).abs() < 1e-6);
    assert!((s(20) - 0.025).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn step_decay_rejects_zero_step_size() {
    assert!(step_decay(0.1, 0.5, 0).is_err());
  }

  #[test]
  fn cosine_decay_at_t0_t_half_t_end_matches_formula() -> Result<()> {
    let s = cosine_decay(0.1, 1000, 0.0)?;
    // t=0: decay = 0.5·(1+cos(0)) = 1 → lr = 0 + 1·0.1 = 0.1
    assert!((s(0) - 0.1).abs() < 1e-6);
    // t=500 (half): decay = 0.5·(1+cos(π/2)) = 0.5 → lr = 0.05
    assert!((s(500) - 0.05).abs() < 1e-5);
    // t=1000 (end): decay = 0.5·(1+cos(π)) = 0 → lr = 0
    assert!((s(1000)).abs() < 1e-5);
    // beyond: constant at end (0)
    assert!((s(2000)).abs() < 1e-5);
    Ok(())
  }

  #[test]
  fn linear_schedule_at_endpoints_matches_formula() -> Result<()> {
    let s = linear_schedule(0.0, 0.1, 100)?;
    assert!((s(0) - 0.0).abs() < 1e-6);
    assert!((s(100) - 0.1).abs() < 1e-6);
    // Halfway:
    assert!((s(50) - 0.05).abs() < 1e-6);
    // Beyond: clamps at end.
    assert!((s(150) - 0.1).abs() < 1e-6);
    Ok(())
  }

  #[test]
  fn join_schedules_switches_at_boundary() -> Result<()> {
    let a = linear_schedule(0.0, 0.1, 10)?;
    let b = cosine_decay(0.1, 100, 0.0)?;
    let joined = join_schedules(vec![a, b], vec![10])?;
    // step 5 (before boundary): linear schedule at 0.05
    assert!((joined(5) - 0.05).abs() < 1e-6);
    // step 10 (at boundary): cosine schedule at (step-10)=0 → 0.1
    assert!((joined(10) - 0.1).abs() < 1e-6);
    // step 110 (full cosine): cosine schedule at (110-10)=100 →
    // decay=0.5·(1+cos(π))·0.1 = 0 (end=0)
    assert!((joined(110)).abs() < 1e-3);
    Ok(())
  }

  #[test]
  fn join_schedules_rejects_wrong_boundary_count() {
    let a = exponential_decay(0.1, 0.9);
    let b = exponential_decay(0.05, 0.9);
    let res = join_schedules(vec![a, b], vec![]);
    assert!(res.is_err());
  }
}
