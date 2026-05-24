//! C9 — `lfilter` IIR recurrence micro-bench.
//!
//! Bench-driven attempt (per user directive 2026-05-24) at SIMD-ifying
//! the previously-rejected C9 candidate. Three variants are compared
//! against the existing generic kernel:
//!
//! - **`generic`** — the existing
//!   `crate::audio::dsp::lfilter_f64_in_place` body (transcribed
//!   inline here to avoid the private-fn import). Tested as the
//!   baseline.
//! - **`biquad_scalar`** — the new
//!   `simd::audio::lfilter::lfilter_biquad_scalar` hand-unrolled
//!   `state_len == 2` specialization.
//! - **`biquad_dispatch`** — the dispatcher (routes to the
//!   `target_feature(enable = "neon")`-annotated arm on aarch64).
//! - **`fir_scalar` / `fir_dispatch`** — the FIR fast-path
//!   (`state_len == 0`, `y[n] = b0 * x[n]`). Cosmetic for K-weighting
//!   (which never hits this arm), but the real wide-SIMD kernel that
//!   does ship.
//!
//! # Sizes
//!
//! Lane sweep from 1024 → 65536 samples, plus the K-weighting fixture
//! (1 second @ 48 kHz = 48000 samples) and realistic long-channel
//! sizes (192000 = 4 s @ 48 kHz, 480000 = 10 s @ 48 kHz) which match
//! the actual `k_weight_channel` driver in `integrated_loudness`
//! (which receives FULL audio channels, not 48 k-sample fixtures —
//! up to `MAX_DECODED_SAMPLES = 64 Mi samples`).
//!
//! # In-place vs out-of-place
//!
//! Both `audio::dsp::lfilter_f64` (out-of-place, public) and
//! `audio::dsp::lfilter_f64_in_place` (in-place, K-weighting consumer)
//! were considered for biquad-dispatcher wiring. The out-of-place path
//! has an EXTRA `Vec::extend_from_slice(x)` memcpy because the kernel
//! is in-place; the per-arm decision needs to see numbers from BOTH
//! call shapes, not just the in-place one.
//!
//! # Unlike other simd_*.rs benches
//!
//! Other in-tree `simd_*.rs` benches are report-only (the project
//! memory rule "SIMD ship NEON regardless" mandates shipping NEON
//! regardless of bench numbers). C9 is the documented EXCEPTION: the
//! ship decision IS the bench. If the biquad NEON arm does not beat
//! the generic loop at the K-weighting workload size, the dispatcher
//! is wired into `audio::dsp::lfilter_f64_in_place` for the FIR
//! fast-path only (cosmetic), and the biquad specialization stays
//! as a documented "tried, didn't pan out" experiment.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mlxrs::simd::audio::lfilter::{
  lfilter_biquad, lfilter_biquad_scalar, lfilter_fir_b0, lfilter_fir_b0_scalar,
};

/// Deterministic f64 sample stream — alternating signs, slowly growing
/// magnitudes to keep the IIR state non-trivial.
fn gen_samples(n: usize) -> Vec<f64> {
  (0..n)
    .map(|i| {
      let mag = 0.1 + (i as f64) * 0.0007;
      if i.is_multiple_of(2) { mag } else { -mag }
    })
    .collect()
}

/// BS.1770 K-weighting high-shelf at 48 kHz — actual workload
/// coefficients (matches `crate::audio::dsp::bs1770_biquad_coefficients`).
fn k_weight_hs_coeffs_48k() -> ([f64; 3], [f64; 3]) {
  use core::f64::consts::PI;
  let gain_db = 4.0_f64;
  let q = 1.0 / 2.0_f64.sqrt();
  let fc = 1500.0_f64;
  let rate = 48000.0_f64;
  let amplitude = 10.0_f64.powf(gain_db / 40.0);
  let omega = 2.0 * PI * (fc / rate);
  let alpha = omega.sin() / (2.0 * q);
  let cos_omega = omega.cos();
  let sqrt_a = amplitude.sqrt();
  let b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
  let b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega);
  let b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
  let a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
  let a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega);
  let a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
  ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

/// BS.1770 K-weighting high-pass at 48 kHz.
fn k_weight_hp_coeffs_48k() -> ([f64; 3], [f64; 3]) {
  use core::f64::consts::PI;
  let q = 0.5_f64;
  let fc = 38.0_f64;
  let rate = 48000.0_f64;
  let omega = 2.0 * PI * (fc / rate);
  let alpha = omega.sin() / (2.0 * q);
  let cos_omega = omega.cos();
  let b0 = (1.0 + cos_omega) / 2.0;
  let b1 = -(1.0 + cos_omega);
  let b2 = (1.0 + cos_omega) / 2.0;
  let a0 = 1.0 + alpha;
  let a1 = -2.0 * cos_omega;
  let a2 = 1.0 - alpha;
  ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

/// Generic in-place kernel body — transcribed from
/// `crate::audio::dsp::lfilter_f64_in_place` (private — can't import).
/// Pre-normalized coefficients (caller divided by `a[0]`).
#[inline(never)]
fn generic_in_place(b: &[f64], a: &[f64], x: &mut [f64]) {
  let b0 = b[0];
  let state_len = a.len().max(b.len()) - 1;
  let mut state = vec![0.0_f64; state_len];
  for slot in x.iter_mut() {
    let sample = *slot;
    let output = b0 * sample + state[0];
    for i in 1..state_len {
      let feedforward = b.get(i).copied().unwrap_or(0.0) * sample;
      let feedback = a.get(i).copied().unwrap_or(0.0) * output;
      state[i - 1] = state[i] + feedforward - feedback;
    }
    let feedforward_last = b.get(state_len).copied().unwrap_or(0.0) * sample;
    let feedback_last = a.get(state_len).copied().unwrap_or(0.0) * output;
    state[state_len - 1] = feedforward_last - feedback_last;
    *slot = output;
  }
}

/// Generic OUT-OF-PLACE kernel body — transcribed from the pre-C9
/// `crate::audio::dsp::lfilter_f64` body (single-pass: reads `x[n]`,
/// writes `y[n]`, no intermediate full-buffer copy). This is the
/// pre-#154 baseline for `lfilter_f64`.
#[inline(never)]
fn generic_out_of_place(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
  let b0 = b[0];
  let n = x.len();
  let state_len = a.len().max(b.len()) - 1;
  let mut state = vec![0.0_f64; state_len];
  let mut y: Vec<f64> = Vec::with_capacity(n);
  for &sample in x {
    let output = b0 * sample + state[0];
    for i in 1..state_len {
      let feedforward = b.get(i).copied().unwrap_or(0.0) * sample;
      let feedback = a.get(i).copied().unwrap_or(0.0) * output;
      state[i - 1] = state[i] + feedforward - feedback;
    }
    let feedforward_last = b.get(state_len).copied().unwrap_or(0.0) * sample;
    let feedback_last = a.get(state_len).copied().unwrap_or(0.0) * output;
    state[state_len - 1] = feedforward_last - feedback_last;
    y.push(output);
  }
  y
}

/// OUT-OF-PLACE biquad-dispatch path — mirrors the post-#154
/// `lfilter_f64`'s shape: `Vec::with_capacity(n)` +
/// `extend_from_slice(x)` + in-place dispatcher kernel. Adds a full
/// `n_samples` memcpy beyond the in-place kernel's work.
#[inline(never)]
fn dispatch_out_of_place(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
  let n = x.len();
  let mut y: Vec<f64> = Vec::with_capacity(n);
  y.extend_from_slice(x);
  lfilter_biquad(&mut y, b, a);
  y
}

#[inline(never)]
fn run_biquad_scalar(x: &mut [f64], b: &[f64], a: &[f64]) {
  lfilter_biquad_scalar(x, b, a);
}

#[inline(never)]
fn run_biquad_dispatch(x: &mut [f64], b: &[f64], a: &[f64]) {
  lfilter_biquad(x, b, a);
}

#[inline(never)]
fn run_fir_scalar(out: &mut [f64], src: &[f64], b0: f64) {
  lfilter_fir_b0_scalar(out, src, b0);
}

#[inline(never)]
fn run_fir_dispatch(out: &mut [f64], src: &[f64], b0: f64) {
  lfilter_fir_b0(out, src, b0);
}

fn bench_biquad_lane_sweep(c: &mut Criterion) {
  let (b, a) = k_weight_hs_coeffs_48k();
  // Lane sweep 1024 → 65536 plus realistic long-channel sizes
  // (192000 = 4 s @ 48 kHz, 480000 = 10 s @ 48 kHz). `k_weight_channel`
  // receives FULL audio channels, not 48 k-sample fixtures.
  for &n in &[1024_usize, 4096, 16384, 48000, 65536, 192_000, 480_000] {
    let src = gen_samples(n);
    let mut group = c.benchmark_group(format!("lfilter_biquad/n={n}"));
    group.throughput(criterion::Throughput::Elements(n as u64));

    group.bench_function("generic", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          generic_in_place(black_box(&b), black_box(&a), x);
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });
    group.bench_function("biquad_scalar", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          run_biquad_scalar(x, black_box(&b), black_box(&a));
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });
    group.bench_function("biquad_dispatch", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          run_biquad_dispatch(x, black_box(&b), black_box(&a));
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });

    group.finish();
  }
}

fn bench_k_weight_chain(c: &mut Criterion) {
  // The actual K-weighting workload: chain HS → HP. Real-world driver
  // for `integrated_loudness`. Sweep multiple sizes including realistic
  // long channels: 1 s, 4 s, and 10 s @ 48 kHz. `k_weight_channel`
  // operates on FULL audio channels, so long-channel numbers matter.
  let (hs_b, hs_a) = k_weight_hs_coeffs_48k();
  let (hp_b, hp_a) = k_weight_hp_coeffs_48k();
  for &n in &[48000_usize, 192_000, 480_000] {
    let src = gen_samples(n);

    let mut group = c.benchmark_group(format!("lfilter_k_weight_chain/n={n}"));
    group.throughput(criterion::Throughput::Elements(n as u64));

    group.bench_function("generic", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          generic_in_place(black_box(&hs_b), black_box(&hs_a), x);
          generic_in_place(black_box(&hp_b), black_box(&hp_a), x);
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });
    group.bench_function("biquad_scalar", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          run_biquad_scalar(x, black_box(&hs_b), black_box(&hs_a));
          run_biquad_scalar(x, black_box(&hp_b), black_box(&hp_a));
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });
    group.bench_function("biquad_dispatch", |bench| {
      bench.iter_batched_ref(
        || src.clone(),
        |x| {
          run_biquad_dispatch(x, black_box(&hs_b), black_box(&hs_a));
          run_biquad_dispatch(x, black_box(&hp_b), black_box(&hp_a));
          black_box(&*x);
        },
        criterion::BatchSize::LargeInput,
      );
    });

    group.finish();
  }
}

fn bench_biquad_out_of_place_lane_sweep(c: &mut Criterion) {
  // Covers the `audio::dsp::lfilter_f64` (out-of-place, public) path:
  // generic single-pass (read `x`, write `y`) vs dispatch (
  // `extend_from_slice(x)` + in-place kernel — extra full-buffer memcpy).
  let (b, a) = k_weight_hs_coeffs_48k();
  for &n in &[1024_usize, 4096, 16384, 48000, 65536, 192_000, 480_000] {
    let src = gen_samples(n);
    let mut group = c.benchmark_group(format!("lfilter_biquad_out_of_place/n={n}"));
    group.throughput(criterion::Throughput::Elements(n as u64));

    group.bench_function("generic", |bench| {
      bench.iter(|| {
        let y = generic_out_of_place(black_box(&b), black_box(&a), black_box(&src));
        black_box(y);
      });
    });
    group.bench_function("dispatch", |bench| {
      bench.iter(|| {
        let y = dispatch_out_of_place(black_box(&b), black_box(&a), black_box(&src));
        black_box(y);
      });
    });

    group.finish();
  }
}

fn bench_fir_fast_path(c: &mut Criterion) {
  let b0 = 0.5_f64;
  for &n in &[1024_usize, 4096, 16384, 48000, 65536] {
    let src = gen_samples(n);
    let mut group = c.benchmark_group(format!("lfilter_fir_b0/n={n}"));
    group.throughput(criterion::Throughput::Elements(n as u64));

    group.bench_function("scalar", |bench| {
      let mut out = vec![0.0_f64; n];
      bench.iter(|| {
        run_fir_scalar(&mut out, black_box(&src), black_box(b0));
        black_box(&out);
      });
    });
    group.bench_function("dispatch", |bench| {
      let mut out = vec![0.0_f64; n];
      bench.iter(|| {
        run_fir_dispatch(&mut out, black_box(&src), black_box(b0));
        black_box(&out);
      });
    });

    group.finish();
  }
}

criterion_group!(
  benches,
  bench_biquad_lane_sweep,
  bench_biquad_out_of_place_lane_sweep,
  bench_k_weight_chain,
  bench_fir_fast_path,
);
criterion_main!(benches);
