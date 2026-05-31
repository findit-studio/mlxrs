//! Audio-side SIMD kernels.
//!
//! This module hosts the `aarch64` NEON + scalar dispatcher triples
//! for the audio CPU-DSP kernels. Each ships as its own dispatcher +
//! `arch::neon` kernel + scalar kernel triple (see [`crate::simd`] for
//! the worked example with `dot`), plus a differential test using the
//! helpers in [`crate::simd::diff`].
//!
//! # Ship policy
//!
//! Every NEON kernel ships **unconditionally on aarch64**.
//! Auto-vectorized scalar perf is compiler-version-dependent (a
//! rustc/LLVM upgrade or stylistic refactor can silently de-vectorize
//! a scalar arm without a test regression), but the hand-rolled NEON
//! kernel is an arch-guaranteed contract that does not depend on LLVM
//! heuristics. Bench numbers are report-only in each module's
//! doc-comment; they never drive ship decisions. The scalar arm
//! remains as the differential-test oracle and the non-aarch64
//! fallback.
//!
//! # Shipped kernels
//!
//! - `save_wav` f32 → i16 quantize ([#152](https://github.com/Findit-AI/mlxrs/issues/152),
//!   `audio/io.rs`). Class: `Exact` (pinned `round_half_to_even`
//!   semantics via NEON `vcvtnq_s32_f32`). Module: [`quantize`].
//! - PCM sample decode → normalized f32 widen ([#146](https://github.com/Findit-AI/mlxrs/issues/146),
//!   `audio/io.rs::push_samples`). Class: `Exact` (lossless
//!   integer-to-fp widen with a constant scale). Multi-dtype
//!   (i8/i16/i24/i32 + offset-binary u-variants). Module:
//!   [`pcm_decode`].
//! - Window generation `symmetric_window` / `build_kaldi_window` ([#157](https://github.com/Findit-AI/mlxrs/issues/157),
//!   `audio/dsp.rs`, `audio/features.rs`). Class: `Tolerance` (cosine
//!   evaluation via NEON polynomial approximation). Module: [`window`].
//! - `mel_filter_bank` triangle construction ([#155](https://github.com/Findit-AI/mlxrs/issues/155),
//!   `audio/dsp.rs`). Class: `Tolerance` (per-row triangular
//!   construction over `all_freqs` vector). Module: [`mel_triangle`].
//! - `get_mel_banks_kaldi` triangle construction ([#156](https://github.com/Findit-AI/mlxrs/issues/156),
//!   `audio/features.rs`). Class: `Tolerance` (per-row triangular
//!   construction with on-the-fly `mel_scale_kaldi`). Module:
//!   [`kaldi_mel`].
//! - `resample_linear` linear interpolation ([#153](https://github.com/Findit-AI/mlxrs/issues/153),
//!   `audio/io.rs`). Class: `Tolerance` (fp mul/add ordering across
//!   lanes; NEON FMA pattern `s1 + (s2-s1)*frac`). Module: [`resample`].
//!
//! # Covered elsewhere
//!
//! - loudness sum-of-squares — covered by the in-tree
//!   [`crate::simd::sum_of_squares`] kernel surface, see
//!   `simd::scalar::sum_of_squares` and `simd::arch::neon::sum_of_squares`.
//!
//! # `lfilter` — FIR fast-path only
//!
//! The `lfilter` recurrence is serial by construction (IIR). See
//! [`lfilter`] for the full write-up. Outcome:
//!
//! - The `state_len == 0` FIR fast-path (`y[n] = b0 * x[n]`) IS
//!   parallel and ships as a NEON `f64x2`-wide kernel. Cosmetic for
//!   the K-weighting workload (which never hits this arm), but a
//!   legit NEON kernel that the dispatcher routes through.
//! - The biquad specialization (`state_len == 2`, K-weighting's
//!   actual workload) is hand-unrolled, but the recurrence is purely
//!   serial so there is no within-stream NEON parallelism to exploit.
//!   Whether the hand-unrolled scalar arm or the
//!   `target_feature(enable = "neon")`-annotated arm beats the
//!   generic loop is benchmark-dependent; the
//!   `mlxrs/benches/simd_lfilter.rs` micro-bench is the authoritative
//!   data source. See [`lfilter`] module doc for the decision and
//!   numbers.

pub mod kaldi_mel;
pub mod lfilter;
pub mod mel_triangle;
pub mod pcm_decode;
pub mod quantize;
pub mod resample;
pub mod window;
