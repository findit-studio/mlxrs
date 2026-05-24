//! Audio-side SIMD kernels.
//!
//! This module hosts the `aarch64` NEON + scalar dispatcher triples
//! for the audio candidates documented in
//! `docs/core-arch-simd-candidates.md` §3 (the "C" series). Each
//! candidate ships as its own dispatcher + `arch::neon` kernel + scalar
//! kernel triple (see [`crate::simd`] for the worked example with
//! `dot`), plus a differential test using the helpers in
//! [`crate::simd::diff`].
//!
//! # Ship-decision rule (per user directive 2026-05-23)
//!
//! Every NEON kernel ships **unconditionally on aarch64** — the
//! §5.4 ≥2× drop-rule is REVERSED per the project memory rule
//! **"SIMD ship NEON regardless"**. Auto-vectorized scalar perf is
//! compiler-version-dependent (a rustc/LLVM upgrade or stylistic
//! refactor can silently de-vectorize a scalar arm without a test
//! regression), but the hand-rolled NEON kernel is an arch-guaranteed
//! contract that does not depend on LLVM heuristics. Bench numbers
//! are report-only in each module's doc-comment; they NEVER drive
//! ship decisions. The scalar arm remains as the differential-test
//! oracle and the non-aarch64 fallback.
//!
//! # Shipped candidates
//!
//! - **C7**  ([#152](https://github.com/Findit-AI/mlxrs/issues/152)) —
//!   `save_wav` f32 → i16 quantize (`audio/io.rs`). Class: `Exact`
//!   (pinned `round_half_to_even` semantics via NEON `vcvtnq_s32_f32`).
//!   Module: [`quantize`].
//! - **C1**  ([#146](https://github.com/Findit-AI/mlxrs/issues/146)) —
//!   PCM sample decode → normalized f32 widen
//!   (`audio/io.rs::push_samples`). Class: `Exact` (lossless
//!   integer-to-fp widen with a constant scale). Multi-dtype
//!   (i8/i16/i24/i32 + offset-binary u-variants). Module:
//!   [`pcm_decode`].
//! - **C12** ([#157](https://github.com/Findit-AI/mlxrs/issues/157)) —
//!   Window generation `symmetric_window` / `build_kaldi_window`
//!   (`audio/dsp.rs`, `audio/features.rs`). Class: `Tolerance` (cosine
//!   evaluation via NEON polynomial approximation). Module: [`window`].
//! - **C10** ([#155](https://github.com/Findit-AI/mlxrs/issues/155)) —
//!   `mel_filter_bank` triangle construction (`audio/dsp.rs`). Class:
//!   `Tolerance` (per-row triangular construction over `all_freqs`
//!   vector). Module: [`mel_triangle`].
//! - **C11** ([#156](https://github.com/Findit-AI/mlxrs/issues/156)) —
//!   `get_mel_banks_kaldi` triangle construction (`audio/features.rs`).
//!   Class: `Tolerance` (per-row triangular construction with on-the-fly
//!   `mel_scale_kaldi`). Module: [`kaldi_mel`].
//! - **C8**  ([#153](https://github.com/Findit-AI/mlxrs/issues/153)) —
//!   `resample_linear` linear interpolation (`audio/io.rs`). Class:
//!   `Tolerance` (fp mul/add ordering across lanes; NEON FMA pattern
//!   `s1 + (s2-s1)*frac`). Module: [`resample`].
//!
//! # Deferred / scoped out
//!
//! - **C2** loudness sum-of-squares — covered by the in-tree
//!   [`crate::simd::sum_of_squares`] kernel surface, see
//!   `simd::scalar::sum_of_squares` and `simd::arch::neon::sum_of_squares`.
//!
//! # C9 — empirical attempt, ships only the FIR fast-path
//!
//! **C9** `lfilter` recurrence — originally documented as
//! non-candidate (IIR is serial by construction). Per user directive
//! 2026-05-24, an empirical bench-driven attempt was made. See
//! [`lfilter`] for the full write-up. Outcome:
//!
//! - The `state_len == 0` FIR fast-path (`y[n] = b0 * x[n]`) IS
//!   parallel and ships as a NEON `f64x2`-wide kernel. Cosmetic for
//!   the K-weighting workload (which never hits this arm), but a
//!   legit NEON kernel that the dispatcher routes through.
//! - The biquad specialization (`state_len == 2`, K-weighting's
//!   actual workload) was tried via hand-unrolling — the recurrence
//!   is purely serial so there is no within-stream NEON parallelism
//!   to exploit. Whether the hand-unrolled scalar arm or the
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
