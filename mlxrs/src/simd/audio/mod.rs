//! Audio-side SIMD kernels (placeholder — populated per-candidate in
//! follow-up PRs).
//!
//! This module receives the `aarch64` NEON + scalar dispatcher triples
//! for the audio candidates documented in
//! `docs/core-arch-simd-candidates.md` §3 (the "C" series). Each
//! candidate ships as its own dispatcher + `arch::neon` kernel + scalar
//! kernel triple (see [`crate::simd`] for the worked example with
//! `dot`), plus a differential test using the helpers in
//! [`crate::simd::diff`].
//!
//! Planned candidates (issue numbers — umbrella
//! [`#143`](https://github.com/Findit-AI/mlxrs/issues/143)):
//!
//! - **C1**  ([#146](https://github.com/Findit-AI/mlxrs/issues/146)) —
//!   PCM sample decode → normalized f32 widen
//!   (`audio/io.rs::decode_buffer_into`). Class: `Exact` (lossless
//!   integer-to-fp widen with a constant scale).
//! - **C2**  ([#147](https://github.com/Findit-AI/mlxrs/issues/147)) —
//!   `integrated_loudness` per-block sum-of-squares
//!   (`audio/dsp.rs`). Class: `Tolerance` (fp-reduction;
//!   reuses the [`crate::simd::sum_of_squares`] kernel surface).
//! - **C7**  ([#152](https://github.com/Findit-AI/mlxrs/issues/152)) —
//!   `save_wav` f32 → i16 quantize (`audio/io.rs`). Class: `Exact` once
//!   the rounding mode is pinned (`round_half_to_even`).
//! - **C8**  ([#153](https://github.com/Findit-AI/mlxrs/issues/153)) —
//!   `resample_linear` linear interpolation (`audio/io.rs`). Class:
//!   `Tolerance` (fp mul/add ordering across lanes).
//! - **C10** ([#155](https://github.com/Findit-AI/mlxrs/issues/155)) —
//!   `mel_filter_bank` triangle construction (`audio/dsp.rs`). Class:
//!   `Tolerance` (cold table build; defer per §5.5).
//! - **C11** ([#156](https://github.com/Findit-AI/mlxrs/issues/156)) —
//!   `get_mel_banks_kaldi` triangle construction (`audio/features.rs`).
//!   Class: `Tolerance` (cold table build; defer per §5.5).
//! - **C12** ([#157](https://github.com/Findit-AI/mlxrs/issues/157)) —
//!   Window generation `symmetric_window` / `build_kaldi_window`
//!   (`audio/dsp.rs`, `audio/features.rs`). Class: `Tolerance` (cold
//!   one-time build; defer per §5.5).
//!
//! X5 (this skeleton) ships **no** kernels — only the module hook so
//! follow-up PRs can land each candidate without re-touching
//! [`crate::simd::mod`](crate::simd).
