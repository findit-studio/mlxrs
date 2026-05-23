//! VLM-side SIMD kernels (placeholder — populated per-candidate in
//! follow-up PRs).
//!
//! This module receives the `aarch64` NEON + scalar dispatcher triples
//! for the VLM image-preprocessing candidates documented in
//! `docs/core-arch-simd-candidates.md` §4 (the "C" series, image arm).
//! Each candidate ships as its own dispatcher + `arch::neon` kernel +
//! scalar kernel triple (see [`crate::simd`] for the worked example
//! with `dot`), plus a differential test using the helpers in
//! [`crate::simd::diff`].
//!
//! Planned candidates (issue numbers — umbrella
//! [`#143`](https://github.com/Findit-AI/mlxrs/issues/143)):
//!
//! - **C3** ([#148](https://github.com/Findit-AI/mlxrs/issues/148)) —
//!   `image_to_array` u8 → f32 RGB widening (`vlm/image.rs`). Class:
//!   `Exact` (lossless integer-to-fp widen + constant scale; only land
//!   if the disassembly check shows LLVM is not already vectorizing).
//! - **C4** ([#149](https://github.com/Findit-AI/mlxrs/issues/149)) —
//!   `image_to_array` BGR R↔B swap widen (`vlm/image.rs`). Class:
//!   `Exact` (`vld3` / `vst3` de-interleave widen).
//! - **C5** ([#150](https://github.com/Findit-AI/mlxrs/issues/150)) —
//!   `rotate_buf` pixel permutation (`vlm/image.rs`). Class: `Exact`
//!   (defer per §5.5; gather-bound).
//! - **C6** ([#151](https://github.com/Findit-AI/mlxrs/issues/151)) —
//!   `pad_to_square` canvas fill (`vlm/image.rs`). Class: `Exact`
//!   (per-3-byte-`extend` fix is the quick-win lead-off per §5.5).
//!
//! X5 (this skeleton) ships **no** kernels — only the module hook so
//! follow-up PRs can land each candidate without re-touching
//! [`crate::simd::mod`](crate::simd).
