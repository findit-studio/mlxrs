//! VLM-side SIMD kernels.
//!
//! This module hosts the `aarch64` NEON + scalar dispatcher triples
//! for the VLM image-preprocessing candidates documented in
//! `docs/core-arch-simd-candidates.md` §4 (the "C" series, image arm).
//! Each candidate ships as its own dispatcher + `arch::neon` kernel +
//! scalar kernel triple (see [`crate::simd`] for the worked example
//! with `dot`), plus a differential test using the helpers in
//! [`crate::simd::diff`].
//!
//! # Shipped (per-candidate PR landing order)
//!
//! - **C6** ([#151](https://github.com/Findit-AI/mlxrs/issues/151)) —
//!   `pad_to_square` canvas fill (`vlm/image.rs`). Class: `Exact`. The
//!   `pad_canvas_fill` submodule holds the dispatcher, the scalar
//!   reference, the 48-byte LCM(3, 16) NEON kernel, and the verify-
//!   before-claim benchmark + decision (per §5.5 execution order —
//!   quick-win lead-off). The submodule is `#[doc(hidden)]` so the
//!   only public surface is the [`crate::vlm::image::pad_to_square`]
//!   call site that consumes it.
//! - **C4** ([#149](https://github.com/Findit-AI/mlxrs/issues/149)) —
//!   `image_to_array` BGR R↔B swap widen (`vlm/image.rs`). Class:
//!   `Exact`. The `bgr_widen` submodule holds the dispatcher, the
//!   scalar reference (`chunks_exact_mut(3) + MaybeUninit::write` —
//!   LLVM auto-vectorizes this shape on aarch64 once the destination
//!   is a sized slice rather than `Vec::push`), the hand-rolled NEON
//!   `vld3q_u8` + permuted `vst3q_f32` 16-pixel-tile kernel (R↔B swap
//!   encoded structurally by feeding the de-interleaved planes to the
//!   interleave-store in `(B, G, R)` order), and the verify-before-
//!   claim benchmark. The NEON kernel **ships unconditionally on
//!   aarch64** despite the bench showing the auto-vec scalar is
//!   ~13–15 % faster at 4096² on M-series silicon — this is a
//!   per-kernel override of the §5.4 2×-rule, per explicit user
//!   directive ("do not trust auto-vectorized, please impl the NEON
//!   backend"). Rationale (full text in the submodule's "Decision —
//!   RULE OVERRIDE" paragraph): auto-vectorization is compiler-
//!   version-dependent and can silently regress on a rustc / LLVM
//!   upgrade or a stylistic refactor; the SIMD module's contract is
//!   to provide a guaranteed arch-specific kernel that does not
//!   depend on auto-vec heuristics; other targets / sizes / surrounding
//!   call-site contexts may not auto-vectorize as cleanly as the
//!   benched M-series shape. The scalar arm remains as the
//!   differential-test oracle and the non-aarch64 fallback. The
//!   submodule is `#[doc(hidden)]` so the only public surface is the
//!   [`crate::vlm::image::image_to_array`] call site.
//!
//! # Planned candidates (issue numbers — umbrella
//! [`#143`](https://github.com/Findit-AI/mlxrs/issues/143))
//!
//! - **C3** ([#148](https://github.com/Findit-AI/mlxrs/issues/148)) —
//!   `image_to_array` u8 → f32 RGB widening (`vlm/image.rs`). Class:
//!   `Exact` (lossless integer-to-fp widen + constant scale; only land
//!   if the disassembly check shows LLVM is not already vectorizing).
//! - **C5** ([#150](https://github.com/Findit-AI/mlxrs/issues/150)) —
//!   `rotate_buf` pixel permutation (`vlm/image.rs`). Class: `Exact`
//!   (defer per §5.5; gather-bound).

#[doc(hidden)]
pub mod bgr_widen;
#[doc(hidden)]
pub mod pad_canvas_fill;

pub(crate) use bgr_widen::bgr_widen;
pub(crate) use pad_canvas_fill::pad_canvas_fill;
