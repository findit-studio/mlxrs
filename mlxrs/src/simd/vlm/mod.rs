//! VLM-side SIMD kernels.
//!
//! This module hosts the `aarch64` NEON + scalar dispatcher triples
//! for the VLM image-preprocessing kernels. Each ships as its own
//! dispatcher + `arch::neon` kernel + scalar kernel triple (see
//! [`crate::simd`] for the worked example with `dot`), plus a
//! differential test using the helpers in [`crate::simd::diff`].
//!
//! # Shipped kernels
//!
//! - `pad_to_square` canvas fill ([#151](https://github.com/Findit-AI/mlxrs/issues/151),
//!   `vlm/image.rs`). Class: `Exact`. The `pad_canvas_fill` submodule
//!   holds the dispatcher, the scalar reference, and the 48-byte
//!   LCM(3, 16) NEON kernel. The submodule is `#[doc(hidden)]` so the
//!   only public surface is the [`crate::vlm::image::pad_to_square`]
//!   call site that consumes it.
//! - `image_to_array` BGR R↔B swap widen ([#149](https://github.com/Findit-AI/mlxrs/issues/149),
//!   `vlm/image.rs`). Class: `Exact`. The `bgr_widen` submodule holds
//!   the dispatcher, the scalar reference (`chunks_exact_mut(3) +
//!   MaybeUninit::write` — LLVM auto-vectorizes this shape on aarch64
//!   once the destination is a sized slice rather than `Vec::push`),
//!   and the hand-rolled NEON `vld3q_u8` + permuted `vst3q_f32`
//!   16-pixel-tile kernel (R↔B swap encoded structurally by feeding
//!   the de-interleaved planes to the interleave-store in `(B, G, R)`
//!   order). The NEON kernel **ships unconditionally on aarch64** even
//!   though the auto-vectorized scalar is ~13–15 % faster at 4096² on
//!   M-series silicon: auto-vectorization is compiler-version-dependent
//!   and can silently regress on a rustc / LLVM upgrade or a stylistic
//!   refactor, whereas the hand-rolled kernel is a guaranteed
//!   arch-specific contract that does not depend on auto-vec
//!   heuristics; other targets / sizes / surrounding call-site contexts
//!   may not auto-vectorize as cleanly as the benched M-series shape.
//!   The scalar arm remains as the differential-test oracle and the
//!   non-aarch64 fallback. The submodule is `#[doc(hidden)]` so the
//!   only public surface is the [`crate::vlm::image::image_to_array`]
//!   call site.
//!
//! # Additional shipped kernels
//!
//! - `image_to_array` u8 → f32 RGB widening ([#148](https://github.com/Findit-AI/mlxrs/issues/148),
//!   `vlm/image.rs`). Class: `Exact`. The `rgb_widen` submodule holds
//!   the dispatcher, the scalar reference, and the 16-byte tile NEON
//!   kernel, which ships unconditionally on aarch64.
//! - `rotate_buf` pixel permutation ([#150](https://github.com/Findit-AI/mlxrs/issues/150),
//!   `vlm/image.rs`). Class: `Exact`. The `rotate_buf` submodule
//!   specialises the **u8 + channels=4** (Rgba8) hot path with a
//!   4-pixel-tile `vld1q_u8` + per-pixel u32 scattered store; all other
//!   type / channel combinations fall back to the scalar arm. Per-pixel
//!   destination scatter is gather-bound (NEON has no scatter), so the
//!   SIMD win is bounded by the per-tile load width. Ships
//!   unconditionally on aarch64.

#[doc(hidden)]
pub mod bgr_widen;
#[doc(hidden)]
pub mod pad_canvas_fill;
#[doc(hidden)]
pub mod rgb_widen;
#[doc(hidden)]
pub mod rotate_buf;

pub(crate) use bgr_widen::bgr_widen;
pub(crate) use pad_canvas_fill::pad_canvas_fill;
pub(crate) use rgb_widen::rgb_widen;
// `rotate_buf::rotate_buf_u8` is re-exported via the public submodule
// (no `pub(crate) use` here yet — the caller wiring lands separately,
// and re-exporting an unused symbol triggers the workspace `-D
// warnings` gate).
