//! Custom Metal kernel safe wrapper — `mlx.fast.metal_kernel`.
//!
//! Mirrors `mlx-swift`'s `Source/MLX/MLXFastKernel.swift` (the
//! `MLXFast.MLXFastKernel` container + `MLXFast.metalKernel` factory) and the
//! python `mlx.fast.metal_kernel` callable. The compiled `MetalKernel`
//! handle is built once via `MetalKernel::new`; each invocation supplies an
//! [`MetalKernelApplyConfig`] describing the per-call grid, thread-group,
//! output shapes/dtypes, optional template arguments, optional init-value,
//! and verbosity flag.
//!
//! ## Foundation slice (this commit)
//!
//! [`KernelTemplateArg`] (enum over `Bool` / `Int` / `Dtype`) and
//! [`MetalKernelApplyConfig`] (per-call config builder) are pure-data Rust
//! types — no FFI yet. The `MetalKernel` handle wrapper and the
//! `mlx_fast_metal_kernel_apply` call shape land in the follow-up commit.

use crate::dtype::Dtype;

/// Template argument for a custom Metal kernel — `bool`, `i32`, or [`Dtype`].
///
/// Mirrors `mlx-swift`'s `KernelTemplateArg` protocol (`Bool` / `Int` /
/// `DType` impls in `MLXFastKernel.swift`) and the python `mlx.fast.metal_kernel`
/// per-call template-args dict, surfaced here as a closed enum so the
/// dispatcher in `MetalKernel::apply` (next commit) is exhaustive at compile
/// time.
///
/// Template arguments are referenced by name from the kernel source (e.g.
/// `template <typename T, int N>` in MSL); the
/// [`MetalKernelApplyConfig::template`] vector pairs each `(name, value)` and
/// forwards into one of `mlx_fast_metal_kernel_config_add_template_arg_{dtype,int,bool}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelTemplateArg {
  /// Boolean template parameter — forwards to
  /// `mlx_fast_metal_kernel_config_add_template_arg_bool`.
  Bool(bool),
  /// Signed-32-bit-integer template parameter — forwards to
  /// `mlx_fast_metal_kernel_config_add_template_arg_int`.
  Int(i32),
  /// MLX dtype template parameter — forwards to
  /// `mlx_fast_metal_kernel_config_add_template_arg_dtype`.
  Dtype(Dtype),
}

/// Per-call configuration for `MetalKernel::apply` (next commit).
///
/// Mirrors the keyword arguments of `MLXFastKernel.callAsFunction`
/// (`grid`, `threadGroup`, `outputShapes`, `outputDTypes`, `template`,
/// `initValue`, `verbose`) and the python `mlx.fast.metal_kernel` per-call
/// kwargs. Each apply call freshly composes an
/// `mlx_fast_metal_kernel_config` from this Rust-side description, then frees
/// the C handle before returning — the config is not retained across calls.
///
/// `output_shapes.len()` must equal `output_dtypes.len()`; both must also
/// equal the number of `output_names` declared when the parent `MetalKernel`
/// was constructed. The apply path enforces these invariants.
///
/// `template` may be empty; an absent entry is equivalent to omitting the
/// kwarg in the swift / python APIs.
///
/// `init_value` is `Some(v)` to pre-fill every output element with `v` before
/// the kernel runs (mlx-c's `_set_init_value`); `None` skips that call,
/// matching the swift / python default.
#[derive(Debug, Clone)]
pub struct MetalKernelApplyConfig {
  /// Launch grid as `(grid_x, grid_y, grid_z)`. Forwarded to
  /// `mlx_fast_metal_kernel_config_set_grid`.
  pub grid: (i32, i32, i32),
  /// Thread-group size as `(x, y, z)`. Forwarded to
  /// `mlx_fast_metal_kernel_config_set_thread_group`.
  pub thread_group: (i32, i32, i32),
  /// One shape per output array, aligned with [`Self::output_dtypes`] and
  /// with the `output_names` declared at parent-kernel construction.
  pub output_shapes: Vec<Vec<i32>>,
  /// One dtype per output array, aligned with [`Self::output_shapes`].
  pub output_dtypes: Vec<Dtype>,
  /// Template arguments, name + value. Empty is allowed.
  pub template: Vec<(String, KernelTemplateArg)>,
  /// Optional pre-fill value for every output element (mlx-c's
  /// `_set_init_value`). `None` skips the call.
  pub init_value: Option<f32>,
  /// If `true`, mlx-c logs the generated kernel source via
  /// `_set_verbose(true)` on each launch.
  pub verbose: bool,
}

impl MetalKernelApplyConfig {
  /// Build a config with the required `grid`, `thread_group`, `output_shapes`,
  /// and `output_dtypes`; optional template / init-value / verbose flags
  /// default to empty / `None` / `false` and can be set on the returned value.
  ///
  /// The constructor is intentionally NOT a `Builder` (the four required
  /// arguments would all be `Option` and most call sites would set every
  /// one); use struct-update on the returned value to override optional
  /// fields:
  ///
  /// ```ignore
  /// MetalKernelApplyConfig {
  ///     template: vec![("ALPHA".to_string(), KernelTemplateArg::Int(2))],
  ///     init_value: Some(0.0),
  ///     verbose: true,
  ///     ..MetalKernelApplyConfig::new(
  ///         (8, 1, 1), (8, 1, 1),
  ///         vec![vec![8]], vec![Dtype::F32],
  ///     )
  /// }
  /// ```
  pub fn new(
    grid: (i32, i32, i32),
    thread_group: (i32, i32, i32),
    output_shapes: Vec<Vec<i32>>,
    output_dtypes: Vec<Dtype>,
  ) -> Self {
    Self {
      grid,
      thread_group,
      output_shapes,
      output_dtypes,
      template: Vec::new(),
      init_value: None,
      verbose: false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // ───────────────────────── KernelTemplateArg ─────────────────────────

  #[test]
  fn template_arg_bool_variant_roundtrip() {
    let a = KernelTemplateArg::Bool(true);
    let b = KernelTemplateArg::Bool(false);
    assert_eq!(a, KernelTemplateArg::Bool(true));
    assert_ne!(a, b);
  }

  #[test]
  fn template_arg_int_variant_roundtrip() {
    let a = KernelTemplateArg::Int(7);
    assert_eq!(a, KernelTemplateArg::Int(7));
    assert_ne!(a, KernelTemplateArg::Int(8));
    assert_ne!(a, KernelTemplateArg::Bool(true));
  }

  #[test]
  fn template_arg_dtype_variant_roundtrip() {
    let a = KernelTemplateArg::Dtype(Dtype::F32);
    assert_eq!(a, KernelTemplateArg::Dtype(Dtype::F32));
    assert_ne!(a, KernelTemplateArg::Dtype(Dtype::F16));
    assert_ne!(a, KernelTemplateArg::Int(0));
  }

  #[test]
  fn template_arg_is_copy_and_clone() {
    // The `Copy` bound matters: the apply path dispatch loop matches the
    // value by reference. A regression to `!Copy` (e.g. adding a `String`
    // arm) would force a clone-or-move rewrite at the call site — the test
    // pins the contract.
    fn assert_copy<T: Copy>() {}
    fn assert_clone<T: Clone>() {}
    assert_copy::<KernelTemplateArg>();
    assert_clone::<KernelTemplateArg>();
    let a = KernelTemplateArg::Int(3);
    let _b = a; // would move if !Copy
    let _c = a; // and again
  }

  // ───────────────────────── MetalKernelApplyConfig ─────────────────────────

  #[test]
  fn config_new_defaults_optional_fields() {
    let cfg = MetalKernelApplyConfig::new((8, 1, 1), (4, 1, 1), vec![vec![8]], vec![Dtype::F32]);
    assert_eq!(cfg.grid, (8, 1, 1));
    assert_eq!(cfg.thread_group, (4, 1, 1));
    assert_eq!(cfg.output_shapes, vec![vec![8]]);
    assert_eq!(cfg.output_dtypes, vec![Dtype::F32]);
    assert!(cfg.template.is_empty());
    assert!(cfg.init_value.is_none());
    assert!(!cfg.verbose);
  }

  #[test]
  fn config_struct_update_overrides_optional_fields() {
    let cfg = MetalKernelApplyConfig {
      template: vec![("ALPHA".to_string(), KernelTemplateArg::Int(2))],
      init_value: Some(0.5),
      verbose: true,
      ..MetalKernelApplyConfig::new((16, 1, 1), (8, 1, 1), vec![vec![16]], vec![Dtype::F16])
    };
    assert_eq!(cfg.grid, (16, 1, 1));
    assert_eq!(cfg.thread_group, (8, 1, 1));
    assert_eq!(cfg.template.len(), 1);
    assert_eq!(cfg.template[0].0, "ALPHA");
    assert_eq!(cfg.template[0].1, KernelTemplateArg::Int(2));
    assert_eq!(cfg.init_value, Some(0.5));
    assert!(cfg.verbose);
  }

  #[test]
  fn config_is_clone_for_repeated_dispatch() {
    // Apply paths that retry / fan out a config over multiple inputs
    // clone the config rather than rebuild it; pin the bound.
    fn assert_clone<T: Clone>() {}
    assert_clone::<MetalKernelApplyConfig>();
    let cfg = MetalKernelApplyConfig::new((1, 1, 1), (1, 1, 1), vec![vec![1]], vec![Dtype::F32]);
    let cloned = cfg.clone();
    assert_eq!(cloned.grid, cfg.grid);
    assert_eq!(cloned.output_shapes, cfg.output_shapes);
  }

  #[test]
  fn config_multi_output_shapes_and_dtypes_align() {
    let cfg = MetalKernelApplyConfig::new(
      (2, 2, 1),
      (1, 1, 1),
      vec![vec![4], vec![4, 4]],
      vec![Dtype::F32, Dtype::I32],
    );
    assert_eq!(cfg.output_shapes.len(), cfg.output_dtypes.len());
    assert_eq!(cfg.output_shapes[1], vec![4, 4]);
    assert_eq!(cfg.output_dtypes[1], Dtype::I32);
  }
}
