//! Custom Metal kernel safe wrapper — `mlx.fast.metal_kernel`.
//!
//! Mirrors `mlx-swift`'s `Source/MLX/MLXFastKernel.swift` (the
//! `MLXFast.MLXFastKernel` container + `MLXFast.metalKernel` factory) and the
//! python `mlx.fast.metal_kernel` callable. The compiled [`MetalKernel`]
//! handle is built once via [`MetalKernel::new`]; each invocation supplies an
//! [`MetalKernelApplyConfig`] describing the per-call grid, thread-group,
//! output shapes/dtypes, optional template arguments, optional init-value,
//! and verbosity flag.
//!
//! Custom Metal kernels require a real Metal device at apply time;
//! construction itself does not. The integration tests in
//! `mlxrs/tests/ops_fast_metal_kernel.rs` cover the apply path behind a
//! `#[cfg(target_os = "macos")] #[ignore]` gate so headless CI does not
//! attempt to launch a Metal pipeline. Unit tests in this file cover the
//! pure-Rust pieces (template-arg variants, config defaults / struct-update,
//! constructor input validation including interior-NUL rejection).

use std::ffi::CString;

use derive_more::{IsVariant, TryUnwrap, Unwrap};

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    EmptyInputPayload, Error, InteriorNulPayload, LengthMismatchPayload, OutOfRangePayload, Result,
    check, check_vector_array_handle,
  },
  stream::default_stream,
};

/// Template argument for a custom Metal kernel — `bool`, `i32`, or [`Dtype`].
///
/// Mirrors `mlx-swift`'s `KernelTemplateArg` protocol (`Bool` / `Int` /
/// `DType` impls in `MLXFastKernel.swift`) and the python `mlx.fast.metal_kernel`
/// per-call template-args dict, surfaced here as a closed enum so the
/// dispatcher in [`MetalKernel::apply`] is exhaustive at compile time.
///
/// Template arguments are referenced by name from the kernel source (e.g.
/// `template <typename T, int N>` in MSL); the
/// [`MetalKernelApplyConfig::template_slice`] vector pairs each `(name, value)` and
/// forwards into one of `mlx_fast_metal_kernel_config_add_template_arg_{dtype,int,bool}`.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Unwrap, TryUnwrap)]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
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

/// Per-call configuration for [`MetalKernel::apply`].
///
/// Mirrors the keyword arguments of `MLXFastKernel.callAsFunction`
/// (`grid`, `threadGroup`, `outputShapes`, `outputDTypes`, `template`,
/// `initValue`, `verbose`) and the python `mlx.fast.metal_kernel` per-call
/// kwargs. Each apply call freshly composes an
/// `mlx_fast_metal_kernel_config` from this Rust-side description, then frees
/// the C handle before returning — the config is not retained across calls.
///
/// `output_shapes.len()` must equal `output_dtypes.len()`; both must also
/// equal the number of `output_names` declared when the parent
/// [`MetalKernel`] was constructed. [`MetalKernel::apply`] enforces these
/// invariants and returns [`Error::ShapeMismatch`] on violation rather than
/// passing through to mlx-c (where the failure surfaces only at JIT time
/// with a less actionable message).
///
/// The optional `template`, `init_value`, and `verbose` fields default to
/// empty / `None` / `false` and can be set via the builder methods
/// [`Self::with_template`], [`Self::with_init_value`], and
/// [`Self::with_verbose`].
///
/// `init_value` is `Some(v)` to pre-fill every output element with `v` before
/// the kernel runs (mlx-c's `_set_init_value`); `None` skips that call,
/// matching the swift / python default.
#[derive(Debug, Clone)]
pub struct MetalKernelApplyConfig {
  /// Launch grid as `[grid_x, grid_y, grid_z]`. Forwarded to
  /// `mlx_fast_metal_kernel_config_set_grid`.
  grid: [u32; 3],
  /// Thread-group size as `[x, y, z]`. Forwarded to
  /// `mlx_fast_metal_kernel_config_set_thread_group`.
  thread_group: [u32; 3],
  /// One shape per output array, aligned with `output_dtypes` and
  /// with the `output_names` declared at parent-kernel construction.
  output_shapes: Vec<Vec<i32>>,
  /// One dtype per output array, aligned with `output_shapes`.
  output_dtypes: Vec<Dtype>,
  /// Template arguments, name + value. Empty is allowed.
  template: Vec<(String, KernelTemplateArg)>,
  /// Optional pre-fill value for every output element (mlx-c's
  /// `_set_init_value`). `None` skips the call.
  init_value: Option<f32>,
  /// If `true`, mlx-c logs the generated kernel source via
  /// `_set_verbose(true)` on each launch.
  verbose: bool,
}

impl MetalKernelApplyConfig {
  /// Build a config with the required `grid`, `thread_group`, `output_shapes`,
  /// and `output_dtypes`; optional fields default to empty / `None` / `false`.
  ///
  /// Use the builder methods [`Self::with_template`], [`Self::with_init_value`],
  /// and [`Self::with_verbose`] to set optional fields:
  ///
  /// ```ignore
  /// MetalKernelApplyConfig::new(
  ///     [8, 1, 1], [8, 1, 1],
  ///     vec![vec![8]], vec![Dtype::F32],
  /// )
  /// .with_template(vec![("ALPHA".to_string(), KernelTemplateArg::Int(2))])
  /// .with_init_value(0.0)
  /// .with_verbose(true)
  /// ```
  pub fn new(
    grid: [u32; 3],
    thread_group: [u32; 3],
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

  /// Set the template arguments for this config.
  #[must_use]
  pub fn with_template(mut self, template: Vec<(String, KernelTemplateArg)>) -> Self {
    self.template = template;
    self
  }

  /// Set the optional pre-fill init value for every output element.
  #[must_use]
  pub fn with_init_value(mut self, value: f32) -> Self {
    self.init_value = Some(value);
    self
  }

  /// Set the verbosity flag (mlx-c logs the generated kernel source when `true`).
  #[must_use]
  pub fn with_verbose(mut self, v: bool) -> Self {
    self.verbose = v;
    self
  }

  /// Launch grid `[grid_x, grid_y, grid_z]`.
  #[inline(always)]
  pub fn grid(&self) -> [u32; 3] {
    self.grid
  }

  /// Thread-group size `[x, y, z]`.
  #[inline(always)]
  pub fn thread_group(&self) -> [u32; 3] {
    self.thread_group
  }

  /// Output shapes — one `Vec<i32>` per declared output.
  #[inline(always)]
  pub fn output_shapes_slice(&self) -> &[Vec<i32>] {
    &self.output_shapes
  }

  /// Output dtypes — one [`Dtype`] per declared output.
  #[inline(always)]
  pub fn output_dtypes_slice(&self) -> &[Dtype] {
    &self.output_dtypes
  }

  /// Template argument pairs `(name, value)`.
  #[inline(always)]
  pub fn template_slice(&self) -> &[(String, KernelTemplateArg)] {
    &self.template
  }

  /// Optional pre-fill init value (`None` means skip the mlx-c `_set_init_value` call).
  #[inline(always)]
  pub fn init_value(&self) -> Option<f32> {
    self.init_value
  }

  /// Verbosity flag.
  #[inline(always)]
  pub fn verbose(&self) -> bool {
    self.verbose
  }
}

/// RAII guard for a temporary `mlx_vector_string`. Frees the underlying
/// `std::vector<std::string>` on drop, including the sentinel NULL-ctx
/// returned by a failed `_new` allocation (mlx-c `_free` is a defined no-op
/// on NULL ctx).
struct VectorStringGuard(mlxrs_sys::mlx_vector_string);
impl Drop for VectorStringGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a
    // defined no-op on NULL ctx so the post-failed-`_new` sentinel is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call
    // `check()`, panic, or unwind across `extern "C"`; the rc is discarded
    // silently per the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_string_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_vector_array`. Same Drop contract as
/// [`VectorStringGuard`] — NULL-ctx-safe, never touches TLS, never panics.
struct VectorArrayGuard(mlxrs_sys::mlx_vector_array);
impl Drop for VectorArrayGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a
    // defined no-op on NULL ctx (sentinel-handle pattern). Runs during
    // `Drop` / thread teardown: discard rc silently.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_array_free(self.0);
    }
  }
}

/// RAII guard for a per-call `mlx_fast_metal_kernel_config`. The config is
/// freshly constructed and freed within [`MetalKernel::apply`]; the guard
/// keeps it alive across the (fallible) `_add_*` / `_set_*` / `_apply` chain
/// so an early `?` does not leak it.
struct MetalKernelConfigGuard(mlxrs_sys::mlx_fast_metal_kernel_config);
impl Drop for MetalKernelConfigGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. The C wrapper
    // `mlx_fast_metal_kernel_config_free` deletes the underlying
    // `mlx_fast_metal_kernel_config_cpp_*` (no-op on NULL ctx — sentinel
    // pattern). Drop contract: no TLS, no panic, no unwind across FFI.
    unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_config_free(self.0);
    }
  }
}

/// Build an `mlx_vector_string` from a slice of `&str` for a kernel-side
/// argument list (input or output names). Interior NULs raise a backend-style
/// error rather than panicking across the FFI boundary. Mirrors the pattern
/// in `crate::io::save_gguf` (the `GgufMetadata::StringList` arm).
fn build_vector_string(items: &[&str], context: &'static str) -> Result<VectorStringGuard> {
  // SAFETY: `mlx_vector_string_new()` returns a fresh empty vector handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`);
  // wrapped in `VectorStringGuard` BEFORE the fallible appends so an early
  // `?` frees it exactly once.
  let vstr = unsafe { mlxrs_sys::mlx_vector_string_new() };
  let guard = VectorStringGuard(vstr);
  for s in items {
    let cs = CString::new(*s).map_err(|_| {
      let _ = s;
      Error::InteriorNul(InteriorNulPayload::new(
        "ops::fast::metal_kernel::vector_string entry append",
        context,
      ))
    })?;
    // SAFETY: `vstr` is the valid vector owned by `guard`; `cs` is a valid
    // in-scope NUL-terminated C string. mlx-c `push_back`s a `std::string`
    // copy, retaining no pointer past the call; rc surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_string_append_value(vstr, cs.as_ptr()) })?;
  }
  Ok(guard)
}

/// Convert a `&str` into a NUL-terminated `CString`, mapping interior NULs to
/// a backend-style error.
fn cstring_or_err(s: &str, context: &'static str) -> Result<CString> {
  CString::new(s).map_err(|_| {
    let _ = s;
    Error::InteriorNul(InteriorNulPayload::new(
      "ops::fast::metal_kernel::cstring_or_err",
      context,
    ))
  })
}

/// Checked-conversion of a `[u32; 3]` dispatch dimension to `[i32; 3]` for
/// the mlx-c `set_grid` / `set_thread_group` FFI. Any component above
/// `i32::MAX` returns [`Error::Backend`] before the call — without this gate
/// the `as i32` cast would wrap to a negative value, which the Metal backend
/// would build a corrupt `MTL::Size(gx, gy, gz)` from. `context` is `"grid"`
/// or `"thread_group"` so the error message identifies which dimension
/// overflowed.
fn to_dispatch_dim(dim: [u32; 3], context: &'static str) -> Result<[i32; 3]> {
  let mut out = [0_i32; 3];
  for (axis, &v) in dim.iter().enumerate() {
    out[axis] = i32::try_from(v).map_err(|_| {
      Error::OutOfRange(OutOfRangePayload::new(
        context,
        "must fit in i32 (mlx-c set_grid / set_thread_group requires i32; reduce the dispatch dimension)",
        format_smolstr!("{context}[{axis}]={v}"),
      ))
    })?;
  }
  Ok(out)
}

/// Compiled custom Metal kernel ready for repeated invocation via
/// [`MetalKernel::apply`].
///
/// Mirrors `mlx-swift`'s `MLXFast.MLXFastKernel` and the python
/// `mlx.fast.metal_kernel` callable. The kernel is constructed once via
/// [`MetalKernel::new`] (mlx-c JIT-compiles + caches the Metal pipeline keyed
/// on `name`); each [`MetalKernel::apply`] launch reuses it.
///
/// ## Threading
///
/// `MetalKernel` is intentionally `!Send` and `!Sync` because the underlying
/// `mlx_fast_metal_kernel` is not concurrency-safe — the per-kernel
/// `CustomKernelFunction` and the kernel cache it indexes into live behind
/// the same thread-local mlx state as [`crate::Array`] and
/// [`crate::Stream`]. The raw pointer in the wrapped handle is enough to
/// make the auto-traits absent; no extra marker is needed.
pub struct MetalKernel {
  inner: mlxrs_sys::mlx_fast_metal_kernel,
  output_names: Vec<String>,
}

impl std::fmt::Debug for MetalKernel {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // Skip the raw FFI handle (a `*mut c_void` whose value is meaningless
    // to a debugger user and would print as a uselessly-unstable hex). The
    // declared output_names are the user-meaningful identifying info.
    f.debug_struct("MetalKernel")
      .field("output_names", &self.output_names)
      .finish_non_exhaustive()
  }
}

impl Drop for MetalKernel {
  fn drop(&mut self) {
    // SAFETY: frees a handle this struct owns exactly once. `_free` is a
    // defined no-op on NULL ctx, so a sentinel handle from a failed `_new()`
    // (caught in [`MetalKernel::new`] before being returned) is safe even on
    // the unreachable path. Drop contract: no TLS, no `check()`, no panic,
    // no unwind across `extern "C"`.
    unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_free(self.inner);
    }
  }
}

impl MetalKernel {
  /// Compile a new custom Metal kernel from MSL source.
  ///
  /// # Arguments
  ///
  /// - `name`: identifier (used by mlx-c as the cache key for the JIT-compiled
  ///   Metal pipeline and in error messages).
  /// - `input_names`: parameter names for the input arrays as they appear in
  ///   the kernel signature mlx-c generates.
  /// - `output_names`: parameter names for the output arrays as they appear in
  ///   the kernel signature mlx-c generates. The wrapper records this list so
  ///   [`MetalKernel::apply`] can validate that the per-call output-shape /
  ///   output-dtype counts line up.
  /// - `source`: the body of the Metal Shading Language kernel function (mlx-c
  ///   wraps it with the auto-generated function signature).
  /// - `header`: optional MSL header content prepended to the generated
  ///   source (helpful for shared helper functions / includes). Pass `""` to
  ///   skip — mlx-c accepts an empty header.
  /// - `ensure_row_contiguous`: if `true`, mlx ensures input arrays are
  ///   row-contiguous before the launch (at a copy-on-mismatch perf cost).
  /// - `atomic_outputs`: if `true`, outputs are declared `device atomic<T>`
  ///   in the generated signature for concurrent-write kernels.
  ///
  /// # Errors
  ///
  /// Returns [`Error::Backend`] if any of the four string arguments contains
  /// an interior NUL byte (rejected before reaching mlx-c so the failure
  /// surfaces as a recoverable [`Error`] rather than as an aborting C++
  /// exception). Returns [`Error::Backend`] if mlx-c's
  /// `mlx_fast_metal_kernel_new` fails — typically a JIT-compile error on the
  /// user-supplied source.
  pub fn new(
    name: &str,
    input_names: &[&str],
    output_names: &[&str],
    source: &str,
    header: &str,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
  ) -> Result<Self> {
    // Install the error handler before any fallible FFI calls so a default
    // printf+exit handler cannot fire on the first failure (mirrors the
    // `ops::shape::concatenate` template).
    crate::error::ensure_handler_installed();

    let name_c = cstring_or_err(name, "`name`")?;
    let source_c = cstring_or_err(source, "`source`")?;
    let header_c = cstring_or_err(header, "`header`")?;

    let input_names_guard = build_vector_string(input_names, "input_names")?;
    let output_names_guard = build_vector_string(output_names, "output_names")?;

    // SAFETY: `name_c` / `source_c` / `header_c` are valid in-scope
    // NUL-terminated C strings; `input_names_guard.0` and
    // `output_names_guard.0` are valid populated vector handles whose
    // guards keep them alive across this call. mlx-c copies the strings +
    // both vectors into its own `std::string` / `std::vector<std::string>`
    // storage (the C wrapper invokes `mlx::core::fast::metal_kernel(...)`
    // which constructs a fresh `CustomKernelFunction` from a `std::string`
    // owned by the cache), retaining none of the Rust-side pointers past
    // the call. On failure mlx-c returns a sentinel `{nullptr}` handle and
    // reports the error via the installed handler; we recover that via
    // `LAST` below.
    let raw = unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_new(
        name_c.as_ptr(),
        input_names_guard.0,
        output_names_guard.0,
        source_c.as_ptr(),
        header_c.as_ptr(),
        ensure_row_contiguous,
        atomic_outputs,
      )
    };
    // Drop the temporary vector_string + CStrings only after the FFI call so
    // mlx-c's read borrow is still live. Explicit drops document intent.
    drop(input_names_guard);
    drop(output_names_guard);
    drop(name_c);
    drop(source_c);
    drop(header_c);

    if raw.ctx.is_null() {
      // Sentinel-handle pattern: mlx-c reported the failure via the error
      // handler and returned `{nullptr}`. Drain `LAST` into `Err` so the
      // backend message survives.
      return Err(
        crate::error::LAST
          .with(|c| c.borrow_mut().take())
          .unwrap_or(Error::Backend(
            "mlx_fast_metal_kernel_new returned NULL handle".into(),
          )),
      );
    }

    Ok(Self {
      inner: raw,
      output_names: output_names.iter().map(|s| (*s).to_string()).collect(),
    })
  }

  /// Number of output arrays this kernel produces, matching the
  /// `output_names` slice passed to [`MetalKernel::new`].
  #[inline(always)]
  pub fn output_arity(&self) -> usize {
    self.output_names.len()
  }

  /// Output parameter names (the slice passed to [`MetalKernel::new`]).
  #[inline(always)]
  pub fn output_names_slice(&self) -> &[String] {
    &self.output_names
  }

  /// Launch the kernel with `inputs` and the per-call `config`. Returns the
  /// output arrays in the same order as `output_names` declared at
  /// construction.
  ///
  /// # Errors
  ///
  /// - [`Error::ShapeMismatch`] if `config.output_shapes.len()` /
  ///   `config.output_dtypes.len()` disagrees with the declared
  ///   `output_names` count, or if those two `Vec`s disagree with each
  ///   other.
  /// - [`Error::ShapeMismatch`] if any entry in `config.output_shapes`
  ///   contains a negative dimension (rejected before mlx-c via
  ///   [`crate::shape::validate_dims`]).
  /// - [`Error::Backend`] if a template-arg name contains an interior NUL
  ///   byte (rejected before mlx-c).
  /// - [`Error::Backend`] if any mlx-c `_set_*` / `_add_*` / `_apply` call
  ///   reports an error (e.g. a runtime Metal pipeline failure).
  pub fn apply(&self, inputs: &[&Array], config: &MetalKernelApplyConfig) -> Result<Vec<Array>> {
    // ensure_handler_installed() is deferred until AFTER all pure-Rust
    // validation (arity, output_shapes, dispatch-dim overflow) so a wrapper
    // error never touches mlx-c — its slow path calls `mlx_set_error_handler`
    // FFI on the stripped-ctor / fallback route.

    let expected = self.output_names.len();
    if config.output_shapes_slice().len() != expected {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "metal_kernel::apply: output_shapes vs kernel output_names",
        expected,
        config.output_shapes_slice().len(),
      )));
    }
    if config.output_dtypes_slice().len() != expected {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "metal_kernel::apply: output_dtypes vs kernel output_names",
        expected,
        config.output_dtypes_slice().len(),
      )));
    }

    // Validate every output shape BEFORE any FFI allocation so a negative dim
    // (which would silently sign-extend as a `usize` inside the C++ vector
    // constructor and corrupt allocation/copy bookkeeping) is rejected at the
    // safe boundary. Mirrors the `validate_dims` precedent in `ops::random`.
    // Also rejects empty (rank-0 / scalar) output shapes: mlx-c custom Metal
    // kernels require a ranked output_arg (`mlx::core::Shape` with `size > 0`)
    // — surfacing the rejection here gives a wrapper-context error instead of
    // a JIT-time backend message, and prevents the empty-slice case from
    // ever reaching the FFI (where it would otherwise rely on `dim_ptr`'s
    // sentinel for defined-pointer semantics).
    for shape in config.output_shapes_slice().iter() {
      if shape.is_empty() {
        return Err(Error::EmptyInput(EmptyInputPayload::new(
          "metal_kernel::apply: output_shapes[idx] (custom Metal kernel outputs must have rank >= 1)",
        )));
      }
      crate::shape::validate_dims(shape)?;
    }

    // Pre-FFI dispatch-dimension overflow check.
    //
    // mlx-c's `set_grid` / `set_thread_group` take `i32` per dimension; the
    // Rust surface accepts `[u32; 3]` so callers cannot pass a negative
    // dimension. Convert via `i32::try_from` HERE — before any mlx-c
    // allocation or setter call — so a u32 value above `i32::MAX` cannot
    // wrap into a negative dimension and reach the Metal backend's
    // `MTL::Size(gx, gy, gz)` construction with corrupt data, and so a
    // grid-overflow doesn't get masked by an earlier `_set_outputs` FFI
    // error reaching `?` first.
    let grid_i32 = to_dispatch_dim(config.grid(), "grid")?;
    let thread_group_i32 = to_dispatch_dim(config.thread_group(), "thread_group")?;

    // Pure-Rust validation done; from here on we may touch mlx-c.
    crate::error::ensure_handler_installed();

    // Resolve the stream FIRST so its cleared-thread poison guard fires
    // before any allocation — matches `ops::quantized::quantize` / `svd`.
    let stream = default_stream();

    // SAFETY: `_config_new` returns a sentinel `{nullptr}` ctx on allocation
    // failure (a defined-safe `_free` input); wrap in the RAII guard BEFORE
    // the fallible `_add_*` / `_set_*` chain so an early `?` frees it. The
    // sentinel itself is checked below.
    let config_raw = unsafe { mlxrs_sys::mlx_fast_metal_kernel_config_new() };
    let _config_guard = MetalKernelConfigGuard(config_raw);
    if config_raw.ctx.is_null() {
      return Err(
        crate::error::LAST
          .with(|c| c.borrow_mut().take())
          .unwrap_or(Error::Backend(
            "mlx_fast_metal_kernel_config_new returned NULL handle".into(),
          )),
      );
    }

    // Output arg slots — one (shape, dtype) per declared output_name. Dim
    // validation already ran above; the per-call `dim_ptr` sentinel keeps an
    // empty `Vec<i32>` from passing a singular dangling pointer into mlx-c's
    // `std::vector<int>` range constructor.
    for (shape, dtype) in config
      .output_shapes_slice()
      .iter()
      .zip(config.output_dtypes_slice().iter())
    {
      // SAFETY: `config_raw` is the valid handle owned by `_config_guard`;
      // `crate::shape::dim_ptr(shape)` is either `shape.as_ptr()` (a live
      // read-only buffer of `shape.len()` `c_int`s, with `shape: &Vec<i32>`
      // borrowed for the full loop iteration) or, for the empty-slice case,
      // a pointer to a static `c_int` sentinel — never a singular dangling
      // pointer. mlx-c copies into a `mlx::core::Shape` (`std::vector<int>`),
      // retaining no pointer past the call; rc via `check()`. The `dtype`
      // enum-to-raw conversion is a const map (`Dtype: Copy`).
      check(unsafe {
        mlxrs_sys::mlx_fast_metal_kernel_config_add_output_arg(
          config_raw,
          crate::shape::dim_ptr(shape),
          shape.len(),
          (*dtype).into(),
        )
      })?;
    }

    // Grid + thread-group (always required by the C config). Dimensions
    // already validated above as `grid_i32` / `thread_group_i32`; the
    // FFI receives the bounds-checked i32 values directly.
    // SAFETY: `config_raw` is the valid handle owned by `_config_guard`;
    // pure-value arguments, no pointer lifetimes; rc via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_config_set_grid(
        config_raw,
        grid_i32[0],
        grid_i32[1],
        grid_i32[2],
      )
    })?;
    // SAFETY: as above; pure-value args, rc via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_config_set_thread_group(
        config_raw,
        thread_group_i32[0],
        thread_group_i32[1],
        thread_group_i32[2],
      )
    })?;

    // Optional init-value (mlx-c skips the pre-fill when `_set_init_value`
    // is never called; we match that contract by only forwarding `Some`).
    if let Some(v) = config.init_value() {
      // SAFETY: as above; pure-value arg, rc via `check()`.
      check(unsafe { mlxrs_sys::mlx_fast_metal_kernel_config_set_init_value(config_raw, v) })?;
    }

    // Always set verbose explicitly (mlx-c defaults to `false`; passing
    // `false` is a no-op apart from honoring an explicit caller flag).
    // SAFETY: as above; pure-value arg, rc via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_config_set_verbose(config_raw, config.verbose())
    })?;

    // Template arguments — dispatch on the Rust enum to one of three mlx-c
    // typed `_add_template_arg_*` calls. Each arg name needs a transient
    // CString that must outlive its call.
    for (arg_name, arg_value) in config.template_slice() {
      let name_c = cstring_or_err(arg_name.as_str(), "template-arg name")?;
      match arg_value {
        KernelTemplateArg::Bool(v) => {
          // SAFETY: `config_raw` is the valid handle owned by `_config_guard`;
          // `name_c.as_ptr()` is a valid in-scope NUL-terminated C string
          // (live through the call). mlx-c copies the name into a
          // `std::string` and stores the bool, retaining nothing past the
          // call; rc via `check()`.
          check(unsafe {
            mlxrs_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
              config_raw,
              name_c.as_ptr(),
              *v,
            )
          })?;
        }
        KernelTemplateArg::Int(v) => {
          // SAFETY: as the Bool arm — borrowed in-scope C string + pure-value
          // payload; rc via `check()`.
          check(unsafe {
            mlxrs_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
              config_raw,
              name_c.as_ptr(),
              *v,
            )
          })?;
        }
        KernelTemplateArg::Dtype(v) => {
          // SAFETY: as the Bool arm — borrowed in-scope C string + the
          // `Dtype: Copy` enum-to-raw const map; rc via `check()`.
          check(unsafe {
            mlxrs_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
              config_raw,
              name_c.as_ptr(),
              (*v).into(),
            )
          })?;
        }
      }
      // `name_c` drops here, AFTER the FFI call — mlx-c retained nothing.
    }

    // Build the input `mlx_vector_array` from a contiguous Vec<mlx_array>
    // (mlx_array is Copy). Mirrors `ops::shape::concatenate`'s
    // CANONICAL VARIADIC-INPUT TEMPLATE.
    let raw_inputs: Vec<mlxrs_sys::mlx_array> = inputs.iter().map(|a| a.0).collect();
    // SAFETY: `raw_inputs` is a contiguous, live `Vec<mlx_array>`
    // (`mlx_array` is `Copy`); `(ptr, len)` is a valid pair. mlx-c copies
    // the handles into its own `std::vector` and does not retain the Rust
    // pointer. Zero-length inputs are allowed (some kernels read no input
    // arrays, e.g. an init-only generator) — `Vec::as_ptr()` on an empty
    // Vec returns a non-null dangling pointer that mlx-c never dereferences
    // when `len == 0`. The RAII guard frees the returned vector (NULL-ctx
    // safe).
    let inputs_vec =
      unsafe { mlxrs_sys::mlx_vector_array_new_data(raw_inputs.as_ptr(), raw_inputs.len()) };
    let _inputs_guard = VectorArrayGuard(inputs_vec);
    if inputs_vec.ctx.is_null() {
      return Err(
        crate::error::LAST
          .with(|c| c.borrow_mut().take())
          .unwrap_or(Error::Backend(
            "mlx_vector_array_new_data returned NULL handle".into(),
          )),
      );
    }

    // Allocate the output `mlx_vector_array` (out-param for `_apply`). The
    // sentinel-handle check uses the shared crate helper.
    // SAFETY: `mlx_vector_array_new()` returns a fresh empty out-param handle
    // (NULL ctx on allocation failure, a defined-safe input to `_free`);
    // wrapped in the RAII guard BEFORE the populating `_apply` call so any
    // early return frees it.
    let mut out_vec = unsafe { mlxrs_sys::mlx_vector_array_new() };
    check_vector_array_handle(out_vec)?;
    let _out_guard = VectorArrayGuard(out_vec);

    // SAFETY:
    // - `&mut out_vec` is the freshly allocated out-param handle (above) —
    //   mlx-c overwrites it in place with the populated vector.
    // - `self.inner` is the valid `mlx_fast_metal_kernel` handle this struct
    //   owns; mlx-c borrows it for the call.
    // - `inputs_vec` is the valid populated input vector owned by
    //   `_inputs_guard` and kept alive across this call.
    // - `config_raw` is the valid populated config owned by `_config_guard`
    //   and kept alive across this call.
    // - `stream` is the per-thread default GPU stream from `default_stream()`
    //   (which installed the error handler + checked the cleared-thread
    //   guard).
    // - mlx-c retains none of these handles past the call; the rc is
    //   surfaced via `check()`.
    check(unsafe {
      mlxrs_sys::mlx_fast_metal_kernel_apply(
        &mut out_vec,
        self.inner,
        inputs_vec,
        config_raw,
        stream,
      )
    })?;

    // Drain the populated output vector into `Vec<Array>` (mirrors
    // `ops::quantized::drain_vector`). The per-element `mlx_array_new`
    // out-params are wrapped in `Array` BEFORE the `_get` call so an early
    // return frees them.
    // SAFETY: pure read of a valid populated `mlx_vector_array`; mlx-c does
    // not mutate or retain it and returns a plain length.
    let n = unsafe { mlxrs_sys::mlx_vector_array_size(out_vec) };
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle
      // (NULL ctx) per the mlx-c convention; wrapped in `Array` FIRST so an
      // early return frees it, then populated by the following `_get`.
      let mut part = Array(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: `&mut part.0` is the fresh out-param; `out_vec` is the valid
      // populated vector and `i < n` is in range. mlx-c writes a fresh
      // `mlx_array` handle into `part.0` (copies the inner shared_ptr — the
      // vector still owns its own +1 reference). rc via `check()`.
      check(unsafe { mlxrs_sys::mlx_vector_array_get(&mut part.0, out_vec, i) })?;
      parts.push(part);
    }

    Ok(parts)
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
    let cfg = MetalKernelApplyConfig::new([8, 1, 1], [4, 1, 1], vec![vec![8]], vec![Dtype::F32]);
    assert_eq!(cfg.grid(), [8, 1, 1]);
    assert_eq!(cfg.thread_group(), [4, 1, 1]);
    assert_eq!(cfg.output_shapes_slice(), &[vec![8]]);
    assert_eq!(cfg.output_dtypes_slice(), &[Dtype::F32]);
    assert!(cfg.template_slice().is_empty());
    assert!(cfg.init_value().is_none());
    assert!(!cfg.verbose());
  }

  #[test]
  fn config_struct_update_overrides_optional_fields() {
    let cfg = MetalKernelApplyConfig::new([16, 1, 1], [8, 1, 1], vec![vec![16]], vec![Dtype::F16])
      .with_template(vec![("ALPHA".to_string(), KernelTemplateArg::Int(2))])
      .with_init_value(0.5)
      .with_verbose(true);
    assert_eq!(cfg.grid(), [16, 1, 1]);
    assert_eq!(cfg.thread_group(), [8, 1, 1]);
    assert_eq!(cfg.template_slice().len(), 1);
    assert_eq!(cfg.template_slice()[0].0, "ALPHA");
    assert_eq!(cfg.template_slice()[0].1, KernelTemplateArg::Int(2));
    assert_eq!(cfg.init_value(), Some(0.5));
    assert!(cfg.verbose());
  }

  #[test]
  fn config_is_clone_for_repeated_dispatch() {
    // Apply paths that retry / fan out a config over multiple inputs
    // clone the config rather than rebuild it; pin the bound.
    fn assert_clone<T: Clone>() {}
    assert_clone::<MetalKernelApplyConfig>();
    let cfg = MetalKernelApplyConfig::new([1, 1, 1], [1, 1, 1], vec![vec![1]], vec![Dtype::F32]);
    let cloned = cfg.clone();
    assert_eq!(cloned.grid(), cfg.grid());
    assert_eq!(cloned.output_shapes_slice(), cfg.output_shapes_slice());
  }

  #[test]
  fn config_multi_output_shapes_and_dtypes_align() {
    let cfg = MetalKernelApplyConfig::new(
      [2, 2, 1],
      [1, 1, 1],
      vec![vec![4], vec![4, 4]],
      vec![Dtype::F32, Dtype::I32],
    );
    assert_eq!(
      cfg.output_shapes_slice().len(),
      cfg.output_dtypes_slice().len()
    );
    assert_eq!(cfg.output_shapes_slice()[1], vec![4, 4]);
    assert_eq!(cfg.output_dtypes_slice()[1], Dtype::I32);
  }

  // ───────────────────────── MetalKernel::new (validation) ─────────────────────────
  //
  // These tests cover the wrapper-side input validation that fires BEFORE
  // the FFI call — interior-NUL rejection in `name` / `source` / `header` /
  // input-output-name slices. The mlx-c `_new` call itself needs a real
  // device only at apply time; construction does not, but to stay
  // headless-CI-safe we exercise validation only.

  fn assert_interior_nul(err: &Error, needle: &str) {
    match err {
      Error::InteriorNul(p) => {
        assert!(
          p.bytes_kind() == needle || p.bytes_kind().contains(needle.trim_matches('`')),
          "expected bytes_kind to match {needle:?}, got: {p:?}"
        );
      }
      other => panic!("expected Error::InteriorNul, got: {other:?}"),
    }
  }

  #[test]
  fn metal_kernel_new_rejects_interior_nul_in_name() {
    let err = MetalKernel::new("bad\0name", &["a"], &["out"], "// noop", "", true, false)
      .expect_err("interior NUL in name should be rejected");
    assert_interior_nul(&err, "`name`");
  }

  #[test]
  fn metal_kernel_new_rejects_interior_nul_in_source() {
    let err = MetalKernel::new("k", &["a"], &["out"], "// bad\0", "", true, false)
      .expect_err("interior NUL in source should be rejected");
    assert_interior_nul(&err, "`source`");
  }

  #[test]
  fn metal_kernel_new_rejects_interior_nul_in_header() {
    let err = MetalKernel::new("k", &["a"], &["out"], "// noop", "hdr\0bad", true, false)
      .expect_err("interior NUL in header should be rejected");
    assert_interior_nul(&err, "`header`");
  }

  #[test]
  fn metal_kernel_new_rejects_interior_nul_in_input_names() {
    let err = MetalKernel::new("k", &["a\0b"], &["out"], "// noop", "", true, false)
      .expect_err("interior NUL in input_names should be rejected");
    assert_interior_nul(&err, "input_names");
  }

  #[test]
  fn metal_kernel_new_rejects_interior_nul_in_output_names() {
    let err = MetalKernel::new("k", &["a"], &["out\0bad"], "// noop", "", true, false)
      .expect_err("interior NUL in output_names should be rejected");
    assert_interior_nul(&err, "output_names");
  }

  // ───────────────────────── MetalKernel::apply (output-shape validation) ─────────────────────────
  //
  // These tests cover the wrapper-side `output_shapes` validation that fires
  // BEFORE any FFI allocation (negative-dim rejection via
  // `crate::shape::validate_dims`, and the empty-slice → static-sentinel
  // routing via `crate::shape::dim_ptr`). They construct a real
  // `MetalKernel` to satisfy `apply`'s `&self` receiver — construction
  // succeeds without a Metal device — and then exercise `apply` only up to
  // the validation-or-route step so headless CI never reaches the Metal
  // pipeline. The real-device round-trip for a valid multi-dim shape lives
  // in `mlxrs/tests/ops_fast_metal_kernel.rs`.

  fn make_validation_kernel(output_names: &[&str]) -> MetalKernel {
    MetalKernel::new(
      "validation_only",
      &["x"],
      output_names,
      "uint elem = thread_position_in_grid.x; out[elem] = x[elem];",
      "",
      true,
      false,
    )
    .expect("construction should not need a Metal device")
  }

  #[test]
  fn apply_rejects_negative_output_dimension() {
    // A negative dim sign-extends as `usize` inside the C++ vector
    // constructor and would silently corrupt allocation bookkeeping.
    // `validate_dims` rejects before the FFI call.
    let kernel = make_validation_kernel(&["out"]);
    let input = Array::ones::<f32>(&(8usize,)).expect("ones alloc");
    let cfg =
      MetalKernelApplyConfig::new([8, 1, 1], [8, 1, 1], vec![vec![-1, 8]], vec![Dtype::F32]);
    let err = kernel
      .apply(&[&input], &cfg)
      .expect_err("negative output dim should be rejected before FFI");
    match err {
      Error::OutOfRange(payload) => {
        assert_eq!(payload.context(), "shape::validate_dims: dim");
        assert_eq!(payload.requirement(), "must be non-negative");
        // `validate_dims` formats the value as `dim[{i}]={d}` for the offending entry.
        assert_eq!(payload.value(), "dim[0]=-1");
      }
      other => panic!("expected OutOfRange, got: {other:?}"),
    }
  }

  #[test]
  fn apply_rejects_scalar_output_shape() {
    // An empty (rank-0) `Vec<i32>` would route through `dim_ptr`'s static
    // sentinel — defined-pointer-wise — but mlx-c custom kernels require a
    // ranked output_arg. The wrapper rejects up-front with a context
    // message rather than waiting for a JIT-time backend error.
    let kernel = make_validation_kernel(&["out"]);
    let input = Array::ones::<f32>(&(8usize,)).expect("ones alloc");
    let cfg = MetalKernelApplyConfig::new([1, 1, 1], [1, 1, 1], vec![vec![]], vec![Dtype::F32]);
    let err = kernel
      .apply(&[&input], &cfg)
      .expect_err("empty output shape should be rejected before FFI");
    match err {
      Error::EmptyInput(payload) => {
        assert_eq!(
          payload.context(),
          "metal_kernel::apply: output_shapes[idx] (custom Metal kernel outputs must have rank >= 1)"
        );
      }
      other => panic!("expected EmptyInput, got: {other:?}"),
    }
  }
}
