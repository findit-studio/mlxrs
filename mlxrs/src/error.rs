//! Error model for the safe wrapper.
//!
//! Two failure-surfacing paths (verified against `mlx-c/mlx/c/array.cpp`):
//!   - rc pattern: most `int`-returning fns return 0 on success, non-zero on
//!     failure. Internal `check` helper drains the captured message.
//!   - sentinel-handle pattern: `mlx_array`-returning constructors return
//!     a handle with NULL `ctx` on failure. Internal `check_handle` does
//!     the same drain.
//!
//! In both cases the error message itself is delivered via the global
//! `mlx_set_error_handler` callback we install eagerly via `#[ctor::ctor(unsafe)]`.
//! That callback writes into a thread-local; check drains it.
//!
//! The handler MUST be installed before any fallible mlx-c call. The default
//! mlx-c handler is `printf + exit(-1)`, which would terminate the process
//! before our `rc` ever reaches `check()`. Every safe-layer entry point that
//! invokes mlx-c calls `ensure_handler_installed` first as defense-in-depth
//! against a stripped/disabled `#[ctor]`.

use std::{
  cell::RefCell,
  ffi::{CStr, c_char, c_int, c_void},
  panic::{AssertUnwindSafe, catch_unwind},
  path::PathBuf,
  ptr,
  sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
  },
};

use crate::Dtype;

/// The kind of file-system or I/O operation that failed, for [`Error::FileIo`].
///
/// Variants cover the operations observed across mlxrs: `Create`, `Write`,
/// `Flush`, `Read`, `Open`, `Stat`, `Copy`, `Remove`, `Rename`, `Fsync`,
/// `HardLink`, `Decode`. `Other` covers any operation not in the above list —
/// callers match only the variants they care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOp {
  /// `File::create` / `OpenOptions::create_new`.
  Create,
  /// Any write to a file or stream.
  Write,
  /// `flush()` on a buffered writer or stdout.
  Flush,
  /// `File::open` / `read_to_string` / `read_exact`.
  Read,
  /// `File::open` (open-only, no read implied).
  Open,
  /// `metadata()` / `symlink_metadata()` / `stat`.
  Stat,
  /// `fs::copy`.
  Copy,
  /// `fs::remove_file` / `fs::remove_dir_all`.
  Remove,
  /// `fs::rename` / `hard_link`.
  Rename,
  /// `File::sync_all` / `sync_data` / `fsync_dir`.
  Fsync,
  /// Any other named operation not in the list above.
  Other(&'static str),
}

/// Display helper for the `lengths` field of [`Error::MultiLengthMismatch`].
/// Formats a `Vec<(&'static str, usize)>` as `"axes=3, low=2, high=3"`.
///
/// Used directly in the `#[error("…")]` attribute via the
/// `thiserror` positional-arg form so the variant needs no extra
/// `impl Display for Error` override.
struct MultiLengthsFmt<'a>(&'a Vec<(&'static str, usize)>);

impl std::fmt::Display for MultiLengthsFmt<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut first = true;
    for (name, len) in self.0 {
      if !first {
        f.write_str(", ")?;
      }
      write!(f, "{name}={len}")?;
      first = false;
    }
    Ok(())
  }
}

impl std::fmt::Display for FileOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Create => f.write_str("create"),
      Self::Write => f.write_str("write"),
      Self::Flush => f.write_str("flush"),
      Self::Read => f.write_str("read"),
      Self::Open => f.write_str("open"),
      Self::Stat => f.write_str("stat"),
      Self::Copy => f.write_str("copy"),
      Self::Remove => f.write_str("remove"),
      Self::Rename => f.write_str("rename"),
      Self::Fsync => f.write_str("fsync"),
      Self::Other(s) => f.write_str(s),
    }
  }
}

/// Payload for [`Error::DtypeMismatch`]: the expected and actual dtypes.
///
/// Accessors are `const fn` for the `Copy` inner type. Construct via
/// [`DtypeMismatchPayload::new`]; destructure via the accessors (not struct
/// literal syntax — this type follows §2 of the rust-golden-skills conventions
/// which forbids public struct-style variant fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeMismatchPayload {
  expected: Dtype,
  got: Dtype,
}

impl DtypeMismatchPayload {
  /// Construct a new payload.
  pub fn new(expected: Dtype, got: Dtype) -> Self {
    Self { expected, got }
  }

  /// The dtype the caller asserted.
  #[inline(always)]
  pub const fn expected(&self) -> Dtype {
    self.expected
  }

  /// The actual dtype of the array.
  #[inline(always)]
  pub const fn got(&self) -> Dtype {
    self.got
  }
}

impl std::fmt::Display for DtypeMismatchPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "expected {:?}, got {:?}", self.expected, self.got)
  }
}

impl std::error::Error for DtypeMismatchPayload {}

/// Payload for [`Error::FfiNullHandle`]: the mlx-c function name that returned
/// a NULL handle.
///
/// Construct via [`FfiNullHandlePayload::new`]; access the field via
/// [`FfiNullHandlePayload::fn_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiNullHandlePayload {
  fn_name: &'static str,
}

impl FfiNullHandlePayload {
  /// Construct a new payload.
  pub fn new(fn_name: &'static str) -> Self {
    Self { fn_name }
  }

  /// The mlx-c function name that returned a NULL handle
  /// (e.g. `"mlx_array_new_float32"`).
  #[inline(always)]
  pub const fn fn_name(&self) -> &'static str {
    self.fn_name
  }
}

impl std::fmt::Display for FfiNullHandlePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "FFI: {} returned NULL handle", self.fn_name)
  }
}

impl std::error::Error for FfiNullHandlePayload {}

/// Payload for [`Error::MissingField`]: the parent type and missing field path.
///
/// Construct via [`MissingFieldPayload::new`]; access fields via
/// `type_name()` and `field()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingFieldPayload {
  type_name: &'static str,
  field: &'static str,
}

impl MissingFieldPayload {
  /// Construct a new payload.
  pub fn new(type_name: &'static str, field: &'static str) -> Self {
    Self { type_name, field }
  }

  /// The parent type or config context (e.g. `"SentencePieceTokenizer"`).
  #[inline(always)]
  pub const fn type_name(&self) -> &'static str {
    self.type_name
  }

  /// The missing field path (e.g. `"model.unk_id"`).
  #[inline(always)]
  pub const fn field(&self) -> &'static str {
    self.field
  }
}

impl std::fmt::Display for MissingFieldPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: missing required field `{}`",
      self.type_name, self.field
    )
  }
}

impl std::error::Error for MissingFieldPayload {}

/// Payload for [`Error::ArithmeticOverflow`]: the operation context and result type.
///
/// Construct via [`ArithmeticOverflowPayload::new`]; access fields via
/// `context()` and `op_type()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArithmeticOverflowPayload {
  context: &'static str,
  op_type: &'static str,
}

impl ArithmeticOverflowPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, op_type: &'static str) -> Self {
    Self { context, op_type }
  }

  /// The expression or operation that overflowed
  /// (e.g. `"vocab_size_base + added"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The result type that overflowed (e.g. `"u32"`, `"usize"`).
  #[inline(always)]
  pub const fn op_type(&self) -> &'static str {
    self.op_type
  }
}

impl std::fmt::Display for ArithmeticOverflowPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}: overflow ({})", self.context, self.op_type)
  }
}

impl std::error::Error for ArithmeticOverflowPayload {}

/// Payload for [`Error::EmptyInput`]: the call-site context label.
///
/// Construct via [`EmptyInputPayload::new`]; access the field via
/// [`EmptyInputPayload::context`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyInputPayload {
  context: &'static str,
}

impl EmptyInputPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str) -> Self {
    Self { context }
  }

  /// The parameter or collection that was empty
  /// (e.g. `"value_and_grad: argnums"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
}

impl std::fmt::Display for EmptyInputPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{} is empty (at least one element required)",
      self.context
    )
  }
}

impl std::error::Error for EmptyInputPayload {}

/// Payload for [`Error::InvariantViolation`]: the call-site context and the
/// violated requirement.
///
/// Construct via [`InvariantViolationPayload::new`]; access fields via
/// `context()` and `requirement()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvariantViolationPayload {
  context: &'static str,
  requirement: &'static str,
}

impl InvariantViolationPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, requirement: &'static str) -> Self {
    Self {
      context,
      requirement,
    }
  }

  /// The parameter or field that violated the invariant
  /// (e.g. `"train: steps_per_eval"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The violated constraint (e.g. `"must be >= 1"`, `"must be > 0"`).
  #[inline(always)]
  pub const fn requirement(&self) -> &'static str {
    self.requirement
  }
}

impl std::fmt::Display for InvariantViolationPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "backend: {} {}", self.context, self.requirement)
  }
}

impl std::error::Error for InvariantViolationPayload {}

/// Payload for [`Error::RankMismatch`]: the call-site context, observed rank,
/// and the full observed shape.
///
/// Construct via [`RankMismatchPayload::new`]; access fields via
/// `context()`, `actual()`, and `actual_shape()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankMismatchPayload {
  context: &'static str,
  actual: u32,
  actual_shape: Vec<usize>,
}

impl RankMismatchPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, actual: u32, actual_shape: Vec<usize>) -> Self {
    Self {
      context,
      actual,
      actual_shape,
    }
  }

  /// The call-site label + expected-rank description
  /// (e.g. `"token_embeddings must be rank-3 (batch, seq_len, hidden)"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The observed rank of the array.
  #[inline(always)]
  pub const fn actual(&self) -> u32 {
    self.actual
  }

  /// The full observed shape, for diagnostics (may be empty for rank-0 scalars).
  pub fn actual_shape(&self) -> &[usize] {
    &self.actual_shape
  }
}

impl std::fmt::Display for RankMismatchPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "shape mismatch: {}: got rank {} (shape {:?})",
      self.context, self.actual, self.actual_shape
    )
  }
}

impl std::error::Error for RankMismatchPayload {}

/// Payload for [`Error::LengthMismatch`]: the call-site context and the
/// expected vs observed lengths.
///
/// Construct via [`LengthMismatchPayload::new`]; access fields via
/// `context()`, `expected()`, and `actual()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatchPayload {
  context: &'static str,
  expected: usize,
  actual: usize,
}

impl LengthMismatchPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, expected: usize, actual: usize) -> Self {
    Self {
      context,
      expected,
      actual,
    }
  }

  /// The call-site label identifying what lengths are being compared
  /// (e.g. `"pad: axes vs low/high"`, `"gather: indices.len() vs axes.len()"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The required length.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// The observed length.
  #[inline(always)]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

impl std::fmt::Display for LengthMismatchPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "shape mismatch: {}: expected length {}, got {}",
      self.context, self.expected, self.actual
    )
  }
}

impl std::error::Error for LengthMismatchPayload {}

/// Payload for [`Error::OutOfRange`]: the call-site context, the violated
/// constraint, and the formatted runtime value.
///
/// Construct via [`OutOfRangePayload::new`]; access fields via `context()`,
/// `requirement()`, and `value()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfRangePayload {
  context: &'static str,
  requirement: &'static str,
  value: String,
}

impl OutOfRangePayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, requirement: &'static str, value: String) -> Self {
    Self {
      context,
      requirement,
      value,
    }
  }

  /// The parameter name and call-site context
  /// (e.g. `"top_k: parameter"`, `"max_audio_seconds"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The violated constraint as a static phrase
  /// (e.g. `"must be in (0, vocab_size)"`).
  #[inline(always)]
  pub const fn requirement(&self) -> &'static str {
    self.requirement
  }

  /// The formatted runtime value that violated the range.
  pub fn value(&self) -> &str {
    &self.value
  }
}

impl std::fmt::Display for OutOfRangePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "shape mismatch: {}: {}, got {}",
      self.context, self.requirement, self.value
    )
  }
}

impl std::error::Error for OutOfRangePayload {}

/// Payload for [`Error::FileIo`]: the call-site context, file-op kind, path,
/// and the underlying `std::io::Error`.
///
/// Construct via [`FileIoPayload::new`]; access fields via `context()`,
/// `op()`, `path()`, and `inner()`. The underlying io::Error is also
/// reachable via [`std::error::Error::source`].
#[derive(Debug)]
pub struct FileIoPayload {
  context: &'static str,
  op: FileOp,
  path: PathBuf,
  inner: std::io::Error,
}

impl FileIoPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, op: FileOp, path: PathBuf, inner: std::io::Error) -> Self {
    Self {
      context,
      op,
      path,
      inner,
    }
  }

  /// Call-site label identifying the function or subsystem
  /// (e.g. `"save_as_txt"`, `"load_audio"`, `"save_model"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The operation kind that failed.
  #[inline(always)]
  pub const fn op(&self) -> FileOp {
    self.op
  }

  /// The file or directory path involved.
  pub fn path(&self) -> &std::path::Path {
    &self.path
  }

  /// The underlying I/O error.
  pub fn inner(&self) -> &std::io::Error {
    &self.inner
  }
}

impl std::fmt::Display for FileIoPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "io: {}: {} {}: {}",
      self.context,
      self.op,
      self.path.display(),
      self.inner
    )
  }
}

impl std::error::Error for FileIoPayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(&self.inner)
  }
}

/// Payload for [`Error::MultiLengthMismatch`]: the call-site context and the
/// list of `(name, length)` pairs.
///
/// Construct via [`MultiLengthMismatchPayload::new`]; access fields via
/// `context()` and `lengths()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiLengthMismatchPayload {
  context: &'static str,
  lengths: Vec<(&'static str, usize)>,
}

impl MultiLengthMismatchPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, lengths: Vec<(&'static str, usize)>) -> Self {
    Self { context, lengths }
  }

  /// The call-site label and collections being compared
  /// (e.g. `"pad: axes/low/high"`, `"slice: start/stop/strides"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// The `(name, length)` pairs for each collection, in order.
  pub fn lengths(&self) -> &[(&'static str, usize)] {
    &self.lengths
  }
}

impl std::fmt::Display for MultiLengthMismatchPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "shape mismatch: {}: length mismatch — {}",
      self.context,
      MultiLengthsFmt(&self.lengths)
    )
  }
}

impl std::error::Error for MultiLengthMismatchPayload {}

/// Payload for [`Error::DurabilityWarning`]: whether the commit succeeded and the IO error.
///
/// Construct via [`DurabilityWarningPayload::new`]; access fields via the
/// `committed()` and `source()` accessors.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
#[derive(Debug)]
pub struct DurabilityWarningPayload {
  committed: bool,
  source: std::io::Error,
}

#[cfg(feature = "lm")]
impl DurabilityWarningPayload {
  /// Construct a new payload.
  pub fn new(committed: bool, source: std::io::Error) -> Self {
    Self { committed, source }
  }

  /// Whether the checkpoint commit point was reached before the fsync warning.
  /// Always `true` for instances produced by `save` / `save_model`.
  #[inline(always)]
  pub const fn committed(&self) -> bool {
    self.committed
  }

  /// The underlying `fsync_dir` IO error.
  pub fn source(&self) -> &std::io::Error {
    &self.source
  }

  /// Consume `self` and return the underlying IO error (owned).
  pub fn into_source(self) -> std::io::Error {
    self.source
  }
}

#[cfg(feature = "lm")]
impl std::fmt::Display for DurabilityWarningPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "save committed but durability fsync failed (committed={}): {}",
      self.committed, self.source
    )
  }
}

#[cfg(feature = "lm")]
impl std::error::Error for DurabilityWarningPayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(&self.source)
  }
}

/// Payload for [`Error::ConvertPostSavePartial`]: committed flag, optional save warning, and
/// the copy error.
///
/// Construct via [`ConvertPostSavePartialPayload::new`]; access fields via
/// `committed()`, `save_warning()`, and `copy_error()`.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
#[derive(Debug)]
pub struct ConvertPostSavePartialPayload {
  committed: bool,
  save_warning: Option<std::io::Error>,
  // `Box` breaks the recursive size cycle (Error → Payload → Error).
  // Carrying the crate `Error` directly preserves the typed structure
  // of the underlying failure — e.g. `Error::FileIo(FileIoPayload { .. })`
  // survives end-to-end with `op` and `path` intact for recovery code,
  // instead of being stringified into an opaque `io::Error::other(...)`
  // (R-final finding on PR #243).
  copy_error: Box<Error>,
}

#[cfg(feature = "lm")]
impl ConvertPostSavePartialPayload {
  /// Construct a new payload. `copy_error` is boxed to break the
  /// recursive size cycle; pass any `crate::Error` directly (typically
  /// the `Error::FileIo(..)` returned by `copy_tokenizer_and_extras`)
  /// to preserve its structured fields end-to-end.
  pub fn new(committed: bool, save_warning: Option<std::io::Error>, copy_error: Error) -> Self {
    Self {
      committed,
      save_warning,
      copy_error: Box::new(copy_error),
    }
  }

  /// Whether the checkpoint commit point was reached. Always `true` for
  /// instances produced by `convert`.
  #[inline(always)]
  pub const fn committed(&self) -> bool {
    self.committed
  }

  /// The durability-fsync warning from the save phase if the save returned
  /// [`Error::DurabilityWarning`]; `None` if the save returned plain `Ok(())`.
  pub fn save_warning(&self) -> Option<&std::io::Error> {
    self.save_warning.as_ref()
  }

  /// The tokenizer-extras copy failure (typed `crate::Error`, typically
  /// `Error::FileIo(FileIoPayload { .. })`).
  pub fn copy_error(&self) -> &Error {
    &self.copy_error
  }

  /// Consume `self` and return the constituent parts (owned).
  pub fn into_parts(self) -> (bool, Option<std::io::Error>, Error) {
    (self.committed, self.save_warning, *self.copy_error)
  }
}

#[cfg(feature = "lm")]
impl std::fmt::Display for ConvertPostSavePartialPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "convert: save committed but post-save extras copy partially failed (committed={}); \
       destination directory may be incomplete (missing tokenizer/extras files)",
      self.committed
    )
  }
}

#[cfg(feature = "lm")]
impl std::error::Error for ConvertPostSavePartialPayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(self.copy_error.as_ref() as &(dyn std::error::Error + 'static))
  }
}

/// Errors surfaced from the mlx backend or detected at the safe-wrapper boundary.
#[derive(Debug, thiserror::Error, derive_more::IsVariant)]
#[non_exhaustive]
pub enum Error {
  /// Shape mismatch detected by mlx during graph construction or eval.
  #[error("shape mismatch: {0}")]
  ShapeMismatch(String),

  /// Dtype mismatch (e.g. requesting `as_slice::<f32>` on an i32 array).
  #[error("dtype mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  DtypeMismatch(DtypeMismatchPayload),

  /// `TryFrom<mlxrs_sys::mlx_dtype>` failed — mlx returned a dtype we don't recognize.
  #[error("unknown dtype value from mlx: {0}")]
  UnknownDtype(u32),

  /// Out-of-memory during allocation (best-effort detection).
  #[error("out of memory")]
  OutOfMemory,

  /// `as_slice` or `to_vec` called on a non-contiguous (post-transpose,
  /// broadcast, or strided-slice) array. M2 will add `.contiguous()` to
  /// materialize a row-contiguous copy.
  #[error("array is not contiguous; M2 will add .contiguous() to materialize")]
  NonContiguous,

  /// Generic backend error with the message captured from mlx-c.
  #[error("mlx backend: {0}")]
  Backend(String),

  /// FFI handle creation returned a NULL pointer (mlx-c idiomatic
  /// "constructor failed"). The `fn_name` identifies which mlx-c
  /// function returned the NULL, so callers can route on the failure
  /// site without parsing strings.
  #[error(transparent)]
  FfiNullHandle(FfiNullHandlePayload),

  /// A required field is missing from a parsed config / state body.
  /// `type_name` is the parent type (e.g. `"SentencePieceTokenizer"`,
  /// `"VlmBaseConfig"`); `field` is the missing field path
  /// (e.g. `"model.unk_id"`, `"mock_image_size"`).
  #[error(transparent)]
  MissingField(MissingFieldPayload),

  /// An integer arithmetic operation overflowed. `context` identifies
  /// the operation site (e.g. `"vocab_size_base + added"`,
  /// `"CacheList::state_count"`); `op_type` describes the result type
  /// (e.g. `"u32"`, `"usize"`).
  #[error(transparent)]
  ArithmeticOverflow(ArithmeticOverflowPayload),

  /// A scalar / collection input was empty when at least one element
  /// was required. `context` identifies the call site
  /// (e.g. `"stack: arrays slice"`, `"value_and_grad: argnums"`).
  #[error(transparent)]
  EmptyInput(EmptyInputPayload),

  /// A scalar invariant was violated (e.g. "must be >= 1", "must be > 0").
  /// `context` identifies the call site (e.g. `"train: steps_per_eval"`,
  /// `"step_decay: step_size"`); `requirement` describes the violated
  /// constraint (e.g. `"must be >= 1"`, `"must be > 0"`).
  #[error(transparent)]
  InvariantViolation(InvariantViolationPayload),

  /// A tensor has the wrong rank. `context` gives the call site and the
  /// expected rank (e.g. `"token_embeddings must be rank-3 (batch, seq_len, hidden)"`);
  /// `actual` is the observed rank; `actual_shape` is the full observed shape
  /// for diagnostics. The expected rank is encoded in `context` so callers with
  /// non-integer requirements (e.g. ">= 2") can use a natural English phrase.
  #[error(transparent)]
  RankMismatch(RankMismatchPayload),

  /// Two collections / dimensions / counts have mismatched lengths.
  /// `context` identifies the call site and what was being compared
  /// (e.g. `"slice: start/stop/strides length"`);
  /// `expected` is the required length; `actual` is the observed length.
  #[error(transparent)]
  LengthMismatch(LengthMismatchPayload),

  /// A scalar value is outside its allowed range. `context` identifies the
  /// parameter and call site (e.g. `"top_k: parameter"`); `requirement`
  /// describes the violated bound as a static phrase
  /// (e.g. `"must be in (0, vocab_size)"`, `"must be a finite positive float"`);
  /// `value` is the formatted runtime value that violated the range.
  #[error(transparent)]
  OutOfRange(OutOfRangePayload),

  /// A file I/O operation failed. `context` identifies the call site
  /// (e.g. `"save_as_txt"`, `"load_audio"`, `"save_model"`); `op` is the
  /// [`FileOp`] kind (Create / Write / Flush / Read / …); `path` is the
  /// file or directory path involved; `inner` is the underlying
  /// [`std::io::Error`] (accessible via [`std::error::Error::source`]).
  ///
  /// Prefer this over `Backend(format!("ctx: op {} failed: {e}", path.display()))`
  /// at any call site where the path is owned or cheaply cloned (e.g. a
  /// `PathBuf` local, a `with_extension` result). The structured fields let
  /// callers branch on [`FileOp`] kind, inspect the path, and walk the
  /// source chain without string parsing.
  #[error(transparent)]
  FileIo(FileIoPayload),

  /// Multiple parallel collections / dimensions disagreed on their lengths
  /// when they must all agree. `context` identifies the call site and what
  /// is being compared (e.g. `"pad: axes/low/high"`, `"slice:
  /// start/stop/strides"`); `lengths` is a sequence of `(name, length)`
  /// pairs for each mismatched collection.
  ///
  /// Prefer this over `ShapeMismatch(format!("pad: length mismatch — axes={}, low={}, high={}", …))`
  /// at call sites that check three-or-more parallel slices (where
  /// [`Error::LengthMismatch`]'s single expected/actual pair is insufficient).
  #[error(transparent)]
  MultiLengthMismatch(MultiLengthMismatchPayload),

  /// Tokenizer subsystem error (HF tokenizer load/encode/decode, chat-template
  /// render, tool-call parse). Only constructed when the `tokenizer` feature
  /// is enabled. The message carries the underlying cause.
  #[cfg(feature = "tokenizer")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer")))]
  #[error("tokenizer: {0}")]
  Tokenizer(String),

  /// Defense-in-depth shard-path collision:
  /// [`crate::lm::load::save_model`]'s atomic no-replace
  /// `std::fs::hard_link` of a shard tempfile onto its final shard path
  /// failed with [`std::io::ErrorKind::AlreadyExists`], meaning a file
  /// already occupies that final path. `link(2)` is atomic + no-replace
  /// by spec, so this surfaces in a single syscall with no silent-
  /// replace window (a `rename`-based publish would race a concurrent
  /// writer here). The collision-resistant `gen_id` (timestamp µs,
  /// PID, per-process counter) makes this statistically unreachable in
  /// normal operation; surfacing it as a hard `Err` keeps the save
  /// fail-closed (never silently overwrite a foreign file). Constructed
  /// only when the `lm` feature is enabled.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  #[error("shard path collision: {}", .0.display())]
  ShardPathCollision(std::path::PathBuf),

  /// Post-commit durability warning: a checkpoint or config file was
  /// successfully renamed into place (so the new content IS visible on
  /// disk + would be observed by a subsequent
  /// [`crate::lm::load::load_weights`] / [`crate::lm::load::load_config`])
  /// but a follow-up `fsync` of the parent directory failed. The
  /// directory-rename entry may not yet be durable on disk: a power loss
  /// before the filesystem internally drains could leave the directory
  /// pointing at the OLD entry. The caller knows the save is **logically
  /// committed**.
  ///
  /// Returned by [`crate::lm::load::save`] when [`crate::lm::load::save_model`]
  /// or the post-commit config rename produced a
  /// [`crate::lm::load::CommitOutcome::CommittedWithDurabilityWarning`].
  /// Constructed only when the `lm` feature is enabled.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  // `transparent` delegates BOTH `Display` AND `source()` to the inner
  // payload. The payload's `std::error::Error::source()` returns the inner
  // `std::io::Error` directly (not the payload itself), preserving the
  // documented source-chain contract: the first `source()` hop is the
  // actionable io::Error, not an opaque wrapper.
  #[error(transparent)]
  DurabilityWarning(DurabilityWarningPayload),

  /// [`crate::lm::convert::convert`] reached the post-save extras-copy step
  /// AFTER the index rename succeeded (so the weights + shard index +
  /// config ARE visible on disk and a follow-up
  /// [`crate::lm::load::load_weights`] / [`crate::lm::load::load_config`]
  /// would observe the committed checkpoint) but
  /// [`crate::lm::convert::copy_tokenizer_and_extras`] partially failed —
  /// at least one tokenizer / `*.py` / `generation_config.json` file
  /// did NOT make it to the destination directory.
  ///
  /// **Semantically distinct from [`Error::DurabilityWarning`]**: a
  /// `DurabilityWarning` with `committed: true` means the on-disk
  /// checkpoint is **logically complete** (weights + index + config + the
  /// tokenizer-extras copy all landed; only the parent-directory `fsync`
  /// returned an error and so a power loss before the FS internally drains
  /// MAY revert the rename). A `ConvertPostSavePartial`, by contrast,
  /// means the on-disk destination directory is **structurally
  /// incomplete** — tokenizer files are missing — and a downstream
  /// [`crate::lm::load::load`] would either fail (missing tokenizer.json)
  /// or silently produce a checkpoint with the wrong tokenizer. Callers
  /// MUST decide whether to retry the copy, copy the missing files by
  /// hand, or treat the whole convert as failed and delete the
  /// destination.
  ///
  /// This variant is constructed only when the `lm` feature is enabled.
  /// It is machine-detectable via the payload accessors: the `save_warning`
  /// accessor disambiguates the save side (a durability fsync warning vs a
  /// clean save) and `copy_error` carries the original tokenizer-copy
  /// failure.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  // `transparent` delegates BOTH `Display` AND `source()` to the inner
  // payload. The payload's `std::error::Error::source()` returns
  // `copy_error` (the actionable tokenizer-copy failure) directly — NOT
  // the payload wrapper — so the documented source-chain contract is
  // preserved: walking `.source()` lands on the io::Error in one hop,
  // not two.
  #[error(transparent)]
  ConvertPostSavePartial(ConvertPostSavePartialPayload),

  /// [`crate::lm::convert::convert`] observed durability-fsync warnings
  /// at **two or more** fsync boundaries (the save-side parent-directory
  /// fsync, the post-copy per-file fsync, and/or the post-copy
  /// destination-directory fsync). Same "logically committed, durability
  /// uncertain" contract as [`Error::DurabilityWarning`], but the
  /// MULTI-warning shape carries each underlying [`std::io::Error`] in a
  /// separate `Option` field so the caller can machine-detect WHICH
  /// boundaries warned without a string parse.
  ///
  /// **Distinct from [`Error::DurabilityWarning`]** which carries
  /// EXACTLY ONE underlying [`std::io::Error`]; this variant is reserved
  /// for the strict-aggregate case (>= 2 boundaries warned in the same
  /// convert). The single-warning case continues to use
  /// [`Error::DurabilityWarning`] so existing callers' "exactly one
  /// fsync warning" recovery path is unchanged.
  ///
  /// **Distinct from [`Error::ConvertPostSavePartial`]** which carries a
  /// **hard copy failure** ([`std::fs::copy`] itself returned `Err` — the
  /// file did NOT reach disk). A `ConvertDurabilityWarnings` means EVERY
  /// file reached disk (every [`std::fs::copy`] returned `Ok`); only the
  /// fsync boundaries warned, and the destination is logically complete
  /// (a subsequent [`crate::lm::load::load`] would observe it on a running
  /// kernel — only a power loss before the FS internally drains could
  /// revert).
  ///
  /// The F7 R4 fix had folded multi-warning sources into a single
  /// free-form `std::io::Error::other(format!(...))` message inside
  /// the [`Error::DurabilityWarning`] `source` field, losing typed
  /// access to the individual [`std::io::Error`]s. This R5 fix routes
  /// the multi-warning
  /// case to this new variant so each warning is reachable via direct
  /// destructuring (and the first non-None warning is reachable via
  /// [`std::error::Error::source`] for chain walkers — see
  /// [`ConvertDurabilityWarnings`]).
  ///
  /// Constructed only when the `lm` feature is enabled.
  #[cfg(feature = "lm")]
  #[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
  #[error(transparent)]
  ConvertDurabilityWarnings(#[from] ConvertDurabilityWarnings),
}

/// Structured aggregate of `convert()`-time durability warnings —
/// the inner shape carried by [`Error::ConvertDurabilityWarnings`].
///
/// Each field is `Some(_)` iff the corresponding fsync boundary returned
/// `Err` AFTER its underlying write/copy succeeded; the data is on disk
/// either way, only durability across a power loss is uncertain.
///
/// **Machine-detectable**: callers destructure to learn WHICH boundaries
/// warned (no string parse). The
/// [`std::error::Error::source`] impl returns the first non-`None`
/// warning in deterministic `save -> post_copy_file -> post_copy_dir`
/// priority order so the chain walk reaches the most-actionable warning.
#[cfg(feature = "lm")]
#[cfg_attr(docsrs, doc(cfg(feature = "lm")))]
#[derive(Debug)]
pub struct ConvertDurabilityWarnings {
  /// Always `true` — this aggregate is reachable only after the
  /// observable commit point (the index rename) has succeeded. Kept
  /// in the public shape so a future caller can branch on it without
  /// an API break.
  pub(crate) committed: bool,
  /// Durability warning from the save phase. `Some(_)` iff
  /// [`crate::lm::load::save_model`] returned
  /// [`Error::DurabilityWarning`] (the save-side parent-directory
  /// fsync warned); `None` if the save returned plain `Ok(())`.
  pub(crate) save: Option<std::io::Error>,
  /// Durability warning from the post-copy per-file fsync. `Some(_)`
  /// iff at least one copied file's `fsync_path` returned `Err`
  /// AFTER its [`std::fs::copy`] succeeded (data IS on disk, only
  /// durability uncertain); `None` if every per-file fsync passed.
  pub(crate) post_copy_file: Option<std::io::Error>,
  /// Durability warning from the post-copy destination-directory
  /// fsync. `Some(_)` iff `fsync_dir(dst)` returned `Err` AFTER every
  /// [`std::fs::copy`] succeeded (data IS on disk, only the directory
  /// inode metadata's durability uncertain); `None` if the dir fsync
  /// passed.
  pub(crate) post_copy_dir: Option<std::io::Error>,
}

#[cfg(feature = "lm")]
impl std::fmt::Display for ConvertDurabilityWarnings {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "convert: save committed but post-save durability warnings (committed={}); \
       destination is on-disk and load-correct, but one or more fsync boundaries returned a warning",
      self.committed
    )
  }
}

#[cfg(feature = "lm")]
impl std::error::Error for ConvertDurabilityWarnings {
  /// Returns the FIRST non-`None` underlying [`std::io::Error`] in
  /// deterministic `save -> post_copy_file -> post_copy_dir` priority
  /// order — the most-actionable warning for a chain walker. Returns
  /// `None` only if every field is `None` (which the convert()
  /// call-site never constructs — the aggregate is reserved for the
  /// 2+-non-None-fields case — but is well-defined here so a caller
  /// constructing it directly observes a consistent contract).
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    self
      .first_warning()
      .map(|e| e as &(dyn std::error::Error + 'static))
  }
}

#[cfg(feature = "lm")]
impl ConvertDurabilityWarnings {
  /// Construct the aggregate directly. Always passes `committed = true`
  /// (this type is only reachable after the observable commit point).
  /// External callers use this instead of struct literal syntax.
  pub fn new(
    committed: bool,
    save: Option<std::io::Error>,
    post_copy_file: Option<std::io::Error>,
    post_copy_dir: Option<std::io::Error>,
  ) -> Self {
    Self {
      committed,
      save,
      post_copy_file,
      post_copy_dir,
    }
  }

  /// Whether the checkpoint commit point was reached. Always `true` for
  /// instances produced by [`crate::lm::convert::convert`].
  #[inline(always)]
  pub fn committed(&self) -> bool {
    self.committed
  }

  /// Durability warning from the save phase, if any.
  pub fn save(&self) -> Option<&std::io::Error> {
    self.save.as_ref()
  }

  /// Durability warning from the post-copy per-file fsync, if any.
  pub fn post_copy_file(&self) -> Option<&std::io::Error> {
    self.post_copy_file.as_ref()
  }

  /// Durability warning from the post-copy destination-directory fsync, if any.
  pub fn post_copy_dir(&self) -> Option<&std::io::Error> {
    self.post_copy_dir.as_ref()
  }

  /// Decompose into the four constituent fields (owned).
  ///
  /// `std::io::Error` is `!Clone`, so this is the only way for a downstream
  /// recovery path to move out the underlying typed errors (preserving
  /// `kind()`/`source()` fidelity) — the borrowed getters above are read-only.
  /// Used internally by the convert pipeline to route the single-warning case
  /// into [`Error::DurabilityWarning`]; surfaced publicly for the same need
  /// in external consumers.
  pub fn into_parts(
    self,
  ) -> (
    bool,
    Option<std::io::Error>,
    Option<std::io::Error>,
    Option<std::io::Error>,
  ) {
    (
      self.committed,
      self.save,
      self.post_copy_file,
      self.post_copy_dir,
    )
  }

  /// Return the first non-`None` underlying [`std::io::Error`] in
  /// deterministic `save -> post_copy_file -> post_copy_dir` priority
  /// order. Used by the [`std::error::Error::source`] impl; exposed
  /// publicly so callers that prefer the inherent accessor over a
  /// `dyn Error` chain walk get the same most-actionable warning
  /// without re-implementing the priority.
  pub fn first_warning(&self) -> Option<&std::io::Error> {
    self
      .save
      .as_ref()
      .or(self.post_copy_file.as_ref())
      .or(self.post_copy_dir.as_ref())
  }

  /// Count the number of non-`None` warning fields. The convert()
  /// call-site uses this to decide between `Ok(())` (0), the existing
  /// [`Error::DurabilityWarning`] (1), and [`Error::ConvertDurabilityWarnings`]
  /// (>= 2) so the multi-warning case is the only one routed through
  /// this aggregate.
  pub fn count(&self) -> usize {
    usize::from(self.save.is_some())
      + usize::from(self.post_copy_file.is_some())
      + usize::from(self.post_copy_dir.is_some())
  }
}

#[cfg(feature = "tokenizer")]
impl Error {
  /// Construct a [`Error::Tokenizer`] from anything stringifiable. Used
  /// throughout the `tokenizer` module to funnel HF / minijinja / serde
  /// failures into the crate's unified error type.
  pub(crate) fn tokenizer(message: impl Into<String>) -> Self {
    Self::Tokenizer(message.into())
  }
}

/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Allocate a `Vec<T>` reserving exactly `cap` capacity, returning
/// [`Error::OutOfMemory`] instead of aborting the process on allocator
/// failure (which `Vec::with_capacity` / `vec![…; n]` do). Use for any
/// REQUEST-SCALED allocation on a hot path (sequence length, token /
/// image counts) so an oversized or hostile input fails recoverably
/// rather than terminating the process.
///
/// For small fixed-size allocations (a handful of elements) the infallible
/// `Vec::with_capacity` remains fine — this is for input-proportional
/// buffers.
///
/// Consumed by the `lm`, `vlm`, `audio`, and `embeddings` modules for
/// request-scaled host-side buffers (the VLM-9 allocation-hardening pass,
/// now extended across lm/audio/embeddings). Gated to exactly the features
/// that use it so `cargo hack --each-feature` sees no dead code (`vlm` and
/// `audio` both enable `lm`).
#[cfg(any(feature = "lm", feature = "embeddings"))]
pub(crate) fn try_with_capacity<T>(cap: usize) -> Result<Vec<T>> {
  let mut v = Vec::new();
  v.try_reserve_exact(cap).map_err(|_| Error::OutOfMemory)?;
  Ok(v)
}

/// Fallible [`slice::to_vec`]: clone `slice` into a freshly-reserved
/// `Vec`, returning [`Error::OutOfMemory`] instead of aborting on
/// allocation failure. The recoverable analogue of `slice.to_vec()` for
/// request-scaled slices. (Only the `vlm` module needs the owned-clone
/// form; lm/audio/embeddings use `try_with_capacity` + `extend` or
/// `try_extend_from_slice` directly, hence the narrower gate.)
#[cfg(feature = "vlm")]
pub(crate) fn try_to_vec<T: Clone>(slice: &[T]) -> Result<Vec<T>> {
  let mut v = try_with_capacity(slice.len())?;
  v.extend_from_slice(slice);
  Ok(v)
}

/// Fallible [`Vec::extend_from_slice`]: reserve room for `slice` and append,
/// returning [`Error::OutOfMemory`] instead of aborting on allocation
/// failure. Uses the AMORTIZED `try_reserve` (NOT `try_reserve_exact`):
/// callers grow the same `Vec` repeatedly (processor history accumulates the
/// prefill prompt, then one token per decode step), so exact reservation
/// would reallocate on every append and turn an O(n) accumulation into
/// O(n²). The recoverable analogue of `vec.extend_from_slice(slice)`.
#[cfg(feature = "lm")]
pub(crate) fn try_extend_from_slice<T: Clone>(v: &mut Vec<T>, slice: &[T]) -> Result<()> {
  v.try_reserve(slice.len()).map_err(|_| Error::OutOfMemory)?;
  v.extend_from_slice(slice);
  Ok(())
}

thread_local! {
  pub(crate) static LAST: RefCell<Option<Error>> = const { RefCell::new(None) };
}

/// Take (drain) the TLS `LAST` error slot. Returns `Some(Error)` if a backend
/// error is pending, `None` otherwise. Used by trampolines and
/// fallible-handle constructors to surface the most recent backend error
/// alongside a NULL-ctx or non-zero rc return.
#[inline]
pub(crate) fn take_last() -> Option<Error> {
  LAST.with(|c| c.borrow_mut().take())
}

/// Stash an error into the TLS `LAST` slot — for use from `extern "C"`
/// trampolines that need to forward a Rust-side `Error` through the rc
/// channel back to a safe-layer caller's `check(rc)`. Non-panicking
/// `try_with`/`try_borrow_mut` keeps it safe under nested panics.
#[inline]
pub(crate) fn set_last(err: Error) {
  let _ = LAST.try_with(|c| {
    if let Ok(mut g) = c.try_borrow_mut() {
      *g = Some(err);
    }
  });
}

/// The most recent backend error recorded on this thread, if any. Used by
/// [`crate::diagnostics`] to surface mlx context when a panic follows a
/// backend failure. Non-panicking: `try_with` keeps it safe during thread
/// teardown, and `try_borrow` keeps it safe when called from inside a panic
/// hook that interrupted code already holding the `RefCell` borrow — a
/// borrow conflict yields `None` rather than a (double-)panic.
pub(crate) fn last_error_message() -> Option<String> {
  LAST
    .try_with(|c| {
      c.try_borrow()
        .ok()
        .and_then(|g| g.as_ref().map(|e| e.to_string()))
    })
    .ok()
    .flatten()
}

/// Set to `true` by the `#[ctor]` install. Read by the static-init smoke test
/// to verify the eager install ran (vs the lazy fallback rescuing it).
pub(crate) static INIT_VIA_CTOR: AtomicBool = AtomicBool::new(false);

extern "C" fn handler(msg: *const c_char, _data: *mut c_void) {
  // Panics across `extern "C"` are UB. Wrap everything in catch_unwind.
  let _ = catch_unwind(AssertUnwindSafe(|| {
    // SAFETY: mlx-c guarantees `msg` is a valid NUL-terminated C string for the
    // duration of this error-handler callback; the owned `String` copies it
    // out so nothing escapes the callback.
    let s = unsafe { CStr::from_ptr(msg) }
      .to_string_lossy()
      .into_owned();
    // Preserve a trampoline-set user error against mlx-c's generic
    // closure-non-zero wrapper. When a [`crate::transforms::closure`]
    // trampoline returns a non-zero rc, mlx-c's C++ wrappers (see
    // vendored `mlx-c/mlx/c/closure.cpp` lines 50, 83, 122, 188, 233,
    // 310, 357, 448, 495, 587, 638, 730, 784) throw
    // `std::runtime_error("mlx_closure...returned a non-zero value")`,
    // caught by the outer `mlx_closure*_apply` (or
    // `mlx_value_and_grad` / `mlx_vjp` / `mlx_jvp` / `mlx_custom_vjp` /
    // …) and surfaced via `mlx_error(e.what())` — which invokes this
    // handler. Without the preserve check below, that generic message
    // would overwrite the user's actual error (the trampoline already
    // called [`set_last`] with the user's `Err` or panic message before
    // returning the non-zero rc), and callers would see
    // "mlx_closure returned a non-zero value" instead of their own
    // error / panic payload.
    //
    // The wrapper messages all share the shape
    // `mlx_closure[_kind] returned a non-zero value at <file>:<line>`
    // — mlx-c surfaces them via `mlx_error(e.what())` where `e.what()`
    // includes the C++ source location suffix appended by the
    // `std::runtime_error` thrown at vendored
    // `mlx-c/mlx/c/closure.cpp` lines 50, 83, 122, 188, 233, 310, 357,
    // 448, 495, 587, 638, 730, 784. Match on the common prefix +
    // load-bearing inner phrase (NOT `ends_with`, since the trailing
    // `at <file>:<line>` varies by build root) so every variant is
    // covered without an explicit enumeration of the kinds.
    let is_generic_closure_wrapper =
      s.starts_with("mlx_closure") && s.contains("returned a non-zero value");
    let _ = LAST.try_with(|c| {
      if let Ok(mut g) = c.try_borrow_mut() {
        if is_generic_closure_wrapper && g.is_some() {
          // Preserve the trampoline's already-set user error.
          return;
        }
        *g = Some(Error::Backend(s));
      }
    });
  }));
}

#[ctor::ctor(unsafe)]
fn install_handler() {
  // **TEST-ONLY** env-var opt-out for the issue #223 stripped-ctor regression
  // fixture. When `MLXRS_DISABLE_CTOR_FOR_TEST=1` is set in the child process's
  // environment, the ctor skips its eager install. The
  // `ensure_handler_installed()` defense-in-depth call inside every safe-layer
  // FFI entry point is then the ONLY thing standing between mlx-c's default
  // `printf + exit(-1)` and a normal `Err` return — which is exactly what the
  // regression test `stripped_ctor_try_item::try_item_survives_stripped_ctor_environment`
  // exercises. Production binaries do NOT set this env var; it is read only
  // here, only on a process-start ctor, and only as `Result<_, _>::is_ok()`.
  // The check itself is async-signal-safe-equivalent (no allocator, no FFI)
  // and adds one env-var lookup per process start.
  if std::env::var("MLXRS_DISABLE_CTOR_FOR_TEST").is_ok() {
    return;
  }
  // SAFETY: handler is a valid extern "C" fn; null data ptr; no dtor needed.
  unsafe {
    mlxrs_sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
  }
  INIT_VIA_CTOR.store(true, Ordering::Relaxed);
}

/// Defense-in-depth installer. Every safe-layer entry point that invokes
/// mlx-c calls this before the FFI call so that, if the eager `#[ctor]`
/// install was skipped (older rustc toolchains below the rust#133491 fix
/// MSRV, consumer binaries that never reference any `mlxrs` symbol so the
/// linker drops the ctor section, or sandbox environments that disable
/// `__attribute__((constructor))`), the handler is installed before mlx-c
/// can invoke its default `printf + exit(-1)` and terminate the process.
///
/// Fast path is an atomic load + branch — `INIT_VIA_CTOR` is `true` after
/// either the ctor or this fallback has run, so subsequent calls return
/// immediately without touching the OnceLock.
#[inline]
pub(crate) fn ensure_handler_installed() {
  if INIT_VIA_CTOR.load(Ordering::Relaxed) {
    return;
  }
  ensure_handler_installed_slow();
}

#[cold]
#[inline(never)]
fn ensure_handler_installed_slow() {
  static FALLBACK: OnceLock<()> = OnceLock::new();
  FALLBACK.get_or_init(|| {
    // SAFETY: `handler` is a valid `extern "C"` fn pointer, the data pointer is
    // NULL, and no destructor is needed; installs the process-global mlx-c
    // error handler.
    unsafe {
      mlxrs_sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
    }
    INIT_VIA_CTOR.store(true, Ordering::Relaxed);
  });
}

/// Hot path: rc-pattern check. Returns `Ok(())` if `rc == 0`, else drains
/// the TLS slot into `Err`. Does NOT install the handler — callers must
/// have called `ensure_handler_installed` before the FFI call, since by the
/// time `check` runs the default abort handler would already have fired.
#[inline]
pub(crate) fn check(rc: c_int) -> Result<()> {
  if rc == 0 {
    Ok(())
  } else {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend(format!("mlx returned {rc} with no message"))),
    )
  }
}

/// Sentinel-handle pattern: for constructors that return `mlx_array` directly
/// with NULL `ctx` on failure (e.g. `mlx_array_new_data`). Same install
/// contract as [`check`].
#[inline]
pub(crate) fn check_handle(handle: mlxrs_sys::mlx_array) -> Result<crate::Array> {
  if handle.ctx.is_null() {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend("mlx returned null handle".into())),
    )
  } else {
    Ok(crate::Array(handle))
  }
}

/// Sentinel-handle pattern for `mlx_vector_array`-returning constructors
/// (e.g. `mlx_vector_array_new`): they report failure via the error handler
/// and return a handle with NULL `ctx`. Unlike [`check_handle`] the caller
/// keeps ownership of its handle (it is passed by value into the subsequent
/// mlx-c call and freed by its own RAII guard), so this returns `Result<()>`
/// like [`check`] — draining `LAST` into `Err` when `ctx` is null. Same
/// install contract as [`check`].
#[inline]
pub(crate) fn check_vector_array_handle(handle: mlxrs_sys::mlx_vector_array) -> Result<()> {
  if handle.ctx.is_null() {
    Err(
      LAST
        .with(|c| c.borrow_mut().take())
        .unwrap_or(Error::Backend(
          "mlx returned null vector_array handle".into(),
        )),
    )
  } else {
    Ok(())
  }
}

#[cfg(test)]
mod init_smoke {
  use super::*;

  #[test]
  fn ctor_fired() {
    assert!(
      INIT_VIA_CTOR.load(Ordering::Relaxed),
      "ctor install did not fire — likely symbol stripping or static-init ordering issue"
    );
  }

  #[test]
  fn failing_op_returns_err_not_abort() {
    // Clear stale TLS first — cargo test runs #[test] fns on the same
    // thread within a binary; a prior failing op could leave Some(..)
    // and produce a false-positive pass.
    super::LAST.with(|c| *c.borrow_mut() = None);

    let r = crate::Array::ones::<f32>(&(2, 2)).and_then(|a| a.reshape(&(3,)));

    assert!(
      matches!(r, Err(crate::Error::Backend(_))),
      "failing op aborted process or produced wrong error variant; \
       mlx-c++ may have overwritten our handler post-ctor — got: {r:?}"
    );
  }
}
