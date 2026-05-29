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
  collections::TryReserveError,
  ffi::{CStr, c_char, c_int, c_void},
  panic::{AssertUnwindSafe, catch_unwind},
  path::PathBuf,
  ptr,
  sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
  },
};

use smallvec::SmallVec;
use smol_str::SmolStr;

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

/// Payload for [`Error::ArithmeticOverflow`]: the operation context, result type, and
/// the offending runtime operands.
///
/// Construct via [`ArithmeticOverflowPayload::new`] (no operands) or
/// [`ArithmeticOverflowPayload::with_operands`]; access fields via
/// `context()`, `op_type()`, and `operands()`. The `operands` field
/// preserves the actual runtime values that triggered the overflow so
/// the caller can diagnose oversized inputs without re-running the
/// failing op or parsing message text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticOverflowPayload {
  context: &'static str,
  op_type: &'static str,
  operands: SmallVec<[(&'static str, u64); 4]>,
}

impl ArithmeticOverflowPayload {
  /// Construct a payload without operands. Prefer [`Self::with_operands`]
  /// at every site where the runtime value(s) that triggered the
  /// overflow are reachable — dropping the operands loses the
  /// diagnostic at the cost of one `format!` to recover it.
  pub fn new(context: &'static str, op_type: &'static str) -> Self {
    Self {
      context,
      op_type,
      operands: SmallVec::new(),
    }
  }

  /// Construct a payload carrying the named runtime operands that
  /// triggered the overflow (e.g. `[("a", 1u64 << 32), ("b", 4)]` for an
  /// overflowing `a * b`).
  pub fn with_operands(
    context: &'static str,
    op_type: &'static str,
    operands: impl IntoIterator<Item = (&'static str, u64)>,
  ) -> Self {
    Self {
      context,
      op_type,
      operands: operands.into_iter().collect(),
    }
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

  /// The offending runtime operands (`(name, value)` pairs). Empty when
  /// constructed via [`Self::new`].
  #[inline(always)]
  pub fn operands(&self) -> &[(&'static str, u64)] {
    &self.operands
  }
}

impl std::fmt::Display for ArithmeticOverflowPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.operands.is_empty() {
      write!(f, "{}: overflow ({})", self.context, self.op_type)
    } else {
      write!(
        f,
        "{}: overflow ({}) with operands",
        self.context, self.op_type
      )?;
      let mut first = true;
      f.write_str(" ")?;
      for (name, value) in &self.operands {
        if !first {
          f.write_str(", ")?;
        }
        write!(f, "{name}={value}")?;
        first = false;
      }
      Ok(())
    }
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
/// `value` is a [`SmolStr`] — most range-violation values are short
/// numeric strings ("1024", "-3.14") that fit the inline storage
/// without a heap allocation.
///
/// Construct via [`OutOfRangePayload::new`]; access fields via `context()`,
/// `requirement()`, and `value()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfRangePayload {
  context: &'static str,
  requirement: &'static str,
  value: SmolStr,
}

impl OutOfRangePayload {
  /// Construct a new payload. `value` accepts anything `impl Into<SmolStr>`
  /// (`&str`, `String`, `SmolStr`, `format_smolstr!(…)` output) so call
  /// sites do not need to think about which string type to pass.
  pub fn new(context: &'static str, requirement: &'static str, value: impl Into<SmolStr>) -> Self {
    Self {
      context,
      requirement,
      value: value.into(),
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
///
/// # Modeling discipline (the §5 rule)
///
/// **Never construct an error variant with a `format!()`-built string payload.**
/// Every error must use a typed variant whose payload carries every runtime
/// substitution as a discrete field — this preserves typed access for
/// callers and downstream tooling, and is the entire reason this is an
/// `enum` rather than a `type Error = String`. If an existing variant
/// doesn't fit the call site's needs, ADD A NEW PAYLOAD STRUCT to this
/// module rather than falling back to [`Error::MlxC`] / [`Error::ShapeMismatch`].
///
/// The free-form-string variants in this enum are reserved for two
/// narrow use cases ONLY:
///
/// 1. [`Error::MlxC`] is for the raw mlx-c handler thread-local drain
///    when [`MlxOpKind::parse_prefix`] could not extract a typed
///    `[op_name]` prefix from the upstream message. **Never construct
///    this elsewhere** — every other call site has structured runtime
///    data and must use a typed variant.
/// 2. [`Error::ShapeMismatch`] is a stepping-stone for the in-progress
///    migration of pre-§5 shape-mismatch sites and is being phased out
///    in favor of [`Error::RankMismatch`] / [`Error::LengthMismatch`] /
///    [`Error::MultiLengthMismatch`] / [`Error::ShapePairMismatch`] /
///    [`Error::DivisibilityConstraint`]. **Never construct new
///    `ShapeMismatch` sites** — pick the typed shape variant whose
///    fields carry every runtime substitution.
#[derive(Debug, thiserror::Error, derive_more::IsVariant)]
#[non_exhaustive]
pub enum Error {
  /// **DEPRECATED for new construction** — see the enum doc. Reserved for
  /// in-progress migration of pre-§5 shape-mismatch sites; new code must
  /// pick the typed shape variant whose fields carry every runtime
  /// substitution ([`Error::RankMismatch`] / [`Error::LengthMismatch`] /
  /// [`Error::MultiLengthMismatch`] / [`Error::ShapePairMismatch`] /
  /// [`Error::DivisibilityConstraint`]).
  #[error("shape mismatch: {0}")]
  ShapeMismatch(String),

  /// Dtype mismatch (e.g. requesting `as_slice::<f32>` on an i32 array).
  #[error("dtype mismatch: expected {:?}, got {:?}", .0.expected(), .0.got())]
  DtypeMismatch(DtypeMismatchPayload),

  /// A dtype gate rejected an input dtype that is not in the allowed set
  /// (e.g. `Adam: weights must be floating, got Int32`). Carries the
  /// rejected dtype and the static list of supported dtypes.
  #[error(transparent)]
  UnsupportedDtype(UnsupportedDtypePayload),

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

  /// Typed mlx-c boundary error — the mlx C++ exception message starts
  /// with a `[op_name]` prefix that we parsed into a typed
  /// [`MlxOpKind`]. Carries the typed op + the full original message.
  ///
  /// Constructed by the crate-private `check` / `check_handle` boundary
  /// helpers when [`MlxOpKind::parse_prefix`] succeeded. Most mlx-c
  /// errors land here rather than [`Error::MlxC`] because mlx C++
  /// consistently emits the `[op_name]` prefix.
  #[error(transparent)]
  MlxOp(MlxOpPayload),

  /// Raw mlx-c handler message — fallback for when
  /// [`MlxOpKind::parse_prefix`] could not extract a typed `[op_name]`
  /// prefix from the upstream message (e.g. internal mlx debug strings,
  /// non-primitive errors).
  ///
  /// **This is the ONLY string-typed error variant for new code, and
  /// it MUST be constructed only from the mlx-c handler drain** inside
  /// the crate-private `check` / `check_handle` /
  /// `check_vector_array_handle` boundary helpers. Every other call
  /// site has structured runtime data and must use a typed variant.
  #[error("mlx-c: {0}")]
  MlxC(SmolStr),

  /// **DEPRECATED for new construction** — see the enum doc. Reserved
  /// for in-progress migration of pre-§5 Backend(format!) sites. New
  /// code MUST pick a typed variant (or [`Error::MlxOp`] / [`Error::MlxC`]
  /// for genuine mlx-c handler pass-through). Final cleanup PR removes
  /// this variant after the 6 per-module migrations land.
  #[error("mlx backend: {0}")]
  Backend(String),

  /// A serialized / structured input is malformed, truncated, or in an
  /// unsupported shape, as detected by one of mlxrs's own hand-rolled
  /// readers / validators (e.g. a corrupt SentencePiece protobuf field,
  /// a `tokenizer.json` `model.vocab` entry with the wrong arity, or a
  /// fine-tuning jsonl record matching no supported dataset format).
  /// `context` identifies the reader / format; `detail` is a static
  /// description of the violation. Distinct from [`Error::Parse`] which
  /// wraps an inner external-parser error.
  #[error(transparent)]
  MalformedData(MalformedDataPayload),

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
  #[error(transparent)]
  MultiLengthMismatch(MultiLengthMismatchPayload),

  /// Two FULL shapes disagree (e.g. `expected [B, S, D]`, `actual [B, T, D]`).
  /// Distinct from [`Error::RankMismatch`] (rank differs) and
  /// [`Error::LengthMismatch`] (single dim differs); carries both
  /// complete shapes for downstream tooling.
  #[error(transparent)]
  ShapePairMismatch(ShapePairMismatchPayload),

  /// A dividend is not a multiple of a divisor (e.g. AWQ `in_features`
  /// not divisible by `group_size`).
  #[error(transparent)]
  DivisibilityConstraint(DivisibilityConstraintPayload),

  /// A scalar that must be finite was NaN or Inf.
  #[error(transparent)]
  NonFiniteScalar(NonFiniteScalarPayload),

  /// A runtime-keyed lookup failed (e.g. missing layer weight; distinct
  /// from [`Error::MissingField`] which carries a static field name).
  #[error(transparent)]
  MissingKey(MissingKeyPayload),

  /// A string-keyed dispatch missed every known variant (e.g. unknown
  /// pooling mode); carries the static list of supported names.
  #[error(transparent)]
  UnknownEnumValue(UnknownEnumValuePayload),

  /// Two mutually-exclusive keys were both present (e.g. AWQ qweight +
  /// the converted weight in the same checkpoint).
  #[error(transparent)]
  KeyCollision(KeyCollisionPayload),

  /// An input string/byte slice contained an interior NUL byte
  /// (rejecting it before passing to mlx-c which uses NUL-terminated
  /// strings).
  #[error(transparent)]
  InteriorNul(InteriorNulPayload),

  /// An input or computed quantity exceeded a documented cap.
  #[error(transparent)]
  CapExceeded(CapExceededPayload),

  /// A request-scaled `try_reserve`/`try_reserve_exact` failed (carries
  /// the request size and the underlying allocator error).
  #[error(transparent)]
  AllocFailure(AllocFailurePayload),

  /// An external parser (JSON, regex, tokenizer.json) failed (carries
  /// the inner error as `Box<dyn Error>` for source-chain walking).
  #[error(transparent)]
  Parse(ParsePayload),

  /// An external-library runtime / device-backend operation failed
  /// (e.g. `cpal` audio device backend, `image` decoder backend). Distinct
  /// from [`Error::Parse`] (external PARSERS) and [`Error::MlxOp`] /
  /// [`Error::MlxC`] (the mlx-c boundary).
  #[error(transparent)]
  ExternalOp(ExternalOpPayload),

  /// A decoder produced more elements than the documented cap
  /// (e.g. truncated/malicious audio stream).
  #[error(transparent)]
  BoundedDecode(BoundedDecodePayload),

  /// A typed inner error from a specific named layer — allows wrapping
  /// any sub-error with a runtime layer identifier without losing the
  /// inner variant.
  #[error(transparent)]
  LayerKeyed(LayerKeyedPayload),

  /// Tokenizer subsystem error (HF tokenizer load/encode/decode, chat-template
  /// render, tool-call parse). Only constructed when the `tokenizer` feature
  /// is enabled. The message carries the underlying cause.
  #[cfg(feature = "tokenizer")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tokenizer")))]
  #[error("tokenizer: {0}")]
  Tokenizer(SmolStr),

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
  pub(crate) fn tokenizer(message: impl Into<SmolStr>) -> Self {
    Self::Tokenizer(message.into())
  }
}

// ────────────────────────────────────────────────────────────────────────────
// mlx-c boundary error: typed prefix + raw message fallback
// ────────────────────────────────────────────────────────────────────────────

/// The mlx-c primitive whose C++-side exception surfaced via the error
/// handler — parsed from the `[op_name] …` prefix the mlx C++ side
/// consistently emits via `std::invalid_argument` / `std::runtime_error`.
///
/// Used as the `op` field of [`MlxOpPayload`] so callers can branch on
/// the failing op without parsing message text. The catch-all
/// [`MlxOpKind::Other`] carries the raw prefix when the message starts
/// with `[…]` but the contents aren't in this enum's enumerated set; if
/// the message has no `[…]` prefix at all, the boundary emits
/// [`Error::MlxC`] (raw `SmolStr`) instead.
///
/// All discriminants except [`MlxOpKind::Other`] are statically known;
/// [`MlxOpKind::Other`] carries a [`SmolStr`] of the unknown op prefix
/// (inline-stored when short) so this enum is `Clone` but not `Copy`.
///
/// The enumerated set is built from the actual vendored mlx C++ throw
/// prefixes (`mlxrs-sys/vendor/mlx/mlx/*.cpp`); both the lowercase
/// function-name form (`[matmul]`) and the CapCase class-name form
/// (`[QuantizedMatmul::vjp]`) map to the same typed variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlxOpKind {
  /// `[matmul]` / `[addmm]` / `[QuantizedMatmul]` (and `::vjp` / `::jvp`
  /// / `::vmap` qualifiers) / `[block_masked_mm]` / `[BlockMaskedMM]` /
  /// `[gather_mm]` / `[GatherMM]` / `[gather_qmm]` / `[GatherQMM::vjp]` /
  /// `[qqmm]` / `[segmented_mm]` / `[inner]` / `[tensordot]` / `[kron]`.
  Matmul,
  /// `[reshape]` / `[unflatten]` / `[Unflatten]` / `[flatten]` /
  /// `[expand_dims]` / `[squeeze]` / `[transpose]` / `[swapaxes]` /
  /// `[moveaxis]` / `[view]`.
  Reshape,
  /// `[broadcast_shapes]` / `[broadcast_to]` / `[broadcast_arrays]` /
  /// `[Broadcast]`.
  Broadcast,
  /// `[shape]` (mlx array shape accessor).
  Shape,
  /// `[slice]` / `[slice_update]` / `[SliceUpdate]` / `[DynamicSlice]`
  /// (with `::vjp` / `::vmap` qualifiers) / `[DynamicSliceUpdate]` /
  /// `[split]` / `[trace]` / `[diag]` / `[diagonal]` / `[tril]` / `[triu]`.
  Slice,
  /// `[concatenate]` / `[stack]` / `[repeat]` / `[meshgrid]`.
  Concat,
  /// `[gather]` / `[Gather]` / `[gather_axis]`.
  Gather,
  /// `[scatter]` / `[scatter_axis]` / `[scatter_add_axis]` /
  /// `[masked_scatter]` / `[put_along_axis]`.
  Scatter,
  /// `[take]` / `[take_along_axis]`.
  Take,
  /// `[fftn]` / `[fftfreq]` / `[rfftfreq]` / `[fftshift]` / `[ifftshift]` /
  /// `[hadamard_transform]`.
  Fft,
  /// `[quantize]` / `[quantized_matmul]` / `[from_fp8]` / `[to_fp8]`
  /// (the `GatherQMM` / `QuantizedMatmul` aliases are routed to
  /// [`Self::Matmul`]).
  Quantize,
  /// `[dequantize]`.
  Dequantize,
  /// `[conv]` (1-D/2-D/3-D/transpose variants share this prefix).
  Conv,
  /// Reduction ops: `[sum]` / `[mean]` / `[max]` / `[min]` / `[prod]` /
  /// `[median]` / `[var]` / `[any]` / `[all]` / `[logsumexp]` /
  /// `[logcumsumexp]` / `[cumsum]` / `[cumprod]` / `[cummax]` / `[cummin]` /
  /// `[number_of_elements]` / `[softmax]` / `[topk]`.
  Pool,
  /// `[eval]` / `[async_eval]` / `[Compiled]` (compiled-graph eval-time
  /// failure) / `[Primitive::vjp]` / `[Primitive::jvp]` / `[Primitive::vmap]` /
  /// `[Primitive::output_shapes]`.
  Eval,
  /// `[sort]` / `[partition]`.
  Sort,
  /// `[argsort]` / `[argpartition]` / `[argmax]` / `[argmin]`.
  ArgSort,
  /// `[layer_norm]` / `[rms_norm]` / `[rope]` / `[RoPE::vjp]` /
  /// `[scaled_dot_product_attention]` / `[scale_dot_product_attention]`.
  Norm,
  /// `[linalg::*]` family — `cholesky` / `cholesky_inv` / `cross` /
  /// `eig` / `eigh` / `eigvals` / `eigvalsh` / `inv` / `lu` /
  /// `lu_factor` / `norm` / `pinv` / `qr` / `solve` /
  /// `solve_triangular` / `svd`.
  Linalg,
  /// Random-sampler ops: `[arange]` / `[full]` / `[eye]` / `[linspace]` /
  /// `[uniform]` / `[normal]` / `[trunc_normal]` / `[bernoulli]` /
  /// `[categorical]` / `[multivariate_normal]` / `[laplace]` /
  /// `[randint]` / `[bits]` / `[finfo]` / `[iinfo]`.
  Random,
  /// `[grad]` / `[vjp]` / `[jvp]` / `[vmap]` / `[compile]` (transform-time
  /// failure outside an eval).
  Transform,
  /// `[astype]` / `[nan_to_num]` / `[negative]` / `[floor]` /
  /// `[bitwise_invert]` / `[divmod]` / `[Pad::vmap]` / general
  /// element-wise op failure with no more-specific category.
  Elementwise,
  /// `[import_function]` / `[import_function::call]` / `[export_function]` /
  /// `[deserialize_variant]` / `[Event::stream]` / `[StreamContext]` /
  /// `[ThreadPool::enqueue]` / `[set_default_device]` /
  /// `[set_default_stream]` / `[default_stream]` — runtime/system
  /// infrastructure failure.
  System,
  /// `[roll]` / `[unflatten]`-style positional ops not better fit
  /// elsewhere — currently empty in vendor, reserved for future.
  Positional,
  /// Distributed-collective ops: `[AllGather::eval_gpu]` /
  /// `[AllReduce::eval_gpu]` / `[ReduceScatter]` / `[Send::eval_gpu]` /
  /// `[Recv::eval_gpu]` / `[sum_scatter]` / `[distributed]` /
  /// `[mpi]` / `[nccl]` / `[jaccl]` / `[ring]`.
  Distributed,
  /// IO primitives: `[load]` / `[save]` / `[load_safetensors]` /
  /// `[save_safetensors]` / `[load_gguf]` / `[save_gguf]` /
  /// `[safetensor]` / `[read]` / `[write]` / `[Load::eval_gpu]`.
  ///
  /// **Distinct from [`Error::FileIo`]** — this is the mlx C++ primitive
  /// name surfaced via the boundary handler; [`Error::FileIo`] is the
  /// safe-wrapper typed io payload constructed at Rust-side `std::fs`
  /// call sites.
  Io,
  /// Any other `[op]` prefix not mapped above. Carries the raw prefix
  /// (after stripping `::method` and `.qualifier` suffixes) as a
  /// [`SmolStr`] so a future caller can pattern-match on the raw op
  /// name without re-parsing the full message.
  Other(SmolStr),
}

impl std::fmt::Display for MlxOpKind {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Matmul => f.write_str("matmul"),
      Self::Reshape => f.write_str("reshape"),
      Self::Broadcast => f.write_str("broadcast"),
      Self::Shape => f.write_str("shape"),
      Self::Slice => f.write_str("slice"),
      Self::Concat => f.write_str("concat"),
      Self::Gather => f.write_str("gather"),
      Self::Scatter => f.write_str("scatter"),
      Self::Take => f.write_str("take"),
      Self::Fft => f.write_str("fft"),
      Self::Quantize => f.write_str("quantize"),
      Self::Dequantize => f.write_str("dequantize"),
      Self::Conv => f.write_str("conv"),
      Self::Pool => f.write_str("reduce"),
      Self::Eval => f.write_str("eval"),
      Self::Sort => f.write_str("sort"),
      Self::ArgSort => f.write_str("argsort"),
      Self::Norm => f.write_str("norm"),
      Self::Linalg => f.write_str("linalg"),
      Self::Random => f.write_str("random"),
      Self::Transform => f.write_str("transform"),
      Self::Elementwise => f.write_str("elementwise"),
      Self::System => f.write_str("system"),
      Self::Positional => f.write_str("positional"),
      Self::Distributed => f.write_str("distributed"),
      Self::Io => f.write_str("io"),
      Self::Other(s) => f.write_str(s),
    }
  }
}

impl MlxOpKind {
  /// Parse the leading `[op_name]` prefix from an mlx-c handler message.
  ///
  /// mlx C++ consistently emits messages starting with `[op_name] …` for
  /// every `std::invalid_argument` / `std::runtime_error` raised inside
  /// the primitive validation paths. The op_name comes in three syntactic
  /// shapes that all normalize to the same primary identifier here:
  ///
  /// 1. Bare function name: `[matmul]`, `[broadcast_shapes]`.
  /// 2. Class name (UpperCamelCase): `[Broadcast]`, `[GatherMM]`,
  ///    `[BlockMaskedMM]`, `[Compiled]`, `[SliceUpdate]`, `[Unflatten]`.
  /// 3. Qualified class method: `[QuantizedMatmul::vjp]`,
  ///    `[DynamicSlice::vmap]`, `[GatherQMM::vjp]`, `[Pad::vmap]`,
  ///    `[Primitive::output_shapes]`, `[linalg::cholesky]`,
  ///    `[import_function::call]`.
  ///
  /// The parser strips any `::method` and `.qualifier` suffixes, then
  /// matches the primary identifier against the enumerated set
  /// case-insensitively (so `Broadcast` and `broadcast_shapes` both
  /// land in [`MlxOpKind::Broadcast`]).
  ///
  /// Returns `None` if the message has NO leading `[…]` prefix — the
  /// caller should then emit [`Error::MlxC`] (raw message) instead.
  ///
  /// The full enumerated mapping is verified by
  /// `init_smoke::mlx_op_kind_parses_real_vendor_prefixes` against the
  /// actual prefixes extracted from `mlxrs-sys/vendor/mlx/mlx/*.cpp`.
  pub fn parse_prefix(msg: &str) -> Option<Self> {
    let rest = msg.strip_prefix('[')?;
    let end = rest.find(']')?;
    let prefix = &rest[..end];
    // Namespace routing — the leading `namespace::` of a class path
    // determines the typed bucket regardless of the trailing class /
    // method name. Checked before the generic `::`-strip so the
    // namespace itself doesn't get discarded as the "primary" name.
    if let Some(rest_after_ns) = prefix.strip_prefix("linalg::") {
      let _ = rest_after_ns;
      return Some(Self::Linalg);
    }
    if let Some(rest_after_ns) = prefix.strip_prefix("Primitive::") {
      let _ = rest_after_ns;
      return Some(Self::Eval);
    }
    if let Some(rest_after_ns) = prefix.strip_prefix("import_function::") {
      let _ = rest_after_ns;
      return Some(Self::System);
    }
    // gpu::* / metal::* / Metal::* are all backend-runtime / system
    // infrastructure prefixes from mlx's GPU / Metal backend layer.
    if prefix.starts_with("gpu::")
      || prefix.starts_with("metal::")
      || prefix.starts_with("Metal::")
      || prefix.starts_with("Event::")
      || prefix.starts_with("Fence::")
    {
      return Some(Self::System);
    }
    // `fast::*` is the mlx fast-path namespace — strip it (iteratively,
    // not recursively — adversarial input like `fast::fast::fast::…`
    // would otherwise blow the stack and abort the process. `parse_prefix`
    // is reachable from the mlx-c handler, which is called on inputs we
    // do not control.) so `[fast::Quantize::eval_cpu]` reduces to
    // `Quantize::eval_cpu` before the generic `::method`-strip runs below.
    let stripped = {
      let mut s = prefix;
      while let Some(rest) = s.strip_prefix("fast::") {
        s = rest;
      }
      s
    };
    // Strip `::method` qualifier (class-method form, e.g.
    // `[QuantizedMatmul::vjp]`, `[Cholesky::eval_cpu]`), then `.qualifier`
    // suffix (compiled-graph / sub-op form, e.g. `[matmul.gemm]`).
    let no_method = stripped.split("::").next().unwrap_or(stripped);
    let primary = no_method.split('.').next().unwrap_or(no_method);
    // Case-insensitive primary-name match (mlx uses `[Broadcast]`
    // class-style alongside `[broadcast_shapes]` function-style).
    let lower = primary.to_lowercase();
    Some(match lower.as_str() {
      // matmul-family — both function-name forms (`[matmul]`,
      // `[quantized_matmul]`) and class forms (`[QuantizedMatmul]`,
      // `[GatherMM]`, `[Matmul::eval_cpu]`, `[QQMatmul::eval_gpu]`)
      // route here; `quantized_matmul` is semantically a matmul
      // operating on quantized weights, same as the class-form sibling
      // `[QuantizedMatmul]`.
      "matmul" | "addmm" | "quantizedmatmul" | "quantized_matmul" | "block_masked_mm"
      | "blockmaskedmm" | "gather_mm" | "gathermm" | "gather_qmm" | "gatherqmm" | "qqmm"
      | "qqmatmul" | "segmented_mm" | "inner" | "tensordot" | "kron" | "gemm_and_bias" => {
        Self::Matmul
      }
      // reshape-family
      "reshape" | "unflatten" | "flatten" | "expand_dims" | "squeeze" | "transpose"
      | "swapaxes" | "moveaxis" | "view" => Self::Reshape,
      // broadcast-family
      "broadcast" | "broadcast_shapes" | "broadcast_to" | "broadcast_arrays" => Self::Broadcast,
      // shape
      "shape" => Self::Shape,
      // slice-family (incl. trace/diag-family which thiserror-style trace
      // shares the slice-into-array semantic)
      "slice"
      | "slice_update"
      | "sliceupdate"
      | "dynamicslice"
      | "dynamicsliceupdate"
      | "dynamic_slice"
      | "dynamic_slice_update"
      | "split"
      | "trace"
      | "diag"
      | "diagonal"
      | "tril"
      | "triu" => Self::Slice,
      // concat-family
      "concatenate" | "stack" | "repeat" | "meshgrid" => Self::Concat,
      // gather-family (incl. backend class `[GatherAxis::eval_cpu]`)
      "gather" | "gather_axis" | "gatheraxis" => Self::Gather,
      // scatter-family (incl. masked_scatter, put_along_axis, and backend
      // class `[ScatterAxis::eval_cpu]`, `[MaskedScatter::eval_cpu]`)
      "scatter" | "scatter_axis" | "scatter_add_axis" | "scatter_add" | "scatter_max"
      | "scatter_min" | "scatter_prod" | "scatteraxis" | "masked_scatter" | "maskedscatter"
      | "put_along_axis" => Self::Scatter,
      // take-family
      "take" | "take_along_axis" => Self::Take,
      // fft-family (incl. bare `hadamard` and CapCase `[FFT]` via lowercase)
      "fft" | "ifft" | "rfft" | "irfft" | "fft2" | "ifft2" | "fftn" | "ifftn" | "fftfreq"
      | "rfftfreq" | "fftshift" | "ifftshift" | "hadamard" | "hadamard_transform" => Self::Fft,
      // quantize-family (`quantized_matmul` is routed to Matmul above as
      // a matmul-family op).
      "quantize" | "block_quantized" | "from_fp8" | "to_fp8" | "quantize_dequantize" => {
        Self::Quantize
      }
      // dequantize
      "dequantize" => Self::Dequantize,
      // conv-family (both bare `conv` and class-form `[Convolution::eval]`)
      "conv" | "conv1d" | "conv2d" | "conv3d" | "conv_transpose" | "convolution" => Self::Conv,
      // reduction-family
      "max_pool" | "avg_pool" | "reduce" | "all" | "any" | "sum" | "prod" | "mean" | "var"
      | "max" | "min" | "median" | "logsumexp" | "logcumsumexp" | "cumsum" | "cumprod"
      | "cummax" | "cummin" | "number_of_elements" | "softmax" | "topk" => Self::Pool,
      // eval / compiled-graph (incl. `[Copy::eval_gpu]`, `[Compile::eval_cpu]`,
      // `[NanEqual::eval_cpu]` — primitive eval-time failures with no more-
      // specific bucket).
      "compiledarray" | "compiled" | "compile" | "eval" | "async_eval" | "copy" | "nanequal" => {
        Self::Eval
      }
      // sort-family
      "sort" | "partition" => Self::Sort,
      // argsort-family
      "argsort" | "argpartition" | "argmax" | "argmin" => Self::ArgSort,
      // norm-family (incl. backend `[vjp_layer_norm]`)
      "mlx_norm"
      | "layer_norm"
      | "rms_norm"
      | "group_norm"
      | "rope"
      | "scaled_dot_product_attention"
      | "scale_dot_product_attention"
      | "vjp_layer_norm" => Self::Norm,
      // transform-family (`grad` / `vjp` / `jvp` / `vmap` — the bare-form
      // prefixes; the `Primitive::*` / `*::vjp` / `*::vmap` forms are
      // routed to the parent op above; the bare `compile` already routed
      // to Eval above).
      "grad" | "vjp" | "jvp" | "vmap" | "pad" => Self::Transform,
      // random / array-builder ops (incl. CapCase eval-time forms
      // `[Arange::eval_gpu]`, `[RandomBits::eval_gpu]`)
      "arange"
      | "full"
      | "eye"
      | "linspace"
      | "uniform"
      | "normal"
      | "trunc_normal"
      | "bernoulli"
      | "categorical"
      | "multivariate_normal"
      | "laplace"
      | "randint"
      | "bits"
      | "finfo"
      | "iinfo"
      | "randombits" => Self::Random,
      // elementwise / one-off (incl. backend dispatchers for binary/unary
      // kernels and the bare `[Abs]` form)
      "astype"
      | "nan_to_num"
      | "negative"
      | "floor"
      | "bitwise_invert"
      | "divmod"
      | "roll"
      | "abs"
      | "binary_float"
      | "binary_int"
      | "unary_fp"
      | "unary_int"
      | "unary_real"
      | "extract_tensor_data" => Self::Elementwise,
      // system / runtime infrastructure (incl. `Metal` / `metal` / `METAL`
      // bare-form prefixes, `[malloc]` allocator, `[new_stream]`, and the
      // `metal_kernel` / `cuda_kernel` / `custom_kernel` kernel-launch
      // failures; the `metal::*` / `Metal::*` / `gpu::*` / `Event::*` /
      // `Fence::*` namespace forms are routed in the up-front namespace
      // check above)
      "event"
      | "streamcontext"
      | "threadpool"
      | "set_default_device"
      | "set_default_stream"
      | "default_stream"
      | "deserialize_variant"
      | "export_function"
      | "import_function"
      | "rope::vjp"
      | "metal"
      | "malloc"
      | "new_stream"
      | "metal_kernel"
      | "cuda_kernel"
      | "custom_kernel" => Self::System,
      // linalg primitive class names (mlx C++ backend throws these as
      // bare CapCase `[Cholesky::eval_cpu]`, `[SVD::eval_gpu]`, etc. —
      // same family as the `linalg::*` namespace forms routed up-front).
      "cholesky" | "eig" | "eigh" | "inverse" | "luf" | "qrf" | "svd" => Self::Linalg,
      // distributed-collective ops (mpi/nccl/jaccl/ring are the transport
      // backends; AllGather/AllReduce/ReduceScatter/Send/Recv are the
      // primitive op names; sum_scatter is the function form)
      "allgather" | "allreduce" | "reducescatter" | "send" | "recv" | "sum_scatter"
      | "distributed" | "mpi" | "nccl" | "jaccl" | "ring" => Self::Distributed,
      // IO primitives — mlx-c's load/save handlers (distinct from our
      // safe-wrapper `Error::FileIo` payload at Rust `std::fs` sites).
      "load" | "save" | "load_safetensors" | "save_safetensors" | "load_gguf" | "save_gguf"
      | "safetensor" | "read" | "write" | "from_str" => Self::Io,
      // Typed catch-all: preserve the original bracket-content as a
      // SmolStr (post-stripping) so a future caller can pattern-match on
      // the raw op name without re-parsing the full message.
      _ => Self::Other(SmolStr::new(primary)),
    })
  }
}

/// Payload for [`Error::MlxOp`]: the typed mlx-c op + full message.
///
/// Construct via [`MlxOpPayload::new`]; access fields via `op()` and `message()`.
/// The message preserves the FULL original mlx-c handler text (including the
/// `[op]` prefix) so downstream display/logging matches what mlx emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxOpPayload {
  op: MlxOpKind,
  message: SmolStr,
}

impl MlxOpPayload {
  /// Construct a new payload. `op` should be derived via [`MlxOpKind::parse_prefix`]
  /// from `message`'s `[op]` prefix; the constructor does not re-parse.
  pub fn new(op: MlxOpKind, message: impl Into<SmolStr>) -> Self {
    Self {
      op,
      message: message.into(),
    }
  }

  /// The typed mlx-c op kind (parsed from the `[op]` prefix).
  pub fn op(&self) -> &MlxOpKind {
    &self.op
  }

  /// The full mlx-c handler message, including the leading `[op]` prefix.
  #[inline(always)]
  pub fn message(&self) -> &str {
    &self.message
  }
}

impl std::fmt::Display for MlxOpPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "mlx {}: {}", self.op, self.message)
  }
}

impl std::error::Error for MlxOpPayload {}

// ────────────────────────────────────────────────────────────────────────────
// 13 new typed payload structs (foundation-PR: replace Backend(format!) +
// ShapeMismatch(format!) sites with these in the per-module migration PRs).
// ────────────────────────────────────────────────────────────────────────────

/// Payload for [`Error::MissingKey`]: a runtime-keyed lookup failure
/// (e.g. "layer X is missing weight Y"). Distinct from
/// [`Error::MissingField`] which carries a static field name; this
/// variant carries a RUNTIME key (layer prefix, file name, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingKeyPayload {
  context: &'static str,
  key: SmolStr,
}

impl MissingKeyPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, key: impl Into<SmolStr>) -> Self {
    Self {
      context,
      key: key.into(),
    }
  }
  /// Call-site label (e.g. `"dequantize_weights: missing .weight"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The runtime key that was looked up (e.g. layer prefix).
  #[inline(always)]
  pub fn key(&self) -> &str {
    &self.key
  }
}

impl std::fmt::Display for MissingKeyPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}: key `{}` not found", self.context, self.key)
  }
}

impl std::error::Error for MissingKeyPayload {}

/// Payload for [`Error::UnknownEnumValue`]: a string-keyed dispatch
/// missed every known variant. `supported` is a static list of valid
/// names so the error message can suggest them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEnumValuePayload {
  type_name: &'static str,
  value: SmolStr,
  supported: &'static [&'static str],
}

impl UnknownEnumValuePayload {
  /// Construct a new payload.
  pub fn new(
    type_name: &'static str,
    value: impl Into<SmolStr>,
    supported: &'static [&'static str],
  ) -> Self {
    Self {
      type_name,
      value: value.into(),
      supported,
    }
  }
  /// The parent type whose enum the value didn't match
  /// (e.g. `"PoolingStrategy"`).
  #[inline(always)]
  pub const fn type_name(&self) -> &'static str {
    self.type_name
  }
  /// The runtime value that didn't match any known variant.
  #[inline(always)]
  pub fn value(&self) -> &str {
    &self.value
  }
  /// The static list of supported variant names (for suggesting).
  #[inline(always)]
  pub const fn supported(&self) -> &'static [&'static str] {
    self.supported
  }
}

impl std::fmt::Display for UnknownEnumValuePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: unknown value `{}` (supported: {:?})",
      self.type_name, self.value, self.supported
    )
  }
}

impl std::error::Error for UnknownEnumValuePayload {}

/// Payload for [`Error::NonFiniteScalar`]: a scalar that must be finite
/// was NaN or Inf. `value` is the offending f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonFiniteScalarPayload {
  context: &'static str,
  value: f64,
}

impl NonFiniteScalarPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, value: f64) -> Self {
    Self { context, value }
  }
  /// Call-site label (e.g. `"LearningRate: resolved value at step 5"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The offending non-finite value.
  #[inline(always)]
  pub const fn value(&self) -> f64 {
    self.value
  }
}

impl std::fmt::Display for NonFiniteScalarPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: value is non-finite (NaN or Inf): {}",
      self.context, self.value
    )
  }
}

impl std::error::Error for NonFiniteScalarPayload {}

/// Payload for [`Error::KeyCollision`]: two mutually-exclusive keys
/// were both present (e.g. AWQ "qweight" + the converted "weight" in
/// the same checkpoint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyCollisionPayload {
  context: &'static str,
  key: SmolStr,
}

impl KeyCollisionPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, key: impl Into<SmolStr>) -> Self {
    Self {
      context,
      key: key.into(),
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The key that collided.
  #[inline(always)]
  pub fn key(&self) -> &str {
    &self.key
  }
}

impl std::fmt::Display for KeyCollisionPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}: key `{}` collides", self.context, self.key)
  }
}

impl std::error::Error for KeyCollisionPayload {}

/// Payload for [`Error::InteriorNul`]: an input string/byte slice
/// contained an interior NUL byte (rejecting it before passing to mlx-c
/// which uses NUL-terminated strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteriorNulPayload {
  context: &'static str,
  bytes_kind: &'static str,
}

impl InteriorNulPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, bytes_kind: &'static str) -> Self {
    Self {
      context,
      bytes_kind,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The kind of input that contained the NUL (e.g. `"array key"`,
  /// `"metadata value"`).
  #[inline(always)]
  pub const fn bytes_kind(&self) -> &'static str {
    self.bytes_kind
  }
}

impl std::fmt::Display for InteriorNulPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: {} contains an interior NUL byte",
      self.context, self.bytes_kind
    )
  }
}

impl std::error::Error for InteriorNulPayload {}

/// Payload for [`Error::CapExceeded`]: an input or computed quantity
/// exceeded a documented cap (e.g. `MAX_DECODED_SAMPLES`,
/// `MAX_LFILTER_SAMPLES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapExceededPayload {
  context: &'static str,
  cap_name: &'static str,
  cap: u64,
  observed: u64,
}

impl CapExceededPayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, cap_name: &'static str, cap: u64, observed: u64) -> Self {
    Self {
      context,
      cap_name,
      cap,
      observed,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The name of the cap that was exceeded.
  #[inline(always)]
  pub const fn cap_name(&self) -> &'static str {
    self.cap_name
  }
  /// The cap value.
  #[inline(always)]
  pub const fn cap(&self) -> u64 {
    self.cap
  }
  /// The observed value that exceeded the cap.
  #[inline(always)]
  pub const fn observed(&self) -> u64 {
    self.observed
  }
}

impl std::fmt::Display for CapExceededPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: observed {} exceeds cap {} ({})",
      self.context, self.observed, self.cap_name, self.cap
    )
  }
}

impl std::error::Error for CapExceededPayload {}

/// Payload for [`Error::ShapePairMismatch`]: two full shapes disagree
/// (e.g. `expected [B, S, D]`, `actual [B, T, D]`). Distinct from
/// [`Error::RankMismatch`] (rank differs) and [`Error::LengthMismatch`]
/// (single dim differs); this variant carries BOTH complete shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapePairMismatchPayload {
  context: &'static str,
  expected: SmallVec<[usize; 4]>,
  actual: SmallVec<[usize; 4]>,
}

impl ShapePairMismatchPayload {
  /// Construct a new payload.
  pub fn new(
    context: &'static str,
    expected: impl Into<SmallVec<[usize; 4]>>,
    actual: impl Into<SmallVec<[usize; 4]>>,
  ) -> Self {
    Self {
      context,
      expected: expected.into(),
      actual: actual.into(),
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The expected full shape.
  #[inline(always)]
  pub fn expected(&self) -> &[usize] {
    &self.expected
  }
  /// The observed full shape.
  #[inline(always)]
  pub fn actual(&self) -> &[usize] {
    &self.actual
  }
}

impl std::fmt::Display for ShapePairMismatchPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "shape mismatch: {}: expected {:?}, got {:?}",
      self.context,
      self.expected.as_slice(),
      self.actual.as_slice()
    )
  }
}

impl std::error::Error for ShapePairMismatchPayload {}

/// Payload for [`Error::DivisibilityConstraint`]: `dividend` is not a
/// multiple of `divisor` (e.g. `in_features` not divisible by
/// `group_size` in AWQ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivisibilityConstraintPayload {
  context: &'static str,
  name_dividend: &'static str,
  name_divisor: &'static str,
  dividend: u64,
  divisor: u64,
}

impl DivisibilityConstraintPayload {
  /// Construct a new payload.
  pub fn new(
    context: &'static str,
    name_dividend: &'static str,
    dividend: u64,
    name_divisor: &'static str,
    divisor: u64,
  ) -> Self {
    Self {
      context,
      name_dividend,
      name_divisor,
      dividend,
      divisor,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// Name of the dividend operand.
  #[inline(always)]
  pub const fn name_dividend(&self) -> &'static str {
    self.name_dividend
  }
  /// Name of the divisor operand.
  #[inline(always)]
  pub const fn name_divisor(&self) -> &'static str {
    self.name_divisor
  }
  /// Dividend value.
  #[inline(always)]
  pub const fn dividend(&self) -> u64 {
    self.dividend
  }
  /// Divisor value.
  #[inline(always)]
  pub const fn divisor(&self) -> u64 {
    self.divisor
  }
}

impl std::fmt::Display for DivisibilityConstraintPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: {} ({}) must be divisible by {} ({})",
      self.context, self.name_dividend, self.dividend, self.name_divisor, self.divisor
    )
  }
}

impl std::error::Error for DivisibilityConstraintPayload {}

/// Payload for [`Error::UnsupportedDtype`]: a dtype gate rejected an
/// input dtype that's not in the allowed set (e.g. `Adam: weights must
/// be floating, got Int32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedDtypePayload {
  context: &'static str,
  dtype: Dtype,
  supported: &'static [Dtype],
}

impl UnsupportedDtypePayload {
  /// Construct a new payload.
  pub const fn new(context: &'static str, dtype: Dtype, supported: &'static [Dtype]) -> Self {
    Self {
      context,
      dtype,
      supported,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The dtype that was rejected.
  #[inline(always)]
  pub const fn dtype(&self) -> Dtype {
    self.dtype
  }
  /// The static list of supported dtypes.
  #[inline(always)]
  pub const fn supported(&self) -> &'static [Dtype] {
    self.supported
  }
}

impl std::fmt::Display for UnsupportedDtypePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: unsupported dtype {:?} (supported: {:?})",
      self.context, self.dtype, self.supported
    )
  }
}

impl std::error::Error for UnsupportedDtypePayload {}

/// Payload for [`Error::AllocFailure`]: a `try_reserve` / `try_reserve_exact`
/// failed (request-scaled allocation that the OOM guard turned into a
/// recoverable error rather than `Vec::with_capacity`'s abort).
#[derive(Debug)]
pub struct AllocFailurePayload {
  context: &'static str,
  item: &'static str,
  count: u64,
  inner: TryReserveError,
}

impl AllocFailurePayload {
  /// Construct a new payload.
  pub fn new(
    context: &'static str,
    item: &'static str,
    count: u64,
    inner: TryReserveError,
  ) -> Self {
    Self {
      context,
      item,
      count,
      inner,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The kind of item being reserved (e.g. `"samples"`, `"f32 elements"`).
  #[inline(always)]
  pub const fn item(&self) -> &'static str {
    self.item
  }
  /// The number of items the allocator could not satisfy.
  #[inline(always)]
  pub const fn count(&self) -> u64 {
    self.count
  }
  /// The underlying allocator error.
  #[inline(always)]
  pub fn inner(&self) -> &TryReserveError {
    &self.inner
  }
}

impl std::fmt::Display for AllocFailurePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: reservation for {} {} failed: {}",
      self.context, self.count, self.item, self.inner
    )
  }
}

impl std::error::Error for AllocFailurePayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(&self.inner)
  }
}

/// Payload for [`Error::Parse`]: an external parser (JSON, regex,
/// tokenizer.json, GGUF metadata) failed.
#[derive(Debug)]
pub struct ParsePayload {
  context: &'static str,
  input_kind: &'static str,
  inner: Box<dyn std::error::Error + Send + Sync>,
}

impl ParsePayload {
  /// Construct a new payload.
  pub fn new(
    context: &'static str,
    input_kind: &'static str,
    inner: impl Into<Box<dyn std::error::Error + Send + Sync>>,
  ) -> Self {
    Self {
      context,
      input_kind,
      inner: inner.into(),
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The kind of input being parsed (e.g. `"JSON"`, `"tokenizer.json"`).
  #[inline(always)]
  pub const fn input_kind(&self) -> &'static str {
    self.input_kind
  }
  /// The underlying parser error.
  pub fn inner(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
    self.inner.as_ref()
  }
}

impl std::fmt::Display for ParsePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: parse {} failed: {}",
      self.context, self.input_kind, self.inner
    )
  }
}

impl std::error::Error for ParsePayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(self.inner.as_ref())
  }
}

/// Payload for [`Error::ExternalOp`]: an external-library runtime /
/// device-backend operation failed (e.g. `cpal` audio device backend's
/// `build_output_stream` / `play` / `pause`, `image` crate's decode, a
/// future GPU backend's command-queue submit).
///
/// **Distinct from [`Error::Parse`]** (which is for external PARSERS:
/// JSON, regex, tokenizer.json), [`Error::MlxOp`] / [`Error::MlxC`]
/// (which are the mlx-c C++ boundary), and [`Error::FileIo`] (which is
/// for `std::fs` operations). This variant is for external libraries
/// whose failures are device / runtime / capability errors, not
/// parser errors.
#[derive(Debug)]
pub struct ExternalOpPayload {
  context: &'static str,
  op_kind: &'static str,
  inner: Box<dyn std::error::Error + Send + Sync>,
}

impl ExternalOpPayload {
  /// Construct a new payload.
  pub fn new(
    context: &'static str,
    op_kind: &'static str,
    inner: impl Into<Box<dyn std::error::Error + Send + Sync>>,
  ) -> Self {
    Self {
      context,
      op_kind,
      inner: inner.into(),
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The external operation kind (e.g. `"cpal stream"`, `"image decode"`).
  #[inline(always)]
  pub const fn op_kind(&self) -> &'static str {
    self.op_kind
  }
  /// The underlying library error.
  pub fn inner(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
    self.inner.as_ref()
  }
}

impl std::fmt::Display for ExternalOpPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: external {} failed: {}",
      self.context, self.op_kind, self.inner
    )
  }
}

impl std::error::Error for ExternalOpPayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(self.inner.as_ref())
  }
}

/// Payload for [`Error::BoundedDecode`]: a decoder produced more
/// elements than the documented cap (e.g. truncated/malicious audio
/// stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedDecodePayload {
  context: &'static str,
  cap: u64,
  observed: u64,
}

impl BoundedDecodePayload {
  /// Construct a new payload.
  pub fn new(context: &'static str, cap: u64, observed: u64) -> Self {
    Self {
      context,
      cap,
      observed,
    }
  }
  /// Call-site label.
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }
  /// The cap value.
  #[inline(always)]
  pub const fn cap(&self) -> u64 {
    self.cap
  }
  /// The observed element count.
  #[inline(always)]
  pub const fn observed(&self) -> u64 {
    self.observed
  }
}

impl std::fmt::Display for BoundedDecodePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}: decoder produced {} elements (cap {})",
      self.context, self.observed, self.cap
    )
  }
}

impl std::error::Error for BoundedDecodePayload {}

/// Payload for [`Error::LayerKeyed`]: a typed inner error from a
/// specific named layer. Allows wrapping any typed sub-error with a
/// runtime layer identifier without losing the inner variant.
///
/// `Box<Error>` breaks the recursive size cycle.
#[derive(Debug)]
pub struct LayerKeyedPayload {
  layer: SmolStr,
  inner: Box<Error>,
}

impl LayerKeyedPayload {
  /// Construct a new payload.
  pub fn new(layer: impl Into<SmolStr>, inner: Error) -> Self {
    Self {
      layer: layer.into(),
      inner: Box::new(inner),
    }
  }
  /// The runtime layer identifier.
  #[inline(always)]
  pub fn layer(&self) -> &str {
    &self.layer
  }
  /// The wrapped typed sub-error.
  #[inline(always)]
  pub fn inner(&self) -> &Error {
    self.inner.as_ref()
  }
}

impl std::fmt::Display for LayerKeyedPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "layer `{}`: {}", self.layer, self.inner)
  }
}

impl std::error::Error for LayerKeyedPayload {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    Some(self.inner.as_ref() as &(dyn std::error::Error + 'static))
  }
}

/// Payload for [`Error::MalformedData`]: a serialized / structured input
/// is malformed, truncated, or in an unsupported shape (e.g. a corrupt
/// SentencePiece protobuf field, a `tokenizer.json` `model.vocab` entry
/// with the wrong arity / element type, or a fine-tuning jsonl record
/// that matches none of the supported dataset formats).
///
/// **Distinct from [`Error::Parse`]** (which wraps an inner `std::error`
/// from an external parser like serde_json) — `MalformedData` is for
/// structural violations detected by mlxrs's OWN hand-rolled readers /
/// validators, where there is no inner library error to carry, only a
/// static call-site context and a static description of what was wrong.
///
/// Construct via [`MalformedDataPayload::new`]; access fields via
/// `context()` and `detail()`. Both fields are `&'static str` (no
/// `format!`) — the §5 typed-error rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalformedDataPayload {
  context: &'static str,
  detail: &'static str,
}

impl MalformedDataPayload {
  /// Construct a new payload.
  pub const fn new(context: &'static str, detail: &'static str) -> Self {
    Self { context, detail }
  }

  /// The call-site label identifying the reader / format
  /// (e.g. `"SentencePiece protobuf"`, `"SentencePieceTokenizer: model.vocab"`).
  #[inline(always)]
  pub const fn context(&self) -> &'static str {
    self.context
  }

  /// A static description of how the data was malformed
  /// (e.g. `"truncated length-delimited field"`, `"entry must be a [token, score] pair"`).
  #[inline(always)]
  pub const fn detail(&self) -> &'static str {
    self.detail
  }
}

impl std::fmt::Display for MalformedDataPayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "malformed data: {}: {}", self.context, self.detail)
  }
}

impl std::error::Error for MalformedDataPayload {}

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
    // duration of this error-handler callback; the owned copy escapes via
    // [`SmolStr`] (inline-stored for the typical short mlx-c message).
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
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
        // Try to extract a typed `[op_name]` prefix into the typed
        // [`Error::MlxOp`] variant; fall back to [`Error::MlxC`] (raw)
        // when the message has no parseable prefix. Both variants carry
        // the original message via [`SmolStr`] (inline-stored for short
        // strings, no heap alloc).
        let payload: SmolStr = s.as_ref().into();
        *g = Some(match MlxOpKind::parse_prefix(&payload) {
          Some(op) => Error::MlxOp(MlxOpPayload::new(op, payload)),
          None => Error::MlxC(payload),
        });
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
    Err(LAST.with(|c| c.borrow_mut().take()).unwrap_or_else(|| {
      // No handler-set message — emit the rare "rc-set-but-no-handler-msg"
      // case as raw `MlxC` (the genuine catch-all for an unparseable
      // boundary signal). Use [`smol_str::format_smolstr`] so the small
      // rc-only message stores inline without a heap allocation.
      Error::MlxC(smol_str::format_smolstr!(
        "mlx returned {rc} with no message"
      ))
    }))
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
        .unwrap_or_else(|| Error::MlxC(SmolStr::new_static("mlx returned null handle"))),
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
      LAST.with(|c| c.borrow_mut().take()).unwrap_or_else(|| {
        Error::MlxC(SmolStr::new_static("mlx returned null vector_array handle"))
      }),
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

    // mlx C++ emits `[reshape] Cannot reshape …` — the handler's
    // [`MlxOpKind::parse_prefix`] should map that to `MlxOpKind::Reshape`
    // and emit [`Error::MlxOp`]. Anything else means either: (a) the
    // handler is misinstalled and mlx-c aborted the process, (b) the
    // prefix parser is missing a known op, or (c) mlx upstream changed
    // their message prefix format (a regression worth catching).
    assert!(
      matches!(
        &r,
        Err(crate::Error::MlxOp(p)) if matches!(p.op(), crate::error::MlxOpKind::Reshape)
      ),
      "failing reshape did not surface as MlxOp(Reshape); \
       got: {r:?}"
    );
  }

  #[test]
  fn mlx_op_kind_parses_real_vendor_prefixes() {
    use super::MlxOpKind;
    // The enumerated cases are extracted verbatim from real mlx C++ throw
    // sites in `mlxrs-sys/vendor/mlx/mlx/*.cpp`. If upstream mlx adds a
    // new primitive whose prefix doesn't match, the boundary still
    // surfaces it as `MlxOpKind::Other(prefix)` rather than a free-form
    // string — but the migration follow-ups depend on the COMMON prefixes
    // landing in their typed buckets.
    let cases: &[(&str, MlxOpKind)] = &[
      // matmul-family — both `[matmul]` bare and CapCase class forms
      ("[matmul]", MlxOpKind::Matmul),
      ("[addmm]", MlxOpKind::Matmul),
      ("[block_masked_mm]", MlxOpKind::Matmul),
      ("[BlockMaskedMM]", MlxOpKind::Matmul),
      ("[gather_mm]", MlxOpKind::Matmul),
      ("[GatherMM]", MlxOpKind::Matmul),
      ("[gather_qmm]", MlxOpKind::Matmul),
      ("[GatherQMM::vjp]", MlxOpKind::Matmul),
      ("[QuantizedMatmul::vjp]", MlxOpKind::Matmul),
      ("[QuantizedMatmul::jvp]", MlxOpKind::Matmul),
      ("[QuantizedMatmul::vmap]", MlxOpKind::Matmul),
      ("[qqmm]", MlxOpKind::Matmul),
      ("[segmented_mm]", MlxOpKind::Matmul),
      ("[inner]", MlxOpKind::Matmul),
      ("[tensordot]", MlxOpKind::Matmul),
      ("[kron]", MlxOpKind::Matmul),
      // reshape-family
      ("[reshape]", MlxOpKind::Reshape),
      ("[unflatten]", MlxOpKind::Reshape),
      ("[Unflatten]", MlxOpKind::Reshape),
      ("[flatten]", MlxOpKind::Reshape),
      ("[expand_dims]", MlxOpKind::Reshape),
      ("[squeeze]", MlxOpKind::Reshape),
      ("[transpose]", MlxOpKind::Reshape),
      ("[swapaxes]", MlxOpKind::Reshape),
      ("[moveaxis]", MlxOpKind::Reshape),
      ("[view]", MlxOpKind::Reshape),
      // broadcast-family
      ("[broadcast_shapes]", MlxOpKind::Broadcast),
      ("[broadcast_arrays]", MlxOpKind::Broadcast),
      ("[Broadcast]", MlxOpKind::Broadcast),
      // slice-family
      ("[slice]", MlxOpKind::Slice),
      ("[slice_update]", MlxOpKind::Slice),
      ("[SliceUpdate]", MlxOpKind::Slice),
      ("[DynamicSlice::vjp]", MlxOpKind::Slice),
      ("[DynamicSlice::vmap]", MlxOpKind::Slice),
      ("[DynamicSliceUpdate::vjp]", MlxOpKind::Slice),
      ("[DynamicSliceUpdate::vmap]", MlxOpKind::Slice),
      ("[split]", MlxOpKind::Slice),
      ("[trace]", MlxOpKind::Slice),
      ("[diag]", MlxOpKind::Slice),
      ("[diagonal]", MlxOpKind::Slice),
      ("[tril]", MlxOpKind::Slice),
      ("[triu]", MlxOpKind::Slice),
      // concat-family
      ("[concatenate]", MlxOpKind::Concat),
      ("[stack]", MlxOpKind::Concat),
      ("[repeat]", MlxOpKind::Concat),
      ("[meshgrid]", MlxOpKind::Concat),
      // gather/scatter/take
      ("[gather]", MlxOpKind::Gather),
      ("[Gather]", MlxOpKind::Gather),
      ("[gather_axis]", MlxOpKind::Gather),
      ("[scatter]", MlxOpKind::Scatter),
      ("[scatter_axis]", MlxOpKind::Scatter),
      ("[scatter_add_axis]", MlxOpKind::Scatter),
      ("[masked_scatter]", MlxOpKind::Scatter),
      ("[put_along_axis]", MlxOpKind::Scatter),
      ("[take]", MlxOpKind::Take),
      ("[take_along_axis]", MlxOpKind::Take),
      // fft-family
      ("[fftn]", MlxOpKind::Fft),
      ("[fftfreq]", MlxOpKind::Fft),
      ("[rfftfreq]", MlxOpKind::Fft),
      ("[fftshift]", MlxOpKind::Fft),
      ("[ifftshift]", MlxOpKind::Fft),
      ("[hadamard_transform]", MlxOpKind::Fft),
      // quantize / dequantize
      ("[quantize]", MlxOpKind::Quantize),
      ("[quantized_matmul]", MlxOpKind::Matmul), // matmul-aliased
      ("[from_fp8]", MlxOpKind::Quantize),
      ("[to_fp8]", MlxOpKind::Quantize),
      ("[dequantize]", MlxOpKind::Dequantize),
      // conv
      ("[conv]", MlxOpKind::Conv),
      // reductions
      ("[sum]", MlxOpKind::Pool),
      ("[max]", MlxOpKind::Pool),
      ("[min]", MlxOpKind::Pool),
      ("[mean]", MlxOpKind::Pool),
      ("[prod]", MlxOpKind::Pool),
      ("[median]", MlxOpKind::Pool),
      ("[logsumexp]", MlxOpKind::Pool),
      ("[logcumsumexp]", MlxOpKind::Pool),
      ("[cumsum]", MlxOpKind::Pool),
      ("[cumprod]", MlxOpKind::Pool),
      ("[cummax]", MlxOpKind::Pool),
      ("[cummin]", MlxOpKind::Pool),
      ("[number_of_elements]", MlxOpKind::Pool),
      ("[softmax]", MlxOpKind::Pool),
      ("[topk]", MlxOpKind::Pool),
      // eval / compiled
      ("[eval]", MlxOpKind::Eval),
      ("[async_eval]", MlxOpKind::Eval),
      ("[Compiled]", MlxOpKind::Eval),
      ("[Primitive::vjp]", MlxOpKind::Eval),
      ("[Primitive::jvp]", MlxOpKind::Eval),
      ("[Primitive::vmap]", MlxOpKind::Eval),
      ("[Primitive::output_shapes]", MlxOpKind::Eval),
      // sort / argsort
      ("[sort]", MlxOpKind::Sort),
      ("[partition]", MlxOpKind::Sort),
      ("[argsort]", MlxOpKind::ArgSort),
      ("[argpartition]", MlxOpKind::ArgSort),
      ("[argmax]", MlxOpKind::ArgSort),
      ("[argmin]", MlxOpKind::ArgSort),
      // norm-family
      ("[layer_norm]", MlxOpKind::Norm),
      ("[rms_norm]", MlxOpKind::Norm),
      ("[rope]", MlxOpKind::Norm),
      ("[scaled_dot_product_attention]", MlxOpKind::Norm),
      ("[scale_dot_product_attention]", MlxOpKind::Norm),
      // linalg::* family
      ("[linalg::cholesky]", MlxOpKind::Linalg),
      ("[linalg::cholesky_inv]", MlxOpKind::Linalg),
      ("[linalg::cross]", MlxOpKind::Linalg),
      ("[linalg::eig]", MlxOpKind::Linalg),
      ("[linalg::eigh]", MlxOpKind::Linalg),
      ("[linalg::eigvals]", MlxOpKind::Linalg),
      ("[linalg::eigvalsh]", MlxOpKind::Linalg),
      ("[linalg::inv]", MlxOpKind::Linalg),
      ("[linalg::lu]", MlxOpKind::Linalg),
      ("[linalg::lu_factor]", MlxOpKind::Linalg),
      ("[linalg::norm]", MlxOpKind::Linalg),
      ("[linalg::pinv]", MlxOpKind::Linalg),
      ("[linalg::qr]", MlxOpKind::Linalg),
      ("[linalg::solve]", MlxOpKind::Linalg),
      ("[linalg::solve_triangular]", MlxOpKind::Linalg),
      ("[linalg::svd]", MlxOpKind::Linalg),
      // transform
      ("[grad]", MlxOpKind::Transform),
      ("[vjp]", MlxOpKind::Transform),
      ("[jvp]", MlxOpKind::Transform),
      ("[vmap]", MlxOpKind::Transform),
      // `compile` is routed to Eval because the backend-eval form
      // `[Compile::eval_cpu]` strips to the same primary identifier
      // — both surface compile-failure paths together. The pure
      // graph-setup failure is rare; eval-time compile failures dominate.
      ("[compile]", MlxOpKind::Eval),
      ("[Pad::vmap]", MlxOpKind::Transform),
      // random / builder
      ("[arange]", MlxOpKind::Random),
      ("[full]", MlxOpKind::Random),
      ("[eye]", MlxOpKind::Random),
      ("[linspace]", MlxOpKind::Random),
      ("[uniform]", MlxOpKind::Random),
      ("[normal]", MlxOpKind::Random),
      ("[trunc_normal]", MlxOpKind::Random),
      ("[bernoulli]", MlxOpKind::Random),
      ("[categorical]", MlxOpKind::Random),
      ("[multivariate_normal]", MlxOpKind::Random),
      ("[laplace]", MlxOpKind::Random),
      ("[randint]", MlxOpKind::Random),
      ("[bits]", MlxOpKind::Random),
      ("[finfo]", MlxOpKind::Random),
      ("[iinfo]", MlxOpKind::Random),
      // elementwise / one-off
      ("[astype]", MlxOpKind::Elementwise),
      ("[nan_to_num]", MlxOpKind::Elementwise),
      ("[negative]", MlxOpKind::Elementwise),
      ("[floor]", MlxOpKind::Elementwise),
      ("[bitwise_invert]", MlxOpKind::Elementwise),
      ("[divmod]", MlxOpKind::Elementwise),
      ("[roll]", MlxOpKind::Elementwise),
      // system / infra (incl. the recursive vendor-tree backend prefixes
      // `metal::*` / `Metal::*` / `gpu::*` / `Event::*` / `Fence::*`)
      ("[Event::stream]", MlxOpKind::System),
      ("[Event::Event]", MlxOpKind::System),
      ("[Event::wait]", MlxOpKind::System),
      ("[Fence::update]", MlxOpKind::System),
      ("[Fence::wait]", MlxOpKind::System),
      ("[StreamContext]", MlxOpKind::System),
      ("[ThreadPool::enqueue]", MlxOpKind::System),
      ("[set_default_device]", MlxOpKind::System),
      ("[set_default_stream]", MlxOpKind::System),
      ("[default_stream]", MlxOpKind::System),
      ("[deserialize_variant]", MlxOpKind::System),
      ("[export_function]", MlxOpKind::System),
      ("[import_function]", MlxOpKind::System),
      ("[import_function::call]", MlxOpKind::System),
      ("[gpu::eval]", MlxOpKind::System),
      ("[gpu::finalize]", MlxOpKind::System),
      ("[gpu::synchronize]", MlxOpKind::System),
      ("[metal::CommandEncoder]", MlxOpKind::System),
      ("[metal::Device]", MlxOpKind::System),
      ("[metal::device_info]", MlxOpKind::System),
      ("[metal::load_device]", MlxOpKind::System),
      ("[metal::malloc]", MlxOpKind::System),
      ("[metal::set_wired_limit]", MlxOpKind::System),
      ("[metal::start_capture]", MlxOpKind::System),
      ("[Metal::binary]", MlxOpKind::System),
      ("[Metal::compiled]", MlxOpKind::System),
      ("[Metal::copy]", MlxOpKind::System),
      ("[Metal::ternary]", MlxOpKind::System),
      ("[Metal::unary]", MlxOpKind::System),
      ("[METAL]", MlxOpKind::System),
      ("[malloc]", MlxOpKind::System),
      ("[new_stream]", MlxOpKind::System),
      ("[metal_kernel]", MlxOpKind::System),
      ("[cuda_kernel]", MlxOpKind::System),
      ("[custom_kernel]", MlxOpKind::System),
      // backend eval prefixes — the CapCase class-form variants from
      // mlxrs-sys/vendor/mlx/mlx/backend/*/*.cpp recursive throw sites
      ("[Matmul::eval_cpu]", MlxOpKind::Matmul),
      ("[QQMatmul]", MlxOpKind::Matmul),
      ("[QQMatmul::eval_gpu]", MlxOpKind::Matmul),
      ("[BlockMaskedMM::eval]", MlxOpKind::Matmul),
      ("[GatherMM::eval]", MlxOpKind::Matmul),
      ("[gemm_and_bias]", MlxOpKind::Matmul),
      ("[Gather::eval_cpu]", MlxOpKind::Gather),
      ("[Gather::eval_gpu]", MlxOpKind::Gather),
      ("[GatherAxis::eval_cpu]", MlxOpKind::Gather),
      ("[Scatter::eval_cpu]", MlxOpKind::Scatter),
      ("[Scatter::eval_gpu]", MlxOpKind::Scatter),
      ("[ScatterAxis::eval_cpu]", MlxOpKind::Scatter),
      ("[MaskedScatter::eval_cpu]", MlxOpKind::Scatter),
      ("[Sort::eval_gpu]", MlxOpKind::Sort),
      ("[Convolution::eval]", MlxOpKind::Conv),
      ("[Convolution::eval_gpu]", MlxOpKind::Conv),
      // linalg primitive class forms — both `linalg::*` namespace and
      // bare CapCase backend `[Cholesky::eval_cpu]` route to Linalg
      ("[Cholesky::eval_cpu]", MlxOpKind::Linalg),
      ("[Cholesky::eval_gpu]", MlxOpKind::Linalg),
      ("[Eig::eval_cpu]", MlxOpKind::Linalg),
      ("[Eig::eval_gpu]", MlxOpKind::Linalg),
      ("[Eigh::eval_cpu]", MlxOpKind::Linalg),
      ("[Eigh::eval_gpu]", MlxOpKind::Linalg),
      ("[Inverse::eval_cpu]", MlxOpKind::Linalg),
      ("[Inverse::eval_gpu]", MlxOpKind::Linalg),
      ("[LUF::eval_cpu]", MlxOpKind::Linalg),
      ("[LUF::eval_gpu]", MlxOpKind::Linalg),
      ("[QRF::eval_cpu]", MlxOpKind::Linalg),
      ("[QRF::eval_gpu]", MlxOpKind::Linalg),
      ("[SVD::eval_cpu]", MlxOpKind::Linalg),
      ("[SVD::eval_gpu]", MlxOpKind::Linalg),
      // eval / compile primitive class forms
      ("[Compile::eval_cpu]", MlxOpKind::Eval),
      ("[Compiled::eval_cpu]", MlxOpKind::Eval),
      ("[Copy::eval_gpu]", MlxOpKind::Eval),
      ("[NanEqual::eval_cpu]", MlxOpKind::Eval),
      // quantize fast-path namespace + backend forms
      ("[Quantize::eval_gpu]", MlxOpKind::Quantize),
      ("[fast::Quantize::eval_cpu]", MlxOpKind::Quantize),
      ("[quantize_dequantize]", MlxOpKind::Quantize),
      // random backend forms
      ("[Arange::eval_gpu]", MlxOpKind::Random),
      ("[RandomBits::eval_gpu]", MlxOpKind::Random),
      // FFT backend forms
      ("[FFT]", MlxOpKind::Fft),
      ("[hadamard]", MlxOpKind::Fft),
      // norm backend form
      ("[vjp_layer_norm]", MlxOpKind::Norm),
      // distributed-collective ops
      ("[AllGather::eval_gpu]", MlxOpKind::Distributed),
      ("[AllReduce::eval_gpu]", MlxOpKind::Distributed),
      ("[ReduceScatter]", MlxOpKind::Distributed),
      ("[ReduceScatter::eval_gpu]", MlxOpKind::Distributed),
      ("[Recv::eval_gpu]", MlxOpKind::Distributed),
      ("[Send::eval_gpu]", MlxOpKind::Distributed),
      ("[sum_scatter]", MlxOpKind::Distributed),
      ("[distributed]", MlxOpKind::Distributed),
      ("[mpi]", MlxOpKind::Distributed),
      ("[nccl]", MlxOpKind::Distributed),
      ("[jaccl]", MlxOpKind::Distributed),
      ("[ring]", MlxOpKind::Distributed),
      // IO primitives
      ("[load]", MlxOpKind::Io),
      ("[save]", MlxOpKind::Io),
      ("[load_safetensors]", MlxOpKind::Io),
      ("[save_safetensors]", MlxOpKind::Io),
      ("[load_gguf]", MlxOpKind::Io),
      ("[save_gguf]", MlxOpKind::Io),
      ("[safetensor]", MlxOpKind::Io),
      ("[read]", MlxOpKind::Io),
      ("[write]", MlxOpKind::Io),
      ("[Load::eval_gpu]", MlxOpKind::Io),
      ("[from_str]", MlxOpKind::Io),
      // elementwise backend dispatchers + bare CapCase forms
      ("[Abs]", MlxOpKind::Elementwise),
      ("[DivMod]", MlxOpKind::Elementwise),
      ("[binary_float]", MlxOpKind::Elementwise),
      ("[binary_int]", MlxOpKind::Elementwise),
      ("[unary_fp]", MlxOpKind::Elementwise),
      ("[unary_int]", MlxOpKind::Elementwise),
      ("[unary_real]", MlxOpKind::Elementwise),
      ("[extract_tensor_data]", MlxOpKind::Elementwise),
    ];
    for (vendor_prefix, expected) in cases {
      let parsed = MlxOpKind::parse_prefix(&format!("{vendor_prefix} some message"))
        .unwrap_or_else(|| {
          panic!("vendor prefix {vendor_prefix:?} should classify, not return None")
        });
      assert_eq!(
        &parsed, expected,
        "vendor prefix {vendor_prefix:?} should classify as {expected:?}, got {parsed:?}",
      );
    }
    // The `Other(SmolStr)` fallback preserves the post-stripped primary
    // name verbatim so future tooling can still see what op the upstream
    // message named.
    let other = MlxOpKind::parse_prefix("[totally_invented_op] foo");
    assert!(
      matches!(other, Some(MlxOpKind::Other(ref s)) if s == "totally_invented_op"),
      "got {other:?}",
    );
    // `::method` qualifier is stripped before fallback name preservation.
    let other_method = MlxOpKind::parse_prefix("[totally_invented_op::vjp] foo");
    assert!(
      matches!(other_method, Some(MlxOpKind::Other(ref s)) if s == "totally_invented_op"),
      "got {other_method:?}",
    );
    // No `[…]` prefix → None (boundary emits MlxC(SmolStr) instead).
    assert!(MlxOpKind::parse_prefix("plain message without prefix").is_none());
    assert!(MlxOpKind::parse_prefix("").is_none());
    assert!(MlxOpKind::parse_prefix("[unterminated bracket").is_none());

    // Adversarial: many repeated `fast::` segments must NOT blow the
    // stack. The parser strips the namespace iteratively (not via
    // recursive self-call) so any depth of nesting is safe.
    let deeply_nested = "[".to_string() + &"fast::".repeat(10_000) + "Quantize::eval_cpu]";
    assert_eq!(
      MlxOpKind::parse_prefix(&deeply_nested),
      Some(MlxOpKind::Quantize),
      "deeply-nested fast:: must reduce to the inner op without recursing",
    );
  }
}
