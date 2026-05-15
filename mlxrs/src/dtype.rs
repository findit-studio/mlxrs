//! `Dtype` enum + sealed `Element` trait. M1 shipped impls for `bool`, `i32`,
//! `u32`, `f32`, `half::f16`; M2a extends to every non-complex variant
//! (`u8/u16/u64/i8/i16/i64/f64/half::bf16`).

use crate::error::{Error, Result, check};

/// Element type of an `Array`. Mirrors mlx-c's `mlx_dtype` 1:1.
///
/// `#[non_exhaustive]` is intentionally NOT applied — the enum already covers
/// all 14 mlx-c dtypes; adding a new dtype upstream is rare and warrants a
/// SemVer-major bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
  /// Boolean.
  Bool,
  /// Unsigned 8-bit integer.
  U8,
  /// Unsigned 16-bit integer.
  U16,
  /// Unsigned 32-bit integer.
  U32,
  /// Unsigned 64-bit integer.
  U64,
  /// Signed 8-bit integer.
  I8,
  /// Signed 16-bit integer.
  I16,
  /// Signed 32-bit integer.
  I32,
  /// Signed 64-bit integer.
  I64,
  /// 16-bit IEEE-754 half-precision float.
  F16,
  /// 32-bit IEEE-754 single-precision float.
  F32,
  /// 64-bit IEEE-754 double-precision float (no native Apple silicon support).
  F64,
  /// Brain-float 16-bit (truncated f32).
  BF16,
  /// 64-bit complex (2× f32).
  Complex64,
}

impl TryFrom<mlxrs_sys::mlx_dtype> for Dtype {
  type Error = Error;
  fn try_from(raw: mlxrs_sys::mlx_dtype) -> Result<Self> {
    match raw {
      mlxrs_sys::mlx_dtype__MLX_BOOL => Ok(Self::Bool),
      mlxrs_sys::mlx_dtype__MLX_UINT8 => Ok(Self::U8),
      mlxrs_sys::mlx_dtype__MLX_UINT16 => Ok(Self::U16),
      mlxrs_sys::mlx_dtype__MLX_UINT32 => Ok(Self::U32),
      mlxrs_sys::mlx_dtype__MLX_UINT64 => Ok(Self::U64),
      mlxrs_sys::mlx_dtype__MLX_INT8 => Ok(Self::I8),
      mlxrs_sys::mlx_dtype__MLX_INT16 => Ok(Self::I16),
      mlxrs_sys::mlx_dtype__MLX_INT32 => Ok(Self::I32),
      mlxrs_sys::mlx_dtype__MLX_INT64 => Ok(Self::I64),
      mlxrs_sys::mlx_dtype__MLX_FLOAT16 => Ok(Self::F16),
      mlxrs_sys::mlx_dtype__MLX_FLOAT32 => Ok(Self::F32),
      mlxrs_sys::mlx_dtype__MLX_FLOAT64 => Ok(Self::F64),
      mlxrs_sys::mlx_dtype__MLX_BFLOAT16 => Ok(Self::BF16),
      mlxrs_sys::mlx_dtype__MLX_COMPLEX64 => Ok(Self::Complex64),
      other => Err(Error::UnknownDtype(other)),
    }
  }
}

impl From<Dtype> for mlxrs_sys::mlx_dtype {
  fn from(d: Dtype) -> Self {
    match d {
      Dtype::Bool => mlxrs_sys::mlx_dtype__MLX_BOOL,
      Dtype::U8 => mlxrs_sys::mlx_dtype__MLX_UINT8,
      Dtype::U16 => mlxrs_sys::mlx_dtype__MLX_UINT16,
      Dtype::U32 => mlxrs_sys::mlx_dtype__MLX_UINT32,
      Dtype::U64 => mlxrs_sys::mlx_dtype__MLX_UINT64,
      Dtype::I8 => mlxrs_sys::mlx_dtype__MLX_INT8,
      Dtype::I16 => mlxrs_sys::mlx_dtype__MLX_INT16,
      Dtype::I32 => mlxrs_sys::mlx_dtype__MLX_INT32,
      Dtype::I64 => mlxrs_sys::mlx_dtype__MLX_INT64,
      Dtype::F16 => mlxrs_sys::mlx_dtype__MLX_FLOAT16,
      Dtype::F32 => mlxrs_sys::mlx_dtype__MLX_FLOAT32,
      Dtype::F64 => mlxrs_sys::mlx_dtype__MLX_FLOAT64,
      Dtype::BF16 => mlxrs_sys::mlx_dtype__MLX_BFLOAT16,
      Dtype::Complex64 => mlxrs_sys::mlx_dtype__MLX_COMPLEX64,
    }
  }
}

/// Sealed trait for types that can serve as `Array` elements.
///
/// M1 ships impls for `bool`, `i32`, `u32`, `f32`, `half::f16`. M2a adds
/// `u8`, `u16`, `u64`, `i8`, `i16`, `i64`, `f64`, `half::bf16` — covering
/// every non-complex `Dtype` variant. `f64` lives in mlx-c's CPU-only path
/// (Metal has no native f64); use sparingly.
/// `Complex64` has no native Rust scalar — use `Array::astype` if needed.
pub trait Element: sealed::Sealed + Copy + 'static {
  /// The mlx dtype this Rust type represents.
  const DTYPE: Dtype;

  /// Item extractor for scalar arrays.
  ///
  /// # Safety
  /// `arr` must be evaluated and have dtype `DTYPE`.
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self>;

  /// Data-slice accessor.
  ///
  /// # Safety
  /// `arr` must be evaluated, have dtype `DTYPE`, AND be row-contiguous.
  /// mlx-c does not expose a contiguity predicate, so callers either do the
  /// check themselves (shape × strides — the safe layer's
  /// `array::conversion::is_row_contiguous` helper) or route through
  /// `Array::as_slice` / `Array::to_vec`, which guard.
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize);

  /// Pointer to a real `static Self` for use as an empty-data sentinel at
  /// FFI boundaries (e.g. `from_slice` with a zero-element shape). Casting a
  /// `[u8]` allocation to `*const T` is not associated with a real `T`
  /// allocation and is the same UB class the safe layer is trying to close;
  /// each impl provides a typed static so the pointer is valid for `+0`.
  fn sentinel_ptr() -> *const Self;
}

mod sealed {
  pub trait Sealed {}
}

impl sealed::Sealed for bool {}
impl sealed::Sealed for u8 {}
impl sealed::Sealed for u16 {}
impl sealed::Sealed for u32 {}
impl sealed::Sealed for u64 {}
impl sealed::Sealed for i8 {}
impl sealed::Sealed for i16 {}
impl sealed::Sealed for i32 {}
impl sealed::Sealed for i64 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for f64 {}
impl sealed::Sealed for half::f16 {}
impl sealed::Sealed for half::bf16 {}

impl Element for bool {
  const DTYPE: Dtype = Dtype::Bool;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: bool = false;
    check(unsafe { mlxrs_sys::mlx_array_item_bool(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_bool(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: bool = false;
    &V
  }
}

impl Element for i32 {
  const DTYPE: Dtype = Dtype::I32;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: i32 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_int32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_int32(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: i32 = 0;
    &V
  }
}

impl Element for u32 {
  const DTYPE: Dtype = Dtype::U32;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: u32 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_uint32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_uint32(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: u32 = 0;
    &V
  }
}

impl Element for f32 {
  const DTYPE: Dtype = Dtype::F32;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: f32 = 0.0;
    check(unsafe { mlxrs_sys::mlx_array_item_float32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_float32(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: f32 = 0.0;
    &V
  }
}

// half::f16 — bindgen exposes float16_t as a #[repr(transparent)] newtype
// `__BindgenFloat16(pub u16)`, and half::f16 is also a #[repr(transparent)]
// newtype around u16. Both are 16-bit IEEE-754 binary16 with identical layout.
// transmute_copy avoids requiring float16_t: Copy publicly.
impl Element for half::f16 {
  const DTYPE: Dtype = Dtype::F16;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut raw: mlxrs_sys::float16_t = unsafe { std::mem::zeroed() };
    check(unsafe { mlxrs_sys::mlx_array_item_float16(&mut raw, arr) })?;
    // SAFETY: float16_t and half::f16 are both #[repr(transparent)] newtypes
    // around u16, identical IEEE-754 binary16 representation.
    Ok(unsafe { std::mem::transmute_copy(&raw) })
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_float16(arr) as *const half::f16,
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: half::f16 = half::f16::ZERO;
    &V
  }
}

impl Element for u8 {
  const DTYPE: Dtype = Dtype::U8;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: u8 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_uint8(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_uint8(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: u8 = 0;
    &V
  }
}

impl Element for u16 {
  const DTYPE: Dtype = Dtype::U16;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: u16 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_uint16(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_uint16(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: u16 = 0;
    &V
  }
}

impl Element for u64 {
  const DTYPE: Dtype = Dtype::U64;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: u64 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_uint64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_uint64(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: u64 = 0;
    &V
  }
}

impl Element for i8 {
  const DTYPE: Dtype = Dtype::I8;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: i8 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_int8(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_int8(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: i8 = 0;
    &V
  }
}

impl Element for i16 {
  const DTYPE: Dtype = Dtype::I16;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: i16 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_int16(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_int16(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: i16 = 0;
    &V
  }
}

impl Element for i64 {
  const DTYPE: Dtype = Dtype::I64;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: i64 = 0;
    check(unsafe { mlxrs_sys::mlx_array_item_int64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_int64(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: i64 = 0;
    &V
  }
}

impl Element for f64 {
  const DTYPE: Dtype = Dtype::F64;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: f64 = 0.0;
    check(unsafe { mlxrs_sys::mlx_array_item_float64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_float64(arr),
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: f64 = 0.0;
    &V
  }
}

// half::bf16 — bindgen exposes bfloat16_t as a plain `u16` (not a newtype, see
// `mlxrs_sys::bfloat16_t = u16`). half::bf16 is a #[repr(transparent)] newtype
// around u16. Both are 16-bit brain-float with identical layout. transmute_copy
// matches the f16 pattern and avoids relying on bfloat16_t's internal layout
// staying a bare u16 across bindgen revs.
impl Element for half::bf16 {
  const DTYPE: Dtype = Dtype::BF16;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut raw: mlxrs_sys::bfloat16_t = unsafe { std::mem::zeroed() };
    check(unsafe { mlxrs_sys::mlx_array_item_bfloat16(&mut raw, arr) })?;
    // SAFETY: bfloat16_t and half::bf16 are both 16-bit (the former a bindgen
    // typedef of u16, the latter a #[repr(transparent)] newtype around u16).
    Ok(unsafe { std::mem::transmute_copy(&raw) })
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    unsafe {
      (
        mlxrs_sys::mlx_array_data_bfloat16(arr) as *const half::bf16,
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: half::bf16 = half::bf16::ZERO;
    &V
  }
}
