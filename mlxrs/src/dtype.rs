//! `Dtype` enum + sealed `Element` trait. M1 shipped impls for `bool`, `i32`,
//! `u32`, `f32`, `half::f16`; M2a extends to every non-complex variant
//! (`u8/u16/u64/i8/i16/i64/f64/half::bf16`).

use crate::error::{Error, Result, check};

/// Element type of an `Array`. Mirrors mlx-c's `mlx_dtype` 1:1.
///
/// `#[non_exhaustive]` is intentionally NOT applied — the enum already covers
/// all 14 mlx-c dtypes; adding a new dtype upstream is rare and warrants a
/// SemVer-major bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display, derive_more::IsVariant)]
#[display("{}", self.as_str())]
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

impl Dtype {
  /// Canonical string name — matches mlx-c++'s `dtype_to_string` exactly.
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Bool => "bool",
      Self::U8 => "uint8",
      Self::U16 => "uint16",
      Self::U32 => "uint32",
      Self::U64 => "uint64",
      Self::I8 => "int8",
      Self::I16 => "int16",
      Self::I32 => "int32",
      Self::I64 => "int64",
      Self::F16 => "float16",
      Self::F32 => "float32",
      Self::F64 => "float64",
      Self::BF16 => "bfloat16",
      Self::Complex64 => "complex64",
    }
  }
}

impl std::str::FromStr for Dtype {
  type Err = crate::error::Error;

  /// Parse a canonical dtype name back into a [`Dtype`] — the inverse of
  /// [`Dtype::as_str`] / the derived [`Display`](std::fmt::Display) (audit
  /// #257). The accepted strings are exactly the ones `as_str`
  /// emits (`"bool"`, `"uint8"`, …, `"complex64"`), so
  /// `Dtype::from_str(d.as_str()) == Ok(d)` round-trips for every variant.
  /// Any other string yields a typed [`Error::UnknownEnumValue`] carrying
  /// the rejected value and the full set of accepted names.
  fn from_str(s: &str) -> Result<Self> {
    match s {
      "bool" => Ok(Self::Bool),
      "uint8" => Ok(Self::U8),
      "uint16" => Ok(Self::U16),
      "uint32" => Ok(Self::U32),
      "uint64" => Ok(Self::U64),
      "int8" => Ok(Self::I8),
      "int16" => Ok(Self::I16),
      "int32" => Ok(Self::I32),
      "int64" => Ok(Self::I64),
      "float16" => Ok(Self::F16),
      "float32" => Ok(Self::F32),
      "float64" => Ok(Self::F64),
      "bfloat16" => Ok(Self::BF16),
      "complex64" => Ok(Self::Complex64),
      _ => Err(Error::UnknownEnumValue(
        crate::error::UnknownEnumValuePayload::new(
          "Dtype",
          s,
          &[
            "bool",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "complex64",
          ],
        ),
      )),
    }
  }
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

/// A 64-bit complex value — the Rust counterpart of mlx-c's
/// `mlx_complex64_t` (`std::complex<float>`): two `f32` lanes, real and
/// imaginary.
///
/// Rust has no native complex scalar, so [`Dtype::Complex64`] arrays had no
/// safe data-extraction path (#257 M2). `Complex64` closes that: it
/// implements [`Element`], giving a safe round-trip through the generic
/// `Array` API — `Array::from_slice::<Complex64>`, `Array::item::<Complex64>`,
/// `Array::as_slice::<Complex64>`, `Array::to_vec::<Complex64>`.
///
/// `#[repr(C)]` with `re` first then `im` matches `mlx_complex64_t`
/// (`__BindgenComplex<f32> { re, im }`) byte-for-byte, which is what makes the
/// `data()` pointer-cast and the `from_slice` buffer copy sound.
///
/// No `Eq`/`Hash` — the `f32` lanes preclude them (NaN ≠ NaN), matching the
/// type-convention rule for floating-point fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
  re: f32,
  im: f32,
}

impl Complex64 {
  /// Construct from real and imaginary parts.
  #[inline(always)]
  pub const fn new(re: f32, im: f32) -> Self {
    Self { re, im }
  }

  /// The real part.
  #[inline(always)]
  pub const fn re(&self) -> f32 {
    self.re
  }

  /// The imaginary part.
  #[inline(always)]
  pub const fn im(&self) -> f32 {
    self.im
  }

  /// The `(real, imaginary)` parts as a tuple — the ergonomic extraction the
  /// audit asked for (#257 M2). Pairs with the `From<(f32, f32)>` /
  /// `From<Complex64> for (f32, f32)` conversions below.
  #[inline(always)]
  pub const fn as_parts(&self) -> (f32, f32) {
    (self.re, self.im)
  }
}

impl From<(f32, f32)> for Complex64 {
  /// `(re, im)` → `Complex64`.
  #[inline(always)]
  fn from((re, im): (f32, f32)) -> Self {
    Self::new(re, im)
  }
}

impl From<Complex64> for (f32, f32) {
  /// `Complex64` → `(re, im)`.
  #[inline(always)]
  fn from(c: Complex64) -> Self {
    (c.re, c.im)
  }
}

/// Sealed trait for types that can serve as `Array` elements.
///
/// M1 ships impls for `bool`, `i32`, `u32`, `f32`, `half::f16`. M2a adds
/// `u8`, `u16`, `u64`, `i8`, `i16`, `i64`, `f64`, `half::bf16` — covering
/// every non-complex `Dtype` variant. `f64` lives in mlx-c's CPU-only path
/// (Metal has no native f64); use sparingly.
///
/// The 14th `Dtype` variant, [`Dtype::Complex64`], is represented by the
/// crate's own [`Complex64`] value type (Rust has no native complex scalar),
/// which also implements `Element` — so `Array::{from_slice, item, as_slice,
/// to_vec}::<Complex64>()` give a safe round-trip for complex data (#257 M2).
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
impl sealed::Sealed for Complex64 {}

impl Element for bool {
  const DTYPE: Dtype = Dtype::Bool;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    let mut out: bool = false;
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_bool(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_int32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_uint32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_float32(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: the raw bindgen 16-bit float type is a plain integer newtype; an
    // all-zero bit pattern is a valid (zero) value, overwritten by the
    // following `mlx_array_item_*` call before it is read.
    let mut raw: mlxrs_sys::float16_t = unsafe { std::mem::zeroed() };
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_float16(&mut raw, arr) })?;
    // SAFETY: float16_t and half::f16 are both #[repr(transparent)] newtypes
    // around u16, identical IEEE-754 binary16 representation.
    Ok(unsafe { std::mem::transmute_copy(&raw) })
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_uint8(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_uint16(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_uint64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_int8(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_int16(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_int64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_float64(&mut out, arr) })?;
    Ok(out)
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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
    // SAFETY: the raw bindgen 16-bit float type is a plain integer newtype; an
    // all-zero bit pattern is a valid (zero) value, overwritten by the
    // following `mlx_array_item_*` call before it is read.
    let mut raw: mlxrs_sys::bfloat16_t = unsafe { std::mem::zeroed() };
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `out` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_bfloat16(&mut raw, arr) })?;
    // SAFETY: bfloat16_t and half::bf16 are both 16-bit (the former a bindgen
    // typedef of u16, the latter a #[repr(transparent)] newtype around u16).
    Ok(unsafe { std::mem::transmute_copy(&raw) })
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc).
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

// Complex64 — the crate's own value type (no native Rust complex scalar). It
// is `#[repr(C)] { re: f32, im: f32 }`, byte-identical to mlx-c's
// `mlx_complex64_t = __BindgenComplex<f32> { re, im }`, so the item out-param
// and the data pointer-cast are layout-sound. Closes #257 M2: gives complex
// arrays a safe data-extraction path through the generic `Array` accessors.
impl Element for Complex64 {
  const DTYPE: Dtype = Dtype::Complex64;
  unsafe fn item(arr: mlxrs_sys::mlx_array) -> Result<Self> {
    // `re`/`im` are plain `f32`s; the zero value is valid and is overwritten
    // by the `mlx_array_item_complex64` call below before it is read.
    let mut raw = mlxrs_sys::mlx_complex64_t { re: 0.0, im: 0.0 };
    // SAFETY: trait contract (see `Element::item` # Safety): `arr` is a valid,
    // evaluated handle whose dtype the caller verified `== Self::DTYPE`;
    // `raw` is a live stack slot; the rc is surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_array_item_complex64(&mut raw, arr) })?;
    Ok(Self::new(raw.re, raw.im))
  }
  unsafe fn data(arr: mlxrs_sys::mlx_array) -> (*const Self, usize) {
    // SAFETY: trait contract (see `Element::data` # Safety): `arr` is a valid,
    // evaluated, dtype-/contiguity-checked handle; the call returns a
    // borrowed `(ptr, len)` view into mlx's buffer (no retain, no rc). The
    // cast is sound: `Complex64` and `mlx_complex64_t` are both
    // `#[repr(C)] { re: f32, im: f32 }` with identical size/align/layout.
    unsafe {
      (
        mlxrs_sys::mlx_array_data_complex64(arr) as *const Complex64,
        mlxrs_sys::mlx_array_size(arr),
      )
    }
  }
  fn sentinel_ptr() -> *const Self {
    static V: Complex64 = Complex64::new(0.0, 0.0);
    &V
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn complex64_parts_round_trip() {
    let c = Complex64::new(1.5, -2.25);
    assert_eq!(c.re(), 1.5);
    assert_eq!(c.im(), -2.25);
    assert_eq!(c.as_parts(), (1.5, -2.25));
  }

  #[test]
  fn complex64_tuple_conversions() {
    let c: Complex64 = (3.0_f32, 4.0_f32).into();
    assert_eq!(c, Complex64::new(3.0, 4.0));
    let parts: (f32, f32) = c.into();
    assert_eq!(parts, (3.0, 4.0));
  }

  #[test]
  fn complex64_default_is_zero() {
    assert_eq!(Complex64::default(), Complex64::new(0.0, 0.0));
  }

  // M2 soundness guard: `Element::data` casts `*const mlx_complex64_t` to
  // `*const Complex64`, and `from_slice` copies a `&[Complex64]` buffer into an
  // mlx complex array. Both are sound only if the two types share layout. This
  // pins that invariant so a future field reorder / repr change fails here
  // rather than as silent UB at the FFI boundary.
  #[test]
  fn complex64_layout_matches_mlx_complex64_t() {
    use std::mem::{align_of, size_of};
    assert_eq!(
      size_of::<Complex64>(),
      size_of::<mlxrs_sys::mlx_complex64_t>(),
      "Complex64 size must match mlx_complex64_t"
    );
    assert_eq!(
      align_of::<Complex64>(),
      align_of::<mlxrs_sys::mlx_complex64_t>(),
      "Complex64 align must match mlx_complex64_t"
    );
    // Field order: a known bit pattern set through the FFI struct must read
    // back identically through Complex64 (re first, im second).
    let raw = mlxrs_sys::mlx_complex64_t { re: 7.0, im: 9.0 };
    // SAFETY: identical `#[repr(C)] { f32, f32 }` layout asserted just above.
    let as_c: Complex64 = unsafe { std::mem::transmute(raw) };
    assert_eq!(as_c, Complex64::new(7.0, 9.0));
  }

  #[test]
  fn complex64_dtype_is_complex64() {
    assert_eq!(<Complex64 as Element>::DTYPE, Dtype::Complex64);
  }

  #[test]
  fn dtype_hash_consistent_with_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Dtype::F32);
    set.insert(Dtype::Complex64);
    set.insert(Dtype::F32); // duplicate value
    assert_eq!(set.len(), 2);
    assert!(set.contains(&Dtype::F32));
    assert!(set.contains(&Dtype::Complex64));
    assert!(!set.contains(&Dtype::I32));
  }

  // `FromStr` is the inverse of `as_str` (audit #257): parsing the
  // canonical name of EVERY variant must reproduce that variant exactly, and
  // an unknown name must surface a typed `UnknownEnumValue` error.
  #[test]
  fn dtype_from_str_round_trips_every_variant() {
    use std::str::FromStr;
    const ALL: &[Dtype] = &[
      Dtype::Bool,
      Dtype::U8,
      Dtype::U16,
      Dtype::U32,
      Dtype::U64,
      Dtype::I8,
      Dtype::I16,
      Dtype::I32,
      Dtype::I64,
      Dtype::F16,
      Dtype::F32,
      Dtype::F64,
      Dtype::BF16,
      Dtype::Complex64,
    ];
    for &d in ALL {
      // `Error` has no `PartialEq`, so assert on the unwrapped `Ok` value
      // (the round-trip contract `from_str(as_str()) == Ok(d)`) rather than
      // comparing the whole `Result`.
      assert_eq!(
        Dtype::from_str(d.as_str()).expect("as_str output must parse"),
        d,
        "round-trip failed for {d:?}"
      );
    }

    let err = Dtype::from_str("not_a_dtype").unwrap_err();
    assert!(
      matches!(err, Error::UnknownEnumValue(_)),
      "unknown dtype name must yield UnknownEnumValue, got {err:?}"
    );
  }
}
