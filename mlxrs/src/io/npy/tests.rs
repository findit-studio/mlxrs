//! Tests for NumPy `.npy` / `.npz` load + save.
//!
//! The load oracle is the exact byte buffer: each test hand-builds the magic +
//! version + header dict + element bytes (matching what MLX `mx.save` /
//! numpy `np.save` write) and asserts `load_npy` / `load_npz` returns the
//! expected shapes, dtypes, and values — the expected values are the literals
//! written into the buffer, never produced by the code under test. A second
//! group exercises real save→load round-trips through the public save fns.

use std::collections::HashMap;

use half::{bf16, f16};

use super::*;
use crate::dtype::Dtype;

/// A fresh, writable per-test temp directory (the crate's no-`tempfile`-crate
/// convention — `temp_dir()` + pid + a process-unique counter).
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs-npy-{tag}-{}-{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

/// Hand-build a v1 `.npy` byte buffer for `descr` / `fortran_order` / `shape`
/// with the given raw little-endian element `data`. The header dict matches
/// MLX's exact text (`{'descr': '<d>', 'fortran_order': <f>, 'shape': (i, )}`)
/// including the per-dimension trailing `, `, then is padded with spaces +
/// `\n` so `magic(6) + version(2) + len(2) + dict` is a multiple of 16.
fn make_npy(descr: &str, fortran: bool, shape: &[usize], data: &[u8]) -> Vec<u8> {
  let mut dict = String::new();
  dict.push_str("{'descr': '");
  dict.push_str(descr);
  dict.push_str("', 'fortran_order': ");
  dict.push_str(if fortran { "True" } else { "False" });
  dict.push_str(", 'shape': (");
  for d in shape {
    dict.push_str(&d.to_string());
    dict.push_str(", ");
  }
  dict.push_str(")}");
  let header_len = dict.len();
  let padding = (6 + 2 + 2 + header_len + 1) % 16;
  dict.push_str(&" ".repeat(padding));
  dict.push('\n');

  let mut out = Vec::new();
  out.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
  out.push(0x01);
  out.push(0x00);
  out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
  out.extend_from_slice(dict.as_bytes());
  out.extend_from_slice(data);
  out
}

/// Hand-build a v1 `.npy` byte buffer wrapping an EXACT, caller-supplied header
/// `dict` body (no trailing `\n` — this helper appends the padding + newline),
/// then the raw `data`. Unlike [`make_npy`] this places no constraints on the
/// dict text, so a corrupt/malformed header (missing key, duplicate key,
/// trailing junk) can be exercised against the parser.
fn make_npy_raw_dict(dict: &str, data: &[u8]) -> Vec<u8> {
  let mut dict = dict.to_string();
  let header_len = dict.len();
  let padding = (6 + 2 + 2 + header_len + 1) % 16;
  dict.push_str(&" ".repeat(padding));
  dict.push('\n');

  let mut out = Vec::new();
  out.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
  out.push(0x01);
  out.push(0x00);
  out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
  out.extend_from_slice(dict.as_bytes());
  out.extend_from_slice(data);
  out
}

/// Hand-build a `.npy` byte buffer for an EXACT `(major, minor)` version,
/// choosing the header-length field width from the major exactly as the format
/// dictates (`u16` for v1, `u32` for v2/v3) and padding `magic + version +
/// len + dict + \n` to a multiple of 16. Used to exercise version validation:
/// the supported `(major, 0)` happy paths and the nonzero-minor rejections.
fn make_npy_versioned(major: u8, minor: u8, descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
  let mut dict = String::new();
  dict.push_str("{'descr': '");
  dict.push_str(descr);
  dict.push_str("', 'fortran_order': False, 'shape': (");
  for d in shape {
    dict.push_str(&d.to_string());
    dict.push_str(", ");
  }
  dict.push_str(")}");
  // v1 uses a 2-byte header-length field, v2/v3 a 4-byte one.
  let len_field = if major == 1 { 2 } else { 4 };
  let padding = (6 + 2 + len_field + dict.len() + 1) % 16;
  dict.push_str(&" ".repeat(padding));
  dict.push('\n');

  let mut out = Vec::new();
  out.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
  out.push(major);
  out.push(minor);
  if len_field == 2 {
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
  } else {
    out.extend_from_slice(&(dict.len() as u32).to_le_bytes());
  }
  out.extend_from_slice(dict.as_bytes());
  out.extend_from_slice(data);
  out
}

/// Write `bytes` to a fresh temp `.npy` file and return the path.
fn write_npy_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
  let dir = fresh_dir(tag);
  let path = dir.join("a.npy");
  std::fs::write(&path, bytes).unwrap();
  path
}

// ─────────────────────────── dtype load oracles ───────────────────────────

#[test]
fn load_npy_f32_2x2_values_shape_dtype() {
  // [[1, 2], [3, 4]] row-major f32, little-endian.
  let mut data = Vec::new();
  for v in [1.0f32, 2.0, 3.0, 4.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<f4", false, &[2, 2], &data);
  let path = write_npy_file("f32", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![2, 2]);
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn load_npy_i32_1d() {
  let mut data = Vec::new();
  for v in [10i32, -20, 30] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<i4", false, &[3], &data);
  let path = write_npy_file("i32", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![3]);
  assert_eq!(arr.dtype().unwrap(), Dtype::I32);
  assert_eq!(arr.to_vec::<i32>().unwrap(), vec![10, -20, 30]);
}

#[test]
fn load_npy_u8_1byte_descr_bar() {
  // u8 has itemsize 1 → numpy/MLX write the `|` byte-order flag.
  let data = vec![0u8, 127, 255];
  let bytes = make_npy("|u1", false, &[3], &data);
  let path = write_npy_file("u8", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::U8);
  assert_eq!(arr.to_vec::<u8>().unwrap(), vec![0, 127, 255]);
}

#[test]
fn load_npy_bool_1byte() {
  // numpy stores bool as one byte per element (0 / 1).
  let data = vec![1u8, 0, 1, 1];
  let bytes = make_npy("|b1", false, &[4], &data);
  let path = write_npy_file("bool", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::Bool);
  assert_eq!(arr.to_vec::<bool>().unwrap(), vec![true, false, true, true]);
}

#[test]
fn load_npy_f16() {
  let vals = [1.5f32, -2.25, 0.0];
  let mut data = Vec::new();
  for v in vals {
    data.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
  }
  let bytes = make_npy("<f2", false, &[3], &data);
  let path = write_npy_file("f16", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::F16);
  let got = arr.to_vec::<f16>().unwrap();
  assert_eq!(
    got,
    vec![f16::from_f32(1.5), f16::from_f32(-2.25), f16::from_f32(0.0)]
  );
}

#[test]
fn load_npy_bfloat16_v2_typestring() {
  // MLX encodes bfloat16 with the nonstandard "V2" typestring (numpy void,
  // size 2). The 3-char descr "<V2" must map to Dtype::BF16.
  let vals = [1.0f32, -0.5, 3.25];
  let mut data = Vec::new();
  for v in vals {
    data.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
  }
  let bytes = make_npy("<V2", false, &[3], &data);
  let path = write_npy_file("bf16", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::BF16);
  let got = arr.to_vec::<bf16>().unwrap();
  assert_eq!(
    got,
    vec![
      bf16::from_f32(1.0),
      bf16::from_f32(-0.5),
      bf16::from_f32(3.25)
    ]
  );
}

#[test]
fn load_npy_scalar_0d() {
  // A 0-d (scalar) array has the empty shape tuple `()`.
  let data = 42.0f32.to_le_bytes().to_vec();
  let bytes = make_npy("<f4", false, &[], &data);
  let path = write_npy_file("scalar", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.ndim(), 0);
  assert_eq!(arr.shape(), Vec::<usize>::new());
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(arr.item::<f32>().unwrap(), 42.0);
}

#[test]
fn load_npy_3d_shape() {
  // shape (2, 1, 3) row-major u32: 0..6.
  let mut data = Vec::new();
  for v in 0u32..6 {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<u4", false, &[2, 1, 3], &data);
  let path = write_npy_file("u32-3d", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![2, 1, 3]);
  assert_eq!(arr.dtype().unwrap(), Dtype::U32);
  assert_eq!(arr.to_vec::<u32>().unwrap(), (0u32..6).collect::<Vec<_>>());
}

#[test]
fn load_npy_complex64() {
  // complex64 = (re, im) f32 lanes.
  let pairs = [(1.0f32, 2.0f32), (-3.0, 4.0)];
  let mut data = Vec::new();
  for (re, im) in pairs {
    data.extend_from_slice(&re.to_le_bytes());
    data.extend_from_slice(&im.to_le_bytes());
  }
  let bytes = make_npy("<c8", false, &[2], &data);
  let path = write_npy_file("c64", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::Complex64);
  let got = arr.to_vec::<Complex64>().unwrap();
  assert_eq!(got[0].as_parts(), (1.0, 2.0));
  assert_eq!(got[1].as_parts(), (-3.0, 4.0));
}

#[test]
fn load_npy_big_endian_complex64_swaps_each_lane_in_place() {
  // A big-endian (`>c8`) complex64 file: each f32 lane is stored MSB-first and
  // the (re, im) lanes are contiguous. On a little-endian host the loader must
  // byte-swap each lane independently and KEEP the (re, im) ordering — numpy's
  // per-lane semantics. Real and imaginary are asymmetric here (1.5 vs -2.25),
  // so the correct per-lane swap and a whole-element 8-byte reversal (which
  // would also swap real and imaginary) give different results.
  let re = 1.5f32;
  let im = -2.25f32;
  let mut data = Vec::new();
  data.extend_from_slice(&re.to_be_bytes());
  data.extend_from_slice(&im.to_be_bytes());
  let bytes = make_npy(">c8", false, &[1], &data);
  let path = write_npy_file("be-c64", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::Complex64);
  let got = arr.to_vec::<Complex64>().unwrap();
  // Per-lane swap: re and im recovered in place. A whole-element reversal would
  // instead yield (im, re) = (-2.25, 1.5), which this asserts against.
  assert_eq!(got[0].as_parts(), (1.5, -2.25));
}

#[test]
fn load_npy_i64_and_u64_and_i16_u16_i8() {
  // i64
  let mut d = Vec::new();
  for v in [1i64, -2, 3] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let mut a = load_npy(&write_npy_file("i64", &make_npy("<i8", false, &[3], &d))).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I64);
  assert_eq!(a.to_vec::<i64>().unwrap(), vec![1, -2, 3]);

  // u64
  let mut d = Vec::new();
  for v in [1u64, 2, u64::MAX] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let mut a = load_npy(&write_npy_file("u64", &make_npy("<u8", false, &[3], &d))).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U64);
  assert_eq!(a.to_vec::<u64>().unwrap(), vec![1, 2, u64::MAX]);

  // i16
  let mut d = Vec::new();
  for v in [1i16, -2, 300] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let mut a = load_npy(&write_npy_file("i16", &make_npy("<i2", false, &[3], &d))).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I16);
  assert_eq!(a.to_vec::<i16>().unwrap(), vec![1, -2, 300]);

  // u16
  let mut d = Vec::new();
  for v in [1u16, 2, 65535] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let mut a = load_npy(&write_npy_file("u16", &make_npy("<u2", false, &[3], &d))).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::U16);
  assert_eq!(a.to_vec::<u16>().unwrap(), vec![1, 2, 65535]);

  // i8 (1-byte → "|i1")
  let d: Vec<u8> = [(-1i8), 2, 127].iter().map(|&v| v as u8).collect();
  let mut a = load_npy(&write_npy_file("i8", &make_npy("|i1", false, &[3], &d))).unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I8);
  assert_eq!(a.to_vec::<i8>().unwrap(), vec![-1, 2, 127]);
}

#[test]
fn load_npy_numpy_style_shape_no_trailing_space() {
  // numpy writes `(2, 2)` (no trailing `, ` after the last dim, single dim
  // `(3,)`). The parser must accept this in addition to MLX's `(2, 2, )`.
  let mut dict = String::from("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2)}");
  let padding = (6 + 2 + 2 + dict.len() + 1) % 16;
  dict.push_str(&" ".repeat(padding));
  dict.push('\n');
  let mut bytes = Vec::new();
  bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00]);
  bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
  bytes.extend_from_slice(dict.as_bytes());
  for v in [1.0f32, 2.0, 3.0, 4.0] {
    bytes.extend_from_slice(&v.to_le_bytes());
  }
  let path = write_npy_file("numpy-shape", &bytes);
  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![2, 2]);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

// ─────────────────────────── fortran order ───────────────────────────

#[test]
fn load_npy_fortran_order_transposes_to_row_major() {
  // Logical 2x3 matrix:
  //   [[1, 2, 3],
  //    [4, 5, 6]]
  // Column-major (fortran) storage is column-by-column: 1,4,2,5,3,6.
  // MLX reads the buffer into the REVERSED shape (3, 2) row-major then
  // transposes → the original (2, 3) row-major logical array.
  let col_major = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
  let mut data = Vec::new();
  for v in col_major {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<f4", true, &[2, 3], &data);
  let path = write_npy_file("fortran", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![2, 3]);
  // Row-major readout must be the logical order 1..6.
  assert_eq!(
    arr.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

// ─────────────────────────── big-endian descr ───────────────────────────

#[test]
fn load_npy_big_endian_descr_swaps() {
  // A big-endian (`>i4`) file: bytes are stored MSB-first. On a little-endian
  // host the parser must byte-swap each element.
  let vals = [1i32, 258, -1];
  let mut data = Vec::new();
  for v in vals {
    data.extend_from_slice(&v.to_be_bytes());
  }
  let bytes = make_npy(">i4", false, &[3], &data);
  let path = write_npy_file("be", &bytes);

  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::I32);
  assert_eq!(arr.to_vec::<i32>().unwrap(), vec![1, 258, -1]);
}

// ─────────────────────────── corrupt / truncated ───────────────────────────

#[test]
fn load_npy_wrong_magic_is_typed_error() {
  let path = write_npy_file("badmagic", b"\x93NUMPZ\x01\x00\x10\x00garbage");
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_too_short_is_typed_error() {
  // Fewer than 8 bytes: no panic, typed Parse error.
  let path = write_npy_file("short", &[0x93, b'N', b'U']);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_truncated_data_is_typed_error() {
  // Header declares 4 f32 (16 bytes) but only 8 bytes of data follow.
  let mut data = Vec::new();
  data.extend_from_slice(&1.0f32.to_le_bytes());
  data.extend_from_slice(&2.0f32.to_le_bytes());
  let bytes = make_npy("<f4", false, &[2, 2], &data);
  let path = write_npy_file("truncdata", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_trailing_data_byte_is_typed_error() {
  // The payload length is validated EXACTLY: a file whose element bytes EXCEED
  // the declared `shape × itemsize` (here ONE extra byte after a complete 4-f32
  // payload) is corrupt / partially-concatenated and must be rejected, not
  // decoded as a clean prefix. Build a valid (2, 2) f32 buffer, then append a
  // single trailing byte.
  let mut data = Vec::new();
  for v in [1.0f32, 2.0, 3.0, 4.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let mut bytes = make_npy("<f4", false, &[2, 2], &data);
  bytes.push(0xAB); // one trailing payload byte
  let path = write_npy_file("trailing-byte", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_truncated_header_is_typed_error() {
  // Declare a header_len longer than the bytes actually present.
  let mut bytes = Vec::new();
  bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00]);
  bytes.extend_from_slice(&9999u16.to_le_bytes());
  bytes.extend_from_slice(b"{'descr': '<f4'");
  let path = write_npy_file("trunchdr", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_unsupported_version_is_typed_error() {
  let mut bytes = Vec::new();
  // Major version 9 is not supported.
  bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 0x09, 0x00]);
  bytes.extend_from_slice(&16u16.to_le_bytes());
  bytes.extend_from_slice(b"{'descr':'<f4'}\n");
  let path = write_npy_file("badver", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_version_2_0_loads() {
  // v2.0 uses a 4-byte header-length field but is otherwise identical content.
  let mut data = Vec::new();
  for v in [1.0f32, 2.0, 3.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy_versioned(2, 0, "<f4", &[3], &data);
  let path = write_npy_file("v2-0", &bytes);
  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![3]);
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn load_npy_version_3_0_loads() {
  // v3.0 shares v2's 4-byte header-length field (it only changes the header's
  // text encoding to UTF-8); the byte-oriented dict scan accepts it.
  let mut data = Vec::new();
  for v in [4.0f32, 5.0, 6.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy_versioned(3, 0, "<f4", &[3], &data);
  let path = write_npy_file("v3-0", &bytes);
  let mut arr = load_npy(&path).unwrap();
  assert_eq!(arr.shape(), vec![3]);
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn load_npy_nonzero_minor_version_is_typed_error() {
  // A supported MAJOR with a NONZERO minor is an unsupported (future or
  // bit-corrupted) revision: it must be rejected, not loaded under semantics
  // this reader has not agreed to support. The numpy format defines only `x.0`
  // versions, so only (1, 0), (2, 0) and (3, 0) are accepted.
  let mut data = Vec::new();
  for v in [1.0f32, 2.0, 3.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  for (major, minor) in [(1u8, 255u8), (2, 1), (3, 99)] {
    let bytes = make_npy_versioned(major, minor, "<f4", &[3], &data);
    let path = write_npy_file(&format!("minor-{major}-{minor}"), &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(
      matches!(err, Error::Parse(_)),
      "version {major}.{minor} must be a typed Parse error, got {err:?}"
    );
  }
}

#[test]
fn load_npy_unsupported_descr_is_typed_error() {
  // A float128 (`<f16` → kind f, size... actually 'f' '1' then '6'): use a
  // descr the mapping rejects, e.g. an unknown kind.
  let data = vec![0u8; 16];
  // Build a valid f64 buffer, then corrupt the descr to an unsupported one of
  // the same length: overwrite the kind char with an unknown 'x'.
  let mut bytes = make_npy("<f8", false, &[2], &data);
  let pos = bytes
    .windows(3)
    .position(|w| w == b"<f8")
    .expect("descr present");
  bytes[pos + 1] = b'x';
  let path = write_npy_file("baddescr", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_non_ascii_3byte_descr_is_typed_error() {
  // A corrupt file can carry a valid-UTF-8 NON-ASCII descr whose byte length
  // is 3 (e.g. "€", U+20AC = 0xE2 0x82 0xAC) but whose interior byte indices
  // are not char boundaries. The dtype parser must return a typed Parse error,
  // never panic from slicing a `&str` at a non-char-boundary.
  let descr = "€";
  assert_eq!(descr.len(), 3, "the regression input must be 3 bytes long");
  let bytes = make_npy(descr, false, &[1], &0.0f32.to_le_bytes());
  let path = write_npy_file("descr-euro", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_non_ascii_2byte_descr_is_typed_error() {
  // A 2-byte NON-ASCII descr (e.g. "¢", U+00A2 = 0xC2 0xA2): no byte-order
  // flag, the two bytes are treated as kind+size. Must be a typed Parse error
  // (no panic, no out-of-bounds) — the non-digit size byte is rejected.
  let descr = "¢";
  assert_eq!(descr.len(), 2, "the regression input must be 2 bytes long");
  let bytes = make_npy(descr, false, &[1], &0.0f32.to_le_bytes());
  let path = write_npy_file("descr-cent", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_invalid_byte_order_flag_is_typed_error() {
  // A corrupt 3-char descr whose leading byte is not a numpy byte-order flag
  // (`<`/`>`/`|`/`=`) must be rejected, not silently treated as little-endian
  // and loaded as the trailing kind+size. `?f4` and `xf4` would otherwise have
  // been silently misread as little-endian f32.
  for descr in ["?f4", "xf4"] {
    let bytes = make_npy(descr, false, &[1], &0.0f32.to_le_bytes());
    let path = write_npy_file("badflag", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{descr:?} → {err:?}");
  }
}

#[test]
fn load_npy_bar_flag_on_multibyte_dtype_is_typed_error() {
  // The `|` flag means "byte order not applicable", valid per the numpy spec
  // ONLY for single-byte types. On a MULTI-byte dtype it is malformed (and would
  // suppress a required byte-swap), so the parser rejects it — for the float
  // `|f4`, the integer `|i4`, AND the 2-byte bfloat16 sentinel `|V2`. This is
  // stricter than MLX (which drops the flag and would map `|f4` to f32), but
  // numpy-spec-correct: MLX/numpy never WRITE `|` for a multi-byte type, so no
  // real weight file is affected.
  for descr in ["|f4", "|i4", "|V2"] {
    assert!(
      parse_descr_dtype(descr).is_err(),
      "{descr:?} must be a typed error at the dtype parser"
    );
    let bytes = make_npy(descr, false, &[3], &[0u8; 6]);
    let path = write_npy_file("barflag-multibyte", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{descr:?} → {err:?}");
  }
}

#[test]
fn load_npy_bar_flag_on_single_byte_dtype_still_loads() {
  // `|` IS correct for single-byte types (`i1`, `u1`, `b1`) — numpy/MLX write
  // exactly this flag for them. These must still parse and load identically;
  // only the multi-byte `|` cases changed.
  let i1 = parse_descr_dtype("|i1").unwrap();
  assert_eq!(i1, (Dtype::I8, ByteOrder::LittleOrNative));
  let u1 = parse_descr_dtype("|u1").unwrap();
  assert_eq!(u1, (Dtype::U8, ByteOrder::LittleOrNative));
  let b1 = parse_descr_dtype("|b1").unwrap();
  assert_eq!(b1, (Dtype::Bool, ByteOrder::LittleOrNative));

  // And a full load through each single-byte `|` descr.
  let mut a = load_npy(&write_npy_file(
    "bar-i1",
    &make_npy("|i1", false, &[3], &[1, 255, 7]),
  ))
  .unwrap();
  assert_eq!(a.dtype().unwrap(), Dtype::I8);
  assert_eq!(a.to_vec::<i8>().unwrap(), vec![1, -1, 7]);
  let mut b = load_npy(&write_npy_file(
    "bar-u1",
    &make_npy("|u1", false, &[3], &[1, 2, 255]),
  ))
  .unwrap();
  assert_eq!(b.dtype().unwrap(), Dtype::U8);
  assert_eq!(b.to_vec::<u8>().unwrap(), vec![1, 2, 255]);
  let mut c = load_npy(&write_npy_file(
    "bar-b1",
    &make_npy("|b1", false, &[3], &[0, 1, 2]),
  ))
  .unwrap();
  assert_eq!(c.dtype().unwrap(), Dtype::Bool);
  assert_eq!(c.to_vec::<bool>().unwrap(), vec![false, true, true]);
}

#[test]
fn load_npy_unrecognized_kind_is_typed_error() {
  // A valid byte-order flag but an unrecognized kind char (`z`) — MLX's
  // `switch` falls through to its `throw`; here it is a typed Parse error.
  let bytes = make_npy("<z4", false, &[1], &0.0f32.to_le_bytes());
  let path = write_npy_file("badkind", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_invalid_itemsize_is_typed_error() {
  // A recognized kind with an itemsize outside its accepted set must be
  // rejected (MLX's per-kind `switch` arms `break` → fall through to `throw`):
  // `<f3` (f ∉ {2,4,8}) and `<i7` (i ∉ {1,2,4,8}).
  for descr in ["<f3", "<i7"] {
    let bytes = make_npy(descr, false, &[1], &0.0f32.to_le_bytes());
    let path = write_npy_file("badsize", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{descr:?} → {err:?}");
  }
}

// ───────────────── corrupt / missing / duplicate / junk header ─────────────────
//
// The header dict is the Python literal `{'descr': <str>, 'fortran_order':
// <bool>, 'shape': <tuple>}`. The parser is a single CONSUMING pass that
// validates the entire `{ ... }` structure left-to-right, accounting for every
// byte: each required key must appear exactly once, be followed by a colon and a
// structurally-valid value, every value must be delimited by exactly `,` or `}`,
// and only padding may follow the closing brace. These exercise the rejection of
// a missing key, a duplicate key, a malformed value, comma-delimited junk
// BETWEEN fields, a missing opening brace, an unknown extra key, a bare token
// where a key is expected, and trailing junk — while the standard numpy/MLX
// layouts (covered above) still parse identically.

#[test]
fn load_npy_missing_shape_key_with_other_tuple_is_typed_error() {
  // The corrupt-file contract: a header with NO 'shape' key but ANOTHER
  // parenthesized field (`not_shape: (1,)`) must NOT have that stray tuple
  // silently adopted as the shape. The old last-`(` scan loaded it as [1]; the
  // key-anchored parser rejects the missing 'shape' key.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'not_shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("missing-shape", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_trailing_paren_junk_after_shape_is_typed_error() {
  // A valid shape followed by a trailing parenthesized expression must be
  // rejected — a duplicate/overriding tuple cannot silently win.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), (9, )}";
  let mut data = Vec::new();
  for v in [1.0f32, 2.0, 3.0, 4.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy_raw_dict(dict, &data);
  let path = write_npy_file("shape-trailing-junk", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_duplicate_shape_key_is_typed_error() {
  // Two 'shape' keys are ambiguous; the parser rejects the duplicate rather
  // than silently picking one.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, ), 'shape': (2, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("dup-shape", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_missing_descr_key_is_typed_error() {
  // No 'descr' key → typed error (the dtype is unknown, never guessed).
  let dict = "{'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("missing-descr", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_duplicate_descr_key_is_typed_error() {
  let dict = "{'descr': '<f4', 'descr': '<i4', 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("dup-descr", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_missing_fortran_order_key_is_typed_error() {
  // No 'fortran_order' key → typed error (column-vs-row order is never guessed).
  let dict = "{'descr': '<f4', 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("missing-fortran", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_duplicate_fortran_order_key_is_typed_error() {
  let dict = "{'descr': '<f4', 'fortran_order': False, 'fortran_order': True, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("dup-fortran", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_non_bool_fortran_order_is_typed_error() {
  // A 'fortran_order' value that is neither `True` nor `False` is rejected
  // (a prefix like `Truthy` or a number must not be accepted).
  for bad in ["Maybe", "1", "Truthy", "Falsey"] {
    let dict = format!("{{'descr': '<f4', 'fortran_order': {bad}, 'shape': (1, )}}");
    let bytes = make_npy_raw_dict(&dict, &1.0f32.to_le_bytes());
    let path = write_npy_file("nonbool-fortran", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{bad:?} → {err:?}");
  }
}

#[test]
fn load_npy_trailing_junk_after_descr_is_typed_error() {
  // After the closing quote of the descr value, the next non-space byte must be
  // a field separator `,` or the dict close `}` — a stray token is rejected.
  let dict = "{'descr': '<f4' XYZ, 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("descr-trailing-junk", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_empty_interior_dimension_is_typed_error() {
  // An empty token in the MIDDLE of the shape tuple (`(1, , 2)`) is malformed —
  // only the scalar `()` and the singleton trailing comma `(n,)` may be empty.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, , 2)}";
  let mut data = Vec::new();
  for v in [1.0f32, 2.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy_raw_dict(dict, &data);
  let path = write_npy_file("shape-empty-mid", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_parenthesized_integer_without_comma_is_typed_error() {
  // `(1)` is a parenthesized integer, NOT a tuple — Python requires the trailing
  // comma `(1,)` for a singleton. A header carrying the bare `(1)` form must be a
  // typed parse error, never silently loaded as the 1-d shape `[1]`.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1)}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("shape-no-comma", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_singleton_with_trailing_comma_still_loads() {
  // The valid singleton `(1,)` (with the mandatory trailing comma) still parses
  // to the 1-d shape `[1]` — the no-comma rejection above must not regress it.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,)}";
  let bytes = make_npy_raw_dict(dict, &7.5f32.to_le_bytes());
  let mut arr = load_npy(&write_npy_file("shape-singleton", &bytes)).unwrap();
  assert_eq!(arr.shape(), vec![1]);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![7.5]);
}

#[test]
fn load_npy_scalar_empty_tuple_still_loads() {
  // The scalar `()` (the one valid comma-less form) still parses to the 0-d
  // shape `[]`; the no-comma rejection applies only to a NON-empty single token.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ()}";
  let bytes = make_npy_raw_dict(dict, &42.0f32.to_le_bytes());
  let mut arr = load_npy(&write_npy_file("shape-scalar-empty", &bytes)).unwrap();
  assert_eq!(arr.ndim(), 0);
  assert_eq!(arr.item::<f32>().unwrap(), 42.0);
}

#[test]
fn load_npy_leading_double_and_bare_comma_shapes_are_typed_errors() {
  // A leading comma `(,1)`, a double comma `(1,,2)`, and a bare comma `(,)` are
  // all malformed tuple syntax — none is a form numpy/MLX emit — and each must be
  // a typed parse error.
  let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
  for shape in ["(,1)", "(1,,2)", "(,)"] {
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}}}");
    let bytes = make_npy_raw_dict(&dict, &data);
    let path = write_npy_file("shape-bad-comma", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{shape:?} → {err:?}");
  }
}

#[test]
fn load_npy_mlx_and_numpy_multidim_shapes_load() {
  // The two faithful multi-dimensional forms — the MLX trailing comma `(2, 2, )`
  // and the numpy `(2, 3)` — load to the expected shapes (re-confirmed alongside
  // the tightened grammar so neither regresses).
  let data4: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
  let mlx = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, )}";
  let a = load_npy(&write_npy_file(
    "shape-mlx-trailing",
    &make_npy_raw_dict(mlx, &data4),
  ))
  .unwrap();
  assert_eq!(a.shape(), vec![2, 2]);

  let data6: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
  let np = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3)}";
  let a = load_npy(&write_npy_file(
    "shape-np-2x3",
    &make_npy_raw_dict(np, &data6),
  ))
  .unwrap();
  assert_eq!(a.shape(), vec![2, 3]);
}

#[test]
fn load_npy_missing_colon_after_key_is_typed_error() {
  // A key not followed by a colon is structurally invalid.
  let dict = "{'descr' '<f4', 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("no-colon", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_standard_layouts_parse_via_raw_dict() {
  // Lock the faithful forms: both the MLX layout (trailing `, ` inside the
  // shape) and the numpy layout (no trailing space, singleton `(n,)`, scalar
  // `()`) parse to the same shape via the raw-dict path the corrupt tests use.
  let data4: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();

  // MLX layout: trailing ", " after each dim.
  let mlx = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, )}";
  let a = load_npy(&write_npy_file("std-mlx", &make_npy_raw_dict(mlx, &data4))).unwrap();
  assert_eq!(a.shape(), vec![2, 2]);
  assert_eq!(a.dtype().unwrap(), Dtype::F32);

  // numpy layout: no trailing space.
  let np = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2)}";
  let a = load_npy(&write_npy_file("std-np", &make_npy_raw_dict(np, &data4))).unwrap();
  assert_eq!(a.shape(), vec![2, 2]);

  // singleton (3,).
  let data3: Vec<u8> = [1.0f32, 2.0, 3.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
  let single = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,)}";
  let a = load_npy(&write_npy_file(
    "std-single",
    &make_npy_raw_dict(single, &data3),
  ))
  .unwrap();
  assert_eq!(a.shape(), vec![3]);

  // scalar ().
  let scalar = "{'descr': '<f4', 'fortran_order': False, 'shape': ()}";
  let mut a = load_npy(&write_npy_file(
    "std-scalar",
    &make_npy_raw_dict(scalar, &42.0f32.to_le_bytes()),
  ))
  .unwrap();
  assert_eq!(a.ndim(), 0);
  assert_eq!(a.item::<f32>().unwrap(), 42.0);
}

#[test]
fn load_npy_comma_delimited_junk_between_fields_is_typed_error() {
  // The consuming parser closes the lenient-header class by construction: a
  // header with a bare junk token comma-spliced BETWEEN two real fields is not a
  // valid Python-literal dict, so it must NOT load with attacker-controlled
  // dtype/order/shape. After a value, the next pair must begin with a quoted key
  // — a bare `@@@` (or any other non-key token) is `Error::Parse`. Both the
  // descr→fortran gap and the fortran→shape gap are exercised.
  let dicts = [
    "{'descr': '<f4', @@@, 'fortran_order': False, 'shape': (1, )}",
    "{'descr': '<f4', 'fortran_order': False, @@@, 'shape': (1, )}",
  ];
  for dict in dicts {
    let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
    let path = write_npy_file("junk-between-fields", &bytes);
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{dict:?} → {err:?}");
  }
}

#[test]
fn load_npy_missing_opening_brace_is_typed_error() {
  // A header that does not begin (after optional whitespace) with `{` is not a
  // dict at all — rejected before any field is read.
  let dict = "'descr': '<f4', 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("no-open-brace", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_trailing_junk_after_closing_brace_is_typed_error() {
  // After the closing `}` only structural padding (spaces / the `\n` numpy pads
  // with) may remain; a stray token appended after the brace is rejected so a
  // second expression cannot ride along behind a valid dict.
  let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, )} EVIL";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("trailing-after-brace", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_unknown_extra_key_is_typed_error() {
  // An unknown key alongside the three required ones is rejected — the parser
  // accepts only `descr` / `fortran_order` / `shape`.
  let dict = "{'descr': '<f4', 'evil': 1, 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("unknown-key", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npy_bare_token_where_key_expected_is_typed_error() {
  // Where a key is expected the parser requires a single-quoted token; a bare
  // (unquoted) token like `descr` without quotes is structurally invalid.
  let dict = "{descr: '<f4', 'fortran_order': False, 'shape': (1, )}";
  let bytes = make_npy_raw_dict(dict, &1.0f32.to_le_bytes());
  let path = write_npy_file("bare-key", &bytes);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

// ─────────────────────────── npz load (stored) ───────────────────────────

/// Build a `.npz` (ZIP) buffer in memory from `(name, npy_bytes)` members
/// with the given compression method, via the `zip` crate (an independent
/// path from the loader under test).
fn make_npz(members: &[(&str, Vec<u8>)], method: zip::CompressionMethod) -> Vec<u8> {
  use std::io::{Cursor, Write};
  let mut buf = Vec::new();
  {
    let mut zw = zip::ZipWriter::new(Cursor::new(&mut buf));
    let opts: zip::write::FileOptions<'_, ()> =
      zip::write::FileOptions::default().compression_method(method);
    for (name, bytes) in members {
      zw.start_file(*name, opts).unwrap();
      zw.write_all(bytes).unwrap();
    }
    zw.finish().unwrap();
  }
  buf
}

#[test]
fn load_npz_stored_two_members() {
  // weight: f32 (2,) = [1.5, 2.5]; bias: i32 (1,) = [7].
  let mut wd = Vec::new();
  for v in [1.5f32, 2.5] {
    wd.extend_from_slice(&v.to_le_bytes());
  }
  let w = make_npy("<f4", false, &[2], &wd);
  let bd = 7i32.to_le_bytes().to_vec();
  let b = make_npy("<i4", false, &[1], &bd);

  let npz = make_npz(
    &[("weight.npy", w), ("bias.npy", b)],
    zip::CompressionMethod::Stored,
  );
  let dir = fresh_dir("npz-stored");
  let path = dir.join("w.npz");
  std::fs::write(&path, &npz).unwrap();

  let mut loaded = load_npz(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  let w = loaded.get_mut("weight").unwrap();
  assert_eq!(w.shape(), vec![2]);
  assert_eq!(w.dtype().unwrap(), Dtype::F32);
  assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.5, 2.5]);
  let b = loaded.get_mut("bias").unwrap();
  assert_eq!(b.dtype().unwrap(), Dtype::I32);
  assert_eq!(b.to_vec::<i32>().unwrap(), vec![7]);
}

#[test]
fn load_npz_deflate_member() {
  // A DEFLATE-compressed member (mx.savez_compressed). Use a payload large
  // enough that deflate actually engages.
  let mut data = Vec::new();
  for v in 0u32..256 {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let m = make_npy("<u4", false, &[256], &data);
  let npz = make_npz(&[("big.npy", m)], zip::CompressionMethod::Deflated);
  let dir = fresh_dir("npz-deflate");
  let path = dir.join("c.npz");
  std::fs::write(&path, &npz).unwrap();

  let mut loaded = load_npz(&path).unwrap();
  let a = loaded.get_mut("big").unwrap();
  assert_eq!(a.shape(), vec![256]);
  assert_eq!(a.to_vec::<u32>().unwrap(), (0u32..256).collect::<Vec<_>>());
}

#[test]
fn load_npz_corrupt_archive_is_typed_error() {
  let dir = fresh_dir("npz-corrupt");
  let path = dir.join("bad.npz");
  std::fs::write(&path, b"not a zip file at all").unwrap();
  let err = load_npz(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npz_duplicate_normalized_key_is_typed_error() {
  // Two members normalize to the same key: `w.npy` strips to `w`, and a bare
  // `w` member is also `w`. Silently keeping one would replace a weight with no
  // error, so the loader rejects the ambiguous archive with `Error::Parse`.
  let mut d = Vec::new();
  for v in [1.0f32, 2.0] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let a = make_npy("<f4", false, &[2], &d);
  let b = make_npy("<f4", false, &[2], &d);
  let npz = make_npz(&[("w.npy", a), ("w", b)], zip::CompressionMethod::Stored);
  let dir = fresh_dir("npz-dup");
  let path = dir.join("dup.npz");
  std::fs::write(&path, &npz).unwrap();

  let err = load_npz(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

#[test]
fn load_npz_member_trailing_data_byte_is_typed_error() {
  // The exact-length payload check applies per npz member too: a member whose
  // `.npy` body carries ONE byte beyond its declared `shape × itemsize` is a
  // corrupt member and must fail with a typed error, not load a clean prefix.
  let mut d = Vec::new();
  for v in [1.5f32, 2.5] {
    d.extend_from_slice(&v.to_le_bytes());
  }
  let mut w = make_npy("<f4", false, &[2], &d);
  w.push(0xAB); // one trailing payload byte inside the member
  let npz = make_npz(&[("weight.npy", w)], zip::CompressionMethod::Stored);
  let dir = fresh_dir("npz-member-trailing");
  let path = dir.join("t.npz");
  std::fs::write(&path, &npz).unwrap();

  let err = load_npz(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}

// ─────────────────────────── save → load round-trips ───────────────────────────

#[test]
fn save_then_load_npy_round_trips_f32() {
  let dir = fresh_dir("rt-npy-f32");
  let path = dir.join("a.npy");
  let mut arr = Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &(2usize, 3)).unwrap();
  save_npy(&path, &mut arr).unwrap();

  let mut back = load_npy(&path).unwrap();
  assert_eq!(back.shape(), vec![2, 3]);
  assert_eq!(back.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    back.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
}

#[test]
fn save_then_load_npy_round_trips_dtypes() {
  let dir = fresh_dir("rt-npy-dtypes");

  // bf16
  let p = dir.join("bf16.npy");
  let src: Vec<bf16> = [1.0f32, -2.0, 3.5]
    .iter()
    .map(|&v| bf16::from_f32(v))
    .collect();
  let mut a = Array::from_slice::<bf16>(&src, &(3usize,)).unwrap();
  save_npy(&p, &mut a).unwrap();
  let mut b = load_npy(&p).unwrap();
  assert_eq!(b.dtype().unwrap(), Dtype::BF16);
  assert_eq!(b.to_vec::<bf16>().unwrap(), src);

  // i64
  let p = dir.join("i64.npy");
  let mut a = Array::from_slice::<i64>(&[-1, 2, 3], &(3usize,)).unwrap();
  save_npy(&p, &mut a).unwrap();
  let mut b = load_npy(&p).unwrap();
  assert_eq!(b.dtype().unwrap(), Dtype::I64);
  assert_eq!(b.to_vec::<i64>().unwrap(), vec![-1, 2, 3]);

  // bool
  let p = dir.join("bool.npy");
  let mut a = Array::from_slice::<bool>(&[true, false, true], &(3usize,)).unwrap();
  save_npy(&p, &mut a).unwrap();
  let mut b = load_npy(&p).unwrap();
  assert_eq!(b.dtype().unwrap(), Dtype::Bool);
  assert_eq!(b.to_vec::<bool>().unwrap(), vec![true, false, true]);
}

#[test]
fn save_then_load_npy_scalar() {
  let dir = fresh_dir("rt-npy-scalar");
  let path = dir.join("s.npy");
  let mut arr = Array::from_slice::<f32>(&[3.5], &Vec::<usize>::new()).unwrap();
  assert_eq!(arr.ndim(), 0);
  save_npy(&path, &mut arr).unwrap();
  let mut back = load_npy(&path).unwrap();
  assert_eq!(back.ndim(), 0);
  assert_eq!(back.item::<f32>().unwrap(), 3.5);
}

#[test]
fn save_npy_empty_array_is_typed_error() {
  let dir = fresh_dir("rt-npy-empty");
  let path = dir.join("e.npy");
  let mut arr = Array::from_slice::<f32>(&[], &(0usize,)).unwrap();
  assert_eq!(arr.size(), 0);
  let err = save_npy(&path, &mut arr).unwrap_err();
  assert!(matches!(err, Error::UnsupportedDtype(_)), "got {err:?}");
}

#[test]
fn save_then_load_npz_stored_round_trips() {
  let dir = fresh_dir("rt-npz-stored");
  let path = dir.join("w.npz");
  let mut arrays: HashMap<String, Array> = HashMap::new();
  arrays.insert(
    "layer.weight".to_string(),
    Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap(),
  );
  arrays.insert(
    "layer.bias".to_string(),
    Array::from_slice::<i32>(&[5, 6], &(2usize,)).unwrap(),
  );
  save_npz(&path, &mut arrays).unwrap();

  let mut loaded = load_npz(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  let w = loaded.get_mut("layer.weight").unwrap();
  assert_eq!(w.shape(), vec![2, 2]);
  assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
  let b = loaded.get_mut("layer.bias").unwrap();
  assert_eq!(b.to_vec::<i32>().unwrap(), vec![5, 6]);
}

#[test]
fn save_then_load_npz_compressed_round_trips() {
  let dir = fresh_dir("rt-npz-compressed");
  let path = dir.join("c.npz");
  let mut arrays: HashMap<String, Array> = HashMap::new();
  let big: Vec<f32> = (0..512).map(|i| i as f32).collect();
  arrays.insert(
    "big".to_string(),
    Array::from_slice::<f32>(&big, &(512usize,)).unwrap(),
  );
  save_npz_compressed(&path, &mut arrays).unwrap();

  let mut loaded = load_npz(&path).unwrap();
  let a = loaded.get_mut("big").unwrap();
  assert_eq!(a.shape(), vec![512]);
  assert_eq!(a.to_vec::<f32>().unwrap(), big);
}

// ─────────────────────────── descr mapping unit ───────────────────────────

#[test]
fn dtype_descr_round_trips_every_dtype() {
  // Every MLX dtype's written descr must map back to the same dtype.
  for dtype in [
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
  ] {
    let descr = dtype_to_descr(dtype);
    assert_eq!(
      parse_descr_dtype(&descr).unwrap().0,
      dtype,
      "descr {descr:?} for {dtype:?}"
    );
  }
}

#[test]
fn dtype_descr_matches_mlx_exact_strings() {
  // The exact typestrings MLX `dtype_to_array_protocol` writes (little-endian
  // host). bfloat16 → "<V2" is the notable nonstandard case.
  assert_eq!(dtype_to_descr(Dtype::Bool), "|b1");
  assert_eq!(dtype_to_descr(Dtype::U8), "|u1");
  assert_eq!(dtype_to_descr(Dtype::I8), "|i1");
  assert_eq!(dtype_to_descr(Dtype::U16), "<u2");
  assert_eq!(dtype_to_descr(Dtype::I16), "<i2");
  assert_eq!(dtype_to_descr(Dtype::U32), "<u4");
  assert_eq!(dtype_to_descr(Dtype::I32), "<i4");
  assert_eq!(dtype_to_descr(Dtype::U64), "<u8");
  assert_eq!(dtype_to_descr(Dtype::I64), "<i8");
  assert_eq!(dtype_to_descr(Dtype::F16), "<f2");
  assert_eq!(dtype_to_descr(Dtype::F32), "<f4");
  assert_eq!(dtype_to_descr(Dtype::F64), "<f8");
  assert_eq!(dtype_to_descr(Dtype::BF16), "<V2");
  assert_eq!(dtype_to_descr(Dtype::Complex64), "<c8");
}

// ─────────────────────────── mmap load path ───────────────────────────
//
// `load_npy(path)` memory-maps the file and builds the array directly from the
// mapped bytes; the common case (little-endian, C-order, aligned) takes the
// zero-copy `&[u8] -> &[T]` reinterpret in `build_typed_le_view`, and the
// transform cases (big-endian, fortran-order) keep the copying path operating
// on the mapped bytes. Every dtype-oracle test above already drives `load_npy`
// through a real temp file (hence the mmap), but these assert the mmap path and
// the in-memory `load_npy_bytes` path agree byte-for-byte for both a zero-copy
// dtype and the transform cases, and exercise the alignment fallback directly.

#[test]
fn mmap_and_in_memory_agree_f32_zero_copy() {
  // f32, little-endian, C-order → the zero-copy reinterpret path. The file
  // (mmap) load and the in-memory `load_npy_bytes` load must agree exactly.
  let mut data = Vec::new();
  for v in [1.0f32, -2.5, 3.25, 4.0, 5.5, 6.0] {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<f4", false, &[2, 3], &data);

  let mut from_mem = load_npy_bytes(&bytes).unwrap();
  let mut from_file = load_npy(&write_npy_file("mmap-f32", &bytes)).unwrap();
  assert_eq!(from_file.shape(), vec![2, 3]);
  assert_eq!(from_file.dtype().unwrap(), Dtype::F32);
  assert_eq!(
    from_file.to_vec::<f32>().unwrap(),
    vec![1.0, -2.5, 3.25, 4.0, 5.5, 6.0]
  );
  assert_eq!(
    from_file.to_vec::<f32>().unwrap(),
    from_mem.to_vec::<f32>().unwrap()
  );
}

#[test]
fn mmap_and_in_memory_agree_complex64_zero_copy() {
  // complex64 also takes the zero-copy view (its `#[repr(C)]` (re, im) f32 lanes
  // match the on-disk little-endian layout). Confirm the mmap-loaded values.
  let pairs = [(1.0f32, 2.0f32), (-3.0, 4.0), (0.5, -0.25)];
  let mut data = Vec::new();
  for (re, im) in pairs {
    data.extend_from_slice(&re.to_le_bytes());
    data.extend_from_slice(&im.to_le_bytes());
  }
  let bytes = make_npy("<c8", false, &[3], &data);
  let mut arr = load_npy(&write_npy_file("mmap-c64", &bytes)).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::Complex64);
  let got = arr.to_vec::<Complex64>().unwrap();
  assert_eq!(got[0].as_parts(), (1.0, 2.0));
  assert_eq!(got[1].as_parts(), (-3.0, 4.0));
  assert_eq!(got[2].as_parts(), (0.5, -0.25));
}

#[test]
fn mmap_and_in_memory_agree_big_endian_transform() {
  // A big-endian (`>i4`) file forces the byte-swap transform path (NOT the
  // zero-copy view). The mmap load and in-memory load must still agree.
  let vals = [1i32, 258, -1, 70000];
  let mut data = Vec::new();
  for v in vals {
    data.extend_from_slice(&v.to_be_bytes());
  }
  let bytes = make_npy(">i4", false, &[4], &data);

  let mut from_mem = load_npy_bytes(&bytes).unwrap();
  let mut from_file = load_npy(&write_npy_file("mmap-be", &bytes)).unwrap();
  assert_eq!(from_file.dtype().unwrap(), Dtype::I32);
  assert_eq!(from_file.to_vec::<i32>().unwrap(), vec![1, 258, -1, 70000]);
  assert_eq!(
    from_file.to_vec::<i32>().unwrap(),
    from_mem.to_vec::<i32>().unwrap()
  );
}

#[test]
fn mmap_and_in_memory_agree_fortran_transform() {
  // A fortran-ordered file forces the reverse-shape + transpose transform path.
  // Logical 2x3 [[1,2,3],[4,5,6]] stored column-major = 1,4,2,5,3,6.
  let col_major = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
  let mut data = Vec::new();
  for v in col_major {
    data.extend_from_slice(&v.to_le_bytes());
  }
  let bytes = make_npy("<f4", true, &[2, 3], &data);

  let mut from_mem = load_npy_bytes(&bytes).unwrap();
  let mut from_file = load_npy(&write_npy_file("mmap-fortran", &bytes)).unwrap();
  assert_eq!(from_file.shape(), vec![2, 3]);
  assert_eq!(
    from_file.to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  );
  assert_eq!(
    from_file.to_vec::<f32>().unwrap(),
    from_mem.to_vec::<f32>().unwrap()
  );
}

#[test]
fn build_typed_le_view_falls_back_when_misaligned_and_yields_correct_values() {
  // The zero-copy view is taken only when the data slice is aligned for `T`; a
  // misaligned slice must return `None` so the caller falls back to the copying
  // path. A real numpy/mmap file is always aligned (header-padded + page-aligned
  // mmap), so this fabricates a deliberately misaligned f32 sub-slice to drive
  // the fallback and confirm correctness regardless of which branch runs.
  //
  // Build a byte buffer with a 1-byte prefix, then 3 f32 values. Slicing off the
  // prefix yields a slice whose start pointer is (almost certainly) NOT 4-byte
  // aligned; assert the helper falls back, and that `build_array` over the same
  // unaligned bytes still decodes the exact values via the copy path.
  let vals = [1.5f32, -2.25, 3.75];
  let mut buf = vec![0xAAu8]; // 1-byte misaligning prefix
  for v in vals {
    buf.extend_from_slice(&v.to_le_bytes());
  }
  let data = &buf[1..]; // 12 bytes of f32 data, offset by 1 from buf's start

  // If this sub-slice happens to be aligned (rare, allocator-dependent), the
  // helper legitimately returns Some — only assert the fallback when we have
  // actually produced a misaligned pointer, so the test is not flaky.
  let misaligned = !(data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>());
  let viewed = build_typed_le_view::<f32>(data, 3, &[3]).unwrap();
  if misaligned {
    assert!(
      viewed.is_none(),
      "a misaligned data slice must fall back (None), not take the unaligned view"
    );
  }

  // Regardless of the branch, decoding the same bytes through the full builder
  // must yield the exact values — the copy fallback is value-identical.
  let header = NpyHeader {
    dtype: Dtype::F32,
    swap_endianness: false,
    fortran_order: false,
    shape: vec![3],
    data_offset: 0,
  };
  let mut arr = build_array(&header, data).unwrap();
  assert_eq!(arr.dtype().unwrap(), Dtype::F32);
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![1.5, -2.25, 3.75]);
}

#[test]
fn build_typed_le_view_aligned_takes_the_zero_copy_path() {
  // The positive case: an aligned data slice (an `[f32; N]`'s bytes are 4-byte
  // aligned) must take the view (`Some`) and yield correct values — locking in
  // that the fast path is actually exercised, not silently always falling back.
  let vals = [10.0f32, 20.0, 30.0, 40.0];
  // SAFETY: `vals` is a live `[f32; 4]`; viewing it as `4 * size_of::<f32>()`
  // bytes is sound (f32 is plain data) and the pointer is f32-aligned because
  // the array's alignment is `align_of::<f32>()`. Used only to feed the helper
  // an aligned byte slice in-test.
  let bytes = unsafe {
    std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), std::mem::size_of_val(&vals[..]))
  };
  assert!(
    (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
    "an [f32]'s bytes are f32-aligned"
  );
  let mut arr = build_typed_le_view::<f32>(bytes, 4, &[4])
    .unwrap()
    .expect("aligned data must take the zero-copy view");
  assert_eq!(arr.to_vec::<f32>().unwrap(), vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn load_npz_stored_member_mmap_zero_copy_slice_loads() {
  // A STORED npz member is loaded by slicing the mmap'd zip directly over the
  // member's `data_start()..data_start()+size()` range (no per-member copy) and
  // parsing it as `.npy`. Confirm a STORED archive round-trips through that path.
  let mut wd = Vec::new();
  for v in [1.5f32, 2.5, -3.5, 4.5] {
    wd.extend_from_slice(&v.to_le_bytes());
  }
  let w = make_npy("<f4", false, &[2, 2], &wd);
  let bd = 7i32.to_le_bytes().to_vec();
  let b = make_npy("<i4", false, &[1], &bd);
  let npz = make_npz(
    &[("weight.npy", w), ("bias.npy", b)],
    zip::CompressionMethod::Stored,
  );
  let dir = fresh_dir("npz-mmap-stored");
  let path = dir.join("w.npz");
  std::fs::write(&path, &npz).unwrap();

  let mut loaded = load_npz(&path).unwrap();
  assert_eq!(loaded.len(), 2);
  let w = loaded.get_mut("weight").unwrap();
  assert_eq!(w.shape(), vec![2, 2]);
  assert_eq!(w.to_vec::<f32>().unwrap(), vec![1.5, 2.5, -3.5, 4.5]);
  let b = loaded.get_mut("bias").unwrap();
  assert_eq!(b.to_vec::<i32>().unwrap(), vec![7]);
}

#[test]
fn load_npy_empty_file_via_mmap_is_typed_parse_error() {
  // A zero-byte file: memmapix maps it to an EMPTY slice (it does not error on
  // empty files), so the header parser sees `&[]` and returns the SAME typed
  // `Error::Parse` the read-to-end path produced — not a panic, not a FileIo
  // error from the map itself. Locks in that the mmap switch preserves the
  // short-file robustness contract at its 0-byte boundary.
  let path = write_npy_file("empty", &[]);
  let err = load_npy(&path).unwrap_err();
  assert!(matches!(err, Error::Parse(_)), "got {err:?}");
}
