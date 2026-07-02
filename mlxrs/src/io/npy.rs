//! NumPy `.npy` / `.npz` array IO — load what `mx.save` / `mx.savez` write.
//!
//! Mirrors MLX core's own format handling (`mlx/io/load.cpp`) byte-for-byte so
//! a file produced by `mx.save`, `mx.savez`, or `mx.savez_compressed` (and,
//! equivalently, `numpy.save` / `numpy.savez`) round-trips exactly:
//!
//! - `.npy` is a single array: a 6-byte magic `\x93NUMPY`, a 1-byte major +
//!   1-byte minor version, a little-endian header length (`u16` for v1, `u32`
//!   for v2/v3), then an ASCII Python-dict header carrying `descr`,
//!   `fortran_order`, and `shape`, then the raw element bytes.
//! - `.npz` is a ZIP archive whose members are `<name>.npy` entries, either
//!   STORED (`mx.savez`) or DEFLATE-compressed (`mx.savez_compressed`). The
//!   array name is the member name with the trailing `.npy` stripped.
//!
//! The numpy `descr` typestring is mapped to an MLX [`Dtype`](crate::Dtype)
//! using the same rules as MLX's `dtype_from_array_protocol` /
//! `dtype_to_array_protocol`, including MLX's nonstandard `V2` typestring for
//! `bfloat16` (numpy has no native bfloat16). A `fortran_order` (column-major)
//! array is reordered to row-major exactly as MLX does on load (reverse the
//! shape, then transpose).
//!
//! Parsing is fully bounds-checked: a truncated or corrupt file yields a typed
//! [`Error`](crate::Error) (never a panic, never an out-of-bounds read). The
//! payload length is validated EXACTLY against the declared `shape × itemsize`,
//! so a file with missing OR trailing element bytes (a partially-concatenated or
//! corrupt weight file) is rejected rather than decoded as a prefix. There are
//! no size caps — a large-but-valid file is loaded via checked arithmetic +
//! fallible allocation, surfacing a typed error only on genuine overflow / OOM.

use std::{
  collections::HashMap,
  fs::File,
  io::{Read, Write},
  path::Path,
};

use half::{bf16, f16};

use crate::{
  array::Array,
  dtype::{Complex64, Dtype, Element},
  error::{
    AllocFailurePayload, Error, FileIoPayload, FileOp, ParsePayload, Result,
    UnsupportedDtypePayload,
  },
  ops::shape::{contiguous, transpose},
};

/// The 6-byte numpy magic prefix `\x93NUMPY` (mlx `MAGIC`, `load.cpp`).
const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// Inner parse error for the `.npy` header scanner. Carries a free-form
/// detail string; wrapped in [`Error::Parse`] (via [`ParsePayload`]) so the
/// dynamic diagnostic is preserved without a `Backend(format!)` string,
/// mirroring `embeddings::config`'s `PoolingJsonParseError` idiom.
#[derive(Debug)]
struct NpyParseError {
  detail: String,
}

impl std::fmt::Display for NpyParseError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(&self.detail)
  }
}

impl std::error::Error for NpyParseError {}

/// Build an [`Error::Parse`] for a malformed `.npy` byte stream.
fn npy_err(detail: impl Into<String>) -> Error {
  Error::Parse(ParsePayload::new(
    "io::npy",
    "npy",
    NpyParseError {
      detail: detail.into(),
    },
  ))
}

// ───────────────────────── dtype ↔ numpy descr ─────────────────────────

/// The byte order a numpy array-protocol typestring declares, parsed from its
/// leading byte-order flag. `Big` is the `>` flag; `LittleOrNative` covers `<`
/// (little), `|` (not-applicable, size-1 types) and `=` (native), plus the
/// flagless 2-char form. Used to decide whether the on-disk element bytes must
/// be reversed (MLX swaps iff the file's order is big-endian on a little-endian
/// host, i.e. `read_is_big_endian != is_big_endian()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
  Big,
  LittleOrNative,
}

impl ByteOrder {
  /// `true` iff the stored byte order differs from the host's, so each
  /// element's on-disk bytes must be reversed before interpretation — the
  /// faithful port of MLX's `swap_endianness = read_is_big_endian !=
  /// is_big_endian()`.
  const fn swaps_on_host(self) -> bool {
    matches!(self, ByteOrder::Big) != cfg!(target_endian = "big")
  }
}

/// Parse a numpy array-protocol typestring into its MLX [`Dtype`] and declared
/// [`ByteOrder`] in a single strict pass — the faithful port of MLX's
/// `dtype_from_array_protocol` (`load.cpp`), hardened so a malformed component
/// is a typed [`Error::Parse`] rather than silently misinterpreted data.
///
/// The typestring is 2 or 3 ASCII chars. When 3 chars, the first is the
/// byte-order flag and the remaining two are the kind + itemsize; when 2 chars
/// there is no byte-order flag (treated as native). MLX's special case: the
/// kind+size pair `V2` (void, size 2) denotes `bfloat16` (numpy has no native
/// bfloat16 dtype).
///
/// Every component is validated against MLX's accepted set, and the leading
/// byte-order flag is validated *before* being stripped (MLX's reader drops it
/// and treats any non-`>` byte as little-endian, silently accepting a corrupt
/// flag — this port rejects it instead): the flag must be `<`/`>`/`|`/`=`, the
/// kind must be `b`/`i`/`u`/`f`/`c` (or the `V2` bfloat16 sentinel), and the
/// itemsize must be in that kind's accepted set. The `|` flag (byte order not
/// applicable) is additionally restricted to single-byte types (`i1`/`u1`/`b1`)
/// — its only valid use per the numpy spec; `|` on any multi-byte dtype
/// (`|f4`, `|i4`, the 2-byte `|V2`, …) is rejected. This is stricter than MLX
/// (which accepts `|f4`), but numpy-spec-correct and breaks no real weight file
/// since MLX/numpy only emit `<`/`>` for multi-byte types.
fn parse_descr_dtype(descr: &str) -> Result<(Dtype, ByteOrder)> {
  // Operate on the raw bytes: the typestring is untrusted header text, so a
  // corrupt file can supply a valid-UTF-8 non-ASCII `descr` (e.g. "€", whose
  // byte length is 3 but whose interior byte indices are not char boundaries).
  // Indexing a byte slice is bounds-checked and never panics on a char
  // boundary, whereas `&str` byte-index slicing would. A well-formed numpy
  // typestring is pure ASCII, so the byte view maps every valid descr exactly.
  let descr_bytes = descr.as_bytes();
  // Mirror MLX: accept only length-2 or length-3 typestrings. When 3 chars,
  // validate the byte-order flag before taking the trailing two (kind + size);
  // when 2 chars there is no flag (native order). Keep the raw flag byte so the
  // `|` (not-applicable) flag can be validated against the resolved itemsize
  // below: numpy permits `|` only for single-byte types.
  let (order, flag, bytes) = match descr_bytes.len() {
    2 => (ByteOrder::LittleOrNative, None, descr_bytes),
    3 => {
      let flag = descr_bytes[0];
      let order = match flag {
        b'>' => ByteOrder::Big,
        // `<` little, `|` not-applicable (size-1 types), `=` native — every
        // byte-order flag numpy/MLX emit. Any other leading byte is corrupt.
        b'<' | b'|' | b'=' => ByteOrder::LittleOrNative,
        _ => {
          return Err(npy_err(format!(
            "unsupported array-protocol typestring {descr:?} (invalid byte-order flag)"
          )));
        }
      };
      (order, Some(flag), &descr_bytes[1..3])
    }
    _ => {
      return Err(npy_err(format!(
        "unsupported array-protocol typestring {descr:?} (expected 2 or 3 chars)"
      )));
    }
  };
  // The `|` flag means "byte order not applicable", which numpy permits ONLY
  // for single-byte types (`i1`, `u1`, `b1`). For any multi-byte dtype `|` is
  // malformed per the numpy array-protocol spec — and it would silently suppress
  // a byte-swap the data may require — so reject it once the dtype (hence its
  // itemsize) is resolved. This is stricter than MLX, which drops the flag and
  // accepts `|f4`; MLX itself never WRITES `|` for a multi-byte type (it emits
  // `<`/`>`), so rejecting it breaks no real MLX/numpy weight file.
  let reject_bar_on_multibyte = |dtype: Dtype| -> Result<()> {
    if flag == Some(b'|') && dtype_itemsize(dtype) > 1 {
      return Err(npy_err(format!(
        "unsupported array-protocol typestring {descr:?} (`|` byte-order flag is valid only for single-byte types)"
      )));
    }
    Ok(())
  };

  // MLX's bfloat16 sentinel: the kind+size pair "V2" (a 2-byte type, so a
  // leading `|` flag is rejected by the check above).
  if bytes == b"V2" {
    reject_bar_on_multibyte(Dtype::BF16)?;
    return Ok((Dtype::BF16, order));
  }
  let kind = bytes[0];
  // `r[1] - '0'` in MLX; reject a non-digit size byte rather than wrap.
  let size = bytes[1]
    .checked_sub(b'0')
    .filter(|s| *s <= 9)
    .ok_or_else(|| {
      npy_err(format!(
        "unsupported array-protocol typestring {descr:?} (non-digit itemsize)"
      ))
    })?;
  // Validate kind + itemsize against MLX's accepted set; an unrecognized kind
  // or an out-of-set itemsize for a recognized kind is a typed error (MLX's
  // `switch` falls through to `throw` in both cases).
  let dtype = match (kind, size) {
    (b'b', 1) => Dtype::Bool,
    (b'i', 1) => Dtype::I8,
    (b'i', 2) => Dtype::I16,
    (b'i', 4) => Dtype::I32,
    (b'i', 8) => Dtype::I64,
    (b'u', 1) => Dtype::U8,
    (b'u', 2) => Dtype::U16,
    (b'u', 4) => Dtype::U32,
    (b'u', 8) => Dtype::U64,
    (b'f', 2) => Dtype::F16,
    (b'f', 4) => Dtype::F32,
    (b'f', 8) => Dtype::F64,
    (b'c', 8) => Dtype::Complex64,
    _ => {
      return Err(npy_err(format!(
        "unsupported array-protocol typestring {descr:?}"
      )));
    }
  };
  reject_bar_on_multibyte(dtype)?;
  Ok((dtype, order))
}

/// Single-character numpy "kind" code for an MLX [`Dtype`] — the faithful
/// port of MLX's `kindof` streamed by `dtype_to_array_protocol`. `bfloat16`
/// uses `V` (void), matching MLX (which serializes it as the `V2`
/// typestring).
const fn dtype_kind(dtype: Dtype) -> u8 {
  match dtype {
    Dtype::Bool => b'b',
    Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 => b'u',
    Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 => b'i',
    Dtype::F16 | Dtype::F32 | Dtype::F64 => b'f',
    Dtype::BF16 => b'V',
    Dtype::Complex64 => b'c',
  }
}

/// On-disk itemsize (bytes) of an MLX [`Dtype`] — equals MLX's `size_of`.
const fn dtype_itemsize(dtype: Dtype) -> usize {
  match dtype {
    Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
    Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
    Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
    Dtype::U64 | Dtype::I64 | Dtype::F64 | Dtype::Complex64 => 8,
  }
}

/// Build the numpy array-protocol typestring MLX writes for a [`Dtype`] (the
/// faithful port of `dtype_to_array_protocol`): a `|` byte-order flag for
/// 1-byte types, otherwise the host byte order — which on every Apple-silicon
/// / x86 target this crate builds for is little-endian (`<`) — followed by the
/// kind char and the integer itemsize.
fn dtype_to_descr(dtype: Dtype) -> String {
  let size = dtype_itemsize(dtype);
  let order = if size > 1 {
    // Host byte order; little-endian on every target this crate supports.
    if cfg!(target_endian = "big") {
      '>'
    } else {
      '<'
    }
  } else {
    '|'
  };
  // bfloat16 serializes its "size" as 2 with kind `V` → "V2"; for every other
  // dtype the kind char + numeric itemsize already match numpy.
  format!("{order}{}{size}", dtype_kind(dtype) as char)
}

// ───────────────────────────── header parse ─────────────────────────────

/// The parsed contents of a `.npy` header dict.
struct NpyHeader {
  dtype: Dtype,
  /// `true` iff the stored byte order differs from the host's — the element
  /// bytes must be reversed per-element before interpretation.
  swap_endianness: bool,
  fortran_order: bool,
  shape: Vec<usize>,
  /// Byte offset of the raw element data (just past the header).
  data_offset: usize,
}

/// Parse the leading `.npy` header (magic + version + header dict) from
/// `bytes`, returning the decoded [`NpyHeader`]. Every field is bounds-checked
/// against the buffer length; a short or malformed stream is a typed error.
fn parse_header(bytes: &[u8]) -> Result<NpyHeader> {
  // Magic (6) + major (1) + minor (1).
  let prefix = bytes
    .get(0..8)
    .ok_or_else(|| npy_err("file shorter than the 8-byte magic + version prefix"))?;
  if prefix[0..6] != MAGIC {
    return Err(npy_err("invalid magic (not a numpy .npy stream)"));
  }
  // Validate the FULL (major, minor) version tuple before choosing the
  // header-length field width. The numpy `.npy` format defines only `x.0`
  // versions, so the supported set is exactly (1, 0), (2, 0) and (3, 0): the
  // major selects the header-length width (v1 → u16, v2/v3 → u32) and the minor
  // must be 0. MLX accepts majors 1 and 2; numpy v3 shares v2's 4-byte
  // header-length field (it only changes the header's text encoding to UTF-8, a
  // superset of the ASCII MLX writes), so v3 is accepted too for forward
  // compatibility — the dict scan below is byte-oriented and encoding-safe. A
  // nonzero minor (an unsupported future or bit-corrupted revision) or an
  // unsupported major is rejected rather than loaded under semantics this reader
  // has not agreed to support.
  let (major, minor) = (prefix[6], prefix[7]);
  let header_len_size = match (major, minor) {
    (1, 0) => 2usize,
    (2, 0) | (3, 0) => 4usize,
    _ => return Err(npy_err("unsupported .npy format version")),
  };

  // Header length: little-endian u16 (v1) or u32 (v2/v3).
  let len_start = 8;
  let len_end = len_start + header_len_size;
  let len_bytes = bytes
    .get(len_start..len_end)
    .ok_or_else(|| npy_err("truncated header-length field"))?;
  let header_len = match header_len_size {
    2 => u16::from_le_bytes([len_bytes[0], len_bytes[1]]) as usize,
    _ => u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize,
  };

  // Header dict bytes.
  let dict_start = len_end;
  let dict_end = dict_start
    .checked_add(header_len)
    .ok_or_else(|| npy_err("header-length overflow"))?;
  let dict_bytes = bytes
    .get(dict_start..dict_end)
    .ok_or_else(|| npy_err("header dict extends past end of file"))?;
  let header =
    std::str::from_utf8(dict_bytes).map_err(|_| npy_err("header dict is not valid UTF-8"))?;

  // Parse the Python-literal dict ONCE into its three required fields with a
  // single consuming parser that validates the ENTIRE `{ ... }` structure
  // left-to-right, accounting for every byte. Each `key : value` pair is read in
  // order (quoted key, colon, then the key's structural value reader), the
  // delimiter after every value must be exactly `,` or `}`, each required key
  // must appear exactly once, an unknown key is rejected, and only padding may
  // follow the closing `}`. There is no position at which junk between fields can
  // sit unvalidated, so an attacker cannot smuggle a dtype/order/shape past a
  // structurally-invalid dict.
  let HeaderFields {
    descr: dtype_descr,
    fortran_order,
    shape,
  } = parse_header_fields(header)?;
  // One strict pass for the typestring: the byte-order flag is validated (a
  // corrupt flag is a typed error, not silently treated as little-endian) and
  // yields both the dtype and the declared byte order, from which the host swap
  // is derived.
  let (dtype, byte_order) = parse_descr_dtype(&dtype_descr)?;
  let swap_endianness = byte_order.swaps_on_host();

  Ok(NpyHeader {
    dtype,
    swap_endianness,
    fortran_order,
    shape,
    data_offset: dict_end,
  })
}

/// The three required fields of a `.npy` header dict, read in order from the
/// consuming dict parser.
struct HeaderFields {
  /// The `descr` typestring (between its surrounding single quotes).
  descr: String,
  fortran_order: bool,
  shape: Vec<usize>,
}

/// Advance `i` past any spaces / tabs in `b`, returning the index of the first
/// non-whitespace byte (or `b.len()`). Pure index arithmetic — never slices a
/// `&str`, never reads out of bounds.
fn skip_ws(b: &[u8], mut i: usize) -> usize {
  while matches!(b.get(i), Some(b' ' | b'\t')) {
    i += 1;
  }
  i
}

/// Read a single-quoted token `'...'` starting at `b[i]` (which must be the
/// opening `'`), returning the unquoted byte contents and the index just past
/// the closing `'`. The contents stop at the FIRST closing quote (matching the
/// prior `descr` reader exactly). A missing opening or closing quote, or
/// non-UTF-8 contents, is a typed [`Error::Parse`]. Used for both the quoted
/// dict keys and the `descr` value, so their accepted form is identical.
fn read_single_quoted(b: &[u8], i: usize) -> Result<(String, usize)> {
  if b.get(i) != Some(&b'\'') {
    return Err(npy_err("expected a single-quoted token"));
  }
  let start = i + 1;
  let mut j = start;
  loop {
    match b.get(j) {
      Some(b'\'') => break,
      Some(_) => j += 1,
      None => return Err(npy_err("unterminated single-quoted token")),
    }
  }
  // `start..j` is the validated content range; reconstruct it as a String
  // without panicking on a corrupt non-ASCII payload (a well-formed descr/key is
  // pure ASCII, so this is exact for every valid header).
  let content = std::str::from_utf8(&b[start..j])
    .map_err(|_| npy_err("single-quoted token is not valid UTF-8"))?
    .to_string();
  Ok((content, j + 1))
}

/// Read the `fortran_order` boolean literal starting at `b[i]`: exactly `True`
/// or `False` (not merely a prefix — the byte after the keyword, if any, is left
/// for the caller's delimiter check). Returns the bool and the index just past
/// the keyword. Any other token is a typed [`Error::Parse`].
fn read_bool_at(b: &[u8], i: usize) -> Result<(bool, usize)> {
  if b.get(i..i + 4) == Some(b"True") {
    Ok((true, i + 4))
  } else if b.get(i..i + 5) == Some(b"False") {
    Ok((false, i + 5))
  } else {
    Err(npy_err(
      "malformed 'fortran_order' value (expected True/False)",
    ))
  }
}

/// Read the `shape` tuple starting at `b[i]` (which must be the opening `(`):
/// one parenthesized list of non-negative integer dimensions, returning the
/// dims and the index just past the closing `)`. The grammar matches Python
/// tuple syntax exactly, accepting precisely the forms numpy/MLX emit:
///
/// * `()` → the 0-d (scalar) shape `[]`.
/// * `(n,)` → a singleton `[n]`; a non-empty tuple REQUIRES at least one comma,
///   so the trailing comma is mandatory for a single element.
/// * `(a, b)` / the MLX trailing-comma `(a, b, )` → `[a, b]`.
///
/// A parenthesized integer with no comma (`(1)`) is NOT a tuple in Python — it
/// is a parenthesized expression — and is rejected as a typed [`Error::Parse`],
/// as is any leading/double/bare comma (`(,1)`, `(1,,2)`, `(,)`), a non-integer
/// or negative dim, and a missing `(` or `)`. Surrounding whitespace around each
/// dimension is allowed.
fn read_shape_tuple_at(b: &[u8], i: usize) -> Result<(Vec<usize>, usize)> {
  if b.get(i) != Some(&b'(') {
    return Err(npy_err("malformed 'shape' value (no opening '(')"));
  }
  let start = i + 1;
  let mut j = start;
  loop {
    match b.get(j) {
      Some(b')') => break,
      Some(_) => j += 1,
      None => return Err(npy_err("malformed 'shape' value (no closing ')')")),
    }
  }
  // `start..j` is the tuple interior. It is pure ASCII in every valid header
  // (digits, commas, spaces); validate as UTF-8 so a corrupt non-ASCII interior
  // is a typed error rather than a panic, then split on commas to recover the
  // dimension tokens.
  let inner = std::str::from_utf8(&b[start..j])
    .map_err(|_| npy_err("malformed 'shape' value (non-UTF-8 interior)"))?;
  let tokens: Vec<&str> = inner.split(',').collect();

  // A non-empty tuple must contain at least one comma. With no comma the
  // interior is a single token: an EMPTY one is the scalar `()` (shape `[]`); a
  // non-empty one is a bare parenthesized integer such as `(1)`, which Python
  // does not treat as a tuple — reject it rather than load it as a 1-d shape.
  if tokens.len() == 1 {
    return if tokens[0].trim_matches([' ', '\t']).is_empty() {
      Ok((Vec::new(), j + 1))
    } else {
      Err(npy_err(
        "malformed 'shape' value (parenthesized integer is not a tuple; expected a trailing comma)",
      ))
    };
  }

  // At least one comma is present. Each dimension token is a non-negative
  // integer with optional surrounding whitespace; the FINAL token may be empty
  // (the numpy/MLX trailing comma, e.g. `(n,)` / `(2, 2, )`). An empty token in
  // any other position is a leading or double comma (`(,1)`, `(1,,2)`, `(,)`)
  // and is malformed.
  let mut shape = Vec::with_capacity(tokens.len());
  for (k, tok) in tokens.iter().enumerate() {
    let trimmed = tok.trim_matches([' ', '\t']);
    if trimmed.is_empty() {
      if k + 1 == tokens.len() {
        continue;
      }
      return Err(npy_err("empty dimension in shape tuple"));
    }
    let dim: usize = trimmed
      .parse()
      .map_err(|_| npy_err("non-integer dimension in shape tuple"))?;
    shape.push(dim);
  }
  Ok((shape, j + 1))
}

/// Parse the Python-literal header dict `{'descr': ..., 'fortran_order': ...,
/// 'shape': ...}` into its three required fields with a single CONSUMING pass
/// that validates the entire dict structure left-to-right, accounting for every
/// byte so no junk can sit between fields unvalidated.
///
/// The parser indexes raw bytes (never slices a `&str` at an arbitrary index,
/// so a truncated / non-ASCII / corrupt header can never panic or read out of
/// bounds) and proceeds:
///
/// 1. skip leading whitespace, require an opening `{`;
/// 2. loop over `key : value` pairs — read a single-quoted key, require `:`,
///    then read the value with the key's structural reader (`descr` → a quoted
///    string, `fortran_order` → exactly `True`/`False`, `shape` → one `( ... )`
///    tuple). After each value the next non-whitespace byte must be exactly `,`
///    (more pairs follow — this also accepts the trailing comma MLX writes
///    before `}`) or `}` (the dict ends); anything else, or a bare/unquoted
///    token where a key is expected, is a typed error;
/// 3. each of `descr` / `fortran_order` / `shape` must appear EXACTLY once
///    (missing or duplicate → error); an unknown key is rejected;
/// 4. after the closing `}` only padding (whitespace / the trailing `\n` numpy
///    pads with) may remain — trailing junk is a typed error.
///
/// Every well-formed numpy/MLX header (the MLX trailing-comma `(2, 2, )` form,
/// the numpy `(2, 2)` form, the scalar `()`, the singleton `(3,)`) parses
/// identically; a malformed dict in any of these positions is [`Error::Parse`].
fn parse_header_fields(header: &str) -> Result<HeaderFields> {
  let b = header.as_bytes();
  let mut i = skip_ws(b, 0);
  // Require the opening brace.
  if b.get(i) != Some(&b'{') {
    return Err(npy_err("header dict missing opening '{'"));
  }
  i += 1;

  let mut descr: Option<String> = None;
  let mut fortran_order: Option<bool> = None;
  let mut shape: Option<Vec<usize>> = None;

  loop {
    i = skip_ws(b, i);
    match b.get(i) {
      // Empty dict or trailing comma before `}` lands here: the dict ends.
      Some(b'}') => {
        i += 1;
        break;
      }
      // A key must be a single-quoted token; anything else (a bare/unquoted
      // token, an unexpected delimiter, end of input) is structurally invalid.
      Some(b'\'') => {}
      _ => return Err(npy_err("expected a quoted key or '}' in header dict")),
    }

    let (key, after_key) = read_single_quoted(b, i)?;
    i = skip_ws(b, after_key);
    if b.get(i) != Some(&b':') {
      return Err(npy_err(format!(
        "header dict '{key}' key not followed by ':'"
      )));
    }
    i = skip_ws(b, i + 1);

    // Dispatch on the key with its structural value reader, rejecting a
    // duplicate (already-set) or unknown key.
    match key.as_str() {
      "descr" => {
        if descr.is_some() {
          return Err(npy_err("header dict has duplicate 'descr' key"));
        }
        let (v, after) = read_single_quoted(b, i)?;
        descr = Some(v);
        i = after;
      }
      "fortran_order" => {
        if fortran_order.is_some() {
          return Err(npy_err("header dict has duplicate 'fortran_order' key"));
        }
        let (v, after) = read_bool_at(b, i)?;
        fortran_order = Some(v);
        i = after;
      }
      "shape" => {
        if shape.is_some() {
          return Err(npy_err("header dict has duplicate 'shape' key"));
        }
        let (v, after) = read_shape_tuple_at(b, i)?;
        shape = Some(v);
        i = after;
      }
      _ => return Err(npy_err(format!("header dict has unknown key '{key}'"))),
    }

    // After every value the next non-whitespace byte must be a pair separator
    // `,` (more pairs follow — also the trailing comma MLX writes) or the dict
    // close `}`. Any other byte (or end of input) is stray junk.
    i = skip_ws(b, i);
    match b.get(i) {
      Some(b',') => {
        i += 1;
        continue;
      }
      Some(b'}') => {
        i += 1;
        break;
      }
      _ => return Err(npy_err(format!("trailing junk after '{key}' value"))),
    }
  }

  // After the closing `}` only structural padding may remain (numpy pads the
  // header with spaces and a trailing `\n`); any other byte is trailing junk.
  i = skip_ws(b, i);
  while matches!(b.get(i), Some(b'\n' | b'\r')) {
    i += 1;
    i = skip_ws(b, i);
  }
  if i != b.len() {
    return Err(npy_err("trailing junk after header dict"));
  }

  // Every required key must have been read exactly once (duplicates already
  // rejected above); a missing key is a typed error.
  let descr = descr.ok_or_else(|| npy_err("header dict missing 'descr' key"))?;
  let fortran_order =
    fortran_order.ok_or_else(|| npy_err("header dict missing 'fortran_order' key"))?;
  let shape = shape.ok_or_else(|| npy_err("header dict missing 'shape' key"))?;

  Ok(HeaderFields {
    descr,
    fortran_order,
    shape,
  })
}

// ───────────────────────────── data → Array ─────────────────────────────

/// Total element count of `shape` (1 for a 0-d scalar), with checked
/// multiplication so a pathological shape product surfaces a typed overflow
/// error rather than wrapping.
fn shape_numel(shape: &[usize]) -> Result<usize> {
  shape.iter().try_fold(1usize, |acc, &d| {
    acc.checked_mul(d).ok_or_else(|| {
      Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(
        "io::npy: shape product",
        "usize",
      ))
    })
  })
}

/// Reinterpret `data` (exactly `numel * size_of::<T>()` host-endian, in-order
/// element bytes) as a borrowed `&[T]` view WITHOUT copying it into an
/// intermediate `Vec<T>`, then hand that view to [`Array::from_slice`] — which
/// performs the single, unavoidable copy into the MLX array (MLX arrays own
/// their data). This is the common-case fast path: it removes the per-element
/// typed-`Vec` copy [`build_typed`] makes, taking peak load memory toward ~1×
/// the data (just the MLX array) instead of ~3× (whole-file buffer + typed Vec
/// + MLX array).
///
/// Returns `Ok(None)` when the borrow is NOT sound — the data pointer is not
/// aligned for `T` — so the caller falls back to the copying [`build_typed`]
/// path (which is byte-by-byte and alignment-agnostic). This only happens for a
/// hand-crafted / odd file: numpy pads the `.npy` header so the data start is
/// 64-byte aligned, and an mmap is page-aligned, so a real file is always
/// aligned for every `T` here (max align 8).
///
/// The caller ([`build_array`], guarded by its `can_view` flag) must ALSO have
/// established that no transform is required: little-endian file on a
/// little-endian host (`!swap`) and C-order (`!fortran`). Under those conditions
/// the on-disk bytes are already the in-memory representation of `[T]`, so the
/// reinterpret is exact; this helper itself only guards length + alignment.
fn build_typed_le_view<T>(data: &[u8], numel: usize, shape: &[usize]) -> Result<Option<Array>>
where
  T: Element,
{
  let size = std::mem::size_of::<T>();
  let need = numel.checked_mul(size).ok_or_else(|| {
    Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(
      "io::npy: numel * itemsize",
      "usize",
    ))
  })?;
  if data.len() != need {
    return Err(npy_err(format!(
      "data length mismatch: need {need} bytes, have {}",
      data.len()
    )));
  }
  let ptr = data.as_ptr();
  // Alignment guard: reading `*const T` from an unaligned pointer is UB, so only
  // take the zero-copy view when the data slice is aligned for `T`; otherwise
  // signal a fallback to the byte-by-byte copy path. (`need == numel * size`
  // was just checked, so the reinterpreted slice length is exact.)
  if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
    return Ok(None);
  }
  // SAFETY: `ptr` is the start of `data`, a live `&[u8]` of exactly `need ==
  // numel * size_of::<T>()` bytes; it is aligned for `T` (checked just above);
  // and every `T` reaching this path is a plain-old-data numeric type whose
  // every bit pattern is a valid value (f16/f32/f64, i8..i64, u8..u64,
  // Complex64 = two f32 lanes) — there are no invalid bit patterns or
  // padding-sensitive invariants — so the bytes are a valid `[T; numel]`. The
  // view borrows `data` only for this call; `Array::from_slice` COPIES it into
  // the MLX array, so no borrow of `data` (hence of any backing mmap) escapes.
  let view: &[T] = unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), numel) };
  Ok(Some(Array::from_slice(view, &shape.to_vec())?))
}

/// Decode `data` (the raw element bytes of `numel` elements of element type
/// `T`, little/big-endian per `swap`) into an owned `Vec<T>`, then build a
/// row-major [`Array`] of `shape`. The byte length must equal `numel *
/// size_of::<T>()` exactly (the caller [`build_array`] already enforces this for
/// every dtype; this is the same invariant re-checked locally on the slice this
/// reader consumes) — too short OR too long is a typed error.
///
/// `from_le` reconstructs one `T` from its `size_of::<T>()` on-disk bytes,
/// applying the host's native interpretation; `swap` reverses each element's
/// bytes first when the file's byte order differs from the host's.
///
/// This is the copying / transform path. For the common case (little-endian
/// file on a little-endian host, C-order, aligned data) [`build_array`] takes
/// the zero-copy [`build_typed_le_view`] instead; this path runs for a
/// byte-swap (big-endian) file or a misaligned hand-crafted buffer, where a
/// transformed buffer is required regardless.
fn build_typed<T, const N: usize>(
  data: &[u8],
  numel: usize,
  shape: &[usize],
  swap: bool,
  from_le: impl Fn([u8; N]) -> T,
) -> Result<Array>
where
  T: Element,
{
  debug_assert_eq!(N, std::mem::size_of::<T>());
  let need = numel.checked_mul(N).ok_or_else(|| {
    Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(
      "io::npy: numel * itemsize",
      "usize",
    ))
  })?;
  if data.len() != need {
    return Err(npy_err(format!(
      "data length mismatch: need {need} bytes, have {}",
      data.len()
    )));
  }
  let mut out: Vec<T> = Vec::new();
  out.try_reserve_exact(numel).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "io::npy: element buffer",
      "elements",
      numel as u64,
      e,
    ))
  })?;
  for chunk in data[..need].as_chunks::<N>().0 {
    let mut elem = [0u8; N];
    elem.copy_from_slice(chunk);
    if swap {
      elem.reverse();
    }
    out.push(from_le(elem));
  }
  Array::from_slice(&out, &shape.to_vec())
}

/// Decode `data` (exactly `need == numel * 8` bytes) into a `Vec<Complex64>` by
/// reading each element's two contiguous f32 lanes, reversing each lane's bytes
/// in place when `swap` (a big-endian `>c8` file), then build a row-major
/// [`Array`] of `shape`. This is the complex64 transform/copy path; the common
/// little-endian case takes the zero-copy view in [`build_array`] instead.
///
/// Reversing each lane independently (not the whole 8-byte element) is numpy's
/// per-lane semantics, a deliberate numpy-correct divergence from MLX's
/// whole-itemsize swap (which would also swap the real/imaginary lanes); see the
/// extended note at the call site. `swap` only matters for a big-endian file,
/// which never appears in a real little-endian weight file.
fn build_complex64_swapped(
  data: &[u8],
  numel: usize,
  need: usize,
  shape: &[usize],
  swap: bool,
) -> Result<Array> {
  let mut out: Vec<Complex64> = Vec::new();
  out.try_reserve_exact(numel).map_err(|e| {
    Error::AllocFailure(AllocFailurePayload::new(
      "io::npy: complex buffer",
      "elements",
      numel as u64,
      e,
    ))
  })?;
  for chunk in data[..need].as_chunks::<8>().0 {
    let mut re = [chunk[0], chunk[1], chunk[2], chunk[3]];
    let mut im = [chunk[4], chunk[5], chunk[6], chunk[7]];
    if swap {
      re.reverse();
      im.reverse();
    }
    out.push(Complex64::new(
      f32::from_le_bytes(re),
      f32::from_le_bytes(im),
    ));
  }
  Array::from_slice(&out, &shape.to_vec())
}

/// Build an [`Array`] of `header.dtype` + `header.shape` from the raw element
/// bytes `data`, applying byte-swap + fortran-order reordering exactly as MLX
/// does on load.
fn build_array(header: &NpyHeader, data: &[u8]) -> Result<Array> {
  let numel = shape_numel(&header.shape)?;
  let swap = header.swap_endianness;

  // Validate the payload length EXACTLY, once, for every dtype path. The
  // declared element count times the dtype's on-disk itemsize is the precise
  // number of element bytes the header promises; `data` is the entire remainder
  // of the file (or npz member) past the header. A buffer that is too SHORT is
  // truncated, and one that is too LONG carries trailing bytes the format does
  // not account for (a corrupt or partially-concatenated weight file) — both are
  // a typed error rather than a silently-decoded prefix. With this exact check
  // the per-dtype readers below always see a slice of precisely `need` bytes.
  let need = numel
    .checked_mul(dtype_itemsize(header.dtype))
    .ok_or_else(|| {
      Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(
        "io::npy: numel * itemsize",
        "usize",
      ))
    })?;
  if data.len() != need {
    return Err(npy_err(format!(
      "data length mismatch: header declares {need} element bytes, file has {}",
      data.len()
    )));
  }

  // For a fortran-ordered (column-major) array MLX reverses the shape, reads
  // the buffer as row-major into the reversed shape, then transposes back —
  // reproduce that exactly so the logical array matches.
  let read_shape: Vec<usize> = if header.fortran_order {
    header.shape.iter().rev().copied().collect()
  } else {
    header.shape.clone()
  };

  // The zero-copy reinterpret in `build_typed_le_view` is correct ONLY when the
  // on-disk bytes already match `T`'s in-memory representation: the file's byte
  // order equals the host's (`!swap`) AND the host is little-endian (where
  // numpy/MLX weight files live; AND'd in per the soundness contract even though
  // `!swap` alone already implies file-order == host-order). The fortran/bool
  // transform paths are handled separately and never use the view. When this is
  // false (a big-endian file, or a big-endian host) every numeric dtype takes
  // the copying `build_typed` path, which reconstructs each element from bytes.
  let can_view = !swap && cfg!(target_endian = "little");

  // For each numeric dtype: try the zero-copy `&[u8] -> &[T]` view first (when
  // `can_view`), which removes the intermediate typed `Vec<T>` copy; if the data
  // is misaligned for `T` the view returns `None` and we fall back to the
  // byte-by-byte `build_typed`. The fortran reorder (when present) is applied to
  // the resulting array below, identically for both paths.
  macro_rules! build_numeric {
    ($T:ty, $N:literal, $from_le:expr) => {{
      let viewed = if can_view {
        build_typed_le_view::<$T>(data, numel, &read_shape)?
      } else {
        None
      };
      match viewed {
        Some(a) => a,
        None => build_typed::<$T, $N>(data, numel, &read_shape, swap, $from_le)?,
      }
    }};
  }

  let arr = match header.dtype {
    Dtype::Bool => {
      // numpy stores bool as one byte per element (0 / 1); MLX's `bool` is
      // likewise a single byte. No byte order applies (size 1). `bool` keeps its
      // own copy path: an arbitrary byte is not a valid `bool` bit pattern (only
      // 0 / 1 are), so a raw reinterpret would be UB — each byte is normalized to
      // `b != 0` exactly as numpy decodes it.
      let need = numel;
      if data.len() != need {
        return Err(npy_err(format!(
          "data length mismatch: need {need} bytes, have {}",
          data.len()
        )));
      }
      let mut out: Vec<bool> = Vec::new();
      out.try_reserve_exact(numel).map_err(|e| {
        Error::AllocFailure(AllocFailurePayload::new(
          "io::npy: bool buffer",
          "elements",
          numel as u64,
          e,
        ))
      })?;
      out.extend(data[..need].iter().map(|&b| b != 0));
      Array::from_slice(&out, &read_shape)?
    }
    Dtype::U8 => build_numeric!(u8, 1, |b| b[0]),
    Dtype::I8 => build_numeric!(i8, 1, |b| b[0] as i8),
    Dtype::U16 => build_numeric!(u16, 2, u16::from_le_bytes),
    Dtype::I16 => build_numeric!(i16, 2, i16::from_le_bytes),
    Dtype::U32 => build_numeric!(u32, 4, u32::from_le_bytes),
    Dtype::I32 => build_numeric!(i32, 4, i32::from_le_bytes),
    Dtype::U64 => build_numeric!(u64, 8, u64::from_le_bytes),
    Dtype::I64 => build_numeric!(i64, 8, i64::from_le_bytes),
    Dtype::F16 => build_numeric!(f16, 2, |b| f16::from_bits(u16::from_le_bytes(b))),
    Dtype::BF16 => build_numeric!(bf16, 2, |b| bf16::from_bits(u16::from_le_bytes(b))),
    Dtype::F32 => build_numeric!(f32, 4, f32::from_le_bytes),
    Dtype::F64 => build_numeric!(f64, 8, f64::from_le_bytes),
    Dtype::Complex64 => {
      // complex64 = two f32 lanes (re, im) per element, 8 bytes. For a
      // big-endian descr each f32 lane is byte-swapped independently (4 bytes
      // each): numpy stores the two lanes contiguously, each in the declared
      // byte order, so the correct conversion reverses each lane in place and
      // leaves the (re, im) ordering intact.
      //
      // This is an intentional, numpy-correct divergence from MLX's CPU
      // `Load::eval_cpu`, which byte-swaps by `out.itemsize()` and therefore
      // reverses all 8 bytes of a complex64 element as one unit. Reversing the
      // whole `[re_be(4)][im_be(4)]` element yields `[reverse(im)][reverse(re)]`
      // — it swaps real and imaginary in addition to fixing each lane's
      // endianness, which is numerically wrong. We follow numpy's per-lane
      // semantics instead; this only affects big-endian complex64 (`>c8`),
      // which never appears in real little-endian weight files.
      let need = numel.checked_mul(8).ok_or_else(|| {
        Error::ArithmeticOverflow(crate::error::ArithmeticOverflowPayload::new(
          "io::npy: complex numel * 8",
          "usize",
        ))
      })?;
      if data.len() != need {
        return Err(npy_err(format!(
          "data length mismatch: need {need} bytes, have {}",
          data.len()
        )));
      }
      // Common case (`!swap`, little-endian host): the on-disk `[re_le(4)]
      // [im_le(4)]` element bytes are exactly `Complex64`'s `#[repr(C)]`
      // layout, so the same zero-copy `&[u8] -> &[Complex64]` view used by the
      // numeric dtypes is sound — try it first, falling back to the per-lane
      // copy when misaligned. Only the big-endian (`>c8`) path needs the
      // per-lane byte reversal in `build_complex64_swapped`.
      let viewed = if can_view {
        build_typed_le_view::<Complex64>(data, numel, &read_shape)?
      } else {
        None
      };
      match viewed {
        Some(arr) => arr,
        None => build_complex64_swapped(data, numel, need, &read_shape, swap)?,
      }
    }
  };

  if header.fortran_order {
    // MLX returns the lazy transpose of the reversed-shape read (a strided
    // view); materialize it row-major so the logical array is directly
    // readable via the contiguity-requiring data accessors (`to_vec` /
    // `as_slice`). `contiguous(_, false)` is a no-op + refcount bump when the
    // array is already row-contiguous and a fresh row-major copy otherwise.
    contiguous(&transpose(&arr)?, false)
  } else {
    Ok(arr)
  }
}

/// Parse a complete `.npy` byte stream (header + data) into an [`Array`].
///
/// The buffer must contain the whole file. Used directly by [`load_npy`] and
/// per-member by [`load_npz`].
fn load_npy_bytes(bytes: &[u8]) -> Result<Array> {
  let header = parse_header(bytes)?;
  let data = bytes
    .get(header.data_offset..)
    .ok_or_else(|| npy_err("data offset past end of file"))?;
  build_array(&header, data)
}

// ───────────────────────────── public load ─────────────────────────────

/// Memory-map `path` read-only, returning a [`memmapix::Mmap`] whose `Deref`
/// to `&[u8]` is a view of the whole file with NO intervening whole-file `Vec`.
/// This is the load building block for both [`load_npy`] and [`load_npz`]: it
/// replaces a `read_to_end` of the entire file, removing one full copy of the
/// data from peak load memory.
///
/// SAFETY contract for the caller: the returned map aliases the on-disk file
/// for its lifetime. Like every memory-mapped weight loader (this is exactly how
/// `safetensors` loads), the standard weight-loading assumption applies — the
/// file must not be truncated or modified by another process while the map is
/// live, since the OS would then fault or surface changed bytes through the
/// view. A weight file being loaded is not concurrently rewritten in practice;
/// the resulting [`Array`] OWNS its data (`Array::from_slice` copies), so once
/// the build completes the map can be dropped with no borrow of it escaping.
fn mmap_file(path: &Path, op: &'static str) -> Result<memmapix::Mmap> {
  let file = File::open(path)
    .map_err(|e| Error::FileIo(FileIoPayload::new(op, FileOp::Open, path.to_path_buf(), e)))?;
  // SAFETY: `file` is a freshly opened read-only handle to `path`. Mapping it is
  // sound under the documented weight-loading assumption above (the file is not
  // concurrently truncated/modified during the load); we never write through the
  // map (it is an `Mmap`, not `MmapMut`), and every read of the resulting `&[u8]`
  // below is bounds-checked against `map.len()` by the existing parser. The map
  // is dropped when this loader returns; no `&[u8]` borrow of it escapes, because
  // `Array::from_slice` copies the bytes into the MLX array.
  unsafe { memmapix::Mmap::map(&file) }
    .map_err(|e| Error::FileIo(FileIoPayload::new(op, FileOp::Read, path.to_path_buf(), e)))
}

/// Load a NumPy `.npy` file (a single array) into an [`Array`].
///
/// Mirrors `mlx.core.load` for the `npy` format: validates the magic +
/// version, decodes the header (`descr` / `fortran_order` / `shape`), maps the
/// numpy dtype to an MLX [`Dtype`] (including MLX's `V2` bfloat16 encoding),
/// and reads the raw little/big-endian element bytes into a row-major array. A
/// `fortran_order` array is reordered to row-major as MLX does. A truncated or
/// corrupt file yields a typed [`Error`] rather than panicking.
///
/// The file is memory-mapped (not read into a whole-file buffer): the array is
/// built directly from the mapped bytes, and for the common case (little-endian,
/// C-order, aligned) the element bytes are reinterpreted as a typed view and
/// copied once straight into the MLX array — so peak load memory is ~1× the data
/// (the MLX array, the irreducible floor) rather than ~3×.
pub fn load_npy(path: &Path) -> Result<Array> {
  let map = mmap_file(path, "io::load_npy: open")?;
  // `&map[..]` (via `Deref<Target = [u8]>`) is the whole-file byte view; the
  // shared in-memory entry parses the header + builds the array from it. The
  // array owns its data, so the map is free to drop when this returns.
  load_npy_bytes(&map)
}

/// Load a NumPy `.npz` file (a ZIP archive of `<name>.npy` members) into a map
/// of named arrays.
///
/// Mirrors `mlx.core.load` for the `npz` format and `numpy.load` on a `.npz`:
/// each ZIP member is decoded as a `.npy` array and keyed by its member name
/// with the trailing `.npy` stripped (matching MLX). Both STORED
/// (`mx.savez`) and DEFLATE-compressed (`mx.savez_compressed`) members are
/// supported. A corrupt archive or a malformed member yields a typed
/// [`Error`].
///
/// Unlike MLX's python helper — which uses `unordered_map::insert` and so
/// silently keeps the first of any colliding keys — two members that normalize
/// to the same key (e.g. `w.npy` twice, or `w.npy` plus a bare `w`) are
/// rejected with a typed [`Error::Parse`]. A duplicate weight key is ambiguous;
/// for a weight loader, failing loudly is preferable to silently dropping one
/// array.
pub fn load_npz(path: &Path) -> Result<HashMap<String, Array>> {
  // Memory-map the zip file and run the archive reader over the mapped bytes
  // (`Cursor<&[u8]>` is `Read + Seek`): the central directory and every member
  // are read from the map, never from a separate whole-file `Vec`.
  let map = mmap_file(path, "io::load_npz: open")?;
  let mapped: &[u8] = &map;
  let mut archive = zip::ZipArchive::new(std::io::Cursor::new(mapped))
    .map_err(|e| Error::Parse(ParsePayload::new("io::load_npz", "npz (zip archive)", e)))?;
  let mut out = HashMap::new();
  for i in 0..archive.len() {
    // Read the member's metadata first. For a STORED (uncompressed) member the
    // zip crate exposes its data byte range via `data_start()` (the offset of
    // the member's data past its local header) and `size()` (the uncompressed
    // length, which for STORED equals the on-disk length): we slice the mmap
    // directly over that range and parse it as `.npy` with NO per-member copy.
    // For a DEFLATE member the data must be inflated into a buffer, so we re-read
    // it through the zip reader. (Zero-copy for STORED members is bounded by what
    // the zip crate's safe API exposes — `data_start()` / `size()` — so we do not
    // hand-roll any zip parsing.)
    let member = archive
      .by_index(i)
      .map_err(|e| Error::Parse(ParsePayload::new("io::load_npz: member", "npz member", e)))?;
    let name = member.name().to_string();
    let is_stored = member.compression() == zip::CompressionMethod::Stored;
    // `data_start()` is populated by `by_index` above (it sets up the member
    // reader, which records the data offset). The zip crate returns it as an
    // `Option<u64>` (`None` only if the offset was never resolved); a `None`
    // here is a malformed/unsupported member, surfaced as a typed parse error.
    let data_start = member.data_start();
    let uncompressed = member.size();
    drop(member);

    let arr = if is_stored {
      // Slice the mmap over the STORED member's data range, fully bounds-checked
      // against the mapped length (a corrupt central directory could declare an
      // out-of-range offset/length): `start + len` must not overflow and must lie
      // within `mapped`. The resulting slice is parsed exactly like a standalone
      // `.npy` file — same header validation, same exact payload-length check.
      let data_start =
        data_start.ok_or_else(|| npy_err("npz member has no resolved data offset"))?;
      let start =
        usize::try_from(data_start).map_err(|_| npy_err("npz member data offset exceeds usize"))?;
      let len =
        usize::try_from(uncompressed).map_err(|_| npy_err("npz member length exceeds usize"))?;
      let end = start
        .checked_add(len)
        .ok_or_else(|| npy_err("npz member data range overflows"))?;
      let member_bytes = mapped
        .get(start..end)
        .ok_or_else(|| npy_err("npz member data range extends past end of archive"))?;
      load_npy_bytes(member_bytes)?
    } else {
      // DEFLATE member: inflation is unavoidable, so re-acquire the member and
      // decompress it into a buffer bounded by this ONE member's uncompressed
      // size (not the whole archive), then build directly from it.
      let mut member = archive
        .by_index(i)
        .map_err(|e| Error::Parse(ParsePayload::new("io::load_npz: member", "npz member", e)))?;
      let mut bytes = Vec::new();
      member.read_to_end(&mut bytes).map_err(|e| {
        Error::FileIo(FileIoPayload::new(
          "io::load_npz: member read",
          FileOp::Read,
          path.to_path_buf(),
          e,
        ))
      })?;
      load_npy_bytes(&bytes)?
    };
    // Strip a trailing `.npy` from the member name, matching MLX.
    let key = name
      .strip_suffix(".npy")
      .map(str::to_string)
      .unwrap_or(name);
    // Reject a collision on the normalized key (e.g. `w.npy` twice, or `w.npy`
    // alongside a bare `w`). MLX's python helper uses `unordered_map::insert`,
    // which silently keeps the FIRST occurrence; we are deliberately stricter:
    // an archive with two members mapping to the same weight name is
    // ambiguous/corrupt, so for a weight loader the robust choice is to fail
    // loudly rather than silently drop one array.
    match out.entry(key) {
      std::collections::hash_map::Entry::Occupied(e) => {
        return Err(Error::Parse(ParsePayload::new(
          "io::load_npz",
          "npz",
          NpyParseError {
            detail: format!("duplicate array key {:?} in npz archive", e.key()),
          },
        )));
      }
      std::collections::hash_map::Entry::Vacant(e) => {
        e.insert(arr);
      }
    }
  }
  Ok(out)
}

// ───────────────────────────── header build ─────────────────────────────

/// Serialize the `.npy` header bytes (magic + version + padded dict) for an
/// array of `dtype`, `shape`, row-major (`fortran_order = False`) — the
/// faithful port of MLX's `save` header construction (`load.cpp`).
fn build_header_bytes(dtype: Dtype, shape: &[usize]) -> Result<Vec<u8>> {
  // Dict text, matching MLX's `header << "{'descr': '" ... "', 'shape': (...)}"`.
  let descr = dtype_to_descr(dtype);
  let mut dict = String::new();
  dict.push_str("{'descr': '");
  dict.push_str(&descr);
  dict.push_str("', 'fortran_order': False, 'shape': (");
  for d in shape {
    // MLX writes every dim followed by ", " (including the last).
    dict.push_str(&d.to_string());
    dict.push_str(", ");
  }
  dict.push_str(")}");

  let header_len = dict.len();
  // MLX: a v1 header is used iff header_len + 15 < u16::MAX.
  let is_v1 = header_len + 15 < u16::MAX as usize;
  // Pad magic(6) + version(2) + len-field(2 or 4) + dict + '\n' to a multiple
  // of 16. MLX computes the padding as the modulus and then appends a '\n'.
  let len_field = if is_v1 { 2usize } else { 4usize };
  let padding = (6 + 2 + len_field + header_len + 1) % 16;
  dict.push_str(&" ".repeat(padding));
  dict.push('\n');

  let total_header_len = dict.len();
  let mut out = Vec::new();
  out.extend_from_slice(&MAGIC);
  if is_v1 {
    out.push(0x01);
    out.push(0x00);
    let len = u16::try_from(total_header_len)
      .map_err(|_| npy_err("header length does not fit in the v1 u16 length field"))?;
    out.extend_from_slice(&len.to_le_bytes());
  } else {
    out.push(0x02);
    out.push(0x00);
    let len = u32::try_from(total_header_len)
      .map_err(|_| npy_err("header length does not fit in the v2 u32 length field"))?;
    out.extend_from_slice(&len.to_le_bytes());
  }
  out.extend_from_slice(dict.as_bytes());
  Ok(out)
}

/// Reinterpret an array's contiguous typed buffer as raw little-endian element
/// bytes, the on-disk payload. The array is materialized + dtype-checked by
/// the caller's `as_slice` (forces eval). On every target this crate supports
/// the host is little-endian, so the in-memory bytes are already the on-disk
/// order; this helper never byte-swaps (matching MLX, which writes host order).
fn typed_bytes<T: Element>(arr: &mut Array) -> Result<Vec<u8>> {
  let slice = arr.as_slice::<T>()?;
  // SAFETY: `slice` is a valid `&[T]` of `slice.len()` elements; reinterpreting
  // it as a byte slice of `len * size_of::<T>()` bytes is sound (every `T` here
  // is a plain-old-data numeric/bool/Complex64 with no padding-sensitive
  // invariants), and `size_of::<T>()` cannot overflow `usize` for a slice that
  // already exists in memory.
  let bytes = unsafe {
    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice))
  };
  Ok(bytes.to_vec())
}

/// Append an array's raw element bytes (dispatched on dtype) to `out`.
fn append_array_bytes(out: &mut Vec<u8>, arr: &mut Array) -> Result<()> {
  let dtype = arr.dtype()?;
  let bytes = match dtype {
    Dtype::Bool => typed_bytes::<bool>(arr)?,
    Dtype::U8 => typed_bytes::<u8>(arr)?,
    Dtype::I8 => typed_bytes::<i8>(arr)?,
    Dtype::U16 => typed_bytes::<u16>(arr)?,
    Dtype::I16 => typed_bytes::<i16>(arr)?,
    Dtype::U32 => typed_bytes::<u32>(arr)?,
    Dtype::I32 => typed_bytes::<i32>(arr)?,
    Dtype::U64 => typed_bytes::<u64>(arr)?,
    Dtype::I64 => typed_bytes::<i64>(arr)?,
    Dtype::F16 => typed_bytes::<f16>(arr)?,
    Dtype::BF16 => typed_bytes::<bf16>(arr)?,
    Dtype::F32 => typed_bytes::<f32>(arr)?,
    Dtype::F64 => typed_bytes::<f64>(arr)?,
    Dtype::Complex64 => typed_bytes::<Complex64>(arr)?,
  };
  out.extend_from_slice(&bytes);
  Ok(())
}

/// Serialize an [`Array`] to the in-memory `.npy` byte representation MLX
/// writes (`mx.save`). The array is materialized (eval) by the byte extraction;
/// a non-contiguous array is made contiguous implicitly via `as_slice`'s
/// contiguity contract (an [`Error::NonContiguous`] is surfaced if it is
/// strided — callers should pass a row-major array).
fn save_npy_bytes(arr: &mut Array) -> Result<Vec<u8>> {
  let dtype = arr.dtype()?;
  // MLX refuses to serialize an empty (zero-element) array.
  if arr.size() == 0 {
    return Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
      "io::save_npy: empty array",
      dtype,
      &[],
    )));
  }
  let shape = arr.shape();
  let mut out = build_header_bytes(dtype, &shape)?;
  append_array_bytes(&mut out, arr)?;
  Ok(out)
}

// ───────────────────────────── public save ─────────────────────────────

/// Save an [`Array`] to a NumPy `.npy` file, byte-compatible with `mx.save`
/// (and `numpy.load`). Writes the same magic + version + header dict + raw
/// host-order element bytes MLX writes. The array is evaluated; an empty
/// (zero-element) array is rejected with a typed error, matching MLX.
pub fn save_npy(path: &Path, array: &mut Array) -> Result<()> {
  let bytes = save_npy_bytes(array)?;
  let mut file = File::create(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "io::save_npy: create",
      FileOp::Create,
      path.to_path_buf(),
      e,
    ))
  })?;
  file.write_all(&bytes).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "io::save_npy: write",
      FileOp::Write,
      path.to_path_buf(),
      e,
    ))
  })
}

/// Save a map of named arrays to a NumPy `.npz` file (a ZIP archive of
/// `<name>.npy` members), byte-compatible with `mx.savez` / `numpy.savez`.
///
/// Members are STORED (uncompressed), matching `mx.savez`. Each array is
/// serialized via the same `.npy` writer as [`save_npy`]. The companion
/// [`load_npz`] (and `numpy.load`) round-trips the result.
pub fn save_npz(path: &Path, arrays: &mut HashMap<String, Array>) -> Result<()> {
  save_npz_impl(path, arrays, zip::CompressionMethod::Stored)
}

/// Save a map of named arrays to a DEFLATE-compressed NumPy `.npz` file,
/// byte-compatible with `mx.savez_compressed` / `numpy.savez_compressed`.
pub fn save_npz_compressed(path: &Path, arrays: &mut HashMap<String, Array>) -> Result<()> {
  save_npz_impl(path, arrays, zip::CompressionMethod::Deflated)
}

fn save_npz_impl(
  path: &Path,
  arrays: &mut HashMap<String, Array>,
  method: zip::CompressionMethod,
) -> Result<()> {
  let file = File::create(path).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "io::save_npz: create",
      FileOp::Create,
      path.to_path_buf(),
      e,
    ))
  })?;
  let mut writer = zip::ZipWriter::new(file);
  let options: zip::write::FileOptions<'_, ()> =
    zip::write::FileOptions::default().compression_method(method);
  for (name, arr) in arrays.iter_mut() {
    let bytes = save_npy_bytes(arr)?;
    let member = format!("{name}.npy");
    writer.start_file(member, options).map_err(|e| {
      Error::Parse(ParsePayload::new(
        "io::save_npz: start_file",
        "npz member",
        e,
      ))
    })?;
    writer.write_all(&bytes).map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "io::save_npz: member write",
        FileOp::Write,
        path.to_path_buf(),
        e,
      ))
    })?;
  }
  writer
    .finish()
    .map_err(|e| Error::Parse(ParsePayload::new("io::save_npz: finish", "npz archive", e)))?;
  Ok(())
}

#[cfg(test)]
mod tests;
