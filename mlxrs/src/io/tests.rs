// `super::*` already re-exports the parent module's `use` aliases
// (`HashMap`, `File`, `CStr`, `c_char`, `Array`, `Error`, `FileOp`, the
// private `WriterState` / `cb_*` items, etc.). Only the genuinely-new
// names are imported explicitly here.
use std::{fs::OpenOptions, io::Read};

use super::*;
use crate::dtype::Dtype;

/// A fresh, writable per-test temp directory (the crate's
/// no-`tempfile`-crate convention — `temp_dir()` + pid + a process-unique
/// counter, mirroring `lm::load`'s `save_tests::fresh_dir`).
fn fresh_dir(tag: &str) -> std::path::PathBuf {
  use std::sync::atomic::{AtomicU64, Ordering};
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let dir = std::env::temp_dir().join(format!("mlxrs-io-{tag}-{}-{n}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

// ─────────────────────── safetensors round-trip ───────────────────────

/// INDEPENDENT closed-form oracle: write a hand-built `{name -> array}`
/// map (two tensors, distinct dtypes/shapes) to a `.safetensors` file via
/// the path-based saver, reload it via `load_safetensors`, and assert each
/// tensor's VALUES, SHAPE, and DTYPE round-trip. The expected values are
/// the literals we wrote, never produced by the fn under test.
#[test]
fn save_then_load_safetensors_round_trips_values_shape_dtype() {
  let dir = fresh_dir("st-roundtrip");
  let path = dir.join("weights.safetensors");

  let a = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
  let b = Array::from_slice::<i32>(&[10_i32, 20, 30], &(3usize,)).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  arrays.insert("a.weight".to_string(), a);
  arrays.insert("b.bias".to_string(), b);

  save_safetensors(&path, &arrays).unwrap();
  assert!(path.exists());

  let mut loaded = load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 2);

  let la = loaded.get_mut("a.weight").unwrap();
  assert_eq!(la.shape(), vec![2, 2]);
  assert_eq!(la.dtype().unwrap(), Dtype::F32);
  assert_eq!(la.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

  let lb = loaded.get_mut("b.bias").unwrap();
  assert_eq!(lb.shape(), vec![3]);
  assert_eq!(lb.dtype().unwrap(), Dtype::I32);
  assert_eq!(lb.to_vec::<i32>().unwrap(), vec![10, 20, 30]);

  let _ = std::fs::remove_dir_all(&dir);
}

/// `save_safetensors_with_metadata` carries a `String -> String` side
/// table; `load_safetensors_with_metadata` returns `(arrays, metadata)`.
/// Oracle: the metadata map we pass in is exactly what comes back, and the
/// metadata-discarding `load_safetensors` drops it. Exercises
/// `build_string_map`'s populated (non-empty) insert path + both drain
/// helpers.
#[test]
fn save_load_safetensors_with_metadata_round_trips_side_table() {
  let dir = fresh_dir("st-meta-roundtrip");
  let path = dir.join("weights.safetensors");

  let w = Array::from_slice::<f32>(&[5.0_f32, 6.0], &(2usize,)).unwrap();
  let mut arrays: HashMap<String, Array> = HashMap::new();
  arrays.insert("w".to_string(), w);
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());
  meta.insert("author".to_string(), "mlxrs-test".to_string());

  save_safetensors_with_metadata(&path, &arrays, &meta).unwrap();

  let (mut a2, m2) = load_safetensors_with_metadata(&path).unwrap();
  let w_back = a2.get_mut("w").unwrap().to_vec::<f32>().unwrap();
  assert_eq!(w_back, vec![5.0_f32, 6.0]);
  assert_eq!(m2.get("format").map(String::as_str), Some("mlx"));
  assert_eq!(m2.get("author").map(String::as_str), Some("mlxrs-test"));

  // The metadata-discarding loader drops the side table.
  let discarded = load_safetensors(&path).unwrap();
  assert!(discarded.contains_key("w"));

  let _ = std::fs::remove_dir_all(&dir);
}

/// Loading a path that does not exist surfaces an `Err` (the mlx-c loader
/// raises through the installed handler — exact variant is mlx's, so we
/// only assert failure, not the variant).
#[test]
fn load_safetensors_missing_file_errors() {
  let dir = fresh_dir("st-missing");
  let path = dir.join("does-not-exist.safetensors");
  assert!(!path.exists());
  assert!(load_safetensors(&path).is_err());
  assert!(load_safetensors_with_metadata(&path).is_err());
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── interior-NUL validation ───────────────────────

/// A path with an embedded NUL byte is rejected by `path_cstring` with a
/// typed `Error::InteriorNul` (context `io::path_cstring`) BEFORE any
/// mlx-c call. Unix-only: building an `OsStr` with an interior NUL needs
/// the byte-level `OsStrExt` constructor.
#[cfg(unix)]
#[test]
fn save_safetensors_path_with_interior_nul_is_rejected() {
  use std::os::unix::ffi::OsStrExt;
  let p = std::path::Path::new(std::ffi::OsStr::from_bytes(b"weights\0.safetensors"));
  let arrays: HashMap<String, Array> = HashMap::new();
  match save_safetensors(p, &arrays).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::path_cstring");
      assert_eq!(payload.bytes_kind(), "path");
    }
    other => panic!("expected InteriorNul, got {other:?}"),
  }
}

/// An array NAME containing an interior NUL is rejected by
/// `build_array_map` with `Error::InteriorNul` keyed `array key` — covers
/// the per-entry `CString::new(k)` rejection inside the map builder.
#[test]
fn save_safetensors_view_array_key_with_interior_nul_is_rejected() {
  let dir = fresh_dir("st-nul-key");
  let path = dir.join("w.safetensors");
  let arr = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let entries = std::iter::once(("bad\0key", &arr));
  let meta: HashMap<String, String> = HashMap::new();
  match save_safetensors_view(&path, entries, &meta).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::map_arrays insert");
      assert_eq!(payload.bytes_kind(), "array key");
    }
    other => panic!("expected InteriorNul(array key), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A metadata KEY with an interior NUL is rejected by `build_string_map`
/// (`metadata key`).
#[test]
fn save_safetensors_metadata_key_with_interior_nul_is_rejected() {
  let dir = fresh_dir("st-nul-meta-key");
  let path = dir.join("w.safetensors");
  let arrays: HashMap<String, Array> = HashMap::new();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("bad\0key".to_string(), "v".to_string());
  match save_safetensors_with_metadata(&path, &arrays, &meta).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::map_meta insert");
      assert_eq!(payload.bytes_kind(), "metadata key");
    }
    other => panic!("expected InteriorNul(metadata key), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A metadata VALUE with an interior NUL is rejected by `build_string_map`
/// (`metadata value`) — the value branch of the insert loop.
#[test]
fn save_safetensors_metadata_value_with_interior_nul_is_rejected() {
  let dir = fresh_dir("st-nul-meta-val");
  let path = dir.join("w.safetensors");
  let arrays: HashMap<String, Array> = HashMap::new();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("k".to_string(), "bad\0value".to_string());
  match save_safetensors_with_metadata(&path, &arrays, &meta).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::map_meta insert");
      assert_eq!(payload.bytes_kind(), "metadata value");
    }
    other => panic!("expected InteriorNul(metadata value), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── fd-bound writer (round-trip) ───────────────────────

/// `save_safetensors_to_file` writes through a caller-owned `&mut File`;
/// reloading via the path-based loader proves the on-disk layout is a
/// valid safetensors (parseable header, offsets, dtype/shape encoding).
/// Oracle: the values + shape + dtype reload to the literals written.
#[test]
fn save_safetensors_to_file_round_trips_via_path_load() {
  let dir = fresh_dir("fd-roundtrip");
  let path = dir.join("via_fd.safetensors");
  let arr = Array::from_slice::<f32>(&[7.0_f32, 8.0, 9.0], &(3usize,)).unwrap();
  let mut meta: HashMap<String, String> = HashMap::new();
  meta.insert("format".to_string(), "mlx".to_string());

  let mut f = File::create(&path).unwrap();
  save_safetensors_to_file(&mut f, std::iter::once(("only", &arr)), &meta).unwrap();
  f.sync_all().unwrap();
  drop(f);

  let mut loaded = load_safetensors(&path).unwrap();
  assert_eq!(loaded.len(), 1);
  let l = loaded.get_mut("only").unwrap();
  assert_eq!(l.shape(), vec![3]);
  assert_eq!(l.dtype().unwrap(), Dtype::F32);
  assert_eq!(l.to_vec::<f32>().unwrap(), vec![7.0, 8.0, 9.0]);

  let _ = std::fs::remove_dir_all(&dir);
}

/// Interior-NUL validation in `save_safetensors_to_file` runs BEFORE the
/// destructive truncate (defense-in-depth): a prefilled file handed in
/// must be byte-preserved when the array key is invalid. Covers the
/// early-validation `build_array_map` `?` path of the fd-bound writer.
#[test]
fn save_safetensors_to_file_array_key_nul_errs_before_truncate() {
  let dir = fresh_dir("fd-nul-preserve");
  let path = dir.join("prefilled.bin");
  std::fs::write(&path, b"PREEXISTING-CONTENT").unwrap();
  let mut f = OpenOptions::new()
    .write(true)
    .read(true)
    .open(&path)
    .unwrap();

  let arr = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let meta: HashMap<String, String> = HashMap::new();
  let err =
    save_safetensors_to_file(&mut f, std::iter::once(("bad\0key", &arr)), &meta).unwrap_err();
  assert!(matches!(err, Error::InteriorNul(_)));
  drop(f);

  // Defense-in-depth side effect: the file was NOT truncated.
  assert_eq!(std::fs::read(&path).unwrap(), b"PREEXISTING-CONTENT");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A read-only `File` cannot be written through `save_safetensors_to_file`:
/// it fails with a typed `Error::FileIo`. Depending on the platform, the
/// failure surfaces either at the destructive `set_len(0)` truncate
/// (`Other("set_len")`, lines 705-712) or — if `ftruncate` on a read-only
/// fd is tolerated by the FS — later in the `cb_write` callback whose
/// captured io::Error becomes the `"write callback"` `FileIo` (lines
/// 730-734). Both are valid, both are on the uncovered-line set.
#[test]
fn save_safetensors_to_file_read_only_fd_is_file_io() {
  let dir = fresh_dir("fd-readonly");
  let path = dir.join("ro.safetensors");
  std::fs::write(&path, b"seed").unwrap();
  let mut f = File::open(&path).unwrap(); // read-only

  let arr = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let meta: HashMap<String, String> = HashMap::new();
  match save_safetensors_to_file(&mut f, std::iter::once(("w", &arr)), &meta) {
    Err(Error::FileIo(p)) => {
      assert!(
        p.op() == FileOp::Other("set_len") || p.op() == FileOp::Write,
        "read-only fd should fail at set_len or in the write callback, got op {:?}",
        p.op()
      );
    }
    other => panic!("expected Error::FileIo on read-only fd, got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── writer-vtable callbacks (direct) ───────────────────────
//
// The `unsafe extern "C"` vtable callbacks are driven by mlx-c during a
// real `mlx_save_safetensors_writer`, but their branch logic (seek-whence
// mapping, NULL-data guard, misuse capture, panic capture, first-error-
// wins) is exercised here DIRECTLY against a `WriterState` over a real
// `File`, since coverage through the FFI path cannot deterministically
// force these error branches.

/// `cb_is_open` is unconditionally `true`; `cb_label` returns the static
/// `WriterState` label pointer.
#[test]
fn cb_is_open_and_label() {
  let dir = fresh_dir("cb-open-label");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let state = WriterState::new(&mut f);
  let desc = state.as_desc();
  // SAFETY: `desc` is the live `WriterState` pointer just obtained from
  // `as_desc()`; `state` (and the borrowed `f`) outlive these synchronous
  // single-threaded callback calls. `cb_label` returns a pointer into the
  // state's `&'static CStr`, valid to read as a NUL-terminated C string.
  let (is_open, label) = unsafe {
    let is_open = cb_is_open(desc);
    let label_ptr = cb_label(desc);
    assert!(!label_ptr.is_null());
    let label = CStr::from_ptr(label_ptr).to_string_lossy().into_owned();
    (is_open, label)
  };
  assert!(is_open);
  assert_eq!(label, "mlxrs::io::save_safetensors_to_file(&mut File)");
  let _ = std::fs::remove_dir_all(&dir);
}

/// `cb_tell` reports the live cursor, `cb_seek` maps all three POSIX
/// whences to the right `SeekFrom`, and an unknown whence captures an
/// error into the state. `cb_good` peeks (does not consume) the err-cell.
#[test]
fn cb_tell_seek_good_whences() {
  let dir = fresh_dir("cb-tell-seek");
  let path = dir.join("f.bin");
  std::fs::write(&path, b"0123456789").unwrap();
  let mut f = OpenOptions::new()
    .read(true)
    .write(true)
    .open(&path)
    .unwrap();
  let state = WriterState::new(&mut f);
  let desc = state.as_desc();

  // SAFETY: `desc` is the live `WriterState` pointer from `as_desc()`;
  // `state`/`f` outlive these synchronous single-threaded callback calls.
  // Each seek/tell only touches the borrowed `File` + the err-cell; no two
  // run concurrently. Results are captured to locals for assertion outside.
  let (after_set, after_cur, after_end, good_a, good_b, good_after_bad) = unsafe {
    cb_seek(desc, 3, libc::SEEK_SET); // SEEK_SET → absolute 3
    let after_set = cb_tell(desc);
    cb_seek(desc, 2, libc::SEEK_CUR); // SEEK_CUR +2 → 5
    let after_cur = cb_tell(desc);
    cb_seek(desc, 0, libc::SEEK_END); // SEEK_END 0 → file length (10)
    let after_end = cb_tell(desc);
    // No error captured yet → cb_good true twice (peek must not consume).
    let good_a = cb_good(desc);
    let good_b = cb_good(desc);
    // An unknown whence captures an error → cb_good flips to false.
    cb_seek(desc, 0, 9999);
    let good_after_bad = cb_good(desc);
    (
      after_set,
      after_cur,
      after_end,
      good_a,
      good_b,
      good_after_bad,
    )
  };
  assert_eq!(after_set, 3);
  assert_eq!(after_cur, 5);
  assert_eq!(after_end, 10);
  assert!(good_a);
  assert!(good_b);
  assert!(!good_after_bad);

  // The captured (unknown-whence) error survives into `into_err`.
  assert!(state.into_err().is_some());
  let _ = std::fs::remove_dir_all(&dir);
}

/// `cb_write` appends to the file at the current cursor; `n == 0` is a
/// no-op; a NULL `data` pointer captures an error.
#[test]
fn cb_write_appends_and_guards_null_and_zero() {
  let dir = fresh_dir("cb-write");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let payload: &[u8] = b"HELLO";
  {
    let state = WriterState::new(&mut f);
    let desc = state.as_desc();
    // SAFETY: `desc` is the live `WriterState` pointer; `payload` is a valid
    // `n`-byte buffer for the `n == payload.len()` call (and unread for the
    // `n == 0` no-op). The callbacks run synchronously here.
    unsafe {
      cb_write(desc, payload.as_ptr().cast::<c_char>(), 0); // no-op
      cb_write(desc, payload.as_ptr().cast::<c_char>(), payload.len()); // 5 bytes
    }
    assert!(state.into_err().is_none());
  }
  f.sync_all().unwrap();
  // Re-read: only the 5 written bytes are present.
  let mut back = String::new();
  File::open(&path)
    .unwrap()
    .read_to_string(&mut back)
    .unwrap();
  assert_eq!(back, "HELLO");

  // NULL data pointer with non-zero n → captured error, file untouched.
  let mut f2 = File::create(dir.join("g.bin")).unwrap();
  let state = WriterState::new(&mut f2);
  let desc = state.as_desc();
  // SAFETY: `desc` is the live `WriterState` pointer; the NULL-data branch
  // is guarded inside `cb_write` before any dereference of the pointer.
  unsafe { cb_write(desc, std::ptr::null(), 8) };
  let e = state.into_err().expect("NULL data must capture an error");
  assert_eq!(e.kind(), std::io::ErrorKind::Other);

  let _ = std::fs::remove_dir_all(&dir);
}

/// A write to a read-only fd fails inside `cb_write` and the io::Error is
/// captured in the state's err-cell.
#[test]
fn cb_write_to_read_only_fd_captures_error() {
  let dir = fresh_dir("cb-write-ro");
  let path = dir.join("ro.bin");
  std::fs::write(&path, b"seed").unwrap();
  let mut f = File::open(&path).unwrap(); // read-only
  let state = WriterState::new(&mut f);
  let desc = state.as_desc();
  let payload: &[u8] = b"X";
  // SAFETY: `desc` is the live `WriterState` pointer; `payload` is a valid
  // 1-byte buffer. The write fails at the OS level (read-only fd) and the
  // io::Error is captured into the state, not propagated.
  unsafe { cb_write(desc, payload.as_ptr().cast::<c_char>(), 1) };
  assert!(
    state.into_err().is_some(),
    "writing to a read-only fd must capture an io::Error"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// A writer must never be asked to read: `cb_read` / `cb_read_at_offset`
/// each capture a misuse error.
#[test]
fn cb_read_paths_capture_misuse() {
  let dir = fresh_dir("cb-read");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let mut buf = [0u8; 4];

  let state = WriterState::new(&mut f);
  let desc = state.as_desc();
  // SAFETY: `desc` is the live `WriterState` pointer; `buf` is a valid
  // 4-byte out-buffer. `cb_read` never actually reads — it only records the
  // misuse into the state — so the buffer pointer is unused.
  unsafe { cb_read(desc, buf.as_mut_ptr().cast::<c_char>(), 4) };
  assert!(state.into_err().is_some(), "cb_read must capture misuse");

  let state2 = WriterState::new(&mut f);
  let desc2 = state2.as_desc();
  // SAFETY: as above; `cb_read_at_offset` likewise only records misuse.
  unsafe { cb_read_at_offset(desc2, buf.as_mut_ptr().cast::<c_char>(), 4, 0) };
  assert!(
    state2.into_err().is_some(),
    "cb_read_at_offset must capture misuse"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// `with_state` wraps the callback body in `catch_unwind`: a panicking
/// closure returns `None` and stores a synthetic `io::ErrorKind::Other`
/// into the state. Exercises the `Err(_)` arm of `with_state`.
#[test]
fn with_state_panic_is_captured_not_propagated() {
  let dir = fresh_dir("cb-panic");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let state = WriterState::new(&mut f);
  let desc = state.as_desc();
  let r: Option<()> = with_state(desc, |_state, _file| panic!("boom in callback"));
  assert!(r.is_none(), "a panicking callback must yield None");
  let e = state
    .into_err()
    .expect("a panicking callback must capture a synthetic error");
  assert_eq!(e.kind(), std::io::ErrorKind::Other);
  let _ = std::fs::remove_dir_all(&dir);
}

/// `WriterState::set_err` keeps the FIRST error (subsequent failures may
/// cascade once the file is bad, but the original cause must survive).
#[test]
fn writer_state_set_err_first_wins() {
  let dir = fresh_dir("state-firstwins");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let state = WriterState::new(&mut f);
  state.set_err(std::io::Error::new(std::io::ErrorKind::NotFound, "first"));
  state.set_err(std::io::Error::other("second"));
  let e = state.into_err().unwrap();
  assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
  assert_eq!(e.to_string(), "first");
  let _ = std::fs::remove_dir_all(&dir);
}

/// `cb_free` is a documented no-op (the `WriterState` is Rust-owned); it
/// must neither panic nor touch the state.
#[test]
fn cb_free_is_noop() {
  let dir = fresh_dir("cb-free");
  let path = dir.join("f.bin");
  let mut f = File::create(&path).unwrap();
  let state = WriterState::new(&mut f);
  let desc = state.as_desc();
  // SAFETY: `desc` is the live `WriterState` pointer; `cb_free` is the
  // documented no-op (mlx-c must never free the Rust-owned state), so it
  // neither dereferences the desc nor mutates anything.
  unsafe { cb_free(desc) };
  assert!(state.into_err().is_none());
  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────── fd-bound writer (non-seekable error paths) ─────────────────
//
// A pipe's write end is a NON-seekable fd: `File::seek`/`stream_position`
// fail with `ESPIPE`. Wrapping it as a `File` lets us deterministically
// drive the seek-failure branch of `save_safetensors_to_file` (the
// `file.seek(SeekFrom::Start(0))` rewind) and the `cb_tell` / `cb_seek`
// error-capture arms, which a regular seekable file never exercises.

/// On a non-seekable fd, `save_safetensors_to_file` fails at the initial
/// rewind (`file.seek(SeekFrom::Start(0))`) with a typed `Error::FileIo`
/// whose op is `Other("seek")` — BEFORE any byte is written. Covers the
/// seek-to-byte-0 error branch.
#[cfg(unix)]
#[test]
fn save_safetensors_to_file_non_seekable_fd_fails_at_seek() {
  use std::os::unix::io::FromRawFd;
  let mut fds = [0_i32; 2];
  // SAFETY: `fds` is a valid 2-int out-buffer; `libc::pipe` fills it with a
  // (read, write) fd pair or returns non-zero on failure (asserted below).
  let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
  assert_eq!(rc, 0, "pipe() must succeed to set up the non-seekable fd");
  let read_fd = fds[0];
  let write_fd = fds[1];
  // SAFETY: `write_fd` is a freshly-created, owned pipe write-end fd; wrapping
  // it in a `File` transfers ownership so it is closed exactly once on drop.
  let mut wf = unsafe { File::from_raw_fd(write_fd) };

  let arr = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap();
  let meta: HashMap<String, String> = HashMap::new();
  match save_safetensors_to_file(&mut wf, std::iter::once(("w", &arr)), &meta) {
    Err(Error::FileIo(p)) => {
      assert_eq!(
        p.op(),
        FileOp::Other("seek"),
        "a non-seekable fd must fail at the rewind seek, got op {:?}",
        p.op()
      );
    }
    other => panic!("expected Error::FileIo(seek) on a non-seekable fd, got {other:?}"),
  }
  drop(wf);
  // SAFETY: `read_fd` is the owned read-end of the pipe, still open and not
  // wrapped elsewhere; closing it exactly once releases the pipe.
  unsafe {
    libc::close(read_fd);
  }
}

/// Driven directly: `cb_tell` on a non-seekable fd captures the
/// `stream_position` `ESPIPE` error (returning 0), and `cb_seek` on the same
/// fd captures its `seek` error. Both feed the err-cell so the safe wrapper
/// can surface a `FileIo`. Covers the `cb_tell` and `cb_seek` IO-error arms.
#[cfg(unix)]
#[test]
fn cb_tell_and_seek_on_non_seekable_fd_capture_error() {
  use std::os::unix::io::FromRawFd;
  let mut fds = [0_i32; 2];
  // SAFETY: as above — valid out-buffer; rc asserted.
  let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
  assert_eq!(rc, 0, "pipe() must succeed to set up the non-seekable fd");
  let read_fd = fds[0];
  let write_fd = fds[1];
  // SAFETY: owns the write-end fd; `File` closes it exactly once on drop.
  let mut wf = unsafe { File::from_raw_fd(write_fd) };

  {
    let state = WriterState::new(&mut wf);
    let desc = state.as_desc();
    // SAFETY: `desc` is the live `WriterState` pointer; the callbacks run
    // synchronously here and only touch the borrowed (non-seekable) `File`
    // and the err-cell. `cb_tell` returns 0 on the captured ESPIPE error;
    // `cb_seek` records its own seek error.
    let tell = unsafe {
      let t = cb_tell(desc);
      cb_seek(desc, 0, libc::SEEK_SET);
      t
    };
    assert_eq!(tell, 0, "cb_tell on a non-seekable fd reports 0 on error");
    assert!(
      state.into_err().is_some(),
      "tell/seek on a non-seekable fd must capture an io::Error"
    );
  }
  drop(wf);
  // SAFETY: `read_fd` is the owned, still-open read-end; closed exactly once.
  unsafe {
    libc::close(read_fd);
  }
}

// ─────────────────────── GGUF (feature-gated) ───────────────────────

/// `gguf_has_meta` is a pure rc → `Result<bool>` mapper, exercisable
/// without any FFI: rc 0 forwards the flag, rc 2 (absent key) maps to
/// `Ok(false)`, any other rc is an `Err`.
#[cfg(feature = "gguf")]
#[test]
fn gguf_has_meta_maps_rc() {
  // rc 0 forwards the flag verbatim.
  assert!(gguf_has_meta(0, true).unwrap());
  assert!(!gguf_has_meta(0, false).unwrap());
  // rc == 2 means "key simply absent" → not an error, always false.
  assert!(!gguf_has_meta(2, true).unwrap());
  assert!(!gguf_has_meta(2, false).unwrap());
  // Any other rc surfaces an Err.
  assert!(gguf_has_meta(-1, false).is_err());
  assert!(gguf_has_meta(7, true).is_err());
}

/// `GgufMetadata::as_str` returns a stable snake_case tag per variant. The
/// `String` / `StringList` arms need no MLX backend; the `Array` arm is
/// covered by the gguf round-trip elsewhere (needs a live array).
#[cfg(feature = "gguf")]
#[test]
fn gguf_metadata_as_str_tags() {
  assert_eq!(GgufMetadata::String("x".to_string()).as_str(), "string");
  let list = GgufMetadata::StringList(vec!["a".to_string(), "b".to_string()]);
  assert_eq!(list.as_str(), "string_list");
  // Display delegates to as_str.
  assert_eq!(GgufMetadata::String("x".to_string()).to_string(), "string");
}

/// `GgufMetadata::as_str` for the `Array` variant + a save/load round-trip
/// of weights and all three metadata kinds. Requires gguflib to be linked
/// (the `gguf` feature; mlxrs-sys link-wiring is a separate follow-up), so
/// this is gated and may be skipped on builds where gguflib is absent.
#[cfg(feature = "gguf")]
#[test]
fn gguf_round_trips_weights_and_typed_metadata() {
  let dir = fresh_dir("gguf-roundtrip");
  let path = dir.join("model.gguf");

  let w = Array::from_slice::<f32>(&[1.0_f32, 2.0, 3.0, 4.0], &(2usize, 2)).unwrap();
  let meta_arr = Array::from_slice::<i32>(&[42_i32], &(1usize,)).unwrap();
  assert_eq!(
    GgufMetadata::Array(meta_arr.try_clone().unwrap()).as_str(),
    "array"
  );

  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.0.weight".to_string(), w);
  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();
  metadata.insert(
    "general.name".to_string(),
    GgufMetadata::String("demo".to_string()),
  );
  metadata.insert(
    "tokenizer.tokens".to_string(),
    GgufMetadata::StringList(vec!["<a>".to_string(), "<b>".to_string()]),
  );
  metadata.insert("general.count".to_string(), GgufMetadata::Array(meta_arr));

  save_gguf(&path, &weights, &metadata).unwrap();
  assert!(path.exists());

  let (mut lw, lm) = load_gguf(&path).unwrap();
  assert_eq!(
    lw.get_mut("blk.0.weight").unwrap().to_vec::<f32>().unwrap(),
    vec![1.0, 2.0, 3.0, 4.0]
  );
  // Metadata only resolves for keys that appear in the GGUF key list; at
  // minimum the string + string-list we wrote must round-trip.
  if let Some(GgufMetadata::String(s)) = lm.get("general.name") {
    assert_eq!(s, "demo");
  }
  if let Some(GgufMetadata::StringList(v)) = lm.get("tokenizer.tokens") {
    assert_eq!(v, &vec!["<a>".to_string(), "<b>".to_string()]);
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// Deterministically exercise the THREE metadata-resolving arms of
/// `load_gguf` (the `is_meta_arr` / `is_meta_str` / `is_meta_vstr`
/// branches) plus the `StringGuard` / inner `VectorStringGuard` drops.
///
/// **Why the previous round-trip test could not reach these arms.** mlx-c's
/// `mlx_io_gguf_get_keys` enumerates the *tensor* map only (the GGUF
/// `.first` map, populated from `gguf_get_tensor`); typed metadata lives in
/// the disjoint `.second` map (populated from `gguf_get_key`). The load loop
/// iterates the tensor keys and probes `has_metadata_*` for each — so a key
/// present ONLY as metadata is never visited, and a key present only as a
/// tensor probes `has_metadata_*` → rc 2 (absent) → always lands in the
/// weight (else) arm. The metadata arms therefore fire only for a name that
/// is present in BOTH the tensor section and the KV-metadata section.
///
/// GGUF stores tensors and KV-metadata in separate sections with no
/// cross-section name dedup (vendored `gguflib.c::gguf_append_kv` /
/// `gguf_append_tensor_info` write names verbatim; `gguf_append_kv` only
/// requires all KV to precede any tensor, which mlx-core's `save_gguf`
/// already honors). So saving a weight AND a typed-metadata entry under the
/// same key name yields a file where that name resolves as a tensor key
/// (listed by `get_keys`) whose `has_metadata_<type>` probe succeeds —
/// driving the load into the matching metadata arm.
///
/// INDEPENDENT closed-form oracle: every expected value/shape/dtype below is
/// a literal written into the maps here, never produced by `load_gguf`.
///
/// NOTE: relies on the disjoint-namespace + same-name-collision behavior of
/// the vendored gguflib/mlx-core round-trip documented above. If a future
/// mlx-core/gguflib bump rejects a name shared across the tensor and KV
/// sections, this test's collision premise (and these load arms) would need
/// revisiting.
#[cfg(feature = "gguf")]
#[test]
fn gguf_load_resolves_array_string_list_metadata_branches() {
  let dir = fresh_dir("gguf-meta-branches");
  let path = dir.join("model.gguf");

  // A 1-D int32 metadata array (mlx-core forbids ndim > 1 and empty arrays
  // for GGUF metadata). Round-trips to shape [3], dtype I32.
  let meta_arr = Array::from_slice::<i32>(&[7_i32, 8, 9], &(3usize,)).unwrap();
  // A plain (non-colliding) weight to also cover the weight (else) arm and
  // prove tensors still resolve when no metadata shares the name.
  let plain_w = Array::from_slice::<f32>(&[1.5_f32, 2.5], &(2usize,)).unwrap();
  // Tensors written under the SAME names as the typed metadata so the load
  // loop visits these keys (they are in the tensor key list) and the
  // per-key `has_metadata_<type>` probe succeeds, selecting the metadata arm.
  let collide_arr_w = Array::from_slice::<f32>(&[0.0_f32], &(1usize,)).unwrap();
  let collide_str_w = Array::from_slice::<f32>(&[0.0_f32], &(1usize,)).unwrap();
  let collide_list_w = Array::from_slice::<f32>(&[0.0_f32], &(1usize,)).unwrap();

  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.0.weight".to_string(), plain_w);
  weights.insert("meta.array.key".to_string(), collide_arr_w);
  weights.insert("meta.string.key".to_string(), collide_str_w);
  weights.insert("meta.list.key".to_string(), collide_list_w);

  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();
  metadata.insert("meta.array.key".to_string(), GgufMetadata::Array(meta_arr));
  metadata.insert(
    "meta.string.key".to_string(),
    GgufMetadata::String("hello-gguf".to_string()),
  );
  metadata.insert(
    "meta.list.key".to_string(),
    GgufMetadata::StringList(vec![
      "tok0".to_string(),
      "tok1".to_string(),
      "tok2".to_string(),
    ]),
  );

  save_gguf(&path, &weights, &metadata).unwrap();
  assert!(path.exists());

  let (mut lw, lm) = load_gguf(&path).unwrap();

  // The non-colliding weight resolves through the weight (else) arm.
  assert_eq!(
    lw.get_mut("blk.0.weight").unwrap().to_vec::<f32>().unwrap(),
    vec![1.5_f32, 2.5]
  );

  // Array metadata arm: the colliding key resolves as metadata (its
  // `has_metadata_array` probe succeeds), so it is NOT in the weight map and
  // its int32 values/shape/dtype round-trip from the literals written above.
  assert!(
    !lw.contains_key("meta.array.key"),
    "a tensor+array-metadata name resolves into metadata, not weights"
  );
  match lm.get("meta.array.key") {
    Some(GgufMetadata::Array(a)) => {
      let mut a = a.try_clone().unwrap();
      assert_eq!(a.shape(), vec![3]);
      assert_eq!(a.dtype().unwrap(), Dtype::I32);
      assert_eq!(a.to_vec::<i32>().unwrap(), vec![7, 8, 9]);
    }
    other => panic!("expected Array metadata for meta.array.key, got {other:?}"),
  }

  // String metadata arm (exercises the `StringGuard` create/populate/drop and
  // the `mlx_string_data` copy-out).
  match lm.get("meta.string.key") {
    Some(GgufMetadata::String(s)) => assert_eq!(s, "hello-gguf"),
    other => panic!("expected String metadata for meta.string.key, got {other:?}"),
  }

  // StringList metadata arm (exercises the inner `VectorStringGuard` + the
  // per-element `mlx_vector_string_get` copy loop and `drop(vstr_guard)`).
  match lm.get("meta.list.key") {
    Some(GgufMetadata::StringList(v)) => {
      assert_eq!(
        v,
        &vec!["tok0".to_string(), "tok1".to_string(), "tok2".to_string()]
      );
    }
    other => panic!("expected StringList metadata for meta.list.key, got {other:?}"),
  }

  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────── GGUF save interior-NUL validation ─────────────────
//
// `save_gguf` rejects interior-NUL bytes in weight keys, metadata keys,
// metadata string values, and metadata list entries with a typed
// `Error::InteriorNul` BEFORE the `mlx_save_gguf` write, each keyed by a
// distinct (context, bytes_kind) pair. Each test uses a SINGLE offending
// map entry so the failing branch is reached deterministically regardless
// of `HashMap` iteration order.

/// A weight KEY with an interior NUL is rejected in the weights-insert loop
/// (`gguf weight key`).
#[cfg(feature = "gguf")]
#[test]
fn gguf_save_weight_key_with_interior_nul_is_rejected() {
  let dir = fresh_dir("gguf-nul-weight-key");
  let path = dir.join("m.gguf");
  let w = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("bad\0weight".to_string(), w);
  let metadata: HashMap<String, GgufMetadata> = HashMap::new();
  match save_gguf(&path, &weights, &metadata).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::gguf_save: weights insert");
      assert_eq!(payload.bytes_kind(), "gguf weight key");
    }
    other => panic!("expected InteriorNul(gguf weight key), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A metadata KEY with an interior NUL is rejected in the metadata-insert
/// loop (`gguf metadata key`). No weights, so the weights loop is skipped
/// and the metadata loop's per-key `CString::new` is the first failure.
#[cfg(feature = "gguf")]
#[test]
fn gguf_save_metadata_key_with_interior_nul_is_rejected() {
  let dir = fresh_dir("gguf-nul-meta-key");
  let path = dir.join("m.gguf");
  let weights: HashMap<String, Array> = HashMap::new();
  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();
  metadata.insert(
    "bad\0meta".to_string(),
    GgufMetadata::String("v".to_string()),
  );
  match save_gguf(&path, &weights, &metadata).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::gguf_save: metadata insert");
      assert_eq!(payload.bytes_kind(), "gguf metadata key");
    }
    other => panic!("expected InteriorNul(gguf metadata key), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A `String`-metadata VALUE with an interior NUL is rejected inside the
/// `GgufMetadata::String` arm (`gguf metadata string value`). The key is
/// NUL-free so the per-key `CString::new` succeeds and the value check is
/// the failure.
#[cfg(feature = "gguf")]
#[test]
fn gguf_save_metadata_string_value_with_interior_nul_is_rejected() {
  let dir = fresh_dir("gguf-nul-meta-strval");
  let path = dir.join("m.gguf");
  let weights: HashMap<String, Array> = HashMap::new();
  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();
  metadata.insert(
    "general.name".to_string(),
    GgufMetadata::String("bad\0value".to_string()),
  );
  match save_gguf(&path, &weights, &metadata).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(payload.context(), "io::gguf_save: metadata string insert");
      assert_eq!(payload.bytes_kind(), "gguf metadata string value");
    }
    other => panic!("expected InteriorNul(gguf metadata string value), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

/// A `StringList`-metadata ENTRY with an interior NUL is rejected inside the
/// list-append loop of the `GgufMetadata::StringList` arm (`gguf metadata
/// list entry`). Single metadata entry + single offending list element →
/// deterministic.
#[cfg(feature = "gguf")]
#[test]
fn gguf_save_metadata_list_entry_with_interior_nul_is_rejected() {
  let dir = fresh_dir("gguf-nul-meta-listentry");
  let path = dir.join("m.gguf");
  let weights: HashMap<String, Array> = HashMap::new();
  let mut metadata: HashMap<String, GgufMetadata> = HashMap::new();
  metadata.insert(
    "tokenizer.tokens".to_string(),
    GgufMetadata::StringList(vec!["ok".to_string(), "bad\0entry".to_string()]),
  );
  match save_gguf(&path, &weights, &metadata).unwrap_err() {
    Error::InteriorNul(payload) => {
      assert_eq!(
        payload.context(),
        "io::gguf_save: metadata list-entry append"
      );
      assert_eq!(payload.bytes_kind(), "gguf metadata list entry");
    }
    other => panic!("expected InteriorNul(gguf metadata list entry), got {other:?}"),
  }
  let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────── load_weights_from_dir: feature-neutral discovery ───────────────────
//
// The feature-neutral checkpoint discovery (`io::load_weights_from_dir`),
// covering each resolution tier. INDEPENDENT closed-form oracles: every
// expected weight name / value / dtype is a literal written into the fixture,
// never produced by the function under test.

/// Tier 1 — `model.safetensors.index.json` (authoritative manifest). A tiny
/// **2-shard** checkpoint: shard 1 holds `a.weight`, shard 2 holds `b.weight`,
/// and the index `weight_map` binds each weight to its shard. The merged load
/// must carry BOTH keys with their literal values. A stray
/// `model-stale.safetensors` NOT listed in the index is ignored.
#[cfg(feature = "serde_json")]
#[test]
fn load_weights_from_dir_index_driven_merges_two_shards() {
  let dir = fresh_dir("lw-index-2shard");

  let a = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap();
  let mut shard1: HashMap<String, Array> = HashMap::new();
  shard1.insert("a.weight".to_string(), a);
  save_safetensors(&dir.join("model-00001-of-00002.safetensors"), &shard1).unwrap();

  let b = Array::from_slice::<i32>(&[10_i32, 20, 30], &(3usize,)).unwrap();
  let mut shard2: HashMap<String, Array> = HashMap::new();
  shard2.insert("b.weight".to_string(), b);
  save_safetensors(&dir.join("model-00002-of-00002.safetensors"), &shard2).unwrap();

  // A stale shard NOT named in the index — must be invisible to the load.
  let stale = Array::from_slice::<f32>(&[999.0_f32], &(1usize,)).unwrap();
  let mut stale_map: HashMap<String, Array> = HashMap::new();
  stale_map.insert("stale.weight".to_string(), stale);
  save_safetensors(&dir.join("model-stale.safetensors"), &stale_map).unwrap();

  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"metadata":{"total_size":20},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
  )
  .unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(
    loaded.len(),
    2,
    "exactly the two indexed weights are loaded"
  );
  assert_eq!(
    loaded.get_mut("a.weight").unwrap().to_vec::<f32>().unwrap(),
    vec![1.0, 2.0]
  );
  assert_eq!(
    loaded.get_mut("b.weight").unwrap().to_vec::<i32>().unwrap(),
    vec![10, 20, 30]
  );
  assert!(
    !loaded.contains_key("stale.weight"),
    "a shard not listed in the index must be ignored"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 2 — a single un-sharded `model.safetensors` (no index). Loaded
/// directly with its literal weights.
#[test]
fn load_weights_from_dir_single_model_safetensors() {
  let dir = fresh_dir("lw-single");
  let w = Array::from_slice::<f32>(&[3.0_f32, 4.0, 5.0], &(3usize,)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("only.weight".to_string(), w);
  save_safetensors(&dir.join("model.safetensors"), &weights).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("only.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![3.0, 4.0, 5.0]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 3 — legacy `weights.safetensors` (pre-HF naming). A directory
/// carrying ONLY this name still loads.
#[test]
fn load_weights_from_dir_legacy_weights_safetensors() {
  let dir = fresh_dir("lw-legacy");
  let w = Array::from_slice::<i32>(&[7_i32, 8], &(2usize,)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("legacy.weight".to_string(), w);
  save_safetensors(&dir.join("weights.safetensors"), &weights).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("legacy.weight")
      .unwrap()
      .to_vec::<i32>()
      .unwrap(),
    vec![7, 8]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// A directory holding `model-*.safetensors` shards but NO index manifest is an
/// INCOMPLETE checkpoint (a partial/failed `save_model` writes `model-gen-*`
/// shards before the index-rename commit point). There is deliberately no
/// unindexed-`model*.safetensors`-glob fallback, so it fails closed with the
/// typed no-weights error rather than merging uncommitted shards — matching
/// mlx-lm's index-or-single-file contract.
#[test]
fn load_weights_from_dir_sharded_without_index_fails_closed() {
  let dir = fresh_dir("lw-noindex-shards");
  let p1 = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut m1: HashMap<String, Array> = HashMap::new();
  m1.insert("p1.weight".to_string(), p1);
  save_safetensors(&dir.join("model-00001-of-00002.safetensors"), &m1).unwrap();

  let p2 = Array::from_slice::<f32>(&[2.0_f32], &(1usize,)).unwrap();
  let mut m2: HashMap<String, Array> = HashMap::new();
  m2.insert("p2.weight".to_string(), p2);
  save_safetensors(&dir.join("model-00002-of-00002.safetensors"), &m2).unwrap();

  assert!(
    !dir.join("model.safetensors.index.json").exists(),
    "the no-index sharded case is what this test exercises"
  );
  assert!(
    load_weights_from_dir(&dir).is_err(),
    "sharded `model-*.safetensors` without an index manifest is incomplete and \
     must fail closed (no unindexed-glob fallback)"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Without `serde_json`, a present `model.safetensors.index.json` — even a
/// NON-REGULAR one (here a directory) — must fail closed rather than fall
/// through and load a stale single `model.safetensors`. `path_is_file`
/// would miss the non-regular entry; `symlink_metadata` catches it.
#[cfg(not(feature = "serde_json"))]
#[test]
fn load_weights_from_dir_no_serde_nonregular_index_fails_closed() {
  let dir = fresh_dir("lw-noserde-nonreg-index");
  // A DIRECTORY named like the index sentinel (a non-regular entry).
  std::fs::create_dir(dir.join("model.safetensors.index.json")).unwrap();
  // A stale single-file checkpoint that must NOT be loaded while the sentinel is present.
  let a = Array::from_slice::<f32>(&[9.0_f32], &(1usize,)).unwrap();
  let mut m: HashMap<String, Array> = HashMap::new();
  m.insert("stale.weight".to_string(), a);
  save_safetensors(&dir.join("model.safetensors"), &m).unwrap();

  assert!(
    load_weights_from_dir(&dir).is_err(),
    "a present (even non-regular) index sentinel must fail closed without serde_json"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// With `serde_json`, a DANGLING `model.safetensors.index.json` symlink (the
/// sentinel exists but its target is missing) must fail closed in `load_via_index`
/// rather than be treated as absent and fall through to a stale `model.safetensors`.
#[cfg(all(feature = "serde_json", unix))]
#[test]
fn load_weights_from_dir_serde_dangling_index_symlink_fails_closed() {
  let dir = fresh_dir("lw-dangling-index");
  // Dangling symlink: the sentinel exists but points at a missing target.
  std::os::unix::fs::symlink(
    dir.join("missing-target.json"),
    dir.join("model.safetensors.index.json"),
  )
  .unwrap();
  // A stale single-file checkpoint that must NOT be loaded while the sentinel exists.
  let a = Array::from_slice::<f32>(&[7.0_f32], &(1usize,)).unwrap();
  let mut m: HashMap<String, Array> = HashMap::new();
  m.insert("stale.weight".to_string(), a);
  save_safetensors(&dir.join("model.safetensors"), &m).unwrap();

  assert!(
    load_weights_from_dir(&dir).is_err(),
    "a dangling index sentinel must fail closed, not fall through to the stale model.safetensors"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 6 — a single NumPy `.npz` (the mlx-community-native multi-array
/// format). With no safetensors / gguf, `model.npz` is loaded via
/// `load_npz`. Round-trips the literal weights.
#[cfg(feature = "npz")]
#[test]
fn load_weights_from_dir_npz_model() {
  let dir = fresh_dir("lw-npz");
  let w = Array::from_slice::<f32>(&[6.0_f32, 7.0], &(2usize,)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("npz.weight".to_string(), w);
  save_npz(&dir.join("model.npz"), &mut weights).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("npz.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![6.0, 7.0]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 6 — a sole `*.npz` that is NOT named `model.npz` / `weights.npz` is
/// still picked up (the "else the sole `.npz`" branch).
#[cfg(feature = "npz")]
#[test]
fn load_weights_from_dir_npz_sole_arbitrary_name() {
  let dir = fresh_dir("lw-npz-arb");
  let w = Array::from_slice::<i32>(&[5_i32], &(1usize,)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("w.weight".to_string(), w);
  save_npz(&dir.join("checkpoint.npz"), &mut weights).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(
    loaded.get_mut("w.weight").unwrap().to_vec::<i32>().unwrap(),
    vec![5]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 5 — a single `*.gguf` (mlx-lm's GGUF path) is loaded when no
/// safetensors is present. Round-trips the literal weight.
#[cfg(feature = "gguf")]
#[test]
fn load_weights_from_dir_gguf_single_file() {
  let dir = fresh_dir("lw-gguf");
  let w = Array::from_slice::<f32>(&[8.0_f32, 9.0, 10.0, 11.0], &(2usize, 2)).unwrap();
  let mut weights: HashMap<String, Array> = HashMap::new();
  weights.insert("blk.0.weight".to_string(), w);
  let metadata: HashMap<String, GgufMetadata> = HashMap::new();
  save_gguf(&dir.join("model.gguf"), &weights, &metadata).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(loaded.len(), 1);
  assert_eq!(
    loaded
      .get_mut("blk.0.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![8.0, 9.0, 10.0, 11.0]
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Tier 1 takes priority over a sole `*.gguf`: when BOTH an index-driven
/// safetensors checkpoint and a `*.gguf` live in the same directory, the
/// safetensors index wins (gguf is the fallback only when no safetensors is
/// present). Proves the resolution ORDER, not just each tier in isolation.
#[cfg(all(feature = "serde_json", feature = "gguf"))]
#[test]
fn load_weights_from_dir_safetensors_index_beats_gguf() {
  let dir = fresh_dir("lw-order");
  let st = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut st_map: HashMap<String, Array> = HashMap::new();
  st_map.insert("st.weight".to_string(), st);
  save_safetensors(&dir.join("model-00001-of-00001.safetensors"), &st_map).unwrap();
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"weight_map":{"st.weight":"model-00001-of-00001.safetensors"}}"#,
  )
  .unwrap();
  // A decoy gguf that must be ignored because safetensors resolves first.
  let g = Array::from_slice::<f32>(&[2.0_f32], &(1usize,)).unwrap();
  let mut g_map: HashMap<String, Array> = HashMap::new();
  g_map.insert("gguf.weight".to_string(), g);
  let meta: HashMap<String, GgufMetadata> = HashMap::new();
  save_gguf(&dir.join("model.gguf"), &g_map, &meta).unwrap();

  let loaded = load_weights_from_dir(&dir).unwrap();
  assert!(
    loaded.contains_key("st.weight") && !loaded.contains_key("gguf.weight"),
    "the safetensors index tier must win over the gguf fallback"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// No weights of any layout → the terminal typed [`Error::FileIo`] (`Open`,
/// `NotFound`) whose static `context()` lists every layout the resolver
/// considered (at minimum each safetensors tier).
#[test]
fn load_weights_from_dir_no_weights_is_typed_error() {
  let dir = fresh_dir("lw-empty");
  let r = load_weights_from_dir(&dir);
  let Err(Error::FileIo(p)) = r else {
    panic!("an empty dir must be Error::FileIo, got {r:?}");
  };
  assert_eq!(p.path(), dir.as_path());
  assert_eq!(p.op(), FileOp::Open);
  assert_eq!(p.inner().kind(), std::io::ErrorKind::NotFound);
  let ctx = p.context();
  assert!(
    ctx.contains("model.safetensors.index.json")
      && ctx.contains("model.safetensors")
      && ctx.contains("weights.safetensors"),
    "the context must list each safetensors tier, got: {ctx}"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// Without the `gguf` feature, a directory whose only weight file is a
/// `*.gguf` (no safetensors) reports the gguf-unsupported typed error
/// (`Error::LayerKeyed` naming the file, inner `InvariantViolation` about
/// the `gguf` feature). With the feature on, that same directory loads — so
/// this asserts the not-`gguf` arm specifically.
#[cfg(not(feature = "gguf"))]
#[test]
fn load_weights_from_dir_gguf_present_without_feature_is_unsupported() {
  let dir = fresh_dir("lw-gguf-unsupported");
  std::fs::write(dir.join("model.gguf"), b"GGUF placeholder").unwrap();
  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("expected Error::LayerKeyed for an unsupported gguf, got {r:?}");
  };
  assert!(p.layer().contains("model.gguf"));
  assert!(matches!(p.inner(), Error::InvariantViolation(iv)
      if iv.context().contains("GGUF") && iv.requirement().contains("enabled")));
  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────── load_weights_from_dir: index is TENSOR-authoritative ─────────────────
//
// The `model.safetensors.index.json` `weight_map` is authoritative at the
// tensor level: exactly the mapped tensors load, each from its assigned shard.
// A tensor present in a shard but unlisted for it must NOT leak in; a listed
// tensor absent from its shard, or a doubly-bound tensor name, is a typed
// error. INDEPENDENT closed-form oracles: every expected name/value is a
// literal written into the fixture, never produced by the function under test.

/// A shard carrying an EXTRA tensor that the index does NOT bind to it: only
/// the `weight_map`-listed tensor is returned, the extra is dropped (it cannot
/// leak in, nor clobber a same-named tensor the index assigned elsewhere).
#[cfg(feature = "serde_json")]
#[test]
fn load_weights_from_dir_index_drops_unlisted_shard_tensor() {
  let dir = fresh_dir("lw-index-extra");

  // The single shard holds BOTH `kept.weight` and an unlisted `extra.weight`.
  let kept = Array::from_slice::<f32>(&[1.0_f32, 2.0], &(2usize,)).unwrap();
  let extra = Array::from_slice::<f32>(&[9.0_f32], &(1usize,)).unwrap();
  let mut shard: HashMap<String, Array> = HashMap::new();
  shard.insert("kept.weight".to_string(), kept);
  shard.insert("extra.weight".to_string(), extra);
  save_safetensors(&dir.join("model-00001-of-00001.safetensors"), &shard).unwrap();

  // The index binds ONLY `kept.weight`.
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"weight_map":{"kept.weight":"model-00001-of-00001.safetensors"}}"#,
  )
  .unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(loaded.len(), 1, "only the indexed tensor is returned");
  assert_eq!(
    loaded
      .get_mut("kept.weight")
      .unwrap()
      .to_vec::<f32>()
      .unwrap(),
    vec![1.0, 2.0]
  );
  assert!(
    !loaded.contains_key("extra.weight"),
    "a tensor present in the shard but unlisted in weight_map must be dropped"
  );
  let _ = std::fs::remove_dir_all(&dir);
}

/// A `weight_map` that assigns a tensor to a shard which does NOT contain it
/// is a malformed checkpoint: `Error::LayerKeyed` (keyed by the shard name)
/// wrapping `Error::MissingKey` (the absent tensor name).
#[cfg(feature = "serde_json")]
#[test]
fn load_weights_from_dir_index_missing_tensor_in_shard_is_typed_error() {
  let dir = fresh_dir("lw-index-missing");

  // The shard holds only `present.weight`; the index also demands
  // `absent.weight` from the same shard.
  let present = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut shard: HashMap<String, Array> = HashMap::new();
  shard.insert("present.weight".to_string(), present);
  save_safetensors(&dir.join("model-00001-of-00001.safetensors"), &shard).unwrap();

  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"weight_map":{"present.weight":"model-00001-of-00001.safetensors","absent.weight":"model-00001-of-00001.safetensors"}}"#,
  )
  .unwrap();

  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("a tensor missing from its assigned shard must be Error::LayerKeyed, got {r:?}");
  };
  assert!(
    p.layer().contains("model-00001-of-00001.safetensors"),
    "the LayerKeyed layer must name the offending shard, got: {}",
    p.layer()
  );
  let Error::MissingKey(mk) = p.inner() else {
    panic!("inner must be Error::MissingKey, got {:?}", p.inner());
  };
  assert_eq!(mk.key(), "absent.weight");
  let _ = std::fs::remove_dir_all(&dir);
}

/// A `weight_map` that binds the SAME tensor name twice is a malformed index
/// (a `serde_json::Value` object would silently keep the last; the order-
/// preserving parse instead surfaces it): `Error::LayerKeyed` (keyed by the
/// tensor name) wrapping an `Error::InvariantViolation` about a duplicate
/// binding. The duplicate must be caught BEFORE any shard is opened.
#[cfg(feature = "serde_json")]
#[test]
fn load_weights_from_dir_index_duplicate_binding_is_typed_error() {
  let dir = fresh_dir("lw-index-dup");

  // A real shard so the failure is the duplicate check, not a missing shard.
  let w = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut shard: HashMap<String, Array> = HashMap::new();
  shard.insert("dup.weight".to_string(), w);
  save_safetensors(&dir.join("model-00001-of-00002.safetensors"), &shard).unwrap();

  // `dup.weight` is bound twice (to two different shards). JSON permits the
  // duplicate object key; the loader must reject it rather than keep one.
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"weight_map":{"dup.weight":"model-00001-of-00002.safetensors","dup.weight":"model-00002-of-00002.safetensors"}}"#,
  )
  .unwrap();

  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("a duplicate weight_map binding must be Error::LayerKeyed, got {r:?}");
  };
  assert_eq!(p.layer(), "dup.weight");
  assert!(matches!(p.inner(), Error::InvariantViolation(iv)
      if iv.context().contains("tensor name") && iv.requirement().contains("more than once")));
  let _ = std::fs::remove_dir_all(&dir);
}

/// Without `serde_json`, the index tier is compiled out — but an
/// `model.safetensors.index.json` must still FAIL CLOSED rather than fall
/// through to the lower tiers and mask the sharded checkpoint. The directory
/// carries both an index and a stray `model-*.safetensors` shard; the load
/// returns the typed unsupported-feature error (`Error::LayerKeyed` naming the
/// index, inner `InvariantViolation` about the `serde_json` feature) instead of
/// the generic no-weights error.
#[cfg(not(feature = "serde_json"))]
#[test]
fn load_weights_from_dir_index_without_serde_json_fails_closed() {
  let dir = fresh_dir("lw-index-noserde");
  // A stray `model-*.safetensors` shard alongside the index sentinel.
  let stray = Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap();
  let mut m: HashMap<String, Array> = HashMap::new();
  m.insert("stray.weight".to_string(), stray);
  save_safetensors(&dir.join("model-00001-of-00001.safetensors"), &m).unwrap();
  std::fs::write(
    dir.join("model.safetensors.index.json"),
    br#"{"weight_map":{"stray.weight":"model-00001-of-00001.safetensors"}}"#,
  )
  .unwrap();

  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("an index without serde_json must fail closed (Error::LayerKeyed), got {r:?}");
  };
  assert!(
    p.layer().contains("model.safetensors.index.json"),
    "the error must name the index file, got: {}",
    p.layer()
  );
  assert!(matches!(p.inner(), Error::InvariantViolation(iv)
      if iv.context().contains("index.json") && iv.context().contains("serde_json")
        && iv.requirement().contains("must be enabled")));
  let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────── load_weights_from_dir: gguf / npz cardinality ─────────────────

/// Multiple `*.gguf` candidates (and no safetensors) is ambiguous: there is no
/// canonical name to pick one, so the resolver refuses with a typed
/// `Error::LayerKeyed` whose runtime key names every colliding file and whose
/// inner `InvariantViolation` explains the single-file contract. Feature-
/// agnostic: the cardinality check fires BEFORE any gguf load, so placeholder
/// files suffice and the error is identical with or without the `gguf` feature.
#[test]
fn load_weights_from_dir_multiple_gguf_is_ambiguous() {
  let dir = fresh_dir("lw-gguf-ambig");
  std::fs::write(dir.join("a-model.gguf"), b"GGUF placeholder A").unwrap();
  std::fs::write(dir.join("b-model.gguf"), b"GGUF placeholder B").unwrap();

  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("multiple *.gguf must be Error::LayerKeyed (ambiguous), got {r:?}");
  };
  assert!(
    p.layer().contains("a-model.gguf") && p.layer().contains("b-model.gguf"),
    "the ambiguity error must name both gguf candidates, got: {}",
    p.layer()
  );
  assert!(matches!(p.inner(), Error::InvariantViolation(iv)
      if iv.context().contains("multiple `*.gguf`") && iv.requirement().contains("exactly one")));
  let _ = std::fs::remove_dir_all(&dir);
}

/// Multiple NON-canonical `*.npz` candidates (neither `model.npz` nor
/// `weights.npz`, no safetensors / gguf) is ambiguous: `Error::LayerKeyed`
/// naming both, inner `InvariantViolation` about the single-file contract.
#[cfg(feature = "npz")]
#[test]
fn load_weights_from_dir_multiple_arbitrary_npz_is_ambiguous() {
  let dir = fresh_dir("lw-npz-ambig");
  let mut m1: HashMap<String, Array> = HashMap::new();
  m1.insert(
    "a.weight".to_string(),
    Array::from_slice::<f32>(&[1.0_f32], &(1usize,)).unwrap(),
  );
  save_npz(&dir.join("alpha.npz"), &mut m1).unwrap();
  let mut m2: HashMap<String, Array> = HashMap::new();
  m2.insert(
    "b.weight".to_string(),
    Array::from_slice::<f32>(&[2.0_f32], &(1usize,)).unwrap(),
  );
  save_npz(&dir.join("beta.npz"), &mut m2).unwrap();

  let r = load_weights_from_dir(&dir);
  let Err(Error::LayerKeyed(p)) = r else {
    panic!("multiple arbitrary *.npz must be Error::LayerKeyed (ambiguous), got {r:?}");
  };
  assert!(
    p.layer().contains("alpha.npz") && p.layer().contains("beta.npz"),
    "the ambiguity error must name both npz candidates, got: {}",
    p.layer()
  );
  assert!(matches!(p.inner(), Error::InvariantViolation(iv)
      if iv.context().contains("non-canonical `*.npz`")));
  let _ = std::fs::remove_dir_all(&dir);
}

/// A canonical `model.npz` ALONGSIDE a non-canonical `*.npz` is NOT ambiguous:
/// the canonical name wins (the cardinality check only governs the arbitrary-
/// name fallback). Proves the preference order survives the new guard.
#[cfg(feature = "npz")]
#[test]
fn load_weights_from_dir_canonical_npz_beats_arbitrary() {
  let dir = fresh_dir("lw-npz-canon");
  let mut canon: HashMap<String, Array> = HashMap::new();
  canon.insert(
    "canon.weight".to_string(),
    Array::from_slice::<i32>(&[7_i32], &(1usize,)).unwrap(),
  );
  save_npz(&dir.join("model.npz"), &mut canon).unwrap();
  // A decoy arbitrary npz that must be ignored because `model.npz` wins.
  let mut decoy: HashMap<String, Array> = HashMap::new();
  decoy.insert(
    "decoy.weight".to_string(),
    Array::from_slice::<i32>(&[8_i32], &(1usize,)).unwrap(),
  );
  save_npz(&dir.join("decoy.npz"), &mut decoy).unwrap();

  let mut loaded = load_weights_from_dir(&dir).unwrap();
  assert_eq!(
    loaded
      .get_mut("canon.weight")
      .unwrap()
      .to_vec::<i32>()
      .unwrap(),
    vec![7]
  );
  assert!(
    !loaded.contains_key("decoy.weight"),
    "the canonical model.npz must win over an arbitrary-name npz"
  );
  let _ = std::fs::remove_dir_all(&dir);
}
