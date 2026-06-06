//! Model IO — safetensors, GGUF, and NumPy `.npy` / `.npz` load/save.
//!
//! safetensors + GGUF are thin wrappers over mlx-c `io.h`; the NumPy formats
//! are parsed in Rust (the `npy` submodule, behind the `npz` feature)
//! mirroring MLX core's own byte format. Local-file IO only; no HF-hub
//! download.
//!
//! - safetensors: a map of named arrays plus an optional `String -> String`
//!   metadata side-table. Mirrors `mlx.core.load/save_safetensors` and
//!   mlx-swift `loadArrays` / `loadArraysAndMetadata` / `save`.
//! - GGUF: a map of named tensors plus typed metadata entries (array /
//!   string / list-of-strings), mirroring mlx-c's `mlx_io_gguf` API and
//!   `mlx.core.load_gguf` (returns weights + metadata).
//! - NumPy (`npz` feature): `.npy` a single array, `.npz` a ZIP archive of
//!   `<name>.npy` members (the mlx-community-native multi-array weight
//!   format). Mirrors `mlx.core.load` / `mlx.core.save` / `savez` /
//!   `savez_compressed`. See the `npy` submodule (`load_npy` / `load_npz` /
//!   `save_npy` / `save_npz` / `save_npz_compressed`).
//!
//! Validation (bad file, missing key, dtype quirks) is left to mlx-c for
//! safetensors/GGUF and done with bounds-checked Rust parsing for NumPy; both
//! surface through [`Result`]. See
//! [mlx io docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html).

/// NumPy `.npy` / `.npz` array IO — load/save what `mx.save` / `mx.savez`
/// write, with a bounds-checked Rust parser mirroring MLX's byte format.
#[cfg(feature = "npz")]
#[cfg_attr(docsrs, doc(cfg(feature = "npz")))]
pub mod npy;

#[cfg(feature = "npz")]
#[cfg_attr(docsrs, doc(cfg(feature = "npz")))]
pub use npy::{load_npy, load_npz, save_npy, save_npz, save_npz_compressed};

use std::{
  cell::Cell,
  collections::HashMap,
  ffi::{CStr, CString},
  fs::File,
  io::{Seek, SeekFrom, Write},
  os::raw::{c_char, c_int, c_void},
  path::Path,
};

use crate::{
  array::Array,
  error::{Error, FileIoPayload, FileOp, InteriorNulPayload, Result, check},
};

thread_local! {
  static CPU_STREAM: Cell<Option<mlxrs_sys::mlx_stream>> = const { Cell::new(None) };
}

/// Per-thread CPU stream for IO ops. `mlx_load*` materialize through
/// `Load::eval_gpu`, which is unimplemented in mlx-c — loads must run on a
/// CPU stream (matches mlx-swift's `StreamOrDevice = .cpu` default for IO).
/// Pattern mirrors `crate::ops::linalg_full::linalg_cpu_stream`: lazy
/// per-thread init, never freed (CPU stream teardown can crash at exit).
fn io_cpu_stream() -> mlxrs_sys::mlx_stream {
  crate::error::ensure_handler_installed();
  // Honor the #13 cleared-thread poison contract (as `default_stream()` /
  // `Stream::default_cpu()` do): a CPU-routed op on a poisoned thread must
  // fail fast, not continue into mlx with torn-down stream state.
  crate::stream::assert_streams_not_cleared();
  CPU_STREAM.with(|cell| {
    if let Some(s) = cell.get() {
      return s;
    }
    // SAFETY: `mlx_default_cpu_stream_new()` returns the thread's default CPU
    // stream handle; the error handler is installed first (above) and the
    // NULL-ctx case is checked just below before the handle is cached/used.
    let s = unsafe { mlxrs_sys::mlx_default_cpu_stream_new() };
    if s.ctx.is_null() {
      panic!(
        "mlxrs::io: mlx_default_cpu_stream_new returned NULL ctx — \
         CPU stream initialization failed. Aborting."
      );
    }
    cell.set(Some(s));
    s
  })
}

// ─────────────────────────── helpers ───────────────────────────

/// Convert a path to a NUL-terminated C string, rejecting embedded NULs.
fn path_cstring(path: &Path) -> Result<CString> {
  CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
    let _ = path;
    Error::InteriorNul(InteriorNulPayload::new("io::path_cstring", "path"))
  })
}

/// RAII guard for a temporary `mlx_map_string_to_array`.
struct ArrayMapGuard(mlxrs_sys::mlx_map_string_to_array);
impl Drop for ArrayMapGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a defined
    // no-op on a NULL ctx, so a sentinel handle from a failed `_new()` is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently per
    // the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_map_string_to_array_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_map_string_to_string`.
struct StringMapGuard(mlxrs_sys::mlx_map_string_to_string);
impl Drop for StringMapGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a defined
    // no-op on a NULL ctx, so a sentinel handle from a failed `_new()` is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently per
    // the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_map_string_to_string_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_io_gguf`.
#[cfg(feature = "gguf")]
struct GgufGuard(mlxrs_sys::mlx_io_gguf);
#[cfg(feature = "gguf")]
impl Drop for GgufGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a defined
    // no-op on a NULL ctx, so a sentinel handle from a failed `_new()` is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently per
    // the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_io_gguf_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_vector_string`.
#[cfg(feature = "gguf")]
struct VectorStringGuard(mlxrs_sys::mlx_vector_string);
#[cfg(feature = "gguf")]
impl Drop for VectorStringGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a defined
    // no-op on a NULL ctx, so a sentinel handle from a failed `_new()` is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently per
    // the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_vector_string_free(self.0);
    }
  }
}

/// RAII guard for a temporary `mlx_string`.
#[cfg(feature = "gguf")]
struct StringGuard(mlxrs_sys::mlx_string);
#[cfg(feature = "gguf")]
impl Drop for StringGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a defined
    // no-op on a NULL ctx, so a sentinel handle from a failed `_new()` is safe.
    // Runs during `Drop` / thread teardown: must not touch TLS, call `check()`,
    // panic, or unwind across `extern "C"`; the rc is discarded silently per
    // the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_string_free(self.0);
    }
  }
}

/// `mlx_io_gguf_has_metadata_*` returns `2` when the key is simply absent
/// from the GGUF metadata map — that is NOT an error (a weight-only key,
/// the common case). Map `2` to `Ok(false)`, `0` to `Ok(flag)`, and any
/// other rc to a backend error via [`check`].
#[cfg(feature = "gguf")]
fn gguf_has_meta(rc: std::os::raw::c_int, flag: bool) -> Result<bool> {
  match rc {
    0 => Ok(flag),
    2 => Ok(false),
    _ => {
      check(rc)?;
      Ok(false) // unreachable: `check` returns `Err` for any non-zero rc here
    }
  }
}

/// Drain an `mlx_map_string_to_array` into a `HashMap<String, Array>`.
fn drain_array_map(map: mlxrs_sys::mlx_map_string_to_array) -> HashMap<String, Array> {
  // SAFETY: `map` is a valid populated handle that the caller's guard keeps
  // alive for the whole of this call. mlx-c's iterator borrows the map (it
  // stores `&cpp_map` internally), so the map MUST outlive `it` — guaranteed
  // because `it` is created and freed entirely within this function while the
  // caller still owns the map. On allocation failure mlx-c returns a NULL-ctx
  // iterator and raises an mlx error; the first `..._next` then returns
  // non-zero (caught internally), so the loop below exits without UB.
  let it = unsafe { mlxrs_sys::mlx_map_string_to_array_iterator_new(map) };
  let mut out = HashMap::new();
  loop {
    let mut key: *const std::os::raw::c_char = std::ptr::null();
    // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
    // ctx) per the mlx-c convention; it is wrapped in the RAII newtype FIRST
    // so a `break`/early drop frees it, then populated by the next call.
    let mut value = Array(unsafe { mlxrs_sys::mlx_array_new() });
    // SAFETY: `it` is the valid iterator from above; `&mut key` / `&mut
    // value.0` are valid out-params (mlx-c writes a borrowed `*key` pointer
    // into the map's live `std::string` and copies the entry into the
    // freshly-allocated `value`). Not retained by mlx past the call.
    let rc =
      unsafe { mlxrs_sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value.0, it) };
    // mlx-c iterators return non-zero once exhausted.
    if rc != 0 {
      break;
    }
    // SAFETY: on `rc == 0`, `key` is the non-NULL pointer mlx-c just wrote; it
    // points at a NUL-terminated buffer inside the map's `std::string` key,
    // owned by the still-live `map` for the duration of this borrow. The
    // `into_owned()` copies it out before the next iteration / map free.
    let k = unsafe { CStr::from_ptr(key) }
      .to_string_lossy()
      .into_owned();
    out.insert(k, value);
  }
  // SAFETY: `it` is the valid iterator from above, freed exactly once here
  // (the map it borrowed is still alive, owned by the caller); rc discarded.
  unsafe {
    let _ = mlxrs_sys::mlx_map_string_to_array_iterator_free(it);
  }
  out
}

/// Drain an `mlx_map_string_to_string` into a `HashMap<String, String>`.
fn drain_string_map(map: mlxrs_sys::mlx_map_string_to_string) -> HashMap<String, String> {
  // SAFETY: `map` is a valid populated handle the caller's guard keeps alive
  // for the whole of this call. mlx-c's iterator borrows the map (stores
  // `&cpp_map`), so the map MUST outlive `it` — guaranteed because `it` is
  // created and freed within this function while the caller still owns the
  // map. On allocation failure mlx-c returns a NULL-ctx iterator and raises
  // an mlx error; the first `..._next` then returns non-zero (caught
  // internally), so the loop below exits without UB.
  let it = unsafe { mlxrs_sys::mlx_map_string_to_string_iterator_new(map) };
  let mut out = HashMap::new();
  loop {
    let mut key: *const std::os::raw::c_char = std::ptr::null();
    let mut value: *const std::os::raw::c_char = std::ptr::null();
    // SAFETY: `it` is the valid iterator from above; `&mut key` / `&mut value`
    // are valid out-params into which mlx-c writes borrowed pointers aimed at
    // the map's live `std::string` key/value (not retained past the call).
    let rc = unsafe { mlxrs_sys::mlx_map_string_to_string_iterator_next(&mut key, &mut value, it) };
    if rc != 0 {
      break;
    }
    // SAFETY: on `rc == 0`, `key` is the non-NULL pointer mlx-c just wrote,
    // pointing at a NUL-terminated buffer inside the map's `std::string` key,
    // owned by the still-live `map`; `into_owned()` copies it out before the
    // next iteration / map free.
    let k = unsafe { CStr::from_ptr(key) }
      .to_string_lossy()
      .into_owned();
    // SAFETY: as above for `value` — non-NULL on `rc == 0`, NUL-terminated,
    // backed by the map's live `std::string` value; copied out immediately.
    let v = unsafe { CStr::from_ptr(value) }
      .to_string_lossy()
      .into_owned();
    out.insert(k, v);
  }
  // SAFETY: `it` is the valid iterator from above, freed exactly once here
  // (the map it borrowed is still alive, owned by the caller); rc discarded.
  unsafe {
    let _ = mlxrs_sys::mlx_map_string_to_string_iterator_free(it);
  }
  out
}

/// Build a temporary `mlx_map_string_to_array` from any iterator of borrowed
/// `(name, array)` pairs. Caller wraps the returned handle in an
/// [`ArrayMapGuard`].
///
/// Generic over the entry iterator so both an owned `&HashMap<String, Array>`
/// (via `HashMap`'s `(&String, &Array)` iterator) and a borrowed shard view
/// (`&HashMap<&str, &Array>`) feed the same map-builder without cloning any
/// `Array` — the shard-save path ([`save_safetensors_view`]) needs the
/// no-clone form.
fn build_array_map<'a, I>(arrays: I) -> Result<mlxrs_sys::mlx_map_string_to_array>
where
  I: IntoIterator<Item = (&'a str, &'a Array)>,
{
  // Install the mlx-c error handler BEFORE the `_new()` call so a
  // `std::bad_alloc` (or any other exception) caught by the constructor's
  // try/catch surfaces into `crate::error::LAST` via `mlx_error(e.what())`
  // rather than mlx-c's default `printf + exit(-1)`. The constructor then
  // returns a sentinel handle with `ctx == nullptr` (vendored
  // `mlx-c/mlx/c/map.cpp::mlx_map_string_to_array_new` line 10-17), which
  // the NULL-check below drains into `Err` so the empty-input case (no
  // `_insert` calls inside the loop below) cannot silently propagate
  // a useless empty handle to a downstream FFI call.
  crate::error::ensure_handler_installed();
  // SAFETY: `mlx_map_string_to_array_new()` returns a fresh empty map handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`),
  // wrapped in an `ArrayMapGuard` IMMEDIATELY so any `?` below (interior-NUL
  // key, insert allocation failure) frees the partially-built map. On success
  // ownership is transferred to the caller via `mem::forget` (suppressing
  // this guard's `Drop`); the caller re-wraps the returned raw handle.
  let guard = ArrayMapGuard(unsafe { mlxrs_sys::mlx_map_string_to_array_new() });
  // Reject the NULL-ctx sentinel from a failed `_new()` before the caller
  // can act on a useless empty handle. Drain `LAST` (NOT peek — leaving
  // a stale `Err` in the TLS would poison the next unrelated mlx-c call
  // on this thread).
  if guard.0.ctx.is_null() {
    let last = crate::error::take_last();
    return Err(last.unwrap_or(Error::FfiNullHandle(
      crate::error::FfiNullHandlePayload::new("mlx_map_string_to_array_new"),
    )));
  }
  for (k, v) in arrays {
    let ck = CString::new(k).map_err(|_| {
      let _ = k;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::map_arrays insert",
        "array key",
      ))
    })?;
    // SAFETY: `guard.0` is the valid handle from above; `ck.as_ptr()` is a
    // valid NUL-terminated C string that outlives the call (`ck` still in
    // scope); `v.0` is a valid borrowed `mlx_array`. mlx-c copies the key
    // into a `std::string` and the array via `insert_or_assign`, retaining
    // neither pointer past the call; the rc is surfaced via `check()` (an
    // `Err` here drops `guard`, freeing the partial map — no leak).
    check(unsafe { mlxrs_sys::mlx_map_string_to_array_insert(guard.0, ck.as_ptr(), v.0) })?;
  }
  let raw = guard.0;
  std::mem::forget(guard);
  Ok(raw)
}

/// Build a temporary `mlx_map_string_to_string` from a Rust map. Caller wraps
/// the returned handle in a [`StringMapGuard`].
fn build_string_map(meta: &HashMap<String, String>) -> Result<mlxrs_sys::mlx_map_string_to_string> {
  // Install the mlx-c error handler BEFORE the `_new()` call so a
  // `std::bad_alloc` (or any other exception) caught by the constructor's
  // try/catch surfaces into `crate::error::LAST` via `mlx_error(e.what())`
  // rather than mlx-c's default `printf + exit(-1)`. The constructor then
  // returns a sentinel handle with `ctx == nullptr` (vendored
  // `mlx-c/mlx/c/map.cpp::mlx_map_string_to_string_new` line 119-126),
  // which the NULL-check below drains into `Err`. The empty-`HashMap`
  // metadata case (the common call site for `save_safetensors`) makes no
  // `_insert` calls inside the loop below, so without this guard an
  // allocation-failure sentinel would silently return `Ok(NULL)`.
  crate::error::ensure_handler_installed();
  // SAFETY: `mlx_map_string_to_string_new()` returns a fresh empty map handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`),
  // wrapped in a `StringMapGuard` IMMEDIATELY so any `?` below (interior-NUL
  // key/value, insert allocation failure) frees the partially-built map. On
  // success ownership is transferred to the caller via `mem::forget`
  // (suppressing this guard's `Drop`); the caller re-wraps the raw handle.
  let guard = StringMapGuard(unsafe { mlxrs_sys::mlx_map_string_to_string_new() });
  // Reject the NULL-ctx sentinel from a failed `_new()` before the loop
  // runs — when `meta` is empty (the no-metadata `save_safetensors`
  // path) no `_insert` call would be made, so an allocation-failure
  // sentinel would otherwise pass through `Ok(NULL)` to the caller.
  // Drain `LAST` (NOT peek — leaving a stale `Err` in the TLS would
  // poison the next unrelated mlx-c call on this thread).
  if guard.0.ctx.is_null() {
    let last = crate::error::take_last();
    return Err(last.unwrap_or(Error::FfiNullHandle(
      crate::error::FfiNullHandlePayload::new("mlx_map_string_to_string_new"),
    )));
  }
  for (k, v) in meta {
    let ck = CString::new(k.as_str()).map_err(|_| {
      let _ = k;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::map_meta insert",
        "metadata key",
      ))
    })?;
    let cv = CString::new(v.as_str()).map_err(|_| {
      let _ = v;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::map_meta insert",
        "metadata value",
      ))
    })?;
    // SAFETY: `guard.0` is the valid handle from above; `ck`/`cv` are valid
    // NUL-terminated C strings still in scope for the call. mlx-c copies both
    // into `std::string`s via `insert_or_assign`, retaining neither pointer
    // past the call; the rc is surfaced via `check()` (an `Err` here drops
    // `guard`, freeing the partial map — no leak).
    check(unsafe {
      mlxrs_sys::mlx_map_string_to_string_insert(guard.0, ck.as_ptr(), cv.as_ptr())
    })?;
  }
  let raw = guard.0;
  std::mem::forget(guard);
  Ok(raw)
}

// ─────────────────────────── safetensors ───────────────────────────

/// Load a `.safetensors` file into a map of named arrays, discarding metadata.
///
/// Mirrors `mlx.core.load` / mlx-swift `loadArrays`.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Array>> {
  Ok(load_safetensors_with_metadata(path)?.0)
}

/// Load a `.safetensors` file, returning `(arrays, metadata)`.
///
/// Mirrors mlx-swift `loadArraysAndMetadata` / `mlx.core.load(..., return_metadata=True)`.
pub fn load_safetensors_with_metadata(
  path: &Path,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
  let cpath = path_cstring(path)?;
  // SAFETY: each `_new()` returns a fresh empty map handle (NULL ctx on
  // allocation failure, a defined-safe input to `_free`). Both are wrapped in
  // their RAII guards (below) BEFORE the fallible `mlx_load_safetensors` so an
  // early `?` frees them.
  let mut arrays = unsafe { mlxrs_sys::mlx_map_string_to_array_new() };
  // SAFETY: as above for the string-to-string metadata map.
  let mut meta = unsafe { mlxrs_sys::mlx_map_string_to_string_new() };
  let arrays_guard = ArrayMapGuard(arrays);
  let meta_guard = StringMapGuard(meta);
  // SAFETY: `&mut arrays` / `&mut meta` are out-params holding the freshly
  // allocated handles already owned by the guards above; mlx-c fills them
  // in-place (`mlx_map_*_set_(*res, ...)` mutates the existing ctx rather
  // than replacing the handle, so the guards still free the right objects).
  // `cpath` is a valid NUL-terminated path string live for the call;
  // `io_cpu_stream()` is a valid CPU stream; the rc is surfaced via `check()`.
  check(unsafe {
    mlxrs_sys::mlx_load_safetensors(&mut arrays, &mut meta, cpath.as_ptr(), io_cpu_stream())
  })?;
  let a = drain_array_map(arrays);
  let m = drain_string_map(meta);
  drop(arrays_guard);
  drop(meta_guard);
  Ok((a, m))
}

/// Save a map of named arrays to a `.safetensors` file (no metadata).
pub fn save_safetensors(path: &Path, arrays: &HashMap<String, Array>) -> Result<()> {
  save_safetensors_with_metadata(path, arrays, &HashMap::new())
}

/// Save a map of named arrays plus `String -> String` metadata to a
/// `.safetensors` file.
pub fn save_safetensors_with_metadata(
  path: &Path,
  arrays: &HashMap<String, Array>,
  metadata: &HashMap<String, String>,
) -> Result<()> {
  save_safetensors_view(path, arrays.iter().map(|(k, v)| (k.as_str(), v)), metadata)
}

/// Save an arbitrary borrowed `(name, array)` view plus `String -> String`
/// metadata to a `.safetensors` file — the no-clone shard-write primitive.
///
/// Generalizes [`save_safetensors_with_metadata`] over the entry iterator so
/// a sub-map of borrowed arrays (a shard, `HashMap<&str, &Array>` — see
/// [`crate::lm::load::make_shards`]) can be written **without** refcount-
/// cloning every `Array` into a fresh owned `HashMap<String, Array>` first.
/// `save_safetensors_with_metadata` is the owned-map convenience wrapper over
/// this. Behavior is otherwise identical to
/// [`save_safetensors_with_metadata`]: the named arrays + metadata are
/// handed to `mlx_save_safetensors` on the IO CPU stream.
///
/// **TOCTOU note.** This entry point creates / truncates `path` by name via
/// mlx-c's path-taking `mlx_save_safetensors`, so it must NOT be used as
/// part of a same-directory "stage to `O_EXCL` tempfile, then rename"
/// flow that wants to keep the original-open identity: between the
/// `O_EXCL` create + this reopen-by-name, an attacker with directory
/// write access could `unlink(path) + symlink(path, /etc/passwd)` and
/// redirect the write. Atomic-staging code paths
/// ([`crate::lm::load::save_model`], [`crate::lm::load::save_config`])
/// instead use the fd-bound [`save_safetensors_to_file`] which writes
/// through an already-open [`File`].
pub fn save_safetensors_view<'a, I>(
  path: &Path,
  arrays: I,
  metadata: &HashMap<String, String>,
) -> Result<()>
where
  I: IntoIterator<Item = (&'a str, &'a Array)>,
{
  let cpath = path_cstring(path)?;
  let amap = build_array_map(arrays)?;
  let amap_guard = ArrayMapGuard(amap);
  let mmap = build_string_map(metadata)?;
  let mmap_guard = StringMapGuard(mmap);
  // SAFETY: `cpath` is a valid NUL-terminated path string live for the call;
  // `amap` / `mmap` are valid populated map handles owned by the guards above
  // and kept alive across this call. mlx-c reads them by const reference and
  // retains nothing past the call; the rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_save_safetensors(cpath.as_ptr(), amap, mmap) })?;
  drop(amap_guard);
  drop(mmap_guard);
  Ok(())
}

/// Save an arbitrary borrowed `(name, array)` view plus metadata to an
/// **already-open** [`File`] — the **fd-bound** safetensors writer.
///
/// Same surface as [`save_safetensors_view`] but pinned to the caller's
/// own [`File`] handle instead of a path. Internally builds a custom
/// `mlx_io_writer` whose vtable delegates `is_open`/`good`/`tell`/`seek`/
/// `write` to the supplied `&mut File`, then hands it to mlx-c's
/// `mlx_save_safetensors_writer`. The `File` is borrowed for the call only
/// — mlx-c performs an eager `eval` + synchronous writes inside
/// `save_safetensors_writer` (see vendored
/// `mlx/io/safetensors.cpp::save_safetensors(writer, ...)`), so no callback
/// is ever invoked after this function returns.
///
/// **Why a `&mut File` rather than a path** — closes the TOCTOU window
/// `O_EXCL`-created-then-reopened-by-name leaves open. Callers in
/// [`crate::lm::load`]'s atomic-save path stage shards to same-directory
/// `O_EXCL` tempfiles and then must continue to write through the
/// `O_EXCL` open's identity (an attacker who can write the destination
/// directory could otherwise `unlink + symlink` the path between the
/// `O_EXCL` create + a reopen-by-name and redirect the write to e.g.
/// `/etc/passwd`). The path-taking [`save_safetensors_view`] remains for
/// direct path-based saves where the caller accepts the path semantics
/// (creates / truncates the target).
///
/// # Destructive mutation
///
/// This function destructively mutates the file. On `Err`, the file may be
/// in any of these states:
///
/// * **Untouched** — if Err occurred during input validation (interior NUL
///   bytes in array names or metadata, NULL-sentinel from the mlx-c map or
///   writer constructors). These early-validation Errs are returned before
///   the file is truncated, so the prior contents are preserved as a
///   defense-in-depth side effect. This is NOT a contract: callers MUST
///   NOT rely on byte preservation across save failures.
/// * **Partially mutated or zero-length** — if Err occurred during
///   `mlx_save_safetensors_writer` (eager `eval` failure, MLX-internal
///   rejection of the array set such as zero-element arrays, header-build
///   failure, or any error returned by the underlying write callbacks).
///   The file has been truncated to zero and may contain a partial
///   safetensors header.
///
/// **For write-redirection-safe staging in an atomic-replace flow**, use
/// the fd-bound tempfile-staging pattern (the open/write/fsync/drop
/// steps below are exemplified by [`crate::lm::load::save_model`] in
/// `mlxrs/src/lm/load.rs:1359-1372`):
///
/// 1. Open a [`File`] for a tempfile in the SAME directory as the target
///    (so the eventual rename stays atomic on the same filesystem), e.g.
///    via `OpenOptions::new().create_new(true).write(true).open(...)`
///    with a unique tempfile name like `target.tmp.<rand>`.
/// 2. Pass that `&mut File` to `save_safetensors_to_file(...)`.
/// 3. On success, `file.sync_all()?` then
///    `std::fs::rename(temp_path, target_path)?` (or
///    `std::fs::hard_link` + unlink for atomic no-replace publish, as
///    [`crate::lm::load::save_model`] does).
/// 4. On error, the temp file is destructively mutated (per the contract
///    above) but the original target file is untouched. Unlink the temp.
///
/// **Scope of this guarantee.** The fd-bound `&mut File` argument
/// protects the WRITE PATH: an attacker with directory write access
/// cannot redirect the bytes via `unlink + symlink` between when this
/// function rewinds + truncates + writes the safetensors payload,
/// because every write goes through the caller-owned fd rather than
/// reopening by name. The SUBSEQUENT publication step in the recipe
/// above (`std::fs::rename(temp_path, ...)` or
/// `std::fs::hard_link(temp_path, ...) + unlink(temp_path)`) operates
/// by PATHNAME and is therefore still subject to directory-entry
/// races: an attacker with write access to the staging directory can
/// `unlink(temp_path)` and substitute their own file at the same
/// name; the subsequent rename / hard_link then atomically publishes
/// the attacker's inode rather than the one this function wrote. The
/// full attack window is the lifetime of the temp NAME —
/// substitution can occur ANY TIME after the `O_EXCL` create and
/// before publication, not only after fsync. The fd-bound write
/// itself remains safe (every byte goes to the inode the caller
/// holds), but the temp directory entry is no longer bound to that
/// inode.
///
/// Avoiding this requires ONE of:
///
/// * **A trusted staging directory** (one that is not user-writable)
///   — the simplest and most portable solution. The publication step
///   is safe because no attacker can substitute the temp entry.
/// * **Platform-specific publish-by-fd primitives** that link the open
///   file descriptor (or an unnamed temp inode) into the target name in
///   one step. The exact requirements are non-trivial and OS-specific
///   (Linux's `O_TMPFILE` + `linkat(AT_EMPTY_PATH)` has multiple
///   preconditions; macOS has no equivalent). **This crate does NOT
///   provide such a primitive.** Callers needing this property must
///   either implement it directly against their OS's syscalls (consult
///   the relevant man-pages for the full constraint set) or use a
///   security-audited library that explicitly documents fd-bound
///   publication semantics. Path-based "atomic persist" APIs (including
///   ones in popular crates) do NOT satisfy this property — they persist
///   by pathname and remain vulnerable to the temp-name substitution
///   race documented above.
///
/// Note that `openat`-family syscalls with a directory file
/// descriptor (e.g. `renameat`, `linkat` by name) DO NOT close this
/// race: they anchor the parent directory but still look up the
/// mutable temp entry by name, so an attacker who can unlink and
/// replace `temp_path` can still cause the substituted inode to be
/// published. Neither of the two safe options above is provided by
/// this API. **Do NOT use the path-taking
/// [`save_safetensors_view`] for atomic replacement** — that API
/// reopens by name and permits `unlink + symlink` write redirection
/// in hostile directories (see its docstring's TOCTOU note); it is
/// appropriate ONLY for callers who accept path-reopen semantics.
///
/// The `&mut File` API exists specifically for callers who need fd-bound
/// semantics (e.g. TOCTOU mitigation when the target path is
/// attacker-controllable, or writes to seekable file descriptors that
/// lack a stable pathname like memfds created via `memfd_create(2)`).
/// The descriptor must be **seekable** — non-seekable descriptors
/// (pipes, sockets, ttys) deterministically fail at the `seek(0)` step
/// below before any write. For non-seekable targets, save to a regular
/// file first and stream the bytes separately.
///
/// Returns an error if any of the fallible setup steps fails (interior-NUL
/// validation, the map ctors, or `mlx_io_writer_new()` itself), if the
/// rewind / truncate fails, if the underlying `File` write fails (surfaced
/// through the captured `WriterState::err`), or if mlx-c raises
/// (surfaced via the installed error handler).
pub fn save_safetensors_to_file<'a, I>(
  file: &mut File,
  arrays: I,
  metadata: &HashMap<String, String>,
) -> Result<()>
where
  I: IntoIterator<Item = (&'a str, &'a Array)>,
{
  // Validation runs first as a defense-in-depth side effect; on early-
  // validation Err the file remains untouched. This is NOT a contract —
  // see the "Destructive mutation" section of the doc comment above. The
  // ordering keeps interior-NUL Errs (`build_array_map` /
  // `build_string_map`) from truncating a caller-owned prefilled file
  // before surfacing the error.
  let amap = build_array_map(arrays)?;
  let amap_guard = ArrayMapGuard(amap);
  let mmap = build_string_map(metadata)?;
  let mmap_guard = StringMapGuard(mmap);
  // Defense-in-depth: build the `mlx_io_writer` before the destructive
  // truncate so an allocation failure inside the vendored
  // `mlx_io_writer_new_` ctor (which catches `std::bad_alloc` and returns
  // a `mlx_io_writer({nullptr})` sentinel — vendored
  // `mlx-c/mlx/c/io_types.cpp:48-54`) is surfaced as Err before the
  // caller's file is mutated. Wrapped in `WriterGuard` immediately so a
  // NULL or mid-function `?` frees the partial handle.
  //
  // `WriterState::new(file)` reborrows `&mut File` only to cast it to
  // `*mut File`; the reborrow ends at function return, so `file` is
  // re-usable as `&mut File` for the `seek`/`set_len` below (the raw
  // pointer in `state.file` is only dereferenced from inside the
  // vtable callbacks invoked by `mlx_save_safetensors_writer`, which
  // runs strictly after both `seek` and `set_len` complete).
  let state = WriterState::new(file);
  // Install the mlx-c error handler BEFORE `mlx_io_writer_new` so a
  // `std::bad_alloc` caught by the constructor's try/catch surfaces
  // into `crate::error::LAST` via `mlx_error(e.what())` rather than
  // mlx-c's default `printf + exit(-1)`. The NULL-ctx branch below
  // then drains that captured message into the returned `Err`.
  crate::error::ensure_handler_installed();
  // SAFETY: `state.as_desc()` returns a `*mut c_void` aliasing the local
  // `WriterState`; it must outlive the `mlx_io_writer` that captures it.
  // We build the writer + immediately wrap it in a `WriterGuard` so any `?`
  // below frees the writer (which DOES drop the shared_ptr<CWriter> — but
  // CWriter only holds `desc + vtable` by value and never calls
  // `vtable.free(desc)` on our side because we set it to a no-op). The
  // `state` local outlives BOTH `writer_guard` (and thus any callback
  // mlx-c could invoke) AND the entire `save_safetensors_writer` call —
  // by the time `state` goes out of scope, the writer + its
  // `shared_ptr<CWriter>` are already freed (writer_guard drop above).
  // `mlx_io_writer_free` is a defined no-op on a NULL-ctx sentinel
  // (vendored `mlx-c/mlx/c/private/io.h:138-142` checks `io.ctx` first),
  // so the guard is safe to install unconditionally.
  let writer = unsafe { mlxrs_sys::mlx_io_writer_new(state.as_desc(), make_writer_vtable()) };
  let writer_guard = WriterGuard(writer);
  // Defense-in-depth: surface a NULL-ctx sentinel from a failed
  // `mlx_io_writer_new` before the destructive truncate. Drain `LAST`
  // (NOT peek — leaving a stale `Err` in the TLS would poison the next
  // unrelated mlx-c call on this thread). The drop order at the early
  // return is: writer_guard (frees the NULL-ctx sentinel — defined
  // no-op), then mmap_guard, then amap_guard (Rust drop is reverse
  // declaration order); `state` is Drop-less.
  if writer_guard.0.ctx.is_null() {
    let last = crate::error::take_last();
    return Err(last.unwrap_or(Error::FfiNullHandle(
      crate::error::FfiNullHandlePayload::new("mlx_io_writer_new"),
    )));
  }
  // Now that every fallible Rust- and FFI-level setup step has confirmed
  // Ok / non-NULL handles, rewind the file to byte 0 and truncate to a
  // clean canvas. Without these, a prefilled `File` handed in at a
  // non-zero cursor would receive a prefix-corrupted safetensors (mlx-c's
  // `cb_write` writes at the current cursor), and a prefilled `File`
  // longer than the new payload would retain stale trailing bytes after
  // the new (shorter) safetensors. Both surfaces are propagated as
  // `Error::Backend` with the same `save_safetensors_to_file:` prefix
  // the rest of this function uses, so logs stay greppable.
  //
  // After this point the file IS destructively mutated — see the
  // "Destructive mutation" section of the doc comment. `seek` /
  // `set_len` can themselves `Err` (partial OS-level mutation is
  // possible), and `mlx_save_safetensors_writer` can `Err` mid-stream
  // (eval failure, MLX-internal rejection, write-callback failure).
  // Callers that need atomic-replace semantics should use the fd-bound
  // tempfile-staging pattern (open a same-directory `O_EXCL` `File`,
  // write through it, `sync_all`, then `rename` / `hard_link` to the
  // final path) — see `save_model` in `mlxrs/src/lm/load.rs:1359-1372`
  // and the "Destructive mutation" section of this function's doc
  // comment. Do NOT route through the path-taking
  // `save_safetensors_view` for atomic replacement: it reopens by name
  // and reintroduces the TOCTOU window this API was built to close.
  file.seek(SeekFrom::Start(0)).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save_safetensors_to_file: seek to byte 0",
      FileOp::Other("seek"),
      std::path::PathBuf::new(),
      e,
    ))
  })?;
  file.set_len(0).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "save_safetensors_to_file: truncate to 0",
      FileOp::Other("set_len"),
      std::path::PathBuf::new(),
      e,
    ))
  })?;
  // SAFETY: `writer` is the valid populated handle owned by `writer_guard`,
  // valid for the duration of this call. `amap` / `mmap` are valid
  // populated map handles owned by their guards above. mlx-c reads all
  // three by const reference; rc surfaced via `check()`. On error the
  // guards free everything in reverse-construction order.
  let rc = unsafe { mlxrs_sys::mlx_save_safetensors_writer(writer, amap, mmap) };
  // Drop the writer FIRST (mlx-c's `mlx_io_writer_free` destroys the
  // C++ `shared_ptr<CWriter>` that aliases `state`), THEN check the rc
  // and surface any captured `state.err`. This ordering guarantees no
  // further callback into `state` is possible while we still hold the
  // borrow.
  drop(writer_guard);
  drop(amap_guard);
  drop(mmap_guard);
  // A captured io::Error from the write callback takes precedence over the
  // mlx-c rc (mlx-c will have raised once the write failed; rc != 0).
  if let Some(e) = state.into_err() {
    return Err(Error::FileIo(FileIoPayload::new(
      "save_safetensors_to_file: write callback",
      FileOp::Write,
      std::path::PathBuf::new(),
      e,
    )));
  }
  check(rc)?;
  Ok(())
}

// ─────────────────────── mlx_io_writer backed by &mut File ───────────────────────

/// State the writer-vtable callbacks operate on: a borrowed [`File`] plus a
/// cell capturing the first [`std::io::Error`] any write/seek hit (so the
/// caller can surface it after the FFI call returns). Layout is opaque on
/// the C side (mlx-c only sees the `*mut c_void` we hand it via
/// [`Self::as_desc`]; the callbacks cast back to `*mut WriterState`), so
/// the default Rust layout suffices.
struct WriterState {
  /// `*mut File` rather than `&mut File` so the field is `Copy` + the type
  /// stays trivially `Send`-checkable; the borrow is enforced at the
  /// API surface by `&mut File`, not at the field level.
  file: *mut File,
  /// First IO error from any callback, captured here so
  /// [`save_safetensors_to_file`] can surface it after the FFI call.
  err: std::cell::Cell<Option<std::io::Error>>,
  /// Static C-string label returned by the `label` callback — included in
  /// mlx-c's "[save_safetensors] Failed to open ..." error messages.
  label: &'static CStr,
}

impl WriterState {
  fn new(file: &mut File) -> Self {
    Self {
      file: file as *mut File,
      err: std::cell::Cell::new(None),
      label: c"mlxrs::io::save_safetensors_to_file(&mut File)",
    }
  }

  fn as_desc(&self) -> *mut c_void {
    (self as *const Self as *mut Self).cast::<c_void>()
  }

  fn into_err(self) -> Option<std::io::Error> {
    self.err.into_inner()
  }

  /// Set the captured error IFF none was already captured (so the FIRST
  /// failure wins — subsequent callbacks may also fail because the file
  /// is now in a bad state, but the original cause is what matters).
  fn set_err(&self, e: std::io::Error) {
    let prev = self.err.take();
    self.err.set(prev.or(Some(e)));
  }
}

/// RAII guard for an `mlx_io_writer`.
struct WriterGuard(mlxrs_sys::mlx_io_writer);
impl Drop for WriterGuard {
  fn drop(&mut self) {
    // SAFETY: frees a handle this guard owns exactly once. `_free` is a
    // defined no-op on a NULL ctx, so a sentinel handle from a failed
    // `_new()` is safe. Runs during `Drop` / thread teardown: must not
    // touch TLS, call `check()`, panic, or unwind across `extern "C"`;
    // the rc is discarded silently per the crate's Drop convention.
    unsafe {
      let _ = mlxrs_sys::mlx_io_writer_free(self.0);
    }
  }
}

// vtable callback panic-safety contract: every callback is reached from
// mlx-c (`extern "C"`), so a Rust panic crossing the FFI boundary is UB.
// Each callback wraps its body in `catch_unwind` and converts a caught
// panic into `state.set_err(...)` — the FFI call then either short-
// circuits (writes that follow turn the `File` into a bad state mlx-c
// notices) or completes with a captured error the safe wrapper surfaces
// before returning.

/// `WriterState::with_state(desc, f)` — recover the borrowed `&WriterState`
/// from the opaque desc pointer mlx-c hands the callback, and run `f`
/// inside `catch_unwind`. A caught panic stores a synthetic
/// `io::ErrorKind::Other` into `state.err` and returns `None`.
fn with_state<R>(desc: *mut c_void, f: impl FnOnce(&WriterState, &mut File) -> R) -> Option<R> {
  let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    // SAFETY: `desc` was set by `WriterState::as_desc` to a `*mut
    // WriterState` aliasing a local that outlives this callback (callers
    // of `save_safetensors_to_file` hold the `WriterState` on the stack
    // for the whole `mlx_save_safetensors_writer` call). The same pointer
    // is what every callback receives; we materialize `&WriterState` for
    // the err-cell access. The inner `*mut File` is then re-borrowed as
    // `&mut File` for the duration of `f` — mlx-c's safetensors writer is
    // single-threaded inside the call, so no two callbacks alias the file
    // at the same time.
    let state = unsafe { &*(desc as *const WriterState) };
    // SAFETY: as above; `state.file` is the `*mut File` derived from the
    // `&mut File` the caller passed to `save_safetensors_to_file`; that
    // borrow is exclusive for the duration of the FFI call, so we can
    // safely materialize a `&mut File` here. Callbacks never re-enter
    // each other (single-threaded inside mlx-c's safetensors writer).
    let file = unsafe { &mut *state.file };
    f(state, file)
  }));
  match result {
    Ok(r) => Some(r),
    Err(_) => {
      // Capture a synthetic error so the safe wrapper surfaces a
      // panic-in-callback as an `Error::Backend` rather than silently
      // succeeding. We cannot resume_unwind across the FFI boundary.
      // SAFETY: same recover-state pattern as above; only the err-cell
      // is touched (no &mut File).
      let state = unsafe { &*(desc as *const WriterState) };
      state.set_err(std::io::Error::other(
        "mlxrs::io::save_safetensors_to_file callback panicked",
      ));
      None
    }
  }
}

unsafe extern "C" fn cb_is_open(_desc: *mut c_void) -> bool {
  // The File is open by construction (caller owns it); always true. No
  // syscall, so panic-free.
  true
}

unsafe extern "C" fn cb_good(desc: *mut c_void) -> bool {
  // `good` in C++ iostream semantics = "no error state". We model it as
  // "no captured error". A captured error means a previous callback hit
  // an IO failure; mlx-c will see `!good()` and abort the save. Peek at
  // the err-cell without consuming the captured error.
  with_state(desc, |state, _file| {
    let prev = state.err.take();
    let is_good = prev.is_none();
    state.err.set(prev);
    is_good
  })
  .unwrap_or(false)
}

unsafe extern "C" fn cb_tell(desc: *mut c_void) -> usize {
  with_state(desc, |state, file| match file.stream_position() {
    Ok(p) => p as usize,
    Err(e) => {
      state.set_err(e);
      0
    }
  })
  .unwrap_or(0)
}

unsafe extern "C" fn cb_seek(desc: *mut c_void, off: i64, whence: c_int) {
  with_state(desc, |state, file| {
    // Map POSIX whence -> SeekFrom. The vendored `private/io.h` translates
    // `std::ios_base::seekdir` -> `SEEK_SET`/`SEEK_CUR`/`SEEK_END`.
    let pos = match whence {
      x if x == libc::SEEK_SET => SeekFrom::Start(off as u64),
      x if x == libc::SEEK_CUR => SeekFrom::Current(off),
      x if x == libc::SEEK_END => SeekFrom::End(off),
      _ => {
        state.set_err(std::io::Error::other(format!(
          "save_safetensors_to_file: unknown seek whence {whence}"
        )));
        return;
      }
    };
    if let Err(e) = file.seek(pos) {
      state.set_err(e);
    }
  });
}

unsafe extern "C" fn cb_read(desc: *mut c_void, _data: *mut c_char, _n: usize) {
  // A writer should never be asked to read; capture the misuse.
  with_state(desc, |state, _file| {
    state.set_err(std::io::Error::other(
      "save_safetensors_to_file: writer.read called (writer-only sink)",
    ));
  });
}

unsafe extern "C" fn cb_read_at_offset(
  desc: *mut c_void,
  _data: *mut c_char,
  _n: usize,
  _off: usize,
) {
  with_state(desc, |state, _file| {
    state.set_err(std::io::Error::other(
      "save_safetensors_to_file: writer.read_at_offset called (writer-only sink)",
    ));
  });
}

unsafe extern "C" fn cb_write(desc: *mut c_void, data: *const c_char, n: usize) {
  with_state(desc, |state, file| {
    if n == 0 {
      return;
    }
    if data.is_null() {
      state.set_err(std::io::Error::other(
        "save_safetensors_to_file: write callback received NULL data",
      ));
      return;
    }
    // SAFETY: mlx-c's safetensors writer hands us a non-NULL `data`
    // pointer to a contiguous run of `n` bytes (the JSON header bytes
    // then each `arr.data<char>()` of `arr.nbytes()`); the pointer is
    // valid for the duration of the synchronous callback only and is
    // not retained by us.
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, n) };
    if let Err(e) = file.write_all(bytes) {
      state.set_err(e);
    }
  });
}

unsafe extern "C" fn cb_label(desc: *mut c_void) -> *const c_char {
  // Best-effort static label for mlx-c's "Failed to open ..." error
  // formatting. Returning a pointer into `state.label` is safe because
  // the WriterState (and its `&'static CStr`) outlive every callback.
  with_state(desc, |state, _file| state.label.as_ptr()).unwrap_or(c"<panic>".as_ptr())
}

unsafe extern "C" fn cb_free(_desc: *mut c_void) {
  // The `WriterState` is owned by the Rust caller (stack-allocated in
  // `save_safetensors_to_file`); mlx-c MUST NOT free it. This is the
  // explicit no-op contract the `mlx_io_vtable.free` slot accepts.
}

fn make_writer_vtable() -> mlxrs_sys::mlx_io_vtable {
  mlxrs_sys::mlx_io_vtable {
    is_open: Some(cb_is_open),
    good: Some(cb_good),
    tell: Some(cb_tell),
    seek: Some(cb_seek),
    read: Some(cb_read),
    read_at_offset: Some(cb_read_at_offset),
    write: Some(cb_write),
    label: Some(cb_label),
    free: Some(cb_free),
  }
}

// ─────────────────────────── GGUF ───────────────────────────
//
// GGUF load/save is gated behind the `gguf` cargo feature (off by default —
// opt-in for the GGUF dep weight in the link line + the public surface).
// `mlxrs-sys/build.rs` links `gguflib` unconditionally (a self-contained
// ld64 archive built by MLX core's FetchContent), so non-`gguf` binaries
// pull no `gguf_*` objects (the linker only loads members that resolve
// referenced symbols) and the default build is byte-for-byte unaffected.

/// A typed GGUF metadata entry. GGUF metadata values are one of: a scalar/
/// tensor [`Array`], a string, or a list of strings (matches mlx-c's
/// `mlx_io_gguf_*_metadata_*` accessors).
#[cfg(feature = "gguf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gguf")))]
#[non_exhaustive]
#[derive(
  Debug, derive_more::Display, derive_more::IsVariant, derive_more::Unwrap, derive_more::TryUnwrap,
)]
#[display("{}", self.as_str())]
#[unwrap(ref, ref_mut)]
#[try_unwrap(ref, ref_mut)]
pub enum GgufMetadata {
  /// An array-valued metadata entry.
  Array(Array),
  /// A string-valued metadata entry.
  String(String),
  /// A list-of-strings metadata entry.
  StringList(Vec<String>),
}

#[cfg(feature = "gguf")]
impl GgufMetadata {
  /// Stable snake_case tag for the active variant — single source of truth
  /// for [`std::fmt::Display`], log keys, and error messages. Data-carrying
  /// enums get `as_str -> &str` (non-`const`); the returned strings are
  /// `&'static str` here only because every variant maps to a literal, not
  /// because `as_str` itself is `const`.
  pub fn as_str(&self) -> &'static str {
    match self {
      Self::Array(_) => "array",
      Self::String(_) => "string",
      Self::StringList(_) => "string_list",
    }
  }
}

/// Load a `.gguf` file, returning `(weights, metadata)`.
///
/// Mirrors `mlx.core.load_gguf`. mlx-c does not expose metadata-key
/// enumeration, so the returned metadata map only carries entries whose
/// keys also appear in the GGUF key list and resolve via the typed
/// `has_metadata_*` probes.
#[cfg(feature = "gguf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gguf")))]
pub fn load_gguf(path: &Path) -> Result<(HashMap<String, Array>, HashMap<String, GgufMetadata>)> {
  let cpath = path_cstring(path)?;
  // Seed the guard with a NULL-ctx sentinel (NOT `mlx_io_gguf_new()`, which
  // heap-allocates an empty `GGUFLoad`). This makes ownership airtight
  // through mlx-c's non-allocation-safe `mlx_io_gguf_set_`: with `d.ctx ==
  // null`, `set_`'s `if (d.ctx) delete` is skipped — it NEVER deletes, only
  // `d.ctx = new GGUFLoad(...)` on success — so there is no delete-before-new
  // window on ANY path. Constructing the `repr(C)` handle is safe; only
  // *using* the ctx pointer (next call) is `unsafe`.
  let mut guard = GgufGuard(mlxrs_sys::mlx_io_gguf {
    ctx: std::ptr::null_mut(),
  });
  // SAFETY: `&mut guard.0` is the null-seeded out-param owned by `guard`.
  // `mlx_load_gguf` → `mlx_io_gguf_set_` (vendored `private/gguf.h`): since
  // `guard.0.ctx` is null it does NOT delete, only `d.ctx = new GGUFLoad`
  // on success. So at drop the guard owns exactly one of: the post-load
  // handle (success → freed once) or still-null (if `load_gguf` throws, or
  // `set_`'s `new` throws before assigning — nothing was deleted), and
  // `mlx_io_gguf_free` is a defined no-op on a null ctx. No path double-frees
  // or leaks. `cpath` is a valid NUL-terminated path live for the call;
  // `io_cpu_stream()` is a valid CPU stream; rc surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_load_gguf(&mut guard.0, cpath.as_ptr(), io_cpu_stream()) })?;
  // Borrowed handle for the read-only accessors below; `guard` remains the
  // sole owner/freer of the (post-load) ctx.
  let gguf = guard.0;

  // SAFETY: `mlx_vector_string_new()` returns a fresh empty vector handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`); wrapped
  // in `VectorStringGuard` (below) BEFORE the fallible `get_keys`.
  let mut keys = unsafe { mlxrs_sys::mlx_vector_string_new() };
  let keys_guard = VectorStringGuard(keys);
  // SAFETY: `&mut keys` is an out-param holding the freshly allocated vector
  // owned by `keys_guard`; mlx-c overwrites it in-place with the GGUF key set.
  // `gguf` is the valid borrowed handle populated above, not retained past
  // the call; the rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_io_gguf_get_keys(&mut keys, gguf) })?;
  // SAFETY: `keys` is the valid populated vector from above; mlx-c returns its
  // plain element count and does not mutate or retain it.
  let n = unsafe { mlxrs_sys::mlx_vector_string_size(keys) };

  let mut weights = HashMap::new();
  let mut metadata = HashMap::new();
  for i in 0..n {
    let mut raw: *mut std::os::raw::c_char = std::ptr::null_mut();
    // SAFETY: `&mut raw` is a valid out-pointer; `keys` is the valid vector
    // from above and `i < n` is in range. mlx-c writes into `raw` a borrowed
    // pointer to the `i`-th `std::string`'s buffer inside the still-live
    // `keys` vector (not retained past the call); rc surfaced via `check()`.
    check(unsafe { mlxrs_sys::mlx_vector_string_get(&mut raw, keys, i) })?;
    // SAFETY: on `rc == 0`, `raw` is the non-NULL pointer mlx-c just wrote,
    // aimed at a NUL-terminated buffer inside `keys`'s `std::string`, owned by
    // the still-live `keys` (its guard is dropped only after this loop);
    // `into_owned()` copies it out before any further mutation.
    let key = unsafe { CStr::from_ptr(raw) }
      .to_string_lossy()
      .into_owned();
    let ckey = CString::new(key.as_str()).map_err(|_| {
      let _ = &key;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::gguf_load: key lookup",
        "gguf key",
      ))
    })?;

    let mut f_arr = false;
    // SAFETY: `&mut f_arr` is a valid `bool` out-param; `gguf` is the valid
    // borrowed handle from above; `ckey.as_ptr()` is a valid NUL-terminated
    // C string live for the call (`ckey` still in scope). mlx-c retains
    // nothing past the call; the rc (0 / 2 / error) is mapped by
    // `gguf_has_meta` (rc 2 = key absent, not an error).
    let rc_arr =
      unsafe { mlxrs_sys::mlx_io_gguf_has_metadata_array(&mut f_arr, gguf, ckey.as_ptr()) };
    let is_meta_arr = gguf_has_meta(rc_arr, f_arr)?;
    let mut f_str = false;
    // SAFETY: as above — valid `bool` out-param, valid borrowed `gguf`, valid
    // in-scope NUL-terminated `ckey`; nothing retained; rc via `gguf_has_meta`.
    let rc_str =
      unsafe { mlxrs_sys::mlx_io_gguf_has_metadata_string(&mut f_str, gguf, ckey.as_ptr()) };
    let is_meta_str = gguf_has_meta(rc_str, f_str)?;
    let mut f_vstr = false;
    // SAFETY: as above — valid `bool` out-param, valid borrowed `gguf`, valid
    // in-scope NUL-terminated `ckey`; nothing retained; rc via `gguf_has_meta`.
    let rc_vstr = unsafe {
      mlxrs_sys::mlx_io_gguf_has_metadata_vector_string(&mut f_vstr, gguf, ckey.as_ptr())
    };
    let is_meta_vstr = gguf_has_meta(rc_vstr, f_vstr)?;

    if is_meta_arr {
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
      // ctx) per the mlx-c convention; wrapped in the RAII newtype FIRST so an
      // early `?` frees it, then populated by the next call.
      let mut arr = Array(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: `&mut arr.0` is the freshly allocated out-param from above;
      // `gguf` is the valid borrowed handle; `ckey` is a valid in-scope
      // NUL-terminated C string. mlx-c copies the entry into `arr` and retains
      // nothing past the call; the rc is surfaced via `check()`.
      check(unsafe { mlxrs_sys::mlx_io_gguf_get_metadata_array(&mut arr.0, gguf, ckey.as_ptr()) })?;
      metadata.insert(key, GgufMetadata::Array(arr));
    } else if is_meta_str {
      // SAFETY: `mlx_string_new()` returns a fresh empty out-param `mlx_string`
      // (NULL ctx) per the mlx-c convention; wrapped in the `StringGuard` RAII
      // newtype FIRST so an early `?` / panic frees it exactly once, then
      // populated by the next call.
      let mut s = StringGuard(unsafe { mlxrs_sys::mlx_string_new() });
      // SAFETY: `&mut s.0` is the fresh `mlx_string` out-param from above;
      // `gguf` is the valid borrowed handle; `ckey` is a valid in-scope
      // NUL-terminated C string. mlx-c overwrites `s.0` with the metadata
      // string and retains nothing past the call; the rc is surfaced via
      // `check()` — an `Err` here drops `s`, freeing it (no leak).
      check(unsafe { mlxrs_sys::mlx_io_gguf_get_metadata_string(&mut s.0, gguf, ckey.as_ptr()) })?;
      // SAFETY: reaching here means `check` passed, so mlx-c wrote a valid
      // `std::string` into `s.0`; `mlx_string_data` then returns that string's
      // `c_str()` — a non-NULL, NUL-terminated buffer owned by the still-live
      // `s` (freed only when its `StringGuard` drops at end of scope, after
      // this borrow). `into_owned()` copies it out before that drop.
      let v = unsafe { CStr::from_ptr(mlxrs_sys::mlx_string_data(s.0)) }
        .to_string_lossy()
        .into_owned();
      metadata.insert(key, GgufMetadata::String(v));
    } else if is_meta_vstr {
      // SAFETY: `mlx_vector_string_new()` returns a fresh empty vector handle
      // (NULL ctx on allocation failure, a defined-safe input to `_free`);
      // wrapped in `VectorStringGuard` (next line) BEFORE the fallible call.
      let mut vstr = unsafe { mlxrs_sys::mlx_vector_string_new() };
      let vstr_guard = VectorStringGuard(vstr);
      // SAFETY: `&mut vstr` is the freshly allocated out-param owned by
      // `vstr_guard`; mlx-c overwrites it in-place with the string list.
      // `gguf` is the valid borrowed handle; `ckey` is a valid in-scope
      // NUL-terminated C string; nothing retained; rc via `check()`.
      check(unsafe {
        mlxrs_sys::mlx_io_gguf_get_metadata_vector_string(&mut vstr, gguf, ckey.as_ptr())
      })?;
      // SAFETY: `vstr` is the valid populated vector from above; mlx-c returns
      // its plain element count and neither mutates nor retains it.
      let m = unsafe { mlxrs_sys::mlx_vector_string_size(vstr) };
      let mut list = Vec::with_capacity(m);
      for j in 0..m {
        let mut sp: *mut std::os::raw::c_char = std::ptr::null_mut();
        // SAFETY: `&mut sp` is a valid out-pointer; `vstr` is the valid vector
        // and `j < m` is in range. mlx-c writes into `sp` a borrowed pointer
        // to the `j`-th `std::string`'s buffer inside the still-live `vstr`
        // (not retained past the call); rc surfaced via `check()`.
        check(unsafe { mlxrs_sys::mlx_vector_string_get(&mut sp, vstr, j) })?;
        // SAFETY: on `rc == 0`, `sp` is the non-NULL pointer mlx-c just wrote,
        // aimed at a NUL-terminated buffer inside `vstr`'s `std::string`,
        // owned by `vstr` (its guard is dropped only after this loop);
        // `into_owned()` copies it out immediately.
        list.push(unsafe { CStr::from_ptr(sp) }.to_string_lossy().into_owned());
      }
      drop(vstr_guard);
      metadata.insert(key, GgufMetadata::StringList(list));
    } else {
      // SAFETY: `mlx_array_new()` returns a fresh empty out-param handle (NULL
      // ctx) per the mlx-c convention; wrapped in the RAII newtype FIRST so an
      // early `?` frees it, then populated by the next call.
      let mut arr = Array(unsafe { mlxrs_sys::mlx_array_new() });
      // SAFETY: `&mut arr.0` is the freshly allocated out-param from above;
      // `gguf` is the valid borrowed handle; `ckey` is a valid in-scope
      // NUL-terminated C string. mlx-c copies the weight tensor into `arr` and
      // retains nothing past the call; the rc is surfaced via `check()`.
      check(unsafe { mlxrs_sys::mlx_io_gguf_get_array(&mut arr.0, gguf, ckey.as_ptr()) })?;
      weights.insert(key, arr);
    }
  }
  drop(keys_guard);
  drop(guard);
  Ok((weights, metadata))
}

/// Save weights plus typed metadata to a `.gguf` file.
///
/// Mirrors `mlx.core.save_gguf`.
#[cfg(feature = "gguf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gguf")))]
pub fn save_gguf(
  path: &Path,
  weights: &HashMap<String, Array>,
  metadata: &HashMap<String, GgufMetadata>,
) -> Result<()> {
  let cpath = path_cstring(path)?;
  // SAFETY: `mlx_io_gguf_new()` returns a fresh empty GGUF handle (NULL ctx on
  // allocation failure, a defined-safe input to `_free`); wrapped in
  // `GgufGuard` (next line) immediately so any later `?` frees it.
  let gguf = unsafe { mlxrs_sys::mlx_io_gguf_new() };
  let guard = GgufGuard(gguf);

  for (k, v) in weights {
    let ck = CString::new(k.as_str()).map_err(|_| {
      let _ = &k;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::gguf_save: weights insert",
        "gguf weight key",
      ))
    })?;
    // SAFETY: `gguf` is the valid handle owned by `guard`; `ck.as_ptr()` is a
    // valid in-scope NUL-terminated C string; `v.0` is a valid borrowed
    // `mlx_array`. mlx-c copies the key into a `std::string` and the array via
    // `insert`, retaining neither pointer past the call; rc via `check()`.
    check(unsafe { mlxrs_sys::mlx_io_gguf_set_array(gguf, ck.as_ptr(), v.0) })?;
  }

  for (k, v) in metadata {
    let ck = CString::new(k.as_str()).map_err(|_| {
      let _ = &k;
      Error::InteriorNul(InteriorNulPayload::new(
        "io::gguf_save: metadata insert",
        "gguf metadata key",
      ))
    })?;
    match v {
      GgufMetadata::Array(arr) => {
        // SAFETY: `gguf` is the valid handle owned by `guard`; `ck` is a valid
        // in-scope NUL-terminated C string; `arr.0` is a valid borrowed
        // `mlx_array`. mlx-c copies key + array via `insert`, retaining
        // neither pointer past the call; rc surfaced via `check()`.
        check(unsafe { mlxrs_sys::mlx_io_gguf_set_metadata_array(gguf, ck.as_ptr(), arr.0) })?;
      }
      GgufMetadata::String(s) => {
        let cs = CString::new(s.as_str()).map_err(|_| {
          let _ = &s;
          Error::InteriorNul(InteriorNulPayload::new(
            "io::gguf_save: metadata string insert",
            "gguf metadata string value",
          ))
        })?;
        // SAFETY: `gguf` is the valid handle owned by `guard`; `ck` and `cs`
        // are valid in-scope NUL-terminated C strings. mlx-c copies both into
        // `std::string`s via `insert`, retaining neither pointer past the
        // call; the rc is surfaced via `check()`.
        check(unsafe {
          mlxrs_sys::mlx_io_gguf_set_metadata_string(gguf, ck.as_ptr(), cs.as_ptr())
        })?;
      }
      GgufMetadata::StringList(list) => {
        // SAFETY: `mlx_vector_string_new()` returns a fresh empty vector handle
        // (NULL ctx on allocation failure, a defined-safe input to `_free`);
        // wrapped in `VectorStringGuard` (next line) immediately so any later
        // `?` in this arm frees it.
        let vstr = unsafe { mlxrs_sys::mlx_vector_string_new() };
        let vstr_guard = VectorStringGuard(vstr);
        for s in list {
          let cs = CString::new(s.as_str()).map_err(|_| {
            let _ = &s;
            Error::InteriorNul(InteriorNulPayload::new(
              "io::gguf_save: metadata list-entry append",
              "gguf metadata list entry",
            ))
          })?;
          // SAFETY: `vstr` is the valid vector owned by `vstr_guard`; `cs` is
          // a valid in-scope NUL-terminated C string. mlx-c `push_back`s a
          // `std::string` copy, retaining no pointer past the call; rc via
          // `check()`.
          check(unsafe { mlxrs_sys::mlx_vector_string_append_value(vstr, cs.as_ptr()) })?;
        }
        // SAFETY: `gguf` is the valid handle owned by `guard`; `ck` is a valid
        // in-scope NUL-terminated C string; `vstr` is the valid populated
        // vector owned by `vstr_guard`. mlx-c copies key + a clone of the
        // string vector via `insert`, retaining neither past the call; rc via
        // `check()`.
        check(unsafe {
          mlxrs_sys::mlx_io_gguf_set_metadata_vector_string(gguf, ck.as_ptr(), vstr)
        })?;
        drop(vstr_guard);
      }
    }
  }

  // SAFETY: `cpath` is a valid NUL-terminated path string live for the call;
  // `gguf` is the valid populated handle owned by `guard` and kept alive
  // across this call. mlx-c reads it and writes the file, retaining nothing
  // past the call; the rc is surfaced via `check()`.
  check(unsafe { mlxrs_sys::mlx_save_gguf(cpath.as_ptr(), gguf) })?;
  drop(guard);
  Ok(())
}

// ─────────────────────── checkpoint weight discovery ───────────────────────
//
// Feature-neutral HF/safetensors checkpoint discovery, ported from the
// weight-loading half of `mlx_lm.utils.load_model`. Lives here (always
// compiled) rather than in the `lm`-gated `crate::lm::load` so every
// consumer — `embeddings`, `vlm`, `audio`, and the LM loader — reaches the
// *same* sharding-aware multi-format loader instead of hand-rolling a
// weaker single-`model.safetensors` probe. `crate::lm::load::load_weights`
// delegates here.

/// Upper bound on a `config.json`-style file read into memory, mirroring
/// `embeddings::config`'s `MAX_ST_POOLING_CONFIG_BYTES`. A real model's
/// `config.json` is well under 1 MiB; a hostile model dir cannot make us
/// allocate unbounded memory by planting a huge `config.json`.
///
/// `pub(crate)` so the `crate::lm::load` / `crate::lm::factory` config-read
/// path shares the *one* bound (rather than restating it) — they all read
/// `config.json` through the same cap. `#[allow(dead_code)]`: this shared
/// bounded-IO surface is exercised by the feature-gated config readers
/// (`lm` / `vlm` / `audio`), so a minimal feature build leaves it unused.
#[allow(dead_code)]
pub(crate) const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// Upper bound on a `model.safetensors.index.json` read into memory. The
/// index carries one `weight_name -> shard_name` entry per tensor; even a
/// Llama-3-405B-class model lists well under 100 000 tensors, comfortably
/// under 16 MiB of JSON. A hostile model directory cannot OOM us by planting
/// a multi-GB index. Twin of [`MAX_CONFIG_BYTES`] for the larger
/// per-tensor-keyed index file. Gated on `serde_json` — the only reader is
/// the JSON index path [`load_via_index`].
#[cfg(feature = "serde_json")]
const MAX_INDEX_BYTES: u64 = 16 << 20;

/// Discover and merge a model's weights from `dir`, mirroring the
/// weight-loading half of `mlx_lm.utils.load_model` while honoring the
/// HF/safetensors `model.safetensors.index.json` weight-map as the
/// **authoritative** shard manifest.
///
/// **Feature-neutral.** Lives in the always-compiled `io` module (not the
/// `lm`-gated `crate::lm::load`) so every consumer — `embeddings`, `vlm`,
/// `audio`, and the LM loader — reuses the same sharding-aware loader rather
/// than hand-rolling a single-filename probe. `crate::lm::load::load_weights`
/// is a thin delegate to this function.
///
/// Resolution order (first match wins):
///
/// 1. **`model.safetensors.index.json` (authoritative).** When the index
///    file is present, it is the SINGLE source of truth for which shards
///    belong to the checkpoint. The unique shard filenames listed in its
///    `weight_map` are loaded from `<dir>/<shard>` and merged. Stale
///    `model*.safetensors` files in `dir` whose names are NOT in the index
///    are **ignored** (the standard HF safetensors-sharded convention, and
///    the safe foundation for the [`crate::lm::load::save_model`]
///    index-rename single-commit-point). Parsing the index requires JSON
///    support (the `serde_json` feature); without it this tier is skipped —
///    a present index then fails closed (it is the authoritative manifest and
///    must not be bypassed), while an index-less directory still resolves via
///    the single-file tiers below.
/// 2. **Single `model.safetensors`** (no index). The HF un-sharded
///    convention: load the one file directly.
/// 3. **Legacy `weights.safetensors`.** Pre-HF-convention back-compat: a
///    directory carrying only this name still loads.
/// 4. **`*.gguf`** (`gguf` feature): a single `*.gguf` is loaded via
///    `load_gguf` (mlx-lm's GGUF path). Without the feature, a present
///    `*.gguf` is reported as unsupported.
/// 5. **`*.npz`** (`npz` feature): a single NumPy `.npz` (the
///    mlx-community-native multi-array format) is loaded via `load_npz`,
///    preferring `model.npz` / `weights.npz`, else the sole `.npz` in `dir`.
///
/// There is deliberately **no unindexed `model*.safetensors` glob fallback**:
/// a complete sharded checkpoint always ships its
/// `model.safetensors.index.json` manifest, so a directory holding
/// `model-*.safetensors` shards but no index is INCOMPLETE (a partial/failed
/// `save_model`, which writes `model-gen-*` shards before the index-rename
/// commit point) and is treated as having no weights rather than glob-merged —
/// matching mlx-lm's index-or-single-file contract.
///
/// No safetensors and no usable GGUF/NPZ → [`Error::FileIo`] (mlx-lm's
/// `FileNotFoundError("No safetensors found in {model_path}")`). Keys are
/// returned **verbatim** (no remap/sanitize).
///
/// The index is parsed with the same bounded-IO / `O_NONBLOCK` /
/// non-regular-reject discipline the shared `read_bounded_config_file`
/// primitive uses for `config.json` (capped at 16 MiB — well above a
/// Llama-3-405B-class index); a malformed or out-of-spec index is a
/// recoverable typed error.
pub fn load_weights_from_dir(dir: &Path) -> Result<HashMap<String, Array>> {
  // 1. Index-honoring path: the index, if present, IS the authoritative
  //    shard manifest. Stale `model*.safetensors` files NOT listed in the
  //    index are invisible to load. Gated on JSON support — when the
  //    `serde_json` feature is off (e.g. a bare `embeddings` build) the
  //    index cannot be parsed, so this tier is skipped — a present index then
  //    fails closed (1b below) rather than being silently bypassed.
  #[cfg(feature = "serde_json")]
  if let Some(weights) = load_via_index(dir)? {
    return Ok(weights);
  }

  // 1b. Fail closed on an index we cannot honor. Without `serde_json` the
  //     index tier above is compiled out, but a `model.safetensors.index.json`
  //     is still the authoritative manifest — silently falling through to the
  //     single-file tiers would load a stale `model.safetensors` (or miss the
  //     sharded checkpoint entirely). So when an index file is present but JSON
  //     support is absent, refuse rather than fall through to the lower tiers.
  #[cfg(not(feature = "serde_json"))]
  {
    let index_path = dir.join("model.safetensors.index.json");
    // Fail closed on ANY existing index entry — regular file, directory, FIFO,
    // or symlink (including a dangling one) — not just a regular file: a present
    // index sentinel means the checkpoint is sharded-by-manifest and must NOT be
    // bypassed without `serde_json`. `symlink_metadata` stats the entry itself
    // (so a dangling symlink still counts as present), unlike `is_file`/`exists`.
    if std::fs::symlink_metadata(&index_path).is_ok() {
      return Err(Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
        index_path.display().to_string(),
        Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
          "load_weights: `model.safetensors.index.json` present but the `serde_json` feature",
          "must be enabled to honor a sharded-checkpoint index",
        )),
      )));
    }
  }

  // 2. Single, un-sharded `model.safetensors` (HF convention without an
  //    index file).
  let single = dir.join("model.safetensors");
  if path_is_file(&single)? {
    return load_safetensors(&single);
  }

  // 3. Legacy back-compat: a `weights.safetensors`-only directory (pre-HF
  //    naming). Kept so a hand-rolled or older checkpoint that uses this
  //    name still loads.
  let legacy = dir.join("weights.safetensors");
  if path_is_file(&legacy)? {
    return load_safetensors(&legacy);
  }

  // No unindexed `model*.safetensors` glob fallback: a complete sharded
  // checkpoint always ships its `model.safetensors.index.json` manifest, so a
  // directory with `model-*.safetensors` shards but no index is INCOMPLETE — an
  // in-progress or failed `save_model` (which writes `model-gen-*` shards before
  // the index-rename commit point) or a malformed checkout. Globbing those would
  // return weights from an uncommitted/partial checkpoint, so we fail closed and
  // fall through to the gguf / npz tiers and the typed no-weights error —
  // matching mlx-lm's index-or-single-file contract.

  // 4. No safetensors → try a single `*.gguf` (mlx-lm's GGUF load path). The
  //    contract is a SINGLE gguf; if several are present there is no canonical
  //    name to disambiguate them, so refuse rather than silently load whichever
  //    sorts first.
  let ggufs = collect_sorted(dir, |name| name.ends_with(".gguf"))?;
  if ggufs.len() > 1 {
    return Err(Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
      ambiguity_list(&ggufs),
      Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
        "load_weights: multiple `*.gguf` weight files in the directory",
        "exactly one `*.gguf` is supported (no canonical name to disambiguate)",
      )),
    )));
  }
  if let Some(gguf) = ggufs.first() {
    #[cfg(feature = "gguf")]
    {
      let (weights, _meta) = load_gguf(gguf)?;
      return Ok(weights);
    }
    #[cfg(not(feature = "gguf"))]
    {
      return Err(Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
        gguf.display().to_string(),
        Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
          "load_weights: GGUF weight file present but `gguf` feature",
          "must be enabled to load GGUF checkpoints",
        )),
      )));
    }
  }

  // 5. No safetensors and no GGUF → try a single NumPy `*.npz` (the
  //    mlx-community-native multi-array weight format). Prefer the
  //    conventional `model.npz` / `weights.npz`; otherwise the sole `.npz`.
  #[cfg(feature = "npz")]
  {
    let preferred = dir.join("model.npz");
    if path_is_file(&preferred)? {
      return load_npz(&preferred);
    }
    let legacy_npz = dir.join("weights.npz");
    if path_is_file(&legacy_npz)? {
      return load_npz(&legacy_npz);
    }
    // Arbitrary-name fallback: the SOLE `*.npz`. With neither canonical name
    // present, several non-canonical `.npz` files have no preference order, so
    // refuse rather than silently load whichever sorts first.
    let npzs = collect_sorted(dir, |name| name.ends_with(".npz"))?;
    if npzs.len() > 1 {
      return Err(Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
        ambiguity_list(&npzs),
        Error::InvariantViolation(crate::error::InvariantViolationPayload::new(
          "load_weights: multiple non-canonical `*.npz` weight files in the directory",
          "name one `model.npz` / `weights.npz`, or keep exactly one `*.npz`",
        )),
      )));
    }
    if let Some(npz) = npzs.first() {
      return load_npz(npz);
    }
  }

  Err(Error::FileIo(FileIoPayload::new(
    NO_WEIGHTS_CONTEXT,
    FileOp::Open,
    dir.to_path_buf(),
    std::io::Error::from(std::io::ErrorKind::NotFound),
  )))
}

/// Static `context()` label for the no-weights terminal error, listing every
/// layout the resolver considered (the safetensors tiers always, plus
/// `*.gguf` / `*.npz` when those features are compiled in). The
/// `crate::lm::load` discovery tests assert this lists each safetensors tier.
#[cfg(all(feature = "gguf", feature = "npz"))]
const NO_WEIGHTS_CONTEXT: &str = "load_weights: no model weights file (expected `model.safetensors.index.json`, \
   `model.safetensors`, `weights.safetensors`, a single `*.gguf`, \
   or a single `*.npz`)";
#[cfg(all(feature = "gguf", not(feature = "npz")))]
const NO_WEIGHTS_CONTEXT: &str = "load_weights: no model weights file (expected `model.safetensors.index.json`, \
   `model.safetensors`, `weights.safetensors`, or a single `*.gguf`)";
#[cfg(all(not(feature = "gguf"), feature = "npz"))]
const NO_WEIGHTS_CONTEXT: &str = "load_weights: no model weights file (expected `model.safetensors.index.json`, \
   `model.safetensors`, `weights.safetensors`, or a single `*.npz`)";
#[cfg(all(not(feature = "gguf"), not(feature = "npz")))]
const NO_WEIGHTS_CONTEXT: &str = "load_weights: no model weights file (expected `model.safetensors.index.json`, \
   `model.safetensors`, or `weights.safetensors`)";

/// The deserialized shape of a `model.safetensors.index.json` — only its
/// `weight_map` is read (any sibling key such as `metadata` is ignored). The
/// field is optional so an *absent* `weight_map` (`None`) is distinguished
/// from a present one, letting the loader raise the precise "must contain a
/// `weight_map`" diagnostic.
#[cfg(feature = "serde_json")]
#[derive(serde::Deserialize)]
struct IndexFile {
  #[serde(default)]
  weight_map: Option<RawWeightMap>,
}

/// The raw `weight_map` value, tolerant of a non-object so the loader can emit
/// the precise "`weight_map` must be an object" diagnostic itself (rather than
/// a generic serde type error). [`RawWeightMap::Object`] is tried first; a
/// non-object (string, array, number, …) falls to [`RawWeightMap::Other`],
/// whose [`IgnoredAny`] consumes the value without retaining it (it is only a
/// "this was not an object" marker).
///
/// [`IgnoredAny`]: serde::de::IgnoredAny
#[cfg(feature = "serde_json")]
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum RawWeightMap {
  Object(IndexEntries),
  Other(serde::de::IgnoredAny),
}

/// The `weight_map` object as an **order-preserving** `Vec<(tensor_name,
/// shard_value)>` rather than a deduplicating map. A `serde_json::Value`/map
/// object collapses duplicate JSON keys (serde keeps the last), which would
/// hide a malformed index that binds one tensor name twice; visiting via
/// [`MapAccess`] instead yields *every* entry in on-disk order, so
/// [`load_via_index`] sees — and rejects — the duplicate. The shard value is
/// kept as a raw [`serde_json::Value`] (not forced to `String`) so a
/// non-string value is reported by `load_via_index` as a precise typed
/// [`Error::InvariantViolation`] rather than a generic serde parse error.
/// (Mirrors `lm::lora`'s `OrderedPattern`: the behavior does not depend on
/// `serde_json`'s `preserve_order` feature.)
///
/// [`MapAccess`]: serde::de::MapAccess
#[cfg(feature = "serde_json")]
struct IndexEntries(Vec<(String, serde_json::Value)>);

#[cfg(feature = "serde_json")]
impl<'de> serde::Deserialize<'de> for IndexEntries {
  fn deserialize<D: serde::Deserializer<'de>>(
    deserializer: D,
  ) -> std::result::Result<Self, D::Error> {
    struct EntriesVisitor;

    impl<'de> serde::de::Visitor<'de> for EntriesVisitor {
      type Value = Vec<(String, serde_json::Value)>;

      fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a `weight_map` JSON object {tensor_name: shard_file_name}")
      }

      fn visit_map<M: serde::de::MapAccess<'de>>(
        self,
        mut access: M,
      ) -> std::result::Result<Self::Value, M::Error> {
        // `next_entry` yields entries in on-disk order WITHOUT collapsing a
        // duplicate key (unlike building a `Value` map), so a doubly-bound
        // tensor name survives for `load_via_index` to reject.
        let mut out = Vec::with_capacity(access.size_hint().unwrap_or(0));
        while let Some((k, v)) = access.next_entry::<String, serde_json::Value>()? {
          out.push((k, v));
        }
        Ok(out)
      }
    }

    deserializer
      .deserialize_map(EntriesVisitor)
      .map(IndexEntries)
  }
}

/// Load a checkpoint via its `model.safetensors.index.json`, if present.
///
/// Returns `Ok(Some(weights))` when an index file is found and successfully
/// drives the load; `Ok(None)` when no index file is present (the caller's
/// "try the next candidate" signal); `Err` on any structural problem
/// (malformed index, missing referenced shard, IO failure).
///
/// The index is the HF/safetensors authoritative weight manifest: its
/// `weight_map` lists every weight-name → shard-file-name binding, and the
/// load is authoritative at the **tensor** level — exactly the `weight_map`
/// tensors are returned, each taken from its assigned shard. Each referenced
/// shard is loaded once (in sorted filename order for determinism), but only
/// the tensors the index assigned to that shard are inserted into the result;
/// any extra tensor present in a shard but NOT bound to it by `weight_map` is
/// dropped (stale duplicates cannot leak in or clobber an index-assigned
/// tensor). The contract is enforced both ways:
///
/// - a shard file named in the index but absent on disk → [`Error::FileIo`]
///   naming the offending shard;
/// - a weight key the index assigns to a shard but whose tensor is NOT in that
///   shard → [`Error::LayerKeyed`] (keyed by shard) wrapping an
///   [`Error::MissingKey`] (the absent tensor name);
/// - the same weight key assigned/inserted twice → [`Error::LayerKeyed`]
///   wrapping an [`Error::InvariantViolation`] (a duplicate `weight_map`
///   binding for one tensor name).
///
/// The index body is bounded at [`MAX_INDEX_BYTES`] via the shared
/// [`read_bounded_text_file`] primitive.
#[cfg(feature = "serde_json")]
fn load_via_index(dir: &Path) -> Result<Option<HashMap<String, Array>>> {
  use crate::error::{
    InvariantViolationPayload, LayerKeyedPayload, MissingKeyPayload, ParsePayload,
  };

  let index_path = dir.join("model.safetensors.index.json");
  // Detect index PRESENCE with `symlink_metadata` (lstat), mirroring the
  // no-serde sentinel guard: a present sentinel — INCLUDING a dangling symlink —
  // must gate the lower single-file tiers and fail CLOSED if unreadable,
  // not be treated as absent. `read_bounded_text_file` opens the target, so a
  // dangling symlink would otherwise yield `NotFound` → `Ok(None)` → a silent
  // fall-through that loads a stale `model.safetensors` despite the sentinel.
  // Only a true lstat `NotFound` (no entry at all) means "no index".
  if matches!(std::fs::symlink_metadata(&index_path), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
  {
    return Ok(None);
  }
  let text = read_bounded_text_file(&index_path, "model weight index", MAX_INDEX_BYTES)?
    .ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        index_path.display().to_string(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_via_index: `model.safetensors.index.json` sentinel present but unreadable",
          "a present index (e.g. a dangling symlink) must be a readable manifest, not fall through to lower tiers",
        )),
      ))
    })?;

  // Parse the index, reading its `weight_map` (the authoritative tensor→shard
  // manifest) through a visitor that yields entries in on-disk order. A
  // `serde_json::Value` object would silently *collapse* a duplicate tensor
  // key (serde keeps the last binding), so a malformed index with two bindings
  // for one tensor would load the later one undetected; the ordered
  // [`IndexEntries`] visitor instead preserves every entry so the
  // duplicate-binding check below can reject it.
  let index: IndexFile = serde_json::from_str(&text).map_err(|e| {
    Error::LayerKeyed(LayerKeyedPayload::new(
      index_path.display().to_string(),
      Error::Parse(ParsePayload::new(
        "load_via_index: model weight index",
        "JSON",
        e,
      )),
    ))
  })?;
  // An absent `weight_map`, or one that is not a JSON object, is a malformed
  // index (the index MUST carry the tensor→shard manifest as an object).
  let weight_map = match index.weight_map {
    Some(RawWeightMap::Object(entries)) => entries,
    None | Some(RawWeightMap::Other(_)) => {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        index_path.display().to_string(),
        Error::MissingKey(MissingKeyPayload::new(
          "load_via_index: model weight index must contain a `weight_map` object",
          "weight_map",
        )),
      )));
    }
  };

  // Invert `weight_map` into shard-basename → the set of tensor keys the index
  // assigns to that shard. A `BTreeMap` keyed on the shard name keeps the load
  // order deterministic (sorted-filename order), and lets a shard be opened
  // exactly once even when it
  // holds many tensors. Reject an empty shard name, or one carrying a path
  // separator / `.` / `..` (an absolute or parent-traversing shard name would
  // escape `dir`; the HF convention is bare basenames living in the same
  // directory). A tensor key bound twice across the (order-preserving) entries
  // is a malformed index — two bindings for one tensor name — and is rejected.
  let mut shard_tensors: std::collections::BTreeMap<&str, std::collections::BTreeSet<&str>> =
    std::collections::BTreeMap::new();
  // Tensor names seen across ALL entries — the duplicate check is GLOBAL, not
  // per-shard: a tensor name bound to two *different* shards is precisely the
  // "stale duplicate overwrites the index-assigned tensor" hazard this guard
  // closes, so it must be caught even though the two bindings land in distinct
  // per-shard sets.
  let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
  for (weight_key, shard_value) in &weight_map.0 {
    let weight_key = weight_key.as_str();
    let shard = shard_value.as_str().ok_or_else(|| {
      Error::LayerKeyed(LayerKeyedPayload::new(
        format!("weight_map[{weight_key}]"),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_via_index: weight_map shard value",
          "must be a string",
        )),
      ))
    })?;
    if shard.is_empty()
      || shard.contains('/')
      || shard.contains('\\')
      || shard == "."
      || shard == ".."
    {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        format!("weight_map[{weight_key}] -> {shard:?}"),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_via_index: weight_map shard name",
          "must be a bare basename (no path separators or `.`/`..`; lives in the same directory as the index)",
        )),
      )));
    }
    if !seen.insert(weight_key) {
      return Err(Error::LayerKeyed(LayerKeyedPayload::new(
        weight_key.to_string(),
        Error::InvariantViolation(InvariantViolationPayload::new(
          "load_via_index: weight_map tensor name",
          "is assigned more than once (a duplicate `weight_map` binding)",
        )),
      )));
    }
    shard_tensors.entry(shard).or_default().insert(weight_key);
  }

  // The index is authoritative at the TENSOR level: open each referenced
  // shard once and pull out exactly the tensors the index bound to it. An
  // extra tensor that happens to live in the shard but is unlisted for it is
  // dropped (a stale duplicate cannot leak in or clobber an index-assigned
  // tensor in another shard). A listed tensor that is absent from its assigned
  // shard is a malformed checkpoint.
  let mut weights: HashMap<String, Array> = HashMap::with_capacity(weight_map.0.len());
  for (shard, expected) in &shard_tensors {
    let shard_path = dir.join(shard);
    if !path_is_file(&shard_path)? {
      return Err(Error::FileIo(FileIoPayload::new(
        "load_via_index: shard listed by the model weight index is missing on disk",
        FileOp::Stat,
        shard_path,
        std::io::Error::from(std::io::ErrorKind::NotFound),
      )));
    }
    let mut part = load_safetensors(&shard_path)?;
    for &key in expected {
      let Some(tensor) = part.remove(key) else {
        return Err(Error::LayerKeyed(LayerKeyedPayload::new(
          shard.to_string(),
          Error::MissingKey(MissingKeyPayload::new(
            "load_via_index: tensor assigned by the model weight index is missing from its shard",
            key,
          )),
        )));
      };
      // A unique-per-name `weight_map` (enforced above) means no key can be
      // produced twice across shards, so this insert never overwrites.
      weights.insert(key.to_string(), tensor);
    }
  }
  Ok(Some(weights))
}

/// Render a sorted candidate list (from [`collect_sorted`]) as a single
/// comma-separated string of base names for an ambiguity diagnostic, e.g.
/// `"a.gguf, b.gguf"`. Base names are used (not full paths) so the message
/// names the colliding files without leaking the absolute temp/cache prefix;
/// the directory is already named by the surrounding `load_weights` context.
/// Carried in a [`LayerKeyedPayload`]'s runtime-key channel — the project's
/// typed-error idiom for a dynamic identifier — so the static
/// [`InvariantViolationPayload`] phrase stays allocation-free.
fn ambiguity_list(paths: &[std::path::PathBuf]) -> String {
  let mut names: Vec<&str> = paths
    .iter()
    .map(|p| {
      p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<non-utf8>")
    })
    .collect();
  names.sort_unstable();
  names.join(", ")
}

/// Whether `path` exists AND its (symlink-resolved) target is a regular
/// file. A symlink whose target is a regular file qualifies (HF Hub snapshot
/// dirs store these as symlinks into `blobs/<hash>`, the same convention the
/// [`collect_sorted`] / [`read_bounded_config_file`] paths intentionally
/// follow). A missing path is `Ok(false)`; any other stat failure is an
/// [`Error::FileIo`] (`Stat`).
///
/// `pub(crate)` so `crate::lm::load`'s discovery tests reach it via the
/// shared module path.
pub(crate) fn path_is_file(path: &Path) -> Result<bool> {
  match std::fs::metadata(path) {
    Ok(m) => Ok(m.is_file()),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
    Err(e) => Err(Error::FileIo(FileIoPayload::new(
      "path_is_file",
      FileOp::Stat,
      path.to_path_buf(),
      e,
    ))),
  }
}

/// List the entries of `dir` whose file name matches `pred`, returning their
/// full paths sorted by name. A non-readable directory (absent / not a
/// directory / permission) maps to [`Error::FileIo`] (`Read`). Only regular
/// files are considered (a directory named `model….safetensors` is ignored).
///
/// `pub(crate)` so `crate::lm::load`'s discovery tests reach it via the
/// shared module path.
pub(crate) fn collect_sorted(
  dir: &Path,
  pred: impl Fn(&str) -> bool,
) -> Result<Vec<std::path::PathBuf>> {
  let entries = std::fs::read_dir(dir).map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      "cannot read model directory",
      FileOp::Read,
      dir.to_path_buf(),
      e,
    ))
  })?;
  let mut out = Vec::new();
  for entry in entries {
    let entry = entry.map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        "cannot read an entry of",
        FileOp::Read,
        dir.to_path_buf(),
        e,
      ))
    })?;
    let name = entry.file_name();
    let Some(name) = name.to_str() else { continue };
    if !pred(name) {
      continue;
    }
    // Require the *resolved* target to be a regular file (a hostile dir
    // could name a subdir / FIFO `model.safetensors`; mlx-c would then fail
    // opaquely on open). `DirEntry::file_type()` does NOT follow symlinks,
    // but HF Hub snapshot dirs store weight files as symlinks into
    // `blobs/<hash>` (mlx-lm's `glob(...) + mx.load(wf)` follows them) — so
    // resolve via `fs::metadata` (follows symlinks) and gate on the target.
    // The original (possibly-symlink) path is passed through; the IO loader
    // opens it, following the link.
    match std::fs::metadata(entry.path()) {
      Ok(m) if m.is_file() => out.push(entry.path()),
      Ok(_) => continue,
      Err(e) => {
        return Err(Error::FileIo(FileIoPayload::new(
          "collect_sorted: cannot stat entry",
          FileOp::Stat,
          entry.path(),
          e,
        )));
      }
    }
  }
  out.sort();
  Ok(out)
}

/// Bounded, TOCTOU-closed read of a config-style file at `path`.
///
/// Shared bounded-config-file primitive used by every config-JSON reader in
/// the loader (`config.json`, `generation_config.json`,
/// `(pre)processor_config.json`, VLM base-config). Behavior:
///
/// - `Ok(Some(text))` on a successful, bounded, valid-UTF-8 read.
/// - `Ok(None)` if the file is absent (`ENOENT`) — the caller's "try the
///   next candidate" / "absent is OK" signal. The caller decides whether
///   absence is a hard error or simply *no override*.
/// - `Err(Error::FileIo)` / `Err(Error::CapExceeded)` / `Err(Error::LayerKeyed)`
///   on every other failure (open failure other than `NotFound`, not a
///   regular file, oversized, IO failure during read, non-UTF-8).
///
/// Discipline mirrors `embeddings::config`'s pooling-config read: open
/// **once** with `O_NONBLOCK | O_CLOEXEC` on unix (so a planted FIFO returns
/// immediately and never hangs the loader), post-open `is_file()` fstat
/// rejects non-regular targets even when reached via a symlink (HF Hub
/// snapshot caches store these files as symlinks into `blobs/<hash>`, which
/// is intentionally followed since the post-open stat enforces the
/// guarantee on the *resolved* target), and the body is capped at
/// [`MAX_CONFIG_BYTES`] via `Read::take` so a hostile model directory
/// cannot OOM us by planting a huge config.
///
/// `pub(crate)` so `crate::lm::load` / `crate::vlm::load` / `crate::audio`
/// readers funnel through the one hardened path. `#[allow(dead_code)]`: a
/// minimal feature build with no config readers leaves this shared surface
/// unused.
#[allow(dead_code)]
pub(crate) fn read_bounded_config_file(path: &Path, label: &'static str) -> Result<Option<String>> {
  read_bounded_text_file(path, label, MAX_CONFIG_BYTES)
}

/// Shared bounded-text-file primitive parametrized on the byte cap. Identical
/// hardening (open-once + non-regular-reject + `O_NONBLOCK | O_CLOEXEC` on
/// unix + cap-via-`Read::take`) as [`read_bounded_config_file`]; factored out
/// so the larger [`MAX_INDEX_BYTES`] cap for `model.safetensors.index.json`
/// can reuse the *one* hardening path rather than restating it.
///
/// Adds a UTF-8 validation pass on top of the shared
/// [`read_bounded_bytes_file`] byte read (a non-UTF-8 body is a typed parse
/// error); the byte primitive owns the open/stat/cap hardening so both the
/// text and the binary-asset readers share the *one* path.
///
/// `pub(crate)` so `crate::lm::load`'s discovery tests exercise the explicit
/// `max_bytes` cap directly. `#[allow(dead_code)]`: a minimal feature build
/// with no config/index reader leaves this shared primitive unused.
#[allow(dead_code)]
pub(crate) fn read_bounded_text_file(
  path: &Path,
  label: &'static str,
  max_bytes: u64,
) -> Result<Option<String>> {
  let Some(bytes) = read_bounded_bytes_file(path, label, max_bytes)? else {
    return Ok(None);
  };
  let text = String::from_utf8(bytes).map_err(|e| {
    Error::LayerKeyed(crate::error::LayerKeyedPayload::new(
      path.display().to_string(),
      Error::Parse(crate::error::ParsePayload::new(label, "UTF-8", e)),
    ))
  })?;
  Ok(Some(text))
}

/// Shared bounded-**bytes**-file primitive parametrized on the byte cap — the
/// binary-asset twin of [`read_bounded_text_file`], returning the raw bytes
/// (no UTF-8 validation) for files that are not text (e.g. a SentencePiece
/// `.model` protobuf).
///
/// Identical TOCTOU-closed hardening as [`read_bounded_config_file`]: open
/// **once** with `O_NONBLOCK | O_CLOEXEC` on unix (so a planted FIFO returns
/// immediately and never hangs the loader), post-open `is_file()` fstat
/// rejects non-regular targets even when reached via a symlink, and the body
/// is capped at `max_bytes` via `Read::take` so a hostile model directory
/// cannot OOM the loader by planting a huge file.
///
/// `Ok(Some(bytes))` on a successful bounded read, `Ok(None)` if the file is
/// absent (`ENOENT`), `Err` on every other failure (open failure other than
/// `NotFound`, not a regular file, oversized, IO failure during read).
///
/// `pub(crate)` so a per-usecase asset reader (e.g. the SenseVoice SPM
/// `.model` loader) can read a binary asset through the *one* bounded-read
/// path with its own generous cap, rather than restating the hardening.
/// `#[allow(dead_code)]`: a minimal feature build with no asset reader
/// leaves this shared primitive unused.
#[allow(dead_code)]
pub(crate) fn read_bounded_bytes_file(
  path: &Path,
  label: &'static str,
  max_bytes: u64,
) -> Result<Option<Vec<u8>>> {
  use std::io::Read;

  #[cfg(unix)]
  let open_result = {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
      .open(path)
  };
  #[cfg(not(unix))]
  let open_result = std::fs::File::open(path);

  let file = match open_result {
    Ok(f) => f,
    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
    Err(e) => {
      return Err(Error::FileIo(FileIoPayload::new(
        label,
        FileOp::Open,
        path.to_path_buf(),
        e,
      )));
    }
  };

  let meta = file.metadata().map_err(|e| {
    Error::FileIo(FileIoPayload::new(
      label,
      FileOp::Stat,
      path.to_path_buf(),
      e,
    ))
  })?;
  if !meta.is_file() {
    return Err(Error::FileIo(FileIoPayload::new(
      label,
      FileOp::Stat,
      path.to_path_buf(),
      std::io::Error::from(std::io::ErrorKind::InvalidInput),
    )));
  }

  let mut bytes = Vec::new();
  file
    .take(max_bytes + 1)
    .read_to_end(&mut bytes)
    .map_err(|e| {
      Error::FileIo(FileIoPayload::new(
        label,
        FileOp::Read,
        path.to_path_buf(),
        e,
      ))
    })?;
  if bytes.len() as u64 > max_bytes {
    return Err(Error::CapExceeded(crate::error::CapExceededPayload::new(
      label,
      "max_bytes",
      max_bytes,
      bytes.len() as u64,
    )));
  }

  Ok(Some(bytes))
}

#[cfg(test)]
mod tests;
