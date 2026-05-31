//! Model IO — safetensors and GGUF load/save.
//!
//! Thin wrappers over mlx-c `io.h`. Local-file IO only; no HF-hub download.
//!
//! - safetensors: a map of named arrays plus an optional `String -> String`
//!   metadata side-table. Mirrors `mlx.core.load/save_safetensors` and
//!   mlx-swift `loadArrays` / `loadArraysAndMetadata` / `save`.
//! - GGUF: a map of named tensors plus typed metadata entries (array /
//!   string / list-of-strings), mirroring mlx-c's `mlx_io_gguf` API and
//!   `mlx.core.load_gguf` (returns weights + metadata).
//!
//! Validation (bad file, missing key, dtype quirks) is left to mlx-c and
//! surfaced through [`Result`]. See
//! [mlx io docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html).

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
    return Err(last.unwrap_or(Error::Backend(
      "mlx_map_string_to_array_new() returned NULL sentinel (allocation failure)".into(),
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
    return Err(last.unwrap_or(Error::Backend(
      "mlx_map_string_to_string_new() returned NULL sentinel (allocation failure)".into(),
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
    return Err(last.unwrap_or(Error::Backend(
      "mlx_io_writer_new() returned NULL sentinel (allocation failure)".into(),
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
  derive_more::Display, derive_more::IsVariant, derive_more::Unwrap, derive_more::TryUnwrap,
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
