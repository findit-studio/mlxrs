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
  path::Path,
};

use crate::{
  array::Array,
  error::{Error, Result, check},
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
  CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| Error::Backend {
    message: format!("path contains an interior NUL byte: {}", path.display()),
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
  // SAFETY: `mlx_map_string_to_array_new()` returns a fresh empty map handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`),
  // wrapped in an `ArrayMapGuard` IMMEDIATELY so any `?` below (interior-NUL
  // key, insert allocation failure) frees the partially-built map. On success
  // ownership is transferred to the caller via `mem::forget` (suppressing
  // this guard's `Drop`); the caller re-wraps the returned raw handle.
  let guard = ArrayMapGuard(unsafe { mlxrs_sys::mlx_map_string_to_array_new() });
  for (k, v) in arrays {
    let ck = CString::new(k).map_err(|_| Error::Backend {
      message: format!("array key contains an interior NUL byte: {k:?}"),
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
  // SAFETY: `mlx_map_string_to_string_new()` returns a fresh empty map handle
  // (NULL ctx on allocation failure, a defined-safe input to `_free`),
  // wrapped in a `StringMapGuard` IMMEDIATELY so any `?` below (interior-NUL
  // key/value, insert allocation failure) frees the partially-built map. On
  // success ownership is transferred to the caller via `mem::forget`
  // (suppressing this guard's `Drop`); the caller re-wraps the raw handle.
  let guard = StringMapGuard(unsafe { mlxrs_sys::mlx_map_string_to_string_new() });
  for (k, v) in meta {
    let ck = CString::new(k.as_str()).map_err(|_| Error::Backend {
      message: format!("metadata key contains an interior NUL byte: {k:?}"),
    })?;
    let cv = CString::new(v.as_str()).map_err(|_| Error::Backend {
      message: format!("metadata value contains an interior NUL byte: {v:?}"),
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

// ─────────────────────────── GGUF ───────────────────────────
//
// GGUF load/save is gated behind the `gguf` cargo feature (OFF by default):
// mlx core's `gguf.cpp` depends on the third-party `gguflib` (`gguf_open`,
// `gguf_create`, …) which `mlxrs-sys` does not yet fetch or link, so linking
// these symbols fails until the sys crate vendors gguflib.

/// A typed GGUF metadata entry. GGUF metadata values are one of: a scalar/
/// tensor [`Array`], a string, or a list of strings (matches mlx-c's
/// `mlx_io_gguf_*_metadata_*` accessors).
#[cfg(feature = "gguf")]
#[cfg_attr(docsrs, doc(cfg(feature = "gguf")))]
#[non_exhaustive]
pub enum GgufMetadata {
  /// An array-valued metadata entry.
  Array(Array),
  /// A string-valued metadata entry.
  String(String),
  /// A list-of-strings metadata entry.
  StringList(Vec<String>),
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
    let ckey = CString::new(key.as_str()).map_err(|_| Error::Backend {
      message: format!("gguf key contains an interior NUL byte: {key:?}"),
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
    let ck = CString::new(k.as_str()).map_err(|_| Error::Backend {
      message: format!("gguf weight key contains an interior NUL byte: {k:?}"),
    })?;
    // SAFETY: `gguf` is the valid handle owned by `guard`; `ck.as_ptr()` is a
    // valid in-scope NUL-terminated C string; `v.0` is a valid borrowed
    // `mlx_array`. mlx-c copies the key into a `std::string` and the array via
    // `insert`, retaining neither pointer past the call; rc via `check()`.
    check(unsafe { mlxrs_sys::mlx_io_gguf_set_array(gguf, ck.as_ptr(), v.0) })?;
  }

  for (k, v) in metadata {
    let ck = CString::new(k.as_str()).map_err(|_| Error::Backend {
      message: format!("gguf metadata key contains an interior NUL byte: {k:?}"),
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
        let cs = CString::new(s.as_str()).map_err(|_| Error::Backend {
          message: format!("gguf metadata string contains an interior NUL byte: {s:?}"),
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
          let cs = CString::new(s.as_str()).map_err(|_| Error::Backend {
            message: format!("gguf metadata list entry contains an interior NUL byte: {s:?}"),
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
