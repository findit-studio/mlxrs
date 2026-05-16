//! `SharedArray` — a frozen, `Send + Sync`, lock-free read-only view of an
//! `Array`, modeled on `bytes`: `Array` ≈ `BytesMut` (lazy/mutable, `!Send`),
//! `SharedArray` ≈ `Bytes` (frozen, immutable, cheaply cloned, shared reads).
//!
//! `Array` is `!Send + !Sync` (see `array/mod.rs`): cheap `Clone` is
//! refcount-shared, and the first data access lazily `eval`s — mutating the
//! C++ `array_desc->status` non-atomically through `const`. Two clones on two
//! threads each `eval`ing would race that write.
//!
//! [`Array::freeze`] closes that window once, up front: it consumes the
//! `Array`, drives it to mlx `Status::available` via `eval()` (a full
//! synchronous `mlx::core::eval`, or `wait()` blocking on the compute event —
//! either way `set_status(available)`), then wraps it in `Arc<Array>`.
//! `available` is terminal and monotonic; mlx ops never mutate an existing
//! `array_desc` in place (they build new arrays). `SharedArray` exposes only
//! pure data-read accessors — `mlx_array_{data,size,shape,strides,dtype}` —
//! which never touch `eval`/`wait`/`is_available`/`set_status`, the only
//! `status` writers. So post-freeze the shared `array_desc` is immutable:
//! concurrent `&self` reads from any number of threads are pure reads of
//! immutable memory plus the atomic `shared_ptr` refcount (`Clone`/`Drop`).
//! No `Mutex`, no serialization — reads are lock-free.
//!
//! ## Cross-thread eval caveat
//!
//! `freeze` runs the `eval()` on the **calling** thread (it uses that
//! thread's default mlx stream — see `stream.rs`). After freezing, reads need
//! no stream, so a `SharedArray` may be read from any thread. The realistic
//! pattern — materialize on the producer/loader thread, then share the frozen
//! result read-only to worker threads — is fully supported. What is *not*
//! possible (an mlx-TLS limitation, not a `SharedArray` one) is deferring the
//! `eval` itself to a different thread than the one that built the graph.

use std::sync::Arc;

use crate::{
  array::{Array, conversion::is_row_contiguous},
  dtype::{Dtype, Element},
  error::{Error, Result},
};

impl Array {
  /// Freeze this array into a `Send + Sync`, cheaply-cloneable, read-only
  /// [`SharedArray`]. Consumes `self` and evaluates once (to mlx
  /// `Status::available`) so every subsequent read is a lock-free pure
  /// memory access. Mirrors `bytes::BytesMut::freeze`.
  ///
  /// Returns `Err` if the internal `eval` fails (e.g. a backend error, or
  /// the thread's mlx streams were cleared).
  pub fn freeze(mut self) -> Result<SharedArray> {
    self.eval()?;
    // SAFETY-adjacent: `Array` is `!Send`, so `Arc<Array>` trips
    // `clippy::arc_with_non_send_sync`. The lint cannot see the manual
    // `unsafe impl Send + Sync for SharedArray` below (justified there);
    // the `Arc<Array>` is only ever reached through `SharedArray`.
    #[allow(clippy::arc_with_non_send_sync)]
    Ok(SharedArray(Arc::new(self)))
  }
}

/// Frozen, immutable, `Send + Sync` view of an `Array`. Construct via
/// [`Array::freeze`]. `Clone` is a cheap `Arc` refcount bump (no data copy,
/// no `mlx_array_set`); every clone shares the same materialized buffer.
///
/// ```no_run
/// # fn run() -> mlxrs::Result<()> {
/// use mlxrs::Array;
///
/// let shared = Array::ones::<f32>(&(2, 2))?.freeze()?;
/// let s2 = shared.clone();
///
/// std::thread::spawn(move || -> mlxrs::Result<()> {
///   // Lock-free read from another thread — no &mut, no eval.
///   assert_eq!(s2.to_vec::<f32>()?, vec![1.0; 4]);
///   Ok(())
/// })
/// .join()
/// .unwrap()?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct SharedArray(Arc<Array>);

// SAFETY: `Array` is `!Send + !Sync` solely because cheap `Clone` + lazy
// `eval` let two threads race the non-atomic `array_desc->status` write
// during the lazy→materialized transition. `Array::freeze` performs that
// transition exactly once (to mlx `Status::available`, verified terminal &
// monotonic in mlx `array.h`) before any sharing, and `SharedArray` exposes
// no API that re-mutates: every accessor below is a pure read via
// `mlx_array_{data,size,shape,strides,dtype}`, none of which call
// `eval`/`wait`/`is_available`/`set_status`. The only post-freeze writes are
// the atomic `shared_ptr` refcount via `Clone`/`Drop`. Even a leftover lazy
// `Array` alias (same `array_desc`, stranded on the constructing thread —
// `!Send`) cannot introduce a write: its `eval()` short-circuits because
// `is_available()` is already true, so `wait()` performs no `set_status`.
// The shared `array_desc` is therefore immutable across threads, exactly the
// `bytes::BytesMut::freeze` → `Bytes` invariant.
unsafe impl Send for SharedArray {}
unsafe impl Sync for SharedArray {}

// Compile-time guarantees colocated with the type definition.
static_assertions::assert_impl_all!(SharedArray: Send, Sync, Clone);

impl SharedArray {
  /// Number of dimensions.
  pub fn ndim(&self) -> usize {
    self.0.ndim()
  }

  /// Total number of elements.
  pub fn size(&self) -> usize {
    self.0.size()
  }

  /// Whether the array has zero elements.
  pub fn is_empty(&self) -> bool {
    self.0.size() == 0
  }

  /// Element type.
  pub fn dtype(&self) -> Result<Dtype> {
    self.0.dtype()
  }

  /// Shape as a `Vec<usize>`.
  pub fn shape(&self) -> Vec<usize> {
    self.0.shape()
  }

  /// Scalar extraction (size-1 arrays). No `eval` — `freeze` already
  /// materialized the buffer.
  pub fn item<T: Element>(&self) -> Result<T> {
    self.check_dtype::<T>()?;
    unsafe { T::item(self.0.0) }
  }

  /// Copy the materialized buffer into a `Vec<T>`. Errors with
  /// [`Error::NonContiguous`] for strided/broadcast views (same contract as
  /// [`Array::to_vec`]); no `eval` (frozen).
  pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
    Ok(self.as_slice::<T>()?.to_vec())
  }

  /// Borrow the materialized buffer as `&[T]` (lifetime tied to `&self`; the
  /// frozen buffer is immutable and the `Arc` keeps it alive for the
  /// borrow). Errors with [`Error::NonContiguous`] for strided views.
  pub fn as_slice<T: Element>(&self) -> Result<&[T]> {
    self.check_dtype::<T>()?;
    if !is_row_contiguous(self.0.0) {
      return Err(Error::NonContiguous);
    }
    unsafe {
      let (ptr, len) = T::data(self.0.0);
      // Zero-element arrays yield a NULL data pointer from mlx;
      // `from_raw_parts(NULL, 0)` is UB, so short-circuit.
      if len == 0 {
        return Ok(&[]);
      }
      assert!(!ptr.is_null(), "mlx data pointer NULL on a frozen array");
      Ok(std::slice::from_raw_parts(ptr, len))
    }
  }

  /// Recover the inner `Array` if this is the sole handle (no other clones).
  /// Returns `None` while any clone is still alive — drop them first, or keep
  /// using the shared read-only API.
  pub fn into_inner(self) -> Option<Array> {
    Arc::try_unwrap(self.0).ok()
  }

  fn check_dtype<T: Element>(&self) -> Result<()> {
    let actual = self.0.dtype()?;
    if actual != T::DTYPE {
      return Err(Error::DtypeMismatch {
        expected: T::DTYPE,
        got: actual,
      });
    }
    Ok(())
  }
}
