//! Process-cached MLX version string accessor.

use std::{ffi::CStr, sync::OnceLock};

static VERSION: OnceLock<String> = OnceLock::new();

/// Returns the MLX C library version (e.g. `"0.31.2"`).
///
/// Cached on first call; subsequent calls return the same `&'static str`.
pub fn version() -> &'static str {
  VERSION.get_or_init(|| {
    // SAFETY: mlx_string_new + mlx_version + mlx_string_data + mlx_string_free
    // are an idiomatic mlx-c sequence. Errors here would surface via the global
    // error handler once it is installed; until then we trust the call.
    unsafe {
      let mut s = mlxrs_sys::mlx_string_new();
      let rc = mlxrs_sys::mlx_version(&mut s);
      assert_eq!(rc, 0, "mlx_version returned {rc}");
      let data = mlxrs_sys::mlx_string_data(s);
      assert!(!data.is_null(), "mlx_string_data returned NULL");
      let owned = CStr::from_ptr(data).to_string_lossy().into_owned();
      let _ = mlxrs_sys::mlx_string_free(s);
      owned
    }
  })
}
