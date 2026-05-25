// First-party C++ shim for `mlx::core` APIs that the vendored mlx-c layer
// does not expose. Compiled by build.rs against libmlx and hand-declared
// (NOT bindgen-generated) in src/lib.rs, so this stays out of the mlx-c
// bindgen drift gate.
//
// Policy: prefer mlx-c bindings for everything. Add a shim here ONLY for a
// genuine mlx-c coverage gap, and track it for upstreaming to
// ml-explore/mlx-c so this file shrinks over time.
//
// Current gaps bridged:
//   - mlx::core::clear_streams()  (mlx/stream.h) — no mlx-c equivalent.
//
// Upstream tracking:
//   - mlx-c: request a `mlx_clear_streams()` binding for the above so this
//     shim can be retired (pure binding gap; the C++ symbol exists).
//   - mlx (ml-explore/mlx): there is NO per-stream teardown anywhere in
//     mlx C++ — verified by exhaustive grep, the only stream destroy
//     primitive is the bulk thread-wide `clear_streams()`. Real per-value
//     RAII for the safe `Stream` wrapper is therefore impossible at the
//     source level, not just unavailable via mlx-c. If scoped stream
//     lifetimes ever matter, the only path is an upstream feature request
//     for a `free_stream(Stream)` / per-stream destructor in mlx itself.

#include "mlx/stream.h"

extern "C" {

// Destroy all streams created on the *current thread*, freeing their Metal
// command encoders. This is mlx's only stream-teardown primitive: it is
// thread-wide and bulk (there is no per-stream free), which is why the safe
// `Stream` wrapper cannot map it to `Drop`. Returns 0 on success, 1 if the
// underlying C++ call threw (it clears an unordered_map; throwing is not
// expected in practice, but we never let a C++ exception cross into Rust).
int mlxrs_shim_clear_streams(void) {
  try {
    mlx::core::clear_streams();
    return 0;
  } catch (...) {
    return 1;
  }
}

} // extern "C"
