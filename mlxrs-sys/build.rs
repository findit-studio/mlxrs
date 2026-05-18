//! Build script for mlxrs-sys.
//!
//! Invokes cmake on the vendored mlx-c submodule, extracts libmlx.a from the
//! transitive MLX FetchContent build, and emits link directives for libmlxc +
//! libmlx + the Apple frameworks mlx requires (Metal, MetalPerformanceShaders,
//! Foundation, QuartzCore, Accelerate) plus libc++.
//!
//! M1 is aarch64-apple-darwin only; Phase 2 will plug optional bindgen here
//! under `cfg(feature = "buildtime_bindgen")`.

use std::{
  env,
  path::{Path, PathBuf},
};

fn main() {
  // Re-run if these change
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=wrapper.h");
  println!("cargo:rerun-if-changed=vendor/mlx-c");
  println!("cargo:rerun-if-changed=shim/mlxrs_shim.cpp");

  // Target check — M1 is macOS arm64 only.
  let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
  let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
  if target_os != "macos" || target_arch != "aarch64" {
    panic!(
      "mlxrs-sys M1 supports aarch64-apple-darwin only; \
             got target_os={target_os}, target_arch={target_arch}"
    );
  }

  // Submodule check — fail fast if user forgot --recurse-submodules.
  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let mlx_c_root = manifest_dir.join("vendor/mlx-c");
  if !mlx_c_root.join("CMakeLists.txt").exists() {
    panic!(
      "vendor/mlx-c/CMakeLists.txt missing. Run:\n\
             \tgit submodule update --init --recursive"
    );
  }

  // CMake invocation. cmake-rs installs to ${dst}/lib/, ${dst}/include/.
  let dst = cmake::Config::new(&mlx_c_root)
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("MLX_C_BUILD_EXAMPLES", "OFF")
    .define("MLX_BUILD_TESTS", "OFF")
    .define("MLX_BUILD_BENCHMARKS", "OFF")
    .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
    .define("CMAKE_CXX_STANDARD", "20")
    .define("CMAKE_BUILD_TYPE", "Release")
    .build();

  // mlx-c's CMakeLists only does install(TARGETS mlxc ...), so libmlxc.a
  // lands in ${dst}/lib/ but libmlx.a is buried under
  // ${dst}/build/_deps/mlx-build/mlx/. Walk and copy it next to libmlxc.a.
  let lib_dir = dst.join("lib");
  let build_dir = dst.join("build");
  let libmlx_src = find_libmlx(&build_dir)
    .unwrap_or_else(|| panic!("could not find libmlx.a under {}", build_dir.display()));
  let libmlx_dst = lib_dir.join("libmlx.a");
  std::fs::copy(&libmlx_src, &libmlx_dst).unwrap_or_else(|e| {
    panic!(
      "failed to copy {} -> {}: {e}",
      libmlx_src.display(),
      libmlx_dst.display()
    )
  });

  // MLX core's `mlx/io/CMakeLists.txt` compiles `gguf.cpp` into libmlx.a
  // whenever `MLX_BUILD_GGUF` is ON (its unconditional upstream default — we
  // don't override it), leaving libmlx.a with undefined `gguf_*` symbols
  // (`gguf_open`, `gguf_create`, `gguf_get_tensor`, …). The same CMakeLists
  // FetchContent's antirez/gguf-tools and builds `add_library(gguflib STATIC
  // fp16.c gguflib.c)` to satisfy them, but links it via
  // `target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:gguflib>)` — a
  // BUILD_INTERFACE-only edge. Since we copy libmlx.a out and link it
  // standalone (not through MLX's export/usage interface), that edge is
  // dropped and gguflib never reaches our link line. So locate the already
  // built libgguflib.a (same buried-archive walk as libmlx.a) and copy it
  // beside libmlxc.a. This is unconditional (not gated on the `mlxrs` `gguf`
  // feature, which mlxrs-sys can't observe anyway): libgguflib.a is a
  // self-contained archive (fp16.c.o + gguflib.c.o; external deps are libc
  // only), and ld64 only pulls archive members that resolve referenced
  // symbols — non-gguf binaries reference no `gguf_*`, so its objects are
  // never pulled and default builds are byte-for-byte unaffected.
  let libgguflib_src = find_libgguflib(&build_dir).unwrap_or_else(|| {
    panic!(
      "could not find libgguflib.a under {} (MLX core builds it from \
             FetchContent'd gguf-tools when MLX_BUILD_GGUF is ON, its default)",
      build_dir.display()
    )
  });
  let libgguflib_dst = lib_dir.join("libgguflib.a");
  std::fs::copy(&libgguflib_src, &libgguflib_dst).unwrap_or_else(|e| {
    panic!(
      "failed to copy {} -> {}: {e}",
      libgguflib_src.display(),
      libgguflib_dst.display()
    )
  });

  // First-party C++ shim for mlx::core APIs that mlx-c does not expose
  // (currently just `clear_streams`). Compiled against the FetchContent'd
  // mlx C++ headers and linked alongside libmlx. cc emits its own
  // `cargo:rustc-link-lib=static=mlxrs_shim` + search-path directives HERE,
  // i.e. BEFORE the `static=mlx` directive below, so a single-pass linker
  // (GNU ld) resolves the shim's `mlx::core::clear_streams` reference into
  // libmlx. macOS ld64 is order-insensitive for archives but we keep the
  // correct order regardless.
  let mlx_src = build_dir.join("_deps/mlx-src");
  if !mlx_src.join("mlx/stream.h").exists() {
    panic!(
      "mlx C++ headers not found at {} (expected mlx/stream.h). The shim \
             needs the FetchContent'd mlx-src tree.",
      mlx_src.display()
    );
  }
  cc::Build::new()
    .cpp(true)
    .std("c++20")
    .file("shim/mlxrs_shim.cpp")
    .include(&mlx_src)
    .compile("mlxrs_shim");

  // Link declarations.
  println!("cargo:rustc-link-search=native={}", lib_dir.display());
  println!("cargo:rustc-link-lib=static=mlxc");
  println!("cargo:rustc-link-lib=static=mlx");
  // After `static=mlx`: libmlx.a's gguf.cpp references gguflib's `gguf_*`
  // symbols, so gguflib must follow it for a single-pass linker (GNU ld).
  // macOS ld64 is order-insensitive for archives, but we keep the correct
  // order regardless (same convention as the libmlxc → libmlx ordering).
  println!("cargo:rustc-link-lib=static=gguflib");
  println!("cargo:rustc-link-lib=framework=Metal");
  println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
  println!("cargo:rustc-link-lib=framework=Foundation");
  println!("cargo:rustc-link-lib=framework=QuartzCore");
  println!("cargo:rustc-link-lib=framework=Accelerate");
  println!("cargo:rustc-link-lib=dylib=c++");

  // The mlx-c FFI declarations come from the pre-committed
  // mlxrs-sys/src/generated/bindings.rs (regenerated via
  // `cargo run -p xtask -- regen-bindings`); this build script does NOT
  // invoke bindgen. The first-party shim's declaration is hand-written in
  // src/lib.rs (intentionally kept out of the bindgen drift gate, since it
  // is not part of the vendored mlx-c surface). This script's job is to
  // compile the shim + link libmlxc + libmlx + libmlxrs_shim + the Apple
  // frameworks.
}

fn find_libmlx(start: &Path) -> Option<PathBuf> {
  // Assert exactly one match so we don't silently link a stale or
  // wrong-architecture libmlx.a if MLX upstream ever produces multiple
  // archives with this name (e.g., a cpu+metal split, or a stale archive
  // left from a previous failed build).
  let matches: Vec<PathBuf> = walkdir::WalkDir::new(start)
    .into_iter()
    .filter_map(Result::ok)
    .filter(|e| e.file_name() == "libmlx.a")
    .map(|e| e.into_path())
    .collect();
  match matches.len() {
    0 => None,
    1 => matches.into_iter().next(),
    _ => panic!(
      "found {} libmlx.a candidates under {}; expected exactly one. \
             Either MLX upstream changed its build layout, or a stale archive \
             was left behind. Candidates:\n  {}",
      matches.len(),
      start.display(),
      matches
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n  "),
    ),
  }
}

fn find_libgguflib(start: &Path) -> Option<PathBuf> {
  // Same buried-archive walk + exactly-one assertion as find_libmlx: MLX
  // core's add_library(gguflib STATIC ...) emits a single libgguflib.a under
  // the mlx-build tree (_deps/mlx-build/mlx/io/). Asserting uniqueness guards
  // against silently linking a stale archive if MLX upstream ever changes
  // its gguf build layout.
  let matches: Vec<PathBuf> = walkdir::WalkDir::new(start)
    .into_iter()
    .filter_map(Result::ok)
    .filter(|e| e.file_name() == "libgguflib.a")
    .map(|e| e.into_path())
    .collect();
  match matches.len() {
    0 => None,
    1 => matches.into_iter().next(),
    _ => panic!(
      "found {} libgguflib.a candidates under {}; expected exactly one. \
             Either MLX upstream changed its gguf build layout, or a stale \
             archive was left behind. Candidates:\n  {}",
      matches.len(),
      start.display(),
      matches
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n  "),
    ),
  }
}
