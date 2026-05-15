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

  // Link declarations.
  println!("cargo:rustc-link-search=native={}", lib_dir.display());
  println!("cargo:rustc-link-lib=static=mlxc");
  println!("cargo:rustc-link-lib=static=mlx");
  println!("cargo:rustc-link-lib=framework=Metal");
  println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
  println!("cargo:rustc-link-lib=framework=Foundation");
  println!("cargo:rustc-link-lib=framework=QuartzCore");
  println!("cargo:rustc-link-lib=framework=Accelerate");
  println!("cargo:rustc-link-lib=dylib=c++");

  // The FFI declarations themselves come from the pre-committed
  // mlxrs-sys/src/generated/bindings.rs (regenerated via
  // `cargo run -p xtask -- regen-bindings`). This build script's job is
  // strictly to compile + link libmlxc + libmlx + the Apple frameworks;
  // it does NOT invoke bindgen.
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
