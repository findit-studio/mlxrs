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
  process::Command,
};

// ───── Submodule revision pins ─────
//
// These MUST match what `mlxrs-sys/vendor/mlx-c/CMakeLists.txt` declares
// in its `FetchContent_Declare(mlx ... GIT_TAG v0.31.2)`, and what
// `mlxrs-sys/vendor/mlx/mlx/io/CMakeLists.txt` declares in its
// `FetchContent_Declare(gguflib ... GIT_TAG <sha>)`. With
// `FETCHCONTENT_SOURCE_DIR_MLX` / `FETCHCONTENT_SOURCE_DIR_GGUFLIB` wired
// below, cmake skips the GIT_TAG enforcement entirely and builds from
// whatever sits at those paths — a submodule that's been manually
// `git checkout`'d to a different SHA (or skipped during a lockstep
// mlx-c bump) would silently compile against the wrong version. The
// `check_submodule_rev` preflight below catches that drift.
//
// If a future mlx-c bump moves the mlx pin (or a future mlx bump moves
// the gguflib pin):
//   1. Update the submodule: `cd mlxrs-sys/vendor/<name> && git checkout
//      <new_tag> && cd - && git add mlxrs-sys/vendor/<name>`.
//   2. Update the corresponding `EXPECTED_*_REV` const below.
//   3. Both must land in the same change so CI catches drift.
// ──────────────────────────────────
const EXPECTED_MLX_REV: &str = "68cf2fddd8de5edd8ab3d926391772b2e2cedad8"; // v0.31.2
const EXPECTED_GGUFLIB_REV: &str = "8fa6eb65236618e28fd7710a0fba565f7faa1848";

fn main() {
  // Re-run if these change
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=wrapper.h");
  println!("cargo:rerun-if-changed=vendor/mlx-c");
  println!("cargo:rerun-if-changed=vendor/mlx");
  println!("cargo:rerun-if-changed=vendor/gguflib");
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
  // Three submodules under vendor/: mlx-c (which we cmake-configure
  // directly), and mlx + gguflib (which mlx-c's CMakeLists.txt and mlx's
  // io/CMakeLists.txt would otherwise FetchContent over the network, but
  // which we redirect to these local paths via FETCHCONTENT_SOURCE_DIR_*
  // below for offline builds).
  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let vendor_dir = manifest_dir.join("vendor");
  let mlx_c_root = vendor_dir.join("mlx-c");
  let mlx_root = vendor_dir.join("mlx");
  let gguflib_root = vendor_dir.join("gguflib");
  for (name, root, sentinel) in [
    ("mlx-c", &mlx_c_root, "CMakeLists.txt"),
    ("mlx", &mlx_root, "CMakeLists.txt"),
    ("gguflib", &gguflib_root, "gguflib.c"),
  ] {
    if !root.join(sentinel).exists() {
      panic!(
        "vendor/{name}/{sentinel} missing. Run:\n\
               \tgit submodule update --init --recursive"
      );
    }
  }

  // Submodule revision check — verify each vendored submodule is at the
  // pinned commit recorded in EXPECTED_*_REV. The sentinel-file preflight
  // above only proves the source tree exists; with FETCHCONTENT_SOURCE_DIR_*
  // below, cmake never enforces GIT_TAG, so a stale checkout (e.g. forgot
  // `git submodule update` after a rebase) would otherwise silently build
  // against the wrong upstream. See `check_submodule_rev` doc for the
  // trust boundary (same level as `~/.cargo/registry/`: we trust the user
  // to not tamper with their own vendored sources).
  check_submodule_rev(&mlx_root, EXPECTED_MLX_REV, "mlx").unwrap_or_else(|msg| panic!("{msg}"));
  check_submodule_rev(&gguflib_root, EXPECTED_GGUFLIB_REV, "gguflib")
    .unwrap_or_else(|msg| panic!("{msg}"));

  // CMake invocation. cmake-rs installs to ${dst}/lib/, ${dst}/include/.
  //
  // FETCHCONTENT_SOURCE_DIR_<uppercaseName> is cmake's standard override
  // for FetchContent_Declare: when set, cmake skips the GIT_REPOSITORY
  // clone for that dependency and uses the local path as the source dir,
  // then proceeds to build it in-place exactly as if it had fetched it.
  // We wire two:
  //   * FETCHCONTENT_SOURCE_DIR_MLX  → redirects mlx-c/CMakeLists.txt's
  //     `FetchContent_Declare(mlx GIT_REPOSITORY ... GIT_TAG v0.31.2)`.
  //   * FETCHCONTENT_SOURCE_DIR_GGUFLIB → redirects mlx core's
  //     mlx/io/CMakeLists.txt's `FetchContent_Declare(gguflib ... GIT_TAG
  //     8fa6eb65236618e28fd7710a0fba565f7faa1848)` (active when
  //     MLX_BUILD_GGUF is ON — its unconditional upstream default).
  // The vendored submodule pins MUST match these GIT_TAG values. If a
  // future mlx-c bump changes the mlx pin, or a future mlx bump changes
  // the gguflib pin, the corresponding submodule under vendor/ must be
  // updated in lockstep (`git submodule update --remote` followed by an
  // explicit `git checkout <newtag>` inside the submodule).
  // Ref: https://cmake.org/cmake/help/latest/module/FetchContent.html
  //      #variable:FETCHCONTENT_SOURCE_DIR_%3CuppercaseName%3E
  let dst = cmake::Config::new(&mlx_c_root)
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("MLX_C_BUILD_EXAMPLES", "OFF")
    .define("MLX_BUILD_TESTS", "OFF")
    .define("MLX_BUILD_BENCHMARKS", "OFF")
    .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
    .define("CMAKE_CXX_STANDARD", "20")
    .define("CMAKE_BUILD_TYPE", "Release")
    .define(
      "FETCHCONTENT_SOURCE_DIR_MLX",
      mlx_root.to_str().expect("mlx vendor path is valid UTF-8"),
    )
    .define(
      "FETCHCONTENT_SOURCE_DIR_GGUFLIB",
      gguflib_root
        .to_str()
        .expect("gguflib vendor path is valid UTF-8"),
    )
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
  // (currently just `clear_streams`). Compiled against the mlx C++ headers
  // and linked alongside libmlx. cc emits its own
  // `cargo:rustc-link-lib=static=mlxrs_shim` + search-path directives HERE,
  // i.e. BEFORE the `static=mlx` directive below, so a single-pass linker
  // (GNU ld) resolves the shim's `mlx::core::clear_streams` reference into
  // libmlx. macOS ld64 is order-insensitive for archives but we keep the
  // correct order regardless.
  //
  // Header source: `vendor/mlx/` directly. With FETCHCONTENT_SOURCE_DIR_MLX
  // wired above, cmake does NOT clone into `_deps/mlx-src/` — it uses the
  // vendored submodule path as the cmake source dir in-place. Pointing the
  // shim include at the submodule source guarantees the shim sees the same
  // tagged-v0.31.2 headers that libmlx.a was compiled from, byte-identical
  // to what FetchContent would have cloned.
  if !mlx_root.join("mlx/stream.h").exists() {
    panic!(
      "mlx C++ headers not found at {} (expected mlx/stream.h). The shim \
             needs the vendored mlx submodule.",
      mlx_root.display()
    );
  }
  cc::Build::new()
    .cpp(true)
    .std("c++20")
    .file("shim/mlxrs_shim.cpp")
    .include(&mlx_root)
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

/// Verify the vendored submodule is checked out at the pinned upstream
/// revision recorded in [`EXPECTED_MLX_REV`] / [`EXPECTED_GGUFLIB_REV`].
///
/// # Trust boundary
///
/// This check confirms the submodule is INITIALIZED and at the expected
/// COMMIT. It does NOT verify the working-tree contents are byte-identical
/// to that commit. A developer with write access to the vendored worktree
/// can locally modify or replace tracked files; this check will pass if
/// `HEAD` still points to the pinned commit.
///
/// The trust model is the same as for files under `~/.cargo/registry/`:
/// we trust the user to not tamper with their own vendored sources. If
/// the user accidentally drifts (e.g. forgot to run `git submodule update`
/// after a rebase), the HEAD-mismatch check catches that. If the user
/// actively tampers, that is outside this build script's threat model.
///
/// Callers needing stricter integrity should:
/// 1. Build from a clean checkout (`git clone` + `git submodule update --init --recursive`)
/// 2. Or use cargo's vendored-sources mechanism with checksum verification
///
/// # Errors
///
/// Returns `Err` if:
/// - The submodule directory exists with `.git` metadata BUT `git rev-parse HEAD`
///   reports a different commit than expected. Remediation: run
///   `git submodule update --init --recursive` from the parent repo.
/// - The git command starts but reports non-zero exit (broken metadata,
///   safety rejection, etc.). Remediation: re-initialize the submodule.
///
/// Returns `Ok(())` (intentional skip) if:
/// - The submodule directory has no `.git` metadata. This happens when the
///   crate is consumed as a packaged tarball (e.g. from crates.io) where the
///   submodule was vendored as a plain directory. The packaging machinery is
///   trusted to have captured the correct revision.
/// - The `git` command cannot be executed (`Command::output()` returns Err).
///   This typically means git is not on the PATH; we cannot verify, but we
///   should not block the build for users who legitimately lack git.
fn check_submodule_rev(
  submodule_path: &Path,
  expected: &str,
  friendly_name: &str,
) -> Result<(), String> {
  // A live submodule checkout has either a `.git` directory (standalone
  // clone) or a `.git` file (gitlink form, typical when nested under a
  // superproject). Either form means git can resolve HEAD here. Absence
  // is the packaged-tarball case — see docstring "Returns Ok" bullet 1.
  if !submodule_path.join(".git").exists() {
    return Ok(());
  }

  let path_str = match submodule_path.to_str() {
    Some(s) => s,
    None => return Ok(()), // non-UTF-8 path; can't pass to git CLI, best-effort skip.
  };

  let output = match Command::new("git")
    .args(["-C", path_str, "rev-parse", "HEAD"])
    .output()
  {
    Ok(o) => o,
    Err(_) => return Ok(()), // git not on PATH; best-effort skip (process never launched).
  };

  if !output.status.success() {
    // Process started but reported failure — broken gitfile, unreadable
    // gitdir, safe-directory rejection, partial checkout, etc. Fail
    // closed so drift protection isn't accidentally disabled by corrupt
    // metadata (R2 hardening preserved through simplification).
    return Err(format!(
      "vendored submodule `{friendly_name}` at {} has git metadata but \
       `git rev-parse HEAD` failed (exit {:?}, stderr: {}). Cannot verify \
       pinned revision. Re-initialize the submodule:\n\
       \tgit submodule deinit -f {}\n\
       \tgit submodule update --init --recursive",
      submodule_path.display(),
      output.status.code(),
      String::from_utf8_lossy(&output.stderr).trim_end(),
      submodule_path.display(),
    ));
  }

  let actual = String::from_utf8_lossy(&output.stdout).trim().to_string();
  if actual != expected {
    let name_upper = friendly_name.to_uppercase();
    return Err(format!(
      "vendored submodule `{friendly_name}` at {} is at revision {actual} \
       but mlxrs-sys expects {expected}.\n\
       \n\
       If you intentionally updated the submodule, also update \
       EXPECTED_{name_upper}_REV in mlxrs-sys/build.rs (both must commit \
       in the same change so CI catches drift).\n\
       \n\
       Otherwise restore the pinned revision:\n\
       \tgit -C {} checkout {expected}\n\
       \tgit add {}",
      submodule_path.display(),
      submodule_path.display(),
      submodule_path.display(),
    ));
  }

  Ok(())
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
