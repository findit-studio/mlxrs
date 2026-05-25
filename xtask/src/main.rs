//! xtask: maintainer-only commands. Run via `cargo run -p xtask -- <subcommand>`.
//!
//! `regen-bindings` builds with no extra deps. The `codegen` subcommand and
//! its heavy `tokenizers`/`serde_json`/`toml` deps are gated behind the
//! optional `codegen` feature; invoke it via `cargo xtask-codegen` (alias
//! for `cargo run -p xtask --features codegen -- codegen`).

#[cfg(feature = "codegen")]
mod codegen;

use std::{env, path::PathBuf, process::ExitCode};

fn main() -> ExitCode {
  let mut args = env::args().skip(1);
  let cmd = args.next().unwrap_or_else(|| "help".to_string());
  match cmd.as_str() {
    "regen-bindings" => regen_bindings(),
    "codegen" => {
      #[cfg(feature = "codegen")]
      {
        let check = args.any(|a| a == "--check");
        codegen::run(check)
      }
      #[cfg(not(feature = "codegen"))]
      {
        eprintln!(
          "`codegen` requires the `codegen` feature (it pulls the heavy \
           tokenizers/serde_json/toml toolchain)."
        );
        eprintln!("Run with: cargo xtask-codegen        (alias for");
        eprintln!("          cargo run -p xtask --features codegen -- codegen)");
        eprintln!("Drift guard: cargo xtask-codegen --check");
        ExitCode::FAILURE
      }
    }
    "help" | "--help" | "-h" => {
      print_help();
      ExitCode::SUCCESS
    }
    other => {
      eprintln!("unknown subcommand: {other}");
      print_help();
      ExitCode::FAILURE
    }
  }
}

fn print_help() {
  eprintln!("xtask — maintainer commands");
  eprintln!();
  eprintln!("USAGE:");
  eprintln!("  cargo run -p xtask -- <subcommand>");
  eprintln!();
  eprintln!("SUBCOMMANDS:");
  eprintln!("  regen-bindings    Re-run bindgen against vendor/mlx-c headers and");
  eprintln!("                    write to mlxrs-sys/src/generated/bindings.rs");
  eprintln!("  codegen [--check] Regenerate the committed tokenizer artifacts from");
  eprintln!("                    mlxrs/data/tokenizer/ (--check: diff-only, no writes).");
  eprintln!("                    Requires the `codegen` feature — invoke as");
  eprintln!("                    `cargo xtask-codegen [--check]` (alias for");
  eprintln!("                    `cargo run -p xtask --features codegen -- codegen`).");
  eprintln!("  help              Show this message");
}

fn regen_bindings() -> ExitCode {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let workspace_root = manifest_dir.parent().expect("xtask is in workspace");
  let sys_dir = workspace_root.join("mlxrs-sys");
  let mlx_c = sys_dir.join("vendor/mlx-c");
  let wrapper = sys_dir.join("wrapper.h");
  let out = sys_dir.join("src/generated/bindings.rs");

  if !wrapper.exists() {
    eprintln!("wrapper.h not found at {}", wrapper.display());
    return ExitCode::FAILURE;
  }
  if !mlx_c.join("mlx/c/mlx.h").exists() {
    eprintln!("mlx-c headers not found; run: git submodule update --init --recursive");
    return ExitCode::FAILURE;
  }

  std::fs::create_dir_all(out.parent().unwrap()).unwrap();

  // mlx-c's `io.h` declares two fns taking `*mut FILE` (mlx_export_to_dot,
  // mlx_print_graph). Without blocking, bindgen expands `FILE` → macOS's
  // private `__sFILE` struct with full layout tests, which makes the
  // bindings drift on every Xcode SDK update. Block both names and let
  // downstream code use `libc::FILE` (or `*mut c_void`) directly.
  let bindings = bindgen::Builder::default()
    .header(wrapper.to_string_lossy())
    .clang_arg(format!("-I{}", mlx_c.display()))
    .clang_arg("-std=c11")
    .allowlist_function("mlx_.*")
    .allowlist_type("mlx_.*")
    .allowlist_var("MLX_.*")
    .blocklist_item("_mlx_error")
    .blocklist_file(".*/private/.*")
    .blocklist_type("FILE")
    .blocklist_type("__sFILE")
    .blocklist_type("__sbuf")
    .blocklist_type("__sFILEX")
    // Re-declare FILE as an opaque c_void since mlx_export_to_dot /
    // mlx_print_graph still reference it. mlxrs's safe wrapper doesn't
    // expose graph-export in M1; the bindings just need a type for the FFI.
    .raw_line("type FILE = ::std::ffi::c_void;")
    .layout_tests(true)
    .derive_default(false)
    .derive_debug(false)
    .generate_comments(false)
    .formatter(bindgen::Formatter::Rustfmt)
    .generate()
    .expect("bindgen generate failed");

  bindings
    .write_to_file(&out)
    .expect("write bindings.rs failed");
  eprintln!("wrote {}", out.display());
  ExitCode::SUCCESS
}
