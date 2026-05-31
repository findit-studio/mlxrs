# Changelog

## [0.1.0] — 2026-05-31

First release. Safe Rust bindings for Apple's MLX array framework on Apple
silicon (`aarch64-apple-darwin`) via the `mlx-c` FFI layer, plus opt-in
higher-level support surfaces ported from MLX's companion projects.

### Added

#### Core

- **`mlxrs-sys`** — pre-committed bindgen output for `mlx-c`. Builds vendored mlx-c (+ gguflib) via cmake-rs; links libmlxc + libmlx + Metal/Accelerate.
- **`Array`** RAII handle, the `Dtype` enum, and the `Element` trait (bool, integer, and float / half / bfloat / complex element types).
- **Lazy evaluation** (`Array::eval`) — reading data via `item` / `to_vec` forces evaluation. `Array` does **not** implement `Clone` (the only duplication is the fallible `try_clone`) and is `!Send + !Sync` (single-thread, like MLX's own APIs); move results across threads as owned data from `to_vec` / `item`.
- **Ops** across `arithmetic`, `reduction`, `comparison`, `logical`, `shape`, `indexing`, `linalg`, `fft`, `fast`, and `misc`, plus `transforms` (autodiff + graph transformations) and a `memory` API.
- **Public `Stream` / `Device` API** — thread-affine, non-RAII handles for explicit stream/device placement.
- **`io`** — safetensors and GGUF load/save.
- **`simd`** — arch-gated NEON kernels (with scalar fallbacks) for hot image/audio paths.
- **Operator overloads** (`&a + &b`, `-&a`, …) gated behind the off-by-default `unstable-ops-overload` feature; they **panic** on shape/dtype error, so library authors must never enable them transitively — the fallible `a.add(&b)?` form is the load-bearing API.

#### Optional feature surfaces (off by default)

- **`lm`** — language models: HF tokenizers (BPE / SentencePiece / chat templates / tool-call parsing), KV-caches (rotating / chunked / batched / quantized), samplers + logits processors, quantization, LoRA / DoRA, optimizers (Adam, AdamW, Adamax, Adafactor, Muon, SGD, Lion, Adadelta, RMSprop), and the generation loop + chat session.
- **`vlm`** — vision-language models (implies `lm`): image preprocessing, prompt assembly, multimodal generation.
- **`audio`** — audio (implies `lm`): STFT / mel DSP, WAV I/O, STT / TTS serializers, playback.
- **`embeddings`** — embedding-model loading, pooling modes, and the encode pipeline.
- **`llguidance`** — grammar-constrained / structured decoding.
- **`gguf`** — GGUF load/save (gguflib is vendored + statically linked by `mlxrs-sys`).
- Finer-grained `tokenizer-*` flags expose individual tokenizer pieces without the full `lm` surface.

#### Tooling & CI

- **xtask `regen-bindings`** — re-run bindgen against the vendored mlx-c headers.
- Per-crate CI (`mlxrs-sys.yml`, `mlxrs.yml`) with matrix feature builds, clippy / fmt / docs gates, a bindings-drift gate, a coverage job, and a weekly `dep-watch.yml` dependency cron.
- Extensive unit-test coverage (~90% of the testable surface; device-bound playback / Metal-kernel dispatch and unreachable defensive guards are excluded).

### Architecture decisions

- **`aarch64-apple-darwin` only.** Other targets (`x86_64-apple-darwin`, Linux + CUDA, distributed) are roadmapped.
- **No dependency pinning** in any manifest (loose semver only); `Cargo.lock` is gitignored. The reproducibility check is the weekly `dep-watch.yml` cron, not a checked-in lockfile.
- **Pre-committed bindings + a drift CI gate** anchor binding stability — `regen-bindings` must be re-run + committed when the mlx-c submodule moves.
- **Async Metal kernel failures intentionally abort the process.** The rc/sentinel chain only catches synchronous errors; a `set_terminate`-style recovery shim is not implementable (mlx-c exposes no hook), and only diagnostics are planned.
- **No per-model architectures are bundled** — the `lm` / `vlm` / `audio` / `embeddings` features ship the support surface (loaders, tokenizers, caches, samplers, processors, generation loops, audio I/O), not specific model implementations; those are added per use-case.

### Safety audits

- Entry refcount audit (`docs/audits/send-soundness.md`, local-only) — verified MLX `array_desc_` is a `std::shared_ptr` with an atomic refcount, but per-clone `Send` is unsound because `set_status` mutates non-atomic state through `const`. Final design is `!Send + !Sync`.
- Adversarial code review on every PR — caught and fixed empty-slice dangling-pointer UB across `slice` / `sum_axes` / `concatenate` / `gather` / `pad` / shape-taking ops; introduced `dim_ptr` / `data_ptr` / per-`Element` `sentinel_ptr` helpers; sealed `IntoShape`; centralized `validate_dims` at every FFI boundary; routed empty-axes reductions through MLX; and added bounded overflow / cap guards on FFI op wrappers that reach unchecked C++ arithmetic.

[0.1.0]: https://github.com/Findit-AI/mlxrs/releases/tag/v0.1.0
