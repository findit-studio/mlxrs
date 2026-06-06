# CoreML → ANE from Rust — feasibility spike findings

Status: **PARTIAL** (interrupted by a server-side API rate-limit mid-spike).
Captures what was determined before interruption; the load+predict execution
proof was not reached. Re-run to complete steps 3–5.

## Verdict so far: FEASIBLE-looking, no blocker found yet

The two hardest discovery questions both came back favorable:

### 1. Binding crate: `objc2-core-ml` 0.3 (over `objc2` 0.6) — VIABLE
- `objc2-core-ml = "0.3"` + `objc2 = "0.6"` + `objc2-foundation = "0.3"`.
- The required CoreML classes — **`MLModel`, `MLMultiArray`, `MLModelConfiguration`,
  and crucially `MLState`** — are all present, behind per-class cargo features
  that are **ON by default** (79 of 80 features default-on).
- `MLState` being present is the key result: it is the stateful-models / KV-cache
  mechanism WhisperKit's `TextDecoder` uses, and the main thing that would have
  forced an Objective-C shim if it were missing. It is **not** missing.
- `MLModelConfiguration.computeUnits` (`.all` / `.cpuAndNeuralEngine`) is the
  ANE-targeting knob; expected reachable via objc2-core-ml (confirm in the re-run).
- => A pure-`objc2-core-ml` path looks viable; an Obj-C shim may NOT be required.
  (Re-run must confirm the `MLState` + async `prediction(from:using:)` ergonomics
  are actually usable, not just present as symbols.)

### 2. Model artifacts: obtained
- Downloaded the real WhisperKit tiny CoreML models from `argmaxinc/whisperkit-coreml`
  into `../models/whisperkit/openai_whisper-tiny/` (gitignored):
  `AudioEncoder.mlmodelc`, `MelSpectrogram.mlmodelc`, `TextDecoder.mlmodelc`.
- These are the ANE-tuned compiled models — the right starting point (vs a naive
  coremltools convert that may not stay on-ANE).

## NOT yet done (interrupted — the re-run picks up here)
- `src/main.rs`: the actual load-`.mlmodelc` + build dummy `MLMultiArray` mel +
  `prediction` + read output. (The `[[bin]]` exists in Cargo.toml; the source was
  not written before interruption.)
- Record each model's exact input/output feature names + shapes + dtypes.
- Confirm `computeUnits = .cpuAndNeuralEngine` runs (ANE-eligible).
- Drive the stateful `TextDecoder` (`MLState` create + thread through prediction).
- ANE-usage confirmation (powermetrics / Console compute-unit logs).

## Isolation
This is a standalone crate with its own empty `[workspace]` table — it is NOT a
member of the mlxrs workspace, so the `objc2-core-ml` dep graph never touches an
mlxrs build. `cargo build` at the mlxrs workspace root is provably unaffected.

## Recommended approach for the full backend (provisional)
Pure `objc2-core-ml` (no shim) wiring a `Backend` seam over mlxrs whisper's
existing decode pipeline (the lfm-style seam): `encode(mel)`, `decoder_step(tokens,
MLState) -> logits`, `cross_qk` for word timestamps. ~2.5–4K net-new Rust LoC,
gated to Apple targets, falling back to MLX off-platform. Re-confirm once the
load+predict proof lands.
