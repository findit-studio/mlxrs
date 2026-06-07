# CoreML -> ANE from Rust — feasibility spike findings

Status: **COMPLETE — FEASIBLE.** Pure Rust (via `objc2-core-ml` 0.3, no
Objective-C shim) loads compiled WhisperKit `.mlmodelc` models, runs predictions
on the Neural Engine, threads explicit KV-cache tensors through the decoder, and
creates/uses an `MLState`. A CPU-only vs ANE timing differential confirms the
Neural Engine actually executes the graph.

## FINAL FEASIBILITY VERDICT: FEASIBLE (pure objc2-core-ml, no shim)

Every make-or-break question came back green. The spike binary
(`src/main.rs`, macOS-gated) loads all three tiny-whisper models, predicts on
each, and exits 0. Reproduce with `cd coreml-spike && cargo run --release`.

---

## 1. Model I/O contracts (from each `.mlmodelc/metadata.json`)

All tensors are first-major contiguous `MLMultiArray`. Storage/compute precision
is Float16 (mixed Float16/Int32). Models are CoreML `mlProgram`, spec v7,
macOS 13+.

### MelSpectrogram.mlmodelc
- in : `audio`  Float16 `[480000]`  (30 s @ 16 kHz)
- out: `melspectrogram_features`  Float16 `[1, 80, 1, 3000]`

### AudioEncoder.mlmodelc
- in : `melspectrogram_features`  Float16 `[1, 80, 1, 3000]`
- out: `encoder_output_embeds`  Float16 `[1, 384, 1, 1500]`

### TextDecoder.mlmodelc  (one decode step; EXPLICIT-I/O KV cache)
- in:
  - `input_ids`  Int32 `[1]`
  - `cache_length`  Int32 `[1]`
  - `key_cache`  Float16 `[1, 1536, 1, 224]`
  - `value_cache`  Float16 `[1, 1536, 1, 224]`
  - `kv_cache_update_mask`  Float16 `[1, 224]`
  - `encoder_output_embeds`  Float16 `[1, 384, 1, 1500]`
  - `decoder_key_padding_mask`  Float16 `[1, 224]`
- out:
  - `logits`  Float16 `[1, 1, 51865]`
  - `key_cache_updates`  Float16 `[1, 1536, 1, 1]`
  - `value_cache_updates`  Float16 `[1, 1536, 1, 1]`
  - `alignment_heads_weights`  Float16 `[1, 1500]`  (cross-attn for word timestamps)

> Note: `1536 = 6 layers x 256` (key/value width); `224` is the max decoder
> context. The decoder is the **classic WhisperKit explicit-cache** design: the
> caller carries `key_cache`/`value_cache` in and stitches the `*_cache_updates`
> back each step (using `kv_cache_update_mask` + `cache_length`). It is **NOT**
> declared as an iOS18 stateful (`MLState`) model — see section 4.

---

## 2. Load + predict from Rust — WORKS

Loaded each `.mlmodelc` with
`MLModel::modelWithContentsOfURL_configuration_error` and ran
`predictionFromFeatures_error`. Inputs built as `MLMultiArray`
(`initWithShape:dataType:error:`, written through `dataPointer`), wrapped in an
`MLDictionaryFeatureProvider` (`initWithDictionary:error:`), outputs read back
via `MLFeatureProvider::featureValueForName` -> `multiArrayValue`.

Observed (silent/zero input; pipeline chains real mel -> encoder -> decoder):

```
MelSpectrogram -> melspectrogram_features [1, 80, 1, 3000]
    head [-0.807, -0.807, -0.807, -0.807, -0.807]  finite=true
AudioEncoder   -> encoder_output_embeds   [1, 384, 1, 1500]
    head [0.109, -0.778, -0.300, -0.150, -0.380]   finite=true
TextDecoder    -> logits                  [1, 1, 51865]
    argmax token = 50362  finite=true
```

Shapes match the declared contracts exactly and all sampled outputs are finite.
(`-0.807` is the expected log-mel floor for silence; argmax `50362` is a
sensible first decode from `<|startoftranscript|>` with an empty cache.)
**Load + predict is proven.**

## 3. ANE-eligible config — WORKS

`MLModelConfiguration::setComputeUnits(MLComputeUnits::CPUAndNeuralEngine)`
(`.All` works too). All three models load + predict under it without error.
`MLComputeUnits` constants confirmed: `CPUOnly=0, CPUAndGPU=1, All=2,
CPUAndNeuralEngine=3`.

## 4. MLState (stateful-decoder KV path) — BINDING USABLE

`objc2-core-ml` 0.3 fully exposes the stateful API (all default-on features):
- `MLModel::newState() -> Retained<MLState>` — works; returns a state handle.
- `MLModel::predictionFromFeatures_usingState_error(input, state)` — works;
  reachable + runnable from Rust (re-ran a decode step through it -> logits
  `[1,1,51865]`).
- `MLState::getMultiArrayForStateNamed_handler(name, block)` is present for
  initializing/reading state buffers (needs the `block2` feature; not exercised).

Important nuance for the Whisper backend: **the WhisperKit tiny `TextDecoder` is
NOT a stateful model** — it uses explicit cache tensors (section 1). So
`newState()` returns an *empty* state and stateful prediction is equivalent to
stateless for *these* artifacts. The KV cache for this decoder is driven by
passing `key_cache`/`value_cache` as ordinary inputs and consuming
`*_cache_updates` outputs (which the spike does in step [3]). The `MLState` path
is nonetheless fully wired from Rust and would be the mechanism for any
genuinely-stateful CoreML model (e.g. a future stateful Whisper export).
**No shim needed for either cache style.**

## 5. ANE confirmation — CONFIRMED (no sudo)

Three independent best-effort signals; `powermetrics --samplers ane_power`
needs sudo (unavailable non-interactively), so the spike confirms ANE
programmatically instead:

- **Device enumeration** (`MLAllComputeDevices()`): reports
  `device[0] = NeuralEngine (ANE), totalCoreCount = 16`, plus GPU and CPU.
  The Neural Engine is an available CoreML scheduling target.
- **Throughput**: AudioEncoder sustains ~90 encodes/s (~11 ms/encode) in a
  300-iter loop — ANE-class for a Float16 attention encoder.
- **CPU-only vs ANE differential** (the clincher): the *same* AudioEncoder
  loaded `.cpuOnly` runs ~125.6 ms/encode vs ~10.6 ms/encode on
  `.cpuAndNeuralEngine` — a **~11.8x speedup**. A differential that large is
  direct evidence the Neural Engine is executing the graph, not merely listed.

(A fully deterministic per-operation ANE *assignment* report is also available
via `MLComputePlan.loadContentsOfURL` ->
`computeDeviceUsageForMLProgramOperation` -> `preferredComputeDevice`
[`is MLNeuralEngineComputeDevice`]; that API is present in the crate but uses
`block2` async loading, so it was left as a documented next step rather than
wired into the spike. The 11.8x differential is already conclusive.)

---

## Binding crate: `objc2-core-ml` 0.3 (over `objc2` 0.6) — VIABLE, no shim

- `objc2-core-ml = "0.3"` + `objc2 = "0.6"` + `objc2-foundation = "0.3"`.
  All required classes are behind per-class cargo features that are ON by
  default (79/80): `MLModel`, `MLMultiArray`, `MLModelConfiguration`,
  `MLDictionaryFeatureProvider`, `MLFeatureValue`, `MLFeatureProvider`,
  `MLState`, `MLAllComputeDevices`, `MLNeuralEngineComputeDevice`,
  `MLComputePlan`, ...
- Idioms that worked (for the eventual backend):
  - alloc via the `objc2::AllocAnyThread` trait (`MLMultiArray::alloc()` etc.).
  - `MLMultiArray` raw access through `dataPointer()` (marked deprecated upstream
    in favour of the `getMutableBytesWithHandler:` block API, but direct
    contiguous access is simplest and still exposed — `#[allow(deprecated)]`).
    Float16 read/write done with a tiny inline half<->f32 codec (no `half` dep).
  - dictionary built with `NSDictionary::from_slices(&[&NSString], &[&AnyObject])`
    where each `MLFeatureValue` coerces to `&AnyObject` via its `AsRef` chain.
  - errors surface cleanly through the generated `error:_` -> `Result<_, Retained<NSError>>`.
- Almost all methods are `unsafe fn` (objc2 convention); the spike documents a
  `SAFETY:` rationale at each call. A real backend would wrap these in a small
  safe seam.

## Recommended approach for the full backend

**Pure `objc2-core-ml` (no Objective-C shim).** Add an Apple-gated CoreML
`Backend` behind mlxrs whisper's existing decode pipeline:
`encode(mel) -> embeds`, `decode_step(tokens, kv_cache) -> (logits, kv_updates)`
(explicit-cache style matching these artifacts), plus
`alignment_heads_weights` for word timestamps. Fall back to the MLX path
off-Apple. The mel front-end can be CoreML (`MelSpectrogram`) or kept in mlxrs.

Effort estimate (unchanged from provisional, now de-risked): ~2.5–4K net-new
Rust LoC — a thin safe wrapper over the objc2 calls proven here, the
cache-stitching bookkeeping (`cache_length` / `kv_cache_update_mask`), tokenizer
+ seek-loop glue (reuse mlxrs whisper's), and model discovery/loading. Gated to
`target_vendor = "apple"`.

## Remaining blockers / open items (none blocking)

- Per-op ANE assignment via `MLComputePlan` (block2 async) — nice-to-have,
  not required (differential already proves ANE execution).
- Real-audio end-to-end parity (mel from actual PCM -> transcript) — belongs in
  the backend PR, not this load+predict spike.
- `MLState` buffer init/read via `getMultiArrayForStateNamed:handler:` (block2)
  — only relevant if a future *stateful* Whisper export is used; not needed for
  the current explicit-cache WhisperKit decoder.

## Isolation (mlxrs build unaffected)

`coreml-spike` is a **standalone crate with its own empty `[workspace]` table**,
so it is NOT a member of the mlxrs workspace — `cargo metadata` at the worktree
root resolves exactly `{mlxrs, mlxrs-sys, xtask}` (coreml-spike excluded), and
the `objc2-core-ml` dependency graph never reaches an mlxrs build. The only diff
vs the spike base is `coreml-spike/src/main.rs`. (A fresh worktree's mlxrs build
separately requires `git submodule update --init --recursive` for
`vendor/{mlx,mlx-c,gguflib}` — a pre-existing setup step, orthogonal to this
spike.)
