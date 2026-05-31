//! Wired-memory policy layer — `WiredMemoryPolicy` trait + 4 concrete
//! policies ([`WiredSumPolicy`], [`WiredMaxPolicy`], [`WiredFixedPolicy`],
//! [`WiredBudgetPolicy`]) + the [`WiredMemoryMeasurement`] runtime-measurement
//! result struct.
//!
//! ## References
//! - Swift trait: [`mlx-swift/Source/MLX/WiredMemory.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/WiredMemory.swift)
//!   lines 116-148 — the base `WiredMemoryPolicy` protocol (`limit` +
//!   `canAdmit` with a default-true admission policy).
//! - Swift LM policies: [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
//!   — the 4 LM-side cap-aware policies (Sum / Max / Fixed / Budget). The
//!   mlx-swift base `WiredSumPolicy` / `WiredMaxPolicy` are uncapped; the
//!   LM-side shadows extend them with optional caps (Sum / Budget) or alter
//!   the semantics (Max becomes `max(baseline, max(active))` rather than
//!   the base's `baseline + max(active)`).
//! - Swift measurement: [`mlx-swift-lm/.../WiredMemoryUtils.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryUtils.swift)
//!   — the `WiredMemoryMeasurement` struct + the `WiredMemoryUtils.tune`
//!   static helper.
//! - Tests: [`mlx-swift-lm/Tests/MLXLMTests/WiredMemoryPolicyTests.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Tests/MLXLMTests/WiredMemoryPolicyTests.swift)
//!   — the authoritative behavior contract this module is verified against.
//!
//! ## Formulas (mirrored verbatim from `WiredMemoryPolicies.swift`)
//! - [`WiredSumPolicy`]: `clamp(baseline + sum(active_sizes))` where
//!   `clamp(v) = min(v, cap.unwrap_or(recommended))` and `recommended` is
//!   [`recommended_working_set_bytes`](super::recommended_working_set_bytes).
//! - [`WiredMaxPolicy`]: `max(baseline, max(active_sizes))` — no cap (the
//!   LM-side max policy intentionally omits both the base's
//!   `baseline + max` form and the cap path; only the largest active
//!   ticket OR the baseline floor wins).
//! - [`WiredFixedPolicy`]: `limit` — ignores `baseline` and `active_sizes`.
//! - [`WiredBudgetPolicy`]: `clamp(baseline + base_bytes + sum(active_sizes))`
//!   — same clamp as Sum; `base_bytes` is the precomputed budget (weights
//!   plus workspace). Hashable / equality are by the policy's stable `id`
//!   (a string handle, mirroring Swift's `UUID`-based grouping) so multiple
//!   tickets can reference the same logical policy instance.
//!
//! ## Divergences from Swift
//! Two intentional adaptations to fit Rust idioms; neither alters policy
//! math:
//!   - **No protocol-side `Identifiable<ID = AnyHashable>`**: Rust's trait
//!     system does not have Swift's `Identifiable where ID == AnyHashable`
//!     erasure idiom (used by mlx-swift's `WiredMemoryManager` to key its
//!     internal policy map). The grouping `id` lives on each concrete
//!     policy struct as a public `&str` accessor (`id()`); a future
//!     manager port can wrap it however it likes.
//!   - **`WiredBudgetPolicy::id` is a `String`, not `UUID`**: mlxrs has no
//!     UUID dependency. A caller-supplied `String` (default: an internally
//!     monotonic counter) gives the same grouping contract — equal IDs →
//!     same `Hash` / `Eq` / `PartialEq` outcome — without pulling `uuid`.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
  Stream,
  error::{Error, InvariantViolationPayload, Result},
};

use super::recommended_working_set_bytes;

/// Compute a process-global wired-memory limit for a set of in-flight
/// tickets. Port of mlx-swift's `WiredMemoryPolicy` protocol
/// (`Source/MLX/WiredMemory.swift` lines 123-138).
///
/// Each policy maps the current `(baseline, active_sizes)` snapshot to a
/// desired limit in bytes. A higher-level manager (out of scope for this
/// port; a future LM concurrency surface) groups tickets by policy and uses
/// the per-policy maximum across groups as the actual limit it pushes to
/// [`set_wired_limit`](super::set_wired_limit).
///
/// The [`Self::can_admit`] method gates whether a new ticket of size
/// `new_size` should be admitted; by default it returns `true` (mirroring
/// the Swift protocol's default extension). Policies with a cap
/// ([`WiredSumPolicy`] / [`WiredBudgetPolicy`]) override it to deny
/// admission when the projected limit would exceed the cap.
pub trait WiredMemoryPolicy: Send + Sync {
  /// The desired wired-memory limit in bytes for the current active set.
  ///
  /// `baseline` is the always-on memory pressure (e.g. weights already
  /// resident); `active_sizes` is the per-ticket additional pressure
  /// (e.g. KV-cache bytes for each in-flight generation). Implementations
  /// MUST be pure (no side effects, no I/O) and idempotent.
  fn limit(&self, baseline: u64, active_sizes: &[u64]) -> u64;

  /// Whether a new ticket of size `new_size` can be admitted given the
  /// current `(baseline, active_sizes)` snapshot. Defaults to `true`
  /// (mirroring Swift's protocol-extension default).
  ///
  /// Cap-aware policies ([`WiredSumPolicy`] / [`WiredBudgetPolicy`])
  /// override this to deny admission whose projected limit would exceed
  /// the cap.
  fn can_admit(&self, _baseline: u64, _active_sizes: &[u64], _new_size: u64) -> bool {
    true
  }

  /// Stable grouping identifier — mirrors Swift's `Identifiable.id` shape.
  /// Default `""` matches the base mlx-swift `WiredMemoryPolicy where Self:
  /// Hashable` extension's implicit per-type identity (for policies whose
  /// equality is by full value rather than by an explicit grouping key).
  fn id(&self) -> &str {
    ""
  }
}

/// Sum policy: `clamp(baseline + sum(active_sizes))` — the most common LM
/// inference policy.
///
/// Each ticket adds to the wired limit; the total is proportional to the
/// concurrent demand. If `cap` is [`None`], the policy clamps to
/// [`recommended_working_set_bytes`](super::recommended_working_set_bytes)
/// when available; otherwise the clamp is a no-op (returns the raw sum).
///
/// Mirrors the Swift `WiredSumPolicy` in
/// [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
/// lines 31-61.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WiredSumPolicy {
  /// Optional absolute cap in bytes. [`None`] → clamp to
  /// [`recommended_working_set_bytes`](super::recommended_working_set_bytes).
  cap: Option<u64>,
}

impl WiredSumPolicy {
  /// Create a sum policy with the given optional cap. Mirrors Swift
  /// `init(cap: Int? = nil)`.
  pub fn new(cap: Option<u64>) -> Self {
    Self { cap }
  }

  /// The optional cap in bytes.
  #[inline(always)]
  pub const fn cap(&self) -> Option<u64> {
    self.cap
  }

  /// Builder: set the cap.
  #[must_use]
  pub fn with_cap(mut self, cap: Option<u64>) -> Self {
    self.cap = cap;
    self
  }

  /// Clamp `value` to the cap (if set) or the recommended working set (if
  /// available). Identical to Swift `private func clamp(_ value: Int) ->
  /// Int`.
  fn clamp(&self, value: u64) -> u64 {
    if let Some(cap) = self.cap {
      return value.min(cap);
    }
    if let Ok(Some(max_bytes)) = recommended_working_set_bytes() {
      return value.min(max_bytes);
    }
    value
  }
}

impl WiredMemoryPolicy for WiredSumPolicy {
  fn limit(&self, baseline: u64, active_sizes: &[u64]) -> u64 {
    let sum: u64 = active_sizes.iter().copied().sum();
    self.clamp(baseline.saturating_add(sum))
  }

  fn can_admit(&self, baseline: u64, active_sizes: &[u64], new_size: u64) -> bool {
    let projected = baseline
      .saturating_add(active_sizes.iter().copied().sum())
      .saturating_add(new_size);
    self.clamp(projected) == projected
  }
}

/// Max policy: `max(baseline, max(active_sizes))` — no cap.
///
/// Tracks the single largest demand (the largest active ticket OR the
/// baseline floor). Useful when you want the limit to scale with the
/// largest in-flight request rather than the sum.
///
/// Mirrors the Swift LM-side `WiredMaxPolicy` in
/// [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
/// lines 77-86. Note this is **distinct** from the base mlx-swift
/// `WiredMaxPolicy` (which uses `baseline + max(active)`); the LM variant
/// omits the additive form.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct WiredMaxPolicy;

impl WiredMaxPolicy {
  /// Create a max policy. Mirrors Swift `init()`.
  pub fn new() -> Self {
    Self
  }
}

impl WiredMemoryPolicy for WiredMaxPolicy {
  fn limit(&self, baseline: u64, active_sizes: &[u64]) -> u64 {
    let max_active = active_sizes.iter().copied().max().unwrap_or(0);
    baseline.max(max_active)
  }
}

/// Fixed policy: constant `limit` regardless of baseline / active sizes.
///
/// Mirrors the Swift `WiredFixedPolicy` in
/// [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
/// lines 101-114.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WiredFixedPolicy {
  /// The constant limit in bytes returned by [`WiredMemoryPolicy::limit`].
  limit_bytes: u64,
}

impl WiredFixedPolicy {
  /// Create a fixed policy with the given byte limit. Mirrors Swift
  /// `init(limit: Int)`.
  pub fn new(limit_bytes: u64) -> Self {
    Self { limit_bytes }
  }

  /// The constant limit in bytes.
  #[inline(always)]
  pub const fn limit_bytes(&self) -> u64 {
    self.limit_bytes
  }

  /// Builder: set the limit in bytes.
  #[must_use]
  pub fn with_limit_bytes(mut self, b: u64) -> Self {
    self.limit_bytes = b;
    self
  }
}

impl WiredMemoryPolicy for WiredFixedPolicy {
  fn limit(&self, _baseline: u64, _active_sizes: &[u64]) -> u64 {
    self.limit_bytes
  }
}

/// Budget policy: `clamp(baseline + base_bytes + sum(active_sizes))` with
/// stable id-based grouping.
///
/// Bakes a precomputed `base_bytes` budget (e.g. weights + workspace, as
/// produced by a [`WiredMemoryMeasurement`] pass) into the limit while
/// still accounting for active tickets. Equality / hashing is by `id` so
/// multiple ticket sites can reference the same logical policy without
/// reconstructing the struct.
///
/// Mirrors the Swift `WiredBudgetPolicy` in
/// [`mlx-swift-lm/.../WiredMemoryPolicies.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryPolicies.swift)
/// lines 134-182.
#[derive(Debug, Clone)]
pub struct WiredBudgetPolicy {
  /// Stable grouping identifier — mirrors Swift `UUID`-based grouping.
  /// Two budget policies with the same `id` compare equal and produce the
  /// same hash; their `base_bytes` / `cap` values are NOT consulted for
  /// equality (a Swift parity choice — see
  /// `WiredMemoryPolicies.swift::==`).
  id: String,
  /// Base budget in bytes (e.g. weights + workspace). Clamped to `>= 0` at
  /// construction; mirrors Swift `self.baseBytes = max(0, baseBytes)`. The
  /// Rust signature already enforces `u64`, so the clamp is implicit.
  base_bytes: u64,
  /// Optional absolute cap in bytes. [`None`] → clamp to
  /// [`recommended_working_set_bytes`](super::recommended_working_set_bytes).
  cap: Option<u64>,
}

/// Process-wide monotonic counter for the default `id` of a
/// [`WiredBudgetPolicy`] constructed without an explicit id. Mirrors
/// Swift's `id: UUID = UUID()` default (fresh-id-per-instance, with the
/// uniqueness guarantee scoped to the running process — adequate for
/// in-process grouping; a remote / cross-process consumer that needs
/// stronger uniqueness can pass an explicit `id`).
static AUTO_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl WiredBudgetPolicy {
  /// Create a budget policy with the given `base_bytes` + optional `cap`,
  /// auto-assigning a fresh in-process `id` (the format `"auto-{N}"` with
  /// a monotonic per-process counter). Mirrors Swift
  /// `init(baseBytes: Int, cap: Int? = nil, id: UUID = UUID())`.
  ///
  /// For deterministic grouping across recreations, use
  /// [`Self::with_id`] instead.
  pub fn new(base_bytes: u64, cap: Option<u64>) -> Self {
    let n = AUTO_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    Self::with_id(format!("auto-{n}"), base_bytes, cap)
  }

  /// Create a budget policy with an explicit grouping `id`. Two budget
  /// policies sharing an `id` compare equal even if their `base_bytes` /
  /// `cap` differ — matching the Swift `static func == (lhs, rhs) ->
  /// Bool { lhs.identifier == rhs.identifier }` contract.
  pub fn with_id(id: impl Into<String>, base_bytes: u64, cap: Option<u64>) -> Self {
    Self {
      id: id.into(),
      base_bytes,
      cap,
    }
  }

  /// The grouping identifier — same string as [`WiredMemoryPolicy::id`],
  /// surfaced as an inherent method so callers that hold a
  /// `WiredBudgetPolicy` directly do not have to import the trait to read
  /// it (Rust trait-method dispatch requires the trait be in scope).
  #[inline(always)]
  pub fn id(&self) -> &str {
    &self.id
  }

  /// Legacy alias of [`Self::id`] retained for one minor-version cycle.
  /// New code should use [`Self::id`] directly (a `String` getter takes the
  /// field name, not a `_str` suffix).
  #[inline(always)]
  pub fn id_str(&self) -> &str {
    self.id()
  }

  /// The base budget in bytes.
  #[inline(always)]
  pub const fn base_bytes(&self) -> u64 {
    self.base_bytes
  }

  /// Builder: set the base budget in bytes.
  #[must_use]
  pub fn with_base_bytes(mut self, b: u64) -> Self {
    self.base_bytes = b;
    self
  }

  /// The optional cap in bytes.
  #[inline(always)]
  pub const fn cap(&self) -> Option<u64> {
    self.cap
  }

  /// Builder: set the cap.
  #[must_use]
  pub fn with_cap(mut self, cap: Option<u64>) -> Self {
    self.cap = cap;
    self
  }

  /// Clamp `value` to the cap (if set) or the recommended working set (if
  /// available). Identical to Swift `private func clamp(_ value: Int) ->
  /// Int`.
  fn clamp(&self, value: u64) -> u64 {
    if let Some(cap) = self.cap {
      return value.min(cap);
    }
    if let Ok(Some(max_bytes)) = recommended_working_set_bytes() {
      return value.min(max_bytes);
    }
    value
  }
}

impl PartialEq for WiredBudgetPolicy {
  /// Equality is by `id` alone — matches Swift
  /// `static func == (lhs, rhs) -> Bool { lhs.identifier == rhs.identifier }`.
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl Eq for WiredBudgetPolicy {}

impl std::hash::Hash for WiredBudgetPolicy {
  /// Hash is by `id` alone — matches Swift `hash(into:) { hasher.combine(identifier) }`.
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.id.hash(state);
  }
}

impl WiredMemoryPolicy for WiredBudgetPolicy {
  fn limit(&self, baseline: u64, active_sizes: &[u64]) -> u64 {
    let sum: u64 = active_sizes.iter().copied().sum();
    self.clamp(baseline.saturating_add(self.base_bytes).saturating_add(sum))
  }

  fn can_admit(&self, baseline: u64, active_sizes: &[u64], new_size: u64) -> bool {
    let projected = baseline
      .saturating_add(self.base_bytes)
      .saturating_add(active_sizes.iter().copied().sum())
      .saturating_add(new_size);
    self.clamp(projected) == projected
  }

  fn id(&self) -> &str {
    &self.id
  }
}

/// Result of a runtime wired-memory measurement pass — the data needed to
/// construct a `WiredBudgetPolicy` or compare against a manual estimate.
///
/// Mirrors the Swift `WiredMemoryMeasurement` in
/// [`mlx-swift-lm/.../WiredMemoryUtils.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryUtils.swift)
/// lines 8-26.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WiredMemoryMeasurement {
  /// Total bytes for model weights — sum of `Array::nbytes()` over the
  /// model's weight tree. Mirrors Swift `weightBytes`.
  weight_bytes: u64,
  /// Total bytes for KV-cache state after prefill (sum of `Array::nbytes()`
  /// over every cache array). Mirrors Swift `kvBytes`.
  kv_bytes: u64,
  /// Estimated transient workspace bytes — `max(0, peak_active_bytes -
  /// weight_bytes - kv_bytes)`. Mirrors Swift `workspaceBytes`.
  workspace_bytes: u64,
  /// Peak [`crate::memory::active_memory`](super::active_memory) observed
  /// during prefill. Mirrors Swift `peakActiveBytes`.
  peak_active_bytes: u64,
  /// Number of tokens used during the prefill measurement. Mirrors Swift
  /// `tokenCount`.
  token_count: usize,
  /// Prefill step size used for the measurement. Mirrors Swift
  /// `prefillStepSize`.
  prefill_step_size: usize,
}

impl WiredMemoryMeasurement {
  /// Construct a measurement record from its components.
  pub fn new(
    weight_bytes: u64,
    kv_bytes: u64,
    workspace_bytes: u64,
    peak_active_bytes: u64,
    token_count: usize,
    prefill_step_size: usize,
  ) -> Self {
    Self {
      weight_bytes,
      kv_bytes,
      workspace_bytes,
      peak_active_bytes,
      token_count,
      prefill_step_size,
    }
  }

  /// Total bytes for model weights. Mirrors Swift `weightBytes`.
  #[inline(always)]
  pub fn weight_bytes(&self) -> u64 {
    self.weight_bytes
  }

  /// Total bytes for KV-cache state after prefill. Mirrors Swift `kvBytes`.
  #[inline(always)]
  pub fn kv_bytes(&self) -> u64 {
    self.kv_bytes
  }

  /// Estimated transient workspace bytes. Mirrors Swift `workspaceBytes`.
  #[inline(always)]
  pub fn workspace_bytes(&self) -> u64 {
    self.workspace_bytes
  }

  /// Peak active memory observed during prefill. Mirrors Swift `peakActiveBytes`.
  #[inline(always)]
  pub fn peak_active_bytes(&self) -> u64 {
    self.peak_active_bytes
  }

  /// Number of tokens used during the prefill measurement. Mirrors Swift `tokenCount`.
  #[inline(always)]
  pub fn token_count(&self) -> usize {
    self.token_count
  }

  /// Prefill step size used for the measurement. Mirrors Swift `prefillStepSize`.
  #[inline(always)]
  pub fn prefill_step_size(&self) -> usize {
    self.prefill_step_size
  }

  /// Combined budget suggestion = `weight_bytes + kv_bytes + workspace_bytes`.
  /// Mirrors Swift `var totalBytes: Int`.
  pub fn total_bytes(&self) -> u64 {
    self
      .weight_bytes
      .saturating_add(self.kv_bytes)
      .saturating_add(self.workspace_bytes)
  }
}

/// Run a trial prefill pass and return a [`WiredMemoryMeasurement`].
///
/// Port of Swift `WiredMemoryUtils.tune(...)` in
/// [`mlx-swift-lm/.../WiredMemoryUtils.swift`](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/WiredMemoryUtils.swift)
/// lines 137-247.
///
/// **STATUS: not yet implemented.** The Swift implementation requires a
/// model-trait `prefill_only` entry point that lives in mlxrs's still-stub
/// LM concurrency surface (the equivalent of mlx-swift-lm's
/// `LanguageModel.prepare(_:cache:windowSize:)`). The signature is shipped
/// here so callers can program against it; the body returns
/// [`Error::InvariantViolation`] with an actionable message until the upstream
/// concurrency surface lands. See follow-up
/// [issue #168](https://github.com/findit-studio/mlxrs/issues/168)
/// for the tracking item.
///
/// Caller contract (matches Swift): runs the model's prefill loop with
/// exactly `token_count` synthetic tokens, observes the peak active memory,
/// then returns the {weights, kv, workspace, peak} breakdown.
pub fn tune(
  _model_bytes: u64,
  _token_count: usize,
  _prefill_step_size: usize,
  _streams: &[Stream],
) -> Result<WiredMemoryMeasurement> {
  Err(Error::InvariantViolation(InvariantViolationPayload::new(
    "WiredMemoryUtils::tune",
    "not yet implemented — requires Model::prefill_only (stub); see issue #168",
  )))
}
