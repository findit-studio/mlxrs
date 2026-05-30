//! Distillation training losses — `kl_div_loss` + `js_div_loss`.
//!
//! Faithful port of `mlx-lm/mlx_lm/tuner/losses.py` (798 lines): two
//! distillation losses, each backed by hand-written Metal kernels for the
//! forward AND backward passes plus a [`crate::transforms::custom_vjp`] that
//! wires those kernels into autograd. mlx-lm uses these hand-rolled kernels
//! (instead of an autograd-derivable formulation) because the naive
//! `mx.exp(p) * (logp - logq)` transcription is numerically unstable at the
//! long-sequence training scales where distillation matters.
//!
//! # MSL source verbatim from python
//!
//! The four Metal Shading Language kernel sources below (KL forward / KL
//! backward / JS forward / JS backward) are **byte-for-byte transliterations**
//! of the source-strings emitted by `_make_kl_forward_kernel` /
//! `_make_kl_backward_kernel` / `_make_js_forward_kernel` /
//! `_make_js_backward_kernel` in the python reference. Preserving the
//! emitted source means mlx-c's JIT produces an identical Metal IR (and
//! therefore identical numerics) to the python implementation.
//!
//! # Lazy-init pattern
//!
//! Each kernel is constructed once per thread via a [`thread_local!`] +
//! [`std::cell::OnceCell`] — [`MetalKernel`] is `!Send + !Sync` (the
//! underlying `mlx_fast_metal_kernel` shares the same thread-local backend
//! state as [`Array`] / [`Stream`]), so a `static OnceLock<MetalKernel>`
//! would not compile. mlx-c itself caches the JIT-compiled pipeline keyed by
//! the kernel `name`, so the per-thread `MetalKernel::new` call after the
//! first thread is cheap (handle-construction only — no recompile).
//!
//! # Surface
//!
//! - [`kl_div_loss`] — KL-divergence loss `D_KL(p || q)` between logits.
//!   Returns an array shaped `logits_q.shape[:-1]`. Differentiable via the
//!   custom backward kernel.
//! - [`js_div_loss`] — Jensen-Shannon divergence loss
//!   `0.5 * (D_KL(p || m) + D_KL(q || m))` where `m = mean(p, q)`. Returns
//!   an array shaped `logits_q.shape[:-1]`. Differentiable via the custom
//!   backward kernel.
//!
//! [`Array`]: crate::array::Array
//! [`Stream`]: crate::stream::Stream

use std::cell::OnceCell;

use smol_str::format_smolstr;

use crate::{
  array::Array,
  dtype::Dtype,
  error::{
    DtypeMismatchPayload, EmptyInputPayload, Error, LengthMismatchPayload, OutOfRangePayload,
    RankMismatchPayload, Result, ShapePairMismatchPayload, UnsupportedDtypePayload,
  },
  ops::fast::metal_kernel::{KernelTemplateArg, MetalKernel, MetalKernelApplyConfig},
  transforms::custom_vjp,
};

// ───────────────────────── MSL kernel sources (verbatim from python) ─────────────────────────

/// MSL source for the KL-divergence forward kernel. Byte-for-byte transcription
/// of `_make_kl_forward_kernel`'s `source` string in
/// `mlx_lm/tuner/losses.py` (the body between `source = """` and the closing
/// `"""`). Re-formatting / re-flowing this string changes the emitted Metal IR
/// and breaks numerical parity with python — leave the verbatim layout intact.
const fn kl_forward_msl_source() -> &'static str {
  r#"
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out += out_idx;

    float lse_q_minus_p;
    float lse_p;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q_minus_p = max_q + metal::fast::log(sum_exp_q) - lse_p;
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float kl = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and add to the kl
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                kl += metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                kl += metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p);
            }
        }

        // Add the kl across the threadgroup
        kl = simd_sum(kl);
        if (simd_lane_id == 0) {
            shared[simd_group_id] = kl;
        }
        threadgroup_barrier(mem_flags::mem_none);
        kl = shared[simd_lane_id];
        kl = simd_sum(kl);

        if (thread_index_in_threadgroup == 0) {
            out[0] = static_cast<T>(kl);
        }
    }
    "#
}

/// MSL source for the KL-divergence backward kernel — byte-for-byte from
/// `_make_kl_backward_kernel` in the python reference.
const fn kl_backward_msl_source() -> &'static str {
  r#"
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out += out_idx * V;
    cotan += out_idx;

    float lse_q;
    float lse_p;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q = max_q + metal::fast::log(sum_exp_q);
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float kl = 0;
        float c = cotan[0];

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and add to the kl
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                out[offset + j] = static_cast<T>(
                    c * (metal::fast::exp(vals_q[j] - lse_q) - metal::fast::exp(vals_p[j] - lse_p)));
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                if (offset + j < V) {
                    out[offset + j] = static_cast<T>(
                        c * (metal::fast::exp(vals_q[j] - lse_q) - metal::fast::exp(vals_p[j] - lse_p)));
                }
            }
        }
    }
    "#
}

/// MSL source for the JS-divergence forward kernel — byte-for-byte from
/// `_make_js_forward_kernel` in the python reference. Emits TWO outputs:
/// the loss `out` and a per-row `out_kl_q` consumed by the backward kernel.
const fn js_forward_msl_source() -> &'static str {
  r#"
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out += out_idx;
    out_kl_q += out_idx;

    float lse_p;
    float lse_q;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q = max_q + metal::fast::log(sum_exp_q);
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float kl_p = 0;
        float kl_q = 0;
        const float logtwo = metal::fast::log(static_cast<float>(2));

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and add to the kl_p and kl_q
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                float logp_j = vals_p[j] - lse_p;
                float logq_j = vals_q[j] - lse_q;
                float p_j = metal::fast::exp(logp_j);
                float q_j = metal::fast::exp(logq_j);
                kl_p += p_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logq_j - logp_j)));
                kl_q += q_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j)));
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                float logp_j = vals_p[j] - lse_p;
                float logq_j = vals_q[j] - lse_q;
                float p_j = metal::fast::exp(logp_j);
                float q_j = metal::fast::exp(logq_j);
                kl_p += p_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logq_j - logp_j)));
                kl_q += q_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j)));
            }
        }

        // Add the kl_p and kl_q across the threadgroup
        kl_p = simd_sum(kl_p);
        kl_q = simd_sum(kl_q);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = kl_p;
            shared[simd_group_id * 2 + 1] = kl_q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kl_p = shared[simd_lane_id * 2 + 0];
        kl_q = shared[simd_lane_id * 2 + 1];
        kl_p = simd_sum(kl_p);
        kl_q = simd_sum(kl_q);

        if (thread_index_in_threadgroup == 0) {
            out[0] = static_cast<T>(0.5 * kl_p + 0.5 * kl_q);
            out_kl_q[0] = static_cast<T>(kl_q);
        }
    }
    "#
}

/// MSL source for the JS-divergence backward kernel — byte-for-byte from
/// `_make_js_backward_kernel` in the python reference.
const fn js_backward_msl_source() -> &'static str {
  r#"
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out_q += out_idx * V;
    cotan += out_idx;
    output_kl_q += out_idx;

    float lse_q;
    float lse_p;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q = max_q + metal::fast::log(sum_exp_q);
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float c = cotan[0];
        const float logtwo = metal::fast::log(static_cast<float>(2));
        float kl_q = output_kl_q[0];

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and compute vjp for logits_q
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                float logp_j = vals_p[j] - lse_p;
                float logq_j = vals_q[j] - lse_q;
                float q_j = metal::fast::exp(logq_j);
                out_q[offset + j] = static_cast<T>(
                    c * 0.5 * q_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j)) - kl_q)
                );
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                if (offset + j < V) {
                    float logp_j = vals_p[j] - lse_p;
                    float logq_j = vals_q[j] - lse_q;
                    float q_j = metal::fast::exp(logq_j);
                    out_q[offset + j] = static_cast<T>(
                        c * 0.5 * q_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j)) - kl_q)
                    );
                }
            }
        }
    }
    "#
}

// ───────────────────────── Lazy-init helpers (thread_local OnceCell) ─────────────────────────

// `MetalKernel` is `!Send + !Sync` (see `ops::fast::metal_kernel::MetalKernel`
// docs — its `mlx_fast_metal_kernel` shares the same thread-local backend
// state as `Array` / `Stream`). Therefore a `static OnceLock<MetalKernel>`
// will not compile (`OnceLock` requires `T: Sync`), and we route the lazy
// init through `thread_local!` + `std::cell::OnceCell` instead.
//
// mlx-c itself caches the JIT-compiled pipeline keyed on the kernel `name`,
// so the per-thread `MetalKernel::new` after the first thread is cheap
// handle-construction (no recompile). The first-call init on each thread
// allocates a fresh wrapper, and subsequent calls on that thread reuse it.

thread_local! {
  static KL_FORWARD: OnceCell<MetalKernel> = const { OnceCell::new() };
  static KL_BACKWARD: OnceCell<MetalKernel> = const { OnceCell::new() };
  static JS_FORWARD: OnceCell<MetalKernel> = const { OnceCell::new() };
  static JS_BACKWARD: OnceCell<MetalKernel> = const { OnceCell::new() };
}

/// `with` over a thread-local `OnceCell<MetalKernel>` that lazily constructs
/// the kernel via `build` on the first hit and hands the same `&MetalKernel`
/// to `f` on every call.
///
/// The `build` closure is fallible (kernel construction can fail if the MSL
/// source has a syntax error mlx-c can detect early); the outer `f` is
/// fallible because the calling apply may fail at the FFI boundary. We chain
/// both through `Result` so an init failure on the FIRST call surfaces as an
/// error to that caller (subsequent calls would re-attempt — an init failure
/// does not "stick" in the cell).
fn with_kernel<F, R>(
  cell: &'static std::thread::LocalKey<OnceCell<MetalKernel>>,
  build: impl FnOnce() -> Result<MetalKernel>,
  f: F,
) -> Result<R>
where
  F: FnOnce(&MetalKernel) -> Result<R>,
{
  // Step 1: try the fast path — if the cell is initialized, run `f` and return.
  // If not initialized, we fall through to step 2 and build outside of `with`
  // to avoid borrowing the LocalKey across a fallible build.
  let already_initialized = cell.with(|c| c.get().is_some());
  if !already_initialized {
    let kernel = build()?;
    // `set` succeeds if the cell was still empty; if another `set` won the
    // race on this same thread (impossible since `thread_local` is single-
    // threaded but harmless), we discard our `kernel` and use the winner.
    let _ = cell.with(|c| c.set(kernel));
  }
  cell.with(|c| {
    let kernel = c
      .get()
      .expect("kernel must be initialized by the preceding set");
    f(kernel)
  })
}

/// Build the KL forward kernel. mlx-c JIT-compiles + caches the Metal
/// pipeline keyed on the `name`, so each thread's first `new` after the
/// pipeline is cached is just handle-construction work.
fn build_kl_forward_kernel() -> Result<MetalKernel> {
  MetalKernel::new(
    "kl_forward",
    &["logits_q", "logits_p"],
    &["out"],
    kl_forward_msl_source(),
    /* header */ "",
    /* ensure_row_contiguous */ true,
    /* atomic_outputs */ false,
  )
}

/// Build the KL backward kernel.
fn build_kl_backward_kernel() -> Result<MetalKernel> {
  MetalKernel::new(
    "kl_backward",
    &["logits_q", "logits_p", "cotan"],
    &["out"],
    kl_backward_msl_source(),
    "",
    true,
    false,
  )
}

/// Build the JS forward kernel. Emits TWO outputs (`out`, `out_kl_q`).
fn build_js_forward_kernel() -> Result<MetalKernel> {
  MetalKernel::new(
    "js_forward",
    &["logits_q", "logits_p"],
    &["out", "out_kl_q"],
    js_forward_msl_source(),
    "",
    true,
    false,
  )
}

/// Build the JS backward kernel.
fn build_js_backward_kernel() -> Result<MetalKernel> {
  MetalKernel::new(
    "js_backward",
    &["logits_q", "logits_p", "cotan", "output_kl_q"],
    &["out_q"],
    js_backward_msl_source(),
    "",
    true,
    false,
  )
}

// ───────────────────────── Shape + dtype helpers ─────────────────────────

/// Compute `n_outs = logits.size // logits.shape[-1]` — the number of
/// "rows" the kernel iterates over (mlx-lm Python computes this as
/// `logits_q.size // logits_q.shape[-1]`). Returned as an `i32` because
/// mlx-c's `grid` / `thread_group` are i32-typed.
fn n_outs_of(logits: &Array) -> Result<i32> {
  let shape = logits.shape();
  let v = shape.last().copied().ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      "mlxrs::lm::tuner::losses: logits must have rank >= 1",
      0,
      Vec::new(),
    ))
  })?;
  if v == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      "mlxrs::lm::tuner::losses: logits last dimension",
      "must be > 0",
      "0",
    )));
  }
  let total: usize = shape.iter().product();
  let n_outs = total / v;
  i32::try_from(n_outs).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mlxrs::lm::tuner::losses: n_outs",
      "must fit in i32",
      format_smolstr!("{n_outs}"),
    ))
  })
}

/// Compute `V = logits.shape[-1]` cast to `i32` (the per-row vocab size).
fn vocab_of(logits: &Array) -> Result<i32> {
  let shape = logits.shape();
  let v = shape.last().copied().ok_or_else(|| {
    Error::RankMismatch(RankMismatchPayload::new(
      "mlxrs::lm::tuner::losses: logits must have rank >= 1",
      0,
      Vec::new(),
    ))
  })?;
  i32::try_from(v).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mlxrs::lm::tuner::losses: vocab size",
      "must fit in i32",
      format_smolstr!("{v}"),
    ))
  })
}

/// Return `logits_q.shape[:-1]` as a `Vec<i32>` for the kernel's
/// `output_shapes` slot. The python `_kl_div_loss` uses
/// `logits_q.shape[:-1]` directly; we mirror that shape rather than
/// flattening to `[n_outs]` (the runtime element count is identical;
/// preserving the rank makes the output `[B, S]` rather than `[B*S]` for a
/// `[B, S, V]` input, matching python).
fn leading_shape_i32(logits: &Array) -> Result<Vec<i32>> {
  let shape = logits.shape();
  if shape.is_empty() {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      "mlxrs::lm::tuner::losses: logits must have rank >= 1",
      0,
      Vec::new(),
    )));
  }
  shape[..shape.len() - 1]
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "mlxrs::lm::tuner::losses: shape dim",
          "must fit in i32",
          format_smolstr!("{d}"),
        ))
      })
    })
    .collect()
}

/// Return the full `logits_q.shape` as `Vec<i32>` for the backward kernel's
/// `output_shapes` slot (the backward emits `[..., V]` — same shape as the
/// primal).
fn full_shape_i32(logits: &Array) -> Result<Vec<i32>> {
  logits
    .shape()
    .iter()
    .map(|&d| {
      i32::try_from(d).map_err(|_| {
        Error::OutOfRange(OutOfRangePayload::new(
          "mlxrs::lm::tuner::losses: shape dim",
          "must fit in i32",
          format_smolstr!("{d}"),
        ))
      })
    })
    .collect()
}

/// Validate that `logits_q` and `logits_p` agree in shape AND dtype, that
/// the rank is at least 2 (required for the kernel's `[..., V]` layout —
/// see contract note on [`kl_div_loss`] / [`js_div_loss`]), and that the
/// dtype is one of the deliberately-supported floating types (`F32`,
/// `F16`, `BF16`). mlx-c would error at JIT time on a shape/dtype mismatch
/// (the template `T` is shared, the kernel reads element-wise into both
/// buffers); we reject early so the caller gets a precise wrapper-level
/// error.
///
/// Rank-1 inputs `[V]` would yield `logits.shape[:-1] == []` (a scalar
/// Metal output), which the shared kernel wrapper rejects with `custom
/// Metal kernel outputs must have rank >= 1`; we surface a clearer
/// contract message before the kernel apply. Upstream `mlx_lm` always
/// invokes these losses on `[B, ..., V]`-shaped logits from a model's
/// forward pass, so this matches real usage; scalar-output support would
/// require kernel-side changes not justified for v1.
///
/// Integer / boolean / `F64` / `Complex64` dtypes are rejected because the
/// kernel uses the input dtype as its template `T` and casts the
/// floating-point divergence and gradient expressions back to `T` — same-
/// dtype integer inputs would silently truncate to integer KL / JS values
/// and derivatives rather than rejecting an invalid numerical mode. Cast
/// with [`Array::astype`] before calling if your logits are not already
/// floating-point.
///
/// Zero-width vocab (last dim == 0) and i32-overflowing dims are also
/// rejected here with [`Error::OutOfRange`], BEFORE the dtype checks,
/// to match the documented precedence on [`kl_div_loss`] /
/// [`js_div_loss`] (step 2 = shape-class errors, step 3 = dtype mismatch,
/// step 4 = dtype admissibility). The shape-class rejections live in this
/// validator so the public contract holds for every reachable dtype
/// combination, including unsupported-but-equal dtypes (e.g. `i32 vs
/// i32`) and mismatched dtypes (e.g. `f32 vs f16`).
///
/// [`Array::astype`]: crate::array::Array::astype
#[allow(clippy::too_many_arguments)]
fn validate_inputs(
  logits_q: &Array,
  logits_p: &Array,
  ctx_q: &'static str,
  ctx_p: &'static str,
  ctx_pair: &'static str,
  ctx_last: &'static str,
  ctx_dim: &'static str,
  ctx_dtype: &'static str,
) -> Result<()> {
  let sq = logits_q.shape();
  let sp = logits_p.shape();
  // Rank >= 2 runs BEFORE the shape comparison so that a mismatched-rank
  // pair (e.g. rank-1 `logits_q` vs rank-2 `logits_p`) surfaces the
  // rank-rejection guidance — which tells the caller how to fix it
  // (reshape to `[1, V]`) — rather than a generic ShapePairMismatch that
  // hides the underlying contract. We check BOTH inputs so the error
  // names whichever side is rank-deficient. Scalar Metal outputs
  // (shape `[]`) are rejected by the shared kernel wrapper; surface a
  // contract message here instead. Mirrors the upstream `mlx_lm`
  // convention of `[B, ..., V]` logits from a model's forward pass.
  if logits_q.ndim() < 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      ctx_q,
      logits_q.ndim() as u32,
      sq.to_vec(),
    )));
  }
  if logits_p.ndim() < 2 {
    return Err(Error::RankMismatch(RankMismatchPayload::new(
      ctx_p,
      logits_p.ndim() as u32,
      sp.to_vec(),
    )));
  }
  if sq != sp {
    return Err(Error::ShapePairMismatch(ShapePairMismatchPayload::new(
      ctx_pair,
      sq.to_vec(),
      sp.to_vec(),
    )));
  }
  // Zero-last-dim + i32-overflow checks run BEFORE dtype checks so the
  // documented precedence (step 2 = OutOfRange for "last dim is 0 or
  // any dim overflows i32") matches actual behavior. Without this, equal
  // `i32` arrays shaped `[1, 0]` would route through the dtype-admissibility
  // Backend error (step 4) and mixed-dtype `[1, 0]` arrays would route
  // through DtypeMismatch (step 3) — contrary to the contract. We also
  // mirror these checks defensively in `n_outs_of` / `leading_shape_i32` /
  // `full_shape_i32`, but they'll never fire on the public path now.
  let last = *sq.last().expect("rank>=2 guaranteed by checks above");
  if last == 0 {
    return Err(Error::OutOfRange(OutOfRangePayload::new(
      ctx_last,
      "must be > 0",
      "0",
    )));
  }
  for &d in sq.iter() {
    if i32::try_from(d).is_err() {
      return Err(Error::OutOfRange(OutOfRangePayload::new(
        ctx_dim,
        "must fit in i32",
        format_smolstr!("{d}"),
      )));
    }
  }
  let dq = logits_q.dtype()?;
  let dp = logits_p.dtype()?;
  if dq != dp {
    return Err(Error::DtypeMismatch(DtypeMismatchPayload::new(dq, dp)));
  }
  // Floating-only: the kernel casts the divergence + gradient expressions
  // back to `T == input dtype`, so an integer / boolean `T` would silently
  // truncate the result. Reject explicitly so users either cast up-front or
  // see a clear error rather than corrupted numerics.
  match dq {
    Dtype::F32 | Dtype::F16 | Dtype::BF16 => {}
    _ => {
      return Err(Error::UnsupportedDtype(UnsupportedDtypePayload::new(
        ctx_dtype,
        dq,
        &[Dtype::F32, Dtype::F16, Dtype::BF16],
      )));
    }
  }
  Ok(())
}

/// Build the `template = [("T", dtype), ("V", vocab)]` slot common to all
/// four kernels.
fn template_for(dtype: Dtype, vocab: i32) -> Vec<(String, KernelTemplateArg)> {
  vec![
    ("T".to_string(), KernelTemplateArg::Dtype(dtype)),
    ("V".to_string(), KernelTemplateArg::Int(vocab)),
  ]
}

// ───────────────────────── KL forward / backward apply ─────────────────────────

/// Apply the KL forward kernel — single output `[..., 1]`-rank reduced.
/// Python ref `_kl_div_loss` (forward):
/// ```python
/// _kl_forward_kernel(
///     inputs=[logits_q, logits_p],
///     output_shapes=[logits_q.shape[:-1]],
///     output_dtypes=[dt],
///     template=[("T", dt), ("V", logits_q.shape[-1])],
///     grid=(1024, n_outs, 1),
///     threadgroup=(1024, 1, 1),
/// )[0]
/// ```
fn kl_forward_apply(logits_q: &Array, logits_p: &Array) -> Result<Array> {
  let dtype = logits_q.dtype()?;
  let vocab = vocab_of(logits_q)?;
  let n_outs = n_outs_of(logits_q)?;
  let out_shape = leading_shape_i32(logits_q)?;

  let cfg = MetalKernelApplyConfig::new(
    /* grid */ [1024, n_outs as u32, 1],
    /* thread_group */ [1024, 1, 1],
    /* output_shapes */ vec![out_shape],
    /* output_dtypes */ vec![dtype],
  )?
  .with_template(template_for(dtype, vocab));

  with_kernel(&KL_FORWARD, build_kl_forward_kernel, |kernel| {
    let mut outputs = kernel.apply(&[logits_q, logits_p], &cfg)?;
    Ok(outputs.swap_remove(0))
  })
}

/// Apply the KL backward kernel — full-rank gradient `[..., V]` w.r.t. the
/// primal `logits_q`. Python ref `_kl_div_loss.vjp`:
/// ```python
/// _kl_backward_kernel(
///     inputs=[logits_q, logits_p, cotangent],
///     output_shapes=[logits_q.shape],
///     output_dtypes=[dt],
///     template=[("T", dt), ("V", logits_q.shape[-1])],
///     grid=(1024, cotangent.size, 1),
///     threadgroup=(1024, 1, 1),
/// )[0]
/// ```
/// `cotangent.size` is the count of cotangent elements — which equals
/// `n_outs` (the cotangent is shaped `[..., 1]` after the forward's
/// reduction, with `n_outs` total elements).
fn kl_backward_apply(logits_q: &Array, logits_p: &Array, cotangent: &Array) -> Result<Array> {
  let dtype = logits_q.dtype()?;
  let vocab = vocab_of(logits_q)?;
  let cot_size = i32::try_from(cotangent.size()).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mlxrs::lm::tuner::losses::kl_backward: cotangent size",
      "must fit in i32",
      format_smolstr!("{}", cotangent.size()),
    ))
  })?;
  let out_shape = full_shape_i32(logits_q)?;

  let cfg = MetalKernelApplyConfig::new(
    [1024, cot_size as u32, 1],
    [1024, 1, 1],
    vec![out_shape],
    vec![dtype],
  )?
  .with_template(template_for(dtype, vocab));

  with_kernel(&KL_BACKWARD, build_kl_backward_kernel, |kernel| {
    let mut outputs = kernel.apply(&[logits_q, logits_p, cotangent], &cfg)?;
    Ok(outputs.swap_remove(0))
  })
}

// ───────────────────────── JS forward / backward apply ─────────────────────────

/// Apply the JS forward kernel — emits TWO outputs:
/// `(loss, kl_q)`. Both have shape `logits_q.shape[:-1]`. The python ref
/// wraps `kl_q` in `mx.stop_gradient`; in Rust the custom_vjp completely
/// overrides autograd (the user-supplied backward is the only path through
/// which gradients reach the primals), so the explicit stop_gradient is
/// unnecessary — gradients never flow through `kl_q` regardless.
fn js_forward_apply(logits_q: &Array, logits_p: &Array) -> Result<(Array, Array)> {
  let dtype = logits_q.dtype()?;
  let vocab = vocab_of(logits_q)?;
  let n_outs = n_outs_of(logits_q)?;
  let leading = leading_shape_i32(logits_q)?;

  let cfg = MetalKernelApplyConfig::new(
    [1024, n_outs as u32, 1],
    [1024, 1, 1],
    vec![leading.clone(), leading],
    vec![dtype, dtype],
  )?
  .with_template(template_for(dtype, vocab));

  with_kernel(&JS_FORWARD, build_js_forward_kernel, |kernel| {
    let mut outputs = kernel.apply(&[logits_q, logits_p], &cfg)?;
    if outputs.len() != 2 {
      return Err(Error::LengthMismatch(LengthMismatchPayload::new(
        "mlxrs::lm::tuner::losses::js_forward: kernel outputs",
        2,
        outputs.len(),
      )));
    }
    let kl_q = outputs.swap_remove(1);
    let loss = outputs.swap_remove(0);
    Ok((loss, kl_q))
  })
}

/// Apply the JS backward kernel — full-rank gradient w.r.t. `logits_q`.
/// Python ref `_js_div_loss.vjp` passes `[logits_q, logits_p, cotan, kl_q]`
/// as inputs.
fn js_backward_apply(
  logits_q: &Array,
  logits_p: &Array,
  cotan: &Array,
  kl_q: &Array,
) -> Result<Array> {
  let dtype = logits_q.dtype()?;
  let vocab = vocab_of(logits_q)?;
  let cot_size = i32::try_from(cotan.size()).map_err(|_| {
    Error::OutOfRange(OutOfRangePayload::new(
      "mlxrs::lm::tuner::losses::js_backward: cotan size",
      "must fit in i32",
      format_smolstr!("{}", cotan.size()),
    ))
  })?;
  let out_shape = full_shape_i32(logits_q)?;

  let cfg = MetalKernelApplyConfig::new(
    [1024, cot_size as u32, 1],
    [1024, 1, 1],
    vec![out_shape],
    vec![dtype],
  )?
  .with_template(template_for(dtype, vocab));

  with_kernel(&JS_BACKWARD, build_js_backward_kernel, |kernel| {
    let mut outputs = kernel.apply(&[logits_q, logits_p, cotan, kl_q], &cfg)?;
    Ok(outputs.swap_remove(0))
  })
}

// ───────────────────────── Public API ─────────────────────────

/// Kullback-Leibler divergence loss `D_KL(p || q)` between two logit tensors,
/// with a hand-written Metal kernel + custom VJP for numerical stability at
/// long-sequence training scales.
///
/// `logits_q` and `logits_p` must have identical shape and dtype. The
/// returned array has shape `logits_q.shape[:-1]` (the last "vocab" axis is
/// reduced) and the same dtype.
///
/// # Contract
///
/// - **Rank >= 2.** Logits must be shaped `[B, ..., V]` (matching upstream
///   `mlx_lm`'s convention of logits emitted from a model forward pass).
///   Rank-1 input `[V]` is rejected with [`Error::Backend`] — scalar Metal
///   outputs are not supported by the shared kernel wrapper. Reshape to
///   `[1, V]` for a scalar-like loss.
/// - **Floating dtype only.** Logits must be `Dtype::F32`, `Dtype::F16`, or
///   `Dtype::BF16`. The kernel uses the input dtype as the template
///   parameter `T` and casts intermediate KL expressions back to `T`, so
///   integer / boolean / `F64` / `Complex64` inputs would silently truncate
///   to corrupted numerics and are rejected with [`Error::Backend`]. Cast
///   with [`Array::astype`] before calling if your logits aren't already
///   floating.
///
/// Differentiable via the custom backward kernel — wrap in
/// [`crate::transforms::grad`] / [`crate::transforms::value_and_grad`] to
/// compute gradients w.r.t. `logits_q` (the gradient w.r.t. `logits_p` is
/// always zero, matching the python reference).
///
/// Mirrors `mlx_lm.tuner.losses.kl_div_loss` (the `can_run_metal()` branch).
/// This Rust surface does NOT include the python fallback (the
/// `nn.losses.kl_div_loss` softmax path used when Metal is unavailable)
/// because mlxrs is macOS-Metal-first; on a Metal-less host the apply call
/// errors clearly rather than silently routing through a slower path.
///
/// # Errors
///
/// Validation precedence (in order):
///
/// 1. Rank check (BOTH inputs) — rank `< 2` (including rank 0) returns
///    [`Error::Backend`] with the "rank >= 2 required" message. This
///    runs BEFORE shape comparison, so a rank-1 `logits_q` paired with a
///    rank-2 `logits_p` returns the rank error and NOT [`Error::ShapePairMismatch`].
/// 2. Shape comparison — [`Error::ShapePairMismatch`] if
///    `logits_q.shape() != logits_p.shape()`; [`Error::OutOfRange`] if the
///    last dimension is 0 or any dimension overflows `i32`.
/// 3. Dtype comparison — [`Error::DtypeMismatch`] if the two arrays have
///    different dtypes.
/// 4. Dtype admissibility — [`Error::Backend`] if the dtype is not one of
///    `F32` / `F16` / `BF16`.
///
/// Also returns [`Error::Backend`] if the Metal kernel apply fails (e.g.
/// no Metal device available, kernel compile error).
///
/// [`Array::astype`]: crate::array::Array::astype
pub fn kl_div_loss(logits_q: &Array, logits_p: &Array) -> Result<Array> {
  validate_inputs(
    logits_q,
    logits_p,
    "kl_div_loss: logits_q rank (must be >= 2; reshape rank-1 [V] to [1, V])",
    "kl_div_loss: logits_p rank (must be >= 2; reshape rank-1 [V] to [1, V])",
    "kl_div_loss: logits_q vs logits_p shape",
    "kl_div_loss: logits last dimension",
    "kl_div_loss: shape dim",
    "kl_div_loss: logits dtype (cast with .astype(Dtype::F32) before calling)",
  )?;

  // Build the custom_vjp closure once per call (the cost is a few FFI calls;
  // the kernel itself is cached by the thread-local OnceCell). The forward
  // closure captures nothing external; the VJP captures nothing external —
  // both reach into the same thread-local kernel cache via the apply helpers.
  let wrapped = custom_vjp(
    |inputs: &[Array]| -> Result<Vec<Array>> {
      let out = kl_forward_apply(&inputs[0], &inputs[1])?;
      Ok(vec![out])
    },
    // MLX core invokes the custom VJP callback with positional order
    // `(primals, cotangents, outputs)` — see `CustomTransforms::vjp` in
    // `mlx/primitives.cpp` upstream (`vjp_fun_(inputs, cotangents,
    // outputs)`). The trampoline in `transforms::closure::trampoline_custom`
    // preserves that order, so the closure binds `cotangents` second and
    // `outputs` third.
    |primals: &[Array], cotangents: &[Array], _outputs: &[Array]| -> Result<Vec<Array>> {
      let logits_q = &primals[0];
      let logits_p = &primals[1];
      let cotangent = &cotangents[0];
      let dq = kl_backward_apply(logits_q, logits_p, cotangent)?;
      // Gradient w.r.t. `logits_p` is zero (python ref uses `mx.zeros_like`).
      let dp = crate::ops::misc::zeros_like(logits_p)?;
      Ok(vec![dq, dp])
    },
  )?;

  let inputs = [logits_q.try_clone()?, logits_p.try_clone()?];
  let mut outputs = wrapped(&inputs)?;
  if outputs.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "kl_div_loss: forward closure output",
    )));
  }
  Ok(outputs.swap_remove(0))
}

/// Jensen-Shannon divergence loss
/// `0.5 * (D_KL(p || m) + D_KL(q || m))` where `m = (p + q) / 2`, between
/// two logit tensors, with a hand-written Metal kernel + custom VJP for
/// numerical stability.
///
/// `logits_q` and `logits_p` must have identical shape and dtype. The
/// returned array has shape `logits_q.shape[:-1]` and the same dtype.
///
/// # Contract
///
/// Same as [`kl_div_loss`]: rank must be `>= 2` (logits must be shaped
/// `[B, ..., V]`; rank-1 `[V]` is rejected — reshape to `[1, V]`) and the
/// dtype must be one of `F32`, `F16`, `BF16` (integer / boolean / `F64` /
/// `Complex64` are rejected to avoid silent numerical truncation; cast
/// with [`Array::astype`] before calling).
///
/// Differentiable via the custom backward kernel — wrap in
/// [`crate::transforms::grad`] / [`crate::transforms::value_and_grad`] to
/// compute gradients w.r.t. `logits_q` (the gradient w.r.t. `logits_p` is
/// always zero, matching the python reference).
///
/// Mirrors `mlx_lm.tuner.losses.js_div_loss` (the `can_run_metal()` branch).
/// The non-Metal fallback is omitted for the same reason as
/// [`kl_div_loss`].
///
/// # Errors
///
/// Same as [`kl_div_loss`].
///
/// [`Array::astype`]: crate::array::Array::astype
pub fn js_div_loss(logits_q: &Array, logits_p: &Array) -> Result<Array> {
  validate_inputs(
    logits_q,
    logits_p,
    "js_div_loss: logits_q rank (must be >= 2; reshape rank-1 [V] to [1, V])",
    "js_div_loss: logits_p rank (must be >= 2; reshape rank-1 [V] to [1, V])",
    "js_div_loss: logits_q vs logits_p shape",
    "js_div_loss: logits last dimension",
    "js_div_loss: shape dim",
    "js_div_loss: logits dtype (cast with .astype(Dtype::F32) before calling)",
  )?;

  let wrapped = custom_vjp(
    |inputs: &[Array]| -> Result<Vec<Array>> {
      let (loss, kl_q) = js_forward_apply(&inputs[0], &inputs[1])?;
      // The python forward returns `(loss, mx.stop_gradient(kl_q))`. In
      // mlxrs the custom_vjp overrides autograd entirely — the user-supplied
      // backward is the ONLY gradient path — so an explicit stop_gradient on
      // `kl_q` is unnecessary; gradients never flow through `kl_q`.
      Ok(vec![loss, kl_q])
    },
    // MLX core's `CustomTransforms::vjp` invokes its callback with positional
    // order `(primals, cotangents, outputs)`; see the matching note in
    // `kl_div_loss`.
    |primals: &[Array], cotangents: &[Array], outputs: &[Array]| -> Result<Vec<Array>> {
      let logits_q = &primals[0];
      let logits_p = &primals[1];
      // cotangents[0] is for `loss`; cotangents[1] is for `kl_q` and is
      // ignored (python `cotan, _ = cotangents`).
      let cotan = &cotangents[0];
      let kl_q = &outputs[1];
      let dq = js_backward_apply(logits_q, logits_p, cotan, kl_q)?;
      let dp = crate::ops::misc::zeros_like(logits_p)?;
      Ok(vec![dq, dp])
    },
  )?;

  let inputs = [logits_q.try_clone()?, logits_p.try_clone()?];
  let mut outputs = wrapped(&inputs)?;
  if outputs.is_empty() {
    return Err(Error::EmptyInput(EmptyInputPayload::new(
      "js_div_loss: forward closure output",
    )));
  }
  // The forward emits (loss, kl_q); the python public surface returns
  // `_js_div_loss(...)[0]`, the loss. We mirror that.
  Ok(outputs.swap_remove(0))
}

// ───────────────────────── Unit tests (no Metal device required) ─────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  // MSL source identity: ensures the Rust kernel-source emitters return
  // strings that match the python reference. The four sources are pinned
  // by length + characteristic-substring assertions; the FULL byte-for-byte
  // contents live in the python ref, and a manual `diff` between the
  // python `source = """..."""` body and the `r#"..."#` body in this
  // module is the source of truth. Re-flowing or re-formatting the MSL
  // changes the emitted Metal IR and breaks numerical parity, so these
  // pinning tests catch accidental formatter-driven edits.

  #[test]
  fn kl_forward_msl_source_contains_signature_landmarks() {
    let s = kl_forward_msl_source();
    // Constants / shared-memory layout from the python ref.
    assert!(s.contains("constexpr int M = 4;"));
    assert!(s.contains("constexpr int block = 1024 * M;"));
    assert!(s.contains("threadgroup float shared[32 * 2];"));
    // Buffer offsets — KL forward shifts `out` by `out_idx` (scalar per row).
    assert!(s.contains("logits_q += out_idx * V;"));
    assert!(s.contains("logits_p += out_idx * V;"));
    assert!(s.contains("out += out_idx;"));
    // The final write — `out[0] = static_cast<T>(kl);`.
    assert!(s.contains("out[0] = static_cast<T>(kl);"));
    // Distinctive: KL forward uses `lse_q_minus_p`, NOT `lse_q`.
    assert!(s.contains("lse_q_minus_p"));
    assert!(!s.contains("kl_p +="));
  }

  #[test]
  fn kl_backward_msl_source_contains_signature_landmarks() {
    let s = kl_backward_msl_source();
    assert!(s.contains("constexpr int M = 4;"));
    // Backward shifts `out` by `out_idx * V` (full-shape gradient).
    assert!(s.contains("out += out_idx * V;"));
    assert!(s.contains("cotan += out_idx;"));
    // The gradient write — `c * (exp(q-lse_q) - exp(p-lse_p))`.
    assert!(
      s.contains("c * (metal::fast::exp(vals_q[j] - lse_q) - metal::fast::exp(vals_p[j] - lse_p))")
    );
    // No `kl_q` accumulation here (that's only in the JS backward).
    assert!(!s.contains("output_kl_q"));
  }

  #[test]
  fn js_forward_msl_source_contains_signature_landmarks() {
    let s = js_forward_msl_source();
    assert!(s.contains("constexpr int M = 4;"));
    // JS forward has TWO outputs.
    assert!(s.contains("out += out_idx;"));
    assert!(s.contains("out_kl_q += out_idx;"));
    // The pinnable JS-distinct expression — kl_p+kl_q accumulation with
    // log(2) - log(1 + exp(...)) structure.
    assert!(s.contains("logtwo - metal::fast::log(1 + metal::fast::exp(logq_j - logp_j))"));
    assert!(s.contains("logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j))"));
    // Final writes: loss = 0.5*(kl_p+kl_q); kl_q kept verbatim.
    assert!(s.contains("out[0] = static_cast<T>(0.5 * kl_p + 0.5 * kl_q);"));
    assert!(s.contains("out_kl_q[0] = static_cast<T>(kl_q);"));
  }

  #[test]
  fn js_backward_msl_source_contains_signature_landmarks() {
    let s = js_backward_msl_source();
    assert!(s.contains("constexpr int M = 4;"));
    // JS backward reads `output_kl_q` (the per-row cached value).
    assert!(s.contains("output_kl_q += out_idx;"));
    assert!(s.contains("float kl_q = output_kl_q[0];"));
    // The gradient expression — c * 0.5 * q_j * (logtwo - log(1+exp(...)) - kl_q).
    assert!(s.contains(
      "c * 0.5 * q_j * (logtwo - metal::fast::log(1 + metal::fast::exp(logp_j - logq_j)) - kl_q)"
    ));
  }

  // Shape / dtype validation — runs without a Metal device because all
  // validation is in the wrapper layer, before any FFI allocation.

  #[test]
  fn validate_inputs_rejects_shape_mismatch() {
    let a = Array::ones::<f32>(&[2, 4]).unwrap();
    let b = Array::ones::<f32>(&[2, 8]).unwrap();
    let err = validate_inputs(
      &a,
      &b,
      "kl_div_loss: logits_q rank",
      "kl_div_loss: logits_p rank",
      "kl_div_loss: logits_q vs logits_p shape",
      "kl_div_loss: logits last dimension",
      "kl_div_loss: shape dim",
      "kl_div_loss: logits dtype",
    )
    .unwrap_err();
    match err {
      Error::ShapePairMismatch(p) => {
        assert_eq!(p.expected(), &[2, 4]);
        assert_eq!(p.actual(), &[2, 8]);
      }
      other => panic!("expected ShapePairMismatch, got: {other:?}"),
    }
  }

  #[test]
  fn validate_inputs_rejects_dtype_mismatch() {
    let a = Array::ones::<f32>(&[2, 4]).unwrap();
    let b = Array::ones::<half::f16>(&[2, 4]).unwrap();
    let err = validate_inputs(
      &a,
      &b,
      "kl_div_loss: logits_q rank",
      "kl_div_loss: logits_p rank",
      "kl_div_loss: logits_q vs logits_p shape",
      "kl_div_loss: logits last dimension",
      "kl_div_loss: shape dim",
      "kl_div_loss: logits dtype",
    )
    .unwrap_err();
    match err {
      Error::DtypeMismatch(p) => {
        assert_eq!(p.expected(), Dtype::F32);
        assert_eq!(p.got(), Dtype::F16);
      }
      other => panic!("expected DtypeMismatch, got: {other:?}"),
    }
  }

  #[test]
  fn kl_div_loss_rejects_shape_mismatch() {
    let a = Array::ones::<f32>(&[2, 4]).unwrap();
    let b = Array::ones::<f32>(&[2, 8]).unwrap();
    let err = kl_div_loss(&a, &b).unwrap_err();
    match err {
      Error::ShapePairMismatch(p) => {
        assert!(p.context().contains("kl_div_loss"), "got: {p:?}");
      }
      other => panic!("expected ShapePairMismatch, got: {other:?}"),
    }
  }

  #[test]
  fn js_div_loss_rejects_shape_mismatch() {
    let a = Array::ones::<f32>(&[2, 4]).unwrap();
    let b = Array::ones::<f32>(&[2, 8]).unwrap();
    let err = js_div_loss(&a, &b).unwrap_err();
    match err {
      Error::ShapePairMismatch(p) => {
        assert!(p.context().contains("js_div_loss"), "got: {p:?}");
      }
      other => panic!("expected ShapePairMismatch, got: {other:?}"),
    }
  }

  #[test]
  fn n_outs_of_computes_total_over_last_dim() {
    let a = Array::ones::<f32>(&[3, 5, 7]).unwrap();
    // 3 * 5 * 7 / 7 = 15
    assert_eq!(n_outs_of(&a).unwrap(), 15);
  }

  #[test]
  fn n_outs_of_rejects_rank_0() {
    let a = Array::full::<f32>(&[0i32; 0], 1.0).unwrap();
    let err = n_outs_of(&a).unwrap_err();
    match err {
      Error::RankMismatch(p) => {
        assert!(p.context().contains("rank"));
        assert_eq!(p.actual(), 0);
      }
      other => panic!("expected RankMismatch, got: {other:?}"),
    }
  }

  #[test]
  fn vocab_of_returns_last_dim_as_i32() {
    let a = Array::ones::<f32>(&[2, 4, 128]).unwrap();
    assert_eq!(vocab_of(&a).unwrap(), 128);
  }

  #[test]
  fn leading_shape_strips_last_axis() {
    let a = Array::ones::<f32>(&[3, 5, 7]).unwrap();
    assert_eq!(leading_shape_i32(&a).unwrap(), vec![3i32, 5]);
  }

  #[test]
  fn full_shape_preserves_all_dims() {
    let a = Array::ones::<f32>(&[3, 5, 7]).unwrap();
    assert_eq!(full_shape_i32(&a).unwrap(), vec![3i32, 5, 7]);
  }

  #[test]
  fn template_for_emits_t_and_v_in_canonical_order() {
    let t = template_for(Dtype::F32, 128);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].0, "T");
    assert_eq!(t[0].1, KernelTemplateArg::Dtype(Dtype::F32));
    assert_eq!(t[1].0, "V");
    assert_eq!(t[1].1, KernelTemplateArg::Int(128));
  }
}
