<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# MoE LLM - Comparative Analysis: MI300X vs H100

## Executive Summary

Comparative analysis of MoE (Mixture-of-Experts) inference workload on MI300X (Trace 1) vs H100 (Trace 2). Trace 2 completes the same logical MoE layer in 0.80 ms using a single fused kernel, while Trace 1 takes 1.26 ms with four separate kernels. Although each individual kernel on Trace 1 is faster than the single fused kernel on Trace 2 in direct time comparison, the multi-kernel decomposition on Trace 1 introduces idle gaps and launch overhead that inflate the total iteration time. The dominant optimization opportunity is kernel fusion — consolidating the four MoE kernels on Trace 1 into a single fused launch.

| Metric | Trace 1 - (MI300X) | Trace 2 - (H100) | Difference |
|--------|----------------------------|-------------------------------|------------|
| Total Time | 1.26 ms | 0.80 ms | -0.46 ms (-36.51%) |
| Compute % | 92.06% | 100.00% | +7.94% |
| Idle % | 7.94% | 0.00% | -7.94% |
| Exposed Communication % | 0.00% | 0.00% | 0.00% |
| Top Bottleneck Category | MoE Fused (92.06%) | MoE Fused (100.00%) | — |

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).
Summaries of recommendations from Step 7 sub-agents, focused on individual kernel efficiency.

### Top Operations

| Rank | Category | Trace 1 Time (ms) | Trace 2 Time (ms) | % of Compute Time | Ops | Difference (ms) | Potential improvement (time, E2E %) |
|------|----------|-------------------|-------------------|-------------------|-----|-----------------|-------------------------------------|
| 1 | MoE Fused | 1.16 | 0.80 | 100.00% | 4 | -0.36 | — |

✅ No compute kernel optimization opportunities identified. All categories are within target performance bounds.

---

## Kernel Fusion Opportunities (Experimental)

> **Note:** Kernel fusion analysis is experimental.

### 🟡 P1: MoE fused execution path (1.16 ms, 1 instances)

**Insight**: The `moe_layer` module launches four separate GPU kernels per instance on Trace 1 versus one fused custom GPU kernel on Trace 2, indicating that consolidating this mixture-of-experts pipeline into fewer launches is feasible.

**Action**: Prefer a fused mixture-of-experts implementation for this block (single launch covering routing, expert compute, and combine) where numerics and routing rules allow, mirroring the comparison trace rather than retaining the multi-kernel decomposition on Trace 1.

**Impact**: ~0.27–0.36 ms savings (21.43–28.57% of E2E)

**Confidence**: Medium — trace1 vs trace2 kernel counts and measured GPU-time gap are clear; kernel type metadata is coarse (`Unknown`) on this workload.

→ *See [Detailed Analysis: Kernel fusion insights > P1](#detailed-analysis-fusion-P1) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

> **Comparative note:** System-level analysis is performed on the primary trace (Trace 1) only. Cross-trace system-level comparison is not yet supported.

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 92.06% computation, with negligible memcpy and communication overhead.

---

## Detailed Analysis

### Compute Kernel Insights

No actionable compute kernel optimization opportunities were identified. The E2E gap is structural (4 launches vs 1 fused kernel), addressed in Kernel Fusion below.

### Kernel Fusion Insights

> **Note:** Kernel fusion analysis is experimental. Savings estimates use a roofline projection model (75-100% of peak) with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Actual savings depend on implementation feasibility and interaction effects.

<a id="detailed-analysis-fusion-P1"></a>
#### 🟡 P1: MoE fused execution path (1.16 ms, 1 instances)

**Identification:** Comparative fusion screening surfaced `moe_layer`: trace1 runs four distinct GPU kernels for one instance while trace2 runs one fused GPU kernel for the same module scope. (source: `fusion_candidates.json` → `module_name`, `kernel_count_trace1`, `kernel_count_trace2`, `kernels_trace1[]`, `kernels_trace2[]`)

**Data:**

**Trace1 kernels:**

| Kernel | Type | Duration (us) |
|--------|------|--------------|
| moe_sorting_kernel<BF16> | Unknown | 120.0 |
| ck_moe_stage1_kernel<BF16, 128, 128> | Unknown | 480.0 |
| ck_moe_stage2_kernel<BF16, 128, 128> | Unknown | 480.0 |
| moe_sum_kernel<BF16> | Unknown | 80.0 |

**Trace2 kernels:**

| Kernel | Type | Duration (us) |
|--------|------|--------------|
| triton__moe_fused_kernel_0d1d2d3d4d5d6d7d8d9d10d | Unknown | 800.0 |

**Impact estimate:**

- Low end (75% gap target): 0.270 ms savings (21.43% E2E)
- High end (100% gap target): 0.360 ms savings (28.57% E2E)
- Fusion pattern: memory-bound
- Confidence: Medium — kernel-count split and measured gap are clear; candidate typing data is coarse for some kernels

### System-Level Insights

No actionable system-level issues detected. GPU utilization is 92.06% compute with idle time at 7.94% (below the 15% threshold for investigation).

---

## Appendix

### Model Architecture
- **Model**: MoE LLM
- **Architecture**: Transformer
- **Scale**: Cannot be inferred from trace
- **Precision**: BF16

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
