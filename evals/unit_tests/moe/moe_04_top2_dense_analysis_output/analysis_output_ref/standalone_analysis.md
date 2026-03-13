# moe_04_top2_dense — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of synthetic test trace `moe_04_top2_dense` on MI300X. The trace contains 2 compute kernel categories and 0 system-level categories.

| Metric | Value |
|--------|-------|
| Total Compute Time | 2.79 ms |
| Computation | 67.7% |
| Idle Time | 32.3% |
| Exposed Communication | 0.0000% |
| Top Bottleneck Category | MoE Fused (1.800 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Category | GPU Time (ms) | % of Compute |
|------|----------|---------------|--------------|
| 1 | MoE Fused | 1.800 | 95.2% |
| 2 | elementwise | 0.090 | 4.8% |

### 🔴 P1: MoE Fused Optimization

**Insight**: MoE Fused operations at 0.0% average efficiency, consuming 1.800 ms (64.5% of compute).

**Action**: Review kernel configurations and consider algorithmic optimizations.

**Impact**: Up to 0.000 ms savings through kernel tuning.

→ *See Detailed Analysis: MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 67.7% computation, with negligible memcpy and communication overhead.

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (95.2% of compute)

# MoE Analysis Findings

**Status**: SUCCESS
**Analysis Tier**: Compute Kernel
**Total Time**: 1.800 ms (64.5% of compute)
**Operation Count**: 1
**Average Efficiency**: 0.0%

## Operations Summary

| Operation | Count | Time (ms) | % of Category | Efficiency (%) | Bound |
|-----------|-------|-----------|---------------|----------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 1.800 | 100.0% | N/A | N/A |

## Operation Classification

- **vllm::rocm_aiter_fused_moe**: Fused MoE (end-to-end kernel)

## Bottleneck Analysis

### vllm::rocm_aiter_fused_moe
- **Time**: 1.800 ms (100.0% of category)
- **Efficiency**: N/A (no perf model — MoE ops lack direct perf model mapping)
- **Assessment**: MoE operations are typically already fused; focus on expert routing balance

## Recommendations

### Note: No Efficiency Metrics Available
- MoE operations in this trace lack a direct performance model mapping
- Focus analysis on expert routing balance and token distribution
- Consider profiling expert utilization at the application level

### Algorithmic: Check Expert Routing Balance
- Verify token distribution across experts is balanced
- Adjust capacity factor if experts are underutilized
- Consider auxiliary load-balancing loss

## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Expert load rebalancing | algorithmic | N/A | low |


---

### 2. elementwise (4.8% of compute)

# Elementwise Analysis Findings

**Status**: SUCCESS
**Total Time**: 0.090 ms (3.2% of compute)
**Average Efficiency**: 2.0%

Elementwise operations are minor (3.2% of compute). No significant optimization opportunity.

## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|


---

## Detailed Analysis: System-Level

## Appendix

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS

*Generated: 2026-02-25 13:39:16*
