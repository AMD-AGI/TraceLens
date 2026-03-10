# MoE Fused Operations Analysis Findings

**Status**: SUCCESS  
**Platform**: MI300X (Peak HBM BW: 5.3 TB/s, Resolved Peak MAF: 708 TFLOPS)

## Summary

MoE (Mixture of Experts) operations were identified in the trace. The category consists of a single fused MoE kernel with total GPU time of 0.95 ms.

## Operations Overview

| Operation | Type | Count | Time (ms) | % of Category | Efficiency | Peak (TFLOPS) |
|-----------|------|-------|-----------|---------------|------------|---------------|
| vllm::rocm_aiter_fused_moe | Fused | 1 | 0.95 | 100.0 | N/A | 708 |

**Operation classification:**
- **Fused**: `vllm::rocm_aiter_fused_moe` — End-to-end fused MoE kernel combining routing and expert compute in a single GPU kernel.

## Bottleneck Assessment

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| Time | > 100 ms or > 5% of category | 0.95 ms — below threshold; no time-based bottleneck |
| Efficiency | < 70% of peak TFLOPS | Efficiency data unavailable (measurement not resolved) |

**Findings:**
- Total MoE category time is 0.95 ms — negligible relative to typical workload scales.
- Efficiency metrics (TFLOPS achieved, efficiency %) are not available for this operation; kernel efficiency cannot be assessed.
- No efficiency anomalies (> 100%) observed.

## Recommendations

### Algorithmic
- **Routing balance**: Validate token distribution across experts; skewed routing can underutilize capacity.
- **Capacity factors**: Review expert capacity settings if scaling to larger batch sizes or longer sequences.
- **Top-k selection**: For top-1 sparse routing, confirm that the routing policy matches intended load balance.

### Kernel Optimization
- MoE kernels in this trace are already fused; fusion opportunities are limited.
- Efficiency data was not resolved — consider generating a replay artifact if deeper kernel analysis is needed.
- No kernel_tuning impact estimates are available from the metrics; no quantified savings can be reported.

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|

*No actionable bottlenecks identified; impact_estimates is empty. No kernel_tuning recommendations with pre-computed savings.*
