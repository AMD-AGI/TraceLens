# GEMM Analysis Summary

## Overview

GEMM operations account for **100%** of compute time in this trace (1.071 ms total). The workload consists of a single operation type with **22.74%** average efficiency relative to peak MAF. This represents a significant optimization opportunity.

**Platform:** MI300X  
**Total trace time:** 1.071 ms  
**GPU utilization:** 100% computation, 0% idle

---

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::mm | 10 | 1.071 | 100.0 | 22.74% | 482.33 | compute-bound |

---

## Root Cause Analysis

### Non-Power-of-2 Dimensions

The primary cause of low efficiency is **non-power-of-2 M and N dimensions** (M=1025, N=1025). K=8192 is power-of-2 and large enough for good utilization.

**Impact on tile-based GEMM kernels:**

1. **Padding waste:** The kernel uses tile config `MT128x64x64` (128×64 output tiles). For M=1025, N=1025:
   - M dimension: ceil(1025/128) = 9 tiles → 9×128 = 1152 elements (127 wasted per tile row)
   - N dimension: ceil(1025/64) = 17 tiles → 17×64 = 1088 elements (63 wasted per tile column)
   - Significant compute cycles are spent on padded regions that produce discarded results.

2. **Uneven work distribution:** 9×17 = 153 total tiles across 304 CUs on MI300X. The tile count does not evenly divide across CUs, leading to load imbalance and underutilization.

3. **Compute-bound but underperforming:** FLOPS/Byte of 482.33 confirms compute-bound behavior. At 161 TFLOPS achieved vs 708 TFLOPS peak (matrix_bf16), the kernel is running at ~23% of theoretical peak—consistent with tile padding and occupancy inefficiencies from non-pow2 dimensions.

---

## Key Bottlenecks

### 1. aten::mm (M=1025, N=1025, K=8192)

- **Time:** 1.071 ms (100% of compute)
- **Efficiency:** 22.74% of peak MAF (161.0 TFLOPS achieved vs 708 TFLOPS peak matrix_bf16)
- **Issue:** Non-power-of-2 M and N dimensions cause tile padding waste and uneven work distribution across CUs.
- **Algorithmic:** Consider padding inputs to power-of-2 (e.g., 1024×1024) at the model/framework level when acceptable for accuracy. Alternatively, batch multiple small GEMMs together to amortize overhead.
- **Kernel:** Generate replay artifact for kernel team to evaluate alternative tile sizes (e.g., MT64x64) or specialized kernels for non-pow2 dimensions. Profile with hardware counters to diagnose wave occupancy and tile utilization.

---

## Recommendations

### Algorithmic Recommendations

| Recommendation | Description | Priority |
|----------------|-------------|----------|
| Pad to power-of-2 | Pad M and N to 1024 when accuracy permits; reduces tile waste and improves utilization | High |
| Batch GEMMs | If this pattern repeats (10 invocations), explore batching to improve parallelism | Medium |
| Alternative layouts | Evaluate if dimensions can be restructured (e.g., transpose) to align with kernel-friendly shapes | Medium |

### Kernel Optimization Focus

| Recommendation | Description | Priority |
|----------------|-------------|----------|
| Tile size tuning | Test alternative tile configs (MT64x64, MT64x128) for non-pow2 dimensions | High |
| Specialized kernels | Investigate kernels optimized for non-pow2 M/N (e.g., 1025, 2049) | Medium |
| Replay artifact | Generate reproducer for kernel team to profile occupancy and memory access patterns | High |

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|----------------|------|------------------------|------------|
| Kernel tuning for non-pow2 dimensions | kernel_tuning | 0.827 | medium |

**Note:** The impact estimate (0.827 ms) assumes kernel optimizations could improve efficiency from ~23% toward 70%+ of peak. Actual savings depend on kernel team findings from replay profiling.

---

## Additional Notes

- **Missing perf models:** 0
- **Quantized GEMMs detected:** 0
- **Kernel:** Cijk_Ailk_Bljk_BBS_BH_MT128x64x64_MI32x32x8x1 (rocBLAS Tensile)
- **Data type:** BFloat16

---

## Priority Ranking

1. **High:** Pad M/N to power-of-2 at model level (if feasible); generate replay artifact for kernel team.
2. **High:** Kernel tile size tuning for non-pow2 dimensions.
3. **Medium:** Investigate specialized kernels for common non-pow2 shapes (1025, 2049).
4. **Medium:** Explore batching the 10 invocations if they originate from a batchable pattern.
