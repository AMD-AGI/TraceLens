# GEMM Analysis Summary

## Overview

**Status:** SUCCESS

GEMMs account for 100% of compute time (10.89 ms GPU kernel time). Average efficiency: 4.46%. One unique GEMM signature with 10 invocations.

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::mm | 10 | 10.89 | 100.0 | 4.46% | 170.44 | compute-bound |

## Key Bottlenecks

### 1. aten::mm (M=256, N=512, K=131072)

- **Time:** 10.89 ms (100% of compute)
- **Efficiency:** 4.46% of peak MAF (31.55 TFLOPS/s achieved vs 708 TFLOPS/s peak matrix_bf16)
- **Issue:** Compute-bound GEMM achieving only 4.46% of peak. Shape (M=256, N=512, K=131072) yields few tiles on GPU—large K dimension with small M/N limits parallelism and wave occupancy. 10 separate invocations suggest batching opportunity.
- **Algorithmic:** Batch the 10 GEMM invocations together using `torch.bmm` or grouped operations if semantically valid; check if `torch.compile` can batch these automatically. Consider whether smaller batch sizes or different batching strategies could improve GPU utilization.
- **Kernel:** Generate replay artifact for kernel team to tune tile sizes. Flag suboptimal GEMM kernel selection for this shape; investigate wave occupancy and tile configuration for compute-bound GEMMs with tall-skinny K dimension.

## Additional Notes

- Missing perf models: 0
- Quantized GEMMs detected: 0

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Kernel tuning for aten::mm (M=256, N=512, K=131072) | kernel_tuning | 10.404 | high |
