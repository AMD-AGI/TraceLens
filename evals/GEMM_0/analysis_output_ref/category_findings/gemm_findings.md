# GEMM Analysis Summary

**Status:** SUCCESS

## Overview

GEMMs account for 93.75% of GPU compute time (0.135 ms GPU kernel time out of 0.144 ms total). A single `aten::mm` operation dominates the category. Average efficiency is 23.51% of peak memory bandwidth, indicating significant optimization opportunity.

**Time Breakdown (from manifest):**
- GPU kernel time: 0.135 ms
- CPU duration: 0.207 ms (CPU/GPU ratio: 1.53x — no sync bottleneck)
- Sync time: 0 ms

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::mm  | 10    | 0.135     | 100.0%        | 23.51%     | 16.0       | memory-bound |

## Key Bottlenecks

### 1. aten::mm — Tall-Skinny GEMM with Low Memory Bandwidth Utilization

- **Time:** 0.135 ms (100% of GEMM compute)
- **Shape:** M=131072, N=32, K=32 (FP16 inputs)
- **Invocations:** 10
- **Efficiency:** 23.51% of peak HBM bandwidth (1.25 TB/s achieved vs 5.3 TB/s peak)
- **Compute throughput:** 19.93 TFLOPS/s achieved vs 654 TFLOPS/s peak FP16 (3.0% — expected for memory-bound)
- **Bound type:** Memory-bound (FLOPS/Byte = 16.0)
- **Kernel:** `Cijk_Ailk_Bljk_HHS_BH_MT64x128x32_MI16x16x16x1_SN_...` (vendor GEMM library — Tensile)

**Issue:** This is a highly skewed tall-skinny matrix multiply (131072x32 × 32x32). With only 16 FLOPS/Byte, the operation is firmly memory-bound, yet achieves only 23.51% of peak HBM bandwidth. The tile configuration (MT64x128x32) may be suboptimal for this extreme aspect ratio where M >> N, K.

**Algorithmic:**
- Consider fusing this operation with adjacent element-wise or reduction operations to reduce memory round-trips
- If this GEMM is part of a linear layer with small hidden dim (32), evaluate whether `torch.compile` can fuse surrounding ops
- With N=K=32, check if a custom fused kernel (e.g., for projection layers) would be more efficient than a general GEMM call

**Kernel:**
- Generate replay artifact for kernel team — the 64x128 tile size is likely oversized for N=32 output columns, leading to wasted work and poor wave occupancy
- A tile configuration better matched to the skinny output (e.g., tall-skinny specific tiles) could improve bandwidth utilization
- Profile with hardware counters to diagnose whether the bottleneck is L2 cache thrashing, TLB pressure, or suboptimal memory access patterns

## Additional Notes

- Missing perf models: 0
- Quantized GEMMs detected: 0
- All operations use FP16 precision (`c10::Half`)
- No parent chain information available in tree data — unable to determine calling context (attention, MLP, etc.)

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune kernel tile size for tall-skinny shape (M=131072, N=32, K=32) | kernel_tuning | 0.103 | medium |
| Fuse with adjacent ops to reduce memory traffic | algorithmic | 0.030 | low |
