# GEMM_0 - MI300X Standalone Analysis

## Executive Summary

This trace contains a single `aten::mm` operation (tall-skinny GEMM: 131072x32 x 32x32 in FP16) invoked 10 times. GPU utilization is high at 93.75% computation, with minimal idle time (6.25%) and no communication or memory transfer overhead. The GEMM achieves only 23.51% of peak memory bandwidth (1.25 TB/s vs 5.3 TB/s peak), indicating significant kernel tuning opportunity for this extreme aspect-ratio shape.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.144 ms |
| Computation | 93.75% |
| Idle Time | 6.25% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |
| Top Bottleneck Category | GEMM (100%) |

![Performance Improvement](perf_improvement.svg)

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis focused on individual kernel efficiency.

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::mm (131072x32 x 32x32, FP16) | GEMM | 0.135 | 100.0% |

### 🔴 P1: Tall-Skinny GEMM Kernel Under-Utilizing Memory Bandwidth

**Issue**: `aten::mm` (M=131072, N=32, K=32) is memory-bound (FLOPS/Byte=16) but achieves only 23.51% of peak HBM bandwidth (1.25 TB/s vs 5.3 TB/s). The vendor GEMM library tile configuration (MT64x128x32) is oversized for the narrow N=32 output dimension.

**Action**: Generate replay artifact for kernel team to investigate tile size selection for this extreme tall-skinny shape. A narrower tile matched to the 32-column output could improve bandwidth utilization significantly.

**Impact**: Estimated 0.103 ms savings (76.5% of GEMM kernel time) if bandwidth utilization reaches peak — medium confidence given the extreme aspect ratio.

→ *See [Detailed Analysis: Compute Kernels > GEMM](#1-gemm-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 93.75% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. GEMM (100% of compute)

**Status:** SUCCESS

**Overview:**
GEMMs account for 93.75% of GPU compute time (0.135 ms GPU kernel time out of 0.144 ms total). A single `aten::mm` operation dominates the category. Average efficiency is 23.51% of peak memory bandwidth.

**Time Breakdown:**
- GPU kernel time: 0.135 ms
- CPU duration: 0.207 ms (CPU/GPU ratio: 1.53x — no sync bottleneck)
- Sync time: 0 ms

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::mm  | 10    | 0.135     | 100.0%        | 23.51%     | 16.0       | memory-bound |

**Key Bottleneck: aten::mm — Tall-Skinny GEMM with Low Memory Bandwidth Utilization**

- **Time:** 0.135 ms (100% of GEMM compute)
- **Shape:** M=131072, N=32, K=32 (FP16 inputs)
- **Invocations:** 10
- **Efficiency:** 23.51% of peak HBM bandwidth (1.25 TB/s achieved vs 5.3 TB/s peak)
- **Compute throughput:** 19.93 TFLOPS/s achieved vs 654 TFLOPS/s peak FP16 (3.0% — expected for memory-bound)
- **Bound type:** Memory-bound (FLOPS/Byte = 16.0)
- **Kernel:** `Cijk_Ailk_Bljk_HHS_BH_MT64x128x32_MI16x16x16x1_SN_...` (vendor GEMM library)

**Issue:** This is a highly skewed tall-skinny matrix multiply (131072x32 x 32x32). With only 16 FLOPS/Byte, the operation is firmly memory-bound, yet achieves only 23.51% of peak HBM bandwidth. The tile configuration (MT64x128x32) may be suboptimal for this extreme aspect ratio where M >> N, K.

**Algorithmic Recommendations:**
- Consider fusing this operation with adjacent element-wise or reduction operations to reduce memory round-trips
- If this GEMM is part of a linear layer with small hidden dim (32), evaluate whether `torch.compile` can fuse surrounding ops
- With N=K=32, check if a custom fused kernel (e.g., for projection layers) would be more efficient than a general GEMM call

**Kernel Optimization Focus:**
- Generate replay artifact for kernel team — the 64x128 tile size is likely oversized for N=32 output columns, leading to wasted work and poor wave occupancy
- A tile configuration better matched to the skinny output (e.g., tall-skinny specific tiles) could improve bandwidth utilization
- Profile with hardware counters to diagnose whether the bottleneck is L2 cache thrashing, TLB pressure, or suboptimal memory access patterns

**Additional Notes:**
- Missing perf models: 0
- Quantized GEMMs detected: 0
- All operations use FP16 precision (`c10::Half`)

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune kernel tile size for tall-skinny shape (M=131072, N=32, K=32) | kernel_tuning | 0.103 | medium |
| Fuse with adjacent ops to reduce memory traffic | algorithmic | 0.030 | low |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 0.144 ms |
| Computation | 93.75% |
| Idle Time | 6.25% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |

CPU/Idle and Multi-Kernel analyses were not invoked as idle time (6.25%) is below the 15% threshold and no memcpy/NCCL events were detected in the trace.

---

## Appendix

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (FP16)**: 654 TFLOPS
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP32)**: 163 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
- **Memory**: 192 GB HBM3
