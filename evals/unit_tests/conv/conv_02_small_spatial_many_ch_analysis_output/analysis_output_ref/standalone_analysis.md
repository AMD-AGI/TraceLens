# conv_02_small_spatial_many_ch — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_02_small_spatial_many_ch` on MI300X. The trace contains a single `aten::convolution` with input [4,2048,4,4], weight [2048,2048,3,3], stride=(1,1), padding=(1,1). The operation achieves only 2.47% of peak FP32 (4.03 TFLOPS vs 163 TFLOPS) due to extreme weight-dominated workload with tiny spatial dimensions (4×4) and poor spatial parallelism.

| Metric | Value |
|--------|-------|
| Total Compute Time | 1.4 ms |
| Computation | 85.7% |
| Idle Time | 14.3% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (1.2 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution [4,2048,4,4]×[2048,2048,3,3] | Convolution | 1.200 | 100.0% |

### 🔴 P1: Small-Spatial Many-Channel Convolution — Poor Spatial Parallelism

**Issue**: Tiny spatial dimensions (4×4) with huge channels (2048) create an extreme weight-dominated workload. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt64x64x16` (grid=[128,1,1], block=[128,1,1]) achieves only 2.47% of peak FP32 (4.03 TFLOPS vs 163 TFLOPS).

**Action**: Investigate batching across spatial positions or alternative GEMM decompositions. Consider fusing with adjacent layers to increase effective batch size. Profile for L2 cache thrashing from large weight tensor (2048×2048×3×3 ≈ 50M elements).

**Impact**: ~1.17 ms savings from closing efficiency gaps (pre-computed).

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 85.7% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `aten::convolution` dominates the trace. Input [4,2048,4,4], weight [2048,2048,3,3], FP32. Stride=(1,1), padding=(1,1), dilation=(1,1), groups=1. Achieves 2.47% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 1.2 ms
- CPU duration: 1.4 ms (CPU/GPU ratio: 1.17x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution | 1 | 1.200 | 100.0% | 2.47% | 31.78 | compute |

**Key Bottleneck: aten::convolution — Extreme Weight-Dominated Shape**

- **Time:** 1.2 ms (100% of compute)
- **Shape:** Input [4,2048,4,4], Weight [2048,2048,3,3], FP32
- **Params:** Stride=(1,1), Padding=(1,1), Dilation=(1,1), Groups=1
- **Kernel:** `igemm_fwd_gtcx_nhwc_fp32_bx0_bt64x64x16`, grid=[128,1,1], block=[128,1,1]
- **Efficiency:** 2.47% of peak FP32 (4.03 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.13 TB/s achieved vs 5.3 TB/s peak (2.5% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 31.78)

**Issue:** With only 4×4=16 spatial positions per batch, the 64×64×16 tile cannot effectively parallelize across space. The massive weight tensor (2048×2048×9) dominates memory traffic and may cause cache pressure.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune igemm tile for small-spatial many-channel shape | kernel_tuning | 1.17 | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 1.4 ms |
| Computation | 85.7% |
| Idle Time | 14.3% |
| Exposed Communication | 0.00% |

---

## Appendix

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (FP32)**: 163 TFLOPS
- **Peak MAF (FP16)**: 654 TFLOPS
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
- **Memory**: 192 GB HBM3
