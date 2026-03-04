# conv_01_large_spatial_small_ch — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_01_large_spatial_small_ch` on MI300X. The trace contains a single `aten::convolution` with input [1,3,2048,2048], weight [64,3,7,7], stride=(2,2), padding=(3,3). The operation achieves only 2.69% of peak FP32 (4.38 TFLOPS vs 163 TFLOPS) due to poor utilization of igemm tiles for this large-spatial, small-channel workload.

| Metric | Value |
|--------|-------|
| Total Compute Time | 4.7 ms |
| Computation | 95.7% |
| Idle Time | 4.3% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (4.5 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution [1,3,2048,2048]×[64,3,7,7] | Convolution | 4.500 | 100.0% |

### 🔴 P1: Large-Spatial Small-Channel Convolution — Poor igemm Tile Utilization

**Issue**: Very large spatial dimensions (2048×2048) with tiny input channels (3) cause poor utilization of the igemm tiles. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_ex0_bt128x128x32` (grid=[1024,1,1], block=[256,1,1]) achieves only 2.69% of peak FP32 (4.38 TFLOPS vs 163 TFLOPS).

**Action**: Investigate alternative tile configurations or specialized kernels for large-spatial/small-channel workloads. Consider im2col+GEMM fusion strategies that better amortize weight loads across spatial positions.

**Impact**: Estimated 4.38 ms savings (97.3% of kernel time) if efficiency approaches peak — medium confidence given the extreme shape mismatch.

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 95.7% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `aten::convolution` dominates the trace. Input [1,3,2048,2048], weight [64,3,7,7], FP32. Stride=(2,2), padding=(3,3), dilation=(1,1), groups=1. Achieves 2.69% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 4.5 ms
- CPU duration: 4.7 ms (CPU/GPU ratio: 1.04x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution | 1 | 4.500 | 100.0% | 2.69% | 61.89 | compute |

**Key Bottleneck: aten::convolution — Large Spatial, Small Channels**

- **Time:** 4.5 ms (100% of compute)
- **Shape:** Input [1,3,2048,2048], Weight [64,3,7,7], FP32
- **Params:** Stride=(2,2), Padding=(3,3), Dilation=(1,1), Groups=1
- **Kernel:** `igemm_fwd_gtcx_nhwc_fp32_bx0_ex0_bt128x128x32`, grid=[1024,1,1], block=[256,1,1]
- **Efficiency:** 2.69% of peak FP32 (4.38 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.07 TB/s achieved vs 5.3 TB/s peak (1.3% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 61.89)

**Issue:** The 128×128×32 tile configuration is poorly matched to the 3 input channels. Each spatial position has minimal channel dimension to amortize tile overhead, leading to severe underutilization.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune igemm tile for large-spatial small-channel shape | kernel_tuning | 4.379 | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 4.7 ms |
| Computation | 95.7% |
| Idle Time | 4.3% |
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
