# conv_03_asymmetric_kernel — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_03_asymmetric_kernel` on MI300X. The trace contains a single `aten::convolution` with input [2,256,56,56], weight [512,256,1,7] (asymmetric 1×7 filter), stride=(1,1), padding=(0,3). The operation achieves 8.83% of peak FP32 (14.39 TFLOPS vs 163 TFLOPS). The non-square 1×7 kernel may cause suboptimal igemm tile utilization.

| Metric | Value |
|--------|-------|
| Total Compute Time | 1.0 ms |
| Computation | 80.0% |
| Idle Time | 20.0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (0.8 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution [2,256,56,56]×[512,256,1,7] | Convolution | 0.800 | 100.0% |

### 🔴 P1: Asymmetric 1×7 Kernel — Suboptimal igemm Tile for Non-Square Filter

**Issue**: Asymmetric 1×7 kernel shape causes the igemm tile (`igemm_fwd_gtcx_nhwc_fp32_bx0_bt128x128x8`, grid=[64,1,1], block=[256,1,1]) to be suboptimal for non-square filters. Achieves 8.83% of peak FP32 (14.39 TFLOPS vs 163 TFLOPS).

**Action**: Generate replay artifact for kernel team to evaluate specialized asymmetric-filter paths. Consider splitting 1×7 into 1×1 + 1×6 or using im2col with tailored GEMM shapes. Profile memory access patterns for the elongated filter dimension.

**Impact**: Estimated 0.73 ms savings (91.2% of kernel time) if efficiency approaches peak — medium confidence given the non-standard filter shape.

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 80.0% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `aten::convolution` with asymmetric 1×7 filter. Input [2,256,56,56], weight [512,256,1,7], FP32. Stride=(1,1), padding=(0,3), dilation=(1,1), groups=1. Achieves 8.83% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 0.8 ms
- CPU duration: 1.0 ms (CPU/GPU ratio: 1.25x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution | 1 | 0.800 | 100.0% | 8.83% | 501.76 | compute |

**Key Bottleneck: aten::convolution — Asymmetric Filter Shape**

- **Time:** 0.8 ms (100% of compute)
- **Shape:** Input [2,256,56,56], Weight [512,256,1,7], FP32
- **Params:** Stride=(1,1), Padding=(0,3), Dilation=(1,1), Groups=1
- **Kernel:** `igemm_fwd_gtcx_nhwc_fp32_bx0_bt128x128x8`, grid=[64,1,1], block=[256,1,1]
- **Efficiency:** 8.83% of peak FP32 (14.39 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.03 TB/s achieved vs 5.3 TB/s peak (0.6% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 501.76)

**Issue:** Square igemm tiles (128×128×8) are designed for symmetric or near-square filter dimensions. The 1×7 filter creates an elongated im2col/GEMM shape that may underutilize tile dimensions or cause irregular memory access.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune igemm tile for asymmetric 1×7 filter | kernel_tuning | 0.73 | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 1.0 ms |
| Computation | 80.0% |
| Idle Time | 20.0% |
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
