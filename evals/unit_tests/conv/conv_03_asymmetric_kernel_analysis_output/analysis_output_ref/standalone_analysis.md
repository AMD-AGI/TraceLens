# TraceLens Standalone Analysis Report

**Trace:** `conv_03_asymmetric_kernel.json`
**Platform:** MI300X
**Total GPU Time:** 1.00 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of convolution trace `conv_03_asymmetric_kernel` on MI300X. The trace contains a single `aten::convolution` with input [2,256,56,56], weight [512,256,1,7] (asymmetric 1×7 filter), stride=(1,1), padding=(0,3). The operation achieves 8.83% of peak FP32 (14.39 TFLOPS vs 163 TFLOPS). The non-square 1×7 kernel may cause suboptimal igemm tile utilization.

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 1.0 ms |
| Computation | 80.0% |
| Communication | 0.0% |
| MemCpy | 0.0% |
| Idle | 20.0% |

GPU is 80.0% utilized for computation with 20.0% idle time.

---

## Prioritized Recommendations

### P1: Asymmetric 1×7 Kernel — Suboptimal igemm Tile for Non-Square Filter

| | |
|---|---|
| **Category** | Convolution |
| **Operation** | aten::convolution (Input [2,256,56,56], Weight [512,256,1,7], FP32) |
| **Current Efficiency** | 8.83% of peak MAF (14.39 TFLOPS vs 163 TFLOPS) |
| **Issue** | Asymmetric 1×7 kernel shape causes the igemm tile to be suboptimal for non-square filters. Square igemm tiles (128×128×8) are designed for symmetric or near-square filter dimensions. |
| **Recommendation** | Generate replay artifact for kernel team to evaluate specialized asymmetric-filter paths. Consider splitting 1×7 into 1×1 + 1×6 or using im2col with tailored GEMM shapes. Profile memory access patterns for the elongated filter dimension. |
| **Estimated Savings** | Up to 0.73 ms (medium confidence) |
| **Confidence** | Medium |

**Detail:** The aten::convolution (Input [2,256,56,56], Weight [512,256,1,7], FP32, stride=(1,1), padding=(0,3)) achieves 8.83% of peak FP32. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt128x128x8` (grid=[64,1,1], block=[256,1,1]) uses square tiles designed for symmetric or near-square filter dimensions. The 1×7 filter creates an elongated im2col/GEMM shape that may underutilize tile dimensions or cause irregular memory access. HBM BW: 0.03 TB/s achieved vs 5.3 TB/s peak (0.6% of peak). Compute-bound (FLOPS/Byte = 501.76).

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

No system-level bottlenecks detected. GPU activity breakdown shows 80.0% computation, with negligible memcpy and communication overhead.

### CPU/Idle Time Analysis

GPU idle time is 20.0% of total trace time (1.0 ms). CPU duration: 1.0 ms, GPU kernel time: 0.8 ms (CPU/GPU ratio: 1.25x — no sync bottleneck).

### Multi-Kernel Issues

No multi-kernel issues detected:
- **Memcpy events:** 0 (no D2H/H2D transfers)
- **Collective communication events:** 0
- **Exposed communication time:** 0.0%
- **Exposed memcpy time:** 0.0%

No memory transfer or communication overlap issues to report.

---

## Compute Kernel Analysis

### 1. Convolution (100% of compute)

**Status:** SUCCESS

Convolution accounts for 100% of compute time (0.8 ms GPU kernel time). Single `aten::convolution` with asymmetric 1×7 filter. Input [2,256,56,56], weight [512,256,1,7], FP32. Stride=(1,1), padding=(0,3), dilation=(1,1), groups=1. Achieves 8.83% of peak FP32.

#### Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::convolution | 1 | 0.800 | 100.0 | 8.83% | 501.76 | compute-bound |

#### Key Bottleneck: aten::convolution — Asymmetric Filter Shape

- **Time:** 0.8 ms (100% of compute)
- **Efficiency:** 8.83% of peak MAF (14.39 TFLOPS achieved vs 163 TFLOPS peak FP32)
- **Issue:** Compute-bound convolution achieving 8.83% of peak. Shape Input [2,256,56,56], Weight [512,256,1,7] (stride=(1,1), padding=(0,3), dilation=(1,1), groups=1). Square igemm tiles (128×128×8) are designed for symmetric or near-square filter dimensions. The 1×7 filter creates an elongated im2col/GEMM shape that may underutilize tile dimensions or cause irregular memory access. HBM BW: 0.03 TB/s achieved vs 5.3 TB/s peak (0.6% of peak).
- **Algorithmic:** Consider splitting 1×7 into 1×1 + 1×6 or using im2col with tailored GEMM shapes. Profile memory access patterns for the elongated filter dimension.
- **Kernel:** Generate replay artifact for kernel team to evaluate specialized asymmetric-filter paths. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt128x128x8` (grid=[64,1,1], block=[256,1,1]) — investigate tile configurations optimized for non-square filter dimensions.

#### Additional Notes

- Missing perf models: 0
- Quantized GEMMs detected: 0

---

## Validation Summary

| Check | Status |
|-------|--------|
| Time Sanity | PASS |
| Efficiency Anomalies | PASS |
| Coverage | PASS |
| Priority Consistency | INFO — Top by GPU time: [convolution] |

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Tune igemm tile for asymmetric 1×7 filter | kernel_tuning | 0.73 | medium |

---

### Hardware Reference

- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (FP32)**: 163 TFLOPS
- **Peak MAF (FP16)**: 654 TFLOPS
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
- **Memory**: 192 GB HBM3

---

*Report generated by TraceLens AgenticMode Standalone Analysis*
