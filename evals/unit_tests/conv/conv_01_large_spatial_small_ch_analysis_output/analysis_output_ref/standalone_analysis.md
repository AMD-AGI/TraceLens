# TraceLens Standalone Analysis Report

**Trace:** `conv_01_large_spatial_small_ch.json`
**Platform:** MI300X
**Total GPU Time:** 4.70 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of convolution trace `conv_01_large_spatial_small_ch` on MI300X. The trace contains a single `aten::convolution` with input [1,3,2048,2048], weight [64,3,7,7], stride=(2,2), padding=(3,3). The operation achieves only 2.69% of peak FP32 (4.38 TFLOPS vs 163 TFLOPS) due to poor utilization of igemm tiles for this large-spatial, small-channel workload.

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 4.7 ms |
| Computation | 95.7% |
| Communication | 0.0% |
| MemCpy | 0.0% |
| Idle | 4.3% |

GPU is 95.7% utilized for computation with minimal idle time.

---

## Prioritized Recommendations

### P1: Large-Spatial Small-Channel Convolution — Poor igemm Tile Utilization

| | |
|---|---|
| **Category** | Convolution |
| **Operation** | aten::convolution (Input [1,3,2048,2048], Weight [64,3,7,7], FP32) |
| **Current Efficiency** | 2.69% of peak MAF (4.38 TFLOPS vs 163 TFLOPS) |
| **Issue** | Very large spatial dimensions (2048×2048) with tiny input channels (3) cause poor utilization of the igemm tiles. The 128×128×32 tile configuration is poorly matched to the 3 input channels. |
| **Recommendation** | Investigate alternative tile configurations or specialized kernels for large-spatial/small-channel workloads. Consider im2col+GEMM fusion strategies that better amortize weight loads across spatial positions. |
| **Estimated Savings** | Up to 4.379 ms (medium confidence) |
| **Confidence** | Medium |

**Detail:** The aten::convolution (Input [1,3,2048,2048], Weight [64,3,7,7], FP32, stride=(2,2), padding=(3,3)) achieves only 2.69% of peak FP32. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_ex0_bt128x128x32` (grid=[1024,1,1], block=[256,1,1]) is poorly matched to the 3 input channels — each spatial position has minimal channel dimension to amortize tile overhead, leading to severe underutilization. HBM BW: 0.07 TB/s achieved vs 5.3 TB/s peak (1.3% of peak). Compute-bound (FLOPS/Byte = 61.89).

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

No system-level bottlenecks detected. GPU activity breakdown shows 95.7% computation, with negligible memcpy and communication overhead.

### CPU/Idle Time Analysis

GPU idle time is 4.3% of total trace time (4.7 ms). CPU duration: 4.7 ms, GPU kernel time: 4.5 ms (CPU/GPU ratio: 1.04x — no sync bottleneck).

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

Convolution accounts for 100% of compute time (4.5 ms GPU kernel time). Single `aten::convolution` dominates the trace. Input [1,3,2048,2048], weight [64,3,7,7], FP32. Stride=(2,2), padding=(3,3), dilation=(1,1), groups=1. Achieves 2.69% of peak FP32.

#### Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::convolution | 1 | 4.500 | 100.0 | 2.69% | 61.89 | compute-bound |

#### Key Bottleneck: aten::convolution — Large Spatial, Small Channels

- **Time:** 4.5 ms (100% of compute)
- **Efficiency:** 2.69% of peak MAF (4.38 TFLOPS achieved vs 163 TFLOPS peak FP32)
- **Issue:** Compute-bound convolution achieving only 2.69% of peak. Shape Input [1,3,2048,2048], Weight [64,3,7,7] (stride=(2,2), padding=(3,3), dilation=(1,1), groups=1). The 128×128×32 tile configuration is poorly matched to the 3 input channels. Each spatial position has minimal channel dimension to amortize tile overhead, leading to severe underutilization. HBM BW: 0.07 TB/s achieved vs 5.3 TB/s peak (1.3% of peak).
- **Algorithmic:** Investigate im2col+GEMM fusion strategies that better amortize weight loads across spatial positions. Consider alternative decompositions for large-spatial/small-channel workloads.
- **Kernel:** Generate replay artifact for kernel team to evaluate tile sizes. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_ex0_bt128x128x32` (grid=[1024,1,1], block=[256,1,1]) — investigate alternative tile configurations optimized for small input channel counts.

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
| Tune igemm tile for large-spatial small-channel shape | kernel_tuning | 4.379 | medium |

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
