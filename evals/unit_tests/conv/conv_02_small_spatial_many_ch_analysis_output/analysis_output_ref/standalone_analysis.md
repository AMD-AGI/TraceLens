# TraceLens Standalone Analysis Report

**Trace:** `conv_02_small_spatial_many_ch.json`
**Platform:** MI300X
**Total GPU Time:** 1.40 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of convolution trace `conv_02_small_spatial_many_ch` on MI300X. The trace contains a single `aten::convolution` with input [4,2048,4,4], weight [2048,2048,3,3], stride=(1,1), padding=(1,1). The operation achieves only 2.47% of peak FP32 (4.03 TFLOPS vs 163 TFLOPS) due to extreme weight-dominated workload with tiny spatial dimensions (4×4) and poor spatial parallelism.

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 1.4 ms |
| Computation | 85.7% |
| Communication | 0.0% |
| MemCpy | 0.0% |
| Idle | 14.3% |

GPU is 85.7% utilized for computation with 14.3% idle time.

---

## Prioritized Recommendations

### P1: Small-Spatial Many-Channel Convolution — Poor Spatial Parallelism

| | |
|---|---|
| **Category** | Convolution |
| **Operation** | aten::convolution (Input [4,2048,4,4], Weight [2048,2048,3,3], FP32) |
| **Current Efficiency** | 2.47% of peak MAF (4.03 TFLOPS vs 163 TFLOPS) |
| **Issue** | Tiny spatial dimensions (4×4) with huge channels (2048) create an extreme weight-dominated workload. The 64×64×16 tile cannot effectively parallelize across space. |
| **Recommendation** | Investigate batching across spatial positions or alternative GEMM decompositions. Consider fusing with adjacent layers to increase effective batch size. Profile for L2 cache thrashing from large weight tensor (2048×2048×3×3). |
| **Estimated Savings** | Up to 1.17 ms (medium confidence) |
| **Confidence** | Medium |

**Detail:** The aten::convolution (Input [4,2048,4,4], Weight [2048,2048,3,3], FP32, stride=(1,1), padding=(1,1)) achieves only 2.47% of peak FP32. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt64x64x16` (grid=[128,1,1], block=[128,1,1]) cannot effectively parallelize across space with only 4×4=16 spatial positions per batch. The massive weight tensor (2048×2048×9 ≈ 50M elements) dominates memory traffic and may cause cache pressure. HBM BW: 0.13 TB/s achieved vs 5.3 TB/s peak (2.5% of peak). Compute-bound (FLOPS/Byte = 31.78).

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

No system-level bottlenecks detected. GPU activity breakdown shows 85.7% computation, with negligible memcpy and communication overhead.

### CPU/Idle Time Analysis

GPU idle time is 14.3% of total trace time (1.4 ms). CPU duration: 1.4 ms, GPU kernel time: 1.2 ms (CPU/GPU ratio: 1.17x — no sync bottleneck).

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

Convolution accounts for 100% of compute time (1.2 ms GPU kernel time). Single `aten::convolution` dominates the trace. Input [4,2048,4,4], weight [2048,2048,3,3], FP32. Stride=(1,1), padding=(1,1), dilation=(1,1), groups=1. Achieves 2.47% of peak FP32.

#### Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::convolution | 1 | 1.200 | 100.0 | 2.47% | 31.78 | compute-bound |

#### Key Bottleneck: aten::convolution — Extreme Weight-Dominated Shape

- **Time:** 1.2 ms (100% of compute)
- **Efficiency:** 2.47% of peak MAF (4.03 TFLOPS achieved vs 163 TFLOPS peak FP32)
- **Issue:** Compute-bound convolution achieving only 2.47% of peak. Shape Input [4,2048,4,4], Weight [2048,2048,3,3] (stride=(1,1), padding=(1,1), dilation=(1,1), groups=1). With only 4×4=16 spatial positions per batch, the 64×64×16 tile cannot effectively parallelize across space. The massive weight tensor (2048×2048×9) dominates memory traffic and may cause cache pressure. HBM BW: 0.13 TB/s achieved vs 5.3 TB/s peak (2.5% of peak).
- **Algorithmic:** Investigate batching across spatial positions or alternative GEMM decompositions. Consider fusing with adjacent layers to increase effective batch size. Profile for L2 cache thrashing from large weight tensor (2048×2048×3×3 ≈ 50M elements).
- **Kernel:** Generate replay artifact for kernel team to evaluate tile sizes. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt64x64x16` (grid=[128,1,1], block=[128,1,1]) — investigate alternative tile configurations for small-spatial many-channel workloads.

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
| Tune igemm tile for small-spatial many-channel shape | kernel_tuning | 1.17 | medium |

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
