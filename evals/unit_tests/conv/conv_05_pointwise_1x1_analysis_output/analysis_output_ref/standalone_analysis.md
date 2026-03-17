# TraceLens Standalone Analysis Report

**Trace:** `conv_05_pointwise_1x1.json`
**Platform:** MI300X
**Total GPU Time:** 0.8 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of convolution trace `conv_05_pointwise_1x1` on MI300X. The trace contains a single 1×1 pointwise `aten::convolution` with input [16,1024,14,14], weight [2048,1024,1,1], stride=(1,1), padding=(0,0). The operation achieves 13.45% of peak FP32 (21.92 TFLOPS vs 163 TFLOPS) — the best efficiency among the convolution traces — but still leaves significant headroom. 1×1 pointwise is essentially a batched GEMM.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.8 ms |
| Computation | 75.0% |
| Idle Time | 25.0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (0.6 ms) |

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 0.8 ms |
| Computation | 75.0% |
| Idle Time | 25.0% |
| Exposed Communication | 0.00% |

---

## Prioritized Recommendations

### P1: 1×1 Pointwise Convolution — Best Efficiency but Still Suboptimal

| | |
|---|---|
| **Category** | Convolution |
| **Operation** | aten::convolution (1×1) [16,1024,14,14]×[2048,1024,1,1] |
| **Current Efficiency** | 13.45% of peak FP32 (21.92 TFLOPS vs 163 TFLOPS) |
| **HBM BW** | 0.08 TB/s vs 5.3 TB/s peak (1.5%) |
| **Issue** | 1×1 pointwise (batched GEMM) achieves 13.45% of peak FP32 — best among convolutions — but still only ~1/7 of peak. Kernel `igemm_fwd_gtcx_nhwc_fp32_bx0_bt256x128x16` (grid=[512,1,1], block=[256,1,1]) has FLOPS/Byte=280.31 (compute-bound) |
| **Recommendation** | Generate replay artifact for kernel team. 1×1 conv maps cleanly to GEMM; investigate whether tile sizes 256×128×16 are optimal for batch×spatial=16×14×14=3136. Consider `torch.compile` or cuDNN/cuBLAS backend selection for pointwise-heavy models |
| **Estimated Savings** | ~0.52 ms from closing efficiency gaps (pre-computed) |
| **Confidence** | Medium |

**Detail:** 1×1 conv reduces to batched GEMM with M=batch×H×W, N=out_ch, K=in_ch. Shape (16×14×14, 2048, 1024) = (3136, 2048, 1024) is a reasonable GEMM; the 256×128×16 tile may not fully utilize the 3136 rows or 2048 columns.

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

No system-level bottlenecks detected. GPU activity breakdown shows 75.0% computation, with negligible memcpy and communication overhead.

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

**Overview:**
Single 1×1 pointwise `aten::convolution`. Input [16,1024,14,14], weight [2048,1024,1,1], FP32. Stride=(1,1), padding=(0,0), dilation=(1,1), groups=1. Achieves 13.45% of peak FP32 — best among convolution traces.

**Time Breakdown:**
- GPU kernel time: 0.6 ms
- CPU duration: 0.8 ms (CPU/GPU ratio: 1.33x — no sync bottleneck)

**Top Operations:**

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution (1×1) [16,1024,14,14]×[2048,1024,1,1] | Convolution | 0.600 | 100.0% |

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution (1×1) | 1 | 0.600 | 100.0% | 13.45% | 280.31 | compute |

**Key Bottleneck: aten::convolution — 1×1 Pointwise (Batched GEMM)**

- **Time:** 0.6 ms (100% of compute)
- **Shape:** Input [16,1024,14,14], Weight [2048,1024,1,1], FP32
- **Params:** Stride=(1,1), Padding=(0,0), Dilation=(1,1), Groups=1
- **Kernel:** `igemm_fwd_gtcx_nhwc_fp32_bx0_bt256x128x16`, grid=[512,1,1], block=[256,1,1]
- **Efficiency:** 13.45% of peak FP32 (21.92 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.08 TB/s achieved vs 5.3 TB/s peak (1.5% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 280.31)

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
| Tune igemm tile for 1×1 pointwise (batched GEMM) shape | kernel_tuning | 0.52 | medium |

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
