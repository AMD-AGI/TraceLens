# conv_07_transposed — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_07_transposed` on MI300X. The trace contains a single transposed (deconvolution) `aten::convolution` with input [4,512,8,8], weight [512,256,4,4], stride=(2,2), padding=(1,1), transposed=True. The operation achieves only 0.73% of peak FP32 (1.19 TFLOPS vs 163 TFLOPS). The backward-data kernel path (`igemm_bwd_data_gtcx_nhwc_fp32_bx0_bt128x128x16`) is less optimized than forward path, resulting in very low efficiency.

| Metric | Value |
|--------|-------|
| Total Compute Time | 1.1 ms |
| Computation | 81.8% |
| Idle Time | 18.2% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (0.9 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution (transposed) [4,512,8,8]×[512,256,4,4] | Convolution | 0.900 | 100.0% |

### 🔴 P1: Transposed Convolution — Backward-Data Kernel Path Less Optimized

**Issue**: Transposed conv uses the backward-data kernel path `igemm_bwd_data_gtcx_nhwc_fp32_bx0_bt128x128x16` (grid=[256,1,1], block=[256,1,1]), which is typically less optimized than the forward path. Achieves only 0.73% of peak FP32 (1.19 TFLOPS vs 163 TFLOPS). HBM BW is 0.01 TB/s vs 5.3 TB/s peak (0.2%).

**Action**: Generate replay artifact for kernel team to prioritize transposed/deconv optimization. Compare against cuDNN transposed conv performance. Consider algorithmic alternatives: replace transposed conv with bilinear upsample + standard conv where acceptable for the model.

**Impact**: ~0.89 ms savings from closing efficiency gaps (pre-computed).

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 81.8% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single transposed (deconvolution) `aten::convolution`. Input [4,512,8,8], weight [512,256,4,4], FP32, transposed=True. Stride=(2,2), padding=(1,1), dilation=(1,1), groups=1. Achieves 0.73% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 0.9 ms
- CPU duration: 1.1 ms (CPU/GPU ratio: 1.22x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution (transposed) | 1 | 0.900 | 100.0% | 0.73% | 107.79 | compute |

**Key Bottleneck: aten::convolution — Transposed (Deconvolution) with Backward-Data Path**

- **Time:** 0.9 ms (100% of compute)
- **Shape:** Input [4,512,8,8], Weight [512,256,4,4], FP32, Transposed=True
- **Params:** Stride=(2,2), Padding=(1,1), Dilation=(1,1), Groups=1
- **Kernel:** `igemm_bwd_data_gtcx_nhwc_fp32_bx0_bt128x128x16`, grid=[256,1,1], block=[256,1,1]
- **Efficiency:** 0.73% of peak FP32 (1.19 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.01 TB/s achieved vs 5.3 TB/s peak (0.2% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 107.79)

**Issue:** Transposed convolution (deconvolution) is implemented via the backward-data gradient path (gradient w.r.t. input). This path receives less optimization attention than the forward path. The 128×128×16 tile and scatter-style output writes for transposed conv may cause poor coalescing and low utilization.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Optimize transposed conv (bwd_data) kernel path | kernel_tuning | 0.89 | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 1.1 ms |
| Computation | 81.8% |
| Idle Time | 18.2% |
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
