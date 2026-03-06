# conv_06_large_dilation — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_06_large_dilation` on MI300X. The trace contains a single `aten::convolution` with input [2,256,128,128], weight [256,256,3,3], stride=(1,1), padding=(16,16), dilation=(16,16). The operation achieves 10.78% of peak FP32 (17.57 TFLOPS vs 163 TFLOPS). Large dilation (16×16) causes highly non-contiguous memory access patterns, reducing effective bandwidth.

| Metric | Value |
|--------|-------|
| Total Compute Time | 2.4 ms |
| Computation | 91.7% |
| Idle Time | 8.3% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (2.2 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution [2,256,128,128]×[256,256,3,3], dilation=16 | Convolution | 2.200 | 100.0% |

### 🔴 P1: Large Dilation (16×16) — Non-Contiguous Memory Access Reducing Effective Bandwidth

**Issue**: Dilation=(16,16) causes the 3×3 filter to sample input at 16-pixel strides, creating highly non-contiguous memory access patterns. Kernel `igemm_fwd_gtcx_nhwc_fp32_dilation_bx0` (grid=[512,1,1], block=[256,1,1]) achieves 10.78% of peak FP32 (17.57 TFLOPS vs 163 TFLOPS). HBM BW is only 0.03 TB/s vs 5.3 TB/s peak (0.6%).

**Action**: Generate replay artifact for kernel team to profile dilated convolution memory access. Investigate specialized dilated im2col or gather-scatter patterns. Consider algorithmic alternatives: decompose dilated conv into separable operations or use atrous convolution optimizations.

**Impact**: Estimated 1.96 ms savings (89.1% of kernel time) if efficiency approaches peak — medium confidence given the known difficulty of dilated conv memory patterns.

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 91.7% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single dilated `aten::convolution`. Input [2,256,128,128], weight [256,256,3,3], FP32. Stride=(1,1), padding=(16,16), dilation=(16,16), groups=1. Achieves 10.78% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 2.2 ms
- CPU duration: 2.4 ms (CPU/GPU ratio: 1.09x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution (dilated) | 1 | 2.200 | 100.0% | 10.78% | 556.44 | compute |

**Key Bottleneck: aten::convolution — Large Dilation Causing Strided Access**

- **Time:** 2.2 ms (100% of compute)
- **Shape:** Input [2,256,128,128], Weight [256,256,3,3], FP32
- **Params:** Stride=(1,1), Padding=(16,16), Dilation=(16,16), Groups=1
- **Kernel:** `igemm_fwd_gtcx_nhwc_fp32_dilation_bx0`, grid=[512,1,1], block=[256,1,1]
- **Efficiency:** 10.78% of peak FP32 (17.57 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.03 TB/s achieved vs 5.3 TB/s peak (0.6% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 556.44)

**Issue:** With dilation=16, each of the 9 filter positions samples input 16 pixels apart. This creates strided, non-contiguous reads that break coalescing and reduce effective bandwidth. The dilation-specific kernel path exists but may not fully optimize for this extreme dilation factor.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Optimize dilated conv memory access patterns | kernel_tuning | 1.96 | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 2.4 ms |
| Computation | 91.7% |
| Idle Time | 8.3% |
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
