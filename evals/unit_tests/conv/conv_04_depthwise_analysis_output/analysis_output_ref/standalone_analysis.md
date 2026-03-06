# conv_04_depthwise — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of convolution trace `conv_04_depthwise` on MI300X. The trace contains a single depthwise `aten::convolution` with input [8,1024,32,32], weight [1024,1,3,3], groups=1024. The operation achieves only 0.26% of peak FP32 (0.43 TFLOPS vs 163 TFLOPS). Depthwise conv is inherently memory-bound (FLOPS/Byte=2.25) yet classified as compute-bound; the specialized `depthwise_conv2d_fwd_fp32_kernel` shows extremely poor utilization.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.55 ms |
| Computation | 63.6% |
| Idle Time | 36.4% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | Convolution (0.35 ms) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::convolution (depthwise) [8,1024,32,32]×[1024,1,3,3] | Convolution | 0.350 | 100.0% |

### 🔴 P1: Depthwise Convolution — Extremely Low Efficiency Despite Specialized Kernel

**Issue**: Depthwise conv (groups=1024) has FLOPS/Byte=2.25 — inherently memory-bound — yet achieves only 0.26% of peak FP32 (0.43 TFLOPS vs 163 TFLOPS). The specialized `depthwise_conv2d_fwd_fp32_kernel` (grid=[256,1,1], block=[256,1,1]) shows very poor utilization; HBM BW is 0.19 TB/s vs 5.3 TB/s peak (3.6%).

**Action**: Profile the depthwise kernel for memory coalescing, warp occupancy, and L2 utilization. Consider algorithmic alternatives: fuse with adjacent pointwise conv (depthwise-separable block), or evaluate if im2col+GEMM path performs better for this shape. Investigate whether FP16/BF16 would improve throughput.

**Impact**: Estimated 0.35 ms savings (99.7% of kernel time) if efficiency improves — low confidence given the fundamental memory-bound nature of depthwise ops.

→ *See [Detailed Analysis: Compute Kernels > Convolution](#1-convolution-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 63.6% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. Convolution (100% of compute)

**Status:** SUCCESS

**Overview:**
Single depthwise `aten::convolution`. Input [8,1024,32,32], weight [1024,1,3,3], FP32. Stride=(1,1), padding=(1,1), dilation=(1,1), groups=1024. Achieves 0.26% of peak FP32.

**Time Breakdown:**
- GPU kernel time: 0.35 ms
- CPU duration: 0.55 ms (CPU/GPU ratio: 1.57x — no sync bottleneck)

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound |
|-----------|-------|-----------|---------------|------------|------------|-------|
| aten::convolution (depthwise) | 1 | 0.350 | 100.0% | 0.26% | 2.25 | compute |

**Key Bottleneck: aten::convolution — Depthwise with Poor Bandwidth Utilization**

- **Time:** 0.35 ms (100% of compute)
- **Shape:** Input [8,1024,32,32], Weight [1024,1,3,3], FP32, Groups=1024
- **Params:** Stride=(1,1), Padding=(1,1), Dilation=(1,1)
- **Kernel:** `depthwise_conv2d_fwd_fp32_kernel`, grid=[256,1,1], block=[256,1,1]
- **Efficiency:** 0.26% of peak FP32 (0.43 TFLOPS achieved vs 163 TFLOPS peak)
- **HBM BW:** 0.19 TB/s achieved vs 5.3 TB/s peak (3.6% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 2.25) — note: depthwise is inherently memory-bound; low FLOPS/Byte suggests bandwidth should dominate, yet efficiency is extremely poor

**Issue:** Depthwise convolutions have minimal arithmetic per byte (2.25 FLOPS/Byte), so they are typically memory-bound. Achieving only 3.6% of peak HBM bandwidth indicates severe kernel inefficiency — likely poor memory coalescing, low occupancy, or suboptimal tiling for the 32×32 spatial tiles with 1024 channels.

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Optimize depthwise kernel memory access patterns | kernel_tuning | 0.349 | low |
| Fuse with adjacent pointwise (depthwise-separable block) | algorithmic | TBD | medium |

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 0.55 ms |
| Computation | 63.6% |
| Idle Time | 36.4% |
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
