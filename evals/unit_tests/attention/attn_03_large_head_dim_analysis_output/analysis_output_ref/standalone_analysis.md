# TraceLens Standalone Analysis Report

**Trace:** `attn_03_large_head_dim.json`
**Platform:** MI300X
**Total GPU Time:** 3.2 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of attention trace `attn_03_large_head_dim` on MI300X. The trace contains a single Flash Attention SDPA operation with large head dimension (d=256) and few heads (H=8). GPU utilization is 100% computation with zero idle time. Efficiency is 6.07% of peak BF16 (42.95 TFLOPS achieved vs 708 TFLOPS peak), indicating significant kernel tuning opportunity for this head-dim configuration.

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 3.2 ms |
| Computation | 100% |
| Communication | 0.0% |
| MemCpy | 0.0% |
| Idle | 0% |

---

## Prioritized Recommendations

### P1: Large Head-Dim Flash Attention — Low Efficiency, Kernel Tuning Opportunity

| | |
|---|---|
| **Category** | SDPA_fwd |
| **Operation** | aten::_scaled_dot_product_flash_attention (B=4, N=2048, H=8, d=256, BF16) |
| **Issue** | `aten::_scaled_dot_product_flash_attention` with shape B=4, N=2048, H=8, d_h=256 achieves only 6.07% of peak BF16 efficiency (42.95 TFLOPS vs 708 TFLOPS). The kernel (`flash_fwd_kernel_bf16_hdim256`) is compute-bound (FLOPS/Byte=1024.0) but underutilizes the GPU. Large head dim (256) with few heads (8) may stress tile/warp configuration differently than the common d=128 case. |
| **Recommendation** | (1) Generate replay artifact for kernel team to profile `flash_fwd_kernel_bf16_hdim256` — tile size, warp occupancy, and register usage for d=256. (2) Compare against d=128 kernel configurations; d=256 may need different tuning. (3) Evaluate whether GQA or multi-query variants could improve utilization for this head configuration. |
| **Estimated Savings** | ~3.006 ms |
| **Confidence** | Medium |

**Detail:** The d=256 head dimension uses a specialized kernel (`flash_fwd_kernel_bf16_hdim256`). With FLOPS/Byte=1024.0, the operation is firmly compute-bound. At 6.07% efficiency, the kernel is underutilizing the GPU — likely due to suboptimal tile/warp configuration for the d=256 case, which is less common than d=128. Few heads (8) with large dim may also limit parallelism.

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

No system-level bottlenecks detected. GPU activity breakdown shows 100% computation. CPU duration (3.5 ms) is slightly higher than GPU kernel time (3.2 ms), indicating minimal launch overhead (~9% ratio).

No CPU/idle or multi-kernel bottlenecks. Single-kernel trace with full GPU utilization during compute.

---

## Compute Kernel Analysis

### 1. SDPA_fwd (100% of compute)

**Status:** SUCCESS

SDPA_fwd accounts for 100% of GPU compute time (3.2 ms). A single `aten::_scaled_dot_product_flash_attention` with large head dimension (d=256) and 8 heads. Flash Attention is active. Average efficiency is 6.07% — below expected range for this workload size; kernel tuning opportunity.

#### Time Breakdown

- GPU kernel time: 3.2 ms (3200 µs)
- CPU duration: 3.5 ms (CPU/GPU ratio: 1.09x — minimal sync overhead)
- Kernel: `flash_fwd_kernel_bf16_hdim256`, duration: 3200 µs

#### Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound | Attention Type |
|-----------|-------|-----------|---------------|------------|------------|-------|----------------|
| aten::_scaled_dot_product_flash_attention | 1 | 3.2 | 100.0% | 6.07% | 1024.0 | compute | flash |

#### Key Metrics

- **Shape:** B=4, N_Q=2048, H_Q=8, N_KV=2048, H_KV=8, d_h_qk=256, d_h_v=256
- **Input Dims:** (4, 8, 2048, 256) ×3, BFloat16
- **TFLOPS achieved:** 42.95 vs 708 TFLOPS peak BF16
- **HBM BW achieved:** 0.04 TB/s vs 5.3 TB/s peak (0.8% of peak — expected for compute-bound)
- **Bound type:** Compute-bound (FLOPS/Byte = 1024.0)
- **Flash attention:** True | **Causal:** False | **Dropout:** 0.0

#### Bottleneck Analysis

The d=256 head dimension uses a specialized kernel (`flash_fwd_kernel_bf16_hdim256`). With FLOPS/Byte=1024.0, the operation is firmly compute-bound. At 6.07% efficiency, the kernel is underutilizing the GPU — likely due to suboptimal tile/warp configuration for the d=256 case, which is less common than d=128. Few heads (8) with large dim may also limit parallelism.

#### Recommendations

- **Kernel tuning:** Profile with hardware counters (occupancy, warp stalls, memory coalescing) for d=256
- **Tile configuration:** The d=256 kernel may need different tile sizes than d=128; generate replay artifact for kernel team
- **Algorithmic:** If model flexibility allows, d=128 with 16 heads has better-tuned kernels; evaluate trade-off
- Flash Attention is the right backend; focus on kernel implementation for d=256

#### Additional Notes

- Achieved TFLOPS: 42.95 (BF16)
- HBM BW Achieved: 0.04 TB/s
- FLOPS/Byte: 1024.0

---

## Validation Summary

| Check | Status |
|-------|--------|
| Time Sanity | PASS |
| Efficiency Anomalies | PASS |
| Coverage | PASS |
| Priority Consistency | INFO — Top by GPU time: [SDPA_fwd] |

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|
| aten::_scaled_dot_product_flash_attention kernel tuning (d=256) | kernel_tuning | 3.006 | medium |
| Tile/warp configuration for flash_fwd_kernel_bf16_hdim256 | kernel_tuning | 1.0–1.5 | medium |

---

### Hardware Reference

- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP16)**: 654 TFLOPS
- **Peak MAF (FP32)**: 163 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
- **Memory**: 192 GB HBM3

---

*Report generated by TraceLens AgenticMode Standalone Analysis*
