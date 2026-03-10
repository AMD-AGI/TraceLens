# TraceLens Standalone Analysis Report

**Trace:** `attn_02_short_seq_overhead.json`
**Platform:** MI300X
**Total GPU Time:** 0.025 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

Standalone performance analysis of attention trace `attn_02_short_seq_overhead` on MI300X. The trace contains a single Flash Attention SDPA operation with very short sequence length (N=16) and large batch (B=32). GPU compute time is only 0.025 ms, but CPU duration is 0.2 ms — an 8× launch-overhead ratio indicating extreme kernel launch overhead dominates over actual compute. Efficiency is 0.76% of peak BF16 (5.37 TFLOPS achieved vs 708 TFLOPS peak).

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 0.025 ms |
| Computation | 100% |
| Communication | 0.0% |
| MemCpy | 0.0% |
| Idle | 0% |

---

## Prioritized Recommendations

### P1: Short-Sequence Flash Attention — Launch Overhead Dominates Compute

| | |
|---|---|
| **Category** | SDPA_fwd |
| **Operation** | aten::_scaled_dot_product_flash_attention (B=32, N=16, H=32, d=128, BF16) |
| **Issue** | `aten::_scaled_dot_product_flash_attention` with shape B=32, N_Q=16, N_KV=16 achieves only 0.76% of peak BF16 efficiency (5.37 TFLOPS vs 708 TFLOPS). The kernel (`flash_fwd_kernel_bf16`) runs for 25 µs while CPU duration is 200 µs — launch/scheduling overhead is ~7× the actual compute time. With FLOPS/Byte=8.0 (compute-bound), the tiny workload cannot amortize fixed launch costs. |
| **Recommendation** | (1) Batch/sequence consolidation — aggregate multiple short sequences into fewer, longer batches to amortize launch overhead. (2) Kernel fusion — if this SDPA is part of a larger block (e.g., transformer layer), fuse with adjacent ops to reduce kernel launches. (3) Evaluate whether Flash Attention is the right backend for N=16; a simpler implementation may have lower overhead. |
| **Estimated Savings** | ~0.025 ms |
| **Confidence** | Low |

**Detail:** The workload is extremely small: 32×16×32×128 ≈ 2M elements per tensor. With FLOPS/Byte=8.0, the operation is compute-bound, but the kernel runs for only 25 µs. Fixed launch overhead (driver, scheduler, kernel dispatch) dominates. No amount of kernel tuning can overcome this — the solution is structural (batching, fusion, or different backend for short sequences).

---

## System-Level Analysis

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

Launch overhead dominates. CPU duration (0.2 ms) is 8× GPU kernel time (0.025 ms). This pattern indicates:
- Multiple short kernels with high per-launch overhead
- Opportunity for batch consolidation or kernel fusion at the model/application level
- Consider `torch.compile` or custom fused blocks to reduce launch count

### Launch Overhead

| Metric | Value |
|--------|-------|
| GPU Kernel Time | 0.025 ms |
| CPU Duration | 0.2 ms |
| Overhead Ratio | 8.0× |
| Overhead Time | ~0.175 ms |

The 8× CPU/GPU ratio indicates that launch and scheduling overhead dominate. System-level optimization (batching, fusion) is the primary lever.

---

## Compute Kernel Analysis

### 1. SDPA_fwd (100% of compute)

**Status:** SUCCESS

SDPA_fwd accounts for 100% of GPU compute time (0.025 ms). A single `aten::_scaled_dot_product_flash_attention` with very short sequence (N=16) and batch 32. Flash Attention is active. Efficiency is 0.76% — dominated by launch overhead, not kernel efficiency.

#### Time Breakdown

- GPU kernel time: 0.025 ms (25 µs)
- CPU duration: 0.2 ms (200 µs) — 8× overhead
- Kernel: `flash_fwd_kernel_bf16`, duration: 25 µs

#### Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound | Attention Type |
|-----------|-------|-----------|---------------|------------|------------|-------|----------------|
| aten::_scaled_dot_product_flash_attention | 1 | 0.025 | 100.0% | 0.76% | 8.0 | compute | flash |

#### Key Metrics

- **Shape:** B=32, N_Q=16, H_Q=32, N_KV=16, H_KV=32, d_h_qk=128, d_h_v=128
- **Input Dims:** (32, 32, 16, 128) ×3, BFloat16
- **TFLOPS achieved:** 5.37 vs 708 TFLOPS peak BF16
- **HBM BW achieved:** 0.67 TB/s vs 5.3 TB/s peak (12.6% of peak)
- **Bound type:** Compute-bound (FLOPS/Byte = 8.0)
- **Flash attention:** True | **Causal:** False | **Dropout:** 0.0

#### Bottleneck Analysis

The workload is extremely small: 32×16×32×128 ≈ 2M elements per tensor. With FLOPS/Byte=8.0, the operation is compute-bound, but the kernel runs for only 25 µs. Fixed launch overhead (driver, scheduler, kernel dispatch) dominates. No amount of kernel tuning can overcome this — the solution is structural (batching, fusion, or different backend for short sequences).

#### Recommendations

- **Batch/sequence consolidation:** Merge short sequences into longer batches to amortize launch overhead
- **Kernel fusion:** Fuse SDPA with surrounding ops (e.g., QKV projection, output projection) to reduce launch count
- **Backend selection:** For N=16, evaluate whether a non-Flash path (e.g., math attention) has lower overhead
- Flash Attention is optimal for long sequences; for N=16, the overhead may outweigh benefits

#### Additional Notes

- CPU/GPU Duration Ratio: 8.0× (0.2 ms / 0.025 ms)
- Achieved TFLOPS: 5.37 (BF16)
- HBM BW Achieved: 0.67 TB/s

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
| Kernel tuning (limited upside) | kernel_tuning | 0.025 | low |

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
