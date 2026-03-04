# attn_01_long_seq_flash — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of attention trace `attn_01_long_seq_flash` on MI300X. The trace contains a single Flash Attention SDPA operation with long sequence length (N=32768). GPU utilization is 100% computation with zero idle time. **⚠️ [ANOMALY]** Compute efficiency reports 138.04% of peak BF16 (977.34 TFLOPS achieved vs 708 TFLOPS peak) — verify measurement methodology or peak specification.

| Metric | Value |
|--------|-------|
| Total Compute Time | 18.0 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | SDPA_fwd (18.0 ms) |
| Achieved TFLOPS | 977.34 (BF16) |
| Peak TFLOPS | 708 (BF16) |
| HBM BW Achieved | 0.06 TB/s |
| HBM BW Peak | 5.3 TB/s |
| FLOPS/Byte | 16384.0 |
| Bound Type | compute |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | aten::_scaled_dot_product_flash_attention (B=1, N=32768, H=32, d=128) | SDPA_fwd | 18.0 | 100.0% |

### 🔴 P1: [ANOMALY] Compute Efficiency Exceeds Peak — Verification Required

**Issue**: `aten::_scaled_dot_product_flash_attention` reports 138.04% of peak BF16 efficiency (977.34 TFLOPS achieved vs 708 TFLOPS peak). Compute efficiency cannot exceed 100% — this indicates a measurement artifact, peak spec mismatch, or FLOP-count model error.

**Action**: Verify (1) FLOP model for Flash Attention on this shape, (2) MI300X peak BF16 spec (708 TFLOPS), (3) kernel timing measurement (18.0 ms GPU kernel time vs 18.2 ms CPU duration). Do not optimize based on this metric until anomaly is resolved.

**Impact**: 0 ms savings — no actionable kernel optimization until anomaly is explained. High confidence that current kernel (`flash_fwd_splitkv_kernel_bf16`) is performing well for long-sequence Flash Attention.

→ *See [Detailed Analysis: Compute Kernels > SDPA_fwd](#1-sdpa_fwd-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation, with negligible memcpy and communication overhead. CPU duration (18.2 ms) closely matches GPU kernel time (18.0 ms), indicating minimal launch/sync overhead.

---

## Detailed Analysis: Compute Kernels

### 1. SDPA_fwd (100% of compute)

**Status:** SUCCESS

**Overview:**
SDPA_fwd accounts for 100% of GPU compute time (18.0 ms). A single `aten::_scaled_dot_product_flash_attention` operation dominates. Flash Attention is active. **⚠️ [ANOMALY]** Average efficiency reported at 138.04% of peak BF16.

**Time Breakdown:**
- GPU kernel time: 18.0 ms
- CPU duration: 18.2 ms (CPU/GPU ratio: 1.01x — minimal sync overhead)
- Kernel: `flash_fwd_splitkv_kernel_bf16`, duration: 18000 µs

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Bound | Attention Type |
|-----------|-------|-----------|---------------|------------|------------|-------|----------------|
| aten::_scaled_dot_product_flash_attention | 1 | 18.0 | 100.0% | 138.04% [ANOMALY] | 16384.0 | compute | flash |

**Key Metrics:**
- **Shape:** B=1, N_Q=32768, H_Q=32, N_KV=32768, H_KV=32, d_h_qk=128, d_h_v=128
- **Input Dims:** (1, 32, 32768, 128) ×3, BFloat16
- **TFLOPS achieved:** 977.34 vs 708 TFLOPS peak BF16
- **HBM BW achieved:** 0.06 TB/s vs 5.3 TB/s peak (1.1% of peak — expected for compute-bound)
- **Bound type:** Compute-bound (FLOPS/Byte = 16384.0)
- **Flash attention:** True | **Causal:** False | **Dropout:** 0.0

**Anomaly Note:**
Compute efficiency exceeds peak by 38.0%. Possible causes: (1) FLOP model overcounts for fused Flash Attention, (2) peak spec understates achievable throughput for this kernel, (3) timing includes overlapping work. Flag for verification before any optimization decisions.

**Recommendations:**
- Flash Attention is already the optimal implementation for this long-sequence case
- No kernel tuning recommended until anomaly is resolved
- Consider documenting this shape as a reference for Flash Attention efficiency validation

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|
| Verify measurement/peak spec before optimization | verification | N/A | high |
| Kernel already optimal for long-sequence Flash Attention | — | 0 | high |

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 18.0 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |

No CPU/idle or multi-kernel bottlenecks. Single-kernel trace with full GPU utilization.

---

## Appendix

### Hardware Reference
- **Platform:** MI300X
- **Peak HBM BW:** 5.3 TB/s
- **Peak MAF (BF16):** 708 TFLOPS
- **Peak MAF (FP16):** 654 TFLOPS
- **Peak MAF (FP32):** 163 TFLOPS
- **Peak MAF (FP8):** 1273 TFLOPS
- **Memory:** 192 GB HBM3
