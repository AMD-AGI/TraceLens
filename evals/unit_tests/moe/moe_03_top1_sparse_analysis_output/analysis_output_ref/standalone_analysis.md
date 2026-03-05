# moe_03_top1_sparse — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of MoE trace `moe_03_top1_sparse` on MI300X. The trace contains a single fused MoE operation with 16 experts, topk=1, hidden=7168, intermediate=14336, and 512 tokens. **Sparse routing (topk=1)** — each expert gets ~32 tokens on average; load imbalance possible.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.95 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | MoE Fused (0.95 ms) |
| Approx. FLOPs | ~210.5 GFLOPs |
| Efficiency (TFLOPS/BW) | N/A (no perf model for fused MoE) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute |
|------|-----------|----------|-----------|--------------|
| 1 | vllm::rocm_aiter_fused_moe | MoE Fused | 0.95 | 100.0% |

### 🔴 P1: Sparse Routing (topk=1) — Potential Load Imbalance

**Issue**: 16 experts with 512 tokens and topk=1 yields 512 expert-token pairs across 16 experts (~32 tokens/expert on average). Sparse routing can cause load imbalance — some experts may receive 0 tokens while others are overloaded. Fused MoE kernels have no efficiency metrics (TFLOPS, BW all null); this is expected.

**Action**: (1) Profile expert utilization at runtime; if imbalance >2× (e.g., one expert 64 tokens, another 0), consider capacity factor or load-balancing loss. (2) If balanced, topk=1 is compute-efficient (half the FLOPs of topk=2). (3) Compare with moe_04_top2_dense (same model, topk=2) — 1.8 ms vs 0.95 ms shows topk=2 doubles compute as expected.

**Impact**: Not quantifiable from trace data.

→ *See Detailed Analysis: Compute Kernels > MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation, 0% idle. CPU duration (1.15 ms) vs GPU kernel time (0.95 ms) indicates ~0.20 ms launch/sync overhead — acceptable.

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `vllm::rocm_aiter_fused_moe` operation dominates 100% of GPU compute (0.95 ms). Kernel `aiter_fused_moe_kernel_bf16`, grid=[128,1,1], block=[256,1,1]. **Efficiency metrics N/A** — fused MoE ops have no direct performance model mapping; this is expected.

**Time Breakdown:**
- GPU kernel time: 0.95 ms
- CPU duration: 1.15 ms (CPU/GPU ratio: 1.21x — launch overhead ~0.20 ms)
- Kernel: `aiter_fused_moe_kernel_bf16`, grid=[128,1,1], block=[256,1,1]

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound |
|-----------|-------|-----------|---------------|------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 0.95 | 100.0% | N/A | N/A |

**Key Metrics:**
- **Shapes:** tokens=[512,7168], w1=[16,28672,7168], w2=[16,7168,14336], topk_weights=[512,1]
- **Config:** 16 experts, topk=1, hidden=7168, intermediate=14336, 512 tokens
- **dtype:** BFloat16
- **Expert-token pairs:** 512 × 1 = 512 across 16 experts
- **Approx. FLOPs:** tokens × topk × (2 × hidden × intermediate × 2) ≈ 512 × 1 × 4 × 7168 × 14336 ≈ 210.5 GFLOPs

**Expert Utilization & Load Balancing:**
- 512 expert-token pairs / 16 experts ≈ 32 tokens per expert on average
- topk=1 (sparse) — each token routes to exactly one expert; routing distribution determines balance
- If routing is uniform, load is balanced; if skewed (e.g., popular experts), some experts idle

**Recommendations:**
- **Profiling:** Measure expert utilization; target <2× max/min ratio
- **Algorithmic:** If imbalanced, apply auxiliary load-balancing loss during training
- **Comparison:** topk=1 is 2× faster than topk=2 for same model (0.95 ms vs 1.8 ms in moe_04)

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.95 ms |
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
