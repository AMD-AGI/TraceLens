# moe_06_tiny_batch_many_experts — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of MoE trace `moe_06_tiny_batch_many_experts` on MI300X. The trace contains a single fused MoE operation with 128 experts, topk=8, hidden=4096, intermediate=11008, and 4 tokens. **Extreme case**: 32 expert-token pairs across 128 experts — most experts idle. Launch overhead (0.38 ms CPU vs 0.18 ms GPU) suggests kernel launch is significant relative to compute.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.18 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | MoE Fused (0.18 ms) |
| Approx. FLOPs | ~5.77 GFLOPs |
| Efficiency (TFLOPS/BW) | N/A (no perf model for fused MoE) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute |
|------|-----------|----------|-----------|--------------|
| 1 | vllm::rocm_aiter_fused_moe | MoE Fused | 0.18 | 100.0% |

### 🔴 P1: Extreme Expert Under-Utilization + Launch Overhead Dominant

**Issue**: 128 experts with only 4 tokens and topk=8 yields 32 expert-token pairs across 128 experts — ~0.25 tokens per expert on average. Most experts receive zero tokens and remain idle. **Launch overhead is significant**: CPU duration 0.38 ms vs GPU kernel time 0.18 ms — 0.20 ms (53% of total) is launch/sync overhead. Fused MoE kernels have no efficiency metrics (TFLOPS, BW all null); this is expected.

**Action**: (1) **Batching**: Increase batch size dramatically (e.g., 64+ tokens) to amortize launch overhead and improve expert utilization. (2) **Expert count**: 128 experts for 4 tokens is severe overkill — consider 8–16 experts for tiny-batch workloads. (3) **topk**: topk=8 with 4 tokens means each token uses 8 experts; 32 pairs across 128 experts maximizes idle experts.

**Impact**: Algorithmic — batching and reducing experts can improve utilization. Launch overhead (0.20 ms) exceeds kernel time (0.18 ms); batching would amortize this.

→ *See Detailed Analysis: Compute Kernels > MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

### 🟡 P1: Kernel Launch Overhead Exceeds Compute Time

**Issue**: CPU duration (0.38 ms) exceeds GPU kernel time (0.18 ms) by 0.20 ms — launch/sync overhead is 53% of total wall-clock time. For such short kernels, launch overhead dominates.

**Action**: Batch multiple MoE invocations or increase token count to amortize launch overhead. Consider GPU graph capture to reduce per-kernel launch cost if applicable.

**Impact**: Batching to 64+ tokens could reduce overhead fraction from 53% to <10%.

→ *See Detailed Analysis: System-Level below*

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `vllm::rocm_aiter_fused_moe` operation dominates 100% of GPU compute (0.18 ms). Kernel `aiter_fused_moe_kernel_bf16`, grid=[32,1,1], block=[256,1,1]. **Efficiency metrics N/A** — fused MoE ops have no direct performance model mapping; this is expected.

**Time Breakdown:**
- GPU kernel time: 0.18 ms
- CPU duration: 0.38 ms (CPU/GPU ratio: 2.11x — launch overhead ~0.20 ms, 53% of total)
- Kernel: `aiter_fused_moe_kernel_bf16`, grid=[32,1,1], block=[256,1,1]

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound |
|-----------|-------|-----------|---------------|------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 0.18 | 100.0% | N/A | N/A |

**Key Metrics:**
- **Shapes:** tokens=[4,4096], w1=[128,22016,4096], w2=[128,4096,11008], topk_weights=[4,8]
- **Config:** 128 experts, topk=8, hidden=4096, intermediate=11008, 4 tokens
- **dtype:** BFloat16
- **Expert-token pairs:** 4 × 8 = 32 across 128 experts
- **Approx. FLOPs:** tokens × topk × (2 × hidden × intermediate × 2) ≈ 4 × 8 × 4 × 4096 × 11008 ≈ 5.77 GFLOPs

**Expert Utilization & Load Balancing:**
- 32 expert-token pairs / 128 experts ≈ 0.25 tokens per expert on average
- topk=8 with 4 tokens — each token routes to 8 experts; 32 pairs total across 128 experts
- Severe under-utilization: 96+ experts likely receive 0 tokens

**Recommendations:**
- **Algorithmic:** Batch to 64+ tokens to amortize launch overhead and improve utilization
- **Algorithmic:** Reduce expert count (8–16) for tiny-batch workloads
- **System:** GPU graph capture may reduce per-kernel launch cost if multiple MoE calls are batched

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|
| Batch to 64+ tokens | algorithmic | Amortize 0.20 ms overhead | high |
| Reduce expert count for tiny batches | algorithmic | N/A | medium |

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.18 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |

### Launch Overhead Analysis

| Metric | Value |
|--------|-------|
| GPU Kernel Time | 0.18 ms |
| CPU Duration | 0.38 ms |
| Launch/Sync Overhead | 0.20 ms (53% of total) |

**Note:** Launch overhead (0.20 ms) exceeds kernel compute time (0.18 ms). For such short kernels, batching or GPU graph capture is recommended to amortize per-kernel launch cost.

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
