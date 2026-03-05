# moe_01_many_experts_few_tokens — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of MoE trace `moe_01_many_experts_few_tokens` on MI300X. The trace contains a single fused MoE operation with 64 experts, topk=2, hidden=4096, intermediate=11008, and 16 tokens. **Extreme expert under-utilization**: 32 expert-token pairs distributed across 64 experts — most experts idle.

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.28 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | MoE Fused (0.28 ms) |
| Approx. FLOPs | ~5.77 GFLOPs |
| Efficiency (TFLOPS/BW) | N/A (no perf model for fused MoE) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute |
|------|-----------|----------|-----------|--------------|
| 1 | vllm::rocm_aiter_fused_moe | MoE Fused | 0.28 | 100.0% |

### 🔴 P1: Extreme Expert Under-Utilization — Many Experts, Few Tokens

**Issue**: 64 experts with only 16 tokens and topk=2 yields 32 expert-token pairs across 64 experts. On average, only ~0.5 tokens per expert — most experts receive zero tokens and remain idle. Fused MoE kernels have no efficiency metrics (TFLOPS, BW all null); this is expected for fused MoE.

**Action**: (1) Increase batch size or sequence length to improve token count (e.g., 256+ tokens for 64 experts). (2) Consider reducing expert count if small batches are inherent (e.g., 8–16 experts for 16-token workloads). (3) Use auxiliary load-balancing loss during training to encourage more uniform routing.

**Impact**: Not quantifiable from trace data.

→ *See Detailed Analysis: Compute Kernels > MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation, 0% idle. CPU duration (0.48 ms) vs GPU kernel time (0.28 ms) indicates ~0.20 ms launch/sync overhead — acceptable for this short kernel.

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `vllm::rocm_aiter_fused_moe` operation dominates 100% of GPU compute (0.28 ms). Kernel `aiter_fused_moe_kernel_bf16`, grid=[64,1,1], block=[256,1,1]. **Efficiency metrics N/A** — fused MoE ops have no direct performance model mapping (TFLOPS, BW all null); this is expected.

**Time Breakdown:**
- GPU kernel time: 0.28 ms
- CPU duration: 0.48 ms (CPU/GPU ratio: 1.71x — launch overhead ~0.20 ms)
- Kernel: `aiter_fused_moe_kernel_bf16`, grid=[64,1,1], block=[256,1,1]

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound |
|-----------|-------|-----------|---------------|------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 0.28 | 100.0% | N/A | N/A |

**Key Metrics:**
- **Shapes:** tokens=[16,4096], w1=[64,22016,4096], w2=[64,4096,11008], topk_weights=[16,2]
- **Config:** 64 experts, topk=2, hidden=4096, intermediate=11008, 16 tokens
- **dtype:** BFloat16
- **Expert-token pairs:** 16 × 2 = 32 across 64 experts
- **Approx. FLOPs:** tokens × topk × (2 × hidden × intermediate × 2) ≈ 16 × 2 × 4 × 4096 × 11008 ≈ 5.77 GFLOPs

**Expert Utilization & Load Balancing:**
- 32 expert-token pairs / 64 experts ≈ 0.5 tokens per expert on average
- With topk=2 and random routing, many experts receive 0 tokens; load imbalance is severe
- Routing implications: Sparse routing (topk=2) with many experts and few tokens maximizes idle experts

**Recommendations:**
- **Algorithmic:** Increase batch/sequence length to 256+ tokens for better expert utilization
- **Algorithmic:** Consider fewer experts (8–16) if small batches are inherent
- **Training:** Auxiliary load-balancing loss to encourage uniform routing

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 0.28 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |

No CPU/idle or multi-kernel bottlenecks. Single-kernel trace with full GPU utilization during kernel execution. Launch overhead (0.20 ms) is modest relative to 0.28 ms kernel time.

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
