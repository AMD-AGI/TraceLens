# moe_02_few_experts_many_tokens — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of MoE trace `moe_02_few_experts_many_tokens` on MI300X. The trace contains a single fused MoE operation with 8 experts, topk=1, hidden=4096, intermediate=11008, and 8192 tokens. **Good load balance expected**: each expert processes ~1024 tokens on average.

| Metric | Value |
|--------|-------|
| Total Compute Time | 6.5 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | MoE Fused (6.5 ms) |
| Approx. FLOPs | ~1.48 TFLOPs |
| Efficiency (TFLOPS/BW) | N/A (no perf model for fused MoE) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute |
|------|-----------|----------|-----------|--------------|
| 1 | vllm::rocm_aiter_fused_moe | MoE Fused | 6.5 | 100.0% |

### 🔴 P1: MoE Fused — Focus on Algorithmic Tuning (No Kernel Efficiency Metrics)

**Issue**: MoE Fused operations consume 6.5 ms (100% of compute). Fused MoE kernels have no efficiency metrics (TFLOPS, BW all null) — this is expected for fused MoE. With 8 experts, topk=1, and 8192 tokens, load balance is favorable (~1024 tokens/expert on average).

**Action**: (1) Verify token distribution across experts at runtime — if routing is skewed, consider auxiliary load-balancing loss. (2) Profile expert utilization; if balanced, kernel is likely well-utilized. (3) For further speedup, consider FP8 quantization if model supports it (MI300X peak FP8: 1273 TFLOPS vs BF16: 708 TFLOPS).

**Impact**: Algorithmic tuning (load balancing, FP8) may yield gains. Kernel-level efficiency cannot be quantified without perf model.

→ *See Detailed Analysis: Compute Kernels > MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation, 0% idle. CPU duration (6.7 ms) closely matches GPU kernel time (6.5 ms), indicating minimal launch/sync overhead (~0.2 ms, 3% of total).

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `vllm::rocm_aiter_fused_moe` operation dominates 100% of GPU compute (6.5 ms). Kernel `aiter_fused_moe_kernel_bf16`, grid=[512,1,1], block=[256,1,1]. **Efficiency metrics N/A** — fused MoE ops have no direct performance model mapping; this is expected.

**Time Breakdown:**
- GPU kernel time: 6.5 ms
- CPU duration: 6.7 ms (CPU/GPU ratio: 1.03x — minimal sync overhead)
- Kernel: `aiter_fused_moe_kernel_bf16`, grid=[512,1,1], block=[256,1,1]

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound |
|-----------|-------|-----------|---------------|------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 6.5 | 100.0% | N/A | N/A |

**Key Metrics:**
- **Shapes:** tokens=[8192,4096], w1=[8,22016,4096], w2=[8,4096,11008], topk_weights=[8192,1]
- **Config:** 8 experts, topk=1, hidden=4096, intermediate=11008, 8192 tokens
- **dtype:** BFloat16
- **Expert-token pairs:** 8192 × 1 = 8192 across 8 experts
- **Approx. FLOPs:** tokens × topk × (2 × hidden × intermediate × 2) ≈ 8192 × 1 × 4 × 4096 × 11008 ≈ 1.48 TFLOPs

**Expert Utilization & Load Balancing:**
- 8192 expert-token pairs / 8 experts ≈ 1024 tokens per expert on average
- topk=1 (sparse routing) with few experts and many tokens yields good load balance
- Each expert processes a large, contiguous workload — favorable for GPU utilization

**Recommendations:**
- **Algorithmic:** Verify routing distribution; if skewed, apply load-balancing loss
- **Quantization:** Consider FP8 for 1.8× theoretical peak uplift (if supported)
- **Profiling:** Confirm expert utilization at application level

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|
| Load-balancing verification | algorithmic | N/A | low |
| FP8 quantization (if supported) | algorithmic | N/A | medium |

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 6.5 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Exposed MemCpy | 0.00% |

No CPU/idle or multi-kernel bottlenecks. Single-kernel trace with full GPU utilization. Launch overhead (~0.2 ms) is negligible (3% of total).

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
