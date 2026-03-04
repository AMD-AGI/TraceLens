# moe_04_top2_dense — MI300X Standalone Analysis

## Executive Summary

Standalone performance analysis of MoE trace `moe_04_top2_dense` on MI300X. The trace contains a single fused MoE operation with 16 experts, topk=2, hidden=7168, intermediate=14336, and 512 tokens. **Same model as moe_03** but with topk=2 — nearly 2× compute time (1.8 ms vs 0.95 ms). Denser routing improves expert utilization but doubles compute.

| Metric | Value |
|--------|-------|
| Total Compute Time | 1.8 ms |
| Computation | 100% |
| Idle Time | 0% |
| Exposed Communication | 0.00% |
| Top Bottleneck Category | MoE Fused (1.8 ms) |
| Approx. FLOPs | ~421 GFLOPs |
| Efficiency (TFLOPS/BW) | N/A (no perf model for fused MoE) |

---

## Compute Kernel Optimizations

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute |
|------|-----------|----------|-----------|--------------|
| 1 | vllm::rocm_aiter_fused_moe | MoE Fused | 1.8 | 100.0% |

### 🔴 P1: topk=2 Dense Routing — 2× Compute vs topk=1 (Expected)

**Issue**: 16 experts, topk=2, 512 tokens yields 1024 expert-token pairs across 16 experts (~64 tokens/expert on average). Denser routing (topk=2) doubles FLOPs vs topk=1 — kernel time 1.8 ms vs 0.95 ms in moe_03, as expected. Fused MoE kernels have no efficiency metrics (TFLOPS, BW all null); this is expected.

**Action**: (1) topk=2 improves expert utilization and model quality but costs 2× compute — trade-off is inherent. (2) If quality allows, consider topk=1 for 2× speedup (moe_03: 0.95 ms). (3) Verify load balance — 1024 pairs / 16 experts should distribute well; profile if concerned.

**Impact**: Algorithmic trade-off — topk=1 saves ~0.85 ms (47%) at potential quality cost. Kernel tuning not applicable (fused MoE lacks perf model).

→ *See Detailed Analysis: Compute Kernels > MoE Fused below*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation, 0% idle. CPU duration (2.0 ms) vs GPU kernel time (1.8 ms) indicates ~0.20 ms launch/sync overhead — acceptable.

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** SUCCESS

**Overview:**
Single `vllm::rocm_aiter_fused_moe` operation dominates 100% of GPU compute (1.8 ms). Kernel `aiter_fused_moe_kernel_bf16`, grid=[256,1,1], block=[256,1,1]. **Efficiency metrics N/A** — fused MoE ops have no direct performance model mapping; this is expected.

**Time Breakdown:**
- GPU kernel time: 1.8 ms
- CPU duration: 2.0 ms (CPU/GPU ratio: 1.11x — launch overhead ~0.20 ms)
- Kernel: `aiter_fused_moe_kernel_bf16`, grid=[256,1,1], block=[256,1,1]

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound |
|-----------|-------|-----------|---------------|------------|-------|
| vllm::rocm_aiter_fused_moe | 1 | 1.8 | 100.0% | N/A | N/A |

**Key Metrics:**
- **Shapes:** tokens=[512,7168], w1=[16,28672,7168], w2=[16,7168,14336], topk_weights=[512,2]
- **Config:** 16 experts, topk=2, hidden=7168, intermediate=14336, 512 tokens
- **dtype:** BFloat16
- **Expert-token pairs:** 512 × 2 = 1024 across 16 experts
- **Approx. FLOPs:** tokens × topk × (2 × hidden × intermediate × 2) ≈ 512 × 2 × 4 × 7168 × 14336 ≈ 421 GFLOPs

**Expert Utilization & Load Balancing:**
- 1024 expert-token pairs / 16 experts ≈ 64 tokens per expert on average
- topk=2 (denser routing) — better expert utilization than topk=1; each token uses 2 experts
- Load balance expected to be good with 64 tokens/expert; routing distribution may still cause variance

**Recommendations:**
- **Trade-off:** topk=2 vs topk=1 — 1.8 ms vs 0.95 ms; consider topk=1 if quality permits
- **Profiling:** Verify expert utilization; 64 tokens/expert should balance well
- **Comparison:** FLOPs scale linearly with topk (421 GFLOPs vs 210.5 GFLOPs for moe_03)

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|
| Reduce to topk=1 if quality allows | algorithmic | ~0.85 ms (47%) | medium |
| Load-balancing verification | algorithmic | N/A | low |

---

## Detailed Analysis: System-Level

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Compute Time | 1.8 ms |
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
