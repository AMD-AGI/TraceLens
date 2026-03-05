# moe_05_large_capacity - MI300X Standalone Analysis

## Executive Summary

This trace contains a single `vllm::rocm_aiter_fused_moe` operation processing 4096 tokens through an 8-expert MoE layer with topk=2 routing, hidden dimension 8192, and intermediate dimension 22016 in BFloat16. GPU utilization is 100% computation with no idle time. The fused MoE kernel runs for 12.0 ms, making this the most compute-intensive MoE trace in the set. Approximate FLOPs: 4096 × 2 × (2 × 8192 × 22016 × 2) ≈ 5.91 TFLOPs. No efficiency metrics are available for the fused kernel.

| Metric | Value |
|--------|-------|
| Total Compute Time | 12.0 ms |
| Computation | 100.0% |
| Idle Time | 0.0% |
| Exposed Communication | 0.0% |
| Exposed MemCpy | 0.0% |
| Top Bottleneck Category | MoE Fused (100%) |
| Approx. FLOPs | 5.91 TFLOPs |

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis focused on individual kernel efficiency.

### Top Operations

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | vllm::rocm_aiter_fused_moe (4096 tokens, 8 experts, topk=2, hidden=8192) | MoE Fused | 12.0 | 100.0% |

### 🟡 P2: Large MoE Layer — Verify Expert Load Balance at Scale

**Issue**: `vllm::rocm_aiter_fused_moe` processes 4096 tokens with topk=2 across 8 experts, yielding ~1024 expert-token pairs per expert on average. With large dimensions (hidden=8192, intermediate=22016), the kernel processes approximately 5.91 TFLOPs. At 12.0 ms, this implies ~493 effective TFLOPS — 69.6% of peak BF16 MAF (708 TFLOPS). While no direct efficiency metric is available for fused MoE, this throughput estimate suggests reasonable but not peak utilization.

**Action**: Profile with hardware counters to verify whether the bottleneck is compute-bound (matrix units) or memory-bound (weight loading). With 8 experts and topk=2, each expert's w1 and w2 weights total 2 × 8192 × 22016 × 2 bytes ≈ 689 MB per expert, totaling ~5.5 GB for all experts — potentially causing HBM bandwidth pressure.

**Impact**: Not quantifiable from trace data.

→ *See [Detailed Analysis: Compute Kernels > MoE Fused](#1-moe-fused-100-of-compute) for details*

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

✅ No system-level bottlenecks detected. GPU activity breakdown shows 100% computation with negligible CPU overhead (CPU dur: 12.2 ms vs GPU: 12.0 ms). See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

---

## Detailed Analysis: Compute Kernels

### 1. MoE Fused (100% of compute)

**Status:** OK — No efficiency model available for fused MoE kernels

**Overview:**
MoE Fused operations account for 100% of GPU compute time (12.0 ms). A single `vllm::rocm_aiter_fused_moe` invocation handles the full gating, expert dispatch, and combine. No direct TFLOPS or bandwidth efficiency metrics are available for this fused kernel.

**Time Breakdown:**
- GPU kernel time: 12.0 ms
- CPU duration: 12.2 ms (CPU/GPU ratio: 1.02x — no launch overhead bottleneck)
- Sync time: 0 ms

**Operations Breakdown:**

| Operation | Count | Time (ms) | % of Category | Efficiency | Bound Type |
|-----------|-------|-----------|---------------|------------|------------|
| vllm::rocm_aiter_fused_moe | 1 | 12.0 | 100.0% | N/A (fused) | N/A |

**MoE Configuration:**
- **Tokens:** 4096
- **Experts:** 8
- **Top-k:** 2
- **Hidden dimension:** 8192
- **Intermediate dimension:** 22016
- **Precision:** BFloat16
- **Kernel:** `aiter_fused_moe_kernel_bf16`, grid=[1024,1,1], block=[256,1,1]

**Expert Utilization Analysis:**
- Total expert-token pairs: 4096 × 2 = 8192
- Average tokens per expert: 8192 / 8 = 1024
- With 1024 tokens per expert and large intermediate (22016), each expert performs substantial compute
- Load balance: With only 8 experts and high token count, statistical balancing is favorable

**Approximate Throughput Estimate:**
- Approx. FLOPs: 4096 × 2 × (2 × 8192 × 22016 × 2) ≈ 5.91 TFLOPs
- Implied throughput: 5.91 TFLOPs / 12.0 ms ≈ 493 TFLOPS
- vs peak BF16 MAF: 493 / 708 = 69.6% — reasonable utilization estimate
- Note: This is an approximation; actual FLOPs depend on gating, activation functions, and fused operations within the kernel

**Recommendations:**
- Profile with hardware counters to determine if compute-bound or memory-bound
- With 8 experts × (22016 × 8192 + 8192 × 22016) × 2 bytes ≈ 5.5 GB total weight data, verify HBM bandwidth is not the bottleneck
- Consider FP8 quantization for expert weights to reduce memory traffic and potentially improve throughput (peak FP8: 1273 TFLOPS)

**Impact Summary:**

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|------------------------|------------|

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Total Time | 12.0 ms |
| Computation | 100.0% |
| Idle Time | 0.0% |
| Exposed Communication | 0.0% |
| Exposed MemCpy | 0.0% |

CPU/Idle and Multi-Kernel analyses were not invoked as idle time (0.0%) is below the 15% threshold and no memcpy/NCCL events were detected in the trace.

---

## Appendix

### Hardware Reference
- **Platform**: MI300X
- **Peak HBM BW**: 5.3 TB/s
- **Peak MAF (FP16)**: 654 TFLOPS
- **Peak MAF (BF16)**: 708 TFLOPS
- **Peak MAF (FP32)**: 163 TFLOPS
- **Peak MAF (FP8)**: 1273 TFLOPS
- **Memory**: 192 GB HBM3
