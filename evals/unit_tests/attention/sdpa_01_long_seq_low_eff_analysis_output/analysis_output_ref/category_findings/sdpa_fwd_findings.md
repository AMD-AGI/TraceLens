# SDPA Forward Analysis Findings

**Status:** SUCCESS
**Platform:** MI300X | Peak BF16 MAF: 708 TFLOPS | Peak HBM BW: 5.3 TB/s
**Attention Implementation:** Flash Attention (all operations)
**Total GPU Kernel Time:** 658.46 ms | **Operation Count:** 10
**Percent of Compute:** 100.0%
**Average Efficiency:** 40.44% of peak MAF

---

## Attention Implementation Details

All 10 SDPA forward operations use **Flash Attention** via the `aten::_scaled_dot_product_flash_attention` backend, dispatching to the **attn_fwd.kd** kernel. No Paged Attention or unfused attention patterns detected. All operations share the same shape and use **MHA** (Multi-Head Attention) with H_Q = H_KV = 32 heads and d_head = 128.

---

## Operation Summary

| # | Operation | Shape (B, H, N_Q, N_KV, d) | Count | GPU Time (ms) | % of Category | TFLOPS Achieved | Efficiency (% of 708 peak) | Bound | FLOPS/Byte |
|---|-----------|---------------------------|-------|---------------|--------------|-----------------|---------------------------|-------|------------|
| 1 | aten::_scaled_dot_product_flash_attention | (1, 32, 32768, 32768, 128) | 10 | 658.46 | 100.0% | 286.31 | 40.44% | compute | 16384.0 |

---

## Bottleneck Identification

Bottleneck criteria: GPU time > 100 ms OR > 5% of category time, AND efficiency < 70% of peak.

### Bottleneck: Flash Attention at N=32768 — CRITICAL

- **GPU Time:** 658.46 ms (100% of SDPA category) — exceeds 100 ms threshold
- **Efficiency:** 40.44% of 708 TFLOPS peak (286.31 TFLOPS achieved)
- **Kernel:** attn_fwd.kd × 10 invocations, mean 65.8 ms, median 56.6 ms (σ = 21.8 ms)
- **Shape:** B=1, H=32, N_Q=N_KV=32768, d=128, non-causal, bfloat16
- **Efficiency Context:** For N=32768 (> 4096), expected efficiency is 50–70%. Achieved 40.44% — **below expected range**, indicating a kernel optimization opportunity of ~10–30% headroom.
- **Root Cause:** The workload is strongly compute-bound (FLOPS/Byte = 16384) and has favorable dimensions (long sequence, 32 heads). The kernel is not fully exploiting MI300X compute capacity for this configuration. Possible causes include suboptimal tile sizes, wave occupancy, or memory access patterns for the (B=1, H=32, N=32768, d=128) shape.

---

## Kernel Breakdown

| Kernel | Stream | Invocations | Mean (µs) | Median (µs) | Min (µs) | Max (µs) | σ (µs) | CV |
|--------|-------|-------------|-----------|-------------|----------|----------|--------|-----|
| attn_fwd.kd | 1 | 10 | 65846 | 56553 | 54177 | 125690 | 21797 | 33% |

**Observations:**
- **High invocation variance:** Coefficient of variation (CV) ≈ 33%. The max duration (125.7 ms) is 2.3× the min (54.2 ms).
- **Warm-up outlier:** Median (56.6 ms) is close to min, while mean (65.8 ms) is pulled up by the max. This suggests the first invocation (or an early one) may be a warm-up outlier (~125 ms) with subsequent invocations clustering around 54–57 ms.
- **Single kernel:** All SDPA work maps to one kernel type; no multi-kernel composition.

---

## Workload Profile

| Property | Value |
|----------|-------|
| Attention Pattern | MHA (Multi-Head Attention) |
| Head Count (Q/KV) | 32 / 32 |
| Head Dimension | 128 |
| Batch Size | 1 |
| Sequence Length | 32768 |
| Causal Masking | No |
| Dropout | 0.0 |
| Flash Implementation | Yes |
| Paged Attention | No |
| Dtype | BFloat16 |
| Approx FLOPs/call | ~17.6 TFLOP |
| Approx Data Moved | ~805 MB |

**Workload Characterization:**
- Long-sequence self-attention with high arithmetic intensity (FLOPS/Byte = 16384).
- Small batch (B=1) limits batch-level parallelism; 32 heads provide head-level parallelism.
- GPU utilization is 99.99% computation with negligible idle time.

---

## Anomaly Check

No efficiency values exceed 100%. No anomalies detected.

---

## Recommendations

### 1. Kernel Tuning for attn_fwd at N=32768 (kernel_tuning)

**Problem:** The attn_fwd kernel achieves 40.44% efficiency for a long-sequence (N=32768) workload where 50–70% is expected. The configuration (B=1, H=32, N=32768, d=128) is compute-bound and should be well-suited to high utilization.

**Recommendation:** Profile the attn_fwd kernel for this shape and tune tile sizes, block dimensions, and wave occupancy. Generate a replay artifact for targeted kernel optimization. The ~10–30% efficiency gap suggests room for tile-size and occupancy improvements.

**Estimated Savings:** 392.179 ms (from impact_estimates)

### 2. Investigate Invocation Variance (kernel_tuning)

**Problem:** Kernel duration varies 2.3× (54 ms to 126 ms) with 33% CV. The first or early invocation may be a warm-up outlier.

**Recommendation:** If the trace includes cold-start or first-call behavior, consider separating warm-up from steady-state analysis. For production workloads, ensure kernels are warmed before latency-sensitive measurements.

### 3. Batch Size Scaling (algorithmic)

**Problem:** B=1 limits batch-level parallelism. With 32 heads, the kernel has head parallelism but no batch parallelism.

**Recommendation:** If the use case supports batching, increasing batch size would improve GPU utilization and amortize per-invocation overhead across more work.

**Estimated Savings:** Model-dependent; potentially 10–25% improvement.
**Confidence:** low

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Kernel tuning for attn_fwd (N=32768, B=1, H=32) | kernel_tuning | 392.179 | high |
| Investigate invocation variance | kernel_tuning | — | medium |
| Batch size scaling | algorithmic | model-dependent | low |
