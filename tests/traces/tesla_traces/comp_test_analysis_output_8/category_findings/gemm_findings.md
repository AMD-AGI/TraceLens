<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GEMM Analysis Summary

## Overview

GEMMs account for **20.57%** of compute time (GPU kernel time), with **106.41 ms** total GEMM kernel time on trace 1 and **591** invocations across reported signatures. Comparative efficiency (**100 × trace2 kernel time / trace1 kernel time** per matched signature) is mostly in the **low 80s–low 90s percent** on the largest rows, so trace 2 spends less GPU time than trace 1 on those matches. One high-time shape is slightly above **100%** (trace 2 slower). Analysis uses **GPU kernel time** from precomputed category data (not CPU op duration).

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
| aten::mm | 130 | 12.499 | 11.75 | 91.15% | 1365.33 | compute-bound |
| aten::mm | 130 | 12.470 | 11.72 | 93.55% | 1365.33 | compute-bound |
| aten::addmm | 130 | 11.611 | 10.91 | 90.74% | 1365.39 | compute-bound |
| aten::addmm | 32 | 11.608 | 10.91 | 81.73% | 1820.47 | compute-bound |
| aten::mm | 32 | 11.502 | 10.81 | 86.64% | 1820.44 | compute-bound |
| aten::mm | 32 | 11.310 | 10.63 | 88.27% | 1820.44 | compute-bound |
| aten::mm | 32 | 11.152 | 10.48 | 89.50% | 1820.44 | compute-bound |
| aten::addmm | 32 | 10.612 | 9.97 | 92.58% | 1820.45 | compute-bound |
| aten::mm | 32 | 10.288 | 9.67 | 100.49% | 1820.44 | compute-bound |
| aten::mm | 2 | 1.232 | 1.16 | 66.52% | 1874.16 | compute-bound |
| aten::addmm | 2 | 1.186 | 1.11 | 71.82% | 1874.35 | compute-bound |
| aten::mm | 2 | 0.892 | 0.84 | 102.78% | 1874.16 | compute-bound |
| aten::mm | 1 | 0.017 | 0.02 | 52.35% | 31.51 | memory-bound |
| aten::mm | 1 | 0.016 | 0.01 | 55.88% | 31.51 | memory-bound |
| aten::addmm | 1 | 0.013 | 0.01 | 60.72% | 31.51 | memory-bound |

*Efficiency: comparative mode — **100 × (trace2 kernel time) / (trace1 kernel time)** for that signature. Below **100%**, trace 2 uses less kernel time; above **100%**, trace 2 uses more.*

## Key Bottlenecks

Bottlenecks use **GPU kernel time** on trace 1: rows with **> 5% of category time** (threshold **~5.32 ms** on **106.409 ms** total) or **> 100 ms** (none).

### 1. `aten::mm` — 4096×4096 (contiguous)

- **Time:** 12.499 ms (11.75% of GEMM category GPU time)
- **Efficiency:** 91.15% (trace 2 faster than trace 1)
- **Trace 1 context:** ~1430 TFLOPS/s achieved vs **1686 TFLOPS** peak multiply-accumulate (`matrix_bf16`, compute-bound)
- **Algorithmic:** Preserve large, contiguous tiles; avoid extra transposes that force a slower library path.
- **Kernel:** On the slower trace, tune toward the compute peak; trace 2 is already ahead on duration for this signature.

### 2. `aten::mm` — 4096×4096 (alternate layout / stride)

- **Time:** 12.470 ms (11.72%)
- **Efficiency:** 93.55% (trace 2 faster)
- **Trace 1 context:** ~1433 TFLOPS/s vs **1686 TFLOPS** peak (`matrix_bf16`)
- **Algorithmic:** Prefer layouts that select the higher-throughput kernel family when possible.
- **Kernel:** Align tiling and library selection for this stride case with the faster trace.

### 3. `aten::addmm` — 4096×4096 with bias

- **Time:** 11.611 ms (10.91%)
- **Efficiency:** 90.74% (trace 2 faster)
- **Trace 1 context:** ~1543 TFLOPS/s vs **1686 TFLOPS** peak (`matrix_bf16`)
- **Algorithmic:** Fuse bias with matmul where the stack allows; keep bias tensors layout-friendly.
- **Kernel:** Close headroom to peak on the slower trace; compare fused vs unfused epilogue paths.

### 4. `aten::addmm` — wide rectangular (4096 × 16384, bias)

- **Time:** 11.608 ms (10.91%)
- **Efficiency:** 81.73% (largest comparative gap among high-time rows — trace 2 much faster)
- **Trace 1 context:** ~1517 TFLOPS/s vs **1686 TFLOPS** peak (`matrix_bf16`)
- **Algorithmic:** Same as above for bias + wide K/N; avoid layout thrash between forward and backward.
- **Kernel:** Primary tuning target on trace 1 to approach trace 2 and the compute peak.

### 5. Rectangular `aten::mm` / `aten::addmm` — 4096 / 16384 family

- **Time:** 10.612–11.502 ms (9.67–10.81% each)
- **Efficiency:** 86.64–92.58% (trace 2 faster on each)
- **Trace 1 context:** ~1530–1660 TFLOPS/s vs **1686 TFLOPS** peak (`matrix_bf16`)
- **Algorithmic:** Stabilize operand layouts so forward and backward hit the same fast kernels.
- **Kernel:** Match library version and tuning between traces for these BF16 rectangular cases.

### 6. `aten::mm` — 4096×4096 with K = 16384 (fat inner dimension)

- **Time:** 10.288 ms (9.67%)
- **Efficiency:** 100.49% (trace 2 slightly **slower** — regression vs trace 1)
- **Trace 1 context:** ~1710 TFLOPS/s vs **1686 TFLOPS** peak (`matrix_bf16`)
- **Algorithmic:** Confirm matching dtype, accumulation, and epilogue across traces for this contraction.
- **Kernel:** Compare kernel selection and launch parameters on trace 2; investigate why trace 2 exceeds trace 1 despite strong trace 1 throughput.

### 7. Tail shapes — 22080×4096 (below 5% category time; kernel-tuning impact bands)

- **Time:** ~1.23 ms / ~1.19 ms (~1.1% each)
- **Efficiency:** 66.52% / 71.82% (trace 2 much faster; trace 1 has larger gap to peak)
- **Algorithmic:** Batch or pad odd leading dimensions if repeated to improve wave utilization.
- **Kernel:** Tune non-power-of-two M toward the compute peak on the slower trace.

## Additional Notes

- Missing perf models: **0**
- Quantized GEMMs: **0**
- Three **memory-bound** micro-GEMMs (M or K = 32) are negligible in GPU time; low arithmetic intensity vs **8.0 TB/s** peak HBM bandwidth reference in metadata.

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|-------------------------------|------------|
| Kernel tuning: `aten::addmm` wide rectangular 4096×16384 | kernel_tuning | 0–2.121 | 0–0.41% | medium |
| Kernel tuning: `aten::mm` 22080×4096 | kernel_tuning | 0.139–0.412 | 0.03–0.08% | medium |
| Kernel tuning: `aten::addmm` 22080×4096 | kernel_tuning | 0.05–0.334 | 0.01–0.06% | medium |
| Kernel tuning: `aten::mm` 4096×16384 rectangular | kernel_tuning | 0–1.537 | 0–0.30% | medium |
| Kernel tuning: `aten::mm` 4096×4096 contiguous | kernel_tuning | 0–1.106 | 0–0.21% | medium |
| Kernel tuning: `aten::mm` 4096×4096 alternate layout | kernel_tuning | 0–0.804 | 0–0.16% | medium |
| Kernel tuning: `aten::addmm` 4096×4096 bias | kernel_tuning | 0–1.075 | 0–0.21% | medium |
| Kernel tuning: `aten::mm` 16384×4096 | kernel_tuning | 0–1.327 | 0–0.26% | medium |
| Kernel tuning: `aten::mm` 4096×16384 (alternate layout) | kernel_tuning | 0–1.171 | 0–0.23% | medium |
| Kernel tuning: `aten::addmm` K=16384 fused path | kernel_tuning | 0–0.787 | 0–0.15% | medium |

*Ranges from `category_data/gemm_metrics.json` → `impact_estimates`, **`type: kernel_tuning` only** (`savings_ms_low`–`savings_ms_high`, `e2e_pct_low`–`e2e_pct_high`).*

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->

#### Comparative GEMM workload

**Identification:** Rows flagged by GEMM **GPU kernel duration** exceeding **5% of category time** or carrying material **kernel_tuning** impact bands in metrics, using comparative **efficiency_percent** as **100 × trace2 / trace1** kernel time (source: `gemm_metrics.json`).

**Data:**

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| aten::mm | 12.499 | 11.392 | 130/130 | −1.107 | 1365.33 | compute-bound |
| aten::mm | 12.470 | 11.666 | 130/130 | −0.804 | 1365.33 | compute-bound |
| aten::addmm | 11.611 | 10.535 | 130/130 | −1.076 | 1365.39 | compute-bound |
| aten::addmm | 11.608 | 9.487 | 32/32 | −2.121 | 1820.47 | compute-bound |
| aten::mm | 11.502 | 9.965 | 32/32 | −1.537 | 1820.44 | compute-bound |
| aten::mm | 11.310 | 9.984 | 32/32 | −1.326 | 1820.44 | compute-bound |
| aten::mm | 11.152 | 9.982 | 32/32 | −1.170 | 1820.44 | compute-bound |
| aten::addmm | 10.612 | 9.825 | 32/32 | −0.787 | 1820.45 | compute-bound |
| aten::mm | 10.288 | 10.339 | 32/32 | +0.051 | 1820.44 | compute-bound |
| aten::mm | 1.232 | 0.820 | 2/2 | −0.412 | 1874.16 | compute-bound |
| aten::addmm | 1.186 | 0.852 | 2/2 | −0.334 | 1874.35 | compute-bound |
| aten::mm | 0.892 | 0.917 | 2/2 | +0.025 | 1874.16 | compute-bound |

*Trace 2 times = trace 1 × (`efficiency_percent` / 100). Difference = trace 2 − trace 1.*

**Reasoning:** Most high-time BF16 GEMMs show trace 2 kernel time below trace 1 (efficiency in the low 80s–low 90s percent), so trace 1 dominates matched GPU time on those signatures. One large compute-bound case near 4096×4096 with K = 16384 is ~**100.5%**, so trace 2 is slightly slower despite high achieved TFLOPS/s on trace 1. Smaller **22080×4096** cases combine lower achieved TFLOPS/s versus the **1686 TFLOPS** `matrix_bf16` peak with a strong trace 2 advantage (mid-60s to low-70s percent efficiency).

**Resolution:** Prioritize aligning trace 1 with trace 2 on the wide **`aten::addmm`** and rectangular **4096×16384** paths where the comparative gap is largest. For the ~**100.5%** row, treat trace 2 as the regression: verify dtype, epilogue, and kernel selection. Use GEMM tuning or library upgrades on compute-bound rows.

**Impact estimate:** Rollup from `impact_estimates` (`kernel_tuning` only) — see Impact Summary table; persisted to metadata via `write_impact_estimates`.

### Compute Kernel Insights

_(Orchestrator may merge this section into the final report.)_

### System-Level Insights

_(None for this compute-tier category.)_
