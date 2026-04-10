<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Convolution Analysis Summary

**Status:** SUCCESS

## Overview

Convolution accounts for **18.71%** of GPU compute kernel time (**96.795 ms** category GPU kernel time, **1472** op invocations summed across grouped CSV rows). All bottleneck analysis uses **GPU kernel time** on trace 1, not CPU op duration. This run is **comparative**: **efficiency** is **100 × (trace2 kernel time) ÷ (trace1 kernel time)** as a percentage. Values **below 100%** mean trace 2 spent less kernel time than trace 1 for that matched grouping. Values **above 100%** mean trace 2 spent more kernel time than trace 1 (regression on trace 2 for that row). Roofline **bound** on trace 1 uses peaks **1686 TFLOP/s** (`matrix_bf16`) and **8.0 TB/s** high-bandwidth memory from metadata. `convolution_metrics.json` reports **transpose_overhead_percent** **0.0%** — no material layout-transpose overhead in this slice.

## Operations Breakdown

*Trace 1 kernel time; efficiency is trace2/trace1 as a percentage; type is roofline bound on trace 1.*

| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|------:|----------:|--------------:|-----------:|-----------:|:-----|
| `aten::convolution_backward` | 152 | 14.801 | 15.29 | 39.56% (T2/T1) | 430.06 | compute-bound |
| `aten::convolution_backward` | 72 | 11.150 | 11.52 | 55.34% (T2/T1) | 246.22 | compute-bound |
| `aten::convolution_backward` | 56 | 6.571 | 6.79 | 42.98% (T2/T1) | 222.84 | compute-bound |
| `aten::convolution_backward` | 24 | 5.837 | 6.03 | 105.03% (T2/T1) | 250.53 | compute-bound |
| `aten::convolution_backward` | 8 | 4.724 | 4.88 | 252.98% (T2/T1) | 100.85 | memory-bound |
| `aten::convolution` | 152 | 4.468 | 4.62 | 50.80% (T2/T1) | 430.06 | compute-bound |
| `aten::convolution_backward` | 8 | 3.540 | 3.66 | 103.48% (T2/T1) | 251.63 | compute-bound |
| `aten::convolution_backward` | 24 | 2.859 | 2.95 | 67.92% (T2/T1) | 111.93 | memory-bound |
| `aten::convolution_backward` | 8 | 2.496 | 2.58 | 213.45% (T2/T1) | 101.13 | memory-bound |
| `aten::convolution` | 72 | 2.484 | 2.57 | 107.03% (T2/T1) | 246.22 | compute-bound |
| `aten::convolution_backward` | 8 | 1.993 | 2.06 | 53.46% (T2/T1) | 907.04 | compute-bound |
| `aten::convolution_backward` | 72 | 1.964 | 2.03 | 39.02% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution` | 56 | 1.789 | 1.85 | 63.55% (T2/T1) | 222.84 | compute-bound |
| `aten::convolution` | 8 | 1.659 | 1.71 | 71.54% (T2/T1) | 100.85 | memory-bound |
| `aten::convolution_backward` | 72 | 1.642 | 1.70 | 50.53% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution` | 24 | 1.562 | 1.61 | 95.68% (T2/T1) | 250.53 | compute-bound |
| `aten::convolution_backward` | 8 | 1.497 | 1.55 | 57.59% (T2/T1) | 149.20 | memory-bound |
| `aten::convolution_backward` | 8 | 1.471 | 1.52 | 161.57% (T2/T1) | 99.86 | memory-bound |
| `aten::convolution_backward` | 8 | 1.440 | 1.49 | 42.75% (T2/T1) | 296.60 | compute-bound |
| `aten::convolution_backward` | 8 | 1.399 | 1.45 | 53.92% (T2/T1) | 586.43 | compute-bound |
| `aten::convolution_backward` | 8 | 1.295 | 1.34 | 74.89% (T2/T1) | 28.00 | memory-bound |
| `aten::convolution_backward` | 8 | 1.194 | 1.23 | 218.19% (T2/T1) | 19.64 | memory-bound |
| `aten::convolution_backward` | 8 | 1.047 | 1.08 | 23.87% (T2/T1) | 293.41 | compute-bound |
| `aten::convolution_backward` | 8 | 1.027 | 1.06 | 173.41% (T2/T1) | 1.71 | memory-bound |
| `aten::convolution_backward` | 8 | 1.012 | 1.05 | 1296.22% (T2/T1) | 99.35 | memory-bound |
| `aten::convolution` | 8 | 1.004 | 1.04 | 94.56% (T2/T1) | 251.63 | compute-bound |
| `aten::convolution` | 24 | 0.973 | 1.00 | 80.94% (T2/T1) | 111.93 | memory-bound |
| `aten::convolution_backward` | 8 | 0.920 | 0.95 | 41.41% (T2/T1) | 738.34 | compute-bound |
| `aten::convolution_backward` | 8 | 0.868 | 0.90 | 52.80% (T2/T1) | 74.88 | memory-bound |
| `aten::convolution_backward` | 8 | 0.774 | 0.80 | 39.37% (T2/T1) | 147.29 | memory-bound |
| `aten::convolution` | 72 | 0.746 | 0.77 | 71.17% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution` | 8 | 0.740 | 0.76 | 69.86% (T2/T1) | 101.13 | memory-bound |
| `aten::convolution_backward` | 8 | 0.616 | 0.64 | 98.86% (T2/T1) | 20.37 | memory-bound |
| `aten::convolution` | 72 | 0.584 | 0.60 | 67.64% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution_backward` | 24 | 0.558 | 0.58 | 45.74% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 8 | 0.518 | 0.54 | 88.41% (T2/T1) | 149.20 | memory-bound |
| `aten::convolution_backward` | 24 | 0.492 | 0.51 | 51.06% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 8 | 0.485 | 0.50 | 147.89% (T2/T1) | 28.00 | memory-bound |
| `aten::convolution` | 8 | 0.458 | 0.47 | 60.95% (T2/T1) | 907.04 | compute-bound |
| `aten::convolution` | 8 | 0.447 | 0.46 | 51.28% (T2/T1) | 586.43 | compute-bound |
| `aten::convolution` | 8 | 0.415 | 0.43 | 67.62% (T2/T1) | 296.60 | compute-bound |
| `aten::convolution` | 8 | 0.374 | 0.39 | 272.18% (T2/T1) | 1.71 | memory-bound |
| `aten::convolution` | 8 | 0.335 | 0.35 | 80.36% (T2/T1) | 99.86 | memory-bound |
| `aten::convolution` | 8 | 0.333 | 0.34 | 63.57% (T2/T1) | 738.34 | compute-bound |
| `aten::convolution_backward` | 8 | 0.278 | 0.29 | 32.97% (T2/T1) | 1.91 | memory-bound |
| `aten::convolution` | 8 | 0.276 | 0.28 | 71.38% (T2/T1) | 99.35 | memory-bound |
| `aten::convolution` | 8 | 0.261 | 0.27 | 261.84% (T2/T1) | 19.64 | memory-bound |
| `aten::convolution` | 24 | 0.226 | 0.23 | 57.52% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution_backward` | 8 | 0.210 | 0.22 | 40.76% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 24 | 0.209 | 0.22 | 58.20% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution_backward` | 8 | 0.196 | 0.20 | 43.62% (T2/T1) | 1.92 | memory-bound |
| `aten::convolution_backward` | 8 | 0.195 | 0.20 | 50.52% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution_backward` | 8 | 0.191 | 0.20 | 45.49% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution_backward` | 8 | 0.179 | 0.18 | 50.94% (T2/T1) | 1.92 | memory-bound |
| `aten::convolution` | 8 | 0.178 | 0.18 | 60.64% (T2/T1) | 293.41 | compute-bound |
| `aten::convolution_backward` | 8 | 0.178 | 0.18 | 46.47% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 8 | 0.172 | 0.18 | 88.54% (T2/T1) | 74.88 | memory-bound |
| `aten::convolution` | 8 | 0.159 | 0.16 | 65.57% (T2/T1) | 147.29 | memory-bound |
| `aten::convolution_backward` | 8 | 0.149 | 0.15 | 48.51% (T2/T1) | 1.59 | memory-bound |
| `aten::convolution` | 8 | 0.146 | 0.15 | 143.13% (T2/T1) | 20.37 | memory-bound |
| `aten::convolution_backward` | 8 | 0.145 | 0.15 | 53.59% (T2/T1) | 1.91 | memory-bound |
| `aten::convolution_backward` | 8 | 0.142 | 0.15 | 49.69% (T2/T1) | 1.59 | memory-bound |
| `aten::convolution` | 8 | 0.114 | 0.12 | 54.37% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution` | 8 | 0.083 | 0.09 | 67.57% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 8 | 0.077 | 0.08 | 78.93% (T2/T1) | 1.98 | memory-bound |
| `aten::convolution` | 8 | 0.076 | 0.08 | 44.65% (T2/T1) | 1.92 | memory-bound |
| `aten::convolution` | 8 | 0.070 | 0.07 | 56.68% (T2/T1) | 1.92 | memory-bound |
| `aten::convolution` | 8 | 0.068 | 0.07 | 60.26% (T2/T1) | 1.96 | memory-bound |
| `aten::convolution` | 8 | 0.065 | 0.07 | 50.37% (T2/T1) | 1.91 | memory-bound |
| `aten::convolution` | 8 | 0.065 | 0.07 | 49.60% (T2/T1) | 1.59 | memory-bound |
| `aten::convolution` | 8 | 0.064 | 0.07 | 62.31% (T2/T1) | 1.91 | memory-bound |
| `aten::convolution` | 8 | 0.061 | 0.06 | 52.48% (T2/T1) | 1.59 | memory-bound |

## Key Bottlenecks

1. **Time share (>5% of category GPU kernel time; none exceed 100 ms per row)** — The four largest groupings are all `aten::convolution_backward` at **~15.3%**, **~11.5%**, **~6.8%**, and **~6.0%** of category time (**14.8 ms**, **11.2 ms**, **6.6 ms**, **5.8 ms** on trace 1). They dominate convolution kernel time despite sub-100 ms per-row durations.
2. **Large comparative gap (T2/T1 well below 100%)** — The top three backward rows show **~40%**, **~55%**, and **~43%** comparative efficiency: trace 2 uses substantially less kernel time for the matched work. Improving trace 1 toward trace 2’s kernel time is the highest-leverage tuning story.
3. **Comparative regression (T2/T1 above 100%)** — Examples include rows at **~105%**, **~253%**, **~1296%**, and other **>100%** lines: trace 2 kernel time exceeds trace 1 on those matched groupings. Treat these as alignment, fusion, or structural-diff effects until validated.
4. **Forward path** — `aten::convolution` at **~4.6%** of category (**4.47 ms**) is **compute-bound** with **~50.8%** comparative ratio; secondary to backward but still material.

## Impact Summary

*Only `kernel_tuning` rows from pre-computed `convolution_metrics.json` → `impact_estimates` (75–100% gap-to-trace2 band).*

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|------------------------------|------------|
| `aten::convolution_backward` | kernel_tuning | 6.994–8.946 | 1.35–1.73% | high |
| `aten::convolution_backward` | kernel_tuning | 2.923–4.980 | 0.57–0.96% | high |
| `aten::convolution_backward` | kernel_tuning | 2.805–3.747 | 0.54–0.72% | high |
| `aten::convolution` | kernel_tuning | 1.442–2.198 | 0.28–0.43% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.942–1.198 | 0.18–0.23% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.572–0.928 | 0.11–0.18% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.714–0.797 | 0.14–0.15% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.619–0.824 | 0.12–0.16% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.536–0.812 | 0.10–0.16% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.270–0.917 | 0.05–0.18% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.393–0.645 | 0.08–0.12% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.348–0.635 | 0.07–0.12% | medium |
| `aten::convolution` | kernel_tuning | 0.273–0.652 | 0.05–0.13% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.412–0.539 | 0.08–0.10% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.368–0.469 | 0.07–0.09% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.257–0.410 | 0.05–0.08% | medium |
| `aten::convolution` | kernel_tuning | 0.077–0.472 | 0.01–0.09% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.218–0.303 | 0.04–0.06% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.157–0.241 | 0.03–0.05% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.002–0.325 | 0.00–0.06% | medium |
| `aten::convolution` | kernel_tuning | 0.141–0.218 | 0.03–0.04% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.156–0.186 | 0.03–0.04% | medium |
| `aten::convolution` | kernel_tuning | 0.051–0.223 | 0.01–0.04% | medium |
| `aten::convolution` | kernel_tuning | 0.038–0.215 | 0.01–0.04% | medium |
| `aten::convolution` | kernel_tuning | 0.086–0.179 | 0.02–0.03% | medium |
| `aten::convolution` | kernel_tuning | 0.057–0.189 | 0.01–0.04% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.096–0.124 | 0.02–0.02% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.082–0.111 | 0.02–0.02% | medium |
| `aten::convolution` | kernel_tuning | 0.041–0.134 | 0.01–0.03% | medium |
| `aten::convolution_backward` | kernel_tuning | 0.075–0.104 | 0.01–0.02% | medium |
| `aten::convolution` | kernel_tuning | 0.051–0.121 | 0.01–0.02% | medium |
| `aten::convolution` | kernel_tuning | 0.000–0.185 | 0.00–0.04% | medium |

*Aggregate band (sum of pre-computed rows): **~21.196–32.027 ms** potential kernel-time savings on trace 1 if trace 2-level kernel time were reached on modeled rows, **~4.10–6.18%** of trace GPU time (E2E % columns summed as emitted in metrics).*

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->

#### Convolution backward and forward — large gap to trace 2, mixed regressions

**Identification:** Groupings were prioritized by trace 1 **GPU kernel time** share of the convolution category, by comparative **efficiency** (trace 2 versus trace 1 kernel time), and by **bound type** on trace 1 for tuning focus. (source: `convolution_metrics.json` → `operations[].time_ms`, `operations[].percent_of_category`, `operations[].efficiency.efficiency_percent`, `operations[].efficiency.bound_type`, `operations[].count`)

**Data:**

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|------------------:|------------------:|---------------|----------------:|:-----------|
| `aten::convolution_backward` | 14.801 | 5.855 | 152/152 | 430.06 | compute |
| `aten::convolution_backward` | 11.150 | 6.171 | 72/72 | 246.22 | compute |
| `aten::convolution_backward` | 6.571 | 2.825 | 56/56 | 222.84 | compute |
| `aten::convolution_backward` | 5.837 | 6.131 | 24/24 | 250.53 | compute |
| `aten::convolution_backward` | 4.724 | 11.952 | 8/8 | 100.85 | memory |
| `aten::convolution` | 4.468 | 2.270 | 152/152 | 430.06 | compute |
| `aten::convolution_backward` | 3.540 | 3.663 | 8/8 | 251.63 | compute |
| `aten::convolution_backward` | 2.859 | 1.941 | 24/24 | 111.93 | memory |
| `aten::convolution_backward` | 2.496 | 5.328 | 8/8 | 101.13 | memory |
| `aten::convolution` | 2.484 | 2.659 | 72/72 | 246.22 | compute |
| `aten::convolution_backward` | 1.993 | 1.065 | 8/24 | 907.04 | compute |
| `aten::convolution_backward` | 1.964 | 0.766 | 72/216 | 1.98 | memory |

*Trace 2 times derived as trace 1 GPU kernel time × (efficiency_percent / 100).*

**Reasoning for slowdown:** The heaviest **backward** rows concentrate category time on **compute-bound** shapes with high FLOPS/byte on trace 1, yet comparative ratios near **40–55%** show trace 2 completes the matched work with much less GPU kernel time. Several **memory-bound** micro-shapes appear with very low FLOPS/byte and uneven comparative ratios. Multiple rows exceed **100%** comparative efficiency, so trace 2 kernel time is **higher** than trace 1 for those groupings; those lines need validation before optimization.

**Resolution:** Align tensor **memory layout** (for example channels-last where it matches the deep learning framework’s preferred format) to reduce layout friction, and tune **backward** kernel selection and algorithmic paths so trace 1 approaches trace 2’s observed kernel time on the large buckets. For **memory-bound** rows, improve access regularity and fusion opportunities. For comparative **regressions** on trace 2, diff framework and library versions, epilogue or fusion, and structural-diff row matching.

**Impact estimate:**

- Low end (75% gap target): 21.196 ms savings (4.10% E2E)
- High end (100% gap target): 32.027 ms savings (6.18% E2E)
- Range: 21.196–32.027 ms (4.10–6.18% E2E)

### Compute Kernel Insights

- **Backward `aten::convolution_backward`** dominates trace 1 convolution kernel time; **forward `aten::convolution`** is smaller but still visible in the top dozen rows.
- **Transpose overhead** is negligible in category aggregates (`transpose_overhead_percent` **0%**); bottlenecks here are kernel-time gaps and bound-type mix, not explicit transpose rows in the CSV.
- Treat **comparative ratios above 100%** as trace 2 slowdowns or alignment artifacts until reconciled with call counts and kernel fusion.

### System-Level Insights

- Convolution metadata lists **has_sync_bottleneck: false** and points to **gpu_kernel_time_ms** for prioritization; no extra host-sync story is asserted from this category slice alone. See **system_findings** for cross-category effects.
