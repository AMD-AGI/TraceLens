<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Elementwise Analysis Findings

**Status:** OK

**Comparison scope:** comparative

**Platform:** MI355X (from `elementwise_metadata.json`)

## Overview

| Metric | Value |
|--------|------:|
| Category GPU kernel time | 82.23 ms |
| Share of compute | 15.9% |
| Logical operation count | 6541 |

Elementwise work mixes **in-place adds** (`aten::add_`), **activation-related backward** ops (`aten::threshold_backward`, `aten::sigmoid_backward`), **casts and copies** (`aten::copy_` across data types), **arithmetic** (`aten::clamp_min_`, `aten::mul`, `aten::div`, `aten::add`), and smaller **utility** rows (`aten::fill_`, `aten::native_dropout_backward`, loss-related elementwise). Bottleneck ordering uses **GPU kernel time** from `elementwise_metrics.json` → `total_time_ms` and per-row `operations[].time_ms`, not CPU-side duration. Peak device memory bandwidth reference for simple memory-bound vector work is **8.0 TB/s** (`elementwise_metrics.json` → `category_specific.peak_hbm_bw_tbs`).

## Semantic groupings (by op name)

| Grouping | Examples in this slice | Notes |
|----------|------------------------|--------|
| Baseline | `aten::add_`, `aten::add` | Several large tensor shapes; largest share of trace 1 elementwise GPU kernel time. |
| Activation | `aten::threshold_backward`, `aten::sigmoid`, `aten::sigmoid_backward` | Paired with piecewise-linear and sigmoid paths; backward rows add non-trivial time. |
| Cast / copy | `aten::copy_` | Narrow ↔ wide numeric copies; comparative ratios vary strongly by shape and direction. |
| Arithmetic | `aten::clamp_min_`, `aten::clamp_min`, `aten::mul`, `aten::div` | In-place clamp, multiply, and scale-by-scalar style ops. |
| Other | `aten::fill_`, `aten::native_dropout_backward`, MSE-related elementwise | Small per-row time in aggregate but many invocations across the trace. |

## Top contributors (>5% of category GPU kernel time or >10 ms)

No single matched row exceeds **10 ms** of GPU kernel time. None exceed **5%** of category time; the largest row is **`aten::add_` at 4.92%** (4.05 ms). The **>5%** formal threshold for this category is **~4.11 ms** (5% of 82.23 ms), so the top **`aten::add_`** rows sit just under that cutoff but still dominate the head of the distribution.

## Comparative efficiency (trace 2 vs trace 1)

Comparative **`efficiency_percent`** in `elementwise_metrics.json` is **100 × (trace 2 kernel time) / (trace 1 kernel time)** (aligned with `elementwise_ops.csv` → `speedup (trace2/trace1)`). **Below 100%** means trace 2 spent **less** GPU kernel time on the matched row; **above 100%** means trace 2 was **slower**. This is **not** roofline utilization in comparative mode.

| Operation | Trace 1 GPU kernel time (ms) | Trace 2 GPU kernel time (ms) | Count (T1/T2) | Comparative efficiency (%) |
|-----------|-----------------------------:|-----------------------------:|---------------|---------------------------:|
| `aten::add_` | 4.046 | 3.338 | 80/80 | 82.49 |
| `aten::add_` | 3.781 | 3.218 | 125/125 | 85.10 |
| `aten::add_` | 2.931 | 2.565 | 32/32 | 87.52 |
| `aten::add_` | 2.676 | 2.346 | 16/16 | 87.66 |
| `aten::threshold_backward` | 2.524 | 1.718 | 232/232 | 68.06 |
| `aten::copy_` | 2.484 | 6.153 | 132/132 | 247.69 |
| `aten::threshold_backward` | 2.468 | 2.033 | 32/32 | 82.37 |
| `aten::copy_` | 2.401 | 5.831 | 127/127 | 242.85 |
| `aten::copy_` | 2.360 | 2.059 | 132/132 | 87.25 |
| `aten::copy_` | 2.100 | 2.031 | 127/127 | 96.73 |
| `aten::copy_` | 2.091 | 1.851 | 32/32 | 88.50 |
| `aten::copy_` | 2.052 | 1.889 | 32/32 | 92.05 |

Trace 2 times above are **trace 1 × efficiency_percent / 100** using values stored in `elementwise_metrics.json`.

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|--------------------------------|------------|
| Close pre-computed comparative elementwise gaps (rollup of all `kernel_tuning` rows in `elementwise_metrics.json` → `impact_estimates`) | kernel_tuning | 2.728–9.227 | 0.54–1.77% | medium |

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Elementwise time is spread across adds, activation backward, and dtype copies; comparative gaps favor trace 2 on most adds but several copies are slower on trace 2

**Identification:** The category is a **double-digit share of compute** (**15.9%**) with **6541** logical invocations, but per matched diff row no single shape exceeds the **10 ms / 5%** bottleneck cutoff. The head of the distribution is still informative: **`aten::add_`** variants, **`aten::threshold_backward`**, and **`aten::copy_`** rows account for most of the top trace 1 GPU kernel milliseconds. Comparative efficiency is **100 × trace 2 / trace 1** on GPU kernel time, not roofline. (source: `elementwise_metrics.json` → `percent_of_compute`, `operation_count`, `operations[].time_ms`, `operations[].percent_of_category`, `operations[].efficiency.efficiency_percent`)

**Data:**

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|------------------:|------------------:|---------------|----------------:|------------|
| `aten::add_` | 4.046 | 3.338 | 80/80 | 0.17 | memory |
| `aten::add_` | 3.781 | 3.218 | 125/125 | 0.08 | memory |
| `aten::add_` | 2.931 | 2.565 | 32/32 | 0.17 | memory |
| `aten::add_` | 2.676 | 2.346 | 16/16 | 0.17 | memory |
| `aten::threshold_backward` | 2.524 | 1.718 | 232/232 | 0.17 | memory |
| `aten::copy_` | 2.484 | 6.153 | 132/132 | 0.17 | memory |
| `aten::threshold_backward` | 2.468 | 2.033 | 32/32 | 0.17 | memory |
| `aten::copy_` | 2.401 | 5.831 | 127/127 | 0.17 | memory |
| `aten::copy_` | 2.360 | 2.059 | 132/132 | 0.17 | memory |
| `aten::copy_` | 2.100 | 2.031 | 127/127 | 0.17 | memory |
| `aten::copy_` | 2.091 | 1.851 | 32/32 | 0.17 | memory |
| `aten::copy_` | 2.052 | 1.889 | 32/32 | 0.17 | memory |

**Reasoning for slowdown:** On the largest **`aten::add_`** rows, trace 2 uses roughly **82–88%** of trace 1’s GPU kernel time, so trace 1 remains slower for those in-place adds even though no row crosses the strict **5%** category-time bar alone. **`aten::threshold_backward`** includes a large BF16 spatial row near **~68%** comparative efficiency, so trace 2 is markedly faster there while trace 1 still pays a few milliseconds. Several **`aten::copy_`** rows show comparative efficiency **well above 100%**, meaning trace 2 is **slower** on those dtype conversions—so trace 1 is not uniformly the slower trace for elementwise work. That split points to differing memory access patterns, framework/library paths, or launch scheduling between runs rather than a single global elementwise regression.

**Resolution:** **Algorithmically**, fuse chains of elementwise ops where possible (multiply–add patterns, clamp, and activation backward sequences) to cut round-trips through memory. **For kernel-focused work**, treat the favorable comparative gaps on **`aten::add_`** and large **`threshold_backward`** rows as opportunities to align trace 1 with trace 2’s faster kernels or tensor layouts. For **`copy_`** rows where trace 2 is slower, avoid treating trace 2 as the automatic performance target; compare data types, contiguity, and whether conversions can be removed by keeping one precision through a subgraph. Graph compilation or fused custom kernels can reduce launch count when many small elementwise kernels appear back-to-back.

**Impact estimate:**

- Low end (75% gap target): 2.728 ms savings (0.54% E2E)
- High end (100% gap target): 9.227 ms savings (1.77% E2E)
- Range: 2.728–9.227 ms (0.54–1.77% E2E)

*Rollup matches all `kernel_tuning` entries in `elementwise_metrics.json` → `impact_estimates` after merging into `elementwise_metadata.json` via `write_impact_estimates` for this single reasoning candidate.*

### Compute Kernel Insights

GPU kernel time and trace 1 / trace 2 comparative ratios for the top elementwise rows are summarized in the **Data** table and the **Comparative efficiency** section above. No `[HIGH VARIANCE]` rows were flagged in `elementwise_metrics.json`.

### System-Level Insights

Host-device sync for this category is summarized in `elementwise_metadata.json` → `time_breakdown` when present; no separate system-tier signals were evaluated in this elementwise slice.
