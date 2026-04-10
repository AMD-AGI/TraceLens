<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Reduce Analysis Findings

**Status:** OK

> **Performance model caveat:** TraceLens does not provide dedicated performance models for reduction kernels. Roofline-style fields (for example, achieved throughput labels and generic memory-bound classification) are approximate context only. Do not treat them as precise kernel efficiency scores. Comparative **efficiency_percent** is **100 × (trace 2 GPU kernel time) / (trace 1 GPU kernel time)** on matched rows and is likewise approximate.

**Comparison scope:** comparative

**Platform:** MI355X (from `reduce_metadata.json`)

## Overview

| Metric | Value |
|--------|------:|
| Category GPU kernel time | 18.25 ms |
| Share of compute | 3.53% |
| Logical operation count | 472 |
| Softmax ops in category | 0 |

Work is dominated by **`aten::sum`** and **`aten::mean`** over BF16 tensors (spatial and axis reductions), with minor FP32 rows. No softmax, max, or min ops appear in this slice. Bottleneck ordering uses **GPU kernel time** from `reduce_metadata.json` → `time_breakdown.gpu_kernel_time_ms` and per-operation kernel times in `reduce_metrics.json`, not CPU-side durations.

## Operations by semantic type

| Type | Ops present | Role in this trace |
|------|-------------|-------------------|
| **Sum** | `aten::sum` | Largest share of category GPU kernel time (several shape variants). |
| **Mean** | `aten::mean` | Paired spatial reductions alongside matching sum patterns. |
| **Softmax** | — | None (`softmax_count` = 0). |
| **Max / Min** | — | No matching rows in `reduce_ops.csv`. |

## Top contributors (>5% of category GPU kernel time)

None exceed 10 ms. Rows above **~0.91 ms** meet the **>5% of category** threshold (5% of 18.248 ms).

| Operation | GPU kernel time (ms) | % of category | Count |
|-----------|---------------------:|--------------:|------:|
| `aten::sum` | 6.459 | 35.4 | 32 |
| `aten::sum` | 3.697 | 20.3 | 162 |
| `aten::sum` | 2.158 | 11.8 | 80 |
| `aten::mean` | 2.093 | 11.5 | 80 |
| `aten::sum` | 0.956 | 5.2 | 32 |
| `aten::mean` | 0.920 | 5.0 | 32 |

## Comparative efficiency (trace 2 vs trace 1)

Comparative **`efficiency_percent`** in `reduce_metrics.json` equals **100 × (trace 2 kernel time) / (trace 1 kernel time)**, derived from `reduce_ops.csv` → `speedup (trace2/trace1)` × 100. Values **below 100%** mean trace 2 spent **less** GPU kernel time than trace 1 on the matched row; **above 100%** means trace 2 was **slower** on that row.

| Operation | Trace 1 GPU kernel time (ms) | Count | Efficiency (comparative %) |
|-----------|-----------------------------:|------:|---------------------------:|
| `aten::sum` | 6.459 | 32 | 19.1 |
| `aten::sum` | 3.697 | 162 | 94.5 |
| `aten::sum` | 2.158 | 80 | 71.5 |
| `aten::mean` | 2.093 | 80 | 68.1 |
| `aten::sum` | 0.956 | 32 | 137.2 |
| `aten::mean` | 0.920 | 32 | 138.9 |
| `aten::mean` | 0.740 | 16 | 195.4 |
| `aten::sum` | 0.727 | 16 | 206.8 |

Roofline-style quantities in `reduce_metrics.json` (for example TB/s achieved) are **qualitative** framing for this category. Pre-computed **`kernel_tuning`** savings assume closing modeled gaps toward **75–100%** of the referenced target (midpoint **87.5%**).

## Impact Summary

Only **`kernel_tuning`** rows from `reduce_metrics.json` → `impact_estimates` are used below. Per reduce methodology, **`kernel_tuning` confidence is low** for every entry (no dedicated reduce models; comparative gap targets are uncertain).

| Recommendation | Type | Estimated savings (ms) | Estimated improvement (E2E %) | Confidence |
|----------------|------|------------------------|---------------------------------|------------|
| Memory-oriented tuning target: `aten::sum` (largest row) | kernel_tuning | 4.818–5.228 | 0.93–1.01 | low |
| Memory-oriented tuning target: `aten::mean` (80-count spatial pair) | kernel_tuning | 0.193–0.668 | 0.04–0.13 | low |
| Memory-oriented tuning target: `aten::sum` (80-count spatial pair) | kernel_tuning | 0.102–0.616 | 0.02–0.12 | low |
| Memory-oriented tuning target: `aten::sum` (162-count row) | kernel_tuning | 0–0.203 | 0.00–0.04 | low |

**Rollup (sum of estimate ranges):** **5.113–6.715 ms** kernel-time headroom if every modeled gap closed (**0.99–1.30%** E2E versus `reduce_metadata.json` → `gpu_utilization.total_time_ms`)—highly uncertain without dedicated reduce models.

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### BF16 sum and mean reductions dominate; mixed comparative outcome on matched rows

**Identification:** The reduce category is a modest fraction of overall compute (**3.53%**) but internally concentrated in **`aten::sum`** and **`aten::mean`** groups over large BF16 activations. No softmax or min/max ops appear in the category extract. (source: `reduce_metrics.json` → `percent_of_compute`, `operations[].name`, `operations[].time_ms`, `operations[].percent_of_category`, `category_specific.softmax_count`)

**Data:**

| Operation | Trace 1 time (ms) | Trace 2 time (ms) | Count (T1/T2) | FLOPS/byte (T1) | Bound label (T1) |
|-----------|------------------:|------------------:|---------------|----------------:|------------------|
| `aten::sum` | 6.459 | 1.231 | 32/32 | 0.50 | memory |
| `aten::sum` | 3.697 | 3.494 | 162/162 | 0.50 | memory |
| `aten::sum` | 2.158 | 1.541 | 80/80 | 0.50 | memory |
| `aten::mean` | 2.093 | 1.425 | 80/80 | 0.50 | memory |
| `aten::sum` | 0.956 | 1.311 | 32/32 | 0.50 | memory |
| `aten::mean` | 0.920 | 1.277 | 32/32 | 0.50 | memory |
| `aten::mean` | 0.740 | 1.446 | 16/16 | 0.50 | memory |
| `aten::sum` | 0.727 | 1.504 | 16/16 | 0.50 | memory |

Trace 1 and trace 2 times are GPU kernel totals from `reduce_ops.csv` → `Kernel Time (µs)_sum` and `Kernel Time (µs)_trace2_sum`. Comparative efficiency uses **`speedup (trace2/trace1)`** × 100 as stored in `operations[].efficiency.efficiency_percent`.

**Reasoning:** Several matched rows show trace 2 at **roughly 68–95%** of trace 1 GPU kernel time, so trace 1 remains slower for those reduction patterns. Rows **above 100%** comparative efficiency indicate trace 2 **slower** on the same matched shapes—often **~1.4×** for mid-sized spatial pairs and **~2×** for the larger **16-invocation** spatial reductions—consistent with scheduling, framework or device-library version differences, or memory access pattern shifts rather than a single root cause. The largest absolute slice of trace 1 reduce time (**~6.5 ms**) pairs with a **low** comparative ratio (**~19%**), meaning trace 2 is much faster on that pattern while trace 1 still pays most of the category cost.

**Resolution:** Prefer **algorithmic** wins where possible, such as fusing normalization or scale steps with neighboring work to avoid separate reduction passes. For **`kernel_tuning`**, verify reduction axes and layouts, avoid redundant back-to-back sum and mean on the same logical tensor when a fused path exists, and align framework and device-library versions. Treat kernel-level tuning as **low** confidence until dedicated reduce models exist.

**Impact estimate:**

- Low end (75% gap target): 5.113 ms savings (0.99% E2E)
- High end (100% gap target): 6.715 ms savings (1.30% E2E)
- Range: 5.113–6.715 ms (0.99–1.30% E2E)

*Rollup matches `reduce_metadata.json` → `impact_estimates` after `write_impact_estimates` merges all per-operation `kernel_tuning` entries for this single reasoning candidate.*
