<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Normalization (Norm) Category Findings

**Status:** OK  
**Platform:** MI355X  
**Comparison scope:** comparative  

All times below are **GPU kernel time** (microseconds summed per row in `norm_ops.csv`, reported as milliseconds). Trace 1 is the reference column (`Kernel Time (µs)_sum`); Trace 2 uses `Kernel Time (µs)_trace2_sum` where present. **Comparative efficiency** is **100 × (Trace 2 kernel time) / (Trace 1 kernel time)**, stored as `operations[].efficiency.efficiency_percent` in `category_data/norm_metrics.json` (regenerate with `norm_analysis.py --comparison_scope comparative`).

## Overview

| Metric | Value |
|--------|------:|
| Category GPU kernel time (Trace 1, summed) | 61.21 ms |
| Share of compute (`percent_of_compute`) | 11.83% |
| Distinct aggregated groups | 22 |
| Logical op instances (`operation_count`) | 1049 |

`metadata/norm_metadata.json` → `time_breakdown.gpu_kernel_time_ms` matches category kernel time for prioritization (not CPU duration).

## Norm type mix

| Semantic class | Primary ATen / library ops | Role in this slice |
|----------------|----------------------------|--------------------|
| **BatchNorm** | `aten::batch_norm`, `aten::miopen_batch_norm_backward` | Dominates wall time; spatial training forward and library backward. |
| **LayerNorm** | `aten::layer_norm`, `aten::native_layer_norm_backward` | Smaller but visible; transformer-style blocks (FP32 vectors on largest rows). |
| **GroupNorm / InstanceNorm** | — | Not present as named ops in this extract. |

## Top operations (Trace 1 GPU time)

**T2/T1 %** = `efficiency_percent` in comparative mode. **Below 100%** means Trace 2 spent **less** kernel time on the matched row; **above 100%** means Trace 2 was **slower**.

| Operation | Trace 1 (ms) | Trace 2 (ms) | Count (T1 / T2) | FLOPS/Byte (T1) | T2/T1 % | Bound (T1) |
|-----------|-------------:|-------------:|----------------:|----------------:|--------:|:----------|
| `aten::batch_norm` | 8.102 | 13.124 | 240 / 480 | 1.25 | 162.0 | memory |
| `aten::miopen_batch_norm_backward` | 8.013 | 8.850 | 240 / 480 | 1.00 | 110.5 | memory |
| `aten::miopen_batch_norm_backward` | 5.969 | 5.751 | 48 / 96 | 1.00 | 96.4 | memory |
| `aten::miopen_batch_norm_backward` | 5.181 | 6.834 | 96 / 192 | 1.00 | 131.9 | memory |
| `aten::batch_norm` | 5.042 | 9.714 | 96 / 192 | 1.25 | 192.7 | memory |
| `aten::batch_norm` | 4.924 | 11.599 | 48 / 96 | 1.25 | 235.6 | memory |
| `aten::miopen_batch_norm_backward` | 4.078 | 3.417 | 8 / 16 | 1.00 | 83.8 | memory |
| `aten::batch_norm` | 3.876 | 8.191 | 8 / 16 | 1.25 | 211.3 | memory |
| `aten::native_layer_norm_backward` | 3.564 | 5.988 | 64 / 64 | 0.75 | 168.0 | memory |
| `aten::layer_norm` | 2.019 | 2.788 | 63 / 64 | 0.62 | 138.1 | memory |

Trace 2 milliseconds follow `norm_ops.csv` (`Kernel Time (µs)_trace2_sum`) or Trace 1 × (T2/T1 %)/100. Counts follow `operation_count` / `operation_count_trace2` on the CSV rows.

## Bottlenecks

**Time (Trace 1):** rows with **> 5%** of category time (**~3.06 ms**) or **> 10 ms** (none here). Every row from **~3.9 ms** upward through the top **BatchNorm** clusters qualifies on the **5%** rule; no single aggregate exceeds **10 ms**.

**Comparative efficiency:** interpret **only** as **T2/T1 × 100**, not roofline. Rows **well above 100%** flag where Trace 2 is disproportionately heavy on the same matched shape; rows **below ~100%** flag where Trace 2 is faster and Trace 1 may still leave headroom versus that target.

**High variance:** no `high_variance: true` entries appear in `norm_metrics.json` for this extract.

**BatchNorm focus:** the largest shares are **BatchNorm** forward and **library** backward clusters. Several high-time rows sit at **~160–270%** T2/T1, so Trace 2 spends **much more** kernel time than Trace 1 on those shapes. A smaller set of backward rows (**~80–96%**) sit below **100%**; those drive the pre-computed **`kernel_tuning`** impact band in `norm_metrics.json` (gap-closure toward the faster trace on matched work).

**LayerNorm:** the **~3.6 ms** backward row shows **~168%** T2/T1 (Trace 2 slower on that pattern); it is **not** in the sub-100% **`kernel_tuning`** band.

## Impact Summary

Only **`kernel_tuning`** rows from `category_data/norm_metrics.json` → `impact_estimates` are used for quantified savings. Rollup matches a **single** reasoning candidate after `write_impact_estimates`.

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|-------------------------------|------------|
| Close comparative gap on batch-norm backward rows with sub-100% T2/T1 (Trace 1 vs Trace 2) | kernel_tuning | 0–1.028 | 0–0.20 | medium |

<!-- reasoning-candidate tier=compute rank=1 -->

## Detailed Analysis

#### BatchNorm comparative hotspots and a narrow backward gap band

**Identification:** Normalization is a **double-digit** share of compute on this platform for this pair (**~11.8%** of GPU time in the manifest), dominated by **BatchNorm** forward and **library** backward groups. Several top rows exceed **~150%** comparative **T2/T1**, so Trace 2 spends **much more** kernel time than Trace 1 on the same matched shapes. Separately, a few **backward** aggregates fall **under 100%** comparative efficiency; those are the only rows that produce quantified **`kernel_tuning`** savings in `norm_metrics.json` for this category.

**Data:**

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|------------------:|-------------------:|--------------:|----------------:|:----------|
| `aten::batch_norm` | 8.102 | 13.124 | 240/480 | 1.25 | memory |
| `aten::miopen_batch_norm_backward` | 8.013 | 8.850 | 240/480 | 1.00 | memory |
| `aten::miopen_batch_norm_backward` | 5.969 | 5.751 | 48/96 | 1.00 | memory |
| `aten::miopen_batch_norm_backward` | 4.078 | 3.417 | 8/16 | 1.00 | memory |
| `aten::miopen_batch_norm_backward` | 0.769 | 0.619 | 8/16 | 1.00 | memory |

*Comparative **T2/T1** is reported as **efficiency_percent** in metrics when `analysis_mode` is **comparative** (not roofline).*

**Reasoning for Slowdown:** The **>100%** rows mean Trace 2’s matched kernel time is **longer** than Trace 1’s for the same op label and shape bucket—consistent with heavier launches or less favorable execution on Trace 2 for those **BatchNorm** paths. The **sub-100%** **backward** rows mean Trace 2 **already finishes faster** on those buckets; the comparative model therefore attributes **kernel_tuning** headroom on Trace 1 toward closing that gap (band encoded in pre-computed `impact_estimates`).

**Resolution:** For Trace-2 regression shapes (**T2/T1** far above **100%**), prioritize alignment of training mode, layout, dtype, and fusion with adjacent math so normalization does not dominate those blocks versus Trace 1. For the **sub-100%** backward rows, **kernel/library** tuning on Trace 1 is the path aligned with the quantified band—without claiming micro-architectural root causes beyond what the trace shows.

**Impact estimate:**

- Low end (rollup `savings_ms_low` sum): 0 ms savings (0% E2E)
- High end (rollup `savings_ms_high` sum): 1.028 ms savings (~0.20% E2E)
- Range: **0–1.028 ms** (**0–~0.20%** E2E), from **`kernel_tuning`** rows only in `norm_metrics.json`

*Rollup matches `metadata/norm_metadata.json` → `impact_estimates` after `write_impact_estimates` merges all per-operation **`kernel_tuning`** entries for this single reasoning candidate.*

### Compute kernel insights

- **BatchNorm** forward and **library** backward carry most normalization GPU time; comparative **T2/T1** is mixed, with several **large** rows far above **100%** (Trace 2 slower) and a thin tail below **100%** that carries the only automated **`kernel_tuning`** savings band.
- **LayerNorm** paths are secondary in absolute time here but show **above-100%** comparative ratios on the largest modeled row—track separately from the backward gap rows.

### System-level insights

- No sync or host-side bottleneck flags appear in `norm_metadata.json` → `time_breakdown` for this category extract (`has_sync_bottleneck`: false); the story is **kernel-time** comparative structure, not CPU-reported duration.
