<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Multi-Tensor Apply Operations Analysis

**Status:** OK

**Comparison scope:** comparative

**Platform:** MI355X (from `multi_tensor_apply_metadata.json`)

## Overview

Seven `multi_tensor_apply` category operations account for **109.63 ms** of **GPU kernel time** in the reference trace (Trace 1), or **21.2%** of compute-time in this run. All seven are classified as **miscellaneous** (no graph or communication sub-category). In comparative mode, **`efficiency_percent`** is **100 × (Trace 2 kernel time) / (Trace 1 kernel time)** on matched rows—**not** roofline utilization. Bottleneck ordering and category share use **Trace 1 GPU kernel time** only.

Sub-categories: **0** graph, **7** miscellaneous.

Metadata notes **~49 ms** of host–device sync attributed to this slice (`time_breakdown.sync_time_ms`); the tables below still prioritize **device kernel time** for compute-focused comparison.

## Operations Breakdown

| Operation | Count | Trace 1 kernel time (ms) | Trace 2 kernel time (ms) | % of category (T1) | Comparative efficiency (100×T2/T1) | Sub-Category |
|-----------|-------|--------------------------|--------------------------|--------------------|-------------------------------------|--------------|
| aten::_foreach_addcdiv_ | 1 | 24.749 | 18.594 | 22.57 | 75.13% | miscellaneous |
| aten::_foreach_addcmul_ | 1 | 18.076 | 14.222 | 16.49 | 78.68% | miscellaneous |
| aten::_foreach_lerp_ | 1 | 17.882 | 14.121 | 16.31 | 78.97% | miscellaneous |
| aten::_foreach_sqrt | 1 | 12.737 | 12.210 | 11.62 | 95.86% | miscellaneous |
| aten::_foreach_div_ | 1 | 12.703 | 12.291 | 11.59 | 96.76% | miscellaneous |
| aten::_foreach_mul_ | 1 | 11.848 | 11.081 | 10.81 | 93.53% | miscellaneous |
| aten::_foreach_add_ | 1 | 11.640 | 11.074 | 10.62 | 95.14% | miscellaneous |

Trace 2 kernel times are **derived** as Trace 1 × (comparative efficiency / 100), consistent with `multi_tensor_apply_metrics.json`. Values **below 100%** mean Trace 2 spent **less** GPU kernel time on the matched op.

## Key Findings

### 1. aten::_foreach_addcdiv_

- **Trace 1 GPU kernel time:** 24.75 ms (largest share of this category).
- **Comparative efficiency:** **75.13%** — Trace 2 uses roughly **three quarters** of Trace 1’s kernel time for the same logical op.
- **Role:** List-parallel **fused multiply–divide–add** style update (optimizer-style tensor lists), implemented via **`multi_tensor_apply`** device kernels.
- **Tree data:** Parent chains are empty in `multi_tensor_apply_tree_data.json`; operand shapes are listed as empty tuples in the precomputed snapshot.

### 2. aten::_foreach_addcmul_ and aten::_foreach_lerp_

- **Trace 1:** 18.08 ms and 17.88 ms (~16.5% and ~16.3% of category kernel time).
- **Comparative efficiency:** **78.68%** and **78.97%** — same pattern as addcdiv: Trace 2 is materially faster on kernel time.
- **Role:** **addcmul** (fused multiply-add across lists) and **lerp** (linear interpolation across lists), typical of fused optimizer paths.

### 3. Remaining foreach ops (sqrt, div, mul, add)

- **Trace 1:** ~11.6–12.7 ms each; comparative efficiency **~93–97%**.
- Trace 1 and Trace 2 are **much closer** than on addcdiv / addcmul / lerp; lower priority for comparative kernel-time closure unless end-to-end budgets require every millisecond.

## GPU Graph Operations

No graph sub-category operations detected (`graph_count`: 0).

## Impact Summary

Pre-computed estimates use only **`kernel_tuning`** rows from `multi_tensor_apply_metrics.json` → `impact_estimates`.

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|-------------------------------|------------|
| Close pre-computed comparative multi-tensor apply gaps (rollup of all `kernel_tuning` rows in metrics) | kernel_tuning | 0–16.042 | 0–3.11% | medium |

## Notes

- Communication kernels are excluded from this slice when present; use the multi-rank / collective analysis path in TraceLens for those.
- Memcpy overlap and CPU/idle findings belong in system-level reports.
- All times above emphasize **GPU kernel time**, not CPU op duration.

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Multi-tensor foreach kernels: largest comparative gaps on addcdiv, addcmul, and lerp

**Identification:** **`aten::_foreach_addcdiv_`**, **`aten::_foreach_addcmul_`**, and **`aten::_foreach_lerp_`** dominate **absolute** category GPU kernel time on Trace 1 and show the **largest** comparative gaps (efficiency **~75–79%**), so they are the clearest multi-tensor apply targets when aligning Trace 1 with Trace 2. Remaining foreach variants are within a few percent. (source: `multi_tensor_apply_metrics.json` → `operations[].time_ms`, `operations[].efficiency.efficiency_percent`, `percent_of_compute`)

**Data:**

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|------------------:|------------------:|---------------|----------------:|------------|
| aten::_foreach_addcdiv_ | 24.749 | 18.594 | 1 / 1 | — | — |
| aten::_foreach_addcmul_ | 18.076 | 14.222 | 1 / 1 | — | — |
| aten::_foreach_lerp_ | 17.882 | 14.121 | 1 / 1 | — | — |
| aten::_foreach_sqrt | 12.737 | 12.210 | 1 / 1 | — | — |
| aten::_foreach_div_ | 12.703 | 12.291 | 1 / 1 | — | — |
| aten::_foreach_mul_ | 11.848 | 11.081 | 1 / 1 | — | — |
| aten::_foreach_add_ | 11.640 | 11.074 | 1 / 1 | — | — |

**Reasoning for gap:** The trace pair does not attach roofline bounds to these ops in comparative mode; the signal is purely **relative GPU kernel time**. The **~25%** kernel-time reduction on **`_foreach_addcdiv_`** (and similar on addcmul/lerp) suggests different effective **launch grouping**, **memory access patterns**, or **library / runtime** behavior between traces, not a single fused-GEMM-style kernel.

**Resolution:** Align **data types**, **tensor contiguity**, and **tensor-list batching** so `multi_tensor_apply` does not process extra bytes or extra rounds on Trace 1. If Trace 1 corresponds to a slower software stack, compare **PyTorch** and **device runtime** versions and confirm both runs use the same **foreach** code path (e.g. `_multi_tensor_adam`-style scopes noted elsewhere in fusion analysis). For kernel-level work, profile **`multi_tensor_apply_kernel`** template names and verify occupancy and memory coalescing on the slower trace.

**Impact estimate:** Rollup of pre-computed **`kernel_tuning`** rows: **0–16.042 ms** device-side savings band vs. Trace 2, **0–3.11%** end-to-end upper bound on this trace’s GPU span (medium confidence).

### Compute Kernel Insights

*(Orchestrator may merge this section into the final report.)*

### System-Level Insights

Host–device synchronization flagged in category metadata is out of scope for this compute-tier file; see system-level findings for sync and memcpy.
