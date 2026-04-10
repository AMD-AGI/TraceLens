<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Uncategorized Operations Analysis

**Status:** OK

**Comparison scope:** comparative (GPU kernel time; **efficiency_percent** = 100 × trace2 kernel time / trace1 kernel time, i.e. 100 × the `speedup (trace2/trace1)` column in `other_ops.csv` when present.)

## Overview

Seven distinct uncategorized operation signatures (**165** rolled-up GPU invocations) account for **4.66%** of compute GPU kernel time (**24.12 ms** in this category on **MI355X**). Sub-categories: **0** graph, **7** miscellaneous. No communication kernels were skipped in this extract (`communication_ops_skipped` absent). Bottleneck ordering uses **GPU kernel time** from `other_metadata.json` → `time_breakdown.gpu_kernel_time_ms` and per-operation `time_ms` in `other_metrics.json`, not CPU-side duration.

## Operations Breakdown

| Operation | Count | Time (ms) | % of Category | Efficiency (T2/T1 %) | Notes |
|-----------|------:|----------:|---------------:|---------------------:|-------|
| `aiter::fmha_v3_bwd` | 32 | 16.99 | 70.43 | 192.47 | From CSV speedup 1.9247 |
| `hipLaunchKernel->…cuComputePartGradGammaBeta<float, float>… (Synthetic Op)` | 64 | 2.783 | 11.54 | — | No comparative columns in CSV row |
| `aiter::fmha_v3_bwd` | 1 | 2.359 | 9.78 | 242.53 | Longer sequence dim in inputs |
| `aten::native_dropout` | 64 | 1.711 | 7.09 | 88.65 | Trace2 faster on matched row |
| `aten::cat` | 1 | 0.138 | 0.57 | 132.08 | Trace2 slower |
| `aten::cat` | 2 | 0.137 | 0.57 | 135.91 | Trace2 slower |
| `aten::_local_scalar_dense` | 1 | 0.005 | 0.02 | 94.81 | Negligible; DtoD memcpy |

**Efficiency** values above use **`other_ops.csv`** → `speedup (trace2/trace1)` × **100** where populated. **Above 100%** means trace2 spent more aggregate GPU kernel time than trace1 on the matched row; **below 100%** means trace2 was faster. `other_metrics.json` currently leaves `operations[].efficiency.efficiency_percent` null for these rows; use the CSV for comparative ratios.

## Key Findings

### 1. `aiter::fmha_v3_bwd` (32 invocations, standard shapes)

- **GPU kernel time (trace1):** 16.99 ms (**70.4%** of this category).
- **Comparative:** **~192%** efficiency (trace2 / trace1): trace2 uses roughly **1.9×** the trace1 kernel time on the matched op. Trace2 aggregate kernel time **~32.7 ms** (`Kernel Time (µs)_trace2_sum` in `other_ops.csv`).
- **Kernels:** Tile-based FMHA backward stages plus `aiter::fmha_bwd_hd128_bf16_a32_psskddv` dominate `trunc_kernel_details`.
- **Categorization:** This is fused multi-head attention **backward**; it is often better handled under SDPA / fused-attention analysis than “other.”
- **Action:** Align attention backward path (library version, sequence length, fused vs decomposed) between traces; confirm both runs select the same backend before micro-tuning individual kernels.

### 2. Layer-norm partial γ/β gradient (`cuComputePartGradGammaBeta`)

- **GPU kernel time:** 2.78 ms (**11.5%** of category); **64** launches ~**43 µs** kernel time each.
- **Comparative:** Not available (empty speedup / trace2 columns on this CSV row).
- **Categorization:** Norm-family backward; consider the **Norm** category for a fuller view.
- **Action:** Prefer fused norm backward where the framework provides it; consider fusion with adjacent backward ops if this stays hot.

### 3. `aiter::fmha_v3_bwd` (1 invocation, extended sequence)

- **GPU kernel time (trace1):** 2.36 ms (**9.8%** of category).
- **Comparative:** **~243%** — trace2 even slower relative to trace1 than the high-count row (shapes include longer sequence tensors, e.g. **11040** in `other_ops.csv`).
- **Action:** Same family as item 1; long-sequence backward is sensitive to tile policy and memory traffic—compare stack versions and attention configuration between traces.

### 4. `aten::native_dropout`

- **GPU kernel time (trace1):** 1.71 ms (**7.1%** of category).
- **Comparative:** **~88.7%** — trace1 spends **more** kernel time than trace2 on the fused dropout kernel (`fused_dropout_kernel_vec` in kernel details). Aggregate trace2 kernel time **~1.52 ms** on the matched row.
- **Action:** Align framework build, RNG path, and tensor contiguity; profile the fused dropout path on the slower trace if the gap matters at full-job scale.

### 5. `aten::cat` and `aten::_local_scalar_dense`

- **GPU kernel time:** Sub-millisecond combined; low share of this category.
- **Comparative:** `aten::cat` rows show trace2 **~132–136%** of trace1 kernel time; `_local_scalar_dense` is negligible.
- **Action:** If concat cost grows elsewhere, reduce concat boundaries or reuse buffers; scalar sync path is not worth tuning here.

## Communication Kernels

Not applicable — no communication kernels in this extract. For collective traffic, use TraceLens’s **NCCL Analyzer** (vendor-agnostic: multi-device collective analysis).

## GPU Graph Operations

None detected (`graph_count` = 0).

## Impact Summary

`other_metrics.json` → **`impact_estimates`** is **empty** for this run, so there are **no pre-computed `kernel_tuning` rows** to cite for automated savings. Do not treat ad-hoc numbers as TraceLens impact estimates until the metrics pipeline populates that array.

**Qualitative comparative priorities (GPU kernel time, from CSV + category totals):**

| Priority | Topic | Direction (trace2 vs trace1) | Category share |
|----------|--------|------------------------------|----------------|
| 1 | `aiter::fmha_v3_bwd` (both shapes) | Trace2 slower (192% / 243%) | ~80% combined |
| 2 | `cuComputePartGradGammaBeta` | Not compared in CSV | ~11.5% |
| 3 | `aten::native_dropout` | Trace2 faster (~88.7%) | ~7.1% |
| 4 | `aten::cat` | Trace2 slower (~132–136%) | ~1.1% combined |

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Fused dropout: trace1 uses more GPU kernel time than trace2 on the matched row

**Identification:** `aten::native_dropout` is the only comparatively **faster-on-trace2** row among the smaller contributors (efficiency **< 100%** using 100 × trace2/trace1 from `other_ops.csv`). Dominant category time remains FMHA backward, where trace2 is **slower** (efficiency **> 100%**).

**Data:**

| Field | Value |
|--------|--------|
| Trace1 kernel sum (`Kernel Time (µs)_sum`) | ~1711 µs (~1.71 ms) |
| Trace2 kernel sum (`Kernel Time (µs)_trace2_sum`) | ~1517 µs (~1.52 ms) |
| Invocations (both traces) | 64 |
| 100 × trace2/trace1 | ~88.65% |
| Kernel | `fused_dropout_kernel_vec` (BF16, Philox) |

**Reasoning:** Same invocation count with lower aggregate kernel time on trace2 points to environment or implementation differences (build, RNG, launch/contiguity), not a missing op on either side.

**Resolution:** Match PyTorch and accelerator stack versions; confirm dropout mode and tensor layout. Profile fused dropout on the slower trace to match the faster trace’s kernel duration without changing intended numerics.

**Impact estimate:** No `kernel_tuning` object in `other_metrics.json` → `impact_estimates` for this run. Any end-to-end percentage must be recomputed against the full-job GPU timeline when estimates are regenerated.
