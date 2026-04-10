<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# SDPA Forward Analysis Findings

**Status:** OK (with data caveat, see below)

**Comparison scope:** comparative

**Platform:** MI355X (from `sdpa_fwd_metadata.json`)

> **Metric basis:** All bottleneck ordering and comparative ratios below use **GPU kernel time**, not CPU op duration.

> **Comparative efficiency:** **efficiency_percent = 100 × (trace 2 GPU kernel time) / (trace 1 GPU kernel time)** on matched logical rows (values above 100% mean trace 2 is slower on the GPU for that match).

## Attention implementation

- **Type:** Flash-style fused forward attention (not paged attention).
- **Evidence:** `sdpa_fwd_metrics.json` → `category_specific.flash_attention_detected` is true; `paged_attention_detected` is false. Summarized GPU work maps to a single fused forward kernel in `kernel_details_summary`.

## Data caveat (trace 1 kernel column vs JSON)

`sdpa_fwd_ops.csv` has **`Kernel Time (µs)_sum` = 0** for every row, so `sdpa_fwd_metrics.json` reports **`total_time_ms`: 0**, per-operation **`time_ms`: 0, null **`efficiency.efficiency_percent`**, and an empty **`impact_estimates`** array. The comparative path in tooling requires a positive trace 1 **`Kernel Time (µs)_sum`** to combine with **`speedup (trace2/trace1)`** / **`delta_us (trace2 - trace1)`**; those speedup/delta fields are also unset here.

The tables below recover **trace 1** GPU totals from each row’s **`kernel_details_summary`** → `total_duration_us`, and **trace 2** from **`Kernel Time (µs)_trace2_sum`**, then apply the comparative efficiency definition above.

## Category totals (GPU kernel time, trace 1)

| Metric | Value |
|--------|------:|
| Sum of trace 1 GPU kernel time (2 logical rows) | **4.48 ms** |
| Logical invocations (trace 1) | 33 |
| Share of overall compute (`percent_of_compute` in metrics) | 0% (metrics incomplete; see caveat) |

## Operations breakdown

| Operation | Count | Trace 1 GPU time (ms) | Trace 2 GPU time (ms) | % of category (T1) | Efficiency (100×T2/T1) | FLOPS/Byte | Type |
|-----------|------:|----------------------:|------------------------:|-------------------:|-------------------------:|:----------:|:----:|
| `FlashAttnFunc` | 1 | 0.572 | 1.738 | 12.8 | 304.0% | — | flash |
| `FlashAttnFunc` | 32 | 3.904 | 10.581 | 87.2 | 270.8% | — | flash |

**FLOPS/Byte** and roofline **bound** labels are omitted: those columns are empty in `sdpa_fwd_ops.csv`, so standalone roofline classification is not available from this export.

## Bottleneck assessment (GPU kernel time)

Thresholds: **> 100 ms** absolute GPU kernel time, or **> 5%** of category GPU time on trace 1.

- Neither row exceeds **100 ms** of GPU kernel time.
- **Both** rows exceed **5%** of category time on trace 1. The **32-invocation** `FlashAttnFunc` variant (**~3.9 ms**, **~87%** of category GPU time on trace 1) is the dominant slice.

**Comparative interpretation:** For both matched shapes, comparative efficiency is **~271–304%**, so **trace 2 spends roughly 2.7–3.0× the GPU kernel time of trace 1** for the same fused forward attention work. Treat this as a large gap only after confirming aligned workloads (same step, shapes, software stack, and stable clocks).

**Variance:** The 32-call row shows low relative spread in the summarized kernel stats (standard deviation small vs mean); nothing in the export flags **high_variance**.

## Recommendations (prioritized)

1. **Pipeline / export:** Regenerate the category CSV so **`Kernel Time (µs)_sum`** reflects trace 1 GPU kernel time; then re-run `sdpa_analysis.py` with **`--comparison_scope comparative`** so `sdpa_fwd_metrics.json`, **`time_ms`**, **`efficiency_percent`**, and **`impact_estimates`** populate from the same definitions.
2. **Fair comparison:** Validate alignment on sequence length, batch, head count, head dimension, and attention backend. The two rows differ in **Input Dims** (long key/value sequence vs shorter); analyze them as **two distinct matched pairs**, not a single homogeneous pool.
3. **Stack and kernel path:** If traces are fairly matched, investigate why trace 2 is much slower on the fused forward kernel (framework and library versions, compiler, kernel selection, power/clocks). Avoid recommending extra “fusion” for this path; it is already a fused forward implementation.

## Impact Summary

Pre-computed **`kernel_tuning`** rows are absent in `sdpa_fwd_metrics.json` → **`impact_estimates`** because per-operation **`time_ms`** is zero in JSON (trace 1 kernel sum column missing). Only those automated **`kernel_tuning`** entries are valid for quantified rollups; manual guesses are not substituted here.

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|----------------|------|------------------------|-------------------------------|------------|

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Fused forward attention kernel shows large trace 2 vs trace 1 GPU time on both SDPA rows

**Identification:** Forward scaled dot-product attention uses a fused flash-style implementation on this platform. Exported **`Kernel Time (µs)_sum`** is zero in `sdpa_fwd_ops.csv`, so automated comparative fields in `sdpa_fwd_metrics.json` did not receive **`efficiency_percent`** or **`impact_estimates`**. GPU-time comparison is recovered from **`kernel_details_summary`** (trace 1) and **`Kernel Time (µs)_trace2_sum`** (trace 2).

**Data:**

| Operation | Trace 1 GPU time (ms) | Trace 2 GPU time (ms) | Count |
|-----------|----------------------:|----------------------:|------:|
| `FlashAttnFunc` | 0.572 | 1.738 | 1 |
| `FlashAttnFunc` | 3.904 | 10.581 | 32 |

Trace 1 values are GPU totals from `kernel_details_summary` → `total_duration_us`. Trace 2 values are **`Kernel Time (µs)_trace2_sum` ÷ 1000**.

**Reasoning:** Trace 1 spends **~4.48 ms** total GPU time in this category, with **~87%** on the 32-invocation variant. For each row, trace 2 GPU time is **about three times** trace 1 (**~271–304%** comparative efficiency), indicating a substantial slowdown on the second trace for the fused forward kernel, subject to comparison fairness.

**Resolution:** Fix upstream reporting so trace 1 kernel aggregates fill **`Kernel Time (µs)_sum`**, then re-run category analysis. If data is already correct downstream, focus on reproducible micro-benchmarks and stack diff for the fused forward path rather than generic compute tuning without a bound label.

**Impact estimate:** Not quantifiable from pre-computed **`kernel_tuning`** impact rows (none produced).

### Compute kernel insights

The dominant cost is the **32-invocation** `FlashAttnFunc` cluster on the fused forward kernel. Comparative GPU kernel time is **several times higher** in trace 2 for both shapes, pending confirmation of aligned traces.

### System-level insights

`sdpa_fwd_metadata.json` → `time_breakdown` reports **no sync bottleneck** (`has_sync_bottleneck`: false). Host-side duration in that block is not used for the ordering above.
