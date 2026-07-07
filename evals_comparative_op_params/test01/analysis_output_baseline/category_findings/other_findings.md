<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Other Operations Analysis

## Overview

This report covers GPU operations in the **other** category — operations that do not fit standard categories (GEMM, SDPA, Elementwise, Reduce, Norm, Convolution, MoE, Triton). Analysis is performed in **comparative** mode: `efficiency_percent` = 100 × (Trace 2 kernel time) / (Trace 1 kernel time). Values below 100% indicate Trace 1 is slower than Trace 2.

**Platform:** MI300X
**Total category kernel time (Trace 1):** 0.52 ms
**Category share of E2E GPU time:** 100.0%
**Operations analyzed:** 1

## Operations Breakdown

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Efficiency (T2/T1) | Count (T1/T2) | Sub-Category |
|-----------|-------------------|-------------------|--------------------|---------------|--------------|
| aten::scaled_dot_product_attention | 0.52 | 0.28 | 53.85% | 1 / 1 | miscellaneous |

## Key Findings

One operation (`aten::scaled_dot_product_attention`) is 46% slower in Trace 1 than Trace 2. Trace 1 executes the `attn_fwd` kernel in 0.52 ms versus 0.28 ms in Trace 2 — a gap of 0.24 ms. This operation accounts for 100% of the E2E GPU time in this trace segment, making it the primary bottleneck.

**Note on sub-category attribution:** Sub-category is heuristic — verify against op semantics before acting on it.

**Note on potential miscategorization:** `aten::scaled_dot_product_attention` is semantically an SDPA operation. Its presence in the `other` category suggests it was not matched by the SDPA category filter (possibly due to a missing or non-standard kernel signature). Cross-category fusion potential not assessed here — defer to the kernel fusion analysis.

---

## Recommendations

### P1: aten::scaled_dot_product_attention — Trace 1 Slower Than Trace 2

**Insight**: `aten::scaled_dot_product_attention` takes 0.52 ms in Trace 1 versus 0.28 ms in Trace 2, making Trace 1 approximately 46% slower for this operation.
**Action**: Profile the `attn_fwd` kernel in Trace 1 for tile-size and wave-occupancy tuning relative to the Trace 2 configuration. Compare the launch parameters (batch size, head count, sequence length, head dimension) and kernel dispatch settings between both traces to identify what changed between configurations.
<!-- impact-begin kind=p_item low=34.61 mid=40.38 high=46.15 -->
**Impact**: impact_score: 40.38
<!-- impact-end -->

---

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### aten::scaled_dot_product_attention — Trace 1 Slower Than Trace 2

**Identification:** `aten::scaled_dot_product_attention` was flagged because its Trace 2 kernel time is 53.85% of its Trace 1 kernel time (efficiency_percent < 100%), indicating Trace 1 is the slower configuration. The operation dispatches the `attn_fwd` GPU kernel with input shape (8, 16, 512, 64) in bfloat16. No launcher path was resolved in the trace. No model module chain is available (source: `other_metrics.json` → `operations[].efficiency.efficiency_percent`, `operations[].call_chain`).

**Data:**

| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) | Sub-Category |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|--------------|
| aten::scaled_dot_product_attention | (8,16,512,64) bf16<br>(8,16,512,64) bf16<br>(8,16,512,64) bf16 | 0.52 | 0.28 | 1 / 1 | -0.24 | — | — | miscellaneous |

**Reasoning for Slowdown:** Trace 1 takes 0.52 ms for this operation versus 0.28 ms in Trace 2 — Trace 2 is approximately 46% faster (absolute gap: 0.24 ms). Both traces run the same `attn_fwd` kernel on identical input shapes (batch=8, heads=16, seq=512, head_dim=64, dtype=bfloat16), so the performance difference is not due to shape or dtype divergence. The bound type cannot be determined from the trace alone (no FLOPS/byte estimate available); profiling with hardware counters is needed to diagnose whether the gap stems from occupancy, memory access pattern differences, or kernel parameter differences between the two runs. Bottleneck identified — generate a reproducer for the kernel team.

**Resolution:** Comparing the kernel dispatch configuration and any surrounding context (e.g., compiler flags, runtime settings, or op fusion changes) between the two traces may reveal what changed to produce the 46% speedup in Trace 2. If the improvement in Trace 2 results from a different kernel variant or updated tuning parameters, applying the same configuration to Trace 1 would close the gap. Kernel running slower than expected — profile occupancy with hardware counters.

**Impact estimate:**
<!-- impact-begin kind=detail_estimate low=34.61 high=46.15 -->
- Low end impact_score: 34.61
- High end impact_score: 46.15
<!-- impact-end -->
