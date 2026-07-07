<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GEMM Analysis

**Status:** SUCCESS
**Platform:** MI300X
**Total GEMM Time:** 0.65 ms
**GEMM Share of Compute:** 100.0%
**Comparison Scope:** comparative

## Operations Breakdown

| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| aten::mm | (1024,4096) bf16<br>(4096,2048) bf16 | 0.65 | 0.32 | 1 / 1 | -0.33 | 585.14 | compute-bound |

## Recommendations

### P1: Compute-Bound GEMM Slower in Trace 1 (Tensile)
**Insight**: The aten::mm operation (1024x2048x4096 BF16) takes 0.65 ms in Trace 1 versus 0.32 ms in Trace 2, making Trace 1 approximately 103% slower.
**Action**: Profile the Trace 1 kernel for tile-size and wave-occupancy tuning. Compare the Tensile kernel configuration between traces to identify which tuning parameters improved Trace 2 performance.
<!-- impact-begin kind=p_item low=38.08 mid=44.42 high=50.77 -->
**Impact**: impact_score: 44.42
<!-- impact-end -->

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Compute-Bound GEMM Slower in Trace 1
**Identification:** The aten::mm operation was flagged because Trace 1 takes 0.65 ms while Trace 2 takes only 0.32 ms, a difference of 0.33 ms. The kernel is launched via the Tensile backend. (source: `gemm_metrics.json` → `operations[].time_ms`, `operations[].t2_time_ms`, `operations[].library`)

**Data:**

| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| aten::mm | (1024,4096) bf16<br>(4096,2048) bf16 | 0.65 | 0.32 | 1 / 1 | -0.33 | 585.14 | compute-bound |

**Reasoning for Slowdown:** Trace 1 is approximately 103% slower than Trace 2 for this GEMM operation, with an absolute time gap of 0.33 ms. The operation is strongly compute-bound (585.14 FLOPS/Byte). Tile size not visible — profile the kernel for tile-size tuning. Decomposition strategy not visible — profile the kernel for tiling layout.

**Resolution:** Profiling the Trace 1 Tensile kernel against the Trace 2 kernel can reveal which tile-size or decomposition strategy change accounts for the 2x speedup. If the Trace 2 configuration uses a more efficient tiling layout for the (1024, 4096) x (4096, 2048) shape, adopting that configuration in Trace 1 would eliminate the 0.33 ms gap.

**Impact estimate:**
<!-- impact-begin kind=detail_estimate low=38.08 high=50.77 -->
- Low end impact_score: 38.08
- High end impact_score: 50.77
<!-- impact-end -->
