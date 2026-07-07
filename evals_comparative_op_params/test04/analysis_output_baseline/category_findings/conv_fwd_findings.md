<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Convolution Forward Analysis

**Status:** SUCCESS
**Category:** conv_fwd
**Comparison Scope:** comparative
**Total Category Time (Trace 1):** 1.9 ms
**% of E2E Compute:** 91.79%
**Operation Count:** 1

## Overview

This category contains a single `aten::convolution` operation performing a standard 2D convolution (3x3 kernel) in bf16 precision within a `Conv2d` module. In comparative mode, Trace 1 is slower than Trace 2 by 0.7 ms (Trace 1: 1.9 ms, Trace 2: 1.2 ms), indicating an optimization opportunity for Trace 1.

No transpose overhead was detected (0%), so memory layout mismatch is not a factor.

## Recommendations

### P1: Standard 2D Convolution Slower in Trace 1
**Insight**: A single 3x3 convolution accounts for 91.79% of E2E GPU time and is 36.84% slower in Trace 1 than Trace 2.
**Action**: Profile the dominant kernel for tile-size and wave-occupancy tuning to close the performance gap observed between the two traces.
<!-- impact-begin kind=p_item low=25.36 mid=29.59 high=33.81 -->
**Impact**: impact_score: 29.59
<!-- impact-end -->

## Detailed Analysis

<!-- reasoning-candidate tier=compute rank=1 -->
#### Standard 2D Convolution Slower in Trace 1

**Identification:** A standard 2D convolution (`aten::convolution`) in the `Conv2d` model layer runs 0.7 ms slower in Trace 1 (1.9 ms) than Trace 2 (1.2 ms). This operation dominates GPU time at 91.79% of E2E (source: `conv_fwd_metrics.json` → `operations[].percent_of_total`, `operations[].difference_ms`).

**Data:**

| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| aten::convolution | (4,256,56,56) bf16<br>(512,256,3,3) bf16<br>() bf16 | 1.9 | 1.2 | 1 / 1 | -0.7 | 1368.67 | compute-bound |

**Reasoning for Slowdown:** Trace 1 is 36.84% slower than Trace 2 for this convolution, with an absolute gap of 0.7 ms. The operation is compute-bound with a FLOPS/Byte ratio of 1368.67. Given that the same 3x3 convolution with identical shapes runs faster in Trace 2, the performance difference likely stems from kernel algorithm selection or configuration differences between the two traces. Bottleneck identified -- generate reproducer for kernel team.

**Resolution:** Profiling the GPU kernel with hardware counters can reveal whether tile-size selection or wave occupancy differs between the two traces. Matching the kernel configuration used in Trace 2 would close the 0.7 ms gap per invocation, directly reducing the dominant time contributor.

**Impact estimate:**

<!-- impact-begin kind=detail_estimate low=25.36 high=33.81 -->
- Low end impact_score: 25.36
- High end impact_score: 33.81
<!-- impact-end -->
