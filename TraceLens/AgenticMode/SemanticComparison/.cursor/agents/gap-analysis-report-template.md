<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Gap Analysis Report Template

Use this template when writing `semantic_comparison_report.md` in Step 8.
Replace placeholders with data from `comparison.csv`, `priority.json`, and
both `semantic_labels.json` files.

```markdown
# [Name A] vs [Name B]: Semantic Comparison Analysis

## Executive Summary

[1 paragraph: model architecture, platform comparison, overall ratio,
which trace is faster and by how much]

| Metric | [Name A] | [Name B] |
|--------|----------|----------|
| Total Iteration Time | X us | Y us |
| Model | architecture | architecture |
| Kernel Count | N | M |
| Blocks Compared | K matched, J only in one trace |
| Overall Ratio (A/B) | Z | - |

## Priority Improvement Targets

<!-- P1=red, P2=yellow, P3+=green -->

### P1: <semantic_block> -- <perf_category>

**Issue**: [Name A] is Xx slower than [Name B] (A: Y us vs B: Z us).

**Action**: [Recommendation based on perf_category.]

**Impact**: X us gap, Y% of total [Name A] runtime.

---

[Repeat for P2, P3, ...]

## Detailed Analysis: Kernel Matching

[For each semantic_group, then each semantic_block:]

### <semantic_group>

#### <semantic_block>

- **[Name A]**: `kernel_name_a` (N kernels, X us)
- **[Name B]**: `kernel_name_b` (M kernels, Y us)
- **Ratio**: Z (A/B)
- **Analysis**: [Why one is faster]

## Blocks Where [Name A] is Competitive or Faster

| semantic_block | perf_category | [Name A] (us) | [Name B] (us) | Ratio |
|---|---|---|---|---|

## Blocks Present in Only One Trace

| semantic_block | Present In | Reason |
|---|---|---|

## Key Takeaways

[3-5 bullet points:
1. Overall performance comparison
2. Largest improvement opportunities
3. Architecture/implementation differences driving the gap
4. Recommended next steps]
```

**For each semantic block, reason about:**

1. **Matching confidence**: Are these clearly the same operation?
2. **Fusion differences**: Does one platform fuse ops into one kernel?
3. **Implementation differences**: Different libraries?
4. **Architecture advantages**: Hardware features that explain gaps?
