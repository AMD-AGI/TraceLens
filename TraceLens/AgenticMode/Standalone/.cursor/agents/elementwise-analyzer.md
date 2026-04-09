<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: elementwise-analyzer
description: Analyze elementwise operations for performance bottlenecks and optimization opportunities. Use when orchestrator needs elementwise category analysis.
model: inherit
---

# Elementwise Analysis Subagent

Analyze elementwise operations for memory bandwidth efficiency and optimization opportunities.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/elementwise_ops.csv` - Filtered elementwise operations
2. `<output_dir>/metadata/elementwise_metadata.json` - Hardware specs
3. `<output_dir>/category_data/elementwise_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/elementwise_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No elementwise operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "memory bandwidth" not vendor-specific terms
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the analysis script using the command prefix:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/elementwise_analysis.py \
  --output-dir <output_dir>
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/elementwise_metrics.json
```

Use `category_specific.peak_hbm_bw_tbs` as the peak HBM bandwidth reference for estimating expected efficiency of elementwise ops.

### Step 3: Classify Operations by Name

Each entry in `metrics['operations']` has a `name` field (e.g. `aten::add_`, `aten::sigmoid`, `aten::gelu`). Classify each operation semantically from its name rather than relying on a pre-computed label. Use these groupings for your analysis:

- **Baseline ops** (simple memory-bound; expect >70% HBM BW): add, mul, copy, fill
- **Arithmetic**: sub, div, remainder, fmod, neg, abs, clamp
- **Activation**: sigmoid, relu, gelu, silu, swish, tanh, mish, hardswish, leaky_relu
- **Cast / Convert**: to, _to_copy, type_as, float, half, bfloat16
- **Math**: exp, log, pow, sqrt, rsqrt, reciprocal, erf
- **Comparison / Mask**: where, masked_fill, eq, ne, gt, lt, ge, le
- **Other**: anything not matching the above

These groupings are guidelines. If you encounter an operation that doesn't fit neatly, use your understanding of the operation's semantics to classify it. Operations you classify as baseline should be used for the baseline bandwidth comparison in Step 4.

### Step 4: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 10ms OR > 5% of category time
- Efficiency: < 70% of peak HBM BW (compared to baseline)

**Special considerations:**
- Simple elementwise ops (add, mul, copy) should achieve >70% of peak HBM BW
- Complex elementwise ops may have lower efficiency

### Step 5: Generate Markdown Tables

Build operations table from `metrics['operations']`.

### Step 6: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- Use torch.compile to auto-fuse operations
- For fusion opportunities, defer to the kernel fusion analysis

**Kernel Optimization Focus:**
- If baseline ops (add, mul, copy) have low efficiency, investigate kernel issues
- Compare to baseline bandwidth to identify anomalies
- Check for memory access pattern issues

### Step 7: Write Category Findings

Write `<output_dir>/category_findings/elementwise_findings.md` using the command prefix.

The findings file **must** include **Impact Summary** followed by **Detailed Analysis**.

Impact Summary template:

```markdown
## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|---------------|------|----------------------|-------------------------------|------------|
| <rec title>   | kernel_tuning | X.X–Y.Y | X.X–Y.Y ms (X.X–Y.Y%) | high/medium/low |
```

**Detailed Analysis block:** Follow [`utils/templates/reasoning_block_template.md`](../utils/templates/reasoning_block_template.md) for the full block schema.

**Note:** `kernel_tuning` impact estimates are pre-computed in `category_data/elementwise_metrics.json` under the `impact_estimates` key. Each estimate includes `savings_ms_low` (75% roofline target), `savings_ms_high` (100% roofline target), `savings_ms` (87.5% midpoint), `e2e_pct_low`, and `e2e_pct_high` (savings as % of E2E time). Use `savings_ms_low–savings_ms_high` for the Estimated Savings column and format the Estimated Improvement column as `savings_ms_low–savings_ms_high ms (e2e_pct_low–e2e_pct_high%)`.

### Step 7.5: Write Impact Estimates to Metadata

Run the script below, then render impact bullets in your `## Detailed Analysis` block per `reasoning_block_template.md`.

```bash
<prefix> python3 -c "from TraceLens.AgenticMode.Standalone.utils.category_utils import write_impact_estimates; write_impact_estimates('<output_dir>', 'elementwise', 'compute')"
```

**Impact estimation guidelines:**
- `kernel_tuning`: Use the range from `impact_estimates` in the metrics JSON (`savings_ms_low`–`savings_ms_high` for savings; `e2e_pct_low`–`e2e_pct_high` for E2E %)
- Do NOT manually estimate algorithmic, fusion, or system savings. Only `kernel_tuning` rows from pre-computed data are valid.
- **Confidence**: `high` = clear fusion opportunity; `medium` = depends on kernel tuning quality; `low` = rough estimate
- **Self-check:** Before finishing, verify the Impact Summary table has ONLY `kernel_tuning` type rows. If `impact_estimates` is empty, leave the table with zero data rows (header and separator only). Do NOT add placeholder rows or rows with Type `algorithmic`, `system`, `—`, or any other value.

---

## Common Patterns for Elementwise Analysis

### Low Baseline Efficiency
- **Symptoms:** Simple ops (add_, mul, copy_) at <50% of peak HBM BW
- **Expected:** >70% efficiency for these operations
- **Kernel:** Investigate memory access patterns, kernel launch overhead

### High Invocation Count
- **Symptoms:** >1000 invocations of similar elementwise ops
- **Indicates:** Batching or fusion opportunity
- **Algorithmic:** Restructure computation to batch operations

---

## Key Principles

1. **Baseline comparison** - Compare complex ops to simple ops (add, mul, copy)
2. **Memory-bound** - Elementwise ops should hit peak HBM BW
3. **Fusion opportunities** - If chains of elementwise ops suggest fusion candidates, note the observation but defer fusion analysis to the kernel fusion module
4. **Provide BOTH recommendation types** - Algorithmic and kernel-level
5. **High variance** - If `high_variance: true` in metrics, mark `[HIGH VARIANCE]` and exclude from bottleneck prioritization

---

## Efficiency Thresholds

| Efficiency | Assessment |
|------------|------------|
| >70% | Good |
| <70% | Significant gap - investigate kernel issues |
