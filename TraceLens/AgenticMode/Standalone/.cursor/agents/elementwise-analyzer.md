<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: elementwise-analyzer
description: Analyze elementwise operations for performance bottlenecks and optimization opportunities. Use when orchestrator needs elementwise category analysis.
model: claude-4.6-sonnet-medium-thinking
---

# Elementwise Analysis Subagent

Analyze elementwise operations for memory-bandwidth efficiency. Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/elementwise_ops.csv` - Filtered elementwise operations
2. `<output_dir>/metadata/elementwise_metadata.json` - Hardware specs
3. `<output_dir>/category_data/elementwise_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/elementwise_findings.md`

**Critical:** Do NOT load the trace file directly. Use only the pre-computed data files.

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

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/elementwise_analysis.py \
  --output-dir <output_dir> \
  --comparison_scope <comparison_scope>
```

### Step 2: Read metrics

```bash
cat <output_dir>/category_data/elementwise_metrics.json
```

`category_specific.peak_hbm_bw_tbs` is the HBM BW reference for elementwise efficiency expectations.

### Step 3: Classify members by name

Each `category_findings[i].members[j].operation` carries a torch op name (e.g. `aten::add_`, `aten::sigmoid`, `aten::gelu`). Classify each member semantically when describing the finding:

- **Baseline ops** (simple memory-bound; expect >70% HBM BW): `add`, `mul`, `copy`, `fill`.
- **Arithmetic**: `sub`, `div`, `remainder`, `fmod`, `neg`, `abs`, `clamp`.
- **Activation**: `sigmoid`, `relu`, `gelu`, `silu`, `swish`, `tanh`, `mish`, `hardswish`, `leaky_relu`.
- **Cast / Convert**: `to`, `_to_copy`, `type_as`, `float`, `half`, `bfloat16`.
- **Math**: `exp`, `log`, `pow`, `sqrt`, `rsqrt`, `reciprocal`, `erf`.
- **Comparison / Mask**: `where`, `masked_fill`, `eq`, `ne`, `gt`, `lt`, `ge`, `le`.
- **Other**: anything not matching the above.

Baseline ops anchor the bandwidth comparison — if a baseline op underperforms while a complex op meets expectations, it points at a kernel issue, not an algorithmic one.

### Step 4: Render P-items from `category_findings`

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

Per [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `time_ms`, `library`) using the Action Prose Guidance, Expected Efficiency, and Common Patterns below. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the **CANNOT Infer** rows, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `memory` | Optimize memory access patterns of the dominant member kernels. For chains of memory-bound elementwise ops in the same parent module (activation + bias-add + dropout, etc.), defer to the kernel fusion analysis — fusion eliminates the intermediate write-back. For very high invocation counts of identically-shaped ops, batch upstream so each launch amortizes the load. |
| `compute` | Rare for elementwise; if it occurs, profile the kernel for tile-size tuning and confirm the operation isn't actually a small reduction or transcendental being misclassified. |

---

## Common Patterns

### Low baseline efficiency
- **Symptoms:** Simple ops (`add_`, `mul`, `copy_`) at <50% of peak HBM BW.
- **Reasoning:** Baseline elementwise should approach peak HBM BW; well below indicates kernel-level memory-access or launch-overhead issues.
- **Kernel:** Investigate memory access patterns and per-launch overhead.

### High invocation count
- **Symptoms:** >1000 invocations of similar elementwise ops.
- **Reasoning:** Per-launch overhead dominates; batching or fusion likely available.
- **Algorithmic:** Restructure to batch operations; chains in the same parent module → defer to kernel fusion analysis.

---

## Validate findings

Per [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) § Validate findings, run:

```bash
<prefix> python3 -c "
import sys
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/category_findings/elementwise_findings.md' 'compute'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
<<<<<<< HEAD

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
2. **Calculate efficiency** -
  **Standalone:** Compare achieved TB/s vs peak HBM BW (memory-bound elementwise). Elementwise ops should hit peak HBM BW
  **Comparative:** Compare achieved runtime in trace1 vs achieved runtime in trace2. use roofline fields only as supplementary context if needed
3. **Memory-bound** - Elementwise ops should hit peak HBM BW
4. **Fusion opportunities** - If chains of elementwise ops suggest fusion candidates, note the observation but defer fusion analysis to the kernel fusion module
5. **Provide BOTH recommendation types** - Algorithmic and kernel-level
6. **High variance** - If `high_variance: true` in metrics, mark `[HIGH VARIANCE]` and exclude from bottleneck prioritization

---

## Efficiency Thresholds

| Efficiency | Assessment |
|------------|------------|
| >70% | Good |
| <70% | Significant gap - investigate kernel issues |
=======
>>>>>>> staging
