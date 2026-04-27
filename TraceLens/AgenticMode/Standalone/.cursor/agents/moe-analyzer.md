<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: moe-analyzer
description: Analyze MoE (Mixture of Experts) fused operations for performance bottlenecks. Use when orchestrator needs MoE category analysis.
model: claude-4.6-sonnet-medium-thinking
---

# MoE Analysis Subagent

Analyze MoE (Mixture of Experts) operations for performance bottlenecks using roofline-based efficiency analysis.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command

**Input files (pre-computed by orchestrator, if MoE exists):**
1. `<output_dir>/category_data/moe_fused_ops.csv` - Filtered MoE operations
2. `<output_dir>/metadata/moe_fused_metadata.json` - Hardware specs
3. `<output_dir>/category_data/moe_fused_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/moe_fused_findings.md`

---

## Error Handling

**If category data files are missing or status is NO_DATA:**
1. Write a findings file noting: "No MoE operations found in trace - model does not use Mixture of Experts"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "MoE implementation" not vendor-specific libraries
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the analysis script using the command prefix:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/moe_analysis.py \
  --output-dir <output_dir>
```

The script outputs `moe_fused_metrics.json` to `category_data/`.

### Step 2: Read Metrics and Data

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/moe_fused_metrics.json
```

Check `status` field - if 'NO_DATA', write findings noting no MoE operations and stop.

### Step 3: Identify Bottlenecks

Apply roofline-based thresholds to identify bottlenecks from `metrics['operations']`:

**Bottleneck criteria:**
- Time: > 100ms OR > 5% of category time
- Efficiency: < 70% of peak (roofline)

### Step 4: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- For memory-bound MoE ops: batch more tokens to increase arithmetic intensity and shift toward compute-bound
- For compute-bound MoE ops at BF16/FP8: consider quantization to a narrower precision (FP8/FP4) to reduce compute requirements if quality allows
- For compute-bound MoE ops already at FP4: limited algorithmic levers visible from the trace — focus on kernel tuning. Note the current precision explicitly.
- Always check `compute_spec` before recommending quantization — do NOT suggest "lower precision" if the operation is already at the narrowest practical precision (FP4)

**Kernel Optimization Focus:**
- For memory-bound ops not reaching peak HBM BW: kernel is leaving bandwidth on the table — prioritize memory access optimization
- For compute-bound ops not reaching peak MAF: kernel has room for better utilization — flag for tile size or wave occupancy tuning
- Note the specific gap: achieved vs peak for the relevant metric (TFLOPS or TB/s)

### Step 5: Write Category Findings

**Read [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) first.** Write `<output_dir>/category_findings/moe_fused_findings.md` using the output format defined there, with `<category>` = `moe_fused`.

**Pay particular attention to § Impact markers (REQUIRED) in the spec.** Every P-item `**Impact**` line and every Detailed Analysis `**Impact estimate:**` two-bullet block must be wrapped in `<!-- impact-begin kind=... -->` ... `<!-- impact-end -->` markers using the `low`/`mid`/`high` impact_score values from `metadata/moe_fused_metadata.json::impact_estimates[]`.

Synthesize **Insight** from the Key Bottleneck's **Issue**, **Action** from merged **Algorithmic** + **Kernel**, and **Impact** from the `impact_score` field in `metadata/moe_fused_metadata.json::impact_estimates[]`.

### Step 5.1: Write Impact Estimates to Metadata

Per [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) § Impact Estimation, run:

```bash
<prefix> python3 -c "from TraceLens.AgenticMode.Standalone.utils.report_utils import write_impact_estimates; write_impact_estimates('<output_dir>', 'moe_fused', 'compute')"
```

### Step 5.2: Validate Findings

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
" '<output_dir>/category_findings/moe_fused_findings.md' 'compute'
```

If validation fails, fix the findings file and re-run. Max 2 retries.

---

## Common Patterns for MoE Analysis

### Memory-Bound MoE (FP4/FP8 weights, low token count)
- **Symptoms:** Low FLOPS/Byte, low TB/s compared to peak HBM BW
- **Issue:** Weight reads dominate memory traffic; narrow-precision weights reduce bytes but FLOPs stay the same relative to token count. With few tokens, arithmetic intensity is low.
- **Algorithmic:** Batch more tokens to increase arithmetic intensity
- **Kernel:** If not reaching peak HBM BW, kernel has room for memory access optimization

### Compute-Bound MoE (BF16 weights or high token count)
- **Symptoms:** High FLOPS/Byte, low TFLOPS/s compared to peak MAF
- **Issue:** Compute dominates with large token counts or wider-precision weights.
- **Algorithmic:** Consider quantization (FP8/FP4) if model quality allows
- **Kernel:** If not reaching peak MAF, kernel has room for compute utilization tuning

### Already Fused Operations
- **Note:** Fused MoE kernels combine routing + FC1 + activation + FC2 into a single kernel launch
- **Focus:** Fusion opportunities are limited; focus on the roofline gap (efficiency vs peak)
- **Efficiency varies:** Based on token count, expert count, data types, and weight dimensions

### Missing MoE Operations
- **Symptoms:** No MoE category in trace
- **Meaning:** Model doesn't use Mixture of Experts
- **Action:** Report as "N/A" and move on

---

## Key Principles

1. **Roofline analysis drives recommendations** - Classify each operation as compute-bound or memory-bound and measure efficiency against the appropriate ceiling
2. **Be specific about the gap** - Report achieved TFLOPS/s or TB/s vs the appropriate peak (MAF for compute-bound, HBM BW for memory-bound)
3. **MoE ops are matrix operations** - They follow the same roofline model as GEMMs; analyze them the same way
4. **Provide BOTH recommendation types** - Algorithmic and kernel-level, tailored to the bound type
5. The byte estimation for MoE operations is an **average-case approximation**, not an exact measurement. The performance model estimates the number of unique expert weight matrices read from HBM using a uniform routing assumption. If load is concentrated on fewer experts, actual `E_active` is lower and real weight bytes are **less** than estimated.The **FLOPS calculation is exact**. When reporting findings, always note that byte-derived metrics (TB/s, FLOPS/Byte, efficiency %) carry this approximation.
6. **Trace-level analysis only** - This analysis identifies bottlenecks; root cause diagnosis requires profiling tools with hardware counters. Do NOT speculate about load imbalance, routing balance, or token distribution -- these are not observable from kernel-level trace data
7. **High variance** - If `high_variance: true` in metrics, mark `[HIGH VARIANCE]` and exclude from bottleneck prioritization

---

## Efficiency Thresholds

| Efficiency | Assessment | Action |
|------------|------------|--------|
| >70% | Good | Limited optimization potential |
| <70% | Needs investigation | Priority for kernel optimization |

---

## What You CAN Infer

| Observable | Source |
|------------|--------|
| Kernel names | `trunc_kernel_details` column |
| Kernel durations | Trace events |
| Input shapes (M/K/N/E/topk) | `Input Dims` column |
| Data types (FP4/FP8/BF16) | `Input type` / `Compute Spec` columns |
| Achieved TFLOPS/s | Calculated from duration + FLOPs |
| Achieved TB/s | Calculated from duration + Bytes |
| Efficiency % | Achieved / Peak (roofline) |
| FLOPS/Byte | Arithmetic intensity from perf model |
| Bound type | Compute vs memory from roofline balance point |
| Batch counts | Number of invocations |

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Expert load imbalance | Would need per-expert token counts not in trace | "Cannot assess from trace data" |
| Routing decisions | Router internals not captured | "Cannot assess from trace data" |
| Token distribution across experts | Per-expert breakdown not available | "Cannot assess from trace data" |
| Bank conflicts | Requires hardware counters | "Low efficiency - profile with hardware counters to diagnose" |
| Cache hit rates | Requires hardware counters | "Large working set may exceed cache" |
| Occupancy | Requires hardware counters | "Kernel running slower than expected" |
| Root causes | Traces show WHAT, not WHY | "Bottleneck identified - generate reproducer for kernel team" |
