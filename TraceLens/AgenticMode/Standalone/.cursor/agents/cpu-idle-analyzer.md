<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: cpu-idle-analyzer
description: Report GPU idle time percentage and utilization breakdown. Invoked when idle_time_percent exceeds 15%.
model: inherit
---

# CPU/Idle Analysis Subagent

Report GPU idle time percentage, utilization breakdown, and kernel launch statistics. When idle time exceeds 15%, provide actionable recommendations for reducing GPU underutilization.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/cpu_idle_ops.csv` - Timeline data for idle analysis
2. `<output_dir>/metadata/cpu_idle_metadata.json` - GPU utilization breakdown
3. `<output_dir>/category_data/category_manifest.json` - Contains gpu_utilization metrics

**Output file you must write:**
- `<output_dir>/system_findings/cpu_idle_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Read gpu_utilization directly from category_data/category_manifest.json
2. Provide analysis based on available data
3. Note limitations in findings

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT skip idle time recommendations**
3. Provide basic recommendations based on idle percentage alone

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU graph" not "CUDA graph" or "HIP graph"
- "kernel launch overhead" not vendor-specific terms
- "device synchronization" not "cudaDeviceSynchronize"
- Focus on patterns and solutions, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the analysis script using the command prefix:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/cpu_idle_analysis.py \
  --output-dir <output_dir>
```

The script outputs `cpu_idle_metrics.json` to `category_data/`.

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/cpu_idle_metrics.json
```

Key metrics to analyze:
- `idle_flagged`: Boolean -- whether idle time exceeds 15%
- `gpu_utilization.idle_time_percent`: Percentage of total time GPU is idle
- `gpu_utilization.idle_time_ms`: Absolute idle time in milliseconds
- `kernel_analysis.total_kernel_count`: Total GPU kernel launches
- `kernel_analysis.short_kernel_count`: Number of kernels under 10μs
- `kernel_analysis.avg_kernel_time_us`: Average kernel duration

### Step 3: Write Findings

Write `<output_dir>/system_findings/cpu_idle_findings.md` using the command prefix:

```markdown
# CPU/Idle Time Analysis Findings

> **Note:** This analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

**Status**: SUCCESS
**Idle Time**: X% (Y ms out of Z ms total)

## Utilization Breakdown

| Metric | Value |
|--------|-------|
| Computation | X% |
| Idle | Y% |
| Communication | Z% |
| MemCpy | W% |

## Kernel Launch Statistics

| Metric | Value |
|--------|-------|
| Total Kernels | N |
| Short Kernels (<10µs) | N (X%) |
| Avg Kernel Time | X.X µs |

## Recommendations

[If idle > 15%, provide actionable recommendations based on the kernel analysis data.
Use the Common Recommendations table below as guidance. If idle <= 15%, state that
idle time is within acceptable range and no action is needed.]

### [Recommendation Title]
**Insight**: [1 sentence description]
**Action**: [Specific steps to take]

## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|---------------|------|----------------------|-------------------------------|------------|
```

**Impact estimates are not produced for system-level analyses.** The Impact Summary table header must be present but must have **zero data rows**. Do NOT estimate savings for idle time reduction, GPU graph mode, sync reduction, or any other system-level recommendation.

---

## Key Principles

1. **Report factual data** - Idle percentage and kernel statistics from the metrics JSON
2. **Provide actionable solutions** - Specific steps, not vague suggestions
3. **No impact estimates** - System-level impact cannot be reliably estimated from trace data
4. **Vendor-agnostic recommendations** - Focus on patterns and solutions
5. **Consider trade-offs** - Some solutions have costs (memory, complexity)

---

## Common Recommendations Summary

| Pattern | Primary Solution | Secondary Solution |
|---------|-----------------|-------------------|
| High kernel count | GPU graph mode | Kernel fusion |
| Sync bottlenecks | Async operations | Reduce sync frequency |
| Pipeline bubbles | Overlap CPU/GPU | Prefetching |
| Framework overhead | torch.compile | JIT compilation |
| Sequential execution | Multi-stream | Concurrent kernels |