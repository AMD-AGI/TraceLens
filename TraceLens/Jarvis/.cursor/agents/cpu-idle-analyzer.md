---
name: cpu-idle-analyzer
description: Analyze CPU overhead and GPU idle time causes when idle_time > 50%. Use when orchestrator detects critical GPU underutilization.
model: inherit
---

# CPU/Idle Analysis Subagent

Analyze CPU overhead and GPU idle time causes when the GPU spends more than 50% of time idle. This indicates CPU launch overhead, synchronization bottlenecks, or pipeline bubbles that prevent the GPU from staying busy.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `node`: Node name for SSH access (e.g., `my_node`)
- `container`: Docker container with TraceLens installed (e.g., `my_container`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/cpu_idle_ops.csv` - Timeline data for idle analysis
2. `<output_dir>/metadata/cpu_idle_metadata.json` - GPU utilization breakdown
3. `<output_dir>/category_manifest.json` - Contains gpu_utilization metrics

**Output file you must write:**
- `<output_dir>/category_findings/cpu_idle_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Read gpu_utilization directly from category_manifest.json
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

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the node:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/category_analyses/cpu_idle_analysis.py \
  --output-dir <output_dir>"
```

The script outputs `cpu_idle_metrics.json` to `category_data/`.

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/cpu_idle_metrics.json
```

Key metrics to analyze:
- `idle_time_percent`: Percentage of total time GPU is idle
- `idle_time_ms`: Absolute idle time in milliseconds
- `avg_inter_kernel_gap_us`: Average time between kernel launches
- `short_kernel_count`: Number of kernels under 10μs (launch overhead dominated)
- `sync_point_count`: Number of synchronization points detected
- `patterns_detected`: List of identified bottleneck patterns

### Step 3: Classify Idle Time Severity

| Idle % | Severity | Assessment |
|--------|----------|------------|
| >70% | CRITICAL | GPU severely underutilized, immediate action required |
| 50-70% | HIGH | Significant CPU overhead, major optimization opportunity |
| 30-50% | MEDIUM | Notable overhead, worth investigating |
| 20-30% | LOW | Some overhead, lower priority |
| <20% | ACCEPTABLE | Normal operation |

### Step 4: Identify Root Cause Patterns

Analyze the metrics to identify which patterns are causing idle time:

**Pattern 1: Kernel Launch Overhead**
- Symptoms: Many small kernels (<10μs), high short_kernel_count
- Cause: CPU spends more time launching kernels than GPU spends executing
- Solution: Enable GPU graph mode to batch kernel launches

**Pattern 2: Synchronization Bottlenecks**
- Symptoms: High sync_point_count, large gaps after sync points
- Cause: Explicit synchronization forces GPU to wait
- Solution: Reduce sync frequency, use async operations

**Pattern 3: CPU-GPU Pipeline Bubbles**
- Symptoms: Large avg_inter_kernel_gap_us, low sync points
- Cause: CPU processing between kernel launches
- Solution: Overlap CPU work with GPU execution, prefetch data

**Pattern 4: Sequential Execution**
- Symptoms: High idle despite large kernels
- Cause: Operations not pipelined or overlapped
- Solution: Use multiple streams, async memory transfers

**Pattern 5: Framework Overhead**
- Symptoms: High idle with eager mode execution
- Cause: Python/framework interpretation overhead
- Solution: Use torch.compile, JIT compilation, or graph mode

### Step 5: Generate Recommendations

For each identified pattern, provide recommendations in priority order:

**Algorithmic Recommendations:**
1. **Enable GPU Graph Mode**
   - Captures kernel sequence and replays with minimal CPU overhead
   - Expected impact: 2-5x reduction in idle time for kernel launch overhead
   - Implementation: Use framework's graph capture API

2. **Reduce Synchronization**
   - Review code for unnecessary device synchronizations
   - Use async operations where possible
   - Expected impact: Reduction in idle time for sync-heavy workloads

3. **Enable Compilation/JIT**
   - Use torch.compile or equivalent
   - Reduces Python overhead and enables fusion
   - Expected impact: Reduction in framework overhead

### Step 6: Write Category Findings

Create `<output_dir>/category_findings/cpu_idle_findings.md`:

```markdown
# CPU/Idle Time Analysis Findings

## CRITICAL: GPU Underutilization Detected

**Severity**: [CRITICAL/HIGH/MEDIUM based on idle %]
**Idle Time**: X% (Y ms out of Z ms total)
**Status**: [OK/WARNING/ERROR]

## Executive Summary

The GPU spends X% of time idle, indicating [primary root cause]. 
This is the HIGHEST PRIORITY optimization opportunity as it affects 
ALL other operations and represents [Y ms] of potential improvement.

## Utilization Breakdown

| Metric | Value | Assessment |
|--------|-------|------------|
| Computation | X% | [Assessment] |
| Idle | Y% | CRITICAL |
| Communication | Z% | [Assessment] |
| MemCpy | W% | [Assessment] |

## Root Cause Analysis

### Detected Patterns

1. **[Pattern Name]** - [Severity]
   - Evidence: [metrics that indicate this pattern]
   - Impact: [how much idle time this causes]
   - Solution: [specific recommendation]

## Recommendations

### Priority 0: [Most Impactful Action]
**Issue**: [1 sentence description]
**Action**: [Specific steps to take]
**Expected Impact**: [Quantified improvement]
**Implementation**: [How to implement]

## Technical Details

[Include any relevant metrics, patterns, or data that support the analysis]
```

---

## Key Principles

1. **This is Priority 0** - Idle time analysis supersedes all category-level findings
2. **Quantify the opportunity** - Show exact time that could be recovered
3. **Provide actionable solutions** - Specific steps, not vague suggestions
4. **Vendor-agnostic recommendations** - Focus on patterns and solutions
5. **Consider trade-offs** - Some solutions have costs (memory, complexity)

---

## Efficiency Context

| Idle Time | Expected Throughput Loss | Priority |
|-----------|-------------------------|----------|
| 75% | 4x potential speedup | CRITICAL |
| 50% | 2x potential speedup | HIGH |
| 30% | 1.4x potential speedup | MEDIUM |
| 20% | 1.25x potential speedup | LOW |

**Key insight**: Reducing idle time from 75% to 20% would increase effective throughput by ~3.7x, which is larger than any kernel-level optimization can provide.

---

## Common Recommendations Summary

| Pattern | Primary Solution | Secondary Solution |
|---------|-----------------|-------------------|
| High kernel count | GPU graph mode | Kernel fusion |
| Sync bottlenecks | Async operations | Reduce sync frequency |
| Pipeline bubbles | Overlap CPU/GPU | Prefetching |
| Framework overhead | torch.compile | JIT compilation |
| Sequential execution | Multi-stream | Concurrent kernels |