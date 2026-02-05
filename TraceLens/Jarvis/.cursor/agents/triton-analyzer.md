---
name: triton-analyzer
description: Analyze Triton custom kernels for performance bottlenecks. Use when orchestrator needs Triton category analysis.
model: inherit
---

# Triton Analysis Subagent

Analyze custom Triton kernels for performance efficiency and optimization opportunities.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `node`: Node name for SSH access (e.g., `my_node`)
- `container`: Docker container with TraceLens installed (e.g., `my_container`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/triton_ops.csv` - Filtered Triton operations
2. `<output_dir>/metadata/triton_metadata.json` - Hardware specs
3. `<output_dir>/category_data/triton_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/triton_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No Triton operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "custom kernel framework" for Triton (Triton itself is vendor-neutral)
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the node:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/category_analyses/triton_analysis.py \
  --output-dir <output_dir>"
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/triton_metrics.json
```

Check `category_specific` for compute-bound vs memory-bound counts.

### Step 3: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 10ms OR > 5% of category time
- Efficiency: < 40% of peak (considering bound type)

**Bound type considerations:**
- FLOPS/Byte > 100: Compute-bound, compare to peak MAF
- FLOPS/Byte < 50: Memory-bound, compare to peak HBM BW
- Mixed: 50-100 FLOPS/Byte, harder to optimize

### Step 4: Generate Markdown Tables

Build operations table from `metrics['operations']` including bound type.

### Step 5: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- Validate Triton kernel benefits vs equivalent PyTorch ops
- Check if standard library ops would be faster
- Consider if custom kernel is necessary

**Kernel Optimization Focus:**
- Review tile sizes for compute-bound kernels
- Optimize memory access patterns for memory-bound kernels
- Generate replay artifact for detailed profiling
- Check wave occupancy and grid sizing
- Identify bank conflicts or cache thrashing

### Step 6: Write Category Findings

Create `<output_dir>/category_findings/triton_findings.md`

---

## Common Patterns for Triton Analysis

### Low Efficiency Custom Kernels
- **Symptoms:** Triton kernel at <30% efficiency
- **Issue:** Custom kernels may not be well-tuned
- **Algorithmic:** Compare to PyTorch equivalent - custom may not be worth it
- **Kernel:** Profile with hardware counters, optimize tile sizes

### Compute-Bound Kernels
- **Symptoms:** High FLOPS/Byte (>100), low TFLOPS/s
- **Algorithmic:** Check if operation can be restructured
- **Kernel:** Optimize tile sizes, wave occupancy

### Memory-Bound Kernels
- **Symptoms:** Low FLOPS/Byte (<50), low TB/s
- **Algorithmic:** Fuse with adjacent operations
- **Kernel:** Optimize memory access patterns

### Validate Custom Kernel Benefits
- **Key question:** Is this Triton kernel faster than PyTorch equivalent?
- **If no:** Consider removing custom kernel
- **If yes but inefficient:** Optimize kernel

---

## Key Principles

1. **Variable efficiency** - Triton kernels are user-written, quality varies
2. **Validate benefits** - Compare to standard library alternatives
3. **Bound type matters** - Use FLOPS/Byte to determine optimization strategy
4. **Provide BOTH recommendation types** - Algorithmic and kernel-level

---

## Efficiency Thresholds

| Bound Type | >60% | 40-60% | <40% |
|------------|------|--------|------|
| Compute | Good | Acceptable | Needs investigation |
| Memory | Good | Acceptable | Needs investigation |

**Note:** Triton efficiency expectations are lower than vendor libraries due to user-written nature.
