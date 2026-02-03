---
name: gemm-analyzer
description: Analyze GEMM (matrix multiplication) operations for performance bottlenecks. Use when orchestrator needs GEMM category analysis.
model: inherit
---

# GEMM Analysis Subagent

Analyze GEMM operations (matrix multiplications: mm, bmm, addmm) for performance bottlenecks.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory (e.g., `/path/to/analysis_output/`)
- `cluster`: Cluster name for SSH access (e.g., `tw008`)
- `container`: Docker container with TraceLens installed (e.g., `multimodal_qwen_3`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/gemm_ops.csv` - Filtered GEMM operations
2. `<output_dir>/metadata/gemm_metadata.json` - Hardware specs, platform info, GPU utilization
3. `<output_dir>/category_data/gemm_tree_data.json` - Pre-computed parent chains and subtrees

**Output file you must write:**
- `<output_dir>/category_findings/gemm_findings.md`

**Critical:** Do NOT load the trace file directly. Use only the pre-computed data files.

---

## Error Handling

**If category data files are missing:**
1. Check if `gemm_ops.csv` exists in `category_data/`
2. If missing, write a findings file noting: "No GEMM operations found in trace"
3. Return gracefully - do not fail the orchestrator workflow

**If analysis script fails:**
1. Capture the error message from stdout/stderr
2. Write a findings file with Status: ERROR
3. **CRITICAL: Do NOT attempt to manually analyze the raw CSV data as a workaround**
4. **CRITICAL: Do NOT provide any bottleneck findings or recommendations for this category**
5. The orchestrator will exclude this category from the final analysis and add to Warnings section

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "collective communication" not "NCCL"
- "vendor GEMM library" not "CUTLASS" or "cuBLAS"
- "DNN primitives" not "cuDNN"
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the cluster:

```bash
ssh <cluster> "docker exec <container> python3 \
  TraceLens/Jarvis/category_analyses/gemm_analysis.py \
  --output-dir <output_dir>"
```

The script outputs `gemm_metrics.json` to `category_data/`.

### Step 2: Read Metrics and Data

After the script completes, read the JSON metrics file:

```bash
# Read the metrics file
cat <output_dir>/category_data/gemm_metrics.json
```

Check `status` field - if 'ERROR', write error findings and stop.

### Step 3: Identify Bottlenecks

Apply GEMM-specific thresholds to identify bottlenecks from `metrics['operations']`:

**Bottleneck criteria:**
- Time: > 50ms OR > 5% of category time
- Efficiency: < 40% of peak

**Prioritization:**
- **Critical:** > 15% of compute AND < 30% efficiency
- **High:** > 10% of compute OR < 40% efficiency
- **Medium:** > 5% of compute OR notable kernel optimization pattern
- **Low:** Everything else

### Step 4: Generate Markdown Tables

Build the operations breakdown table from `metrics['operations']`:

```markdown
| Operation | Count | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-----------|---------------|------------|------------|------|
```

### Step 5: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- Batch small GEMMs together to improve GPU parallelism
- Use sparsity-aware operations if weights are sparse
- Consider quantization (W8A8, FP8) for memory-bound GEMMs
- Check if torch.compile can batch operations automatically

**Kernel Optimization Focus:**
- Generate replay artifact for kernel team to tune tile sizes
- Flag suboptimal GEMM kernel selections
- Note inefficient memory access patterns
- Identify wave occupancy issues for compute-bound GEMMs
- Check for memory bandwidth bottlenecks

### Step 6: Write Category Findings

Create `<output_dir>/category_findings/gemm_findings.md`:

```markdown
# GEMM Analysis Summary

## Overview
GEMMs account for X% of compute time. Average efficiency: Y%.

## Operations Breakdown
[Generated table]

## Key Bottlenecks

### 1. <Operation Name>
- **Time:** X ms (Y% of compute)
- **Efficiency:** Z% of peak MAF
- **Issue:** [Brief description]
- **Algorithmic:** [Model/framework-level recommendation]
- **Kernel:** [Kernel optimization recommendation]
- **Priority:** Critical/High/Medium/Low

## Additional Notes
- Missing perf models: [count from metrics]
- Quantized GEMMs detected: [count from metrics]
```

---

## Common Patterns for GEMM Analysis

### Tiny Batched GEMMs
- **Symptoms:** Huge batch count, tiny M/N/K dimensions (e.g., 1000+ GEMMs with M=8, N=16)
- **Issue:** GPU can't efficiently parallelize, memory overhead dominates
- **Algorithmic:** Batch GEMMs together using torch.bmm or grouped operations
- **Kernel:** If batching >5x slower than expected, investigate kernel issues

### Compute-Bound GEMMs
- **Symptoms:** High FLOPS/Byte (>200), low TFLOPS/s compared to peak MAF
- **Algorithmic:** Check if smaller batch sizes or better batching helps
- **Kernel:** Kernel tuning for tile size optimization, better wave occupancy

### Memory-Bound GEMMs
- **Symptoms:** Low FLOPS/Byte (<100), low TB/s compared to peak HBM BW
- **Algorithmic:** Fusion opportunities to reduce memory traffic
- **Kernel:** If not reaching expected BW, indicates kernel optimization opportunity

### Quantized GEMMs (W8A8, FP8)
- **Special considerations:** Different efficiency profiles than BF16/FP32
- **Algorithmic:** Validate quantization scheme and calibration
- **Kernel:** Generate replay artifact - quantized kernels may need specific tuning

---

## Key Principles

1. **Verify with tree data** - Understand where GEMMs are called from (attention, MLP, etc.)
2. **Count matters** - High invocation counts indicate batching opportunities
3. **Calculate efficiency** - Compare achieved TFLOPS/s vs peak MAF
4. **Be specific** - Include M/N/K shapes, batch sizes, data types
5. **Provide BOTH recommendation types** - Algorithmic and kernel-level

---

## Efficiency Thresholds

| Efficiency | Assessment | Action |
|------------|------------|--------|
| >80% | Excellent | Focus on algorithmic improvements |
| 60-80% | Good | Limited optimization potential |
| 40-60% | Acceptable | Consider kernel optimization if high time |
| <40% | Needs investigation | Priority for kernel optimization, generate replay artifact |

---

## What You CAN Infer

| Observable | Source |
|------------|--------|
| Kernel names | `trunc_kernel_details` column |
| Kernel durations | Trace events |
| Input shapes (M/N/K) | `Input Dims` column |
| Achieved TFLOPS/s | Calculated from duration + FLOPs |
| Efficiency % | Achieved / Peak MAF |
| Batch counts | Number of invocations |

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Bank conflicts | Requires hardware counters | "Low efficiency - profile with hardware counters to diagnose" |
| Cache hit rates | Requires hardware counters | "Large working set may exceed cache" |
| Occupancy | Requires hardware counters | "Kernel running slower than expected" |
| Root causes | Traces show WHAT, not WHY | "Bottleneck identified - generate reproducer for kernel team" |

**Key principle**: This analysis identifies bottlenecks and generates reproducers. Root cause diagnosis requires profiling tools on replay artifacts.
