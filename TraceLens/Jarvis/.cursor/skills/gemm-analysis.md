---
name: GEMM Analysis
description: Analyze GEMM (matrix multiplication) operations for performance bottlenecks and optimization opportunities
triggers:
  - gemm analysis
  - analyze gemm
  - matrix multiplication analysis
tools:
  - terminal
  - file_read
  - file_write
---

# GEMM Analysis Skill

Analyze GEMM operations (matrix multiplications: mm, bmm, addmm) for performance bottlenecks.

---

## Input Contract

Expects pre-filtered data in organized folders:
1. **`category_data/gemm_ops.csv`** - Filtered GEMM operations only
2. **`metadata/gemm_metadata.json`** - Hardware specs, trace path, GPU utilization
3. **`category_data/gemm_tree_data.json`** - Pre-computed parent chains and subtrees

**DO NOT load trace** - use pre-computed data.

---

## Output Contract

Produces **`category_findings/gemm_findings.md`** - Markdown summary of analysis with:
- Category significance assessment
- Validated bottlenecks with prioritization
- Optimization recommendations (Path A & B)
- Additional notes

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the Python analysis script:

```bash
python3 Jarvis/category_analyses/gemm_analysis.py --output-dir <output_dir>
```

The script outputs markdown analysis to stdout containing:
- Operations breakdown table
- Potential bottlenecks (flagged by script)
- Key metrics

### Step 2: Interpret Results

Read the markdown output from the script. The script flags potential bottlenecks, but you (LLM) must:

1. **Assess Category Significance** - Is GEMM worth optimizing? (>10% of compute = high priority)

2. **Validate Bottlenecks** - Review flagged operations:
   - Is the efficiency genuinely problematic for this GEMM type?
   - Is the time significant enough to warrant action?
   - Consider GEMM characteristics (tiny batched, quantized, large shapes)

3. **Contextualize Issues** - Consider:
   - Tiny batched GEMMs naturally have low efficiency (parallelism issues)
   - Quantized GEMMs (W8A8, FP8) may have different efficiency profiles
   - Very large GEMMs may be limited by memory bandwidth
   - High batch counts suggest batching opportunities

4. **Identify Additional Issues** - Look beyond script flags:
   - Patterns of many small GEMMs (batching opportunity)
   - Missing perf models on critical GEMMs
   - Unexpected kernel selections

### Step 3: Trace Call Stacks (if needed)

For significant bottlenecks, review pre-computed tree data for context:

```python
import json
with open('<output_dir>/category_data/gemm_tree_data.json') as f:
    tree_data = json.load(f)

# tree_data format: {op_uid: {"name": ..., "parent_chain": [...], "time_ms": ...}}
# Review parent chain to understand where GEMM is called from
```

### Step 4: Determine Optimization Paths

For each **validated bottleneck** (confirmed as worth addressing), assign priority:

**Bottleneck Prioritization:**
- **Critical (Priority 1):** >15% of compute time AND <30% efficiency
- **High (Priority 2):** >10% of compute time OR <40% efficiency
- **Medium (Priority 3):** >5% of compute time OR notable pattern (batching opportunity)
- **Low (Priority 4):** Everything else flagged by script

Then determine recommendations for BOTH paths:

**Path A - Fusion/Algorithmic:**
- Batch small GEMMs together to improve GPU parallelism
- Use sparsity-aware operations if weights are sparse
- Consider quantization (W8A8, FP8) for memory-bound GEMMs
- Check if torch.compile can batch operations automatically

**Path B - Kernel Optimization:**
- Generate replay artifact for kernel team to tune tile sizes
- Flag suboptimal GEMM kernel selections
- Note inefficient memory access patterns
- Identify opportunities for better kernel tuning

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/gemm_findings.md`:

```markdown
# GEMM Analysis Summary

## Overview
GEMMs account for X% of compute time. Average efficiency: Y%.

## Key Bottlenecks

### 1. <Operation Name>
- **Time:** X ms (Y% of compute)
- **Efficiency:** Z% of peak MAF
- **Issue:** [Brief description - e.g., "Tiny batched GEMMs with low parallelism"]
- **Path A:** [Fusion/algorithmic recommendation]
- **Path B:** [Kernel optimization recommendation]
- **Priority:** Critical/High/Medium/Low

[Repeat for top 3-5 validated bottlenecks]

## Additional Notes
- Missing perf models: [list if any]
- Quantized GEMMs detected: [count]
- Batching opportunities: [describe if applicable]
```

---

## Common Patterns for GEMM Analysis

### Tiny Batched GEMMs
- **Symptoms:** Huge batch count, tiny M/N/K dimensions (e.g., 1000+ GEMMs with M=8, N=16)
- **Issue:** GPU can't efficiently parallelize, memory overhead dominates
- **Path A:** Batch GEMMs together using torch.bmm or grouped operations
- **Path B:** If batching >5x slower on one platform vs reference, investigate kernel issues
- **Worth investigating if:** One platform is >5x slower than expected

### Compute-Bound GEMMs
- **Symptoms:** High FLOPS/Byte (>200), low TFLOPS/s compared to peak MAF
- **Expected:** Platforms with higher MAF should win
- **Path A:** Check if smaller batch sizes or better batching helps
- **Path B:** Kernel tuning for tile size optimization, better wave occupancy

### Memory-Bound GEMMs
- **Symptoms:** Low FLOPS/Byte (<100), low TB/s compared to peak HBM BW
- **Expected:** Platforms with higher HBM BW should perform better
- **Path A:** Fusion opportunities to reduce memory traffic
- **Path B:** If not reaching expected BW, indicates kernel optimization opportunity

### Quantized GEMMs (W8A8, FP8)
- **Special considerations:** Different efficiency profiles than BF16/FP32
- **Expected:** Should be faster than full-precision, but may have lower efficiency
- **Path A:** Validate quantization scheme and calibration
- **Path B:** Generate replay artifact - quantized kernels may need specific tuning

---

## Key Principles

1. **Verify with tree data** - Understand where GEMMs are called from (attention, MLP, etc.)
2. **Count matters** - High invocation counts indicate batching opportunities
3. **Calculate efficiency** - Compare achieved TFLOPS/s vs peak MAF
4. **Be specific** - Include M/N/K shapes, batch sizes, data types
5. **Provide BOTH paths** - User decides which applies to their situation

---

## Efficiency Thresholds

| Efficiency | Assessment | Action |
|------------|------------|--------|
| >80% | Excellent | Focus on Path A (algorithmic) |
| 60-80% | Good | Limited optimization potential |
| 40-60% | Acceptable | Consider Path B if high time |
| <40% | Needs investigation | Priority for Path B (kernel optimization) |

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
| Bank conflicts | Requires hardware counters (rocprof) | "Low efficiency - profile with rocprof to diagnose" |
| Cache hit rates | Requires hardware counters | "Large working set may exceed cache" |
| Occupancy | Requires hardware counters | "Kernel running slower than expected" |
| Root causes | Traces show WHAT, not WHY | "Bottleneck identified - generate reproducer for kernel team" |

**Key principle**: JARVIS identifies bottlenecks and generates reproducers. Root cause diagnosis requires profiling tools (rocprof, nsight-compute) on replay artifacts.
