---
name: Generic Op Analysis
description: Analyze other/generic operations including MoE, BatchNorm, Convolutions, and Communication
triggers:
  - generic op analysis
  - analyze other ops
  - other operations analysis
tools:
  - terminal
  - file_read
  - file_write
---

# Generic/Other Operations Analysis Skill

Analyze miscellaneous operations that don't fit into standard categories: Communication, Graph operations, and other uncategorized operations.

**Note:** MoE, BatchNorm, and Convolutions now have dedicated analysis skills (@moe-analysis, @batchnorm-analysis, @convolution-analysis).

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/other_ops.csv`**
2. **`metadata/other_metadata.json`**
3. **`category_data/other_tree_data.json`**

---

## Output Contract

Produces **`category_findings/other_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/other_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - "Other" is catch-all for uncategorized operations

2. **Validate Bottlenecks** - Different subcategories:
   - **Communication:** Single rank limitation, observe collective types and sizes
   - **Graph operations:** CUDA/HIP graph capture and replay
   - **Miscellaneous:** Memory operations, synchronization, etc.

3. **Contextualize Issues**:
   - Communication can only be analyzed from single rank perspective
   - Graph operations may indicate kernel launch optimization attempts
   - Miscellaneous operations are typically low overhead

### Step 3: Trace Call Stacks

Use tree data to understand operation context

### Step 4: Determine Optimization Paths

**For Communication Operations:**
- **Single rank limitation:** Can only observe collective types, message sizes, total time
- **Cannot diagnose:** Straggler vs communication time split without all ranks
- **Action:** Note collective types/sizes, recommend checking topology/configuration
- **If communication overhead high:** Review collective strategy and network topology

**For Graph Operations:**
- **Path A:** Graph capture may have overhead - validate benefits vs overhead
- **Path B:** Profile graph capture and replay separately
- **Action:** Ensure graph mode provides benefits for the workload

**For Miscellaneous Operations:**
- **Path A:** Usually low overhead - focus on high-value optimizations first
- **Path B:** Only investigate if taking significant time (>5% of compute)

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/other_findings.md`

---

## Common Patterns

### Communication (DDP/Multi-GPU)
- **Single rank limitation:** Can only observe collective types, message sizes, total time from one rank
- **Cannot diagnose:** Straggler vs communication time split without all ranks
- **If communication overhead high:** Note collective types/sizes, recommend checking topology/configuration. Use vendor neutral terminology.
- **Same collective types + message sizes across platforms:** Same collectives being used

### Graph Operations
- **CUDA/HIP Graphs:** Kernel launch optimization technique
- **Trade-off:** Graph capture overhead vs replay benefits
- **Check:** Ensure graph mode provides net benefit for the workload

---

## Key Principles

1. **Category-specific analysis** - Each subcategory has different expectations
2. **Communication limitations** - Single rank perspective only
3. **Graph operations** - Validate capture/replay benefits
4. **Low priority** - "Other" category typically has lower optimization potential
5. **Focus on high-value targets** - Prioritize major categories (GEMM, SDPA, etc.) first
