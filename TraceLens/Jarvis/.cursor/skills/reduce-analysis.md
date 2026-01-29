---
name: Reduce Analysis
description: Analyze reduction operations (softmax, sum, mean) for performance bottlenecks
triggers:
  - reduce analysis
  - analyze reduce
  - softmax analysis
tools:
  - terminal
  - file_read
  - file_write
---

# Reduce Operations Analysis Skill

Analyze reduction operations for memory bandwidth efficiency and fusion opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/reduce_ops.csv`**
2. **`metadata/reduce_metadata.json`**
3. **`category_data/reduce_tree_data.json`**

---

## Output Contract

Produces **`category_findings/reduce_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/reduce_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - Reduce ops typically 5-10% of compute

2. **Validate Bottlenecks**:
   - Softmax is memory-bound, expected efficiency 50-70%
   - Check if softmax appears in unfused attention patterns
   - Sum/mean operations should be well-optimized

3. **Contextualize Issues**:
   - Softmax in attention context may be fusable
   - Standalone softmax may be harder to optimize
   - Large reductions may be limited by memory bandwidth

### Step 3: Trace Call Stacks

Use tree data to identify attention patterns with softmax:

```python
# Look for softmax within attention modules
# If found, check if part of unfused attention pattern
```

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- **Softmax in attention:** Migrate to Flash Attention (fuses softmax with other ops)
- **Softmax standalone:** Limited fusion opportunities
- **Sum/mean:** Check if part of layer norm or other fusable patterns

**Path B - Kernel Optimization:**
- Generate replay artifact if efficiency significantly below expected
- Check memory access patterns for reduction kernels

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/reduce_findings.md`

---

## Common Patterns

### Softmax in Attention
- **High impact:** Softmax in attention should be fused via Flash Attention
- **Unfused attention pattern:** softmax + bmm + mul typically indicates unfused attention
- **Path A (primary):** Migrate to Flash Attention
- **Expected efficiency:** Standalone softmax 50-70% of peak HBM BW

### Standalone Reductions
- **Memory-bound:** Expected efficiency 50-70%
- **Limited optimization:** Usually well-optimized already
- **Focus on time:** Only investigate if taking significant time (>5% of compute)

---

## Key Principles

1. **Softmax context matters** - In attention vs standalone
2. **Memory-bound expected** - 50-70% efficiency typical
3. **Fusion primary strategy** - Especially for attention patterns
