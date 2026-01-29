---
name: Triton Analysis
description: Analyze custom Triton kernels for performance efficiency
triggers:
  - triton analysis
  - analyze triton
  - custom kernel analysis
tools:
  - terminal
  - file_read
  - file_write
---

# Triton Kernel Analysis Skill

Analyze custom Triton kernels for performance efficiency and optimization opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/triton_ops.csv`**
2. **`metadata/triton_metadata.json`**
3. **`category_data/triton_tree_data.json`**

---

## Output Contract

Produces **`category_findings/triton_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/triton_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - Triton kernels vary widely in compute %

2. **Validate Bottlenecks**:
   - Custom kernels have highly variable efficiency
   - Check if memory-bound or compute-bound
   - Compare to equivalent PyTorch operations if known

3. **Contextualize Issues**:
   - Triton kernels are user-written, performance varies
   - Low efficiency may indicate suboptimal tile sizes
   - Memory access patterns matter significantly

### Step 3: Trace Call Stacks

Use tree data to understand kernel purpose:

```python
# Review parent modules to understand what the kernel does
# This helps determine if optimization is worthwhile
```

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- Review Triton kernel implementation for optimization opportunities
- Check tile sizes and memory access patterns
- Consider if standard PyTorch ops could be used instead
- Validate that custom kernel provides benefits

**Path B - Kernel Optimization:**
- Generate replay artifact with kernel source
- Compare performance to equivalent PyTorch operations
- Benchmark different tile size configurations

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/triton_findings.md`

---

## Common Patterns

### Low Efficiency Custom Kernels
- **Symptoms:** <40% efficiency, unclear if memory or compute bound
- **Path A:** Review Triton implementation for tile size and memory access optimization
- **Path B:** Generate replay artifact and compare to PyTorch baseline

### Validate Custom Kernel Benefits
- **Question:** Is custom kernel faster than equivalent PyTorch ops?
- **Action:** Compare performance to standard operations
- **If slower:** Consider removing custom kernel and using PyTorch ops

---

## Key Principles

1. **Custom kernels vary widely** - No standard efficiency expectations
2. **Tile sizes matter** - Often primary optimization parameter
3. **Validate benefits** - Ensure custom kernel actually improves performance
4. **Memory access patterns** - Critical for Triton kernel performance
