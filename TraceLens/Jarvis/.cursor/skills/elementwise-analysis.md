---
name: Elementwise Analysis
description: Analyze elementwise operations for memory bandwidth efficiency and fusion opportunities
triggers:
  - elementwise analysis
  - analyze elementwise
tools:
  - terminal
  - file_read
  - file_write
---

# Elementwise Operations Analysis Skill

Analyze elementwise operations (add, mul, copy, etc.) for memory bandwidth efficiency and kernel fusion opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/elementwise_ops.csv`**
2. **`metadata/elementwise_metadata.json`**
3. **`category_data/elementwise_tree_data.json`**

---

## Output Contract

Produces **`category_findings/elementwise_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/elementwise_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - Elementwise ops typically 5-15% of compute

2. **Validate Bottlenecks**:
   - Compare to baseline bandwidth (simple ops like add_, mul should achieve 70-80% of peak)
   - Low efficiency on simple ops indicates kernel issues
   - Many small elementwise ops suggest fusion opportunities

3. **Contextualize Issues**:
   - Memory-bound (FLOPS/Byte < 50) - efficiency depends on memory access patterns
   - Simple ops (add, mul, copy) should be well-optimized
   - Complex ops (activation functions) may have lower efficiency

### Step 3: Trace Call Stacks

Use tree data to identify fusion opportunities:

```python
# Find module instances with many elementwise ops
# Example: RMSNorm, LayerNorm often have 5-8 elementwise ops that could be fused
```

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- **Kernel fusion** through torch.compile
- **Custom fused kernels** for common patterns (fused layer norm, fused RMSNorm)
- **Algorithmic changes** (e.g., RMSNorm instead of LayerNorm reduces ops)

**Path B - Kernel Optimization:**
- Generate replay artifact if simple ops achieve <50% of baseline
- Check memory access patterns
- Investigate if kernel is memory-bound as expected

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/elementwise_findings.md`

---

## Common Patterns

### Memory-Bound Operations
- **Symptoms:** Low FLOPS/Byte (<50), achieved TB/s far below peak
- **Path A:** Kernel fusion to reduce memory traffic
- **Path B:** Generate replay artifact for kernel team optimization
- **In comparison:** Platform with higher HBM BW should perform better - if not, indicates kernel optimization opportunity

### Elementwise Baseline Technique
When an op shows low memory bandwidth efficiency, **compare to simple elementwise ops** in the same trace:

```python
# From unified_perf_summary, find simple memory-bound ops
elementwise = df[df['name'].isin(['aten::add_', 'aten::mul', 'aten::copy_'])]
baseline_bw = elementwise['TB/s_mean'].mean()  # What the hardware CAN achieve
```

If elementwise ops achieve 70-80% of peak but your target op achieves <20%, the issue is the kernel, not the hardware.

### Fusion Opportunity Identification
Use tree traversal to identify fusion opportunities in unfused patterns:

```python
# Find a module instance (e.g., RMSNorm)
# Look for pattern:
# Module_RMSNorm
# ├── copy_ (dtype cast)     - 22 µs
# ├── pow (x²)               - 18 µs  
# ├── mean (variance)        - 8 µs
# ├── add (+epsilon)         - 2 µs
# ├── rsqrt                  - 2 µs
# ├── mul                    - 18 µs
# ├── copy_ (dtype cast)     - 15 µs
# └── mul (weight)           - 19 µs
```

Sum the kernel times to get fusion opportunity: 8 ops × N instances = potential savings.

---

## Key Principles

1. **Baseline comparison is critical** - Always compare to simple elementwise ops
2. **Many small ops = fusion opportunity** - Look for repeated patterns
3. **Memory-bound expected** - FLOPS/Byte < 50 is typical
4. **70-80% efficiency is good** - For simple elementwise operations
