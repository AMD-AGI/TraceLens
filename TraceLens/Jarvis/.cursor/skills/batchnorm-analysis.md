---
name: BatchNorm Analysis
description: Analyze BatchNormalization operations for performance bottlenecks
triggers:
  - batchnorm analysis
  - analyze batchnorm
  - batch normalization analysis
tools:
  - terminal
  - file_read
  - file_write
---

# BatchNorm Operations Analysis Skill

Analyze BatchNormalization operations for memory bandwidth efficiency and optimization opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/batchnorm_ops.csv`**
2. **`metadata/batchnorm_metadata.json`**
3. **`category_data/batchnorm_tree_data.json`**

---

## Output Contract

Produces **`category_findings/batchnorm_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/batchnorm_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - BatchNorm is often 10-50% of compute in CNNs (ResNet, EfficientNet)

2. **Validate Bottlenecks**:
   - Compare to elementwise baseline (simple ops like add_, mul should achieve 70-80% of peak)
   - BatchNorm uses PyTorch native kernels, not vendor-optimized BLAS
   - Expected efficiency: Should match simple elementwise ops (50-70% of peak HBM BW)

3. **Contextualize Issues**:
   - **No TraceLens perf model** - calculations must be done manually
   - Memory-bound operation (low FLOPS/Byte)
   - If BatchNorm <20% of peak BW while elementwise >70%, it's a kernel issue

### Step 3: Trace Call Stacks

Review tree data to identify BatchNorm usage patterns in model architecture

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- Consider alternatives: GroupNorm, LayerNorm may have better-optimized kernels
- Fusion opportunities with adjacent convolutions (if applicable)
- Some frameworks offer fused Conv+BatchNorm kernels

**Path B - Kernel Optimization:**
- Compare BatchNorm efficiency to simple elementwise operations (baseline technique)
- If significantly lower than baseline, generate replay artifact for kernel team
- Profile with rocprof to diagnose specific memory access issues

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/batchnorm_findings.md`

---

## Common Patterns

### BatchNorm (Common Bottleneck in CNNs)
- **No TraceLens perf model** - must calculate manually
- **Often 10-50% of compute in CNNs** (ResNet, EfficientNet, etc.)
- **Uses PyTorch native kernels** by default, not vendor BLAS
- **Key check:** Compare achieved BW to what simple elementwise ops achieve (see baseline technique below)
- **If BatchNorm <20% of peak BW** while elementwise ops >70%, it's a kernel issue

### Elementwise Baseline Technique
When BatchNorm shows low memory bandwidth efficiency, **compare to simple elementwise ops** in the same trace:

```python
# From unified_perf_summary, find simple memory-bound ops
elementwise = df[df['name'].isin(['aten::add_', 'aten::mul', 'aten::copy_'])]
baseline_bw = elementwise['TB/s_mean'].mean()  # What the hardware CAN achieve
```

If elementwise ops achieve 70-80% of peak but BatchNorm achieves <20%, the issue is the kernel, not the hardware.

---

## Key Principles

1. **Baseline comparison is critical** - Always compare to simple elementwise ops
2. **CNNs are BatchNorm-heavy** - ResNet-50 can be 30-40% BatchNorm
3. **No perf model** - Manual calculation required
4. **Consider alternatives** - GroupNorm, LayerNorm may perform better

---

## Efficiency Context

| Efficiency | Assessment | Action |
|------------|------------|--------|
| >70% | Excellent | Well-optimized |
| 50-70% | Good | Acceptable performance |
| 30-50% | Acceptable | Monitor for improvements |
| <30% | Needs investigation | Compare to elementwise baseline |
