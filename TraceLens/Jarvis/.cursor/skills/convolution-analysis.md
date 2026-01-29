---
name: Convolution Analysis
description: Analyze Convolution operations for performance bottlenecks and layout optimization
triggers:
  - convolution analysis
  - analyze convolution
  - conv analysis
tools:
  - terminal
  - file_read
  - file_write
---

# Convolution Operations Analysis Skill

Analyze convolution operations for compute efficiency and memory layout optimization opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/convolution_ops.csv`**
2. **`metadata/convolution_metadata.json`**
3. **`category_data/convolution_tree_data.json`**

---

## Output Contract

Produces **`category_findings/convolution_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/convolution_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - Convolutions dominate in CNN workloads (can be 60-80% of compute)

2. **Validate Bottlenecks**:
   - Check for transpose operations (indicates layout mismatch)
   - Transpose overhead of 30-45% is typical for NCHW → NHWC conversions
   - Convolution efficiency should be 60-80% of peak MAF for large kernels

3. **Contextualize Issues**:
   - **Layout mismatch:** PyTorch defaults to NCHW, but MIOpen/cuDNN prefer NHWC
   - **Transpose overhead:** `batched_transpose` kernels before/after convolutions
   - **Small convolutions:** May be memory-bound rather than compute-bound

### Step 3: Trace Call Stacks

Review tree data to identify convolution patterns and transpose operations

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- **Primary fix:** Convert model to channels_last memory format
  - `model.to(memory_format=torch.channels_last)`
  - Eliminates 30-45% transpose overhead
- Fused Conv+BatchNorm kernels (some frameworks support this)
- Consider depthwise separable convolutions if applicable

**Path B - Kernel Optimization:**
- If already using channels_last and efficiency still low, generate replay artifact
- Check tile size optimization for convolution kernels
- Profile specific convolution kernel selections

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/convolution_findings.md`

---

## Common Patterns

### Convolution Transpose Overhead (NCHW vs NHWC)
- **MIOpen/cuDNN kernels often prefer NHWC** layout
- **PyTorch defaults to NCHW** layout
- **Result:** `batched_transpose` kernels before/after each convolution (30-45% overhead)
- **Solution:** `model.to(memory_format=torch.channels_last)`
- **Check:** `trunc_kernel_details` for transpose kernels to estimate overhead

### Detection Pattern
Look for:
```
aten::conv2d - 100ms
├── batched_transpose - 20ms (input NCHW → NHWC)
├── MIOpen_conv_kernel - 60ms (actual convolution)
└── batched_transpose - 20ms (output NHWC → NCHW)
```

**Total overhead:** 40ms / 100ms = 40% wasted on layout conversions

### Compute-Bound vs Memory-Bound
- **Large kernels (7x7, 5x5):** Typically compute-bound, expect 60-80% MAF efficiency
- **Small kernels (1x1, 3x3):** May be memory-bound, expect 50-70% HBM BW efficiency
- **Depthwise convolutions:** Usually memory-bound

---

## Key Principles

1. **Layout matters** - NCHW vs NHWC can cause 30-45% overhead
2. **Transpose detection** - Look for batched_transpose in kernel details
3. **Primary optimization** - Convert to channels_last format
4. **CNN-specific** - Convolutions dominate CNN workloads

---

## Efficiency Context

| Operation Type | Expected Efficiency | Metric |
|----------------|---------------------|--------|
| Large convolutions (7x7, 5x5) | 60-80% of peak MAF | Compute-bound |
| Small convolutions (3x3, 1x1) | 50-70% of peak HBM BW | Memory-bound |
| With transpose overhead | 30-45% overhead | Layout mismatch |
| channels_last format | 60-80% (no transpose) | Optimized layout |
