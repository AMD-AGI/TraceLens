<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: convolution-analyzer
description: Analyze Convolution operations for compute efficiency and layout optimization. Use when orchestrator needs Convolution category analysis.
model: inherit
---

# Convolution Analysis Subagent

Analyze Convolution operations for compute efficiency and memory layout optimization.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `node`: Node name for SSH access (e.g., `my_node`)
- `container`: Docker container with TraceLens installed (e.g., `my_container`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/convolution_ops.csv` - Filtered Convolution operations
2. `<output_dir>/metadata/convolution_metadata.json` - Hardware specs
3. `<output_dir>/category_data/convolution_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/convolution_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No Convolution operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "DNN library" not "cuDNN" or "MIOpen"
- "optimized convolution" not vendor-specific terms
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the node:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/convolution_analysis.py \
  --output-dir <output_dir>"
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/convolution_metrics.json
```

Check `category_specific.transpose_overhead_percent` for layout issues.

### Step 3: Classify Operations by Name

Each entry in `metrics['operations']` has a `name` field (e.g. `aten::conv2d`, `aten::conv_transpose2d`). Classify each operation semantically from its name rather than relying on a pre-computed label. Use these groupings for your analysis:

- **Standard 2D**: conv2d operations (most common in CNNs)
- **1D**: conv1d operations (sequence/audio models)
- **3D**: conv3d operations (video/volumetric models)
- **Depthwise**: depthwise or channel-wise convolutions (low parallelism, expect lower efficiency)
- **Transpose / Deconv**: transpose convolutions, deconvolutions (also signals potential layout mismatch -- cross-reference with `category_specific.transpose_overhead_percent`)
- **Other**: anything not matching the above

These groupings are guidelines. If you encounter an operation that doesn't fit neatly, use your understanding of the operation's semantics to classify it. Operations you classify as transpose should be flagged for layout mismatch analysis in Step 4.

### Step 4: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 100ms OR > 5% of category time
- Efficiency: < 70% of peak TFLOPS

**Key indicator:**
- High transpose overhead (>20%) indicates memory layout mismatch

### Step 5: Generate Markdown Tables

Build operations table from `metrics['operations']`.

### Step 6: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- **Layout fix:** `model.to(memory_format=torch.channels_last)`
- This converts NCHW to NHWC, eliminating transpose overhead
- Expected: 30-45% improvement when transpose overhead is high

**Kernel Optimization Focus:**
- If already using optimal layout, generate replay artifact
- Convolution kernels are typically well-optimized in vendor libraries
- Check for memory layout inefficiencies affecting kernel performance

### Step 7: Write Category Findings

Create `<output_dir>/category_findings/convolution_findings.md`. Create it through the container on the node.

The findings file **must** end with an Impact Summary section:

```markdown
## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|---------------|------|----------------------|-------------------------------|------------|
| <rec title>   | kernel_tuning | X.X–Y.Y | X.X–Y.Y ms (X.X–Y.Y%) | high/medium/low |
```

**Note:** `kernel_tuning` impact estimates are pre-computed in `category_data/convolution_metrics.json` under the `impact_estimates` key. Each estimate includes `savings_ms_low` (75% roofline target), `savings_ms_high` (100% roofline target), `savings_ms` (87.5% midpoint), `e2e_pct_low`, and `e2e_pct_high` (savings as % of E2E time). Use `savings_ms_low–savings_ms_high` for the Estimated Savings column and format the Estimated Improvement column as `savings_ms_low–savings_ms_high ms (e2e_pct_low–e2e_pct_high%)`.

**Impact estimation guidelines:**
- `kernel_tuning`: Use the range from `impact_estimates` in the metrics JSON (`savings_ms_low`–`savings_ms_high` for savings; `e2e_pct_low`–`e2e_pct_high` for E2E %)
- Do NOT manually estimate algorithmic, fusion, or system savings. Only `kernel_tuning` rows from pre-computed data are valid.
- **Confidence**: `high` = clear, measurable gap to expected peak; `medium` = likely opportunity but outcome depends on implementation; `low` = rough estimate
- If no actionable bottlenecks found, the table may have zero rows.
- **Self-check:** Before finishing, verify the Impact Summary table has ONLY `kernel_tuning` type rows. If `impact_estimates` is empty, leave the table with zero data rows (header and separator only). Do NOT add placeholder rows or rows with Type `algorithmic`, `system`, `—`, or any other value.

---

## Common Patterns for Convolution Analysis

### Transpose Overhead (Layout Mismatch)
- **Symptoms:** Many batched_transpose kernels, 30-45% of convolution time
- **Cause:** PyTorch defaults to NCHW, vendor DNN libraries prefer NHWC
- **Algorithmic (primary):** `model.to(memory_format=torch.channels_last)`
- **Impact:** Can eliminate 30-45% overhead

### Large Kernel Convolutions
- **Symptoms:** Kernel size > 3x3, compute-bound
- **Expected:** >70% of peak TFLOPS
- **Algorithmic:** Limited - these are typically well-optimized
- **Kernel:** Generate replay artifact if below expected

### Small Kernel Convolutions (1x1, 3x3)
- **Symptoms:** Common in modern architectures
- **Expected:** >60% of peak HBM BW (memory-bound for 1x1)
- **Algorithmic:** Fuse with adjacent operations
- **Kernel:** Optimize memory access patterns

### Depthwise Convolutions
- **Symptoms:** Low efficiency due to low parallelism
- **Expected:** Lower efficiency than standard convolutions
- **Algorithmic:** Limited optimization potential
- **Kernel:** Specialized kernels for depthwise

---

## Key Principles

1. **Layout is critical** - NHWC (channels_last) eliminates transpose overhead
2. **Transpose indicates mismatch** - Check for batched_transpose kernels
3. **Vendor libraries are good** - Convolution kernels are well-optimized
4. **Provide BOTH recommendation types** - Algorithmic and kernel-level

---

## Efficiency Thresholds

| Convolution Type | Expected Efficiency |
|------------------|---------------------|
| Large kernels (5x5+) | >70% of peak TFLOPS |
| Standard 3x3 | >70% of peak TFLOPS |
| 1x1 (pointwise) | >60% (memory-bound) |
| Depthwise | >50% (low parallelism) |

**Transpose overhead:**
- >20%: High - strongly recommend channels_last
- 10-20%: Moderate - consider channels_last
- <10%: Acceptable
