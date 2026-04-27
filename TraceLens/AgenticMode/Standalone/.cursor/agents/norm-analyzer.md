<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: norm-analyzer
description: Analyze normalization operations (BatchNorm, LayerNorm, GroupNorm, etc.) for memory bandwidth efficiency. Use when orchestrator needs norm category analysis.
model: claude-4.6-sonnet-medium-thinking
---

# Normalization Analysis Subagent

Analyze normalization operations (BatchNorm, LayerNorm, GroupNorm, InstanceNorm) for memory bandwidth efficiency.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/norm_ops.csv` - Filtered normalization operations
2. `<output_dir>/metadata/norm_metadata.json` - Hardware specs
3. `<output_dir>/category_data/norm_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/norm_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No normalization operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "native normalization kernels" not vendor-specific terms
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

Execute the analysis script using the command prefix:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/category_analyses/norm_analysis.py \
  --output-dir <output_dir>
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/norm_metrics.json
```

### Step 3: Classify Operations by Name

Each entry in `metrics['operations']` has a `name` field (e.g. `aten::batch_norm`, `aten::layer_norm`, `aten::group_norm`). Classify each operation semantically from its name rather than relying on a pre-computed label. Use these groupings for your analysis:

- **BatchNorm**: batch_norm, batchnorm (per-channel normalization, common in CNNs)
- **LayerNorm**: layer_norm, layernorm (per-token normalization, common in Transformers)
- **GroupNorm**: group_norm, groupnorm (hybrid approach, used in diffusion models)
- **InstanceNorm**: instance_norm (per-instance normalization, used in style transfer)
- **Other**: anything not matching the above

These groupings are guidelines. If you encounter an operation that doesn't fit neatly, use your understanding of the operation's semantics to classify it. Note that different norm types may have different efficiency characteristics due to their implementations.

### Step 4: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 10ms OR > 5% of category time
- Efficiency: < 70% of peak HBM BW

**Baseline comparison:**
- Compare to simple elementwise ops (add_, mul, copy_)
- If normalization ops <20% while elementwise >70%, indicates kernel issue

### Step 5: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- Consider alternatives: LayerNorm, GroupNorm may have better kernels
- Check if torch.compile helps
- For fusion opportunities, defer to the kernel fusion analysis

**Kernel Optimization Focus:**
- Normalization ops use native PyTorch kernels, not optimized BLAS
- If significantly below baseline, investigate kernel issues

### Step 6: Write Category Findings

**Read [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) first.** Write `<output_dir>/category_findings/norm_findings.md` using the output format defined there, with `<category>` = `norm`.

### Step 6.1: Write Impact Estimates to Metadata

Per [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) § Impact Estimation, run:

```bash
<prefix> python3 -c "from TraceLens.AgenticMode.Standalone.utils.report_utils import write_impact_estimates; write_impact_estimates('<output_dir>', 'norm', 'compute')"
```

### Step 6.2: Validate Findings

Per [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) § Validate findings, run:

```bash
<prefix> python3 -c "
import sys
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/category_findings/norm_findings.md' 'compute'
```

If validation fails, fix the findings file and re-run. Max 2 retries.

---

## Common Patterns for Normalization Analysis

### Low Efficiency vs Baseline
- **Symptoms:** Normalization at <20% while elementwise at >70%
- **Issue:** Normalization kernel may be suboptimal
- **Algorithmic:** Try LayerNorm or GroupNorm alternatives
- **Kernel:** Profile kernel if efficiency is below expected threshold

### CNN-Heavy Workloads
- **Symptoms:** BatchNorm is 10-50% of compute
- **Common in:** ResNet, EfficientNet, etc.
- **Algorithmic:** Consider channels_last memory format
- **Kernel:** Optimize normalization kernels

### Norm Type Variations
- **BatchNorm:** Per-channel normalization
- **LayerNorm:** Per-token normalization
- **GroupNorm:** Hybrid approach
- **Note:** Different implementations may have different efficiency

---

## Key Principles

1. **Baseline comparison critical** - Compare to simple elementwise ops
2. **Memory-bound** - Should match elementwise efficiency
3. **Native kernels** - Uses PyTorch native, not vendor BLAS
4. **Alternatives exist** - LayerNorm/GroupNorm may perform better
5. **Provide BOTH recommendation types** - Algorithmic and kernel-level
6. **High variance** - If `high_variance: true` in metrics, mark `[HIGH VARIANCE]` and exclude from bottleneck prioritization

---

## Efficiency Thresholds

| Efficiency | Assessment |
|------------|------------|
| >70% | Good |
| <70% | Compare to baseline, may indicate kernel issue |