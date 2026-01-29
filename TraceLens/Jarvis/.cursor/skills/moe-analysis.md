---
name: MoE Analysis
description: Analyze Mixture of Experts (MoE) fused operations for performance bottlenecks
triggers:
  - moe analysis
  - analyze moe
  - mixture of experts analysis
tools:
  - terminal
  - file_read
  - file_write
---

# MoE Analysis Skill

Analyze MoE (Mixture of Experts) fused operations for performance bottlenecks.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/moe_fused_ops.csv`**
2. **`metadata/moe_fused_metadata.json`**
3. **`category_data/moe_fused_tree_data.json`**

---

## Output Contract

Produces **`category_findings/moe_fused_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

**Note:** If `moe_fused_ops.csv` does not exist or is empty, **report this category as missing** and skip analysis.

```bash
python3 Jarvis/category_analyses/moe_analysis.py --output-dir <output_dir>
```

(Note: This script doesn't exist yet - for now, manually analyze the CSV if present)

### Step 2: Interpret Results

1. **Assess Category Significance** - MoE can be 20-40% of compute in MoE models

2. **Validate Bottlenecks**:
   - MoE operations are typically already fused
   - Check efficiency compared to expected performance
   - Look for expert load imbalance issues

3. **Contextualize Issues**:
   - MoE kernels are specialized and already optimized
   - Efficiency depends on token distribution across experts
   - Load imbalance can cause underutilization

### Step 3: Trace Call Stacks

Use tree data to identify MoE layer structure

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- MoE is already fused - focus on expert routing balance
- Check token distribution across experts
- Review expert selection strategy (top-k)
- Consider expert parallelism configuration

**Path B - Kernel Optimization:**
- Generate replay artifact if efficiency notably low
- Profile for load imbalance or routing inefficiency
- Check if using latest fused MoE implementation

### Step 5: Write Category Findings

If MoE category is missing:

```markdown
# MoE Analysis Summary

## Overview
No MoE operations detected in this trace.

## Additional Notes
- Model does not use Mixture of Experts architecture
```

Otherwise, create `<output_dir>/category_findings/moe_fused_findings.md`

---

## Common Patterns

### MoE Operations
- **Already fused:** Typically use highly optimized fused kernels
- **Focus on:** Expert routing balance and load distribution
- **Efficiency:** Depends heavily on token distribution
- **Compare:** To reference MoE implementations

### Expert Load Imbalance
- **Symptoms:** Some experts heavily loaded, others underutilized
- **Impact:** Overall efficiency suffers from imbalance
- **Path A:** Review routing algorithm and token distribution
- **Path B:** Limited kernel-level optimization available

---

## Key Principles

1. **MoE typically already optimized** - Fusion already done
2. **Routing balance matters** - Token distribution affects efficiency
3. **Report if missing** - Many models don't use MoE
