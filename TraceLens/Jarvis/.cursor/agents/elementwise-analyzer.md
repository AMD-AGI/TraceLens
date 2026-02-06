---
name: elementwise-analyzer
description: Analyze elementwise operations for performance bottlenecks and optimization opportunities. Use when orchestrator needs elementwise category analysis.
model: inherit
---

# Elementwise Analysis Subagent

Analyze elementwise operations (add, mul, copy, div) for memory bandwidth efficiency and optimization opportunities.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `node`: Node name for SSH access (e.g., `my_node`)
- `container`: Docker container with TraceLens installed (e.g., `my_container`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/elementwise_ops.csv` - Filtered elementwise operations
2. `<output_dir>/metadata/elementwise_metadata.json` - Hardware specs
3. `<output_dir>/category_data/elementwise_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/elementwise_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No elementwise operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "memory bandwidth" not vendor-specific terms
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the node:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/category_analyses/elementwise_analysis.py \
  --output-dir <output_dir>"
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/elementwise_metrics.json
```

Check `category_specific.baseline_efficiency_percent` for expected efficiency.

### Step 3: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 10ms OR > 5% of category time
- Efficiency: < 40% of peak HBM BW (compared to baseline)

**Special considerations:**
- Simple elementwise ops (add, mul, copy) should achieve 70-80% of peak HBM BW
- Complex elementwise ops may have lower efficiency
- High count indicates fusion opportunities

### Step 4: Generate Markdown Tables

Build operations table from `metrics['operations']`.

### Step 5: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations in both categories:

**Algorithmic Recommendations:**
- Fuse chains of elementwise ops (RMSNorm, LayerNorm patterns)
- Use torch.compile to auto-fuse operations
- Look for patterns like: mul → add → mul (normalization)

**Kernel Optimization Focus:**
- If baseline ops (add, mul, copy) have low efficiency, investigate kernel issues
- Compare to baseline bandwidth to identify anomalies
- Generate replay artifact for ops with unexpectedly low HBM bandwidth
- Check for memory access pattern issues

### Step 6: Write Category Findings

Create `<output_dir>/category_findings/elementwise_findings.md`. Create it through the container on the node:

---

## Common Patterns for Elementwise Analysis

### Fusion Opportunities
- **Symptoms:** Many small elementwise ops in sequence
- **Look for:** RMSNorm (mul, rsqrt, mul), LayerNorm patterns
- **Algorithmic:** Fuse using torch.compile or custom Triton kernels
- **Impact:** Can reduce memory traffic by 2-3x

### Low Baseline Efficiency
- **Symptoms:** Simple ops (add_, mul, copy_) at <50% of peak HBM BW
- **Expected:** 70-80% efficiency for these operations
- **Kernel:** Investigate memory access patterns, kernel launch overhead

### High Invocation Count
- **Symptoms:** >1000 invocations of similar elementwise ops
- **Indicates:** Batching or fusion opportunity
- **Algorithmic:** Restructure computation to batch operations

---

## Key Principles

1. **Baseline comparison** - Compare complex ops to simple ops (add, mul, copy)
2. **Memory-bound** - Elementwise ops should hit peak HBM BW
3. **Fusion is primary algorithmic optimization** - Look for chains of ops
4. **Provide BOTH recommendation types** - Algorithmic and kernel-level

---

## Efficiency Thresholds

| Efficiency | Assessment |
|------------|------------|
| >70% | Excellent - near optimal |
| 50-70% | Good |
| 30-50% | Investigate fusion opportunities |
| <30% | Anomaly - investigate kernel issues |
