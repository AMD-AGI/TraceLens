---
name: Standalone Analysis Orchestrator
description: Orchestrate modular standalone performance analysis - queries user, generates reports, pre-computes tree data, invokes category subagents in parallel
triggers:
  - standalone analysis
  - analyze trace standalone
  - performance analysis single platform
tools:
  - terminal
  - file_read
  - file_write
---

# Standalone Analysis Orchestrator

Orchestrate modular standalone PyTorch trace analysis. Coordinate workflow, pre-compute expensive operations, invoke category-specific subagents in parallel.

**Role**: Load trace once, pre-compute tree data, filter by category, invoke subagents in parallel, aggregate results, generate final report.

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective communication, vendor GEMM library, DNN primitives, GPU graph, etc. Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from traces, it's acceptable to include the actual name for identification.

---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path, Cluster, Container)
1. Generate Performance Report
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Category Filtering)
6. Invoke Category-Specific Subagents (PARALLEL)
7. Aggregate Results and Determine Optimization Recommendations
8. Generate Replay Artifacts
9. Generate Final Report (with Warnings section if errors)
```

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Trace File Path**
   - Ask: "Please provide the full path to your PyTorch trace file (.json or .json.gz)"
   - Example: `/home/user/traces/model_trace.json`

2. **Platform**
   - Ask: "Which platform are you analyzing?"
   - Options:
     1. **MI300X** - 5.3 TB/s HBM, 708 TFLOPS BF16, 192 GB
     2. **MI325X** - 6.0 TB/s HBM, 708 TFLOPS BF16, 256 GB
     3. **MI355X** - 6.5 TB/s HBM, 850 TFLOPS BF16, 288 GB
     4. **MI400** - 7.0 TB/s HBM, 1000 TFLOPS BF16, 320 GB

3. **Cluster Name**
   - Ask: "Which cluster should we use for analysis?"

4. **Container Name**
   - Ask: "Which Docker container has TraceLens installed?"

5. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

---

## Step 1: Generate Performance Report

Execute TraceLens CLI in the container:

```bash
ssh <cluster> "docker exec <container> \
  TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs"
```

This generates:
- `perf_report.xlsx` - Excel report with all sheets
- `perf_report_csvs/` directory with CSV files

**Duration:** ~60-120s depending on trace size

---

## Steps 2-5: Prepare Category Data

Execute the Jarvis orchestrator preparation script in the container:

```bash
ssh <cluster> "docker exec <container> python3 \
  TraceLens/Jarvis/orchestrator_prepare.py \
  --trace-path <trace_path> \
  --platform <platform> \
  --output-dir <output_dir>"
```

This script performs:
- **Step 2:** Assess GPU utilization (computation, idle, communication times)
- **Step 3:** Identify top 10 operations by GPU time
- **Step 4:** Pre-compute tree data for bottleneck operations (load trace ONCE)
- **Step 5:** Filter and export category-specific data

**Outputs:**
- `category_data/<category>_ops.csv` - Filtered operations per category
- `metadata/<category>_metadata.json` - Platform specs, GPU utilization, config
- `category_data/<category>_tree_data.json` - Pre-computed tree analysis
- `category_manifest.json` - Workflow metadata with categories_with_data

**Duration:** ~60-90s for tree data pre-computation

---

## Step 6: Invoke Category-Specific Subagents (PARALLEL)

### 6.1 Read Manifest and Identify Valid Categories

```python
import json
with open('<output_dir>/category_manifest.json') as f:
    manifest = json.load(f)
valid_categories = manifest.get('categories_with_data', [])
```

### 6.2 Launch ALL Valid Subagents in PARALLEL

For each category in `valid_categories`, invoke the corresponding subagent **simultaneously**.
Do NOT wait between invocations - launch all at once.

**Category to Subagent Mapping:**

- `gemm` → /gemm-analyzer
- `sdpa_fwd` → /sdpa-analyzer
- `elementwise` → /elementwise-analyzer
- `reduce` → /reduce-analyzer
- `triton` → /triton-analyzer
- `moe_fused` → /moe-analyzer
- `batchnorm` → /batchnorm-analyzer
- `convolution` → /convolution-analyzer
- `other` → /generic-op-analyzer

**Subagent Invocation Format:**

Pass only the execution context - let the subagent handle script execution:

```
/gemm-analyzer
- Output directory: <output_dir>
- Cluster: <cluster>
- Container: <container>
- Input files: category_data/gemm_ops.csv, metadata/gemm_metadata.json, category_data/gemm_tree_data.json
- Output file: category_findings/gemm_findings.md
```

**CRITICAL:** The orchestrator does NOT run any analysis scripts. Each subagent is responsible for:
1. Running its Python script inside the container on the cluster
2. Reading the metrics JSON output
3. Identifying bottlenecks and generating findings

### 6.3 Wait for All Subagents to Complete

All subagents must complete before proceeding to Step 7.
Each subagent writes its findings to `category_findings/<category>_findings.md`.

### 6.4 Verify Outputs and Collect Errors

After all subagents complete:

1. Check each `<category>_findings.md` for "Status: ERROR"
2. Collect list of failed categories and their error summaries
3. **CRITICAL: Exclude failed categories from aggregation and recommendations**
4. **CRITICAL: Do NOT attempt to manually analyze failed categories**

---

## Step 7: Aggregate and Determine Optimization Recommendations

### Read All Category Findings

```python
import os
import json

# Load category findings
findings_summaries = {}git add 
failed_categories = []
findings_dir = '<output_dir>/category_findings/'

for f in os.listdir(findings_dir):
    if f.endswith('_findings.md'):
        with open(os.path.join(findings_dir, f)) as file:
            content = file.read()
            if 'Status: ERROR' in content:
                # Extract error and add to failed list
                failed_categories.append({
                    'category': f.replace('_findings.md', ''),
                    'content': content
                })
            else:
                findings_summaries[f] = content

# Load category manifest for top operations
with open('<output_dir>/category_manifest.json', 'r') as f:
    manifest = json.load(f)
    top_ops = manifest.get('top_operations', [])
```

### Aggregate Recommendations

Each subagent has produced algorithmic and kernel optimization recommendations.
Consolidate these, cross-reference with `top_ops`, and prioritize by impact.

**Prioritization Framework:**

| Priority | Criteria |
|----------|----------|
| 🔴 Priority 1 | In top_ops + Low efficiency (<30%) + High category % (>15%) |
| 🟡 Priority 2 | In top_ops OR (Low efficiency <40% + >10% category time) |
| 🟢 Priority 3 | >5% category time OR notable optimization pattern |

---

## Step 8: Generate Replay Artifacts

**When to Generate:**
1. Op is a significant bottleneck (>10% of compute)
2. Efficiency is notably low (<30% of peak)
3. Kernel team needs a minimal reproducer

```bash
ssh <cluster> "docker exec <container> python3 \
  TraceLens/Jarvis/generate_replay_artifacts.py \
  --output-dir <output_dir> \
  --perf-report-path <output_dir>/perf_report.xlsx \
  --op-names <op1> <op2> <op3>"
```

---

## Step 9: Generate Final Report

Create `standalone_analysis.md` in `<output_dir>`:

**Purpose:** Clean stakeholder report with prioritized recommendations

```markdown
# <Model> - <Platform> Standalone Analysis

## Executive Summary
[1 paragraph overview + key metrics table]

| Metric | Value |
|--------|-------|
| Total Compute Time | X ms |
| GPU Utilization | Y% |
| Top Bottleneck Category | Category (Z%) |
| Flash Attention Usage | Yes/No |

## Warnings

**Include this section ONLY if any subagent failed:**

The following categories could not be analyzed due to script failures:

| Category | Error Summary |
|----------|---------------|
| <category> | <brief error description> |

These categories are excluded from the recommendations below.

---

## Top Operations

| Rank | Operation | Category | Time (ms) | % of Total Compute |
|------|-----------|----------|-----------|-------------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

---

## Recommendations

### 🔴 Priority 1: <Brief Title>
**Issue**: [1 sentence - what's wrong]
**Action**: [1-2 sentences - what to do]
**Impact**: [Expected improvement]
→ *See [Detailed Analysis: Section](#section-link) for details*

---

### 🟡 Priority 2: <Brief Title>
**Issue**: [1 sentence]
**Action**: [1-2 sentences]
**Impact**: [Expected improvement]
→ *See [Detailed Analysis: Section](#section-link) for details*

---

### 🟢 Priority 3: <Brief Title>
**Issue**: [1 sentence]
**Action**: [1-2 sentences]
**Impact**: [Expected improvement]
→ *See [Detailed Analysis: Section](#section-link) for details*

---

## Detailed Analysis

### 1. <Operation Category> (X% of compute)
[All kernel breakdowns, calculations, tables, explanations from category findings]

### 2. <Operation Category> (X% of compute)
[...]

---

## Appendix

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF**: Y TFLOPS

### Replay Artifacts
[List of generated replay packages if any]
```

**Key formatting rules:**
1. **Warnings section**: Only include if there were errors; omit entirely if all succeeded
2. **Executive Summary**: Max ~20 lines
3. **Recommendations**: Max ~10 lines PER recommendation
4. **Detailed Analysis**: All tables, calculations, explanations go HERE
5. **No redundancy**: Information appears in ONE place only

---

## Key Principles

1. **Always verify with TraceLens** - Don't assume sources, use call stack analysis
2. **Include counts** - Operation counts help identify hotspots
3. **Calculate efficiency** - Compare achieved vs peak performance
4. **Be specific** - Include shapes, kernel names, tile sizes
5. **Provide BOTH optimization paths** - User decides which applies
6. **Vendor-agnostic language** - Use generic terms for all recommendations
7. **Hardware-agnostic analysis** - Don't hardcode GPU specs, use provided values
8. **Focus on bottlenecks + reproducers** - Identify bottlenecks, generate reproducers for kernel teams

---

## Error Handling

### Before Invoking Subagents
- Read `category_manifest.json` to get valid categories
- Only invoke subagents for categories that exist in manifest
- Skip categories with no operations

### After All Subagents Complete
- Check each `<category>_findings.md` for "Status: ERROR"
- Collect list of failed categories and their error summaries
- **CRITICAL: Exclude failed categories from aggregation and recommendations**
- **CRITICAL: Do NOT attempt to manually analyze failed categories**

### In Final Report
- Include Warnings section listing failed categories (only if errors occurred)
- Provide recommendations only for successfully analyzed categories
- If no errors, omit the Warnings section entirely

---

## TraceLens API Reference

```python
# Load trace
from TraceLens.TreePerf import TreePerfAnalyzer
analyzer = TreePerfAnalyzer.from_file(trace_file, add_python_func=True)
tree = analyzer.tree

# Traverse parent chain
tree.traverse_parents_and_print(event)

# Traverse subtree  
tree.traverse_subtree_and_print(event, cpu_op_fields=('Input Dims', 'Input type'))
```

---

## Efficiency Thresholds (General)

| Efficiency | Assessment |
|------------|------------|
| >80% | Excellent |
| 60-80% | Good |
| 40-60% | Acceptable |
| <40% | Needs investigation |

---

## Workflow Tips

1. **Parallel subagent invocation** - Launch all category subagents simultaneously
2. **Wait for completion** - All subagents must finish before aggregation
3. **Load trace ONCE** - Tree data pre-computation is the expensive operation
4. **Context isolation** - Each subagent gets only its data
5. **Single report** - Clean stakeholder report with prioritized recommendations
6. **Handle errors gracefully** - Failed categories go to Warnings, not manual analysis
