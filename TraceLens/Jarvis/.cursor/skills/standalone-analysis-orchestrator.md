---
name: Standalone Analysis Orchestrator
description: Orchestrate modular standalone performance analysis - queries user, generates reports, pre-computes tree data, invokes category skills
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

Orchestrate modular standalone PyTorch trace analysis. Coordinate workflow, pre-compute expensive operations, invoke category-specific skills.

**Role**: Load trace once, pre-compute tree data, filter by category, invoke skills, aggregate results, generate dual reports.

---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path, Cluster, Container)
1. Generate Performance Report
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Category Filtering)
6. Invoke Category-Specific Skills
7. Determine Optimization Paths
8. Generate Replay Artifacts (Optional - Path B)
9. Generate Dual Reports
```

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Trace File Path**
   - Ask: "Please provide the full path to your PyTorch trace file (.json or .json.gz)"
   - Example: `/home/user/traces/model_trace.json`

2. **Platform**
   - Ask: "Which AMD platform are you analyzing?"
   - Options:
     1. **MI300X** - 5.3 TB/s HBM, 708 TFLOPS BF16, 192 GB
     2. **MI325X** - 6.0 TB/s HBM, 708 TFLOPS BF16, 256 GB
     3. **MI355X** - 6.5 TB/s HBM, 850 TFLOPS BF16, 288 GB
     4. **MI400** - 7.0 TB/s HBM, 1000 TFLOPS BF16, 320 GB

3. **Cluster Name**
   - Ask: "Which cluster should we use for analysis? (e.g., tw008)"

4. **Container Name**
   - Ask: "Which Docker container has TraceLens installed? (e.g., multimodal_qwen_3)"

5. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

**Example prompt format:**
```
To run standalone analysis, I need the following information:

1. Trace file path: _____
2. Platform (MI300X/MI325X/MI355X/MI400): _____
3. Cluster name: _____
4. Container name: _____
5. Output directory (optional): _____
```

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
- `perf_report_csvs/` directory with CSV files including `gpu_timeline.csv`, `unified_perf_summary.csv`

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
- `category_manifest.json` - Workflow metadata

**Duration:** ~60-90s for tree data pre-computation

---

## Step 6: Invoke Category-Specific Skills for operations

For each category with data (from `category_manifest.json`), invoke the corresponding skill:

- `gemm` → @gemm-analysis
- `sdpa_fwd` → @sdpa-analysis
- `elementwise` → @elementwise-analysis
- `reduce` → @reduce-analysis
- `triton` → @triton-analysis
- `moe_fused` → @moe-analysis
- `batchnorm` → @batchnorm-analysis
- `convolution` → @convolution-analysis
- `other` → @generic-op-analysis

**How Skills Work:**
1. Each skill runs its Python analysis script (outputs markdown to stdout)
2. LLM interprets the markdown output
3. LLM validates bottlenecks and determines optimization paths
4. LLM writes findings to `category_findings/<category>_findings.md`

**Duration:** ~5-10s per category (script execution + LLM interpretation)

---

## Step 7: Aggregate and Determine Optimization Paths

### Read All Category Findings and Top Operations

Load all findings files and the category manifest (which includes top operations):

```python
import os
import json

# Load category findings
findings_summaries = {}
findings_dir = '<output_dir>/category_findings/'
for f in os.listdir(findings_dir):
    if f.endswith('_findings.md'):
        with open(os.path.join(findings_dir, f)) as file:
            findings_summaries[f] = file.read()

# Load category manifest (includes top operations from Step 3)
with open('<output_dir>/category_manifest.json', 'r') as f:
    manifest = json.load(f)
    top_ops = manifest.get('top_operations', [])
```

**Use top operations data for prioritization:**
- The `top_operations` list from Step 3 shows the highest time operations across ALL categories
- Cross-reference category findings with top_ops to identify high-impact bottlenecks
- Prioritize recommendations that address operations in the top_ops list

### Provide Recommendations for BOTH Paths

**CRITICAL**: Present recommendations for both optimization approaches:

#### Path A: Fusion / Algorithmic Changes
*For when user CAN modify model or use different PyTorch APIs*

- Flash Attention for unfused attention patterns (3-10x speedup)
- torch.compile for kernel fusion opportunities
- Custom fused kernels (e.g., fused layer norm, fused MLP)
- Algorithmic changes (e.g., RMSNorm instead of LayerNorm)
- Batching small operations together (e.g., tiny batched GEMMs)
- Memory layout changes (NCHW → NHWC for convolutions)

#### Path B: Kernel Optimization Only
*For when user MUST keep same torch code and can only improve kernels*

- Generate replay artifacts for kernel team to investigate
- Identify suboptimal kernel selections
- Flag tile size issues or inefficient kernel launches
- Recommend tuning specific kernel parameters
- Note memory access pattern issues

**Always present both paths** - let user decide which applies to their situation.

### Prioritize Recommendations Using Top Operations

**Critical: Cross-reference category bottlenecks with top_ops from Step 3**

For each bottleneck identified in category findings:
1. **Check if it appears in top_ops list** (from Step 3 / category_manifest.json)
2. **If yes:** This is a high-priority bottleneck (affects overall compute time)
3. **If no:** Still important within its category, but lower overall impact

**Prioritization Framework:**

| Priority | Criteria |
|----------|----------|
| 🔴 **Critical** | In top_ops + Low efficiency (<30%) + High category % (>15%) |
| 🟡 **High** | In top_ops OR (Low efficiency <40% + >10% category time) |
| 🟢 **Medium** | >5% category time OR notable optimization pattern |
| ⚪ **Low** | Everything else |

### Estimate Optimization Impact

For each recommendation, provide impact ranges to help prioritize:

**Impact Projection Framework:**

```markdown
**Impact Projection** (if op improves to X% of peak):

| Target Efficiency | Time | E2E Improvement |
|-------------------|------|-----------------|
| 20% of peak | 9.5 ms | ~49% faster for this op |
| 50% of peak | 3.8 ms | ~50% faster for this op |
```

**Calculate E2E Impact:**
- Use operation's % of total compute time (from top_ops or category data)
- Example: If op is 10% of compute and improves 50% → 5% E2E speedup

This acknowledges uncertainty while still enabling prioritization.

**For Path A (fusion):** Calculate expected benefit using TraceLens perf models where available.

**For Path B (kernel optimization):** Estimate ceiling:
- Memory-bound ops: What's the gap to peak HBM bandwidth?
- Compute-bound ops: What's the gap to peak MAF?

---

## Step 8: Generate Replay Artifacts (Optional - Path B)

**When to Generate Replay Artifacts:**
1. Op is a significant bottleneck (>10% of compute)
2. Efficiency is notably low (<30% of peak)
3. Kernel team needs a minimal reproducer

**Frame it as**: "Replay artifact recommended for kernel team to investigate and optimize."

For significant bottlenecks, generate replay artifacts:

```bash
ssh <cluster> "docker exec <container> python3 \
  TraceLens/Jarvis/generate_replay_artifacts.py \
  --output-dir <output_dir> \
  --perf-report-path <output_dir>/perf_report.xlsx \
  --op-names <op1> <op2> <op3>"
```

Replace `<op1>`, `<op2>`, etc. with the specific operation names to generate replay artifacts for.

**Outputs:** Creates `<output_dir>/replay_packages/<op_name>_replay_package.zip` for each operation

---

## Step 9: Generate Final Reports

Create two reports in `<output_dir>`:

### 1. `standalone_analysis_rough.md`

**Purpose:** Working notes, process documentation, raw data

Document the analysis process:
- Analysis steps taken
- Raw data and calculations (tables from category findings)
- Tree analysis performed (parent chains reviewed)
- TraceLens gaps encountered
- Questions for further investigation

**Structure:**
```markdown
# <Model> - <Platform> Standalone Analysis (Rough)

## Analysis Process Summary
### Step 1: Identify Traces & Setup
[Document user inputs, platform selection]

### Step 2: Generate Reports
[Document TraceLens report generation]

### Step 3-5: Prepare Category Data
[Document GPU utilization, top ops, tree data pre-computation]

### Step 6: Category Analyses
[Document each category analysis invocation and key observations]

### Step 7: Optimization Paths Determination
[Document how Path A and Path B recommendations were derived]

## Raw Data Exploration
[All the data tables from category findings]

## Tree Analysis
[Parent chains explored, subtrees reviewed]

## TraceLens Gaps
[Missing features that would have helped]

## Questions for Further Investigation
[Open questions, areas needing more data]
```

### 2. `standalone_analysis_fair.md`

**Purpose:** Clean stakeholder report with prioritized recommendations

**Structure:**
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
| ...additional key metrics... |

### Top Operations (from Step 3)
**These operations consume the most GPU time - prioritize optimizations here**

| Rank | Operation | Category | Time (ms) | % of Total Compute |
|------|-----------|----------|-----------|-------------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

---

## Recommendations

**Note:** Recommendations are prioritized based on:
1. Presence in Top Operations list (highest impact)
2. Efficiency gap (how much improvement is possible)
3. Category compute percentage

### 🔴 Critical Priority: <Brief Title>
**Operation**: [Name - from top_ops if applicable]
**Issue**: [1 sentence - what's wrong]
**Action**: [1-2 sentences - what to do]
**Impact**: [Expected E2E improvement - calculated from % of compute]
→ *See [Detailed Analysis: Section](#section-link) for details*

---

### 🟡 High Priority: <Brief Title>
**Operation**: [Name - from top_ops if applicable]
[Same brief format]

---

### 🟢 Medium Priority: <Brief Title>
[Same brief format]

---

## Detailed Analysis

### 1. <Category Name> (X% of compute)
[Paste relevant content from category findings]
[All kernel breakdowns, calculations, tables, explanations]

### 2. <Category Name> (Y% of compute)
[...]

---

## Appendix

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF**: Y TFLOPS
- **Memory**: Z GB

### Replay Artifacts
[List of generated replay packages if any]
```

**Key formatting rules for Fair reports:**
1. **Executive Summary**: Max ~20 lines - metrics table + bottleneck ranking
2. **Recommendations**: Max ~10 lines PER recommendation - Issue/Action/Impact only
3. **Detailed Analysis**: All kernel breakdowns, math, explanations go HERE
4. **No redundancy**: Information appears in ONE place only
5. **Cross-references**: Recommendations link to detailed sections

---

## Key Principles

1. **Always verify with TraceLens** - Don't assume sources, use call stack analysis
2. **Include counts** - Operation counts help identify hotspots
3. **Calculate efficiency** - Compare achieved vs peak performance
4. **Be specific** - Include shapes, kernel names, tile sizes
5. **Provide BOTH optimization paths** - User decides which applies
6. **Neutral comparative framing** - Focus on improving target, not declaring winners
7. **Hardware-agnostic analysis** - Don't hardcode GPU specs, use provided values
8. **Separate expected vs unexpected differences** - Hardware limits vs software issues
9. **Focus on bottlenecks + reproducers** - JARVIS role is to identify performance bottlenecks and generate minimal reproducers for kernel teams, not to diagnose root causes

### What You CAN Infer from Traces

| Observable | Source |
|------------|--------|
| Kernel names | `trunc_kernel_details` column |
| Kernel durations | Trace events |
| Input shapes | `Input Dims` column |
| Achieved TB/s, TFLOPS/s | Calculated from duration + data moved |
| Efficiency % | Achieved / Peak |
| Call stack | TreePerfAnalyzer |
| Kernel lowering differences | Compare kernel breakdown between shapes |

### What You CANNOT Infer (Avoid Speculation)

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Bank conflicts | Requires hardware counters (rocprof/nsight) | "Low efficiency - profile with rocprof to diagnose" |
| Memory coalescing | Requires hardware counters | "Replay artifact provided for kernel team investigation" |
| Occupancy | Requires hardware counters | "Kernel running slower than expected" |
| Cache hit rates | Requires hardware counters | "Large working set may exceed cache" |
| Specific root causes | Traces show WHAT, not WHY | "Bottleneck identified - generate reproducer for kernel team" |

**Key principle**: JARVIS identifies bottlenecks and generates reproducers. Root cause diagnosis requires profiling tools (rocprof, nsight-compute) on the replay artifacts.

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

# Find events by name
for evt in tree.events:
    if 'softmax' in evt.get('name', ''):
        # process event

# SDPA FLOPS calculation
from TraceLens.PerfModel.perf_model import SDPA
flops = SDPA.flops_func(B, N_Q, H_Q, N_KV, H_KV, d_h_qk, d_h_v, causal)
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

1. **Load trace ONCE** (Step 4) - Tree data pre-computation is the expensive operation
2. **Context isolation** - Each skill gets only its data, preventing context pollution
3. **Dual reports** - Rough for process documentation, Fair for stakeholders
4. **Priority sorting** - Aggregate recommendations by impact
5. **Handle missing categories gracefully** - Not all traces have all operation types
6. **Use relative paths** - Scripts use paths relative to TraceLens repo root
