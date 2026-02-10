---
name: Standalone Analysis Orchestrator
description: Orchestrate two-tier standalone performance analysis - system-level (CPU/idle, multi-kernel) and compute kernel tiers with independently composable reports
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

Orchestrate modular standalone PyTorch trace analysis using a **two-tier architecture**:
- **System-Level Analysis** (Step 6): CPU/idle time + multi-kernel issues (memcpy, NCCL blocking, overlap)
- **Compute Kernel Analysis** (Step 7): Per-category kernel efficiency (GEMM, SDPA, elementwise, etc.)

**Role**: Load trace once, pre-compute tree data, filter by category, invoke system-level and compute kernel subagents in parallel, aggregate results into independently composable report sections.

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective communication, vendor GEMM library, DNN primitives, GPU graph, etc. Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from traces, it's acceptable to include the actual name for identification.

---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path, Node, Container)
1. Generate Performance Report
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
6. System-Level Analysis (CPU/Idle + Multi-Kernel, PARALLEL) → system_findings/
7. Invoke Compute Kernel Subagents (PARALLEL) → category_findings/
8. Validate Subagent Outputs (system_findings/ + category_findings/)
9. Aggregate Results: System-Level + Compute Kernel Recommendations
10. Generate Replay Artifacts
11. Generate Final Report (composable System + Compute sections)
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
     1. **MI300X**
     2. **MI325X**
     3. **MI355X**
     4. **MI400**

3. **Node Name**
   - Ask: "Which Node should we use for analysis?"

4. **Container Name**
   - Ask: "Which Docker container has TraceLens installed?"

5. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

---

## Step 1: Generate Performance Report

Execute TraceLens CLI in the container:

```bash
ssh <node> "docker exec <container> \
  TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs"
```

This generates:
- `perf_report.xlsx` - Excel report with all sheets
- `perf_report_csvs/` directory with CSV files

---

## Steps 2-5: Prepare Category Data

Execute the Jarvis orchestrator preparation script in the container:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/orchestrator_prepare.py \
  --trace-path <trace_path> \
  --platform <platform> \
  --output-dir <output_dir>"
```

This script performs:
- **Step 2:** Assess GPU utilization (computation, idle, communication times)
- **Step 3:** Identify top 10 operations by GPU time
- **Step 4:** Pre-compute tree data for bottleneck operations (load trace ONCE)
- **Step 4.5:** Pre-compute multi-kernel issue data (memcpy by direction, NCCL events, overlap metrics)
- **Step 5:** Filter and export category-specific data

**Outputs:**
- `category_data/<category>_ops.csv` - Filtered operations per category
- `metadata/<category>_metadata.json` - Platform specs, GPU utilization, config
- `category_data/<category>_tree_data.json` - Pre-computed tree analysis
- `category_data/multi_kernel_data.json` - Memcpy/NCCL/overlap pre-computed data
- `category_data/category_manifest.json` - Workflow metadata with categories (includes `tier` field: `system` or `compute_kernel`)
- `system_findings/` - Directory for system-level analysis outputs
- `category_findings/` - Directory for compute kernel analysis outputs

**Duration:** ~60-90s for tree data pre-computation

---

## Step 6: System-Level Analysis (PARALLEL)

System-level analysis examines issues that affect the GPU pipeline as a whole -- idle time, memory transfer patterns, and communication/compute overlap. These are **not** about individual kernel efficiency.

**Output directory:** `system_findings/`

### 6.1 Read Manifest and Identify System-Level Subagents

```python
import json
with open('<output_dir>/category_data/category_manifest.json') as f:
    manifest = json.load(f)

gpu_util = manifest.get('gpu_utilization', {})
system_categories = [c for c in manifest.get('categories', []) if c.get('tier') == 'system']
```

### 6.2 Launch System-Level Subagents in PARALLEL

Launch **both** sub-agents simultaneously. Do NOT wait between invocations.

**System-Level Subagent Mapping:**

- `cpu_idle` → /cpu-idle-analyzer (invoke if `idle_time_percent > 50%`)
- `multi_kernel` → /multi-kernel-analyzer (invoke if memcpy/NCCL events exist in trace)

**Invocation conditions:**
- **CPU/Idle**: `manifest['cpu_idle_critical'] == True` OR `gpu_util['idle_time_percent'] > 50`
- **Multi-Kernel**: `multi_kernel` category exists in manifest OR `gpu_util['exposed_comm_time_percent'] > 0` OR `gpu_util['exposed_memcpy_time_percent'] > 0`

**CPU/Idle Subagent Invocation:**

```
/cpu-idle-analyzer
- Output directory: <output_dir>
- Node: <node>
- Container: <container>
- Input files: category_data/cpu_idle_ops.csv, metadata/cpu_idle_metadata.json
- Output file: system_findings/cpu_idle_findings.md
```

**Multi-Kernel Subagent Invocation:**

```
/multi-kernel-analyzer
- Output directory: <output_dir>
- Node: <node>
- Container: <container>
- Input files: category_data/multi_kernel_data.json, metadata/multi_kernel_metadata.json
- Output file: system_findings/multi_kernel_findings.md
```

**CRITICAL:** The orchestrator does NOT generate and run any analysis scripts. Each sub-agent is responsible for:
1. Running its Python script inside the container on the node
2. Reading the metrics JSON output
3. Identifying issues and generating findings

### 6.3 Wait for System-Level Subagents to Complete

Both subagents must complete before proceeding to Step 7.
Each writes findings to `system_findings/<name>_findings.md`.

### 6.4 Verify System Outputs

After both subagents complete:

1. Check each findings file in `system_findings/` for "Status: ERROR"
2. Collect failed analyses and error summaries
3. **CRITICAL: Exclude failed analyses from aggregation**
4. **CRITICAL: Do NOT attempt manual analysis of failed system checks**

---

## Step 7: Invoke Compute Kernel Subagents (PARALLEL)

Compute kernel analysis examines individual operation category efficiency. CPU/Idle is NOT in this tier -- it was handled in Step 6.

**Output directory:** `category_findings/`

### 7.1 Read Manifest and Identify Compute Kernel Categories

```python
import json
with open('<output_dir>/category_data/category_manifest.json') as f:
    manifest = json.load(f)

# Only compute kernel tier categories (exclude system tier)
compute_categories = [c for c in manifest.get('categories', [])
                      if c.get('tier') == 'compute_kernel']
```

### 7.2 Launch ALL Compute Kernel Subagents in PARALLEL

For each category in `compute_categories`, invoke the corresponding subagent **simultaneously**.
Do NOT wait between invocations - launch all at once.

**Compute Kernel Subagent Mapping:**

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
- Node: <node>
- Container: <container>
- Input files: category_data/gemm_ops.csv, metadata/gemm_metadata.json, category_data/gemm_tree_data.json
- Output file: category_findings/gemm_findings.md
```

**CRITICAL:** The orchestrator does NOT generate and run any analysis scripts. Each sub-agent is responsible for:
1. Running its Python script inside the container on the node
2. Reading the metrics JSON output
3. Identifying bottlenecks and generating findings

---

### CRITICAL CONSTRAINTS for Compute Kernel Subagents

Include these constraints in EVERY compute kernel subagent invocation prompt:

#### 1. Use GPU Kernel Time for Prioritization

- **ALWAYS** use `gpu_kernel_time_ms` or `Kernel Time (µs)_sum` for bottleneck ranking
- **CPU duration** (`cpu_duration_ms` or `total_duration_us`) is for sync/overhead analysis ONLY
- **NEVER** conflate the two metrics - they measure different things:
  - GPU kernel time = actual GPU execution (optimization target for kernel teams)
  - CPU duration = total operation time including sync, launch overhead
  
```python
# CORRECT: Use GPU kernel time for prioritization
bottleneck_score = gpu_kernel_time_ms * (100 - efficiency_percent)

# WRONG: Using CPU duration for kernel bottleneck analysis
# bottleneck_score = cpu_duration_ms * (100 - efficiency_percent)
```

#### 2. Flag Efficiency Anomalies

- Any efficiency > 100% **MUST** be noted as `[ANOMALY] - verify measurement`
- Do **NOT** use >100% values to claim "excellent performance"
- Report the anomaly but base recommendations on other operations
- Efficiency anomalies indicate:
  - Wrong peak spec for the platform
  - Measurement timing issues
  - Workload characteristics outside normal bounds

```markdown
<!-- CORRECT -->
| Operation | Efficiency | Note |
|-----------|------------|------|
| gemm_1    | 127.3%     | [ANOMALY] Exceeds peak - verify measurement |

<!-- WRONG -->
| Operation | Efficiency | Assessment |
|-----------|------------|------------|
| gemm_1    | 127.3%     | Excellent  |
```

#### 3. Cross-Reference with Manifest Time Breakdown

- Read `time_breakdown` from metadata JSON for context
- Use `gpu_kernel_time_ms` from manifest for category prioritization
- If `has_sync_bottleneck: true`, note this as a separate issue (framework/model, not kernel)

```python
# Read metadata for time breakdown
with open('<output_dir>/metadata/<category>_metadata.json') as f:
    metadata = json.load(f)
    
time_breakdown = metadata.get('time_breakdown', {})
gpu_time = time_breakdown.get('gpu_kernel_time_ms', 0)
sync_time = time_breakdown.get('sync_time_ms', 0)
has_sync = time_breakdown.get('has_sync_bottleneck', False)

if has_sync:
    print("⚠️ Sync bottleneck detected - recommend investigating host-device sync points")
```

#### 4. Sync Time Detection

- If CPU duration >> GPU kernel time (>5x), flag as **sync bottleneck**
- These are model/framework issues, NOT kernel issues
- Recommend: "Investigate host-device synchronization points"
- Do NOT attribute sync time to kernel inefficiency

```markdown
<!-- When sync_time is significant -->
### Sync Bottleneck Detected

**Issue:** Operations show {sync_time}ms sync overhead vs {gpu_time}ms GPU execution
**Root Cause:** Host-device synchronization (e.g., `cudaDeviceSynchronize`, `_local_scalar_dense`)
**Recommendation:** Review model code for unnecessary sync points; consider async execution
**Note:** This is a framework/model issue, not a kernel optimization target
```

#### 5. Output Consistency

- Status must be `SUCCESS` or `ERROR`
- Time values in milliseconds (ms) unless otherwise noted
- Efficiency values as percentages (0-100% typically; flag >100% as anomaly)
- Always include operation count for context

---

**Compute Kernel Subagent Prompt Template:**

When invoking a compute kernel subagent, use this template:

```
You are analyzing {category} operations for a PyTorch trace on {platform}.

**CRITICAL - READ FIRST:**
- Use GPU kernel time (not CPU duration) for all bottleneck analysis
- Flag any efficiency > 100% as "[ANOMALY] - verify measurement"
- If CPU duration >> GPU kernel time (>5x), flag as sync bottleneck (framework issue, not kernel)
- Check metadata time_breakdown for has_sync_bottleneck flag

**Platform Specs:**
- Peak HBM BW: {peak_hbm_bw} TB/s
- Max Achievable TFLOPS (from metadata max_achievable_tflops dict, keyed by compute spec e.g. matrix_bf16, matrix_fp8)

**Input files:**
- category_data/{category}_ops.csv
- metadata/{category}_metadata.json
- category_data/{category}_tree_data.json (if available)

**Output:** category_findings/{category}_findings.md
```

### 7.3 Wait for All Compute Kernel Subagents to Complete

All subagents must complete before proceeding to Step 8.
Each subagent writes its findings to `category_findings/<category>_findings.md`.

### 7.4 Verify Outputs and Collect Errors

After all subagents complete:

1. Check each `<category>_findings.md` for "Status: ERROR"
2. Collect list of failed categories and their error summaries
3. **CRITICAL: Exclude failed categories from aggregation and recommendations**
4. **CRITICAL: Do NOT attempt to manually analyze failed categories**

---

## Step 8: Validate Subagent Outputs

Before aggregating results, validate outputs from **both** tiers (system_findings/ and category_findings/):

### 1. Time Sanity Check

```python
import json
import os

# Load manifest with ground truth
with open('<output_dir>/category_data/category_manifest.json') as f:
    manifest = json.load(f)

# Sum of compute kernel GPU kernel times should ~= computation time
total_category_time = sum(
    cat.get('gpu_kernel_time_ms', 0) 
    for cat in manifest['categories']
    if cat.get('tier') == 'compute_kernel'
)

computation_time = manifest['gpu_utilization']['total_time_ms'] * \
                   manifest['gpu_utilization']['computation_time_percent'] / 100

discrepancy = abs(total_category_time - computation_time) / computation_time * 100 if computation_time > 0 else 0
if discrepancy > 15:
    print(f"⚠️ Time discrepancy: Category sum ({total_category_time:.1f}ms) vs " +
          f"Computation time ({computation_time:.1f}ms) = {discrepancy:.1f}% difference")
```

### 2. Efficiency Anomaly Check

Scan compute kernel findings for efficiency values > 100%:

```python
import re

findings_dir = '<output_dir>/category_findings/'
anomalies = []

for f in os.listdir(findings_dir):
    if f.endswith('_findings.md'):
        with open(os.path.join(findings_dir, f)) as file:
            content = file.read()
            matches = re.findall(r'(\d{3,}\.?\d*)\s*%', content)
            for m in matches:
                if float(m) > 100:
                    anomalies.append({
                        'category': f.replace('_findings.md', ''),
                        'value': f"{m}%"
                    })

if anomalies:
    print("⚠️ Efficiency anomalies detected (>100%):")
    for a in anomalies:
        print(f"  - {a['category']}: {a['value']}")
    print("Note: Anomalies indicate measurement issues - do not use for prioritization")
```

### 3. Coverage Check

Verify all expected categories have findings in the correct directories:

```python
# System-level coverage
system_findings_dir = '<output_dir>/system_findings/'
expected_system = [c['name'] for c in manifest['categories'] if c.get('tier') == 'system']
found_system = [f.replace('_findings.md', '') 
                for f in os.listdir(system_findings_dir) 
                if f.endswith('_findings.md')]
missing_system = set(expected_system) - set(found_system)
if missing_system:
    print(f"⚠️ Missing system findings for: {', '.join(missing_system)}")

# Compute kernel coverage
category_findings_dir = '<output_dir>/category_findings/'
expected_compute = [c['name'] for c in manifest['categories'] if c.get('tier') == 'compute_kernel']
found_compute = [f.replace('_findings.md', '') 
                 for f in os.listdir(category_findings_dir) 
                 if f.endswith('_findings.md')]
missing_compute = set(expected_compute) - set(found_compute)
if missing_compute:
    print(f"⚠️ Missing compute kernel findings for: {', '.join(missing_compute)}")
```

### 4. Priority Consistency Check

Ensure categories with highest GPU kernel time % are given highest priority in the compute kernel tier:

```python
sorted_cats = sorted(
    [c for c in manifest['categories'] if c.get('tier') == 'compute_kernel'],
    key=lambda x: x.get('gpu_kernel_time_ms', 0),
    reverse=True
)

top_time_categories = [c['name'] for c in sorted_cats[:3]]
print(f"Top 3 compute kernel categories by GPU time: {top_time_categories}")
print("Verify these receive P1-P3 in compute kernel recommendations")
```

### Validation Output

```markdown
## Validation Summary
- Time Check: [PASS/WARN] - Compute kernel sum vs computation time discrepancy
- Efficiency Check: [PASS/WARN] - Anomalies > 100%
- System Coverage: [PASS/WARN] - Missing system findings
- Compute Coverage: [PASS/WARN] - Missing compute kernel findings
- Priority Check: [INFO] - Top time categories for priority verification
```

---

## Step 9: Aggregate Results -- Two-Tier Recommendations

### Read Findings from BOTH Tiers

```python
import os
import json

# --- System-level findings ---
system_findings = {}
failed_system = []
system_dir = '<output_dir>/system_findings/'

for f in os.listdir(system_dir):
    if f.endswith('_findings.md'):
        with open(os.path.join(system_dir, f)) as file:
            content = file.read()
            name = f.replace('_findings.md', '')
            if 'Status: ERROR' in content:
                failed_system.append({'category': name, 'content': content})
            else:
                system_findings[name] = content

# --- Compute kernel findings ---
compute_findings = {}
failed_compute = []
compute_dir = '<output_dir>/category_findings/'

for f in os.listdir(compute_dir):
    if f.endswith('_findings.md'):
        with open(os.path.join(compute_dir, f)) as file:
            content = file.read()
            name = f.replace('_findings.md', '')
            if 'Status: ERROR' in content:
                failed_compute.append({'category': name, 'content': content})
            else:
                compute_findings[name] = content

# Load manifest for top operations and metadata
with open('<output_dir>/category_data/category_manifest.json', 'r') as f:
    manifest = json.load(f)
    top_ops = manifest.get('top_operations', [])
```

### Aggregate System-Level Recommendations

System-level recommendations address pipeline/framework issues that affect ALL operations.

**System-Level Prioritization:**

Assign priorities sequentially starting from P1 based on which analyses are present. If CPU/Idle is skipped, multi-kernel issues start at P1.

| Order | Source | Criteria | Included When |
|-------|--------|----------|---------------|
| First | CPU/Idle | `idle_time_percent > 30%` | Only if CPU/Idle analysis was invoked |
| Next | Multi-Kernel | Highest severity multi-kernel issue (memcpy/NCCL blocking/overlap) | Only if severity is not NONE/N/A |
| Next | Multi-Kernel | Next severity multi-kernel issue | If additional actionable issues exist |

**CRITICAL: No-Issue Handling:**
- If **all** system-level analyses report NONE/N/A severity (no actionable issues), do **NOT** generate any P1/P2/P3 recommendations for the System-Level Optimizations section.
- Instead, display a short summary: "No system-level bottlenecks detected. GPU utilization is healthy at X%, with negligible memcpy/communication overhead."
- This avoids misleading stakeholders with a red P1 icon for a non-issue.

**System-Level Icon Mapping (by priority number, NOT severity):**

| Priority | Icon |
|----------|------|
| P1 | 🔴 |
| P2 | 🟡 |
| P3+ | 🟢 |

### Aggregate Compute Kernel Recommendations

Compute kernel recommendations address individual operation efficiency. Each subagent has produced algorithmic and kernel optimization recommendations. Consolidate these, cross-reference with `top_ops`, and prioritize by impact.

**Compute Kernel Prioritization:**

Select kernels in top_ops and order by highest impact on end-to-end time. Focus on areas with low efficiency and high category time.

| Priority | Icon | Criteria |
|----------|------|----------|
| P1 | 🔴 | In top_ops + (Sorted by efficiency and high category %) |
| P2 | 🟡 | In top_ops + (Sorted by efficiency and high category %) |
| P3+ | 🟢 | In top_ops + (Sorted by efficiency and high category %) |

**Compute Kernel Icon Mapping (by priority number, NOT severity):**

| Priority | Icon |
|----------|------|
| P1 | 🔴 |
| P2 | 🟡 |
| P3+ | 🟢 |

---

## Step 10: Generate Replay Artifacts

**When to Generate:**
1. Op is a significant bottleneck (>10% of compute)
2. Efficiency is notably low (<30% of peak)
3. Kernel team needs a minimal reproducer

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/generate_replay_artifacts.py \
  --output-dir <output_dir> \
  --perf-report-path <output_dir>/perf_report.xlsx \
  --op-names <op1> <op2> <op3>"
```

---

## Step 11: Generate Final Report

Create `standalone_analysis.md` in `<output_dir>`. The report uses a **two-section structure**: System-Level Optimizations and Compute Kernel Optimizations. Each section is independently composable and can stand alone as a deliverable.

Validate the report before sharing the priority recommendations on the chat and prompt the user to review the report.

**Purpose:** Stakeholder report with prioritized recommendations in two tiers

```markdown
# <Model> - <Platform> Standalone Analysis

## Executive Summary
[1 paragraph overview + key metrics table]

| Metric | Value |
|--------|-------|
| Total Compute Time | X ms |
| GPU Utilization | Y% |
| Idle Time | Z% |
| Exposed Communication | W% |
| Top Bottleneck Category | Category (V%) |

## Warnings

**Include this section ONLY if any subagent failed:**

The following analyses could not be completed due to script failures:

| Analysis | Tier | Error Summary |
|----------|------|---------------|
| <name> | System / Compute Kernel | <brief error description> |

These are excluded from the recommendations below.

---

## Top Operations

| Rank | Operation | Category | Time (ms) | % of Total Compute |
|------|-----------|----------|-----------|-------------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

---

## System-Level Optimizations

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

<!-- CONDITIONAL: If NO actionable system-level issues found (all severities are NONE/N/A), use the clean template below. -->
<!-- Otherwise, number priorities sequentially starting from P1. Include CPU/Idle only if invoked. -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Title format: Descriptive name only. Do NOT append severity labels like (CRITICAL) or (MEDIUM). -->

<!-- === TEMPLATE A: No actionable system-level issues === -->
<!-- Use this when all system-level analyses report NONE/N/A severity -->

✅ No system-level bottlenecks detected. GPU utilization is healthy at X% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

<!-- === TEMPLATE B: Actionable issues found === -->
<!-- Use this when at least one system-level analysis reports an actionable severity (LOW/MEDIUM/HIGH/CRITICAL) -->

### 🔴 P1: <CPU/Idle Title OR Multi-Kernel Issue Title>

**Issue**: [1-2 sentences - what's wrong]

**Action**: [1-2 sentences - what to do]

**Impact**: [Expected improvement]

→ *See [Detailed Analysis: System-Level > CPU/Idle Time](#cpu-idle-time-analysis) for details* OR → *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

---

### 🟡 P2: <Multi-Kernel Issue Title>

**Issue**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

**Impact**: [Expected improvement]

→ *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

---

### 🟢 P3: <Next Multi-Kernel Issue>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [Expected improvement]

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).
Summaries of recommendations from Step 7 sub-agents, focused on individual kernel efficiency.

<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->

### 🔴 P1: <Brief Title>

**Issue**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

**Impact**: [Expected improvement]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟡 P2: <Brief Title>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [Expected improvement]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟢 P3: <Brief Title>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [Expected improvement]

---

## Detailed Analysis: System-Level

### 1. CPU/Idle Time Analysis
[Full cpu_idle_findings.md content from system_findings/]

### 2. Multi-Kernel Issues
[Full multi_kernel_findings.md content from system_findings/]

---

## Detailed Analysis: Compute Kernels

### 1. <Operation Category> (X% of compute)
[All kernel breakdowns, calculations, tables, explanations from category_findings/]

### 2. <Operation Category> (X% of compute)
[...]

---

## Appendix

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)

### Replay Artifacts
[List of generated replay packages if any]
```

**Key formatting rules:**
1. **Warnings section**: Only include if there were errors; omit entirely if all succeeded
2. **Executive Summary**: Max ~20 lines
3. **System-Level Optimizations**: If all system-level analyses report no actionable issues (NONE/N/A severity), use a single "✅ No system-level bottlenecks detected" summary instead of P1/P2/P3 recommendations. Only generate numbered priorities when at least one actionable issue exists (Number sequentially from P1, including CPU/Idle first if invoked)
4. **Compute Kernel Optimizations**: P1-P3+ from category subagent findings
5. **Each section is independently composable** -- can be shared standalone
6. **System and Compute tiers use separate sequential P1/P2/P3 numbering (no gaps)**
7. **Priority icons are assigned by PRIORITY NUMBER, not severity:**
   - **System-Level:** 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ... (only when actionable issues exist)
   - **Compute Kernel:** 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ...
8. **Detailed Analysis**: Split into System-Level and Compute Kernels subsections. Always include the Detailed Analysis: System-Level section with full metrics even when no actionable issues exist.
9. **No redundancy**: Information appears in ONE place only
10. **Recommendations**: Max ~10 lines PER recommendation

---

## Key Principles

1. **Always verify with TraceLens** - Don't assume sources, use call stack analysis
2. **Include counts** - Operation counts help identify hotspots
3. **Calculate efficiency** - Compare achieved vs peak performance
4. **Be specific** - Include shapes, kernel names, tile sizes
5. **Provide multiple optimization paths** - User decides which applies
6. **Vendor-agnostic language** - Use generic terms for all recommendations
7. **Hardware-agnostic analysis** - Don't hardcode GPU specs, use provided values
8. **Focus on bottlenecks + reproducers** - Identify bottlenecks, generate reproducers for kernel teams

---

## Error Handling

### Before Invoking Subagents
- Read `category_data/category_manifest.json` to get valid categories
- Use `tier` field to determine which subagents belong to Step 6 (system) vs Step 7 (compute kernel)
- Only invoke subagents for categories that exist in manifest
- Skip categories with no operations

### After System-Level Subagents Complete (Step 6)
- Check each file in `system_findings/` for "Status: ERROR"
- Collect failed analyses and error summaries
- **CRITICAL: Exclude failed analyses from aggregation**
- **CRITICAL: Do NOT attempt to manually analyze failed system checks**

### After Compute Kernel Subagents Complete (Step 7)
- Check each file in `category_findings/` for "Status: ERROR"
- Collect list of failed categories and their error summaries
- **CRITICAL: Exclude failed categories from aggregation and recommendations**
- **CRITICAL: Do NOT attempt to manually analyze failed categories**

### In Final Report
- Include Warnings section listing failed analyses from BOTH tiers (only if errors occurred)
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

1. **Two-tier parallelism** - Launch system-level (Step 6) and compute kernel (Step 7) subagents in parallel within each tier
2. **Wait for completion** - All subagents in both tiers must finish before validation (Step 8)
3. **Load trace ONCE** - Tree data and multi-kernel data pre-computation happen in a single trace load
4. **Context isolation** - Each subagent gets only its data; system subagents read from `system_findings/`, compute from `category_findings/`
5. **Composable reports** - System-Level and Compute Kernel sections can stand alone as independent deliverables
6. **Sequential priority numbering per tier** - System and Compute tiers each number P1/P2/P3 independently with no gaps (if CPU/Idle is skipped, multi-kernel starts at P1). Icons follow priority number: System 🔴→🟡→🟢, Compute 🔴→🟡→🟢
7. **Handle errors gracefully** - Failed analyses go to Warnings, not manual analysis
