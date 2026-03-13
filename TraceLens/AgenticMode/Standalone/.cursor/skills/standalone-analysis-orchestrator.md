<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

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

## Shell Permissions for Remote Commands
 
**All `ssh` commands** (and any shell commands that connect to the node/container) should first be attempted with `required_permissions: ["full_network"]`. If the command fails due to SSH authentication or agent access issues (e.g., "Permission denied", "Connection refused", or the command is rejected/aborted by the sandbox), ask the user for permission to retry with `required_permissions: ["all"]`, explaining that full sandbox access is needed for SSH agent and config access. Once the user approves `"all"` permissions, use `"all"` for all subsequent SSH commands in the session without asking again.
 
---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path, Environment Setup)
1. Generate Performance Report
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
6. System-Level Analysis (CPU/Idle + Multi-Kernel, PARALLEL) → system_findings/
7. Invoke Compute Kernel Subagents (PARALLEL) → category_findings/
8. Validate Subagent Outputs (system_findings/ + category_findings/)
9. Aggregate Results: System-Level + Compute Kernel Recommendations
10. Generate Final Report (composable System + Compute sections)
10.1. Generate and Embed Performance Improvement Plot (single atomic call: plot_data + matplotlib PNG + base64 embed)
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
     3. **MI350X**
     4. **MI355X**
     5. **MI400**

3. **Environment Setup**
   - Ask: "Are you running locally or on a cluster?"
     - If **local**: No further environment questions — prefix is blank (commands run directly).
     - If **cluster**:
       - Ask "Which node should we use?" → `<node>`
       - Ask "Are you working in a containerized environment (e.g. Docker)?" → if yes, ask for container name → `<container>`
       - Ask "Are you using a virtual environment?" → if yes, ask for venv path (e.g. `/opt/venvs/tracelens`) → `<venv_path>`

4. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

### Build and Cache Command Prefix

After collecting inputs, build a command template and save it to `<output_dir>/cache/cmd_prefix.txt`. Create the directory with `mkdir -p <output_dir>/cache`.

The template uses `{CMD}` as a placeholder for the actual command.

**Cluster:** Before building the prefix, locate the TraceLens project root on the remote environment:

1. SSH into the node (and enter the container if applicable).
2. Run `find / -maxdepth 5 -type d -name "TraceLens" 2>/dev/null | head -5` to find candidate directories.
3. Confirm the correct one (it should contain `TraceLens/AgenticMode/`). Store as `<tracelens_dir>`.

Build the cluster prefix using this lookup:

| Container | Venv | Template |
|-----------|------|----------|
| No | No | `ssh <node> "cd <tracelens_dir> && {CMD}"` |
| Yes | No | `ssh <node> "docker exec <container> bash -c 'cd <tracelens_dir> && {CMD}'"` |
| No | Yes | `ssh <node> "bash -c 'source <venv_path>/bin/activate && cd <tracelens_dir> && {CMD}'"` |
| Yes | Yes | `ssh <node> "docker exec <container> bash -c 'source <venv_path>/bin/activate && cd <tracelens_dir> && {CMD}'"` |

Write the resolved template (with actual node/container/venv/tracelens_dir values substituted) to `<output_dir>/cache/cmd_prefix.txt`.

### Command Execution Pattern

**Before executing any command**, read `<output_dir>/cache/cmd_prefix.txt`. It contains a template with a `{CMD}` placeholder. Substitute `{CMD}` with the actual command. All commands below use `<prefix>` to represent this resolved template.

---

## Step 1: Generate Performance Report

Execute TraceLens CLI:

```bash
<prefix> TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs \
  --enable_pseudo_ops
```

This generates:
- `perf_report.xlsx` - Excel report with all sheets
- `perf_report_csvs/` directory with CSV files

---

## Steps 2-5: Prepare Category Data

Execute the TraceLens Agentic Mode orchestrator preparation script:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path <trace_path> \
  --platform <platform> \
  --output-dir <output_dir>
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

---

## Agent File Map

The following maps category names to their agent definition files. These files contain specialized analysis instructions, severity thresholds, root cause patterns, and finding templates for each category.

**Base path:** `TraceLens/AgenticMode/Standalone/.cursor/agents/`

| Category | Agent File |
|----------|-----------|
| cpu_idle | cpu-idle-analyzer.md |
| multi_kernel | multi-kernel-analyzer.md |
| gemm | gemm-analyzer.md |
| sdpa_fwd | sdpa-analyzer.md |
| moe_fused | moe-analyzer.md |
| elementwise | elementwise-analyzer.md |
| triton | triton-analyzer.md |
| reduce | reduce-analyzer.md |
| norm | norm-analyzer.md |
| convolution | convolution-analyzer.md |
| other | generic-op-analyzer.md |

---

## Step 6: System-Level Analysis (PARALLEL)

System-level analysis examines issues that affect the GPU pipeline as a whole -- idle time, memory transfer patterns, and communication/compute overlap. These are **not** about individual kernel efficiency.

**Output directory:** `system_findings/`

### 6.1 Read Manifest and Identify System-Level Subagents

```bash
<prefix> python3 -c "
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_manifest_categories
load_manifest_categories('<output_dir>')
"
```

This prints `system_categories` and `compute_categories` lists. Use `system_categories` for Step 6 and `compute_categories` for Step 7.

### 6.2 Launch System-Level Subagents in PARALLEL

Launch **both** sub-agents simultaneously using the Task tool. Do NOT wait between invocations.

**For each system-level subagent:**

1. **Read** the agent definition file from the Agent File Map:
   `TraceLens/AgenticMode/Standalone/.cursor/agents/<agent-file>.md`

2. **Launch a Task subagent** (subagent_type: generalPurpose) with a prompt that includes:
   - The **full contents** of the agent definition file (this provides the specialized
     analysis instructions, severity thresholds, and finding templates)
   - The execution context appended after the agent instructions

**System-Level Subagent Mapping:**

- `cpu_idle` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/cpu-idle-analyzer.md` (invoke if `idle_flagged` is `true`)
- `multi_kernel` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/multi-kernel-analyzer.md` (invoke if memcpy/NCCL events exist in trace)

**Invocation conditions:**
- **CPU/Idle**: Read `category_data/cpu_idle_metrics.json` and check `idle_flagged`. Only invoke the subagent if `idle_flagged` is `true` (idle > 15%). Skip if `false` -- the deterministic script already captured the factual data.
- **Multi-Kernel**: `multi_kernel` category exists in manifest OR `gpu_util['exposed_comm_time_percent'] > 0` OR `gpu_util['exposed_memcpy_time_percent'] > 0`

**Task prompt structure for each subagent:**

```
Read the agent file first:
  TraceLens/AgenticMode/Standalone/.cursor/agents/<agent-file>.md

Then launch a Task subagent with the following prompt:

---BEGIN AGENT INSTRUCTIONS---
<full contents of the agent .md file>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- Input files: <list from agent file's "Input files" section>
- Output file: <from agent file's "Output file" section>

Follow the agent instructions above to complete the analysis.
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

Use `compute_categories` from the `load_manifest_categories()` call in Step 6.1 (no need to re-read the manifest).

### 7.2 Launch ALL Compute Kernel Subagents in PARALLEL

For each category in `compute_categories`, launch the corresponding subagent **simultaneously** using the Task tool. Do NOT wait between invocations - launch all at once.

**For each compute kernel subagent:**

1. **Read** the agent definition file from the Agent File Map:
   `TraceLens/AgenticMode/Standalone/.cursor/agents/<agent-file>.md`

2. **Launch a Task subagent** (subagent_type: generalPurpose) with a prompt that includes:
   - The **full contents** of the agent definition file
   - The execution context
   - The CRITICAL CONSTRAINTS (from Section 7, below)

**Compute Kernel Subagent Mapping:**

- `gemm` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/gemm-analyzer.md`
- `sdpa_fwd` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/sdpa-analyzer.md`
- `elementwise` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/elementwise-analyzer.md`
- `reduce` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/reduce-analyzer.md`
- `triton` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/triton-analyzer.md`
- `moe_fused` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/moe-analyzer.md`
- `norm` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/norm-analyzer.md`
- `convolution` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/convolution-analyzer.md`
- `other` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/generic-op-analyzer.md`

**Task prompt structure for each subagent:**

```
Read the agent file first:
  TraceLens/AgenticMode/Standalone/.cursor/agents/<agent-file>.md

Then launch a Task subagent with the following prompt:

---BEGIN AGENT INSTRUCTIONS---
<full contents of the agent .md file>
---END AGENT INSTRUCTIONS---

**CRITICAL CONSTRAINTS:**
<include the constraints from the orchestrator skill Section 7 below>

**Execution Context:**
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- Input files: category_data/<category>_ops.csv, metadata/<category>_metadata.json,
  category_data/<category>_tree_data.json (if available)
- Output file: category_findings/<category>_findings.md

Follow the agent instructions above to complete the analysis.
```

**CRITICAL:** The orchestrator does NOT generate and run any analysis scripts. Each sub-agent is responsible for:
1. Running its Python script inside the container on the node
2. Reading the metrics JSON output
3. Identifying bottlenecks and generating findings

---

### CRITICAL CONSTRAINTS for Compute Kernel Subagents

Include these constraints in EVERY compute kernel subagent invocation prompt:

#### 1. Flag Efficiency Anomalies

- Any efficiency > 100% **MUST** be noted as `[ANOMALY] - verify measurement`
- Do **NOT** use > 100% values to claim "excellent performance"
- Report the anomaly but base recommendations on other operations
- Efficiency anomalies indicate:
  - Wrong peak spec for the platform
  - Measurement timing issues
  - Workload characteristics outside normal bounds

#### 2. Output Consistency

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
- When citing peak TFLOPS, use each operation's `efficiency.resolved_peak_maf` from the metrics JSON (precision-specific)

**Platform Specs:**
- Peak HBM BW: {peak_hbm_bw} TB/s
- Resolved Peak MAF: Each operation's `efficiency.resolved_peak_maf` has the precision-correct peak — use this when citing peak TFLOPS

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

Before aggregating results, validate outputs from **both** tiers (system_findings/ and category_findings/).

```bash
<prefix> python3 -c "
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_subagent_outputs
validate_subagent_outputs('<output_dir>')
"
```

This runs four checks:
1. **Time Sanity** -- category GPU kernel time sum vs computation time (WARN if >15% discrepancy)
2. **Efficiency Anomalies** -- findings with efficiency >100% (measurement issues)
3. **Coverage** -- all expected system and compute findings present
4. **Priority Consistency** -- top 3 categories by GPU time for P1-P3 verification

---

## Step 9: Aggregate Results -- Two-Tier Recommendations

### Read Findings from BOTH Tiers

```bash
<prefix> python3 -c "
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_findings
load_findings('<output_dir>')
"
```

Then read the individual findings files through the container as needed for report assembly.

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

### Aggregate System-Level Recommendations

System-level recommendations address pipeline/framework issues that affect ALL operations.

**System-Level Prioritization:**

Assign priorities sequentially starting from P1 based on which analyses are present. If CPU/Idle is below threshold, multi-kernel issues start at P1.

| Order | Source | Criteria | Included When |
|-------|--------|----------|---------------|
| First | CPU/Idle | `idle_time_percent > 15%` | Only if idle exceeds 15% threshold |
| Next | Multi-Kernel | Flagged multi-kernel issue (memcpy/NCCL blocking/overlap) | Only if `flagged` is `true` for any assessment |
| Next | Multi-Kernel | Additional flagged multi-kernel issue | If multiple assessments are flagged |

**CRITICAL: No-Issue Handling:**
- If **all** system-level analyses report no actionable issues (idle <= 15% and all multi-kernel assessments have `flagged: false`), do **NOT** generate any P1/P2/P3 recommendations for the System-Level Optimizations section.
- Instead, display a short summary: "No system-level bottlenecks detected. GPU activity breakdown shows X% computation, with negligible memcpy and communication overhead."
- This avoids misleading stakeholders with a red P1 icon for a non-issue.

**System-Level Icon Mapping (by priority number, NOT severity):**

| Priority | Icon |
|----------|------|
| P1 | 🔴 |
| P2 | 🟡 |
| P3+ | 🟢 |

---

## Step 10: Generate Final Report

Create `standalone_analysis.md` in `<output_dir>` **using `<prefix>`** (e.g., via `<prefix> tee <path> << 'REPORT_EOF' ... REPORT_EOF`). When running on a cluster, do **not** use the local Write/file-write tool — the report must be written on the same NFS client that Step 10.1 will use to read and modify it.

The report uses a **two-section structure**: Compute Kernel Optimizations and System-Level Optimizations. Each section is independently composable and can stand alone as a deliverable.

The report **must** use these exact `##` headers — do NOT rename them:
1. `## Executive Summary`
2. `## Compute Kernel Optimizations`
3. `## System-Level Optimizations`
4. `## Detailed Analysis: Compute Kernels`
5. `## Detailed Analysis: System-Level`
6. `## Appendix`

Each compute kernel P-item must use **Issue** / **Action** / **Impact** fields.

Validate the report before sharing the priority recommendations on the chat and prompt the user to review the report.

**Purpose:** Stakeholder report with prioritized recommendations in two tiers

```markdown
# <Model> - <Platform> Standalone Analysis

## Executive Summary
[1 paragraph overview + key metrics table]

| Metric | Value |
|--------|-------|
| Total Compute Time | X ms |
| Computation | Y% |
| Idle Time | Z% |
| Exposed Communication | W% |
| Top Bottleneck Category | Category (V%) |

{{PERF_PLOT}}

## Warnings

**Include this section ONLY if any subagent failed:**

The following analyses could not be completed due to script failures:

| Analysis | Tier | Error Summary |
|----------|------|---------------|
| <name> | System / Compute Kernel | <brief error description> |

These are excluded from the recommendations below.

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).
Summaries of recommendations from Step 7 sub-agents, focused on individual kernel efficiency.

### Top Operations

Use **% of computation time** (not % of total trace time) so readers can see each operation's share of the GPU compute budget. Compute the denominator as `total_time_ms * computation_time_percent / 100` from the manifest `gpu_utilization`.

| Rank | Operation | Category | Time (ms) | % of Compute Time |
|------|-----------|----------|-----------|-------------------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |

<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->

### 🔴 P1: <Brief Title>

**Issue**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

**Impact**: [~X.X ms savings from closing efficiency gaps (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟡 P2: <Brief Title>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X ms savings from closing efficiency gaps (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟢 P3: <Brief Title>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X ms savings from closing efficiency gaps (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

<!-- CONDITIONAL: If NO actionable system-level issues found (idle <= 15% and all multi-kernel assessments flagged: false), use Template A. -->
<!-- Otherwise, number priorities sequentially starting from P1. Include CPU/Idle only if idle > 15%. -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Title format: Descriptive name only. -->
<!-- System-level recommendations have NO **Impact** field -- impact is not quantifiable for system-level issues. -->

<!-- === TEMPLATE A: No actionable system-level issues === -->
<!-- Use this when idle <= 15% and all multi-kernel assessments have flagged: false -->

✅ No system-level bottlenecks detected. GPU activity breakdown shows X% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

<!-- === TEMPLATE B: Actionable issues found === -->
<!-- Use this when idle > 15% or at least one multi-kernel assessment has flagged: true -->

### 🔴 P1: <CPU/Idle Title OR Multi-Kernel Issue Title>

**Issue**: [1-2 sentences - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-Level > CPU/Idle Time](#cpu-idle-time-analysis) for details* OR → *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

---

### 🟡 P2: <Multi-Kernel Issue Title>

**Issue**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

---

### 🟢 P3: <Next Multi-Kernel Issue>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

---

## Detailed Analysis: Compute Kernels

### 1. <Operation Category> (X% of compute)
[All kernel breakdowns, calculations, tables, explanations from category_findings/]

### 2. <Operation Category> (X% of compute)
[...]

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

### 1. CPU/Idle Time Analysis
[Full cpu_idle_findings.md content from system_findings/]

### 2. Multi-Kernel Issues
[Full multi_kernel_findings.md content from system_findings/]

---

## Appendix

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)

```

### 10.1 Validate Report Structure (Retry up to 2x)

After writing `standalone_analysis.md`, validate that the report contains all 6 required `##` section headers. If validation fails, modify the report with the missing sections.
**Validation procedure:**

```bash
<prefix> python3 -c "
from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import validate_report
passed, missing = validate_report('<output_dir>')
if not passed:
    print('FAIL: Missing sections: ' + ', '.join(missing))
    import sys; sys.exit(1)
print('PASS: All required sections present')
"
```

**If validation fails (exit code 1):**

1. Read the FAIL output to identify missing sections
2. Prompt the LLM with a correction request:
   > "Your report at `<output_dir>/standalone_analysis.md` is missing these sections: [list]. The report **must** use these exact `##` headers: Executive Summary, Compute Kernel Optimizations, System-Level Optimizations, Detailed Analysis: Compute Kernels, Detailed Analysis: System-Level, Appendix. Rewrite `standalone_analysis.md` with all required sections, keeping existing content."
3. After the LLM rewrites, run validation again
4. Maximum 2 retry attempts. If still failing after retries, proceed to Step 10.1 with a warning

---

### 10.2 Generate and Embed Performance Improvement Plot

After writing `standalone_analysis.md` with the `{{PERF_PLOT}}` placeholder, run a **single command** that generates `plot_data.json`, renders `perf_improvement.png.

**Important:** The plot data is sourced from deterministic `impact_estimates` pre-computed by the analysis scripts (stored in each `*_metrics.json`). Do **not** parse the `## Impact Summary` markdown tables in findings files for the plot -- those tables are for human readability only.

```bash
<prefix> python3 -c "
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_and_embed_plot
generate_and_embed_plot('<output_dir>', '<Model> on <Platform> — Kernel Tuning Potential')
"
```

If the plot is skipped, the `{{PERF_PLOT}}` placeholder is removed so the report remains clean.

**Key formatting rules:**
1. **Warnings section**: Only include if there were errors; omit entirely if all succeeded
2. **Executive Summary**: Max ~20 lines
3. **Performance plot**: The `{{PERF_PLOT}}` placeholder is replaced by Step 10.1 with a base64-embedded PNG data URI (`![Performance Improvement](data:image/png;base64,...)`). The plot shows **kernel tuning potential only**.
4. **Compute Kernel Optimizations**: P1-P3+ from category subagent findings
5. **System-Level Optimizations**: If all system-level analyses report no actionable issues (NONE/N/A severity), use a single "✅ No system-level bottlenecks detected" summary instead of P1/P2/P3 recommendations. Only generate numbered priorities when at least one actionable issue exists (Number sequentially from P1, including CPU/Idle first if invoked)
6. **Each section is independently composable** -- can be shared standalone
7. **Compute and System tiers use separate sequential P1/P2/P3 numbering (no gaps)**
8. **Priority icons are assigned by PRIORITY NUMBER, not severity:**
   - **Compute Kernel:** 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ...
   - **System-Level:** 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ... (only when actionable issues exist)
9. **Detailed Analysis**: Split into Compute Kernels and System-Level subsections. Always include the Detailed Analysis: System-Level section with full metrics even when no actionable issues exist.
10. **No redundancy**: Information appears in ONE place only
11. **Recommendations**: Max ~10 lines PER recommendation

---

## Error Handling

### Unsupported Trace Features

If Steps 1 or many of Steps 2-5 fail or produce unexpected results, check whether the trace uses unsupported features before retrying:

- **Torch Compile**: `ops_summary.csv` contains op names matching `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, or `CompiledFunction`.
- **GPU Graph Replay**: raw trace JSON contains `hipGraphLaunch` or `cudaGraphLaunch`.

If found, inform the user which feature was detected and that TraceLens Agentic Mode currently supports eager-mode PyTorch traces only. **Abort** -- do not retry or continue.

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

## Efficiency Thresholds (General)

| Efficiency | Assessment |
|------------|------------|
| >70% | Good |
| <70% | Needs investigation |

