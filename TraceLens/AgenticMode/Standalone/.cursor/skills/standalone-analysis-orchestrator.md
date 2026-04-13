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

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path, Analysis Mode, Environment Setup)
1. Generate Performance Report (branches on analysis mode: training vs inference)
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
5.5. Model Identification (subagent) → metadata/model_info.json
6. System-Level Analysis (CPU/Idle + Multi-Kernel, PARALLEL) → system_findings/
7. Invoke Compute Kernel Subagents (PARALLEL) → category_findings/
8. Validate Subagent Outputs (system_findings/ + category_findings/)
9. Aggregate Results: System-Level + Compute Kernel Recommendations
10. Generate Final Report (composable System + Compute sections)
10.1. Generate and Embed Performance Improvement Plot (single atomic call: plot_data + matplotlib PNG + base64 embed)
```

**Subagent usage:** Only invoke Task subagents in steps that explicitly say "subagent" (Steps 5.5, 6, 7). All other steps must be performed directly by the orchestrator using the command prefix.

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Trace File Path** → `<trace_path>`
   - Ask: "Please provide the full path to your PyTorch trace file (.json or .json.gz)"

2. **Platform** → `<platform>`
   - Ask: "Which platform are you analyzing?"
   - Options:
     1. **MI300X**
     2. **MI325X**
     3. **MI350X**
     4. **MI355X**
     5. **MI455X**

3. **Analysis Mode** → `<analysis_mode>`
   - If the user's prompt explicitly specifies an analysis mode or mentions inference/vLLM/SGLang, use that. Otherwise, default to `default` without asking.
   - Options:
     1. **Default (training and non-vLLM/SGLang eager inference)** (`<analysis_mode>` = `default`) — uses `TraceLens_generate_perf_report_pytorch`
     2. **Inference analysis (vLLM/SGLang)** (`<analysis_mode>` = `inference`) — uses `TraceLens_generate_perf_report_pytorch_inference`
   - If **Inference (vLLM/SGLang)** is selected, ask **Execution Mode** → `<inference_exec_mode>`:
     1. **Eager mode** (`<inference_exec_mode>` = `eager`) — only the trace file is needed
     2. **Graph replay + capture** (`<inference_exec_mode>` = `graph_capture`) — also requires a capture folder path
   - If **Graph replay + capture**, ask for **Capture Folder Path** → `<capture_folder_path>`:
     - Ask: "Please provide the full path to the graph capture traces folder"
     - Example: `/home/user/traces/capture_traces/`

4. **Environment Setup**
   - Ask: "Are you running locally or on a cluster?"
     - If **local**: No further environment questions — prefix is blank (commands run directly).
     - If **cluster**:
       - Ask "Which node should we use?" → `<node>`
       - Ask "Are you working in a containerized environment (e.g. Docker)?" → if yes, ask for container name → `<container>`
       - Ask "Are you using a virtual environment?" → if yes, ask for venv path → `<venv_path>`

5. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

### Build and Cache Command Prefix

After collecting inputs, build a command template and save it to `<output_dir>/cache/cmd_prefix.txt`. Create the directory with `mkdir -p <output_dir>/cache`.

The template uses `{CMD}` as a placeholder for the actual command.

**Cluster:** Before building the prefix, locate the TraceLens project root on the remote environment.

Run the following command (adjust for container if applicable):

```bash
# Without container:
ssh <node> "find / -maxdepth 5 -type d -name 'TraceLens' 2>/dev/null | head -5"

# With container:
ssh <node> "docker exec <container> bash -c 'find / -maxdepth 5 -type d -name TraceLens 2>/dev/null | head -5'"
```

Pick the result that contains `TraceLens/AgenticMode/` and store it as `<tracelens_dir>`.

Build the cluster prefix using this lookup:

| Container | Venv | Template |
|-----------|------|----------|
| No | No | `ssh <node> "cd <tracelens_dir> && {CMD}"` |
| Yes | No | `ssh <node> "docker exec <container> bash -c 'cd <tracelens_dir> && {CMD}'"` |
| No | Yes | `ssh <node> "bash -c 'source <venv_path>/bin/activate && cd <tracelens_dir> && {CMD}'"` |

Write the resolved template (with actual node/container/venv/tracelens_dir values substituted) to `<output_dir>/cache/cmd_prefix.txt`.

### Command Execution Pattern

**Before executing any command**, read `<output_dir>/cache/cmd_prefix.txt`. It contains a template with a `{CMD}` placeholder. Substitute `{CMD}` with the actual command. All commands below use `<prefix>` to represent this resolved template.

---

## Step 1: Generate Performance Report

Use the analysis mode selected in Step 0 to determine which CLI tool to run.

**Default (training and non-vLLM/SGLang eager inference)** (analysis_mode = `default`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs \
  --gpu_arch_json_path TraceLens/AgenticMode/Standalone/utils/arch/<platform>.json \
  --group_by_parent_module \
  --enable_pseudo_ops \
  --group_by_num_kernels
```

**Inference eager mode** (analysis_mode = `inference`, inference_exec_mode = `eager`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs \
  --gpu_arch_json_path TraceLens/AgenticMode/Standalone/utils/arch/<platform>.json \
  --group_by_parent_module \
  --enable_pseudo_ops \
  --group_by_num_kernels
```

**Inference graph replay + capture mode** (analysis_mode = `inference`, inference_exec_mode = `graph_capture`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path <trace_path> \
  --capture_folder <capture_folder_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs \
  --gpu_arch_json_path TraceLens/AgenticMode/Standalone/utils/arch/<platform>.json \
  --group_by_parent_module \
  --enable_pseudo_ops \
  --group_by_num_kernels
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

## Step 5.5: Model Identification (Subagent)

Launch a Task subagent (generalPurpose) with the full contents of `TraceLens/AgenticMode/Standalone/.cursor/agents/model-identification-agent.md` and context: <output_dir>. Wait for completion. On failure, write fallback `metadata/model_info.json` with all four fields `"Cannot be inferred from trace"`.

Assign <Model> to model value in `<output_dir>/metadata/model_info.json` or "Workload" if model is "Cannot be inferred from trace". Wait for completion before proceeding further.

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
| sdpa_bwd | sdpa-analyzer.md |
| moe_fused | moe-analyzer.md |
| elementwise | elementwise-analyzer.md |
| triton | triton-analyzer.md |
| reduce | reduce-analyzer.md |
| norm | norm-analyzer.md |
| convolution | convolution-analyzer.md |
| kernel_fusion | kernel-fusion-analyzer.md |
| other | generic-op-analyzer.md |

---

## Step 6: System-Level Analysis (PARALLEL)

System-level analysis examines issues that affect the GPU pipeline as a whole -- idle time, memory transfer patterns, and communication/compute overlap. These are **not** about individual kernel efficiency.

**Output directory:** `system_findings/`

### 6.1 Read Manifest and Identify System-Level Subagents

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_manifest_categories
load_manifest_categories(sys.argv[1])
\" '<output_dir>'"
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
- `kernel_fusion` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/kernel-fusion-analyzer.md` (invoke if `kernel_fusion` category exists in manifest)

**Invocation conditions:**
- **CPU/Idle**: Read `category_data/category_manifest.json` and check `gpu_utilization.idle_time_percent`. Only invoke the subagent if `idle_time_percent > 15`. Skip otherwise -- the deterministic script already captured the factual data.
- **Multi-Kernel**: `multi_kernel` category exists in manifest OR `gpu_util['exposed_comm_time_percent'] > 0` OR `gpu_util['exposed_memcpy_time_percent'] > 0`
- **Kernel Fusion**: `kernel_fusion` category exists in manifest

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
1. Running its Python script using the command prefix
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
- `sdpa_fwd` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/sdpa-analyzer.md` (pass `--category sdpa_fwd`)
- `sdpa_bwd` → Read `TraceLens/AgenticMode/Standalone/.cursor/agents/sdpa-analyzer.md` (pass `--category sdpa_bwd`)
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
1. Running its Python script using the command prefix
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

#### 3. Fusion-Tagged Operations

- Operations with `fusion_flagged: true` in the metrics JSON are already covered by a high-confidence kernel fusion candidate — do **NOT** flag them as bottlenecks or write kernel_tuning recommendations
- The analysis scripts already exclude them from `impact_estimates`

---

**Compute Kernel Subagent Prompt Template:**

When invoking a compute kernel subagent, use this template:

```
You are analyzing {category} operations for a PyTorch trace on {platform}.

**CRITICAL - READ FIRST:**
- Use GPU kernel time (not CPU duration) for all bottleneck analysis
- Flag any efficiency > 100% as "[ANOMALY] - verify measurement"
- When citing peak performance, use bound-type-aware references: `efficiency.resolved_peak_maf` (TFLOPS) for compute-bound ops, `efficiency.resolved_peak_hbm_bw` (TB/s) for memory-bound ops

**Platform Specs:**
- Peak HBM BW: {peak_hbm_bw} TB/s
- Peak references are bound-type-aware: each operation's `efficiency.resolved_peak_maf` has the precision-correct compute peak (TFLOPS); `efficiency.resolved_peak_hbm_bw` has the memory bandwidth peak (TB/s). Use the one matching `efficiency.bound_type`
- Impact estimates assume tuning can reach 75–100% of roofline (midpoint 87.5% used for plots)

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
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_subagent_outputs
validate_subagent_outputs(sys.argv[1])
\" '<output_dir>'"
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
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_findings
load_findings(sys.argv[1])
\" '<output_dir>'"
```

Then read the individual findings files using the command prefix as needed for report assembly.

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

## Step 9.5: Generate Performance Improvement Plot

**Important:** The plot data is sourced from deterministic `impact_estimates` pre-computed by the analysis scripts (stored in each `*_metrics.json`). Do **not** parse the `## Impact Summary` markdown tables in findings files for the plot -- those tables are for human readability only.

### 9.5.1 Ensure matplotlib is available

```bash
<prefix> python3 -c "import matplotlib" 2>/dev/null || <prefix> pip install matplotlib
```

### 9.5.2 Generate plot_data.json

Run the `generate_plot_data()` utility to aggregate all `impact_estimates` from `*_metrics.json` files into a single `plot_data.json`:

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import generate_plot_data
generate_plot_data(sys.argv[1])
\" '<output_dir>'
```

### 9.5.3 Generate Plot and Base64 File

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_perf_plot
generate_perf_plot(sys.argv[1], sys.argv[2])
\" '<output_dir>' '<Model> on <Platform> — Kernel Tuning Potential'
```

If the plot fails or is skipped, proceed to Step 10 without the plot and note the failure in the report.

---

## Step 10: Generate Final Report (<output_dir>/standalone_analysis.md)

1. **Read** the report template: `TraceLens/AgenticMode/Standalone/utils/templates/standalone_analysis_template.md`
2. **Copy** it to `<output_dir>/standalone_analysis.md` using `<prefix>` (e.g., via `<prefix> cp ...` or `<prefix> tee ...`). Do **not** use the local Write/file-write tool — the report must be written on the same NFS client that Step 10.2 will use to read and modify it.
3. **Fill in** each section by substituting placeholders with data using `<prefix>`. Never retain template placeholders (`<Brief Title>`, `X ms`, `Y%`, `<platform>`, `<model>`) — every field must contain actual data.
   - `category_data/category_manifest.json` (metrics, GPU utilization)
   - `category_findings/*.md` (compute kernel P-items)
   - `system_findings/*.md` (system-level P-items)
   - `category_data/*_metrics.json` (per-op tables, impact estimates)
4. **Paste `## Detailed Analysis`:** For each P-item in priority order, take the `## Detailed Analysis` candidate from the matching `category_findings/*.md` or `system_findings/*.md` file and place it in the report.

The report at `<output_dir>/standalone_analysis.md` must use these exact `##` headers — do NOT rename them:
1. `## Executive Summary`
2. `## Compute Kernel Optimizations`
3. `## Kernel Fusion Opportunities (Experimental)`
4. `## System-Level Optimizations`
5. `## Detailed Analysis`
6. `## Appendix`

Each compute kernel P-item must use **Insight** / **Action** / **Impact** fields.

Validate the report before sharing the priority recommendations on the chat and prompt the user to review the report.

### 10.1 Validate Report Structure (Retry up to 2x)

After writing `standalone_analysis.md`, validate that the report contains all required `##` section headers. If validation fails, modify the report with the missing sections.

**Validation procedure:**

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import validate_report
passed, missing = validate_report(sys.argv[1])
if not passed:
    print('FAIL:')
    for m in missing:
        print('  - ' + m)
    sys.exit(1)
print('PASS: All required sections present')
\" '<output_dir>'
```

**If validation fails (exit code 1):**

1. Read the FAIL output to identify missing sections
2. Check if the report contains similar but incorrectly named headers (e.g., `## Compute Kernel Analysis` instead of `## Compute Kernel Optimizations`, or `## System-Level Analysis` instead of `## System-Level Optimizations`) and rename them to match the exact required names using string replacement. Do NOT rewrite the report from scratch.
3. If sections are entirely absent, add them with the correct `##` headers, keeping existing content
4. Run validation again
5. Maximum 3 retry attempts. If still failing after retry, proceed to Step 10.2 with a warning

---

### 10.2 Generate and Embed Performance Improvement Plot

After writing `standalone_analysis.md` with the `{{PERF_PLOT}}` placeholder, run a **single command** that generates `plot_data.json`, renders `perf_improvement.png`, and embeds the base64-encoded plot into the report.

**Important:** The plot data is sourced from deterministic `impact_estimates` pre-computed by the analysis scripts (stored in each `*_metrics.json`). Do **not** parse the `## Impact Summary` markdown tables in findings files for the plot -- those tables are for human readability only.

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_and_embed_plot
generate_and_embed_plot(sys.argv[1], sys.argv[2])
\" '<output_dir>' '<Model> on <Platform> — Kernel Tuning Potential'
```

If the plot is skipped, the `{{PERF_PLOT}}` placeholder is removed so the report remains clean.

---

## Error Handling

### Unsupported Trace Features

If Steps 1 or many of Steps 2-5 fail or produce unexpected results, check whether the trace uses unsupported features before retrying:

- **Torch Compile**: `ops_summary.csv` contains op names matching `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, or `CompiledFunction`. If found, inform the user and **abort**.
- **GPU Graph Replay**: raw trace JSON contains `hipGraphLaunch` or `cudaGraphLaunch`.
  - **Default mode** (analysis_mode = `default`): Inform the user that GPU graph replay was detected and that the default analysis mode supports typical PyTorch traces. **Abort** -- do not retry or continue.
  - **Inference mode** (analysis_mode = `inference`): Graph launches are expected and supported. Do **NOT** abort. If inference_exec_mode is `eager` (no capture folder was provided), log a warning that analysis may be limited without graph capture traces, but continue.

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

