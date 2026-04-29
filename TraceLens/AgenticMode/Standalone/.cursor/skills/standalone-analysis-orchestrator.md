<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: Standalone Analysis Orchestrator
description: Orchestrate two-tier PyTorch trace performance analysis - system-level (CPU/idle, multi-kernel) and compute kernel tiers with independently composable reports
triggers:
  - standalone analysis
  - comparative analysis
  - analyze trace standalone
  - compare two traces
  - performance analysis single platform
tools:
  - terminal
  - file_read
  - file_write
---

# Standalone Analysis Orchestrator

Orchestrate modular PyTorch trace analysis using a **two-tier architecture**:
- **System-Level Analysis** (Step 6): CPU/idle time + multi-kernel issues (memcpy, NCCL blocking, overlap)
- **Compute Kernel Analysis** (Step 7): Per-category kernel efficiency (GEMM, SDPA, elementwise, etc.)

**Role**: Load trace once (primary trace), pre-compute tree data, filter by category, invoke system-level and compute kernel subagents in parallel, aggregate results into independently composable report sections.

**Comparison scope (`<comparison_scope>`):** **`standalone`** — single trace, roofline analysis. **`comparative`** — primary trace vs second trace comparative analysis

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective communication, vendor GEMM library, DNN primitives, GPU graph, etc. Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from traces, it's acceptable to include the actual name for identification.
 
---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path(s), Analysis Mode, Environment Setup)
1. Generate Performance Report (branches on analysis mode: training vs inference then, comparison scope)
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
6. System-Level Analysis (CPU/Idle + Multi-Kernel, PARALLEL) → system_findings/
7. Invoke Compute Kernel Subagents (PARALLEL) → category_findings/
8. Validate Subagent Outputs (system_findings/ + category_findings/)
9. Prepare Report Data (load_findings) + Model Identification (subagent) → metadata/model_info.json
10. Generate Performance Improvement Plot (reads priority_data.json → PNG + base64 embed)
11. Generate Final Report (composable System + Compute sections)
11.3. **Comparative only:** Cumulative TraceDiff plot (`{{COMPARATIVE_CUMULATIVE_PLOT}}`) — see Step 11.3
```

**Subagent usage:** Only invoke Task subagents in steps that explicitly say "subagent" (Steps 6, 7, 9). All other steps must be performed directly by the orchestrator using the command prefix.

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

0. **Comparison scope** → `<comparison_scope>`
   - Set from the user’s intent **before** deep-diving on paths:
     - **`comparative`** if the skill was triggered by **“comparative analysis”**, **“compare two traces”**, or the user supplies **two** trace paths / explicitly asks to compare trace A vs B.
     - **`standalone`** otherwise (including triggers **“standalone analysis”**, **“analyze trace standalone”**, single trace only).

1. **Trace File Path(s)**
   - **`standalone`:** **Trace File Path** → `<trace_path>`
     - Ask: "Please provide the full path to your PyTorch trace file (.json or .json.gz)"
   - **`comparative`:** ask for both:
     - **Primary trace (trace1)** → `<trace_path>`
     - **Comparison trace (trace2)** → `<trace2_path>`
     - Ask: "Please the full path to your primary trace file and your comparison trace file (.json or .json.gz)"

2. **Platform** → `<platform>`
   **`standalone`**: Ask: "Which platform are you analyzing?"
   **`comparative`**: Ask: "Which platform is baseline trace (trace1)?"
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

Pick the result containing `AgenticMode/` and strip the trailing `/TraceLens` to get `<tracelens_dir>`.

Build the cluster prefix using this lookup:

| Container | Venv | Template |
|-----------|------|----------|
| No | No | `ssh <node> "cd <tracelens_dir> && {CMD}"` |
| Yes | No | `ssh <node> "docker exec <container> bash -c 'cd <tracelens_dir> && {CMD}'"` |
| No | Yes | `ssh <node> "bash -c 'source <venv_path>/bin/activate && cd <tracelens_dir> && {CMD}'"` |

Write the resolved template to `<output_dir>/cache/cmd_prefix.txt`. Then validate it works:

```bash
<prefix> python3 -c "import TraceLens; print('PREFIX_OK')"
```

If this fails, check that `<tracelens_dir>` is the **parent** of TraceLens (not the repo root itself), verify the container/venv is accessible, rebuild, and retry. Do NOT proceed to Step 1 until validation passes.

### Command Execution Pattern

**Before executing any command**, read `<output_dir>/cache/cmd_prefix.txt`. It contains a template with a `{CMD}` placeholder. Substitute `{CMD}` with the actual command. All commands below use `<prefix>` to represent this resolved template.

---

## Step 1: Generate Performance Report

Use **`<analysis_mode>`** to determine which CLI tool to run and then **`<comparison_scope>`** to determine arguments.

**Output paths:**
**`standalone`** — one run: `--output_xlsx_path <output_dir>/perf_report.xlsx`, `--output_csvs_dir <output_dir>/perf_report_csvs`.
**`comparative`** — two runs: trace1 → `<output_dir>/perf_report_trace1.xlsx`, `<output_dir>/perf_report_trace1_csvs`; trace2 → `--profile_json_path <trace2_path>`, `<output_dir>/perf_report_trace2.xlsx`, `<output_dir>/perf_report_trace2_csvs`.

**Extension flags (comparative, trace1 only):** When `<comparison_scope>` = `comparative`, append **only** to the trace1 command:

```text
  --extension_file TraceLens/Reporting/tracediff_comparison_extension.py \
  --extension_args <trace2_path>
```

Do **not** pass `--extension_*` on the trace2 command.

---

**Default (training and non-vLLM/SGLang eager inference)** (`<analysis_mode>` = `default`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --output_xlsx_path <output_dir>/perf_report.xlsx \
  --output_csvs_dir <output_dir>/perf_report_csvs \
  --gpu_arch_json_path TraceLens/AgenticMode/Standalone/utils/arch/<platform>.json \
  --enable_pseudo_ops \
  --group_by_num_kernels
```

**Inference eager mode** (`<analysis_mode>` = `inference`, `<inference_exec_mode>` = `eager`):

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

When `<comparison_scope>` = `comparative`, append the same `--extension_file` / `--extension_args <trace2_path>` pair as in the default-mode example (inference generator supports `--extension_args` for TraceDiff).

**Inference graph replay + capture mode** (`<analysis_mode>` = `inference`, `<inference_exec_mode>` = `graph_capture`):

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
**`standalone`:** `perf_report.xlsx` and `perf_report_csvs/`. 
**`comparative`:** `perf_report_trace1.xlsx`, `perf_report_trace1_csvs/`, `perf_report_trace2.xlsx`, `perf_report_trace2_csvs/`.

Excel report contains all sheets. CSV directory contains individual sheets in report

---

## Steps 2-5: Prepare Category Data

Execute the TraceLens Agentic Mode orchestrator preparation script:

```bash
<prefix> python3 \
  TraceLens/AgenticMode/Standalone/utils/orchestrator_prepare.py \
  --trace-path <trace_path> \
  --platform <platform> \
  --output-dir <output_dir>
  --comparison-scope <comparison_scope>
```

**Perf CSV directories by scope:** 
**`standalone`** → `<output_dir>/perf_report_csvs`. 
**`comparative`** → trace 1 from `<output_dir>/perf_report_trace1_csvs`, trace 2 from `<output_dir>/perf_report_trace2_csvs`

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

Launch system-level sub-agents simultaneously using the Task tool. Do NOT wait between invocations.

**System-Level Agent File Map:**

**Base path:** `TraceLens/AgenticMode/Standalone/.cursor/agents/`

| Category | Agent file |
|----------|-----------|
| `cpu_idle` | `cpu-idle-analyzer.md` |
| `multi_kernel` | `multi-kernel-analyzer.md` |
| `kernel_fusion` | `kernel-fusion-analyzer.md` |

**Invocation conditions:**
- **CPU/Idle**: Read `category_data/category_manifest.json` and check `gpu_utilization.idle_time_percent`. Only invoke the subagent if `idle_time_percent > 15`. Skip otherwise -- the deterministic script already captured the factual data.
- **Multi-Kernel**: `multi_kernel` category exists in manifest OR `gpu_util['exposed_comm_time_percent'] > 0` OR `gpu_util['exposed_memcpy_time_percent'] > 0`
- **Kernel Fusion**: `kernel_fusion` category exists in manifest

**Task prompt structure for each system-level subagent:**

The subagent reads its own agent file — the orchestrator does NOT read or paste agent file contents.

```
Read and follow the FULL instructions in:
  TraceLens/AgenticMode/Standalone/.cursor/agents/<agent-file>.md

**Execution Context:**
- Comparison scope: `<comparison_scope>`
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template
  with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- Input files: <list from agent file's "Input files" section>
- Output file: <from agent file's "Output file" section>

Execute every step in the agent file. Return "DONE" when complete.
```

**CRITICAL:** The orchestrator does NOT read agent files or run analysis scripts. Each sub-agent is responsible for:
1. Reading its own agent `.md` file
2. Running its Python script using the command prefix
3. Reading the metrics JSON output
4. Identifying issues and generating findings

### 6.3 Wait for System-Level Subagents to Complete

The three subagents must complete before proceeding to Step 6.4.
Each writes findings to `system_findings/<name>_findings.md`.

### 6.4 Verify System Outputs and Retry Failures (up to 1 retry per subagent)

After all system-level subagents complete:

1. For each expected system category from the manifest, check:
   - Does `system_findings/<category>_findings.md` exist?
   - If it exists, does it contain "Status: ERROR"?
2. Collect a list of **failed** categories (missing file OR Status: ERROR).
3. **Retry each failed category exactly once** by re-launching its subagent with the same prompt from Step 6.2. Wait for all retries to complete before proceeding.
4. After retries, re-check outputs. Any category that still fails is excluded from aggregation.
5. **CRITICAL: Do NOT attempt manual analysis of failed system checks — only automated subagent retry is allowed.**

---

## Step 7: Invoke Compute Kernel Subagents (PARALLEL)

Compute kernel analysis examines individual operation category efficiency.

**Output directory:** `category_findings/`

### 7.1 Read Manifest and Identify Compute Kernel Categories

Use `compute_categories` from the `load_manifest_categories()` call in Step 6.1.

### 7.2 Launch Compute Kernel Subagents in PARALLEL

**Compute Kernel Agent File Map:**

| Category | Agent file |
|----------|-----------|
| `gemm` | `gemm-analyzer.md` |
| `sdpa_fwd` | `sdpa-analyzer.md` |
| `sdpa_bwd` | `sdpa-analyzer.md` |
| `elementwise` | `elementwise-analyzer.md` |
| `reduce` | `reduce-analyzer.md` |
| `triton` | `triton-analyzer.md` |
| `moe_fused` | `moe-analyzer.md` |
| `norm` | `norm-analyzer.md` |
| `convolution` | `convolution-analyzer.md` |
| `other` | `generic-op-analyzer.md` |

**Base path:** `TraceLens/AgenticMode/Standalone/.cursor/agents/`

#### Subagent Selection

For each category in `compute_categories`, resolve `{agent_file}`:
- If the category is in the Agent File Map above, use the listed agent file.
- Otherwise (unmapped category), fall back to `generic-op-analyzer.md` — it is `<cat>`-parameterized and handles any category by substitution.

Launch all subagents simultaneously in a single parallel batch.

---

#### Shared Compute Kernel Preamble

Include this block in every compute kernel subagent prompt:

<Shared Compute Kernel Preamble>:
```
comparison_scope: {comparison_scope}

**CRITICAL - READ FIRST:**
- Use GPU kernel time (not CPU duration) for all bottleneck analysis
- **Standalone only:** flag roofline `efficiency_percent` > 100% as "[ANOMALY] - verify measurement". In **comparative** mode, > 100% means Trace 2 is slower — NOT an anomaly.
- When citing peak performance, use bound-type-aware references: `efficiency.resolved_peak_maf` (TFLOPS) for compute-bound ops, `efficiency.resolved_peak_hbm_bw` (TB/s) for memory-bound ops

**Platform Specs:**
- Peak HBM BW: {peak_hbm_bw} TB/s
- Peak references are bound-type-aware: each operation's `efficiency.resolved_peak_maf` has the precision-correct compute peak (TFLOPS); `efficiency.resolved_peak_hbm_bw` has the memory bandwidth peak (TB/s). Use the one matching `efficiency.bound_type`
- Impact estimates assume tuning can reach 75–100% of peak performance (midpoint 87.5% used for plots)

**CRITICAL CONSTRAINTS:**
1. **Standalone only:** Any efficiency > 100% → `[ANOMALY] - verify measurement`. **Comparative:** efficiency > 100% means Trace 2 is slower — NOT an anomaly.
2. Status must be SUCCESS or ERROR; times in ms; efficiencies as percentages
3. Operations with `fusion_flagged: true` in the metrics JSON are already covered by
   a high-confidence kernel fusion candidate — do NOT flag them as bottlenecks or write
   kernel_tuning recommendations. The analysis scripts already exclude them from `impact_estimates`.

**Execution Context:**
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template
  with `{CMD}` placeholder; substitute `{CMD}` with the actual command
```

---

#### Compute Kernel Subagent Prompt

For each category, launch a Task (subagent_type: generalPurpose):

```
You are analyzing {category} operations for a PyTorch trace on {platform}.

<Shared Compute Kernel Preamble>

Read and follow the FULL instructions in:
  TraceLens/AgenticMode/Standalone/.cursor/agents/{agent_file}

- Category: {category}
- Input files: category_data/{category}_ops.csv, metadata/{category}_metadata.json,
  category_data/{category}_tree_data.json (if available)
- Output file: category_findings/{category}_findings.md

Execute every step in the agent file. Return "DONE" when complete.
```

### 7.3 Wait for All Compute Kernel Subagents to Complete

All subagents must complete before proceeding to Step 7.4.
Each subagent writes its findings to `category_findings/<category>_findings.md`.

### 7.4 Verify Outputs and Retry Failures (up to 1 retry per subagent)

After all compute kernel subagents complete:

1. For each category in the manifest with `tier: compute_kernel`, check:
   - Does `category_findings/<category>_findings.md` exist?
   - If it exists, does it contain "Status: ERROR"?
2. Collect a list of **failed** categories (missing file OR Status: ERROR).
3. **Retry each failed category exactly once** by re-launching its subagent with the same prompt structure from Step 7.2. Launch all retries in parallel and wait for completion.
4. After retries, re-check outputs. Any category that still fails is excluded from aggregation and recommendations.
5. **CRITICAL: Do NOT attempt to manually analyze failed categories — only automated subagent retry is allowed.**

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
2. **Efficiency Anomalies** -- findings with efficiency >100% (measurement issues) when `<comparison_scope>` = `standalone`
3. **Coverage** -- all expected system and compute findings present
4. **Priority Consistency** -- top 3 categories by GPU time for P1-P3 verification

---

## Step 9: Prepare Report Data + Model Identification

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.report_utils import load_findings
load_findings(sys.argv[1])
\" '<output_dir>'"
```

### 9.1 Model Identification (Subagent, retry once on failure)

Launch a Task subagent (generalPurpose) that reads and follows `TraceLens/AgenticMode/Standalone/.cursor/agents/model-identification-agent.md` with context: <output_dir>. Wait for completion.

**On failure (subagent error, timeout, or `model_info.json` not written):**
1. **Retry exactly once** by re-launching the same subagent with the same prompt.
2. If the retry also fails, write fallback `metadata/model_info.json` with all four fields set to `"Cannot be inferred from trace"`.

Assign <Model> to model value in `<output_dir>/metadata/model_info.json` or "Workload" if model is "Cannot be inferred from trace".

---

## Step 10: Generate Performance Improvement Plot

**Important:** The plot data is sourced from deterministic `impact_estimates` pre-computed by the analysis scripts (stored in each `*_metrics.json`). Do **not** parse the `## Impact Summary` markdown tables in findings files for the plot -- those tables are for human readability only.

### 10.1 Ensure matplotlib is available

```bash
<prefix> python3 -c "import matplotlib" 2>/dev/null || <prefix> pip install matplotlib
```

### 10.2 Generate priority_data.json

Run `generate_priority_data()` to aggregate all `impact_estimates` from `*_metrics.json` files and manifest fallback categories into `priority_data.json` — the single source of truth for both report P-item ordering and the plot:

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_priority_data
generate_priority_data(sys.argv[1])
\" '<output_dir>'
```

### 10.3 Generate Plot and Base64 File

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_perf_plot
generate_perf_plot(sys.argv[1], sys.argv[2])
\" '<output_dir>' '<Model> on <Platform> — Kernel Tuning Potential'
```

If the plot fails, retry once. If still failing, proceed to Step 11 without the plot.

---

## Step 11: Generate Final Report

**Output filename:** `standalone_analysis.md` when `<comparison_scope>` = `standalone`; `comparative_analysis.md` when `<comparison_scope>` = `comparative`. Referred to as `<report_filename>` below.

**CRITICAL: Do NOT delegate Step 10 to a Task subagent.** The orchestrator must write the report directly.

1. **Read** the report template: `TraceLens/AgenticMode/Standalone/utils/templates/standalone_analysis_template.md`
2. **Copy** it to `<output_dir>/<report_filename>` using `<prefix>` (e.g., via `<prefix> cp ...` or `<prefix> tee ...`). Do **not** use the local Write/file-write tool — the report must be written on the same NFS client that Step 11.2 will use to read and modify it.
3. **Fill in** each section by substituting placeholders with data using `<prefix>`. Never retain template placeholders (`<Brief Title>`, `X ms`, `Y%`, `<platform>`, `<model>`) — every field must contain actual data.
   - `category_data/category_manifest.json` (metrics, GPU utilization)
   - `category_findings/*.md` (compute kernel P-items)
   - `system_findings/*.md` (system-level P-items)
   - `category_data/*_metrics.json` (per-op tables, impact estimates)
   - `priority_data.json` — use `priorities` array for compute kernel P-item ordering (P1 = rank 1, P2 = rank 2, P3+ = rest). Categories with `source: "manifest_fallback"` use Impact: "Not quantifiable from trace data".
   - `metadata/model_info.json` — for `### Model Architecture` in Appendix: substitute `<model>`, `<architecture>`, `<scale>`, `<precision>` with the four field values.
   - Platform arch file — read `platform` from `category_manifest.json`, then read `TraceLens/AgenticMode/Standalone/utils/arch/<platform>.json`. For `### Hardware Reference`: substitute `<platform>`, Peak HBM BW = `mem_bw_gbps / 1000` TB/s, Peak MAF (BF16) = `max_achievable_tflops.matrix_bf16` TFLOPS, Peak MAF (FP8) = `max_achievable_tflops.matrix_fp8` TFLOPS if present.
   - **Card sourcing:** For each findings file, copy its `## Recommendations` P-items into the report card slots and its `## Detailed Analysis` blocks into the Detailed Analysis section. Follow the template for formatting.
   - **No-findings categories:** If a findings file contains `<!-- no-actionable-findings -->`, the category has no actionable recommendations. Include it in the Top Operations table but do **not** generate a P-item card for it in the Compute Kernel Optimizations section. If **all** compute categories have no actionable findings, use: "✅ No compute kernel optimization opportunities identified. All categories are within expected performance bounds."
   - **Exclude failures:** Skip any category listed in `load_findings()` output as `failed_system` or `failed_compute`. Include a Warnings section only if failures exist.

The report at `<output_dir>/<report_filename>` must use these exact `##` headers — do NOT rename them:
1. `## Executive Summary`
2. `## Compute Kernel Optimizations`
3. `## Kernel Fusion Opportunities (Experimental)`
4. `## System-Level Optimizations`
5. `## Detailed Analysis`
6. `## Appendix`


### 11.1 Validate Report Structure (Retry up to 2x)

After writing `<report_filename>`, validate that the report contains all required `##` section headers. If validation fails, modify the report with the missing sections.

**Validation procedure:**

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_report
passed, missing = validate_report(sys.argv[1], sys.argv[2])
if not passed:
    print('FAIL:')
    for m in missing:
        print('  - ' + m)
    sys.exit(1)
print('PASS: All required sections present')
\" '<output_dir>' '<report_filename>'
```

**If validation fails (exit code 1):**

1. Read the FAIL output to identify the issue. Fix in-place, do NOT rewrite the report from scratch.
a. Check if the report contains similar but incorrectly named headers and rename them to match the exact required names. 
b. If sections are entirely absent, add them with the correct `##` headers, keeping existing content.
c. For "Missing metrics row" errors: add the row to the Executive Summary table using values from `category_data/category_manifest.json` (`gpu_utilization` keys) and `priority_data.json` (top bottleneck).
d. For placeholder values (`X ms`, `Y%`, `Z%`, `W%`) in the Executive Summary metrics table: replace each with the actual value from `category_manifest.json` -> `gpu_utilization`.
e. For unfilled `<Brief Title>` / `<Library>` / `<platform>` placeholders: substitute the real title/backend/platform from the corresponding findings file or `metadata/*_metadata.json`.
f. For Args cell mismatches: copy the matching `operations[].args` value verbatim (preserving `<br>`) from the corresponding `category_data/<cat>_metrics.json` and string-replace the bad cell.
2. Run validation again.
3. Maximum 2 retry attempts. If still failing after retry, proceed to Step 11.2 with a warning.

---

### 11.2 Generate and Embed Performance Improvement Plot

Render `perf_improvement.png`, and embed the base64-encoded plot into the report.

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.plot_utils import generate_and_embed_plot
generate_and_embed_plot(sys.argv[1], sys.argv[2])
\" '<output_dir>' '<Model> on <Platform> — Kernel Tuning Potential'
```

If the plot is skipped, the `{{PERF_PLOT}}` placeholder is removed so the report remains clean.

---

### 10.3 Comparative cumulative kernel-time plot (comparative scope only)

When `<comparison_scope>` = **`comparative`**, after Step 10.2 (or immediately after the report file exists with `{{COMPARATIVE_CUMULATIVE_PLOT}}` in the comparative Executive Summary), run **one** command to build a **stacked Baseline → Projection** chart from TraceDiff-enriched `unified_perf_summary.csv`, and embed it in the markdown.

**Labels:** Use the same naming you used in the report for the two traces (e.g. **Trace 1** = `<trace_path>` platform, **Trace 2** = comparison platform).

```bash
<prefix> python3 -c \"
import sys
from TraceLens.AgenticMode.Standalone.utils.comparative_cumulative_plot import (
    generate_and_embed_comparative_cumulative_plot,
)
generate_and_embed_comparative_cumulative_plot(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    title=sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None,
)
\" '<output_dir>' '<Plaform1>' '<Platform2>' '<Optional suptitle>'
```

- If generation fails (missing CSV, no comparative columns), the placeholder is **removed** so the report stays valid.

For **`standalone`** scope, the comparative block (including `{{COMPARATIVE_CUMULATIVE_PLOT}}`) is deleted per template rules — no Step 10.3 run.

---

## Error Handling

### Unsupported Trace Features

If Steps 1 or many of Steps 2-5 fail or produce unexpected results, check whether the trace uses unsupported features before retrying:

- **Torch Compile**: `ops_summary.csv` contains op names matching `triton_poi_fused_*`, `triton_red_fused_*`, `triton_per_fused_*`, or `CompiledFunction`. If found, inform the user.
- **GPU Graph Replay**: raw trace JSON contains `hipGraphLaunch` or `cudaGraphLaunch`.
  - **Default mode** (analysis_mode = `default`): Inform the user that GPU graph replay was detected and that the default analysis mode supports typical PyTorch traces. **Abort** -- do not retry or continue.
  - **Inference mode** (analysis_mode = `inference`): Graph launches are expected and supported if graph capture folder is provided, do not abort. If inference_exec_mode is `eager` (no capture folder was provided), continue.

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



