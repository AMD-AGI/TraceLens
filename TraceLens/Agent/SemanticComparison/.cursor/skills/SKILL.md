<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: Semantic Comparison
description: End-to-end semantic gap analysis for two GPU traces. Runs breakdown agents in parallel (extraction + tree context + classification + pattern finding + LLM labeling per trace), harmonizes labels across traces, then runs comparison pipeline and generates a stakeholder gap analysis report.
triggers:
  - semantic comparison
  - semantic gap analysis
  - compare two traces semantically
  - trace comparison
tools:
  - terminal
  - file_read
  - file_write
---

# Semantic Comparison

Orchestrate end-to-end semantic comparison of two GPU traces. The user
provides two raw trace files and the orchestrator handles everything:
parallel breakdown with LLM-driven labeling, cross-trace harmonization,
comparison pipeline, and final stakeholder report.

Use vendor-agnostic terminology (GPU kernels, vendor GEMM library, etc.)
except when quoting actual kernel names from traces.

---

## Workflow Steps

```
0.   Query User Inputs
1.   Semantic Breakdown (PARALLEL subagents, one per trace)
2.   Semantic Harmonization (single agent, cross-trace)
3.   Generate TraceDiff Output (script)
4.   Generate Comparison CSV (script)
5.   Compute Priority Ranking (script)
6.   Verify Outputs (script)
7.   Generate Comparison Report (script) -> Excel + CSVs
8.   Write Gap Analysis Report (LLM)
9.   Validate Report
```

---

## Step 0: Query User Inputs

Ask the user for:

**Required:**
- Trace A path (.json or .json.gz)
- Trace B path (.json or .json.gz)
- Short labels for each trace (e.g., MI355 / B200)

**Optional:**
- HuggingFace config.json path (enriches model_info)
- Output directory (default: `comparison_output/`)
- Capture trace A/B paths (for graph-mode traces with companion captures)

**Auto-detection:** If the user provides a directory, look for paired
`*_graph*` and `*_capture*` files.

**vLLM traces** (auto-detected via `gpu_user_annotation` events): use
`--split-vllm` to split by annotation timeline, analyze all steady-state
regions, match by region descriptor for comparison.

---

## Step 1: Semantic Breakdown (PARALLEL subagents)

### 1.1 Read the Agent Definition

Read: `TraceLens/AgenticMode/SemanticComparison/.cursor/agents/semantic-breakdown-agent.md`

### 1.2 Launch Both Breakdown Subagents in PARALLEL

Launch **both** simultaneously using the Task tool.

**Task prompt structure for each subagent:**

```
---BEGIN AGENT INSTRUCTIONS---
<full contents of semantic-breakdown-agent.md>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Trace path: <trace_path>
- Trace name: <trace_name>
- Output directory: <output_dir>/_work/<trace_name>/
- Capture trace path: <capture_trace_path>  (omit if not provided)

Follow the agent instructions above to complete the breakdown.
Return the summary described in the "Return Value" section.
```

**Output directories:**
- Trace A: `<output_dir>/_work/<name_a>/`
- Trace B: `<output_dir>/_work/<name_b>/`

For vLLM traces, output has per-region subdirs under each trace directory.

### 1.3 Verify Breakdown Outputs

After both subagents complete, verify each directory contains
`extracted.json`, `pattern.json`, `classified.json`, `tree_context.json`,
`breakdown_context.json`, `llm_labels.json`, and `semantic_labels.json`.
If either failed, report the error and stop.

---

## Step 2: Semantic Harmonization

### 2.1 Read the Agent Definition

Read: `TraceLens/AgenticMode/SemanticComparison/.cursor/agents/semantic-harmonization-agent.md`

### 2.2 Launch Harmonization Agent

```
---BEGIN AGENT INSTRUCTIONS---
<full contents of semantic-harmonization-agent.md>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Trace A directory: <output_dir>/_work/<name_a>/
- Trace B directory: <output_dir>/_work/<name_b>/
- Output directory: <output_dir>/_work/
- Name A: <name_a>
- Name B: <name_b>

Follow the agent instructions above to harmonize the labels.
Return the summary described in the "Return Value" section.
```

For multi-region vLLM: run once per matching region pair.

### 2.3 Verify Harmonization

Check that `alignment.json`, `harmonization_context.json`, and
`harmonization_corrections.json` exist in `<output_dir>/_work/`.

---

## Step 3: Generate TraceDiff Output [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_semantic_diff.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/tracediff_output/
```

Produces in `<output_dir>/tracediff_output/`:
- `diff_stats.csv` -- per-kernel rows matching TraceDiff schema
- `diff_stats_unique_args_summary.csv` -- aggregated by semantic block
- `cpu_op_map.json`, `cpu_op_map_trace1.json`, `cpu_op_map_trace2.json`
- `merged_tree_output.txt`

This is a **final deliverable** directory for downstream TraceDiff consumers.

---

## Step 4: Generate Comparison CSV [S]

**Single-region mode:**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/match_and_compare.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison.csv
```

**Multi-region mode (vLLM):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/match_and_compare.py \
    --regions-dir-a <output_dir>/_work/<name_a> \
    --regions-dir-b <output_dir>/_work/<name_b> \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison.csv
```

---

## Step 5: Compute Priority Ranking [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/compute_priority.py \
    <output_dir>/_work/comparison.csv \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/priority.json --top 5
```

---

## Step 6: Verify Outputs [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/verify_comparison.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    <output_dir>/_work/comparison.csv \
    --name-a <name_a> --name-b <name_b>
```

Must exit 0. Fix any FAIL assertions before proceeding.

---

## Step 7: Generate Comparison Report [S]

Call **once**, even for multi-region traces.

**Single-region mode:**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_comparison_report.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --comparison <output_dir>/_work/comparison.csv \
    --diff-dir <output_dir>/tracediff_output/ \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/comparison_report.xlsx \
    --output-csvs-dir <output_dir>/_work/comparison_report_csvs
```

**Multi-region mode:** Use a representative region's labels and point
`--diff-dir` at the TraceDiff output directory:
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_comparison_report.py \
    <output_dir>/_work/<name_a>/<representative_region>/semantic_labels.json \
    <output_dir>/_work/<name_b>/<representative_region>/semantic_labels.json \
    --comparison <output_dir>/_work/comparison.csv \
    --diff-dir <output_dir>/tracediff_output/ \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/comparison_report.xlsx \
    --output-csvs-dir <output_dir>/_work/comparison_report_csvs
```

---

## Step 8: Write Gap Analysis Report [LLM]

Read the report template from:
`TraceLens/AgenticMode/SemanticComparison/.cursor/agents/gap-analysis-report-template.md`

Using `_work/comparison.csv`, `_work/priority.json`, and both
`_work/<name>/semantic_labels.json` files, produce
`<output_dir>/semantic_comparison_report.md` following the template.

---

## Step 9: Validate Report

- All `priority.json` entries appear in the P1/P2/P3 section
- All semantic blocks from `comparison.csv` appear in detailed analysis
- No blocks are missing

Prompt the user to review the report.

---

## Key Principles

1. **Seamless flow** -- user provides two trace paths, orchestrator handles
   everything
2. **Parallel breakdown** -- both traces processed as subagents simultaneously
3. **Cross-trace harmonization** -- separate step unifies labels for
   consistent comparison
4. **Vendor-agnostic language** -- generic terms for all recommendations
5. **Complete coverage** -- every semantic block appears in the report
6. **No script creation** -- subagents use only existing scripts

---

## Final Deliverables

```
<output_dir>/
  comparison_report.xlsx              # Multi-sheet Excel workbook
  semantic_comparison_report.md       # Stakeholder gap analysis report
  tracediff_output/                    # TraceDiff-compatible output
    diff_stats.csv
    diff_stats_unique_args_summary.csv
    cpu_op_map.json
    cpu_op_map_trace1.json
    cpu_op_map_trace2.json
    merged_tree_output.txt
```
