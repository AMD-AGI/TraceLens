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

Orchestrate end-to-end semantic comparison of two GPU traces. The user provides
two raw trace files and the orchestrator handles everything: parallel breakdown
with LLM-driven labeling per trace, cross-trace harmonization, comparison
pipeline, and final stakeholder report.

**Role**: Run breakdown subagents in parallel (each produces
`semantic_labels.json` independently), then run the harmonization agent to
unify labels across traces, execute comparison scripts, generate Excel/CSV
reports, and write a prioritized gap analysis report.

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective
communication, vendor GEMM library, DNN primitives, etc. Focus on operation
semantics, not vendor implementation details.

**Exception:** When quoting kernel names from traces, include the actual name.

---

## Workflow Steps

```
0.   Query User Inputs (Trace Paths, Names, Config)
1.   Semantic Breakdown (PARALLEL subagents, one per trace)
     -> Each produces semantic_labels.json independently
2.   Semantic Harmonization (single agent, cross-trace)
     -> Aligns + harmonizes labels across both traces
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

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Trace A Path**
   - Ask: "Please provide the full path to Trace A (.json or .json.gz)"

2. **Trace B Path**
   - Ask: "Please provide the full path to Trace B (.json or .json.gz)"

3. **Name A / Name B**
   - Ask: "What short labels should we use for each trace? (e.g., MI355 / B200)"

### Optional Information:

4. **Model config.json Path** (optional, enriches model_info)
   - Ask: "If available, provide the path to the HuggingFace config.json for the model"

5. **Output Directory**
   - Ask: "Where should we save results? (default: comparison_output/ in the current directory)"

### vLLM Trace Handling (Auto-Detected)

When traces contain `gpu_user_annotation` events (execute_context_*_generation_*):
- Use `extract_trace_data.py --split-vllm` to split by GPU annotation timeline
- Splitting uses `TraceLens.TraceUtils.split_inference_trace_annotation` under the hood (steady-state detection, phase extraction)
- Analyze **all** steady-state regions (prefill-only, decode-only, prefill-decode)
- Output: one breakdown per region; comparison matches regions by (prefill, decode) descriptor
- Report: multi-section Excel (one sheet per region, reference.xlsx style)

---

## Step 1: Semantic Breakdown (PARALLEL subagents)

Run the full semantic breakdown pipeline on both traces simultaneously.
Each breakdown agent now produces a complete `semantic_labels.json`
independently -- including LLM-driven labeling using trace tree context.

### 1.1 Read the Agent Definition

Read the breakdown agent file:
`TraceLens/AgenticMode/SemanticComparison/.cursor/agents/semantic-breakdown-agent.md`

### 1.2 Launch Both Breakdown Subagents in PARALLEL

Launch **both** subagents simultaneously using the Task tool. Do NOT wait
between invocations.

**Task prompt structure for each subagent:**

```
---BEGIN AGENT INSTRUCTIONS---
<full contents of semantic-breakdown-agent.md>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Trace path: <trace_path>
- Trace name: <trace_name>
- Output directory: <output_dir>/_work/<trace_name>/

Follow the agent instructions above to complete the breakdown.
Return the summary described in the "Return Value" section.
```

**Output directories (all intermediates go under `_work/`):**
- Trace A: `<output_dir>/_work/<name_a>/`
- Trace B: `<output_dir>/_work/<name_b>/`

**For vLLM traces (--split-vllm):** Output has per-region subdirs, e.g.:
- `<output_dir>/_work/<name_a>/prefill_only_3072/`, `decode_only_4/`, `prefill_decode_3072_1/`
- Each region has its own full pipeline output including `semantic_labels.json`

### 1.3 Wait for Both Subagents to Complete

Both subagents must complete before proceeding to Step 2.

### 1.4 Verify Breakdown Outputs

After both subagents complete:

1. Check that each output directory contains `extracted.json`, `pattern.json`,
   `classified.json`, `tree_context.json`, `breakdown_context.json`,
   `llm_labels.json`, and `semantic_labels.json`
2. Check each subagent's return status for SUCCESS/ERROR
3. If either failed, report the error and stop

```python
import os

required_files = [
    'extracted.json', 'tree_context.json', 'pattern.json',
    'classified.json', 'breakdown_context.json', 'llm_labels.json',
    'semantic_labels.json',
]
for trace_dir in ['<output_dir>/_work/<name_a>', '<output_dir>/_work/<name_b>']:
    for fname in required_files:
        fpath = os.path.join(trace_dir, fname)
        assert os.path.exists(fpath), f"Missing {fpath}"
    print(f"{trace_dir}: all prerequisite files present")
```

---

## Step 2: Semantic Harmonization

After both traces have been independently labeled, harmonize labels across
traces to ensure cross-trace consistency.

### 2.1 Read the Agent Definition

Read the harmonization agent file:
`TraceLens/AgenticMode/SemanticComparison/.cursor/agents/semantic-harmonization-agent.md`

### 2.2 Launch Harmonization Agent

Launch a single Task subagent for harmonization.

**Task prompt structure:**

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

**For multi-region mode (vLLM):** Run the harmonization agent once per
matching region pair. For each region that appears in both trace output
directories, launch a harmonization with the region's subdirectory paths.

### 2.3 Verify Harmonization

After the harmonization agent completes:

1. Check that `alignment.json`, `harmonization_context.json`, and
   `harmonization_corrections.json` exist in `<output_dir>/_work/`
2. Verify both `semantic_labels.json` files have been updated
3. Check the agent's return status

---

## Step 3: Generate TraceDiff Output [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_semantic_diff.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison_report_csvs/
```

Produces in `<output_dir>/_work/comparison_report_csvs/`:
- `diff_stats.csv` -- per-kernel rows matching TraceDiff schema
- `diff_stats_unique_args_summary.csv` -- aggregated by semantic block
- `cpu_op_map.json`, `cpu_op_map_trace1.json`, `cpu_op_map_trace2.json`

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

**Multi-region mode (vLLM per steady-state):**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/match_and_compare.py \
    --regions-dir-a <output_dir>/_work/<name_a> \
    --regions-dir-b <output_dir>/_work/<name_b> \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison.csv
```
Matches regions by subdir name (prefill_only_3072, decode_only_4, etc.); compares only corresponding regions.

The comparison CSV includes:
- `semantic_block`, `semantic_group`, `perf_category`, `algorithm_order`
- Per trace: `kernel_names`, `kernel_count`, `total_us`, `avg_us`, `pct`
- `ratio` (trace_a / trace_b) and `gap_us` (absolute difference)

---

## Step 5: Compute Priority Ranking [S]

```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/compute_priority.py \
    <output_dir>/_work/comparison.csv \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/priority.json --top 5
```

Output: `priority.json` with top N optimization targets ranked by
`pct * (ratio - 1)`.

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

**IMPORTANT:** Call this script exactly **once**, even for multi-region vLLM
traces. The script handles multi-region data automatically.

**Single-region mode:**
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_comparison_report.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --comparison <output_dir>/_work/comparison.csv \
    --diff-dir <output_dir>/_work/ \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/comparison_report.xlsx \
    --output-csvs-dir <output_dir>/_work/comparison_report_csvs
```

**Multi-region mode (vLLM per steady-state):** Pass a representative region's
`semantic_labels.json` for the kernel mapping, and `--diff-dir` pointing to the
parent directory containing per-region subdirs:
```bash
python TraceLens/AgenticMode/SemanticComparison/trace_comparison/generate_comparison_report.py \
    <output_dir>/_work/<name_a>/<representative_region>/semantic_labels.json \
    <output_dir>/_work/<name_b>/<representative_region>/semantic_labels.json \
    --comparison <output_dir>/_work/comparison.csv \
    --diff-dir <output_dir>/_work/comparison_report_csvs \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/comparison_report.xlsx \
    --output-csvs-dir <output_dir>/_work/comparison_report_csvs
```

The script auto-detects the `region` column in `comparison.csv` and creates
per-region comparison sheets. `_load_diff_stats()` scans region subdirs under
`--diff-dir` when no top-level `diff_stats.csv` exists.

Produces:

- `comparison_report.xlsx` -- multi-sheet Excel workbook
- `comparison_report_csvs/` -- individual CSV files per sheet

**Excel sheets:**

| Sheet | Content |
|-------|---------|
| `kernel_mapping` | Human-readable kernel-to-kernel mapping |
| `gpu_scope_metrics` | Per-region GPU busy/idle metrics |
| `comparison_<region>` | Per-region block-level timing (one sheet per region) |
| `comparison` | Combined block-level timing (all regions) |
| `diff_stats` | TraceDiff-compatible per-kernel diff |
| `diff_stats_summary` | Aggregated diff stats |

Comparison sheets include a color-coded `<name_a> vs <name_b> %` column:
green = trace A faster, yellow = 0-10% gap, red = >10% gap.

---

## Step 8: Write Gap Analysis Report [LLM]

Using `_work/comparison.csv`, `_work/priority.json`, and both
`_work/<name>/semantic_labels.json` files (plus TraceDiff outputs in `_work/`),
produce `<output_dir>/semantic_comparison_report.md`.

The report follows the **standalone analysis visual language**: P1/P2/P3
priorities with icons, Issue/Action/Impact structure, Executive Summary,
Detailed Analysis sections.

**Report template:**

```markdown
# [Name A] vs [Name B]: Semantic Comparison Analysis

## Executive Summary

[1 paragraph overview: model architecture, platform comparison, overall ratio,
which trace is faster and by how much]

| Metric | [Name A] | [Name B] |
|--------|----------|----------|
| Total Iteration Time | X us | Y us |
| Model | architecture | architecture |
| Kernel Count | N | M |
| Blocks Compared | K matched, J only in one trace |
| Overall Ratio (A/B) | Z | - |

## Priority Improvement Targets

<!-- Icon mapping by PRIORITY NUMBER: P1=red, P2=yellow, P3+=green -->

### P1: <semantic_block> -- <perf_category>

**Issue**: [Name A] is Xx slower than [Name B] for this block (A: Y us vs B:
Z us).

**Action**: [Recommendation based on kernel timing analysis and perf_category.]

**Impact**: Xus gap, Y% of total [Name A] runtime.

---

### P2: <semantic_block> -- <perf_category>

**Issue**: [1-2 sentences]

**Action**: [1-2 sentences]

**Impact**: [gap_us savings, % of total]

---

### P3: <semantic_block> -- <perf_category>

**Issue**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [gap_us savings, % of total]

---

## Detailed Analysis: Kernel Matching

[For each semantic_group, then for each semantic_block within the group:]

### Self-Attention

#### Pre-Attn Norm

- **[Name A]**: `kernel_name_a` (N kernels, X us)
- **[Name B]**: `kernel_name_b` (M kernels, Y us)
- **Ratio**: Z (A/B)
- **Analysis**: [Why one is faster -- architecture features, kernel quality,
  implementation differences]

[... continue for all semantic blocks ...]

## Blocks Where [Name A] is Competitive or Faster

| semantic_block | perf_category | [Name A] (us) | [Name B] (us) | Ratio |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Blocks Present in Only One Trace

| semantic_block | Present In | Reason |
|---|---|---|
| ... | [Name A/B] only | ... |

## Key Takeaways

[3-5 bullet points:
1. Overall performance comparison
2. Largest improvement opportunities
3. Architecture/implementation differences driving the gap
4. Recommended next steps]
```

**For each semantic block, reason about:**

1. **Matching confidence**: Are these clearly the same operation?
2. **Fusion differences**: Does one platform fuse multiple ops into one kernel?
3. **Implementation differences**: Different libraries?
4. **Architecture advantages**: Hardware features that explain gaps?

---

## Step 9: Validate Report

- Check that all `_work/priority.json` entries appear in the P1/P2/P3 section
- Check that all semantic blocks from `_work/comparison.csv` appear in the detailed
  analysis
- Verify no blocks are missing from the report

Prompt the user to review the report.

---

## Agent File Map

**Base path:** `TraceLens/AgenticMode/SemanticComparison/.cursor/agents/`

| Agent | File | Purpose |
|-------|------|---------|
| Semantic Breakdown | `semantic-breakdown-agent.md` | Full breakdown + LLM labeling on a single trace |
| Semantic Harmonization | `semantic-harmonization-agent.md` | Cross-trace label alignment and harmonization |

---

## Output File Structure

The output directory contains **only** the two final deliverables.
All intermediate / working files are under `_work/`.

```
<output_dir>/
  comparison_report.xlsx                  # Final Excel workbook
  semantic_comparison_report.md           # Final stakeholder report
  _work/                                  # All intermediate outputs
    alignment.json                        # Cross-trace block alignment
    harmonization_context.json            # LLM harmonization context
    harmonization_corrections.json        # LLM harmonization corrections
    comparison.csv                        # Block-level comparison
    priority.json                         # Ranked optimization targets
    comparison_report_csvs/               # Per-sheet CSV exports
      kernel_mapping.csv
      comparison.csv
      diff_stats.csv
      diff_stats_summary.csv
      diff_stats_unique_args_summary.csv
      cpu_op_map.json
      cpu_op_map_trace1.json
      cpu_op_map_trace2.json
    <name_a>/<region>/
      extracted.json                      # Raw trace data
      tree_context.json                   # Tree context (cpu_op, nn_module)
      pattern.json                        # Repeating patterns
      classified.json                     # Kernel classification
      breakdown_context.json              # LLM breakdown context
      llm_labels.json                     # LLM-assigned labels
      semantic_labels.json                # Final labeled kernels
    <name_b>/<region>/
      ...                                 # Same structure as name_a
```

---

## Error Handling

### Before Step 1 (Breakdown)
- Verify both trace files exist and are readable
- If config.json was provided, verify it exists
- Create the output directory if it doesn't exist

### After Step 1 (Breakdown)
- Check each subagent's return status
- Verify all required files exist for both traces (see Step 1.4)
- If either breakdown failed, report the error and stop

### After Step 2 (Harmonization)
- Verify harmonization outputs exist
- Check that both `semantic_labels.json` files have been updated
- Verify label consistency across traces

### After Step 6 (Verify)
- If verification fails, report which assertions failed
- Common fixes: check label assignment, re-run harmonization

### In Final Report
- If any step failed, note it in the report
- Provide recommendations only for successfully compared blocks

---

## Key Principles

1. **Seamless flow** -- user provides two trace paths and the orchestrator
   handles everything from breakdown to final report
2. **Parallel breakdown** -- both traces are processed simultaneously as
   subagents for speed; each produces semantic_labels.json independently
3. **Single-trace labeling** -- each trace gets functional labels using tree
   context + LLM knowledge, no cross-trace dependency for initial labeling
4. **Cross-trace harmonization** -- a separate step aligns and unifies labels
   across traces for consistent comparison
5. **Vendor-agnostic language** -- use generic terms for all recommendations
6. **Standalone visual language** -- P1/P2/P3 priorities, Issue/Action/Impact
   format, Executive Summary structure
7. **Complete coverage** -- every semantic block appears in the report
8. **No script creation** -- subagents must use only existing scripts. They
   must never create new Python scripts or other new files.
