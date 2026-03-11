---
name: Workflow LLM Eval
description: Run LLM-based workflow evals on a standalone analysis output and produce a results CSV.
triggers:
  - workflow LLM eval
  - run workflow LLM eval
tools:
  - file_read
  - file_write
---

# Workflow LLM Eval

Evaluate the standalone analysis report for correct formatting, content structure, and template adherence.

## Inputs

When triggered, the prompt will specify:
- **output_dir**: path to the `analysis_output/` directory to evaluate
- **results_path**: path to write the results CSV

## Files to Read

Read ALL of these before evaluating:

- `<output_dir>/standalone_analysis.md`
- `<output_dir>/perf_report_csvs/gpu_timeline.csv`
- `<output_dir>/category_data/category_manifest.json`
- All `<output_dir>/category_findings/*_findings.md`
- All `<output_dir>/system_findings/*_findings.md`

## Evals

Evaluate ALL 5 checks below. Write ALL 5 rows to the results CSV. If a file listed above does not exist, skip it gracefully (do not FAIL solely because an optional system findings file is absent).

### workflow_eval_8: Report Template Rendering

**Category:** Workflow
**Issue Summary:** Report Template Rendering

Read `standalone_analysis.md` and check for the presence of ALL these section headers (exact or close match):

- `## Executive Summary`
- `## Compute Kernel Optimizations`
- `## System-Level Optimizations`
- `## Detailed Analysis: Compute Kernels`
- `## Detailed Analysis: System-Level`
- `## Appendix`

Also verify the Executive Summary contains a markdown table (lines with `|`).

**PASS** if all headers found. **FAIL** with list of missing headers.

### workflow_eval_9: Executive Summary has metrics table

**Category:** Workflow
**Issue Summary:** Executive Summary has metrics table

In `standalone_analysis.md`, find the Executive Summary section. Verify the metrics table contains rows for:

- Total Compute Time (or Total Time)
- Computation (percentage)
- Idle Time (percentage)
- Exposed Communication (percentage)
- Top Bottleneck Category

Cross-check numeric values against `gpu_timeline.csv`:
- Computation % should match `computation_time_percent` within 1%
- Idle % should match `idle_time_percent` within 1%

**PASS** if all rows present and values align. **FAIL** with specifics.

### workflow_eval_10: Issue Template rendering

**Category:** Workflow
**Issue Summary:** Issue Template rendering

In `standalone_analysis.md`, find every priority item (lines matching `### ... P1:`, `### ... P2:`, `### ... P3:`, etc.). For each P-item, determine which section it belongs to and verify it has the correct fields:

**Compute Kernel P-items** (under `## Compute Kernel Optimizations`):
- **Issue**: 1-2 sentences describing the problem
- **Action**: 1-2 sentences describing what to do
- **Impact**: expected improvement (e.g., "~X.X ms savings..." or "Not quantifiable from trace data")

**System-Level P-items** (under `## System-Level Optimizations`):
- **Issue**: 1-2 sentences describing the problem
- **Action**: 1-2 sentences describing what to do
- (No **Impact** field — system-level issues are not quantified)

**PASS** if every P-item has the correct fields for its section. **FAIL** listing which P-items are missing required fields or have unexpected fields.

### workflow_eval_11: Hardware Reference in Appendix

**Category:** Workflow
**Issue Summary:** Hardware Reference in Appendix

In `standalone_analysis.md`, find the `## Appendix` section. Verify it contains:

- Platform name
- Peak HBM BW value
- At least one Peak MAF value

**PASS** if all present. **FAIL** with what's missing.

### workflow_eval_12: Sub-agent findings structure and Impact Summary types

**Category:** Workflow
**Issue Summary:** Sub-agent findings structure and Impact Summary types

Read `category_manifest.json` to get categories by tier. For each **compute_kernel** category, read `category_findings/<cat>_findings.md`. For each **system** category, read `system_findings/<cat>_findings.md`.

**Compute kernel findings** must each contain:

- An operations table (markdown table with columns like Operation, Count, Time)
- At least one key bottleneck section with Time, Efficiency, and recommendation text
- An `## Impact Summary` section with a markdown table where every data row has Type = `kernel_tuning`. Rows with Type = `algorithmic` or `system` are **not allowed**. The table may have zero data rows (header only) if no actionable bottlenecks were found.

**System findings** (e.g. `cpu_idle_findings.md`, `multi_kernel_findings.md`) must each contain:

- An `## Impact Summary` section with the table header row but **zero data rows**. System-level analyses do not produce impact estimates.

**PASS** if all findings have the required structure and valid Impact Summary types. **FAIL** listing which categories have missing structure or invalid Impact Summary rows (e.g. `algorithmic` or `system` type entries in compute findings, or data rows in system findings).

## Output

Write a CSV to the specified `results_path` with exactly these 5 columns and 5 data rows:

`index,category,issue_summary,result,details`

Use `workflow_eval_8` through `workflow_eval_12` as the `index` values. Do not add any other columns.
