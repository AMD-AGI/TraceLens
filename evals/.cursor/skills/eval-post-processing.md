<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: Eval Post Processing
description: Aggregate eval results, classify failures, and generate PR + fix-ticket reports from completed repeatability results.
triggers:
  - eval post processing
  - run eval post processing
tools:
  - terminal
  - file_read
  - file_write
---

# Eval Post Processing

Short-lived skill invoked after the repeatability harness finishes (or manually on existing results). Aggregates per-run CSVs, classifies failures, and writes two markdown reports.

## Inputs

The prompt provides these key=value parameters:

- **results_root**: path to the repeatability results tree (contains `<trace_id>/run_<n>/` directories)
- **suite**: `unit` or `e2e`
- **test_traces_csv**: path to the trace CSV used (e.g. `evals/e2e_test_traces.csv`)
- **report_dir**: where to write output reports
- **container**: Docker container name (used in reproducer commands)

## Step 4 — Aggregate

Run `aggregate_repeatability.py` to merge all per-run `eval_summary.csv` files and parse stream logs:

```bash
RESULTS_ROOT=<results_root> OUTPUT_DIR=<report_dir>/aggregates \
  python3 evals/eval_utils/aggregate_repeatability.py
```

This produces three files in `<report_dir>/aggregates/`:

- `aggregated_results.csv` — columns: `trace_id,run_id,eval_index,eval_category,issue_summary,result,details`
- `pass_rate_summary.csv` — columns: `trace_id,<eval_index>,...,overall_pass_rate` (values like `13/15` or `N/A`)
- `stream_diagnostics.csv` — columns: `trace_id,run_id,outcome,duration_ms,input_tokens,output_tokens,cache_read_tokens,turns,tool_calls,report_written,report_headers,last_step_reached`

Verify the script exits 0 and all three files exist before proceeding.

## Step 5 — Read and Classify

Read these files:

1. `<report_dir>/aggregates/aggregated_results.csv`
2. `<report_dir>/aggregates/pass_rate_summary.csv`
3. `<report_dir>/aggregates/stream_diagnostics.csv`
4. `evals/eval_utils/report_section_rules.yaml` — classification guide (JSON format despite `.yaml` extension)
5. `<test_traces_csv>` — for trace metadata (id, platform, trace_path)

From `aggregated_results.csv`, compute:

- **Global metrics**: count PASS, FAIL, MISSING rows; compute pass rate
- **Per-suite metrics**: same counts grouped by suite
- **Top failure issues**: group FAIL rows by `issue_summary`, count, sort descending
- **Per-trace failure counts**: group FAIL rows by `trace_id`, count, sort descending

### Classification

Use `report_section_rules.yaml` as a guide to classify each FAIL row into:

- **Base sections** (broad categories like "Reasoning", "Kernel Fusion", "Others"): match `eval_index`, `eval_category`, `issue_summary`, or `details` against `base_section_rules`. Unmatched rows go to "Others".
- **Standalone sections** (report sections like "Detailed Analysis", "Executive Summary"): match against `standalone_section_rules`. Unmatched rows go to "Detailed Analysis".
- **Failure modes** (likely cause + suggested fix): match against `failure_mode_rules`. You may improve on the suggested causes and fixes using your own judgment based on the failure details — the rules file is a starting point, not a rigid contract.

## Step 6 — Write Reports

Write two markdown files. Follow the exact structure below.

### `<report_dir>/pr_report.md`

```markdown
# Automated Eval Report (PR)

Generated at: `<ISO 8601 timestamp>`

| Metric | Value |
|---|---|
| Overall PASS | <count> |
| Overall FAIL | <count> |
| Overall MISSING | <count> |
| Overall pass rate | <percentage>% |

## Suite Summary

| Suite | PASS | FAIL | MISSING | Pass rate |
|---|---|---|---|---|
| <suite> | <pass> | <fail> | <missing> | <rate>% |

## Failure Sections

| Section | Failures |
|---|---|
| <base_section> | <count> |
...

## Sections

| Section | Failures |
|---|---|
| <standalone_section> | <count> |
...sorted descending by count

## Top Failure Issues

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...top 10, sorted descending
```

### `<report_dir>/fix_ticket_report.md`

```markdown
# Automated Eval Report (Fix Ticket)

Generated at: `<ISO 8601 timestamp>`

| Metric | Value |
|---|---|
| Overall PASS | <count> |
| Overall FAIL | <count> |
| Overall MISSING | <count> |
| Overall pass rate | <percentage>% |

## Suite: <suite_name>

### Sections

| Section | Failures |
|---|---|
| <standalone_section> | <count> |
...sorted descending

### Top Failure Issues

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...all issues, sorted descending

### Failure Modes (Concise)

| Issue | Count | Likely cause | Suggested fix |
|---|---|---|---|
| <issue_summary> | <count> | <cause> | <fix> |
...top 8 issues with cause/fix analysis

### Top Reproducers

| Trace/Case | Failures | Platform | Reproducer command |
|---|---|---|---|
| <trace_id> | <count> | <platform> | `CONTAINER=<container> NUM_REPEATS=1 MAX_PARALLEL=1 TEST_IDS="<trace_id>" TEST_TRACES_CSV="<absolute_csv_path>" bash evals/eval_scripts/run_repeatability_parallel.sh` |
...top 5 traces by failure count
```

For the reproducer commands:
- Use the `container` value from inputs
- Use the absolute path of `test_traces_csv`
- The `platform` comes from the test traces CSV

## Step 7 — Save and Summarize

1. Copy `<report_dir>` to `evals/eval_reports/latest/` (remove existing `latest/` first if present):
   ```bash
   rm -rf evals/eval_reports/latest
   cp -r <report_dir> evals/eval_reports/latest
   ```

2. Print a summary to the user:
   - Overall pass rate and PASS/FAIL/MISSING counts
   - Top 3 worst-performing traces
   - Top 3 failure issues
   - Paths to `pr_report.md` and `fix_ticket_report.md` for manual GitHub posting
