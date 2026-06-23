<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Compare two traces

This guide shows how to quantify the impact of a change by comparing two
TraceLens reports side by side — for example, a baseline against a candidate
after a code, library, or hardware change.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- Two generated reports (`.xlsx`), one per configuration. Generate them with
  [TraceLens_generate_perf_report_pytorch](./generate-perf-report.md).

## Step 1: Generate the two reports

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path traces/baseline.json --output_xlsx_path baseline.xlsx
TraceLens_generate_perf_report_pytorch --profile_json_path traces/candidate.json --output_xlsx_path candidate.xlsx
```

## Step 2: Run the comparison

```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

- `--names` sets the display tags used in the comparison sheets (the count must
  match the number of reports).
- `--sheets` selects sheet groups: `gpu_timeline`, `ops_summary`,
  `kernel_summary`, `ops_all`, `roofline`, or `all`.
- `-o` sets the output workbook path; add `--output_csvs_dir` to also emit CSVs.

**Expected output:** `comparison.xlsx`, a workbook with side-by-side columns for
each report. Because the comparison is performed at the CPU-dispatch level, it
produces meaningful diffs even across different hardware or software versions.

You can also pass directories of per-sheet `.csv` files instead of `.xlsx`
reports, and compare more than two reports at once.

## Step 3: Interpret the comparison

Use the comparison sheets to find operations or categories that regressed or
improved, shifts in the GPU-timeline breakdown (for example, more idle time or
exposed communication), and changes in roofline efficiency.

## Alternative: inline diff during report generation

To compare against a second trace while generating the primary report, pass
`--comparison_json_path`. This runs TraceDiff and adds speedup, delta, and LCA
columns to `unified_perf_summary`, plus a `diff_stats` sheet:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path traces/candidate.json \
    --comparison_json_path traces/baseline.json
```

See `docs/TraceDiff.md` in the repository for the tree-based comparison details.
