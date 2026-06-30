---
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
myst:
    html_meta:
        "description": "Learn how to compare two TraceLens performance reports side by side to quantify the impact of code, library, or hardware changes on GPU performance."
        "keywords": "TraceLens, trace comparison, TraceDiff, performance regression, GPU benchmark, ROCm, PyTorch profiler, diff, roofline, GEMM"
---

# Compare two traces in TraceLens

This guide shows how to quantify the impact of a change by comparing two
TraceLens reports side by side — for example, a baseline against a candidate
after a code, library, or hardware change.

TraceLens offers two ways to compare traces, depending on how precise you need
the matching to be:

- **[Perf-report comparison](#perf-report-comparison)** — compares two finished
  reports by **op name**, matching each sheet's rows (per-op aggregates, kernels,
  and modelled ops like GEMM/SDPA) across the two reports. CLI-driven, and
  produces a side-by-side Excel workbook.
- **[TraceDiff comparison](#tracediff-comparison-morphological)** — an SDK that
  compares traces by their **morphological tree structure**, matching ops at the
  lowest common node. Use it when op names differ between traces (e.g. across
  hardware, libraries, or framework versions) or when you need finer-grained,
  programmatic analysis.

## Prerequisites

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- Two generated reports (`.xlsx`), one per configuration. Generate them with
  [TraceLens_generate_perf_report_pytorch](./generate-perf-report-pytorch.md).

## Perf-report comparison

This approach compares two already-generated reports by matching rows on op
**name** within each sheet — per-op aggregates, kernel summaries, and modelled
ops such as GEMM and SDPA in the roofline groups.

### Step 1: Generate the two reports

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path traces/baseline.json --output_xlsx_path baseline.xlsx
TraceLens_generate_perf_report_pytorch --profile_json_path traces/candidate.json --output_xlsx_path candidate.xlsx
```

### Step 2: Run the comparison

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
each report. Rows are matched by op name, so this works best when the two runs
share the same op names (typically the same workload across configurations). When
names diverge — for example across hardware or library versions — use the
[TraceDiff comparison](#tracediff-comparison-morphological) instead.

You can also pass directories of per-sheet `.csv` files instead of `.xlsx`
reports, and compare more than two reports at once.

### Step 3: Read the comparison sheets

The output workbook contains one sheet per group you requested with `--sheets`:

| Sheet | When you get it | What it shows |
|-------|-----------------|---------------|
| `gpu_timeline` | `gpu_timeline` or `all` | GPU activity by type (compute, memcpy, etc.) with per-report timings plus `time ms__<tag>_diff` and `time ms__<tag>_pct`. |
| `ops_summary` | `ops_summary` or `all` | Per-op aggregates keyed on `name`, sorted by the baseline's total kernel time. Generated for PyTorch traces; for rocprof traces without `ops_summary` it falls back to `kernel_summary`. |
| `kernel_summary` | `kernel_summary` or `all` | Kernel-level summary for rocprof reports keyed on `name`, using rocprof columns like `Total Kernel Time (ms)` and `Count`. |
| `ops_all_*` | `ops_all` or `all` | Three sheets per variant tag: `ops_all_intersect_<tag>` (ops in both), `ops_all_only_baseline_<tag>` (baseline only), `ops_all_only_variant_<tag>` (variant only). |
| `<roofline>_*` | `roofline` or `all` | Same intersect / only-baseline / only-variant breakdown for each roofline group: `GEMM`, `SDPA_fwd`, `SDPA_bwd`, `CONV_fwd`, `CONV_bwd`, `UnaryElementwise` (tab `un_eltwise`), `BinaryElementwise` (tab `bin_eltwise`). |

For every tracked metric, the comparison adds two columns:

```text
metric__<tag>_diff      # variant - baseline
metric__<tag>_pct       # 100 * diff / baseline
```

Things to know when reading the workbook:

- **Outer merge, never inner** — if an op exists in only one report it still
  appears (in the `only_baseline` / `only_variant` sheets), so you can see ops
  that vanished or were newly introduced.
- **Baseline is the first report** you pass — choose its order deliberately.
- **Column prefixing** — every metric is written as `<tag>::metric`, so multiple
  reports can be compared safely.
- **Noise is hidden, not deleted** — columns like `median`, `std`, `min`, `max`,
  and `ex_UID` are hidden in Excel for readability; unhide them if you need them.

Use these to find operations or categories that regressed or improved, shifts in
the GPU-timeline breakdown (more idle time or exposed communication), and changes
in roofline efficiency.

## TraceDiff comparison (morphological)

Where the perf-report comparison matches rows by op name, `TraceDiff` compares
the two traces by their *morphological tree structure*. It is an SDK you call
directly, giving finer-grained, programmatic analysis.

### Why morphological diffing

Unlike a leaf-level operation comparison, TraceDiff considers the morphological
structure of each trace to find the **lowest common node** between them. This
matters when the same logical operation lowers to different leaf ops — for
example, `aten::convolution` becomes `aten::miopen_convolution` on ROCm and
`aten::cudnn_convolution` on CUDA. A leaf-level diff would treat these as
unrelated; TraceDiff matches them at the `convolution` level, making it suitable
for comparing across hardware, libraries, or framework versions.

### Compare two traces with the SDK

```python
from TraceLens import TreePerfAnalyzer, TraceDiff

# Build a tree for each trace
tree1 = TreePerfAnalyzer.from_file("/path/to/baseline.json").tree
tree2 = TreePerfAnalyzer.from_file("/path/to/candidate.json").tree

td = TraceDiff(tree1, tree2)          # tree1 is the baseline
td.generate_tracediff_report()        # builds DataFrames; writes no files
td.print_tracediff_report_files("rprt_diff", prune_non_gpu=True)  # writes files
```

`generate_tracediff_report()` populates the result DataFrames in memory;
`print_tracediff_report_files()` writes them to disk (pass `prune_non_gpu=True`
to drop subtrees with no GPU work).

```{note}
TraceDiff operates on PyTorch profiler JSON traces, which it builds into the
`TraceToTree` objects it compares.
```

### Output files

| File | Contents |
|------|----------|
| `merged_tree_output.txt` | Text visualization of the merged tree, showing matched and unmatched nodes. |
| `diff_stats.csv` | Detailed per-op-instance statistics: input shapes, types, kernel times, and kernel names. |
| `diff_stats_unique_args_summary.csv` | Statistics aggregated per op name plus unique argument combination. |
| `cpu_op_map_trace1.json` / `cpu_op_map_trace2.json` | Per-trace map from each CPU op to its kernels. |

### Access DataFrames and UID mapping

After `generate_tracediff_report()`, the same data is available as DataFrames,
and `merged_uid_map` cross-references events between the two trees:

```python
df_stats = td.diff_stats_df                       # detailed per-op
df_args  = td.diff_stats_unique_args_summary_df    # per op name + unique args

# Find the tree2 UID corresponding to a tree1 UID (-1 if no match)
uid1 = next(iter(td.baseline.cpu_root_nodes))
uid2 = td.get_corresponding_uid(1, uid1)
```

### Inline diff during report generation

To run TraceDiff automatically while generating the primary PyTorch report, pass
`--comparison_json_path`. This adds speedup, delta, and LCA columns to
`unified_perf_summary`, plus a `diff_stats` sheet:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path traces/candidate.json \
    --comparison_json_path traces/baseline.json
```

See the
[`trace_diff_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/trace_diff_example.ipynb)
notebook for a worked example.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a JAX performance report](./generate-perf-report-jax.md)
- [Generate a rocprof performance report](./generate-perf-report-rocprof.md)
- [Generate a collective-communication report](./collective-report.md)
- [API reference](../reference/api-reference.md)
