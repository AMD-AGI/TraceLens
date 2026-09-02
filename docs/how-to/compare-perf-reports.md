<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Compare performance reports in TraceLens
```{meta}
:description: Learn how to compare two or more TraceLens performance reports with the compare_perf_reports_pytorch tool to diff per-op, kernel, and roofline metrics in a single workbook.
:keywords: TraceLens, compare performance reports, compare_perf_reports_pytorch, xlsx, diff, ops_summary, kernel_summary, roofline, ROCm, PyTorch profiler
```

This topic shows how to compare multiple performance reports that you've already
generated with `generate_perf_report_pytorch.py`, using the
`compare_perf_reports_pytorch.py` script
(`TraceLens/Reporting/compare_perf_reports_pytorch.py`). The result is a single
workbook with side-by-side metrics and per-metric diffs.

```{note}
This topic covers comparing generated *reports* (the `.xlsx` or `.csv` outputs).
To compare raw traces by their morphological tree structure, see
[Compare two traces in TraceLens](./compare-traces.md), which also documents the
SDK-based `TraceDiff` workflow.
```

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- At least two reports (`.xlsx`) generated with
  [TraceLens_generate_perf_report_pytorch](./generate-perf-report-pytorch.md).

## What it takes in

The script accepts the following inputs.

| Input | Description |
|-------|-------------|
| `*.xlsx` files | TraceLens reports you want to compare. Provide at least two. |
| Optional `--names` | Human-readable tags for each report. If omitted, the script falls back to the base filenames. |

## How to call it

Pass the report files you want to compare along with any optional flags.

```bash
python compare_perf_reports.py \
    baseline.xlsx \
    candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

Alternatively, you can call the entry point directly after installing TraceLens,
which avoids needing to know the script's path:

```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx \
    candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

Common flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `-o, --output` | `comparison.xlsx` | Name of the merged workbook. |
| `--names` | `<file stem>` | Custom tags (must match the number of reports). |
| `--sheets` | `all` | Limit processing to a subset: `gpu_timeline`, `ops_summary`, `kernel_summary`, `ops_all`, `roofline`, or `all`. |

## What comes out

The script writes one workbook (`comparison.xlsx` unless you override it)
containing multiple sheets:

| Sheet | When you get it | What it shows |
|-------|-----------------|---------------|
| `gpu_timeline` | `--sheets gpu_timeline` or `all` | End-to-end GPU activity by type (`compute`, `memcpy`, and so on) with per-report timings plus `time ms__<tag>_diff` and `time ms__<tag>_pct`. |
| `ops_summary` | `--sheets ops_summary` or `all` | Per-op aggregates keyed on `name`. Sorted by the baseline's `total_direct_kernel_time_ms`. Unhelpful columns (for example cumulative %) are stripped from non-baseline reports. When comparing PyTorch traces, this sheet is generated. For rocprof traces without `ops_summary`, it automatically falls back to `kernel_summary`. |
| `kernel_summary` | `--sheets kernel_summary` or `all` | Kernel-level summary for rocprof reports, keyed on `name`. Similar to `ops_summary` but uses rocprof-specific columns like `Total Kernel Time (ms)` and `Count`. When `--sheets all` is used with rocprof reports, this is generated automatically if `ops_summary` is not available. |
| `ops_all_*` | `--sheets ops_all` or `all` | Three sheets per variant tag: `ops_all_intersect_<tag>` (op instances present in both baseline and variant), `ops_all_only_baseline_<tag>` (ops only the baseline ran), and `ops_all_only_variant_<tag>` (ops only the variant ran). Columns irrelevant to a given view are hidden, not deleted. |
| `<roofline>_*` | `--sheets roofline` or `all` | Same intersect / only-baseline / only-variant breakdown for each roofline group: `GEMM`, `SDPA_fwd`, `SDPA_bwd`, `CONV_fwd`, `CONV_bwd`, `UnaryElementwise`, `BinaryElementwise`. |

Hidden columns stay in the file (for power users) but are invisible in Excel by
default.

## Diff math

For every metric you ask it to track (`diff_cols` in the code), the script
computes:

```text
metric__<tag>_diff      # variant - baseline
metric__<tag>_pct       # 100 * diff / baseline
```

## Design decisions to know

The following behaviors are intentional and affect how you interpret the output.

- **Outer merge, never inner**: If an op vanished, you'll see it.
- **Baseline is the first report**: Choose its order deliberately.
- **Column prefixing**: Every metric becomes `<tag>::metric`, so you can safely concatenate arbitrary reports.
- **Sheet-specific pruning**: The script aggressively hides noise (for example `median`, `UID`) to keep the output readable. You can always unhide these in Excel if you need them.
- **Excel 31-char rule**: Sheet names are truncated to fit; no data loss, just shorter labels.

```{note}
A planned enhancement is morphology-aware diffing: understanding the call-stack
tree and comparing at the lowest common call-stack level. For example, if a
baseline leaf op is `cudnn_convolution` and the variant is `miopen_convolution`,
the diff would recognize that the lowest common level is `convolution` and
compare the two there.
```

## Related topics

- [Compare two traces in TraceLens](./compare-traces.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a rocprof performance report](./generate-perf-report-rocprof.md)
- [Generate a report](./generate-reports.md)
- [API reference](../reference/api-reference.md)
