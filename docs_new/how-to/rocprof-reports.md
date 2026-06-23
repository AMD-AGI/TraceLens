<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Analyze rocprofv3 JSON traces

This guide shows how to generate a TraceLens report from a rocprofv3
`*_results.json` trace captured with the AMD ROCm rocprofiler-sdk.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A rocprofv3 `*_results.json` trace.

## Step 1: Capture a trace (if needed)

Use `rocprofv3` to record kernel and HIP activity for your application. Refer to
the ROCm rocprofiler-sdk documentation for the capture options appropriate to
your workload; the result is a `*_results.json` file.

## Step 2: Generate the report

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace_results.json \
    --short_kernel_study \
    --kernel_details
```

- `--kernel_details` includes per-kernel grid/block dimensions.
- `--short_kernel_study` adds short-kernel analysis (tune with
  `--short_kernel_threshold_us` and `--short_kernel_histogram_bins`).
- `--topk_kernels N` limits kernel details to the top N kernels by time.
- Kernel-summary sheets are enabled by default; disable them with
  `--disable_kernel_summary`.

**Expected output:** an Excel report with automatically categorized kernels (for
example, GEMM, attention, elementwise), kernel summaries, and — when requested —
short-kernel analysis and per-kernel grid/block detail.

## Step 3: Output to CSVs (optional)

To write per-sheet CSVs instead of (or in addition to) Excel:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace_results.json \
    --output_csvs_dir ./rocprof_csvs
```

## Next steps

- For Perfetto-style `.pftrace` output from rocprofv3, see
  [Analyze rocprofv3 pftrace files](./pftrace-reports.md).
- See `docs/generate_perf_report_rocprof.md` in the repository for full details.
