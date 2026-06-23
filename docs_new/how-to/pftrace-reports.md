<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Analyze rocprofv3 pftrace files

This guide shows how to generate reports from Perfetto-style `.pftrace` files
produced by `rocprofv3 --output-format pftrace`. TraceLens provides three
complementary pftrace reports: activity, API↔kernel, and memory copy.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A `.pftrace` (or `.json` / `.json.gz`) trace.
- `traceconv` is required to convert `.pftrace` to JSON. You do not need to
  install it manually: TraceLens uses it from `PATH` if present, otherwise
  downloads it automatically. You can also pass `--traceconv /path/to/traceconv`.

## Step 1: Record a pftrace (if needed)

```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace \
    --output-format pftrace -d ./v3_traces -- python3 your_app.py
```

## Step 2: Activity report (category summary)

Produces an NSYS-style per-GPU category summary plus kernel/HIP/XLA summaries,
with optional Markdown output:

```bash
TraceLens_generate_perf_report_pftrace_hip_activity \
    --trace_path sample.pftrace \
    --write_md
```

- `--write_md` (or `--output_md_path`) writes a Markdown report.
- `--merge_kernels` merges kernel names by stripping digits.
- `--min_event_ns` drops events shorter than the given nanosecond threshold
  (default 5000).
- `--kernel_summary_include_rccl` includes RCCL kernels in the kernel summary.

## Step 3: API↔kernel report (latency breakdown)

Correlates HIP API calls with the kernels they launch and reports the latency
breakdown `T = A + Q + K` (API duration, queue delay, kernel duration):

```bash
TraceLens_generate_perf_report_pftrace_hip_api --trace_path sample.pftrace
```

- `--allow_multi_kernel_per_api` allows multiple kernels per API correlation ID.
- `--include_nonlaunch_apis` includes API rows that did not launch a kernel.
- `--exclude_kernel_regex` excludes kernel names matching a regex (defaults to
  the redzone checker kernel).

A large queue-delay (`Q`) component relative to kernel duration (`K`) suggests
launch overhead or host-side bottlenecks rather than slow kernels.

## Step 4: Memory-copy report

Summarizes memory copies grouped by `copy_bytes`, with direction (h2d, d2h, d2d)
and the GPUs involved:

```bash
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace
```

**Expected output:** for each report, an Excel workbook (or CSVs with
`--output_csvs_dir`) is written next to the trace; the activity report can also
emit Markdown.

## Next steps

- See `docs/generate_perf_report_rocprof_pftrace.md` in the repository for the
  full pftrace report reference.
