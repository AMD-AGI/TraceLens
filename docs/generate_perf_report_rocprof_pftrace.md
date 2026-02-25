<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Generate Performance Reports from pftrace (rocprofv3 / Perfetto)

For **Perfetto-style** traces (e.g. rocprofv3 with `--output-format pftrace`, or JSON with a `traceEvents` array), TraceLens provides two report generators that share the same trace loading and optional **traceconv** handling (for `.pftrace` → JSON conversion).

## CLI Reports

| CLI | Description | Output |
|-----|-------------|--------|
| `TraceLens_generate_perf_report_pftrace_hip_activity` | Category summary per GPU, kernel/HIP/XLA summaries (NSYS-style), optional Markdown | Excel, CSV, optional `.md` |
| `TraceLens_generate_perf_report_pftrace_hip_api` | HIP API ↔ kernel correlation; latency breakdown **T = A + Q + K** (API duration, queue delay, kernel duration) | Excel, CSV |

## Input Formats

- `.json` — Perfetto-style JSON with a `traceEvents` array
- `.json.gz` — Gzip-compressed JSON
- `.pftrace` — Perfetto binary format

For `.pftrace` files, **traceconv** is resolved from `PATH` or downloaded into the trace file's directory if not provided via `--traceconv`.

## Quick Start

### 1. Record a pftrace

```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace \
    --output-format pftrace -d ./v3_traces -- python3 your_app.py
```

### 2. Generate reports

```bash
# Activity report: category summary per GPU, kernel/HIP/XLA summaries (NSYS-style). Optional Markdown output.
TraceLens_generate_perf_report_pftrace_hip_activity --trace_path sample.pftrace --write_md

# API↔Kernel report: latency breakdown T = A + Q + K (API duration, queue delay, kernel duration)
TraceLens_generate_perf_report_pftrace_hip_api --trace_path sample.pftrace

# Memory copy report: count per copy_bytes with direction (h2d/d2h/d2d and which GPU(s))
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace

# Custom Excel path:
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace --output_xlsx_path report.xlsx

# CSV directory:
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace --output_csvs_dir ./out
```

## Library Usage

Both scripts can be imported and called with a trace path; they return a dictionary of pandas DataFrames (e.g. `api_kernel_summary`, `category_summary`, `kernel_summary`, `hip_summary`).

## Testing

```bash
python -m pytest tests/test_pftrace_hip_api_perf_report.py tests/test_pftrace_hip_activity_report.py -v
```
