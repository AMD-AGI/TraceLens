---
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
myst:
    html_meta:
        "description": "Learn how to build a performance report from AMD ROCm rocprofiler-sdk traces (rocprofv3 JSON and Perfetto pftrace) using TraceLens."
        "keywords": "TraceLens, rocprofv3, rocprofiler-sdk, ROCm, pftrace, Perfetto, GPU trace, kernel summary, performance report, AMD Instinct, HIP"
---

# Generate a rocprof performance report

Build a performance report from AMD ROCm rocprofiler-sdk traces. TraceLens
supports both `rocprofv3` JSON results and Perfetto-style `.pftrace` files.

## Prerequisites

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- A `rocprofv3` trace — either a `*_results.json` file or a `.pftrace` file.

## rocprofv3 JSON

Generate a report from a rocprofv3 `*_results.json` trace captured with the AMD
ROCm rocprofiler-sdk:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace_results.json \
    --short_kernel_study \
    --kernel_details
```

The report sheets are:

| Sheet | Description |
|-------|-------------|
| `gpu_timeline` | GPU activity: kernel execution, memory operations, and idle time. |
| `kernel_summary` | Per-kernel statistics — count, total/mean/median/std/min/max duration, and percentage of total kernel time. |
| `kernel_summary_by_category` | Kernel time grouped by category (GEMM, elementwise, attention, …). |
| `kernel_details` | Per-dispatch grid/block dimensions — added with `--kernel_details`. |
| `short_kernels_summary`, `short_kernel_histogram` | Short-kernel table and histogram — added with `--short_kernel_study`. |

Options:

- `--kernel_details` includes per-kernel grid/block dimensions; pair with
  `--topk_kernels N` to limit to the top N kernels by time.
- `--short_kernel_study` adds short-kernel analysis (tune with
  `--short_kernel_threshold_us` and `--short_kernel_histogram_bins`).
- Kernel-summary sheets are on by default; disable with `--disable_kernel_summary`.
- `--output_xlsx_path` / `--output_csvs_dir` control output.

Kernels are categorized from name patterns (GEMM, elementwise, reduction,
convolution, normalization, attention, memory, and other).

Because rocprofv3 captures GPU activity directly, it has **no CPU call stack or
operator hierarchy**, categorization is name-based rather than semantic, and
PyTorch-specific metadata (input shapes, op args) is not available. For
operator-level shape analysis, use a PyTorch trace instead. The two sources
compare as follows:

| Feature | rocprofv3 | PyTorch profiler |
|---------|-----------|------------------|
| Format | rocprofiler-sdk JSON | Chrome Trace Event |
| Kernel names | Direct from ROCm | Via PyTorch ops |
| Grid/block dims | Available | Available |
| CPU operations | Limited | Full trace |
| API calls | HIP / HSA | CUDA runtime |
| CPU call-stack tree | Not available | Available |

**Troubleshooting:**

- *"Not a valid rocprofv3 file"* — ensure the input is a `*_results.json` file
  from rocprofv3, not another format.
- *"No kernel events found"* — the trace captured no GPU activity; check that GPU
  work ran during profiling and that rocprofv3 recorded kernel dispatches.
- *openpyxl not installed* — install it for Excel output (`pip install openpyxl`)
  or use `--output_csvs_dir` for CSV output instead.

## rocprofv3 pftrace

For Perfetto-style `.pftrace` files produced by
`rocprofv3 --output-format pftrace`, TraceLens provides three complementary
reports. Each takes `--trace_path` and writes an Excel workbook (or CSVs with
`--output_csvs_dir`) next to the trace. Accepted inputs are `.pftrace` (Perfetto
binary), `.json` (Perfetto-style JSON with a `traceEvents` array), and `.json.gz`.

```{note}
`.pftrace` input requires `traceconv` to convert the trace to JSON. You do not
need to install it manually: TraceLens uses it from `PATH` if present, otherwise
downloads it into the trace's directory. You can also pass
`--traceconv /path/to/traceconv`.
```

**Activity report** — an NSYS-style per-GPU category summary plus kernel/HIP/XLA
summaries, with optional Markdown output:

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

**API↔kernel report** — correlates HIP API calls with the kernels they launch
and reports the latency breakdown `T = A + Q + K` (total = API duration + queue
delay + kernel duration):

```bash
TraceLens_generate_perf_report_pftrace_hip_api --trace_path sample.pftrace
```

- `--allow_multi_kernel_per_api` allows multiple kernels per API correlation ID.
- `--include_nonlaunch_apis` includes API rows that did not launch a kernel.
- `--exclude_kernel_regex` excludes kernel names matching a regex (defaults to
  the redzone checker kernel).

A large queue-delay (`Q`) relative to kernel duration (`K`) points to launch
overhead or host-side bottlenecks rather than slow kernels.

**Memory-copy report** — summarizes memory copies grouped by `copy_bytes`, with
direction (`h2d`, `d2h`, `d2d`) and the GPUs involved:

```bash
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace
```

Each pftrace generator can also be imported and called with a trace path,
returning a dictionary of pandas DataFrames (for example `api_kernel_summary`,
`category_summary`, `kernel_summary`, `hip_summary`).

## Next steps

- Quantify the effect of a change by [comparing two traces](./compare-traces.md).
- Analyze multi-GPU collectives with a
  [collective-communication report](./collective-report.md).
- Analyze [PyTorch](./generate-perf-report-pytorch.md) or
  [JAX](./generate-perf-report-jax.md) traces.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [API reference](../reference/api-reference.md)
