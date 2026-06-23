<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Generate a PyTorch performance report

This guide shows how to produce a TraceLens performance report from a PyTorch
profiler trace and how to read the most important sheets.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A PyTorch profiler Chrome trace (`.json`, `.json.gz`, or `.zip`). If you do
  not have one, use a bundled demo trace under `tests/traces/` or follow the
  PyTorch profiling guide in the repository.

## Step 1: Generate the report

Run the report generator on your trace:

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```

**Expected output:** an Excel workbook (`.xlsx`) is written next to the trace.
It contains several sheets, including a GPU timeline, operator-category and
operator summaries, a unique-argument table, and roofline metrics.

To control the output location or also emit CSVs:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/your/trace.json \
    --output_xlsx_path report.xlsx \
    --output_csvs_dir ./report_csvs
```

## Step 2: Read the GPU timeline

The `gpu_timeline` sheet breaks GPU time into computation, communication, memory
copy, and idle time. A low computation percentage with significant idle time
indicates an inefficient workload with room to improve compute/communication
overlap. Use `--micro_idle_thresh_us` to separate very short idle gaps into
their own category.

## Step 3: Drill into operator categories

- `ops_summary_by_category` groups the CPU operations that launch kernels into
  broad categories (for example, GEMM, convolution forward, convolution
  backward). This is the most aggregated view and quickly shows which category
  dominates GPU time.
- The category-level sheets break the same data down further so you can find the
  specific operations within a category that drive inefficiency.

## Step 4: Inspect unique argument shapes

The `ops_unique_args` sheet is the most detailed view. It groups operations by
name plus the combination of input shape, dtypes, strides, and concrete inputs,
which helps you find the input patterns responsible for slow performance. For
example:

| Operation | Input dims |
|-----------|------------|
| `aten::miopen_batch_norm` | `((256, 64, 8, 8), (64,), (64,), (64,), (64,), (), (), ())` |
| `aten::miopen_batch_norm` | `((256, 256, 8, 8), (256,), (256,), (256,), (256,), (), (), ())` |

Comparing the two rows, the second is slower, as shown by the kernel median run
time.

## Step 5: Optional deeper analysis

- `--enable_kernel_summary` adds a kernel-summary sheet.
- `--short_kernel_study` adds a short-kernel study (tune with
  `--short_kernel_threshold_us` and `--short_kernel_histogram_bins`).
- `--detect_recompute` flags activation recomputation (checkpointing).
- `--include_overlap_info` adds kernel-overlap sheets.

See [Performance report column definitions](https://github.com/AMD-AGI/TraceLens/blob/main/docs/perf_report_columns.md)
in the repository for what each column means.

## Next steps

- Add efficiency metrics with [roofline analysis](./roofline-analysis.md).
- Quantify the effect of a change by [comparing two traces](./compare-traces.md).
