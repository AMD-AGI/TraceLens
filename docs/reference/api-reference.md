---
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
myst:
    html_meta:
        "description": "Complete reference for TraceLens command-line tools and Python SDK modules, including arguments and output descriptions for all report generators."
        "keywords": "TraceLens, API reference, command-line tools, Python SDK, ROCm, GPU trace, performance report, PyTorch, JAX, rocprofv3, roofline, CLI"
---

# TraceLens API reference

TraceLens exposes two complementary interfaces:

- A set of **command-line tools** (installed as `console_scripts`) for
  generating and comparing reports.
- A **Python SDK** for building custom analysis workflows.

This page documents the command-line tools and their main arguments, and
summarizes the SDK modules. Run any tool with `--help` for the complete,
version-specific argument list.

## Command-line tools

The following sections document each tool's main arguments and expected output.

### TraceLens_generate_perf_report_pytorch

Generate a multi-sheet Excel report from a PyTorch (`torch.profiler`) trace.

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_json_path` | required | Path to the `profile.json` or `.json.gz` trace. |
| `--output_xlsx_path` | auto | Path to the output Excel file. |
| `--output_csvs_dir` | None | Directory to write per-sheet CSV files instead of/with Excel. |
| `--enable_kernel_summary` | off | Add a kernel-summary sheet. |
| `--short_kernel_study` | off | Add a short-kernel study; tune with `--short_kernel_threshold_us`, `--short_kernel_histogram_bins`, `--topk_short_kernels`. |
| `--disable_coll_analysis` | on | Disable the collective-analysis section (enabled by default). |
| `--include_unlinked_kernels` | off | Include kernels with no linked CPU op in the GPU-timeline analysis. |
| `--micro_idle_thresh_us` | None | Threshold (µs) to classify an idle interval as micro-idle. |
| `--comparison_json_path` | None | Second trace to compare against; runs TraceDiff and adds speedup/delta/LCA columns plus a `diff_stats` sheet. |
| `--enable-origami` | off | Use Origami for simulated GEMM/SDPA times when a GPU arch JSON is provided. |
| `--detect_recompute` | off | Detect activation recomputation and add an `is_recompute` column. |
| `--include_overlap_info` | off | Add kernel-overlap sheets. |
| `--topk_ops`, `--topk_roofline_ops` | None | Limit rows in the unique-args and roofline tables. |
| `--extension_file` | None | Custom extensions for `TraceTree` and `PerfModel`. |

**Output:** an `.xlsx` workbook with the GPU-timeline, operator-category,
operator, unique-argument, and roofline sheets.

### TraceLens_generate_perf_report_pytorch_inference

Inference-oriented variant of the PyTorch report.

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_json_path` | required | Path to the trace. |
| `--group_by_parent_module` | off | Group kernel-launcher summaries by parent `nn.Module`. |
| `--capture_folder` | None | Path to the capture-trace folder. |
| `--include_overlap_info` | off | Add `*_kl_overlap` sheets when data exists. |

Shares most options with `TraceLens_generate_perf_report_pytorch` (output
paths, short-kernel study, roofline/Origami, comparison, call stack). Run with
`--help` for the full list.

### TraceLens_generate_perf_report_jax

Generate a report from a JAX XPlane protobuf trace (also accepts a PyTorch
trace).

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_path` | required | Path to the trace (`trace.json` or JAX `xplane.pb`). |
| `--output_xlsx_path` | auto | Output Excel file. |
| `--output_csvs_dir` | None | Directory for CSV output. |
| `--kernel_metadata_keyword_filters` | None | Only analyze events whose metadata contains the given keyword(s), for example `remat checkpoint`. |
| `--enable-origami` | off | Use Origami simulated GEMM/SDPA times when a GPU arch JSON is provided. |

### TraceLens_generate_perf_report_rocprof

Generate a report from a rocprofv3 `*_results.json` trace.

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_json_path` | required | Path to the rocprofv3 `*_results.json` trace. |
| `--output_xlsx_path` / `--output_csvs_dir` | auto / None | Output destinations. |
| `--kernel_details` | off | Include per-kernel detail with grid/block dimensions. |
| `--disable_kernel_summary` | on | Disable kernel-summary sheets (enabled by default). |
| `--short_kernel_study` | off | Add short-kernel analysis; tune with `--short_kernel_threshold_us`, `--short_kernel_histogram_bins`. |
| `--topk_kernels` | all | Limit kernel details to the top K kernels by time. |

### TraceLens_compare_perf_reports_pytorch

Compare two or more previously generated reports.

| Argument | Default | Description |
|----------|---------|-------------|
| `reports` | required | One or more TraceLens reports: `.xlsx` files or directories of per-sheet `.csv` files. |
| `--names` | None | Optional display tags for each report (count must match). |
| `--sheets` | `all` | Sheet groups to compare: `gpu_timeline`, `ops_summary`, `kernel_summary`, `ops_all`, `roofline`, or `all`. |
| `-o`, `--output` | `comparison.xlsx` | Output Excel file. |
| `--output_csvs_dir` | None | Also write each comparison sheet as a CSV here. |

### TraceLens_generate_multi_rank_collective_report_pytorch

Generate a collective-communication report across ranks.

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_dir` | — | Directory containing per-rank trace files. |
| `--trace_pattern` | — | Template path with a single `*` placeholder for rank. |
| `--trace_glob` | — | Glob for arbitrarily named trace files (requires `--world_size`, uses `--rank_regex`). |
| `--world_size` | required | Number of ranks. |
| `--agg_metrics` | `mean median min max` | Aggregation metrics in the summary. |
| `--gpus_per_node` | auto | Adds `node_id`/`node_span` columns and labels each process group `intra_node` or `inter_node`. |
| `--all2allv_heatmap` | off | Add an `nccl_all2allv_heatmap` sheet with per rank-pair send volumes. |
| `--use_multiprocessing` / `--max_workers` | off / cpu_count | Parallel trace loading. |

`--trace_dir`, `--trace_pattern`, and `--trace_glob` are mutually exclusive
ways to locate the per-rank traces.

### TraceLens_generate_perf_report_pftrace_hip_activity

Per-GPU category and kernel/HIP/XLA activity report from a Perfetto-style
trace.

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_path` | required | Path to `.json`, `.json.gz`, or `.pftrace`. |
| `--write_md` / `--output_md_path` | off / None | Write a Markdown report. |
| `--merge_kernels` | off | Merge kernel names by stripping digits. |
| `--min_event_ns` | 5000 | Drop events shorter than this (ns). |
| `--kernel_summary_baseline` | `total` | Baseline for the kernel summary. |
| `--kernel_summary_group` | `config` | Grouping key for the kernel summary. |
| `--kernel_summary_include_rccl` | off | Include RCCL kernels in the kernel summary. |
| `--traceconv` | auto | Path to `traceconv` (auto-resolved or downloaded for `.pftrace`). |

### TraceLens_generate_perf_report_pftrace_hip_api

HIP API ↔ kernel correlation with the latency breakdown `T = A + Q + K` (API
duration, queue delay, kernel duration).

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_path` | required | Path to `.json`, `.json.gz`, or `.pftrace`. |
| `--output_xlsx_path` / `--output_csvs_dir` | auto / None | Output destinations. |
| `--exclude_kernel_regex` | redzone checker | Regex of kernel names to exclude. |
| `--allow_multi_kernel_per_api` | off | Allow multiple kernels per API correlation ID. |
| `--include_nonlaunch_apis` | off | Include API rows that have no linked kernel. |
| `--traceconv` | auto | Path to `traceconv`. |

### TraceLens_generate_perf_report_pftrace_memory_copy

Memory-copy report grouped by `copy_bytes` with direction and the GPUs
involved.

| Argument | Default | Description |
|----------|---------|-------------|
| `--trace_path` | required | Path to `.json`, `.json.gz`, or `.pftrace`. |
| `--output_xlsx_path` | auto | Custom Excel output path. |
| `--output_csvs_dir` | None | Directory for CSV output. |
| `--traceconv` | auto | Path to `traceconv`. |

### TraceLens_split_inference_trace

Split an inference trace into per-iteration or per-phase sub-traces.

| Argument | Default | Description |
|----------|---------|-------------|
| `trace_path` | required | Path to the trace (`.json` or `.json.gz`). |
| `-o`, `--output-dir` | required | Output directory. |
| `-i`, `--iterations` | `all` | Iteration range: `all`, a single index (`50`), or a range (`10:20`). |
| `-d`, `--dummy` | `all` | Dummy-run range. |
| `--find-steady-state` / `--num-steps` | off / 32 | Extract a steady-state region of N iterations. |
| `--divide-phases` | off | Store steady-state steps into `prefilldecodemix/` and `decode_only/` sub-folders. |
| `--CONC`, `--OSL`, `--R` | None | Expected concurrency and output-sequence-length window parameters. |

## Python SDK

The SDK modules live under the `TraceLens` package and can be imported to build
custom workflows. Each module has a dedicated guide in the repository `docs_original/`
directory and an example notebook under `examples/`.

| Module | Purpose | Reference |
|--------|---------|-----------|
| `Trace2Tree` | Build and navigate the hierarchical event tree (Python ops → CPU dispatch → GPU kernels). | `docs_original/Trace2Tree.md`, `examples/trace2tree_example.ipynb` |
| `TreePerf` | GPU-timeline breakdown, per-op performance, and roofline metrics. | `docs_original/TreePerf.md`, `examples/tree_perf_example.ipynb` |
| `PerfModel` | Compute and roofline performance models for operators. | `docs_original/gemm_dim_eff.md`, `docs_original/triton_perf_model_walkthrough.md` |
| `NcclAnalyser` | Multi-rank collective latency/bandwidth/skew analysis. | `docs_original/NcclAnalyser.md`, `examples/nccl_analyser_example.ipynb` |
| `TraceDiff` | Morphological comparison of two trace trees. | `docs_original/TraceDiff.md`, `examples/trace_diff_example.ipynb` |
| `EventReplay` | Extract and replay isolated operations. | `docs_original/EventReplay.md`, `examples/event_replayer_example.ipynb` |
| `TraceFusion` | Merge multi-rank traces for Perfetto visualization. | `docs_original/TraceFusion.md`, `examples/trace_fusion_example.py` |
| `Reporting` | The report generators behind the CLI tools; importable to return pandas data frames. | `docs_original/generate_perf_report.md` |
| `TraceUtils` | Trace utilities, including inference-trace splitting. | — |

For report-column definitions across all sheets, see
`docs_original/perf_report_columns.md` in the repository.

```{note}
For a class- and function-level SDK reference generated directly from
docstrings, build the documentation against the TraceLens source with a Sphinx
autodoc/autosummary extension.
```

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [Compatibility matrix](../reference/compatibility.md)
- [Release notes](../about/release-notes.md)
