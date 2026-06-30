---
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
myst:
    html_meta:
        "description": "Learn how to generate a multi-sheet Excel performance report from a PyTorch torch.profiler trace using TraceLens, including roofline analysis."
        "keywords": "TraceLens, PyTorch profiler, torch.profiler, GPU trace, performance report, roofline, GEMM, ROCm, AMD Instinct, activation recompute, CUDA migration"
---

# Generate a PyTorch performance report

Turn a `torch.profiler` Chrome trace into a multi-sheet Excel (or CSV)
performance report, then read the sheets to find what dominates GPU time.

## Prerequisites

Before generating a report, confirm you have the following:

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- A `torch.profiler` Chrome trace (`.json` or `.json.gz`).

## Generate the report

Pass the trace path to generate the default Excel report:

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/trace.json
```

Set a custom Excel path, or write per-sheet CSVs instead:

```bash
# Custom Excel path
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --output_xlsx_path report.xlsx

# Per-sheet CSVs instead of Excel
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --output_csvs_dir ./report_csvs
```

**Output behavior:** by default a single Excel workbook is written next to the
trace, with the name inferred from the trace (`profile.json` →
`profile_perf_report.xlsx`). `--output_xlsx_path` changes that location.
`--output_csvs_dir` writes one CSV per sheet; passing it alone replaces the Excel
output, while passing it together with `--output_xlsx_path` produces both. The
`openpyxl` package is only needed for Excel output and is auto-installed if
missing.

## The report sheets

The generated workbook contains the following sheets:

| Sheet | Description |
|-------|-------------|
| `gpu_timeline` | End-to-end GPU activity: computation, communication, memory copy, and idle time. |
| `ops_summary_by_category` | Compute time grouped by operation category (GEMM, SDPA_fwd, elementwise, …) — the most aggregated view. |
| `ops_summary` | Per-operation aggregate; one row per unique operation name. |
| `ops_unique_args` | Most detailed view; one row per unique (operation name, argument) combination. |
| `unified_perf_summary` | Unified perf metrics for ops with perf models or leaf ops that launch kernels — `GFLOPS`, `TFLOPS/s`, `Data Moved (MB)`, `FLOPS/Byte`, `TB/s`, aggregated by unique args. |
| `coll_analysis` | Collective-communication analysis (enabled by default; disable with `--disable_coll_analysis`). |
| Roofline sheets | One per operation category (`GEMM`, `CONV_fwd`, `SDPA_fwd`, …) with the intensity/roofline metrics described below. |
| `kernel_summary` | Per-kernel summary — added with `--enable_kernel_summary`. |
| `short_kernels_summary`, `short_kernel_histogram` | Short-kernel table and duration histogram — added with `--short_kernel_study`. |

For the GPU timeline, a low computation percentage with significant idle time
indicates poor compute/communication overlap; use `--micro_idle_thresh_us` to
split very short idle gaps into their own category.

See [Performance report column definitions](https://github.com/AMD-AGI/TraceLens/blob/main/docs_original/perf_report_columns.md)
for what each column means.

## Roofline classification

Every per-category roofline sheet includes operation-intensity columns by
default: `GFLOPS`, `Data Moved (MB)`, `FLOPS/Byte`, `TFLOPS/s`, and `TB/s`.

To add the roofline **bound classification**, supply a GPU architecture spec.
This adds:

- **`Compute Spec`** — combined compute type and precision (e.g. `matrix_bf16`,
  `vector_fp32`).
- **`Roofline Time (µs)`** — theoretical minimum time from the GPU's peak
  capabilities.
- **`Roofline Bound`** — `COMPUTE_BOUND` or `MEMORY_BOUND`.
- **`Pct Roofline`** — how close the measured kernel time runs to the roofline.

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --gpu_arch_platform MI300X
```

- `--gpu_arch_platform` takes a bundled platform name (`MI300X`, `MI325X`, under
  `TraceLens/Agent/Analysis/utils/arch/`); use `--gpu_arch_json_path` to supply
  your own spec. The two flags are mutually exclusive.
- An operation is classified by comparing its compute time (`FLOPs / peak FLOPS`)
  against its memory time (`bytes / peak bandwidth`): the larger term wins.
  Equivalently, operations whose arithmetic intensity (FLOPs/byte) sits below the
  roofline knee point (peak FLOPS / peak bandwidth) are memory-bound; those above
  it are compute-bound.
- Add `--enable-origami` to use Origami-simulated GEMM/SDPA times when a GPU arch
  spec is provided.

The arch JSON specifies Max Achievable FLOPS (MAF) per compute type and
precision; see the
[GPU architecture example](https://github.com/AMD-AGI/TraceLens/blob/main/examples/gpu_arch_example.md)
for the format and the
[AMD MAF measurements](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html#amd-maf-results)
for reference values. To plot roofline charts for specific operators through the
SDK, see the
[`roofline_plots_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/roofline_plots_example.ipynb)
notebook.

## Detect activation recompute

When training with activation checkpointing (`torch.utils.checkpoint`), some
forward-pass ops are recomputed during the backward pass to save memory.
`--detect_recompute` identifies these and adds an `is_recompute` column so you
can see how much GPU time and compute is spent on recomputation:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --detect_recompute
```

TraceLens walks the CPU call-stack tree and marks all ops under
`recompute_fn` subtrees (`python_function` events from `torch/utils/checkpoint.py`)
as `is_recompute=True`. This requires `python_function` events in the trace,
which the flag enables automatically. The `is_recompute` column is added to the
`gpu_timeline`, `ops_summary_by_category`, `ops_summary`, `ops_unique_args`, and
`unified_perf_summary` sheets, splitting rows into recompute vs non-recompute.
Use it to answer questions like what percentage of GPU time is
recomputation, which layers are recomputed and at what cost, and whether the
overhead is acceptable for the memory saved. When the flag is not set there is
zero overhead — no extra columns and no `python_function` parsing.

The same split is available through the SDK:

```python
from TraceLens.TreePerf import TreePerfAnalyzer

analyzer = TreePerfAnalyzer.from_file("trace.json", detect_recompute=True)
df = analyzer.get_df_kernel_launchers(include_kernel_details=True)
print(df["is_recompute"].value_counts())
```

## Extend the report (custom hooks)

`--extension_file` injects custom logic into the report pipeline — useful for
pseudo-op injection, custom perf models, or new op categories. The Python file
can define any of:

| Symbol | Type | Purpose |
|--------|------|---------|
| `tree_postprocess_extension` | `Callable` | Called with `perf_analyzer.tree`; update the tree post-construction. |
| `perf_model_extension` | `dict` | Map op name → custom perf-model class; overrides or extends built-in models. |
| `op_category_extension` | `dict` | Map category-only op names to final categories, so an op appears in unified reports without a perf model. |

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --extension_file my_extension.py
```

See the example extension file for MegatronLM in the
[`examples/`](https://github.com/AMD-AGI/TraceLens/tree/main/examples) directory.

## Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_xlsx_path PATH` | auto-inferred | Excel output path (see output behavior above). |
| `--output_csvs_dir DIR` | `None` | Write each sheet as a CSV in this directory. |
| `--gpu_arch_platform NAME` | `None` | Bundled GPU arch for roofline classification (`MI300X`, `MI325X`). |
| `--gpu_arch_json_path PATH` | `None` | Custom GPU arch JSON (mutually exclusive with `--gpu_arch_platform`). |
| `--enable-origami` | `False` | Use Origami-simulated GEMM/SDPA times when an arch is provided. |
| `--detect_recompute` | `False` | Add an `is_recompute` column for activation checkpointing (see above). |
| `--extension_file PATH` | `None` | Custom tree / perf-model / op-category hooks (see above). |
| `--enable_kernel_summary` | `False` | Add the `kernel_summary` sheet. |
| `--short_kernel_study` | `False` | Add the short-kernel study sheets. |
| `--short_kernel_threshold_us X` | `10` | Threshold (µs) to classify a kernel as "short". |
| `--short_kernel_histogram_bins B` | `100` | Number of bins for the short-kernel histogram. |
| `--enable_pseudo_ops` | `False` | Augment the tree with pseudo-ops to isolate kernels (e.g. `FusedMoE`). |
| `--include_overlap_info` | `False` | Add kernel-overlap sheets. |
| `--include_unlinked_kernels` | `False` | Include kernels not linked to a host call stack in the GPU timeline. |
| `--micro_idle_thresh_us X` | `None` | Split idle gaps shorter than this into a separate micro-idle category. |
| `--disable_coll_analysis` | (on) | Disable the `coll_analysis` sheet (collective analysis is on by default). |
| `--topk_ops N` | `None` | Cap rows in the unique-args (`ops_unique_args`) table. |
| `--topk_short_kernels N` | `None` | Cap rows in the short-kernel table. |
| `--topk_roofline_ops N` | `None` | Cap rows in the roofline sheets. |

## Next steps

- Quantify the effect of a change by [comparing two traces](./compare-traces.md).
- Analyze multi-GPU collectives with a
  [collective-communication report](./collective-report.md).
- Isolate a single operation into a reproducer with
  [EventReplay](./event-replay.md).
- Analyze [JAX](./generate-perf-report-jax.md) or
  [rocprof](./generate-perf-report-rocprof.md) traces.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [API reference](../reference/api-reference.md)
