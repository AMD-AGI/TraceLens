<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Generate a PyTorch performance report
```{meta}
:description: Learn how to generate a multi-sheet Excel performance report from a PyTorch torch.profiler trace using TraceLens, including roofline analysis.
:keywords: TraceLens, PyTorch profiler, torch.profiler, GPU trace, performance report, roofline, GEMM, ROCm, AMD Instinct, activation recompute, CUDA migration
```

Turn a `torch.profiler` Chrome trace into a multi-sheet Excel (or CSV)
performance report, then read the sheets to find what dominates GPU time.

## Before you begin

Before generating a report, confirm you have the following:

- [TraceLens installed](../install/install.md).
- A `torch.profiler` Chrome trace (`.json` or `.json.gz`).

If you don't have a trace yet, capture one as shown below. The
[PyTorch profiling walkthrough](https://github.com/AMD-AGI/TraceLens/blob/main/notebooks/torch-profiling.ipynb)
walks through it end to end.

## Collect a trace

The quality of the TraceLens Analysis report depends on
the quality of the trace. 

```python
import torch

def export(prof):
    prof.export_chrome_trace("trace.json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
    on_trace_ready=export,
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(num_steps):   # num_steps >= 4 to cover wait + warmup + active
        model(inputs)
        torch.cuda.synchronize()
        prof.step()
```

Two flags on `torch.profiler.profile` change what TraceLens can report.

| Flag | What it captures | Why TraceLens needs it |
|---|---|---|
| `record_shapes=True` | Input argument shapes and dtypes for each operator | Roofline and compute modeling need shapes to compute FLOPs, bytes, and arithmetic intensity. Without them, per-operator efficiency metrics are unavailable. |
| `with_stack=True` | The Python call stack above each operator | Required for the call-stack views — GPU time grouped by `nn.Module` or by the line of Python that launched the work. Reports run with `--include_call_stack` need it. |

```{note}
Inference frameworks that run in HIP graph mode (vLLM, SGLang, ATOM, xDiT)
need a different capture path. See
[Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md).
```

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

**Output behavior**: By default a single Excel workbook is written next to the
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
| `ops_summary_by_category` | Compute time grouped by operation category (GEMM, SDPA_fwd, elementwise, and so on) — the most aggregated view. |
| `ops_summary` | Per-operation aggregate; one row per unique operation name. |
| `ops_unique_args` | Most detailed view; one row per unique (operation name, argument) combination. |
| `unified_perf_summary` | Unified perf metrics for ops with perf models or leaf ops that launch kernels — `GFLOPS`, `TFLOPS/s`, `Data Moved (MB)`, `FLOPS/Byte`, `TB/s`, aggregated by unique args. |
| `coll_analysis` | Collective-communication analysis (enabled by default; disable with `--disable_coll_analysis`). |
| Roofline sheets | One per operation category (`GEMM`, `CONV_fwd`, `SDPA_fwd`, and so on) with the intensity/roofline metrics described below. |
| `kernel_summary` | Per-kernel summary — added with `--enable_kernel_summary`. |
| `short_kernels_summary`, `short_kernel_histogram` | Short-kernel table and duration histogram — added with `--short_kernel_study`. |

For the GPU timeline, a low computation percentage with significant idle time
indicates poor compute and communication overlap; use `--micro_idle_thresh_us` to
split very short idle gaps into their own category.

See [Performance report column reference](../reference/perf-report-columns.md)
for what each column means.

## Roofline classification

Every per-category roofline sheet includes operation-intensity columns by
default: `GFLOPS`, `Data Moved (MB)`, `FLOPS/Byte`, `TFLOPS/s`, and `TB/s`.

To add the roofline *bound classification*, supply a GPU architecture spec.
This adds:

- `Compute Spec:` combined compute type and precision (for example, `matrix_bf16`,
  `vector_fp32`).
- `Roofline Time (µs):` theoretical minimum time from the GPU's peak
  capabilities.
- `Roofline TFLOPS/s:` throughput from dividing modeled FLOPs by that time.
- `Roofline TB/s:` bandwidth from dividing modeled bytes by that time.
- `Roofline Bound:` `COMPUTE_BOUND` or `MEMORY_BOUND`.
- `Pct Roofline:` how close the measured kernel time runs to the roofline.

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
- Add `--enable-origami-gemm` for Origami GEMM times when a GPU arch spec is
  provided.
- Add `--enable-origami-sdpa-tile` for attention times from TraceLens's SDPA
  tile model, with Origami timing each tile GEMM (see below).
- To add your own model, see [Add an op model](#add-an-op-model).

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
| `op_models` | `dict` | Map a label → op model `fn(category, params, arch)`; see [Add an op model](#add-an-op-model). |
| `external_op_model` | `Callable` | Same signature; registered under the label `External`. |
| `kernel_filters` | `dict` | Map a label → `fn(kernel_event)` returning whether to count the kernel. Each label fills `<label> Kernel Time (µs)` and `<label> TFLOPS/s` with the busy time of the op's kept kernels. |

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path path/to/trace.json \
    --extension_file my_extension.py
```

### Add an op model

A *perf model* says what work an op does: its parameters, FLOPs, and bytes.
An *op model* takes that description and returns anything you want reported
per op: usually a predicted time, but also values such as the kernel
configuration a library would pick, an occupancy estimate, or a flag.
TraceLens reports each op model under a label, next to the measured metrics
and the roofline.

The contract:

```python
def my_model(category, params, arch):
    ...
    return None            # this op isn't handled
    return 12.5            # predicted time in µs
    return {"time_us": 12.5, "Tile": "256x128x64"}  # time plus extra columns
```

- `category` is the perf-model category (`"GEMM"`, `"SDPA_fwd"`, ...). For a
  backward op it is the backward category (`"SDPA_bwd"`, `"CONV_bwd"`, ...)
  with the forward op's params; ops without one, such as GEMM backward, are
  skipped.
- `params` is a copy of the perf model's parameters, the same values as the
  report's `param:` columns; look at those columns to see what a category
  provides. Any key can be missing: `transpose`, for example, is only set
  when the GEMM's kernel name parses. `dtype_A_B` holds the trace's dtype
  strings, such as `"c10::BFloat16"`;
  `TraceLens.PerfModel.utils.torch_dtype_map` turns them into `"bf16"`.
- `arch` is the report's GPU arch dict, or `None` without one. A model is
  free to ignore it and target another GPU.
- A dict's `time_us` is optional. Every other key becomes a `<label> <key>`
  column. Only scalars (numbers, strings, bools) are written.
- The model is called once per op launch, so the same shape can arrive many
  times. Cache results in the model if a call is slow.
- If the model raises, TraceLens leaves its columns empty for that op, keeps
  the op's other metrics, and prints one warning per model, category, and
  error type. Return `None` to skip ops on purpose.

Register models in the extension file:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def _gemm_time(M, N, K, B, transpose):
    return my_library.gemm(M, N, K, B, transpose)

def gemm_model(category, params, arch):
    if category != "GEMM":
        return None
    result = _gemm_time(
        params["M"], params["N"], params["K"], params.get("B", 1),
        str(params.get("transpose")),
    )
    return {"time_us": result.time_us, "Tile": result.tile}

op_models = {"MyModel": gemm_model}
```

This adds `MyModel Time (µs)`, `MyModel TFLOPS/s`, `MyModel TB/s`,
`MyModel Tile`, and `Pct MyModel` (the time as a percentage of the measured
kernel time); the summary sheets pick up every label (see the
[column reference](../reference/perf-report-columns.md#op-model-columns)).
Keep the import and call of a proprietary library in the extension file. A
placeholder is in `examples/external_op_model_stub.py`.

From Python, register a model with
`TreePerfAnalyzer.register_op_model(label, fn)`, then
`compute_perf_metrics(event)` runs every registered model on one op, which is
handy when iterating on a model:

```python
from TraceLens import TreePerfAnalyzer

analyzer = TreePerfAnalyzer.from_file("trace.json", arch=arch)
analyzer.register_op_model("MyModel", gemm_model)
gemm = next(e for e in analyzer.tree.events if e["name"] == "aten::mm")
metrics = analyzer.compute_perf_metrics(gemm)  # param: ..., MyModel Time (µs), ...
```

Behind this, TraceLens builds the op's perf model from the event, describes
its work with `TraceLens.PerfModel.op_models.op_work(perf_model)`, and calls
each op model on that work. Roofline and the Origami models use the same
interface, in `TraceLens/PerfModel/op_models.py`.

Origami models GEMMs only. For attention, `--enable-origami-sdpa-tile` adds
TraceLens's SDPA tile model: it times one Q·Kᵀ tile and one P·V tile on one
CU with Origami, scales them by the number of waves, and adds softmax and
memory terms. Its columns are labeled `SDPA Tile Origami`. It needs
`num_cus`, `gemm_units_per_cu`, and `mem_bw_gbps` in the arch, and
`l1_bw_gbps` for backward. The tile model, in
`TraceLens/PerfModel/sdpa_tile.py`, times its tiles with whatever GEMM model
it's given as `gemm_time`; the op model passes Origami's. To use another GEMM
model, write an op model that calls
`sdpa_tile_time_us(work.perf_model, arch, your_gemm_time, bwd=work.bwd)`.

### Add a kernel filter

A kernel filter reports the op's throughput over a subset of its kernels, for
example without copy and transpose kernels. `examples/kernel_filter_example.py`
defines one:

```python
def non_data_mov_filter(kernel):
    return not any(p in kernel["name"] for p in ("direct_copy_kernel", "transpose_"))

kernel_filters = {"Non-Data-Mov": non_data_mov_filter}
```

From Python, use `TreePerfAnalyzer.register_kernel_filter(label, fn)`.

See the example extension file for MegatronLM in the
[`examples/`](https://github.com/AMD-AGI/TraceLens/tree/main/examples) directory.

## Optional arguments

The following table describes all optional arguments.

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_xlsx_path PATH` | auto-inferred | Excel output path (see output behavior above). |
| `--output_csvs_dir DIR` | `None` | Write each sheet as a CSV in this directory. |
| `--gpu_arch_platform NAME` | `None` | Bundled GPU arch for roofline classification (`MI300X`, `MI325X`). |
| `--gpu_arch_json_path PATH` | `None` | Custom GPU arch JSON (mutually exclusive with `--gpu_arch_platform`). |
| `--enable-origami-gemm` | `False` | Add `Origami` GEMM times when an arch is provided. |
| `--enable-origami-sdpa-tile` | `False` | Add `SDPA Tile Origami` attention times from TraceLens's tile model, with Origami timing each tile GEMM. |
| `--detect_recompute` | `False` | Add an `is_recompute` column for activation checkpointing (see above). |
| `--extension_file PATH` | `None` | Custom tree, perf-model, op-category, op-model, and kernel-filter hooks (see above). |
| `--enable_kernel_summary` | `False` | Add the `kernel_summary` sheet. |
| `--short_kernel_study` | `False` | Add the short-kernel study sheets. |
| `--short_kernel_threshold_us X` | `10` | Threshold (µs) to classify a kernel as "short". |
| `--short_kernel_histogram_bins B` | `100` | Number of bins for the short-kernel histogram. |
| `--enable_pseudo_ops` | `False` | Augment the tree with pseudo-ops to isolate kernels (for example, `FusedMoE`). |
| `--include_overlap_info` | `False` | Add kernel-overlap sheets. |
| `--include_unlinked_kernels` | `False` | Include kernels not linked to a host call stack in the GPU timeline. |
| `--micro_idle_thresh_us X` | `None` | Split idle gaps shorter than this into a separate micro-idle category. |
| `--disable_coll_analysis` | (on) | Disable the `coll_analysis` sheet (collective analysis is on by default). |
| `--topk_ops N` | `None` | Cap rows in the unique-args (`ops_unique_args`) table. |
| `--topk_short_kernels N` | `None` | Cap rows in the short-kernel table. |
| `--topk_roofline_ops N` | `None` | Cap rows in the roofline sheets. |

## Related topics

- Quantify the effect of a change by [comparing two traces](./compare-traces.md).
- Analyze multi-GPU collectives with a
  [collective-communication report](./collective-report.md).
- Isolate a single operation into a reproducer with
  [EventReplay](./event-replay.md).
- Catalog many reports for search with
  [Index a corpus of traces](./trace-index.md).
- Analyze [JAX](./generate-perf-report-jax.md) or
  [rocprof](./generate-perf-report-rocprof.md) traces.

