<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

---
myst:
    html_meta:
        "description": "Learn how to generate a TraceLens performance report from a JAX XPlane protobuf trace, including GPU-event and GEMM analysis."
        "keywords": "TraceLens, JAX, XPlane, protobuf, GPU trace, GEMM, performance report, roofline, ROCm, AMD Instinct, xprof"
---

# Generate a JAX performance report

TraceLens offers two JAX reports from an XPlane protobuf trace: the **standard
report** (operator and roofline analysis, like the PyTorch report) and a
**GPU-event and GEMM analysis** focused on event-type breakdowns and GEMM
performance.

## Prerequisites

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- A JAX XPlane protobuf trace (`xplane.pb`). JAX parsing uses the `xprof`
  dependency, installed automatically with TraceLens.

```{note}
JAX protobuf parsing has been validated with `tensorboard` 2.19.0,
`tensorboard-plugin-profile` 2.19.0, and `protobuf` 5.29.2. Other versions might
not work.
```

## Standard report

Generate a report from a JAX XPlane protobuf trace (`xplane.pb`):

```bash
TraceLens_generate_perf_report_jax --profile_path path/to/xplane.pb
```

The same `--profile_path` argument also accepts a PyTorch `trace.json`.

**Expected output:** an Excel report analogous to the
[PyTorch report](./generate-perf-report-pytorch.md), with the operator and
roofline analysis derived from the XPlane trace (the operator sheets are named
`kernel_launchers*`, alongside `xla_summary` and `df_xla_perf`). JAX output does
**not** include the `short_kernels_summary` / `short_kernel_histogram` sheets —
those are PyTorch-only.

Options:

- `--kernel_metadata_keyword_filters <kw> ...` restricts the analysis to events
  whose metadata contains the given keywords (for example, `remat checkpoint` to
  focus on rematerialization/checkpointing scopes).
- `--enable-origami` uses Origami-simulated GEMM/SDPA times when a GPU arch JSON
  is available.
- `--output_xlsx_path` and `--output_csvs_dir` control output paths. (JAX
  currently supports only these output options.)

For bandwidth analysis of JAX collective operations from XPlane traces, see the
[`jax_nccl_analyser_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/jax_nccl_analyser_example.ipynb)
notebook.

## GPU-event and GEMM analysis

For a JAX-specific breakdown of GPU event statistics and GEMM performance, use
the analysis report. It has no console entry point, so run it as a module:

```bash
python -m TraceLens.Reporting.generate_perf_report_jax_analysis \
    --profile_xplane_pb_path path/to/xplane.pb \
    --output_path ./jax_analysis \
    --num_cus 304
```

This writes one file per table into `--output_path`, named
`<output_filename><suffix>` (default `output_filename` is
`trace_analysis_results`), in both `.xlsx` and `.csv` by default:

| File suffix | Contents |
|-------------|----------|
| `_gpu_events_averages` | Average time and percentage per GPU event type, including total/exposed/overlapped communication time. |
| `_gpu_events_categorized_mean` | GPU event statistics grouped by category. |
| `_xla_grouped` | XLA computations grouped by base name (digits/trailing underscores stripped), sorted by percentage of total time. |
| `_gemms` | Summary of GEMM events. |
| `_gemms_detailed` | Detailed GEMM performance metrics, computed against the GPU compute units (`--num_cus`). |

Options:

- `--profile_xplane_pb_path PATH` (required) — the JAX XPlane protobuf trace.
- `--output_path DIR` (required) — output directory.
- `--num_cus N` — GPU compute units for the GEMM model (default `304` for MI300X;
  use `104` for MI210). `--name` sets the architecture label (default `mi300x`).
- `--output_table_formats {.xlsx,.csv} ...` — one or both output formats (default
  both).
- `--output_filename NAME` — base name for the output files (default
  `trace_analysis_results`).

## Next steps

- Quantify the effect of a change by [comparing two traces](./compare-traces.md).
- Analyze multi-GPU collectives with a
  [collective-communication report](./collective-report.md).
- Analyze [PyTorch](./generate-perf-report-pytorch.md) or
  [rocprof](./generate-perf-report-rocprof.md) traces.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [API reference](../reference/api-reference.md)
