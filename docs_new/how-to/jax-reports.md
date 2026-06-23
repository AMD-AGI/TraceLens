<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Analyze JAX traces

This guide shows how to generate a TraceLens report from a JAX XPlane protobuf
trace.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A JAX XPlane protobuf trace (`xplane.pb`). JAX parsing uses the `xprof`
  dependency, which is installed automatically with TraceLens.

## Step 1: Generate the report

```bash
TraceLens_generate_perf_report_jax --profile_path path/to/xplane.pb
```

The tool also accepts a PyTorch `trace.json` via the same `--profile_path`
argument.

**Expected output:** an Excel report analogous to the PyTorch report, with
operator and roofline analysis derived from the XPlane trace.

## Step 2: Filter by kernel metadata (optional)

Restrict the analysis to events whose metadata contains specific keywords — for
example, to focus on rematerialization/checkpointing scopes:

```bash
TraceLens_generate_perf_report_jax \
    --profile_path path/to/xplane.pb \
    --kernel_metadata_keyword_filters remat checkpoint
```

## Step 3: Add simulated GEMM/SDPA timings (optional)

Pass `--enable-origami` to use Origami simulated GEMM/SDPA times when a GPU
architecture JSON is available:

```bash
TraceLens_generate_perf_report_jax \
    --profile_path path/to/xplane.pb \
    --enable-origami
```

## JAX collective analysis

For bandwidth analysis of JAX collective operations from XPlane traces, see the
`examples/jax_nccl_analyser_example.ipynb` notebook and `docs/jax_analyses.md`
in the repository.

## Next steps

- Customize output paths with `--output_xlsx_path` and `--output_csvs_dir`.
- Compare JAX and PyTorch reports with
  [TraceLens_compare_perf_reports_pytorch](./compare-traces.md).
