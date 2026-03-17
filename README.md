<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens

TraceLens is a Python library focused on **automating analysis from trace files** and enabling rich performance insights. Designed with **simplicity and extensibility** in mind, this library provides tools to simplify the process of profiling and debugging complex distributed training and inference systems.

## Key Features

**Hierarchical Performance Breakdowns** — Pinpoint bottlenecks with a top-down view, moving from the overall GPU timeline (idle/busy) to operator categories, individual operators, and right down to unique argument shapes.

**Compute & Roofline Modeling** — Automatically translate raw timings into efficiency metrics like TFLOP/s and TB/s for popular operations. Determine if an op is compute- or memory-bound and see how effectively your code uses the hardware.

**Multi-GPU Communication Analysis** — Accurately diagnose scaling issues by dissecting collective operations. TraceLens separates pure communication time from synchronization skew and calculates effective bandwidth on your real workload.

**Trace Comparison** — Quantify the impact of your changes with powerful trace diffing. By analyzing performance at the CPU dispatch level, TraceLens enables meaningful side-by-side comparisons across different hardware and software versions.

**Event Replay** — Isolate any operation for focused debugging. TraceLens generates minimal, self-contained replay scripts directly from trace metadata, making it simple to share IP-safe test cases with kernel developers.

**Extensible SDK** — Get started instantly with ready-to-use scripts, then build your own custom workflows using a flexible and hackable Python API.

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

### 2. Generate a report from your PyTorch trace

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```

This produces an Excel workbook with GPU timeline breakdown, ops summary, roofline metrics, and more.
See [Performance Report Column Definitions](docs/perf_report_columns.md) for what each column means.

**Don't have a trace yet?** Follow the [PyTorch profiling guide](docs/conceptual/torch_profiling_guide.ipynb), or use the [demo traces](tests/traces) bundled in the repo.

For the full CLI reference (JAX, rocprofv3, trace comparison, multi-rank collectives), see [Supported Profile Formats](#supported-profile-formats) below.

## Examples & Notebooks

Hands-on notebooks that walk through the core TraceLens features:

| Example | What it covers |
|---------|----------------|
| [Trace2Tree](examples/trace2tree_example.ipynb) | Navigate the hierarchical event tree — link Python ops, CPU dispatches, and GPU kernels |
| [TreePerf](examples/tree_perf_example.ipynb) | GPU timeline breakdown, per-op performance, and roofline metrics via the SDK |
| [NN Module View](examples/nn_module_view.ipynb) | See GPU time broken down by `nn.Module` — useful for model developers |
| [NCCL Analyser](examples/nccl_analyser_example.ipynb) | Multi-rank collective analysis: latency, bandwidth, skew |
| [Trace Diff](examples/trace_diff_example.ipynb) | Morphological comparison of two trace trees to pinpoint structural divergences |
| [Event Replay](examples/event_replayer_example.ipynb) | Extract and replay operations for isolated debugging |
| [Trace Fusion](examples/trace_fusion_example.py) | Merge multi-rank PyTorch traces into a single file for Perfetto visualization |
| [Roofline Plots](examples/roofline_plots_example.ipynb) | Build roofline-style visualizations for specific operators |
| [JAX NCCL Analyser](examples/jax_nccl_analyser_example.ipynb) | Bandwidth analysis for JAX collective operations from XPlane traces |

For community-contributed utilities — including interactive trace dashboards (**traceMap**), roofline analysis tooling, and a Streamlit UI — see [`examples/custom_workflows/`](examples/custom_workflows/).

## Supported Profile Formats

| Format | Tool | Documentation |
|--------|------|---------------|
| **PyTorch** | `torch.profiler` | [docs/generate_perf_report.md](docs/generate_perf_report.md) |
| **JAX** | XPlane protobuf | [docs/jax_analyses.md](docs/jax_analyses.md) |
| **rocprofv3 JSON** | AMD ROCm rocprofiler-sdk | [docs/generate_perf_report_rocprof.md](docs/generate_perf_report_rocprof.md) |
| **rocprofv3 pftrace** | Perfetto-style | [docs/generate_perf_report_rocprof_pftrace.md](docs/generate_perf_report_rocprof_pftrace.md) |

### PyTorch

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```

Detailed docs: [generate_perf_report.md](docs/generate_perf_report.md). Supports compressed traces (`.zip`, `.gz`).

### Compare PyTorch reports

```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

Detailed docs: [compare_perf_reports_pytorch.md](docs/compare_perf_reports_pytorch.md).

### Multi-rank collective report

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8
```

Detailed docs: [generate_multi_rank_collective_report_pytorch.md](docs/generate_multi_rank_collective_report_pytorch.md).

### rocprofv3 JSON

For `*_results.json` from rocprofv3:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace_results.json \
    --short_kernel_study --kernel_details
```

Detailed docs: [generate_perf_report_rocprof.md](docs/generate_perf_report_rocprof.md).

### pftrace (rocprofv3 / Perfetto)

For Perfetto-style traces (e.g. `rocprofv3 --output-format pftrace`):

```bash
# Record a pftrace
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace \
    --output-format pftrace -d ./v3_traces -- python3 your_app.py

# Activity report (NSYS-style category summary, optional Markdown)
TraceLens_generate_perf_report_pftrace_hip_activity --trace_path sample.pftrace --write_md

# API↔Kernel report (latency breakdown T = A + Q + K)
TraceLens_generate_perf_report_pftrace_hip_api --trace_path sample.pftrace

# Memory copy report
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace
```

`.pftrace` is Perfetto's binary format that needs to be converted to JSON for parsing. `traceconv` (a Perfetto tool) is optional — if not on `PATH`, it is downloaded automatically. You can also pass `--traceconv /path/to/traceconv` explicitly.

## Documentation

Deeper dives on the core modules:

| Module | Doc |
|--------|-----|
| Trace2Tree | [docs/Trace2Tree.md](docs/Trace2Tree.md) |
| TreePerf | [docs/TreePerf.md](docs/TreePerf.md) |
| NCCL Analyser | [docs/NcclAnalyser.md](docs/NcclAnalyser.md) |
| TraceDiff | [docs/TraceDiff.md](docs/TraceDiff.md) |
| Event Replay | [docs/EventReplay.md](docs/EventReplay.md) |
| TraceFusion | [docs/TraceFusion.md](docs/TraceFusion.md) |
| GPU Event Analyser | [docs/gpu_event_analyser.md](docs/gpu_event_analyser.md) |
| JAX Analyses | [docs/jax_analyses.md](docs/jax_analyses.md) |
| pftrace Reports | [docs/generate_perf_report_rocprof_pftrace.md](docs/generate_perf_report_rocprof_pftrace.md) |
| Performance Report Columns | [docs/perf_report_columns.md](docs/perf_report_columns.md) |

## Development

```bash
git clone https://github.com/AMD-AGI/TraceLens.git && cd TraceLens
pip install -e .[dev]
python -m pytest tests/ -v
```

## Contributing

We welcome contributions across the entire project — new analysis modules, performance models, docs, examples, or bug fixes. Whether you're adding a new metric or building a custom workflow, the SDK is designed to make that easy.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branching, commit style, and project structure.

## Additional Resources

- [PyTorch Conference 2025 Poster](docs/TraceLens%20-%20Democratizing%20AI%20Performance%20Analysis%20-%20Adeem%20Jassani%2C%20AMD.pdf)
- [GEMMs in AI Models — Conceptual Tutorial](docs/conceptual/aimodels_gemms.md)
- [Trace2Tree Motivation](docs/conceptual/trace2tree_motivation.md)
- [PyTorch Profiling Guide](docs/conceptual/torch_profiling_guide.ipynb)
