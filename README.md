<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens

[![Tests](https://github.com/AMD-AGI/TraceLens/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/AMD-AGI/TraceLens/actions/workflows/regression-tests.yml)
[![Lint](https://github.com/AMD-AGI/TraceLens/actions/workflows/lint.yml/badge.svg)](https://github.com/AMD-AGI/TraceLens/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%2B-blue)](setup.py)

TraceLens is a Python library for **automated performance analysis of training and inference workloads** from trace files. It reads profiler traces (PyTorch, JAX, rocprofv3) and shows you where GPU time actually goes: which kernels are slow, whether they are compute or memory bound, and where communication or idle gaps are costing you, so you can find and fix bottlenecks without hand-reading traces.

---

## Key Features

**Hierarchical Performance Breakdowns**: Pinpoint bottlenecks with a top-down view, moving from the overall GPU timeline (idle/busy) to operator categories, individual operators, and right down to unique argument shapes.

**Compute & Roofline Modeling**: Automatically translate raw timings into efficiency metrics like TFLOP/s and TB/s for popular operations. Determine if an op is compute or memory bound and see how effectively your workload utilizes the hardware.

**Multi-GPU Communication Analysis**: Accurately diagnose scaling issues by dissecting collective operations. TraceLens separates pure communication time from synchronization skew and calculates effective bandwidth on your workload.

**Trace Comparison**: Quantify the impact of your changes with powerful trace diffing. By analyzing performance at the CPU dispatch level, TraceLens enables meaningful side-by-side comparisons across different hardware and software versions.

**Event Replay**: Isolate any operation for focused debugging. TraceLens generates minimal, self-contained replay scripts directly from trace metadata, making it simple to share IP-safe test cases with kernel developers.

**Extensible SDK**: Get started instantly with ready-to-use scripts, then build your own custom workflows using a flexible and hackable Python API.

**TraceLens Agent**: Receive a prioritized human-readable optimization report, derived through an agentic workflow, covering compute kernels, system bottlenecks, and kernel fusion opportunities with root-cause reasoning and concrete resolutions.

---

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

### 2. Collect Traces

TraceLens analyses profiler traces from PyTorch, JAX, and AMD rocprofv3; see [Supported Profile Formats](#supported-profile-formats) for the full list. The instructions below cover collecting a PyTorch trace:
- **Generic Eager Traces**: Instrument your loop with `torch.profiler.profile(...)`, enabling CPU-side call-stack and shape capture (`with_stack=True`, `record_shapes=True`). Profile a representative steady-state window (a handful of steps, post-warmup) and log the trace with `prof.export_chrome_trace(...)`. A single rank's trace is enough for per-rank analysis. The [PyTorch profiling guide](docs_original/conceptual/torch_profiling_guide.ipynb) walks through this end to end.
- **Inference Traces with Graph Capture**: Collection has framework-specific requirements. Follow guidelines in [Inference Analysis](docs_original/Inference_analysis.md). The [Profiling skill](TraceLens/Agent/Profiling/README.md) automates vLLM/SGLang benchmarking and PyTorch profiler trace collection via [Magpie](https://github.com/AMD-AGI/Magpie), producing analysis-ready traces.

To try out TraceLens without collecting your own trace, use the [demo traces](tests/traces) bundled in the repository.

### 3. Analyze your Workload

### Building with TraceLens

Generate a performance analysis report from an eager execution PyTorch trace with a single command:

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```

This produces an Excel workbook with GPU timeline breakdown, ops summary, roofline metrics and more. For additional details, see [Performance Report Generation](docs_original/generate_perf_report.md) and [Performance Report Column Definitions](docs_original/perf_report_columns.md). For other input formats, see [Supported Profile Formats](#supported-profile-formats).

Compare two reports to quantify the impact of a change (see [compare_perf_reports_pytorch.md](docs_original/compare_perf_reports_pytorch.md)):

```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

For multi-rank runs, generate a collective-communication report across ranks (see [generate_multi_rank_collective_report_pytorch.md](docs_original/generate_multi_rank_collective_report_pytorch.md)):

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8
```

To dig deeper, call TraceLens modules directly and build your own analysis. These hands-on notebooks walk through the core features:

| Example                                                       | What it covers                                                                            |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [Trace2Tree](examples/trace2tree_example.ipynb)               | Navigate the hierarchical event tree, linking Python ops, CPU dispatches, and GPU kernels |
| [TreePerf](examples/tree_perf_example.ipynb)                  | GPU timeline breakdown, per-op performance, and roofline metrics via the SDK              |
| [NN Module View](examples/nn_module_view.ipynb)               | See GPU time broken down by `nn.Module`, useful for model developers                      |
| [NCCL Analyser](examples/nccl_analyser_example.ipynb)         | Multi-rank collective analysis: latency, bandwidth, skew                                  |
| [Trace Diff](examples/trace_diff_example.ipynb)               | Morphological comparison of two trace trees to pinpoint structural divergences            |
| [Event Replay](examples/event_replayer_example.ipynb)         | Extract and replay operations for isolated debugging                                      |
| [Trace Fusion](examples/trace_fusion_example.py)              | Merge multi-rank PyTorch traces into a single file for Perfetto visualization             |
| [Roofline Plots](examples/roofline_plots_example.ipynb)       | Build roofline-style visualizations for specific operators                                |
| [JAX NCCL Analyser](examples/jax_nccl_analyser_example.ipynb) | Bandwidth analysis for JAX collective operations from XPlane traces                       |

For community-contributed utilities, including interactive trace dashboards (traceMap), roofline analysis tooling, and a Streamlit UI, see [examples/custom_workflows/](examples/custom_workflows/).

### TraceLens Agent

Analyze a workload autonomously using an agentic system that automates performance analysis and bottleneck prioritization for PyTorch traces. The agent orchestrates the entire analysis workflow to pinpoint underperforming kernels, highlight kernel fusion opportunities, and flag system-level bottlenecks, each backed by root-cause reasoning and a concrete resolution. The result is a prioritized, human-readable markdown report that turns a raw trace into a ranked action list. This report can also be plugged directly into automated performance optimization platforms to drive kernel tuning, system configuration, kernel fusion, and model-code changes. Refer to [TraceLens Agent](TraceLens/Agent/Analysis/README.md) for more details.

---

## Supported Profile Formats

| Format                | Tool                     | Documentation                                                                                                  |
| --------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **PyTorch**           | `torch.profiler`         | [docs_original/generate_perf_report.md](docs_original/generate_perf_report.md)                                 |
| **JAX**               | XPlane protobuf          | [docs_original/jax_analyses.md](docs_original/jax_analyses.md)                                                 |
| **rocprofv3 JSON**    | AMD ROCm rocprofiler-sdk | [docs_original/generate_perf_report_rocprof.md](docs_original/generate_perf_report_rocprof.md)                 |
| **rocprofv3 pftrace** | Perfetto-style           | [docs_original/generate_perf_report_rocprof_pftrace.md](docs_original/generate_perf_report_rocprof_pftrace.md) |

Each format's linked doc covers its full CLI reference. For PyTorch report comparison and multi-rank collective analysis, see the corresponding docs in the [Documentation](#documentation) table.

---

## Documentation

| Module                       | Doc                                                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Trace2Tree                   | [docs_original/Trace2Tree.md](docs_original/Trace2Tree.md)                                                                       |
| TreePerf                     | [docs_original/TreePerf.md](docs_original/TreePerf.md)                                                                           |
| NCCL Analyser                | [docs_original/NcclAnalyser.md](docs_original/NcclAnalyser.md)                                                                   |
| TraceDiff                    | [docs_original/TraceDiff.md](docs_original/TraceDiff.md)                                                                         |
| Event Replay                 | [docs_original/EventReplay.md](docs_original/EventReplay.md)                                                                     |
| TraceFusion                  | [docs_original/TraceFusion.md](docs_original/TraceFusion.md)                                                                     |
| GPU Event Analyser           | [docs_original/gpu_event_analyser.md](docs_original/gpu_event_analyser.md)                                                       |
| JAX Analyses                 | [docs_original/jax_analyses.md](docs_original/jax_analyses.md)                                                                   |
| pftrace Reports              | [docs_original/generate_perf_report_rocprof_pftrace.md](docs_original/generate_perf_report_rocprof_pftrace.md)                   |
| Compare PyTorch Reports      | [docs_original/compare_perf_reports_pytorch.md](docs_original/compare_perf_reports_pytorch.md)                                   |
| Multi-Rank Collective Report | [docs_original/generate_multi_rank_collective_report_pytorch.md](docs_original/generate_multi_rank_collective_report_pytorch.md) |
| Performance Report Columns   | [docs_original/perf_report_columns.md](docs_original/perf_report_columns.md)                                                     |

---

## Development

```bash
git clone https://github.com/AMD-AGI/TraceLens.git && cd TraceLens
pip install -e .[dev]
python -m pytest tests/ -v
```

---

## Contributing

Contributions are welcome across the entire project, including new analysis modules, performance models, documentation, examples, and bug fixes.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branching, commit style, and project structure.

---

## Additional Resources

- [PyTorch Conference 2025 Poster](docs_original/TraceLens%20-%20Democratizing%20AI%20Performance%20Analysis%20-%20Adeem%20Jassani%2C%20AMD.pdf)
- [GEMMs in AI Models: Conceptual Tutorial](docs_original/conceptual/aimodels_gemms.md)
- [Trace2Tree Motivation](docs_original/conceptual/trace2tree_motivation.md)
- [PyTorch Profiling Guide](docs_original/conceptual/torch_profiling_guide.ipynb)

For more background and conceptual tutorials, browse `[docs_original/conceptual/](docs_original/conceptual/)`.