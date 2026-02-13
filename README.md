<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens

TraceLens is a Python library focused on **automating analysis from trace files** and enabling rich performance insights. Designed with **simplicity and extensibility** in mind, this library provides tools to simplify the process of profiling and debugging complex distributed training and inference systems.
Find the PyTorch Conference 2025 poster for TraceLens [here](docs/TraceLens%20-%20Democratizing%20AI%20Performance%20Analysis%20-%20Adeem%20Jassani%2C%20AMD.pdf).
## Key Features

✨ **Hierarchical Performance Breakdowns**: Pinpoint bottlenecks with a top-down view, moving from the overall GPU timeline (idle/busy) to operator categories (e.g., convolutions), individual operators, and right down to unique argument shapes.

⚙️ **Compute & Roofline Modeling**: Automatically translate raw timings into efficiency metrics like **TFLOP/s** and **TB/s** for popular operations. Determine if an op is compute- or memory-bound and see how effectively your code is using the hardware.

🔗 **Multi-GPU Communication Analysis**: Accurately diagnose scaling issues by dissecting collective operations. TraceLens separates pure communication time from synchronization skew and calculates effective bandwidth on your real workload, not a synthetic benchmark.

🔄 **Trace Comparison**: Quantify the impact of your changes with powerful trace diffing. By analyzing performance at the CPU dispatch level, TraceLens enables meaningful side-by-side comparisons across different hardware and software versions.

▶️ **Event Replay**: Isolate any operation for focused debugging. TraceLens generates minimal, self-contained replay scripts directly from trace metadata, making it simple to share IP-safe test cases with kernel developers.

🔧 **Extensible SDK**: Get started instantly with ready-to-use scripts, then build your own custom workflows using a flexible and hackable Python API.

## Quick Start

### Installation

**1. Install TraceLens directly from GitHub:**

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

**2. Command Line Scripts for popular analyses**

- **Generate Excel Reports from Traces** — Detailed docs [here](docs/generate_perf_report.md). For report column definitions, see [Performance Report Column Definitions](docs/perf_report_columns.md). You can use compressed traces (e.g. `.zip`, `.gz`).

```bash
# PyTorch profiler traces
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```

- **rocprofv3 pftrace (Perfetto-style)** — For traces produced with `rocprofv3 --output-format pftrace` (or any Perfetto-style JSON). Two report types are available:

```bash
# 1) Record a pftrace
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace --output-format pftrace -d ./v3_traces -- python3 your_app.py

# 2a) Activity report: category summary per GPU, kernel/HIP/XLA summaries (NSYS-style). Optional Markdown output.
TraceLens_generate_perf_report_pftrace_hip_activity --trace_path sample.pftrace --write_md

# 2b) API↔Kernel report: latency breakdown T = A + Q + K (API duration, queue delay, kernel duration)
TraceLens_generate_perf_report_pftrace_hip_api --trace_path sample.pftrace

# 2c) Memory copy report: count per copy_bytes with direction (h2d/d2h/d2d and which GPU(s))
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace
# Custom Excel path:
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace --output_xlsx_path report.xlsx
# CSV directory:
TraceLens_generate_perf_report_pftrace_memory_copy --trace_path sample.pftrace --output_csvs_dir ./out
```

For `.pftrace` files, `traceconv` is optional: if not on `PATH`, it is downloaded automatically into the trace directory. You can also pass `--traceconv /path/to/traceconv` explicitly.

- **rocprofv3 JSON** — For `*_results.json` from rocprofv3. See [docs/generate_perf_report_rocprof.md](docs/generate_perf_report_rocprof.md).

```bash
TraceLens_generate_perf_report_rocprof --profile_json_path path/to/trace_results.json --short_kernel_study --kernel_details
```

- **Compare Traces** — Detailed docs [here](docs/compare_perf_reports_pytorch.md).

```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx \
    candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```

- **Generate Collective Performance Report** — Detailed docs [here](docs/generate_multi_rank_collective_report_pytorch.md).

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8 \
```

Refer to the individual module docs in the `docs/` directory and the example notebooks under `examples/` for further guidance.

**Development & testing** — Install in editable mode and run tests:

```bash
git clone https://github.com/AMD-AGI/TraceLens.git && cd TraceLens
pip install -e .
python -m pytest tests/ -v
```

Pftrace report tests only: `python -m pytest tests/test_pftrace_hip_api_perf_report.py tests/test_pftrace_hip_activity_report.py -v`.

**📦 Custom Workflows**: Check out [examples/custom_workflows/](examples/custom_workflows/) for community-contributed utilities including **roofline_analyzer** and **traceMap** — powerful tools we're working on integrating more tightly into the core library.

## Supported Profile Formats

TraceLens supports multiple profiling formats:

| Format | Tool | Documentation |
|--------|------|---------------|
| **PyTorch** | `torch.profiler` | [docs/generate_perf_report.md](docs/generate_perf_report.md) |
| **JAX** | XPlane protobuf | [docs/generate_perf_report_jax.md](docs/generate_perf_report_jax.md) |
| **rocprofv3 JSON** | AMD ROCm rocprofiler-sdk | [docs/generate_perf_report_rocprof.md](docs/generate_perf_report_rocprof.md) |
| **rocprofv3 pftrace** | Perfetto-style (e.g. rocprofv3 `--output-format pftrace`) | See [pftrace reports](#pftrace-rocprofv3--perfetto) below |

### rocprofv3 JSON

For rocprofv3 `*_results.json` traces:

```bash
TraceLens_generate_perf_report_rocprof \
    --profile_json_path trace_results.json \
    --short_kernel_study \
    --kernel_details
```

Features: GPU timeline breakdown (kernel, memory, idle time), kernel summary with statistical analysis, automatic kernel categorization (GEMM, Attention, Elementwise, etc.), short kernel analysis, grid/block dimension tracking. See [docs/generate_perf_report_rocprof.md](docs/generate_perf_report_rocprof.md) for detailed usage.

### pftrace (rocprofv3 / Perfetto)

For **Perfetto-style** traces (e.g. rocprofv3 with `--output-format pftrace`, or JSON with a `traceEvents` array), TraceLens provides two report generators that share the same trace loading and optional **traceconv** handling (for `.pftrace` → JSON conversion):

| CLI | Description | Output |
|-----|-------------|--------|
| `TraceLens_generate_perf_report_pftrace_hip_activity` | Category summary per GPU, kernel/HIP/XLA summaries (NSYS-style), optional Markdown | Excel, CSV, optional `.md` |
| `TraceLens_generate_perf_report_pftrace_hip_api` | HIP API ↔ kernel correlation; latency breakdown **T = A + Q + K** (API duration, queue delay, kernel duration) | Excel, CSV |

- **Input:** `.json`, `.json.gz`, or `.pftrace`. For `.pftrace`, **traceconv** is resolved from `PATH` or downloaded into the trace file’s directory if not provided via `--traceconv`.
- **Library usage:** Both scripts can be imported and called with a trace path; they return a dictionary of pandas DataFrames (e.g. `api_kernel_summary`, `category_summary`, `kernel_summary`, `hip_summary`).

## Contributing

We welcome issues, bug reports, and pull requests. Feel free to open discussions in the GitHub repository
or contribute new performance models, operator mappings or analysis modules. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
