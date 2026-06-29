<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens documentation

TraceLens is an open-source Python library developed by AMD that automates
performance analysis from GPU trace files. Instead of manually inspecting raw
profiling data in tools such as Perfetto or Chrome Trace Viewer, TraceLens
parses traces from PyTorch, JAX, and `rocprofv3` and produces structured
performance reports — including hierarchical GPU-timeline breakdowns,
per-operator roofline analysis (TFLOP/s, TB/s), and multi-GPU communication
diagnostics.

TraceLens can pinpoint whether a kernel is compute-bound or memory-bound,
separate true communication time from synchronization skew in distributed
workloads, and compare traces side by side to quantify the impact of code or
configuration changes. It is available both as a set of command-line tools for
quick report generation and as a Python SDK for building custom analysis
workflows, making it useful for one-off debugging and repeatable performance
regression testing alike.

The TraceLens source code is hosted at
[github.com/AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens).

## What TraceLens does

- **Hierarchical performance breakdowns** — Pinpoint bottlenecks with a
  top-down view, moving from the overall GPU timeline (idle/busy) to operator
  categories, individual operators, and right down to unique argument shapes.
- **Compute and roofline modeling** — Translate raw timings into efficiency
  metrics such as TFLOP/s and TB/s. Determine whether an operation is compute-
  or memory-bound and see how effectively your code uses the hardware.
- **Multi-GPU communication analysis** — Diagnose scaling issues by dissecting
  collective operations. TraceLens separates pure communication time from
  synchronization skew and calculates effective bandwidth on your real
  workload.
- **Trace comparison** — Quantify the impact of changes with trace diffing at
  the CPU-dispatch level, enabling meaningful side-by-side comparisons across
  hardware and software versions.
- **Event replay** — Isolate any operation for focused debugging. TraceLens
  generates minimal, self-contained replay scripts from trace metadata, making
  it simple to share IP-safe reproducers with kernel developers.
- **Extensible SDK** — Start with ready-to-use scripts, then build custom
  workflows with a flexible Python API.

## Use cases

- **Performance debugging** — Find the operators and kernels responsible for low
  GPU utilization in a training or inference workload.
- **Roofline efficiency analysis** — Measure how close each operator runs to the
  hardware's compute and memory-bandwidth limits on a given accelerator.
- **Distributed scaling analysis** — Quantify exposed communication and
  synchronization skew across ranks in multi-GPU and multi-node runs.
- **Regression testing** — Compare a baseline trace against a candidate to
  quantify the effect of a code, library, or hardware change.
- **Reproducer generation** — Extract a single operator into a standalone replay
  script to share with kernel or framework developers.

## Documentation overview

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Release information
- [Release notes](./about/release-notes.md)
- [Compatibility matrix](./reference/compatibility.md)
:::

:::{grid-item-card} Install
- [Installation instructions](./install/installation.md)
:::

:::{grid-item-card} Reference
- [API reference](./reference/api-reference.md)
:::

:::{grid-item-card} How to
- [How-to guides](./how-to/index.md)
:::

:::{grid-item-card} About
- [License](./about/license.md)
:::

::::

## Supported profile formats

| Format | Source tool | Report CLI |
|--------|-------------|------------|
| PyTorch | `torch.profiler` | `TraceLens_generate_perf_report_pytorch` |
| PyTorch (inference) | `torch.profiler` | `TraceLens_generate_perf_report_pytorch_inference` |
| JAX | XPlane protobuf | `TraceLens_generate_perf_report_jax` |
| rocprofv3 JSON | AMD ROCm rocprofiler-sdk | `TraceLens_generate_perf_report_rocprof` |
| rocprofv3 pftrace (Perfetto-style) | `rocprofv3 --output-format pftrace` | `TraceLens_generate_perf_report_pftrace_hip_activity`, `..._pftrace_hip_api`, `..._pftrace_memory_copy` |

## Example notebooks

Hands-on notebooks in the repository walk through the core features:

| Example | What it covers |
|---------|----------------|
| `examples/trace2tree_example.ipynb` | Navigate the hierarchical event tree — link Python ops, CPU dispatches, and GPU kernels |
| `examples/tree_perf_example.ipynb` | GPU-timeline breakdown, per-op performance, and roofline metrics via the SDK |
| `examples/nn_module_view.ipynb` | GPU time broken down by `nn.Module` |
| `examples/nccl_analyser_example.ipynb` | Multi-rank collective analysis: latency, bandwidth, skew |
| `examples/trace_diff_example.ipynb` | Morphological comparison of two trace trees |
| `examples/event_replayer_example.ipynb` | Extract and replay operations for isolated debugging |
| `examples/trace_fusion_example.py` | Merge multi-rank PyTorch traces for Perfetto visualization |
| `examples/roofline_plots_example.ipynb` | Roofline-style visualizations for specific operators |
| `examples/jax_nccl_analyser_example.ipynb` | Bandwidth analysis for JAX collectives from XPlane traces |

## License

TraceLens is released under the MIT License. For the full text, see the
[License](./about/license.md) page.
