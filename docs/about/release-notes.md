<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Release notes

This page summarizes the features available in each TraceLens release. For the
hardware and software versions validated for a release, see the
[Compatibility matrix](../reference/compatibility.md).

## TraceLens 0.1.0 (initial release)

The initial release establishes TraceLens as an end-to-end toolkit for
automated GPU trace analysis, available as command-line tools and a Python SDK.

### Report generation

- **PyTorch profiler reports** — `TraceLens_generate_perf_report_pytorch`
  builds a multi-sheet Excel report from a `torch.profiler` Chrome trace,
  including a GPU-timeline breakdown, operator-category and operator summaries,
  a unique-argument table, roofline metrics, and an optional short-kernel study.
  Compressed traces (`.zip`, `.gz`) are supported.
- **PyTorch inference reports** — `TraceLens_generate_perf_report_pytorch_inference`
  adds analysis for LLM-serving traces (vLLM/SGLang). It merges CUDA-graph
  capture traces into graph-replay traces (`--capture_folder`) to recover
  call-stack and shape information lost in graph mode, and uses per-step request
  annotations (prefill/decode counts and token statistics) to drive
  inference-specific roofline models for paged attention and fused MoE.
- **JAX reports** — `TraceLens_generate_perf_report_jax` builds reports from JAX
  XPlane protobuf traces, with optional kernel-metadata keyword filtering.
- **rocprofv3 JSON reports** — `TraceLens_generate_perf_report_rocprof` builds
  reports from rocprofv3 `*_results.json` traces, with kernel summaries,
  automatic categorization (GEMM, attention, elementwise, and others),
  short-kernel analysis, and optional grid/block detail.
- **rocprofv3 pftrace reports** — for Perfetto-style traces produced with
  `rocprofv3 --output-format pftrace`:
  - `TraceLens_generate_perf_report_pftrace_hip_activity` — per-GPU category
    summary plus kernel/HIP/XLA summaries (NSYS-style), with optional Markdown.
  - `TraceLens_generate_perf_report_pftrace_hip_api` — HIP API ↔ kernel
    correlation with the latency breakdown `T = A + Q + K`.
  - `TraceLens_generate_perf_report_pftrace_memory_copy` — memory-copy counts
    per `copy_bytes` with direction (h2d/d2h/d2d) and the GPUs involved.

### Analysis features

- **Hierarchical GPU-timeline breakdown** — Splits GPU time into computation,
  communication, memory copy, and idle time, with optional micro-idle
  classification.
- **Operator categorization** — Groups CPU operations that launch kernels into
  categories for a portable, reproducible view.
- **Unique-argument analysis** — Groups operations by name plus input shapes,
  dtypes, strides, and concrete inputs to isolate problematic input patterns.
- **Roofline modeling** — Computes arithmetic intensity (FLOPs/byte) and
  classifies operations as compute- or memory-bound relative to a target
  accelerator's roofline knee point. Optional Origami-based simulated GEMM/SDPA
  timings are available when a GPU architecture specification is provided.
- **Activation-recompute detection** and **kernel-overlap** sheets for deeper
  PyTorch analysis.

### Multi-GPU and comparison

- **Collective communication analysis** —
  `TraceLens_generate_multi_rank_collective_report_pytorch` reports time spent
  in collectives across ranks, including aggregation metrics, intra-/inter-node
  labeling, and an optional all-to-all-v heatmap.
- **Trace comparison** — `TraceLens_compare_perf_reports_pytorch` diffs two
  generated reports at the CPU-dispatch level to quantify the impact of a change
  across hardware or software versions.
- **Trace diff** — `TraceDiff` provides morphological comparison of two trace
  trees to pinpoint structural divergences (also available inline in the
  PyTorch report via `--comparison_json_path`).

### SDK modules

- **Trace2Tree** — Build and navigate the hierarchical event tree.
- **TreePerf** — GPU-timeline breakdown, per-op performance, and roofline
  metrics through the SDK.
- **PerfModel** — Compute and roofline performance models.
- **NcclAnalyser** — Multi-rank collective latency/bandwidth/skew analysis.
- **TraceDiff** — Tree-based trace comparison.
- **EventReplay** — Extract and replay isolated operations.
- **TraceFusion** — Merge multi-rank traces for Perfetto visualization.
- **TraceUtils** — Trace utilities, including inference-trace splitting
  (`TraceLens_split_inference_trace`).

```{note}
Version numbers, dates, and per-release validated configurations should be
updated in this page and in the [Compatibility matrix](../reference/compatibility.md)
with each new TraceLens release.
```
