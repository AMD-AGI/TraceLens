<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# How-to guides

These guides walk through common TraceLens workflows with step-by-step
instructions and the output you should expect. Each guide assumes you have
already installed TraceLens (see
[Installation instructions](../install/installation.md)).

- [Generate a PyTorch performance report](./generate-perf-report.md) — Produce a
  multi-sheet Excel report from a `torch.profiler` trace and interpret the key
  sheets.
- [Add roofline analysis](./roofline-analysis.md) — Classify operations as
  compute-bound or memory-bound using a GPU architecture specification.
- [Compare two traces](./compare-traces.md) — Quantify the impact of a change by
  diffing two reports.
- [Generate a collective-communication report](./collective-report.md) — Analyze
  multi-GPU collective operations across ranks.
- [Analyze rocprofv3 JSON traces](./rocprof-reports.md) — Generate a report from
  a rocprofv3 `*_results.json` trace.
- [Analyze rocprofv3 pftrace files](./pftrace-reports.md) — Generate activity,
  API↔kernel, and memory-copy reports from Perfetto-style traces.
- [Analyze JAX traces](./jax-reports.md) — Generate a report from a JAX XPlane
  protobuf trace.
- [Replay a single operation](./event-replay.md) — Extract one operation into a
  standalone reproducer.
- [Fuse multi-rank traces](./trace-fusion.md) — Merge per-rank traces for
  visualization in Perfetto.
