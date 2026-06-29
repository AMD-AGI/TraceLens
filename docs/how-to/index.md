<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# How-to guides

These guides walk through common TraceLens workflows with step-by-step
instructions and the output you should expect. Each guide assumes you have
already installed TraceLens (see
[Installation instructions](../install/installation.md)).

- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md) —
  Produce a multi-sheet Excel report from a `torch.profiler` trace, add roofline
  classification and recompute detection, and interpret the key sheets.
- [Generate a JAX performance report](./generate-perf-report-jax.md) — Analyze an
  XPlane protobuf trace, including GPU-event and GEMM analysis.
- [Generate a rocprof performance report](./generate-perf-report-rocprof.md) —
  Analyze `rocprofv3` JSON and `.pftrace` traces.
- [Compare two traces](./compare-traces.md) — Quantify the impact of a change by
  diffing two reports.
- [Generate a collective-communication report](./collective-report.md) — Analyze
  multi-GPU collective operations across ranks.
- [Replay a single operation](./event-replay.md) — Extract one operation into a
  standalone reproducer.
- [Fuse multi-rank traces](./trace-fusion.md) — Merge per-rank traces for
  visualization in Perfetto.
