<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Fuse multi-rank traces

This guide shows how to merge per-rank PyTorch traces from a distributed run
into a single file that can be visualized together in Perfetto.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- Per-rank PyTorch profiler traces (one trace per rank).

## Why fuse traces

By default, each rank in a distributed run produces its own trace. Inspecting
them separately makes it hard to see how ranks interact — for example, when one
rank stalls waiting on a collective. `TraceFusion` merges the per-rank traces
into a single timeline so all ranks can be viewed together in Perfetto.

## Step 1: Fuse the traces

Follow the `examples/trace_fusion_example.py` script in the repository, which
loads the per-rank traces and writes a single merged trace file.

```bash
python examples/trace_fusion_example.py
```

**Expected output:** a single merged trace file containing the events from all
ranks, aligned on a common timeline.

## Step 2: Visualize in Perfetto

Open the merged trace in [Perfetto UI](https://ui.perfetto.dev/) (or the
Chrome trace viewer) to inspect cross-rank behavior, such as exposed
communication and synchronization gaps.

## Next steps

- For quantitative collective analysis instead of visualization, see
  [Generate a collective-communication report](./collective-report.md).
- See `docs/TraceFusion.md` in the repository for the full module reference.
