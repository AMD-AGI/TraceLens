<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Replay a single operation

This guide shows how to isolate a single operation from a trace into a minimal,
self-contained replay script — useful for focused debugging and for sharing
IP-safe reproducers with kernel or framework developers.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A PyTorch profiler trace containing the operation you want to isolate.

## How it works

The `EventReplay` module reads an operation's metadata from the trace —
operation name, input shapes, dtypes, strides, and other arguments — and
generates a standalone Python script that reconstructs and re-runs just that
operation. Because the replay script is built from trace metadata rather than
your model code, it can be shared without exposing model IP.

## Step 1: Identify the operation

Generate a PyTorch report (see
[Generate a PyTorch performance report](./generate-perf-report.md)) and use the
`ops_unique_args` sheet to find the operation and input shape you want to
isolate.

## Step 2: Generate the replay script

Follow the `examples/event_replayer_example.ipynb` notebook, which walks through
loading the trace, building the event tree, selecting the target event, and
emitting a replay script through the `EventReplay` SDK module.

**Expected output:** a self-contained Python script that allocates inputs
matching the recorded shapes and dtypes and invokes the isolated operation,
ready to run independently of the original workload.

## Step 3: Run and share

Run the generated script on the target hardware to reproduce the operation in
isolation, or share it with kernel developers as a minimal reproducer.

## Next steps

- See `docs/EventReplay.md` in the repository for the full module reference.
