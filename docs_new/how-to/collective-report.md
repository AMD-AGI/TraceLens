<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Generate a collective-communication report

This guide shows how to analyze multi-GPU collective operations across ranks to
diagnose scaling issues, separating true communication time from
synchronization skew.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- Per-rank PyTorch profiler traces from a distributed run (one trace per rank).

## Step 1: Generate the report

If your traces are in a single directory:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8
```

If the per-rank files follow a naming pattern, use `--trace_pattern` with a
single `*` placeholder for the rank:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_pattern "/path/to/trace_rank_*_step_3.json" \
    --world_size 8
```

For arbitrarily named files, use `--trace_glob` together with `--world_size`
(and `--rank_regex` to map files to ranks).

**Expected output:** an Excel report summarizing collective operations across
ranks, with aggregation metrics (`mean`, `median`, `min`, `max` by default).

## Step 2: Interpret communication versus skew

For each collective, TraceLens separates the time spent in actual data movement
from the time a rank spends waiting for other ranks to arrive (synchronization
skew). High skew points to load imbalance rather than a slow network; high pure
communication time with low skew points to a bandwidth or topology limit.

## Step 3: Add topology and heatmaps (optional)

- `--gpus_per_node N` adds `node_id` and `node_span` columns and labels each
  collective's process group as `intra_node` or `inter_node`. If omitted,
  TraceLens auto-detects it from trace metadata.
- `--all2allv_heatmap` adds an `nccl_all2allv_heatmap` sheet with per rank-pair
  send volumes — useful for spotting imbalanced all-to-all communication.
- `--use_multiprocessing` (with optional `--max_workers`) speeds up loading many
  large traces.

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8 \
    --gpus_per_node 8 \
    --all2allv_heatmap \
    --use_multiprocessing
```

## Next steps

- See `docs/NcclAnalyser.md` and `examples/nccl_analyser_example.ipynb` in the
  repository for the SDK-level collective analysis.
- For JAX collectives, see [Analyze JAX traces](./jax-reports.md).
