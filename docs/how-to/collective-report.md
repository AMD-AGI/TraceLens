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

For arbitrarily named files (for example, traces written by
`torch.profiler.tensorboard_trace_handler`, which are not named
`rank0_trace.json`), use `--trace_glob` with `--rank_regex` to extract the rank
from each filename:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_glob "/path/to/tensorboard/**/*.pt.trace.json.gz" \
    --rank_regex "rank\[(?P<rank>\d+)\]" \
    --world_size 16
```

`--rank_regex` must contain a named group `rank` (or a single capture group).

**Expected output:** an Excel report summarizing collective operations across
ranks, with aggregation metrics (`mean`, `median`, `min`, `max` by default). If
neither `--output_xlsx_path` nor `--output_csvs_dir` is given, the workbook is
written to `--trace_dir` (or the lowest common directory of the resolved
traces).

## Step 2: Read the report sheets

Each collective is assigned a `collective_id` (its Process Group name plus an
`index_in_group` reflecting its order in the trace); this is how the tool
correlates the same collective across ranks. The workbook contains a sheet per
collective type and analysis. The most useful:

| Sheet | Granularity | What it contains |
|-------|-------------|------------------|
| `nccl_summary_implicit_sync` | one row per (collective name, dtype, message size) | Aggregated latency, skew, and bandwidth (algorithm and bus) for collectives that incur implicit sync (AllReduce, AllGather, ReduceScatter, balanced AllToAll). |
| `nccl_summary_long` | one row per (collective name, dtype, message size) | Duration aggregates (sum/mean/std/min/max) and counts across all ranks. |
| `nccl_implicit_sync` | one row per collective | Per-collective implicit-sync breakdown; per-rank timestamps, durations, and `wait_time` are spread across `rank_<i>_*` columns. |
| `nccl_long` | one row per (collective, rank) | Per-event, per-rank NCCL records (timestamps, durations, stream, message sizes) for drilling into specific ops. |
| `nccl_summary_all2allv` / `nccl_all2allv` | per (process group, dtype) / per (collective, rank) | All-to-all-v metrics (throughput, wall time, size imbalance, skew), aggregated and per-rank. |
| `nccl_all2allv_heatmap` | one row per rank, one column per rank | Per rank-pair send volumes (only with `--all2allv_heatmap`). |
| `straggler_summary` | one row per rank | Aggregated straggler metrics — see Step 3. |

Each sheet is produced only when the corresponding collective type is present in
the trace (for example, the all2allv sheets are skipped if the run has no
all-to-all-v). The detailed per-event sheets (`nccl_long`, `nccl_implicit_sync`,
`nccl_all2allv`) are controlled by `--detailed_analysis`, which is on by default.

### Interpret communication versus skew

The implicit-sync sheets separate the time spent in actual data movement from the
time a rank spends waiting for other ranks to arrive. The key metrics:

| Metric | Meaning |
|--------|---------|
| `comm_latency` | Communication time for the collective, taken as the **minimum duration across ranks** — this strips out per-rank waiting so it reflects real data movement. |
| `algo bw` | Algorithm bandwidth: message size / `comm_latency`. |
| `bus bw` | Bus bandwidth (link utilization). |
| `wait_time`, `skew in start time` | How much later some ranks arrive than the earliest rank — the synchronization cost. |

High skew points to load imbalance upstream rather than a slow network; high
`comm_latency` with low skew points to a bandwidth or topology limit.

### Interpret all-to-all-v collectives (MoE / expert parallel)

All-to-all-v sends a different amount of data per rank, so there is no single
message size and the ring/tree `algo bw` / `bus bw` formulas do not apply.
The all2allv sheets instead report:

| Metric | Meaning |
|--------|---------|
| `throughput (GB/s)` | Total data moved / wall-clock time. Compare against link bandwidth to gauge efficiency. |
| `wall_time` | End-to-end time from the first rank entering to the last finishing. |
| `size_imbalance` | Max rank's data ÷ mean. `1.0` = balanced; `>> 1.0` = some ranks carry disproportionate load (common when some MoE experts are hotter than others). |
| `skew in start time` | How far apart ranks enter the collective — large skew means some ranks are blocked by upstream compute. |

Reading it: low `throughput` with `size_imbalance` ≈ 1.0 → software/driver
overhead; low `throughput` with `size_imbalance` >> 1.0 → expert-routing
imbalance, where the busiest rank gates the collective. Use
`--all2allv_heatmap` to see which rank pairs carry the most traffic.

## Step 3: Find the straggler rank

A *straggler* is the rank that consistently arrives last at collectives, forcing
the others to wait in implicit synchronization. Open the `straggler_summary`
sheet: it is sorted so the **straggler is the first row** (lowest total wait
time — it arrives last, so it rarely waits itself).

| Column | What it tells you |
|--------|-------------------|
| `total_wait_time_us` | Sum of this rank's wait across all collectives. The straggler has the **lowest** value. |
| `times_arrived_last` / `pct_arrived_last` | How often this rank was last. The straggler dominates these. |
| `times_arrived_first` | How often this rank was first; the rank highest here pays the biggest sync penalty. |
| `total_nccl_dur_us` | Time inside NCCL kernels; the straggler is typically **lowest** (others are inflated by wait time). |

Once you know the straggler, merge its trace with a fast rank using
[trace fusion](./trace-fusion.md) and open the result in Perfetto to see what
compute or memory work delays the straggler before each collective.

## Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--trace_dir` / `--trace_pattern` / `--trace_glob` | — | Mutually exclusive ways to locate the per-rank traces (directory, `*`-pattern, or glob). Exactly one is required. |
| `--world_size` | required | Number of ranks. |
| `--rank_regex` | `rank[\[\-_/]?(?P<rank>\d+)` | With `--trace_glob`, regex (named group `rank` or one capture group) used to extract the rank from each filename. |
| `--output_xlsx_path` | auto | Output Excel path. If omitted, written to the trace directory / lowest common directory of the traces. |
| `--output_csvs_dir` | None | Write each sheet as a separate CSV here instead of (or alongside) Excel. |
| `--detailed_analysis` | on | Emit the per-event detail sheets (`nccl_long`, `nccl_implicit_sync`, `nccl_all2allv`) in addition to the summaries. |
| `--agg_metrics` | `mean median min max` | Aggregation metrics in the summary sheets. |
| `--gpus_per_node` | auto | Adds `node_id` / `node_span` columns and labels each process group `intra_node` or `inter_node`. Auto-detected from trace `deviceProperties` if omitted. |
| `--all2allv_heatmap` | off | Add the `nccl_all2allv_heatmap` sheet (per rank-pair send volumes) — useful for spotting imbalanced all-to-all / MoE routing. |
| `--use_multiprocessing` | off | Load traces in parallel — a significant speedup when processing many large per-rank traces. |
| `--max_workers` | `os.cpu_count()` | Cap on worker processes (requires `--use_multiprocessing`). |

Example combining topology labeling, the heatmap, and parallel loading:

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8 \
    --gpus_per_node 8 \
    --all2allv_heatmap \
    --use_multiprocessing
```

## Next steps

- For SDK-level collective analysis, see the
  [`nccl_analyser_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/nccl_analyser_example.ipynb)
  notebook in the repository.
- For JAX collectives, see [Generate a JAX performance report](./generate-perf-report-jax.md).
