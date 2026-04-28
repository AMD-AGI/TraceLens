<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Generate Multi-Rank Collective Report

This utility analyzes **PyTorch JSON profile traces** from **multiple ranks** and produces a comprehensive **NCCL communication report** (Excel workbook and/or CSVs). 

---

## 🚀 Quick Start

Run the script with **one** of: a directory containing trace files, a trace file **pattern** with a single `*` placeholder for rank id, or a **glob** pattern with a regex to extract the rank.

### Directory mode
```bash
python generate_multi_rank_collective_report_pytorch.py   --trace_dir /path/to/traces   --world_size 8
```

### Pattern mode (strict rank substitution)
```bash
python generate_multi_rank_collective_report_pytorch.py   --trace_pattern "/logs/job123/rank*/trace.json"   --world_size 8
```

### Glob mode (tensorboard-style / arbitrary filenames)
When traces are produced by `torch.profiler.tensorboard_trace_handler`, filenames are typically not `rank{i}_trace.json`. Use `--trace_glob` with an optional `--rank_regex` to extract the rank id:
```bash
python generate_multi_rank_collective_report_pytorch.py \
  --trace_glob "/path/to/tensorboard/**/*.pt.trace.json.gz" \
  --rank_regex "rank\[(?P<rank>\d+)\]" \
  --world_size 16
```

### Parallel mode (for faster processing)
```bash
python generate_multi_rank_collective_report_pytorch.py   --trace_dir /path/to/traces   --world_size 8   --use_multiprocessing
```

> **Note:** Speedup varies based on available CPU cores, trace file sizes, and disk I/O.

> The pattern must contain exactly one `*`, which is replaced with `0, 1, ..., world_size-1` to enumerate rank file paths.

### Installed entry point (if packaged)
If you install this script as part of a package with a console entry point, you can call it directly, e.g.:
```bash
TraceLens_generate_multi_rank_collective_report_pytorch   --trace_dir /path/to/traces   --world_size 8
```
---

## 📥 Input

- **PyTorch JSON traces** captured per-rank (e.g., via `torch.profiler` with JSON export).
- Naming patterns:
  - Directory mode (files inside `--trace_dir`): `rank0_trace.json`, `rank1_trace.json`, ...
  - Pattern mode: `--trace_pattern "/path/to/rank*_trace.json"`
  - Glob mode: `--trace_glob "/path/to/**/*.pt.trace.json.gz"` with `--rank_regex` for rank extraction

---

## ⚙️ Command-Line Options

> Provide **exactly one** of `--trace_dir`, `--trace_pattern`, or `--trace_glob`.

| Argument | Default | Description |
|---|---|---|
| `--trace_dir` | `None` | Directory containing trace files (expects names like `rank*_trace.json`). |
| `--trace_pattern` | `None` | Template path with a **single** `*` placeholder for the rank id (strict substitution: `* → 0..world_size-1`). |
| `--trace_glob` | `None` | Glob for trace files with arbitrary names (supports `**`). Requires `--world_size` and uses `--rank_regex` to map files to ranks. |
| `--rank_regex` | `rank[\[\-_/]?(?P<rank>\d+)` | Regex used with `--trace_glob` to extract rank id. Must contain a named group `rank` or a single capture group. |
| `--world_size` | `Required` | Number of ranks in the distributed run.  |
| `--output_xlsx_path` | _auto-inferred_ | Path to save the Excel workbook. If omitted, a sensible default is created (see Output Behavior). |
| `--output_csvs_dir` | `None` | If provided, writes each sheet as an individual `.csv` under this directory. |
| `--detailed_analysis` | `False` | Include detailed per-rank sheets (see below). |
| `--agg_metrics` | `mean median min max` | Aggregations to compute in summaries. Allowed: `mean`, `median`, `min`, `max` (space-separated). |
| `--use_multiprocessing` | `False` | Enable parallel trace loading using multiprocessing. Can provide speedup (system-dependent) but uses more CPU resources. |
| `--max_workers` | `os.cpu_count()` | Maximum number of worker processes for parallel loading (requires `--use_multiprocessing`). Override to limit resource usage if needed. |
| `--all2allv_heatmap` | `False` | Add an `nccl_all2allv_heatmap` sheet with per rank-pair send volumes across all all2allv invocations. Useful for diagnosing MoE/expert-parallel routing imbalance. |
| `--gpus_per_node` | _auto-detected_ | Number of GPUs per node. When known, `node_id` and `node_span` columns are added to report sheets, labeling each collective's process group as `intra_node` or `inter_node`. Auto-detected from trace `deviceProperties` if omitted. |


---

## 📊 Excel Workbook Sheets

| Sheet Name | What it contains |
|---|---|
| `nccl_summary_implicit_sync` | Aggregated view of collectives that incurred implicit sync (AllReduce, AllGather, ReduceScatter, AllToAll balanced). Provides latency, skew, and aggregated bandwidth metrics (algorithm and bus). |
| `nccl_summary_long` | Aggregated NCCL operation stats per (rank, process group, collective, dtype, size). Includes counts and duration aggregates (sum/mean/std/min/max). |
| `nccl_long` | Per-event, per-rank NCCL records with timestamps, durations, stream, message sizes, and other attributes. Useful for drilling into specific slow ops. |
| `nccl_implicit_sync` | Per-collective implicit synchronization breakdown. Includes per-rank wait_time columns indicating time lost to implicit sync, plus timing/skew and bandwidth metrics. |
| `nccl_summary_all2allv` | Aggregated all2allv metrics by (Process Group, dtype). Includes throughput, wall time, size imbalance, and timing skew. Unlike implicit-sync collectives, all2allv uses aggregate throughput rather than algo/bus bandwidth since per-rank data sizes vary. |
| `nccl_all2allv` | Detailed All2AllV analysis: variable send/recv sizes and splits per rank, with timing and skew columns per rank. |
| `nccl_all2allv_heatmap` | *(when `--all2allv_heatmap` is set)* Per rank-pair total send volumes across all all2allv invocations. Useful for identifying hot pairs in MoE/expert-parallel routing. |
| `straggler_summary` | One row per rank with aggregated straggler metrics: total/mean wait time, how often the rank arrived last or first, and total NCCL duration. Sorted ascending by wait time so the straggler (lowest wait time) is at the top. See [Identifying Straggler Ranks](#-identifying-straggler-ranks). |

_Notes:_
- Summary sheets (`nccl_summary_*`, `nccl_summary_all2allv`) are **always** produced when the corresponding collective type exists in the trace; detailed per-event sheets (`nccl_long`, `nccl_implicit_sync`, `nccl_all2allv`) appear when `--detailed_analysis` is set.
- When `gpus_per_node` is known (auto-detected or set via `--gpus_per_node`), sheets that contain `rank` and/or `Process Group Ranks` columns gain `node_id` and `node_span` columns (fully aggregated summary sheets without these columns remain unchanged). `node_span` is `intra_node` when all ranks in the process group reside on the same node, or `inter_node` otherwise. You can filter or pivot on these columns in Excel to compare intra- vs inter-node communication.
- Column sets can evolve; the above reflects the provided example workbook.
---

## 📤 Output Behavior

- If **neither** `--output_xlsx_path` nor `--output_csvs_dir` is specified, the tool writes an Excel file to:
  - the provided `--trace_dir`, or
  - the **lowest common directory** of the resolved trace files (from `--trace_pattern` or `--trace_glob`).
- If `--output_csvs_dir` is set, all sheets are written as individual CSV files under that directory.
- If `--output_xlsx_path` is set, a single Excel workbook containing all sheets is created.
- `openpyxl` is required for Excel; the tool will attempt to install it when missing.

**Lowest Common Directory (pattern mode)**  
When using `--trace_pattern`, the tool expands the pattern by replacing `*` with `0..world_size-1`, then computes the **lowest common ancestor directory** of those paths to pick a default output location.

---

## 🔍 Identifying Straggler Ranks

A **straggler** is the rank that consistently arrives last at collectives, forcing all other ranks to wait in implicit synchronization.

### Quick answer: open `straggler_summary`

The `straggler_summary` sheet is sorted so **the straggler is in the first row** (lowest total wait time). Example from an 8-rank Llama 70B FSDP run:

| rank | total_wait_time_us | mean_wait_time_us | times_arrived_last | times_arrived_first | pct_arrived_last | num_collectives | total_nccl_dur_us |
|------|-------------------|-------------------|-------------------|--------------------|-----------------:|----------------:|------------------:|
| 4 | 27,195 | 55.7 | 420 | 6 | 86.1% | 488 | 3,910,753 |
| 5 | 4,660,353 | 9,549.9 | 39 | 10 | 8.0% | 488 | 8,463,896 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 0 | 13,358,753 | 27,374.5 | 3 | 320 | 0.6% | 488 | 17,203,432 |

**How to read it:**

| Column | What it tells you |
|--------|-------------------|
| `total_wait_time_us` | Sum of this rank's wait time across all collectives. The straggler has the **lowest** value — it arrives last, so it rarely waits. |
| `times_arrived_last` | Number of collectives where this rank was the last to arrive. The straggler dominates this column. |
| `times_arrived_first` | Number of collectives where this rank arrived first. The rank that appears here most often pays the biggest sync penalty. |
| `pct_arrived_last` | `times_arrived_last / num_collectives`. A rank at 86% is a persistent straggler. |
| `total_nccl_dur_us` | Total time inside NCCL kernels. The straggler typically has the **lowest** value — other ranks' durations are inflated by implicit-sync wait time. |

### Next step: visualize with TraceFusion

Once you know the straggler rank, use [TraceFusion](TraceFusion.md) to merge the straggler's trace with a fast rank (e.g. the one with the highest `times_arrived_first`) into a single file and open it in Perfetto UI. Viewing both timelines side-by-side makes it straightforward to see what compute or memory work is delaying the straggler before each collective.

```python
from TraceLens import TraceFuse

fuser = TraceFuse(
    trace_files=["rank4_trace.json.gz", "rank0_trace.json.gz"],
    output_file="straggler_vs_fast.json.gz",
)
fuser.fuse()
```

---

## Interpreting all2allv Metrics

**Why different metrics from other collectives?**

All2allv sends variable amounts of data per rank — there is no single "message size" or uniform workload. The standard NCCL `algo bw` and `bus bw` formulas assume ring/tree algorithms with equal data on every rank, so they don't apply.

| Metric | What it tells you |
|--------|-------------------|
| `throughput (GB/s)` | Total data moved / wall-clock time of the collective. Compare against theoretical link bandwidth to gauge efficiency. |
| `wall_time (us)` | End-to-end time from first rank entering to last rank finishing. High values + low throughput = bottleneck. |
| `size_imbalance` | Ratio of max rank's data to mean. 1.0 = balanced. Values >> 1.0 indicate some ranks carry disproportionate load — common in MoE when some experts are "hotter" than others. |
| `max_rank_dur / min_rank_dur` | Spread in per-rank kernel time. Large spread + high imbalance = consider rebalancing expert routing or token dropping. |
| `skew in start time` | How far apart ranks enter the collective. High skew means some ranks are blocked by prior compute. |

**Actionable guidance:**
- If `throughput` is much lower than link bandwidth and `size_imbalance` ≈ 1.0 → likely a software/driver overhead issue.
- If `throughput` is low and `size_imbalance` >> 1.0 → expert routing imbalance; the busiest rank gates the collective.
- If `skew in start time` is large → ranks are entering the collective at very different times, indicating upstream compute imbalance.
- Use the heatmap sheet (`--all2allv_heatmap`) to identify which rank pairs carry the most traffic.

---

