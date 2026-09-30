<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analyze GPU idle time in TraceLens
```{meta}
:description: Learn how to classify GPU idle gaps in a PyTorch profiler trace by root cause with TraceLens, and read the idle_overview, idle_summary, and idle_intervals sheets.
:keywords: TraceLens, GPU idle time, idle gaps, launch latency, synchronization, CPU overhead, PyTorch profiler, Perfetto, ROCm, AMD Instinct
```

TraceLens can classify every gap between consecutive GPU events (kernels,
memory copies, and memsets) by root cause, instead of reporting GPU idle time as
a single number. This topic explains the background, shows how to enable the
analysis, and walks through reading the resulting sheets.

Each idle interval is classified along two axes:

- **`drain_type`:** why the GPU queue emptied — a synchronization call drained it
  (`sync_drain`), or the CPU couldn't submit work fast enough (`starved`).
- **`cpu_during_gap`:** what the CPU was doing during the gap —
  `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`,
  `CPU_DOMINATED`, or `CPU_UNTRACED`.

## Before you begin

Before running the analysis, confirm you have the following:

- [TraceLens installed](../install/install.md).
- A `torch.profiler` Chrome trace (`.json` or `.json.gz`) with CPU and GPU
  activity. See
  [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
  for capture settings. Capturing with `with_stack=True` improves CPU-side
  attribution.

## Background concepts

### CPU-GPU asynchronous execution

The CPU submits work to the GPU by enqueueing kernel launches (and similar
operations) into one or more GPU queues. The GPU consumes that queue
independently. While the GPU runs kernels, the CPU can keep preparing and
enqueueing more work, so the two processors overlap in time.

If the CPU stops submitting new work and the queue empties, the GPU has nothing
left to run and becomes *idle* until the next launch arrives. Long or frequent
idle stretches mean the GPU isn't doing useful computation during that window.

### GPU queue and launch latency

After the CPU issues a launch (for example, through `hipLaunchKernel` or
`cudaLaunchKernel`), there's a short delay before the kernel begins executing on
the GPU. That delay is *launch latency*; it's typically on the order of 5–20
microseconds, depending on driver, runtime, and workload. If the CPU keeps the
queue full, successive kernels can start back-to-back and this per-launch gap is
largely hidden.

### Synchronization and queue draining

Operations such as `hipDeviceSynchronize`, `hipStreamSynchronize`, event waits,
or blocking device-to-host copies force the CPU to wait until the GPU reaches a
defined point. While the CPU waits, it usually can't enqueue additional kernels,
so the GPU queue may drain and the GPU can sit idle even though the host process
is still busy from a logical standpoint.

### Why idle time matters

Idle GPU time is time the hardware isn't applying to your model. In training,
reducing idle time often improves step time and throughput. Common contributors
include CPU overhead between operators, synchronization points, allocator or
runtime stalls, and cases where the host can't feed the GPU fast enough.

## Run the analysis

Add `--enable_idle_analysis` to the PyTorch report to append the
`idle_overview`, `idle_summary`, and `idle_intervals` sheets:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis
```

Add `--enable_augmented_trace` to also write
`<trace>_idle_augmented.json.gz` next to the input trace. It's the original trace
plus idle annotation tracks on the GPU process, for viewing in
[Perfetto](https://ui.perfetto.dev):

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis \
    --enable_augmented_trace
```

Gaps shorter than `--micro_idle_thresh_us` (default `5` µs for idle analysis)
are labeled noise and skipped by the classifier.

To classify idle time without generating the full report, run the standalone
module. It writes an Excel workbook with the three idle sheets and an augmented
trace (`<trace>_idle_classified.json.gz` by default; override with `-o`):

```bash
python -m TraceLens.Reporting.classify_idle_time trace.json --micro-thresh 5.0
```

Or use the Python API, which returns the same sheets as pandas `DataFrame`
objects:

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser

pa = TreePerfAnalyzer.from_file("trace.json")
analyser = IdleTimeAnalyser(pa.tree, micro_thresh_us=5.0)
dfs = analyser.get_dataframes()        # {"idle_overview": ..., "idle_summary": ..., "idle_intervals": ...}
intervals = analyser.classify()        # list of per-interval dicts
```

## Read the report

### Step 1: idle_overview

Start here for a coarse picture. Check `gpu_utilization_pct` (GPU busy time as a
share of busy plus macro idle time). If utilization is already very high (for
example, above roughly 95%), idle time is unlikely to be the primary bottleneck;
other limits (memory, kernel duration, algorithmic cost) may dominate.

Use the breakdown by `drain_type` and `cpu_during_gap` to see which combination
of "why the queue emptied" and "what the CPU was doing" accounts for most idle
time.

### Step 2: idle_summary

Scan by `cumulative_pct` and total time. The first rows after the aggregate row
usually show where to invest effort.

- If `LAUNCH_ANOMALY` dominates, the issue is along the GPU dispatch path (slow
  pickup of queued work, or unusually large launch-to-start gaps), not
  necessarily slow Python on the host.
- If `CPU_DOMINATED` dominates, framework or operator overhead on the CPU is the
  main story; the `dominant_op` column narrows the target.
- `RUNTIME_DOMINATED` points at runtime API activity (allocation,
  synchronization, launch stalls); inspect the detail fields for the specific
  call or sub-type.

### Step 3: idle_intervals

Use this sheet to inspect individual gaps. `idle_id` ties each interval to
annotations in the augmented Perfetto trace (labels of the form `idle#N`). The
UID columns (`following_gpu_uid`, `following_launch_uid`, `sync_event_uid`) link
rows to events in the [Trace2Tree](../conceptual/trace2tree.md) model through
`tree.events_by_uid[uid]`, so you can recover call stacks and surrounding context
for deep dives.

## Classification reference

The following table summarizes each `cpu_during_gap` category and the typical
response.

| Category | Meaning | Typical action |
|----------|---------|----------------|
| `LAUNCH_ANOMALY` | GPU slow to pick up the next kernel relative to expectations. | Consider CUDA/HIP graphs or other batching of dispatch; investigate driver or platform behavior for persistent micro-gaps. |
| `LAUNCH_OVERHEAD_ONLY` | Gap explained by normal launch latency; the CPU submitted work promptly. | Usually not actionable beyond accepting inherent overhead. |
| `RUNTIME_DOMINATED` | A runtime API call (for example, malloc, sync, or a launch stall) occupies a large share of the gap. | Depends on sub-type; reduce synchronization, allocator churn, or the specific API hotspot. |
| `CPU_DOMINATED` | CPU-side framework or operator work dominated the gap. | Optimize or fuse the dominant op; reduce host-side overhead. |
| `CPU_UNTRACED` | CPU activity during the gap isn't well represented in the trace. | Re-profile with Python function tracing or `with_stack=True` so self-time attribution is reliable. |

For thresholds, column definitions, and annotation track details, see the
[idle time sheet reference](../reference/idle-time-columns.md).

## Example workflow

Suppose about 30% of idle time is attributed to `LAUNCH_ANOMALY`:

1. Open `idle_overview`: confirm that a substantial fraction of total idle falls
   under `LAUNCH_ANOMALY`, and that overall idle is worth fixing given
   `gpu_utilization_pct`.
2. Open `idle_summary`: check whether most of that time is grouped under
   `prequeued` intervals. That means the host launched the kernel in time, but
   the GPU still showed a larger-than-expected gap before execution started.
3. Open `idle_intervals`: note typical `duration_us` (for example, many gaps of
   roughly 5–15 microseconds between kernels).
4. Conclude: the pattern is consistent with GPU-side dispatch or scheduling
   overhead rather than a single slow CPU op. Mitigations include CUDA/HIP graphs
   to amortize launch and submission, and platform or driver follow-up if the gap
   is uniform and limits throughput.

## Performance overhead

Enabling idle time analysis adds classification and DataFrame construction on
top of the normal trace load. The following table shows measurements on
representative PyTorch traces.

| Trace | Size | Intervals | Load | Classify + DataFrames | Overhead |
|-------|------|-----------|------|-----------------------|----------|
| ResNet-26t (single GPU) | 1.8 MB | 154 | 0.09 s | 0.81 s | ~9x (dominated by first-time NumPy initialization) |
| ResNet (training) | 5.3 MB | 548 | 0.47 s | 0.20 s | ~43% |

The first-run overhead includes one-time NumPy and pandas import costs. On
subsequent calls within the same process, the overhead is closer to 20–40% of
trace load time for medium traces.

For very large traces (more than 100 MB, thousands of idle intervals),
classification can take several minutes. Raise the noise threshold
(`--micro_idle_thresh_us`, or `micro_thresh_us` in the API) so more short gaps
are skipped, or run the analysis in a separate process with a timeout.

## Related topics

- [Idle time sheet reference](../reference/idle-time-columns.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Trace2Tree data model](../conceptual/trace2tree.md)
- [Analyze traces with the SDK](./sdk-analysis.md)
