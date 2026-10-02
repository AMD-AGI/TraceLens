<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Idle time sheet reference
```{meta}
:description: Column reference for the TraceLens idle_overview, idle_summary, and idle_intervals sheets, plus the drain_type and cpu_during_gap classification rules.
:keywords: TraceLens, GPU idle time, idle_overview, idle_summary, idle_intervals, drain_type, cpu_during_gap, column reference
```

These sheets are added to the PyTorch performance report when idle time analysis
is enabled with `--enable_idle_analysis`. For a conceptual introduction and
usage examples, see [Analyze GPU idle time](../how-to/analyze-idle-time.md).

## idle_overview

High-level summary grouped by `(drain_type, cpu_during_gap)`. It's typically
5–8 rows, giving an immediate picture of where idle time lives, and is sorted by
`total_time_ms` in descending order. The following table describes all columns.

| Column | Type | Description |
|--------|------|-------------|
| `drain_type` | str | `sync_drain`, `starved`, or `ALL` (totals row). |
| `cpu_during_gap` | str | `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, `CPU_UNTRACED`, or `ALL`. |
| `count` | int | Number of idle intervals in this group. |
| `total_time_ms` | float | Sum of all interval durations (milliseconds). |
| `pct_of_idle` | float | This group's share of total macro idle time (%). |
| `pct_of_trace` | float | This group's share of GPU busy time plus macro idle time (%). Present in the perf report output. |
| `gpu_utilization_pct` | float | GPU busy time as a share of GPU busy time plus macro idle time (%); same value on every row. Present in the perf report output. |
| `mean_us` | float | Mean interval duration (microseconds). |
| `median_us` | float | Median interval duration. |
| `min_us` | float | Shortest interval. |
| `max_us` | float | Longest interval. |

The first row (`ALL`) holds totals across all macro intervals.

## idle_summary

Grouped summary of macro idle intervals (those at or above the noise threshold,
default 5 µs). The following table describes all columns.

| Column | Type | Description |
|--------|------|-------------|
| `drain_type` | str | Why the GPU queue became empty. `sync_drain` means a sync call blocked the CPU and drained the queue; `starved` means the CPU couldn't submit work fast enough. `ALL` for the totals row, `—` for noise. |
| `cpu_during_gap` | str | What the CPU was doing during the idle gap. One of `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, `CPU_UNTRACED`, `ALL`, or `noise`. |
| `dominant_op` | str | Grouping key within a category. For `CPU_DOMINATED` and `CPU_UNTRACED`: the `cpu_op` or `python_function` with the highest self-time. For `RUNTIME_DOMINATED`: sub-type and function name (for example, `SYNC_CALL: hipDeviceSynchronize`). For `LAUNCH_ANOMALY` and `LAUNCH_OVERHEAD_ONLY`: `prequeued` or `launched_during_gap`. |
| `count` | int | Number of idle intervals in this group. |
| `total_time_ms` | float | Sum of all interval durations in this group (milliseconds). |
| `pct_of_idle` | float | This group's share of total macro idle time (%). |
| `cumulative_pct` | float | Running sum of `pct_of_idle` across groups (sorted by `total_time_ms` descending). Useful for finding the top groups that cover, for example, 80% of idle time. |
| `mean_us` | float | Mean interval duration (microseconds). |
| `median_us` | float | Median interval duration. |
| `std_us` | float | Standard deviation of interval durations. |
| `min_us` | float | Shortest interval in the group. |
| `max_us` | float | Longest interval in the group. |
| `idle_ids` | str | Comma-separated `idle_id` values in this group. Cross-reference with `idle_intervals` and the Perfetto annotations. |

The first row (`ALL`) holds totals across all macro intervals. The second row
(`noise`) holds stats for intervals below the noise threshold; its `pct_of_idle`
is null.

## idle_intervals

Per-interval detail for every macro idle interval, sorted by duration in
descending order. The following table describes all columns.

| Column | Type | Description |
|--------|------|-------------|
| `idle_id` | int | Unique identifier. Matches Perfetto annotation labels (`idle#N`). Non-negative for macro intervals, negative for noise. |
| `group` | str | Human-readable group key: `drain_type \| cpu_during_gap \| dominant_op`. |
| `start_us` | float | Timestamp (microseconds) where the GPU became idle. |
| `end_us` | float | Timestamp where the GPU resumed work. |
| `duration_us` | float | `end_us - start_us`. |
| `drain_type` | str | `sync_drain` or `starved`. |
| `sync_type` | str or null | The sync mechanism of the causal sync event (`DEVICE_SYNC`, `STREAM_SYNC`, `EVENT_SYNC`, `D2H_COPY`, `H2D_COPY`). Null if no causal sync was found. |
| `sync_event_name` | str or null | Runtime API name of the sync call (for example, `hipMemcpyWithStream` or `hipStreamSynchronize`). |
| `sync_event_correlation` | int or null | Correlation or External ID linking to the GPU-side event. |
| `sync_event_dur` | float or null | Duration of the sync runtime event (microseconds). |
| `cpu_during_gap` | str | `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, or `CPU_UNTRACED`. |
| `cpu_during_gap_detail` | str or null | For `CPU_DOMINATED`: top three ops by self-time with percentages. For `CPU_UNTRACED`: self-time coverage. For `RUNTIME_DOMINATED`: sub-type and function name. For `LAUNCH_ANOMALY`: launch timing. For `LAUNCH_OVERHEAD_ONLY`: application overhead and `launch_to_exec`. |
| `dominant_op` | str or null | The `cpu_op` or `python_function` event with the highest self-time during this gap. Null for `RUNTIME_DOMINATED` and the launch categories. |
| `preceding_gpu_event` | str or null | Name of the GPU event that ended just before this idle gap. |
| `following_gpu_event` | str or null | Name of the GPU event that started just after this idle gap. |
| `following_launch_name` | str or null | Runtime launch call for the following GPU kernel (for example, `hipLaunchKernel`). |
| `following_gpu_uid` | int or null | TraceLens UID of the following GPU event. Use with `tree.events_by_uid[uid]` for deep analysis. |
| `following_launch_uid` | int or null | TraceLens UID of the launch runtime event for the following kernel. |
| `sync_event_uid` | int or null | TraceLens UID of the causal sync event. |
| `launch_to_exec_us` | float or null | Time from the end of the launch call to the GPU kernel start. |
| `kernel_prequeued` | bool or null | True if the following kernel's launch completed well before this idle gap started (see below). |

## Classification rules

### drain_type

Answers: why was the GPU queue empty?

- **`sync_drain`:** A synchronization event (device-to-host copy, or
  device/stream/event sync) blocked the CPU, so it stopped submitting work while
  the GPU drained its queue. The sync is *causal*: it ended before the next
  kernel's launch began, and it started more than 5 µs before the preceding GPU
  event finished.
- **`starved`:** The CPU couldn't submit work fast enough. This includes cases
  where a sync was present but the queue was already empty (the sync was
  redundant from a queue perspective).

### cpu_during_gap

Answers: what was the CPU doing during the idle gap? Categories are checked in
the order listed.

- **`LAUNCH_ANOMALY`:** The GPU was slow to pick up the next kernel. For
  kernels launched during the gap: `launch_to_exec > 10 µs` (typical is 7–8 µs)
  *and* `launch_to_exec` explains more than 25% of the gap. For prequeued
  kernels: gap duration above 5 µs, since an already-queued kernel should start
  almost immediately. Indicates GPU scheduler or dispatch overhead.
- **`LAUNCH_OVERHEAD_ONLY`:** The gap is almost entirely explained by kernel
  launch latency. The CPU dispatched promptly; application overhead
  (`duration - launch_to_exec`) is under 4 µs. Not actionable.
- **`RUNTIME_DOMINATED`:** A runtime call category (allocation/free, launch
  stall, sync, other runtime) occupied at least 25% of the gap duration. The
  detail shows the sub-type and specific function (for example,
  `MEMORY_ALLOC: hipMalloc`).
- **`CPU_DOMINATED`:** Framework or operator dispatch, or Python overhead,
  dominated the gap. The `dominant_op` column shows the most specific operation
  by self-time. Self-time coverage is at least 20% of the gap.
- **`CPU_UNTRACED`:** CPU ops overlap the gap but their self-time covers less
  than 20% of it, or no CPU ops overlap at all. The actual work is untraced
  Python or framework code. Re-profile with `with_stack=True` or Python function
  tracing enabled.

### Self-time for dominant_op

The dominant op is selected by *self-time* (exclusive time), not inclusive time.
Self-time subtracts time covered by child events, so `aten::miopen_convolution`
(the backend call) is reported instead of its wrapper `aten::conv2d`, because
the wrapper's self-time only includes dispatch overhead.

Both `cpu_op` and `python_function` events are included. Python-level frames
(for example, `model.py(234): forward`) appear when they have genuine exclusive
time, such as Python setup work before calling into C++ ops.

### kernel_prequeued

A kernel is *prequeued* if `launch_end + 8 µs < gap_start`, where 8 µs is the
typical launch latency. That means the CPU returned from the launch call early
enough for the kernel to reach the GPU hardware queue before the gap began.

## Augmented trace annotations

The augmented trace (from `--enable_augmented_trace` or the standalone
`classify_idle_time` module) adds three annotation tracks to the GPU process,
visible in Perfetto:

1. **Idle: Noise/Macro:** labels each gap as `noise` or `idle#N`.
2. **Idle: Drain Type:** labels each macro gap with `sync_drain` and the sync
   type, or `starved`.
3. **Idle: CPU During Gap:** labels each macro gap with its `cpu_during_gap`
   classification plus detail.

## Related topics

- [Analyze GPU idle time](../how-to/analyze-idle-time.md)
- [Performance report column reference](./perf-report-columns.md)
- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
