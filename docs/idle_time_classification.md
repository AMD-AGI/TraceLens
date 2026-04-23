# Idle Time Classification Sheets -- Column Documentation

These sheets are added to the `generate_perf_report` Excel output when the idle
time extension is enabled (`--extension_file idle_time_extension.py`).

## Sheet: `idle_overview`

High-level summary grouped by `(drain_type, cpu_during_gap)`. Typically 5-8 rows,
giving an immediate picture of where idle time lives. Sorted by `total_time_ms` descending.

| Column | Type | Description |
|--------|------|-------------|
| `drain_type` | str | `sync_drain`, `starved`, or `ALL` (totals row). |
| `cpu_during_gap` | str | `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, `CPU_UNTRACED`, or `ALL`. |
| `count` | int | Number of idle intervals in this group. |
| `total_time_ms` | float | Sum of all interval durations (milliseconds). |
| `pct_of_idle` | float | This group's share of total macro idle time (%). |
| `mean_us` | float | Mean interval duration (microseconds). |
| `median_us` | float | Median interval duration. |
| `min_us` | float | Shortest interval. |
| `max_us` | float | Longest interval. |

### Special rows

- **ALL row** (first): totals across all macro intervals.

---

## Sheet: `idle_summary`

Grouped summary of macro idle intervals (those above the noise threshold, default 5µs).

| Column | Type | Description |
|--------|------|-------------|
| `drain_type` | str | Why the GPU queue became empty. `sync_drain` = a sync call blocked the CPU and drained the queue. `starved` = CPU couldn't submit work fast enough. `ALL` for the totals row, `—` for noise. |
| `cpu_during_gap` | str | What the CPU was doing during the idle gap. One of `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, `CPU_UNTRACED`, `ALL`, or `noise`. |
| `dominant_op` | str | For `CPU_DOMINATED`/`CPU_UNTRACED`: the cpu_op or python_function with the highest self-time. For `RUNTIME_DOMINATED`: sub-type + function name (e.g. `SYNC_CALL: hipDeviceSynchronize`). For `LAUNCH_ANOMALY`/`LAUNCH_OVERHEAD_ONLY`: `prequeued` or `launched_during_gap`. |
| `count` | int | Number of idle intervals in this group. |
| `total_time_ms` | float | Sum of all interval durations in this group (milliseconds). |
| `pct_of_idle` | float | This group's share of total macro idle time (%). |
| `cumulative_pct` | float | Running sum of `pct_of_idle` across groups (sorted by total_time_ms desc). Useful for identifying the top-N groups that cover e.g. 80% of idle time. |
| `mean_us` | float | Mean interval duration (microseconds). |
| `median_us` | float | Median interval duration. |
| `std_us` | float | Standard deviation of interval durations. |
| `min_us` | float | Shortest interval in the group. |
| `max_us` | float | Longest interval in the group. |
| `idle_ids` | str | Comma-separated list of `idle_id` values belonging to this group. Cross-reference with `idle_intervals` sheet and Perfetto annotations. |

### Special rows

- **ALL row** (first): totals across all macro intervals.
- **noise row** (second): stats for intervals below the noise threshold; `pct_of_idle` is null.

---

## Sheet: `idle_intervals`

Per-interval detail for every macro idle interval, sorted by duration descending.

| Column | Type | Description |
|--------|------|-------------|
| `idle_id` | int | Unique identifier. Matches Perfetto annotation labels (`idle#N`). Non-negative for macro, negative for noise. |
| `group` | str | Human-readable group key: `drain_type \| cpu_during_gap \| dominant_op`. |
| `start_us` | float | Timestamp (microseconds) where GPU became idle. |
| `end_us` | float | Timestamp where GPU resumed work. |
| `duration_us` | float | `end_us - start_us`. |
| `drain_type` | str | `sync_drain` or `starved`. |
| `sync_type` | str or null | If sync_drain: the sync mechanism (`DEVICE_SYNC`, `STREAM_SYNC`, `EVENT_SYNC`, `D2H_COPY`, `H2D_COPY`). Null if starved. |
| `sync_event_name` | str or null | Runtime API name of the sync call (e.g. `hipMemcpyWithStream`, `hipStreamSynchronize`). |
| `sync_event_correlation` | int or null | Correlation / External ID linking to the GPU-side event. |
| `sync_event_dur` | float or null | Duration of the sync runtime event (microseconds). |
| `cpu_during_gap` | str | `LAUNCH_ANOMALY`, `LAUNCH_OVERHEAD_ONLY`, `RUNTIME_DOMINATED`, `CPU_DOMINATED`, or `CPU_UNTRACED`. |
| `cpu_during_gap_detail` | str or null | Details. For `CPU_DOMINATED`: top-3 ops by self-time with percentages. For `CPU_UNTRACED`: self-time coverage %. For `RUNTIME_DOMINATED`: sub-type + function name. For `LAUNCH_ANOMALY`: launch timing. For `LAUNCH_OVERHEAD_ONLY`: app overhead and launch_to_exec. |
| `dominant_op` | str or null | The cpu_op or python_function event name with the highest self-time during this gap. Null for `RUNTIME_DOMINATED` and `LAUNCH_ANOMALY`. |
| `preceding_gpu_event` | str or null | Name of the GPU event that ended just before this idle gap. |
| `following_gpu_event` | str or null | Name of the GPU event that started just after this idle gap. |
| `following_launch_name` | str or null | Runtime launch call for the following GPU kernel (e.g. `hipLaunchKernel`). |
| `following_gpu_uid` | int or null | TraceLens UID of the following GPU event. Use with `tree.events_by_uid[uid]` for deep analysis. |
| `following_launch_uid` | int or null | TraceLens UID of the launch runtime event for the following kernel. |
| `sync_event_uid` | int or null | TraceLens UID of the causal sync event (if `sync_drain`). |
| `launch_to_exec_us` | float or null | Time from launch call end to GPU kernel start. Can be negative for pre-queued kernels. |
| `kernel_prequeued` | bool or null | True if the following kernel's launch completed well before this idle gap started (launch_end + typical_latency < gap_start). |

---

## Classification Axes

### drain_type

Answers: **why was the GPU queue empty?**

- **`sync_drain`**: A synchronization event (D2H memcpy, device/stream/event sync) blocked the CPU, causing it to stop submitting work while the GPU drained its queue. The sync is *causal* -- it ended before the next kernel's launch.
- **`starved`**: The CPU couldn't submit work fast enough. This includes cases where a sync was present but the queue was already empty (the sync was "redundant" from a queue perspective).

### cpu_during_gap

Answers: **what was the CPU doing during the idle gap?**

- **`LAUNCH_ANOMALY`**: The GPU was slow to pick up the next kernel. For non-prequeued kernels: `launch_to_exec > 10µs` (typical is 7-8µs). For prequeued kernels: `gap_duration > 5µs` (kernel was already queued, should start near-immediately). Indicates GPU scheduler/dispatch overhead.
- **`LAUNCH_OVERHEAD_ONLY`**: The gap is almost entirely explained by inherent kernel launch latency. The CPU dispatched promptly; application overhead is negligible (< max(2µs, 0.5 × typical_launch_latency)). Not actionable.
- **`RUNTIME_DOMINATED`**: A runtime call (malloc, free, launch stall, sync) occupied ≥25% of the gap duration. Detail shows the sub-type and specific function (e.g. `MEMORY_ALLOC: hipMalloc`).
- **`CPU_DOMINATED`**: Framework/operator dispatch or Python overhead dominated the gap. The `dominant_op` column shows the most specific operation (using self-time). Self-time coverage ≥ 20% of the gap.
- **`CPU_UNTRACED`**: CPU ops overlap the gap but their self-time covers < 20% of it, meaning the actual work is untraced Python/framework code. Signal to re-profile with `with_stack=True` or python functions enabled.

### Self-time for dominant_op

The dominant op is selected using **self-time** (exclusive time), not inclusive time. Self-time subtracts time covered by child events. This means `aten::miopen_convolution` (the actual backend call) is reported instead of its wrapper `aten::conv2d`, because the wrapper's self-time only includes dispatch overhead, not the child's duration.

Both `cpu_op` and `python_function` events are included. Python-level frames (e.g. `model.py(234): forward`) appear when they have genuine exclusive time (e.g. Python setup work before calling into C++ ops).

### kernel_prequeued

A kernel is considered **prequeued** if `launch_end + TYPICAL_LAUNCH_LATENCY < gap_start`, meaning the CPU returned from the launch call with enough time for the kernel to have been submitted to the GPU hardware queue before the gap began.

---

## Augmented Trace Annotations

When using `classify_idle_time.py` directly, three annotation tracks are added to the GPU process in the output trace (visible in Perfetto):

1. **Idle: Noise/Macro** -- labels each gap as `noise` or `idle#N`
2. **Idle: Drain Type** -- labels each macro gap with `sync_drain` or `starved`
3. **Idle: CPU During Gap** -- labels with the cpu_during_gap classification plus detail
