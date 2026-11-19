<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Idle time sheet reference
```{meta}
:description: Column reference for the TraceLens idle_overview, idle_groups, and idle_intervals sheets, plus the idle gap parts, labels, drain_type rules, and thresholds.
:keywords: TraceLens, GPU idle time, idle_overview, idle_groups, idle_intervals, drain_type, launch latency, column reference
```

This topic describes the `idle_overview`, `idle_groups`, and `idle_intervals`
sheets, which `--enable_idle_analysis` adds to the PyTorch performance report,
and the rules that classify each gap. For how to run the analysis and read the
results, see [Analyze GPU idle time](../how-to/analyze-idle-time.md).

## idle_overview

`idle_overview` totals idle time by cause. The `section` column groups its rows:

- **`total`:** `gpu_utilization_pct`, followed by the `ALL idle`, `noise`, and
  `macro` totals. `ALL idle` equals the `idle_time` row of the `gpu_timeline`
  sheet. `gpu_utilization_pct` needs the GPU busy time, which the perf report and
  the `IdleTimeAnalyser` API always provide.
- **`label`:** one row per label over macro gaps, largest first.
- **`sub_label`:** one row per sub-label, after its label.
- **`drain`:** `starved`, then one `sync_drain: <sync call>` row per sync call,
  largest first. With `--include_call_stack`, the sync rows are split by call
  site: `sync_drain: <sync call> @ <sync_entry_point>`.
- **`threshold`:** the `IdleThresholds` values used, as `name = value`.

The `macro`, `label`, `sub_label`, and `drain` rows have one `*_ms` column per
[part](#parts), summed over the row's gaps. They also have `sync_drain_ms`,
`starved_ms`, and `stream_switch_ms`, which split the row's time by `drain_type`
and `stream_switch`.

| Column | Type | Description |
|--------|------|-------------|
| `section` | str | `total`, `label`, `sub_label`, `drain`, or `threshold`. |
| `row` | str | Row name, for example `ALL idle`, `CPU_OP_DOMINATED`, `sync_drain: hipFree`, or `noise_us = 5.0`. |
| `gaps` | int or null | Number of gaps. Null for `threshold` rows. |
| `total_ms` | float | Idle time (ms). |
| `pct_of_idle` | float | Share of all idle time, including noise (%). |
| `pct_of_trace` | float or null | Share of GPU busy time plus all idle time (%). On the `gpu_utilization_pct` row, the utilization itself. |
| `sync_drain_ms` | float or null | Time in `sync_drain` gaps. |
| `sync_gain_ms` | float or null | Sync rows of the `drain` section only: `sync_gain_us` summed over the row's gaps. The most that removing or deferring the sync could save. |
| `starved_ms` | float or null | Time in `starved` gaps. `sync_drain_ms` plus `starved_ms` equals `total_ms`. |
| `stream_switch_ms` | float or null | Time in gaps where `stream_switch` is true. |
| `cpu_op_ms` ... `unattributed_ms` | float or null | Each part summed over the row's gaps (ms). They add up to `total_ms`. Null on the `ALL idle` and `noise` rows. |

## idle_groups

`idle_groups` groups macro gaps by `label`, `sub_label`, `drain_type`, and
`covering_op`, sorted by `total_ms`, largest first. `covering_op` is the group's
code location. Groups under 1% of idle time are folded into one
`other (N groups)` row per label, sub-label, and `drain_type`. The three largest
groups of each label and sub-label are never folded.

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | int | Row number. The `group_id` column of `idle_intervals` refers to it. |
| `label` | str | Label of the group's gaps. |
| `sub_label` | str or null | Sub-label, if any. |
| `drain_type` | str | `sync_drain` or `starved`. |
| `covering_op` | str | The gaps' `covering_op`, with memory addresses removed. `(no CPU op)` if no CPU op ran on the launching thread; `(no launch)` for `UNATTRIBUTED`; `other (N groups)` for folded rows. |
| `top_runtime` | str or null | Runtime call with the most time in the group's gaps; for `other_thread` groups, the `other_thread_top_runtime` with the most time. Null if that call holds less than 10% of the group's idle time. |
| `top_launch` | str or null | The `following_launch_name` with the most idle time before it, for example `hipGraphLaunch`. |
| `top_next_gpu_op` | str or null | The `following_gpu_event` with the most idle time before it. |
| `gaps` | int | Number of gaps. |
| `total_ms` | float | Sum of gap durations (ms). |
| `mean_us`, `median_us`, `p90_us`, `max_us` | float | Gap duration statistics (µs). |
| `pct_of_idle` | float | Share of all idle time (%). Noise gaps aren't grouped, so the column sums to the `pct_of_idle` of the `macro` row, not to 100. |
| `cumulative_pct` | float | Running sum of `pct_of_idle` down the sheet. |
| `worst_idle_id` | int | `idle_id` of the group's longest gap. |
| `folded_top_ops` | str or null | `other` rows only: the `covering_op` of the three largest folded groups. |
| `cpu_op_ms` ... `unattributed_ms` | float | Each part summed over the group (ms). |

## idle_intervals

`idle_intervals` has one row per macro gap, sorted by duration, longest first.

| Column | Type | Description |
|--------|------|-------------|
| `idle_id` | int | Gap number, in time order. Matches the `idle#N` labels in the augmented trace. |
| `group_id` | int | The gap's `idle_groups` row. |
| `label` | str | See [Labels](#labels). |
| `sub_label` | str or null | See [Labels](#labels). |
| `case` | str | `unqueued`, `queued`, or `no_launch`. See [Parts](#parts). |
| `drain_type` | str | `sync_drain` or `starved`. See [drain_type](#drain_type). |
| `stream_switch` | bool or null | True if the GPU ops before and after the gap ran on different streams. Null if either op wasn't found. |
| `start_us`, `end_us`, `duration_us` | float | Gap start, end, and length (µs). |
| `cpu_op_us` ... `unattributed_us` | float | The gap's parts. They add up to `duration_us`. |
| `detail` | str | The nonzero parts, largest first. For queued and no-launch gaps, a short explanation. |
| `covering_op` | str or null | The deepest CPU op on the launching thread that spans at least half the gap; if none does, the CPU op with the most time in the gap. Runtime calls and annotations don't count. Set for every gap with a launch, including queued gaps. Null if no CPU op ran during the gap. |
| `top_op` | str or null | Op with the most exclusive time in the CPU window. |
| `top_runtime` | str or null | Runtime call with the most exclusive time in the CPU window. |
| `launch_parent_op` | str or null | Nearest `cpu_op` that encloses the launch call. |
| `following_launch_name` | str or null | Launch call of the next GPU op, for example `hipLaunchKernel`. |
| `launch_tid` | int or null | Thread of the launch call. |
| `launch_to_start_us` | float or null | Next GPU op start minus launch call return. Negative when the op started before the call returned, for example the transfer of a blocking copy. |
| `launch_lead_us` | float or null | Gap start minus launch call return. Positive on queued gaps: how long before the gap the next op was launched. Negative when the launch returned during the gap; a value close to `-duration_us` means it returned at the end of the gap. |
| `launch_call_x_ref` | float or null | CPU duration of the whole launch call, including any part before the gap, divided by `launch_call_ref_us` (50 µs). Above 1, the call was slow: usually a graph launch, otherwise a descheduled or contended thread. Kernel and graph launches only; null for copies and memsets, which can block on the GPU. |
| `other_thread_top_runtime` | str or null | Runtime call with the most time during the gap on any other thread. |
| `other_thread_runtime_us` | float or null | Time of that call during the gap (µs). |
| `preceding_gpu_event`, `following_gpu_event` | str or null | GPU ops that end and start the gap. |
| `preceding_stream`, `following_stream` | int or null | Their streams. |
| `sync_event_name`, `sync_event_dur`, `sync_event_correlation` | str, float, int or null | Sync call running at the gap start on the launching thread: its name, duration (µs), and correlation or External ID. The call can start while the GPU is still busy, so `sync_event_dur` can be longer than `duration_us`. Also set on `starved` gaps that had a sync call that didn't drain the queue. |
| `sync_gain_us` | float or null | `sync_drain` gaps only: `min(sync_event_dur, duration_us)`, the most that removing or deferring the sync could save on this gap. Counts only the sync at the gap start. If more blocking calls follow in the gap, for example several `hipFree` calls, removing all of them can save more. |
| `preceding_gpu_uid`, `following_gpu_uid`, `following_launch_uid`, `sync_event_uid` | int or null | TraceLens UIDs, for `tree.events_by_uid[uid]`. |
| `sync_entry_point` | str or null | With `--include_call_stack`: the user code frame that issued the sync call, found the same way as `entry_point` in `unified_perf_summary`. `Not found` if the stack has no Python frame. Null if the gap has no sync call. |
| `sync_call_stack_full` | str or null | With `--include_call_stack`: call stack from the root to the sync call, in the `call_stack_full` format of `unified_perf_summary`, for example `['train.py(41): main', 'aten::item', 'aten::_local_scalar_dense', 'hipMemcpyWithStream']`. Python frames appear only if the trace has them (`with_stack=True`) and the tree was built with them. |
| `covering_call_stack_full` | str or null | With `--include_call_stack`: call stack from the root to `covering_op`, in the same format. Shows the code behind a generic `covering_op` such as `aten::copy_`. Null if the gap has no `covering_op`. |

## Classification rules

### Parts

Parts describe only the thread that launched the GPU op at the end of the gap.
`launch_end` is the time that launch call returned.

- **Unqueued** (`launch_end + launch_latency_us > gap start`): the time from the
  gap start to the launch call is split by exclusive time into:
  - `cpu_op_us`: CPU ops and Python functions.
  - `cpu_runtime_us`: runtime calls.
  - `cpu_untraced_us`: no traced event, or only a `user_annotation` range such as
    `ProfilerStep#N`.

  The part of the launch call inside the gap is `launch_call_us`, or
  `cpu_runtime_us` for a blocking call such as `hipMemcpyWithStream`. The time
  from the later of `launch_end` and the gap start to the op start is
  `submit_latency_us`.
- **Queued** (`launch_end + launch_latency_us <= gap start`): the whole gap is
  `gpu_wait_us`.
- **No launch:** the trace has no launch call for the next op. The whole gap is
  `unattributed_us`.

### Labels

| Label | Sub-label | Rule |
|-------|-----------|------|
| `CPU_OP_DOMINATED` | | Unqueued, and `cpu_op_us` is the largest part. |
| `RUNTIME_DOMINATED` | | Unqueued, and `cpu_runtime_us` is the largest part. |
| `RUNTIME_DOMINATED` | `other_thread` | Unqueued, `cpu_runtime_us` isn't the largest part, and another thread spent more than `other_thread_frac` of the gap in one runtime call. |
| `UNTRACED_DOMINATED` | | Unqueued, and `cpu_untraced_us` is the largest part. |
| `LAUNCH_DOMINATED` | | Unqueued, `launch_call_us + submit_latency_us` is the largest part, and `launch_to_start_us <= launch_latency_us`. |
| `UNEXPLAINED_GPU_WAIT` | `queued` | Queued. The next op was launched before the gap, so the whole gap is on the GPU side. |
| `UNEXPLAINED_GPU_WAIT` | `unqueued` | As `LAUNCH_DOMINATED`, but `launch_to_start_us > launch_latency_us`: the op was launched during the gap and took longer than normal to start. This is the only label not chosen by largest part. |
| `UNATTRIBUTED` | | No launch call in the trace for the next GPU op. |

A change of stream across the gap is recorded in the `stream_switch` column, not
as a sub-label, because it can occur with any label.

### drain_type

`drain_type` records whether a sync call caused the gap. It's set independently
of the label.

- **`sync_drain`:** the launching thread was blocked in a sync call while the GPU
  finished its queued work. All of the following are true:
  - The sync call is on the thread that launched the GPU op after the gap.
  - The call was running at the gap start, or ended less than 2 µs before it.
  - The call returned before that launch call started.
  - The call started more than `drain_overlap_us` before the GPU finished its
    last op. A copy's own transfer doesn't count as that op; the last op before
    the transfer is used instead.
- **`starved`:** every other gap. The GPU finished its work before the CPU
  launched the next op. This includes a copy issued after the GPU went idle: the
  host is blocked in the copy call, but no GPU work was queued behind it.

For example, a `.cpu()` call starts at t = 0 µs while kernels run until
t = 700 µs. The copy's transfer runs from 700 to 705 µs, and the call returns at
710 µs. The gap after the transfer is `sync_drain`. If the kernels had finished
at t = −50 µs, the same call would make the gap `starved`.

The following runtime calls count as sync calls:

| Calls | Why they block |
|-------|----------------|
| `hipDeviceSynchronize`, `hipStreamSynchronize`, `hipEventSynchronize`, and the `cuda*` equivalents | Explicit syncs. |
| `hipFree`, `hipHostFree`, `hipFreeHost`, `cudaFree`, `cudaFreeHost` | Documented to synchronize the device implicitly. Memory from `hipMallocAsync` or `cudaMallocAsync`, for example with PyTorch's `backend:cudaMallocAsync` allocator, is freed without a sync; a long free of such memory that spans the end of the GPU work is still counted. |
| `hipMemcpy`, `hipMemcpyWithStream`, `hipMemcpyAsync`, `hipMemcpy2DAsync`, `hipMemcpyDtoH`, and the `cuda*` equivalents | Block the host when host memory is pageable, in either direction, including the `Async` forms. |

A listed call counts only if it meets the timing conditions above, so a copy that
returns immediately doesn't count.

Notes:

- **Copies differ by vendor:** `.item()`, `.cpu()`, and `.tolist()` appear as
  `hipMemcpyWithStream` on ROCm, and usually as `cudaMemcpyAsync` to pageable
  memory on CUDA. Copying to pinned host memory avoids the block.
- **`hipFree` inside an op usually means allocator retries:** when an allocation
  fails, the PyTorch caching allocator frees cached blocks and retries. The cause
  is memory pressure, not the op that allocated. Check `num_alloc_retries` in
  `torch.cuda.memory_stats()`.
- **Some syncs aren't detected:** a sync on another thread that the launching
  thread waits for, or a sync inside a library that isn't a listed call, makes the
  gap `starved`.

### Thresholds

Thresholds are fields of `IdleThresholds`, passed to `IdleTimeAnalyser`.

| Field | Default | Description |
|-------|---------|-------------|
| `noise_us` | `5.0` | Gaps shorter than this are noise: counted in the totals but not classified. Set with `--micro_idle_thresh_us` (perf report) or `--micro-thresh` (standalone). |
| `launch_latency_us` | `15.0` | Normal time from launch call return to op start. Separates queued from unqueued gaps, and `LAUNCH_DOMINATED` from `UNEXPLAINED_GPU_WAIT/unqueued`. Set with `--idle_launch_latency_us` (perf report) or `--launch-latency` (standalone). |
| `launch_latency_rel` | `None` | If set, the launch latency limit is `max(launch_latency_us, launch_latency_rel * median launch_to_start_us)` over ops launched during gaps, so the limit adapts to the platform. |
| `other_thread_frac` | `0.5` | Share of the gap that another thread's runtime call must exceed for `other_thread`. |
| `drain_overlap_us` | `5.0` | Minimum time from the start of a sync call to the end of the GPU's last queued op for the gap to be `sync_drain`. |
| `launch_call_ref_us` | `50.0` | Reference CPU duration of one launch call, used for `launch_call_x_ref`. Kernel launches take a few µs; graph launches normally take much longer. |

## Augmented trace annotations

The augmented trace, written by `--enable_augmented_trace` or the standalone
`classify_idle_time` module, adds three annotation tracks to the GPU process in
Perfetto:

1. **Idle: Noise/Macro:** labels each gap `noise` or `idle#N`.
2. **Idle: Drain Type:** labels each macro gap `sync_drain` with the sync call,
   or `starved`.
3. **Idle: Label:** labels each macro gap with its label and sub-label. The slice
   arguments hold the case, detail, and parts.

## Related topics

- [Analyze GPU idle time](../how-to/analyze-idle-time.md)
- [Performance report column reference](./perf-report-columns.md)
- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
