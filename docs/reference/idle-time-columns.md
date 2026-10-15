<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Idle time sheet reference
```{meta}
:description: Column reference for the TraceLens idle_overview, idle_groups, and idle_intervals sheets, plus the idle gap parts, labels, drain_type rules, and thresholds.
:keywords: TraceLens, GPU idle time, idle_overview, idle_groups, idle_intervals, drain_type, launch latency, column reference
```

These sheets are added to the PyTorch performance report when idle time analysis
is enabled with `--enable_idle_analysis`. For a conceptual introduction and
usage examples, see [Analyze GPU idle time](../how-to/analyze-idle-time.md).

## idle_overview

How much GPU time is idle, and why. Each row belongs to one `section`:

- **`total`:** `gpu_utilization_pct` (only when GPU busy time is known, which is
  always the case in the perf report and the `IdleTimeAnalyser` API), then
  `ALL idle`, `noise`, and `macro` totals. `ALL idle` equals the `idle_time` row
  of the `gpu_timeline` sheet.
- **`label`:** one row per label over macro gaps, largest first.
- **`sub_label`:** one row per sub-label, after its label.
- **`drain`:** `starved`, and one `sync_drain: <sync call>` row per sync call that
  drained the queue, largest first. With `--include_call_stack`, the rows are split
  by call site, `sync_drain: <sync call> @ <sync_entry_point>`, so each row is one
  line of code to change.
- **`threshold`:** the `IdleThresholds` values used, as `name = value`.

The `macro`, `label`, `sub_label`, and `drain` rows also carry one `*_ms` column
per part, so each row shows what its gaps were made of. A label names only the
largest part of each gap; the part columns show the rest. For example, a
`CPU_OP_DOMINATED` row with a large `launch_call_ms` means launch cost hides
inside gaps where CPU ops took the largest share. Features that are independent
of the label, `drain_type` and `stream_switch`, are split columns on every row
(`sync_drain_ms`, `starved_ms`, `stream_switch_ms`) rather than sub-labels, so
they don't add rows.

| Column | Type | Description |
|--------|------|-------------|
| `section` | str | `total`, `label`, `sub_label`, `drain`, or `threshold`. |
| `row` | str | Row name, for example `ALL idle`, `CPU_OP_DOMINATED`, `sync_drain: hipFree`, or `noise_us = 5.0`. |
| `gaps` | int or null | Number of gaps. Null for `threshold` rows. |
| `total_ms` | float | Idle time in milliseconds. |
| `pct_of_idle` | float | Share of all idle time, including noise (%). |
| `pct_of_trace` | float or null | Share of GPU busy time plus all idle time (%). For `gpu_utilization_pct`, the utilization itself. |
| `sync_drain_ms` | float or null | Time in `sync_drain` gaps. |
| `sync_gain_ms` | float or null | On `sync_drain` drain rows only: `sync_gain_us` summed over the row's gaps, the most that removing or deferring the sync could save. |
| `starved_ms` | float or null | Time in `starved` gaps. With `sync_drain_ms`, adds up to `total_ms`. |
| `stream_switch_ms` | float or null | Time in gaps where `stream_switch` is true. |
| `cpu_op_ms` ... `unattributed_ms` | float or null | Each part summed over the row's gaps (milliseconds). They add up to `total_ms`. Null for the `ALL idle` and `noise` rows. |

## idle_groups

Macro gaps grouped by `(label, sub_label, drain_type, covering_op)`, sorted by
`total_ms` in descending order. Groups under 1% of idle time are folded into one
`other (N groups)` row per label, sub-label, and `drain_type`, except the three
largest groups of each label and sub-label.

The group's code location is the same for every label: the CPU op that was
running on the launching thread during the gap. The `top_runtime`, `top_launch`,
and `top_next_gpu_op` columns add what the runtime and the GPU were doing.

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | int | Row number. The `group_id` column of `idle_intervals` points here. |
| `label` | str | Label of the gaps in this group. |
| `sub_label` | str or null | Sub-label, if any. |
| `drain_type` | str | `sync_drain` or `starved`. |
| `covering_op` | str | The `covering_op` of the gaps. `(no CPU op)` if no CPU op ran on the launching thread during the gap; `(no launch)` for `UNATTRIBUTED`; `other (N groups)` for folded rows. Memory addresses are removed. |
| `top_runtime` | str or null | Runtime call with the most time over the group's gaps: `top_runtime`, or `other_thread_top_runtime` for `other_thread`. Null if that call's time is under 10% of the group's idle time, so incidental calls aren't named. |
| `top_launch` | str or null | The `following_launch_name` with the most idle time before it, for example `hipGraphLaunch`. |
| `top_next_gpu_op` | str or null | The `following_gpu_event` with the most idle time before it. Most useful for `UNEXPLAINED_GPU_WAIT`. |
| `gaps` | int | Number of gaps. |
| `total_ms` | float | Sum of gap durations (milliseconds). |
| `mean_us`, `median_us`, `p90_us`, `max_us` | float | Gap duration statistics (microseconds). |
| `pct_of_idle` | float | Share of all idle time (%). Noise gaps aren't grouped, so the column sums to the `macro` row's `pct_of_idle`, not 100. |
| `cumulative_pct` | float | Running sum of `pct_of_idle` down the sheet. |
| `worst_idle_id` | int | `idle_id` of the longest gap in the group. |
| `folded_top_ops` | str or null | For `other` rows: the `covering_op` of the three largest folded groups. |
| `cpu_op_ms` ... `unattributed_ms` | float | Each part summed over the group (milliseconds). |

## idle_intervals

One row per macro gap, sorted by duration in descending order.

| Column | Type | Description |
|--------|------|-------------|
| `idle_id` | int | Gap identifier, in time order. Matches Perfetto annotation labels (`idle#N`). |
| `group_id` | int | The `idle_groups` row this gap belongs to. |
| `label` | str | See [Labels](#labels). |
| `sub_label` | str or null | See [Labels](#labels). |
| `case` | str | `unqueued`, `queued`, or `no_launch`. See [Parts](#parts). |
| `drain_type` | str | `sync_drain` or `starved`. See the `drain_type` rules below. |
| `stream_switch` | bool or null | The GPU ops before and after the gap ran on different streams. Null if either op wasn't found. |
| `start_us`, `end_us`, `duration_us` | float | Gap boundaries and length (microseconds). |
| `cpu_op_us` ... `unattributed_us` | float | The gap's parts. They add up to `duration_us`. |
| `detail` | str | The non-zero parts, largest first, or a short explanation for queued and no-launch gaps. |
| `covering_op` | str or null | Deepest CPU op (not a runtime call or annotation) on the launching thread spanning at least half of the gap; if none does, the CPU op with the most time in the gap. Filled for every gap with a launch, including queued ones. Null if no CPU op ran during the gap. |
| `top_op` | str or null | Op with the most exclusive time in the CPU window. |
| `top_runtime` | str or null | Runtime call with the most exclusive time in the CPU window. |
| `launch_parent_op` | str or null | Nearest `cpu_op` enclosing the launch call. |
| `following_launch_name` | str or null | Launch call for the next GPU op (for example, `hipLaunchKernel`). |
| `launch_tid` | int or null | Thread that made the launch call. |
| `launch_to_start_us` | float or null | Launch return to next GPU op start. Negative when the GPU op started before the launch call returned, for example a copy issued by a blocking memcpy call. |
| `launch_lead_us` | float or null | Launch return to gap start. Negative if the launch returned during the gap: `-launch_lead_us` is how far into the gap it returned, so a value close to `-duration_us` means the launch came at the end of the gap. Positive values (queued gaps) are how long before the gap the next op was already launched. |
| `launch_call_x_ref` | float or null | Full CPU duration of the next op's launch call divided by `launch_call_ref_us` (50 µs). Above 1, the launch call is slow: usually a graph launch, otherwise a descheduled or contended thread. Uses the whole call, including any part before the gap. Kernel and graph launches only; null for copies and memsets, which can block on the GPU. |
| `other_thread_top_runtime` | str or null | Runtime call with the most time during the gap on other threads. |
| `other_thread_runtime_us` | float or null | Time of that call during the gap. |
| `preceding_gpu_event`, `following_gpu_event` | str or null | GPU ops ending and starting the gap. |
| `preceding_stream`, `following_stream` | int or null | Their streams. |
| `sync_event_name`, `sync_event_dur`, `sync_event_correlation` | str, float, int or null | The sync call found at the gap start on the launching thread, its duration in µs, and its correlation or External ID. The call can start long before the gap (while the GPU was still busy), so `sync_event_dur` can be much longer than `duration_us`; only the part after the GPU went idle counts as idle time. Also filled for `starved` gaps when a sync was present but started too late to drain the queue. |
| `sync_gain_us` | float or null | `sync_drain` gaps only: `min(sync_event_dur, duration_us)`, the most that removing or deferring the sync could save on this gap. Counts only the sync at the gap start; if more blocking calls follow in the same gap (for example one `hipFree` after another), removing them all can save more. |
| `preceding_gpu_uid`, `following_gpu_uid`, `following_launch_uid`, `sync_event_uid` | int or null | TraceLens UIDs. Use with `tree.events_by_uid[uid]`. |
| `sync_entry_point` | str or null | With `--include_call_stack`: the user code frame that issued the sync call, found the same way as `entry_point` in `unified_perf_summary`. `Not found` if the stack has no Python frame; null if the gap has no sync. |
| `sync_call_stack_full` | str or null | With `--include_call_stack`: call stack from the root to the sync call, in the same format as `call_stack_full` in `unified_perf_summary`: a list of event names, for example `['train.py(41): main', 'aten::item', 'aten::_local_scalar_dense', 'hipMemcpyWithStream']`. Python frames appear only if the trace has them (`with_stack=True`) and the tree was built with them. |
| `covering_call_stack_full` | str or null | With `--include_call_stack`: call stack from the root to `covering_op`, in the same format. Use it to find the code behind a `covering_op` that's a generic op such as `aten::copy_`. Null if the gap has no `covering_op`. |

## Classification rules

### Parts

Parts use only the thread that launched the GPU op ending the gap. Let
`launch_end` be when that launch call returned:

- **Unqueued** (`launch_end + launch_latency_us > gap start`): the CPU window
  from gap start to the launch call is split by exclusive time into `cpu_op_us`
  (CPU ops and Python functions), `cpu_runtime_us` (runtime calls), and
  `cpu_untraced_us` (no traced event, or only a `user_annotation` range such as
  `ProfilerStep#N`). The launch call itself inside the gap is `launch_call_us`,
  or `cpu_runtime_us` for blocking calls such as `hipMemcpyWithStream`. Launch
  return (or gap start, if later) to op start is `submit_latency_us`.
- **Queued** (`launch_end + launch_latency_us <= gap start`): the whole gap is
  `gpu_wait_us`.
- **No launch:** the whole gap is `unattributed_us`.

### Labels

| Label | Sub-label | Rule |
|-------|-----------|------|
| `CPU_OP_DOMINATED` | | Unqueued; `cpu_op_us` is the largest part. |
| `RUNTIME_DOMINATED` | | Unqueued; `cpu_runtime_us` is the largest part. |
| `RUNTIME_DOMINATED` | `other_thread` | Unqueued; another thread was in one runtime call for more than `other_thread_frac` of the gap, and `cpu_runtime_us` isn't the largest part. |
| `UNTRACED_DOMINATED` | | Unqueued; `cpu_untraced_us` is the largest part. |
| `LAUNCH_DOMINATED` | | Unqueued; `launch_call_us + submit_latency_us` is the largest, and `launch_to_start_us <= launch_latency_us`. |
| `UNEXPLAINED_GPU_WAIT` | `queued` | Queued: the next op was submitted before the gap, so the whole gap is GPU-side. |
| `UNEXPLAINED_GPU_WAIT` | `unqueued` | As `LAUNCH_DOMINATED`, but `launch_to_start_us > launch_latency_us`: the op was launched during the gap and took longer than normal to start. The only exception to "largest part wins". |
| `UNATTRIBUTED` | | No traced launch for the next GPU op. |

Whether the gap crossed streams is not a sub-label but the `stream_switch` column,
because it applies to every case and label.

### drain_type

Answers: why was the GPU queue empty? It's independent of the label, which says
what the CPU did during the gap.

- **`sync_drain`:** A sync call on the launching thread was running within 2 µs
  before the gap start, returned before the next launch began, and started more than
  `drain_overlap_us` before the preceding GPU op finished. The CPU was blocked
  while the GPU drained its queue. Removing or moving the sync lets the CPU work
  after it overlap with GPU work.
- **`starved`:** No such sync. The CPU didn't submit work in time. This includes
  gaps where a sync was present but the queue was already empty. Only less CPU
  work per GPU op helps.

These runtime calls count as syncs:

| Calls | Why they block |
|-------|----------------|
| `hipDeviceSynchronize`, `hipStreamSynchronize`, `hipEventSynchronize` and the `cuda*` equivalents | Explicit syncs. |
| `hipFree`, `hipHostFree`, `hipFreeHost`, `cudaFree`, `cudaFreeHost` | Documented to synchronize the device implicitly, except for memory from `hipMallocAsync` or `cudaMallocAsync` (for example, PyTorch's `backend:cudaMallocAsync` allocator). Such a free counts as `sync_drain` only if it happens to span the end of the GPU work, so a long non-blocking free could still be counted. |
| `hipMemcpy`, `hipMemcpyWithStream`, `hipMemcpyAsync`, `hipMemcpy2DAsync`, `hipMemcpyDtoH` and the `cuda*` equivalents | Copies block the host when host memory is pageable, in either direction, including the `Async` forms. |

A call on this list counts only if the timing rules above show that it blocked,
so a copy that returned right away never counts.

Notes for reading `sync_drain` gaps:

- **Copies look different by vendor.** PyTorch `.item()`, `.cpu()`, and `.tolist()`
  show up as `hipMemcpyWithStream` on ROCm, and usually as `cudaMemcpyAsync` to
  pageable memory on CUDA. Use `sync_entry_point` to find the line of code, and
  pinned host memory to avoid the block.
- **`hipFree` inside an op usually means allocator retries.** The PyTorch caching
  allocator frees cached blocks when an allocation fails, then retries. The op
  that happened to allocate isn't the cause; memory pressure is. Check
  `num_alloc_retries` in `torch.cuda.memory_stats()`.
- **Some syncs aren't visible.** A sync on another thread that the launching
  thread waits for, or a sync inside a library that doesn't appear as one of the
  calls above, shows as `starved`.

### Thresholds

All thresholds are fields of `IdleThresholds`, passed to `IdleTimeAnalyser`.

| Field | Default | Description |
|-------|---------|-------------|
| `noise_us` | `5.0` | Gaps shorter than this are noise: counted in the totals but not classified. Set with `--micro_idle_thresh_us` (perf report) or `--micro-thresh` (standalone). |
| `launch_latency_us` | `15.0` | Normal launch return to op start. Separates queued from unqueued gaps and `LAUNCH_DOMINATED` from `UNEXPLAINED_GPU_WAIT/unqueued`. Set with `--idle_launch_latency_us` (perf report) or `--launch-latency` (standalone). |
| `launch_latency_rel` | `None` | If set, the launch latency limit is `max(launch_latency_us, launch_latency_rel * median launch_to_start_us)` over ops launched during gaps, so it adapts to the platform. |
| `other_thread_frac` | `0.5` | Fraction of the gap another thread's runtime call must exceed for `other_thread`. |
| `drain_overlap_us` | `5.0` | How long before the GPU finished a sync must start to count as `sync_drain`. |
| `launch_call_ref_us` | `50.0` | Reference CPU duration of one launch call, used for `launch_call_x_ref`. Kernel launches take a few µs; graph launches normally take much longer. Separate from `launch_latency_us`, which is launch return to op start. |

## Augmented trace annotations

The augmented trace (from `--enable_augmented_trace` or the standalone
`classify_idle_time` module) adds three annotation tracks to the GPU process,
visible in Perfetto:

1. **Idle: Noise/Macro:** labels each gap as `noise` or `idle#N`.
2. **Idle: Drain Type:** labels each macro gap with `sync_drain` and the sync
   call, or `starved`.
3. **Idle: Label:** labels each macro gap with its label and sub-label. The
   slice arguments hold the case, detail, and parts.

## Related topics

- [Analyze GPU idle time](../how-to/analyze-idle-time.md)
- [Performance report column reference](./perf-report-columns.md)
- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
