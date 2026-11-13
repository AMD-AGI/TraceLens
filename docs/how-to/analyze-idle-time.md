<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analyze GPU idle time in TraceLens
```{meta}
:description: Learn how to find where GPU idle time comes from in a PyTorch profiler trace with TraceLens, from the idle_overview, idle_groups, and idle_intervals sheets down to the line of code.
:keywords: TraceLens, GPU idle time, idle gaps, launch latency, synchronization, CPU overhead, PyTorch profiler, Perfetto, ROCm, AMD Instinct
```

TraceLens classifies every gap between consecutive GPU operations (kernels,
memory copies, and memsets) in a PyTorch profiler trace. For each gap, it records
what the CPU thread that launched the next operation was doing, whether a
synchronization call emptied the GPU queue, and the code involved. This topic
shows how to run the analysis and how to use its three sheets to find the code
that causes idle time.

## Before you begin

- [TraceLens installed](../install/install.md).
- A `torch.profiler` Chrome trace (`.json` or `.json.gz`) that includes CPU and
  GPU activity. Capture it with `with_stack=True`. Without Python frames, the
  analysis can name the op behind a gap but not the line of code. See
  [Generate a PyTorch performance report](./generate-perf-report-pytorch.md) for
  capture settings.
- For multi-GPU runs, one trace per rank. Analyze each rank's trace separately,
  not a file that merges several ranks.

## Run the analysis

Add `--enable_idle_analysis` to the PyTorch performance report command:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis \
    --include_call_stack \
    --enable_augmented_trace
```

`--enable_idle_analysis` adds three sheets to the report workbook:
`idle_overview`, `idle_groups`, and `idle_intervals`. The other options are:

- `--include_call_stack`: records the call stack of each sync call and covering
  op, and splits the sync rows of `idle_overview` by the line of code that issued
  the sync. Use it when you diagnose idle time.
- `--enable_augmented_trace`: writes `<trace>_idle_augmented.json.gz` next to the
  input trace. It's the original trace plus annotation tracks that label each
  gap, for viewing in [Perfetto](https://ui.perfetto.dev).
- `--micro_idle_thresh_us` (default `5`): gaps shorter than this are counted in
  the idle total but not classified.
- `--idle_launch_latency_us` (default `15`): the normal time from a launch call's
  return to the start of the GPU op.

To run only the idle analysis, use the standalone module:

```bash
python -m TraceLens.Reporting.classify_idle_time trace.json --include-call-stack
```

It prints `idle_overview`, writes an Excel workbook with the three idle sheets,
and writes an augmented trace to `<trace>_idle_classified.json.gz` (change the
path with `-o`).

The Python API returns the sheets as pandas `DataFrame` objects. Build the tree
with `add_python_func=True` to get call stacks. Thresholds are fields of
`IdleThresholds`.

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.IdleTimeAnalyser import IdleThresholds, IdleTimeAnalyser

pa = TreePerfAnalyzer.from_file("trace.json", add_python_func=True)
analyser = IdleTimeAnalyser(pa.tree, IdleThresholds(), include_call_stack=True)
dfs = analyser.get_dataframes()   # {"idle_overview": ..., "idle_groups": ..., "idle_intervals": ...}
gaps = analyser.classify()        # list of per-gap dicts
```

## How a gap is classified

A gap ends when the GPU starts its next operation. The analysis finds the
runtime call that launched that operation and the thread it ran on, then
describes the gap with three fields: the case and its parts, `drain_type`, and
`stream_switch`.

### Case and parts

The case says whether the next op had been launched when the gap started:

- `unqueued`: the next op was launched during the gap, or less than
  `launch_latency_us` before it. The GPU was waiting for the CPU. The gap is split
  into parts by what the launching thread did:
  - `cpu_op_us`: CPU ops and Python functions.
  - `cpu_runtime_us`: runtime calls other than the launch, such as `hipMalloc` or
    a blocking copy.
  - `cpu_untraced_us`: nothing traced.
  - `launch_call_us`: the launch call itself.
  - `submit_latency_us`: from the launch call's return to the start of the op.
- `queued`: the next op was launched before the gap, so the CPU wasn't the
  bottleneck. The whole gap is `gpu_wait_us`.
- `no_launch`: the trace has no launch call for the next op. The whole gap is
  `unattributed_us`.

The parts add up to the gap's duration. The gap's label names its largest part:

| Label | Largest part |
|-------|--------------|
| `CPU_OP_DOMINATED` | `cpu_op_us` |
| `RUNTIME_DOMINATED` | `cpu_runtime_us` |
| `UNTRACED_DOMINATED` | `cpu_untraced_us` |
| `LAUNCH_DOMINATED` | `launch_call_us` plus `submit_latency_us` |
| `UNEXPLAINED_GPU_WAIT` | `gpu_wait_us`, or a launch during the gap after which the op took longer than `launch_latency_us` to start |
| `UNATTRIBUTED` | `unattributed_us` |

### drain_type

`drain_type` says why the GPU queue was empty:

- `sync_drain`: the launching thread was blocked in a sync call while the GPU
  finished its queued work. Examples are `hipStreamSynchronize`, `hipFree`, and
  the blocking copy behind `.item()` or `.cpu()`. The gap also includes the CPU
  work after the sync returned, up to the next launch.
- `starved`: the GPU finished its work before the CPU launched the next op, and
  no sync call was waiting on that work.

The two need different fixes. For a `sync_drain` gap, removing or deferring the
sync lets the CPU work after it run while the GPU is still busy, so the gap
shrinks even if that work stays as slow. A `starved` gap only shrinks if the CPU
does less work between GPU ops.

### stream_switch

`stream_switch` is true when the GPU ops before and after the gap ran on
different streams. The next op might have waited on work in another stream.

The [idle time sheet reference](../reference/idle-time-columns.md) gives the
exact rules and every column.

## Diagnose idle time

`idle_overview` gives idle time totals by cause, `idle_groups` groups gaps by
code location, and `idle_intervals` lists each gap. The following steps use them
in that order.

### Step 1: Check GPU utilization

In `idle_overview`, `gpu_utilization_pct` is GPU busy time divided by busy time
plus idle time. It matches the `gpu_timeline` sheet. Above about 95%, idle time
is unlikely to be the main bottleneck; look at kernel performance instead.

Next, compare the `noise` and `macro` rows. Macro gaps are the gaps of 5 µs or
longer, which the analysis classifies. If `noise` holds a large share of idle
time, the GPU is running many short kernels back to back and is limited by the
launch rate. Kernel fusion and graph capture (CUDA or HIP graphs) reduce this
kind of idle time.

### Step 2: Separate sync drains from starvation

On the `macro` row, `sync_drain_ms` and `starved_ms` add up to `total_ms`. The
`drain` rows break this down into `starved` and one row per sync call. With
`--include_call_stack`, each sync row is one line of code:

```text
sync_drain: hipMemcpyWithStream @ vllm/v1/attention/backends/rocm_aiter_fa.py(459): build
```

A sync row without `@` means the sync's call stack has no Python frame, for
example a sync issued from the autograd thread during the backward pass. Its
`sync_entry_point` is `Not found`, and the row can combine several call paths. To
split it, group `idle_intervals` by `sync_call_stack_full`. A stack of op names
still identifies the call: `aten::item` → `aten::_local_scalar_dense` →
`hipMemcpyWithStream` is an `.item()`.

A sync row's `total_ms` isn't what removing the sync saves. Each gap holds the
time the sync blocked plus the CPU work after it. Removing the sync lets that
work start earlier by the time the sync blocked during the gap, so the most it
saves per gap is `sync_gain_us = min(sync_event_dur, duration_us)`.
`sync_event_dur` is often longer than the gap, because the call started while the
GPU was still busy. The `sync_gain_ms` column of each drain row sums
`sync_gain_us` over the row's gaps. Compare it with the row's `total_ms`:

- If `sync_gain_ms` is close to `total_ms`, removing or deferring the sync
  removes most of the row's idle time.
- If `sync_gain_ms` is a small part of `total_ms`, most of the time is CPU work
  that runs after the sync anyway, for example preprocessing the next input after
  copying results out. Moving the sync doesn't help. Find that work in
  `idle_groups` (the `covering_op` of the `sync_drain` groups) and make it faster,
  or run it ahead of the GPU with pipelining or data loader workers. If the work
  is folded into an `other` group, filter `idle_intervals` by `sync_event_name`
  and `sync_entry_point` instead.

`sync_gain_us` counts only the sync at the start of the gap. If `cpu_runtime_ms`
is most of the row and `top_runtime` is the same call, more calls of it follow in
the gap, for example one `hipFree` after another. Removing all of them removes
the gap.

The estimate assumes the work after the sync doesn't use the sync's result. Code
that branches on an `.item()` value has to move the branch as well.

### Step 3: See what the CPU was doing

The `label` rows give macro idle time per label, split into `sync_drain_ms`,
`starved_ms`, and `stream_switch_ms`. The part columns, `cpu_op_ms` through
`unattributed_ms`, show what the gaps contained. A label names only each gap's
largest part, so read the part columns too. For example, a large `launch_call_ms`
on the `CPU_OP_DOMINATED` row means launch cost is also significant in gaps where
CPU ops took the largest share.

The `sub_label` rows split `RUNTIME_DOMINATED` into `other_thread`, and
`UNEXPLAINED_GPU_WAIT` into `queued` and `unqueued`.

For `RUNTIME_DOMINATED/other_thread`, the part columns describe the launching
thread, which can show mostly `cpu_op_ms`, for example while it waits inside
`run_backward`. The blocking call is on another thread. In `idle_intervals`,
`other_thread_top_runtime` names it and `other_thread_runtime_us` gives its time.

### Step 4: Find the code location

Each `idle_groups` row holds the gaps that share a label, sub-label,
`drain_type`, and `covering_op`, sorted by total time. `cumulative_pct` shows how
many groups cover most of the idle time. For each large group, read:

- `covering_op`: the CPU op on the launching thread during the gap, for example
  `aten::addmm`, a Python frame such as `model.py(42): forward`, or a C function
  called from Python such as `<built-in function zoom_shift>`. It's the deepest op
  that spans at least half the gap, or otherwise the op with the most time in the
  gap. `(no CPU op)` means the thread ran nothing traced, and `(no launch)` marks
  `UNATTRIBUTED` gaps. For a generic op such as `aten::copy_`, the
  `covering_call_stack_full` column of `idle_intervals` shows the calling code.
- `top_runtime`: the runtime call with the most time in the group's gaps. It's
  blank when that call holds less than 10% of the group's time.
- `top_launch`: the launch call with the most idle time before it.
  `hipGraphLaunch` or `cudaGraphLaunch` means the gaps end in graph launches.
- `top_next_gpu_op`: the GPU op with the most idle time before it. It's most
  useful for `UNEXPLAINED_GPU_WAIT`. A group can contain several next ops; to see
  them, sum `duration_us` by `following_gpu_event` in `idle_intervals`.
- `gaps`, `mean_us`, and `median_us`: many short gaps indicate per-op overhead,
  and a few long gaps indicate a specific event.

`covering_op` is the op that ran during the gap, which isn't always the cause:

- For `sync_drain` gaps, the cause is the sync, and the Step 2 drain rows give its
  line of code. The CPU work after a sync spreads over whichever ops run next, so
  one sync can appear as several `sync_drain` groups with the same `top_runtime`.
  If `sync_gain_ms` covers most of the drain row, removing the sync fixes all of
  those groups. If it doesn't, treat each large group as separate CPU work.
- For `UNEXPLAINED_GPU_WAIT/queued` gaps, the next op was already launched, so
  `covering_op` is only an op that overlapped the gap, for example an `.item()`
  waiting for the GPU. All parts except `gpu_wait_ms` are zero. Read
  `top_next_gpu_op` and the streams instead.
- For `starved` gaps, `launch_parent_op` in `idle_intervals`, the op that launched
  the next GPU op, is a second code location.

Groups under 1% of idle time are folded into one `other (N groups)` row per
label, sub-label, and `drain_type`. The three largest groups of each label and
sub-label are never folded. `folded_top_ops` names the `covering_op` of the
largest folded groups. A large `other` row means the idle time is spread over
many ops, which points to per-op overhead rather than one hotspot.

### Step 5: Inspect individual gaps

In `idle_intervals`, filter by `group_id`, or look up a group's `worst_idle_id`.
The most useful columns are:

- `detail`: the nonzero parts, largest first. For queued and no-launch gaps, a
  short explanation instead.
- `sync_event_name`, `sync_event_dur`, `sync_entry_point`, and
  `sync_call_stack_full`: the sync call, how long it blocked in µs, the user frame
  that issued it, and its call stack from the root. Stack frames are Python
  functions (`file(line): function`) when the trace has them, otherwise op names.
- `launch_parent_op`, `following_launch_name`, and `launch_to_start_us`: how the
  next op was launched. `launch_tid` tells threads apart, for example the main
  thread and the autograd thread. `launch_call_x_ref` is the launch call's CPU
  time divided by 50 µs; above 1, the launch call was slow.
- `preceding_gpu_event`, `following_gpu_event`, and their streams.
- `start_us`: sort a group's gaps by it. Gaps at a regular spacing happen once per
  step or once per layer.

To see a gap in context, open the augmented trace in Perfetto and search for
`idle#<idle_id>`. To find the code that launched the next op, follow the UID
columns into the [Trace2Tree](../conceptual/trace2tree.md) model:

```python
tree = pa.tree
gap = dfs["idle_intervals"].iloc[0]          # the longest gap
evt = tree.events_by_uid[gap["following_launch_uid"]]
while evt is not None:                       # launch call -> ops -> Python frames
    print(evt["name"])
    evt = tree.events_by_uid.get(evt.get("parent"))
```

Then read the source at the file and line from `sync_entry_point` or the launch
stack.

## Common patterns

The following table lists common patterns, their usual cause, and what to try.
Confirm a pattern against the part columns and a few individual gaps before you
act on it.

| Pattern | Usual cause | What to try |
|---------|-------------|-------------|
| Large `sync_drain` row for `hipMemcpyWithStream` or `cudaMemcpyAsync` at a line of user code, with `sync_gain_ms` close to `total_ms` | `.item()`, `.cpu()`, `.tolist()`, printing a tensor, or a Python condition on a GPU value | Remove it, defer it to the end of the step, or copy to pinned memory with `non_blocking=True` and sync later. |
| Large `sync_drain` row with `sync_gain_ms` much smaller than `total_ms` | CPU work that runs after the sync anyway, for example preprocessing the next input | Make that work faster or run it ahead of the GPU (see Step 2). |
| Large `sync_drain: hipFree` or `cudaFree` row, often without `@` | The caching allocator is disabled (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`), or it frees cached blocks to retry a failed allocation | If every op frees memory, check that the caching allocator is enabled. Otherwise check `num_alloc_retries` in `torch.cuda.memory_stats()`, then reduce peak memory or set `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` if your PyTorch build supports it. |
| Large `sync_drain` row for `hipDeviceSynchronize`, `hipStreamSynchronize`, or an event sync | An explicit sync, often `torch.cuda.synchronize()` for timing or logging | Remove it or move it out of the step. |
| `starved` and `CPU_OP_DOMINATED`, many short gaps under one `covering_op` | Host overhead per op: dispatch, Python, small kernels | Fuse ops, use `torch.compile`, capture a graph, or give each op more work with a larger batch. |
| `starved` and `LAUNCH_DOMINATED`, many gaps of about 10 µs | Launch rate: each small kernel finishes before the next one arrives | Capture a graph or fuse kernels. |
| `LAUNCH_DOMINATED` with `top_launch` `hipGraphLaunch` or `cudaGraphLaunch` and `launch_call_us` of hundreds of µs or more | CPU cost of launching a large graph | Launch the next graph before you sync on the previous one, so the launch overlaps GPU work. If graph launches run back to back, capture fewer, larger graphs. |
| `launch_call_x_ref` above 1 on `hipLaunchKernel` or `cudaLaunchKernel` | The launching thread was descheduled or contended, for example too few CPU cores per process | Check CPU affinity and the number of threads per process. |
| `RUNTIME_DOMINATED` and `starved`, with `top_runtime` `hipMalloc` or `hipFree` | Memory allocation on the critical path, for example with the caching allocator disabled | Keep the caching allocator enabled and preallocate buffers. |
| `RUNTIME_DOMINATED/other_thread` | A long runtime call on another thread (data loader, communication, allocator) blocked the launching thread | Find that thread in Perfetto and move its work off the critical path. |
| `UNTRACED_DOMINATED` | CPU work the profiler didn't record: Python without `with_stack`, data loading, waiting for the GIL | Profile again with `with_stack=True` or `record_function` ranges, or use a Python profiler. |
| `UNEXPLAINED_GPU_WAIT/queued` with a large `stream_switch_ms` | The next op waited on another stream, for example on an event or a collective | Check the other stream in Perfetto. For collectives, compare ranks to find a straggler. |
| `UNEXPLAINED_GPU_WAIT` on the same stream | Runtime, driver, or hardware behavior, or profiler overhead | Compare with profiling off. Report it if it's large. |
| `UNATTRIBUTED` | The trace has no CPU activity for the next op | Check the profiler settings, and whether another process launched the op. |

## Pitfalls

- A label names only the largest part of a gap. Read the part columns before you
  decide that a gap is CPU-bound or launch-bound.
- For `sync_drain` gaps, `covering_op` is the work after the sync, not the sync.
  The drain rows and `sync_entry_point` point at the sync.
- A sync row's `total_ms` isn't the gain from removing the sync. Use
  `sync_gain_ms`.
- Some syncs show as `starved`: a sync on another thread, a sync inside a library
  call that isn't a known sync call, and a copy issued after the GPU was already
  idle.
- Profiler overhead adds CPU time, especially between small ops. Confirm a gain
  with profiling off.
- Removing a gap shortens the step only if the work after it is on the critical
  path, so idle time is an upper bound on the gain.

## Examples

### Diffusion model inference limited by the CPU

A diffusion model inference trace has 62% GPU utilization:

1. Macro idle time is 35% of the trace, so idle time matters.
2. Nearly all idle time is `starved`. The sync rows add up to about 0.1% of idle
   time, so syncs aren't the problem.
3. `CPU_OP_DOMINATED` holds 65% of idle time and `LAUNCH_DOMINATED` holds 27%. The
   part columns of the `CPU_OP_DOMINATED` row hold another 53 ms of launch time,
   so launch cost is larger than the `LAUNCH_DOMINATED` row shows.
4. The largest `idle_groups` row is `CPU_OP_DOMINATED` under `aten::addmm`, with
   24% of idle time. The next are an attention processor's `__call__` and
   `aten::mm`. The `LAUNCH_DOMINATED` time is spread over many small groups of
   short gaps.
5. In the augmented trace, the worst `aten::addmm` gaps show the GPU finishing its
   kernels while the CPU is still inside `aten::addmm`, before it launches the
   GEMM.
6. The CPU can't keep up with many short kernels. Reducing CPU overhead around
   the linear layers, fusing ops, or capturing the model in a graph should reduce
   idle time.

### LLM serving with a blocking copy

A vLLM serving trace has 89% GPU utilization and 31 ms of macro idle time:

1. One drain row, `sync_drain: hipMemcpyWithStream @
   vllm/v1/attention/backends/rocm_aiter_fa.py(459): build`, holds 22 ms over 126
   gaps, or 67.5% of idle time.
2. The row's `sync_gain_ms` is also 22 ms: each copy blocked for longer than its
   gap. Removing the wait would remove nearly all of the row's idle time.
3. The row's `cpu_op_ms` is 20 ms, and its label is `CPU_OP_DOMINATED`. After each
   copy returns, the CPU runs ops for about 160 µs before it launches the next GPU
   op. Without the blocking copy, that work would overlap GPU work.
4. In `idle_groups`, `hipMemcpyWithStream` is the `top_runtime` of several
   `sync_drain` groups with different `covering_op` values (`execute_model`,
   `build`, and `_build_attn_group_metadata`). These groups are the CPU work after
   the same copy, not separate causes.
5. `sync_call_stack_full` shows that the copy is a `.cpu()` call in the attention
   backend's metadata `build`.
6. Make the copy asynchronous with pinned memory and a later sync, or move it
   after the next launches. The metadata build then overlaps GPU work without
   being made faster.

## Performance overhead

Idle time analysis adds classification and `DataFrame` construction to the
normal trace load. The following table shows measurements on representative
PyTorch traces.

| Trace | Size (gzip) | Macro gaps | Load | Idle analysis |
|-------|-------------|------------|------|---------------|
| ResNet-18 DDP training | 0.6 MB | 696 | 0.6 s | 0.5 s |
| Llama 70B FP8 inference | 1.7 MB | 141 | 1.2 s | 0.6 s |
| LLM serving (vLLM) | 5.5 MB | 501 | 5.8 s | 3.4 s |
| Diffusion model inference | 21 MB | 14,882 | 23.7 s | 15.7 s |

On very large traces, raise the noise threshold (`--micro_idle_thresh_us`, or
`noise_us` in `IdleThresholds`) to classify fewer short gaps.

## Related topics

- [Idle time sheet reference](../reference/idle-time-columns.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Trace2Tree data model](../conceptual/trace2tree.md)
- [Analyze traces with the SDK](./sdk-analysis.md)
