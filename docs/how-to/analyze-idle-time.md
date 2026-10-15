<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analyze GPU idle time in TraceLens
```{meta}
:description: Learn how to find where GPU idle time comes from in a PyTorch profiler trace with TraceLens, from the idle_overview, idle_groups, and idle_intervals sheets down to the line of code.
:keywords: TraceLens, GPU idle time, idle gaps, launch latency, synchronization, CPU overhead, PyTorch profiler, Perfetto, ROCm, AMD Instinct
```

TraceLens explains every gap between consecutive GPU events (kernels, memory
copies, and memsets) instead of reporting GPU idle time as a single number. This
topic shows how to run the analysis and then walks through a diagnosis: from how
much time is idle, to what kind of problem it is, to the line of code that causes
it. The steps work the same whether you read the sheets yourself or hand them to
an AI agent.

## Before you begin

Before running the analysis, confirm you have the following:

- [TraceLens installed](../install/install.md).
- A `torch.profiler` Chrome trace (`.json` or `.json.gz`) with CPU and GPU
  activity. See
  [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
  for capture settings. Capture with `with_stack=True` so the analysis can name
  the Python code behind each gap.
- For multi-GPU runs, one trace per rank. Analyze each rank's trace separately,
  not a file that merges several ranks.

## Run the analysis

Add `--enable_idle_analysis` to the PyTorch report to append the
`idle_overview`, `idle_groups`, and `idle_intervals` sheets. For a diagnosis, also
add `--include_call_stack`: it names the line of code behind each sync, and
splits the sync rows of the overview by that line. Add `--enable_augmented_trace`
to write `<trace>_idle_augmented.json.gz` next to the input trace, the original
trace plus idle annotation tracks for [Perfetto](https://ui.perfetto.dev):

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis \
    --include_call_stack \
    --enable_augmented_trace
```

Gaps shorter than `--micro_idle_thresh_us` (default `5` µs for idle analysis)
count as noise: they're included in the idle total but not classified. Use
`--idle_launch_latency_us` to change the launch latency threshold (default 15 µs).

To analyze idle time without generating the full report, run the standalone
module. It prints the overview, writes an Excel workbook with the three idle
sheets, and writes an augmented trace (`<trace>_idle_classified.json.gz` by
default; override with `-o`):

```bash
python -m TraceLens.Reporting.classify_idle_time trace.json --include-call-stack
```

Or use the Python API, which returns the same sheets as pandas `DataFrame`
objects. All thresholds are fields of `IdleThresholds`. Call stacks need the
tree built with Python functions:

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.IdleTimeAnalyser import IdleThresholds, IdleTimeAnalyser

pa = TreePerfAnalyzer.from_file("trace.json", add_python_func=True)
analyser = IdleTimeAnalyser(pa.tree, IdleThresholds(), include_call_stack=True)
dfs = analyser.get_dataframes()   # {"idle_overview": ..., "idle_groups": ..., "idle_intervals": ...}
gaps = analyser.classify()        # list of per-gap dicts
```

## How a gap is described

The CPU submits work to the GPU by enqueueing launches into GPU queues, and the
GPU runs them independently. The GPU goes idle when its queue is empty, or when
it doesn't start work that's already queued. For each gap, the analysis looks at
the GPU op that ends the gap and at the thread that launched it, and records
three things.

**Case and parts: what the launching thread was doing.** The gap is split into
parts that add up to its duration:

- *Unqueued:* the next op was launched during the gap (or less than
  `launch_latency_us` before it), so the GPU was waiting for the CPU. The time
  before the launch call is split into `cpu_op_us` (CPU ops and Python
  functions), `cpu_runtime_us` (runtime calls such as `hipMalloc` or a blocking
  copy), and `cpu_untraced_us` (nothing traced). Then come the launch call
  itself (`launch_call_us`) and launch return to op start (`submit_latency_us`).
- *Queued:* the next op was launched before the gap, so the CPU wasn't the
  bottleneck. The whole gap is `gpu_wait_us`.
- *No launch:* the trace has no launch for the next op. The whole gap is
  `unattributed_us`.

The **label** names the largest part: `CPU_OP_DOMINATED`, `RUNTIME_DOMINATED`,
`UNTRACED_DOMINATED`, `LAUNCH_DOMINATED` (launch call plus normal launch
latency), `UNEXPLAINED_GPU_WAIT` (queued, or launched but late to start), or
`UNATTRIBUTED`.

**`drain_type`: why the queue was empty.** `sync_drain` means a sync call on the
launching thread was blocking the CPU when the GPU finished its queue, for example
`hipStreamSynchronize`, `.item()` (a blocking copy), or `hipFree`. The gap also
includes whatever the CPU did after the sync returned, so the sync itself may be
a small part of it (see Step 2). `starved`
means the CPU just didn't submit work in time. The two need different fixes:
removing or moving a sync lets the CPU work after it start earlier and overlap
with GPU work, even if that work stays as slow. A starved gap only shrinks if the
CPU does less work per GPU op.

**`stream_switch`: whether the ops on either side of the gap ran on different
streams.** A switch can mean the next op waited on another stream.

The [idle time sheet reference](../reference/idle-time-columns.md) defines every
column and rule.

## Diagnose idle time

The three sheets go from coarse to fine: `idle_overview` says how much time is
idle and why, `idle_groups` says where in the code, and `idle_intervals` lists
each gap. Work through the following steps in order. Each one narrows the
question for the next.

### Step 1: Is idle time the problem?

In `idle_overview`, read the `gpu_utilization_pct` row. It's GPU busy time as a
share of busy time plus all idle time, and it matches the `gpu_timeline` sheet.
Above about 95%, idle time is unlikely to be the main bottleneck; look at kernel
performance instead.

Then compare the `noise` and `macro` rows. Macro gaps (5 µs or longer) are what
the rest of the analysis explains. If noise holds a large share of idle time, the
GPU runs many short kernels back to back and is limited by launch rate; fusion and
graph capture (CUDA or HIP graphs) help more than any single fix.

### Step 2: Sync or starvation?

On the `macro` row, compare `sync_drain_ms` with `starved_ms`. Then read the
`drain` rows: `starved`, and one row per sync call that drained the queue. With
`--include_call_stack`, each sync row is one line of code,
`sync_drain: <call> @ <file>(<line>): <function>`. If a sync row has no `@`, the
sync was issued with no Python frame in its call stack, for example from the
autograd thread in the backward pass (`sync_entry_point` is then `Not found`),
and the row can mix several call paths. Group `sync_call_stack_full` in
`idle_intervals` to split it (a row with `@` can also mix paths that share the
same innermost user frame); an op-only stack such as `aten::item` →
`aten::_local_scalar_dense` → `hipMemcpyWithStream` is enough to recognize the
call.

A sync row names where the queue drained, not how much removing the sync would
save. Removing a sync lets the CPU start the work after it earlier, by as long
as the sync blocked, so on each gap the most it can save is
`min(sync_event_dur, duration_us)` (both in µs). `sync_event_dur` is often much
longer than the gap, because the call started while the GPU was still busy;
only the gap is idle time, hence the cap. The bound is the `sync_gain_us` column
of `idle_intervals`, summed per row in `sync_gain_ms` on the drain rows. Compare
it with the row's `total_ms`:

- *The bound is the gain from removing or deferring the sync,* whatever the
  gap's parts are: the blocking call itself, CPU ops, or untraced time that
  waited behind it (as in the vLLM example below). It counts only the sync at
  the gap start. If `cpu_runtime_ms` is most of the row and `top_runtime` is the
  same call, more calls of it follow in the gap (for example one `hipFree` after
  another), and removing them all removes the gap.
- *The rest of the row's time is CPU work that runs after the sync anyway.* If
  that's most of the row, for example preprocessing the next input after copying
  the results out, find that work in `idle_groups` (`covering_op` of the
  `sync_drain` groups) and make it faster or run it ahead of the GPU (pipelining,
  data loader workers). Moving the sync doesn't help. For a small row, that work
  may be folded into an `other` group; filter `idle_intervals` by
  `sync_event_name` and `sync_entry_point` instead.

The bound assumes the work after the sync doesn't need its result; for example,
code that branches on an `.item()` value has to move the branch too.

### Step 3: What was the CPU doing?

The `label` rows show how much macro idle each label holds, split into
`sync_drain_ms`, `starved_ms`, and `stream_switch_ms`. The `*_ms` part columns
show what the gaps were made of. A label names only each gap's largest part, so
read the part columns too. For example, a large `launch_call_ms` on the
`CPU_OP_DOMINATED` row means launch cost hides inside gaps where CPU ops took the
largest share. The `sub_label` rows split `RUNTIME_DOMINATED` (`other_thread`)
and `UNEXPLAINED_GPU_WAIT` (`queued`, `unqueued`).

`RUNTIME_DOMINATED/other_thread` is the exception to reading the part columns:
the parts describe the launching thread, which may show mostly `cpu_op_ms` (for
example, waiting inside `run_backward`). The blocking call is on another thread;
it's in `other_thread_top_runtime` and `other_thread_runtime_us` in
`idle_intervals`.

### Step 4: Where in the code?

Each `idle_groups` row is one label, sub-label, `drain_type`, and `covering_op`,
sorted by total time; `cumulative_pct` shows how few groups cover most of the
idle time. Columns to read for each large group:

- `covering_op`: the CPU op running on the launching thread during the gap, for
  example `aten::addmm`, a Python frame such as `model.py(42): forward`, or a C
  function called from Python such as `<built-in function zoom_shift>` (often a
  library outside PyTorch; find its caller in Perfetto or the tree). It's
  the deepest op spanning at least half the gap, or the op with the most time in
  the gap if none does. `(no CPU op)` means the thread ran nothing traced;
  `(no launch)` is for `UNATTRIBUTED`. If it's a generic op such as
  `aten::copy_`, `covering_call_stack_full` in `idle_intervals` (with
  `--include_call_stack`) shows the code that called it.
- `top_runtime`: the runtime call with the most time in the group's gaps. It's
  blank when that call holds under 10% of the group's time, because it's then
  incidental.
- `top_launch`: the launch call for the next op with the most idle time before
  it. `hipGraphLaunch` or `cudaGraphLaunch` here means the group's gaps end in
  graph launches.
- `top_next_gpu_op`: the single GPU op with the most idle time before it. For
  `UNEXPLAINED_GPU_WAIT`, it's the main clue, but a group can mix several next
  ops; sum `following_gpu_event` in `idle_intervals` to see the mix. For gaps the
  CPU caused, it's only the op that ran after the gap.
- `gaps`, `mean_us`, `median_us`: many short gaps point at per-op overhead; a few
  long ones point at a specific event.

`covering_op` is the op running *during* the gap, not necessarily the op that
caused it:

- After a sync, the CPU work that follows is spread over whichever ops run next,
  so one sync line can appear as several `sync_drain` groups with the same
  `top_runtime`. For syncs, the Step 2 drain rows are the code location; use the
  groups to see what the CPU did afterward. Whether those groups are one cause
  depends on the Step 2 bound: if it covers the row, removing the sync fixes all
  of them; if it's small, each large group is CPU work to fix on its own.
- For queued gaps (`UNEXPLAINED_GPU_WAIT/queued`), the CPU had already launched
  the next op, so `covering_op` is only whatever long-running op overlapped the
  gap, for example an `.item()` waiting for the GPU. All part columns except
  `gpu_wait_ms` are zero; read `top_next_gpu_op` and the streams instead. The
  same `.item()` can still be a real `sync_drain` on the gap where it returns.

For starved gaps, `launch_parent_op` in `idle_intervals` (the op that launched the
next GPU op) is a second code location next to `covering_op`.

Groups under 1% of idle time are folded into one `other (N groups)` row per label,
sub-label, and `drain_type`, except the three largest groups of each label and
sub-label, which always stay. `folded_top_ops` names the `covering_op` of the
largest folded groups. If an `other` row is among the largest groups, the idle
time is spread over many ops: per-op overhead, not one hotspot.

### Step 5: Drill down to individual gaps

In `idle_intervals`, filter by `group_id`, or look up a group's `worst_idle_id`.
Useful columns per gap:

- `detail`: the non-zero parts, largest first; for queued and no-launch gaps, a
  short explanation instead.
- `sync_event_name`, `sync_event_dur`, `sync_entry_point`, `sync_call_stack_full`:
  the sync, how long it blocked, and the stack that issued it. The stack is a list
  from the root to the sync call; frames are Python functions (`file(line):
  function`) when the trace has them, otherwise op names only.
- `launch_parent_op`, `following_launch_name`, `launch_to_start_us`: how the next
  op was launched. `launch_tid` tells threads apart, for example the main thread
  from the autograd thread. `launch_call_x_ref` is the launch call's CPU time
  over 50 µs; above 1, the launch itself was slow.
- `preceding_gpu_event`, `following_gpu_event`, and their streams.
- `start_us`: sort a group's gaps by it. Gaps at a regular spacing are once per
  step (or per layer), which tells you how much each step loses.

To see the gap in context, open the augmented trace in Perfetto and search for
`idle#<idle_id>`. To get the code that launched the next op, follow the UID
columns into the [Trace2Tree](../conceptual/trace2tree.md) model:

```python
tree = pa.tree
gap = dfs["idle_intervals"].iloc[0]          # the longest gap
evt = tree.events_by_uid[gap["following_launch_uid"]]
while evt is not None:                       # launch call -> ops -> Python frames
    print(evt["name"])
    evt = tree.events_by_uid.get(evt.get("parent"))
```

With a file and line from `sync_entry_point` or the launch stack, read the source
around that line to see why it syncs or why it's slow.

## Common patterns

The following table maps what you see in the sheets to the likely cause and what
to try. Check the pattern against the part columns and a few gaps before acting
on it.

| Pattern | Likely cause | What to try |
|---------|--------------|-------------|
| Large `sync_drain: hipMemcpyWithStream` or `cudaMemcpyAsync` row (at a line of user code, or with `aten::item` or `aten::_local_scalar_dense` in the stack), with a large Step 2 bound | `.item()`, `.cpu()`, `.tolist()`, `print` of a tensor, or a host-side condition on a GPU value | Remove it, defer it to the end of the step, or copy to pinned memory with `non_blocking=True` and sync later. |
| Large sync row with a Step 2 bound that's a small share of its time | CPU work that follows the sync anyway, for example preprocessing the next input | Find that work in `idle_groups` (`covering_op` of the `sync_drain` groups); make it faster or run it ahead. Moving the sync doesn't help. |
| Large `sync_drain: hipFree` or `cudaFree` row, often with no `@` site | Frees that synchronize the device: either the caching allocator is disabled (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`), so every free is a `hipFree`, or the caching allocator freed cached blocks to retry a failed allocation (memory pressure). Either way the op that freed isn't the cause | If the frees repeat once per op every step, check the allocator is enabled first. Otherwise check `num_alloc_retries` in `torch.cuda.memory_stats()`; reduce peak memory (smaller batch, activation checkpointing), or try `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` where your PyTorch build supports it. |
| Large `sync_drain` row for `hipDeviceSynchronize`, `hipStreamSynchronize`, or an event sync | Explicit sync, often timing, logging, or a `torch.cuda.synchronize()` | Remove it or move it out of the step. |
| `starved`, `CPU_OP_DOMINATED`, many short gaps under one `covering_op` | Host overhead per op (dispatch, Python, small kernels) | Fuse ops, use `torch.compile`, capture a graph, or give each op more work (larger batch). |
| `starved`, `LAUNCH_DOMINATED`, many gaps near 10 µs | Launch rate: the GPU finishes each small kernel before the next one arrives | Graph capture or kernel fusion. |
| `LAUNCH_DOMINATED` with `top_launch` or `following_launch_name` `cudaGraphLaunch` or `hipGraphLaunch` and `launch_call_us` in the hundreds of µs or more | The CPU cost of launching a large graph | Capture fewer, larger graphs per step only if the launches are back to back; otherwise overlap the graph launch with GPU work (launch the next graph before syncing on the previous one). |
| `launch_call_x_ref` above 1 on ordinary kernel launches (`hipLaunchKernel`, `cudaLaunchKernel`), up to the ms range | The launching thread was descheduled or contended (too few CPU cores per process, CPU oversubscription). For graph launches, see the previous row | Check CPU affinity and the number of threads per process. |
| `RUNTIME_DOMINATED` with `top_runtime` `hipMalloc` or `hipFree`, `starved` | Allocation on the critical path, for example with the caching allocator disabled | Keep the caching allocator enabled; preallocate buffers. |
| `RUNTIME_DOMINATED/other_thread` | Another thread (data loader, communication, allocator) was in a long runtime call and blocked the launching thread | Find that thread in Perfetto; move the work off the critical path. |
| `UNTRACED_DOMINATED` | CPU work that the profiler didn't record: Python without `with_stack`, data loading, waiting for the GIL | Re-profile with `with_stack=True` or `record_function` ranges, or use a Python profiler. |
| `UNEXPLAINED_GPU_WAIT/queued` with high `stream_switch_ms` | The next op waited on work in another stream, for example an event wait or a collective | Check the other stream in Perfetto; for collectives, compare ranks for a straggler. |
| `UNEXPLAINED_GPU_WAIT` on the same stream | Runtime, driver, or hardware behavior, or profiler overhead | Not fixable from Python. Compare with profiling off; report it if it's large. |
| `UNATTRIBUTED` | The trace has no CPU activity for the next op | Check the profiler settings and whether another process launched the op. |

## Pitfalls

- *A label names only the largest part.* Read the part columns before deciding a
  gap is "CPU" or "launch".
- *`covering_op` isn't the cause of a sync.* For `sync_drain` gaps, the drain rows
  and `sync_entry_point` point at the sync; `covering_op` shows the work after it.
- *A sync row's time isn't the gain from removing it.* Check the Step 2 bound.
- *A sync can hide.* A sync on another thread, or one inside a library call that
  isn't a known sync call, shows as `starved`.
- *Profiler overhead inflates CPU time.* Short gaps between small ops are partly
  profiler cost. Confirm the gain with the profiler off.
- *Idle time is the upper bound of the gain.* Removing a gap only shortens the
  step if the work after it was on the critical path.

## Report the findings

Whether you or an agent write it up, a short report with the following structure
is easy to act on:

1. GPU utilization, macro idle time, and its split into `sync_drain` and
   `starved`.
2. The top three to five causes, each with its share of idle time, label and
   `drain_type`, code location (`sync_entry_point`, `covering_op`,
   `launch_parent_op`, or the launch stack), gap count and typical length, and the
   evidence from the part columns. For a sync, include the Step 2 bound
   (`sync_gain_ms`).
3. For each cause, what to try first and how confident you are.
4. What's left unexplained (`UNEXPLAINED_GPU_WAIT`, `UNATTRIBUTED`, untraced
   time) and what would explain it.

## Example workflows

### CPU-bound inference: make the CPU faster

The following example is a diffusion model inference trace with 62% GPU
utilization:

1. Utilization is 62%, so idle time matters. Macro idle is 35% of the trace.
2. The `drain` rows show almost all idle time is `starved`; the few sync rows add
   up to about 0.1%. Synchronization isn't the issue.
3. `CPU_OP_DOMINATED` holds 65% of idle time and `LAUNCH_DOMINATED` 27%. The part
   columns show 53 ms of launch time inside the `CPU_OP_DOMINATED` gaps, so launch
   cost is larger than the `LAUNCH_DOMINATED` row alone suggests.
4. In `idle_groups`, the largest group is `CPU_OP_DOMINATED` under `aten::addmm`
   (24% of idle), followed by an attention processor's `__call__` and
   `aten::mm`. The `LAUNCH_DOMINATED` time is spread over many small groups of
   short gaps.
5. Filter `idle_intervals` by the `aten::addmm` group and open its worst gaps in
   the augmented trace: the GPU finishes the previous kernels while the CPU is
   still inside `aten::addmm`, before it launches the GEMM.
6. Conclusion: the host can't keep up with many short kernels. Reducing CPU
   overhead around the linear layers, fusing, or capturing the model in a graph is
   likely to reduce idle time.

### LLM serving: remove the sync

The following example is a vLLM serving trace with 89% GPU utilization:

1. Utilization is 89%; macro idle is 31 ms.
2. One drain row holds 68% of idle time: `sync_drain: hipMemcpyWithStream @
   vllm/v1/attention/backends/rocm_aiter_fa.py(459): build`, 22 ms over 128 gaps.
   Its `cpu_op_ms` is 20 ms: after each copy drains the queue, the CPU runs ops
   for a while before the next launch, and the GPU waits. The Step 2 bound
   (`sync_gain_ms`) is 22.0 ms: each copy blocked for longer than its gap, so removing the wait
   would let that CPU work run while the GPU is still busy.
3. The label is `CPU_OP_DOMINATED`, which agrees: the gaps are CPU work after the
   sync.
4. In `idle_groups`, the same `top_runtime` (`hipMemcpyWithStream`) appears on
   several `sync_drain` groups with different `covering_op` values
   (`execute_model`, `build`, `_build_attn_group_metadata`): that's the CPU work
   that follows the one sync, not separate causes.
5. `sync_call_stack_full` in `idle_intervals` shows the copy is a `.cpu()` call in
   the attention backend's metadata `build`.
6. Conclusion: making the copy asynchronous (pinned memory and a later sync), or
   moving it after the next launches, lets the metadata build overlap with GPU
   work, without making the build itself faster.

## Performance overhead

Idle time analysis adds classification and DataFrame construction on top of the
normal trace load. The following table shows measurements on representative
PyTorch traces.

| Trace | Size (gzip) | Macro gaps | Load | Idle analysis |
|-------|-------------|------------|------|---------------|
| ResNet-18 DDP training | 0.6 MB | 696 | 0.6 s | 0.5 s |
| Llama 70B FP8 inference | 1.7 MB | 141 | 1.2 s | 0.6 s |
| LLM serving (vLLM) | 5.5 MB | 501 | 5.8 s | 3.4 s |
| Diffusion model inference | 21 MB | 14,882 | 23.7 s | 15.7 s |

To reduce the cost on very large traces, raise the noise threshold
(`--micro_idle_thresh_us`, or `noise_us` in `IdleThresholds`) so fewer short gaps
are classified.

## Related topics

- [Idle time sheet reference](../reference/idle-time-columns.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Trace2Tree data model](../conceptual/trace2tree.md)
- [Analyze traces with the SDK](./sdk-analysis.md)
