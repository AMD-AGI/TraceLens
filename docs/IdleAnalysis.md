# GPU Idle Time Analysis

## Overview

TraceLens can classify every idle gap on a GPU stream by its root cause. Rather than reporting GPU idle time as a single number, the idle time analyser breaks it down into categories that explain *why* the GPU was idle, giving engineers actionable information for optimisation.

The analysis is per-stream: for each GPU stream, it looks at every gap between consecutive GPU events (kernels, memcpy, memset) and classifies the gap by correlating it with the CPU-side call stack, CUDA/HIP runtime events, and the kernel launch timeline.

## Key Concept: The Pipelined Launch Model

Modern GPU programming uses an asynchronous launch model. The CPU issues kernel launches via runtime APIs (`cudaLaunchKernel` / `hipLaunchKernel`), which enqueue work onto a GPU stream. The GPU executes kernels from the queue independently. When the CPU stays ahead of the GPU (i.e., the next launch is issued before the current kernel finishes), the GPU pipeline stays full and gaps are minimal.

The entire classification algorithm hinges on one question:

> **Was the CPU-side launch for the next kernel issued *before* or *after* the gap started?**

This divides every gap into one of two paths:

```
For each gap between kernel[i] and kernel[i+1] on the same stream:

    Find the CPU-side launch event for kernel[i+1]

    if launch.ts <= gap_start:
        --> PATH A: Launch was pipelined (GPU-side gap)
    else:
        --> PATH B: Launch was late (CPU-side gap)
```

### Path A: Launch Was Pipelined (GPU-Side Gap)

The CPU did its job -- it issued the launch before the previous kernel even finished. The gap exists because the GPU itself couldn't start the next kernel immediately. This means the root cause is on the GPU side, and we should **not** blame CPU-side events (like a sync call on another thread) that merely happen to overlap temporally.

### Path B: Launch Was Late (CPU-Side Gap)

The CPU didn't issue the launch until after the previous kernel had already finished. The GPU was starved -- it had nothing to execute. The root cause is on the CPU side, and we investigate *what* the CPU was doing during the gap that delayed the launch.

This two-path split is critical for avoiding false classifications. Without it, a long-running `hipEventSynchronize` on a profiling thread (which doesn't block kernel launches at all) would be falsely blamed for every tiny inter-kernel gap that happens to overlap with it temporally.

## Classification Taxonomy

### Path A Classifications (GPU-side gaps)

#### A1. Cross-Stream Dependency (`cross_stream_dep`)

A `cudaStreamWaitEvent` / `hipStreamWaitEvent` was issued during the gap interval. This API makes a GPU stream wait until an event recorded on another stream completes. Stream-wait affects the GPU stream directly regardless of which CPU thread issued it.

**How detected:** Find `cuda_runtime` events with names matching stream-wait patterns whose time interval overlaps the gap.

**Typical cause:** Multi-stream workloads where one stream depends on another's output (e.g., a compute stream waiting for a data transfer stream).

#### A2. Launch Overhead (`launch_overhead`)

The gap is small (below a configurable threshold, default 10us) and the launch was pipelined. This is the irreducible cost of the GPU hardware dispatch pipeline -- the time between one kernel finishing and the next starting, even when work is already queued.

**How detected:** `gap_duration <= launch_overhead_thresh_us` and the launch was pipelined.

**Typical cause:** Normal GPU operation. These gaps are expected and typically sub-microsecond to single-digit microseconds. A high count with tiny durations is healthy.

#### A3. Scheduler Saturation (`scheduler_saturation`)

The launch was pipelined, but the gap is larger than the launch overhead threshold and no stream-wait was found. The CPU issued the launch in time, the GPU just didn't start the kernel promptly.

**How detected:** Pipelined launch + gap exceeds threshold + no stream-wait found.

**Typical cause:** GPU hardware scheduler contention -- too many pending kernels across streams, resource conflicts (registers, shared memory), or hardware queue depth limits. Can also indicate resource-intensive kernels that prevent the next kernel from starting.

### Path B Classifications (CPU-side gaps)

For Path B, we investigate what was happening on the **same CPU thread** as the kernel launch. This is important because CUDA/HIP runtime APIs block only the *calling* thread. An API running on thread A does not prevent `hipLaunchKernel` on thread B.

#### B1. Cross-Stream Dependency (`cross_stream_dep`)

Same as A1, but found on the late-launch path. A stream-wait was active during the gap even though the launch was late.

#### B2. Memory Operation Stall (`memory_op_stall`)

A synchronous memory API was active on the launch thread during the gap, blocking it from issuing the next launch.

**How detected:** Find `cuda_runtime`/`cuda_driver` events with memory-pattern names that (a) overlap the gap interval and (b) ran on the **same (pid, tid)** as the launch event.

**Matched API names include:**
- `cudaMemcpy` / `hipMemcpy` (synchronous variants)
- `cudaMalloc` / `hipMalloc`
- `cudaFree` / `hipFree`
- `cudaMemset` / `hipMemset`
- And their async/2D/3D variants

**Typical cause:** Synchronous host-device data transfers, runtime memory allocation (which may trigger device synchronisation internally), or memory deallocation.

#### B3. CPU Bottleneck (`cpu_bottleneck`)

The launch was late but no specific blocking API (memory, stream-wait) was found on the launch thread. The CPU was simply busy doing other work -- framework overhead, Python execution, tensor metadata computation, etc.

**How detected:** Fallback when the launch was late and no specific blocking API matched. The analyser reports the CPU delay (how late the launch was) and the nearest `cpu_op` ancestor in the call stack to help identify what the CPU was doing.

**Typical cause:** Heavy CPU-side computation between kernel launches (e.g., complex control flow, dynamic shape computation, Python overhead in eager mode). Also common for operators that are CPU-bound (sorting, data-dependent indexing).

### Special Case: Unknown (`unknown`)

The next GPU event has no linked CPU-side launch event (common for `gpu_memset` and `gpu_memcpy` operations that are issued via runtime APIs other than `cudaLaunchKernel` and may not have ac2g correlation links in the trace).

## Classification Decision Flowchart

```
                    ┌─────────────────────────────┐
                    │  Gap between GPU event[i]    │
                    │  and GPU event[i+1]          │
                    │  on the same stream          │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  Find launch event for        │
                    │  GPU event[i+1]               │
                    └──────────────┬───────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Launch found?    │
                          └───┬─────────┬───┘
                          No  │         │ Yes
                              │         │
                  ┌───────────▼──┐  ┌───▼──────────────────┐
                  │ Check for    │  │ launch.ts <= gap_start│
                  │ stream wait  │  │ (was it pipelined?)   │
                  │ → cross_     │  └───┬──────────────┬───┘
                  │   stream_dep │    Yes│              │No
                  │ else:        │      │              │
                  │ → unknown    │      │              │
                  └──────────────┘  ┌───▼────┐   ┌────▼────────┐
                                    │ PATH A │   │   PATH B    │
                                    │GPU-side│   │  CPU-side   │
                                    └───┬────┘   └────┬────────┘
                                        │             │
                        ┌───────────────▼──┐   ┌──────▼───────────┐
                        │ Stream wait?     │   │ Stream wait?     │
                        │ → cross_stream_  │   │ → cross_stream_  │
                        │   dep            │   │   dep            │
                        │                  │   │                  │
                        │ Small gap?       │   │ Memory op on     │
                        │ → launch_        │   │ same thread?     │
                        │   overhead       │   │ → memory_op_     │
                        │                  │   │   stall          │
                        │ Large gap?       │   │                  │
                        │ → scheduler_     │   │ Else:            │
                        │   saturation     │   │ → cpu_bottleneck │
                        └──────────────────┘   └──────────────────┘
```

## The Same-Thread Check

A subtle but important detail: CUDA/HIP runtime APIs and memory operations block only the **calling CPU thread**. A `hipEventSynchronize` on thread 42 does not prevent `hipLaunchKernel` from being called on thread 100.

In real traces, profiling infrastructure often calls `hipEventSynchronize` on a background thread for timing purposes. Without the same-thread check, these profiling-related calls would be falsely blamed for every gap that happens to overlap them temporally -- potentially misclassifying hundreds of thousands of gaps.

The analyser only blames memory APIs when they ran on the **same (pid, tid)** as the launch event for the next kernel. Cross-stream waits (`cudaStreamWaitEvent`) are exempt from this check because they affect the GPU stream directly, regardless of which CPU thread issued them.

## Usage

```python
from TraceLens import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

tree = TraceToTree(events)
perf = TreePerfAnalyzer(tree)

# Detailed per-gap classification
idle_df = perf.get_df_idle_analysis()

# Summary breakdown by reason
summary = perf.get_idle_summary_df()

# Top N largest gaps
top = perf.get_top_idle_gaps(n=10)

# Filter to a specific stream
idle_stream7 = perf.get_df_idle_analysis(stream_id=7)

# Adjust the launch overhead threshold (default 10us)
idle_df = perf.get_df_idle_analysis(launch_overhead_thresh_us=20.0)
```

## Output Columns

### `get_df_idle_analysis()` -- one row per gap

| Column | Description |
|---|---|
| `stream` | GPU stream ID |
| `gap_start` | Timestamp (us) where the previous GPU event ended |
| `gap_end` | Timestamp (us) where the next GPU event started |
| `duration_us` | Gap duration in microseconds |
| `reason` | Classification label (see taxonomy above) |
| `prev_gpu_event_uid` | UID of the GPU event before the gap |
| `prev_gpu_event_name` | Kernel/memcpy/memset name before the gap |
| `next_gpu_event_uid` | UID of the GPU event after the gap |
| `next_gpu_event_name` | Kernel/memcpy/memset name after the gap |
| `launch_event_uid` | UID of the CPU-side launch for the next kernel |
| `launch_event_name` | Name of the launch API (e.g., `hipLaunchKernel`) |
| `cpu_op_ancestor_uid` | UID of the nearest `cpu_op` ancestor of the launch |
| `cpu_op_ancestor_name` | Name of the cpu_op (e.g., `aten::matmul`) |
| `details` | Human-readable explanation of the classification |

### `get_idle_summary_df()` -- aggregated by reason

| Column | Description |
|---|---|
| `reason` | Classification label |
| `count` | Number of gaps with this reason |
| `total_time_us` | Total idle time for this reason |
| `mean_duration_us` | Average gap duration |
| `max_duration_us` | Largest single gap |
| `pct_of_total_idle` | Percentage of total idle time |

## Note on Per-Stream vs. Global Idle

This analysis is **per-stream**: it counts every gap on every stream independently. This means the total idle time across all streams will generally be **larger** than the global GPU idle time reported by `get_df_gpu_timeline()`, because when stream A is idle but stream B is busy, this analysis counts stream A's gap while the global metric does not (since the GPU as a whole was not idle).

Both views are useful:
- **Per-stream idle** tells you how well-utilised each individual stream is and what's causing each stream to stall.
- **Global idle** (from `get_df_gpu_timeline()`) tells you when the entire GPU had no work at all.
