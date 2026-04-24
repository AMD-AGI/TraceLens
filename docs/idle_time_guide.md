<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GPU Idle Time Analysis — User Guide

This guide explains how to interpret TraceLens idle-time classification in performance reports and augmented traces. It is aimed at readers who are familiar with ML training workflows but not with low-level GPU execution details.

---

## 1. Background Concepts

### CPU–GPU asynchronous execution

The CPU submits work to the GPU by enqueueing kernel launches (and similar operations) into one or more GPU queues. The GPU consumes that queue independently. While the GPU runs kernels, the CPU can continue preparing and enqueueing more work, so the two processors overlap in time.

If the CPU stops submitting new work and the queue empties, the GPU has nothing left to run and becomes **idle** until the next launch arrives. Long or frequent idle stretches mean the accelerator is not doing useful computation during that window.

### GPU queue and launch latency

After the CPU issues a launch (for example via `hipLaunchKernel` or `cudaLaunchKernel`), there is a short delay before the kernel begins executing on the GPU. That delay is **launch latency**; on typical platforms it is often on the order of roughly 5–20 microseconds, depending on driver, runtime, and workload. If the CPU keeps the queue full, successive kernels can start back-to-back and this per-launch gap is largely hidden.

### Synchronization and queue draining

Operations such as `hipDeviceSynchronize`, `hipStreamSynchronize`, event waits, or blocking device-to-host copies force the CPU to wait until the GPU reaches a defined point. While the CPU waits, it usually cannot enqueue additional kernels, so the GPU queue may drain and the GPU can sit idle even though the host process is still “busy” from a logical standpoint.

### Why idle time matters

Idle GPU time is time the hardware is not applying to your model. In training, reducing idle time often improves step time and throughput. Common contributors include CPU overhead between operators, synchronization points, allocator or runtime stalls, and cases where the host cannot feed the GPU fast enough.

---

## 2. Quick Start

Three typical entry points:

```bash
# As part of a perf report
python -m TraceLens.Reporting.generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis

# With augmented trace for Perfetto visualization
python -m TraceLens.Reporting.generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --enable_idle_analysis \
    --enable_augmented_trace

# Python API
from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser
from TraceLens import TreePerfAnalyzer
pa = TreePerfAnalyzer.from_file("trace.json")
analyser = IdleTimeAnalyser(pa.tree)
dfs = analyser.get_dataframes()
```

The Python API returns a mapping of sheet names to pandas `DataFrame` objects (for example `idle_overview`, `idle_summary`, `idle_intervals`), consistent with the Excel report.

---

## 3. Reading the Report (Walkthrough)

### Step 1: `idle_overview`

Start here for a coarse picture. Check **`gpu_utilization_pct`** (or the equivalent summary of GPU busy versus idle time for your trace window). If utilization is already very high (for example above roughly 95%), idle time is unlikely to be the primary bottleneck; other limits (memory, kernel duration, algorithmic cost) may dominate.

Use the breakdown by **`drain_type`** and **`cpu_during_gap`** to see which combination of “why the queue emptied” and “what the CPU was doing” accounts for most idle time.

### Step 2: `idle_summary`

Sort or scan by **`cumulative_pct`** and total time. The first rows after the aggregate row usually show where to invest effort.

- If **`LAUNCH_ANOMALY`** dominates, the issue is along the GPU dispatch path (slow pickup of queued work, or unusually large launch-to-start gaps), not necessarily slow Python on the host.
- If **`CPU_DOMINATED`** dominates, framework or operator overhead on the CPU is the main story; the **`dominant_op`** column narrows the target.
- **`RUNTIME_DOMINATED`** points at runtime API activity (allocation, synchronization, launch stalls); inspect the detail fields for the specific call or sub-type.

### Step 3: `idle_intervals`

Use this sheet to inspect individual gaps. **`idle_id`** ties each interval to annotations in an augmented Perfetto trace (for example labels of the form `idle#N`). **UID** columns (`following_gpu_uid`, `following_launch_uid`, `sync_event_uid`, and similar) link rows to events in TraceLens’s Trace2Tree model so you can recover call stacks and surrounding context for deep dives.

---

## 4. Classification Reference

| Category | Meaning | Typical action |
|----------|---------|----------------|
| `LAUNCH_ANOMALY` | GPU slow to pick up the next kernel relative to expectations | Consider CUDA/HIP graphs or other batching of dispatch; investigate driver/platform behavior for persistent micro-gaps |
| `LAUNCH_OVERHEAD_ONLY` | Gap explained by normal launch latency; CPU submitted work promptly | Usually not actionable beyond accepting inherent overhead |
| `RUNTIME_DOMINATED` | Runtime API call (for example malloc, sync, launch stall) occupies a large share of the gap | Depends on sub-type; reduce sync, allocator churn, or the specific API hotspot |
| `CPU_DOMINATED` | CPU-side framework or operator work dominated the gap | Optimize or fuse the dominant op; reduce host-side overhead |
| `CPU_UNTRACED` | CPU activity during the gap is not well represented in the trace | Re-profile with Python function tracing or stacks enabled so self-time attribution is reliable |

For column-level definitions and annotation track details, see `docs/idle_time_classification.md`.

---

## 5. Output Formats

- **Excel / DataFrames**: Structured tables suitable for sorting, filtering, and programmatic use (including automation or agent-assisted analysis). UIDs support cross-referencing with the in-memory tree.
- **Augmented trace** (`--enable_augmented_trace`): The original Chrome trace is extended with idle annotation tracks aligned to the GPU process timeline in Perfetto, so idle gaps appear in context next to surrounding kernels and runtime activity.

---

## 6. Example Workflow

**Scenario:** “I see about 30% of idle time attributed to `LAUNCH_ANOMALY` on MI325X — what should I conclude?”

1. Open **`idle_overview`**: confirm that a substantial fraction of total idle falls under `LAUNCH_ANOMALY` (and that overall idle is worth fixing given **`gpu_utilization_pct`**).
2. Open **`idle_summary`**: check whether most of that time is grouped under **`prequeued`** intervals — meaning the host launched the kernel in time, but the GPU still exhibited a larger-than-expected gap before execution started.
3. Open **`idle_intervals`**: note typical **`duration_us`** (for example many gaps of roughly 5–15 microseconds between kernels).
4. **Conclusion:** the pattern is consistent with GPU-side dispatch or scheduling overhead rather than a single slow CPU op. Mitigations may include CUDA/HIP graphs (or similar) to amortize launch and submission, and platform or driver follow-up if the gap is uniform and limits throughput.

---

## 7. Performance Overhead

Enabling idle time analysis adds classification and DataFrame construction on top
of the normal trace load.  Measured on representative PyTorch traces:

| Trace | Size | Intervals | Load | Classify + DF | Overhead |
|-------|------|-----------|------|---------------|----------|
| ResNet-26t (single GPU) | 1.8 MB | 154 | 0.09 s | 0.81 s | ~9x (dominated by first-time numpy init) |
| ResNet (training) | 5.3 MB | 548 | 0.47 s | 0.20 s | ~43% |

The first-run overhead includes one-time numpy/pandas import costs.  On
subsequent calls within the same process the overhead is closer to 20-40% of
trace load time for medium traces.

For very large traces (> 100 MB, thousands of idle intervals) classification can
take several minutes.  Consider using the `micro_thresh_us` parameter to filter
noise intervals early or running analysis in a separate process with a timeout.

---

## See also

- `docs/idle_time_classification.md` — full column reference and classification axes.
