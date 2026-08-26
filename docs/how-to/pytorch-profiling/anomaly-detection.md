<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Investigate PyTorch training performance anomalies

```{meta}
:description: Learn how to detect and capture anomalous training steps in PyTorch using always-on CUDA-only profiling with threshold-based selective trace capture.
:keywords: TraceLens, PyTorch profiler, anomaly detection, GPU trace, performance monitoring, ROCm, AMD Instinct, torch.profiler, schedule, CUDA, training diagnostics
```

Some workloads have anomalous steps — most steps are fine, but every so often one is 2×, 5×, or 50× slower. A fixed-window profiler almost never catches them.

Trace every step continuously with CUDA-only activities, flush the buffer every step, and only write to disk when the step exceeds a threshold. Saved traces are tight — one anomaly equals one small trace, with no surrounding context.

This tutorial walks through the pattern with a ResNet18 example, a synthetic anomaly injector, and the TraceLens perf-report generator. It builds on [Configure and run the PyTorch profiler](./torch-profiling.md).

## Set up your environment

Set up the imports, model, optimizer, and synthetic data used throughout this tutorial.

```python
import random
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, schedule

device = "cuda"
dtype = torch.bfloat16
torch.manual_seed(0)
random.seed(0)

model = models.resnet18().to(device).to(dtype)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

B, C, H, W = 5, 3, 224, 224
num_classes = 1000
dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
dummy_target = torch.randn(B, num_classes, device=device, dtype=dtype)
```

## Simulate training anomalies

For a runnable example, `train_step()` does a normal forward pass, backward pass, and optimizer step and an extra GPU matmul on a random ~5% of steps. Tensors are pre-allocated so the anomaly is pure compute, not allocator pressure. The detection loop treats `train_step()` as a black box — you call it and time it, you don't know which calls will be anomalous.

```python
EXTRA_MATMUL_DIM = 8192
EXTRA_MATMUL_PROB = 0.05
EXTRA_MATMUL_COUNT = 2

extra_a = torch.randn(EXTRA_MATMUL_DIM, EXTRA_MATMUL_DIM, device=device, dtype=dtype)
extra_b = torch.randn(EXTRA_MATMUL_DIM, EXTRA_MATMUL_DIM, device=device, dtype=dtype)


def train_step() -> None:
    optimizer.zero_grad(set_to_none=True)
    output = model(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    loss.backward()
    optimizer.step()

    if random.random() < EXTRA_MATMUL_PROB:
        x = extra_a
        for _ in range(EXTRA_MATMUL_COUNT):
            x = x @ extra_b
        x.sum()
```

## Establish a timing baseline

Run a few hundred steps without the profiler, plot them, and take the median as the baseline. Set `threshold = 1.5 × median`. Each step is timed with `time.perf_counter()` plus a trailing `torch.cuda.synchronize()` so the timer accounts for all GPU work.

```python
# Warm up so CUDA context init and autotune don't show up as outliers.
for _ in range(5):
    train_step()
torch.cuda.synchronize()

DIAGNOSTIC_STEPS = 200
step_durs_ms: list[float] = []
for _ in range(DIAGNOSTIC_STEPS):
    t0 = time.perf_counter()
    train_step()
    torch.cuda.synchronize()
    step_durs_ms.append((time.perf_counter() - t0) * 1000.0)

baseline_median = statistics.median(step_durs_ms)
THRESHOLD_MULTIPLIER = 1.5
threshold_ms = baseline_median * THRESHOLD_MULTIPLIER
print(f"baseline median: {baseline_median:.2f} ms   threshold (1.5x): {threshold_ms:.2f} ms")

fig, ax = plt.subplots(figsize=(11, 4))
x = list(range(len(step_durs_ms)))
above_x = [i for i, d in enumerate(step_durs_ms) if d > threshold_ms]
above_y = [step_durs_ms[i] for i in above_x]
ax.plot(x, step_durs_ms, lw=0.8, color="#1f77b4", label="step duration")
ax.axhline(threshold_ms, color="red", lw=1.2, ls="--", label=f"threshold ({threshold_ms:.1f} ms)")
ax.axhline(baseline_median, color="green", lw=1.0, ls=":", label=f"baseline median ({baseline_median:.1f} ms)")
ax.scatter(above_x, above_y, color="red", s=30, zorder=5, label=f"above threshold ({len(above_x)})")
ax.set_xlabel("step")
ax.set_ylabel("duration (ms)")
ax.set_title("Per-step duration timeline")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Points above the threshold (red) sit well above the baseline. If your timeline shows a noisy band with no clear gap between the baseline and the high points, the workload is jittery rather than anomalous and threshold capture won't help much.

## Configure per-step profiler flushing

Each saved trace should contain *only* the anomalous step's events — no surrounding healthy steps. To achieve this, make the profiler flush its buffer on *every* `prof.step()` boundary, then let the handler decide whether to keep the file.

`schedule(wait=0, warmup=0, active=1)` returns `RECORD_AND_SAVE` every step, which flushes the buffer to `on_trace_ready` and resets on every `prof.step()`. Each handler call sees exactly one step's events.

```{note}
PyTorch prints a `Profiler won't be using warmup` warning with this schedule. This is safe to ignore — the CUDA context was already warmed up in the baseline section. See the [Configure and run the PyTorch profiler](./torch-profiling.md) for the `schedule()` factory's general form.
```

Create the schedule:

```python
sched = schedule(wait=0, warmup=0, active=1)
```

## Filter traces in the handler

The handler decides whether the just-finished step is worth keeping. At the end of each loop iteration, push `(step, dur)` onto a FIFO. The handler pops one entry per invocation and writes to disk only if the duration exceeds the threshold. Non-anomalous steps fall through — their events are flushed from the in-memory buffer and discarded by the profiler, never serialized to JSON, and never touching disk.

```python
OUTPUT_DIR = Path("./anomaly_traces")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
for old in OUTPUT_DIR.glob("anomaly_*.json"):
    old.unlink()
for old in OUTPUT_DIR.glob("anomaly_*.xlsx"):
    old.unlink()

step_info_queue: list[tuple[int, float]] = []
saved_files: list[Path] = []


def trace_handler(prof):
    if not step_info_queue:
        return
    step, duration_ms = step_info_queue.pop(0)
    if duration_ms <= threshold_ms:
        return  # not anomalous — discard, nothing written to disk
    out = OUTPUT_DIR / f"anomaly_step{step:05d}_dur{duration_ms:.1f}ms.json"
    prof.export_chrome_trace(str(out))
    saved_files.append(out)
    print(f"    saved: {out.name}  ({out.stat().st_size / 1024:.1f} KB)")
```

## Run the always-on profiling loop

Generic training loop: time each step, push `(step, dur)`, call `prof.step()`. Detection is purely duration-based — the training loop has no knowledge of which steps will be anomalous. Use `activities=[ProfilerActivity.CUDA]` only (no CPU op decode, no Python stacks) to keep per-step overhead small enough for always-on profiling. The handler fires synchronously inside `prof.step()` and either writes the just-finished step's trace or returns silently.

```python
NUM_STEPS = 200
print(f"Running {NUM_STEPS} steps with always-on CUDA-only profiling, threshold={threshold_ms:.2f} ms\n")

loop_start = time.perf_counter()
with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=sched,
    on_trace_ready=trace_handler,
    record_shapes=False,
    with_stack=False,
) as prof:
    for global_step in range(NUM_STEPS):
        step_start = time.perf_counter()
        train_step()
        torch.cuda.synchronize()
        dur = (time.perf_counter() - step_start) * 1000.0
        step_info_queue.append((global_step, dur))
        prof.step()

elapsed = time.perf_counter() - loop_start
print(f"\nWall time: {elapsed:.2f} s ({elapsed * 1000 / NUM_STEPS:.2f} ms/step including profiler overhead)")
```

Example output:

```text
Running 200 steps with always-on CUDA-only profiling, threshold=7.90 ms

    saved: anomaly_step00020_dur23.4ms.json  (830.6 KB)
    saved: anomaly_step00043_dur23.0ms.json  (830.6 KB)
    ...

Wall time: 2.68 s (13.42 ms/step including profiler overhead)
```

## Analyze anomaly traces with TraceLens

Each saved file contains exactly one anomalous step's CUDA events — small, focused, and ready to drop into [Perfetto](https://ui.perfetto.dev/) or feed to the TraceLens [perf-report generator](../generate-perf-report-pytorch.md). For CUDA-only traces, TraceLens emits `gpu_timeline` and `kernel_summary` (the op-level sheets need CPU events). Pass `include_unlinked_kernels=True` so the GPU timeline counts kernels that have no host call-stack to link against.

```python
from TraceLens.Reporting.generate_perf_report_pytorch import generate_perf_report_pytorch

target_trace = saved_files[0]
print(f"Generating perf report for: {target_trace.name}\n")

dfs = generate_perf_report_pytorch(
    profile_json_path=str(target_trace),
    include_unlinked_kernels=True,
    kernel_summary=True,
)
print(f"\nSheets returned: {list(dfs.keys())}")
```

Example output:

```text
Generating perf report for: anomaly_step00020_dur23.4ms.json

Building tree with add_python_func=False
Detected GPU-only trace. Skipping CPU-dependent analysis and generating only GPU timeline and kernel summary.
DataFrames successfully written to anomaly_traces/anomaly_step00020_dur23.4ms_perf_report.xlsx

Sheets returned: ['gpu_timeline', 'kernel_summary']
```

### Inspect the GPU timeline

Compute vs idle vs communication vs memory-copy breakdown for the anomalous step. A compute-heavy anomaly points at extra work; high idle points at host stalls; a communication anomaly points at a slow collective.

```python
df_timeline = dfs["gpu_timeline"].copy()
df_timeline["time ms"] = df_timeline["time ms"].round(3)
df_timeline["percent"] = df_timeline["percent"].round(2)
df_timeline
```

| Type | Time (ms) | Percent |
|------|-----------|---------|
| computation_time | 22.479 | 96.92 |
| exposed_comm_time | 0.000 | 0.00 |
| exposed_memcpy_time | 0.000 | 0.00 |
| busy_time | 22.479 | 96.92 |
| idle_time | 0.713 | 3.08 |
| total_time | 23.192 | 100.00 |
| total_comm_time | 0.000 | 0.00 |
| total_memcpy_time | 0.000 | 0.00 |

### Inspect the kernel summary

Top kernels by total time within the anomalous step.

```python
rename_map = {
    "Kernel duration (µs)_sum": "total (µs)",
    "Kernel duration (µs)_count": "count",
    "Kernel duration (µs)_mean": "mean (µs)",
    "Percent of total time (%)": "% total",
}
df_kern = (
    dfs["kernel_summary"]
    .sort_values("Kernel duration (µs)_sum", ascending=False)
    .head(10)
    .copy()
)
df_kern["kernel (truncated)"] = df_kern["Kernel name"].str.slice(0, 60) + "..."
df_kern_view = (
    df_kern[["kernel (truncated)", *rename_map.keys()]]
    .rename(columns=rename_map)
    .round({"total (µs)": 1, "mean (µs)": 2, "% total": 2})
)
```

Because the trace contains only the anomalous step, the kernels responsible sit at the very top — small `count`, large `sum`. A synthetic 8192×8192 bfloat16 matmul lights up exactly that way. The same data is also saved as `.xlsx`.

## Summary of the detection pattern

The following steps summarize the anomaly detection approach used in this tutorial.

- Time each step with `time.perf_counter()` plus a trailing `cuda.synchronize()`.
- Run one short diagnostic pass to set `threshold = 1.5 × median`.
- Use `schedule(wait=0, warmup=0, active=1)` so the buffer flushes every step.
- The trace handler pops `(step, dur)` from a single-entry FIFO and writes to disk only if `dur > threshold`.
- Use `activities=[ProfilerActivity.CUDA]` for always-on profiling; the training loop is generic and unaware of detection.

Once you have per-anomaly trace files, run them through `generate_perf_report_pytorch` or the CLI `TraceLens_generate_perf_report_pytorch`.

## Related topics

- [Configure and run the PyTorch profiler](./torch-profiling.md)
- [Generate a PyTorch performance report](../generate-perf-report-pytorch.md)
- [Understanding PyTorch traces](../../conceptual/torch-profiling-analysis.md)
