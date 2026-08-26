<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Configure and run the PyTorch profiler

```{meta}
:description: Learn how to configure and run the PyTorch built-in profiler for deep learning models, from basic timeline capture to scheduling long training runs.
:keywords: TraceLens, PyTorch profiler, torch.profiler, GPU trace, Chrome trace, ROCm, AMD Instinct, performance profiling, CUDA, schedule, record_shapes, with_stack
```

Learn how to use PyTorch's built-in profiler. Use it analyze and diagnose performance bottlenecks in deep learning models.

By the end of this guide, you'll be able to:

- Configure and run the profiler for both short and long training runs.
- Understand key profiler options such as `record_shapes` and `with_stack`.
- Generate Chrome trace timelines.
- Use profiler scheduling and trace handlers to profile long-running training jobs efficiently.

This guide uses a ResNet18 model with synthetic data to stay focused on profiling behavior—not model convergence or accuracy.

## Set up the model and environment

Set up the model and device for profiling:

```python
import torch
import torchvision.models as models

device = "cuda"
dtype = torch.bfloat16
model = models.resnet18().to(device).to(dtype)
```

## Define the training step

Create a fixed random batch and define a single reusable training step:

```python
B, C, H, W = 5, 3, 224, 224
num_classes = 1000

dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)
dummy_target = torch.randn(B, num_classes, device=device, dtype=dtype)


def train_step():
    output = model(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    loss.backward()


# test it out
train_step()
```

## Warm up the CUDA context

Run a short warm-up loop to initialize the CUDA context before recording:

```python
def warm_up(iters: int = 10):
    for _ in range(iters):
        train_step()
    torch.cuda.synchronize()
```

## Capture a basic trace timeline

This section shows how to capture a minimal execution trace using PyTorch's profiler. The `activities` argument specifies which device events to track:

- `ProfilerActivity.CPU` – records CPU-side operator execution.
- `ProfilerActivity.CUDA` – records GPU kernel launches and durations.

```python
from torch.profiler import profile, ProfilerActivity

warm_up()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    train_step()
p.export_chrome_trace("trace_minimal.json")
```

Open the generated JSON file in [Perfetto](https://ui.perfetto.dev/).

## Record tensor shapes

Enable `record_shapes` to capture the tensor dimensions seen by each operator:

```python
warm_up()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as p:
    train_step()
p.export_chrome_trace("trace_shapes.json")
```

## Include Python stack traces

Add `with_stack=True` to record the Python call stack alongside each operator, making it easier to trace GPU activity back to its origin in your model code:

```python
warm_up()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as p:
    train_step()
p.export_chrome_trace("trace_stack.json")
```

```{note}
`with_stack=True` notably increases the trace file size. Switch it off for large profiling runs.
```

## Profile long training runs with a schedule

When profiling longer training runs, capturing every iteration is too expensive and unnecessary. PyTorch's `schedule()` allows fine-grained control over when to record and when to skip. Here's what the arguments mean:

- `wait`: Number of iterations to skip before profiling begins.
- `warmup`: Profiler enabled but not yet saving traces — it measures kernel timings internally so the system reaches steady-state, but these iterations are discarded.
- `active`: Number of iterations to record traces for.
- `repeat`: How many times to repeat the `wait → warmup → active` cycle.

This lets you profile windows of activity in long runs without generating massive trace files.

By default, `repeat=0`, which means the profiler continues executing `wait → warmup → active` cycles *indefinitely* until the job ends. Setting `repeat=1` means only one such cycle is run.

```python
from torch.profiler import schedule

sched_wait, sched_warmup, sched_active, sched_repeat = 10, 5, 3, 2
sched = schedule(
    wait=sched_wait, warmup=sched_warmup, active=sched_active, repeat=sched_repeat
)


def trace_handler(p):
    # Called at the end of the active window.
    # p.step_num is the last iteration of the active window.
    start = p.step_num - sched_active + 1
    end = p.step_num
    p.export_chrome_trace(f"trace_iter{start}_{end}.json")


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=sched,
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler,
) as p:
    warm_up()
    for _ in range(100):
        train_step()
        p.step()  # marks iteration boundary
```

## Related topics

- [Understanding PyTorch traces](../../conceptual/torch-profiling-analysis.md)
- [Generate a PyTorch performance report](../generate-perf-report-pytorch.md)
- [Investigate PyTorch training performance anomalies](./anomaly-detection.md)
- [Profile distributed PyTorch workloads](./distributed-profiling.md)
