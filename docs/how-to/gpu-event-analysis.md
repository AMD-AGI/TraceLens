<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analyze a GPU timeline with GPUEventAnalyser
```{meta}
:description: Learn how to use the TraceLens GPUEventAnalyser component standalone to extract GPU timeline metrics such as computation, communication, memcpy, and idle time from a PyTorch or JAX trace.
:keywords: TraceLens, GPUEventAnalyser, JaxGPUEventAnalyser, GPU timeline, computation time, communication time, memcpy, idle time, exposed communication, PyTorch profiler, JAX, ROCm
```

`GPUEventAnalyser` is a reusable component designed to analyze a GPU timeline and extract key performance metrics. While it's used within [TreePerf](../how-to/tree-perf-analysis.md), it can also be used independently, and this topic focuses on that standalone usage — you don't need the CPU-side tree to run it.

## Before you begin

- TraceLens installed (see [Install TraceLens](../install/install.md)).
- A trace file containing GPU events: an unzipped PyTorch profiler JSON trace, or a gzipped JAX trace for the JAX workflow.

## Key features

- *GPU timeline breakdown*: computes key GPU activity metrics, including:
   - *Computation time*: time spent in computation kernels (for example, matrix multiplications, convolutions).
   - *Communication time*: time spent in communication kernels (for example, NCCL operations for distributed training).
   - *Memcpy time*: time spent in memory copy operations between host and device or across devices.
   - *Idle time*: periods where the GPU is not executing any computation, communication, or memcpy operations.
   - *Exposed communication*: communication time that does not overlap with computation.
   - *Exposed memcpy*: memcpy time that does not overlap with computation or communication.
- *Reusable across profiling formats*: although `GPUEventAnalyser` is designed for PyTorch's JSON trace format, it can be adapted to other profiling formats by inheriting the class and reimplementing `get_gpu_event_lists()`.

## Analyze a PyTorch trace (standalone)

`GPUEventAnalyser` works directly from a list of trace events, so you can run it on its own without building the CPU-side tree. Load the trace JSON, pass its `traceEvents` to `GPUEventAnalyser`, and call `get_breakdown_df()`:

```python
import json
import sys
from TraceLens import GPUEventAnalyser

path = sys.argv[1]  # this expects the JSON file (unzipped)

with open(path, 'r') as f:
    data = json.load(f)

events = data['traceEvents']
my_gpu_event_analyser = GPUEventAnalyser(events)
df = my_gpu_event_analyser.get_breakdown_df()
print(df)
```

Example output:

| type                  | time ms   | percent   |
| --------------------- | --------- | --------- |
| computation_time      | 4184.32   | 96.10     |
| exposed_comm_time     | 160.85    | 3.69      |
| exposed_memcpy_time   | 0.19      | 0.00      |
| busy_time             | 4345.36   | 99.80     |
| idle_time             | 8.53      | 0.20      |
| total_time            | 4353.88   | 100.00    |
| total_comm_time       | 292.92    | 6.73      |
| total_memcpy_time     | 0.19      | 0.00      |

## Analyze a JAX trace

JAX describes events slightly differently, and includes all GPUs in a single trace, using the `JaxGPUEventAnalyser` class. When you use `GPUEventAnalyser` for JAX traces (enabled by passing `jax=True` to the analysis functions), the `compute_metrics()` and `get_breakdown_df()` methods return events from GPU 0.

To access the traces for all devices, call `get_gpu_event_lists_jax()` to obtain a dictionary of `{pid: event_lists}`, where `pid` is the process id (1-n for n GPUs, a number greater than 100 for the CPU event list). To create a Pandas dataframe for each GPU, call `get_breakdown_df_multigpu()`, which returns a dictionary of `{gpu_index: DataFrame}` for `gpuindex` in `[0, num_gpus)`. The inherited `get_breakdown_df` function returns the results from GPU 0.

```python
import gzip
import json
import sys
from TraceLens import JaxGPUEventAnalyser

path = sys.argv[1]  # this expects the zipped JSON file produced by the profiler

with gzip.open(path, 'r') as fin:
    data = json.loads(fin.read().decode('utf-8'))

events = data['traceEvents']
my_gpu_event_analyser = JaxGPUEventAnalyser(events)
for gpu, df in my_gpu_event_analyser.get_breakdown_df_multigpu().items():
    print(f"GPU {gpu}")
    print(df)
```

## Customize for other profiling formats

To adapt `GPUEventAnalyser` for other profiling formats, subclass it and reimplement the `get_gpu_event_lists()` method to correctly extract GPU events. See the `JaxGPUEventAnalyser` class for an example.

## Related topics

- [Analyze GPU performance with TreePerf](../how-to/tree-perf-analysis.md)
- [Trace2Tree](../conceptual/trace2tree.md)
- [Generate a JAX performance report](../how-to/generate-perf-report-jax.md)
- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
- [API reference](../reference/api-reference.md)
