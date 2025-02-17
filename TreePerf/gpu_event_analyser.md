# GPUEventAnalyser

GPUEventAnalyser is a reusable component designed to analyze GPU profiling traces and extract key performance metrics. While it is used within TreePerf, it can also be used independently to analyze GPU execution time, compute and communication breakdowns, and exposed latencies.

---

## Key Features

1. **GPU Timeline Breakdown**: Computes key GPU activity metrics, including:
   - **Computation Time**: Time spent in computation kernels (e.g., matrix multiplications, convolutions).
   - **Communication Time**: Time spent in communication kernels (e.g., NCCL operations for distributed training).
   - **Memcpy Time**: Time spent in memory copy operations between host and device or across devices.
   - **Idle Time**: Periods where the GPU is not executing any computation, communication, or memcpy operations.
   - **Exposed Communication**: Communication time that does not overlap with computation.
   - **Exposed Memcpy**: Memcpy time that does not overlap with computation or communication.

2. **Reusable Across Profiling Formats**: Although GPUEventAnalyser is designed for PyTorch's JSON trace format, it can be adapted to other profiling formats by inheriting the class and reimplementing `get_gpu_event_lists()`.

---

## Usage Example

```python
import json
from gpu_event_analyser import GPUEventAnalyser

path = '/path/to/profile.json'

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
| busy_time            | 4345.36   | 99.80     |
| idle_time            | 8.53      | 0.20      |
| total_time           | 4353.88   | 100.00    |
| total_comm_time      | 292.92    | 6.73      |
| total_memcpy_time    | 0.19      | 0.00      |

---

## Customizing for Other Profiling Formats

To adapt GPUEventAnalyser for other profiling formats, subclass it and reimplement the `get_gpu_event_lists()` method to correctly extract GPU events.

**TODO: Add an example demonstrating subclassing for a different profiling format.**

---

