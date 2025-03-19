# GPUEventAnalyser

GPUEventAnalyser is a reusable component designed to analyze GPU timeline and extract key performance metrics. While it is used within TreePerf, it can also be used independently.

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
import sys
from TraceLens import GPUEventAnalyser

path = sys.argv[1] # this expects the JSON file (unzipped)

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

## Jax support
Jax describes events slightly differently, and includes all GPUs in a single trace, using the JaxGPUEventAnalyser class.
When using GPUEventAnalyser for JAX traces (enabled by passing "jax=True" to the analysis functions),
the compute_metrics() and get_breakdown_df() methods will return events from GPU 0.
To access the traces for all devices, call get_gpu_event_lists_jax() to obtain a dictionary of
{pid: event_lists} where pid is the process id (1-n for n GPUs, a number >100 for CPU the CPU event list).
To create a Pandas dataframe for each GPU, call get_breakdown_df_multigpu(), which will return a dictionary of
{gpu_index: DataFrame} for gpuindex in [0, num_gpus).
The inherited function get_breakdown_df will return the results from GPU 0

### Jax Usage Example
```python
import gzip
import json
import sys
from TraceLens import JaxGPUEventAnalyser

path = sys.argv[1] # this expects the zipped JSON file produced by the

with gzip.open(path, 'r') as fin:
    data = json.loads(fin.read().decode('utf-8'))

events = data['traceEvents']
my_gpu_event_analyser = JaxGPUEventAnalyser(events)
for gpu, df in my_gpu_event_analyser.get_breakdown_df_multigpu().items():
    print(f"GPU {gpu}")
    print(df)
```

## Customizing for Other Profiling Formats

To adapt GPUEventAnalyser for other profiling formats, subclass it and reimplement the `get_gpu_event_lists()` method to correctly extract GPU events.
See the class JaxGPUEventAnalyser for an example.

---

