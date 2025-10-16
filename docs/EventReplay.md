<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Event Replay

Optimizing GPU performance in deep learning requires isolating and benchmarking individual operations to identify bottlenecks. However, reproducing operations directly from complex model code or large profiles can be cumbersome.

Event Replay is a Python-based tool within TraceLens that extracts and replays almost arbitrary PyTorch operations using minimal, portable Intermediate Representation (IR). It enables users to easily reproduce, analyze, and benchmark specific operators independently from the original model execution, streamlining performance optimization workflows.

---

## Key Features

- **Generic Operator Replay**: Reconstructs and benchmarks any PyTorch operator from profile data, including convolutions, GEMMs, reductions, element-wise operations, and more.
- **Minimalistic IR**: Extracts essential operator attributes (tensor shapes, strides, dtypes, and other arguments) into a lightweight, portable JSON-based IR.
- **Portable Artifacts**: Enables sharing standalone artifacts (JSON IR and scripts) with teammates or upstream repositories without requiring access to the model or TraceLens.

---


## Quick Start

### Example: Replay a Single Event

```python
from TraceLens import TreePerfAnalyzer, EventReplayer

# Load profile and get event
perf_analyzer = TreePerfAnalyzer.from_file('/path/to/profile.json')
uid = 12345  # Replace with actual UID of interest
event = perf_analyzer.tree.get_UID2event(uid)

# Initialize and replay
replayer = EventReplayer(event, device='cuda')
replayer.replay()
```

---

## Batch Replay and Benchmark

### Extract Operator IR from TraceLens Profiles

```python
import json

# Extract replay IR for events of interest
repro_data = [EventReplayer(event, lazy=True).get_repro_info() for event in events_of_interest]

with open('event_replay_ir.json', 'w') as f:
    json.dump(repro_data, f, indent=4)
```

```bash
python batched_replay.py event_replay_ir.json
```

#### Example Output

```
[7/11] Replaying: aten::convolution
  Reconstructing arguments for 'aten::convolution'...
  Positional Args:
  input Tensor: {'shape': [20, 128, 28, 28], 'dtype': 'c10::BFloat16', 'strides': [100352, 784, 28, 1]}
  weight Tensor: {'shape': [256, 128, 3, 3], 'dtype': 'c10::BFloat16', 'strides': [1152, 9, 3, 1]}
  bias Tensor?: None
  stride SymInt[]: [2, 2]
  padding SymInt[]: [1, 1]
  dilation SymInt[]: [1, 1]
  transposed bool: False
  output_padding SymInt[]: [0, 0]
  groups SymInt: 1
  Keyword Args:
  Average time taken: 100.38 microseconds
  Successfully executed aten::convolution.
  Result: Tensor(shape=torch.Size([20, 256, 14, 14]), dtype=torch.bfloat16, device=cuda:0)

[8/11] Replaying: aten::convolution
  Reconstructing arguments for 'aten::convolution'...
  Positional Args:
  input Tensor: {'shape': [20, 256, 14, 14], 'dtype': 'c10::BFloat16', 'strides': [50176, 196, 14, 1]}
  weight Tensor: {'shape': [512, 256, 3, 3], 'dtype': 'c10::BFloat16', 'strides': [2304, 9, 3, 1]}
  bias Tensor?: None
  stride SymInt[]: [2, 2]
  padding SymInt[]: [1, 1]
  dilation SymInt[]: [1, 1]
  transposed bool: False
  output_padding SymInt[]: [0, 0]
  groups SymInt: 1
  Keyword Args:
  Average time taken: 92.83 microseconds
  Successfully executed aten::convolution.
  Result: Tensor(shape=torch.Size([20, 512, 7, 7]), dtype=torch.bfloat16, device=cuda:0)
...
--- Replay Summary ---
Total operations in file: 11
Attempted replays: 11
Successful replays: 11
Errors encountered: 0

```
-------------------
### Creating Standalone Replay Artifacts

You can optionally package the extracted replay IR and scripts into a standalone zip file for easy sharing and reproduction, independent of the original model code or TraceLens repository.

Artifacts included:
- `event_replay_ir.json`: Serialized operator replay instructions.
- `utils.py`: Tensor creation and helper utilities.
- `batched_replay.py`: Script to batch replay and benchmark operations.
- `batched_replay_readme.md`: Instructions for running the replay.

Example packaging code:

```python
import zipfile
import os
from TraceLens.EventReplay import utils as tl_utils
from TraceLens.EventReplay import batched_replay

files = [
    OUTPUT_REPRO_FILE,
    tl_utils.__file__,
    batched_replay.__file__,
    batched_replay.__file__.replace('batched_replay.py', 'batched_replay_readme.md')
]

zip_file_path = '/path/to/replay_code.zip'
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    for file in files:
        zipf.write(file, arcname=os.path.basename(file))

print(f"Created zip file: {zip_file_path}")
```
---

## Use Cases

- **Performance Debugging**: Quickly isolate and reproduce performance issues from large models.
- **Regression Testing**: Automate benchmarks to detect performance regressions at the operator level.
- **Kernel Development**: Extract minimal reproducers for GPU kernel optimization and debugging.
- **Numerical Validation**: Evaluate numerical correctness and stability of isolated operations across hardware.
- **Hardware Counter Profiling**: Use with hardware counters to analyze performance bottlenecks in specific operations.

---

## Notes

- Event Replay uses randomized data based on extracted tensor shapes; thus, replay timings approximate real-world performance.
