## TraceFusion

In distributed deep learning, diagnosing issues like straggling ranks, load imbalance, or bottlenecks requires a **global view** of events across all ranks. TraceFusion is a Python SDK for merging trace files across ranks in distributed training and inference setups. With customization options for filtering events and defining file paths, TraceFusion simplifies the preparation of traces for seamless rendering in **Perfetto**.
Note that: TraceFusion is **only for visual analysis in PerfettoUI** and not for automated analysis. 

---

## Key Features

- **Custom Filtering**: Easily define filtering logic to include or exclude specific events. For example, merge traces for a subset of ranks, focus only on GPU events, or narrow down further to NCCL kernel events only.
- **PyTorch Support**: Built primarily for PyTorch trace files, with potential support for other frameworks in the future.
- **Lightweight and Simple**: A dependency-free and straightforward codebase makes it easy to integrate and extend.

---

## Quick Start

Here’s how to use TraceFusion to merge and process trace files for distributed training or inference:

### Example 1: Basic Usage

```python
from iLoveTrace import TraceFuse
import os

# Define file paths for each rank
root_profiles = '/path/to/profiles/'
world_size = 8
list_profile_files = [os.path.join(root_profiles, f'pytorch_profile_rank{i}_step120.json') for i in range(world_size)]

# Initialize TraceFusion
fuser = TraceFuse(list_profile_files)

# Merge and Save traces
output_file = os.path.join(root_profiles, 'merged_trace_all_events.json')
fuser.merge_and_save(output_file)

# By default, Python function category events are skipped to save memory.
# To include them, set include_pyfunc=True.
```

### Example 2: Advanced Usage

```python
from iLoveTrace import TraceFuse
import os

# Define file paths for rank 0 on each node
world_size = 64
profile_files = {i: os.path.join('/path/to/profiles/', f'pytorch_profile_rank{i}_step120.json') for i in range(0, world_size, 8)}

# Initialize TraceFusion
fuser = TraceFuse(profile_files)

# Custom filter for NCCL kernels
def filter_nccl_kernels(event):
    return ('cat' in event and 'args' in event and event['cat'] in ['kernel', 'gpu_user_annotation'] and 'nccl' in event['name'])

# Merge and Save traces
output_file = '/path/to/profiles/merged_trace_nccl.json'
fuser.merge_and_save(output_file, filter_fn=filter_nccl_kernels)
```

---

### What's Inside?

TraceFusion merges `traceEvents` across ranks by:
1. **Appending Events**: Combines all events from multiple ranks into a single list.
2. **Adjusting Process IDs**: Modifies `pid` so that traces for each rank render correctly in the UI.
3. **Correcting Flow Linking**: Updates `External id` for events and `id` for corresponding `ac2g` events to ensure accurate flow linking in the UI.

These adjustments ensure seamless visualization in **Perfetto**, with clear rank separation and correct flow rendering.
