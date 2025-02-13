# TraceFusion

In distributed deep learning, diagnosing issues like straggling ranks, load imbalance, or bottlenecks requires a **global view** of events across all ranks. TraceFusion is a Python SDK for merging trace files across ranks in distributed training and inference setups. With customization options for filtering events and defining file paths, TraceFusion simplifies the preparation of traces for seamless rendering in **Perfetto** .

---

## Key Features

- **Custom Filtering**: Easily define filtering logic to include or exclude specific events. For example, merge traces for a subset of ranks, focus only on GPU events, or narrow down further to NCCL kernel events only.
- **Customizable File Mapping**: Flexible rank-to-file mapping to match your directory structure and naming conventions.
- **PyTorch Support**: Currently built for PyTorch trace files, with the potential to extend support to other frameworks.
- **Lightweight and Simple**: No dependencies and straightforward codebase make it easy to integrate and extend.

---

## Quick Start

Here’s how to use TraceFusion to merge and process trace files for distributed training or inference:

### Example 1: Merge All Events from Selected Ranks

```python
from TraceFusion.trace_fuse import TraceFuse
import os

profiles_root_dir = '/path/to/profiles'
world_size = 32
# Initialize TraceFusion
fuser = TraceFuse(profiles_root_dir, world_size)

# Define rank-to-file mapping
def rank2file(rank):
    return os.path.join(profiles_root_dir, f'pytorch_profile_rank{rank}_step120.json')
fuser.set_rank2file_fn(rank2file)

# Merge and Save traces
output_file = os.path.join(profiles_root_dir, 'merged_trace_all_events_rank0-7.json')
ranks_to_merge = range(8)
fuser.merge_and_save(ranks_to_merge, output_file)
```

### Example 2: Merge Only NCCL Kernels from All Ranks

```python
from TraceFusion.trace_fuse import TraceFuse
import os

profiles_root_dir = '/path/to/profiles/'
world_size = 32

# Initialize TraceFusion
fuser = TraceFuse(profiles_root_dir, world_size)

# Custom filter for NCCL kernels
def filter_nccl_kernels(event):
    if 'cat' not in event or 'args' not in event:
        return False
    if event['cat'] not in ['kernel', 'gpu_user_annotation']:
        return False
    if 'nccl' not in event['name']:
        return False
    return True
fuser.set_filter(filter_nccl_kernels)

# Define rank-to-file mapping
def rank2file(rank):
    return os.path.join(profiles_root_dir, f'pytorch_profile_rank{rank}_step120.json')
fuser.set_rank2file_fn(rank2file)

# Merge and Save traces
output_file = os.path.join(profiles_root_dir, 'merged_trace_nccl.json')
ranks_to_merge = range(world_size)
fuser.merge_and_save(ranks_to_merge, output_file)
```

### What's Inside?

TraceFusion merges `traceEvents` across ranks by:  
1. **Appending Events**: Combines all events from multiple ranks into a single list.   
2. **Adjusting Process IDs**: Modifies `pid` to so that the traces for each rank is rendered correctly in the UI.  
3. **Correcting Flow Linking**: Updates `External id` for events and `id` for corresponding `ac2g` events to ensure accurate flow linking in the UI. 

These adjustments ensure seamless visualization in **Perfetto**, with clear rank separation and correct flow rendering.

