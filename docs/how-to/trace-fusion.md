<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Fuse multi-rank traces in TraceLens
```{meta}
:description: Learn how to merge per-rank PyTorch traces from a distributed run into a single file for cross-rank visualization in Perfetto using TraceLens.
:keywords: TraceLens, TraceFuse, trace fusion, multi-rank, distributed training, Perfetto, PyTorch profiler, ROCm, visualization, straggler
```


This guide shows how to merge per-rank PyTorch traces from a distributed run into
a single file that can be visualized together in Perfetto, using the `TraceFuse`
SDK.

## Prerequisites

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- Per-rank PyTorch profiler traces (one trace per rank).

## Why fuse traces

By default, each rank in a distributed run produces its own trace. Inspecting
them separately makes it hard to diagnose cross-rank issues — straggling ranks,
load imbalance, or a rank stalling on a collective — because you can't see how
ranks line up on a common timeline. `TraceFuse` merges the per-rank traces into a
single file so all ranks render together in Perfetto, with clear per-rank CPU/GPU
separation and correct flow linking.

`TraceFuse` is **for visual analysis in the Perfetto UI only** — not for
automated analysis. For quantitative cross-rank metrics, use a
[collective-communication report](./collective-report.md) instead.

## Step 1: Fuse the traces

Pass `TraceFuse` the per-rank trace files (a list ordered by rank, or a
`{rank: filepath}` dict), then call `merge_and_save`:

```python
import os
from TraceLens import TraceFuse

profiles_root_dir = "path/to/your/profiles"
world_size = 8
list_profile_filepaths = [
    os.path.join(profiles_root_dir, f"rank_{i}.json") for i in range(world_size)
]

fuser = TraceFuse(list_profile_filepaths)

output_file = os.path.join(profiles_root_dir, "merged_trace.json")
fuser.merge_and_save(output_file)
```

A ready-to-edit version of this script is at
[`examples/trace_fusion_example.py`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/trace_fusion_example.py).

**Expected output:** a single merged trace file containing the events from all
ranks, aligned on a common timeline with per-rank `RANK N - CPU` / `RANK N - GPU`
process labels.

```{note}
Python-function category events (`python_function`) are skipped by default to
save memory. To keep them, pass `include_pyfunc=True` to `merge_and_save`.
```

### Speed up loading with multiprocessing

Loading many large per-rank traces is the slow part. Enable parallel loading
(speedup is system-dependent):

```python
# Defaults to os.cpu_count() workers
fuser = TraceFuse(list_profile_filepaths, use_multiprocessing=True)

# Or cap the worker count
fuser = TraceFuse(list_profile_filepaths, use_multiprocessing=True, max_workers=32)
```

### Filter the events you merge

For large runs, merging every event from every rank is heavy. Pass a `filter_fn`
to `merge_and_save` to keep only the events you care about — for example, NCCL
kernels:

```python
def filter_nccl_kernels(event):
    return (
        event.get("cat") in ["kernel", "gpu_user_annotation"]
        and "nccl" in event.get("name", "").lower()
    )

fuser.merge_and_save(output_file, filter_fn=filter_nccl_kernels)
```

You can also narrow the run itself — pass a dict to merge only a subset of ranks
(for instance, rank 0 of each node on a 64-rank job):

```python
profile_files = {
    i: os.path.join(profiles_root_dir, f"rank_{i}.json")
    for i in range(0, 64, 8)
}
fuser = TraceFuse(profile_files)
```

## Step 2: Visualize in Perfetto

Open the merged trace in the [Perfetto UI](https://ui.perfetto.dev/) (or the
Chrome trace viewer) to inspect cross-rank behavior, such as exposed
communication and synchronization gaps. Each rank appears as separate
`RANK N - CPU` and `RANK N - GPU` tracks.

## What the merge does

`TraceFuse` combines the `traceEvents` from each rank into one list and adjusts
them so they render correctly together:

- **Process IDs** (`pid`) are offset per rank so ranks don't collide in the UI,
  and `process_name` metadata is added to label each track `RANK N - CPU` /
  `RANK N - GPU`.
- **Flow linking** is preserved by offsetting the linking key (`correlation` or
  `External id`) and the `ac2g` flow `id`s per rank, so CPU→GPU launch arrows
  stay correct.

## Next steps

- For quantitative collective analysis instead of visualization, see
  [Generate a collective-communication report](./collective-report.md).
- Profile a single rank's operations with a
  [PyTorch performance report](./generate-perf-report-pytorch.md).

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [Compare two traces](./compare-traces.md)
- [API reference](../reference/api-reference.md)
