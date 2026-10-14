<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens performance report column reference
```{meta}
:description: Reference for every sheet and column in a TraceLens performance report, including the gpu_timeline breakdown, operation-analysis sheets, roofline metrics, and collective communication analysis.
:keywords: TraceLens, performance report, column reference, gpu_timeline, ops, ops_summary, ops_unique_args, roofline, GEMM, SDPA, convolution, NCCL, coll_analysis, GFLOPS, TFLOPS, arithmetic intensity, ROCm, AMD Instinct, MI300X
```

This topic explains the columns in each sheet of the TraceLens performance report. The report is an Excel file with multiple sheets, and each sheet analyzes a different aspect of GPU performance. Use it as a lookup while you read a report generated with one of the how-to topics listed under [Related topics](#related-topics).

The performance report Excel file contains multiple sheets analyzing different aspects of GPU performance. The core sheets are:

| Sheet | Description |
|-------|-------------|
| `gpu_timeline` | High-level GPU time breakdown. |
| `ops` | Detailed per-operation data (base data). |
| `ops_summary_by_category` | Operations summarized by category. |
| `ops_summary` | Operations summarized by name. |
| `ops_unique_args` | Operations summarized by unique argument combinations. |

Additional sheets can include operation-specific analysis (GEMM, SDPA_fwd, CONV_fwd, and so on), kernel summary, short kernels, and collective analysis.

### Unit conventions

- **Time**: All times from the trace are in microseconds (µs) unless explicitly stated otherwise (for example, `time ms` in `gpu_timeline` is in milliseconds).
- **Compute**: GFLOPS (billions of FLOPs), TFLOPS/s (trillions of FLOPs per second).
- **Memory**: MB (megabytes), GB/s (gigabytes per second), TB/s (terabytes per second).

## The `gpu_timeline` sheet

The `gpu_timeline` sheet provides a high-level breakdown of GPU time into computation, communication, memory copy, and idle time, accounting for overlaps between different types of operations.

### Example output

Here's a typical `gpu_timeline` sheet from a distributed training workload:

| type                  | time ms   | percent   |
| --------------------- | --------- | --------- |
| computation_time      | 56305.19  | 99.30     |
| exposed_comm_time     | 240.88    | 0.42      |
| exposed_memcpy_time   | 14.44     | 0.03      |
| busy_time             | 56560.52  | 99.75     |
| idle_time             | 143.16    | 0.25      |
| total_time            | 56703.68  | 100.00    |
| total_comm_time       | 17203.43  | 30.34     |
| total_memcpy_time     | 14.47     | 0.03      |

In this example:

- 99.30% computation time, so the GPU is used efficiently.
- 0.42% exposed communication, which is excellent. Most communication (30.34% total - 0.42% exposed = 29.92%) overlaps with computation.
- 0.25% idle time, so there are minimal gaps and good kernel launch efficiency.
- This workload demonstrates effective computation and communication overlap.

### Event classification

GPU events are classified into three categories based on their `cat` field and kernel name:

```
GPU Event
│
├─ cat = "gpu_memcpy"?
│  └─ YES → Memory Copy (H2D, D2H, D2D)
│
├─ cat = "kernel"?
│  └─ YES → Does kernel name contain "nccl"?
│     ├─ YES → Communication (AllReduce, AllGather, ReduceScatter, etc.)
│     └─ NO  → Computation (GEMM, Conv, Elementwise, etc.)
│
└─ cat = "gpu_memset"?
   └─ YES → Computation (grouped here for simplicity; typically very short)
```

```{note}
- "Computation" includes all kernels doing work (GEMM, convolution, elementwise, and so on), not just compute-bound operations.
- Communication kernels (NCCL) include both synchronization delay and actual data transfer time. In single-rank traces, these components can't be separated, so `total_comm_time` represents the total duration. See the NCCL Analyzer documentation for multi-rank analysis that can break down communication into sync and transfer phases.
```

### How time is calculated

1. **Merge events by category**: Events of the same category are merged across all GPU streams into non-overlapping intervals.

   Example, merging computation events:

   ```
   Stream 0: ──[Kernel A]────────[Kernel B]──
   Stream 1: ─────[Kernel C]──[Kernel D]─────
                   ↓
   Merged:   ──[──────────]──[──────────]──
             (overlapping kernels merged into single intervals)
   ```

2. **Create interval sets**: After merging, there are four sets of non-overlapping intervals:

   - `COMP` = merged computation intervals
   - `COMM` = merged communication intervals
   - `MEMCPY` = merged memcpy intervals
   - `ALL_GPU` = merged intervals of ALL GPU events (computation + communication + memcpy)

3. **Apply set arithmetic**: Exposed metrics are calculated by subtracting overlaps:

   ```
   exposed_comm intervals     = COMM - COMP
   exposed_memcpy intervals   = MEMCPY - COMP - COMM
   ```

   In plain terms:

   - Exposed communication is communication time that doesn't overlap with computation.
   - Exposed memcpy is memcpy time that doesn't overlap with computation or communication.

4. **Sum interval durations**: Time for each metric is the sum of durations of all intervals in that set:

   ```
   computation_time      = sum of durations in COMP
   exposed_comm_time     = sum of durations in (COMM - COMP)
   exposed_memcpy_time   = sum of durations in (MEMCPY - COMP - COMM)
   busy_time            = sum of durations in ALL_GPU
   idle_time            = total_time - busy_time
   total_comm_time      = sum of durations in COMM
   total_memcpy_time    = sum of durations in MEMCPY
   total_time           = end of last GPU event - start of first GPU event
   ```

   The resulting equation:

   ```
   computation_time + exposed_comm_time + exposed_memcpy_time + idle_time = total_time
   ```

### Columns

| Column | Description | Values |
|--------|-------------|--------|
| `type` | Category of GPU time | `computation_time`, `exposed_comm_time`, `exposed_memcpy_time`, `busy_time`, `idle_time` (or `micro_idle_time` and `macro_idle_time`), `total_comm_time`, `total_memcpy_time`, `total_time` |
| `time ms` | Time in milliseconds | Actual duration for each category |
| `percent` | Percentage of total time | `(time ms / total_time) * 100` |

### Interpreting results

- High `computation_time` (>80%): workload is efficiently using the GPU.
- High `exposed_comm_time`: communication isn't overlapped with computation, so optimize with computation and communication overlap.
- High `exposed_memcpy_time`: memory transfers are blocking work, so optimize data movement.
- High `idle_time`: the GPU is sitting idle, so check for CPU bottlenecks, kernel launch overhead, or synchronization.

Generated by `TreePerfAnalyzer.get_df_gpu_timeline()` using `GPUEventAnalyser.compute_metrics()`.

## Operation analysis

The following sections analyze CPU operations that launch GPU kernels. Each operation is analyzed for its direct GPU time impact.

### The ops sheet (base data)

The `ops` sheet provides detailed per-operation data showing each CPU operation that launches GPU kernels. This is the base unsummarized data; the following sheets provide different ways of summarizing this information. Each row represents one CPU operation instance from your workload that launched GPU kernels, showing exactly which kernels it launched and how long they took.

Generated by `TreePerfAnalyzer.get_df_kernel_launchers()`.

#### Why analyze operations instead of kernels

When you look at a PyTorch trace, you might be tempted to analyze GPU kernels directly. However, this has significant limitations:

- Kernel names are cryptic and ambiguous. A kernel named `Cijk_Ailk_Bljk_BBS_BH_Bias_HAS_SAV...` tells you almost nothing about what computation it's doing. The same kernel name can also map to different computations depending on input shape, dtype, and memory layout.
- Kernel names vary across platforms. The same matrix multiply appears as `nvjet_*` or `cutlass_*` on NVIDIA GPUs, but as `Cijk_*` on AMD GPUs. This makes cross-platform comparison nearly impossible.
- Operations are stable and meaningful. Names like `aten::addmm` (matrix multiply with add) or `aten::conv2d` are platform-independent and immediately tell you what computation is happening. Combined with argument information (shapes, dtypes, strides), they fully define the computation in a reproducible way.

By analyzing at the operation level, you get insights that are portable across platforms, interpretable without deep kernel knowledge, and reproducible.

#### Which operations are analyzed

TraceLens analyzes leaf operations, the lowest-level CPU operations in the call stack that directly launch GPU kernels. This gives you the most granular view without double-counting.

Example call stack (showing all layers from the trace):

```
└── (python_function) nn.Module: Linear_75
    └── (python_function) forward @ linear.py:124
        └── (cpu_op) aten::linear
            ├── (cpu_op) aten::to
            │   └── (cpu_op) aten::_to_copy
            │       └── (cpu_op) aten::copy_                    ← Leaf op
            │           └── (cuda_runtime) hipLaunchKernel      ← Runtime layer
            │               └── (kernel) elementwise_kernel...   ← GPU kernel
            ├── (cpu_op) aten::to
            │   └── (cpu_op) aten::_to_copy
            │       └── (cpu_op) aten::copy_                    ← Leaf op
            │           └── (cuda_runtime) hipLaunchKernel      ← Runtime layer
            │               └── (kernel) elementwise_kernel...   ← GPU kernel
            └── (cpu_op) aten::linear
                └── (cpu_op) aten::addmm                        ← Leaf op
                    ├── (cuda_runtime) hipLaunchKernel          ← Runtime layer
                    │   └── (kernel) elementwise_kernel...       ← GPU kernel
                    └── (cuda_runtime) hipExtModuleLaunchKernel ← Runtime layer
                        └── (kernel) Cijk_Alik_Bljk_BBS_BH...   ← GPU kernel
```

```{note}
The `cuda_runtime` label is used by the PyTorch profiler even on ROCm platforms. It's just a naming convention.
```

In this example:

- Leaf operations analyzed: `aten::copy_` (2 instances) and `aten::addmm`.
- Not included: `aten::linear`, `aten::to`, `aten::_to_copy` (these are higher-level and don't directly launch kernels).
- Why this matters: if higher-level operations were included, the same GPU time would be counted multiple times. By focusing on leaf operations, each kernel's time is attributed to exactly one operation.

#### What CPU operations (cpu_op) are

When you write PyTorch code in Python (like `output = model(input)`), it goes through several layers before reaching the GPU. The trace captures all these layers:

1. *Python frontend*: your Python code calls `nn.Module` methods (labeled `python_function` in the trace).
2. *Torch dispatcher*: operations are routed through PyTorch's dispatcher, which selects the appropriate implementation based on device, dtype, and other factors (labeled `cpu_op` in the trace).
3. *Runtime layer*: CUDA/HIP API calls to launch kernels (labeled `cuda_runtime` or `cuda_driver` in the trace).
4. *GPU kernels*: the actual computation on the GPU (labeled `kernel`, `gpu_memcpy`, or `gpu_memset` in the trace).

Operations marked as `cpu_op` in the trace are operations registered in the PyTorch dispatcher. These can be registered in C++ or Python, and represent various levels of abstraction in the execution hierarchy. The key distinction is that TraceLens analyzes the ones that directly launch kernels (the leaf operations in the call stack). The dispatcher layer is where PyTorch attaches rich metadata about the computation.

```{note}
In the trace hierarchy there's also a runtime layer (labeled `cuda_runtime` or `cuda_driver`) between CPU operations and GPU kernels. This represents the CUDA/HIP API calls that actually launch kernels. The PyTorch profiler uses the `cuda_runtime`/`cuda_driver` naming convention even on ROCm/HIP platforms. It's a naming convention inherited from CUDA.
```

#### What arguments operations contain

Each operation in the trace contains argument information that fully characterizes the computation. These arguments come directly from the JSON event's `args` field that the PyTorch profiler captures.

Example JSON event from a trace:

```json
{
  "name": "aten::addmm",
  "cat": "cpu_op",
  "ts": 1234567890,
  "dur": 150,
  "args": {
    "Input Dims": [[1024, 512], [512, 256], [256]],
    "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
    "Input Strides": [[512, 1], [256, 1], [1]],
    "Concrete Inputs": ["", "", "1.0"]
  },
  ...
}
```

These arguments are extracted and presented in the `ops` sheet:

| Argument type | Description | Example (from the `aten::addmm` above) | Why it matters |
|---------------|-------------|----------------------------------------|----------------|
| `Input Dims` | Shape of input tensors | `((1024, 512), (512, 256), (256,))` | Different shapes give different performance (tiling, memory access patterns). |
| `Input type` | Data types | `('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16')` | FP32 vs BF16 vs FP16 affects speed and memory. |
| `Input Strides` | Memory layout (row-major, column-major, and so on) | `((512, 1), (256, 1), (1,))` | Strided vs contiguous affects memory bandwidth. |
| `Concrete Inputs` | Scalar/string arguments | `('', '', '1.0')` | Additional parameters that can affect algorithm selection. |

Why shapes matter:

```
aten::addmm with shape (1024, 512) × (512, 256):  Fast (optimized tiling)
aten::addmm with shape (1023, 511) × (511, 255):  Slower (odd dimensions, less efficient)
```

Why strides matter:

```
Same operation, same shape, different strides can have different performance:
  stride=(512, 1)  vs  stride=(1, 512)
```

Strides affect memory access patterns. Some strides work better for certain operations depending on how kernels access data. TraceLens captures strides so you can identify stride-dependent performance variations.

By capturing these arguments, you can group operations by their unique argument combinations. This enables targeted analysis:

- Identify slow cases: "Which specific input shapes are causing slowdowns?"
- Reproduce issues: "I can reproduce the slow case with these exact inputs."
- Compare across platforms: "Does this shape perform better on NVIDIA or AMD?"

This combination of operation name plus arguments forms a unique signature that can be aggregated in the summary sheets (`ops_summary_by_category`, `ops_summary`, `ops_unique_args`), enabling you to ask questions at different levels of granularity.

#### How GPU time is calculated

For each operation, TraceLens calculates `total_direct_kernel_time_us` using GPU Event Analyzer to account for kernel overlaps. When an operation launches multiple kernels, some can run concurrently on different GPU streams. GPU Event Analyzer computes the actual "busy time", the wall-clock time during which at least one of the operation's kernels is executing.

Example:

```
Operation: aten::addmm
  ├── Kernel A: 100 µs (stream 0, executes 0-100 µs)
  └── Kernel B: 80 µs  (stream 1, executes 20-100 µs, overlaps with Kernel A)

Naive sum:              100 + 80 = 180 µs  ✗ (incorrect)
Actual busy time:       100 µs             ✓ (kernels overlapped from 20-100 µs)
```

This gives you the actual wall-clock time each operation's kernels occupied the GPU.

#### Understanding kernel_details

`kernel_details` captures detailed information about each GPU kernel launched by an operation: kernel name, duration (in microseconds), and stream ID.

Example from `aten::addmm`:

```python
[
    {'name': 'elementwise_kernel', 'dur': 2.3, 'stream': 7},      # dur in µs
    {'name': 'Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV...', 'dur': 45.2, 'stream': 7}  # dur in µs
]

# Analysis: addmm = elementwise (2.3 µs) + GEMM (45.2 µs)
# → GEMM is bottleneck (95% of time), both sequential (same stream)
# → Cijk_* = Tensile GEMM (AMD), cutlass_* would be NVIDIA
```

Key insights from `kernel_details`:

- Backend implementation: kernel names reveal which library is used.
  - `Cijk_*`: Tensile GEMM kernels (AMD ROCm)
  - `cutlass_*`: CUTLASS kernels (NVIDIA)
  - `ck_tile::kentry<...FmhaFwd...>`: Composable Kernel (CK) Flash Attention (AMD)
  - `void at::native::*`: PyTorch native kernels (element-wise, and so on)
- Execution breakdown: shows the performance contribution of each kernel in multi-kernel operations.
- Concurrency: same stream = sequential, different streams = potentially concurrent.

#### Columns

| Column | Description | Details |
|--------|-------------|---------|
| `name` | Operation name | PyTorch operation name (for example, `aten::addmm`, `aten::index_select`) |
| `op category` | Operation category | Categorized type (for example, GEMM, CONV_fwd, SDPA_fwd, other) |
| `UID` | Unique event identifier | Unique ID for this specific operation instance in the trace |
| `total_direct_kernel_time` | GPU busy time in microseconds | Wall-clock time the GPU was busy executing this operation's kernels (accounts for overlaps using GPU Event Analyzer) |
| `direct_kernel_count` | Number of kernels launched | How many GPU kernels this operation launched |
| `Input Dims` | Input tensor shapes | Tuple of shapes for each input tensor, for example `((30522, 512), (), (141,))` |
| `Input type` | Input data types | Tuple of data types for each input, for example `('c10::BFloat16', 'Scalar', 'long int')` |
| `Input Strides` | Input tensor strides | Tuple of stride tuples for each input, for example `((512, 1), (), (1,))` |
| `Concrete Inputs` | Scalar/string arguments | Additional arguments passed to the operation, for example `('', '0', '')` |
| `kernel_details` | Kernel execution details | List of dicts with kernel name, duration (µs), and stream for each kernel launched, for example `[{'name': '...', 'dur': 3.136, 'stream': 7}]` |

#### What this data represents

The PyTorch trace has a hierarchical structure (Python → Operations → Runtime → Kernels). This sheet provides a MECE (mutually exclusive, collectively exhaustive) flat representation focusing on the leaf operations that launch GPU work.

- Granularity: each row is a single CPU operation instance from the trace.
- Completeness: only operations that launch GPU kernels are included (pure CPU operations aren't shown).
- Raw data: this is unsummarized, so you see every individual operation occurrence. The following sheets (`ops_summary_by_category`, `ops_summary`, `ops_unique_args`) provide different ways to aggregate this data.

The following three sheets provide different levels of aggregation of the base `ops` data, following a progressive disclosure of complexity.

### The `ops_summary_by_category` sheet (most aggregated)

This is the highest-level summary. It groups operations into broad computational categories (GEMM, convolution, attention, and so on). Use it to quickly identify which types of operations dominate your workload.

#### Columns

| Column | Description | Calculation |
|--------|-------------|-------------|
| `op category` | Operation category | Possible values: GEMM, CONV_fwd, CONV_bwd, SDPA_fwd, SDPA_bwd, BN_fwd, BN_bwd, triton, elementwise, reduce, multi_tensor_apply, other |
| `Count` | Number of operations in this category | Count of unique CPU operations |
| `total_direct_kernel_time_ms` | Total GPU time in milliseconds | Sum of all GPU kernel durations launched by operations in this category |
| `Percentage (%)` | Percentage of total GPU time | `(total_direct_kernel_time_ms / sum of all kernel time) * 100` |
| `Cumulative Percentage (%)` | Running cumulative percentage | Sum of percentages from top to current row |

#### How operations are categorized

Operations are automatically categorized based on name patterns and kernel characteristics. Some categories are detected from the operation name, while others are detected by inspecting the launched kernel names.

| Category | Detection method | Example operations |
|----------|------------------|--------------------|
| GEMM | Operation name | `aten::addmm`, `aten::mm`, `aten::bmm`, `aten::baddbmm` |
| CONV_fwd | Operation name | `aten::convolution`, `aten::miopen_convolution`, `aten::cudnn_convolution` |
| CONV_bwd | Operation name | `aten::convolution_backward` |
| SDPA_fwd | Operation name | `aten::_scaled_dot_product_flash_attention`, `aten::_flash_attention_forward` |
| SDPA_bwd | Operation name | `aten::_scaled_dot_product_flash_attention_backward` |
| BN_fwd | Operation name | `aten::batch_norm`, `aten::native_batch_norm`, `aten::cudnn_batch_norm` |
| BN_bwd | Operation name | `aten::native_batch_norm_backward`, `aten::cudnn_batch_norm_backward` |
| triton | Operation name | Operations starting with `triton` |
| elementwise | Kernel name inspection | Operations launching `at::native` elementwise kernels (for example, `aten::relu`, `aten::add`, `aten::mul`) |
| reduce | Kernel name inspection | Operations launching `at::native` reduce kernels |
| multi_tensor_apply | Kernel name inspection | Operations launching `multi_tensor_apply` kernels |
| other | Default | Operations not matching any above patterns |

Use this sheet when you want a high-level answer to "What types of operations are taking the most time?"

Generated by `TreePerfAnalyzer.get_df_kernel_launchers_summary_by_category()`.

### The `ops_summary` sheet (by operation name)

This is a mid-level summary. It groups operations by their name (for example, all `aten::addmm` calls together, all `aten::conv2d` calls together). Use it when you know which category is expensive and want to see which specific operations within that category are the culprits.

#### Columns

| Column | Description | Calculation |
|--------|-------------|-------------|
| `name` | Operation name | PyTorch operation name (for example, `aten::addmm`, `aten::_flash_attention_forward`) |
| `total_direct_kernel_time_sum` | Total GPU time in microseconds | Sum of all GPU kernel durations launched by this operation type |
| `Count` | Number of operations | Count of instances of this operation |
| `total_direct_kernel_time_ms` | Total GPU time in milliseconds | `total_direct_kernel_time_sum / 1000` |
| `Percentage (%)` | Percentage of total GPU time | `(total_direct_kernel_time_ms / sum of all kernel time) * 100` |
| `Cumulative Percentage (%)` | Running cumulative percentage | Sum of percentages from top to current row |
| `is_recompute` | *(Optional)* Whether this operation is part of activation recomputation | `True` or `False`. Only present when `--detect_recompute` is enabled. The same op name can appear in two rows (one for recompute, one for non-recompute). |

What this sheet shows:

- Each unique operation name gets one row (for example, one row for all `aten::addmm` calls).
- All instances of the same operation are aggregated together, regardless of their input shapes, dtypes, or other arguments.
- More granular than the category view, but still hides variation due to different input arguments.
- When `--detect_recompute` is enabled, rows are further split by recompute status.

Use this sheet when you're asking, "I see GEMM is expensive. Which specific GEMM operation is the problem: `aten::addmm`, `aten::mm`, or `aten::bmm`?"

Generated by `TreePerfAnalyzer.get_df_kernel_launchers_summary()`.

### The `ops_unique_args` sheet (most detailed)

This is the most detailed summary. It groups operations by unique combinations of operation name AND input arguments (shapes, dtypes, strides, concrete inputs). Use it to identify which specific input patterns are causing performance issues.

#### Columns

| Column | Description | Details |
|--------|-------------|---------|
| `name` | Operation name | PyTorch operation name |
| `Input Dims` | Input tensor dimensions | Shape of input tensors (for example, `[[1, 512, 768], [768, 768]]`) |
| `Input type` | Input tensor data types | Data types of inputs (for example, `['c10::BFloat16', 'c10::BFloat16']`) |
| `Input Strides` | Input tensor strides | Memory layout strides for each input tensor |
| `Concrete Inputs` | Scalar input values | Non-tensor inputs like kernel size, stride, padding |
| `operation_count` | Number of occurrences | How many times this exact operation+args combination appeared |
| `total_direct_kernel_time_sum` | Total GPU time in microseconds | Sum of GPU kernel time for all occurrences |
| `total_direct_kernel_time_mean` | Mean GPU time in microseconds | Average GPU time per occurrence |
| `total_direct_kernel_time_median` | Median GPU time in microseconds | Median GPU time across occurrences |
| `total_direct_kernel_time_std` | Standard deviation | Standard deviation of GPU time across occurrences (microseconds) |
| `total_direct_kernel_time_min` | Minimum GPU time | Fastest occurrence (microseconds) |
| `total_direct_kernel_time_max` | Maximum GPU time | Slowest occurrence (microseconds) |
| `ex_UID` | Example event UID | UID of one example event (for further analysis) |
| `kernel_details_summary` | Kernel execution details | Summary of which GPU kernels were launched and their statistics |
| `trunc_kernel_details` | Truncated kernel details | Shortened version of `kernel_details_summary` for readability |
| `Percentage (%)` | Percentage of total GPU time | `(total_direct_kernel_time_sum / sum of all kernel time) * 100` |
| `Cumulative Percentage (%)` | Running cumulative percentage | Sum of percentages from top to current row |
| `is_recompute` | *(Optional)* Whether this operation is part of activation recomputation | `True` or `False`. Only present when `--detect_recompute` is enabled. |

What makes arguments "unique":

- Same operation with different input shapes gives different rows (for example, `aten::addmm` with (1024, 512) vs (2048, 1024)).
- Same operation with different data types gives different rows (for example, BF16 vs FP32).
- Same operation with different strides gives different rows (contiguous vs transposed).
- Each unique combination of (name + Input Dims + Input type + Input Strides + Concrete Inputs) gets one row.

What this sheet shows:

- Statistics for each unique operation+args combination (count, mean, median, std, min, max).
- Which specific input patterns are slow vs fast.
- Performance variation across different calls to the same operation.

Use this sheet when you're asking, "I see `aten::addmm` is expensive. Is it slow for all input shapes, or just specific ones? Are there outliers?"

#### Understanding `kernel_details_summary`

`kernel_details_summary` provides aggregated kernel statistics across all occurrences of an operation+args combination. It shows which kernels are consistently launched and their performance distribution. All durations are in microseconds (µs).

Example for `aten::addmm` (aggregated across 150 occurrences):

```python
[
    {'kernel_name': 'elementwise_kernel', 'count': 150, 'mean_duration_us': 2.3, 'std_dev_duration_us': 0.1},
    {'kernel_name': 'Cijk_Alik_Bljk_BBS_BH...', 'count': 150, 'mean_duration_us': 45.2, 'std_dev_duration_us': 3.5}
]

# Analysis: addmm = elementwise (2.3 µs) + GEMM (45.2 µs) across 150 calls
# → GEMM is bottleneck, with moderate variance (std_dev=3.5) → check for outliers with ex_UID
# → elementwise is consistent (std_dev=0.1)
```

Use this to identify:

- Bottlenecks: which kernel in a multi-kernel operation dominates time. For example, in the GPT-3 XL analysis, `aten::addmm` consistently showed a Tensile GEMM kernel taking 97 µs (mean) while setup kernels took <5 µs each.
- Consistency: low `std_dev` = stable, high `std_dev` = investigate outliers with `ex_UID`. For example, if `std_dev_duration_us` is 15.3 when `mean_duration_us` is 120.5, that's 12.7% variance, worth investigating.
- Backend recipe: which kernels are always launched together (for example, elementwise + GEMM for addmm).
- Call count validation: verify all occurrences launch the same kernels (same `count`).

```{note}
`trunc_kernel_details` provides a shortened version for spreadsheet readability.
```

#### Deep-dive with `ex_UID`

The `ex_UID` column provides a UID of one example event with this operation+arguments combination. You can use it to access the actual event object for deeper analysis:

```python
# Get the event object
event = tree.get_UID2event(ex_UID)

# Analyze call stack and context:
parent = tree.get_parent_event(event)           # Get parent operation
children = tree.get_children_events(event)      # Get child operations
gpu_events = tree.get_gpu_events(event)         # Get launched GPU kernels
tree.traverse_parents_and_print(event)          # See full call stack above this
tree.traverse_subtree_and_print(event)          # See full call stack below this

# Replay the operation for benchmarking:
from TraceLens import EventReplayer
replayer = EventReplayer(event, device='cuda')
replayer.replay()  # Replays the exact operation with same inputs/args
```

This is useful when you want to investigate a specific slow case in detail or benchmark it in isolation. See [Replay a traced operation](../how-to/event-replay.md) for more on replaying operations.

Replaying from the perf report (no trace required): you can also replay operations directly from the Excel report without needing the original trace file:

```python
import pandas as pd
import ast
from TraceLens import EventReplayer

# Read ops from perf report
df = pd.read_excel('perf_report.xlsx', sheet_name='ops_unique_args')

# Convert row to event format
def row_to_event(row):
    return {
        'name': row['name'],
        'args': {
            'Input Dims': ast.literal_eval(row['Input Dims']),
            'Input Strides': ast.literal_eval(row['Input Strides']),
            'Input type': ast.literal_eval(row['Input type']),
            'Concrete Inputs': ast.literal_eval(row['Concrete Inputs']),
        }
    }

# Replay an operation
row = df.iloc[0]  # or filter by name/args
event = row_to_event(row)
replayer = EventReplayer(event, device='cuda')
replayer.replay()

# Get standalone repro artifacts
repro_info = replayer.get_repro_info()  # Returns serializable JSON
```

This is particularly useful for creating standalone reproducers or benchmarking specific operations without the full model or trace. See [`examples/event_replayer_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/event_replayer_example.ipynb) for complete examples including batched replay.

Generated by `TreePerfAnalyzer.get_df_kernel_launchers_unique_args()`.

## Performance metrics sheets (roofline analysis)

For certain operation categories (GEMM, CONV, SDPA, UnaryElementwise, BinaryElementwise), TraceLens generates additional sheets with roofline model metrics. These sheets help you understand how efficiently operations are using the GPU's computational and memory bandwidth capabilities.

```{note}
While hardware counter profilers like `rocprof compute` and `nsight compute` reveal what the GPU actually executed, including effects of padding, redundant memory movement, and cache behavior, TraceLens focuses on the useful work dictated by operator semantics. Used together, these two perspectives provide a richer picture: hardware counters expose low-level execution characteristics, while TraceLens reveals the efficiency of the computation in context.
```

### Understanding the metrics pipeline

The performance metrics are calculated through a 4-step pipeline:

```
Trace Event → Parameter Extraction → Static Metrics + Runtime → Performance Metrics
                                          ↑              ↑
                                    (Compute Model) (GPU Time)
```

#### Step 1: event from trace

Starting with an operation event from the trace JSON:

```json
{
  'name': 'aten::addmm',
  'args': {
    'Input Dims': [[6144], [40960, 1536], [1536, 6144], [], []],
    'Input type': ['c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', 'Scalar', 'Scalar'],
    'Input Strides': [...]
  }
}
```

#### Step 2: parameter extraction

TraceLens uses operation-specific performance models to extract relevant parameters from the event's `args` field:

```python
# For aten::addmm: A @ B + C where A is (M, K), B is (K, N), C is (M, N)
params = {
    'M': 40960,      # From Input Dims[1][0]
    'N': 6144,       # From Input Dims[2][1]  
    'K': 1536,       # From Input Dims[1][1]
    'bias': True,    # C tensor present
    'dtype': 'c10::BFloat16'  # From Input type
}
```

Each operation type (GEMM, Conv, SDPA, and so on) has its own `get_param_details()` method that knows how to extract the relevant parameters from that operation's argument structure. See [Operation parameters reference](#operation-parameters-reference) for details on what gets extracted for each operation type.

#### Step 3: parallel calculation

Once parameters are extracted, two pieces of information are computed in parallel.

Static operation metrics (through the compute model): the performance model calculates the theoretical work based on the operation type:

```python
# GEMM compute model
GFLOPS = (2 * M * N * K) / 1e9                    # 773.35 GFLOPS
Bytes_moved = (M*K + K*N + M*N) * bytes_per_elem  # 618.01 MB (for BF16: 2 bytes/elem)
FLOPS_per_Byte = GFLOPS * 1e9 / Bytes_moved       # 1193.38 (arithmetic intensity)
```

These metrics are static; they depend only on the operation parameters, not on actual execution.

Kernel time (from Trace2Tree): the actual GPU execution time is extracted by:

1. Finding all GPU kernels launched by this operation (using Trace2Tree's hierarchical analysis).
2. Using `GPUEventAnalyser` to compute total busy time, accounting for kernel overlaps.

```python
T = 1884 µs  # Measured kernel execution time
```

#### Step 4: runtime performance metrics

Finally, static work metrics are combined with measured time to produce runtime performance:

```python
TFLOPS/s = GFLOPS / (T / 1e6) = 773.35 / (1884 / 1e6) = 410.48 TFLOPS/s
TB/s = (Bytes_moved / 1e12) / (T / 1e6) = 0.34 TB/s
```

These metrics tell you:

- TFLOPS/s: how many trillion floating-point operations per second were achieved.
- TB/s: how many terabytes per second of memory bandwidth were utilized.

### Why this matters: roofline analysis

The combination of TFLOPS/s, TB/s, and FLOPS/Byte (arithmetic intensity) allows you to perform roofline analysis:

- High FLOPS/Byte (compute-intensive): performance is limited by compute throughput (TFLOPS/s).
  - Example: large GEMM operations (for example, 6144×2048 × 2048×8192).
  - Goal: maximize TFLOPS/s (approach the GPU's peak compute).
- Low FLOPS/Byte (memory-intensive): performance is limited by memory bandwidth (TB/s).
  - Example: element-wise operations, small GEMMs (for example, 2048×2048).
  - Goal: maximize TB/s (approach the GPU's peak bandwidth).

The roofline "knee point": the boundary between memory-bound and compute-bound is determined by the GPU's hardware characteristics:

```
Arithmetic Intensity Threshold = Peak FLOPS / Peak Bandwidth
```

For example:

- MI325X: 1300 TFLOPS / 6000 GB/s = ~217 FLOPs/Byte
- H100: 1000 TFLOPS / 3350 GB/s = ~298 FLOPs/Byte
- MI300X: 1300 TFLOPS / 5300 GB/s = ~245 FLOPs/Byte

Operations with FLOPs/Byte below this threshold are memory-bound; operations above are compute-bound. This is why small GEMMs (FLOPs/Byte ~8-50) are memory-bound, while large GEMMs (FLOPs/Byte ~100-300) are compute-bound on these GPUs. By comparing your achieved TFLOPS/s and TB/s against the GPU's theoretical peaks, you can identify optimization opportunities.

Interpreting performance numbers: understanding what "good" performance looks like requires comparing against theoretical peaks and max-achievable performance:

| GPU | Peak compute (BF16) | Peak memory BW | Example utilization |
|-----|---------------------|----------------|---------------------|
| AMD Instinct™ MI325X | ~1.3 PFLOPS | ~6 TB/s | 500-800 TFLOPS/s = 38-62% of peak (typical for medium GEMMs) |
| H100 | ~1.0 PFLOPS | ~3.35 TB/s | 400-700 TFLOPS/s = 40-70% of peak |
| AMD Instinct™ MI300X | ~1.3 PFLOPS | ~5.3 TB/s | Similar to AMD Instinct™ MI325X |

Understanding theoretical vs. real-world performance: TraceLens uses idealized assumptions that represent upper bounds on performance. The actual roofline model has two key differences from the theoretical one:

1. Peak FLOPS: the theoretical peak represents hardware limits, but real-world applications typically achieve lower performance due to realistic constraints. AMD's Max-Achievable FLOPS (MAF) methodology provides more realistic performance targets. For details, see:
   - [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)
   - [Measuring Max-Achievable FLOPS (Part 2)](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html)
2. Arithmetic intensity: TraceLens assumes 100% cache hit rate, so each memory location is accessed once from global memory, then perfectly cached. For example, in a GEMM, matrices A and B are counted only once even if elements are reused. In reality:
   - Cache misses, evictions, and redundant loads cause actual memory movement to be higher.
   - This means actual arithmetic intensity is lower than TraceLens calculates.
   - Impact: an operation that TraceLens shows as compute-bound (high FLOPs/Byte) might actually be memory-bound in practice.
   - This is a common practice in performance modeling (see [NVIDIA's approach](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem-bounds)).
   - For typical compute-bound operations (large GEMMs, SDPA, convolutions), this limitation has minimal practical impact since they remain compute-bound even with lower arithmetic intensity.

```{note}
Hardware profilers like `rocprof compute` or `nsight compute` measure actual memory transactions and are needed to determine true arithmetic intensity and memory-boundedness.
```

Real examples (from GPT-3 XL on MI325X, BF16):

```
Operation                    TFLOPS/s   TB/s    FLOPs/Byte   Interpretation
──────────────────────────────────────────────────────────────────────────────
mm(256×256 × 256×256)        45        2.1     ~8           Memory-bound, 35% BW utilization
mm(2048×6144 × 6144×2048)    531       0.61    ~138         Compute-bound, 40% efficiency
mm(6144×2048 × 2048×2048)    624       0.71    ~117         Compute-bound, 48% efficiency  
addmm(6144×2048 × 2048×8192) 762       0.59    ~203         Compute-bound, 58% efficiency
```

Understanding compute-bound vs. memory-bound:

- Memory-bound (FLOPs/Byte < ~50): small GEMM with arithmetic intensity of ~8. Performance is limited by memory bandwidth (~2.1 TB/s), not compute. To optimize, focus on improving memory access patterns.
- Compute-bound (FLOPs/Byte > ~100): large GEMMs with high arithmetic intensity. Performance is limited by compute throughput (531-762 TFLOPS/s). The 40% vs 58% efficiency difference reflects kernel optimization quality (tile sizes, wave occupancy), not the compute vs. memory boundedness.

```{note}
Arithmetic intensity (FLOPs/Byte) determines whether an operation is compute-bound or memory-bound. The percentage of peak achieved indicates optimization quality within that constraint. See [NVIDIA's GEMM Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) for more on this distinction.
```

### What these sheets contain

Performance metrics sheets are generated for operations with available performance models:

- GEMM: matrix multiply operations (addmm, mm, bmm, baddbmm, and so on).
- CONV_fwd / CONV_bwd: convolution operations.
- SDPA_fwd / SDPA_bwd: scaled dot-product attention.
- UnaryElementwise / BinaryElementwise: element-wise operations.

Each sheet contains:

- All columns from `ops_unique_args` (operation name, arguments, occurrences, and so on).
- Static metrics: GFLOPS, Data Moved (MB), FLOPS/Byte (calculated once from parameters).
- Compute Spec: combined compute type and precision (for example, `matrix_bf16`, `vector_fp32`). This indicates:
  - `matrix_*`: operations using matrix compute units (GEMM, CONV, SDPA).
  - `vector_*`: operations using vector compute units (elementwise).
  - Precision: `fp8`, `fp16`, `bf16`, `fp32`, `fp64`.
- Roofline metrics (when `--gpu_arch_json_path` is provided): Roofline Time (µs), Pct Roofline; compares achieved time to the theoretical roofline bound.
- Runtime metrics: Kernel Time (µs), TFLOPS/s, TB/s (statistics across all occurrences).
  - `mean`, `median`: central tendency; use median if variance is high.
  - `std_dev`: variability; high `std_dev` (>10% of mean) suggests inconsistent performance.
  - `min`, `max`: range; a large spread can indicate outliers worth investigating.

```{note}
These sheets contain all the columns from `ops_unique_args`, so you can replay operations from these sheets using the same approach described in [Understanding kernel_details_summary](#understanding-kernel_details_summary). Read the desired performance metrics sheet (for example, `GEMM`, `SDPA_fwd`) instead of `ops_unique_args`.
```

Generated by `TreePerfAnalyzer.build_df_perf_metrics()` and `TreePerfAnalyzer.summarize_df_perf_metrics()`.

### Operation parameters reference

TraceLens extracts operation-specific parameters from trace events to calculate theoretical FLOPs and memory traffic. Each operation type has its own parameter extraction logic based on what information is needed for its performance model.

Common sources:

- `Input Dims`: tensor shapes.
- `Input type`: data types.
- `Concrete Inputs`: scalar arguments (stride, padding, and so on).

Data type sizes: `fp64` = 8 bytes, `fp32` = 4 bytes, `bf16/fp16` = 2 bytes, `fp8` = 1 byte.

#### GEMM (matrix multiply)

| Parameter | Meaning | Used in |
|-----------|---------|---------|
| `M` | Number of rows in first matrix | All GEMM ops |
| `N` | Number of columns in second matrix | All GEMM ops |
| `K` | Inner dimension (A cols = B rows) | All GEMM ops |
| `B` | Batch size | bmm, baddbmm |
| `bias` | Whether bias is added | addmm, baddbmm |
| `dtype` | Data type (BFloat16, Float32, and so on) | All GEMM ops |

#### Convolution

| Parameter | Meaning |
|-----------|---------|
| `input_shape` | Input tensor shape (N, C_in, H, W, ...) |
| `filter_shape` | Filter/weight shape (C_out, C_in/groups, kH, kW, ...) |
| `stride` | Stride for convolution (for example, (1, 1)) |
| `padding` | Padding applied (for example, (0, 0)) |
| `dilation` | Dilation factor |
| `groups` | Number of groups for grouped convolution |
| `bias` | Whether bias is present |
| `dtype` | Data type |

#### SDPA (scaled dot-product attention)

| Parameter | Meaning |
|-----------|---------|
| `B` | Batch size |
| `N_Q` | Query sequence length |
| `N_KV` | Key/Value sequence length |
| `H_Q` | Number of query attention heads |
| `H_KV` | Number of key/value heads (for GQA/MQA) |
| `d_h_qk` | Head dimension for Q and K |
| `d_h_v` | Head dimension for V |
| `causal` | Whether causal masking is applied |
| `dropout` | Dropout probability |
| `dtype` | Data type |

```{note}
Different attention implementations use different tensor layouts (BHND vs BNHD).
```

#### Element-wise operations

| Parameter | Meaning |
|-----------|---------|
| `shape` | Tensor shape |
| `dtype` | Data type |

For binary ops (add, mul, and so on), two shapes are extracted and broadcasting is handled automatically.

For the specific formulas used to calculate FLOPs and memory traffic from these parameters, see the performance model implementations in [`TraceLens/PerfModel/perf_model.py`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/PerfModel/perf_model.py).

Extending with custom operations: performance metrics sheets are only generated for operations that have a registered performance model. If your workload includes custom operations (for example, from Megatron, vLLM, or other libraries), you can extend TraceLens by:

1. Creating a performance model class for your operation (inherit from `GEMM`, `CONV`, `SDPA`, and so on).
2. Implementing `get_param_details()`, `flops()`, and `bytes()` methods.
3. Passing an extension file using `--extension_file` when generating the report.

See [`examples/example_megatron_extension.py`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/example_megatron_extension.py) for a complete example of extending TraceLens with custom Megatron operations.

```{note}
If your custom operation is frequently used across multiple projects, it can be added as a native operation in TraceLens. Open an issue or reach out to discuss integration.
```

## Collective communication analysis

The `coll_analysis` sheet analyzes NCCL collective communication operations from a single rank's perspective. It shows which collectives are taking the most time on this rank.

```{note}
This is single-rank analysis. For multi-rank synchronization analysis (skew, stragglers), see [Generate a multi-rank collective report](../how-to/collective-report.md).
```

This sheet provides aggregated statistics for collective operations seen by this rank, grouped by collective type, process group, data type, and message size.

### Collective identification

| Column | Description |
|--------|-------------|
| `rank` | Rank ID (single rank from this trace) |
| `Process Group Name` | Process group identifier |
| `Process Group Ranks` | List of ranks in this process group (for example, `[0, 1, 2, 3, 4, 5, 6, 7]`) |
| `Collective name` | Type of collective operation (for example, `allreduce`, `allgather`, `reduce_scatter`) |
| `Group size` | Number of ranks in the process group |
| `dtype` | Data type (for example, `Float`, `BFloat16`) |
| `In msg nelems` | Number of elements in input message |
| `Out msg nelems` | Number of elements in output message |
| `In split size` | Input split configuration (for split collectives) |
| `Out split size` | Output split configuration (for split collectives) |
| `stream` | GPU stream ID where the collective executes |

### Message sizes

| Column | Description |
|--------|-------------|
| `In msg size (MB)_first` | Input message size in megabytes |
| `Out msg size (MB)_first` | Output message size in megabytes |

### Duration statistics

All times are in microseconds.

| Column | Description |
|--------|-------------|
| `dur_sum` | Total duration across all occurrences |
| `dur_mean` | Mean duration per occurrence |
| `dur_std` | Standard deviation of duration |
| `dur_min` | Minimum duration |
| `dur_max` | Maximum duration |
| `operation_count` | Number of times this collective appeared |

### Understanding the metrics

Duration is the time this rank spends in the collective operation (in microseconds). It includes:

- Synchronization delay waiting for other ranks.
- Actual data transfer time.

```{note}
From a single rank's perspective, you can't determine whether time is spent waiting for other ranks (sync) or in actual communication. For multi-rank analysis with skew detection and straggler identification, use NCCL Analyzer with traces from all ranks. See [Generate a multi-rank collective report](../how-to/collective-report.md).
```

### Interpreting results

- High total duration: sort by `dur_sum` to find the most time-consuming collective operations on this rank.
- High variance (large `dur_std` or difference between `dur_min` and `dur_max`):
  - Indicates inconsistent performance.
  - Can suggest network contention or varying degrees of synchronization delay.
  - Consider investigating with multi-rank traces to identify stragglers.
- Frequent operations (high `operation_count`):
  - Even if individual operations are fast, high frequency can add up (`dur_sum` accounts for this).
  - Check if collectives can be batched or overlapped with computation.

Example analysis:

```
If you see:
- allreduce with dur_mean=3596 µs (3.6 ms), operation_count=2
- But dur_std=35.5 µs (low variance)
→ Collective is consistent and moderately fast

If you see:
- allreduce with dur_mean=192.6 µs, dur_std=313.7 µs, dur_max=1019.4 µs
→ High variance! One occurrence took 1ms while another took 60µs
→ Investigate with multi-rank trace to find if this rank or another is the straggler
```

Generated by `NcclAnalyser.build_df_summary_long()`. Enable with the `--disable_coll_analysis` flag (enabled by default).

## Additional sheets

Depending on the command-line arguments used when generating the report, additional analysis sheets can be present:

- `kernel_summary`: per-kernel statistics aggregated by kernel name (enabled with `--kernel_summary`).
- `short_kernel_histogram`, `short_kernels_summary`: analysis of very short kernels (enabled with `--short_kernel_study`).
- `unified_perf_summary`: unified perf metrics for all ops with perf models or leaf ops that launch GPU kernels. Includes GFLOPS, TFLOPS/s, Data Moved, FLOPS/Byte, TB/s metrics aggregated by unique args. When `--detect_recompute` is enabled, an `is_recompute` column is added to split rows by recompute status.

## Common analysis workflows

A typical top-down analysis flow:

1. Start with `gpu_timeline`: get a high-level time breakdown (computation, communication, idle).
   - High idle time points to a CPU bottleneck or kernel launch issues.
   - High exposed communication points to poor overlap with computation.
2. Identify bottleneck categories (`ops_summary_by_category`): which operation types dominate? GEMM, convolution, attention, elementwise, and so on.
3. Find expensive operations (`ops_summary`): which specific operations take the most time?
4. Analyze variants (`ops_unique_args`): which input shapes/arguments are slow?
   - Compare performance across different dimensions.
   - Check for stride or alignment issues.
5. Deep dive with roofline (GEMM, SDPA_fwd, and so on): assess efficiency vs. hardware peaks.
6. Check collectives (`coll_analysis`): analyze communication costs in distributed training.
7. Replay operations: reproduce and benchmark specific cases in isolation.

## Related topics

- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
- [Generate a JAX performance report](../how-to/generate-perf-report-jax.md)
- [Generate a rocprofv3 performance report](../how-to/generate-perf-report-rocprof.md)
- [Replay a traced operation](../how-to/event-replay.md)
- [API reference](../reference/api-reference.md)
