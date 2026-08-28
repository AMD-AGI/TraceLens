<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analyze traces with the TraceLens SDK
```{meta}
:description: Learn how to use the TraceLens Python SDK (TreePerfAnalyzer) to build a call tree from a trace, navigate it, and compute GPU-timeline, per-operation, roofline, and nn.Module-level performance metrics.
:keywords: TraceLens, TreePerfAnalyzer, TreePerf, Trace2Tree, GPUEventAnalyser, SDK, GPU timeline, roofline, FLOPS, nn.Module, PyTorch profiler, JAX, ROCm
```

The TraceLens SDK builds a hierarchical call tree from a profiler trace and
computes performance metrics on top of it. `TreePerfAnalyzer` is the main entry
point: it wraps [`Trace2Tree`](../conceptual/trace2tree.md) (which parses the
trace into a CPU-to-GPU dependency tree) and exposes methods to navigate that tree
and derive metrics at the timeline, operation, roofline, and `nn.Module` levels.

This topic is the narrated reference for the SDK. Each section links a runnable
notebook you can execute against your own trace.

If you need the standard metrics in a spreadsheet, the
[`generate_perf_report_pytorch.py`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Reporting/generate_perf_report_pytorch.py)
script packages everything covered here into a ready-made Excel report. Reach for
the SDK when you need more than that report offers — custom filtering, ad-hoc
aggregations, support for a new operation, or programmatic access to the tree.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- A PyTorch profiler trace JSON file for the model you want to analyze.
- A basic understanding of how `Trace2Tree` builds the tree (see
  [Trace2Tree data model](../conceptual/trace2tree.md)).

## Build and navigate the tree

Load a trace with `TreePerfAnalyzer.from_file`. Pass `add_python_func=True` to
include the Python call stack, so you can trace GPU kernels all the way back to
your Python code:

```python
from TraceLens.TreePerf import TreePerfAnalyzer

analyzer = TreePerfAnalyzer.from_file("/path/to/trace.json", add_python_func=True)
tree = analyzer.tree
```

Once you have an event of interest, the tree exposes navigation helpers:

- `tree.traverse_subtree_and_print(event)` — visualize the subtree rooted at an
  operation (its runtime calls and GPU kernels).
- `tree.traverse_parents_and_print(event, cpu_op_fields=(...))` — walk the parent
  chain up to the Python frame that launched it; `cpu_op_fields` optionally shows
  op details (`'Input Dims'`, `'Input type'`, `'Input Strides'`, `'Concrete Inputs'`).
- `tree.get_parent_event(event)`, `tree.get_children_events(event)`, and
  `tree.get_gpu_events(event)` — direct parent, child, and kernel access.

For example, `traverse_subtree_and_print` on an `aten::convolution` op shows how
the dispatch op lowers to runtime launches and kernels:

```
└── UID: 41, Category: cpu_op, Name: aten::convolution
    └── UID: 42, Category: cpu_op, Name: aten::_convolution
        └── UID: 43, Category: cpu_op, Name: aten::miopen_convolution
            ├── UID: 104314, Category: cuda_runtime, Name: hipExtModuleLaunchKernel
            │   └── UID: 107846, Category: kernel, Name: Im2d2Col_v2, Duration: 45.063
            └── UID: 104318, Category: cuda_runtime, Name: hipExtModuleLaunchKernel
                └── UID: 107848, Category: kernel, Name: Cijk_Ailk_Bljk_BBS_BH...
```

Walking the other direction, `traverse_parents_and_print` traces an op back
through the runtime and Python frames all the way to the model's `forward`, and
with `cpu_op_fields` it annotates each level with argument metadata:

```
Node:
  UID: 41, Category: cpu_op, Name: aten::convolution
    Input Dims: [[1, 768, 24, 24], [768, 768, 3, 3], []]
    Input type: [float, float, float]
1-up:
  UID: 40, Category: cpu_op, Name: aten::conv2d
2-up:
  UID: 40139, Category: python_function, Name: <built-in method conv2d ...>
...
5-up:
  UID: 40136, Category: python_function, Name: torch/nn/modules/conv.py(558): forward
```

For a full interactive walkthrough of tree navigation, see
[`trace2tree_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/trace2tree_example.ipynb).

## GPU timeline breakdown

`get_df_gpu_timeline()` returns a high-level view of GPU activity — computation,
communication, memcpy, exposed (non-overlapped) communication, busy, and idle
time:

```python
analyzer.get_df_gpu_timeline()
```

Example output:

| type | time ms | percent |
|------|---------|---------|
| `busy_time` | 6521.46 | 99.93 |
| `computation_time` | 6318.26 | 96.81 |
| `exposed_communication_time` | 203.06 | 3.11 |
| `exposed_memcpy_time` | 0.14 | 0.00 |
| `idle_time` | 4.72 | 0.07 |
| `total_time` | 6526.18 | 100.00 |

The metrics are defined as:

- `computation_time`: Time in compute kernels (GEMMs, convolutions, and so on).
- `communication_time`: Time in collective/communication kernels (for example, NCCL/RCCL).
- `memcpy_time`: Time in host-device or device-device memory copies.
- `exposed_communication_time` / `exposed_memcpy_time`: The portion of
  communication or memcpy that does *not* overlap computation (the part that
  actually costs wall-clock time).
- `busy_time` / `idle_time`: Time the GPU is doing anything vs. nothing.

This breakdown is computed by `GPUEventAnalyser`, which works directly from a list
of GPU events. It operates on any GPU timeline (PyTorch, JAX, and others) and
uses no host-side call stack, so you can run it standalone without building
the CPU-side tree:

```python
import json
from TraceLens import GPUEventAnalyser

with open("/path/to/trace.json") as f:
    events = json.load(f)["traceEvents"]

df = GPUEventAnalyser(events).get_breakdown_df()
```

For JAX traces (which describe events differently and include all GPUs in one
trace), use `JaxGPUEventAnalyser`; `get_breakdown_df_multigpu()` returns a
`{gpu_index: DataFrame}` mapping. To adapt the analyzer to another profiling
format, subclass it and reimplement `get_gpu_event_lists()`.

## Per-operation GPU time

Break GPU time down by the lowest-level CPU operations (from the call-stack
perspective) and the time they *induce* on the GPU. This is a more stable,
interpretable abstraction than raw kernel durations:

```python
df = analyzer.get_df_kernel_launchers(include_kernel_details=True)
analyzer.get_df_kernel_launchers_summary(df)                     # grouped by op name
analyzer.get_df_kernel_launchers_unique_args(df, include_pct=True)  # by op + unique args
```

Example op-name summary (top rows by induced GPU time):

| name | Count | total_direct_kernel_time_ms | Percentage (%) |
|------|-------|-----------------------------|----------------|
| aten::mm | 126 | 5576.76 | 88.73 |
| flash_attn::_flash_attn_backward | 8 | 213.62 | 3.40 |
| flash_attn::_flash_attn_forward | 8 | 119.65 | 1.90 |
| aten::copy_ | 4 | 69.96 | 1.11 |

Pass `event_name="aten::mm"` to `get_df_kernel_launchers_unique_args` to drill
into a single op's argument combinations.

## Roofline metrics

For modeled op classes (GEMM, SDPA, CONV, and more), `TreePerf` computes
analytical FLOPs and bytes and combines them with measured time to yield achieved
efficiency (`TFLOPS/s`, `TB/s`, `FLOPS/Byte`):

```python
from TraceLens.PerfModel.torch_op_mapping import build_sheet_category_to_op_names

categories = build_sheet_category_to_op_names(analyzer.op_to_perf_model_class_map)
gemm_events = [e for e in analyzer.tree.events if e["name"] in categories["GEMM"]]

df_gemm = analyzer.build_df_perf_metrics(gemm_events, include_kernel_details=True)
analyzer.summarize_df_perf_metrics(df_gemm, ["mean"])   # group by M, N, K, bias
```

Example `aten::mm` summary (top rows by induced GPU time, grouped by shape):

| name | param: M | param: N | param: K | param: bias | FLOPS/Byte_first | TFLOPS/s_mean |
|---------|----------|----------|----------|-------------|------------------|---------------|
| aten::mm | 73728 | 28672 | 8192 | False | 5864.73 | 698.19 |
| aten::mm | 73728 | 8192 | 28672 | False | 5864.73 | 719.59 |
| aten::mm | 28672 | 8192 | 73728 | False | 5864.73 | 740.51 |
| aten::mm | 73728 | 128256 | 8192 | False | 6972.01 | 628.10 |

`FLOPS/Byte` is the operation's arithmetic intensity; comparing achieved
`TFLOPS/s_mean` against the device peak tells you whether each shape is compute-
or memory-bound.

For how these metrics are defined and modeled, see
[GEMM analysis](../conceptual/gemm-analysis.md) and the
[performance report columns](../reference/perf-report-columns.md) reference. To
compute the same metrics from shapes alone, without a trace, see
[Model op performance without a trace](./perf-model-without-trace.md). The
[`tree_perf_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/tree_perf_example.ipynb)
notebook walks through roofline metrics for every modeled category.

### Add a new operation

Roofline metrics are only available for modeled op classes, but extending the
perf model to a new op is straightforward, and contributions are welcome:

- In [`perf_model.py`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/PerfModel/perf_model.py):
  parse the operation's shapes from the trace and write (or reuse) a performance
  model that returns FLOPs and bytes.
- In [`torch_op_mapping.py`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/PerfModel/torch_op_mapping.py):
  map the operation name to that perf model.

Once mapped, the op is picked up automatically by `build_df_perf_metrics` and by
the report generator.

## nn.Module attribution

GPU time can be attributed all the way up to the Python `nn.Module` hierarchy,
making performance insights accessible without call-stack expertise. Build a
latency tree rooted at a module and traverse it:

```python
analyzer = TreePerfAnalyzer.from_file("/path/to/trace.json", add_python_func=True)
tree = analyzer.tree

event = next(e for e in tree.events if e["name"] == "nn.Module: DeepseekV2DecoderLayer_2")
analyzer.build_nn_module_latency_tree(event)

# navigate the module hierarchy
children = tree.get_nn_module_children(event)   # list of module UIDs
parent = tree.get_nn_module_parent(event)       # parent module UID
```

Each node carries its aggregated `GPU Time`, so a recursive walk prints a
module-level breakdown that mirrors your model architecture. The example below is
a single DeepSeek-V2 decoder layer, with each module annotated by its GPU time
(µs) and a `Non-nn.Module GPU Ops` entry for GPU work not owned by a child
module:

```
                                                            (GPU Time μs)
└── nn.Module: DeepseekV2DecoderLayer_2 ..................... (10233.51)
    ├── nn.Module: RMSNorm_8 ................................ (148.46)
    ├── nn.Module: DeepseekV2AttentionMLA_2 ................. (6775.96)
    │   ├── nn.Module: ReplicatedLinear_4 ................... (817.78)
    │   ├── nn.Module: RMSNorm_9 ............................ (19.67)
    │   ├── nn.Module: ColumnParallelLinear_2 ............... (268.80)
    │   ├── nn.Module: ReplicatedLinear_5 ................... (469.98)
    │   ├── nn.Module: RMSNorm_10 ........................... (10.21)
    │   ├── nn.Module: DeepseekScalingRotaryEmbedding_0 ..... (139.27)
    │   ├── nn.Module: RadixAttention_2 ..................... (3204.62)
    │   ├── nn.Module: RowParallelLinear_4 ................. (1446.40)
    │   └── Non-nn.Module GPU Ops ........................... (399.22)
    ├── nn.Module: RMSNorm_11 .............................. (154.43)
    └── nn.Module: DeepseekV2MLP_2 ......................... (3154.66)
        ├── nn.Module: MergedColumnParallelLinear_2 ........ (1592.94)
        ├── nn.Module: SiluAndMul_2 ....................... (84.44)
        └── nn.Module: RowParallelLinear_5 ................ (1477.28)
```

See
[`nn_module_view.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/nn_module_view.ipynb)
for the full traversal against your own trace, including the recursive
pretty-printer.

## Related topics

- [The Trace2Tree data model](../conceptual/trace2tree.md)
- [Model op performance without a trace](./perf-model-without-trace.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Replay a single operation in TraceLens](./event-replay.md)
- [API reference](../reference/api-reference.md)
