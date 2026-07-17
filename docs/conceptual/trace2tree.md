<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# The Trace2Tree data model
```{meta}
:description: Understand why Trace2Tree exists and how its four-layer call tree links Python code and backend operations down to GPU kernels for portable, interpretable performance analysis.
:keywords: TraceLens, Trace2Tree, TreePerfAnalyzer, call tree, GPU kernel, PyTorch profiler, cpu_op, aten, HLO, JAX, ROCm, AMD, performance analysis
```

In GPU application performance analysis, understanding the relationship between host CPU operations and the corresponding GPU kernel executions is crucial for finding bottlenecks. The PyTorch profiler provides a JSON trace file containing events with timestamps and durations, but it lacks explicit call-stack dependency information.

`Trace2Tree` is the underlying tree-structure component that TraceLens uses to parse trace files and build hierarchical dependency relationships, from host CPU operations down to GPU kernels.

```{note}
It's recommended that you access this functionality through the `TreePerfAnalyzer` interface rather than using `Trace2Tree` directly.
```

## Why kernel names aren't enough

Directly inspecting GPU kernel names has two fundamental limitations:

* *Ambiguous semantics (and weak reproducibility)*: a single kernel name can map to many different computations depending on shape, dtype, strides or layout, and so on. Shape strongly affects performance — one shape might select a fast tiled path while another shape of the same op type falls onto a slower algorithm. Because the name omits this argument context, you can't reliably understand, compare, or reproduce the workload from the kernel string alone. Many kernel names are also cryptic and unreadable, for example `Cijk_Ailk_Bljk_*` or `void cutlass_*`.

* *Platform-dependent, unstable naming (and weak cross-platform comparison)*: the same high-level operation appears under different kernel names across platforms. For example, a single GEMM shows up as `nvjet_*` or `cutlass_*` on NVIDIA H100, and as a Tensile kernel `Cijk_Ailk_Bljk_*` on AMD MI300. These names also shift across software versions, so raw kernel strings aren't a stable abstraction for comparison.

## What Trace2Tree does

`Trace2Tree` reconstructs a full call tree from the Python front end down to each GPU kernel. The tree has exactly four layers:

1. *Python front end* — user code or `nn.Module`.
2. *Operation* — on PyTorch this is the dispatch operation (`cpu_op`), for example `aten::mm`, `aten::addmm`, and so on. On JAX the corresponding layer is the HLO operations. This is the layer that contains the argument information used to contextualize the kernel.
3. *HIP / CUDA runtime* — launch API calls.
4. *GPU kernel* — the executed kernel.

## How this solves the problem

* *Disambiguates semantics*: argument metadata at the backend op layer lets you group identical computations, attribute time, and deterministically reproduce slow cases.
* *Enables fair comparison*: operations such as `aten::mm` and HLO are stable across platforms. By anchoring analysis there, you can compare the same operation and arguments regardless of how the kernels are named underneath.
* *Flexible attribution*: GPU time can be viewed at any level — module (through its backend ops), dispatch op, runtime, or kernel — depending on the question. As an additional benefit, time can be attributed all the way up to the Python `nn.Module` level, making performance insights accessible even to users outside the performance-engineering field. This helps bridge the gap between model developers and low-level hardware execution.

Kernel names are volatile and context-free. Trace2Tree anchors analysis at the stable backend operation, enriches it with arguments, and maps the full execution stack to deliver portable, interpretable performance insight.

That said, kernel names are often useful — they can offer clues about the backend implementation, algorithm variant, or compiler choices. TraceLens intends to serve as a one-stop solution for extracting every bit of useful signal from a trace file, so it includes features to extract and parse relevant information from kernel names where applicable. But it treats them as supplementary, not foundational.

## Key features

- *Hierarchical dependency tree*: constructs a tree structure linking CPU operations to GPU kernel launches, enabling detailed analysis of ops lowering and performance.
- *Extensible SDK*: provides a framework for custom analyses, such as identifying GPU time for CPU operations or pinpointing bottlenecks.
- *Lightweight design*: minimal dependencies and a straightforward codebase for easy integration and use.
- *PyTorch support*: built for PyTorch profiler JSON traces, with potential for future support of other frameworks.

## Build and traverse the tree

```{note}
See [`examples/trace2tree_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/trace2tree_example.ipynb) for a complete interactive tutorial.
```

### Load the trace data through TreePerfAnalyzer

```python
from TraceLens.TreePerf import TreePerfAnalyzer

# Load trace data using TreePerfAnalyzer
# Set add_python_func=True to include the Python function call stack in the tree
# This lets you trace GPU kernels all the way back to your Python code
trace_file = '/path/to/trace.json'
analyzer = TreePerfAnalyzer.from_file(trace_file, add_python_func=True)

# Access the underlying tree structure
tree = analyzer.tree
```

### Find an operation to analyze

```python
# Find an operation of interest
event_interest = next(
    evt for evt in tree.events
    if evt.get('name') == 'aten::convolution' and evt.get('cat') == 'cpu_op'
)
```

### Traverse a subtree

Visualize the entire subtree rooted at this operation:

```python
tree.traverse_subtree_and_print(event_interest)
```

```
└── UID: 41, Category: cpu_op, Name: aten::convolution
    └── UID: 42, Category: cpu_op, Name: aten::_convolution
        └── UID: 43, Category: cpu_op, Name: aten::miopen_convolution
            ├── UID: 104314, Category: cuda_runtime, Name: hipExtModuleLaunchKernel
            │   └── UID: 107846, Category: kernel, Name: Im2d2Col_v2, Duration: 45.063
            └── UID: 104318, Category: cuda_runtime, Name: hipExtModuleLaunchKernel
                └── UID: 107848, Category: kernel, Name: Cijk_Ailk_Bljk_BBS_BH...
```

### Traverse the parent chain

Trace back through all parent events to see the full call stack. You can optionally include CPU operation details such as input dimensions, types, and strides using the `cpu_op_fields` parameter.

Available fields: `'Input Dims'`, `'Input type'`, `'Input Strides'`, `'Concrete Inputs'`.

```python
root = tree.traverse_parents_and_print(
    event_interest,
    cpu_op_fields=('Input Dims', 'Input type')
)
```

```
Node:
  UID: 41, Category: cpu_op, Name: aten::convolution
    Input Dims: [[1, 768, 24, 24], [768, 768, 3, 3], []]
    Input type: [float, float, float]
1-up:
  UID: 40, Category: cpu_op, Name: aten::conv2d
    Input Dims: [[1, 768, 24, 24], [768, 768, 3, 3], [1], [2, 2], [1, 1], [1, 1], [1]]
    Input type: [float, float, int, int, int, int, int]
2-up:
  UID: 40139, Category: python_function, Name: <built-in method conv2d of type object at 0x...>
3-up:
  UID: 40138, Category: python_function, Name: torch/utils/_device.py(100): __torch_function__
4-up:
  UID: 40137, Category: python_function, Name: torch/nn/modules/conv.py(554): _conv_forward
5-up:
  UID: 40136, Category: python_function, Name: torch/nn/modules/conv.py(558): forward
6-up:
  UID: 40135, Category: python_function, Name: torch/nn/modules/module.py(1736): _wrapped_call_impl
7-up:
  UID: 40134, Category: python_function, Name: torch/nn/modules/module.py(1747): _call_impl
8-up:
  UID: 40133, Category: python_function, Name: transformers/models/owlv2/modeling_owlv2.py(395): forward
...
```

## Related topics

- [Tree performance analysis](../how-to/tree-perf-analysis.md)
- [Event replay](../how-to/event-replay.md)
- [Torch profiling analysis](../conceptual/torch-profiling-analysis.md)
- [API reference](../reference/api-reference.md)
- [What is TraceLens?](../what-is-tracelens.md)
