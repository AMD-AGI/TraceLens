<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Trace2Tree

In GPU applications performance analysis, understanding the relationship between host CPU operations and corresponding GPU kernel executions is crucial for analyzing bottlenecks. The PyTorch profiler provides a JSON trace file containing events with timestamps and durations but lacks explicit call stack dependency information.

Trace2Tree is the underlying tree structure component used by TraceLens to parse trace files and build hierarchical dependency relationships from host CPU operations to GPU kernels. **It is recommended to access this functionality through the `TreePerfAnalyzer` interface** rather than using Trace2Tree directly.

---

## Key Features

- **Hierarchical Dependency Tree**: Constructs a tree structure linking CPU operations to GPU kernel launches, enabling detailed analysis of the ops lowering and perfomance.
- **Extensible SDK**: Provides a framework for custom analyses, such as identifying GPU time for CPU operations or pinpointing bottlenecks.
- **Lightweight Design**: Minimal dependencies and a straightforward codebase for easy integration and use.
- **PyTorch Support**: Built for PyTorch profiler JSON traces, with potential for future support of other frameworks.

---

## Quick Start

> **💡 Tip:** See [`examples/trace2tree_example.ipynb`](../examples/trace2tree_example.ipynb) for a complete interactive tutorial.

### Example: Build and traverse tree

#### 1. Load the Trace data via TreePerfAnalyzer
```python
from TraceLens.TreePerf import TreePerfAnalyzer

# Load trace data using TreePerfAnalyzer
# Set add_python_func=True to include Python function call stack in the tree
# This allows you to trace GPU kernels all the way back to your Python code
trace_file = '/path/to/trace.json'
analyzer = TreePerfAnalyzer.from_file(trace_file, add_python_func=True)

# Access the underlying tree structure
tree = analyzer.tree
```

#### 2. Find an Operation to Analyze

```python
# Find an operation of interest
event_interest = next(
    evt for evt in tree.events 
    if evt.get('name') == 'aten::convolution' and evt.get('cat') == 'cpu_op'
)
```

#### 3. Traverse Subtree

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

#### 4. Traverse Parent Chain

Trace back through all parent events to see the full call stack. You can optionally include CPU operation details like input dimensions, types, and strides using the `cpu_op_fields` parameter:

Available fields: `'Input Dims'`, `'Input type'`, `'Input Strides'`, `'Concrete Inputs'`

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