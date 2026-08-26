<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Trace2Tree data model in TraceLens
```{meta}
:description: Understand why Trace2Tree exists and how its four-layer call tree links Python code and backend operations down to GPU kernels for portable, interpretable performance analysis.
:keywords: TraceLens, Trace2Tree, TreePerfAnalyzer, call tree, GPU kernel, PyTorch profiler, cpu_op, aten, HLO, JAX, ROCm, AMD, performance analysis
```

In GPU application performance analysis, understanding the relationship between host CPU operations and the corresponding GPU kernel executions is crucial for finding bottlenecks. The PyTorch profiler provides a JSON trace file containing events with timestamps and durations, but it lacks explicit call-stack dependency information.

`Trace2Tree` is the underlying tree-structure component that TraceLens uses to parse trace files and build hierarchical dependency relationships, from host CPU operations down to GPU kernels.

```{note}
Access this functionality through the `TreePerfAnalyzer` interface rather than using `Trace2Tree` directly.
```

## Why kernel names aren't enough

Directly inspecting GPU kernel names has two fundamental limitations:

* Ambiguous semantics (and weak reproducibility): A single kernel name can map to many different computations depending on shape, dtype, strides or layout, and so on. Shape strongly affects performance — one shape might select a fast tiled path while another shape of the same op type falls onto a slower algorithm. Because the name omits this argument context, you can't reliably understand, compare, or reproduce the workload from the kernel string alone. Many kernel names are also cryptic and unreadable, for example `Cijk_Ailk_Bljk_*` or `void cutlass_*`.

* Platform-dependent, unstable naming (and weak cross-platform comparison): The same high-level operation appears under different kernel names across platforms. For example, a single GEMM shows up as `nvjet_*` or `cutlass_*` on NVIDIA H100, and as a Tensile kernel `Cijk_Ailk_Bljk_*` on AMD Instinct™ MI300. These names also shift across software versions, so raw kernel strings aren't a stable abstraction for comparison.

## What Trace2Tree does

`Trace2Tree` reconstructs a full call tree from the Python front end down to each GPU kernel. The tree has exactly four layers:

* Python front end — user code or `nn.Module`.
* Operation — on PyTorch this is the dispatch operation (`cpu_op`), for example `aten::mm`, `aten::addmm`, and so on. On JAX the corresponding layer is the HLO operations. This is the layer that contains the argument information used to contextualize the kernel.
* HIP / CUDA runtime — launch API calls.
* GPU kernel — the executed kernel.

## How this solves the problem

The tree structure addresses the kernel-name limitations described above in three ways.

* Disambiguates semantics: Argument metadata at the backend op layer lets you group identical computations, attribute time, and deterministically reproduce slow cases.
* Enables fair comparison: Operations such as `aten::mm` and HLO are stable across platforms. By anchoring analysis there, you can compare the same operation and arguments regardless of how the kernels are named underneath.
* Flexible attribution: GPU time can be viewed at any level — module (through its backend ops), dispatch op, runtime, or kernel — depending on the question. As an additional benefit, time can be attributed all the way up to the Python `nn.Module` level, making performance insights accessible even to users outside the performance-engineering field. This helps bridge the gap between model developers and low-level hardware execution.

Kernel names are volatile and context-free. Trace2Tree anchors analysis at the stable backend operation, enriches it with arguments, and maps the full execution stack to deliver portable, interpretable performance insight.

That said, kernel names are often useful — they can offer clues about the backend implementation, algorithm variant, or compiler choices. TraceLens intends to serve as a one-stop solution for extracting every bit of useful signal from a trace file, so it includes features to extract and parse relevant information from kernel names where applicable. But it treats them as supplementary, not foundational.

## A powerful IR for deeper analysis

This tree is a powerful intermediate representation (IR) that forms the foundation of many TraceLens features. It captures both the structural and performance semantics of a workload: the hierarchical dependency between CPU operations and GPU kernel launches, the argument metadata that disambiguates each operation, and the forward-versus-backward relationship between operations. Because it's a lightweight, framework-agnostic structure (built today for PyTorch profiler JSON, with JAX support and room for more), higher-level analyses — GPU-timeline breakdowns, roofline metrics, `nn.Module` attribution, event replay, and trace diffing — all build on top of it.

You typically don't construct or traverse the tree by hand. It's accessed through the `TreePerfAnalyzer` interface, which builds the tree from a trace and exposes the navigation and metric APIs. To work with the tree in practice, see [Analyze traces with the TraceLens SDK](../how-to/sdk-analysis.md).

## Related topics

- [Analyze traces with the TraceLens SDK](../how-to/sdk-analysis.md)
- [Replay a single operation in TraceLens](../how-to/event-replay.md)
- [Understanding PyTorch traces](../conceptual/torch-profiling-analysis.md)
- [API reference](../reference/api-reference.md)
- [What is TraceLens?](../what-is-tracelens.md)
