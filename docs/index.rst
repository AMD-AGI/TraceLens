..
   Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

.. meta::
   :description: TraceLens is an open-source Python library for automated GPU trace analysis. Generate performance reports from PyTorch, JAX, and rocprofv3 traces.
   :keywords: TraceLens, GPU trace analysis, ROCm, AMD Instinct, PyTorch profiler, JAX, rocprofv3, roofline analysis, performance report, distributed training, CUDA migration

***********************
TraceLens documentation
***********************

TraceLens is an open-source Python library developed by AMD that automates
performance analysis from GPU trace files. Instead of manually inspecting raw
profiling data in tools such as Perfetto or Chrome Trace Viewer, TraceLens
parses traces from PyTorch, JAX, and ``rocprofv3`` and produces structured
performance reports — including hierarchical GPU-timeline breakdowns,
per-operator roofline analysis (TFLOP/s, TB/s), and multi-GPU communication
diagnostics.

The TraceLens source code is hosted at `github.com/AMD-AGI/TraceLens <https://github.com/AMD-AGI/TraceLens>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install TraceLens <install/install>`

   .. grid-item-card:: How to

      * :doc:`Generate reports <how-to/generate-reports>`

        * :doc:`PyTorch performance report <how-to/generate-perf-report-pytorch>`
        * :doc:`PyTorch inference performance report <how-to/generate-perf-report-pytorch-inference>`
        * :doc:`JAX performance report <how-to/generate-perf-report-jax>`
        * :doc:`rocprof performance report <how-to/generate-perf-report-rocprof>`
        * :doc:`Collective-communication report <how-to/collective-report>`
        * :doc:`Compare performance reports <how-to/compare-perf-reports>`

      * :doc:`Compare two traces <how-to/compare-traces>`
      * :doc:`Replay a single operation <how-to/event-replay>`
      * :doc:`Fuse multi-rank traces <how-to/trace-fusion>`
      * :doc:`Analyze traces with the SDK <how-to/sdk-analysis>`
      * :doc:`Model op performance without a trace <how-to/perf-model-without-trace>`
      * :doc:`Analyze collective communication <how-to/nccl-analysis>`
      * :doc:`Estimate kernel times with Origami <how-to/origami-integration>`
      * :doc:`Agentic performance analysis with the TraceLens Agent <how-to/agent>`

   .. grid-item-card:: Concepts

      * :doc:`Trace2Tree data model <conceptual/trace2tree>`
      * :doc:`PyTorch traces <conceptual/torch-profiling-analysis>`
      * :doc:`Tensor shape metadata <conceptual/shape-metadata>`
      * :doc:`GEMM analysis <conceptual/gemm-analysis>`
      * :doc:`Inference performance analysis <conceptual/inference-analysis>`
      * :doc:`Triton kernel performance model <conceptual/triton-perf-model-walkthrough>`

   .. grid-item-card:: Reference

      * :doc:`API reference <reference/api-reference>`
      * :doc:`Performance report columns <reference/perf-report-columns>`
      * :doc:`Compatibility matrix <reference/compatibility>`

For information on contributing to TraceLens, see the
`Contributing guide <https://github.com/AMD-AGI/TraceLens/blob/main/CONTRIBUTING.md>`_.

TraceLens is released under the MIT License. For details, see the
:doc:`License <about/license>` page.
