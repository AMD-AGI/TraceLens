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

      * :doc:` Install TraceLens <install/installation>`

   .. grid-item-card:: How to

      * :doc:`Generate a PyTorch performance report <how-to/generate-perf-report-pytorch>`
      * :doc:`Generate a JAX performance report <how-to/generate-perf-report-jax>`
      * :doc:`Generate a rocprof performance report <how-to/generate-perf-report-rocprof>`
      * :doc:`Generate a collective-communication report <how-to/collective-report>`
      * :doc:`Compare two traces <how-to/compare-traces>`
      * :doc:`Replay a single operation <how-to/event-replay>`
      * :doc:`Fuse multi-rank traces <how-to/trace-fusion>`

   .. grid-item-card:: Reference

      * :doc:`API reference <reference/api-reference>`

For information on contributing to TraceLens, see the
`Contributing guide <https://github.com/AMD-AGI/TraceLens/blob/main/CONTRIBUTING.md>`_.

TraceLens is released under the MIT License. For details, see the
:doc:`License <about/license>` page.
