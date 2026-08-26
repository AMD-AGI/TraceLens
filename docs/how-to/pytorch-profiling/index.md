<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Run the PyTorch profiler

```{meta}
:description: Learn how to capture, schedule, and analyze PyTorch profiler traces for single-GPU, distributed, and anomaly-detection workloads using TraceLens.
:keywords: TraceLens, PyTorch profiler, torch.profiler, GPU trace, distributed profiling, anomaly detection, ROCm, AMD Instinct, CUDA, performance profiling
```

These guides cover the end-to-end workflow for collecting PyTorch profiler traces and feeding them into TraceLens for analysis.

- [Configure and run the PyTorch profiler](./torch-profiling.md) — capture single-GPU traces with `torch.profiler`, from a minimal timeline to scheduled long-run profiling.
- [Profile distributed PyTorch workloads](./distributed-profiling.md) — collect rank-separated traces with DDP and `torchrun` without file races.
- [Investigate PyTorch training performance anomalies](./anomaly-detection.md) — use always-on CUDA-only profiling with threshold-based capture to isolate anomalous steps.
