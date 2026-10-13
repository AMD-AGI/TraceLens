<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens trace quality checks
```{meta}
:description: Reference for TraceLens trace quality checks that validate traces before and after analysis, including check descriptions, severity levels, detailed checks, and environment variables.
:keywords: TraceLens, trace check, trace quality, validation, dropped kernels, call stack, graph mode, with_stack, capture trace, TRACELENS_SKIP_QUALITY_CHECK, TRACELENS_DETAILED_QUALITY_CHECKS, ROCm, AMD Instinct
```

Collecting a trace can be difficult. Profilers require specific flags and settings to capture the data that
downstream analysis needs: call stacks, kernel correlation IDs, graph-capture
traces, etc. On top of this profilers can be buggy and drop crucial information needed for analysis. Without early detection these traces can lead
to incomplete reports, misleading roofline numbers, or hard-to-diagnose inaccuracies in the analysis.

TraceLens addresses this with a set of light-weight trace quality checks that run
automatically every time a trace is loaded. The checks validate that the trace
contains the data required for analysis and surface actionable warnings or
errors before any expensive tree building or performance modeling begins.

## When checks run

Checks are organized into three stages:

- **Pre-report checks** run inside `DataLoader.load_trace_events()`, before
  the call-stack tree is built. Every code path that loads a trace file runs
  basic pre-report checks. Detailed pre-report checks run when
  `TRACELENS_DETAILED_QUALITY_CHECKS=1` is set.

- **Post-tree checks** run inside `TreePerfAnalyzer.__init__()` and
  `JaxTreePerfAnalyzer.__init__()` after `build_tree()` completes. These checks
  require the tree structure (for example, correlation-ID linking between
  runtime launches and GPU kernels).

- **Post-report checks** run after perf report generation completes. These
  validate the generated report DataFrames (idle time, op consistency, shape
  coverage). Only run when `TRACELENS_DETAILED_QUALITY_CHECKS=1` is set.

## Complete check reference

| Check | Phase | Detailed | Framework | Level | Description |
|-------|-------|----------|-----------|-------|-------------|
| `_check_trace_file_size` | pre-report | no | all | warn | Flags trace files smaller than 100 KB (may be empty or warmup-only) or larger than 5 GB (consider splitting). |
| `_check_kernels_present` | pre-report | no | all | error | Verifies at least one GPU event exists. Analysis cannot proceed without GPU work. |
| `_check_call_stack` | pre-report | no | all | warn | Checks for python events indicating call-stack data was captured. Missing call stacks limit call-stack-based analysis and may affect cross-trace comparisons. |
| `_check_graph_mode` | pre-report | no | all | warn | Detects `cudaGraphLaunch` or `hipGraphLaunch` events without a corresponding capture trace. Pass `--capture_folder` to the inference report generator for full graph-mode analysis. |
| `_check_cpu_op_shapes_basic` | pre-report | no | PyTorch | warn | Checks whether CPU operations carry `Input Dims` in their args. Warns if fewer than 10 have shapes. Re-profile with `record_shapes=True`. |
| `_check_cpu_op_shapes_detailed` | pre-report | yes | PyTorch | warn | Computes shape coverage as a percentage across all `cpu_op` events. Warns below 80% coverage, flags below 40%. |
| `_check_cpu_gpu_ratio` | pre-report | yes | PyTorch | warn | Computes the ratio of CPU events to GPU events. Flags ratios outside the 0.2–50 range as suspicious, indicating the profiler may not have captured both CPU and GPU sides properly. |
| `_check_runtime_variability` | pre-report | yes | all | warn | Groups kernels by (name, grid, block) and computes the coefficient of variation of durations per group. Warns when more than 10% of eligible groups exceed CV of 0.75, indicating profiler timing jitter or workload instability. |
| `_check_gpu_busy_idle` | pre-report | yes | all | warn | Merges GPU event intervals and computes idle percentage. Warns at 30% idle, flags at 60%. |
| `_check_jax_metadata_richness` | pre-report | yes | JAX | warn | Checks coverage of `hlo_op`, `hlo_module`, and `tf_op` metadata fields on GPU events. Flags when minimum coverage is below 40%. |
| `_check_kernels_dropped_correlation` | post-report | no | all | warn | Checks each CUDA runtime launch event for a matching GPU kernel via correlation ID after tree building. Reports the count of launches with no corresponding kernel. |
| `_check_idle_timeline` | post-report | yes | all | warn | Reads the `idle_time` row from the report's `gpu_timeline` sheet. Warns at 30% idle, flags at 60%. Cross-checks the pre-report busy/idle analysis against the report's own computation. |
| `_check_ops_have_shapes` | post-report | yes | all | warn | Checks `Input Dims` coverage in the `ops_unique_args` sheet. Flags below 40% — shape data did not survive through the analysis pipeline. |
| `_check_high_idle_ops` | post-report | yes | all | warn | Flags operations where `subtree_kernel_time - direct_kernel_time` exceeds 40% of subtree time, indicating significant non-kernel overhead. |
| `_check_attention_kernel_count` | post-report | yes | JAX | warn | Validates observed attention kernel launches against a model config formula: `num_gpus * num_steps * num_layers * calls * unique_kernels`, computed separately for forward and backward passes. Requires `AttnCountConfig` with `num_layers` and `num_steps` set. |

## Severity levels

| Level | Behavior |
|-------|----------|
| error | `log_findings()` raises a `ValueError` and analysis stops. |
| warn  | `log_findings()` emits a `UserWarning`. Analysis continues. |

## Environment variables

| Variable | Effect |
|----------|--------|
| `TRACELENS_SKIP_QUALITY_CHECK=1` | Skip all pre-report checks. Post-tree and post-report checks still run. |
| `TRACELENS_DETAILED_QUALITY_CHECKS=1` | Enable detailed pre-report and post-report checks in addition to the basic checks. |

```bash
# Skip all pre-report checks
TRACELENS_SKIP_QUALITY_CHECK=1 TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json

# Enable detailed checks
TRACELENS_DETAILED_QUALITY_CHECKS=1 TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json
```

## Related topics

- [Generate a PyTorch performance report](../how-to/generate-perf-report-pytorch.md)
- [Generate a PyTorch inference performance report](../how-to/generate-perf-report-pytorch-inference.md)
- [Compare traces](../how-to/compare-traces.md)
- [API reference](api-reference.md)
