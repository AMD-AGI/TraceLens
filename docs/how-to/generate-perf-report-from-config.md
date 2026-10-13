<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Generate TraceLens performance reports from a YAML config
```{meta}
:description: Learn how to describe a TraceLens performance-report run in one YAML file covering training, inference, graph capture, prefill and decode, and trace comparison.
:keywords: TraceLens, YAML, run config, performance report, PyTorch profiler, inference, vLLM, SGLang, graph capture, speculative decoding, prefill, decode, TraceDiff, ROCm
```

This topic shows how to run TraceLens performance reports from a single YAML
file. It's the recommended way to generate PyTorch reports: one command covers
training and inference traces, and the file records exactly which options were
used. `TraceLens_generate_perf_report_pytorch` and
`TraceLens_generate_perf_report_pytorch_inference` keep working unchanged.

## Before you begin

Before running a config, confirm you have the following:

- [TraceLens installed](../install/install.md).
- One or more `torch.profiler` traces (`.json` or `.json.gz`).
- For graph-mode inference, the graph-capture folder for each trace. See
  [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md).

## Run a config

```bash
TraceLens_generate_perf_report_from_config --config run.yaml
```

The same entry point is available as
`python -m TraceLens.Reporting.generate_perf_report_from_config --config run.yaml`.

The whole file is checked before any trace is opened, so a typo in the last job
fails in seconds instead of after the first report finishes.

## Write a config

A config has three required keys and a list of jobs. Each job produces one
report.

```yaml
analysis_mode: default
output_dir: /data/llama_train/analysis_output
jobs:
  - trace_path: /data/llama_train/trace.json.gz
    platform: MI300X
```

### Top-level keys

| Key | Required | Description |
|-----|----------|-------------|
| `analysis_mode` | yes | `default` for training and plain PyTorch traces. `inference` for vLLM, SGLang, ATOM, and xDiT traces, with or without graph capture. |
| `output_dir` | yes | Directory for the reports. |
| `jobs` | yes | List of jobs. See the job keys below. |
| any report option | no | Any parameter of the selected report function, by its Python name (for example `kernel_summary: true`, `topk_ops: 50`). Applies to every job. |

### Job keys

| Key | Required | Description |
|-----|----------|-------------|
| `name` | when there's more than one job | Used in the output file names. Letters, digits, `_`, `-`, and `.`. |
| `trace_path` | yes | Trace to analyze. |
| `platform` | one of the two | Bundled or `TL_EXTENSION` architecture name, for example `MI300X`. |
| `gpu_arch_json_path` | one of the two | Path to an architecture JSON file, for hardware that isn't bundled. |
| `capture_folder` | no | Graph-capture folder. `inference` only. |
| `compare_to` | no | `trace_path` of a second trace, plus `capture_folder` in `inference`. Same as `--comparison_json_path`. |
| `precomputed_diff_stats` | no | A `diff_stats.csv` to use instead of running TraceDiff. Can't be combined with `compare_to`. |

Report options go at the top level, not on a job. Each job can use a different
architecture, so one config can cover traces from different GPUs.

## Defaults

Unless the config sets them, these options are on. They match the TraceLens
Agent's standard report:

| Option | Mode | Default here | Default in the CLIs |
|--------|------|--------------|---------------------|
| `enable_pseudo_ops` | both | `true` | `false` |
| `group_by_num_kernels` | both | `true` | `false` |
| `include_call_stack` | both | `true` | `false` |
| `group_by_parent_module` | `inference` | `true` | `false` |

Every other option has the same default as the matching CLI. For example,
`collective_analysis` is on in `default` and off in `inference`.

```{note}
`include_call_stack` adds the `call_stack_trimmed` and `call_stack_full`
columns and makes large traces slower to load. Set
`include_call_stack: false` if you don't need them.
```

## Outputs

| Jobs | Files in `output_dir` |
|------|-----------------------|
| One, without `name` | `perf_report.xlsx`, `perf_report_csvs/` |
| Any job with `name` | `perf_report_<name>.xlsx`, `perf_report_<name>_csvs/` |

Before the first job runs, TraceLens writes `run_config.resolved.yaml` to
`output_dir`. It's the same config with every report option spelled out and
every path absolute, so you can check later which options produced a report,
or rerun it as is.

Jobs run in the order listed. If a job fails, the run stops with a non-zero
exit code and the reports already written are kept. Relative paths are resolved
against the directory you run the command from, as in the CLIs.

## Examples

### Change report options

```yaml
analysis_mode: default
output_dir: /data/llama_train/analysis_output
include_call_stack: false
kernel_summary: true
topk_ops: 50
jobs:
  - trace_path: /data/llama_train/trace.json.gz
    platform: MI300X
```

### Inference, eager mode

```yaml
analysis_mode: inference
output_dir: /data/qwen_vllm/analysis_output
jobs:
  - trace_path: /data/qwen_vllm/rank0.json.gz
    platform: MI300X
```

### Inference with graph capture

```yaml
analysis_mode: inference
output_dir: /data/qwen_vllm/analysis_output
jobs:
  - trace_path: /data/qwen_vllm/rank0.json.gz
    platform: MI300X
    capture_folder: /data/qwen_vllm/capture_traces
```

Speculative decoding uses the same config. The drafter and target graphs are
in the same capture folder.

### Prefill and decode

Prefill and decode are two inference jobs, each with its own trace and capture
folder:

```yaml
analysis_mode: inference
output_dir: /data/dsv2lite_pd/analysis_output
jobs:
  - name: prefill_rank0
    trace_path: /data/dsv2lite_pd/prefill/rank0.json.gz
    platform: MI300X
    capture_folder: /data/dsv2lite_pd/prefill/capture_traces
  - name: decode_rank0
    trace_path: /data/dsv2lite_pd/decode/rank0.json.gz
    platform: MI300X
    capture_folder: /data/dsv2lite_pd/decode/capture_traces
```

### Compare two traces

`compare_to` adds speedup, delta, and lowest common ancestor (LCA) columns to
the job's `unified_perf_summary`, plus a `diff_stats` sheet. It doesn't write a
report for the compared trace. Add a second job if you want one:

```yaml
analysis_mode: inference
output_dir: /data/ab/analysis_output
jobs:
  - name: new
    trace_path: /data/ab/new.json.gz
    platform: MI300X
    capture_folder: /data/ab/new_capture
    compare_to:
      trace_path: /data/ab/old.json.gz
      capture_folder: /data/ab/old_capture
  - name: old
    trace_path: /data/ab/old.json.gz
    platform: MI300X
    capture_folder: /data/ab/old_capture
```

### Hardware that isn't bundled

```yaml
analysis_mode: default
output_dir: /data/llama_train/analysis_output
jobs:
  - trace_path: /data/llama_train/trace.json.gz
    gpu_arch_json_path: /data/arch/my_gpu.json
```

## Related topics

- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md)
- [Compare two traces](./compare-traces.md)
- [API reference](../reference/api-reference.md)
