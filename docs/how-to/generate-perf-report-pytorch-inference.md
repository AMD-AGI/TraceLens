<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Generate a PyTorch inference performance report
```{meta}
:description: Learn how to generate a TraceLens performance report from a PyTorch LLM-serving (vLLM/SGLang) trace captured in graph mode, including capture-trace merging.
:keywords: TraceLens, PyTorch profiler, inference, vLLM, SGLang, CUDA graph, HIP graph, graph capture, LLM serving, FusedMoE, roofline, ROCm
```

`TraceLens_generate_perf_report_pytorch_inference` is the inference-oriented
variant of the PyTorch report. It targets LLM-serving traces (for example, from
vLLM or SGLang) that run in CUDA/HIP graph mode, and can merge the graph-capture
traces back into the graph-replay trace to recover the call-stack and input-shape
metadata that graph execution drops.

For training or single-run PyTorch traces, use
[Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
instead.

## Before you begin

Before generating a report, confirm you have the following:

- TraceLens installed (see [Install TraceLens](../install/install.md)).
- A graph-replay `torch.profiler` trace (`.json` or `.json.gz`).
- (Optional) The folder of graph-capture traces from the same run, if you want to
  recover shapes and call stacks (see [Merge capture traces](#merge-capture-traces)).

## Generate the report

Pass the graph-replay trace to generate the default Excel report:

```bash
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path tests/traces/inference/graph_full/graph_execution.json.gz
```

Output behavior matches the PyTorch report: a single Excel workbook is written
next to the trace by default; use `--output_xlsx_path` or `--output_csvs_dir` to
change the destination.

## Merge capture traces

In graph mode, the replay trace records kernel launches but loses the CPU
call-stack and input-shape metadata attached to the original operations. If you
captured the graphs, point `--capture_folder` at that folder to merge the capture
traces back into the replay trace and restore that metadata for richer operator
and roofline analysis:

```bash
TraceLens_generate_perf_report_pytorch_inference \
    --profile_json_path tests/traces/inference/graph_full/graph_execution.json.gz \
    --capture_folder tests/traces/inference/graph_full/capture_traces
```

When `--capture_folder` is set, TraceLens first classifies the capture traces
(writing an `execution_details.json` in the folder if one isn't already present),
then merges the matching subtrees into the graph tree before running the standard
analysis.

```{note}
`--capture_folder` and `--comparison_json_path` can't be used together: the
TraceDiff comparison doesn't support graph-capture traces.
```

## Inference-oriented options

The inference report shares most options with the PyTorch report (output paths,
short-kernel study, roofline and Origami, collective analysis). The options most
relevant to serving traces:

| Argument | Default | Description |
|----------|---------|-------------|
| `--profile_json_path` | required | Path to the graph-replay `torch.profiler` trace (`.json` or `.json.gz`). |
| `--capture_folder PATH` | `None` | Folder of graph-capture traces to merge into the replay trace (recovers shapes and call stacks). Mutually exclusive with `--comparison_json_path`. |
| `--group_by_parent_module` | `False` | Group kernel-launcher summaries by parent `nn.Module` in addition to operation name. |
| `--group_by_num_kernels` | `False` | Group summary rows by the number of kernels. |
| `--include_call_stack` | `False` | Add the CPU call stack to the report. |
| `--include_overlap_info` | `False` | Add kernel-overlap sheets (`*_kl_overlap`) when overlap data exists. |
| `--enable_pseudo_ops` | `False` | Augment the tree with pseudo-ops to isolate kernels (for example, `FusedMoE`). |

Run the tool with `--help` for the complete, version-specific argument list.

## Related topics

- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate TraceLens reports](./generate-reports.md)
- [Compare two traces](./compare-traces.md)
- [API reference](../reference/api-reference.md)
