<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Generate a PyTorch inference performance report
```{meta}
:description: Learn how to collect, split, and analyze PyTorch LLM-serving (vLLM/SGLang) traces with TraceLens, including graph-capture merging for graph-mode inference.
:keywords: TraceLens, PyTorch profiler, inference, vLLM, SGLang, HIP graph, CUDA graph, graph capture, LLM serving, FusedMoE, roofline, trace splitting, steady state, ROCm
```

`TraceLens_generate_perf_report_pytorch_inference` is the inference-oriented
variant of the PyTorch report. It targets LLM-serving traces (for example, from
vLLM or SGLang) that run in CUDA/HIP graph mode, and can merge the graph-capture
traces back into the graph-replay trace to recover the call-stack and input-shape
metadata that graph execution drops.

This topic covers the end-to-end inference workflow: collecting traces, splitting
them into steady-state windows, and generating the report. For training or
single-run PyTorch traces, use
[Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
instead. For the underlying concepts (the paged-attention roofline model and the
steady-state region), see
[Inference performance analysis in TraceLens](../conceptual/inference-analysis.md).

## Supported frameworks and execution modes

TraceLens inference features are primarily tested with vLLM and SGLang. Feature
coverage depends on how the model is executed:

| Mode | Shapes / roofline analysis | Agent analysis | Limitations |
|------|----------------------------|----------------|-------------|
| Eager only | Yes | Supported; proposed patches are recommended to include roofline information for attention operations | Eager execution may use different compilation strategies, resulting in different kernels and fusions than graph execution. |
| Graph execution only | Non-graph kernels | Limited | Categorization, call stacks, and shapes are available only for attention kernels if `full_and_piecewise` mode is used. |
| Graph execution + eager-mode trace | Limited | Limited | Kernel categorization may be less accurate than eager or graph+capture. |
| Graph execution + graph capture [^1] | Yes | Yes (patches required) | |

[^1]: Graph-mode analysis using graph-capture and graph-replay traces is
supported for vLLM and SGLang (proposed patches required).

## Before you begin

- TraceLens installed (see [Install TraceLens](../install/install.md)).
- An LLM inference setup to profile (this guide uses vLLM or SGLang on AMD Instinct™ MI300X).

## Collect inference traces

There are two ways to collect inference traces for TraceLens analysis.

### Option A: Profiling agent (recommended)

The Profiling agent is an agentic skill bundled with TraceLens that drives an LLM
inference benchmark, applies the Docker/framework patches for you, tunes the
profiler window, runs the benchmark, and splits the resulting trace into
analysis-ready windows. Follow
[`TraceLens/Agent/Profiling/README.md`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Profiling/README.md).
The agent handles collection and splitting end-to-end; if you use it you can skip
the rest of this section and [Split inference traces](#split-inference-traces-optional).

### Option B: Manual profiling

Use manual profiling for full control over every step. Build (or patch) an image,
configure the profiler yourself, and run the benchmark by hand.

#### Build a Docker image

Build scripts for vLLM and SGLang are provided under
[`examples/custom_workflows/inference_analysis/`](https://github.com/AMD-AGI/TraceLens/tree/main/examples/custom_workflows/inference_analysis).

**vLLM.** A unified build script supports multiple vLLM versions. It takes a
version tag as its first argument, followed by the path to your local TraceLens
clone and any standard `docker build` flags. The script selects the correct base
image and patch file automatically:

| Version | Base image | vLLM version | Patch file |
|---------|-----------|--------------|------------|
| `v14` | `rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120` | v0.14.0 | `config_vllm_v0.14.0.patch` |
| `v15` | `rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130` | v0.15.0 | `config_vllm_v0.15.0.patch` |
| `v16` | `rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223` | v0.16.0 | `config_vllm_v0.16.0.patch` |
| `v17` | `vllm/vllm-openai-rocm:v0.17.0` | v0.17.0 | `config_vllm_v0.17.0.patch` |
| `v18` | `vllm/vllm-openai-rocm:v0.18.0` | v0.18.0 | `config_vllm_v0.18.0.patch` |
| `v19` | `vllm/vllm-openai-rocm:v0.19.0` | v0.19.0 | `config_vllm_v0.19.0.patch` |
| `v20` | `rocm/vllm-dev:preview_v0.20.0_20260429` | v0.20.0 | `config_vllm_v0.20.0.patch` |
| `v21` | `vllm/vllm-openai-rocm:v0.21.0` | v0.21.0 | `config_vllm_v0.21.0.patch` |
| `v22` | `vllm/vllm-openai-rocm:v0.22.0` | v0.22.0 | `config_vllm_v0.22.0.patch` |
| `v23` | `vllm/vllm-openai-rocm:v0.23.0` | v0.23.0 | `config_vllm_v0.23.0.patch` |
| `v24` | `vllm/vllm-openai-rocm:v0.24.0` | v0.24.0 | `config_vllm_v0.24.0.patch` |

```bash
bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    v16 \
    /path/to/TraceLens \
    -t tracelens-vllm
```

To use a custom base image instead of the default for the selected version, pass
`--base-image`:

```bash
bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    v18 \
    /path/to/TraceLens \
    --base-image my-registry/vllm:nightly \
    -t tracelens-vllm:custom
```

**SGLang.** The SGLang build script takes the path to your local TraceLens clone,
the SGLang version (`--sglang-version`, default `0.5.9`), and the GPU type
(`--gpu-type`, default `mi350`). MI300 and MI350/MI355 are supported:

| SGLang version | GPU type | Base image |
|----------------|----------|------------|
| `0.5.9` | MI300 | `lmsysorg/sglang:v0.5.9-rocm700-mi30x` |
| `0.5.9` | MI350/MI355 | `lmsysorg/sglang:v0.5.9-rocm700-mi35x` |
| `0.5.11` | MI300 | `lmsysorg/sglang:v0.5.11-rocm720-mi30x` |
| `0.5.11` | MI350/MI355 | `lmsysorg/sglang:v0.5.11-rocm720-mi35x` |
| `0.5.12` | MI300 | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` |
| `0.5.12` | MI350/MI355 | `lmsysorg/sglang:v0.5.12-rocm720-mi35x` |
| `0.5.13` | MI300 | `lmsysorg/sglang:v0.5.13-rocm720-mi30x` |
| `0.5.13` | MI350/MI355 | `lmsysorg/sglang:v0.5.13-rocm720-mi35x` |
| `0.5.14` | MI300 | `lmsysorg/sglang:v0.5.14-rocm720-mi30x` |
| `0.5.14` | MI350/MI355 | `lmsysorg/sglang:v0.5.14-rocm720-mi35x` |
| `0.5.15` | MI300 | `lmsysorg/sglang:v0.5.15-rocm720-mi30x` |
| `0.5.15` | MI350/MI355 | `lmsysorg/sglang:v0.5.15-rocm720-mi35x` |
| `0.5.16` | MI300 | `lmsysorg/sglang:v0.5.16-rocm720-mi30x` |
| `0.5.16` | MI350/MI355 | `lmsysorg/sglang:v0.5.16-rocm720-mi35x` |
| `0.5.17` | MI300 | `lmsysorg/sglang:v0.5.17-rocm720-mi30x` |
| `0.5.17` | MI350/MI355 | `lmsysorg/sglang:v0.5.17-rocm720-mi35x` |

On SGLang **0.5.13 / 0.5.14**, kernel-shape wrapping is incompatible with the
EAGLE/MTP speculative *overlap* decode, so the speculative patches disable capture
profiling on the speculative graph runners (and, on 0.5.14, the target-verify
graph) to keep MTP runs fault-free; non-MTP shape profiling is unaffected. Full MTP
shape profiling (capture + execution) works on **0.5.15 and later**.

```bash
bash examples/custom_workflows/inference_analysis/build_docker_sglang.sh \
    /path/to/TraceLens \
    --sglang-version 0.5.11 \
    --gpu-type mi350 \
    -t tracelens-sglang
```

Create a container from the resulting image.

#### Apply framework patches manually

If you prefer to patch an existing environment instead of building a new image,
apply patches to your inference framework to:

- Add custom annotations with request-packing information (these annotations feed
  the [roofline model](../conceptual/inference-analysis.md#roofline-analysis-for-inference-attention)).
- Capture graph-mode execution phases for augmentation by TraceLens.

To apply them:

1. Locate your inference engine:

   ```bash
   # vLLM
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
   # SGLang
   python -c "import sglang; import os; print(os.path.dirname(sglang.__file__))"
   ```

2. Select the patch for your framework and version and apply it:

   ```bash
   cd /path/to/framework/../ && git apply /path/to/patchfile
   ```

   vLLM patches are in
   [`vllm_patches`](https://github.com/AMD-AGI/TraceLens/tree/main/examples/custom_workflows/inference_analysis/vllm_patches);
   SGLang patches are in
   [`sglang_roofline_patches`](https://github.com/AMD-AGI/TraceLens/tree/main/examples/custom_workflows/inference_analysis/sglang_roofline_patches).

#### Collection parameters

- **Steady-state window**: Inference traces are large. Most serving benchmarks use
  `NUM_PROMPTS = 10 * CONC` with OSL sampling ratio R. Trace
  `((R+1)/2) * 5 * OSL ± (16 * OSL / CONC)` execution steps to capture peak
  concurrency with a prefill-decode mix. See
  [Steady-state region](../conceptual/inference-analysis.md#steady-state-region)
  for the derivation. You may need to raise the trace-store timeout in some
  frameworks to allow writing the trace mid-execution (for example
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200` for vLLM).
- **Graph-capture mode**: The recommended patch traces the graph-capture phase and
  stores the corresponding trace files.
- **Profiler setup**: Enable CPU-side call-stack and shape capture (for example,
  vLLM's `profiler-config.torch_profiler_record_shapes` and
  `profiler-config.torch_profiler_with_stack`).

#### Trace collection flags

**vLLM.** The `config_vllm_v*.patch` patches (available for v0.14-v0.24) add two
`ProfilerConfig` flags. Pass them as server arguments:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--profiler-config.capture_torch_profiler_dir DIR` | `str` | `""` | Directory where a profiler trace of the HIP/CUDA-graph capture phase is saved (rank 0 only). Requires `--profiler-config.profiler torch`. Leave empty to disable graph-capture profiling. |
| `--profiler-config.detailed_trace_annotation` | `bool` | `False` | When `True`, execution-step annotations include roofline metrics (`sk`, `sqsq`, `sqsk`) for context and generation requests. When `False`, annotations record only request and token counts. Enable for full roofline analysis. |

Example - enable both flags alongside a steady-state window profile:

```bash
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /workspace/torch_trace \
--profiler-config.capture_torch_profiler_dir /workspace/torch_trace/capture_traces \
--profiler-config.detailed_trace_annotation True \
--profiler-config.delay_iterations 5402 \
--profiler-config.max_iterations 256 \
--profiler-config.ignore_frontend True
```

```{note}
`capture_torch_profiler_dir` is only available when `--profiler-config.profiler torch`
is set. The capture trace is written once at server startup during HIP/CUDA-graph
construction; the steady-state replay trace is written to `torch_profiler_dir`
during the benchmark. Pass both paths to the report generator using
`--capture_folder` and `--profile_json_path` respectively.
```

**SGLang.** In the profile request for the execution step:

1. Pass `shape_discovery=True` to enable shape discovery and registration for
   operations not covered by the default SGLang profile.
2. Pass `detailed_annotations=True` to annotate the trace with the detailed
   information used for roofline analysis.
3. To profile the graph-capture phase, pass the `--enable-profile-cuda-graph`
   server argument at startup. This saves one trace file per batch size but misses
   shape information for some operations; add
   `--enable-shape-discovery-for-cuda-graph-profile` for more diverse coverage.
4. On SGLang **0.5.13 and later**, writing the per-batch-size graph-capture trace
   files to disk is opt-in via the `SGLANG_GRAPH_BATCH_CAPTURE=1`
   environment variable (default off). Set it *in addition to*
   `--enable-profile-cuda-graph`; without it the capture profiler still runs (for
   the summary tables and memory snapshot) but no per-batch-size trace is written to
   the `capture_traces/` folder. On 0.5.12 and earlier the capture traces are
   written unconditionally and this variable does not exist.

```{note}
On SGLang 0.5.13 / 0.5.14 the graph-capture shape profiling is intentionally
disabled for the EAGLE/MTP speculative graphs (and, on 0.5.14, the target-verify
graph) to avoid a GPU fault in the speculative overlap decode, so
`SGLANG_GRAPH_BATCH_CAPTURE` only yields capture traces for the
non-speculative graphs there. Full MTP graph-capture profiling works on 0.5.15+.
```

## Split inference traces (optional)

Large traces can be split into steady-state windows or per-step files before
report generation, using `TraceLens.TraceUtils.split_inference_trace_annotation`.
Splitting assumes vLLM v0.14 or higher (tested through v0.24) or SGLang v0.5.9,
or use of the provided patches, so that annotations (batch size, request counts,
and so on) are present in the execution-step metadata.

**Find the steady-state region** (highest concurrency) and separate
prefill-decode from decode-only steps. This is the recommended option for large
traces where you want a few representative steps extracted automatically:

```bash
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz \
    -o ./steady_state_analysis \
    --find-steady-state --num-steps 256
```

The output is a trace of `--num-steps` contiguous steps near peak concurrency,
plus contiguous prefill-decode-mix and decode-only steady-state traces extracted
from that window with no idle gaps between steps.

By default the mixed window is selected by matching the empirically observed
prefill-decode-to-total-steps ratio. If the benchmark parameters are known,
passing `--CONC`, `--OSL`, and `--R` computes the *ideal* ratio analytically and
uses that to drive window selection instead:

| Argument | Type | Description |
|----------|------|-------------|
| `--CONC` | `int` | Expected peak concurrency (number of concurrent requests). A warning is printed if the observed trace peak differs. |
| `--OSL` | `float` | Maximum output sequence length (decode tokens per request). Each request's OSL is sampled from `[R * OSL, OSL]`. |
| `--R` | `float` | OSL sampling range ratio in `[0, 1]`. `R=0` means all requests use exactly `OSL` tokens; `R=1` means OSL is uniform in `[0, OSL]`. |

When all three are provided, the tool derives:

```
ideal_prefilldecodemix_to_totalsteps_ratio = (CONC * 2) / (OSL * (1 + R))
```

and uses it as the reference ratio, overriding the empirical estimate.
`--num-steps` is automatically raised to
`ceil(1 / ideal_prefilldecodemix_to_totalsteps_ratio)` if it is too small to
capture a representative mix:

```bash
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz \
    -o ./steady_state_analysis \
    --find-steady-state --num-steps 256 \
    --CONC 32 --OSL 1024 --R 0.8
```

**One trace file per execution step** (vLLM v0.13+, SGLang v0.5.9, Atom 0.1.1) -
use when you want to analyze an isolated execution step:

```bash
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz \
    -o ./output --store-single-iteration
```

**Limit the steady-state search to a window** with `--iterations`:

```bash
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz \
    -o ./output --iterations 10:20 --find-steady-state --num-steps 256 \
    --CONC 32 --OSL 1024 --R 0.8
```

See [Steady-state region](../conceptual/inference-analysis.md#steady-state-region)
for what these phases mean and why this window is the one worth profiling.

## Generate the report

Report generation is supported for both eager-mode and graph-mode (capture +
replay) traces. Pass the graph-replay trace to generate the default Excel report:

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

- [Inference performance analysis in TraceLens](../conceptual/inference-analysis.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Compare two traces](./compare-traces.md)
- [Agentically orchestrate and generate optimization recommendations](./agent.md)
- [API reference](../reference/api-reference.md)
