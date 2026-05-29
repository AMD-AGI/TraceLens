<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# 🚀 TraceLens Inference Performance Analysis

TraceLens now provides comprehensive support for inference use cases, with a focus on inference serving optimization. This documentation covers:

- 📋 **Overview** - New features for inference trace analysis
- 🔧 **Trace Collection** - Methodologies and setup
- 📊 **Analysis Tools** - Available workflows and usage
- 🗺️ **Roadmap** - Upcoming improvements

## ✨ Key Features


| Feature                      | Description                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Agentic Analysis**         | Agentic workflows for single-trace analysis for performance improvement recommendations                                      |
| **Graph Execution Analysis** | Merge graph capture and graph reply traecfiles for augmenting graph execution tracefile with callstack and shape information |
| **TraceDiff**                | Extended to support inference traces with Lowest Common Ancestor (LCA) analysis for kernel correlation across platforms      |
| **Roofline Analysis**        | Custom roofline models for key inference operations (fused MoE, unified attention) with prefill/decode request annotations.  |
| **Trace Splitting**          | Splitting of large tracefiles into steady-state regions, per-iteration traces, and phase-specific analyses                   |


## Supported Frameworks and Execution Modes

TraceLens features for inference analysis have been primarily tested with vLLM and SGLang. Here is a summary of the different execution modes and supported features.


| Mode                                | Shapes/Roofline analysis | Agent Analysis                                                                                       | Limitations                                                                                                                                                |
| ----------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eager only                          | Yes                      | Supported; proposed patches are recommended to include roofline information for attention operations | Eager mode execution may employ different compilation strategies, which can result in differences in kernels and fusions compared to graph execution mode. |
| Graph execution only                | Non‑graph kernels        | Limited                                                                                              | Categorization, call stacks, and shapes are available only for attention kernels if full_and_piecewise mode is used                                        |
| Graph execution + eager mode trace  | Limited                  | Limited                                                                                              | Kernel categorization might not be as accurate as eager or graph+capture                                                                                   |
| Graph execution + Graph capture$^1$ | Yes                      | Yes (patches required)                                                                               |                                                                                                                                                            |


  $^1$ Graph mode analysis using graph capture and graph replay traces is supported for vLLM and SGLang (proposed patches required).

## 📖 Quickstart Guide

### Step 1: Installation

Install TraceLens from GitHub (requires AMD-AGI organization access):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

### Step 2: Trace Collection

There are two ways to collect inference traces for TraceLens analysis:

- **Option A — Use the Profiling agent** *(recommended)*: an agentic skill bundled with TraceLens that drives a Magpie LLM inference benchmark, applies the Docker / framework patches for you, tunes the profiler window, runs the benchmark, and splits the resulting trace into analysis-ready windows.
- **Option B — Manual profiling**: build (or patch) your own image, configure the profiler yourself, and run the benchmark by hand. Use this if you are not using Magpie or want full control over every step.

#### Option A: Use the Profiling Agent (recommended)

Follow [`TraceLens/Agent/Profiling/README.md`](../TraceLens/Agent/Profiling/README.md). The skill handles trace collection and splitting end-to-end; you can skip the rest of Step 2 and Step 3.

#### Option B: Manual Profiling

The remainder of Step 2 covers manual trace collection — building or patching the inference framework, configuring the profiler, and running the benchmark yourself.

##### Build a Docker image using the [provided scripts](../examples/custom_workflows/inference_analysis/) (recommended for manual flow)

###### vLLM Script

A unified build script is provided that supports multiple vLLM versions. It takes a version tag (`v14`, `v15`, `v16`, `v17`, `v18`, `v19`, `v20`, or `v21`) as the first argument, followed by the path to your local TraceLens clone and any standard `docker build` flags. The script selects the correct base image and patch file automatically.


| Version | Base Image                                                    | vLLM Version | Patch File                  |
| ------- | ------------------------------------------------------------- | ------------ | --------------------------- |
| `v14`   | `rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120`        | v0.14.0      | `config_vllm_v0.14.0.patch` |
| `v15`   | `rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130`        | v0.15.0      | `config_vllm_v0.15.0.patch` |
| `v16`   | `rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223` | v0.16.0      | `config_vllm_v0.16.0.patch` |
| `v17`   | `vllm/vllm-openai-rocm:v0.17.0`                               | v0.17.0      | `config_vllm_v0.17.0.patch` |
| `v18`   | `vllm/vllm-openai-rocm:v0.18.0`                               | v0.18.0      | `config_vllm_v0.18.0.patch` |
| `v19`   | `vllm/vllm-openai-rocm:v0.19.0`                               | v0.19.0      | `config_vllm_v0.19.0.patch` |
| `v20`   | `rocm/vllm-dev:preview_v0.20.0_20260429`                      | v0.20.0      | `config_vllm_v0.20.0.patch` |
| `v21`   | `vllm/vllm-openai-rocm:v0.21.0`                               | v0.21.0      | `config_vllm_v0.21.0.patch` |


```bash
bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    v16 \
    /path/to/TraceLens \
    -t tracelens-vllm
```

To use a custom base Docker image instead of the default for the selected version, pass `--base-image`:

```bash
bash examples/custom_workflows/inference_analysis/build_docker_vllm.sh \
    v18 \
    /path/to/TraceLens \
    --base-image my-registry/vllm:nightly \
    -t tracelens-vllm:custom
```

Then create a container from the image.

###### SGLang Script

The build script for SGLang supports SGLang 0.5.9, 0.5.11, and 0.5.12. It takes the path to the local TraceLens clone, the SGLang version (`--sglang-version`, default 0.5.9), and the GPU type (`--gpu-type`, default mi350). MI300 and MI350/MI355 are supported.

**SGLang versions** (`--sglang-version`):

| Version | Patch directory | ROCm | Description |
| ------- | --------------- | ---- | ----------- |
| `0.5.9` (default) | `sglang_roofline_patches/sglang_0_5_9/` | 7.0 | Legacy stack |
| `0.5.11` | `sglang_roofline_patches/sglang_0_5_11/` | 7.2 | Previous stable |
| `0.5.12` | `sglang_roofline_patches/sglang_0_5_12/` | 7.2 | Recommended for newer stacks |

Each version subdirectory contains the same set of patch files (`cuda_graph_runner`, `fused_moe_triton_kernels`, `http_server`, `io_struct`, `kernel_shape_profiler`, `profile_utils`, `scheduler`, `scheduler_profiler_mixin`, `server_args`, `tokenizer_communicator_mixin`). Use the directory that matches your SGLang version exactly.

**Base images** (overridable with `--base-image`):

| SGLang Version | GPU Type | Base Image |
| -------------- | -------- | ---------- |
| `0.5.9` | MI300 | `lmsysorg/sglang:v0.5.9-rocm700-mi30x` |
| `0.5.9` | MI350/MI355 | `lmsysorg/sglang:v0.5.9-rocm700-mi35x` |
| `0.5.11` | MI300 | `lmsysorg/sglang:v0.5.11-rocm720-mi30x` |
| `0.5.11` | MI350/MI355 | `lmsysorg/sglang:v0.5.11-rocm720-mi35x` |
| `0.5.12` | MI300 | `lmsysorg/sglang:v0.5.12-rocm720-mi30x` |
| `0.5.12` | MI350/MI355 | `lmsysorg/sglang:v0.5.12-rocm720-mi35x` |

**Examples:**

```bash
# SGLang 0.5.12 on MI300X (recommended)
bash examples/custom_workflows/inference_analysis/build_docker_sglang.sh \
    /path/to/TraceLens \
    --sglang-version 0.5.12 \
    --gpu-type mi300 \
    -t tracelens-sglang:0.5.12-mi300

# SGLang 0.5.11 on MI355X
bash examples/custom_workflows/inference_analysis/build_docker_sglang.sh \
    /path/to/TraceLens \
    --sglang-version 0.5.11 \
    --gpu-type mi355 \
    -t tracelens-sglang:0.5.11-mi355

# SGLang 0.5.9 on MI350 (legacy default)
bash examples/custom_workflows/inference_analysis/build_docker_sglang.sh \
    /path/to/TraceLens \
    --gpu-type mi350 \
    -t tracelens-sglang:0.5.9-mi350
```

Then create a container from the image.

##### Apply framework patches manually (alternative to building a new image)

If you prefer to patch an existing environment instead of building a new image, apply patches to your inference framework to:

- Add custom annotations with request packing information (See [roofline conceptual details](#roofline-analysis))
- Capture graph mode execution phases for augmentation by TraceLens

**Steps:**

1. **Locate your inference engine:**
  For vLLM: 
  ```bash
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
  ```
  For SGLang:
  ```bash
   python -c "import sglang; import os; print(os.path.dirname(sglang.__file__))"
  ```
  
2. **Find and apply the relevant patch:**
  - Select by framework and version
  - Apply: `cd /path/to/framework/../ && git apply /path/to/patchfile`
  vLLM patches are in [vllm_roofline_patches](../examples/custom_workflows/inference_analysis/vllm_patches)
  SGLang patches are in [sglang_roofline_patches](../examples/custom_workflows/inference_analysis/sglang_roofline_patches/) — use the `sglang_0_5_9/`, `sglang_0_5_11/`, or `sglang_0_5_12/` subdirectory matching your SGLang version

##### Collection Parameters

- **Eager or Graph Execution Steady-State Window:** Large tracefiles are expected. Most inference serving benchmarks use `NUM_PROMPTS = 10 × CONC` with OSL sampling ratio R. We recommend tracing `(((R+1)/2) * 5 * OSL) ± (16 * OSL / CONC)` execution steps (which represents peak concurrency with prefill-decode mix). See [steady-state region identification](#steady-state-region-and-trace-splitting) for more details. User might need to increase the timeout limit in certain inference frameworks to allow storing the trace in the middle of the execution (e.g., VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 for vLLM).
- **Graph Capture Mode:** The recommended patchfile will trace the graph capture phase and store corresponding tracefiles.
- **Profiler Setup:** Enable CPU-side call-stack and shape capture. For example, vLLM supports `profiler-config.torch_profiler_record_shapes` and `profiler-config.torch_profiler_with_stack`.

##### Trace collection options

###### vLLM

The `config_vllm_v*.patch` patches add two `ProfilerConfig` flags that control graph-capture profiling and trace annotation detail. These patches are available for v0.14–v0.20. Pass the flags as server arguments:


| Flag                                               | Type   | Default | Description                                                                                                                                                                                                                                                                                |
| -------------------------------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--profiler-config.capture_torch_profiler_dir DIR` | `str`  | `""`    | Absolute path to a directory where a PyTorch profiler trace of the CUDA graph capture phase will be saved (rank 0 only). Requires `--profiler-config.profiler torch`. Leave empty to disable graph capture profiling.                                                                      |
| `--profiler-config.detailed_trace_annotation`      | `bool` | `False` | When `True`, execution-step annotations include roofline metrics (`sk`, `sqsq`, `sqsk`) for both context and generation requests. When `False`, annotations record only request counts and token counts. Enable this for full roofline analysis; leave disabled for lighter-weight traces. |


**Example** — enable both flags alongside a standard steady-state window profile:

```bash
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /workspace/torch_trace \
--profiler-config.capture_torch_profiler_dir /workspace/torch_trace/capture_traces \
--profiler-config.detailed_trace_annotation True \
--profiler-config.delay_iterations 5402 \
--profiler-config.max_iterations 256 \
--profiler-config.ignore_frontend True
```

> **Note:** `capture_torch_profiler_dir` is only available when `--profiler-config.profiler torch` is set. The capture trace is written once at server startup during CUDA graph construction; the steady-state replay trace is written to `torch_profiler_dir` during the benchmark run. Pass both paths to `generate_perf_report_pytorch_inference` via `--capture_folder` and `--profile_json_path` respectively.

###### SGLang

1. While doing the profiling of the execution step, pass the parameter `shape_discovery=True` in the profile request to enable shape discovery and registration for operations which are not covered in default SGLang profile.
2. While doing the profiling of the execution step, pass the parameter `roofline_annotations=True` in the profile request to annotate trace with more detailed information useful for roofline annotations.
3. To profile the graph capture phase, while server startup provide the `--enable-profile-cuda-graph` server argument. This will save a trace file per batch size but it misses shape information for some operations, to ensure more diverse coverage, provide the `--enable-shape-discovery-for-cuda-graph-profile` server argument.

### Step 3: Trace Preparation (Optional)

This optional step reads the collected trace and splits it into smaller trace files or execution‑phase‑specific trace files.

Option 1: Find steady-state region of execution (highest concurrency) and separate prefill-decode and decode-only execution steps (supports vLLM v0.14–v0.20 and SGLang v0.5.9–v0.5.12; using the patchfile is recommended). This is recommended if the tracefile is large and the user wants to extract a few representative steps automatically.

```python
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz  -o ./steady_state_analysis \
     --find-steady-state --num-steps 256
```

Output: A tracefile containing {num-steps} contiguous execution steps where close to maximum concurrency is observed, plus contiguous prefill-decode mix and decode-only steady-state tracefiles extracted from this window with no idle gaps between execution steps.

**Refining steady-state window selection with `--CONC`, `--OSL`, and `--R`**

By default, the mixed steady-state window is selected by matching the empirically observed prefill-decode to total-steps ratio of the trace. If the benchmark parameters are known, passing `--CONC`, `--OSL`, and `--R` lets the tool compute an *ideal* perfilldecodemix_steps/total_steps ratio analytically and use that to drive window selection instead:


| Argument | Type    | Description                                                                                                                         |
| -------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `--CONC` | `int`   | Expected peak concurrency (number of concurrent requests). A warning is printed if the observed trace peak differs from this value. |
| `--OSL`  | `float` | Maximum output sequence length (decode tokens per request). Each request's OSL is sampled from `[R × OSL, OSL]`.                    |
| `--R`    | `float` | OSL sampling range ratio in `[0, 1]`. `R=0` means all requests use exactly `OSL` tokens; `R=1` means OSL is uniform in `[0, OSL]`.  |


When all three are provided, the tool derives:

```
ideal_prefilldecodemix_to_totalsteps_ratio = (CONC × 2) / (OSL × (1 + R))
```

and uses this as the reference ratio for mixed-window selection, overriding the empirical estimate. `--num-steps` is also automatically raised to `ceil(1 / ideal_prefilldecodemix_to_totalsteps_ratio)` if it is too small to capture a representative decode/prefill-decode mix.

Example — benchmark with CONC=32, OSL=1024, R=0.8:

```python
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz \
    -o ./steady_state_analysis \
    --find-steady-state --num-steps 256 \
    --CONC 32 --OSL 1024 --R 0.8
```

Option 2: One tracefile per eager/graph execution step (supports vLLM v0.13 or higher, SGLang v0.5.9–v0.5.12, and Atom 0.1.1). This is recommended if the user wants to perform analysis on an isolated execution step.

```python
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz -o ./output --store-single-iteration
```

Output: Single trace file per execution step.

Option 3: Limit the search of steady-state region to a limited window

```python
python -m TraceLens.TraceUtils.split_inference_trace_annotation trace.json.gz -o ./output --iterations 10:20 --find-steady-state --num-steps 256 \
    --CONC 32 --OSL 1024 --R 0.8
```

### Step 4: Generate Performance Report

Performance report generation is supported for both eager-mode and graph-mode (capture + replay) traces.

**Eager or graph replay traces (no graph capture folder):**

```bash
python -m TraceLens.Reporting.generate_perf_report_pytorch_inference \
  --profile_json_path /path/to/trace.json \
  --output_xlsx_path perf_report.xlsx \
  --group_by_parent_module \
  --group_by_kernel_count \
  --enable_pseudo_ops
```

**Graph replay traces augmented with graph capture traces:**

When a `--capture_folder` is provided, the script automatically classifies graph capture traces (batch sizes, full vs. piecewise mode) and merges their call-stack and shape information into the graph replay tree before generating the report.

```bash
python -m TraceLens.Reporting.generate_perf_report_pytorch_inference \
  --profile_json_path /path/to/graph/replay/trace.json \
  --capture_folder /path/to/capture/traces/folder \
  --output_xlsx_path perf_report.xlsx \
  --group_by_parent_module \
  --group_by_kernel_count \
  --enable_pseudo_ops
```

**Additional options:**


| Flag                        | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `--output_csvs_dir DIR`     | Write per-sheet CSV files instead of a single Excel workbook |
| `--gpu_arch_json_path PATH` | Provide a GPU architecture spec for roofline analysis        |


### Step 5: Compare Traces with TraceDiff (Eager-mode only)

Compare two tracefiles and analyze execution differences using Lowest Common Ancestor (LCA) analysis:

```python
import sys
from TraceLens import TreePerfAnalyzer, TraceDiff

file1, file2 = sys.argv[1], sys.argv[2]

# Build performance trees
print(f"Creating tree1 from {file1}...")
perf_analyzer1 = TreePerfAnalyzer.from_file(file1, add_python_func=True)
tree1 = perf_analyzer1.tree

print(f"Creating tree2 from {file2}...")
perf_analyzer2 = TreePerfAnalyzer.from_file(file2, add_python_func=True)
tree2 = perf_analyzer2.tree

# Generate diff report
td = TraceDiff(tree1, tree2)
td.generate_tracediff_report()
td.print_tracediff_report_files("rprt_diff_pruned", prune_non_gpu=True)

print("✅ Pruned TraceDiff reports (GPU only) written to rprt_diff_pruned/")
```

> **Recommendations:** Ensure both tracefiles use similar execution setup (profiled steps, OSL range, concurrency) and the same execution mode (eager/graph) for meaningful comparisons.

### Step 6: Agentic Trace Analysis (Skip Step 4 and 5)

Generate a performance analysis and comparison report (if comparing two traces), along with optimization opportunity analysis, automatically using an LLM agent.

- Performance analysis: This is the recommended first step, and it leverages TraceLens roofline models for performance bridge gap analysis. Please follow [these instructions](../TraceLens/Agent/Analysis/README.md).

---

## 🐞 TraceLens: Report a Bug or Feature Request

Please include the following details when reporting an issue. Please use internal or direct channels to share sensitive data.

1. 🖥️ Environment Details


| Item                             | Details                             |
| -------------------------------- | ----------------------------------- |
| **Inference Engine and Version** | (e.g., vLLM, SGLang)                |
| **Execution Mode**               | (e.g., Eager, Graph, Graph+Capture) |
| **Hardware**                     | (e.g., GPU model)                   |
| **Profiler Config**              | (e.g. Torch profiler config)        |


1. ▶️ Scripts/Commands Used

The scripts and commands used to generate a performance analysis report using TraceLens to reproduce the issue.

1. ❗ Error/Unexpected Behavior
2. 📂 Trace Files Used for Analysis
3. (Optional) 🧪 Expected Output Overview for Feature Request

---

## 📚 Examples & Use Cases

*Example notebooks and scripts coming soon* 🔄

---

## 🔬 Conceptual Details

### [TraceDiff: Lowest Common Ancestor (LCA) Analysis](#tracediff-lca-analysis)

`TraceDiff` is a trace comparison tool that analyzes execution differences between two inference traces (baseline and variant) by constructing a merged tree and identifying structural similarities and differences using **Lowest Common Ancestor (LCA) analysis**.

#### The Problem

When comparing two execution traces from different platforms, frameworks, or configurations:

- Kernel names may differ (e.g., platform-specific optimizations)
- Execution paths may have insertions, deletions, or reorderings
- GPU operations may be fused differently
- We need to **correlate related operations** across traces

#### The Solution: Lowest Common Ancestor

The **Lowest Common Ancestor** is the nearest parent CPU operation or Python function that is **common to both traces** in the merged execution tree. A combination of **position**- and **name**-based comparison rules is used to match two operations or functions. It serves as an anchor point for correlating GPU kernels and operations that differ between traces.

**Key Insight:** If two GPU kernels have the same LCA, they likely serve the same computational purpose, even if their implementations differ.

#### How It Works

1. Tree Alignment with Wagner-Fischer Algorithm

- Uses dynamic programming to align execution trees from both traces
- Identifies three types of nodes:
  - **Combined**: Operations present in both traces (potential LCA candidates)
  - **Trace1-only**: Operations unique to baseline
  - **Trace2-only**: Operations unique to variant
- Normalizes operation names by removing variable parts (memory addresses, line numbers)

1. Merged Tree Construction
  Creates a unified tree structure where:

- Each node has a unique `merged_id`
- Nodes track UIDs from both original traces (`uid1`, `uid2`)
- Parent-child relationships are preserved from both traces
- The merged tree maintains execution hierarchy

1. LCA Identification
  For GPU operations that differ between traces:

```
Trace 1:                    Trace 2:
┌─────────────┐            ┌─────────────┐
│  attention  │ ◄── LCA ──►│  attention  │  (Combined node)
└──────┬──────┘            └──────┬──────┘
       │                          │
   ┌───┴───┐                  ┌───┴───┐
   │       │                  │       │
 GPU_K1  GPU_K2             GPU_K3  GPU_K4  (Different kernels)
(trace1-only)             (trace2-only)
```

The `attention` operation is the **LCA** for all four GPU kernels, indicating they all serve the same high-level computation despite different implementations.

1. Performance Correlation
  The LCA enables meaningful comparisons:

- **Kernel Grouping**: All GPU kernels under the same LCA are functionally related
- **Time Aggregation**: Sum kernel times under each LCA for apples-to-apples comparison
- **Shape Analysis**: Compare input dimensions at the LCA level
- **Optimization Identification**: Spot fusion opportunities or inefficiencies

1. LCA example:

Example snippet

```
├── nn.Module: Attention_0
│   └── torch/nn/modules/module.py(1779): _call_impl
│       └── combined: vllm/attention/layer.py(310): forward | vllm/attention/layer.py(290): forward
│           ├── combined: torch/_ops.py(1243): __call__ | torch/_ops.py(1244): __call__
│           │   └── combined: <built-in method unified_attention_with_output of PyCapsule object at 0x7f3755e18810> | <built-in method unified_attention_with_output of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object at 0x7fdc74225cf0>
│           │       └── vllm::unified_attention_with_output
│           │           └── vllm/attention/utils/kv_transfer_utils.py(36): wrapper
│           │               └── combined: vllm/attention/layer.py(858): unified_attention_with_output | vllm/attention/layer.py(852): unified_attention_with_output
│           │                   └── combined: vllm/v1/attention/backends/rocm_attn.py(256): forward | vllm/v1/attention/backends/flashinfer.py(1064): forward
│           │                       ├── combined: vllm/attention/ops/paged_attn.py(31): write_to_paged_cache | torch/_ops.py(1244): __call__
│           │                       │   └── combined: vllm/_custom_ops.py(2156): reshape_and_cache | <built-in method reshape_and_cache_flash of pybind11_builtins.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1 object at 0x7fdc743f79f0>
│           │                       │       └── combined: torch/_ops.py(1243): __call__ | _C_cache_ops::reshape_and_cache_flash
│           │                       │           ├── >> trace1: <built-in method reshape_and_cache of PyCapsule object at 0x7f37450f4900>
│           │                       │           │   └── >> trace1: _C_cache_ops::reshape_and_cache
│           │                       │           │       └── >> trace1: hipLaunchKernel
│           │                       │           │           └── >> trace1: void vllm::reshape_and_cache_kernel<__hip_bfloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0>(__hip_bfloat16 const*, __hip_bfloat16 const*, __hip_bfloat16*, __hip_bfloat16*, long const*, int, int, int, int, int, int, float const*, float const*)
│           │                       │           └── << trace2: cudaLaunchKernel
│           │                       │               └── << trace2: void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1>(__nv_bfloat16 const*, __nv_bfloat16 const*, unsigned char*, unsigned char*, long const*, long, long, long, long, long, int, int, int, float const*, float const*)
```

#### Output: TraceDiff Report

The generated report includes:


| Column                            | Description                          |
| --------------------------------- | ------------------------------------ |
| `name`                            | GPU kernel name                      |
| `cpu_op_name`                     | Immediate parent CPU operation       |
| `source`                          | `trace1` or `trace2`                 |
| `Input Dims`                      | Tensor shapes at CPU operation level |
| `kernel_time`                     | GPU kernel execution time (μs)       |
| `**lowest_common_ancestor_name**` | **Name of the LCA operation**        |
| `**lowest_common_ancestor_id`**   | **Merged tree ID of the LCA**        |
| `nn_module_stack`                 | PyTorch module hierarchy             |


### [Roofline Analysis](#roofline-analysis)

#### Inference Attention

In inference serving, multiple requests are batched together. Each request has its own sequence lengths (N_Q, N_KV).
**Notation:**


| Symbol            | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| B                 | Batch size (1 per request in paged attention)                       |
| N_Q               | Number of query tokens                                              |
| N_KV              | Number of key/value tokens (context length)                         |
| H_Q               | Number of query heads                                               |
| H_KV              | Number of KV heads (H_KV ≤ H_Q; equal for MHA, smaller for GQA/MQA) |
| d_h_qk            | Head dimension for queries and keys                                 |
| d_h_v             | Head dimension for values                                           |
| R_C               | Number of context (prefill) requests in the batch                   |
| R_G               | Number of generation (decode) requests in the batch                 |
| R                 | Total number of requests (R = R_C + R_G)                            |
| N_Q^(i), N_KV^(i) | Query and KV token counts for the i-th request                      |


**Standard SDPA Attention (Single Request)**

Attention consists of two matrix multiplications per head:

1. **QK^T (score computation):** `2 * B * N_Q * N_KV * H_Q * d_h_qk`
2. **Score × V (value aggregation):** `2 * B * N_Q * N_KV * H_Q * d_h_v`

For causal attention, roughly half the score matrix is masked out:

```
FLOPS = (2 * B * N_Q * N_KV * H_Q * d_h_qk + 2 * B * N_Q * N_KV * H_Q * d_h_v) / 2
```

```
Elements Moved = 
Q:      B * N_Q  * H_Q  * d_h_qk
K:      B * N_KV * H_KV * d_h_qk
V:      B * N_KV * H_KV * d_h_v
Output: B * N_Q  * H_Q  * d_h_v
```

**Inference Paged Attention**

For calculating total Flops and bytes moved for inference paged attention, we **sum over the computation requirement of all requests individually** (B = 1 per request).

Requests fall into two categories:

- **Context (prefill) requests** — processing input tokens; attention is causal within the current chunk
- **Generation (decode) requests** — generating new tokens; attention is non-causal (queries attend to all past KV tokens). Typically N_Q = 1, but approaches like speculative decoding may produce multiple query tokens per request.

**1. Flops Calculation**

**Prefill Requests (First Chunk or Full Context)**

When chunked prefill is not enabled, this is the first (and only) chunk, so N_KV = N_Q and attention is causal:

```
                  R_C
FLOPS_prefill  =   Σ   (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v) / 2
                  i=1
```

**Prefill Requests (Chunked Prefill, nth Chunk)**

With chunked prefill, the nth chunk has KV tokens from all previous chunks already cached. The attention matrix for one such request looks like:

```
                                  Keys (N_KV)
              ◄──── N_KV - N_Q ────►◄──── N_Q ───►
             ┌──────────────────────┬──────────────┐ ▲
             │                      │╲             │ │
             │                      │  ╲  (masked) │ │
             │      Non-causal      │    ╲         │ │
    Queries  │    (full rectangle)  │      ╲       │ N_Q
             │                      │        ╲     │ │
             │  attend to previous  │ Causal   ╲   │ │
             │    chunks' KV cache  │  (self)    ╲ │ │
             └──────────────────────┴──────────────┘ ▼
             ◄────── previous ──────►◄── current ──►
                     chunks               chunk
```

The attention computation splits into two regions:

a. **Current chunk attending to previous chunks** — this region is a full (non-causal) rectangle of shape N_Q × (N_KV − N_Q)
b. **Current chunk attending to itself** — this region is causal (lower-triangular), so we halve

For the **first chunk** (no chunking, or first chunk of chunked prefill), N_KV = N_Q, so the rectangle vanishes and the entire matrix is causal:

```
           Keys (N_KV = N_Q)
          ◄──── N_Q ─────►
         ┌──────────────┐ ▲
         │╲             │ │
         │  ╲  (masked) │ │
         │    ╲         │ │
         │      ╲       │ N_Q   Queries
         │        ╲     │ │
         │ Causal   ╲   │ │
         │ (entire)   ╲ │ │
         └──────────────┘ ▼
```

```
                  R_C
FLOPS_chunked  =   Σ  [ 2 * N_Q(i) * (N_KV(i) - N_Q(i)) * H_Q * d_h_qk
                  i=1
                       + 2 * N_Q(i) * (N_KV(i) - N_Q(i)) * H_Q * d_h_v
                       + (2 * N_Q(i)² * H_Q * d_h_qk + 2 * N_Q(i)² * H_Q * d_h_v) / 2 ]
```

Both cases (first chunk and nth chunk) simplify to a **single unified formula**:

```
                  R_C
FLOPS_context  =   Σ  [ (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v)
                  i=1
                       - (2 * N_Q(i)² * H_Q * d_h_qk + 2 * N_Q(i)² * H_Q * d_h_v) / 2 ]
```

This works because:

- When N_KV = N_Q (first chunk): `full - full/2 = full/2`, which is causal
- When N_KV > N_Q (nth chunk): `full rectangle - self-triangle`, which is the non-causal rectangle plus the causal self-attention

**Generation Requests**

Generation requests attend to all cached KV tokens (N_KV = context length so far). Typically N_Q = 1 (autoregressive decoding), but techniques like speculative decoding may have N_Q > 1. The attention is non-causal:

```
                     R_G
FLOPS_generation  =   Σ   (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v)
                     i=1
```

```
FLOPS_total = FLOPS_context + FLOPS_generation
```

**2. Elements Moved**

The total memory traffic sums over all requests. Each request reads its Q, K, V tensors and writes the output. With GQA/MQA, the KV cache uses H_KV heads (not H_Q), reducing KV memory traffic.

Ignoring cases with shared KV pages between requests:

```
                    R
Elements_moved  =   Σ  ( N_Q(i)  * H_Q  * d_h_qk        // Q read
                   i=1
                       +  N_KV(i) * H_KV * d_h_qk        // K read (from paged KV cache)
                       +  N_KV(i) * H_KV * d_h_v          // V read (from paged KV cache)
                       +  N_Q(i)  * H_Q  * d_h_v  )       // Output write
```

where R = R_C + R_G is the total number of requests.

Note that B = 1 per request in paged attention, so the batch dimension is absorbed into the summation.

**Practical Roofline Analysis Without Per-Request Details**

Importantly, we do **not** need the details of individual requests to perform roofline analysis. Inspecting the formulas above, the only per-request quantities that appear are `N_Q(i)`, `N_KV(i)`, and their products. The full FLOPS and memory traffic expressions can be evaluated using just these **aggregate statistics**, computed separately for context and generation requests:


| Aggregate      | Used in                                              |
| -------------- | ---------------------------------------------------- |
| R_C, R_G       | Request counts                                       |
| Σ N_Q          | Elements moved (Q read, Output write)                |
| Σ N_KV         | Elements moved (K read, V read)                      |
| Σ (N_Q * N_KV) | FLOPS (full rectangle term)                          |
| Σ (N_Q²)       | FLOPS (causal self-attention correction for prefill) |


We obtain these aggregates by applying `torch.record_function(annotation)` to vLLM's execution steps. A single execution step can contain a mix of both context (prefill) and generation (decode) requests, so the annotation encodes the aggregate statistics **separately** for context and generation requests within that step (e.g., R_C, R_G, Σ N_Q for context, Σ N_Q for generation, etc.). These annotations are stored as `user_annotation` events in the PyTorch profiler trace, making roofline analysis possible directly from the trace without any additional instrumentation or runtime logging.

### [Steady-State Region and Trace Splitting](#steady-state-region-and-trace-splitting)

Inference serving execution consists of three phases:

1. **Ramp-up:** The initial few steps where one or more requests are being batched.
2. **Ramp-down:** The last few trailing steps where the final batch of requests finishes.
3. **Steady state:** Defined as the execution steps with the highest concurrency. Once steady state is reached, execution consists of:
  - Decode‑only steps
  - Prefill‑decode steps, typically containing one prefill request packed with ~CONC−1 decode requests.

For performance analysis, we are interested in profiling only the steady‑state steps:

1. Prefill‑decode steps
2. Decode‑only steps with large context sizes (towards the end of a request)

**Parameters Relevant to Inference Serving**

- **NUM_PROMPTS**: typically `10 × CONC`
- **CONC**: number of concurrent requests that can be batched together
- **R**: Random‑range ratio used for sampling ISL and OSL
- **OSL**: Maximum output sequence length. Output sequence length per request is sampled uniformly in:
`[ R × OSL , OSL ]`
- **ISL**: assumed to be lower than the chunk size

We assume inference serving benchmark schedules requests at an infinite rate, and conservatively treat the first **CONC** steps as the *ramp‑up* phase.

With these parameters, execution step ranges where groups of CONC requests complete:

```
1 × R × OSL  to  1 × OSL      e.g. 0.8 OSL – 1 OSL
2 × R × OSL  to  2 × OSL      e.g. 1.6 OSL – 2 OSL
3 × R × OSL  to  3 × OSL      e.g. 2.4 OSL – 3 OSL
4 × R × OSL  to  4 × OSL      e.g. 3.2 OSL – 4 OSL
...
N × R × OSL  to  N × OSL

Where,  N = NUM_PROMPTS / CONC
```

```
TOTAL_STEPS = NUM_PROMPTS × Avg(OSL) / CONC
TOTAL_PrefillDecode_Steps = NUM_PROMPTS
TOTAL_DecodeOnly_Steps= NUM_PROMPTS × ( (Avg(OSL) − CONC) / CONC )
```

Since inference serving benchmarks commonly use `R = Random Range Ratio` to sample output sequence lengths from `[R * OSL, OSL]`, the most useful steady‑state profiling window lies in:

```
5 * R * OSL – 5 * OSL
```

Here, the probability distribution of a request finishing at step t is non-uniform, with the highest probability at ((R+1)/2) * 5 * OSL.

This region exhibits:

- Fully saturated concurrency
- Representative mix of decode-only and prefill-decode steps
- Minimal warm-up or tail artifacts

This makes the recommended window for performance profiling:

```
max_iters = ( 16 * OSL/CONC ) # The number of execution steps to be profiled. The multiplier 16 is used to ensure that the profiling window has ~16 execution steps with prefill+decode mix requests. The max_iters value might need clamping for extreme values of OSL/CONC. 

delay_iters = ( ((R+1)/2) * 5 * OSL ) - (max_iters/2) # The execution step where profiler starts. 
```

#### Trace Splitting

The trace splitting workflow provides three key features. Note that trace splitting assumes vLLM v0.14 or higher (tested through v0.20), or use of our provided patches, to ensure that relevant annotations (batch size, request counts, etc.) are included in execution step metadata.

1. **Split into individual execution steps:** Decompose the entire trace into per-step files, extracting batch size from annotations or kernels for shape-focused analysis and comparison.
2. **Identify steady-state region:** Detect execution steps with near-maximum concurrency. The algorithm identifies large windows with concurrency close to peak levels and selects a representative steady-state region based on prefill-decode and decode-only step composition. When benchmark parameters are known, pass `--CONC`, `--OSL`, and `--R` to `split_inference_trace_annotation` to override the empirical reference ratio with the analytically derived ideal PD ratio — see [Step 3](#step-3-trace-preparation-optional) for details.
3. **Separate phase analysis:** Further decompose steady-state into prefill-decode and decode-only traces. Since prefill and decode have different computational bottlenecks, separate analysis enables targeted performance optimization.

### [Trace Availability-Analysis Trade-off](#trace-availability-analysis-trade-off)

Balancing complete trace capture versus analysis complexity.

---

**Last Updated:** May 2026
**Maintainers:** AMD-AGI Performance and Optimization Team
**Repository:** [github.com/AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens)
