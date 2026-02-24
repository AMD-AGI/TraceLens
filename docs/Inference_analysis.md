# 🚀 TraceLens Inference Performance Analysis

TraceLens-internal extends the open-source TraceLens tooling to provide comprehensive support for inference use cases, with a focus on InferenceMax optimization. This documentation covers:

- 📋 **Overview** - New features for inference trace analysis
- 🔧 **Trace Collection** - Methodologies and setup
- 📊 **Analysis Tools** - Available workflows and usage
- 🗺️ **Roadmap** - Upcoming improvements

## ✨ Key Features

| Feature                     | Description                                                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic Analysis**  | Agentic workflows for standalone trace (single trace) and comparative trace analysis for performance improvement recommendations |
| **TraceDiff**         | Extended to support inference traces with Lowest Common Ancestor (LCA) analysis for kernel correlation across platforms          |
| **Roofline Analysis** | Custom roofline models for key inference operations (fused MoE, unified attention) with prefill/decode request annotations.      |
| **Trace Splitting**   | Splitting of large tracefiles into steady-state regions, per-iteration traces, and phase-specific analyses                       |

## Supported Frameworks and Execution Modes

TraceLens features for inference analysis have been primarily tested with vLLM, with active efforts underway to extend support to other frameworks such as SGLang and Atom. Here is the summary of different execution modes and supported features.

| Mode                                  | Shapes/Roofline analysis | Standalone Analysis                                                                              | Comparative Analysis | Limitations                                                                                                                                                |
| ------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eager only                            | Yes                      | Supported, proposed patches recommended to include roofline information for attention operations | Yes                  | Eager mode execution may employ different compilation strategies, which can result in differences in kernels and fusions compared to graph execution mode. |
| Graph execution only                  | Non‑graph kernels       | Limited                                                                                          | Limited              | Categorization, call stack and shapes are available only for attention kernels if full_and_piecewise mode is used                                          |
| Graph execution + eager mode trace    | Limited                  | Planned                                                                                          | Planned              | Kernel categorization might not be as accurate as eager or graph+capture                                                                                   |
| Graph execution + Graph capture$^1$ | Yes                      | Yes (proposed patches required)                                                                  | Planned              |                                                                                                                                                            |

  $^1$ Graph mode analysis using graph capture and graph replay traces is supported for vLLM (proposed patches to vLLM required), and similar support for other inference engines are coming soon.

## 📖 Quickstart Guide

### Step 1: Trace Collection

#### Apply Framework Patches

We recommend applying patches to your inference framework to:

- Add custom annotations with request packing information (See [roofline conceptual details](#roofline-analysis))
- Capture graph mode execution phases for augmentation by TraceLens

**Steps:**

1. **Locate your inference engine:**

   ```bash
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
   ```
2. **Find and apply the relevant patch:**

   - Browse available patches: [inference patches](../examples/custom_workflows/inference_analysis/)
   - Select by framework and version
   - Apply: `cd /path/to/vllm && git apply /path/to/patchfile`

#### Collection Parameters

- **Eager or Graph Execution Steady-State Window:** Large tracefiles are expected. InferenceMax uses `NUM_PROMPTS = 10 × CONC` with OSL sampling ratio (R) = 0.8. We recommend tracing 1.6–2.0 OSL execution steps (which represents peak concurrency with prefill-decode mix). See [steady-state region identification](#steady-state-region-and-trace-splitting) for more details.
- **Graph Capture Mode:** The recommended patchfile will trace the graph capture phase and store corresponding tracefiles.
- **Profiler Setup:** Enable CPU-side call-stack and shape capture. An example script to run GPT-OSS using InferenceMax can be [found here](../examples/custom_workflows/inference_analysis/gptoss_fp4_mi355_vllm_docker.sh).

### Step 2: Installation

Install TraceLens from GitHub (requires AMD-AGI organization access):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens-internal.git
```

### Step 3: Trace Preparation (Optional)

This optional step reads the collected trace and splits it into smaller trace files or execution‑phase‑specific trace files.

Option 1: Find steady-state region of execution (highest concurrency) and separate prefill-decode and decode-only execution steps (supports vLLM v0.14 or higher; using the patchfile is recommended). This is recommended if the tracefile is large and the user wants to extract a few representative steps automatically.

```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz  -o ./steady_state_analysis \\
     --find-steady-state --num-steps 256
```

Output: A tracefile containing {num-steps} contiguous execution steps where close to maximum concurrency is observed, a tracefile containing prefill-decode mix steps from this window, and a tracefile containing deocde-only steps from this window. The tracefiles with prefill-decode and decode-only steps are non-contiguous and will have huge idle time between execution steps.

Option 2: One tracefile per eager/graph execution step (supports vLLM v0.13 or higher). This is recommended if the user wants to perform analysis on isolated execution step.

```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz -o ./output --store-single-iteration
```

Output: Single trace file per execution step.

Option 3: Extract execution steps from a specified range and separate prefill-decode and decode-only execution steps (supports vLLM v0.14 or higher; using the patchfile is recommended).

```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz -o ./output --iterations 10:20
```

### Step 4: Generate Performance Report

Run standalone performance analysis on eager or graph mode traces:

```bash
python TraceLens/Reporting/generate_perf_report_pytorch_vllm.py \
  --profile_json_path /path/to/trace.json \
  --output_xlsx_path perf_report.xlsx \
  --group_by_parent_module \
  --enable_pseudo_ops
```

Run standalone performance analysis on graph replay and graph capture traces:

```bash
python TraceLens/Reporting/generate_perf_report_pytorch_vllm_graph.py \
  --capture_folder path/to/capture/traces/folder \
  --graph_json_path path/to/graph/replay/trace \
  --output_xlsx_path perf_report.xlsx \
  --group_by_parent_module \
  --enable_pseudo_ops
```

### Step 5: Compare Traces with TraceDiff

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

Generate performance analysis and comparison report (if comparing two traces), along with optimization opportunity analysis automatically using an LLM agent.

- Standalone performance analysis: This is the recommended first step, and it leverages TraceLens roofline models for performance bridge gap analysis. Please follow [these instructions](../TraceLens/AgenticMode/Standalone/README.md).
- Comparative analysis: This is an optional step for comparing two traces. Please follow [these instructions](../TraceLens/AgenticMode/Comparative/README.md).

---

## 🐞 TraceLens-internal: Report a Bug or Feature Request

Please include the following details when reporting an issue. Please use the TraceLens-internal private repo to share sensitive data.

1. 🖥️ Environment Details

| Item                                   | Details                             |
| -------------------------------------- | ----------------------------------- |
| **Inference Engine and Version** | (e.g., vLLM, SGLang)                |
| **Execution Mode**               | (e.g., Eager, Graph, Graph+Capture) |
| **Hardware**                     | (e.g., GPU model)                   |
| **Profiler Config**              | (e.g. Torch profiler config)        |

1. ▶️ Scripts/Commands Used

The scripts and commands used to generate a performance analysis report using TraceLens for reproducing the issue.

3. ❗ Error/Unexpected Behavior
4. 📂 Trace Files Used for Analysis
5. (Optional) 🧪 Expected Output Overview for Feature Request

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

The **Lowest Common Ancestor** is the nearest parent CPU operation or Python function that is **common to both traces** in the merged execution tree. A combination of **position** and **name** based comparison rules are used to match two operations or functions. It serves as an anchor point for correlating GPU kernels and operations that differ between traces.

**Key Insight:** If two GPU kernels have the same LCA, they likely serve the same computational purpose, even if their implementations differ.

#### How It Works

1. Tree Alignment with Wagner-Fischer Algorithm

- Uses dynamic programming to align execution trees from both traces
- Identifies three types of nodes:
  - **Combined**: Operations present in both traces (potential LCA candidates)
  - **Trace1-only**: Operations unique to baseline
  - **Trace2-only**: Operations unique to variant
- Normalizes operation names by removing variable parts (memory addresses, line numbers)

2. Merged Tree Construction
   Creates a unified tree structure where:

- Each node has a unique `merged_id`
- Nodes track UIDs from both original traces (`uid1`, `uid2`)
- Parent-child relationships are preserved from both traces
- The merged tree maintains execution hierarchy

3. LCA Identification
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

4. Performance Correlation
   The LCA enables meaningful comparisons:

- **Kernel Grouping**: All GPU kernels under the same LCA are functionally related
- **Time Aggregation**: Sum kernel times under each LCA for apples-to-apples comparison
- **Shape Analysis**: Compare input dimensions at the LCA level
- **Optimization Identification**: Spot fusion opportunities or inefficiencies

5. LCA example:

<details>
  <summary>Example snippet</summary>

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

</details>

#### Output: TraceDiff Report

The generated report includes:

| Column                                    | Description                          |
| ----------------------------------------- | ------------------------------------ |
| `name`                                  | GPU kernel name                      |
| `cpu_op_name`                           | Immediate parent CPU operation       |
| `source`                                | `trace1` or `trace2`             |
| `Input Dims`                            | Tensor shapes at CPU operation level |
| `kernel_time`                           | GPU kernel execution time (μs)      |
| **`lowest_common_ancestor_name`** | **Name of the LCA operation**  |
| **`lowest_common_ancestor_id`**   | **Merged tree ID of the LCA**  |
| `nn_module_stack`                       | PyTorch module hierarchy             |

### [Roofline Analysis](#roofline-analysis)

#### Inference Attention

In inference serving, multiple requests are batched together. Each request has its own sequence lengths (N_Q, N_KV).
**Notation:**

| Symbol            | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| B                 | Batch size (1 per request in paged attention)                        |
| N_Q               | Number of query tokens                                               |
| N_KV              | Number of key/value tokens (context length)                          |
| H_Q               | Number of query heads                                                |
| H_KV              | Number of KV heads (H_KV ≤ H_Q; equal for MHA, smaller for GQA/MQA) |
| d_h_qk            | Head dimension for queries and keys                                  |
| d_h_v             | Head dimension for values                                            |
| R_C               | Number of context (prefill) requests in the batch                    |
| R_G               | Number of generation (decode) requests in the batch                  |
| R                 | Total number of requests (R = R_C + R_G)                             |
| N_Q^(i), N_KV^(i) | Query and KV token counts for the i-th request                       |

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

For calculating total Flops and byted moved fro inference paged attention, we **sum over the computation requirement of all requests individually** (B = 1 per request).

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

| Aggregate       | Used in                                              |
| --------------- | ---------------------------------------------------- |
| R_C, R_G        | Request counts                                       |
| Σ N_Q          | Elements moved (Q read, Output write)                |
| Σ N_KV         | Elements moved (K read, V read)                      |
| Σ (N_Q * N_KV) | FLOPS (full rectangle term)                          |
| Σ (N_Q²)      | FLOPS (causal self-attention correction for prefill) |

We obtain these aggregates by applying `torch.record_function(annotation)` to vLLM's execution steps. A single execution step can contain a mix of both context (prefill) and generation (decode) requests, so the annotation encodes the aggregate statistics **separately** for context and generation requests within that step (e.g., R_C, R_G, Σ N_Q for context, Σ N_Q for generation, etc.). These annotations are stored as `user_annotation` events in the PyTorch profiler trace, making roofline analysis possible directly from the trace without any additional instrumentation or runtime logging.

### [Steady-State Region and Trace Splitting](#steady-state-region-and-trace-splitting)

Inference serving execution consists of three phases:

1. **Ramp‑up**Initial few steps where one or more requests are batching.
2. **Ramp‑down**The last few tailing steps where the final batch of requests finishes.
3. **Steady state**Defined as the execution steps with the highest concurrency.Once steady state is reached, execution consists of:

   - Decode‑only steps
   - Prefill‑decode steps, typically containing one prefill request packed with ~CONC−1 decode requests.

For performance analysis, we are interested in profiling only the steady‑state steps:

1. Prefill‑decode steps
2. Decode‑only steps with large context sizes (towards the end of a request)

**Parameters Relevant to InferenceMax**

- **NUM_PROMPTS**: typically `10 × CONC`
- **CONC**: number of concurrent requests that can be batched together
- **R**: Random‑range ratio used for sampling ISL and OSL
- **OSL**: Maximum output sequence lengthOutput sequence length per request is sampled uniformly in:
  `  [ R × OSL , OSL ]`
- **ISL**: assumed to be lower than the chunk size

InferenceMax can schedule requests at infinite rate, but we conservatively treat the first **CONC** steps as the *ramp‑up* phase.

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

Since InferenceMax commonly uses `R = 0.8`, the most useful steady‑state profiling window lies in:

```
1.6 OSL – 2 OSL
```

This region exhibits:

- Fully saturated concurrency
- Representative mix of decode-only and prefill-decode steps
- Minimal warm-up or tail artifacts

This makes it the recommended window for performance profiling.

#### Trace Splitting

The trace splitting workflow provides three key features. Note that trace splitting assumes vLLM v0.14 or higher, or use of our provided patches, to ensure that relevant annotations (batch size, request counts, etc.) are included in execution step metadata.

1. **Split into individual execution steps:** Decompose the entire trace into per-step files, extracting batch size from annotations or kernels for shape-focused analysis and comparison.
2. **Identify steady-state region:** Detect execution steps with near-maximum concurrency. The algorithm identifies large windows with concurrency close to peak levels and selects a representative steady-state region based on prefill-decode and decode-only step composition.
3. **Separate phase analysis:** Further decompose steady-state into prefill-decode and decode-only traces. Since prefill and decode have different computational bottlenecks, separate analysis enables targeted performance optimization.

### [Trace Availability-Analysis Trade-off](#trace-availability-analysis-trade-off)

Balancing complete trace capture versus analysis complexity.

---

## 🗺️ Roadmap

### 🔄 In Progress

- [ ] Extend graph execution analysis using TraceDiff reports from eager phase.
- [ ] Improve roofline analysis for sparse attention to use model architecture information
- [ ] Extending support for other inference engines

### 🚀 Future Improvements

- [ ] Unified interface for performance analysis
- [ ] Critical path analysis for accurate end-to-end performance projection
- [ ] Integration with performance projection tools

---

**Last Updated:** February 2026
**Maintainers:** AMD-AGI Performance and Optimization Team
**Repository:** [github.com/AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal)
