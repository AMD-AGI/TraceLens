
# 🚀 TraceLens Inference Performance Analysis

TraceLens-internal extends the open-source TraceLens tooling to provide comprehensive support for inference use cases, with a focus on InferenceMax optimization. This documentation covers:

- 📋 **Overview** - New features for inference trace analysis
- 🔧 **Trace Collection** - Methodologies and setup
- 📊 **Analysis Tools** - Available workflows and usage
- 🗺️ **Roadmap** - Upcoming improvements

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Trace Comparison** | Automated workflow for comparing two traces and generating performance improvement recommendations |
| **Agentic Analysis** | Autonomous trace analysis workflows for standalone trace analysis and performance recommendations |
| **TraceDiff** | Extended to support inference traces with Lowest Common Ancestor (LCA) analysis for kernel correlation across platforms |
| **Roofline Analysis** | Custom roofline models for key inference operations (fused MoE, unified attention) with prefill/decode request annotations |
| **Trace Splitting** | Splitting of large tracefiles into steady-state regions, per-iteration traces, and phase-specific analyses |


  
  
  

## Supported Frameworks and Execution Modes

TraceLens features for inference analysis have been primarily tested with vLLM, with active efforts underway to extend support to other frameworks such as SGLang and Atom. Here is the summary of different execution modes and supported features.

| Mode | Kernel Categorization | Shapes/Roofline analysis | TraceDiff | Comments |
|----------------------------------------|------------------------|----------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Eager only $^1$ | Yes | Yes | Yes | |
| Graph execution only | Low | Non‑graph kernels | Coarse granularity | Categorization, call stack and shapes are available only for attention kernels if full_and_piecewise mode is used. |
| Graph execution + eager mode trace $^2$ | Approximate | Non‑graph kernels | Coarse granularity | Kernel categorization might not be as accurate as eager or graph+capture since we don’t have callstack for all kernels |
| Graph execution + Graph capture $^2$ | Yes | Yes | Yes | |

  $^1$ Eager mode execution may employ different compilation strategies, which can result in differences in kernels and fusions compared to graph execution mode. 

  $^2$ Graph mode analysis improvement using eager or capture phase trace is coming soon. 

## 📖 Quickstart Guide

### Step 1: Trace Collection

#### Apply Framework Patches
We recommend applying patches to your inference framework to:
- Add custom annotations with request packing information
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
- **Eager or Graph Execution Steady-State Window:** Large tracefiles are expected. InferenceMax uses `NUM_PROMPTS = 10 × CONC` with OSL sampling ratio (R) = 0.8. We recommend tracing 1.6–2.0 OSL execution steps (which represents peak concurrency with prefill-decode mix). See [steady-state region identification](#steady-state-region-identification) for more details. 
- **Graph Capture Mode:** The recommended patchfile will trace the graph capture phase and store corresponding tracefiles.
- **Profiler Setup:** Enable CPU-side callstack and shape capture. An example script to run GPT-OSS using InferenceMax can be [found here](../examples/custom_workflows/inference_analysis/gptoss_fp4_mi355_vllm_docker.sh). 

### Step 2: Installation


Install TraceLens from GitHub (requires AMD-AGI organization access):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens-internal.git
```

### Step 3: Trace Preparation

Read the collected trace and split it into smaller tracefiles:

Option 1: One tracefile per eager/graph execution step (supports vLLM v0.13 or higher). 
```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz -o ./output --store-single-iteration
```

Option 2: Extract execution steps from a specified range and separate prefill-decode and decode-only execution steps (supports vLLM v0.14 or higher; using the patchfile is recommended). 
```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz -o ./output --iterations 10:20
```

Option 3: Find steady-state region of execution (highest concurrency) and separate prefill-decode and decode-only execution steps (supports vLLM v0.14 or higher; using the patchfile is recommended). 

```python
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz  -o ./steady_state_analysis \\
     --find-steady-state --num-steps 256
```

Option 4: Split capture phase tracefile into individual tracefile per shape.
```
python examples/custom_workflows/split_vllm_trace_annotation.py trace.json.gz  -o ./dummy_runs --store-single-iteration
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

### Step 6: Automated Trace Comparison 

Generate performance comparison report and analysis, along with optimization opportunity analysis by comparing two traces. See [Performance comparison flow](../TraceLens/ComparativeMode/README.md) for instructions. 

or

### Step 6: Automated Standalone Performance Analysis

Generate advanced optimization recommendations automatically based on roofline analysis. See [Agentic Mode](../TraceLens/AgenticMode/README.md) for instructions. 


---

## 📚 Examples & Use Cases

*Example notebooks and scripts coming soon* 🔄

---

## 🔬 Conceptual Details *(Coming Soon)*


### [TraceDiff LCA analysis](#tracediff-lca-analysis)


### [Roofline Analysis](#roofline-analysis)

Custom roofline models tailored for inference workloads with prefill/decode-aware metrics.

### [Steady-State Region and Trace Splitting](#steady-state-region-identification)

Inferenc serving execution consists of three phases:

1. **Ramp‑up**  
   Initial few steps where one or more requests are batching.

2. **Ramp‑down**  
   The last few tailing steps where the final batch of requests finishes.

3. **Steady state**  
   Defined as the execution steps with the highest concurrency.  
   Once steady state is reached, execution consists of:
   - Decode‑only steps  
   - Prefill‑decode steps, typically containing one prefill request packed with ~CONC−1 decode requests.

For performance analysis, we are interested in profiling only the steady‑state steps:
1. Prefill‑decode steps  
2. Decode‑only steps with large context sizes (towards the end of a request)


**Parameters Relevant to InferenceMax**

- **NUM_PROMPTS**: typically `10 × CONC`
- **CONC**: number of concurrent requests that can be batched together
- **R**: Random‑range ratio used for sampling ISL and OSL
- **OSL**: Maximum output sequence length  
  Output sequence length per request is sampled uniformly in:
  `  [ R × OSL , OSL ]`
- **ISL**: assumed to be lower than the chunk size

InferenceMax can schedule requests at infinite rate, but we conservatively treat the first **CONC** steps as the *ramp‑up* phase.

With these parameters, execution step Ranges where groups of CONC requests complete:


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

This region contains:
- Fully saturated concurrency  
- Representative mix of decode‑only and prefill‑decode steps  
- Minimal warm‑up or tail artifacts  

Thus it is the recommended window for performance profiling.



### [Trace Availability-Analysis Trade-off](#trace-availability-analysis-trade-off)

Balancing complete trace capture versus analysis complexity.




---

## 🗺️ Roadmap

### 🔄 In Progress

- [ ] Extend graph execution analysis using TraceDiff reports from eager phase.
- [ ] Extend graph execution analysis by linking it to capture phase trace.
- [ ] Extending support for other inference engines.

### 🚀 Future Improvements

- [ ] Unified interface for performance analysis.
- [ ] Critical path analysis for accurate end-to-end performance projection.
- [ ] Integration with performance projection tools.

---

**Last Updated:** February 2026  
**Maintainers:** AMD-AGI Performance and Optimization Team  
**Repository:** [github.com/AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal)
