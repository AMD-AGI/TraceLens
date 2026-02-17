
# 🚀 TraceLens Inference Trace Analysis

TraceLens-internal extends the open-source TraceLens tooling to provide comprehensive support for inference use cases, with a focus on InferenceMax optimization. This documentation covers:

- 📋 **Overview** - New features for inference trace analysis
- 🔧 **Trace Collection** - Methodologies and setup
- 📊 **Analysis Tools** - Available workflows and usage
- 🗺️ **Roadmap** - Upcoming improvements

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Trace Comparison** | Automated flow for inference trace comparison and summarization [ToDo: Arseny] |
| **TraceDiff** | Extended to support inference traces with Lowest Common Ancestor (LCA) analysis for kernel correlation across platforms |
| **Roofline Analysis** | Custom roofline models for key inference operations (fused MoE, unified attention) with prefill/decode request annotations |
| **Trace Splitting** | Splitting of large tracefiles into steady-state regions, per-iteration traces, and phase-specific analyses |
| **Agentic Analysis** | Agentic trace analysis workflows for performance optimization [ToDo: Tharun/Adeem] |

  
  
  

## Supported Frameworks and Execution mode

 TraceLens features for Inference analysis have been tested for vLLM and SGLang, with active efforts towards extending it to other frameworks. Here is the summary of different execution modes and supported features.

| Mode | Kernel Categorization | Shapes/Roofline analysis | TraceDiff | Comments |
|----------------------------------------|------------------------|----------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Eager only | Yes* | Yes* | Yes* | |
| Graph execution only | Low | Non‑graph kernels | Coarse granularity | Categorization, call stack and shapes are available only for attention kernels if full_and_piecewise mode is used. |
| Graph execution + eager mode trace | Approximate | Non‑graph kernels | Coarse granularity | Kernel categorization might not be as accurate as eager or graph+capture since we don’t have callstack for all kernels |
| Graph execution + Graph capture | Yes | Yes | WIP | |
  \*Eager mode execution might enforece different compilation strategy which might result in difference in kernels and fusions performed in eager and graph execution mode. 

## 📖 Quickstart Guide

### Step 1: Trace Collection

#### Apply Framework Patches
We recommend applying patches to your inference framework to:
- Add custom annotations with request packing information
- Capture graph mode execution phases (augmented by TraceLens)

**Steps:**
1. **Locate your inference engine:**
   ```bash
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
   ```

2. **Find and apply the relevant patch:**
   - Browse available patches: [inference patches](https://github.com/AMD-AGI/TraceLens-internal)
   - Select by framework and version
   - Apply: `cd /path/to/vllm && git apply /path/to/patchfile`
  
#### Collection Parameters
- **Eager or Graph execution steady-state window:** Large tracefiles expected. InferenceMax uses  `NUM_PROMPTS = 10 × CONC` with OSL sampling ratio (R) = 0.8. We recommend tracing 1.6–2.0 OSL execution steps (represents peak concurrency with prefill-decode mix)
- **Graph capture Mode:** Captures multiple shape/phase combinations; store corresponding tracefiles
- **Profiler Setup:** Enable CPU-side callstack and shape capture

### Step 2: Installation

## 📖 Quickstart Guide

### Step 1: Trace Collection

#### Apply Framework Patches
We recommend applying patches to your inference framework to:
- Add custom annotations with request packing information
- Capture graph mode execution phases (augmented by TraceLens)

**Steps:**
1. **Locate your inference engine:**
   ```bash
   python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
   ```

2. **Find and apply the relevant patch:**
   - Browse available patches: [inference patches](https://github.com/AMD-AGI/TraceLens-internal)
   - Select by framework and version
   - Apply: `cd /path/to/vllm && git apply /path/to/patchfile`

#### Collection Parameters
- **Eager Mode:** Large tracefiles expected. Use `NUM_PROMPTS = 10 × CONC` with OSL sampling ratio (R) = 0.8
- **Steady-State Window:** Trace 1.6–2.0 OSL execution steps (represents peak concurrency with prefill-decode mix)
- **Graph Mode:** Captures multiple shape/phase combinations; store corresponding tracefiles
- **Profiler Setup:** Enable CPU-side callstack and shape capture

**Example:** [InferenceMax collection scripts]()

### Step 2: Installation

Install TraceLens from GitHub (requires AMD-AGI organization access):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens-internal.git
```

### Step 3: Generate Performance Report

Run standalone performance analysis on eager or graph mode traces:

```bash
python TraceLens/Reporting/generate_perf_report_pytorch_vllm.py \
  --profile_json_path /path/to/trace.json \
  --output_xlsx_path perf_report.xlsx \
  --group_by_parent_module \
  --enable_pseudo_ops
```

### Step 4: Generate TraceDiff Report

Compare two tracefiles and analyze execution differences via Lowest Common Ancestor (LCA) analysis:

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

### Step 5: Automated Trace Comparison *(Coming Soon)*

Run the complete trace comparison workflow:
```bash
python TraceLens/tools/compare_traces.py trace1.json trace2.json --output results/
```
*(More details available in [Automated Trace Comparison](#automated-trace-comparison-flow))*

### Step 6: Agentic Performance Analysis

Leverage autonomous trace analysis workflows:
- See [Agentic Analysis README]() for detailed setup
- Advanced optimization recommendations generated automatically


---

## 📚 Examples & Use Cases

*Example notebooks and scripts coming soon* 🔄

---

## 🔬 Technical Details

### [Automated Trace Comparison Flow](#automated-trace-comparison-flow)

Automated end-to-end workflow for comparing two inference execution traces, generating actionable insights on performance differences and optimization opportunities.

### [Roofline Analysis](#roofline-analysis)

Custom roofline models for inference operations with accurate memory bandwidth and compute utilization metrics, including prefill/decode-aware analysis.

### [Steady State Region Identification](#steady-state-region-identification)

Algorithms to automatically detect and isolate steady-state execution regions, enabling representative performance analysis without warmup/cooldown noise.

### [Trace Splitting for Performance Analysis](#trace-splitting-for-performance-analysis)

Fine-grained trace decomposition into individual execution steps, steady-state windows, and phase-specific (prefill-decode/decode) analyses for targeted optimization.

### [Trace Availability-Analysis Trade-off](#trace-availability-analysis-trade-off)

Balancing complete trace capture versus analysis complexity; techniques for selective event recording to reduce memory while maintaining analysis fidelity.

### [Agentic Workflow](#agentic-workflow)

Autonomous trace analysis framework leveraging AI/ML agents for intelligent optimization recommendations and performance insights generation.

---

## 🗺️ Roadmap

### 🔄 In Progress

- [ ] Extend graph execution analysis using TraceDiff reports from eager phase
- [ ] Refined graph execution ↔ capture phase linkage for inference workflows
- [ ] Automated trace comparison flow (scheduled for Q2 2026)

### 🚀 Future Improvements

- [ ] Multi-framework support (PyTorch, TensorFlow, JAX inference engines)
- [ ] Real-time trace analysis and streaming profiling
- [ ] Advanced ML-based bottleneck detection
- [ ] Integration with optimization frameworks
- [ ] Enhanced visualization dashboards

---

**Last Updated:** February 2026  
**Maintainers:** AMD-AGI TraceLens Team  
**Repository:** [github.com/AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal)