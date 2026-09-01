<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Generate optimization recommendations with the TraceLens Agent
```{meta}
:description: Learn how to run the TraceLens Agent, an agentic workflow that turns a GPU trace into a prioritized, stakeholder-facing performance optimization report.
:keywords: TraceLens, TraceLens Agent, agentic analysis, performance optimization, roofline, kernel fusion, GEMM, ROCm, AMD Instinct, optimization report
```

This topic shows how to run the *TraceLens Agent*, an agentic performance-analysis
workflow that reads a GPU trace and produces a single stakeholder-facing report,
`analysis.md`, organized as a prioritized bottleneck list. Findings are ranked and
grouped into three tiers: compute kernel optimizations, kernel fusion
opportunities, and system-level optimizations. Each finding carries the supporting
evidence, the reasoning behind the call-out, and a concrete resolution.

The agent combines a structured, skill-driven workflow with codified TraceLens
analysis for repeatable, reliable results.

## Analysis modes

The agent runs in one of two modes:

- **Standalone**: Single-trace roofline analysis. Use this when you have one trace
  and want to find where performance falls short of hardware limits, find system bottlenecks or fusion opportunities. This is the recommended default.
- **Comparative**: Two-trace gap analysis. The agent compares a primary trace
  against a reference trace, for example a different platform or a tuned config,
  and identifies inefficiencies in the primary trace relative to the reference.
  Comparative analysis works best when both traces come from the same framework.
  Cross-framework comparisons can produce misleading gap estimates because of
  structural differences in operation call stacks.

Support depends on the execution mode of the traced workload:

| Execution mode | Standalone | Comparative |
|---|---|---|
| Eager | Supported | Supported |
| Graph + capture | Supported | Supported |
| Graph | Not supported | Not supported |

## Before you begin

Complete the following setup steps before running the agent.

### Install TraceLens

Install locally, or into your container or virtual environment (see
[Install TraceLens](../install/install.md)):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

### Collect a trace

The orchestrator runs against a single `torch.profiler` trace (`.json` or
`.json.gz`). Collection is workload-specific:

- **Generic Eager Traces**: Instrument your loop with
  `torch.profiler.profile(...)`, enabling CPU-side call-stack and shape capture
  (`with_stack=True`, `record_shapes=True`). Profile a representative steady-state
  window of a handful of post-warmup steps, then log the trace with
  `prof.export_chrome_trace(...)`. A single rank's trace is enough for per-rank
  analysis.
- **Inference Traces with Graph Capture**: Collection has framework-specific
  requirements. Follow
  [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md).
  The Profiling Skill automates
  vLLM, SGLang, and ATOM benchmarking and PyTorch profiler trace collection using
  Magpie, producing analysis-ready traces. For
  graph-mode workloads you produce two artifacts: a graph-replay trace and a
  graph-capture folder. In inference mode with execution mode
  `graph replay + capture`, TraceLens merges call-stack and shape information from
  the capture folder into the replay tree before analysis.

### Establish a hardware baseline

Roofline analysis compares each measured kernel against your GPU's max-achievable
TFLOPS and HBM bandwidth, so it needs a `<platform>.json` arch file for your
hardware. Bundled arch files ship with the package. If your platform isn't
included, or you want stack-specific measured values instead of published specs,
generate benchmark-derived peak TFLOPS and HBM bandwidth with the GPU
microbenchmarking suite. It writes the arch JSON in the shape the roofline
expects.

## Run the agent from a chat

Invoke the agent from any chat session with a capable model using one of the following prompts.

```{note}
The orchestrator skills are portable and work with agentic runners that support skill-file discovery.
```

In a chat with a capable model, invoke one of:

- Standalone (single trace):

  ```text
  Follow the analysis orchestrator installed with TraceLens and run the full
  agentic analysis workflow on <path_to_trace.json>
  ```

- Comparative (two traces)

  ```text
  Follow the analysis orchestrator installed with TraceLens and run the full
  agentic analysis workflow on <path_to_trace1.json> and <path_to_trace2.json>
  ```

If prompted, provide the trace file path, the platform of the first trace, the
analysis mode (`default` for training and eager inference outside vLLM, SGLang, and ATOM,
or `inference` for vLLM, SGLang, or ATOM), the execution mode and capture-folder path for
inference, environment details (node, container, or virtual environment), and an
optional output directory.

## Read the results

Only `analysis.md` is intended for end-user review. Everything else under
`analysis_output/` is agent internal: intermediates the orchestrator and
sub-agents pass between steps.

Every report has the same top-level structure:

1. *Executive summary*: A one-paragraph workload characterization, a metrics
   table (total time, compute percentage, idle percentage, exposed-communication
   percentage, and top bottleneck category), and a representative chart.
2. *Compute kernel optimizations*: Top-operations table followed by per-category
   priority items (`P1`, `P2`, and so on) sorted by `impact_score`. Each item
   carries an Insight, Action, and Impact triplet.
3. *Kernel fusion opportunities* (experimental): Candidate modules to fuse.
4. *System-level optimizations* (experimental): Idle time, memory-copy overhead,
   and compute/communication overlap.
5. *Detailed analysis*: Per-priority-item drill-down with identification
   rationale, data tables, and reasoning.

### Programmatic interface

Every report embeds HTML comment markers so a downstream system can consume it
without parsing prose:

- `<!-- impact-begin kind=p_item category=<cat> low=<x> mid=<y> high=<z> -->` …
  `<!-- impact-end -->` wraps each priority item's Impact line; `mid` is the
  canonical `impact_score` and `<cat>` is the analyzer category (`gemm`,
  `sdpa_fwd`, `elementwise`, `norm_fwd`, and so on).
- `<!-- impact-begin kind=top_ops -->` … `<!-- impact-end -->` wraps the Top
  Operations table at the start of the Compute kernel optimizations section.
- Per-item anchors (`#detailed-analysis-compute-pN`, `#detailed-analysis-fusion-PN`,
  `#detailed-analysis-system-pN`) link each summary card to its detailed reasoning
  section.

## Analysis workflow

The workflow splits into three independently composable tiers:

- **System-level optimizations**: Issues that affect the GPU pipeline as a whole,
  such as idle time, memory-copy overhead, collective-communication blocking, and
  compute/communication overlap.
- **Kernel fusion opportunities** (experimental): Multi-kernel modules that could be
  fused, with estimated savings.
- **Compute kernel optimizations**: Per-category kernel analysis (GEMM, attention,
  elementwise, etc.) focused on individual operation efficiency.

The analysis orchestrator skill coordinates the workflow: it queries user
inputs, runs TraceLens to pre-compute trace data, invokes the system-level and
compute-kernel sub-agents in parallel, aggregates and validates their findings,
identifies the model, and generates the prioritized report. The orchestrator and
its sub-agents run on a capable reasoning model declared in each agent's front
matter.

The high-level steps are:

1. Query user inputs (comparison scope, trace paths, platforms, analysis mode,
   environment setup).
2. Generate the performance report (branching on analysis mode and comparison
   scope).
3. Prepare category data (GPU utilization, top operations, tree data, multi-kernel
   data, category filtering) and extract fusion candidates.
4. Run system-level analysis (CPU/idle, multi-kernel, and kernel fusion) in
   parallel.
5. Run the compute-kernel sub-agents in parallel.
6. Aggregate the per-category findings into a globally ranked list.
7. Validate sub-agent outputs for time sanity, efficiency anomalies, and coverage.
8. Prepare report data and identify the model.
9. Generate the final `analysis.md` report.

### Sub-agents

System-level sub-agents cover CPU and idle analysis, memory-copy and
collective-communication patterns, and kernel fusion. Compute-kernel sub-agents
cover GEMM, scaled dot-product attention, elementwise,
reduction, Triton-compiled kernels, Mixture-of-Experts, normalization,
convolution, and a generic analyzer for uncategorized operations.

### Execution environments

During the input step the orchestrator asks whether you're running locally or on a
cluster, and builds the appropriate command prefixes automatically — direct
execution, a virtual-environment activation prefix, an SSH wrapper for a remote
node, or an SSH plus container-exec wrapper for a containerized node.

## Related topics

- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md)
- [Analyze traces with the TraceLens SDK](./sdk-analysis.md)
- [Trace2Tree data model](../conceptual/trace2tree.md)
- [GEMM analysis](../conceptual/gemm-analysis.md)