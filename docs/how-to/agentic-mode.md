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
grouped into three tiers — compute kernel optimizations, kernel fusion
opportunities, and system-level optimizations — and each finding carries the
supporting evidence, the reasoning behind the call-out, and a concrete resolution.

The agent combines a structured, skill-driven workflow with codified TraceLens
analysis for repeatable, reliable results.

## Analysis modes

The agent runs in one of two modes:

- *Standalone* — single-trace roofline analysis. Use this when you have one trace
  and want to find where performance falls short of hardware limits. This is the
  recommended default.
- *Comparative* — two-trace gap analysis. The agent compares a primary trace
  against a reference trace (for example, a different platform or a tuned config)
  and identifies inefficiencies in the primary trace relative to the reference.
  Comparative analysis works best when both traces come from the same framework;
  cross-framework comparisons can produce misleading gap estimates because of
  structural differences in operation call stacks.

Support depends on the execution mode of the traced workload:

| Execution mode | Standalone | Comparative |
|---|---|---|
| Eager | Supported | Supported |
| Graph + capture | Supported | Not supported |
| Graph | Not supported | Not supported |

## Before you begin

### Install TraceLens

Install locally, or into your container or virtual environment (see
[Install TraceLens](../install/install.md)):

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

### Collect a trace

The orchestrator runs against a single `torch.profiler` trace (`.json` or
`.json.gz`). Collection is workload-specific:

- *Training and eager inference traces* — instrument your loop with
  `torch.profiler.profile(...)`, enabling CPU-side call-stack and shape capture
  (`with_stack=True`, `record_shapes=True`). Profile a representative steady-state
  window (a handful of post-warmup steps) and export with
  `prof.export_chrome_trace(...)`. A single rank's trace is enough for per-rank
  analysis.
- *vLLM or SGLang inference traces* — collection has framework-, version-, and
  execution-mode-specific requirements. Follow
  [Generate a PyTorch inference performance report](./generate-perf-report-pytorch-inference.md).
  For graph-mode workloads you produce two artifacts: a graph-replay trace and a
  graph-capture folder. In inference mode with execution mode
  `graph replay + capture`, TraceLens merges call-stack and shape information from
  the capture folder into the replay tree before analysis.

### Establish a hardware baseline

Roofline analysis compares each measured kernel against your GPU's max-achievable
TFLOPS and HBM bandwidth, so it needs a `<platform>.json` arch file for your
hardware. Bundled arch files ship with the package; if your platform isn't
included — or you want stack-specific measured values instead of published specs —
generate benchmark-derived peak TFLOPS and HBM bandwidth with the GPU
microbenchmarking suite, which writes the arch JSON in the shape the roofline
expects.

## Run the agent from a chat

```{note}
The examples use the Cursor IDE and CLI, but the orchestrator skills are portable
and also work with other agentic runners that support skill-file discovery.
```

In a chat with a capable model, invoke one of:

- Standalone (single trace):

  ```text
  Follow the analysis orchestrator installed with TraceLens and run the full
  agentic analysis workflow on <path_to_trace.json>
  ```

- Comparative (two traces) — always pass the *baseline* trace first:

  ```text
  Follow the analysis orchestrator installed with TraceLens and run the full
  agentic analysis workflow on <path_to_trace1.json> and <path_to_trace2.json>
  ```

If prompted, provide the trace file path, the platform of the first trace, the
analysis mode (`default` for training and non-vLLM/SGLang eager inference, or
`inference` for vLLM/SGLang), the execution mode and capture-folder path for
inference, environment details (node, container, or virtual environment), and an
optional output directory.

## Run the agent headless (CLI)

Use the `agent` CLI to run the orchestrator non-interactively. Install it with:

```bash
curl https://cursor.com/install -fsS | bash
```

Pass every parameter inline so no interactive prompts are needed — useful for
batch runs and continuous-integration pipelines. For example, for a default
standalone run on a remote node with a container:

```bash
agent --model <model> --print --force --trust \
    "Follow the analysis orchestrator installed with the TraceLens pip package
    (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package
    installation directory) and run the full agentic analysis workflow on
    <path_to_trace.json> with platform <platform>, analysis mode default,
    node <node>, container <container>, output to <output_dir>"
```

For vLLM or SGLang inference, set `analysis mode inference` and add
`execution mode eager`, or `execution mode graph replay + capture` together with
`capture folder <path_to_capture_folder>`.

## Read the results

Only `analysis.md` is intended for end-user review. Everything else under
`analysis_output/` is agent internals — intermediates the orchestrator and
sub-agents pass between steps (for example `system_findings/`,
`category_findings/`, `category_data/`, `metadata/`, `perf_report_csvs/`, and
`perf_report.xlsx`).

Every report has the same top-level structure:

1. *Executive summary* — a one-paragraph workload characterization, a metrics
   table (total time, compute percentage, idle percentage, exposed-communication
   percentage, and top bottleneck category), and a performance-improvement chart.
2. *Compute kernel optimizations* — a top-operations table followed by per-category
   priority items (`P1`, `P2`, and so on) sorted by `impact_score`, each with an
   Insight, Action, and Impact triplet.
3. *Kernel fusion opportunities* (experimental) — candidate modules to fuse.
4. *System-level optimizations* (experimental) — idle time, memory-copy overhead,
   and compute/communication overlap.
5. *Detailed analysis* — per-priority-item drill-down with identification
   rationale, data tables, and reasoning.

### Programmatic interface

Every report embeds HTML comment markers so a downstream system can consume it
without parsing prose:

- `<!-- impact-begin ... -->` … `<!-- impact-end -->` wraps each priority item's
  Impact line; `mid` is the canonical `impact_score` and the category is the
  analyzer category (`gemm`, `sdpa_fwd`, `elementwise`, `norm_fwd`, and so on).
- Per-item anchors (`#detailed-analysis-compute-pN`, `#detailed-analysis-fusion-PN`,
  `#detailed-analysis-system-pN`) link each summary card to its detailed reasoning
  section.

## How the analysis works

The workflow splits into three independently composable tiers:

- *System-level optimizations* — issues that affect the GPU pipeline as a whole,
  such as idle time, memory-copy overhead, collective-communication blocking, and
  compute/communication overlap.
- *Kernel fusion opportunities* (experimental) — multi-kernel modules that could be
  fused, with estimated savings.
- *Compute kernel optimizations* — per-category kernel analysis (GEMM, attention,
  elementwise, and so on) focused on individual operation efficiency.

The *analysis orchestrator* skill coordinates the workflow: it queries user
inputs, runs TraceLens to pre-compute trace data, invokes the system-level and
compute-kernel sub-agents in parallel, then validates outputs, aggregates
findings, and generates the prioritized report. The orchestrator and its
sub-agents run on a capable reasoning model declared in each agent's front matter.

The high-level steps are:

1. Query user inputs (comparison scope, trace paths, platforms, analysis mode,
   environment setup).
2. Generate the performance report (branching on analysis mode and comparison
   scope).
3. Prepare category data (GPU utilization, top operations, tree data, multi-kernel
   data, category filtering) and extract fusion candidates.
4. Identify the model.
5. Run system-level analysis (CPU/idle, multi-kernel, and kernel fusion) in
   parallel.
6. Run the compute-kernel sub-agents in parallel.
7. Validate sub-agent outputs for time sanity, efficiency anomalies, and coverage.
8. Aggregate results across all three tiers.
9. Generate the final `analysis.md` report.

### Sub-agents

System-level sub-agents cover CPU and idle analysis, memory-copy and
collective-communication patterns, and kernel fusion. Compute-kernel sub-agents
cover GEMM (`mm`, `bmm`, `addmm`), scaled dot-product attention, elementwise,
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
- [Analyze performance with TreePerf](./tree-perf-analysis.md)
- [GEMM analysis](../conceptual/gemm-analysis.md)
- [The Trace2Tree data model](../conceptual/trace2tree.md)
