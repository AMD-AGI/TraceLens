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

The orchestrator runs against a PyTorch profiler trace. Collection is workload-specific:

- **Generic Eager Traces**: Instrument your loop with `torch.profiler.profile(...)`, enabling CPU-side call-stack and shape capture (`with_stack=True`, `record_shapes=True`). Profile a representative steady-state window (a handful of steps, post-warmup) and log the trace with `prof.export_chrome_trace(...)`. A single rank's trace is enough for per-rank analysis. See [Collect a trace](../../../docs/how-to/generate-perf-report-pytorch.md#collect-a-trace).
- **Inference Traces with Graph Capture**: Collection has framework-specific requirements. Follow guidelines in [Generate a PyTorch inference report](../../../docs/how-to/generate-perf-report-pytorch-inference.md). The [Profiling skill](../Profiling/README.md) automates vLLM/SGLang/ATOM benchmarking and PyTorch profiler trace collection via [Magpie](https://github.com/AMD-AGI/Magpie), producing analysis-ready traces. For graph-mode workloads you produce two artifacts: a graph-replay trace and a graph-capture folder. In inference mode with execution mode `graph replay + capture`, TraceLens merges call-stack and shape information from the capture folder into the replay tree before analysis.

### Establish a hardware baseline

Roofline analysis compares each measured kernel against your GPU's max-achievable
TFLOPS and memory bandwidth, so it needs a `<platform>.json` arch file for your
hardware. Bundled arch files ship with the package. If your platform isn't
included, or you want stack-specific measured values instead of published specs,
generate benchmark-derived peak TFLOPS and memory bandwidth with the GPU
microbenchmarking suite. It writes the arch JSON in the shape the roofline
expects.

## Run the agent from a chat

Invoke the agent from any chat session with a capable model using one of the following prompts.

```{note}
The orchestrator skills are portable and work with agentic runners that support skill-file discovery (for example, Claude Code, Cursor, or Codex).
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

### Run the agent from a YAML config

If you already have a run config, point the agent at it instead of answering the
prompts:

```text
Follow the analysis orchestrator installed with TraceLens and run the full
agentic analysis workflow from config <run.yaml>, <scope>
```

Replace `<scope>` with `standalone`, `comparative`, `semantic`, or `multi_trace`. You can
leave the scope out when the config already implies it: a single job runs as `standalone`,
two jobs with `compare_to` on the first job run as `comparative`, and a config with an
`aggregation` block runs as `multi_trace`.

| Scope | Config shape |
|---|---|
| `standalone` | One job. |
| `comparative` | Two jobs, each with an optional `capture_folder`. `compare_to` on the first job can name the second job instead of repeating its paths. |
| `semantic` | Two graph-replay-only jobs with `analysis_mode: inference` and no `capture_folder`. |
| `multi_trace` | Two or more jobs on the same platform, each with a `phase` and a `node` (its index within the phase), and an `aggregation` block with policy `combine_by_node`. Each job is one node, `<phase>_<index>`, and the first job is the one that kernel fusion and system analysis focus on. |

For example, a `multi_trace` config for a prefill trace and a decode trace:

```yaml
analysis_mode: inference
output_dir: /abs/path/analysis_output
aggregation:
  policy: combine_by_node
jobs:
  - phase: prefill
    node: 0
    trace_path: /abs/path/prefill_0.json.gz
    platform: MI300X
  - phase: decode
    node: 0
    trace_path: /abs/path/decode.json.gz
    platform: MI300X
```

Every job needs a `platform`, and all trace, capture folder, and output paths must be
absolute. For the full schema and more examples, see
[Generate performance reports from a YAML config](./generate-perf-report-from-config.md).

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

### Example output

The excerpts below show one detailed-analysis card from each tier of a standalone
report.

*Note: All performance data shown here are example outputs from TraceLens, intended
to illustrate the agent's capabilities. They are not official performance
benchmarks.*

#### Compute kernel analysis

> <a id="detailed-analysis-compute-p1"></a>
> <!-- reasoning-candidate tier=compute rank=1 -->
> **🔴 P1: Compute-bound BF16 GEMMs underrunning roofline (Tensile)**
>
> **Identification:** 18 BF16 `aten::mm` shapes were flagged as the dominant GEMM
> cluster in the trace, accounting for 45.7 s of GPU kernel time (~80.6% of compute
> time). All flagged operations dispatch through the **Tensile** backend with
> `bound_type = compute`. The cluster spans the large MLP shapes (M=24576, K=8192,
> N=28672), the gate/up shapes (N=10240), the QKV/output shapes (N=8192), and the
> LM-head shapes (N=128256). (source: `gemm_metrics.json` →
> `category_findings[0].members[]`, `operations[].library`,
> `operations[].efficiency.bound_type`)
>
> **Data:**
>
> | Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
> |-----------|------|-------------|-------------|-----------|------|-------|------------|------------|-------|
> | aten::mm | (24576,8192) × (8192,28672) bf16 | ... | ... | 7607.463 | 13.42 | 320 | 5059.76 | 68.74% of 708 TFLOPS | compute-bound |
> | aten::mm | (24576,8192) × (8192,28672) bf16 | ... | ... | 6636.191 | 11.70 | 320 | 5059.76 | 79.04% of 708 TFLOPS | compute-bound |
> | aten::mm | (28672,24576) × (24576,8192) bf16 | ... | ... | 6313.337 | 11.13 | 320 | 5059.76 | 82.70% of 708 TFLOPS | compute-bound |
> | aten::mm | (24576,28672) × (28672,8192) bf16 | ... | ... | 6071.557 | 10.71 | 320 | 5059.76 | 85.99% of 708 TFLOPS | compute-bound |
>
> **Reasoning for Slowdown:** Every flagged shape has very high arithmetic
> intensity (FLOPS/Byte 3510–5863), so each GEMM is correctly compute-bound against
> the BF16 matrix-FP roofline (708 TFLOPS) rather than HBM-bound. The achieved
> TFLOPS/s sit between 486.7 and 613.5, which is 68.7%–86.7% of peak — the heaviest
> shape (`(24576,8192) × (8192,28672)`, 7.6 s of kernel time, count = 320) lands at
> only 68.7%, and a second 6.6 s instance of the same shape only reaches 79.0%,
> indicating the same logical GEMM is hitting more than one Tensile kernel and at
> least one is sub-optimal.
>
> **Resolution:** Tile / wave-occupancy tuning targets the specific Tensile kernels
> selected for these shapes so the same FLOPs are issued under a better-utilized
> MFMA pipeline, raising achieved TFLOPS/s without changing the algorithm.
> Narrowing precision (BF16 → FP8/FP4 where the model tolerates it) doubles or
> quadruples the matrix roofline (708 → 1273 TFLOPS at FP8, higher at FP4),
> directly lowering the compute floor for these compute-bound shapes. Generate
> reproducers for the kernel team for the lowest-efficiency shapes (68.7%, 69.7%,
> 74.0%) so the Tensile selector can be revisited for those tile/M-N-K
> combinations.
>
> **Impact estimate:**
> <!-- impact-begin kind=detail_estimate low=12.97 mid=15.12 high=17.27 -->
> - Low end impact_score: 12.97
> - High end impact_score: 17.27
> <!-- impact-end -->

#### Kernel fusion analysis

> <a id="detailed-analysis-fusion-P1"></a>
> <!-- reasoning-candidate tier=fusion rank=1 -->
> **🔴 P1: Unfused Attention**
>
> **Identification:** Attention runs as separate bmm, softmax and bmm kernels, so
> the 5185x5185 score matrix round-trips through HBM between launches (source:
> `fusion_candidates.json`).
>
> **Data:**
>
> | Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
> |---|---|---|---|---|---|---|---|---|---|
> | aten::bmm | (16,5185,64) bf16<br>(16,64,5185) bf16 | ... | ... | 40.460 | 4.45 | 120 | 62.46 | 49.39% of 5.3 TB/s | memory-bound |
> | aten::_softmax | (16,5185,5185) bf16 | ... | ... | 153.698 | 16.92 | 120 | — | — | — |
> | aten::bmm | (16,5185,5185) bf16<br>(16,5185,64) bf16 | ... | ... | 55.056 | 6.06 | 120 | 62.46 | 36.51% of 5.3 TB/s | memory-bound |
> | aten::bmm | (12,5,64) bf16<br>(12,64,5) bf16 | ... | ... | 0.356 | 0.04 | 60 | 2.41 | 0.05% of 5.3 TB/s | memory-bound |
> | aten::_softmax | (12,5,5) bf16 | ... | ... | 0.295 | 0.03 | 60 | — | — | — |
> | aten::bmm | (12,5,5) bf16<br>(12,5,64) bf16 | ... | ... | 0.385 | 0.04 | 60 | 2.41 | 0.05% of 5.3 TB/s | memory-bound |
>
> **Resolution:** Call a vendor library's fused attention if one exists; otherwise
> write a fused kernel.
>
> **Impact estimate:**
> <!-- impact-begin kind=detail_estimate low=11.5 mid=13.5 high=15.5 -->
> - Low end impact_score: 11.50
> - High end impact_score: 15.50
> <!-- impact-end -->

#### System-level analysis

> <a id="detailed-analysis-system-p1"></a>
> <!-- reasoning-candidate tier=system rank=1 -->
> **🔴 P1: Exposed Collective Communication**
>
> **Identification:** The compute/communication overlap threshold tripped. Overlap
> is 0.0%, below the 70% target. The exposed-communication threshold did not trip,
> at 4.87% of GPU time against a 5% gate. The trace has 8064 collective all-reduce
> kernels (two ranks, bf16 payload). That is about 63 all-reduces per decode step
> over 128 steps. (source: `multi_kernel_metrics.json` →
> `overlap_assessment.overlap_percent` < 70,
> `nccl_blocking_assessment.exposed_percent_of_total`, `nccl_summary.total_count`,
> `nccl_summary.top_ops[].name`)
>
> **Data:**
>
> | Metric | Value |
> |--------|-------|
> | Exposed Communication Time | 188.72 ms |
> | Exposed Communication % | 4.87% |
> | Compute/Comm Overlap % | 0.0% |
> | Total Communication Time | 188.72 ms |
>
> **Reasoning for Slowdown:** Exposed communication time equals total communication
> time (188.72 ms). No collective kernel ran concurrently with any compute kernel.
> Each tensor-parallel all-reduce is therefore a serial step between layers, and
> the GPU does no useful compute while it runs. The individual kernels are short
> (23.4 us on average, 49.36 us at most). The cost comes from the count: roughly 63
> serialized all-reduces in every decode step. Exposed communication sits just
> under the 5% gate. Even so, it is the single largest overlappable block of time
> this analysis can see.
>
> **Resolution:** Splitting the batch into two micro-batches gives the GPU
> independent work during each all-reduce. While micro-batch A runs its all-reduce
> on the communication stream, micro-batch B can run its attention or MoE compute
> on the compute stream. This converts serialized collective time into overlapped
> time without changing the collective payload or algorithm. The expected movement
> is a higher compute/communication overlap ratio, up from 0.0%, and lower exposed
> communication time, down from 188.72 ms. Payload compression is not recommended
> here because the problem is scheduling (zero overlap), not the size of each
> transfer.
>
> **Impact estimate:** Impact estimate is not quantifiable from trace data.

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

1. Query user inputs, or read a run YAML (environment setup is always asked).
2. Generate the performance reports with one call on the completed run config.
3. Check trace quality, with a fallback for graph-under-recorded traces.
4. Prepare category data (GPU utilization, top operations, multi-kernel
   data, category filtering) and extract fusion candidates.
5. Run system-level analysis (CPU/idle, multi-kernel, and kernel fusion) in
   parallel.
6. Run the compute-kernel sub-agents in parallel.
7. Aggregate the per-category findings into a globally ranked list.
8. Validate sub-agent outputs for time sanity, efficiency anomalies, and coverage.
9. Prepare report data and identify the model.
10. Render the performance plot when no `agent_extension.py` is present.
11. Generate the final `analysis.md` report.

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
- [Inference performance analysis](../conceptual/inference-analysis.md)
- [Analyze traces with the TraceLens SDK](./sdk-analysis.md)
- [Trace2Tree data model](../conceptual/trace2tree.md)
- [GEMM analysis](../conceptual/gemm-analysis.md)