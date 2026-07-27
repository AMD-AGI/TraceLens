---
name: magpie-benchmark-profiling
description: >-
  Runs Magpie LLM inference benchmarks (vLLM or SGLang), applies TraceLens profiling
  patches and profiler tuning, collects PyTorch profiler traces on remote GPU nodes,
  verifies trace quality, and splits traces for TraceLens inference analysis. Use when
  the user asks to follow the Magpie Benchmark + Profiling skill, profile LLM inference,
  collect vLLM or SGLang traces, or prepare traces for the analysis orchestrator.
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Magpie benchmark + profiling

Drive end-to-end **Magpie** benchmark runs with **PyTorch profiler** traces suitable for **`TraceLens_generate_perf_report_pytorch_inference`** and the [Analysis orchestrator](../../Analysis/README.md): patched Docker images when needed, YAML and InferenceX script edits, targeted vs full profiling modes, remote execution, trace QA, and steady-state trace splitting.

## Full procedure

Follow **[reference.md](reference.md)** for every step (SSH/conda prompts, `build_docker_*.sh` usage, `EXTRA_VLLM_ARGS` / SGLang flags, `benchmark_lib.sh` / `benchmark_serving.py` patches, cleanup, split command, and pitfalls).

## Workflow index

```
0. Gather execution environment (node, conda env, conda prefix)
1. Read benchmark YAML; optionally build TraceLens-patched Docker image (vLLM or SGLang)
2. Ensure torch profiler enabled; apply common graph/capture flags; choose targeted vs full profiling
3. Run: python -m Magpie benchmark --benchmark-config <yaml> (long-running; monitor via docker)
4. Monitor container / logs on the remote node
5. Verify trace files and GPU kernel categories in torch_trace/
6. split_inference_trace_annotation on rank-0 trace; print (do not run) generate_perf_report_pytorch_inference.py command
```

## Rules

- Do **not** guess node names or conda envs — ask the user (see reference Step 0).
- Discover supported Docker image tags by **reading** `examples/custom_workflows/inference_analysis/build_docker_vllm.sh` / `build_docker_sglang.sh` at runtime; do not hardcode version matrices in chat.
- After Step 6, **print only** the suggested `generate_perf_report_pytorch_inference.py` invocation for the user; the skill does not substitute for their analysis run unless they ask.

## Primary outputs

- **`results/benchmark_{framework}_{timestamp}/`**: `torch_trace/`, logs, `benchmark_report.json`, etc. (see reference **Output Structure**).

## Skill layout

Bundled as: `TraceLens/Agent/Profiling/skills/magpie-benchmark-profiling/` (`SKILL.md`, `reference.md`). Cursor’s default project skill discovery uses `.cursor/skills/`; symlink or copy this folder there if you rely on automatic skill pickup.
