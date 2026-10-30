---
name: model-roofline-projection
description: >-
  Projects a kernel-by-kernel roofline (FLOPs, bytes, compute-bound vs
  memory-bound time) for a whole model from its checkpoint config and modeling
  source alone, with no trace and no GPU run. Use when the user asks to
  estimate or size a model's roofline/FLOPs/bytes/arithmetic-intensity before
  it runs, wants to know what GPU or how many GPUs a checkpoint needs, or
  wants that projection for one GPU's share under a tensor-, pipeline-, or
  expert-parallel split (Hugging Face plans, vLLM, or SGLang). Measuring a
  trace that already exists, visualizing the architecture graph, and
  collecting benchmark traces are other skills.
license: MIT
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model roofline projection

Drive **`TraceLens_model_roofline_projection`** so a Hugging Face checkpoint, a local checkpoint directory, or a whitelisted modeling repo gets a kernel-by-kernel roofline table and a model-level rollup, without a trace and without a GPU run. The tool builds the same merged Model Explorer graph the visualizer draws (statically parsing `modeling_*.py`), stops expansion at every op with a direct `TraceLens.PerfModel` formula, and costs each one's FLOPs/bytes against an optional GPU architecture spec.

## Full procedure

Follow **[reference.md](reference.md)** for every step: inputs, install check, the single CLI invocation, GPU architecture selection, device sharding, output formats, verification, and footguns. The flag list and op-coverage caveats live in [the how-to guide](../../../../../docs/how-to/generate-model-roofline-projection.md).

## Workflow index

```
0. Resolve every stated option (source, shape, GPU arch, sharding source and counts, output format)
1. Ensure the Visualizer extra is installed
2. Run one TraceLens_model_roofline_projection command
3. Confirm the table printed (or the CSV/JSON artifact) and read the rollup line
4. Report the roofline numbers and the modeled-element coverage to the user
```

## Rules

- **One tool.** Run `TraceLens_model_roofline_projection`. Fall back to `python -m TraceLens.Visualizer.model_explorer_export.roofline_cli` only when that console script is absent. Do not hand-compute FLOPs/bytes or rebuild the graph in ad-hoc Python.
- **CPU only, no trace.** This is a static projection: no trace file is read, no checkpoint weights are loaded, and no GPU is required to run the tool itself. A trace that already exists is the perf-report / analysis-orchestrator workflow, not this one.
- **No GPU arch means no roofline time.** Without `--gpu-arch-name` or `--gpu-arch-json-path`, FLOPs and bytes are still printed but every roofline-time column is empty. Ask which GPU (`MI300X`, `MI325X`, or a custom arch JSON) before reporting "no roofline time" as the final answer, unless the user only asked for FLOPs/bytes.
- **Shape is `--batch-size`/`--seq-len`, or `--dim KEY=VALUE` for anything more specific** (e.g. `--dim past_len=1024` for a decode step's cache length). `--dim` wins over `--batch-size`/`--seq-len` for the same key.
- **Device sharding mirrors the visualizer.** `--tp`/`--pp`/`--ep` project one GPU's own share of the work; `--engine vllm`/`--engine sglang` switches whose parallel-layer classes decide the split. No `--engine` uses the checkpoint's own Hugging Face plans. Omitting a flag leaves that axis unsplit (count 1).
- **Unmodeled is reported, not hidden.** Embeddings, convolutions, and anything else with no `TraceLens.PerfModel` formula show up as unmodeled rows and lower the rollup's "% of output elements modeled" line. Surface that coverage figure in the summary rather than only the top-N table.
- **Config overrides, attention, and runtime regime are first-class**, exactly as in the model visualizer: `--config-override KEY=VALUE`, `--attention`, `--phase`/`--cache`/`--sequence-layout`/`--mode`/`--device`. An option the user named is never dropped because some other default exists.
- **Graph export and benchmarking are other skills.** A request to open the architecture in Model Explorer is `model-visualizer`. A request to collect a PyTorch profiler trace or run a benchmark is Magpie profiling. A request to analyze a trace that already exists is the analysis orchestrator.

## Primary outputs

- **Deliverable:** the stdout table (ranked by `total_roofline_time_us`, top 20 rows by default) plus the rollup line(s): total FLOPs/bytes, modeled vs. unmodeled op counts, modeled-output-element share, and, when a GPU arch was given, the compute-bound/memory-bound time split.
- **Optional:** the full per-op table as CSV (`--output-csv`) or JSON with the rollup included (`--output-json`).

## Skill layout

`TraceLens/Agent/Visualizer/skills/model-roofline-projection/` (`SKILL.md`, `reference.md`, `skill-card.md`). Cursor project discovery reads `.cursor/skills/`; symlink or copy this folder there for automatic pickup. Headless runs should open this directory inside the installed TraceLens package.
