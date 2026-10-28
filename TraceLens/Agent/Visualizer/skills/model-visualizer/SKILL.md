---
name: model-visualizer
description: >-
  Exports a Hugging Face checkpoint or local modeling source to an interactive
  Model Explorer architecture graph as a standalone HTML file, or serves that
  graph locally. Use when the user asks to visualize a model, draw an
  architecture or computation graph, open a checkpoint in Model Explorer,
  export visualize_model_in_explorer HTML, show symbolic tensor shapes,
  override config keys, choose an attention implementation (eager, sdpa, flash
  attention), or shard with Hugging Face plans, vLLM, or SGLang (tensor,
  pipeline, or expert parallel). Trace files, roofline reports, and analysis.md
  belong to the analysis orchestrator. Collecting benchmark traces belongs to
  Magpie benchmark profiling.
license: MIT
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model visualizer

Drive **`visualize_model_in_explorer`** so a Hugging Face checkpoint, a local checkpoint directory, or a whitelisted modeling repo becomes one Model Explorer graph. The export is static analysis plus an optional meta-device shape pass. It runs on CPU and does not load checkpoint weights.

## Full procedure

Follow **[reference.md](reference.md)** for every step: inputs, install check, the single CLI invocation, config overrides, attention selection, Hugging Face / vLLM / SGLang sharding, the rest of the flag table, verification, and footguns. The tool's long flag list lives in [the Visualizer README](../../../../Visualizer/README.md).

## Workflow index

```
0. Resolve every stated option (source, output, viewer, regime, attention, config overrides, sharding source and counts)
1. Ensure the Visualizer extra is installed
2. Run one visualize_model_in_explorer command
3. Confirm the HTML or JSON artifact (or the viewer URL)
4. Tell the user the path or URL
```

## Rules

- **One exporter.** Run `visualize_model_in_explorer`. Fall back to `python -m TraceLens.Visualizer.model_explorer_export.cli` only when that console script is absent. Do not rebuild the graph in ad-hoc Python.
- **CPU only.** Do not request a GPU, set `HIP_VISIBLE_DEVICES`, or download weight shards. Config and modeling source are enough.
- **Every stated option is a flag.** Walk the decision procedures in reference.md before building the command. Config overrides, the attention implementation, and Hugging Face versus vLLM versus SGLang sharding are first-class. An option the user named is never dropped because some other default exists.
- **Default artifact.** Write a self-contained `.html` file. Add `--serve` only when the user asks to view the graph live.
- **Defaults stay on** unless the user opts out: `--auto-venv`, `--meta-shapes`, symbolic shapes, `--drop-constants` for HTML. Leave `--attention` unset when the user did not name an implementation, so the checkpoint's own setting survives.
- **`--device` is the accelerator being drawn** (default `hip`). It does not probe the machine running the export.
- **Traces and benchmarks** are other skills. A `.json` Chrome trace, a request for `analysis.md`, or a Magpie profiling run is not this workflow.

## Primary outputs

- **Deliverable:** `<model>.html` in the working directory, or the path passed with `-o`. Device partitions write `*_subgraph_<i>.html` beside it.
- **Optional:** payload `.json` (`-o model.json`), operator JSON (`--operators-json`), or a local viewer URL when `--serve` was requested.

## Skill layout

`TraceLens/Agent/Visualizer/skills/model-visualizer/` (`SKILL.md`, `reference.md`, `skill-card.md`). Cursor project discovery reads `.cursor/skills/`; symlink or copy this folder there for automatic pickup. Headless runs should open this directory inside the installed TraceLens package.
