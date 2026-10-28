<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Visualize a model with the TraceLens Agent
```{meta}
:description: Learn how to run the TraceLens model visualizer agent, which exports a Hugging Face checkpoint to a standalone Model Explorer architecture graph.
:keywords: TraceLens, model visualizer, Model Explorer, Hugging Face, architecture graph, tensor parallelism, agent
```

This topic shows how to run the *model visualizer* agent. The agent exports a
Hugging Face checkpoint, a local checkpoint directory, or a whitelisted modeling
repo to an interactive Model Explorer graph. The export parses the model's
`modeling_*.py` source and fills symbolic tensor shapes. It runs on CPU and
does not load checkpoint weights.

The skill and the step-by-step procedure live in the package at
`TraceLens/Agent/Visualizer/`. The exporter's flag reference is
`TraceLens/Visualizer/README.md`.

## Before you begin

Install TraceLens with the Visualizer extra (see
[Install TraceLens](../install/install.md)):

```bash
pip install -e ".[Visualizer]"
```

From a published install:

```bash
pip install "TraceLens[Visualizer] @ git+https://github.com/AMD-AGI/TraceLens.git"
```

You need Python 3.10 or newer and network access the first time a Hub model is
fetched. No GPU is required. Set `HF_TOKEN` for a gated checkpoint.

## Run the agent from a chat

```{note}
The skill is portable and works with agentic runners that support skill-file
discovery (for example, Claude Code, Cursor, or Codex).
```

In a chat with a capable model, invoke:

```text
Follow the model visualizer installed with TraceLens and export <model> to <path>.html
```

`<model>` is a Hugging Face id or a local checkpoint directory. If prompted,
provide the output path, whether you want a live viewer instead of a file, the
runtime regime (packed sequences, decode, warm cache, training), the attention
implementation (`eager`, `sdpa`, `flash_attention_2`, or another name; omit it
to keep the checkpoint's), any `KEY=VALUE` config overrides, and the sharding
source plus counts. Hugging Face plans are the default. Name vLLM or SGLang
when the drawing should follow that engine, and give `--tp`, `--pp`, and
`--ep` for the axes you care about.

### Headless run

Any agent CLI that can discover the installed skill can run the export without
prompts. Pass the model and the output path in the prompt:

```bash
agent --model claude-opus-4-8-thinking-medium --print --force --trust \
    "Follow the model visualizer installed with the TraceLens pip package (look under TraceLens/Agent/Visualizer/skills/model-visualizer/ in the package installation directory) and export <model> to <path>.html"
```

## Read the results

The file you open is `<path>.html`. It embeds the graph payload, the TraceLens
viewer, and the layout worker. The `<model-explorer-visualizer>` element and
its font textures load from unpkg the first time you open the file.

| Request | Artifact |
| --- | --- |
| Default export | one `.html` file |
| `--tp`, `--pp`, or `--ep` with more than one GPU group | `<stem>_subgraph_<i>.html` |
| `-o model.json` | Model Explorer payload; render it later with `--from-payload` |
| `--operators-json` | flat operator list with inferred shapes |
| `--serve` | local viewer at `http://127.0.0.1:8765/` (the process stays running) |

Shapes use the symbols `B` (batch), `S` (sequence), `H` (hidden), and the
attention, expert, and vocabulary symbols documented in the Visualizer README.
The default drawing is a cold rectangular prefill in inference mode, with
attention taken from the checkpoint and `--device hip` selecting HIP branches
in the modeling source. `--device` does not choose the machine that runs the
export.

## What the agent runs

The agent runs a single `visualize_model_in_explorer` command after confirming
the Visualizer extra imports. It checks that an HTML file contains
`model-explorer-visualizer` or that a JSON payload contains `graphCollections`,
then reports the absolute path.
