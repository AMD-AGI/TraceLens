<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Agent: Model Visualizer

The model visualizer agent exports a Hugging Face checkpoint, a local checkpoint directory, or a whitelisted modeling repo to an interactive [Model Explorer](https://github.com/google-ai-edge/model-explorer) graph. A skill drives the `visualize_model_in_explorer` CLI: install check, one export command, then a check that the HTML or JSON file is real. The export is static analysis of `modeling_*.py` plus a meta-device shape pass. It runs on CPU and does not load checkpoint weights.

The tool itself is documented in [TraceLens/Visualizer/README.md](../../Visualizer/README.md).

---

## Prerequisites

### 1. Install TraceLens with the Visualizer extra

Installing the package ships this skill under `TraceLens/Agent/Visualizer/skills/model-visualizer/`. The exporter is the `[Visualizer]` extra.

**Local checkout:**

```bash
pip install -e ".[Visualizer]"
```

**From GitHub:**

```bash
pip install "TraceLens[Visualizer] @ git+https://github.com/AMD-AGI/TraceLens.git"
```

Python 3.10 or newer. No GPU, ROCm, or CUDA wheel is required. A gated checkpoint needs `HF_TOKEN`.

### 2. Model source

One of:

- a Hugging Face model id
- a local directory that contains `config.json`
- a GitHub modeling repo that is already allowlisted, or passed with `--allow-repo`

Network is required the first time a Hub model or an unpkg viewer asset is fetched. Cached checkpoints export offline.

---

## Quick Start - How to Use

> **Note**: The skill is portable across agentic runners that discover skill files (Cursor, Claude Code, Codex). The instructions below use the Cursor IDE and the `agent` CLI.

> **Skill path in the package:** `TraceLens/Agent/Visualizer/skills/model-visualizer/` (`SKILL.md`, `reference.md`, `skill-card.md`). Cursor project discovery reads `.cursor/skills/`; symlink or copy this folder there for automatic pickup.

### To run via chat

1. **Invoke:**

   ```text
   Follow the model visualizer installed with TraceLens and export <model> to <path>.html
   ```

2. **Provide if prompted:**
   - Hugging Face id or local checkpoint path
   - HTML path, or a request to serve the graph locally
   - Runtime regime when it is not a cold rectangular prefill (packed layout, decode, warm cache, training)
   - Attention implementation when it should not follow the checkpoint (`eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, or another `_attn_implementation` string)
   - Config overrides as `KEY=VALUE` (repeatable; dotted keys reach nested configs)
   - Sharding source and counts: Hugging Face plans by default, or `--engine vllm` / `--engine sglang`, with `--tp`, `--pp`, and `--ep`

3. **Results:**
   - **Primary output:** a self-contained `.html` file. Open it in a browser. The visualizer element loads from unpkg on first open.
   - **Several GPU groups:** `<stem>_subgraph_<i>.html` next to that file.
   - **Optional:** a Model Explorer JSON payload (`-o model.json`) and a flat operator JSON (`--operators-json`).

### To run headless

Install the Cursor CLI if you want a non-interactive run:

```bash
curl https://cursor.com/install -fsS | bash
```

Pass the model and the output path in the prompt so the skill does not ask questions:

```bash
agent --model claude-opus-4-8-thinking-medium --print --force --trust \
    "Follow the model visualizer installed with the TraceLens pip package (look under TraceLens/Agent/Visualizer/skills/model-visualizer/ in the package installation directory) and export <model> to <path>.html"
```

Interactive chat does not need the CLI.

---

## Output files

```text
<model>.html                         # standalone viewer (default)
<model>.json                         # Model Explorer payload, when -o ends in .json
<model>_operators.json               # flat operator export, when --operators-json is set
<model>_subgraph_<i>.html            # one file per GPU group, when tp/pp/ep splits the model
```

A live viewer (`--serve`, default port 8765) prints `Open viewer: http://127.0.0.1:8765/` and holds the process. Ask for that only when you want a server instead of a file.

---

## Workflow

```text
0. Resolve the model source, output path, regime, attention, config overrides, and sharding source plus counts
1. Install the Visualizer extra if the exporter does not import
2. Run visualize_model_in_explorer once
3. Confirm the HTML contains model-explorer-visualizer, or the JSON has graphCollections
4. Report the absolute path (or the viewer URL)
```

The procedure and footguns are in [reference.md](skills/model-visualizer/reference.md).

| Choice | What the skill does |
| --- | --- |
| Save a file | `visualize_model_in_explorer <model> -o <file>.html` |
| JSON, then HTML | `-o model.json`, then `--from-payload model.json -o model.html` |
| Hugging Face sharding | `--tp`, `--pp`, `--ep` and no `--engine` |
| vLLM or SGLang sharding | `--engine vllm` or `--engine sglang` with `--tp`, `--pp`, `--ep` |
| Attention implementation | `--attention` (`eager`, `sdpa`, `flash_attention_2`, …). Omit it to keep the checkpoint's setting |
| Config overrides | repeat `--config-override KEY=VALUE` |
| Decode or packed tokens | `--phase decode`, `--sequence-layout packed` |
| Skip shapes or the meta pass | `--no-shapes`, `--no-meta-shapes` |

`--device` selects which accelerator branches the drawing follows (`hip` by default). It does not select the machine that runs the export.

---

## Tests

Skill routing and behavior cases live in [skills/model-visualizer/evals/](skills/model-visualizer/evals/README.md). Structural checks run in CI through `.github/workflows/skill-evals.yml`. The exporter's own unit tests stay in `tests/Visualizer/` and `tests/ModelUtils/`:

```bash
pip install -e ".[Visualizer,dev]"
python3 -m pytest tests/Visualizer/ tests/ModelUtils/
```
