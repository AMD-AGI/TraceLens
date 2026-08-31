# TraceLens Visualizer

Analyze Hugging Face model configs and modeling source code, then export interactive architecture graphs for [Model Explorer](https://github.com/google-ai-edge/model-explorer). The visualizer runs entirely on CPU — it reads `config.json` and parses `modeling_*.py` via AST inspection; no weights or GPU are required.

## Features

- **Model Explorer export** — interactive layered graph with fact sheet, kernel styling, and collapsible pipelines
- **Optional shape inference** — symbolic tensor shapes on expanded detail nodes (`--shapes`)
- **Operator JSON export** — flat per-op shape lists for tooling (`--operators-json`)
- **Model graph IR** — serializable computation graph (nodes, edges, inline frames, subgraphs)
- **Flexible sources** — Hugging Face Hub checkpoints, local checkpoint directories, or GitHub modeling repos

## Installation

From the TraceLens repository root:

```bash
pip install -e ".[Visualizer]"
```

For local development and tests, also install the dev extra:

```bash
pip install -e ".[Visualizer,dev]"
```

Or install only from the `Visualizer/` directory with `PYTHONPATH=.` (see below).

Legacy standalone setup:

```bash
cd ~/tracelens/Visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Quick start

Serve an interactive Model Explorer viewer for a Hugging Face model:

```bash
python visualize_model_in_explorer.py moonshotai/Kimi-K3 --serve --open
```

Export a standalone HTML file:

```bash
python visualize_model_in_explorer.py moonshotai/Kimi-K3 -o
# writes moonshotai_Kimi-K3.html
```

Export raw Model Explorer JSON:

```bash
python visualize_model_in_explorer.py moonshotai/Kimi-K3 -o kimi.json
```

Enable symbolic shape annotations on graph nodes:

```bash
python visualize_model_in_explorer.py moonshotai/Kimi-K3 --serve --open --shapes
```

Write a flat operator export with inferred shapes (also embeds `operatorExport` in the payload when used with `-o`):

```bash
python visualize_model_in_explorer.py moonshotai/Kimi-K3 --operators-json kimi_operators.json
```

## CLI reference

### Model Explorer (`visualize_model_in_explorer.py`)

| Option | Description |
|--------|-------------|
| `SOURCE` / `--checkpoint`, `-c` | Hugging Face model id or local checkpoint directory |
| `--github`, `-g` | GitHub repo URL or `github:owner/repo@ref:path` for modeling source |
| `-o`, `--output` | Output path: `.html` (standalone viewer), `.json` (Model Explorer payload), or default model name when used alone |
| `--title` | Architecture display name override |
| `--serve` | Start local HTTP server with embedded payload |
| `--open` | Open the viewer URL in a browser (with `--serve`) |
| `--port` | Port for `--serve` (default: `8765`) |
| `--config-path` | Explicit `config.json` path inside a checkpoint |
| `--code-path` | Explicit path to `modeling_*.py` when auto-discovery fails |
| `--dump-ast` | Write parsed modeling-file AST dump to a path |
| `--basic-op-add REGEX` | Treat matching block names as leaf/basic ops (repeatable) |
| `--basic-op-remove REGEX` | Remove a default basic-op pattern (repeatable) |
| `--all-tensor-ops` | Include tensor housekeeping ops in detailed graphs |
| `--shapes` | Add `output_shape` / `output_dtype` on expanded detail nodes |
| `--operators-json PATH` | Also write flat operator export JSON with inferred shapes |

Shape inference is **off by default**. When enabled, shapes are attached to nodes inside expanded detail sections (MoE internals, kernel pipelines, etc.), not to single-tile overview spine nodes.

## Source resolution

The visualizer resolves inputs in this order:

1. **Hugging Face checkpoint** — downloads or reads `config.json`, then locates `modeling_*.py` in the same repo
2. **GitHub** — fetches modeling source from a repo URL or `github:owner/repo@ref:path` shorthand
3. **Explicit paths** — `--config-path` and `--code-path` override auto-discovery

Checkpoints for transformers-native architectures (Qwen3, MiniMax-M3, …) ship no
modeling code, so `modeling_<model_type>.py` is read from `huggingface/transformers`
on GitHub — for a multimodal wrapper, the nested text backbone's `model_type` is tried
too. The file is cached under `~/.cache/tracelens`, and if the fetch fails the export
falls back to config heuristics.

## How it works

1. **Config parsing** — reads `hidden_size`, layer counts, attention type, MoE settings, etc. from `config.json`
2. **AST analysis** — parses `modeling_*.py` to discover module structure, `forward()` call order, norms, gates, and custom kernels
3. **Block trees** — builds recursive block diagrams from the parsed structure
4. **Computation graph** — converts block trees to a directed graph for Model Explorer export
5. **Shape inference** (optional) — propagates symbolic dimensions (`B`, `T`, `H`, …) through the model graph and annotates exported nodes

Composite GPU kernels (e.g. fused attention, MoE dispatch) remain opaque leaf nodes unless expanded by custom kernel pipeline parsing.

## Project layout

```
Visualizer/
├── visualize_model_in_explorer.py   # CLI entry point
├── model_explorer_export/           # Model Explorer payload, viewer, and server
│   ├── build.py                     # Payload assembly
│   ├── merge.py                     # Single merged graph export
│   ├── shapes.py                    # Shape annotation for Model Explorer nodes
│   └── viewer/                      # Bundled viewer shell (app.js, index.html)
├── visualizer/
│   ├── extract.py                   # ArchitectureSpec and config/code merge
│   ├── ast_analyze.py               # Modeling file AST analysis
│   ├── block_tree.py                # Recursive block trees and expansion
│   ├── computation_graph.py         # Graph construction from block trees
│   ├── model_graph.py               # Serializable model graph IR
│   ├── shape_inference.py           # Symbolic shape and operator export
│   └── loader.py                    # High-level model spec loading
├── tests/                           # Pytest suite
└── requirements.txt
```

## Tests

```bash
cd Visualizer
PYTHONPATH=. pytest tests/
```

## Python API

```python
from visualizer.loader import load_model_spec
from visualizer.shape_inference import build_operator_export
from model_explorer_export.build import build_model_explorer_payload

spec = load_model_spec("moonshotai/Kimi-K3", detailed=True)

# Model Explorer payload (shapes off by default)
payload = build_model_explorer_payload(spec)
payload_with_shapes = build_model_explorer_payload(spec, include_shapes=True)

# Flat operator list with inferred shapes
operators = build_operator_export(spec)
```

## Requirements

- Python 3.10+
- huggingface_hub ≥ 0.20

Network access is needed when loading models from the Hugging Face Hub unless the checkpoint and modeling files are cached locally.
