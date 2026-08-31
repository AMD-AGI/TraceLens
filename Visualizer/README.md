# TraceLens Visualizer

Analyze Hugging Face model configs and modeling source code, then export interactive architecture graphs for [Model Explorer](https://github.com/google-ai-edge/model-explorer). The visualizer runs entirely on CPU — it reads `config.json` and parses `modeling_*.py` via AST inspection; no weights or GPU are required.

## Features

- **Model Explorer export** — interactive layered graph with fact sheet, kernel styling, and collapsible pipelines
- **Model graph export** — serializable computation graph IR (nodes, edges, inline frames, subgraphs)
- **Flexible sources** — Hugging Face Hub checkpoints, local checkpoint directories, or GitHub modeling repos

## Installation

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

## CLI reference

### Model Explorer (`visualize_model_in_explorer.py`)

| Option | Description |
|--------|-------------|
| `SOURCE` / `--checkpoint`, `-c` | Hugging Face model id or local checkpoint directory |
| `--github`, `-g` | GitHub repo URL or `github:owner/repo@ref:path` for modeling source |
| `-o`, `--output` | Output path: `.html` (standalone viewer), `.json` (Model Explorer payload), or default model name when used alone |
| `--serve` | Start local HTTP server with embedded payload |
| `--open` | Open the viewer URL in a browser (with `--serve`) |
| `--host`, `--port` | Bind address for `--serve` (default: `127.0.0.1:8765`) |
| `--config-only` | Skip AST inspection; use config heuristics only |
| `--config-path` | Explicit `config.json` path inside a checkpoint |
| `--code-path` | Explicit path to `modeling_*.py` when auto-discovery fails |
| `--basic-op-add REGEX` | Treat matching block names as leaf/basic ops (repeatable) |
| `--basic-op-remove REGEX` | Remove a default basic-op pattern (repeatable) |

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

Composite GPU kernels (e.g. fused attention, MoE dispatch) remain opaque leaf nodes unless expanded by custom kernel pipeline parsing.

## Project layout

```
Visualizer/
├── visualize_model_in_explorer.py   # CLI entry point
├── model_explorer_export/           # Model Explorer payload, viewer, and server
├── visualizer/
│   ├── extract.py                   # ArchitectureSpec and config/code merge
│   ├── ast_analyze.py               # Modeling file AST analysis
│   ├── block_tree.py                # Recursive block trees and expansion
│   ├── computation_graph.py         # Graph construction from block trees
│   ├── model_graph.py               # Serializable model graph IR
│   └── loader.py                    # High-level model spec loading
├── tests/                           # Pytest suite
└── requirements.txt
```

## Tests

```bash
pytest tests/
```

## Python API

```python
from visualizer.loader import load_model_spec
from model_explorer_export.build import build_model_explorer_payload

spec = load_model_spec("moonshotai/Kimi-K3", detailed=True)
payload = build_model_explorer_payload(spec)
```

## Requirements

- Python 3.10+
- huggingface_hub ≥ 0.20

Network access is needed when loading models from the Hugging Face Hub unless the checkpoint and modeling files are cached locally.
