# TraceLens Visualizer

Generate Raschka-style LLM architecture diagrams from Hugging Face model configs and modeling source code. The visualizer runs entirely on CPU — it reads `config.json` and parses `modeling_*.py` via AST inspection; no weights or GPU are required.

## Features

- **Overview diagrams** — high-level model stack (embeddings, decoder blocks, LM head) from config and heuristics
- **Detailed diagrams** (`--detailed`) — recursive internal block diagrams parsed from `forward()` in modeling code
- **Operator export** — flat operator list with symbolic tensor shapes/dtypes as JSON
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

Generate a detailed architecture diagram for a Hugging Face model:

```bash
python generate_diagram.py moonshotai/Kimi-K3 --detailed -o Kimi-K3_architecture_detailed.svg
```

Or run the module directly:

```bash
python -m visualizer moonshotai/Kimi-K3 --detailed
```

Export an operator graph with inferred shapes:

```bash
python export_operator_graph.py moonshotai/Kimi-K3 -o kimi_operators.json
```

## CLI reference

### Diagram generator (`generate_diagram.py` / `python -m visualizer`)

| Option | Description |
|--------|-------------|
| `SOURCE` / `--checkpoint`, `-c` | Hugging Face model id or local checkpoint directory |
| `--github`, `-g` | GitHub repo URL or `github:owner/repo@ref:path` for modeling source |
| `-o`, `--output` | Output path (`.svg`, `.png`, `.pdf`). Default: `<model>_architecture.svg` or `_detailed.svg` |
| `--title` | Diagram title override |
| `--detailed` | Include recursive internal block diagrams (requires modeling source) |
| `--config-only` | Skip AST inspection; use config heuristics only |
| `--config-path` | Explicit `config.json` path inside a checkpoint (e.g. `text_encoder/config.json`) |
| `--code-path` | Explicit path to `modeling_*.py` when auto-discovery fails |
| `--dump-ast` | Write parsed Python AST dump for the modeling file |
| `--json` | Also write architecture metadata to JSON |
| `--graph-json` | Write model graph IR to JSON (requires `--detailed`) |
| `--facts` | Print fact sheet to stdout |
| `--basic-op-add REGEX` | Treat matching block names as leaf/basic ops (repeatable) |
| `--basic-op-remove REGEX` | Remove a default basic-op pattern (repeatable) |
| `--no-inline-linear-frames` | Disable dotted frames around inlined straight-line sub-blocks |
| `--dpi` | DPI for raster output (default: 150) |

### Operator export (`export_operator_graph.py`)

Same checkpoint/GitHub/code options as the diagram generator. Writes a JSON file with sections of operators, each including name, computation, operation kind, inputs, and inferred output shape/dtype.

| Option | Description |
|--------|-------------|
| `-o`, `--output` | Output JSON path (default: `<model>_operators.json`) |
| `--no-model-output` | Omit the terminal output/logits operator |
| Other flags | Same source and basic-op flags as the diagram CLI |

## Source resolution

The visualizer resolves inputs in this order:

1. **Hugging Face checkpoint** — downloads or reads `config.json`, then locates `modeling_*.py` in the same repo
2. **GitHub** — fetches modeling source from a repo URL or `github:owner/repo@ref:path` shorthand
3. **Explicit paths** — `--config-path` and `--code-path` override auto-discovery

Checkpoints for transformers-native architectures (Qwen3, MiniMax-M3, …) ship no
modeling code, so `modeling_<model_type>.py` is read from `huggingface/transformers`
on GitHub — for a multimodal wrapper, the nested text backbone's `model_type` is tried
too. The file is cached under `~/.cache/tracelens`, and if the fetch fails the diagram
falls back to config heuristics. Diagrams name the source in the fact sheet.

Example with GitHub fallback when the HF repo lacks modeling code:

```bash
python generate_diagram.py org/model-name \
  --github github:org/model-repo@main:src/modeling.py \
  --detailed
```

## How it works

1. **Config parsing** — reads `hidden_size`, layer counts, attention type, MoE settings, etc. from `config.json`
2. **AST analysis** — parses `modeling_*.py` to discover module structure, `forward()` call order, norms, gates, and custom kernels
3. **Block trees** — builds recursive block diagrams from the parsed structure
4. **Diagram preparation** — expands straight-line composites in-place and substitutes single-op subgraph wrappers before layout
5. **Computation graph** — converts block trees to a directed graph for layout and rendering
6. **Shape inference** — assigns symbolic dimensions (`B`, `T`, `H`, …) to tensors for operator export

Composite GPU kernels (e.g. fused attention, MoE dispatch) remain opaque leaf nodes unless expanded by custom kernel pipeline parsing.

## Project layout

```
Visualizer/
├── generate_diagram.py       # CLI entry point for diagrams
├── export_operator_graph.py  # CLI entry point for operator JSON
├── visualizer/
│   ├── cli.py                # Argument parsing and main diagram flow
│   ├── extract.py            # ArchitectureSpec and config/code merge
│   ├── ast_analyze.py        # Modeling file AST analysis
│   ├── block_tree.py         # Recursive block trees and expansion
│   ├── computation_graph.py  # Graph construction and layout
│   ├── render.py             # Matplotlib SVG/PNG/PDF rendering
│   ├── shape_inference.py    # Symbolic shape inference and operator export
│   ├── model_graph.py        # Serializable model graph IR
│   └── loader.py             # High-level model spec loading
├── tests/                    # Pytest suite
└── requirements.txt
```

## Tests

```bash
pytest tests/
```

Run a focused subset:

```bash
pytest tests/test_shape_inference.py tests/test_detailed.py -q
```

## Examples

**Overview diagram (config heuristics only):**

```bash
python generate_diagram.py moonshotai/Kimi-K3 --config-only
```

**Detailed diagram with metadata JSON:**

```bash
python generate_diagram.py moonshotai/Kimi-K3 --detailed \
  --json kimi_meta.json \
  -o Kimi-K3_architecture_detailed.svg
```

**Treat Linear and RMSNorm as basic leaf ops in detailed view:**

```bash
python generate_diagram.py moonshotai/Kimi-K3 --detailed \
  --basic-op-add '(?i)^Linear$' \
  --basic-op-add '(?i)^RMSNorm$'
```

**Local checkpoint directory:**

```bash
python generate_diagram.py /path/to/checkpoint --detailed -o local_model.svg
```

## Python API

```python
from visualizer import load_architecture, render_diagram, build_operator_export
from visualizer.basic_ops import BasicOpFilter

spec = load_architecture(
    "moonshotai/Kimi-K3",
    detailed=True,
    basic_ops=BasicOpFilter.for_detailed(),
)
render_diagram(spec, "Kimi-K3_architecture_detailed.svg", detailed=True)

payload = build_operator_export(spec)
```

## Requirements

- Python 3.10+
- matplotlib ≥ 3.7
- huggingface_hub ≥ 0.20
- graph-layout ≥ 0.4.1

Network access is needed when loading models from the Hugging Face Hub unless the checkpoint and modeling files are cached locally.
