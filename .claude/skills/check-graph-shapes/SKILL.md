---
name: check-graph-shapes
description: Run after editing graph-export, block-tree, computation-graph, ast, or shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph and flags every node whose rendered output shape carries an unresolved dim (a `?` placeholder, an unfolded reshape `-1`, or a dropped/empty axis), reporting a shape-inference fidelity gap to fix upstream.
---

# Check for unresolved output shapes in the merged model graph

Every operation node in the export carries a rendered `output_shape` attr (the
string form the export writes, e.g. `[B, S, 4096] bfloat16`). A fully resolved
shape names every axis either as a concrete int (`4096`) or as a named symbolic
dim (`S`, `Pv/4`, a collapsed product `B*S*8192`, `index_head_dim`, `S + S`). A
symbolic dim is resolved — it just is not numeric — and is **not** a problem.

An axis that renders as one of these is *unresolved* and is flagged:

- `?` — shape inference could not determine the axis at all;
- `-1` — an unfolded `reshape`/`view` placeholder that should have been resolved
  against the operand's real dims (the `_resolve_view_shape` class of work);
- an empty token (`[B, , S]`) — a dropped axis.

`type_check_graph_nodes` collects these as `_unresolved_shape_warnings` and emits
them as **warnings** (never errors): the export still builds, but a warning flags
a shape-inference fidelity gap. The check keys purely on the dim token, so it uses
no op-name or class-name list and never fires on a legitimate symbolic dim.
Constant nodes (learned weights/buffers, filtered from the drawn graph) and rank-0
scalars (empty bracket `[]` → no dim tokens) are not flagged.

## When to run

Run this immediately after any edit to:
- `TraceLens/ModelUtils/ast_analyze.py`, `block_tree.py`, `computation_graph.py`,
  `shape_inference.py`
- `TraceLens/Visualizer/model_explorer_export/merge.py` (and siblings)

## How to run

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python \
  -m pytest tests/Visualizer/test_graph_type_check.py -q
```

Or, for an ad-hoc scan of a real model that prints each unresolved-shape warning:

```python
import os
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M
from TraceLens.Visualizer.model_explorer_export.type_check import (
    _unresolved_shape_warnings,
)

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
warnings = _unresolved_shape_warnings(graph["nodes"])
print("UNRESOLVED-SHAPE WARNINGS:", len(warnings))
for line in warnings:
    print("  -", line)
```

## Interpreting results

- **No unresolved-shape warnings** — every rendered axis is resolved (concrete or
  named symbolic). This is the expected state for all four tracked models
  (GLM-5.3-Flash / Kimi-K3 / MiniMax-M3 / DeepSeek-V4-Flash).
- **A `?` warning** — shape inference produced nothing for that axis. Trace the
  node back to its `_infer_node_output` branch in `shape_inference.py` and add /
  fix the rule so the axis resolves (often a reduction or a module-shape lookup).
- **A `-1` warning** — an unfolded reshape placeholder. The `-1` should be
  resolved against the operand's real dims in `_resolve_view_shape` /
  `_resolve_meta_op` (`shape_inference.py`); the star-ref + lone-`-1` head-collapse
  branch (`_resolve_view_shape`) is the reference pattern.
- **An empty-axis warning** — a dropped axis; find where the shape list lost an
  entry (a slice/select whose dim fell out, or a merge that dropped a token).

The fix is always upstream in shape inference — resolve the dim; never render an
unresolved shape and never suppress the warning.
