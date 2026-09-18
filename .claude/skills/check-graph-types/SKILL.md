---
name: check-graph-types
description: Run after editing graph-export, block-tree, computation-graph, ast, or shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph and type-checks every operation node whose operand contract is known, reporting warnings for mis-wired/mis-typed inputs (e.g. an axis op fed a second tensor operand where a scalar dim belongs).
---

# Type-check operation nodes in the merged model graph

Every operation node in the export carries a PyTorch-profiler-style operand
description — `op_type` plus parallel `input_shapes` / `input_types` /
`concrete_inputs` attrs (attached by `merge._annotate_op_input_signatures`). A
scalar argument that is not a graph edge (an `unsqueeze`'s `dim`, a `select`'s
index) appears as a `[]` shape / `"Scalar"` type / concrete-value entry, exactly
like the profiler's `Input Dims` / `Input type` / `Concrete Inputs`.

`type_check_graph_nodes` validates the ops whose operand contract is known and
emits **warnings** (never errors): the export still builds, but a warning flags a
wiring/extraction fidelity bug. The canonical case: an axis op
(`unsqueeze`/`squeeze`/`select`) that takes one tensor + a scalar `dim` but was
wired a *second tensor* operand (a caller argument mis-dumped onto it). The fix is
always upstream — correct the extraction/wiring — never suppress the warning.

Ops with no known contract are skipped (no false positives).

## When to run

Run this immediately after any edit to:
- `TraceLens/ModelUtils/ast_analyze.py`, `block_tree.py`, `computation_graph.py`,
  `shape_inference.py`
- `TraceLens/Visualizer/model_explorer_export/merge.py` (and siblings)

## How to run

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python \
  -m pytest tests/Visualizer/test_glm53_linear_attn_graph.py::test_glm53_graph_type_check_no_axis_op_violations -q
```

Or, for an ad-hoc scan that prints each warning:

```python
import os
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M
from TraceLens.Visualizer.model_explorer_export.type_check import type_check_graph_nodes

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
warnings = type_check_graph_nodes(graph["nodes"])
print("TYPE-CHECK WARNINGS:", len(warnings))
for line in warnings:
    print("  -", line)
```

## Interpreting results

- **No axis-op warnings** — the owner-flagged class (an axis op with a second
  tensor operand) is clean.
- **An axis-op warning** — a wiring regression. Trace the op back through the
  extractor (`ast_analyze.py` operand/`param_inputs` extraction) and the graph
  builder (`computation_graph.py` `_wire_all_predecessor_edges` /
  `_first_op_entry_params`, which fabricated the spurious `hidden_states` edge on
  the vision-rotary unsqueeze) and fix the wiring so the op reads only its real
  operand. Do not suppress the warning or add per-name hacks.
- **Known `concat` rank warnings** — two `concat` rank warnings from the
  documented advanced-index phantom-rank legs (see the
  `project_view_squeeze_flatten_phantom_rank` memory) are expected and tracked
  separately; they are warnings by design, not a regression.

The agent-based counterpart that actually applies fixes is the `graph-type-fix`
agent.
