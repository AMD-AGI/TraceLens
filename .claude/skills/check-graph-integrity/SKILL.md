---
name: check-graph-integrity
description: Run after editing graph-export, block-tree, computation-graph, ast, or shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph and runs three structural-integrity checks (I1 dead-node, I2 no-source/orphan, I3 constant-soundness) on BOTH the built graph and the render-filtered graph, reporting warnings for wiring/tagging fidelity bugs (a sourceless op, an orphaned tile, a real activation hidden as a constant).
---

# Structural-integrity check of the merged model graph

Beyond the per-operation operand type-check (`check-graph-types`), the export must
satisfy three whole-graph structural invariants. `integrity_check_graph_nodes`
(`TraceLens/Visualizer/model_explorer_export/type_check.py`) validates them and
emits **warnings** (never errors): the export still builds, but a warning flags a
wiring/tagging fidelity bug. The fix is always upstream — correct the
extraction/wiring/tagging — never suppress the warning.

The three invariants:

- **I1 dead-node** — every non-exempt node's value is consumed by some other
  node. Exempt sinks: synthetic `@input`/`@output` boundaries, `@loop_carried`
  tiles, and the top-level `@output`. A dead node means a real tensor was
  extracted but its consumer edge was never rebuilt — reconstruct the edge, never
  prune.
- **I2 no-source / orphan** — every node that is not a boundary
  (`@input`/`@output`), not a `constant` leaf (a `constant` tag with no inputs),
  and not a `@loop_carried_in` (seed + back-edge fed) has ≥1 incoming edge. This
  catches a sourceless op (e.g. `pre_b, post_b, comb_b = self.base.split(...)`
  with no materialized root leaf) and an orphaned passthrough/slice tile.
- **I3 constant-soundness** — no `constant`-tagged node carries a raw
  floating-point *activation* operand. A genuinely constant node reads only other
  constants (learned weights / buffers, annotated `"Constant"`), integer *indices*
  (a dynamic weight/expert selection like `self.gate_up_proj[expert_idx]`, wired
  from an int64 routing index), or scalar args — never a raw float tensor. A float
  dtype in `input_types` means a real activation is being hidden at render, i.e.
  the node is mistagged.

Both the built graph and the **render-filtered** graph (constants dropped, via
`viewer_page._graph_without_constants`) are checked — dropping the constant
closure can orphan a survivor that lost its only constant producer, so orphaning
often appears only post-filter.

## When to run

Run this immediately after any edit to:
- `TraceLens/ModelUtils/ast_analyze.py`, `block_tree.py`, `computation_graph.py`,
  `shape_inference.py`
- `TraceLens/Visualizer/model_explorer_export/merge.py` (and siblings)

## How to run

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python \
  -m pytest tests/Visualizer/test_glm53_linear_attn_graph.py::test_glm53_graph_integrity_checks_emit_no_warnings -q
```

Or, for an ad-hoc scan that prints each warning on both graphs:

```python
import os
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M
from TraceLens.Visualizer.model_explorer_export.type_check import integrity_check_graph_nodes
from TraceLens.Visualizer.model_explorer_export.viewer_page import _graph_without_constants

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))

built = integrity_check_graph_nodes(graph["nodes"], label="built")
print("BUILT INTEGRITY WARNINGS:", len(built))
for line in built:
    print("  -", line)

filtered = integrity_check_graph_nodes(_graph_without_constants(graph)["nodes"], label="render-filtered")
print("RENDER-FILTERED INTEGRITY WARNINGS:", len(filtered))
for line in filtered:
    print("  -", line)
```

## Interpreting results

- **0 warnings on both graphs** — good, ship it.
- **I1 dead-node** — a wiring regression: trace the node's producer through the
  extractor (`ast_analyze.py` `var_producer`/`step_predecessor_*`) and the graph
  builder (`computation_graph.py`, `merge.py` `_inject_group_outputs`) and rebuild
  the missing consumer edge. Reuse the exemption predicates — do not invent a new
  exemption to mask a real dead node.
- **I2 no-source** — a sourceless op or orphaned tile: a constant unpack op needs
  its root parameter leaf materialized (`_materialize_external_input_constants`),
  or a slice/passthrough tile needs its `constant` tag propagated
  (`merge._add_split_slice_tiles`) so it drops with its closure instead of being
  orphaned at render.
- **I3 constant-unsound** — an over-tagging bug: a node with a real float operand
  was tagged constant and will hide real computation at render. Fix the tagging
  pass (`_tag_weight_only_ops` / `_tag_buffer_only_ops` /
  `_tag_linear_weight_operands` / `_propagate_constant_closure`) so only the
  weight/buffer closure is tagged, never a float activation.

For root-causing and fixing, use the `graph-integrity-fix` agent.
