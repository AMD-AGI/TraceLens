---
name: check-dead-nodes
description: Run after editing graph-export, block-tree, computation-graph, or shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph and fails if any node has no consumer (a "dead" node), which signals a wiring regression.
---

# Check for dead nodes in the merged model graph

A **dead node** is a graph node whose id is never referenced as the
`sourceNodeId` of any other node's `incomingEdges` — i.e. it produces a value
nothing consumes. Dead nodes almost always mean a wiring bug: a real tensor
(`value_states` from `expand_kv`, `up` from an experts gate, `valid_keys` from
the indexer) was extracted but its consumer edge was never reconstructed. The
fix is to **reconstruct the missing consumer edge**, never to prune the node
(pruning hides a real computation, violating the "no opaque/collapsed
computation" invariant).

Legitimate sinks are exempt: synthetic `@output` boundaries/mirrors, synthetic
`@input` boundaries, `@loop_carried` tiles, and the top-level `@output`.

## When to run

Run this immediately after any edit to:
- `TraceLens/ModelUtils/ast_analyze.py`, `block_tree.py`, `computation_graph.py`,
  `shape_inference.py`
- `TraceLens/Visualizer/model_explorer_export/merge.py` (and siblings)

## How to run

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python \
  -m pytest tests/Visualizer/test_glm53_linear_attn_graph.py::test_glm53_graph_has_no_dead_nodes -q
```

Or, for an ad-hoc scan that prints each offender:

```python
import os
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
nodes = graph["nodes"]
consumed = {e["sourceNodeId"] for n in nodes for e in n.get("incomingEdges", [])}

def exempt(n):
    if M._is_synthetic_output(n) or M._is_synthetic_input(n):
        return True
    if M._node_attr(n, "synthetic") == "@loop_carried":
        return True
    return n.get("id") == "@output"

dead = [n for n in nodes if n["id"] not in consumed and not exempt(n)]
print("DEAD COUNT:", len(dead))
for n in dead:
    print("  ", n["id"], "| label=", n.get("label"))
```

## Interpreting results

- **`DEAD COUNT: 0`** — good, ship it.
- **Any dead node** — a wiring regression. Trace the node's producer back through
  the extractor (`ast_analyze.py` `var_producer` / `var_output_ordinal` /
  `step_predecessor_*`) and the graph builder (`computation_graph.py`
  pipeline/kernel port wiring, `merge.py` `_inject_group_outputs`) and
  reconstruct the missing consumer edge. Reuse the exemption predicates from
  `merge.py` (`_is_synthetic_output`, `_is_synthetic_input`, `@loop_carried`,
  `@output`) — do not invent new exemptions to mask a real dead node.
