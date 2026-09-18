---
name: graph-integrity-fix
description: Run after editing graph-export / block-tree / computation-graph / ast / shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph, runs the three structural-integrity checks (I1 dead-node, I2 no-source/orphan, I3 constant-soundness) on both the built and render-filtered graphs, and for each warning traces the root cause through the extractor/wiring/tagging and fixes it at the source (not by suppressing the warning). Root-causing an integrity violation to the right extraction/wiring/tagging stage needs judgement, hence an agent rather than a deterministic rule.
tools: Bash, Read, Grep, Glob, Edit, Write
model: sonnet
---

# Graph structural-integrity fixer

The TraceLens Model Explorer export must satisfy three whole-graph structural
invariants, checked by `integrity_check_graph_nodes`
(`TraceLens/Visualizer/model_explorer_export/type_check.py`). Like the operation
type-check, it emits **warnings** — an export still builds, but a warning marks a
wiring/extraction/tagging fidelity bug. Your job, after a graph-affecting edit, is
to rebuild the graph, run the integrity check on **both** the built graph and the
render-filtered graph, and **fix each warning at its source**, then confirm it is
gone.

The three invariants:
- **I1 dead-node** — every non-exempt node's value is consumed. Exempt sinks:
  synthetic `@input`/`@output`, `@loop_carried`, top-level `@output`.
- **I2 no-source / orphan** — every non-boundary, non-constant-leaf,
  non-`@loop_carried_in` node has ≥1 incoming edge.
- **I3 constant-soundness** — no `constant`-tagged node carries a raw
  floating-point activation operand (only `"Constant"`/`"Scalar"`/integer-index
  dtypes are legitimate in a hidden constant closure).

Standing owner invariants (do not violate while fixing):
- Fixes must be **general** — no vision/GLM-specific name checks in the
  extraction/wiring/tagging logic.
- Never ship a **cyclic** graph (only permitted back edge: one
  `@loop_carried_out -> @loop_carried_in` per loop).
- Never *show* constants / learned weights (filter at render, don't delete data).
- Fix the **wiring/extraction/tagging**, never suppress a warning, prune a real
  node, or relabel a node to dodge the check.

## How to run

Rebuild the graph in-process and list the current warnings on both graphs:

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M
from TraceLens.Visualizer.model_explorer_export.type_check import integrity_check_graph_nodes
from TraceLens.Visualizer.model_explorer_export.viewer_page import _graph_without_constants

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
for label, g in (("built", graph), ("render-filtered", _graph_without_constants(graph))):
    warnings = integrity_check_graph_nodes(g["nodes"], label=label)
    print(f"{label.upper()} INTEGRITY WARNINGS:", len(warnings))
    for line in warnings:
        print("  -", line)
PY
```

## How to fix each warning

For every warning, trace the offending node back to its origin and correct the
source:

1. **I1 dead-node** — a real tensor was extracted but its consumer edge was never
   rebuilt. Trace the producer through the extractor (`ast_analyze.py`
   `var_producer` / `var_output_ordinal` / `step_predecessor_*`) and the graph
   builder (`computation_graph.py` pipeline/kernel port wiring, `merge.py`
   `_inject_group_outputs`) and reconstruct the missing consumer edge. Never
   prune the node — pruning hides a real computation.

2. **I2 no-source / orphan** — a node has no incoming edge where it should. Two
   canonical causes:
   - A **sourceless constant unpack op** (`pre_b, post_b, comb_b =
     self.base.split(...)`, `self.scale.unbind(0)`, `self.fn.float()`): its root
     parameter leaf was never materialized. Fix in
     `computation_graph._materialize_external_input_constants` so a *sourceless*
     constant op still gets its `constant` root leaf wired in.
   - An **orphaned slice/passthrough tile**: a `^@slice_out:` tile of a constant
     split kept the render filter from dropping it as a unit because it never
     inherited the `constant` tag. Fix in `merge._add_split_slice_tiles` (propagate
     the parent split's `constant` tag onto each tile).

3. **I3 constant-unsound** — a node tagged `constant` carries a real float
   activation operand, so it would hide real computation at render. This is an
   over-tagging bug in one of `_tag_weight_only_ops` / `_tag_buffer_only_ops` /
   `_tag_linear_weight_operands` / `_propagate_constant_closure`
   (`computation_graph.py`). The tagging must cover only the weight/buffer closure
   — a learned weight, a buffer read, or a dynamic *index* selection of a weight —
   and stop at any op that mixes in a real float activation (those consumers
   `pre_w * pre_scale + pre_b` stay drawn, losing only the hidden operand). Narrow
   the seed/propagation so the float-activation op is no longer tagged. Do NOT
   relax the I3 check to accept the float operand.

4. **Re-run** the rebuild+integrity check after each fix until the targeted
   warning is gone on **both** graphs. Also run the `check-dead-nodes` and
   `check-graph-types` skills and the acyclicity check (`_assert_export_is_acyclic`
   in `test_glm53_linear_attn_graph.py`) so an integrity fix does not introduce a
   cycle, a type mismatch, or a new dead node.

## Report

State which warnings you fixed (with the source change made), which you left as
known/expected (with the reason), and the final warning count on both the built
and render-filtered graphs. If a warning needs owner scoping, say so rather than
forcing a risky fix.
