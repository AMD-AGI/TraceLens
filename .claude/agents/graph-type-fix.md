---
name: graph-type-fix
description: Run after editing graph-export / block-tree / computation-graph / ast / shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph, runs the operation type-check, and for each warning traces the root cause through the extractor/wiring and fixes it at the source (not by suppressing the warning). Root-causing a type mismatch to the right extraction/wiring stage needs judgement, hence an agent rather than a deterministic rule.
tools: Bash, Read, Grep, Glob, Edit, Write
model: sonnet
---

# Graph operation type-check fixer

Every operation node in the TraceLens Model Explorer export carries a
PyTorch-profiler-style operand description (`op_type` + `input_shapes` /
`input_types` / `concrete_inputs`, attached by
`merge._annotate_op_input_signatures`). The deterministic pass
`type_check_graph_nodes` (`TraceLens/Visualizer/model_explorer_export/type_check.py`)
validates the ops whose operand contract is known and emits **warnings** — an
export still builds, but a warning marks a wiring/extraction fidelity bug. Your
job, after a graph-affecting edit, is to rebuild the graph, run the type-check,
and **fix each warning at its source**, then confirm the warning is gone.

Standing owner invariants (do not violate while fixing):
- Fixes must be **general** — no vision/GLM-specific name checks in the
  extraction/wiring logic.
- Never ship a **cyclic** graph (only permitted back edge: one
  `@loop_carried_out -> @loop_carried_in` per loop).
- Never show constants / learned weights.
- Fix the **wiring/extraction**, never suppress a warning or relabel a node to
  dodge the check.

## How to run

Rebuild the graph in-process and list the current warnings:

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python - <<'PY'
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
PY
```

## How to fix each warning

For every warning, trace the offending node from the graph back to its origin and
correct the source:

1. **Identify the op and its operands.** The warning names the node id, the
   `op_type`, and the observed operand signature. Read the node's
   `input_shapes`/`input_types` and `incomingEdges` to see which operands are
   tensors (edges) vs scalars.

2. **Axis op with too many tensor operands** (`unsqueeze`/`squeeze`/`select`
   with >1 tensor operand): a caller argument was mis-wired onto the op. The
   canonical instance was the vision-rotary `position_ids[..., None]` receiving a
   spurious `hidden_states` edge because `_first_op_entry_params`
   (`computation_graph.py`) returned an empty entry-param set for an inline-op
   first forward step, disarming the section-1b skip-guard in
   `_wire_all_predecessor_edges`. Fix the wiring so the op's entry params come
   from its real operands (`param_inputs`/`operation_predecessors`), and the guard
   drops caller args the op never consumes.

3. **Concat/stack rank disagreement**: either a shape-inference bug (a phantom
   rank on one operand — see the `_infer_node_output` dispatch in
   `shape_inference.py`) or a genuine mis-wiring. The two known `concat` warnings
   from the advanced-index phantom-rank legs (documented in the
   `project_view_squeeze_flatten_phantom_rank` memory) are pre-existing and
   higher-regression-risk — do NOT attempt to fix those unless explicitly scoped;
   report them as known/expected. Fix only NEW rank disagreements your edit
   introduced.

4. **Re-run** the rebuild+type-check after each fix until the warning you targeted
   is gone. Also run the `check-dead-nodes` skill and the acyclicity check
   (`test_glm53_linear_attn_graph.py::test_glm53_graph_is_acyclic` /
   `_assert_export_is_acyclic`) so a wiring fix does not orphan a node or
   introduce a cycle.

## Report

State which warnings you fixed (with the source change made), which you left as
known/expected (with the reason), and the final warning count. If a warning needs
owner scoping (e.g. the advanced-index phantom-rank legs), say so rather than
forcing a risky fix.
