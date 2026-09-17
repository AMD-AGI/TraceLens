---
name: opaque-node-audit
description: Run after editing graph-export / block-tree / computation-graph / ast / shape-inference code (TraceLens/ModelUtils/*, TraceLens/Visualizer/model_explorer_export/*). Rebuilds the merged Model Explorer graph and reviews every leaf node whose label is a module / free-function / custom name, reporting any that hide real tensor computation behind a single opaque box. Needs judgement (a sanctioned fused kernel vs an opaque composite), hence an agent rather than a deterministic rule.
tools: Bash, Read, Grep, Glob
model: sonnet
---

# Opaque computation-node audit

A standing owner invariant for the TraceLens Model Explorer export is: **no
opaque / collapsed computation nodes** — every tensor computation must be
visible as its own op. This class of defect has recurred (Logits-as-Linear, the
vision index helpers, `get_pooled_states`), so audit for it after every
graph-affecting edit.

An **opaque computation node** is a single leaf graph node whose label is a
module / method / free-function / custom name (not an atomic torch op) that
*hides* real tensor math inside it — the graph should instead show that math as
individual ops. The fix is always to **expand** the node into its inner ops
(via the free-function / multi-op-method frame machinery), never to relabel it.

## Sanctioned atomic leaves (NOT opaque — do not flag)

- Fused attention kernels: `recurrent_kimi_delta_attention`,
  `Recurrent gated delta`, vision attention kernel, `Causal Conv1D`, and other
  `AttentionOp`/`KernelOp`/`AttentionMerge` nodes.
- Pure atomic torch ops: `Masked scatter`, `TopK`, `Token Embedding`,
  `Scatter`, `Gather`, `Softmax`, `Linear`, `Cast`, `RMSNorm`, elementwise ops,
  reshape/view/split/unbind, etc.
- Synthetic boundaries: `@input`, `@output`, `@loop_carried_*`, kernel ports,
  slice-passthrough (`^@slice_out:`) tiles.

## What IS opaque (flag it)

A leaf whose label is a `snake_case` free-function name, a `CamelCase` module
name, or a method name (e.g. `get pooled states`, `Get vision position ids`,
`expand kv`) AND that is not one of the sanctioned kernels above. Such a leaf
means its callee body was never expanded — real computation is collapsed.

## How to run

Rebuild the graph in-process and list candidate opaque leaves, then judge each:

```bash
HF_HOME=/home/AMD/gabweisz/huggingface \
  /home/AMD/gabweisz/venv_tracelens_visualizer/bin/python - <<'PY'
import os, re
os.environ.setdefault("HF_HOME", "/home/AMD/gabweisz/huggingface")
from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export import merge as M

spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
graph = M.build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
nodes = graph["nodes"]

# A leaf = a node with no descendant nodes sharing its id-prefix namespace.
ids = [n["id"] for n in nodes]
def is_leaf(n):
    p = n["id"] + "/"
    return not any(other != n["id"] and other.startswith(p) for other in ids)

SANCTIONED = re.compile(
    r"recurrent|delta|conv1d|attention|kernel|embedding|topk|scatter|gather|"
    r"softmax|linear|rmsnorm|cast|split|unbind|view|reshape|@|result",
    re.IGNORECASE,
)
for n in nodes:
    label = (n.get("label") or "").strip()
    if not label or not is_leaf(n):
        continue
    # Multi-word snake/camel names that are not plain atomic ops.
    if SANCTIONED.search(label):
        continue
    # Heuristic: a name containing '_' or 3+ CamelCase humps is a callee name.
    if "_" in label or len(re.findall(r"[A-Z][a-z]+", label)) >= 2:
        print("CANDIDATE:", n["id"], "|", label)
PY
```

For each `CANDIDATE`, open the source of the named callee (grep the analyzed
model file) and decide: does it perform tensor math that should be individual
ops? If yes, it is a real opaque node — report it with the callee name, the node
id, and the source location, and recommend expanding it through the existing
free-function / multi-op-method frame machinery (see
`_multi_op_free_functions` / `_multi_op_forward_methods` in
`TraceLens/ModelUtils/ast_analyze.py`). If it is a sanctioned fused kernel,
say so and do not flag it.

Report a concise list of confirmed opaque nodes (or "none found"), most-impactful
first. Do not edit code — this is an audit.
