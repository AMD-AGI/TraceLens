###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for relabeling a weight-operand matmul as ``Linear``.

A ``torch.matmul`` / ``torch.bmm`` that contracts a real activation against a
*learned constant* weight is an affine projection -- semantically an
``nn.Linear`` -- so its tile should read ``Linear`` and (once the render filter
drops the constant weight edge) draw with a single input.
``ast_analyze.classify_matmul_label`` only catches the case where the weight is
a *direct* operand of the call; a grouped-linear
``torch.bmm(x, self.weight.view(...).transpose(...))`` flows the weight in
through a local variable, so the relabel must happen at the graph level where the
weight's whole producer subgraph is already ``constant``-tagged
(``merge._relabel_constant_operand_matmul_as_linear``).
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.loader import load_model_spec
from TraceLens.ModelUtils.shape_inference import ShapeInferencer
from TraceLens.Visualizer.model_explorer_export.merge import (
    _relabel_constant_operand_matmul_as_linear,
    build_merged_model_graph,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    _graph_without_constants,
)


def _op_node(node_id: str, label: str, producers: list[str]) -> dict:
    return {
        "id": node_id,
        "label": label,
        "attrs": [{"key": "class_name", "value": label}],
        "incomingEdges": [{"sourceNodeId": p} for p in producers],
    }


def _const_node(node_id: str) -> dict:
    return {
        "id": node_id,
        "label": "weight",
        "attrs": [{"key": "constant", "value": "true"}],
    }


def _activation_node(node_id: str) -> dict:
    return {"id": node_id, "label": "Transpose", "attrs": []}


def test_bmm_against_learned_weight_relabels_to_linear():
    """A ``BatchMatMul`` with one constant operand and one activation -> ``Linear``."""
    nodes = [
        _activation_node("x"),
        _const_node("w"),
        _op_node("bmm", "BatchMatMul", ["x", "w"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    bmm = nodes[-1]
    assert bmm["label"] == "Linear"
    assert any(
        a["key"] == "class_name" and a["value"] == "Linear" for a in bmm["attrs"]
    )


def test_matmul_of_two_activations_stays_matmul():
    """A genuine activation-activation contraction (attention scores) is untouched."""
    nodes = [
        _activation_node("q"),
        _activation_node("k"),
        _op_node("scores", "MatMul", ["q", "k"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    assert nodes[-1]["label"] == "MatMul"


def test_matmul_of_two_constants_stays_matmul():
    """No activation operand means it is not an activation projection -- left as is."""
    nodes = [
        _const_node("a"),
        _const_node("b"),
        _op_node("folded", "MatMul", ["a", "b"]),
    ]
    _relabel_constant_operand_matmul_as_linear(nodes)
    assert nodes[-1]["label"] == "MatMul"


def _attr(node: dict, key: str):
    for a in node.get("attrs", []) or []:
        if a.get("key") == key:
            return a.get("value")
    return None


def test_deepseek_v4_grouped_linear_bmm_renders_as_single_input_linear():
    """The DeepSeek ``o_a_proj`` grouped-linear ``bmm(x, self.weight...)`` reads
    ``Linear`` and, after the constant weight edge is filtered, draws one input.

    Keyed structurally: the sole ``bmm`` op whose annotated operands include a
    ``Constant`` -- no hardcoded node id or line number.
    """
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec("deepseek-ai/DeepSeek-V4-Flash", detailed=True)
    graph = build_merged_model_graph(spec, shape_inferencer=ShapeInferencer(spec))
    nodes = graph["nodes"]

    weighted_bmm = [
        node
        for node in nodes
        if _attr(node, "raw_op") == "bmm"
        and "Constant" in (_attr(node, "input_types") or "")
    ]
    assert weighted_bmm, "expected a bmm with a constant weight operand"
    for node in weighted_bmm:
        assert node.get("label") == "Linear", node.get("label")
        assert _attr(node, "op_type") == "Linear"

    filtered = _graph_without_constants(graph)
    kept = {n["id"] for n in filtered["nodes"]}
    by_id = {n["id"]: n for n in filtered["nodes"]}
    for node in weighted_bmm:
        assert node["id"] in kept
        drawn = by_id[node["id"]]
        assert len(drawn.get("incomingEdges", [])) == 1, drawn.get("incomingEdges")
