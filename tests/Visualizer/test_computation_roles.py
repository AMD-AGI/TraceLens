###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A leaf's role is read from what it computes, not from its parent module."""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.computation_roles import (
    ATTENTION_ROLE,
    ROLE_BASIS_COMPUTATION,
    attention_core_ids,
    stamp_computation_roles,
)


def _node(node_id, op_type, shape, sources=(), role="other"):
    attrs = [
        {"key": "role", "value": role},
        {"key": "output_shape", "value": f"[{shape}] bfloat16"},
    ]
    if op_type:
        attrs.append({"key": "op_type", "value": op_type})
    return {
        "id": node_id,
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0", "targetNodeInputId": str(i)}
            for i, s in enumerate(sources)
        ],
    }


def _attr(node, key):
    return next((a["value"] for a in node["attrs"] if a["key"] == key), None)


def _attention_graph():
    """Eager attention with a mask built from its own matmul and a sink."""
    scores = "B, 8, S, K"
    return [
        _node("x", "Linear", "B, S, 64"),
        _node("scores", "MatMul", scores, ["x"]),
        _node("scaled", "Multiply", scores, ["scores"]),
        # The mask is built from an unrelated matmul of its own (an indexer
        # score); it does not share the score tensor's heads dimension.
        _node("index_scores", "MatMul", "B, S, 4, K", ["x"]),
        _node("mask", "Where", "B, 1, S, K", ["index_scores"]),
        _node("masked", "Add", scores, ["scaled", "mask"]),
        _node("softmax", "Softmax", scores, ["masked"]),
        _node("weights", "Cast", scores, ["softmax"]),
        _node("values", "MatMul", "B, 8, S, 16", ["weights", "x"]),
        _node("merged", "Transpose", "B, S, 8, 16", ["values"]),
    ]


def test_the_score_softmax_and_value_path_is_attention():
    nodes = _attention_graph()
    assert attention_core_ids(nodes) == {
        "scores",
        "scaled",
        "masked",
        "softmax",
        "weights",
        "values",
    }


def test_the_mask_branch_and_the_projection_are_not_attention():
    nodes = _attention_graph()
    stamp_computation_roles({"nodes": nodes})
    by_id = {node["id"]: node for node in nodes}
    for node_id in ("x", "index_scores", "mask", "merged"):
        assert _attr(by_id[node_id], "role") == "other"
        assert _attr(by_id[node_id], "role_basis") is None


def test_stamped_leaves_carry_the_role_and_its_basis():
    nodes = _attention_graph()
    stamped = stamp_computation_roles({"nodes": nodes})
    assert stamped == 6
    by_id = {node["id"]: node for node in nodes}
    for node_id in ("scores", "scaled", "masked", "softmax", "weights", "values"):
        assert _attr(by_id[node_id], "role") == ATTENTION_ROLE
        assert _attr(by_id[node_id], "role_basis") == ROLE_BASIS_COMPUTATION


def test_a_router_softmax_is_not_attention():
    # Linear -> Softmax -> TopK: a projection feeds it, not a matmul, and no
    # matmul follows.
    nodes = [
        _node("tokens", "Linear", "T, 64"),
        _node("logits", "Linear", "T, 16", ["tokens"]),
        _node("probs", "Softmax", "T, 16", ["logits"]),
        _node("top", "TopK", "T, 2", ["probs"]),
    ]
    assert attention_core_ids(nodes) == set()


def test_a_pooling_softmax_with_no_matmul_after_it_is_not_attention():
    # A KV compressor softmaxes over a window and sums. A matmul feeds it, but
    # none consumes it.
    nodes = [
        _node("a", "MatMul", "B, W, 4, 8"),
        _node("probs", "Softmax", "B, W, 4, 8", ["a"]),
        _node("pooled", "Sum", "B, W, 8", ["probs"]),
    ]
    assert attention_core_ids(nodes) == set()


def test_a_softmax_without_a_shape_is_left_alone():
    nodes = _attention_graph()
    for node in nodes:
        if node["id"] == "softmax":
            node["attrs"] = [a for a in node["attrs"] if a["key"] != "output_shape"]
    assert attention_core_ids(nodes) == set()


def test_a_boundary_port_carries_the_walk_between_frames():
    nodes = _attention_graph()
    # The softmax's operand now crosses a frame boundary through a port that
    # has no op type of its own.
    nodes.insert(6, _node("port", None, "B, 8, S, K", ["masked"]))
    for node in nodes:
        if node["id"] == "softmax":
            node["incomingEdges"] = [
                {"sourceNodeId": "port", "sourceNodeOutputId": "0", "targetNodeInputId": "0"}
            ]
    core = attention_core_ids(nodes)
    assert {"scores", "masked", "softmax", "values"} <= core
    assert "port" not in core


def test_a_role_already_read_from_the_module_is_kept():
    nodes = _attention_graph()
    for node in nodes:
        if node["id"] == "scaled":
            for attr in node["attrs"]:
                if attr["key"] == "role":
                    attr["value"] = "norm"
    stamp_computation_roles({"nodes": nodes})
    by_id = {node["id"]: node for node in nodes}
    assert _attr(by_id["scaled"], "role") == "norm"
    assert _attr(by_id["scores"], "role") == ATTENTION_ROLE
