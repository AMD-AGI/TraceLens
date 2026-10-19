###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mutually-exclusive layer variants converge in a phi, not side by side.

A repeat stack whose layers are not all alike is drawn as one group per
variant: GLM's decoder splits into ``31x_``/``11x_``/``3x_``, DeepSeek's into
four, Kimi's into three, MiniMax's into two. Exactly one of them is what any
given layer IS, so they are ALTERNATIVES.

Each group's output was wired straight to whatever follows the stack, so a
reader saw three tensors arriving at once and GLM's ``hc_head`` -- a ``mean`` --
appeared to take three operands. Mutually-exclusive arms cannot enter a node as
if they were ordinary parallel operands; the graph already has the node for
this, the phi it emits for the two arms of an unresolved ``if``.

The phi is labelled ``Mutually exclusive branch`` and its identity stays
``Merge``. Not ``Select``: ``torch.select`` exists (two aten schemas) and means
indexing one position along a dimension, which REMOVES that dimension -- the
opposite of a node whose output has the same shape as each arm. No aten schema
anywhere contains "merge", so the identity cannot be mistaken for an operator,
and the arity check correctly resolves no contract for it: a phi legitimately
takes N arms.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.ast_analyze import PHI_CLASS, PHI_LABEL
from TraceLens.Visualizer.model_explorer_export.merge import (
    _converge_layer_variants_through_a_phi,
)


def _node(node_id, namespace, label="op", sources=()):
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [{"key": "output_shape", "value": "[B, S, H] bfloat16"}],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0", "targetNodeInputId": str(i)}
            for i, s in enumerate(sources)
        ],
    }


STACK = "45x_Decoder"


def _phis(nodes):
    return [n for n in nodes if "@variant_merge:" in n["id"]]


class TestTheArmsConverge:
    def test_three_variants_reach_their_consumer_through_one_phi(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("c", f"{STACK}/3x_C"),
            _node("head", "", label="Mean", sources=["a", "b", "c"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        phis = _phis(nodes)
        assert len(phis) == 1
        assert phis[0]["label"] == PHI_LABEL
        head = next(n for n in nodes if n["id"] == "head")
        assert [e["sourceNodeId"] for e in head["incomingEdges"]] == [phis[0]["id"]]
        assert {e["sourceNodeId"] for e in phis[0]["incomingEdges"]} == {"a", "b", "c"}

    def test_the_phi_is_identified_as_a_merge(self) -> None:
        """The label is for a reader; every rule keys on the identity."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        attrs = {a["key"]: a["value"] for a in _phis(nodes)[0]["attrs"]}
        assert attrs["class_name"] == PHI_CLASS

    def test_it_sits_where_the_variants_and_the_consumer_meet(self) -> None:
        """Inside the stack its value would leave without an output tile."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert _phis(nodes)[0]["namespace"] == ""

    def test_it_carries_the_arms_shape(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        attrs = {a["key"]: a["value"] for a in _phis(nodes)[0]["attrs"]}
        assert attrs["output_shape"] == "[B, S, H] bfloat16"

    def test_it_is_drawn_like_every_other_operation(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert isinstance(_phis(nodes)[0].get("style"), dict)

    def test_the_innermost_variant_is_the_one_chosen_between(self) -> None:
        """The stack nests in its own ``45x_`` wrapper; that is not a variant."""
        nodes = [
            _node("a", f"{STACK}/31x_A/inner"),
            _node("b", f"{STACK}/11x_B/inner"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert len(_phis(nodes)) == 1


class TestWhatIsNotAnAlternative:
    def test_two_tensors_out_of_one_variant(self) -> None:
        """Two edges from the SAME group are two tensors, not a choice."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/31x_A"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert _phis(nodes) == []

    def test_a_uniform_stack(self) -> None:
        """GPT-2's layers are all alike: one group, nothing to choose."""
        nodes = [
            _node("a", "12x_Block"),
            _node("x", ""),
            _node("op", "", sources=["a", "x"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert _phis(nodes) == []

    def test_producers_outside_any_repeat_group(self) -> None:
        nodes = [
            _node("a", "enc"),
            _node("b", "dec"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        assert _phis(nodes) == []

    def test_a_single_operand(self) -> None:
        nodes = [_node("a", f"{STACK}/31x_A"), _node("op", "", sources=["a"])]
        _converge_layer_variants_through_a_phi(nodes)
        assert _phis(nodes) == []

    def test_an_unrelated_operand_is_left_alone(self) -> None:
        """Only the alternatives converge; a genuine second input stays put."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("w", "", label="weight"),
            _node("op", "", sources=["a", "b", "w"]),
        ]
        _converge_layer_variants_through_a_phi(nodes)
        op = next(n for n in nodes if n["id"] == "op")
        sources = {e["sourceNodeId"] for e in op["incomingEdges"]}
        assert "w" in sources and len(sources) == 2

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _converge_layer_variants_through_a_phi(nodes)
        assert nodes == []


def test_merge_is_a_name_no_torch_op_uses() -> None:
    """Why the identity is ``Merge`` and not ``Select``."""
    torch = pytest.importorskip("torch")
    assert not hasattr(torch, "merge")
    assert not [
        s for s in torch._C._jit_get_all_schemas() if "merge" in str(s.name).lower()
    ]
    assert hasattr(torch, "select")  # the name that WOULD have collided
