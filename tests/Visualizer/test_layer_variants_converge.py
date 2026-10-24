###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mutually-exclusive layer variants leave through the stack's output tile.

A repeat stack whose layers are not all alike is drawn as one group per
variant: GLM's decoder splits into ``31x_``/``11x_``/``3x_``, DeepSeek's into
four, Kimi's into three, MiniMax's into two. Exactly one of them is what any
given layer IS, so they are ALTERNATIVES.

Each group's output was wired straight to whatever follows the stack, so a
reader saw three tensors arriving at once and GLM's ``hc_head`` -- a ``mean`` --
appeared to take three operands. Each variant already has the input and output
tiles a consumer reads. Those alternatives meet in one output tile of the
stack, the same port every other block uses to hand a value on, and the
consumer reads that tile.
"""

from __future__ import annotations

import pytest

from TraceLens.Visualizer.model_explorer_export.merge import (
    _converge_layer_variants_through_an_output_tile,
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


def _outputs(nodes):
    return [
        n
        for n in nodes
        if any(
            a.get("key") == "synthetic" and a.get("value") == "@output"
            for a in n.get("attrs") or []
        )
    ]


class TestTheArmsConverge:
    def test_three_variants_reach_their_consumer_through_one_output_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("c", f"{STACK}/3x_C"),
            _node("head", "", label="Mean", sources=["a", "b", "c"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tiles = _outputs(nodes)
        assert len(tiles) == 1
        assert tiles[0]["label"] == "output"
        head = next(n for n in nodes if n["id"] == "head")
        assert [e["sourceNodeId"] for e in head["incomingEdges"]] == [tiles[0]["id"]]
        assert {e["sourceNodeId"] for e in tiles[0]["incomingEdges"]} == {"a", "b", "c"}

    def test_the_join_is_an_output_tile_for_the_consumer(self) -> None:
        """The consumer reads an output tile, the same port every block leaves through."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        attrs = {a["key"]: a["value"] for a in _outputs(nodes)[0]["attrs"]}
        assert attrs["synthetic"] == "@output"
        assert "class_name" not in attrs

    def test_loop_ends_name_the_tile_the_consumer_reads(self) -> None:
        """The variant output tiles already name the tensor. The stack tile keeps it."""
        nodes = [
            _node(
                "a",
                f"{STACK}/48x_A",
                label="loop out: hidden_states - 48 of 64 iterations",
            ),
            _node(
                "b",
                f"{STACK}/16x_B",
                label="loop out: hidden_states - 16 of 64 iterations",
            ),
            _node("head", "norm", label="x", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["label"] == "hidden_states"
        head = next(n for n in nodes if n["id"] == "head")
        assert [e["sourceNodeId"] for e in head["incomingEdges"]] == [tile["id"]]

    def test_two_consumers_of_the_same_arms_share_the_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
            _node("tail", "other", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tiles = _outputs(nodes)
        assert len(tiles) == 1
        for node_id in ("head", "tail"):
            node = next(n for n in nodes if n["id"] == node_id)
            assert [e["sourceNodeId"] for e in node["incomingEdges"]] == [
                tiles[0]["id"]
            ]

    def test_it_is_drawn_beside_the_stack_that_holds_the_variants(self) -> None:
        """The stack box holds the variants. Their join sits one level up."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["namespace"] == ""
        assert tile["id"] == f"{STACK}/@output:output"

        nested = "decoder/45x_Decoder"
        nodes = [
            _node("a", f"{nested}/31x_A"),
            _node("b", f"{nested}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        tile = _outputs(nodes)[0]
        assert tile["namespace"] == "decoder"
        assert tile["id"] == f"{nested}/@output:output"

    def test_it_carries_the_arms_shape(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        attrs = {a["key"]: a["value"] for a in _outputs(nodes)[0]["attrs"]}
        assert attrs["output_shape"] == "[B, S, H] bfloat16"

    def test_it_is_drawn_like_every_other_output_tile(self) -> None:
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert isinstance(_outputs(nodes)[0].get("style"), dict)

    def test_the_innermost_variant_is_the_one_chosen_between(self) -> None:
        """The stack nests in its own ``45x_`` wrapper; that is not a variant."""
        nodes = [
            _node("a", f"{STACK}/31x_A/inner"),
            _node("b", f"{STACK}/11x_B/inner"),
            _node("head", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert len(_outputs(nodes)) == 1


class TestWhatIsNotAnAlternative:
    def test_two_tensors_out_of_one_variant(self) -> None:
        """Two edges from the SAME group are two tensors, not a choice."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/31x_A"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_a_uniform_stack(self) -> None:
        """GPT-2's layers are all alike: one group, nothing to choose."""
        nodes = [
            _node("a", "12x_Block"),
            _node("x", ""),
            _node("op", "", sources=["a", "x"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_producers_outside_any_repeat_group(self) -> None:
        nodes = [
            _node("a", "enc"),
            _node("b", "dec"),
            _node("op", "", sources=["a", "b"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_a_single_operand(self) -> None:
        nodes = [_node("a", f"{STACK}/31x_A"), _node("op", "", sources=["a"])]
        _converge_layer_variants_through_an_output_tile(nodes)
        assert _outputs(nodes) == []

    def test_an_unrelated_operand_is_left_alone(self) -> None:
        """Only the alternatives converge; a genuine second input stays put."""
        nodes = [
            _node("a", f"{STACK}/31x_A"),
            _node("b", f"{STACK}/11x_B"),
            _node("w", "", label="weight"),
            _node("op", "", sources=["a", "b", "w"]),
        ]
        _converge_layer_variants_through_an_output_tile(nodes)
        op = next(n for n in nodes if n["id"] == "op")
        sources = {e["sourceNodeId"] for e in op["incomingEdges"]}
        assert "w" in sources and len(sources) == 2

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _converge_layer_variants_through_an_output_tile(nodes)
        assert nodes == []


def test_merge_is_a_name_no_torch_op_uses() -> None:
    """Why the identity is ``Merge`` and not ``Select``."""
    torch = pytest.importorskip("torch")
    assert not hasattr(torch, "merge")
    assert not [
        s for s in torch._C._jit_get_all_schemas() if "merge" in str(s.name).lower()
    ]
    assert hasattr(torch, "select")  # the name that WOULD have collided
