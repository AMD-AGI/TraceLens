###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A boundary tile carries the shape of what crosses it.

A boundary does not compute: it marks where a tensor enters a module, so what
it carries is exactly what its producer made. Several passes materialise these
tiles AFTER shapes have been filled in, and those arrived with none at all --
GLM's attention mask read ``[B, S] bool`` at the mirror and then nothing at
either of the two module levels it enters, and MiniMax's ``position_ids`` did
the same at four.

The tiles chain, outermost first, so the carry runs to a fixed point: the outer
level has to be given its shape before the level inside it can take it.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _carry_producer_shape_to_boundary_tiles,
)


def _tile(node_id, sources=(), synthetic="@input", **attrs):
    node = {
        "id": node_id,
        "label": node_id.rsplit(":", 1)[-1],
        "attrs": [{"key": "synthetic", "value": synthetic}] if synthetic else [],
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": "0"} for s in sources
        ],
    }
    node["attrs"].extend({"key": k, "value": v} for k, v in attrs.items())
    return node


def _shape(nodes, node_id):
    node = next(n for n in nodes if n["id"] == node_id)
    return next(
        (a["value"] for a in node.get("attrs", []) if a["key"] == "output_shape"), None
    )


class TestTheShapeTravelsDownTheChain:
    def test_one_level(self) -> None:
        nodes = [
            _tile("mirror", synthetic="@input_mirror", output_shape="[B, S] bool"),
            _tile("outer/@input:attention_mask", sources=["mirror"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:attention_mask") == "[B, S] bool"

    def test_every_level_of_a_chain(self) -> None:
        """Run to a fixed point, or only the outermost would be filled."""
        nodes = [
            _tile("mirror", synthetic="@input_mirror", output_shape="[B, S] bool"),
            _tile("a/@input:attention_mask", sources=["mirror"]),
            _tile("a/b/@input:attention_mask", sources=["a/@input:attention_mask"]),
            _tile("a/b/c/@input:attention_mask", sources=["a/b/@input:attention_mask"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        for node_id in (
            "a/@input:attention_mask",
            "a/b/@input:attention_mask",
            "a/b/c/@input:attention_mask",
        ):
            assert _shape(nodes, node_id) == "[B, S] bool", node_id

    def test_the_dtype_comes_with_it(self) -> None:
        nodes = [
            _tile(
                "producer",
                synthetic="@input_mirror",
                output_shape="[B, S] bool",
                output_dtype="bool",
            ),
            _tile("outer/@input:mask", sources=["producer"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        tile = next(n for n in nodes if n["id"] == "outer/@input:mask")
        assert any(
            a["key"] == "output_dtype" and a["value"] == "bool" for a in tile["attrs"]
        )


class TestWhatIsLeftAlone:
    def test_a_stale_boundary_shape_is_replaced_by_its_producer(self) -> None:
        nodes = [
            _tile(
                "producer",
                synthetic="@input_mirror",
                output_shape="[B, 1, S, S] bfloat16",
                output_dtype="bfloat16",
            ),
            _tile(
                "outer/@input:mask",
                sources=["producer"],
                output_shape="[B, S] int64",
                output_dtype="int64",
            ),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") == "[B, 1, S, S] bfloat16"
        tile = next(n for n in nodes if n["id"] == "outer/@input:mask")
        assert (
            next(a["value"] for a in tile["attrs"] if a["key"] == "output_dtype")
            == "bfloat16"
        )

    def test_a_tile_with_several_producers(self) -> None:
        """Not a pass-through; it would have to pick one, so it says nothing."""
        nodes = [
            _tile("p1", synthetic="@input_mirror", output_shape="[B, S] bool"),
            _tile("p2", synthetic="@input_mirror", output_shape="[B, S, H] bfloat16"),
            _tile("outer/@input:mask", sources=["p1", "p2"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_producer_that_is_not_in_the_graph(self) -> None:
        nodes = [_tile("outer/@input:mask", sources=["absent"])]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_producer_with_no_shape_of_its_own(self) -> None:
        nodes = [
            _tile("producer", synthetic="@input_mirror"),
            _tile("outer/@input:mask", sources=["producer"]),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_tile_with_no_producer(self) -> None:
        nodes = [_tile("outer/@input:mask")]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@input:mask") is None

    def test_a_node_that_is_not_a_boundary(self) -> None:
        """An op computes its own shape; it does not inherit one."""
        nodes = [
            _tile("producer", synthetic="@input_mirror", output_shape="[B, S] bool"),
            _tile("outer/@op_add", sources=["producer"], synthetic=None),
        ]
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert _shape(nodes, "outer/@op_add") is None

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _carry_producer_shape_to_boundary_tiles(nodes)
        assert nodes == []
