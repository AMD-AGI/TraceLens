###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A tile that knows its shape has to publish it, or the reader sees ``?``.

Model Explorer draws a tile's shape, and labels every edge leaving it, from
``outputsMetadata``. The node-level ``output_shape`` attr is text in the side
panel and nothing else. ``apply_shape_attrs`` writes both at once, so anything
shaped by inference is fine -- but several passes create tiles AFTER inference
has run, take the shape from the producer they pass through, and write the attr
alone. Those tiles named their own shape one panel away while drawing ``?``:
Kimi's ``v`` into ``chunk_kda_fwd_intra`` (a kernel port interposed late, which
then absorbed the named tile that did publish one), the boundary tiles
``_carry_producer_shape_to_boundary_tiles`` fills, and the layer-variant phi --
5 on GLM, 12 on DeepSeek, 23 on MiniMax, 5 on Kimi when this was written.

The invariant is the whole point and is stated here as one: a node that states
a shape publishes it. A single pass at the end of the build enforces it, so a
future tile-making pass cannot reintroduce the gap one tile at a time.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.merge import (
    _publish_declared_shapes_as_output_ports,
)
from TraceLens.Visualizer.model_explorer_export.shapes import stamp_full_shapes


def _node(node_id, shape=None, dtype=None, ports=None, sources=()):
    attrs = []
    if shape:
        attrs.append({"key": "output_shape", "value": shape})
    if dtype:
        attrs.append({"key": "output_dtype", "value": dtype})
    node = {
        "id": node_id,
        "label": node_id,
        "namespace": "box",
        "attrs": attrs,
        "incomingEdges": [
            {"sourceNodeId": s, "sourceNodeOutputId": p} for s, p in sources
        ],
    }
    if ports is not None:
        node["outputsMetadata"] = ports
    return node


def _ports(node):
    return node.get("outputsMetadata") or []


def _attrs(port):
    return {attr["key"]: attr["value"] for attr in port.get("attrs", [])}


class TestAStatedShapeIsPublished:
    def test_the_shape_reaches_a_port(self) -> None:
        nodes = [_node("v", shape="[B, S, 96, 128] bfloat16", dtype="bfloat16")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _attrs(_ports(nodes[0])[0]) == {
            "shape": "[B, S, 96, 128] bfloat16",
            "dtype": "bfloat16",
        }

    def test_a_shape_with_no_dtype_still_publishes(self) -> None:
        nodes = [_node("v", shape="[B, S]")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _attrs(_ports(nodes[0])[0]) == {"shape": "[B, S]"}

    def test_a_node_stating_nothing_publishes_nothing(self) -> None:
        """Inventing a shape is worse than drawing ``?``."""
        nodes = [_node("v")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0]) == []

    def test_what_a_node_already_published_is_left_alone(self) -> None:
        """Several ports is where several ports come from: never overwrite."""
        existing = [
            {"id": "cos", "attrs": [{"key": "shape", "value": "[B, S, D]"}]},
            {"id": "sin", "attrs": [{"key": "shape", "value": "[B, S, D]"}]},
        ]
        nodes = [_node("rotary", shape="[B, S, D]", ports=existing)]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0]) == existing

    def test_the_port_takes_the_id_its_readers_dock_on(self) -> None:
        """A tile read at a named slot keeps that name, or nothing matches it."""
        nodes = [
            _node("tile", shape="[B, S]"),
            _node("reader", sources=[("tile", "result")]),
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "result"

    def test_readers_docking_on_different_slots_fall_back_to_zero(self) -> None:
        nodes = [
            _node("tile", shape="[B, S]"),
            _node("a", sources=[("tile", "cos")]),
            _node("b", sources=[("tile", "sin")]),
        ]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "0"

    def test_a_tile_nothing_reads_publishes_on_zero(self) -> None:
        nodes = [_node("tile", shape="[B, S]")]
        _publish_declared_shapes_as_output_ports(nodes)
        assert _ports(nodes[0])[0]["id"] == "0"

    def test_an_empty_graph(self) -> None:
        nodes: list[dict] = []
        _publish_declared_shapes_as_output_ports(nodes)
        assert nodes == []


def test_the_full_shape_is_recorded_even_when_the_graph_is_not_split() -> None:
    node = _node(
        "linear",
        shape="[B, S, 8192] bfloat16",
        ports=[
            {"id": "0", "attrs": [{"key": "shape", "value": "[B, S, 8192] bfloat16"}]}
        ],
    )
    stamp_full_shapes([node])
    assert {item["key"]: item["value"] for item in node["attrs"]}["full_shape"] == (
        "[B, S, 8192] bfloat16"
    )
    assert node["outputsMetadata"][0]["attrs"][1] == {
        "key": "full_shape",
        "value": "[B, S, 8192] bfloat16",
    }
    # A second pass must not replace a full shape already saved.
    node["attrs"][0]["value"] = "[B, S, 4096] bfloat16"
    stamp_full_shapes([node])
    assert {item["key"]: item["value"] for item in node["attrs"]}["full_shape"] == (
        "[B, S, 8192] bfloat16"
    )
